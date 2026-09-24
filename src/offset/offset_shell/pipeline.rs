use super::*;

/// Put every face on one normal convention (surface normal on the CCW
/// parameter-winding side) and then restore outward normals. The offset shell's
/// source and offset skins are assembled as independent components whose
/// per-face `same_sense` can follow opposite conventions; `validate()` only
/// checks coedge-direction coherence (the `forward` flags), so a joined,
/// watertight, coedge-coherent manifold can still be normal-incoherent and
/// report the wrong volume. Re-deriving `same_sense` from the loop winding makes
/// the coedge-coherent manifold normal-coherent; the signed-volume flip then
/// orients it outward. Only invoked after a non-coplanar rim weld, so the
/// coplanar cylinder/prism shells keep their exact assembled orientation.
fn coheres_face_normals(solid: &mut BrepSolid) -> Result<(), String> {
    for face in solid.shells.iter_mut().flat_map(|shell| &mut shell.faces) {
        face.same_sense = parameter_space_area(face)? > 0.0;
    }
    // Outward is a PER-SHELL property: a through-hole shelled at both ends
    // legitimately produces two disjoint closed components (the outer wall
    // tube and the hole wall tube), and a single global signed-volume flip
    // cannot orient both.
    for shell_index in 0..solid.shells.len() {
        let single = BrepSolid {
            id: solid.id,
            vertices: solid.vertices.clone(),
            edges: solid.edges.clone(),
            shells: vec![solid.shells[shell_index].clone()],
            genus: 0,
        };
        if crate::solid_signed_volume(&single)? < 0.0 {
            flip_shell_faces(&mut solid.shells[shell_index])?;
        }
    }
    Ok(())
}

/// Mint a vertex at every SELF-TOUCH of the source's faces before the shell
/// reads their trims.
///
/// A face whose loops touch at an EDGE INTERIOR is a pinched region: the
/// reported 2026-09-09 document's r = 6 through-hole kisses both r = 4 fillet
/// setbacks on the top face, at (16, 20, 10) and (10, 20, 16), with no vertex
/// there.  The fragment stage's 2D arrangement resolves such a pinch by
/// carving the touched-off corner into its own region, whose loop walks the
/// closed hole chain in a PARTIAL run — which `build_loop` can only assemble
/// when the touch is a vertex of both touching edges, and otherwise refuses
/// with "chain N fragmented into an incomplete run".
///
/// The boolean lane mints that vertex inside `build_imprints`
/// (`imprint/self_touch.rs`).  Nothing minted it here: the shell fragments
/// its SOURCE against an EMPTY imprint, and its offset carriers — whose trims
/// are copied from the source faces — deliberately keep their boundaries
/// un-split, so both arrived at the arrangement pinched.  Splitting the
/// source once, before the carriers are built from it, fixes both: every
/// carrier inherits the split trim by construction.
///
/// Returns `None` when no face self-touches, so every part that was never
/// pinched keeps its exact previous path.  Escape hatch:
/// `BREP_SELF_TOUCH_SPLIT=0`, the same one the imprint lane honours.
fn split_self_touching_source(source: &BrepSolid) -> Result<Option<BrepSolid>, String> {
    if std::env::var("BREP_SELF_TOUCH_SPLIT").as_deref() == Ok("0") {
        return Ok(None);
    }
    // The same derivation `offset_shell_impl` uses for every other band, so
    // the source scan and the carrier-pair scans agree on what "touching"
    // means.
    let scale = crate::solid_scale(source);
    let tolerance = 1e-7f64.max(scale * 1e-9);
    let edge_splits = crate::imprint::self_touch_edge_splits(source, SOURCE_OPERAND, tolerance, scale)?;
    if edge_splits.is_empty() {
        return Ok(None);
    }
    os_debug!(
        "self-touch: source split on {} edge(s): {:?}",
        edge_splits.len(),
        edge_splits
            .iter()
            .map(|split| split.edge_id)
            .collect::<Vec<_>>(),
    );
    let imprint = ImprintResultRecord {
        edge_splits,
        ..empty_imprint()
    };
    Ok(Some(apply_edge_splits(source, SOURCE_OPERAND, &imprint)?))
}

pub fn offset_shell(
    source: &BrepSolid,
    opening_face_ids: &[u64],
    distance: f64,
) -> Result<OffsetShellResultRecord, String> {
    // Pinched source faces are un-modellable by the arrangement until the
    // touch is a vertex; do it once here so BOTH `offset_shell_impl` attempts
    // and every carrier built from the source see the same topology.
    let prepared = split_self_touching_source(source)?;
    let source = prepared.as_ref().unwrap_or(source);
    // First try with oblique hole-wall carriers extended past their opening
    // planes (the lane that closes tilted/conical through-holes). If that
    // extension fired but the arrangement could not assemble the harder
    // configuration it produced (e.g. the extended bore genuinely intersects
    // other offset walls), fall back to the un-extended pipeline so every
    // previously-supported shape keeps its exact path and every refusal keeps
    // the canonical honesty-gate message.
    let mut extension_fired = false;
    let mut reflex_rebuilds = 0usize;
    match offset_shell_impl(
        source,
        opening_face_ids,
        distance,
        true,
        &mut extension_fired,
        &mut reflex_rebuilds,
    ) {
        Ok(result) => {
            // Attribution marker: which pipeline produced this result. A
            // retry-produced Ok can silently mask a broken extension lane —
            // always say which attempt won. The reflex-rebuild count rides
            // along because that lane fires on BOTH attempts: "extension
            // idle" alone would misread as "the plain pipeline did this".
            os_debug!(
                "offset_shell result: FIRST attempt (extension {}, reflex-rebuilt {})",
                if extension_fired { "fired" } else { "idle" },
                reflex_rebuilds,
            );
            Ok(result)
        }
        Err(error) if extension_fired => {
            os_debug!("extended pipeline failed ({error}); retrying without carrier extension");
            let mut unused = false;
            let mut retry_reflex = 0usize;
            let retried = offset_shell_impl(
                source,
                opening_face_ids,
                distance,
                false,
                &mut unused,
                &mut retry_reflex,
            );
            os_debug!(
                "offset_shell result: RETRY without extension ({}, reflex-rebuilt {retry_reflex})",
                if retried.is_ok() { "ok" } else { "err" },
            );
            retried
        }
        Err(error) => Err(error),
    }
}

fn offset_shell_impl(
    source: &BrepSolid,
    opening_face_ids: &[u64],
    distance: f64,
    extend_oblique_carriers: bool,
    extension_fired: &mut bool,
    reflex_rebuilds: &mut usize,
) -> Result<OffsetShellResultRecord, String> {
    if !distance.is_finite() || distance.abs() <= 1e-7 {
        return Err("offset_shell: distance must be finite and non-zero".into());
    }
    if opening_face_ids.is_empty() {
        return Err("offset_shell: at least one opening face is required".into());
    }
    let source_faces = source
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .collect::<Vec<_>>();
    let opening_set = opening_face_ids.iter().copied().collect::<HashSet<_>>();
    if opening_set
        .iter()
        .any(|id| !source_faces.iter().any(|face| face.id == *id))
    {
        return Err("offset_shell: opening face does not belong to source".into());
    }
    let retained = source_faces
        .iter()
        .filter(|face| !opening_set.contains(&face.id))
        .map(|face| face.id)
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err("offset_shell: removing every face cannot produce a shell".into());
    }
    let mut carriers = Vec::new();
    for face_id in &retained {
        let face = source_faces.iter().find(|face| face.id == *face_id).unwrap();
        if offset_support_collapsed(face, distance)? {
            os_debug!("omitting collapsed offset support for source face {face_id}");
            continue;
        }
        // A planar carrier whose source face meets a neighbour at a REFLEX
        // (concave) edge extends its trim by |distance|: at such a junction
        // the offset skin must GROW past the source footprint to meet the
        // neighbour's offset (a pocket wall's source [5,15] → skin
        // [3.5,16.5]); a source-sized trim leaves a gap no miter can bridge.
        // The surplus at the face's convex ends is clipped by the imprints +
        // on-skin filter. Convex-only faces keep their exact source-sized
        // trim — extending them perturbs delicately-selected fragment
        // landscapes for no benefit.
        let miter = source_faces
            .iter()
            .find(|face| face.id == *face_id)
            .map(|face| face_reflex_miter_tan(source, face))
            .transpose()?
            .flatten();
        // The offsets of two faces joined at a reflex edge meet d·tan(θn/2)
        // past the source rim — d only for a perpendicular join. Scale the
        // extension to the worst sampled miter (×1.5 headroom so the true
        // intersection lands strictly INSIDE the extended trim, never on its
        // boundary where the arrangement drops boundary-coincident cuts) and
        // cap it: a near-tangential join would ask for an unbounded trim.
        let reflex_extension = miter.map(|worst_tan| {
            (distance.abs() * worst_tan * 1.5)
                .max(distance.abs())
                .min(distance.abs() * 12.0)
        });
        if let Some(extension) = reflex_extension {
            os_debug!("carrier for src face {face_id}: reflex-extended by {extension:.4}");
        }
        // OUTWARD: every retained carrier grows by |distance| across its
        // sharp and opening-facing sides (the offsets of a convex junction
        // meet past both source rims; a wall plane cuts the surplus), and
        // by nothing across a smooth junction into a retained neighbour —
        // see `outward_carrier_extension`. A reflex junction needs no
        // outward growth at all (those offsets cross inside the trims).
        let extension = if distance < 0.0 {
            outward_carrier_extension(source, face, &source_faces, &opening_set, distance.abs())?
        } else {
            crate::CarrierExtension::uniform(reflex_extension.unwrap_or(0.0))
        };
        if distance < 0.0 {
            os_debug!(
                "carrier for src face {face_id}: outward extension u=({:.3},{:.3}) v=({:.3},{:.3})",
                extension.u_min,
                extension.u_max,
                extension.v_min,
                extension.v_max,
            );
        }
        let mut carrier = carrier_solid_sided(source, *face_id, distance, &extension)?;
        if (distance < 0.0 || reflex_extension.is_some()) && face.surface.is_affine()? {
            // PADDED planes: the pad slides the plane's net while the trim's
            // loops are cloned in uv, so an INTERIOR loop (a bore or boss rim
            // in the face) is carried outward with the grown outline instead
            // of following the neighbour's offset.
            //   OUTWARD: a bore's rim grew (r=4.68 → 5.15) while the shrunk
            //   offset bore (r=3.68) that must hole this plane fell inside the
            //   enlarged void, so neither the imprint nor the orphan-rim weld
            //   could cut it.
            //   INWARD, reflex-extended: the same slide moves a REFLEX hole
            //   rim the wrong way — a sphere sitting under a frustum's base
            //   annulus (the user's 2026-09-07 document) had its r=8.3 rim
            //   scaled to 9.22 while the shrunk offset sphere (r=7.3) crosses
            //   the offset plane at r=7.23, inside the enlarged hole: the
            //   pair imprint found nothing and the rim stayed one-use.
            // Drop the interior loops (as the wall carriers do): the
            // neighbour's carrier imprints — or its rim, welded in as a
            // hole — cut the true opening, and the seeded selection drops
            // the disk over it. A CONVEX interior rim on the same face (a
            // through-bore in a pocket floor) is safe to drop too: its
            // offset bore imprints at r+d and the ring inside is rejected by
            // class (Out) or by the on-skin filter, no seed involved.
            drop_wall_interior_loops(&mut carrier)?;
        }
        carriers.push(Carrier {
            solid: carrier,
            source_face_id: *face_id,
            kind: OffsetFaceRole::Offset,
        });
    }
    // APEX-CAP carriers: a retained face whose DEGENERATE edge (an apex
    // point, e.g. a conic crater's tip) offsets into a RING is an EXTERIOR
    // cone offset — the parallel surface truncates at the ring and the exact
    // rolling-ball offset closes it with a SPHERE of radius |distance| about
    // the apex. Emit that sphere as an extra carrier: the ring imprint
    // (sphere × offset frustum) splits it, the on-skin filter keeps exactly
    // the cap sector (its points are |d| from the apex and ≥|d| from the
    // rest of the face), and the ring becomes two-use.
    let mut apex_cap_sources = HashSet::default();
    let mut apex_cap_pairs: Vec<(usize, usize)> = Vec::new();
    {
        let retained_count = carriers.len();
        for index in 0..retained_count {
            let source_face = source_by_id_lookup(&source_faces, carriers[index].source_face_id);
            let Some(source_face) = source_face else {
                continue;
            };
            let carrier_surface = carriers[index].solid.shells[0].faces[0].surface.clone();
            for coedge in source_face
                .loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
            {
                let Some(edge) = source
                    .edges
                    .iter()
                    .find(|edge| edge.id == coedge.edge_id)
                else {
                    continue;
                };
                if !edge.degenerate {
                    continue;
                }
                // Image of the degenerate edge on the offset surface.
                let [p0, p1] = coedge.pcurve.domain()?;
                let mut ring = Vec::new();
                for sample in 0..=8 {
                    let uv = coedge
                        .pcurve
                        .evaluate(p0 + (p1 - p0) * sample as f64 / 8.0)?;
                    ring.push(carrier_surface.evaluate(uv.x, uv.y)?);
                }
                let spread = ring
                    .iter()
                    .map(|point| point.sub(ring[0]).length())
                    .fold(0.0f64, f64::max);
                if spread <= distance.abs() * 1e-2 {
                    continue;
                }
                let apex = source
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == edge.start_vertex_id)
                    .map(|vertex| vertex.point)
                    .ok_or_else(|| "offset_shell: apex vertex missing".to_string())?;
                let centroid = ring
                    .iter()
                    .fold(Vec3::default(), |sum, point| sum.add(*point))
                    .scale(1.0 / ring.len() as f64);
                let axis = apex.sub(centroid);
                let axis = if axis.length() > 1e-9 {
                    axis.normalized()?
                } else {
                    Vec3::new(0.0, 0.0, 1.0)
                };
                os_debug!(
                    "apex-cap carrier for src face {} at ({:.3},{:.3},{:.3}) r={:.3}",
                    carriers[index].source_face_id,
                    apex.x,
                    apex.y,
                    apex.z,
                    distance.abs(),
                );
                apex_cap_sources.insert(index);
                apex_cap_sources.insert(carriers.len());
                apex_cap_pairs.push((index, carriers.len()));
                carriers.push(Carrier {
                    solid: crate::make_sphere_brep(apex, distance.abs(), axis)?,
                    source_face_id: carriers[index].source_face_id,
                    kind: OffsetFaceRole::Offset,
                });
            }
        }
    }
    for face_id in opening_face_ids {
        let source_face = source_faces
            .iter()
            .find(|face| face.id == *face_id)
            .unwrap();
        carriers.push(Carrier {
            solid: if distance < 0.0 && source_face.surface.is_affine()? {
                let pad = outward_wall_pad(
                    source,
                    source_face,
                    &source_faces,
                    &opening_set,
                    distance.abs(),
                )?;
                os_debug!("wall for opening face {face_id}: outward pad {pad:.4}");
                let mut wall = carrier_solid(source, *face_id, 0.0, pad)?;
                // The pad extends the wall by STRETCHING the plane's 2x2 net
                // while the trim pcurves are cloned in UV, so every loop
                // scales about the patch centre by (w+2|d|)/w. For the OUTER
                // loop of a full-domain rectangle that lands exactly on the
                // grown outline — the intended wall extent. But an INTERIOR
                // loop (a carve outline: a bore, gouge or crater rim in the
                // opening face) is scaled OUTWARD too, and the enlarged hole
                // then swallows both circles the wall must be cut by (the
                // source carve rim from the wall x source imprint and the
                // shrunk rim from the wall x offset-carrier imprint) — the
                // imprints clip to nothing and the rims dangle one-use. Drop
                // interior loops instead: the wall covers the carve mouth,
                // the imprints cut the true rims, and selection (void-wall
                // guard + on-skin filter) drops the fragments over the void.
                drop_wall_interior_loops(&mut wall)?;
                wall
            } else {
                standalone_face(source, *face_id)?
            },
            source_face_id: *face_id,
            kind: OffsetFaceRole::Wall,
        });
    }
    if carriers.len() >= SOURCE_OPERAND as usize {
        return Err("offset_shell: too many carrier faces for current ABI".into());
    }
    // The characteristic length every tolerance below is derived from.  This
    // was `max ‖vertex‖` — the distance from the WORLD ORIGIN — until audit
    // slice 0: that form made every band widen as the part was moved away from
    // the origin while ignoring the part's actual size, so the same shell
    // solved to different tolerances depending only on where it was modelled.
    // `solid_scale` is the bbox diagonal, floored at 1.0 exactly as the old
    // fold was, so this changes ONE thing: placement-dependence.
    let scale = crate::solid_scale(source);
    crate::report_scale_migration("offset_shell", scale, || {
        source
            .vertices
            .iter()
            .map(|vertex| vertex.point.length())
            .fold(1.0, f64::max)
    });
    let smooth = synchronize_smooth_offset_boundaries(&mut carriers, &source_faces, scale)?;
    // Reflex-rim rebuilds run on BOTH attempts: the fallback for a broken
    // extension is an honest refusal, but the fallback for a skipped reflex
    // rebuild is a silently WRONG (membrane-capped) shell.
    let reflex_rebuilt = rebuild_reflex_rim_carriers(
        &mut carriers,
        source,
        &source_faces,
        &smooth.pairs,
        scale,
        distance,
    )?;
    *reflex_rebuilds = reflex_rebuilt.len();
    if extend_oblique_carriers {
        let oblique_extended = extend_offset_carriers_past_open_hole_rims(
            &mut carriers,
            source,
            &source_faces,
            &opening_set,
            &smooth.pairs,
            scale,
        )?;
        let mut already_extended = oblique_extended.clone();
        already_extended.extend(reflex_rebuilt.iter().copied());
        let fallshort_extended = extend_fallshort_curved_carriers(
            &mut carriers,
            source,
            &source_faces,
            &opening_set,
            &smooth.pairs,
            &already_extended,
            scale,
            distance,
        )?;
        *extension_fired = !oblique_extended.is_empty() || fallshort_extended > 0;
        os_debug!(
            "extended {} oblique + {fallshort_extended} fall-short carriers past their opening planes",
            oblique_extended.len(),
        );
    }
    let tolerance = 1e-7f64.max(scale * 1e-9);
    let pair_tolerance = (scale * 1e-8).max(2e-6);
    let reach = distance.abs() * 4.0 + pair_tolerance;
    let carrier_samples = carriers
        .iter()
        .map(|carrier| carrier_extent_points(&carrier.solid))
        .collect::<Result<Vec<_>, _>>()?;
    let carrier_bounds = carrier_samples
        .iter()
        .map(|samples| Bounds::from_points(samples))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "offset_shell: carrier has no boundary samples".to_string())?;
    let source_by_id = source_faces
        .iter()
        .map(|face| (face.id, *face))
        .collect::<HashMap<_, _>>();
    let source_samples = source_faces
        .iter()
        .map(|face| {
            let standalone = standalone_face(source, face.id)?;
            Ok((face.id, edge_sample_points(&standalone)?))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;
    if debug_enabled() {
        for (index, carrier) in carriers.iter().enumerate() {
            let face = &carrier.solid.shells[0].faces[0];
            let samples = &carrier_samples[index];
            let bounds =
                samples
                    .iter()
                    .fold((Vec3::default(), Vec3::default()), |(low, high), point| {
                        (
                            Vec3 {
                                x: low.x.min(point.x),
                                y: low.y.min(point.y),
                                z: low.z.min(point.z),
                            },
                            Vec3 {
                                x: high.x.max(point.x),
                                y: high.y.max(point.y),
                                z: high.z.max(point.z),
                            },
                        )
                    });
            let surface = &face.surface;
            let rational = surface
                .control_points
                .iter()
                .flatten()
                .any(|point| (point.w - 1.0).abs() > 1e-12);
            os_debug!(
                "carrier[{index}] kind={:?} source_face={} loops={} deg=({},{}) net={}x{} rational={rational} bounds=({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3})",
                carrier.kind, carrier.source_face_id, face.loops.len(),
                surface.degree_u, surface.degree_v,
                surface.control_points.len(),
                surface.control_points.first().map(|row| row.len()).unwrap_or(0),
                bounds.0.x, bounds.0.y, bounds.0.z, bounds.1.x, bounds.1.y, bounds.1.z,
            );
        }
        for (index, carrier) in carriers.iter().enumerate() {
            if !matches!(carrier.kind, OffsetFaceRole::Offset) {
                continue;
            }
            let source_face = source_by_id[&carrier.source_face_id];
            let carrier_surface = &carrier.solid.shells[0].faces[0].surface;
            let (ku, kv) = (
                crate::KnotVector::new(
                    source_face.surface.knots_u.clone(),
                    source_face.surface.degree_u,
                ),
                crate::KnotVector::new(
                    source_face.surface.knots_v.clone(),
                    source_face.surface.degree_v,
                ),
            );
            if let (Ok(ku), Ok(kv)) = (ku, kv) {
                let [u0, u1] = ku.domain();
                let [v0, v1] = kv.domain();
                let mut worst = 0.0f64;
                let mut worst_at = (0.0f64, 0.0f64);
                for iu in 0..=24 {
                    for iv in 0..=24 {
                        let u = u0 + (u1 - u0) * iu as f64 / 24.0;
                        let v = v0 + (v1 - v0) * iv as f64 / 24.0;
                        let expected = face_offsets(source_face)
                            .at(u, v, -distance)
                            .map(|sample| sample.point);
                        let actual = carrier_surface.evaluate(u, v);
                        if let (Ok(expected), Ok(actual)) = (expected, actual) {
                            let error = expected.sub(actual).length();
                            if error > worst {
                                worst = error;
                                worst_at = (u, v);
                            }
                        }
                    }
                }
                let mut normal_swing = 0.0f64;
                let mut previous: Option<Vec3> = None;
                for iu in 0..=48 {
                    let u = u0 + (u1 - u0) * iu as f64 / 48.0;
                    let v = (v0 + v1) / 2.0;
                    if let Ok(normal) = face_normal(source_face, u, v) {
                        if let Some(previous) = previous {
                            normal_swing = normal_swing.max(previous.sub(normal).length());
                        }
                        previous = Some(normal);
                    }
                }
                let weights = source_face
                    .surface
                    .control_points
                    .iter()
                    .flatten()
                    .map(|point| point.w)
                    .fold((f64::MAX, f64::MIN), |(low, high), w| {
                        (low.min(w), high.max(w))
                    });
                os_debug!(
                    "carrier[{index}] offset fit error max={worst:.6} at=({:.4},{:.4}) domain_u=({u0:.4},{u1:.4}) normal_step_max={normal_swing:.4} src_net={}x{} src_deg=({},{}) src_w=({:.4},{:.4}) knots_u={:?}",
                    worst_at.0, worst_at.1,
                    source_face.surface.control_points.len(),
                    source_face.surface.control_points.first().map(|row| row.len()).unwrap_or(0),
                    source_face.surface.degree_u, source_face.surface.degree_v,
                    weights.0, weights.1,
                    &source_face.surface.knots_u,
                );
            }
        }
        os_debug!("smooth pairs: {:?}", smooth.pairs);
    }
    let mut imprint = empty_imprint();
    let mut next_piece_id = 1;
    let mut next_vertex_id = 1;
    for first in 0..carriers.len() {
        for second in first + 1..carriers.len() {
            let first_source = source_by_id[&carriers[first].source_face_id];
            let second_source = source_by_id[&carriers[second].source_face_id];
            if matches!(carriers[first].kind, OffsetFaceRole::Wall)
                && matches!(carriers[second].kind, OffsetFaceRole::Wall)
            {
                // INWARD walls are the opening faces themselves: two openings
                // never overlap and adjacent ones only touch along their
                // shared edge, so there is nothing to imprint. OUTWARD walls
                // are the opening planes EXTENDED by |distance|: the walls of
                // two ADJACENT openings cross along the line of their shared
                // source edge, and each wall ring must stop there (the sheet
                // past that line lies in front of the other opening). Without
                // the cut the wall × source chain below stays open on the
                // shared-edge side (its neighbour there is an opening, not a
                // retained face), the wall never fragments into its ring, and
                // every source rim along both openings dangles one-use.
                if distance > 0.0 || !source_faces_adjacent(first_source, second_source) {
                    continue;
                }
            }
            if smooth.pairs.contains(&(first, second)) {
                os_debug!("pair ({first},{second}) skipped: smooth");
                continue;
            }
            if !carrier_bounds[first].intersects(carrier_bounds[second], pair_tolerance) {
                os_debug!("pair ({first},{second}) skipped: bounds");
                continue;
            }
            if !source_faces_adjacent(first_source, second_source)
                && sample_separation(
                    &source_samples[&first_source.id],
                    &source_samples[&second_source.id],
                ) > reach
            {
                os_debug!("pair ({first},{second}) skipped: separation");
                continue;
            }
            let pair = match build_imprints(
                &carriers[first].solid,
                &carriers[second].solid,
                &ImprintOptions {
                    tolerance,
                    maximum_fit_points: 96,
                    local_fit: true,
                    fit_chunk_points: None,
                    // Offset carriers can meet in short, tightly curved
                    // branches. The general SSI default (diagonal / 15) is
                    // too coarse here and lets independently fitted branches
                    // miss a shared junction when thickness crosses one of
                    // those branches.
                    maximum_ssi_step: Some((scale * 0.0005).max(tolerance * 100.0)),
                },
            ) {
                Ok(pair) => pair,
                // A failed pair imprint (e.g. the marching SSI exhausting its
                // step budget on a long closed sphere/plane intersection) is
                // not fatal to the shell: the affected rim simply stays
                // unsplit and the rim-weld passes below get to close it. The
                // final watertightness gate still refuses anything the welds
                // cannot reach, so tolerating the miss only enlarges the
                // honest success set.
                Err(error) => {
                    os_debug!("pair ({first},{second}) imprint failed: {error}");
                    continue;
                }
            };
            let pieces_before = imprint.pieces.len();
            merge_pair_imprint(
                &mut imprint,
                pair,
                first as u8,
                second as u8,
                &mut next_piece_id,
                &mut next_vertex_id,
                tolerance,
                scale,
            );
            os_debug!(
                "pair ({first},{second}) src=({},{}) pieces+={}",
                carriers[first].source_face_id,
                carriers[second].source_face_id,
                imprint.pieces.len() - pieces_before,
            );
        }
    }
    if distance < 0.0 {
        // OUTWARD shells: the opening-wall carrier is the opening face's
        // plane EXTENDED by |distance|, so its trim carries only the GROWN
        // outline. Its inner boundary — where the wall ring stops at the
        // source solid — is the source outline, and the retained SOURCE faces
        // are not carriers, so no carrier×carrier pair ever imprints it.
        // Un-cut, the wall stays one whole-plane fragment covering the
        // opening void, and the source faces' rim edges dangle one-use.
        // Imprint each wall carrier against the retained source faces that
        // share an edge with its opening: the wall then fragments into the
        // ring (kept) and the opening hole (dropped by the void guard).
        for index in 0..carriers.len() {
            if !matches!(carriers[index].kind, OffsetFaceRole::Wall) {
                continue;
            }
            let opening = source_by_id[&carriers[index].source_face_id];
            for retained_id in &retained {
                let retained_face = source_by_id[retained_id];
                if !source_faces_adjacent(opening, retained_face) {
                    continue;
                }
                let neighbor = standalone_face(source, *retained_id)?;
                let pair = match build_imprints(
                    &carriers[index].solid,
                    &neighbor,
                    &ImprintOptions {
                        tolerance,
                        maximum_fit_points: 96,
                        local_fit: true,
                        fit_chunk_points: None,
                        maximum_ssi_step: Some((scale * 0.0005).max(tolerance * 100.0)),
                    },
                ) {
                    Ok(pair) => pair,
                    Err(error) => {
                        os_debug!(
                            "wall×source pair ({index},{retained_id}) imprint failed: {error}"
                        );
                        continue;
                    }
                };
                let pieces_before = imprint.pieces.len();
                merge_pair_imprint(
                    &mut imprint,
                    pair,
                    index as u8,
                    SOURCE_OPERAND,
                    &mut next_piece_id,
                    &mut next_vertex_id,
                    tolerance,
                    scale,
                );
                os_debug!(
                    "wall×source pair ({index},{retained_id}) pieces+={}",
                    imprint.pieces.len() - pieces_before,
                );
            }
        }
    }
    // APEX-CAP ring splits: the cap × cone section lies ON the cone
    // carrier's promoted ring edge (an overlap, not a crossing), so
    // `build_imprints` mints no edge splits for the cone side. Without them
    // the cone keeps ONE closed ring edge while the cap carries the section
    // as arcs — mismatched segmentation, both sides one-use. Record the arc
    // junctions as splits on the cone's coincident edge.
    for (cone_index, cap_index) in &apex_cap_pairs {
        let cap_face_id = carriers[*cap_index].solid.shells[0].faces[0].id;
        let piece_ids = imprint
            .by_face
            .iter()
            .find(|entry| entry.operand == *cap_index as u8 && entry.face_id == cap_face_id)
            .map(|entry| entry.piece_ids.clone())
            .unwrap_or_default();
        let coincidence_band = 2e-5f64.max(scale * 1e-7);
        for piece_id in piece_ids {
            let Some((curve, t0, t1)) = imprint
                .pieces
                .iter()
                .find(|piece| piece.id == piece_id)
                .map(|piece| (piece.curve.clone(), piece.t0, piece.t1))
            else {
                continue;
            };
            for endpoint in [curve.evaluate(t0)?, curve.evaluate(t1)?] {
                for edge in &carriers[*cone_index].solid.edges {
                    if edge.degenerate {
                        continue;
                    }
                    let projection = crate::project_point_to_curve(&edge.curve, endpoint)?;
                    if projection.distance > coincidence_band
                        || projection.u < edge.t0 + 1e-9
                        || projection.u > edge.t1 - 1e-9
                    {
                        continue;
                    }
                    let parameter = projection.u;
                    if let Some(existing) = imprint.edge_splits.iter_mut().find(|split| {
                        split.operand == *cone_index as u8 && split.edge_id == edge.id
                    }) {
                        if !existing
                            .parameters
                            .iter()
                            .any(|value| (*value - parameter).abs() <= 1e-8)
                        {
                            existing.parameters.push(parameter);
                        }
                    } else {
                        imprint.edge_splits.push(EdgeSplitRecord {
                            operand: *cone_index as u8,
                            edge_id: edge.id,
                            parameters: vec![parameter],
                        });
                    }
                    os_debug!(
                        "apex-cap ring split: cone carrier {cone_index} edge {} at t={parameter:.6}",
                        edge.id,
                    );
                }
            }
        }
    }
    if debug_enabled() {
        for piece in &imprint.pieces {
            let start = piece.curve.evaluate(piece.t0);
            let mid = piece.curve.evaluate((piece.t0 + piece.t1) * 0.5);
            let end = piece.curve.evaluate(piece.t1);
            if let (Ok(start), Ok(mid), Ok(end)) = (start, mid, end) {
                os_debug!(
                    "piece[{}] ({:.4},{:.4},{:.4})..({:.4},{:.4},{:.4})..({:.4},{:.4},{:.4})",
                    piece.id,
                    start.x,
                    start.y,
                    start.z,
                    mid.x,
                    mid.y,
                    mid.z,
                    end.x,
                    end.y,
                    end.z,
                );
            }
        }
    }
    synchronize_smooth_edge_splits(&mut imprint, &smooth.edge_pairs);
    // This decides whether an imprint is genuinely the carrier's existing
    // trim, not whether independently fitted curves can later be sewn. The
    // looser assembly tolerance can erase a short separating branch while it
    // is still departing a boundary.
    let boundary_coincidence_tolerance = 2e-5f64.max(scale * 1e-7);
    let piece_geometry = imprint
        .pieces
        .iter()
        .map(|piece| {
            (
                piece.id,
                FragmentEdgeGeometry {
                    curve: piece.curve.clone(),
                    t0: piece.t0,
                    t1: piece.t1,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    for (index, carrier) in carriers.iter().enumerate() {
        if !matches!(carrier.kind, OffsetFaceRole::Offset) {
            continue;
        }
        let boundaries = carrier
            .solid
            .edges
            .iter()
            .map(|edge| FragmentEdgeGeometry {
                curve: edge.curve.clone(),
                t0: edge.t0,
                t1: edge.t1,
            })
            .collect::<Vec<_>>();
        if let Some(by_face) = imprint
            .by_face
            .iter_mut()
            .find(|entry| entry.operand == index as u8)
        {
            let before = by_face.piece_ids.clone();
            by_face.piece_ids.retain(|piece_id| {
                let piece = &piece_geometry[piece_id];
                !boundaries.iter().any(|boundary| {
                    fragment_edge_lies_on(piece, boundary, boundary_coincidence_tolerance)
                        .unwrap_or(false)
                })
            });
            if debug_enabled() && before.len() != by_face.piece_ids.len() {
                os_debug!(
                    "carrier[{index}] dropped boundary-coincident pieces: {:?} -> {:?}",
                    before,
                    by_face.piece_ids,
                );
            }
        }
    }

    if debug_enabled() {
        for entry in &imprint.by_face {
            os_debug!(
                "by_face operand={} face={} pieces={:?}",
                entry.operand,
                entry.face_id,
                entry.piece_ids,
            );
        }
    }
    let mut split_carriers = Vec::new();
    let mut fragments_by_carrier = Vec::new();
    for (index, carrier) in carriers.iter().enumerate() {
        // Offset carriers keep their boundary un-split (their trims never
        // coincide with imprints — the boundary-coincidence filter drops such
        // pieces instead). An apex-cap ring breaks that assumption: the
        // frustum's ring is its TRIM boundary while the cap crosses it as TWO
        // imprint arcs — without splitting, the two sides segment differently
        // and both stay one-use. Split exactly the cap-affected carriers.
        let split = if matches!(carrier.kind, OffsetFaceRole::Offset)
            && !apex_cap_sources.contains(&index)
        {
            carrier.solid.clone()
        } else {
            apply_edge_splits(&carrier.solid, index as u8, &imprint)?
        };
        let fragments = fragment_solid(&split, index as u8, &imprint)?;
        os_debug!(
            "carrier[{index}] kind={:?} src={} fragments={}",
            carrier.kind,
            carrier.source_face_id,
            fragments.len(),
        );
        if debug_enabled() {
            for (fragment_index, fragment) in fragments.iter().enumerate() {
                let mut boundary_count = 0usize;
                let mut imprint_pieces = Vec::new();
                let mut derived_count = 0usize;
                for coedge in fragment
                    .loops
                    .iter()
                    .flat_map(|loop_record| &loop_record.coedges)
                {
                    match &coedge.source {
                        FragmentEdgeSource::Boundary { .. }
                        | FragmentEdgeSource::SharedBoundary { .. } => boundary_count += 1,
                        FragmentEdgeSource::Imprint { piece_id } => imprint_pieces.push(*piece_id),
                        FragmentEdgeSource::Derived { .. } => derived_count += 1,
                    }
                }
                os_debug!(
                    "  frag[{index}.{fragment_index}] test=({:.3},{:.3},{:.3}) uv=({:.3},{:.3}) boundary={boundary_count} derived={derived_count} pieces={imprint_pieces:?}",
                    fragment.test_point.x, fragment.test_point.y, fragment.test_point.z,
                    fragment.test_uv.x, fragment.test_uv.y,
                );
            }
        }
        split_carriers.push(split);
        fragments_by_carrier.push(fragments);
    }
    let mut assembly_sources = [(SOURCE_OPERAND, source)]
        .into_iter()
        .collect::<HashMap<_, _>>();
    for (index, carrier) in split_carriers.iter().enumerate() {
        assembly_sources.insert(index as u8, carrier);
    }

    let source_edge_by_id = source
        .edges
        .iter()
        .map(|edge| (edge.id, edge))
        .collect::<HashMap<_, _>>();
    let retained_faces_with_edges = retained
        .iter()
        .map(|face_id| {
            let face = source_by_id[face_id];
            let edges = face
                .loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
                .filter_map(|coedge| source_edge_by_id.get(&coedge.edge_id).copied())
                .collect::<Vec<_>>();
            (face, edges)
        })
        .collect::<Vec<_>>();
    let offset_skin_tolerance = 2e-3f64.max(scale * 5e-5).max(distance.abs() * 1e-3);
    // OUTWARD shells: a wall fragment lying IN FRONT of another opening —
    // past that opening's plane on its void side, within the extent of its
    // (extended) wall carrier — is not material: an opening removes
    // everything in front of it. Such a fragment is the sheet of this wall
    // between the source outline and the grown outline on the far side of an
    // ADJACENT opening's shared edge (the x∈[−1,0] strip of an opened −Z
    // face's wall, in front of the opened −X face). It sits within |distance|
    // of a retained face's corner edge, so the void-cross-section guard reads
    // it as wall thickness; only the other opening's plane tells it apart.
    let mut opening_planes: Vec<(usize, Vec3, Vec3)> = Vec::new();
    if distance < 0.0 {
        for (index, carrier) in carriers.iter().enumerate() {
            if !matches!(carrier.kind, OffsetFaceRole::Wall) {
                continue;
            }
            let opening = source_by_id[&carrier.source_face_id];
            if !opening.surface.is_affine()? {
                continue;
            }
            let [u0, u1] = opening.surface.domain_u()?;
            let [v0, v1] = opening.surface.domain_v()?;
            let (u, v) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
            opening_planes.push((
                index,
                opening.surface.evaluate(u, v)?,
                face_normal(opening, u, v)?,
            ));
        }
    }
    let in_front_of_other_opening = |wall_index: usize, point: Vec3| -> Result<bool, String> {
        let band = 2e-3f64.max(scale * 5e-5);
        for (index, plane_point, outward) in &opening_planes {
            if *index == wall_index || point.sub(*plane_point).dot(*outward) <= band {
                continue;
            }
            let face = &split_carriers[*index].shells[0].faces[0];
            let projection = project_point_to_surface(&face.surface, point)?;
            let uv = Vec2 {
                x: projection.u,
                y: projection.v,
            };
            if parameter_point_in_face(face, uv, 1e-8)? != PolygonClass::Outside {
                return Ok(true);
            }
        }
        Ok(false)
    };

    let mut chosen_offsets = Vec::new();
    let mut chosen_walls = Vec::new();
    for (index, carrier) in carriers.iter().enumerate() {
        let raw_seed = source_seed(source, carrier.source_face_id)?;
        let seed = if matches!(carrier.kind, OffsetFaceRole::Offset) {
            offset_seed(
                source_by_id[&carrier.source_face_id],
                raw_seed,
                &split_carriers[index].shells[0].faces[0].surface,
                distance,
                2e-3f64.max(scale * 5e-5),
            )?
            .unwrap_or(raw_seed)
        } else {
            raw_seed
        };
        if seed.sub(raw_seed).length() > 1e-9 {
            os_debug!(
                "  seed[{index}] moved from ({:.4},{:.4}) to the offset image ({:.4},{:.4})",
                raw_seed.x,
                raw_seed.y,
                seed.x,
                seed.y
            );
        }
        if matches!(carrier.kind, OffsetFaceRole::Offset) {
            // Fragments whose test point sits at the full offset distance
            // from every retained source face are the true offset-skin
            // regions. Fragments closer to some other face are shadowed
            // pockets/strips that another carrier owns. Only fall back to the
            // unfiltered set when nothing qualifies (numerical safety net).
            let on_skin = fragments_by_carrier[index]
                .iter()
                .map(|fragment| {
                    point_on_offset_skin(
                        fragment.test_point,
                        &retained_faces_with_edges,
                        distance,
                        offset_skin_tolerance,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            let skin_filter_active = on_skin.iter().any(|flag| *flag);
            let mut classified = Vec::new();
            let mut viable = Vec::new();
            for (fragment_index, fragment) in fragments_by_carrier[index].iter().enumerate() {
                let class = classify_point(fragment.test_point, source, tolerance * 10.0)?.class;
                let excluded_by_class = (distance > 0.0 && class == PointClass::Out)
                    || (distance < 0.0 && class == PointClass::In);
                let contact = if excluded_by_class {
                    false
                } else {
                    distance > 0.0
                        && fragment_has_sustained_source_contact(
                            fragment,
                            &source_faces,
                            &assembly_sources,
                            &imprint,
                            2e-3f64.max(scale * 5e-5),
                        )?
                };
                os_debug!(
                    "  select[{index}.{fragment_index}] class={class:?} contact={contact} on_skin={} seed_in={:?} uv=({:.3},{:.3})",
                    on_skin[fragment_index],
                    parameter_point_in_face(&fragment_as_trim(fragment), seed, 1e-8),
                    fragment.test_uv.x, fragment.test_uv.y,
                );
                if excluded_by_class || (skin_filter_active && !on_skin[fragment_index]) {
                    continue;
                }
                classified.push(fragment.clone());
                if contact {
                    continue;
                }
                viable.push(fragment.clone());
            }
            os_debug!("  select[{index}] seed=({:.4},{:.4})", seed.x, seed.y);
            let seeded = viable
                .iter()
                .filter(|fragment| {
                    parameter_point_in_face(&fragment_as_trim(fragment), seed, 1e-8)
                        .is_ok_and(|class| class != PolygonClass::Outside)
                })
                .cloned()
                .collect::<Vec<_>>();
            let seeded_classified = classified
                .iter()
                .filter(|fragment| {
                    parameter_point_in_face(&fragment_as_trim(fragment), seed, 1e-8)
                        .is_ok_and(|class| class != PolygonClass::Outside)
                })
                .cloned()
                .collect::<Vec<_>>();
            let selected = if viable.is_empty() {
                if !skin_filter_active {
                    // NOTHING on this carrier reads on-skin: every classified
                    // fragment's test point is measurably CLOSER than the
                    // offset distance to some retained face — a buried strip
                    // another carrier owns. Forcing the largest one in anyway
                    // plants a wrong skin patch that collides with the
                    // completion phase's reconstruction (3-use edges).
                    // Contribute nothing: `complete_opening_boundary_cycles`
                    // rebuilds the true skin from the carrier surface, and the
                    // watertight gate still refuses if it cannot.
                    None
                } else {
                    let mut by_area = classified
                        .into_iter()
                        .map(|fragment| {
                            let area = parameter_space_area(&fragment_as_trim(&fragment))?.abs();
                            Ok((fragment, area))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    by_area.sort_by(|a, b| b.1.total_cmp(&a.1));
                    by_area.into_iter().next().map(|entry| entry.0)
                }
            } else if !seeded_classified.is_empty() {
                let mut candidates = seeded_classified;
                candidates.sort_by(|a, b| {
                    a.test_uv
                        .sub(seed)
                        .length()
                        .total_cmp(&b.test_uv.sub(seed).length())
                });
                candidates.into_iter().next()
            } else if viable.len() > 2 {
                let mut by_area = viable
                    .into_iter()
                    .map(|fragment| {
                        let area = parameter_space_area(&fragment_as_trim(&fragment))?.abs();
                        Ok((fragment, area))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                by_area.sort_by(|a, b| b.1.total_cmp(&a.1));
                by_area.into_iter().next().map(|entry| entry.0)
            } else {
                let mut candidates = if seeded.is_empty() { viable } else { seeded };
                candidates.sort_by(|a, b| {
                    a.test_uv
                        .sub(seed)
                        .length()
                        .total_cmp(&b.test_uv.sub(seed).length())
                });
                candidates.into_iter().next()
            };
            if let Some(fragment) = selected {
                chosen_offsets.push(fragment);
            }
        } else {
            // Keep arrangement order stable. Besides making the result
            // reproducible, this matches the reference kernel's Set insertion
            // order when the final manifold-wall guard admits fragments.
            let opening = source_by_id[&carrier.source_face_id];
            let wall_seeds = opening_wall_seeds(
                opening,
                &split_carriers[index].shells[0].faces[0],
                source,
                &opening_set,
                distance,
            )?;
            // The wall-seed exemptions below rest on "the coplanar-rim weld
            // cuts the void hole later" — which only ever happens on PLANAR
            // opening carriers. A curved carrier (e.g. the sphere face used
            // directly as the opening) keeps its un-split fragment forever:
            // exempting it would seal the opening with the source face itself
            // and fabricate a closed hollow solid. Curved carriers therefore
            // get the strict void guard.
            let carrier_planar =
                surface_is_planar(&split_carriers[index].shells[0].faces[0].surface, tolerance)
                    .unwrap_or(false);
            let mut selected_indices = Vec::new();
            for (fragment_index, fragment) in fragments_by_carrier[index].iter().enumerate() {
                if parameter_point_in_face(&fragment_as_trim(fragment), seed, 1e-8)?
                    == PolygonClass::Outside
                {
                    if in_front_of_other_opening(index, fragment.test_point)? {
                        os_debug!(
                            "  wall[{index}.{fragment_index}] skipped: in front of another \
                             opening at ({:.3},{:.3},{:.3})",
                            fragment.test_point.x,
                            fragment.test_point.y,
                            fragment.test_point.z,
                        );
                        continue;
                    }
                    // Wall = the material cross-section exposed at the
                    // opening: every point of a genuine wall fragment lies
                    // WITHIN the offset distance of some retained face (it is
                    // the thickness between that face and its offset skin). A
                    // fragment at ≥ distance from EVERY retained face — and
                    // containing no expected wall seed (an un-split cap
                    // fragment can span both the void AND the wall annulus;
                    // the coplanar-rim weld cuts its hole later) — is the
                    // core's own cross-section: a HOLE in the opening (e.g.
                    // the corner void where a carve's offset circle passes
                    // just inside the opening's corner). Welding it shut
                    // would fabricate a wall over the void.
                    if point_on_offset_skin(
                        fragment.test_point,
                        &retained_faces_with_edges,
                        distance,
                        offset_skin_tolerance,
                    )? && !(carrier_planar
                        && wall_seeds.iter().any(|wall_seed| {
                            parameter_point_in_face(&fragment_as_trim(fragment), *wall_seed, 1e-8)
                                .is_ok_and(|class| class != PolygonClass::Outside)
                        }))
                    {
                        os_debug!(
                            "  wall[{index}.{fragment_index}] skipped: void cross-section at \
                             ({:.3},{:.3},{:.3})",
                            fragment.test_point.x,
                            fragment.test_point.y,
                            fragment.test_point.z,
                        );
                        continue;
                    }
                    selected_indices.push(fragment_index);
                }
            }
            for wall_seed in wall_seeds {
                let fragments = &fragments_by_carrier[index];
                let nearest_containing = fragments
                    .iter()
                    .enumerate()
                    .filter(|(_, fragment)| {
                        parameter_point_in_face(&fragment_as_trim(fragment), wall_seed, 1e-8)
                            .is_ok_and(|class| class != PolygonClass::Outside)
                    })
                    .min_by(|(_, first), (_, second)| {
                        first
                            .test_uv
                            .sub(wall_seed)
                            .length()
                            .total_cmp(&second.test_uv.sub(wall_seed).length())
                    })
                    .map(|(fragment_index, _)| fragment_index);
                // Arrangement fitting can leave the seed just outside every
                // fragment by a small amount. Match the reference kernel's
                // geometric fallback instead of silently omitting the wall
                // fragment and leaving the assembled shell open.
                let nearest = if nearest_containing.is_some() {
                    nearest_containing
                } else {
                    let seed_point = split_carriers[index].shells[0].faces[0]
                        .surface
                        .evaluate(wall_seed.x, wall_seed.y)?;
                    fragments
                        .iter()
                        .enumerate()
                        .min_by(|(_, first), (_, second)| {
                            first
                                .test_point
                                .sub(seed_point)
                                .length()
                                .total_cmp(&second.test_point.sub(seed_point).length())
                        })
                        .map(|(fragment_index, _)| fragment_index)
                };
                if let Some(fragment_index) = nearest {
                    let candidate = &fragments_by_carrier[index][fragment_index];
                    if in_front_of_other_opening(index, candidate.test_point)? {
                        continue;
                    }
                    // Void-cross-section guard. On a PLANAR carrier it applies
                    // to the GEOMETRIC FALLBACK only: a fragment that
                    // genuinely contains this wall seed is wall content by
                    // construction (even when un-split it also spans void —
                    // the rim weld cuts the hole later), but a merely-nearest
                    // fragment must not resurrect a hole in the opening. On a
                    // CURVED carrier no later weld cuts the hole, so even a
                    // seed-containing fragment gets the strict guard.
                    if (nearest_containing.is_none() || !carrier_planar)
                        && point_on_offset_skin(
                            candidate.test_point,
                            &retained_faces_with_edges,
                            distance,
                            offset_skin_tolerance,
                        )?
                    {
                        os_debug!(
                            "  wall[{index}.{fragment_index}] seed-hit skipped: void \
                             cross-section (planar={carrier_planar}) at ({:.3},{:.3},{:.3})",
                            candidate.test_point.x,
                            candidate.test_point.y,
                            candidate.test_point.z,
                        );
                        continue;
                    }
                    let existing = chosen_offsets.iter().cloned().chain(
                        selected_indices
                            .iter()
                            .map(|selected| fragments_by_carrier[index][*selected].clone()),
                    );
                    if !selected_indices.contains(&fragment_index)
                        && !would_overuse_existing_boundary(
                            candidate,
                            existing,
                            &assembly_sources,
                            &imprint,
                            2e-3,
                        )?
                    {
                        selected_indices.push(fragment_index);
                    }
                }
            }
            for fragment_index in selected_indices.iter().copied() {
                chosen_walls.push(fragments_by_carrier[index][fragment_index].clone());
            }
        }
    }
    if debug_enabled() {
        for fragment in &chosen_offsets {
            os_debug!(
                "chosen offset: operand={} src={} test=({:.3},{:.3},{:.3})",
                fragment.operand,
                carriers[fragment.operand as usize].source_face_id,
                fragment.test_point.x,
                fragment.test_point.y,
                fragment.test_point.z,
            );
            for (loop_index, loop_record) in fragment.loops.iter().enumerate() {
                for coedge in &loop_record.coedges {
                    let surface = &fragment.surface;
                    let [c0, c1] = coedge.pcurve.domain().unwrap_or([0.0, 0.0]);
                    let uv0 = coedge.pcurve.evaluate(c0);
                    let uv1 = coedge.pcurve.evaluate(c1);
                    if let (Ok(uv0), Ok(uv1)) = (uv0, uv1) {
                        let p0 = surface.evaluate(uv0.x, uv0.y);
                        let p1 = surface.evaluate(uv1.x, uv1.y);
                        if let (Ok(p0), Ok(p1)) = (p0, p1) {
                            let kind = match &coedge.source {
                                FragmentEdgeSource::Boundary { edge_id, .. } => {
                                    format!("bnd:{edge_id}")
                                }
                                FragmentEdgeSource::SharedBoundary { edge_id, .. } => {
                                    format!("sbnd:{edge_id}")
                                }
                                FragmentEdgeSource::Imprint { piece_id } => {
                                    format!("imp:{piece_id}")
                                }
                                FragmentEdgeSource::Derived { .. } => "derived".to_string(),
                            };
                            os_debug!(
                                "    loop{loop_index} {kind} ({:.4},{:.4},{:.4})..({:.4},{:.4},{:.4})",
                                p0.x, p0.y, p0.z, p1.x, p1.y, p1.z,
                            );
                        }
                    }
                }
            }
        }
        for fragment in &chosen_walls {
            os_debug!(
                "chosen wall: operand={} src={} test=({:.3},{:.3},{:.3})",
                fragment.operand,
                carriers[fragment.operand as usize].source_face_id,
                fragment.test_point.x,
                fragment.test_point.y,
                fragment.test_point.z,
            );
        }
    }
    if chosen_offsets.is_empty() {
        return Err("offset_shell: carrier intersections produced no offset boundary".into());
    }
    // Opening-wall fragments can overlap at dense carrier junctions. The
    // seed-time guard above handles geometric containment; this final pass
    // mirrors the topology-level rule and counts exact fragment edge sources.
    let mut boundary_use_count = HashMap::<[i64; 9], usize>::default();
    for fragment in &chosen_offsets {
        for coedge in fragment
            .loops
            .iter()
            .flat_map(|loop_record| &loop_record.coedges)
        {
            let key = fragment_edge_segment_key(&coedge.source, &assembly_sources, &imprint)?;
            *boundary_use_count.entry(key).or_default() += 1;
        }
    }
    let mut manifold_walls = Vec::new();
    for fragment in chosen_walls {
        let keys = fragment
            .loops
            .iter()
            .flat_map(|loop_record| &loop_record.coedges)
            .map(|coedge| fragment_edge_segment_key(&coedge.source, &assembly_sources, &imprint))
            .collect::<Result<Vec<_>, _>>()?;
        if !keys
            .iter()
            .any(|key| boundary_use_count.get(key).copied().unwrap_or(0) >= 2)
        {
            for key in keys {
                *boundary_use_count.entry(key).or_default() += 1;
            }
            manifold_walls.push(fragment);
        }
    }
    os_debug!(
        "manifold walls kept: {} operands={:?}",
        manifold_walls.len(),
        manifold_walls
            .iter()
            .map(|fragment| fragment.operand)
            .collect::<Vec<_>>(),
    );
    let chosen_walls = manifold_walls;

    let source_fragments = fragment_solid(source, SOURCE_OPERAND, &empty_imprint())?;
    let mut selected = source_fragments
        .into_iter()
        .filter(|fragment| retained.contains(&fragment.source_face_id))
        .collect::<Vec<_>>();
    let mut face_images = selected
        .iter()
        .map(|fragment| OffsetShellFaceImageRecord {
            role: OffsetFaceRole::Source,
            source_face_id: fragment.source_face_id,
        })
        .collect::<Vec<_>>();
    if distance < 0.0 {
        for fragment in &mut selected {
            flip_fragment(fragment)?;
        }
    }
    if distance > 0.0 {
        for fragment in &mut chosen_offsets {
            flip_fragment(fragment)?;
        }
    }
    face_images.extend(
        chosen_offsets
            .iter()
            .map(|fragment| OffsetShellFaceImageRecord {
                role: OffsetFaceRole::Offset,
                source_face_id: carriers[fragment.operand as usize].source_face_id,
            }),
    );
    face_images.extend(
        chosen_walls
            .iter()
            .map(|fragment| OffsetShellFaceImageRecord {
                role: OffsetFaceRole::Wall,
                source_face_id: carriers[fragment.operand as usize].source_face_id,
            }),
    );
    selected.extend(chosen_offsets);
    selected.extend(chosen_walls);
    let mut solid = assemble_open_fragments(
        selected,
        &assembly_sources,
        &imprint,
        2e-3f64.max(scale * 5e-5),
    )?;
    orient_open_solid_faces(&mut solid)?;
    let connector_images = complete_sharp_offset_connectors(
        &mut solid,
        source,
        &face_images,
        distance,
        2e-3f64.max(scale * 5e-5),
    )?;
    os_debug!(
        "sharp connector completion added {} faces",
        connector_images.len()
    );
    if !connector_images.is_empty() {
        // Wall fragments were arranged before the sharp seam was closed, so
        // their prospective inner boundary was an open chain. Rebuild the
        // wall from the now-closed source/offset cycles instead of retaining
        // that pre-connector fragment.
        for shell in &mut solid.shells {
            shell.faces.retain(|face| {
                !matches!(
                    face_images
                        .get(face.id.saturating_sub(1) as usize)
                        .map(|image| image.role),
                    Some(OffsetFaceRole::Wall)
                )
            });
        }
        solid.shells.retain(|shell| !shell.faces.is_empty());
    }
    face_images.extend(connector_images);
    orient_open_solid_faces(&mut solid)?;
    // Weld curved-face offset rims that the wall fragmentation left orphaned
    // into their coplanar opening cap (cylinder / straight-hole shells). This
    // is the seam-rim consistency step the offset pipeline previously lacked.
    let welded_rims = weld_coplanar_orphan_rims(&mut solid, 2e-3f64.max(scale * 5e-5))?;
    os_debug!("welded {welded_rims} coplanar orphan rims into opening caps");
    // Weld oblique-wall offset rims (truncated cone) into a ruled frustum
    // opening-wall band, rebuilding the flat cap the pipeline mis-builds.
    let ruled_rims = weld_ruled_offset_rims(&mut solid, 2e-3f64.max(scale * 5e-5))?;
    os_debug!("welded {ruled_rims} non-coplanar offset rims into ruled bands");
    // Weld leftover coaxial coplanar rim PAIRS (a through-hole's exposed wall
    // thickness at each opening) into new planar annulus faces.
    let pair_rims =
        weld_coplanar_rim_pair_annuli(&mut solid, &mut face_images, 2e-3f64.max(scale * 5e-5))?;
    os_debug!("welded {pair_rims} coplanar rim pairs into annulus walls");
    if debug_enabled() {
        for (shell_index, shell) in solid.shells.iter().enumerate() {
            for face in &shell.faces {
                os_debug!(
                    "POSTWELD shell {} face {} same_sense={} loops={}",
                    shell_index,
                    face.id,
                    face.same_sense,
                    face.loops.len()
                );
                for (loop_index, loop_record) in face.loops.iter().enumerate() {
                    let walk = loop_record
                        .coedges
                        .iter()
                        .map(|coedge| {
                            let spin = (|| -> Result<f64, String> {
                                let [d0, d1] = coedge.pcurve.domain()?;
                                let a = coedge.pcurve.evaluate(d0 + (d1 - d0) * 0.45)?;
                                let b = coedge.pcurve.evaluate(d0 + (d1 - d0) * 0.55)?;
                                let pa = face.surface.evaluate(a.x, a.y)?;
                                let pb = face.surface.evaluate(b.x, b.y)?;
                                Ok(pa.cross(pb).z)
                            })()
                            .unwrap_or(f64::NAN);
                            format!(
                                "{}{}(spin{:+.0})",
                                if coedge.forward { "+" } else { "-" },
                                coedge.edge_id,
                                spin.signum()
                            )
                        })
                        .collect::<Vec<_>>();
                    os_debug!("    loop {} {:?}", loop_index, walk);
                }
            }
        }
    }
    let completion_carriers = carriers.iter().collect::<Vec<_>>();
    let completion_tolerance = 2e-3f64.max(scale * 5e-5);
    // Unify geometric duplicate arc records BEFORE completion: a ring that is
    // already covered by both its faces (frustum + apex cap) must not read as
    // an open boundary, or completion papers over it with duplicate faces.
    let duplicate_welds = weld_duplicate_one_use_arcs(&mut solid, completion_tolerance)?;
    os_debug!("welded {duplicate_welds} duplicate one-use boundary arcs");
    if debug_enabled() {
        let mut use_counts = HashMap::<u64, usize>::default();
        for coedge in solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .flat_map(|face| &face.loops)
            .flat_map(|loop_record| &loop_record.coedges)
        {
            *use_counts.entry(coedge.edge_id).or_default() += 1;
        }
        let open = use_counts.values().filter(|count| **count == 1).count();
        os_debug!(
            "assembled: faces={} open_edges={open}",
            solid
                .shells
                .iter()
                .map(|shell| shell.faces.len())
                .sum::<usize>()
        );
    }
    let completed_images = complete_opening_boundary_cycles(
        &mut solid,
        &completion_carriers,
        &face_images,
        completion_tolerance,
    )?;
    os_debug!("completion added {} faces", completed_images.len());
    face_images.extend(completed_images);
    let solid = merge_same_surface_faces_open(&solid, (1e-7f64).max(scale * 1e-9))?;
    let assembly_tolerance = 2e-3f64.max(scale * 5e-5);
    // No faceted fallback here: an offset shell must be made of real analytic
    // surfaces. Failing loudly beats silently returning a tessellated BREP
    // with destroyed face provenance.
    let mut solid = finalize_assembled_solid(solid, assembly_tolerance)?;
    if ruled_rims + pair_rims > 0 {
        // A ruled-band or rim-pair weld joins independently-oriented skins;
        // put the whole finalized manifold on one normal convention so the
        // shell reports its exact wall volume (coplanar-only shells keep
        // their assembled sense).
        coheres_face_normals(&mut solid)?;
    }
    let solid = solid;
    face_images = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .map(|face| {
            face_images
                .get(face.id.saturating_sub(1) as usize)
                .cloned()
                .ok_or_else(|| "offset_shell: merged face lost provenance".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A bore's end ring carries the bore wall's provenance whichever
    // construction built it (the opening's fragmentation or the rim-pair
    // weld), so its name does not depend on the path the rims took.
    let bore_rings = claim_bore_end_rings(&solid, &mut face_images)?;
    os_debug!("{bore_rings} bore end rings took the bore wall's provenance");
    // Honesty gate: a real rim left one-use is a hole in the shell that the
    // degenerate marking hides from `validate()` (degenerate edges are allowed
    // to be face-local). Refuse such a result instead of shipping a
    // non-watertight solid as valid. A rim is a CLOSED edge whose interior
    // sweeps measurably away from its seam vertex — a genuine circle, not a
    // legitimate pole point placeholder (which stays one-use by design).
    let mut rim_use_counts = HashMap::<u64, usize>::default();
    for coedge in solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .flat_map(|face| &face.loops)
        .flat_map(|loop_record| &loop_record.coedges)
    {
        *rim_use_counts.entry(coedge.edge_id).or_default() += 1;
    }
    if debug_enabled() {
        for (shell_index, shell) in solid.shells.iter().enumerate() {
            for face in &shell.faces {
                let role = face_images
                    .get(face.id.saturating_sub(1) as usize)
                    .map(|image| image.role);
                let c = face.surface.evaluate(0.5, 0.5).unwrap_or_default();
                os_debug!(
                    "DUMPFACE shell {} face {} affine={} role={:?} loops={} center=({:.3},{:.3},{:.3})",
                    shell_index, face.id, face.surface.is_affine().unwrap_or(false), role,
                    face.loops.len(), c.x, c.y, c.z
                );
                for (li, lp) in face.loops.iter().enumerate() {
                    let ids = lp.coedges.iter().map(|c| c.edge_id).collect::<Vec<_>>();
                    os_debug!("    loop {} edges {:?}", li, ids);
                }
            }
        }
        for edge in &solid.edges {
            let uses = rim_use_counts.get(&edge.id).copied().unwrap_or(0);
            let s = edge.curve.evaluate(edge.t0)?;
            let m = edge.curve.evaluate((edge.t0 + edge.t1) * 0.5)?;
            let q = edge.curve.evaluate(edge.t0 + (edge.t1 - edge.t0) * 0.25)?;
            os_debug!(
                "DUMPEDGE edge {} uses={} closed={} deg={} cpts={} cdeg={} knots={:?} start=({:.3},{:.3},{:.3}) q=({:.3},{:.3},{:.3}) mid=({:.3},{:.3},{:.3})",
                edge.id, uses, edge.start_vertex_id == edge.end_vertex_id, edge.degenerate,
                edge.curve.control_points.len(), edge.curve.degree, edge.curve.knots,
                s.x, s.y, s.z, q.x, q.y, q.z, m.x, m.y, m.z
            );
        }
    }
    for edge in &solid.edges {
        if rim_use_counts.get(&edge.id).copied().unwrap_or(0) != 1
            || edge.start_vertex_id != edge.end_vertex_id
        {
            continue;
        }
        let anchor = edge.curve.evaluate(edge.t0)?;
        let sweep_threshold = 1e-4f64.max(scale * 1e-6);
        let sweeps = [0.25, 0.5, 0.75].into_iter().any(|fraction| {
            edge.curve
                .evaluate(edge.t0 + (edge.t1 - edge.t0) * fraction)
                .map(|point| point.sub(anchor).length() > sweep_threshold)
                .unwrap_or(false)
        });
        if sweeps {
            if debug_enabled() {
                let rim_mid = edge.curve.evaluate((edge.t0 + edge.t1) * 0.5)?;
                os_debug!(
                    "PROBE orphan rim edge {} anchor=({:.4},{:.4},{:.4}) mid=({:.4},{:.4},{:.4})",
                    edge.id,
                    anchor.x,
                    anchor.y,
                    anchor.z,
                    rim_mid.x,
                    rim_mid.y,
                    rim_mid.z
                );
                for (shell_index, shell) in solid.shells.iter().enumerate() {
                    for face in &shell.faces {
                        let uses_rim = face
                            .loops
                            .iter()
                            .flat_map(|l| &l.coedges)
                            .any(|c| c.edge_id == edge.id);
                        let affine = face.surface.is_affine().unwrap_or(false);
                        let on = edge_on_surface(edge, &face.surface, 2e-3f64.max(scale * 5e-5))
                            .unwrap_or(false);
                        let proj = project_point_to_surface(&face.surface, rim_mid)?;
                        let contains = if on {
                            parameter_point_in_face(
                                face,
                                Vec2 {
                                    x: proj.u,
                                    y: proj.v,
                                },
                                2e-3f64.max(scale * 5e-5),
                            )? != PolygonClass::Outside
                        } else {
                            false
                        };
                        if on || uses_rim {
                            os_debug!(
                                "  face {} shell {} affine={} uses_rim={} on_surf={} proj_dist={:.5} contains={}",
                                face.id, shell_index, affine, uses_rim, on, proj.distance, contains
                            );
                        }
                    }
                }
            }
            return Err(format!(
                "offset_shell: unwelded rim leaves a non-watertight shell \
                 (one-use closed edge {} near ({:.3},{:.3},{:.3})); this \
                 curved-solid opening is not yet supported",
                edge.id, anchor.x, anchor.y, anchor.z
            ));
        }
    }
    Ok(OffsetShellResultRecord { solid, face_images })
}

/// Run offset shell with stable counters and validation diagnostics suitable
/// for automated regressions and UI presentation.
pub fn offset_shell_with_diagnostics(
    source: &BrepSolid,
    opening_face_ids: &[u64],
    distance: f64,
    tolerances: Option<KernelTolerances>,
) -> Result<KernelOutcome<OffsetShellResultRecord>, String> {
    let policy = tolerances.unwrap_or_else(|| KernelTolerances::for_solid(source, 1e-7));
    policy.check()?;
    let mut diagnostics = KernelDiagnostics::default();
    diagnostics.count_n(
        "collect.source_faces",
        source
            .shells
            .iter()
            .map(|shell| shell.faces.len() as u64)
            .sum(),
    );
    diagnostics.count_n("collect.opening_faces", opening_face_ids.len() as u64);
    diagnostics.measure_max("offset.distance", distance.abs());
    diagnostics.measure_max("tolerance.model", policy.model);
    let result = offset_shell(source, opening_face_ids, distance)?;
    diagnostics.count_n(
        "sew.output_faces",
        result
            .solid
            .shells
            .iter()
            .map(|shell| shell.faces.len() as u64)
            .sum(),
    );
    diagnostics.count_n("sew.face_images", result.face_images.len() as u64);
    let validation = result.solid.validate_detailed(&policy);
    diagnostics.measure_max("validate.max_pcurve_error", validation.max_pcurve_error);
    diagnostics.count_n("validate.issues", validation.issues.len() as u64);
    diagnostics.count_n(
        "validate.wire_warnings",
        validation.wire_warnings.len() as u64,
    );
    for warning in validation.wire_warnings {
        diagnostics.event(
            DiagnosticSeverity::Warning,
            KernelStage::Validate,
            "validate.uv_wire",
            warning.message,
        );
    }
    if !validation.issues.is_empty() {
        for issue in &validation.issues {
            diagnostics.event(
                DiagnosticSeverity::Error,
                KernelStage::Validate,
                "validate.brep",
                issue.message.clone(),
            );
        }
        return Err(format!(
            "offset_shell: invalid diagnostic result: {:?}",
            validation.issues
        ));
    }
    Ok(KernelOutcome {
        value: result,
        diagnostics,
    })
}
