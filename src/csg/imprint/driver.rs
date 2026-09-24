use crate::{KernelRefusal, KernelStage, OrRefuse, RefusalClass};
use super::*;

pub fn build_imprints(
    solid_a: &BrepSolid,
    solid_b: &BrepSolid,
    options: &ImprintOptions,
) -> Result<ImprintResultRecord, KernelRefusal> {
    let mut section_evidence = false;
    let edges = edge_map(solid_a, solid_b);
    let mut builder = ImprintBuilder {
        edges,
        tolerance: options.tolerance,
        barrier_edges: HashSet::default(),
        overlap_ridden_edges: HashSet::default(),
        scale: solid_scale(solid_a).max(solid_scale(solid_b)),
        vertices: Vec::new(),
        vertex_radii: HashMap::default(),
        pieces: Vec::new(),
        by_face: HashMap::default(),
        edge_splits: HashMap::default(),
        next_id: 1,
    };
    let first_faces = faces(solid_a, 0);
    let second_faces = faces(solid_b, 1);
    // Per-face trim-window carriers, computed once: tighter BVH bounds, and
    // the seed/march stages walk the window instead of the full carrier
    // (identical geometry and parameterization inside the window).
    let first_restricted: Vec<Option<NurbsSurface>> = first_faces
        .iter()
        .map(|tagged| restricted_carrier(tagged.face))
        .collect();
    let second_restricted: Vec<Option<NurbsSurface>> = second_faces
        .iter()
        .map(|tagged| restricted_carrier(tagged.face))
        .collect();
    let first_bounds = first_faces
        .iter()
        .zip(&first_restricted)
        .map(|(tagged, restricted)| {
            face_bounds(
                tagged.face,
                restricted.as_ref(),
                &builder.edges,
                tagged.operand,
                options.tolerance,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let second_bounds = second_faces
        .iter()
        .zip(&second_restricted)
        .map(|(tagged, restricted)| {
            face_bounds(
                tagged.face,
                restricted.as_ref(),
                &builder.edges,
                tagged.operand,
                options.tolerance,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let second_bvh = Bvh::build(&second_bounds);
    // Per-face classifier data and edge lists/subcurves, computed once and
    // reused across every pair the face participates in.
    let first_classify = first_faces
        .iter()
        .map(|tagged| SurfaceClassifyData::build(&tagged.face.surface, options.tolerance))
        .collect::<Result<Vec<_>, _>>().or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
    let second_classify = second_faces
        .iter()
        .map(|tagged| SurfaceClassifyData::build(&tagged.face.surface, options.tolerance))
        .collect::<Result<Vec<_>, _>>().or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
    let mut face_edge_lists: HashMap<FaceKey, Vec<&EdgeRecord>> = HashMap::default();
    for tagged in first_faces.iter().chain(second_faces.iter()) {
        face_edge_lists.insert(tagged.key(), face_edges(*tagged, &builder.edges)?);
    }
    let mut subcurves: HashMap<(u8, u64), NurbsCurve> = HashMap::default();
    for (&key, edge) in &builder.edges {
        if !edge.degenerate {
            // Failures fall through: the use sites recompute and surface the
            // original error exactly where the uncached code did.
            if let Ok(curve) = edge_subcurve(edge) {
                subcurves.insert(key, curve);
            }
        }
    }
    let cached_subcurve = |operand: u8, edge: &EdgeRecord| -> Result<NurbsCurve, KernelRefusal> {
        match subcurves.get(&(operand, edge.id)) {
            Some(curve) => Ok(curve.clone()),
            None => edge_subcurve(edge),
        }
    };
    let mut profile = ImprintProfile::new();
    let mut tangent_nodes: Vec<Vec3> = Vec::new();
    let mut paired = Vec::new();
    for (first_index, first) in first_faces.iter().enumerate() {
        let first = *first;
        paired.clear();
        second_bvh.overlapping(
            first_bounds[first_index],
            options.tolerance * 100.0,
            &mut paired,
        );
        paired.sort_unstable();
        let debug_pairs = std::env::var("BREP_DEBUG_PAIRS").is_ok();
        if debug_pairs {
            eprintln!(
                "bvh first_face={} -> {} candidate pairs {:?}",
                first.face.id,
                paired.len(),
                paired
                    .iter()
                    .map(|&index| second_faces[index].face.id)
                    .collect::<Vec<_>>()
            );
        }
        for &second_index in &paired {
            let second = second_faces[second_index];
            profile.pairs += 1;
            // Set when this pair's section passes through an isolated TANGENT
            // NODE inside both trims (see the classification below). The march
            // is attempted anyway — that is the whole point — but a marcher
            // that cannot get through a node must refuse in the tangent-node
            // class naming the node, not leak its own step-budget message.
            let mut tangent_node: Option<Vec3> = None;
            let mut lap_start = None;
            profile.lap(&mut lap_start);
            let pair_classification = classify_surface_pair_cached(
                &first.face.surface,
                &first_classify[first_index],
                &second.face.surface,
                &second_classify[second_index],
                options.tolerance,
                PAIR_ANGULAR_TOLERANCE,
            ).or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
            profile.classify += profile.lap(&mut lap_start);
            if pair_classification.relation == SurfacePairRelation::Disjoint {
                if debug_pairs {
                    eprintln!(
                        "pair {}x{}: DISJOINT-cull sep={:.4e}",
                        first.face.id, second.face.id, pair_classification.minimum_separation
                    );
                }
                continue;
            }
            // Coincident carriers are never marched: their true intersection
            // is a 2D region, not a curve, so anything the marcher traces on
            // them is noise ("similar faces we do not intersect" — Golovanov
            // §6.2).  The sampled classification catches coincident pairs
            // the strict reconstruction test misses (partial overlaps,
            // differing parameterizations); both route to the boundary-curve
            // exchange below.
            let is_cosurface = pair_classification.relation == SurfacePairRelation::Cosurface
                || cosurface_pair(&first.face.surface, &second.face.surface, options.tolerance)?;
            profile.cosurface += profile.lap(&mut lap_start);
            if is_cosurface {
                if debug_pairs {
                    eprintln!("pair {}x{}: cosurface", first.face.id, second.face.id);
                }
                for edge in &face_edge_lists[&second.key()] {
                    if !edge.degenerate {
                        builder.process_curve(
                            cached_subcurve(second.operand, edge)?,
                            first,
                            second,
                            &[first],
                            &[first],
                            false,
                        )?;
                    }
                }
                for edge in &face_edge_lists[&first.key()] {
                    if !edge.degenerate {
                        builder.process_curve(
                            cached_subcurve(first.operand, edge)?,
                            first,
                            second,
                            &[second],
                            &[second],
                            false,
                        )?;
                    }
                }
                profile.process_curve += profile.lap(&mut lap_start);
                continue;
            }

            for edge in &face_edge_lists[&first.key()] {
                if !edge.degenerate {
                    let curve = cached_subcurve(first.operand, edge)?;
                    if curve_lies_on_surface(&curve, &second.face.surface, options.tolerance)? {
                        builder.process_curve(curve, first, second, &[second], &[second], false)?;
                    }
                }
            }
            for edge in &face_edge_lists[&second.key()] {
                if !edge.degenerate {
                    let curve = cached_subcurve(second.operand, edge)?;
                    if curve_lies_on_surface(&curve, &first.face.surface, options.tolerance)? {
                        builder.process_curve(curve, first, second, &[first], &[first], false)?;
                    }
                }
            }
            profile.lies_on += profile.lap(&mut lap_start);

            let planar_iso = planar_iso_intersection(
                &first.face.surface,
                &second.face.surface,
                options.tolerance,
            )?;
            profile.planar_iso += profile.lap(&mut lap_start);
            if let Some(curve) = planar_iso {
                if debug_pairs {
                    eprintln!("pair {}x{}: planar_iso", first.face.id, second.face.id);
                }
                builder.process_curve(
                    curve,
                    first,
                    second,
                    &[first, second],
                    &[first, second],
                    true,
                )?;
                profile.process_curve += profile.lap(&mut lap_start);
                continue;
            }

            // Recognized analytic pairs produce their exact intersection
            // curves (lines, circles, ellipses) directly — no marching, no
            // polyline fitting, no chord-sag drift. An empty result is a
            // proof of non-intersection and also skips the marcher.
            let analytic = crate::intersect_analytic_pair(
                &first.face.surface,
                &second.face.surface,
                options.tolerance,
            );
            profile.analytic += profile.lap(&mut lap_start);
            if let Some(curves) = analytic {
                if debug_pairs {
                    eprintln!(
                        "pair {}x{}: analytic x{}",
                        first.face.id,
                        second.face.id,
                        curves.len()
                    );
                }
                for curve in curves {
                    builder.process_curve(
                        curve,
                        first,
                        second,
                        &[first, second],
                        &[first, second],
                        true,
                    )?;
                }
                profile.process_curve += profile.lap(&mut lap_start);
                continue;
            }

            // A pair whose every touching sample is TANGENTIAL cannot
            // contain a transverse intersection curve: the marcher would
            // walk the tangency band's noise, which neither closes nor
            // reaches a boundary (the distance-0 glue-extrude runaway).
            // Shared topology at a tangential contact comes from the
            // boundary-curve exchange above ("similar faces we do not
            // intersect" extended to tangential contacts — Golovanov §6.2;
            // where surfaces CROSS through a tangency, off-line samples
            // have non-parallel normals and the pair still marches).
            if pair_classification.relation == SurfacePairRelation::NearTangent
                && pair_classification.tangential_only
            {
                // The coarse 5×5 classifier can flag `tangential_only` off an
                // incidental tangential KISS between two curved carriers and
                // miss the transverse loop where they actually cross (two
                // overlapping tori: their tubes cross while their inner walls
                // just touch). Before honouring the skip, a GATED supplemental
                // detector (denser two-sided seeding, tangency-band seeds
                // rejected, trace-exhaustion swallowed, transverse-only
                // branches) checks whether a genuine transverse curve of
                // meaningful length lies inside BOTH trims.
                //
                // Such a pair's intersection is SINGULAR where a tangency
                // sits on it, and what the imprint can do about that depends
                // entirely on the SHAPE of the tangency — see
                // `imprint/tangent_contact.rs`. An isolated NODE (two branches
                // crossing, as every equal-radius pair produces) is already
                // assembled correctly by the ordinary march plus the 2D
                // arrangement's pinch carving; an EXTENDED contact along a
                // whole curve has no section to imprint at all and tears the
                // shell. Both are classified below and only the second is
                // refused. Skipping either silently would DROP the
                // intersection (torus∪torus double-counts; torus−torus removes
                // nothing — the two structural bugs the semantic oracle
                // found), so nothing here ever falls through quietly.
                let first_march = first_restricted[first_index]
                    .as_ref()
                    .unwrap_or(&first.face.surface);
                let second_march = second_restricted[second_index]
                    .as_ref()
                    .unwrap_or(&second.face.surface);
                let supplemental = intersect_surfaces_supplemental(
                    first_march,
                    second_march,
                    &SurfaceIntersectionOptions {
                        tolerance: options.tolerance,
                        maximum_step: options.maximum_ssi_step,
                        ..Default::default()
                    },
                ).or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
                let mut dropped_length = 0.0f64;
                let mut clipped: Vec<Vec<Vec3>> = Vec::new();
                for branch in &supplemental {
                    for run in clip_branch_to_trims(&branch.points, first, second)? {
                        if run.len() < 2 {
                            continue;
                        }
                        let length: f64 = run
                            .windows(2)
                            .map(|pair| pair[1].sub(pair[0]).length())
                            .sum();
                        dropped_length = dropped_length.max(length);
                        clipped.push(run);
                    }
                }
                if dropped_length <= options.tolerance * 100.0 {
                    if debug_pairs {
                        eprintln!(
                            "pair {}x{}: tangential-only contact, march skipped",
                            first.face.id, second.face.id
                        );
                    }
                    continue;
                }
                // A transverse curve EXISTS. What would make it unimprintable is
                // a tangent NODE on it — and the node need not lie inside the
                // trims. Two equal-radius pipe arms are tangent to each other
                // where their axes' common perpendicular leaves the junction,
                // and a joint ball wider than the arms trims that crotch off
                // both faces; what is left inside the trims is an ordinary
                // crossing. So the question is not "does a transverse curve
                // exist" but "does the curve INSIDE BOTH TRIMS reach a
                // tangency". Only the latter is the checkerboard case.
                let mut tangency_inside_trims = false;
                let mut tangency_witness = Vec3::default();
                'clipped: for run in &clipped {
                    for (index, &point) in run.iter().enumerate() {
                        if pair_normals_parallel_at(first, second, point)? {
                            tangency_inside_trims = true;
                            tangency_witness = point;
                            if debug_pairs {
                                eprintln!(
                                    "pair {}x{}: TANGENCY at run index {}/{} ({:.6},{:.6},{:.6}) dropped_len={:.4e}",
                                    first.face.id, second.face.id, index, run.len(),
                                    point.x, point.y, point.z, dropped_length
                                );
                            }
                            break 'clipped;
                        }
                    }
                }
                if tangency_inside_trims {
                    // WHAT SHAPE is the tangency? The witness is only a marched
                    // sample within the transverse-seed angular gate, so it is
                    // REFINED onto the contact before being classified — at the
                    // raw witness a G1 cylinder/torus join and a genuine
                    // equal-radius node are eighteen-fold apart, which would be
                    // a band; at the refined contact they are thirty orders
                    // apart, which is a rank question. See
                    // `imprint/tangent_contact.rs`.
                    let contact = classify_tangent_contact(
                        &first.face.surface,
                        &second.face.surface,
                        tangency_witness,
                        options.tolerance,
                    )?;
                    if debug_pairs {
                        eprintln!(
                            "pair {}x{}: contact classification {:?}",
                            first.face.id, second.face.id, contact
                        );
                    }
                    match contact {
                        // An isolated node: the section is a curve everywhere
                        // but that one point, so the ordinary march below runs
                        // and the 2D arrangement carves the pinch on both
                        // faces. `tangent_node` records it so a march that
                        // cannot get through still refuses in THIS class rather
                        // than leaking the marcher's own message.
                        Some(contact) if contact.is_isolated_node() => {
                            tangent_node = Some(contact.point);
                            tangent_nodes.push(contact.point);
                        }
                        // An extended contact (rank-deficient) or a witness we
                        // could not refine onto any contact at all (None — so
                        // nothing is proven and the pre-classification refusal
                        // stands). Neither has a section curve the imprint can
                        // represent.
                        other => {
                            let shape = match other {
                                Some(contact) => contact.describe(),
                                None => format!(
                                    "the tangency near ({:.6},{:.6},{:.6}) could not be refined \
                                     onto a contact, so its shape is unproven",
                                    tangency_witness.x, tangency_witness.y, tangency_witness.z
                                ),
                            };
                            return Err(KernelRefusal::new(
                                RefusalClass::TangentNodeSingularity,
                                KernelStage::Intersect,
                                format!(
                                    "boolean: unsupported singular/tangent-node surface intersection \
                                     between faces {} and {}: {shape}",
                                    first.face.id, second.face.id
                                ),
                            ));
                        }
                    }
                }
                if debug_pairs && tangent_node.is_none() {
                    // Transverse the whole way inside both trims: the coarse
                    // 5x5 classifier only saw the tangency the trims cut away.
                    eprintln!(
                        "pair {}x{}: classified tangential-only, but the curve inside both \
                         trims is transverse ({dropped_length:.4}) — marching",
                        first.face.id, second.face.id
                    );
                }
            }

            // The trim-window carriers: same surface and parameterization
            // over the window, so hit (u, v) values remain valid on the
            // originals; out-of-window intersections could never survive
            // clip_branch_to_trims and are not walked at all.
            let first_march = first_restricted[first_index]
                .as_ref()
                .unwrap_or(&first.face.surface);
            let second_march = second_restricted[second_index]
                .as_ref()
                .unwrap_or(&second.face.surface);
            let mut seed_points = Vec::new();
            // Smallest |edge_tangent · surface_normal| over accepted seeds — the
            // local grazing measure at the trim crossings (near 0 = the edge
            // pierces the other surface tangentially). Gates the near-tangent
            // clip-order rescue below to genuinely grazing pairs.
            let mut min_seed_tangency = f64::INFINITY;
            for (face, other_march, other) in
                [(first, second_march, second), (second, first_march, first)]
            {
                for edge in &face_edge_lists[&face.key()] {
                    if edge.degenerate {
                        continue;
                    }
                    for hit in intersect_curve_surface(&edge.curve, other_march, options.tolerance).or_refuse(KernelStage::Intersect, "intersect_curve_surface")?
                    {
                        if hit.t < edge.t0 - 1e-9 || hit.t > edge.t1 + 1e-9 {
                            continue;
                        }
                        let tangent = edge.curve.derivatives(hit.t, 1).or_refuse(KernelStage::Intersect, "derivatives")?[1].normalized().or_refuse(KernelStage::Intersect, "normalized")?;
                        let normal = match other.face.surface.normal(hit.u, hit.v) {
                            Ok(normal) => normal,
                            Err(_) => continue,
                        };
                        if debug_pairs {
                            eprintln!(
                                "pair {}x{}: seed edge {} t={:.6} p=({:.5},{:.5},{:.5}) |tan.n|={:.4} {}",
                                first.face.id,
                                second.face.id,
                                edge.id,
                                hit.t,
                                hit.point.x,
                                hit.point.y,
                                hit.point.z,
                                tangent.dot(normal).abs(),
                                if tangent.dot(normal).abs() >= 0.1 { "ACCEPT" } else { "reject" }
                            );
                        }
                        if tangent.dot(normal).abs() >= 0.1 {
                            seed_points.push(hit.point);
                            min_seed_tangency = min_seed_tangency.min(tangent.dot(normal).abs());
                            section_evidence = true;
                        }
                    }
                }
            }
            profile.seeds += profile.lap(&mut lap_start);
            profile.marched_pairs += 1;
            if debug_pairs {
                eprintln!(
                    "pair {}x{}: marching (relation {:?})...",
                    first.face.id, second.face.id, pair_classification.relation
                );
            }
            let marched = intersect_surfaces(
                first_march,
                second_march,
                &SurfaceIntersectionOptions {
                    tolerance: options.tolerance,
                    maximum_step: march_maximum_step(
                        &pair_classification,
                        options.maximum_ssi_step,
                        builder.scale,
                    ),
                    seed_points: seed_points.clone(),
                    ..Default::default()
                },
            );
            // A pair carrying a tangent node keeps its OWN refusal class when
            // the march fails: the node is why the trace cannot close, and the
            // equal-radius torus pair depends on being told so rather than on
            // reading a step-budget message it cannot act on.
            let marched = match (marched, tangent_node) {
                (Err(error), Some(node)) => {
                    return Err(KernelRefusal::new(
                        RefusalClass::TangentNodeSingularity,
                        KernelStage::Intersect,
                        format!(
                            "boolean: unsupported singular/tangent-node surface intersection \
                             between faces {} and {}: the section through the tangent node at \
                             ({:.6},{:.6},{:.6}) could not be marched ({error})",
                            first.face.id, second.face.id, node.x, node.y, node.z
                        ),
                    ));
                }
                (marched, _) => marched,
            };
            let mut branches = marched
            .map_err(|error| {
                format!(
                    "{error} (marching faces {} and {})",
                    first.face.id, second.face.id
                )
            }).or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
            profile.march += profile.lap(&mut lap_start);
            // MARCH-ORDER SWAP RESCUE (hatch BREP_MARCH_SWAP_RESCUE=0).
            // `intersect_surfaces` is not order-symmetric: the coupled Newton
            // trace can fail to start/continue from a valid seed when the two
            // surfaces are presented in one operand order yet succeed in the
            // other. This is the sole reason t217's `subtract(sphere, step)`
            // fails while every other op passes — pair 118x842 (sphere-first)
            // marches ZERO branches, while 842x118 (step-first, the working
            // a\b order) marches the section from the IDENTICAL accepted pierce
            // seeds. When the forward order returns no branch AND an accepted
            // transverse pierce seed exists (so a real section provably crosses
            // both trims), retry the march with the surfaces swapped — i.e.
            // reproduce the exact call the working operand order makes for this
            // pair (measured: seed_only alone does NOT recover it — the section
            // is found from an auto-grid start, not from the near-tangent pierce
            // seeds, which the seed normal-cross gate rejects). The returned
            // branch points are 3D and therefore order-independent, so no
            // parameter remap is needed; they flow through the identical clip /
            // process_curve gates below, which drop any out-of-trim or
            // sub-length run exactly as today. The rescue only ever runs when
            // the forward order found NOTHING, so it cannot alter a pair that
            // already marched.
            if branches.is_empty()
                && !seed_points.is_empty()
                && pair_classification.relation == SurfacePairRelation::Candidate
                && std::env::var("BREP_MARCH_SWAP_RESCUE").as_deref() != Ok("0")
            {
                // FAIL-SOFT: the forward order already returned Ok(empty); a
                // swapped-march error (trace-exhaustion is a real error class —
                // `intersect_surfaces_supplemental` swallows it for exactly this
                // reason) must NOT convert that graceful empty into a hard error.
                // On Err, keep the (empty) forward result and carry on.
                let swapped = intersect_surfaces(
                    second_march,
                    first_march,
                    &SurfaceIntersectionOptions {
                        tolerance: options.tolerance,
                        maximum_step: march_maximum_step(
                            &pair_classification,
                            options.maximum_ssi_step,
                            builder.scale,
                        ),
                        seed_points: seed_points.clone(),
                        ..Default::default()
                    },
                )
                .unwrap_or_default();
                if debug_pairs {
                    eprintln!(
                        "pair {}x{}: MARCH-SWAP RESCUE attempt -> {} branches, pts {:?}",
                        first.face.id,
                        second.face.id,
                        swapped.len(),
                        swapped.iter().map(|b| b.points.len()).collect::<Vec<_>>()
                    );
                }
                if !swapped.is_empty() {
                    branches = swapped;
                }
            }
            // NEAR-TANGENT CLIP-ORDER RESCUE (hatch BREP_MARCH_SWAP_CLIP_RESCUE=0).
            // Companion to the empty-branch MARCH-SWAP RESCUE above:
            // `intersect_surfaces` is order-asymmetric not only in WHETHER it
            // marches, but in the exact sample positions of the section
            // polyline. On a NEAR-TANGENT graze, `clip_branch_to_trims`
            // classifies those samples against the mutual trims, and a sample
            // landing just past the near-tangent boundary is dropped as a
            // false-Outside that TRUNCATES the clipped section. The two operand
            // orders drop DIFFERENT near-tangent tail samples, so one order
            // yields a section ~0.1-0.2mm shorter at ONE endpoint (t660: b\a's
            // step-first order clips pairs 223x105/399x105 ~0.22/0.14mm short of
            // the section a\b's cyl-first order keeps, stranding edge 173
            // one-use). Near-tangent clip errors are almost exclusively
            // false-Outside (a point truly outside a trim rarely projects to an
            // in-trim uv), so the order with the LONGER clipped section suffered
            // fewer drops and is the more complete one. When the swapped order
            // marches the SAME branch structure with a meaningfully longer
            // clipped total, adopt its branches (3D points, so they flow through
            // the identical clip/process_curve gates below). Gated to Candidate
            // pairs (the near-tangent/ambiguous class; clean Transverse
            // crossings clip identically in both orders and never differ) with
            // an accepted GRAZING pierce seed (min |tan·n| < 0.5 — a real
            // section provably crosses, and does so near-tangentially, the only
            // regime where the clip wobbles; this also bounds the extra march to
            // grazing pairs), and requires a >0.1%-of-length margin so
            // raw-sampling jitter (t660: 0.02%) cannot flip a clean pair.
            // Fail-soft: any swapped-march or comparison-clip error keeps the
            // forward result.
            if !branches.is_empty()
                && !seed_points.is_empty()
                && min_seed_tangency < 0.5
                && pair_classification.relation == SurfacePairRelation::Candidate
                && std::env::var("BREP_MARCH_SWAP_CLIP_RESCUE").as_deref() != Ok("0")
            {
                let swapped = intersect_surfaces(
                    second_march,
                    first_march,
                    &SurfaceIntersectionOptions {
                        tolerance: options.tolerance,
                        maximum_step: march_maximum_step(
                            &pair_classification,
                            options.maximum_ssi_step,
                            builder.scale,
                        ),
                        seed_points: seed_points.clone(),
                        ..Default::default()
                    },
                )
                .unwrap_or_default();
                // Only a swap that reproduces the SAME branch count — a
                // refined-endpoint variant of the same section, not a different
                // branch decomposition (guards against adopting a spurious
                // extra branch as "longer").
                if !swapped.is_empty() && swapped.len() == branches.len() {
                    let totals = (|| -> Result<(f64, f64), KernelRefusal> {
                        let mut fwd = 0.0;
                        for b in &branches {
                            let refined = insert_seed_points_into_branch(
                                &b.points,
                                &seed_points,
                                options.tolerance,
                            );
                            for run in clip_branch_to_trims(&refined, first, second)? {
                                fwd += run
                                    .windows(2)
                                    .map(|p| p[1].sub(p[0]).length())
                                    .sum::<f64>();
                            }
                        }
                        let mut swp = 0.0;
                        for b in &swapped {
                            let refined = insert_seed_points_into_branch(
                                &b.points,
                                &seed_points,
                                options.tolerance,
                            );
                            for run in clip_branch_to_trims(&refined, first, second)? {
                                swp += run
                                    .windows(2)
                                    .map(|p| p[1].sub(p[0]).length())
                                    .sum::<f64>();
                            }
                        }
                        Ok((fwd, swp))
                    })();
                    if let Ok((fwd_clip, swp_clip)) = totals {
                        let margin = fwd_clip.max(swp_clip) * 1.0e-3;
                        let adopt = swp_clip > fwd_clip + margin;
                        if debug_pairs {
                            eprintln!(
                                "pair {}x{}: CLIP-SWAP RESCUE fwd_clip={:.6} swp_clip={:.6} margin={:.6}{}",
                                first.face.id,
                                second.face.id,
                                fwd_clip,
                                swp_clip,
                                margin,
                                if adopt { " ADOPT" } else { "" }
                            );
                        }
                        if adopt {
                            branches = swapped;
                        }
                    }
                }
            }
            if branches.iter().any(|branch| branch.points.len() >= 2) {
                section_evidence = true;
            }
            if debug_pairs {
                eprintln!(
                    "pair {}x{}: MARCH {} branches, pts {:?}",
                    first.face.id,
                    second.face.id,
                    branches.len(),
                    branches.iter().map(|b| b.points.len()).collect::<Vec<_>>()
                );
            }
            for branch in branches {
                // The pierce seeds are the section's exact trim-crossing
                // points — insert them so no trim interval shorter than the
                // march step is invisible to the point-classification clip.
                let refined_points =
                    insert_seed_points_into_branch(&branch.points, &seed_points, options.tolerance);
                for run in clip_branch_to_trims(&refined_points, first, second)? {
                    if debug_pairs {
                        eprintln!(
                            "pair {}x{}: clip run len_pts={} length={:.4e}",
                            first.face.id,
                            second.face.id,
                            run.len(),
                            run.windows(2)
                                .map(|pair| pair[1].sub(pair[0]).length())
                                .sum::<f64>()
                        );
                    }
                    if run.len() < 2 {
                        continue;
                    }
                    let length: f64 = run
                        .windows(2)
                        .map(|pair| pair[1].sub(pair[0]).length())
                        .sum();
                    if length <= options.tolerance * 100.0 {
                        continue;
                    }
                    if branch_follows_shared_boundary(
                        &run,
                        first,
                        second,
                        &builder.edges,
                        options.tolerance,
                        builder.scale,
                    )? {
                        if debug_pairs {
                            eprintln!(
                                "pair {}x{}: run dropped (follows shared boundary)",
                                first.face.id, second.face.id
                            );
                        }
                        continue;
                    }
                    let pieces_before = builder.pieces.len();
                    let chunk_points = options.fit_chunk_points.unwrap_or(run.len()).max(2);
                    let mut start = 0;
                    while start + 1 < run.len() {
                        let end = (start + chunk_points - 1).min(run.len() - 1);
                        let fit = fit_polyline(
                            &run[start..=end],
                            options.tolerance.max(1e-7),
                            options.maximum_fit_points,
                            options.local_fit,
                        ).or_refuse(KernelStage::Intersect, "csg.imprint.driver")?;
                        builder.process_curve(
                            fit.curve,
                            first,
                            second,
                            &[first, second],
                            &[first, second],
                            true,
                        )?;
                        start = end;
                    }
                    if debug_pairs {
                        eprintln!(
                            "pair {}x{}: run -> {} pieces",
                            first.face.id,
                            second.face.id,
                            builder.pieces.len() - pieces_before
                        );
                    }
                }
            }
            profile.clip_and_fit += profile.lap(&mut lap_start);
        }
    }
    profile.report();
    // SELF-TOUCH SPLIT: a face whose loops touch at an edge interior (a hole
    // tangent to a fillet setback) gets a vertex minted at the touch on BOTH
    // edges, so the fragment arrangement's pinch resolution assembles — see
    // `imprint/self_touch.rs`. Escape hatch: BREP_SELF_TOUCH_SPLIT=0.
    if std::env::var("BREP_SELF_TOUCH_SPLIT").as_deref() != Ok("0") {
        builder.split_self_touching_loops(&first_faces, &face_edge_lists, &cached_subcurve)?;
        builder.split_self_touching_loops(&second_faces, &face_edge_lists, &cached_subcurve)?;
    }
    // PIECE-ENDPOINT EXCHANGE post-pass: with every pair's process_curve
    // done, the piece set is final for the ridden-edge class — imprint each
    // open riding piece's junction endpoints onto the ridden edges (see the
    // method doc; hatch BREP_OVERLAP_PIECE_ENDPOINT_SPLIT=0).
    builder.exchange_piece_endpoint_junctions()?;
    let mut edge_splits = builder
        .edge_splits
        .into_iter()
        .map(|((operand, edge_id), mut parameters)| {
            parameters.sort_by(f64::total_cmp);
            EdgeSplitRecord {
                operand,
                edge_id,
                parameters,
            }
        })
        .collect::<Vec<_>>();
    edge_splits.sort_by_key(|record| (record.operand, record.edge_id));
    let mut by_face = builder
        .by_face
        .into_iter()
        .map(|(face, piece_ids)| FaceImprints {
            operand: face.operand,
            face_id: face.face_id,
            piece_ids,
        })
        .collect::<Vec<_>>();
    by_face.sort_by_key(|record| (record.operand, record.face_id));
    let section_evidence = section_evidence || !builder.pieces.is_empty();
    let mut result = ImprintResultRecord {
        tangent_nodes,
        vertices: builder.vertices,
        pieces: builder.pieces,
        by_face,
        edge_splits,
        barrier_edges: builder.barrier_edges.into_iter().collect(),
        section_evidence,
    };
    // COINCIDENT-PIECE MERGE (problemInbox equator-tangent, and the generic
    // one-circle-from-many-pairs class): the SAME section curve can be minted
    // by several pairs — a cosurface boundary-edge copy (the cylinder cap
    // ring lying ON the inscribed sphere) AND the cap-plane's analytic
    // section ring are one circle minted twice, each carrying only its own
    // pair's supports/pcurves. Assembly then builds duplicate edges that
    // cannot both be two-use → one-use strands. Merge pieces whose curves
    // coincide along their whole span (bidirectional max deviation within
    // the weld band): keep the first, union the supports/pcurves/by_face
    // registrations of the rest into it. Escape hatch:
    // BREP_COINCIDENT_PIECE_MERGE=0.
    if std::env::var("BREP_COINCIDENT_PIECE_MERGE").as_deref() != Ok("0") {
        let weld = assembler_weld(options.tolerance).max(options.tolerance * 10.0);
        let mut removed: Vec<u64> = Vec::new();
        let mut index = 0;
        while index < result.pieces.len() {
            let mut other = index + 1;
            while other < result.pieces.len() {
                let coincide = {
                    let a = &result.pieces[index];
                    let b = &result.pieces[other];
                    max_curve_deviation(&a.curve, &b.curve)? <= weld
                        && max_curve_deviation(&b.curve, &a.curve)? <= weld
                };
                if coincide {
                    let absorbed = result.pieces.remove(other);
                    removed.push(absorbed.id);
                    let keeper = &mut result.pieces[index];
                    for pcurve in absorbed.pcurves {
                        if !keeper
                            .pcurves
                            .iter()
                            .any(|existing| {
                                existing.operand == pcurve.operand
                                    && existing.face_id == pcurve.face_id
                            })
                        {
                            keeper.pcurves.push(pcurve);
                        }
                    }
                    let keeper_id = keeper.id;
                    for record in &mut result.by_face {
                        if let Some(position) =
                            record.piece_ids.iter().position(|&id| id == absorbed.id)
                        {
                            if record.piece_ids.contains(&keeper_id) {
                                record.piece_ids.remove(position);
                            } else {
                                record.piece_ids[position] = keeper_id;
                            }
                        }
                    }
                } else {
                    other += 1;
                }
            }
            index += 1;
        }
        if !removed.is_empty() && std::env::var("BREP_DEBUG_BOOL").is_ok() {
            eprintln!("coincident-piece merge: absorbed {:?}", removed);
        }
    }
    // Rescue near-tangent SSI truncations BEFORE canonicalization so the added
    // bridge pieces' endpoints (existing crossing/stub vertices) fold into the
    // same junction merges as every other section.
    extend_truncated_sections(
        &mut result,
        &face_edge_lists,
        solid_a,
        solid_b,
        options.tolerance,
    )?;
    canonicalize_imprint_junctions(&mut result, solid_a, solid_b, options.tolerance)?;
    // B2: reuse an existing boundary edge as the shared section edge wherever a
    // section coincides with one along its whole span (vertices are final after
    // canonicalization; `face_edge_lists` holds each face's boundary edges on
    // the healed operands). Runs here so both operands reference ONE edge.
    reuse_boundary_section_edges(
        &mut result,
        &face_edge_lists,
        solid_a,
        solid_b,
        options.tolerance,
    )?;
    // Capstone step 1 — instrumentation only, zero behavior change: report
    // every (section piece × boundary edge) contact where the piece runs
    // within the scale-derived band of the edge over a real span. Measuring
    // the bands here first validates the graze-contact model on the
    // acceptance suite before step 2's common-block machinery replaces a
    // grazed overlap with a shared edge.
    report_graze_contacts(&result, &face_edge_lists, solid_a, solid_b, options.tolerance)?;
    Ok(result)
}

/// Debug-only graze-contact survey (`BREP_DEBUG_GRAZE=1`): for each section
/// piece and each boundary edge of its support faces, sample the piece and
/// measure distance to the edge; report contacts whose in-band span exceeds
/// both the weld scale and 4× the minimum deviation (span-wise proximity, not
/// a point touch). `band_cap` reuses the residual-merge `sep_cap` ceiling —
/// measured, never grown. The output is the raw material for capstone step 2
/// (partial-span common-block): which contacts exist, their spans, and their
/// measured bands.
fn report_graze_contacts(
    result: &ImprintResultRecord,
    face_edge_lists: &HashMap<FaceKey, Vec<&EdgeRecord>>,
    solid_a: &BrepSolid,
    solid_b: &BrepSolid,
    tolerance: f64,
) -> Result<(), KernelRefusal> {
    if std::env::var("BREP_DEBUG_GRAZE").as_deref() != Ok("1") {
        return Ok(());
    }
    let raw_extent = raw_solid_extent(solid_a).max(raw_solid_extent(solid_b));
    let (_, band_cap) = residual_merge_bands(raw_extent, tolerance);
    const SAMPLES: usize = 17;
    for piece in &result.pieces {
        let [t0, t1] = [piece.t0, piece.t1];
        if !(t1 > t0) {
            continue;
        }
        for key in piece.support_faces {
            let Some(edges) = face_edge_lists.get(&key) else {
                continue;
            };
            for edge in edges {
                if edge.degenerate {
                    continue;
                }
                let mut in_band = 0usize;
                let mut min_dev = f64::INFINITY;
                let mut max_dev_in_band = 0.0f64;
                let mut span = 0.0f64;
                let mut prev: Option<(bool, Vec3)> = None;
                for k in 0..SAMPLES {
                    let t = t0 + (t1 - t0) * k as f64 / (SAMPLES - 1) as f64;
                    let point = piece.curve.evaluate(t).or_refuse(KernelStage::Intersect, "evaluate")?;
                    let deviation = project_point_to_curve(&edge.curve, point).or_refuse(KernelStage::Intersect, "project_point_to_curve")?.distance;
                    min_dev = min_dev.min(deviation);
                    let inside = deviation <= band_cap;
                    if inside {
                        in_band += 1;
                        max_dev_in_band = max_dev_in_band.max(deviation);
                        if let Some((true, prev_point)) = prev {
                            span += point.sub(prev_point).length();
                        }
                    }
                    prev = Some((inside, point));
                }
                // Span-wise contact: several consecutive samples in band and a
                // span that dwarfs the closest-approach (not a transversal
                // crossing, which dips in and out at one sample).
                if in_band >= 3 && span > (4.0 * min_dev).max(assembler_weld(tolerance)) {
                    eprintln!(
                        "graze: piece {} sup=[{}:{},{}:{}] ~ edge {}:{} span={:.3e} band=[{:.3e},{:.3e}] samples_in_band={}/{}",
                        piece.id,
                        piece.support_faces[0].operand,
                        piece.support_faces[0].face_id,
                        piece.support_faces[1].operand,
                        piece.support_faces[1].face_id,
                        key.operand,
                        edge.id,
                        span,
                        min_dev,
                        max_dev_in_band,
                        in_band,
                        SAMPLES
                    );
                }
            }
        }
    }
    Ok(())
}

/// Are the two carriers TANGENT (normals parallel) where they both pass through
/// `point`?
///
/// The test the tangential-only refusal above needs: a point on the marched
/// intersection is a TANGENT NODE when the two surface normals there are
/// parallel. The threshold is the transversality bound the supplemental
/// detector already accepts seeds by (`TRANSVERSE_SEED_CROSS`), not the
/// far tighter pair-classifier bound — a node the march merely passes CLOSE to
/// still poisons the assembly, so this errs toward calling a pair singular.
fn pair_normals_parallel_at(
    first: TaggedFace<'_>,
    second: TaggedFace<'_>,
    point: Vec3,
) -> Result<bool, KernelRefusal> {
    let mut normals = [Vec3::default(); 2];
    for (slot, face) in normals.iter_mut().zip([first, second]) {
        let projection = project_point_to_surface(&face.face.surface, point)
            .or_refuse(KernelStage::Intersect, "project_point_to_surface")?;
        let Ok(normal) = face.face.surface.normal(projection.u, projection.v) else {
            // A pole/singular parameter point cannot witness transversality;
            // treat it as tangential so the pair stays refused.
            return Ok(true);
        };
        *slot = normal;
    }
    Ok(normals[0].cross(normals[1]).length() <= crate::TRANSVERSE_SEED_CROSS)
}
