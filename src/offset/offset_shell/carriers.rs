use super::*;

pub(super) fn standalone_face(solid: &BrepSolid, face_id: u64) -> Result<BrepSolid, String> {
    let face = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .find(|face| face.id == face_id)
        .cloned()
        .ok_or_else(|| format!("offset_shell: missing face {face_id}"))?;
    let edge_ids = face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .map(|coedge| coedge.edge_id)
        .collect::<HashSet<_>>();
    let edges = solid
        .edges
        .iter()
        .filter(|edge| edge_ids.contains(&edge.id))
        .cloned()
        .collect::<Vec<_>>();
    let vertex_ids = edges
        .iter()
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect::<HashSet<_>>();
    let vertices = solid
        .vertices
        .iter()
        .filter(|vertex| vertex_ids.contains(&vertex.id))
        .cloned()
        .collect();
    Ok(BrepSolid {
        id: solid.id,
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: 1,
            faces: vec![face],
        }],
        genus: 0,
    })
}

/// A cylinder or sphere has no regular inner skin once its signed radius
/// reaches zero. Do this on the source support, before fitting an offset:
/// fitting a zero-radius surface leaves line edges, and fitting past zero
/// creates a reflected surface that is no longer part of the shell boundary.
/// Keep the source face itself for the outer skin and distance classification.
pub(super) fn offset_support_collapsed(face: &FaceRecord, distance: f64) -> Result<bool, String> {
    use crate::AnalyticSurface;
    let (center, axis, radius) = match face.surface.analytic() {
        Some(AnalyticSurface::Sphere { frame, radius }) => (frame.origin, None, *radius),
        Some(AnalyticSurface::RuledRevolution {
            frame, rho0, rho1, ..
        }) if (rho0 - rho1).abs() <= 1e-9 * rho0.abs().max(1.0) => {
            (frame.origin, Some(frame.axis), (rho0 + rho1) * 0.5)
        }
        Some(AnalyticSurface::Revolution {
            frame, generatrix, ..
        }) if generatrix.degree == 1 && generatrix.control_points.len() == 2 => {
            let a = generatrix.control_points[0].point()?.sub(frame.origin);
            let b = generatrix.control_points[1].point()?.sub(frame.origin);
            let ra = a.sub(frame.axis.scale(a.dot(frame.axis)));
            let rb = b.sub(frame.axis.scale(b.dot(frame.axis)));
            if ra.sub(rb).length() > 1e-9 * ra.length().max(1.0) {
                return Ok(false);
            }
            (frame.origin, Some(frame.axis), ra.length())
        }
        _ => return offset_curvature_collapsed(face, distance),
    };
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let (u, v) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let mut radial = face.surface.evaluate(u, v)?.sub(center);
    if let Some(axis) = axis {
        radial = radial.sub(axis.scale(radial.dot(axis)));
    }
    let normal = face.surface.normal(u, v)?;
    let sense = if face.same_sense { 1.0 } else { -1.0 };
    let inward_change = distance * sense * normal.dot(radial.normalized()?);
    Ok(radius - inward_change <= 1e-9 * radius.max(distance.abs()).max(1.0))
}

/// The lane [`offset_support_collapsed`] takes for a support with no radius to
/// read: a FITTED face — a fillet blend along a curved spine, a swept
/// band, an imported patch — is never `analytic()`, so the radius lanes above
/// see nothing and the inner skin of a fillet thinner than the wall is fitted
/// and kept as a REFLECTED band. Read the support's principal curvatures
/// instead, which is the same regularity condition `thicken`'s
/// `ensure_offsets_regular` states for a sheet: the equidistant surface folds
/// through the evolute wherever the per-direction area factor reaches zero.
///
/// Sign. [`crate::NurbsSurface::principal_curvatures`] signs κ against the
/// PARAMETRIZATION normal `Su × Sv`; the face's outward normal is
/// `same_sense · n`; and offset-shell's positive `distance` moves OPPOSITE that
/// outward normal. The factor is therefore `1 + distance · κ_outward` with
/// `κ_outward = same_sense · κ_n`. Shelled by 1, a bore of radius 6 reads 7/6,
/// the r=18 convex blend it runs into reads 17/18, and an r=0.5 convex fillet
/// between them reads −1 — that fillet's inner skin is a band bounded by the
/// same two rails traced the wrong way round, which no weld can make part of
/// the shell. Dropping it lets the two neighbours meet in the sharp inner miter
/// that a d > r fillet is supposed to leave.
///
/// WHOLE-FACE only. A constant-radius fillet collapses at every sample, which
/// is the case this answers. A support that collapses only in PART — a cone
/// reaching its apex inside the trim — is a different question (the regular
/// part still belongs to the skin), so keep it and say so.
fn offset_curvature_collapsed(face: &FaceRecord, distance: f64) -> Result<bool, String> {
    // The factor a fold has to reach. `thicken` refuses a sheet at the same
    // 1e-6: below it the offset's area element has collapsed, and the fitted
    // supports this lane reads carry curvature noise far under that.
    const COLLAPSE_FACTOR: f64 = 1e-6;
    const SAMPLES: usize = 7;
    if distance == 0.0 || face.surface.is_affine()? {
        return Ok(false);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let sense = if face.same_sense { 1.0 } else { -1.0 };
    let mut sampled = 0usize;
    let mut collapsed = 0usize;
    let mut worst = f64::INFINITY;
    // Strictly interior samples: a pole or a seam sits ON the domain boundary,
    // where the parametrization is degenerate and κ is unreadable.
    for iu in 0..SAMPLES {
        let u = u0 + (u1 - u0) * (iu as f64 + 0.5) / SAMPLES as f64;
        for iv in 0..SAMPLES {
            let v = v0 + (v1 - v0) * (iv as f64 + 0.5) / SAMPLES as f64;
            let Ok((kappa_min, kappa_max)) = face.surface.principal_curvatures(u, v) else {
                continue;
            };
            sampled += 1;
            let factor = [kappa_min, kappa_max]
                .into_iter()
                .map(|kappa| 1.0 + distance * sense * kappa)
                .fold(f64::INFINITY, f64::min);
            worst = worst.min(factor);
            if factor <= COLLAPSE_FACTOR {
                collapsed += 1;
            }
        }
    }
    os_debug!(
        "curvature collapse probe face {}: worst factor {:.6} over {}/{} collapsed samples",
        face.id,
        worst,
        collapsed,
        sampled
    );
    Ok(sampled > 0 && collapsed == sampled)
}

pub(super) fn carrier_solid(
    source: &BrepSolid,
    face_id: u64,
    distance: f64,
    planar_extension: f64,
) -> Result<BrepSolid, String> {
    carrier_solid_sided(
        source,
        face_id,
        distance,
        &crate::CarrierExtension::uniform(planar_extension),
    )
}

pub(super) fn carrier_solid_sided(
    source: &BrepSolid,
    face_id: u64,
    distance: f64,
    extension: &crate::CarrierExtension,
) -> Result<BrepSolid, String> {
    let carrier = crate::offset_face_carrier_sided(source, face_id, distance, extension)?;
    Ok(BrepSolid {
        id: 1,
        vertices: carrier.vertices,
        edges: carrier.edges,
        shells: vec![ShellRecord {
            id: 1,
            faces: vec![carrier.face],
        }],
        genus: 0,
    })
}

/// Keep only the outer (largest |UV area|) loop of a one-face wall carrier and
/// prune the edges/vertices the dropped interior loops referenced. See the
/// call site in `offset_shell_impl` for why interior loops must go.
pub(super) fn drop_wall_interior_loops(wall: &mut BrepSolid) -> Result<(), String> {
    let face = &mut wall.shells[0].faces[0];
    if face.loops.len() < 2 {
        return Ok(());
    }
    let mut outer = 0usize;
    let mut outer_area = f64::NEG_INFINITY;
    for (index, loop_record) in face.loops.iter().enumerate() {
        let area = parameter_space_area(&FaceRecord {
            id: 0,
            surface: face.surface.clone(),
            same_sense: true,
            loops: vec![loop_record.clone()],
            name: None,
        })?
        .abs();
        if area > outer_area {
            outer_area = area;
            outer = index;
        }
    }
    let kept = face.loops[outer].clone();
    face.loops = vec![kept];
    let used_edges = face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .map(|coedge| coedge.edge_id)
        .collect::<HashSet<_>>();
    wall.edges.retain(|edge| used_edges.contains(&edge.id));
    let used_vertices = wall
        .edges
        .iter()
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect::<HashSet<_>>();
    wall.vertices
        .retain(|vertex| used_vertices.contains(&vertex.id));
    Ok(())
}

fn merge_vertex(
    vertices: &mut Vec<ImprintVertex>,
    point: Vec3,
    next_id: &mut u64,
    tolerance: f64,
    scale: f64,
) -> u64 {
    // `crate::tolerance::merge_scale` IS this band's size factor, and its doc
    // records why: the former `1 + ‖point‖` coupling gave a part far from the
    // origin a wrongly-inflated merge band, so the same geometry translated
    // away produced different volumes.  The shared imprint was moved off that
    // anti-pattern; offset-shell's private copy of `merge_vertex` was not, and
    // still read the distance from the world origin until audit slice 0.
    let merge = tolerance.max(1e-5) * crate::tolerance::merge_scale(scale);
    if let Some(vertex) = vertices
        .iter()
        .find(|vertex| vertex.point.sub(point).length() <= merge)
    {
        return vertex.id;
    }
    let id = *next_id;
    *next_id += 1;
    vertices.push(ImprintVertex { id, point });
    id
}

pub(super) fn merge_pair_imprint(
    global: &mut ImprintResultRecord,
    pair: ImprintResultRecord,
    first_operand: u8,
    second_operand: u8,
    next_piece_id: &mut u64,
    next_vertex_id: &mut u64,
    tolerance: f64,
    scale: f64,
) {
    let remap_operand = |operand| {
        if operand == 0 {
            first_operand
        } else {
            second_operand
        }
    };
    let mut vertex_map = HashMap::default();
    for vertex in pair.vertices {
        vertex_map.insert(
            vertex.id,
            merge_vertex(
                &mut global.vertices,
                vertex.point,
                next_vertex_id,
                tolerance,
                scale,
            ),
        );
    }
    let mut piece_map = HashMap::default();
    for piece in pair.pieces {
        let id = *next_piece_id;
        *next_piece_id += 1;
        piece_map.insert(piece.id, id);
        global.pieces.push(ImprintPieceRecord {
            id,
            curve: piece.curve,
            t0: piece.t0,
            t1: piece.t1,
            start_vertex_id: vertex_map[&piece.start_vertex_id],
            end_vertex_id: vertex_map[&piece.end_vertex_id],
            pcurves: piece
                .pcurves
                .into_iter()
                .map(|pcurve| FacePcurve {
                    operand: remap_operand(pcurve.operand),
                    face_id: pcurve.face_id,
                    pcurve: pcurve.pcurve,
                })
                .collect(),
            support_faces: piece.support_faces.map(|face| FaceKey {
                operand: remap_operand(face.operand),
                face_id: face.face_id,
            }),
            shared_edge: piece
                .shared_edge
                .map(|(operand, edge_id, aligned)| (remap_operand(operand), edge_id, aligned)),
        });
    }
    for by_face in pair.by_face {
        let operand = remap_operand(by_face.operand);
        let entry = if let Some(entry) = global
            .by_face
            .iter_mut()
            .find(|entry| entry.operand == operand && entry.face_id == by_face.face_id)
        {
            entry
        } else {
            global.by_face.push(FaceImprints {
                operand,
                face_id: by_face.face_id,
                piece_ids: Vec::new(),
            });
            global.by_face.last_mut().unwrap()
        };
        entry.piece_ids.extend(
            by_face
                .piece_ids
                .into_iter()
                .filter_map(|id| piece_map.get(&id).copied()),
        );
    }
    for split in pair.edge_splits {
        let operand = remap_operand(split.operand);
        if let Some(existing) = global
            .edge_splits
            .iter_mut()
            .find(|entry| entry.operand == operand && entry.edge_id == split.edge_id)
        {
            for parameter in split.parameters {
                if !existing
                    .parameters
                    .iter()
                    .any(|value| (*value - parameter).abs() <= 1e-8)
                {
                    existing.parameters.push(parameter);
                }
            }
        } else {
            global.edge_splits.push(EdgeSplitRecord {
                operand,
                edge_id: split.edge_id,
                parameters: split.parameters,
            });
        }
    }
}

pub(super) fn empty_imprint() -> ImprintResultRecord {
    ImprintResultRecord {
        vertices: Vec::new(),
        pieces: Vec::new(),
        by_face: Vec::new(),
        edge_splits: Vec::new(),
        barrier_edges: Vec::new(),
        tangent_nodes: Vec::new(),
        section_evidence: false,
    }
}

pub(super) fn flip_fragment(fragment: &mut FaceFragmentRecord) -> Result<(), String> {
    fragment.same_sense = !fragment.same_sense;
    for loop_record in &mut fragment.loops {
        loop_record.coedges.reverse();
        for coedge in &mut loop_record.coedges {
            coedge.forward = !coedge.forward;
            coedge.pcurve = coedge.pcurve.reversed()?;
        }
    }
    Ok(())
}

pub(super) fn fragment_as_trim(fragment: &FaceFragmentRecord) -> FaceRecord {
    FaceRecord {
        id: fragment.source_face_id,
        surface: fragment.surface.clone(),
        same_sense: fragment.same_sense,
        loops: fragment
            .loops
            .iter()
            .enumerate()
            .map(|(loop_index, loop_record)| LoopRecord {
                id: loop_index as u64 + 1,
                coedges: loop_record
                    .coedges
                    .iter()
                    .enumerate()
                    .map(|(index, coedge)| CoedgeRecord {
                        id: index as u64 + 1,
                        edge_id: index as u64 + 1,
                        forward: coedge.forward,
                        pcurve: coedge.pcurve.clone(),
                    })
                    .collect(),
            })
            .collect(),
        name: None,
    }
}

/// Distance from a point to a TRIMMED face: surface projection when it lands
/// inside the trim, otherwise the nearest point on the face's boundary edges.
fn distance_to_trimmed_face(
    face: &FaceRecord,
    edges: &[&EdgeRecord],
    point: Vec3,
) -> Result<f64, String> {
    let projection = project_point_to_surface(&face.surface, point)?;
    let uv = Vec2 {
        x: projection.u,
        y: projection.v,
    };
    if parameter_point_in_face(face, uv, 1e-9)? != PolygonClass::Outside {
        return Ok(projection.distance);
    }
    let mut best = f64::INFINITY;
    for edge in edges {
        if edge.degenerate {
            continue;
        }
        let samples = 48;
        for index in 0..=samples {
            let parameter = edge.t0 + (edge.t1 - edge.t0) * index as f64 / samples as f64;
            best = best.min(edge.curve.evaluate(parameter)?.sub(point).length());
        }
    }
    Ok(best)
}

/// True when the point lies at the full offset distance from every retained
/// source face — the defining property of the shell's offset skin. Fragments
/// that sit closer to some OTHER retained face belong to that face's offset
/// region (they are "shadowed") and must not be selected, no matter where the
/// legacy parameter-space seed lands.
pub(super) fn point_on_offset_skin(
    point: Vec3,
    retained_faces: &[(&FaceRecord, Vec<&EdgeRecord>)],
    distance: f64,
    tolerance: f64,
) -> Result<bool, String> {
    let target = distance.abs();
    for (face, edges) in retained_faces {
        let separation = distance_to_trimmed_face(face, edges, point)?;
        if separation < target - tolerance {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn source_seed(source: &BrepSolid, face_id: u64) -> Result<Vec2, String> {
    let face = standalone_face(source, face_id)?;
    let fragments = fragment_solid(&face, SOURCE_OPERAND, &empty_imprint())?;
    fragments
        .first()
        .map(|fragment| fragment.test_uv)
        .ok_or_else(|| "offset_shell: source face has no interior seed".into())
}

/// The seed an OFFSET carrier's fragment selection should prefer: the
/// source face's interior seed point moved `distance` along the face's
/// (inward-for-positive) offset direction, then located on the carrier
/// surface by projection.
///
/// Reusing the source seed's raw uv assumes the carrier is parameterised
/// like the source face. A padded plane is not — the pad slides the plane's
/// net under the cloned pcurves, so the source uv lands `extension` short of
/// (or past) its true offset image; a full-domain sphere rebuild replaces the
/// fitted net outright. Both put the raw seed in the wrong fragment (the
/// frustum-base annulus above: the seed read OUTSIDE its own annulus, and
/// the disk over the dropped hole survived only by a distance coin-flip).
/// `None` when the moved point does not lie on the carrier (a shadowed or
/// collapsed offset — keep the raw seed and let the fallbacks decide).
pub(super) fn offset_seed(
    source_face: &FaceRecord,
    raw_seed: Vec2,
    carrier_surface: &crate::NurbsSurface,
    distance: f64,
    band: f64,
) -> Result<Option<Vec2>, String> {
    // `face_offsets` is signed ALONG the outward normal; offset-shell's
    // positive distance moves opposite it.
    let sample = face_offsets(source_face).at(raw_seed.x, raw_seed.y, -distance)?;
    let projection = project_point_to_surface(carrier_surface, sample.point)?;
    if !projection.distance.is_finite() || projection.distance > band {
        return Ok(None);
    }
    Ok(Some(Vec2 {
        x: projection.u,
        y: projection.v,
    }))
}

#[derive(Clone)]
pub(super) struct FragmentEdgeGeometry {
    pub(super) curve: crate::NurbsCurve,
    pub(super) t0: f64,
    pub(super) t1: f64,
}

pub(super) fn fragment_edge_segment_key(
    source: &FragmentEdgeSource,
    solids: &HashMap<u8, &BrepSolid>,
    imprint: &ImprintResultRecord,
) -> Result<[i64; 9], String> {
    let geometry = fragment_edge_geometry(source, solids, imprint)?;
    let mut start = geometry.curve.evaluate(geometry.t0)?;
    let middle = geometry.curve.evaluate((geometry.t0 + geometry.t1) * 0.5)?;
    let mut end = geometry.curve.evaluate(geometry.t1)?;
    if (end.x, end.y, end.z) < (start.x, start.y, start.z) {
        std::mem::swap(&mut start, &mut end);
    }
    let quantize = |value: f64| (value / 1e-6).round() as i64;
    Ok([
        quantize(start.x),
        quantize(start.y),
        quantize(start.z),
        quantize(middle.x),
        quantize(middle.y),
        quantize(middle.z),
        quantize(end.x),
        quantize(end.y),
        quantize(end.z),
    ])
}

fn fragment_edge_geometry(
    source: &FragmentEdgeSource,
    solids: &HashMap<u8, &BrepSolid>,
    imprint: &ImprintResultRecord,
) -> Result<FragmentEdgeGeometry, String> {
    match source {
        FragmentEdgeSource::Boundary { operand, edge_id }
        | FragmentEdgeSource::SharedBoundary { operand, edge_id } => {
            let edge = solids
                .get(operand)
                .and_then(|solid| solid.edges.iter().find(|edge| edge.id == *edge_id))
                .ok_or_else(|| {
                    format!("offset_shell: missing boundary edge {operand}:{edge_id}")
                })?;
            Ok(FragmentEdgeGeometry {
                curve: edge.curve.clone(),
                t0: edge.t0,
                t1: edge.t1,
            })
        }
        FragmentEdgeSource::Imprint { piece_id } => {
            let piece = imprint
                .pieces
                .iter()
                .find(|piece| piece.id == *piece_id)
                .ok_or_else(|| format!("offset_shell: missing imprint piece {piece_id}"))?;
            Ok(FragmentEdgeGeometry {
                curve: piece.curve.clone(),
                t0: piece.t0,
                t1: piece.t1,
            })
        }
        FragmentEdgeSource::Derived { curve, t0, t1, .. } => Ok(FragmentEdgeGeometry {
            curve: curve.clone(),
            t0: *t0,
            t1: *t1,
        }),
    }
}

pub(super) fn fragment_edge_lies_on(
    source: &FragmentEdgeGeometry,
    target: &FragmentEdgeGeometry,
    tolerance: f64,
) -> Result<bool, String> {
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let point = source
            .curve
            .evaluate(source.t0 + (source.t1 - source.t0) * fraction)?;
        let projection = crate::project_point_to_curve(&target.curve, point)?;
        if projection.distance > tolerance
            || projection.u < target.t0 - tolerance
            || projection.u > target.t1 + tolerance
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn fragment_edges_overlap(
    first: &FragmentEdgeGeometry,
    second: &FragmentEdgeGeometry,
    tolerance: f64,
) -> Result<bool, String> {
    Ok(fragment_edge_lies_on(first, second, tolerance)?
        || fragment_edge_lies_on(second, first, tolerance)?)
}

fn fragment_edge_geometries(
    fragment: &FaceFragmentRecord,
    solids: &HashMap<u8, &BrepSolid>,
    imprint: &ImprintResultRecord,
) -> Result<Vec<FragmentEdgeGeometry>, String> {
    fragment
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .map(|coedge| fragment_edge_geometry(&coedge.source, solids, imprint))
        .collect()
}

pub(super) fn would_overuse_existing_boundary(
    candidate: &FaceFragmentRecord,
    existing: impl Iterator<Item = FaceFragmentRecord>,
    solids: &HashMap<u8, &BrepSolid>,
    imprint: &ImprintResultRecord,
    tolerance: f64,
) -> Result<bool, String> {
    let existing_edges = existing
        .map(|fragment| fragment_edge_geometries(&fragment, solids, imprint))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    for candidate_edge in fragment_edge_geometries(candidate, solids, imprint)? {
        let mut matches = 0;
        for existing_edge in &existing_edges {
            if fragment_edges_overlap(&candidate_edge, existing_edge, tolerance)? {
                matches += 1;
                if matches >= 2 {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

pub(super) fn fragment_has_sustained_source_contact(
    fragment: &FaceFragmentRecord,
    source_faces: &[&FaceRecord],
    solids: &HashMap<u8, &BrepSolid>,
    imprint: &ImprintResultRecord,
    tolerance: f64,
) -> Result<bool, String> {
    for edge in fragment_edge_geometries(fragment, solids, imprint)? {
        for source_face in source_faces {
            let mut sustained = true;
            for fraction in [0.2, 0.5, 0.8] {
                let point = edge
                    .curve
                    .evaluate(edge.t0 + (edge.t1 - edge.t0) * fraction)?;
                let projection = project_point_to_surface(&source_face.surface, point)?;
                if projection.distance > tolerance
                    || parameter_point_in_face(
                        source_face,
                        Vec2 {
                            x: projection.u,
                            y: projection.v,
                        },
                        tolerance,
                    )? == PolygonClass::Outside
                {
                    sustained = false;
                    break;
                }
            }
            if sustained {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
