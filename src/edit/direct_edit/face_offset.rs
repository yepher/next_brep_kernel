use super::*;

// ---------------------------------------------------------------------------
// Push a CURVED analytic face (cylinder / cone) by OFFSETTING its carrier.
//
// The methodology (user directive 2026-08-28, the architecture principle made
// literal): a face is a TRIMMED region of an infinite carrier surface. To push
// a curved face we OFFSET its untrimmed carrier surface, then re-derive the trim
// as the INTERSECTIONS of that offset surface with the (unchanged) untrimmed
// carriers of the neighbour faces — `intersect_analytic_pair` does the surface ∩
// surface, `build_pcurve_on_surface` re-trims. This is the same engine the
// boolean imprint uses; here one surface (the pushed face's) is replaced by its
// offset and the incident edges are recut.
//
// Slice 1 scope (honest refusals, never a bad solid): the PUSHED face is a
// cylinder or cone (`RuledRevolution`); its neighbours across each boundary edge
// are PLANAR. Ruled/sphere/torus neighbours and free-form pushed faces are
// deferred.
// ---------------------------------------------------------------------------

/// The exact analytic offset of a ruled-revolution CARRIER surface (cylinder or
/// cone) by `signed_distance` along its OUTWARD normal — the untrimmed surface
/// S′ whose re-intersection with the neighbour carriers gives the new trim.
///
/// A normal offset of a ruled revolution is another ruled revolution with the
/// SAME axis and half-angle: in the (axial z, radial ρ) meridian, the generatrix
/// line `ρ = rho0 + m·z` (m = dρ/dz = (rho1−rho0)/height; m = 0 for a cylinder)
/// offsets to the PARALLEL line `ρ = rho0 + m·z + δ·√(1+m²)` — a uniform radial
/// growth `grow = δ·√(1+m²)` at every z. So S′ is built by revolving the grown
/// generatrix about the SAME frame, over the SAME axial span `[0, height]` and
/// seam azimuth (`frame.x_axis`) as the source — the span keeps the neighbour
/// caps IN the surface's domain (an axis-translation offset would slide the
/// finite patch off them) and the shared seam meridian stays put.
///
/// `signed_distance > 0` grows the surface outward (radius increases). Refuses a
/// carrier whose grown radius reaches/crosses the axis at either end.
fn offset_ruled_carrier(
    surface: &NurbsSurface,
    signed_distance: f64,
    tolerance: f64,
) -> Result<NurbsSurface, String> {
    let Some(AnalyticSurface::RuledRevolution {
        frame,
        rho0,
        rho1,
        height,
    }) = surface.analytic()
    else {
        return Err("offset_ruled_carrier: face carrier is not a ruled revolution".into());
    };
    let (frame, rho0, rho1, height) = (frame.clone(), *rho0, *rho1, *height);
    let slope = (rho1 - rho0) / height;
    let grow = signed_distance * (1.0 + slope * slope).sqrt();
    let rho0_new = rho0 + grow;
    let rho1_new = rho1 + grow;
    if rho0_new <= tolerance || rho1_new <= tolerance {
        return Err(
            "offset (push): the ruled carrier collapses to or past its axis — refusing".into(),
        );
    }
    // Revolve the grown generatrix over the SAME axial span + seam azimuth.
    let start = frame.origin.add(frame.x_axis.scale(rho0_new));
    let end = frame
        .origin
        .add(frame.axis.scale(height))
        .add(frame.x_axis.scale(rho1_new));
    let generatrix = make_line(start, end)?;
    make_revolution(frame.origin, frame.axis, &generatrix, std::f64::consts::TAU)
}

/// The face's OUTWARD unit normal at its parameter-domain midpoint, oriented by
/// `same_sense` (the convention `push_face::planar_face_normal` uses).
fn outward_normal_mid(face: &FaceRecord) -> Result<(Vec3, Vec3), String> {
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let (um, vm) = (0.5 * (u0 + u1), 0.5 * (v0 + v1));
    let point = face.surface.evaluate(um, vm)?;
    let mut normal = face.surface.normal(um, vm)?;
    if !face.same_sense {
        normal = normal.scale(-1.0);
    }
    Ok((point, normal))
}

/// Push a CYLINDER or CONE face along its outward normal by `distance` (positive
/// grows the solid) by OFFSETTING its carrier surface and re-deriving the trim as
/// the intersection of that offset surface with the neighbour carriers.
///
/// Slice-1 scope (honest refusal, never a bad solid): a full-revolution
/// cylinder/cone SIDE face whose neighbour across every non-seam boundary edge is
/// PLANAR (its end caps). The pushed face's own periodic SEAM edge rides the
/// offset carrier's meridian; each rim edge re-intersects its cap
/// (`intersect_analytic_pair`, exact circle); the caps re-trim as planar faces.
/// Ruled/curved neighbours and non-full-revolution ruled faces are deferred.
pub fn offset_ruled_face(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
) -> Result<BrepSolid, String> {
    if !distance.is_finite() {
        return Err("offset_ruled_face: distance must be finite".into());
    }
    let scale = solid_model_scale(solid);
    let tolerance = (scale * 1e-7).max(1e-9);
    let plane_tolerance = (scale * 1e-6).max(1e-7);

    let (pshell, pface) =
        find_face(solid, face_id).ok_or_else(|| format!("offset_ruled_face: no face {face_id}"))?;
    let pushed = &solid.shells[pshell].faces[pface];
    let Some(AnalyticSurface::RuledRevolution { frame, .. }) = pushed.surface.analytic() else {
        return Err("offset_ruled_face: the pushed face is not a cylinder or cone".into());
    };
    let (frame_origin, frame_axis) = (frame.origin, frame.axis);

    // Sign the offset: the carrier grows outward (radius +) along the face's
    // OUTWARD normal. If that normal points radially inward (a hole wall), a
    // positive push shrinks the radius.
    let (mid_point, mid_normal) = outward_normal_mid(pushed)?;
    let radial = {
        let d = mid_point.sub(frame_origin);
        d.sub(frame_axis.scale(d.dot(frame_axis)))
    };
    if radial.length() <= tolerance {
        return Err("offset_ruled_face: degenerate radial direction (face on the axis)".into());
    }
    let outward_sign = if mid_normal.dot(radial) >= 0.0 { 1.0 } else { -1.0 };
    let signed_distance = distance * outward_sign;

    let s_prime = offset_ruled_carrier(&pushed.surface, signed_distance, tolerance)?;

    // Edge -> incident faces, to classify each of the pushed face's boundary
    // edges as a SEAM (both incidences are the pushed face) or a RIM (the other
    // face is the fixed neighbour cap).
    let mut faces_of_edge: HashMap<u64, Vec<u64>> = HashMap::default();
    for shell in &solid.shells {
        for face in &shell.faces {
            for loop_record in &face.loops {
                for coedge in &loop_record.coedges {
                    faces_of_edge
                        .entry(coedge.edge_id)
                        .or_default()
                        .push(face.id);
                }
            }
        }
    }
    let edge_by_id: HashMap<u64, &EdgeRecord> =
        solid.edges.iter().map(|edge| (edge.id, edge)).collect();

    let mut new_curve: HashMap<u64, NurbsCurve> = HashMap::default();
    let mut new_vertex: HashMap<u64, Vec3> = HashMap::default();
    let mut cap_faces: HashSet<u64> = HashSet::default();
    // Coaxial ruled neighbours (a stepped/telescoping bore or boss: a cylinder
    // meeting a coaxial cone, or two coaxial cones) — retrimmed on their own
    // (axially-grown) carriers rather than as planar caps.
    let mut ruled_faces: HashSet<u64> = HashSet::default();
    // CURVED neighbours re-intersected through the shared generic lane — a
    // sphere dome, a fillet torus, a general revolution, a non-coaxial
    // cylinder. Their carriers do NOT move and need no growth (unlike a plane's
    // finite patch or a ruled band's axial span, a sphere's and a torus's
    // domains are already closed over their whole surface, and a re-intersected
    // rim by construction lands on the carrier it was intersected with), so
    // they are re-trimmed in place: pcurves rebuilt around the moved rim, and
    // every OTHER edge they own re-trimmed by parameter on the curve it already
    // carries — see `retrim_curved_neighbour_ranges`.
    let mut curved_faces: HashSet<u64> = HashSet::default();
    let mut seam_edges: Vec<u64> = Vec::new();
    // Current vertex positions, for rebuilding a ruled neighbour's seam whose
    // far endpoint (its unmoved rim vertex) is not relocated by any rim.
    let vertex_pos: HashMap<u64, Vec3> =
        solid.vertices.iter().map(|v| (v.id, v.point)).collect();

    // OPEN rim arcs (a multi-loop pushed face: a window / slot cut through the
    // wall). Collected here and trimmed in a second pass, once every corner
    // vertex has been solved, so both arcs meeting at a corner agree on it.
    let mut open_rims: Vec<OpenRim> = Vec::new();
    // Every RIM edge rebuilt below, closed circles and open arcs alike: the
    // pushed face's pcurves for these are refitted when a multi-loop rebuild
    // ran, because the replacement conic carries its own parameterization.
    let mut rim_edges: Vec<u64> = Vec::new();

    // PRE-PASS — a HOLE cut clean through the wall by ONE crossing carrier.
    //
    // A window bored by a crossing cylinder, a dome or a torus leaves an
    // interior loop every one of whose edges is shared with the SAME neighbour
    // face, and whose corner vertices have valence two: no third face meets
    // there, so no triple point determines them and `resolve_open_rim_end` —
    // which solves a triple point against a PLANE — has nothing to solve. The
    // loop is one closed section that the arrangement stored as arcs, and the
    // right rebuild is to re-intersect once and cut the section at the samples
    // nearest the vertices it replaces. Handled here, ahead of the per-edge
    // loop, because the decision is a property of the whole loop.
    let mut hole_edges: HashSet<u64> = HashSet::default();
    for loop_record in &pushed.loops {
        let Some(hole) = single_neighbour_hole(
            loop_record,
            face_id,
            solid,
            &faces_of_edge,
            &edge_by_id,
            frame_origin,
            frame_axis,
            scale,
        )?
        else {
            continue;
        };
        rebuild_single_neighbour_hole(
            solid,
            &s_prime,
            &hole,
            &edge_by_id,
            &vertex_pos,
            tolerance,
            scale,
            &mut new_curve,
            &mut new_vertex,
        )?;
        for edge_id in &hole.edge_ids {
            hole_edges.insert(*edge_id);
            if !rim_edges.contains(edge_id) {
                rim_edges.push(*edge_id);
            }
        }
        curved_faces.insert(hole.neighbour);
    }

    for loop_record in &pushed.loops {
        let coedge_count = loop_record.coedges.len();
        for coedge_index in 0..coedge_count {
            let coedge = &loop_record.coedges[coedge_index];
            let edge = *edge_by_id
                .get(&coedge.edge_id)
                .ok_or_else(|| format!("offset_ruled_face: missing edge {}", coedge.edge_id))?;
            if hole_edges.contains(&edge.id) {
                continue; // rebuilt whole, by the pre-pass above.
            }
            let incident = faces_of_edge.get(&edge.id).cloned().unwrap_or_default();
            let is_seam = incident.iter().all(|f| *f == face_id);
            if is_seam {
                if !seam_edges.contains(&edge.id) {
                    seam_edges.push(edge.id);
                }
                continue;
            }
            // RIM: the fixed neighbour is the other face. Three lanes, in
            // decreasing exactness and increasing generality:
            //
            //   * a PLANE (an end cap) or a ruled revolution COAXIAL with the
            //     pushed carrier (a stepped/telescoping bore or boss) — the two
            //     original lanes, whose closed forms are reached by the exact
            //     call this function has always made, unchanged;
            //   * ANY OTHER carrier — sphere, torus, general revolution,
            //     non-coaxial cylinder/cone, free-form — through the shared
            //     re-intersection service (`offset/reintersect.rs`), which tries
            //     the same closed forms first and marches the pair when they
            //     decline. This is the lane the audit's §2.3 says push-face
            //     lacks and offset-shell has.
            //
            // The generic lane is admitted only where THIS rebuild can express
            // the answer: a CLOSED rim, whose single vertex is a bookkeeping
            // seam point rather than a triple point. An open arc against a
            // curved neighbour needs `resolve_open_rim_end` to solve a triple
            // point against a curved `N'`, which it cannot (it takes a plane),
            // so that stays a refusal with its own reason.
            let neighbour = *incident
                .iter()
                .find(|f| **f != face_id)
                .ok_or_else(|| format!("offset_ruled_face: edge {} has no neighbour", edge.id))?;
            let (nshell, nface) = find_face(solid, neighbour)
                .ok_or_else(|| format!("offset_ruled_face: missing neighbour {neighbour}"))?;
            let neighbour_surface = &solid.shells[nshell].faces[nface].surface;
            let is_plane =
                matches!(neighbour_surface.analytic(), Some(AnalyticSurface::Plane { .. }));
            let is_coaxial_ruled =
                ruled_neighbour_is_coaxial(neighbour_surface, frame_origin, frame_axis, scale);
            let is_curved = !is_plane && !is_coaxial_ruled;
            let separated = || {
                "offset_ruled_face: the pushed carrier no longer meets a neighbour \
                 (the push separated them, or the rim left the neighbour's domain) — refusing"
                    .to_string()
            };
            // New rim = offset carrier ∩ neighbour carrier, the branch nearest
            // the old edge.
            let old_mid = edge.curve.evaluate(0.5 * (edge.t0 + edge.t1))?;
            let mut marched = false;
            let curves = if is_curved {
                if edge.start_vertex_id != edge.end_vertex_id {
                    return Err(format!(
                        "offset_ruled_face: the rim against curved neighbour {neighbour} is an \
                         OPEN arc (edge {}); an arc's corner is a triple point solved against a \
                         PLANE only — deferred (refusing)",
                        edge.id
                    ));
                }
                let policy = MarchPolicy {
                    tolerance,
                    // The rebuilt rim must sit on BOTH carriers tightly enough
                    // that its pcurves build and `validate` accepts them; the
                    // free-form push's own residual gate (0.05% of model scale)
                    // is the in-tree precedent for "how far an approximate
                    // offset result may be off".
                    residual_tolerance: (scale * 5e-4).max(5e-6),
                    // The boundary being replaced is the best seed set for the
                    // boundary replacing it.
                    seeds: edge_seeds(edge, 9)?,
                };
                match reintersect_carriers(&s_prime, neighbour_surface, &policy) {
                    Ok(found) => {
                        marched = found.lane == RimLane::Marched;
                        if std::env::var("BREP_PUSH_HOLE_DEBUG").is_ok() {
                            eprintln!(
                                "RIM neighbour {neighbour}: lane {:?}, {} branch(es), \
                                 residual {:.3e} (gate {:.3e})",
                                found.lane,
                                found.sections.len(),
                                found.residual,
                                policy.residual_tolerance
                            );
                        }
                        found.curves()
                    }
                    Err(ReintersectRefusal::Separated) => return Err(separated()),
                    Err(other) => {
                        return Err(format!("offset_ruled_face: {}", other.describe()));
                    }
                }
            } else {
                // UNCHANGED: the exact call, with the same arguments in the same
                // order, so every pair this function answered before is answered
                // bit-identically now.
                intersect_analytic_pair(&s_prime, neighbour_surface, tolerance)
                    .filter(|curves| !curves.is_empty())
                    .ok_or_else(separated)?
            };
            let rim = nearest_curve(&curves, old_mid)?;
            if !rim_edges.contains(&edge.id) {
                rim_edges.push(edge.id);
            }
            if edge.start_vertex_id == edge.end_vertex_id {
                // A closed rim is a whole conic, so it must also run the way the
                // edge it replaces ran — see `match_closed_rim_direction`.
                let rim = if marched {
                    // A MARCHED section begins wherever the trace was seeded,
                    // not on the carrier's seam, so the start-tangent test would
                    // compare two unrelated places. Ask the same question at the
                    // old curve's nearest parameter instead.
                    match_marched_rim_direction(rim, edge)?
                } else {
                    match_closed_rim_direction(rim, edge)?
                };
                // CLOSED rim (the canonical cap circle of a single-loop push):
                // its seam-azimuth point is the relocated corner and matches the
                // offset carrier's meridian. A recognized carrier's u = 0 IS its
                // seam, and the shared rim's one seam vertex sits on both
                // carriers' seams, so this placement keeps the coaxial
                // neighbour's straight meridian on its carrier too. A marched
                // section has no such convention, so its own domain start is
                // where its single vertex goes — the vertex is a bookkeeping
                // split of a closed curve, not a geometric corner.
                let seam_point = if marched {
                    let [d0, _] = rim.domain()?;
                    rim.evaluate(d0)?
                } else {
                    rim.evaluate(0.0)?
                };
                new_vertex.insert(edge.start_vertex_id, seam_point);
                new_vertex.insert(edge.end_vertex_id, seam_point);
                new_curve.insert(edge.id, rim);
            } else {
                // OPEN rim: an arc / generatrix bounding a window cut through
                // the wall. `rim` is the WHOLE conic the two carriers share, so
                // each endpoint must be re-solved (it is the triple point
                // `S' ∩ N ∩ N'` against the ADJACENT rim's neighbour, or the
                // pushed carrier's own seam) and the conic trimmed between them.
                // Collapsing both onto `rim.evaluate(0.0)` — the closed-rim rule
                // — is what used to tear a multi-loop push apart.
                let previous =
                    &loop_record.coedges[(coedge_index + coedge_count - 1) % coedge_count];
                let next = &loop_record.coedges[(coedge_index + 1) % coedge_count];
                let (at_start, at_end) = if coedge.forward {
                    (previous, next)
                } else {
                    (next, previous)
                };
                let mut resolve = |adjacent_coedge: &CoedgeRecord,
                                   vertex_id: u64|
                 -> Result<RimEnd, String> {
                    let adjacent = *edge_by_id.get(&adjacent_coedge.edge_id).ok_or_else(|| {
                        format!(
                            "offset_ruled_face: missing edge {}",
                            adjacent_coedge.edge_id
                        )
                    })?;
                    let old_point = vertex_pos
                        .get(&vertex_id)
                        .copied()
                        .ok_or_else(|| format!("offset_ruled_face: missing vertex {vertex_id}"))?;
                    let (end, point) = resolve_open_rim_end(
                        solid,
                        face_id,
                        &faces_of_edge,
                        &rim,
                        adjacent,
                        old_point,
                        plane_tolerance,
                    )?;
                    // Both arcs meeting at a corner solve the SAME triple point
                    // from their own conic; a disagreement means the branch pick
                    // went to different roots, so refuse rather than tear.
                    match new_vertex.get(&vertex_id).copied() {
                        Some(existing) if existing.sub(point).length() > plane_tolerance => {
                            return Err(
                                "offset_ruled_face: the two rims meeting at a multi-loop corner \
                                 disagree on its new position — refusing"
                                    .into(),
                            )
                        }
                        Some(_) => {}
                        None => {
                            new_vertex.insert(vertex_id, point);
                        }
                    }
                    Ok(end)
                };
                let start = resolve(at_start, edge.start_vertex_id)?;
                let end = resolve(at_end, edge.end_vertex_id)?;
                open_rims.push(OpenRim {
                    edge_id: edge.id,
                    conic: rim,
                    start,
                    end,
                    old_mid,
                    start_vertex_id: edge.start_vertex_id,
                    end_vertex_id: edge.end_vertex_id,
                });
            }
            if is_plane {
                cap_faces.insert(neighbour);
            } else if is_coaxial_ruled {
                ruled_faces.insert(neighbour);
            } else {
                curved_faces.insert(neighbour);
            }
        }
    }

    // Second pass: trim each open rim's conic to the arc BETWEEN its two solved
    // corners — the one that contains the old edge, picked by the old midpoint's
    // parameter — and orient it start-vertex → end-vertex.
    for rim in &open_rims {
        let [d0, d1] = rim.conic.domain()?;
        let middle = project_point_to_curve(&rim.conic, rim.old_mid)?.u;
        let (from, to) = match (rim.start, rim.end) {
            (RimEnd::Corner(a), RimEnd::Corner(b)) => {
                let (low, high) = if a <= b { (a, b) } else { (b, a) };
                if middle < low || middle > high {
                    return Err(
                        "offset_ruled_face: a multi-loop rim arc wraps the pushed carrier's \
                         periodic seam — deferred (refusing)"
                            .into(),
                    );
                }
                (low, high)
            }
            (RimEnd::Seam, RimEnd::Corner(corner)) | (RimEnd::Corner(corner), RimEnd::Seam) => {
                if middle < corner {
                    (d0, corner)
                } else {
                    (corner, d1)
                }
            }
            (RimEnd::Seam, RimEnd::Seam) => {
                return Err(
                    "offset_ruled_face: a multi-loop rim arc ends on the seam at BOTH ends — \
                     refusing"
                        .into(),
                )
            }
        };
        let mut trimmed = subcurve(&rim.conic, from, to)?;
        let start_point = *new_vertex.get(&rim.start_vertex_id).ok_or_else(|| {
            "offset_ruled_face: a multi-loop rim corner was not relocated — refusing".to_string()
        })?;
        let end_point = *new_vertex.get(&rim.end_vertex_id).ok_or_else(|| {
            "offset_ruled_face: a multi-loop rim corner was not relocated — refusing".to_string()
        })?;
        let [t0, _] = trimmed.domain()?;
        let head = trimmed.evaluate(t0)?;
        if head.sub(start_point).length() > head.sub(end_point).length() {
            trimmed = trimmed.reversed()?;
        }
        new_curve.insert(rim.edge_id, trimmed);
    }

    let pos = |vid: u64| -> Vec3 {
        new_vertex
            .get(&vid)
            .copied()
            .unwrap_or_else(|| vertex_pos[&vid])
    };

    // Seam edge(s): the straight generatrix segment of the offset carrier, from
    // the relocated bottom seam vertex to the top one. Both endpoints were placed
    // on S′'s seam meridian by the rim re-intersections above, so a line between
    // them rides u = 0 of S′ over the SAME axial range the seam had — using the
    // full `iso_curve_u` meridian would wrongly span the untrimmed carrier (a
    // boolean-drilled wall's surface runs past the plate faces).
    for seam_id in &seam_edges {
        let seam = *edge_by_id
            .get(seam_id)
            .ok_or_else(|| format!("offset_ruled_face: missing seam edge {seam_id}"))?;
        let start = *new_vertex.get(&seam.start_vertex_id).ok_or_else(|| {
            "offset_ruled_face: seam endpoint was not relocated by a rim — refusing".to_string()
        })?;
        let end = *new_vertex.get(&seam.end_vertex_id).ok_or_else(|| {
            "offset_ruled_face: seam endpoint was not relocated by a rim — refusing".to_string()
        })?;
        new_curve.insert(*seam_id, make_line(start, end)?);
    }

    // A COAXIAL ruled neighbour also has its own straight seam meridian ending
    // at the shared rim's (now moved) seam vertex. Rebuild it as the chord
    // between its endpoints — one relocated by the rim, the other its unmoved
    // rim vertex — which, both lying on the neighbour's straight generatrix, is
    // exactly the meridian segment.
    for ruled in &ruled_faces {
        let (nshell, nface) = find_face(solid, *ruled)
            .ok_or_else(|| format!("offset_ruled_face: missing ruled neighbour {ruled}"))?;
        for loop_record in &solid.shells[nshell].faces[nface].loops {
            for coedge in &loop_record.coedges {
                let edge = *edge_by_id.get(&coedge.edge_id).ok_or_else(|| {
                    format!("offset_ruled_face: missing edge {}", coedge.edge_id)
                })?;
                let incident = faces_of_edge.get(&edge.id).cloned().unwrap_or_default();
                let is_seam = incident.iter().all(|f| *f == *ruled);
                let touches_moved = new_vertex.contains_key(&edge.start_vertex_id)
                    || new_vertex.contains_key(&edge.end_vertex_id);
                if is_seam && touches_moved && !new_curve.contains_key(&edge.id) {
                    new_curve.insert(
                        edge.id,
                        make_line(pos(edge.start_vertex_id), pos(edge.end_vertex_id))?,
                    );
                }
            }
        }
    }

    // A CURVED neighbour keeps its surface, so it keeps every 3D CURVE it owns
    // — a sphere's seam meridian is the same great-circle arc after the push as
    // before it. What changes is where that curve is TRIMMED: the endpoint the
    // moved rim carried with it slides along the curve to a new parameter.
    // Re-solving it as a parameter on the existing curve is EXACT, and it is
    // strictly better than rebuilding: the coaxial-ruled lane above can chord
    // its seam only because a ruled revolution's meridian is straight, and a
    // sphere's is not.
    let mut new_range: HashMap<u64, (f64, f64)> = HashMap::default();
    for curved in &curved_faces {
        let (nshell, nface) = find_face(solid, *curved)
            .ok_or_else(|| format!("offset_ruled_face: missing curved neighbour {curved}"))?;
        for loop_record in &solid.shells[nshell].faces[nface].loops {
            for coedge in &loop_record.coedges {
                let edge = *edge_by_id.get(&coedge.edge_id).ok_or_else(|| {
                    format!("offset_ruled_face: missing edge {}", coedge.edge_id)
                })?;
                if new_curve.contains_key(&edge.id) || new_range.contains_key(&edge.id) {
                    continue;
                }
                let start_moved = new_vertex.get(&edge.start_vertex_id).copied();
                let end_moved = new_vertex.get(&edge.end_vertex_id).copied();
                if start_moved.is_none() && end_moved.is_none() {
                    continue;
                }
                if edge.start_vertex_id == edge.end_vertex_id {
                    // A closed or degenerate edge of the neighbour (its own
                    // opposite rim, or a pole) whose vertex a rim relocated:
                    // the whole curve would have to move, which this lane does
                    // not do.
                    return Err(format!(
                        "offset_ruled_face: the push moved the vertex of curved neighbour \
                         {curved}'s CLOSED edge {} — deferred (refusing)",
                        edge.id
                    ));
                }
                let mut range = (edge.t0, edge.t1);
                for (moved, slot) in [(start_moved, 0usize), (end_moved, 1usize)] {
                    let Some(point) = moved else { continue };
                    let projection = project_point_to_curve(&edge.curve, point)?;
                    if projection.distance > plane_tolerance {
                        return Err(format!(
                            "offset_ruled_face: a relocated rim vertex left curved neighbour \
                             {curved}'s edge {} (off by {:.3e}) — refusing",
                            edge.id, projection.distance
                        ));
                    }
                    if slot == 0 {
                        range.0 = projection.u;
                    } else {
                        range.1 = projection.u;
                    }
                }
                new_range.insert(edge.id, range);
            }
        }
    }
    // Fail-safe: every relocated vertex must belong only to edges this push
    // rebuilt or re-trimmed. A vertex that also ends an edge of some other face
    // would leave that face's boundary behind, which is exactly the silent
    // tear the ruled lane's own corner guard refuses.
    if !curved_faces.is_empty() {
        for edge in &solid.edges {
            if new_curve.contains_key(&edge.id) || new_range.contains_key(&edge.id) {
                continue;
            }
            if new_vertex.contains_key(&edge.start_vertex_id)
                || new_vertex.contains_key(&edge.end_vertex_id)
            {
                return Err(format!(
                    "offset_ruled_face: relocating a curved neighbour's rim moved the end of \
                     edge {}, which this push does not rebuild — refusing",
                    edge.id
                ));
            }
        }
    }

    // A multi-loop window's corner is a TRIPLE point, so it is also the end of a
    // third edge that belongs to neither the pushed face nor a rim: the two
    // fixed neighbour planes' own shared edge (a slot floor meeting a slot
    // side wall). Both carriers are fixed, so that edge stays on their
    // intersection line and the chord between its (possibly relocated)
    // endpoints IS the rebuilt edge. Skipped entirely when no open rim was
    // rebuilt, so the single-loop paths are untouched.
    if !open_rims.is_empty() {
        let mut corner_edges: Vec<(u64, NurbsCurve)> = Vec::new();
        for edge in &solid.edges {
            if new_curve.contains_key(&edge.id) {
                continue;
            }
            if !new_vertex.contains_key(&edge.start_vertex_id)
                && !new_vertex.contains_key(&edge.end_vertex_id)
            {
                continue;
            }
            if edge.curve.degree != 1 || edge.curve.control_points.len() != 2 {
                return Err(
                    "offset_ruled_face: the push moved the end of a CURVED edge that is not a \
                     rebuilt rim — refusing"
                        .into(),
                );
            }
            let start = pos(edge.start_vertex_id);
            let end = pos(edge.end_vertex_id);
            for neighbour in faces_of_edge.get(&edge.id).cloned().unwrap_or_default() {
                if !cap_faces.contains(&neighbour) {
                    return Err(
                        "offset_ruled_face: a relocated corner borders a face this push does not \
                         re-trim — refusing"
                            .into(),
                    );
                }
                let (nshell, nface) = find_face(solid, neighbour).ok_or_else(|| {
                    format!("offset_ruled_face: missing neighbour {neighbour}")
                })?;
                let plane = plane_of_surface(
                    &solid.shells[nshell].faces[nface].surface,
                    plane_tolerance,
                    "offset_ruled_face",
                )?;
                for point in [start, end] {
                    if point.sub(plane.origin).dot(plane.normal).abs() > plane_tolerance {
                        return Err(
                            "offset_ruled_face: a relocated corner left one of its fixed \
                             neighbour planes — refusing"
                                .into(),
                        );
                    }
                }
            }
            corner_edges.push((edge.id, make_line(start, end)?));
        }
        for (edge_id, curve) in corner_edges {
            new_curve.insert(edge_id, curve);
        }
    }

    // Every edge this push touched, by curve or by trim range — the selective
    // re-fit's work list.
    let changed_edges: HashSet<u64> = new_curve
        .keys()
        .chain(new_range.keys())
        .copied()
        .collect();

    // --- Apply to a fresh clone (the input is never mutated) ---------------
    let mut result = solid.clone();
    for edge in &mut result.edges {
        if let Some(curve) = new_curve.get(&edge.id) {
            // A trimmed rim arc carries its PARENT conic's parameter range
            // (`NurbsCurve::split` preserves knot values), so the edge range
            // comes from the curve. Every whole-curve rebuild — the closed rims,
            // the seam lines — still lands on [0, 1] exactly as before.
            let [d0, d1] = curve.domain()?;
            edge.curve = curve.clone();
            edge.t0 = d0;
            edge.t1 = d1;
        } else if let Some((t0, t1)) = new_range.get(&edge.id) {
            // A curved neighbour's own edge: same curve, new trim.
            edge.t0 = *t0;
            edge.t1 = *t1;
        }
    }
    for vertex in &mut result.vertices {
        if let Some(point) = new_vertex.get(&vertex.id) {
            vertex.point = *point;
        }
    }
    // The pushed face rides the offset carrier. S′ shares the source's parameter
    // domain + seam azimuth (same make_revolution frame/span), so the face's
    // existing (u, v) pcurves stay valid when every rim stayed at its axial
    // station (the planar-cap push) — only the surface swaps.
    result.shells[pshell].faces[pface].surface = s_prime;

    let final_edges: HashMap<u64, EdgeRecord> =
        result.edges.iter().map(|e| (e.id, e.clone())).collect();

    // A COAXIAL ruled neighbour moves the shared rim ALONG the axis, so the
    // pushed face's pcurve for that rim (and the seam whose endpoint moved) is
    // no longer at its old v — refit both the pushed face and every ruled
    // neighbour on their (axially-grown-if-needed) carriers. The planar-only
    // push skips this and keeps the exact pcurve reuse above.
    //
    // A CURVED neighbour moves the shared rim just as much (a dome's rim climbs
    // the sphere as the rod it caps grows), so it joins the same pass — but its
    // own carrier neither moves nor grows: a sphere, a torus and a full
    // revolution are already closed over their whole surface, and the rim was
    // intersected WITH that surface, so it is on it by construction. Its retrim
    // is therefore the same driver with a growth that does nothing.
    if !ruled_faces.is_empty() {
        retrim_offset_ruled_face(&mut result, face_id, &final_edges, tolerance)?;
        for ruled in &ruled_faces {
            retrim_offset_ruled_face(&mut result, *ruled, &final_edges, tolerance)?;
        }
        for curved in &curved_faces {
            refit_changed_pcurves(&mut result, *curved, &final_edges, &changed_edges, tolerance)?;
        }
    } else if !curved_faces.is_empty() {
        // The pushed carrier still has to GROW where the rim climbed past its
        // axial span — the same exact prolongation the ruled lane uses — but its
        // pcurves are re-fitted selectively, not wholesale.
        let (gshell, gface) = find_face(&result, face_id)
            .ok_or_else(|| format!("offset_ruled_face: missing face {face_id}"))?;
        let samples = boundary_samples(
            &result.shells[gshell].faces[gface],
            &final_edges,
            "offset_ruled_face",
        )?;
        extend_ruled_neighbour_over(&mut result, face_id, &samples, tolerance)?;
        refit_changed_pcurves(&mut result, face_id, &final_edges, &changed_edges, tolerance)?;
        for curved in &curved_faces {
            refit_changed_pcurves(&mut result, *curved, &final_edges, &changed_edges, tolerance)?;
        }
    } else if !rim_edges.is_empty() {
        // Every rebuilt RIM needs a fresh pcurve on S'. An open rim because its
        // azimuth span (an arc) or station (a generatrix) genuinely moved; a
        // CLOSED rim because the replacement conic carries the INTERSECTOR's
        // parameterization, which need not be the one the incoming curve had —
        // a boolean-cut cylinder's cap circle comes back re-parameterized, and
        // a closed rim under an OBLIQUE planar cap is an ellipse whose height
        // profile v(u) changes with the offset (the cap plane climbs as the
        // carrier grows: measured 0.382 against a 0.206 limit when the old
        // pcurve was kept on the oblique-cut cone of PushFaceTest2). So this
        // pass runs for every rebuilt rim, not only when an open rim exists;
        // for a perpendicular cap circle it rebuilds the same v = const line.
        // `validate` pairs pcurve to curve by FRACTION of their domains, so a
        // pointwise-identical circle with a different parameter distribution
        // still reads as a gross deviation (measured 5.96 and 12.0 on the slot
        // fixture, against a 0.394 limit).
        //
        // The SEAM edges are deliberately left alone: each is rebuilt as a
        // straight chord over the same axial range, so fraction ↦ height is
        // unchanged and their exact (u, v) — including the two coedges sitting
        // on OPPOSITE sides of the periodic seam — survives untouched.
        let rebuilt: HashSet<u64> = rim_edges.iter().copied().collect();
        let surface = result.shells[pshell].faces[pface].surface.clone();
        let [u_start, u_end] = surface.domain_u()?;
        let u_period = u_end - u_start;
        for loop_record in &mut result.shells[pshell].faces[pface].loops {
            for coedge in &mut loop_record.coedges {
                if !rebuilt.contains(&coedge.edge_id) {
                    continue;
                }
                let edge = final_edges
                    .get(&coedge.edge_id)
                    .ok_or_else(|| format!("offset_ruled_face: missing edge {}", coedge.edge_id))?;
                let mut pcurve = build_pcurve_on_surface(&surface, &edge.curve)?;
                if !coedge.forward {
                    pcurve = pcurve.reversed()?;
                }
                coedge.pcurve = reanchor_pcurve_u(&pcurve, &coedge.pcurve, u_period)?;
            }
        }
    }

    // The fixed caps re-trim as planar faces around their grown rim circles.
    for cap in &cap_faces {
        let (cshell, cface) = find_face(&result, *cap)
            .ok_or_else(|| format!("offset_ruled_face: missing cap {cap}"))?;
        let plane = plane_of_surface(
            &result.shells[cshell].faces[cface].surface,
            plane_tolerance,
            "offset_ruled_face",
        )?;
        retrim_planar_face(
            &mut result.shells[cshell].faces[cface],
            &plane,
            &final_edges,
            scale,
            "offset_ruled_face",
        )?;
    }

    let issues = result.validate();
    if !issues.is_empty() {
        if std::env::var("BREP_PUSH_HOLE_DEBUG").is_ok() {
            // Which coedge of the pushed face drifted, and by how much, with
            // the pcurve's endpoints so a re-anchoring or direction mistake is
            // visible next to a fitting one.
            let (pshell, pface) = find_face(&result, face_id).expect("pushed face survives");
            let face = &result.shells[pshell].faces[pface];
            for (li, loop_record) in face.loops.iter().enumerate() {
                for coedge in &loop_record.coedges {
                    let Some(edge) = result.edges.iter().find(|e| e.id == coedge.edge_id) else {
                        continue;
                    };
                    let (Ok([p0, p1]), Ok(c0)) = (coedge.pcurve.domain(), edge.curve.evaluate(edge.t0))
                    else {
                        continue;
                    };
                    let mut worst = 0.0f64;
                    for k in 0..=32 {
                        let f = k as f64 / 32.0;
                        let t = if coedge.forward {
                            edge.t0 + (edge.t1 - edge.t0) * f
                        } else {
                            edge.t1 - (edge.t1 - edge.t0) * f
                        };
                        if let (Ok(q), Ok(target)) =
                            (coedge.pcurve.evaluate(p0 + (p1 - p0) * f), edge.curve.evaluate(t))
                        {
                            if let Ok(on_s) = face.surface.evaluate(q.x, q.y) {
                                worst = worst.max(on_s.sub(target).length());
                            }
                        }
                    }
                    let (Ok(a), Ok(b)) = (coedge.pcurve.evaluate(p0), coedge.pcurve.evaluate(p1)) else {
                        continue;
                    };
                    eprintln!(
                        "PUSH-DEBUG face {} loop {li} edge {} fwd={} rebuilt={} t=[{:.4},{:.4}] dom={:?} \
                         curve(t0)=({:.4},{:.4},{:.4}) pcurve ({:.4},{:.4})->({:.4},{:.4}) dev={worst:.4}",
                        face.id,
                        edge.id,
                        coedge.forward,
                        rim_edges.contains(&edge.id),
                        edge.t0,
                        edge.t1,
                        edge.curve.domain().ok(),
                        c0.x,
                        c0.y,
                        c0.z,
                        a.x,
                        a.y,
                        b.x,
                        b.y
                    );
                }
            }
        }
        return Err(format!(
            "offset_ruled_face: pushed solid failed validation: {issues:?}"
        ));
    }
    if let (Ok(before), Ok(after)) =
        (solid_signed_volume(solid), solid_signed_volume(&result))
    {
        if before * after <= 0.0 {
            return Err("offset_ruled_face: the push inverts the solid — refusing".into());
        }
    }
    Ok(result)
}

/// Where an OPEN rim arc's endpoint sits on the rebuilt conic.
#[derive(Clone, Copy)]
enum RimEnd {
    /// A genuine corner: the TRIPLE point `offset carrier ∩ this rim's
    /// neighbour ∩ the ADJACENT rim's neighbour`, carried as the conic
    /// parameter that lands on it.
    Corner(f64),
    /// The pushed carrier's periodic SEAM split this rim in two, so the
    /// endpoint rides the offset carrier's seam meridian — which is exactly
    /// parameter 0 (== 1) of the conic, because `S'` is revolved about the
    /// SAME frame and seam azimuth as the source carrier.
    Seam,
}

/// One open rim arc, held between the two passes of the rebuild.
struct OpenRim {
    edge_id: u64,
    /// The WHOLE conic `S' ∩ N`, before trimming.
    conic: NurbsCurve,
    start: RimEnd,
    end: RimEnd,
    /// The OLD edge's midpoint: picks which of the two complementary arcs
    /// between the corners is the one this edge actually was.
    old_mid: Vec3,
    start_vertex_id: u64,
    end_vertex_id: u64,
}

/// Resolve ONE endpoint of an open rim arc against the edge that follows it
/// around the loop.
///
/// Two cases, and nothing else is admitted:
/// * the adjacent edge is the pushed face's own SEAM — the endpoint rides the
///   offset carrier's seam meridian;
/// * the adjacent edge's fixed neighbour `N'` is a PLANE — the endpoint is the
///   triple point `S' ∩ N ∩ N'`, i.e. where this rim's conic (already the
///   `S' ∩ N` curve) crosses `N'`. The crossing nearest the old vertex is the
///   branch, so a conic that meets `N'` twice picks the right corner.
///
/// A conic that LIES IN `N'` means the boolean merely split one rim in two
/// (`N == N'`); there is no triple point, and the split simply keeps its
/// azimuth on the rebuilt conic. Any other adjacent neighbour (a curved
/// carrier) makes `plane_of_surface` refuse, which is the fail-safe.
fn resolve_open_rim_end(
    solid: &BrepSolid,
    face_id: u64,
    faces_of_edge: &HashMap<u64, Vec<u64>>,
    conic: &NurbsCurve,
    adjacent: &EdgeRecord,
    old_point: Vec3,
    plane_tolerance: f64,
) -> Result<(RimEnd, Vec3), String> {
    let incident = faces_of_edge.get(&adjacent.id).cloned().unwrap_or_default();
    if incident.iter().all(|f| *f == face_id) {
        let [d0, _] = conic.domain()?;
        return Ok((RimEnd::Seam, conic.evaluate(d0)?));
    }
    let other = *incident
        .iter()
        .find(|f| **f != face_id)
        .ok_or_else(|| format!("offset_ruled_face: edge {} has no neighbour", adjacent.id))?;
    let (nshell, nface) = find_face(solid, other)
        .ok_or_else(|| format!("offset_ruled_face: missing neighbour {other}"))?;
    let plane = plane_of_surface(
        &solid.shells[nshell].faces[nface].surface,
        plane_tolerance,
        "offset_ruled_face",
    )?;
    if curve_lies_in_plane(conic, &plane, plane_tolerance)? {
        let projection = project_point_to_curve(conic, old_point)?;
        return Ok((RimEnd::Corner(projection.u), conic.evaluate(projection.u)?));
    }
    let mut best: Option<(f64, f64)> = None;
    for parameter in plane_crossing_params(conic, &plane)? {
        let distance = conic.evaluate(parameter)?.sub(old_point).length();
        if best.map(|(best, _)| distance < best).unwrap_or(true) {
            best = Some((distance, parameter));
        }
    }
    let (_, parameter) = best.ok_or_else(|| {
        "offset_ruled_face: a multi-loop rim corner no longer meets its adjacent neighbour \
         (the push pulled the window off it) — refusing"
            .to_string()
    })?;
    Ok((RimEnd::Corner(parameter), conic.evaluate(parameter)?))
}

/// TRUE when the whole curve lies in the plane (so it cannot cross it): the
/// adjacent rim shares this rim's carrier plane and the shared vertex is a
/// SPLIT point, not a triple point.
fn curve_lies_in_plane(
    curve: &NurbsCurve,
    plane: &Plane,
    plane_tolerance: f64,
) -> Result<bool, String> {
    let [d0, d1] = curve.domain()?;
    for step in 0..=16 {
        let t = d0 + (d1 - d0) * step as f64 / 16.0;
        if curve
            .evaluate(t)?
            .sub(plane.origin)
            .dot(plane.normal)
            .abs()
            > plane_tolerance
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Every parameter at which `curve` crosses `plane`, by dense sampling of the
/// signed plane distance plus bisection on each sign change. The curves here
/// are conics (a circle crosses a slot wall twice, a generatrix line crosses a
/// slot floor once), so a sampled sign-change sweep finds every root; bisection
/// then drives it to the last bit rather than to a fit tolerance.
fn plane_crossing_params(curve: &NurbsCurve, plane: &Plane) -> Result<Vec<f64>, String> {
    let [d0, d1] = curve.domain()?;
    let signed = |t: f64| -> Result<f64, String> {
        Ok(curve.evaluate(t)?.sub(plane.origin).dot(plane.normal))
    };
    const SAMPLES: usize = 512;
    let mut roots = Vec::new();
    let mut previous = (d0, signed(d0)?);
    for index in 1..=SAMPLES {
        let t = d0 + (d1 - d0) * index as f64 / SAMPLES as f64;
        let value = signed(t)?;
        if previous.1 == 0.0 {
            roots.push(previous.0);
        } else if (previous.1 < 0.0) != (value < 0.0) {
            let (mut low, mut high) = (previous.0, t);
            let mut low_value = previous.1;
            for _ in 0..100 {
                let middle = 0.5 * (low + high);
                if middle <= low || middle >= high {
                    break;
                }
                let middle_value = signed(middle)?;
                if (low_value < 0.0) != (middle_value < 0.0) {
                    high = middle;
                } else {
                    low = middle;
                    low_value = middle_value;
                }
            }
            roots.push(0.5 * (low + high));
        }
        previous = (t, value);
    }
    if previous.1 == 0.0 {
        roots.push(previous.0);
    }
    Ok(roots)
}

/// The piece of `curve` over `[from, to]`. `NurbsCurve::split` keeps the parent
/// parameterization, so the result's own domain IS `[from, to]` — which is what
/// the edge's `t0`/`t1` are then set from.
fn subcurve(curve: &NurbsCurve, from: f64, to: f64) -> Result<NurbsCurve, String> {
    let [d0, d1] = curve.domain()?;
    let span = (d1 - d0).max(1e-12);
    if to - from <= 1e-9 * span {
        return Err("offset_ruled_face: a rebuilt rim arc collapsed to a point — refusing".into());
    }
    let mut trimmed = curve.clone();
    if to < d1 - 1e-9 * span {
        trimmed = trimmed.split(to)?.0;
    }
    if from > d0 + 1e-9 * span {
        trimmed = trimmed.split(from)?.1;
    }
    Ok(trimmed)
}

/// Orient a rebuilt CLOSED rim the way the edge it replaces ran.
///
/// `intersect_analytic_pair` always emits its conics in its OWN direction
/// (increasing azimuth about the carrier frame). The edge being replaced need
/// not run that way: a wall's top and bottom cap circles are traversed in
/// OPPOSITE directions around the face loop, so one of them arrives decreasing.
/// Both are the same point set and a closed rim's two endpoints are the same
/// vertex, so `validate` cannot tell them apart — but the face's pcurve for the
/// reversed one runs `u: 1 → 0`, and handing the loop an increasing rim tears it
/// open in parameter space, which the Green's-theorem volume integral reads as a
/// completely different solid (measured 138.18 against a true 1024.11).
///
/// Decided on the START TANGENT, which is local and exact: a closed rim starts
/// on the carrier's seam and so does the rebuilt conic, so the two tangents
/// there either agree or oppose. OPEN rims are not orientated here — they are
/// trimmed to their solved corners first and then turned to run
/// start-vertex → end-vertex.
fn match_closed_rim_direction(
    rim: NurbsCurve,
    previous: &EdgeRecord,
) -> Result<NurbsCurve, String> {
    let [d0, _] = rim.domain()?;
    let incoming = previous.curve.derivatives(previous.t0, 1)?;
    let rebuilt = rim.derivatives(d0, 1)?;
    if incoming[1].dot(rebuilt[1]) < 0.0 {
        return rim.reversed();
    }
    Ok(rim)
}

/// Put a freshly fitted pcurve back on the periodic BRANCH of `u` that the
/// pcurve it replaces used, by the whole-period shift that aligns their starts.
///
/// This is not cosmetic. `build_pcurve_on_surface` CLAMPS an analytic carrier's
/// parameters into `[u0, u1]`, but a face loop on a full revolution legitimately
/// carries `u` one period PAST the domain: the two coedges of the periodic seam
/// sit on opposite sides of it, so the coedge that closes the loop across the
/// seam runs (say) `u: 1 → 2`. `validate` accepts either branch — it evaluates
/// the surface with `evaluate_extended`, which wraps — but the Green's-theorem
/// area/volume integral does NOT: it integrates the wire as drawn in parameter
/// space, and a rim dropped back to `u: 0 → 1` tears the loop open and yields a
/// nonsense volume (measured 138.18 for a solid whose true volume is 1024.11).
///
/// The replacement rim traverses the same path in the same direction as the edge
/// it replaces — only its parameter DISTRIBUTION and (for an open arc) its
/// endpoints change — so aligning the starts fixes the branch, and the endpoint
/// check refuses anything that is not merely re-anchored.
fn reanchor_pcurve_u(
    pcurve: &NurbsCurve,
    previous: &NurbsCurve,
    u_period: f64,
) -> Result<NurbsCurve, String> {
    if !(u_period.is_finite() && u_period > 0.0) {
        return Ok(pcurve.clone());
    }
    let [a0, a1] = pcurve.domain()?;
    let [b0, b1] = previous.domain()?;
    let shift = ((previous.evaluate(b0)?.x - pcurve.evaluate(a0)?.x) / u_period).round() * u_period;
    let shifted = if shift == 0.0 {
        pcurve.clone()
    } else {
        let controls = pcurve
            .control_points
            .iter()
            .map(|control| {
                let mut point = control.point()?;
                point.x += shift;
                Ok(crate::Vec4::from_point(point, control.w))
            })
            .collect::<Result<Vec<_>, String>>()?;
        NurbsCurve::new(pcurve.degree, pcurve.knots.clone(), controls)?
    };
    // Both ends must land within a quarter period of the ones they replace: a
    // rim that reversed direction, or that jumped a branch mid-curve, is not a
    // re-anchoring and must not be silently accepted.
    let end_drift = (shifted.evaluate(a1)?.x - previous.evaluate(b1)?.x).abs();
    if end_drift > 0.25 * u_period {
        return Err(format!(
            "offset_ruled_face: a rebuilt rim traverses the carrier's periodic parameter \
             differently from the edge it replaces (end drift {end_drift}) — refusing"
        ));
    }
    Ok(shifted)
}

/// TRUE when `neighbour` is a ruled revolution (cylinder or cone) sharing the
/// pushed carrier's axis LINE — same axis direction (up to sign) and the axes
/// coincident. Only such a neighbour has a closed-form re-intersection with the
/// offset carrier (`intersect_coaxial_revolutions`); the tolerances mirror that
/// intersector so an accepted neighbour is one it can actually solve.
pub(super) fn ruled_neighbour_is_coaxial(
    neighbour: &NurbsSurface,
    axis_origin: Vec3,
    axis_dir: Vec3,
    scale: f64,
) -> bool {
    let Some(AnalyticSurface::RuledRevolution { frame, .. }) = neighbour.analytic() else {
        return false;
    };
    if frame.axis.dot(axis_dir).abs() < 1.0 - 1e-9 {
        return false;
    }
    let offset = frame.origin.sub(axis_origin);
    let perpendicular = offset.sub(axis_dir.scale(offset.dot(axis_dir)));
    perpendicular.length() <= 1e-9 * scale.max(1.0)
}

/// Re-trim a ruled-revolution face (the pushed carrier itself, now S′, or a
/// coaxial ruled neighbour) whose boundary moved axially: grow the carrier along
/// its axis to cover the updated boundary (`extend_ruled_neighbour_over`, exact),
/// then rebuild every pcurve from the already-updated edge curves. The offset
/// analogue of `retrim_planar_face` — all loops are visited, so holes carry.
///
/// The three-phase body is `crate::offset_retrim::retrim_face_in_solid`; this is
/// the ruled growth strategy and the `offset_ruled_face` refusal prefix.
pub(super) fn retrim_offset_ruled_face(
    result: &mut BrepSolid,
    face_id: u64,
    final_edges: &HashMap<u64, EdgeRecord>,
    tolerance: f64,
) -> Result<(), String> {
    let (shell, face_pos) = find_face(result, face_id)
        .ok_or_else(|| format!("offset_ruled_face: missing ruled face {face_id}"))?;
    retrim_face_in_solid(
        result,
        shell,
        face_pos,
        final_edges,
        |solid, points| extend_ruled_neighbour_over(solid, face_id, points, tolerance),
        PcurveFit::SubrangeAware { tolerance },
        "offset_ruled_face",
    )
}

/// A window through the wall bounded entirely by ONE crossing carrier.
struct SingleNeighbourHole {
    neighbour: u64,
    /// Every edge of the loop, in loop order, de-duplicated.
    edge_ids: Vec<u64>,
    /// Corner vertices that are ALSO the end of one of the neighbour's own
    /// edges — its seam meridian, in every case the corpus reaches. Such a
    /// corner is NOT a free bookkeeping split: it is pinned to that meridian,
    /// so it moves along it rather than to the nearest point of the new
    /// section. `(vertex id, the neighbour edge it is pinned to)`.
    pinned: Vec<(u64, u64)>,
}

/// The axial half-plane a revolution carrier's SEAM meridian lies in.
///
/// Every seam of every revolution carrier this lane admits — a cylinder, a
/// cone, a sphere, a torus, a general revolution — is a meridian, and a
/// meridian lies in the half-plane spanned by the axis and its own radial
/// direction. So "where does the new section cross the neighbour's seam" is a
/// curve × plane crossing, in closed form, for all five at once. Returns `None`
/// for a carrier that is not a revolution or a seam that runs ON the axis.
fn seam_axial_plane(surface: &NurbsSurface, seam: &EdgeRecord) -> Option<Plane> {
    let structure = crate::revolution_structure(surface)?;
    let midpoint = seam
        .curve
        .evaluate(0.5 * (seam.t0 + seam.t1))
        .ok()?;
    let offset = midpoint.sub(structure.frame.origin);
    let radial = offset
        .sub(structure.frame.axis.scale(offset.dot(structure.frame.axis)))
        .normalized()
        .ok()?;
    let normal = structure.frame.axis.cross(radial).normalized().ok()?;
    Some(Plane {
        origin: structure.frame.origin,
        u_dir: structure.frame.axis,
        v_dir: radial,
        normal,
    })
}

/// Recognise a loop that is a hole cut by ONE crossing carrier, or say why not.
///
/// Four conditions, each of which the rebuild depends on and none of which it
/// can check afterwards:
///
/// * every coedge's other face is the SAME neighbour — otherwise a corner IS a
///   triple point and belongs to `resolve_open_rim_end`;
/// * that neighbour is neither planar nor coaxial-ruled — the two lanes with
///   their own exact rebuilds, which must keep them (bit-identity);
/// * every edge is OPEN (a closed edge is the single-rim lane's business);
/// * every corner vertex has valence two across the WHOLE solid, so relocating
///   it cannot strand a third face's boundary. This is what rules out the hole
///   that straddles the pushed carrier's periodic seam: its arcs are stitched
///   into the outer loop, so the loop fails the first condition long before the
///   valence test — and either way it is refused rather than torn.
///
/// Returns `Ok(None)` for a loop that is simply not this shape (the outer loop
/// of any ordinary push), so the caller falls through to the lanes it always
/// used.
#[allow(clippy::too_many_arguments)]
fn single_neighbour_hole(
    loop_record: &crate::topology::LoopRecord,
    face_id: u64,
    solid: &BrepSolid,
    faces_of_edge: &HashMap<u64, Vec<u64>>,
    edge_by_id: &HashMap<u64, &EdgeRecord>,
    frame_origin: Vec3,
    frame_axis: Vec3,
    scale: f64,
) -> Result<Option<SingleNeighbourHole>, String> {
    let debug = std::env::var("BREP_PUSH_HOLE_DEBUG").is_ok();
    let mut neighbour: Option<u64> = None;
    let mut edge_ids: Vec<u64> = Vec::new();
    for coedge in &loop_record.coedges {
        let edge = *edge_by_id
            .get(&coedge.edge_id)
            .ok_or_else(|| format!("offset_ruled_face: missing edge {}", coedge.edge_id))?;
        if edge.start_vertex_id == edge.end_vertex_id {
            if debug { eprintln!("HOLE reject: edge {} is closed", edge.id); }
            return Ok(None); // a closed rim: the single-rim lane owns it.
        }
        let incident = faces_of_edge.get(&edge.id).cloned().unwrap_or_default();
        let others: Vec<u64> = incident.into_iter().filter(|f| *f != face_id).collect();
        if others.len() != 1 {
            if debug { eprintln!("HOLE reject: edge {} has {} others", edge.id, others.len()); }
            return Ok(None); // a seam, or a non-manifold edge.
        }
        match neighbour {
            Some(known) if known != others[0] => {
                if debug { eprintln!("HOLE reject: mixed neighbours {known} / {}", others[0]); }
                return Ok(None);
            }
            Some(_) => {}
            None => neighbour = Some(others[0]),
        }
        if !edge_ids.contains(&edge.id) {
            edge_ids.push(edge.id);
        }
    }
    let Some(neighbour) = neighbour else {
        return Ok(None);
    };
    if edge_ids.len() < 2 {
        if debug { eprintln!("HOLE reject: only {} edges", edge_ids.len()); }
        return Ok(None);
    }
    let (nshell, nface) = find_face(solid, neighbour)
        .ok_or_else(|| format!("offset_ruled_face: missing neighbour {neighbour}"))?;
    let surface = &solid.shells[nshell].faces[nface].surface;
    if matches!(surface.analytic(), Some(AnalyticSurface::Plane { .. }))
        || ruled_neighbour_is_coaxial(surface, frame_origin, frame_axis, scale)
    {
        if debug { eprintln!("HOLE reject: neighbour {neighbour} is planar/coaxial"); }
        return Ok(None); // the two lanes that already have exact rebuilds.
    }
    // Every corner is either a free bookkeeping split (nothing else ends there)
    // or PINNED to one edge of the neighbour — its seam. Anything else ending
    // at a corner makes it a genuine junction of three or more faces, which
    // this lane does not solve.
    let neighbour_edges: HashSet<u64> = solid.shells[nshell].faces[nface]
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .map(|coedge| coedge.edge_id)
        .collect();
    let mut pinned: Vec<(u64, u64)> = Vec::new();
    for edge_id in &edge_ids {
        let edge = *edge_by_id
            .get(edge_id)
            .ok_or_else(|| format!("offset_ruled_face: missing edge {edge_id}"))?;
        for vertex_id in [edge.start_vertex_id, edge.end_vertex_id] {
            for other in &solid.edges {
                if other.start_vertex_id != vertex_id && other.end_vertex_id != vertex_id {
                    continue;
                }
                if edge_ids.contains(&other.id) {
                    continue;
                }
                if !neighbour_edges.contains(&other.id) {
                    if debug {
                        eprintln!("HOLE reject: vertex {vertex_id} also on foreign edge {}", other.id);
                    }
                    return Ok(None);
                }
                if pinned
                    .iter()
                    .any(|(known, pin)| *known == vertex_id && *pin != other.id)
                {
                    if debug {
                        eprintln!("HOLE reject: vertex {vertex_id} pinned by two neighbour edges");
                    }
                    return Ok(None);
                }
                if !pinned.iter().any(|(known, _)| *known == vertex_id) {
                    pinned.push((vertex_id, other.id));
                }
            }
        }
    }
    if debug {
        eprintln!("HOLE accept: neighbour {neighbour} edges {edge_ids:?} pinned {pinned:?}");
    }
    Ok(Some(SingleNeighbourHole {
        neighbour,
        edge_ids,
        pinned,
    }))
}

/// Rebuild a single-neighbour hole: re-intersect once, then cut the section.
#[allow(clippy::too_many_arguments)]
fn rebuild_single_neighbour_hole(
    solid: &BrepSolid,
    s_prime: &NurbsSurface,
    hole: &SingleNeighbourHole,
    edge_by_id: &HashMap<u64, &EdgeRecord>,
    vertex_pos: &HashMap<u64, Vec3>,
    tolerance: f64,
    scale: f64,
    new_curve: &mut HashMap<u64, NurbsCurve>,
    new_vertex: &mut HashMap<u64, Vec3>,
) -> Result<(), String> {
    let (nshell, nface) = find_face(solid, hole.neighbour)
        .ok_or_else(|| format!("offset_ruled_face: missing neighbour {}", hole.neighbour))?;
    let neighbour_surface = &solid.shells[nshell].faces[nface].surface;

    // Seed the march with the WHOLE old loop: the section replacing it runs
    // near it for any push small against the feature, and a blind seed grid on
    // two large carriers can miss a small window entirely.
    let mut seeds: Vec<Vec3> = Vec::new();
    let mut reference: Vec<Vec3> = Vec::new();
    for edge_id in &hole.edge_ids {
        let edge = *edge_by_id
            .get(edge_id)
            .ok_or_else(|| format!("offset_ruled_face: missing edge {edge_id}"))?;
        let samples = edge_seeds(edge, 9)?;
        reference.extend(samples.iter().copied());
        seeds.extend(samples);
    }
    let policy = MarchPolicy {
        tolerance,
        residual_tolerance: (scale * 5e-4).max(5e-6),
        seeds,
    };
    let found = match reintersect_carriers(s_prime, neighbour_surface, &policy) {
        Ok(found) => found,
        Err(ReintersectRefusal::Separated) => {
            return Err(
                "offset_ruled_face: the pushed carrier no longer meets a neighbour \
                 (the push separated them, or the rim left the neighbour's domain) — refusing"
                    .into(),
            )
        }
        Err(other) => return Err(format!("offset_ruled_face: {}", other.describe())),
    };
    if found.lane != RimLane::Marched {
        // An analytic section has no polyline to cut, and the exact lanes that
        // produce one already have their own arc rebuild. Refusing here keeps
        // this lane from re-deciding a case the closed forms own.
        return Err(format!(
            "offset_ruled_face: the window bounded by face {} re-intersects in CLOSED FORM, \
             whose arc rebuild is the analytic lane's — deferred (refusing)",
            hole.neighbour
        ));
    }
    if std::env::var("BREP_PUSH_HOLE_DEBUG").is_ok() {
        eprintln!(
            "HOLE rebuild neighbour {}: lane {:?}, {} branch(es), residual {:.3e} \
             (gate {:.3e})",
            hole.neighbour,
            found.lane,
            found.sections.len(),
            found.residual,
            policy.residual_tolerance
        );
    }
    // Which branch is THIS window: two windows cut by one crossing carrier
    // share a surface and so come back as two branches of one section set.
    let section = found
        .nearest_section(&reference)
        .map_err(|error| format!("offset_ruled_face: {error}"))?;

    // Corners first, so every arc is cut against the same relocated vertices.
    //
    // A corner PINNED to the neighbour's seam is not free to go to the nearest
    // sample: it must stay on that meridian, so it goes where the new section
    // CROSSES the meridian's axial half-plane. Placing it at the nearest sample
    // instead leaves it off the seam by the amount the section drifted, which
    // the neighbour's own re-trim then reports as "a relocated rim vertex left
    // curved neighbour N's edge" — a refusal where a correct answer exists.
    for edge_id in &hole.edge_ids {
        let edge = *edge_by_id
            .get(edge_id)
            .ok_or_else(|| format!("offset_ruled_face: missing edge {edge_id}"))?;
        for vertex_id in [edge.start_vertex_id, edge.end_vertex_id] {
            if new_vertex.contains_key(&vertex_id) {
                continue;
            }
            let old = *vertex_pos
                .get(&vertex_id)
                .ok_or_else(|| format!("offset_ruled_face: missing vertex {vertex_id}"))?;
            let point = match hole
                .pinned
                .iter()
                .find(|(known, _)| *known == vertex_id)
                .map(|(_, pin)| *pin)
            {
                Some(pin) => {
                    let seam = *edge_by_id
                        .get(&pin)
                        .ok_or_else(|| format!("offset_ruled_face: missing edge {pin}"))?;
                    let plane = seam_axial_plane(neighbour_surface, seam).ok_or_else(|| {
                        format!(
                            "offset_ruled_face: the window's corner is pinned to edge {pin} of a \
                             neighbour that is not a surface of revolution — refusing"
                        )
                    })?;
                    // The correct crossing is inside the window itself, so the
                    // search reach is the window's own extent about this corner
                    // — the opposite meridian cannot win.
                    let reach = reference
                        .iter()
                        .map(|point| point.sub(old).length())
                        .fold(0.0f64, f64::max)
                        .max(tolerance * 100.0);
                    curve_plane_crossing_near(&section.curve, &plane, old, reach)
                        .map(|(_, point)| point)
                        .ok_or_else(|| {
                            format!(
                                "offset_ruled_face: the rebuilt window never crosses the seam \
                                 (edge {pin}) its corner is pinned to — refusing"
                            )
                        })?
                }
                None => section_corner(section, old)
                    .map_err(|error| format!("offset_ruled_face: {error}"))?,
            };
            new_vertex.insert(vertex_id, point);
        }
    }
    for edge_id in &hole.edge_ids {
        let edge = *edge_by_id
            .get(edge_id)
            .ok_or_else(|| format!("offset_ruled_face: missing edge {edge_id}"))?;
        let from = new_vertex[&edge.start_vertex_id];
        let to = new_vertex[&edge.end_vertex_id];
        let through = edge.curve.evaluate(0.5 * (edge.t0 + edge.t1))?;
        let arc = arc_of_section(section, from, to, through, tolerance)
            .map_err(|error| format!("offset_ruled_face: {error}"))?;
        new_curve.insert(edge.id, arc);
    }
    Ok(())
}

/// Re-fit ONLY the pcurves whose edge actually changed, each back onto the
/// periodic branch of `u` the pcurve it replaces was on.
///
/// This is the CURVED-neighbour lane's re-trim, and it is deliberately not
/// [`crate::offset_retrim::retrim_face_in_solid`]. That driver rebuilds *every*
/// pcurve of the face, which is right for a ruled revolution (whose seam
/// coedges are straight generatrices whose rebuilt pcurves land back on their
/// own sides by luck of the parameterization) and **measurably wrong** for a
/// sphere or a torus. Measured on the ball-capped rod, `d = +0.5`:
///
/// | face | rebuilt-all | correct (independently built r = 3.5 solid) |
/// |---|---|---|
/// | wall seam coedge | `u: 0 → 0` | `u: 1 → 1` |
/// | dome seam coedge | `u: 0 → 0` | `u: 1 → 1` |
/// | dome pole coedge | `(0,1) → (0,1)` | `(1,1) → (0,1)` |
///
/// with a resulting solid that `validate()` accepts and whose Green's-theorem
/// volume reads **207.86 against the true 534.10** — the dome's parameter-space
/// loop collapsed to zero area because both of its seam sides ended up on the
/// same branch. Exactly the failure [`reanchor_pcurve_u`] was written for.
///
/// Two rules together fix it, and both are needed:
/// * **touch only what moved.** A neighbour that keeps its surface keeps every
///   pcurve whose edge did not change; rebuilding one is a chance to land on
///   the wrong branch for no gain.
/// * **re-anchor what is rebuilt**, by the whole-period shift that aligns its
///   start with the pcurve it replaces — the rebuilt rim traverses the same path
///   in the same direction, so aligning the starts fixes the branch, and
///   [`reanchor_pcurve_u`]'s end-drift check refuses anything that is not
///   merely re-anchored.
fn refit_changed_pcurves(
    result: &mut BrepSolid,
    face_id: u64,
    final_edges: &HashMap<u64, EdgeRecord>,
    changed: &HashSet<u64>,
    tolerance: f64,
) -> Result<(), String> {
    let (shell, face_pos) = find_face(result, face_id)
        .ok_or_else(|| format!("offset_ruled_face: missing face {face_id}"))?;
    let surface = result.shells[shell].faces[face_pos].surface.clone();
    let [u_start, u_end] = surface.domain_u()?;
    let u_period = u_end - u_start;
    for loop_record in &mut result.shells[shell].faces[face_pos].loops {
        for coedge in &mut loop_record.coedges {
            if !changed.contains(&coedge.edge_id) {
                continue;
            }
            let edge = final_edges
                .get(&coedge.edge_id)
                .ok_or_else(|| format!("offset_ruled_face: missing edge {}", coedge.edge_id))?;
            // Same subrange rule as `PcurveFit::SubrangeAware`, so a rim that is
            // a strict piece of a full-domain curve keeps the range-aware fit it
            // has always had.
            let [d0, d1] = edge.curve.domain()?;
            let span = (d1 - d0).max(1e-12);
            let is_subrange =
                (edge.t0 - d0).abs() > 1e-9 * span || (edge.t1 - d1).abs() > 1e-9 * span;
            let pcurve = if is_subrange {
                build_pcurve_on_surface_range(
                    &surface,
                    &edge.curve,
                    edge.t0,
                    edge.t1,
                    coedge.forward,
                    tolerance,
                )?
            } else {
                let mut pcurve = build_pcurve_on_surface(&surface, &edge.curve)?;
                if !coedge.forward {
                    pcurve = pcurve.reversed()?;
                }
                pcurve
            };
            coedge.pcurve = reanchor_pcurve_u(&pcurve, &coedge.pcurve, u_period)?;
        }
    }
    Ok(())
}

/// The curve in `curves` whose midpoint is nearest `reference` (branch selection
/// for a surface∩surface intersection that returns more than one component).
fn nearest_curve(curves: &[NurbsCurve], reference: Vec3) -> Result<NurbsCurve, String> {
    let mut best: Option<(f64, &NurbsCurve)> = None;
    for curve in curves {
        let mid = curve.evaluate(0.5)?;
        let d = mid.sub(reference).length();
        if best.map(|(best_d, _)| d < best_d).unwrap_or(true) {
            best = Some((d, curve));
        }
    }
    best.map(|(_, curve)| curve.clone())
        .ok_or_else(|| "offset_ruled_face: empty intersection".into())
}

// BREP private tests: 37e0c543416098c0
