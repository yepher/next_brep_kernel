use super::*;

/// Push a general surface of revolution with planar neighbours.
///
/// The meridian is densely offset and interpolated, then revolved through the
/// original full or partial sweep. A dense normal-distance residual check
/// bounds that approximation. Endpoint rims and start/end meridians are rebuilt,
/// and planar axial/radial neighbours are re-trimmed around the new boundary.
///
/// A boundary whose pcurve is NEITHER a constant-u nor a constant-v line is no
/// longer refused outright: its 3D image on the offset carrier comes from
/// [`crate::image_curve`]. That transfer is only valid where the boundary is
/// definitional, so a general image against a FIXED planar cap is measured
/// against that cap's plane and refused — with the number — when it leaves it,
/// because the honest answer there is a re-intersection this site does not do.
pub fn offset_revolution_face(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
) -> Result<BrepSolid, String> {
    if !distance.is_finite() {
        return Err("offset_revolution_face: distance must be finite".into());
    }
    let scale = solid_model_scale(solid);
    let tolerance = (scale * 1e-7).max(1e-9);
    let residual_tolerance = (scale * 1e-4).max(1e-6);
    // Bar for the general 3D-image ladder: a CURVE-fit accuracy target, which
    // is a different question from `residual_tolerance`'s bound on the offset
    // SURFACE, so it takes the kernel's single named fit-accuracy field.
    let fit_tolerance = crate::KernelTolerances::for_scale(scale, 1e-7).intersection_fit;
    let (shell_index, face_index) = find_face(solid, face_id)
        .ok_or_else(|| format!("offset_revolution_face: no face {face_id}"))?;
    let face = &solid.shells[shell_index].faces[face_index];
    let Some(AnalyticSurface::Revolution { sweep, .. }) = face.surface.analytic() else {
        return Err("offset_revolution_face: the pushed face is not a general revolution".into());
    };

    let source_structure = crate::revolution_structure(&face.surface).ok_or_else(|| {
        "offset_revolution_face: source carrier lost its revolution structure".to_string()
    })?;
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    if (v0.abs() + (v1 - 1.0).abs()) > 1e-9 {
        return Err(
            "offset_revolution_face: a non-normalized generatrix domain is deferred".into(),
        );
    }
    // Offset the meridian densely, then revolve it. A general curve offset is
    // not exactly rational; using 33 cubic interpolation stations materially
    // lowers the residual compared with fitting the whole surface back onto its
    // original sparse control net, while retaining the [0,1] pcurve contract.
    let u_probe = u0 + (u1 - u0) * 0.137;
    // The dense meridian sample is a pointwise offset — the shared evaluator's
    // `Face` lane, whose distance is signed ALONG the oriented normal exactly
    // like direct-edit's own convention (audit §4.1: this side of the kernel
    // grows the solid on a positive distance).
    let offsets = OffsetEvaluator::new(
        "push_revolution",
        &face.surface,
        OffsetNormal::Face {
            same_sense: face.same_sense,
        },
    );
    let mut meridian_points = Vec::with_capacity(33);
    let mut meridian_parameters = Vec::with_capacity(33);
    for index in 0..=32 {
        let parameter = index as f64 / 32.0;
        let v = v0 + (v1 - v0) * parameter;
        let target_point = offsets.at(u_probe, v, distance)?.point;
        let relative = target_point.sub(source_structure.frame.origin);
        let axial = relative.dot(source_structure.frame.axis);
        let radial = relative
            .sub(source_structure.frame.axis.scale(axial))
            .length();
        meridian_points.push(
            source_structure
                .frame
                .origin
                .add(source_structure.frame.x_axis.scale(radial))
                .add(source_structure.frame.axis.scale(axial)),
        );
        meridian_parameters.push(parameter);
    }
    let offset_generatrix = crate::interpolate_curve(&meridian_points, 3, &meridian_parameters)?;
    let offset = crate::make_revolution(
        source_structure.frame.origin,
        source_structure.frame.axis,
        &offset_generatrix,
        *sweep,
    )?;
    let structure = crate::revolution_structure(&offset).ok_or_else(|| {
        "offset_revolution_face: offset carrier lost its revolution structure — refusing"
            .to_string()
    })?;
    if !matches!(offset.analytic(), Some(AnalyticSurface::Revolution { .. })) {
        return Err(
            "offset_revolution_face: offset carrier changed analytic kind — refusing".into(),
        );
    }

    // A general interpolated offset is approximate. Admit it only when its
    // same-parameter displacement stays normal and at the requested distance.
    let target = distance.abs();
    // Stagger away from repeated knots: derivatives at an exact full-period
    // seam or a multiple internal knot are side-dependent and may be zero in
    // the generic evaluator even though the carrier is regular there.
    for iu in 0..13 {
        for iv in 0..13 {
            let u = u0 + (u1 - u0) * (iu as f64 + 0.37) / 13.0;
            let v = v0 + (v1 - v0) * (iv as f64 + 0.41) / 13.0;
            let source_point = face.surface.evaluate(u, v)?;
            let offset_point = offset.evaluate(u, v)?;
            let delta = offset_point.sub(source_point);
            let normal = offsets
                .normal(u, v)
                .map_err(|error| format!("offset_revolution_face: sample normal: {error}"))?;
            let distance_error = (delta.length() - target).abs();
            let normal_error = if delta.length() > tolerance {
                1.0 - delta
                    .normalized()
                    .map_err(|error| format!("offset_revolution_face: sample delta: {error}"))?
                    .dot(normal)
                    .abs()
            } else {
                0.0
            };
            if distance_error > residual_tolerance || normal_error > 5e-4 {
                return Err(format!(
                    "offset_revolution_face: offset fit exceeds tolerance \
                     (distance error {distance_error:.3e}, normal error {normal_error:.3e})"
                ));
            }
        }
    }

    let mut faces_of_edge: HashMap<u64, Vec<u64>> = HashMap::default();
    for shell in &solid.shells {
        for candidate in &shell.faces {
            for loop_record in &candidate.loops {
                for coedge in &loop_record.coedges {
                    faces_of_edge
                        .entry(coedge.edge_id)
                        .or_default()
                        .push(candidate.id);
                }
            }
        }
    }
    let edge_by_id: HashMap<u64, &EdgeRecord> =
        solid.edges.iter().map(|edge| (edge.id, edge)).collect();

    // The offset generatrix endpoints revolve into the only two possible rims.
    let [g0, g1] = structure.generatrix.domain()?;
    let mut rim_candidates = Vec::new();
    for parameter in [g0, g1] {
        let point = structure.generatrix.evaluate(parameter)?;
        let relative = point.sub(structure.frame.origin);
        let axial = relative.dot(structure.frame.axis);
        let radial = relative.sub(structure.frame.axis.scale(axial));
        let radius = radial.length();
        if radius <= tolerance {
            return Err(format!(
                "offset_revolution_face: an offset rim collapses onto the axis \
                 (point {point:?}, frame {:?}) — refusing",
                structure.frame
            ));
        }
        rim_candidates.push(crate::make_arc(
            structure
                .frame
                .origin
                .add(structure.frame.axis.scale(axial)),
            structure.frame.x_axis,
            structure.frame.y_axis,
            radius,
            0.0,
            structure.sweep,
        )?);
    }

    let mut new_curves: HashMap<u64, NurbsCurve> = HashMap::default();
    let mut new_vertices: HashMap<u64, Vec3> = HashMap::default();
    let mut planar_neighbours: HashSet<u64> = HashSet::default();
    for loop_record in &face.loops {
        for coedge in &loop_record.coedges {
            if new_curves.contains_key(&coedge.edge_id) {
                continue;
            }
            let edge = *edge_by_id.get(&coedge.edge_id).ok_or_else(|| {
                format!("offset_revolution_face: missing edge {}", coedge.edge_id)
            })?;
            let incident = faces_of_edge.get(&edge.id).cloned().unwrap_or_default();
            // Kept for the general lane below: an edge shared with a FIXED
            // planar cap has to stay in that cap's plane, and only this loop
            // knows which plane that is.
            let mut neighbour_plane = None;
            if let Some(neighbour) = incident.iter().find(|candidate| **candidate != face_id) {
                let (ns, nf) = find_face(solid, *neighbour).ok_or_else(|| {
                    format!("offset_revolution_face: missing neighbour {neighbour}")
                })?;
                match plane_of_surface(
                    &solid.shells[ns].faces[nf].surface,
                    residual_tolerance,
                    "offset_revolution_face",
                ) {
                    Err(_) => {
                        return Err(
                            "offset_revolution_face: only planar neighbours are supported \
                                for a general revolution — refusing"
                                .into(),
                        );
                    }
                    Ok(plane) => neighbour_plane = Some(plane),
                }
                planar_neighbours.insert(*neighbour);
            }

            // Constant-v pcurve → a revolved endpoint rim. Constant-u → a
            // start/end meridian (or the intrinsic full-sweep seam).
            let [q0, q1] = coedge.pcurve.domain()?;
            let uv0 = coedge.pcurve.evaluate(q0)?;
            let uv1 = coedge.pcurve.evaluate(q1)?;
            let uvm = coedge.pcurve.evaluate(0.5 * (q0 + q1))?;
            let uv_tolerance = 1e-7;
            let mut general_image = false;
            let mut curve = if (uv0.y - uv1.y).abs() <= uv_tolerance
                && (uv0.y - uvm.y).abs() <= uv_tolerance
            {
                let old_mid = edge.curve.evaluate(0.5 * (edge.t0 + edge.t1))?;
                let mut best = rim_candidates[0].clone();
                let mut best_distance = f64::INFINITY;
                for candidate in &rim_candidates {
                    let [c0, c1] = candidate.domain()?;
                    let d = candidate.evaluate(0.5 * (c0 + c1))?.sub(old_mid).length();
                    if d < best_distance {
                        best_distance = d;
                        best = candidate.clone();
                    }
                }
                best
            } else if (uv0.x - uv1.x).abs() <= uv_tolerance && (uv0.x - uvm.x).abs() <= uv_tolerance
            {
                offset.iso_curve_u(uv0.x)?
            } else {
                // A GENERAL boundary pcurve — what this site used to refuse.
                // Its image on the offset carrier is built by the shared
                // ladder, oriented off `coedge.forward` (the authoritative
                // pcurve↔edge relation) rather than the iso lanes' geometric
                // nearest-endpoint match, because a general image need not
                // share an endpoint with the old curve at all.
                general_image = true;
                let image = crate::image_curve(
                    &offset,
                    &coedge.pcurve,
                    fit_tolerance,
                    "offset_revolution_face",
                )?;
                if coedge.forward {
                    image.curve
                } else {
                    image.curve.reversed()?
                }
            };
            // The transfer is only legitimate where the boundary is
            // DEFINITIONAL. Against a FIXED planar cap the true new rim is
            // `offset carrier ∩ cap plane`, and the image of the old pcurve
            // leaves that plane by the offset's out-of-plane component. The
            // validator's `pcurve_acceptance` band (2.5% of the model diagonal)
            // is far too loose to catch that, so measure it here and refuse
            // with the number rather than let a wrong rim through.
            //
            // Scoped to the general lane on purpose: the iso lanes rebuild
            // their rims from the offset generatrix and carry their own,
            // separately tracked, off-plane drift (refusal census §3.2). Making
            // this gate global would change results this slice did not measure.
            if general_image {
                if let Some(plane) = neighbour_plane {
                    let [g0, g1] = curve.domain()?;
                    let mut worst = 0.0f64;
                    for index in 0..=32 {
                        let t = g0 + (g1 - g0) * index as f64 / 32.0;
                        let point = curve.evaluate(t)?;
                        worst = worst.max(point.sub(plane.origin).dot(plane.normal).abs());
                    }
                    if worst > residual_tolerance {
                        return Err(format!(
                            "offset_revolution_face: the general image of boundary edge {} \
                             leaves its fixed planar neighbour by {worst:.3e} (limit \
                             {residual_tolerance:.3e}) — re-intersecting the offset carrier \
                             with a fixed neighbour is a separate capability; refusing",
                            edge.id
                        ));
                    }
                }
            }
            if !general_image {
                let [c0, c1] = curve.domain()?;
                let old_start = edge.curve.evaluate(edge.t0)?;
                if curve.evaluate(c0)?.sub(old_start).length()
                    > curve.evaluate(c1)?.sub(old_start).length()
                {
                    curve = curve.reversed()?;
                }
            }
            let [d0, d1] = curve.domain()?;
            let start = curve.evaluate(d0)?;
            let end = curve.evaluate(d1)?;
            new_vertices.insert(edge.start_vertex_id, start);
            new_vertices.insert(edge.end_vertex_id, end);
            new_curves.insert(edge.id, curve);
        }
    }

    let mut result = solid.clone();
    result.shells[shell_index].faces[face_index].surface = offset;
    for edge in &mut result.edges {
        if let Some(curve) = new_curves.get(&edge.id) {
            let [d0, d1] = curve.domain()?;
            edge.curve = curve.clone();
            edge.t0 = d0;
            edge.t1 = d1;
        }
    }
    for vertex in &mut result.vertices {
        if let Some(point) = new_vertices.get(&vertex.id) {
            vertex.point = *point;
        }
    }
    // A full-revolve planar disk/annulus may carry its own radial seam from an
    // axis vertex to the shared circular rim. Moving the rim relocates one end
    // of that seam, so rebuild the straight edge before re-trimming the cap.
    let result_vertices: HashMap<u64, Vec3> = result
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.point))
        .collect();
    let mut cap_seams = HashSet::default();
    for neighbour in &planar_neighbours {
        let (ns, nf) = find_face(&result, *neighbour)
            .ok_or_else(|| format!("offset_revolution_face: missing cap {neighbour}"))?;
        for loop_record in &result.shells[ns].faces[nf].loops {
            for coedge in &loop_record.coedges {
                let edge = *edge_by_id.get(&coedge.edge_id).ok_or_else(|| {
                    format!(
                        "offset_revolution_face: missing cap edge {}",
                        coedge.edge_id
                    )
                })?;
                let touches_moved = new_vertices.contains_key(&edge.start_vertex_id)
                    || new_vertices.contains_key(&edge.end_vertex_id);
                if touches_moved
                    && !new_curves.contains_key(&edge.id)
                    && !edge.degenerate
                    && edge.curve.degree == 1
                {
                    cap_seams.insert(edge.id);
                }
            }
        }
    }
    for seam_id in cap_seams {
        let source = edge_by_id[&seam_id];
        let curve = make_line(
            result_vertices[&source.start_vertex_id],
            result_vertices[&source.end_vertex_id],
        )?;
        let [d0, d1] = curve.domain()?;
        if let Some(edge) = result.edges.iter_mut().find(|edge| edge.id == seam_id) {
            edge.curve = curve;
            edge.t0 = d0;
            edge.t1 = d1;
        }
    }
    let final_edges: HashMap<u64, EdgeRecord> = result
        .edges
        .iter()
        .map(|edge| (edge.id, edge.clone()))
        .collect();
    for neighbour in planar_neighbours {
        let (ns, nf) = find_face(&result, neighbour)
            .ok_or_else(|| format!("offset_revolution_face: missing cap {neighbour}"))?;
        let plane = plane_of_surface(
            &result.shells[ns].faces[nf].surface,
            residual_tolerance,
            "offset_revolution_face",
        )?;
        retrim_planar_face(
            &mut result.shells[ns].faces[nf],
            &plane,
            &final_edges,
            scale,
            "offset_revolution_face",
        )
        .map_err(|error| format!("offset_revolution_face: cap retrim: {error}"))?;
    }
    let issues = result.validate();
    if !issues.is_empty() {
        return Err(format!(
            "offset_revolution_face: pushed solid failed validation: {issues:?}"
        ));
    }
    if let (Ok(before), Ok(after)) = (solid_signed_volume(solid), solid_signed_volume(&result)) {
        if before * after <= 0.0 {
            return Err("offset_revolution_face: the push inverts the solid — refusing".into());
        }
    }
    Ok(result)
}

// BREP private tests: e1a110296b52d090
