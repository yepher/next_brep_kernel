use super::*;

/// Grow a ruled-revolution neighbour along its axis so its domain covers
/// `points` (with a small margin) — the extension is EXACT (same frame, the
/// generatrix prolonged linearly) and every existing pcurve is remapped
/// affinely in v, so the face's trims are untouched geometrically. Planar or
/// already-covering carriers are left alone.
pub(super) fn extend_ruled_neighbour_over(
    solid: &mut BrepSolid,
    face_id: u64,
    points: &[Vec3],
    tolerance: f64,
) -> Result<(), String> {
    let op = "delete_face_and_heal";
    let (shell, face_pos) =
        find_face(solid, face_id).ok_or_else(|| format!("{op}: missing face {face_id}"))?;
    extend_ruled_carrier(&mut solid.shells[shell].faces[face_pos], points, tolerance)
}

/// The face-local half of [`extend_ruled_neighbour_over`]: everything it does
/// happens inside one `FaceRecord`, and the rejoin in `delete_faces.rs` grows a
/// candidate face it has not put into a solid yet.
pub(super) fn extend_ruled_carrier(
    face: &mut FaceRecord,
    points: &[Vec3],
    tolerance: f64,
) -> Result<(), String> {
    let Some(AnalyticSurface::RuledRevolution {
        frame,
        rho0,
        rho1,
        height,
    }) = face.surface.analytic().cloned()
    else {
        return Ok(());
    };
    let mut low = 0.0f64;
    let mut high = height;
    for point in points {
        let axial = point.sub(frame.origin).dot(frame.axis);
        low = low.min(axial - tolerance);
        high = high.max(axial + tolerance);
    }
    if low >= -tolerance && high <= height + tolerance {
        return Ok(());
    }
    let rho_at = |z: f64| rho0 + (rho1 - rho0) * z / height;
    let base = frame.origin.add(frame.axis.scale(low));
    let start = base.add(frame.x_axis.scale(rho_at(low)));
    let end = frame
        .origin
        .add(frame.axis.scale(high))
        .add(frame.x_axis.scale(rho_at(high)));
    let generatrix = make_line(start, end)?;
    let extended = make_revolution(base, frame.axis, &generatrix, std::f64::consts::TAU)?;
    // v remap old -> new: z = old_v * height; new_v = (z - low)/(high - low).
    let v_scale = height / (high - low);
    let v_offset = -low / (high - low);
    face.surface = extended;
    for coedge in face
        .loops
        .iter_mut()
        .flat_map(|loop_record| &mut loop_record.coedges)
    {
        for control in &mut coedge.pcurve.control_points {
            control.y = control.y * v_scale + control.w * v_offset;
        }
    }
    Ok(())
}

/// Grow a partial-sweep straight-generatrix `Revolution` neighbour (a fillet
/// band / partial cylinder-or-cone wall) along its axis so its domain covers
/// `points`. `extend_ruled_neighbour_over` only knows the full-2π
/// `RuledRevolution` variant and no-ops on a partial `Revolution`; this is its
/// partial-sweep analogue. The straight generatrix is prolonged along its own
/// 3D line — which keeps its azimuth and radius law exactly — to the required
/// axial span, and the band is re-revolved over the SAME sweep. `retrim_ruled_face`
/// rebuilds every pcurve from scratch on the grown surface afterwards, so no
/// v-remap is needed here. A non-`Revolution` surface, or one that already
/// covers the boundary, is left untouched.
pub(super) fn extend_revolution_carrier_over(
    solid: &mut BrepSolid,
    face_id: u64,
    points: &[Vec3],
    tolerance: f64,
) -> Result<(), String> {
    let (shell, face_pos) = find_face(solid, face_id)
        .ok_or_else(|| format!("move_faces: missing ruled face {face_id}"))?;
    extend_revolution_carrier(&mut solid.shells[shell].faces[face_pos], points, tolerance)
}

/// The face-local half of [`extend_revolution_carrier_over`] — see
/// [`extend_ruled_carrier`] for why the rejoin needs one.
pub(super) fn extend_revolution_carrier(
    face: &mut FaceRecord,
    points: &[Vec3],
    tolerance: f64,
) -> Result<(), String> {
    let Some(AnalyticSurface::Revolution {
        frame,
        sweep,
        generatrix,
        ..
    }) = face.surface.analytic().cloned()
    else {
        return Ok(());
    };
    let controls = &generatrix.control_points;
    if generatrix.degree != 1 || controls.len() != 2 {
        return Ok(()); // curved generatrix — not a ruled band, nothing to grow
    }
    let p0 = controls[0].point()?;
    let p1 = controls[1].point()?;
    let axial = |p: Vec3| p.sub(frame.origin).dot(frame.axis);
    let (z0, z1) = (axial(p0), axial(p1));
    let (span_lo, span_hi) = (z0.min(z1), z0.max(z1));
    let mut low = span_lo;
    let mut high = span_hi;
    for &point in points {
        let a = axial(point);
        low = low.min(a - tolerance);
        high = high.max(a + tolerance);
    }
    if low >= span_lo - tolerance && high <= span_hi + tolerance {
        return Ok(()); // the current band already covers the moved boundary
    }
    let denom = z1 - z0;
    if denom.abs() <= tolerance {
        return Ok(());
    }
    // Prolong the generatrix line to the required axial span (same 3D line ⇒
    // same azimuth and radius law). Recognition validated the carrier by exact
    // reconstruction, so `make_revolution(frame.origin, frame.axis, generatrix,
    // sweep)` reproduces the SOURCE surface bit-for-bit — including its normal
    // orientation — PROVIDED the generatrix keeps its control[0]→control[1]
    // direction. So extend along that same sense (not blindly low→high), else
    // the rebuilt surface's normal flips and the face reads as inside-out.
    let point_at = |z: f64| -> Vec3 {
        let t = (z - z0) / denom;
        p0.add(p1.sub(p0).scale(t))
    };
    let (start_axial, end_axial) = if z1 >= z0 { (low, high) } else { (high, low) };
    let extended = make_line(point_at(start_axial), point_at(end_axial))?;
    let grown = make_revolution(frame.origin, frame.axis, &extended, sweep)?;
    face.surface = grown;
    Ok(())
}

/// Grow a face's carrier until it spans `points`, then rebuild EVERY pcurve on
/// the grown carrier from the edge curves.
///
/// The rejoin in `delete_faces.rs` hands one survivor face the loops of its
/// cosurface twin on the far side of a deleted band. Those loops' pcurves were
/// fitted to the TWIN's patch, and the patch that keeps them is a finite piece
/// of surface that stops where its own face stopped — so it has to grow over
/// the twin's extent and the bridges between them, and everything it now
/// carries has to be re-fitted to it. Both halves are the shared re-trim's
/// (`crate::offset_retrim`), taken face-locally because the rejoin is still
/// assembling a candidate face rather than editing one inside a solid.
///
/// The growth is EXACT for every carrier it accepts — a plane rebuilt over the
/// boundary's own box in its own frame, a ruled or partial-sweep revolution
/// prolonged along its axis — and a carrier it does not recognise is left
/// alone, so the refit below refuses if the loops fall outside it.
pub(super) fn regrow_and_refit_carrier(
    face: &mut FaceRecord,
    edges: &HashMap<u64, EdgeRecord>,
    scale: f64,
    op: &str,
) -> Result<(), String> {
    let tolerance = (scale * 1e-7).max(1e-9);
    let points = boundary_samples(face, edges, op)?;
    if let Ok(plane) = plane_of_surface(&face.surface, (scale * 1e-6).max(1e-7), op) {
        // The planar rebuild is `retrim_planar_face`'s, inlined for its
        // SUBRANGE-AWARE refit: a rejoined face keeps pinch-split edges that
        // are strict subranges of their own curve, and a whole-curve pcurve
        // for one of those lands the loop's ends a whole edge away.
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for point in &points {
            let delta = point.sub(plane.origin);
            u_min = u_min.min(delta.dot(plane.u_dir));
            u_max = u_max.max(delta.dot(plane.u_dir));
            v_min = v_min.min(delta.dot(plane.v_dir));
            v_max = v_max.max(delta.dot(plane.v_dir));
        }
        if !(u_min.is_finite() && v_min.is_finite()) {
            return Err(format!("{op}: empty face boundary"));
        }
        let margin = ((u_max - u_min).max(v_max - v_min) * 0.25).max(scale * 1e-3);
        let origin = plane
            .origin
            .add(plane.u_dir.scale(u_min - margin))
            .add(plane.v_dir.scale(v_min - margin));
        face.surface = crate::make_plane(
            origin,
            plane.u_dir,
            plane.v_dir,
            (u_max - u_min) + 2.0 * margin,
            (v_max - v_min) + 2.0 * margin,
        )?;
    } else {
        extend_ruled_carrier(face, &points, tolerance)?;
        extend_revolution_carrier(face, &points, tolerance)?;
    }
    let surface = face.surface.clone();
    crate::offset_retrim::rebuild_loop_pcurves(
        face,
        edges,
        &surface,
        PcurveFit::SubrangeAware { tolerance },
        op,
    )
}

// ---------------------------------------------------------------------------
// §6.12 OPEN-chain heal with curved neighbours: curved PRIMARIES (plane ×
// cylinder/ruled revolution) and curved LATERAL CAPS (a strip ending against
// a curved analytic wall) alike.
// ---------------------------------------------------------------------------

/// Which analytic carrier a neighbour of an open transition strip offers.
/// `Curved` is any recognized non-planar analytic (cylinder/cone wall,
/// partial revolution with a straight generatrix, …) — `intersect_analytic_pair`
/// decides below whether the actual PAIR has a closed-form re-intersection.
pub(super) enum OpenNeighbourCarrier {
    Planar(Plane),
    Curved,
}

/// The parameter/point where `curve` crosses `plane`, restricted to crossings
/// within `reach` of `center` (the deleted strip's region gate); the nearest
/// such crossing wins. Sampled sign changes refined by bisection — accurate to
/// ~1e-13 of the domain span, which is exact-in-practice for the analytic
/// branches this heal clips.
pub(super) fn curve_plane_crossing_near(
    curve: &NurbsCurve,
    plane: &Plane,
    center: Vec3,
    reach: f64,
) -> Option<(f64, Vec3)> {
    let [t0, t1] = curve.domain().ok()?;
    let height = |t: f64| -> Option<f64> {
        curve
            .evaluate(t)
            .ok()
            .map(|point| point.sub(plane.origin).dot(plane.normal))
    };
    const SAMPLES: usize = 64;
    let mut best: Option<(f64, Vec3, f64)> = None;
    let mut previous_t = t0;
    let mut previous_h = height(previous_t)?;
    for index in 1..=SAMPLES {
        let t = t0 + (t1 - t0) * index as f64 / SAMPLES as f64;
        let h = height(t)?;
        if previous_h * h <= 0.0 {
            let (mut low, mut high, mut low_h) = (previous_t, t, previous_h);
            for _ in 0..80 {
                let mid = 0.5 * (low + high);
                let mid_h = height(mid)?;
                if low_h * mid_h <= 0.0 {
                    high = mid;
                } else {
                    low = mid;
                    low_h = mid_h;
                }
            }
            let root = 0.5 * (low + high);
            let point = curve.evaluate(root).ok()?;
            let distance = point.sub(center).length();
            if distance <= reach && best.map(|(_, _, known)| distance < known).unwrap_or(true) {
                best = Some((root, point, distance));
            }
        }
        previous_t = t;
        previous_h = h;
    }
    best.map(|(t, point, _)| (t, point))
}

/// The parameter/point where `curve` crosses a lateral cap's ANALYTIC carrier,
/// restricted to crossings within `reach` of `center` (the deleted strip's
/// region gate); the nearest such crossing wins. A PLANAR cap uses the exact
/// bisection solver above; a CURVED analytic cap uses the Newton
/// curve×surface intersector against the cap face's carrier surface.
pub(super) fn curve_cap_crossing_near(
    curve: &NurbsCurve,
    cap_carrier: &OpenNeighbourCarrier,
    cap_surface: &NurbsSurface,
    center: Vec3,
    reach: f64,
    tolerance: f64,
) -> Option<(f64, Vec3)> {
    match cap_carrier {
        OpenNeighbourCarrier::Planar(plane) => {
            curve_plane_crossing_near(curve, plane, center, reach)
        }
        OpenNeighbourCarrier::Curved => {
            let hits = intersect_curve_surface(curve, cap_surface, tolerance).ok()?;
            let mut best: Option<(f64, Vec3, f64)> = None;
            for hit in hits {
                let point = curve.evaluate(hit.t).ok()?;
                let distance = point.sub(center).length();
                if distance <= reach && best.map(|(_, _, known)| distance < known).unwrap_or(true) {
                    best = Some((hit.t, point, distance));
                }
            }
            best.map(|(t, point, _)| (t, point))
        }
    }
}

/// Move a side edge's endpoint(s) onto recovered corner(s). Prefers WIDENING:
/// the corner is projected onto the edge's underlying curve and, when it lands
/// on it (a blend trimmed this very curve back, so growing the represented
/// interval restores it exactly — line, arc, or spline alike), only t0/t1 and
/// the vertex ids change and the geometry is bit-exact. A straight line whose
/// curve stops short of the corner is rebuilt from its resolved endpoints
/// instead. Returns Ok(false) — leaving the edge untouched — when the stored
/// curve stops short and is not a line (booleans re-fit curves to exactly the
/// represented range); the caller may then re-derive the edge in closed form
/// from its flanking carriers. Hard refusals (degenerate edge, zero-length
/// collapse of a line) stay errors.
pub(super) fn relocate_open_side_edge(
    edge: &mut EdgeRecord,
    start_target: Option<(u64, Vec3)>,
    end_target: Option<(u64, Vec3)>,
    on_curve_tolerance: f64,
    tolerance: f64,
    op: &str,
) -> Result<bool, String> {
    if edge.degenerate {
        return Err(format!(
            "{op}: degenerate side edge {} cannot be relocated (deferred)",
            edge.id
        ));
    }
    let [d0, d1] = edge.curve.domain()?;
    let span = (d1 - d0).max(1e-12);
    let snap = |t: f64| -> f64 {
        if (t - d0).abs() <= 1e-9 * span {
            d0
        } else if (t - d1).abs() <= 1e-9 * span {
            d1
        } else {
            t
        }
    };
    let widen = |target: Vec3| -> Option<f64> {
        let projection = project_point_to_curve(&edge.curve, target).ok()?;
        (projection.distance <= on_curve_tolerance).then(|| snap(projection.u))
    };
    let new_t0 = match &start_target {
        Some((_, point)) => widen(*point),
        None => Some(edge.t0),
    };
    let new_t1 = match &end_target {
        Some((_, point)) => widen(*point),
        None => Some(edge.t1),
    };
    if let (Some(new_t0), Some(new_t1)) = (new_t0, new_t1) {
        if new_t1 - new_t0 > 1e-9 * span {
            edge.t0 = new_t0;
            edge.t1 = new_t1;
            if let Some((vertex, _)) = start_target {
                edge.start_vertex_id = vertex;
            }
            if let Some((vertex, _)) = end_target {
                edge.end_vertex_id = vertex;
            }
            return Ok(true);
        }
    }
    // The curve itself does not reach the corner: rebuild straight lines,
    // hand everything else back to the caller for closed-form re-derivation.
    if edge.curve.degree == 1 && edge.curve.control_points.len() == 2 {
        let start_point = match &start_target {
            Some((_, point)) => *point,
            None => edge.curve.evaluate(edge.t0)?,
        };
        let end_point = match &end_target {
            Some((_, point)) => *point,
            None => edge.curve.evaluate(edge.t1)?,
        };
        if start_point.sub(end_point).length() <= tolerance {
            return Err(format!(
                "{op}: healing would collapse side edge {} to zero length",
                edge.id
            ));
        }
        edge.curve = make_line(start_point, end_point)?;
        edge.t0 = 0.0;
        edge.t1 = 1.0;
        if let Some((vertex, _)) = start_target {
            edge.start_vertex_id = vertex;
        }
        if let Some((vertex, _)) = end_target {
            edge.end_vertex_id = vertex;
        }
        return Ok(true);
    }
    Ok(false)
}

/// Closed-form fallback for a side edge whose STORED curve stops short of the
/// recovered corner (booleans re-fit intersection curves to exactly the
/// represented range, so widening has nothing to widen into). The side edge
/// lies on the intersection of its two flanking faces' analytic carriers:
/// re-derive that intersection with `intersect_analytic_pair`, pick the branch
/// that carries BOTH resolved endpoints AND the existing edge (its midpoint
/// must project onto the branch inside the new interval — this also rejects a
/// periodic branch whose direct parameter interval runs the wrong way around),
/// and re-trim it from the preserved endpoint to the corner. Refused honestly
/// when the flanking pair has no closed form or no branch qualifies.
pub(super) fn rederive_side_edge_on_carriers(
    solid: &BrepSolid,
    edge: &EdgeRecord,
    start_target: Option<(u64, Vec3)>,
    end_target: Option<(u64, Vec3)>,
    on_curve_tolerance: f64,
    tolerance: f64,
    op: &str,
) -> Result<EdgeRecord, String> {
    let mut adjacent: Vec<&NurbsSurface> = Vec::new();
    for shell in &solid.shells {
        for face in &shell.faces {
            if face
                .loops
                .iter()
                .any(|loop_record| loop_record.coedges.iter().any(|c| c.edge_id == edge.id))
            {
                adjacent.push(&face.surface);
            }
        }
    }
    if adjacent.len() != 2 {
        return Err(format!(
            "{op}: side edge {} is not shared by exactly two faces (found {})",
            edge.id,
            adjacent.len()
        ));
    }
    let start_point = match &start_target {
        Some((_, point)) => *point,
        None => edge.curve.evaluate(edge.t0)?,
    };
    let end_point = match &end_target {
        Some((_, point)) => *point,
        None => edge.curve.evaluate(edge.t1)?,
    };
    if start_point.sub(end_point).length() <= tolerance {
        return Err(format!(
            "{op}: healing would collapse side edge {} to zero length",
            edge.id
        ));
    }
    let middle_point = edge.curve.evaluate(0.5 * (edge.t0 + edge.t1))?;
    let branches =
        intersect_analytic_pair(adjacent[0], adjacent[1], tolerance).ok_or_else(|| {
            format!(
                "{op}: side edge {} cannot be extended onto the recovered corner (its curve \
             stops short and its flanking carriers are not a closed-form analytic pair — \
             deferred)",
                edge.id
            )
        })?;
    for branch in branches {
        let Ok(projected_start) = project_point_to_curve(&branch, start_point) else {
            continue;
        };
        let Ok(projected_end) = project_point_to_curve(&branch, end_point) else {
            continue;
        };
        let Ok(projected_middle) = project_point_to_curve(&branch, middle_point) else {
            continue;
        };
        if projected_start.distance > on_curve_tolerance
            || projected_end.distance > on_curve_tolerance
            || projected_middle.distance > on_curve_tolerance
        {
            continue;
        }
        let [d0, d1] = branch.domain()?;
        let branch_span = (d1 - d0).max(1e-12);
        let (t_start, t_end) = (projected_start.u, projected_end.u);
        if (t_end - t_start).abs() <= 1e-9 * branch_span {
            continue;
        }
        let (low, high) = if t_start < t_end {
            (t_start, t_end)
        } else {
            (t_end, t_start)
        };
        let slack = 1e-6 * branch_span;
        if projected_middle.u < low - slack || projected_middle.u > high + slack {
            continue;
        }
        // Preserve the edge's start→end direction (coedge forward flags and
        // vertex bindings reference it).
        let (curve, new_t0, new_t1) = if t_start < t_end {
            (branch, t_start, t_end)
        } else {
            let reversed = branch.reversed()?;
            (reversed, d0 + d1 - t_start, d0 + d1 - t_end)
        };
        let mut record = edge.clone();
        record.curve = curve;
        record.t0 = new_t0;
        record.t1 = new_t1;
        if let Some((vertex, _)) = start_target {
            record.start_vertex_id = vertex;
        }
        if let Some((vertex, _)) = end_target {
            record.end_vertex_id = vertex;
        }
        return Ok(record);
    }
    Err(format!(
        "{op}: side edge {} cannot be extended onto the recovered corner (no closed-form \
         intersection branch of its flanking carriers reaches both endpoints — deferred)",
        edge.id
    ))
}

/// Refit the pcurves of every coedge referencing a touched (widened or newly
/// created) edge against the face's CURRENT carrier, sampling exactly the
/// edge's represented interval — so extended pcurve endpoints land on the NEW
/// edge, not the old blend rim (the closed-heal endpoint-station lesson).
/// With `only_subrange`, coedges whose edge represents its curve's full domain
/// are left alone (the affine full-curve pcurve a planar retrim just built is
/// already exact there).
pub(super) fn refit_touched_pcurves(
    face: &mut FaceRecord,
    edges: &HashMap<u64, EdgeRecord>,
    touched: &HashSet<u64>,
    only_subrange: bool,
    tolerance: f64,
    op: &str,
) -> Result<(), String> {
    let surface = face.surface.clone();
    for loop_record in &mut face.loops {
        for coedge in &mut loop_record.coedges {
            if !touched.contains(&coedge.edge_id) {
                continue;
            }
            let edge = edges
                .get(&coedge.edge_id)
                .ok_or_else(|| format!("{op}: missing edge {}", coedge.edge_id))?;
            if only_subrange {
                let [d0, d1] = edge.curve.domain()?;
                let span = (d1 - d0).max(1e-12);
                if (edge.t0 - d0).abs() <= 1e-9 * span && (edge.t1 - d1).abs() <= 1e-9 * span {
                    continue;
                }
            }
            coedge.pcurve = build_pcurve_on_surface_range(
                &surface,
                &edge.curve,
                edge.t0,
                edge.t1,
                coedge.forward,
                tolerance,
            )?;
        }
    }
    Ok(())
}
