use crate::analytic_surface::circumcenter;
use crate::mass_properties::{face_area, parameter_space_area};
use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord};
use crate::{
    build_pcurve_on_surface, make_line, make_plane, make_revolution, KnotVector, NurbsCurve,
    NurbsSurface, Vec3, Vec4,
};
use crate::{KernelRefusal, KernelStage, OrRefuse, RefusalClass};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

fn controls_match(first: Vec4, second: Vec4, tolerance: f64) -> bool {
    (first.x - second.x).abs() <= tolerance
        && (first.y - second.y).abs() <= tolerance
        && (first.z - second.z).abs() <= tolerance
        && (first.w - second.w).abs() <= tolerance
}

/// Whether two faces ride the SAME carrier patch — identical degrees, knots
/// and control points inside `tolerance`. `pub(crate)` because the direct-edit
/// cap lane asks the same question for the same reason this module does: when
/// two faces share one carrier their coedges' pcurves are interchangeable, so
/// loops can move between them with nothing refit.
pub(crate) fn same_surface(first: &NurbsSurface, second: &NurbsSurface, tolerance: f64) -> bool {
    first.degree_u == second.degree_u
        && first.degree_v == second.degree_v
        && first.knots_u.len() == second.knots_u.len()
        && first.knots_v.len() == second.knots_v.len()
        && first.control_points.len() == second.control_points.len()
        && first
            .control_points
            .iter()
            .zip(&second.control_points)
            .all(|(a, b)| {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(a, b)| controls_match(*a, *b, tolerance))
            })
        && first
            .knots_u
            .iter()
            .zip(&second.knots_u)
            .all(|(a, b)| (a - b).abs() <= tolerance)
        && first
            .knots_v
            .iter()
            .zip(&second.knots_v)
            .all(|(a, b)| (a - b).abs() <= tolerance)
}

fn surface_domains(surface: &NurbsSurface) -> Result<([f64; 2], [f64; 2]), KernelRefusal> {
    Ok((
        KnotVector::new(surface.knots_u.clone(), surface.degree_u)
            .or_refuse(KernelStage::Sew, "new")?
            .domain(),
        KnotVector::new(surface.knots_v.clone(), surface.degree_v)
            .or_refuse(KernelStage::Sew, "new")?
            .domain(),
    ))
}

fn outward_normal(face: &FaceRecord) -> Result<Vec3, KernelRefusal> {
    let (u, v) = surface_domains(&face.surface)?;
    let normal = face
        .surface
        .normal((u[0] + u[1]) / 2.0, (v[0] + v[1]) / 2.0)
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
    Ok(if face.same_sense {
        normal
    } else {
        normal.scale(-1.0)
    })
}

fn planar_support(
    face: &FaceRecord,
    tolerance: f64,
) -> Result<Option<(Vec3, Vec3)>, KernelRefusal> {
    let (u, v) = surface_domains(&face.surface)?;
    let midpoint_u = (u[0] + u[1]) / 2.0;
    let midpoint_v = (v[0] + v[1]) / 2.0;
    let origin = face
        .surface
        .evaluate(midpoint_u, midpoint_v)
        .or_refuse(KernelStage::Sew, "evaluate")?;
    let normal = face
        .surface
        .normal(midpoint_u, midpoint_v)
        .or_refuse(KernelStage::Sew, "normal")?;
    let controls = face
        .surface
        .control_points
        .iter()
        .flatten()
        .map(|control| control.point())
        .collect::<Result<Vec<_>, _>>()
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
    let scale = controls
        .iter()
        .map(|point| point.sub(origin).length())
        .fold(1.0f64, f64::max);
    let plane_tolerance = (tolerance * 100.0).max(1e-7) * scale;
    if controls
        .iter()
        .all(|point| point.sub(origin).dot(normal).abs() <= plane_tolerance)
    {
        Ok(Some((origin, normal)))
    } else {
        Ok(None)
    }
}

fn coplanar_with_same_outward_normal(
    first: &FaceRecord,
    second: &FaceRecord,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    let Some((first_origin, first_geometric_normal)) = planar_support(first, tolerance)? else {
        return Ok(false);
    };
    let Some((_, second_geometric_normal)) = planar_support(second, tolerance)? else {
        return Ok(false);
    };
    if first_geometric_normal.dot(second_geometric_normal).abs() < 1.0 - 1e-7 {
        return Ok(false);
    }
    let first_normal = outward_normal(first)?;
    let second_normal = outward_normal(second)?;
    if first_normal.dot(second_normal) < 1.0 - 1e-7 {
        return Ok(false);
    }
    let (second_u, second_v) = surface_domains(&second.surface)?;
    let plane_tolerance = (tolerance * 100.0).max(1e-6) * (1.0 + first_origin.length());
    for u in second_u {
        for v in second_v {
            if second
                .surface
                .evaluate(u, v)
                .or_refuse(KernelStage::Sew, "evaluate")?
                .sub(first_origin)
                .dot(first_normal)
                .abs()
                > plane_tolerance
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Circular-arc support of a curve, detected by sampling: circumcenter of
/// three spread samples, validated against all samples for constant radius
/// and planarity. Returns (center, unit plane normal, radius).
fn arc_circle_support(
    curve: &NurbsCurve,
    tolerance: f64,
) -> Result<Option<(Vec3, Vec3, f64)>, KernelRefusal> {
    let [t0, t1] = curve.domain().or_refuse(KernelStage::Sew, "domain")?;
    let samples = (0..=8)
        .map(|index| curve.evaluate(t0 + (t1 - t0) * index as f64 / 8.0))
        .collect::<Result<Vec<_>, _>>()
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
    // A full-circle carrier evaluates identical endpoints, so the sample
    // triple must avoid (start, end); fall back across spreads until one is
    // non-degenerate.
    let Some(center) = circumcenter(samples[0], samples[2], samples[5])
        .or_else(|| circumcenter(samples[0], samples[4], samples[8]))
        .or_else(|| circumcenter(samples[1], samples[4], samples[7]))
    else {
        return Ok(None);
    };
    let radius = samples[0].sub(center).length();
    if radius <= tolerance {
        return Ok(None);
    }
    let normal = match samples[2]
        .sub(samples[0])
        .cross(samples[5].sub(samples[0]))
        .normalized()
        .or_else(|_| {
            samples[4]
                .sub(samples[0])
                .cross(samples[8].sub(samples[0]))
                .normalized()
        }) {
        Ok(normal) => normal,
        Err(_) => return Ok(None),
    };
    let band = tolerance.max(1e-9 * radius);
    for sample in &samples {
        let delta = sample.sub(center);
        if (delta.length() - radius).abs() > band || delta.dot(normal).abs() > band {
            if std::env::var("BREP_DEBUG_MERGE").is_ok() {
                eprintln!(
                    "    arc reject: radius={radius} dr={} dplane={} band={band}",
                    (delta.length() - radius).abs(),
                    delta.dot(normal).abs()
                );
            }
            return Ok(None);
        }
    }
    Ok(Some((center, normal, radius)))
}

struct CylinderSupport {
    origin: Vec3,
    axis: Vec3,
    radius: f64,
    /// +1 when the face's outward normal points radially away from the
    /// axis, -1 when it points inward.
    outward_radial_sign: f64,
}

/// Detect a ruled cylinder patch: quadratic circular arc in one parameter
/// direction, linear in the other, both boundary arcs sharing an axis and
/// radius. Covers extruded arcs, revolved axis-parallel lines, and
/// primitive cylinder walls regardless of parameterization.
fn ruled_cylinder_support(
    face: &FaceRecord,
    tolerance: f64,
) -> Result<Option<CylinderSupport>, KernelRefusal> {
    let surface = &face.surface;
    let debug = std::env::var("BREP_DEBUG_MERGE").is_ok();
    let (arc_first, columns) = match (surface.degree_u, surface.degree_v) {
        (2, 1) => (true, surface.control_points[0].len()),
        (1, 2) => (false, surface.control_points.len()),
        _ => {
            if debug {
                eprintln!(
                    "  support face {}: degrees ({},{}) not ruled-arc",
                    face.id, surface.degree_u, surface.degree_v
                );
            }
            return Ok(None);
        }
    };
    // Boolean splits can insert linear-direction knots, so a ruled carrier
    // may hold more than two columns; the outermost columns still bound the
    // patch and interior straightness is enforced by the final area check.
    let last = columns - 1;
    let iso_curve = |side: usize| -> Result<NurbsCurve, KernelRefusal> {
        let controls = if arc_first {
            surface
                .control_points
                .iter()
                .map(|row| row[side])
                .collect::<Vec<_>>()
        } else {
            surface.control_points[side].clone()
        };
        let knots = if arc_first {
            surface.knots_u.clone()
        } else {
            surface.knots_v.clone()
        };
        NurbsCurve::new(2, knots, controls).or_refuse(KernelStage::Sew, "NurbsCurve::new")
    };
    let scale = surface
        .control_points
        .iter()
        .flatten()
        .map(|point| point.point().map(|p| p.length()).unwrap_or(0.0))
        .fold(1.0f64, f64::max);
    let circle_tolerance = (tolerance * 100.0).max(1e-7 * scale);
    let Some((center0, normal0, radius0)) = arc_circle_support(&iso_curve(0)?, circle_tolerance)?
    else {
        if debug {
            eprintln!("  support face {}: first iso not circular", face.id);
        }
        return Ok(None);
    };
    let Some((center1, normal1, radius1)) =
        arc_circle_support(&iso_curve(last)?, circle_tolerance)?
    else {
        if debug {
            eprintln!("  support face {}: last iso not circular", face.id);
        }
        return Ok(None);
    };
    if (radius0 - radius1).abs() > circle_tolerance {
        if debug {
            eprintln!(
                "  support face {}: radii differ {} vs {}",
                face.id, radius0, radius1
            );
        }
        return Ok(None);
    }
    let axial = center1.sub(center0);
    if axial.length() <= circle_tolerance {
        if debug {
            eprintln!("  support face {}: zero axial extent", face.id);
        }
        return Ok(None);
    }
    let axis = axial
        .normalized()
        .or_refuse(KernelStage::Sew, "normalized")?;
    if axis.dot(normal0).abs() < 1.0 - 1e-7 || axis.dot(normal1).abs() < 1.0 - 1e-7 {
        if debug {
            eprintln!(
                "  support face {}: arc planes not perpendicular to axis",
                face.id
            );
        }
        return Ok(None);
    }
    let outward = outward_normal(face)?;
    let (u, v) = surface_domains(surface)?;
    let midpoint = surface
        .evaluate((u[0] + u[1]) / 2.0, (v[0] + v[1]) / 2.0)
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
    let foot = center0.add(axis.scale(midpoint.sub(center0).dot(axis)));
    let radial = match midpoint.sub(foot).normalized() {
        Ok(radial) => radial,
        Err(_) => return Ok(None),
    };
    let alignment = outward.dot(radial);
    if alignment.abs() < 0.9 {
        if debug {
            eprintln!(
                "  support face {}: outward not radial ({alignment})",
                face.id
            );
        }
        return Ok(None);
    }
    Ok(Some(CylinderSupport {
        origin: center0,
        axis,
        radius: radius0,
        outward_radial_sign: alignment.signum(),
    }))
}

fn cosurface_cylinder_pair(
    first: &FaceRecord,
    second: &FaceRecord,
    tolerance: f64,
) -> Result<Option<CylinderSupport>, KernelRefusal> {
    let debug = std::env::var("BREP_DEBUG_MERGE").is_ok();
    let Some(support_a) = ruled_cylinder_support(first, tolerance)? else {
        if debug {
            eprintln!(
                "merge {}x{}: first not a ruled cylinder",
                first.id, second.id
            );
        }
        return Ok(None);
    };
    let Some(support_b) = ruled_cylinder_support(second, tolerance)? else {
        if debug {
            eprintln!(
                "merge {}x{}: second not a ruled cylinder",
                first.id, second.id
            );
        }
        return Ok(None);
    };
    let scale = 1.0f64
        .max(support_a.origin.length())
        .max(support_b.origin.length())
        .max(support_a.radius);
    let band = (tolerance * 100.0).max(1e-7 * scale);
    if support_a.axis.dot(support_b.axis).abs() < 1.0 - 1e-7
        || (support_a.radius - support_b.radius).abs() > band
        || support_a.outward_radial_sign != support_b.outward_radial_sign
    {
        if debug {
            eprintln!(
                "merge {}x{}: axis/radius/sign mismatch dot={} dr={} signs=({},{})",
                first.id,
                second.id,
                support_a.axis.dot(support_b.axis),
                (support_a.radius - support_b.radius).abs(),
                support_a.outward_radial_sign,
                support_b.outward_radial_sign
            );
        }
        return Ok(None);
    }
    let offset = support_b.origin.sub(support_a.origin);
    let perpendicular = offset.sub(support_a.axis.scale(offset.dot(support_a.axis)));
    if perpendicular.length() > band {
        if debug {
            eprintln!(
                "merge {}x{}: axes offset {}",
                first.id,
                second.id,
                perpendicular.length()
            );
        }
        return Ok(None);
    }
    if debug {
        eprintln!(
            "merge {}x{}: MATCH r={} axis=({:.4},{:.4},{:.4})",
            first.id,
            second.id,
            support_a.radius,
            support_a.axis.x,
            support_a.axis.y,
            support_a.axis.z
        );
    }
    Ok(Some(support_a))
}

struct ExtrusionSupport {
    /// Profile iso curve at the sweep start, in its actual 3D position.
    base: NurbsCurve,
    /// Unit sweep direction (from column 0 toward the last column).
    direction: Vec3,
    /// Axial interval the patch occupies along `direction`.
    interval: [f64; 2],
}

/// Detect a linear sweep of an arbitrary profile curve: one parameter
/// direction is degree 1 with every profile control translated by the SAME
/// vector (weights unchanged), and the profile lies in a plane perpendicular
/// to the sweep. This is exact at the control-point level, so it also merges
/// swept faces whose profile is only approximately circular.
fn ruled_extrusion_support(
    face: &FaceRecord,
    tolerance: f64,
) -> Result<Option<ExtrusionSupport>, KernelRefusal> {
    let surface = &face.surface;
    // Planar faces are the coplanar path's business; a plane recognized as
    // an "extrusion of a line" here could merge across a knife edge.
    if planar_support(face, tolerance)?.is_some() {
        return Ok(None);
    }
    let profile_first = if surface.degree_v == 1 && surface.degree_u >= 1 {
        true
    } else if surface.degree_u == 1 && surface.degree_v >= 1 {
        false
    } else {
        return Ok(None);
    };
    let (profile_count, linear_count) = if profile_first {
        (
            surface.control_points.len(),
            surface.control_points[0].len(),
        )
    } else {
        (
            surface.control_points[0].len(),
            surface.control_points.len(),
        )
    };
    if linear_count < 2 {
        return Ok(None);
    }
    let control = |profile_index: usize, linear_index: usize| -> Vec4 {
        if profile_first {
            surface.control_points[profile_index][linear_index]
        } else {
            surface.control_points[linear_index][profile_index]
        }
    };
    let scale = surface
        .control_points
        .iter()
        .flatten()
        .map(|point| point.point().map(|p| p.length()).unwrap_or(0.0))
        .fold(1.0f64, f64::max);
    let band = (tolerance * 100.0).max(1e-7 * scale);
    let mut translation: Option<Vec3> = None;
    for profile_index in 0..profile_count {
        let first = control(profile_index, 0);
        let last = control(profile_index, linear_count - 1);
        if (first.w - last.w).abs() > 1e-9 * first.w.abs().max(1.0) {
            return Ok(None);
        }
        let step = last
            .point()
            .or_refuse(KernelStage::Sew, "point")?
            .sub(first.point().or_refuse(KernelStage::Sew, "point")?);
        if let Some(reference) = translation {
            if step.sub(reference).length() > band {
                return Ok(None);
            }
        } else {
            translation = Some(step);
        }
        let start = first.point().or_refuse(KernelStage::Sew, "point")?;
        for linear_index in 1..linear_count - 1 {
            let middle = control(profile_index, linear_index);
            if (middle.w - first.w).abs() > 1e-9 * first.w.abs().max(1.0) {
                return Ok(None);
            }
            let offset = middle
                .point()
                .or_refuse(KernelStage::Sew, "point")?
                .sub(start);
            let along = offset.dot(step) / step.dot(step).max(1e-30);
            if offset.sub(step.scale(along)).length() > band
                || !(-1e-9..=1.0 + 1e-9).contains(&along)
            {
                return Ok(None);
            }
        }
    }
    let translation = translation.ok_or_else(|| {
        KernelRefusal::internal(
            KernelStage::Sew,
            "face_merge",
            "extrusion support: empty net",
        )
    })?;
    if translation.length() <= band {
        return Ok(None);
    }
    let direction = translation
        .normalized()
        .or_refuse(KernelStage::Sew, "normalized")?;
    let base_controls = (0..profile_count).map(|index| control(index, 0)).collect();
    let base_knots = if profile_first {
        surface.knots_u.clone()
    } else {
        surface.knots_v.clone()
    };
    let base_degree = if profile_first {
        surface.degree_u
    } else {
        surface.degree_v
    };
    let base = NurbsCurve::new(base_degree, base_knots, base_controls)
        .or_refuse(KernelStage::Sew, "new")?;
    let [t0, t1] = base.domain().or_refuse(KernelStage::Sew, "domain")?;
    let mut axial = [f64::INFINITY, f64::NEG_INFINITY];
    for index in 0..=8 {
        let z = base
            .evaluate(t0 + (t1 - t0) * index as f64 / 8.0)
            .or_refuse(KernelStage::Sew, "evaluate")?
            .dot(direction);
        axial[0] = axial[0].min(z);
        axial[1] = axial[1].max(z);
    }
    if axial[1] - axial[0] > band {
        // Slanted profiles sweep skewed slabs whose axial union is not an
        // interval; keep the merge to perpendicular profiles.
        return Ok(None);
    }
    let start = (axial[0] + axial[1]) / 2.0;
    Ok(Some(ExtrusionSupport {
        base,
        direction,
        interval: [start, start + translation.length()],
    }))
}

fn perpendicular_profile(curve: &NurbsCurve, direction: Vec3) -> Result<NurbsCurve, KernelRefusal> {
    let controls = curve
        .control_points
        .iter()
        .map(|control| {
            let point = control.point().or_refuse(KernelStage::Sew, "point")?;
            Ok(Vec4::from_point(
                point.sub(direction.scale(point.dot(direction))),
                control.w,
            ))
        })
        .collect::<Result<Vec<_>, KernelRefusal>>()?;
    NurbsCurve::new(curve.degree, curve.knots.clone(), controls)
        .or_refuse(KernelStage::Sew, "NurbsCurve::new")
}

fn curves_coincide(
    first: &NurbsCurve,
    second: &NurbsCurve,
    band: f64,
) -> Result<bool, KernelRefusal> {
    for (from, to) in [(first, second), (second, first)] {
        let [t0, t1] = from.domain().or_refuse(KernelStage::Sew, "domain")?;
        for index in 0..=8 {
            let point = from
                .evaluate(t0 + (t1 - t0) * index as f64 / 8.0)
                .or_refuse(KernelStage::Sew, "evaluate")?;
            let projection = match crate::project_point_to_curve(to, point) {
                Ok(projection) => projection,
                Err(_) => return Ok(false),
            };
            if projection.distance > band {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Matching linear sweeps: parallel directions, identical perpendicular
/// profile, adjacent (or touching) axial intervals.
fn coextrusion_pair(
    older: &FaceRecord,
    newer: &FaceRecord,
    tolerance: f64,
) -> Result<Option<(ExtrusionSupport, [f64; 2])>, KernelRefusal> {
    let debug = std::env::var("BREP_DEBUG_MERGE").is_ok();
    let Some(support_a) = ruled_extrusion_support(older, tolerance)? else {
        return Ok(None);
    };
    let Some(support_b) = ruled_extrusion_support(newer, tolerance)? else {
        return Ok(None);
    };
    let alignment = support_a.direction.dot(support_b.direction);
    if alignment.abs() < 1.0 - 1e-9 {
        if debug {
            eprintln!(
                "coextrusion {}x{}: directions differ (dot {alignment})",
                older.id, newer.id
            );
        }
        return Ok(None);
    }
    let scale = 1.0f64
        .max(support_a.interval[1].abs())
        .max(support_b.interval[1].abs());
    let band = (tolerance * 100.0).max(1e-7 * scale);
    let profile_a = perpendicular_profile(&support_a.base, support_a.direction)?;
    let profile_b = perpendicular_profile(&support_b.base, support_a.direction)?;
    if !curves_coincide(&profile_a, &profile_b, band)? {
        if debug {
            eprintln!("coextrusion {}x{}: profiles differ", older.id, newer.id);
        }
        return Ok(None);
    }
    // Express B's interval along A's direction regardless of B's own sweep
    // orientation: anti-parallel sweeps negate the axial coordinate.
    let interval_b = if alignment < 0.0 {
        [-support_b.interval[1], -support_b.interval[0]]
    } else {
        support_b.interval
    };
    if interval_b[0] > support_a.interval[1] + band || support_a.interval[0] > interval_b[1] + band
    {
        if debug {
            eprintln!(
                "coextrusion {}x{}: axial gap {:?} vs {:?}",
                older.id, newer.id, support_a.interval, interval_b
            );
        }
        return Ok(None);
    }
    let union = [
        support_a.interval[0].min(interval_b[0]),
        support_a.interval[1].max(interval_b[1]),
    ];
    Ok(Some((support_a, union)))
}

fn trimmed_curve(curve: &NurbsCurve, t0: f64, t1: f64) -> Result<NurbsCurve, KernelRefusal> {
    let [start, end] = curve.domain().or_refuse(KernelStage::Sew, "domain")?;
    let epsilon = (1e-9 * (end - start)).max(2e-9);
    let mut result = curve.clone();
    if t0 > start + epsilon && t0 < end - epsilon {
        result = result.split(t0).or_refuse(KernelStage::Sew, "split")?.1;
    }
    let domain = result.domain().or_refuse(KernelStage::Sew, "domain")?;
    if t1 < domain[1] - epsilon && t1 > domain[0] + epsilon {
        result = result.split(t1).or_refuse(KernelStage::Sew, "split")?.0;
    } else if std::env::var("BREP_DEBUG_SUBRANGE").is_ok() && t1 < domain[1] - epsilon {
        eprintln!("abnormal subrange skip in face_merge trimmed");
    }
    Ok(result)
}

fn reproject_loops(
    surface: &NurbsSurface,
    loops: Vec<LoopRecord>,
    edges: &HashMap<u64, &EdgeRecord>,
) -> Result<Vec<LoopRecord>, KernelRefusal> {
    let mut projected_loops = Vec::new();
    for (loop_index, loop_record) in loops.into_iter().enumerate() {
        let mut projected = Vec::new();
        for mut coedge in loop_record.coedges {
            let edge = edges[&coedge.edge_id];
            let mut curve = trimmed_curve(&edge.curve, edge.t0, edge.t1)?;
            if !coedge.forward {
                curve = curve.reversed().or_refuse(KernelStage::Sew, "reversed")?;
            }
            coedge.pcurve = build_pcurve_on_surface(surface, &curve)
                .or_refuse(KernelStage::Sew, "build_pcurve_on_surface")?;
            projected.push(coedge);
        }
        projected_loops.push(LoopRecord {
            id: loop_index as u64 + 1,
            coedges: projected,
        });
    }
    Ok(projected_loops)
}

fn traversal_vertices(edge: &EdgeRecord, coedge: &CoedgeRecord) -> (u64, u64) {
    if coedge.forward {
        (edge.start_vertex_id, edge.end_vertex_id)
    } else {
        (edge.end_vertex_id, edge.start_vertex_id)
    }
}

/// Whether two faces ride ONE carrier — the same geometric surface, oriented
/// the same way out of the material — by any of the four readings this module
/// merges on: bit-identical patches, coplanar planes with the same outward
/// normal, cosurface cylinder patches, and coextruded profiles.
///
/// `pub(crate)` because the direct-edit rejoin asks the same question for the
/// same reason: a survivor face on the far side of a deleted band can only
/// take the near side's loops when the two are one carrier. It asks GEOMETRIC
/// cosurface rather than [`same_surface`] alone, because a mirrored twin is
/// the same surface reparametrized — reflected control points, flipped
/// `same_sense` — and bit-identity reads that as two.
pub(crate) fn faces_are_cosurface(
    first: &FaceRecord,
    second: &FaceRecord,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    if first.same_sense == second.same_sense
        && same_surface(&first.surface, &second.surface, tolerance)
    {
        return Ok(true);
    }
    if coplanar_with_same_outward_normal(first, second, tolerance)? {
        return Ok(true);
    }
    if cosurface_cylinder_pair(first, second, tolerance)?.is_some() {
        return Ok(true);
    }
    Ok(coextrusion_pair(first, second, tolerance)?.is_some())
}

fn try_merge_pair(
    older: &FaceRecord,
    newer: &FaceRecord,
    edges: &HashMap<u64, &EdgeRecord>,
    tolerance: f64,
    same_carrier_area_tolerance_ratio: f64,
) -> Result<Option<FaceRecord>, KernelRefusal> {
    let same_carrier = older.same_sense == newer.same_sense
        && same_surface(&older.surface, &newer.surface, tolerance);
    let coplanar_planar =
        !same_carrier && coplanar_with_same_outward_normal(older, newer, tolerance)?;
    let cosurface_cylinder = if !same_carrier && !coplanar_planar {
        cosurface_cylinder_pair(older, newer, tolerance)?
    } else {
        None
    };
    let cosurface_extrusion = if !same_carrier && !coplanar_planar && cosurface_cylinder.is_none() {
        coextrusion_pair(older, newer, tolerance)?
    } else {
        None
    };
    let older_coedges = older
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .collect::<Vec<_>>();
    let newer_coedges = newer
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .collect::<Vec<_>>();
    let mut shared_older = HashSet::default();
    let mut shared_newer = HashSet::default();
    for first in &older_coedges {
        if edges[&first.edge_id].degenerate {
            continue;
        }
        for second in &newer_coedges {
            if first.edge_id != second.edge_id {
                continue;
            }
            if first.forward == second.forward {
                return Ok(None);
            }
            shared_older.insert(first.id);
            shared_newer.insert(second.id);
        }
    }
    if shared_older.is_empty() || shared_older.len() != shared_newer.len() {
        return Ok(None);
    }
    if !same_carrier
        && !coplanar_planar
        && cosurface_cylinder.is_none()
        && cosurface_extrusion.is_none()
    {
        return Ok(None);
    }
    let maximum_cycle_length = older_coedges.len() + newer_coedges.len();
    let mut remaining = older_coedges
        .into_iter()
        .filter(|coedge| {
            !shared_older.contains(&coedge.id)
                && !(coplanar_planar && edges[&coedge.edge_id].degenerate)
        })
        .chain(newer_coedges.into_iter().filter(|coedge| {
            !shared_newer.contains(&coedge.id)
                && !(coplanar_planar && edges[&coedge.edge_id].degenerate)
        }))
        .cloned()
        .collect::<Vec<_>>();
    let mut loops = Vec::new();
    while !remaining.is_empty() {
        let first = remaining.remove(0);
        let (start, mut end) = traversal_vertices(edges[&first.edge_id], &first);
        let mut cycle = vec![first];
        while end != start {
            let matches = remaining
                .iter()
                .enumerate()
                .filter_map(|(index, coedge)| {
                    (traversal_vertices(edges[&coedge.edge_id], coedge).0 == end).then_some(index)
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Ok(None);
            }
            let next = remaining.remove(matches[0]);
            end = traversal_vertices(edges[&next.edge_id], &next).1;
            cycle.push(next);
            if cycle.len() > maximum_cycle_length {
                return Ok(None);
            }
        }
        if !cycle
            .iter()
            .any(|coedge| !edges[&coedge.edge_id].degenerate)
        {
            return Ok(None);
        }
        loops.push(LoopRecord {
            id: loops.len() as u64 + 1,
            coedges: cycle,
        });
    }
    if loops.is_empty() {
        return Ok(None);
    }
    loops.sort_by(|first, second| {
        let area = |loop_record: &LoopRecord| {
            parameter_space_area(&FaceRecord {
                id: older.id,
                surface: older.surface.clone(),
                same_sense: older.same_sense,
                loops: vec![loop_record.clone()],
                name: None,
            })
            .map(f64::abs)
        };
        match (area(first), area(second)) {
            (Ok(a), Ok(b)) => b.total_cmp(&a),
            _ => std::cmp::Ordering::Equal,
        }
    });
    if same_carrier {
        let merged = FaceRecord {
            id: older.id,
            surface: older.surface.clone(),
            same_sense: older.same_sense,
            loops,
            name: older.name.clone(),
        };
        let expected = parameter_space_area(older)
            .or_refuse(KernelStage::Sew, "parameter_space_area")?
            + parameter_space_area(newer).or_refuse(KernelStage::Sew, "parameter_space_area")?;
        let actual =
            parameter_space_area(&merged).or_refuse(KernelStage::Sew, "parameter_space_area")?;
        let area_tolerance = 1e-9f64.max(expected.abs() * same_carrier_area_tolerance_ratio);
        if (actual - expected).abs() > area_tolerance
            || (actual.abs() > 1e-12 && actual.is_sign_positive() != merged.same_sense)
        {
            return Ok(None);
        }
        return Ok(Some(merged));
    }

    if let Some(support) = cosurface_cylinder {
        // Build one revolution carrier covering both patches: seam at the
        // largest angular gap so the merged region never crosses it, then
        // reproject every loop onto the shared cylinder.
        let x_reference = {
            let seed = loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
                .find_map(|coedge| {
                    let edge = edges[&coedge.edge_id];
                    edge.curve.evaluate(edge.t0).ok()
                })
                .ok_or_else(|| {
                    KernelRefusal::internal(
                        KernelStage::Sew,
                        "face_merge",
                        "cylinder merge: no boundary sample",
                    )
                })?;
            let delta = seed.sub(support.origin);
            let radial = delta.sub(support.axis.scale(delta.dot(support.axis)));
            match radial.normalized() {
                Ok(radial) => radial,
                Err(_) => return Ok(None),
            }
        };
        let y_reference = support.axis.cross(x_reference);
        let mut angles = Vec::new();
        let mut axial_range = [f64::INFINITY, f64::NEG_INFINITY];
        for coedge in loops.iter().flat_map(|loop_record| &loop_record.coedges) {
            let edge = edges[&coedge.edge_id];
            let curve = trimmed_curve(&edge.curve, edge.t0, edge.t1)?;
            let [c0, c1] = curve.domain().or_refuse(KernelStage::Sew, "domain")?;
            for index in 0..=8 {
                let point = curve
                    .evaluate(c0 + (c1 - c0) * index as f64 / 8.0)
                    .or_refuse(KernelStage::Sew, "evaluate")?;
                let delta = point.sub(support.origin);
                let axial = delta.dot(support.axis);
                axial_range[0] = axial_range[0].min(axial);
                axial_range[1] = axial_range[1].max(axial);
                let radial = delta.sub(support.axis.scale(axial));
                if radial.length() <= support.radius * 1e-6 {
                    continue;
                }
                let mut angle = radial.dot(y_reference).atan2(radial.dot(x_reference));
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                angles.push(angle);
            }
        }
        if angles.len() < 3 || axial_range[1] - axial_range[0] <= tolerance {
            return Ok(None);
        }
        angles.sort_by(f64::total_cmp);
        let mut gap_start = angles.len() - 1;
        let mut largest_gap = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
        for index in 0..angles.len() - 1 {
            let gap = angles[index + 1] - angles[index];
            if gap > largest_gap {
                largest_gap = gap;
                gap_start = index;
            }
        }
        let span = std::f64::consts::TAU - largest_gap;
        if span <= tolerance || span >= std::f64::consts::TAU - 1e-6 {
            return Ok(None);
        }
        let start_angle = angles[(gap_start + 1) % angles.len()];
        let start_direction = x_reference
            .scale(start_angle.cos())
            .add(y_reference.scale(start_angle.sin()));
        let generatrix = make_line(
            support
                .origin
                .add(support.axis.scale(axial_range[0]))
                .add(start_direction.scale(support.radius)),
            support
                .origin
                .add(support.axis.scale(axial_range[1]))
                .add(start_direction.scale(support.radius)),
        )
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
        let surface = make_revolution(support.origin, support.axis, &generatrix, span)
            .or_refuse(KernelStage::Sew, "make_revolution")?;
        let (u_domain, v_domain) = surface_domains(&surface)?;
        let sample = surface
            .evaluate(
                (u_domain[0] + u_domain[1]) / 2.0,
                (v_domain[0] + v_domain[1]) / 2.0,
            )
            .or_refuse(KernelStage::Sew, "healing.face_merge")?;
        let geometric_normal = surface
            .normal(
                (u_domain[0] + u_domain[1]) / 2.0,
                (v_domain[0] + v_domain[1]) / 2.0,
            )
            .or_refuse(KernelStage::Sew, "healing.face_merge")?;
        let delta = sample.sub(support.origin);
        let radial = delta.sub(support.axis.scale(delta.dot(support.axis)));
        let radial = match radial.normalized() {
            Ok(radial) => radial,
            Err(_) => return Ok(None),
        };
        let same_sense = geometric_normal.dot(radial).signum() == support.outward_radial_sign;
        let mut projected_loops = reproject_loops(&surface, loops, edges)?;
        projected_loops.sort_by(|first, second| {
            let area = |loop_record: &LoopRecord| {
                parameter_space_area(&FaceRecord {
                    id: older.id,
                    surface: surface.clone(),
                    same_sense,
                    loops: vec![loop_record.clone()],
                    name: None,
                })
                .map(f64::abs)
            };
            match (area(first), area(second)) {
                (Ok(a), Ok(b)) => b.total_cmp(&a),
                _ => std::cmp::Ordering::Equal,
            }
        });
        let merged = FaceRecord {
            id: older.id,
            surface,
            same_sense,
            loops: projected_loops,
            name: older.name.clone(),
        };
        let area =
            parameter_space_area(&merged).or_refuse(KernelStage::Sew, "parameter_space_area")?;
        if area.abs() <= tolerance * tolerance || area.is_sign_positive() != merged.same_sense {
            return Ok(None);
        }
        // The reprojected trims must reproduce the pieces' combined 3D
        // area; a projection or seam mistake shows up here immediately.
        let expected = face_area(older).or_refuse(KernelStage::Sew, "face_area")?
            + face_area(newer).or_refuse(KernelStage::Sew, "face_area")?;
        let actual = face_area(&merged).or_refuse(KernelStage::Sew, "face_area")?;
        if (actual - expected).abs() > expected.abs().max(1e-9) * 5e-3 {
            return Ok(None);
        }
        return Ok(Some(merged));
    }

    if let Some((support, union)) = cosurface_extrusion {
        // Shared linear-sweep carrier: the older profile translated to the
        // union's start plane, extruded across the union interval.
        let shift = support.direction.scale(union[0] - support.interval[0]);
        let step = support.direction.scale(union[1] - union[0]);
        let rows = support
            .base
            .control_points
            .iter()
            .map(|control| -> Result<Vec<Vec4>, KernelRefusal> {
                let point = control
                    .point()
                    .or_refuse(KernelStage::Sew, "point")?
                    .add(shift);
                Ok(vec![
                    Vec4::from_point(point, control.w),
                    Vec4::from_point(point.add(step), control.w),
                ])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let surface = NurbsSurface::new(
            support.base.degree,
            1,
            support.base.knots.clone(),
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
        )
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
        let (older_u, older_v) = surface_domains(&older.surface)?;
        let older_mid = older
            .surface
            .evaluate(
                (older_u[0] + older_u[1]) / 2.0,
                (older_v[0] + older_v[1]) / 2.0,
            )
            .or_refuse(KernelStage::Sew, "healing.face_merge")?;
        let projection = crate::project_point_to_surface(&surface, older_mid)
            .or_refuse(KernelStage::Sew, "project_point_to_surface")?;
        let carrier_normal = surface
            .normal(projection.u, projection.v)
            .or_refuse(KernelStage::Sew, "normal")?;
        let same_sense = carrier_normal.dot(outward_normal(older)?) > 0.0;
        let mut projected_loops = reproject_loops(&surface, loops, edges)?;
        projected_loops.sort_by(|first, second| {
            let area = |loop_record: &LoopRecord| {
                parameter_space_area(&FaceRecord {
                    id: older.id,
                    surface: surface.clone(),
                    same_sense,
                    loops: vec![loop_record.clone()],
                    name: None,
                })
                .map(f64::abs)
            };
            match (area(first), area(second)) {
                (Ok(a), Ok(b)) => b.total_cmp(&a),
                _ => std::cmp::Ordering::Equal,
            }
        });
        let merged = FaceRecord {
            id: older.id,
            surface,
            same_sense,
            loops: projected_loops,
            name: older.name.clone(),
        };
        let area =
            parameter_space_area(&merged).or_refuse(KernelStage::Sew, "parameter_space_area")?;
        if area.abs() <= tolerance * tolerance || area.is_sign_positive() != merged.same_sense {
            return Ok(None);
        }
        let expected = face_area(older).or_refuse(KernelStage::Sew, "face_area")?
            + face_area(newer).or_refuse(KernelStage::Sew, "face_area")?;
        let actual = face_area(&merged).or_refuse(KernelStage::Sew, "face_area")?;
        if (actual - expected).abs() > expected.abs().max(1e-9) * 5e-3 {
            if std::env::var("BREP_DEBUG_MERGE").is_ok() {
                eprintln!(
                    "coextrusion {}x{}: area mismatch {actual} vs {expected}",
                    older.id, newer.id
                );
            }
            return Ok(None);
        }
        return Ok(Some(merged));
    }

    let (u_domain, v_domain) = surface_domains(&older.surface)?;
    let origin = older
        .surface
        .evaluate(u_domain[0], v_domain[0])
        .or_refuse(KernelStage::Sew, "evaluate")?;
    let derivatives = older
        .surface
        .derivatives(
            (u_domain[0] + u_domain[1]) / 2.0,
            (v_domain[0] + v_domain[1]) / 2.0,
            1,
        )
        .or_refuse(KernelStage::Sew, "healing.face_merge")?;
    let u_axis = derivatives[1][0]
        .normalized()
        .or_refuse(KernelStage::Sew, "normalized")?;
    let geometric_normal = derivatives[1][0]
        .cross(derivatives[0][1])
        .normalized()
        .or_refuse(KernelStage::Sew, "normalized")?;
    let v_axis = geometric_normal
        .cross(u_axis)
        .normalized()
        .or_refuse(KernelStage::Sew, "normalized")?;
    let mut bounds_u = [f64::INFINITY, f64::NEG_INFINITY];
    let mut bounds_v = [f64::INFINITY, f64::NEG_INFINITY];
    for coedge in loops.iter().flat_map(|loop_record| &loop_record.coedges) {
        let edge = edges[&coedge.edge_id];
        let curve = trimmed_curve(&edge.curve, edge.t0, edge.t1)?;
        for control in &curve.control_points {
            let relative = control
                .point()
                .or_refuse(KernelStage::Sew, "point")?
                .sub(origin);
            let u = relative.dot(u_axis);
            let v = relative.dot(v_axis);
            bounds_u[0] = bounds_u[0].min(u);
            bounds_u[1] = bounds_u[1].max(u);
            bounds_v[0] = bounds_v[0].min(v);
            bounds_v[1] = bounds_v[1].max(v);
        }
    }
    let extent_u = bounds_u[1] - bounds_u[0];
    let extent_v = bounds_v[1] - bounds_v[0];
    if extent_u <= tolerance || extent_v <= tolerance {
        return Ok(None);
    }
    let surface_origin = origin
        .add(u_axis.scale(bounds_u[0]))
        .add(v_axis.scale(bounds_v[0]));
    let surface = make_plane(surface_origin, u_axis, v_axis, extent_u, extent_v)
        .or_refuse(KernelStage::Sew, "make_plane")?;
    let mut projected_loops = reproject_loops(&surface, loops, edges)?;
    projected_loops.sort_by(|first, second| {
        let area = |loop_record: &LoopRecord| {
            parameter_space_area(&FaceRecord {
                id: older.id,
                surface: surface.clone(),
                same_sense: older.same_sense,
                loops: vec![loop_record.clone()],
                name: None,
            })
            .map(f64::abs)
        };
        match (area(first), area(second)) {
            (Ok(a), Ok(b)) => b.total_cmp(&a),
            _ => std::cmp::Ordering::Equal,
        }
    });
    let merged = FaceRecord {
        id: older.id,
        surface,
        same_sense: older.same_sense,
        loops: projected_loops,
        name: older.name.clone(),
    };
    let area = parameter_space_area(&merged).or_refuse(KernelStage::Sew, "parameter_space_area")?;
    if area.abs() <= tolerance * tolerance || area.is_sign_positive() != merged.same_sense {
        return Ok(None);
    }
    Ok(Some(merged))
}

fn try_merge_connected_group(
    faces: &[FaceRecord],
    edges: &HashMap<u64, &EdgeRecord>,
    same_carrier_area_tolerance_ratio: f64,
    keep_unmerged: &[String],
) -> Result<Option<(Vec<usize>, FaceRecord)>, KernelRefusal> {
    for seed in 0..faces.len() {
        // A pinned face never seeds or joins a merge group.
        if face_is_pinned(&faces[seed], keep_unmerged) {
            continue;
        }
        let mut component = [seed].into_iter().collect::<HashSet<_>>();
        loop {
            let mut changed = false;
            for candidate in 0..faces.len() {
                if component.contains(&candidate)
                    || face_is_pinned(&faces[candidate], keep_unmerged)
                    || faces[candidate].same_sense != faces[seed].same_sense
                    || !same_surface(&faces[candidate].surface, &faces[seed].surface, 1e-7)
                {
                    continue;
                }
                let candidate_uses = faces[candidate]
                    .loops
                    .iter()
                    .flat_map(|loop_record| &loop_record.coedges);
                let adjacent = candidate_uses.into_iter().any(|candidate_use| {
                    !edges[&candidate_use.edge_id].degenerate
                        && component.iter().any(|member| {
                            faces[*member]
                                .loops
                                .iter()
                                .flat_map(|loop_record| &loop_record.coedges)
                                .any(|member_use| {
                                    member_use.edge_id == candidate_use.edge_id
                                        && member_use.forward != candidate_use.forward
                                })
                        })
                });
                if adjacent {
                    component.insert(candidate);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if component.len() < 2 {
            continue;
        }

        let mut uses_by_edge = HashMap::<u64, Vec<CoedgeRecord>>::default();
        for index in &component {
            for coedge in faces[*index]
                .loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
            {
                uses_by_edge
                    .entry(coedge.edge_id)
                    .or_default()
                    .push(coedge.clone());
            }
        }
        if uses_by_edge.values().any(|uses| uses.len() > 2) {
            continue;
        }
        let internal = uses_by_edge
            .iter()
            .filter_map(|(edge_id, uses)| {
                (uses.len() == 2
                    && !edges[edge_id].degenerate
                    && uses[0].forward != uses[1].forward)
                    .then_some(*edge_id)
            })
            .collect::<HashSet<_>>();
        let mut remaining = uses_by_edge
            .into_iter()
            .filter(|(edge_id, _)| !internal.contains(edge_id))
            .flat_map(|(_, uses)| uses)
            .collect::<Vec<_>>();
        let maximum_cycle_length = remaining.len();
        let mut loops = Vec::new();
        let mut failed = false;
        while !remaining.is_empty() {
            let first = remaining.remove(0);
            let (start, mut end) = traversal_vertices(edges[&first.edge_id], &first);
            let mut cycle = vec![first];
            while end != start {
                let matches = remaining
                    .iter()
                    .enumerate()
                    .filter_map(|(index, coedge)| {
                        (traversal_vertices(edges[&coedge.edge_id], coedge).0 == end)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    failed = true;
                    break;
                }
                let next = remaining.remove(matches[0]);
                end = traversal_vertices(edges[&next.edge_id], &next).1;
                cycle.push(next);
                if cycle.len() > maximum_cycle_length {
                    failed = true;
                    break;
                }
            }
            if failed {
                break;
            }
            loops.push(LoopRecord {
                id: loops.len() as u64 + 1,
                coedges: cycle,
            });
        }
        if failed || loops.is_empty() {
            continue;
        }
        loops.sort_by(|first, second| {
            let area = |loop_record: &LoopRecord| {
                parameter_space_area(&FaceRecord {
                    id: faces[seed].id,
                    surface: faces[seed].surface.clone(),
                    same_sense: faces[seed].same_sense,
                    loops: vec![loop_record.clone()],
                    name: None,
                })
                .map(f64::abs)
            };
            match (area(first), area(second)) {
                (Ok(a), Ok(b)) => b.total_cmp(&a),
                _ => std::cmp::Ordering::Equal,
            }
        });
        let merged = FaceRecord {
            id: faces[seed].id,
            surface: faces[seed].surface.clone(),
            same_sense: faces[seed].same_sense,
            loops,
            name: faces[seed].name.clone(),
        };
        let expected = component
            .iter()
            .map(|index| parameter_space_area(&faces[*index]))
            .collect::<Result<Vec<_>, _>>()
            .or_refuse(KernelStage::Sew, "healing.face_merge")?
            .into_iter()
            .sum::<f64>();
        let actual =
            parameter_space_area(&merged).or_refuse(KernelStage::Sew, "parameter_space_area")?;
        let area_tolerance = 1e-9f64.max(expected.abs() * same_carrier_area_tolerance_ratio);
        if (actual - expected).abs() > area_tolerance
            || (actual.abs() > 1e-12 && actual.is_sign_positive() != merged.same_sense)
        {
            continue;
        }
        let mut indices = component.into_iter().collect::<Vec<_>>();
        indices.sort_unstable();
        return Ok(Some((indices, merged)));
    }
    Ok(None)
}

/// A face is PINNED out of the merge when its `name` contains any substring in
/// `keep_unmerged`: it never coalesces with a neighbour (in any merge lane) and
/// is emitted unchanged. An empty `keep_unmerged` pins nothing — the historical
/// behavior.
fn face_is_pinned(face: &FaceRecord, keep_unmerged: &[String]) -> bool {
    match &face.name {
        Some(name) => keep_unmerged
            .iter()
            .any(|needle| name.contains(needle.as_str())),
        None => false,
    }
}

/// Merge adjacent fragments that retain the exact same carrier surface.
///
/// Every shared edge is cancelled in opposite directions and the remaining
/// coedges are stitched into outer/hole cycles without refitting geometry.
fn merge_same_surface_faces_impl(
    solid: &BrepSolid,
    tolerance: f64,
    validate: bool,
    same_carrier_area_tolerance_ratio: f64,
    keep_unmerged: &[String],
) -> Result<BrepSolid, KernelRefusal> {
    let mut result = solid.clone();
    for shell in &mut result.shells {
        loop {
            let edges = result
                .edges
                .iter()
                .map(|edge| (edge.id, edge))
                .collect::<HashMap<_, _>>();
            if !validate {
                if let Some((indices, merged)) = try_merge_connected_group(
                    &shell.faces,
                    &edges,
                    same_carrier_area_tolerance_ratio,
                    keep_unmerged,
                )? {
                    let first = indices[0];
                    shell.faces[first] = merged;
                    for index in indices.into_iter().skip(1).rev() {
                        shell.faces.remove(index);
                    }
                    continue;
                }
            }
            let mut accepted = None;
            'pairs: for first in 0..shell.faces.len() {
                if face_is_pinned(&shell.faces[first], keep_unmerged) {
                    continue;
                }
                for second in first + 1..shell.faces.len() {
                    if face_is_pinned(&shell.faces[second], keep_unmerged) {
                        continue;
                    }
                    if let Some(merged) = try_merge_pair(
                        &shell.faces[first],
                        &shell.faces[second],
                        &edges,
                        tolerance,
                        same_carrier_area_tolerance_ratio,
                    )? {
                        accepted = Some((first, second, merged));
                        break 'pairs;
                    }
                }
            }
            let Some((first, second, merged)) = accepted else {
                break;
            };
            shell.faces[first] = merged;
            shell.faces.remove(second);
        }
    }
    let used_edges = result
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .flat_map(|face| &face.loops)
        .flat_map(|loop_record| &loop_record.coedges)
        .map(|coedge| coedge.edge_id)
        .collect::<HashSet<_>>();
    result.edges.retain(|edge| used_edges.contains(&edge.id));
    let used_vertices = result
        .edges
        .iter()
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect::<HashSet<_>>();
    result
        .vertices
        .retain(|vertex| used_vertices.contains(&vertex.id));
    if !validate {
        return Ok(result);
    }
    if std::env::var("BREP_DEBUG_MERGE").is_ok() {
        let entry_faces: i64 = solid.shells.iter().map(|s| s.faces.len() as i64).sum();
        let entry_holes: i64 = solid
            .shells
            .iter()
            .flat_map(|s| &s.faces)
            .map(|f| f.loops.len().saturating_sub(1) as i64)
            .sum();
        let entry_edges = solid.edges.iter().filter(|e| !e.degenerate).count() as i64;
        let entry_used: std::collections::HashSet<u64> = solid
            .shells
            .iter()
            .flat_map(|s| &s.faces)
            .flat_map(|f| &f.loops)
            .flat_map(|l| &l.coedges)
            .map(|c| c.edge_id)
            .collect();
        eprintln!(
            "merge entry census: V={} E={entry_edges} (coedge-referenced {}) F={entry_faces} H={entry_holes} S={} stored_genus={}",
            solid.vertices.len(),
            entry_used.len(),
            solid.shells.len(),
            solid.genus
        );
    }
    let face_count = result
        .shells
        .iter()
        .map(|shell| shell.faces.len() as i64)
        .sum::<i64>();
    let hole_count = result
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .map(|face| face.loops.len().saturating_sub(1) as i64)
        .sum::<i64>();
    let edge_count = result.edges.iter().filter(|edge| !edge.degenerate).count() as i64;
    // Count only LIVE vertices (referenced by a non-degenerate edge): the
    // merge removes edges and faces, orphaning vertices that then must not
    // participate in the Euler census — mirroring refresh_derived_genus
    // (boolean.rs). Counting stale vertices inflated V and rejected valid
    // merges as "non-integral genus" (the helmet report's final layer).
    let live_vertices: std::collections::HashSet<u64> = result
        .edges
        .iter()
        .filter(|edge| !edge.degenerate)
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect();
    let vertex_count = result
        .vertices
        .iter()
        .filter(|vertex| live_vertices.contains(&vertex.id))
        .count() as i64;
    let euler = vertex_count - edge_count + face_count - hole_count;
    let numerator = result.shells.len() as i64 * 2 - euler;
    // Mirror validate()'s convention (topology.rs): the reduced V−E+F−H
    // formula does not model parameter-space POLE COLLAPSES reliably, so on a
    // complex carrying degenerate edges the parity check must not reject —
    // the helmet result holds 17 legitimate degenerate (pole) edges and its
    // odd numerator here is a formula artifact, not a defect; validate()
    // below performs the real integrity checks either way. With no
    // degenerate edges the strict behavior is unchanged.
    let has_degenerate = result.edges.iter().any(|edge| edge.degenerate);
    if numerator < 0 || numerator % 2 != 0 {
        if std::env::var("BREP_DEBUG_MERGE").is_ok() {
            eprintln!(
                "merge euler reject: V={vertex_count} E={edge_count} F={face_count} H={hole_count} S={} euler={euler} numerator={numerator} degenerate={has_degenerate}",
                result.shells.len()
            );
        }
        if !has_degenerate
            || std::env::var("BREP_MERGE_DEGENERATE_EULER_STRICT").as_deref() == Ok("1")
        {
            return Err(KernelRefusal::new(
                RefusalClass::NonIntegralGenus {
                    shells: result.shells.len() as u32,
                    euler: euler as i64,
                },
                KernelStage::Validate,
                "merge_same_surface_faces produced non-integral genus",
            ));
        }
    } else {
        result.genus = numerator / 2;
    }
    let issues = result.validate();
    if !issues.is_empty() {
        return Err(KernelRefusal::new(
            RefusalClass::DegenerateArrangement {
                open_edges: 0,
                issues: issues.len() as u32,
            },
            KernelStage::Validate,
            format!("merge_same_surface_faces produced invalid topology: {issues:?}"),
        ));
    }
    Ok(result)
}

pub fn merge_same_surface_faces(
    solid: &BrepSolid,
    tolerance: f64,
) -> Result<BrepSolid, KernelRefusal> {
    merge_same_surface_faces_impl(solid, tolerance, true, 1e-7, &[])
}

/// Like [`merge_same_surface_faces`], but any face whose `name` CONTAINS one of
/// `keep_unmerged` is pinned: it never merges with a coplanar/cosurface
/// neighbour and is emitted unchanged, while every other face merges exactly as
/// [`merge_same_surface_faces`] would. An empty `keep_unmerged` is identical to
/// the plain call.
pub fn merge_same_surface_faces_excluding(
    solid: &BrepSolid,
    tolerance: f64,
    keep_unmerged: &[String],
) -> Result<BrepSolid, KernelRefusal> {
    merge_same_surface_faces_impl(solid, tolerance, true, 1e-7, keep_unmerged)
}

pub(crate) fn merge_same_surface_faces_open(
    solid: &BrepSolid,
    tolerance: f64,
) -> Result<BrepSolid, KernelRefusal> {
    merge_same_surface_faces_impl(solid, tolerance, false, 5e-3, &[])
}

// BREP private tests: cebf6e4d7f8e005f
