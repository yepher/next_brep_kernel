use super::*;
use crate::{KernelRefusal, KernelStage, OrRefuse};

/// Split every iso-line of `surface` along one direction and keep the range
/// [range[0], range[1]]. Knot insertion is exact, so the result is the same
/// geometry restricted to the range and its control hull bounds it.
pub(super) fn restrict_direction(
    surface: &NurbsSurface,
    range: [f64; 2],
    direction_u: bool,
) -> Result<NurbsSurface, KernelRefusal> {
    let (knots, degree, domain) = if direction_u {
        (
            &surface.knots_u,
            surface.degree_u,
            surface
                .domain_u()
                .or_refuse(KernelStage::Intersect, "domain_u")?,
        )
    } else {
        (
            &surface.knots_v,
            surface.degree_v,
            surface
                .domain_v()
                .or_refuse(KernelStage::Intersect, "domain_v")?,
        )
    };
    // NurbsCurve::split refuses parameters within its absolute knot identity
    // tolerance of the domain ends (same guard as edge_subcurve). Mirror the
    // single source (curve.rs KNOT_IDENTITY_TOL) rather than a bare literal.
    let guard = ((domain[1] - domain[0]) * KNOT_IDENTITY_TOL).max(2e-9);
    let mut curves: Vec<NurbsCurve> = if direction_u {
        (0..surface.control_points[0].len())
            .map(|column| {
                NurbsCurve::new(
                    degree,
                    knots.clone(),
                    surface
                        .control_points
                        .iter()
                        .map(|row| row[column])
                        .collect(),
                )
            })
            .collect::<Result<_, _>>()
            .or_refuse(KernelStage::Intersect, "csg.imprint.support")?
    } else {
        surface
            .control_points
            .iter()
            .map(|row| NurbsCurve::new(degree, knots.clone(), row.clone()))
            .collect::<Result<_, _>>()
            .or_refuse(KernelStage::Intersect, "csg.imprint.support")?
    };
    if range[1] < domain[1] - guard && range[1] > domain[0] + guard {
        curves = curves
            .into_iter()
            .map(|curve| curve.split(range[1]).map(|(left, _)| left))
            .collect::<Result<_, _>>()
            .or_refuse(KernelStage::Intersect, "csg.imprint.support")?;
    }
    if range[0] > domain[0] + guard && range[0] < domain[1] - guard {
        curves = curves
            .into_iter()
            .map(|curve| curve.split(range[0]).map(|(_, right)| right))
            .collect::<Result<_, _>>()
            .or_refuse(KernelStage::Intersect, "csg.imprint.support")?;
    }
    let new_knots = curves[0].knots.clone();
    if direction_u {
        let rows = curves[0].control_points.len();
        let control = (0..rows)
            .map(|row| {
                curves
                    .iter()
                    .map(|curve| curve.control_points[row])
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            degree,
            surface.degree_v,
            new_knots,
            surface.knots_v.clone(),
            control,
        )
        .or_refuse(KernelStage::Intersect, "NurbsSurface::new")
    } else {
        let control = curves
            .iter()
            .map(|curve| curve.control_points.clone())
            .collect();
        NurbsSurface::new(
            surface.degree_u,
            degree,
            surface.knots_u.clone(),
            new_knots,
            control,
        )
        .or_refuse(KernelStage::Intersect, "NurbsSurface::new")
    }
}

/// The carrier restricted (by exact knot-insertion splitting) to the uv
/// bounding box of the trim pcurves. The pcurve control hulls bound the
/// loops, the loops bound the trimmed region, and splitting preserves the
/// global parameterization, so the result is the same surface over a window
/// that still contains the whole trimmed region — usable interchangeably
/// with the carrier for bounding, seeding, and marching. Returns None when
/// the trims span the whole carrier (nothing to gain) or the restriction
/// fails (callers keep the carrier, which is always conservative).
pub(super) fn restricted_carrier(face: &FaceRecord) -> Option<NurbsSurface> {
    let [u0, u1] = face.surface.domain_u().ok()?;
    let [v0, v1] = face.surface.domain_v().ok()?;
    let mut u_low = f64::INFINITY;
    let mut u_high = f64::NEG_INFINITY;
    let mut v_low = f64::INFINITY;
    let mut v_high = f64::NEG_INFINITY;
    let mut found = false;
    for coedge in face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
    {
        for control in &coedge.pcurve.control_points {
            let point = control.point().ok()?;
            u_low = u_low.min(point.x);
            u_high = u_high.max(point.x);
            v_low = v_low.min(point.y);
            v_high = v_high.max(point.y);
            found = true;
        }
    }
    if !found {
        return None;
    }
    // Generous margin: the window must strictly contain the trims so that
    // trim-boundary crossings (which the clip stage locates) stay interior.
    let u_margin = ((u1 - u0) * 1e-3).max(2e-9);
    let v_margin = ((v1 - v0) * 1e-3).max(2e-9);
    u_low = (u_low - u_margin).max(u0);
    u_high = (u_high + u_margin).min(u1);
    v_low = (v_low - v_margin).max(v0);
    v_high = (v_high + v_margin).min(v1);
    if u_high <= u_low || v_high <= v_low {
        return None;
    }
    let mut full_u = u_low <= u0 + u_margin && u_high >= u1 - u_margin;
    let mut full_v = v_low <= v0 + v_margin && v_high >= v1 - v_margin;
    // WRAPPED-BAND HULL GUARD: on a CLOSED direction the pcurve hull does NOT
    // necessarily bound the material — a band face whose two rims sit at the
    // hull edges can have its material in the WRAPPED COMPLEMENT of the hull
    // window (trial 35's torus face: loops at v=0.25/1.0, material v∈[0,0.25]).
    // Restricting the march surface to such a hull excludes every interior
    // material point AND breaks closure, so the SSI trace terminates at the
    // rim as a false "boundary" and the section over the band is never
    // marched. The complement strip of the full hull contains no loop curves
    // at all, so it is uniformly material or uniformly not: ONE sample at its
    // middle decides exactly. If that sample is Inside, the hull is invalid in
    // that direction — keep the full period (the unrestricted carrier is
    // always conservative). Escape hatch: BREP_RESTRICT_WRAP_GUARD=0.
    if std::env::var("BREP_RESTRICT_WRAP_GUARD").as_deref() != Ok("0") {
        let (closed_u, closed_v) = face.surface.closed_directions().ok()?;
        let sample_inside = |uv: crate::arrangement::Vec2| -> bool {
            matches!(
                parameter_point_in_face(face, uv, 1e-9),
                Ok(PolygonClass::Inside)
            )
        };
        if closed_u && !full_u {
            let period = u1 - u0;
            let gap = period - (u_high - u_low);
            let mut mid = u_high + gap / 2.0;
            if mid > u1 {
                mid -= period;
            }
            if sample_inside(crate::arrangement::Vec2 {
                x: mid,
                y: (v_low + v_high) / 2.0,
            }) {
                full_u = true;
            }
        }
        if closed_v && !full_v {
            let period = v1 - v0;
            let gap = period - (v_high - v_low);
            let mut mid = v_high + gap / 2.0;
            if mid > v1 {
                mid -= period;
            }
            if sample_inside(crate::arrangement::Vec2 {
                x: (u_low + u_high) / 2.0,
                y: mid,
            }) {
                full_v = true;
            }
        }
    }
    if full_u && full_v {
        return None;
    }
    let restricted_u = if full_u {
        None
    } else {
        Some(restrict_direction(&face.surface, [u_low, u_high], true).ok()?)
    };
    if full_v {
        restricted_u
    } else {
        let base = restricted_u.as_ref().unwrap_or(&face.surface);
        Some(restrict_direction(base, [v_low, v_high], false).ok()?)
    }
}

pub(super) fn face_bounds(
    face: &FaceRecord,
    restricted: Option<&NurbsSurface>,
    edges: &HashMap<(u8, u64), &EdgeRecord>,
    operand: u8,
    tolerance: f64,
) -> Result<Aabb, KernelRefusal> {
    let [u0, u1] = face
        .surface
        .domain_u()
        .or_refuse(KernelStage::Intersect, "domain_u")?;
    let [v0, v1] = face
        .surface
        .domain_v()
        .or_refuse(KernelStage::Intersect, "domain_v")?;
    let origin = face
        .surface
        .evaluate((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        .or_refuse(KernelStage::Intersect, "csg.imprint.support")?;
    let normal = face
        .surface
        .normal((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        .or_refuse(KernelStage::Intersect, "csg.imprint.support")?;
    let surface_controls = face
        .surface
        .control_points
        .iter()
        .flatten()
        .map(|control| control.point())
        .collect::<Result<Vec<_>, _>>()
        .or_refuse(KernelStage::Intersect, "csg.imprint.support")?;
    let scale = surface_controls
        .iter()
        .map(|point| point.sub(origin).length())
        .fold(1.0f64, f64::max);
    let plane_tolerance = (tolerance * 100.0).max(1e-7) * scale;
    if surface_controls
        .iter()
        .any(|point| point.sub(origin).dot(normal).abs() > plane_tolerance)
    {
        // Curved carrier: bound the trimmed uv sub-patch when the trims
        // cover only part of it (a revolve wall's full carrier box spans
        // the whole ring and pairs with everything).
        if let Some(restricted) = restricted {
            return Ok(Aabb::from_surface_controls(restricted)
                .or_refuse(KernelStage::Intersect, "from_surface_controls")?
                .expanded(plane_tolerance));
        }
        return Aabb::from_surface_controls(&face.surface)
            .or_refuse(KernelStage::Intersect, "from_surface_controls");
    }

    // A planar trimmed region lies inside the convex hull of its boundary.
    // Bounding its exact trimmed edge curves is therefore conservative and
    // substantially tighter than the often much larger carrier patch.
    let mut bounds = Aabb::empty();
    let mut found = false;
    for coedge in face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
    {
        let edge = edges.get(&(operand, coedge.edge_id)).ok_or_else(|| {
            KernelRefusal::internal(
                KernelStage::Intersect,
                "imprint.support",
                format!("missing edge {} on operand {operand}", coedge.edge_id),
            )
        })?;
        let curve = edge_subcurve(edge)?;
        for control in &curve.control_points {
            bounds.include_point(control.point().or_refuse(KernelStage::Intersect, "point")?);
            found = true;
        }
    }
    if !found {
        return Aabb::from_surface_controls(&face.surface)
            .or_refuse(KernelStage::Intersect, "from_surface_controls");
    }
    Ok(bounds.expanded(plane_tolerance))
}

#[derive(Clone, Copy)]
pub(super) struct TaggedFace<'a> {
    pub(super) operand: u8,
    pub(super) face: &'a FaceRecord,
}

impl TaggedFace<'_> {
    pub(super) fn key(self) -> FaceKey {
        FaceKey {
            operand: self.operand,
            face_id: self.face.id,
        }
    }
}

pub(super) fn faces(solid: &BrepSolid, operand: u8) -> Vec<TaggedFace<'_>> {
    solid
        .shells
        .iter()
        .flat_map(|shell| shell.faces.iter())
        .map(|face| TaggedFace { operand, face })
        .collect()
}

pub(super) fn edge_map<'a>(
    solid_a: &'a BrepSolid,
    solid_b: &'a BrepSolid,
) -> HashMap<(u8, u64), &'a EdgeRecord> {
    solid_a
        .edges
        .iter()
        .map(|edge| ((0, edge.id), edge))
        .chain(solid_b.edges.iter().map(|edge| ((1, edge.id), edge)))
        .collect()
}

pub(super) fn face_edges<'a>(
    face: TaggedFace<'_>,
    edges: &HashMap<(u8, u64), &'a EdgeRecord>,
) -> Result<Vec<&'a EdgeRecord>, KernelRefusal> {
    let mut seen = HashSet::default();
    let mut result = Vec::new();
    for coedge in face
        .face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
    {
        if seen.insert(coedge.edge_id) {
            result.push(*edges.get(&(face.operand, coedge.edge_id)).ok_or_else(|| {
                KernelRefusal::internal(
                    KernelStage::Intersect,
                    "imprint.support",
                    format!("face {} references missing edge", face.face.id),
                )
            })?);
        }
    }
    Ok(result)
}

pub(super) fn edge_subcurve(edge: &EdgeRecord) -> Result<NurbsCurve, KernelRefusal> {
    let mut curve = edge.curve.clone();
    let [start, end] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    // NurbsCurve::split refuses parameters within its ABSOLUTE knot identity
    // tolerance (curve.rs KNOT_IDENTITY_TOL) of the domain ends; the skip
    // epsilon must cover that or a near-edge trim parameter slips past the
    // guard and errors.
    let epsilon = (KNOT_IDENTITY_TOL * (end - start)).max(2e-9);
    if edge.t0 > start + epsilon && edge.t0 < end - epsilon {
        curve = curve
            .split(edge.t0)
            .or_refuse(KernelStage::Intersect, "split")?
            .1;
    }
    let domain = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    if edge.t1 < domain[1] - epsilon && edge.t1 > domain[0] + epsilon {
        curve = curve
            .split(edge.t1)
            .or_refuse(KernelStage::Intersect, "split")?
            .0;
    } else if std::env::var("BREP_DEBUG_SUBRANGE").is_ok() && edge.t1 < domain[1] - epsilon {
        eprintln!("abnormal subrange skip in imprint edge_subcurve");
    }
    Ok(curve)
}

pub(super) fn curve_length_rough(curve: &NurbsCurve) -> Result<f64, KernelRefusal> {
    let [start, end] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let mut previous = curve
        .evaluate(start)
        .or_refuse(KernelStage::Intersect, "evaluate")?;
    let mut length = 0.0;
    for index in 1..=8 {
        let point = curve
            .evaluate(start + (end - start) * index as f64 / 8.0)
            .or_refuse(KernelStage::Intersect, "evaluate")?;
        length += point.sub(previous).length();
        previous = point;
    }
    Ok(length)
}

pub(super) fn parameter_tolerance(
    curve: &NurbsCurve,
    parameter: f64,
    tolerance: f64,
) -> Result<f64, KernelRefusal> {
    let tangent = curve
        .derivatives(parameter, 1)
        .or_refuse(KernelStage::Intersect, "derivatives")?[1]
        .length();
    Ok(tolerance / tangent.max(1e-12))
}

pub(super) fn snap_curve_ends(
    curve: &NurbsCurve,
    start: Vec3,
    end: Vec3,
) -> Result<NurbsCurve, KernelRefusal> {
    let [t0, t1] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let p0 = curve
        .evaluate(t0)
        .or_refuse(KernelStage::Intersect, "evaluate")?;
    let p1 = curve
        .evaluate(t1)
        .or_refuse(KernelStage::Intersect, "evaluate")?;
    let cap = 5e-3 * (1.0 + p0.length());
    let d0 = p0.sub(start).length();
    let d1 = p1.sub(end).length();
    if (d0 <= 1e-12 && d1 <= 1e-12) || d0 > cap || d1 > cap {
        return Ok(curve.clone());
    }
    let mut controls = curve.control_points.clone();
    if d0 > 1e-12 {
        controls[0] = Vec4::from_point(start, controls[0].w);
    }
    let last = controls.len() - 1;
    if d1 > 1e-12 {
        controls[last] = Vec4::from_point(end, controls[last].w);
    }
    NurbsCurve::new(curve.degree, curve.knots.clone(), controls)
        .or_refuse(KernelStage::Intersect, "NurbsCurve::new")
}

pub(super) fn curve_lies_on_surface(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    let [start, end] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let scaled_tolerance = (tolerance * 100.0).max(2e-5);
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let point = curve
            .evaluate(start + (end - start) * fraction)
            .or_refuse(KernelStage::Intersect, "evaluate")?;
        if project_point_to_surface(surface, point)
            .or_refuse(KernelStage::Intersect, "project_point_to_surface")?
            .distance
            > scaled_tolerance * (1.0 + point.length())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Recover an exact isoparametric intersection when an affine plane slices
/// a surface that is linear across one parameter direction. General SSI can
/// miss this case when the other direction is a high-degree rational curve:
/// all seeds lie on two distant trim boundaries even though the intersection
/// itself is a simple interior iso-curve.
fn planar_linear_iso_intersection(
    plane: &NurbsSurface,
    ruled: &NurbsSurface,
    tolerance: f64,
) -> Result<Option<NurbsCurve>, KernelRefusal> {
    if !plane
        .is_affine()
        .or_refuse(KernelStage::Intersect, "is_affine")?
    {
        return Ok(None);
    }
    let plane_u = KnotVector::new(plane.knots_u.clone(), plane.degree_u)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let plane_v = KnotVector::new(plane.knots_v.clone(), plane.degree_v)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let derivatives = plane
        .derivatives(plane_u, plane_v, 1)
        .or_refuse(KernelStage::Intersect, "derivatives")?;
    let origin = derivatives[0][0];
    let normal = derivatives[1][0]
        .cross(derivatives[0][1])
        .normalized()
        .or_refuse(KernelStage::Intersect, "normalized")?;
    let signed_distance = |point: Vec3| point.sub(origin).dot(normal);
    let root_tolerance = (tolerance * 100.0).max(1e-7);

    let try_direction = |linear_v: bool| -> Result<Option<NurbsCurve>, KernelRefusal> {
        if linear_v {
            if ruled.degree_v != 1 || ruled.control_points.iter().any(|row| row.len() != 2) {
                return Ok(None);
            }
        } else if ruled.degree_u != 1 || ruled.control_points.len() != 2 {
            return Ok(None);
        }
        let u_domain = KnotVector::new(ruled.knots_u.clone(), ruled.degree_u)
            .or_refuse(KernelStage::Intersect, "new")?
            .domain();
        let v_domain = KnotVector::new(ruled.knots_v.clone(), ruled.degree_v)
            .or_refuse(KernelStage::Intersect, "new")?
            .domain();
        let mut roots = Vec::new();
        for fraction in [0.0, 0.2, 0.5, 0.8, 1.0] {
            let (first, second) = if linear_v {
                let u = u_domain[0] + (u_domain[1] - u_domain[0]) * fraction;
                (
                    ruled
                        .evaluate(u, v_domain[0])
                        .or_refuse(KernelStage::Intersect, "evaluate")?,
                    ruled
                        .evaluate(u, v_domain[1])
                        .or_refuse(KernelStage::Intersect, "evaluate")?,
                )
            } else {
                let v = v_domain[0] + (v_domain[1] - v_domain[0]) * fraction;
                (
                    ruled
                        .evaluate(u_domain[0], v)
                        .or_refuse(KernelStage::Intersect, "evaluate")?,
                    ruled
                        .evaluate(u_domain[1], v)
                        .or_refuse(KernelStage::Intersect, "evaluate")?,
                )
            };
            let first_distance = signed_distance(first);
            let second_distance = signed_distance(second);
            let denominator = first_distance - second_distance;
            if denominator.abs() <= root_tolerance {
                return Ok(None);
            }
            let root = first_distance / denominator;
            if root < -root_tolerance || root > 1.0 + root_tolerance {
                return Ok(None);
            }
            roots.push(root.clamp(0.0, 1.0));
        }
        let root = roots.iter().sum::<f64>() / roots.len() as f64;
        if roots
            .iter()
            .any(|candidate| (candidate - root).abs() > root_tolerance * 10.0)
        {
            return Ok(None);
        }
        let curve = if linear_v {
            ruled
                .iso_curve_v(v_domain[0] + (v_domain[1] - v_domain[0]) * root)
                .or_refuse(KernelStage::Intersect, "iso_curve_v")?
        } else {
            ruled
                .iso_curve_u(u_domain[0] + (u_domain[1] - u_domain[0]) * root)
                .or_refuse(KernelStage::Intersect, "iso_curve_u")?
        };
        if !curve_lies_on_surface(&curve, plane, tolerance)? {
            return Ok(None);
        }
        Ok(Some(curve))
    };

    if let Some(curve) = try_direction(true)? {
        return Ok(Some(curve));
    }
    try_direction(false)
}

pub(super) fn planar_iso_intersection(
    first: &NurbsSurface,
    second: &NurbsSurface,
    tolerance: f64,
) -> Result<Option<NurbsCurve>, KernelRefusal> {
    if let Some(curve) = planar_linear_iso_intersection(first, second, tolerance)? {
        return Ok(Some(curve));
    }
    planar_linear_iso_intersection(second, first, tolerance)
}

fn coplanar_pair(
    first: &NurbsSurface,
    second: &NurbsSurface,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    if first.degree_u != 1 || first.degree_v != 1 || second.degree_u != 1 || second.degree_v != 1 {
        return Ok(false);
    }
    let first_u = KnotVector::new(first.knots_u.clone(), first.degree_u)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let first_v = KnotVector::new(first.knots_v.clone(), first.degree_v)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let second_u = KnotVector::new(second.knots_u.clone(), second.degree_u)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let second_v = KnotVector::new(second.knots_v.clone(), second.degree_v)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain()[0];
    let a = first
        .derivatives(first_u, first_v, 1)
        .or_refuse(KernelStage::Intersect, "derivatives")?;
    let b = second
        .derivatives(second_u, second_v, 1)
        .or_refuse(KernelStage::Intersect, "derivatives")?;
    let normal_a = a[1][0]
        .cross(a[0][1])
        .normalized()
        .or_refuse(KernelStage::Intersect, "normalized")?;
    let normal_b = b[1][0]
        .cross(b[0][1])
        .normalized()
        .or_refuse(KernelStage::Intersect, "normalized")?;
    Ok(normal_a.dot(normal_b).abs() >= 1.0 - 1e-9
        && b[0][0].sub(a[0][0]).dot(normal_a).abs()
            <= (tolerance * 10.0).max(COINCIDENCE_DISTANCE_FLOOR))
}

pub(super) fn cosurface_pair(
    first: &NurbsSurface,
    second: &NurbsSurface,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    let first_planar = first.degree_u == 1 && first.degree_v == 1;
    let second_planar = second.degree_u == 1 && second.degree_v == 1;
    if first_planar && second_planar {
        return coplanar_pair(first, second, tolerance);
    }
    if first_planar != second_planar {
        return Ok(false);
    }
    let bounds_a = Aabb::from_surface_controls(first)
        .or_refuse(KernelStage::Intersect, "from_surface_controls")?;
    let bounds_b = Aabb::from_surface_controls(second)
        .or_refuse(KernelStage::Intersect, "from_surface_controls")?;
    if !bounds_a.intersects(bounds_b, tolerance * 10.0) {
        return Ok(false);
    }
    let scale = bounds_a.diagonal().max(bounds_b.diagonal());
    let probe = bounds_b.expanded(scale * 0.01);
    let u_domain = KnotVector::new(first.knots_u.clone(), first.degree_u)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain();
    let v_domain = KnotVector::new(first.knots_v.clone(), first.degree_v)
        .or_refuse(KernelStage::Intersect, "new")?
        .domain();
    let mut samples = Vec::new();
    for fu in [0.13, 0.37, 0.61, 0.89] {
        for fv in [0.17, 0.43, 0.71, 0.93] {
            let u = u_domain[0] + (u_domain[1] - u_domain[0]) * fu;
            let v = v_domain[0] + (v_domain[1] - v_domain[0]) * fv;
            let point = first
                .evaluate(u, v)
                .or_refuse(KernelStage::Intersect, "evaluate")?;
            if probe.contains(point) {
                samples.push((point, u, v));
            }
        }
    }
    if samples.len() < 3 {
        return Ok(false);
    }
    for (point, u, v) in samples {
        let projection = project_point_to_surface(second, point)
            .or_refuse(KernelStage::Intersect, "project_point_to_surface")?;
        if projection.distance > (tolerance * 100.0).max(COINCIDENCE_DISTANCE_FLOOR) {
            return Ok(false);
        }
        if let (Ok(normal_a), Ok(normal_b)) = (
            first.normal(u, v),
            second.normal(projection.u, projection.v),
        ) {
            if normal_a.dot(normal_b).abs() < 1.0 - 1e-6 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
