use crate::sweep_topology::{curve_to_plane_parameters, parameter_line};
use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{make_line, make_plane, make_revolution, NurbsCurve, NurbsSurface, Vec3, Vec4};

const TOLERANCE: f64 = 1e-5;

fn next_id(counter: &mut u64) -> u64 {
    let id = *counter;
    *counter += 1;
    id
}

fn rotate_point(point: Vec3, origin: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let vector = point.sub(origin);
    let cosine = angle.cos();
    let sine = angle.sin();
    origin
        .add(vector.scale(cosine))
        .add(axis.cross(vector).scale(sine))
        .add(axis.scale(axis.dot(vector) * (1.0 - cosine)))
}

fn rotate_curve(
    curve: &NurbsCurve,
    origin: Vec3,
    axis: Vec3,
    angle: f64,
) -> Result<NurbsCurve, String> {
    NurbsCurve::new(
        curve.degree,
        curve.knots.clone(),
        curve
            .control_points
            .iter()
            .map(|point| {
                let euclidean = Vec3::new(point.x / point.w, point.y / point.w, point.z / point.w);
                Vec4::from_point(rotate_point(euclidean, origin, axis, angle), point.w)
            })
            .collect(),
    )
}

fn signed_profile_area(
    curves: &[NurbsCurve],
    origin: Vec3,
    radial: Vec3,
    axis: Vec3,
) -> Result<f64, String> {
    let mut area = 0.0;
    for curve in curves {
        let [start, end] = curve.domain()?;
        let mut previous = curve.evaluate(start)?;
        for index in 1..=64 {
            let point = curve.evaluate(start + (end - start) * index as f64 / 64.0)?;
            let a = previous.sub(origin);
            let b = point.sub(origin);
            area += 0.5 * (a.dot(radial) * b.dot(axis) - b.dot(radial) * a.dot(axis));
            previous = point;
        }
    }
    Ok(area)
}

struct RevolveProfile {
    curves: Vec<NurbsCurve>,
    points: Vec<Vec3>,
    axis_points: Vec<bool>,
    axis_curves: Vec<bool>,
    /// For each entry of `curves`, true when it is a straight segment lying
    /// (within tolerance) in a plane PERPENDICULAR to the axis — a radial
    /// generatrix.  Revolving one sweeps a flat disk/annulus sector, so the
    /// partial-revolve side face is genuinely PLANAR and is built with a
    /// `make_plane` carrier (not `make_revolution`); otherwise the shallow
    /// "cone" carrier is misrecognized (conical, not planar) and its boolean
    /// intersections fall to the marcher and shed slivers.
    radial_curves: Vec<bool>,
    radial: Vec3,
    axis: Vec3,
    /// For each entry of `curves`, the index of the INPUT curve it came
    /// from.  Winding normalization reverses the loop and axis-lying
    /// curves produce no side face, so callers cannot recover this mapping
    /// from the emitted face order — side-face names must travel through
    /// it (the extrude naming permutation bug, revolve edition).
    input_indices: Vec<usize>,
}

fn prepare_profile(
    input: &[NurbsCurve],
    axis_point: Vec3,
    axis_direction: Vec3,
) -> Result<RevolveProfile, String> {
    if input.len() < 2 {
        return Err("profile needs at least 2 curves".into());
    }
    let axis = axis_direction.normalized()?;
    let mut curves: Vec<NurbsCurve> = input
        .iter()
        .map(|curve| {
            NurbsCurve::new(
                curve.degree,
                curve.knots.clone(),
                curve.control_points.clone(),
            )
        })
        .collect::<Result<_, _>>()?;
    let closure_points = |curves: &[NurbsCurve]| -> Result<Vec<Vec3>, String> {
        let mut points = Vec::with_capacity(curves.len());
        for index in 0..curves.len() {
            let [start, end] = curves[index].domain()?;
            let next_start = curves[(index + 1) % curves.len()].domain()?[0];
            let point = curves[index].evaluate(start)?;
            if curves[index]
                .evaluate(end)?
                .sub(curves[(index + 1) % curves.len()].evaluate(next_start)?)
                .length()
                > TOLERANCE
            {
                return Err(format!("profile not closed at curve {index}"));
            }
            points.push(point);
        }
        Ok(points)
    };
    let mut points = closure_points(&curves)?;

    let radial_distance = |point: Vec3| {
        let delta = point.sub(axis_point);
        delta.sub(axis.scale(delta.dot(axis))).length()
    };
    let mut radial = None;
    'outer: for curve in &curves {
        let [start, end] = curve.domain()?;
        for index in 0..=24 {
            let point = curve.evaluate(start + (end - start) * index as f64 / 24.0)?;
            let delta = point.sub(axis_point);
            let candidate = delta.sub(axis.scale(delta.dot(axis)));
            if candidate.length() > TOLERANCE * 100.0 {
                radial = Some(candidate.normalized()?);
                break 'outer;
            }
        }
    }
    let radial = radial.ok_or_else(|| "revolveSolid: profile touches the axis".to_string())?;
    let curve_on_axis = |curve: &NurbsCurve| -> Result<bool, String> {
        let [start, end] = curve.domain()?;
        for index in 0..=8 {
            if radial_distance(curve.evaluate(start + (end - start) * index as f64 / 8.0)?)
                > TOLERANCE * 100.0
            {
                return Ok(false);
            }
        }
        Ok(true)
    };
    for curve in &curves {
        let [start, end] = curve.domain()?;
        let on_axis = curve_on_axis(curve)?;
        for index in 0..=24 {
            let point = curve.evaluate(start + (end - start) * index as f64 / 24.0)?;
            let delta = point.sub(axis_point);
            let in_plane_radial = delta.dot(radial);
            let off_plane = delta
                .sub(axis.scale(delta.dot(axis)))
                .sub(radial.scale(in_plane_radial))
                .length();
            if off_plane > TOLERANCE * 100.0 {
                return Err("revolveSolid: profile not in a half-plane containing the axis".into());
            }
            if in_plane_radial < -TOLERANCE * 100.0 {
                return Err("revolveSolid: profile crosses the revolve axis".into());
            }
            if in_plane_radial < TOLERANCE * 100.0 && index != 0 && index != 24 && !on_axis {
                return Err(
                    "revolveSolid: profile may only touch the axis at boundary vertices or axis edges"
                        .into(),
                );
            }
        }
    }
    let mut input_indices: Vec<usize> = (0..curves.len()).collect();
    if signed_profile_area(&curves, axis_point, radial, axis)? < 0.0 {
        curves = curves
            .iter()
            .rev()
            .map(NurbsCurve::reversed)
            .collect::<Result<_, _>>()?;
        points = closure_points(&curves)?;
        input_indices.reverse();
    }
    let mut axis_points: Vec<bool> = points
        .iter()
        .map(|point| radial_distance(*point) <= TOLERANCE * 100.0)
        .collect();
    if axis_points.iter().any(|flag| *flag) {
        let snapped: Vec<Vec3> = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                if axis_points[index] {
                    axis_point.add(axis.scale(point.sub(axis_point).dot(axis)))
                } else {
                    *point
                }
            })
            .collect();
        for index in 0..curves.len() {
            let next = (index + 1) % curves.len();
            if !axis_points[index] && !axis_points[next] {
                continue;
            }
            let mut control_points = curves[index].control_points.clone();
            if axis_points[index] {
                let weight = control_points[0].w;
                control_points[0] = Vec4::from_point(snapped[index], weight);
            }
            if axis_points[next] {
                let last = control_points.len() - 1;
                let weight = control_points[last].w;
                control_points[last] = Vec4::from_point(snapped[next], weight);
            }
            curves[index] = NurbsCurve::new(
                curves[index].degree,
                curves[index].knots.clone(),
                control_points,
            )?;
        }
        points = closure_points(&curves)?;
        axis_points = points
            .iter()
            .map(|point| radial_distance(*point) <= TOLERANCE * 100.0)
            .collect();
    }
    let axis_curves = curves
        .iter()
        .map(curve_on_axis)
        .collect::<Result<Vec<_>, _>>()?;
    // A radial generatrix is a straight segment whose direction is
    // perpendicular to the axis: its two endpoints share an axial coordinate,
    // so the revolved patch is a flat disk/annulus sector (a plane), not a
    // cone.  Curves that lie ON the axis produce no side face and are excluded.
    let radial_curves = curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            if axis_curves[index] {
                return Ok(false);
            }
            Ok(is_straight_radial(curve, axis)?)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RevolveProfile {
        curves,
        points,
        axis_points,
        axis_curves,
        radial_curves,
        radial,
        axis,
        input_indices,
    })
}

/// True when `curve` is a straight, unit-weight line segment whose direction is
/// perpendicular to `axis` (a radial generatrix).  The threshold is a small
/// relative slope (~0.057 deg from perpendicular): it captures sketch-solver
/// noise on an intended flat cap without swallowing a genuine shallow cone.
fn is_straight_radial(curve: &NurbsCurve, axis: Vec3) -> Result<bool, String> {
    if curve.degree != 1 || curve.control_points.len() != 2 {
        return Ok(false);
    }
    let [start, end] = curve.domain()?;
    let p0 = curve.evaluate(start)?;
    let p1 = curve.evaluate(end)?;
    let direction = p1.sub(p0);
    let length = direction.length();
    if length <= TOLERANCE {
        return Ok(false);
    }
    Ok(direction.dot(axis).abs() <= 1e-3 * length)
}

fn junction_curve(
    index: usize,
    surfaces: &[Option<NurbsSurface>],
    points: &[Vec3],
) -> Result<NurbsCurve, String> {
    if let Some(surface) = &surfaces[index] {
        return surface.iso_curve_v(surface.knots_v[surface.degree_v]);
    }
    let previous = (index + surfaces.len() - 1) % surfaces.len();
    if let Some(surface) = &surfaces[previous] {
        return surface.iso_curve_v(surface.knots_v[surface.knots_v.len() - 1 - surface.degree_v]);
    }
    make_line(points[index], points[index])
}

/// Build the PLANAR side face of a partial revolve whose generatrix is radial
/// (perpendicular to the axis): a flat disk/annulus sector.  The face carrier
/// is an affine `make_plane` — so it recognizes as a `Plane` (not a shallow
/// "cone"), reads planar in the app, and its boolean intersections take the
/// exact analytic path instead of the sliver-prone marcher.  The bounding
/// loop reuses the same edges as the revolution branch; only the surface and
/// its pcurves differ (the edges are projected into the plane's frame).
#[allow(clippy::too_many_arguments)]
fn radial_plane_face(
    profile: &RevolveProfile,
    index: usize,
    count: usize,
    origin: Vec3,
    revolution: &NurbsSurface,
    arc_curves: &[NurbsCurve],
    end_curves: &[NurbsCurve],
    arc_ids: &[u64],
    start_edge_ids: &[u64],
    end_edge_ids: &[u64],
    counter: &mut u64,
    name: Option<String>,
) -> Result<FaceRecord, String> {
    let axis = profile.axis;
    let a_start = profile.points[index].sub(origin).dot(axis);
    let a_end = profile.points[(index + 1) % count].sub(origin).dot(axis);
    let plane_center = origin.add(axis.scale(0.5 * (a_start + a_end)));
    let e1 = profile.radial;
    let e2 = axis.cross(e1);

    let next = (index + 1) % count;
    let boundary: [(NurbsCurve, u64, bool); 4] = [
        (arc_curves[index].clone(), arc_ids[index], true),
        (end_curves[index].clone(), end_edge_ids[index], true),
        (arc_curves[next].clone(), arc_ids[next], false),
        (profile.curves[index].clone(), start_edge_ids[index], false),
    ];

    let mut min1 = f64::INFINITY;
    let mut max1 = f64::NEG_INFINITY;
    let mut min2 = f64::INFINITY;
    let mut max2 = f64::NEG_INFINITY;
    for (curve, _, _) in &boundary {
        let [s, e] = curve.domain()?;
        for k in 0..=16 {
            let point = curve.evaluate(s + (e - s) * k as f64 / 16.0)?;
            let d = point.sub(plane_center);
            let c1 = d.dot(e1);
            let c2 = d.dot(e2);
            min1 = min1.min(c1);
            max1 = max1.max(c1);
            min2 = min2.min(c2);
            max2 = max2.max(c2);
        }
    }
    let padding = (max1 - min1).max(max2 - min2) * 0.05 + 1e-6;
    let width = max1 - min1 + 2.0 * padding;
    let height = max2 - min2 + 2.0 * padding;
    let corner = plane_center
        .add(e1.scale(min1 - padding))
        .add(e2.scale(min2 - padding));
    let plane = make_plane(corner, e1, e2, width, height)?;

    // Preserve the revolution carrier's outward orientation: the plane's
    // geometric normal is e1 x e2, so same_sense follows whether that agrees
    // with the revolution's normal at the patch midpoint.
    let [ru0, ru1] = revolution.domain_u()?;
    let [rv0, rv1] = revolution.domain_v()?;
    let rev_normal = revolution.normal((ru0 + ru1) * 0.5, (rv0 + rv1) * 0.5)?;
    let same_sense = rev_normal.dot(e1.cross(e2)) >= 0.0;

    let mut coedges = Vec::with_capacity(4);
    for (curve, edge_id, forward) in boundary {
        let mapped = curve_to_plane_parameters(&curve, corner, e1, e2)?;
        let pcurve = if forward { mapped } else { mapped.reversed()? };
        coedges.push(CoedgeRecord {
            id: next_id(counter),
            edge_id,
            forward,
            pcurve,
        });
    }
    let loop_id = next_id(counter);
    Ok(FaceRecord {
        id: next_id(counter),
        surface: plane,
        same_sense,
        loops: vec![LoopRecord {
            id: loop_id,
            coedges,
        }],
        name,
    })
}

pub fn revolve_profile_brep(
    input: &[NurbsCurve],
    axis_point: Vec3,
    axis_direction: Vec3,
    angle: f64,
) -> Result<BrepSolid, String> {
    revolve_profile_brep_named(input, axis_point, axis_direction, angle, &[], &[])
}

/// Revolve with caller-supplied names.  `side_names` is aligned with the
/// INPUT curves; the kernel carries it through winding normalization and
/// axis-curve skips, which the caller cannot reconstruct from the emitted
/// face order.  `cap_names` is `[start_cap, end_cap]` for partial
/// revolutions.  Empty slices leave faces unnamed.
pub fn revolve_profile_brep_named(
    input: &[NurbsCurve],
    axis_point: Vec3,
    axis_direction: Vec3,
    angle: f64,
    side_names: &[Option<String>],
    cap_names: &[Option<String>],
) -> Result<BrepSolid, String> {
    if angle <= 1e-9 || angle > std::f64::consts::TAU + 1e-9 {
        return Err("revolveSolid: angle must be in (0, 2*PI]".into());
    }
    if !side_names.is_empty() && side_names.len() != input.len() {
        return Err(format!(
            "revolveSolid: {} side names for {} profile curves",
            side_names.len(),
            input.len()
        ));
    }
    let full_turn = angle > std::f64::consts::TAU - 1e-9;
    let profile = prepare_profile(input, axis_point, axis_direction)?;
    if full_turn {
        revolve_full(profile, axis_point, side_names)
    } else {
        revolve_partial(profile, axis_point, angle, side_names, cap_names)
    }
}

fn side_name(
    profile: &RevolveProfile,
    curve_index: usize,
    side_names: &[Option<String>],
) -> Option<String> {
    profile
        .input_indices
        .get(curve_index)
        .and_then(|input_index| side_names.get(*input_index))
        .cloned()
        .flatten()
}

fn revolve_full(
    profile: RevolveProfile,
    origin: Vec3,
    side_names: &[Option<String>],
) -> Result<BrepSolid, String> {
    let count = profile.curves.len();
    let surfaces: Vec<Option<NurbsSurface>> = profile
        .curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            if profile.axis_curves[index] {
                Ok(None)
            } else {
                make_revolution(origin, profile.axis, curve, std::f64::consts::TAU).map(Some)
            }
        })
        .collect::<Result<_, String>>()?;
    let vertices: Vec<VertexRecord> = profile
        .points
        .iter()
        .enumerate()
        .map(|(index, point)| VertexRecord {
            id: index as u64 + 1,
            point: *point,
        })
        .collect();
    let mut counter = 100_u64;
    let mut edges = Vec::new();
    let mut circle_ids = Vec::with_capacity(count);
    for index in 0..count {
        let curve = junction_curve(index, &surfaces, &profile.points)?;
        let [start, end] = curve.domain()?;
        let id = next_id(&mut counter);
        circle_ids.push(id);
        edges.push(EdgeRecord {
            id,
            curve,
            t0: start,
            t1: end,
            start_vertex_id: index as u64 + 1,
            end_vertex_id: index as u64 + 1,
            degenerate: profile.axis_points[index],
            name: None,
        });
    }
    let mut faces = Vec::new();
    for index in 0..count {
        if profile.axis_curves[index] {
            continue;
        }
        let curve = profile.curves[index].clone();
        let [start, end] = curve.domain()?;
        let seam_id = next_id(&mut counter);
        edges.push(EdgeRecord {
            id: seam_id,
            curve,
            t0: start,
            t1: end,
            start_vertex_id: index as u64 + 1,
            end_vertex_id: ((index + 1) % count) as u64 + 1,
            degenerate: false,
            name: None,
        });
        let coedges = vec![
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: circle_ids[index],
                forward: true,
                pcurve: parameter_line(0.0, start, 1.0, start)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: seam_id,
                forward: true,
                pcurve: parameter_line(1.0, start, 1.0, end)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: circle_ids[(index + 1) % count],
                forward: false,
                pcurve: parameter_line(1.0, end, 0.0, end)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: seam_id,
                forward: false,
                pcurve: parameter_line(0.0, end, 0.0, start)?,
            },
        ];
        let loop_id = next_id(&mut counter);
        faces.push(FaceRecord {
            id: next_id(&mut counter),
            surface: surfaces[index].clone().unwrap(),
            same_sense: true,
            loops: vec![LoopRecord {
                id: loop_id,
                coedges,
            }],
            name: side_name(&profile, index, side_names),
        });
    }
    let solid = BrepSolid {
        id: next_id(&mut counter),
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: next_id(&mut counter),
            faces,
        }],
        genus: if profile.axis_points.iter().any(|flag| *flag) {
            0
        } else {
            1
        },
    };
    let issues = solid.validate();
    if issues.is_empty() {
        Ok(solid)
    } else {
        Err(format!("Rust full revolution is invalid: {issues:?}"))
    }
}

fn revolve_partial(
    profile: RevolveProfile,
    origin: Vec3,
    angle: f64,
    side_names: &[Option<String>],
    cap_names: &[Option<String>],
) -> Result<BrepSolid, String> {
    let count = profile.curves.len();
    let end_curves: Vec<NurbsCurve> = profile
        .curves
        .iter()
        .map(|curve| rotate_curve(curve, origin, profile.axis, angle))
        .collect::<Result<_, _>>()?;
    let surfaces: Vec<Option<NurbsSurface>> = profile
        .curves
        .iter()
        .enumerate()
        .map(|(index, curve)| {
            if profile.axis_curves[index] {
                Ok(None)
            } else {
                make_revolution(origin, profile.axis, curve, angle).map(Some)
            }
        })
        .collect::<Result<_, String>>()?;
    let mut vertices = Vec::new();
    let mut start_vertex_ids = Vec::with_capacity(count);
    let mut end_vertex_ids = Vec::with_capacity(count);
    let mut counter = 100_u64;
    for point in &profile.points {
        let id = next_id(&mut counter);
        start_vertex_ids.push(id);
        vertices.push(VertexRecord { id, point: *point });
    }
    for index in 0..count {
        if profile.axis_points[index] {
            end_vertex_ids.push(start_vertex_ids[index]);
        } else {
            let id = next_id(&mut counter);
            end_vertex_ids.push(id);
            vertices.push(VertexRecord {
                id,
                point: rotate_point(profile.points[index], origin, profile.axis, angle),
            });
        }
    }

    let mut edges = Vec::new();
    let mut arc_ids = Vec::with_capacity(count);
    let mut arc_curves = Vec::with_capacity(count);
    for index in 0..count {
        let curve = junction_curve(index, &surfaces, &profile.points)?;
        let [start, end] = curve.domain()?;
        let id = next_id(&mut counter);
        arc_ids.push(id);
        arc_curves.push(curve.clone());
        edges.push(EdgeRecord {
            id,
            curve,
            t0: start,
            t1: end,
            start_vertex_id: start_vertex_ids[index],
            end_vertex_id: end_vertex_ids[index],
            degenerate: profile.axis_points[index],
            name: None,
        });
    }
    let mut start_edge_ids = Vec::with_capacity(count);
    let mut end_edge_ids = Vec::with_capacity(count);
    for index in 0..count {
        let [start, end] = profile.curves[index].domain()?;
        let start_id = next_id(&mut counter);
        start_edge_ids.push(start_id);
        edges.push(EdgeRecord {
            id: start_id,
            curve: profile.curves[index].clone(),
            t0: start,
            t1: end,
            start_vertex_id: start_vertex_ids[index],
            end_vertex_id: start_vertex_ids[(index + 1) % count],
            degenerate: false,
            name: None,
        });
        if profile.axis_curves[index] {
            end_edge_ids.push(start_id);
        } else {
            let end_id = next_id(&mut counter);
            end_edge_ids.push(end_id);
            edges.push(EdgeRecord {
                id: end_id,
                curve: end_curves[index].clone(),
                t0: start,
                t1: end,
                start_vertex_id: end_vertex_ids[index],
                end_vertex_id: end_vertex_ids[(index + 1) % count],
                degenerate: false,
                name: None,
            });
        }
    }

    let mut faces = Vec::new();
    for index in 0..count {
        if profile.axis_curves[index] {
            continue;
        }
        if profile.radial_curves[index] {
            faces.push(radial_plane_face(
                &profile,
                index,
                count,
                origin,
                surfaces[index].as_ref().unwrap(),
                &arc_curves,
                &end_curves,
                &arc_ids,
                &start_edge_ids,
                &end_edge_ids,
                &mut counter,
                side_name(&profile, index, side_names),
            )?);
            continue;
        }
        let [start, end] = profile.curves[index].domain()?;
        let coedges = vec![
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: arc_ids[index],
                forward: true,
                pcurve: parameter_line(0.0, start, 1.0, start)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: end_edge_ids[index],
                forward: true,
                pcurve: parameter_line(1.0, start, 1.0, end)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: arc_ids[(index + 1) % count],
                forward: false,
                pcurve: parameter_line(1.0, end, 0.0, end)?,
            },
            CoedgeRecord {
                id: next_id(&mut counter),
                edge_id: start_edge_ids[index],
                forward: false,
                pcurve: parameter_line(0.0, end, 0.0, start)?,
            },
        ];
        let loop_id = next_id(&mut counter);
        faces.push(FaceRecord {
            id: next_id(&mut counter),
            surface: surfaces[index].clone().unwrap(),
            same_sense: true,
            loops: vec![LoopRecord {
                id: loop_id,
                coedges,
            }],
            name: side_name(&profile, index, side_names),
        });
    }

    let mut min_radial = f64::INFINITY;
    let mut max_radial = f64::NEG_INFINITY;
    let mut min_axis = f64::INFINITY;
    let mut max_axis = f64::NEG_INFINITY;
    for curve in &profile.curves {
        let [start, end] = curve.domain()?;
        for index in 0..=24 {
            let delta = curve
                .evaluate(start + (end - start) * index as f64 / 24.0)?
                .sub(origin);
            let radial = delta.dot(profile.radial);
            let axial = delta.dot(profile.axis);
            min_radial = min_radial.min(radial);
            max_radial = max_radial.max(radial);
            min_axis = min_axis.min(axial);
            max_axis = max_axis.max(axial);
        }
    }
    let padding = (max_radial - min_radial).max(max_axis - min_axis) * 0.05 + 1e-6;
    let width = max_radial - min_radial + 2.0 * padding;
    let height = max_axis - min_axis + 2.0 * padding;
    let start_origin = origin
        .add(profile.radial.scale(min_radial - padding))
        .add(profile.axis.scale(min_axis - padding));
    let mut start_coedges = Vec::with_capacity(count);
    for index in 0..count {
        start_coedges.push(CoedgeRecord {
            id: next_id(&mut counter),
            edge_id: start_edge_ids[index],
            forward: true,
            pcurve: curve_to_plane_parameters(
                &profile.curves[index],
                start_origin,
                profile.radial,
                profile.axis,
            )?,
        });
    }
    let start_loop_id = next_id(&mut counter);
    faces.push(FaceRecord {
        id: next_id(&mut counter),
        surface: make_plane(start_origin, profile.radial, profile.axis, width, height)?,
        same_sense: true,
        loops: vec![LoopRecord {
            id: start_loop_id,
            coedges: start_coedges,
        }],
        name: cap_names.first().cloned().flatten(),
    });

    let end_radial = rotate_point(origin.add(profile.radial), origin, profile.axis, angle)
        .sub(origin)
        .normalized()?;
    let end_origin = origin
        .add(end_radial.scale(min_radial - padding))
        .add(profile.axis.scale(min_axis - padding));
    let mut end_coedges = Vec::with_capacity(count);
    for index in (0..count).rev() {
        end_coedges.push(CoedgeRecord {
            id: next_id(&mut counter),
            edge_id: end_edge_ids[index],
            forward: false,
            pcurve: curve_to_plane_parameters(
                &end_curves[index],
                end_origin,
                profile.axis,
                end_radial,
            )?
            .reversed()?,
        });
    }
    let end_loop_id = next_id(&mut counter);
    faces.push(FaceRecord {
        id: next_id(&mut counter),
        surface: make_plane(end_origin, profile.axis, end_radial, height, width)?,
        same_sense: true,
        loops: vec![LoopRecord {
            id: end_loop_id,
            coedges: end_coedges,
        }],
        name: cap_names.get(1).cloned().flatten(),
    });

    let solid = BrepSolid {
        id: next_id(&mut counter),
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: next_id(&mut counter),
            faces,
        }],
        genus: 0,
    };
    let issues = solid.validate();
    if issues.is_empty() {
        Ok(solid)
    } else {
        Err(format!("Rust partial revolution is invalid: {issues:?}"))
    }
}

// BREP private tests: 36d0585030357ab9
