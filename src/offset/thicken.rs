//! §5.9 Sheet → solid THICKEN (Golovanov): turn an open surface patch —
//! either its full parameter rectangle or a region TRIMMED by pcurve loops —
//! into a slab solid bounded by the sheet offset on one or both sides plus
//! ruled side walls along every boundary pcurve.
//!
//! The offset carriers come from `offset_surface` (§3.15 equidistant
//! surface), which preserves the source degree / knots / weights — planes
//! stay exact planes and cylinders / spheres / tori stay exact rational
//! carriers.  Because the two sheets share one basis, each side wall is the
//! HOMOGENEOUS ruling between corresponding boundary curves: with equal
//! per-column weights the ruling evaluates to the pointwise segment
//! (1−w)·B(s) + w·T(s), so wall pcurves are exact parameter lines, a planar
//! rectangle thickens to an EXACT box, and a circular hole in a planar sheet
//! grows an EXACT cylindrical tube.
//!
//! Trim loops follow the FaceRecord convention (`validate_uv_wire`): the
//! outer loop runs counter-clockwise in (u, v) and hole loops run clockwise,
//! exactly as they would appear on a `same_sense = true` face.  Each hole
//! adds one handle: the result of thickening a sheet with h holes is a
//! genus-h solid, and the Euler check V − E + F − H = 2(1 − genus) closes.
//!
//! Where the equidistant surface would degenerate — a concave principal
//! curvature radius smaller than the offset distance folds the offset
//! through its evolute — the builder refuses with an honest `Err` instead
//! of assembling silent garbage.

use crate::offset::offset_surface;
use crate::sweep_topology::parameter_line;
use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::image_curve::{affine_image_curve, image_curve_pair};
use crate::{make_line, NurbsCurve, NurbsSurface, Vec3};

/// The sheet's principal curvatures at (u, v) — [`NurbsSurface::principal_curvatures`],
/// with this operation's name on the refusal.
fn principal_curvatures(surface: &NurbsSurface, u: f64, v: f64) -> Result<(f64, f64), String> {
    surface
        .principal_curvatures(u, v)
        .map_err(|error| format!("thickenSheet: {error}"))
}

/// Refuse offsets that fold through the sheet's evolute: at every sampled
/// (u, v) and for every requested signed offset distance d the per-direction
/// area factor 1 − d·κ must stay positive, or the equidistant surface
/// self-intersects (concave curvature radius ≤ offset distance).
fn ensure_offsets_regular(surface: &NurbsSurface, distances: &[f64]) -> Result<(), String> {
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    const SAMPLES: usize = 33;
    for i in 0..SAMPLES {
        let u = u0 + (u1 - u0) * i as f64 / (SAMPLES - 1) as f64;
        for j in 0..SAMPLES {
            let v = v0 + (v1 - v0) * j as f64 / (SAMPLES - 1) as f64;
            let (kappa_min, kappa_max) = principal_curvatures(surface, u, v)?;
            for &distance in distances {
                if distance == 0.0 {
                    continue;
                }
                for kappa in [kappa_min, kappa_max] {
                    let factor = 1.0 - distance * kappa;
                    if factor <= 1e-6 {
                        let radius = 1.0 / kappa.abs().max(1e-300);
                        return Err(format!(
                            "thickenSheet: offset by {distance:.6} self-intersects — the \
                             sheet's concave curvature radius {radius:.6} at (u={u:.4}, \
                             v={v:.4}) is not larger than the offset distance"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Equidistant sheet moved `distance` along the parametrization normal
/// n = Su × Sv (positive = +n side).  `offset_surface`'s positive distance
/// moves OPPOSITE the face normal, hence the negation.
fn offset_sheet(surface: &NurbsSurface, distance: f64) -> Result<NurbsSurface, String> {
    if distance == 0.0 {
        return Ok(surface.clone());
    }
    let carrier = FaceRecord {
        id: 1,
        surface: surface.clone(),
        same_sense: true,
        loops: vec![],
        name: None,
    };
    offset_surface(&carrier, -distance, 0.0)
}

/// Ruled wall between corresponding boundary curves of the bottom and top
/// sheets.  Requires the shared basis that `offset_surface` guarantees
/// (same degree, knots, and per-column weights); with equal weights the
/// homogeneous ruling evaluates to the exact pointwise segment
/// (1−w)·bottom(s) + w·top(s).
fn ruled_wall(bottom: &NurbsCurve, top: &NurbsCurve) -> Result<NurbsSurface, String> {
    if bottom.degree != top.degree
        || bottom.knots.len() != top.knots.len()
        || bottom
            .knots
            .iter()
            .zip(&top.knots)
            .any(|(a, b)| (a - b).abs() > 1e-12)
        || bottom
            .control_points
            .iter()
            .zip(&top.control_points)
            .any(|(a, b)| (a.w - b.w).abs() > 1e-9)
    {
        return Err("thickenSheet: internal error — offset sheet basis mismatch".into());
    }
    let rows = bottom
        .control_points
        .iter()
        .zip(&top.control_points)
        .map(|(b, t)| vec![*b, *t])
        .collect();
    NurbsSurface::new(
        bottom.degree,
        1,
        bottom.knots.clone(),
        vec![0.0, 0.0, 1.0, 1.0],
        rows,
    )
}

const GAUSS_X: [f64; 8] = [
    -0.9602898564975363,
    -0.7966664774136267,
    -0.525532409916329,
    -0.18343464249564978,
    0.18343464249564978,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975363,
];
const GAUSS_W: [f64; 8] = [
    0.10122853629037669,
    0.22238103445337445,
    0.31370664587788727,
    0.362683783378362,
    0.362683783378362,
    0.31370664587788727,
    0.22238103445337445,
    0.10122853629037669,
];

/// Green's-theorem signed-area contribution ∮ (x·y' − y·x')/2 of one pcurve,
/// integrated per knot span with 8-point Gauss.  Summed over a closed loop
/// this is the loop's signed (u, v) area — the orientation oracle for the
/// outer-CCW / hole-CW convention.
fn pcurve_signed_area(curve: &NurbsCurve) -> Result<f64, String> {
    let [q0, q1] = curve.domain()?;
    let mut breaks = vec![q0];
    for &knot in &curve.knots {
        if knot > q0 + 1e-12 && knot < q1 - 1e-12 && (knot - breaks[breaks.len() - 1]).abs() > 1e-12
        {
            breaks.push(knot);
        }
    }
    breaks.push(q1);
    let mut area = 0.0;
    for pair in breaks.windows(2) {
        let half = (pair[1] - pair[0]) * 0.5;
        let middle = (pair[1] + pair[0]) * 0.5;
        for index in 0..GAUSS_X.len() {
            let derivatives = curve.derivatives(middle + half * GAUSS_X[index], 1)?;
            let point = derivatives[0];
            let tangent = derivatives[1];
            area += GAUSS_W[index] * half * 0.5 * (point.x * tangent.y - point.y * tangent.x);
        }
    }
    Ok(area)
}

/// Distance between two pcurve evaluations in the (u, v) plane (the z slot
/// of a parameter-space curve is dead weight).
fn planar_gap(first: Vec3, second: Vec3) -> f64 {
    let du = first.x - second.x;
    let dv = first.y - second.y;
    (du * du + dv * dv).sqrt()
}

/// 3D images of one boundary pcurve on the bottom and top sheets, plus the
/// edge parameter range they are represented over and whether the stored
/// loop direction runs with increasing edge parameter.
struct BoundaryImages {
    bottom: NurbsCurve,
    top: NurbsCurve,
    t0: f64,
    t1: f64,
    /// Stored pcurve direction == increasing edge parameter.
    dir: bool,
}

/// Build the bottom/top 3D images of one boundary pcurve.
///
/// * Affine sheets take ANY rational pcurve (exact homogeneous mapping).
/// * Curved sheets take iso-parameter LINE segments (u = const or
///   v = const): the image is the shared-basis isocurve of each sheet,
///   trimmed to the segment's parameter range.  A degree-1 equal-weight
///   pcurve maps its parameter linearly onto the iso parameter, so the
///   validator's fraction-matched pcurve consistency check is exact.
/// * Anything else on a curved sheet goes to the GENERAL image ladder
///   ([`crate::image_curve::image_curve_pair`]): the composed curve-on-surface
///   is sampled and fitted on both sheets over ONE parameter set, which is what
///   makes the pair basis-identical for [`ruled_wall`].  The transfer is exact
///   in the sense that matters here — unlike a push against a FIXED neighbour,
///   both sheets move together, so the image of the shared trim IS the
///   boundary, not an approximation of some other curve.  The ladder refuses
///   with its measured deviation rather than returning an unvalidated fit.
fn boundary_images(
    base_affine: bool,
    bottom: &NurbsSurface,
    top: &NurbsSurface,
    pcurve: &NurbsCurve,
    eps_u: f64,
    eps_v: f64,
    fit_tolerance: f64,
) -> Result<BoundaryImages, String> {
    if base_affine {
        let [q0, q1] = pcurve.domain()?;
        return Ok(BoundaryImages {
            bottom: affine_image_curve(bottom, pcurve)?,
            top: affine_image_curve(top, pcurve)?,
            t0: q0,
            t1: q1,
            dir: true,
        });
    }
    if pcurve.degree == 1 && pcurve.control_points.len() == 2 {
        let first = pcurve.control_points[0];
        let second = pcurve.control_points[1];
        if (first.w - second.w).abs() <= 1e-12 {
            let (ua, va) = (first.x / first.w, first.y / first.w);
            let (ub, vb) = (second.x / second.w, second.y / second.w);
            if (ua - ub).abs() <= eps_u && (va - vb).abs() > eps_v {
                let u_constant = (ua + ub) * 0.5;
                return Ok(BoundaryImages {
                    bottom: bottom.iso_curve_u(u_constant)?,
                    top: top.iso_curve_u(u_constant)?,
                    t0: va.min(vb),
                    t1: va.max(vb),
                    dir: vb > va,
                });
            }
            if (va - vb).abs() <= eps_v && (ua - ub).abs() > eps_u {
                let v_constant = (va + vb) * 0.5;
                return Ok(BoundaryImages {
                    bottom: bottom.iso_curve_v(v_constant)?,
                    top: top.iso_curve_v(v_constant)?,
                    t0: ua.min(ub),
                    t1: ua.max(ub),
                    dir: ub > ua,
                });
            }
        }
    }
    // The general trim: no closed form, so fit the composed curve on both
    // sheets and let the ladder measure itself.
    let (bottom_image, top_image) =
        image_curve_pair(bottom, top, pcurve, fit_tolerance, "thickenSheet")?;
    let forward = bottom_image.t0 <= bottom_image.t1;
    Ok(BoundaryImages {
        bottom: bottom_image.curve,
        top: top_image.curve,
        t0: bottom_image.t0.min(bottom_image.t1),
        t1: bottom_image.t0.max(bottom_image.t1),
        dir: forward,
    })
}

/// §5.9 THICKEN a TRIMMED sheet region into a closed solid.
///
/// `loops` are parameter-space curves on `surface` exactly as FaceRecord
/// loops store them: the outer loop first, running counter-clockwise in
/// (u, v), followed by optional hole loops running clockwise (the
/// `same_sense = true` convention of `validate_uv_wire`).  Each loop's
/// pcurves must chain tip-to-tail and close; a loop may also be a single
/// closed pcurve (e.g. a rational circle).
///
/// * `symmetric = false`: the solid occupies the space between the sheet and
///   its offset at signed `thickness` along the sheet normal n = Su × Sv
///   (negative thickness grows the solid on the −n side).
/// * `symmetric = true`: the material splits evenly, |thickness|/2 on each
///   side of the sheet (the sheet becomes the mid-surface).
///
/// The caps are the bottom/top offset sheets trimmed by the SAME pcurve
/// loops (the offset shares the sheet's basis, so pcurves transfer
/// verbatim); every boundary pcurve contributes one ruled side wall between
/// its bottom and top 3D images, ruled pointwise at equal pcurve parameter —
/// for an offset pair that ruling runs along the surface normal, so walls
/// are exact wherever the full-domain walls were.  Hole loops produce inner
/// wall tubes and each adds one handle: `genus = loops.len() - 1`.
///
/// Every boundary edge is a single EdgeRecord shared by exactly two coedges
/// (cap + wall); wall-to-wall junction edges are likewise shared.
///
/// A general (non-iso) pcurve on a CURVED sheet is no longer refused: its two
/// 3D images come from [`crate::image_curve::image_curve_pair`], which fits the
/// composed curve-on-surface on both sheets over ONE shared parameter set (the
/// basis identity [`ruled_wall`] requires) and refuses only when the fit misses
/// its measured tolerance.  Unlike a push against a FIXED neighbour, both
/// sheets here move together, so the image of the shared trim IS the boundary.
///
/// Refuses (Err) on: zero/non-finite thickness, closed sheets, open or
/// misoriented loops, pinched loops, a general pcurve image that cannot be
/// fitted to tolerance, and offsets through the evolute.
pub fn thicken_trimmed_sheet(
    surface: &NurbsSurface,
    loops: &[Vec<NurbsCurve>],
    thickness: f64,
    symmetric: bool,
) -> Result<BrepSolid, String> {
    if !thickness.is_finite() || thickness.abs() <= 1e-12 {
        return Err("thickenSheet: thickness must be a nonzero finite value".into());
    }
    if loops.is_empty() || loops.iter().any(|loop_curves| loop_curves.is_empty()) {
        return Err("thickenSheet: at least one non-empty pcurve loop is required".into());
    }
    let (closed_u, closed_v) = surface.closed_directions()?;
    if closed_u || closed_v {
        return Err(
            "thickenSheet: closed sheets are not supported (split the patch at its seam first)"
                .into(),
        );
    }
    let (distance_bottom, distance_top) = if symmetric {
        (-thickness.abs() * 0.5, thickness.abs() * 0.5)
    } else if thickness > 0.0 {
        (0.0, thickness)
    } else {
        (thickness, 0.0)
    };
    ensure_offsets_regular(surface, &[distance_bottom, distance_top])?;

    let bottom = offset_sheet(surface, distance_bottom)?;
    let top = offset_sheet(surface, distance_top)?;
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let uv_tolerance = 1e-7 * (u1 - u0).max(v1 - v0);
    let minimum_area = 1e-10 * (u1 - u0) * (v1 - v0);
    let eps_u = 1e-9 * (u1 - u0);
    let eps_v = 1e-9 * (v1 - v0);
    let base_affine = surface.is_affine()?;
    // Fit-accuracy bar for the general image ladder.  `intersection_fit` is
    // the kernel's single NAMED "maximum geometric error accepted while
    // fitting" field (`geometry/tolerance.rs`), sized to this sheet — an
    // accuracy target, deliberately not the validator's loose
    // `pcurve_acceptance` identity band, which would admit a fit that misses
    // the true boundary by 2.5% of the model.
    let sheet_points = bottom
        .control_points
        .iter()
        .flatten()
        .map(|control| control.point())
        .collect::<Result<Vec<_>, String>>()?;
    let fit_tolerance =
        crate::KernelTolerances::for_scale(crate::model_scale(sheet_points), 1e-7).intersection_fit;

    let mut vertices: Vec<VertexRecord> = Vec::new();
    let mut edges: Vec<EdgeRecord> = Vec::new();
    let mut faces: Vec<FaceRecord> = Vec::new();
    let mut top_cap_loops: Vec<LoopRecord> = Vec::new();
    let mut bottom_cap_loops: Vec<LoopRecord> = Vec::new();
    let mut bottom_junction_points: Vec<Vec3> = Vec::new();
    let mut next_id = 1u64;

    for (loop_index, loop_curves) in loops.iter().enumerate() {
        let count = loop_curves.len();

        // ---- Parameter-space checks: closure, pinches, orientation. ----
        let mut starts = Vec::with_capacity(count);
        let mut ends = Vec::with_capacity(count);
        for curve in loop_curves {
            let [q0, q1] = curve.domain()?;
            starts.push(curve.evaluate(q0)?);
            ends.push(curve.evaluate(q1)?);
        }
        for index in 0..count {
            let next_index = (index + 1) % count;
            let gap = planar_gap(ends[index], starts[next_index]);
            if gap > uv_tolerance {
                return Err(format!(
                    "thickenSheet: loop {loop_index} is open — pcurve {index} ends at \
                     (u={:.6}, v={:.6}) but pcurve {next_index} starts at (u={:.6}, v={:.6}) \
                     (parameter-space gap {gap:.3e})",
                    ends[index].x, ends[index].y, starts[next_index].x, starts[next_index].y
                ));
            }
        }
        if count == 1 {
            let [q0, q1] = loop_curves[0].domain()?;
            let middle = loop_curves[0].evaluate((q0 + q1) * 0.5)?;
            if planar_gap(middle, starts[0]) <= uv_tolerance {
                return Err(format!(
                    "thickenSheet: loop {loop_index} is degenerate (zero parameter-space extent)"
                ));
            }
        } else {
            for index in 0..count {
                if planar_gap(ends[index], starts[index]) <= uv_tolerance {
                    return Err(format!(
                        "thickenSheet: pcurve {index} of loop {loop_index} closes on itself \
                         inside a multi-curve loop (pinched loop)"
                    ));
                }
            }
        }
        let mut area = 0.0;
        for curve in loop_curves {
            area += pcurve_signed_area(curve)?;
        }
        if loop_index == 0 {
            if area <= minimum_area {
                return Err(format!(
                    "thickenSheet: outer loop must run counter-clockwise in (u, v) \
                     (signed area {area:.3e})"
                ));
            }
        } else if area >= -minimum_area {
            return Err(format!(
                "thickenSheet: hole loop {loop_index} must run clockwise in (u, v) \
                 (signed area {area:.3e})"
            ));
        }

        // ---- Junction vertices: junction j = start of pcurve j. ----
        let mut bottom_vertex_ids = Vec::with_capacity(count);
        let mut top_vertex_ids = Vec::with_capacity(count);
        let mut bottom_points = Vec::with_capacity(count);
        let mut top_points = Vec::with_capacity(count);
        for start in &starts {
            let bottom_point = bottom.evaluate(start.x, start.y)?;
            let top_point = top.evaluate(start.x, start.y)?;
            vertices.push(VertexRecord {
                id: next_id,
                point: bottom_point,
            });
            bottom_vertex_ids.push(next_id);
            next_id += 1;
            vertices.push(VertexRecord {
                id: next_id,
                point: top_point,
            });
            top_vertex_ids.push(next_id);
            next_id += 1;
            bottom_points.push(bottom_point);
            top_points.push(top_point);
            bottom_junction_points.push(bottom_point);
        }

        // ---- Boundary edges on both sheets + vertical junction edges. ----
        let mut images = Vec::with_capacity(count);
        for curve in loop_curves {
            images.push(boundary_images(
                base_affine,
                &bottom,
                &top,
                curve,
                eps_u,
                eps_v,
                fit_tolerance,
            )?);
        }
        let mut bottom_edge_ids = Vec::with_capacity(count);
        let mut top_edge_ids = Vec::with_capacity(count);
        for (index, image) in images.iter().enumerate() {
            let next_index = (index + 1) % count;
            let (start_j, end_j) = if image.dir {
                (index, next_index)
            } else {
                (next_index, index)
            };
            edges.push(EdgeRecord {
                id: next_id,
                curve: image.bottom.clone(),
                t0: image.t0,
                t1: image.t1,
                start_vertex_id: bottom_vertex_ids[start_j],
                end_vertex_id: bottom_vertex_ids[end_j],
                degenerate: false,
                name: None,
            });
            bottom_edge_ids.push(next_id);
            next_id += 1;
            edges.push(EdgeRecord {
                id: next_id,
                curve: image.top.clone(),
                t0: image.t0,
                t1: image.t1,
                start_vertex_id: top_vertex_ids[start_j],
                end_vertex_id: top_vertex_ids[end_j],
                degenerate: false,
                name: None,
            });
            top_edge_ids.push(next_id);
            next_id += 1;
        }
        let mut vertical_edge_ids = Vec::with_capacity(count);
        for junction in 0..count {
            edges.push(EdgeRecord {
                id: next_id,
                curve: make_line(bottom_points[junction], top_points[junction])?,
                t0: 0.0,
                t1: 1.0,
                start_vertex_id: bottom_vertex_ids[junction],
                end_vertex_id: top_vertex_ids[junction],
                degenerate: false,
                name: None,
            });
            vertical_edge_ids.push(next_id);
            next_id += 1;
        }

        // ---- One ruled wall per boundary pcurve. ----
        //
        // The wall's s parameter is the edge parameter; its loop traverses
        // the bottom edge ALONG the stored loop direction.  With the top
        // sheet at the larger offset, W_w = Δd·n with Δd > 0, so the wall's
        // natural normal W_s × W_w points to the RIGHT of the walk — which
        // is outward for a CCW outer loop (material on the left) AND for a
        // CW hole loop (material on the left, void on the right).  Hence
        // same_sense = dir uniformly, with the loop winding to match.
        for (index, image) in images.iter().enumerate() {
            let next_index = (index + 1) % count;
            let wall = ruled_wall(&image.bottom, &image.top)?;
            let (s_start, s_end) = if image.dir {
                (image.t0, image.t1)
            } else {
                (image.t1, image.t0)
            };
            let mut coedges = Vec::with_capacity(4);
            for (edge_id, forward, pcurve) in [
                (
                    bottom_edge_ids[index],
                    image.dir,
                    parameter_line(s_start, 0.0, s_end, 0.0)?,
                ),
                (
                    vertical_edge_ids[next_index],
                    true,
                    parameter_line(s_end, 0.0, s_end, 1.0)?,
                ),
                (
                    top_edge_ids[index],
                    !image.dir,
                    parameter_line(s_end, 1.0, s_start, 1.0)?,
                ),
                (
                    vertical_edge_ids[index],
                    false,
                    parameter_line(s_start, 1.0, s_start, 0.0)?,
                ),
            ] {
                coedges.push(CoedgeRecord {
                    id: next_id,
                    edge_id,
                    forward,
                    pcurve,
                });
                next_id += 1;
            }
            let loop_id = next_id;
            next_id += 1;
            faces.push(FaceRecord {
                id: next_id,
                surface: wall,
                same_sense: image.dir,
                loops: vec![LoopRecord {
                    id: loop_id,
                    coedges,
                }],
                name: None,
            });
            next_id += 1;
        }

        // ---- Cap loops: verbatim pcurves on top, reversed on bottom. ----
        let mut top_coedges = Vec::with_capacity(count);
        for (index, image) in images.iter().enumerate() {
            top_coedges.push(CoedgeRecord {
                id: next_id,
                edge_id: top_edge_ids[index],
                forward: image.dir,
                pcurve: loop_curves[index].clone(),
            });
            next_id += 1;
        }
        top_cap_loops.push(LoopRecord {
            id: next_id,
            coedges: top_coedges,
        });
        next_id += 1;
        let mut bottom_coedges = Vec::with_capacity(count);
        for index in (0..count).rev() {
            bottom_coedges.push(CoedgeRecord {
                id: next_id,
                edge_id: bottom_edge_ids[index],
                forward: !images[index].dir,
                pcurve: loop_curves[index].reversed()?,
            });
            next_id += 1;
        }
        bottom_cap_loops.push(LoopRecord {
            id: next_id,
            coedges: bottom_coedges,
        });
        next_id += 1;
    }

    // Coincident junction vertices — v1's coincident-corner refusal
    // generalized: a repeated junction point pinches the boundary into a
    // non-manifold vertex (this also catches a hole touching the rim).
    // The junction ring's own extent, not its distance from the world origin:
    // the same sheet modelled 5 m out must be judged degenerate on the same
    // evidence.  For a two-junction boundary this is the pair's separation, so
    // the test still answers "coincident" only at a true collapse.
    let scale = crate::model_scale(bottom_junction_points.iter().copied());
    for first in 0..bottom_junction_points.len() {
        for second in first + 1..bottom_junction_points.len() {
            if bottom_junction_points[first]
                .sub(bottom_junction_points[second])
                .length()
                <= 1e-7 * scale
            {
                return Err(
                    "thickenSheet: sheet boundary is degenerate (coincident junction vertices)"
                        .into(),
                );
            }
        }
    }

    // Caps: outward is +n on the top sheet, −n on the bottom; the given
    // loop orientation matches same_sense = true, its reversal the bottom.
    faces.push(FaceRecord {
        id: next_id,
        surface: top,
        same_sense: true,
        loops: top_cap_loops,
        name: None,
    });
    next_id += 1;
    faces.push(FaceRecord {
        id: next_id,
        surface: bottom,
        same_sense: false,
        loops: bottom_cap_loops,
        name: None,
    });
    next_id += 1;

    let shell_id = next_id;
    let solid = BrepSolid {
        id: next_id + 1,
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: shell_id,
            faces,
        }],
        genus: loops.len() as i64 - 1,
    };
    let issues = solid.validate();
    if !issues.is_empty() {
        return Err(format!(
            "thickenSheet: assembled solid failed validation: {issues:?}"
        ));
    }
    let volume = crate::solid_signed_volume(&solid)?;
    if volume <= 0.0 {
        return Err(format!(
            "thickenSheet: internal orientation error (signed volume {volume})"
        ));
    }
    Ok(solid)
}

/// §5.9 THICKEN a sheet (an open surface patch over its full parameter
/// domain) into a closed solid.
///
/// Thin wrapper over [`thicken_trimmed_sheet`] passing the full-domain
/// rectangle as the (counter-clockwise) outer loop.  The result is a
/// genus-0 solid with 8 vertices, 12 edges, and 6 faces (offset caps + four
/// ruled walls), oriented outward and validated.  Refuses (Err) on:
/// zero/non-finite thickness, closed sheets (split at the seam first),
/// degenerate boundaries, and offsets that would self-intersect because a
/// concave curvature radius is smaller than the offset distance.
pub fn thicken_face_sheet(
    surface: &NurbsSurface,
    thickness: f64,
    symmetric: bool,
) -> Result<BrepSolid, String> {
    if !thickness.is_finite() || thickness.abs() <= 1e-12 {
        return Err("thickenSheet: thickness must be a nonzero finite value".into());
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let rectangle = vec![
        parameter_line(u0, v0, u1, v0)?,
        parameter_line(u1, v0, u1, v1)?,
        parameter_line(u1, v1, u0, v1)?,
        parameter_line(u0, v1, u0, v0)?,
    ];
    thicken_trimmed_sheet(surface, &[rectangle], thickness, symmetric)
}

// BREP private tests: 9d43676f8eb94b42
