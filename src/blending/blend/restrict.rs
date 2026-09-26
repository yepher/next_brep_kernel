//! Restricting a sewn stripe against a THIRD face it crosses.
//!
//! The march plus §6.9 surgery re-trims *the blended edge's two MATING faces
//! and nothing else*.  That is complete only while the swept volume — the
//! sliver a convex blend removes — touches nothing but those two mates.  Let a
//! third face cross it (a through hole under the rail, a pocket, a neighbouring
//! boss) and the blend wall comes out whole and running through the other face,
//! the mate keeps an inner loop that no longer lies inside its own trimmed
//! region, and the answer is self-intersecting.  `check_blend_interference`
//! (`fillet/analyze.rs`) states that precondition and refuses when it fails;
//! this module RESTRICTS the stripe instead, so the refusal — and the cutter
//! composition behind it — is not reached.
//!
//! # Where the crossing points come from
//!
//! Each end of the trim curve is the point where the blend surface, the mate
//! and the obstacle all meet: a three-surface point (Golovanov §4.5).  It is
//! not solved as one.  Two of the three pairwise intersection curves are
//! already in hand as TOPOLOGY — blend ∩ mate is the contact RAIL the march
//! just fitted, and mate ∩ obstacle is an EDGE the input solid already carried
//! — so the triple point drops to a curve-curve intersection between two
//! existing curves.  That is cheaper, better conditioned, and it lands the
//! point exactly ON both curves instead of a fit-accuracy away from each.
//!
//! It also decides the class this module covers.  An obstacle that meets
//! NEITHER mate — a void wholly under the blend's arc — has no such edge, and
//! is refused by name here; `check_blend_interference`'s surface-grid pass is
//! what still catches it, and it still routes to the cutter.
//!
//! # The trim curve
//!
//! With both endpoints known exactly, the curve between them is TRACED by the
//! kernel's own surface/surface marcher, seeded once in the middle of the
//! crossing window.  The march stops on the blend carrier's parametric domain
//! boundary, and that boundary IS the rail, so the traced branch's two ends are
//! the two crossings — the curve is clipped to the face by construction rather
//! than by a filter, and it comes back with a parameter pair on each carrier.
//! The exact crossings are then committed over the traced endpoints, because a
//! curve-curve intersection of two existing curves is sharper than anything the
//! march produces.
//!
//! A per-SECTION root solve was tried first — for each `u` across the window,
//! the single root of the obstacle's signed distance along that blend section
//! — and it is wrong at the ends.  On the measured through hole the trim curve
//! leaves each crossing and turns BACK in `u`: the bore is widest at z = 10,
//! half a millimetre past the rail at z = 9.5, so the curve has a vertical
//! tangent there and is not a graph over `u` at all.  A fit through per-section
//! roots cuts that turn off and converges at FIRST order — 1.7e-3 at 49
//! stations, still 5.7e-5 at 1483 — while the traced curve clears the kernel's
//! own SSI fit contract on its first pass.  The section solve survives as the
//! seed and as a structural guard: a section with no root, or more than one, is
//! a shape this construction does not describe and is refused by name.
//!
//! # What it rebuilds
//!
//! Three loops, sharing one new edge:
//!
//! * the BLEND's loop swaps the rail coedge for `rail-low, trim, rail-high`;
//! * the OBSTACLE's loop swaps the arc the blend ate for the same trim edge;
//! * the MATE's two loops MERGE — the rail's own loop and the loop the obstacle
//!   bounded become one, because the rail now runs into the hole and out of it.

use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, VertexRecord};
use crate::{
    build_pcurve_on_surface_range, intersect_curves, interpolate_curve,
    project_point_to_surface, KernelTolerances, NurbsCurve, Vec3,
};

/// The faces one sewn stripe contributed, as the orchestrator knows them.
pub(in crate::blend) struct StripeFaces {
    pub(in crate::blend) blend_face: u64,
    pub(in crate::blend) mates: [u64; 2],
    /// Which mates the stripe consumed whole (`SewnStripe::consumed`).  A
    /// consumed mate is no longer in the solid, so there is nothing on it to
    /// restrict; it stays in `mates` because it is still not an obstacle.
    pub(in crate::blend) consumed: [bool; 2],
    /// A CONVEX stripe removes a sliver; a concave one adds a pad.  Only the
    /// convex case is restricted here — a pad crossing a third face leaves a
    /// different topology (the pad closes over the obstacle rather than being
    /// notched by it), and guessing at it would be worse than the refusal.
    pub(in crate::blend) convex: bool,
}

/// How many obstacles one stripe/mate pair may carry before the search is
/// called a mis-detection rather than a shape.
const OBSTACLE_BUDGET: usize = 8;

/// Sections used only to prove the root along one blend section is UNIQUE.
const UNIQUENESS_SAMPLES: usize = 24;

/// Refinement passes, and the traced-point count at which the fit is called
/// impossible rather than under-refined.
const REFINEMENTS: usize = 7;
const STATION_BUDGET: usize = 40000;

/// Samples used to MEASURE a fit against its carriers.
const VERIFY: usize = 97;

/// Restrict every sewn stripe against the third faces it crosses.
///
/// `Ok(n)` restricted `n` obstacles (`n = 0` when nothing crosses).  `Err` is a
/// NAMED refusal: a crossing was found that this construction does not
/// describe, and the caller should treat the whole network result as refused.
pub(in crate::blend) fn restrict_third_faces(
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    stripes: &[StripeFaces],
    policy: &KernelTolerances,
) -> Result<usize, String> {
    let mut applied = 0usize;
    for stripe in stripes {
        let mut mates = Vec::with_capacity(2);
        for side in 0..2 {
            let mate = stripe.mates[side];
            if !stripe.consumed[side] && !mates.contains(&mate) {
                mates.push(mate);
            }
        }
        for mate in mates {
            for _ in 0..OBSTACLE_BUDGET {
                let Some(plan) = plan_restriction(
                    result,
                    stripe.blend_face,
                    mate,
                    &stripe.mates,
                    stripe.convex,
                    policy,
                )?
                else {
                    break;
                };
                apply_restriction(result, take_id, plan)?;
                applied += 1;
            }
        }
    }
    Ok(applied)
}

/// One obstacle, resolved against one (blend face, mate face) pair.
struct Restriction {
    blend_face: u64,
    mate_face: u64,
    obstacle_face: u64,
    /// The rail: the edge the blend and the mate share.
    rail_edge: u64,
    /// Where the obstacle's boundary crosses the rail, in the rail's own
    /// parameter, ordered.
    rail_parameters: [f64; 2],
    /// The mate's loop that carries the rail, and the loop the obstacle bounds.
    rail_loop: usize,
    obstacle_loop: usize,
    /// The obstacle's boundary edge on the mate, its position in that loop, and
    /// the sub-range of it the blend ate.
    obstacle_edge: u64,
    obstacle_position: usize,
    /// The obstacle edge's parameters bounding the run of it the blend ate,
    /// in the edge's own direction.  That run never passes through the edge's
    /// own endpoint: the arc that would is refused where it is detected.
    obstacle_parameters: [f64; 2],
    /// The two crossing points, matched to `rail_parameters`.
    points: [Vec3; 2],
    /// The trim curve, its pcurve on the blend and its pcurve on the obstacle,
    /// all running from `points[0]` to `points[1]`.
    trim_curve: NurbsCurve,
    trim_on_blend: NurbsCurve,
    trim_on_obstacle: NurbsCurve,
    /// The worst distance the fitted trim curve sits from either carrier.
    deviation: f64,
}

/// The edges a face's loops use, with the loop each one sits in.
fn face_edges(face: &FaceRecord) -> Vec<(usize, usize, u64)> {
    face.loops
        .iter()
        .enumerate()
        .flat_map(|(loop_index, loop_record)| {
            loop_record
                .coedges
                .iter()
                .enumerate()
                .map(move |(position, coedge)| (loop_index, position, coedge.edge_id))
        })
        .collect()
}

fn face_of<'a>(solid: &'a BrepSolid, id: u64) -> Option<&'a FaceRecord> {
    solid.shells.iter().flat_map(|shell| &shell.faces).find(|face| face.id == id)
}

fn edge_of<'a>(solid: &'a BrepSolid, id: u64) -> Option<&'a EdgeRecord> {
    solid.edges.iter().find(|edge| edge.id == id)
}

/// The outward normal of `face` at (u, v): the surface normal, flipped when the
/// face uses its carrier reversed.
fn outward_normal(face: &FaceRecord, u: f64, v: f64) -> Result<Vec3, String> {
    let normal = face.surface.normal(u, v)?;
    Ok(if face.same_sense { normal } else { normal.scale(-1.0) })
}

/// Signed distance from `point` to the obstacle's carrier: POSITIVE on the
/// material side (behind the face's outward normal is material, so a point
/// outside the solid across that face reads positive).
///
/// The sign is read from the face's own orientation rather than assumed, so a
/// bore (material outside the cylinder) and a boss (material inside it) are the
/// same expression.
fn material_side_distance(face: &FaceRecord, point: Vec3) -> Result<f64, String> {
    let projection = project_point_to_surface(&face.surface, point)?;
    let normal = outward_normal(face, projection.u, projection.v)?;
    let offset = point.sub(projection.point);
    // `-` because the outward normal points AWAY from material.
    Ok(-offset.dot(normal))
}

/// Walk a loop's pcurves into one uv polygon, in traversal order.
///
/// A coedge's pcurve is already parameterized in that coedge's traversal
/// direction, so the polygon is the concatenation of each pcurve sampled from
/// its own domain start to its own domain end.
fn loop_polygon(face: &FaceRecord, loop_index: usize) -> Result<Vec<[f64; 2]>, String> {
    const PER_COEDGE: usize = 24;
    let mut polygon = Vec::new();
    for coedge in &face.loops[loop_index].coedges {
        let [q0, q1] = coedge.pcurve.domain()?;
        for step in 0..PER_COEDGE {
            let q = q0 + (q1 - q0) * step as f64 / PER_COEDGE as f64;
            let point = coedge.pcurve.evaluate(q)?;
            polygon.push([point.x, point.y]);
        }
    }
    if polygon.len() < 3 {
        return Err("blend restriction: a loop has no uv polygon".into());
    }
    Ok(polygon)
}

/// Even-odd parity of `point` against a uv polygon.
fn polygon_contains(polygon: &[[f64; 2]], point: [f64; 2]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        let straddles = (current[1] > point[1]) != (previous[1] > point[1]);
        if straddles {
            let t = (point[1] - current[1]) / (previous[1] - current[1]);
            if point[0] < current[0] + t * (previous[0] - current[0]) {
                inside = !inside;
            }
        }
        previous = *current;
    }
    inside
}

/// The normalized position of `parameter` along an edge's own span.
fn along(edge: &EdgeRecord, parameter: f64) -> f64 {
    (parameter - edge.t0) / (edge.t1 - edge.t0)
}

/// The uv a coedge's pcurve reaches at `along` of its edge's span.
fn pcurve_at(coedge: &CoedgeRecord, along: f64) -> Result<[f64; 2], String> {
    let [q0, q1] = coedge.pcurve.domain()?;
    let walked = if coedge.forward { along } else { 1.0 - along };
    let point = coedge.pcurve.evaluate(q0 + (q1 - q0) * walked)?;
    Ok([point.x, point.y])
}

/// Plan the next restriction for one (blend, mate) pair, or `Ok(None)` when
/// nothing on that mate crosses the blend's rail.
fn plan_restriction(
    solid: &BrepSolid,
    blend_face_id: u64,
    mate_face_id: u64,
    stripe_mates: &[u64; 2],
    convex: bool,
    policy: &KernelTolerances,
) -> Result<Option<Restriction>, String> {
    let blend = face_of(solid, blend_face_id)
        .ok_or("blend restriction: the blend face is missing")?;
    let mate = face_of(solid, mate_face_id)
        .ok_or("blend restriction: the mate face is missing")?;
    let blend_edges: Vec<u64> = face_edges(blend).into_iter().map(|(_, _, id)| id).collect();

    // Every rail: an edge the blend and the mate share.  After one obstacle has
    // been restricted the rail is in pieces, so this is a set, not a single id.
    let mut rails: Vec<(usize, usize, u64)> = face_edges(mate)
        .into_iter()
        .filter(|(_, _, id)| blend_edges.contains(id))
        .collect();
    rails.sort_by_key(|(_, _, id)| *id);

    for (rail_loop, _rail_position, rail_id) in rails {
        let rail = edge_of(solid, rail_id)
            .ok_or("blend restriction: a rail edge vanished")?;
        let band = policy.pcurve_consistency;
        let mut hits: Vec<(usize, usize, u64, f64, f64, Vec3)> = Vec::new();
        for (loop_index, position, edge_id) in face_edges(mate) {
            if loop_index == rail_loop || edge_id == rail_id {
                continue;
            }
            let other = edge_of(solid, edge_id)
                .ok_or("blend restriction: a mate boundary edge vanished")?;
            for crossing in intersect_curves(&rail.curve, &other.curve, band)? {
                let inside_rail = within(crossing.s, rail.t0, rail.t1);
                let inside_other = within(crossing.t, other.t0, other.t1);
                if !inside_rail || !inside_other {
                    continue;
                }
                // A TANGENTIAL meeting is not a crossing: the rail grazes the
                // obstacle's boundary and passes no material through it, so
                // the rails alone still trim the blend correctly.  This is the
                // pinched bore mouth, where the corner setbacks pull the top
                // face in until its boundary just touches the rim — refusing
                // it would drop a selection the network already builds.
                if crossing.tangential {
                    continue;
                }
                hits.push((loop_index, position, edge_id, crossing.s, crossing.t, crossing.point));
            }
        }
        if hits.is_empty() {
            continue;
        }
        // One obstacle at a time: a rail may cross several, and each
        // restriction re-scans what is left of it.
        let mut edges: Vec<u64> = hits.iter().map(|hit| hit.2).collect();
        edges.sort_unstable();
        edges.dedup();
        for edge_id in edges {
            let group: Vec<_> = hits.iter().copied().filter(|hit| hit.2 == edge_id).collect();
            if group.len() != 2 {
                return Err(format!(
                    "blend restriction: the blend's rail {} crosses edge {edge_id} of face {} at \
                     {} points (expected two); a stripe that enters an obstacle without leaving \
                     it is not restricted directly",
                    rail.id,
                    mate.id,
                    group.len()
                ));
            }
        }
        let first = hits[0].2;
        let group: Vec<_> = hits.iter().copied().filter(|hit| hit.2 == first).collect();
        return plan_from_hits(
            solid,
            blend,
            mate,
            rail,
            rail_loop,
            &group,
            stripe_mates,
            convex,
            policy,
        )
        .map(Some);
    }
    Ok(None)
}

/// Strictly inside a span, by a margin that keeps a crossing at a rail END out
/// of the class (there the trim would degenerate into the stripe's own cap).
fn within(parameter: f64, low: f64, high: f64) -> bool {
    let margin = 1e-6 * (high - low).abs();
    parameter > low + margin && parameter < high - margin
}

fn plan_from_hits(
    solid: &BrepSolid,
    blend: &FaceRecord,
    mate: &FaceRecord,
    rail: &EdgeRecord,
    rail_loop: usize,
    hits: &[(usize, usize, u64, f64, f64, Vec3)],
    stripe_mates: &[u64; 2],
    convex: bool,
    policy: &KernelTolerances,
) -> Result<Restriction, String> {
    if hits.len() != 2 {
        return Err(format!(
            "blend restriction: the blend's rail {} crosses the rest of face {} at {} points \
             (expected two); a stripe entering or leaving more than one obstacle is not \
             restricted directly",
            rail.id,
            mate.id,
            hits.len()
        ));
    }
    let (loop_a, position_a, edge_a, _, _, _) = hits[0];
    let (loop_b, position_b, edge_b, _, _, _) = hits[1];
    if loop_a != loop_b || edge_a != edge_b || position_a != position_b {
        return Err(format!(
            "blend restriction: the blend's rail {} enters face {}'s obstacle on edge {edge_a} \
             and leaves it on edge {edge_b}; a crossing spanning more than one boundary edge is \
             not restricted directly",
            rail.id, mate.id
        ));
    }
    // Order by the RAIL's parameter: the trim curve runs the way the rail does.
    let mut ordered = [hits[0], hits[1]];
    if ordered[0].3 > ordered[1].3 {
        ordered.swap(0, 1);
    }
    let rail_parameters = [ordered[0].3, ordered[1].3];
    let points = [ordered[0].5, ordered[1].5];
    if (rail_parameters[1] - rail_parameters[0]).abs() <= 1e-6 * (rail.t1 - rail.t0).abs() {
        return Err(format!(
            "blend restriction: the blend's rail {} touches face {}'s obstacle at one point \
             rather than crossing it; a tangential contact removes nothing and is not restricted",
            rail.id, mate.id
        ));
    }

    let obstacle_edge = edge_of(solid, edge_a)
        .ok_or("blend restriction: the obstacle boundary edge vanished")?;
    let closed = obstacle_edge.start_vertex_id == obstacle_edge.end_vertex_id;

    // Which run of the obstacle's boundary did the blend EAT?  The one outside
    // the rail's own loop: that loop is the mate's material after the surgery,
    // and the rail is its new boundary there.
    let polygon = loop_polygon(mate, rail_loop)?;
    let coedge = &mate.loops[loop_a].coedges[position_a];
    let low = ordered[0].4.min(ordered[1].4);
    let high = ordered[0].4.max(ordered[1].4);
    let inner_mid = 0.5 * (low + high);
    let inner_outside = !polygon_contains(&polygon, pcurve_at(coedge, along(obstacle_edge, inner_mid))?);
    let obstacle_parameters = if inner_outside {
        [low, high]
    } else if closed {
        // The complement of [low, high] on a closed edge runs through its own
        // start/end vertex.  That vertex would have to go with the arc — the
        // obstacle's seam edge then loses its foot — so it is refused here.
        return Err(format!(
            "blend restriction: the arc of edge {} the blend ate runs through that edge's own \
             endpoint on face {}, so the obstacle's seam would lose its foot; only an arc \
             interior to one boundary edge is restricted directly",
            obstacle_edge.id, mate.id
        ));
    } else {
        return Err(format!(
            "blend restriction: neither run of edge {} between the rail crossings lies outside \
             the mate's own loop on face {}; the obstacle's boundary is not resolved",
            obstacle_edge.id, mate.id
        ));
    };

    // The obstacle: the OTHER face using that boundary edge.
    let mut obstacle_faces: Vec<u64> = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .filter(|face| face.id != mate.id)
        .filter(|face| {
            face.loops
                .iter()
                .any(|l| l.coedges.iter().any(|c| c.edge_id == obstacle_edge.id))
        })
        .map(|face| face.id)
        .collect();
    obstacle_faces.sort_unstable();
    obstacle_faces.dedup();
    let [obstacle_face] = obstacle_faces[..] else {
        return Err(format!(
            "blend restriction: edge {} is shared by {} faces besides the mate; the obstacle is \
             not a single face",
            obstacle_edge.id,
            obstacle_faces.len()
        ));
    };
    if obstacle_face == blend.id || stripe_mates.contains(&obstacle_face) {
        return Err(format!(
            "blend restriction: the face across edge {} is the stripe's own blend or mate ({}), \
             not a third face; a stripe crossing its own supports is not restricted here",
            obstacle_edge.id, obstacle_face
        ));
    }
    if !convex {
        return Err(format!(
            "blend restriction: the concave stripe on face {} crosses obstacle face \
             {obstacle_face}; a pad that runs over a third face closes OVER it rather than \
             being notched by it, and is not restricted directly",
            blend.id
        ));
    }
    let obstacle = face_of(solid, obstacle_face)
        .ok_or("blend restriction: the obstacle face is missing")?;

    let (trim_curve, trim_on_blend, trim_on_obstacle, deviation) = solve_trim_curve(
        blend,
        obstacle,
        rail,
        rail_parameters,
        points,
        policy,
    )?;

    Ok(Restriction {
        blend_face: blend.id,
        mate_face: mate.id,
        obstacle_face,
        rail_edge: rail.id,
        rail_parameters,
        rail_loop,
        obstacle_loop: loop_a,
        obstacle_edge: obstacle_edge.id,
        obstacle_position: position_a,
        obstacle_parameters,
        points,
        trim_curve,
        trim_on_blend,
        trim_on_obstacle,
        deviation,
    })
}

/// Solve the trim curve in the blend's own domain, between two known endpoints.
fn solve_trim_curve(
    blend: &FaceRecord,
    obstacle: &FaceRecord,
    rail: &EdgeRecord,
    rail_parameters: [f64; 2],
    points: [Vec3; 2],
    policy: &KernelTolerances,
) -> Result<(NurbsCurve, NurbsCurve, NurbsCurve, f64), String> {
    // The rail on the BLEND is an isoparm of its carrier: one of the two
    // parameters is constant along it, and that constant is the trim curve's
    // own start and end.
    let rail_coedge = blend
        .loops
        .iter()
        .flat_map(|l| &l.coedges)
        .find(|c| c.edge_id == rail.id)
        .ok_or("blend restriction: the blend face does not use its own rail")?;
    let ends = [
        pcurve_at(rail_coedge, along(rail, rail_parameters[0]))?,
        pcurve_at(rail_coedge, along(rail, rail_parameters[1]))?,
    ];
    let [du0, du1] = blend.surface.domain_u()?;
    let [dv0, dv1] = blend.surface.domain_v()?;
    let span_u = du1 - du0;
    let span_v = dv1 - dv0;
    let iso_band = 1e-6;
    let along_u = (ends[0][1] - ends[1][1]).abs() <= iso_band * span_v.abs();
    let along_v = (ends[0][0] - ends[1][0]).abs() <= iso_band * span_u.abs();
    if along_u == along_v {
        return Err(format!(
            "blend restriction: the contact rail {} is not an isoparm of the blend carrier \
             (its pcurve runs ({:.9}, {:.9}) to ({:.9}, {:.9})); the trim curve needs the \
             blend's own section direction",
            rail.id, ends[0][0], ends[0][1], ends[1][0], ends[1][1]
        ));
    }
    // `section` is the constant coordinate of the rail; `run` varies along it.
    let section_is_v = along_u;
    let section0 = if section_is_v { ends[0][1] } else { ends[0][0] };
    let (section_low, section_high) = if section_is_v { (dv0, dv1) } else { (du0, du1) };
    // The far side of the section: the OTHER rail's end of the domain.
    let section1 = if (section0 - section_low).abs() >= (section0 - section_high).abs() {
        section_low
    } else {
        section_high
    };
    let run = [
        if section_is_v { ends[0][0] } else { ends[0][1] },
        if section_is_v { ends[1][0] } else { ends[1][1] },
    ];

    let at = |run_value: f64, section_value: f64| -> Result<Vec3, String> {
        if section_is_v {
            blend.surface.evaluate(run_value, section_value)
        } else {
            blend.surface.evaluate(section_value, run_value)
        }
    };
    // The parameters must reproduce the crossing points, or the rail's pcurve
    // and its 3D fit disagree and everything below is built on the wrong uv.
    let seat_band = (policy.pcurve_consistency * 10.0).max(1e-9);
    for (index, point) in points.iter().enumerate() {
        let seated = at(run[index], section0)?;
        let miss = seated.sub(*point).length();
        if miss > seat_band {
            return Err(format!(
                "blend restriction: the rail crossing at ({:.6}, {:.6}, {:.6}) sits {miss:.3e} \
                 from the blend carrier's own point there (band {seat_band:.3e}); the rail's \
                 pcurve and its 3D fit disagree too far to trim against",
                point.x, point.y, point.z
            ));
        }
    }

    // Each section: one root of the obstacle's signed distance, and only one.
    let root_on_section = |run_value: f64| -> Result<f64, String> {
        let mut previous = material_side_distance(obstacle, at(run_value, section0)?)?;
        if previous >= 0.0 {
            return Err(format!(
                "blend restriction: the blend's rail at run {run_value:.6} is already on the \
                 material side of the obstacle, so the crossing window is not the region the \
                 blend ate"
            ));
        }
        let mut bracket: Option<[f64; 2]> = None;
        let mut crossings = 0usize;
        for step in 1..=UNIQUENESS_SAMPLES {
            let s = section0 + (section1 - section0) * step as f64 / UNIQUENESS_SAMPLES as f64;
            let value = material_side_distance(obstacle, at(run_value, s)?)?;
            if (value >= 0.0) != (previous >= 0.0) {
                crossings += 1;
                if bracket.is_none() {
                    let previous_s =
                        section0 + (section1 - section0) * (step - 1) as f64 / UNIQUENESS_SAMPLES as f64;
                    bracket = Some([previous_s, s]);
                }
            }
            previous = value;
        }
        let Some([mut low, mut high]) = bracket else {
            return Err(format!(
                "blend restriction: the blend's section at run {run_value:.6} never reaches the \
                 material side of the obstacle; the obstacle cuts the whole section and the \
                 stripe would be severed, which is not restricted directly"
            ));
        };
        if crossings != 1 {
            return Err(format!(
                "blend restriction: the blend's section at run {run_value:.6} crosses the \
                 obstacle {crossings} times; only a single crossing per section is restricted \
                 directly"
            ));
        }
        // Bisection: the bracket is a genuine sign change, so it converges
        // without a derivative and cannot leave the section.
        for _ in 0..80 {
            let middle = 0.5 * (low + high);
            let value = material_side_distance(obstacle, at(run_value, middle)?)?;
            if value >= 0.0 {
                high = middle;
            } else {
                low = middle;
            }
            if (high - low).abs() <= 1e-15 * (section1 - section0).abs().max(1.0) {
                break;
            }
        }
        Ok(0.5 * (low + high))
    };

    // The curve between the two crossings is TRACED, not solved section by
    // section.  A section solve was tried first and is wrong at the ends: on
    // the measured through hole the trim curve leaves each crossing and turns
    // BACK in the run parameter — the bore's widest point sits at z = 10, just
    // past the rail at z = 9.5 — so it has a vertical tangent there and is not
    // a graph over the run at all.  A fit through per-section roots cuts that
    // turn off and converges at first order (1.7e-3 at 49 stations, still
    // 5.7e-5 at 1483).  The kernel's own marcher walks arc length instead and
    // STOPS on the carrier's domain boundary, which is the rail: seeded once
    // in the middle, its two ends are the two crossings.
    let fit_band = policy.intersection_fit;
    let drift_band = policy.pcurve_consistency;
    let seed_run = 0.5 * (run[0] + run[1]);
    let seed_point = at(seed_run, root_on_section(seed_run)?)?;
    let reach = points[0].sub(points[1]).length().max(1.0);
    let mut step = reach / 32.0;
    let mut outcome: Option<(NurbsCurve, NurbsCurve, NurbsCurve, f64)> = None;
    let mut worst = f64::INFINITY;
    let mut traced = 0usize;
    for _pass in 0..REFINEMENTS {
        let options = crate::SurfaceIntersectionOptions {
            tolerance: fit_band,
            seed_points: vec![seed_point],
            seed_only: true,
            maximum_step: Some(step),
            maximum_steps: STATION_BUDGET,
            ..Default::default()
        };
        let branches = crate::intersect_surfaces(&blend.surface, &obstacle.surface, &options)?;
        let [branch] = &branches[..] else {
            return Err(format!(
                "blend restriction: the blend and obstacle face {} meet in {} branches through \
                 the crossing window; a trim curve that is not one arc is not restricted \
                 directly",
                obstacle.id,
                branches.len()
            ));
        };
        if branch.closed || branch.points.len() < 4 {
            return Err(format!(
                "blend restriction: the trim curve traced against obstacle face {} came back \
                 {} with {} points; it does not run from one rail crossing to the other",
                obstacle.id,
                if branch.closed { "closed" } else { "open" },
                branch.points.len()
            ));
        }
        // The march stops on the blend carrier's own domain boundary, which is
        // the rail.  Both ends must therefore BE the crossings; anything else
        // means the curve left the face somewhere this lane does not describe.
        let ends = [branch.points[0], branch.points[branch.points.len() - 1]];
        let straight = ends[0].sub(points[0]).length().max(ends[1].sub(points[1]).length());
        let swapped = ends[0].sub(points[1]).length().max(ends[1].sub(points[0]).length());
        let reversed = swapped < straight;
        let landing = straight.min(swapped);
        if landing > drift_band {
            return Err(format!(
                "blend restriction: the traced trim curve stops {landing:.3e} from the rail \
                 crossings it must join (band {drift_band:.3e}); it leaves the blend face \
                 somewhere other than the rail"
            ));
        }
        let mut spatial: Vec<Vec3> = branch.points.clone();
        if reversed {
            spatial.reverse();
        }
        // Commit the ends EXACTLY: they are curve-curve intersections of two
        // existing curves, which is sharper than anything the march produces.
        let last = spatial.len() - 1;
        spatial[0] = points[0];
        spatial[last] = points[1];
        traced = spatial.len();

        let parameters = chord_parameters(&spatial)?;
        let degree = 3.min(spatial.len() - 1);
        let trim_curve = interpolate_curve(&spatial, degree, &parameters)?;
        let [t0, t1] = trim_curve.domain()?;
        let trim_on_blend = build_pcurve_on_surface_range(
            &blend.surface,
            &trim_curve,
            t0,
            t1,
            true,
            fit_band,
        )?;
        let trim_on_obstacle = build_pcurve_on_surface_range(
            &obstacle.surface,
            &trim_curve,
            t0,
            t1,
            true,
            fit_band,
        )?;
        let [ou0, ou1] = obstacle.surface.domain_u()?;
        let [ov0, ov1] = obstacle.surface.domain_v()?;
        let [op0, op1] = trim_on_obstacle.domain()?;
        for index in 0..VERIFY {
            let a = trim_on_obstacle.evaluate(op0 + (op1 - op0) * index as f64 / VERIFY as f64)?;
            let b =
                trim_on_obstacle.evaluate(op0 + (op1 - op0) * (index + 1) as f64 / VERIFY as f64)?;
            if (b.x - a.x).abs() > 0.25 * (ou1 - ou0).abs()
                || (b.y - a.y).abs() > 0.25 * (ov1 - ov0).abs()
            {
                return Err(format!(
                    "blend restriction: the trim curve's pcurve on obstacle face {} jumps a \
                     large fraction of its domain, so it crosses that carrier's seam; a \
                     seam-crossing trim is not restricted directly",
                    obstacle.id
                ));
            }
        }

        // What the fit is worth, measured: how far the committed edge sits from
        // each carrier it is supposed to lie ON, and how far each pcurve's
        // image sits from that edge.
        let mut deviation: f64 = 0.0;
        let mut drift: f64 = 0.0;
        let mut off_carrier = [0.0f64; 2];
        let mut per_carrier = [0.0f64; 2];
        for index in 0..=VERIFY {
            let fraction = index as f64 / VERIFY as f64;
            let on_curve = trim_curve.evaluate(t0 + (t1 - t0) * fraction)?;
            for (which, (surface, pcurve)) in [
                (&blend.surface, &trim_on_blend),
                (&obstacle.surface, &trim_on_obstacle),
            ]
            .into_iter()
            .enumerate()
            {
                let off = project_point_to_surface(surface, on_curve)?.distance;
                off_carrier[which] = off_carrier[which].max(off);
                deviation = deviation.max(off);
                let [p0, p1] = pcurve.domain()?;
                let node = pcurve.evaluate(p0 + (p1 - p0) * fraction)?;
                let miss = surface.evaluate(node.x, node.y)?.sub(on_curve).length();
                per_carrier[which] = per_carrier[which].max(miss);
                drift = drift.max(miss);
            }
        }
        worst = deviation.max(drift);
        if std::env::var("BREP_DEBUG_NETWORK").is_ok() {
            eprintln!(
                "blend restriction: traced {traced} points at step {step:.4} — off-carrier \
                 blend={:.3e} obstacle={:.3e} drift blend={:.3e} obstacle={:.3e} \
                 (bands {fit_band:.3e} / {drift_band:.3e})",
                off_carrier[0], off_carrier[1], per_carrier[0], per_carrier[1],
            );
        }
        if deviation <= fit_band && drift <= drift_band {
            outcome = Some((trim_curve, trim_on_blend, trim_on_obstacle, worst));
            break;
        }
        step *= 0.5;
    }
    let Some((trim_curve, trim_on_blend, trim_on_obstacle, measured)) = outcome else {
        return Err(format!(
            "blend restriction: the trim curve could not be fitted to its carriers — {traced} \
             traced points still miss by {worst:.3e}, past the {fit_band:.3e} an SSI fit is \
             allowed; the restriction would be a new defect rather than a fix"
        ));
    };
    Ok((trim_curve, trim_on_blend, trim_on_obstacle, measured))
}

/// Chord-length parameters normalized to [0, 1].
fn chord_parameters(points: &[Vec3]) -> Result<Vec<f64>, String> {
    let mut total = 0.0;
    let mut running = vec![0.0];
    for pair in points.windows(2) {
        total += pair[1].sub(pair[0]).length();
        running.push(total);
    }
    if !(total > 0.0) {
        return Err("blend restriction: the trim curve's stations collapse to a point".into());
    }
    Ok(running.into_iter().map(|value| value / total).collect())
}

/// Split one coedge into three at two normalized positions along its edge.
fn split_coedge_twice(
    coedge: &CoedgeRecord,
    first: f64,
    second: f64,
    edges: [u64; 3],
    take_id: &mut dyn FnMut() -> u64,
) -> Result<[CoedgeRecord; 3], String> {
    let [q0, q1] = coedge.pcurve.domain()?;
    let walk = |value: f64| if coedge.forward { value } else { 1.0 - value };
    let mut cuts = [walk(first), walk(second)];
    if cuts[0] > cuts[1] {
        cuts.swap(0, 1);
    }
    let (head, rest) = coedge.pcurve.split(q0 + (q1 - q0) * cuts[0])?;
    let (middle, tail) = rest.split(q0 + (q1 - q0) * cuts[1])?;
    // The walk meets the edge pieces in the edge's own order when forward.
    let order = if coedge.forward {
        [edges[0], edges[1], edges[2]]
    } else {
        [edges[2], edges[1], edges[0]]
    };
    Ok([
        CoedgeRecord {
            id: coedge.id,
            edge_id: order[0],
            forward: coedge.forward,
            pcurve: head,
        },
        CoedgeRecord {
            id: take_id(),
            edge_id: order[1],
            forward: coedge.forward,
            pcurve: middle,
        },
        CoedgeRecord {
            id: take_id(),
            edge_id: order[2],
            forward: coedge.forward,
            pcurve: tail,
        },
    ])
}

/// Cut an edge in three at two parameters, minting the pieces and the two
/// crossing vertices, and splice every loop that uses it.
fn cut_edge_in_three(
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    edge_id: u64,
    parameters: [f64; 2],
    vertices: [u64; 2],
) -> Result<[u64; 3], String> {
    let edge = edge_of(result, edge_id)
        .ok_or("blend restriction: an edge vanished before its cut")?
        .clone();
    let (low_curve, rest) = edge.curve.split(parameters[0])?;
    let (middle_curve, high_curve) = rest.split(parameters[1])?;
    let ids = [take_id(), take_id(), take_id()];
    for (index, (curve, span, ends)) in [
        (low_curve, [edge.t0, parameters[0]], [edge.start_vertex_id, vertices[0]]),
        (middle_curve, [parameters[0], parameters[1]], [vertices[0], vertices[1]]),
        (high_curve, [parameters[1], edge.t1], [vertices[1], edge.end_vertex_id]),
    ]
    .into_iter()
    .enumerate()
    {
        result.edges.push(EdgeRecord {
            id: ids[index],
            curve,
            t0: span[0],
            t1: span[1],
            start_vertex_id: ends[0],
            end_vertex_id: ends[1],
            degenerate: false,
            name: edge.name.clone(),
        });
    }
    let first = along(&edge, parameters[0]);
    let second = along(&edge, parameters[1]);
    let uses: usize = result
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .flat_map(|face| &face.loops)
        .map(|l| l.coedges.iter().filter(|c| c.edge_id == edge_id).count())
        .sum();
    if uses != 2 {
        return Err(format!(
            "blend restriction: edge {edge_id} has {uses} coedges, not two; a seam or a \
             non-manifold use is not cut here"
        ));
    }
    let mut spliced = 0usize;
    for shell in &mut result.shells {
        for face in &mut shell.faces {
            for loop_record in &mut face.loops {
                let Some(position) = loop_record.coedges.iter().position(|c| c.edge_id == edge_id)
                else {
                    continue;
                };
                let pieces =
                    split_coedge_twice(&loop_record.coedges[position], first, second, ids, take_id)?;
                loop_record.coedges.splice(position..=position, pieces);
                spliced += 1;
            }
        }
    }
    if spliced != 2 {
        return Err(format!(
            "blend restriction: edge {edge_id} spliced into {spliced} loops, not two"
        ));
    }
    result.edges.retain(|candidate| candidate.id != edge_id);
    Ok(ids)
}

fn apply_restriction(
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    plan: Restriction,
) -> Result<(), String> {
    // The two crossing vertices, shared by every piece below.
    let crossing_vertices = [take_id(), take_id()];
    for (index, id) in crossing_vertices.iter().enumerate() {
        result.vertices.push(VertexRecord {
            id: *id,
            point: plan.points[index],
        });
    }
    // Which crossing does the obstacle's LOW parameter meet?
    let obstacle_edge = edge_of(result, plan.obstacle_edge)
        .ok_or("blend restriction: the obstacle boundary edge vanished")?;
    let low_point = obstacle_edge.curve.evaluate(plan.obstacle_parameters[0])?;
    let low_meets_first =
        low_point.sub(plan.points[0]).length() <= low_point.sub(plan.points[1]).length();
    let obstacle_vertices = if low_meets_first {
        [crossing_vertices[0], crossing_vertices[1]]
    } else {
        [crossing_vertices[1], crossing_vertices[0]]
    };

    let rail_pieces = cut_edge_in_three(
        result,
        take_id,
        plan.rail_edge,
        plan.rail_parameters,
        crossing_vertices,
    )?;
    let obstacle_pieces = cut_edge_in_three(
        result,
        take_id,
        plan.obstacle_edge,
        plan.obstacle_parameters,
        obstacle_vertices,
    )?;

    // The trim edge, running from crossing 0 to crossing 1.
    let trim_edge = take_id();
    let [t0, t1] = plan.trim_curve.domain()?;
    result.edges.push(EdgeRecord {
        id: trim_edge,
        curve: plan.trim_curve.clone(),
        t0,
        t1,
        start_vertex_id: crossing_vertices[0],
        end_vertex_id: crossing_vertices[1],
        degenerate: false,
        name: None,
    });

    // --- The blend face: the rail's middle piece becomes the trim edge. ---
    replace_coedge(
        result,
        plan.blend_face,
        rail_pieces[1],
        trim_edge,
        &plan.trim_on_blend,
        crossing_vertices,
    )?;
    // --- The obstacle face: the eaten arc becomes the same trim edge. ---
    replace_coedge(
        result,
        plan.obstacle_face,
        obstacle_pieces[1],
        trim_edge,
        &plan.trim_on_obstacle,
        crossing_vertices,
    )?;

    // --- The mate: the rail's middle piece and the eaten arc both go, and the
    //     two loops they left open become one. ---
    merge_mate_loops(result, plan.mate_face, rail_pieces[1], obstacle_pieces[1])?;

    result
        .edges
        .retain(|edge| edge.id != rail_pieces[1] && edge.id != obstacle_pieces[1]);
    for face in [plan.blend_face, plan.obstacle_face, plan.mate_face] {
        check_loops_closed(result, face)?;
    }
    if std::env::var("BREP_DEBUG_NETWORK").is_ok() {
        eprintln!(
            "blend restriction: face {} trimmed against obstacle face {} along edge {} \
             (mate {} loops {} and {}, position {}) — fit {:.3e}",
            plan.blend_face,
            plan.obstacle_face,
            trim_edge,
            plan.mate_face,
            plan.rail_loop,
            plan.obstacle_loop,
            plan.obstacle_position,
            plan.deviation,
        );
    }
    Ok(())
}

/// Swap one coedge for the trim edge, keeping the loop's traversal sense.
fn replace_coedge(
    result: &mut BrepSolid,
    face_id: u64,
    old_edge: u64,
    trim_edge: u64,
    pcurve: &NurbsCurve,
    trim_vertices: [u64; 2],
) -> Result<(), String> {
    let old = edge_of(result, old_edge)
        .ok_or("blend restriction: the piece being replaced vanished")?;
    // The trim edge runs `trim_vertices[0] -> trim_vertices[1]`, and the piece
    // it replaces runs between the same two vertices, so the walk decides the
    // sense: a coedge that STARTS at `trim_vertices[0]` uses the trim forward.
    let (old_start, old_end) = (old.start_vertex_id, old.end_vertex_id);
    let face = result
        .shells
        .iter_mut()
        .flat_map(|shell| &mut shell.faces)
        .find(|face| face.id == face_id)
        .ok_or("blend restriction: a face vanished before its splice")?;
    for loop_record in &mut face.loops {
        let Some(position) = loop_record.coedges.iter().position(|c| c.edge_id == old_edge) else {
            continue;
        };
        let walked_from = if loop_record.coedges[position].forward {
            old_start
        } else {
            old_end
        };
        let forward = walked_from == trim_vertices[0];
        loop_record.coedges[position] = CoedgeRecord {
            id: loop_record.coedges[position].id,
            edge_id: trim_edge,
            forward,
            pcurve: if forward { pcurve.clone() } else { pcurve.reversed()? },
        };
        return Ok(());
    }
    Err(format!(
        "blend restriction: face {face_id} does not use edge {old_edge}"
    ))
}

/// Merge the mate's rail loop and obstacle loop into one, dropping the two
/// coedges the restriction removed.
fn merge_mate_loops(
    result: &mut BrepSolid,
    mate_face: u64,
    rail_middle: u64,
    obstacle_middle: u64,
) -> Result<(), String> {
    let face = result
        .shells
        .iter_mut()
        .flat_map(|shell| &mut shell.faces)
        .find(|face| face.id == mate_face)
        .ok_or("blend restriction: the mate face vanished before its merge")?;
    let Some(rail_loop) = face
        .loops
        .iter()
        .position(|l| l.coedges.iter().any(|c| c.edge_id == rail_middle))
    else {
        return Err("blend restriction: the mate lost the rail's middle piece".into());
    };
    let Some(obstacle_loop) = face
        .loops
        .iter()
        .position(|l| l.coedges.iter().any(|c| c.edge_id == obstacle_middle))
    else {
        return Err("blend restriction: the mate lost the obstacle's eaten arc".into());
    };
    if rail_loop == obstacle_loop {
        return Err(
            "blend restriction: the rail and the obstacle arc are already in one loop on the \
             mate; the merge has nothing to join"
                .into(),
        );
    }
    // Rotate each loop so the removed coedge sits LAST, then drop it: what is
    // left of each is an open chain, and the two chains join end to end.
    let mut rail_chain = face.loops[rail_loop].coedges.clone();
    let rail_at = rail_chain
        .iter()
        .position(|c| c.edge_id == rail_middle)
        .expect("located above");
    let rail_length = rail_chain.len();
    rail_chain.rotate_left((rail_at + 1) % rail_length);
    rail_chain.pop();
    let mut obstacle_chain = face.loops[obstacle_loop].coedges.clone();
    let obstacle_at = obstacle_chain
        .iter()
        .position(|c| c.edge_id == obstacle_middle)
        .expect("located above");
    let obstacle_length = obstacle_chain.len();
    obstacle_chain.rotate_left((obstacle_at + 1) % obstacle_length);
    obstacle_chain.pop();
    rail_chain.extend(obstacle_chain);
    face.loops[rail_loop].coedges = rail_chain;
    face.loops.remove(obstacle_loop);
    Ok(())
    // Whether the two chains meet end to end is not asserted here: the caller
    // walks every touched loop through `check_loops_closed`, which names the
    // pair of vertices that failed to meet rather than a bare orientation
    // claim.
}

/// Every loop of a face must walk vertex to vertex: each coedge's finishing
/// vertex is the next coedge's starting one.  Checked here rather than left to
/// `validate`, because a merge that joins the wrong ends still validates as
/// two-use incidence.
fn check_loops_closed(result: &BrepSolid, face_id: u64) -> Result<(), String> {
    let face = face_of(result, face_id)
        .ok_or("blend restriction: a face vanished before its loop check")?;
    for (loop_index, loop_record) in face.loops.iter().enumerate() {
        let mut walk: Vec<(u64, u64)> = Vec::with_capacity(loop_record.coedges.len());
        for coedge in &loop_record.coedges {
            let edge = edge_of(result, coedge.edge_id).ok_or_else(|| {
                format!(
                    "blend restriction: face {face_id} loop {loop_index} uses missing edge {}",
                    coedge.edge_id
                )
            })?;
            walk.push(if coedge.forward {
                (edge.start_vertex_id, edge.end_vertex_id)
            } else {
                (edge.end_vertex_id, edge.start_vertex_id)
            });
        }
        for index in 0..walk.len() {
            let next = (index + 1) % walk.len();
            if walk[index].1 != walk[next].0 {
                return Err(format!(
                    "blend restriction: face {face_id} loop {loop_index} does not close — \
                     coedge {} finishes at vertex {} and the next starts at {}",
                    loop_record.coedges[index].edge_id, walk[index].1, walk[next].0
                ));
            }
        }
    }
    Ok(())
}
