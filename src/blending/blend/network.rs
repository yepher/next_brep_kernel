//! The stripe network: a whole selection blended as one construction.
//!
//! # Why this is not a loop over single-edge fillets
//!
//! Blending one edge at a time means every blend after the first is built
//! against a solid the previous ones have already cut, and a corner where two
//! or three of them meet has to be RECONSTRUCTED afterwards from whatever they
//! left behind.  That is the origin of both classic failures: a cutter run
//! past its edge to make a boolean assemble removes material the fillet never
//! touches, and a corner closure that has to identify leftover caps by
//! distance heuristics silently gives up and leaves the corner unmade.
//!
//! Here nothing is cut until everything is known.  Every stripe is marched
//! against the ORIGINAL solid, so the result cannot depend on selection order;
//! every corner is solved from the corner ball ([`solve_corner_ball`]) before
//! any topology changes; and each stripe is then trimmed to the station where
//! its own rolling ball IS the corner ball.  At that station the stripe's
//! contact points ARE the ball's tangency points, so:
//!
//!   * the two stripes sharing a face end their contact rails at the same
//!     point and the face's loop closes with no connector and no leftover;
//!   * at a STAR (every edge at the vertex selected) the stripe's end
//!     cross-section is an arc of the corner sphere, so it can be the shared
//!     edge with the corner patch — exactly tangent, with nothing to intersect;
//!   * at a MITER (two edges selected, the third sharp) the two blends run on
//!     past the ball and are trimmed against each other along their seam,
//!     marched as a level set from the ball's tangency point (`miter.rs`);
//!   * no boolean is involved anywhere, so no cutter can overshoot.
//!
//! Every vertex of the selection is classified once — free end, miter, or
//! star — so a selection mixing them (a face perimeter with one star corner)
//! is one construction, not a composition of per-corner special cases.
//! Configurations the lane does not construct refuse BY NAME: a smooth (G1)
//! two-edge vertex, which is a chain and not a corner; mixed-convexity
//! corners (their closure is a torus sector, not a sphere); re-entrant
//! miters (the blends wrap the concave edge instead of meeting in a seam);
//! no-common-ball stars.  A CHAMFER reaches the lane too: its stripes are the
//! same march with a chord section (`cross_section_basis`), and its STAR closes
//! with the planar facet those chords bound (`build_chamfer_corner_facet`).
//! Its other corners refuse by name, one message per vertex, and take the
//! cutter composition.

use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, VertexRecord};
use crate::{NurbsCurve, Vec3};

use super::corner::*;
use super::edge::*;
use super::miter::*;
use super::stations::*;

/// One selected edge, marched and fitted against the original solid.
struct Stripe<'a> {
    edge: &'a EdgeRecord,
    first: BlendMate<'a>,
    second: BlendMate<'a>,
    rows: FittedRows,
    name: Option<String>,
}

impl Stripe<'_> {
    /// The blend loop's walk over an end slot runs first-rim -> second-rim
    /// exactly when this is true (see `build_open_surgery`).
    fn walk_first_to_second(&self, at_start: bool) -> bool {
        let blend_cr_forward = !self.first.coedge.forward;
        if at_start {
            !blend_cr_forward
        } else {
            blend_cr_forward
        }
    }

    fn at_start(&self, vertex: u64) -> bool {
        self.edge.start_vertex_id == vertex
    }
}

/// A seated corner ball with its tangency vertices committed.
struct Corner {
    center: Vec3,
    /// True when the ball sits inside the material (every incident edge
    /// convex); false when it sits in the air (every incident edge concave).
    outward: bool,
    /// Face id -> the ball's tangency point on that face, and the vertex id
    /// committed for it.
    tangency: Vec<(u64, Vec3, u64)>,
}

impl Corner {
    fn tangency_on(&self, face_id: u64) -> Result<(Vec3, u64), String> {
        self.tangency
            .iter()
            .find(|(id, _, _)| *id == face_id)
            .map(|(_, point, vertex)| (*point, *vertex))
            .ok_or_else(|| {
                format!("blend network: the corner ball has no tangency on face {face_id}")
            })
    }
}

/// What the selection does at one vertex.
enum VertexKind<'a> {
    /// A star: every edge at the vertex is selected; closed by a patch of the
    /// corner ball.
    Star { corner: Corner },
    /// A miter: two selected edges, the third sharp.  `shared` is the face
    /// both stripes touch; `sharp` the unselected edge; the ball still seats
    /// on the three faces and supplies P, the point both shared-face rails
    /// stop at.
    Miter {
        corner: Corner,
        shared: &'a FaceRecord,
        sharp: &'a EdgeRecord,
    },
    /// A flush join: two selected edges meeting SMOOTHLY (G1).  The rolling
    /// ball is the same ball on the same (or tangent-continuous) faces on
    /// both sides, so the two sections at the vertex coincide: both stripes
    /// stop at their own vertex station and share ONE arc, with no patch.
    /// When the two stripes' side faces differ (a planar wall running into a
    /// cylindrical one), the seam between those faces is trimmed at the
    /// shared side rim.
    Flush,
    /// A re-entrant corner: two selected CONVEX edges on a cap, meeting where
    /// the cap's perimeter turns inward (their third edge concave).  The two
    /// blends never meet in a seam — the ball rolls round the concave edge
    /// instead, touching it at one point, and sweeps a HORN TORUS (major
    /// radius = minor radius = r) whose pole is that point.  Each stripe stops
    /// where its wall rail reaches the concave edge; the sector between their
    /// end sections is the closure.
    Reentrant {
        cap: &'a FaceRecord,
        sharp: &'a EdgeRecord,
    },
}

/// The overshoot ladder for the marches, as fractions of the edge span.  A
/// stripe that ends at a corner needs no overshoot there (its stop station is
/// INSIDE the edge, a tangency setback short of the vertex); the ladder is
/// still needed for free ends, whose support crossings sit at or past the
/// endpoint.
const OVERSHOOTS: [f64; 4] = [0.08, 0.16, 0.28, 0.45];

/// Unit tangent of `edge` at `vertex`, pointing AWAY from it.
fn tangent_away_from(edge: &EdgeRecord, vertex: u64) -> Result<Vec3, String> {
    let (t, sign) = if edge.start_vertex_id == vertex {
        (edge.t0, 1.0)
    } else {
        (edge.t1, -1.0)
    };
    let derivatives = edge.curve.derivatives(t, 1)?;
    derivatives[1].normalized().map(|tangent| tangent.scale(sign))
}

/// The two mating faces of every selected edge, with the signed radius the
/// march offsets each of them by.  Read off the ORIGINAL solid: `rho` is the
/// side the rolling ball sits on, so it is the edge's convexity in the form
/// the corner solve needs.
fn stripe_mates<'a>(
    solid: &'a BrepSolid,
    edge_ids: &[u64],
    radius: f64,
) -> Result<Vec<(&'a EdgeRecord, BlendMate<'a>, BlendMate<'a>)>, String> {
    let mut mates: Vec<(&EdgeRecord, BlendMate, BlendMate)> = Vec::with_capacity(edge_ids.len());
    for &edge_id in edge_ids {
        let edge = solid
            .edges
            .iter()
            .find(|candidate| candidate.id == edge_id)
            .ok_or_else(|| format!("blend network: edge {edge_id} not found"))?;
        if edge.start_vertex_id == edge.end_vertex_id {
            return Err(format!(
                "blend network: edge {edge_id} is closed and has no corner to share"
            ));
        }
        let (first_face, first_loop, first_coedge) = locate_mate(solid, edge_id, None)?;
        let (second_face, second_loop, second_coedge) =
            locate_mate(solid, edge_id, Some((first_face.id, first_loop)))?;
        let (rho1, rho2) = signed_radii(
            edge,
            first_face,
            first_coedge,
            second_face,
            second_coedge,
            radius,
        )?;
        mates.push((
            edge,
            BlendMate {
                face: first_face,
                coedge: first_coedge,
                loop_index: first_loop,
                rho: rho1,
            },
            BlendMate {
                face: second_face,
                coedge: second_coedge,
                loop_index: second_loop,
                rho: rho2,
            },
        ));
    }
    Ok(mates)
}

/// Which selected edges (indices into `mates`) meet at which vertex.
fn stripes_by_vertex(mates: &[(&EdgeRecord, BlendMate, BlendMate)]) -> Vec<(u64, Vec<usize>)> {
    let mut vertex_stripes: Vec<(u64, Vec<usize>)> = Vec::new();
    for (index, (edge, _, _)) in mates.iter().enumerate() {
        for vertex in [edge.start_vertex_id, edge.end_vertex_id] {
            match vertex_stripes.iter_mut().find(|(id, _)| *id == vertex) {
                Some((_, list)) => list.push(index),
                None => vertex_stripes.push((vertex, vec![index])),
            }
        }
    }
    vertex_stripes
}

/// A corner where the selection mixes convexity: at `point` two of the
/// selected edges share a face but want the rolling ball on OPPOSITE sides of
/// it, one blend cutting material away from that face while the other adds
/// material against it.  `convex` and `concave` are indices into the selection.
pub(crate) struct MixedCorner {
    pub(crate) point: Vec3,
    pub(crate) convex: Vec<usize>,
    pub(crate) concave: Vec<usize>,
}

/// The one network refusal the fillet ladder must not paper over: a corner
/// where MORE THAN ONE selected edge is concave and at least one is convex.
///
/// Mixing convexity at a vertex is not by itself unbuildable. The re-entrant
/// vertex of a notch — ONE concave edge with the convex edges that run into
/// it — is closed by the horn-torus sector about that concave edge's own
/// blend, because the ball can roll round it from one convex blend to the
/// other (`round_concave_chain_corner`, `round_convex_corner`, §6.9.7). Its
/// MIRROR — two or more concave edges meeting a convex one — has no closure
/// here at all: the convex blend RUNS OUT against the concave beads partway
/// along its edge, which is the vertex blend nothing in this kernel
/// constructs (see `docs/developer/kernel-plans/fillet-stripe-network.md`).
/// That boundary is empirical, one concave edge against two, not a rule
/// derived from the balls' sides — the notch splits those the same way and is
/// built.
///
/// So `fillet_edges` reports that class as a TERMINAL refusal rather than
/// retrying it on the cutter, whose answer for it is a shredded solid: sliver
/// end caps at every unclosed corner and a renamed carrier face.
pub(crate) fn mixed_convexity_corner(
    solid: &BrepSolid,
    edge_ids: &[u64],
    radius: f64,
) -> Option<MixedCorner> {
    let mates = stripe_mates(solid, edge_ids, radius).ok()?;
    for (vertex_id, stripe_indices) in stripes_by_vertex(&mates) {
        if stripe_indices.len() < 2 {
            continue;
        }
        let mut faces: Vec<(u64, f64)> = Vec::new();
        let mut mixed = false;
        for &index in &stripe_indices {
            for mate in [&mates[index].1, &mates[index].2] {
                match faces.iter().find(|(id, _)| *id == mate.face.id) {
                    Some((_, rho)) => mixed |= (*rho - mate.rho).abs() > 1e-9 * (1.0 + radius),
                    None => faces.push((mate.face.id, mate.rho)),
                }
            }
        }
        if !mixed {
            continue;
        }
        let Some(point) = solid
            .vertices
            .iter()
            .find(|vertex| vertex.id == vertex_id)
            .map(|vertex| vertex.point)
        else {
            continue;
        };
        let mut convex = Vec::new();
        let mut concave = Vec::new();
        for &index in &stripe_indices {
            let (edge, first, second) = &mates[index];
            match edge_is_convex(edge, first, second) {
                Ok(true) => convex.push(index),
                Ok(false) => concave.push(index),
                // Unclassifiable: the corner is still mixed, it just cannot be
                // described edge by edge.  Report it with the side left empty.
                Err(_) => {}
            }
        }
        // Exactly one concave edge is the re-entrant closure's own shape, and
        // it is constructed; only its mirror is refused.
        if concave.len() < 2 || convex.is_empty() {
            continue;
        }
        return Some(MixedCorner {
            point,
            convex,
            concave,
        });
    }
    None
}

/// Blend every edge in `edge_ids` as one network.
///
/// `edge_names` is parallel to `edge_ids` (the per-edge blend-face name);
/// `corner_name` names each star patch after the input indices meeting there.
/// Errors — always named — leave the caller's solid untouched.
pub(crate) fn blend_star_network(
    solid: &BrepSolid,
    edge_ids: &[u64],
    radius: f64,
    chamfer: bool,
    edge_names: &[Option<String>],
    corner_name: &dyn Fn(&[usize]) -> Option<String>,
) -> Result<BrepSolid, String> {
    if edge_ids.is_empty() {
        return Err("blend network: no edges selected".into());
    }
    if !(radius > 0.0) || !radius.is_finite() {
        return Err("blend network: radius must be positive".into());
    }
    // ---- 1. Mates and signed radii, from the ORIGINAL solid. ----
    let mates = stripe_mates(solid, edge_ids, radius)?;

    // ---- 2. Which selected edges meet at which vertex. ----
    let vertex_stripes = stripes_by_vertex(&mates);

    // ---- 3. Seat a ball in every shared vertex, before anything is cut. ----
    let mut result = solid.clone();
    let mut take_id = fresh_id_source(solid);
    let scale = solid
        .vertices
        .iter()
        .fold(0.0f64, |worst, vertex| worst.max(vertex.point.length()))
        .max(1.0);
    let mut corners: Vec<(u64, Vec<usize>, VertexKind)> = Vec::new();
    for (vertex_id, stripe_indices) in &vertex_stripes {
        if stripe_indices.len() < 2 {
            continue;
        }
        let point = solid
            .vertices
            .iter()
            .find(|vertex| vertex.id == *vertex_id)
            .ok_or_else(|| format!("blend network: vertex {vertex_id} missing"))?
            .point;
        // The faces around the corner are the mates of the stripes that end
        // there, with the signed offset each stripe was marched at.
        let mut faces: Vec<(&FaceRecord, f64, usize)> = Vec::new();
        for &index in stripe_indices {
            for mate in [&mates[index].1, &mates[index].2] {
                match faces.iter_mut().find(|(face, _, _)| face.id == mate.face.id) {
                    Some((_, rho, uses)) => {
                        if (*rho - mate.rho).abs() > 1e-9 * (1.0 + radius) {
                            return Err(format!(
                                "blend network: the stripes at vertex {vertex_id} disagree on \
                                 which side of face {} the ball sits (mixed convexity)",
                                mate.face.id
                            ));
                        }
                        *uses += 1;
                    }
                    None => faces.push((mate.face, mate.rho, 1)),
                }
            }
        }
        let is_star = stripe_indices.len() >= 3;
        if is_star && faces.len() != stripe_indices.len() {
            return Err(format!(
                "blend network: vertex {vertex_id} has {} selected edges around {} faces — \
                 not every edge at the corner is selected, so no ball seats there",
                stripe_indices.len(),
                faces.len()
            ));
        }
        // A two-edge vertex: a G1 continuation is a CHAIN, not a corner (one
        // stripe should run through it — `blend_smooth_chain`); otherwise the
        // two stripes must share exactly one face with one sharp edge
        // between the other two.
        let mut miter: Option<(&FaceRecord, &EdgeRecord)> = None;
        if !is_star {
            let [a, b] = [stripe_indices[0], stripe_indices[1]];
            let tangent_a = tangent_away_from(mates[a].0, *vertex_id)?;
            let tangent_b = tangent_away_from(mates[b].0, *vertex_id)?;
            if tangent_a.dot(tangent_b) <= -(1.0 - 1e-6) {
                corners.push((*vertex_id, stripe_indices.clone(), VertexKind::Flush));
                continue;
            }
            if faces.len() != 3 {
                return Err(format!(
                    "blend network: vertex {vertex_id} has two selected edges around {} faces \
                     (expected exactly three: one shared, one on each side)",
                    faces.len()
                ));
            }
            let shared = faces
                .iter()
                .find(|(_, _, uses)| *uses == 2)
                .map(|(face, _, _)| *face)
                .ok_or_else(|| {
                    format!("blend network: the two edges at vertex {vertex_id} share no face")
                })?;
            let others: Vec<&FaceRecord> = faces
                .iter()
                .filter(|(face, _, _)| face.id != shared.id)
                .map(|(face, _, _)| *face)
                .collect();
            // The sharp edge borders both non-shared faces at the vertex.
            let (other_a, other_b) = (others[0], others[1]);
            let sharp_a = boundary_edge_at_vertex(solid, other_a, *vertex_id, mates[a].0.id)
                .or_else(|_| boundary_edge_at_vertex(solid, other_a, *vertex_id, mates[b].0.id))?;
            let sharp_b = boundary_edge_at_vertex(solid, other_b, *vertex_id, mates[a].0.id)
                .or_else(|_| boundary_edge_at_vertex(solid, other_b, *vertex_id, mates[b].0.id))?;
            if sharp_a != sharp_b {
                return Err(format!(
                    "blend network: vertex {vertex_id} has more than one unselected edge \
                     between the side faces (found {sharp_a} and {sharp_b})"
                ));
            }
            let sharp = solid
                .edges
                .iter()
                .find(|edge| edge.id == sharp_a)
                .ok_or("blend network: the sharp edge vanished")?;
            if sharp.start_vertex_id == sharp.end_vertex_id {
                return Err("blend network: the sharp edge at a miter is closed".into());
            }
            // The sharp edge's own convexity decides the closure.  A CONVEX
            // sharp edge leaves the two blends meeting in a seam (the miter);
            // a CONCAVE one — the cap's perimeter turning inward — puts the
            // seam nowhere: the ball rolls round that edge and the closure is
            // the horn-torus sector, with no ball to seat on the three faces.
            if !sharp_edge_is_convex(solid, sharp, other_a, other_b, radius)? {
                let selected_convex = [a, b].iter().all(|&index| {
                    let (edge, first, second) = &mates[index];
                    edge_is_convex(edge, first, second).unwrap_or(false)
                });
                if !selected_convex {
                    return Err(format!(
                        "blend network: vertex {vertex_id} is re-entrant with concave selected \
                         edges — the mirror of the horn-torus closure is not in this lane"
                    ));
                }
                corners.push((
                    *vertex_id,
                    stripe_indices.clone(),
                    VertexKind::Reentrant { cap: shared, sharp },
                ));
                continue;
            }
            miter = Some((shared, sharp));
        }
        let face_refs: Vec<&FaceRecord> = faces.iter().map(|(face, _, _)| *face).collect();
        let rho_of = |id: u64| {
            faces
                .iter()
                .find(|(face, _, _)| face.id == id)
                .map(|(_, rho, _)| *rho)
        };
        let prepared = corner_faces(&face_refs, &rho_of, point)?;
        let ball = solve_corner_ball(&prepared, scale)?;
        // The residual IS the common-ball test: N = 3 always has a root, and
        // beyond that a non-zero residual means the fillet cylinders have no
        // single inscribed sphere and the notch needs an N-sided fill.
        let bar = 1e-7 * (1.0 + scale);
        if ball.residual > bar {
            return Err(format!(
                "blend network: the {} fillets at vertex {vertex_id} have no common inscribed \
                 ball (residual {:.3e} > {bar:.3e}); an N-sided corner fill is not implemented",
                stripe_indices.len(),
                ball.residual
            ));
        }
        // Which side of the walls the ball sits on is the corner's
        // convexity: inside the material on every face (convex, the patch
        // faces away from the centre) or in the air on every face (concave,
        // the patch faces toward it).  One of each is a MIXED corner, whose
        // closure is not a sphere at all (the torus-sector class) and is
        // refused here by name.
        let mut inside = 0usize;
        let mut in_air = 0usize;
        for contact in &ball.contacts {
            let face = face_refs
                .iter()
                .find(|face| face.id == contact.face_id)
                .ok_or("blend network: the ball touched a face that is not at the corner")?;
            let face_outward = if face.same_sense {
                contact.normal
            } else {
                contact.normal.scale(-1.0)
            };
            let ball_inside_material = ball.center.sub(contact.point).dot(face_outward) < 0.0;
            if ball_inside_material {
                inside += 1;
            } else {
                in_air += 1;
            }
            // At a miter whose ball sits INSIDE the material the ball must
            // touch each SIDE face on the face itself, not on its carrier
            // beyond a concave edge: two convex edges meeting at a re-entrant
            // vertex (their third edge concave) have no seam — the ball rolls
            // round the concave edge instead and the closure is a horn torus,
            // which this lane does not build.
            //
            // A ball in the AIR is the mirror case and this test says nothing
            // there: two CONCAVE edges mitering over a convex sharp edge (the
            // base of a boss, a rib standing on a floor) seat their ball on
            // the far side of that sharp edge, so its contact with each side
            // wall is ALWAYS beyond the wall's own boundary — the sliver runs
            // past the corner and the two blends meet in the seam through
            // that contact.  The re-entrant class those two are told apart by
            // is the sharp edge's convexity, which `sharp_edge_is_convex`
            // already decided above.
            if let Some((shared, _)) = miter.filter(|_| ball_inside_material) {
                if contact.face_id != shared.id {
                    let class = crate::parameter_point_in_face(
                        face,
                        crate::Vec2 {
                            x: contact.uv[0],
                            y: contact.uv[1],
                        },
                        1e-6,
                    )?;
                    if class != crate::PolygonClass::Inside {
                        return Err(format!(
                            "blend network: vertex {vertex_id} is re-entrant — the ball \
                             touches face {} beyond its edge, so the two blends do not meet \
                             in a seam but wrap the concave edge (horn-torus closure, not in \
                             this lane)",
                            contact.face_id
                        ));
                    }
                }
            }
        }
        let outward = match (inside, in_air) {
            (_, 0) => true,
            (0, _) => false,
            _ => {
                return Err(format!(
                    "blend network: vertex {vertex_id} mixes convex and concave edges ({inside} \
                     faces with the ball inside, {in_air} with it in the air); a mixed corner \
                     is not closed by a sphere and this lane does not build it"
                ))
            }
        };
        // A corner is always OUTSIDE its own inscribed ball (a cube corner
        // sits at r√3 from the centre, a nearly flat one at r).  A centre
        // solved on the wrong branch fails this before anything is cut.
        if point.sub(ball.center).length() < radius * (1.0 - 1e-6) {
            return Err(format!(
                "blend network: the ball solved at vertex {vertex_id} swallows the corner \
                 itself ({:.6} < {radius}) — wrong branch",
                point.sub(ball.center).length()
            ));
        }
        // Commit the tangency vertices a star needs on every face; a miter
        // only needs the one on the shared face, and creates its other rim
        // vertices from the seam.
        let mut tangency = Vec::with_capacity(ball.contacts.len());
        for contact in &ball.contacts {
            let needed = match miter {
                Some((shared, _)) => contact.face_id == shared.id,
                None => true,
            };
            let id = if needed {
                let id = take_id();
                result.vertices.push(VertexRecord {
                    id,
                    point: contact.point,
                });
                id
            } else {
                0
            };
            tangency.push((contact.face_id, contact.point, id));
        }
        let corner = Corner {
            center: ball.center,
            outward,
            tangency,
        };
        let kind = match miter {
            Some((shared, sharp)) => VertexKind::Miter {
                corner,
                shared,
                sharp,
            },
            None => VertexKind::Star { corner },
        };
        corners.push((*vertex_id, stripe_indices.clone(), kind));
    }

    // §6.11 closes a chamfered corner differently from a filleted one, and only
    // the STAR is built here so far: its stripes stop at the ball and their end
    // CHORDS bound a planar facet. The others are still the cutter
    // composition's, and each says which one it is rather than the whole
    // selection failing under one message.
    if chamfer {
        for (vertex_id, _, kind) in &corners {
            let unbuilt = match kind {
                VertexKind::Star { .. } => None,
                VertexKind::Miter { .. } => Some(
                    "a chamfer's two-edge corner is the mutual trim of the two bevels, and                      the seam march solves for the sibling's rolling-ball canal, which a                      bevel is not",
                ),
                VertexKind::Flush => Some(
                    "a chamfer's smooth (G1) two-edge join has no fitted section to share yet",
                ),
                VertexKind::Reentrant { .. } => Some(
                    "a chamfer's re-entrant corner has no horn-torus sector to sweep",
                ),
            };
            if let Some(reason) = unbuilt {
                return Err(format!(
                    "blend network: vertex {vertex_id} — {reason} (§6.11)"
                ));
            }
        }
    }

    // ---- 4. March and fit every stripe, against the ORIGINAL solid. ----
    let radius_at = |_: f64| radius;
    let mut stripes: Vec<Stripe> = Vec::with_capacity(mates.len());
    for (index, (edge, first, second)) in mates.iter().enumerate() {
        let mut chosen: Option<FittedRows> = None;
        let mut last_error: Option<String> = None;
        for &overshoot in &OVERSHOOTS {
            match march_open_stations(edge, first, second, &radius_at, overshoot) {
                Ok((stations, vertex_indices)) => {
                    let parameters = station_parameters(&stations);
                    let at_vertices = vertex_stations(&parameters, vertex_indices);
                    match fit_open_rows(&stations, &parameters, chamfer) {
                        Ok(mut rows) => {
                            rows.vertex_stations = at_vertices;
                            let settled = [
                                (edge.start_vertex_id, true),
                                (edge.end_vertex_id, false),
                            ]
                            .into_iter()
                            .all(|(vertex, at_start)| {
                                if let Some((_, _, kind)) =
                                    corners.iter().find(|(id, _, _)| *id == vertex)
                                {
                                    return corner_rails_reach(
                                        &rows,
                                        first,
                                        second,
                                        kind,
                                        corner_station_bar(scale, radius),
                                    );
                                }
                                matches!(
                                    resolve_free_end(
                                        solid, edge.id, first, second, &rows, vertex, at_start
                                    ),
                                    Ok((_, true))
                                )
                            });
                            if chosen.is_none() || settled {
                                chosen = Some(rows);
                            }
                            if settled {
                                break;
                            }
                        }
                        Err(error) => last_error = Some(error),
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        let rows = chosen.ok_or_else(|| {
            last_error.unwrap_or_else(|| {
                format!("blend network: the march failed on edge {}", edge.id)
            })
        })?;
        stripes.push(Stripe {
            edge,
            first: BlendMate {
                face: first.face,
                coedge: first.coedge,
                loop_index: first.loop_index,
                rho: first.rho,
            },
            second: BlendMate {
                face: second.face,
                coedge: second.coedge,
                loop_index: second.loop_index,
                rho: second.rho,
            },
            rows,
            name: edge_names.get(index).cloned().flatten(),
        });
    }

    // Row parameter where `stripe`'s rail on `face_id` passes through
    // `point` — the stop station of a corner.  Both rows share one
    // parameterisation, so either rail reports the same station; the caller
    // asks for the rail it is about to trim.
    let station_on = |stripe: &Stripe, face_id: u64, point: Vec3| -> Result<f64, String> {
        let row = if stripe.first.face.id == face_id {
            &stripe.rows.cr
        } else if stripe.second.face.id == face_id {
            &stripe.rows.cs
        } else {
            return Err(format!(
                "blend network: edge {} does not touch face {face_id}",
                stripe.edge.id
            ));
        };
        rail_station(row, point, corner_station_bar(scale, radius)).map_err(|reason| match reason {
            RailMiss::OffRail(distance) => format!(
                "blend network: edge {}'s contact rail on face {face_id} misses the corner \
                 ball's tangency point by {distance:.3e}",
                stripe.edge.id
            ),
            RailMiss::OffSpan(station) => format!(
                "blend network: the corner station for edge {} lands at {station:.6}, outside \
                 the marched rows — the radius is too large for this edge",
                stripe.edge.id
            ),
            RailMiss::Projection(error) => error,
        })
    };

    // Free ends and the horn-torus lane both climb `support_crossing`'s
    // tolerance ladder; a crossing that only resolves on a looser rung builds
    // what the tight rung refused, so the network's answer is gated on being a
    // valid solid (see `SpokeCrossings`).
    let mut crossings = SpokeCrossings::default();

    // ---- 5. Turn every corner into per-stripe end plans. ----
    struct PendingPatch {
        center: Vec3,
        outward: bool,
        normals: Vec<Vec3>,
        arcs: Vec<EdgeRecord>,
        name: Option<String>,
    }
    /// A sharp edge to trim once the stripes are in, and the connector to
    /// splice next to a stripe's rail on a face.
    struct PendingMiter {
        vertex: u64,
        sharp_edge: u64,
        sharp_parameter: f64,
        sharp_vertex: u64,
        /// (stripe index whose rail the connector continues from, the face
        /// it lies on, connector edge id, its pcurve on that face — stored
        /// from the rail's end to the sharp edge)
        connector: Option<(usize, u64, u64, NurbsCurve)>,
    }
    let mut end_plans: Vec<[Option<EndPlan>; 2]> =
        (0..stripes.len()).map(|_| [None, None]).collect();
    let mut patches: Vec<PendingPatch> = Vec::new();
    let mut pending_miters: Vec<PendingMiter> = Vec::new();
    for (vertex_id, stripe_indices, kind) in &corners {
        match kind {
            // Planned below, once every star and miter is in.
            VertexKind::Reentrant { .. } | VertexKind::Flush => continue,
            VertexKind::Star { corner } => {
                let mut arcs: Vec<EdgeRecord> = Vec::with_capacity(stripe_indices.len());
                let mut normals: Vec<Vec3> = Vec::with_capacity(stripe_indices.len());
                for (_, point, _) in &corner.tangency {
                    normals.push(point.sub(corner.center).normalized()?);
                }
                for &index in stripe_indices {
                    let stripe = &stripes[index];
                    let at_start = stripe.at_start(*vertex_id);
                    let (first_point, first_vertex) = corner.tangency_on(stripe.first.face.id)?;
                    let (second_point, second_vertex) =
                        corner.tangency_on(stripe.second.face.id)?;
                    let station_first = station_on(stripe, stripe.first.face.id, first_point)?;
                    let station_second =
                        station_on(stripe, stripe.second.face.id, second_point)?;
                    if (station_first - station_second).abs() > 1e-3 {
                        return Err(format!(
                            "blend network: edge {} meets vertex {vertex_id} at inconsistent \
                             stations ({station_first:.6} on one support, {station_second:.6} \
                             on the other)",
                            stripe.edge.id
                        ));
                    }
                    let station = (station_first + station_second) * 0.5;
                    // The cross-section at the stop station is an arc of the
                    // corner sphere from one tangency point to the other.  It
                    // is stored in the direction the blend face's loop walks
                    // it, so the blend uses it forward and the patch, which
                    // always takes its boundary the other way round, reversed.
                    let walk_first_to_second = stripe.walk_first_to_second(at_start);
                    let (from, to, from_vertex, to_vertex) = if walk_first_to_second {
                        (first_point, second_point, first_vertex, second_vertex)
                    } else {
                        (second_point, first_point, second_vertex, first_vertex)
                    };
                    // §6.11: the chamfer's section at the stop station is the
                    // CHORD between the same two tangency points the fillet
                    // arcs between — the iso-parameter line of a ruled bevel.
                    let arc = if chamfer {
                        crate::make_line(from, to)?
                    } else {
                        corner_section_arc(corner.center, radius, from, to)?
                    };
                    let arc_domain = arc.domain()?;
                    let arc_edge_id = take_id();
                    let record = EdgeRecord {
                        id: arc_edge_id,
                        curve: arc,
                        t0: arc_domain[0],
                        t1: arc_domain[1],
                        start_vertex_id: from_vertex,
                        end_vertex_id: to_vertex,
                        degenerate: false,
                        name: None,
                    };
                    result.edges.push(record.clone());
                    arcs.push(record);
                    // The blend surface's own parameter space: z runs 0 -> 1
                    // from the first support row to the second, so the
                    // cross-section is the iso-parameter line at the stop
                    // station, walked in the same direction as the arc.
                    let arc_blend_pcurve = if walk_first_to_second {
                        crate::sweep_topology::parameter_line(station, 0.0, station, 1.0)?
                    } else {
                        crate::sweep_topology::parameter_line(station, 1.0, station, 0.0)?
                    };
                    end_plans[index][usize::from(!at_start)] = Some(EndPlan::Corner(CornerEnd {
                        cr_parameter: station,
                        cs_parameter: station,
                        first_vertex,
                        second_vertex,
                        arc_edge_id,
                        arc_blend_pcurve,
                    }));
                }
                patches.push(PendingPatch {
                    center: corner.center,
                    outward: corner.outward,
                    normals,
                    arcs,
                    name: corner_name(stripe_indices),
                });
            }
            VertexKind::Miter {
                corner,
                shared,
                sharp,
            } => {
                let [ia, ib] = [stripe_indices[0], stripe_indices[1]];
                let (p_point, p_vertex) = corner.tangency_on(shared.id)?;
                let describe = |index: usize| -> Result<MiterStripe, String> {
                    let stripe = &stripes[index];
                    let shared_first = stripe.first.face.id == shared.id;
                    let other_face = if shared_first {
                        stripe.second.face
                    } else {
                        stripe.first.face
                    };
                    Ok(MiterStripe {
                        rows: &stripe.rows,
                        shared_first,
                        at_start: stripe.at_start(*vertex_id),
                        u_ball: station_on(stripe, shared.id, p_point)?,
                        other_face,
                    })
                };
                let described = [describe(ia)?, describe(ib)?];
                let closure = solve_miter([&described[0], &described[1]], radius, sharp, scale)?;
                if std::env::var("BREP_DEBUG_NETWORK").is_ok() {
                    let seam = match &closure {
                        MiterClosure::Symmetric { seam, .. } => seam,
                        MiterClosure::Asymmetric { seam, .. } => seam,
                    };
                    let center_b = stripes[ib].rows.center.as_ref();
                    let center_a = stripes[ia].rows.center.as_ref();
                    let mut worst = 0.0f64;
                    for point in &seam.points {
                        for center in [center_a, center_b].into_iter().flatten() {
                            let d = crate::project_point_to_curve(center, *point)
                                .map(|p| (p.distance - radius).abs())
                                .unwrap_or(f64::NAN);
                            worst = worst.max(d);
                        }
                    }
                    eprintln!(
                        "network seam at vertex {vertex_id}: {} samples, worst |dist-to-centre - r| {worst:.3e}",
                        seam.points.len()
                    );
                }

                // Commit the vertices and edges the closure names, then hand
                // each stripe its end plan in its loop's walk order.
                let commit_curve = |result: &mut BrepSolid,
                                    take_id: &mut dyn FnMut() -> u64,
                                    curve: NurbsCurve,
                                    from: u64,
                                    to: u64|
                 -> Result<u64, String> {
                    let domain = curve.domain()?;
                    let id = take_id();
                    result.edges.push(EdgeRecord {
                        id,
                        curve,
                        t0: domain[0],
                        t1: domain[1],
                        start_vertex_id: from,
                        end_vertex_id: to,
                        degenerate: false,
                        name: None,
                    });
                    Ok(id)
                };
                let commit_vertex = |result: &mut BrepSolid,
                                     take_id: &mut dyn FnMut() -> u64,
                                     point: Vec3|
                 -> u64 {
                    let id = take_id();
                    result.vertices.push(VertexRecord { id, point });
                    id
                };
                // A plan for stripe `index` whose end is closed by `edges`
                // listed from P (the shared rim) to the far rim, with pcurves
                // on this stripe's blend in that same direction.
                let plan_for = |index: usize,
                                far_station: f64,
                                far_vertex: u64,
                                edges: Vec<(u64, NurbsCurve)>|
                 -> Result<EndPlan, String> {
                    let stripe = &stripes[index];
                    let at_start = stripe.at_start(*vertex_id);
                    let shared_first = stripe.first.face.id == shared.id;
                    let u_ball = described[usize::from(index == ib)].u_ball;
                    let (cr_parameter, cs_parameter, first_vertex, second_vertex) =
                        if shared_first {
                            (u_ball, far_station, p_vertex, far_vertex)
                        } else {
                            (far_station, u_ball, far_vertex, p_vertex)
                        };
                    // The walk runs P -> far exactly when the loop walks
                    // first -> second and P is on the first side, or the
                    // reverse of both.
                    let walk_p_to_far = stripe.walk_first_to_second(at_start) == shared_first;
                    let slot = if walk_p_to_far {
                        edges
                            .into_iter()
                            .map(|(id, pcurve)| (id, true, pcurve))
                            .collect()
                    } else {
                        edges
                            .into_iter()
                            .rev()
                            .map(|(id, pcurve)| Ok((id, false, pcurve.reversed()?)))
                            .collect::<Result<Vec<_>, String>>()?
                    };
                    Ok(EndPlan::Miter(MiterEnd {
                        cr_parameter,
                        cs_parameter,
                        first_vertex,
                        second_vertex,
                        edges: slot,
                    }))
                };
                let slot_of = |index: usize| usize::from(!stripes[index].at_start(*vertex_id));
                match closure {
                    MiterClosure::Symmetric {
                        seam,
                        exit,
                        sharp_parameter,
                    } => {
                        let q_point = sharp.curve.evaluate(sharp_parameter)?;
                        let q_vertex = commit_vertex(&mut result, &mut take_id, q_point);
                        let (curve, on_a, on_b) = fit_marched(&seam)?;
                        let seam_id =
                            commit_curve(&mut result, &mut take_id, curve, p_vertex, q_vertex)?;
                        end_plans[ia][slot_of(ia)] =
                            Some(plan_for(ia, exit[0], q_vertex, vec![(seam_id, on_a)])?);
                        end_plans[ib][slot_of(ib)] =
                            Some(plan_for(ib, exit[1], q_vertex, vec![(seam_id, on_b)])?);
                        pending_miters.push(PendingMiter {
                            vertex: *vertex_id,
                            sharp_edge: sharp.id,
                            sharp_parameter,
                            sharp_vertex: q_vertex,
                            connector: None,
                        });
                    }
                    MiterClosure::Asymmetric {
                        seam,
                        leader,
                        exit_leader,
                        connector,
                        exit_follower,
                        sharp_parameter,
                    } => {
                        let (leader_index, follower_index) = if leader == 0 {
                            (ia, ib)
                        } else {
                            (ib, ia)
                        };
                        let exit_point = *seam.points.last().ok_or("miter: empty seam")?;
                        let x_vertex = commit_vertex(&mut result, &mut take_id, exit_point);
                        let q_point = sharp.curve.evaluate(sharp_parameter)?;
                        let q_vertex = commit_vertex(&mut result, &mut take_id, q_point);
                        let (seam_curve, seam_on_a, seam_on_b) = fit_marched(&seam)?;
                        let seam_id = commit_curve(
                            &mut result,
                            &mut take_id,
                            seam_curve,
                            p_vertex,
                            x_vertex,
                        )?;
                        let (connector_curve, on_follower, on_face) = fit_marched(&connector)?;
                        let connector_id = commit_curve(
                            &mut result,
                            &mut take_id,
                            connector_curve,
                            x_vertex,
                            q_vertex,
                        )?;
                        // Seam images are stored on stripe 0 then stripe 1.
                        let (seam_on_leader, seam_on_follower) = if leader == 0 {
                            (seam_on_a, seam_on_b)
                        } else {
                            (seam_on_b, seam_on_a)
                        };
                        end_plans[leader_index][slot_of(leader_index)] = Some(plan_for(
                            leader_index,
                            exit_leader,
                            x_vertex,
                            vec![(seam_id, seam_on_leader)],
                        )?);
                        end_plans[follower_index][slot_of(follower_index)] = Some(plan_for(
                            follower_index,
                            exit_follower,
                            q_vertex,
                            vec![(seam_id, seam_on_follower), (connector_id, on_follower)],
                        )?);
                        pending_miters.push(PendingMiter {
                            vertex: *vertex_id,
                            sharp_edge: sharp.id,
                            sharp_parameter,
                            sharp_vertex: q_vertex,
                            connector: Some((
                                leader_index,
                                described[usize::from(leader != 0)].other_face.id,
                                connector_id,
                                on_face,
                            )),
                        });
                    }
                }
            }
        }
    }

    // Re-entrant corners: stop each stripe where its wall rail reaches the
    // concave edge, share the pole there, and close with the horn-torus
    // sector between the two end sections plus the cap arc between the two
    // cap rails.
    struct PendingHorn {
        vertex: u64,
        cap_face: u64,
        sharp_edge: u64,
        sharp_parameter: f64,
        pole_vertex: u64,
        pole: Vec3,
        axis: Vec3,
        /// Per stripe (in `stripe_indices` order): its index, the stored
        /// section arc (pole <-> cap point), its cap-rail vertex and point,
        /// and its ball centre at the stop.
        ends: Vec<(usize, EdgeRecord, u64, Vec3, Vec3)>,
        name: Option<String>,
    }
    let mut pending_horns: Vec<PendingHorn> = Vec::new();
    for (vertex_id, stripe_indices, kind) in &corners {
        let VertexKind::Reentrant { cap, sharp } = kind else {
            continue;
        };
        let corner_point = solid
            .vertices
            .iter()
            .find(|vertex| vertex.id == *vertex_id)
            .ok_or("blend network: re-entrant vertex missing")?
            .point;
        // Each stripe's wall rail meets the concave edge at the pole; both
        // must meet it at the SAME point (equal wall dihedrals), and the ball
        // centre there must sit over the pole exactly one radius out.
        let mut ends: Vec<(usize, f64, f64, Vec3, Vec3, Vec3)> = Vec::new(); // (index, u_stop, t_pole, pole, cap point, centre)
        for &index in stripe_indices {
            let stripe = &stripes[index];
            let at_start = stripe.at_start(*vertex_id);
            let cap_first = stripe.first.face.id == cap.id;
            let (wall_rail, cap_rail) = if cap_first {
                (&stripe.rows.cs, &stripe.rows.cr)
            } else {
                (&stripe.rows.cr, &stripe.rows.cs)
            };
            let (u_stop, t_pole, pole, escalated) = support_crossing(solid, wall_rail, sharp, at_start)
                .map_err(|error| {
                    format!(
                        "blend network: edge {}'s wall rail does not reach the concave edge at \
                         vertex {vertex_id} ({error})",
                        stripe.edge.id
                    )
                })?;
            crossings.note(escalated);
            if !(u_stop > RIM_MARGIN && u_stop < 1.0 - RIM_MARGIN) {
                return Err(format!(
                    "blend network: edge {}'s stop at the concave edge lands at {u_stop:.6}, \
                     outside the marched rows",
                    stripe.edge.id
                ));
            }
            let cap_point = cap_rail.evaluate(u_stop)?;
            let center = stripe
                .rows
                .center
                .as_ref()
                .ok_or("blend network: stripe rows carry no centre path")?
                .evaluate(u_stop)?;
            ends.push((index, u_stop, t_pole, pole, cap_point, center));
        }
        let pole = ends[0].3;
        let pole_gap = ends[1].3.sub(pole).length();
        if pole_gap > 1e-6 * (1.0 + radius) {
            return Err(format!(
                "blend network: the two blends at vertex {vertex_id} reach the concave edge \
                 {pole_gap:.3e} apart (unequal wall dihedrals) — the closure is not a horn torus \
                 and this lane does not build it"
            ));
        }
        let axis = tangent_away_from(sharp, *vertex_id)?;
        for (index, u_stop, _, _, cap_point, center) in &ends {
            // The section at the stop must be a meridian of the torus about
            // the concave edge: its plane (normal = the stripe's direction
            // there) must contain the axis, and its centre must sit one
            // radius from the pole, perpendicular to the axis.
            let stripe = &stripes[*index];
            let direction = stripe
                .rows
                .center
                .as_ref()
                .ok_or("blend network: stripe rows carry no centre path")?
                .derivatives(*u_stop, 1)?[1]
                .normalized()?;
            let radial = center.sub(pole);
            if direction.dot(axis).abs() > 1e-6
                || (radial.length() - radius).abs() > 1e-6 * (1.0 + radius)
                || radial.dot(axis).abs() > 1e-6 * (1.0 + radius)
            {
                return Err(format!(
                    "blend network: edge {}'s section at vertex {vertex_id} is not a meridian \
                     of the horn torus about the concave edge (tilted class) — not in this lane",
                    stripe.edge.id
                ));
            }
            if (cap_point.sub(corner_point).length() - radius).abs() > 1e-6 * (1.0 + radius) {
                return Err(format!(
                    "blend network: edge {}'s cap rail stops {:.6} from the re-entrant vertex, \
                     not one radius — the cap is not planar or the walls are not perpendicular \
                     to it",
                    stripe.edge.id,
                    cap_point.sub(corner_point).length()
                ));
            }
        }
        let pole_vertex = {
            let id = take_id();
            result.vertices.push(VertexRecord { id, point: pole });
            id
        };
        let mut committed: Vec<(usize, EdgeRecord, u64, Vec3, Vec3)> = Vec::new();
        for (index, u_stop, _, _, cap_point, center) in &ends {
            let stripe = &stripes[*index];
            let at_start = stripe.at_start(*vertex_id);
            let cap_first = stripe.first.face.id == cap.id;
            let cap_vertex = take_id();
            result.vertices.push(VertexRecord {
                id: cap_vertex,
                point: *cap_point,
            });
            // The section arc of the ball at the stop, from the cap point to
            // the pole, stored in the blend loop's walk direction like a star
            // arc.
            let (first_point, first_vertex, second_point, second_vertex) = if cap_first {
                (*cap_point, cap_vertex, pole, pole_vertex)
            } else {
                (pole, pole_vertex, *cap_point, cap_vertex)
            };
            let walk_first_to_second = stripe.walk_first_to_second(at_start);
            let (from, to, from_vertex, to_vertex) = if walk_first_to_second {
                (first_point, second_point, first_vertex, second_vertex)
            } else {
                (second_point, first_point, second_vertex, first_vertex)
            };
            let arc = corner_section_arc(*center, radius, from, to)?;
            let arc_domain = arc.domain()?;
            let arc_edge_id = take_id();
            let record = EdgeRecord {
                id: arc_edge_id,
                curve: arc,
                t0: arc_domain[0],
                t1: arc_domain[1],
                start_vertex_id: from_vertex,
                end_vertex_id: to_vertex,
                degenerate: false,
                name: None,
            };
            result.edges.push(record.clone());
            let arc_blend_pcurve = if walk_first_to_second {
                crate::sweep_topology::parameter_line(*u_stop, 0.0, *u_stop, 1.0)?
            } else {
                crate::sweep_topology::parameter_line(*u_stop, 1.0, *u_stop, 0.0)?
            };
            end_plans[*index][usize::from(!at_start)] = Some(EndPlan::Corner(CornerEnd {
                cr_parameter: *u_stop,
                cs_parameter: *u_stop,
                first_vertex,
                second_vertex,
                arc_edge_id,
                arc_blend_pcurve,
            }));
            committed.push((*index, record, cap_vertex, *cap_point, *center));
        }
        pending_horns.push(PendingHorn {
            vertex: *vertex_id,
            cap_face: cap.id,
            sharp_edge: sharp.id,
            sharp_parameter: ends[0].2,
            pole_vertex,
            pole,
            axis,
            ends: committed,
            name: corner_name(stripe_indices),
        });
    }

    // Flush joins: both stripes stop at their own vertex section, which is
    // the same ball on both sides, and share one arc.
    struct PendingFlush {
        vertex: u64,
        /// The seam edge between the two stripes' differing side faces, if
        /// any, to trim at the shared side rim: (edge id, parameter, rim).
        seam_trim: Option<(u64, f64, u64)>,
    }
    let mut pending_flush: Vec<PendingFlush> = Vec::new();
    for (vertex_id, stripe_indices, kind) in &corners {
        let VertexKind::Flush = kind else {
            continue;
        };
        let [ia, ib] = [stripe_indices[0], stripe_indices[1]];
        let section = |index: usize| -> Result<(f64, Vec3, Vec3, Vec3), String> {
            let stripe = &stripes[index];
            let at_start = stripe.at_start(*vertex_id);
            let station = stripe
                .rows
                .vertex_stations
                .ok_or("blend network: stripe rows carry no vertex stations")?
                [usize::from(!at_start)];
            if !(station > RIM_MARGIN && station < 1.0 - RIM_MARGIN) {
                return Err(format!(
                    "blend network: edge {}'s vertex section lands at {station:.6}, outside \
                     the marched rows",
                    stripe.edge.id
                ));
            }
            let center = stripe
                .rows
                .center
                .as_ref()
                .ok_or("blend network: stripe rows carry no centre path")?
                .evaluate(station)?;
            Ok((
                station,
                stripe.rows.cr.evaluate(station)?,
                stripe.rows.cs.evaluate(station)?,
                center,
            ))
        };
        let (station_a, cr_a, cs_a, center_a) = section(ia)?;
        let (station_b, cr_b, cs_b, center_b) = section(ib)?;
        let bar = 1e-6 * (1.0 + radius);
        if center_a.sub(center_b).length() > bar {
            return Err(format!(
                "blend network: the two blends at the smooth vertex {vertex_id} sit on \
                 different balls ({:.3e} apart) — not a flush join",
                center_a.sub(center_b).length()
            ));
        }
        // Pair stripe b's rims with stripe a's by position (the mates may be
        // different faces that are tangent there).
        let (b_first_matches_a_first, mismatch) = {
            let direct = cr_b.sub(cr_a).length().max(cs_b.sub(cs_a).length());
            let crossed = cr_b.sub(cs_a).length().max(cs_b.sub(cr_a).length());
            if direct <= crossed {
                (true, direct)
            } else {
                (false, crossed)
            }
        };
        if mismatch > bar {
            return Err(format!(
                "blend network: the two sections at the smooth vertex {vertex_id} differ by \
                 {mismatch:.3e} — not a flush join"
            ));
        }
        // Shared rim vertices, keyed by stripe a's first/second sides.
        let rim_first = take_id();
        result.vertices.push(VertexRecord {
            id: rim_first,
            point: cr_a,
        });
        let rim_second = take_id();
        result.vertices.push(VertexRecord {
            id: rim_second,
            point: cs_a,
        });
        // One arc, stored in stripe a's walk direction; stripe b walks it the
        // other way (its loop runs opposite across the shared edge), which
        // the corner plan expresses by storing... no: a corner plan always
        // uses its arc forward.  So commit the arc in a's walk direction and
        // give b a plan whose arc pcurve runs the same 3D way — b's walk is
        // guaranteed opposite by manifoldness, so b uses `forward = false`.
        // `CornerEnd` cannot say that; a `MiterEnd` can.
        let stripe_a = &stripes[ia];
        let at_start_a = stripe_a.at_start(*vertex_id);
        let walk_a_first_to_second = stripe_a.walk_first_to_second(at_start_a);
        let (from, to, from_vertex, to_vertex) = if walk_a_first_to_second {
            (cr_a, cs_a, rim_first, rim_second)
        } else {
            (cs_a, cr_a, rim_second, rim_first)
        };
        let arc = corner_section_arc(center_a, radius, from, to)?;
        let arc_domain = arc.domain()?;
        let arc_edge_id = take_id();
        result.edges.push(EdgeRecord {
            id: arc_edge_id,
            curve: arc,
            t0: arc_domain[0],
            t1: arc_domain[1],
            start_vertex_id: from_vertex,
            end_vertex_id: to_vertex,
            degenerate: false,
            name: None,
        });
        let pcurve_a = if walk_a_first_to_second {
            crate::sweep_topology::parameter_line(station_a, 0.0, station_a, 1.0)?
        } else {
            crate::sweep_topology::parameter_line(station_a, 1.0, station_a, 0.0)?
        };
        end_plans[ia][usize::from(!at_start_a)] = Some(EndPlan::Corner(CornerEnd {
            cr_parameter: station_a,
            cs_parameter: station_a,
            first_vertex: rim_first,
            second_vertex: rim_second,
            arc_edge_id,
            arc_blend_pcurve: pcurve_a,
        }));
        // Stripe b: its first/second rims map onto a's by the pairing above.
        let stripe_b = &stripes[ib];
        let at_start_b = stripe_b.at_start(*vertex_id);
        let (b_first_vertex, b_second_vertex) = if b_first_matches_a_first {
            (rim_first, rim_second)
        } else {
            (rim_second, rim_first)
        };
        // b's walk over the slot, first-rim -> second-rim or the reverse, in
        // terms of the arc's stored direction (a's walk).  The stored arc
        // runs a-first -> a-second when `walk_a_first_to_second`.
        let b_walk_first_to_second = stripe_b.walk_first_to_second(at_start_b);
        let b_walk_in_a_terms = b_walk_first_to_second == b_first_matches_a_first;
        let forward = b_walk_in_a_terms == walk_a_first_to_second;
        let pcurve_b = if b_walk_first_to_second {
            crate::sweep_topology::parameter_line(station_b, 0.0, station_b, 1.0)?
        } else {
            crate::sweep_topology::parameter_line(station_b, 1.0, station_b, 0.0)?
        };
        end_plans[ib][usize::from(!at_start_b)] = Some(EndPlan::Miter(MiterEnd {
            cr_parameter: station_b,
            cs_parameter: station_b,
            first_vertex: b_first_vertex,
            second_vertex: b_second_vertex,
            edges: vec![(arc_edge_id, forward, pcurve_b)],
        }));
        if forward {
            return Err(format!(
                "blend network: the two blends at the smooth vertex {vertex_id} walk their \
                 shared section the same way — inconsistent orientation"
            ));
        }
        // If the side faces differ, the seam between them passes through the
        // vertex and must be trimmed at the shared side rim.
        let mut seam_trim = None;
        for (side_a, rim, rim_point) in [
            (stripe_a.first.face.id, rim_first, cr_a),
            (stripe_a.second.face.id, rim_second, cs_a),
        ] {
            let b_face = if b_first_matches_a_first {
                if side_a == stripe_a.first.face.id { stripe_b.first.face.id } else { stripe_b.second.face.id }
            } else if side_a == stripe_a.first.face.id {
                stripe_b.second.face.id
            } else {
                stripe_b.first.face.id
            };
            if b_face == side_a {
                continue;
            }
            // The seam: the edge at the vertex bordering both differing faces,
            // other than the two selected edges.
            let seam = solid.edges.iter().find(|edge| {
                (edge.start_vertex_id == *vertex_id || edge.end_vertex_id == *vertex_id)
                    && edge.id != stripe_a.edge.id
                    && edge.id != stripe_b.edge.id
                    && [side_a, b_face].iter().all(|face_id| {
                        solid
                            .shells
                            .iter()
                            .flat_map(|shell| &shell.faces)
                            .filter(|face| face.id == *face_id)
                            .flat_map(|face| &face.loops)
                            .flat_map(|l| &l.coedges)
                            .any(|c| c.edge_id == edge.id)
                    })
            });
            let Some(seam) = seam else {
                return Err(format!(
                    "blend network: the side faces at the smooth vertex {vertex_id} differ \
                     but share no seam edge there"
                ));
            };
            let on_seam = crate::project_point_to_curve(&seam.curve, rim_point)?;
            if on_seam.distance > bar {
                return Err(format!(
                    "blend network: the shared side rim at the smooth vertex {vertex_id} is \
                     {:.3e} off the seam between the side faces",
                    on_seam.distance
                ));
            }
            if seam_trim.is_some() {
                return Err(format!(
                    "blend network: both side pairs differ at the smooth vertex {vertex_id}"
                ));
            }
            seam_trim = Some((seam.id, on_seam.u, rim));
        }
        pending_flush.push(PendingFlush {
            vertex: *vertex_id,
            seam_trim,
        });
    }

    let mut pending_caps: Vec<PendingCap> = Vec::new();

    // ---- 6. Free ends keep the §6.9 transverse construction — or, where
    //         the edge runs smoothly into an unselected one, a CAP. ----
    for (index, stripe) in stripes.iter().enumerate() {
        for (slot, (vertex, at_start)) in [
            (stripe.edge.start_vertex_id, true),
            (stripe.edge.end_vertex_id, false),
        ]
        .into_iter()
        .enumerate()
        {
            if end_plans[index][slot].is_some() {
                continue;
            }
            match resolve_free_end(
                solid,
                stripe.edge.id,
                &stripe.first,
                &stripe.second,
                &stripe.rows,
                vertex,
                at_start,
            ) {
                Ok((end, _)) => {
                    crossings.note(end.escalated);
                    end_plans[index][slot] = Some(EndPlan::Free(end));
                }
                Err(free_error) => {
                    // A boundary that continues the edge smoothly has no
                    // transverse crossing: cap the stripe at its own section.
                    let continues = [stripe.first.face, stripe.second.face]
                        .iter()
                        .any(|face| {
                            boundary_edge_at_vertex(solid, face, vertex, stripe.edge.id)
                                .ok()
                                .and_then(|id| solid.edges.iter().find(|e| e.id == id))
                                .and_then(|boundary| {
                                    let own = tangent_away_from(stripe.edge, vertex).ok()?;
                                    let other = tangent_away_from(boundary, vertex).ok()?;
                                    Some(own.dot(other) <= -(1.0 - 1e-6))
                                })
                                .unwrap_or(false)
                        });
                    if !continues {
                        return Err(free_error);
                    }
                    let plan = plan_cap_end(
                        solid,
                        &mut result,
                        &mut take_id,
                        stripe,
                        vertex,
                        at_start,
                        radius,
                        &mut pending_caps,
                    )?;
                    end_plans[index][slot] = Some(EndPlan::Cap(plan));
                }
            }
        }
    }

    // ---- 7. Sew: every stripe, then the corner patches and the miter
    //         trims, then one prune. ----
    let mut sewn: Vec<SewnStripe> = Vec::with_capacity(stripes.len());
    for (index, stripe) in stripes.iter().enumerate() {
        let [start_plan, finish_plan] = std::mem::replace(&mut end_plans[index], [None, None]);
        let (Some(start_plan), Some(finish_plan)) = (start_plan, finish_plan) else {
            return Err("blend network: a stripe end was left unplanned".into());
        };
        // The two ends must leave some rail between them.  When the corner
        // setbacks from both ends meet or cross, the edge is shorter than the
        // fillet is wide and its blend strip pinches out — a configuration
        // this lane names rather than builds (pinch splitting is not
        // implemented).
        for (start, finish, which) in [
            (start_plan.cr_parameter(), finish_plan.cr_parameter(), "first"),
            (start_plan.cs_parameter(), finish_plan.cs_parameter(), "second"),
        ] {
            if finish - start <= RIM_MARGIN {
                return Err(format!(
                    "blend network: edge {} is shorter than the two corner setbacks that meet                      on it (its {which} rail would run from {start:.4} to {finish:.4}); the                      blend strip pinches out and pinch splitting is not implemented",
                    stripe.edge.id
                ));
            }
        }
        let rows = FittedRows {
            surface: stripe.rows.surface.clone(),
            cr: stripe.rows.cr.clone(),
            cs: stripe.rows.cs.clone(),
            cr_pcurve: stripe.rows.cr_pcurve.clone(),
            cs_pcurve: stripe.rows.cs_pcurve.clone(),
            u_domain: stripe.rows.u_domain,
            center: None,
            exact_extrusion: stripe.rows.exact_extrusion,
            vertex_stations: stripe.rows.vertex_stations,
        };
        sewn.push(build_open_surgery(
            solid,
            &mut result,
            &mut take_id,
            stripe.edge,
            &stripe.first,
            &stripe.second,
            rows,
            [start_plan, finish_plan],
            stripe.name.as_deref(),
        )?);
    }
    for patch in patches {
        let face = if chamfer {
            build_chamfer_corner_facet(
                patch.center,
                &patch.normals,
                &patch.arcs,
                patch.outward,
                patch.name.as_deref(),
                &mut take_id,
            )?
        } else {
            build_general_corner_patch(
                patch.center,
                radius,
                &patch.normals,
                &patch.arcs,
                patch.outward,
                patch.name.as_deref(),
                &mut take_id,
            )?
        };
        let shell_index = result
            .shells
            .iter()
            .position(|shell| !shell.faces.is_empty())
            .ok_or("blend network: the solid has no shell to sew the corner patch into")?;
        result.shells[shell_index].faces.push(face);
    }
    for miter in pending_miters {
        // The sharp edge loses its corner end: it now starts at the seam's
        // exit on it, on both faces it borders.
        trim_edge_at(
            &mut result,
            miter.sharp_edge,
            miter.sharp_parameter,
            miter.vertex,
            miter.sharp_vertex,
        )?;
        let Some((leader_index, face_id, connector_id, on_face)) = miter.connector else {
            continue;
        };
        // The connector sits on the leader's other face between the
        // leader's rail there (which ends at the seam's exit) and the
        // trimmed sharp edge (which starts where the connector ends).
        let leader = &stripes[leader_index];
        let rail_id = if leader.first.face.id == face_id {
            sewn[leader_index].cr_edge_id
        } else {
            sewn[leader_index].cs_edge_id
        };
        let coedge_id = take_id();
        let face = result
            .shells
            .iter_mut()
            .flat_map(|shell| &mut shell.faces)
            .find(|face| face.id == face_id)
            .ok_or("blend network: the connector's face is missing")?;
        let mut spliced = false;
        for loop_record in &mut face.loops {
            let count = loop_record.coedges.len();
            let Some(rail_at) = loop_record
                .coedges
                .iter()
                .position(|coedge| coedge.edge_id == rail_id)
            else {
                continue;
            };
            let Some(sharp_at) = loop_record
                .coedges
                .iter()
                .position(|coedge| coedge.edge_id == miter.sharp_edge)
            else {
                continue;
            };
            let (insert_at, forward, pcurve) = if (rail_at + 1) % count == sharp_at {
                // rail -> connector -> sharp: the walk runs from the rail's
                // end (the seam's exit) to the sharp edge, the stored sense.
                (rail_at + 1, true, on_face.clone())
            } else if (sharp_at + 1) % count == rail_at {
                (sharp_at + 1, false, on_face.reversed()?)
            } else {
                return Err(
                    "blend network: the leader's rail and the sharp edge are not adjacent on \
                     the connector's face"
                        .into(),
                );
            };
            loop_record.coedges.insert(
                insert_at,
                CoedgeRecord {
                    id: coedge_id,
                    edge_id: connector_id,
                    forward,
                    pcurve,
                },
            );
            spliced = true;
            break;
        }
        if !spliced {
            return Err(
                "blend network: no loop on the connector's face holds the leader's rail and \
                 the sharp edge"
                    .into(),
            );
        }
    }
    for horn in pending_horns {
        trim_edge_at(
            &mut result,
            horn.sharp_edge,
            horn.sharp_parameter,
            horn.vertex,
            horn.pole_vertex,
        )?;
        let corner_point = solid
            .vertices
            .iter()
            .find(|vertex| vertex.id == horn.vertex)
            .ok_or("blend network: re-entrant vertex missing")?
            .point;
        let [(index_a, arc_a, cap_a, cap_point_a, center_a), (index_b, arc_b, cap_b, cap_point_b, center_b)] =
            <[_; 2]>::try_from(horn.ends).map_err(|_| "blend network: a horn needs two ends")?;
        // The cap arc: the torus's tangency with the cap, a circle of one
        // radius about the re-entrant vertex from one cap point to the other.
        let cap_arc = corner_section_arc(corner_point, radius, cap_point_a, cap_point_b)?;
        let cap_domain = cap_arc.domain()?;
        let cap_arc_id = take_id();
        result.edges.push(EdgeRecord {
            id: cap_arc_id,
            curve: cap_arc.clone(),
            t0: cap_domain[0],
            t1: cap_domain[1],
            start_vertex_id: cap_a,
            end_vertex_id: cap_b,
            degenerate: false,
            name: None,
        });
        // Splice it into the cap's loop between the two cap rails.
        let cap_forward_on_cap: bool;
        let rail_a = if stripes[index_a].first.face.id == horn.cap_face {
            sewn[index_a].cr_edge_id
        } else {
            sewn[index_a].cs_edge_id
        };
        let rail_b = if stripes[index_b].first.face.id == horn.cap_face {
            sewn[index_b].cr_edge_id
        } else {
            sewn[index_b].cs_edge_id
        };
        {
            let cap_face = result
                .shells
                .iter_mut()
                .flat_map(|shell| &mut shell.faces)
                .find(|face| face.id == horn.cap_face)
                .ok_or("blend network: the cap face is missing")?;
            let cap_pcurve = {
                // The cap is planar here (checked above): its pcurve is the
                // projection of the arc's samples.
                let mut samples = Vec::with_capacity(17);
                let mut parameters = Vec::with_capacity(17);
                for k in 0..=16 {
                    let t = cap_domain[0] + (cap_domain[1] - cap_domain[0]) * k as f64 / 16.0;
                    let point = cap_arc.evaluate(t)?;
                    let projection = crate::project_point_to_surface(&cap_face.surface, point)?;
                    samples.push(crate::Vec4::from_point(
                        Vec3::new(projection.u, projection.v, 0.0),
                        1.0,
                    ));
                    parameters.push(k as f64 / 16.0);
                }
                crate::fit::interpolate_homogeneous(&samples, 3, &parameters)?
            };
            let mut spliced: Option<bool> = None;
            for loop_record in &mut cap_face.loops {
                let count = loop_record.coedges.len();
                let (Some(at_a), Some(at_b)) = (
                    loop_record.coedges.iter().position(|c| c.edge_id == rail_a),
                    loop_record.coedges.iter().position(|c| c.edge_id == rail_b),
                ) else {
                    continue;
                };
                let (insert_at, forward, pcurve) = if (at_a + 1) % count == at_b {
                    (at_a + 1, true, cap_pcurve.clone())
                } else if (at_b + 1) % count == at_a {
                    (at_b + 1, false, cap_pcurve.reversed()?)
                } else {
                    return Err(
                        "blend network: the two cap rails are not adjacent on the cap".into(),
                    );
                };
                loop_record.coedges.insert(
                    insert_at,
                    CoedgeRecord {
                        id: take_id(),
                        edge_id: cap_arc_id,
                        forward,
                        pcurve,
                    },
                );
                spliced = Some(forward);
                break;
            }
            let Some(forward) = spliced else {
                return Err("blend network: no cap loop holds both cap rails".into());
            };
            cap_forward_on_cap = forward;
        }
        // The horn-torus sector: the generatrix is the first stripe's section
        // arc run from the pole to its cap point, revolved about the concave
        // edge through the pole by the angle between the two ball centres.
        let radial_a = center_a.sub(horn.pole).normalized()?;
        let radial_b = center_b.sub(horn.pole).normalized()?;
        let sweep = radial_a.dot(radial_b).clamp(-1.0, 1.0).acos();
        if sweep < 1e-9 {
            return Err("blend network: degenerate horn-torus sweep".into());
        }
        let revolve_axis = if radial_a.cross(radial_b).dot(horn.axis) > 0.0 {
            horn.axis
        } else {
            horn.axis.scale(-1.0)
        };
        let generatrix = corner_section_arc(center_a, radius, horn.pole, cap_point_a)?;
        let surface = crate::make_revolution(horn.pole, revolve_axis, &generatrix, sweep)?;
        // Sanity: the far meridian lands on the second stripe's cap point.
        let far = surface.evaluate(1.0, 1.0)?;
        if far.sub(cap_point_b).length() > 1e-6 * (1.0 + radius) {
            return Err(format!(
                "blend network: the horn torus does not close on the second cap point ({:.3e} off)",
                far.sub(cap_point_b).length()
            ));
        }
        let pole_edge_id = take_id();
        result.edges.push(EdgeRecord {
            id: pole_edge_id,
            curve: crate::make_line(horn.pole, horn.pole)?,
            t0: 0.0,
            t1: 1.0,
            start_vertex_id: horn.pole_vertex,
            end_vertex_id: horn.pole_vertex,
            degenerate: true,
            name: None,
        });
        let pl = |u0, v0, u1, v1| crate::sweep_topology::parameter_line(u0, v0, u1, v1);
        // Every shared edge is used with the sense OPPOSITE to its
        // neighbour: the blends use their section arcs forward (they were
        // stored in the blends' walk direction), the cap uses the cap arc
        // with the sense the splice gave it.  Those three senses fix the
        // walk; the u/v boundaries are: u = 0 meridian = arc_a, v = 1 = the
        // cap arc, u = 1 meridian = arc_b, v = 0 = the pole.
        let a_walks_down = arc_a.start_vertex_id == horn.pole_vertex; // forward=false walks cap_a -> pole
        let b_walks_up = arc_b.end_vertex_id == horn.pole_vertex; // forward=false walks pole -> cap_b
        let cap_walks_back = cap_forward_on_cap; // horn uses !cap_forward: cap_b -> cap_a iff the cap walked a -> b
        if a_walks_down != b_walks_up || a_walks_down != cap_walks_back {
            return Err(
                "blend network: the blends and the cap do not walk the horn's boundary \
                 consistently"
                    .into(),
            );
        }
        let coedges = if a_walks_down {
            // cap_a -> pole -> (pole edge) -> cap_b -> cap_a
            vec![
                CoedgeRecord {
                    id: take_id(),
                    edge_id: arc_a.id,
                    forward: false,
                    pcurve: pl(0.0, 1.0, 0.0, 0.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: pole_edge_id,
                    forward: true,
                    pcurve: pl(0.0, 0.0, 1.0, 0.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: arc_b.id,
                    forward: false,
                    pcurve: pl(1.0, 0.0, 1.0, 1.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: cap_arc_id,
                    forward: false,
                    pcurve: pl(1.0, 1.0, 0.0, 1.0)?,
                },
            ]
        } else {
            // pole -> cap_a -> cap_b -> pole -> (pole edge)
            vec![
                CoedgeRecord {
                    id: take_id(),
                    edge_id: arc_a.id,
                    forward: false,
                    pcurve: pl(0.0, 0.0, 0.0, 1.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: cap_arc_id,
                    forward: true,
                    pcurve: pl(0.0, 1.0, 1.0, 1.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: arc_b.id,
                    forward: false,
                    pcurve: pl(1.0, 1.0, 1.0, 0.0)?,
                },
                CoedgeRecord {
                    id: take_id(),
                    edge_id: pole_edge_id,
                    forward: true,
                    pcurve: pl(1.0, 0.0, 0.0, 0.0)?,
                },
            ]
        };
        // Convex: the sector's normal points away from the ball centre under
        // each point (the centre circle, one radius from the pole).
        let mid = surface.evaluate(0.5, 0.5)?;
        let mid_center = horn.pole.add(
            radial_a
                .scale((sweep * 0.5).cos())
                .add(revolve_axis.cross(radial_a).scale((sweep * 0.5).sin())),
        );
        let surface_normal = surface.normal(0.5, 0.5)?;
        let same_sense = surface_normal.dot(mid.sub(mid_center)) > 0.0;
        let loop_id = take_id();
        let face = FaceRecord {
            id: take_id(),
            surface,
            same_sense,
            loops: vec![crate::topology::LoopRecord {
                id: loop_id,
                coedges,
            }],
            name: horn.name,
        };
        let shell_index = result
            .shells
            .iter()
            .position(|shell| !shell.faces.is_empty())
            .ok_or("blend network: the solid has no shell to sew the horn torus into")?;
        result.shells[shell_index].faces.push(face);
    }
    for flush in pending_flush {
        if let Some((seam_id, parameter, rim)) = flush.seam_trim {
            trim_edge_at(&mut result, seam_id, parameter, flush.vertex, rim)?;
        }
    }
    for cap in pending_caps {
        sew_cap(&mut result, &mut take_id, cap)?;
    }
    // A rail that wrapped a PERIODIC mate crosses that carrier's seam
    // meridian; split it there and trim the seam edge the blend just ate the
    // end of, so the carrier keeps its seam-in-one-loop structure.
    let mut rail_faces: Vec<(u64, u64)> = Vec::with_capacity(sewn.len() * 2);
    for (index, stripe) in stripes.iter().enumerate() {
        rail_faces.push((stripe.first.face.id, sewn[index].cr_edge_id));
        rail_faces.push((stripe.second.face.id, sewn[index].cs_edge_id));
    }
    let rail_edges: Vec<u64> = rail_faces.iter().map(|(_, rail)| *rail).collect();
    let mut seam_faces: Vec<u64> = rail_faces.iter().map(|(face, _)| *face).collect();
    seam_faces.sort_unstable();
    seam_faces.dedup();
    for face_id in seam_faces {
        // One seam pair per periodic direction, so a second pass only ever
        // finds the u-and-v case; the bound keeps a mis-detection from looping.
        for _ in 0..2 {
            let Some(plan) = plan_carrier_seam_split(&result, &rail_edges, face_id)? else {
                break;
            };
            apply_carrier_seam_split(&mut result, &mut take_id, plan)?;
        }
    }
    prune_orphan_vertices(&mut result);
    crossings.gate(result)
}

/// A cap's bulkhead, waiting for the stripes to be in so its arc and legs
/// exist: the plane's frame and the three edges in the bulkhead's own walk.
struct PendingCap {
    plane_origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    plane_normal: Vec3,
    /// (edge id, stored start vertex, stored end vertex)
    arc: (u64, u64, u64),
    first_leg: (u64, u64, u64),
    second_leg: (u64, u64, u64),
    /// The bulkhead's outward normal must point along this (away from the
    /// stripe): the sliver's cross-section faces the continuing edge.
    outward: Vec3,
    name: Option<String>,
}

/// Plan a cap for `stripe`'s end at `vertex`: commit the section arc at the
/// vertex station, the two straight legs from its ends to the sharp vertex
/// (each verified to lie on its mate), and record the bulkhead.
#[allow(clippy::too_many_arguments)]
fn plan_cap_end(
    solid: &BrepSolid,
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    stripe: &Stripe,
    vertex: u64,
    at_start: bool,
    radius: f64,
    pending: &mut Vec<PendingCap>,
) -> Result<CapEnd, String> {
    let sharp_point = solid
        .vertices
        .iter()
        .find(|candidate| candidate.id == vertex)
        .ok_or("blend network: cap vertex missing")?
        .point;
    let station = stripe
        .rows
        .vertex_stations
        .ok_or("blend network: stripe rows carry no vertex stations")?[usize::from(!at_start)];
    if !(station > RIM_MARGIN && station < 1.0 - RIM_MARGIN) {
        return Err(format!(
            "blend network: edge {}'s vertex section lands at {station:.6}, outside the \
             marched rows",
            stripe.edge.id
        ));
    }
    let first_point = stripe.rows.cr.evaluate(station)?;
    let second_point = stripe.rows.cs.evaluate(station)?;
    let center = stripe
        .rows
        .center
        .as_ref()
        .ok_or("blend network: stripe rows carry no centre path")?
        .evaluate(station)?;
    // The section plane: through the vertex, normal = the edge's tangent.
    let normal = tangent_away_from(stripe.edge, vertex)?;
    for point in [first_point, second_point, center] {
        if point.sub(sharp_point).dot(normal).abs() > 1e-5 * (1.0 + radius) {
            return Err(format!(
                "blend network: edge {}'s vertex section is not in the vertex's normal plane \
                 ({:.3e} off)",
                stripe.edge.id,
                point.sub(sharp_point).dot(normal).abs()
            ));
        }
    }
    // Legs: straight, and they must lie ON the mates (planar mates, or a
    // ruled mate whose ruling is the leg).
    for (point, mate) in [(first_point, &stripe.first), (second_point, &stripe.second)] {
        let mid = point.add(sharp_point).scale(0.5);
        let projection = crate::project_point_to_surface(&mate.face.surface, mid)?;
        if projection.distance > 1e-6 * (1.0 + radius) {
            return Err(format!(
                "blend network: the cap leg on face {} at edge {} leaves the face by {:.3e} — \
                 a curved section leg is not in this lane",
                mate.face.id, stripe.edge.id, projection.distance
            ));
        }
    }
    // The legs meet at the EXISTING vertex — the continuing edge still ends
    // there, and a fresh vertex at the same point would leave every loop
    // through it open.
    let sharp_vertex = vertex;
    let first_vertex = take_id();
    result.vertices.push(VertexRecord {
        id: first_vertex,
        point: first_point,
    });
    let second_vertex = take_id();
    result.vertices.push(VertexRecord {
        id: second_vertex,
        point: second_point,
    });
    // The arc in the blend loop's walk direction.
    let walk_first_to_second = stripe.walk_first_to_second(at_start);
    let (from, to, from_vertex, to_vertex) = if walk_first_to_second {
        (first_point, second_point, first_vertex, second_vertex)
    } else {
        (second_point, first_point, second_vertex, first_vertex)
    };
    let arc = corner_section_arc(center, radius, from, to)?;
    let arc_domain = arc.domain()?;
    let arc_edge_id = take_id();
    result.edges.push(EdgeRecord {
        id: arc_edge_id,
        curve: arc,
        t0: arc_domain[0],
        t1: arc_domain[1],
        start_vertex_id: from_vertex,
        end_vertex_id: to_vertex,
        degenerate: false,
        name: None,
    });
    let arc_blend_pcurve = if walk_first_to_second {
        crate::sweep_topology::parameter_line(station, 0.0, station, 1.0)?
    } else {
        crate::sweep_topology::parameter_line(station, 1.0, station, 0.0)?
    };
    // Legs from rim to vertex, with pcurves on their mates from the
    // projections of both ends (a straight line in UV for planar mates; a
    // ruling's UV image is straight on a cylinder too).
    let mut leg = |rim_point: Vec3,
                   rim_vertex: u64,
                   face: &FaceRecord|
     -> Result<(u64, NurbsCurve), String> {
        let id = take_id();
        result.edges.push(EdgeRecord {
            id,
            curve: crate::make_line(rim_point, sharp_point)?,
            t0: 0.0,
            t1: 1.0,
            start_vertex_id: rim_vertex,
            end_vertex_id: sharp_vertex,
            degenerate: false,
            name: None,
        });
        let a = crate::project_point_to_surface(&face.surface, rim_point)?;
        let b = crate::project_point_to_surface(&face.surface, sharp_point)?;
        Ok((id, crate::sweep_topology::parameter_line(a.u, a.v, b.u, b.v)?))
    };
    let first_leg = leg(first_point, first_vertex, stripe.first.face)?;
    let second_leg = leg(second_point, second_vertex, stripe.second.face)?;
    // The bulkhead plane's frame: u along first->second rim, v completing.
    let plane_u = second_point.sub(first_point).normalized()?;
    let plane_v = normal.cross(plane_u).normalized()?;
    pending.push(PendingCap {
        plane_origin: sharp_point,
        plane_u,
        plane_v,
        plane_normal: normal,
        arc: (arc_edge_id, from_vertex, to_vertex),
        first_leg: (first_leg.0, first_vertex, sharp_vertex),
        second_leg: (second_leg.0, second_vertex, sharp_vertex),
        outward: normal.scale(-1.0),
        name: stripe.name.clone(),
    });
    Ok(CapEnd {
        station,
        first_vertex,
        second_vertex,
        arc_edge_id,
        arc_blend_pcurve,
        first_leg,
        second_leg,
        sharp_vertex,
    })
}

/// Sew a cap's bulkhead: a planar face in the section plane whose loop is
/// leg (vertex -> first rim), arc (first rim -> second rim), leg (second rim
/// -> vertex), each used with the sense OPPOSITE to its neighbour (the blend
/// walks the arc forward; the mates walk the legs rim -> vertex where the
/// rail ends at the rim).
fn sew_cap(
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    cap: PendingCap,
) -> Result<(), String> {
    // A plane patch large enough to hold the bulkhead with its origin set
    // back so the cap sits inside the patch's parameter square: the section
    // spans at most a few radii around the vertex.
    let reach = {
        let a = vertex_point_of(result, cap.first_leg.1)?;
        let b = vertex_point_of(result, cap.second_leg.1)?;
        a.sub(cap.plane_origin)
            .length()
            .max(b.sub(cap.plane_origin).length())
            .max(1e-6)
            * 4.0
    };
    let origin = cap
        .plane_origin
        .sub(cap.plane_u.scale(reach))
        .sub(cap.plane_v.scale(reach));
    let plane = crate::make_plane(origin, cap.plane_u, cap.plane_v, 2.0 * reach, 2.0 * reach)?;
    let uv_of = |point: Vec3| -> Result<(f64, f64), String> {
        let projection = crate::project_point_to_surface(&plane, point)?;
        Ok((projection.u, projection.v))
    };
    let edge_by_id = |id: u64| -> Result<&EdgeRecord, String> {
        result
            .edges
            .iter()
            .find(|edge| edge.id == id)
            .ok_or_else(|| format!("blend network: cap edge {id} missing"))
    };
    // How the blend uses the arc: forward.  The bulkhead uses it reversed,
    // so its walk runs arc-end -> arc-start; the legs then run start-rim
    // -> vertex ... wait: walk = [leg from vertex to arc-END rim]? Build the
    // loop by chaining vertices: start at the sharp vertex, take the leg
    // ending at the arc's stored END, the arc reversed, then the other leg
    // back to the vertex.
    let (arc_id, arc_start, arc_end) = cap.arc;
    let leg_to = |rim: u64| -> Result<(u64, u64, u64), String> {
        for leg in [cap.first_leg, cap.second_leg] {
            if leg.1 == rim {
                return Ok(leg);
            }
        }
        Err("blend network: cap leg for the rim is missing".into())
    };
    let leg_in = leg_to(arc_end)?; // stored rim -> vertex; walked vertex -> rim: forward = false
    let leg_out = leg_to(arc_start)?; // stored rim -> vertex; walked rim -> vertex: forward = true
    let pcurve_between = |from: Vec3, to: Vec3| -> Result<NurbsCurve, String> {
        let (u0, v0) = uv_of(from)?;
        let (u1, v1) = uv_of(to)?;
        crate::sweep_topology::parameter_line(u0, v0, u1, v1)
    };
    let vertex_point = |id: u64| -> Result<Vec3, String> {
        result
            .vertices
            .iter()
            .find(|vertex| vertex.id == id)
            .map(|vertex| vertex.point)
            .ok_or_else(|| format!("blend network: cap vertex {id} missing"))
    };
    let sharp = vertex_point(leg_in.2)?;
    let end_point = vertex_point(arc_end)?;
    let start_point = vertex_point(arc_start)?;
    // The arc's pcurve on the plane: map its rational control points.
    let arc_edge = edge_by_id(arc_id)?;
    let mut mapped = Vec::with_capacity(arc_edge.curve.control_points.len());
    for control in &arc_edge.curve.control_points {
        let point = control.point()?;
        let (u, v) = uv_of(point)?;
        mapped.push(crate::Vec4 {
            x: u * control.w,
            y: v * control.w,
            z: 0.0,
            w: control.w,
        });
    }
    let arc_pcurve_forward =
        NurbsCurve::new(arc_edge.curve.degree, arc_edge.curve.knots.clone(), mapped)?;
    let coedges = vec![
        CoedgeRecord {
            id: take_id(),
            edge_id: leg_in.0,
            forward: false,
            pcurve: pcurve_between(sharp, end_point)?,
        },
        CoedgeRecord {
            id: take_id(),
            edge_id: arc_id,
            forward: false,
            pcurve: arc_pcurve_forward.reversed()?,
        },
        CoedgeRecord {
            id: take_id(),
            edge_id: leg_out.0,
            forward: true,
            pcurve: pcurve_between(start_point, sharp)?,
        },
    ];
    let plane_normal = plane.normal(0.5, 0.5)?;
    let same_sense = plane_normal.dot(cap.outward) > 0.0;
    let loop_id = take_id();
    let face = FaceRecord {
        id: take_id(),
        surface: plane,
        same_sense,
        loops: vec![crate::topology::LoopRecord {
            id: loop_id,
            coedges,
        }],
        name: cap.name.map(|name| format!("{name}:CAP")),
    };
    let _ = cap.plane_normal;
    let shell_index = result
        .shells
        .iter()
        .position(|shell| !shell.faces.is_empty())
        .ok_or("blend network: the solid has no shell to sew the cap into")?;
    result.shells[shell_index].faces.push(face);
    Ok(())
}

fn vertex_point_of(solid: &BrepSolid, id: u64) -> Result<Vec3, String> {
    solid
        .vertices
        .iter()
        .find(|vertex| vertex.id == id)
        .map(|vertex| vertex.point)
        .ok_or_else(|| format!("blend network: vertex {id} missing"))
}

/// Why a corner's tangency point is not on a stripe's fitted rail.
enum RailMiss {
    /// The rail passes `f64` away from the point — a fit disagreement, not a
    /// span that stops short.
    OffRail(f64),
    /// The rail passes THROUGH the point, but at a station outside the
    /// marched span (the row would have to be extrapolated to reach it).
    OffSpan(f64),
    Projection(String),
}

/// A fitted rail on a curved carrier reproduces the marched contacts between
/// stations only to its interpolation error, which grows with the carrier's
/// size (a radius-50 rail with 64 stations lands ~3e-5 off), so the bar is the
/// model scale's, not the radius's.
fn corner_station_bar(scale: f64, radius: f64) -> f64 {
    1e-6 * (1.0 + scale.max(radius))
}

/// The row parameter where `row` passes through `point`, or why it does not.
/// The shared vertex sits at the ball's point; the rail end is healed onto it
/// afterwards.
fn rail_station(row: &NurbsCurve, point: Vec3, bar: f64) -> Result<f64, RailMiss> {
    let projection =
        crate::project_point_to_curve(row, point).map_err(RailMiss::Projection)?;
    if projection.distance > bar {
        return Err(RailMiss::OffRail(projection.distance));
    }
    if !(projection.u > RIM_MARGIN && projection.u < 1.0 - RIM_MARGIN) {
        return Err(RailMiss::OffSpan(projection.u));
    }
    Ok(projection.u)
}

/// Whether `rows` — a stripe marched at one rung of the overshoot ladder —
/// already reaches the tangency points the corner at one of its ends was
/// solved on, which is what the surgery will ask of it (`station_on`).
///
/// A CONVEX corner seats its ball inside the material, so its tangency points
/// sit on the rails INSIDE the edge's own span (the familiar setback) and any
/// rung reaches them.  A CONCAVE one seats it in the air PAST the vertex — the
/// base of a boss, a rib standing on a floor: the two floor rails cross one
/// radius beyond the corner — so the march has to overshoot far enough to
/// carry the rail there.  Reporting that here is what makes the ladder climb
/// for it instead of settling on the first rung and refusing later.
fn corner_rails_reach(
    rows: &FittedRows,
    first: &BlendMate,
    second: &BlendMate,
    kind: &VertexKind,
    bar: f64,
) -> bool {
    let reaches = |face_id: u64, point: Vec3| {
        let row = if first.face.id == face_id {
            &rows.cr
        } else if second.face.id == face_id {
            &rows.cs
        } else {
            return false;
        };
        rail_station(row, point, bar).is_ok()
    };
    let on = |corner: &Corner, face_id: u64| {
        corner
            .tangency_on(face_id)
            .map(|(point, _)| reaches(face_id, point))
            .unwrap_or(false)
    };
    match kind {
        VertexKind::Star { corner } => {
            on(corner, first.face.id) && on(corner, second.face.id)
        }
        // A miter only reads the tangency on the SHARED face; its other rim
        // vertices come from the seam.
        VertexKind::Miter { corner, shared, .. } => on(corner, shared.id),
        // Their own lanes decide where these stop.
        VertexKind::Reentrant { .. } | VertexKind::Flush => true,
    }
}

/// Whether the SHARP edge between `wall_a` and `wall_b` is convex, by the
/// rule `analyze_edge` uses: one wall's into-material direction dives below
/// the other wall's plane.
fn sharp_edge_is_convex(
    solid: &BrepSolid,
    sharp: &EdgeRecord,
    wall_a: &FaceRecord,
    wall_b: &FaceRecord,
    radius: f64,
) -> Result<bool, String> {
    let _ = solid;
    let t = (sharp.t0 + sharp.t1) * 0.5;
    let point = sharp.curve.evaluate(t)?;
    let tangent = sharp.curve.derivatives(t, 1)?[1].normalized()?;
    let probe = (radius * 0.25).max(1e-4);
    let into_a = crate::fillet::into_face_direction(wall_a, point, tangent, point, tangent, probe)?;
    let into_b = crate::fillet::into_face_direction(wall_b, point, tangent, point, tangent, probe)?;
    let outward = |face: &FaceRecord| -> Result<Vec3, String> {
        let projection = crate::project_point_to_surface(&face.surface, point)?;
        let normal = raw_normal(&face.surface, projection.u, projection.v)?;
        Ok(if face.same_sense { normal } else { normal.scale(-1.0) })
    };
    let (n_a, n_b) = (outward(wall_a)?, outward(wall_b)?);
    Ok(into_a.dot(n_b) < -1e-9 || into_b.dot(n_a) < -1e-9)
}

/// Whether a selected edge is convex: its ball (offset by the signed radii
/// the march uses) sits inside the material on both mates.
fn edge_is_convex(edge: &EdgeRecord, first: &BlendMate, second: &BlendMate) -> Result<bool, String> {
    let t = (edge.t0 + edge.t1) * 0.5;
    let mut inside = true;
    for mate in [first, second] {
        let uv = edge_uv_on_face(mate.coedge, edge, t)?;
        let normal = raw_normal(&mate.face.surface, uv[0], uv[1])?;
        let face_outward = if mate.face.same_sense { normal } else { normal.scale(-1.0) };
        // centre = p + rho * n_raw; inside the material iff that offset runs
        // against the outward normal.
        inside &= normal.scale(mate.rho).dot(face_outward) < 0.0;
    }
    Ok(inside)
}

/// The corner ball's cross-section between two of its tangency points: the
/// radius-`radius` arc about `center` running the short way from `from` to
/// `to`.  This is simultaneously the stripe's end section and one side of the
/// spherical corner patch, which is why the two meet exactly.
fn corner_section_arc(
    center: Vec3,
    radius: f64,
    from: Vec3,
    to: Vec3,
) -> Result<NurbsCurve, String> {
    let x_axis = from.sub(center).normalized()?;
    let toward = to.sub(center).normalized()?;
    let y_axis = toward.sub(x_axis.scale(toward.dot(x_axis))).normalized()?;
    let sweep = toward.dot(x_axis).clamp(-1.0, 1.0).acos();
    if !(sweep > 1e-9) {
        return Err("blend network: the corner section arc is degenerate".into());
    }
    crate::make_arc(center, x_axis, y_axis, radius, 0.0, sweep)
}

/// One planned split of a blend rail on a PERIODIC mate's seam meridian.
struct CarrierSeamSplit {
    face_id: u64,
    loop_index: usize,
    /// Position in the loop of the coedge that crosses the meridian.
    crossing_position: usize,
    /// Positions of the two seam coedges, the one ARRIVING at the eaten
    /// vertex and the one LEAVING it.
    seam_in_position: usize,
    seam_out_position: usize,
    seam_edge: u64,
    /// The vertex the blend ate, which the seam edge still ends at.
    dangling_vertex: u64,
    /// The mate's u level each seam coedge rides.
    level_in: f64,
    level_out: f64,
    period: f64,
    rail_edge: u64,
    /// Parameter on the rail edge's own curve where it crosses the meridian.
    rail_parameter: f64,
}

/// The traversal endpoints of a coedge: its walk runs start -> end, which is
/// the edge's t0 -> t1 only when the coedge is forward.
fn coedge_walk(solid: &BrepSolid, coedge: &CoedgeRecord) -> Result<(u64, u64), String> {
    let edge = solid
        .edges
        .iter()
        .find(|edge| edge.id == coedge.edge_id)
        .ok_or("blend network: loop references a missing edge")?;
    Ok(if coedge.forward {
        (edge.start_vertex_id, edge.end_vertex_id)
    } else {
        (edge.end_vertex_id, edge.start_vertex_id)
    })
}

/// The u range a coedge's pcurve covers, sampled (a fitted rail is not
/// monotone in its control points).
fn pcurve_u_range(pcurve: &NurbsCurve) -> Result<(f64, f64), String> {
    let [q0, q1] = pcurve.domain()?;
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for index in 0..=SEAM_SAMPLES {
        let q = q0 + (q1 - q0) * index as f64 / SEAM_SAMPLES as f64;
        let u = pcurve.evaluate(q)?.x;
        low = low.min(u);
        high = high.max(u);
    }
    Ok((low, high))
}

/// Sample count for the seam-crossing bracket and its bisection budget: the
/// rail is a cubic fit, so a 256-interval bracket cannot step over a single
/// crossing, and 64 bisections take the bracket to double precision.
const SEAM_SAMPLES: usize = 256;
const SEAM_BISECTIONS: usize = 64;

/// The pcurve parameter where `pcurve`'s u reaches `level`, bracketed on a
/// dense sample and bisected — no derivative, so a fitted rail that grazes the
/// meridian cannot send a Newton off the end of the row.
fn pcurve_u_crossing(pcurve: &NurbsCurve, level: f64) -> Result<Option<f64>, String> {
    let [q0, q1] = pcurve.domain()?;
    let at = |q: f64| -> Result<f64, String> { Ok(pcurve.evaluate(q)?.x - level) };
    let mut previous = (q0, at(q0)?);
    for index in 1..=SEAM_SAMPLES {
        let q = q0 + (q1 - q0) * index as f64 / SEAM_SAMPLES as f64;
        let value = at(q)?;
        if previous.1 == 0.0 {
            return Ok(Some(previous.0));
        }
        if (previous.1 < 0.0) != (value < 0.0) {
            let (mut low, mut high) = (previous, (q, value));
            for _ in 0..SEAM_BISECTIONS {
                let mid = 0.5 * (low.0 + high.0);
                let value = at(mid)?;
                if (low.1 < 0.0) != (value < 0.0) {
                    high = (mid, value);
                } else {
                    low = (mid, value);
                }
            }
            return Ok(Some(0.5 * (low.0 + high.0)));
        }
        previous = (q, value);
    }
    Ok(None)
}

/// Plan the seam split for one mate face, or `None` when its loops close.
///
/// A carrier closed in u carries its seam as ONE edge used TWICE by the same
/// loop, running from the far rim up to a vertex on the rim being blended.
/// When the selection blends the WHOLE of that rim, the new contact rail wraps
/// the carrier and crosses the meridian, while the seam edge still ends at the
/// vertex the blend ate: the loop is then open at both seam coedges.  The
/// repair keeps the carrier's seam-in-one-loop structure — the same structure
/// `edge/closed.rs` preserves for a closed blended rim — by splitting the
/// crossing rail at the meridian and trimming the seam to it.
fn plan_carrier_seam_split(
    result: &BrepSolid,
    rail_edges: &[u64],
    face_id: u64,
) -> Result<Option<CarrierSeamSplit>, String> {
    let face = result
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .find(|face| face.id == face_id)
        .ok_or("blend network: mate face lost before the seam split")?;
    if !face.surface.closed_directions()?.0 {
        return Ok(None);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let period = u1 - u0;
    if !(period > 0.0) {
        return Ok(None);
    }
    for (loop_index, loop_record) in face.loops.iter().enumerate() {
        let count = loop_record.coedges.len();
        if count < 3 {
            continue;
        }
        let mut walks = Vec::with_capacity(count);
        for coedge in &loop_record.coedges {
            walks.push(coedge_walk(result, coedge)?);
        }
        let breaks: Vec<usize> = (0..count)
            .filter(|&index| walks[index].1 != walks[(index + 1) % count].0)
            .collect();
        // Exactly two breaks, both at the same edge used twice: the seam pair
        // left behind at the eaten vertex.  Anything else is not this defect,
        // and `validate` reports it rather than this pass guessing.
        if breaks.len() != 2 {
            continue;
        }
        // One break follows the seam coedge ARRIVING at the eaten vertex, the
        // other precedes the one LEAVING it.  Which of the two comes first in
        // the stored array depends on where the loop happens to start, so try
        // both readings rather than assuming the arriving one leads.
        let read = |arriving_break: usize, leaving_break: usize| {
            let seam_in_position = arriving_break;
            let seam_out_position = (leaving_break + 1) % count;
            let seam_edge = loop_record.coedges[seam_in_position].edge_id;
            let paired = loop_record.coedges[seam_out_position].edge_id == seam_edge
                && loop_record
                    .coedges
                    .iter()
                    .filter(|coedge| coedge.edge_id == seam_edge)
                    .count()
                    == 2
                && walks[seam_in_position].1 == walks[seam_out_position].0;
            paired.then_some((seam_in_position, seam_out_position, seam_edge))
        };
        let Some((seam_in_position, seam_out_position, seam_edge)) =
            read(breaks[0], breaks[1]).or_else(|| read(breaks[1], breaks[0]))
        else {
            continue;
        };
        let dangling_vertex = walks[seam_in_position].1;
        // The u level each seam coedge rides, read at the eaten end.
        let level_at = |position: usize| -> Result<f64, String> {
            let coedge = &loop_record.coedges[position];
            let [q0, q1] = coedge.pcurve.domain()?;
            // The end that meets the eaten vertex: the walk's END for the
            // arriving coedge, its START for the leaving one.  A coedge's walk
            // runs its pcurve domain low -> high only when it is forward.
            let at_walk_end = position == seam_in_position;
            Ok(coedge
                .pcurve
                .evaluate(if at_walk_end == coedge.forward { q1 } else { q0 })?
                .x)
        };
        let level_in = level_at(seam_in_position)?;
        let level_out = level_at(seam_out_position)?;
        // The rail that crosses one of those meridians, strictly inside its
        // own u range.  Only a rail this operation just built is eligible:
        // pre-existing trims are never re-cut here.
        let band = 1e-9 * period;
        for (position, coedge) in loop_record.coedges.iter().enumerate() {
            if !rail_edges.contains(&coedge.edge_id) {
                continue;
            }
            let (low, high) = pcurve_u_range(&coedge.pcurve)?;
            for level in [level_in, level_out] {
                if low >= level - band || high <= level + band {
                    continue;
                }
                let Some(q) = pcurve_u_crossing(&coedge.pcurve, level)? else {
                    continue;
                };
                let edge = result
                    .edges
                    .iter()
                    .find(|edge| edge.id == coedge.edge_id)
                    .ok_or("blend network: the crossing rail vanished")?;
                let [q0, q1] = coedge.pcurve.domain()?;
                let walked = (q - q0) / (q1 - q0);
                let along = if coedge.forward { walked } else { 1.0 - walked };
                return Ok(Some(CarrierSeamSplit {
                    face_id,
                    loop_index,
                    crossing_position: position,
                    seam_in_position,
                    seam_out_position,
                    seam_edge,
                    dangling_vertex,
                    level_in,
                    level_out,
                    period,
                    rail_edge: edge.id,
                    rail_parameter: edge.t0 + (edge.t1 - edge.t0) * along,
                }));
            }
        }
    }
    Ok(None)
}

/// Split one coedge of `edge` at `along` (a fraction of the edge's own t
/// range) into the two pieces its walk meets first and second.
fn split_coedge(
    coedge: &CoedgeRecord,
    along: f64,
    low_edge: u64,
    high_edge: u64,
    take_id: &mut dyn FnMut() -> u64,
) -> Result<[CoedgeRecord; 2], String> {
    let [q0, q1] = coedge.pcurve.domain()?;
    let walked = if coedge.forward { along } else { 1.0 - along };
    let (first_pcurve, second_pcurve) = coedge.pcurve.split(q0 + (q1 - q0) * walked)?;
    // The walk meets the LOW piece of the edge first only when forward.
    let (first_edge, second_edge) = if coedge.forward {
        (low_edge, high_edge)
    } else {
        (high_edge, low_edge)
    };
    Ok([
        CoedgeRecord {
            id: coedge.id,
            edge_id: first_edge,
            forward: coedge.forward,
            pcurve: first_pcurve,
        },
        CoedgeRecord {
            id: take_id(),
            edge_id: second_edge,
            forward: coedge.forward,
            pcurve: second_pcurve,
        },
    ])
}

/// Carry out one [`plan_carrier_seam_split`]: split the rail edge at the
/// meridian (3D curve, both faces' pcurves), fold the wrapped piece back into
/// the carrier's domain, re-order the mate loop around the seam, and trim the
/// seam edge to the crossing.
fn apply_carrier_seam_split(
    result: &mut BrepSolid,
    take_id: &mut dyn FnMut() -> u64,
    plan: CarrierSeamSplit,
) -> Result<(), String> {
    let rail = result
        .edges
        .iter()
        .find(|edge| edge.id == plan.rail_edge)
        .ok_or("blend network: the crossing rail vanished before its split")?
        .clone();
    let seam_curve = result
        .edges
        .iter()
        .find(|edge| edge.id == plan.seam_edge)
        .ok_or("blend network: the carrier seam edge vanished")?
        .curve
        .clone();
    // The pcurve seed is only as good as the row's uv fit agrees with its 3D
    // fit — the `pcurve_consistency` drift, 1.5e-4 on this carrier.  Both the
    // rail and the seam meridian are on the carrier, so their 3D distance has
    // a genuine zero: refine the parameter onto it rather than trimming the
    // seam to a point 1.5e-4 off itself.
    let rail_parameter =
        refine_seam_crossing(&rail, &seam_curve, plan.rail_parameter)?;
    let span = rail.t1 - rail.t0;
    let along = (rail_parameter - rail.t0) / span;
    if !(1e-9..=1.0 - 1e-9).contains(&along) {
        return Err(format!(
            "blend network: the seam crossing on rail {} lands at {along:.9} of its own span — \
             the rail does not cross the carrier's meridian inside itself",
            rail.id
        ));
    }
    let crossing_point = rail.curve.evaluate(rail_parameter)?;
    let (low_curve, high_curve) = rail.curve.split(rail_parameter)?;
    let crossing_vertex = take_id();
    result.vertices.push(VertexRecord {
        id: crossing_vertex,
        point: crossing_point,
    });
    let low_edge = take_id();
    let high_edge = take_id();
    result.edges.push(EdgeRecord {
        id: low_edge,
        curve: low_curve,
        t0: rail.t0,
        t1: rail_parameter,
        start_vertex_id: rail.start_vertex_id,
        end_vertex_id: crossing_vertex,
        degenerate: false,
        name: rail.name.clone(),
    });
    result.edges.push(EdgeRecord {
        id: high_edge,
        curve: high_curve,
        t0: rail_parameter,
        t1: rail.t1,
        start_vertex_id: crossing_vertex,
        end_vertex_id: rail.end_vertex_id,
        degenerate: false,
        name: rail.name.clone(),
    });

    // Every OTHER loop that uses the rail — the blend face's own, and any
    // neighbour that was sewn onto it — takes the two pieces in place; only
    // the carrier's loop below needs re-ordering, because only it rides the
    // seam.  Splicing everywhere means no loop is left holding an edge id
    // that no longer exists.
    let mut spliced_elsewhere = false;
    for shell in &mut result.shells {
        for face in &mut shell.faces {
            let on_carrier = face.id == plan.face_id;
            for (loop_index, loop_record) in face.loops.iter_mut().enumerate() {
                if on_carrier && loop_index == plan.loop_index {
                    continue;
                }
                let Some(position) = loop_record
                    .coedges
                    .iter()
                    .position(|coedge| coedge.edge_id == plan.rail_edge)
                else {
                    continue;
                };
                let pieces = split_coedge(
                    &loop_record.coedges[position],
                    along,
                    low_edge,
                    high_edge,
                    take_id,
                )?;
                loop_record.coedges.splice(position..=position, pieces);
                spliced_elsewhere = true;
            }
        }
    }
    if !spliced_elsewhere {
        return Err(format!(
            "blend network: rail {} is used only by the carrier — its blend face is missing",
            rail.id
        ));
    }

    // The mate: split, fold the out-of-domain piece one period, then place the
    // piece that leaves the arriving meridian right after the seam and the one
    // that reaches the leaving meridian right before it.
    let face = result
        .shells
        .iter_mut()
        .flat_map(|shell| &mut shell.faces)
        .find(|face| face.id == plan.face_id)
        .ok_or("blend network: mate face lost during the seam split")?;
    let loop_record = &mut face.loops[plan.loop_index];
    let mut pieces = split_coedge(
        &loop_record.coedges[plan.crossing_position],
        along,
        low_edge,
        high_edge,
        take_id,
    )?;
    let centre = 0.5 * (plan.level_in + plan.level_out);
    for piece in &mut pieces {
        let [q0, q1] = piece.pcurve.domain()?;
        let middle = piece.pcurve.evaluate(0.5 * (q0 + q1))?.x;
        let shift = plan.period * ((middle - centre) / plan.period).round();
        if shift != 0.0 {
            for control in piece.pcurve.control_points.iter_mut() {
                control.x -= shift * control.w;
            }
        }
    }
    // The split parameter came from the 3D crossing, which the row's uv fit
    // reproduces only to `pcurve_consistency`; commit the endpoint the two
    // pieces provably share to the meridian exactly, so the mate's loop closes
    // in parameter space as well as in space.  The correction is the fit's own
    // residual, and a larger one means this is not the meridian at all.
    let band = 1e-6 * plan.period;
    let snap_band = 1e-3 * plan.period;
    for (index_of_piece, at_end) in [(0usize, true), (1usize, false)] {
        let controls = &mut pieces[index_of_piece].pcurve.control_points;
        let index = if at_end { controls.len() - 1 } else { 0 };
        let u = controls[index].x / controls[index].w;
        let level = if (u - plan.level_in).abs() <= (u - plan.level_out).abs() {
            plan.level_in
        } else {
            plan.level_out
        };
        if (u - level).abs() > snap_band {
            return Err(format!(
                "blend network: the split of rail {} lands {:.3e} from the carrier's meridian \
                 — the crossing is not on the seam",
                rail.id,
                (u - level).abs()
            ));
        }
        controls[index].x = level * controls[index].w;
    }
    // Which piece leaves the arriving meridian: the walk goes seam -> piece.
    let walk_start_u = |piece: &CoedgeRecord| -> Result<f64, String> {
        let [q0, q1] = piece.pcurve.domain()?;
        Ok(piece.pcurve.evaluate(if piece.forward { q0 } else { q1 })?.x)
    };
    let leaving_first = (walk_start_u(&pieces[0])? - plan.level_in).abs() <= band;
    let (leaving, arriving) = if leaving_first {
        (pieces[0].clone(), pieces[1].clone())
    } else {
        (pieces[1].clone(), pieces[0].clone())
    };
    if (walk_start_u(&leaving)? - plan.level_in).abs() > band {
        return Err(format!(
            "blend network: neither piece of rail {} starts on the carrier's meridian at \
             {:.6}",
            rail.id, plan.level_in
        ));
    }
    let mut rebuilt = Vec::with_capacity(loop_record.coedges.len() + 1);
    for (position, coedge) in loop_record.coedges.iter().enumerate() {
        if position == plan.crossing_position {
            continue;
        }
        if position == plan.seam_out_position {
            rebuilt.push(arriving.clone());
        }
        rebuilt.push(coedge.clone());
        if position == plan.seam_in_position {
            rebuilt.push(leaving.clone());
        }
    }
    loop_record.coedges = rebuilt;

    // The seam edge now ends on the crossing instead of the eaten vertex.
    let projection = crate::project_point_to_curve(&seam_curve, crossing_point)?;
    // The rail is a fit; the meridian is not.  Judge their meeting against the
    // kernel's named contract for how far a fitted edge may sit from the
    // carrier it belongs to, exactly as `support_crossing_uv` does.
    let reach = crate::KernelTolerances::for_solid(result, 1e-7)
        .pcurve_consistency
        .max(1e-6 * (1.0 + crossing_point.length()));
    if projection.distance > reach {
        return Err(format!(
            "blend network: the carrier seam edge {} does not pass through the rail's meridian \
             crossing (off by {:.3e})",
            plan.seam_edge, projection.distance
        ));
    }
    trim_edge_at(
        result,
        plan.seam_edge,
        projection.u,
        plan.dangling_vertex,
        crossing_vertex,
    )?;
    result.edges.retain(|edge| edge.id != plan.rail_edge);
    Ok(())
}

/// Refine a seam crossing seeded from the row's uv fit onto the 3D meeting of
/// the rail and the carrier's seam meridian.  Both curves lie on the carrier,
/// so their distance has a genuine minimum of zero there; a ternary search
/// over a bracket around the seed converges on it without a derivative (the
/// distance is V-shaped at a transversal crossing, and a fitted rail's
/// derivative is not worth trusting at 1e-9).
fn refine_seam_crossing(
    rail: &EdgeRecord,
    seam: &NurbsCurve,
    seed: f64,
) -> Result<f64, String> {
    let span = rail.t1 - rail.t0;
    let distance = |t: f64| -> Result<f64, String> {
        let point = rail.curve.evaluate(t)?;
        Ok(crate::project_point_to_curve(seam, point)?.distance)
    };
    // The uv fit and the 3D fit agree to the row's own accuracy, so the true
    // crossing is a small fraction of the rail away from the seed.
    let reach = (span.abs() * 0.05).max(1e-12);
    let (mut low, mut high) = (
        (seed - reach).max(rail.t0.min(rail.t1)),
        (seed + reach).min(rail.t0.max(rail.t1)),
    );
    for _ in 0..96 {
        if high - low <= 1e-15 * (1.0 + span.abs()) {
            break;
        }
        let a = low + (high - low) / 3.0;
        let b = high - (high - low) / 3.0;
        if distance(a)? <= distance(b)? {
            high = b;
        } else {
            low = a;
        }
    }
    let refined = 0.5 * (low + high);
    // Never take a refinement that is worse than the seed.
    if distance(refined)? <= distance(seed)? {
        Ok(refined)
    } else {
        Ok(seed)
    }
}
