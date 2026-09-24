//! Deleting a whole CORNER BLEND — a vertex blend and the strips that meet on
//! it — as ONE question.
//!
//! Round the three edges that meet at a box corner and the fillet leaves four
//! faces: three cylindrical STRIPS, one per edge, and the spherical CORNER the
//! rolling ball leaves where they run together. Selecting all four and asking
//! for them gone is "un-round this corner", and it is the commonest thing a
//! user does to a fillet after making it.
//!
//! One face at a time it cannot be answered, and the refusal was honest. A
//! strip's four neighbours are its two walls, the face it runs out into, and —
//! at the corner end — the vertex blend. The heal a strip asks for is "extend
//! the two walls, re-intersect them, and clip the recovered sharp edge against
//! both end caps"; here the recovered edge is the box's own sharp edge, and it
//! passes the corner sphere at `r·√2` — it never meets it. So
//! [`heal_open_transition_mixed`](super::open_heal) reported exactly what it
//! found:
//!
//! ```text
//! delete_face_and_heal: no re-intersection branch of the extended carriers
//! spans the deleted strip — refusing rather than emitting an invalid solid
//! ```
//!
//! No chain order fixes that. Deleting one strip out of a rounded corner is
//! not an under-solved problem, it is an ILL-POSED one: the corner sphere is
//! tangent to the strip that is leaving and to the two that are staying, and
//! what bounds it once its neighbour is gone is not a question the input
//! answers. The set is the question, and the set has a closed form.
//!
//! ## The construction
//!
//! The group is read off its own topology, never guessed:
//!
//! * the CORNER is the selected face with three boundary edges, and each of
//!   those edges is shared with another selected face;
//! * each of those three is a STRIP: a four-sided face, four distinct edges,
//!   four distinct neighbours, exactly one of them the corner. Opposite the
//!   corner edge is the strip's far CAP; flanking it are the two WALLS;
//! * the three strips' walls must be three distinct carriers, each carrying
//!   two strips. That is what makes the group a corner rather than three
//!   fillets that happen to touch: the walls close a cycle.
//!
//! Then, for each strip, exactly the single-strip heal's own closed form: the
//! two walls re-intersect in the recovered sharp edge, and the far cap clips
//! one end of it. What replaces the corner sphere is the OTHER end — the
//! recovered edge meets the third wall (the one this strip does not use) in a
//! point, and all three strips must name the SAME point. That agreement is the
//! whole gate on the geometry: three lines, each solved from a different pair
//! of planes, meeting at one vertex to within the plane tolerance. Disagree
//! and the group is refused.
//!
//! The topology that follows is the single-strip heal repeated: the corner
//! sphere's three vertices collapse onto the recovered corner vertex, each
//! strip's two far vertices collapse onto its own recovered far corner, every
//! surviving edge that ended on a collapsed vertex is widened along its own
//! line, each wall swaps its two strip rims for the two sharp edges that now
//! meet on it, each cap drops the transverse blend edge it wore, and every
//! touched carrier is re-trimmed and its pcurves refit.
//!
//! ## Nothing is left for the coalescer
//!
//! `delete_faces_and_heal` finishes every lane through
//! `merge_curve_continuation_edges`, which collapses the tangent continuation
//! edges a heal leaves behind. This lane leaves none, and that is structural
//! rather than lucky: a wall's two strip rims are REPLACED by the two recovered
//! sharp edges rather than having new pieces appended beside them, and every
//! surviving side edge is WIDENED along its own line rather than continued by a
//! second edge. So there is no continuation pair to find. Measured on both the
//! plain cube and the reported shelled document: the coalesced result is
//! bit-for-bit the counts the heal produced (6/12/8 and 14/33/21 faces / edges
//! / vertices), and a further coalesce takes 0 edges and 0 vertices.
//!
//! ## Scope
//!
//! Three strips on three PLANAR walls with planar caps — the box corner, and
//! any corner whose walls and caps are planes. A four-way star corner is not
//! taken (its corner face is four-sided, which the one-at-a-time chain still
//! reads as a transition strip; that behaviour is left alone), and a curved
//! wall or cap falls through to the chain and its existing refusal. See
//! `docs/developer/kernel-plans/direct-edit-heal-tail.md` §1/§4.

use super::*;

/// One blend strip of a corner group, read off its own loop.
pub(super) struct CornerStrip {
    /// The strip face itself.
    face_id: u64,
    /// The strip's boundary edge ids, in loop order.
    boundary: [u64; 4],
    /// The face across each boundary edge, in the same order.
    neighbours: [u64; 4],
    /// Index into `boundary` of the edge shared with the corner face.
    corner_index: usize,
    /// The two indices flanking `corner_index` — the walls whose
    /// re-intersection is the recovered sharp edge.
    walls: [usize; 2],
    /// The index opposite `corner_index` — the face the strip runs out into.
    cap_index: usize,
}

impl CornerStrip {
    fn wall_ids(&self) -> [u64; 2] {
        [self.neighbours[self.walls[0]], self.neighbours[self.walls[1]]]
    }

    fn cap_id(&self) -> u64 {
        self.neighbours[self.cap_index]
    }
}

/// A selection that reads as one vertex blend plus every strip that meets on
/// it.
pub(super) struct CornerBlendGroup {
    /// The shell the group lives in (all four faces share one).
    shell_index: usize,
    /// The vertex blend.
    corner_id: u64,
    /// The corner face's boundary edges.
    corner_edges: Vec<u64>,
    /// The strips, in the corner's own loop order.
    strips: Vec<CornerStrip>,
}

/// Read one strip of a corner group off its loop, or decline.
fn read_corner_strip(
    solid: &BrepSolid,
    strip_id: u64,
    corner_id: u64,
    selected: &HashSet<u64>,
) -> Option<CornerStrip> {
    let (shell_index, face_index) = find_face(solid, strip_id)?;
    let face = &solid.shells[shell_index].faces[face_index];
    if face.loops.len() != 1 || face.loops[0].coedges.len() != 4 {
        return None;
    }
    let mut boundary = [0u64; 4];
    for (index, coedge) in face.loops[0].coedges.iter().enumerate() {
        boundary[index] = coedge.edge_id;
    }
    if boundary.iter().copied().collect::<HashSet<u64>>().len() != 4 {
        return None;
    }
    let mut neighbours = [0u64; 4];
    for (index, &edge_id) in boundary.iter().enumerate() {
        neighbours[index] = other_face_of_edge(solid, edge_id, strip_id).ok()?;
    }
    if neighbours.iter().copied().collect::<HashSet<u64>>().len() != 4 {
        return None;
    }
    let corner_index = neighbours.iter().position(|id| *id == corner_id)?;
    let walls = [(corner_index + 1) % 4, (corner_index + 3) % 4];
    let cap_index = (corner_index + 2) % 4;
    // Everything the strip leans on has to SURVIVE — a wall or a cap that is
    // itself selected is a bigger question than one corner.
    for index in [walls[0], walls[1], cap_index] {
        if selected.contains(&neighbours[index]) {
            return None;
        }
    }
    Some(CornerStrip {
        face_id: strip_id,
        boundary,
        neighbours,
        corner_index,
        walls,
        cap_index,
    })
}

/// Does this selection read as one vertex blend plus every strip that meets on
/// it? Structural only — the geometry is gated in
/// [`heal_corner_blend_group`].
pub(super) fn classify_corner_blend_group(
    solid: &BrepSolid,
    face_ids: &[u64],
) -> Option<CornerBlendGroup> {
    // One corner + three strips. A four-way star corner wears a four-sided
    // corner face, which the one-at-a-time chain already reads its own way.
    if face_ids.len() != 4 {
        return None;
    }
    let selected: HashSet<u64> = face_ids.iter().copied().collect();
    if selected.len() != 4 {
        return None;
    }
    for &corner_id in face_ids {
        let Some((shell_index, face_index)) = find_face(solid, corner_id) else {
            return None;
        };
        let corner = &solid.shells[shell_index].faces[face_index];
        if corner.loops.len() != 1 || corner.loops[0].coedges.len() != 3 {
            continue;
        }
        let corner_edges: Vec<u64> = corner.loops[0]
            .coedges
            .iter()
            .map(|coedge| coedge.edge_id)
            .collect();
        if corner_edges.iter().copied().collect::<HashSet<u64>>().len() != 3 {
            continue;
        }
        let mut strips: Vec<CornerStrip> = Vec::with_capacity(3);
        for &edge_id in &corner_edges {
            let Ok(strip_id) = other_face_of_edge(solid, edge_id, corner_id) else {
                break;
            };
            if !selected.contains(&strip_id) {
                break;
            }
            let Some(strip) = read_corner_strip(solid, strip_id, corner_id, &selected) else {
                break;
            };
            // Every face of the group lives in the corner's own shell.
            if find_face(solid, strip_id).map(|(shell, _)| shell) != Some(shell_index) {
                break;
            }
            strips.push(strip);
        }
        if strips.len() != 3 {
            continue;
        }
        if strips
            .iter()
            .map(|strip| strip.face_id)
            .collect::<HashSet<u64>>()
            .len()
            != 3
        {
            continue;
        }
        // The walls close a CYCLE: three carriers, each carrying two of the
        // three strips. Anything else is not one corner.
        let mut wall_uses: HashMap<u64, usize> = HashMap::default();
        for strip in &strips {
            for wall in strip.wall_ids() {
                *wall_uses.entry(wall).or_insert(0) += 1;
            }
        }
        if wall_uses.len() != 3 || wall_uses.values().any(|count| *count != 2) {
            continue;
        }
        // A cap that is also a wall would be re-trimmed against a boundary it
        // is simultaneously rewriting; that is a different question.
        if strips
            .iter()
            .any(|strip| wall_uses.contains_key(&strip.cap_id()))
        {
            continue;
        }
        return Some(CornerBlendGroup {
            shell_index,
            corner_id,
            corner_edges,
            strips,
        });
    }
    None
}

/// Re-derive a side edge from its two flanking PLANES.
///
/// `relocate_open_side_edge` widens along the stored curve, and rebuilds only a
/// two-control-point line; `rederive_side_edge_on_carriers` goes through
/// `intersect_analytic_pair`, which declines plane × plane by design (there is
/// no closed form there because there is one HERE). A shelled solid's rim weld
/// leaves its straight edges as degree-1 curves with three control points, so
/// both of those decline an edge that is a plain line between two planes. This
/// is that line: the flanking pair re-intersected exactly, re-trimmed from the
/// endpoint it keeps to the corner it now reaches.
fn rederive_side_edge_on_planes(
    solid: &BrepSolid,
    edge: &EdgeRecord,
    planes: &HashMap<u64, Plane>,
    start_target: Option<(u64, Vec3)>,
    end_target: Option<(u64, Vec3)>,
    plane_tolerance: f64,
    tolerance: f64,
) -> Result<Option<EdgeRecord>, String> {
    let mut flanking: Vec<u64> = Vec::new();
    for shell in &solid.shells {
        for face in &shell.faces {
            if face
                .loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
                .any(|coedge| coedge.edge_id == edge.id)
            {
                flanking.push(face.id);
            }
        }
    }
    if flanking.len() != 2 {
        return Ok(None);
    }
    let (Some(plane_a), Some(plane_b)) = (planes.get(&flanking[0]), planes.get(&flanking[1])) else {
        return Ok(None);
    };
    let Some(line) = intersect_planes(plane_a, plane_b) else {
        return Ok(None);
    };
    let on_line = |point: Vec3| -> bool {
        let along = point.sub(line.point).dot(line.dir);
        point.sub(line.point.add(line.dir.scale(along))).length() <= plane_tolerance
    };
    let start_point = match start_target {
        Some((_, point)) => point,
        None => edge.curve.evaluate(edge.t0)?,
    };
    let end_point = match end_target {
        Some((_, point)) => point,
        None => edge.curve.evaluate(edge.t1)?,
    };
    if !on_line(start_point) || !on_line(end_point) {
        return Ok(None);
    }
    if start_point.sub(end_point).length() <= tolerance {
        return Ok(None);
    }
    let mut record = edge.clone();
    record.curve = make_line(start_point, end_point)?;
    record.t0 = 0.0;
    record.t1 = 1.0;
    if let Some((vertex, _)) = start_target {
        record.start_vertex_id = vertex;
    }
    if let Some((vertex, _)) = end_target {
        record.end_vertex_id = vertex;
    }
    Ok(Some(record))
}

/// The recovered geometry of one strip: the sharp edge its two walls make, and
/// where its far cap clips it.
struct StripPlan {
    /// The recovered far corner (sharp edge × the strip's cap plane).
    far_point: Vec3,
    /// The recovered corner vertex, read from this strip's own line.
    corner_point: Vec3,
}

/// Heal a corner blend group: the strips' walls re-intersect into three sharp
/// edges meeting at one recovered vertex, and the vertex blend goes with them.
pub(super) fn heal_corner_blend_group(
    solid: &BrepSolid,
    group: &CornerBlendGroup,
    op: &str,
) -> Result<BrepSolid, String> {
    let mut solid = solid.clone();
    let scale = solid_model_scale(&solid);
    let tolerance = (scale * 1e-7).max(1e-9);
    let plane_tolerance = (scale * 1e-6).max(1e-7);

    // --- Carriers: every wall and every cap must be planar in this slice ---
    let mut planes: HashMap<u64, Plane> = HashMap::default();
    for strip in &group.strips {
        for neighbour_id in strip
            .wall_ids()
            .into_iter()
            .chain(std::iter::once(strip.cap_id()))
        {
            if planes.contains_key(&neighbour_id) {
                continue;
            }
            let (ns, nf) = find_face(&solid, neighbour_id)
                .ok_or_else(|| format!("{op}: missing neighbour {neighbour_id}"))?;
            let plane = plane_of_surface(
                &solid.shells[ns].faces[nf].surface,
                plane_tolerance,
                op,
            )?;
            planes.insert(neighbour_id, plane);
        }
    }
    let wall_ids: HashSet<u64> = group
        .strips
        .iter()
        .flat_map(|strip| strip.wall_ids())
        .collect();

    // --- The group's own region gate (centre + reach), from its vertices ---
    let group_edges: HashSet<u64> = group
        .corner_edges
        .iter()
        .copied()
        .chain(
            group
                .strips
                .iter()
                .flat_map(|strip| strip.boundary.iter().copied()),
        )
        .collect();
    let mut group_vertices: Vec<u64> = Vec::new();
    for edge in &solid.edges {
        if group_edges.contains(&edge.id) {
            group_vertices.push(edge.start_vertex_id);
            group_vertices.push(edge.end_vertex_id);
        }
    }
    group_vertices.sort_unstable();
    group_vertices.dedup();
    if group_vertices.is_empty() {
        return Err(format!("{op}: corner blend group has no vertices"));
    }
    let mut centre = Vec3::default();
    for &vertex_id in &group_vertices {
        centre = centre.add(edge_point(&solid, vertex_id)?);
    }
    centre = centre.scale(1.0 / group_vertices.len() as f64);
    let mut reach = 0.0f64;
    for &vertex_id in &group_vertices {
        reach = reach.max(edge_point(&solid, vertex_id)?.sub(centre).length());
    }
    let reach = reach * 3.0 + tolerance;

    // --- Each strip's recovered edge, in closed form -----------------------
    let mut plans: Vec<StripPlan> = Vec::with_capacity(3);
    for strip in &group.strips {
        let [wall_a, wall_b] = strip.wall_ids();
        let line = intersect_planes(&planes[&wall_a], &planes[&wall_b]).ok_or_else(|| {
            format!(
                "{op}: the walls flanking blend face {} are parallel — they cannot \
                 re-intersect into a sharp edge",
                strip.face_id
            )
        })?;
        let far_point = intersect_line_plane(&line, &planes[&strip.cap_id()])
            .filter(|point| point.sub(centre).length() <= reach)
            .ok_or_else(|| {
                format!(
                    "{op}: the sharp edge recovered for blend face {} does not cross the \
                     face it runs out into",
                    strip.face_id
                )
            })?;
        // The corner vertex is where this strip's recovered edge meets the
        // wall it does NOT use — the third carrier of the cycle.
        let third = wall_ids
            .iter()
            .copied()
            .find(|id| *id != wall_a && *id != wall_b)
            .ok_or_else(|| format!("{op}: the corner's walls do not close a cycle"))?;
        let corner_point = intersect_line_plane(&line, &planes[&third])
            .filter(|point| point.sub(centre).length() <= reach)
            .ok_or_else(|| {
                format!(
                    "{op}: the sharp edge recovered for blend face {} never reaches the \
                     corner's third wall",
                    strip.face_id
                )
            })?;
        plans.push(StripPlan {
            far_point,
            corner_point,
        });
    }

    // The three lines were each solved from a different pair of planes; they
    // have to name ONE vertex. This is the gate on the geometry.
    let mut corner_point = Vec3::default();
    for plan in &plans {
        corner_point = corner_point.add(plan.corner_point);
    }
    corner_point = corner_point.scale(1.0 / plans.len() as f64);
    for plan in &plans {
        let drift = plan.corner_point.sub(corner_point).length();
        if drift > plane_tolerance {
            return Err(format!(
                "{op}: the blend strips' recovered edges miss each other by {drift:.6} — \
                 the selection is not one corner"
            ));
        }
    }
    for (index, plan) in plans.iter().enumerate() {
        if plan.far_point.sub(corner_point).length() <= tolerance {
            return Err(format!(
                "{op}: the sharp edge recovered for blend face {} collapses to the corner \
                 vertex",
                group.strips[index].face_id
            ));
        }
    }

    // --- New vertices and the three new sharp edges ------------------------
    let mut next_id = max_topology_id(&solid) + 1;
    let mut alloc = || {
        let value = next_id;
        next_id += 1;
        value
    };
    let corner_vertex = alloc();
    let mut new_vertices: Vec<(u64, Vec3)> = vec![(corner_vertex, corner_point)];
    let mut vertex_points: HashMap<u64, Vec3> = HashMap::default();
    vertex_points.insert(corner_vertex, corner_point);
    let mut sharp_edges: Vec<EdgeRecord> = Vec::with_capacity(3);
    // Old vertex -> the recovered vertex it collapses onto.
    let mut collapse: HashMap<u64, u64> = HashMap::default();
    for (index, strip) in group.strips.iter().enumerate() {
        let far_vertex = alloc();
        new_vertices.push((far_vertex, plans[index].far_point));
        vertex_points.insert(far_vertex, plans[index].far_point);
        let sharp_edge_id = alloc();
        sharp_edges.push(EdgeRecord {
            id: sharp_edge_id,
            curve: make_line(plans[index].far_point, corner_point)?,
            t0: 0.0,
            t1: 1.0,
            start_vertex_id: far_vertex,
            end_vertex_id: corner_vertex,
            degenerate: false,
            name: None,
        });
        // The strip's far vertices are the ends of its cap edge; its corner
        // vertices are the ends of the edge it shares with the vertex blend.
        for (boundary_index, target) in [
            (strip.cap_index, far_vertex),
            (strip.corner_index, corner_vertex),
        ] {
            let edge_id = strip.boundary[boundary_index];
            let edge = solid
                .edges
                .iter()
                .find(|edge| edge.id == edge_id)
                .ok_or_else(|| format!("{op}: missing edge {edge_id}"))?;
            collapse.insert(edge.start_vertex_id, target);
            collapse.insert(edge.end_vertex_id, target);
        }
    }
    // Every vertex the group wore has to go somewhere; an unmapped one would
    // be left dangling by the prune below.
    for &vertex_id in &group_vertices {
        if !collapse.contains_key(&vertex_id) {
            return Err(format!(
                "{op}: corner blend vertex {vertex_id} is not on a cap or corner edge \
                 (unexpected strip ordering)"
            ));
        }
    }

    // --- Relocate every surviving edge that ended on a collapsed vertex ----
    let pending = pending_edge_relocations(&solid, &group_edges, &collapse, &vertex_points);
    for (index, start_target, end_target) in pending {
        let mut record = solid.edges[index].clone();
        let widened = relocate_open_side_edge(
            &mut record,
            start_target,
            end_target,
            plane_tolerance,
            tolerance,
            op,
        )?;
        if !widened {
            // The flanking pair is two planes wherever the group's own walls
            // and caps meet, and that is the exact line the edge lies on.
            match rederive_side_edge_on_planes(
                &solid,
                &solid.edges[index],
                &planes,
                start_target,
                end_target,
                plane_tolerance,
                tolerance,
            )? {
                Some(rebuilt) => record = rebuilt,
                None => {
                    record = rederive_side_edge_on_carriers(
                        &solid,
                        &solid.edges[index],
                        start_target,
                        end_target,
                        plane_tolerance,
                        tolerance,
                        op,
                    )?
                }
            }
        }
        solid.edges[index] = record;
    }

    // Index the (now relocated) edges for the loop rewrites.
    let mut edges_by_id: HashMap<u64, EdgeRecord> = solid
        .edges
        .iter()
        .map(|edge| (edge.id, edge.clone()))
        .collect();
    for edge in &sharp_edges {
        edges_by_id.insert(edge.id, edge.clone());
    }
    // A coedge on a relocated edge already carries its new vertex; one on a
    // strip rim still carries the vertex that is about to collapse.
    let resolve = |vertex_id: u64| -> u64 { collapse.get(&vertex_id).copied().unwrap_or(vertex_id) };

    // --- Walls: swap each strip rim for the sharp edge that replaces it ----
    for (index, strip) in group.strips.iter().enumerate() {
        let sharp_edge_id = sharp_edges[index].id;
        let far_vertex = sharp_edges[index].start_vertex_id;
        for &wall_index in &strip.walls {
            let rim_edge = strip.boundary[wall_index];
            let neighbour_id = strip.neighbours[wall_index];
            let (ns, nf) = find_face(&solid, neighbour_id)
                .ok_or_else(|| format!("{op}: missing wall {neighbour_id}"))?;
            let face = &mut solid.shells[ns].faces[nf];
            let (loop_index, coedge_index) = locate_coedge(face, rim_edge).ok_or_else(|| {
                format!("{op}: wall {neighbour_id} does not use edge {rim_edge}")
            })?;
            let coedges = &face.loops[loop_index].coedges;
            let count = coedges.len();
            let previous = &coedges[(coedge_index + count - 1) % count];
            let next = &coedges[(coedge_index + 1) % count];
            let required_from = coedge_to_vertex(previous, &edges_by_id)
                .map(resolve)
                .ok_or_else(|| format!("{op}: could not resolve loop connectivity"))?;
            let required_to = coedge_from_vertex(next, &edges_by_id)
                .map(resolve)
                .ok_or_else(|| format!("{op}: could not resolve loop connectivity"))?;
            let forward = if required_from == far_vertex && required_to == corner_vertex {
                true
            } else if required_from == corner_vertex && required_to == far_vertex {
                false
            } else {
                return Err(format!(
                    "{op}: the recovered sharp edge does not close wall {neighbour_id}'s loop \
                     (unexpected connectivity)"
                ));
            };
            let new_coedge = CoedgeRecord {
                id: alloc(),
                edge_id: sharp_edge_id,
                forward,
                // Placeholder; the re-trim/refit pass below recomputes it.
                pcurve: make_line(Vec3::default(), Vec3::new(1.0, 0.0, 0.0))?,
            };
            face.loops[loop_index].coedges[coedge_index] = new_coedge;
        }
    }

    // --- Caps: drop the transverse blend edge the strip ran out on ---------
    for strip in &group.strips {
        let cap_edge = strip.boundary[strip.cap_index];
        let neighbour_id = strip.cap_id();
        let (ns, nf) = find_face(&solid, neighbour_id)
            .ok_or_else(|| format!("{op}: missing cap {neighbour_id}"))?;
        let face = &mut solid.shells[ns].faces[nf];
        let (loop_index, coedge_index) = locate_coedge(face, cap_edge)
            .ok_or_else(|| format!("{op}: cap {neighbour_id} does not use edge {cap_edge}"))?;
        face.loops[loop_index].coedges.remove(coedge_index);
        if face.loops[loop_index].coedges.is_empty() {
            return Err(format!("{op}: healing emptied a cap face loop"));
        }
    }

    // --- Prune the group, its edges and its vertices -----------------------
    let removed_faces: HashSet<u64> = group
        .strips
        .iter()
        .map(|strip| strip.face_id)
        .chain(std::iter::once(group.corner_id))
        .collect();
    solid.shells[group.shell_index]
        .faces
        .retain(|face| !removed_faces.contains(&face.id));
    solid.edges.retain(|edge| !group_edges.contains(&edge.id));
    let removed_vertices: HashSet<u64> = group_vertices.iter().copied().collect();
    solid
        .vertices
        .retain(|vertex| !removed_vertices.contains(&vertex.id));
    for edge in &sharp_edges {
        solid.edges.push(edge.clone());
    }
    for &(vertex_id, point) in &new_vertices {
        solid.vertices.push(VertexRecord {
            id: vertex_id,
            point,
        });
    }

    // --- Re-trim and refit every wall and cap ------------------------------
    let final_edges: HashMap<u64, EdgeRecord> = solid
        .edges
        .iter()
        .map(|edge| (edge.id, edge.clone()))
        .collect();
    let mut touched: HashSet<u64> = HashSet::default();
    let mut retrim_order: Vec<u64> = planes.keys().copied().collect();
    retrim_order.sort_unstable();
    for neighbour_id in retrim_order {
        let plane = &planes[&neighbour_id];
        let (ns, nf) = find_face(&solid, neighbour_id)
            .ok_or_else(|| format!("{op}: missing neighbour {neighbour_id}"))?;
        retrim_planar_face(
            &mut solid.shells[ns].faces[nf],
            plane,
            &final_edges,
            scale,
            op,
        )?;
        // The planar re-trim maps each edge's WHOLE curve; every edge of the
        // face that represents a strict SUBRANGE of its curve — a widened side
        // edge, but equally an untouched edge an earlier boolean trimmed —
        // needs the range fitter so its pcurve spans exactly `[t0, t1]`.
        touched.clear();
        touched.extend(
            solid.shells[ns].faces[nf]
                .loops
                .iter()
                .flat_map(|loop_record| loop_record.coedges.iter().map(|coedge| coedge.edge_id)),
        );
        refit_touched_pcurves(
            &mut solid.shells[ns].faces[nf],
            &final_edges,
            &touched,
            true,
            tolerance,
            op,
        )?;
    }
    let issues = solid.validate();
    if !issues.is_empty() {
        return Err(format!(
            "{op}: corner blend heal failed validation: {issues:?}"
        ));
    }
    Ok(solid)
}
