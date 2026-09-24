use super::*;

/// Constant-radius rolling-ball fillet of one edge (Golovanov §4.9 march,
/// §6.9 surgery; the cutter only where the march refuses).
pub fn fillet_edge(
    solid: &BrepSolid,
    edge_id: u64,
    radius: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    check_mixed_concavity(solid, edge_id, "fillet_edge")?;
    check_support_extent(solid, edge_id, radius, "fillet_edge")?;
    fillet_or_chamfer(solid, edge_id, radius, false, name, ToolEnds::default(), Lane::GeneralFirst)
}

/// `fillet_edge` with the cutter first: the input the corner-closure lanes
/// (`round_convex_corner`, the mixed-convexity closures) were written
/// against.  Their tests build it directly; production reaches those lanes
/// only through the group's sequential composition, which is cutter-first
/// for the same reason.
pub(crate) fn fillet_edge_cutter(
    solid: &BrepSolid,
    edge_id: u64,
    radius: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    check_mixed_concavity(solid, edge_id, "fillet_edge")?;
    check_support_extent(solid, edge_id, radius, "fillet_edge")?;
    fillet_or_chamfer(solid, edge_id, radius, false, name, ToolEnds::default(), Lane::CutterFirst)
}

/// Equal-leg chamfer of one edge (Golovanov §6.11: the same construction
/// with the arc's chord).
pub fn chamfer_edge(
    solid: &BrepSolid,
    edge_id: u64,
    distance: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    check_mixed_concavity(solid, edge_id, "chamfer_edge")?;
    check_support_extent(solid, edge_id, distance, "chamfer_edge")?;
    fillet_or_chamfer(solid, edge_id, distance, true, name, ToolEnds::default(), Lane::GeneralFirst)
}

/// Build and apply the chamfer tool for one straight edge from an already-built
/// cross-section profile (curve index 1 is the chamfer chord / blend wall).
/// Shared by the two-distance and distance-angle asymmetric entries.
fn apply_chamfer_offsets_profile(
    solid: &BrepSolid,
    cross: &EdgeCross,
    profile: &[NurbsCurve],
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    let mut tool = match &cross.path {
        EdgePath::Straight { direction, length } => {
            extrude_profile_brep(profile, *direction, *length)?
        }
        EdgePath::Circular { .. } => {
            return Err(
                "chamfer_edge_asymmetric: only straight edges on planar faces are supported \
                 in this slice (asymmetric chamfer on general/curved edges is out of scope)"
                    .into(),
            );
        }
    };
    if let Some(name) = name {
        // Side faces are emitted in input-curve order; the chamfer wall is the
        // second profile curve (index 1).
        let mut side_index = 0usize;
        for shell in &mut tool.shells {
            for face in &mut shell.faces {
                if side_index == 1 && face.name.is_none() {
                    face.name = Some(name.to_string());
                }
                side_index += 1;
                if side_index >= profile.len() {
                    break;
                }
            }
        }
    }
    apply_tool(solid, &tool, cross.convex)
}

/// Asymmetric (two-distance) chamfer of one STRAIGHT edge between two planar
/// faces (Golovanov §6.11): setback `d1` along face 1 and `d2` along face 2 —
/// the standard CAD "d1 × d2" bevel.  General/curved edges are out of scope for
/// this slice and return a clear error.
pub fn chamfer_edge_asymmetric(
    solid: &BrepSolid,
    edge_id: u64,
    d1: f64,
    d2: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if !(d1 > 0.0) || !(d2 > 0.0) || !d1.is_finite() || !d2.is_finite() {
        return Err("chamfer_edge_asymmetric: both setback distances must be positive".into());
    }
    // `analyze_edge` only uses the radius to size the orientation probe step;
    // the smaller setback keeps that probe inside both faces.
    let cross = analyze_edge(solid, edge_id, d1.min(d2))?;
    let profile = chamfer_cross_section_offsets(&cross, d1, d2)?;
    apply_chamfer_offsets_profile(solid, &cross, &profile, name)
}

/// Distance-angle chamfer of one STRAIGHT edge between two planar faces
/// (Golovanov §6.11): setback `d1` along face 1 and angle `angle_rad` between
/// the chamfer face and face 1.  `d2` is constructed geometrically in the
/// cross-section plane (see `chamfer_angle_second_distance`), then the
/// two-distance builder is applied.
pub fn chamfer_edge_angle(
    solid: &BrepSolid,
    edge_id: u64,
    d1: f64,
    angle_rad: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if !(d1 > 0.0) || !d1.is_finite() {
        return Err("chamfer_edge_angle: setback distance d1 must be positive".into());
    }
    let cross = analyze_edge(solid, edge_id, d1)?;
    let d2 = chamfer_angle_second_distance(&cross, d1, angle_rad)?;
    let profile = chamfer_cross_section_offsets(&cross, d1, d2)?;
    apply_chamfer_offsets_profile(solid, &cross, &profile, name)
}

/// Snap small trim/intersection gaps between blend edges and their vertices.
/// The boolean endpoint welder preserves gaps already inside the validation
/// band and limits repairs to `max(|radius| * 1e-3, 1e-4)`.
pub(super) fn heal_edge_vertex_gaps(solid: &mut BrepSolid, radius: f64) -> Result<(), String> {
    let search = (radius.abs() * 1e-3).max(1e-7);
    crate::boolean::commit_nearby_edge_endpoints(solid, search).map_err(String::from)
}

/// Resolve a point within 1e-3 of a trimmed edge.
fn resolve_edge_by_point(solid: &BrepSolid, point: Vec3) -> Result<u64, String> {
    match crate::topology::nearest_edge(solid, point) {
        Some((edge_id, distance)) if distance <= 1e-3 => Ok(edge_id),
        Some((_, distance)) => Err(format!(
            "fillet_edges: no edge within tolerance of the point (nearest {distance:.6})"
        )),
        None => Err("fillet_edges: solid has no edges".into()),
    }
}

/// Fillet (or chamfer) a GROUP of edges as ONE operation, and — for fillets —
/// round the convex "star" vertices where three or more of the selected edges
/// meet (Golovanov §6.9.7).  This is the whole multi-edge fillet in a single
/// kernel call: the caller passes the object plus one 3D point on each edge,
/// and the kernel orchestrates the filleting and corner blending against the
/// full topology (so acute corners resolve coherently instead of being
/// stitched edge-by-edge by the app).  A corner the kernel cannot round (e.g.
/// non-orthogonal beyond support, or a general no-common-ball star) is left as
/// the edge fillets rather than failing the whole group.
///
/// When the WHOLE selection cannot be blended as one unit — a shared convex
/// corner where a revolve axis/pole edge meets the adjacent cap edges can
/// defeat the sequential corner surgery even though each edge and every proper
/// SUBSET of the selection blends cleanly (the three fillets converge on the
/// pole with no single end face across the corner) — we do NOT hard-reject the
/// whole selection (which makes the app refuse it outright with "does not yet
/// support the selected edge geometry").  Instead we blend the LARGEST subset
/// of the selected edges that yields a VALID solid, dropping only the edge(s)
/// that cannot co-blend at the corner.  A selection that already composes is
/// returned unchanged (byte-identical) — the subset search only runs after the
/// full-group attempt errors.
///
/// `edge_names` (when `Some`) is the per-edge blend-FACE name parallel to
/// `edge_points` — each grown wall is named after ITS originating edge; `None`
/// names every wall with the single base `name` (the legacy/test behavior,
/// byte-identical to before). The whole `*_edges` family takes the same
/// `edge_names` slot in the same position.
pub fn fillet_edges(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    radius: f64,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if !(radius > 0.0) || !radius.is_finite() {
        return Err("fillet_edges: radius must be positive".into());
    }
    if edge_points.is_empty() {
        return Err("fillet_edges: no edges selected".into());
    }
    // The rolling ball must reach both supports of every SELECTED edge — see
    // `check_support_extent`.  Checked ONCE, here, against the solid the
    // selection was made on: inside the group build the faces have already
    // been eaten into by earlier blends of the same selection, and a fillet
    // legitimately runs off that remainder (two rounds pinching on a shared
    // face).  Dropping edges cannot rescue an oversized radius either, so
    // this runs before the subset search rather than inside it.
    let entry = if chamfer { "chamfer_edges" } else { "fillet_edges" };
    let mut selected_ids = Vec::with_capacity(edge_points.len());
    for point in edge_points {
        let edge_id = resolve_edge_by_point(solid, *point)?;
        check_mixed_concavity(solid, edge_id, entry)?;
        check_support_extent(solid, edge_id, radius, entry)?;
        selected_ids.push(edge_id);
    }
    // A corner where the selection mixes convexity — a convex edge dying into
    // the concave edges it meets — has no rolling-ball closure at all: the
    // blends run out against each other partway along, which is the vertex
    // blend the stripe network does not construct yet.  It is TERMINAL, and
    // for the same reason as the two checks above it runs once, here: neither
    // the cutter composition nor the subset search below can rescue it.  The
    // cutter "succeeds" on this shape by shredding the solid (a sliver end cap
    // per corner, the carrier face split and renamed), which is worse than
    // the named refusal — see the 2026-09-09 rib-spine report.
    if !chamfer {
        check_mixed_corner_convexity(solid, &selected_ids, edge_points, edge_names, radius, name)?;
    }

    match fillet_edges_group(solid, edge_points, edge_names, radius, chamfer, name) {
        Ok(result) => Ok(result),
        Err(group_err) => {
            // Fewer than two edges: nothing to drop, so the group error is final.
            // Cap the combinatorial search so a large malformed selection cannot
            // explode (the full group carries the common case; the search is a
            // rare fallback).
            let n = edge_points.len();
            if n < 2 || n > 12 {
                return Err(group_err);
            }
            // Drop the fewest edges first (largest surviving subset), trying the
            // drop-sets in lexicographic order so the result is deterministic.
            // Return the first subset that blends to a VALID (watertight) solid.
            for drop in 1..n {
                for dropped in index_combinations(n, drop) {
                    let kept: Vec<Vec3> = (0..n)
                        .filter(|i| !dropped.contains(i))
                        .map(|i| edge_points[i])
                        .collect();
                    // Subset the per-edge blend-face names with the IDENTICAL
                    // drop-set so `kept_names[k]` still names `kept[k]`.
                    let kept_names: Option<Vec<String>> = edge_names.map(|names| {
                        (0..n)
                            .filter(|i| !dropped.contains(i))
                            .map(|i| names[i].clone())
                            .collect()
                    });
                    if let Ok(result) = fillet_edges_group(
                        solid,
                        &kept,
                        kept_names.as_deref(),
                        radius,
                        chamfer,
                        name,
                    ) {
                        if result.validate().is_empty() {
                            return Ok(result);
                        }
                    }
                }
            }
            Err(group_err)
        }
    }
}

/// The blend-FACE name for the `i`-th selected edge: its per-edge name when the
/// caller supplied the parallel `edge_names` (feature path — each wall named
/// after its originating edge, `{fid}:BLEND:{edge}`), else the single base
/// `name` for every wall (the legacy/test path, byte-identical to before).
fn per_edge_name<'a>(
    edge_names: Option<&'a [String]>,
    base: Option<&'a str>,
    i: usize,
) -> Option<&'a str> {
    match edge_names {
        Some(names) => names.get(i).map(|value| value.as_str()),
        None => base,
    }
}

/// The star-corner patch name: `{base}:CORNER:{sorted+join of adjacent edge
/// names}` when per-edge names were supplied (feature path), else the single
/// `base` (legacy/test path — the corner keeps the wall name, pre-change
/// behavior).  `base` is `{fid}:BLEND` and each `edge_names[i]` is the composed
/// `{fid}:BLEND:{edge}`, so stripping the `{base}:` prefix recovers the bare
/// originating-edge name for the join.  Unique per corner: two distinct corners
/// never share the same set of >=3 selected edges.
fn corner_face_name(
    edge_names: Option<&[String]>,
    base: Option<&str>,
    adjacent: &[usize],
) -> Option<String> {
    match (edge_names, base) {
        (Some(names), Some(base)) => {
            let prefix = format!("{base}:");
            let mut raws: Vec<&str> = adjacent
                .iter()
                .filter_map(|&i| names.get(i))
                .map(|composed| composed.strip_prefix(&prefix).unwrap_or(composed.as_str()))
                .collect();
            raws.sort_unstable();
            raws.dedup();
            Some(format!("{base}:CORNER:{}", raws.join("+")))
        }
        _ => base.map(|value| value.to_string()),
    }
}

/// **A corner with two concave edges cannot also take a convex one.**
///
/// Where selected edges meet, the rolling ball has to touch every face around
/// the vertex from ONE side.  A convex edge running into the concave edges it
/// meets (a rib spine dying into the fillets at its own base) asks the ball to
/// sit under the shared face for one blend and over it for the other, so no
/// ball seats there and the corner has no closure: physically the convex blend
/// runs out against the concave beads partway along the edge, which is a vertex
/// blend the stripe network does not construct
/// (`docs/developer/kernel-plans/fillet-stripe-network.md`).
///
/// The re-entrant vertex of a notch — ONE concave edge, the rest convex — is
/// NOT this: the ball rolls round that single concave edge from one convex
/// blend to the next, and `round_concave_chain_corner` / `round_convex_corner`
/// close it with the horn-torus sector.  Only its mirror is refused.
///
/// This is TERMINAL — deliberately not a fall-through to the cutter
/// composition.  The cutter answers this shape with a watertight but shredded
/// solid: a sliver end cap at each unclosed corner and the carrier face split
/// so the original name lands on a fragment, which breaks every downstream
/// reference to it.  A named refusal that says which edges to separate is
/// worth more than that solid.
fn check_mixed_corner_convexity(
    solid: &BrepSolid,
    edge_ids: &[u64],
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    radius: f64,
    name: Option<&str>,
) -> Result<(), String> {
    match crate::blend::mixed_convexity_corner(solid, edge_ids, radius) {
        Some(corner) => Err(mixed_corner_message(&corner, edge_points, edge_names, name)),
        None => Ok(()),
    }
}

/// The refusal text for [`check_mixed_corner_convexity`], naming the edges the
/// way the user selected them (their originating edge names when the feature
/// layer supplied them, else the picked point).
fn mixed_corner_message(
    corner: &crate::blend::MixedCorner,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    name: Option<&str>,
) -> String {
    let label = |index: usize| -> String {
        let named = edge_names.and_then(|names| names.get(index)).map(|composed| {
            match name {
                Some(base) => composed
                    .strip_prefix(&format!("{base}:"))
                    .unwrap_or(composed)
                    .to_string(),
                None => composed.clone(),
            }
        });
        named.unwrap_or_else(|| match edge_points.get(index) {
            Some(point) => format!("the edge at ({:.3}, {:.3}, {:.3})", point.x, point.y, point.z),
            None => format!("selection #{}", index + 1),
        })
    };
    let list = |indices: &[usize]| -> String {
        let parts: Vec<String> = indices.iter().map(|index| label(*index)).collect();
        match parts.split_last() {
            None => "none".to_string(),
            Some((last, [])) => last.clone(),
            Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        }
    };
    format!(
        "fillet_edges: the selection mixes convexity at the corner ({:.3}, {:.3}, {:.3}) — convex \
         {} meets concave {} there. One rolling ball cannot touch the face they share from both \
         sides at once, so that corner has no closure: the convex blend runs out against the \
         concave ones partway along the edge, and this kernel does not build that vertex blend \
         yet. Blend the concave edges in one fillet and the convex ones in a later \
         fillet — both orders build, and concave-first is the tidier result.",
        corner.point.x,
        corner.point.y,
        corner.point.z,
        list(&corner.convex),
        list(&corner.concave),
    )
}

/// All ways to choose `k` distinct indices from `0..n`, in lexicographic order.
fn index_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // Advance to the next combination (like counting with carry).
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Blend the WHOLE selection as one group (the single-shot multi-edge fillet).
/// Errors if any selected edge or the shared-corner surgery cannot compose;
/// `fillet_edges` wraps this with a maximal-valid-subset fallback.
fn fillet_edges_group(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    radius: f64,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    use rustc_hash::FxHashSet as HashSet;

    // Fuse-first operand heal (Lever A) before the multi-edge surgery: snap the
    // input's near-coincident / off-plane vertices (e.g. a revolve pole apex
    // sitting a few microns off the axis) to exact and re-anchor incident
    // edges.  The selected edges are resolved geometrically below, so a
    // sub-heal_tol vertex move never changes which edges are picked; a clean
    // input is left byte-identical.
    let mut healed_input = solid.clone();
    let heal_policy = crate::KernelTolerances::for_solid(&healed_input, 1e-7);
    crate::heal::heal_operands(&mut healed_input, &heal_policy)?;
    let solid = &healed_input;

    // 1. Detect convex corners from the ORIGINAL solid: a point that is an
    //    endpoint of >=3 of the selected edges (a cube/prism-style vertex).
    let mut endpoints: Vec<(Vec3, usize)> = Vec::with_capacity(edge_points.len() * 2);
    // The extent each selected edge has BEFORE any blend trims it, so the
    // sequential build can run every cutter through the shared corners.
    let mut original_extents: Vec<(Vec3, Vec3)> = Vec::with_capacity(edge_points.len());
    let mut selected_edge_ids: Vec<u64> = Vec::with_capacity(edge_points.len());
    for (i, point) in edge_points.iter().enumerate() {
        let edge_id = resolve_edge_by_point(solid, *point)?;
        let edge = solid
            .edges
            .iter()
            .find(|e| e.id == edge_id)
            .ok_or("fillet_edges: resolved edge vanished")?;
        let (start, end) = (edge.curve.evaluate(edge.t0)?, edge.curve.evaluate(edge.t1)?);
        endpoints.push((start, i));
        endpoints.push((end, i));
        original_extents.push((start, end));
        selected_edge_ids.push(edge_id);
    }
    let mut corners: Vec<Vec3> = Vec::new();
    // The selected-edge INPUT INDICES meeting at each star corner (parallel to
    // `corners`), sorted — used to name the corner patch after its adjacent
    // edges (`{fid}:BLEND:CORNER:{e_a}+{e_b}+…`), UNIQUE per corner because no
    // two distinct corners share the same set of >=3 selected edges.
    let mut corner_edges: Vec<Vec<usize>> = Vec::new();
    let mut chain_corner_count = 0usize;
    let mut chain_corners: Vec<(Vec3, [usize; 2])> = Vec::new();
    let mut used = vec![false; endpoints.len()];
    for i in 0..endpoints.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut edges_here: HashSet<usize> = HashSet::default();
        edges_here.insert(endpoints[i].1);
        for j in (i + 1)..endpoints.len() {
            if used[j] {
                continue;
            }
            if endpoints[i].0.sub(endpoints[j].0).length() < 1e-6 {
                used[j] = true;
                edges_here.insert(endpoints[j].1);
            }
        }
        if edges_here.len() >= 3 {
            corners.push(endpoints[i].0);
            let mut adjacent: Vec<usize> = edges_here.into_iter().collect();
            adjacent.sort_unstable();
            corner_edges.push(adjacent);
        } else if edges_here.len() == 2 {
            chain_corner_count += 1;
            let mut adjacent = edges_here.into_iter().collect::<Vec<_>>();
            adjacent.sort_unstable();
            chain_corners.push((endpoints[i].0, [adjacent[0], adjacent[1]]));
        }
    }

    // A FACE selection may contain several disconnected boundary components
    // (the common example is an outer perimeter plus a circular hole rim).
    // Miter composition is only meaningful inside one connected edge graph.
    // Feeding every component into the same INTERSECT/UNION combines multiple
    // independently filleted copies of otherwise untouched support faces; the
    // boolean then imprints those coincident copies and can leave redundant
    // seams (the reported through-hole cylinder split into three faces).
    //
    // Partition by shared original endpoints and process components in input
    // order.  Each recursive call sees one connected component, so it follows
    // the existing miter/sequential path without recursion cycling.  Distinct
    // components are then composed sequentially on the evolving solid.
    let mut component_of = vec![usize::MAX; edge_points.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for seed in 0..edge_points.len() {
        if component_of[seed] != usize::MAX {
            continue;
        }
        let component_index = components.len();
        component_of[seed] = component_index;
        let mut component = vec![seed];
        let mut cursor = 0;
        while cursor < component.len() {
            let current = component[cursor];
            cursor += 1;
            for candidate in 0..edge_points.len() {
                if component_of[candidate] != usize::MAX {
                    continue;
                }
                let connected = [original_extents[current].0, original_extents[current].1]
                    .into_iter()
                    .any(|a| {
                        [original_extents[candidate].0, original_extents[candidate].1]
                            .into_iter()
                            .any(|b| a.sub(b).length() < 1e-6)
                    });
                if connected {
                    component_of[candidate] = component_index;
                    component.push(candidate);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    if components.len() > 1 {
        let mut separated = solid.clone();
        for component in components {
            let points = component
                .iter()
                .map(|index| edge_points[*index])
                .collect::<Vec<_>>();
            let names = edge_names.map(|all| {
                component
                    .iter()
                    .map(|index| all[*index].clone())
                    .collect::<Vec<_>>()
            });
            separated =
                fillet_edges_group(&separated, &points, names.as_deref(), radius, chamfer, name)?;
        }
        return Ok(separated);
    }

    // 2a. The stripe network (blend/network.rs) is the lane for EVERY
    //     selection: each stripe is marched against THIS solid, each shared
    //     vertex is solved before anything is cut — a star closed by a patch
    //     of the corner ball, a two-edge corner by the seam between the two
    //     blends, a re-entrant corner by a horn torus, a tangent pair by a
    //     flush join, an unselected tangent continuation by a cap — and
    //     nothing is subtracted, so no cutter can overshoot into a neighbour
    //     and no leftover cap has to be identified afterwards.  It refuses
    //     BY NAME on what it does not yet construct (mixed-convexity corners,
    //     no-common-ball stars, chamfer corners, pinched edges), and those
    //     fall through to the cutter composition below.
    if std::env::var("BREP_NO_NETWORK").is_err() {
        let network_names: Vec<Option<String>> = (0..edge_points.len())
            .map(|index| per_edge_name(edge_names, name, index).map(str::to_string))
            .collect();
        let network_corner_name =
            |adjacent: &[usize]| corner_face_name(edge_names, name, adjacent);
        match crate::blend::blend_star_network(
            solid,
            &selected_edge_ids,
            radius,
            chamfer,
            &network_names,
            &network_corner_name,
        ) {
            Ok(mut network) => {
                // Fail-safe like the rest of the ladder: a heal or validation
                // problem in the network result falls through to the cutter,
                // it does not fail the group.  So does an UNTRIMMED result: the
                // network re-trims only each stripe's two mates, and
                // `check_blend_interference` is what notices a third face
                // crossing the swept volume -- `validate` cannot, because a
                // self-intersecting solid is still a watertight one.
                let entry = if chamfer { "chamfer_edges" } else { "fillet_edges" };
                let healed = heal_edge_vertex_gaps(&mut network, radius);
                let issues = network.validate();
                let interference =
                    check_blend_interference(solid, &network, &selected_edge_ids, entry);
                if healed.is_ok() && issues.is_empty() && interference.is_ok() {
                    return Ok(network);
                }
                if std::env::var("BREP_DEBUG_NETWORK").is_ok() {
                    eprintln!(
                        "network result rejected: heal={healed:?} issues={issues:?} \
                         interference={interference:?}"
                    );
                    dump_loops_debug("INPUT", solid);
                    dump_loops_debug("RESULT", &network);
                }
            }
            Err(refusal) => {
                if std::env::var("BREP_DEBUG_NETWORK").is_ok() {
                    eprintln!("network refused: {refusal}");
                }
            }
        }
    }

    // 2. Build the blends.
    //
    //    CHAIN corners (§6.9.6 — exactly TWO selected edges share a vertex):
    //    the sequential build truncates the later blend where it runs into the
    //    earlier one and closes it with a flat bulkhead across the fillet
    //    channel — a hard step, not a transition.  For selections containing
    //    chain corners, blend each edge FULL-LENGTH on the ORIGINAL solid and
    //    INTERSECT the per-edge results instead: the removal volumes union, so
    //    adjacent blends run through the shared corner and trim each other
    //    along their intersection curve — the standard MITER corner, tangent
    //    to the shared face at the seam's tangency end.  Non-adjacent edges
    //    are unaffected (their removals are disjoint, intersection ≡
    //    sequential).
    //
    //    Selections without chain corners keep the sequential build unchanged
    //    (star corners are rounded in step 3 against exactly the sequential
    //    geometry round_convex_corner was built for).
    //    CONVEXITY GUARD: a convex blend REMOVES material (fillet = orig −
    //    cut), a concave blend ADDS it (orig + pad).  Full-length blends
    //    combine as orig − ∪cuts + ∪pads, so the per-edge results compose by
    //    INTERSECTION when every edge is convex and by UNION when every edge
    //    is concave; a mixed selection has no single composition and falls
    //    back to the sequential build.
    // Fillet stars keep the sequential topology required by their sphere
    // patches. Chamfer stars have no subsequent corner patch: compose their
    // full-length removals too, so all bevel planes meet at the corner.
    let miter_operation = if (chain_corner_count > 0 || !corners.is_empty())
        && (chamfer || corners.is_empty())
    {
        let mut any_convex = false;
        let mut any_concave = false;
        for point in edge_points {
            // Convexity is a DIHEDRAL property, so it is read from the general
            // scan (`scan_dihedral`) — the same one `check_mixed_concavity`
            // has already run over every selected edge.  `analyze_edge` used
            // to answer here and it ALSO refuses an edge the EXACT CUTTER
            // cannot build: a straight edge whose mates are not both planes,
            // a circular one whose mates do not share its axis.  Reading that
            // refusal as "unknown convexity" dropped the §6.9.6 miter for
            // every selection touching a curved carrier — the groove rim of a
            // bored box, say — and the sequential build then left a bulkhead
            // face standing in the shared corner (the 2026-09-10 "chamfer
            // produced a spurious face" report).
            match resolve_edge_by_point(solid, *point)
                .and_then(|edge_id| super::analyze::scan_dihedral(solid, edge_id))
            {
                // A mixed edge has no single composition either; it takes the
                // `_ => None` arm below through both flags.
                Ok(profile) if profile.samples > 0 && !profile.is_mixed() => {
                    any_convex |= profile.any_convex;
                    any_concave |= profile.any_concave;
                }
                // Unknown edge class: let the sequential path produce its own
                // (more specific) error or result.
                _ => {
                    any_convex = true;
                    any_concave = true;
                    break;
                }
            }
        }
        match (any_convex, any_concave) {
            (true, false) => Some(crate::BooleanOperation::Intersect),
            (false, true) => Some(crate::BooleanOperation::Union),
            _ => None,
        }
    } else {
        None
    };

    // Fillet/chamfer each edge in turn, resolving its point on the evolving
    // solid (ids shift as earlier fillets rewrite topology; the midpoint of an
    // edge is untouched by the corner surgery of the others).  This is the
    // baseline build used directly for non-miter selections AND as the
    // fallback when the miter composition below cannot reassemble.
    // The composition's per-edge lane: cutter-first ONLY when a corner
    // closure will run afterwards and read cutter-shaped topology (a star or
    // a chain corner).  A selection with no shared vertex — a lone closed
    // rim, a lone chamfer, disconnected edges — has no such closure, so each
    // edge gets the march first there too.
    let composition_lane = if corners.is_empty() && chain_corner_count == 0 {
        Lane::GeneralFirst
    } else {
        Lane::CutterFirst
    };
    let build_sequential = || -> Result<BrepSolid, String> {
        let mut sequential = solid.clone();
        for (index, point) in edge_points.iter().enumerate() {
            let edge_id = resolve_edge_by_point(&sequential, *point)?;
            // Earlier cutters in this loop TRIM the edges that share a corner
            // with them; extend this cutter back over what they took so the
            // two removal volumes union through the corner (§6.9.6 miter)
            // instead of leaving a wedge of material standing behind a flush
            // end cap.  Untouched edges get a zero pad and the historical
            // flush cutter.
            let ends = tool_ends_to_original_extent(
                &sequential,
                edge_id,
                radius,
                original_extents[index],
                if chamfer { &[] } else { &corners },
            );
            // This edge's blend wall carries the name of THIS input edge
            // (`edge_names[index]`); a smooth chain that engulfs several edges
            // is named after the FIRST such input edge processed here (input
            // order), the chain's representative.
            let edge_name = per_edge_name(edge_names, name, index);
            sequential = fillet_or_chamfer(
                &sequential,
                edge_id,
                radius,
                chamfer,
                edge_name,
                ends,
                composition_lane,
            )?;
        }
        Ok(sequential)
    };

    let mut result = if let Some(operation) = miter_operation {
        let options = crate::BooleanOptions::default();
        let mut combined: Result<Option<BrepSolid>, String> = Ok(None);
        for (index, point) in edge_points.iter().enumerate() {
            let edge_id = resolve_edge_by_point(solid, *point)?;
            let edge_name = per_edge_name(edge_names, name, index);
            let blended = {
                let entry = if chamfer { "chamfer_edges" } else { "fillet_edges" };
                check_mixed_concavity(solid, edge_id, entry)?;
                check_support_extent(solid, edge_id, radius, entry)?;
                fillet_or_chamfer(
                    solid,
                    edge_id,
                    radius,
                    chamfer,
                    edge_name,
                    ToolEnds::default(),
                    Lane::CutterFirst,
                )?
            };
            combined = match combined {
                Err(e) => Err(e),
                Ok(None) => Ok(Some(blended)),
                Ok(Some(previous)) => {
                    crate::boolean_operation(&previous, &blended, operation, &options)
                        .map(Some)
                        .map_err(|error| {
                            format!("fillet_edges: chain-corner miter composition failed: {error}")
                        })
                }
            };
            if combined.is_err() {
                break;
            }
        }
        // A §6.9.6 miter (per-edge blends intersected/unioned through the
        // shared chain corners) can fail to reassemble on faces whose
        // fragmented boundary does not close — e.g. a planar cap whose ENTIRE
        // perimeter is selected, where fragment_face reports an "incomplete
        // run".  Rather than let `fillet_edges` silently DROP a selected edge
        // to recover a valid subset (the reported defect: not every edge of
        // the face gets a fillet), fall back to the sequential build, which
        // blends EVERY selected edge (chain corners get a flat bulkhead
        // instead of a miter).  Only if that also fails to produce a valid
        // solid do we surface the miter error so the caller's
        // maximal-valid-subset search can still run.
        match combined {
            Ok(Some(mitered)) => mitered,
            Ok(None) => return Err("fillet_edges: no edges selected".into()),
            Err(miter_err) => match build_sequential() {
                Ok(seq) if seq.validate().is_empty() => seq,
                _ => return Err(miter_err),
            },
        }
    } else {
        build_sequential()?
    };

    // A re-entrant vertex of a selected face perimeter has two selected,
    // convex cap-wall edges and one UNSELECTED concave wall-wall edge.  The
    // two cutter volumes only touch there, so the boolean miter leaves a
    // triangular planar end-cap instead of carrying the rolling ball around
    // the corner.  Close that exact orthogonal class with its horn-torus
    // sector (major radius = minor radius = fillet radius).
    if !chamfer {
        for (corner, adjacent) in &chain_corners {
            let has_concave_unselected_edge = solid.edges.iter().any(|edge| {
                if selected_edge_ids.contains(&edge.id) {
                    return false;
                }
                let Ok(a) = edge.curve.evaluate(edge.t0) else {
                    return false;
                };
                let Ok(b) = edge.curve.evaluate(edge.t1) else {
                    return false;
                };
                (a.sub(*corner).length() < 1e-6 || b.sub(*corner).length() < 1e-6)
                    && analyze_edge(solid, edge.id, radius)
                        .map(|cross| !cross.convex)
                        .unwrap_or(false)
            });
            if !has_concave_unselected_edge {
                continue;
            }
            let corner_name = corner_face_name(edge_names, name, adjacent);
            if let Ok(rounded) = crate::blend::round_concave_chain_corner(
                &result,
                solid,
                *corner,
                [
                    selected_edge_ids[adjacent[0]],
                    selected_edge_ids[adjacent[1]],
                ],
                radius,
                corner_name.as_deref(),
            ) {
                result = rounded;
            }
        }
    }

    // 3. Round the convex corners (fillets only — chamfers keep sharp
    //    vertices).  A corner that cannot be rounded is left as the edge
    //    fillets so the group still succeeds.  Each corner patch is named after
    //    the selected edges meeting there (`{fid}:BLEND:CORNER:{e_a}+…`) so no
    //    two corners collide and the patch stays under the `{fid}:BLEND` prefix.
    if !chamfer {
        for (ci, corner) in corners.iter().enumerate() {
            let corner_name = corner_face_name(edge_names, name, &corner_edges[ci]);
            if let Ok(rounded) =
                crate::blend::round_convex_corner(&result, *corner, radius, corner_name.as_deref())
            {
                result = rounded;
            }
        }
    }

    // Heal any residual vertex/edge gaps introduced by the corner-rounding
    // surgery (the per-edge results are already healed inside fillet_or_chamfer).
    heal_edge_vertex_gaps(&mut result, radius)?;
    Ok(result)
}

/// Variable-radius fillet/chamfer of a GROUP of edges (§4.9.5), the app entry
/// for tapered blends: each selected edge (resolved by a point on it) is
/// blended with the SAME radius profile `radii` — a list of (edge-fraction,
/// radius) stops in [0,1] — applied along that edge's own parameterization.
/// Edges are blended independently (no shared-vertex corner rounding; a
/// variable-radius star has no single tangent ball), so this is the tapered
/// counterpart of `fillet_edges` for the constant case.
pub fn fillet_edges_variable(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    radii: &[(f64, f64)],
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if edge_points.is_empty() {
        return Err("fillet_edges_variable: no edges selected".into());
    }
    let per_edge_stops: Vec<Vec<(f64, f64)>> = vec![radii.to_vec(); edge_points.len()];
    fillet_edges_variable_impl(
        solid,
        edge_points,
        edge_names,
        &per_edge_stops,
        chamfer,
        name,
        "fillet_edges_variable",
        true,
    )
}

/// The shared variable-radius group core: each selected edge `i` is blended
/// with ITS OWN stop list `per_edge_stops[i]` (the legacy entry replicates one
/// list; the law entries sample a chain-abscissa [`crate::law::RadiusLaw`] per
/// edge).  `allow_tapered_sequential` keeps the legacy entry's sequential
/// fallback for tapered chains (whose end state is the honest
/// "mismatched radii" validation gate); the law entries pass `false` because
/// their stop fractions are computed against the ORIGINAL edges — the
/// sequential build re-resolves edges on the evolving solid whose shared
/// corners are already TRIMMED by earlier blends, which would silently distort
/// the law's abscissa mapping (constant stops are immune, so they may still
/// fall back).
#[allow(clippy::too_many_arguments)]
fn fillet_edges_variable_impl(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    per_edge_stops: &[Vec<(f64, f64)>],
    chamfer: bool,
    name: Option<&str>,
    entry: &str,
    allow_tapered_sequential: bool,
) -> Result<BrepSolid, String> {
    debug_assert_eq!(per_edge_stops.len(), edge_points.len());
    // A law that is one constant everywhere IS the constant-radius fillet:
    // hand it to the constant group, whose corners are constructed (the
    // stripe network) rather than composed by boolean.
    if let Some(constant) = per_edge_stops
        .first()
        .and_then(|stops| stops.first())
        .map(|(_, radius)| *radius)
    {
        let uniform = per_edge_stops.iter().all(|stops| {
            stops
                .iter()
                .all(|(_, radius)| (radius - constant).abs() <= 1e-12 * (1.0 + constant.abs()))
        });
        if uniform && constant > 0.0 {
            return fillet_edges(solid, edge_points, edge_names, constant, chamfer, name);
        }
    }
    let max_radius = per_edge_stops
        .iter()
        .flat_map(|stops| stops.iter())
        .map(|(_, r)| r.abs())
        .fold(0.0_f64, f64::max);

    // Chain corners miter exactly like the constant-radius group (§6.9.6):
    // per-edge blends on the ORIGINAL solid composed by boolean — Intersect
    // when every edge is convex, Union when every edge is concave.  Mixed or
    // unclassifiable selections keep the sequential build.  Variable blends
    // never round star vertices, so unlike the constant group there is no
    // sequential-only star path to protect.
    let mut chain_corner = false;
    {
        use rustc_hash::FxHashSet as HashSet;
        let mut endpoints: Vec<(Vec3, usize)> = Vec::with_capacity(edge_points.len() * 2);
        for (i, point) in edge_points.iter().enumerate() {
            if let Ok(edge_id) = resolve_edge_by_point(solid, *point) {
                if let Some(edge) = solid.edges.iter().find(|e| e.id == edge_id) {
                    if let (Ok(a), Ok(b)) =
                        (edge.curve.evaluate(edge.t0), edge.curve.evaluate(edge.t1))
                    {
                        endpoints.push((a, i));
                        endpoints.push((b, i));
                    }
                }
            }
        }
        let mut used = vec![false; endpoints.len()];
        for i in 0..endpoints.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut edges_here: HashSet<usize> = HashSet::default();
            edges_here.insert(endpoints[i].1);
            for j in (i + 1)..endpoints.len() {
                if used[j] {
                    continue;
                }
                if endpoints[i].0.sub(endpoints[j].0).length() < 1e-6 {
                    used[j] = true;
                    edges_here.insert(endpoints[j].1);
                }
            }
            if edges_here.len() == 2 {
                chain_corner = true;
            }
        }
    }
    let miter_operation = if chain_corner {
        let probe_radius = if max_radius > 0.0 { max_radius } else { 1.0 };
        let mut any_convex = false;
        let mut any_concave = false;
        for point in edge_points {
            match resolve_edge_by_point(solid, *point)
                .and_then(|edge_id| analyze_edge(solid, edge_id, probe_radius))
            {
                Ok(cross) if cross.convex => any_convex = true,
                Ok(_) => any_concave = true,
                Err(_) => {
                    any_convex = true;
                    any_concave = true;
                    break;
                }
            }
        }
        match (any_convex, any_concave) {
            (true, false) => Some(crate::BooleanOperation::Intersect),
            (false, true) => Some(crate::BooleanOperation::Union),
            _ => None,
        }
    } else {
        None
    };

    // Try the miter first; the variable blend's FITTED boundary curves are
    // only ~1e-3 accurate at blend-blend tangencies (unlike the exact
    // constant-radius cylinders), so the composition can fail — fall back to
    // the sequential build then, which is never worse than the pre-miter
    // behavior.  Tightening the taper surface's endpoint fitting is the
    // documented follow-up that would make the miter stick.
    let miter_attempt: Option<BrepSolid> = if let Some(operation) = miter_operation {
        let options = crate::BooleanOptions::default();
        let mut combined: Option<BrepSolid> = None;
        let mut failed = false;
        for (index, point) in edge_points.iter().enumerate() {
            let Ok(edge_id) = resolve_edge_by_point(solid, *point) else {
                failed = true;
                break;
            };
            let edge_name = per_edge_name(edge_names, name, index);
            let Ok(blended) = crate::blend::blend_edge_variable(
                solid,
                edge_id,
                &per_edge_stops[index],
                chamfer,
                edge_name,
            ) else {
                failed = true;
                break;
            };
            let next = match combined.take() {
                None => blended,
                Some(previous) => {
                    match crate::boolean_operation(&previous, &blended, operation, &options) {
                        Ok(next) => next,
                        Err(_) => {
                            failed = true;
                            break;
                        }
                    }
                }
            };
            combined = Some(next);
        }
        if failed {
            None
        } else {
            combined.filter(|s| s.validate().is_empty())
        }
    } else {
        None
    };
    let mut result = match miter_attempt {
        Some(mitered) => mitered,
        None => {
            // Any stop list that is NOT radius-uniform (the same 1e-12
            // relative criterion `blend_edge_variable` uses for its exact
            // constant-radius degeneration) makes the sequential build
            // abscissa-distorting on trimmed chain edges; law entries refuse
            // instead of silently shifting the law.
            let tapered = per_edge_stops.iter().any(|stops| {
                stops.first().is_some_and(|(_, first)| {
                    stops
                        .iter()
                        .any(|(_, r)| (r - first).abs() > 1e-12 * (1.0 + first.abs()))
                })
            });
            if chain_corner && tapered && !allow_tapered_sequential {
                return Err(format!(
                    "{entry}: the tapered blends across the selected chain's shared corners \
                     did not compose to a valid solid (the fitted blend boundaries could not \
                     be mitered); fillet fewer edges per operation or reduce the taper"
                ));
            }
            let mut sequential = solid.clone();
            for (index, point) in edge_points.iter().enumerate() {
                let edge_id = resolve_edge_by_point(&sequential, *point)?;
                let edge_name = per_edge_name(edge_names, name, index);
                sequential = crate::blend::blend_edge_variable(
                    &sequential,
                    edge_id,
                    &per_edge_stops[index],
                    chamfer,
                    edge_name,
                )?;
            }
            sequential
        }
    };
    // Heal §6.9 surgery so re-trimmed edges meet their vertices exactly; scale
    // the heal bound by the largest radius stop in the taper profile.
    heal_edge_vertex_gaps(&mut result, max_radius)?;
    // Final honesty gate: a genuinely TAPERED chain (different radii at the
    // shared corner) has mismatched trim stations there — the blends cannot
    // meet without a transition patch (not implemented), and the sequential
    // surgery silently left broken topology before this gate existed.
    let issues = result.validate();
    if !issues.is_empty() {
        let detail = if allow_tapered_sequential {
            "tapered blends meet at a shared chain vertex with \
             mismatched radii — the radius-transition corner patch is not implemented; \
             fillet the edges in separate operations or use matching stop radii"
        } else {
            "the composed radius-law blend produced invalid topology — the blends \
             across a shared chain corner failed to reassemble"
        };
        return Err(format!(
            "{entry}: {detail} ({} validation issues, first: {})",
            issues.len(),
            issues
                .first()
                .map(|issue| issue.message.clone())
                .unwrap_or_default()
        ));
    }
    Ok(result)
}

/// One selected edge placed on the ordered chain, with its arc-length
/// parameterization (the caller-side mapping from chain abscissa to the
/// `blend_edge_variable` per-edge parameter-fraction stop seam).
struct ChainLink {
    /// Position of this edge in the caller's `edge_points` selection.
    input_index: usize,
    /// True when the edge's own t0→t1 parameter direction runs WITH the chain.
    forward: bool,
    /// Cumulative chord-length table from the edge's t0 end:
    /// `(parameter fraction, arc length)`, uniformly spaced in fraction.
    arc: Vec<(f64, f64)>,
    /// Total arc length of the edge.
    length: f64,
    /// Chain abscissa at the link's ENTRY vertex (the end reached first when
    /// walking the chain from its start).
    abscissa: f64,
}

/// Cumulative chord-length table of one edge, `(parameter fraction, arc
/// length)` at uniform fractions.  The sample count doubles until two
/// successive total-length estimates agree within `tol` (chord length
/// converges O(N⁻²) for smooth curves, so the agreement of the N and 2N
/// estimates bounds the remaining error at the same order); straight edges
/// converge on the first doubling.  The 16-sample start resolves any
/// single-span arc of up to half a turn to sub-percent before refinement; the
/// 4096 cap (8 doublings) guards adversarial curves — beyond it the table is
/// two orders denser than the blend march's station grid, so finer chords
/// cannot move any station's sampled radius meaningfully.
fn edge_arc_table(
    curve: &NurbsCurve,
    t0: f64,
    t1: f64,
    tol: f64,
) -> Result<(Vec<(f64, f64)>, f64), String> {
    let build = |n: usize| -> Result<(Vec<(f64, f64)>, f64), String> {
        let mut table = Vec::with_capacity(n + 1);
        let mut cumulative = 0.0;
        let mut previous = curve.evaluate(t0)?;
        table.push((0.0, 0.0));
        for j in 1..=n {
            let fraction = j as f64 / n as f64;
            let point = curve.evaluate(t0 + (t1 - t0) * fraction)?;
            cumulative += point.sub(previous).length();
            previous = point;
            table.push((fraction, cumulative));
        }
        Ok((table, cumulative))
    };
    let mut n = 16usize;
    let (mut table, mut length) = build(n)?;
    while n < 4096 {
        n *= 2;
        let (next_table, next_length) = build(n)?;
        let converged = (next_length - length).abs() <= tol;
        table = next_table;
        length = next_length;
        if converged {
            break;
        }
    }
    Ok((table, length))
}

/// Arc length from the edge's t0 end at `fraction` of its parameter span,
/// linearly interpolated in the uniform chord table.
fn arc_length_at_fraction(table: &[(f64, f64)], fraction: f64) -> f64 {
    let fraction = fraction.clamp(0.0, 1.0);
    let intervals = table.len() - 1;
    let scaled = fraction * intervals as f64;
    let index = (scaled.floor() as usize).min(intervals - 1);
    let local = scaled - index as f64;
    let (_, a) = table[index];
    let (_, b) = table[index + 1];
    a + (b - a) * local
}

/// Resolve the selected edges and order them into ONE OPEN CHAIN with
/// cumulative arc-length abscissas.  The chain starts at the free endpoint
/// belonging to the EARLIEST-selected end edge (so users get the natural
/// "first pick carries the law start" orientation); a single selected edge is
/// its own chain oriented t0→t1 (which also admits a closed edge — the law's
/// end radii must then match, enforced downstream by `blend_edge_variable`).
/// Branching (a vertex shared by 3+ selected edges), closed rings of several
/// edges, and disconnected selections refuse with named errors.
fn resolve_selected_chain(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    entry: &str,
    tol: f64,
) -> Result<Vec<ChainLink>, String> {
    // Endpoint identity uses the kernel-wide COINCIDENCE_DISTANCE_FLOOR — the
    // same band the constant-radius group's corner detector applies.
    let band = crate::tolerance::COINCIDENCE_DISTANCE_FLOOR;

    struct Resolved {
        edge_id: u64,
        start: Vec3,
        end: Vec3,
        arc: Vec<(f64, f64)>,
        length: f64,
    }
    let mut resolved: Vec<Resolved> = Vec::with_capacity(edge_points.len());
    for point in edge_points {
        let edge_id = resolve_edge_by_point(solid, *point)?;
        if resolved.iter().any(|r| r.edge_id == edge_id) {
            return Err(format!("{entry}: the same edge was selected more than once"));
        }
        let edge = solid
            .edges
            .iter()
            .find(|e| e.id == edge_id)
            .ok_or_else(|| format!("{entry}: resolved edge vanished"))?;
        let (arc, length) = edge_arc_table(&edge.curve, edge.t0, edge.t1, tol)?;
        if !(length > 0.0) {
            return Err(format!("{entry}: selected edge has zero length"));
        }
        resolved.push(Resolved {
            edge_id,
            start: edge.curve.evaluate(edge.t0)?,
            end: edge.curve.evaluate(edge.t1)?,
            arc,
            length,
        });
    }
    let n = resolved.len();
    if n == 1 {
        let only = resolved.remove(0);
        return Ok(vec![ChainLink {
            input_index: 0,
            forward: true,
            arc: only.arc,
            length: only.length,
            abscissa: 0.0,
        }]);
    }

    // Cluster the 2n endpoints; each entry is (edge index, is_start_end).
    let mut clusters: Vec<(Vec3, Vec<(usize, bool)>)> = Vec::new();
    for (i, r) in resolved.iter().enumerate() {
        for (point, is_start) in [(r.start, true), (r.end, false)] {
            match clusters
                .iter_mut()
                .find(|(anchor, _)| anchor.sub(point).length() < band)
            {
                Some((_, members)) => members.push((i, is_start)),
                None => clusters.push((point, vec![(i, is_start)])),
            }
        }
    }
    if clusters.iter().any(|(_, members)| members.len() > 2) {
        return Err(format!(
            "{entry}: selected edges must form one open chain \
             (a vertex is shared by three or more selected edges)"
        ));
    }
    let free: Vec<usize> = clusters
        .iter()
        .enumerate()
        .filter(|(_, (_, members))| members.len() == 1)
        .map(|(c, _)| c)
        .collect();
    if free.len() != 2 {
        return Err(format!(
            "{entry}: selected edges must form one OPEN chain \
             (closed rings and disconnected selections are not supported)"
        ));
    }
    // Start at the free end whose edge appears EARLIEST in the selection.
    let start_cluster = *free
        .iter()
        .min_by_key(|&&c| clusters[c].1[0].0)
        .expect("two free ends");

    // Walk the chain.
    let mut links: Vec<ChainLink> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut abscissa = 0.0_f64;
    let mut cluster = start_cluster;
    for _ in 0..n {
        let Some(&(edge_index, entered_at_start)) = clusters[cluster]
            .1
            .iter()
            .find(|(edge_index, _)| !visited[*edge_index])
        else {
            return Err(format!(
                "{entry}: selected edges are not connected into one chain"
            ));
        };
        visited[edge_index] = true;
        let r = &resolved[edge_index];
        links.push(ChainLink {
            input_index: edge_index,
            forward: entered_at_start,
            arc: r.arc.clone(),
            length: r.length,
            abscissa,
        });
        abscissa += r.length;
        let exit_point = if entered_at_start { r.end } else { r.start };
        cluster = clusters
            .iter()
            .position(|(anchor, _)| anchor.sub(exit_point).length() < band)
            .ok_or_else(|| format!("{entry}: chain walk lost an endpoint cluster"))?;
    }
    if visited.iter().any(|v| !v) {
        return Err(format!(
            "{entry}: selected edges are not connected into one chain"
        ));
    }
    Ok(links)
}

/// Sample the law into one edge's `(parameter fraction, radius)` stop list —
/// the caller-side bridge from chain abscissa into the `radius_at` closure
/// seam that `blend_edge_variable` builds over its stops.
///
/// `scale` maps chain abscissa into the law's own abscissa units
/// (`law.total_length() / chain_length` — proportional, so a law built from
/// the measured chain lengths maps 1:1).  The base stop count derives from
/// the law's curvature: piecewise-linear sampling of a C1, piecewise-C2
/// function over step `h` errs at most `h²·max|r''|/8`, so
/// `h = sqrt(8·tol/max|r''|)` holds the sampling error under the SSI fit
/// tolerance the variable lane is built to.  That bound is exact for
/// arc-length-linear (straight) edges; curved edges bend the
/// parameter→abscissa map, so each interval is additionally midpoint-checked
/// against the law and bisected on violation (up to 8 halvings — a 4⁸ ≈ 6·10⁴
/// error reduction, decisive for any C1 law).  The base count is capped at
/// 256 intervals: 4× the blend march's 64-station density
/// (blend/stations.rs), beyond which denser stops cannot move any station's
/// sampled radius by more than the fit tolerance.
fn law_stops_for_link(
    link: &ChainLink,
    law: &crate::law::RadiusLaw,
    scale: f64,
    tol: f64,
) -> Vec<(f64, f64)> {
    let radius_at_fraction = |fraction: f64| -> f64 {
        let arc = arc_length_at_fraction(&link.arc, fraction);
        let chain_s = if link.forward {
            link.abscissa + arc
        } else {
            link.abscissa + (link.length - arc)
        };
        law.radius_at(chain_s * scale)
    };
    // Curvature of the law in edge-fraction units: d²r/df² ≤
    // max|r''|·(scale·length)² for the arc-length-linear map.
    let curvature = law
        .max_second_derivative(link.abscissa * scale, (link.abscissa + link.length) * scale)
        * (scale * link.length).powi(2);
    let base = if curvature * 0.125 <= tol {
        1usize
    } else {
        ((curvature / (8.0 * tol)).sqrt().ceil() as usize).clamp(1, 256)
    };
    let mut stops: Vec<(f64, f64)> = (0..=base)
        .map(|j| {
            let fraction = j as f64 / base as f64;
            (fraction, radius_at_fraction(fraction))
        })
        .collect();
    // Midpoint refinement for curved parameter→abscissa maps.
    let mut depth = 0usize;
    while depth < 8 {
        let mut refined: Vec<(f64, f64)> = Vec::with_capacity(stops.len());
        let mut inserted = false;
        for pair in stops.windows(2) {
            refined.push(pair[0]);
            let mid_fraction = 0.5 * (pair[0].0 + pair[1].0);
            let law_mid = radius_at_fraction(mid_fraction);
            let linear_mid = 0.5 * (pair[0].1 + pair[1].1);
            if (law_mid - linear_mid).abs() > tol {
                refined.push((mid_fraction, law_mid));
                inserted = true;
            }
        }
        refined.push(*stops.last().expect("at least two stops"));
        stops = refined;
        if !inserted {
            break;
        }
        depth += 1;
    }
    stops
}

/// Fillet (or chamfer) a chain of edges under a composable radius law
/// evaluated on the chain's cumulative arc-length abscissa (the OCCT
/// `Law_Composite` model; see [`crate::law::RadiusLaw`]).  The selected edges
/// must form ONE OPEN CHAIN (or be a single edge); the chain starts at the
/// free end of the earliest-selected end edge, and the law's abscissa maps
/// proportionally onto the chain's measured arc length (a law built with the
/// chain's own lengths — e.g. [`crate::law::RadiusLaw::from_vertex_radii`] —
/// maps 1:1).  Endpoint radii are met exactly; radii at shared chain vertices
/// match by the law's continuity, so the per-edge blends miter through the
/// corners; a chain whose blends cannot be mitered REFUSES rather than
/// distorting the law through the sequential rebuild.
pub fn fillet_edges_variable_law(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    law: &crate::law::RadiusLaw,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    const ENTRY: &str = "fillet_edges_variable_law";
    if edge_points.is_empty() {
        return Err(format!("{ENTRY}: no edges selected"));
    }
    let tolerances = crate::KernelTolerances::for_solid(solid, 1e-7);
    let chain = resolve_selected_chain(solid, edge_points, ENTRY, tolerances.intersection_fit)?;
    fillet_variable_law_on_chain(
        solid,
        edge_points,
        edge_names,
        &chain,
        law,
        chamfer,
        name,
        ENTRY,
        tolerances.intersection_fit,
    )
}

/// The natural per-vertex user model: radius `vertex_radii[i]` at chain
/// vertex `i` (in CHAIN order, starting at the free end of the
/// earliest-selected edge), smoothly interpolated along the chain
/// (monotone C1 — every vertex radius met exactly, no overshoot).  Requires
/// exactly one radius per chain vertex (`edges + 1`).
pub fn fillet_edges_variable_vertex_radii(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    vertex_radii: &[f64],
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    const ENTRY: &str = "fillet_edges_variable_vertex_radii";
    if edge_points.is_empty() {
        return Err(format!("{ENTRY}: no edges selected"));
    }
    let tolerances = crate::KernelTolerances::for_solid(solid, 1e-7);
    let chain = resolve_selected_chain(solid, edge_points, ENTRY, tolerances.intersection_fit)?;
    let lengths: Vec<f64> = chain.iter().map(|link| link.length).collect();
    let law = crate::law::RadiusLaw::from_vertex_radii(&lengths, vertex_radii)
        .map_err(|error| format!("{ENTRY}: {error}"))?;
    fillet_variable_law_on_chain(
        solid,
        edge_points,
        edge_names,
        &chain,
        &law,
        chamfer,
        name,
        ENTRY,
        tolerances.intersection_fit,
    )
}

/// Shared law-entry tail: sample per-edge stop lists from the law over the
/// resolved chain and run the variable group core (miter-or-refuse: no
/// sequential fallback for tapered chains — see `fillet_edges_variable_impl`).
#[allow(clippy::too_many_arguments)]
fn fillet_variable_law_on_chain(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    chain: &[ChainLink],
    law: &crate::law::RadiusLaw,
    chamfer: bool,
    name: Option<&str>,
    entry: &str,
    tol: f64,
) -> Result<BrepSolid, String> {
    let chain_length: f64 = chain.iter().map(|link| link.length).sum();
    let scale = law.total_length() / chain_length;
    let mut per_edge_stops: Vec<Vec<(f64, f64)>> = vec![Vec::new(); edge_points.len()];
    for link in chain {
        per_edge_stops[link.input_index] = law_stops_for_link(link, law, scale, tol);
    }
    fillet_edges_variable_impl(
        solid,
        edge_points,
        edge_names,
        &per_edge_stops,
        chamfer,
        name,
        entry,
        false,
    )
}

/// Asymmetric (two-distance) chamfer of a GROUP of edges, the app entry: each
/// selected edge (resolved by a point on it) gets a `d1 × d2` bevel (§6.11).
/// Edges are chamfered independently — asymmetric chamfers keep sharp vertices,
/// so there is no shared-corner blending.
pub fn chamfer_edges_asymmetric(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    d1: f64,
    d2: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if edge_points.is_empty() {
        return Err("chamfer_edges_asymmetric: no edges selected".into());
    }
    let mut result = solid.clone();
    // `edge_names[index]` is keyed by INPUT position; the loop resolves each
    // point on the EVOLVING `result`, but enumerates the input points in order,
    // so the index alignment holds.
    for (index, point) in edge_points.iter().enumerate() {
        let edge_id = resolve_edge_by_point(&result, *point)?;
        let edge_name = per_edge_name(edge_names, name, index);
        result = chamfer_edge_asymmetric(&result, edge_id, d1, d2, edge_name)?;
    }
    Ok(result)
}

/// Distance-angle chamfer of a GROUP of edges, the app entry: each selected
/// edge (resolved by a point on it) gets a setback `d1` on face 1 and a chamfer
/// face at `angle_rad` from face 1 (§6.11); `d2` is constructed per edge.
pub fn chamfer_edges_angle(
    solid: &BrepSolid,
    edge_points: &[Vec3],
    edge_names: Option<&[String]>,
    d1: f64,
    angle_rad: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if edge_points.is_empty() {
        return Err("chamfer_edges_angle: no edges selected".into());
    }
    let mut result = solid.clone();
    for (index, point) in edge_points.iter().enumerate() {
        let edge_id = resolve_edge_by_point(&result, *point)?;
        let edge_name = per_edge_name(edge_names, name, index);
        result = chamfer_edge_angle(&result, edge_id, d1, angle_rad, edge_name)?;
    }
    Ok(result)
}

/// Debug aid (`BREP_DEBUG_NETWORK`): every face loop as its coedge walk, with
/// each edge's stored vertices, so a loop the surgery left open can be read
/// against the input it was built from.
fn dump_loops_debug(label: &str, solid: &BrepSolid) {
    for shell in &solid.shells {
        for face in &shell.faces {
            for (index, loop_record) in face.loops.iter().enumerate() {
                let walk: Vec<String> = loop_record
                    .coedges
                    .iter()
                    .map(|coedge| {
                        match solid.edges.iter().find(|e| e.id == coedge.edge_id) {
                            Some(edge) => format!(
                                "c{}:e{}{}({}->{})",
                                coedge.id,
                                edge.id,
                                if coedge.forward { "+" } else { "-" },
                                edge.start_vertex_id,
                                edge.end_vertex_id
                            ),
                            None => format!("c{}:e{}?MISSING", coedge.id, coedge.edge_id),
                        }
                    })
                    .collect();
                eprintln!(
                    "{label} face {} ({:?}) loop {index}: {}",
                    face.id,
                    face.name,
                    walk.join(" ")
                );
            }
        }
    }
}
