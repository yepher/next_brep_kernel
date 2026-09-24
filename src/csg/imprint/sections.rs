use crate::{KernelRefusal, KernelStage, OrRefuse};
use super::*;

/// A marched intersection branch that runs along existing boundary edges of
/// BOTH faces cannot bound any new fragment — the boundary is already there.
/// Worse, the fitted polyline wiggles within tolerance of the true curve and
/// its crossings spray micro edge-splits over the very edges it follows
/// (glue unions march the touching pair in one operand order and produced
/// thousands of noise splits). Skip such branches entirely.
pub(super) fn branch_follows_shared_boundary(
    points: &[Vec3],
    first: TaggedFace<'_>,
    second: TaggedFace<'_>,
    edges: &HashMap<(u8, u64), &EdgeRecord>,
    tolerance: f64,
    scale: f64,
) -> Result<bool, KernelRefusal> {
    let Some(last) = points.last() else {
        return Ok(false);
    };
    let weld_limit = assembler_weld(tolerance);
    let step = (points.len() / 16).max(1);
    for face in [first, second] {
        let face_edges = face_edges(face, edges)?;
        for point in points.iter().step_by(step).chain(std::iter::once(last)) {
            // A tangential contact curve is only determinable to the
            // square-root of the tolerance times the local size (the
            // intersection is ill-conditioned there), so the marched points
            // legitimately wander that far from the exact shared boundary.
            // Widen the proximity band accordingly where the two surface
            // normals are (anti-)parallel.
            let first_projection = project_point_to_surface(&first.face.surface, *point).or_refuse(KernelStage::Intersect, "project_point_to_surface")?;
            let second_projection = project_point_to_surface(&second.face.surface, *point).or_refuse(KernelStage::Intersect, "project_point_to_surface")?;
            // At a surface pole — e.g. the apex a partial revolve leaves where a
            // profile edge touches the axis — the normal is undefined (the patch
            // collapses to a point, so the derivative cross product is zero).
            // The normals here only decide whether to WIDEN the proximity band
            // for an ill-conditioned tangential contact; a pole is exactly such an
            // ill-conditioned spot. Treat a missing normal as tangential (widen)
            // instead of propagating the zero-length error and aborting the whole
            // boolean.
            let first_normal = first
                .face
                .surface
                .normal(first_projection.u, first_projection.v)
                .ok();
            let second_normal = second
                .face
                .surface
                .normal(second_projection.u, second_projection.v)
                .ok();
            let tangential = match (first_normal, second_normal) {
                (Some(a), Some(b)) => a.cross(b).length() <= 1e-2,
                _ => true,
            };
            let limit = if tangential {
                (tolerance.max(1e-7) * merge_scale(scale))
                    .sqrt()
                    .max(weld_limit)
            } else {
                weld_limit
            };
            let mut near = false;
            let mut best = f64::INFINITY;
            for edge in &face_edges {
                if edge.degenerate {
                    continue;
                }
                let projection = project_point_to_curve(&edge.curve, *point).or_refuse(KernelStage::Intersect, "project_point_to_curve")?;
                let low = edge.t0.min(edge.t1);
                let high = edge.t0.max(edge.t1);
                let on = edge.curve.evaluate(projection.u.clamp(low, high)).or_refuse(KernelStage::Intersect, "evaluate")?;
                let distance = on.sub(*point).length();
                if distance < best {
                    best = distance;
                }
                if distance <= limit {
                    near = true;
                    break;
                }
            }
            if !near {
                if std::env::var("BREP_DEBUG_BRANCH").is_ok() {
                    eprintln!(
                        "branch reject: face {} best={:.3e} limit={:.3e} point=({:.6},{:.6},{:.6})",
                        face.face.id, best, limit, point.x, point.y, point.z
                    );
                }
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Keep only the runs of a marched branch that lie inside (or on the
/// boundary of) BOTH faces' trimmed regions. The marcher works on the
/// untrimmed carriers and happily chases intersections far outside the
/// actual faces (an extruded wall's carrier spans the full tool travel);
/// those runaway sections then spray noise splits over real edges during
/// imprint processing. Points are tested on a subsample grid with one-step
/// margin so genuine trim crossings keep a little slack for the exact
/// piece-level clipping that follows.
/// SEED-REFINED CLIP (multi-interval section rescue). `clip_branch_to_trims`
/// classifies only the branch's own SAMPLE points, so a trim interval SHORTER
/// than the march step is invisible: a section line crossing an annular face
/// twice (solid 00000327's serrated rings vs a box wall) had its ~1.9e-3 exit
/// interval fall entirely between two ~3.5e-3-apart samples — the clip saw no
/// inside point, dropped the interval, the strip never split, and the whole
/// outline chain stranded one-use (non-integral genus, `did not converge`
/// churn downstream). The boundary-pierce SEEDS already computed for the
/// marcher are the EXACT crossing points of the section with the faces' trim
/// boundaries — the interval endpoints. Inserting every seed that lies ON the
/// branch polyline into it (ordered along its segment) makes each interval
/// carry its own endpoints, so the point classification cannot miss it.
/// Purely additive: points are only ADDED, every previously-kept run is still
/// kept. The insertion gate (`max(tolerance*10, 0.1*segment)`) admits the
/// zero-sag analytic cases and the small sag of curved marches while
/// rejecting seeds of OTHER nearby branches. Escape hatch
/// `BREP_SEED_CLIP_REFINE=0`.
pub(super) fn insert_seed_points_into_branch(points: &[Vec3], seeds: &[Vec3], tolerance: f64) -> Vec<Vec3> {
    if points.len() < 2
        || seeds.is_empty()
        || std::env::var("BREP_SEED_CLIP_REFINE").as_deref() == Ok("0")
    {
        return points.to_vec();
    }
    // Per segment, the seeds that project onto it, ordered by fraction.
    let mut insertions: Vec<Vec<(f64, Vec3)>> = vec![Vec::new(); points.len() - 1];
    for seed in seeds {
        let mut best: Option<(usize, f64, f64)> = None; // (segment, distance, fraction)
        for (index, pair) in points.windows(2).enumerate() {
            let direction = pair[1].sub(pair[0]);
            let length_squared = direction.dot(direction);
            if length_squared <= 0.0 {
                continue;
            }
            let fraction = (seed.sub(pair[0]).dot(direction) / length_squared).clamp(0.0, 1.0);
            let distance = seed.sub(pair[0].add(direction.scale(fraction))).length();
            if best.is_none_or(|(_, d, _)| distance < d) {
                best = Some((index, distance, fraction));
            }
        }
        let Some((index, distance, fraction)) = best else {
            continue;
        };
        let segment_length = points[index + 1].sub(points[index]).length();
        if distance > (tolerance * 10.0).max(segment_length * 0.1) {
            continue; // not on this branch
        }
        // An interior insertion only — a seed at an existing sample adds nothing.
        let near_end = fraction * segment_length <= tolerance
            || (1.0 - fraction) * segment_length <= tolerance;
        if near_end {
            continue;
        }
        insertions[index].push((fraction, *seed));
    }
    if insertions.iter().all(Vec::is_empty) {
        return points.to_vec();
    }
    let mut refined = Vec::with_capacity(points.len() + seeds.len());
    for (index, point) in points.iter().enumerate() {
        refined.push(*point);
        if index < insertions.len() {
            let segment = &mut insertions[index];
            segment.sort_by(|a, b| a.0.total_cmp(&b.0));
            refined.extend(segment.iter().map(|(_, seed)| *seed));
        }
    }
    refined
}

pub(super) fn clip_branch_to_trims(
    points: &[Vec3],
    first: TaggedFace<'_>,
    second: TaggedFace<'_>,
) -> Result<Vec<Vec<Vec3>>, KernelRefusal> {
    let count = points.len();
    if count < 2 {
        return Ok(Vec::new());
    }
    let debug_clip = std::env::var("BREP_DEBUG_CLIP").is_ok_and(|value| {
        value == "all"
            || value == first.face.id.to_string()
            || value == second.face.id.to_string()
    });
    let inside_both = |point: Vec3| -> Result<bool, KernelRefusal> {
        for face in [first, second] {
            let projection = project_point_to_surface(&face.face.surface, point).or_refuse(KernelStage::Intersect, "project_point_to_surface")?;
            let status = parameter_point_in_face(
                face.face,
                Vec2 {
                    x: projection.u,
                    y: projection.v,
                },
                1e-6,
            ).or_refuse(KernelStage::Intersect, "csg.imprint.sections")?;
            if debug_clip {
                eprintln!(
                    "clip {}x{}: p=({:.5},{:.5},{:.5}) on face {} uv=({:.5},{:.5}) dist={:.2e} {:?}",
                    first.face.id,
                    second.face.id,
                    point.x,
                    point.y,
                    point.z,
                    face.face.id,
                    projection.u,
                    projection.v,
                    projection.distance,
                    status
                );
            }
            if status == PolygonClass::Outside {
                return Ok(false);
            }
        }
        Ok(true)
    };
    let step = (count / 256).max(1);
    let mut keep = vec![false; count];
    let mut index = 0;
    loop {
        let sample = index.min(count - 1);
        if inside_both(points[sample])? {
            let low = sample.saturating_sub(step);
            let high = (sample + step).min(count - 1);
            for flag in &mut keep[low..=high] {
                *flag = true;
            }
        }
        if sample == count - 1 {
            break;
        }
        index += step;
    }
    let mut runs = Vec::new();
    let mut current: Vec<Vec3> = Vec::new();
    for (point, kept) in points.iter().zip(&keep) {
        if *kept {
            current.push(*point);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    Ok(runs)
}

/// True when at least three of five interior samples of `curve` lie on the
/// trimmed span of `edge` — a (near-)coincident overlap rather than a
/// transversal crossing.
pub(super) fn curve_overlaps_edge(curve: &NurbsCurve, edge: &EdgeRecord, limit: f64) -> Result<bool, KernelRefusal> {
    let [c0, c1] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let low = edge.t0.min(edge.t1);
    let high = edge.t0.max(edge.t1);
    let mut on = 0;
    for fraction in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let point = curve.evaluate(c0 + (c1 - c0) * fraction).or_refuse(KernelStage::Intersect, "evaluate")?;
        let projection = project_point_to_curve(&edge.curve, point).or_refuse(KernelStage::Intersect, "project_point_to_curve")?;
        let nearest = edge.curve.evaluate(projection.u.clamp(low, high)).or_refuse(KernelStage::Intersect, "evaluate")?;
        if nearest.sub(point).length() <= limit {
            on += 1;
        }
    }
    Ok(on >= 3)
}

/// Worst distance from any sample of `a` to the curve `b` (a one-sided Hausdorff
/// distance, sampled). Used bidirectionally so the B2 span-coincidence test
/// rejects partial overlaps and same-endpoint-but-different-path curves.
pub(super) fn max_curve_deviation(a: &NurbsCurve, b: &NurbsCurve) -> Result<f64, KernelRefusal> {
    let [d0, d1] = a.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let mut worst = 0.0f64;
    for k in 0..=32 {
        let t = d0 + (d1 - d0) * (k as f64 / 32.0);
        let point = a.evaluate(t).or_refuse(KernelStage::Intersect, "evaluate")?;
        worst = worst.max(project_point_to_curve(b, point).or_refuse(KernelStage::Intersect, "project_point_to_curve")?.distance);
    }
    Ok(worst)
}

/// The sub-curve of `curve` over the parameter window `[lo, hi]`, by exact
/// knot-insertion splitting (mirrors `boolean::curve_subrange` but by absolute
/// parameter rather than fraction). `lo`/`hi` are clamped away from the domain
/// ends by the split guard, so a window flush with an end keeps that end.
fn curve_param_subrange(curve: &NurbsCurve, lo: f64, hi: f64) -> Result<NurbsCurve, KernelRefusal> {
    let [d0, d1] = curve.domain().or_refuse(KernelStage::Intersect, "domain")?;
    let epsilon = ((d1 - d0).abs() * 1e-9).max(2e-9);
    let mut result = curve.clone();
    if lo > d0 + epsilon && lo < d1 - epsilon {
        result = result.split(lo).or_refuse(KernelStage::Intersect, "split")?.1;
    }
    let domain = result.domain().or_refuse(KernelStage::Intersect, "domain")?;
    if hi < domain[1] - epsilon && hi > domain[0] + epsilon {
        result = result.split(hi).or_refuse(KernelStage::Intersect, "split")?.0;
    }
    Ok(result)
}

/// B2 (OCCT `BOPDS_CommonBlock` / `IsExistingPaveBlock`): a SSI section piece
/// whose supporting curve coincides ALONG ITS WHOLE SPAN with a SUB-ARC of an
/// EXISTING boundary edge of one of its support faces is not a new 1-cell — it
/// IS that boundary. `apply_edge_splits` will mint that sub-arc as its own edge
/// (splitting the parent at the section's endpoints), which the OWNING face
/// then carries as a boundary. So instead of the section drifting a graze-gap
/// away from that sub-arc (the marched/analytic chord vs the vendor arc — the
/// one-use pair), we REPLACE the section's geometry with the sub-arc itself and
/// drop the now-redundant cut from the owning face. The section on the KEPT
/// (other) face then welds to the split sub-arc by identical geometry — one
/// shared edge, no duplicate — exactly OCCT's "reuse the existing pave block".
///
/// The decision is IDENTITY-grounded, never a bare proximity grab (per the B1
/// falsification): BOTH section endpoints must project ONTO the boundary edge's
/// curve within the weld radius (at parameters inside its live window), and the
/// section must lie within a size-relative span band of the sub-arc in BOTH
/// directions across its whole span. It must ALSO deviate from the sub-arc by
/// MORE than the weld radius somewhere along the span — a section already within
/// the weld radius is a coincident / cosurface boundary-exchange copy the
/// assembler welds as-is, and reusing it would strand a coincident face's split.
/// Any geometry step that fails (sub-arc extraction, pcurve rebuild on the kept
/// face) aborts that one reuse and keeps the historical behaviour. Escape hatch
/// `BREP_SHARED_SECTION_EDGE=0` restores the mint-a-duplicate path wholesale
/// (the clean A/B lever).
pub(super) fn reuse_boundary_section_edges(
    result: &mut ImprintResultRecord,
    face_edge_lists: &HashMap<FaceKey, Vec<&EdgeRecord>>,
    solid_a: &BrepSolid,
    solid_b: &BrepSolid,
    tolerance: f64,
) -> Result<(), KernelRefusal> {
    if std::env::var("BREP_SHARED_SECTION_EDGE").as_deref() == Ok("0") {
        return Ok(());
    }
    let debug = std::env::var("BREP_DEBUG_SHARED_SECTION").is_ok();
    // Endpoint gate: the section's endpoints must lie ON the boundary edge's
    // curve (not merely near its end vertices) — the coincident edge is a SUB-
    // ARC that `apply_edge_splits` later mints at exactly these junction points.
    // Measured on fixture 09: matching endpoints project at ~1e-9; the nearest
    // non-matching edge is ≥3.5e-3 — 2.5 orders of separation.
    let endpoint_gate = assembler_weld(tolerance);
    let scale = solid_scale(solid_a).max(solid_scale(solid_b));
    let span_band = (SHARED_SECTION_BAND_FRACTION * scale).max(endpoint_gate);
    let piece_point: HashMap<u64, Vec3> =
        result.vertices.iter().map(|v| (v.id, v.point)).collect();
    let face_surface = |key: &FaceKey| -> Option<&NurbsSurface> {
        let solid = if key.operand == 0 { solid_a } else { solid_b };
        solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .find(|face| face.id == key.face_id)
            .map(|face| &face.surface)
    };

    struct Reuse {
        piece_index: usize,
        owning: FaceKey,
        edge_id: u64,
        arc: NurbsCurve,
        aligned: bool,
    }
    // Decide first (immutable borrow of result.pieces), apply after.
    let mut decisions: Vec<Reuse> = Vec::new();
    for (piece_index, piece) in result.pieces.iter().enumerate() {
        if piece.shared_edge.is_some() {
            continue;
        }
        let (Some(&ps), Some(&pe)) = (
            piece_point.get(&piece.start_vertex_id),
            piece_point.get(&piece.end_vertex_id),
        ) else {
            continue;
        };
        if ps.sub(pe).length() <= endpoint_gate {
            // CLOSED-RING REUSE (problemInbox equator-tangent): a closed
            // section ring coincident with an existing closed boundary edge
            // (the inscribed sphere's equator IS the cylinder's cap-edge
            // circle) can never be unified by the assembler's endpoint weld —
            // closed rings have no distinct endpoints to weld — so the two
            // records stay separate and strand one-use. Unlike the open-arc
            // path below, EXACT coincidence must be accepted here (no lower
            // floor): reuse is the ONLY identity mechanism available to
            // rings. Escape hatch: BREP_SHARED_SECTION_RING=0.
            if std::env::var("BREP_SHARED_SECTION_RING").as_deref() != Ok("0") {
                let mut chosen: Option<Reuse> = None;
                let mut best_score = f64::INFINITY;
                for face_key in &piece.support_faces {
                    let Some(edges) = face_edge_lists.get(face_key) else {
                        continue;
                    };
                    for edge in edges {
                        if edge.degenerate {
                            continue;
                        }
                        let (Ok(es), Ok(ee)) =
                            (edge.curve.evaluate(edge.t0), edge.curve.evaluate(edge.t1))
                        else {
                            continue;
                        };
                        if es.sub(ee).length() > endpoint_gate {
                            continue; // open edge cannot host a ring
                        }
                        let Ok(ring) = edge_subcurve(edge) else {
                            continue;
                        };
                        let (Ok(dev_a), Ok(dev_b)) = (
                            max_curve_deviation(&piece.curve, &ring),
                            max_curve_deviation(&ring, &piece.curve),
                        ) else {
                            continue;
                        };
                        let deviation = dev_a.max(dev_b);
                        // Rings ride a boundary edge only on EXACT identity
                        // (weld-scale): a near-coincident-but-distinct ring
                        // (t149: dev 1.1e-2) is real separate topology, and
                        // absorbing it corrupts the arrangement. The open-arc
                        // path's generous span_band does not apply here.
                        if deviation > endpoint_gate || deviation >= best_score {
                            continue;
                        }
                        // Orientation: compare tangents at the piece start's
                        // image on the ring.
                        let Ok(projection) = project_point_to_curve(&ring, ps) else {
                            continue;
                        };
                        let (Ok((_, piece_tangent)), Ok((_, ring_tangent))) =
                            (piece.curve.deriv1(piece.t0), ring.deriv1(projection.u))
                        else {
                            continue;
                        };
                        best_score = deviation;
                        chosen = Some(Reuse {
                            piece_index,
                            owning: *face_key,
                            edge_id: edge.id,
                            arc: ring.clone(),
                            aligned: piece_tangent.dot(ring_tangent) >= 0.0,
                        });
                    }
                }
                if let Some(reuse) = chosen {
                    if debug {
                        eprintln!(
                            "shared-section RING piece {} -> edge {}:{} dev={best_score:.3e}",
                            piece.id, reuse.owning.operand, reuse.edge_id
                        );
                    }
                    decisions.push(reuse);
                }
            }
            continue;
        }
        let mut chosen: Option<Reuse> = None;
        let mut best_score = f64::INFINITY;
        for face_key in &piece.support_faces {
            let Some(edges) = face_edge_lists.get(face_key) else {
                continue;
            };
            for edge in edges {
                if edge.degenerate {
                    continue;
                }
                // Both section endpoints must lie ON the edge's curve, at
                // parameters inside the edge's live window (closed rings
                // included — the parent is often a ring the section grazes).
                let (Ok(proj_s), Ok(proj_e)) = (
                    project_point_to_curve(&edge.curve, ps),
                    project_point_to_curve(&edge.curve, pe),
                ) else {
                    continue;
                };
                if proj_s.distance > endpoint_gate || proj_e.distance > endpoint_gate {
                    continue;
                }
                let lo_t = edge.t0.min(edge.t1);
                let hi_t = edge.t0.max(edge.t1);
                let margin = (hi_t - lo_t).abs() * 1e-6 + 1e-9;
                if proj_s.u < lo_t - margin
                    || proj_s.u > hi_t + margin
                    || proj_e.u < lo_t - margin
                    || proj_e.u > hi_t + margin
                {
                    continue;
                }
                let aligned = proj_s.u <= proj_e.u;
                let lo = proj_s.u.min(proj_e.u);
                let hi = proj_s.u.max(proj_e.u);
                if hi - lo <= margin {
                    continue; // sub-arc collapses to a point
                }
                // The sub-arc of THIS edge between the two junctions — the exact
                // 1-cell apply_edge_splits will mint. Orient it ps→pe and snap
                // its ends to the section's canonical vertices.
                let Ok(mut arc) = curve_param_subrange(&edge.curve, lo, hi) else {
                    continue;
                };
                if !aligned {
                    let Ok(reversed) = arc.reversed() else {
                        continue;
                    };
                    arc = reversed;
                }
                let Ok(arc) = snap_curve_ends(&arc, ps, pe) else {
                    continue;
                };
                // Bidirectional whole-span coincidence between the section and
                // the SUB-ARC (rejects partial overlap / different-path curves).
                let sec_to_arc = max_curve_deviation(&piece.curve, &arc)?;
                let arc_to_sec = max_curve_deviation(&arc, &piece.curve)?;
                if debug {
                    eprintln!(
                        "shared-section? piece {} sup=[{}:{},{}:{}] vs edge {}:{} \
                         proj=({:.2e},{:.2e}) sec->arc={:.3e} arc->sec={:.3e} band={:.3e}",
                        piece.id,
                        piece.support_faces[0].operand,
                        piece.support_faces[0].face_id,
                        piece.support_faces[1].operand,
                        piece.support_faces[1].face_id,
                        face_key.operand,
                        edge.id,
                        proj_s.distance,
                        proj_e.distance,
                        sec_to_arc,
                        arc_to_sec,
                        span_band,
                    );
                }
                // Lower floor is the crux of the identity discrimination: only
                // reuse when the section deviates from the sub-arc by MORE than
                // the weld radius — a genuine GRAZE the assembler's geometric
                // weld cannot fuse (fixture 09: 8.2e-4/1.67e-3). A section
                // ALREADY within the weld radius of a boundary edge is a
                // coincident / cosurface boundary-exchange copy (a section that
                // IS a boundary edge, deviation ~1e-15..1e-6); the assembler
                // welds those as-is, and dropping their cut would strand the
                // coincident face's split (revolve_pole_union one-use edges).
                let span_dev = sec_to_arc.max(arc_to_sec);
                if span_dev > endpoint_gate && span_dev <= span_band && span_dev < best_score {
                    best_score = span_dev;
                    chosen = Some(Reuse {
                        piece_index,
                        owning: *face_key,
                        edge_id: edge.id,
                        arc,
                        aligned,
                    });
                }
            }
        }
        if let Some(reuse) = chosen {
            decisions.push(reuse);
        }
    }

    // Apply: swap the section's geometry for the reused sub-arc, recompute the
    // pcurves on the KEPT faces, drop the redundant cut from the owning face.
    // Any failure in this path aborts this one reuse and keeps today's behaviour.
    for reuse in decisions {
        let piece = &result.pieces[reuse.piece_index];
        let mut new_pcurves = Vec::with_capacity(piece.pcurves.len());
        let mut ok = true;
        for facepc in &piece.pcurves {
            let key = FaceKey {
                operand: facepc.operand,
                face_id: facepc.face_id,
            };
            if key == reuse.owning {
                continue; // dropped — owning face already carries the edge
            }
            let Some(surface) = face_surface(&key) else {
                ok = false;
                break;
            };
            match build_pcurve_on_surface(surface, &reuse.arc) {
                Ok(pcurve) => new_pcurves.push(FacePcurve {
                    operand: key.operand,
                    face_id: key.face_id,
                    pcurve,
                }),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        let Ok([nt0, nt1]) = reuse.arc.domain() else {
            continue;
        };
        if !ok || new_pcurves.is_empty() {
            if debug {
                eprintln!(
                    "shared-section SKIP piece {} (pcurve rebuild on kept face failed)",
                    result.pieces[reuse.piece_index].id
                );
            }
            continue;
        }
        let piece = &mut result.pieces[reuse.piece_index];
        let piece_id = piece.id;
        piece.curve = reuse.arc;
        piece.t0 = nt0;
        piece.t1 = nt1;
        piece.pcurves = new_pcurves;
        piece.shared_edge = Some((reuse.owning.operand, reuse.edge_id, reuse.aligned));
        if let Some(entry) = result
            .by_face
            .iter_mut()
            .find(|f| f.operand == reuse.owning.operand && f.face_id == reuse.owning.face_id)
        {
            entry.piece_ids.retain(|id| *id != piece_id);
        }
        if debug {
            eprintln!(
                "shared-section REUSE piece {} -> boundary edge {}:{} sub-arc \
                 aligned={} (dropped redundant cut from face {}:{})",
                piece_id,
                reuse.owning.operand,
                reuse.edge_id,
                reuse.aligned,
                reuse.owning.operand,
                reuse.owning.face_id,
            );
        }
    }
    Ok(())
}

/// Canonical unordered key for a support pair.
fn support_pair_key(a: FaceKey, b: FaceKey) -> (FaceKey, FaceKey) {
    if (a.operand, a.face_id) <= (b.operand, b.face_id) {
        (a, b)
    } else {
        (b, a)
    }
}

/// TRUNCATED-SECTION EXTENSION (near-tangent SSI under-capture rescue).
///
/// Where a face F grazes another face G near-tangentially, the SSI marcher can
/// terminate F∩G's section curve at an INTERIOR turning point instead of the
/// crossing vertex where F's trim boundary actually meets G — the section piece
/// is left ~a-percent short of the corner. The stranded stub cannot separate
/// F's domain (a cut that reaches the boundary at only one end leaves the face
/// whole), and on G's side the cross-section loop gaps exactly there, so both
/// operands' fragments strand their neighbours' shared section edges as one-use
/// (non-integral assembly genus). The gap sits in the `[arr_tol, junction_
/// radius]` band the per-face bridges cannot close, and because F and G index
/// the SAME piece, healing it once at the imprint fixes both.
///
/// Add ONE bridging section piece from the dangling endpoint D to the orphaned
/// crossing vertex O, supported on the same pair {F, G}. Gate: D is a valence-1
/// piece endpoint interior to F (not at any crossing), O is a valence-1 vertex
/// ON F's trim boundary that carries NO {F,G} section endpoint, both share G,
/// within a span tied to the stub's own length, and the chord's midpoint sits
/// on F. A closed section loop has no such orphaned crossing, so the gate never
/// trips a face the marcher already reconstructed. Escape hatch
/// `BREP_EXTEND_TRUNCATED_SECTIONS=0`.
pub(super) fn extend_truncated_sections(
    result: &mut ImprintResultRecord,
    face_edge_lists: &HashMap<FaceKey, Vec<&EdgeRecord>>,
    solid_a: &BrepSolid,
    solid_b: &BrepSolid,
    tolerance: f64,
) -> Result<usize, KernelRefusal> {
    if std::env::var("BREP_EXTEND_TRUNCATED_SECTIONS").as_deref() == Ok("0") {
        return Ok(0);
    }
    let debug = std::env::var("BREP_DEBUG_EXTEND_SECTIONS").is_ok();

    let vpoint: HashMap<u64, Vec3> = result.vertices.iter().map(|v| (v.id, v.point)).collect();
    // Section-endpoint valence and each unordered support pair's endpoint set.
    let mut valence: HashMap<u64, usize> = HashMap::default();
    let mut pair_endpoints: HashMap<(FaceKey, FaceKey), HashSet<u64>> = HashMap::default();
    for piece in &result.pieces {
        for vid in [piece.start_vertex_id, piece.end_vertex_id] {
            *valence.entry(vid).or_insert(0) += 1;
        }
        let entry = pair_endpoints
            .entry(support_pair_key(piece.support_faces[0], piece.support_faces[1]))
            .or_default();
        entry.insert(piece.start_vertex_id);
        entry.insert(piece.end_vertex_id);
    }

    let surface_of = |key: FaceKey| -> Option<&NurbsSurface> {
        let solid = if key.operand == 0 { solid_a } else { solid_b };
        solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .find(|face| face.id == key.face_id)
            .map(|face| &face.surface)
    };
    // "On F's trim boundary" uses the same clearance polish_endpoint snaps
    // section termini to, so a genuine crossing (~tolerance*100 off) reads On
    // while a marcher truncation (a measurable stub short) reads Off.
    let on_boundary = |pt: Vec3, fkey: FaceKey| -> Result<bool, KernelRefusal> {
        let tol = (tolerance * 100.0).max(1e-6) * (1.0 + pt.length());
        if let Some(edges) = face_edge_lists.get(&fkey) {
            for edge in edges {
                if edge.degenerate {
                    continue;
                }
                if project_point_to_curve(&edge.curve, pt).or_refuse(KernelStage::Intersect, "project_point_to_curve")?.distance <= tol {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    };

    struct BridgeSpec {
        f: FaceKey,
        g: FaceKey,
        o: u64,
        d: u64,
        d_param: Vec3,
    }
    let mut bridges: Vec<BridgeSpec> = Vec::new();
    let mut claimed: HashSet<(u64, u64)> = HashSet::default();

    for piece in &result.pieces {
        let support = piece.support_faces;
        for &(f, g) in &[(support[0], support[1]), (support[1], support[0])] {
            let fg = support_pair_key(f, g);
            for (d, d_is_start) in [(piece.start_vertex_id, true), (piece.end_vertex_id, false)] {
                if valence.get(&d).copied().unwrap_or(0) != 1 {
                    continue;
                }
                let d_pt = vpoint[&d];
                // A dangling stub is interior to BOTH support faces — a genuine
                // terminus sits on a trim boundary.
                if on_boundary(d_pt, f)? || on_boundary(d_pt, g)? {
                    if debug {
                        eprintln!(
                            "extend: dangler v{d} F={}:{} G={}:{} REJECT on-boundary(f={},g={})",
                            f.operand,
                            f.face_id,
                            g.operand,
                            g.face_id,
                            on_boundary(d_pt, f)?,
                            on_boundary(d_pt, g)?
                        );
                    }
                    continue;
                }
                let plen = piece.curve.evaluate(piece.t0).or_refuse(KernelStage::Intersect, "evaluate")?.sub(piece.curve.evaluate(piece.t1).or_refuse(KernelStage::Intersect, "evaluate")?).length();
                let span_bound = (plen * 5.0).max(tolerance * 100.0);
                // Nearest orphaned crossing vertex on F's boundary that shares G.
                let mut best: Option<(f64, u64)> = None;
                for other in &result.pieces {
                    if other.support_faces[0] != g && other.support_faces[1] != g {
                        continue;
                    }
                    for o in [other.start_vertex_id, other.end_vertex_id] {
                        if o == d {
                            continue;
                        }
                        if pair_endpoints.get(&fg).is_some_and(|set| set.contains(&o)) {
                            continue; // already a {F,G} terminus — not orphaned
                        }
                        let o_pt = vpoint[&o];
                        let gap = o_pt.sub(d_pt).length();
                        if gap <= tolerance * 10.0 || gap > span_bound {
                            continue;
                        }
                        if !on_boundary(o_pt, f)? {
                            continue;
                        }
                        if best.is_none_or(|(g0, _)| gap < g0) {
                            best = Some((gap, o));
                        }
                    }
                }
                let Some((gap, o)) = best else {
                    if debug {
                        eprintln!(
                            "extend: dangler v{d} F={}:{} G={}:{} REJECT no-orphan (span_bound={span_bound:.3e})",
                            f.operand, f.face_id, g.operand, g.face_id
                        );
                    }
                    continue;
                };
                if claimed.contains(&(o, d)) || claimed.contains(&(d, o)) {
                    continue;
                }
                // Chord validity: the straight O→D extension must lie on F (it
                // approximates the missing near-tangent arc; reject if F bows away).
                let Some(fsurf) = surface_of(f) else { continue };
                let mid = vpoint[&o].add(d_pt).scale(0.5);
                let dev = project_point_to_surface(fsurf, mid).or_refuse(KernelStage::Intersect, "project_point_to_surface")?.distance;
                if dev > (tolerance * 100.0).max(0.25 * gap) * (1.0 + mid.length()) {
                    if debug {
                        eprintln!(
                            "extend: reject F={}:{} G={}:{} d={d} o={o} gap={gap:.3e} midpoint-dev={dev:.3e}",
                            f.operand, f.face_id, g.operand, g.face_id
                        );
                    }
                    continue;
                }
                // D's parameter on F, read from the stub's own F pcurve.
                let f_pcurve = pcurve_for_piece(piece, f)?;
                let [pd0, pd1] = f_pcurve.domain().or_refuse(KernelStage::Intersect, "domain")?;
                let d_param = f_pcurve.evaluate(if d_is_start { pd0 } else { pd1 }).or_refuse(KernelStage::Intersect, "evaluate")?;
                claimed.insert((o, d));
                if debug {
                    eprintln!(
                        "extend: bridge F={}:{} G={}:{} d={d} o={o} gap={gap:.3e}",
                        f.operand, f.face_id, g.operand, g.face_id
                    );
                }
                bridges.push(BridgeSpec { f, g, o, d, d_param });
            }
        }
    }

    if bridges.is_empty() {
        return Ok(0);
    }
    let mut next_id = result
        .vertices
        .iter()
        .map(|v| v.id)
        .chain(result.pieces.iter().map(|p| p.id))
        .max()
        .unwrap_or(0)
        + 1;
    let mut added = 0usize;
    for spec in bridges {
        let o_pt = vpoint[&spec.o];
        let d_pt = vpoint[&spec.d];
        let (Some(fsurf), Some(gsurf)) = (surface_of(spec.f), surface_of(spec.g)) else {
            continue;
        };
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Vec4::from_point(o_pt, 1.0), Vec4::from_point(d_pt, 1.0)],
        ).or_refuse(KernelStage::Intersect, "csg.imprint.sections")?;
        // G pcurve: exact for an affine carrier (the common cutting plane),
        // sampled-and-fit otherwise.
        let g_pcurve = build_pcurve_on_surface(gsurf, &line).or_refuse(KernelStage::Intersect, "build_pcurve_on_surface")?;
        // F pcurve: a param-space chord whose O end is projected SEEDED from the
        // D end, so a seam-straddling F cannot flip O onto the wrong branch.
        let o_proj = project_point_to_surface_seeded(fsurf, o_pt, spec.d_param.x, spec.d_param.y).or_refuse(KernelStage::Intersect, "project_point_to_surface_seeded")?;
        // On a seam-closed F, the projection can return O on the far seam copy
        // (u+period) while D sits on the near one; a param-space chord between
        // them would cross the seam and the arrangement then splits D onto the
        // opposite side, disconnecting the bridge from the stub it extends.
        // Unwrap O onto the seam representative NEAREST D so the chord stays on
        // one sheet and its ends coincide with the stub and boundary chains.
        let (closed_u, closed_v) = fsurf.closed_directions().or_refuse(KernelStage::Intersect, "closed_directions")?;
        let [fu0, fu1] = fsurf.domain_u().or_refuse(KernelStage::Intersect, "domain_u")?;
        let [fv0, fv1] = fsurf.domain_v().or_refuse(KernelStage::Intersect, "domain_v")?;
        let unwrap = |value: f64, target: f64, lo: f64, hi: f64, closed: bool| {
            if !closed {
                return value;
            }
            let period = hi - lo;
            if period <= 0.0 {
                return value;
            }
            let mut result = value;
            while result - target > period * 0.5 {
                result -= period;
            }
            while target - result > period * 0.5 {
                result += period;
            }
            result
        };
        let o_u = unwrap(o_proj.u, spec.d_param.x, fu0, fu1, closed_u);
        let o_v = unwrap(o_proj.v, spec.d_param.y, fv0, fv1, closed_v);
        if std::env::var("BREP_DEBUG_EXTEND_SECTIONS").is_ok() {
            eprintln!(
                "extend apply: F={}:{} d_param=({:.4},{:.4}) o_param=({:.4},{:.4},dist={:.2e})",
                spec.f.operand, spec.f.face_id, spec.d_param.x, spec.d_param.y,
                o_u, o_v, o_proj.distance
            );
        }
        let f_pcurve = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                Vec4::from_point(Vec3::new(o_u, o_v, 0.0), 1.0),
                Vec4::from_point(Vec3::new(spec.d_param.x, spec.d_param.y, 0.0), 1.0),
            ],
        ).or_refuse(KernelStage::Intersect, "csg.imprint.sections")?;
        let id = next_id;
        next_id += 1;
        // operand-0 face first, matching the freshly-minted-piece convention.
        let (support_faces, pcurves) = if spec.f.operand <= spec.g.operand {
            (
                [spec.f, spec.g],
                vec![
                    FacePcurve { operand: spec.f.operand, face_id: spec.f.face_id, pcurve: f_pcurve },
                    FacePcurve { operand: spec.g.operand, face_id: spec.g.face_id, pcurve: g_pcurve },
                ],
            )
        } else {
            (
                [spec.g, spec.f],
                vec![
                    FacePcurve { operand: spec.g.operand, face_id: spec.g.face_id, pcurve: g_pcurve },
                    FacePcurve { operand: spec.f.operand, face_id: spec.f.face_id, pcurve: f_pcurve },
                ],
            )
        };
        result.pieces.push(ImprintPieceRecord {
            id,
            curve: line,
            t0: 0.0,
            t1: 1.0,
            start_vertex_id: spec.o,
            end_vertex_id: spec.d,
            pcurves,
            support_faces,
            shared_edge: None,
        });
        for key in [spec.f, spec.g] {
            match result.by_face.iter_mut().find(|record| {
                record.operand == key.operand && record.face_id == key.face_id
            }) {
                Some(record) => record.piece_ids.push(id),
                None => result.by_face.push(FaceImprints {
                    operand: key.operand,
                    face_id: key.face_id,
                    piece_ids: vec![id],
                }),
            }
        }
        added += 1;
    }
    Ok(added)
}

fn pcurve_for_piece(piece: &ImprintPieceRecord, key: FaceKey) -> Result<&NurbsCurve, KernelRefusal> {
    piece
        .pcurves
        .iter()
        .find(|pcurve| pcurve.operand == key.operand && pcurve.face_id == key.face_id)
        .map(|pcurve| &pcurve.pcurve)
        .ok_or_else(|| KernelRefusal::internal(KernelStage::Intersect, "imprint.sections", "extend_truncated_sections: piece lacks face pcurve"))
}
