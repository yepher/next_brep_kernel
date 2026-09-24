use super::*;

struct RegionOutcome {
    fit: CandidateFit,
    accepted: bool,
    area: f64,
    bbox: (Vec3, Vec3),
    triangle_count: usize,
}

/// Try the analytic cascade plane → cylinder → cone → sphere → torus on one
/// region; the first carrier within tolerance wins.
fn fit_region(data: &MeshData, tri_ids: &[u32], options: &SegmentOptions) -> RegionOutcome {
    let vert_ids = region_vertices(data, tri_ids);
    let bbox = region_bbox(data, &vert_ids);
    let area: f64 = tri_ids.iter().map(|&t| data.tris[t as usize].area).sum();
    let scale = {
        let diag = bbox.1.sub(bbox.0).length();
        if diag > 0.0 {
            diag
        } else {
            data.diag.max(1e-12)
        }
    };
    let tol_abs = options.fit_tolerance * scale;
    let mut outcome = RegionOutcome {
        fit: CandidateFit {
            carrier: RegionCarrier::Freeform,
            max_dev: 0.0,
            rms_dev: 0.0,
            max_normal_angle_deg: 0.0,
        },
        accepted: false,
        area,
        bbox,
        triangle_count: tri_ids.len(),
    };
    if tri_ids.len() < options.min_region_triangles || vert_ids.len() < 3 {
        return outcome;
    }
    let candidates = [
        fit_plane(data, tri_ids, &vert_ids),
        fit_cylinder(data, tri_ids, &vert_ids),
        fit_cone(data, tri_ids, &vert_ids, tol_abs),
        fit_sphere(data, tri_ids, &vert_ids),
        fit_torus(data, tri_ids, &vert_ids),
    ];
    for candidate in candidates {
        if let Some(fit) = candidate {
            if std::env::var("BREP_DEBUG_SEG").is_ok() {
                eprintln!(
                    "[seg] region tris={} candidate={} max_dev={:.3e} tol_abs={:.3e} max_ang={:.3}",
                    tri_ids.len(),
                    fit.carrier.kind(),
                    fit.max_dev,
                    tol_abs,
                    fit.max_normal_angle_deg
                );
            }
            if fit.max_dev.is_finite()
                && fit.max_dev <= tol_abs
                && fit.max_normal_angle_deg <= options.normal_tolerance_deg
            {
                outcome.fit = fit;
                outcome.accepted = true;
                return outcome;
            }
        }
    }
    outcome
}

// ---------------------------------------------------------------------------
// Region growing and refinement
// ---------------------------------------------------------------------------

fn grow_regions(data: &MeshData, options: &SegmentOptions) -> Vec<Vec<u32>> {
    let cos_gate = options.deflection_angle_deg.to_radians().cos();
    let tri_count = data.tris.len();
    let mut assigned = vec![false; tri_count];
    let mut regions = Vec::new();
    let mut stack = Vec::new();
    for seed in 0..tri_count {
        if assigned[seed] || data.tris[seed].skip {
            continue;
        }
        assigned[seed] = true;
        stack.push(seed as u32);
        let mut region = Vec::new();
        while let Some(t) = stack.pop() {
            region.push(t);
            let tri = &data.tris[t as usize];
            for corner in 0..3 {
                let first = tri.verts[corner];
                let second = tri.verts[(corner + 1) % 3];
                if first == second {
                    continue;
                }
                let Some(incident) = data.edges.get(&edge_key(first, second)) else {
                    continue;
                };
                // Only manifold (two-sided) edges are smooth crossings.
                if incident.len() != 2 {
                    continue;
                }
                for &other in incident {
                    if other == t || assigned[other as usize] {
                        continue;
                    }
                    let neighbor = &data.tris[other as usize];
                    if neighbor.skip {
                        continue;
                    }
                    if tri.normal.dot(neighbor.normal) >= cos_gate {
                        assigned[other as usize] = true;
                        stack.push(other);
                    }
                }
            }
        }
        region.sort_unstable();
        regions.push(region);
    }
    regions
}

/// Regions whose area is below this fraction of the total mesh area are
/// dissolved into their largest neighboring region before fitting.  Needle
/// triangles at tessellation poles wall themselves off into 1–2-triangle
/// islands; the app's minimum-region-size gating served the same purpose.
const REGION_ABSORB_AREA_FRACTION: f64 = 1e-6;

/// Merge negligible regions into their largest-area neighbor (repeats until
/// stable so chains of islands resolve).  Regions with no neighbor at all
/// stay as they are.
fn absorb_tiny_regions(data: &MeshData, regions: &mut Vec<Vec<u32>>) {
    let total_area: f64 = data
        .tris
        .iter()
        .filter(|tri| !tri.skip)
        .map(|tri| tri.area)
        .sum();
    let floor = total_area * REGION_ABSORB_AREA_FRACTION;
    if !(floor > 0.0) {
        return;
    }
    let mut region_of = vec![usize::MAX; data.tris.len()];
    for (r, list) in regions.iter().enumerate() {
        for &t in list {
            region_of[t as usize] = r;
        }
    }
    let mut areas: Vec<f64> = regions
        .iter()
        .map(|list| list.iter().map(|&t| data.tris[t as usize].area).sum())
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for r in 0..regions.len() {
            if regions[r].is_empty() || areas[r] >= floor {
                continue;
            }
            // Find adjacent regions, traversing THROUGH region-less
            // (skipped sliver) triangles — pole islands are often fenced
            // off from the main region by a ring of skipped needles.
            let mut target: Option<usize> = None;
            let mut visited: rustc_hash::FxHashSet<u32> = regions[r].iter().copied().collect();
            let mut frontier: Vec<u32> = regions[r].clone();
            while let Some(t) = frontier.pop() {
                let tri = &data.tris[t as usize];
                for corner in 0..3 {
                    let first = tri.verts[corner];
                    let second = tri.verts[(corner + 1) % 3];
                    if first == second {
                        continue;
                    }
                    if let Some(incident) = data.edges.get(&edge_key(first, second)) {
                        for &other in incident {
                            if visited.contains(&other) {
                                continue;
                            }
                            let neighbor = region_of[other as usize];
                            if neighbor == usize::MAX {
                                // Skipped triangle: pass through it.
                                visited.insert(other);
                                frontier.push(other);
                                continue;
                            }
                            if neighbor == r {
                                continue;
                            }
                            target = match target {
                                None => Some(neighbor),
                                Some(current) => Some(if areas[neighbor] > areas[current] {
                                    neighbor
                                } else {
                                    current
                                }),
                            };
                        }
                    }
                }
            }
            if let Some(target) = target {
                let moved = std::mem::take(&mut regions[r]);
                for &t in &moved {
                    region_of[t as usize] = target;
                }
                areas[target] += areas[r];
                areas[r] = 0.0;
                regions[target].extend(moved);
                changed = true;
            }
        }
    }
    regions.retain(|list| !list.is_empty());
    for list in regions.iter_mut() {
        list.sort_unstable();
    }
}

/// Connected components of `tri_ids` across shared (welded) edges,
/// restricted to the given set.
fn connected_components(data: &MeshData, tri_ids: &[u32]) -> Vec<Vec<u32>> {
    let member: rustc_hash::FxHashSet<u32> = tri_ids.iter().copied().collect();
    let mut assigned: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    let mut components = Vec::new();
    let mut stack = Vec::new();
    for &seed in tri_ids {
        if assigned.contains(&seed) {
            continue;
        }
        assigned.insert(seed);
        stack.push(seed);
        let mut component = Vec::new();
        while let Some(t) = stack.pop() {
            component.push(t);
            let tri = &data.tris[t as usize];
            for corner in 0..3 {
                let first = tri.verts[corner];
                let second = tri.verts[(corner + 1) % 3];
                if first == second {
                    continue;
                }
                if let Some(incident) = data.edges.get(&edge_key(first, second)) {
                    for &other in incident {
                        if other != t && member.contains(&other) && !assigned.contains(&other) {
                            assigned.insert(other);
                            stack.push(other);
                        }
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// Split a region the cascade could not classify: peel off seed-plane-
/// anchored coplanar groups (the app's planar extraction), then re-fit the
/// remaining connected components.  Returns the partition with each piece's
/// fit outcome.
fn refine_region(
    data: &MeshData,
    tri_ids: &[u32],
    options: &SegmentOptions,
) -> Vec<(Vec<u32>, RegionOutcome)> {
    let vert_ids = region_vertices(data, tri_ids);
    let (low, high) = region_bbox(data, &vert_ids);
    let scale = {
        let diag = high.sub(low).length();
        if diag > 0.0 {
            diag
        } else {
            data.diag.max(1e-12)
        }
    };
    let dist_gate = options.fit_tolerance * scale;
    let cos_gate = options.planar_extraction_angle_deg.to_radians().cos();
    let region_area: f64 = tri_ids.iter().map(|&t| data.tris[t as usize].area).sum();
    let min_group_area = region_area * (options.planar_min_area_percent / 100.0).clamp(0.0, 1.0);
    let min_group_tris = options.min_region_triangles.max(2);

    let member: rustc_hash::FxHashSet<u32> = tri_ids.iter().copied().collect();
    let mut order: Vec<u32> = tri_ids.to_vec();
    order.sort_by(|&a, &b| {
        data.tris[b as usize]
            .area
            .total_cmp(&data.tris[a as usize].area)
            .then(a.cmp(&b))
    });

    let mut group_of: HashMap<u32, usize> = HashMap::default();
    let mut planar_groups: Vec<Vec<u32>> = Vec::new();
    let mut stack = Vec::new();
    for &seed in &order {
        if group_of.contains_key(&seed) {
            continue;
        }
        let seed_tri = &data.tris[seed as usize];
        let seed_normal = seed_tri.normal;
        let seed_origin = seed_tri.centroid;
        let coplanar = |t: u32| -> bool {
            let tri = &data.tris[t as usize];
            if tri.normal.dot(seed_normal) < cos_gate {
                return false;
            }
            tri.verts
                .iter()
                .all(|&v| data.verts[v].sub(seed_origin).dot(seed_normal).abs() <= dist_gate)
        };
        if !coplanar(seed) {
            continue;
        }
        // Grow the seed plane over not-yet-grouped region triangles.
        let mut visited: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
        visited.insert(seed);
        stack.push(seed);
        let mut group = Vec::new();
        while let Some(t) = stack.pop() {
            group.push(t);
            let tri = &data.tris[t as usize];
            for corner in 0..3 {
                let first = tri.verts[corner];
                let second = tri.verts[(corner + 1) % 3];
                if first == second {
                    continue;
                }
                if let Some(incident) = data.edges.get(&edge_key(first, second)) {
                    for &other in incident {
                        if other == t
                            || !member.contains(&other)
                            || group_of.contains_key(&other)
                            || visited.contains(&other)
                        {
                            continue;
                        }
                        if coplanar(other) {
                            visited.insert(other);
                            stack.push(other);
                        }
                    }
                }
            }
        }
        let group_area: f64 = group.iter().map(|&t| data.tris[t as usize].area).sum();
        if group.len() >= min_group_tris && group_area >= min_group_area {
            let id = planar_groups.len();
            for &t in &group {
                group_of.insert(t, id);
            }
            group.sort_unstable();
            planar_groups.push(group);
        }
    }

    let remainder: Vec<u32> = tri_ids
        .iter()
        .copied()
        .filter(|t| !group_of.contains_key(t))
        .collect();

    let mut pieces = Vec::new();
    for group in planar_groups {
        let outcome = fit_region(data, &group, options);
        pieces.push((group, outcome));
    }
    for component in connected_components(data, &remainder) {
        let outcome = fit_region(data, &component, options);
        pieces.push((component, outcome));
    }
    pieces
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Segment a triangle mesh into smooth regions and recognize each region's
/// analytic carrier.  `positions` are xyz triples; `indices` is a triangle
/// index buffer, or empty to treat `positions` as a raw soup of consecutive
/// triangles (the binary-STL layout of `read_binary_stl`).
pub fn segment_mesh_faces(
    positions: &[f64],
    indices: &[u32],
    options: &SegmentOptions,
) -> Result<MeshSegmentation, String> {
    if !(options.deflection_angle_deg > 0.0) || !options.deflection_angle_deg.is_finite() {
        return Err("segment_mesh_faces: deflection angle must be positive".into());
    }
    if !(options.fit_tolerance > 0.0) || !options.fit_tolerance.is_finite() {
        return Err("segment_mesh_faces: fit tolerance must be positive".into());
    }
    let data = build_mesh_data(positions, indices, options)?;
    Ok(segment_prepared(&data, options))
}

/// Segmentation core on an already-welded mesh: shared by the analysis entry
/// (`segment_mesh_faces`) and the BREP rebuild (`mesh_regions_to_brep`), so
/// both see identical triangle/vertex indexing.
pub(super) fn segment_prepared(data: &MeshData, options: &SegmentOptions) -> MeshSegmentation {
    let mut raw_regions = grow_regions(data, options);
    absorb_tiny_regions(data, &mut raw_regions);

    let mut final_regions: Vec<(Vec<u32>, RegionOutcome)> = Vec::new();
    for region in raw_regions {
        let outcome = fit_region(data, &region, options);
        if outcome.accepted {
            final_regions.push((region, outcome));
            continue;
        }
        // Tangent-smooth compound (or genuinely freeform): try to split it.
        let pieces = refine_region(data, &region, options);
        if pieces.len() <= 1 {
            // Refinement did not split anything; keep the region whole.
            final_regions.push((region, outcome));
        } else {
            final_regions.extend(pieces);
        }
    }

    // Reunite regions the growing/refinement pipeline split although they
    // lie on ONE carrier (a cap triangle isolated behind skipped slivers, a
    // blend peeled into two coaxial pieces).  Downstream consumers — the
    // BREP rebuild above all — rely on "one region per face".
    merge_compatible_regions(data, &mut final_regions, options);

    let mut triangle_region_ids = vec![UNASSIGNED_REGION; data.tris.len()];
    let mut regions = Vec::with_capacity(final_regions.len());
    for (id, (tri_ids, outcome)) in final_regions.into_iter().enumerate() {
        for &t in &tri_ids {
            triangle_region_ids[t as usize] = id as u32;
        }
        regions.push(MeshRegion {
            id: id as u32,
            triangle_count: outcome.triangle_count,
            area: outcome.area,
            bbox_min: outcome.bbox.0,
            bbox_max: outcome.bbox.1,
            carrier: outcome.fit.carrier,
            max_deviation: outcome.fit.max_dev,
            rms_deviation: outcome.fit.rms_dev,
            max_normal_angle_deg: outcome.fit.max_normal_angle_deg,
        });
    }

    // Attach degenerate/micro triangles to an adjacent region when one
    // exists.  Multi-pass so chains of skipped slivers resolve through
    // their assigned neighbors.
    let mut changed = true;
    while changed {
        changed = false;
        for t in 0..data.tris.len() {
            if triangle_region_ids[t] != UNASSIGNED_REGION {
                continue;
            }
            let tri = &data.tris[t];
            let mut adopted = UNASSIGNED_REGION;
            for corner in 0..3 {
                let first = tri.verts[corner];
                let second = tri.verts[(corner + 1) % 3];
                if first == second {
                    continue;
                }
                if let Some(incident) = data.edges.get(&edge_key(first, second)) {
                    for &other in incident {
                        let region = triangle_region_ids[other as usize];
                        if other as usize != t && region != UNASSIGNED_REGION {
                            adopted = adopted.min(region);
                        }
                    }
                }
            }
            if adopted != UNASSIGNED_REGION {
                triangle_region_ids[t] = adopted;
                changed = true;
            }
        }
    }

    MeshSegmentation {
        triangle_region_ids,
        regions,
        triangle_count: data.tris.len(),
        welded_vertex_count: data.verts.len(),
    }
}

/// Do two accepted carriers describe the same surface (within `tol` for
/// distances, tight angular bands for directions)?  Freeform never matches.
fn carriers_compatible(a: &RegionCarrier, b: &RegionCarrier, tol: f64) -> bool {
    const ANGULAR: f64 = 1e-6;
    match (a, b) {
        (
            RegionCarrier::Plane {
                origin: o1,
                normal: n1,
            },
            RegionCarrier::Plane {
                origin: o2,
                normal: n2,
            },
        ) => {
            n1.cross(*n2).length() <= ANGULAR
                && n1.dot(*n2) > 0.0
                && o2.sub(*o1).dot(*n1).abs() <= tol
        }
        (
            RegionCarrier::Cylinder {
                axis_point: p1,
                axis_dir: a1,
                radius: r1,
                sense: s1,
            },
            RegionCarrier::Cylinder {
                axis_point: p2,
                axis_dir: a2,
                radius: r2,
                sense: s2,
            },
        ) => {
            let offset = p2.sub(*p1);
            s1 == s2
                && a1.cross(*a2).length() <= ANGULAR
                && (r1 - r2).abs() <= tol
                && offset.sub(a1.scale(offset.dot(*a1))).length() <= tol
        }
        (
            RegionCarrier::Cone {
                apex: p1,
                axis_dir: a1,
                half_angle_rad: h1,
                sense: s1,
            },
            RegionCarrier::Cone {
                apex: p2,
                axis_dir: a2,
                half_angle_rad: h2,
                sense: s2,
            },
        ) => {
            s1 == s2
                && a1.cross(*a2).length() <= ANGULAR
                && a1.dot(*a2) > 0.0
                && (h1 - h2).abs() <= ANGULAR
                && p2.sub(*p1).length() <= tol
        }
        (
            RegionCarrier::Sphere {
                center: c1,
                radius: r1,
                sense: s1,
            },
            RegionCarrier::Sphere {
                center: c2,
                radius: r2,
                sense: s2,
            },
        ) => s1 == s2 && c2.sub(*c1).length() <= tol && (r1 - r2).abs() <= tol,
        (
            RegionCarrier::Torus {
                center: c1,
                axis_dir: a1,
                major_radius: mj1,
                minor_radius: mn1,
                sense: s1,
            },
            RegionCarrier::Torus {
                center: c2,
                axis_dir: a2,
                major_radius: mj2,
                minor_radius: mn2,
                sense: s2,
            },
        ) => {
            s1 == s2
                && a1.cross(*a2).length() <= ANGULAR
                && c2.sub(*c1).length() <= tol
                && (mj1 - mj2).abs() <= tol
                && (mn1 - mn2).abs() <= tol
        }
        _ => false,
    }
}

/// Merge edge-adjacent regions whose accepted carriers agree, repeating
/// until stable.  Adjacency counts paths THROUGH skipped (degenerate/micro)
/// triangles — exactly the fences that isolate cap islands in the first
/// place.  A merge is only applied when the union re-fits cleanly on the
/// same carrier kind, so this pass can only improve the segmentation.
fn merge_compatible_regions(
    data: &MeshData,
    regions: &mut Vec<(Vec<u32>, RegionOutcome)>,
    options: &SegmentOptions,
) {
    let tol = options.fit_tolerance * data.diag.max(1e-12);
    loop {
        let mut region_of = vec![usize::MAX; data.tris.len()];
        for (index, (tris, _)) in regions.iter().enumerate() {
            for &t in tris {
                region_of[t as usize] = index;
            }
        }
        // Directly edge-adjacent pairs.
        let mut pairs: std::collections::BTreeSet<(usize, usize)> =
            std::collections::BTreeSet::new();
        for incident in data.edges.values() {
            if incident.len() != 2 {
                continue;
            }
            let (a, b) = (
                region_of[incident[0] as usize],
                region_of[incident[1] as usize],
            );
            if a != b && a != usize::MAX && b != usize::MAX {
                pairs.insert((a.min(b), a.max(b)));
            }
        }
        // Pairs bridged by connected runs of region-less (skipped) triangles.
        let mut seen = vec![false; data.tris.len()];
        for seed in 0..data.tris.len() {
            if seen[seed] || region_of[seed] != usize::MAX {
                continue;
            }
            let mut stack = vec![seed];
            let mut touched: Vec<usize> = Vec::new();
            seen[seed] = true;
            while let Some(t) = stack.pop() {
                let tri = &data.tris[t];
                for corner in 0..3 {
                    let first = tri.verts[corner];
                    let second = tri.verts[(corner + 1) % 3];
                    if first == second {
                        continue;
                    }
                    if let Some(incident) = data.edges.get(&edge_key(first, second)) {
                        for &other in incident {
                            let other = other as usize;
                            if other == t {
                                continue;
                            }
                            let region = region_of[other];
                            if region == usize::MAX {
                                if !seen[other] {
                                    seen[other] = true;
                                    stack.push(other);
                                }
                            } else if !touched.contains(&region) {
                                touched.push(region);
                            }
                        }
                    }
                }
            }
            touched.sort_unstable();
            for i in 0..touched.len() {
                for j in i + 1..touched.len() {
                    pairs.insert((touched[i], touched[j]));
                }
            }
        }

        let mut merged_any = false;
        'pairs: for (a, b) in pairs {
            let (first, second) = (&regions[a], &regions[b]);
            if !first.1.accepted || !second.1.accepted {
                continue;
            }
            if !carriers_compatible(&first.1.fit.carrier, &second.1.fit.carrier, tol) {
                continue;
            }
            let mut union: Vec<u32> = first.0.iter().chain(second.0.iter()).copied().collect();
            union.sort_unstable();
            let outcome = fit_region(data, &union, options);
            if !outcome.accepted || outcome.fit.carrier.kind() != first.1.fit.carrier.kind() {
                continue;
            }
            regions[a] = (union, outcome);
            regions.remove(b);
            merged_any = true;
            break 'pairs;
        }
        if !merged_any {
            return;
        }
    }
}
