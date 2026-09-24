use crate::mesh_weld::{edge_key, weld_vertex};
use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{make_line, make_plane, tessellate_face, Mesh, TessellationOptions, Vec3};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Convert an open exact assembly to a closed, welded triangular BREP.
///
/// This is a last-resort native recovery for self-intersection transition
/// values where carrier selection leaves an otherwise useful surface mesh
/// with boundary cycles. Exact assemblies never pass through this function.
pub(crate) fn close_as_faceted_brep(
    solid: &BrepSolid,
    tolerance: f64,
) -> Result<BrepSolid, String> {
    let options = TessellationOptions {
        slabs_per_span_u: 12,
        steps_per_span_v: 12,
    };
    let mut mesh = Mesh::default();
    let mut face_id = 0;
    for face in solid.shells.iter().flat_map(|shell| &shell.faces) {
        if let Ok(face_mesh) = tessellate_face(face, options, face_id) {
            let base = (mesh.positions.len() / 3) as u32;
            mesh.positions.extend(face_mesh.positions);
            mesh.normals.extend(face_mesh.normals);
            mesh.indices
                .extend(face_mesh.indices.into_iter().map(|index| index + base));
            mesh.face_ids.extend(face_mesh.face_ids);
        }
        face_id += 1;
    }
    if mesh.indices.is_empty() {
        return Err("faceted shell repair could not tessellate any faces".into());
    }
    triangle_soup_to_faceted_brep(&mesh.positions, Some(&mesh.indices), tolerance.max(2e-3))
}

/// Build a faceted (planar-triangle) BREP solid from a triangle soup or an
/// indexed triangle mesh — the STL/mesh import entry (§8.6 inverse: mesh →
/// body). Vertices weld within `tolerance` (pass `<= 0` to derive it from the
/// bounding-box diagonal), duplicate and degenerate triangles are dropped,
/// surplus sheets at non-manifold edges are pruned, and remaining boundary
/// cycles are capped with a fan so the result closes watertight whenever the
/// input is close to a manifold. Validation is authoritative: an input too
/// broken to close returns an error, never a silently-invalid solid.
pub fn mesh_to_faceted_brep(
    positions: &[f64],
    indices: Option<&[u32]>,
    tolerance: f64,
) -> Result<BrepSolid, String> {
    if positions.is_empty() || positions.len() % 3 != 0 {
        return Err("mesh_to_faceted_brep: positions must be xyz triples".into());
    }
    let weld_tolerance = if tolerance > 0.0 && tolerance.is_finite() {
        tolerance
    } else {
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for point in positions.chunks_exact(3) {
            for axis in 0..3 {
                low[axis] = low[axis].min(point[axis]);
                high[axis] = high[axis].max(point[axis]);
            }
        }
        let diagonal = (0..3)
            .map(|axis| (high[axis] - low[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        (diagonal * 1e-6).max(1e-12)
    };
    triangle_soup_to_faceted_brep(positions, indices, weld_tolerance)
}

/// Shared weld → prune → cap → build pipeline. `indices` of `None` treats
/// `positions` as a raw soup (three consecutive vertices per triangle, the
/// binary-STL layout).
fn triangle_soup_to_faceted_brep(
    positions: &[f64],
    indices: Option<&[u32]>,
    weld_tolerance: f64,
) -> Result<BrepSolid, String> {
    let mut points = Vec::<Vec3>::new();
    let mut buckets = HashMap::<[i64; 3], Vec<usize>>::default();
    let mut mesh_to_welded = Vec::with_capacity(positions.len() / 3);
    for position in positions.chunks_exact(3) {
        mesh_to_welded.push(weld_vertex(
            Vec3::new(position[0], position[1], position[2]),
            weld_tolerance,
            &mut points,
            &mut buckets,
        ));
    }
    let sequential;
    let indices = match indices {
        Some(indices) => indices,
        None => {
            sequential = (0..mesh_to_welded.len() as u32).collect::<Vec<_>>();
            &sequential
        }
    };
    let mut triangles = Vec::<[usize; 3]>::new();
    let mut seen = HashSet::<[usize; 3]>::default();
    for triangle in indices.chunks_exact(3) {
        if triangle
            .iter()
            .any(|index| *index as usize >= mesh_to_welded.len())
        {
            return Err("mesh_to_faceted_brep: triangle index outside positions".into());
        }
        let value = [
            mesh_to_welded[triangle[0] as usize],
            mesh_to_welded[triangle[1] as usize],
            mesh_to_welded[triangle[2] as usize],
        ];
        if value[0] == value[1] || value[1] == value[2] || value[2] == value[0] {
            continue;
        }
        let mut key = value;
        key.sort_unstable();
        if seen.insert(key) {
            triangles.push(value);
        }
    }
    if triangles.is_empty() {
        return Err("mesh_to_faceted_brep: no non-degenerate triangles".into());
    }

    // Remove surplus coincident sheets at non-manifold triangle edges,
    // retaining one use in each direction whenever possible.
    loop {
        let mut edge_uses = HashMap::<(usize, usize), Vec<(usize, bool)>>::default();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            for side in 0..3 {
                let first = triangle[side];
                let second = triangle[(side + 1) % 3];
                edge_uses
                    .entry(edge_key(first, second))
                    .or_default()
                    .push((triangle_index, first < second));
            }
        }
        let Some(uses) = edge_uses.values().find(|uses| uses.len() > 2) else {
            break;
        };
        let keep_forward = uses.iter().find(|(_, forward)| *forward).map(|use_| use_.0);
        let keep_reverse = uses
            .iter()
            .find(|(_, forward)| !*forward)
            .map(|use_| use_.0);
        let keep = [keep_forward, keep_reverse]
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>();
        let remove = uses
            .iter()
            .map(|use_| use_.0)
            .filter(|index| !keep.contains(index))
            .collect::<HashSet<_>>();
        if remove.is_empty() {
            break;
        }
        triangles = triangles
            .into_iter()
            .enumerate()
            .filter_map(|(index, triangle)| (!remove.contains(&index)).then_some(triangle))
            .collect();
    }

    let mut directed_boundary = Vec::<(usize, usize)>::new();
    let mut counts = HashMap::<(usize, usize), usize>::default();
    for triangle in &triangles {
        for side in 0..3 {
            *counts
                .entry(edge_key(triangle[side], triangle[(side + 1) % 3]))
                .or_default() += 1;
        }
    }
    for triangle in &triangles {
        for side in 0..3 {
            let first = triangle[side];
            let second = triangle[(side + 1) % 3];
            if counts[&edge_key(first, second)] == 1 {
                directed_boundary.push((first, second));
            }
        }
    }
    while let Some((start, mut end)) = directed_boundary.pop() {
        let mut loop_vertices = vec![start, end];
        while end != start {
            let Some(index) = directed_boundary
                .iter()
                .position(|(candidate_start, _)| *candidate_start == end)
            else {
                break;
            };
            let (_, next) = directed_boundary.swap_remove(index);
            end = next;
            if end != start {
                loop_vertices.push(end);
            }
            if loop_vertices.len() > points.len() + 1 {
                break;
            }
        }
        if end != start || loop_vertices.len() < 3 {
            continue;
        }
        let center = loop_vertices
            .iter()
            .fold(Vec3::default(), |sum, index| sum.add(points[*index]))
            .scale(1.0 / loop_vertices.len() as f64);
        let center_index = points.len();
        points.push(center);
        for index in 0..loop_vertices.len() {
            let first = loop_vertices[index];
            let second = loop_vertices[(index + 1) % loop_vertices.len()];
            // Reverse the existing boundary direction on the new cap.
            triangles.push([second, first, center_index]);
        }
    }

    let vertices = points
        .iter()
        .enumerate()
        .map(|(index, point)| VertexRecord {
            id: index as u64 + 1,
            point: *point,
        })
        .collect::<Vec<_>>();
    let mut edges = Vec::<EdgeRecord>::new();
    let mut edge_ids = HashMap::<(usize, usize), u64>::default();
    let mut faces = Vec::<FaceRecord>::new();
    let mut next_id = vertices.len() as u64 + 1;
    for triangle in triangles {
        let a = points[triangle[0]];
        let b = points[triangle[1]];
        let c = points[triangle[2]];
        let u = b.sub(a);
        let u_length = u.length();
        if u_length <= 1e-10 {
            continue;
        }
        let u_direction = u.scale(1.0 / u_length);
        let c_delta = c.sub(a);
        let c_u = c_delta.dot(u_direction);
        let v = c_delta.sub(u_direction.scale(c_u));
        let v_length = v.length();
        if v_length <= 1e-10 {
            continue;
        }
        let v_direction = v.scale(1.0 / v_length);
        // An OBTUSE triangle puts the apex's u coordinate outside [0, |ab|];
        // evaluation clamps to the surface domain, so the plane must span the
        // triangle's full uv bounding box or the apex edges' pcurves map off
        // the triangle (sphere tessellations expose this; boxes never do).
        let u_low = c_u.min(0.0);
        let u_high = c_u.max(u_length);
        let surface = make_plane(
            a.add(u_direction.scale(u_low)),
            u_direction,
            v_direction,
            u_high - u_low,
            v_length,
        )?;
        let uv = [
            Vec3::new(-u_low, 0.0, 0.0),
            Vec3::new(u_length - u_low, 0.0, 0.0),
            Vec3::new(c_u - u_low, v_length, 0.0),
        ];
        let mut coedges = Vec::with_capacity(3);
        for side in 0..3 {
            let first = triangle[side];
            let second = triangle[(side + 1) % 3];
            let key = edge_key(first, second);
            let edge_id = if let Some(id) = edge_ids.get(&key) {
                *id
            } else {
                let id = next_id;
                next_id += 1;
                edges.push(EdgeRecord {
                    id,
                    curve: make_line(points[key.0], points[key.1])?,
                    t0: 0.0,
                    t1: 1.0,
                    start_vertex_id: key.0 as u64 + 1,
                    end_vertex_id: key.1 as u64 + 1,
                    degenerate: false,
                    name: None,
                });
                edge_ids.insert(key, id);
                id
            };
            coedges.push(CoedgeRecord {
                id: next_id,
                edge_id,
                forward: first < second,
                pcurve: make_line(uv[side], uv[(side + 1) % 3])?,
            });
            next_id += 1;
        }
        let loop_record = LoopRecord {
            id: next_id,
            coedges,
        };
        next_id += 1;
        faces.push(FaceRecord {
            id: next_id,
            surface,
            same_sense: true,
            loops: vec![loop_record],
            name: None,
        });
        next_id += 1;
    }
    let mut result = BrepSolid {
        id: next_id + 1,
        vertices,
        edges,
        shells: vec![ShellRecord { id: next_id, faces }],
        genus: 0,
    };
    let vertex_count = result.vertices.len() as i64;
    let edge_count = result.edges.len() as i64;
    let face_count = result.shells[0].faces.len() as i64;
    let numerator = 2 - (vertex_count - edge_count + face_count);
    if numerator >= 0 && numerator % 2 == 0 {
        result.genus = numerator / 2;
    }
    let issues = result.validate();
    if !issues.is_empty() {
        return Err(format!(
            "faceted shell repair produced invalid topology: {issues:?}"
        ));
    }
    for (index, face) in result.shells[0].faces.iter().enumerate() {
        tessellate_face(face, TessellationOptions::default(), index as u32)
            .map_err(|error| format!("faceted face {index} cannot be tessellated: {error}"))?;
    }
    Ok(result)
}

// BREP private tests: 09be33c0d41cac3c
