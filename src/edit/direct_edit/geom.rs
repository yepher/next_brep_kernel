use super::*;

pub(super) use crate::offset_retrim::{plane_of_surface, Plane, PARALLEL_EPS};

#[derive(Clone, Copy)]
pub(super) struct Line {
    pub(super) point: Vec3,
    pub(super) dir: Vec3,
}

pub(super) fn max_topology_id(solid: &BrepSolid) -> u64 {
    let mut maximum = solid.id;
    for vertex in &solid.vertices {
        maximum = maximum.max(vertex.id);
    }
    for edge in &solid.edges {
        maximum = maximum.max(edge.id);
    }
    for shell in &solid.shells {
        maximum = maximum.max(shell.id);
        for face in &shell.faces {
            maximum = maximum.max(face.id);
            for loop_record in &face.loops {
                maximum = maximum.max(loop_record.id);
                for coedge in &loop_record.coedges {
                    maximum = maximum.max(coedge.id);
                }
            }
        }
    }
    maximum
}

/// Apply a face's rebuilt intrinsic boundaries, then validate topology and
/// reject a volume-sign reversal when both volume calculations succeed.
pub(super) fn finish_intrinsic_face_offset(
    solid: &BrepSolid,
    face_location: (usize, usize),
    offset: NurbsSurface,
    rebuilt_edges: &HashMap<u64, (NurbsCurve, f64, f64)>,
    rebuilt_vertices: &HashMap<u64, Vec3>,
    operation: &str,
) -> Result<BrepSolid, String> {
    let (shell_index, face_index) = face_location;
    let mut result = solid.clone();
    result.shells[shell_index].faces[face_index].surface = offset;
    for edge in &mut result.edges {
        if let Some((curve, t0, t1)) = rebuilt_edges.get(&edge.id) {
            edge.curve = curve.clone();
            edge.t0 = *t0;
            edge.t1 = *t1;
        }
    }
    for vertex in &mut result.vertices {
        if let Some(point) = rebuilt_vertices.get(&vertex.id) {
            vertex.point = *point;
        }
    }
    let issues = result.validate();
    if !issues.is_empty() {
        return Err(format!(
            "{operation}: pushed solid failed validation: {issues:?}"
        ));
    }
    if let (Ok(before), Ok(after)) = (solid_signed_volume(solid), solid_signed_volume(&result)) {
        if before * after <= 0.0 {
            return Err(format!("{operation}: the push inverts the solid — refusing"));
        }
    }
    Ok(result)
}

pub(super) fn find_face(solid: &BrepSolid, face_id: u64) -> Option<(usize, usize)> {
    for (shell_index, shell) in solid.shells.iter().enumerate() {
        for (face_index, face) in shell.faces.iter().enumerate() {
            if face.id == face_id {
                return Some((shell_index, face_index));
            }
        }
    }
    None
}

/// The one face (other than `exclude`) that uses `edge_id`. Errs unless the
/// edge is shared by exactly one other face (manifold interior edge).
pub(super) fn other_face_of_edge(solid: &BrepSolid, edge_id: u64, exclude: u64) -> Result<u64, String> {
    let mut found: Option<u64> = None;
    for shell in &solid.shells {
        for face in &shell.faces {
            if face.id == exclude {
                continue;
            }
            let uses = face
                .loops
                .iter()
                .flat_map(|loop_record| &loop_record.coedges)
                .any(|coedge| coedge.edge_id == edge_id);
            if uses {
                if found.is_some_and(|other| other != face.id) {
                    return Err(format!(
                        "delete_face_and_heal: edge {edge_id} is shared by more than two faces"
                    ));
                }
                found = Some(face.id);
            }
        }
    }
    found.ok_or_else(|| {
        format!("delete_face_and_heal: boundary edge {edge_id} has no opposite face")
    })
}

pub(super) fn edge_point(solid: &BrepSolid, vertex_id: u64) -> Result<Vec3, String> {
    solid
        .vertices
        .iter()
        .find(|vertex| vertex.id == vertex_id)
        .map(|vertex| vertex.point)
        .ok_or_else(|| format!("delete_face_and_heal: missing vertex {vertex_id}"))
}

pub(super) fn intersect_planes(a: &Plane, b: &Plane) -> Option<Line> {
    let dir = a.normal.cross(b.normal);
    if dir.length() <= PARALLEL_EPS {
        return None;
    }
    let ca = a.normal.dot(a.origin);
    let cb = b.normal.dot(b.origin);
    let naa = a.normal.dot(a.normal);
    let nab = a.normal.dot(b.normal);
    let nbb = b.normal.dot(b.normal);
    let determinant = naa * nbb - nab * nab;
    if determinant.abs() <= PARALLEL_EPS {
        return None;
    }
    let alpha = (ca * nbb - cb * nab) / determinant;
    let beta = (cb * naa - ca * nab) / determinant;
    let point = a.normal.scale(alpha).add(b.normal.scale(beta));
    let dir = dir.normalized().ok()?;
    Some(Line { point, dir })
}

pub(super) fn intersect_line_plane(line: &Line, plane: &Plane) -> Option<Vec3> {
    let denominator = line.dir.dot(plane.normal);
    if denominator.abs() <= PARALLEL_EPS {
        return None;
    }
    let t = (plane.normal.dot(plane.origin) - plane.normal.dot(line.point)) / denominator;
    Some(line.point.add(line.dir.scale(t)))
}

type EndpointTarget = Option<(u64, Vec3)>;

/// Collect surviving edges affected by a vertex collapse before mutating them.
pub(super) fn pending_edge_relocations(
    solid: &BrepSolid,
    excluded: &HashSet<u64>,
    collapse: &HashMap<u64, u64>,
    vertex_points: &HashMap<u64, Vec3>,
) -> Vec<(usize, EndpointTarget, EndpointTarget)> {
    let mut pending = Vec::new();
    for (index, edge) in solid.edges.iter().enumerate() {
        if excluded.contains(&edge.id) {
            continue;
        }
        let start = collapse
            .get(&edge.start_vertex_id)
            .map(|vertex| (*vertex, vertex_points[vertex]));
        let end = collapse
            .get(&edge.end_vertex_id)
            .map(|vertex| (*vertex, vertex_points[vertex]));
        if start.is_some() || end.is_some() {
            pending.push((index, start, end));
        }
    }
    pending
}
