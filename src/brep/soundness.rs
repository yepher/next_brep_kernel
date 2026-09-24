//! # Structural soundness — the two checks `validate()` deliberately does NOT make
//!
//! [`BrepSolid::validate`](crate::BrepSolid::validate) is an
//! **incidence** test: ids resolve, loops close, an edge is used twice with
//! opposite senses, a pcurve tracks its edge, Euler closes. A great deal of
//! code reads "validate passed" as exactly that claim, so this module does not
//! widen it. What `validate()` never asks:
//!
//! * **Is the thing connected?** Nothing in the incidence test notices that a
//!   shell's faces fall into two mutually unreachable pieces. Every local
//!   incidence can be perfect while the "solid" is two solids in one record.
//! * **Does it pass through itself?** `validate()`'s one string carrying the
//!   words "self-intersects" is a *uv-wire* warning — one loop's sampled
//!   pcurve polygon crossing itself in parameter space. It is not, and never
//!   was, a face-versus-face test.
//!
//! Both live here as SEPARATE detectors, following the precedent set by
//! `csg/oracle.rs`: an independent wrongness detector that a caller asks for,
//! never a silent widening of an existing verdict. Nothing in the kernel's own
//! build path calls either of them; what a CALLER does with the verdict — gate
//! or record — is the table below, and it is the case gate that acts on it.
//!
//! ## Gate or diagnostic — the standing decision
//!
//! | check | cost | verdict |
//! |---|---|---|
//! | [`solid_connectivity`] — per-shell edge-connected components, cross-shell edges | pure topology, one pass over the coedges, no geometry | **hard gate.** A split shell is never acceptable and the test cannot be noisy: it reads ids, not numbers. [`ConnectivityReport::is_disconnected`] is the gate predicate. |
//! | `ConnectivityReport::pinch_vertices` — vertices whose incident faces form more than one fan | same pass | **diagnostic.** A pinch is a real non-manifold defect, but pole/seam vertices are false-positive bait, so it is reported beside the gate rather than inside it. Vertices with an incident degenerate edge are skipped outright and counted in `pinch_skipped`. |
//! | [`solid_self_intersections`] — face-versus-face crossing | tessellates the solid and walks a triangle BVH | **gate for NEW occurrences.** [`SelfIntersectionReport::is_flagged`] is the predicate. It shipped opt-in because it costs a full tessellation plus a BVH sweep; the decision was reversed once the first corpus run found a case labelled `correct` whose faces cross four orders of magnitude past the band, because a gate that fires only when someone remembers a flag is a diagnostic with extra steps. The case gate now runs it by default, reports a crossing its baseline already holds as a note, and fails a crossing on any other face pair. |
//!
//! ## How the self-intersection test avoids being noisy
//!
//! A mesh-only test flags every tangent neighbour: two faces meeting G1 along
//! a shared edge chord-cross each other one triangle row in, and a fillet
//! corpus is nothing but tangent neighbours. Two rules keep this one quiet:
//!
//! 1. **Welded-vertex skip.** The watertight tessellator pins shared-edge
//!    samples to bit-identical positions, so triangles that meet along a
//!    shared edge share a welded vertex and are never compared.
//! 2. **Exact-surface confirmation.** A mesh crossing is only a *candidate*.
//!    It is confirmed by projecting each triangle's corners onto the OTHER
//!    face's carrier surface and requiring the signed distances to straddle
//!    zero by more than [`SelfIntersectionOptions::straddle_band`]. Every
//!    tessellation vertex lies exactly on its own surface, so chord error
//!    cannot manufacture a straddle: a tangent neighbour's corners all sit on
//!    one side of the other carrier, while a genuine piercing has corners on
//!    both. The confirmation runs only on BVH+tri-tri survivors, so it is
//!    paid a handful of times per solid, not per triangle.
//!
//! ## Known blind spot
//!
//! Only DISTINCT face pairs are confirmed. When both triangles lie on one
//! face, the straddle test is degenerate by construction — every corner of
//! both triangles is on that carrier at distance zero — so a single face
//! folding through itself is counted in
//! [`SelfIntersectionReport::same_face_candidates`] and reported, never
//! confirmed. The parameter-space side of that defect is what `validate()`'s
//! uv-wire warning already looks at.

use crate::projection::project_point_to_surface;
use crate::spatial::{Aabb, Bvh};
use crate::topology::BrepSolid;
use crate::{NurbsSurface, Vec3};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Disjoint set — shared by the shell and the vertex-fan scans
// ---------------------------------------------------------------------------

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
        }
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    fn union(&mut self, first: usize, second: usize) {
        let (a, b) = (self.find(first), self.find(second));
        if a != b {
            self.parent[b] = a;
        }
    }
}

// ---------------------------------------------------------------------------
// Connectivity
// ---------------------------------------------------------------------------

/// One shell's connectivity: how many edge-connected pieces its faces form.
#[derive(Clone, Debug, Serialize)]
pub struct ShellConnectivity {
    pub shell_id: u64,
    pub faces: usize,
    /// Edge-connected components of this shell's faces. Anything but 1 means
    /// the record holds several surfaces under one shell.
    pub components: usize,
    /// Face ids of every component EXCEPT the largest, smallest first — the
    /// pieces that would each have had to be their own shell. Empty when
    /// `components == 1`.
    pub detached: Vec<Vec<u64>>,
}

/// Outcome of [`solid_connectivity`].
#[derive(Clone, Debug, Serialize)]
pub struct ConnectivityReport {
    pub shells: Vec<ShellConnectivity>,
    /// Edges used by faces belonging to two DIFFERENT shells. A shell is a
    /// closed surface in its own right; sharing an edge across shells means
    /// the shell split is fiction.
    pub cross_shell_edges: Vec<u64>,
    /// Vertices where the incident face corners form more than one fan — two
    /// pieces of surface touching at a point and nowhere else. Edge
    /// connectivity cannot see this: both pieces are edge-connected, just not
    /// through this vertex. Diagnostic, not part of [`Self::is_disconnected`].
    pub pinch_vertices: Vec<u64>,
    /// Vertices the fan scan skipped because a degenerate (pole) edge meets
    /// there — a collapsed parameter boundary is not an ordinary fan and the
    /// scan would read it as a pinch.
    pub pinch_skipped: usize,
}

impl ConnectivityReport {
    /// The HARD-GATE predicate: the record claims to be one closed surface per
    /// shell and it is not. Pinch vertices are deliberately excluded — see the
    /// module table.
    pub fn is_disconnected(&self) -> bool {
        self.shells.iter().any(|shell| shell.components > 1)
            || !self.cross_shell_edges.is_empty()
    }

    /// One line naming what is wrong, or `None` when the gate predicate is
    /// clear. Pinch vertices are appended when present so a caller printing
    /// the summary still sees them.
    pub fn summary(&self) -> Option<String> {
        let mut parts = Vec::new();
        for shell in &self.shells {
            if shell.components > 1 {
                let sizes: Vec<usize> = shell.detached.iter().map(Vec::len).collect();
                parts.push(format!(
                    "shell {} splits into {} edge-connected components ({} face(s), detached sizes {sizes:?}, first detached faces {:?})",
                    shell.shell_id,
                    shell.components,
                    shell.faces,
                    shell.detached.first().map(Vec::as_slice).unwrap_or(&[]),
                ));
            }
        }
        if !self.cross_shell_edges.is_empty() {
            parts.push(format!(
                "{} edge(s) shared across shells ({:?})",
                self.cross_shell_edges.len(),
                &self.cross_shell_edges[..self.cross_shell_edges.len().min(8)]
            ));
        }
        if !self.pinch_vertices.is_empty() {
            parts.push(format!(
                "{} pinch vertex/vertices ({:?})",
                self.pinch_vertices.len(),
                &self.pinch_vertices[..self.pinch_vertices.len().min(8)]
            ));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("; "))
        }
    }
}

/// Connectivity of a solid's topology: are each shell's faces reachable from
/// one another across shared edges, do any two shells share an edge, and does
/// any vertex pinch two otherwise-separate fans together.
///
/// Pure topology — ids and use counts, never a coordinate — so it is
/// tolerance-free and cannot be noisy. Degenerate (pole) edges DO count as
/// connections: a collapsed boundary shared by two faces genuinely joins them.
pub fn solid_connectivity(solid: &BrepSolid) -> ConnectivityReport {
    // Global face indexing over (shell, face) in record order.
    let mut face_shell: Vec<usize> = Vec::new();
    let mut face_ids: Vec<u64> = Vec::new();
    for (shell_index, shell) in solid.shells.iter().enumerate() {
        for face in &shell.faces {
            face_shell.push(shell_index);
            face_ids.push(face.id);
        }
    }

    // edge id -> the global face indices that use it.
    let mut edge_faces: HashMap<u64, Vec<usize>> = HashMap::default();
    let mut global_index = 0usize;
    for shell in &solid.shells {
        for face in &shell.faces {
            let _ = face;
            for loop_record in &face.loops {
                for coedge in &loop_record.coedges {
                    let users = edge_faces.entry(coedge.edge_id).or_default();
                    if users.last() != Some(&global_index) {
                        users.push(global_index);
                    }
                }
            }
            global_index += 1;
        }
    }

    let mut sets = DisjointSet::new(face_ids.len());
    let mut cross_shell_edges = Vec::new();
    for (edge_id, users) in &edge_faces {
        for pair in users.windows(2) {
            sets.union(pair[0], pair[1]);
        }
        let shells_touched: HashSet<usize> =
            users.iter().map(|index| face_shell[*index]).collect();
        if shells_touched.len() > 1 {
            cross_shell_edges.push(*edge_id);
        }
    }
    cross_shell_edges.sort_unstable();

    let mut shells = Vec::with_capacity(solid.shells.len());
    let mut base = 0usize;
    for shell in &solid.shells {
        let count = shell.faces.len();
        let mut buckets: HashMap<usize, Vec<u64>> = HashMap::default();
        for offset in 0..count {
            let root = sets.find(base + offset);
            buckets.entry(root).or_default().push(face_ids[base + offset]);
        }
        let mut groups: Vec<Vec<u64>> = buckets.into_values().collect();
        // Largest first so `detached` is "everything but the main piece", and
        // deterministic: ties break on the smallest face id.
        groups.sort_by(|a, b| b.len().cmp(&a.len()).then(a.first().cmp(&b.first())));
        let components = groups.len();
        let mut detached: Vec<Vec<u64>> = groups.into_iter().skip(1).collect();
        detached.sort_by_key(Vec::len);
        for group in &mut detached {
            group.sort_unstable();
        }
        shells.push(ShellConnectivity {
            shell_id: shell.id,
            faces: count,
            components,
            detached,
        });
        base += count;
    }

    let (pinch_vertices, pinch_skipped) = pinch_vertices(solid);

    ConnectivityReport {
        shells,
        cross_shell_edges,
        pinch_vertices,
        pinch_skipped,
    }
}

/// Vertices whose incident face corners form more than one fan.
///
/// A *corner* is one consecutive coedge pair of a loop meeting at the vertex.
/// Each non-degenerate edge at the vertex is used by exactly two coedges, each
/// belonging to exactly one corner there, so the edge links those two corners.
/// One fan = one component. Two cones touching at their apex are edge-
/// connected everywhere else and split into two fans exactly here.
///
/// Any vertex with an incident DEGENERATE edge is skipped: a collapsed pole
/// boundary is not an ordinary fan and would read as a pinch.
fn pinch_vertices(solid: &BrepSolid) -> (Vec<u64>, usize) {
    let mut degenerate_at: HashSet<u64> = HashSet::default();
    for edge in &solid.edges {
        if edge.degenerate {
            degenerate_at.insert(edge.start_vertex_id);
            degenerate_at.insert(edge.end_vertex_id);
        }
    }
    let edges: HashMap<u64, &crate::topology::EdgeRecord> =
        solid.edges.iter().map(|edge| (edge.id, edge)).collect();

    // vertex id -> corners; a corner is the pair of edge ids meeting there.
    let mut corners: HashMap<u64, Vec<[u64; 2]>> = HashMap::default();
    for shell in &solid.shells {
        for face in &shell.faces {
            for loop_record in &face.loops {
                let count = loop_record.coedges.len();
                if count == 0 {
                    continue;
                }
                for index in 0..count {
                    let current = &loop_record.coedges[index];
                    let next = &loop_record.coedges[(index + 1) % count];
                    let Some(current_edge) = edges.get(&current.edge_id) else {
                        continue;
                    };
                    let vertex = if current.forward {
                        current_edge.end_vertex_id
                    } else {
                        current_edge.start_vertex_id
                    };
                    corners
                        .entry(vertex)
                        .or_default()
                        .push([current.edge_id, next.edge_id]);
                }
            }
        }
    }

    let mut pinched = Vec::new();
    let mut skipped = 0usize;
    for (vertex, list) in &corners {
        if degenerate_at.contains(vertex) {
            skipped += 1;
            continue;
        }
        if list.len() < 2 {
            continue;
        }
        let mut sets = DisjointSet::new(list.len());
        let mut by_edge: HashMap<u64, usize> = HashMap::default();
        for (index, corner) in list.iter().enumerate() {
            for edge_id in corner {
                match by_edge.entry(*edge_id) {
                    std::collections::hash_map::Entry::Occupied(slot) => {
                        sets.union(*slot.get(), index);
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(index);
                    }
                }
            }
        }
        let fans: HashSet<usize> = (0..list.len()).map(|index| sets.find(index)).collect();
        if fans.len() > 1 {
            pinched.push(*vertex);
        }
    }
    pinched.sort_unstable();
    (pinched, skipped)
}

// ---------------------------------------------------------------------------
// Face-versus-face self-intersection
// ---------------------------------------------------------------------------

/// Fraction of the solid's bounding diagonal a confirmed crossing must
/// straddle the other carrier by. Tessellation vertices are exact on their own
/// surface, so this band only has to clear projection/Newton residue, not
/// chord error; it is deliberately far below any real penetration.
const STRADDLE_FRACTION: f64 = 1e-6;

/// Fraction of the diagonal a mesh crossing segment must be longer than. A
/// crossing shorter than this is a corner graze, not a penetration.
const CROSSING_FRACTION: f64 = 1e-7;

/// Fraction of the diagonal used to weld coincident tessellation vertices.
/// Shared-edge samples are pinned to identical values by the watertight
/// tessellator, so this only has to survive being read back out of the mesh
/// buffers.
const WELD_FRACTION: f64 = 1e-9;

/// Knobs for [`solid_self_intersections`]. [`SelfIntersectionOptions::for_solid`]
/// derives every one from the solid's own size.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SelfIntersectionOptions {
    /// Tessellation density. The detector resolves nothing finer than this.
    pub chord_tolerance: f64,
    /// Signed-distance margin a confirmed crossing must straddle the other
    /// carrier surface by, on BOTH sides.
    pub straddle_band: f64,
    /// Shortest mesh crossing segment that counts as a candidate.
    pub min_crossing_length: f64,
    /// Coincidence radius for welding tessellation vertices.
    pub weld: f64,
    /// Refuse to scan a mesh larger than this (the report comes back
    /// `truncated`), so a pathological solid cannot hang a gate.
    pub max_triangles: usize,
}

impl SelfIntersectionOptions {
    /// Every band derived from the solid's own extent: the display chord
    /// tolerance the app meshes at, and diagonal-relative bands.
    pub fn for_solid(solid: &BrepSolid) -> Self {
        let diagonal = solid_diagonal(solid).max(1e-9);
        Self {
            chord_tolerance: crate::display_chord_tolerance(solid, 1.0),
            straddle_band: diagonal * STRADDLE_FRACTION,
            min_crossing_length: diagonal * CROSSING_FRACTION,
            weld: diagonal * WELD_FRACTION,
            max_triangles: 400_000,
        }
    }
}

/// One confirmed face-versus-face crossing.
#[derive(Clone, Debug, Serialize)]
pub struct FaceCrossing {
    pub face_a: u64,
    pub face_b: u64,
    pub name_a: Option<String>,
    pub name_b: Option<String>,
    /// Midpoint of the mesh crossing segment.
    pub point: Vec3,
    /// Length of the mesh crossing segment.
    pub crossing_length: f64,
    /// How far the confirmation straddled the other carrier: the smaller of
    /// the two one-sided margins, over both directions of the pair.
    pub straddle: f64,
}

/// Outcome of [`solid_self_intersections`].
#[derive(Clone, Debug, Serialize)]
pub struct SelfIntersectionReport {
    pub triangles: usize,
    /// Triangle pairs whose boxes overlapped and that survived the welded-
    /// vertex skip — the work the exact confirmation was offered.
    pub candidate_pairs: usize,
    /// Mesh crossings between DISTINCT faces that the exact-surface straddle
    /// test confirmed, one entry per face pair (deepest kept).
    pub confirmed: Vec<FaceCrossing>,
    /// Mesh crossings between distinct faces the straddle test REJECTED —
    /// tangent neighbours and chord error. A large number here beside zero
    /// confirmations is the detector working, not failing.
    pub rejected: usize,
    /// Mesh crossings whose confirmation could not be evaluated (projection or
    /// normal failed). Never counted as either verdict.
    pub undecided: usize,
    /// Mesh crossings between two triangles of ONE face — the documented blind
    /// spot; reported, never confirmed.
    pub same_face_candidates: usize,
    /// The mesh exceeded `max_triangles` and was not scanned.
    pub truncated: bool,
    pub options: SelfIntersectionOptions,
}

impl SelfIntersectionReport {
    /// The predicate for a caller that has opted into this check: the solid
    /// passes through itself.
    pub fn is_flagged(&self) -> bool {
        !self.confirmed.is_empty()
    }

    /// One line naming the worst crossing, or `None` when nothing was
    /// confirmed.
    pub fn summary(&self) -> Option<String> {
        let worst = self
            .confirmed
            .iter()
            .max_by(|a, b| a.straddle.total_cmp(&b.straddle))?;
        Some(format!(
            "{} face pair(s) self-intersect; worst faces {} and {} near ({:.4}, {:.4}, {:.4}) (crossing {:.3e}, straddle {:.3e})",
            self.confirmed.len(),
            worst.face_a,
            worst.face_b,
            worst.point.x,
            worst.point.y,
            worst.point.z,
            worst.crossing_length,
            worst.straddle
        ))
    }
}

fn solid_diagonal(solid: &BrepSolid) -> f64 {
    let mut bounds = Aabb::empty();
    for vertex in &solid.vertices {
        bounds.include_point(vertex.point);
    }
    let diagonal = bounds.diagonal();
    if diagonal.is_finite() && diagonal > 0.0 {
        return diagonal;
    }
    // Vertex-free solids (a full sphere, a full torus) fall back to the
    // control hull, exactly as the display tolerance does.
    let mut hull = Aabb::empty();
    for shell in &solid.shells {
        for face in &shell.faces {
            if let Ok(box_of) = Aabb::from_surface_controls(&face.surface) {
                hull.include(box_of);
            }
        }
    }
    let diagonal = hull.diagonal();
    if diagonal.is_finite() {
        diagonal
    } else {
        0.0
    }
}

/// Does this solid pass through itself? Tessellates every face once, walks a
/// triangle BVH for mesh crossings, and confirms each candidate against the
/// two carrier SURFACES so tangent neighbours and chord error cannot
/// manufacture a finding. See the module docs for the design and its one
/// documented blind spot.
///
/// Nothing in the kernel's build path calls this — a caller asks for it and
/// reads [`SelfIntersectionReport::is_flagged`]. The case gate is that caller
/// and treats a flag as a failure unless its baseline already holds the same
/// face pair; see the module table.
pub fn solid_self_intersections(
    solid: &BrepSolid,
    options: SelfIntersectionOptions,
) -> Result<SelfIntersectionReport, String> {
    let mut report = SelfIntersectionReport {
        triangles: 0,
        candidate_pairs: 0,
        confirmed: Vec::new(),
        rejected: 0,
        undecided: 0,
        same_face_candidates: 0,
        truncated: false,
        options,
    };
    if !(options.chord_tolerance > 0.0) || !options.chord_tolerance.is_finite() {
        return Err("solid_self_intersections: chord tolerance must be positive".into());
    }

    // The face-stride entry point rather than `tessellate_brep_watertight`:
    // the latter runs a coherent-orientation pass and a mesh validation that
    // can fail on exactly the broken solids this detector exists to inspect,
    // and winding is irrelevant to a crossing test.
    let mesh = crate::watertight_tessellation::tessellate_brep_watertight_face_stride(
        solid,
        options.chord_tolerance,
        1,
        0,
    )?;
    let triangle_count = mesh.indices.len() / 3;
    report.triangles = triangle_count;
    if triangle_count == 0 {
        return Ok(report);
    }
    if triangle_count > options.max_triangles {
        report.truncated = true;
        return Ok(report);
    }

    // Faces in the tessellator's sequential order, so a triangle's `face_ids`
    // entry indexes straight into this.
    let faces: Vec<&crate::topology::FaceRecord> = solid
        .shells
        .iter()
        .flat_map(|shell| shell.faces.iter())
        .collect();

    let point_of = |index: u32| -> Vec3 {
        let base = index as usize * 3;
        Vec3::new(
            mesh.positions[base],
            mesh.positions[base + 1],
            mesh.positions[base + 2],
        )
    };

    let welded = weld_positions(&mesh.positions, options.weld);

    let mut boxes = Vec::with_capacity(triangle_count);
    let mut corners: Vec<[Vec3; 3]> = Vec::with_capacity(triangle_count);
    for triangle in 0..triangle_count {
        let indices = [
            mesh.indices[triangle * 3],
            mesh.indices[triangle * 3 + 1],
            mesh.indices[triangle * 3 + 2],
        ];
        let points = [
            point_of(indices[0]),
            point_of(indices[1]),
            point_of(indices[2]),
        ];
        boxes.push(Aabb::from_points(points));
        corners.push(points);
    }
    let bvh = Bvh::build(&boxes);

    // Deepest confirmed crossing per unordered face pair.
    let mut best: HashMap<(u64, u64), FaceCrossing> = HashMap::default();
    let mut hits = Vec::new();
    for first in 0..triangle_count {
        hits.clear();
        bvh.overlapping(boxes[first], 0.0, &mut hits);
        for second in hits.iter().copied() {
            if second <= first {
                continue;
            }
            let shares_vertex = (0..3).any(|a| {
                let wa = welded[mesh.indices[first * 3 + a] as usize];
                (0..3).any(|b| wa == welded[mesh.indices[second * 3 + b] as usize])
            });
            if shares_vertex {
                continue;
            }
            report.candidate_pairs += 1;
            let Some((start, end)) = triangle_crossing(&corners[first], &corners[second]) else {
                continue;
            };
            let length = end.sub(start).length();
            if length <= options.min_crossing_length {
                continue;
            }
            let face_first = mesh.face_ids[first] as usize;
            let face_second = mesh.face_ids[second] as usize;
            if face_first == face_second {
                report.same_face_candidates += 1;
                continue;
            }
            let (Some(face_a), Some(face_b)) = (faces.get(face_first), faces.get(face_second))
            else {
                report.undecided += 1;
                continue;
            };
            let forward = straddle(&face_b.surface, &corners[first], options.straddle_band);
            let backward = straddle(&face_a.surface, &corners[second], options.straddle_band);
            let (Some(forward), Some(backward)) = (forward, backward) else {
                report.undecided += 1;
                continue;
            };
            if forward <= options.straddle_band || backward <= options.straddle_band {
                report.rejected += 1;
                continue;
            }
            let key = if face_a.id <= face_b.id {
                (face_a.id, face_b.id)
            } else {
                (face_b.id, face_a.id)
            };
            let crossing = FaceCrossing {
                face_a: key.0,
                face_b: key.1,
                name_a: if face_a.id <= face_b.id {
                    face_a.name.clone()
                } else {
                    face_b.name.clone()
                },
                name_b: if face_a.id <= face_b.id {
                    face_b.name.clone()
                } else {
                    face_a.name.clone()
                },
                point: start.add(end).scale(0.5),
                crossing_length: length,
                straddle: forward.min(backward),
            };
            let slot = best.entry(key).or_insert_with(|| crossing.clone());
            if crossing.straddle > slot.straddle {
                *slot = crossing;
            }
        }
    }

    report.confirmed = best.into_values().collect();
    report
        .confirmed
        .sort_by(|a, b| b.straddle.total_cmp(&a.straddle).then(a.face_a.cmp(&b.face_a)));
    Ok(report)
}

/// Map every mesh vertex to a representative index, welding positions that
/// coincide within `weld`. Bucketed on a `weld`-sized lattice with the 27
/// neighbouring cells consulted, so a pair straddling a cell boundary still
/// welds.
fn weld_positions(positions: &[f64], weld: f64) -> Vec<u32> {
    let count = positions.len() / 3;
    let mut representative = vec![0u32; count];
    let cell = weld.max(f64::MIN_POSITIVE);
    let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::default();
    for index in 0..count {
        let base = index * 3;
        let point = Vec3::new(positions[base], positions[base + 1], positions[base + 2]);
        let key = (
            (point.x / cell).floor() as i64,
            (point.y / cell).floor() as i64,
            (point.z / cell).floor() as i64,
        );
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = buckets.get(&(key.0 + dx, key.1 + dy, key.2 + dz)) else {
                        continue;
                    };
                    for candidate in list {
                        let base = *candidate as usize * 3;
                        let other = Vec3::new(
                            positions[base],
                            positions[base + 1],
                            positions[base + 2],
                        );
                        if other.sub(point).length() <= weld {
                            found = Some(representative[*candidate as usize]);
                            break 'search;
                        }
                    }
                }
            }
        }
        let value = found.unwrap_or(index as u32);
        representative[index] = value;
        buckets.entry(key).or_default().push(index as u32);
    }
    representative
}

/// Signed distances of a triangle's corners to `surface`, reduced to the
/// smaller of the two one-sided margins: positive means the corners genuinely
/// sit on both sides of the carrier by at least that much, zero means they do
/// not straddle it at all. `None` when a projection or a normal could not be
/// evaluated — never a verdict.
///
/// Three projections, and only for a triangle pair that already survived the
/// BVH, the welded-vertex skip and the mesh crossing test.
fn straddle(surface: &NurbsSurface, triangle: &[Vec3; 3], band: f64) -> Option<f64> {
    let mut above = 0.0f64;
    let mut below = 0.0f64;
    for corner in triangle {
        let projection = project_point_to_surface(surface, *corner).ok()?;
        if projection.distance <= band {
            continue;
        }
        let normal = surface.normal(projection.u, projection.v).ok()?;
        let normal = normal.normalized().ok()?;
        let signed = corner.sub(projection.point).dot(normal);
        if signed > 0.0 {
            above = above.max(signed);
        } else {
            below = below.max(-signed);
        }
    }
    Some(above.min(below))
}

/// The segment along which two triangles cross, or `None` when they do not
/// cross transversally. Coplanar pairs return `None` — a coplanar overlap is a
/// different defect and this detector does not claim it.
fn triangle_crossing(first: &[Vec3; 3], second: &[Vec3; 3]) -> Option<(Vec3, Vec3)> {
    let plane_first = triangle_plane(first)?;
    let plane_second = triangle_plane(second)?;
    let direction = plane_first.0.cross(plane_second.0);
    let length = direction.length();
    if length <= 1e-12 {
        return None; // parallel or coplanar
    }
    let direction = direction.scale(1.0 / length);

    let first_span = plane_span(first, plane_second, direction)?;
    let second_span = plane_span(second, plane_first, direction)?;
    let low = first_span.0.max(second_span.0);
    let high = first_span.1.min(second_span.1);
    if high < low {
        return None;
    }
    // Rebuild 3D points by interpolating the first triangle's own crossing
    // chord, so the returned segment is on the mesh rather than reconstructed
    // from a line equation.
    let span = first_span.1 - first_span.0;
    let lerp = |value: f64| {
        if span <= 0.0 {
            first_span.2
        } else {
            let fraction = ((value - first_span.0) / span).clamp(0.0, 1.0);
            first_span
                .2
                .add(first_span.3.sub(first_span.2).scale(fraction))
        }
    };
    Some((lerp(low), lerp(high)))
}

/// Unit normal and plane offset of a triangle, `None` for a degenerate one.
fn triangle_plane(triangle: &[Vec3; 3]) -> Option<(Vec3, f64)> {
    let normal = triangle[1]
        .sub(triangle[0])
        .cross(triangle[2].sub(triangle[0]));
    let length = normal.length();
    if length <= 1e-18 {
        return None;
    }
    let normal = normal.scale(1.0 / length);
    Some((normal, -normal.dot(triangle[0])))
}

/// Where `triangle` crosses `plane`, as `(low, high, low_point, high_point)`
/// parametrized by `point · direction`. `None` when the triangle does not
/// reach the plane.
fn plane_span(
    triangle: &[Vec3; 3],
    plane: (Vec3, f64),
    direction: Vec3,
) -> Option<(f64, f64, Vec3, Vec3)> {
    let distance = |point: Vec3| plane.0.dot(point) + plane.1;
    let distances = [
        distance(triangle[0]),
        distance(triangle[1]),
        distance(triangle[2]),
    ];
    if distances.iter().all(|value| *value > 0.0) || distances.iter().all(|value| *value < 0.0) {
        return None;
    }
    let mut points: Vec<Vec3> = Vec::with_capacity(3);
    for index in 0..3 {
        let next = (index + 1) % 3;
        if distances[index] == 0.0 {
            points.push(triangle[index]);
        }
        if (distances[index] < 0.0) != (distances[next] < 0.0)
            && distances[index] != 0.0
            && distances[next] != 0.0
        {
            let fraction = distances[index] / (distances[index] - distances[next]);
            points.push(
                triangle[index].add(triangle[next].sub(triangle[index]).scale(fraction)),
            );
        }
    }
    if points.len() < 2 {
        return None;
    }
    let mut low = (f64::INFINITY, Vec3::default());
    let mut high = (f64::NEG_INFINITY, Vec3::default());
    for point in points {
        let value = point.dot(direction);
        if value < low.0 {
            low = (value, point);
        }
        if value > high.0 {
            high = (value, point);
        }
    }
    Some((low.0, high.0, low.1, high.1))
}

// BREP private tests: ba1a0a116d86088e
