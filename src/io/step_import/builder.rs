use super::*;

// Resolver coordinates are already in millimetres. Identity repair uses the
// existing absolute floor: neither world position nor unrelated large features
// justify merging separate vertices or collapsing an edge. Widening this band
// requires bounded, measured representation error (not a model-size heuristic).
const IMPORT_IDENTITY_TOLERANCE: f64 = 1e-7;

struct SolidBuilder<'a> {
    resolver: &'a Resolver<'a>,
    next_id: u64,
    vertices: Vec<VertexRecord>,
    edges: Vec<EdgeRecord>,
    vertex_of_ref: HashMap<usize, u64>,
    edge_of_ref: HashMap<usize, u64>,
    /// FACETED_BREP welding: an implicit polygon edge is shared by the two faces
    /// that reference the same consecutive point pair, keyed by unordered vertex
    /// ids so the two coedges reuse (and oppositely orient) one `EdgeRecord`.
    edge_of_vertex_pair: HashMap<(u64, u64), u64>,
    /// id -> index into `vertices` / `edges`, kept in lock-step with every push
    /// (via `push_vertex`/`push_edge`) and the single `split_edge_at` insert
    /// (via `rebuild_edge_index`). `vertex_point`/`edge_record` were O(N) linear
    /// scans called per coedge; these make them O(1) and yield the same record
    /// (records are only appended and mutated in place, never reordered).
    vertex_index: HashMap<u64, usize>,
    edge_index: HashMap<u64, usize>,
}

/// A face resolved to its carrier surface and ordered coedge bounds, with
/// pcurve/loop assembly deferred so a global seam pre-pass can run in between.
struct PendingFace {
    surface: NurbsSurface,
    same_sense: bool,
    /// `(ordered (edge_id, forward) coedges, is_outer_bound)`, outer bounds first.
    bounds: Vec<(Vec<(u64, bool)>, bool)>,
}

mod collect;
mod seams;
mod rims;
mod loops;
mod solids;

pub(super) use solids::*;
