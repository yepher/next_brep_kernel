use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, VertexRecord};
use crate::{fit, NurbsCurve, NurbsSurface, Vec3, Vec4};
use super::stations::*;
use super::corner::*;

mod closed;
mod keep;
mod open;
mod support;

pub use closed::{blend_closed_edge, blend_edge_variable};
pub use open::blend_open_edge;
pub(super) use keep::{blend_closed_edge_keep, fit_preserved_pcurve, KeepStation};
pub(super) use support::{
    boundary_edge_at_vertex, detect_row_coincidence, end_face_id, spoke_crossing, support_crossing,
    support_crossing_uv, transverse_curve, trim_edge_at, CapEnd, CornerEnd, EndPlan, EndSurgery,
    MiterEnd, RimResolution, RowSew, SewnStripe, SpokeCrossings,
};
pub(super) use support::{locate_end_corner, resolve_free_end, splice_end_corner, RIM_MARGIN};
pub(super) use open::{build_open_surgery, fresh_id_source, prune_orphan_vertices};
pub(super) use support::{fit_open_rows, march_open_stations, vertex_stations};

use open::blend_open_edge_impl;
