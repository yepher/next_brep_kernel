use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{
    boolean_operation, build_pcurve_on_surface, interpolate_curve, loft_profile_brep, make_arc,
    make_extrusion, make_line, make_plane, project_point_to_curve, BooleanOperation,
    BooleanOptions, NurbsCurve, NurbsSurface, Vec3, Vec4,
};

#[path = "sweep_topology/extrude.rs"]
mod extrude;
#[path = "sweep_topology/sweep.rs"]
mod sweep;
#[path = "sweep_topology/draft.rs"]
mod draft;
#[path = "sweep_topology/rib.rs"]
mod rib;
// BREP private tests: d406958354a473b5

pub use extrude::extrude_profile_brep;
pub(crate) use crate::curve::{curve_to_plane_parameters, parameter_line};
pub(crate) use extrude::profile_area;

pub use sweep::{
    fit_helix_curve, helix_sample_points, profile_anchor, sweep_profile_along_chain,
    sweep_profile_along_chain_with_stations,
    sweep_profile_along_path, sweep_profile_along_path_anchored,
    sweep_profile_helix, sweep_profile_twisted, sweep_profile_twisted_anchored, ProfileAnchor,
    SectionPlacement,
};

pub use draft::extrude_profile_brep_draft;

pub use rib::{rib_from_profile, RibExtrusion};
use rib::{
    arc_window_subrange, classify_profile_segments, junction_edge_curve, offset_junction,
    ruled_between, SegGeom,
};
