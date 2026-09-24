use crate::sweep_topology::{curve_to_plane_parameters, parameter_line, profile_area};
use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{interpolate_curve, make_plane, NurbsCurve, NurbsSurface, Vec3, Vec4};

#[path = "loft_topology/basic.rs"]
mod basic;
#[path = "loft_topology/guided.rs"]
mod guided;
#[path = "loft_topology/closed.rs"]
mod closed;
// BREP private tests: fe4de28da8f21108
// BREP private tests: b53b1db70d85dddc
// BREP private tests: 9955026f9a85e303

pub use basic::{loft_profile_brep, loft_profile_brep_tangent};
use basic::closed_points;

pub use closed::loft_profile_brep_closed;

pub use guided::{loft_profile_brep_guided, loft_profile_brep_guided_frame};

#[path = "loft_topology/validation.rs"]
mod validation;
use validation::validate_sections;
