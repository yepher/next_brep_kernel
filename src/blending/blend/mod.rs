//! General rolling-ball blends (Golovanov §4.9 + §6.9 surgery).
//!
//! The blend surface is the rational surface (4.9.1) built from three
//! curves: the two support curves cr(t) ⊂ F1 and cs(t) ⊂ F2 where a
//! sphere of radius ρ touches both faces, and the middle curve at the
//! intersection of the two tangent planes, weighted w(t) = cos(α/2)
//! (4.9.3).  The tangency system (4.9.2) + the section-plane constraint
//! (4.9.5) — the sphere center pinned to the normal plane of the edge —
//! is solved per station by Newton with a finite-difference Jacobian.
//!
//! Topology is rebuilt the §6.9 way, with NO tool solid and NO boolean:
//! each mating face's loop swaps the blended edge's coedge for a coedge
//! on its support curve (trimming the adjacent seam edge when the face
//! is a closed carrier with seam-in-one-loop topology), and the blend
//! face is inserted with its own seam.  Open edges take the same route
//! (`edge/open.rs`), and a whole SELECTION goes through `network.rs`, which
//! marches every stripe against the original solid and solves each shared
//! corner from its ball before anything is cut.

mod carve;
mod chain;
mod collapse;
mod corner;
mod edge;
mod fold;
mod miter;
mod network;
mod planar_chart;
mod restrict;
mod runout;
mod stations;
mod rows;
mod track_fit;


pub use chain::{blend_smooth_chain, blend_smooth_chain_if_closed};
pub(crate) use corner::round_concave_chain_corner;
pub use corner::round_convex_corner;
pub use edge::{blend_closed_edge, blend_edge_variable, blend_open_edge};
pub(crate) use collapse::FULL_WIDTH_COLLAPSE_UNSOUND;
pub(crate) use planar_chart::{
    fit_planar_charts_to_trims, PLANAR_CHART_EDGE_OFF_PLANE, PLANAR_CHART_WIDEN_UNSOUND,
};
pub(crate) use track_fit::PCURVE_OFF_FLOOR;
pub(crate) use edge::{consumed_band, CONSUMED_SNAP_UNSOUND, RAIL_COLLAPSE_UNSUPPORTED};
pub(crate) use network::{
    blend_star_network, degenerate_corner_setbacks, mixed_convexity_corner, DegenerateSetback,
    MixedCorner,
};
pub(crate) use miter::MARCHED_FIT_OFF_CARRIERS;
pub(crate) use runout::{plan_runout, RunoutPlan};
pub(crate) use fold::{is_wall_fold, WALL_FOLDS};
pub(crate) use stations::BALL_OFF_CARRIER;
