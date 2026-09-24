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

mod chain;
mod corner;
mod edge;
mod miter;
mod network;
mod stations;
mod rows;

// BREP private tests: fb8e60d6a94ef81c

pub use chain::{blend_smooth_chain, blend_smooth_chain_if_closed};
pub(crate) use corner::round_concave_chain_corner;
pub use corner::round_convex_corner;
pub use edge::{blend_closed_edge, blend_edge_variable, blend_open_edge};
pub(crate) use network::{blend_star_network, mixed_convexity_corner, MixedCorner};
pub(crate) use stations::BALL_OFF_CARRIER;
