use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, VertexRecord};
use crate::{fit, NurbsCurve, NurbsSurface, Vec3, Vec4};
use super::stations::*;
use super::edge::*;

// ====================================================================
// Smooth-edge chains (§6.9.5): conjugated edges blended as ONE unit
// ====================================================================

mod closed;
mod closed_surgery;
mod collect;
mod march;
mod open;

pub use closed::{blend_smooth_chain, blend_smooth_chain_if_closed};

use closed::{cross_edge_at, pcurve_portion, project_piece_pcurve, ChainRows, RimPiece};
use closed_surgery::chain_surgery;
use collect::{collect_smooth_chain, ChainSegment, SmoothChain};
use march::{march_chain, ChainSample, CHAIN_PER_SEGMENT};
use open::blend_open_smooth_chain;
