//! STEP Part 21 import for NURBS and analytic surfaces, assemblies, and PMI.
//!
//! The importer reconstructs and validates BREP topology. Analytic carriers are
//! converted to NURBS sized to the face bounds. Pcurves are derived by projecting
//! each edge onto its carrier; surface-curve wrappers contribute their 3D curve.
//! Seam edges occupy opposite parameter boundaries, and collapsed pole bounds
//! become degenerate edges so faces close and integrate correctly.

use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{
    build_pcurve_on_surface_range, make_arc, make_extrusion, make_hyperbola, make_line,
    make_parabola, make_plane, make_revolution, offset_surface, transform_brep, AffineTransform,
    NurbsCurve, NurbsSurface, Vec3, Vec4,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;

const TAU: f64 = std::f64::consts::TAU;

mod parse;
mod bodies;
mod assembly;
mod builder;
mod geometry;
mod styles;
mod pmi;
// BREP private tests: c4f20f6ac1ecc685

use crate::appearance::{BodyAppearance, ImportedColor};
use parse::*;
use bodies::*;
use builder::*;
use geometry::*;
use styles::*;
pub use assembly::{read_step_assembly, StepAssembly, StepOccurrence, StepProduct};
pub use pmi::read_step_pmi;
pub(crate) use pmi::decode_step_text as decode_step_text_for_tests;


// ---------------------------------------------------------------------------
// Topology assembly
// ---------------------------------------------------------------------------

/// Public entry: parse a STEP Part 21 document and return one `BrepSolid` per
/// MANIFOLD_SOLID_BREP, FACETED_BREP, or certified BREP_WITH_VOIDS. Every
/// returned solid passes `validate()`.
pub fn import_step(text: &str) -> Result<Vec<BrepSolid>, String> {
    Ok(import_step_with_appearance(text)?.0)
}

/// [`import_step`] plus the file's PRESENTATION colours: one
/// [`BodyAppearance`] per returned solid, in the same order.
///
/// This is the lane the IMPORT3D feature uses, because colour is stamped as
/// scene metadata on the names IT chooses — see
/// `feature_pipeline::features::import3d` and `io/appearance.rs` for the record
/// shape. A file with no presentation entities returns default (empty)
/// appearances, never a short vector, so callers can `zip` unconditionally.
pub fn import_step_with_appearance(
    text: &str,
) -> Result<(Vec<BrepSolid>, Vec<BodyAppearance>), String> {
    let imported = collect_step_solids(text)?;
    if imported.solids.is_empty() {
        return Err(imported.first_error.unwrap_or_else(|| {
            "step_import: no MANIFOLD_SOLID_BREP/FACETED_BREP/BREP_WITH_VOIDS body imported (unsupported or invalid representation)"
                .into()
        }));
    }
    Ok((imported.solids, imported.appearances))
}

/// Like [`import_step`] but reports how many bodies failed and the first error,
/// so callers can distinguish a fully- from a partially-imported assembly.
pub fn import_step_report(text: &str) -> Result<(Vec<BrepSolid>, usize, Option<String>), String> {
    let imported = collect_step_solids(text)?;
    Ok((imported.solids, imported.failed, imported.first_error))
}
