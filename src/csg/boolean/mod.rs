use crate::arrangement::Vec2;
use crate::classification::{classify_point, parameter_point_in_face, PolygonClass};
use crate::fragment::{FaceFragmentRecord, FragmentEdgeSource};
use crate::imprint::{
    EdgeSplitRecord, FaceImprints, FaceKey, ImprintOptions, ImprintPieceRecord,
    ImprintResultRecord, ImprintVertex,
};
use crate::tolerance::{assembler_weld, commit_weld};
use crate::topology::{
    adaptive_coedge_error, BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord,
    ShellRecord, VertexRecord,
};
use crate::{
    KernelRefusal, OrRefuse, RefusalClass,
    apply_edge_splits, apply_edge_splits_with_map, build_imprints, build_pcurve_on_surface,
    build_pcurve_on_surface_range, classify_surface_pair,
    fragment_solid, interpolate_curve, merge_curve_continuation_edges,
    merge_same_surface_faces_excluding,
    project_point_to_curve, project_point_to_surface, solid_signed_volume, AffineTransform,
    DiagnosticSeverity, KernelDiagnostics, KernelOutcome, KernelStage, KernelTolerances, NurbsCurve,
    PointClass, SolidClassifier, SurfacePairRelation, Vec3,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::{Deserialize, Serialize};
use web_time::Instant;

thread_local! {
    /// Running count (per thread) of one-use edges the EDGE-CONFORMANCE
    /// repair lanes merged or bridged.  The lanes only fire on an assembly
    /// that would otherwise refuse, so a zero delta across an operation is
    /// an honest witness that the boolean assembled cleanly without the
    /// repair.  Test-observable; not part of any result.
    pub(crate) static CONFORMANCE_REPAIRS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BooleanOperation {
    Union,
    Intersect,
    Subtract,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BooleanOptions {
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// Optional complete accuracy/search policy.  `tolerance` remains the
    /// backwards-compatible model-identity shorthand when this is absent.
    #[serde(default)]
    pub tolerances: Option<KernelTolerances>,
    #[serde(default)]
    pub imprint: ImprintOptions,
    /// Coalesce adjacent result fragments after assembly. This defaults to
    /// true to preserve the kernel's normal clean-boundary behavior.
    #[serde(default = "default_true")]
    pub merge_coplanar_faces: bool,
    /// Face-name substrings that PIN a result face out of the coplanar/cosurface
    /// merge (only consulted when [`merge_coplanar_faces`] is on). A face whose
    /// `name` CONTAINS any of these is emitted unchanged instead of being
    /// coalesced with a mergeable neighbour; faces matching none merge exactly as
    /// before. Default empty — no caller sees a behavior change unless it opts
    /// in. Sheet metal uses this to keep every thickness/side wall face per
    /// outline segment in the folded solid (so a flange can still attach to a
    /// specific segment: fusing two collinear thickness faces would erase the
    /// segment boundary).
    #[serde(default)]
    pub keep_unmerged_name_substrs: Vec<String>,
}

fn default_tolerance() -> f64 {
    1e-7
}

fn default_true() -> bool {
    true
}

impl Default for BooleanOptions {
    fn default() -> Self {
        Self {
            tolerance: default_tolerance(),
            tolerances: None,
            imprint: ImprintOptions::default(),
            merge_coplanar_faces: true,
            keep_unmerged_name_substrs: Vec::new(),
        }
    }
}

mod select;
mod assemble;
mod rim;
// BREP private tests: 105eeef69a3ee87f

use rim::*;
use select::*;
// BREP private tests: bf6d0f8b58224c78
pub(crate) use assemble::{
    assemble_fragments, assemble_open_fragments, commit_nearby_edge_endpoints,
    edge_interior_lies_on, finalize_assembled_solid,
};
// Currently referenced only inside `assemble`, so the re-export is unused.
#[allow(unused_imports)]
pub(crate) use assemble::apply_assembly_heal_chain;

pub fn boolean_operation(
    first: &BrepSolid,
    second: &BrepSolid,
    operation: BooleanOperation,
    options: &BooleanOptions,
) -> Result<BrepSolid, KernelRefusal> {
    match boolean_operation_with_diagnostics(first, second, operation, options) {
        Ok(outcome) => Ok(outcome.value),
        Err(error) => {
            // Perturbation-fallback lane (Simulation of Simplicity). The exact
            // arrangement only errs on a DEGENERACY here (coincident /
            // near-tangent carrier faces make it structurally inconsistent) —
            // e.g. a partial torus whose equatorial circles lie exactly in a box
            // face. Retry once on a rigidly perturbed second operand; accept only
            // an oracle-clean, volume-consistent result. See
            // `perturbation_retry`.
            if std::env::var("BREP_NO_PERTURB").as_deref() == Ok("1")
                || !error.class.perturbation_eligible()
            {
                return Err(error);
            }
            // INTERNAL-TANGENCY PINCH GATE. Perturbation is only legitimate
            // where the exact arrangement is UNSTABLE but its ANSWER is not:
            // the nearby transversal case must carry the same topology the
            // degenerate one does. An internal tangency in a DIFFERENCE breaks
            // exactly that premise — the exact `A − B` pinches to zero
            // thickness along the contact, so an epsilon either tears the wall
            // open (a through-slot where the exact answer has intact material)
            // or leaves a sub-micron web. Both are within epsilon in Hausdorff
            // distance and in volume, so neither the oracle (boundary skip
            // band) nor the 1% CSG volume bound can see them, and the lane
            // would ship the tolerant-kernel sliver this kernel refuses. Only
            // consulted on the failure path, so nothing that succeeds today can
            // change; the honest exact-path error is returned instead.
            // Escape hatch for tamper-verification: BREP_PINCH_GATE=0.
            if matches!(operation, BooleanOperation::Subtract)
                && std::env::var("BREP_PINCH_GATE").as_deref() != Ok("0")
                && subtract_pinches_at_internal_tangency(first, second, options.tolerance)
                    .unwrap_or(false)
            {
                return Err(error.with_message(|message| format!(
                    "{message}; internal tangency: the operands touch tangentially with \
                     co-directed normals, so the exact difference pinches to zero thickness \
                     (non-manifold, unrepresentable in a boundary model) — refusing rather \
                     than returning a perturbed sliver"
                )));
            }
            match perturbation_retry(first, second, operation, options) {
                Some(solid) => Ok(solid),
                None => Err(error),
            }
        }
    }
}

/// Grid resolution (per parameter direction, endpoints included) used to probe a
/// face pair for a tangential contact. Endpoints and the midpoint are both
/// sampled, so a contact sitting on a periodic seam (a cylinder's `u = 0`
/// ruling) and one sitting at a face's parametric centre are both hit exactly.
const PINCH_PROBE_STEPS: usize = 8;

/// `|n_a × n_b|` bound below which two unit normals count as parallel — the same
/// bound `csg::imprint`'s pair classifier uses (`PAIR_ANGULAR_TOLERANCE`).
const PINCH_ANGULAR_TOLERANCE: f64 = 1e-4;

/// Parameter-space step (fraction of the domain span) taken around a contact
/// sample to prove the contact is LOWER-DIMENSIONAL — a tangency curve/point
/// rather than a cosurface patch.
const PINCH_SEPARATION_STEP: f64 = 1e-2;

/// Would the exact `first − second` PINCH to zero thickness at a tangential
/// contact?
///
/// True when some face of `first` and some face of `second` touch tangentially
/// (surfaces within the contact band, normals parallel) with CO-DIRECTED outward
/// normals, at a contact that is lower-dimensional (the surfaces separate as you
/// step away from it).
///
/// Co-directed outward normals at a tangency mean one solid lies locally INSIDE
/// the other: subtracting leaves material on both sides of the contact that
/// meets there at zero thickness — a non-manifold pinch no boundary model can
/// represent. Anti-directed normals are the harmless EXTERNAL tangency (a
/// cylinder resting against a wall): the difference simply keeps `first` intact.
/// The lower-dimensionality requirement excludes a CO-SURFACE contact (a pocket
/// wall flush with an outer wall — co-directed normals, but the surfaces stay
/// coincident in every direction), where the difference is perfectly well
/// behaved.
///
/// Conservative by construction: it only reports a pinch it can actually witness
/// on the probe grid, and a miss just leaves the perturbation lane to its
/// existing gates.
fn subtract_pinches_at_internal_tangency(
    first: &BrepSolid,
    second: &BrepSolid,
    tolerance: f64,
) -> Result<bool, KernelRefusal> {
    // Same contact band the imprint pair classifier uses for its sampled
    // "these carriers touch" verdict.
    let contact = (tolerance * 100.0).max(1e-12);
    let probes_a = first
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .map(PinchProbe::of)
        .collect::<Result<Vec<_>, _>>()?;
    let probes_b = second
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .map(PinchProbe::of)
        .collect::<Result<Vec<_>, _>>()?;
    for probe_a in &probes_a {
        for probe_b in &probes_b {
            // Sampled-hull cull, so the (failure-path-only) cost stays linear in
            // the faces that actually touch rather than quadratic in every face.
            // The hull is built from the probe grid, so it can under-cover a
            // curved carrier between samples: pad it by a percent of the pair's
            // size before culling. Over-keeping a pair only costs a probe that
            // finds nothing.
            let pad = 1e-2 * probe_a.extent().max(probe_b.extent());
            if probe_a.separation(probe_b) > contact + pad {
                continue;
            }
            // CO-SURFACE contacts are 2-dimensional, not tangencies: two flush
            // walls (a pocket's side coincident with an outer wall, a cylinder
            // cap lying in a box face) have co-directed normals wherever the
            // solids nest, yet the difference there is perfectly well behaved.
            // Only a LOWER-dimensional contact pinches, so drop the pair the
            // pair classifier calls cosurface before probing it.
            if classify_surface_pair(
                &probe_a.face.surface,
                &probe_b.face.surface,
                tolerance,
                PINCH_ANGULAR_TOLERANCE,
            ).or_refuse(KernelStage::Validate, "csg.boolean.mod")?
            .relation
                == SurfacePairRelation::Cosurface
            {
                continue;
            }
            if faces_touch_with_codirected_normals(probe_a, probe_b, contact)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// A face's probe grid plus the grid's bounding box, built once per face so the
/// pair loop only pays for a hull comparison.
struct PinchProbe<'a> {
    face: &'a FaceRecord,
    /// `(u, v, point)` on the face's carrier surface.
    samples: Vec<(f64, f64, Vec3)>,
    minimum: [f64; 3],
    maximum: [f64; 3],
}

impl<'a> PinchProbe<'a> {
    fn of(face: &'a FaceRecord) -> Result<Self, KernelRefusal> {
        let [u0, u1] = face.surface.domain_u().or_refuse(KernelStage::Validate, "domain_u")?;
        let [v0, v1] = face.surface.domain_v().or_refuse(KernelStage::Validate, "domain_v")?;
        let steps = PINCH_PROBE_STEPS as f64;
        let mut samples = Vec::with_capacity((PINCH_PROBE_STEPS + 1).pow(2));
        let mut minimum = [f64::INFINITY; 3];
        let mut maximum = [f64::NEG_INFINITY; 3];
        for i in 0..=PINCH_PROBE_STEPS {
            let u = u0 + (u1 - u0) * i as f64 / steps;
            for j in 0..=PINCH_PROBE_STEPS {
                let v = v0 + (v1 - v0) * j as f64 / steps;
                let Ok(point) = face.surface.evaluate(u, v) else {
                    continue;
                };
                for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                    minimum[axis] = minimum[axis].min(value);
                    maximum[axis] = maximum[axis].max(value);
                }
                samples.push((u, v, point));
            }
        }
        Ok(Self {
            face,
            samples,
            minimum,
            maximum,
        })
    }

    /// Largest side of the sampled hull (0 when nothing sampled).
    fn extent(&self) -> f64 {
        (0..3)
            .map(|axis| self.maximum[axis] - self.minimum[axis])
            .fold(0.0f64, f64::max)
    }

    /// Axis-aligned gap between the two sampled hulls (0 when they overlap).
    fn separation(&self, other: &Self) -> f64 {
        let mut gap: f64 = 0.0;
        for axis in 0..3 {
            gap = gap.max(self.minimum[axis] - other.maximum[axis]);
            gap = gap.max(other.minimum[axis] - self.maximum[axis]);
        }
        gap
    }
}

/// One face pair of [`subtract_pinches_at_internal_tangency`]: probe both
/// surfaces on a grid, keep samples that land on the other surface inside BOTH
/// trims with parallel co-directed outward normals, and accept only where the
/// contact provably separates nearby.
fn faces_touch_with_codirected_normals(
    probe_a: &PinchProbe<'_>,
    probe_b: &PinchProbe<'_>,
    contact: f64,
) -> Result<bool, KernelRefusal> {
    for (probe, other) in [(probe_a, probe_b), (probe_b, probe_a)] {
        let source = probe.face;
        let target = other.face;
        for &(u, v, point) in &probe.samples {
            let projection = project_point_to_surface(&target.surface, point).or_refuse(KernelStage::Validate, "project_point_to_surface")?;
            if projection.distance > contact {
                continue;
            }
            // A pole (collapsed du × dv) has no reliable orientation here;
            // skipping it keeps the predicate conservative.
            let (Ok(source_normal), Ok(target_normal)) = (
                source.surface.normal(u, v),
                target.surface.normal(projection.u, projection.v),
            ) else {
                continue;
            };
            let source_outward = outward_normal(source_normal, source.same_sense);
            let target_outward = outward_normal(target_normal, target.same_sense);
            if source_outward.cross(target_outward).length() > PINCH_ANGULAR_TOLERANCE
                || source_outward.dot(target_outward) <= 0.0
            {
                continue;
            }
            if parameter_point_in_face(source, Vec2 { x: u, y: v }, 1e-6).or_refuse(KernelStage::Validate, "parameter_point_in_face")?
                == PolygonClass::Outside
                || parameter_point_in_face(
                    target,
                    Vec2 {
                        x: projection.u,
                        y: projection.v,
                    },
                    1e-6,
                ).or_refuse(KernelStage::Validate, "csg.boolean.mod")? == PolygonClass::Outside
            {
                continue;
            }
            if contact_separates_locally(source, target, u, v, contact)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn outward_normal(normal: Vec3, same_sense: bool) -> Vec3 {
    if same_sense {
        normal
    } else {
        normal.scale(-1.0)
    }
}

/// Does the contact at `source(u, v)` LEAVE the contact band when you step away
/// from it in parameter space? A tangency curve or point does (stepping across
/// the tangency separates the surfaces quadratically); a cosurface patch does
/// not (it stays coincident in every direction). Steps are clamped into the
/// domain, so a sample on a seam or a domain edge probes only inward.
fn contact_separates_locally(
    source: &FaceRecord,
    target: &FaceRecord,
    u: f64,
    v: f64,
    contact: f64,
) -> Result<bool, KernelRefusal> {
    let [u0, u1] = source.surface.domain_u().or_refuse(KernelStage::Validate, "domain_u")?;
    let [v0, v1] = source.surface.domain_v().or_refuse(KernelStage::Validate, "domain_v")?;
    let step_u = (u1 - u0) * PINCH_SEPARATION_STEP;
    let step_v = (v1 - v0) * PINCH_SEPARATION_STEP;
    for (probe_u, probe_v) in [
        ((u + step_u).min(u1), v),
        ((u - step_u).max(u0), v),
        (u, (v + step_v).min(v1)),
        (u, (v - step_v).max(v0)),
    ] {
        if (probe_u - u).abs() < f64::EPSILON && (probe_v - v).abs() < f64::EPSILON {
            continue;
        }
        let Ok(point) = source.surface.evaluate(probe_u, probe_v) else {
            continue;
        };
        // A generous multiple of the band: the step must clear it decisively,
        // never on projection noise.
        if project_point_to_surface(&target.surface, point).or_refuse(KernelStage::Validate, "project_point_to_surface")?.distance > contact * 10.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Deterministic perturbation-fallback (Simulation of Simplicity), the textbook
/// CAD answer to an exact/near-tangent arrangement degeneracy, adopted after six
/// incremental arrangement/imprint/fragment fixes for the near-tangent class were
/// each falsified, rather than attempting a seventh. Rigidly TRANSLATE the second operand by a tiny
/// epsilon so the degenerate contact becomes a clean transversal crossing, redo
/// the boolean, and accept the result ONLY if it is (a) topologically valid —
/// guaranteed by `assemble`'s own gate returning `Ok` — (b) semantically equal to
/// the exact CSG of the ORIGINAL operands (the point-classification oracle), and
/// (c) volume-consistent with the CSG inequality bounds. A wrong perturbed result
/// is rejected on (b)/(c), so this lane preserves the engine's fail-safe property:
/// it can only turn a degeneracy-error into a CORRECT solid, never a wrong one.
///
/// INVARIANT — no recursion: this calls `boolean_operation_with_diagnostics`
/// (the EXACT path), never `boolean_operation`, so the fallback can never
/// re-enter itself. A future refactor must preserve that.
fn perturbation_retry(
    first: &BrepSolid,
    second: &BrepSolid,
    operation: BooleanOperation,
    options: &BooleanOptions,
) -> Option<BrepSolid> {
    // Generic translation directions: each has a substantial component on ALL
    // three axes, so whatever axis the coincidence normal lies along (an
    // equatorial plane flush with a box face, a cap plane flush with a side
    // face, …) at least one direction has a component that lifts the contact off
    // exact tangency. Axis-aligned directions are deliberately excluded — a nudge
    // parallel to the coincident plane leaves the degeneracy in place (verified:
    // pure ±X / ±Z never rescue the flush-torus case). Fixed, not hash-seeded
    // as the lane was first sketched: the seeding's only purpose was to rule out
    // RNG/clock nondeterminism, which fixed vectors already satisfy, and four
    // diverse directions cover any axis-aligned normal.
    const DIRECTIONS: [[f64; 3]; 4] = [
        [0.4034, 0.7973, 0.4491],
        [0.7973, 0.4491, 0.4034],
        [0.4491, 0.4034, 0.7973],
        [0.5774, -0.5774, 0.5774],
    ];
    // Fractions of operand scale, ordered SMALLEST-FIRST so the accepted solid
    // carries the least geometric offset. Sized above the near-tangent sliver
    // dead zone and below local feature size; the ladder + accept-gate make the
    // exact magnitude non-critical (a rung that lands in a dead zone simply fails
    // the gate and the next rung is tried).
    const FRACTIONS: [f64; 5] = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3];

    let debug = std::env::var("BREP_DEBUG_BOOL").is_ok();
    let scale = crate::tolerance::solid_scale(first)
        .max(crate::tolerance::solid_scale(second))
        .max(1.0);

    // Original operand volumes for the CSG inequality backstop (the oracle skips
    // points near boundaries, so a wrong thin shell hugging the flush face is
    // invisible to it but shows up here).
    let va = solid_signed_volume(first).ok().map(f64::abs);
    let vb = solid_signed_volume(second).ok().map(f64::abs);

    for &fraction in &FRACTIONS {
        let magnitude = scale * fraction;
        for dir in &DIRECTIONS {
            let offset = [dir[0] * magnitude, dir[1] * magnitude, dir[2] * magnitude];
            let translate = match AffineTransform::new([
                1.0, 0.0, 0.0, offset[0], //
                0.0, 1.0, 0.0, offset[1], //
                0.0, 0.0, 1.0, offset[2], //
                0.0, 0.0, 0.0, 1.0,
            ]) {
                Ok(transform) => transform,
                Err(_) => continue,
            };
            let moved = match crate::transform_brep(second, translate, false) {
                Ok(solid) => solid,
                Err(_) => continue,
            };
            let candidate =
                match boolean_operation_with_diagnostics(first, &moved, operation, options) {
                    Ok(outcome) => outcome.value,
                    Err(_) => continue,
                };
            // A perturbation must not "rescue" a case into emptiness.
            if candidate.shells.is_empty() {
                continue;
            }
            // (c) Volume must satisfy the CSG inequality for the ORIGINAL
            // operands (generous 1% relative slack: catches gross wrongness —
            // doubled/halved bodies, missing lobes — without rejecting the
            // legitimate sub-micron offset).
            if let (Some(va), Some(vb)) = (va, vb) {
                if let Ok(vr) = solid_signed_volume(&candidate) {
                    if !volume_within_csg_bounds(operation, va, vb, vr.abs()) {
                        continue;
                    }
                }
            }
            // (b) Semantic agreement with the exact CSG of the ORIGINAL
            // operands. Reject a vacuous verdict (skip-band swallowed every
            // sample so nothing was actually checked).
            match crate::oracle::boolean_semantic_disagreement(
                first, second, operation, &candidate, 400,
            ) {
                Ok(report) if report.considered >= 30 && !report.is_flagged() => {
                    if debug {
                        eprintln!(
                            "[perturb] rescued {operation:?}: dir={dir:?} fraction={fraction:.1e} \
                             magnitude={magnitude:.3e} offset={offset:?} \
                             (oracle considered={} rate={:.4})",
                            report.considered, report.disagreement_rate
                        );
                    }
                    return Some(candidate);
                }
                _ => continue,
            }
        }
    }
    if debug {
        eprintln!("[perturb] ladder exhausted for {operation:?}; no rung produced a clean result");
    }
    None
}

/// CSG volume inequality bounds (no extra boolean needed) with 1% relative slack.
/// `va`, `vb`, `vr` are absolute volumes of operand A, operand B, and the result.
fn volume_within_csg_bounds(operation: BooleanOperation, va: f64, vb: f64, vr: f64) -> bool {
    let slack = 1e-2 * (va + vb).max(1.0);
    match operation {
        // max(A,B) ≤ A∪B ≤ A+B
        BooleanOperation::Union => vr >= va.max(vb) - slack && vr <= va + vb + slack,
        // A−B ≤ A (and ≥ 0)
        BooleanOperation::Subtract => vr <= va + slack && vr >= -slack,
        // A∩B ≤ min(A,B)
        BooleanOperation::Intersect => vr <= va.min(vb) + slack && vr >= -slack,
    }
}

// ---------------------------------------------------------------------------
// N-ary booleans (Golovanov T4.4)
//
// Union / intersect / subtract of N solids in ONE imprint+fragment pass.  This
// is a NEW ADDITIVE entry point: the binary `boolean_operation` above is left
// byte-identical.  The pipeline mirrors the binary one but generalizes the two
// operand-specific stages to N operands:
//
//   1. Imprint EVERY unordered pair (i, j) so every mutual intersection edge is
//      present on both operands' faces, and accumulate all pairs into ONE global
//      imprint (operand indices remapped to 0..N, piece/vertex ids made globally
//      unique, per-face piece lists and per-edge split parameters merged).
//   2. Split + fragment each operand by that single global imprint, so each
//      operand's faces are cut by ALL of its intersections at once.
//   3. Select fragments in one pass: the per-fragment keep decision generalizes
//      from "vs one other solid" to "vs the SET of other solids" (see
//      `nary_keep`).  Because every pair is imprinted, a fragment's interior lies
//      wholly inside or wholly outside each other operand, so a single interior
//      test point per (fragment, other-solid) is an exact membership verdict.
//   4. Assemble with the same machinery (`assemble_fragments`), then merge and
//      validate exactly as the binary path does.
// ---------------------------------------------------------------------------

/// Reverse a fragment's orientation (surface sense + every coedge), the same
/// transform the binary Subtract path applies to the second operand's kept
/// fragments.  Extracted so the n-ary selector can reuse it verbatim.
fn reverse_fragment(fragment: &mut FaceFragmentRecord) -> Result<(), KernelRefusal> {
    fragment.same_sense = !fragment.same_sense;
    for loop_record in &mut fragment.loops {
        loop_record.coedges.reverse();
        for coedge in &mut loop_record.coedges {
            coedge.forward = !coedge.forward;
            coedge.pcurve = coedge.pcurve.reversed().or_refuse(KernelStage::Validate, "reversed")?;
        }
    }
    Ok(())
}

/// Imprint every unordered pair of operands and fold the pairwise
/// `ImprintResultRecord`s into ONE global imprint whose operand indices are the
/// operands' global positions (0..N).  Piece and vertex ids are re-based to be
/// globally unique; `by_face` piece lists and `edge_splits` parameter lists are
/// MERGED per key so `fragment_face` (which takes the first `by_face` match) and
/// `apply_edge_splits` (which collects one entry per edge) see every cut.
fn build_nary_imprint(
    operands: &[BrepSolid],
    options: &ImprintOptions,
) -> Result<ImprintResultRecord, KernelRefusal> {
    let mut section_evidence = false;
    let mut vertices: Vec<ImprintVertex> = Vec::new();
    let mut pieces: Vec<ImprintPieceRecord> = Vec::new();
    let mut by_face: HashMap<(u8, u64), Vec<u64>> = HashMap::default();
    let mut edge_splits: HashMap<(u8, u64), Vec<f64>> = HashMap::default();
    let mut next_vertex_id: u64 = 1;
    let mut next_piece_id: u64 = 1;

    for i in 0..operands.len() {
        for j in (i + 1)..operands.len() {
            let pair = build_imprints(&operands[i], &operands[j], options)?;
            let remap = |operand: u8| -> u8 {
                if operand == 0 {
                    i as u8
                } else {
                    j as u8
                }
            };

            section_evidence = section_evidence || pair.section_evidence;
            let mut vertex_map: HashMap<u64, u64> = HashMap::default();
            for vertex in &pair.vertices {
                let global = next_vertex_id;
                next_vertex_id += 1;
                vertex_map.insert(vertex.id, global);
                vertices.push(ImprintVertex {
                    id: global,
                    point: vertex.point,
                });
            }

            let mut piece_map: HashMap<u64, u64> = HashMap::default();
            for piece in &pair.pieces {
                let global = next_piece_id;
                next_piece_id += 1;
                piece_map.insert(piece.id, global);
                let mut mapped = piece.clone();
                mapped.id = global;
                mapped.start_vertex_id = *vertex_map
                    .get(&piece.start_vertex_id)
                    .ok_or_else(|| KernelRefusal::internal(KernelStage::Intersect, "nary.imprint", "n-ary imprint: piece references unknown vertex"))?;
                mapped.end_vertex_id = *vertex_map
                    .get(&piece.end_vertex_id)
                    .ok_or_else(|| KernelRefusal::internal(KernelStage::Intersect, "nary.imprint", "n-ary imprint: piece references unknown vertex"))?;
                for pcurve in &mut mapped.pcurves {
                    pcurve.operand = remap(pcurve.operand);
                }
                mapped.support_faces = [
                    FaceKey {
                        operand: remap(piece.support_faces[0].operand),
                        face_id: piece.support_faces[0].face_id,
                    },
                    FaceKey {
                        operand: remap(piece.support_faces[1].operand),
                        face_id: piece.support_faces[1].face_id,
                    },
                ];
                pieces.push(mapped);
            }

            for entry in &pair.by_face {
                let list = by_face
                    .entry((remap(entry.operand), entry.face_id))
                    .or_default();
                for piece_id in &entry.piece_ids {
                    let global = *piece_map.get(piece_id).ok_or_else(|| {
                        KernelRefusal::internal(KernelStage::Intersect, "nary.imprint", "n-ary imprint: by_face references unknown piece")
                    })?;
                    list.push(global);
                }
            }

            for split in &pair.edge_splits {
                edge_splits
                    .entry((remap(split.operand), split.edge_id))
                    .or_default()
                    .extend(split.parameters.iter().copied());
            }
        }
    }

    Ok(ImprintResultRecord {
        vertices,
        pieces,
        tangent_nodes: Vec::new(),
        barrier_edges: Vec::new(),
        section_evidence,
        by_face: by_face
            .into_iter()
            .map(|((operand, face_id), piece_ids)| FaceImprints {
                operand,
                face_id,
                piece_ids,
            })
            .collect(),
        edge_splits: edge_splits
            .into_iter()
            .map(|((operand, edge_id), parameters)| EdgeSplitRecord {
                operand,
                edge_id,
                parameters,
            })
            .collect(),
    })
}

/// N-ary union / intersect / subtract of `operands` in ONE imprint+fragment
/// pass (Golovanov T4.4).  Additive: does not touch the binary path.
///
/// - `Union`   : the boundary of `∪ operands`.
/// - `Intersect`: the boundary of `∩ operands`.
/// - `Subtract`: `operands[0]` minus the union of `operands[1..]`.
///
/// `N = 1` returns the sole operand; `N = 2` runs the same general pipeline and
/// reproduces the binary result (same volume, valid).  Returns a clear `Err`
/// when the operands cannot assemble into a closed, valid solid.
pub fn boolean_operation_nary(
    operands: &[BrepSolid],
    operation: BooleanOperation,
) -> Result<BrepSolid, KernelRefusal> {
    if operands.is_empty() {
        return Err(KernelRefusal::input(KernelStage::Collect, "operands", "boolean_operation_nary: no operands provided"));
    }
    if operands.len() == 1 {
        return Ok(operands[0].clone());
    }
    if operands.len() > 255 {
        return Err(KernelRefusal::input(KernelStage::Collect, "operands", "boolean_operation_nary: at most 255 operands supported"));
    }
    let debug = std::env::var("BREP_DEBUG_BOOL").is_ok();
    let options = BooleanOptions::default();

    // One tolerance policy for the whole set: the largest-scale operand's policy
    // dominates (every derived band is monotonic in scale), matching how the
    // binary path takes `for_pair` = max of the two scales.
    let mut policy = KernelTolerances::for_solid(&operands[0], options.tolerance);
    for operand in &operands[1..] {
        let candidate = KernelTolerances::for_solid(operand, options.tolerance);
        if candidate.sew_search > policy.sew_search {
            policy = candidate;
        }
    }
    policy.check().or_input(KernelStage::Collect, "tolerances")?;
    let tolerance = policy.model;

    // Per-operand fuse-first healing, exactly as the binary path (a no-op on
    // clean input, so clean operands are untouched).
    let mut healed = operands.to_vec();
    for operand in &mut healed {
        crate::heal::heal_operands(operand, &policy).or_refuse(KernelStage::Validate, "heal_operands")?;
        // Same operand normalization as the binary path: seam edges for
        // seamless full-period band faces.
        normalize_operand_band_seams(operand)?;
    }

    // 1. Imprint every pair into one global imprint.
    let mut imprint_options = options.imprint.clone();
    imprint_options.tolerance = tolerance;
    let imprint = build_nary_imprint(&healed, &imprint_options)?;

    // 2. Split every operand's edges, then fragment its faces, by that imprint.
    let mut split = Vec::with_capacity(healed.len());
    for (index, operand) in healed.iter().enumerate() {
        split.push(apply_edge_splits(operand, index as u8, &imprint)?);
    }
    let mut fragments = Vec::with_capacity(split.len());
    for (index, solid) in split.iter().enumerate() {
        fragments.push(fragment_solid(solid, index as u8, &imprint)?);
    }

    // 3. Select fragments in one pass against the set of other operands.
    let selected = select_fragments_nary(&fragments, &healed, operation, tolerance, debug)?;

    // 4. Assemble + normalize + validate, mirroring the binary tail.
    let solids: HashMap<u8, &BrepSolid> = split
        .iter()
        .enumerate()
        .map(|(index, solid)| (index as u8, solid))
        .collect();
    let solid = assemble_fragments(selected, &solids, &imprint, tolerance)?;
    let solid = if options.merge_coplanar_faces
        && std::env::var("BREP_COPLANAR_MERGE").as_deref() != Ok("0")
    {
        let merged = merge_same_surface_faces_excluding(
            &solid,
            tolerance,
            &options.keep_unmerged_name_substrs,
        )?;
        merge_curve_continuation_edges(&merged, tolerance).or_refuse(KernelStage::Validate, "merge_curve_continuation_edges")?
    } else {
        solid
    };
    let validation = solid.validate_detailed(&policy);
    if !validation.issues.is_empty() {
        return Err(KernelRefusal::new(
            RefusalClass::InvalidResultTopology { issues: validation.issues.len() as u32 },
            KernelStage::Validate,
            format!(
            "boolean_operation_nary: invalid result: {:?}",
            validation.issues
        )));
    }
    if debug {
        if let Ok(report) =
            crate::oracle::boolean_semantic_disagreement_nary(&healed, operation, &solid, 300)
        {
            if report.is_flagged() {
                eprintln!(
                    "[oracle] n-ary semantic disagreement rate {:.4} ({} of {} decidable points); sample {:?}",
                    report.disagreement_rate,
                    report.disagreements.len(),
                    report.considered,
                    report.sample_disagreement()
                );
            }
        }
    }
    Ok(solid)
}

/// TANGENT-NODE ATTRIBUTION.
///
/// The imprint's second-order filter decides which tangential contacts are
/// isolated NODES the arrangement can carve, and marches those instead of
/// refusing. It cannot decide a THIRD-order singularity — where the transverse
/// branch is tangent to a contact curve, every second-order quantity vanishes
/// together and the contact reads as a clean saddle at every stopping rule
/// (`imprint/tangent_contact.rs`). The assembly is the authority on whether an
/// admitted node was really imprintable, so a tear downstream of one is
/// reported AGAINST THE NODE rather than as an anonymous degeneracy: same
/// refusal class the whole tangent-node family carries, naming the point.
///
/// Only classes that are ALREADY perturbation-eligible are re-attributed, so
/// the retry gate sees exactly what it sees today — this changes what a
/// refusal SAYS, never which refusals are retried.
fn attribute_to_tangent_node(error: KernelRefusal, nodes: &[Vec3]) -> KernelRefusal {
    let Some(node) = nodes.first() else {
        return error;
    };
    if !error.class.perturbation_eligible()
        || matches!(error.class, RefusalClass::TangentNodeSingularity)
    {
        return error;
    }
    KernelRefusal::new(
        RefusalClass::TangentNodeSingularity,
        error.stage,
        format!(
            "boolean: unsupported singular/tangent-node surface intersection \
             at the tangent node at ({:.6},{:.6},{:.6}): the section through it did not \
             assemble into a closed shell",
            node.x, node.y, node.z
        ),
    )
}

pub fn boolean_operation_with_diagnostics(
    first: &BrepSolid,
    second: &BrepSolid,
    operation: BooleanOperation,
    options: &BooleanOptions,
) -> Result<KernelOutcome<BrepSolid>, KernelRefusal> {
    let mut tangent_nodes = Vec::new();
    boolean_pipeline(first, second, operation, options, &mut tangent_nodes)
        .map_err(|error| attribute_to_tangent_node(error, &tangent_nodes))
}

fn boolean_pipeline(
    first: &BrepSolid,
    second: &BrepSolid,
    operation: BooleanOperation,
    options: &BooleanOptions,
    tangent_nodes: &mut Vec<Vec3>,
) -> Result<KernelOutcome<BrepSolid>, KernelRefusal> {
    let policy = options
        .tolerances
        .unwrap_or_else(|| KernelTolerances::for_pair(first, second, options.tolerance));
    policy.check().or_input(KernelStage::Collect, "tolerances")?;
    let tolerance = policy.model;

    // Fuse-first operand healing (Lever A): before any intersection touches the
    // operands, snap each one's near-coincident / off-plane vertices to exact
    // and re-anchor the incident edge curves onto them, so noisy
    // near-degenerate input (points that should coincide but drifted, vertices
    // a few microns off a planar face) cannot tip a fixed downstream band over
    // the edge.  Healing is per-operand, validate-gated, and — because a clean
    // operand has no vertices within `heal_tol` and none off its planes — a
    // no-op that leaves the boolean output byte-identical on clean inputs.
    let mut first_owned = first.clone();
    let mut second_owned = second.clone();
    crate::heal::heal_operands(&mut first_owned, &policy).or_refuse(KernelStage::Validate, "heal_operands")?;
    crate::heal::heal_operands(&mut second_owned, &policy).or_refuse(KernelStage::Validate, "heal_operands")?;
    // Seamless full-period band faces (STEP import) break the seam-aware
    // imprint/arrangement machinery; normalize them to the seam-carrying
    // topology native periodic faces use. No-op on operands without such
    // faces. See `insert_periodic_band_seam_edges`.
    normalize_operand_band_seams(&mut first_owned)?;
    normalize_operand_band_seams(&mut second_owned)?;
    let first = &first_owned;
    let second = &second_owned;

    let mut diagnostics = KernelDiagnostics::default();
    let operation_started = Instant::now();
    diagnostics.count_n(
        "collect.faces",
        first
            .shells
            .iter()
            .chain(&second.shells)
            .map(|shell| shell.faces.len() as u64)
            .sum(),
    );
    diagnostics.measure_max("tolerance.model", policy.model);
    diagnostics.measure_max("tolerance.sew_search", policy.sew_search);

    let mut imprint_options = options.imprint.clone();
    imprint_options.tolerance = tolerance;
    let stage_started = Instant::now();
    let imprint = build_imprints(first, second, &imprint_options)?;
    tangent_nodes.extend_from_slice(&imprint.tangent_nodes);
    if std::env::var("BREP_DEBUG_BOOL").is_ok() {
        for piece in &imprint.pieces {
            let start = piece.curve.evaluate(piece.t0);
            let end = piece.curve.evaluate(piece.t1);
            eprintln!(
                "piece {} supports={:?} t=[{:.6},{:.6}] start={:?} end={:?}",
                piece.id, piece.support_faces, piece.t0, piece.t1, start, end
            );
        }
        for entry in &imprint.by_face {
            eprintln!(
                "by_face operand={} face={} pieces={:?}",
                entry.operand, entry.face_id, entry.piece_ids
            );
        }
        for split in &imprint.edge_splits {
            eprintln!(
                "edge_split operand={} edge={} params={:?}",
                split.operand, split.edge_id, split.parameters
            );
        }
    }
    diagnostics.measure_max(
        "timing.imprint_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    diagnostics.count_n("intersect.pieces", imprint.pieces.len() as u64);
    diagnostics.count_n("intersect.vertices", imprint.vertices.len() as u64);
    diagnostics.count_n("intersect.edge_splits", imprint.edge_splits.len() as u64);
    let stage_started = Instant::now();
    let (split_first, split_map_first) = apply_edge_splits_with_map(first, 0, &imprint)?;
    let (split_second, split_map_second) = apply_edge_splits_with_map(second, 1, &imprint)?;
    diagnostics.measure_max(
        "timing.edge_split_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    let stage_started = Instant::now();
    let fragments_a = fragment_solid(&split_first, 0, &imprint)?;
    let fragments_b = fragment_solid(&split_second, 1, &imprint)?;
    diagnostics.measure_max(
        "timing.fragment_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    diagnostics.count_n(
        "fragment.candidates",
        (fragments_a.len() + fragments_b.len()) as u64,
    );
    // Operand surface samples for the LEGITIMATE-EMPTY adjudication below:
    // every fragment test point is a 3D point ON its operand's boundary and
    // inside its face trim, so they witness where the operands' materials sit
    // relative to each other without extra geometry work.
    let a_surface_samples: Vec<Vec3> = fragments_a
        .iter()
        .flat_map(|fragment| {
            std::iter::once(fragment.test_point).chain(fragment.extra_test_points.iter().copied())
        })
        .collect();
    let b_surface_samples: Vec<Vec3> = fragments_b
        .iter()
        .flat_map(|fragment| {
            std::iter::once(fragment.test_point).chain(fragment.extra_test_points.iter().copied())
        })
        .collect();
    let stage_started = Instant::now();
    let mut barrier_edges: HashSet<(u8, u64)> = imprint
        .pieces
        .iter()
        .filter_map(|piece| piece.shared_edge.map(|(operand, edge_id, _)| (operand, edge_id)))
        .collect();
    barrier_edges.extend(imprint.barrier_edges.iter().copied());
    // The barrier set is keyed on ORIGINAL edge ids, but a barrier edge the
    // imprint also SPLIT (rotated equator-tangency: the overlapped cap ring
    // gains a junction at the sphere-seam crossing) reaches fragment
    // selection as its minted sub-edge ids. Remap the barrier through the
    // split ledger — keeping the originals too for unsplit references.
    for (operand, split_map) in [(0u8, &split_map_first), (1u8, &split_map_second)] {
        let minted: Vec<(u8, u64)> = barrier_edges
            .iter()
            .filter(|(barrier_operand, _)| *barrier_operand == operand)
            .filter_map(|(_, edge_id)| split_map.get(edge_id))
            .flat_map(|sub_ids| sub_ids.iter().map(|sub_id| (operand, *sub_id)))
            .collect();
        barrier_edges.extend(minted);
    }
    let selected = select_fragments(
        fragments_a,
        fragments_b,
        first,
        second,
        operation,
        tolerance,
        &barrier_edges,
    )?;
    diagnostics.measure_max(
        "timing.select_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    diagnostics.count_n("select.fragments", selected.len() as u64);
    // LEGITIMATE-EMPTY RESULTS: an intersect of DISJOINT operands (or a
    // subtract whose left operand is entirely CONSUMED by the right) selects
    // zero fragments, and assembly used to refuse with "operation produced no
    // boundary faces" — 384 of the 474 post-wrapped-band pool errors were
    // this, not bugs. Empty is only blessed on TWO independent proofs:
    //   (a) the imprint recorded NO section evidence (no accepted pierce
    //       seed, no traced SSI branch, no minted piece) — a silently-lost
    //       section (the trial-489 class) always leaves upstream evidence
    //       even when every downstream sub-segment is clipped away, while
    //       pure material disjointness leaves none (verified: 489's
    //       recreated silent-miss state keeps erroring; per-face surface
    //       samples ALONE missed its 235 mm³ pocket, which is why (a) is
    //       required and sampling alone was rejected);
    //   (b) the operands' surface samples agree (every fragment test point
    //       classified against the other solid):
    //       intersect — no sample of either operand strictly inside the
    //       other; subtract — no left-operand sample strictly outside the
    //       right.
    // Contact/graze pairs (evidence exists, material still disjoint) stay
    // errors — conservative by design. Escape hatch: BREP_EMPTY_BOOLEAN=0
    // restores the unconditional error.
    if selected.is_empty()
        && !imprint.section_evidence
        && std::env::var("BREP_EMPTY_BOOLEAN").as_deref() != Ok("0")
    {
        let strictly_inside = |samples: &[Vec3], other: &BrepSolid| -> Result<bool, KernelRefusal> {
            for &sample in samples {
                if classify_point(sample, other, tolerance).or_refuse(KernelStage::Validate, "classify_point")?.class == PointClass::In {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        let outside_witness = |samples: &[Vec3], other: &BrepSolid| -> Result<bool, KernelRefusal> {
            for &sample in samples {
                if classify_point(sample, other, tolerance).or_refuse(KernelStage::Validate, "classify_point")?.class == PointClass::Out {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        let legitimate = match operation {
            BooleanOperation::Intersect => {
                !strictly_inside(&a_surface_samples, second)?
                    && !strictly_inside(&b_surface_samples, first)?
            }
            BooleanOperation::Subtract => !outside_witness(&a_surface_samples, second)?,
            BooleanOperation::Union => false,
        };
        if legitimate {
            diagnostics.event(
                DiagnosticSeverity::Info,
                KernelStage::Select,
                "boolean.empty_result",
                format!("{operation:?} of witnessed-non-overlapping operands is empty"),
            );
            diagnostics.measure_max(
                "timing.total_ms",
                operation_started.elapsed().as_secs_f64() * 1_000.0,
            );
            return Ok(KernelOutcome {
                value: BrepSolid {
                    id: 0,
                    vertices: Vec::new(),
                    edges: Vec::new(),
                    shells: Vec::new(),
                    genus: 0,
                },
                diagnostics,
            });
        }
    }
    let solids = [(0, &split_first), (1, &split_second)]
        .into_iter()
        .collect::<HashMap<_, _>>();
    let stage_started = Instant::now();
    let solid = assemble_fragments(selected, &solids, &imprint, tolerance)?;
    diagnostics.measure_max(
        "timing.assemble_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    let stage_started = Instant::now();
    if std::env::var("BREP_DEBUG_PREMERGE_VALIDATE").is_ok() {
        let issues = solid.validate();
        eprintln!(
            "pre-merge validation: {} issue(s){}",
            issues.len(),
            issues
                .first()
                .map(|issue| format!(" — first: {}", issue.message))
                .unwrap_or_default()
        );
    }
    let solid = if options.merge_coplanar_faces
        && std::env::var("BREP_COPLANAR_MERGE").as_deref() != Ok("0")
    {
        let merged = merge_same_surface_faces_excluding(
            &solid,
            tolerance,
            &options.keep_unmerged_name_substrs,
        )?;
        // Face merging can make previously separate collinear boundary
        // segments incident to the same pair of faces. Run continuation
        // cleanup again, matching the post-merge normalization performed by
        // the former assembly pipeline.
        merge_curve_continuation_edges(&merged, tolerance).or_refuse(KernelStage::Validate, "merge_curve_continuation_edges")?
    } else {
        solid
    };
    diagnostics.measure_max(
        "timing.face_merge_ms",
        stage_started.elapsed().as_secs_f64() * 1_000.0,
    );
    let validation = solid.validate_detailed(&policy);
    diagnostics.count_n("validate.issues", validation.issues.len() as u64);
    diagnostics.count_n(
        "validate.wire_warnings",
        validation.wire_warnings.len() as u64,
    );
    diagnostics.measure_max("validate.max_pcurve_error", validation.max_pcurve_error);
    for warning in &validation.wire_warnings {
        diagnostics.event(
            DiagnosticSeverity::Warning,
            KernelStage::Validate,
            "validate.uv_wire",
            warning.message.clone(),
        );
    }
    for issue in &validation.issues {
        diagnostics.event(
            DiagnosticSeverity::Error,
            KernelStage::Validate,
            "validate.brep",
            issue.message.clone(),
        );
    }
    diagnostics.measure_max(
        "timing.total_ms",
        operation_started.elapsed().as_secs_f64() * 1_000.0,
    );
    if !validation.issues.is_empty() {
        return Err(KernelRefusal::new(
            RefusalClass::InvalidResultTopology { issues: validation.issues.len() as u32 },
            KernelStage::Validate,
            format!(
            "boolean_operation: invalid result: {:?}",
            validation.issues
        )));
    }
    // Optional semantic oracle (off by default, no perf hit): under
    // BREP_DEBUG_BOOL, cross-check the well-formed result against the CSG
    // point-membership expectation and scan for residual coincident-but-unwelded
    // geometry. Purely diagnostic and NON-rejecting — a statistical check must
    // never fail a valid op (see oracle.rs) — so it only ever warns.
    if std::env::var("BREP_DEBUG_BOOL").is_ok() {
        if let Ok(report) =
            crate::oracle::boolean_semantic_disagreement(first, second, operation, &solid, 200)
        {
            if report.is_flagged() {
                eprintln!(
                    "[oracle] semantic disagreement rate {:.4} ({} of {} decidable points); sample {:?}",
                    report.disagreement_rate,
                    report.disagreements.len(),
                    report.considered,
                    report.sample_disagreement()
                );
            }
        }
        let fusable_band = crate::tolerance::assembler_weld(policy.model);
        let fusables = crate::oracle::boolean_residual_fusables(&solid, fusable_band);
        if !fusables.is_empty() {
            eprintln!(
                "[oracle] {} residual fusable(s) after weld; sample {:?}",
                fusables.len(),
                fusables.first()
            );
        }
    }
    Ok(KernelOutcome {
        value: solid,
        diagnostics,
    })
}

