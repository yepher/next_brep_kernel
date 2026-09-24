//! The ONE definition of "the offset of a surface at a parameter".
//!
//! Push-face, offset-shell, thicken and the blend march are all surface
//! *offsetting* problems, and each of them had written the same three lines by
//! hand: evaluate the carrier, take a unit normal, step `distance` along it.
//! The audit
//! ([offset-unification-audit.md](../../../docs/developer/kernel-plans/offset-unification-audit.md))
//! records four independent copies; a fifth (`blend/edge/keep.rs`) and two more
//! in offset-shell turned up while this module was written. They are collected
//! here.
//!
//! # What this module is NOT
//!
//! It is **not** [`crate::offset_surface`]. That function is a Greville *fit*:
//! it samples the offset at the source basis's Greville parameters and
//! re-interpolates. Two independent reasons a fit is the wrong shared seam are
//! recorded in-tree — handing a fitted surface to the blend march's Newton loop
//! would inject the fit's approximation error into a residual whose bar is
//! `1e-11·(1+scale)` (`blending/blend/stations.rs`), and
//! `edit/direct_edit/face_offset_sphere.rs` records that `offset_surface` was
//! tried for the full-sphere push and *failed*, because the Greville fit
//! degenerates at the poles and the result stops re-recognising as a sphere.
//! The shared foundation is the **pointwise evaluator**; `offset_surface` is one
//! of its consumers (a fit of it), not the other way round.
//!
//! # The sign convention
//!
//! There is exactly ONE here: **`distance` is signed ALONG the returned
//! normal.** `point = source + distance · normal`. The kernel's other
//! convention — `offset_surface`'s "positive distance moves *opposite* the
//! face's outward normal" — is expressed by its adapter negating on the way in,
//! at one labelled place, instead of by four unlabelled hand negations
//! (audit §4.1).
//!
//! # The orientation conventions
//!
//! Orientation is an explicit parameter, never read off a `FaceRecord`, because
//! the callers genuinely disagree (audit §4.2) and both readings are
//! load-bearing. [`OffsetNormal`] has exactly the three that exist in the tree,
//! and each is defined to be *bit-identical* to the formula it replaces — see
//! its variant docs.

use crate::{AnalyticSurface, NurbsSurface, Vec3};

/// Which unit normal the offset rides, and how it is recovered where the
/// parametric normal degenerates.
///
/// Every variant names an existing in-tree formula and is bit-identical to it.
/// They differ ONLY where the parameter leaves the domain or the cross product
/// `S_u × S_v` collapses; a shared evaluator that silently picked one for
/// everybody would move behaviour at exactly the singular points the callers
/// each decided about on purpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetNormal {
    /// The bare `S_u × S_v`, normalized, through the C¹ domain extension
    /// (`deriv1_extended`). Orientation is the *caller's* business — the blend
    /// march carries it in the sign of ρ instead (`signed_radii`), which is why
    /// this variant must not apply `same_sense`.
    ///
    /// No singular recovery: where the cross product collapses this is an
    /// error, deliberately, so the march keeps refusing exactly where it
    /// refuses today.
    ///
    /// Bit-identical to `blending::blend::stations::raw_normal` (and to
    /// `evaluate_extended` for the point).
    Raw,
    /// `NurbsSurface::normal` — the in-domain `derivatives(u, v, 1)` cross —
    /// negated when `same_sense` is false. Clamped, not extended, and no
    /// singular recovery.
    ///
    /// Bit-identical to the `let mut normal = surface.normal(u, v)?; if
    /// !face.same_sense { normal = normal.scale(-1.0) }` block written out in
    /// `face_offset_revolution.rs`, `face_offset_freeform.rs`,
    /// `face_offset_torus.rs` and `offset_shell/smooth_sync.rs::face_normal`.
    Face { same_sense: bool },
    /// [`OffsetNormal::Face`] plus the singular-row recovery that
    /// `offset_surface` depends on: nudge off a point where the normal is
    /// undefined, and at a *collapsed parameter row* (a cone apex) walk deep
    /// inward for the true per-ruling limit rather than accept the axis
    /// direction the at-point cross degenerates to.
    ///
    /// Bit-identical to the former `offset::stable_face_normal`, whose body
    /// this is. The blend march must NOT inherit it — the recovery would
    /// change the residual at poles, so it stays opt-in (audit slice 1,
    /// "keep it as an opt-in flag").
    FaceStable { same_sense: bool },
    /// The exact closed-form normal of the *recognised analytic carrier*,
    /// oriented like [`OffsetNormal::Face`].
    ///
    /// This is the "exact per analytic surface type" lane: a sphere's normal is
    /// `(p − centre)/r` — defined at the poles, where `S_u × S_v` vanishes — a
    /// cylinder's is its radial direction, a cone's is the meridian
    /// perpendicular (defined even AT the apex, from the ruling's own azimuth),
    /// a torus's is `(p − tube centre)/r`. Where there is no exact form (the
    /// general `Revolution`, and every free-form patch) it falls back to
    /// [`OffsetNormal::Face`], so this variant is always at least as defined as
    /// that one.
    ///
    /// `NurbsSurface::analytic` memoizes per instance, so recognition is paid
    /// once per surface, not once per point.
    ///
    /// **No consumer rides this lane yet.** It exists as the exact half of the
    /// evaluator's contract and as the diagnostic's comparator; switching a
    /// caller onto it changes that caller's last digits and is a separate,
    /// measured step. See [`offset_normal_diagnostic`].
    ExactAnalytic { same_sense: bool },
}

impl OffsetNormal {
    /// The `same_sense`-carrying variants, for a caller that has a face.
    fn same_sense(self) -> bool {
        match self {
            OffsetNormal::Raw => true,
            OffsetNormal::Face { same_sense }
            | OffsetNormal::FaceStable { same_sense }
            | OffsetNormal::ExactAnalytic { same_sense } => same_sense,
        }
    }
}

/// One evaluation of the offset: the source point, the unit normal the offset
/// rides, and the offset point itself.
///
/// The normal is also the *offset surface's own* normal wherever the offset is
/// regular — it flips only past the evolute, which is what
/// `thicken::ensure_offsets_regular` (the kernel's only curvature gate) exists
/// to refuse. This struct deliberately carries no other derivative data: the
/// five consumers were audited and not one of them uses `S_u`/`S_v` for
/// anything but the normal, and the blend march's Jacobian is finite-differenced
/// on the residual rather than assembled from partials.
#[derive(Clone, Copy, Debug)]
pub struct OffsetSample {
    /// `S(u, v)` on the source carrier.
    pub source: Vec3,
    /// The unit normal, in the requested orientation.
    pub normal: Vec3,
    /// `source + distance · normal`.
    pub point: Vec3,
}

/// The pointwise offset evaluator, bound to one carrier and one orientation
/// convention.
///
/// Construction is free — it stores three references and reads nothing — so a
/// caller inside a Newton loop may build one per call without paying anything.
/// `site` is a short stable label used only by [`offset_normal_diagnostic`].
#[derive(Clone, Copy)]
pub struct OffsetEvaluator<'a> {
    site: &'static str,
    surface: &'a NurbsSurface,
    convention: OffsetNormal,
}

impl<'a> OffsetEvaluator<'a> {
    pub fn new(site: &'static str, surface: &'a NurbsSurface, convention: OffsetNormal) -> Self {
        Self {
            site,
            surface,
            convention,
        }
    }

    /// The unit normal alone, for callers that only need the direction (a
    /// residual check, or an affine offset that shifts the whole control net by
    /// one vector). Evaluates the point only on the lanes that need it.
    pub fn normal(&self, u: f64, v: f64) -> Result<Vec3, String> {
        let result = self.normal_with(self.convention, u, v);
        offset_normal_diagnostic(self.site, self.surface, self.convention, u, v, &result);
        result
    }

    /// The offset of this carrier at `(u, v)` by `distance` **along the
    /// normal** (see the module's sign convention).
    pub fn at(&self, u: f64, v: f64, distance: f64) -> Result<OffsetSample, String> {
        let (source, normal) = self.evaluate(u, v)?;
        Ok(OffsetSample {
            source,
            normal,
            point: source.add(normal.scale(distance)),
        })
    }

    fn evaluate(&self, u: f64, v: f64) -> Result<(Vec3, Vec3), String> {
        let result = match self.convention {
            // ONE evaluation for the point and the partials the normal needs —
            // `deriv1_extended`'s point is bit-identical to `evaluate_extended`,
            // which is what licenses `blend/edge/keep.rs` to drop its second
            // evaluation of the same surface.
            OffsetNormal::Raw => self
                .surface
                .deriv1_extended(u, v)
                .and_then(|(point, su, sv)| Ok((point, su.cross(sv).normalized()?))),
            convention => self
                .surface
                .evaluate(u, v)
                .and_then(|point| Ok((point, self.normal_with(convention, u, v)?))),
        };
        let normal = result
            .as_ref()
            .map(|(_, normal)| *normal)
            .map_err(Clone::clone);
        offset_normal_diagnostic(self.site, self.surface, self.convention, u, v, &normal);
        result
    }

    fn normal_with(&self, convention: OffsetNormal, u: f64, v: f64) -> Result<Vec3, String> {
        match convention {
            OffsetNormal::Raw => {
                let (_, su, sv) = self.surface.deriv1_extended(u, v)?;
                su.cross(sv).normalized()
            }
            OffsetNormal::Face { same_sense } => {
                Ok(orient(self.surface.normal(u, v)?, same_sense))
            }
            OffsetNormal::FaceStable { same_sense } => {
                stable_normal(self.surface, same_sense, u, v)
            }
            OffsetNormal::ExactAnalytic { same_sense } => {
                let point = self.surface.evaluate(u, v)?;
                match exact_analytic_normal(self.surface, u, v, point)? {
                    Some(normal) => Ok(orient(normal, same_sense)),
                    None => Ok(orient(self.surface.normal(u, v)?, same_sense)),
                }
            }
        }
    }
}

fn orient(normal: Vec3, same_sense: bool) -> Vec3 {
    if same_sense {
        normal
    } else {
        normal.scale(-1.0)
    }
}

fn domains(surface: &NurbsSurface) -> Result<([f64; 2], [f64; 2]), String> {
    Ok((surface.domain_u()?, surface.domain_v()?))
}

/// The body of the former `offset::stable_face_normal`, verbatim, with the
/// `FaceRecord` replaced by the `same_sense` flag it read (audit §4.5: the
/// `FaceRecord` requirement is a type mismatch, not a geometric one — it is why
/// `thicken` fabricates a synthetic face with empty loops just to offset a bare
/// surface).
fn stable_normal(
    surface: &NurbsSurface,
    same_sense: bool,
    u: f64,
    v: f64,
) -> Result<Vec3, String> {
    let normal_at = |u, v| surface.normal(u, v).ok();
    let mut normal = normal_at(u, v);
    let ([u0, u1], [v0, v1]) = domains(surface)?;
    if normal.is_none() {
        let du = (u1 - u0) * 1e-5;
        let dv = (v1 - v0) * 1e-5;
        for (candidate_u, candidate_v) in [
            ((u + du).clamp(u0, u1), v),
            ((u - du).clamp(u0, u1), v),
            (u, (v + dv).clamp(v0, v1)),
            (u, (v - dv).clamp(v0, v1)),
        ] {
            normal = normal_at(candidate_u, candidate_v);
            if normal.is_some() {
                break;
            }
        }
    }
    // SINGULAR-ROW override: at a surface singularity where a whole
    // parameter row collapses to one point (a cone apex), the at-point /
    // nudged normal is the cross of a vanishing partial with noise — the
    // AXIS direction instead of the ruling normal, which offsets the apex
    // row straight down the axis and bends the fitted surface by exactly
    // d·cos(half-angle). Detect the collapse by local point spread, walk
    // DEEP inward for the true per-ruling limit, and replace the at-point
    // value only when the two genuinely DISAGREE. A sphere/dome pole also
    // reads as collapsed, but there the at-point normal (the axis) IS the
    // limit — agreement keeps the exact baseline value.
    //
    // (`OffsetNormal::ExactAnalytic` answers this case in closed form for a
    // recognised cone — the ruling's own azimuth, not a deep-inward probe —
    // but no consumer rides that lane yet, so this stays the `offset_surface`
    // behaviour it always was.)
    let singular_here = {
        let du = (u1 - u0) * 1e-4;
        let dv = (v1 - v0) * 1e-4;
        let here = surface.evaluate(u, v)?;
        let along_u = surface
            .evaluate((u + du).clamp(u0, u1), v)?
            .sub(here)
            .length()
            .max(
                surface
                    .evaluate((u - du).clamp(u0, u1), v)?
                    .sub(here)
                    .length(),
            );
        let along_v = surface
            .evaluate(u, (v + dv).clamp(v0, v1))?
            .sub(here)
            .length()
            .max(
                surface
                    .evaluate(u, (v - dv).clamp(v0, v1))?
                    .sub(here)
                    .length(),
            );
        let scale = along_u.max(along_v);
        scale > 0.0 && along_u.min(along_v) < scale * 1e-6
    };
    if singular_here {
        let v_mid = (v0 + v1) * 0.5;
        let u_mid = (u0 + u1) * 0.5;
        let mut interior = None;
        for fraction in [1e-3, 1e-2, 5e-2, 0.25] {
            let candidate_v = v + (v_mid - v) * fraction;
            let candidate_u = u + (u_mid - u) * fraction;
            for (cu, cv) in [(u, candidate_v), (candidate_u, v), (candidate_u, candidate_v)] {
                if let Some(candidate) = normal_at(cu, cv) {
                    interior = Some(candidate);
                    break;
                }
            }
            if interior.is_some() {
                break;
            }
        }
        normal = match (normal, interior) {
            (Some(at_point), Some(interior)) if at_point.dot(interior) > 1.0 - 1e-6 => {
                Some(at_point)
            }
            (_, Some(interior)) => Some(interior),
            (at_point, None) => at_point,
        };
    }
    let normal =
        normal.ok_or_else(|| "offset_surface: cannot determine surface normal".to_string())?;
    Ok(orient(normal, same_sense))
}

// ---------------------------------------------------------------------------
// The exact analytic lane
// ---------------------------------------------------------------------------

/// The exact unit normal of a recognised analytic carrier at `point`, in the
/// carrier's own canonical orientation *calibrated to agree with `S_u × S_v`*.
///
/// Returns `Ok(None)` when there is no exact closed form for this carrier (the
/// general `Revolution`, and every unrecognised free-form patch), so the caller
/// can fall back.
///
/// **Sign.** The natural closed forms (a sphere's outward radial, a cone's
/// meridian perpendicular) have no fixed sign relation to the patch's
/// parametric normal — a `make_revolution` product's `S_u × S_v` points inward
/// or outward depending on how the generatrix was oriented. So the exact
/// direction is calibrated against the parametric normal once, at `(u, v)` if
/// it is regular there and at the domain midpoint otherwise. If no regular
/// probe exists anywhere tried, there is nothing to calibrate against and the
/// lane declines.
fn exact_analytic_normal(
    surface: &NurbsSurface,
    u: f64,
    v: f64,
    point: Vec3,
) -> Result<Option<Vec3>, String> {
    let Some(analytic) = surface.analytic() else {
        return Ok(None);
    };
    let Some(direction) = exact_direction(surface, analytic, u, v, point)? else {
        return Ok(None);
    };
    let ([u0, u1], [v0, v1]) = domains(surface)?;
    // Calibration probes: here first (free agreement when the patch is regular
    // at the query), then the domain midpoint, then two off-centre staggers
    // that miss a seam or a pole row.
    let probes = [
        (u, v),
        ((u0 + u1) * 0.5, (v0 + v1) * 0.5),
        (u0 + (u1 - u0) * 0.37, v0 + (v1 - v0) * 0.41),
        (u0 + (u1 - u0) * 0.63, v0 + (v1 - v0) * 0.59),
    ];
    for (pu, pv) in probes {
        let Ok(parametric) = surface.normal(pu, pv) else {
            continue;
        };
        let probe_point = if (pu, pv) == (u, v) {
            point
        } else {
            surface.evaluate(pu, pv)?
        };
        let Some(probe_direction) = exact_direction(surface, analytic, pu, pv, probe_point)? else {
            continue;
        };
        let alignment = probe_direction.dot(parametric);
        // A probe whose two readings are near-orthogonal is not a calibration —
        // it is a degenerate row read as noise. Demand a decisive sign.
        if alignment.abs() < 0.5 {
            continue;
        }
        return Ok(Some(if alignment > 0.0 {
            direction
        } else {
            direction.scale(-1.0)
        }));
    }
    Ok(None)
}

/// The closed-form normal direction (unit, canonical orientation, uncalibrated)
/// of one analytic carrier at a point known to lie on it.
fn exact_direction(
    surface: &NurbsSurface,
    analytic: &AnalyticSurface,
    u: f64,
    v: f64,
    point: Vec3,
) -> Result<Option<Vec3>, String> {
    Ok(match analytic {
        AnalyticSurface::Plane { u_dir, v_dir, .. } => u_dir.cross(*v_dir).normalized().ok(),
        AnalyticSurface::Sphere { frame, .. } => {
            // Exact AT the poles, which is precisely where `S_u × S_v`
            // vanishes and the Greville fit that `face_offset_sphere.rs`
            // rejected went wrong.
            point.sub(frame.origin).normalized().ok()
        }
        AnalyticSurface::RuledRevolution {
            frame,
            rho0,
            rho1,
            height,
        } => {
            let radial = match radial_direction(surface, frame.origin, frame.axis, u, v, point)? {
                Some(radial) => radial,
                None => return Ok(None),
            };
            // Meridian generatrix runs (rho0, 0) → (rho1, height) in
            // (radial, axial); the in-meridian perpendicular is
            // (height, −(rho1 − rho0)) over its length.
            let dr = rho1 - rho0;
            let length = (dr * dr + height * height).sqrt();
            if length <= 0.0 {
                return Ok(None);
            }
            radial
                .scale(*height / length)
                .sub(frame.axis.scale(dr / length))
                .normalized()
                .ok()
        }
        AnalyticSurface::Torus {
            frame,
            major_radius,
            minor_radius,
        } => {
            let radial = match radial_direction(surface, frame.origin, frame.axis, u, v, point)? {
                Some(radial) => radial,
                None => return Ok(None),
            };
            if *minor_radius <= 0.0 {
                return Ok(None);
            }
            let tube_centre = frame.origin.add(radial.scale(*major_radius));
            point.sub(tube_centre).normalized().ok()
        }
        // A general revolution's exact normal needs the generatrix tangent at
        // the meridian station, which is a 1D projection — not closed form, and
        // no cheaper than `S_u × S_v`. Decline rather than pretend.
        AnalyticSurface::Revolution { .. } => None,
    })
}

/// Unit radial direction of `point` about the axis.
///
/// On the axis itself (a cone apex, or a sphere/torus degeneracy) the point
/// carries no azimuth — but the *parameter* still does, because `u` names the
/// ruling. Recover it from the opposite end of the same `u` iso-line, which is
/// the exact per-ruling limit the `FaceStable` recovery only approximates by
/// walking inward.
fn radial_direction(
    surface: &NurbsSurface,
    origin: Vec3,
    axis: Vec3,
    u: f64,
    v: f64,
    point: Vec3,
) -> Result<Option<Vec3>, String> {
    let radial_of = |point: Vec3| {
        let relative = point.sub(origin);
        relative.sub(axis.scale(relative.dot(axis)))
    };
    let here = radial_of(point);
    let [v0, v1] = surface.domain_v()?;
    let scale = point.sub(origin).length().max(1.0);
    if here.length() > 1e-12 * scale {
        return Ok(here.normalized().ok());
    }
    let far = if (v - v0).abs() >= (v1 - v).abs() {
        v0
    } else {
        v1
    };
    let candidate = radial_of(surface.evaluate(u, far)?);
    if candidate.length() <= 1e-12 * scale {
        return Ok(None);
    }
    Ok(candidate.normalized().ok())
}

// ---------------------------------------------------------------------------
// Migration diagnostic
// ---------------------------------------------------------------------------

/// Per-site agreement between the normal lanes, aggregated over one thread.
///
/// Enabled by `BREP_OFFSET_DIAG`; the value is a sampling stride (`1` = every
/// call). Printing per call is useless here — one blend march makes hundreds of
/// thousands of evaluations — so the comparison is accumulated and dumped when
/// the thread ends, which for `cargo test` is once per test.
///
/// A site with `calls` but zero disagreement is evidence the corpus does not
/// *exercise* the difference, not proof the lanes are equivalent; a site that
/// never appears was never reached at all. Both readings matter, so the counts
/// are reported alongside the magnitudes.
#[derive(Default, Clone, Copy)]
struct LaneStats {
    compared: u64,
    /// Calls where this lane errored but the selected lane did not.
    lane_errors: u64,
    /// Calls where this lane succeeded and the selected lane did not.
    lane_rescues: u64,
    /// Worst CHORD distance between this lane's unit normal and the selected
    /// lane's, taken as an unsigned DIRECTION: `min(|a − b|, |a + b|)`.
    ///
    /// Chord, not `acos`: two bitwise identical unit vectors have a dot product
    /// one ULP below 1.0 as often as not (three rounded products summed), and
    /// `acos` turns that ULP into a flat 1.49e-8 rad floor that hides every
    /// real signal beneath it.  The chord is `2·sin(θ/2)`, so for a small angle
    /// it IS the angle in radians, and it is exactly 0 for identical vectors.
    ///
    /// Unsigned, because `Raw` carries no orientation while the `Face` lanes
    /// apply `same_sense`: on a reversed face they are exact opposites, and
    /// that is the CONVENTION difference (§4.2), not a geometry difference.
    /// The flips are counted separately so neither reading is lost.
    worst_chord: f64,
    /// Calls where this lane's normal pointed OPPOSITE the selected lane's.
    flipped: u64,
    worst_u: f64,
    worst_v: f64,
}

#[derive(Default, Clone, Copy)]
struct SiteStats {
    calls: u64,
    /// Calls where the lane the caller SELECTED could not produce a normal.
    errors: u64,
    convention: Option<OffsetNormal>,
    lanes: [LaneStats; 4],
}

const LANE_NAMES: [&str; 4] = ["raw", "face", "stable", "exact"];

fn lane_of(index: usize, same_sense: bool) -> OffsetNormal {
    match index {
        0 => OffsetNormal::Raw,
        1 => OffsetNormal::Face { same_sense },
        2 => OffsetNormal::FaceStable { same_sense },
        _ => OffsetNormal::ExactAnalytic { same_sense },
    }
}

struct DiagnosticSink {
    sites: std::collections::BTreeMap<&'static str, SiteStats>,
}

impl Drop for DiagnosticSink {
    fn drop(&mut self) {
        use std::fmt::Write as _;
        for (site, stats) in &self.sites {
            // Build the WHOLE line first and emit it with ONE `eprintln!`.
            // Threads exit concurrently under `cargo test`, and a report
            // assembled from a run of `eprint!`s interleaves mid-line with
            // another thread's — the first version of this lost ~60% of its
            // reports to exactly that, which looked like missing sites rather
            // than shredded ones.
            let mut line = format!(
                "offset-diag site={site} convention={} calls={} errors={}",
                stats
                    .convention
                    .map(|convention| format!("{convention:?}"))
                    .unwrap_or_else(|| "?".to_string()),
                stats.calls,
                stats.errors
            );
            for (index, lane) in stats.lanes.iter().enumerate() {
                if lane.compared == 0 && lane.lane_errors == 0 && lane.lane_rescues == 0 {
                    continue;
                }
                let _ = write!(
                    line,
                    " | {}: n={} chord={:.3e} at=({:.6},{:.6}) flip={} lane_err={} rescue={}",
                    LANE_NAMES[index],
                    lane.compared,
                    lane.worst_chord,
                    lane.worst_u,
                    lane.worst_v,
                    lane.flipped,
                    lane.lane_errors,
                    lane.lane_rescues
                );
            }
            eprintln!("{line}");
        }
    }
}

fn diagnostic_stride() -> u64 {
    static STRIDE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *STRIDE.get_or_init(|| match std::env::var("BREP_OFFSET_DIAG") {
        Ok(value) => value.trim().parse::<u64>().unwrap_or(1).max(1),
        Err(_) => 0,
    })
}

thread_local! {
    static DIAGNOSTIC: std::cell::RefCell<DiagnosticSink> = std::cell::RefCell::new(DiagnosticSink {
        sites: std::collections::BTreeMap::new(),
    });
}

/// Compare every lane against the one the caller selected, and accumulate.
///
/// Off unless `BREP_OFFSET_DIAG` is set, and then the only cost on the hot path
/// is one `OnceLock` read and a modulo.
fn offset_normal_diagnostic(
    site: &'static str,
    surface: &NurbsSurface,
    convention: OffsetNormal,
    u: f64,
    v: f64,
    selected: &Result<Vec3, String>,
) {
    let stride = diagnostic_stride();
    if stride == 0 {
        return;
    }
    let sampled = DIAGNOSTIC.with(|sink| {
        let mut sink = sink.borrow_mut();
        let stats = sink.sites.entry(site).or_default();
        stats.convention = Some(convention);
        stats.calls += 1;
        if selected.is_err() {
            stats.errors += 1;
        }
        stats.calls % stride == 0
    });
    if !sampled {
        return;
    }
    let evaluator = OffsetEvaluator::new(site, surface, convention);
    let same_sense = convention.same_sense();
    let mut readings = [(0usize, f64::NAN, false, false); 4];
    for (index, reading) in readings.iter_mut().enumerate() {
        let lane = lane_of(index, same_sense);
        let value = evaluator.normal_with(lane, u, v);
        *reading = match (selected, &value) {
            (Ok(chosen), Ok(other)) => {
                let aligned = chosen.sub(*other).length();
                let opposed = chosen.add(*other).length();
                (index, aligned.min(opposed), true, opposed < aligned)
            }
            (Ok(_), Err(_)) => (index, f64::NAN, false, false),
            (Err(_), Ok(_)) => (index, f64::NAN, true, false),
            (Err(_), Err(_)) => (index, f64::NAN, false, false),
        };
    }
    DIAGNOSTIC.with(|sink| {
        let mut sink = sink.borrow_mut();
        let stats = sink.sites.entry(site).or_default();
        for (index, chord, lane_ok, flipped) in readings {
            let lane = &mut stats.lanes[index];
            if chord.is_finite() {
                lane.compared += 1;
                if flipped {
                    lane.flipped += 1;
                }
                if chord > lane.worst_chord {
                    lane.worst_chord = chord;
                    lane.worst_u = u;
                    lane.worst_v = v;
                }
            } else if selected.is_err() && lane_ok {
                lane.lane_rescues += 1;
            } else if selected.is_ok() && !lane_ok {
                lane.lane_errors += 1;
            }
        }
    });
}

// BREP private tests: f2fe80f6e7b543ac
