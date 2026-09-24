use super::*;

/// Sweep a CLOSED PLANAR profile loop along a path curve (§3.3/§5.7),
/// TRANSPLANTING the profile onto the path — see [`SectionPlacement`] for the
/// other reading, and for why the choice changes the solid.
///
/// Reuses the loft builder rather than a bespoke swept surface: the path is
/// sampled at `STATIONS` uniform stations, a rotation-minimizing frame is
/// propagated along it (double-reflection RMF, Wang et al. 2008 — the frame
/// does NOT spin at inflections the way raw Frenet does), a rigidly
/// transformed copy of the profile is placed at each station (identical
/// degree/knots/weights, only the control points moved — which guarantees
/// loft's per-section compatibility), and the stations are lofted through
/// to produce the tube plus planar end caps.
///
/// Guards return a clear `Err` on an open/non-planar profile, a degenerate
/// path tangent, or a loft failure. A self-intersecting result (path
/// curvature radius smaller than the profile extent) is OUT OF SCOPE — the
/// caller is responsible for keeping the tube from folding onto itself.
pub fn sweep_profile_along_path(
    profile: &[NurbsCurve],
    path: &NurbsCurve,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    // 32 stations is the original fixed sampling — golden parity pins the
    // emitted geometry to it.  The helix variant raises the count with the
    // turn count instead, hence the shared `_stations` core.
    sweep_profile_along_path_stations(
        profile,
        path,
        name,
        32,
        0.0,
        None,
        SectionPlacement::Transplant,
    )
}

/// Twisted path sweep (§5.7): identical to [`sweep_profile_along_path`], but
/// the profile additionally ROTATES about the path tangent, linearly in ARC
/// LENGTH, from 0 at the sweep start to `twist_angle` radians (right-handed
/// about the tangent) at the end.  The arc-length fraction comes from the
/// sampled station polyline, not the raw path parameter, so a non-uniformly
/// parameterized path still twists uniformly in space.
///
/// STATION LAW: 16 stations per quarter turn of twist, floored at the path
/// sweep's 32 and capped at 1024 (the loft's dense interpolation solve is
/// O(stations³) per control column — the same cap the helix uses).  At
/// 16/quarter-turn the inter-station twist step is Δφ ≈ 5.6°, so the cubic
/// v-interpolation error on a profile point circling at radius r is
/// ≈ r·Δφ⁴/384 ≈ 2.4·10⁻¹⁰·r — far below any geometric tolerance.  The cap
/// holds that density up to 16 full turns (|twist| = 32π = 1024/16 quarter
/// turns); a larger twist would silently alias under the cap, so it is
/// REJECTED with an honest error instead.  Everything else the path sweep
/// documents (profile validity, self-intersection being the caller's
/// responsibility) applies unchanged; twisting about the profile's own
/// centroid adds no new radial extent, so no extra collision guard exists
/// to compute here.
pub fn sweep_profile_twisted(
    profile: &[NurbsCurve],
    path: &NurbsCurve,
    twist_angle: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    use std::f64::consts::{FRAC_PI_2, TAU};

    if !twist_angle.is_finite() {
        return Err("sweep_profile_twisted: twist angle must be finite".into());
    }
    // 16 turns is where the 1024-station cap meets 16 stations/quarter-turn;
    // beyond it the cap would degrade the twist sampling density silently.
    const MAX_TURNS: f64 = 16.0;
    if twist_angle.abs() > MAX_TURNS * TAU {
        return Err(format!(
            "sweep_profile_twisted: twist of {:.3} turns exceeds the {MAX_TURNS}-turn \
             limit the 1024-station cap can resolve at 16 stations per quarter turn; \
             split the sweep or reduce the twist",
            twist_angle.abs() / TAU
        ));
    }
    let quarter_turns = (twist_angle.abs() / FRAC_PI_2).ceil() as usize;
    let stations = (quarter_turns * 16).clamp(32, 1024);
    sweep_profile_along_path_stations(
        profile,
        path,
        name,
        stations,
        twist_angle,
        None,
        SectionPlacement::Transplant,
    )
    .map_err(|error| format!("sweep_profile_twisted: {error}"))
}

/// HOW a section is placed at each station — the two things "sweep a profile
/// along a path" can mean, and the reason they are not the same shape.
///
/// `Transplant` MOVES the profile onto the path: the anchor origin lands on the
/// station point and the profile plane becomes the station's NORMAL plane. The
/// profile's own position and its angle to the path are both discarded — every
/// sweep comes out centred on the path and square to it. That is the classic
/// swept-surface reading, and it is what `SWP` builds.
///
/// `Rigid` CARRIES the profile: the section at station `k` is the profile moved
/// by the same rigid motion the PATH undergoes between its start and station
/// `k`,
///
/// ```text
///     section_k(x) = P_k + R_k · (x − P_0)
/// ```
///
/// where `R_k` is the rotation taking the start frame `(T_0, r_0, s_0)` to the
/// station's `(T_k, r_k, s_k)`. The profile keeps its drawn POSITION and its
/// drawn ANGLE to the path; what changes along the sweep is only what the path
/// itself does. Two consequences are the whole point of the mode:
///   - a STRAIGHT path makes every `R_k` the identity, so the section merely
///     translates and the result is the oblique prism `extrude_profile_brep`
///     builds from the same profile and vector — `pathAlign` and `translate`
///     agree on a straight path rather than merely having the same volume;
///   - a circular ARC makes `R_k` the rotation about the ARC'S OWN CENTRE AXIS
///     (for a planar path the RMF's transport rotation IS that rotation), so
///     the profile is carried round the pivot and the end cap arrives at the
///     angle the start cap had to the tangent. A full-circle path degenerates
///     to exactly a revolve of the profile about that axis.
///
/// `R_k = F_k · F_0ᵀ` does NOT depend on which perpendicular `r_0` the RMF was
/// seeded with: re-seeding rolls `F_0` and every `F_k` by the same angle about
/// the tangent, and the two rolls cancel in the product. That matters because
/// `Transplant` maps `pu → r_k` directly, so under IT the profile's roll about
/// the path is whatever `tangents[0].perpendicular()` happened to return — an
/// arbitrary orientation no caller can predict. `Rigid` has no such freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionPlacement {
    /// Move the profile onto the path, square to it (the classic sweep).
    Transplant,
    /// Carry the profile by the path's own rigid motion (see above).
    Rigid,
}

/// The placement anchor a path sweep transplants its profile with: the plane
/// frame `(origin = boundary-sample centroid, normal, pu, pv)` the station
/// loop maps profile points through (`local = p − origin` → `station + ri·(local
/// ·pu) + si·(local·pv)`). Extracted as data so a HOLE loop can sweep with its
/// OUTER loop's anchor — sweeping each loop with its OWN centroid would
/// re-center every loop onto the path and lose the hole's in-plane offset.
#[derive(Debug, Clone, Copy)]
pub struct ProfileAnchor {
    pub origin: Vec3,
    pub normal: Vec3,
    pub pu: Vec3,
    pub pv: Vec3,
}

/// [`sweep_profile_along_path`] with an explicit placement anchor (see
/// [`ProfileAnchor`]) — the hole-loop cutter path: the swept loop is validated
/// as usual but PLACED in its outer loop's frame.
pub fn sweep_profile_along_path_anchored(
    profile: &[NurbsCurve],
    path: &NurbsCurve,
    name: Option<&str>,
    anchor: ProfileAnchor,
) -> Result<BrepSolid, String> {
    sweep_profile_along_path_stations(
        profile,
        path,
        name,
        32,
        0.0,
        Some(anchor),
        SectionPlacement::Transplant,
    )
}

/// [`sweep_profile_twisted`] with an explicit placement anchor: the hole loop
/// twists about the SAME path axis as its outer loop (shared anchor), so the
/// cutter stays registered with the outer wall through the whole twist.
pub fn sweep_profile_twisted_anchored(
    profile: &[NurbsCurve],
    path: &NurbsCurve,
    twist_angle: f64,
    name: Option<&str>,
    anchor: ProfileAnchor,
) -> Result<BrepSolid, String> {
    use std::f64::consts::{FRAC_PI_2, TAU};

    if !twist_angle.is_finite() {
        return Err("sweep_profile_twisted: twist angle must be finite".into());
    }
    const MAX_TURNS: f64 = 16.0;
    if twist_angle.abs() > MAX_TURNS * TAU {
        return Err(format!(
            "sweep_profile_twisted: twist of {:.3} turns exceeds the {MAX_TURNS}-turn \
             limit the 1024-station cap can resolve at 16 stations per quarter turn; \
             split the sweep or reduce the twist",
            twist_angle.abs() / TAU
        ));
    }
    let quarter_turns = (twist_angle.abs() / FRAC_PI_2).ceil() as usize;
    let stations = (quarter_turns * 16).clamp(32, 1024);
    sweep_profile_along_path_stations(
        profile,
        path,
        name,
        stations,
        twist_angle,
        Some(anchor),
        SectionPlacement::Transplant,
    )
    .map_err(|error| format!("sweep_profile_twisted: {error}"))
}

/// Sweep a CLOSED PLANAR profile along a CHAIN of path curves joined
/// head-to-tail — the multi-segment entry point, with the twist and anchor
/// options the single-curve family exposes as separate functions folded into
/// one signature (the feature layer picks all three per loop, so splitting them
/// four ways here would only push the same match into the caller).
///
/// `segment_names` name the chain's segments for the joint refusal; a short or
/// empty slice degrades to `<unnamed>` rather than failing.
///
/// `corner_advice` is appended to the cornered-joint refusal and is the CALLER's
/// to write, because the builder has two callers with different answers: path
/// sweep sends the user to SW (a different feature), while SW's own `pathAlign`
/// sends them to its `translate` mode. A builder that named one feature would be
/// telling half its users to switch to the feature they are already in.
///
/// A ONE-segment chain delegates to the single-curve builders unchanged, so the
/// geometry every existing path sweep emits is untouched by this entry point
/// existing — the multi-segment sampler is reached only by a path that actually
/// has a joint.
///
/// Joints must be tangent-continuous; see [`MAX_JOINT_TANGENT_BREAK`] for why a
/// corner is refused here instead of rounded off, and where such a path belongs.
pub fn sweep_profile_along_chain(
    profile: &[NurbsCurve],
    chain: &[NurbsCurve],
    segment_names: &[String],
    twist_angle: f64,
    name: Option<&str>,
    anchor: Option<ProfileAnchor>,
    placement_mode: SectionPlacement,
    corner_advice: &str,
) -> Result<BrepSolid, String> {
    use std::f64::consts::{FRAC_PI_2, TAU};

    if chain.is_empty() {
        return Err("sweepSolid: path chain is empty".into());
    }
    // Single segment under `Transplant`: the existing builders, byte for byte.
    // `Rigid` has no single-curve wrapper to delegate to and falls through to the
    // chain sampler, which samples a 1-element chain over exactly the domain the
    // single-curve sampler would (no joints, the whole budget on one segment).
    if let ([single], SectionPlacement::Transplant) = (chain, placement_mode) {
        return match (anchor, twist_angle == 0.0) {
            (None, true) => sweep_profile_along_path(profile, single, name),
            (Some(anchor), true) => {
                sweep_profile_along_path_anchored(profile, single, name, anchor)
            }
            (None, false) => sweep_profile_twisted(profile, single, twist_angle, name),
            (Some(anchor), false) => {
                sweep_profile_twisted_anchored(profile, single, twist_angle, name, anchor)
            }
        };
    }

    // Everything else — any multi-segment chain, and a single segment under
    // `Rigid` — takes the same station law the single-curve twisted builder
    // uses, so density scales with twist identically on a chained path.
    if !twist_angle.is_finite() {
        return Err("sweep_profile_along_chain: twist angle must be finite".into());
    }
    const MAX_TURNS: f64 = 16.0;
    if twist_angle.abs() > MAX_TURNS * TAU {
        return Err(format!(
            "sweep_profile_along_chain: twist of {:.3} turns exceeds the {MAX_TURNS}-turn \
             limit the 1024-station cap can resolve at 16 stations per quarter turn; \
             split the sweep or reduce the twist",
            twist_angle.abs() / TAU
        ));
    }
    let stations = if twist_angle == 0.0 {
        32
    } else {
        let quarter_turns = (twist_angle.abs() / FRAC_PI_2).ceil() as usize;
        (quarter_turns * 16).clamp(32, 1024)
    };
    let _ = name;
    sweep_profile_along_chain_stations(
        profile,
        chain,
        segment_names,
        stations,
        twist_angle,
        anchor,
        placement_mode,
        corner_advice,
    )
}

/// [`sweep_profile_along_chain`] with an explicit STATION budget and no twist
/// — the entry a caller with a long, smooth chain uses when the default 32
/// stations would under-sample it (a wire-harness bundle along a many-anchor
/// spline: three pieces per span, so a handful of spans already starves each
/// piece of stations). The budget is spread over the chain by arc length
/// exactly as the twisted builder's law spreads its own; the 1024-station cap
/// and the tangent-continuity refusal apply unchanged.
pub fn sweep_profile_along_chain_with_stations(
    profile: &[NurbsCurve],
    chain: &[NurbsCurve],
    segment_names: &[String],
    stations: usize,
    corner_advice: &str,
) -> Result<BrepSolid, String> {
    sweep_profile_along_chain_stations(
        profile,
        chain,
        segment_names,
        stations.clamp(2, 1024),
        0.0,
        None,
        SectionPlacement::Transplant,
        corner_advice,
    )
}

/// Validate a closed planar profile loop and derive its placement anchor —
/// the path sweep's §1 block, extracted bit-identically: 16 samples per curve,
/// closure at `tolerance`, Newell normal, boundary-sample-mean origin,
/// planarity at `tolerance * 100`, `pu = np.perpendicular()`, `pv = np × pu`.
pub fn profile_anchor(profile: &[NurbsCurve]) -> Result<ProfileAnchor, String> {
    let tolerance = 1e-6;
    if profile.len() < 2 {
        return Err("sweepSolid: profile needs at least 2 curves forming a closed loop".into());
    }
    let mut samples = Vec::new();
    for (index, curve) in profile.iter().enumerate() {
        let [start, end] = curve.domain()?;
        let next = &profile[(index + 1) % profile.len()];
        let next_start = next.domain()?[0];
        if curve
            .evaluate(end)?
            .sub(next.evaluate(next_start)?)
            .length()
            > tolerance
        {
            return Err(format!(
                "sweepSolid: profile is not closed at curve {index}"
            ));
        }
        for sample in 0..16 {
            samples.push(curve.evaluate(start + (end - start) * sample as f64 / 16.0)?);
        }
    }
    let normal = crate::polygon::newell_normal(&samples);
    let centroid = samples.iter().fold(Vec3::default(), |sum, &point| sum.add(point));
    let np = normal
        .normalized()
        .map_err(|_| "sweepSolid: profile is degenerate (zero enclosed area)".to_string())?;
    let origin = centroid.scale(1.0 / samples.len() as f64);
    if samples
        .iter()
        .any(|point| point.sub(origin).dot(np).abs() > tolerance * 100.0)
    {
        return Err("sweepSolid: profile is not planar".into());
    }
    let pu = np.perpendicular()?;
    let pv = np.cross(pu).normalized()?;
    Ok(ProfileAnchor {
        origin,
        normal: np,
        pu,
        pv,
    })
}

/// Core of the path sweep with an explicit station count.  Every consumer of
/// the station count (path sampling, RMF propagation, section placement)
/// derives from the one `stations` argument so the density scales as a unit.
/// `twist_angle` (radians; 0 for the untwisted variants) rotates the placed
/// profile about the path tangent linearly in sampled arc length — the
/// `twist_angle == 0.0` fast path leaves the RMF axes bit-identical, so the
/// untwisted callers keep golden parity.
fn sweep_profile_along_path_stations(
    profile: &[NurbsCurve],
    path: &NurbsCurve,
    name: Option<&str>,
    stations: usize,
    twist_angle: f64,
    anchor: Option<ProfileAnchor>,
    placement_mode: SectionPlacement,
) -> Result<BrepSolid, String> {
    // Loft carries no face names; the app stamps them onto the emitted face
    // order.  Accept `name` for ABI symmetry with the other builders.
    let _ = name;
    let tolerance = 1e-6;
    if stations < 2 {
        return Err("sweepSolid: need at least 2 stations".into());
    }

    // --- 1. Validate the profile: closed + planar; derive (origin, np, pu, pv).
    //        A caller-supplied anchor OVERRIDES the placement frame (the swept
    //        loop is still validated against its own plane), so a hole loop
    //        rides the path in its outer loop's frame instead of re-centering.
    //        Resolved HERE, before the path is touched, so a bad profile is
    //        still reported ahead of a bad path exactly as it always was.
    let placement = resolve_placement(profile, anchor, placement_mode)?;

    // --- 2. Sample the path; require a non-degenerate tangent at every station.
    let [t0, t1] = path.domain()?;
    if (t1 - t0).abs() <= tolerance {
        return Err("sweepSolid: path domain is degenerate".into());
    }
    let mut points = Vec::with_capacity(stations);
    let mut tangents = Vec::with_capacity(stations);
    for index in 0..stations {
        let t = t0 + (t1 - t0) * index as f64 / (stations - 1) as f64;
        let derivatives = path.derivatives(t, 1)?;
        let tangent = derivatives[1]
            .normalized()
            .map_err(|_| format!("sweepSolid: path tangent is degenerate at station {index}"))?;
        points.push(derivatives[0]);
        tangents.push(tangent);
    }

    sweep_sections_through_samples(
        profile,
        &points,
        &tangents,
        twist_angle,
        placement,
        placement_mode,
    )
}

/// The placement frame a sweep transplants its profile with: the caller's
/// `anchor` when it supplied one (a hole loop riding its outer loop's frame),
/// otherwise the profile's own. Either way the profile is VALIDATED — closed and
/// planar — because `profile_anchor` is what does that validating, and skipping
/// it for an anchored loop would let an open hole loop through.
///
/// Called by each SAMPLER before it touches the path, so the "bad profile" error
/// still precedes the "bad path" one, as it did when the two lived in one
/// function.
fn resolve_placement(
    profile: &[NurbsCurve],
    anchor: Option<ProfileAnchor>,
    placement_mode: SectionPlacement,
) -> Result<ProfileAnchor, String> {
    let computed = profile_anchor(profile)?;
    Ok(match placement_mode {
        // RIGID never transplants, so there is no frame to borrow: every loop
        // keeps its own position already, which is the whole reason the anchor
        // exists under `Transplant`. Taking a caller's anchor here would silently
        // swap in another loop's NORMAL for the fold-back guard below.
        SectionPlacement::Rigid => computed,
        SectionPlacement::Transplant => anchor.unwrap_or(computed),
    })
}

/// The largest tangent break, in radians, a chained path may have at a joint.
///
/// This is a REAL geometric boundary, not a tuning knob. The builder skins
/// consecutive stations by lofting through them, so a corner between two
/// stations is rendered as a smooth blend across the corner — the tube cuts it,
/// and near a sharp one the section sweeps through itself. A path sweep can
/// honestly build a G1 chain (line→fillet→line, spline pieces, an arc train)
/// and cannot honestly build a cornered polyline. `Sweep` (SW) is the feature
/// for a cornered path: it builds one prism per segment and unions them, which
/// is exactly the construction that gives a corner a real mitre.
///
/// ~1.15° (0.02 rad). Loose enough to absorb the tangent disagreement of two
/// curves a sketch chained on endpoint coincidence, tight enough that anything
/// a user would call a corner is refused rather than silently rounded off.
///
/// This is INDEPENDENT of the chainer's join tolerance, and deliberately so.
/// `common::chain_path_segments` joins on POSITION (`1e-5 * scale`); this gate
/// measures DIRECTION. Two collinear segments meeting with a positional gap have
/// a zero tangent break, and two segments meeting exactly at a point can still
/// break 90°. So a run the chainer happily orders head-to-tail may still be
/// refused here — which is not an inconsistency but the SW/SWP split itself: the
/// chainer's job is to decide what order the picks form a run in, and this gate's
/// job is to decide whether that run is one this builder can skin.
const MAX_JOINT_TANGENT_BREAK: f64 = 0.02;

/// Sweep a CLOSED PLANAR profile along a CHAIN of path curves, joined
/// head-to-tail, as one continuous tube.
///
/// The chain is sampled as a single trajectory and handed to the shared core,
/// so ONE rotation-minimizing frame is propagated across the whole path: a joint
/// is just another pair of adjacent stations to the RMF, which is what makes the
/// profile arrive at the far end in the orientation the near end implies rather
/// than snapping at each segment.
///
/// STATION BUDGET. `stations` are distributed across the segments in proportion
/// to estimated ARC LENGTH (chord sums over 16 samples per curve), floored at 2
/// per segment so a short fillet between two long lines still carries its
/// curvature. The shared endpoint of two adjacent segments is sampled ONCE, not
/// twice — a duplicated station would land in the RMF's coincident-step branch
/// and waste a section on zero advance.
///
/// JOINTS must be tangent-continuous to within [`MAX_JOINT_TANGENT_BREAK`];
/// a break beyond it is refused, naming the two segments and the angle.
fn sweep_profile_along_chain_stations(
    profile: &[NurbsCurve],
    chain: &[NurbsCurve],
    segment_names: &[String],
    stations: usize,
    twist_angle: f64,
    anchor: Option<ProfileAnchor>,
    placement_mode: SectionPlacement,
    corner_advice: &str,
) -> Result<BrepSolid, String> {
    let tolerance = 1e-6;
    if chain.is_empty() {
        return Err("sweepSolid: path chain is empty".into());
    }
    if stations < 2 {
        return Err("sweepSolid: need at least 2 stations".into());
    }
    // Profile first, matching the single-curve sampler's order of complaint.
    let placement = resolve_placement(profile, anchor, placement_mode)?;

    // --- Per-segment arc-length estimate, and the domain each one is sampled
    //     over. A degenerate domain is refused here rather than producing a
    //     silently collapsed station block.
    let mut domains = Vec::with_capacity(chain.len());
    let mut lengths = Vec::with_capacity(chain.len());
    for (index, curve) in chain.iter().enumerate() {
        let [t0, t1] = curve.domain()?;
        if (t1 - t0).abs() <= tolerance {
            return Err(format!(
                "sweepSolid: path segment {index} has a degenerate domain"
            ));
        }
        let mut length = 0.0;
        let mut previous = curve.evaluate(t0)?;
        for sample in 1..=16 {
            let point = curve.evaluate(t0 + (t1 - t0) * sample as f64 / 16.0)?;
            length += point.sub(previous).length();
            previous = point;
        }
        domains.push([t0, t1]);
        lengths.push(length);
    }
    let total_length: f64 = lengths.iter().sum();
    if total_length <= tolerance {
        return Err("sweepSolid: path chain has zero length".into());
    }

    // --- Station budget per segment: proportional to arc length, floored at 2
    //     (a segment needs its two ends), and the remainder handed to the
    //     longest segment so the totals land exactly on `stations`.
    let mut budget: Vec<usize> = lengths
        .iter()
        .map(|length| {
            let share = (stations as f64 * length / total_length).round() as usize;
            share.max(2)
        })
        .collect();
    // Each interior joint shares one station between its two segments, so the
    // emitted total is sum(budget) - (segments - 1).
    let joints = chain.len() - 1;
    let emitted: usize = budget.iter().sum::<usize>() - joints;
    // The 2-per-segment floor means a chain of n segments emits at least n+1
    // stations, so a long enough chain cannot be held under the loft's station
    // cap by any distribution. The loft's interpolation solve is O(stations³)
    // per control column, so silently accepting such a path would look like a
    // hang. Refuse with the count instead — the same honesty the twist cap uses.
    const MAX_STATIONS: usize = 1024;
    if emitted > MAX_STATIONS {
        return Err(format!(
            "sweepSolid: a {}-segment path needs at least {emitted} stations, past the \
             {MAX_STATIONS}-station cap the loft solve can carry; sweep it in fewer, \
             longer pieces",
            chain.len()
        ));
    }
    if emitted < stations {
        let longest = lengths
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        budget[longest] += stations - emitted;
    }

    // --- Sample the chain as ONE trajectory, checking tangent continuity at
    //     every joint as it is crossed.
    let mut points: Vec<Vec3> = Vec::new();
    let mut tangents: Vec<Vec3> = Vec::new();
    for (index, curve) in chain.iter().enumerate() {
        let [t0, t1] = domains[index];
        let count = budget[index];

        // The JOINT check runs at this segment's START parameter, which is the
        // point the previous segment ended on — comparing the tangent the path
        // arrives with against the one it departs with. Sampling the next
        // station instead would fold the segment's own curvature into the
        // measured break and let a genuine corner through on a curved segment.
        if index > 0 {
            let departing = curve.derivatives(t0, 1)?[1].normalized().map_err(|_| {
                format!("sweepSolid: path tangent is degenerate at the start of segment {index}")
            })?;
            let arriving = *tangents
                .last()
                .expect("a previous segment emitted stations");
            // Both unit, so the dot is the cosine of the break.
            let break_angle = arriving.dot(departing).clamp(-1.0, 1.0).acos();
            if break_angle > MAX_JOINT_TANGENT_BREAK {
                let previous_name = segment_names
                    .get(index - 1)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                let name = segment_names
                    .get(index)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                return Err(format!(
                    "the path must be tangent-continuous: segments '{previous_name}' and \
                     '{name}' meet at a {:.1}° corner. The profile is skinned between sampled \
                     stations along the path, so a corner would be ROUNDED OFF rather than \
                     mitred. {corner_advice}",
                    break_angle.to_degrees()
                ));
            }
        }

        // Skip the first sample of every segment after the first: that station
        // is the joint, already emitted as the previous segment's last.
        let first = usize::from(index > 0);
        for sample in first..count {
            let t = t0 + (t1 - t0) * sample as f64 / (count - 1) as f64;
            let derivatives = curve.derivatives(t, 1)?;
            let tangent = derivatives[1].normalized().map_err(|_| {
                format!("sweepSolid: path tangent is degenerate on segment {index}")
            })?;
            points.push(derivatives[0]);
            tangents.push(tangent);
        }
    }

    sweep_sections_through_samples(
        profile,
        &points,
        &tangents,
        twist_angle,
        placement,
        placement_mode,
    )
}

/// The part of the sweep that does not care HOW the stations were sampled:
/// validate the profile, distribute the twist over the sampled polyline,
/// propagate the rotation-minimizing frame, place a profile copy per station
/// and loft through them.
///
/// Split out so a MULTI-SEGMENT path can share it
/// ([`sweep_profile_along_chain_stations`]). The single-curve sampler above is
/// deliberately left calling this with its OWN sampling rather than being
/// re-expressed as a one-element chain: two code paths that agree today are not
/// the same thing as one code path, and the untwisted single-curve geometry is
/// pinned by volume tests that must not move.
///
/// The RMF is propagated over the station sequence as a whole, so a chain's
/// frame carries across a joint exactly as it carries across any other pair of
/// adjacent stations — there is no per-segment restart to reconcile.
fn sweep_sections_through_samples(
    profile: &[NurbsCurve],
    points: &[Vec3],
    tangents: &[Vec3],
    twist_angle: f64,
    placement: ProfileAnchor,
    placement_mode: SectionPlacement,
) -> Result<BrepSolid, String> {
    let tolerance = 1e-6;
    let stations = points.len();
    if stations < 2 {
        return Err("sweepSolid: need at least 2 stations".into());
    }

    let ProfileAnchor { origin, pu, pv, .. } = placement;

    // --- 2b. Twist distribution: cumulative ARC-LENGTH fractions over the
    //         sampled station polyline (chord sums), so the twist advances
    //         uniformly in space even on a non-uniformly parameterized path.
    //         Only computed when a twist is actually requested — the
    //         `twist_angle == 0.0` path must stay bit-identical to the
    //         pre-twist builder.
    let twist_fractions: Option<Vec<f64>> = if twist_angle != 0.0 {
        let mut cumulative = vec![0.0; stations];
        let mut total = 0.0;
        for index in 1..stations {
            total += points[index].sub(points[index - 1]).length();
            cumulative[index] = total;
        }
        if total <= tolerance {
            return Err("sweepSolid: path has zero length; cannot distribute the twist".into());
        }
        for length in &mut cumulative {
            *length /= total;
        }
        Some(cumulative)
    } else {
        None
    };

    // --- 3. Rotation-minimizing frames via the double-reflection method.
    let mut r_axes = Vec::with_capacity(stations);
    let mut s_axes = Vec::with_capacity(stations);
    let r0 = tangents[0].perpendicular()?; // any unit vector ⟂ T0
    s_axes.push(tangents[0].cross(r0).normalized()?);
    r_axes.push(r0);
    for index in 0..stations - 1 {
        let t_next = tangents[index + 1];
        let v1 = points[index + 1].sub(points[index]);
        let c1 = v1.dot(v1);
        let r_candidate = if c1 <= 1e-18 {
            // Coincident stations: carry the reference axis forward unchanged.
            r_axes[index]
        } else {
            // First reflection across the plane bisecting the step vector.
            let reflected_r = r_axes[index].sub(v1.scale(2.0 / c1 * v1.dot(r_axes[index])));
            let reflected_t = tangents[index].sub(v1.scale(2.0 / c1 * v1.dot(tangents[index])));
            // Second reflection across the plane bisecting the tangents.
            let v2 = t_next.sub(reflected_t);
            let c2 = v2.dot(v2);
            if c2 <= 1e-18 {
                reflected_r
            } else {
                reflected_r.sub(v2.scale(2.0 / c2 * v2.dot(reflected_r)))
            }
        };
        // Re-orthogonalize against the new tangent to shed floating drift.
        let r_next = r_candidate
            .sub(t_next.scale(r_candidate.dot(t_next)))
            .normalized()
            .map_err(|_| format!("sweepSolid: frame degenerated at station {index}"))?;
        s_axes.push(t_next.cross(r_next).normalized()?);
        r_axes.push(r_next);
    }

    // --- 3b. A CLOSED trajectory cannot be capped: the first and last sections
    //         land on top of each other, and the loft's own complaint about that
    //         ("end sections coincide") says nothing about the path. Name it here
    //         instead, with the two things that do work. The check is on the
    //         PATH's first and last station POINTS, which is enough: a path that
    //         comes back to where it started puts the two caps in the same place
    //         whatever direction it arrives from — coincident when it also
    //         arrives pointing the same way, interpenetrating when it does not,
    //         and neither is a solid worth building.
    //
    //         The scale is the path's own travel, so this is a proportion, not a
    //         length: ends within a thousandth of the distance travelled are the
    //         same place for capping purposes. On a circular arc that band is the
    //         last ~0.36° before closure, so a 359° sweep still builds.
    {
        let travel: f64 = points
            .windows(2)
            .map(|pair| pair[1].sub(pair[0]).length())
            .sum();
        let closure = points[stations - 1].sub(points[0]).length();
        if travel > tolerance && closure <= 1e-3 * travel {
            return Err(format!(
                "sweepSolid: the path returns to where it started ({closure:.3e} apart after \
                 travelling {travel:.3e}), so the sweep's two end caps would land on top of each \
                 other. A closed path swept this way IS a revolution — build it with Revolve \
                 about the same axis, or sweep the run in two halves and union them"
            ));
        }
    }

    // --- 4. Place a transformed copy of the profile at each station, by the mode
    //        the caller asked for (see `SectionPlacement`).
    //
    //   TRANSPLANT maps the profile's own plane frame onto the station frame,
    //       world = P_k + r_k·(l·pu) + s_k·(l·pv),   l = x − origin
    //   which lands the anchor origin ON the path and the profile square to it.
    //   The profile's distance from the path and its angle to the path are both
    //   discarded — there is no term along the tangent to carry them.
    //
    //   RIGID applies the path's own motion to the profile where it stands,
    //       world = P_k + a·T_k + b·r_k + c·s_k,     (a,b,c) = (l·T_0, l·r_0, l·s_0)
    //                                                l = x − P_0
    //   which is `P_k + R_k·(x − P_0)` for `R_k = F_k·F_0ᵀ`, written in the frame
    //   basis rather than as a matrix. The tangent term `a·T_k` is exactly what
    //   TRANSPLANT drops, and carrying it is what preserves both the profile's
    //   offset from the path and its angle to it. At station 0 the frames
    //   coincide and the section IS the profile, so the START cap lies in the
    //   plane it was drawn in.
    let path_start = points[0];
    let (t0, r0, s0) = (tangents[0], r_axes[0], s_axes[0]);
    let rigid = placement_mode == SectionPlacement::Rigid;
    let mut sections: Vec<Vec<NurbsCurve>> = Vec::with_capacity(stations);
    // Only the RIGID guard below reads these; TRANSPLANT leaves them empty.
    let mut station_points: Vec<Vec<Vec3>> = Vec::with_capacity(if rigid { stations } else { 0 });
    let mut section_normals: Vec<Vec3> = Vec::with_capacity(if rigid { stations } else { 0 });
    let mut anchor_track: Vec<Vec3> = Vec::with_capacity(if rigid { stations } else { 0 });
    for station in 0..stations {
        let station_origin = points[station];
        // Rotate the RMF axes about the tangent by the station's twist angle
        // (Rodrigues on vectors ⟂ the tangent: r' = r·cosφ + s·sinφ,
        // s' = s·cosφ − r·sinφ, since s = t × r and t × s = −r).
        let (ri, si) = match &twist_fractions {
            Some(fractions) => {
                let phi = twist_angle * fractions[station];
                let (sin_phi, cos_phi) = phi.sin_cos();
                let r = r_axes[station];
                let s = s_axes[station];
                (
                    r.scale(cos_phi).add(s.scale(sin_phi)),
                    s.scale(cos_phi).sub(r.scale(sin_phi)),
                )
            }
            None => (r_axes[station], s_axes[station]),
        };
        let tk = tangents[station];
        let place = |x: Vec3| -> Vec3 {
            if rigid {
                let local = x.sub(path_start);
                station_origin
                    .add(tk.scale(local.dot(t0)))
                    .add(ri.scale(local.dot(r0)))
                    .add(si.scale(local.dot(s0)))
            } else {
                let local = x.sub(origin);
                station_origin
                    .add(ri.scale(local.dot(pu)))
                    .add(si.scale(local.dot(pv)))
            }
        };
        if rigid {
            // The section's own normal takes the same rotation (a direction, so
            // `R_k` without the translation), and the anchor origin's track is
            // the profile's bulk motion for the near-parallel guard.
            let n0 = placement.normal;
            section_normals.push(
                tk.scale(n0.dot(t0))
                    .add(ri.scale(n0.dot(r0)))
                    .add(si.scale(n0.dot(s0))),
            );
            anchor_track.push(place(origin));
            station_points.push(Vec::with_capacity(
                profile.iter().map(|c| c.control_points.len()).sum(),
            ));
        }
        let mut section = Vec::with_capacity(profile.len());
        for curve in profile {
            let control_points = curve
                .control_points
                .iter()
                .map(|point| {
                    let weight = point.w;
                    let euclidean = Vec3::new(point.x / weight, point.y / weight, point.z / weight);
                    let world = place(euclidean);
                    if rigid {
                        station_points[station].push(world);
                    }
                    Vec4 {
                        x: world.x * weight,
                        y: world.y * weight,
                        z: world.z * weight,
                        w: weight,
                    }
                })
                .collect();
            section.push(NurbsCurve::new(
                curve.degree,
                curve.knots.clone(),
                control_points,
            )?);
        }
        sections.push(section);
    }

    // --- 4b. RIGID only: refuse a sweep whose sections fold through each other.
    //
    // TRANSPLANT keeps every section square to the path, so it can only fold
    // where the path's curvature radius drops below the profile's extent — the
    // case this builder has always documented as the caller's to avoid. RIGID
    // can fold for a second, much more reachable reason: a turning path carries
    // the profile around the TURN'S OWN AXIS, so a profile straddling that axis
    // has one half advancing while the other retreats. That is not a tolerance
    // question, it is a sign question, so it is answered exactly.
    //
    // The placement map is AFFINE in `x`, so the advance is affine in `x` too and
    // its extremes over the section lie on the control points. Checking the hull
    // therefore bounds the advance over the curves AND over the region they
    // enclose — this is a proof, not a sampling.
    if rigid {
        let mut most_positive = 0.0f64;
        let mut most_negative = 0.0f64;
        let mut longest_step = 0.0f64;
        for station in 0..stations - 1 {
            let normal = section_normals[station];
            for (before, after) in station_points[station]
                .iter()
                .zip(&station_points[station + 1])
            {
                let step = after.sub(*before);
                longest_step = longest_step.max(step.length());
                let advance = step.dot(normal);
                most_positive = most_positive.max(advance);
                most_negative = most_negative.min(advance);
            }
        }
        // Scale-relative, and far below any geometric tolerance: this band only
        // has to separate a sign from floating noise, never a small motion from
        // a large one.
        let noise = 1e-9 * longest_step.max(tolerance);
        if most_positive > noise && most_negative < -noise {
            return Err(
                "sweepSolid: the profile sweeps back through itself — part of it advances along \
                 the path while part of it retreats. A turning path carries the profile around \
                 the turn's own axis, so a profile that STRADDLES that axis folds into itself; \
                 move the profile clear of the axis, or sweep the run in pieces"
                    .into(),
            );
        }
        if most_positive <= noise && most_negative >= -noise {
            return Err(
                "sweepSolid: the path does not advance through the profile — it runs inside the \
                 profile plane, so the sweep encloses no volume"
                    .into(),
            );
        }
        // The near-parallel case, on the anchor origin's own track: the same 0.1
        // threshold `extrude_profile_brep` refuses a sliver at, so a STRAIGHT
        // path is accepted here exactly when the translational sweep of the same
        // profile and vector is. A station whose origin barely moves is skipped —
        // an origin sitting on the turn axis means the profile straddles it, and
        // the fold-back guard above has already spoken.
        for station in 0..stations - 1 {
            let step = anchor_track[station + 1].sub(anchor_track[station]);
            let length = step.length();
            if length <= noise {
                continue;
            }
            if step.dot(section_normals[station]).abs() / length < 0.1 {
                return Err(
                    "sweepSolid: the path is nearly parallel to the profile plane, which sweeps a \
                     sliver rather than a solid"
                        .into(),
                );
            }
        }
    }

    // --- 5. Loft through the swept sections (side walls + planar end caps).
    loft_profile_brep(&sections)
        .map_err(|error| format!("sweepSolid: loft through swept sections failed: {error}"))
}

/// Helical sweep (§5.7): sweep a CLOSED PLANAR profile loop along a helix of
/// `helix_radius` about the axis through `axis_origin` with direction
/// `axis_direction`, rising `pitch` per revolution for `turns` revolutions.
///
/// Sample a helix — `turns` revolutions about `axis_direction` from
/// `axis_origin`, starting at angle `start_angle` (radians, measured from the
/// axis frame's `u`) and rising `pitch` per turn, the radius running linearly
/// from `start_radius` to `end_radius` — uniformly in angle: 64 stations per
/// turn (at least 9, at most 4097 in total, so beyond 64 turns the per-turn
/// density thins). Returns the points and their chord parameters `s ∈ [0,1]`
/// (uniform-in-angle IS chord-length for a constant-radius helix, which is
/// what the averaged-knot interpolation assumes). `left_handed` winds the
/// angle the other way. The axis frame is `w = axis`, `u` = the component of
/// `reference` perpendicular to the axis (so angle zero points AT the
/// reference — the helix feature's local +X, or its start point), falling
/// back to `w.perpendicular()` when there is no usable reference, and
/// `v = w × u`. Shared by [`sweep_profile_helix`] and the HX feature, so the
/// coil a sweep builds and the edge a helix feature publishes agree exactly.
///
/// Errors on a degenerate axis, non-finite inputs, `turns ≤ 0`, more than 256
/// turns, a negative radius or pitch, or a helix that is a single point (both
/// radii AND the pitch zero).
#[allow(clippy::too_many_arguments)]
pub fn helix_sample_points(
    axis_origin: Vec3,
    axis_direction: Vec3,
    reference: Option<Vec3>,
    start_radius: f64,
    end_radius: f64,
    pitch: f64,
    turns: f64,
    start_angle: f64,
    left_handed: bool,
) -> Result<(Vec<Vec3>, Vec<f64>), String> {
    use std::f64::consts::TAU;
    let w = axis_direction
        .normalized()
        .map_err(|_| "helix: axis direction is degenerate".to_string())?;
    for (name, value) in [
        ("radius", start_radius),
        ("end radius", end_radius),
        ("pitch", pitch),
        ("turns", turns),
        ("start angle", start_angle),
    ] {
        if !value.is_finite() {
            return Err(format!("helix: {name} must be a finite number"));
        }
    }
    if start_radius < 0.0 || end_radius < 0.0 {
        return Err("helix: radius must not be negative".into());
    }
    if pitch < 0.0 {
        return Err("helix: pitch must not be negative".into());
    }
    if turns <= 0.0 {
        return Err("helix: turns must be positive".into());
    }
    if turns > 256.0 {
        return Err("helix: turns must be at most 256".into());
    }
    if start_radius == 0.0 && end_radius == 0.0 && pitch == 0.0 {
        return Err("helix: zero radius and zero pitch describe a single point".into());
    }
    let u = match reference {
        Some(reference) => {
            let radial = reference.sub(w.scale(reference.dot(w)));
            radial.normalized().or_else(|_| w.perpendicular())?
        }
        None => w.perpendicular()?,
    };
    let v = w.cross(u).normalized()?;
    let total_angle = turns * TAU;
    let height = pitch * turns;
    let sign = if left_handed { -1.0 } else { 1.0 };
    let count = ((turns * 64.0).ceil() as usize + 1).clamp(9, 4097);
    let mut points = Vec::with_capacity(count);
    let mut parameters = Vec::with_capacity(count);
    for index in 0..count {
        let s = index as f64 / (count - 1) as f64;
        let theta = start_angle + sign * total_angle * s;
        let radius = start_radius + (end_radius - start_radius) * s;
        points.push(
            axis_origin
                .add(u.scale(radius * theta.cos()))
                .add(v.scale(radius * theta.sin()))
                .add(w.scale(height * s)),
        );
        parameters.push(s);
    }
    Ok((points, parameters))
}

/// [`helix_sample_points`] fitted as ONE cubic curve (global interpolation —
/// `interpolate_curve`, the machinery every fitted path here uses). Cubic
/// interpolation error on a circle of radius R with node spacing Δθ is
/// ≈ R·Δθ⁴/384: ~2·10⁻⁷·R at 64 stations per turn.
#[allow(clippy::too_many_arguments)]
pub fn fit_helix_curve(
    axis_origin: Vec3,
    axis_direction: Vec3,
    reference: Option<Vec3>,
    start_radius: f64,
    end_radius: f64,
    pitch: f64,
    turns: f64,
    start_angle: f64,
    left_handed: bool,
) -> Result<NurbsCurve, String> {
    let (points, parameters) = helix_sample_points(
        axis_origin,
        axis_direction,
        reference,
        start_radius,
        end_radius,
        pitch,
        turns,
        start_angle,
        left_handed,
    )?;
    interpolate_curve(&points, 3, &parameters)
}

/// The helix is transcendental, not exactly NURBS-representable, so the path
/// is FIT: points (R·cosθ, R·sinθ, pitch·θ/2π) in the axis frame are sampled
/// uniformly in θ and globally interpolated with a cubic (`interpolate_curve`
/// — the same machinery every other fitted path here uses).  DENSITY: 64
/// samples per turn, capped at 1025 total nodes.  Cubic interpolation error
/// on a circle of radius R with node spacing Δθ is ≈ R·Δθ⁴/384: ~2·10⁻⁷·R at
/// 64/turn, and still ~6·10⁻⁵·R at the cap's worst case (16/turn at the
/// 64-turn limit) — orders below any profile a caller could sweep without
/// self-intersecting.  Uniform-in-θ parameters are exact chord-length for a
/// helix (constant speed), which is what the averaged-knot interpolation
/// assumes.
///
/// The fitted path then drives the EXISTING path-sweep core with 32 stations
/// per turn (min 32, capped at 1024 — the loft's dense interpolation solve is
/// O(stations³) per control column, so the cap trades per-turn density, never
/// correctness, at high turn counts).
///
/// GUARDS beyond the path sweep's own: the path sweep documents
/// self-intersection as caller responsibility, so the helix variant — which
/// knows its curvature analytically — rejects the two garbage modes itself:
///   • fold-over: the helix curvature radius (R² + c²)/R (c = pitch/2π) must
///     exceed the profile's max extent about its centroid, or the tube folds
///     through itself on the inner side (the torus tube-radius > major-radius
///     failure, pitch-relaxed);
///   • coil collision (turns ≥ 1): the normal gap between consecutive coils,
///     pitch·2πR/√((2πR)² + pitch²), must exceed the profile diameter.
/// Both use the profile's max sample distance from its centroid — conservative
/// for asymmetric profiles (extent in a harmless direction still counts), but
/// a false reject beats silent garbage.
///
/// SHALLOW COILS BUILD. They did not until 2026-09-08: the loft measured its
/// advance by the END-TO-END chord, which on a whole-turn helix is purely
/// AXIAL while both caps face nearly tangentially, so `|n·chord|` fell under
/// the loft's 0.1 sliver band and everything with pitch ≲ 0.63·R — most
/// springs — was refused. The loft measures the advance at each END now
/// (`loft_topology::basic::advance_from`), which is the direction the tube
/// actually grows through the cap it is orienting, so the band only ever
/// catches a real sliver.
pub fn sweep_profile_helix(
    profile: &[NurbsCurve],
    axis_origin: Vec3,
    axis_direction: Vec3,
    helix_radius: f64,
    pitch: f64,
    turns: f64,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    use std::f64::consts::TAU;

    // --- 1. Validate the helix parameters with honest errors.
    let w = axis_direction
        .normalized()
        .map_err(|_| "sweep_profile_helix: axis direction is degenerate".to_string())?;
    if !(helix_radius.is_finite() && helix_radius > 0.0) {
        return Err("sweep_profile_helix: helix radius must be positive".into());
    }
    if !(pitch.is_finite() && pitch > 0.0) {
        return Err("sweep_profile_helix: pitch must be positive".into());
    }
    if !(turns.is_finite() && turns > 0.0) {
        return Err("sweep_profile_helix: turns must be positive".into());
    }
    // 64 turns bounds the loft's O(stations³) interpolation solve; beyond it
    // the station cap would silently degrade per-turn density anyway.
    if turns > 64.0 {
        return Err("sweep_profile_helix: turns must be at most 64".into());
    }

    // --- 2. Profile extent about its centroid, sampled exactly like the path
    //        sweep derives its placement origin (boundary-sample mean), so the
    //        extent is measured about the point that actually rides the path.
    let mut samples = Vec::new();
    for curve in profile {
        let [start, end] = curve.domain()?;
        for sample in 0..16 {
            samples.push(curve.evaluate(start + (end - start) * sample as f64 / 16.0)?);
        }
    }
    if !samples.is_empty() {
        let mut centroid = Vec3::default();
        for point in &samples {
            centroid = centroid.add(*point);
        }
        let origin = centroid.scale(1.0 / samples.len() as f64);
        let extent = samples
            .iter()
            .map(|point| point.sub(origin).length())
            .fold(0.0, f64::max);
        let c = pitch / TAU; // axial rise per radian
                             // Fold-over: profile reaches past the helix's center of curvature.
        let curvature_radius = (helix_radius * helix_radius + c * c) / helix_radius;
        if extent >= curvature_radius {
            return Err(format!(
                "sweep_profile_helix: profile extent {extent:.6} reaches the helix \
                 curvature radius {curvature_radius:.6}; the tube would fold through \
                 itself — increase the helix radius or pitch, or shrink the profile"
            ));
        }
        // Coil collision: only possible once the sweep spans a full revolution.
        if turns >= 1.0 {
            let circumference = TAU * helix_radius;
            let turn_length = (circumference * circumference + pitch * pitch).sqrt();
            let coil_gap = pitch * circumference / turn_length;
            if coil_gap <= 2.0 * extent {
                return Err(format!(
                    "sweep_profile_helix: consecutive turns would self-intersect — \
                     coil gap {coil_gap:.6} does not clear the profile diameter {:.6}; \
                     increase the pitch or shrink the profile",
                    2.0 * extent
                ));
            }
        }
    }

    // --- 3. Fit the helical path (see the density rationale in the fn docs) —
    //        the SAME sampler + fit the HX feature publishes as its edge.
    let path = fit_helix_curve(
        axis_origin,
        w,
        None,
        helix_radius,
        helix_radius,
        pitch,
        turns,
        0.0,
        false,
    )
    .map_err(|error| format!("sweep_profile_helix: helix path fit failed: {error}"))?;

    // --- 4. Drive the existing sweep core; its (or the loft's) failures
    //        propagate with helix context prepended.
    let stations = ((turns * 32.0).ceil() as usize).clamp(32, 1024);
    // The helix builder places the profile ON the fitted helix (the coil's
    // section is drawn once and carried round), which is `Transplant` by
    // construction — the radius comes from the helix argument, not the profile's
    // own position.
    sweep_profile_along_path_stations(
        profile,
        &path,
        name,
        stations,
        0.0,
        None,
        SectionPlacement::Transplant,
    )
        .map_err(|error| format!("sweep_profile_helix: {error}"))
}
