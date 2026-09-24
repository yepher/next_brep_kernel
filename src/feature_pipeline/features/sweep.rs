//! SW — Sweep.
//!
//! # What this feature is
//!
//! The `Sweep` feature is NOT the dedicated swept-surface op — that is `SWP`
//! (`path_sweep.rs`), which drives the profile along a curve on a
//! rotation-minimizing frame. `SW` is the TRANSLATIONAL sweep: the profile keeps
//! its orientation and slides along the path, which for a polyline path is
//! exactly one oblique prism per path segment, unioned.
//!
//! # The path: a whole SKETCH, or several individually picked EDGES
//!
//! `path` is a `multiple` `["SKETCH","EDGE"]` selection, resolved by
//! `common::resolve_path_chain`:
//!   - a whole SKETCH resolves to its published chain — its longest open run, or a
//!     lone closed loop, which the fold-back rule below then refuses — so one
//!     selection covers the entire trajectory. `SKETCH` is a real pick lane
//!     (`SelectionFilter::sketch`), so clicking the sketch's sheet in the 3D view
//!     picks the whole sketch while clicking one of its drawn segments picks that
//!     edge;
//!   - individual sketch segments (`{sketchId}:G{gid}`) and resident solid EDGES
//!     resolve one curve each, and several of them CHAIN by endpoint coincidence.
//! The FIRST pick seeds the chain in its own drawn direction and every other
//! selection is ordered and reversed to join it, so which of the rest you clicked
//! first is irrelevant — but the first one still decides which WAY the sweep runs,
//! exactly as a single picked edge always has. (A whole-sketch pick is seeded by
//! the sketch's own published chain order.) A selection that does not form ONE
//! connected run names the stragglers and fails.
//!
//! # The build: one PORTION per path segment
//!
//! Portion `i` extrudes a COPY of the profile — translated by the sum of the
//! preceding segments' CHORDS, so consecutive portions share a cap exactly even
//! when individually picked edges meet only within tolerance — along segment `i`'s
//! chord. Portion 0 is un-translated, so a single-segment path builds the very
//! same geometry this feature always did (its faces are named per segment now —
//! see below — with the old spelling kept as their container). The union of the portions IS the exact
//! translational sweep of the profile along the polyline (the swept volume of a
//! rigid body along a polyline is the union of its per-segment sweeps), holes
//! included: each portion subtracts its own hole prisms BEFORE the union, which
//! is what carries a hole channel around a corner.
//!
//! Every segment must be STRAIGHT (degree-1, two control points), must not run
//! (nearly) inside the profile plane (the builder refuses a direction within ~6°
//! of it — `|n̂·d̂| < 0.1`), and must ADVANCE THROUGH THE PROFILE THE SAME WAY as the first — the
//! profile does not rotate, so a segment that reverses relative to it drives the
//! profile back through material already swept. A CLOSED path always reverses
//! somewhere (its advances must sum to zero) and is refused by the same rule.
//! Every refusal names the offending segment.
//!
//! That one rule is also what makes the cap naming exact: with all segments
//! advancing the same way, portion `i` occupies `[s_i, s_{i+1}]` through the
//! profile, so its far cap and portion `i+1`'s near cap are coincident and
//! ANTI-parallel and annihilate in the union. Only the chain's own two ends
//! survive, and they survive as themselves.
//!
//! # Headless profile + path contract
//!
//! Both inputs resolve FROM THE SCENE by name, HEADLESS (the `extrude.rs` /
//! `revolve.rs` template):
//!   - `profile` names a SKETCH whose profile the SKETCH feature already extracted
//!     (`SceneMap::resolve_profile`), OR a resident solid FACE — resolved via
//!     `scene.resolve_face` → `face_profile::face_profile` (the extrude.rs /
//!     revolve.rs two-step; `from_face` switches the cap base to the FACE name
//!     and skips sketch consumption). Sweep reads
//!     each region's OUTER loop curves + carried `{sketchId}:G{gid}` edge names —
//!     no marshaled curves, no caller round-trip. Region holes subtract as the same
//!     over-long through prisms extrude uses (`common::subtract_region_holes`);
//!     region solids union into ONE result.
//!   - `path` names the trajectory, as described above.
//!
//! # Name fidelity — a face per (path segment × profile edge)
//!
//! `extrude_profile_brep` emits faces UNNAMED, in order `[side faces (INPUT curve
//! order), bottom cap (= profile base / START), top cap (= far end / END)]`. Each
//! portion stamps that order:
//!   - side `i`  → `${tag}${edgeName[i]}:${segment}_SW` (the carried
//!                 `{sketchId}:G{gid}` profile edge name, else `EDGE_${i}`, joined
//!                 to the PATH SEGMENT's own name), with the un-keyed
//!                 `${tag}${edgeName[i]}_SW` registered as the CONTAINER standing
//!                 for that profile edge's wall along every segment. So a
//!                 single-segment sweep's reference still resolves exactly, and a
//!                 reference into a multi-segment sweep names both the profile edge
//!                 and the path segment it belongs to — neither moves when the
//!                 OTHER end of the path is edited.
//!   - START cap → `${tag}${faceName}:L{loopId}_START` (portion 0)
//!   - END cap   → `${tag}${faceName}:L{loopId}_END`   (the last portion)
//!     (one pair PER REGION, keyed by the region's outer loop's stable identity —
//!     see the sketch feature's `loop_ids`; the un-keyed `..._START/_END` is
//!     registered as the CONTAINER standing for all of them. A profile with no
//!     loop identity behind it — a FACE profile — keeps the un-keyed spelling.)
//!     Interior caps are keyed by segment too (`common::CapNames::interior`); the
//!     fold-back rule above guarantees they annihilate, so they never reach the
//!     result — the keyed spelling is what they would carry if one ever did.
//!   - hole walls → `${id}:HOLE:{loop key}:${segment}`, with `${id}:HOLE:{loop key}`
//!     as their container — the same per-segment treatment as the sidewalls.
//! where `tag` = `""` for an empty id, else `${id}:`,
//! and `faceName` is the sketch profile face name `{sketchId}:PROFILE`.
//!
//! The portion union PINS `_SW` and `:HOLE:` faces out of the coplanar merge
//! (`common::union_solids_keeping`). Without it, two portions' coplanar walls —
//! guaranteed for collinear segments, common either side of a turn — fuse into one
//! face carrying one of the two names, and WHICH one depends on the boolean's
//! internal face order. Pinning is what makes the per-segment names mean something
//! stable; the cost is a seam edge where two coplanar portions meet.
//!
//! The pin covers THIS feature's own unions (portions, then regions). The user's
//! `boolean` runs through `finalize_solid_grouped` un-pinned, like every other
//! feature's, so folding a multi-segment sweep into a target can still merge
//! portion walls that end up coplanar with each other or with the target.
//!
//! # The two orientation modes, and which path belongs to which
//!
//! | the path is | mode | what it builds |
//! |---|---|---|
//! | cornered (straight segments meeting at an angle) | `translate` | one oblique prism per segment, unioned; the corner is handled; per-SEGMENT wall names |
//! | smooth — curved, or segments meeting tangentially | `pathAlign` | ONE tube: the profile is CARRIED by the path's own rigid motion, keeping the position and angle it was drawn at; per-PROFILE-EDGE wall names |
//! | cornered AND the profile must follow it | neither | unbuilt (a mitred corner); refused by name |
//!
//! The gate between them is the JOINT ANGLE, not curvature: an all-straight
//! cornered polyline is refused under `pathAlign` for the same reason a curved one
//! is. `translate` refuses a CURVED segment for the mirror reason — a chord does
//! not follow a curve.
//!
//! # What `pathAlign` means, exactly
//!
//! The section at path parameter `t` is the profile moved by THE SAME RIGID MOTION
//! THE PATH UNDERGOES between its start and `t`:
//!
//! ```text
//!     section(t) = P(t) + R(t) · (x − P(0))
//! ```
//!
//! `R(t)` is the rotation carrying the path's start tangent frame to its frame at
//! `t` (`SectionPlacement::Rigid` in the builder). So the profile's POSITION
//! relative to the path and its ANGLE to the path are the drawn ones, and the only
//! thing that changes along the sweep is what the path itself does. Two readings
//! fall out, and they are the reason the mode exists:
//!   - a STRAIGHT path has `R ≡ I`, so the profile translates along the segment's
//!     vector for the segment's length — the oblique prism `translate` builds from
//!     the same inputs, exactly, not merely to within the station sampling. The
//!     modes agree on a straight path rather than merely agreeing in volume;
//!   - a circular ARC has `R` = the rotation about the ARC'S OWN CENTRE AXIS, so
//!     the profile is carried round that pivot and the end cap keeps the angle to
//!     the tangent the start cap had. A full circle is exactly a revolve.
//! The START cap therefore lies in the sketch plane. The path need not start on
//! the profile, or touch it: an offset profile sweeps the ring its offset traces.
//!
//! This is NOT what `SWP` (path_sweep.rs) does with the same builder: `SWP`
//! TRANSPLANTS the profile onto the path (centroid on the path, plane square to
//! it), discarding both the offset and the angle. The two features build the same
//! solid only for a profile drawn centred on and square to the path start.
//!
//! NAMING, `pathAlign`: one tube means exactly one wall per profile edge, so a
//! wall takes the un-keyed CONTAINER spelling `{tag}{profileEdge}_SW` rather than
//! the per-segment `{tag}{profileEdge}:{segment}_SW`. Nothing positional is in
//! either: both key on the carried `{sketchId}:G{gid}` name. Because `translate`
//! already registers that un-keyed string as the container standing for that
//! edge's walls across every segment, a reference stored under it resolves under
//! BOTH modes — flipping `orientationMode` does not break a downstream reference.
//! Caps keep the loop-keyed spelling, and one portion has no interior caps.
//!
//! # Deferred paths (clear errors, never a silent geometry drop)
//!
//! - UNDER `translate`, a CURVED path segment — the loft-through-transformed-copies
//!   branch — is not yet migrated; the error names the segment and points at
//!   `pathAlign`. What is deferred is that LANE, not the path: `pathAlign` sweeps a
//!   curved path TODAY, as one tube, and skips the straight-segment loop below
//!   outright (`filter(|_| !path_align)`), so the refusal is never reached in that
//!   mode.
//! - a path that DOUBLES BACK through the profile, a closed path included (see
//!   above) — sweep the run one direction at a time.
//! - `orientationMode: "pathAlign"` on a CORNERED path. The mode itself is built
//!   (see below); what is not built is the MITRE. Rotating the profile at a joint
//!   and mitring the corner is what this codebase has always meant by pathAlign,
//!   and the one-tube build cannot do the second half: it skins between sampled
//!   stations, so a corner would be rounded off rather than mitred. Refused by
//!   name, pointing at `translate` — which handles corners, at the cost of not
//!   rotating the profile. A path that is cornered AND wants the profile to
//!   follow it is genuinely unbuilt, and the refusal says so.
//! - a non-zero `twistAngle` (a no-op upstream, but a geometry-shaped param we refuse to
//!   silently ignore) is not yet migrated.

use crate::extrude_profile_brep;
use crate::feature_pipeline::features::common;
use crate::feature_pipeline::{FeatureContext, FeatureResult, SketchProfile};
use crate::Vec3;

/// Face-name substrings PINNED out of the portion union's coplanar merge — every
/// face this feature names per PATH SEGMENT (see the module header).
const KEEP_UNMERGED: [&str; 2] = ["_SW", ":HOLE:"];

pub fn execute(ctx: &FeatureContext) -> FeatureResult {
    match build(ctx) {
        Ok(result) => result,
        Err(error) => ctx.fail(error),
    }
}

/// One straight path segment, reduced to what the build needs: its source NAME,
/// the OFFSET the profile copy is translated by (the sum of the preceding
/// segments' chords), and the chord itself.
struct Step {
    name: String,
    offset: Vec3,
    direction: Vec3,
    distance: f64,
}

fn build(ctx: &FeatureContext) -> Result<FeatureResult, String> {
    // ---- Profile: resolve the referenced sketch's extracted profile, else a
    // resident FACE via `face_profile` (the extrude.rs/revolve.rs two-step).
    // `from_face` gates the cap-name base and skips the consumed-sketch
    // bookkeeping.
    // Normalize a `{sketch}:FACE` display-sheet pick to the sketch profile base.
    let profile_name = common::normalize_profile_alias(
        common::first_reference_name(ctx.param("profile"))
            .ok_or("sweep: missing `profile` reference selection")?,
    );
    let mut face_profile_storage: Option<SketchProfile> = None;
    let (profile, from_face): (&SketchProfile, bool) =
        match ctx.scene.resolve_profile(&profile_name) {
            Some(profile) => (profile, false),
            None => match ctx.scene.resolve_face(&profile_name) {
                Some(face) => {
                    let built = super::face_profile::face_profile(face)
                        .map_err(|error| format!("sweep: {error}"))?;
                    (&*face_profile_storage.insert(built), true)
                }
                None => {
                    return Err(format!(
                        "sweep: profile '{profile_name}' not found (no sketch profile or resident face)"
                    ));
                }
            },
        };

    // ---- Path: a whole sketch chain and/or individually picked edges, ordered
    // head-to-tail into ONE connected run. A missing selection is a hard error.
    let path_names = common::reference_names(ctx.param("path"));
    if path_names.is_empty() {
        return Err("sweep: requires a path edge selection".into());
    }
    let chain = common::resolve_path_chain(ctx, &path_names)
        .map_err(|error| format!("sweep: {error}"))?;

    // ---- Orientation mode, read BEFORE the straight-segment loop below.
    // ORDER MATTERS: that loop refuses a CURVED segment, and a curved segment is
    // exactly what `pathAlign` exists to sweep. Reading the mode afterwards (as
    // this did) meant a curved path was refused before orientation was ever
    // consulted, so the mode could never take effect on the paths that need it.
    let path_align = ctx
        .param("orientationMode")
        .and_then(|value| value.as_str())
        .map(|mode| mode.trim().eq_ignore_ascii_case("pathAlign"))
        .unwrap_or(false);

    // Each segment reduces to `(offset, chord)`. The offset accumulates the
    // CHORDS, not the resolved endpoints, so portion `i`'s far cap and portion
    // `i+1`'s near cap are the same translated profile exactly — individually
    // picked solid edges that meet only within tolerance would otherwise leave a
    // sliver between the two.
    //
    // TRANSLATE ONLY: `pathAlign` sweeps the whole run as one tube and never
    // reduces a segment to a chord, so it neither needs nor may run this loop.
    let mut steps: Vec<Step> = Vec::with_capacity(chain.len());
    let mut offset = Vec3::default();
    for segment in chain.iter().filter(|_| !path_align) {
        // STRAIGHT segments only: a degree-1 two-control-point curve. Anything
        // else is the curved-sweep (loft) branch, deferred loudly BY NAME.
        if segment.curve.degree != 1 || segment.curve.control_points.len() != 2 {
            return Err(format!(
                "sweep: path segment '{}' is curved, and `orientationMode: translate` slides \
                 the profile along each segment's CHORD, which would not follow the curve. \
                 Set `orientationMode: pathAlign` to sweep this path — the profile then \
                 rotates to follow it. (Path Sweep, SWP, builds the same shape as a \
                 separate feature.)",
                segment.name
            ));
        }
        let [t0, t1] = segment.curve.domain()?;
        let direction = segment.curve.evaluate(t1)?.sub(segment.curve.evaluate(t0)?);
        let distance = direction.length();
        if distance <= 1e-12 {
            return Err(format!(
                "sweep: path segment '{}' has zero length",
                segment.name
            ));
        }
        steps.push(Step {
            name: segment.name.clone(),
            offset,
            direction,
            distance,
        });
        offset = offset.add(direction);
    }

    // A segment lying exactly IN the profile plane sweeps nothing and leaves the
    // fold-back guard below with no sense to compare against. (The builder owns
    // the real threshold — it refuses anything within ~84° of the plane — but it
    // runs after that guard, so the degenerate case is named here first.)
    for step in &steps {
        if step.direction.dot(profile.z_axis).abs() <= 1e-9 * step.distance {
            return Err(format!(
                "sweep: path segment '{}' runs inside the profile plane — it would sweep the \
                 profile through itself rather than along the path",
                step.name
            ));
        }
    }

    // ---- Fold-back guard. Every segment must advance through the profile in the
    // SAME sense. The profile does not rotate here, so a segment that reverses
    // relative to the profile normal drives the profile back through material it
    // has already swept: the result self-intersects, and the two portions either
    // side of the reversal share a cap they now face the SAME way, so the caps
    // MERGE instead of cancelling and the surviving face's name falls to the
    // boolean's face order. A CLOSED path always reverses somewhere (its
    // segments' advances must sum to zero), which is why one is refused here.
    //
    // The profile's own plane normal (`SketchProfile::z_axis`, published by the
    // sketch and by `face_profile` alike) fixes the reference; the sign comes from
    // the FIRST segment and every later segment is compared against it.
    //
    // Comparing to the FIRST rather than to the PREDECESSOR is right HERE, and
    // that is worth saying because the loft's winding check had to go the other
    // way (a section's winding is compared to its predecessor, since past a 90°
    // bend the frame has rotated away from section 0). The difference: the loft's
    // reference frame MOVES along the path, this one does not. The profile never
    // rotates in a translational sweep, so `z_axis` is a global reference and
    // `sign(dᵢ·n) == sign(d₀·n)` is transitive — predecessor and first agree, and
    // only the global form catches a chain that drifts back a segment at a time.
    if steps.len() > 1 {
        let normal = profile.z_axis;
        let sense = steps[0].direction.dot(normal).signum();
        for step in &steps[1..] {
            if step.direction.dot(normal) * sense <= 0.0 {
                return Err(format!(
                    "sweep: path segment '{}' doubles back through the profile — this sweep \
                     TRANSLATES the profile without rotating it, so a segment that reverses \
                     relative to the profile drives it back through material already swept. \
                     Every path segment must advance through the profile the same way (a CLOSED \
                     path never can). Sweep the run one direction at a time.",
                    step.name
                ));
            }
        }
    }

    // ---- Orientation guard: `pathAlign` would rotate the profile at each joint
    // and miter the corners. This build translates instead, so on a multi-segment
    // path the mode is refused rather than silently dropped. On ONE straight
    // segment it changes nothing either way.
    // `pathAlign` on a multi-segment path is BUILT (one tube, single
    // rotation-minimizing frame). What is NOT built is a MITRED CORNER: the tube
    // is skinned between sampled stations, so a corner would be rounded off
    // rather than mitred. `sweep_profile_along_chain` refuses a cornered joint by
    // name and its message points back at `translate`, which handles corners at
    // the cost of not rotating the profile. Nothing to gate here.

    // ---- Twist guard: a non-zero twist is a no-op we refuse to silently honor.
    // Non-finite / absent / zero → treated as 0 (`Number(twist)` NaN-guard).
    if let Some(value) = ctx.param("twistAngle") {
        if !value.is_null() {
            if let Ok(angle) = ctx.number("twistAngle") {
                if angle.is_finite() && angle.abs() > 1e-12 {
                    return Err(
                        "sweep: non-zero `twistAngle` is not yet migrated to the Rust pipeline \
                         (a no-op in the previous kernel)"
                            .into(),
                    );
                }
            }
        }
    }

    // ---- Build: one PORTION per (region × path segment), unioned -------------
    let tag = normalize_feature_tag(&ctx.id);
    // Cap base (the source-profile base name): the sketch profile face name
    // `{sketchId}:PROFILE` for a sketch, the FACE name verbatim for a face
    // profile (the `:PROFILE` branch strips-then-reappends, a no-op).
    let base = if from_face {
        profile_base_name(&tag, &profile_name)
    } else {
        let sketch_id = profile_name
            .strip_suffix(":PROFILE")
            .unwrap_or(&profile_name);
        profile_base_name(&tag, &format!("{sketch_id}:PROFILE"))
    };

    // Per-loop cap names keyed off each region's STABLE loop identity, with the
    // un-keyed `{base}_START/_END` as the containers standing for all of them
    // (see `common::CapNames`).
    let caps = common::CapNames::new(&base, &profile.regions);
    let pinned: Vec<String> = KEEP_UNMERGED.iter().map(|s| s.to_string()).collect();
    // Every cap name the build stamps, so the role pass marks an interior cap a
    // cap in the rare case one survives the union.
    let mut start_names = caps.starts();
    let mut end_names = caps.ends();
    // `container -> members` for the per-segment sidewall and hole-wall names.
    let mut containers = caps.containers();

    // `pathAlign` sweeps the run as ONE tube per region, so it needs the chain as
    // plain curves. It needs NO placement anchor: `SectionPlacement::Rigid`
    // carries every loop from where it was drawn, so a hole's offset from its
    // outer loop, and a later region's offset from the first, are preserved by
    // the construction itself rather than by lending one loop's frame to the
    // rest. (That anchor exists for `Transplant`, which moves each loop onto the
    // path and would otherwise re-centre every one of them. SWP still needs it.)
    let path_curves: Vec<_> = chain.iter().map(|s| s.curve.clone()).collect();
    let path_segment_names: Vec<String> = chain.iter().map(|s| s.name.clone()).collect();

    let mut region_solids = Vec::with_capacity(profile.regions.len());
    for (region_index, region) in profile.regions.iter().enumerate() {
        let outer = region.first().ok_or("sweep: profile has no outer loop")?;
        if outer.curves.len() < 2 {
            return Err(format!(
                "sweep: outer loop needs >= 2 curves, got {}",
                outer.curves.len()
            ));
        }
        let side_count = outer.curves.len();

        if path_align {
            // ONE tube for the whole run, every loop carried by the PATH'S OWN
            // rigid motion from where it was drawn (`SectionPlacement::Rigid`).
            let mut solid = crate::sweep_profile_along_chain(
                &outer.curves,
                &path_curves,
                &path_segment_names,
                0.0,
                Some(&ctx.id),
                None,
                crate::SectionPlacement::Rigid,
                "Set `orientationMode: translate` to sweep this path — that mode builds one portion per segment and handles the corner, at the cost of not rotating the profile to follow the path. A mitred, path-aligned corner is not built.",
            )
            .map_err(|error| format!("sweep: {error}"))?;

            // NAMES. One tube means exactly ONE wall per profile edge, so a wall
            // takes the CONTAINER spelling — the un-keyed `{tag}{profileEdge}_SW`
            // that already stands for "that profile edge's wall along every path
            // segment". Nothing positional survives: the name is keyed on the
            // carried `{sketchId}:G{gid}` profile-edge name, and no path segment
            // appears in it because there is no per-segment wall to name. The
            // payoff is compatibility — a reference stored under the container
            // resolves under BOTH modes, so flipping `orientationMode` does not
            // break a downstream reference. Caps keep the loop-keyed spelling,
            // and with one portion there are no interior caps at all.
            let mut face_names: Vec<String> = Vec::with_capacity(side_count + 2);
            for index in 0..side_count {
                face_names.push(wall_container_name(&tag, &outer.edge_names, index));
            }
            let (chain_start, chain_end) = &caps.per_loop[region_index];
            face_names.push(chain_start.clone());
            face_names.push(chain_end.clone());

            let faces = &mut solid
                .shells
                .get_mut(0)
                .ok_or("sweep builder produced no shell")?
                .faces;
            if faces.len() != face_names.len() {
                return Err(format!(
                    "sweep builder produced {} faces, expected {} ({} sides + 2 caps)",
                    faces.len(),
                    face_names.len(),
                    side_count
                ));
            }
            for (face, name) in faces.iter_mut().zip(&face_names) {
                face.name = Some(name.clone());
            }

            // Hole loops sweep the SAME run and subtract — a translated
            // through-prism would not follow the curve. Under `Rigid` the hole
            // rides the outer wall by construction: both are carried by the same
            // path motion from their own drawn positions, so the channel stays
            // where the sketch put it without an anchor to tie them together.
            if region.len() > 1 {
                solid = common::subtract_region_holes(
                    solid,
                    profile,
                    region,
                    "sweep",
                    &ctx.id,
                    None,
                    &mut |loop_index, _depth| {
                        crate::sweep_profile_along_chain(
                            &region[loop_index].curves,
                            &path_curves,
                            &path_segment_names,
                            0.0,
                            None,
                            None,
                            crate::SectionPlacement::Rigid,
                            "Set `orientationMode: translate` to sweep this path — that mode builds one portion per segment and handles the corner, at the cost of not rotating the profile to follow the path. A mitred, path-aligned corner is not built.",
                        )
                    },
                )?;
            }
            region_solids.push(solid);
            continue;
        }

        let mut portions = Vec::with_capacity(steps.len());
        for (step_index, step) in steps.iter().enumerate() {
            // The profile copy this segment sweeps.
            let outer_curves = common::translate_curves(&outer.curves, step.offset)?;
            let mut solid = extrude_profile_brep(&outer_curves, step.direction, step.distance)
                .map_err(|error| {
                    format!("sweep: path segment '{}': {error}", step.name)
                })?;

            // Face order (sweep_topology.rs): [sidewalls in INPUT order..., bottom, top].
            let mut face_names: Vec<String> = Vec::with_capacity(side_count + 2);
            for index in 0..side_count {
                face_names.push(side_face_name(&tag, &outer.edge_names, index, &step.name));
            }
            // The chain's own two ends keep the plain per-loop cap spelling;
            // every interior cap is keyed by its segment (and cancels in the
            // union — see `CapNames::interior`).
            let (interior_start, interior_end) = caps.interior(region_index, &step.name);
            let (chain_start, chain_end) = &caps.per_loop[region_index];
            let start = if step_index == 0 {
                chain_start.clone()
            } else {
                start_names.push(interior_start.clone());
                interior_start
            };
            let end = if step_index + 1 == steps.len() {
                chain_end.clone()
            } else {
                end_names.push(interior_end.clone());
                interior_end
            };
            face_names.push(start);
            face_names.push(end);

            let faces = &mut solid
                .shells
                .get_mut(0)
                .ok_or("sweep builder produced no shell")?
                .faces;
            if faces.len() != face_names.len() {
                return Err(format!(
                    "sweep builder produced {} faces, expected {} ({} sides + 2 caps)",
                    faces.len(),
                    face_names.len(),
                    side_count
                ));
            }
            for (face, name) in faces.iter_mut().zip(&face_names) {
                face.name = Some(name.clone());
            }

            // Region holes: this portion IS an extrude along its segment's chord,
            // so each hole loop subtracts as the same over-long through prism
            // extrude uses — cut PER PORTION, before the union, which is what
            // carries the hole channel around a corner. The prism margin grows
            // with nesting depth so a nested cutter strictly overhangs its
            // parent's caps.
            if region.len() > 1 {
                solid = common::subtract_region_holes(
                    solid,
                    profile,
                    region,
                    "sweep",
                    &ctx.id,
                    Some(step.name.as_str()),
                    &mut |loop_index, depth| {
                        let hole_curves =
                            common::translate_curves(&region[loop_index].curves, step.offset)?;
                        common::hole_prism(&hole_curves, step.direction, step.distance, depth)
                    },
                )?;
            }
            portions.push(solid);
        }

        // The portions of ONE region union into that region's solid, keeping each
        // portion's own walls addressable (see `KEEP_UNMERGED`).
        region_solids.push(
            common::union_solids_keeping(portions, &pinned)
                .map_err(|error| format!("sweep: {error}"))?,
        );
    }

    // Sidewall + hole-wall containers: the un-keyed name stands for that profile
    // edge's (or hole loop's) wall along EVERY path segment.
    for region in &profile.regions {
        let outer = &region[0];
        for index in 0..outer.curves.len() {
            let container = wall_container_name(&tag, &outer.edge_names, index);
            let members = steps
                .iter()
                .map(|step| side_face_name(&tag, &outer.edge_names, index, &step.name))
                .collect();
            containers.push((container, members));
        }
        for loop_index in 1..region.len() {
            let key = common::hole_key(&region[loop_index], loop_index);
            let container = common::hole_face_name(&ctx.id, &key, None);
            let members = steps
                .iter()
                .map(|step| common::hole_face_name(&ctx.id, &key, Some(&step.name)))
                .collect();
            containers.push((container, members));
        }
    }

    let solid = common::union_solids_keeping(region_solids, &pinned)
        .map_err(|error| format!("sweep: {error}"))?;

    // Role metadata: sidewall/cap convention, from the actual named faces.
    common::stamp_sweep_roles_multi(&solid, "_SW", &start_names, &end_names);

    // Solid name: `featureID || sweeps[0].name` → `${id}` or "Sweep".
    let base_name = if ctx.id.is_empty() { "Sweep" } else { &ctx.id };
    let mut result = common::finalize_solid_grouped(ctx, solid, base_name, &containers);
    if result.error.is_some() {
        return Ok(result);
    }
    // Consume the referenced profile sketch (default true). A FACE profile has
    // no sketch to consume — the source solid stays resident.
    if !from_face {
        common::consume_sketch(ctx, &profile_name, &mut result);
    }
    Ok(result)
}

/// Feature tag: trim; empty → `""`; already `:`-ended →
/// as-is; otherwise append `:`.
fn normalize_feature_tag(id: &str) -> String {
    let tag = id.trim();
    if tag.is_empty() {
        String::new()
    } else if tag.ends_with(':') {
        tag.to_string()
    } else {
        format!("{tag}:")
    }
}

/// The profile edge's own name token: the carried source edge name when present +
/// non-empty, else the INDEX-based default `EDGE_${i}`. Trimmed.
fn profile_edge_token(edge_names: &[Option<String>], index: usize) -> String {
    match edge_names.get(index) {
        Some(Some(name)) if !name.trim().is_empty() => name.trim().to_string(),
        _ => format!("EDGE_{index}"),
    }
}

/// One portion's side face name: the profile edge joined to the PATH SEGMENT that
/// swept it, `${tag}${edge}:${segment}_SW`.
fn side_face_name(
    tag: &str,
    edge_names: &[Option<String>],
    index: usize,
    segment: &str,
) -> String {
    format!("{tag}{}:{segment}_SW", profile_edge_token(edge_names, index))
}

/// The CONTAINER standing for one profile edge's wall along every path segment,
/// `${tag}${edge}_SW` — the spelling a single-segment sweep has always used.
fn wall_container_name(tag: &str, edge_names: &[Option<String>], index: usize) -> String {
    format!("{tag}{}_SW", profile_edge_token(edge_names, index))
}

/// Source-profile base name: `${tag}${faceName}`. The `:PROFILE`
/// branch is a NO-OP (both branches yield `${tag}${faceName}`), so a plain concat
/// is exact; an empty/blank face name falls back to `PROFILE`.
fn profile_base_name(tag: &str, face_name: &str) -> String {
    let trimmed = face_name.trim();
    let name = if trimmed.is_empty() { "PROFILE" } else { trimmed };
    format!("{tag}{name}")
}


/// Context-bar applicability ([`crate::feature_pipeline::context_offer`]):
/// a selected profile drives `profile`, an edge `path`.
pub fn context_applicable(probe: &crate::feature_pipeline::SelectionProbe) -> bool {
    probe.has_profile() || probe.edges > 0
}

pub fn schema() -> serde_json::Value {
    serde_json::json!({
    "type": "SW",
    "shortName": "SW",
    "longName": "Sweep",
    "displayBuilder": false,
    "inputParamsSchema": {
        "id": {
            "type": "string",
            "default_value": null,
            "hint": "unique identifier for the sweep feature"
        },
        "profile": {
            "type": "reference_selection",
            "selectionFilter": [
                "SKETCH",
                "FACE"
            ],
            "multiple": false,
            "default_value": null,
            "hint": "Select the profile to sweep"
        },
        "consumeProfileSketch": {
            "type": "boolean",
            "default_value": true,
            "hint": "Remove the referenced sketch after creating the sweep. Turn off to keep it in the scene."
        },
        "path": {
            "type": "reference_selection",
            "selectionFilter": [
                "SKETCH",
                "EDGE"
            ],
            "multiple": true,
            "default_value": null,
            "hint": "Sweep path: pick a whole sketch, or one or more connected edges — they are chained head-to-tail, with the FIRST pick setting the direction the sweep runs. Straight segments only."
        },
        "orientationMode": {
            "type": "options",
            "options": [
                "translate",
                "pathAlign"
            ],
            "default_value": "translate",
            "hint": "How the profile is carried along the path. 'translate' (default): the profile keeps its orientation and slides along each segment's chord — one portion per segment, so CORNERS are handled, but straight segments only. 'pathAlign': the profile is carried by the PATH'S OWN motion, keeping the position and the angle you drew it at and turning exactly as much as the path turns — a straight path extrudes it from where it sits, an arc swings it about the arc's centre. Swept as one continuous tube; this is the mode for a CURVED path, and it REQUIRES the joints to be smooth (tangent-continuous). A mitred corner is NOT built: a cornered path under 'pathAlign' is refused and names the joint, so use 'translate' for corners."
        },
        "twistAngle": {
            "type": "number",
            "default_value": 0,
            "hint": "Twist angle for the sweep. Must be 0 here — for a twisted sweep use Path Sweep (SWP)."
        },
        "boolean": common::optional_boolean_schema()
    }
})
}

// BREP private tests: 3ea7955a1b9eaa80
