//! SWP — Path Sweep. Rewired to resolve its profile + path FROM THE SCENE by name
//! (the headless cutover), mirroring `extrude.rs` / `revolve.rs`.
//!
//! # What this feature is (audit finding)
//!
//! The DEDICATED swept-surface op — NOT the `Sweep` (SW) feature, which only
//! extrudes along a chord and ignores twist. This feature sweeps a closed
//! planar profile along a path via `sweep_profile_along_chain`, which folds the
//! twisted and anchored single-curve builders behind one call and sweeps a
//! MULTI-segment path as one continuous tube. This port honors twist exactly
//! the same way.
//!
//! # SWP or SW — which sweep a path belongs to
//!
//! The two features now accept the SAME path selection (a whole sketch, or
//! connected edges chained head-to-tail), and they divide on the path's SHAPE
//! rather than on how it was picked:
//!   - **SWP (here)** drives the profile along the path on a
//!     rotation-minimizing frame, so the profile turns to follow a curve. It
//!     needs the run to be TANGENT-CONTINUOUS, because it skins between sampled
//!     stations: a corner between two stations would be rounded off, not
//!     mitred. A cornered run is refused, naming the two segments and the angle.
//!   - **SW** translates the profile along each segment's chord and unions the
//!     portions, which gives a corner a real mitre but never rotates the
//!     profile — and it refuses a CURVED segment.
//! Between them the two refusals partition the space: a smooth run is SWP's, a
//! cornered polyline is SW's, and each refusal names the other feature.
//!
//! # Headless input contract (post-cutover)
//!
//! Both inputs resolve from the live scene, by name (no marshaled curves):
//! - `profile` `reference_selection` names a SKETCH whose profile the SKETCH
//!   feature already extracted (`SceneMap::resolve_profile`). Each region's
//!   OUTER loop `.curves` are a swept ring; its `.edge_names` stamp the sidewall
//!   names. Hole loops sweep along the SAME path (same twist, anchored to the
//!   FIRST region outer's placement) into cutters subtracted via
//!   `common::subtract_region_holes`; further regions sweep anchored too and the
//!   region solids union into ONE result. A profile that resolves to a resident
//!   solid FACE becomes a `SketchProfile` via `face_profile::face_profile` —
//!   the extrude.rs / revolve.rs / sweep.rs two-step, and the reason the
//!   `profile` field's filter is `["SKETCH","FACE"]`. A face profile has no
//!   sketch behind it, so nothing is consumed and the source solid stays
//!   resident.
//! - `path` `reference_selection` (`multiple`) resolves via
//!   `common::resolve_path_chain` — SW's resolver — to an ordered, head-to-tail
//!   run of named world curves: a whole SKETCH contributes its published chain,
//!   individual sketch segments and resident solid EDGES resolve one curve each
//!   and chain by endpoint coincidence, and an edge running the "wrong" way is
//!   reversed. The FIRST pick seeds the direction. The whole run sweeps as ONE
//!   tube under a SINGLE rotation-minimizing frame, so the frame carries across
//!   a joint rather than restarting per segment; a one-segment run delegates to
//!   the single-curve builders and is byte-identical to what it always emitted.
//!
//! # Name fidelity (contract rule 2)
//!
//! Side faces in INPUT-curve order, then bottom cap = START, top cap = END (the
//! loft order `sweep_profile_along_path` produces):
//!   - side `i`  → `${edgeName}_SWP`  (missing name → the `"EDGE"` fallback)
//!   - START cap → `${featureID}_START`
//!   - END cap   → `${featureID}_END`
//! There is NO `${id}:` feature tag here (unlike SW) — the
//! names are bare. Solid name `featureID || "PathSweep"`.
//!
//! The optional `boolean` param folds via `common::finalize_solid`; the referenced
//! sketch is consumed (`common::consume_sketch`).

use crate::feature_pipeline::features::common;
use crate::feature_pipeline::{FeatureContext, FeatureResult, SketchProfile};
use crate::sweep_profile_along_chain;

pub fn execute(ctx: &FeatureContext) -> FeatureResult {
    match build(ctx) {
        Ok(result) => result,
        Err(error) => ctx.fail(error),
    }
}

fn build(ctx: &FeatureContext) -> Result<FeatureResult, String> {
    // --- Profile: resolve the SKETCH profile from the scene by name.
    // Normalize a `{sketch}:FACE` display-sheet pick to the sketch profile base.
    let profile_name = common::normalize_profile_alias(
        common::first_reference_name(ctx.param("profile"))
            .ok_or("path sweep: missing `profile` reference selection")?,
    );
    // Sketch profile first; otherwise a resident FACE becomes a profile via
    // `face_profile` (the extrude.rs / revolve.rs / sweep.rs two-step, same
    // `SketchProfile` shape and the same downstream path). `from_face` skips the
    // consumed-sketch bookkeeping — a face profile has no sketch behind it.
    let mut face_profile_storage: Option<SketchProfile> = None;
    let (profile, from_face): (&SketchProfile, bool) =
        match ctx.scene.resolve_profile(&profile_name) {
            Some(profile) => (profile, false),
            None => match ctx.scene.resolve_face(&profile_name) {
                Some(face) => {
                    let built = super::face_profile::face_profile(face)
                        .map_err(|error| format!("path sweep: {error}"))?;
                    (&*face_profile_storage.insert(built), true)
                }
                None => {
                    return Err(format!(
                        "path sweep: profile '{profile_name}' not found (no sketch profile or resident face)"
                    ));
                }
            },
        };
    let first_outer = profile
        .regions
        .first()
        .and_then(|region| region.first())
        .ok_or("path sweep: profile has no outer loop")?;

    // --- Path: a whole sketch chain and/or individually picked edges, ordered
    //     head-to-tail into ONE connected run — the same resolution SW uses, so
    //     the two sweeps accept the same selection. The whole run is swept as
    //     one continuous tube under a single rotation-minimizing frame; the
    //     builder refuses a run with a CORNER in it, because skinning between
    //     stations would round the corner off rather than mitre it (SW is the
    //     feature for a cornered path). A missing selection is a hard error.
    let path_names = common::reference_names(ctx.param("path"));
    if path_names.is_empty() {
        return Err("path sweep: requires a path edge selection".into());
    }
    let segments = common::resolve_path_chain(ctx, &path_names)
        .map_err(|error| format!("path sweep: {error}"))?;
    let path_curves: Vec<_> = segments.iter().map(|segment| segment.curve.clone()).collect();
    let path_segment_names: Vec<String> =
        segments.iter().map(|segment| segment.name.clone()).collect();

    let feature_id = if ctx.id.is_empty() {
        "PathSweep"
    } else {
        ctx.id.as_str()
    };

    // Twist: degrees → radians (a non-finite `twistAngle` → 0). The gate for the
    // twisted builder is plain truthiness on the RADIANS, i.e. `!= 0.0` exactly
    // (NOT an epsilon).
    let twist_degrees = common::number_or_default(ctx, "twistAngle", 0.0);
    let twist_radians = twist_degrees * std::f64::consts::PI / 180.0;

    // The builder re-centers a swept loop's own centroid onto the path; every
    // loop OTHER than the first region's outer (hole loops, and the outers of
    // further regions) must instead keep its in-plane offset relative to that
    // first outer — they sweep ANCHORED to its placement anchor.
    let anchor = crate::profile_anchor(&first_outer.curves)
        .map_err(|error| format!("path sweep: {error}"))?;

    // Per-loop cap names keyed off each region's STABLE loop identity, with the
    // un-keyed `{feature_id}_START/_END` as the containers standing for all of
    // them (see `common::CapNames`).
    let caps = common::CapNames::new(feature_id, &profile.regions);

    // One swept solid per REGION, unioned below.
    let mut region_solids = Vec::with_capacity(profile.regions.len());
    for (region_index, region) in profile.regions.iter().enumerate() {
        let outer = region.first().ok_or("path sweep: profile has no outer loop")?;
        if outer.curves.len() < 2 {
            return Err(format!(
                "path sweep: requires a closed profile with at least two boundary curves, got {}",
                outer.curves.len()
            ));
        }
        // Region 0 places on its OWN centroid; every later region rides region
        // 0's anchor so its in-plane offset survives (the same rule hole loops
        // follow below). `sweep_profile_along_chain` folds the twisted and
        // anchored variants into one call and delegates a ONE-segment chain
        // straight back to the single-curve builders, so an ordinary
        // single-edge path sweep emits exactly the geometry it always has.
        let mut solid = sweep_profile_along_chain(
            &outer.curves,
            &path_curves,
            &path_segment_names,
            twist_radians,
            Some(feature_id),
            (region_index != 0).then_some(anchor),
            crate::SectionPlacement::Transplant,
            "Use Sweep (SW) with `orientationMode: translate` for a cornered path — it builds one portion per segment, which is what gives a corner a mitre.",
        )
        .map_err(|error| format!("path sweep: {error}"))?;

        // Face names: `[${edgeName}_SWP..., ${id}_START, ${id}_END]` (INPUT-curve order).
        let side_count = outer.curves.len();
        let mut face_names: Vec<String> = Vec::with_capacity(side_count + 2);
        for index in 0..side_count {
            let base = outer
                .edge_names
                .get(index)
                .cloned()
                .flatten()
                .filter(|name| !name.trim().is_empty())
                .unwrap_or_else(|| "EDGE".to_string());
            face_names.push(format!("{base}_SWP"));
        }
        let (start, end) = &caps.per_loop[region_index];
        face_names.push(start.clone());
        face_names.push(end.clone());

        let faces = &mut solid
            .shells
            .get_mut(0)
            .ok_or("path sweep: builder produced no shell")?
            .faces;
        if faces.len() != face_names.len() {
            return Err(format!(
                "path sweep: builder produced {} faces, expected {} ({} sides + 2 caps)",
                faces.len(),
                face_names.len(),
                side_count
            ));
        }
        for (face, name) in faces.iter_mut().zip(&face_names) {
            face.name = Some(name.clone());
        }

        // Region holes: sweep each hole loop along the SAME path with the SAME
        // twist into an anchored cutter and subtract. Cutter caps land coplanar
        // with the outer's caps; the exact boolean's coplanar merge handles them.
        if region.len() > 1 {
            solid = common::subtract_region_holes(
                solid,
                profile,
                region,
                "path sweep",
                feature_id,
                None,
                &mut |loop_index, _depth| {
                    let hole_curves = &region[loop_index].curves;
                    sweep_profile_along_chain(
                        hole_curves,
                        &path_curves,
                        &path_segment_names,
                        twist_radians,
                        None,
                        Some(anchor),
                        crate::SectionPlacement::Transplant,
                        "Use Sweep (SW) with `orientationMode: translate` for a cornered path — it builds one portion per segment, which is what gives a corner a mitre.",
                    )
                },
            )?;
        }
        region_solids.push(solid);
    }
    let solid = common::union_region_solids(region_solids)
        .map_err(|error| format!("path sweep: {error}"))?;

    // Role metadata: sidewall/cap convention, from the actual named faces.
    common::stamp_sweep_roles_multi(&solid, "_SWP", &caps.starts(), &caps.ends());

    let mut result = common::finalize_solid_grouped(ctx, solid, feature_id, &caps.containers());
    if result.error.is_some() {
        return Ok(result);
    }
    // A FACE profile has no sketch to consume — the source solid stays resident.
    if !from_face {
        common::consume_sketch(ctx, &profile_name, &mut result);
    }
    Ok(result)
}

/// Context-bar applicability ([`crate::feature_pipeline::context_offer`]):
/// a selected profile drives `profile`, edges `path`.
pub fn context_applicable(probe: &crate::feature_pipeline::SelectionProbe) -> bool {
    probe.has_profile() || probe.edges > 0
}

pub fn schema() -> serde_json::Value {
    serde_json::json!({
    "type": "SWP",
    "shortName": "SWP",
    "longName": "Path Sweep",
    "displayBuilder": false,
    "inputParamsSchema": {
        "id": {
            "type": "string",
            "default_value": null,
            "hint": "unique identifier for the path sweep feature"
        },
        "profile": {
            "type": "reference_selection",
            "selectionFilter": [
                "SKETCH",
                "FACE"
            ],
            "multiple": false,
            "default_value": null,
            "hint": "Select a single closed planar profile (sketch or face) to sweep"
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
            "hint": "Sweep path: pick a whole sketch, or one or more connected edges — they are chained head-to-tail, with the FIRST pick setting the direction the sweep runs. The joints must be smooth (tangent-continuous); for a path with corners use Sweep (SW)."
        },
        "twistAngle": {
            "type": "number",
            "default_value": 0,
            "label": "Twist (degrees)",
            "hint": "Rotate the profile about the path tangent, linearly in arc length, by this total angle (§5.7). 0 = plain sweep."
        },
        "boolean": common::optional_boolean_schema()
    }
})
}

// BREP private tests: 0619803033c76e79
