//! Rolling-ball edge blends, with optional taper through `radiusEnd`.
//!
//! Edge and face selections resolve to one solid and are blended together,
//! including shared convex corners. Cross-solid selections return no changes.
//! Only AUTO and INSET directions are supported.
//!
//! The result retains the target's name. Generated faces use the shared
//! `{featureID || 'F'}:BLEND:{edgeName}` naming scheme, with corner patches
//! under `{base}:CORNER:{adjacent edge names}`.

use crate::feature_pipeline::features::common;
use crate::feature_pipeline::{FeatureContext, FeatureResult};

pub fn execute(ctx: &FeatureContext) -> FeatureResult {
    match build(ctx) {
        Ok(result) => result,
        Err(error) => ctx.fail(error),
    }
}

fn build(ctx: &FeatureContext) -> Result<FeatureResult, String> {
    common::require_blend_direction(ctx, "fillet")?;

    let selection = common::resolve_blend_selection(ctx)?;
    let mut result = FeatureResult::empty(ctx.id.clone(), ctx.feature_type.clone());
    result.unresolved = selection.unresolved;

    // A cross-solid selection is a SOFT abort (warn + return empty).
    if selection.multi_solid {
        return Ok(result);
    }
    // Nothing resolved: `unresolved` carries the misses (the caller repairs + re-dispatches).
    let Some(target) = selection.target else {
        return Ok(result);
    };

    let radius = ctx.number("radius")?;
    // A non-positive / non-finite radius is a soft no-op.
    if !(radius.is_finite() && radius > 0.0) {
        return Ok(result);
    }

    // `radiusEnd > 0 && |radiusEnd − radius| > 1e-9` selects the VARIABLE
    // (tapered) fillet.
    let taper = common::optional_number(ctx, "radiusEnd")?
        .filter(|value| value.is_finite() && *value > 0.0 && (*value - radius).abs() > 1e-9);

    // Names parallel the sampled edge points; the base also names corner patches.
    let blend_base = common::blend_face_base(&ctx.id);
    let blend_names: Vec<String> = target
        .edge_names
        .iter()
        .map(|edge_name| common::blend_face_name(&ctx.id, edge_name))
        .collect();
    let blended = crate::with_registered_solid_str(target.handle, |solid| match taper {
        Some(radius_end) => crate::fillet_edges_variable(
            solid,
            &target.edge_points,
            Some(&blend_names),
            &[(0.0, radius), (1.0, radius_end)],
            false,
            Some(&blend_base),
        ),
        None => crate::fillet_edges(
            solid,
            &target.edge_points,
            Some(&blend_names),
            radius,
            false,
            Some(&blend_base),
        ),
    })
    .map_err(|error| format!("fillet failed: {error}"))?;

    result.added.push(common::register_added(blended, &target.name));
    result.removed.push(target.name);
    Ok(result)
}


/// Context-bar applicability ([`crate::feature_pipeline::context_offer`]):
/// selected faces/edges drive `edges`.
pub fn context_applicable(probe: &crate::feature_pipeline::SelectionProbe) -> bool {
    probe.faces > 0 || probe.edges > 0
}

pub fn schema() -> serde_json::Value {
    serde_json::json!({
    "type": "F",
    "shortName": "F",
    "longName": "Fillet",
    "displayBuilder": false,
    "inputParamsSchema": {
        "id": {
            "type": "string",
            "default_value": null,
            "hint": "unique identifier for the fillet feature"
        },
        "edges": {
            "type": "reference_selection",
            "selectionFilter": [
                "FACE",
                "EDGE"
            ],
            "timestampDependency": "parentSolid",
            "multiple": true,
            "default_value": null,
            "hint": "Select faces (or an edge) to fillet along shared edges"
        },
        "radius": {
            "type": "number",
            "step": 0.1,
            "default_value": 1,
            "hint": "Fillet radius"
        },
        "radiusEnd": {
            "type": "number",
            "step": 0.1,
            "default_value": 0,
            "hint": "Optional end radius for a VARIABLE (tapered) fillet. 0 = constant radius. When > 0 and different from radius, each selected edge tapers from radius at its start to this value at its end (kernel-exact path; corners are not rounded in variable mode)."
        },
        "direction": {
            "type": "options",
            "options": [
                "AUTO",
                "INSET"
            ],
            "default_value": "AUTO",
            "hint": "AUTO classifies each selected edge as inside/outside and applies subtract/union automatically."
        }
    }
})
}

// BREP private tests: c47b57c8274b747f
