//! ACOMP — Assembly Component: one placed INSTANCE of a parts-library entry
//! (assemblies build-spec §2.2 / §10 item 3). FULLY IMPLEMENTED.
//!
//! `inputParams = { partName, transform, isFixed }` — a reference into the
//! parts library (`parts_library.rs`) with NO embedded payload. `transform` is
//! the rigid instance pose `{ translate: [x,y,z], rotateEulerDeg: [rx,ry,rz] }`,
//! expression-capable per component, composed through the ONE feature-transform
//! convention ([`common::compose_trs_matrix`], intrinsic-XYZ Euler — the same
//! compose XFORM uses and the same one any solver/gizmo pose write-back must
//! use). Namespacing and registration are `component::create_component`'s job;
//! this feature only chooses WHERE the part-local solids come from:
//!
//! - **FAST (default):** decode the library entry's `snapshot` payload straight
//!   into kernel structs — no sub-part history execution.
//! - **SELF-HEAL:** an unreadable/corrupt snapshot (or a DIRTY entry — the
//!   `refresh_library_entry` / edit-in-context signal) re-executes the entry's
//!   embedded `document` in isolation, REWRITES the snapshot, then proceeds as
//!   FAST. The history cache re-runs this feature when the entry's content
//!   changes (see `parts_library::mix_descriptor_fingerprint`).
//!
//! **Auto-ground:** when `isFixed` is ABSENT and no component exists earlier in
//! the history, the instance is grounded (the first component of an empty
//! assembly is auto-fixed). An EXPLICIT `false` is always respected — the rule
//! keys on absence, so un-fixing the sole component stays possible. Wave-3
//! seam: the insert flow should write the resolved value into `inputParams` so
//! it is visible/editable in the dialog.
//!
//! Metadata: the snapshot's captured records are stamped under the NAMESPACED
//! names (`{id}:{name}`), replacing wholesale each run (a refreshed part's
//! dropped keys must not linger); an `inheritsFrom` value is namespaced too so
//! a sub-part's inheritance chain stays inside the sub-part.

use std::collections::BTreeMap;

use crate::feature_pipeline::component::{self, create_component};
use crate::feature_pipeline::features::{common, port};
use crate::feature_pipeline::{
    parts_library, scene_metadata, FeatureContext, FeatureResult, PortRecord,
};
use crate::{restore_solids, AffineTransform, RestoredSnapshot};

pub fn execute(ctx: &FeatureContext) -> FeatureResult {
    match build(ctx) {
        Ok(result) => result,
        Err(error) => ctx.fail(error),
    }
}

fn build(ctx: &FeatureContext) -> Result<FeatureResult, String> {
    // The feature id is the namespace prefix AND the syntactic component
    // reference the resolver/constraint lanes parse (`ACOMP<digits>`), so an
    // off-pattern id must fail loudly here — `split_component_namespace` would
    // silently fail to strip it later.
    if !crate::is_component_reference(&ctx.id) {
        return Err(format!(
            "component feature id '{}' must have the form ACOMP<digits>",
            ctx.id
        ));
    }
    let part_name = ctx
        .string("partName")
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .ok_or("missing required param `partName` (a parts-library entry name)")?;
    let entry = parts_library::active_entry(&part_name).ok_or_else(|| {
        format!("part '{part_name}' is not in the parts library")
    })?;

    // The rigid instance pose (expression-capable vec3s, ONE shared compose).
    let transform_value = ctx.param("transform");
    let translate = common::vec3_from_value(
        ctx.env,
        transform_value.and_then(|value| value.get("translate")),
        "transform.translate",
        [0.0, 0.0, 0.0],
    )?;
    let rotate_deg = common::vec3_from_value(
        ctx.env,
        transform_value.and_then(|value| value.get("rotateEulerDeg")),
        "transform.rotateEulerDeg",
        [0.0, 0.0, 0.0],
    )?;
    let matrix = common::compose_trs_matrix(
        translate,
        [
            rotate_deg[0].to_radians(),
            rotate_deg[1].to_radians(),
            rotate_deg[2].to_radians(),
        ],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    );
    let transform = AffineTransform::new(matrix)?;

    // Auto-ground: absent -> grounded iff no component precedes this one.
    let fixed = match ctx.param("isFixed") {
        None | Some(serde_json::Value::Null) => ctx.scene.components.is_empty(),
        Some(serde_json::Value::Bool(flag)) => *flag,
        Some(other) => return Err(format!("param `isFixed` must be a boolean, found {other}")),
    };

    // FAST lane, with the SELF-HEAL fallback (dirty entry or unreadable
    // snapshot -> re-execute the embedded document, rewrite the snapshot).
    let (restored, ports): (RestoredSnapshot, BTreeMap<String, PortRecord>) = if entry.dirty {
        heal(&part_name, &entry.document)?
    } else {
        match restore_solids(&entry.snapshot) {
            Ok(snapshot) => (snapshot, entry.ports.clone()),
            Err(_) => heal(&part_name, &entry.document)?,
        }
    };

    let members: Vec<(String, crate::BrepSolid)> = restored
        .solids
        .into_iter()
        .map(|solid| (solid.name, solid.solid))
        .collect();
    let source = serde_json::json!({
        "sourceKey": entry.source_key,
        "sourceSignature": entry.source_signature,
    });
    let (mut record, added) =
        create_component(&ctx.id, &part_name, fixed, source, transform, members)?;
    // The part's ports ride along, namespaced and posed like its members.
    let posed_ports = component::attach_component_ports(&mut record, &ports);

    // Stamp the part's captured metadata under the namespaced names.
    for (name, mut metadata_record) in restored.metadata {
        if let Some(serde_json::Value::String(parent)) = metadata_record.get_mut("inheritsFrom") {
            *parent = component::namespaced(&ctx.id, parent);
        }
        scene_metadata::set_record(&component::namespaced(&ctx.id, &name), metadata_record);
    }

    let mut result = FeatureResult::empty(ctx.id.clone(), ctx.feature_type.clone());
    result.added = added;
    result.components = vec![record];
    for (id, posed) in posed_ports {
        port::publish_port(&mut result, &id, posed)?;
    }
    Ok(result)
}

/// The self-heal lane: re-execute the entry's embedded document in isolation,
/// write the fresh snapshot back into the ACTIVE library entry, and decode it —
/// from here on the caller proceeds exactly as the fast lane.
fn heal(
    part_name: &str,
    document: &serde_json::Value,
) -> Result<(RestoredSnapshot, BTreeMap<String, PortRecord>), String> {
    let (payload, ports) = parts_library::rebuild_snapshot(document).map_err(|error| {
        format!("part '{part_name}': snapshot unreadable and document re-execution failed: {error}")
    })?;
    let restored = restore_solids(&payload)
        .map_err(|error| format!("part '{part_name}': rebuilt snapshot did not decode: {error}"))?;
    parts_library::heal_entry(part_name, payload, ports.clone());
    Ok((restored, ports))
}

/// Context-bar applicability ([`crate::feature_pipeline::context_offer`]):
/// never offered - components are inserted from the parts
/// library, not created from a selection.
pub fn context_applicable(_probe: &crate::feature_pipeline::SelectionProbe) -> bool {
    false
}

/// This feature's kernel-owned definition (name + parameter schema), aggregated
/// into the catalogue by `feature_pipeline/schema.rs`. The `partName` field is
/// a plain string for now (the insert flow fills it via `add_part_to_library`;
/// a component-selector widget is a Wave-3 form-engine addition), and the
/// `transform` field type is the nested pose object the form engine binds via
/// its path support (`["transform","translate"]`).
pub fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "ACOMP",
        "shortName": "ACOMP",
        "longName": "Assembly Component",
        "displayBuilder": false,
        "inputParamsSchema": {
            "id": {
                "type": "string",
                "default_value": null,
                "hint": "unique identifier for the component instance"
            },
            "partName": {
                "type": "string",
                "default_value": null,
                "hint": "parts-library entry this instance places (filled by the insert flow)"
            },
            "transform": {
                "type": "transform",
                "default_value": { "translate": [0, 0, 0], "rotateEulerDeg": [0, 0, 0] },
                "hint": "rigid instance pose: translate [x,y,z] + intrinsic-XYZ Euler degrees (expression-capable)"
            },
            "isFixed": {
                "type": "boolean",
                "default_value": false,
                "label": "Fixed",
                "hint": "grounded — the solver never moves it; the first component of an empty assembly is auto-fixed when unset"
            }
        }
    })
}

// BREP private tests: 6f2cd26580b95dc4
