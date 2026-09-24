//! PORT — Port. A named harness connection point: a base point and a unit
//! direction, plus whether it is a cable END (`termination`) or a pass-through
//! `waypoint` that joins spline paths into a network.
//!
//! # What it publishes (all under the feature id)
//!
//! - `scene.ports[{id}]` — the [`PortRecord`] (point, direction, kind,
//!   extension, display length, label). SPLINE anchors attach to it by id and
//!   the wire-harness tail builds its sided routing graph from it.
//! - `scene.points[{id}:Base]` — the base point (a helix / hole placement can
//!   start there; the display draws it as the port's vertex).
//! - `scene.axes[{id}]` — the port line as an axis (point + direction), so a
//!   revolve / sweep can reference a port's direction like a sketch line.
//! - `scene.frames[{id}]` — a plane frame whose normal IS the port direction
//!   (a sketch on a port is a connector face square to the wire).
//! - `scene.paths[{id}:PortLine}` — the drawn port line: from the base point
//!   along the direction for `displayLength` (a termination), or centred on the
//!   base point spanning `displayLength` (a waypoint, whose wire passes
//!   through). The committed-curve display draws it exactly like a helix edge,
//!   so a port is visible and pickable in the viewport.
//!
//! # Placement
//!
//! The base point is `transform.position` (world; the transform gizmo drags
//! it). The direction is the `directionRef` reference when one is set — a
//! FACE's outward normal at its midpoint, an EDGE's chord direction, a
//! PLANE / DATUM frame's normal, or a published sketch line — else the
//! `transform.rotationEuler` (degrees, intrinsic XYZ like every feature
//! transform) applied to `+X`. `reverseDirection` flips it. An unresolved
//! `directionRef` is reported through `unresolved` (contract rule 1: the caller
//! repairs; the port still publishes with the transform direction so nothing
//! downstream vanishes).
//!
//! `extension` is the straight run a wire keeps from the base point before it
//! may bend — an attached spline anchor takes it as its forward/backward
//! distance. `displayLength` only sizes the drawn line.

use crate::feature_pipeline::features::common;
use crate::feature_pipeline::{
    Axis, FeatureContext, FeatureResult, Frame, PortKind, PortRecord, SceneMap, ScenePoint,
};
use crate::{make_line, Vec3};

/// Default straight run before a wire may bend (the schema default).
const DEFAULT_EXTENSION: f64 = 2.0;
/// Default drawn line length (the schema default).
const DEFAULT_DISPLAY_LENGTH: f64 = 6.0;
/// The shortest drawn line — a zero-length line would publish no path.
const MIN_DISPLAY_LENGTH: f64 = 0.1;

pub fn execute(ctx: &FeatureContext) -> FeatureResult {
    match build(ctx) {
        Ok(result) => result,
        Err(error) => ctx.fail(format!("port: {error}")),
    }
}

fn build(ctx: &FeatureContext) -> Result<FeatureResult, String> {
    let id = if ctx.id.is_empty() { "Port" } else { ctx.id.as_str() };
    let mut result = FeatureResult::pass_through(ctx.id.clone(), ctx.feature_type.clone());

    let transform = ctx.param("transform");
    let position = common::vec3_from_value(
        ctx.env,
        transform.and_then(|t| t.get("position")),
        "transform.position",
        [0.0, 0.0, 0.0],
    )?;
    let rotation_deg = common::vec3_from_value(
        ctx.env,
        transform.and_then(|t| t.get("rotationEuler")),
        "transform.rotationEuler",
        [0.0, 0.0, 0.0],
    )?;
    let point = Vec3::new(position[0], position[1], position[2]);

    // The transform's own direction: the rotated +X axis.
    let rotated_x = super::datum::rotate_euler_xyz(
        Vec3::new(1.0, 0.0, 0.0),
        [
            rotation_deg[0].to_radians(),
            rotation_deg[1].to_radians(),
            rotation_deg[2].to_radians(),
        ],
    );
    let mut direction = rotated_x;
    if let Some(name) = common::first_reference_name(ctx.param("directionRef")) {
        match resolve_direction(ctx, &name)? {
            Some(resolved) => direction = resolved,
            None => result.unresolved.push(name),
        }
    }
    let mut direction = direction
        .normalized()
        .map_err(|_| "direction is a zero vector".to_string())?;
    if ctx
        .param("reverseDirection")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        direction = direction.scale(-1.0);
    }

    let kind = match ctx.param("kind").and_then(serde_json::Value::as_str) {
        Some("waypoint") => PortKind::Waypoint,
        Some("termination") | None => PortKind::Termination,
        Some(other) => return Err(format!("unknown port kind '{other}' (termination | waypoint)")),
    };
    let extension = common::optional_number(ctx, "extension")?
        .unwrap_or(DEFAULT_EXTENSION)
        .max(0.0);
    let display_length = common::optional_number(ctx, "displayLength")?
        .unwrap_or(DEFAULT_DISPLAY_LENGTH)
        .max(MIN_DISPLAY_LENGTH);
    let label = ctx
        .param("portName")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .unwrap_or(id)
        .to_string();

    let record = PortRecord {
        point,
        direction,
        kind,
        extension,
        display_length,
        label,
    };

    publish_port(&mut result, id, record)?;
    Ok(result)
}

/// The drawn port line: a termination's wire leaves along the direction; a
/// waypoint's wire passes through, so its line straddles the base point.
pub(crate) fn port_line(record: &PortRecord) -> (Vec3, Vec3) {
    let point = record.point;
    let direction = record.direction;
    let display_length = record.display_length;
    match record.kind {
        PortKind::Termination => (point, point.add(direction.scale(display_length))),
        PortKind::Waypoint => (
            point.sub(direction.scale(display_length * 0.5)),
            point.add(direction.scale(display_length * 0.5)),
        ),
    }
}

/// Publish one port's scene entities onto `result`: the record itself
/// (`ports[id]`), its base point (`{id}:Base`), its axis and its frame (both
/// `id`) and its drawn line (`{id}:PortLine`). The PORT feature and the ACOMP
/// feature (a placed part's ports, namespaced and posed) share this one shape.
pub(crate) fn publish_port(
    result: &mut FeatureResult,
    id: &str,
    record: PortRecord,
) -> Result<(), String> {
    let (line_start, line_end) = port_line(&record);
    result
        .paths
        .push((format!("{id}:PortLine"), vec![make_line(line_start, line_end)?]));
    result
        .points
        .push((format!("{id}:Base"), ScenePoint::model(record.point)));
    result.axes.push((
        id.to_string(),
        Axis { point: record.point, direction: record.direction },
    ));
    result.frames.push((
        id.to_string(),
        Frame::from_origin_normal(record.point, record.direction)?,
    ));
    result.ports.push((id.to_string(), record));
    Ok(())
}

/// The same entities written straight into a scene — the component re-pose
/// lane, which moves a placed part's ports without re-running its feature.
pub(crate) fn place_scene_port(
    scene: &mut SceneMap,
    id: &str,
    record: PortRecord,
) -> Result<(), String> {
    let (line_start, line_end) = port_line(&record);
    scene
        .paths
        .insert(format!("{id}:PortLine"), vec![make_line(line_start, line_end)?]);
    scene
        .points
        .insert(format!("{id}:Base"), ScenePoint::model(record.point));
    scene.axes.insert(
        id.to_string(),
        Axis { point: record.point, direction: record.direction },
    );
    scene.frames.insert(
        id.to_string(),
        Frame::from_origin_normal(record.point, record.direction)?,
    );
    scene.ports.insert(id.to_string(), record);
    Ok(())
}

/// Resolve a `directionRef` name to a unit direction: a resident FACE's
/// outward midpoint normal, a PLANE / DATUM frame's normal, a published sketch
/// line, or a resident EDGE's chord. `Ok(None)` = the name resolves to nothing
/// (an `unresolved` entry, never an error).
fn resolve_direction(ctx: &FeatureContext, name: &str) -> Result<Option<Vec3>, String> {
    if let Some(face) = ctx.scene.resolve_face(name) {
        let (_, normal) = common::face_point_normal(face)?;
        return Ok(Some(normal));
    }
    if let Some(frame) = ctx.scene.resolve_frame(name) {
        return Ok(Some(frame.z_axis));
    }
    if let Some(axis) = ctx.scene.resolve_axis(name) {
        return Ok(Some(axis.direction));
    }
    if let Some(edge) = ctx.scene.resolve_edge(name) {
        return Ok(Some(common::edge_axis(edge)?.direction));
    }
    Ok(None)
}

/// Context-bar applicability ([`crate::feature_pipeline::context_offer`]):
/// a selected face or edge can seat `directionRef`.
pub fn context_applicable(probe: &crate::feature_pipeline::SelectionProbe) -> bool {
    probe.faces > 0 || probe.edges > 0
}

pub fn schema() -> serde_json::Value {
    serde_json::json!({
    "type": "PORT",
    "shortName": "PORT",
    "longName": "Port",
    "displayBuilder": true,
    "inputParamsSchema": {
        "id": {
            "type": "string",
            "default_value": null,
            "hint": "Unique identifier for the port feature."
        },
        "portName": {
            "type": "string",
            "default_value": "Port",
            "label": "Port Name",
            "hint": "Logical name for this electrical termination or waypoint (the harness panel's endpoint label)."
        },
        "kind": {
            "type": "options",
            "options": [
                "termination",
                "waypoint"
            ],
            "default_value": "termination",
            "label": "Port Type",
            "hint": "Termination ports are cable endpoints. Waypoint ports join spline paths into a network; a wire enters on one side and leaves on the other."
        },
        "directionRef": {
            "type": "reference_selection",
            "selectionFilter": [
                "FACE",
                "EDGE",
                "PLANE",
                "DATUM"
            ],
            "multiple": false,
            "default_value": null,
            "label": "Direction Reference",
            "hint": "Optional geometry that sets the port direction: a face normal, an edge direction or a plane normal. Without it the placement rotation's X axis is the direction."
        },
        "transform": {
            "type": "transform",
            "default_value": {
                "position": [
                    0,
                    0,
                    0
                ],
                "rotationEuler": [
                    0,
                    0,
                    0
                ],
                "scale": [
                    1,
                    1,
                    1
                ]
            },
            "label": "Placement",
            "hint": "The port's base point (position) and, without a direction reference, its direction (the rotated X axis)."
        },
        "reverseDirection": {
            "type": "boolean",
            "default_value": false,
            "label": "Reverse Direction",
            "hint": "Flip the port direction vector."
        },
        "extension": {
            "type": "number",
            "default_value": 2,
            "label": "Extension",
            "hint": "How far the wire stays straight from the base point before bending. An attached spline anchor takes it as its forward and backward distance."
        },
        "displayLength": {
            "type": "number",
            "default_value": 6,
            "label": "Display Length",
            "hint": "Length of the drawn port line."
        }
    }
})
}

// BREP private tests: 6280223f6c64bc42
