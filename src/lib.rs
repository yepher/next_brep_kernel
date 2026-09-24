use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// The wasm panic hook every `#[wasm_bindgen]` entry point installs. Replaces
/// the `console_error_panic_hook` crate for the whole BREP family — reached as
/// `brep_kernel::panic_hook` and, through the render engine's re-export, as
/// `brep_render::brep_kernel::panic_hook`.
pub mod panic_hook;

#[path = "geometry/tolerance.rs"]
mod tolerance;
pub use tolerance::{
    curve_model_scale, model_scale, report_scale_migration, solid_model_scale, solid_scale,
    offset_construction_band, vertex_tolerance_from_edges, KernelTolerances,
    MeasuredTolerance, OFFSET_CONSTRUCTION_FLOOR, OFFSET_CONSTRUCTION_REL,
    VERTEX_MATCH_FLOOR,
};
// Per-entity measured tolerances (`per-entity-tolerances.md` slice S1): the
// lazy, capped band an individual edge or vertex earns from its own redundant
// representations, on top of the global policy above. Measurement only — no
// record field, no serialization; see the module doc for invariants I1/I2.
#[path = "geometry/entity_tolerance.rs"]
mod entity_tolerance;
pub use entity_tolerance::{EntityTolerances, EDGE_CAP_FRACTION, VERTEX_CAP_FRACTION};
#[path = "props/diagnostics.rs"]
mod diagnostics;
pub use diagnostics::{
    DiagnosticEvent, DiagnosticSeverity, KernelDiagnostics, KernelOutcome, KernelRefusal, KernelStage,
    OrRefuse, RefusalClass,
};
#[path = "geometry/polygon.rs"]
mod polygon;
#[path = "geometry/curve.rs"]
mod curve;
pub use curve::{
    make_arc, make_circle, make_hyperbola, make_line, make_parabola, uniform_clamped_knots,
    KnotVector, NurbsCurve, Vec4,
};
#[path = "blending/blend/mod.rs"]
mod blend;
pub use blend::{
    blend_closed_edge, blend_edge_variable, blend_open_edge, blend_smooth_chain,
    round_convex_corner,
};
#[path = "blending/fillet.rs"]
mod fillet;
pub use fillet::{
    chamfer_edge, chamfer_edge_angle, chamfer_edge_asymmetric, chamfer_edges_angle,
    chamfer_edges_asymmetric, fillet_edge, fillet_edges, fillet_edges_variable,
    fillet_edges_variable_law, fillet_edges_variable_vertex_radii,
};
#[path = "blending/law.rs"]
mod law;
pub use law::{LawSegment, RadiusLaw};
#[path = "geometry/fit.rs"]
mod fit;
pub use fit::{
    fit_polyline, interpolate_curve, interpolate_curve_closed, interpolate_curve_local,
    interpolate_curve_thinned, interpolate_curve_with_end_tangents, simplify_polyline,
    solve_banded, solve_dense, PolylineFit,
};
#[path = "geometry/image_curve.rs"]
mod image_curve;
pub use image_curve::{affine_image_curve, image_curve, image_curve_pair, ImageCurve, ImageCurveTier};
#[path = "geometry/surface.rs"]
mod surface;
pub use surface::{
    carrier_preview_patch, make_cone_surface, make_cylinder_surface, make_extrusion, make_plane,
    make_revolution, make_sphere_surface, make_sphere_surface_framed, make_torus_surface,
    NurbsSurface,
};
#[path = "geometry/analytic_surface.rs"]
mod analytic_surface;
pub use analytic_surface::{
    circle_angle_to_parameter, intersect_analytic_pair, revolution_structure, AnalyticSurface,
    RevolutionFrame, RevolutionStructure,
};
#[path = "geometry/sphere_chart.rs"]
mod sphere_chart;
pub use sphere_chart::{
    canonicalize_points, chart_grid_divisions, chart_grid_parameters, corner_index,
    cube_edge_corner, cube_edge_divisions, cube_edge_index, cube_edge_parameters,
    cube_edge_parts, Chart, ChartSegment, ChartSide, SphereAtlas,
    SphericalRegion, CHART_COUNT, CUBE_CORNER_COUNT, CUBE_EDGE_COUNT,
};
#[path = "brep/solid_codec.rs"]
mod solid_codec;
pub use solid_codec::{decode_solid, encode_solid, SolidNames, SOLID_CODEC_VERSION};
#[path = "brep/topology.rs"]
mod topology;
pub use topology::{
    make_box_brep, make_cylinder_brep, make_pyramid_brep, BrepSolid, FaceRecord, IssueKind,
    ValidationReport,
};
#[path = "brep/soundness.rs"]
mod soundness;
pub use soundness::{
    solid_connectivity, solid_self_intersections, ConnectivityReport, FaceCrossing,
    SelfIntersectionOptions, SelfIntersectionReport, ShellConnectivity,
};
#[path = "brep/topology_arena.rs"]
mod topology_arena;
pub use topology_arena::{
    ArenaCoedge, ArenaEdge, ArenaFace, ArenaLoop, ArenaShell, ArenaVertex, CoedgeId, EdgeId,
    FaceId, LoopId, ShellId, TopologyArena, VertexId,
};
#[path = "brep/analytic_topology.rs"]
mod analytic_topology;
pub use analytic_topology::{
    make_cone_brep, make_sphere_brep, make_sphere_brep_framed, make_torus_brep,
};
#[path = "construction/sweep_topology.rs"]
mod sweep_topology;
pub use sweep_topology::{
    extrude_profile_brep, extrude_profile_brep_draft, fit_helix_curve, helix_sample_points,
    profile_anchor, rib_from_profile, RibExtrusion,
    sweep_profile_along_chain, sweep_profile_along_chain_with_stations, sweep_profile_along_path,
    sweep_profile_along_path_anchored,
    sweep_profile_helix,
    sweep_profile_twisted, sweep_profile_twisted_anchored, ProfileAnchor, SectionPlacement,
};
#[path = "construction/revolve_topology.rs"]
mod revolve_topology;
pub use revolve_topology::{revolve_profile_brep, revolve_profile_brep_named};
#[path = "construction/loft_topology.rs"]
mod loft_topology;
pub use loft_topology::{
    loft_profile_brep, loft_profile_brep_closed, loft_profile_brep_guided,
    loft_profile_brep_guided_frame, loft_profile_brep_tangent,
};
#[path = "brep/transform_topology.rs"]
mod transform_topology;
pub use transform_topology::{mirror_brep, transform_brep, AffineTransform};
#[path = "edit/split.rs"]
mod split;
pub use split::{
    split_solid_by_face_surface, split_solid_by_plane, split_solid_by_surface, SplitSurface,
};
#[path = "props/mass_properties.rs"]
mod mass_properties;
pub use mass_properties::{
    curve_arc_length, edge_arc_length, face_area, face_boundary_length, face_volume_contribution,
    parameter_space_area, solid_edge_length_total, solid_mass_properties,
    solid_mass_properties_full, solid_signed_volume, trim_polygons, DensityMassProperties,
    FullMassProperties, MassProperties,
};
#[path = "meshing/tessellation.rs"]
mod tessellation;
pub use tessellation::{tessellate_brep, tessellate_face, TessellationOptions};
#[path = "meshing/watertight_tessellation/mod.rs"]
mod watertight_tessellation;
pub use watertight_tessellation::{
    sample_edge_polylines, sample_edges_encoded, tessellate_brep_watertight,
    tessellate_brep_watertight_face_stride, tessellate_brep_watertight_face_stride_with_samples,
};
// The Rust feature-history execution engine (migration-plan Stage 4/5 foundation).
// Deserializes the serialized `{ type, inputParams, ... }` descriptor shape, runs
// each feature against a live scene-map of resident handles, and returns per-feature
// results (handles + face/edge names). See feature_pipeline/mod.rs for the contract.
#[path = "feature_pipeline/mod.rs"]
mod feature_pipeline;
pub use feature_pipeline::execute_history_json;
pub use feature_pipeline::{first_reference_name, reference_names};
// Typed, in-process pipeline surface for native consumers (brep-render): run a
// whole history and read the results without a JSON round trip.
pub use feature_pipeline::{
    clear_history_cache, execute_history, execute_history_observed, AddedSolid, Axis,
    ComponentRecord, FeatureDescriptor, FeatureResult, Frame, HistoryProgress, HistoryRequest,
    HistoryResult, PortKind, PortRecord, ProfileLoop, ScenePoint, SketchProfile,
};
// Wire harness: the document's `wireHarness` block (connections), the tail's
// routing report (per-connection status / length / route, per-segment bundle),
// and the sided-port vocabulary the spline attachments share.
pub use feature_pipeline::wire_harness::{
    bundle_diameter, Attachment as SplineAttachment, PortSide, RouteResult, RouteStatus,
    WireHarnessBundle, WireHarnessConnection,
    WireHarnessEndpoint, WireHarnessReport, WireHarnessState, WIRE_HARNESS_FEATURE_ID,
    WIRE_HARNESS_FEATURE_TYPE,
};
// PMI: the document's `pmi` block (views + annotations), the type table with
// its schemas / selection predicates, the tail's resolved report, and the
// presentation layout shared by the viewport and the AP242 export.
pub use feature_pipeline::pmi::{
    clamp_text_size, format_dimension, format_number, pmi_schema_catalogue, pmi_type,
    resolve_state as pmi_resolve_state, FcfFrame, PmiAnnotation, PmiAnnotationReport, PmiCamera,
    PmiDisplay, PmiGeometry, PmiPlane, PmiProjection, PmiReport, PmiState, PmiStatus, PmiTypeDef, PmiView,
    PmiViewReport, ToleranceBlock, ToleranceMode, PMI_TYPES,
};
pub use feature_pipeline::pmi::layout::{present as pmi_present, LayoutStyle as PmiLayoutStyle, Presentation as PmiPresentation};
pub use feature_pipeline::pmi::annotations::fcf::{characteristic as pmi_characteristic, Characteristic as PmiCharacteristic, CHARACTERISTICS as PMI_CHARACTERISTICS};
// The sketch loop-id write-back: stamps each closed loop's stable id onto its
// geometries so per-loop face names survive deleting the edge an id came from.
// The editor calls it when a sketch is committed.
pub use feature_pipeline::assign_sketch_loop_ids;
// Sheet-metal flat-pattern (unfold) → 2D vector export, keyed by resident handle.
pub use feature_pipeline::{flat_pattern_dxf, flat_pattern_svg, is_sheet_metal_handle};
// Native IMPORT3D payload: finished solids → the `io/snapshot` container with
// IMPORT3D's naming stamped on, i.e. the `inputParams.nativeBrep` an import lane
// bakes into a part document (`features/import3d.rs` reads it back verbatim).
pub use feature_pipeline::{native_import_payload, native_import_payload_with_appearance};
// The scene-metadata isolation bracket a payload producer holds across the
// encode, so ambient records of the LIVE document never leak into a new part.
pub use feature_pipeline::IsolatedSceneMetadata;
// The imported-colour read of the name-keyed scene-metadata store. Thread-local
// like the rest of it, so an off-thread runner reads it and ships the result.
pub use feature_pipeline::scene_metadata_colors_json;
// The assemblies parts library (unique part payloads for ACOMP instances):
// insert / update-refresh mutations and the save-side serialization surface.
pub use feature_pipeline::{
    add_part_to_library, install_parts_library, missing_library_parts, parts_library_json,
    parts_library_map, parts_library_revision, refresh_library_entry, stable_json_hash, PartsLibraryEntry,
    PartsLibraryMap,
};
// Assembly constraint state + exported ABI (state/statuses/DOF/overlay reads,
// constraint CRUD with auto-solve, the document pose/isFixed write-back fold).
pub use feature_pipeline::assembly::{
    assembly_add_constraint_json, assembly_apply_document_json,
    assembly_apply_inferred_constraints_json, assembly_dof_json,
    assembly_infer_constraints_json, assembly_inferable_types_json,
    assembly_move_constraint_json, assembly_overlay_json, assembly_pose_updates_json,
    assembly_remove_constraint_json, assembly_run_solve_json,
    assembly_set_constraint_enabled_json, assembly_set_constraint_open_json,
    assembly_state_json, assembly_statuses_json, assembly_update_constraint_json,
    constraint_schema_catalogue,
};
pub use feature_pipeline::{AssemblyState, ConstraintEntry};
// The ONE matrix → `{translate, rotateEulerDeg}` (intrinsic XYZ, degrees) pose
// encoder, shared by the solver write-back and any out-of-crate ACOMP author.
pub use feature_pipeline::assembly::transform_to_pose_params;
// The feature-schema catalogue (feature definitions: name + `inputParamsSchema`),
// so a native UI can drive schema-driven feature dialogs without the JSON export.
pub use feature_pipeline::feature_schema_catalogue;
// Selection-context applicability: the context bar's per-feature/-constraint
// show/no-show predicates, plus the constraint type table they pair with.
pub use feature_pipeline::assembly::{constraint_type, ConstraintTypeDef, CONSTRAINT_TYPES};
pub use feature_pipeline::{feature_context_applicable, SelectionProbe};
// The pipeline expression evaluator (the history's `expressions` + `configurator`
// variable sheet), so a native UI (brep-render's engine-native sketcher) can
// evaluate a LIVE dimension `valueExpr` against the same variables a committed
// solve would (`features/sketch.rs` `pre_evaluate_expressions`). `Env::build`/
// `eval` are pure arithmetic (no `std::time`), so this stays wasm-clean.
pub use feature_pipeline::Env;

/// One-shot: build an [`Env`] from a history's `expressions` source + its
/// `configurator` JSON, then evaluate `source` to a scalar. The small surface the
/// engine-native sketcher's live dimension-value edit uses (deliverable S5.0) —
/// equivalent to `Env::build(expressions, configurator_json)?.eval(source)`.
pub fn eval_expression(
    expressions: &str,
    configurator_json: &serde_json::Value,
    source: &str,
) -> Result<f64, String> {
    Env::build(expressions, configurator_json).and_then(|env| env.eval(source))
}

// In the shared-memory-threaded wasm build, re-export wasm-bindgen-rayon's
// thread-pool initializer as `initThreadPool(numThreads)`. The JS side must
// await it once after wasm init (needs a cross-origin-isolated page —
// COOP/COEP — for SharedArrayBuffer). With the pool live, the ordinary
// `tessellate_watertight_buffers` parallelizes its per-face loop across threads
// (see `tessellate_faces_stride` under the `parallel` feature).
#[cfg(feature = "wasm-threads")]
pub use wasm_bindgen_rayon::init_thread_pool;
#[path = "geometry/projection.rs"]
mod projection;
pub use projection::{
    project_point_to_curve, project_point_to_surface, project_point_to_surface_seeded,
    CurveProjection, SurfaceProjection,
};
#[path = "intersect/curve_surface_intersection.rs"]
mod curve_surface_intersection;
pub use curve_surface_intersection::{intersect_curve_surface, CurveSurfaceIntersection};
#[path = "intersect/curve_curve_intersection.rs"]
mod curve_curve_intersection;
pub use curve_curve_intersection::{intersect_curves, CurveCurveIntersection};
#[path = "intersect/surface_surface_intersection.rs"]
mod surface_surface_intersection;
pub use surface_surface_intersection::{
    intersect_surfaces, intersect_surfaces_supplemental, SurfaceIntersectionCurve,
    SurfaceIntersectionOptions,
};
pub(crate) use surface_surface_intersection::TRANSVERSE_SEED_CROSS;
#[path = "brep/classification.rs"]
mod classification;
pub use classification::{
    classify_point, parameter_point_in_face, PointClass, PointClassification, PolygonClass,
    SolidClassifier,
};
#[path = "geometry/spatial.rs"]
mod spatial;
pub use spatial::{Aabb, Bvh};
#[path = "brep/pair_classification.rs"]
mod pair_classification;
pub use pair_classification::{
    classify_surface_pair, classify_surface_pair_cached, SurfaceClassifyData,
    SurfacePairClassification, SurfacePairRelation,
};
#[path = "intersect/arrangement.rs"]
mod arrangement;
pub use arrangement::{
    arrange_segments, point_in_polygon, segment_intersection, ArrangementPiece, ArrangementRegion,
    CycleUse, Segment2, Vec2,
};
#[path = "geometry/pcurve.rs"]
mod pcurve;
pub use pcurve::{
    build_pcurve_on_surface, build_pcurve_on_surface_marched, build_pcurve_on_surface_range,
    build_pcurve_on_surface_range_dense,
};
#[path = "csg/imprint.rs"]
mod imprint;
pub use imprint::{
    build_imprints, EdgeSplitRecord, FaceImprints, FaceKey, FacePcurve, ImprintOptions,
    ImprintPieceRecord, ImprintResultRecord, ImprintVertex,
};
#[path = "csg/edge_split.rs"]
mod edge_split;
pub use edge_split::{apply_edge_splits, apply_edge_splits_with_map};
#[path = "csg/fragment.rs"]
mod fragment;
pub use fragment::{
    fragment_face, fragment_solid, FaceFragmentRecord, FragmentCoedge, FragmentEdgeSource,
    FragmentLoop,
};
#[path = "csg/boolean/mod.rs"]
mod boolean;
pub use boolean::{
    boolean_operation, boolean_operation_nary, boolean_operation_with_diagnostics,
    BooleanOperation, BooleanOptions,
};
#[path = "csg/oracle.rs"]
mod oracle;
pub use oracle::{
    boolean_residual_fusables, boolean_semantic_disagreement, boolean_semantic_disagreement_nary,
    OracleReport, ResidualFusable, ResidualKind, SemanticDisagreement, DISAGREEMENT_THRESHOLD,
};
#[path = "healing/heal.rs"]
mod heal;
// The pointwise offset evaluator — ONE definition of "the offset of a surface
// at a parameter", shared by `offset_surface`'s Greville sampling, the blend
// march's tangency residual, and the push-face offset residual gates. Declared
// before `offset` because that module is its first consumer.
#[path = "offset/point.rs"]
mod offset_point;
pub use offset_point::{OffsetEvaluator, OffsetNormal, OffsetSample};
// The LOCUS-level offset seam — ONE definition of "intersect two offset
// analytic carriers in closed form", shared by the blend's corner closures.
// The pointwise evaluator above answers "where is the offset at this (u, v)";
// this answers "where do two offsets MEET", which is what the corner solves
// need and what no pointwise evaluator can produce (audit §9.4, §11).
// Crate-internal: no consumer outside the kernel names these.
#[path = "offset/analytic_pair.rs"]
mod offset_analytic_pair;
// The shared re-trim helpers — ONE definition of "rebuild a face's carrier and
// pcurves around a boundary that has already moved", shared by push-face,
// face-move and the direct-edit heals. This is the RE-INTERSECTION sense of
// trimming; offset-shell's parametric-image trimming is a different operation
// and deliberately not here (see the module doc). Crate-internal: no consumer
// outside the kernel names these.
#[path = "offset/retrim.rs"]
mod offset_retrim;
// The RE-INTERSECTION seam — ONE definition of "intersect two carriers to
// remake a trim boundary", with the exact analytic lane first and the general
// marched lane behind it. `offset_retrim` above re-trims a face around a
// boundary that has ALREADY moved; this is the step that moves it, and it is
// what makes a neighbour re-intersection surface-type-blind the way
// offset-shell's imprint driver already is. Crate-internal: no consumer outside
// the kernel names these.
#[path = "offset/reintersect.rs"]
mod offset_reintersect;
// The MEASURED half of the tolerance model: the deviation actually observed
// between an offset construction and the geometry it was built to reproduce.
// Everything in `geometry/tolerance.rs` above `MeasuredTolerance` DERIVES a
// band from a size before the geometry exists; this measures what the
// construction then did, and compares it against that band. The comparison is
// not itself a gate: the deviation rides the transient construction result
// (`OffsetFaceCarrier::deviation`, `#[serde(skip)]`) and refusing on an
// exceedance is the caller's call, not this module's. Declared before `offset`
// because that module is its first consumer.
#[path = "offset/measure.rs"]
mod offset_measure;
pub use offset_measure::{
    measure_edge_against_pcurve_image, measure_surface_fit_against_pointwise_offset,
    span_midpoint_error, vertex_endpoint_gap,
};
#[path = "offset/offset.rs"]
mod offset;
pub use offset::{
    offset_face_carrier, offset_face_carrier_measured, offset_face_carrier_sided, offset_surface,
    offset_surface_measured, offset_surface_measured_sided, CarrierDeviation, CarrierExtension,
    MeasuredOffsetSurface, OffsetFaceCarrier, OffsetSurfaceLane,
};
#[path = "offset/thicken.rs"]
mod thicken;
pub use thicken::{thicken_face_sheet, thicken_trimmed_sheet};
#[path = "healing/coalesce.rs"]
mod coalesce;
pub use coalesce::{concatenate_exact_curve_pieces, merge_curve_continuation_edges};
#[path = "healing/face_merge.rs"]
mod face_merge;
// Retained for potential mesh-import repair; no exact operation may fall
// back to it (offset shell must produce real surfaces or fail loudly).
#[allow(dead_code)]
#[path = "healing/faceted_repair.rs"]
mod faceted_repair;
pub use face_merge::{merge_same_surface_faces, merge_same_surface_faces_excluding};
pub use faceted_repair::mesh_to_faceted_brep;
#[path = "offset/offset_shell.rs"]
mod offset_shell;
pub use offset_shell::{
    offset_shell, offset_shell_with_diagnostics, OffsetFaceRole, OffsetShellFaceImageRecord,
    OffsetShellResultRecord,
};
#[path = "io/step_matrix.rs"]
mod step_matrix;
#[path = "io/step.rs"]
mod step;
pub use step::{
    assembly_export_tree, audit_step_manifold, audit_step_pcurves, export_step,
    export_step_assembly, export_step_assembly_report, export_step_report,
    export_step_report_named, Mat4, StepAssemblyExport, StepExportOccurrence, StepExportProduct,
    StepExportReport, StepPmi,
};
// Imported appearance (colour) — the carrier between an importer that READ a
// colour and the pipeline that stamps it as name-keyed scene metadata. Not a
// storage layer: see the module doc for the `{"color": "#RRGGBB"}` convention.
#[path = "io/appearance.rs"]
mod appearance;
pub use appearance::{BodyAppearance, ImportedColor, COLOR_METADATA_KEY};
#[path = "io/step_import/mod.rs"]
mod step_import;
pub use step_import::{
    import_step, import_step_report, import_step_with_appearance, read_step_assembly,
    read_step_pmi, StepAssembly, StepOccurrence, StepProduct,
};
/// Test seams for the Part 21 text codec of the PMI writer / reader.
#[doc(hidden)]
pub fn io_step_text_for_tests(value: &str) -> String {
    step::pmi::step_text(value)
}
#[doc(hidden)]
pub fn io_step_decode_text_for_tests(value: &str) -> String {
    step_import::decode_step_text_for_tests(value)
}
#[path = "io/iges/mod.rs"]
mod iges;
pub use iges::{export_iges, import_iges};
#[path = "io/mesh_io.rs"]
mod mesh_io;
pub use mesh_io::{
    read_binary_stl, read_obj, write_binary_stl, write_obj, ObjReadResult, StlReadResult,
};
// Native serialized exact-BREP snapshot: the assemblies parts-library fast
// lane (instant component insert from a cached payload; fails detectably so
// the ACOMP self-heal lane can re-execute the embedded part document).
#[path = "io/snapshot.rs"]
mod snapshot;
pub use snapshot::{
    restore_solids, snapshot_resident_solids, snapshot_solids, RestoredSnapshot, RestoredSolid,
    SNAPSHOT_FORMAT_VERSION,
};
#[path = "solvers/linear_algebra.rs"]
mod solver_linear_algebra;
#[path = "solvers/sketch_solver.rs"]
mod sketch_solver;
pub use sketch_solver::{
    sketch_id_key, solve_sketch, solve_sketch_from_json, SketchSolverSettings, SolveSketchRequest,
};
#[path = "solvers/assembly_solver.rs"]
mod assembly_solver;
pub use assembly_solver::{
    solve_assembly, AssemblyBody, AssemblyMate, AssemblySolution, AssemblySolveOptions, BodyPose,
    MateAlign, MateAxis, MateKind, MatePlane, MateResidualReport, SolveStrategy,
};
// Assembly selection resolution (build-spec §5): a namespaced selection ref
// resolved to an analytic frame (plane/axis/sphere/circle/line/point) read from
// the exact BREP, feeding `MateKind` inputs in component-local coordinates.
#[path = "solvers/assembly_resolve.rs"]
mod assembly_resolve;
pub use assembly_resolve::{
    is_component_reference, resolve_component_point, resolve_edge_selection,
    resolve_face_selection, resolve_named_selection, resolve_vertex_selection,
    split_component_namespace, ResolveError, SelectionGeometry,
};
#[path = "edit/direct_edit.rs"]
mod direct_edit;
pub use direct_edit::{
    delete_face_and_heal, delete_faces_and_heal, move_faces, offset_freeform_face,
    offset_revolution_face, offset_ruled_face, offset_sphere_face, offset_torus_face,
    resolve_face_by_point,
};
#[path = "healing/sew.rs"]
mod sew;
pub use sew::{sew_solid, split_pinched_vertices, SewReport};
#[path = "meshing/mesh_weld.rs"]
mod mesh_weld;
#[path = "meshing/mesh_segment.rs"]
mod mesh_segment;
pub use mesh_segment::{
    mesh_regions_to_brep, segment_mesh_faces, MeshRegion, MeshSegmentation, RegionCarrier,
    SegmentOptions, UNASSIGNED_REGION,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    pub fn normalized(self) -> Result<Self, String> {
        let length = self.length();
        if length <= 1e-12 {
            return Err("Vec3.normalized: zero-length vector".into());
        }
        Ok(self.scale(1.0 / length))
    }
    pub fn perpendicular(self) -> Result<Self, String> {
        let x = self.x.abs();
        let y = self.y.abs();
        let z = self.z.abs();
        let axis = if x <= y && x <= z {
            Self::new(1.0, 0.0, 0.0)
        } else if y <= z {
            Self::new(0.0, 1.0, 0.0)
        } else {
            Self::new(0.0, 0.0, 1.0)
        };
        self.cross(axis).normalized()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Mesh {
    pub positions: Vec<f64>,
    pub normals: Vec<f64>,
    pub indices: Vec<u32>,
    pub face_ids: Vec<u32>,
}

impl Mesh {
    fn push_vertex(&mut self, point: Vec3, normal: Vec3) -> u32 {
        self.positions.extend([point.x, point.y, point.z]);
        self.normals.extend([normal.x, normal.y, normal.z]);
        (self.positions.len() / 3 - 1) as u32
    }

    fn quad(&mut self, points: [Vec3; 4], normal: Vec3, face_id: u32) {
        let base = self.push_vertex(points[0], normal);
        for point in points.iter().skip(1) {
            self.push_vertex(*point, normal);
        }
        self.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        self.face_ids.extend([face_id, face_id]);
    }

    pub fn signed_volume(&self) -> f64 {
        self.indices
            .chunks_exact(3)
            .map(|tri| {
                let point = |index: u32| {
                    let i = index as usize * 3;
                    Vec3::new(
                        self.positions[i],
                        self.positions[i + 1],
                        self.positions[i + 2],
                    )
                };
                point(tri[0]).dot(point(tri[1]).cross(point(tri[2]))) / 6.0
            })
            .sum()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.positions.len() % 3 != 0 || self.normals.len() != self.positions.len() {
            return Err("invalid vertex or normal buffer".into());
        }
        self.validate_geometry()
    }

    /// Geometry-only validation for consumers that use positions + indices but
    /// NOT normals (e.g. `signed_volume`, whose tetrahedron sum never reads a
    /// normal). Mesh-only solids such as sheet-metal bodies supply metrics
    /// meshes with an empty normal buffer, which is legitimate for volume.
    pub fn validate_geometry(&self) -> Result<(), String> {
        if self.positions.len() % 3 != 0 {
            return Err("invalid vertex buffer".into());
        }
        if self.indices.len() % 3 != 0 {
            return Err("index buffer is not triangular".into());
        }
        let count = self.positions.len() / 3;
        if self.indices.iter().any(|index| *index as usize >= count) {
            return Err("index outside vertex buffer".into());
        }
        if !self.positions.iter().all(|value| value.is_finite()) {
            return Err("non-finite vertex".into());
        }
        Ok(())
    }
}

pub fn make_box(size_x: f64, size_y: f64, size_z: f64) -> Mesh {
    let x = size_x.abs();
    let y = size_y.abs();
    let z = size_z.abs();
    let p = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(x, 0.0, 0.0),
        Vec3::new(x, y, 0.0),
        Vec3::new(0.0, y, 0.0),
        Vec3::new(0.0, 0.0, z),
        Vec3::new(x, 0.0, z),
        Vec3::new(x, y, z),
        Vec3::new(0.0, y, z),
    ];
    let mut mesh = Mesh::default();
    mesh.quad([p[0], p[3], p[2], p[1]], Vec3::new(0.0, 0.0, -1.0), 0);
    mesh.quad([p[4], p[5], p[6], p[7]], Vec3::new(0.0, 0.0, 1.0), 1);
    mesh.quad([p[0], p[1], p[5], p[4]], Vec3::new(0.0, -1.0, 0.0), 2);
    mesh.quad([p[3], p[7], p[6], p[2]], Vec3::new(0.0, 1.0, 0.0), 3);
    mesh.quad([p[0], p[4], p[7], p[3]], Vec3::new(-1.0, 0.0, 0.0), 4);
    mesh.quad([p[1], p[2], p[6], p[5]], Vec3::new(1.0, 0.0, 0.0), 5);
    mesh
}

pub fn make_cylinder(radius: f64, height: f64, segments: usize) -> Mesh {
    let radius = radius.abs();
    let height = height.abs();
    let segments = segments.max(3);
    let mut mesh = Mesh::default();
    for i in 0..segments {
        let a = std::f64::consts::TAU * i as f64 / segments as f64;
        let b = std::f64::consts::TAU * (i + 1) as f64 / segments as f64;
        let (sa, ca) = a.sin_cos();
        let (sb, cb) = b.sin_cos();
        let pa = Vec3::new(radius * ca, 0.0, -radius * sa);
        let pb = Vec3::new(radius * cb, 0.0, -radius * sb);
        let ta = Vec3::new(pa.x, height, pa.z);
        let tb = Vec3::new(pb.x, height, pb.z);
        mesh.quad(
            [pa, pb, tb, ta],
            Vec3::new((ca + cb) * 0.5, 0.0, -(sa + sb) * 0.5),
            0,
        );
        let bottom = mesh.push_vertex(Vec3::default(), Vec3::new(0.0, -1.0, 0.0));
        let bi = mesh.push_vertex(pb, Vec3::new(0.0, -1.0, 0.0));
        let bj = mesh.push_vertex(pa, Vec3::new(0.0, -1.0, 0.0));
        mesh.indices.extend([bottom, bi, bj]);
        mesh.face_ids.push(1);
        let top = mesh.push_vertex(Vec3::new(0.0, height, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let ti = mesh.push_vertex(ta, Vec3::new(0.0, 1.0, 0.0));
        let tj = mesh.push_vertex(tb, Vec3::new(0.0, 1.0, 0.0));
        mesh.indices.extend([top, ti, tj]);
        mesh.face_ids.push(2);
    }
    mesh
}


mod abi;
pub use abi::*;

// BREP private tests: e9a96ee1118285df
// BREP private tests: 802ecfa7fc0314e2
