use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum StepBody {
    SolidBrep(usize),
    BrepWithVoids(usize),
    /// One CLOSED_SHELL of a SHELL_BASED_SURFACE_MODEL. That entity is a surface-
    /// model *container* wrapping a LIST of shells; when those shells are closed
    /// they are full solid bodies authored via the surface-model form rather than
    /// MANIFOLD_SOLID_BREP, so each expands to its own body. `model_ref` is the
    /// SHELL_BASED_SURFACE_MODEL entity (identity), `shell_ref` the shell to build.
    SurfaceModelShell { model_ref: usize, shell_ref: usize },
}

impl StepBody {
    /// Total order for a stable, reproducible body sequence. Bodies are gathered
    /// from a HashMap (nondeterministic iteration), so callers sort by this before
    /// building; sibling shells of one surface model share `model_ref`, hence the
    /// `shell_ref` tiebreaker keeps their order fixed run-to-run.
    fn order_key(self) -> (usize, usize) {
        match self {
            Self::SolidBrep(entity_ref) | Self::BrepWithVoids(entity_ref) => (entity_ref, 0),
            Self::SurfaceModelShell {
                model_ref,
                shell_ref,
            } => (model_ref, shell_ref),
        }
    }
}

pub(super) fn step_body_for_entity(entity_ref: usize, entity: &Entity) -> Option<StepBody> {
    if entity.has("BREP_WITH_VOIDS") {
        Some(StepBody::BrepWithVoids(entity_ref))
    } else if entity.has("MANIFOLD_SOLID_BREP") || entity.has("FACETED_BREP") {
        Some(StepBody::SolidBrep(entity_ref))
    } else {
        None
    }
}

/// The CLOSED_SHELL references wrapped by a SHELL_BASED_SURFACE_MODEL
/// (`SHELL_BASED_SURFACE_MODEL(name, (#shell, ...))`), in author order.
fn surface_model_shell_refs(resolver: &Resolver, model_ref: usize) -> Option<Vec<usize>> {
    let args = resolver
        .get(model_ref)
        .ok()?
        .find("SHELL_BASED_SURFACE_MODEL")?;
    let list = args.get(1)?.as_list().ok()?;
    let mut refs = Vec::with_capacity(list.len());
    for value in list {
        refs.push(value.as_ref_id().ok()?);
    }
    Some(refs)
}

/// Append every solid body an entity contributes. Most bodies map 1→1 via
/// [`step_body_for_entity`]; a SHELL_BASED_SURFACE_MODEL fans out 1→many, one
/// [`StepBody::SurfaceModelShell`] per wrapped shell (a closed shell reconstructs
/// as a solid; a genuinely open shell simply fails the shared validation gate).
fn push_step_bodies(
    resolver: &Resolver,
    entity_ref: usize,
    entity: &Entity,
    out: &mut Vec<StepBody>,
) {
    if let Some(body) = step_body_for_entity(entity_ref, entity) {
        out.push(body);
        return;
    }
    if entity.has("SHELL_BASED_SURFACE_MODEL") {
        if let Some(shell_refs) = surface_model_shell_refs(resolver, entity_ref) {
            for shell_ref in shell_refs {
                out.push(StepBody::SurfaceModelShell {
                    model_ref: entity_ref,
                    shell_ref,
                });
            }
        }
    }
}

pub(super) fn build_step_body(resolver: &Resolver, body: StepBody) -> Result<BrepSolid, String> {
    match body {
        StepBody::SolidBrep(entity_ref) => build_solid(resolver, entity_ref),
        StepBody::BrepWithVoids(entity_ref) => build_brep_with_voids(resolver, entity_ref),
        StepBody::SurfaceModelShell { shell_ref, .. } => {
            build_solid_from_shell(resolver, shell_ref)
        }
    }
}

/// The colours the file's presentation entities assign to ONE built body:
/// the body's own, plus one per face in the body's shell/face order.
///
/// **How the per-face pairing is trusted.** Every lane of [`build_step_body`]
/// builds a shell's faces by `zip`-ing straight down the STEP shell's face-ref
/// list ([`build_shell_faces_in_namespace`] / [`build_solid_from_shell`]), and
/// every pass that runs afterwards — the pinched-vertex split, the orientation
/// flips, and the occurrence lane's `transform_brep` — mutates faces IN PLACE.
/// So face `i` of built shell `s` IS `#face_refs[s][i]`. This function re-walks
/// the same shell/face refs and CHECKS that pairing per shell (shell count, then
/// each shell's face count): a mismatch means the two walks have drifted, and
/// the honest answer is no face colours at all rather than colours on the wrong
/// faces. A body colour is unaffected by the check — it is keyed on the body's
/// own entity, not on a position.
pub(super) fn step_body_appearance(
    resolver: &Resolver,
    styles: &StyleTable,
    body: StepBody,
    solid: &BrepSolid,
) -> BodyAppearance {
    let mut appearance = BodyAppearance::default();
    if styles.is_empty() {
        return appearance;
    }
    let shell_refs = step_body_claimed_shells(resolver, body);

    // The body's own colour: the file may style the solid entity, or (for a
    // surface model) the model container, or the OUTER shell that stands for the
    // body. Only the outer shell — a BREP_WITH_VOIDS void shell is a cavity, not
    // the body, and its style is not the body's.
    let mut body_items: Vec<usize> = match body {
        StepBody::SolidBrep(entity_ref) | StepBody::BrepWithVoids(entity_ref) => vec![entity_ref],
        StepBody::SurfaceModelShell { model_ref, .. } => vec![model_ref],
    };
    body_items.extend(shell_refs.as_ref().and_then(|shells| shells.first()).copied());
    appearance.body = body_items
        .into_iter()
        .find_map(|item_ref| styles.color_for(item_ref));

    // Per-face colours, positionally paired and length-checked per shell.
    let Some(shell_refs) = shell_refs else {
        return appearance;
    };
    if shell_refs.len() != solid.shells.len() {
        return appearance;
    }
    let mut faces: Vec<Option<ImportedColor>> = Vec::new();
    let mut any = false;
    for (shell_ref, shell) in shell_refs.into_iter().zip(&solid.shells) {
        let Some(face_refs) = shell_face_refs(resolver, shell_ref) else {
            return appearance;
        };
        if face_refs.len() != shell.faces.len() {
            return appearance;
        }
        for face_ref in face_refs {
            let color = styles.color_for(face_ref);
            any |= color.is_some();
            faces.push(color);
        }
    }
    if any {
        appearance.faces = faces;
    }
    appearance
}

/// A shell's ordered ADVANCED_FACE references. Accepts OPEN_SHELL as well as
/// CLOSED_SHELL (the SHELL_BASED_SURFACE_MODEL lane can hold either), and
/// answers `None` — never an error — for anything it cannot read, because a
/// missing colour must not fail an import.
fn shell_face_refs(resolver: &Resolver, shell_ref: usize) -> Option<Vec<usize>> {
    let entity = resolver.get(shell_ref).ok()?;
    let list = entity
        .find("CLOSED_SHELL")
        .or_else(|| entity.find("OPEN_SHELL"))?
        .get(1)?
        .as_list()
        .ok()?;
    list.iter().map(|value| value.as_ref_id().ok()).collect()
}

pub(super) fn step_body_claimed_shells(resolver: &Resolver, body: StepBody) -> Option<Vec<usize>> {
    match body {
        StepBody::SolidBrep(entity_ref) => resolver
            .get(entity_ref)
            .ok()?
            .find("MANIFOLD_SOLID_BREP")
            .or_else(|| resolver.get(entity_ref).ok()?.find("FACETED_BREP"))
            .and_then(|args| args.get(1))
            .and_then(|value| value.as_ref_id().ok())
            .map(|shell_ref| vec![shell_ref]),
        StepBody::BrepWithVoids(entity_ref) => {
            let args = resolver.get(entity_ref).ok()?.find("BREP_WITH_VOIDS")?;
            let mut claims = vec![args.get(1)?.as_ref_id().ok()?];
            for oriented_value in args.get(2)?.as_list().ok()? {
                let oriented_ref = oriented_value.as_ref_id().ok()?;
                let oriented = resolver
                    .get(oriented_ref)
                    .ok()?
                    .find("ORIENTED_CLOSED_SHELL")?;
                claims.push(oriented.get(2)?.as_ref_id().ok()?);
            }
            Some(claims)
        }
        StepBody::SurfaceModelShell { shell_ref, .. } => Some(vec![shell_ref]),
    }
}

fn dedupe_step_bodies(resolver: &Resolver, bodies: &mut Vec<StepBody>) {
    bodies.sort_unstable_by_key(|body| body.order_key());
    let ordinary_claims: HashSet<usize> = bodies
        .iter()
        .filter(|body| matches!(body, StepBody::SolidBrep(_)))
        .filter_map(|body| step_body_claimed_shells(resolver, *body))
        .flatten()
        .collect();
    let mut seen = HashSet::default();
    bodies.retain(|body| {
        let Some(claims) = step_body_claimed_shells(resolver, *body) else {
            return true;
        };
        if matches!(body, StepBody::BrepWithVoids(_))
            && claims.iter().any(|shell| ordinary_claims.contains(shell))
        {
            return false;
        }
        if claims.iter().any(|shell| seen.contains(shell)) {
            return false;
        }
        seen.extend(claims);
        true
    });
}

/// What one STEP document imports as: the bodies, their appearance, and how
/// gracefully it went.
///
/// `appearances` is PARALLEL to `solids` — index `i` is body `i`'s colour — so
/// every arm that pushes a solid pushes its appearance in the same breath. A
/// file with no presentation entities fills it with defaults rather than leaving
/// it short, which keeps the "parallel" contract a `zip` can rely on.
pub(super) struct ImportedBodies {
    pub(super) solids: Vec<BrepSolid>,
    pub(super) appearances: Vec<BodyAppearance>,
    /// Parallel to `solids`: the STEP face entity of each built face, shell
    /// by shell (face `i` of body `b` IS `face_refs[b][i]`); empty for a body
    /// whose pairing could not be verified. The PMI reader's anchor map.
    pub(super) face_refs: Vec<Vec<usize>>,
    pub(super) failed: usize,
    pub(super) first_error: Option<String>,
}

/// [`shell_face_refs`] for the PMI reader (a sibling module).
pub(super) fn shell_face_refs_pub(resolver: &Resolver, shell_ref: usize) -> Option<Vec<usize>> {
    shell_face_refs(resolver, shell_ref)
}

/// Shared core. Either RESOLVES THE ASSEMBLY — positioning each solid-bearing
/// part occurrence by the world transform composed down the assembly tree, and
/// producing one distinct copy per instance of a re-used part — or, for a file
/// with no assembly structure, imports every solid untransformed (identity)
/// exactly as before. In both paths a body that fails to reconstruct is skipped
/// (counted in `failed`) rather than aborting the whole file.
pub(super) fn collect_step_solids(text: &str) -> Result<ImportedBodies, String> {
    if !text.contains("ISO-10303-21") {
        return Err("step_import: not an ISO-10303-21 Part 21 file".into());
    }
    let debug_timing = std::env::var("BREP_DEBUG_STEP_LOOP").is_ok();
    let parse_started = debug_timing.then(web_time::Instant::now);
    let entities = parse_data_section(text)?;
    if let Some(started) = parse_started {
        eprintln!("PHASE parse: {:?} ({} entities)", started.elapsed(), entities.len());
    }
    let resolver = Resolver {
        entities: &entities,
        length_scale: derive_length_scale_mm(&entities),
    };
    // The file's presentation entities, read ONCE. An uncoloured file leaves the
    // table empty and every appearance lookup short-circuits.
    let styles = StyleTable::read(&entities);

    // A file that carries an assembly graph (NEXT_ASSEMBLY_USAGE_OCCURRENCE) is
    // resolved into positioned leaf occurrences. If the graph resolves to no
    // positioned body (partial/unknown transform encoding) we fall through to
    // the flat import, so a malformed assembly still degrades to bodies-at-origin.
    let edges = assembly_edges(&entities);
    if !edges.is_empty() {
        let assembly_started = debug_timing.then(web_time::Instant::now);
        let occurrences = resolve_assembly(&entities, &resolver, &styles, &edges);
        if let Some(started) = assembly_started {
            eprintln!("PHASE assembly: {:?}", started.elapsed());
        }
        if !occurrences.solids.is_empty() {
            if std::env::var("BREP_DEBUG_STEP_ASM").is_ok() {
                eprintln!(
                    "[step-asm] {} positioned occurrence(s) from {} NAUO edge(s):",
                    occurrences.solids.len(),
                    edges.len()
                );
                for line in &occurrences.debug {
                    eprintln!("[step-asm]   {line}");
                }
            }
            return Ok(ImportedBodies {
                solids: occurrences.solids,
                appearances: occurrences.appearances,
                face_refs: occurrences.face_refs,
                failed: occurrences.failed,
                first_error: occurrences.first_error,
            });
        }
    }

    // Flat import: a single part, or several solids already sharing one frame. A
    // body is a MANIFOLD_SOLID_BREP (analytic/NURBS shell) or a FACETED_BREP
    // (planar polygon closed-shell); both wrap a CLOSED_SHELL as their shell arg.
    let mut bodies: Vec<StepBody> = Vec::new();
    for (id, entity) in entities.iter() {
        push_step_bodies(&resolver, *id, entity, &mut bodies);
    }
    dedupe_step_bodies(&resolver, &mut bodies);
    // Each body reconstructs an independent `BrepSolid` from the shared read-only
    // `Resolver` (its own `SolidBuilder`/id space, no cross-body state), so the
    // bodies fan out across a rayon pool under the `parallel` feature. `collect`
    // preserves body order, so the resulting solids, failure count, and
    // first_error are byte-identical to the sequential path (only faster).
    // Carry whether each body is an OPPORTUNISTIC surface-model shell: a
    // SHELL_BASED_SURFACE_MODEL closed shell that reconstructs becomes a solid,
    // but one that is an OPEN sheet is simply not a solid body — its failure must
    // not count toward `failed`/`first_error`, which callers read to judge whether
    // the genuine solid bodies imported cleanly.
    let loop_started = debug_timing.then(web_time::Instant::now);
    // The appearance is resolved HERE, beside the build, because it needs the
    // built solid (to length-check the per-face pairing) and the body's STEP
    // refs (to look the colours up) in the same place.
    let build = |body: StepBody| {
        (
            matches!(body, StepBody::SurfaceModelShell { .. }),
            build_step_body(&resolver, body).map(|solid| {
                let appearance = step_body_appearance(&resolver, &styles, body, &solid);
                let face_refs = super::pmi::body_face_refs(&resolver, body, &solid);
                (solid, appearance, face_refs)
            }),
        )
    };
    type BuiltBody = (bool, Result<(BrepSolid, BodyAppearance, Vec<usize>), String>);
    #[cfg(feature = "parallel")]
    let results: Vec<BuiltBody> = {
        use rayon::prelude::*;
        bodies.into_par_iter().map(build).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let results: Vec<BuiltBody> = bodies.into_iter().map(build).collect();
    if let Some(started) = loop_started {
        eprintln!("PHASE reconstruct: {:?} ({} bodies)", started.elapsed(), results.len());
    }
    let mut solids = Vec::new();
    let mut appearances = Vec::new();
    let mut face_refs = Vec::new();
    let mut failed = 0usize;
    let mut first_error = None;
    for (opportunistic, result) in results {
        match result {
            // An OPPORTUNISTIC (SHELL_BASED_SURFACE_MODEL) body that closes into a
            // SINGLE face is a full untrimmed primitive the author declared as a
            // surface model — reference/construction geometry, not a part (OCC
            // imports it as a non-solid sheet). Skip it so it isn't solidified into
            // a phantom body (e.g. 00012039's three full-torus sheets -> phantom
            // toroidal solids). Genuine multi-face sewn surface-model solids and all
            // MANIFOLD_SOLID_BREP parts (>= 2 faces) are unaffected.
            Ok((solid, _, _))
                if opportunistic
                    && solid.shells.iter().map(|s| s.faces.len()).sum::<usize>() <= 1 => {}
            // Solid, appearance and face refs are pushed together — the vectors
            // are parallel by construction, through every skip arm below.
            Ok((solid, appearance, refs)) => {
                solids.push(solid);
                appearances.push(appearance);
                face_refs.push(refs);
            }
            Err(_) if opportunistic => {} // open surface-model sheet: not a solid, skip
            Err(error) => {
                failed += 1;
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    Ok(ImportedBodies {
        solids,
        appearances,
        face_refs,
        failed,
        first_error,
    })
}

// ---------------------------------------------------------------------------
// Assembly resolution: place & instance multi-body parts (ISO 10303-214 §5)
// ---------------------------------------------------------------------------
//
// A STEP assembly wires PRODUCT_DEFINITIONs (parts / sub-assemblies) together
// with NEXT_ASSEMBLY_USAGE_OCCURRENCE (NAUO) edges. Each NAUO carries a rigid
// placement of the child's local frame into the parent's frame, encoded as:
//
//   NAUO ─ (PRODUCT_DEFINITION_SHAPE "NAUO PRDDFN") ← CONTEXT_DEPENDENT_SHAPE_
//   REPRESENTATION → ( REPRESENTATION_RELATIONSHIP(rep_child, rep_parent) +
//   REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION( ITEM_DEFINED_TRANSFORMATION
//   (item_1, item_2) ) ) → a pair of AXIS2_PLACEMENT_3D.
//
// The transform mapping child-rep coordinates into the parent rep is
// T = M(item_2) · M(item_1)⁻¹, where M(placement) is that frame's local→world
// matrix. Composing T along the path from the ROOT assembly down to a leaf part
// gives that occurrence's WORLD transform; a part reached along several paths
// (e.g. AS1's six bolts) yields several distinct positioned copies.

pub(super) use crate::step_matrix::{mat4_mul, Mat4};

pub(super) fn mat4_identity() -> Mat4 {
    crate::step_matrix::MAT4_IDENTITY
}

/// The local→world matrix of an orthonormal placement frame: rotation columns
/// are the frame axes, the translation column is the origin.
fn frame_matrix(frame: &Frame) -> Mat4 {
    [
        frame.x.x,
        frame.y.x,
        frame.z.x,
        frame.origin.x,
        frame.x.y,
        frame.y.y,
        frame.z.y,
        frame.origin.y,
        frame.x.z,
        frame.y.z,
        frame.z.z,
        frame.origin.z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// Inverse of an affine 4×4 (invert the 3×3 linear block, then the translation).
fn mat4_affine_inverse(m: &Mat4) -> Result<Mat4, String> {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() <= 1e-14 {
        return Err("step_import: singular assembly transform".into());
    }
    let inv_det = 1.0 / det;
    let r = [
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ];
    let (tx, ty, tz) = (m[3], m[7], m[11]);
    let it = [
        -(r[0] * tx + r[1] * ty + r[2] * tz),
        -(r[3] * tx + r[4] * ty + r[5] * tz),
        -(r[6] * tx + r[7] * ty + r[8] * tz),
    ];
    Ok([
        r[0], r[1], r[2], it[0], r[3], r[4], r[5], it[1], r[6], r[7], r[8], it[2], 0.0, 0.0, 0.0,
        1.0,
    ])
}

/// One parent→child edge of the assembly graph.
#[derive(Clone, Copy)]
pub(crate) struct AssemblyEdge {
    pub(crate) nauo_id: usize,
    pub(crate) parent_pd: usize,
    pub(crate) child_pd: usize,
}

/// The assembly graph edges (NEXT_ASSEMBLY_USAGE_OCCURRENCE), sorted by id for
/// determinism. Empty ⇒ the file has no assembly structure.
pub(crate) fn assembly_edges(entities: &HashMap<usize, Entity>) -> Vec<AssemblyEdge> {
    let mut edges: Vec<AssemblyEdge> = entities
        .iter()
        .filter_map(|(id, entity)| {
            let args = entity.find("NEXT_ASSEMBLY_USAGE_OCCURRENCE")?;
            let parent_pd = args.get(3)?.as_ref_id().ok()?;
            let child_pd = args.get(4)?.as_ref_id().ok()?;
            Some(AssemblyEdge {
                nauo_id: *id,
                parent_pd,
                child_pd,
            })
        })
        .collect();
    edges.sort_by_key(|edge| edge.nauo_id);
    edges
}

/// The arguments of a SHAPE_REPRESENTATION / ADVANCED_BREP_SHAPE_REPRESENTATION
/// (and the faceted/surface variants): `(name, items, context)`.
fn representation_args(entity: &Entity) -> Option<&[Value]> {
    entity
        .find("ADVANCED_BREP_SHAPE_REPRESENTATION")
        .or_else(|| entity.find("FACETED_BREP_SHAPE_REPRESENTATION"))
        .or_else(|| entity.find("MANIFOLD_SURFACE_SHAPE_REPRESENTATION"))
        .or_else(|| entity.find("GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION"))
        .or_else(|| entity.find("SHAPE_REPRESENTATION"))
}

/// The item list of a shape representation — the geometry it carries.
fn representation_items(entity: &Entity) -> Option<&[Value]> {
    representation_args(entity)?.get(1)?.as_list().ok()
}

/// The REPRESENTATION_CONTEXT a shape representation is expressed in — where
/// its unit assignment lives, hence the scale of every length it carries.
pub(super) fn representation_context(entity: &Entity) -> Option<usize> {
    representation_args(entity)?.get(2)?.as_ref_id().ok()
}

/// The root SHAPE_REPRESENTATION(s) that directly define a PRODUCT_DEFINITION's
/// shape (via PRODUCT_DEFINITION_SHAPE → SHAPE_DEFINITION_REPRESENTATION).
pub(super) fn pd_root_srs(entities: &HashMap<usize, Entity>, pd: usize) -> Vec<usize> {
    let pds_ids: Vec<usize> = entities
        .iter()
        .filter_map(|(id, entity)| {
            let args = entity.find("PRODUCT_DEFINITION_SHAPE")?;
            (args.get(2)?.as_ref_id().ok()? == pd).then_some(*id)
        })
        .collect();
    let mut srs: Vec<usize> = entities
        .values()
        .filter_map(|entity| {
            let args = entity.find("SHAPE_DEFINITION_REPRESENTATION")?;
            let definition = args.first()?.as_ref_id().ok()?;
            let used = args.get(1)?.as_ref_id().ok()?;
            pds_ids.contains(&definition).then_some(used)
        })
        .collect();
    srs.sort_unstable();
    srs.dedup();
    srs
}

/// SRs joined to `sr` by an IDENTITY SHAPE_REPRESENTATION_RELATIONSHIP (a
/// product's alternative representations of the same geometry). With-transform
/// relationships (assembly placements between different products) are excluded.
fn identity_related_srs(entities: &HashMap<usize, Entity>, sr: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for entity in entities.values() {
        if entity.has("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION") {
            continue;
        }
        let Some(args) = entity.find("SHAPE_REPRESENTATION_RELATIONSHIP") else {
            continue;
        };
        if args.len() < 4 {
            continue;
        }
        let (Ok(rep1), Ok(rep2)) = (args[2].as_ref_id(), args[3].as_ref_id()) else {
            continue;
        };
        if rep1 == sr {
            out.push(rep2);
        } else if rep2 == sr {
            out.push(rep1);
        }
    }
    out
}

/// Solid-brep entity ids belonging to a PRODUCT_DEFINITION (in the part's own
/// local frame), gathered from its shape representation and every identity-
/// related representation.
pub(super) fn pd_step_bodies(
    entities: &HashMap<usize, Entity>,
    resolver: &Resolver,
    pd: usize,
) -> Vec<StepBody> {
    let mut visited: HashSet<usize> = HashSet::default();
    let mut queue: VecDeque<usize> = pd_root_srs(entities, pd).into_iter().collect();
    let mut bodies = Vec::new();
    while let Some(sr) = queue.pop_front() {
        if !visited.insert(sr) {
            continue;
        }
        if let Ok(entity) = resolver.get(sr) {
            if let Some(items) = representation_items(entity) {
                for item in items {
                    if let Ok(item_id) = item.as_ref_id() {
                        if let Ok(item_entity) = resolver.get(item_id) {
                            push_step_bodies(resolver, item_id, item_entity, &mut bodies);
                        }
                    }
                }
            }
        }
        for other in identity_related_srs(entities, sr) {
            if !visited.contains(&other) {
                queue.push_back(other);
            }
        }
    }
    bodies.sort_unstable_by_key(|body| body.order_key());
    bodies.dedup();
    dedupe_step_bodies(resolver, &mut bodies);
    bodies
}

/// Resolve a NAUO's rigid placement (child-frame → parent-frame) from its
/// CONTEXT_DEPENDENT_SHAPE_REPRESENTATION. Returns identity when any link of the
/// transform chain is absent — a robust graceful fallback per the spec's option
/// that the transform entities may be partial.
pub(crate) fn edge_transform(
    entities: &HashMap<usize, Entity>,
    resolver: &Resolver,
    edge: &AssemblyEdge,
) -> Mat4 {
    let child_srs = pd_root_srs(entities, edge.child_pd);
    for entity in entities.values() {
        let Some(args) = entity.find("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") else {
            continue;
        };
        let Some(pds_id) = args.get(1).and_then(|value| value.as_ref_id().ok()) else {
            continue;
        };
        let Ok(pds) = resolver.get(pds_id) else {
            continue;
        };
        let owns_nauo = pds
            .find("PRODUCT_DEFINITION_SHAPE")
            .and_then(|pds_args| pds_args.get(2))
            .and_then(|value| value.as_ref_id().ok())
            == Some(edge.nauo_id);
        if !owns_nauo {
            continue;
        }
        let Some(rep_rel_id) = args.first().and_then(|value| value.as_ref_id().ok()) else {
            continue;
        };
        if let Some(transform) = resolve_rep_rel_transform(resolver, rep_rel_id, &child_srs) {
            return transform;
        }
    }
    mat4_identity()
}

/// Build the child→parent transform from a (possibly complex) representation
/// relationship carrying a REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION. The
/// stored transform maps rep_1 → rep_2; we orient the result to child → parent
/// using the child's known root SR, so the graph's rep ordering does not matter.
fn resolve_rep_rel_transform(
    resolver: &Resolver,
    rep_rel_id: usize,
    child_srs: &[usize],
) -> Option<Mat4> {
    let entity = resolver.get(rep_rel_id).ok()?;
    let idt_id = entity
        .find("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")?
        .first()?
        .as_ref_id()
        .ok()?;
    let rel = entity.find("REPRESENTATION_RELATIONSHIP")?;
    let rep1 = rel.get(2)?.as_ref_id().ok()?;
    let rep2 = rel.get(3)?.as_ref_id().ok()?;
    let idt = resolver.get(idt_id).ok()?;
    let idt_args = idt.find("ITEM_DEFINED_TRANSFORMATION")?;
    let item1 = idt_args.get(2)?.as_ref_id().ok()?;
    let item2 = idt_args.get(3)?.as_ref_id().ok()?;
    let m1 = frame_matrix(&resolver.placement(item1).ok()?);
    let m2 = frame_matrix(&resolver.placement(item2).ok()?);
    // T maps rep_1 coordinates into rep_2 coordinates.
    let forward = mat4_mul(&m2, &mat4_affine_inverse(&m1).ok()?);
    if child_srs.contains(&rep2) && !child_srs.contains(&rep1) {
        // rep_2 is the child ⇒ invert so the result maps child → parent.
        mat4_affine_inverse(&forward).ok()
    } else {
        Some(forward)
    }
}

pub(super) struct AssemblyResult {
    pub(super) solids: Vec<BrepSolid>,
    /// Parallel to `solids`: each occurrence's STEP face refs (the part's).
    pub(super) face_refs: Vec<Vec<usize>>,
    /// Parallel to `solids`: the styled colour of the PART each occurrence
    /// instantiates. An occurrence is a placed copy of one part body, so every
    /// instance of a re-used part carries that part's colour.
    pub(super) appearances: Vec<BodyAppearance>,
    failed: usize,
    first_error: Option<String>,
    debug: Vec<String>,
}

/// The assembly ROOTS: products that appear as an edge parent but never as an
/// edge child. Sorted and deduped, so the occurrence walk is deterministic.
pub(super) fn assembly_roots(edges: &[AssemblyEdge]) -> Vec<usize> {
    let children: HashSet<usize> = edges.iter().map(|edge| edge.child_pd).collect();
    let mut roots: Vec<usize> = edges
        .iter()
        .map(|edge| edge.parent_pd)
        .filter(|pd| !children.contains(pd))
        .collect();
    roots.sort_unstable();
    roots.dedup();
    roots
}

/// One visited node of the occurrence walk: a PRODUCT_DEFINITION reached along
/// ONE path down from a root, carrying the world transform composed along that
/// path. A product used N times is visited N times, once per path.
pub(super) struct OccurrenceNode {
    pub(super) pd: usize,
    pub(super) world: Mat4,
    /// `root/child/grandchild` product-name path — the occurrence's label.
    pub(super) name: String,
    /// Every product from the root to this node inclusive: the cycle guard.
    ancestors: Vec<usize>,
}

/// Depth-first walk of the occurrence tree. THE one implementation of tree
/// shape, visit order, transform composition and cycle guarding — shared by the
/// flat lane ([`resolve_assembly`], which emits one positioned solid per visit)
/// and the structured lane (`assembly::read_step_assembly`, which uses it to
/// decide which products the structure actually reaches), so the two can never
/// disagree about any of them.
///
/// `transforms` maps a NAUO id to that edge's child-frame -> parent-frame
/// placement; a missing entry composes as identity. Roots are pushed reversed,
/// and each node's child edges are pushed reversed, so the id-sorted edges pop
/// in ascending NAUO order.
pub(super) fn walk_occurrences(
    resolver: &Resolver,
    edges: &[AssemblyEdge],
    transforms: &HashMap<usize, Mat4>,
    mut visit: impl FnMut(&OccurrenceNode),
) {
    let mut stack: Vec<OccurrenceNode> = assembly_roots(edges)
        .iter()
        .rev()
        .map(|&pd| OccurrenceNode {
            pd,
            world: mat4_identity(),
            name: product_name(resolver, pd),
            ancestors: vec![pd],
        })
        .collect();

    while let Some(node) = stack.pop() {
        visit(&node);

        // Descend into children.
        let mut child_edges: Vec<&AssemblyEdge> = edges
            .iter()
            .filter(|edge| edge.parent_pd == node.pd)
            .collect();
        child_edges.sort_by_key(|edge| edge.nauo_id);
        for edge in child_edges.into_iter().rev() {
            if node.ancestors.contains(&edge.child_pd) {
                continue; // cycle guard
            }
            let transform = transforms
                .get(&edge.nauo_id)
                .copied()
                .unwrap_or_else(mat4_identity);
            let world = mat4_mul(&node.world, &transform);
            let mut ancestors = node.ancestors.clone();
            ancestors.push(edge.child_pd);
            stack.push(OccurrenceNode {
                pd: edge.child_pd,
                world,
                name: format!("{}/{}", node.name, product_name(resolver, edge.child_pd)),
                ancestors,
            });
        }
    }
}

/// Enumerate every leaf occurrence in the assembly and emit a positioned solid
/// (the part's BrepSolid with its composed world transform applied). A part used
/// N times produces N distinct positioned copies. Bodies that fail to build or
/// transform are skipped (graceful degradation), never aborting the file.
///
/// A pure CONSUMER of [`walk_occurrences`]: this owns only the "what to emit at
/// each visited node" half, so the structured lane shares its tree walk exactly.
pub(super) fn resolve_assembly(
    entities: &HashMap<usize, Entity>,
    resolver: &Resolver,
    styles: &StyleTable,
    edges: &[AssemblyEdge],
) -> AssemblyResult {
    // Precompute each edge's rigid placement (child-frame -> parent-frame).
    let mut transforms: HashMap<usize, Mat4> = HashMap::default();
    for edge in edges {
        transforms.insert(edge.nauo_id, edge_transform(entities, resolver, edge));
    }

    let mut result = AssemblyResult {
        solids: Vec::new(),
        face_refs: Vec::new(),
        appearances: Vec::new(),
        failed: 0,
        first_error: None,
        debug: Vec::new(),
    };
    // Build each distinct part solid once — with its appearance, which is a
    // property of the PART, not of the placement — then re-use both for every
    // occurrence.
    let mut solid_cache: HashMap<StepBody, Result<(BrepSolid, BodyAppearance, Vec<usize>), String>> =
        HashMap::default();
    let debug_on = std::env::var("BREP_DEBUG_STEP_ASM").is_ok();

    walk_occurrences(resolver, edges, &transforms, |node| {
        // Emit this product's own solids (leaf parts carry geometry; a pure
        // assembly node carries none, so this is a no-op there).
        for body in pd_step_bodies(entities, resolver, node.pd) {
            let base = solid_cache.entry(body).or_insert_with(|| {
                build_step_body(resolver, body)
                    .map(|solid| {
                        let appearance = step_body_appearance(resolver, styles, body, &solid);
                        let face_refs = super::pmi::body_face_refs(resolver, body, &solid);
                        (solid, appearance, face_refs)
                    })
                    .map_err(|error| {
                        format!("step_import: part body {body:?} failed to build: {error}")
                    })
            });
            let (base_solid, base_appearance, base_refs) = match base {
                Ok((solid, appearance, refs)) => (solid.clone(), appearance.clone(), refs.clone()),
                // An OPEN SHELL_BASED_SURFACE_MODEL sheet is not a solid body, so
                // (as in the flat path) its build failure is skipped rather than
                // counted against the genuine solid bodies.
                Err(_) if matches!(body, StepBody::SurfaceModelShell { .. }) => continue,
                Err(error) => {
                    result.failed += 1;
                    if result.first_error.is_none() {
                        result.first_error = Some(error.clone());
                    }
                    continue;
                }
            };
            match AffineTransform::new(node.world)
                .and_then(|transform| transform_brep(&base_solid, transform, false))
            {
                Ok(positioned) => {
                    if debug_on {
                        result.debug.push(format!(
                            "{} body{body:?} @ ({:.1}, {:.1}, {:.1})",
                            node.name, node.world[3], node.world[7], node.world[11]
                        ));
                    }
                    result.solids.push(positioned);
                    // `transform_brep` clones and mutates faces IN PLACE, so the
                    // positioned copy keeps the part's shell/face order and the
                    // part's per-face colour list still pairs index-for-index.
                    result.appearances.push(base_appearance);
                    result.face_refs.push(base_refs);
                }
                Err(error) => {
                    result.failed += 1;
                    if result.first_error.is_none() {
                        result.first_error = Some(error);
                    }
                }
            }
        }
    });

    result
}

/// The PRODUCT entity behind a PRODUCT_DEFINITION:
/// PRODUCT_DEFINITION → PRODUCT_DEFINITION_FORMATION[_WITH_SPECIFIED_SOURCE] →
/// PRODUCT. `None` when any link of the chain is absent.
fn product_entity<'a>(resolver: &'a Resolver, pd: usize) -> Option<&'a Entity> {
    let pd_entity = resolver.get(pd).ok()?;
    let formation_id = pd_entity
        .find("PRODUCT_DEFINITION")?
        .get(2)?
        .as_ref_id()
        .ok()?;
    let formation = resolver.get(formation_id).ok()?;
    let product_ref = formation
        .find("PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE")
        .or_else(|| formation.find("PRODUCT_DEFINITION_FORMATION"))?
        .get(2)?
        .as_ref_id()
        .ok()?;
    resolver.get(product_ref).ok()
}

/// One string argument of the PRODUCT record, or `None` when it is absent or
/// not a string. `PRODUCT(id, name, description, frame_of_reference)`.
fn product_field(resolver: &Resolver, pd: usize, index: usize) -> Option<String> {
    match product_entity(resolver, pd)?.find("PRODUCT")?.get(index)? {
        Value::Str(text) => Some(text.clone()),
        _ => None,
    }
}

/// A PRODUCT_DEFINITION's product NAME — instrumentation, occurrence labels,
/// and (through the structured lane) the imported part's library name. Falls
/// back to the entity id when the PRODUCT chain is absent.
pub(super) fn product_name(resolver: &Resolver, pd: usize) -> String {
    product_field(resolver, pd, 1).unwrap_or_else(|| format!("pd#{pd}"))
}

/// A PRODUCT_DEFINITION's product ID — the vendor part number, which the BOM
/// reports alongside the name. EMPTY when the file carries none: unlike the
/// name there is no sensible synthetic substitute, and a caller that wants one
/// falls back to [`product_name`].
pub(super) fn product_id(resolver: &Resolver, pd: usize) -> String {
    product_field(resolver, pd, 0).unwrap_or_default()
}

