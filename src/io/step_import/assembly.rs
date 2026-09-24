//! STRUCTURED STEP import: read the product structure and keep it.
//!
//! [`collect_step_solids`] resolves a STEP assembly graph and then throws the
//! structure away — it composes each occurrence's world transform, bakes it
//! into a copy of the part, and returns a flat `Vec<BrepSolid>`. That is the
//! right answer for "import these bodies", and it stays exactly as it is.
//!
//! This module is the other answer. It reads the SAME graph in the SAME single
//! parse and returns it: one [`StepProduct`] per PRODUCT_DEFINITION holding
//! that product's own bodies IN ITS OWN LOCAL FRAME, and one [`StepOccurrence`]
//! per NEXT_ASSEMBLY_USAGE_OCCURRENCE edge carrying the child→parent placement.
//! **Nothing here is transformed.** Composing the tree — and so choosing what a
//! "component" is — belongs to the caller, which maps products onto parts-library
//! entries and occurrences onto assembly-component instances.
//!
//! Two invariants make that safe:
//!
//! - **ONE parse, each body built ONCE.** The parse is shared with nothing (the
//!   caller passes the text once), and inside it every distinct [`StepBody`] is
//!   reconstructed a single time and cloned into its product, exactly as
//!   `resolve_assembly` caches per body.
//! - **The same walk.** Tree shape, visit order, transform composition and the
//!   cycle guard all come from [`walk_occurrences`], which the flat lane also
//!   uses, so the two lanes cannot drift apart. The kernel test suite asserts
//!   the strong form of this: replaying this module's output through the flat
//!   lane's own `transform_brep` call reproduces `resolve_assembly`'s solids
//!   BIT-for-bit on every single-context assembly fixture.
//!
//! **Units are resolved PER PRODUCT here** (`step-assembly-import.md` §3.6),
//! not once per file. `derive_length_scale_mm` picks one global scale out of
//! `HashMap` iteration order, which a mixed-unit assembly — each product
//! representation carrying its own `GLOBAL_UNIT_ASSIGNED_CONTEXT` — makes both
//! wrong and nondeterministic. See [`ProductScales`].

use super::*;

/// One PRODUCT_DEFINITION in the file's product structure.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepProduct {
    /// PRODUCT_DEFINITION entity id — the stable identity and dedup key.
    pub pd_ref: usize,
    /// `PRODUCT.name`, or `pd#<id>` when the file carries no PRODUCT chain.
    pub name: String,
    /// `PRODUCT.id` — the vendor part number, for the BOM. May be empty.
    pub id: String,
    /// This product's OWN solids, in its OWN local frame and at its OWN unit
    /// scale. Empty ⇒ a pure assembly node (structure, no geometry).
    pub bodies: Vec<BrepSolid>,
    /// The styled colour of each entry of `bodies`, PARALLEL to it (index `i`
    /// is body `i`). Default-filled when the file carries no presentation
    /// entities, so it is always the same length as `bodies` and a caller can
    /// `zip` without a length check.
    pub appearances: Vec<BodyAppearance>,
    /// Bodies of this product that did not reconstruct. Graceful degradation:
    /// counted, never fatal, and never a reason to reject the file.
    pub failed_bodies: usize,
}

/// One NAUO occurrence: a placement of `child` inside `parent`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepOccurrence {
    /// NEXT_ASSEMBLY_USAGE_OCCURRENCE entity id.
    pub nauo_ref: usize,
    /// Index into [`StepAssembly::products`].
    pub parent: usize,
    /// Index into [`StepAssembly::products`].
    pub child: usize,
    /// The instance label ("bolt-3") — see [`nauo_designator`] for which of the
    /// NAUO's several label attributes wins.
    pub designator: String,
    /// child-local → parent-local, row-major affine, translations in MILLIMETRES
    /// (scaled by the PARENT representation's unit, which is the frame the
    /// placement is expressed in). NOT a world transform: the caller composes
    /// these down from [`StepAssembly::roots`].
    pub placement: [f64; 16],
    /// RᵀR = I and det R = +1 within 1e-9 — i.e. the placement is a pure
    /// rotation + translation, representable as a translate/rotate component
    /// pose. A mirrored or scaled occurrence has no such representation and its
    /// non-rigid factor must be baked into a distinct part by the caller.
    pub rigid: bool,
}

/// A STEP file's product structure, with geometry, and without a single
/// transform applied to it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepAssembly {
    /// Every product the structure mentions, ordered by ascending `pd_ref`.
    pub products: Vec<StepProduct>,
    /// One entry per NAUO edge, ordered by ascending `nauo_ref`.
    pub occurrences: Vec<StepOccurrence>,
    /// Indices of the products that are never an occurrence child — where a
    /// consumer starts composing world transforms.
    pub roots: Vec<usize>,
    /// The first body-reconstruction error, if any body failed.
    pub first_error: Option<String>,
}

/// Read a STEP file's product structure AND build each product's own bodies, in
/// ONE parse, with each distinct body built ONCE.
///
/// `Ok(None)` when the file carries no assembly structure worth using — no NAUO
/// edges at all, or none that the walk reaches a built body through. That is
/// the same condition on which [`collect_step_solids`] abandons the assembly
/// branch, so a caller can treat `None` as "fall back to the flat lane" and get
/// exactly today's behaviour.
///
/// `Err` only for a file that is not Part 21 or does not parse — a broken
/// *assembly* degrades, it does not fail.
pub fn read_step_assembly(text: &str) -> Result<Option<StepAssembly>, String> {
    if !text.contains("ISO-10303-21") {
        return Err("step_import: not an ISO-10303-21 Part 21 file".into());
    }
    let entities = parse_data_section(text)?;
    let edges = assembly_edges(&entities);
    if edges.is_empty() {
        return Ok(None); // no product structure: the flat lane is the answer
    }

    let global_scale = derive_length_scale_mm(&entities);
    let mut scales = ProductScales::new(&entities, global_scale);

    // Every product the structure mentions, by ascending entity id.
    let mut pd_refs: Vec<usize> = edges
        .iter()
        .flat_map(|edge| [edge.parent_pd, edge.child_pd])
        .collect();
    pd_refs.sort_unstable();
    pd_refs.dedup();
    let product_index: HashMap<usize, usize> = pd_refs
        .iter()
        .enumerate()
        .map(|(index, &pd)| (pd, index))
        .collect();

    // Each edge's placement, read through a Resolver at the PARENT product's
    // scale: an ITEM_DEFINED_TRANSFORMATION's AXIS2_PLACEMENT_3D origins are
    // lengths in the parent representation's context (§3.6.2), so the parent's
    // unit — not the file-global one — converts them to millimetres. The
    // rotation block is dimensionless and unaffected.
    //
    // (Strictly, `item_1` lives in the CHILD representation, so a file that
    // BOTH mixes units AND gives item_1 a non-identity origin would want the
    // two frames scaled separately. Every producer we have seen writes item_1
    // as the child rep's origin placement, where the two agree; splitting the
    // scales would mean reimplementing `edge_transform` rather than handing it
    // a Resolver, which is a worse trade for a case no fixture exhibits.)
    let mut transforms: HashMap<usize, Mat4> = HashMap::default();
    for edge in &edges {
        let parent_resolver = Resolver {
            entities: &entities,
            length_scale: scales.get(edge.parent_pd),
        };
        transforms.insert(
            edge.nauo_id,
            edge_transform(&entities, &parent_resolver, edge),
        );
    }

    // Each product's own bodies, built at that product's own unit scale. The
    // cache is keyed by (body, scale) rather than by body alone: two products
    // can only share a body entity by sharing a representation, in which case
    // they share a context and the scale key is redundant — but if they ever do
    // not, the second product gets its own build instead of the first's.
    // The file's presentation entities, read ONCE for the whole document.
    let styles = StyleTable::read(&entities);
    let mut solid_cache: HashMap<(StepBody, u64), Result<(BrepSolid, BodyAppearance), String>> =
        HashMap::default();
    let mut first_error: Option<String> = None;
    let mut products: Vec<StepProduct> = Vec::with_capacity(pd_refs.len());
    for &pd in &pd_refs {
        let scale = scales.get(pd);
        let resolver = Resolver {
            entities: &entities,
            length_scale: scale,
        };
        let scale_key = scale.to_bits();
        let mut bodies = Vec::new();
        let mut appearances = Vec::new();
        let mut failed_bodies = 0usize;
        for body in pd_step_bodies(&entities, &resolver, pd) {
            let built = solid_cache.entry((body, scale_key)).or_insert_with(|| {
                build_step_body(&resolver, body)
                    .map(|solid| {
                        let appearance = step_body_appearance(&resolver, &styles, body, &solid);
                        (solid, appearance)
                    })
                    .map_err(|error| {
                        format!("step_import: part body {body:?} failed to build: {error}")
                    })
            });
            match built {
                // Pushed together, so `bodies` and `appearances` stay parallel
                // through every skip arm below.
                Ok((solid, appearance)) => {
                    bodies.push(solid.clone());
                    appearances.push(appearance.clone());
                }
                // An OPEN SHELL_BASED_SURFACE_MODEL sheet is not a solid body:
                // skipped, not counted, exactly as both existing lanes do.
                Err(_) if matches!(body, StepBody::SurfaceModelShell { .. }) => {}
                Err(error) => {
                    failed_bodies += 1;
                    if first_error.is_none() {
                        first_error = Some(error.clone());
                    }
                }
            }
        }
        products.push(StepProduct {
            pd_ref: pd,
            name: product_name(&resolver, pd),
            id: product_id(&resolver, pd),
            bodies,
            appearances,
            failed_bodies,
        });
    }

    // Which products the structure actually REACHES — through the flat lane's
    // own walk, so the cycle guard and root set are identical. A graph that
    // reaches no geometry is not a structure we can import, so the caller falls
    // back to the flat lane, exactly as `collect_step_solids` does when
    // `resolve_assembly` yields no solid.
    //
    // (One corner differs harmlessly: a body that builds but fails
    // `transform_brep` yields no flat solid, yet counts as reached here.)
    let mut reached: HashSet<usize> = HashSet::default();
    {
        let resolver = Resolver {
            entities: &entities,
            length_scale: global_scale,
        };
        walk_occurrences(&resolver, &edges, &transforms, |node| {
            reached.insert(node.pd);
        });
    }
    let reaches_geometry = products
        .iter()
        .any(|product| !product.bodies.is_empty() && reached.contains(&product.pd_ref));
    if !reaches_geometry {
        return Ok(None);
    }

    let label_resolver = Resolver {
        entities: &entities,
        length_scale: global_scale,
    };
    let occurrences = edges
        .iter()
        .map(|edge| {
            let placement = transforms
                .get(&edge.nauo_id)
                .copied()
                .unwrap_or_else(mat4_identity);
            StepOccurrence {
                nauo_ref: edge.nauo_id,
                parent: product_index[&edge.parent_pd],
                child: product_index[&edge.child_pd],
                designator: nauo_designator(&label_resolver, edge.nauo_id),
                placement,
                rigid: transform_is_rigid(&placement),
            }
        })
        .collect();

    let roots = assembly_roots(&edges)
        .into_iter()
        .map(|pd| product_index[&pd])
        .collect();

    Ok(Some(StepAssembly {
        products,
        occurrences,
        roots,
        first_error,
    }))
}

/// Per-product length units, memoised (`step-assembly-import.md` §3.6).
///
/// A product's geometry is expressed in the REPRESENTATION_CONTEXT of the shape
/// representation that defines it, and that context carries its own unit
/// assignment. `Resolver` is only `{entities, length_scale}`, so building a
/// product at its own scale costs one struct rebuild — which is the whole
/// implementation of "per-product units".
///
/// Falls back to the file-global scale for a product with no resolvable context
/// (a pure assembly node with no shape representation, typically), so a
/// single-context file resolves to the identical scale everywhere and the
/// structured lane's geometry stays byte-identical to the flat lane's.
pub(super) struct ProductScales<'a> {
    entities: &'a HashMap<usize, Entity>,
    global: f64,
    cache: HashMap<usize, f64>,
}

impl<'a> ProductScales<'a> {
    pub(super) fn new(entities: &'a HashMap<usize, Entity>, global: f64) -> Self {
        Self {
            entities,
            global,
            cache: HashMap::default(),
        }
    }

    /// Millimetres per native length unit for `pd`'s own representation. The
    /// candidate SRs come from `pd_root_srs`, which sorts and dedups, so a
    /// product with several representations resolves deterministically (unlike
    /// `derive_length_scale_mm`, which reads whatever the `HashMap` yields).
    pub(super) fn get(&mut self, pd: usize) -> f64 {
        if let Some(&scale) = self.cache.get(&pd) {
            return scale;
        }
        let scale = pd_root_srs(self.entities, pd)
            .into_iter()
            .filter_map(|sr| self.entities.get(&sr).and_then(representation_context))
            .find_map(|context| context_length_scale_mm(self.entities, context))
            .unwrap_or(self.global);
        self.cache.insert(pd, scale);
        scale
    }
}

/// Is this affine's linear block a pure rotation — RᵀR = I and det R = +1 to
/// 1e-9? A mirrored (det = −1) or scaled occurrence is NOT representable as a
/// component pose, and its caller must bake the non-rigid factor into a
/// distinct part rather than silently reuse the unmirrored twin.
///
/// Nothing the importer reads TODAY can fail this: an occurrence placement is
/// built from two AXIS2_PLACEMENT_3Ds, and `Resolver::placement` always returns
/// an orthonormal right-handed frame. It becomes load-bearing the moment
/// CARTESIAN_TRANSFORMATION_OPERATOR_3D placements are read (§3.7), which is
/// why the flag is computed and carried now rather than assumed away.
pub(super) fn transform_is_rigid(matrix: &Mat4) -> bool {
    const TOLERANCE: f64 = 1e-9;
    let column = |index: usize| [matrix[index], matrix[4 + index], matrix[8 + index]];
    let (x, y, z) = (column(0), column(1), column(2));
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (a, b, expected) in [
        (x, x, 1.0),
        (y, y, 1.0),
        (z, z, 1.0),
        (x, y, 0.0),
        (x, z, 0.0),
        (y, z, 0.0),
    ] {
        if (dot(a, b) - expected).abs() > TOLERANCE {
            return false;
        }
    }
    // det R, as the scalar triple product of the columns.
    let cross = [
        y[1] * z[2] - y[2] * z[1],
        y[2] * z[0] - y[0] * z[2],
        y[0] * z[1] - y[1] * z[0],
    ];
    (dot(x, cross) - 1.0).abs() <= TOLERANCE
}

/// The instance label of a NAUO —
/// `NEXT_ASSEMBLY_USAGE_OCCURRENCE(id, name, description, relating, related,
/// reference_designator)`.
///
/// Producers disagree about which attribute carries it, so take the first
/// NON-BLANK of `reference_designator` → `name` → `description` → `id`. The
/// three assembly fixtures each land on a different one: AssemblyExample and
/// io1-ac put it in `name`; as1-ug writes `name` as a single space and the real
/// designator ('ROD', 'NUT1', …) in `description`; a producer that fills
/// `reference_designator` is the standard-conforming case and wins outright.
fn nauo_designator(resolver: &Resolver, nauo: usize) -> String {
    let field = |args: &[Value], index: usize| match args.get(index) {
        Some(Value::Str(text)) if !text.trim().is_empty() => Some(text.trim().to_string()),
        _ => None,
    };
    (|| {
        let args = resolver
            .get(nauo)
            .ok()?
            .find("NEXT_ASSEMBLY_USAGE_OCCURRENCE")?;
        field(args, 5)
            .or_else(|| field(args, 1))
            .or_else(|| field(args, 2))
            .or_else(|| field(args, 0))
    })()
    .unwrap_or_else(|| format!("nauo#{nauo}"))
}
