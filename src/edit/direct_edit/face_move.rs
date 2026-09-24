use super::*;

// ---------------------------------------------------------------------------
// §6.12 sibling operation: move a face group.
// ---------------------------------------------------------------------------

/// How the moved group uses an edge: not at all, on both sides (carried
/// rigidly), or on exactly one side (the seam to re-intersect).
enum EdgeMoveClass {
    Fixed,
    Interior,
    Boundary { moved_face: u64, fixed_face: u64 },
}

enum EdgeMoveAction {
    /// Translate the curve rigidly; parameters and pcurves stay exact.
    Translate,
    /// Replace the curve by the straight line between re-solved endpoints.
    Rebuild { start: Vec3, end: Vec3 },
    /// Map the curve by an exact affine (SM1b: the radial-scale-about-axis of a
    /// rim re-intersecting a ruled carrier under an axis-parallel cap push).
    /// Rational-quadratic circles map exactly, so parametrisation is preserved.
    Transform(AffineTransform),
    /// Replace the curve outright by an exactly RE-BUILT conic arc — the section
    /// `plane ∩ ruled carrier` between two re-solved endpoints
    /// (`conic_arc_on_ruled`). A re-trim cannot serve here: the boolean that
    /// created these edges SPLIT them at the corner, so `domain == [t0, t1]` and
    /// there is no parameter headroom for an outward push (verified on the
    /// flatted-frustum fixture — the flat×cone hyperbolas and the cap's conic
    /// rims alike). Used for a CONE's oblique multi-rim cap, where the rim's
    /// homothety about the apex carries the whole conic exactly but slides the
    /// ARC's endpoints off the fixed wall the corners must stay on.
    Replace {
        curve: NurbsCurve,
    },
}

enum FaceMoveAction {
    /// Shift the whole control net; pcurves stay exact for any carrier type.
    TranslateSurface,
    /// Rebuild the (planar) carrier around the new boundary and recompute
    /// every pcurve — the direct-edit equivalent of "extend the neighbour".
    Retrim(Plane),
}

/// `plane_of_surface` with per-face memoisation, because a face is consulted
/// once per boundary edge, once per touched vertex, and once at re-trim time.
fn cached_plane(
    cache: &mut HashMap<u64, Plane>,
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
    tolerance: f64,
) -> Result<Plane, String> {
    if let Some(plane) = cache.get(&face_id) {
        return Ok(*plane);
    }
    let (shell_index, face_index) = *face_lookup
        .get(&face_id)
        .ok_or_else(|| format!("move_faces: missing face {face_id}"))?;
    let plane = plane_of_surface(
        &solid.shells[shell_index].faces[face_index].surface,
        tolerance,
        "move_faces",
    )?;
    cache.insert(face_id, plane);
    Ok(plane)
}

/// The carrier kind of a face, in the words a user would use — for refusals
/// that must say WHAT the offending face is.
fn carrier_kind_name(surface: &NurbsSurface) -> &'static str {
    match surface.analytic() {
        Some(AnalyticSurface::Plane { .. }) => "plane",
        Some(AnalyticSurface::RuledRevolution { rho0, rho1, .. }) => {
            let scale = rho0.abs().max(rho1.abs()).max(1.0);
            if (rho1 - rho0).abs() <= 1e-9 * scale {
                "cylinder"
            } else {
                "cone"
            }
        }
        Some(AnalyticSurface::Sphere { .. }) => "sphere",
        Some(AnalyticSurface::Torus { .. }) => "torus",
        Some(AnalyticSurface::Revolution { .. }) => "general surface of revolution",
        None => "free-form surface",
    }
}

/// The refusal a face whose carrier this operation cannot use deserves: one
/// that names the FACE, its CARRIER KIND and the ROLE it plays in the push.
///
/// Every call site below reaches `cached_plane` on a face it has already
/// decided is not a supported carrier — so the string a user got was
/// `plane_of_surface`'s own `move_faces: face is not planar (curved neighbours
/// are deferred in this slice)` (`offset/retrim.rs:152`): a message from a
/// shared helper that three features call, naming neither the face nor the
/// neighbour, and carrying slice-scoped wording out of a general predicate. A
/// torus groove, a barrel boss and a NURBS dimple all produced the same
/// sentence, and the earlier hand-read census of these refusals could not tell
/// from the text which of the *nine* such call sites had fired: it attributed
/// the torus groove's refusal to the wrong one.
///
/// This changes no verdict anywhere: it is applied as `map_err`, so a face
/// `plane_of_surface` accepts is still accepted, on exactly the same inputs.
fn unsupported_carrier(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
    role: &str,
) -> String {
    let kind = face_lookup
        .get(&face_id)
        .map(|&(shell, face)| carrier_kind_name(&solid.shells[shell].faces[face].surface))
        .unwrap_or("missing");
    format!(
        "move_faces: {role} (face {face_id}) is a {kind}; the plane push re-intersects only \
         planar and ruled (cylinder / cone) carriers here — refusing"
    )
}

/// Intersect three-or-more planes in one point: take the best-conditioned
/// pair for the line, then the plane most transverse to that line for the
/// point. The caller checks the residual against EVERY plane afterwards, so
/// this only has to find *a* candidate, not prove consistency.
fn solve_corner(planes: &[Plane]) -> Option<Vec3> {
    let mut best_pair: Option<(usize, usize, f64)> = None;
    for first in 0..planes.len() {
        for second in first + 1..planes.len() {
            let spread = planes[first].normal.cross(planes[second].normal).length();
            if best_pair.map(|(_, _, best)| spread > best).unwrap_or(true) {
                best_pair = Some((first, second, spread));
            }
        }
    }
    let (first, second, spread) = best_pair?;
    if spread <= PARALLEL_EPS {
        return None;
    }
    let line = intersect_planes(&planes[first], &planes[second])?;
    let mut best_third: Option<(usize, f64)> = None;
    for third in 0..planes.len() {
        if third == first || third == second {
            continue;
        }
        let transversality = line.dir.dot(planes[third].normal).abs();
        if best_third
            .map(|(_, best)| transversality > best)
            .unwrap_or(true)
        {
            best_third = Some((third, transversality));
        }
    }
    let (third, transversality) = best_third?;
    if transversality <= PARALLEL_EPS {
        return None;
    }
    intersect_line_plane(&line, &planes[third])
}

/// Rebuild a straight edge between two re-solved endpoints. Refuses the
/// degenerate and inverted cases — a zero or reversed chord means the
/// translation drove a moved face onto or past the neighbour this edge
/// belongs to (e.g. pushing a box face through its opposite face).
fn plan_straight_rebuild(
    edge: &EdgeRecord,
    start_old: Vec3,
    end_old: Vec3,
    start_new: Vec3,
    end_new: Vec3,
    tolerance: f64,
) -> Result<EdgeMoveAction, String> {
    if edge.degenerate {
        return Err(format!(
            "move_faces: degenerate edge {} would need re-stretching (deferred)",
            edge.id
        ));
    }
    if edge.curve.degree != 1 || edge.curve.control_points.len() != 2 {
        return Err(format!(
            "move_faces: edge {} must be re-stretched but is not a straight line \
             (curved re-intersection edges are deferred in this slice)",
            edge.id
        ));
    }
    let new_chord = end_new.sub(start_new);
    if new_chord.length() <= tolerance {
        return Err(format!(
            "move_faces: the translation collapses edge {} to zero length (a moved \
             face lands exactly on its neighbour) — refusing",
            edge.id
        ));
    }
    if end_old.sub(start_old).dot(new_chord) <= 0.0 {
        return Err(format!(
            "move_faces: the translation inverts edge {} (a moved face passes beyond \
             its neighbour) — refusing",
            edge.id
        ));
    }
    Ok(EdgeMoveAction::Rebuild {
        start: start_new,
        end: end_new,
    })
}

/// The (linearly-varying) radius of a ruled revolution at axial coordinate
/// `axial`: `rho0` at the base, `rho1` at `height`. Constant for a cylinder.
fn rho_at(rho0: f64, rho1: f64, height: f64, axial: f64) -> f64 {
    if height == 0.0 {
        rho0
    } else {
        rho0 + (rho1 - rho0) * axial / height
    }
}

/// Frame/radii/height of any revolution with a STRAIGHT (degree-1, unit-weight)
/// generatrix — the full-2π `RuledRevolution` quadrics AND partial-sweep
/// `Revolution`s whose generatrix is a line. The latter is how a fillet band /
/// partial cylinder-or-cone wall recognizes (a straight profile swept less than
/// 2π is the general `Revolution`, not `RuledRevolution`), so treating it as a
/// ruled carrier is exactly what lets a face adjacent to a fillet band be pushed.
///
/// This mirrors `analytic_surface::intersect::ruled_revolution_data`, which
/// already extracts the same `(frame, rho0, rho1, height)` from a partial-sweep
/// `Revolution`; that function is private behind a private module (unreachable
/// from here without editing `analytic_surface.rs`), so the tiny extraction is
/// re-derived from the (public) frame basis instead of shared. A CURVED
/// generatrix (sphere/torus-like, non-ruled) returns None → it must still refuse.
fn ruled_revolution_carrier(
    surface: &NurbsSurface,
) -> Option<(crate::RevolutionFrame, f64, f64, f64)> {
    match surface.analytic() {
        Some(AnalyticSurface::RuledRevolution {
            frame,
            rho0,
            rho1,
            height,
        }) => Some((frame.clone(), *rho0, *rho1, *height)),
        Some(AnalyticSurface::Revolution {
            frame, generatrix, ..
        }) => {
            // Unit-weight straight line only; a rational or higher-degree
            // generatrix is a genuinely curved surface of revolution.
            const UNIT_WEIGHT_TOL: f64 = 1e-9;
            let controls = &generatrix.control_points;
            if generatrix.degree != 1
                || controls.len() != 2
                || (controls[0].w - 1.0).abs() > UNIT_WEIGHT_TOL
                || (controls[1].w - 1.0).abs() > UNIT_WEIGHT_TOL
            {
                return None;
            }
            // Cylindrical decomposition (radius, axial) from the public frame
            // basis — `RevolutionFrame::cylindrical` is module-private.
            let decompose = |point: Vec3| -> (f64, f64) {
                let d = point.sub(frame.origin);
                let axial = d.dot(frame.axis);
                let radial = d.sub(frame.axis.scale(axial)).length();
                (radial, axial)
            };
            let (rho0, z0) = decompose(controls[0].point().ok()?);
            let (rho1, z1) = decompose(controls[1].point().ok()?);
            let height = z1 - z0;
            if height.abs() <= 1e-12 * (1.0 + rho0.abs().max(rho1.abs())) {
                return None;
            }
            // Rebase the origin to the generatrix start's axial position so
            // v = axial / height, exactly as the `RuledRevolution` variant does.
            let origin = frame.origin.add(frame.axis.scale(z0));
            Some((
                crate::RevolutionFrame {
                    origin,
                    ..frame.clone()
                },
                rho0,
                rho1,
                height,
            ))
        }
        _ => None,
    }
}

/// The ruled-revolution carrier (cylinder OR cone, full-2π OR partial-sweep
/// fillet band) of a face resolved by id, for ANY push direction (SM1/SM1b
/// axis-parallel, SM1c oblique). A planar cap re-intersecting such a carrier
/// under a push keeps the SAME surface and only re-trims; the rim maps by the
/// exact affine `rim_ruled_map` (a homothety about the cone apex, or an axis
/// translation for a cylinder), so unlike the earlier `axis_parallel_ruled`
/// predicate this no longer requires the push to be parallel to the axis.
fn carrier_ruled(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
) -> Option<(crate::RevolutionFrame, f64, f64, f64)> {
    let &(shell, face) = face_lookup.get(&face_id)?;
    ruled_revolution_carrier(&solid.shells[shell].faces[face].surface)
}

/// The SPHERE carrier (frame + radius) of a face resolved by id, or `None` when
/// it is not an analytic sphere. A planar push that borders a FIXED sphere
/// re-intersects it in an EXACT circle (`intersect_plane_quadric` = plane ×
/// sphere) and re-trims the sphere to that new rim — see
/// `move_planar_face_across_sphere` (backlog #5, Plane × Sphere).
fn carrier_sphere(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
) -> Option<(crate::RevolutionFrame, f64)> {
    let &(shell, face) = face_lookup.get(&face_id)?;
    match solid.shells[shell].faces[face].surface.analytic() {
        Some(AnalyticSurface::Sphere { frame, radius }) => Some((frame.clone(), *radius)),
        _ => None,
    }
}

/// True iff the face's carrier is a plane the push is PARALLEL to — the corner
/// slides within it, carried rigidly by `+translation` (SM1's invariant-plane
/// case, e.g. a box side wall as the top is pushed up).
fn plane_parallel_to(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
    translation: Vec3,
    parallel_tol: f64,
) -> bool {
    let Some(&(shell, face)) = face_lookup.get(&face_id) else {
        return false;
    };
    match solid.shells[shell].faces[face].surface.analytic() {
        Some(AnalyticSurface::Plane { u_dir, v_dir, .. }) => {
            let normal = u_dir.cross(*v_dir);
            let len = normal.length();
            len > 0.0 && translation.dot(normal).abs() / len <= parallel_tol
        }
        _ => false,
    }
}

/// True iff a FIXED carrier is INVARIANT under `translation`, so a corner on it
/// rides rigidly by `+translation` and provably stays on it. Two carriers are:
///   • a plane the push is PARALLEL to (the corner slides in-plane), and
///   • a CYLINDER whose axis the push is PARALLEL to (the point shifts along a
///     generatrix, radius unchanged).
/// A CONE is deliberately excluded: its radius varies with the axial coordinate,
/// so an axis translation moves a point OFF the carrier — a corner there must
/// re-intersect, not ride. This lets a corner shared by a fillet band (an
/// axis-parallel partial cylinder) and a parallel wall ride rigidly under an
/// axis-parallel push, while any oblique push falls through to the (planar-only)
/// re-intersection path and refuses.
fn carrier_invariant_under(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
    translation: Vec3,
    parallel_tol: f64,
) -> bool {
    if plane_parallel_to(solid, face_lookup, face_id, translation, parallel_tol) {
        return true;
    }
    if let Some((frame, rho0, rho1, _height)) = carrier_ruled(solid, face_lookup, face_id) {
        let radius_scale = rho0.abs().max(rho1.abs()).max(1.0);
        let is_cylinder = (rho1 - rho0).abs() <= 1e-9 * radius_scale;
        if is_cylinder {
            // Invariant iff the translation is parallel to the axis, i.e. its
            // component perpendicular to the axis is negligible.
            let axis = frame.axis;
            let perpendicular = translation.sub(axis.scale(translation.dot(axis)));
            return perpendicular.length() <= parallel_tol;
        }
    }
    false
}

/// The EXACT affine that re-intersects a planar cap's rim with a ruled
/// revolution under a push of ANY direction — the SM1c generalisation of SM1b's
/// axis-parallel radial-scale map:
///
/// - **Cone/frustum:** a homothety (central dilation) about the apex with ratio
///   `λ = 1 + (n·T)/D₀`, where `n` is the cap-plane unit normal, `T` the
///   translation, and `D₀ = n·(cap_origin − apex)` the signed apex→plane
///   distance along `n`. The dilation maps the cone to itself and the cap plane
///   to the TRANSLATED cap plane, so it maps the old rim exactly to the new one
///   — for any push direction (oblique sections are hyperbola/ellipse arcs, and
///   an affine maps rational curves control-point-wise, so parametrisation is
///   preserved). For an axis-parallel ⊥-cap this reduces to the radial scale
///   `s = rho_at(z+d)/rho_at(z)`.
/// - **Cylinder (`rho0 == rho1`):** the carrier is invariant under axis
///   translation, so the cap plane's shift maps to a pure axis shift `t·axis`
///   with `t = (n·T)/(n·axis)` (SM1's `n = axis` case gives `t = d`).
///
/// Refuses a cap plane through the apex (`D₀ ≈ 0`), a push to/through the apex
/// (`λ ≤ tol`), and a cap plane parallel to a cylinder axis (`n·axis ≈ 0`, a
/// straight generatrix section handled by the straight-rebuild path).
fn rim_ruled_map(
    frame: &crate::RevolutionFrame,
    rho0: f64,
    rho1: f64,
    height: f64,
    moved_plane: &Plane,
    translation: Vec3,
    tolerance: f64,
) -> Result<AffineTransform, String> {
    let n = moved_plane.normal;
    let axis = frame.axis;
    let radius_scale = rho0.abs().max(rho1.abs()).max(1.0);
    // Cylinder: axis-invariant carrier ⇒ the cap shift is a pure axis translation.
    if (rho1 - rho0).abs() <= 1e-9 * radius_scale {
        let axial_component = n.dot(axis);
        if axial_component.abs() <= 1e-9 {
            return Err(
                "move_faces: the cap plane is parallel to the cylinder axis \
                 (straight generatrix section) — refusing"
                    .into(),
            );
        }
        let shift = axis.scale(n.dot(translation) / axial_component);
        return AffineTransform::new([
            1.0, 0.0, 0.0, shift.x, //
            0.0, 1.0, 0.0, shift.y, //
            0.0, 0.0, 1.0, shift.z, //
            0.0, 0.0, 0.0, 1.0,
        ]);
    }
    // Cone/frustum: homothety about the apex (where rho_at → 0).
    let z_apex = rho0 * height / (rho0 - rho1);
    let apex = frame.origin.add(axis.scale(z_apex));
    let d0 = n.dot(moved_plane.origin.sub(apex));
    if d0.abs() <= tolerance {
        return Err("move_faces: the cap plane passes through the cone apex — refusing".into());
    }
    let lambda = 1.0 + n.dot(translation) / d0;
    if lambda <= tolerance {
        return Err(
            "move_faces: the push drives the cap to or past the cone apex \
             (scale → 0) — refusing"
                .into(),
        );
    }
    // Y = apex + λ·(X − apex) = λ·X + (1−λ)·apex.
    let offset = apex.scale(1.0 - lambda);
    AffineTransform::new([
        lambda, 0.0, 0.0, offset.x, //
        0.0, lambda, 0.0, offset.y, //
        0.0, 0.0, lambda, offset.z, //
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// The checked contract for SM1b (the user's "reapply the trimming" semantics):
/// the constructed rim must lie ON both modified carriers — the fixed ruled
/// carrier (radius `rho_at`) and the translated moved plane. Samples the mapped
/// rim; refuses if any sample drifts off either surface. The map is exact by
/// construction, so this is a fail-safe guard, not the primary computation.
fn verify_rim_on_carriers(
    edge: &EdgeRecord,
    map: &AffineTransform,
    frame: &crate::RevolutionFrame,
    rho0: f64,
    rho1: f64,
    height: f64,
    moved_plane: &Plane,
    translation: Vec3,
    tolerance: f64,
) -> Result<(), String> {
    let (origin, axis) = (frame.origin, frame.axis);
    let normal = moved_plane.normal;
    let plane_point = moved_plane.origin.add(translation);
    for step in 0..=8 {
        let t = edge.t0 + (edge.t1 - edge.t0) * (step as f64 / 8.0);
        let mapped = map.point(edge.curve.evaluate(t)?);
        let delta = mapped.sub(origin);
        let axial = delta.dot(axis);
        let radial = delta.sub(axis.scale(axial)).length();
        let off_ruled = (radial - rho_at(rho0, rho1, height, axial)).abs();
        let off_plane = mapped.sub(plane_point).dot(normal).abs();
        if off_ruled > 10.0 * tolerance || off_plane > 10.0 * tolerance {
            return Err(format!(
                "move_faces: the re-intersected rim does not lie on both modified carriers \
                 (off ruled {off_ruled:.3e}, off plane {off_plane:.3e}) — refusing"
            ));
        }
    }
    Ok(())
}

/// True iff the face's carrier is a plane (the today path — planar retrim).
fn carrier_is_planar(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
) -> bool {
    face_lookup
        .get(&face_id)
        .map(|&(shell, face)| {
            matches!(
                solid.shells[shell].faces[face].surface.analytic(),
                Some(AnalyticSurface::Plane { .. })
            )
        })
        .unwrap_or(false)
}

/// A rebuilt straight edge bordering a curved invariant carrier must still lie
/// ON that carrier (endpoints + midpoint within `10·tol`), else the moved group
/// tore off it — refuse rather than emit a bad solid.
fn verify_chord_on_carrier(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    face_id: u64,
    start: Vec3,
    end: Vec3,
    tolerance: f64,
) -> Result<(), String> {
    let (shell, face) = *face_lookup
        .get(&face_id)
        .ok_or_else(|| format!("move_faces: missing face {face_id}"))?;
    // Measure against the ANALYTIC (unbounded) carrier, not the trimmed NURBS
    // surface: the rebuilt chord routinely lands OUTSIDE the current v-domain
    // (the carrier is grown to cover it afterwards), so a domain-clamped
    // projection would report a false miss. For a cylinder "on the carrier" is
    // just "radial distance from the axis == radius". Accepts a partial-sweep
    // fillet band (`Revolution`) the same way as a full `RuledRevolution`.
    let Some((frame, rho0, rho1, height)) =
        ruled_revolution_carrier(&solid.shells[shell].faces[face].surface)
    else {
        return Err(format!(
            "move_faces: chord-on-carrier check expects a ruled carrier (face {face_id})"
        ));
    };
    let (origin, axis) = (frame.origin, frame.axis);
    let midpoint = start.add(end).scale(0.5);
    for point in [start, midpoint, end] {
        let delta = point.sub(origin);
        let axial = delta.dot(axis);
        let radial = delta.sub(axis.scale(axial)).length();
        // The generatrix radius VARIES along the axis on a cone, so compare
        // against rho_at(axial), not a fixed rho0 (which is cylinder-only).
        let radius = rho_at(rho0, rho1, height, axial);
        if (radial - radius).abs() > 10.0 * tolerance {
            return Err(format!(
                "move_faces: rebuilt edge would leave its curved neighbour \
                 (face {face_id}, off by {:.3e}) — refusing",
                (radial - radius).abs()
            ));
        }
    }
    Ok(())
}

/// Re-trim a FIXED ruled neighbour whose boundary moved under an axis-parallel
/// push: grow the carrier along its axis to cover the new boundary
/// (`extend_ruled_neighbour_over` for a full-2π cylinder/cone,
/// `extend_revolution_carrier_over` for a partial-sweep fillet band — both
/// exact), then recompute every pcurve on the grown carrier from the
/// already-updated edge curves. The direct-edit analogue of `retrim_planar_face`
/// for a translation-invariant cylinder; all loops are visited, so holes on the
/// neighbour are carried.
///
/// The three-phase body is `crate::offset_retrim::retrim_face_in_solid`; this is
/// the two-step growth strategy and the `move_faces` refusal prefix. It differs
/// from `face_offset::retrim_offset_ruled_face` in exactly that second growth
/// call — the partial-sweep `Revolution` prolongation, which the offset push has
/// never run.
fn retrim_ruled_face(
    solid: &mut BrepSolid,
    face_id: u64,
    final_edges: &HashMap<u64, EdgeRecord>,
    tolerance: f64,
) -> Result<(), String> {
    let (shell, face_pos) = find_face(solid, face_id)
        .ok_or_else(|| format!("move_faces: missing ruled face {face_id}"))?;
    retrim_face_in_solid(
        solid,
        shell,
        face_pos,
        final_edges,
        |solid, points| {
            extend_ruled_neighbour_over(solid, face_id, points, tolerance)?;
            extend_revolution_carrier_over(solid, face_id, points, tolerance)
        },
        PcurveFit::SubrangeAware { tolerance },
        "move_faces",
    )
}

/// The EXACT corner where a fixed PLANE, a fixed RULED carrier and the
/// TRANSLATED cap plane meet: the line `fixed plane ∩ translated cap plane`
/// intersected with the ruled carrier (a quadratic in the line parameter),
/// taking the root nearest the corner's OLD position so the corner moves
/// continuously.
///
/// This is the carrier-level solve that a CURVED fixed edge needs.
/// `resolve_corner_on_fixed_edge` brackets strictly INSIDE the fixed edge's
/// domain, and the boolean that produced such an edge split it exactly at the
/// corner (`domain == [t0, t1]`), so riding the edge can only ever move a corner
/// INWARD — an outward push would refuse for a purely representational reason.
/// The carriers have no such horizon.
fn corner_on_plane_and_ruled(
    fixed_plane: &Plane,
    cap_normal: Vec3,
    cap_c: f64,
    frame: &crate::RevolutionFrame,
    rho0: f64,
    rho1: f64,
    height: f64,
    old_corner: Vec3,
    tolerance: f64,
) -> Result<Vec3, String> {
    let direction = fixed_plane.normal.cross(cap_normal);
    if direction.length() <= PARALLEL_EPS {
        return Err(
            "move_faces: the pushed cap plane is parallel to a fixed planar neighbour \
             (no corner) — refusing"
                .into(),
        );
    }
    let direction = direction.normalized()?;
    // A point on both planes, taken in the 2-D span of the two normals.
    let ca = fixed_plane.normal.dot(fixed_plane.origin);
    let naa = fixed_plane.normal.dot(fixed_plane.normal);
    let nab = fixed_plane.normal.dot(cap_normal);
    let nbb = cap_normal.dot(cap_normal);
    let determinant = naa * nbb - nab * nab;
    if determinant.abs() <= PARALLEL_EPS {
        return Err("move_faces: cannot place the corner's carrier line — refusing".into());
    }
    let alpha = (ca * nbb - cap_c * nab) / determinant;
    let beta = (cap_c * naa - ca * nab) / determinant;
    let base = fixed_plane
        .normal
        .scale(alpha)
        .add(cap_normal.scale(beta));
    // |P − O|² − axial² = rho_at(axial)² along P(s) = base + s·direction.
    let offset = base.sub(frame.origin);
    let axis = frame.axis;
    let slope = if height == 0.0 {
        0.0
    } else {
        (rho1 - rho0) / height
    };
    let a0 = offset.dot(axis);
    let a1 = direction.dot(axis);
    let r0 = rho0 + slope * a0;
    let quad = 1.0 - a1 * a1 - slope * slope * a1 * a1;
    let linear = 2.0 * offset.dot(direction) - 2.0 * a0 * a1 - 2.0 * slope * a1 * r0;
    let constant = offset.dot(offset) - a0 * a0 - r0 * r0;
    let mut roots: Vec<f64> = Vec::new();
    if quad.abs() <= 1e-12 {
        if linear.abs() > 1e-12 {
            roots.push(-constant / linear);
        }
    } else {
        let discriminant = linear * linear - 4.0 * quad * constant;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            roots.push((-linear + root) / (2.0 * quad));
            roots.push((-linear - root) / (2.0 * quad));
        }
    }
    let mut best: Option<(Vec3, f64)> = None;
    for s in roots {
        let point = base.add(direction.scale(s));
        // Only the nappe with a NON-NEGATIVE radius is the real carrier.
        if rho_at(rho0, rho1, height, point.sub(frame.origin).dot(axis)) < -tolerance {
            continue;
        }
        let distance = point.sub(old_corner).length();
        if best.map(|(_, best)| distance < best).unwrap_or(true) {
            best = Some((point, distance));
        }
    }
    let (corner, _) = best.ok_or_else(|| {
        "move_faces: the pushed cap plane no longer meets the fixed ruled neighbour \
         (the push drives the corner off the carrier) — refusing"
            .to_string()
    })?;
    Ok(corner)
}

/// True iff `curve[t0..t1]` sweeps in the POSITIVE azimuth sense about `frame`
/// (the sense `make_arc` builds), decided by whether its midpoint's azimuth lies
/// inside the positive sweep from the start's to the end's.
fn arc_sweeps_forward(
    frame: &crate::RevolutionFrame,
    curve: &NurbsCurve,
    t0: f64,
    t1: f64,
) -> Result<bool, String> {
    let azimuth = |point: Vec3| {
        let delta = point.sub(frame.origin);
        delta.dot(frame.y_axis).atan2(delta.dot(frame.x_axis))
    };
    let tau = std::f64::consts::TAU;
    let wrap = |angle: f64| {
        let value = angle % tau;
        if value < 0.0 {
            value + tau
        } else {
            value
        }
    };
    let start = azimuth(curve.evaluate(t0)?);
    let end = azimuth(curve.evaluate(t1)?);
    let middle = azimuth(curve.evaluate(0.5 * (t0 + t1))?);
    Ok(wrap(middle - start) <= wrap(end - start))
}

/// The EXACT arc of `plane ∩ cone` between two endpoints that already lie on
/// both, swept in the given sense.
///
/// A cone's plane section is the PROJECTIVE image, from the apex, of the base
/// circle: the ray `apex → X` meets the plane at `apex + (k/((X−apex)·n))·(X−apex)`
/// with `k = c − n·apex`, which is LINEAR in the homogeneous control point — so
/// the image of a rational-quadratic circular arc is a rational-quadratic conic
/// arc on the SAME knot vector, exactly. Azimuth is constant along a cone ray,
/// so the endpoints' azimuths give the base arc directly. This is exact for
/// ELLIPTIC and HYPERBOLIC sections alike (the full hyperbola cannot be one
/// rational Bezier — its weights change sign — but an arc that stays on one
/// nappe can, which is why the sign check below is the only restriction).
///
/// The kernel's own `intersect_plane_quadric` builds a cone section the same way
/// but only ever for the FULL section, so it refuses exactly the hyperbolic case
/// this arc-restricted form supports.
fn conic_arc_on_ruled(
    frame: &crate::RevolutionFrame,
    rho0: f64,
    rho1: f64,
    height: f64,
    plane_normal: Vec3,
    plane_c: f64,
    start: Vec3,
    end: Vec3,
    forward_sweep: bool,
    tolerance: f64,
) -> Result<NurbsCurve, String> {
    let radius_scale = rho0.abs().max(rho1.abs()).max(1.0);
    if (rho1 - rho0).abs() <= 1e-9 * radius_scale {
        // A cylinder's plane section is an ellipse (or a generatrix pair); its
        // straight sections are already handled by the chord rebuild and no
        // fixture exercises a curved one, so it stays an honest refusal.
        return Err(
            "move_faces: rebuilding a curved section on a CYLINDER carrier is deferred \
             — refusing"
                .into(),
        );
    }
    let axis = frame.axis;
    let apex = frame
        .origin
        .add(axis.scale(rho0 * height / (rho0 - rho1)));
    let k = plane_c - plane_normal.dot(apex);
    if k.abs() <= tolerance {
        return Err("move_faces: the section plane passes through the cone apex — refusing".into());
    }
    // Base circle at whichever end has the LARGER radius, so it never degenerates.
    let (reference_rho, reference_axial) = if rho0.abs() >= rho1.abs() {
        (rho0.abs(), 0.0)
    } else {
        (rho1.abs(), height)
    };
    if reference_rho <= tolerance {
        return Err("move_faces: the cone carrier degenerates to its apex — refusing".into());
    }
    let azimuth = |point: Vec3| {
        let delta = point.sub(frame.origin);
        delta.dot(frame.y_axis).atan2(delta.dot(frame.x_axis))
    };
    let tau = std::f64::consts::TAU;
    let wrap = |angle: f64| {
        let value = angle % tau;
        if value < 0.0 {
            value + tau
        } else {
            value
        }
    };
    let (start_angle, sweep, reverse) = if forward_sweep {
        (azimuth(start), wrap(azimuth(end) - azimuth(start)), false)
    } else {
        (azimuth(end), wrap(azimuth(start) - azimuth(end)), true)
    };
    if sweep <= 1e-9 || sweep >= tau - 1e-9 {
        return Err(
            "move_faces: the re-intersected arc degenerates to a point or a full turn \
             — refusing"
                .into(),
        );
    }
    let circle = crate::make_arc(
        frame.origin.add(axis.scale(reference_axial)),
        frame.x_axis,
        frame.y_axis,
        reference_rho,
        start_angle,
        start_angle + sweep,
    )?;
    let mut mapped: Vec<crate::Vec4> = Vec::with_capacity(circle.control_points.len());
    let mut sign = 0.0f64;
    for control in &circle.control_points {
        let relative = Vec3::new(
            control.x - control.w * apex.x,
            control.y - control.w * apex.y,
            control.z - control.w * apex.z,
        );
        let weight = relative.dot(plane_normal);
        if weight.abs() <= 1e-9 * radius_scale {
            return Err(
                "move_faces: the re-intersected section runs along an asymptotic ruling \
                 of the cone — refusing"
                    .into(),
            );
        }
        if sign == 0.0 {
            sign = weight.signum();
        } else if weight.signum() != sign {
            return Err(
                "move_faces: the re-intersected section crosses the cone's apex plane \
                 (both nappes) — refusing"
                    .into(),
            );
        }
        let scaled = relative.scale(k);
        mapped.push(crate::Vec4 {
            x: apex.x * weight + scaled.x,
            y: apex.y * weight + scaled.y,
            z: apex.z * weight + scaled.z,
            w: weight,
        });
    }
    if sign < 0.0 {
        // Homogeneously identical, but the kernel keeps weights positive.
        for control in &mut mapped {
            control.x = -control.x;
            control.y = -control.y;
            control.z = -control.z;
            control.w = -control.w;
        }
    }
    let mut curve = NurbsCurve::new(circle.degree, circle.knots.clone(), mapped)?;
    if reverse {
        curve = curve.reversed()?;
    }
    // Fail-safe: the rebuilt arc must run between the given corners and lie on
    // BOTH carriers — the same "reapply the trimming" contract the affine rim map
    // is held to in `verify_rim_on_carriers`.
    let [d0, d1] = curve.domain()?;
    for (parameter, target) in [(d0, start), (d1, end)] {
        let drift = curve.evaluate(parameter)?.sub(target).length();
        if drift > 10.0 * tolerance {
            return Err(format!(
                "move_faces: the rebuilt section misses its corner by {drift:.3e} — refusing"
            ));
        }
    }
    for step in 0..=8 {
        let point = curve.evaluate(d0 + (d1 - d0) * (step as f64 / 8.0))?;
        let delta = point.sub(frame.origin);
        let axial = delta.dot(axis);
        let off_ruled = (delta.sub(axis.scale(axial)).length()
            - rho_at(rho0, rho1, height, axial))
        .abs();
        let off_plane = (point.dot(plane_normal) - plane_c).abs();
        if off_ruled > 10.0 * tolerance || off_plane > 10.0 * tolerance {
            return Err(format!(
                "move_faces: the rebuilt section does not lie on both carriers \
                 (off ruled {off_ruled:.3e}, off plane {off_plane:.3e}) — refusing"
            ));
        }
    }
    Ok(curve)
}

/// The fixed PLANE and the fixed RULED carrier an edge is shared by, when it is
/// shared by exactly one of each (the flat×cone hyperbola configuration).
fn plane_and_ruled_carriers(
    solid: &BrepSolid,
    face_lookup: &HashMap<u64, (usize, usize)>,
    planes: &mut HashMap<u64, Plane>,
    face_ids: &[u64],
    plane_tolerance: f64,
) -> Option<(Plane, (crate::RevolutionFrame, f64, f64, f64))> {
    let mut plane = None;
    let mut ruled = None;
    for &face_id in face_ids {
        if let Some(carrier) = carrier_ruled(solid, face_lookup, face_id) {
            if ruled.is_some() {
                return None;
            }
            ruled = Some(carrier);
        } else if carrier_is_planar(solid, face_lookup, face_id) {
            if plane.is_some() {
                return None;
            }
            plane = Some(cached_plane(planes, solid, face_lookup, face_id, plane_tolerance).ok()?);
        } else {
            return None;
        }
    }
    Some((plane?, ruled?))
}

/// Re-solve a cap corner that is shared with a FIXED ruled carrier (a split
/// cylinder/cone band), which the planar 3-plane `solve_corner` cannot place
/// (its carrier is not a plane). The corner rides the ONE fixed edge it shares
/// with the body: the new corner is where the TRANSLATED cap plane crosses that
/// fixed edge's curve. The fixed edge itself stays put — only its cap-side trim
/// endpoint slides along it (`plan_straight_rebuild` re-lays a straight
/// generatrix between the new and the untouched endpoints afterwards).
///
/// This is the SM1c multi-rim generalisation: the single-closed-rim oblique cap
/// (no corners) is the `fixed_at.len() == 1` rim-ride; a cap bounded by several
/// conic rims meeting at seam corners needs each corner re-placed here.
///
/// - **Straight generatrix (degree 1):** exact line solve. A line is its own
///   natural extension, so a crossing OUTSIDE the fixed edge's current span
///   (the outward push that lengthens the wall) is exact and accepted.
/// - **Conic (degree 2+):** bracket a sign change WITHIN the domain only and
///   refine — a rational conic extended past its span rides the end tangent,
///   not the conic, so an out-of-domain crossing is refused. The root nearest
///   the corner's own end parameter is chosen so the corner moves continuously.
///   This is now a FALLBACK: booleans split such an edge exactly at the corner
///   (`domain == [t0, t1]`), so an outward push has no room to bracket into.
///   When the corner's fixed carriers are one plane and one ruled surface, the
///   caller solves on the CARRIERS instead (`corner_on_plane_and_ruled`), which
///   has no such horizon; this arm only runs for configurations that solve does
///   not cover.
///
/// Refuses cleanly when the fixed edge runs in the cap plane (no crossing) or
/// the push drives the corner off the edge's reachable span (a tearing push).
fn resolve_corner_on_fixed_edge(
    fixed_edge: &EdgeRecord,
    seed_param: f64,
    plane_normal: Vec3,
    plane_c: f64,
    tolerance: f64,
) -> Result<Vec3, String> {
    let curve = &fixed_edge.curve;
    if curve.degree == 1 && curve.control_points.len() == 2 {
        let p0 = curve.control_points[0].point()?;
        let p1 = curve.control_points[1].point()?;
        let dir = p1.sub(p0);
        let denom = plane_normal.dot(dir);
        if denom.abs() <= PARALLEL_EPS * (1.0 + dir.length()) {
            return Err(format!(
                "move_faces: fixed edge {} runs parallel to the cap plane (no crossing) — refusing",
                fixed_edge.id
            ));
        }
        let s = (plane_c - plane_normal.dot(p0)) / denom;
        return Ok(p0.add(dir.scale(s)));
    }
    // Curved (conic) fixed edge — closed-form-in-spirit numeric solve strictly
    // within the domain. `n·C(u) − c` has the sign of the numerator polynomial
    // (weights are strictly positive), so its roots are the crossings.
    let [d0, d1] = curve.domain()?;
    let f = |u: f64| -> Result<f64, String> { Ok(plane_normal.dot(curve.evaluate(u)?) - plane_c) };
    const STEPS: usize = 96;
    let mut best: Option<(f64, f64)> = None;
    let mut prev_u = d0;
    let mut prev_f = f(d0)?;
    if prev_f.abs() <= tolerance {
        best = Some((d0, (d0 - seed_param).abs()));
    }
    for i in 1..=STEPS {
        let u = d0 + (d1 - d0) * (i as f64 / STEPS as f64);
        let fu = f(u)?;
        if prev_f * fu < 0.0 {
            let (mut lo, mut hi, mut flo) = (prev_u, u, prev_f);
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                let fm = f(mid)?;
                if flo * fm <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            let root = 0.5 * (lo + hi);
            let dist = (root - seed_param).abs();
            if best.map(|(_, bd)| dist < bd).unwrap_or(true) {
                best = Some((root, dist));
            }
        }
        prev_u = u;
        prev_f = fu;
    }
    let (root, _) = best.ok_or_else(|| {
        format!(
            "move_faces: the pushed cap does not re-cross fixed edge {} within its span \
             (curved-fixed-edge extension deferred, or the push tears the face) — refusing",
            fixed_edge.id
        )
    })?;
    curve.evaluate(root)
}

/// Golovanov §6.12 direct editing — translate a group of faces rigidly and
/// heal the adjacency with the faces that stay behind.
///
/// The moved carriers translate exactly (every control point shifts by the
/// translation, which is exact for ANY surface type), and each boundary edge
/// between a moved face and a fixed face is recomputed as the intersection of
/// the translated moved carrier with the fixed carrier:
///
/// - When the translation is parallel to every fixed plane a boundary vertex
///   touches, the whole neighbourhood translates rigidly — exact for any
///   moved carrier and any edge curve type. This is the extrude-like case:
///   pushing a face along its own normal slides the side walls in-plane.
/// - Otherwise the new corner is re-solved as the common point of ALL carrier
///   planes meeting at the vertex (moved ones translated), and every affected
///   straight edge is rebuilt between the re-solved corners — the same
///   relocate-onto-recovered-corners move `delete_face_and_heal` performs on
///   its side edges.
///
/// Scope (honest refusals, never a bad solid): the moved faces may be any
/// surface type. A FIXED face that must be re-intersected — the fixed side of a
/// boundary edge, or any face whose boundary edges must be rebuilt — must be
/// PLANAR, OR an AXIS-PARALLEL ruled revolution: a cylinder OR a cone whose axis
/// the push is parallel to (SM1/SM1b). Such a carrier keeps its SAME surface and
/// only re-trims — its rim re-intersects it at a new radius, constructed by the
/// exact radial-scale map `EdgeMoveAction::Transform` (`s = rho_at(z+d)/rho_at(z)`;
/// the cylinder is `s = 1`) and verified to lie on both modified carriers, then
/// grown along the axis (`retrim_ruled_face`). Every straight rebuilt edge (a
/// seam/generatrix) is verified on its carrier at its own `rho_at` radius.
///
/// An OBLIQUE cap push is supported on both carriers. The rim is the affine
/// image `rim_ruled_map` (a cylinder's axis translation, a cone's homothety
/// about the apex) — exact for the whole conic — and where that affine's image
/// of a corner disagrees with the corner itself (a CONE, whose homothety slides
/// the arc's endpoint off the fixed flat), the rim and the CURVED fixed edge it
/// meets are RE-BUILT as exact conic sections between the re-solved corners
/// (`conic_arc_on_ruled`, `EdgeMoveAction::Replace`); the corner itself comes
/// from the carrier-level solve `corner_on_plane_and_ruled`. A curved fixed edge
/// on a CYLINDER carrier, spheres, tori, general revolutions and a cap pushed
/// to/through the apex are refused (SM3 is the general offset path).
/// A translation that collapses an adjacent edge to zero length or reverses
/// its direction (moving a box face onto or past its opposite face) is
/// refused, as is a group that tears away from its neighbours. The input is
/// never mutated; the result is returned only when `validate()` is clean.
pub fn move_faces(
    solid: &BrepSolid,
    face_ids: &[u64],
    translation: Vec3,
) -> Result<BrepSolid, String> {
    if !(translation.x.is_finite() && translation.y.is_finite() && translation.z.is_finite()) {
        return Err("move_faces: translation must be finite".into());
    }
    if face_ids.is_empty() {
        return Err("move_faces: no faces selected".into());
    }
    let moved: HashSet<u64> = face_ids.iter().copied().collect();
    // face id -> (shell, face) built once. move_faces never mutates `solid`,
    // so this replaces the O(faces) `find_face` scans in the validation loop
    // below and in `cached_plane` (called up to once per unique fixed face).
    // `or_insert` keeps the first match, mirroring `find_face`.
    let mut face_lookup: HashMap<u64, (usize, usize)> = HashMap::default();
    for (shell_index, shell) in solid.shells.iter().enumerate() {
        for (face_index, face) in shell.faces.iter().enumerate() {
            face_lookup
                .entry(face.id)
                .or_insert((shell_index, face_index));
        }
    }
    for &face_id in face_ids {
        if !face_lookup.contains_key(&face_id) {
            return Err(format!("move_faces: no face with id {face_id}"));
        }
    }

    let scale = solid_model_scale(solid);
    let tolerance = (scale * 1e-7).max(1e-9);
    let plane_tolerance = (scale * 1e-6).max(1e-7);
    // "Parallel to a fixed plane" means the translation's normal component
    // could not move any point off that plane at model precision.
    let parallel_tolerance = (translation.length() * 1e-9).max(1e-12);
    // "Rigid" endpoints moved by exactly the translation (they are assigned
    // `point + translation` verbatim, so this only absorbs rounding noise).
    let rigid_tolerance = (scale * 1e-9).max(1e-12);

    // --- Classify every edge by how the group uses it ----------------------
    let mut faces_of_edge: HashMap<u64, Vec<u64>> = HashMap::default();
    for shell in &solid.shells {
        for face in &shell.faces {
            for loop_record in &face.loops {
                for coedge in &loop_record.coedges {
                    faces_of_edge
                        .entry(coedge.edge_id)
                        .or_default()
                        .push(face.id);
                }
            }
        }
    }
    // --- Plane × Sphere fast path (backlog #5) -----------------------------
    // A planar push whose FIXED neighbour across a boundary edge is a SPHERE
    // cannot be healed by the planar/ruled machinery below: the sphere's seam
    // meridian is a CURVED fixed edge (which `plan_straight_rebuild` refuses),
    // and its periodic u=0/u=2π seam pcurves must be patched in parameter space,
    // not refit from scratch. Route those to a dedicated handler that
    // re-intersects the translated plane with the fixed sphere (an EXACT circle)
    // and re-trims the sphere. Every configuration that handler does not support
    // refuses cleanly there — and every such case refuses in the generic path
    // today too, so this routing can only turn a refusal into a heal (it never
    // changes an already-supported case).
    let borders_a_sphere = solid.edges.iter().any(|edge| {
        let uses = faces_of_edge
            .get(&edge.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let moved_uses = uses.iter().filter(|f| moved.contains(*f)).count();
        moved_uses > 0
            && moved_uses < uses.len()
            && uses
                .iter()
                .any(|f| !moved.contains(f) && carrier_sphere(solid, &face_lookup, *f).is_some())
    });
    if borders_a_sphere {
        return move_planar_face_across_sphere(solid, &moved, &face_lookup, &faces_of_edge, translation);
    }

    let mut classes: HashMap<u64, EdgeMoveClass> = HashMap::default();
    let mut planes: HashMap<u64, Plane> = HashMap::default();
    for edge in &solid.edges {
        let uses = faces_of_edge
            .get(&edge.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let expected = if edge.degenerate { 1 } else { 2 };
        if uses.len() != expected {
            return Err(format!(
                "move_faces: edge {} is used {} times (non-manifold input)",
                edge.id,
                uses.len()
            ));
        }
        let moved_uses = uses
            .iter()
            .filter(|face_id| moved.contains(*face_id))
            .count();
        let class = if moved_uses == 0 {
            EdgeMoveClass::Fixed
        } else if moved_uses == uses.len() {
            EdgeMoveClass::Interior
        } else {
            let moved_face = *uses
                .iter()
                .find(|face_id| moved.contains(*face_id))
                .unwrap();
            let fixed_face = *uses
                .iter()
                .find(|face_id| !moved.contains(*face_id))
                .unwrap();
            // The face left behind across a boundary edge is the carrier we
            // re-intersect against. Planar → cache its plane (today path).
            // Non-planar but a ruled revolution (a cylinder OR a cone) → allowed:
            // the cap rim re-intersects the SAME carrier, re-trimmed as a ruled
            // carrier via the exact `rim_ruled_map` for ANY push direction
            // (SM1/SM1b axis-parallel, SM1c oblique). Any other non-planar
            // carrier still refuses.
            if carrier_is_planar(solid, &face_lookup, fixed_face) {
                cached_plane(&mut planes, solid, &face_lookup, fixed_face, plane_tolerance)?;
            } else if carrier_ruled(solid, &face_lookup, fixed_face).is_none() {
                // A geometrically-planar face the recognizer did not TAG as a
                // plane (an imported B-spline patch, say) still passes here —
                // only its message changes. A genuinely curved carrier refuses,
                // and this is the arm it refuses at: the classification loop,
                // long before the face-action loop the census cites.
                cached_plane(&mut planes, solid, &face_lookup, fixed_face, plane_tolerance)
                    .map_err(|_| {
                        unsupported_carrier(
                            solid,
                            &face_lookup,
                            fixed_face,
                            &format!(
                                "the fixed neighbour across boundary edge {}",
                                edge.id
                            ),
                        )
                    })?;
            }
            EdgeMoveClass::Boundary {
                moved_face,
                fixed_face,
            }
        };
        classes.insert(edge.id, class);
    }

    // --- Relocate every vertex the group touches ---------------------------
    let mut vertex_faces: HashMap<u64, HashSet<u64>> = HashMap::default();
    for edge in &solid.edges {
        if let Some(uses) = faces_of_edge.get(&edge.id) {
            for vertex_id in [edge.start_vertex_id, edge.end_vertex_id] {
                vertex_faces
                    .entry(vertex_id)
                    .or_default()
                    .extend(uses.iter().copied());
            }
        }
    }
    let mut new_vertex: HashMap<u64, Vec3> = HashMap::default();
    for vertex in &solid.vertices {
        let Some(adjacent) = vertex_faces.get(&vertex.id) else {
            continue;
        };
        if !adjacent.iter().any(|face_id| moved.contains(face_id)) {
            continue;
        }
        let fixed_at: Vec<u64> = adjacent
            .iter()
            .copied()
            .filter(|face_id| !moved.contains(face_id))
            .collect();
        if fixed_at.is_empty() {
            // Interior vertex: carried rigidly with the group.
            new_vertex.insert(vertex.id, vertex.point.add(translation));
            continue;
        }
        // SM1b/SM1c: a corner on a single ruled neighbour rides that rim's exact
        // affine map — it stays on the (unchanged) cylinder/cone at the re-
        // intersected position. The map is the cap's homothety about the cone
        // apex (or an axis translation for a cylinder) and no longer requires the
        // push to be axis-parallel, so an OBLIQUE cap push relocates the seam
        // vertex onto the new conic rim exactly.
        if fixed_at.len() == 1 {
            if let Some((frame, rho0, rho1, height)) =
                carrier_ruled(solid, &face_lookup, fixed_at[0])
            {
                // The cap sharing this ruled rim is the moved planar face at the
                // vertex; its plane defines the homothety (D₀ from the apex).
                if let Some(cap) = adjacent.iter().copied().find(|f| moved.contains(f)) {
                    let moved_plane =
                        cached_plane(&mut planes, solid, &face_lookup, cap, plane_tolerance)
                            .map_err(|_| {
                                unsupported_carrier(
                                    solid,
                                    &face_lookup,
                                    cap,
                                    &format!(
                                        "the MOVED cap meeting a ruled neighbour at vertex {}",
                                        vertex.id
                                    ),
                                )
                            })?;
                    let map =
                        rim_ruled_map(&frame, rho0, rho1, height, &moved_plane, translation, tolerance)?;
                    new_vertex.insert(vertex.id, map.point(vertex.point));
                    continue;
                }
            }
        }
        // If every fixed carrier is INVARIANT under the push — a plane parallel
        // to it, or an axis-parallel cylinder (e.g. a fillet band) — the corner
        // rides rigidly (stays on all of them + on every translated moved plane).
        if fixed_at.iter().all(|&face_id| {
            carrier_invariant_under(solid, &face_lookup, face_id, translation, parallel_tolerance)
        }) {
            new_vertex.insert(vertex.id, vertex.point.add(translation));
            continue;
        }
        // SM1c multi-rim: a corner shared with a FIXED ruled carrier (a split
        // cylinder/cone band) cannot be placed by the planar 3-plane solver — its
        // carrier is not a plane. Ride it along the single fixed edge it shares:
        // the new corner is where the TRANSLATED cap plane crosses that fixed
        // edge's curve. Consistent + valid only when the affine rim map that
        // carries the adjacent conic rim agrees with this ride (an axis-parallel
        // cylinder generatrix); a cone homothety moves the corner off the fixed
        // wall, so the consistency gate below refuses that (curved multi-rim
        // cone deferred to the rim re-trim path).
        if fixed_at
            .iter()
            .any(|&f| carrier_ruled(solid, &face_lookup, f).is_some())
        {
            let moved_here: Vec<u64> = adjacent
                .iter()
                .copied()
                .filter(|f| moved.contains(f))
                .collect();
            if moved_here.len() != 1 {
                return Err(format!(
                    "move_faces: corner at vertex {} touches {} moved faces against a ruled \
                     neighbour (single-cap multi-rim only) — refusing",
                    vertex.id,
                    moved_here.len()
                ));
            }
            let cap_plane =
                cached_plane(&mut planes, solid, &face_lookup, moved_here[0], plane_tolerance)
                    .map_err(|_| {
                        unsupported_carrier(
                            solid,
                            &face_lookup,
                            moved_here[0],
                            &format!("the MOVED cap at vertex {}", vertex.id),
                        )
                    })?;
            let fixed_edges: Vec<&EdgeRecord> = solid
                .edges
                .iter()
                .filter(|e| {
                    (e.start_vertex_id == vertex.id || e.end_vertex_id == vertex.id)
                        && matches!(classes.get(&e.id), Some(EdgeMoveClass::Fixed))
                })
                .collect();
            if fixed_edges.len() != 1 {
                return Err(format!(
                    "move_faces: corner at vertex {} rides {} fixed edges against a ruled \
                     neighbour (exactly one required) — refusing",
                    vertex.id,
                    fixed_edges.len()
                ));
            }
            let fixed_edge = fixed_edges[0];
            let seed = if fixed_edge.start_vertex_id == vertex.id {
                fixed_edge.t0
            } else {
                fixed_edge.t1
            };
            let translated_origin = cap_plane.origin.add(translation);
            let plane_c = cap_plane.normal.dot(translated_origin);
            // A STRAIGHT fixed edge (a cylinder generatrix on a flat, or a cone's
            // seam meridian — which passes through the apex, so the rim homothety
            // agrees with it) rides its own line: exact, and unchanged since SM1c.
            // A CURVED fixed edge (the hyperbola where a flat cuts a cone) is
            // solved on the CARRIERS instead: the boolean split it exactly at the
            // corner, so riding it could only ever move the corner inward.
            let straight_fixed_edge = fixed_edge.curve.degree == 1
                && fixed_edge.curve.control_points.len() == 2;
            let carriers = if straight_fixed_edge {
                None
            } else {
                plane_and_ruled_carriers(
                    solid,
                    &face_lookup,
                    &mut planes,
                    &fixed_at,
                    plane_tolerance,
                )
            };
            let corner = match carriers {
                Some((fixed_plane, (frame, rho0, rho1, height))) => corner_on_plane_and_ruled(
                    &fixed_plane,
                    cap_plane.normal,
                    plane_c,
                    &frame,
                    rho0,
                    rho1,
                    height,
                    vertex.point,
                    tolerance,
                )?,
                None => resolve_corner_on_fixed_edge(
                    fixed_edge,
                    seed,
                    cap_plane.normal,
                    plane_c,
                    tolerance,
                )?,
            };
            // The new corner must genuinely sit on EVERY fixed carrier at the
            // vertex (ruled: at its rho_at radius; planar: on the plane), else
            // the group tore away — refuse rather than emit a bad solid.
            for &f in &fixed_at {
                if let Some((frame, rho0, rho1, height)) = carrier_ruled(solid, &face_lookup, f) {
                    let delta = corner.sub(frame.origin);
                    let axial = delta.dot(frame.axis);
                    let radial = delta.sub(frame.axis.scale(axial)).length();
                    let off = (radial - rho_at(rho0, rho1, height, axial)).abs();
                    if off > 10.0 * tolerance {
                        return Err(format!(
                            "move_faces: re-solved corner at vertex {} left its ruled neighbour \
                             (off {off:.3e}) — refusing",
                            vertex.id
                        ));
                    }
                } else {
                    let plane = cached_plane(&mut planes, solid, &face_lookup, f, plane_tolerance)
                        .map_err(|_| {
                            unsupported_carrier(
                                solid,
                                &face_lookup,
                                f,
                                &format!("a FIXED neighbour at vertex {}", vertex.id),
                            )
                        })?;
                    if corner.sub(plane.origin).dot(plane.normal).abs() > 10.0 * tolerance {
                        return Err(format!(
                            "move_faces: re-solved corner at vertex {} left a fixed planar \
                             neighbour — refusing",
                            vertex.id
                        ));
                    }
                }
            }
            // The re-solved corner is TRUTH: it is the only point lying on the
            // fixed wall, the fixed flat AND the translated cap plane at once.
            //
            // The conic rim bordering the fixed ruled carrier is carried by the
            // EXACT affine `rim_ruled_map`, which maps the WHOLE conic correctly
            // (carrier → itself, cap plane → translated cap plane). On a CYLINDER
            // — an axis translation along a generatrix of a flat that contains the
            // axis — its image of the old corner IS this corner, so the rim keeps
            // its affine map untouched. On a CONE it is a homothety about the
            // apex: the rim CURVE is still exact but the arc's ENDPOINT slides off
            // the fixed flat, so the rim edge is RE-BUILT between the re-solved
            // corners instead (`EdgeMoveAction::Replace`, see the Boundary arm).
            // This used to be a consistency gate that refused the cone outright.
            new_vertex.insert(vertex.id, corner);
            continue;
        }
        // Genuine re-intersection: every carrier meeting at the corner must
        // be planar to solve the new corner in closed form.
        let mut corner_planes = Vec::with_capacity(fixed_at.len());
        for &face_id in &fixed_at {
            corner_planes.push(
                cached_plane(&mut planes, solid, &face_lookup, face_id, plane_tolerance).map_err(
                    |_| {
                        unsupported_carrier(
                            solid,
                            &face_lookup,
                            face_id,
                            &format!("a FIXED carrier meeting the moved group at vertex {}", vertex.id),
                        )
                    },
                )?,
            );
        }
        for face_id in adjacent
            .iter()
            .copied()
            .filter(|face_id| moved.contains(face_id))
        {
            let mut plane = cached_plane(&mut planes, solid, &face_lookup, face_id, plane_tolerance)
                .map_err(|_| {
                    unsupported_carrier(
                        solid,
                        &face_lookup,
                        face_id,
                        &format!("a MOVED carrier meeting a fixed neighbour at vertex {}", vertex.id),
                    )
                })?;
            plane.origin = plane.origin.add(translation);
            corner_planes.push(plane);
        }
        let corner = solve_corner(&corner_planes).ok_or_else(|| {
            format!(
                "move_faces: cannot re-intersect the carriers meeting at vertex {} \
                 (parallel or under-constrained planes)",
                vertex.id
            )
        })?;
        // The corner must genuinely sit on EVERY carrier; otherwise the group
        // tears away from its fixed neighbours and no manifold heal exists.
        for plane in &corner_planes {
            if corner.sub(plane.origin).dot(plane.normal).abs() > tolerance {
                return Err(format!(
                    "move_faces: the moved group tears away from its neighbours at \
                     vertex {} — refusing rather than emitting an invalid solid",
                    vertex.id
                ));
            }
        }
        new_vertex.insert(vertex.id, corner);
    }

    // --- Plan every edge update -------------------------------------------
    let vertex_position: HashMap<u64, Vec3> = solid
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.point))
        .collect();
    let mut actions: HashMap<u64, EdgeMoveAction> = HashMap::default();
    for edge in &solid.edges {
        let position = |vertex_id: u64| -> Result<Vec3, String> {
            vertex_position
                .get(&vertex_id)
                .copied()
                .ok_or_else(|| format!("move_faces: missing vertex {vertex_id}"))
        };
        let start_old = position(edge.start_vertex_id)?;
        let end_old = position(edge.end_vertex_id)?;
        let start_new = new_vertex
            .get(&edge.start_vertex_id)
            .copied()
            .unwrap_or(start_old);
        let end_new = new_vertex
            .get(&edge.end_vertex_id)
            .copied()
            .unwrap_or(end_old);
        let rigid = start_new.sub(start_old.add(translation)).length() <= rigid_tolerance
            && end_new.sub(end_old.add(translation)).length() <= rigid_tolerance;
        match &classes[&edge.id] {
            EdgeMoveClass::Fixed => {
                if start_new.sub(start_old).length() == 0.0 && end_new.sub(end_old).length() == 0.0
                {
                    continue; // no endpoint relocated — the edge is untouched
                }
                // A fixed side edge follows its re-solved endpoint, exactly as
                // delete_face_and_heal relocates side edges onto recovered
                // corners. Its faces get re-trimmed: planar faces need their
                // plane; a ruled carrier (the drilled-hole seam, or a cone's
                // extended generatrix) re-trims as a ruled carrier.
                for &face_id in &faces_of_edge[&edge.id] {
                    if carrier_ruled(solid, &face_lookup, face_id).is_none() {
                        cached_plane(&mut planes, solid, &face_lookup, face_id, plane_tolerance)
                            .map_err(|_| {
                                unsupported_carrier(
                                    solid,
                                    &face_lookup,
                                    face_id,
                                    &format!(
                                        "a carrier of fixed edge {}, whose trim the push relocates",
                                        edge.id
                                    ),
                                )
                            })?;
                    }
                }
                if !edge.degenerate
                    && (edge.curve.degree != 1 || edge.curve.control_points.len() != 2)
                {
                    // CURVED fixed edge — the hyperbola where a flat cuts a cone,
                    // whose cap-side endpoint rides along it. BOTH its carriers
                    // stayed put, so the section is unchanged as a SET, but the
                    // boolean split the curve exactly at the corner (`domain ==
                    // [t0, t1]`), leaving no parameter headroom for an outward
                    // push — so the arc is RE-BUILT between its endpoints. (A
                    // straight-chord rebuild, what a degree-1 generatrix gets,
                    // would leave the cone; that is why this used to refuse.)
                    // Same collapse/inversion guards as `plan_straight_rebuild`,
                    // so a tearing push still refuses.
                    let new_chord = end_new.sub(start_new);
                    if new_chord.length() <= tolerance {
                        return Err(format!(
                            "move_faces: the translation collapses edge {} to zero length (a \
                             moved face lands exactly on its neighbour) — refusing",
                            edge.id
                        ));
                    }
                    if end_old.sub(start_old).dot(new_chord) <= 0.0 {
                        return Err(format!(
                            "move_faces: the translation inverts edge {} (a moved face passes \
                             beyond its neighbour) — refusing",
                            edge.id
                        ));
                    }
                    let (fixed_plane, (frame, rho0, rho1, height)) = plane_and_ruled_carriers(
                        solid,
                        &face_lookup,
                        &mut planes,
                        &faces_of_edge[&edge.id],
                        plane_tolerance,
                    )
                    .ok_or_else(|| {
                        format!(
                            "move_faces: curved fixed edge {} is not shared by exactly one plane \
                             and one ruled carrier — refusing",
                            edge.id
                        )
                    })?;
                    let forward =
                        arc_sweeps_forward(&frame, &edge.curve, edge.t0, edge.t1)?;
                    let curve = conic_arc_on_ruled(
                        &frame,
                        rho0,
                        rho1,
                        height,
                        fixed_plane.normal,
                        fixed_plane.normal.dot(fixed_plane.origin),
                        start_new,
                        end_new,
                        forward,
                        tolerance,
                    )?;
                    actions.insert(edge.id, EdgeMoveAction::Replace { curve });
                    continue;
                }
                let action =
                    plan_straight_rebuild(edge, start_old, end_old, start_new, end_new, tolerance)?;
                // A chord rebuilt against a ruled carrier must stay ON it — the
                // seam/generatrix endpoints and midpoint at their own rho_at radius.
                if let EdgeMoveAction::Rebuild { start, end } = &action {
                    for &face_id in &faces_of_edge[&edge.id] {
                        if carrier_ruled(solid, &face_lookup, face_id).is_some() {
                            verify_chord_on_carrier(
                                solid,
                                &face_lookup,
                                face_id,
                                *start,
                                *end,
                                tolerance,
                            )?;
                        }
                    }
                }
                actions.insert(edge.id, action);
            }
            EdgeMoveClass::Interior => {
                if rigid {
                    actions.insert(edge.id, EdgeMoveAction::Translate);
                } else {
                    // A tangential translation left the carriers in place, so
                    // an interior edge must stretch between re-solved corners
                    // instead of riding along (both faces are planar-checked).
                    for &face_id in &faces_of_edge[&edge.id] {
                        cached_plane(&mut planes, solid, &face_lookup, face_id, plane_tolerance)
                            .map_err(|_| {
                                unsupported_carrier(
                                    solid,
                                    &face_lookup,
                                    face_id,
                                    &format!(
                                        "a carrier of interior edge {}, which must stretch between \
                                         re-solved corners",
                                        edge.id
                                    ),
                                )
                            })?;
                    }
                    actions.insert(
                        edge.id,
                        plan_straight_rebuild(
                            edge, start_old, end_old, start_new, end_new, tolerance,
                        )?,
                    );
                }
            }
            EdgeMoveClass::Boundary {
                moved_face,
                fixed_face,
            } => {
                if let Some((frame, rho0, rho1, height)) =
                    carrier_ruled(solid, &face_lookup, *fixed_face)
                {
                    // The rim re-intersects the (unchanged) ruled carrier. Build
                    // that trim as the EXACT affine map of the rim (= the
                    // intersection of the translated cap plane with the carrier):
                    // a homothety about the cone apex, or an axis translation for
                    // a cylinder — for ANY push direction. Then verify it lies on
                    // both modified carriers (the "reapply the trimming"
                    // contract). The moved side must be planar (a cap);
                    // `cached_plane` refuses otherwise.
                    let moved_plane = cached_plane(
                        &mut planes,
                        solid,
                        &face_lookup,
                        *moved_face,
                        plane_tolerance,
                    )
                    .map_err(|_| {
                        unsupported_carrier(
                            solid,
                            &face_lookup,
                            *moved_face,
                            &format!(
                                "the MOVED side of boundary edge {} against a ruled neighbour",
                                edge.id
                            ),
                        )
                    })?;
                    let map =
                        rim_ruled_map(&frame, rho0, rho1, height, &moved_plane, translation, tolerance)?;
                    verify_rim_on_carriers(
                        edge,
                        &map,
                        &frame,
                        rho0,
                        rho1,
                        height,
                        &moved_plane,
                        translation,
                        tolerance,
                    )?;
                    // The affine carries the WHOLE conic exactly, but a CONE's
                    // homothety about the apex slides the ARC's endpoints off the
                    // fixed flat the corners must stay on — and the boolean left
                    // the rim with no parameter headroom (`domain == [t0, t1]`),
                    // so it cannot simply be re-trimmed either. When the map and
                    // the re-solved corners disagree, RE-BUILD the rim as the
                    // exact section of the TRANSLATED cap plane with the carrier,
                    // between those corners. A closed rim (one vertex, no corner)
                    // and a cylinder (whose map already lands on the corners) take
                    // the untouched affine path.
                    let mut replacement = None;
                    if edge.start_vertex_id != edge.end_vertex_id {
                        let old_start = edge.curve.evaluate(edge.t0)?;
                        let old_end = edge.curve.evaluate(edge.t1)?;
                        let rim_start = new_vertex
                            .get(&edge.start_vertex_id)
                            .copied()
                            .unwrap_or(old_start);
                        let rim_end = new_vertex
                            .get(&edge.end_vertex_id)
                            .copied()
                            .unwrap_or(old_end);
                        let drift = map
                            .point(old_start)
                            .sub(rim_start)
                            .length()
                            .max(map.point(old_end).sub(rim_end).length());
                        if drift > 10.0 * tolerance {
                            let forward =
                                arc_sweeps_forward(&frame, &edge.curve, edge.t0, edge.t1)?;
                            let plane_c = moved_plane
                                .normal
                                .dot(moved_plane.origin.add(translation));
                            replacement = Some(conic_arc_on_ruled(
                                &frame,
                                rho0,
                                rho1,
                                height,
                                moved_plane.normal,
                                plane_c,
                                rim_start,
                                rim_end,
                                forward,
                                tolerance,
                            )?);
                        }
                    }
                    actions.insert(
                        edge.id,
                        match replacement {
                            Some(curve) => EdgeMoveAction::Replace { curve },
                            None => EdgeMoveAction::Transform(map),
                        },
                    );
                } else if carrier_is_planar(solid, &face_lookup, *fixed_face) {
                    let fixed_plane = planes[fixed_face];
                    if rigid && translation.dot(fixed_plane.normal).abs() <= parallel_tolerance {
                        // The whole edge slides inside the fixed plane while
                        // staying on the translated moved carrier — exact for any
                        // curve type, no re-intersection needed.
                        actions.insert(edge.id, EdgeMoveAction::Translate);
                    } else {
                        // Real re-intersection: line = translated moved plane ∩
                        // fixed plane, delimited by the re-solved corners. The
                        // moved side must be planar for the chord to stay on it.
                        cached_plane(
                            &mut planes,
                            solid,
                            &face_lookup,
                            *moved_face,
                            plane_tolerance,
                        )
                        .map_err(|_| {
                            unsupported_carrier(
                                solid,
                                &face_lookup,
                                *moved_face,
                                &format!(
                                    "the MOVED side of boundary edge {} against a planar neighbour",
                                    edge.id
                                ),
                            )
                        })?;
                        actions.insert(
                            edge.id,
                            plan_straight_rebuild(
                                edge, start_old, end_old, start_new, end_new, tolerance,
                            )?,
                        );
                    }
                } else {
                    return Err(format!(
                        "move_faces: boundary edge {} borders a non-planar, non-axis-parallel \
                         carrier — refusing (SM3 territory)",
                        edge.id
                    ));
                }
            }
        }
    }

    // --- Plan face updates -------------------------------------------------
    // Edges whose SHAPE changed — rebuilt straight OR affine-transformed rim. A
    // moved face bounding one of these must be RE-TRIMMED (its rim moved to a new
    // curve), not merely surface-translated; else its pcurve goes stale.
    let reshaped: HashSet<u64> = actions
        .iter()
        .filter(|(_, action)| {
            matches!(
                action,
                EdgeMoveAction::Rebuild { .. }
                    | EdgeMoveAction::Transform(_)
                    | EdgeMoveAction::Replace { .. }
            )
        })
        .map(|(edge_id, _)| *edge_id)
        .collect();
    let dirty: HashSet<u64> = actions.keys().copied().collect();
    let mut face_actions: Vec<(usize, usize, FaceMoveAction)> = Vec::new();
    // Fixed cylinder neighbours re-trim in a separate post-pass (they grow the
    // carrier via `extend_ruled_neighbour_over`, which needs `&mut solid`).
    let mut ruled_retrim_faces: Vec<u64> = Vec::new();
    for (shell_index, shell) in solid.shells.iter().enumerate() {
        for (face_index, face) in shell.faces.iter().enumerate() {
            let edge_ids = || {
                face.loops
                    .iter()
                    .flat_map(|loop_record| &loop_record.coedges)
                    .map(|coedge| coedge.edge_id)
            };
            if moved.contains(&face.id) {
                if edge_ids().any(|edge_id| reshaped.contains(&edge_id)) {
                    // A boundary edge changed shape (stretched, or a cap rim grown
                    // by the radial-scale map), so the patch must be re-trimmed
                    // around it; the moved cap is planar (translated), so its plane
                    // simply shifts by `translation` before the retrim.
                    let mut plane =
                        cached_plane(&mut planes, solid, &face_lookup, face.id, plane_tolerance)
                            .map_err(|_| {
                                unsupported_carrier(
                                    solid,
                                    &face_lookup,
                                    face.id,
                                    "the MOVED face whose rim the push reshaped",
                                )
                            })?;
                    plane.origin = plane.origin.add(translation);
                    face_actions.push((shell_index, face_index, FaceMoveAction::Retrim(plane)));
                } else {
                    // Every edge of the face rode along rigidly: shifting the
                    // control net keeps surface, curves, and pcurves in exact
                    // agreement for ANY carrier type.
                    face_actions.push((shell_index, face_index, FaceMoveAction::TranslateSurface));
                }
            } else if edge_ids().any(|edge_id| dirty.contains(&edge_id)) {
                if carrier_is_planar(solid, &face_lookup, face.id) {
                    let plane =
                        cached_plane(&mut planes, solid, &face_lookup, face.id, plane_tolerance)?;
                    face_actions.push((shell_index, face_index, FaceMoveAction::Retrim(plane)));
                } else if carrier_ruled(solid, &face_lookup, face.id).is_some() {
                    // Ruled carrier (drilled hole, boss cap, OR a cone whose trim
                    // extended under an oblique cap push): grow it along its axis +
                    // recompute pcurves in the post-pass.
                    ruled_retrim_faces.push(face.id);
                } else {
                    // Non-planar, non-ruled fixed carrier — refuse (the general
                    // offset path does the genuine curved re-intersection).
                    //
                    // MEASURED, and it corrects a just-landed claim: this arm is
                    // NOT the one a curved fixed neighbour reaches. A curved
                    // neighbour that borders the moved group is refused by the
                    // classification loop's own carrier gate (search
                    // `the fixed neighbour across boundary edge`) hundreds of
                    // lines earlier, so this arm can only be entered by a face
                    // whose edges went dirty WITHOUT it sharing a boundary edge
                    // with the moved group — which needs a vertex of valence
                    // four or more. The earlier census, reading the code rather
                    // than running it, attributed the Plane × Torus refusal to
                    // this arm; the probe (`examples/plane_push_refusal_probe.rs`,
                    // `carrier/torus.*`) shows the classification arm firing.
                    cached_plane(&mut planes, solid, &face_lookup, face.id, plane_tolerance)
                        .map_err(|_| {
                            unsupported_carrier(
                                solid,
                                &face_lookup,
                                face.id,
                                "a FIXED face the push must re-trim",
                            )
                        })?;
                }
            }
        }
    }

    // --- Apply to a fresh clone (the input is never touched) ---------------
    let translate = AffineTransform::new([
        1.0,
        0.0,
        0.0,
        translation.x,
        0.0,
        1.0,
        0.0,
        translation.y,
        0.0,
        0.0,
        1.0,
        translation.z,
        0.0,
        0.0,
        0.0,
        1.0,
    ])?;
    let mut result = solid.clone();
    for edge in &mut result.edges {
        match actions.get(&edge.id) {
            Some(EdgeMoveAction::Translate) => {
                edge.curve = transform_curve(&edge.curve, translate)?;
            }
            Some(EdgeMoveAction::Rebuild { start, end }) => {
                edge.curve = make_line(*start, *end)?;
                edge.t0 = 0.0;
                edge.t1 = 1.0;
            }
            Some(EdgeMoveAction::Transform(map)) => {
                // Affine-map the rim (radial scale about the axis + translate).
                // Rational-quadratic circles map exactly; parameters unchanged.
                edge.curve = transform_curve(&edge.curve, *map)?;
            }
            Some(EdgeMoveAction::Replace { curve }) => {
                // An exactly rebuilt conic arc spans its whole domain by
                // construction, so the trim is the domain.
                let [d0, d1] = curve.domain()?;
                edge.curve = curve.clone();
                edge.t0 = d0;
                edge.t1 = d1;
            }
            None => {}
        }
    }
    for vertex in &mut result.vertices {
        if let Some(point) = new_vertex.get(&vertex.id) {
            vertex.point = *point;
        }
    }
    let final_edges: HashMap<u64, EdgeRecord> = result
        .edges
        .iter()
        .map(|edge| (edge.id, edge.clone()))
        .collect();
    for (shell_index, face_index, action) in face_actions {
        let face = &mut result.shells[shell_index].faces[face_index];
        match action {
            FaceMoveAction::TranslateSurface => {
                face.surface = transform_surface(&face.surface, translate)?;
            }
            FaceMoveAction::Retrim(plane) => {
                retrim_planar_face(face, &plane, &final_edges, scale, "move_faces")?;
                // `retrim_planar_face` maps each edge's WHOLE curve onto the
                // rebuilt plane, so any edge that represents a strict SUBRANGE
                // of its curve (e.g. a box wall edge a fillet trimmed back)
                // needs the range fitter to span exactly [t0, t1]; full-domain
                // edges keep the exact affine pcurve just built. Same pairing
                // the delete/heal planar retrim uses.
                let face_edges: HashSet<u64> = face
                    .loops
                    .iter()
                    .flat_map(|loop_record| loop_record.coedges.iter().map(|c| c.edge_id))
                    .collect();
                refit_touched_pcurves(
                    face,
                    &final_edges,
                    &face_edges,
                    true,
                    tolerance,
                    "move_faces",
                )?;
            }
        }
    }
    // SM1: fixed ruled (cylinder) neighbours grow along their axis and recompute
    // pcurves on the grown carrier — a separate pass because it needs `&mut result`.
    for face_id in ruled_retrim_faces {
        retrim_ruled_face(&mut result, face_id, &final_edges, tolerance)?;
    }

    // Topology (and therefore genus) is untouched — only geometry moved — so
    // validate() re-checks Euler, loop closure, and pcurve agreement.
    let issues = result.validate();
    if !issues.is_empty() {
        return Err(format!(
            "move_faces: moved solid failed validation: {issues:?}"
        ));
    }
    // Belt and braces on top of the per-edge inversion guard: a global
    // inversion flips the signed volume even if every edge kept its direction.
    if let (Ok(before), Ok(after)) = (solid_signed_volume(solid), solid_signed_volume(&result)) {
        if before * after <= 0.0 {
            return Err(
                "move_faces: the translation inverts the solid (signed volume changed sign) \
                 — refusing"
                    .into(),
            );
        }
    }
    Ok(result)
}

/// Backlog #5 (Plane × Sphere): push a single PLANAR face whose FIXED boundary
/// neighbour(s) are SPHERES — e.g. a flat capping a spherical dome / spherical-
/// bottomed pocket. The moved plane translates; where it borders a fixed sphere
/// the new boundary rim is the EXACT circle `translated-plane ∩ sphere` (built
/// seam-aligned from the sphere's own frame so its pcurve crosses the seam
/// cleanly), the sphere's coupled seam meridian slides its rim endpoint to the
/// new latitude (its pcurve is patched in parameter space — the periodic u=0/u=2π
/// pairing is preserved), and both carriers are re-trimmed to the new rim.
///
/// Honest scope (SPHERE neighbours only, this slice): only the AXIS-PERPENDICULAR
/// single-closed-circle rim is supported. Refused cleanly (never a bad solid): a
/// moved GROUP, a moved face that is not planar, any FIXED neighbour that is not
/// a sphere (planar corner re-solve / torus / general revolution are other
/// slices), an OBLIQUE plane × sphere rim, an open / multi-edge rim, a sphere
/// seam that does not lie where the plane re-intersects it, a vanished / tangent
/// cap, and a push that collapses or inverts the solid.
fn move_planar_face_across_sphere(
    solid: &BrepSolid,
    moved: &HashSet<u64>,
    face_lookup: &HashMap<u64, (usize, usize)>,
    faces_of_edge: &HashMap<u64, Vec<u64>>,
    translation: Vec3,
) -> Result<BrepSolid, String> {
    let scale = solid_model_scale(solid);
    let tolerance = (scale * 1e-7).max(1e-9);
    let plane_tolerance = (scale * 1e-6).max(1e-7);

    // Single moved planar face only (groups deferred — they would need the
    // generic corner re-solve against the remaining planar neighbours).
    if moved.len() != 1 {
        return Err(
            "move_faces: a moved GROUP against a sphere neighbour is deferred \
             (single planar face only) — refusing"
                .into(),
        );
    }
    let moved_id = *moved.iter().next().unwrap();
    let &(mshell, mface) = face_lookup
        .get(&moved_id)
        .ok_or_else(|| format!("move_faces: missing moved face {moved_id}"))?;
    // The moved face must itself be planar (it translates rigidly).
    let moved_plane = plane_of_surface(
        &solid.shells[mshell].faces[mface].surface,
        plane_tolerance,
        "move_faces",
    )?;
    let translated_origin = moved_plane.origin.add(translation);

    let edge_by_id: HashMap<u64, &EdgeRecord> =
        solid.edges.iter().map(|e| (e.id, e)).collect();

    // --- Build every sphere rim of the moved planar face -------------------
    struct SphereRim {
        edge_id: u64,
        sphere_id: u64,
        closure_vertex: u64,
        new_circle: NurbsCurve,
        closure_point: Vec3,
        v_rim: f64,
    }
    let mut rims: Vec<SphereRim> = Vec::new();
    let mut closure_vertices: HashSet<u64> = HashSet::default();
    let mut sphere_ids: HashSet<u64> = HashSet::default();

    for loop_record in &solid.shells[mshell].faces[mface].loops {
        for coedge in &loop_record.coedges {
            let edge = *edge_by_id
                .get(&coedge.edge_id)
                .ok_or_else(|| format!("move_faces: missing edge {}", coedge.edge_id))?;
            let uses = faces_of_edge
                .get(&edge.id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            // The one fixed face across this boundary edge.
            let neighbour = uses.iter().copied().find(|f| *f != moved_id);
            let Some(neighbour) = neighbour else {
                return Err(format!(
                    "move_faces: the moved face borders itself along edge {} \
                     (unexpected own edge) — refusing",
                    edge.id
                ));
            };
            let Some((frame, radius)) = carrier_sphere(solid, face_lookup, neighbour) else {
                return Err(format!(
                    "move_faces: the moved planar face borders a non-sphere fixed \
                     neighbour along edge {} (mixed / planar-corner / torus / \
                     revolution neighbours are other slices) — refusing",
                    edge.id
                ));
            };
            // A single CLOSED-circle rim only (start == end vertex).
            if edge.start_vertex_id != edge.end_vertex_id {
                return Err(format!(
                    "move_faces: the plane × sphere rim (edge {}) is not a single closed \
                     circle (open / multi-edge rims are deferred) — refusing",
                    edge.id
                ));
            }
            let center = frame.origin;
            let axis = frame.axis;
            // Only the axis-perpendicular cap keeps the rim a fixed-latitude
            // circle we can build seam-aligned; an oblique section crossing the
            // seam is deferred (matches the sphere-pushed path's own refusal).
            if moved_plane.normal.dot(axis).abs() < 1.0 - 1e-6 {
                return Err(format!(
                    "move_faces: an OBLIQUE plane × sphere rim (edge {}) is deferred \
                     (only axis-perpendicular caps) — refusing",
                    edge.id
                ));
            }
            // Signed axial offset of the TRANSLATED plane from the sphere centre;
            // the new rim is the small circle of radius √(r²−a²) at that height.
            let a = translated_origin.sub(center).dot(axis);
            let rr2 = radius * radius - a * a;
            if rr2 <= tolerance * tolerance {
                return Err(format!(
                    "move_faces: the pushed plane no longer meets the sphere (edge {}: the \
                     cap vanishes / is tangent) — refusing",
                    edge.id
                ));
            }
            let radius_new = rr2.sqrt();
            let circle_center = center.add(axis.scale(a));
            let mut new_circle = crate::make_arc(
                circle_center,
                frame.x_axis,
                frame.y_axis,
                radius_new,
                0.0,
                std::f64::consts::TAU,
            )?;
            // Match the new rim's traversal to the old edge so the preserved
            // coedge `forward` flags keep both loops' winding consistent. A
            // closed circle's reversal keeps its start point, so the seam-aligned
            // closure point is unaffected.
            let closure_old = edge.curve.evaluate(edge.t0)?;
            let old_tan = edge
                .curve
                .evaluate(edge.t0 + 0.01 * (edge.t1 - edge.t0))?
                .sub(closure_old);
            let [c0, c1] = new_circle.domain()?;
            let new_start = new_circle.evaluate(c0)?;
            let new_tan = new_circle.evaluate(c0 + 0.01 * (c1 - c0))?.sub(new_start);
            if old_tan.dot(new_tan) < 0.0 {
                new_circle = new_circle.reversed()?;
            }
            let closure_point = new_circle.evaluate(new_circle.domain()?[0])?;

            // The rim latitude in the (unchanged) sphere's (u, v) space, read off
            // the seam-crossing pcurve (constant v). The coupled seam meridian's
            // rim endpoint slides to this v.
            let (nshell, nface) = find_face(solid, neighbour)
                .ok_or_else(|| format!("move_faces: missing sphere neighbour {neighbour}"))?;
            let sphere_surface = &solid.shells[nshell].faces[nface].surface;
            let rim_pcurve = build_pcurve_on_surface(sphere_surface, &new_circle)?;
            let [q0, q1] = rim_pcurve.domain()?;
            let v_rim = rim_pcurve.evaluate(0.5 * (q0 + q1))?.y;

            closure_vertices.insert(edge.start_vertex_id);
            sphere_ids.insert(neighbour);
            rims.push(SphereRim {
                edge_id: edge.id,
                sphere_id: neighbour,
                closure_vertex: edge.start_vertex_id,
                new_circle,
                closure_point,
                v_rim,
            });
        }
    }
    if rims.is_empty() {
        return Err("move_faces: no plane × sphere rim found — refusing".into());
    }
    // Per closure vertex: where it moves + its new rim latitude (for the meridian).
    let closure_of: HashMap<u64, (Vec3, f64)> = rims
        .iter()
        .map(|r| (r.closure_vertex, (r.closure_point, r.v_rim)))
        .collect();

    // --- Coupled seam meridians (own-only edges of the fixed sphere whose rim
    // endpoint is one of the moved closure vertices) ------------------------
    struct MeridianRetrim {
        edge_id: u64,
        sphere_id: u64,
        moved_end_is_start: bool,
        new_t: f64,
        moved_vertex_old: Vec3,
        moved_vertex_new: Vec3,
        v_rim: f64,
    }
    let mut meridians: Vec<MeridianRetrim> = Vec::new();
    for edge in &solid.edges {
        if edge.degenerate {
            continue; // a pole degeneracy does not move
        }
        let uses = faces_of_edge
            .get(&edge.id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        // Own-only to exactly one of the fixed spheres (a seam meridian).
        let owner = uses.first().copied();
        let Some(owner) = owner else { continue };
        if !sphere_ids.contains(&owner) || !uses.iter().all(|f| *f == owner) {
            continue;
        }
        let start_is_closure = closure_vertices.contains(&edge.start_vertex_id);
        let end_is_closure = closure_vertices.contains(&edge.end_vertex_id);
        if !start_is_closure && !end_is_closure {
            continue; // an uncoupled seam meridian: untouched
        }
        if start_is_closure && end_is_closure {
            return Err(format!(
                "move_faces: sphere seam meridian {} moves at BOTH ends (a sphere zone \
                 with two pushed rims) — deferred, refusing",
                edge.id
            ));
        }
        let moved_end_is_start = start_is_closure;
        let closure_vertex = if moved_end_is_start {
            edge.start_vertex_id
        } else {
            edge.end_vertex_id
        };
        let (moved_vertex_new, v_rim) = *closure_of
            .get(&closure_vertex)
            .ok_or_else(|| "move_faces: seam meridian is not paired with a rim — refusing".to_string())?;
        // Slide the moved endpoint along the (unchanged) meridian curve. The
        // curve is fixed (the sphere is fixed), so this only re-parametrises the
        // trim; project the new rim point onto it and verify it truly lands there
        // (the safety net if `frame.x_axis` were not the seam azimuth).
        let projection = project_point_to_curve(&edge.curve, moved_vertex_new)?;
        if projection.distance > 10.0 * tolerance {
            return Err(format!(
                "move_faces: the re-intersected rim point does not lie on the sphere seam \
                 meridian {} (off {:.3e}) — refusing",
                edge.id, projection.distance
            ));
        }
        let new_t = projection.u;
        let fixed_t = if moved_end_is_start { edge.t1 } else { edge.t0 };
        let old_moved_t = if moved_end_is_start { edge.t0 } else { edge.t1 };
        // The trim must neither collapse nor invert: the moved parameter must
        // stay on the same side of the fixed endpoint as before, with a real span.
        let [dom0, dom1] = edge.curve.domain()?;
        let span = (dom1 - dom0).max(1e-12);
        if (new_t - fixed_t) * (old_moved_t - fixed_t) <= 0.0
            || (new_t - fixed_t).abs() <= 1e-7 * span
            || edge.curve.evaluate(new_t)?.sub(edge.curve.evaluate(fixed_t)?).length() <= tolerance
        {
            return Err(format!(
                "move_faces: the sphere seam meridian {} trim collapses or inverts under \
                 the push — refusing",
                edge.id
            ));
        }
        let moved_vertex_old = edge_point(solid, closure_vertex)?;
        meridians.push(MeridianRetrim {
            edge_id: edge.id,
            sphere_id: owner,
            moved_end_is_start,
            new_t,
            moved_vertex_old,
            moved_vertex_new,
            v_rim,
        });
    }

    // Every moved-face boundary vertex must be an accounted-for rim closure
    // vertex; anything else means an unmodelled corner (refuse rather than leave
    // a vertex un-relocated and fail late).
    for loop_record in &solid.shells[mshell].faces[mface].loops {
        for coedge in &loop_record.coedges {
            let edge = *edge_by_id
                .get(&coedge.edge_id)
                .ok_or_else(|| format!("move_faces: missing edge {}", coedge.edge_id))?;
            for v in [edge.start_vertex_id, edge.end_vertex_id] {
                if !closure_vertices.contains(&v) {
                    return Err(format!(
                        "move_faces: the moved face has a boundary vertex {v} that is not a \
                         sphere-rim closure — refusing",
                    ));
                }
            }
        }
    }

    // --- Apply to a fresh clone (the input is never mutated) ---------------
    let mut result = solid.clone();
    // Rim edges take their new circle.
    for rim in &rims {
        if let Some(edge) = result.edges.iter_mut().find(|e| e.id == rim.edge_id) {
            let [d0, d1] = rim.new_circle.domain()?;
            edge.curve = rim.new_circle.clone();
            edge.t0 = d0;
            edge.t1 = d1;
        }
    }
    // Seam meridians keep their curve; only the moved-end trim slides.
    for mer in &meridians {
        if let Some(edge) = result.edges.iter_mut().find(|e| e.id == mer.edge_id) {
            if mer.moved_end_is_start {
                edge.t0 = mer.new_t;
            } else {
                edge.t1 = mer.new_t;
            }
        }
    }
    // Relocate the rim closure vertices.
    for rim in &rims {
        if let Some(v) = result.vertices.iter_mut().find(|v| v.id == rim.closure_vertex) {
            v.point = rim.closure_point;
        }
    }

    // The moved planar face rides the translated plane and re-trims around its
    // new (smaller/larger) rim circle.
    let final_edges: HashMap<u64, EdgeRecord> =
        result.edges.iter().map(|e| (e.id, e.clone())).collect();
    {
        let mut plane = moved_plane;
        plane.origin = plane.origin.add(translation);
        let face = &mut result.shells[mshell].faces[mface];
        retrim_planar_face(face, &plane, &final_edges, scale, "move_faces")?;
    }

    // Re-trim every fixed sphere: rebuild the rim coedge pcurve on the (unchanged)
    // sphere surface, and patch each coupled seam-meridian coedge's pcurve in
    // parameter space (keep u — preserving the periodic u=0/u=2π pairing — and
    // slide only the moved endpoint's v to the new rim latitude).
    for sphere_id in &sphere_ids {
        let (nshell, nface) = find_face(&result, *sphere_id)
            .ok_or_else(|| format!("move_faces: missing sphere neighbour {sphere_id}"))?;
        let sphere_surface = result.shells[nshell].faces[nface].surface.clone();
        for loop_record in &mut result.shells[nshell].faces[nface].loops {
            for coedge in &mut loop_record.coedges {
                if let Some(rim) = rims
                    .iter()
                    .find(|r| r.edge_id == coedge.edge_id && r.sphere_id == *sphere_id)
                {
                    let mut pcurve = build_pcurve_on_surface(&sphere_surface, &rim.new_circle)?;
                    if !coedge.forward {
                        pcurve = pcurve.reversed()?;
                    }
                    coedge.pcurve = pcurve;
                } else if let Some(mer) = meridians
                    .iter()
                    .find(|m| m.edge_id == coedge.edge_id && m.sphere_id == *sphere_id)
                {
                    coedge.pcurve = patch_seam_meridian_pcurve(
                        &coedge.pcurve,
                        &sphere_surface,
                        mer.moved_vertex_old,
                        mer.moved_vertex_new,
                        mer.v_rim,
                        tolerance,
                    )?;
                }
            }
        }
    }

    let issues = result.validate();
    if !issues.is_empty() {
        return Err(format!(
            "move_faces: moved solid failed validation: {issues:?}"
        ));
    }
    if let (Ok(before), Ok(after)) = (solid_signed_volume(solid), solid_signed_volume(&result)) {
        if before * after <= 0.0 {
            return Err(
                "move_faces: the push inverts the solid (signed volume changed sign) — refusing"
                    .into(),
            );
        }
    }
    Ok(result)
}

/// Slide a sphere seam-meridian pcurve's RIM endpoint to the new latitude,
/// keeping its constant u (so the periodic u=0 / u=2π seam pairing survives) and
/// its fixed (pole / other-rim) endpoint. The moved endpoint is identified by
/// which pcurve end maps (through the surface) to the moved vertex's OLD
/// position — not by pole detection — so it makes no assumption about the cap's
/// topology. Endpoint order (domain start→end) is preserved.
fn patch_seam_meridian_pcurve(
    pcurve: &NurbsCurve,
    surface: &NurbsSurface,
    moved_vertex_old: Vec3,
    moved_vertex_new: Vec3,
    v_rim: f64,
    tolerance: f64,
) -> Result<NurbsCurve, String> {
    let [q0, q1] = pcurve.domain()?;
    let a = pcurve.evaluate(q0)?;
    let b = pcurve.evaluate(q1)?;
    let a3 = surface.evaluate(a.x, a.y)?;
    let b3 = surface.evaluate(b.x, b.y)?;
    let da = a3.sub(moved_vertex_old).length();
    let db = b3.sub(moved_vertex_old).length();
    // Guard: one endpoint must genuinely be the moved vertex, and the surface
    // point at the patched (u, v_rim) must land on the relocated vertex.
    let moved_is_a = da <= db;
    let (moved_uv, fixed_uv) = if moved_is_a { (a, b) } else { (b, a) };
    let patched = Vec3::new(moved_uv.x, v_rim, 0.0);
    if surface.evaluate(patched.x, patched.y)?.sub(moved_vertex_new).length() > 10.0 * tolerance {
        return Err(
            "move_faces: patched seam-meridian pcurve endpoint does not reach the new rim \
             vertex — refusing"
                .into(),
        );
    }
    let fixed = Vec3::new(fixed_uv.x, fixed_uv.y, 0.0);
    if moved_is_a {
        make_line(patched, fixed)
    } else {
        make_line(fixed, patched)
    }
}

// BREP private tests: 3d717f5b2660e67f
