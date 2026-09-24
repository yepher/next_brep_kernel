use super::*;

/// How densely a healed rim is sampled when asking whether it follows one of
/// its neighbour's parameter lines. Coarse on purpose: a rim that drifts off a
/// `v = const` line does so over the whole turn, not between two samples.
const RIM_STATION_SAMPLES: usize = 16;

/// The constant `v` of a closed rim that lies on a `v = const` ISO-CURVE of
/// `surface`, or `None` when it does not lie on one.
///
/// This is the general form of the axial `(z / height)` station a
/// `RuledRevolution` states in closed form, and it is what lets a SPHERE,
/// TORUS or general revolution neighbour be rebound by the same code: the
/// healed rim is sampled, each sample projected onto the carrier, and the
/// station accepted only when
///
/// * every sample sits ON the carrier (within `tolerance`) — a rim that has
///   left the neighbour is not its rim at all, and
/// * their `v` values agree to a millionth of the v-domain — otherwise the
///   rim crosses parameter lines and a straight `v = const` pcurve would be a
///   fiction the pcurve/edge consistency check would (rightly) reject.
///
/// Returning `None` rather than a best-fit station is the point: the caller
/// refuses, and no solid is emitted whose trim curve does not follow the
/// boundary it claims to.
fn rim_iso_station(surface: &NurbsSurface, rim: &NurbsCurve, tolerance: f64) -> Option<f64> {
    let [t0, t1] = rim.domain().ok()?;
    let [v0, v1] = surface.domain_v().ok()?;
    let v_span = (v1 - v0).abs().max(1e-12);
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    let mut total = 0.0f64;
    for index in 0..RIM_STATION_SAMPLES {
        let t = t0 + (t1 - t0) * index as f64 / RIM_STATION_SAMPLES as f64;
        let point = rim.evaluate(t).ok()?;
        let projection = crate::project_point_to_surface(surface, point).ok()?;
        if projection.distance > tolerance {
            return None;
        }
        low = low.min(projection.v);
        high = high.max(projection.v);
        total += projection.v;
    }
    if high - low > 1e-6 * v_span {
        return None;
    }
    Some((total / RIM_STATION_SAMPLES as f64).clamp(v0.min(v1), v0.max(v1)))
}

/// Centre, unit normal and radius of the circle through three points, or
/// `None` when they are collinear. Exact (the circumcentre in closed form).
fn circle_through(a: Vec3, b: Vec3, c: Vec3) -> Option<(Vec3, Vec3, f64)> {
    let u = b.sub(a);
    let v = c.sub(a);
    let n = u.cross(v);
    let n_squared = n.dot(n);
    if n_squared <= 1e-24 {
        return None;
    }
    let offset = v
        .scale(u.dot(u))
        .sub(u.scale(v.dot(v)))
        .cross(n)
        .scale(0.5 / n_squared);
    let center = a.add(offset);
    Some((center, n.normalized().ok()?, a.sub(center).length()))
}

/// Put a healed closed rim's PARAMETER ORIGIN on the neighbour's seam.
///
/// A closed rim edge starts and ends at one vertex, and on a periodic carrier
/// that vertex is not free: it is where the rim crosses the surface's seam, so
/// the seam meridian that ends there can still reach it. The re-intersection
/// that produced the rim does not know this. `intersect_plane_quadric` builds a
/// ruled-revolution or torus section with `frame_circle`, whose origin is the
/// carrier frame's own azimuth — the same azimuth `make_cylinder_brep` and
/// friends put the seam at, so those align by construction. Its SPHERE arm
/// instead starts the circle at `plane.normal.perpendicular()`, an azimuth
/// unrelated to the sphere's seam; a half-ball's equator comes back rotated a
/// quarter turn from where the dome's meridian ends.
///
/// So: leave the curve untouched when it already starts on the seam (the
/// aligned-by-construction cases stay byte-for-byte identical), and otherwise
/// rebuild it as the SAME circle re-origined at the seam. Everything is
/// verified — the rim must be a circle to within `tolerance` at every sample,
/// and every sample of the original must lie on the rebuilt curve — and any
/// miss refuses. A section that is not a circle (an oblique plane's ellipse)
/// has no such closed form and is refused rather than approximated.
fn align_closed_rim_origin(
    curve: &NurbsCurve,
    seam: Vec3,
    tolerance: f64,
    op: &str,
) -> Result<NurbsCurve, String> {
    const SAMPLES: usize = 16;
    let [t0, t1] = curve.domain()?;
    let start = curve.evaluate(t0)?;
    let crossing = project_point_to_curve(curve, seam)?;
    let target = curve.evaluate(crossing.u)?;
    if start.sub(target).length() <= tolerance {
        return Ok(curve.clone());
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for index in 0..SAMPLES {
        samples.push(curve.evaluate(t0 + (t1 - t0) * index as f64 / SAMPLES as f64)?);
    }
    let off_seam = start.sub(target).length();
    let decline = || {
        format!(
            "{op}: the healed rim starts {off_seam:.3e} from the neighbour's seam and is not \
             a circle that can be re-origined there — refusing rather than binding a closed \
             rim to a vertex off the seam (deferred)"
        )
    };
    let (center, normal, radius) =
        circle_through(samples[0], samples[SAMPLES / 3], samples[2 * SAMPLES / 3])
            .ok_or_else(decline)?;
    for sample in &samples {
        let radial = sample.sub(center);
        if (radial.length() - radius).abs() > tolerance || radial.dot(normal).abs() > tolerance {
            return Err(decline());
        }
    }
    let radial = target.sub(center);
    let x_axis = radial.sub(normal.scale(radial.dot(normal))).normalized()?;
    let y_axis = normal.cross(x_axis);
    let rebuilt = crate::make_arc(center, x_axis, y_axis, radius, 0.0, std::f64::consts::TAU)?;
    // The rebuilt rim must be the same point set, not merely a similar one.
    for sample in &samples {
        if project_point_to_curve(&rebuilt, *sample)?.distance > tolerance {
            return Err(decline());
        }
    }
    Ok(rebuilt)
}

/// Signed area of ONE of a face's trim loops in the carrier's own parameter
/// space, through the kernel's existing producer: `parameter_space_area`
/// integrates Green's theorem over a face's coedge pcurves exactly (Gauss
/// quadrature per knot span, so a rational arc is exact, not sampled), and a
/// face carrying the single loop is how every other caller asks it for one
/// loop — `offset_shell::carriers`, `healing::face_merge` and `thicken` all
/// spell it this way.
///
/// The sign is what this is for: on a validly trimmed face the loop that
/// bounds the material and the loops that punch holes in it wind OPPOSITE
/// ways. Which absolute sign means "outer" depends on the carrier's uv
/// handedness and on `same_sense`, and neither is worth assuming here — the
/// only claim made downstream is that two loops disagree.
pub(super) fn loop_signed_area(face: &FaceRecord, loop_index: usize) -> Result<f64, String> {
    parameter_space_area(&FaceRecord {
        id: face.id,
        surface: face.surface.clone(),
        same_sense: face.same_sense,
        loops: vec![face.loops[loop_index].clone()],
        name: None,
    })
}

/// `(shell, face, loop)` of the loop `rim` occupies in `neighbour_id`, when
/// that loop is a HOLE the rim owns ALONE — the neighbour-side signature of a
/// through feature's wall.
///
/// Three things have to hold, and each rules out a transition strip:
///
/// * the rim appears in exactly ONE of the neighbour's loops, and is that
///   loop's only coedge — a fillet's rim on a periodic wall shares its loop
///   with the wall's seam and the wall's other rim, so it fails here;
/// * the neighbour carries at least one further loop — a cap disc whose only
///   loop is the rim has no material left once the loop goes, so it is a
///   multi-face gap, not a cap;
/// * the rim's loop winds AGAINST the neighbour's largest loop, i.e. it is a
///   hole in it rather than the loop that bounds its material.
fn rim_hole_loop(
    solid: &BrepSolid,
    neighbour_id: u64,
    rim: u64,
    op: &str,
) -> Result<Option<(usize, usize, usize)>, String> {
    let (shell, face_pos) =
        find_face(solid, neighbour_id).ok_or_else(|| format!("{op}: missing face {neighbour_id}"))?;
    let face = &solid.shells[shell].faces[face_pos];
    if face.loops.len() < 2 {
        return Ok(None);
    }
    let mut carrying = face
        .loops
        .iter()
        .enumerate()
        .filter(|(_, loop_record)| {
            loop_record
                .coedges
                .iter()
                .any(|coedge| coedge.edge_id == rim)
        })
        .map(|(index, _)| index);
    let Some(loop_index) = carrying.next() else {
        return Ok(None);
    };
    if carrying.next().is_some() || face.loops[loop_index].coedges.len() != 1 {
        return Ok(None);
    }
    let mut areas = Vec::with_capacity(face.loops.len());
    for index in 0..face.loops.len() {
        areas.push(loop_signed_area(face, index)?);
    }
    let host = (0..areas.len())
        .max_by(|a, b| areas[*a].abs().total_cmp(&areas[*b].abs()))
        .expect("at least two loops");
    if host == loop_index || areas[host] * areas[loop_index] >= 0.0 {
        return Ok(None);
    }
    Ok(Some((shell, face_pos, loop_index)))
}

/// Both rims' hole loops, when BOTH rims are through-wall rims; `None` as soon
/// as either is not — a strip with one hole rim and one ordinary rim is a
/// transition strip (a fillet round a hole's mouth is exactly that), and it
/// belongs to the re-intersection path.
fn through_wall_hole_loops(
    solid: &BrepSolid,
    rims: &[u64],
    neighbour_ids: &[u64; 2],
    op: &str,
) -> Result<Option<[(usize, usize, usize); 2]>, String> {
    let mut found = Vec::with_capacity(2);
    for (index, rim) in rims.iter().enumerate() {
        match rim_hole_loop(solid, neighbour_ids[index], *rim, op)? {
            Some(position) => found.push(position),
            None => return Ok(None),
        }
    }
    Ok(Some([found[0], found[1]]))
}

/// The face id of a BLIND pocket's floor, when the strip could be that
/// pocket's wall: one rim is a hole loop (the mouth, in the face the pocket
/// was sunk into) and the other is the ENTIRE boundary of a single-loop
/// neighbour (the floor).
///
/// This is a NAME for a refusal, not a decision to refuse. A dome boss on a
/// plate wears the same shape — hole loop on the plate, one rim loop on the
/// dome — and its base fillet re-intersects (plane x sphere) perfectly well.
/// So the caller reads this only once the re-intersection has already failed,
/// to say WHICH configuration failed instead of naming the intersector.
fn blind_pocket_floor(
    solid: &BrepSolid,
    rims: &[u64],
    neighbour_ids: &[u64; 2],
    op: &str,
) -> Result<Option<u64>, String> {
    for near in 0..2usize {
        let far = 1 - near;
        if rim_hole_loop(solid, neighbour_ids[near], rims[near], op)?.is_none() {
            continue;
        }
        let (shell, face_pos) = find_face(solid, neighbour_ids[far])
            .ok_or_else(|| format!("{op}: missing face {}", neighbour_ids[far]))?;
        let floor = &solid.shells[shell].faces[face_pos];
        if floor.loops.len() == 1
            && floor.loops[0].coedges.len() == 1
            && floor.loops[0].coedges[0].edge_id == rims[far]
        {
            return Ok(Some(neighbour_ids[far]));
        }
    }
    Ok(None)
}

/// Whether every face is reachable from the first by walking shared edges.
///
/// `validate()` deliberately does not ask this — a closed complex is checked
/// incidence by incidence — so a shell that has fallen into two closed pieces
/// passes it. The cap is the one heal that can create that, which is why the
/// question is asked here and not in the shared validator.
pub(super) fn faces_are_connected(faces: &[FaceRecord]) -> bool {
    if faces.is_empty() {
        return true;
    }
    let mut by_edge: HashMap<u64, Vec<usize>> = HashMap::default();
    for (index, face) in faces.iter().enumerate() {
        for coedge in face.loops.iter().flat_map(|loop_record| &loop_record.coedges) {
            by_edge.entry(coedge.edge_id).or_default().push(index);
        }
    }
    let mut seen: HashSet<usize> = HashSet::default();
    let mut stack = vec![0usize];
    seen.insert(0);
    while let Some(index) = stack.pop() {
        for coedge in faces[index]
            .loops
            .iter()
            .flat_map(|loop_record| &loop_record.coedges)
        {
            for neighbour in by_edge.get(&coedge.edge_id).into_iter().flatten() {
                if seen.insert(*neighbour) {
                    stack.push(*neighbour);
                }
            }
        }
    }
    seen.len() == faces.len()
}

/// Heal after deleting the WALL OF A THROUGH FEATURE by capping it: the wall
/// face, its private edges (two rims and the doubled seam) and their vertices
/// go, and each neighbour loses the hole loop the wall was bored through. The
/// neighbours' carriers and outer loops are untouched — a plane with its hole
/// loop removed is the whole plane again — so nothing is refit and the capped
/// faces are bit-identical to what they were before the feature was cut.
///
/// Two things are checked before the result is handed back, because a POST
/// joining two otherwise separate parts of a body wears exactly a through
/// wall's topological signature and capping it would SEVER the solid:
///
/// * the shell must still be CONNECTED — this is the exact statement of "the
///   wall was not the only thing holding the body together", and it is
///   checked directly because `validate()` does not test connectivity: two
///   closed surfaces wearing one shell record satisfy every incidence rule it
///   has;
/// * the GENUS must stay non-negative once decremented. Closing a through
///   hole removes exactly one handle, so the new genus is the old one less
///   one; a negative result means the input's own genus disagreed with its
///   topology, and `validate()`'s Euler check would only report that
///   afterwards, in terms of a formula rather than of the operation.
fn cap_through_wall(
    solid: &BrepSolid,
    shell_index: usize,
    face_index: usize,
    boundary: &[(u64, bool)],
    holes: [(usize, usize, usize); 2],
    op: &str,
) -> Result<BrepSolid, String> {
    let mut solid = solid.clone();
    let wall_edges: HashSet<u64> = boundary.iter().map(|(edge_id, _)| *edge_id).collect();

    // The hole loops go FIRST, by the positions already resolved against this
    // very topology. Dropping a loop cannot move a face, whereas dropping the
    // wall face renumbers every face behind it in its shell — and the two
    // neighbours are distinct faces (checked upstream), so neither removal
    // disturbs the other's index.
    for (shell, face_pos, loop_index) in holes {
        solid.shells[shell].faces[face_pos].loops.remove(loop_index);
    }
    solid.shells[shell_index].faces.remove(face_index);
    solid.edges.retain(|edge| !wall_edges.contains(&edge.id));
    let used: HashSet<u64> = solid
        .edges
        .iter()
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect();
    solid.vertices.retain(|vertex| used.contains(&vertex.id));

    if !faces_are_connected(&solid.shells[shell_index].faces) {
        return Err(format!(
            "{op}: the deleted wall joins two otherwise separate parts of the body — \
             capping it would sever the solid, which this operation cannot represent \
             (deferred)"
        ));
    }
    solid.genus -= 1;
    if solid.genus < 0 {
        return Err(format!(
            "{op}: capping the wall leaves genus {}, so the solid's stated genus did not \
             account for the through feature it carries (deferred)",
            solid.genus
        ));
    }
    let issues = solid.validate();
    if !issues.is_empty() {
        return Err(format!("{op}: through-wall cap failed validation: {issues:?}"));
    }
    Ok(solid)
}

/// Heal after deleting a CLOSED transition strip (a fillet or chamfer that
/// runs around a full rim): the two closed rim edges each border one
/// neighbour; the neighbours' ANALYTIC carriers re-intersect in exactly one
/// closed curve — the re-sharpened rim. Both carriers extend exactly (a
/// plane widens, a ruled revolution grows along its axis with every pcurve
/// remapped affinely), so a fillet-then-heal round-trip reproduces the
/// original body bit-for-bit up to ids.
pub(super) fn heal_closed_transition(
    solid: &BrepSolid,
    shell_index: usize,
    face_index: usize,
    boundary: &[(u64, bool)],
) -> Result<BrepSolid, String> {
    let op = "delete_face_and_heal";
    let mut solid = solid.clone();
    let scale = solid_model_scale(&solid);
    let tolerance = (scale * 1e-6).max(1e-9);
    let face_id = solid.shells[shell_index].faces[face_index].id;

    // Classify the strip boundary: one OPEN seam edge used twice with
    // opposite senses, two distinct CLOSED rims.
    let mut counts: HashMap<u64, usize> = HashMap::default();
    for (edge_id, _) in boundary {
        *counts.entry(*edge_id).or_default() += 1;
    }
    let seam_id = *counts
        .iter()
        .find(|(_, count)| **count == 2)
        .map(|(id, _)| id)
        .ok_or_else(|| format!("{op}: closed transition without a doubled seam"))?;
    let rims: Vec<u64> = boundary
        .iter()
        .map(|(edge_id, _)| *edge_id)
        .filter(|edge_id| *edge_id != seam_id)
        .collect();
    if rims.len() != 2 || rims[0] == rims[1] {
        return Err(format!("{op}: closed transition needs exactly two rims"));
    }
    for rim in &rims {
        let edge = solid
            .edges
            .iter()
            .find(|edge| edge.id == *rim)
            .ok_or_else(|| format!("{op}: missing rim {rim}"))?;
        if edge.start_vertex_id != edge.end_vertex_id {
            return Err(format!("{op}: rim {rim} is not closed"));
        }
    }

    // The two neighbours and their analytic carriers.
    let mut neighbour_ids = [0u64; 2];
    for (index, rim) in rims.iter().enumerate() {
        neighbour_ids[index] = other_face_of_edge(&solid, *rim, face_id)?;
    }
    if neighbour_ids[0] == neighbour_ids[1] {
        return Err(format!(
            "{op}: both rims border the same neighbour (deferred)"
        ));
    }

    // Not every closed strip is a TRANSITION. When each rim is its
    // neighbour's whole hole loop, the strip is the WALL OF A THROUGH
    // FEATURE — a drilled hole's bore — and the neighbours were never meant
    // to meet: they are the two faces the feature punched through. Deleting
    // that wall means UN-punching it, so the heal is to cap: drop the wall
    // and both hole loops, and the neighbours close back over the opening.
    // This is checked BEFORE the re-intersection because it is a topological
    // signature, not a fallback — a strip with this shape has no sharp rim
    // to recover, and a pair of neighbours that happened to re-intersect
    // somewhere far away would be "sharpened" onto geometry the strip never
    // touched.
    if let Some(holes) = through_wall_hole_loops(&solid, &rims, &neighbour_ids, op)? {
        return cap_through_wall(&solid, shell_index, face_index, boundary, holes, op);
    }
    // A BLIND bore wears HALF that signature: the mouth rim is a hole loop,
    // but the far rim is the floor disc's whole boundary, so capping it would
    // leave the floor with no loop at all. That is a two-face gap this
    // one-face-at-a-time operation cannot close — but it is NOT grounds to
    // refuse here, because the same half-signature also fits a blend that
    // re-intersects perfectly well: a dome boss on a plate puts a hole loop on
    // the plate and a single rim loop on the dome, and its base fillet
    // sharpens against a plane x sphere circle. So the reading is only
    // prepared now and spent on the refusal path below, where the
    // re-intersection has already had its say.
    let blind_floor = blind_pocket_floor(&solid, &rims, &neighbour_ids, op)?;
    let deferral = |reason: &str| match blind_floor {
        Some(floor) => format!(
            "{op}: the wall's far rim is the whole boundary of face {floor} — this is a \
             blind pocket, and its wall and floor are one PATCH: select that face too \
             and the whole pocket caps in a single delete"
        ),
        None => format!("{op}: {reason} (deferred)"),
    };

    // The strip's spatial extent: sampled points of BOTH rims. The healed
    // rim (the neighbours' re-intersection) lies within this region.
    let mut strip_points: Vec<Vec3> = Vec::new();
    for rim in &rims {
        let edge = solid
            .edges
            .iter()
            .find(|edge| edge.id == *rim)
            .ok_or_else(|| format!("{op}: missing rim {rim}"))?;
        for sample in 0..8 {
            let t = edge.t0 + (edge.t1 - edge.t0) * sample as f64 / 8.0;
            strip_points.push(edge.curve.evaluate(t)?);
        }
    }
    // EXTEND-FIRST: a trimmed carrier stops at the strip's near rim, so
    // intersecting the trimmed surfaces finds nothing (the fillet removed
    // exactly the region where they meet). Grow every ruled-revolution
    // neighbour along its axis to cover the strip BEFORE intersecting;
    // planes are analytically unbounded and need no growth for the
    // intersection itself (retrim handles their extents afterwards).
    for &neighbour_id in &neighbour_ids {
        extend_ruled_neighbour_over(&mut solid, neighbour_id, &strip_points, tolerance)?;
    }
    let neighbour_surface = |id: u64, solid: &BrepSolid| -> Result<NurbsSurface, String> {
        let (shell, face) =
            find_face(solid, id).ok_or_else(|| format!("{op}: missing face {id}"))?;
        Ok(solid.shells[shell].faces[face].surface.clone())
    };
    let surface_a = neighbour_surface(neighbour_ids[0], &solid)?;
    let surface_b = neighbour_surface(neighbour_ids[1], &solid)?;
    if std::env::var("BREP_DEBUG_HEAL").is_ok() {
        eprintln!(
            "HEAL neighbours {:?} analytic a={:?} b={:?}",
            neighbour_ids,
            surface_a.analytic().map(std::mem::discriminant),
            surface_b.analytic().map(std::mem::discriminant)
        );
    }
    let curves = intersect_analytic_pair(&surface_a, &surface_b, tolerance)
        .ok_or_else(|| deferral("neighbours are not a recognized analytic pair"))?;
    let closed: Vec<_> = curves
        .into_iter()
        .filter(|curve| {
            let [t0, t1] = curve.domain().unwrap_or([0.0, 1.0]);
            curve
                .evaluate(t0)
                .and_then(|a| curve.evaluate(t1).map(|b| a.sub(b).length()))
                .map(|gap| gap <= tolerance)
                .unwrap_or(false)
        })
        .collect();
    // Pick the re-intersection nearest the strip (a cone x plane pair can
    // yield two circles; the healed rim is the one the strip surrounded).
    let strip_centre = {
        let face = &solid.shells[shell_index].faces[face_index];
        face.surface.evaluate(0.5, 0.5)?
    };
    let new_curve = closed
        .into_iter()
        .min_by(|a, b| {
            let mid = |curve: &crate::NurbsCurve| {
                let [t0, t1] = curve.domain().unwrap_or([0.0, 1.0]);
                curve.evaluate(0.5 * (t0 + t1)).unwrap_or_default()
            };
            mid(a)
                .sub(strip_centre)
                .length()
                .total_cmp(&mid(b).sub(strip_centre).length())
        })
        .ok_or_else(|| deferral("neighbours do not re-intersect in a closed rim"))?;

    // The rim's shared vertex must land on the SEAM of every periodic
    // neighbour, because that is where those neighbours' seam meridians end.
    // A planar neighbour has no seam and imposes nothing; two seamed
    // neighbours that disagree are not a configuration this heal can serve.
    let seam_target: Option<Vec3> = {
        let mut wanted: Option<Vec3> = None;
        for (index, rim) in rims.iter().enumerate() {
            let neighbour_id = neighbour_ids[index];
            let (ns, nf) = find_face(&solid, neighbour_id)
                .ok_or_else(|| format!("{op}: missing neighbour {neighbour_id}"))?;
            if matches!(
                solid.shells[ns].faces[nf].surface.analytic(),
                Some(AnalyticSurface::Plane { .. })
            ) {
                continue;
            }
            let edge = solid
                .edges
                .iter()
                .find(|edge| edge.id == *rim)
                .ok_or_else(|| format!("{op}: missing rim {rim}"))?;
            let seam_point = edge.curve.evaluate(edge.t0)?;
            match wanted {
                None => wanted = Some(seam_point),
                Some(known) if known.sub(seam_point).length() <= tolerance => {}
                Some(_) => {
                    return Err(format!(
                        "{op}: the strip's two curved neighbours put their seams in different \
                         places — the healed rim cannot start on both (deferred)"
                    ))
                }
            }
        }
        wanted
    };
    let new_curve = match seam_target {
        Some(seam) => align_closed_rim_origin(&new_curve, seam, tolerance, op)?,
        None => new_curve,
    };

    // New shared rim edge + its seam vertex.
    let mut next_id = max_topology_id(&solid) + 1;
    let mut alloc = || {
        let value = next_id;
        next_id += 1;
        value
    };
    let [nt0, nt1] = new_curve.domain()?;
    let new_vertex_id = alloc();
    let new_point = new_curve.evaluate(nt0)?;
    solid.vertices.push(VertexRecord {
        id: new_vertex_id,
        point: new_point,
    });
    let new_edge_id = alloc();
    solid.edges.push(EdgeRecord {
        id: new_edge_id,
        curve: new_curve.clone(),
        t0: nt0,
        t1: nt1,
        start_vertex_id: new_vertex_id,
        end_vertex_id: new_vertex_id,
        degenerate: false,
        name: None,
    });

    // Rebind each neighbour's rim coedge onto the new edge, extending the
    // carrier when the new rim lies outside its domain.
    for (index, rim) in rims.iter().enumerate() {
        let neighbour_id = neighbour_ids[index];
        let (shell, face_pos) = find_face(&solid, neighbour_id)
            .ok_or_else(|| format!("{op}: missing neighbour {neighbour_id}"))?;
        // Direction of the loop's walk along the OLD rim, preserved for the
        // new one: coaxial closed rims keep azimuth direction under nearest
        // projection.
        let face = &solid.shells[shell].faces[face_pos];
        let old_coedge = face
            .loops
            .iter()
            .flat_map(|loop_record| &loop_record.coedges)
            .find(|coedge| coedge.edge_id == *rim)
            .ok_or_else(|| format!("{op}: neighbour lost its rim coedge"))?;
        let old_forward = old_coedge.forward;
        let old_edge = solid
            .edges
            .iter()
            .find(|edge| edge.id == *rim)
            .ok_or_else(|| format!("{op}: missing rim {rim}"))?
            .clone();
        // Azimuth agreement at a quarter turn decides the new forward flag.
        let quarter = |curve: &crate::NurbsCurve, forward: bool| -> Result<Vec3, String> {
            let [t0, t1] = curve.domain()?;
            let f = if forward { 0.25 } else { 0.75 };
            curve.evaluate(t0 + (t1 - t0) * f)
        };
        let old_quarter = quarter(&old_edge.curve, old_forward)?;
        let forward_gap = quarter(&new_curve, true)?.sub(old_quarter).length();
        let backward_gap = quarter(&new_curve, false)?.sub(old_quarter).length();
        let new_forward = forward_gap <= backward_gap;

        let is_plane = matches!(face.surface.analytic(), Some(AnalyticSurface::Plane { .. }));
        if is_plane {
            let plane = plane_of_surface(&face.surface, tolerance, op)?;
            let face = &mut solid.shells[shell].faces[face_pos];
            for coedge in face
                .loops
                .iter_mut()
                .flat_map(|loop_record| &mut loop_record.coedges)
            {
                if coedge.edge_id == *rim {
                    coedge.edge_id = new_edge_id;
                    coedge.forward = new_forward;
                }
            }
            let edges_by_id: HashMap<u64, EdgeRecord> = solid
                .edges
                .iter()
                .map(|edge| (edge.id, edge.clone()))
                .collect();
            let face = &mut solid.shells[shell].faces[face_pos];
            retrim_planar_face(face, &plane, &edges_by_id, scale, op)?;
            continue;
        }
        // The rebound rim is the carrier's `v = const` ISO-CURVE at the new
        // station, and the whole rebind is that one number. A ruled
        // revolution states it in closed form; every other revolution-family
        // carrier — sphere, torus, general revolution — is MEASURED off the
        // carrier by projecting the healed rim onto it. Nothing else about
        // this branch is carrier-specific, so the ruled-only gate that used
        // to stand here was narrower than the intersector at line 100: for a
        // Sphere×Plane or Torus×Plane pair `intersect_analytic_pair` returns
        // the exact closed rim and the heal then refused to bind it.
        let mut ruled_arm = false;
        let v_new = match face.surface.analytic() {
            Some(AnalyticSurface::RuledRevolution { frame, height, .. }) => {
                ruled_arm = true;
                // The pre-extension covered the strip, so the new rim's
                // station is inside the (already extended) carrier.
                let axial = new_point.sub(frame.origin).dot(frame.axis);
                if !(-tolerance..=height + tolerance).contains(&axial) {
                    return Err(format!(
                        "{op}: healed rim escapes the extended carrier \
                         (station {axial:.6} of {height:.6})"
                    ));
                }
                (axial / *height).clamp(0.0, 1.0)
            }
            Some(_) => rim_iso_station(&face.surface, &new_curve, tolerance).ok_or_else(|| {
                format!(
                    "{op}: the healed rim is not a `v = const` iso-curve of neighbour \
                     {neighbour_id}'s carrier — refusing rather than binding it to a \
                     parameter line it does not follow (deferred)"
                )
            })?,
            None => {
                return Err(format!(
                    "{op}: curved neighbour {neighbour_id} is a free-form surface (deferred)"
                ))
            }
        };
        let [u_low, u_high] = face.surface.domain_u()?;
        // A `v = const` pcurve running the WHOLE u range is only the rim's
        // trace if the rim starts where that u range starts. On a carrier
        // whose seam sits at u_low that is what `align_closed_rim_origin`
        // arranged, but arranging is not knowing: measure it. A rim that
        // begins mid-domain traces the same circle from a different azimuth,
        // and the straight pcurve would be that circle rotated — geometry
        // that is wrong by up to a diameter while looking perfectly plausible.
        let rim_start = crate::project_point_to_surface(&face.surface, new_point)?;
        let u_span = (u_high - u_low).abs().max(1e-12);
        let at_u_end = (rim_start.u - u_low).abs() <= 1e-6 * u_span
            || (rim_start.u - u_high).abs() <= 1e-6 * u_span;
        if !at_u_end {
            return Err(format!(
                "{op}: the healed rim starts at u={:.6} of neighbour {neighbour_id}'s                  [{u_low:.6}, {u_high:.6}] domain, not at its seam — a whole-domain iso                  pcurve would trace the rim from the wrong azimuth (deferred)",
                rim_start.u
            ));
        }
        // Which way the rim's traversal runs in the carrier's OWN u.
        //
        // For a ruled revolution the answer is known by construction: the rim
        // came from `frame_circle` on the same frame `make_revolution` built
        // the carrier from, so the coedge's sense IS the u sense. Nowhere else
        // is that guaranteed — a sphere's u may wind the opposite way round
        // the same axis — and a `v = const` pcurve laid down the wrong way
        // traces the rim mirrored, matching at both ends and a full diameter
        // out at the quarter turns. So measure it: project the quarter point
        // of the TRAVERSAL and see which end of the u domain it sits nearer.
        let ascending = if ruled_arm {
            new_forward
        } else {
            let [rim_t0, rim_t1] = new_curve.domain()?;
            let fraction = if new_forward { 0.25 } else { 0.75 };
            let quarter = new_curve.evaluate(rim_t0 + (rim_t1 - rim_t0) * fraction)?;
            let quarter_u = crate::project_point_to_surface(&face.surface, quarter)?.u;
            (quarter_u - u_low).abs() < (quarter_u - u_high).abs()
        };
        let face = &mut solid.shells[shell].faces[face_pos];
        for coedge in face
            .loops
            .iter_mut()
            .flat_map(|loop_record| &mut loop_record.coedges)
        {
            if coedge.edge_id == *rim {
                coedge.edge_id = new_edge_id;
                coedge.forward = new_forward;
                // The new rim is the v-iso at its own station.
                let (from_u, to_u) = if ascending {
                    (u_low, u_high)
                } else {
                    (u_high, u_low)
                };
                coedge.pcurve =
                    make_line(Vec3::new(from_u, v_new, 0.0), Vec3::new(to_u, v_new, 0.0))?;
            }
        }
    }

    // Collapse the old rim seam vertices onto the new vertex and extend the
    // straight edges that ended there (a wall's meridian grows to the new
    // rim station).
    let mut collapsed: HashSet<u64> = HashSet::default();
    for rim in &rims {
        if let Some(edge) = solid.edges.iter().find(|edge| edge.id == *rim) {
            collapsed.insert(edge.start_vertex_id);
        }
    }
    let strip_edges: HashSet<u64> = boundary.iter().map(|(edge_id, _)| *edge_id).collect();
    let mut extended_edges: HashMap<u64, (bool, bool)> = HashMap::default();
    // Extended edges whose curve is CURVED and was widened along itself. Their
    // pcurves cannot be fixed by sliding one control point's v: a rational arc
    // is not linear in the surface's v, so the straight two-point pcurve that
    // serves a line meridian would cut the chord in parameter space. They are
    // refit against the carrier over their new range instead.
    let mut widened_edges: HashSet<u64> = HashSet::default();
    for edge in &mut solid.edges {
        if strip_edges.contains(&edge.id) || edge.id == new_edge_id {
            continue;
        }
        let start_hit = collapsed.contains(&edge.start_vertex_id);
        let end_hit = collapsed.contains(&edge.end_vertex_id);
        if !(start_hit || end_hit) {
            continue;
        }
        if start_hit {
            edge.start_vertex_id = new_vertex_id;
        }
        if end_hit {
            edge.end_vertex_id = new_vertex_id;
        }
        if edge.curve.degree == 1 && edge.curve.control_points.len() == 2 {
            // A STRAIGHT meridian (a cylinder's or cone's wall seam) is
            // rebuilt from its endpoints: the prolonged line is the same line,
            // so this is exact.
            let anchor = if start_hit {
                edge.curve.evaluate(edge.t1)?
            } else {
                edge.curve.evaluate(edge.t0)?
            };
            let (from, to) = if start_hit {
                (new_point, anchor)
            } else {
                (anchor, new_point)
            };
            edge.curve = make_line(from, to)?;
            [edge.t0, edge.t1] = edge.curve.domain()?;
        } else {
            // A CURVED seam meridian — a dome's is an arc — must be WIDENED
            // along its own curve. Rebuilding it as a line, which is what this
            // branch used to do unconditionally, would replace the arc with
            // its chord: an edge that no longer lies on either carrier, i.e.
            // exactly the silently-wrong solid the fail-safe contract forbids.
            // When the stored curve does not reach the new station there is
            // nothing honest to widen into, so refuse.
            let projection = project_point_to_curve(&edge.curve, new_point)?;
            if projection.distance > tolerance {
                return Err(format!(
                    "{op}: curved seam edge {} does not reach the healed rim (nearest \
                     point {:.3e} away) — refusing rather than replacing it with a straight \
                     chord (deferred)",
                    edge.id, projection.distance
                ));
            }
            if start_hit {
                edge.t0 = projection.u;
            } else {
                edge.t1 = projection.u;
            }
            let [d0, d1] = edge.curve.domain()?;
            if edge.t1 - edge.t0 <= 1e-9 * (d1 - d0).max(1e-12) {
                return Err(format!(
                    "{op}: healing would collapse curved seam edge {} to zero length",
                    edge.id
                ));
            }
            widened_edges.insert(edge.id);
        }
        extended_edges.insert(edge.id, (start_hit, end_hit));
    }
    // Refit every pcurve of a widened curved edge over its new range, on
    // whatever carrier each referencing face has. Same mechanism the open
    // chain's `refit_touched_pcurves` uses, and for the same reason.
    if !widened_edges.is_empty() {
        let edges_by_id: HashMap<u64, EdgeRecord> = solid
            .edges
            .iter()
            .map(|edge| (edge.id, edge.clone()))
            .collect();
        for shell in &mut solid.shells {
            for face in &mut shell.faces {
                refit_touched_pcurves(face, &edges_by_id, &widened_edges, false, tolerance, op)?;
            }
        }
    }
    // The pcurves referencing an extended edge still stop at the OLD rim
    // station; pull the affected endpoint to the new station. The meridian
    // pcurves are straight two-point lines pinned to a seam branch — moving
    // only the endpoint's v keeps the branch pinning intact.
    for shell in &mut solid.shells {
        for face in &mut shell.faces {
            let station = match face.surface.analytic() {
                Some(AnalyticSurface::RuledRevolution { frame, height, .. }) => {
                    Some((new_point.sub(frame.origin).dot(frame.axis) / height).clamp(0.0, 1.0))
                }
                _ => None,
            };
            let Some(v_new) = station else { continue };
            for coedge in face
                .loops
                .iter_mut()
                .flat_map(|loop_record| &mut loop_record.coedges)
            {
                if widened_edges.contains(&coedge.edge_id) {
                    continue; // already refit against the carrier above
                }
                let Some(&(start_moved, end_moved)) = extended_edges.get(&coedge.edge_id) else {
                    continue;
                };
                if coedge.pcurve.degree != 1 || coedge.pcurve.control_points.len() != 2 {
                    return Err(format!(
                        "{op}: extended edge has a non-line pcurve (deferred)"
                    ));
                }
                // Traversal start maps to the edge start when forward.
                let start_index = if coedge.forward { 0 } else { 1 };
                if start_moved {
                    let control = &mut coedge.pcurve.control_points[start_index];
                    control.y = v_new * control.w;
                }
                if end_moved {
                    let control = &mut coedge.pcurve.control_points[1 - start_index];
                    control.y = v_new * control.w;
                }
            }
        }
    }

    // Drop the strip and its private topology.
    solid.shells[shell_index].faces.remove(face_index);
    solid.edges.retain(|edge| !strip_edges.contains(&edge.id));
    let used: HashSet<u64> = solid
        .edges
        .iter()
        .flat_map(|edge| [edge.start_vertex_id, edge.end_vertex_id])
        .collect();
    solid.vertices.retain(|vertex| used.contains(&vertex.id));

    let issues = solid.validate();
    if !issues.is_empty() {
        return Err(format!(
            "{op}: closed-transition heal failed validation: {issues:?}"
        ));
    }
    Ok(solid)
}
