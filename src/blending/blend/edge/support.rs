use super::*;

// ====================================================================
// OPEN edges (§6.9 transverse construction)
// ====================================================================

/// March an OPEN edge with a small overshoot past both ends (extended
/// curve/surface evaluation) so the support fits reach the end-face
/// crossings.
///
/// Returns the stations together with the INDICES of the two the march
/// snapped exactly onto the edge's start and end parameters — the sections
/// that sit in each vertex's normal plane.  Those indices are the march's own
/// bookkeeping and cannot be recovered from the stations afterwards (see
/// [`vertex_stations`]).
pub(in crate::blend) fn march_open_stations(
    edge: &EdgeRecord,
    first: &BlendMate,
    second: &BlendMate,
    radius_at: &dyn Fn(f64) -> f64,
    overshoot_fraction: f64,
) -> Result<(Vec<Station>, [usize; 2]), String> {
    let span = edge.t1 - edge.t0;
    let overshoot = span * overshoot_fraction;
    let radius_extent = [0.0, 0.5, 1.0]
        .into_iter()
        .map(|fraction| radius_at(edge.t0 + span * fraction).abs())
        .fold(0.0, f64::max);
    // The overshot span is deliberately NOT used here: `curve_model_scale`
    // clamps out-of-domain probes, so it would measure the same extent while
    // reading as if the overshoot mattered.
    let scale = march_model_scale(&edge.curve, edge.t0, edge.t1, radius_extent)?;
    crate::report_scale_migration("blend_march_open", scale, || {
        let a = edge.curve.evaluate(edge.t0).unwrap_or_default();
        let b = edge.curve.evaluate(edge.t0 + span * 0.5).unwrap_or_default();
        a.length().max(b.length()).max(1.0)
    });
    let surface1 = &first.face.surface;
    let surface2 = &second.face.surface;
    let signs = [first.rho.signum(), second.rho.signum()];
    let rho_at = |t: f64| -> [f64; 2] {
        let radius = radius_at(t);
        [signs[0] * radius, signs[1] * radius]
    };
    // Seed at the middle (the best-conditioned station), then march
    // outward to both ends so every station is seeded by its neighbour.
    let mid_index = STATIONS / 2;
    // Uniform parameters across the overshot span, with the two marched
    // parameters nearest the BLEND ENDS snapped to exactly t0 / t1: the
    // end stations are analytically knowable (the end radius's tangent
    // offsets on each wall), and making them interpolation points pins
    // the fitted rows THROUGH them — the end trims and any downstream
    // miter boolean read these row endpoints.  A half-step snap keeps the
    // parameter list strictly increasing.
    let (station_ts, vertex_indices): (Vec<f64>, [usize; 2]) = {
        let mut ts: Vec<f64> = (0..=STATIONS)
            .map(|index| {
                edge.t0 - overshoot + (span + 2.0 * overshoot) * index as f64 / STATIONS as f64
            })
            .collect();
        let mut snapped = [0usize; 2];
        for (slot, target) in [edge.t0, edge.t1].into_iter().enumerate() {
            let mut nearest = 0usize;
            for (index, t) in ts.iter().enumerate() {
                if (t - target).abs() < (ts[nearest] - target).abs() {
                    nearest = index;
                }
            }
            ts[nearest] = target;
            snapped[slot] = nearest;
        }
        (ts, snapped)
    };
    let station_t = |index: usize| station_ts[index];
    let solve_at = |t: f64, seed: [f64; 4]| -> Result<[f64; 4], String> {
        let derivatives = edge.curve.derivatives_extended(t, 1)?;
        solve_station(
            surface1,
            surface2,
            rho_at(t),
            seed,
            derivatives[0],
            derivatives[1].normalized()?,
            scale,
        )
    };
    let mid_seed = {
        let t = station_t(mid_index).clamp(edge.t0, edge.t1);
        let uv1 = edge_uv_on_face(first.coedge, edge, t)?;
        let uv2 = edge_uv_on_face(second.coedge, edge, t)?;
        [uv1[0], uv1[1], uv2[0], uv2[1]]
    };
    let mut solutions = vec![[0.0f64; 4]; STATIONS + 1];
    solutions[mid_index] = solve_at(station_t(mid_index), mid_seed)?;
    for index in (0..mid_index).rev() {
        solutions[index] = solve_at(station_t(index), solutions[index + 1])?;
    }
    for index in mid_index + 1..=STATIONS {
        solutions[index] = solve_at(station_t(index), solutions[index - 1])?;
    }
    let mut stations = Vec::with_capacity(STATIONS + 1);
    for (index, uv) in solutions.iter().enumerate() {
        let t = station_t(index);
        let derivatives = edge.curve.derivatives_extended(t, 1)?;
        let section_point = derivatives[0];
        let section_tangent = derivatives[1].normalized()?;
        let (_, p1, p2, center) = tangency_residual(
            surface1,
            surface2,
            rho_at(t),
            *uv,
            section_point,
            section_tangent,
        )?;
        let n1 = raw_normal(surface1, uv[0], uv[1])?;
        let n2 = raw_normal(surface2, uv[2], uv[3])?;
        let cos_alpha = signs[0] * signs[1] * n1.dot(n2);
        let weight = ((1.0 + cos_alpha) * 0.5).max(0.0).sqrt();
        if weight <= 1e-6 {
            return Err("blend: faces are tangent at a station (α = π)".into());
        }
        let apex = apex_point(p1, n1, p2, n2, center)?;
        stations.push(Station {
            uv1: [uv[0], uv[1]],
            uv2: [uv[2], uv[3]],
            p1,
            p2,
            center,
            weight,
            apex,
        });
    }
    Ok((stations, vertex_indices))
}

/// Clamped open interpolation of the blend rows (no wrap, no seam).
/// The row parameters of the two stations the march snapped onto the edge's
/// vertices, start then end.  `vertex_indices` comes straight from
/// [`march_open_stations`], which placed one station EXACTLY at each vertex
/// parameter; the section there is the only one that lies in the vertex's
/// normal plane, and it is what the corner/flush/cap surgery trims to.
///
/// This used to be re-derived geometrically as "the station whose ball centre
/// is nearest the vertex", on the argument that the centre path is an offset
/// of the edge and so turns its closest approach at the vertex.  That is only
/// true when the section's offset direction is constant along the edge (a
/// straight edge, or planar mates).  On a CURVED support the ball centre
/// swings as it rolls, the distance-to-vertex minimum slides off the vertex
/// station, and a NEIGHBOUR wins by a hair: on the reported box+through-hole
/// document the elliptical rim's two arcs picked stations 0.0018 apart in
/// distance but 9.0e-2 apart in space, and the two halves of one smooth rim
/// then refused each other as sitting on "different balls".
pub(in crate::blend) fn vertex_stations(
    parameters: &[f64],
    vertex_indices: [usize; 2],
) -> Option<[f64; 2]> {
    if parameters.len() < 2 || vertex_indices[0] == vertex_indices[1] {
        return None;
    }
    Some([
        *parameters.get(vertex_indices[0])?,
        *parameters.get(vertex_indices[1])?,
    ])
}

pub(in crate::blend) fn fit_open_rows(
    stations: &[Station],
    parameters: &[f64],
    chamfer: bool,
) -> Result<FittedRows, String> {
    let mut samples_cr = Vec::new();
    let mut samples_mid = Vec::new();
    let mut samples_cs = Vec::new();
    let mut samples_uv1 = Vec::new();
    let mut samples_uv2 = Vec::new();
    let mut samples_center = Vec::new();
    for station in stations {
        samples_cr.push(Vec4::from_point(station.p1, 1.0));
        samples_cs.push(Vec4::from_point(station.p2, 1.0));
        samples_center.push(Vec4::from_point(station.center, 1.0));
        samples_mid.push(Vec4 {
            x: station.apex.x * station.weight,
            y: station.apex.y * station.weight,
            z: station.apex.z * station.weight,
            w: station.weight,
        });
        samples_uv1.push(Vec4::from_point(
            Vec3::new(station.uv1[0], station.uv1[1], 0.0),
            1.0,
        ));
        samples_uv2.push(Vec4::from_point(
            Vec3::new(station.uv2[0], station.uv2[1], 0.0),
            1.0,
        ));
    }
    let cr_pcurve = fit::interpolate_homogeneous(&samples_uv1, FIT_DEGREE, parameters)?;
    let cs_pcurve = fit::interpolate_homogeneous(&samples_uv2, FIT_DEGREE, parameters)?;
    // Rigidly translated sections admit exact extrusion rows, preserving an
    // analytic carrier instead of approximating it with a cubic fitted surface.
    if let Some(rows) = exact_extrusion_rows(stations, parameters, chamfer, cr_pcurve.clone(), cs_pcurve.clone())? {
        return Ok(rows);
    }
    let cr = fit::interpolate_homogeneous(&samples_cr, FIT_DEGREE, parameters)?;
    let cs = fit::interpolate_homogeneous(&samples_cs, FIT_DEGREE, parameters)?;
    let mid = if chamfer {
        None
    } else {
        Some(fit::interpolate_homogeneous(
            &samples_mid,
            FIT_DEGREE,
            parameters,
        )?)
    };
    let u_domain = cr.domain()?;
    let surface = crate::blend::rows::surface_from_rows(FIT_DEGREE, &cr, &cs, mid.as_ref(), false)?;
    let center = fit::interpolate_homogeneous(&samples_center, FIT_DEGREE, parameters)?;
    Ok(FittedRows {
        surface,
        cr,
        cs,
        cr_pcurve,
        cs_pcurve,
        u_domain,
        center: Some(center),
        exact_extrusion: false,
        vertex_stations: None,
    })
}

/// The exact rows of a translational march, or `None` when the stations are
/// not rigid translates of the first along one direction.
///
/// Exactness bar: every station's contact points, apex and centre must sit
/// within `1e-9·(1 + scale)` of the first station's translated by the
/// centre's own displacement, and the centre displacements must be collinear
/// to the same bar.  The station solver converges to `1e-11·(1 + scale)`, so
/// a genuinely straight×planar edge passes with two orders of margin, and a
/// curved carrier fails by its sagitta at the second station.
fn exact_extrusion_rows(
    stations: &[Station],
    parameters: &[f64],
    chamfer: bool,
    cr_pcurve: NurbsCurve,
    cs_pcurve: NurbsCurve,
) -> Result<Option<FittedRows>, String> {
    let (Some(first), Some(last)) = (stations.first(), stations.last()) else {
        return Ok(None);
    };
    if stations.len() < 3 {
        return Ok(None);
    }
    let travel = last.center.sub(first.center);
    let length = travel.length();
    if !(length > 1e-12) {
        return Ok(None);
    }
    let direction = travel.scale(1.0 / length);
    let scale = stations
        .iter()
        .fold(0.0f64, |worst, station| worst.max(station.center.length()))
        .max(1.0);
    let bar = 1e-9 * (1.0 + scale);
    for (station, parameter) in stations.iter().zip(parameters) {
        let shift = station.center.sub(first.center);
        // Collinear with the travel, at the row parameter's fraction of it.
        if shift.sub(direction.scale(shift.dot(direction))).length() > bar
            || (shift.dot(direction) - parameter * length).abs() > bar
        {
            return Ok(None);
        }
        let translate = |point: Vec3| point.add(shift);
        if station.p1.sub(translate(first.p1)).length() > bar
            || station.p2.sub(translate(first.p2)).length() > bar
            || station.apex.sub(translate(first.apex)).length() > bar
            || (station.weight - first.weight).abs() > 1e-12
        {
            return Ok(None);
        }
    }
    // Section along v (z), spine along u: the blend's own convention.
    let section = |z_index: usize, station: &Station| -> Vec4 {
        match z_index {
            0 => Vec4::from_point(station.p1, 1.0),
            1 if !chamfer => Vec4 {
                x: station.apex.x * station.weight,
                y: station.apex.y * station.weight,
                z: station.apex.z * station.weight,
                w: station.weight,
            },
            _ => Vec4::from_point(station.p2, 1.0),
        }
    };
    let columns = if chamfer { 2 } else { 3 };
    let control = vec![
        (0..columns).map(|j| section(j, first)).collect::<Vec<_>>(),
        (0..columns).map(|j| section(j, last)).collect::<Vec<_>>(),
    ];
    let (degree_v, knots_v) = crate::blend::rows::cross_section_basis(chamfer);
    let surface = NurbsSurface::new(1, degree_v, vec![0.0, 0.0, 1.0, 1.0], knots_v, control)?;
    let cr = crate::make_line(first.p1, last.p1)?;
    let cs = crate::make_line(first.p2, last.p2)?;
    let center = crate::make_line(first.center, last.center)?;
    Ok(Some(FittedRows {
        surface,
        cr,
        cs,
        cr_pcurve,
        cs_pcurve,
        u_domain: [0.0, 1.0],
        center: Some(center),
        exact_extrusion: true,
        vertex_stations: None,
    }))
}

/// One end of an open blend: the mates' boundary edges meeting the end
/// vertex, the single end face across the corner, and where the support
/// curves cross those boundary edges.
pub(in crate::blend) struct EndSurgery {
    /// Support-curve parameters at the crossings.
    pub(in crate::blend) cr_parameter: f64,
    pub(in crate::blend) cs_parameter: f64,
    /// The boundary edges being trimmed and their crossing parameters.
    pub(in crate::blend) first_edge_id: u64,
    pub(in crate::blend) first_edge_parameter: f64,
    pub(in crate::blend) second_edge_id: u64,
    pub(in crate::blend) second_edge_parameter: f64,
    pub(in crate::blend) end_face_id: u64,
    /// New vertices at the crossings (filled during surgery).
    pub(in crate::blend) transverse_curve: NurbsCurve,
    pub(in crate::blend) transverse_blend_pcurve: NurbsCurve,
    pub(in crate::blend) transverse_end_pcurve: NurbsCurve,
    /// True when a support crossing only resolved on a looser rung of
    /// [`support_crossing_tolerances`] than the historical 1e-7 — everything
    /// built from it is NEW, so the caller gates it through
    /// [`SpokeCrossings::gate`].
    pub(in crate::blend) escalated: bool,
}

/// How one end of a stripe is closed.
///
/// A stripe that runs out on its own terminates on the single face across the
/// corner ([`EndSurgery`], the §6.9 transverse construction).  A stripe that
/// runs into a corner shared with other stripes does NOT: it stops at the
/// corner ball's station, and the closure is the corner patch sewn onto every
/// stripe that meets there.  Both occupy the same slot of the blend face's
/// loop, which is why one enum covers them.
pub(in crate::blend) enum EndPlan {
    Free(EndSurgery),
    Corner(CornerEnd),
    Miter(MiterEnd),
    Cap(CapEnd),
}

/// A stripe end closed by a CAP: the stripe stops at its own vertex's
/// section and the sliver's cross-section — the section arc plus one leg on
/// each mate from the arc's end to the vertex — is filled as a planar
/// bulkhead.  The edge continuing past the vertex is left untouched.  This
/// is the termination a fillet gets where its edge runs smoothly into an
/// edge that is not selected (tangent chain, no propagation), and it is the
/// cutter's flush end cap built without the cutter.
///
/// The orchestrator commits the arc, the two leg edges, the leg vertices and
/// the bulkhead; the surgery references the arc (like a corner end) and
/// splices each leg into its mate's loop between the rail and the
/// continuing edge.
pub(in crate::blend) struct CapEnd {
    /// Row parameter of the vertex's own section on both rows.
    pub(in crate::blend) station: f64,
    /// Rim vertices: where the section arc meets the first and second mate.
    pub(in crate::blend) first_vertex: u64,
    pub(in crate::blend) second_vertex: u64,
    /// The section arc, committed in the blend loop's walk direction.
    pub(in crate::blend) arc_edge_id: u64,
    pub(in crate::blend) arc_blend_pcurve: NurbsCurve,
    /// The leg on each mate, committed from the rim vertex to the sharp
    /// vertex: `(edge id, pcurve on that mate from rim to vertex)`.
    pub(in crate::blend) first_leg: (u64, NurbsCurve),
    pub(in crate::blend) second_leg: (u64, NurbsCurve),
    /// The sharp vertex the legs meet at.
    pub(in crate::blend) sharp_vertex: u64,
}

/// A stripe end that meets ONE other stripe at a vertex whose third edge
/// stays sharp (§6.9.6): the two blend surfaces run past their corner-ball
/// stations and are trimmed against each other along the seam.
///
/// Unlike a star, the two rails of this stripe stop at DIFFERENT stations —
/// the shared-face rail at the ball's tangency point, the other rail where
/// the seam leaves the strip — and the end may be closed by one edge (the
/// seam) or two (the seam and then the sibling's section on this stripe's
/// other mate, when the seam leaves the sibling first).  The orchestrator
/// commits every edge and vertex before the surgery runs; the surgery only
/// references them.
pub(in crate::blend) struct MiterEnd {
    /// Row parameters where the first and second support rows stop.
    pub(in crate::blend) cr_parameter: f64,
    pub(in crate::blend) cs_parameter: f64,
    /// Rim vertices at those stops.
    pub(in crate::blend) first_vertex: u64,
    pub(in crate::blend) second_vertex: u64,
    /// The edges closing this end, in the order the blend face's loop walks
    /// them from the first rim to the second: `(edge id, stored forward along
    /// that walk, pcurve on the blend in walk direction)`.
    pub(in crate::blend) edges: Vec<(u64, bool, NurbsCurve)>,
}

/// A stripe end that stops at a shared corner ball.
///
/// Every field is decided BEFORE any cutting, from the ball solve
/// ([`crate::blend::solve_corner_ball`]): the rows are trimmed at the station
/// where the stripe's rolling ball IS the corner ball, and the rim vertices
/// are that ball's tangency points — the same two vertices the neighbouring
/// stripes on those faces stop at, so the mate loops close with no connector
/// and nothing is left over to identify and delete.
pub(in crate::blend) struct CornerEnd {
    /// Row parameters of the corner station on the two support rows.
    pub(in crate::blend) cr_parameter: f64,
    pub(in crate::blend) cs_parameter: f64,
    /// The ball's tangency vertices on the first and second mate — created
    /// once per corner and shared by every stripe that ends there.
    pub(in crate::blend) first_vertex: u64,
    pub(in crate::blend) second_vertex: u64,
    /// The end cross-section, already committed as an `EdgeRecord` oriented
    /// the way this blend face's loop walks it (so the blend uses it
    /// `forward = true` and the corner patch uses it `forward = false`).
    pub(in crate::blend) arc_edge_id: u64,
    /// Its pcurve in the blend surface's (t, z) space, in that same walk
    /// direction.
    pub(in crate::blend) arc_blend_pcurve: NurbsCurve,
}

impl EndPlan {
    pub(in crate::blend) fn cr_parameter(&self) -> f64 {
        match self {
            EndPlan::Free(end) => end.cr_parameter,
            EndPlan::Corner(end) => end.cr_parameter,
            EndPlan::Miter(end) => end.cr_parameter,
            EndPlan::Cap(end) => end.station,
        }
    }

    pub(in crate::blend) fn cs_parameter(&self) -> f64 {
        match self {
            EndPlan::Free(end) => end.cs_parameter,
            EndPlan::Corner(end) => end.cs_parameter,
            EndPlan::Miter(end) => end.cs_parameter,
            EndPlan::Cap(end) => end.station,
        }
    }

    pub(in crate::blend) fn free(&self) -> Option<&EndSurgery> {
        match self {
            EndPlan::Free(end) => Some(end),
            EndPlan::Corner(_) | EndPlan::Miter(_) | EndPlan::Cap(_) => None,
        }
    }

    /// The rim vertices an already-planned (corner, miter or cap) end
    /// supplies, first mate then second.
    fn planned_rims(&self) -> Option<(u64, u64)> {
        match self {
            EndPlan::Free(_) => None,
            EndPlan::Corner(end) => Some((end.first_vertex, end.second_vertex)),
            EndPlan::Miter(end) => Some((end.first_vertex, end.second_vertex)),
            EndPlan::Cap(end) => Some((end.first_vertex, end.second_vertex)),
        }
    }

    pub(in crate::blend) fn planned_rim(&self, side_first: bool) -> Option<u64> {
        self.planned_rims()
            .map(|(first, second)| if side_first { first } else { second })
    }
}

/// What one stripe's surgery committed, for the orchestrator to splice
/// against: the ids of the two rail edges it created (a network needs them to
/// insert a miter connector between a rail and the trimmed sharp edge) and
/// the blend face.
pub(in crate::blend) struct SewnStripe {
    pub(in crate::blend) cr_edge_id: u64,
    pub(in crate::blend) cs_edge_id: u64,
    pub(in crate::blend) blend_face_id: u64,
}

/// How one of the four rim points of a stripe resolves.
pub(in crate::blend) enum RimResolution {
    /// Pristine free end: a fresh vertex at the row endpoint, with the
    /// boundary edge trimmed back to it.
    Fresh,
    /// A prior fillet already consumed the whole boundary edge — reuse its far
    /// vertex and delete the boundary.
    Consumed(u64),
    /// A shared corner ball supplies the vertex.  Nothing on the mate is
    /// trimmed here: the neighbouring stripe replaces the boundary outright.
    Corner(u64),
}

impl RimResolution {
    /// The existing vertex to reuse, or `None` when a fresh one is needed.
    pub(in crate::blend) fn existing(&self) -> Option<u64> {
        match self {
            RimResolution::Fresh => None,
            RimResolution::Consumed(id) | RimResolution::Corner(id) => Some(*id),
        }
    }

    pub(in crate::blend) fn is_consumed(&self) -> bool {
        matches!(self, RimResolution::Consumed(_))
    }

    /// True when this rim trims a boundary edge of the mate face.
    pub(in crate::blend) fn trims_boundary(&self) -> bool {
        matches!(self, RimResolution::Fresh)
    }
}

/// Resolve one FREE end of a stripe: the §6.9 transverse construction on the
/// single face across the corner.
///
/// Returns the end plan together with whether both support crossings landed
/// strictly inside the fitted-row domain — the caller marches with a growing
/// overshoot until they do.
pub(in crate::blend) fn resolve_free_end(
    solid: &BrepSolid,
    edge_id: u64,
    first: &BlendMate,
    second: &BlendMate,
    rows: &FittedRows,
    vertex: u64,
    at_start: bool,
) -> Result<(EndSurgery, bool), String> {
    let first_face = first.face;
    let second_face = second.face;
    let first_edge_id = boundary_edge_at_vertex(solid, first_face, vertex, edge_id)?;
    let second_edge_id = boundary_edge_at_vertex(solid, second_face, vertex, edge_id)?;
    if first_edge_id == second_edge_id {
        return Err("blend: mates share their end boundary edge (unsupported)".into());
    }
    let end_face = end_face_id(
        solid,
        first_face.id,
        second_face.id,
        first_edge_id,
        second_edge_id,
    )?;
    let first_boundary = solid
        .edges
        .iter()
        .find(|candidate| candidate.id == first_edge_id)
        .ok_or("blend: end boundary edge missing")?;
    let second_boundary = solid
        .edges
        .iter()
        .find(|candidate| candidate.id == second_edge_id)
        .ok_or("blend: end boundary edge missing")?;
    let first_boundary_coedge = first_face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .find(|coedge| coedge.edge_id == first_edge_id)
        .ok_or("blend: mate does not use its end boundary edge")?;
    let second_boundary_coedge = second_face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
        .find(|coedge| coedge.edge_id == second_edge_id)
        .ok_or("blend: mate does not use its end boundary edge")?;
    let (cr_parameter, first_edge_parameter, cr_escalated) =
        match support_crossing(solid, &rows.cr, first_boundary, at_start) {
            Ok((s, t, _, escalated)) => (s, t, escalated),
            Err(_) => support_crossing_uv(
                solid,
                &rows.cr,
                &rows.cr_pcurve,
                first_boundary,
                first_boundary_coedge,
                at_start,
            )?,
        };
    let (cs_parameter, second_edge_parameter, cs_escalated) =
        match support_crossing(solid, &rows.cs, second_boundary, at_start) {
            Ok((s, t, _, escalated)) => (s, t, escalated),
            Err(_) => support_crossing_uv(
                solid,
                &rows.cs,
                &rows.cs_pcurve,
                second_boundary,
                second_boundary_coedge,
                at_start,
            )?,
        };
    // Crossings must land strictly inside the row domain [0, 1] so the
    // surgery can split the support rows.  The start end sits near 0, the
    // finish end near 1.
    let in_range = if at_start {
        cr_parameter >= RIM_MARGIN && cs_parameter >= RIM_MARGIN
    } else {
        cr_parameter <= 1.0 - RIM_MARGIN && cs_parameter <= 1.0 - RIM_MARGIN
    };
    let end_record = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .find(|face| face.id == end_face)
        .ok_or("blend: end face missing")?;
    let clamp01 = |value: f64| value.clamp(0.0, 1.0);
    let (transverse, blend_pcurve, end_pcurve) = match exact_section_transverse(
        rows,
        end_record,
        clamp01(cr_parameter),
        clamp01(cs_parameter),
    )? {
        Some(exact) => exact,
        None => transverse_curve(
            rows,
            end_record,
            clamp01(cr_parameter),
            clamp01(cs_parameter),
        )?,
    };
    Ok((
        EndSurgery {
            cr_parameter,
            cs_parameter,
            first_edge_id,
            first_edge_parameter,
            second_edge_id,
            second_edge_parameter,
            end_face_id: end_face,
            transverse_curve: transverse,
            transverse_blend_pcurve: blend_pcurve,
            transverse_end_pcurve: end_pcurve,
            escalated: cr_escalated || cs_escalated,
        },
        in_range,
    ))
}

/// The EXACT transverse curve of an exact-extrusion blend on a planar end face
/// perpendicular to the extrusion: the section itself — the surface's own
/// u-iso-curve, a rational arc — with its end-face pcurve mapped control
/// point by control point (an affine map keeps a rational curve rational).
/// `None` when the configuration is not that, and the fitted intersection
/// applies.  This is what keeps a straight×planar fillet's ends exact to the
/// mass integrator's bar instead of a cubic's chord error.
fn exact_section_transverse(
    rows: &FittedRows,
    end_face: &FaceRecord,
    cr_parameter: f64,
    cs_parameter: f64,
) -> Result<Option<(NurbsCurve, NurbsCurve, NurbsCurve)>, String> {
    if !rows.exact_extrusion || (cr_parameter - cs_parameter).abs() > 1e-9 {
        return Ok(None);
    }
    let Some(crate::AnalyticSurface::Plane { u_dir, v_dir, .. }) = end_face.surface.analytic()
    else {
        return Ok(None);
    };
    let Ok(normal) = u_dir.cross(*v_dir).normalized() else {
        return Ok(None);
    };
    let [u0, u1] = rows.u_domain;
    let direction = rows
        .cr
        .evaluate(u1)?
        .sub(rows.cr.evaluate(u0)?)
        .normalized()?;
    if direction.cross(normal).length() > 1e-9 {
        return Ok(None);
    }
    let section = rows.surface.iso_curve_u(cr_parameter)?;
    let [v0, v1] = section.domain()?;
    let blend_pcurve =
        crate::sweep_topology::parameter_line(cr_parameter, v0, cr_parameter, v1)?;
    let mut mapped = Vec::with_capacity(section.control_points.len());
    for control in &section.control_points {
        let point = control.point()?;
        let projection = crate::project_point_to_surface(&end_face.surface, point)?;
        mapped.push(Vec4 {
            x: projection.u * control.w,
            y: projection.v * control.w,
            z: 0.0,
            w: control.w,
        });
    }
    let end_pcurve = NurbsCurve::new(section.degree, section.knots.clone(), mapped)?;
    Ok(Some((section, blend_pcurve, end_pcurve)))
}

/// The corner of an END FACE the blend terminates on: the two boundary
/// coedges that meet at the old corner vertex, plus any DEGENERATE coedges
/// standing between them.
///
/// # Why a pole can stand between them
/// A pole is a legitimate corner.  Where the end face's apex sits ON the
/// blended edge's end vertex — revolve a profile about a line through one of
/// its own points and the cone raised on that point has its apex exactly
/// there — the face's loop reads `… first boundary, POLE, second boundary …`
/// and the pair is NOT adjacent.  The blend cuts that apex away, so the pole
/// coedges are located here and dropped by [`splice_end_corner`], and their
/// edges go with the other boundaries the surgery consumes.  With no pole the
/// walk is the plain `index + 1` this replaced.
pub(in crate::blend) struct EndCorner {
    pub(in crate::blend) loop_index: usize,
    /// The coedge the transverse curve is spliced AFTER, and the one it is
    /// spliced BEFORE, by stable id (indices shift when a sibling end shares
    /// this face).
    pub(in crate::blend) x_coedge_id: u64,
    pub(in crate::blend) y_coedge_id: u64,
    /// Whether `x` is the FIRST mate's boundary (it is `y`'s otherwise).
    pub(in crate::blend) x_is_first: bool,
    /// The pole coedges between them, in loop order, and their edges.
    pub(in crate::blend) pole_coedge_ids: Vec<u64>,
    pub(in crate::blend) pole_edge_ids: Vec<u64>,
}

/// Locate [`EndCorner`] on `face`, or `None` when the two boundary edges do
/// not meet at `corner` in any of its loops.
pub(in crate::blend) fn locate_end_corner(
    solid: &BrepSolid,
    face: &FaceRecord,
    corner: u64,
    first_edge_id: u64,
    second_edge_id: u64,
) -> Option<EndCorner> {
    let edge_of = |coedge: &CoedgeRecord| {
        solid
            .edges
            .iter()
            .find(|candidate| candidate.id == coedge.edge_id)
    };
    let traversal = |coedge: &CoedgeRecord| -> Option<(u64, u64)> {
        let edge = edge_of(coedge)?;
        Some(if coedge.forward {
            (edge.start_vertex_id, edge.end_vertex_id)
        } else {
            (edge.end_vertex_id, edge.start_vertex_id)
        })
    };
    // Only a degenerate edge AT THIS CORNER is stepped over — a pole
    // elsewhere on the loop is a boundary like any other.
    let is_pole = |coedge: &CoedgeRecord| -> bool {
        edge_of(coedge).is_some_and(|edge| {
            edge.degenerate && edge.start_vertex_id == corner && edge.end_vertex_id == corner
        })
    };
    for (loop_index, loop_record) in face.loops.iter().enumerate() {
        let count = loop_record.coedges.len();
        for index in 0..count {
            let this = &loop_record.coedges[index];
            if is_pole(this) {
                continue;
            }
            let mut pole_coedge_ids = Vec::new();
            let mut pole_edge_ids = Vec::new();
            let mut step = 1usize;
            while step < count && is_pole(&loop_record.coedges[(index + step) % count]) {
                let pole = &loop_record.coedges[(index + step) % count];
                pole_coedge_ids.push(pole.id);
                pole_edge_ids.push(pole.edge_id);
                step += 1;
            }
            if step >= count {
                continue;
            }
            let next = &loop_record.coedges[(index + step) % count];
            let this_pair = (this.edge_id, next.edge_id);
            if this_pair != (first_edge_id, second_edge_id)
                && this_pair != (second_edge_id, first_edge_id)
            {
                continue;
            }
            let (Some((_, this_end)), Some((next_start, _))) = (traversal(this), traversal(next))
            else {
                continue;
            };
            if this_end != corner || next_start != corner {
                continue;
            }
            return Some(EndCorner {
                loop_index,
                x_coedge_id: this.id,
                y_coedge_id: next.id,
                x_is_first: this.edge_id == first_edge_id,
                pole_coedge_ids,
                pole_edge_ids,
            });
        }
    }
    None
}

/// Splice the transverse coedge into a located [`EndCorner`]: drop the
/// consumed boundary coedge on either side, drop every pole coedge between
/// them, and keep the rest of the loop in order.
pub(in crate::blend) fn splice_end_corner(
    result: &mut BrepSolid,
    end_face: u64,
    corner: &EndCorner,
    transverse: CoedgeRecord,
    x_consumed: bool,
    y_consumed: bool,
) -> Result<(), String> {
    let face = result
        .shells
        .iter_mut()
        .flat_map(|shell| &mut shell.faces)
        .find(|face| face.id == end_face)
        .ok_or("blend: end face lost during surgery")?;
    let loop_record = face
        .loops
        .get_mut(corner.loop_index)
        .ok_or("blend: end-face corner loop lost during surgery")?;
    let count = loop_record.coedges.len();
    let seg_index = loop_record
        .coedges
        .iter()
        .position(|coedge| coedge.id == corner.x_coedge_id)
        .ok_or("blend: end-face corner coedge lost during surgery")?;
    for (offset, pole) in corner.pole_coedge_ids.iter().enumerate() {
        if loop_record.coedges[(seg_index + 1 + offset) % count].id != *pole {
            return Err("blend: end-face corner pole no longer between the boundaries".into());
        }
    }
    let y_index = (seg_index + 1 + corner.pole_coedge_ids.len()) % count;
    if loop_record.coedges[y_index].id != corner.y_coedge_id {
        return Err("blend: end-face corner pair no longer adjacent".into());
    }
    let mut transverse = Some(transverse);
    let mut rebuilt = Vec::with_capacity(count + 1);
    for index in 0..count {
        let coedge = &loop_record.coedges[index];
        if index == seg_index {
            if !x_consumed {
                rebuilt.push(coedge.clone());
            }
            rebuilt.push(transverse.take().expect("transverse spliced once"));
        } else if index == y_index {
            if !y_consumed {
                rebuilt.push(coedge.clone());
            }
        } else if !corner.pole_coedge_ids.contains(&coedge.id) {
            rebuilt.push(coedge.clone());
        }
    }
    check_loop_branches(&face.surface, &rebuilt)?;
    let loop_record = face
        .loops
        .get_mut(corner.loop_index)
        .ok_or("blend: end-face corner loop lost during surgery")?;
    loop_record.coedges = rebuilt;
    Ok(())
}

/// A rebuilt loop must walk ONE branch of a closed carrier's seam.
///
/// # The defect this is the invariant for
/// Where the blend's sliver runs past the corner into material a THIRD face
/// supplies, the §6.9 end cross-section — the genuine blend × end-carrier
/// intersection, exact to 1e-12 in 3D — meets that carrier on the far side of
/// its seam.  The captured case is a profile revolved about a line through one
/// of its own points: the end face is the cone raised on that point, its APEX
/// is the blend's own end vertex, and the arc passes BEHIND the apex, across
/// the part of the cone that lies inside the other body.  Its pcurve then
/// starts at `u = 1` where the boundary it must join ends at `u = 0` — the
/// same 3D point on the same seam, one period apart in parameter — and the
/// loop encloses the COMPLEMENT of the face: a 332-unit² cone came back as
/// 180, the part lost 158 units³ against a 15.6 closed form, and `validate`
/// saw none of it, because every endpoint still coincides in 3D.
///
/// Only a whole branch is refused (half a period or more).  The ordinary
/// junction mismatch is a fit residual — 4e-5 of the domain on that same part
/// — and the §6.9 EXTEND case, whose transverse deliberately lands past a
/// boundary's trim, keeps its endpoints on the branch it started on.
///
/// Representing what the arc actually wants needs the end face SPLIT along the
/// blend and the fragments re-classified (slice E₃ of
/// `docs/developer/kernel-plans/fillet-stripe-network.md` §4); a NURBS
/// carrier's parameter extension does not wrap, so the loop cannot simply be
/// unwrapped across the seam.  Until that lands the surgery refuses by name
/// and the selection routes to the cutter, whose boolean does split crossing
/// faces.
fn check_loop_branches(surface: &NurbsSurface, coedges: &[CoedgeRecord]) -> Result<(), String> {
    let (closed_u, closed_v) = surface.closed_directions().unwrap_or((false, false));
    if !closed_u && !closed_v {
        return Ok(());
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let periods = [
        if closed_u { u1 - u0 } else { 0.0 },
        if closed_v { v1 - v0 } else { 0.0 },
    ];
    for index in 0..coedges.len() {
        let this = &coedges[index];
        let next = &coedges[(index + 1) % coedges.len()];
        let [_, this_end] = this.pcurve.domain()?;
        let [next_start, _] = next.pcurve.domain()?;
        let leaving = this.pcurve.evaluate(this_end)?;
        let arriving = next.pcurve.evaluate(next_start)?;
        for (axis, gap) in [
            (0usize, (leaving.x - arriving.x).abs()),
            (1, (leaving.y - arriving.y).abs()),
        ] {
            if periods[axis] > 0.0 && gap >= periods[axis] * 0.5 {
                return Err(format!(
                    "blend: the end cross-section joins the end face's loop across its SEAM \
                     (the junction jumps {gap:.6} in the closed direction, a period of \
                     {:.6}) — the blend runs past the corner into material a third face \
                     supplies, and closing it needs that face SPLIT along the blend, which \
                     the rolling-ball surgery does not do",
                    periods[axis]
                ));
            }
        }
    }
    Ok(())
}

/// How far inside the fitted-row domain a support crossing must land for the
/// surgery to be able to split the rows there.
pub(in crate::blend) const RIM_MARGIN: f64 = 2.0e-3;

/// Part-per-extent width of the row ↔ existing-edge coincidence band used by
/// [`detect_row_coincidence`]: the coincidence must hold to
/// `heal_band(extent, ROW_COINCIDENCE_REL)` = `max(model, extent · 1e-6)`,
/// where `extent` is the far-vertex separation the row spans.  Chosen orders
/// of magnitude ABOVE the march/fit numerics (stations solve to ~1e-11·scale
/// and the rows interpolate them) and orders BELOW any genuine sliver the
/// endpoint classifier can admit (the r = 9.999-on-width-10 near miss leaves
/// a 1e-4-of-extent gap — 100× this band — so it can never be eaten).
/// Identity/coincidence taxonomy: size-coupled at the site via `heal_band`,
/// the chartered pattern (geometry/tolerance.rs), never a fat magic band.
const ROW_COINCIDENCE_REL: f64 = 1e-6;

/// A detected full-span coincidence of a blend support row with an EXISTING
/// mate-face boundary edge (OCCT PR #1449 lesson 3, the r = face-width
/// class).  The surgery sews the blend face straight onto `edge_id` instead
/// of creating a coincident fresh support edge, and drops the fully-consumed
/// mate face whose loop collapsed between the two coincident curves.
pub(in crate::blend) struct RowSew {
    /// The existing boundary edge the blend row retraces.
    pub(in crate::blend) edge_id: u64,
    /// Traversal sense the blend face's coedge on that edge must carry
    /// (equal to the dropped face's old sense, so manifold pairing with the
    /// surviving neighbour face is preserved).
    pub(in crate::blend) forward: bool,
    /// Lesson-7 leftovers of the collapsing loop: zero-span edges wholly
    /// owned by the dropped face, collapsed (deleted) with it.
    pub(in crate::blend) collapsed_edges: Vec<u64>,
}

/// Detect that the trimmed support `row` coincides with an existing boundary
/// edge of the mate face over that edge's FULL span, so the mate face is
/// fully consumed (OCCT PR #1449 lesson 3 — the curve-level analog of the
/// endpoint-level `classify` rule in open.rs).
///
/// Fail-safe contract: ANY ambiguity returns `Ok(None)` and the caller keeps
/// the existing surgery path (the sliver face survives, downstream validation
/// still guards it).  `Some` requires ALL of:
///
/// * both end crossings already endpoint-consumed onto DISTINCT far vertices;
/// * the mate face is single-loop (dropping a face with holes would orphan
///   them) and its loop consists of the blended edge, the two consumed
///   boundary edges, exactly ONE existing edge joining the far vertices, and
///   (lesson 7) only zero-span collapse leftovers wholly owned by this face;
/// * the row retraces that edge under the AFFINE parameter map the sewn
///   coedge will carry — sampled in 3D, the exact metric `validate`'s
///   `adaptive_coedge_error` measures (the pr1449 doc sketches this check in
///   the mate's UV space; 3D avoids the anisotropic UV-metric conversion and
///   validates the true downstream contract) — within the tolerance-derived
///   [`ROW_COINCIDENCE_REL`] band, with the OPPOSITE map refused;
/// * relative orientation (the parameter-delta sign of the matching map)
///   agrees with the vertex correspondence, with the dropped face's own use
///   of the edge, and with exactly one opposing use by the surviving
///   neighbour (manifold pairing).
#[allow(clippy::too_many_arguments)]
pub(in crate::blend) fn detect_row_coincidence(
    solid: &BrepSolid,
    mate: &BlendMate,
    blended_edge_id: u64,
    boundary_start_id: u64,
    boundary_finish_id: u64,
    start_consumed: bool,
    start_far_vertex: u64,
    finish_consumed: bool,
    finish_far_vertex: u64,
    row: &NurbsCurve,
    row_traversed_from_start: bool,
) -> Result<Option<RowSew>, String> {
    if !(start_consumed && finish_consumed) {
        return Ok(None);
    }
    if start_far_vertex == finish_far_vertex {
        return Ok(None);
    }
    if mate.face.loops.len() != 1 || mate.loop_index != 0 {
        return Ok(None);
    }
    let edge_by_id = |id: u64| solid.edges.iter().find(|candidate| candidate.id == id);
    let vertex_point = |id: u64| {
        solid
            .vertices
            .iter()
            .find(|candidate| candidate.id == id)
            .map(|vertex| vertex.point)
    };
    // The ONE existing edge joining the two far vertices, plus any leftovers.
    let mut sewn_coedge: Option<&CoedgeRecord> = None;
    let mut leftovers: Vec<u64> = Vec::new();
    for coedge in &mate.face.loops[0].coedges {
        if coedge.edge_id == blended_edge_id
            || coedge.edge_id == boundary_start_id
            || coedge.edge_id == boundary_finish_id
        {
            continue;
        }
        let Some(candidate) = edge_by_id(coedge.edge_id) else {
            return Err("blend: mate loop references a missing edge".into());
        };
        let joins = (candidate.start_vertex_id == start_far_vertex
            && candidate.end_vertex_id == finish_far_vertex)
            || (candidate.start_vertex_id == finish_far_vertex
                && candidate.end_vertex_id == start_far_vertex);
        if joins {
            if sewn_coedge.is_some() {
                return Ok(None);
            }
            sewn_coedge = Some(coedge);
        } else if !leftovers.contains(&candidate.id) {
            leftovers.push(candidate.id);
        }
    }
    let Some(sewn_coedge) = sewn_coedge else {
        return Ok(None);
    };
    let Some(target) = edge_by_id(sewn_coedge.edge_id) else {
        return Ok(None);
    };
    let (Some(start_point), Some(finish_point)) = (
        vertex_point(start_far_vertex),
        vertex_point(finish_far_vertex),
    ) else {
        return Ok(None);
    };
    let extent = finish_point.sub(start_point).length();
    let tolerances = crate::KernelTolerances::for_solid(solid, 1e-7);
    let band = tolerances.heal_band(extent, ROW_COINCIDENCE_REL);
    if extent <= band * 4.0 {
        // Far vertices too close to discriminate orientation — ambiguous.
        return Ok(None);
    }
    // Orientation by vertex correspondence; confirmed below by the matching
    // affine map (the parameter-delta sign) and the manifold senses.
    let aligned = target.start_vertex_id == start_far_vertex;
    let [row_t0, row_t1] = row.domain()?;
    const ROW_COINCIDENCE_SAMPLES: usize = 16;
    for sample in 0..=ROW_COINCIDENCE_SAMPLES {
        let fraction = sample as f64 / ROW_COINCIDENCE_SAMPLES as f64;
        let on_row = row.evaluate(row_t0 + (row_t1 - row_t0) * fraction)?;
        let edge_fraction = if aligned { fraction } else { 1.0 - fraction };
        let on_edge = target
            .curve
            .evaluate(target.t0 + (target.t1 - target.t0) * edge_fraction)?;
        if on_row.sub(on_edge).length() > band {
            return Ok(None);
        }
    }
    // The OPPOSITE map must NOT also fit — a both-ways match means the edge
    // is degenerate at this band and orientation is ambiguous.
    let quarter_row = row.evaluate(row_t0 + (row_t1 - row_t0) * 0.25)?;
    let opposite_fraction = if aligned { 0.75 } else { 0.25 };
    let quarter_opposite = target
        .curve
        .evaluate(target.t0 + (target.t1 - target.t0) * opposite_fraction)?;
    if quarter_row.sub(quarter_opposite).length() <= band {
        return Ok(None);
    }
    // Lesson 7: leftover loop edges are tolerated only as ZERO-SPAN collapse
    // edges (both ends on one vertex, no geometric extent) wholly owned by
    // the dropped face — collapsed with it, never a generic deletion.
    for id in &leftovers {
        let Some(leftover) = edge_by_id(*id) else {
            return Ok(None);
        };
        if leftover.start_vertex_id != leftover.end_vertex_id {
            return Ok(None);
        }
        let a = leftover.curve.evaluate(leftover.t0)?;
        let mid = leftover
            .curve
            .evaluate(leftover.t0 + (leftover.t1 - leftover.t0) * 0.5)?;
        let b = leftover.curve.evaluate(leftover.t1)?;
        if a.sub(b).length() > band || a.sub(mid).length() > band {
            return Ok(None);
        }
        let used_elsewhere = solid.shells.iter().flat_map(|shell| &shell.faces).any(|face| {
            face.id != mate.face.id
                && face
                    .loops
                    .iter()
                    .flat_map(|loop_record| &loop_record.coedges)
                    .any(|coedge| coedge.edge_id == *id)
        });
        if used_elsewhere {
            return Ok(None);
        }
    }
    // Manifold pairing: the blend's traversal of the row determines the sense
    // its coedge on the existing edge must carry; it must equal the dropped
    // face's old sense (whose partner is the surviving neighbour) and be
    // opposed by exactly one use elsewhere.
    let traversal_start = if row_traversed_from_start {
        start_far_vertex
    } else {
        finish_far_vertex
    };
    let forward = target.start_vertex_id == traversal_start;
    if sewn_coedge.forward != forward {
        return Ok(None);
    }
    let mut opposing_uses = 0usize;
    for face in solid.shells.iter().flat_map(|shell| &shell.faces) {
        if face.id == mate.face.id {
            continue;
        }
        for coedge in face.loops.iter().flat_map(|loop_record| &loop_record.coedges) {
            if coedge.edge_id != target.id {
                continue;
            }
            if coedge.forward == forward {
                return Ok(None);
            }
            opposing_uses += 1;
        }
    }
    if opposing_uses != 1 {
        return Ok(None);
    }
    Ok(Some(RowSew {
        edge_id: target.id,
        forward,
        collapsed_edges: leftovers,
    }))
}

/// The boundary edge of `face` that meets the blended `exclude` edge at
/// `vertex`.  Uses loop adjacency: the answer is the `exclude` coedge's
/// loop-neighbour that shares `vertex`.  This DISAMBIGUATES the case where a
/// prior fillet has left more than one edge touching `vertex` (the original
/// plus a fresh fillet/support edge) — a plain incidence scan would see
/// several and could not tell which continues this face's border.
pub(in crate::blend) fn boundary_edge_at_vertex(
    solid: &BrepSolid,
    face: &FaceRecord,
    vertex: u64,
    exclude: u64,
) -> Result<u64, String> {
    let edge_at = |edge_id: u64| -> Option<&EdgeRecord> {
        solid.edges.iter().find(|edge| edge.id == edge_id)
    };
    // Primary: the loop-neighbour of the blended edge that shares the vertex.
    for loop_record in &face.loops {
        let count = loop_record.coedges.len();
        if count == 0 {
            continue;
        }
        for index in 0..count {
            if loop_record.coedges[index].edge_id != exclude {
                continue;
            }
            // Prefer the neighbour on the side of `vertex`, then the other.
            for step in [count - 1, 1] {
                let neighbour = &loop_record.coedges[(index + step) % count];
                if neighbour.edge_id == exclude {
                    continue;
                }
                if let Some(edge) = edge_at(neighbour.edge_id) {
                    if edge.start_vertex_id == vertex || edge.end_vertex_id == vertex {
                        return Ok(edge.id);
                    }
                }
            }
        }
    }
    // Fallback: a plain incidence scan (used when the blended edge is not in
    // this face's loops, e.g. degenerate inputs) — accept a single unique hit.
    let mut found = None;
    for loop_record in &face.loops {
        for coedge in &loop_record.coedges {
            if coedge.edge_id == exclude {
                continue;
            }
            let edge = edge_at(coedge.edge_id).ok_or("blend: loop references missing edge")?;
            if edge.start_vertex_id == vertex || edge.end_vertex_id == vertex {
                if found.is_some() && found != Some(edge.id) {
                    return Err("blend: multiple boundary edges at the end vertex".into());
                }
                found = Some(edge.id);
            }
        }
    }
    found.ok_or_else(|| "blend: no boundary edge at the end vertex".to_string())
}

pub(in crate::blend) fn end_face_id(
    solid: &BrepSolid,
    first_face: u64,
    second_face: u64,
    first_edge: u64,
    second_edge: u64,
) -> Result<u64, String> {
    let mut found = None;
    for shell in &solid.shells {
        for face in &shell.faces {
            if face.id == first_face || face.id == second_face {
                continue;
            }
            let uses_first = face
                .loops
                .iter()
                .flat_map(|l| &l.coedges)
                .any(|coedge| coedge.edge_id == first_edge);
            let uses_second = face
                .loops
                .iter()
                .flat_map(|l| &l.coedges)
                .any(|coedge| coedge.edge_id == second_edge);
            if uses_first && uses_second {
                if found.is_some() {
                    return Err("blend: ambiguous end face".into());
                }
                found = Some(face.id);
            }
        }
    }
    found.ok_or_else(|| "blend: no single end face across the corner".to_string())
}

/// Crossing of the fitted support pcurve with a boundary edge's pcurve in
/// the MATE FACE's parameter space — robust for free-form boundaries, and
/// capable of landing BEYOND the boundary's trim (the §6.9 "trim or
/// EXTEND" case).  Returns (support parameter, boundary EDGE parameter).
pub(in crate::blend) fn support_crossing_uv(
    solid: &BrepSolid,
    support: &NurbsCurve,
    support_pcurve: &NurbsCurve,
    boundary: &EdgeRecord,
    boundary_coedge: &CoedgeRecord,
    at_start: bool,
) -> Result<(f64, f64, bool), String> {
    let [s0, s1] = support_pcurve.domain()?;
    let [q0, q1] = boundary_coedge.pcurve.domain()?;
    let seed_s = if at_start { s0 } else { s1 };
    let mut solved = None;
    for seed_fraction in [0.0f64, 1.0, 0.5] {
        let mut x = [seed_s, q0 + (q1 - q0) * seed_fraction];
        let mut converged = false;
        for _ in 0..NEWTON_ITERATIONS {
            let a = support_pcurve.evaluate_extended(x[0])?;
            let b = boundary_coedge.pcurve.evaluate_extended(x[1])?;
            let residual = [a.x - b.x, a.y - b.y];
            if residual[0].abs().max(residual[1].abs()) <= 1e-12 {
                converged = true;
                break;
            }
            let step = 1e-8;
            let a_probe = support_pcurve.evaluate_extended(x[0] + step)?;
            let b_probe = boundary_coedge.pcurve.evaluate_extended(x[1] + step)?;
            let jacobian = [
                [(a_probe.x - a.x) / step, -(b_probe.x - b.x) / step],
                [(a_probe.y - a.y) / step, -(b_probe.y - b.y) / step],
            ];
            let determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if determinant.abs() <= 1e-16 {
                break;
            }
            x[0] -= (residual[0] * jacobian[1][1] - residual[1] * jacobian[0][1]) / determinant;
            x[1] -= (residual[1] * jacobian[0][0] - residual[0] * jacobian[1][0]) / determinant;
        }
        if converged {
            // Prefer the solution nearest the support's queried end.
            let rank = (x[0] - seed_s).abs();
            if solved
                .map(|(existing_s, _): (f64, f64)| (existing_s - seed_s).abs() > rank)
                .unwrap_or(true)
            {
                solved = Some((x[0], x[1]));
            }
        }
    }
    let (s, q) = solved
        .ok_or_else(|| "blend: support curve does not reach the end boundary edge".to_string())?;
    // The row was marched with an overshoot past both vertices, so a genuine
    // crossing always lies within its domain; a solution outside it is the
    // pcurve extrapolated off its surface image.
    let span = (s1 - s0).abs();
    if s < s0 - 1e-6 * span || s > s1 + 1e-6 * span {
        return Err(format!(
            "blend: support curve does not reach the end boundary edge (the parameter-space \
             crossing lands at {s:.4}, outside the row's [{s0:.4}, {s1:.4}])"
        ));
    }
    // Map the pcurve parameter to the boundary EDGE parameter through the
    // affine traversal contract.
    let fraction = (q - q0) / (q1 - q0);
    let fraction = if boundary_coedge.forward {
        fraction
    } else {
        1.0 - fraction
    };
    let t = boundary.t0 + (boundary.t1 - boundary.t0) * fraction;
    // A UV solution is only a crossing if the two 3D curves meet there.  A
    // rational pcurve evaluated past its domain sweeps away from its surface
    // image, so a boundary the rail never reaches — one that CONTINUES the
    // blended edge tangentially, the rail running parallel to it — can still
    // hand Newton a phantom (s, q).  The §6.9 EXTEND case, a short straight
    // boundary met just past its end, keeps its 3D point on the boundary's
    // line and passes.
    let on_support = support.evaluate_extended(s)?;
    let on_boundary = boundary.curve.evaluate_extended(t)?;
    // Model scale from the boundary's own extent — never from the evaluated
    // points, which an extrapolation can throw arbitrarily far and thereby
    // make any gap look small.
    let scale = boundary
        .curve
        .evaluate(boundary.t0)?
        .length()
        .max(boundary.curve.evaluate(boundary.t1)?.length())
        .max(1.0);
    let gap = on_support.sub(on_boundary).length();
    // Both sides of that subtraction are APPROXIMATIONS of the same rail: the
    // support is a cubic fit through the marched stations, and the crossing
    // parameter came out of the pcurve fit through the SAME stations — two
    // interpolants that agree at every station and drift between them by the
    // fit's own accuracy, which the surface's parameterisation amplifies (a
    // revolution's u is projective, not proportional to angle).  On a 208°
    // arc of a r≈15 revolve that drift is 2e-4, so `1e-6·scale` was demanding
    // more precision of the row than the row carries and a genuinely
    // transverse 43° corner was refused.  The kernel already names the size
    // of that disagreement — see [`support_crossing_tolerances`], which makes
    // exactly this argument for the 3D intersector — so the phantom test is
    // judged against `pcurve_consistency`, the same contract, and never
    // against a literal of its own.  A crossing accepted only past the
    // historical bar is ESCALATED: the caller must validate what it builds.
    let historical = 1e-6 * scale;
    let band = crate::KernelTolerances::for_solid(solid, 1e-7).pcurve_consistency;
    if gap > band.max(historical) {
        return Err(format!(
            "blend: support curve does not reach the end boundary edge (the parameter-space \
             crossing is {gap:.3e} off in 3D — the boundary continues the blended edge rather \
             than crossing its rail)"
        ));
    }
    Ok((s, t, gap > historical))
}

/// Where a fitted support curve crosses a boundary edge (nearest genuine
/// crossing to the given end of the support domain).
pub(in crate::blend) fn support_crossing(
    solid: &BrepSolid,
    support: &NurbsCurve,
    boundary: &EdgeRecord,
    at_start: bool,
) -> Result<(f64, f64, Vec3, bool), String> {
    for tolerance in support_crossing_tolerances(solid) {
        if let Some((s, t, point)) = support_crossing_at(support, boundary, at_start, tolerance)? {
            return Ok((s, t, point, tolerance > 1e-7));
        }
    }
    Err("blend: support curve does not reach the end boundary edge".to_string())
}

/// One rung of [`support_crossing`]'s ladder.
fn support_crossing_at(
    support: &NurbsCurve,
    boundary: &EdgeRecord,
    at_start: bool,
    tolerance: f64,
) -> Result<Option<(f64, f64, Vec3)>, String> {
    let hits = crate::intersect_curves(support, &boundary.curve, tolerance)?;
    let [s0, s1] = support.domain()?;
    let mut best: Option<(f64, f64, Vec3)> = None;
    for hit in hits {
        if hit.t < boundary.t0 - 1e-9 || hit.t > boundary.t1 + 1e-9 {
            continue;
        }
        // The support is a fitted row over [s0, s1]; a hit reported outside
        // it is the intersector extrapolating the row's carrier (a tangent
        // continuation lets a circle's extension meet a line far away) and
        // is not a crossing of the rail.
        let margin = 1e-9 * (1.0 + (s1 - s0).abs());
        if hit.s < s0 - margin || hit.s > s1 + margin {
            continue;
        }
        let rank = if at_start { hit.s - s0 } else { s1 - hit.s };
        if best
            .map(|(existing, ..)| {
                let existing_rank = if at_start {
                    existing - s0
                } else {
                    s1 - existing
                };
                rank < existing_rank
            })
            .unwrap_or(true)
        {
            best = Some((hit.s, hit.t, hit.point));
        }
    }
    Ok(best)
}

/// The tolerance ladder a fitted support row is intersected against, tightest
/// first.
///
/// # Why a ladder and not one number
/// A support row is a FIT through marched stations; the spoke/boundary edge it
/// crosses is the carrier's own EXACT curve.  Both lie on the same carrier by
/// construction, so the crossing exists geometrically — but the fitted row only
/// reproduces the carrier to the fit's accuracy, so in 3D the two curves pass
/// each other with a residual gap of that size instead of meeting exactly.
/// [`crate::intersect_curves`] discards any Newton answer whose 3D gap exceeds
/// the tolerance it was given, so demanding 1e-7 of a fitted row asks for more
/// precision than the row carries and the crossing is reported as ABSENT.
///
/// The kernel already names the size of that disagreement, and names it twice:
/// [`crate::KernelTolerances::intersection_fit`] ("maximum geometric error
/// accepted while fitting SSI curves") and
/// [`crate::KernelTolerances::pcurve_consistency`] ("maximum edge/pcurve/
/// carrier disagreement in a valid in-memory BREP", whose own comment records
/// that *existing fitted NURBS intersections can deviate by a few microns*).
/// Those are the contract; 1e-7 was tighter than any of them.
///
/// The ladder starts at the historical 1e-7 so every crossing that resolves
/// today resolves identically — the looser rungs are reached ONLY where the
/// tight one finds nothing, i.e. only where the caller used to hard-refuse —
/// and it stops at the FIRST rung that produces a crossing, so a crossing is
/// never accepted at a looser tolerance than it actually needs.
pub(in crate::blend) fn support_crossing_tolerances(solid: &BrepSolid) -> [f64; 3] {
    let policy = crate::KernelTolerances::for_solid(solid, 1e-7);
    [1e-7, policy.intersection_fit, policy.pcurve_consistency]
}

/// Where a fitted support row crosses a chain SPOKE edge.
///
/// Returns the crossing nearest the chain vertex as `(support parameter, spoke
/// edge parameter, escalated)`, where `escalated` is true when the crossing
/// only resolved on a looser rung than the historical 1e-7 — the caller MUST
/// then validate what the surgery built (see [`SpokeCrossings`]).  `None` means
/// the row genuinely does not cross the spoke on any rung; the caller renders
/// that refusal, because what it means differs between the closed and open
/// chain lanes.
pub(in crate::blend) fn spoke_crossing(
    solid: &BrepSolid,
    support: &NurbsCurve,
    spoke: &EdgeRecord,
    vertex_point: Vec3,
) -> Result<Option<(f64, f64, bool)>, String> {
    for tolerance in support_crossing_tolerances(solid) {
        let hits = crate::intersect_curves(support, &spoke.curve, tolerance)?;
        let mut best: Option<(f64, f64, f64)> = None;
        for hit in hits {
            if hit.t < spoke.t0 - 1e-9 || hit.t > spoke.t1 + 1e-9 {
                continue;
            }
            let distance = hit.point.sub(vertex_point).length();
            if best.map(|(.., known)| distance < known).unwrap_or(true) {
                best = Some((hit.s, hit.t, distance));
            }
        }
        if let Some((crossing_support, crossing_spoke, _)) = best {
            return Ok(Some((crossing_support, crossing_spoke, tolerance > 1e-7)));
        }
    }
    Ok(None)
}

/// Whether any spoke crossing in one chain surgery needed a looser rung than
/// the historical 1e-7, and the gate that fact imposes on the result.
///
/// # Why the escalation is gated and the tight rung is not
/// The tight rung reproduces master exactly, INCLUDING whatever it already
/// builds — so it is never re-judged here; re-judging it would turn an existing
/// output into a refusal, which is a regression whatever the output's quality.
/// The looser rungs, by contrast, only ever run where master hard-refused, so
/// everything they produce is NEW.  A new output is worth having only if it is
/// a valid solid: a refusal is recoverable (the caller's maximal-valid-subset
/// search, or the user's own edit), a silently invalid solid is not.  So the
/// escalated path must earn its answer, and a radius that marches past what the
/// supports can carry goes back to being refused — with a message that says so
/// — instead of shipping a torn blend.
#[derive(Clone, Copy, Default)]
pub(in crate::blend) struct SpokeCrossings {
    escalated: bool,
}

impl SpokeCrossings {
    pub(in crate::blend) fn note(&mut self, escalated: bool) {
        self.escalated |= escalated;
    }

    /// Accept `result` unless it was built on an escalated crossing AND does
    /// not validate.
    pub(in crate::blend) fn gate(&self, result: BrepSolid) -> Result<BrepSolid, String> {
        if !self.escalated {
            return Ok(result);
        }
        let issues = result.validate();
        if issues.is_empty() {
            return Ok(result);
        }
        Err(format!(
            "blend: chain surgery across a fit-tolerance spoke crossing did not \
             produce a valid solid ({})",
            issues
                .first()
                .map(|issue| issue.message.clone())
                .unwrap_or_default()
        ))
    }
}

/// The transverse curve where the blend surface crosses the end face:
/// per-z Newton in t on the signed distance to the end carrier, endpoints
/// pinned to the rim crossings.
pub(in crate::blend) fn transverse_curve(
    rows: &FittedRows,
    end_face: &FaceRecord,
    cr_parameter: f64,
    cs_parameter: f64,
) -> Result<(NurbsCurve, NurbsCurve, NurbsCurve), String> {
    const SECTIONS: usize = 16;
    let mut points3 = Vec::with_capacity(SECTIONS + 1);
    let mut blend_uv = Vec::with_capacity(SECTIONS + 1);
    let mut end_uv: Vec<[f64; 2]> = Vec::with_capacity(SECTIONS + 1);
    let mut parameters = Vec::with_capacity(SECTIONS + 1);
    for section in 0..=SECTIONS {
        let z = section as f64 / SECTIONS as f64;
        let mut t = cr_parameter + (cs_parameter - cr_parameter) * z;
        if section > 0 && section < SECTIONS {
            for _ in 0..NEWTON_ITERATIONS {
                let point = rows.surface.evaluate_extended(t, z)?;
                let projection = crate::project_point_to_surface(&end_face.surface, point)?;
                let normal = raw_normal(&end_face.surface, projection.u, projection.v)?;
                let distance = point.sub(projection.point).dot(normal);
                if distance.abs() <= 1e-11 {
                    break;
                }
                let step = 1e-7;
                let probe = rows.surface.evaluate_extended(t + step, z)?;
                let probe_projection = crate::project_point_to_surface(&end_face.surface, probe)?;
                let probe_distance = probe.sub(probe_projection.point).dot(raw_normal(
                    &end_face.surface,
                    probe_projection.u,
                    probe_projection.v,
                )?);
                let derivative = (probe_distance - distance) / step;
                if derivative.abs() <= 1e-14 {
                    return Err("blend: transverse Newton stalled".into());
                }
                t -= distance / derivative;
            }
        }
        let point = rows.surface.evaluate_extended(t, z)?;
        let projection = crate::project_point_to_surface(&end_face.surface, point)?;
        points3.push(Vec4::from_point(point, 1.0));
        blend_uv.push(Vec4::from_point(Vec3::new(t, z, 0.0), 1.0));
        end_uv.push([projection.u, projection.v]);
        parameters.push(z);
    }
    unwrap_seam_branch(&end_face.surface, &mut end_uv)?;
    let end_uv: Vec<Vec4> = end_uv
        .into_iter()
        .map(|uv| Vec4::from_point(Vec3::new(uv[0], uv[1], 0.0), 1.0))
        .collect();
    let curve = fit::interpolate_homogeneous(&points3, FIT_DEGREE, &parameters)?;
    let blend_pcurve = fit::interpolate_homogeneous(&blend_uv, FIT_DEGREE, &parameters)?;
    let end_pcurve = fit::interpolate_homogeneous(&end_uv, FIT_DEGREE, &parameters)?;
    Ok((curve, blend_pcurve, end_pcurve))
}

/// Make a sampled parameter sequence CONTINUOUS across a closed carrier's
/// seam, in place.
///
/// # Why a projection alone cannot decide this
/// On a surface closed in u, the parameters `u0` and `u1` name the SAME
/// meridian, so a point sitting on it projects to either, and the projector's
/// answer is arbitrary — measured on the captured revolve-about-its-own-edge
/// part, the rim point where the blend's row meets its mate came back at
/// `u = 1.000000` while every neighbouring sample answered near `u = 0`.  That
/// happens at every free end whose mate boundary IS the end face's seam, which
/// a revolve about a line through the profile produces by construction (the
/// shared generatrix is both).  One sample left on the far branch makes the
/// fitted pcurve walk the long way round the carrier.
///
/// The seam is a parameter artefact, so the fix is the parametric one: choose,
/// for each sample, the branch nearest its neighbour.  The walk starts at the
/// MIDDLE sample and runs outward in both directions — an end sample is the
/// one that lands on a seam, the middle is the one that does not.  Whether the
/// branch the WHOLE curve lands on is the one the end face's loop can join is
/// a separate question, and [`check_loop_branches`] is where it is asked.
fn unwrap_seam_branch(surface: &NurbsSurface, samples: &mut [[f64; 2]]) -> Result<(), String> {
    let (closed_u, closed_v) = surface.closed_directions().unwrap_or((false, false));
    if (!closed_u && !closed_v) || samples.len() < 2 {
        return Ok(());
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let periods = [
        if closed_u { u1 - u0 } else { 0.0 },
        if closed_v { v1 - v0 } else { 0.0 },
    ];
    let snap = |value: f64, reference: f64, period: f64| -> f64 {
        if !(period > 0.0) {
            return value;
        }
        let shifted = value - period * ((value - reference) / period).round();
        if (shifted - reference).abs() < (value - reference).abs() {
            shifted
        } else {
            value
        }
    };
    let middle = samples.len() / 2;
    for index in (0..middle).rev() {
        for axis in 0..2 {
            samples[index][axis] = snap(samples[index][axis], samples[index + 1][axis], periods[axis]);
        }
    }
    for index in middle + 1..samples.len() {
        for axis in 0..2 {
            samples[index][axis] = snap(samples[index][axis], samples[index - 1][axis], periods[axis]);
        }
    }
    Ok(())
}

/// Trim `edge_id` at `parameter`, moving the `old_vertex` endpoint to
/// `new_vertex`, and split every referencing coedge pcurve to the
/// matching fraction (the pcurve domain maps affinely onto [t0, t1]).
pub(in crate::blend) fn trim_edge_at(
    solid: &mut BrepSolid,
    edge_id: u64,
    parameter: f64,
    old_vertex: u64,
    new_vertex: u64,
) -> Result<(), String> {
    let (old_t0, old_t1, trims_start) = {
        let edge = solid
            .edges
            .iter()
            .find(|edge| edge.id == edge_id)
            .ok_or("blend: edge to trim is missing")?;
        (edge.t0, edge.t1, edge.start_vertex_id == old_vertex)
    };
    let fraction = (parameter - old_t0) / (old_t1 - old_t0);
    let interior = (1e-9..=1.0 - 1e-9).contains(&fraction);
    let extends = if trims_start {
        fraction < 1e-9
    } else {
        fraction > 1.0 - 1e-9
    };
    if !interior && !extends {
        return Err("blend: end trim degenerates the boundary edge".into());
    }
    for shell in &mut solid.shells {
        for face in &mut shell.faces {
            for loop_record in &mut face.loops {
                for coedge in &mut loop_record.coedges {
                    if coedge.edge_id != edge_id {
                        continue;
                    }
                    let [q0, q1] = coedge.pcurve.domain()?;
                    // Pcurve fraction runs with the traversal: forward
                    // coedges map fraction f -> q0 + (q1-q0)f, reversed
                    // ones map f -> q1 - (q1-q0)f.
                    let split_q = if coedge.forward {
                        q0 + (q1 - q0) * fraction
                    } else {
                        q1 - (q1 - q0) * fraction
                    };
                    if interior {
                        let keep_upper = trims_start == coedge.forward;
                        let (low, high) = coedge.pcurve.split(split_q)?;
                        coedge.pcurve = if keep_upper { high } else { low };
                    } else {
                        // EXTEND (§6.9): the boundary continues past its old
                        // end to meet the blend.  Exact for linear pcurves;
                        // curved pcurves would need re-fitting.
                        if coedge.pcurve.degree != 1 {
                            return Err("blend: cannot extend a curved boundary pcurve".into());
                        }
                        let a = coedge.pcurve.evaluate(q0)?;
                        let b = coedge.pcurve.evaluate(q1)?;
                        let direction = b.sub(a);
                        let extended_fraction = (split_q - q0) / (q1 - q0);
                        let target = a.add(direction.scale(extended_fraction));
                        let controls = &mut coedge.pcurve.control_points;
                        let end_index = if (trims_start == coedge.forward) == false {
                            controls.len() - 1
                        } else {
                            0
                        };
                        let w = controls[end_index].w;
                        controls[end_index] = crate::Vec4::from_point(target, w);
                    }
                }
            }
        }
    }
    let edge = solid
        .edges
        .iter_mut()
        .find(|edge| edge.id == edge_id)
        .ok_or("blend: edge to trim is missing")?;
    if trims_start {
        edge.t0 = parameter;
        edge.start_vertex_id = new_vertex;
    } else {
        edge.t1 = parameter;
        edge.end_vertex_id = new_vertex;
    }
    Ok(())
}
