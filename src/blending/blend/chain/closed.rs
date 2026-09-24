use super::*;

/// Piece of a support rim assigned to one mate face.
pub(super) struct RimPiece {
    pub(super) face_id: u64,
    pub(super) loop_index: usize,
    /// Global-parameter window of this piece.
    pub(super) window: [f64; 2],
    pub(super) edge_id: u64,
}

/// Blend a chain of conjugated edges with one rolling-ball blend face
/// (Golovanov §6.9.5: all conjugated edges processed together).  A CLOSED
/// chain (stadium rim, T-pipe saddle) welds into a periodic blend; an OPEN
/// chain (line->arc->line capped by end faces) rolls a clamped blend that
/// terminates in a transverse edge on each end face.
pub fn blend_smooth_chain(
    solid: &BrepSolid,
    seed_edge_id: u64,
    radius: f64,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if !(radius > 0.0) || !radius.is_finite() {
        return Err("blend: radius must be positive".into());
    }
    let chain = collect_smooth_chain(solid, seed_edge_id)?;
    if chain.segments.len() < 2 {
        return Err("blend: chain collapsed to a single segment".into());
    }
    if chain.closed {
        blend_closed_smooth_chain(solid, &chain.segments, radius, chamfer, name)
    } else {
        blend_open_smooth_chain(solid, &chain, radius, chamfer, name)
    }
}

/// `blend_smooth_chain` restricted to CLOSED chains: an edge that is one arc
/// of a seam-split rim (a cyl×cyl saddle) is blended as the whole rim, but an
/// OPEN tangent chain is not walked — filleting one edge of it must not
/// fillet the unselected edges it runs into.  Tangent propagation is a
/// selection policy for the feature layer; the kernel caps an unselected
/// continuation instead (`network.rs`).
pub fn blend_smooth_chain_if_closed(
    solid: &BrepSolid,
    seed_edge_id: u64,
    radius: f64,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    if !(radius > 0.0) || !radius.is_finite() {
        return Err("blend: radius must be positive".into());
    }
    let chain = collect_smooth_chain(solid, seed_edge_id)?;
    if chain.segments.len() < 2 {
        return Err("blend: chain collapsed to a single segment".into());
    }
    if !chain.closed {
        return Err(
            "blend: the edge is one arc of an OPEN tangent chain; the unselected \
             continuation is capped, not blended"
                .into(),
        );
    }
    blend_closed_smooth_chain(solid, &chain.segments, radius, chamfer, name)
}

/// Blend a CLOSED chain of conjugated edges with one rolling-ball blend
/// face (Golovanov §6.9.5: all conjugated edges processed together).
fn blend_closed_smooth_chain(
    solid: &BrepSolid,
    segments: &[ChainSegment<'_>],
    radius: f64,
    chamfer: bool,
    name: Option<&str>,
) -> Result<BrepSolid, String> {
    let samples = march_chain(segments, radius, false)?;

    // Global rows from the in-segment samples (wrapped-overlap closed fit).
    let mut global: Vec<(f64, &ChainSample)> = samples
        .iter()
        .filter(|sample| (0..CHAIN_PER_SEGMENT as isize).contains(&sample.position))
        .map(|sample| (sample.parameter.rem_euclid(1.0), sample))
        .collect();
    global.sort_by(|a, b| a.0.total_cmp(&b.0));
    let count = global.len();
    let degree = FIT_DEGREE;
    let mut extended_params = Vec::new();
    let mut samples_cr = Vec::new();
    let mut samples_mid = Vec::new();
    let mut samples_cs = Vec::new();
    let mut push = |station: &Station, parameter: f64| {
        extended_params.push(parameter);
        samples_cr.push(Vec4::from_point(station.p1, 1.0));
        samples_cs.push(Vec4::from_point(station.p2, 1.0));
        samples_mid.push(Vec4 {
            x: station.apex.x * station.weight,
            y: station.apex.y * station.weight,
            z: station.apex.z * station.weight,
            w: station.weight,
        });
    };
    for offset in (1..=degree).rev() {
        let (parameter, sample) = &global[count - offset];
        push(&sample.station, parameter - 1.0);
    }
    for (parameter, sample) in &global {
        push(&sample.station, *parameter);
    }
    // Close the loop: repeat the first station at parameter 1, then the
    // wrap continuation.
    push(&global[0].1.station, global[0].0 + 1.0);
    for offset in 1..=degree {
        let (parameter, sample) = &global[offset];
        push(&sample.station, parameter + 1.0);
    }
    let low = extended_params[0];
    let high = *extended_params.last().unwrap();
    let range = high - low;
    let normalized: Vec<f64> = extended_params
        .iter()
        .map(|parameter| (parameter - low) / range)
        .collect();
    let seam_low = (0.0 - low) / range;
    let seam_high = (1.0 - low) / range;
    let fit_row = |row: &[Vec4]| -> Result<NurbsCurve, String> {
        let curve = fit::interpolate_homogeneous(row, degree, &normalized)?;
        let (_, tail) = curve.split(seam_low)?;
        let (middle, _) = tail.split(seam_high)?;
        Ok(middle)
    };
    let cr = fit_row(&samples_cr)?;
    let cs = fit_row(&samples_cs)?;
    let mid = if chamfer {
        None
    } else {
        Some(fit_row(&samples_mid)?)
    };
    let u_domain = cr.domain()?;
    let surface = crate::blend::rows::surface_from_rows(degree, &cr, &cs, mid.as_ref(), true)?;

    // Single-face sides (e.g. one cap around the whole rim) get ONE uv
    // fit in the rows' own normalized space — identical parameterization
    // to cr/cs, so the closed support piece and its pcurve agree exactly.
    let single_face = |side: usize| -> bool {
        let first_id = segments[0].mate(side).face.id;
        segments
            .iter()
            .all(|segment| segment.mate(side).face.id == first_id)
    };
    let mut whole_pcurves: [Option<NurbsCurve>; 2] = [None, None];
    for side in 0..2 {
        if !single_face(side) {
            continue;
        }
        let mut row = Vec::new();
        let select = |station: &Station| -> [f64; 2] {
            if side == 0 {
                station.uv1
            } else {
                station.uv2
            }
        };
        let mut push_uv = |station: &Station| {
            let uv = select(station);
            row.push(Vec4::from_point(Vec3::new(uv[0], uv[1], 0.0), 1.0));
        };
        for offset in (1..=degree).rev() {
            push_uv(&global[count - offset].1.station);
        }
        for (_, sample) in &global {
            push_uv(&sample.station);
        }
        push_uv(&global[0].1.station);
        for offset in 1..=degree {
            push_uv(&global[offset].1.station);
        }
        whole_pcurves[side] = Some(fit_row(&row)?);
    }

    // Per-face pcurve fits over each segment's full sample window
    // (in-segment + overshoot), in global parameters.
    let mut pcurves1 = Vec::with_capacity(segments.len());
    let mut pcurves2 = Vec::with_capacity(segments.len());
    for segment in 0..segments.len() {
        let mut window: Vec<&ChainSample> = samples
            .iter()
            .filter(|sample| sample.segment == segment)
            .collect();
        window.sort_by(|a, b| a.parameter.total_cmp(&b.parameter));
        let params: Vec<f64> = window.iter().map(|sample| sample.parameter).collect();
        let low = params[0];
        let high = *params.last().unwrap();
        let normalized: Vec<f64> = params
            .iter()
            .map(|parameter| (parameter - low) / (high - low))
            .collect();
        let fit_uv = |select: &dyn Fn(&Station) -> [f64; 2]| -> Result<NurbsCurve, String> {
            let points: Vec<Vec4> = window
                .iter()
                .map(|sample| {
                    let uv = select(&sample.station);
                    Vec4::from_point(Vec3::new(uv[0], uv[1], 0.0), 1.0)
                })
                .collect();
            let curve = fit::interpolate_homogeneous(&points, degree, &normalized)?;
            // Re-express in GLOBAL parameters: the curve's [0,1] domain
            // corresponds to [low, high]; keep as-is and remember the
            // affine map through the window bounds.
            Ok(curve)
        };
        pcurves1.push((low, high, fit_uv(&|station| station.uv1)?));
        pcurves2.push((low, high, fit_uv(&|station| station.uv2)?));
    }

    chain_surgery(
        solid,
        segments,
        ChainRows {
            surface,
            cr,
            cs,
            u_domain,
            fit_low: low,
            fit_range: range,
            pcurves1,
            pcurves2,
            whole_pcurves,
        },
        name,
    )
}

pub(super) struct ChainRows {
    pub(super) surface: NurbsSurface,
    pub(super) cr: NurbsCurve,
    pub(super) cs: NurbsCurve,
    pub(super) u_domain: [f64; 2],
    /// Affine map from the rows' fit space to GLOBAL chain parameters:
    /// global = fit_low + p * fit_range (the global fit normalised its
    /// wrap-extended parameters onto [0, 1] before interpolating).
    pub(super) fit_low: f64,
    pub(super) fit_range: f64,
    /// Per-segment (window_low, window_high, uv curve over [0,1]) in
    /// GLOBAL parameters.
    pub(super) pcurves1: Vec<(f64, f64, NurbsCurve)>,
    pub(super) pcurves2: Vec<(f64, f64, NurbsCurve)>,
    /// Whole-chain uv fits (rows' fit space) for single-face sides.
    pub(super) whole_pcurves: [Option<NurbsCurve>; 2],
}

impl ChainRows {
    pub(super) fn to_global(&self, fit_parameter: f64) -> f64 {
        self.fit_low + fit_parameter * self.fit_range
    }
}

/// Extract the pcurve portion for a global window [a, b] from a
/// segment's fitted uv curve, re-parameterised so its domain maps
/// affinely onto the portion (matching the split support edge).
pub(super) fn pcurve_portion(fitted: &(f64, f64, NurbsCurve), a: f64, b: f64) -> Result<NurbsCurve, String> {
    let (low, high, curve) = fitted;
    // The chain parameterisation wraps with period 1; pick the period
    // copy of the window that overlaps this segment's fit range.
    let to_local = |value: f64| {
        let mut best = value;
        for candidate in [value, value + 1.0, value - 1.0] {
            if candidate >= low - 1e-9 && candidate <= high + 1e-9 {
                best = candidate;
                break;
            }
        }
        ((best - low) / (high - low)).clamp(0.0, 1.0)
    };
    let mut local_a = to_local(a);
    let mut local_b = to_local(b);
    if local_a > local_b {
        std::mem::swap(&mut local_a, &mut local_b);
    }
    let epsilon = 1e-9;
    let (_, tail) = if local_a > epsilon {
        curve.split(local_a)?
    } else {
        (curve.clone(), curve.clone())
    };
    let tail = if local_a > epsilon {
        tail
    } else {
        curve.clone()
    };
    let portion = if local_b < 1.0 - epsilon {
        tail.split(local_b)?.0
    } else {
        tail
    };
    Ok(portion)
}

/// The single boundary edge of a side's mate face(s) incident to a chain
/// vertex, EXCLUDING the two chain edges meeting there — i.e. the SEAM
/// (same face on both segments) or the SPOKE (face changes) that the
/// support curve must cross.  `None` when the chain simply flows through
/// (same face, no seam — e.g. a stadium cap rim vertex).
pub(super) fn cross_edge_at(
    solid: &BrepSolid,
    face_before: &FaceRecord,
    loop_before: usize,
    face_after: &FaceRecord,
    loop_after: usize,
    vertex: u64,
    before_edge: u64,
    after_edge: u64,
) -> Result<Option<u64>, String> {
    let mut found: Option<u64> = None;
    let mut consider = |face: &FaceRecord, loop_index: usize| -> Result<(), String> {
        for coedge in &face.loops[loop_index].coedges {
            if coedge.edge_id == before_edge || coedge.edge_id == after_edge {
                continue;
            }
            let edge = solid
                .edges
                .iter()
                .find(|edge| edge.id == coedge.edge_id)
                .ok_or("blend: loop references missing edge")?;
            if edge.start_vertex_id == vertex || edge.end_vertex_id == vertex {
                if found.is_some() && found != Some(edge.id) {
                    return Err("blend: multiple cross edges at a chain vertex".into());
                }
                found = Some(edge.id);
            }
        }
        Ok(())
    };
    consider(face_before, loop_before)?;
    if !(face_after.id == face_before.id && loop_after == loop_before) {
        consider(face_after, loop_after)?;
    }
    Ok(found)
}

/// The nearest periodic image of a seam boundary to `value` — the boundary a
/// crossing endpoint must be pinned to, read off its ADJACENT INTERIOR sample.
fn nearest_seam_image(value: f64, low: f64, high: f64) -> f64 {
    let period = high - low;
    low + ((value - low) / period).round() * period
}

/// The mate-face pcurve of a support PIECE, built by projecting the piece's
/// 3D curve onto the (analytic) mate surface and fitting.  Because a chain
/// support rides EXACTLY on its mate carrier, the projection is exact; the
/// periodic track is unwrapped so it never jumps a period across the carrier
/// seam, and crossing endpoints are pinned ONTO the seam so the piece and the
/// trimmed seam edge close the loop in parameter space.
///
/// The carrier is closed in EITHER direction, and the crossed seam is not
/// always the u meridian.  A cylinder/cone/sphere is closed in u only, so its
/// seam is the u0 ≡ u1 meridian — but a TORUS is biperiodic, and a tube welded
/// through a plane crosses the torus' v0 ≡ v1 PARALLEL (its outer equator)
/// instead.  Pinning u there drags the endpoint a half-carrier away from its
/// own 3D vertex (the collar loop then closes on a uv corner that is nowhere
/// near the vertex, and the mate face's trim is garbage), so the pinned AXIS is
/// chosen GEOMETRICALLY: whichever closed direction's seam actually passes
/// through the endpoint.  A carrier closed in u only can only ever take the u
/// branch, exactly as before.
pub(super) fn project_piece_pcurve(
    piece: &NurbsCurve,
    surface: &NurbsSurface,
    start_is_crossing: bool,
    end_is_crossing: bool,
) -> Result<NurbsCurve, String> {
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let (_, closed_v) = surface.closed_directions()?;
    let period = u1 - u0;
    let period_v = v1 - v0;
    let [d0, d1] = piece.domain()?;
    // Dense reference projection; the pcurve is fit from an adaptively
    // thinned subset kept as coarse as tolerance allows.
    const DENSE: usize = 96;
    let mut proj_u = Vec::with_capacity(DENSE + 1);
    let mut proj_v = Vec::with_capacity(DENSE + 1);
    let mut point3 = Vec::with_capacity(DENSE + 1);
    let mut params = Vec::with_capacity(DENSE + 1);
    let mut previous_u: Option<f64> = None;
    for section in 0..=DENSE {
        let t = d0 + (d1 - d0) * section as f64 / DENSE as f64;
        let point = piece.evaluate(t)?;
        let projection = crate::project_point_to_surface(surface, point)?;
        let mut u = projection.u;
        if let Some(previous) = previous_u {
            while u - previous > period * 0.5 {
                u -= period;
            }
            while previous - u > period * 0.5 {
                u += period;
            }
        }
        previous_u = Some(u);
        proj_u.push(u);
        proj_v.push(projection.v);
        point3.push(point);
        params.push(section as f64 / DENSE as f64);
    }
    // A biperiodic carrier needs the SAME unwrap in v, or a track that walks up
    // to the v seam comes back as 0 at the far end and the fit swings across
    // the whole carrier.  Anchor the chain on the first INTERIOR sample and
    // pull index 0 onto it afterwards: an endpoint sitting exactly ON the seam
    // inverts to either boundary at the solver's whim, and letting that
    // coin flip anchor the chain would shift the whole piece a period.
    if closed_v {
        for index in 2..=DENSE {
            while proj_v[index] - proj_v[index - 1] > period_v * 0.5 {
                proj_v[index] -= period_v;
            }
            while proj_v[index - 1] - proj_v[index] > period_v * 0.5 {
                proj_v[index] += period_v;
            }
        }
        while proj_v[0] - proj_v[1] > period_v * 0.5 {
            proj_v[0] -= period_v;
        }
        while proj_v[1] - proj_v[0] > period_v * 0.5 {
            proj_v[0] += period_v;
        }
    }
    // Interior mean decides which meridian a crossing endpoint sits on, and
    // whether the whole track needs a full-period shift into the domain.
    let mean_u: f64 = proj_u[1..DENSE].iter().sum::<f64>() / (DENSE - 1) as f64;
    let seam_u = if mean_u - u0 > u1 - mean_u { u1 } else { u0 };
    // Pin a crossing endpoint onto the seam it actually crosses.  Each
    // candidate keeps the OTHER coordinate as projected, so the losing axis
    // moves the endpoint bodily off its own 3D vertex — the residual names the
    // winner with no surface-type special-casing.  u keeps the whole-track
    // mean; a v run can legitimately cover a full period, so its boundary is
    // read off the ADJACENT INTERIOR sample instead.  Both pins land in the
    // track's own (pre-shift) period, so the shift below carries them along
    // with the rest of the track.
    let pinned = |index: usize,
                  neighbour: usize,
                  proj_u: &[f64],
                  proj_v: &[f64]|
     -> Result<(f64, f64), String> {
        let point = point3[index];
        let u_error = surface
            .evaluate(seam_u, proj_v[index])?
            .sub(point)
            .length();
        if closed_v {
            let seam_v = nearest_seam_image(proj_v[neighbour], v0, v1);
            let v_error = surface
                .evaluate(proj_u[index], seam_v)?
                .sub(point)
                .length();
            if v_error < u_error {
                return Ok((proj_u[index], seam_v));
            }
        }
        Ok((seam_u, proj_v[index]))
    };
    if start_is_crossing {
        let (u, v) = pinned(0, 1, &proj_u, &proj_v)?;
        proj_u[0] = u;
        proj_v[0] = v;
    }
    if end_is_crossing {
        let (u, v) = pinned(DENSE, DENSE - 1, &proj_u, &proj_v)?;
        proj_u[DENSE] = u;
        proj_v[DENSE] = v;
    }
    let mut shift = 0.0;
    while mean_u + shift > u1 + 1e-9 {
        shift -= period;
    }
    while mean_u + shift < u0 - 1e-9 {
        shift += period;
    }
    for u in proj_u.iter_mut() {
        *u += shift;
    }
    if closed_v {
        let mean_v: f64 = proj_v[1..DENSE].iter().sum::<f64>() / (DENSE - 1) as f64;
        let mut shift_v = 0.0;
        while mean_v + shift_v > v1 + 1e-9 {
            shift_v -= period_v;
        }
        while mean_v + shift_v < v0 - 1e-9 {
            shift_v += period_v;
        }
        for v in proj_v.iter_mut() {
            *v += shift_v;
        }
    }
    // Coarsest interpolation whose fitted pcurve stays inside HALF the
    // validator's pcurve/edge tolerance everywhere along the dense track.
    let tolerance = 0.002;
    let mut best: Option<NurbsCurve> = None;
    for sections in [8usize, 12, 16, 24, 32, 48, 64, 96] {
        if sections > DENSE {
            break;
        }
        let indices: Vec<usize> = (0..=sections)
            .map(|k| (k * DENSE / sections).min(DENSE))
            .collect();
        let points: Vec<Vec4> = indices
            .iter()
            .map(|&i| Vec4::from_point(Vec3::new(proj_u[i], proj_v[i], 0.0), 1.0))
            .collect();
        let knot_params: Vec<f64> = indices.iter().map(|&i| params[i]).collect();
        let curve = fit::interpolate_homogeneous(&points, FIT_DEGREE, &knot_params)?;
        let mut max_deviation: f64 = 0.0;
        for i in 0..=DENSE {
            let uv = curve.evaluate(params[i])?;
            let on_surface = surface.evaluate(uv.x, uv.y)?;
            max_deviation = max_deviation.max(on_surface.sub(point3[i]).length());
        }
        best = Some(curve);
        if max_deviation <= tolerance {
            break;
        }
    }
    best.ok_or_else(|| "blend: piece pcurve projection failed".to_string())
}
