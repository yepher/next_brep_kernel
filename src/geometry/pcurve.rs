use crate::{
    interpolate_curve, project_point_to_surface, project_point_to_surface_seeded, AnalyticSurface,
    KnotVector, NurbsCurve, NurbsSurface, Vec3, Vec4,
};

const EPSILON: f64 = 1e-12;
const LINEAR_TOLERANCE: f64 = 1e-7;

fn surface_domains(surface: &NurbsSurface) -> Result<([f64; 2], [f64; 2]), String> {
    Ok((
        KnotVector::new(surface.knots_u.clone(), surface.degree_u)?.domain(),
        KnotVector::new(surface.knots_v.clone(), surface.degree_v)?.domain(),
    ))
}

fn surface_closedness(surface: &NurbsSurface) -> Result<(bool, bool), String> {
    let ([u0, u1], [v0, v1]) = surface_domains(surface)?;
    let mut closed_u = true;
    let mut closed_v = true;
    for fraction in [0.19, 0.52, 0.87] {
        let v = v0 + (v1 - v0) * fraction;
        if surface
            .evaluate(u0, v)?
            .sub(surface.evaluate(u1, v)?)
            .length()
            > LINEAR_TOLERANCE * 10.0
        {
            closed_u = false;
        }
        let u = u0 + (u1 - u0) * fraction;
        if surface
            .evaluate(u, v0)?
            .sub(surface.evaluate(u, v1)?)
            .length()
            > LINEAR_TOLERANCE * 10.0
        {
            closed_v = false;
        }
    }
    Ok((closed_u, closed_v))
}

fn invert_checked(surface: &NurbsSurface, point: Vec3) -> Result<([f64; 2], f64), String> {
    if surface.is_affine()? {
        let ([u0, _], [v0, _]) = surface_domains(surface)?;
        let (origin, du, dv) = surface.deriv1(u0, v0)?;
        let delta = point.sub(origin);
        let uu = du.dot(du);
        let uv = du.dot(dv);
        let vv = dv.dot(dv);
        let along_u = delta.dot(du);
        let along_v = delta.dot(dv);
        let determinant = uu * vv - uv * uv;
        if determinant.abs() > EPSILON {
            return Ok((
                [
                    u0 + (along_u * vv - along_v * uv) / determinant,
                    v0 + (along_v * uu - along_u * uv) / determinant,
                ],
                0.0,
            ));
        }
    }
    let projection = project_point_to_surface(surface, point)?;
    Ok(([projection.u, projection.v], projection.distance))
}

fn unwrap_periodic(values: &mut [f64], minimum: f64, maximum: f64) {
    let period = maximum - minimum;
    for index in 1..values.len() {
        while values[index] - values[index - 1] > period / 2.0 {
            values[index] -= period;
        }
        while values[index] - values[index - 1] < -period / 2.0 {
            values[index] += period;
        }
    }
    if values.len() > 1 {
        while values[0] - values[1] > period / 2.0 {
            values[0] -= period;
        }
        while values[0] - values[1] < -period / 2.0 {
            values[0] += period;
        }
    }
    // Recenter the whole (now continuous) sequence into the domain by the
    // whole-period shift that leaves the LEAST parameter outside [minimum,
    // maximum]. A single middle/mean sample is NOT representative of a curve
    // that bulges out to touch — or straddle — the periodic seam: when that
    // sample is a boundary-kissing point (the at-seam projection unwraps a hair
    // past the boundary), a single-sample test misfires and shoves the entire
    // curve a full period out of range (ABC helmet 00000011/12 side channels:
    // a touching edge's parameter run snapped a whole period off its true
    // interior, tearing the loop open in parameter space).
    if values.len() > 1 {
        let excursion = |shift: f64| -> f64 {
            values
                .iter()
                .map(|value| {
                    let v = value + shift;
                    (minimum - v).max(0.0) + (v - maximum).max(0.0)
                })
                .sum::<f64>()
        };
        let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
        let center = 0.5 * (minimum + maximum);
        let base_k = ((center - mean) / period).round() as i64;
        let mut best_shift = 0.0;
        let mut best_excursion = f64::INFINITY;
        for k in (base_k - 1)..=(base_k + 1) {
            let shift = k as f64 * period;
            let value = excursion(shift);
            if value < best_excursion {
                best_excursion = value;
                best_shift = shift;
            }
        }
        if best_shift != 0.0 {
            for value in values.iter_mut() {
                *value += best_shift;
            }
        }
    }
}

fn build_interpolant(
    surface: &NurbsSurface,
    raw_parameters: &[[f64; 2]],
    curve_parameters: &[f64],
) -> Result<NurbsCurve, String> {
    let ([u0, u1], [v0, v1]) = surface_domains(surface)?;
    let (closed_u, closed_v) = surface_closedness(surface)?;
    let mut parameters = raw_parameters.to_vec();
    // Longitude is undefined at a sphere pole: the surface's u tangent
    // vanishes there, and closest-point inversion is free to return any u.
    // Letting that arbitrary value participate in periodic unwrapping can move
    // the first real meridian by a whole period (e.g. 0.75 -> -0.25). Analytic
    // carriers are subsequently clamped, collapsing that meridian onto u=0.
    //
    // Keep this deliberately sphere- and singularity-gated. Split the u values
    // into maximal non-pole runs, unwrap each run independently, and leave the
    // pole samples untouched. The existing interpolant degree stays unchanged;
    // lowering it here would also change endpoint-tangent decisions made later
    // while stitching the face loop.
    let sphere_u_singular = if matches!(surface.analytic(), Some(AnalyticSurface::Sphere { .. })) {
        let speeds = parameters
            .iter()
            .map(|parameter| {
                surface
                    .deriv1(parameter[0], parameter[1])
                    .map(|(_, su, _)| su.length())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reference = speeds.iter().copied().fold(0.0_f64, f64::max);
        speeds
            .into_iter()
            // Uniform edge stations need not hit the pole exactly (ABC 5360
            // bottoms out at v=5.5e-4). In this small conditioning band,
            // longitude is already numerically arbitrary and must not connect
            // the two meridian runs across the pole.
            .map(|speed| speed <= reference * 1e-2)
            .collect::<Vec<_>>()
    } else {
        vec![false; parameters.len()]
    };
    // Do not perturb ordinary pole-touching trims. The special path is needed
    // only when a pole band is bracketed by nonsingular samples and its raw
    // longitude changes by at least half a period: a true through-pole branch
    // reset whose two meridian runs must be unwrapped independently.
    // Prefix/suffix pole samples on normal sphere caps retain the legacy
    // interpolation byte-for-byte. Include the exactly-half-period case so
    // floating-point noise cannot decide which entire run is lifted (ABC 5603).
    let has_interior_pole_crossing = sphere_u_singular
        .iter()
        .position(|singular| !singular)
        .zip(sphere_u_singular.iter().rposition(|singular| !singular))
        .is_some_and(|(first, last)| sphere_u_singular[first..=last].iter().any(|value| *value));
    let period = (u1 - u0).abs();
    let interior_pole_branch_reset = closed_u
        && has_interior_pole_crossing
        && parameters.windows(2).zip(sphere_u_singular.windows(2)).any(
            |(parameter_pair, singular_pair)| {
                (singular_pair[0] || singular_pair[1])
                    && (parameter_pair[1][0] - parameter_pair[0][0]).abs()
                        >= 0.5 * period - 1e-12 * period.max(1.0)
            },
        );
    if closed_u {
        let mut values = parameters.iter().map(|value| value[0]).collect::<Vec<_>>();
        if interior_pole_branch_reset {
            let mut start = 0;
            while start < values.len() {
                while start < values.len() && sphere_u_singular[start] {
                    start += 1;
                }
                let mut end = start;
                while end < values.len() && !sphere_u_singular[end] {
                    end += 1;
                }
                unwrap_periodic(&mut values[start..end], u0, u1);
                start = end;
            }
        } else {
            unwrap_periodic(&mut values, u0, u1);
        }
        for (parameter, value) in parameters.iter_mut().zip(values) {
            parameter[0] = value;
        }
    }
    if closed_v {
        let mut values = parameters.iter().map(|value| value[1]).collect::<Vec<_>>();
        unwrap_periodic(&mut values, v0, v1);
        for (parameter, value) in parameters.iter_mut().zip(values) {
            parameter[1] = value;
        }
    }
    // A closed direction WRAPS: an edge that straddles the seam has no single
    // in-domain parameter run, so keep the unwrapped (possibly slightly
    // out-of-domain) values and let the periodic evaluator wrap them — clamping
    // them onto the seam boundary would collapse the straddling geometry (and,
    // via the interpolant, spike neighbouring stations). Restrict this to
    // GENERAL (B-spline) carriers: the analytic cylinders/cones/tori keep their
    // exact, in-domain seam handling (split rims, biperiodic bands), which the
    // straddle path is not meant to replace. Open directions must stay in-domain
    // regardless (their extension is a tangent-plane ruling, not a wrap).
    let wrap = surface.analytic().is_none();
    let points = parameters
        .into_iter()
        .map(|parameter| {
            let u = if closed_u && wrap {
                parameter[0]
            } else {
                parameter[0].clamp(u0, u1)
            };
            let v = if closed_v && wrap {
                parameter[1]
            } else {
                parameter[1].clamp(v0, v1)
            };
            Vec3::new(u, v, 0.0)
        })
        .collect::<Vec<_>>();
    interpolate_curve(&points, 3usize.min(points.len() - 1), curve_parameters)
}

/// Build a parameter-space curve for a 3D curve lying on a surface.
///
/// This mirrors the reference imprint implementation: affine patches map
/// homogeneous control points exactly; general patches use sampled inversion,
/// periodic seam unwrapping, and adaptive 3D residual refinement.
pub fn build_pcurve_on_surface(
    surface: &NurbsSurface,
    curve: &NurbsCurve,
) -> Result<NurbsCurve, String> {
    let [t0, t1] = curve.domain()?;
    if surface.is_affine()? {
        let ([u0, _], [v0, _]) = surface_domains(surface)?;
        let (origin, du, dv) = surface.deriv1(u0, v0)?;
        let uu = du.dot(du);
        let uv = du.dot(dv);
        let vv = dv.dot(dv);
        let determinant = uu * vv - uv * uv;
        if determinant.abs() <= 1e-18 {
            return Err("build_pcurve_on_surface: singular affine parameterization".into());
        }
        let control_points = curve
            .control_points
            .iter()
            .map(|control| {
                let delta = control.point()?.sub(origin);
                let along_u = delta.dot(du);
                let along_v = delta.dot(dv);
                Ok(Vec4::from_point(
                    Vec3::new(
                        u0 + (along_u * vv - along_v * uv) / determinant,
                        v0 + (along_v * uu - along_u * uv) / determinant,
                        0.0,
                    ),
                    control.w,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        return NurbsCurve::new(curve.degree, curve.knots.clone(), control_points);
    }

    let scale = 1.0 + curve.evaluate((t0 + t1) / 2.0)?.length();
    let drop_tolerance = 1e-4 * scale;
    let endpoint_tolerance = 1e-3 * scale;
    let refinement_tolerance = 1e-3f64.min(1e-4 * scale);
    let ([u0, u1], [v0, v1]) = surface_domains(surface)?;
    let invert =
        |fraction: f64| invert_checked(surface, curve.evaluate(t0 + (t1 - t0) * fraction)?);

    let mut parameters = Vec::with_capacity(25);
    let mut raw_surface_parameters = Vec::with_capacity(25);
    for index in 0..=24 {
        let fraction = index as f64 / 24.0;
        let (parameter, distance) = invert(fraction)?;
        if distance > drop_tolerance && index != 0 && index != 24 {
            continue;
        }
        if distance > endpoint_tolerance {
            if std::env::var("BREP_DEBUG_PCURVE").is_ok() {
                let p3 = curve.evaluate(t0 + (t1 - t0) * fraction).ok();
                let p0 = curve.evaluate(t0).ok();
                let p1 = curve.evaluate(t1).ok();
                eprintln!(
                    "PCURVE-FAIL idx={index} frac={fraction} dist={distance} endpt_tol={endpoint_tolerance} scale={scale}\n  curve3D@frac={p3:?}\n  curve3D@t0={p0:?} curve3D@t1={p1:?}\n  surf_domain=u[{u0},{u1}] v[{v0},{v1}] invpar={parameter:?}\n  surf@invpar={:?}",
                    surface.evaluate(parameter[0], parameter[1]).ok()
                );
            }
            return Err(format!(
                "build_pcurve_on_surface: endpoint projection failed (distance={distance})"
            ));
        }
        parameters.push(fraction);
        raw_surface_parameters.push(parameter);
    }

    let mut pcurve = build_interpolant(surface, &raw_surface_parameters, &parameters)?;
    const MAX_SAMPLES: usize = 160;
    for _ in 0..4 {
        if parameters.len() >= MAX_SAMPLES {
            break;
        }
        let mut inserts = Vec::new();
        for index in 0..parameters.len() - 1 {
            if parameters.len() + inserts.len() >= MAX_SAMPLES {
                break;
            }
            let start = parameters[index];
            let end = parameters[index + 1];
            if end - start < 1e-3 {
                continue;
            }
            for local_fraction in [0.25, 0.5, 0.75] {
                if parameters.len() + inserts.len() >= MAX_SAMPLES {
                    break;
                }
                let fraction = start + (end - start) * local_fraction;
                let parameter = pcurve.evaluate(fraction)?;
                let on_surface =
                    surface.evaluate(parameter.x.clamp(u0, u1), parameter.y.clamp(v0, v1))?;
                let on_curve = curve.evaluate(t0 + (t1 - t0) * fraction)?;
                if on_surface.sub(on_curve).length() <= refinement_tolerance {
                    continue;
                }
                let (surface_parameter, distance) = invert(fraction)?;
                if distance <= drop_tolerance {
                    inserts.push((index + 1, fraction, surface_parameter));
                }
            }
        }
        if inserts.is_empty() {
            break;
        }
        for (at, fraction, surface_parameter) in inserts.into_iter().rev() {
            parameters.insert(at, fraction);
            raw_surface_parameters.insert(at, surface_parameter);
        }
        pcurve = build_interpolant(surface, &raw_surface_parameters, &parameters)?;
    }
    Ok(pcurve)
}

/// SECTION-PCURVE SEEDED MARCH (t222: cone × ABC 00000327, non-integral genus).
///
/// `build_pcurve_on_surface` inverts every sample by GLOBAL closest point. Where
/// the carrier surface is COMPRESSED / self-overlapping along the row the
/// section rides, the global search aliases interior samples onto a DISTANT
/// preimage sheet, folding the pcurve out past the trim boundary and re-crossing
/// it. The arrangement then splits the shared section at that phantom crossing,
/// while the mate face (whose carrier is not folded there) keeps the section
/// whole — so the two operands' fragments disagree and the section strands
/// one-use (non-integral genus). Ground truth for t222: 00000327 face 496's
/// v≈0.7875 row maps u∈[0.55,0.81] into a ~0.18mm neighbourhood of the section
/// endpoint A; the global pcurve for piece 9 folded out to u=0.806 and back,
/// re-crossing trim edge 349 at v≈0.784, 0.003 below the corner vertex A.
///
/// This marches the interior inversions with each seeded from the PREVIOUS
/// accepted parameters (`project_point_to_surface_seeded`), so a marched sample
/// adopts the nearby branch instead of the momentarily-closest distant sheet.
/// The two ENDPOINT samples keep the deterministic global inversion (they are
/// the piece's shared junction vertices — same reasoning as
/// `repair_branch_jumps`). Additive + fail-soft: the marched chain is used ONLY
/// when it CLOSES onto the global far endpoint (its last interior sample is
/// parameter-adjacent to it, measured against the chain's own median step);
/// otherwise the plain global build is returned byte-for-byte. A single-preimage
/// carrier (the cone side of the same section) marches to the same samples the
/// global search already had, so it is unchanged there — this only ever alters a
/// carrier that actually presents multiple preimages within tolerance.
///
/// Escape hatch: `BREP_SECTION_PCURVE_MARCH=0`.
pub fn build_pcurve_on_surface_marched(
    surface: &NurbsSurface,
    curve: &NurbsCurve,
) -> Result<NurbsCurve, String> {
    let global = build_pcurve_on_surface(surface, curve)?;
    if std::env::var("BREP_SECTION_PCURVE_MARCH").as_deref() == Ok("0") || surface.is_affine()? {
        return Ok(global);
    }
    let [t0, t1] = curve.domain()?;
    // A closed (ring) section has no distinct endpoints to anchor the march;
    // leave it to the existing path (closed rings are handled elsewhere).
    if curve.evaluate(t0)?.sub(curve.evaluate(t1)?).length() <= 1e-6 {
        return Ok(global);
    }
    const SAMPLES: usize = 24;
    let fractions: Vec<f64> = (0..=SAMPLES).map(|i| i as f64 / SAMPLES as f64).collect();
    let edge: Vec<Vec3> = fractions
        .iter()
        .map(|fraction| curve.evaluate(t0 + (t1 - t0) * fraction))
        .collect::<Result<Vec<_>, _>>()?;
    let scale = 1.0 + curve.evaluate((t0 + t1) / 2.0)?.length();
    let endpoint_tolerance = 1e-3 * scale;
    let (first, first_distance) = invert_checked(surface, edge[0])?;
    let (last, last_distance) = invert_checked(surface, edge[SAMPLES])?;
    if first_distance > endpoint_tolerance || last_distance > endpoint_tolerance {
        return Ok(global);
    }
    let ([u0, u1], [v0, v1]) = surface_domains(surface)?;
    let u_span = (u1 - u0).abs().max(EPSILON);
    let v_span = (v1 - v0).abs().max(EPSILON);
    let (closed_u, closed_v) = surface_closedness(surface)?;
    let norm_step = |a: [f64; 2], b: [f64; 2]| -> f64 {
        let mut du = a[0] - b[0];
        if closed_u {
            while du > 0.5 * u_span {
                du -= u_span;
            }
            while du < -0.5 * u_span {
                du += u_span;
            }
        }
        let mut dv = a[1] - b[1];
        if closed_v {
            while dv > 0.5 * v_span {
                dv -= v_span;
            }
            while dv < -0.5 * v_span {
                dv += v_span;
            }
        }
        ((du / u_span).powi(2) + (dv / v_span).powi(2)).sqrt()
    };
    // Forward seeded march over the interior samples.
    let mut raw = vec![first];
    let mut previous = first;
    for index in 1..SAMPLES {
        let seeded =
            project_point_to_surface_seeded(surface, edge[index], previous[0], previous[1])?;
        let (_, global_distance) = invert_checked(surface, edge[index])?;
        // The seeded footpoint must still sit on the surface — as tight as the
        // fit band or the global answer already had it. If it fell off (the
        // seed led Newton into a valley), abandon the march and keep global.
        if seeded.distance > 4.0 * global_distance + endpoint_tolerance {
            return Ok(global);
        }
        previous = [seeded.u, seeded.v];
        raw.push(previous);
    }
    raw.push(last);
    // CLOSURE GATE: the last marched interior sample must be parameter-adjacent
    // to the global far endpoint — the march stayed on ONE branch the whole way.
    // Measure the endpoint gap against the chain's OWN median step so a genuine
    // long edge is not rejected while a sheet-hop (which leaves a big gap to the
    // endpoint) is.
    let mut steps: Vec<f64> = (1..raw.len())
        .map(|i| norm_step(raw[i], raw[i - 1]))
        .collect();
    let endpoint_gap = steps.pop().unwrap_or(0.0);
    steps.sort_by(f64::total_cmp);
    let median = steps
        .get(steps.len() / 2)
        .copied()
        .unwrap_or(0.0)
        .max(EPSILON);
    if endpoint_gap > (4.0 * median).max(0.05) {
        return Ok(global);
    }
    // Build the interpolant from the marched samples, then refine — re-projecting
    // each insert SEEDED from the interpolant (already on the marched branch), so
    // a mid-refinement global inversion cannot re-alias onto the far sheet.
    let mut parameters = fractions;
    let mut pcurve = build_interpolant(surface, &raw, &parameters)?;
    let refinement_tolerance = 1e-3f64.min(1e-4 * scale);
    const MAX_SAMPLES: usize = 160;
    for _ in 0..4 {
        if parameters.len() >= MAX_SAMPLES {
            break;
        }
        let mut inserts = Vec::new();
        for index in 0..parameters.len() - 1 {
            if parameters.len() + inserts.len() >= MAX_SAMPLES {
                break;
            }
            if parameters[index + 1] - parameters[index] < 1e-3 {
                continue;
            }
            for local in [0.25, 0.5, 0.75] {
                if parameters.len() + inserts.len() >= MAX_SAMPLES {
                    break;
                }
                let fraction =
                    parameters[index] + (parameters[index + 1] - parameters[index]) * local;
                let seed = pcurve.evaluate(fraction)?;
                let on_curve = curve.evaluate(t0 + (t1 - t0) * fraction)?;
                let on_surface = surface.evaluate(seed.x.clamp(u0, u1), seed.y.clamp(v0, v1))?;
                if on_surface.sub(on_curve).length() <= refinement_tolerance {
                    continue;
                }
                let seeded = project_point_to_surface_seeded(surface, on_curve, seed.x, seed.y)?;
                inserts.push((index + 1, fraction, [seeded.u, seeded.v]));
            }
        }
        if inserts.is_empty() {
            break;
        }
        for (at, fraction, uv) in inserts.into_iter().rev() {
            parameters.insert(at, fraction);
            raw.insert(at, uv);
        }
        pcurve = build_interpolant(surface, &raw, &parameters)?;
    }
    Ok(pcurve)
}

/// Build a pcurve for a represented subrange of a larger edge curve.  Unlike
/// trimming the curve's homogeneous control net, this samples only the
/// represented interval, which is essential when off-interval control points
/// do not lie on the target carrier.
pub fn build_pcurve_on_surface_range(
    surface: &NurbsSurface,
    curve: &NurbsCurve,
    edge_start: f64,
    edge_end: f64,
    forward: bool,
    tolerance: f64,
) -> Result<NurbsCurve, String> {
    build_pcurve_on_surface_range_dense(
        surface, curve, edge_start, edge_end, forward, tolerance, 64, 3, 513,
    )
}

fn align_collapsed_endpoints(surface: &NurbsSurface, raw: &mut [[f64; 2]]) -> Result<(), String> {
    let anchor = surface.control_points[0][0].point()?;
    let mut extent = 0.0_f64;
    for control in surface.control_points.iter().flatten() {
        extent = extent.max(control.point()?.sub(anchor).length());
    }
    // This is a geometric identity check, independent of the fitting allowance
    // and of world position. Positive rational weights keep the whole iso-curve
    // inside the convex hull of its Euclidean controls. Cap the size coupling
    // at the linear tolerance so a large patch cannot erase a thin finite row.
    let pole_band = (1e-10 * (1.0 + extent)).min(LINEAR_TOLERANCE);
    for (endpoint, neighbor) in [(0, 1), (raw.len() - 1, raw.len() - 2)] {
        let point = surface.evaluate(raw[endpoint][0], raw[endpoint][1])?;
        for axis in 0..2 {
            let mut candidate = raw[endpoint];
            candidate[axis] = raw[neighbor][axis];
            if candidate == raw[endpoint]
                || surface
                    .evaluate(candidate[0], candidate[1])?
                    .sub(point)
                    .length()
                    > pole_band
            {
                continue;
            }
            let row = if axis == 0 {
                surface.iso_curve_v(raw[endpoint][1])?
            } else {
                surface.iso_curve_u(raw[endpoint][0])?
            };
            let mut collapsed = true;
            for control in &row.control_points {
                if control.point()?.sub(point).length() > pole_band {
                    collapsed = false;
                    break;
                }
            }
            if collapsed {
                raw[endpoint] = candidate;
            }
        }
    }
    Ok(())
}

/// Re-seat any raw inversion sample that BRANCH-JUMPED — snapped to a distant
/// fold of a self-overlapping general carrier — back onto the branch traced by
/// its neighbours.
///
/// Each entry in `raw` is an INDEPENDENT global closest-point inversion of the
/// corresponding 3D edge sample. Every one is geometrically valid (on the
/// surface, at the edge), but they need not be CONTIGUOUS: where a rational
/// B-spline surface folds back over the small trimmed patch it carries, the
/// momentarily-closest fold flips from one sample to the next, so the raw
/// polygon zig-zags across the whole domain and self-crosses — and the region
/// its interpolant bounds no longer covers the true face (its surface flux can
/// exceed its own area). We anchor on the LONGEST run of mutually-continuous
/// samples — the fold the trimmed patch actually lies on, since a small face
/// lives on ONE fold and the jumped samples are the minority the global search
/// snapped elsewhere — then walk outward in both directions, re-seeding Newton
/// from the last good parameters and adopting the continuous footpoint whenever
/// it is geometrically as valid as the global one. Anchoring on consensus (not
/// blindly on sample 0, which can itself be an outlier that would drag the
/// whole edge onto the wrong branch and tear the loop open at its endpoint)
/// leaves continuous nonsingular fits unchanged. Certified collapsed endpoint
/// rows first adopt their neighbour's branch; a bad seed in the subsequent
/// jump repair cannot make a sample worse than the global search already had it.
fn repair_branch_jumps(
    surface: &NurbsSurface,
    edge_points: &[Vec3],
    raw: &mut [[f64; 2]],
    tolerance: f64,
) -> Result<(), String> {
    if raw.len() < 3 {
        return Ok(());
    }
    // Affine carriers invert linearly and exactly — no folds, nothing to chase.
    // Checked first so the common planar/affine face never touches the env.
    if surface.is_affine()? {
        return Ok(());
    }
    // Opt-out for A/B bisection of a STEP-import regression against this repair.
    if std::env::var("BREP_NO_PCURVE_REPAIR").is_ok() {
        return Ok(());
    }
    // A pole has no unique coordinate along its collapsed row. Continuing the
    // adjacent sample's branch is safe only when the ENTIRE row is collapsed;
    // coincident points across an ordinary seam or fold are not enough.
    align_collapsed_endpoints(surface, raw)?;
    let ([u0, u1], [v0, v1]) = surface_domains(surface)?;
    let u_span = (u1 - u0).abs().max(EPSILON);
    let v_span = (v1 - v0).abs().max(EPSILON);
    let (closed_u, closed_v) = surface_closedness(surface)?;
    // Normalised, seam-aware step between two parameter samples. Wrapping the
    // closed directions keeps a legitimate seam crossing SMALL, so the seam /
    // periodic-unwrap machinery elsewhere is never disturbed by this repair.
    let norm_step = |a: [f64; 2], b: [f64; 2]| -> f64 {
        let mut du = a[0] - b[0];
        if closed_u {
            while du > 0.5 * u_span {
                du -= u_span;
            }
            while du < -0.5 * u_span {
                du += u_span;
            }
        }
        let mut dv = a[1] - b[1];
        if closed_v {
            while dv > 0.5 * v_span {
                dv -= v_span;
            }
            while dv < -0.5 * v_span {
                dv += v_span;
            }
        }
        ((du / u_span).powi(2) + (dv / v_span).powi(2)).sqrt()
    };
    // A jump crosses a large fraction of the WHOLE domain — orders above the
    // per-sample motion of any real (even domain-spanning) edge, which advances
    // ~1/N of its traversal between consecutive stations.
    const JUMP_THRESHOLD: f64 = 0.2;
    let count = raw.len();

    // Locate the longest maximal run of consecutive continuous samples — the
    // branch to anchor on. If nothing jumps, this spans [0, count-1] and both
    // re-seat passes below are empty, leaving `raw` untouched.
    let (mut best_start, mut best_len, mut run_start) = (0usize, 1usize, 0usize);
    for index in 1..count {
        if norm_step(raw[index], raw[index - 1]) > JUMP_THRESHOLD {
            if index - run_start > best_len {
                best_len = index - run_start;
                best_start = run_start;
            }
            run_start = index;
        }
    }
    if count - run_start > best_len {
        best_len = count - run_start;
        best_start = run_start;
    }
    let (spine_lo, spine_hi) = (best_start, best_start + best_len - 1);

    // Adopt the continuous footpoint only when it (a) still sits on the surface
    // — as tight as the fit band or the global answer — and (b) genuinely
    // closes the jump rather than trading it for another. Returns the parameter
    // to carry forward as the next seed (the repaired one, or the untouched
    // global when no repair applies).
    let reseat = |edge_point: Vec3,
                  previous: [f64; 2],
                  global: [f64; 2]|
     -> Result<[f64; 2], String> {
        let global_step = norm_step(global, previous);
        if global_step <= JUMP_THRESHOLD {
            return Ok(global);
        }
        let seeded =
            project_point_to_surface_seeded(surface, edge_point, previous[0], previous[1])?;
        let candidate = [seeded.u, seeded.v];
        let global_residual = surface
            .evaluate(global[0], global[1])?
            .sub(edge_point)
            .length();
        let on_surface = seeded.distance <= 4.0 * global_residual + tolerance.max(1e-12);
        if on_surface && norm_step(candidate, previous) < 0.5 * global_step {
            if std::env::var("BREP_DEBUG_PCURVE").is_ok() {
                eprintln!(
                    "pcurve repair: jump {global_step:.4} ({global:?}) -> {:.4} ({candidate:?}) res {:.2e}->{:.2e}",
                    norm_step(candidate, previous),
                    global_residual,
                    seeded.distance
                );
            }
            Ok(candidate)
        } else {
            Ok(global)
        }
    };

    // Walk forward off the spine's high end, then backward off its low end,
    // chaining each repaired sample as the next seed so continuity propagates.
    //
    // Apart from the certified collapsed rows above, the two ENDPOINT samples
    // (fraction 0 and 1) are NEVER moved: they are the
    // edge's shared loop vertices. The global inversion is deterministic, so the
    // two coedges meeting at a vertex land it at the SAME parameters even when
    // that vertex sits on a fold reachable from two branches — moving one side
    // to a different branch tears the loop open there (`synthesize_pole_edge`
    // then rejects the non-collapsed gap). A vertex's genuine fold transition
    // (an edge whose interior rides u≈0.08 but whose endpoint must meet its
    // neighbour at u≈0.95) is exactly this case and must be preserved, not
    // "continuity-repaired" back onto the interior branch.
    let mut previous = raw[spine_hi];
    for index in (spine_hi + 1)..count.saturating_sub(1) {
        raw[index] = reseat(edge_points[index], previous, raw[index])?;
        previous = raw[index];
    }
    let mut previous = raw[spine_lo];
    for index in (1..spine_lo).rev() {
        raw[index] = reseat(edge_points[index], previous, raw[index])?;
        previous = raw[index];
    }
    Ok(())
}

/// Range fitter with explicit sampling knobs. The default entry above keeps
/// the long-standing (base 64, 3 refinement rounds, 513 cap) budget.
///
/// NOTE (2026-09-03): this comment used to say "STEP import retries failed fits
/// with a denser budget". No caller passes these knobs any more — that retry
/// was removed and the sentence outlived it. The knobs are kept because they
/// are the honest way to ASK whether a fit had headroom left, which is what
/// `examples/pcurve_residual_probe.rs` uses them for: on every ABC corpus
/// coedge that failed `validate`'s pcurve check, budgets up to (512, 8, 8193)
/// with a 1000x tighter target reproduced the coarse fit's deviation to six
/// decimals, because that deviation is the edge-vs-surface closest-point
/// residual and a curve ON the surface cannot beat it.
#[allow(clippy::too_many_arguments)]
pub fn build_pcurve_on_surface_range_dense(
    surface: &NurbsSurface,
    curve: &NurbsCurve,
    edge_start: f64,
    edge_end: f64,
    forward: bool,
    tolerance: f64,
    base_samples: usize,
    refinement_rounds: usize,
    parameter_cap: usize,
) -> Result<NurbsCurve, String> {
    if !(edge_start.is_finite() && edge_end.is_finite() && edge_start < edge_end) {
        return Err("build_pcurve_on_surface_range: invalid edge interval".into());
    }
    let evaluate_edge = |fraction: f64| {
        let edge_fraction = if forward { fraction } else { 1.0 - fraction };
        curve.evaluate(edge_start + (edge_end - edge_start) * edge_fraction)
    };
    let mut parameters = (0..=base_samples)
        .map(|index| index as f64 / base_samples as f64)
        .collect::<Vec<_>>();
    // The 3D edge point behind each raw inversion sample, kept parallel so the
    // continuity repair can re-seed a jumped station from its own footpoint.
    let mut edge_points = parameters
        .iter()
        .map(|fraction| evaluate_edge(*fraction))
        .collect::<Result<Vec<_>, _>>()?;
    let mut raw = edge_points
        .iter()
        .map(|point| invert_checked(surface, *point).map(|value| value.0))
        .collect::<Result<Vec<_>, _>>()?;
    repair_branch_jumps(surface, &edge_points, &mut raw, tolerance)?;
    let mut pcurve = build_interpolant(surface, &raw, &parameters)?;
    for _ in 0..refinement_rounds {
        let mut inserts = Vec::new();
        for index in 0..parameters.len() - 1 {
            if parameters.len() + inserts.len() >= parameter_cap {
                break;
            }
            for local in [0.25, 0.5, 0.75] {
                let fraction =
                    parameters[index] + (parameters[index + 1] - parameters[index]) * local;
                let uv = pcurve.evaluate(fraction)?;
                let represented = surface.evaluate(uv.x, uv.y)?;
                let edge_point = evaluate_edge(fraction)?;
                if represented.sub(edge_point).length() <= tolerance {
                    continue;
                }
                inserts.push((
                    index + 1,
                    fraction,
                    edge_point,
                    invert_checked(surface, edge_point)?.0,
                ));
            }
        }
        if inserts.is_empty() {
            break;
        }
        for (index, fraction, edge_point, value) in inserts.into_iter().rev() {
            parameters.insert(index, fraction);
            edge_points.insert(index, edge_point);
            raw.insert(index, value);
        }
        // Inserts are independent global inversions too — re-run the repair so
        // a fold-flip introduced mid-refinement cannot poison the next round's
        // deviation interpolant.
        repair_branch_jumps(surface, &edge_points, &mut raw, tolerance)?;
        pcurve = build_interpolant(surface, &raw, &parameters)?;
    }
    Ok(pcurve)
}

// BREP private tests: cf55fa85fe7032ca
