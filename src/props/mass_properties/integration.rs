pub(super) use crate::curve::interior_knots;
use super::*;

pub(super) fn curve_breaks(curve: &NurbsCurve) -> Result<Vec<f64>, String> {
    let [start, end] = curve.domain()?;
    let mut result = vec![start];
    result.extend(interior_knots(&curve.knots, curve.degree));
    result.push(end);
    Ok(result)
}

pub fn parameter_space_area(face: &FaceRecord) -> Result<f64, String> {
    let mut area = 0.0;
    for loop_record in &face.loops {
        for coedge in &loop_record.coedges {
            for pair in curve_breaks(&coedge.pcurve)?.windows(2) {
                let half = (pair[1] - pair[0]) * 0.5;
                let middle = (pair[1] + pair[0]) * 0.5;
                for index in 0..GAUSS_X.len() {
                    let parameter = middle + half * GAUSS_X[index];
                    let (point, tangent) = coedge.pcurve.deriv1(parameter)?;
                    area +=
                        GAUSS_W[index] * half * 0.5 * (point.x * tangent.y - point.y * tangent.x);
                }
            }
        }
    }
    Ok(area)
}

pub(super) fn is_affine(surface: &NurbsSurface) -> Result<bool, String> {
    // Keep exact metric shortcuts consistent with geometry recognition.
    surface.is_affine()
}

pub(super) fn surface_breaks(surface: &NurbsSurface) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ku = crate::KnotVector::new(surface.knots_u.clone(), surface.degree_u)?;
    let kv = crate::KnotVector::new(surface.knots_v.clone(), surface.degree_v)?;
    let [u0, u1] = ku.domain();
    let [v0, v1] = kv.domain();
    let mut u = vec![u0];
    u.extend(interior_knots(&surface.knots_u, surface.degree_u));
    u.push(u1);
    let mut v = vec![v0];
    v.extend(interior_knots(&surface.knots_v, surface.degree_v));
    v.push(v1);
    Ok((u, v))
}

pub(super) fn integrand_value(kind: Integrand, point: Vec3, weighted_normal: Vec3) -> f64 {
    let (x, y, z) = (point.x, point.y, point.z);
    match kind {
        Integrand::Area => weighted_normal.length(),
        Integrand::Volume => point.dot(weighted_normal),
        Integrand::VolumeAbout(reference) => point.sub(reference).dot(weighted_normal),
        Integrand::MomentX => 0.5 * x * x * weighted_normal.x,
        Integrand::MomentY => 0.5 * y * y * weighted_normal.y,
        Integrand::MomentZ => 0.5 * z * z * weighted_normal.z,
        Integrand::SecondXX => x * x * x / 3.0 * weighted_normal.x,
        Integrand::SecondYY => y * y * y / 3.0 * weighted_normal.y,
        Integrand::SecondZZ => z * z * z / 3.0 * weighted_normal.z,
        Integrand::ProductXY => 0.5 * x * x * y * weighted_normal.x,
        Integrand::ProductXZ => 0.5 * x * x * z * weighted_normal.x,
        Integrand::ProductYZ => 0.5 * y * y * z * weighted_normal.y,
    }
}

pub(super) fn evaluate_integrand(face: &FaceRecord, u: f64, v: f64, kind: Integrand) -> Result<f64, String> {
    let (point, su, sv) = face.surface.deriv1(u, v)?;
    let sign = if face.same_sense { 1.0 } else { -1.0 };
    let weighted_normal = su.cross(sv).scale(sign);
    Ok(integrand_value(kind, point, weighted_normal))
}

pub(super) fn integrate_untrimmed(face: &FaceRecord, kind: Integrand) -> Result<f64, String> {
    let (u_breaks, v_breaks) = surface_breaks(&face.surface)?;
    let mut total = 0.0;
    for upair in u_breaks.windows(2) {
        let half_u = (upair[1] - upair[0]) * 0.5;
        let middle_u = (upair[1] + upair[0]) * 0.5;
        for vpair in v_breaks.windows(2) {
            let half_v = (vpair[1] - vpair[0]) * 0.5;
            let middle_v = (vpair[1] + vpair[0]) * 0.5;
            for i in 0..GAUSS_X.len() {
                for j in 0..GAUSS_X.len() {
                    total += GAUSS_W[i]
                        * GAUSS_W[j]
                        * half_u
                        * half_v
                        * evaluate_integrand(
                            face,
                            middle_u + half_u * GAUSS_X[i],
                            middle_v + half_v * GAUSS_X[j],
                            kind,
                        )?;
                }
            }
        }
    }
    Ok(total)
}

/// A bi-periodic band face (surface closed in both u and v, exactly two loops
/// each a full-wrap constant-cross-level rim) whose trim is NOT expressed with a
/// seam ruling — the fillet-torus bands OCC/STEP emit as two rim circles. Such a
/// face integrates to ZERO on the trimmed path (both loops are degenerate iso
/// lines in parameter space), so it is handled analytically over the seam-cut
/// rectangle instead. See [`biperiodic_band_range`].
#[derive(Clone, Copy)]
pub(super) struct BiBand {
    /// Which parameter is the periodic (full-wrap) one.
    p_is_u: bool,
    /// The two rim cross-levels, ascending (q_lo <= q_hi) in the cross param.
    q_lo: f64,
    q_hi: f64,
    /// True when the material band is the CROSS-SEAM COMPLEMENT of [q_lo, q_hi]
    /// rather than the between-rims strip (mirrors the tessellator's
    /// `close_periodic_trim_dir` orientation test).
    complement: bool,
}

/// Detect the bi-periodic band configuration of `face` and choose which of the
/// two regions its rims bound is material. The choice replicates the watertight
/// tessellator (`watertight_tessellation::close_periodic_trim_dir`): the
/// material is to the LEFT of the boundary traversal, so the between-rims strip
/// is kept unless the cross direction wraps AND the rim senses conclusively name
/// the complement. Returns None for anything that is not a clean two-rim band.
pub(super) fn biperiodic_band_range(face: &FaceRecord) -> Result<Option<BiBand>, String> {
    let surface = &face.surface;
    let (closed_u, closed_v) = surface.closed_directions()?;
    if !(closed_u && closed_v) {
        return Ok(None);
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let u_span = (u1 - u0).abs().max(1e-30);
    let v_span = (v1 - v0).abs().max(1e-30);
    // Sample each loop's pcurves (in coedge/traversal order) into (u,v) points.
    // A DEGENERATE loop — a pole/apex vertex whose whole trace collapses to one
    // parameter point (a fat fillet torus touches its outer equator at a single
    // seam point: ABC 00000039 faces 68/70) — is not a band boundary; drop it
    // and keep the two real rims. Anything else leaves the band ambiguous.
    let mut loop_points: Vec<Vec<[f64; 2]>> = Vec::with_capacity(2);
    for loop_record in &face.loops {
        let mut points = Vec::new();
        for coedge in &loop_record.coedges {
            let [d0, d1] = coedge.pcurve.domain()?;
            let samples = 12;
            for k in 0..=samples {
                let t = d0 + (d1 - d0) * k as f64 / samples as f64;
                let p = coedge.pcurve.evaluate(t)?;
                points.push([p.x, p.y]);
            }
        }
        if points.len() < 2 {
            return Ok(None);
        }
        let (mut umin, mut umax, mut vmin, mut vmax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for pt in &points {
            umin = umin.min(pt[0]);
            umax = umax.max(pt[0]);
            vmin = vmin.min(pt[1]);
            vmax = vmax.max(pt[1]);
        }
        if (umax - umin) <= 1e-3 * u_span && (vmax - vmin) <= 1e-3 * v_span {
            continue; // degenerate pole/apex loop
        }
        loop_points.push(points);
    }
    if loop_points.len() != 2 {
        return Ok(None);
    }
    for p_is_u in [true, false] {
        let (period, _p0, _p1) = if p_is_u {
            (u1 - u0, u0, u1)
        } else {
            (v1 - v0, v0, v1)
        };
        let (q_dom_lo, q_dom_hi) = if p_is_u { (v0, v1) } else { (u0, u1) };
        let q_extent = (q_dom_hi - q_dom_lo).abs().max(1e-30);
        if !(period > 0.0) {
            continue;
        }
        let coord = |pt: &[f64; 2]| -> (f64, f64) {
            if p_is_u {
                (pt[0], pt[1])
            } else {
                (pt[1], pt[0])
            }
        };
        // (level, winding-direction) for each rim; None if any loop is not a
        // clean full-wrap constant-cross-level rim in this p direction.
        let mut rings: Vec<(f64, i32)> = Vec::with_capacity(2);
        let mut clean = true;
        for points in &loop_points {
            let (mut pmin, mut pmax, mut qmin, mut qmax) = (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            );
            for pt in points {
                let (p, q) = coord(pt);
                pmin = pmin.min(p);
                pmax = pmax.max(p);
                qmin = qmin.min(q);
                qmax = qmax.max(q);
            }
            if (pmax - pmin) < 0.6 * period || (qmax - qmin) > 0.05 * q_extent {
                clean = false;
                break;
            }
            let mut net = 0.0;
            for pair in points.windows(2) {
                let mut delta = coord(&pair[1]).0 - coord(&pair[0]).0;
                if delta > 0.5 * period {
                    delta -= period;
                } else if delta < -0.5 * period {
                    delta += period;
                }
                net += delta;
            }
            let direction = if net > 0.25 * period {
                1
            } else if net < -0.25 * period {
                -1
            } else {
                0
            };
            rings.push((0.5 * (qmin + qmax), direction));
        }
        if !clean || rings.len() != 2 {
            continue;
        }
        rings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (q_lo, lower_dir) = rings[0];
        let (q_hi, upper_dir) = rings[1];
        let inconclusive = lower_dir == 0 || upper_dir == 0 || lower_dir == upper_dir;
        let between_rims_is_ccw_uv = if p_is_u { lower_dir > 0 } else { lower_dir < 0 };
        let complement = !inconclusive && between_rims_is_ccw_uv != face.same_sense;
        return Ok(Some(BiBand {
            p_is_u,
            q_lo,
            q_hi,
            complement,
        }));
    }
    Ok(None)
}

/// Gauss–Legendre integral of `kind` over the axis-aligned parameter rectangle
/// [u_lo,u_hi]×[v_lo,v_hi], subdividing at knot spans clipped to the rectangle.
pub(super) fn integrate_rectangle(
    face: &FaceRecord,
    u_lo: f64,
    u_hi: f64,
    v_lo: f64,
    v_hi: f64,
    kind: Integrand,
) -> Result<f64, String> {
    let (u_full, v_full) = surface_breaks(&face.surface)?;
    let clamp = |breaks: &[f64], lo: f64, hi: f64| -> Vec<f64> {
        let eps = 1e-9 * (hi - lo).abs().max(1e-30);
        let mut out = vec![lo];
        for &b in breaks {
            if b > lo + eps && b < hi - eps {
                out.push(b);
            }
        }
        out.push(hi);
        out
    };
    let u_breaks = clamp(&u_full, u_lo, u_hi);
    let v_breaks = clamp(&v_full, v_lo, v_hi);
    let mut total = 0.0;
    for upair in u_breaks.windows(2) {
        let half_u = (upair[1] - upair[0]) * 0.5;
        let middle_u = (upair[1] + upair[0]) * 0.5;
        for vpair in v_breaks.windows(2) {
            let half_v = (vpair[1] - vpair[0]) * 0.5;
            let middle_v = (vpair[1] + vpair[0]) * 0.5;
            for i in 0..GAUSS_X.len() {
                for j in 0..GAUSS_X.len() {
                    total += GAUSS_W[i]
                        * GAUSS_W[j]
                        * half_u
                        * half_v
                        * evaluate_integrand(
                            face,
                            middle_u + half_u * GAUSS_X[i],
                            middle_v + half_v * GAUSS_X[j],
                            kind,
                        )?;
                }
            }
        }
    }
    Ok(total)
}

/// Integrate each of `kinds` over a bi-periodic band face, or return None when
/// `face` is not such a band. The between-rims strip integrates over the full
/// periodic span × the cross-level strip; the complement is the full-domain
/// integral MINUS that strip (the two tile the closed cross period, so no
/// domain-extended evaluation is needed).
pub(super) fn biperiodic_band_integral(
    face: &FaceRecord,
    kinds: &[Integrand],
) -> Result<Option<Vec<f64>>, String> {
    let Some(band) = biperiodic_band_range(face)? else {
        return Ok(None);
    };
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let (u_lo, u_hi, v_lo, v_hi) = if band.p_is_u {
        (u0, u1, band.q_lo, band.q_hi)
    } else {
        (band.q_lo, band.q_hi, v0, v1)
    };
    let mut out = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        let strip = integrate_rectangle(face, u_lo, u_hi, v_lo, v_hi, kind)?;
        let value = if band.complement {
            integrate_untrimmed(face, kind)? - strip
        } else {
            strip
        };
        out.push(value);
    }
    Ok(Some(out))
}
