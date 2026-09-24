use crate::curve::interior_knot_count;
use crate::{project_point_to_surface, NurbsCurve, NurbsSurface, Vec3};
use serde::Serialize;

const EPSILON: f64 = 1e-12;

#[derive(Clone, Copy)]
struct Bounds {
    minimum: Vec3,
    maximum: Vec3,
}

impl Bounds {
    fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut bounds = Self {
            minimum: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            maximum: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        };
        for point in points {
            bounds.minimum.x = bounds.minimum.x.min(point.x);
            bounds.minimum.y = bounds.minimum.y.min(point.y);
            bounds.minimum.z = bounds.minimum.z.min(point.z);
            bounds.maximum.x = bounds.maximum.x.max(point.x);
            bounds.maximum.y = bounds.maximum.y.max(point.y);
            bounds.maximum.z = bounds.maximum.z.max(point.z);
        }
        bounds
    }

    fn expanded(self, amount: f64) -> Self {
        let delta = Vec3::new(amount, amount, amount);
        Self {
            minimum: self.minimum.sub(delta),
            maximum: self.maximum.add(delta),
        }
    }

    fn intersects(self, other: Self) -> bool {
        self.minimum.x <= other.maximum.x
            && self.maximum.x >= other.minimum.x
            && self.minimum.y <= other.maximum.y
            && self.maximum.y >= other.minimum.y
            && self.minimum.z <= other.maximum.z
            && self.maximum.z >= other.minimum.z
    }
}

fn surface_bounds(surface: &NurbsSurface) -> Result<Bounds, String> {
    Ok(Bounds::from_points(
        surface
            .control_points
            .iter()
            .flatten()
            .map(|point| point.point())
            .collect::<Result<Vec<_>, _>>()?,
    ))
}


fn fit_parameter(value: f64, minimum: f64, maximum: f64, closed: bool) -> f64 {
    if closed {
        (value - minimum).rem_euclid(maximum - minimum) + minimum
    } else {
        value.clamp(minimum, maximum)
    }
}

fn solve_three(matrix: [[f64; 3]; 3], rhs: Vec3) -> Option<[f64; 3]> {
    let a = matrix;
    let determinant = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let scale: f64 = a.iter().flatten().map(|value| value.abs()).sum();
    if determinant.abs() <= 1e-14 * scale.max(1.0).powi(3) {
        return None;
    }
    let x = (rhs.x * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (rhs.y * a[2][2] - a[1][2] * rhs.z)
        + a[0][2] * (rhs.y * a[2][1] - a[1][1] * rhs.z))
        / determinant;
    let y = (a[0][0] * (rhs.y * a[2][2] - a[1][2] * rhs.z)
        - rhs.x * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * rhs.z - rhs.y * a[2][0]))
        / determinant;
    let z = (a[0][0] * (a[1][1] * rhs.z - rhs.y * a[2][1])
        - a[0][1] * (a[1][0] * rhs.z - rhs.y * a[2][0])
        + rhs.x * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        / determinant;
    Some([x, y, z])
}

fn newton_curve_surface(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    seed: f64,
) -> Result<[f64; 3], String> {
    let [t0, t1] = curve.domain()?;
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let (closed_u, closed_v) = surface.closed_directions()?;
    let mut t = seed;
    let projection = project_point_to_surface(surface, curve.evaluate(t)?)?;
    let mut u = projection.u;
    let mut v = projection.v;
    for _ in 0..60 {
        let (curve_point, tangent) = curve.deriv1(t)?;
        let (surface_point, du, dv) = surface.deriv1(u, v)?;
        let residual = curve_point.sub(surface_point);
        if residual.length() <= EPSILON {
            return Ok([t, u, v]);
        }
        let Some([mut step_t, mut step_u, mut step_v]) = solve_three(
            [
                [tangent.x, -du.x, -dv.x],
                [tangent.y, -du.y, -dv.y],
                [tangent.z, -du.z, -dv.z],
            ],
            residual.scale(-1.0),
        ) else {
            if tangent.length_squared() <= EPSILON {
                return Ok([t, u, v]);
            }
            let next_t = (t - residual.dot(tangent) / tangent.length_squared()).clamp(t0, t1);
            if (next_t - t).abs() <= 1e-15 * (t1 - t0) {
                return Ok([t, u, v]);
            }
            t = next_t;
            let projection = project_point_to_surface(surface, curve.evaluate(t)?)?;
            u = projection.u;
            v = projection.v;
            continue;
        };
        step_t = step_t.clamp(-(t1 - t0) / 4.0, (t1 - t0) / 4.0);
        step_u = step_u.clamp(-(u1 - u0) / 4.0, (u1 - u0) / 4.0);
        step_v = step_v.clamp(-(v1 - v0) / 4.0, (v1 - v0) / 4.0);
        let next_t = (t + step_t).clamp(t0, t1);
        let next_u = fit_parameter(u + step_u, u0, u1, closed_u);
        let next_v = fit_parameter(v + step_v, v0, v1, closed_v);
        let stalled = (next_t - t).abs() <= 1e-15 * (t1 - t0)
            && (next_u - u).abs() <= 1e-15 * (u1 - u0)
            && (next_v - v).abs() <= 1e-15 * (v1 - v0);
        t = next_t;
        u = next_u;
        v = next_v;
        if stalled {
            return Ok([t, u, v]);
        }
    }
    Ok([t, u, v])
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CurveSurfaceIntersection {
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub point: Vec3,
    pub gap: f64,
    pub tangential: bool,
}

pub fn intersect_curve_surface(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    tolerance: f64,
) -> Result<Vec<CurveSurfaceIntersection>, String> {
    let [t0, t1] = curve.domain()?;
    let surface_box = surface_bounds(surface)?.expanded(tolerance * 10.0);
    let sample_count =
        32usize.max((interior_knot_count(&curve.knots, curve.degree) + 1) * (curve.degree + 1) * 4);
    let mut samples = Vec::with_capacity(sample_count + 1);
    for index in 0..=sample_count {
        let t = t0 + (t1 - t0) * index as f64 / sample_count as f64;
        samples.push((t, curve.evaluate(t)?));
    }
    let mut seeds = Vec::new();
    for pair in samples.windows(2) {
        let segment_box = Bounds::from_points([pair[0].1, pair[1].1]).expanded(tolerance * 10.0);
        if segment_box.intersects(surface_box) {
            seeds.push((pair[0].0 + pair[1].0) * 0.5);
        }
    }
    let mut results: Vec<CurveSurfaceIntersection> = Vec::new();
    for seed in seeds {
        let [t, u, v] = newton_curve_surface(curve, surface, seed)?;
        let on_curve = curve.evaluate(t)?;
        let on_surface = surface.evaluate(u, v)?;
        let gap = on_curve.sub(on_surface).length();
        if gap > tolerance {
            continue;
        }
        let tangent = curve.deriv1(t)?.1;
        let parameter_tolerance =
            ((tolerance / tangent.length().max(EPSILON)) * 10.0).max((t1 - t0) * 1e-9);
        if results.iter().any(|result| {
            (result.t - t).abs() <= parameter_tolerance
                || result.point.sub(on_curve).length() <= tolerance * 10.0
        }) {
            continue;
        }
        let normal = surface.normal(u, v).ok();
        let tangential = normal
            .and_then(|normal| {
                tangent
                    .normalized()
                    .ok()
                    .map(|unit| normal.dot(unit).abs() <= 1e-3)
            })
            .unwrap_or(false);
        results.push(CurveSurfaceIntersection {
            t,
            u,
            v,
            point: on_curve.add(on_surface).scale(0.5),
            gap,
            tangential,
        });
    }
    results.sort_by(|a, b| a.t.total_cmp(&b.t));
    Ok(results)
}

// BREP private tests: 60c495752367c789
