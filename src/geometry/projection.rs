use crate::curve::interior_knots;
use crate::{NurbsCurve, NurbsSurface, Vec3};
use serde::Serialize;

const EPSILON: f64 = 1e-12;
const LINEAR_TOLERANCE: f64 = 1e-7;
const MAX_NEWTON_ITERATIONS: usize = 50;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CurveProjection {
    pub u: f64,
    pub point: Vec3,
    pub distance: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SurfaceProjection {
    pub u: f64,
    pub v: f64,
    pub point: Vec3,
    pub distance: f64,
}

fn fit_parameter(value: f64, minimum: f64, maximum: f64, closed: bool) -> f64 {
    if closed {
        let period = maximum - minimum;
        (value - minimum).rem_euclid(period) + minimum
    } else {
        value.clamp(minimum, maximum)
    }
}

pub fn project_point_to_curve(curve: &NurbsCurve, point: Vec3) -> Result<CurveProjection, String> {
    let [start, end] = curve.domain()?;
    let mut breaks = vec![start];
    breaks.extend(interior_knots(&curve.knots, curve.degree));
    breaks.push(end);
    let samples_per_span = 4usize.max(curve.degree + 2);
    let mut best_u = start;
    let mut best_distance_squared = f64::INFINITY;
    for pair in breaks.windows(2) {
        for index in 0..=samples_per_span {
            let parameter = pair[0] + (pair[1] - pair[0]) * index as f64 / samples_per_span as f64;
            let distance_squared = curve.evaluate(parameter)?.sub(point).length_squared();
            if distance_squared < best_distance_squared {
                best_distance_squared = distance_squared;
                best_u = parameter;
            }
        }
    }

    let mut parameter = best_u;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        let derivatives = curve.derivatives_small(parameter, 2)?;
        let residual = derivatives[0].sub(point);
        let f = derivatives[1].dot(residual);
        let derivative = derivatives[2].dot(residual) + derivatives[1].length_squared();
        let distance_squared = residual.length_squared();
        if distance_squared < best_distance_squared {
            best_distance_squared = distance_squared;
            best_u = parameter;
        }
        let residual_length = distance_squared.sqrt();
        let tangent_length = derivatives[1].length();
        if residual_length <= LINEAR_TOLERANCE
            || f.abs() <= EPSILON + 1e-10 * tangent_length * residual_length
            || derivative.abs() <= EPSILON
        {
            break;
        }
        let mut step = -f / derivative;
        let maximum_step = (end - start) / 4.0;
        if step.abs() > maximum_step {
            step = step.signum() * maximum_step;
        }
        let next = (parameter + step).clamp(start, end);
        if (next - parameter).abs() <= 1e-15 * (end - start) {
            parameter = next;
            break;
        }
        parameter = next;
    }
    for candidate in [start, end, parameter] {
        let distance_squared = curve.evaluate(candidate)?.sub(point).length_squared();
        if distance_squared < best_distance_squared {
            best_distance_squared = distance_squared;
            best_u = candidate;
        }
    }
    let projected = curve.evaluate(best_u)?;
    Ok(CurveProjection {
        u: best_u,
        point: projected,
        distance: projected.sub(point).length(),
    })
}

fn project_linear_revolution(
    surface: &NurbsSurface,
    point: Vec3,
) -> Result<Option<SurfaceProjection>, String> {
    let rows = &surface.control_points;
    if surface.degree_u != 2
        || surface.degree_v != 1
        || rows.len() < 5
        || rows[0].len() != 2
        || !surface.closed_directions()?.0
    {
        return Ok(None);
    }
    let base = rows[0][0].point()?.add(rows[4][0].point()?).scale(0.5);
    let top = rows[0][1].point()?.add(rows[4][1].point()?).scale(0.5);
    let axis_vector = top.sub(base);
    let height = axis_vector.length();
    if height <= EPSILON {
        return Ok(None);
    }
    let axis = axis_vector.normalized()?;
    let [v0, v1] = surface.domain_v()?;
    let axial = point.sub(base).dot(axis).clamp(0.0, height);
    let v = v0 + (v1 - v0) * axial / height;
    let circle = surface.iso_curve_v(v)?;
    let projection = project_point_to_curve(&circle, point)?;
    let [u0, u1] = surface.domain_u()?;
    let u = projection.u.clamp(u0, u1);
    let projected = surface.evaluate(u, v)?;
    Ok(Some(SurfaceProjection {
        u,
        v,
        point: projected,
        distance: projected.sub(point).length(),
    }))
}

#[derive(Clone, Copy)]
struct NewtonResult {
    u: f64,
    v: f64,
    distance_squared: f64,
    converged: bool,
}

fn surface_newton(
    surface: &NurbsSurface,
    point: Vec3,
    seed_u: f64,
    seed_v: f64,
    domains: [f64; 4],
    closed_u: bool,
    closed_v: bool,
) -> Result<NewtonResult, String> {
    let [u0, u1, v0, v1] = domains;
    let mut u = seed_u;
    let mut v = seed_v;
    let mut converged = false;
    let mut best = NewtonResult {
        u,
        v,
        distance_squared: surface.evaluate(u, v)?.sub(point).length_squared(),
        converged,
    };
    for _ in 0..MAX_NEWTON_ITERATIONS {
        let derivatives = surface.derivatives_small(u, v, 2)?;
        let residual = derivatives[0][0].sub(point);
        let distance_squared = residual.length_squared();
        if distance_squared < best.distance_squared {
            best.u = u;
            best.v = v;
            best.distance_squared = distance_squared;
        }
        let f = derivatives[1][0].dot(residual);
        let g = derivatives[0][1].dot(residual);
        let residual_length = distance_squared.sqrt();
        if residual_length <= LINEAR_TOLERANCE {
            converged = true;
            break;
        }
        let tangent_u_length = derivatives[1][0].length();
        let tangent_v_length = derivatives[0][1].length();
        let cosine_u = if tangent_u_length * residual_length <= EPSILON {
            0.0
        } else {
            f.abs() / (tangent_u_length * residual_length)
        };
        let cosine_v = if tangent_v_length * residual_length <= EPSILON {
            0.0
        } else {
            g.abs() / (tangent_v_length * residual_length)
        };
        if cosine_u <= 1e-10 && cosine_v <= 1e-10 {
            converged = true;
            break;
        }
        let j00 = derivatives[2][0].dot(residual) + derivatives[1][0].length_squared();
        let j01 = derivatives[1][1].dot(residual) + derivatives[1][0].dot(derivatives[0][1]);
        let j11 = derivatives[0][2].dot(residual) + derivatives[0][1].length_squared();
        let determinant = j00 * j11 - j01 * j01;
        if determinant.abs() <= EPSILON {
            break;
        }
        let mut du = (-f * j11 + g * j01) / determinant;
        let mut dv = (-g * j00 + f * j01) / determinant;
        du = du.clamp(-(u1 - u0) / 4.0, (u1 - u0) / 4.0);
        dv = dv.clamp(-(v1 - v0) / 4.0, (v1 - v0) / 4.0);
        let next_u = fit_parameter(u + du, u0, u1, closed_u);
        let next_v = fit_parameter(v + dv, v0, v1, closed_v);
        let stalled =
            (next_u - u).abs() <= 1e-15 * (u1 - u0) && (next_v - v).abs() <= 1e-15 * (v1 - v0);
        u = next_u;
        v = next_v;
        if stalled {
            break;
        }
    }
    let final_distance_squared = surface.evaluate(u, v)?.sub(point).length_squared();
    if final_distance_squared < best.distance_squared {
        best.u = u;
        best.v = v;
        best.distance_squared = final_distance_squared;
    }
    best.converged = converged;
    Ok(best)
}

pub fn project_point_to_surface(
    surface: &NurbsSurface,
    point: Vec3,
) -> Result<SurfaceProjection, String> {
    // Recognized analytic carriers (plane / cylinder / cone / sphere /
    // torus) have exact closed-form projections in the surface's own
    // rational parameterization — skip grid seeding and Newton entirely.
    if let Some(analytic) = surface.analytic() {
        if let Some(projection) = analytic.project(surface, point) {
            return Ok(projection);
        }
    }
    project_point_to_surface_general(surface, point)
}

/// Project a point onto a surface starting Newton from an explicit `(u, v)`
/// guess, WITHOUT the global grid seed. This is a footpoint refiner for
/// continuity-preserving curve-on-surface tracing: seeding each edge sample
/// from its neighbour's parameters keeps the fit on ONE branch of a surface
/// that folds back over the small trimmed patch, where an independent global
/// search would snap to whichever fold is momentarily closest and tear the
/// pcurve into a self-crossing zig-zag. The caller compares the returned
/// `distance` against the global answer and only adopts this result when it is
/// geometrically just as valid, so a bad seed can never make a fit worse.
pub fn project_point_to_surface_seeded(
    surface: &NurbsSurface,
    point: Vec3,
    seed_u: f64,
    seed_v: f64,
) -> Result<SurfaceProjection, String> {
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let (closed_u, closed_v) = surface.closed_directions()?;
    let result = surface_newton(
        surface,
        point,
        seed_u,
        seed_v,
        [u0, u1, v0, v1],
        closed_u,
        closed_v,
    )?;
    let projected = surface.evaluate(result.u, result.v)?;
    Ok(SurfaceProjection {
        u: result.u,
        v: result.v,
        point: projected,
        distance: projected.sub(point).length(),
    })
}

/// The Newton seed grid for general projection: (u, v, point) samples over
/// every knot span. A pure function of the surface, cached on it — building
/// the grid costs hundreds of evaluations and projection is the hottest
/// entry point in the kernel.
fn projection_seed_grid(surface: &NurbsSurface) -> Result<&[(f64, f64, Vec3)], String> {
    if let Some(grid) = surface.projection_grid.get() {
        return Ok(grid);
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let (closed_u, closed_v) = surface.closed_directions()?;
    let mut breaks_u = vec![u0];
    breaks_u.extend(interior_knots(&surface.knots_u, surface.degree_u));
    breaks_u.push(u1);
    let mut breaks_v = vec![v0];
    breaks_v.extend(interior_knots(&surface.knots_v, surface.degree_v));
    breaks_v.push(v1);
    let samples_u = if closed_u {
        8usize.max(surface.degree_u * 4)
    } else {
        3usize.max(surface.degree_u + 1)
    };
    let samples_v = if closed_v {
        8usize.max(surface.degree_v * 4)
    } else {
        3usize.max(surface.degree_v + 1)
    };
    let mut grid = Vec::new();
    for u_pair in breaks_u.windows(2) {
        for v_pair in breaks_v.windows(2) {
            for i in 0..=samples_u {
                for j in 0..=samples_v {
                    let u = u_pair[0] + (u_pair[1] - u_pair[0]) * i as f64 / samples_u as f64;
                    let v = v_pair[0] + (v_pair[1] - v_pair[0]) * j as f64 / samples_v as f64;
                    grid.push((u, v, surface.evaluate(u, v)?));
                }
            }
        }
    }
    Ok(surface.projection_grid.get_or_init(|| grid))
}

/// The dense fallback grid for degenerate general projection: (u, v, point)
/// over a `(count_u+1) × (count_v+1)` lattice with `count_u/count_v` derived
/// from the surface's knot spans. A pure function of the surface, cached on
/// it — on metre-unit-mm parts the degenerate fallback fires ~1.7M times and
/// each re-evaluated all ~561 lattice points; caching them makes the argmin a
/// pure distance scan over reused points (bit-identical seed).
fn projection_dense_grid(surface: &NurbsSurface) -> Result<&[(f64, f64, Vec3)], String> {
    if let Some(grid) = surface.projection_dense_grid.get() {
        return Ok(grid);
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let mut breaks_u = vec![u0];
    breaks_u.extend(interior_knots(&surface.knots_u, surface.degree_u));
    breaks_u.push(u1);
    let mut breaks_v = vec![v0];
    breaks_v.extend(interior_knots(&surface.knots_v, surface.degree_v));
    breaks_v.push(v1);
    let count_u = 32usize.max((breaks_u.len() - 1) * 8);
    let count_v = 16usize.max((breaks_v.len() - 1) * 8);
    let mut grid = Vec::with_capacity((count_u + 1) * (count_v + 1));
    for i in 0..=count_u {
        for j in 0..=count_v {
            let u = u0 + (u1 - u0) * i as f64 / count_u as f64;
            let v = v0 + (v1 - v0) * j as f64 / count_v as f64;
            grid.push((u, v, surface.evaluate(u, v)?));
        }
    }
    Ok(surface.projection_dense_grid.get_or_init(|| grid))
}

// Boundary-ring cache geometry: 4 sides × 22 exponents × 33 indices. The scan
// positions depend only on the surface domain and these fixed integers, so the
// evaluated points are a pure function of the surface and cacheable.
const RING_EXPONENTS: usize = 22;
const RING_INDICES: usize = 33; // index 0..=32

#[inline]
fn ring_slot(side: usize, exponent_index: usize, index: usize) -> usize {
    (side * RING_EXPONENTS + exponent_index) * RING_INDICES + index
}

/// The boundary-ring sample points for the degenerate projection fallback,
/// flattened by `ring_slot(side, exponent-1, index)`. Sides: 0 = v-low
/// (`v = v0 + (v1-v0)·offset`), 1 = v-high (`v1 - (v1-v0)·offset`), 2 = u-low
/// (`u0 + (u1-u0)·offset`), 3 = u-high (`u1 - (u1-u0)·offset`), where
/// `offset = 1/2^exponent`. A pure function of the surface, cached on it — the
/// fallback's boundary ring re-evaluated these on every one of ~1.7M calls.
fn projection_ring_grid(surface: &NurbsSurface) -> Result<&[Vec3], String> {
    if let Some(grid) = surface.projection_ring_grid.get() {
        return Ok(grid);
    }
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let mut grid = vec![Vec3::default(); 4 * RING_EXPONENTS * RING_INDICES];
    for exponent in 1..=RING_EXPONENTS as i32 {
        let offset = 1.0 / 2f64.powi(exponent);
        let e = exponent as usize - 1;
        let v_lo = v0 + (v1 - v0) * offset;
        let v_hi = v1 - (v1 - v0) * offset;
        let u_lo = u0 + (u1 - u0) * offset;
        let u_hi = u1 - (u1 - u0) * offset;
        for index in 0..=32 {
            let u = u0 + (u1 - u0) * index as f64 / 32.0;
            grid[ring_slot(0, e, index as usize)] = surface.evaluate(u, v_lo)?;
            grid[ring_slot(1, e, index as usize)] = surface.evaluate(u, v_hi)?;
        }
        for index in 0..=32 {
            let v = v0 + (v1 - v0) * index as f64 / 32.0;
            grid[ring_slot(2, e, index as usize)] = surface.evaluate(u_lo, v)?;
            grid[ring_slot(3, e, index as usize)] = surface.evaluate(u_hi, v)?;
        }
    }
    Ok(surface.projection_ring_grid.get_or_init(|| grid))
}

pub(crate) fn project_point_to_surface_general(
    surface: &NurbsSurface,
    point: Vec3,
) -> Result<SurfaceProjection, String> {
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let (closed_u, closed_v) = surface.closed_directions()?;
    let mut best = NewtonResult {
        u: u0,
        v: v0,
        distance_squared: f64::INFINITY,
        converged: false,
    };
    for &(u, v, sample) in projection_seed_grid(surface)? {
        let distance_squared = sample.sub(point).length_squared();
        if distance_squared < best.distance_squared {
            best.u = u;
            best.v = v;
            best.distance_squared = distance_squared;
        }
    }
    if let Some(revolution) = project_linear_revolution(surface, point)? {
        if revolution.distance * revolution.distance < best.distance_squared {
            best.u = revolution.u;
            best.v = revolution.v;
            best.distance_squared = revolution.distance * revolution.distance;
        }
    }
    let first_newton = surface_newton(
        surface,
        point,
        best.u,
        best.v,
        [u0, u1, v0, v1],
        closed_u,
        closed_v,
    )?;
    if first_newton.distance_squared < best.distance_squared {
        best = first_newton;
    }
    let (_, su, sv) = surface.deriv1(best.u, best.v)?;
    let degenerate = su.cross(sv).length() <= 1e-5 * (1.0 + su.length() + sv.length());
    if degenerate || !first_newton.converged {
        let mut grid = best;
        for &(u, v, sample) in projection_dense_grid(surface)? {
            let distance_squared = sample.sub(point).length_squared();
            if distance_squared < grid.distance_squared {
                grid.u = u;
                grid.v = v;
                grid.distance_squared = distance_squared;
            }
        }
        if grid.distance_squared < best.distance_squared {
            best = grid;
        }
        let polished = surface_newton(
            surface,
            point,
            grid.u,
            grid.v,
            [u0, u1, v0, v1],
            closed_u,
            closed_v,
        )?;
        if polished.distance_squared < best.distance_squared {
            best = polished;
        }

        let (_, su, sv) = surface.deriv1(best.u, best.v)?;
        if su.cross(sv).length() <= 1e-5 * (1.0 + su.length() + sv.length()) {
            let ring_grid = projection_ring_grid(surface)?;
            let mut ring = best;
            // Read the pre-evaluated ring points at the exact same parameter
            // positions the live scans used (side/exponent/index → `ring_slot`),
            // preserving scan order and argmin tie-breaking for bit-identity.
            let scan = |side: usize,
                        exponent_index: usize,
                        along_u: bool,
                        fixed: f64,
                        candidate: &mut NewtonResult| {
                for index in 0..=32 {
                    let moving = if along_u {
                        u0 + (u1 - u0) * index as f64 / 32.0
                    } else {
                        v0 + (v1 - v0) * index as f64 / 32.0
                    };
                    let (u, v) = if along_u {
                        (moving, fixed)
                    } else {
                        (fixed, moving)
                    };
                    let distance_squared = ring_grid
                        [ring_slot(side, exponent_index, index as usize)]
                    .sub(point)
                    .length_squared();
                    if distance_squared < candidate.distance_squared {
                        candidate.u = u;
                        candidate.v = v;
                        candidate.distance_squared = distance_squared;
                    }
                }
            };
            for exponent in 1..=22 {
                let offset = 1.0 / 2f64.powi(exponent);
                let e = exponent as usize - 1;
                if (best.v - v0).abs() <= (v1 - v0) * 0.02 {
                    scan(0, e, true, v0 + (v1 - v0) * offset, &mut ring);
                }
                if (best.v - v1).abs() <= (v1 - v0) * 0.02 {
                    scan(1, e, true, v1 - (v1 - v0) * offset, &mut ring);
                }
                if (best.u - u0).abs() <= (u1 - u0) * 0.02 {
                    scan(2, e, false, u0 + (u1 - u0) * offset, &mut ring);
                }
                if (best.u - u1).abs() <= (u1 - u0) * 0.02 {
                    scan(3, e, false, u1 - (u1 - u0) * offset, &mut ring);
                }
            }
            if ring.distance_squared < best.distance_squared {
                best = ring;
            }
            let polished = surface_newton(
                surface,
                point,
                ring.u,
                ring.v,
                [u0, u1, v0, v1],
                closed_u,
                closed_v,
            )?;
            if polished.distance_squared < best.distance_squared {
                best = polished;
            }
        }
    }
    let projected = surface.evaluate(best.u, best.v)?;
    Ok(SurfaceProjection {
        u: best.u,
        v: best.v,
        point: projected,
        distance: projected.sub(point).length(),
    })
}

// BREP private tests: 142fd6502860c9b6
