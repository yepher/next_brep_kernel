//! The two-edge miter (§6.9.6): two stripes meeting at a vertex whose third
//! edge stays sharp.
//!
//! # Why this is not a surface-surface intersection call
//!
//! The seam between the two blends starts at P, the corner ball's tangency
//! point with the SHARED face — and there the two blend surfaces are tangent
//! to each other (both carry the shared face's normal).  A normal-based SSI
//! refuses P as a seed and is ill-conditioned next to it, which is the
//! blend-blend tangency drift the boolean miter has to weld over.  And two
//! equal-radius canals whose spines cross intersect in TWO branches through
//! P; only the one running toward the sharp edge is the seam.
//!
//! So the seam is marched as a level set on the FIRST stripe's own domain.
//! A constant-radius blend is the radius-r canal of its centre path, so a
//! point of stripe 1 lies on stripe 2 exactly when its distance to stripe 2's
//! centre row is r.  Step the station parameter from P toward the vertex and
//! solve the section coordinate alone: P is exact by construction, the
//! pcurve on stripe 1 is the sample list itself, the branch is chosen by the
//! direction of the march, and the gradient is fine everywhere off P.
//!
//! The seam leaves the strips through one of the two OTHER rails.  When it
//! leaves both at once the exit lies on the sharp edge and the corner is
//! symmetric.  Otherwise the stripe it left first is done, and the other one
//! is closed by a CONNECTOR — its own section by the first stripe's other
//! mate face, from the exit point down to the sharp edge — which is a
//! transversal level set (distance to that face) and marches the same way.

use crate::topology::{EdgeRecord, FaceRecord};
use crate::Vec3;

use super::stations::*;

/// One stripe's part in a miter.
pub(super) struct MiterStripe<'a> {
    pub(super) rows: &'a FittedRows,
    /// True when the shared face is this stripe's FIRST mate (the cr row,
    /// z = 0 on the blend surface).
    pub(super) shared_first: bool,
    /// True when the miter vertex is at the `u_domain[0]` end of the rows.
    pub(super) at_start: bool,
    /// Row parameter of the corner ball's tangency point on the shared rail.
    pub(super) u_ball: f64,
    /// The carrier of this stripe's OTHER mate: the face the sibling's
    /// connector runs on when this stripe is the one the seam leaves first.
    pub(super) other_face: &'a FaceRecord,
}

impl MiterStripe<'_> {
    /// Section coordinate from `f` (0 on the shared rail, 1 on the other).
    fn z(&self, fraction: f64) -> f64 {
        if self.shared_first {
            fraction
        } else {
            1.0 - fraction
        }
    }

    fn fraction_of_z(&self, z: f64) -> f64 {
        if self.shared_first {
            z
        } else {
            1.0 - z
        }
    }

    /// The row parameter at the vertex's end of the marched rows.
    fn u_end(&self) -> f64 {
        if self.at_start {
            self.rows.u_domain[0]
        } else {
            self.rows.u_domain[1]
        }
    }

    /// The row parameter of this stripe's own end vertex — where the seam is
    /// headed from the ball's station.
    fn u_vertex(&self) -> f64 {
        match self.rows.vertex_stations {
            Some(stations) => stations[usize::from(!self.at_start)],
            None => self.u_end(),
        }
    }

    /// The end of the marched rows on the side the seam travels toward.
    ///
    /// A CONVEX corner seats its ball inside the material, so P sits on the
    /// shared rail INSIDE the edge's own span and the seam runs outward, past
    /// the vertex, to the overshot end — `u_end`.  A CONCAVE one seats it in
    /// the air PAST the vertex (the two rails of a boss's base cross one
    /// radius beyond the corner), so P is out in the overshoot and the seam
    /// runs the other way, back into the span.  Both are "toward the vertex";
    /// only the row end that lies beyond it differs.
    fn u_limit(&self) -> f64 {
        if self.u_vertex() >= self.u_ball {
            self.rows.u_domain[1]
        } else {
            self.rows.u_domain[0]
        }
    }

    fn point(&self, u: f64, fraction: f64) -> Result<Vec3, String> {
        self.rows.surface.evaluate_extended(u, self.z(fraction))
    }

    fn center(&self) -> Result<&crate::NurbsCurve, String> {
        self.rows
            .center
            .as_ref()
            .ok_or_else(|| "miter: stripe rows carry no centre path".to_string())
    }
}

/// A marched curve with its images on the two surfaces it lies on.
pub(super) struct MarchedCurve {
    pub(super) points: Vec<Vec3>,
    /// (u, z) on the first surface, per sample.
    pub(super) uv_first: Vec<[f64; 2]>,
    /// (u, z) on the second surface, per sample.
    pub(super) uv_second: Vec<[f64; 2]>,
}

/// How the seam closes the corner.
pub(super) enum MiterClosure {
    /// The seam leaves both strips at one point, on the sharp edge.
    Symmetric {
        /// From P to the exit; images on stripe 0 then stripe 1.
        seam: MarchedCurve,
        /// Row parameter on each stripe where its OTHER rail stops.
        exit: [f64; 2],
        /// Where the exit lands on the sharp edge.
        sharp_parameter: f64,
    },
    /// The seam leaves stripe `leader` first.  The other stripe is closed by
    /// a connector: its section by the leader's other mate face, from the
    /// seam's exit to the sharp edge.
    Asymmetric {
        seam: MarchedCurve,
        leader: usize,
        /// Row parameter on the leader where its other rail stops.
        exit_leader: f64,
        /// The connector, images on the FOLLOWER's blend then on the
        /// leader's other mate face.
        connector: MarchedCurve,
        /// Row parameter on the follower where its other rail stops.
        exit_follower: f64,
        sharp_parameter: f64,
    },
}

/// Bisection to a relative bar on an interval with a sign change.
fn bisect(
    mut low: f64,
    mut high: f64,
    mut f_low: f64,
    f: &mut dyn FnMut(f64) -> Result<f64, String>,
    bar: f64,
) -> Result<f64, String> {
    for _ in 0..80 {
        let mid = (low + high) * 0.5;
        if (high - low).abs() <= bar {
            return Ok(mid);
        }
        let f_mid = f(mid)?;
        if (f_mid > 0.0) == (f_low > 0.0) {
            low = mid;
            f_low = f_mid;
        } else {
            high = mid;
        }
    }
    Ok((low + high) * 0.5)
}

/// Seam steps between the ball station and the vertex's end of the rows, to
/// find where the seam leaves the strips.
const SEAM_STEPS: usize = 48;
/// Samples over the seam's own length once its exit is known.
const SEAM_SAMPLES: usize = 64;

/// March the seam and decide how the corner closes.
///
/// `scale` is the model scale for the tolerance bars; `sharp` the vertex's
/// unselected third edge.
pub(super) fn solve_miter(
    stripes: [&MiterStripe<'_>; 2],
    radius: f64,
    sharp: &EdgeRecord,
    scale: f64,
) -> Result<MiterClosure, String> {
    let [first, second] = stripes;
    let bar = 1e-11 * (1.0 + scale);
    let center_second = second.center()?;
    // Distance-to-the-sibling's-spine level: zero where a point of stripe 0
    // lies on stripe 1.
    let level = |u: f64, fraction: f64| -> Result<f64, String> {
        let point = first.point(u, fraction)?;
        Ok(crate::project_point_to_curve(center_second, point)?.distance - radius)
    };
    // Where a point lands on the second stripe, as (u, fraction from the
    // shared rail, how far the projection had to clamp to get there).  A seam
    // point is ON stripe 1's canal by construction, so a non-zero distance is
    // the strip's own boundary: the point has run PAST a rail and the
    // projection clamped it back onto one.
    let on_second = |point: Vec3| -> Result<([f64; 2], f64, f64), String> {
        let projection = crate::project_point_to_surface(&second.rows.surface, point)?;
        Ok((
            [projection.u, projection.v],
            second.fraction_of_z(projection.v),
            projection.distance,
        ))
    };
    // Signed measure of stripe 1's far rail: negative while the seam is still
    // on the strip, zero on the rail, positive once it has run off.  The
    // fraction alone cannot say — it CLAMPS at 1 — so past the rail the
    // measure is the distance the projection clamped by.
    let past_second = |point: Vec3| -> Result<f64, String> {
        let (_, fraction, distance) = on_second(point)?;
        Ok(if fraction < 1.0 { fraction - 1.0 } else { distance })
    };

    let u_start = first.u_ball;
    let u_end = first.u_limit();
    let span = u_end - u_start;
    if !(span.abs() > 1e-9) {
        return Err("miter: the ball station sits at the vertex end of the rows".into());
    }
    let du = span / SEAM_STEPS as f64;

    let mut points = vec![first.point(u_start, 0.0)?];
    let mut uv_first = vec![[u_start, first.z(0.0)]];
    let mut uv_second = vec![[second.u_ball, second.z(0.0)]];

    // One station of the seam: the fraction where the level crosses zero, or
    // `None` when the level is non-negative on the whole other-rail end (the
    // seam has already left this strip).
    let station = |u: f64| -> Result<Option<(f64, Vec3)>, String> {
        let at_far = level(u, 1.0)?;
        if at_far >= 0.0 {
            return Ok(None);
        }
        let at_near = level(u, 0.0)?;
        if at_near <= 0.0 {
            return Err("miter: the shared rail re-enters the sibling blend".into());
        }
        let fraction = bisect(0.0, 1.0, at_near, &mut |f| level(u, f), 1e-13)?;
        Ok(Some((fraction, first.point(u, fraction)?)))
    };
    // The seam's own point at `u`, or — once stripe 0 has been left — the far
    // rail point the seam ran out through, so both exit measures stay defined
    // (and continuous) across each other's crossing.
    let seam_point = |u: f64| -> Result<Vec3, String> {
        Ok(match station(u)? {
            Some((_, point)) => point,
            None => first.point(u, 1.0)?,
        })
    };

    let mut previous_u = u_start;
    let mut exit: Option<(usize, f64)> = None; // (stripe that exits, u on stripe 0)
    for step in 1..=SEAM_STEPS + 8 {
        let u = u_start + du * step as f64;
        // Past the marched rows: refuse rather than extrapolate.
        if (u - u_end) * span.signum() > du.abs() * 8.0 {
            break;
        }
        let station_here = station(u)?;
        let left_first = station_here.is_none();
        let left_second = past_second(seam_point(u)?)? >= 0.0;
        if let (false, false, Some((fraction, point))) = (left_first, left_second, station_here) {
            let (uv2, _, _) = on_second(point)?;
            points.push(point);
            uv_first.push([u, first.z(fraction)]);
            uv_second.push(uv2);
            previous_u = u;
            continue;
        }
        // One or both strips were left inside THIS step — on an asymmetric
        // corner the two rails reach the sharp edge a fraction of a step
        // apart, so whichever was left first is the one the corner closes on.
        // Bisecting only the one the step happened to notice picks the wrong
        // leader and hands the connector a section it has already run past.
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        if left_first {
            let mut predicate =
                |candidate: f64| -> Result<f64, String> { level(candidate, 1.0) };
            let f_low = predicate(previous_u)?;
            candidates.push((0, bisect(previous_u, u, f_low, &mut predicate, bar)?));
        }
        if left_second {
            let mut predicate =
                |candidate: f64| -> Result<f64, String> { past_second(seam_point(candidate)?) };
            let f_low = predicate(previous_u)?;
            candidates.push((1, bisect(previous_u, u, f_low, &mut predicate, bar)?));
        }
        exit = candidates
            .into_iter()
            .min_by(|a, b| (a.1 - u_start).abs().total_cmp(&(b.1 - u_start).abs()));
        break;
    }
    let Some((leaves_first, u_exit)) = exit else {
        return Err(
            "miter: the seam does not leave either blend within the marched rows — the \
             radius is too large for this corner"
                .into(),
        );
    };
    // The exit is known: resample the seam densely between P and the exit so
    // the fitted edge rides both blends between samples too.  The coarse
    // march above stepped the whole row span and put only a dozen samples on
    // a seam that leaves within a fifth of it; a cubic through those bows
    // ~1e-4 off the cylinders at r = 1.  Sixty-four samples over the seam's
    // own length bring that to the fit's noise.
    let mut points = vec![first.point(u_start, 0.0)?];
    let mut uv_first = vec![[u_start, first.z(0.0)]];
    let mut uv_second = vec![[second.u_ball, second.z(0.0)]];
    for step in 1..SEAM_SAMPLES {
        let u = u_start + (u_exit - u_start) * step as f64 / SEAM_SAMPLES as f64;
        let Some((fraction, point)) = station(u)? else {
            return Err("miter: the seam vanished while being resampled".into());
        };
        let (uv2, _, _) = on_second(point)?;
        points.push(point);
        uv_first.push([u, first.z(fraction)]);
        uv_second.push(uv2);
    }

    // The exit point, and whether the other stripe leaves there too.
    let (exit_point, exit_uv_first, exit_uv_second, exit_u_second) = if leaves_first == 0 {
        let point = first.point(u_exit, 1.0)?;
        let (uv2, _, _) = on_second(point)?;
        (point, [u_exit, first.z(1.0)], uv2, uv2[0])
    } else {
        let (fraction, point) = station(u_exit)?
            .ok_or("miter: the exit station lost the seam")?;
        let (uv2, _, _) = on_second(point)?;
        (point, [u_exit, first.z(fraction)], [uv2[0], second.z(1.0)], uv2[0])
    };
    points.push(exit_point);
    uv_first.push(exit_uv_first);
    uv_second.push(exit_uv_second);
    let seam = MarchedCurve {
        points,
        uv_first,
        uv_second,
    };

    let on_sharp = crate::project_point_to_curve(&sharp.curve, exit_point)?;
    let symmetric_bar = 1e-6 * (1.0 + radius);
    if on_sharp.distance <= symmetric_bar {
        return Ok(MiterClosure::Symmetric {
            seam,
            exit: [u_exit, exit_u_second],
            sharp_parameter: on_sharp.u,
        });
    }

    // Asymmetric: the follower is closed by its section on the leader's
    // other mate face, from the exit point to the sharp edge.
    let (leader, follower) = if leaves_first == 0 {
        (first, second)
    } else {
        (second, first)
    };
    let (follower_uv_at_exit, follower_fraction_at_exit) = if leaves_first == 0 {
        (exit_uv_second, second.fraction_of_z(exit_uv_second[1]))
    } else {
        (exit_uv_first, first.fraction_of_z(exit_uv_first[1]))
    };
    let connector = march_connector(
        follower,
        leader.other_face,
        follower_uv_at_exit[0],
        follower_fraction_at_exit,
        exit_point,
        scale,
    )?;
    let connector_end = *connector
        .points
        .last()
        .ok_or("miter: the connector has no samples")?;
    let on_sharp = crate::project_point_to_curve(&sharp.curve, connector_end)?;
    if on_sharp.distance > 1e-5 * (1.0 + radius) {
        return Err(format!(
            "miter: the connector ends {:.3e} off the sharp edge",
            on_sharp.distance
        ));
    }
    let exit_follower = connector.uv_first.last().map(|uv| uv[0]).unwrap_or(0.0);
    Ok(MiterClosure::Asymmetric {
        seam,
        leader: leaves_first,
        // The seam is marched on stripe 0's domain, so `u_exit` is stripe 0's
        // row parameter.  The caller trims the LEADER's rail at this station,
        // and when the leader is stripe 1 that is the exit's parameter on
        // stripe 1's own rows.
        exit_leader: if leaves_first == 0 { u_exit } else { exit_u_second },
        connector,
        exit_follower,
        sharp_parameter: on_sharp.u,
    })
}

/// Connector steps from the seam's exit to the sharp edge.
const CONNECTOR_STEPS: usize = 24;

/// The follower's section by `face`, from `(u0, fraction0)` (the seam's exit,
/// already on the face) to the follower's other rail.  Parameterised by the
/// section fraction, solving the station parameter at each — the section by
/// a face through the vertex is close to one station of the follower, so the
/// station parameter is the unknown, not the fraction.
fn march_connector(
    follower: &MiterStripe<'_>,
    face: &FaceRecord,
    u0: f64,
    fraction0: f64,
    start: Vec3,
    scale: f64,
) -> Result<MarchedCurve, String> {
    let bar = 1e-11 * (1.0 + scale);
    let signed_distance = |point: Vec3| -> Result<f64, String> {
        let projection = crate::project_point_to_surface(&face.surface, point)?;
        let normal = raw_normal(&face.surface, projection.u, projection.v)?;
        Ok(point.sub(projection.point).dot(normal))
    };
    // Bracket the station parameter around the exit, on the side of the
    // marched rows that reaches the vertex.
    let u_end = follower.u_limit();
    let mut points = vec![start];
    let mut uv_follower = vec![[u0, follower.z(fraction0)]];
    let first_projection = crate::project_point_to_surface(&face.surface, start)?;
    let mut uv_face = vec![[first_projection.u, first_projection.v]];
    let mut previous_u = u0;
    for step in 1..=CONNECTOR_STEPS {
        let fraction = fraction0 + (1.0 - fraction0) * step as f64 / CONNECTOR_STEPS as f64;
        // Signed distance to the face along the station parameter at this
        // fraction; bracket from the previous station toward the row end and
        // back, whichever side changes sign.
        let mut g = |u: f64| -> Result<f64, String> {
            signed_distance(follower.point(u, fraction)?)
        };
        let g_prev = g(previous_u)?;
        let reach = (u_end - previous_u).abs().max(1e-9);
        let mut found = None;
        for direction in [(u_end - previous_u).signum(), -(u_end - previous_u).signum()] {
            let mut width = reach * 0.05;
            for _ in 0..8 {
                let candidate = previous_u + direction * width;
                let g_candidate = g(candidate)?;
                if (g_candidate > 0.0) != (g_prev > 0.0) {
                    let (low, high, f_low) = if candidate > previous_u {
                        (previous_u, candidate, g_prev)
                    } else {
                        (candidate, previous_u, g_candidate)
                    };
                    found = Some(bisect(low, high, f_low, &mut g, bar)?);
                    break;
                }
                width *= 2.0;
            }
            if found.is_some() {
                break;
            }
        }
        let u = match found {
            Some(u) => u,
            None if g_prev.abs() <= 1e-9 * (1.0 + scale) => previous_u,
            None => {
                return Err("miter: the connector left the face without crossing it".into())
            }
        };
        let point = follower.point(u, fraction)?;
        let projection = crate::project_point_to_surface(&face.surface, point)?;
        points.push(point);
        uv_follower.push([u, follower.z(fraction)]);
        uv_face.push([projection.u, projection.v]);
        previous_u = u;
    }
    Ok(MarchedCurve {
        points,
        uv_first: uv_follower,
        uv_second: uv_face,
    })
}

/// Fit a marched curve and its two images with ONE chord-length
/// parameterisation, so the pcurves reproduce the 3D curve parameter for
/// parameter.  Returns (curve, first image, second image).
pub(super) fn fit_marched(
    curve: &MarchedCurve,
) -> Result<(crate::NurbsCurve, crate::NurbsCurve, crate::NurbsCurve), String> {
    use crate::Vec4;
    let count = curve.points.len();
    if count < 2 {
        return Err("miter: a marched curve needs at least two samples".into());
    }
    let mut parameters = Vec::with_capacity(count);
    let mut accumulated = 0.0;
    parameters.push(0.0);
    for pair in curve.points.windows(2) {
        accumulated += pair[1].sub(pair[0]).length();
        parameters.push(accumulated);
    }
    let total = accumulated.max(1e-12);
    for parameter in &mut parameters {
        *parameter /= total;
    }
    let degree = FIT_DEGREE.min(count - 1);
    let as_points = |samples: &[[f64; 2]]| {
        samples
            .iter()
            .map(|uv| Vec4::from_point(Vec3::new(uv[0], uv[1], 0.0), 1.0))
            .collect::<Vec<_>>()
    };
    let points3 = curve
        .points
        .iter()
        .map(|point| Vec4::from_point(*point, 1.0))
        .collect::<Vec<_>>();
    Ok((
        crate::fit::interpolate_homogeneous(&points3, degree, &parameters)?,
        crate::fit::interpolate_homogeneous(&as_points(&curve.uv_first), degree, &parameters)?,
        crate::fit::interpolate_homogeneous(&as_points(&curve.uv_second), degree, &parameters)?,
    ))
}
