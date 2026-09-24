use super::*;

pub(super) const CHAIN_PER_SEGMENT: usize = 24;
const CHAIN_OVERSHOOT: usize = 3;

/// One marched chain station, tagged with its segment and its position
/// (which may overshoot the segment for pcurve fitting).
pub(super) struct ChainSample {
    pub(super) segment: usize,
    /// Station position within the segment: 0..CHAIN_PER_SEGMENT are
    /// in-segment; negatives and > CHAIN_PER_SEGMENT are overshoot.
    pub(super) position: isize,
    pub(super) station: Station,
    /// Global chord parameter (filled after the march).
    pub(super) parameter: f64,
}

pub(super) fn march_chain(
    segments: &[ChainSegment<'_>],
    radius: f64,
    open: bool,
) -> Result<Vec<ChainSample>, String> {
    let mut samples = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        // Signs from this segment's own cross-section seed.
        let (rho1, rho2) = signed_radii(
            segment.edge,
            segment.first.face,
            segment.first.coedge,
            segment.second.face,
            segment.second.coedge,
            radius,
        )?;
        let rho = [rho1, rho2];
        let surface1 = &segment.first.face.surface;
        let surface2 = &segment.second.face.surface;
        let edge = segment.edge;
        let span = edge.t1 - edge.t0;
        let scale = march_model_scale(&edge.curve, edge.t0, edge.t1, radius)?;
        crate::report_scale_migration("blend_march_chain", scale, || {
            edge.curve
                .evaluate(edge.t0)
                .unwrap_or_default()
                .length()
                .max(1.0)
        });
        let chain_t = |position: isize| -> f64 {
            let fraction = position as f64 / CHAIN_PER_SEGMENT as f64;
            if segment.forward {
                edge.t0 + span * fraction
            } else {
                edge.t1 - span * fraction
            }
        };
        let mid_position = CHAIN_PER_SEGMENT as isize / 2;
        let seed_mid = {
            let t = chain_t(mid_position);
            let uv1 = edge_uv_on_face(segment.first.coedge, edge, t)?;
            let uv2 = edge_uv_on_face(segment.second.coedge, edge, t)?;
            [uv1[0], uv1[1], uv2[0], uv2[1]]
        };
        let solve_at = |t: f64, seed: [f64; 4]| -> Result<[f64; 4], String> {
            let derivatives = edge.curve.derivatives_extended(t, 1)?;
            let tangent = derivatives[1].normalized()?;
            let tangent = if segment.forward {
                tangent
            } else {
                tangent.scale(-1.0)
            };
            solve_station(
                surface1,
                surface2,
                rho,
                seed,
                derivatives[0],
                tangent,
                scale,
            )
        };
        let low = -(CHAIN_OVERSHOOT as isize);
        let high = (CHAIN_PER_SEGMENT + CHAIN_OVERSHOOT) as isize;
        let count = (high - low + 1) as usize;
        let mut solutions = vec![[0.0f64; 4]; count];
        let index_of = |position: isize| (position - low) as usize;
        solutions[index_of(mid_position)] = solve_at(chain_t(mid_position), seed_mid)?;
        for position in (low..mid_position).rev() {
            solutions[index_of(position)] =
                solve_at(chain_t(position), solutions[index_of(position + 1)])?;
        }
        for position in mid_position + 1..=high {
            solutions[index_of(position)] =
                solve_at(chain_t(position), solutions[index_of(position - 1)])?;
        }
        for position in low..=high {
            let t = chain_t(position);
            let uv = solutions[index_of(position)];
            let derivatives = edge.curve.derivatives_extended(t, 1)?;
            let tangent = derivatives[1].normalized()?;
            let tangent = if segment.forward {
                tangent
            } else {
                tangent.scale(-1.0)
            };
            let (_, p1, p2, center) =
                tangency_residual(surface1, surface2, rho, uv, derivatives[0], tangent)?;
            let n1 = raw_normal(surface1, uv[0], uv[1])?;
            let n2 = raw_normal(surface2, uv[2], uv[3])?;
            let cos_alpha = rho1.signum() * rho2.signum() * n1.dot(n2);
            let weight = ((1.0 + cos_alpha) * 0.5).max(0.0).sqrt();
            if weight <= 1e-6 {
                return Err("blend: faces are tangent at a chain station".into());
            }
            let apex = apex_point(p1, n1, p2, n2, center)?;
            samples.push(ChainSample {
                segment: index,
                position,
                station: Station {
                    uv1: [uv[0], uv[1]],
                    uv2: [uv[2], uv[3]],
                    p1,
                    p2,
                    center,
                    weight,
                    apex,
                },
                parameter: 0.0,
            });
        }
    }
    // Global chord parameters over the IN-SEGMENT stations (position
    // 0..CHAIN_PER_SEGMENT-1 per segment, in chain order), then assign
    // overshoot samples by extrapolating with their own chords.
    let mut accumulated = 0.0;
    let mut previous: Option<Vec3> = None;
    let mut junction_params = vec![0.0; segments.len() + 1];
    for segment in 0..segments.len() {
        for position in 0..CHAIN_PER_SEGMENT as isize {
            let sample_index = samples
                .iter()
                .position(|sample| sample.segment == segment && sample.position == position)
                .expect("chain sample present");
            let midpoint = samples[sample_index]
                .station
                .p1
                .add(samples[sample_index].station.p2)
                .scale(0.5);
            if let Some(previous_point) = previous {
                accumulated += midpoint.sub(previous_point).length();
            }
            if position == 0 {
                junction_params[segment] = accumulated;
            }
            samples[sample_index].parameter = accumulated;
            previous = Some(midpoint);
        }
    }
    // Wrap chord back to the chain start (CLOSED chains only; an OPEN chain
    // parameterises its in-segment stations onto [0, 1] with no wrap term so
    // the free ends sit at the parameter extremes, clamped like fit_open_rows).
    if !open {
        let first_midpoint = {
            let sample = samples
                .iter()
                .find(|sample| sample.segment == 0 && sample.position == 0)
                .expect("chain start sample");
            sample.station.p1.add(sample.station.p2).scale(0.5)
        };
        accumulated += first_midpoint.sub(previous.unwrap()).length();
    }
    junction_params[segments.len()] = accumulated;
    let total = accumulated.max(1e-12);
    for sample in samples.iter_mut() {
        sample.parameter /= total;
    }
    let junction_params: Vec<f64> = junction_params
        .into_iter()
        .map(|value| value / total)
        .collect();
    // Overshoot samples: REAL chord distances from the boundary
    // in-segment stations (linear extrapolation misparameterises them
    // when the neighbouring segment's station spacing differs, and the
    // support pieces' crossing region lies exactly there).
    for segment in 0..segments.len() {
        let sample_at = |samples: &[ChainSample], position: isize| -> usize {
            samples
                .iter()
                .position(|sample| sample.segment == segment && sample.position == position)
                .expect("chain sample present")
        };
        let midpoint = |samples: &[ChainSample], index: usize| -> Vec3 {
            samples[index]
                .station
                .p1
                .add(samples[index].station.p2)
                .scale(0.5)
        };
        // Below position 0.
        let anchor = sample_at(&samples, 0);
        let mut accumulated = samples[anchor].parameter;
        let mut previous_point = midpoint(&samples, anchor);
        for position in (-(CHAIN_OVERSHOOT as isize)..0).rev() {
            let index = sample_at(&samples, position);
            let point = midpoint(&samples, index);
            accumulated -= point.sub(previous_point).length() / total;
            samples[index].parameter = accumulated;
            previous_point = point;
        }
        // Above the last in-segment position.
        let anchor = sample_at(&samples, CHAIN_PER_SEGMENT as isize - 1);
        let mut accumulated = samples[anchor].parameter;
        let mut previous_point = midpoint(&samples, anchor);
        for position in CHAIN_PER_SEGMENT as isize..=(CHAIN_PER_SEGMENT + CHAIN_OVERSHOOT) as isize
        {
            let index = sample_at(&samples, position);
            let point = midpoint(&samples, index);
            accumulated += point.sub(previous_point).length() / total;
            samples[index].parameter = accumulated;
            previous_point = point;
        }
    }
    let _ = junction_params;
    // Re-origin the global parameter at segment 0's MIDDLE station so the
    // blend seam falls far from every junction (a seam ON a junction
    // collides with that junction's spoke crossing).  Kept UNWRAPPED so
    // every segment's sample window stays contiguous (segment 0 spans
    // negative params); the global closed fit wraps with rem_euclid.  An
    // OPEN chain has no seam, so its parameter stays anchored at the start
    // free end (segment 0's position 0 == 0).
    if !open {
        let p_mid = samples
            .iter()
            .find(|sample| sample.segment == 0 && sample.position == CHAIN_PER_SEGMENT as isize / 2)
            .map(|sample| sample.parameter)
            .expect("chain mid sample");
        for sample in samples.iter_mut() {
            sample.parameter -= p_mid;
        }
    }
    Ok(samples)
}
