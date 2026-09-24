use crate::fit::solve_dense;
use crate::topology::{BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, VertexRecord};
use crate::{
    interpolate_curve, measure_edge_against_pcurve_image,
    measure_surface_fit_against_pointwise_offset, offset_construction_band, solid_model_scale,
    vertex_endpoint_gap, vertex_tolerance_from_edges, KnotVector, MeasuredTolerance, NurbsCurve,
    NurbsSurface, OffsetEvaluator, OffsetNormal, Vec2, Vec3, Vec4,
};
use rustc_hash::FxHashMap as HashMap;
use serde::Serialize;

fn domains(surface: &NurbsSurface) -> Result<([f64; 2], [f64; 2]), String> {
    Ok((
        KnotVector::new(surface.knots_u.clone(), surface.degree_u)?.domain(),
        KnotVector::new(surface.knots_v.clone(), surface.degree_v)?.domain(),
    ))
}

/// The normal `offset_surface` offsets along: the face's oriented normal with
/// the singular-row recovery, i.e. the shared evaluator's
/// [`OffsetNormal::FaceStable`] lane, whose body this function used to be.
fn stable_face_normal(face: &FaceRecord, u: f64, v: f64) -> Result<Vec3, String> {
    OffsetEvaluator::new(
        "offset_surface",
        &face.surface,
        OffsetNormal::FaceStable {
            same_sense: face.same_sense,
        },
    )
    .normal(u, v)
}

fn greville_parameters(knots: &KnotVector) -> Vec<f64> {
    let mut parameters = (0..knots.control_point_count())
        .map(|index| {
            knots.knots[index + 1..=index + knots.degree]
                .iter()
                .sum::<f64>()
                / knots.degree as f64
        })
        .collect::<Vec<_>>();
    let domain = knots.domain();
    parameters[0] = domain[0];
    *parameters.last_mut().unwrap() = domain[1];
    parameters
}

/// Rational collocation matrix: rows are the rational basis functions
/// R_i(t) = N_i(t)·w_i / Σ_k N_k(t)·w_k evaluated at each parameter. With
/// uniform weights this reduces to the ordinary B-spline collocation matrix.
fn collocation_matrix(knots: &KnotVector, parameters: &[f64], weights: &[f64]) -> Vec<Vec<f64>> {
    parameters
        .iter()
        .map(|parameter| {
            let mut row = vec![0.0; knots.control_point_count()];
            let span = knots.find_span(*parameter);
            for (offset, value) in knots
                .basis_functions(span, *parameter)
                .into_iter()
                .enumerate()
            {
                let index = span - knots.degree + offset;
                row[index] = value * weights[index];
            }
            let denominator: f64 = row.iter().sum();
            if denominator.abs() > 0.0 {
                for value in &mut row {
                    *value /= denominator;
                }
            }
            row
        })
        .collect()
}

/// Split the weight grid into per-direction factors when it is separable
/// (w_ij = a_i·b_j), which covers every tensor surface built from rational
/// profile/rail curves (cylinders, cones, spheres, tori, revolves).
fn separable_weights(weights: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<f64>)> {
    let first_row = weights.first()?;
    let anchor = *first_row.first()?;
    if anchor.abs() <= 1e-12 {
        return None;
    }
    let a: Vec<f64> = weights.iter().map(|row| row[0]).collect();
    let b: Vec<f64> = first_row.iter().map(|w| w / anchor).collect();
    for (i, row) in weights.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            if (w - a[i] * b[j]).abs() > 1e-10 * (1.0 + w.abs()) {
                return None;
            }
        }
    }
    Some((a, b))
}

/// Interpolate the sample grid in the SOURCE surface's rational basis (same
/// knots and weights). When the true offset is representable in that basis —
/// planes, cylinders, cones, spheres, tori — collocation at the Greville grid
/// recovers it EXACTLY, so offset carriers stay real analytic surfaces
/// instead of non-rational approximations with span-scale wobble.
fn interpolate_tensor(
    knot_u: &KnotVector,
    knot_v: &KnotVector,
    parameters_u: &[f64],
    parameters_v: &[f64],
    samples: &[Vec<Vec3>],
    weights: &[Vec<f64>],
) -> Result<Vec<Vec<Vec4>>, String> {
    let count_u = parameters_u.len();
    let count_v = parameters_v.len();
    if let Some((weights_u, weights_v)) = separable_weights(weights) {
        let matrix_u = collocation_matrix(knot_u, parameters_u, &weights_u);
        let matrix_v = collocation_matrix(knot_v, parameters_v, &weights_v);
        let mut intermediate = vec![vec![Vec3::default(); count_v]; count_u];
        for column in 0..count_v {
            let solve_axis = |axis: fn(Vec3) -> f64| {
                solve_dense(
                    matrix_u.clone(),
                    samples.iter().map(|row| axis(row[column])).collect(),
                )
            };
            let x = solve_axis(|point| point.x)?;
            let y = solve_axis(|point| point.y)?;
            let z = solve_axis(|point| point.z)?;
            for row in 0..count_u {
                intermediate[row][column] = Vec3::new(x[row], y[row], z[row]);
            }
        }
        let mut controls = vec![vec![Vec4::from_point(Vec3::default(), 1.0); count_v]; count_u];
        for row in 0..count_u {
            let solve_axis = |axis: fn(Vec3) -> f64| {
                solve_dense(
                    matrix_v.clone(),
                    intermediate[row].iter().copied().map(axis).collect(),
                )
            };
            let x = solve_axis(|point| point.x)?;
            let y = solve_axis(|point| point.y)?;
            let z = solve_axis(|point| point.z)?;
            for column in 0..count_v {
                controls[row][column] = Vec4::from_point(
                    Vec3::new(x[column], y[column], z[column]),
                    weights[row][column],
                );
            }
        }
        return Ok(controls);
    }

    // Non-separable weights: solve the full tensor collocation system with
    // the exact 2D rational basis. Nets are small in practice.
    let unknowns = count_u * count_v;
    let mut matrix = vec![vec![0.0; unknowns]; unknowns];
    for (k, &u) in parameters_u.iter().enumerate() {
        let span_u = knot_u.find_span(u);
        let basis_u = knot_u.basis_functions(span_u, u);
        for (l, &v) in parameters_v.iter().enumerate() {
            let span_v = knot_v.find_span(v);
            let basis_v = knot_v.basis_functions(span_v, v);
            let row = &mut matrix[k * count_v + l];
            let mut denominator = 0.0;
            for (du, value_u) in basis_u.iter().enumerate() {
                let i = span_u - knot_u.degree + du;
                for (dv, value_v) in basis_v.iter().enumerate() {
                    let j = span_v - knot_v.degree + dv;
                    let entry = value_u * value_v * weights[i][j];
                    row[i * count_v + j] = entry;
                    denominator += entry;
                }
            }
            if denominator.abs() > 0.0 {
                for value in row.iter_mut() {
                    *value /= denominator;
                }
            }
        }
    }
    let solve_axis = |axis: fn(Vec3) -> f64| {
        solve_dense(
            matrix.clone(),
            samples
                .iter()
                .flat_map(|row| row.iter().copied().map(axis))
                .collect(),
        )
    };
    let x = solve_axis(|point| point.x)?;
    let y = solve_axis(|point| point.y)?;
    let z = solve_axis(|point| point.z)?;
    let mut controls = vec![vec![Vec4::from_point(Vec3::default(), 1.0); count_v]; count_u];
    for row in 0..count_u {
        for column in 0..count_v {
            let index = row * count_v + column;
            controls[row][column] = Vec4::from_point(
                Vec3::new(x[index], y[index], z[index]),
                weights[row][column],
            );
        }
    }
    Ok(controls)
}

/// Which branch of [`offset_surface`] built a carrier — and, with it, whether
/// comparing that carrier against the pointwise offset at the same `(u, v)` is
/// even the right question.
///
/// Recorded rather than inferred, in the pattern `offset/reintersect.rs`
/// established for its own two lanes: two of this function's three branches
/// deliberately move the result off the pointwise offset, and a measurement that
/// did not know which branch ran would report a designed divergence as a fit
/// error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OffsetSurfaceLane {
    /// A rigid control-net shift of an affine (plane-like) carrier. The normal
    /// is constant, so the shifted net IS the pointwise offset — exact by
    /// construction, with no fit to measure.
    Affine,
    /// Greville collocation of the pointwise offset. `S_fit(u, v)` is meant to
    /// BE `offset(u, v)`, and the distance between them is the fit error this
    /// slice measures.
    #[default]
    Fit,
    /// The result was deliberately moved off the pointwise offset: the planar /
    /// ruled EXTENSION (which grows the carrier past its source rim, so the same
    /// `(u, v)` names a different point) or the apex-cone PINCH RETRIM (which
    /// pulls the crossed sample row back to the offset cone's own apex). Both
    /// are correct and intended; neither is comparable pointwise.
    Reparameterised,
}

/// A fitted offset carrier together with what its construction measured about
/// itself.
#[derive(Clone, Debug)]
pub struct MeasuredOffsetSurface {
    pub surface: NurbsSurface,
    pub lane: OffsetSurfaceLane,
    /// `max ‖S_fit(u, v) − offset(u, v)‖` over
    /// [`crate::measure_surface_fit_against_pointwise_offset`]'s grid, against
    /// the band it was judged with.
    ///
    /// `None` for [`OffsetSurfaceLane::Reparameterised`] — see that variant.
    pub fit: Option<MeasuredTolerance>,
}

/// How far an offset carrier's TRIM grows past the source face at each of
/// its four parametric sides, in world units.
///
/// A carrier is trimmed by the parametric IMAGE of the source loops, so the
/// growth is realised by reshaping the carrier surface's net while the cloned
/// pcurves stay put: an affine plane's 2x2 net slides along the plane, a ruled
/// (linear-v) net stretches each sampled ruling. Both are solved so that the
/// trim's own uv bounding box moves by exactly these amounts — a trim that
/// occupies a sub-range of its surface's domain (every booleaned or
/// blend-trimmed face) grows by the same distance as one that spans it. The
/// legacy `planar_extension: f64` entry points are the uniform case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarrierExtension {
    pub u_min: f64,
    pub u_max: f64,
    pub v_min: f64,
    pub v_max: f64,
}

impl CarrierExtension {
    pub const NONE: CarrierExtension = CarrierExtension {
        u_min: 0.0,
        u_max: 0.0,
        v_min: 0.0,
        v_max: 0.0,
    };

    pub fn uniform(amount: f64) -> CarrierExtension {
        CarrierExtension {
            u_min: amount,
            u_max: amount,
            v_min: amount,
            v_max: amount,
        }
    }

    pub fn is_active(&self) -> bool {
        self.u_min > 0.0 || self.u_max > 0.0 || self.v_min > 0.0 || self.v_max > 0.0
    }

    fn extends_v(&self) -> bool {
        self.v_min > 0.0 || self.v_max > 0.0
    }
}

/// The trim's uv bounding box as FRACTIONS of the surface domain:
/// `([ta_u, tb_u], [ta_v, tb_v])`, each in `[0, 1]`. Sampled over every
/// coedge pcurve of every loop.
fn trim_domain_fractions(
    face: &FaceRecord,
    [u0, u1]: [f64; 2],
    [v0, v1]: [f64; 2],
) -> Result<([f64; 2], [f64; 2]), String> {
    let mut low = Vec2 {
        x: f64::INFINITY,
        y: f64::INFINITY,
    };
    let mut high = Vec2 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
    };
    for coedge in face
        .loops
        .iter()
        .flat_map(|loop_record| &loop_record.coedges)
    {
        let [p0, p1] = coedge.pcurve.domain()?;
        for sample in 0..=24 {
            let uv = coedge
                .pcurve
                .evaluate(p0 + (p1 - p0) * sample as f64 / 24.0)?;
            low.x = low.x.min(uv.x);
            low.y = low.y.min(uv.y);
            high.x = high.x.max(uv.x);
            high.y = high.y.max(uv.y);
        }
    }
    let fraction = |value: f64, start: f64, end: f64| {
        let span = end - start;
        if span.abs() <= f64::EPSILON || !value.is_finite() {
            return None;
        }
        Some(((value - start) / span).clamp(0.0, 1.0))
    };
    let u = match (fraction(low.x, u0, u1), fraction(high.x, u0, u1)) {
        (Some(a), Some(b)) if b - a > 1e-6 => [a, b],
        _ => [0.0, 1.0],
    };
    let v = match (fraction(low.y, v0, v1), fraction(high.y, v0, v1)) {
        (Some(a), Some(b)) if b - a > 1e-6 => [a, b],
        _ => [0.0, 1.0],
    };
    Ok((u, v))
}

/// How far the surface's domain ENDS must move (`(back, forward)`: the start
/// end backwards, the far end forwards, world units) so that a trim occupying
/// the domain fractions `[ta, tb]` grows by exactly `grow_min` at its start and
/// `grow_max` at its end when its pcurves are left untouched. A point at
/// fraction `t` of a linearly re-mapped domain moves by
/// `-(1 - t)·back + t·forward`; solving that at `ta` and `tb` gives the pair
/// below. It reduces to `(grow_min, grow_max)` for a full-domain trim, and
/// never shrinks either end (both results are non-negative for non-negative
/// inputs).
fn domain_end_moves(grow_min: f64, grow_max: f64, [ta, tb]: [f64; 2]) -> (f64, f64) {
    let span = tb - ta;
    if span <= 1e-6 {
        return (grow_min, grow_max);
    }
    let rate = (grow_min + grow_max) / span;
    let back = grow_min + ta * rate;
    let forward = (1.0 - ta) * rate - grow_min;
    (back.max(0.0), forward.max(0.0))
}

/// Construct the same fitted offset carrier surface as the reference shell
/// implementation. Positive distance follows its convention and moves
/// opposite the face's outward normal.
pub fn offset_surface(
    face: &FaceRecord,
    distance: f64,
    planar_extension: f64,
) -> Result<NurbsSurface, String> {
    offset_surface_with_lane(face, distance, &CarrierExtension::uniform(planar_extension))
        .map(|(surface, _)| surface)
}

/// [`offset_surface`], plus the deviation MEASURED between the carrier it built
/// and the pointwise offset that carrier approximates.
///
/// The surface is bit-identical to [`offset_surface`]'s — this calls the same
/// body and adds a read-only pass afterwards. Nothing here can change the
/// carrier: measuring the deviation AFTER the carrier is built records what
/// happened; the size-derived band stays the floor and the bar it is judged
/// against. A measurement never widens a band.
pub fn offset_surface_measured(
    face: &FaceRecord,
    distance: f64,
    planar_extension: f64,
    band: f64,
) -> Result<MeasuredOffsetSurface, String> {
    offset_surface_measured_sided(
        face,
        distance,
        &CarrierExtension::uniform(planar_extension),
        band,
    )
}

/// [`offset_surface_measured`] with a per-side [`CarrierExtension`].
pub fn offset_surface_measured_sided(
    face: &FaceRecord,
    distance: f64,
    extension: &CarrierExtension,
    band: f64,
) -> Result<MeasuredOffsetSurface, String> {
    let (surface, lane) = offset_surface_with_lane(face, distance, extension)?;
    let fit = match lane {
        OffsetSurfaceLane::Affine => Some(MeasuredTolerance::exact(band)),
        OffsetSurfaceLane::Fit => Some(measure_surface_fit_against_pointwise_offset(
            &face.surface,
            face.same_sense,
            &surface,
            distance,
            band,
        )?),
        OffsetSurfaceLane::Reparameterised => None,
    };
    Ok(MeasuredOffsetSurface { surface, lane, fit })
}

fn offset_surface_with_lane(
    face: &FaceRecord,
    distance: f64,
    extension: &CarrierExtension,
) -> Result<(NurbsSurface, OffsetSurfaceLane), String> {
    let source = &face.surface;
    let extension_active = extension.is_active();
    if source.is_affine()? {
        let ([u0, u1], [v0, v1]) = domains(source)?;
        let normal = stable_face_normal(face, (u0 + u1) / 2.0, (v0 + v1) / 2.0)?;
        let shift = normal.scale(-distance);
        let mut points = source
            .control_points
            .iter()
            .map(|row| {
                row.iter()
                    .map(|control| Ok(control.point()?.add(shift)))
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;
        if extension_active {
            let p00 = points[0][0];
            let p01 = points[0][1];
            let p10 = points[1][0];
            let direction_u = p10.sub(p00).normalized()?;
            let direction_v = p01.sub(p00).normalized()?;
            // The net slides while the cloned pcurves stay put, so a trim
            // that spans only part of the domain would grow by less than
            // asked (a 20-wide face trimmed to [0,16] by a blend moved its
            // x=16 edge 0.6 for a 1.0 pad). Solve the slide for the TRIM's
            // own bounding box instead.
            let (trim_u, trim_v) = trim_domain_fractions(face, [u0, u1], [v0, v1])?;
            let (back_u, forward_u) = domain_end_moves(extension.u_min, extension.u_max, trim_u);
            let (back_v, forward_v) = domain_end_moves(extension.v_min, extension.v_max, trim_v);
            points[0][0] = p00
                .sub(direction_u.scale(back_u))
                .sub(direction_v.scale(back_v));
            points[0][1] = p01
                .sub(direction_u.scale(back_u))
                .add(direction_v.scale(forward_v));
            points[1][0] = p10
                .add(direction_u.scale(forward_u))
                .sub(direction_v.scale(back_v));
            points[1][1] = points[1][1]
                .add(direction_u.scale(forward_u))
                .add(direction_v.scale(forward_v));
        }
        let controls = points
            .into_iter()
            .enumerate()
            .map(|(row, points)| {
                points
                    .into_iter()
                    .enumerate()
                    .map(|(column, point)| {
                        Vec4::from_point(point, source.control_points[row][column].w)
                    })
                    .collect()
            })
            .collect();
        // The extension slides the control net along the plane, so the same
        // `(u, v)` no longer names the pointwise offset of the same source
        // point; without it the shift is rigid and exact.
        let lane = if extension_active {
            OffsetSurfaceLane::Reparameterised
        } else {
            OffsetSurfaceLane::Affine
        };
        return Ok((
            NurbsSurface::new(
                source.degree_u,
                source.degree_v,
                source.knots_u.clone(),
                source.knots_v.clone(),
                controls,
            )?,
            lane,
        ));
    }

    let knot_u = KnotVector::new(source.knots_u.clone(), source.degree_u)?;
    let knot_v = KnotVector::new(source.knots_v.clone(), source.degree_v)?;
    let parameters_u = greville_parameters(&knot_u);
    let parameters_v = greville_parameters(&knot_v);
    // The Greville sample grid IS a pointwise offset evaluation — this fit is
    // the shared evaluator's consumer, not its peer. `offset_surface`'s
    // positive distance moves OPPOSITE the face normal while the evaluator's
    // moves ALONG it, so the negation happens once, here, with a name on it
    // (audit §4.1's four hand negations get no fifth).
    let evaluator = OffsetEvaluator::new(
        "offset_surface",
        source,
        OffsetNormal::FaceStable {
            same_sense: face.same_sense,
        },
    );
    let mut samples = Vec::new();
    for &u in &parameters_u {
        let mut row = Vec::new();
        for &v in &parameters_v {
            row.push(evaluator.at(u, v, -distance)?.point);
        }
        samples.push(row);
    }
    // APEX-CONE PINCH RETRIM: offsetting an apex cone INWARD moves each
    // ruling past the axis — the sampled far row becomes a ring on the far
    // side (radius d·cos half-angle, mirrored through the axis) and the
    // offset surface self-pinches inside the v-domain. The genuine cavity
    // ends AT the pinch (the offset cone's own apex). For a linear-v net
    // (two sample rows — every made/booleaned cone) the pinch lies on each
    // ruling at the fraction where the radial vector vanishes: detect the
    // inversion (far-row radials anti-parallel to near-row radials about the
    // row centroids) and pull the far row back to the pinch point, so the
    // fitted surface ends in a proper degenerate apex row instead of a
    // parasitic inverted tip ending in an unweldable ring.
    // Both blocks below move the sample grid OFF the pointwise offset on
    // purpose. Recording that is what lets `offset_surface_measured` decline to
    // report a designed divergence as a fit error.
    let mut reparameterised = false;
    if parameters_v.len() == 2 && parameters_u.len() >= 3 {
        let centroid = |column: usize| {
            let mut sum = Vec3::default();
            for row in &samples {
                sum = sum.add(row[column]);
            }
            sum.scale(1.0 / samples.len() as f64)
        };
        let near_centroid = centroid(0);
        let far_centroid = centroid(1);
        let mut inverted = true;
        let mut pinch_fraction = 0.0f64;
        let mut near_mean = 0.0f64;
        let mut far_mean = 0.0f64;
        for row in &samples {
            let near_radial = row[0].sub(near_centroid);
            let far_radial = row[1].sub(far_centroid);
            let near_len = near_radial.length();
            let far_len = far_radial.length();
            if near_len <= 1e-9 || far_len <= 1e-9 {
                inverted = false;
                break;
            }
            if near_radial.dot(far_radial) >= 0.0 {
                inverted = false;
                break;
            }
            pinch_fraction += near_len / (near_len + far_len) / samples.len() as f64;
            near_mean += near_len / samples.len() as f64;
            far_mean += far_len / samples.len() as f64;
        }
        if inverted {
            // Pull the crossed (smaller-ring, past-the-pinch) end back to the
            // pinch point on each ruling.
            reparameterised = true;
            let retrim_far = far_mean <= near_mean;
            for row in &mut samples {
                let near = row[0];
                let far = row[1];
                let pinch = near.add(far.sub(near).scale(pinch_fraction));
                if retrim_far {
                    row[1] = pinch;
                } else {
                    row[0] = pinch;
                }
            }
        }
        // RULED EXTENSION: `planar_extension` is a no-op for curved carriers
        // above, but a cone/cylinder lateral joined at a reflex edge needs
        // its offset skin to GROW past the source rim exactly like a plane
        // (a cylinder piercing a cone: the two offsets only meet past both
        // cloned rims). A linear-v net is ruled — stretching each sampled
        // ruling beyond both ends stays ON the same surface, so the fitted
        // carrier keeps its parameterization (knots/pcurves untouched) while
        // its world image (and with it the cloned trim's image) inflates.
        if extension.extends_v() && !inverted {
            let mut min_ruling = f64::MAX;
            let mut back_allowance = f64::MAX;
            let mut forward_allowance = f64::MAX;
            let mut extendable = true;
            for row in &samples {
                let ruling = row[1].sub(row[0]);
                let length = ruling.length();
                min_ruling = min_ruling.min(length);
                // Radii about the row centroids expose a converging (conic)
                // ruling sheaf; the extension must stop short of its apex or
                // the sheet folds through it.
                let near_radial = row[0].sub(near_centroid).length();
                let far_radial = row[1].sub(far_centroid).length();
                if (far_radial - near_radial).abs() > 1e-9 {
                    let apex_at = near_radial / (near_radial - far_radial);
                    if (-1e-9..=1.0 + 1e-9).contains(&apex_at) {
                        // Apex inside the span: degenerate sheet, do not touch.
                        extendable = false;
                        break;
                    }
                    if apex_at < 0.0 {
                        back_allowance = back_allowance.min(0.9 * -apex_at);
                    } else {
                        forward_allowance = forward_allowance.min(0.9 * (apex_at - 1.0));
                    }
                }
            }
            if extendable && min_ruling > 1e-9 {
                reparameterised = true;
                // Solved for the TRIM's v-extent, not the domain's: a bore
                // face keeps its drill's full-height surface and is trimmed
                // to the part it pierces, so moving the surface's own ends
                // by |d| moved the trim's rims by only a fraction of that
                // (a 30-long ruling trimmed to 20 gave 0.667 for 1.0) — and
                // an outward shell's bore never reached the grown planes.
                let (_, trim_v) = trim_domain_fractions(face, knot_u.domain(), knot_v.domain())?;
                let (back_world, forward_world) =
                    domain_end_moves(extension.v_min, extension.v_max, trim_v);
                let back = (back_world / min_ruling).min(back_allowance);
                let forward = (forward_world / min_ruling).min(forward_allowance);
                for row in &mut samples {
                    let ruling = row[1].sub(row[0]);
                    row[0] = row[0].sub(ruling.scale(back));
                    row[1] = row[1].add(ruling.scale(forward));
                }
            }
        }
    }
    let weights = source
        .control_points
        .iter()
        .map(|row| row.iter().map(|point| point.w).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let lane = if reparameterised {
        OffsetSurfaceLane::Reparameterised
    } else {
        OffsetSurfaceLane::Fit
    };
    Ok((
        NurbsSurface::new(
            source.degree_u,
            source.degree_v,
            source.knots_u.clone(),
            source.knots_v.clone(),
            interpolate_tensor(
                &knot_u,
                &knot_v,
                &parameters_u,
                &parameters_v,
                &samples,
                &weights,
            )?,
        )?,
        lane,
    ))
}

/// The 3D curve of one carrier edge: the image of the source coedge's pcurve
/// on the carrier surface, with the parameter range `(t0, t1)` that matches the
/// pcurve's domain fraction for fraction (the contract every coedge of this
/// carrier relies on, since the source pcurves are reused verbatim).
///
/// Built through the [`crate::image_curve`] ladder — exact on an affine sheet
/// (every planar offset), the exact iso-curve where the pcurve holds one
/// coordinate constant (a blend rail, a cylinder's cap rim, every
/// extrude/revolve boundary), and a fitted curve verified to `fit_tolerance`
/// off its nodes otherwise. Only where the ladder refuses does the edge fall
/// back to the degree-1 interpolant through `points` (the historical
/// construction), so nothing that offset before is refused now.
///
/// Why the polyline is no longer the first choice: its adaptive subdivision
/// stops at a 5e-4 chord sag, and a 128-chord rim on an r = 8.5 cylinder sits
/// up to 1.6e-4 inside the true arc. That is 16x the boolean imprint's
/// coincidence band, so when a later operation's section curve runs along that
/// rim (a cutter's side wall coplanar with the offset shell's opening wall,
/// 2026-09-06 report) the imprint cannot recognise the rim as the section it
/// already is, and reports every chord vertex as a crossing — one edge split
/// into forty. An exact arc is recognised as coincident and left alone.
fn carrier_edge_curve(
    surface: &NurbsSurface,
    pcurve: &NurbsCurve,
    points: &[Vec3],
    parameters: &[f64],
    fit_tolerance: f64,
) -> Result<(NurbsCurve, f64, f64), String> {
    // `BREP_CARRIER_POLYLINE=1` restores the polyline for an A/B comparison.
    let image = if std::env::var("BREP_CARRIER_POLYLINE").as_deref() == Ok("1") {
        Err("BREP_CARRIER_POLYLINE set".to_string())
    } else {
        crate::image_curve::image_curve(surface, pcurve, fit_tolerance, "offset_face_carrier")
    };
    match image {
        Ok(image) if image.t0 <= image.t1 => Ok((image.curve, image.t0, image.t1)),
        Ok(image) => {
            // The image runs against the pcurve: reverse it so the edge range
            // is increasing, reflecting the range through the curve's domain.
            let [start, end] = image.curve.domain()?;
            let curve = image.curve.reversed()?;
            Ok((curve, start + end - image.t0, start + end - image.t1))
        }
        Err(_) => {
            let curve = interpolate_curve(points, 1, parameters)?;
            let [t0, t1] = curve.domain()?;
            Ok((curve, t0, t1))
        }
    }
}

fn mapped_pcurve_polyline(
    surface: &NurbsSurface,
    pcurve: &NurbsCurve,
    degenerate: bool,
) -> Result<(Vec<Vec3>, Vec<f64>), String> {
    let [start, end] = pcurve.domain()?;
    let evaluate = |fraction: f64| {
        let uv = pcurve.evaluate(start + (end - start) * fraction)?;
        surface.evaluate(uv.x, uv.y)
    };
    let first = evaluate(0.0)?;
    let last = evaluate(1.0)?;
    if degenerate {
        return Ok((vec![first, last], vec![0.0, 1.0]));
    }
    fn append(
        evaluate: &impl Fn(f64) -> Result<Vec3, String>,
        a_fraction: f64,
        a: Vec3,
        b_fraction: f64,
        b: Vec3,
        depth: usize,
        parameters: &mut Vec<f64>,
        points: &mut Vec<Vec3>,
    ) -> Result<(), String> {
        let fractions =
            [0.25, 0.5, 0.75].map(|local| a_fraction + (b_fraction - a_fraction) * local);
        let samples = fractions
            .map(evaluate)
            .into_iter()
            .collect::<Result<Vec<_>, String>>()?;
        let deviation = samples
            .iter()
            .enumerate()
            .map(|(index, point)| {
                point
                    .sub(a.add(b.sub(a).scale((index + 1) as f64 * 0.25)))
                    .length()
            })
            .fold(0.0, f64::max);
        if deviation <= 5e-4 || depth >= 10 {
            parameters.push(b_fraction);
            points.push(b);
            return Ok(());
        }
        append(
            evaluate,
            a_fraction,
            a,
            fractions[1],
            samples[1],
            depth + 1,
            parameters,
            points,
        )?;
        append(
            evaluate,
            fractions[1],
            samples[1],
            b_fraction,
            b,
            depth + 1,
            parameters,
            points,
        )
    }
    let mut parameters = vec![0.0];
    let mut points = vec![first];
    append(
        &evaluate,
        0.0,
        first,
        1.0,
        last,
        0,
        &mut parameters,
        &mut points,
    )?;
    Ok((points, parameters))
}

/// Everything an offset carrier's construction MEASURED about itself.
///
/// The half this kernel lacked: a tolerance MEASURED from the geometry that was
/// actually built, rather than a band chosen from `scale` before building. It
/// lives in the one place that needs no durable-format change: alongside the
/// transient construction result, never as a field on a
/// [`crate::BrepSolid`] record. `io/snapshot.rs` is a documented durable format
/// and `SOLID_CODEC_VERSION` a versioned wire layout; persisting per-entity
/// tolerances is real, planned, and separately designed
/// (`docs/developer/kernel-plans/per-entity-tolerances.md` S3). This lands the
/// measurement with no format churn at all.
///
/// Every number here is a RECORD. None of it widens a band — see
/// [`MeasuredTolerance`]'s direction rule.
#[derive(Clone, Debug)]
pub struct CarrierDeviation {
    /// The derived band every measurement below was judged against:
    /// [`crate::offset_construction_band`] of the source solid's extent.
    pub band: f64,
    /// Which branch built the carrier surface.
    pub lane: OffsetSurfaceLane,
    /// The carrier surface's own fit error, when the lane has one.
    pub surface: Option<MeasuredTolerance>,
    /// `max_t ‖C_3d(t) − S_off(p(t))‖` per carrier edge id, folded over every
    /// coedge that references the edge — OCCT's `FillEdgeData` rule
    /// (`BRepOffset_SimpleOffset.cxx:296-310`), which takes the maximum over
    /// **every** adjacent face rather than the first one.
    pub edges: Vec<(u64, MeasuredTolerance)>,
    /// Per carrier vertex id, propagated from the incident edge ends by
    /// [`crate::vertex_tolerance_from_edges`] — which is also where the verdict
    /// on OCCT's 1.001 inflation factor is recorded.
    pub vertices: Vec<(u64, f64)>,
}

impl CarrierDeviation {
    /// The worst thing the construction did, against the tightest band it
    /// faced. `None` only when there was nothing at all to measure.
    pub fn worst(&self) -> Option<MeasuredTolerance> {
        MeasuredTolerance::worst(
            self.surface
                .into_iter()
                .chain(self.edges.iter().map(|(_, measured)| *measured))
                .chain(
                    self.vertices
                        .iter()
                        .map(|(_, gap)| MeasuredTolerance::new(*gap, self.band)),
                ),
        )
    }

    /// The entities whose measured deviation exceeded the derived band — the
    /// interesting case, and the only one any gate acts on.
    pub fn exceedances(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(surface) = self.surface {
            if surface.exceeds_band() {
                out.push(format!("carrier surface fit {}", surface.describe()));
            }
        }
        for (id, measured) in &self.edges {
            if measured.exceeds_band() {
                out.push(format!("edge {id} {}", measured.describe()));
            }
        }
        for (id, gap) in &self.vertices {
            let measured = MeasuredTolerance::new(*gap, self.band);
            if measured.exceeds_band() {
                out.push(format!("vertex {id} {}", measured.describe()));
            }
        }
        out
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OffsetFaceCarrier {
    pub vertices: Vec<VertexRecord>,
    pub edges: Vec<EdgeRecord>,
    pub face: FaceRecord,
    /// What the construction measured about itself, or `None` when it was built
    /// through the unmeasured [`offset_face_carrier`] entry point.
    ///
    /// `#[serde(skip)]` on purpose: this struct crosses the wasm ABI as JSON
    /// (`abi/modeling_b.rs:500`), and a measurement is a diagnostic about a
    /// build, not part of the carrier the caller asked for. Skipping it keeps
    /// that payload byte-identical.
    #[serde(skip)]
    pub deviation: Option<CarrierDeviation>,
}

fn claim_vertex_image(
    source_id: u64,
    point: Vec3,
    vertex_images: &mut HashMap<u64, u64>,
    vertices: &mut Vec<VertexRecord>,
    next_id: &mut u64,
) -> u64 {
    if let Some(id) = vertex_images.get(&source_id) {
        return *id;
    }
    let id = *next_id;
    *next_id += 1;
    vertices.push(VertexRecord { id, point });
    vertex_images.insert(source_id, id);
    id
}

pub fn offset_face_carrier(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
    planar_extension: f64,
) -> Result<OffsetFaceCarrier, String> {
    offset_face_carrier_impl(
        solid,
        face_id,
        distance,
        &CarrierExtension::uniform(planar_extension),
        false,
    )
}

/// [`offset_face_carrier`] with a per-side [`CarrierExtension`].
pub fn offset_face_carrier_sided(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
    extension: &CarrierExtension,
) -> Result<OffsetFaceCarrier, String> {
    offset_face_carrier_impl(solid, face_id, distance, extension, false)
}

/// [`offset_face_carrier`], with every entity it builds measured against the
/// deviation it was meant to reproduce.
///
/// The carrier is bit-identical to [`offset_face_carrier`]'s — same body, same
/// arithmetic, in the same order — with a read-only measurement pass appended.
/// Three quantities land in [`CarrierDeviation`]:
///
/// * the carrier SURFACE's fit against the pointwise offset it interpolates
///   ([`offset_surface_measured`]);
/// * each carrier EDGE's 3D curve against the locus its pcurve traces on that
///   surface — the trim boundary is built by sampling exactly that composition
///   and interpolating the images, so this is the construction's own claim,
///   measured rather than assumed;
/// * each carrier VERTEX, propagated from the incident edge ends.
///
/// The interesting output is [`CarrierDeviation::exceedances`]: an entity whose
/// measured deviation is worse than the size-derived band assumed. That is a
/// construction that went wrong in a way the derived band alone cannot see, and
/// it is the case a caller should refuse on rather than ship.
pub fn offset_face_carrier_measured(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
    planar_extension: f64,
) -> Result<OffsetFaceCarrier, String> {
    offset_face_carrier_impl(
        solid,
        face_id,
        distance,
        &CarrierExtension::uniform(planar_extension),
        true,
    )
}

/// The measured deviations of one built carrier: surface fit, per edge, per
/// vertex.
///
/// Read-only over what the construction produced. The edge measurement is
/// [`crate::measure_edge_against_pcurve_image`] — validate's own
/// `adaptive_coedge_error` floored by a span-midpoint pass, because the sampler
/// alone is aliased against exactly the curves this construction builds (see
/// that module's doc) — taken against the far tighter
/// [`crate::offset_construction_band`] instead of the vendor-forgiving
/// `pcurve_acceptance` validate will use later.
fn measure_carrier(
    surface: &NurbsSurface,
    vertices: &[VertexRecord],
    edges: &[EdgeRecord],
    loops: &[LoopRecord],
    lane: OffsetSurfaceLane,
    surface_fit: Option<MeasuredTolerance>,
    band: f64,
) -> Result<CarrierDeviation, String> {
    let edge_by_id: HashMap<u64, &EdgeRecord> = edges.iter().map(|edge| (edge.id, edge)).collect();
    // Fold over coedges, not edges: a seam edge is referenced twice with two
    // different pcurves, and OCCT's `FillEdgeData` takes the maximum over every
    // adjacent face for exactly that reason.
    let mut per_edge: HashMap<u64, MeasuredTolerance> = HashMap::default();
    for loop_record in loops {
        for coedge in &loop_record.coedges {
            let Some(edge) = edge_by_id.get(&coedge.edge_id) else {
                continue;
            };
            let measured = measure_edge_against_pcurve_image(
                surface,
                &coedge.pcurve,
                edge,
                coedge.forward,
                band,
            )?;
            per_edge
                .entry(edge.id)
                .and_modify(|existing| *existing = existing.worse_of(measured))
                .or_insert(measured);
        }
    }

    let mut measured_edges: Vec<(u64, MeasuredTolerance)> = per_edge.into_iter().collect();
    measured_edges.sort_by_key(|(id, _)| *id);
    let deviation_of: HashMap<u64, f64> = measured_edges
        .iter()
        .map(|(id, measured)| (*id, measured.deviation()))
        .collect();

    // Edge ENDS by the vertex they claim, built once. A degenerate edge claims
    // the same vertex at both ends and contributes both, which is right: the
    // question is how far every representation meeting there actually lands.
    let mut ends_at: HashMap<u64, Vec<(&EdgeRecord, f64)>> = HashMap::default();
    for edge in edges {
        ends_at
            .entry(edge.start_vertex_id)
            .or_default()
            .push((edge, edge.t0));
        ends_at
            .entry(edge.end_vertex_id)
            .or_default()
            .push((edge, edge.t1));
    }

    let mut measured_vertices = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let mut gaps = Vec::new();
        let mut incident = Vec::new();
        for (edge, parameter) in ends_at.get(&vertex.id).into_iter().flatten() {
            gaps.push(vertex_endpoint_gap(
                vertex.point,
                edge.curve.evaluate(*parameter)?,
            ));
            incident.push(deviation_of.get(&edge.id).copied().unwrap_or(0.0));
        }
        measured_vertices.push((vertex.id, vertex_tolerance_from_edges(gaps, incident)));
    }

    Ok(CarrierDeviation {
        band,
        lane,
        surface: surface_fit,
        edges: measured_edges,
        vertices: measured_vertices,
    })
}

fn offset_face_carrier_impl(
    solid: &BrepSolid,
    face_id: u64,
    distance: f64,
    extension: &CarrierExtension,
    measure: bool,
) -> Result<OffsetFaceCarrier, String> {
    let source = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .find(|face| face.id == face_id)
        .ok_or_else(|| format!("offset_face_carrier: missing face {face_id}"))?;
    // The band is derived from the SOURCE SOLID's extent, matching every other
    // direct-edit/offset site (`face_offset.rs:103` and its siblings all take
    // `solid_model_scale`). The alternative basis is the FACE's own extent — the
    // reference shell keys its size-relative offset quantities on the 3D arc
    // length of the face's own boundary iso, never on the assembly. Which basis
    // is better is unsettled: it is a measurement to make over the refusal
    // corpus before switching, not a change to smuggle in here.
    let band = offset_construction_band(solid_model_scale(solid));
    // The accuracy bar a carrier EDGE's 3D curve must meet against the locus its
    // pcurve traces on the carrier surface — the kernel's own SSI fit contract,
    // the same bar `thicken` holds its wall boundaries to. NOT the construction
    // band: that is the surface-fit allowance (5e-4 of the part), 50x looser
    // than the boolean's coincidence band, and an edge fitted only that well is
    // what a later imprint mistakes for a crossing (see `carrier_edge_curve`).
    let fit_tolerance =
        crate::KernelTolerances::for_scale(solid_model_scale(solid), 1e-7).intersection_fit;
    let (surface, lane, surface_fit) = if measure {
        let measured = offset_surface_measured_sided(source, distance, extension, band)?;
        (measured.surface, measured.lane, measured.fit)
    } else {
        let (surface, lane) = offset_surface_with_lane(source, distance, extension)?;
        (surface, lane, None)
    };
    let source_edges = solid
        .edges
        .iter()
        .map(|edge| (edge.id, edge))
        .collect::<HashMap<_, _>>();
    let source_vertices = solid
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex))
        .collect::<HashMap<_, _>>();
    let mut vertices = Vec::new();
    let mut vertex_images = HashMap::default();
    let mut edges = Vec::new();
    let mut edge_images = HashMap::default();
    let mut loops = Vec::new();
    let mut next_id = 1u64;

    for source_loop in &source.loops {
        let mut coedges = Vec::new();
        for source_coedge in &source_loop.coedges {
            let source_edge = source_edges
                .get(&source_coedge.edge_id)
                .ok_or_else(|| "offset_face_carrier: missing source edge".to_string())?;
            let (source_start, source_end) = if source_coedge.forward {
                (source_edge.start_vertex_id, source_edge.end_vertex_id)
            } else {
                (source_edge.end_vertex_id, source_edge.start_vertex_id)
            };
            if !source_vertices.contains_key(&source_start)
                || !source_vertices.contains_key(&source_end)
            {
                return Err("offset_face_carrier: missing source vertex".into());
            }
            // Map even DEGENERATE source edges through the full polyline: a
            // cone apex's image on the offset surface is a genuine CIRCLE
            // (radius d·cos half-angle), not a point — shortcutting to the
            // two endpoints would collapse the ring and leave the carrier's
            // topology inconsistent with its surface. Edges whose image truly
            // collapses (sphere poles, planar corners) still interpolate to a
            // point-sized curve and keep their degenerate flag below.
            // Map even DEGENERATE source edges through the full polyline: an
            // EXTERIOR cone offset turns the apex point into a genuine RING
            // (radius d·cos half-angle) — shortcutting to the endpoints would
            // collapse it and leave the carrier topology inconsistent with
            // its surface (and the ring imprint would be dropped as
            // boundary-coincident with a "degenerate" edge). Images that
            // truly collapse (sphere poles; interior apexes after the pinch
            // retrim) stay degenerate below. The threshold scales with the
            // offset distance: a real ring measures ~d·cos α, while fitted
            // pole rows wobble ~1e-4 absolute.
            let (points, parameters) =
                mapped_pcurve_polyline(&surface, &source_coedge.pcurve, false)?;
            let collapse_tolerance = 1e-6f64.max(distance.abs() * 1e-2);
            let image_collapsed = points
                .iter()
                .all(|point| point.sub(points[0]).length() <= collapse_tolerance);
            let (edge_id, forward) =
                if let Some((edge_id, edge_start_vertex_id, creator_forward)) =
                    edge_images.get(&source_edge.id)
                {
                    if source_start == source_end {
                        // A CLOSED source edge (a seam of a periodic face: both
                        // ends are the same vertex) is referenced twice by the
                        // same loop, once per seam side, and the two references
                        // traverse it in OPPOSITE senses — that is what closes
                        // the loop. The endpoint test below cannot see that:
                        // both ends map to the same vertex image, so it answers
                        // `true` for both references and the returning coedge
                        // comes back mis-oriented.
                        //
                        // Measured on a full torus (one vertex, two seam edges):
                        // the source records `forward: false` on the two
                        // returning coedges and validates clean, while the
                        // carrier recorded `forward: true` on both and its rim
                        // edge measured 12.5 — a whole part diameter — against
                        // its own composed pcurve image. The measured tolerance
                        // this slice adds is what surfaced it; no existing gate
                        // reaches this shape.
                        //
                        // The source coedge's own sense is the answer: the image
                        // edge runs along the CREATING coedge's pcurve, so a
                        // later reference runs with it exactly when the two
                        // source coedges traverse the source edge the same way.
                        (*edge_id, source_coedge.forward == *creator_forward)
                    } else {
                        // UNCHANGED for every open edge: the image edge's start
                        // vertex identifies which way this coedge runs.
                        (
                            *edge_id,
                            vertex_images.get(&source_start) == Some(edge_start_vertex_id),
                        )
                    }
                } else {
                    let (curve, t0, t1) = if image_collapsed {
                        let curve = NurbsCurve::new(
                            1,
                            vec![0.0, 0.0, 1.0, 1.0],
                            vec![
                                Vec4::from_point(points[0], 1.0),
                                Vec4::from_point(points[0], 1.0),
                            ],
                        )?;
                        (curve, 0.0, 1.0)
                    } else {
                        carrier_edge_curve(
                            &surface,
                            &source_coedge.pcurve,
                            &points,
                            &parameters,
                            fit_tolerance,
                        )?
                    };
                    let start_vertex_id = claim_vertex_image(
                        source_start,
                        points[0],
                        &mut vertex_images,
                        &mut vertices,
                        &mut next_id,
                    );
                    let end_vertex_id = claim_vertex_image(
                        source_end,
                        points[points.len() - 1],
                        &mut vertex_images,
                        &mut vertices,
                        &mut next_id,
                    );
                    let id = next_id;
                    next_id += 1;
                    edges.push(EdgeRecord {
                        id,
                        curve,
                        t0,
                        t1,
                        start_vertex_id,
                        end_vertex_id,
                        // Degenerate only if the IMAGE collapsed too — a cone
                        // apex maps to a real ring on the offset surface and
                        // must carry a real closed edge.
                        degenerate: source_edge.degenerate && image_collapsed,
                        // Image of a named source edge on the offset carrier;
                        // suffixed so it cannot collide with the source edge
                        // when both faces survive into one solid.
                        name: source_edge
                            .name
                            .as_ref()
                            .map(|name| format!("{name}_Offset")),
                    });
                    edge_images.insert(
                        source_edge.id,
                        (id, start_vertex_id, source_coedge.forward),
                    );
                    (id, true)
                };
            let id = next_id;
            next_id += 1;
            coedges.push(CoedgeRecord {
                id,
                edge_id,
                forward,
                pcurve: source_coedge.pcurve.clone(),
            });
        }
        let id = next_id;
        next_id += 1;
        loops.push(LoopRecord { id, coedges });
    }
    let deviation = if measure {
        Some(measure_carrier(
            &surface,
            &vertices,
            &edges,
            &loops,
            lane,
            surface_fit,
            band,
        )?)
    } else {
        None
    };
    Ok(OffsetFaceCarrier {
        vertices,
        edges,
        face: FaceRecord {
            id: next_id,
            surface,
            same_sense: source.same_sense,
            loops,
            name: source.name.as_ref().map(|name| format!("{name}_Offset")),
        },
        deviation,
    })
}

// BREP private tests: f18b466be3e719c5
