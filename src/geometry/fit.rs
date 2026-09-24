use crate::{KnotVector, NurbsCurve, Vec3, Vec4};

#[derive(Clone, Debug)]
pub struct PolylineFit {
    pub curve: NurbsCurve,
    pub parameters: Vec<f64>,
    pub kept: Vec<Vec3>,
}

pub fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, String> {
    let count = rhs.len();
    if count == 0 || matrix.len() != count || matrix.iter().any(|row| row.len() != count) {
        return Err("solve_dense: matrix must be square and match RHS".into());
    }
    let scale = matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return Err("solve_dense: singular matrix".into());
    }
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        if matrix[pivot][column].abs() <= 1e-13 * scale {
            return Err("solve_dense: singular matrix".into());
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        for row in column + 1..count {
            let factor = matrix[row][column] / diagonal;
            matrix[row][column] = 0.0;
            for entry in column + 1..count {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut result = vec![0.0; count];
    for row in (0..count).rev() {
        let remainder: f64 = (row + 1..count)
            .map(|column| matrix[row][column] * result[column])
            .sum();
        result[row] = (rhs[row] - remainder) / matrix[row][row];
    }
    Ok(result)
}

/// `solve_dense` for fixed-size systems on the stack — identical pivoting and
/// thresholds, no allocation. `count <= N` solves the leading `count`-sized
/// block (for callers that shrink the system by fixing parameters).
pub fn solve_small<const N: usize>(
    mut matrix: [[f64; N]; N],
    mut rhs: [f64; N],
    count: usize,
) -> Result<[f64; N], String> {
    if count == 0 || count > N {
        return Err("solve_small: invalid system size".into());
    }
    let scale = matrix
        .iter()
        .take(count)
        .flat_map(|row| row.iter().take(count))
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return Err("solve_small: singular matrix".into());
    }
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        if matrix[pivot][column].abs() <= 1e-13 * scale {
            return Err("solve_small: singular matrix".into());
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        for row in column + 1..count {
            let factor = matrix[row][column] / diagonal;
            matrix[row][column] = 0.0;
            for entry in column + 1..count {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut result = [0.0; N];
    for row in (0..count).rev() {
        let remainder: f64 = (row + 1..count)
            .map(|column| matrix[row][column] * result[column])
            .sum();
        result[row] = (rhs[row] - remainder) / matrix[row][row];
    }
    Ok(result)
}

/// Solve a B-spline interpolation (collocation) system. The collocation matrix
/// `N_j(τ_i)` produced by the knot-averaging schemes below is banded with
/// half-bandwidth ≤ `degree` (Schoenberg–Whitney) and totally positive, so a
/// no-pivot BANDED elimination is O(n·degree²) — vs the O(n³) of the general
/// dense solver — and is numerically identical to it on these systems.
///
/// The banded result is accepted only when its in-band residual is tiny;
/// otherwise (a pathological / near-singular system) it falls back to the
/// partial-pivoting dense solver. This keeps large STEP-import pcurve fits
/// (which push `n` toward ~500) from turning each solve into a ~500³ blowup.
pub fn solve_collocation(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    degree: usize,
) -> Result<Vec<f64>, String> {
    let count = rhs.len();
    let bandwidth = degree.max(1).min(count.saturating_sub(1).max(1));
    if let Ok(solution) = solve_banded(matrix, rhs, bandwidth) {
        let scale = rhs
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let tolerance = 1e-9 * scale.max(1.0);
        let mut residual = 0.0_f64;
        for row in 0..count {
            let lo = row.saturating_sub(bandwidth);
            let hi = (row + bandwidth).min(count - 1);
            let mut accumulated = 0.0;
            for col in lo..=hi {
                accumulated += matrix[row][col] * solution[col];
            }
            residual = residual.max((accumulated - rhs[row]).abs());
        }
        if residual <= tolerance {
            return Ok(solution);
        }
    }
    solve_dense(matrix.to_vec(), rhs.to_vec())
}

pub fn solve_banded(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    bandwidth: usize,
) -> Result<Vec<f64>, String> {
    let count = rhs.len();
    if count == 0 || matrix.len() != count || matrix.iter().any(|row| row.len() != count) {
        return Err("solve_banded: matrix must be square and match RHS".into());
    }
    let mut matrix = matrix.to_vec();
    let mut rhs = rhs.to_vec();
    for column in 0..count {
        let pivot = matrix[column][column];
        if pivot.abs() <= 1e-300 {
            return Err("solve_banded: singular matrix".into());
        }
        let maximum_row = (column + bandwidth).min(count - 1);
        for row in column + 1..=maximum_row {
            let factor = matrix[row][column] / pivot;
            if factor == 0.0 {
                continue;
            }
            let maximum_column = (column + bandwidth).min(count - 1);
            for entry in column..=maximum_column {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut result = vec![0.0; count];
    for row in (0..count).rev() {
        let mut value = rhs[row];
        let maximum_column = (row + bandwidth).min(count - 1);
        for column in row + 1..=maximum_column {
            value -= matrix[row][column] * result[column];
        }
        if matrix[row][row].abs() <= 1e-300 {
            return Err("solve_banded: singular matrix".into());
        }
        result[row] = value / matrix[row][row];
    }
    Ok(result)
}

/// Global B-spline interpolation (The NURBS Book A9.1), matching the
/// reference implementation's supplied-parameter path.
/// Interpolate HOMOGENEOUS points (Vec4, rational data) at the given
/// parameters with the same averaged-knot scheme as `interpolate_curve`.
/// Parameters must start at 0 and end at 1. Rows fitted with identical
/// parameters share knots — the contract the blend-surface rows rely on.
pub fn interpolate_homogeneous(
    points: &[Vec4],
    degree: usize,
    parameters: &[f64],
) -> Result<NurbsCurve, String> {
    if points.len() < 2 || parameters.len() != points.len() {
        return Err(
            "interpolate_homogeneous: points and parameters must have matching length >= 2".into(),
        );
    }
    if parameters.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("interpolate_homogeneous: parameters must be strictly increasing".into());
    }
    let n = points.len() - 1;
    let degree = degree.min(n);
    let m = n + degree + 1;
    let mut knots = vec![0.0; m + 1];
    for knot in &mut knots[m - degree..=m] {
        *knot = 1.0;
    }
    for j in 1..=n.saturating_sub(degree) {
        knots[j + degree] = parameters[j..j + degree].iter().sum::<f64>() / degree as f64;
    }
    let knot_vector = KnotVector::new(knots.clone(), degree)?;
    let mut matrix = vec![vec![0.0; n + 1]; n + 1];
    for (row, &parameter) in parameters.iter().enumerate() {
        let span = knot_vector.find_span(parameter);
        let basis = knot_vector.basis_functions(span, parameter);
        for (offset, value) in basis.into_iter().enumerate() {
            matrix[row][span - degree + offset] = value;
        }
    }
    let solve_axis = |axis: fn(&Vec4) -> f64| {
        solve_collocation(
            &matrix,
            &points.iter().map(axis).collect::<Vec<_>>(),
            degree,
        )
    };
    let xs = solve_axis(|point| point.x)?;
    let ys = solve_axis(|point| point.y)?;
    let zs = solve_axis(|point| point.z)?;
    let ws = solve_axis(|point| point.w)?;
    NurbsCurve::new(
        degree,
        knots,
        (0..=n)
            .map(|index| Vec4 {
                x: xs[index],
                y: ys[index],
                z: zs[index],
                w: ws[index],
            })
            .collect(),
    )
}

/// Global B-spline interpolation (The NURBS Book A9.1) through `points` at the
/// supplied strictly increasing `parameters`.
///
/// The result is CLAMPED over exactly `[parameters[0], parameters[n]]` — the
/// caller's own parameter interval, not a normalized `[0, 1]`. A9.1's end knots
/// are `ū_0` and `ū_n`; hardcoding `0.0`/`1.0` there is only correct when the
/// caller already normalized, and silently emits a knot vector that DESCENDS at
/// the clamp when it did not (STEP import's `reconcile_edges_onto_surfaces`
/// re-fits an edge over its own domain — on ABC 00000290 that is `[1.0,
/// 1.2499]`, producing interior knots up to 1.234 followed by the four 1.0 end
/// knots, i.e. a 0.234 descent that `validate_knots` rightly rejected). Since
/// the averaged interior knots always lie strictly between `parameters[0]` and
/// `parameters[n]`, taking the ends from the parameters makes the vector valid
/// for ANY strictly increasing input, and is bit-identical for the already-
/// normalized callers.
pub fn interpolate_curve(
    points: &[Vec3],
    degree: usize,
    parameters: &[f64],
) -> Result<NurbsCurve, String> {
    if points.len() < 2 || parameters.len() != points.len() {
        return Err(
            "interpolate_curve: points and parameters must have matching length >= 2".into(),
        );
    }
    if parameters.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("interpolate_curve: parameters must be strictly increasing".into());
    }
    let n = points.len() - 1;
    let degree = degree.min(n);
    let m = n + degree + 1;
    let mut knots = vec![parameters[0]; m + 1];
    for knot in &mut knots[m - degree..=m] {
        *knot = parameters[n];
    }
    for j in 1..=n.saturating_sub(degree) {
        knots[j + degree] = parameters[j..j + degree].iter().sum::<f64>() / degree as f64;
    }
    let knot_vector = KnotVector::new(knots.clone(), degree)?;
    let mut matrix = vec![vec![0.0; n + 1]; n + 1];
    for (row, &parameter) in parameters.iter().enumerate() {
        let span = knot_vector.find_span(parameter);
        let basis = knot_vector.basis_functions(span, parameter);
        for (offset, value) in basis.into_iter().enumerate() {
            matrix[row][span - degree + offset] = value;
        }
    }
    let solve_axis = |axis: fn(Vec3) -> f64| {
        solve_collocation(
            &matrix,
            &points.iter().copied().map(axis).collect::<Vec<_>>(),
            degree,
        )
    };
    let xs = solve_axis(|point| point.x)?;
    let ys = solve_axis(|point| point.y)?;
    let zs = solve_axis(|point| point.z)?;
    NurbsCurve::new(
        degree,
        knots,
        (0..=n)
            .map(|index| Vec4::from_point(Vec3::new(xs[index], ys[index], zs[index]), 1.0))
            .collect(),
    )
}

/// Cubic interpolation with PRESCRIBED end derivatives (Piegl–Tiller §9.2.2):
/// n+1 points plus two tangent rows give n+3 clamped control points. The
/// derivative conditions use the exact clamped end forms
/// C'(t0) = p/(u_{p+1}−t0)·(Q1−Q0) and C'(t1) = p/(t1−u_{m−p−1})·(Qn−Qn−1),
/// so the requested tangents are reproduced exactly — the §5.8 loft tangency
/// building block.
pub fn interpolate_curve_with_end_tangents(
    points: &[Vec3],
    parameters: &[f64],
    start_tangent: Vec3,
    end_tangent: Vec3,
) -> Result<NurbsCurve, String> {
    if points.len() < 2 || parameters.len() != points.len() {
        return Err(
            "interpolate_curve_with_end_tangents: points and parameters must match, >= 2".into(),
        );
    }
    if parameters.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("interpolate_curve_with_end_tangents: parameters must increase".into());
    }
    let degree = 3usize;
    let n = points.len() - 1;
    let control_count = n + 3;
    let t0 = parameters[0];
    let t1 = parameters[n];
    // Clamped knots with n−1 interior values averaged over parameter runs
    // (the tangent rows consume the two extra controls).
    let mut knots = vec![t0; degree + 1];
    for j in 0..n.saturating_sub(1) {
        let window = &parameters[j + 1..(j + degree).min(n) + 1];
        knots.push(window.iter().sum::<f64>() / window.len() as f64);
    }
    knots.extend(std::iter::repeat(t1).take(degree + 1));
    if knots.len() != control_count + degree + 1 {
        return Err(format!(
            "interpolate_curve_with_end_tangents: internal knot count {} for {} controls",
            knots.len(),
            control_count
        ));
    }
    let knot_vector = KnotVector::new(knots.clone(), degree)?;
    let mut matrix = vec![vec![0.0; control_count]; control_count];
    let mut rhs_points = vec![Vec3::default(); control_count];
    // Row 0: C(t0) = P0; row 1: start tangent; rows 2..=n: interior + end
    // interpolation; row n+1... reorganized: standard layout is
    // [P0, T0, P1..Pn-1, T1, Pn].
    matrix[0][0] = 1.0;
    rhs_points[0] = points[0];
    let start_span = knots[degree + 1] - t0;
    matrix[1][0] = -(degree as f64) / start_span;
    matrix[1][1] = (degree as f64) / start_span;
    rhs_points[1] = start_tangent;
    for (index, &parameter) in parameters.iter().enumerate().take(n).skip(1) {
        let row = index + 1;
        let span = knot_vector.find_span(parameter);
        let basis = knot_vector.basis_functions(span, parameter);
        for (offset, value) in basis.into_iter().enumerate() {
            matrix[row][span - degree + offset] = value;
        }
        rhs_points[row] = points[index];
    }
    let end_span = t1 - knots[control_count - 1];
    matrix[control_count - 2][control_count - 2] = -(degree as f64) / end_span;
    matrix[control_count - 2][control_count - 1] = (degree as f64) / end_span;
    rhs_points[control_count - 2] = end_tangent;
    matrix[control_count - 1][control_count - 1] = 1.0;
    rhs_points[control_count - 1] = points[n];
    let solve_axis = |axis: fn(Vec3) -> f64| {
        solve_dense(
            matrix.clone(),
            rhs_points.iter().copied().map(axis).collect::<Vec<_>>(),
        )
    };
    let xs = solve_axis(|point| point.x)?;
    let ys = solve_axis(|point| point.y)?;
    let zs = solve_axis(|point| point.z)?;
    NurbsCurve::new(
        degree,
        knots,
        (0..control_count)
            .map(|index| Vec4::from_point(Vec3::new(xs[index], ys[index], zs[index]), 1.0))
            .collect(),
    )
}

pub fn interpolate_curve_thinned(
    points: &[Vec3],
    degree: usize,
    maximum_points: usize,
) -> Result<NurbsCurve, String> {
    if maximum_points < 2 {
        return Err("interpolate_curve_thinned: maximum point count must be at least 2".into());
    }
    if points.len() <= maximum_points {
        let parameters = chord_parameters(points);
        return interpolate_curve(points, degree, &parameters);
    }
    let step = (points.len() - 1) as f64 / (maximum_points - 1) as f64;
    let thinned = (0..maximum_points)
        .map(|index| points[(index as f64 * step).round() as usize])
        .collect::<Vec<_>>();
    let parameters = chord_parameters(&thinned);
    interpolate_curve(&thinned, degree, &parameters)
}

fn chord_parameters(points: &[Vec3]) -> Vec<f64> {
    if points.len() < 2 {
        return vec![0.0; points.len()];
    }
    let mut parameters = vec![0.0; points.len()];
    for index in 1..points.len() {
        parameters[index] = parameters[index - 1] + points[index].sub(points[index - 1]).length();
    }
    let length = parameters[points.len() - 1];
    if length <= 1e-15 {
        for (index, parameter) in parameters.iter_mut().enumerate() {
            *parameter = index as f64 / (points.len() - 1) as f64;
        }
    } else {
        for parameter in &mut parameters {
            *parameter /= length;
        }
    }
    parameters
}

pub fn simplify_polyline(points: &[Vec3], tolerance: f64) -> Vec<Vec3> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0usize, points.len() - 1)];
    while let Some((start, end)) = stack.pop() {
        if end - start < 2 {
            continue;
        }
        let a = points[start];
        let direction = points[end].sub(a);
        let length_squared = direction.length_squared().max(1e-300);
        let mut worst = None;
        let mut worst_distance = tolerance;
        for (index, point) in points.iter().enumerate().take(end).skip(start + 1) {
            let fraction = point.sub(a).dot(direction) / length_squared;
            let fraction = fraction.clamp(0.0, 1.0);
            let distance = point.sub(a.add(direction.scale(fraction))).length();
            if distance > worst_distance {
                worst_distance = distance;
                worst = Some(index);
            }
        }
        if let Some(index) = worst {
            keep[index] = true;
            stack.push((start, index));
            stack.push((index, end));
        }
    }
    points
        .iter()
        .copied()
        .zip(keep)
        .filter_map(|(point, keep)| keep.then_some(point))
        .collect()
}

pub fn interpolate_curve_local(
    points: &[Vec3],
    parameters: &[f64],
    tension: f64,
) -> Result<NurbsCurve, String> {
    if points.len() != parameters.len() || points.len() < 2 {
        return Err(
            "interpolate_curve_local: points and parameters must have matching length >= 2".into(),
        );
    }
    if points.len() == 2 {
        return interpolate_curve(points, 1, parameters);
    }
    let count = points.len();
    let mut tangents = vec![Vec3::default(); count];
    tangents[0] = points[1]
        .sub(points[0])
        .scale(1.0 / (parameters[1] - parameters[0]));
    tangents[count - 1] = points[count - 1]
        .sub(points[count - 2])
        .scale(1.0 / (parameters[count - 1] - parameters[count - 2]));
    for index in 1..count - 1 {
        let previous_interval = parameters[index] - parameters[index - 1];
        let next_interval = parameters[index + 1] - parameters[index];
        let total = previous_interval + next_interval;
        let previous_secant = points[index]
            .sub(points[index - 1])
            .scale(1.0 / previous_interval);
        let next_secant = points[index + 1]
            .sub(points[index])
            .scale(1.0 / next_interval);
        let mut tangent =
            points[index - 1]
                .scale(-next_interval / (previous_interval * total))
                .add(points[index].scale(
                    (next_interval - previous_interval) / (previous_interval * next_interval),
                ))
                .add(points[index + 1].scale(previous_interval / (next_interval * total)));
        if previous_secant.dot(next_secant) <= 0.0
            || tangent.dot(previous_secant) <= 0.0
            || tangent.dot(next_secant) <= 0.0
        {
            tangent = Vec3::default();
        } else {
            let maximum = 3.0 * previous_secant.length().min(next_secant.length());
            if tangent.length() > maximum {
                tangent = tangent.normalized()?.scale(maximum);
            }
        }
        tangents[index] = tangent;
    }
    if tension != 1.0 {
        for tangent in &mut tangents {
            *tangent = tangent.scale(tension);
        }
    }
    let mut control_points = vec![Vec4::from_point(points[0], 1.0)];
    for index in 0..count - 1 {
        let interval = parameters[index + 1] - parameters[index];
        control_points.extend([
            Vec4::from_point(
                points[index].add(tangents[index].scale(interval / 3.0)),
                1.0,
            ),
            Vec4::from_point(
                points[index + 1].sub(tangents[index + 1].scale(interval / 3.0)),
                1.0,
            ),
            Vec4::from_point(points[index + 1], 1.0),
        ]);
    }
    let mut knots = vec![parameters[0]; 4];
    for parameter in &parameters[1..count - 1] {
        knots.extend([*parameter; 3]);
    }
    knots.extend([parameters[count - 1]; 4]);
    NurbsCurve::new(3, knots, control_points)
}

pub fn fit_polyline(
    points: &[Vec3],
    tolerance: f64,
    maximum_points: usize,
    local_interpolation: bool,
) -> Result<PolylineFit, String> {
    let mut kept = simplify_polyline(points, tolerance);
    if kept.len() > 2 {
        let total: f64 = kept
            .windows(2)
            .map(|pair| pair[1].sub(pair[0]).length())
            .sum();
        let floor = (tolerance * 0.01).max(total * 1e-4);
        let first = kept[0];
        let last = kept[kept.len() - 1];
        let mut conditioned = vec![first];
        for point in &kept[1..kept.len() - 1] {
            if point.sub(first).length() > floor
                && point.sub(last).length() > floor
                && point.sub(*conditioned.last().unwrap()).length() > floor
            {
                conditioned.push(*point);
            }
        }
        conditioned.push(last);
        kept = conditioned;
    } else {
        let mut distinct = Vec::new();
        for point in kept {
            if distinct
                .last()
                .is_none_or(|previous: &Vec3| point.sub(*previous).length() > tolerance * 0.01)
            {
                distinct.push(point);
            }
        }
        kept = distinct;
    }
    if kept.len() < 2 {
        return Err("fit_polyline: degenerate polyline".into());
    }
    let maximum_points = maximum_points.max(2);
    if kept.len() > maximum_points {
        let step = (kept.len() - 1) as f64 / (maximum_points - 1) as f64;
        kept = (0..maximum_points)
            .map(|index| kept[(index as f64 * step).round() as usize])
            .collect();
    }
    let total: f64 = kept
        .windows(2)
        .map(|pair| pair[1].sub(pair[0]).length())
        .sum();
    if total <= 0.0 {
        return Err("fit_polyline: degenerate polyline".into());
    }
    let mut parameters = vec![0.0; kept.len()];
    let mut accumulated = 0.0;
    for index in 1..kept.len() {
        accumulated += kept[index].sub(kept[index - 1]).length();
        parameters[index] = accumulated / total;
    }
    *parameters.last_mut().unwrap() = 1.0;
    let curve = if local_interpolation {
        interpolate_curve_local(&kept, &parameters, 1.0)?
    } else {
        interpolate_curve(&kept, 3usize.min(kept.len() - 1), &parameters)?
    };
    Ok(PolylineFit {
        curve,
        parameters,
        kept,
    })
}

// BREP private tests: 7d3d630bf075d2f1

// BREP private tests: 05cf36d4895e74b7

/// Cox–de Boor basis over a RAW (possibly unclamped) knot array — the local
/// helper the periodic interpolation needs; `KnotVector` validation rightly
/// rejects unclamped arrays, so this stays private to the fit module.
fn raw_basis(knots: &[f64], degree: usize, span: usize, parameter: f64) -> Vec<f64> {
    let mut basis = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    basis[0] = 1.0;
    for j in 1..=degree {
        left[j] = parameter - knots[span + 1 - j];
        right[j] = knots[span + j] - parameter;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let temp = if denominator.abs() > 0.0 {
                basis[r] / denominator
            } else {
                0.0
            };
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
    basis
}

/// Boehm single-knot insertion on raw arrays (degree fixed by caller).
fn raw_insert_knot(knots: &mut Vec<f64>, controls: &mut Vec<Vec3>, degree: usize, parameter: f64) {
    // span: last index with knots[span] <= parameter, clamped to the valid
    // control range (the textbook find_span clamp — inserting at the domain
    // end otherwise indexes one past the control array).
    let span = knots
        .iter()
        .rposition(|&knot| knot <= parameter + 1e-14)
        .unwrap()
        .min(controls.len() - 1);
    let mut fresh = Vec::with_capacity(controls.len() + 1);
    fresh.extend_from_slice(&controls[..=span - degree]);
    for i in span - degree + 1..=span {
        let denominator = knots[i + degree] - knots[i];
        let alpha = if denominator.abs() > 0.0 {
            (parameter - knots[i]) / denominator
        } else {
            0.0
        };
        fresh.push(
            controls[i - 1]
                .scale(1.0 - alpha)
                .add(controls[i].scale(alpha)),
        );
    }
    fresh.extend_from_slice(&controls[span..]);
    *controls = fresh;
    knots.insert(span + 1, parameter);
}

/// EXACT closed (periodic) cubic interpolation. `points` are the S >= 4
/// distinct stations (first NOT repeated); `parameters` has S+1 strictly
/// increasing values whose last entry closes the period. The cyclic
/// collocation system is solved densely (S is small for lofts), and the
/// periodic B-spline is re-expressed in CLAMPED form by Boehm-inserting the
/// domain ends to full multiplicity — the representation every kernel
/// consumer expects — so the seam is C² by construction, not by welding.
pub fn interpolate_curve_closed(points: &[Vec3], parameters: &[f64]) -> Result<NurbsCurve, String> {
    let degree = 3usize;
    let station_count = points.len();
    if station_count < 4 {
        return Err("interpolate_curve_closed: need at least 4 stations".into());
    }
    if parameters.len() != station_count + 1 {
        return Err(
            "interpolate_curve_closed: parameters must have one more entry than points".into(),
        );
    }
    if parameters.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("interpolate_curve_closed: parameters must increase".into());
    }
    let period = parameters[station_count] - parameters[0];
    // Cyclic knot line u_j = t_{j mod S} + floor(j/S)·T for j in −3..S+4,
    // stored with offset 3: raw[k] = u_{k−3}.
    let cyclic = |j: i64| -> f64 {
        let s = station_count as i64;
        let wrap = j.div_euclid(s);
        parameters[j.rem_euclid(s) as usize] + wrap as f64 * period
    };
    let raw_knots: Vec<f64> = (-3..=(station_count as i64 + 3)).map(cyclic).collect();
    // Collocation: row i evaluates the cubic basis at t_i; the span in the
    // raw array is the one containing t_i (raw index i+3 == u_i).
    let mut matrix = vec![vec![0.0; station_count]; station_count];
    for i in 0..station_count {
        let span = i + 3;
        let basis = raw_basis(&raw_knots, degree, span, parameters[i]);
        for (offset, value) in basis.iter().enumerate() {
            // Control j = span − degree + offset in unclamped indexing, i.e.
            // cyclic control (i + offset − 3) mod S.
            let index = (i as i64 + offset as i64 - 3).rem_euclid(station_count as i64) as usize;
            matrix[i][index] += value;
        }
    }
    let solve_axis = |axis: fn(Vec3) -> f64| {
        solve_dense(
            matrix.clone(),
            points.iter().copied().map(axis).collect::<Vec<_>>(),
        )
    };
    let xs = solve_axis(|point| point.x)?;
    let ys = solve_axis(|point| point.y)?;
    let zs = solve_axis(|point| point.z)?;
    let cyclic_controls: Vec<Vec3> = (0..station_count)
        .map(|index| Vec3::new(xs[index], ys[index], zs[index]))
        .collect();
    // Window covering [t_0, t_S]: controls D_{−3..S−1} cyclically.
    let mut window_controls: Vec<Vec3> = (-3..(station_count as i64))
        .map(|j| cyclic_controls[j.rem_euclid(station_count as i64) as usize])
        .collect();
    let mut window_knots = raw_knots.clone();
    // Clamp both domain ends to full multiplicity (degree insertions each —
    // the ends currently sit at multiplicity 1).
    for _ in 0..degree {
        raw_insert_knot(
            &mut window_knots,
            &mut window_controls,
            degree,
            parameters[0],
        );
    }
    for _ in 0..degree {
        raw_insert_knot(
            &mut window_knots,
            &mut window_controls,
            degree,
            parameters[station_count],
        );
    }
    // Slice out the clamped sub-curve over [t_0, t_S]: knots from the first
    // occurrence of t_0 through the last of t_S, controls aligned so that
    // control k pairs with knot span k..k+degree+1.
    let first = window_knots
        .iter()
        .position(|&knot| (knot - parameters[0]).abs() < 1e-12)
        .ok_or("interpolate_curve_closed: clamp lost the start knot")?;
    let last = window_knots
        .iter()
        .rposition(|&knot| (knot - parameters[station_count]).abs() < 1e-12)
        .ok_or("interpolate_curve_closed: clamp lost the end knot")?;
    let clamped_knots: Vec<f64> = window_knots[first..=last].to_vec();
    let control_count = clamped_knots.len() - degree - 1;
    let clamped_controls: Vec<Vec4> = window_controls[first..first + control_count]
        .iter()
        .map(|point| Vec4::from_point(*point, 1.0))
        .collect();
    NurbsCurve::new(degree, clamped_knots, clamped_controls)
}

// BREP private tests: aa6a4fea97fb3e37
