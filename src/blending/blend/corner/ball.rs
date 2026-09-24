use super::*;

// ======================================================================
// The corner ball (§6.9.7), solved the way a station is solved
// ======================================================================
//
// Every quantity a corner closure needs is a consequence of ONE point: the
// centre C* of the radius-r ball seated in the corner, touching all N faces
// that meet there.  Writing that solve generally is what lets a corner be
// planned BEFORE anything is cut, instead of being reconstructed afterwards
// from whatever a cutter left behind.
//
// The system is the march's own tangency equation (4.9.2) with the section
// plane dropped and one row per additional face:
//
//     C* = p_j(u_j, v_j) + ρ_j · n_j(u_j, v_j)      for every face j
//
// — 2N unknowns (the surface parameters), 3(N−1) equations (every face's
// offset evaluation must land on the same centre).  N = 3 is square and has
// an isolated root; N ≥ 4 is over-determined and its least-squares RESIDUAL
// is the "is there a common ball here" test, which is why no separate
// concurrency check appears anywhere below.
//
// This is the pointwise case of the offset-intersection identity in the
// module header: `p + ρ·n` on each support, required to coincide.  Unlike
// the locus solves next door (`mixed_concave`, `mixed_curved`), a corner
// ball is a POINT, so [`crate::OffsetEvaluator`] answers it directly and no
// carrier-specific algebra is needed — planes, cylinders, cones, spheres,
// tori, revolutions and free-form patches all go through the same Newton.

/// Where the corner ball touches one of the faces meeting at the vertex.
pub(in crate::blend) struct CornerContact {
    pub(in crate::blend) face_id: u64,
    /// Converged surface parameters of the tangency point — where on the
    /// face's carrier the ball touches, which may lie OUTSIDE the face's
    /// trim (a re-entrant corner, where the ball touches the wall's plane
    /// beyond a concave edge).
    pub(in crate::blend) uv: [f64; 2],
    /// The tangency point itself — the vertex at which the two blend stripes
    /// sharing this face both stop.
    pub(in crate::blend) point: Vec3,
    /// Raw surface normal there (the march's `OffsetNormal::Raw` convention,
    /// with orientation carried by the sign of ρ).
    pub(in crate::blend) normal: Vec3,
}

/// The seated ball: its centre, and where it touches each incident face.
pub(in crate::blend) struct CornerBall {
    pub(in crate::blend) center: Vec3,
    pub(in crate::blend) contacts: Vec<CornerContact>,
    /// Worst disagreement between the per-face offset evaluations at
    /// convergence, in model units.  Zero to solver tolerance for N = 3;
    /// for N ≥ 4 a non-zero value means the N cylinders have NO common
    /// inscribed ball and the corner needs an N-sided fill instead of a
    /// spherical patch.
    pub(in crate::blend) residual: f64,
}

/// One face of the corner, with the signed offset the stripes touching it
/// were marched at.
pub(in crate::blend) struct CornerFace<'a> {
    pub(in crate::blend) face: &'a FaceRecord,
    /// Signed radius ρ: `centre = p + ρ·n_raw(p)`.  Taken from the stripe
    /// that lands on this face (`signed_radii`), never re-derived, so the
    /// ball and the stripes are guaranteed to be the same ball.
    pub(in crate::blend) rho: f64,
    /// Starting parameters — normally the sharp corner's own (u, v) on this
    /// face.
    pub(in crate::blend) seed: [f64; 2],
}

/// Damped Gauss-Newton iterations.  Matches the station solver's budget:
/// the residual is the same offset equation and the seed (the sharp corner)
/// is one radius away from the root.
const CORNER_ITERATIONS: usize = 40;

/// Offset-evaluate every face at `x` and return the per-face centres.
fn centers(faces: &[CornerFace<'_>], x: &[f64]) -> Result<Vec<Vec3>, String> {
    faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            blend_offset(&face.face.surface)
                .at(x[2 * index], x[2 * index + 1], face.rho)
                .map(|sample| sample.point)
        })
        .collect()
}

/// Residual: every face's offset centre minus the FIRST face's, stacked.
/// 3(N−1) rows.
fn residual(faces: &[CornerFace<'_>], x: &[f64]) -> Result<Vec<f64>, String> {
    let centers = centers(faces, x)?;
    let mut rows = Vec::with_capacity(3 * (faces.len() - 1));
    for center in &centers[1..] {
        let delta = center.sub(centers[0]);
        rows.extend_from_slice(&[delta.x, delta.y, delta.z]);
    }
    Ok(rows)
}

fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |worst, v| worst.max(v.abs()))
}

/// Seat the radius-r ball in the corner formed by `faces`.
///
/// `scale` is the march's model scale, used for the convergence bar exactly
/// as [`solve_station`] uses it (`1e-11·(1 + scale)`), so a corner solved
/// here is accurate to the same bar as the stations of the stripes that end
/// on it.
pub(in crate::blend) fn solve_corner_ball(
    faces: &[CornerFace<'_>],
    scale: f64,
) -> Result<CornerBall, String> {
    if faces.len() < 3 {
        return Err(format!(
            "corner ball: needs at least three faces at the vertex, got {}",
            faces.len()
        ));
    }
    let unknowns = 2 * faces.len();
    let equations = 3 * (faces.len() - 1);
    let mut x: Vec<f64> = faces
        .iter()
        .flat_map(|face| [face.seed[0], face.seed[1]])
        .collect();
    let tolerance = 1e-11 * (1.0 + scale);
    let step = 1e-7;
    let mut current = residual(faces, &x)?;
    for _ in 0..CORNER_ITERATIONS {
        if norm(&current) <= tolerance {
            break;
        }
        // Finite-difference Jacobian, one column per unknown — the same
        // scheme (and the same step) as the station Newton, so a corner and
        // the stripes ending on it are conditioned alike.
        let mut jacobian = vec![vec![0.0f64; unknowns]; equations];
        for column in 0..unknowns {
            let mut probe = x.clone();
            probe[column] += step;
            let probed = residual(faces, &probe)?;
            for row in 0..equations {
                jacobian[row][column] = (probed[row] - current[row]) / step;
            }
        }
        // Normal equations JᵀJ δ = Jᵀ r.  Square for N = 3 (the exact root)
        // and least-squares for N ≥ 4 (where the residual is the answer).
        let mut normal_matrix = vec![vec![0.0f64; unknowns]; unknowns];
        let mut normal_rhs = vec![0.0f64; unknowns];
        for row in 0..equations {
            for a in 0..unknowns {
                normal_rhs[a] += jacobian[row][a] * current[row];
                for b in 0..unknowns {
                    normal_matrix[a][b] += jacobian[row][a] * jacobian[row][b];
                }
            }
        }
        let delta = fit::solve_dense(normal_matrix, normal_rhs)
            .map_err(|error| format!("corner ball: Gauss-Newton is singular ({error})"))?;
        // Step halving on the residual norm: a corner seed sits a full radius
        // from its root, far enough that an undamped step can overshoot a
        // curved carrier's domain.
        let mut accepted = false;
        let mut damping = 1.0f64;
        for _ in 0..8 {
            let trial: Vec<f64> = x
                .iter()
                .zip(&delta)
                .map(|(value, correction)| value - damping * correction)
                .collect();
            if let Ok(trial_residual) = residual(faces, &trial) {
                if norm(&trial_residual) < norm(&current) {
                    x = trial;
                    current = trial_residual;
                    accepted = true;
                    break;
                }
            }
            damping *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let final_residual = norm(&current);
    let centers = centers(faces, &x)?;
    let center = centers
        .iter()
        .fold(Vec3::default(), |sum, point| sum.add(*point))
        .scale(1.0 / centers.len() as f64);
    let mut contacts = Vec::with_capacity(faces.len());
    for (index, face) in faces.iter().enumerate() {
        let sample = blend_offset(&face.face.surface).at(x[2 * index], x[2 * index + 1], face.rho)?;
        contacts.push(CornerContact {
            face_id: face.face.id,
            uv: [x[2 * index], x[2 * index + 1]],
            point: sample.source,
            normal: sample.normal,
        });
    }
    Ok(CornerBall {
        center,
        contacts,
        residual: final_residual,
    })
}

/// Build the solver input for the faces meeting at `corner`, seeding each
/// from the corner's own projection onto that face.
///
/// `faces` are the incident carriers in any order; `rho` is looked up per
/// face id from the stripes that end at this corner, so the ball is offset
/// on the same side as every stripe it will close.
pub(in crate::blend) fn corner_faces<'a>(
    faces: &[&'a FaceRecord],
    rho_of: &dyn Fn(u64) -> Option<f64>,
    corner: Vec3,
) -> Result<Vec<CornerFace<'a>>, String> {
    let mut prepared = Vec::with_capacity(faces.len());
    for face in faces {
        let rho = rho_of(face.id).ok_or_else(|| {
            format!(
                "corner ball: no stripe supplies a signed radius for face {}",
                face.id
            )
        })?;
        let projection = crate::project_point_to_surface(&face.surface, corner)?;
        prepared.push(CornerFace {
            face,
            rho,
            seed: [projection.u, projection.v],
        });
    }
    Ok(prepared)
}
