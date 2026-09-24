use super::*;

// ======================================================================
// §6.9.7 vertex ("star") blend — convex trihedral corner.
//
// When three edges meeting at a convex vertex are filleted, the rolling
// ball tangent to all three faces sits at ONE position and its sphere is
// the vertex patch (Golovanov §6.9.7). The patch is a spherical octant:
// with the sphere centered at C = P − r·Σ mᵢ (mᵢ = outward face normals),
// its three corners are the tangent points Tᵢ = C + r·mᵢ and its three
// boundary arcs are the tangent circles sphere∩cylinderᵢ, which lie in
// the coordinate planes through C (arc T₁T₂ = sphere∩(edge-3 cylinder),
// in the plane ⟂ m₃, etc.). Built as a sub-rectangle u∈[0,¼], v∈[½,1] of
// a sphere revolution with polar axis m₃ and x-axis m₁: the three arcs
// are exact iso-parameter curves (two meridians + the equator) and the
// v=1 pole is a degenerate edge at T₃. No booleans — the tangent
// surfaces are never intersected, only constructed and sewn.

/// Build ONLY the spherical-octant `FaceRecord` (plus its degenerate pole
/// edge) for a vertex blend, reusing the three EXISTING tangent vertices
/// `t_ids` and the three FRESH tangent-circle arc edges in `arcs`
/// (`(edge_id, start_vertex, end_vertex)`).  The loop, pcurves and
/// same_sense are exactly the spherical-octant construction described above;
/// each arc coedge's
/// `forward` is derived from the stored edge orientation so the octant use
/// is opposite to the fillet use (shared edge, two coedges, opposite sense).
pub(super) fn build_octant_face(
    center: Vec3,
    normals: [Vec3; 3],
    radius: f64,
    t_ids: [u64; 3],
    t_pts: [Vec3; 3],
    arcs: &[(u64, u64, u64)],
    name: Option<&str>,
    next_id: &mut dyn FnMut() -> u64,
) -> Result<(FaceRecord, EdgeRecord), String> {
    let [m1, m2, m3] = normals;
    if m1.cross(m2).dot(m3) < 0.5 {
        return Err("round_convex_corner: octant normals are not right-handed".into());
    }
    let half = std::f64::consts::FRAC_PI_2;
    let meridian = crate::make_arc(center, m1, m3, radius, -half, half)?;
    let surface = crate::make_revolution(center, m3, &meridian, std::f64::consts::TAU)?;

    let [t1, t2, t3] = t_ids;
    let e_pole = next_id();
    let pole_edge = EdgeRecord {
        id: e_pole,
        curve: crate::make_line(t_pts[2], t_pts[2])?,
        t0: 0.0,
        t1: 1.0,
        start_vertex_id: t3,
        end_vertex_id: t3,
        degenerate: true,
        name: None,
    };

    // For a needed traversal `from -> to`, locate the fresh arc joining that
    // pair and report whether its stored orientation already runs that way.
    let find = |from: u64, to: u64| -> Result<(u64, bool), String> {
        for &(eid, s, e) in arcs {
            if s == from && e == to {
                return Ok((eid, true));
            }
            if s == to && e == from {
                return Ok((eid, false));
            }
        }
        Err(format!(
            "round_convex_corner: no tangent arc between vertices {from} and {to}"
        ))
    };
    let (e12, f12) = find(t1, t2)?; // equator T1->T2  (arc ⟂ m3)
    let (e23, f23) = find(t2, t3)?; // meridian u=0.25 T2->T3 (arc ⟂ m1)
    let (e31, f31) = find(t3, t1)?; // meridian u=0    T3->T1 (arc ⟂ m2)

    let pl = |u0, v0, u1, v1| crate::sweep_topology::parameter_line(u0, v0, u1, v1);
    let loop_id = next_id();
    let coedges = vec![
        CoedgeRecord {
            id: next_id(),
            edge_id: e12,
            forward: f12,
            pcurve: pl(0.0, 0.5, 0.25, 0.5)?,
        },
        CoedgeRecord {
            id: next_id(),
            edge_id: e23,
            forward: f23,
            pcurve: pl(0.25, 0.5, 0.25, 1.0)?,
        },
        CoedgeRecord {
            id: next_id(),
            edge_id: e_pole,
            forward: true,
            pcurve: pl(0.25, 1.0, 0.0, 1.0)?,
        },
        CoedgeRecord {
            id: next_id(),
            edge_id: e31,
            forward: f31,
            pcurve: pl(0.0, 1.0, 0.0, 0.5)?,
        },
    ];

    let mid = surface.evaluate(0.125, 0.75)?;
    let surface_normal = surface.normal(0.125, 0.75)?;
    let outward = mid.sub(center).normalized()?;
    let same_sense = surface_normal.dot(outward) > 0.0;

    let face = FaceRecord {
        id: next_id(),
        surface,
        same_sense,
        loops: vec![LoopRecord {
            id: loop_id,
            coedges,
        }],
        name: name.map(|value| value.to_string()),
    };
    Ok((face, pole_edge))
}

/// General convex-corner vertex patch for a NON-orthogonal trihedral corner:
/// a spherical triangle on the ball (`center`, `radius`) bounded by the three
/// FRESH tangent-circle arcs in `arc_edges`.  Unlike [`build_octant_face`],
/// the three boundary arcs are great circles that are NOT iso-parameter curves
/// of any single revolution parameterization, so each coedge pcurve is FITTED
/// by projecting samples of its arc onto the sphere.  The revolution frame is
/// chosen to keep the whole patch OFF the u-seam and OFF both poles.
///
/// `outward` says which way the material lies: a CONVEX corner's ball sits
/// inside the material and the patch faces away from the centre; a CONCAVE
/// corner's ball sits in the air and the patch faces toward it.
pub(in crate::blend) fn build_general_corner_patch(
    center: Vec3,
    radius: f64,
    normals: &[Vec3],
    arc_edges: &[EdgeRecord],
    outward: bool,
    name: Option<&str>,
    next_id: &mut dyn FnMut() -> u64,
) -> Result<FaceRecord, String> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let n_faces = normals.len();
    if n_faces < 3 || arc_edges.len() != n_faces {
        return Err(format!(
            "build_general_corner_patch: expected N≥3 faces with one arc each, \
             found {n_faces} faces and {} arcs",
            arc_edges.len()
        ));
    }
    // Patch centre direction (radially outward through the middle of the
    // spherical N-gon); every tangent point Tᵢ = C + r·nᵢ lies within 90° of
    // it, so a seam placed opposite pc keeps the whole patch off u=0.
    let pc = normals
        .iter()
        .fold(Vec3::default(), |acc, n| acc.add(*n))
        .normalized()?;

    // Revolution frame: polar axis ⟂ pc so the patch straddles the equator
    // (away from both poles).  Among all such axes pick the one that keeps the
    // N tangent points AND the great-circle arc midpoints (normalize(nᵢ+nⱼ))
    // nearest the equator — minimizing the worst |d·polar|.  x-axis = -pc puts
    // the u=0 seam on the far side (patch lands near u≈0.5).
    let e0 = pc.perpendicular()?;
    let e1 = pc.cross(e0).normalized()?;
    let mut crit: Vec<Vec3> = normals.to_vec();
    for i in 0..n_faces {
        for j in (i + 1)..n_faces {
            if let Ok(mid) = normals[i].add(normals[j]).normalized() {
                crit.push(mid);
            }
        }
    }
    let mut best_phi = 0.0f64;
    let mut best_max = f64::INFINITY;
    for k in 0..360 {
        let phi = PI * k as f64 / 360.0;
        let polar = e0.scale(phi.cos()).add(e1.scale(phi.sin()));
        let worst = crit
            .iter()
            .fold(0.0f64, |acc, d| acc.max(d.dot(polar).abs()));
        if worst < best_max {
            best_max = worst;
            best_phi = phi;
        }
    }
    let polar = e0
        .scale(best_phi.cos())
        .add(e1.scale(best_phi.sin()))
        .normalized()?;
    let x_axis = pc.scale(-1.0);
    let meridian = crate::make_arc(center, x_axis, polar, radius, -FRAC_PI_2, FRAC_PI_2)?;
    let surface = crate::make_revolution(center, polar, &meridian, TAU)?;

    // Chain the three arcs into a closed loop traversed OPPOSITE to the fillet
    // use: each arc is stored start->end and used forward=true by its fillet,
    // so the patch reuses it forward=false (end->start).  Following reversed
    // arcs, each successor's stored end meets the current arc's stored start.
    let mut order = vec![0usize];
    let mut used = vec![false; n_faces];
    used[0] = true;
    let mut chain_vertex = arc_edges[0].start_vertex_id;
    for _ in 0..(n_faces - 1) {
        let next = (0..n_faces)
            .find(|&j| !used[j] && arc_edges[j].end_vertex_id == chain_vertex)
            .ok_or("build_general_corner_patch: tangent arcs do not form a closed N-gon")?;
        used[next] = true;
        order.push(next);
        chain_vertex = arc_edges[next].start_vertex_id;
    }
    if chain_vertex != arc_edges[0].end_vertex_id {
        return Err("build_general_corner_patch: tangent arc loop is not closed".into());
    }

    // Fit each coedge pcurve by projecting samples of the arc (taken in the
    // coedge traversal direction, forward=false) onto the sphere.  Sphere
    // projection is exact, so surface(pcurve(t)) reproduces the arc.  ACUTE
    // corners produce fresh tangent arcs that sweep well past 90° (the face
    // normals are nearly anti-parallel), and a degree-3 uv-curve through only
    // 9 samples then bows off the true arc image by > the validator's pcurve
    // limit.  Sample DENSELY (33 stations) so the fitted pcurve tracks the
    // long arc to well inside tolerance at any convex angle.
    const PATCH_PCURVE_SAMPLES: usize = 33;
    let mut coedges = Vec::with_capacity(n_faces);
    for &ai in &order {
        let edge = &arc_edges[ai];
        let mut points = Vec::with_capacity(PATCH_PCURVE_SAMPLES);
        let mut params = Vec::with_capacity(PATCH_PCURVE_SAMPLES);
        let mut previous_u: Option<f64> = None;
        for k in 0..PATCH_PCURVE_SAMPLES {
            let fraction = k as f64 / (PATCH_PCURVE_SAMPLES - 1) as f64;
            // forward = false: fraction 0 -> t1 (end vertex), 1 -> t0 (start).
            let t = edge.t1 - (edge.t1 - edge.t0) * fraction;
            let p = edge.curve.evaluate(t)?;
            let projection = crate::project_point_to_surface(&surface, p)?;
            let mut u = projection.u;
            let v = projection.v;
            // Keep u continuous across the sampled arc (no spurious 2π jump).
            if let Some(pu) = previous_u {
                while u - pu > 0.5 {
                    u -= 1.0;
                }
                while u - pu < -0.5 {
                    u += 1.0;
                }
            }
            previous_u = Some(u);
            points.push(Vec4 {
                x: u,
                y: v,
                z: 0.0,
                w: 1.0,
            });
            params.push(fraction);
        }
        let pcurve = crate::fit::interpolate_homogeneous(&points, 3, &params)?;
        coedges.push(CoedgeRecord {
            id: next_id(),
            edge_id: edge.id,
            forward: false,
            pcurve,
        });
    }

    // same_sense so the face normal points radially away from C for a convex
    // corner, toward it for a concave one.
    let centre_projection =
        crate::project_point_to_surface(&surface, center.add(pc.scale(radius)))?;
    let surface_normal = surface.normal(centre_projection.u, centre_projection.v)?;
    let same_sense = (surface_normal.dot(pc) > 0.0) == outward;

    let loop_id = next_id();
    Ok(FaceRecord {
        id: next_id(),
        surface,
        same_sense,
        loops: vec![LoopRecord {
            id: loop_id,
            coedges,
        }],
        name: name.map(|value| value.to_string()),
    })
}

/// The CHAMFER counterpart of [`build_general_corner_patch`]: the planar facet
/// a chamfered star closes with.
///
/// §6.11 keeps a chamfered vertex sharp only where the bevels meet in a seam.
/// At a STAR — every edge at the vertex selected — each stripe stops at the
/// station where its own rolling ball IS the corner ball, and there its cross
/// section is the CHORD between the ball's two tangency points rather than the
/// arc between them.  The N chords therefore run tangency point to tangency
/// point around the ball, one per face, and close an N-gon whose vertices all
/// lie on the ball.  For N = 3 that N-gon is a triangle, so it is planar by
/// construction and the facet is the one plane through the three tangency
/// points; the chords lie in it exactly, which is what lets the facet be sewn
/// to the stripes with no intersection anywhere.
///
/// N ≥ 4 is refused: four points on a sphere are not coplanar in general, and
/// the fill that closes a non-planar N-gon is the question
/// `general_star.rs` answers for fillets, not this one.
pub(in crate::blend) fn build_chamfer_corner_facet(
    center: Vec3,
    normals: &[Vec3],
    chord_edges: &[EdgeRecord],
    outward: bool,
    name: Option<&str>,
    next_id: &mut dyn FnMut() -> u64,
) -> Result<FaceRecord, String> {
    let n_faces = normals.len();
    if n_faces != 3 || chord_edges.len() != n_faces {
        return Err(format!(
            "blend network: a chamfered star closes with a planar facet, which is only \
             determined for 3 faces — this vertex has {n_faces} faces and {} sections",
            chord_edges.len()
        ));
    }
    // Outward direction through the middle of the corner, exactly as the
    // spherical patch uses it: every tangency point lies within 90 degrees of
    // it, so it fixes the facet's sense.
    let pc = normals
        .iter()
        .fold(Vec3::default(), |acc, n| acc.add(*n))
        .normalized()?;

    // Chain the chords into a closed loop traversed OPPOSITE to the stripes'
    // use, the same walk `build_general_corner_patch` makes over the arcs.
    let mut order = vec![0usize];
    let mut used = vec![false; n_faces];
    used[0] = true;
    let mut chain_vertex = chord_edges[0].start_vertex_id;
    for _ in 0..(n_faces - 1) {
        let next = (0..n_faces)
            .find(|&j| !used[j] && chord_edges[j].end_vertex_id == chain_vertex)
            .ok_or("blend network: the chamfer facet's sections do not form a closed N-gon")?;
        used[next] = true;
        order.push(next);
        chain_vertex = chord_edges[next].start_vertex_id;
    }
    if chain_vertex != chord_edges[0].end_vertex_id {
        return Err("blend network: the chamfer facet's section loop is not closed".into());
    }

    // The plane through the three tangency points. They are `center + r·nᵢ`
    // for a common r, so the plane is well conditioned exactly when the three
    // normals are (which the ball solve has already established).
    let corner_point = |index: usize| -> Result<Vec3, String> {
        let edge = &chord_edges[index];
        edge.curve.evaluate(edge.t0)
    };
    let [a, b, c] = [corner_point(0)?, corner_point(1)?, corner_point(2)?];
    let u_dir = b.sub(a).normalized()?;
    let normal = u_dir.cross(c.sub(a)).normalized()?;
    let v_dir = normal.cross(u_dir).normalized()?;
    // A patch big enough to hold the facet with room to spare, laid out so the
    // facet lands inside the parameter square (the same set-back `sew_cap`
    // uses for its bulkhead).
    let reach = [a, b, c]
        .iter()
        .fold(0.0f64, |worst, point| worst.max(point.sub(a).length()))
        .max(1e-6)
        * 4.0;
    let origin = a.sub(u_dir.scale(reach)).sub(v_dir.scale(reach));
    let plane = crate::make_plane(origin, u_dir, v_dir, 2.0 * reach, 2.0 * reach)?;
    let uv_of = |point: Vec3| -> Result<(f64, f64), String> {
        let projection = crate::project_point_to_surface(&plane, point)?;
        Ok((projection.u, projection.v))
    };

    // A chord is a straight line and the plane's parameterisation is affine,
    // so each pcurve is the straight uv segment between the chord's ends —
    // exact, with nothing to fit. `forward = false` walks stored end -> start.
    let mut coedges = Vec::with_capacity(n_faces);
    for &index in &order {
        let edge = &chord_edges[index];
        let (u0, v0) = uv_of(edge.curve.evaluate(edge.t1)?)?;
        let (u1, v1) = uv_of(edge.curve.evaluate(edge.t0)?)?;
        coedges.push(CoedgeRecord {
            id: next_id(),
            edge_id: edge.id,
            forward: false,
            pcurve: crate::sweep_topology::parameter_line(u0, v0, u1, v1)?,
        });
    }

    // same_sense so the face normal points away from the ball's centre at a
    // convex corner and toward it at a concave one — the patch's own rule.
    let plane_normal = plane.normal(0.5, 0.5)?;
    let same_sense = (plane_normal.dot(pc) > 0.0) == outward;
    let _ = center;

    let loop_id = next_id();
    Ok(FaceRecord {
        id: next_id(),
        surface: plane,
        same_sense,
        loops: vec![LoopRecord {
            id: loop_id,
            coedges,
        }],
        name: name.map(|value| value.to_string()),
    })
}
