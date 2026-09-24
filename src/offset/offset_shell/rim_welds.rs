use super::*;

/// A closed edge sampled as a circle: center, unit axis, radius. `None` when
/// the samples do not lie on a common circle within tolerance.
pub(super) struct RimCircle {
    center: Vec3,
    axis: Vec3,
    radius: f64,
}

pub(super) fn fit_rim_circle(edge: &EdgeRecord, tolerance: f64) -> Result<Option<RimCircle>, String> {
    let samples = 48;
    let mut points = Vec::with_capacity(samples);
    for index in 0..samples {
        let fraction = index as f64 / samples as f64;
        points.push(
            edge.curve
                .evaluate(edge.t0 + (edge.t1 - edge.t0) * fraction)?,
        );
    }
    let count = points.len() as f64;
    let center = points
        .iter()
        .fold(Vec3::default(), |sum, point| sum.add(*point))
        .scale(1.0 / count);
    // Plane normal from the sample covariance's smallest-spread direction,
    // approximated by the average of consecutive radial cross products.
    let mut axis = Vec3::default();
    for index in 0..points.len() {
        let a = points[index].sub(center);
        let b = points[(index + 1) % points.len()].sub(center);
        axis = axis.add(a.cross(b));
    }
    // A degenerate closed edge (pole placeholder) has all samples coincident:
    // the swept-area axis is zero. That is "not a circle", not a hard error.
    let Ok(axis) = axis.normalized() else {
        return Ok(None);
    };
    let radius = points
        .iter()
        .map(|point| point.sub(center).length())
        .sum::<f64>()
        / count;
    if radius <= tolerance {
        return Ok(None);
    }
    for point in &points {
        let radial = point.sub(center);
        if radial.dot(axis).abs() > tolerance.max(radius * 1e-3)
            || (radial.length() - radius).abs() > tolerance.max(radius * 1e-3)
        {
            return Ok(None);
        }
    }
    Ok(Some(RimCircle {
        center,
        axis,
        radius,
    }))
}

/// Weld an offset rim that is NOT coplanar with any opening cap into a ruled
/// (conical-frustum) opening-wall band.
///
/// `weld_coplanar_orphan_rims` handles the perpendicular-wall case where the
/// offset rim lands in the plane of a flat opening cap.  When the retained
/// side face is oblique (a truncated cone), the offset rim leaves that plane:
/// it is a smaller coaxial circle sitting BELOW the opening.  The offset
/// pipeline then mis-builds the opening wall as a flat disk that caps the
/// opening (bounded only by the source rim) and leaves the offset rim dangling
/// one-use.  The correct opening wall is the ruled band between the source rim
/// (radius R on the opening plane) and the offset rim (radius r below it) — an
/// exact conical frustum through the two coaxial circles.
///
/// Rebuild that mis-built flat cap in place: swap its plane for the frustum
/// surface through both rims and add the offset rim as its inner loop, so both
/// rims become two-use and the shell closes watertight.  Reuses the existing
/// `boundary_pcurve` fitter; nothing else in the assembly is disturbed.
///
/// The `forward` flag for a coedge is derived from geometry, not assumed: a
/// coedge whose `pcurve` (mapped through the surface) traces its edge's 3D
/// curve in the SAME direction is `forward = true`, otherwise `false`. When a
/// pcurve's direction is pinned by loop connectivity (a full-circle rim isoline
/// can run either u-way), the flag must follow the resulting 3D walk or
/// `validate()` flags the coedge as pcurve-inconsistent with its edge.
pub(super) fn isoline_forward(
    pcurve: &crate::NurbsCurve,
    surface: &crate::NurbsSurface,
    curve: &crate::NurbsCurve,
) -> Result<bool, String> {
    let mut deviation_forward = 0.0f64;
    let mut deviation_reversed = 0.0f64;
    for sample in 0..=32 {
        let fraction = sample as f64 / 32.0;
        let uv = pcurve.evaluate(fraction)?;
        let on_surface = surface.evaluate(uv.x, uv.y)?;
        deviation_forward =
            deviation_forward.max(on_surface.sub(curve.evaluate(fraction)?).length());
        deviation_reversed =
            deviation_reversed.max(on_surface.sub(curve.evaluate(1.0 - fraction)?).length());
    }
    Ok(deviation_forward <= deviation_reversed)
}

/// Plane frame of a geometrically planar surface: `(origin, unit normal)`,
/// or `None` when the sampled grid leaves one plane by more than `tolerance`.
/// Works for planes in any representation — structural `is_affine` misses
/// planes represented as surfaces of revolution (a revolved radial line is a
/// perfectly flat disk but rational degree (2,1)), which is exactly how
/// revolve-built opening caps arrive here.
pub(super) fn planar_surface_frame(
    surface: &crate::NurbsSurface,
    tolerance: f64,
) -> Result<Option<(Vec3, Vec3)>, String> {
    let [u0, u1] = surface.domain_u()?;
    let [v0, v1] = surface.domain_v()?;
    let origin = surface.evaluate((u0 + u1) * 0.5, (v0 + v1) * 0.5)?;
    let mut points = Vec::new();
    for iu in 0..=6 {
        for iv in 0..=6 {
            let u = u0 + (u1 - u0) * iu as f64 / 6.0;
            let v = v0 + (v1 - v0) * iv as f64 / 6.0;
            points.push(surface.evaluate(u, v)?);
        }
    }
    // A summed cross product cancels on rotationally sampled grids (the
    // per-pair crosses alternate sign about the axis); take the single
    // largest-magnitude cross as the normal estimate instead.
    let mut normal = Vec3::default();
    let mut best = 0.0f64;
    for first in &points {
        for second in &points {
            let cross = first.sub(origin).cross(second.sub(origin));
            let magnitude = cross.length();
            if magnitude > best {
                best = magnitude;
                normal = cross;
            }
        }
    }
    let Ok(normal) = normal.normalized() else {
        return Ok(None);
    };
    if points
        .iter()
        .all(|point| point.sub(origin).dot(normal).abs() <= tolerance)
    {
        Ok(Some((origin, normal)))
    } else {
        Ok(None)
    }
}

/// Geometric planarity: whether every sampled surface point lies on one plane
/// within `tolerance`.
pub(super) fn surface_is_planar(surface: &crate::NurbsSurface, tolerance: f64) -> Result<bool, String> {
    if surface.is_affine()? {
        return Ok(true);
    }
    Ok(planar_surface_frame(surface, tolerance)?.is_some())
}

/// The mis-built opening cap a ruled weld may rebuild: a geometrically planar
/// face whose single loop consists of exactly ONE closed non-degenerate rim
/// plus only face-local plumbing (seam edges used twice by this loop and
/// degenerate pole placeholders). A plane-carried disk has just the rim; a
/// revolve-carried disk arrives as [rim, seam, pole, seam].
fn cap_face_rim(
    face: &FaceRecord,
    edges_by_id: &HashMap<u64, &EdgeRecord>,
    global_use_counts: &HashMap<u64, usize>,
) -> Option<u64> {
    if face.loops.len() != 1 {
        return None;
    }
    let mut local_counts = HashMap::<u64, usize>::default();
    for coedge in &face.loops[0].coedges {
        *local_counts.entry(coedge.edge_id).or_default() += 1;
    }
    let mut rim = None;
    for (&edge_id, &local) in &local_counts {
        let edge = edges_by_id.get(&edge_id)?;
        let global = global_use_counts.get(&edge_id).copied().unwrap_or(0);
        if edge.degenerate {
            // Pole placeholder: fine as long as it is entirely this face's.
            if global != local {
                return None;
            }
            continue;
        }
        if local == 2 {
            // Seam: both uses must be this loop's, or a rebuild orphans it.
            if global != 2 {
                return None;
            }
            continue;
        }
        if local == 1 && edge.start_vertex_id == edge.end_vertex_id {
            if rim.is_some() {
                return None;
            }
            rim = Some(edge_id);
            continue;
        }
        return None;
    }
    rim
}

pub(super) fn weld_ruled_offset_rims(solid: &mut BrepSolid, tolerance: f64) -> Result<usize, String> {
    let use_counts = crate::topology::edge_use_counts(solid);
    let orphan_ids = solid
        .edges
        .iter()
        .filter(|edge| {
            use_counts.get(&edge.id).copied().unwrap_or(0) == 1
                && edge.start_vertex_id == edge.end_vertex_id
        })
        .map(|edge| edge.id)
        .collect::<Vec<_>>();
    let mut welded = 0usize;
    let mut shell_unions: Vec<(usize, usize)> = Vec::new();
    for edge_id in orphan_ids {
        // A previous cap rebuild may have purged this candidate (a pole
        // placeholder that was face-local plumbing of a rebuilt disk).
        let Some(offset_edge) = solid.edges.iter().find(|edge| edge.id == edge_id).cloned() else {
            continue;
        };
        let Some(offset_circle) = fit_rim_circle(&offset_edge, tolerance)? else {
            continue;
        };
        // The single existing use fixes the manifold direction: the band must
        // trace the shared offset rim the opposite way.
        let owner_shell = solid
            .shells
            .iter()
            .enumerate()
            .find_map(|(shell_index, shell)| {
                shell
                    .faces
                    .iter()
                    .flat_map(|face| &face.loops)
                    .flat_map(|loop_record| &loop_record.coedges)
                    .find(|coedge| coedge.edge_id == edge_id)
                    .map(|_| shell_index)
            })
            .ok_or_else(|| "offset_shell: ruled rim has no use".to_string())?;
        // Locate the mis-built flat opening cap: a geometrically planar face
        // whose single loop is one closed circle (the source rim, plus only
        // face-local seam/pole plumbing) coaxial with — but not coplanar
        // with — the offset rim. Several coaxial planar caps can qualify
        // (e.g. the cavity floor); the opening cap is the CLOSEST one along
        // the axis — the offset wall's thin rim, not the far floor.
        let current_use_counts = crate::topology::edge_use_counts(solid);
        let edges_by_id = solid
            .edges
            .iter()
            .map(|edge| (edge.id, edge))
            .collect::<HashMap<_, _>>();
        let mut mate: Option<(usize, usize, EdgeRecord, RimCircle, f64)> = None;
        for (shell_index, shell) in solid.shells.iter().enumerate() {
            for (face_index, face) in shell.faces.iter().enumerate() {
                if !surface_is_planar(&face.surface, tolerance).unwrap_or(false) {
                    os_debug!("RULED mate reject face {}: not planar", face.id);
                    continue;
                }
                let Some(source_edge_id) = cap_face_rim(face, &edges_by_id, &current_use_counts)
                else {
                    os_debug!("RULED mate reject face {}: no single cap rim", face.id);
                    continue;
                };
                if source_edge_id == edge_id {
                    continue;
                }
                let Some(source_edge) = solid
                    .edges
                    .iter()
                    .find(|edge| {
                        edge.id == source_edge_id && edge.start_vertex_id == edge.end_vertex_id
                    })
                    .cloned()
                else {
                    continue;
                };
                let Some(source_circle) = fit_rim_circle(&source_edge, tolerance)? else {
                    continue;
                };
                if offset_circle.axis.dot(source_circle.axis).abs() < 0.999 {
                    continue;
                }
                let between = source_circle.center.sub(offset_circle.center);
                let axial = between.dot(offset_circle.axis);
                // Coaxial: centre offset is purely along the shared axis.
                if between.sub(offset_circle.axis.scale(axial)).length() > tolerance.max(1e-4) {
                    continue;
                }
                // Non-coplanar: an in-plane rim is the coplanar-weld case.
                if axial.abs() <= tolerance.max(1e-4) {
                    continue;
                }
                if mate
                    .as_ref()
                    .is_none_or(|(_, _, _, _, best)| axial.abs() < *best)
                {
                    mate = Some((
                        shell_index,
                        face_index,
                        source_edge,
                        source_circle,
                        axial.abs(),
                    ));
                }
            }
        }
        let Some((cap_shell, cap_face, source_edge, source_circle, _)) = mate else {
            os_debug!("RULED skip: no mate cap for orphan {edge_id}");
            continue;
        };
        os_debug!(
            "RULED mate found: cap face shell {cap_shell} idx {cap_face}, source edge {}",
            source_edge.id
        );
        // The source rim must be a genuine circle NURBS: the band reuses its
        // rotational parametrization so the analytic isoline pcurves are exact.
        if source_edge.curve.degree < 2 {
            os_debug!(
                "RULED skip: source rim degree {} < 2",
                source_edge.curve.degree
            );
            continue;
        }
        let owner_surface = solid.shells[owner_shell]
            .faces
            .iter()
            .find(|face| {
                face.loops
                    .iter()
                    .flat_map(|loop_record| &loop_record.coedges)
                    .any(|coedge| coedge.edge_id == edge_id)
            })
            .map(|face| face.surface.clone())
            .ok_or_else(|| "offset_shell: ruled rim lost its owner".to_string())?;
        // Pin the owner's isoline pcurve to its loop neighbors. A full circle
        // admits either u-direction, but choosing the wrong one can collapse
        // the trim polygon and halve the volume integral.
        let (prev_end, next_start) = crate::topology::coedge_neighbor_endpoints(
            solid.shells[owner_shell].faces.iter().flat_map(|face| &face.loops),
            edge_id,
        )
        .ok_or_else(|| "offset_shell: ruled rim lost its owner loop".to_string())?;
        // Both endpoints sit on the rim's v-isoline; a straight parameter-space
        // line between them IS that isoline traced in the loop's direction.
        if (prev_end.y - next_start.y).abs() > tolerance.max(1e-4) {
            os_debug!("RULED skip: owner rim neighbours not on a common isoline");
            continue;
        }
        let owner_pcurve = crate::make_line(
            Vec3::new(prev_end.x, prev_end.y, 0.0),
            Vec3::new(next_start.x, next_start.y, 0.0),
        )?;
        // Verify the isoline actually traces the re-analyticized rim circle.
        let mut worst = 0.0f64;
        for sample in 0..=32 {
            let fraction = sample as f64 / 32.0;
            let uv = owner_pcurve.evaluate(fraction)?;
            let on_surface = owner_surface.evaluate(uv.x, uv.y)?;
            let radial = on_surface.sub(offset_circle.center);
            let axial = radial.dot(offset_circle.axis);
            let planar = radial.sub(offset_circle.axis.scale(axial)).length();
            worst = worst.max(axial.abs().max((planar - offset_circle.radius).abs()));
        }
        if worst > tolerance.max(1e-3) {
            os_debug!("RULED skip: neighbour-pinned owner pcurve off rim by {worst}");
            continue;
        }
        // Re-analyticize the degenerate offset rim. PREFERRED: the owner
        // surface's own v-isoline at the pinned rim height — exact both on the
        // owner (it IS the surface's boundary curve) and in parametrization
        // (the offset carrier copies the source face's rotational knots), so
        // the band it spans is the exact normal-ruled frustum. The fitted-
        // polyline circle is only a similarity witness: its sampled radius
        // carries the polyline's chord sag (~1e-4 relative), which is exactly
        // the wall-volume error the exact isoline removes. FALLBACK: the
        // source control net scaled radially to the fitted circle, for owners
        // whose u-structure does not match the source rim's.
        let scaled_offset_curve = || -> Result<crate::NurbsCurve, String> {
            let radius_ratio = offset_circle.radius / source_circle.radius;
            let offset_controls = source_edge
                .curve
                .control_points
                .iter()
                .map(|control| {
                    let point = control.point()?;
                    let placed = offset_circle
                        .center
                        .add(point.sub(source_circle.center).scale(radius_ratio));
                    Ok(crate::Vec4::from_point(placed, control.w))
                })
                .collect::<Result<Vec<_>, String>>()?;
            crate::NurbsCurve::new(
                source_edge.curve.degree,
                source_edge.curve.knots.clone(),
                offset_controls,
            )
        };
        let iso_offset_curve = owner_surface.iso_curve_v(prev_end.y).ok().filter(|iso| {
            iso.degree == source_edge.curve.degree
                && iso.knots.len() == source_edge.curve.knots.len()
                && iso
                    .knots
                    .iter()
                    .zip(&source_edge.curve.knots)
                    .all(|(a, b)| (a - b).abs() <= 1e-9)
                && iso.control_points.len() == source_edge.curve.control_points.len()
        });
        let offset_curve = match iso_offset_curve {
            Some(iso) => {
                // The band zips source and offset controls index by index, so
                // the isoline must run angularly in step with the source rim
                // (same start, same direction) — otherwise the ruled surface
                // would twist. Verify against the similarity image.
                let reference = scaled_offset_curve()?;
                let mut aligned = true;
                for sample in 0..=8 {
                    let fraction = sample as f64 / 8.0;
                    if iso
                        .evaluate(fraction)?
                        .sub(reference.evaluate(fraction)?)
                        .length()
                        > tolerance.max(1e-3) * 4.0
                    {
                        aligned = false;
                        break;
                    }
                }
                if aligned {
                    iso
                } else {
                    os_debug!(
                        "RULED: owner isoline out of step with source rim; using scaled fallback"
                    );
                    reference
                }
            }
            None => scaled_offset_curve()?,
        };
        // Verify the reconstructed rim actually lies on its owner face so the
        // re-analyticized curve stays consistent with the offset wall.
        if !edge_on_surface(
            &EdgeRecord {
                curve: offset_curve.clone(),
                t0: 0.0,
                t1: 1.0,
                ..offset_edge.clone()
            },
            &owner_surface,
            tolerance,
        )? {
            os_debug!("RULED skip: re-analyticized rim not on owner surface");
            continue;
        }
        // Build the frustum band as the ruled tensor between the two circles so
        // v=0 is the source rim and v=1 the offset rim, both exact isolines.
        let band_controls = source_edge
            .curve
            .control_points
            .iter()
            .zip(&offset_curve.control_points)
            .map(|(source_control, offset_control)| vec![*source_control, *offset_control])
            .collect::<Vec<_>>();
        let band = crate::NurbsSurface::new(
            source_edge.curve.degree,
            1,
            source_edge.curve.knots.clone(),
            vec![0.0, 0.0, 1.0, 1.0],
            band_controls,
        )?;
        let [band_u0, band_u1] = band.domain_u()?;
        let seam = band.iso_curve_u(band_u0)?;
        // Build the band as a fixed, internally-coherent rectangle (source rim
        // walked +u at v=0, offset rim walked -u at v=1, seam up then down),
        // exactly like make_cone_brep's side face. The source and offset skins
        // were assembled as independent components whose global normal senses
        // are not yet reconciled; a re-run of orient_open_solid_faces after all
        // rims are two-use makes the joined manifold coherent, and a final
        // signed-volume check restores outward normals.
        let ccw = true;
        let next_loop_id = solid.shells[cap_shell].faces[cap_face]
            .loops
            .iter()
            .map(|loop_record| loop_record.id)
            .max()
            .unwrap_or(0)
            + 1;
        let mut next_coedge_id = solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .flat_map(|face| &face.loops)
            .flat_map(|loop_record| &loop_record.coedges)
            .map(|coedge| coedge.id)
            .max()
            .unwrap_or(0)
            + 1;
        let seam_edge_id = solid.edges.iter().map(|edge| edge.id).max().unwrap_or(0) + 1;
        // Straight parameter-space isolines mirroring make_cone_brep's side
        // face rectangle: v=0 source rim, u=u1 seam up, v=1 offset rim, u=u0
        // seam down. The CW variant is the mirror image.
        let line = |u0: f64, v0: f64, u1: f64, v1: f64| -> Result<crate::NurbsCurve, String> {
            crate::make_line(Vec3::new(u0, v0, 0.0), Vec3::new(u1, v1, 0.0))
        };
        let (source_pcurve, seam_up_pcurve, offset_pcurve, seam_down_pcurve) = if ccw {
            (
                line(band_u0, 0.0, band_u1, 0.0)?,
                line(band_u1, 0.0, band_u1, 1.0)?,
                line(band_u1, 1.0, band_u0, 1.0)?,
                line(band_u0, 1.0, band_u0, 0.0)?,
            )
        } else {
            (
                line(band_u1, 0.0, band_u0, 0.0)?,
                line(band_u0, 0.0, band_u0, 1.0)?,
                line(band_u0, 1.0, band_u1, 1.0)?,
                line(band_u1, 1.0, band_u1, 0.0)?,
            )
        };
        // Seam edge joins the two rims' seam vertices along the u0 generatrix;
        // both rectangles walk it up (forward) then down (backward).
        let seam_start_vertex = source_edge.start_vertex_id;
        let seam_end_vertex = offset_edge.start_vertex_id;
        // The rim coedges' `forward` follows the 3D walk their (fixed-direction)
        // band isolines produce, so each stays consistent with its edge curve.
        let source_curve_forward = isoline_forward(&source_pcurve, &band, &source_edge.curve)?;
        let offset_curve_forward = isoline_forward(&offset_pcurve, &band, &offset_curve)?;
        let mut coedges = Vec::new();
        let mut push_coedge = |edge_id: u64, forward: bool, pcurve: crate::NurbsCurve| {
            coedges.push(CoedgeRecord {
                id: next_coedge_id,
                edge_id,
                forward,
                pcurve,
            });
            next_coedge_id += 1;
        };
        if debug_enabled() {
            for (label, pcurve) in [("source", &source_pcurve), ("offset", &offset_pcurve)] {
                let spin = (|| -> Result<f64, String> {
                    let [d0, d1] = pcurve.domain()?;
                    let a = pcurve.evaluate(d0 + (d1 - d0) * 0.45)?;
                    let b = pcurve.evaluate(d0 + (d1 - d0) * 0.55)?;
                    Ok(band.evaluate(a.x, a.y)?.cross(band.evaluate(b.x, b.y)?).z)
                })()
                .unwrap_or(f64::NAN);
                os_debug!("RULED band {label} pcurve spin {:+.3e}", spin);
            }
            os_debug!(
                "RULED source_forward={source_curve_forward} offset_forward={offset_curve_forward}"
            );
        }
        if ccw {
            push_coedge(source_edge.id, source_curve_forward, source_pcurve);
            push_coedge(seam_edge_id, true, seam_up_pcurve);
            push_coedge(edge_id, offset_curve_forward, offset_pcurve);
            push_coedge(seam_edge_id, false, seam_down_pcurve);
        } else {
            push_coedge(seam_edge_id, true, seam_up_pcurve);
            push_coedge(edge_id, offset_curve_forward, offset_pcurve);
            push_coedge(seam_edge_id, false, seam_down_pcurve);
            push_coedge(source_edge.id, source_curve_forward, source_pcurve);
        }
        // Commit: rebuild the flat cap as the frustum band, re-analyticize the
        // offset rim, refit the owner pcurve, and register the seam edge.
        let face = &mut solid.shells[cap_shell].faces[cap_face];
        face.surface = band.clone();
        // The cap's face-local plumbing (a revolve-carried disk's seam and
        // pole placeholder) dies with the old loop; purge the records so the
        // Euler bookkeeping does not count edges no face references anymore.
        let retired_edge_ids = face
            .loops
            .iter()
            .flat_map(|loop_record| &loop_record.coedges)
            .map(|coedge| coedge.edge_id)
            .filter(|retired| *retired != source_edge.id && *retired != edge_id)
            .collect::<HashSet<_>>();
        face.loops = vec![LoopRecord {
            id: next_loop_id,
            coedges,
        }];
        if parameter_space_area(face)? < 0.0 {
            face.same_sense = !face.same_sense;
        }
        solid
            .edges
            .retain(|record| !retired_edge_ids.contains(&record.id));
        // The neighbour-pinned owner pcurve traces the re-analyticized rim in
        // the loop's direction; its `forward` must follow that same 3D walk.
        // Rewrite ONLY the pre-existing owner use: when the cap lives in the
        // same shell as the owner (multi-opening shells are already one
        // connected component here), an unfiltered sweep would clobber the
        // band's own freshly built rim coedge with a pcurve that lives in the
        // OWNER surface's parameter space.
        let owner_forward = isoline_forward(&owner_pcurve, &owner_surface, &offset_curve)?;
        for (face_index, face) in solid.shells[owner_shell].faces.iter_mut().enumerate() {
            if owner_shell == cap_shell && face_index == cap_face {
                continue;
            }
            for coedge in face
                .loops
                .iter_mut()
                .flat_map(|loop_record| &mut loop_record.coedges)
                .filter(|coedge| coedge.edge_id == edge_id)
            {
                coedge.pcurve = owner_pcurve.clone();
                coedge.forward = owner_forward;
            }
        }
        if let Some(record) = solid.edges.iter_mut().find(|record| record.id == edge_id) {
            record.curve = offset_curve.clone();
            record.degenerate = false;
        }
        solid.edges.push(EdgeRecord {
            id: seam_edge_id,
            curve: seam,
            t0: 0.0,
            t1: 1.0,
            start_vertex_id: seam_start_vertex,
            end_vertex_id: seam_end_vertex,
            degenerate: false,
            name: None,
        });
        if owner_shell != cap_shell {
            shell_unions.push((owner_shell, cap_shell));
        }
        welded += 1;
    }
    merge_connected_shells(solid, shell_unions);
    if welded > 0 {
        // The bands are built with a fixed internal winding; now that every rim
        // is two-use, make the shared-edge coedge directions coherent so the
        // finalize validation accepts the joined manifold. Normal-convention
        // coherence (same_sense) is repaired on the finalized solid.
        orient_open_solid_faces(solid)?;
    }
    Ok(welded)
}

/// Weld PAIRS of coaxial, coplanar orphan rims into a NEW planar annulus face.
///
/// A through-hole shelled at both of its openings leaves the retained hole
/// wall and its offset image each ending on a dangling rim in every opening
/// plane. Neither single-rim strategy applies there: no wall fragment covers
/// the hole's surroundings (the wall carriers only produce the opening's
/// border frame), so there is no containing planar face to hole
/// (`weld_coplanar_orphan_rims`) and no mis-built cap to rebuild
/// (`weld_ruled_offset_rims`). The missing geometry is exactly the flat
/// annulus between the two rims — the through-hole wall's exposed thickness.
///
/// Pairing: two one-use closed rims qualify when they are circles on a common
/// axis, lie in a common plane (an out-of-plane mate would need a twisted
/// band — refused, honest), and do not already bound a common face (the two
/// ends of one tube must never be capped against each other). Ambiguity is
/// resolved smallest-radial-gap-first, which picks each rim's true wall mate
/// (the source/offset images differ by exactly the shell thickness).
///
/// Orientation: each rim is traced opposite to its single existing use. The
/// two owning components were assembled independently, so their senses can
/// disagree — detected as the two loops winding the SAME way in the annulus
/// plane, which would corrupt the trim, not just the orientation. When the
/// owners live in different shell components the inner rim's whole component
/// is flipped (legal: a floating component's global sense is arbitrary until
/// first joined); a same-component winding clash is a genuine inconsistency
/// and skips the pair, leaving the honesty gate to refuse the shell.
pub(super) fn weld_coplanar_rim_pair_annuli(
    solid: &mut BrepSolid,
    face_images: &mut Vec<OffsetShellFaceImageRecord>,
    tolerance: f64,
) -> Result<usize, String> {
    let use_counts = crate::topology::edge_use_counts(solid);
    let mut orphans = Vec::new();
    for edge in &solid.edges {
        if use_counts.get(&edge.id).copied().unwrap_or(0) != 1
            || edge.start_vertex_id != edge.end_vertex_id
        {
            continue;
        }
        if let Some(circle) = fit_rim_circle(edge, tolerance)? {
            orphans.push((edge.clone(), circle));
        }
    }
    if orphans.len() < 2 {
        return Ok(0);
    }
    let find_use = |solid: &BrepSolid, edge_id: u64| -> Option<(usize, usize, bool)> {
        solid
            .shells
            .iter()
            .enumerate()
            .find_map(|(shell_index, shell)| {
                shell
                    .faces
                    .iter()
                    .enumerate()
                    .find_map(|(face_index, face)| {
                        face.loops
                            .iter()
                            .flat_map(|loop_record| &loop_record.coedges)
                            .find(|coedge| coedge.edge_id == edge_id)
                            .map(|coedge| (shell_index, face_index, coedge.forward))
                    })
            })
    };
    let coplanar_band = tolerance.max(1e-4);
    let mut candidates = Vec::new();
    for first in 0..orphans.len() {
        for second in first + 1..orphans.len() {
            let (first_edge, first_circle) = &orphans[first];
            let (second_edge, second_circle) = &orphans[second];
            if first_circle.axis.dot(second_circle.axis).abs() < 0.999 {
                continue;
            }
            let between = second_circle.center.sub(first_circle.center);
            let axial = between.dot(first_circle.axis);
            if between.sub(first_circle.axis.scale(axial)).length() > coplanar_band
                || axial.abs() > coplanar_band
            {
                continue;
            }
            let radial_gap = (first_circle.radius - second_circle.radius).abs();
            if radial_gap <= coplanar_band {
                continue;
            }
            // Rims that already bound one face are that face's two ends, not
            // a wall pair (capping them against each other would close the
            // tube's bore with a phantom membrane).
            let share_face = solid
                .shells
                .iter()
                .flat_map(|shell| &shell.faces)
                .any(|face| {
                    let mut uses_first = false;
                    let mut uses_second = false;
                    for coedge in face
                        .loops
                        .iter()
                        .flat_map(|loop_record| &loop_record.coedges)
                    {
                        uses_first |= coedge.edge_id == first_edge.id;
                        uses_second |= coedge.edge_id == second_edge.id;
                    }
                    uses_first && uses_second
                });
            if share_face {
                continue;
            }
            candidates.push((radial_gap, first, second));
        }
    }
    candidates.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| orphans[a.1].0.id.cmp(&orphans[b.1].0.id))
            .then_with(|| orphans[a.2].0.id.cmp(&orphans[b.2].0.id))
    });
    let mut consumed = HashSet::<u64>::default();
    let mut shell_parent = (0..solid.shells.len()).collect::<Vec<_>>();
    fn shell_root(parent: &mut [usize], index: usize) -> usize {
        if parent[index] != index {
            parent[index] = shell_root(parent, parent[index]);
        }
        parent[index]
    }
    let mut welded = 0usize;
    let mut shell_unions: Vec<(usize, usize)> = Vec::new();
    for (_, first, second) in candidates {
        let (first_edge, first_circle) = &orphans[first];
        let (second_edge, second_circle) = &orphans[second];
        if consumed.contains(&first_edge.id) || consumed.contains(&second_edge.id) {
            continue;
        }
        // Outer = larger radius; the annulus is the region between them.
        let (outer_edge, outer_circle, inner_edge, _inner_circle) =
            if first_circle.radius >= second_circle.radius {
                (first_edge, first_circle, second_edge, second_circle)
            } else {
                (second_edge, second_circle, first_edge, first_circle)
            };
        let Some((outer_shell, outer_owner_face, outer_forward)) = find_use(solid, outer_edge.id)
        else {
            continue;
        };
        let Some((inner_shell, inner_owner_face, inner_forward)) = find_use(solid, inner_edge.id)
        else {
            continue;
        };
        // Provenance for the new annulus: the wall thickness it exposes
        // belongs to the outer rim owner's source face (for a through-hole
        // that is the hole wall whose offset produced the outer rim).
        let annulus_source_face_id = face_images
            .get(
                solid.shells[outer_shell].faces[outer_owner_face]
                    .id
                    .saturating_sub(1) as usize,
            )
            .map(|image| image.source_face_id)
            .ok_or_else(|| "offset_shell: rim pair owner lost provenance".to_string())?;
        // Exactness upgrade with full fallback: swap each fitted-polyline rim
        // for its owner surface's exact isoline where possible, so the
        // annulus trims (and the shell's volume) are exact, not chord-sagged.
        let outer_id = outer_edge.id;
        let inner_id = inner_edge.id;
        let (outer_edge, outer_forward) = match upgrade_rim_to_owner_isoline(
            solid,
            outer_id,
            outer_shell,
            outer_owner_face,
            tolerance,
        )? {
            Some((updated, forward_new)) => (updated, forward_new),
            None => (outer_edge.clone(), outer_forward),
        };
        let (inner_edge, inner_forward) = match upgrade_rim_to_owner_isoline(
            solid,
            inner_id,
            inner_shell,
            inner_owner_face,
            tolerance,
        )? {
            Some((updated, forward_new)) => (updated, forward_new),
            None => (inner_edge.clone(), inner_forward),
        };
        let (outer_edge, inner_edge) = (&outer_edge, &inner_edge);
        let axis = outer_circle.axis;
        let Ok(u_direction) = axis.perpendicular() else {
            continue;
        };
        let v_direction = axis.cross(u_direction).normalized()?;
        let extent = outer_circle.radius * 3.0;
        let plane = crate::make_plane(
            outer_circle
                .center
                .sub(u_direction.scale(extent * 0.5))
                .sub(v_direction.scale(extent * 0.5)),
            u_direction,
            v_direction,
            extent,
            extent,
        )?;
        // Manifold rule: the annulus traces each rim opposite its single use.
        let Some(outer_pcurve) = boundary_pcurve(outer_edge, !outer_forward, &plane, tolerance)?
        else {
            continue;
        };
        let Some(mut inner_pcurve) =
            boundary_pcurve(inner_edge, !inner_forward, &plane, tolerance)?
        else {
            continue;
        };
        let mut next_coedge_id = solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .flat_map(|face| &face.loops)
            .flat_map(|loop_record| &loop_record.coedges)
            .map(|coedge| coedge.id)
            .max()
            .unwrap_or(0)
            + 1;
        let loop_area = |pcurve: &crate::NurbsCurve, edge_id: u64, forward: bool| {
            parameter_space_area(&FaceRecord {
                id: 0,
                surface: plane.clone(),
                same_sense: true,
                loops: vec![LoopRecord {
                    id: 1,
                    coedges: vec![CoedgeRecord {
                        id: 1,
                        edge_id,
                        forward,
                        pcurve: pcurve.clone(),
                    }],
                }],
                name: None,
            })
        };
        let outer_area = loop_area(&outer_pcurve, outer_edge.id, !outer_forward)?;
        let mut inner_area = loop_area(&inner_pcurve, inner_edge.id, !inner_forward)?;
        let mut inner_wall_forward = !inner_forward;
        if outer_area.signum() == inner_area.signum() {
            // The two owning components disagree about their global sense.
            // A floating inner component may be flipped wholesale; a shared
            // component means the shell is genuinely inconsistent — refuse.
            let outer_root = shell_root(&mut shell_parent, outer_shell);
            let inner_root = shell_root(&mut shell_parent, inner_shell);
            if outer_root == inner_root {
                os_debug!(
                    "PAIR skip: rims {} and {} wind the same way inside one component",
                    outer_edge.id,
                    inner_edge.id
                );
                continue;
            }
            let flip_shells = (0..solid.shells.len())
                .filter(|index| shell_root(&mut shell_parent, *index) == inner_root)
                .collect::<Vec<_>>();
            for shell_index in flip_shells {
                flip_shell_faces(&mut solid.shells[shell_index])?;
            }
            inner_wall_forward = inner_forward;
            inner_pcurve = boundary_pcurve(inner_edge, inner_wall_forward, &plane, tolerance)?
                .ok_or_else(|| "offset_shell: annulus inner rim left its plane".to_string())?;
            inner_area = loop_area(&inner_pcurve, inner_edge.id, inner_wall_forward)?;
            if outer_area.signum() == inner_area.signum() {
                os_debug!("PAIR skip: inner-component flip did not fix the winding");
                continue;
            }
        }
        let mut face = FaceRecord {
            id: solid
                .shells
                .iter()
                .flat_map(|shell| &shell.faces)
                .map(|face| face.id)
                .max()
                .unwrap_or(0)
                + 1,
            surface: plane,
            same_sense: outer_area + inner_area > 0.0,
            loops: Vec::new(),
            name: None,
        };
        let next_loop_id = solid
            .shells
            .iter()
            .flat_map(|shell| &shell.faces)
            .flat_map(|face| &face.loops)
            .map(|loop_record| loop_record.id)
            .max()
            .unwrap_or(0)
            + 1;
        for (loop_offset, (edge, forward, pcurve)) in [
            (outer_edge, !outer_forward, outer_pcurve),
            (inner_edge, inner_wall_forward, inner_pcurve),
        ]
        .into_iter()
        .enumerate()
        {
            face.loops.push(LoopRecord {
                id: next_loop_id + loop_offset as u64,
                coedges: vec![CoedgeRecord {
                    id: next_coedge_id,
                    edge_id: edge.id,
                    forward,
                    pcurve,
                }],
            });
            next_coedge_id += 1;
        }
        solid.shells[outer_shell].faces.push(face);
        face_images.push(OffsetShellFaceImageRecord {
            role: OffsetFaceRole::Wall,
            source_face_id: annulus_source_face_id,
        });
        for edge_id in [outer_edge.id, inner_edge.id] {
            if let Some(record) = solid.edges.iter_mut().find(|record| record.id == edge_id) {
                record.degenerate = false;
            }
        }
        if outer_shell != inner_shell {
            let outer_root = shell_root(&mut shell_parent, outer_shell);
            let inner_root = shell_root(&mut shell_parent, inner_shell);
            if outer_root != inner_root {
                shell_parent[inner_root] = outer_root;
            }
            shell_unions.push((outer_shell, inner_shell));
        }
        consumed.insert(outer_edge.id);
        consumed.insert(inner_edge.id);
        welded += 1;
    }
    merge_connected_shells(solid, shell_unions);
    if welded > 0 {
        orient_open_solid_faces(solid)?;
    }
    Ok(welded)
}

/// Give every opening-wall face that is a BORE'S END RING the bore wall's
/// provenance, whichever construction built it.
///
/// A through-bore that meets an opening exposes the wall thickness between
/// the hole wall and its offset as a planar ring in the opening's plane. Two
/// constructions produce that ring: the opening carrier's own fragmentation
/// (a wall fragment, provenance the OPENING face) and
/// [`weld_coplanar_rim_pair_annuli`] (provenance the hole wall). Which one
/// runs depends on whether the rims arrived exact enough for the fragmenter
/// to close the ring — so the same face was `Pin_S_1` from one path and
/// `Box_NY_1` from the other, and a document that referenced the ring by
/// name (the 2026-08-24 `BadBoolean` extrude) broke when the offset carrier
/// started building exact rims (2026-09-06).
///
/// The bore's provenance is the one kept: the ring is determined by the bore
/// alone (there is exactly one per bore end), where "the n-th piece of the
/// opening wall" depends on encounter order. A ring is a bore's end ring when
/// it is a two-loop wall whose neighbours are all images of ONE source face
/// and whose INNER rim is that face's source image — the material lies
/// outside the hole. A can shelled through its lid has its source rim
/// OUTSIDE (the offset rim inside) and keeps the opening's name, as before.
///
/// Runs on the finished solid, after the merge remaps `face_images` into
/// face iteration order. Returns how many rings changed provenance.
pub(super) fn claim_bore_end_rings(
    solid: &BrepSolid,
    face_images: &mut [OffsetShellFaceImageRecord],
) -> Result<usize, String> {
    let faces = solid
        .shells
        .iter()
        .flat_map(|shell| &shell.faces)
        .collect::<Vec<_>>();
    if faces.len() != face_images.len() {
        return Err(format!(
            "offset_shell: {} faces but {} provenance records",
            faces.len(),
            face_images.len()
        ));
    }
    let mut users = HashMap::<u64, Vec<usize>>::default();
    for (index, face) in faces.iter().enumerate() {
        for coedge in face.loops.iter().flat_map(|loop_record| &loop_record.coedges) {
            users.entry(coedge.edge_id).or_default().push(index);
        }
    }
    let mut claimed = 0usize;
    for (index, face) in faces.iter().enumerate() {
        if !matches!(face_images[index].role, OffsetFaceRole::Wall) || face.loops.len() != 2 {
            continue;
        }
        let mut areas = [0.0f64; 2];
        for (slot, loop_record) in face.loops.iter().enumerate() {
            areas[slot] = parameter_space_area(&FaceRecord {
                id: 0,
                surface: face.surface.clone(),
                same_sense: true,
                loops: vec![loop_record.clone()],
                name: None,
            })?
            .abs();
        }
        let inner = usize::from(areas[1] < areas[0]);
        let mut source: Option<u64> = None;
        let mut inner_is_source = true;
        let mut outer_is_offset = true;
        let mut neighbours = 0usize;
        let mut consistent = true;
        for (slot, loop_record) in face.loops.iter().enumerate() {
            for coedge in &loop_record.coedges {
                for &other in users.get(&coedge.edge_id).map(Vec::as_slice).unwrap_or(&[]) {
                    if other == index {
                        continue;
                    }
                    neighbours += 1;
                    let image = &face_images[other];
                    match source {
                        None => source = Some(image.source_face_id),
                        Some(known) if known == image.source_face_id => {}
                        Some(_) => consistent = false,
                    }
                    let is_source = matches!(image.role, OffsetFaceRole::Source);
                    let is_offset = matches!(image.role, OffsetFaceRole::Offset);
                    if slot == inner {
                        inner_is_source &= is_source;
                    } else {
                        outer_is_offset &= is_offset;
                    }
                }
            }
        }
        let Some(source) = source else { continue };
        if !consistent
            || neighbours == 0
            || !inner_is_source
            || !outer_is_offset
            || face_images[index].source_face_id == source
        {
            continue;
        }
        face_images[index].source_face_id = source;
        claimed += 1;
    }
    Ok(claimed)
}
