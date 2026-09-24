use super::*;

/// §5.8 CLOSED loft: the sections form a RING (the last flows back into the
/// first), producing a capless genus-1 solid. Every interpolation column runs
/// through `interpolate_curve_closed`, so the ring is C² across the closure —
/// not a welded seam. Topology per skin is the cylinder-wall rectangle turned
/// on its side: u runs along the section curve (open), v through the sections
/// (closed); the section-0 curves themselves serve as the doubled v-seam
/// edges and the corner rings (one closed curve through every section at each
/// section corner) are shared between adjacent skins.
pub fn loft_profile_brep_closed(input_sections: &[Vec<NurbsCurve>]) -> Result<BrepSolid, String> {
    let tolerance = 1e-6;
    let section_count = input_sections.len();
    if section_count < 4 {
        return Err("loftSolid: a closed loft needs at least 4 sections".into());
    }
    let sections = input_sections.to_vec();
    let curve_count = validate_sections(&sections, tolerance, "loftSolid", false)?;

    // Cyclic chord parameters over S stations plus the wrap span back to
    // station 0 — the closed analogue of the open loft's averaging.
    let mut accumulated = vec![0.0; section_count + 1];
    let mut columns = 0usize;
    for curve_index in 0..curve_count {
        for control_index in 0..sections[0][curve_index].control_points.len() {
            let mut chords = vec![0.0; section_count + 1];
            let mut total = 0.0;
            for station in 1..=section_count {
                let previous =
                    sections[station - 1][curve_index].control_points[control_index].point()?;
                let current = sections[station % section_count][curve_index].control_points
                    [control_index]
                    .point()?;
                total += current.sub(previous).length();
                chords[station] = total;
            }
            if total <= tolerance {
                continue;
            }
            for station in 0..=section_count {
                accumulated[station] += chords[station] / total;
            }
            columns += 1;
        }
    }
    if columns == 0 {
        return Err("loftSolid: sections coincide".into());
    }
    let mut parameters = vec![0.0; section_count + 1];
    for station in 0..=section_count {
        parameters[station] = accumulated[station] / columns as f64;
    }
    parameters[0] = 0.0;
    parameters[section_count] = 1.0;
    if parameters.windows(2).any(|pair| pair[1] <= pair[0] + 1e-9) {
        return Err("loftSolid: closed sections are not strictly ordered".into());
    }

    let mut skins = Vec::with_capacity(curve_count);
    for curve_index in 0..curve_count {
        let reference = &sections[0][curve_index];
        let mut grid = Vec::with_capacity(reference.control_points.len());
        let mut knots_v = Vec::new();
        for control_index in 0..reference.control_points.len() {
            let points = sections
                .iter()
                .map(|section| section[curve_index].control_points[control_index].point())
                .collect::<Result<Vec<_>, _>>()?;
            let interpolated = crate::interpolate_curve_closed(&points, &parameters)?;
            knots_v = interpolated.knots.clone();
            let weight = reference.control_points[control_index].w;
            grid.push(
                interpolated
                    .control_points
                    .iter()
                    .map(|point| Vec4::from_point(point.point().unwrap(), weight))
                    .collect(),
            );
        }
        skins.push(NurbsSurface::new(
            reference.degree,
            3,
            reference.knots.clone(),
            knots_v,
            grid,
        )?);
    }

    // Topology: c corner vertices (section-0 corners), c corner-ring edges
    // (closed v-curves at each corner), c seam edges (the section-0 curves),
    // c skin faces. V − E + F = c − 2c + c = 0 → genus 1.
    let base_points = closed_points(&sections[0], tolerance)?;
    let mut vertices = Vec::with_capacity(curve_count);
    for (index, point) in base_points.iter().enumerate() {
        vertices.push(VertexRecord {
            id: index as u64 + 1,
            point: *point,
        });
    }
    let mut edges = Vec::with_capacity(2 * curve_count);
    let mut ring_edge_ids = Vec::with_capacity(curve_count);
    let mut seam_edge_ids = Vec::with_capacity(curve_count);
    for index in 0..curve_count {
        let [u_start, u_end] = sections[0][index].domain()?;
        let _ = u_end;
        let ring = skins[index].iso_curve_u(u_start)?;
        let [ring_start, ring_end] = ring.domain()?;
        let ring_id = 100 + index as u64;
        ring_edge_ids.push(ring_id);
        edges.push(EdgeRecord {
            id: ring_id,
            curve: ring,
            t0: ring_start,
            t1: ring_end,
            start_vertex_id: index as u64 + 1,
            end_vertex_id: index as u64 + 1,
            degenerate: false,
            name: None,
        });
        let seam = sections[0][index].clone();
        let [seam_start, seam_end] = seam.domain()?;
        let seam_id = 100 + curve_count as u64 + index as u64;
        seam_edge_ids.push(seam_id);
        edges.push(EdgeRecord {
            id: seam_id,
            curve: seam,
            t0: seam_start,
            t1: seam_end,
            start_vertex_id: index as u64 + 1,
            end_vertex_id: ((index + 1) % curve_count) as u64 + 1,
            degenerate: false,
            name: None,
        });
    }

    let mut next_id = 1000u64;
    let mut faces = Vec::with_capacity(curve_count);
    for index in 0..curve_count {
        let [u_start, u_end] = sections[0][index].domain()?;
        // Cylinder-wall rectangle: seam at v=0 forward, next corner ring up,
        // seam at v=1 backward, own corner ring down.
        let coedges = vec![
            CoedgeRecord {
                id: next_id,
                edge_id: seam_edge_ids[index],
                forward: true,
                pcurve: parameter_line(u_start, 0.0, u_end, 0.0)?,
            },
            CoedgeRecord {
                id: next_id + 1,
                edge_id: ring_edge_ids[(index + 1) % curve_count],
                forward: true,
                pcurve: parameter_line(u_end, 0.0, u_end, 1.0)?,
            },
            CoedgeRecord {
                id: next_id + 2,
                edge_id: seam_edge_ids[index],
                forward: false,
                pcurve: parameter_line(u_end, 1.0, u_start, 1.0)?,
            },
            CoedgeRecord {
                id: next_id + 3,
                edge_id: ring_edge_ids[index],
                forward: false,
                pcurve: parameter_line(u_start, 1.0, u_start, 0.0)?,
            },
        ];
        next_id += 4;
        let loop_id = next_id;
        next_id += 1;
        let face_id = next_id;
        next_id += 1;
        faces.push(FaceRecord {
            id: face_id,
            surface: skins[index].clone(),
            same_sense: true,
            loops: vec![LoopRecord {
                id: loop_id,
                coedges,
            }],
            name: None,
        });
    }

    let mut solid = BrepSolid {
        id: next_id,
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: next_id + 1,
            faces,
        }],
        genus: 1,
    };
    let issues = solid.validate();
    if !issues.is_empty() {
        return Err(format!(
            "loftSolid: closed loft failed validation: {issues:?}"
        ));
    }
    // The ring's outward side depends on the sections' winding; the signed
    // volume is the arbiter.
    if crate::solid_signed_volume(&solid)? < 0.0 {
        crate::offset_shell::flip_all_faces(&mut solid)?;
    }
    Ok(solid)
}
