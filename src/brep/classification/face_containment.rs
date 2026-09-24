use crate::curve::interior_knot_count;
use super::*;

fn point_segment_distance(point: Vec2, start: Vec2, end: Vec2) -> f64 {
    let segment = end.sub(start);
    let length_squared = segment.dot(segment);
    if length_squared <= 1e-30 {
        return point.sub(start).length();
    }
    let parameter = (point.sub(start).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.sub(start.add(segment.scale(parameter))).length()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolygonClass {
    Inside,
    Outside,
    Boundary,
}

fn point_in_polygon(point: Vec2, polygon: &[Vec2], tolerance: f64) -> PolygonClass {
    for index in 0..polygon.len() {
        if point_segment_distance(point, polygon[index], polygon[(index + 1) % polygon.len()])
            <= tolerance
        {
            return PolygonClass::Boundary;
        }
    }
    let mut inside = false;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        if (a.y > point.y) != (b.y > point.y) {
            let crossing = a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if crossing > point.x {
                inside = !inside;
            }
        }
    }
    if inside {
        PolygonClass::Inside
    } else {
        PolygonClass::Outside
    }
}

struct SegmentReference<'a> {
    start: Vec2,
    end: Vec2,
    curve: &'a NurbsCurve,
    parameter_start: f64,
    parameter_end: f64,
}


/// Point-in-face for a DOUBLY-periodic (torus) seam-band face, whose material
/// region the raw even-odd/tangent test below gets wrong: the constant-level
/// seam rim is a zero-area iso line and the wavy cut's own polygon encloses only
/// the thin strip it bounds, so a point in the band reads Outside (or flips on
/// the fragile nearest-tangent refinement). Reconstruct the band as one simple
/// (u,v) polygon — identical topology to the mass integrator and tessellator —
/// and test the point against it, so all three subsystems agree on the material
/// side. Returns None (keep the general path) for every other face.
fn seam_band_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    tolerance: f64,
) -> Result<Option<PolygonClass>, String> {
    if face.surface.closed_directions()? != (true, true) {
        return Ok(None);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    if crate::topology::doubly_periodic_has_only_collapsed_loops(face)? {
        return Ok(Some(PolygonClass::Inside));
    }
    let u_span = (u1 - u0).abs().max(1e-30);
    let v_span = (v1 - v0).abs().max(1e-30);
    let mut loops_uv: Vec<Vec<[f64; 2]>> = Vec::with_capacity(face.loops.len());
    for loop_record in &face.loops {
        let mut points: Vec<[f64; 2]> = Vec::new();
        for coedge in &loop_record.coedges {
            let [d0, d1] = coedge.pcurve.domain()?;
            let samples = 24;
            for k in 0..=samples {
                let t = d0 + (d1 - d0) * k as f64 / samples as f64;
                let p = coedge.pcurve.evaluate(t)?;
                points.push([p.x, p.y]);
            }
        }
        let (mut umin, mut umax, mut vmin, mut vmax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for p in &points {
            umin = umin.min(p[0]);
            umax = umax.max(p[0]);
            vmin = vmin.min(p[1]);
            vmax = vmax.max(p[1]);
        }
        // Mirror `mass_properties::biperiodic_band_range`: a torus has no
        // geometric pole, so a VERTEX_LOOP whose whole pcurve trace collapses
        // to one parameter point is a zero-area puncture, not a band boundary.
        // Restrict the new behavior to faces that actually contain such an
        // extra loop; ordinary two-loop torus classification stays unchanged.
        if face.loops.len() > 2 && (umax - umin) <= 1e-3 * u_span && (vmax - vmin) <= 1e-3 * v_span
        {
            continue;
        }
        loops_uv.push(points);
    }
    // MERGED SEAM-CARRYING BAND (single loop): `insert_periodic_band_seam_edges`
    // fuses a wrapped band's two rims into ONE loop joined by seam columns, so
    // the two-loop paths below never see it, and the plain even-odd polygon
    // carries a period-magnitude hop at each rim→column junction (a phantom
    // diagonal across the domain) that misclassifies the whole middle zone.
    // Unwrap the loop onto the covering plane (`loop_seam_offsets`, the same
    // fold mass integration and tessellation use), where it IS a simple
    // polygon, and even-odd the query's period images against it. Gated on a
    // genuinely seam-crossing single loop on a doubly-periodic surface; every
    // other face falls through unchanged. Hatch: BREP_SEAM_BAND_MERGED=0.
    if face.loops.len() == 1
        && loops_uv.len() == 1
        && std::env::var("BREP_SEAM_BAND_MERGED").as_deref() != Ok("0")
    {
        let coedges = &face.loops[0].coedges;
        let offsets = crate::topology::loop_seam_offsets(coedges, true, true, u_span, v_span)?;
        if offsets.iter().any(|o| o[0] != 0.0 || o[1] != 0.0) {
            let mut polygon: Vec<Vec2> = Vec::new();
            for (coedge_index, coedge) in coedges.iter().enumerate() {
                let [d0, d1] = coedge.pcurve.domain()?;
                let samples = 24;
                for k in 0..samples {
                    let t = d0 + (d1 - d0) * k as f64 / samples as f64;
                    let p = coedge.pcurve.evaluate(t)?;
                    polygon.push(Vec2 {
                        x: p.x + offsets[coedge_index][0],
                        y: p.y + offsets[coedge_index][1],
                    });
                }
            }
            let mut best = PolygonClass::Outside;
            'images: for du in [-1.0, 0.0, 1.0] {
                for dv in [-1.0, 0.0, 1.0] {
                    let image = Vec2 {
                        x: point.x + du * u_span,
                        y: point.y + dv * v_span,
                    };
                    match point_in_polygon(image, &polygon, tolerance) {
                        PolygonClass::Boundary => {
                            best = PolygonClass::Boundary;
                            break 'images;
                        }
                        PolygonClass::Inside => best = PolygonClass::Inside,
                        PolygonClass::Outside => {}
                    }
                }
            }
            return Ok(Some(best));
        }
    }
    if loops_uv.len() != 2 {
        return Ok(None);
    }
    if let Some(band) = crate::topology::analyze_doubly_periodic_seam_band(
        &loops_uv,
        [u0, u1, v0, v1],
        face.same_sense,
    ) {
        let polygon: Vec<Vec2> =
            crate::topology::seam_band_uv_polygon(&loops_uv, [u0, u1, v0, v1], &band)
                .into_iter()
                .map(|p| Vec2 { x: p[0], y: p[1] })
                .collect();
        return Ok(Some(point_in_polygon(point, &polygon, tolerance)));
    }
    // TWO clean full-wrap rims are the companion `biperiodic_band_range` case,
    // with or without an extra collapsed puncture. Classify only the cross
    // parameter: the material is either the strip between the rims or its
    // periodic complement. The exact two-loop/full-wrap/constant-cross-level
    // gates below mirror mass integration and tessellation; every other torus
    // face retains the general classifier.
    for p_is_u in [true, false] {
        let (period, q_extent) = if p_is_u {
            (u1 - u0, v_span)
        } else {
            (v1 - v0, u_span)
        };
        if !(period > 0.0) {
            continue;
        }
        let coord = |p: &[f64; 2]| if p_is_u { (p[0], p[1]) } else { (p[1], p[0]) };
        let mut rings = Vec::with_capacity(2);
        for points in &loops_uv {
            let (mut pmin, mut pmax, mut qmin, mut qmax) = (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            );
            for p in points {
                let (periodic, cross) = coord(p);
                pmin = pmin.min(periodic);
                pmax = pmax.max(periodic);
                qmin = qmin.min(cross);
                qmax = qmax.max(cross);
            }
            if (pmax - pmin) < 0.6 * period || (qmax - qmin) > 0.05 * q_extent {
                rings.clear();
                break;
            }
            let mut net = 0.0;
            for pair in points.windows(2) {
                let mut delta = coord(&pair[1]).0 - coord(&pair[0]).0;
                if delta > 0.5 * period {
                    delta -= period;
                } else if delta < -0.5 * period {
                    delta += period;
                }
                net += delta;
            }
            let direction = if net > 0.25 * period {
                1
            } else if net < -0.25 * period {
                -1
            } else {
                0
            };
            rings.push((0.5 * (qmin + qmax), direction));
        }
        if rings.len() != 2 {
            continue;
        }
        rings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let [(q_lo, lower_direction), (q_hi, upper_direction)] = rings.as_slice() else {
            unreachable!()
        };
        let inconclusive =
            *lower_direction == 0 || *upper_direction == 0 || lower_direction == upper_direction;
        let between_is_ccw_uv = if p_is_u {
            *lower_direction > 0
        } else {
            *lower_direction < 0
        };
        let complement = !inconclusive && between_is_ccw_uv != face.same_sense;
        let q = if p_is_u { point.y } else { point.x };
        if (q - q_lo).abs() <= tolerance || (q - q_hi).abs() <= tolerance {
            return Ok(Some(PolygonClass::Boundary));
        }
        let between = q > *q_lo && q < *q_hi;
        return Ok(Some(if between != complement {
            PolygonClass::Inside
        } else {
            PolygonClass::Outside
        }));
    }
    Ok(None)
}

/// Point-in-face for a SINGLY-PERIODIC face with a SEAM-STRADDLING loop: a
/// trim loop crossing the u seam is stored with its samples folded into the
/// domain, so its flat polygon is TORN at the seam and plain even-odd
/// misclassifies near the tear (STEP 00000585-family: face 519's big inner
/// loop straddles the seam and a REAL section sub-segment between two accepted
/// pieces read Outside in `process_curve`'s midpoint gate — trial 174's
/// missing middle segment). Mirror fragment.rs's proven seam-straddling
/// fallback: UNWRAP every loop by accumulating period-folded deltas (a
/// zero-winding loop closes in the covering plane), then even-odd the query at
/// its −P/0/+P period images across all unwrapped loops. Fires only when the
/// face is closed in exactly u, has >2 loops, and at least one loop straddles
/// the seam (≥1 fold jump); every other face returns None and keeps the plain
/// classifier bit-identically. Loops that WIND the period (net winding ≠ 0)
/// do not close under unwrapping — bail to the plain path rather than guess.
/// Escape hatch `BREP_HORIZON_CONTAINMENT=0`.
fn wrapped_horizon_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    tolerance: f64,
) -> Result<Option<PolygonClass>, String> {
    if face.surface.closed_directions()? != (true, false) || face.loops.is_empty() {
        return Ok(None);
    }
    if std::env::var("BREP_HORIZON_CONTAINMENT").as_deref() == Ok("0") {
        return Ok(None);
    }
    // SINGLE/DOUBLE-LOOP straddlers (t91's face 675: ONE loop drawn in-domain
    // that hops the u-seam twice, trimming a sliver strip AT the seam) used
    // to be excluded by a `loops.len() <= 2` gate and fell to the plain
    // even-odd, whose verdict on the hopped polygon is INVERTED (the strip
    // interior read Outside, far azimuths read Inside) — minting bogus
    // far-azimuth section pieces and discarding the real ones. The unwrap +
    // period-image even-odd below is loop-count-agnostic, so the gate now
    // only requires a non-empty loop set; non-straddling loops still bail to
    // the plain path unchanged. Escape hatch: BREP_HORIZON_SINGLE_LOOP=0
    // restores the old minimum.
    if face.loops.len() <= 2 && std::env::var("BREP_HORIZON_SINGLE_LOOP").as_deref() == Ok("0") {
        return Ok(None);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let u_period = u1 - u0;
    if u_period <= 0.0 {
        return Ok(None);
    }
    let debug = std::env::var("BREP_DEBUG_HORIZON").is_ok();
    let mut loops: Vec<Vec<Vec2>> = Vec::new();
    let mut straddling = false;
    for loop_record in &face.loops {
        let mut points: Vec<Vec2> = Vec::new();
        for coedge in &loop_record.coedges {
            let curve = &coedge.pcurve;
            let [start, end] = curve.domain()?;
            let sample_count =
                2usize.max((interior_knot_count(&curve.knots, curve.degree) + 1) * (curve.degree + 1) * 4);
            for index in 0..sample_count {
                let parameter = start + (end - start) * index as f64 / sample_count as f64;
                let evaluated = curve.evaluate(parameter)?;
                points.push(Vec2 {
                    x: evaluated.x,
                    y: evaluated.y,
                });
            }
        }
        if points.len() < 3 {
            continue;
        }
        // Unwrap into the covering plane: place each sample at the previous
        // one plus the period-folded delta. A seam-straddling loop becomes a
        // continuous closed polygon; a WINDING loop does not close — bail.
        let mut unwrapped = Vec::with_capacity(points.len());
        let mut jumps = 0usize;
        let mut cursor = points[0];
        unwrapped.push(cursor);
        for pair in points.windows(2) {
            let mut du = pair[1].x - pair[0].x;
            let folded = du - u_period * (du / u_period).round();
            if (du - folded).abs() > 0.25 * u_period {
                jumps += 1;
            }
            du = folded;
            cursor = Vec2 {
                x: cursor.x + du,
                y: pair[1].y,
            };
            unwrapped.push(cursor);
        }
        let closure = (unwrapped[0].x - unwrapped[unwrapped.len() - 1].x).abs();
        if closure > 0.25 * u_period {
            if debug {
                eprintln!(
                    "horizon: face {} loop winds the period (closure {closure:.3}) — bail",
                    face.id
                );
            }
            return Ok(None); // winding loop: unwrapping cannot close it
        }
        if jumps > 0 {
            straddling = true;
        }
        loops.push(unwrapped);
    }
    if !straddling || loops.is_empty() {
        return Ok(None);
    }
    // Even-odd on the QUOTIENT cylinder: each unwrapped loop is tested at
    // every period image of the query and contributes its own parity. The old
    // combining ("any single image with an odd crossing count ⇒ Inside") is
    // wrong whenever two loops live in DIFFERENT period frames after
    // unwrapping — t181/00000312#p7: face 519's seam-straddling HOLE unwraps
    // to u∈[−0.039, 0.039] while the outer rectangle spans [0, 1], so a query
    // inside the hole at u≈0.986 is contained by the outer loop at shift 0
    // (odd ⇒ Inside) and by the hole at shift −P (odd ⇒ Inside again); the
    // true even count (outer + hole = 2 ⇒ Outside) is never assembled at any
    // single image. Fragment selection then kept a PHANTOM half-hole region
    // whose boundary (half the hole loop + a derived chord) minted same-sense
    // coedges at assembly. Summing per-loop parities across images restores
    // the covering-space even-odd. Escape hatch: BREP_HORIZON_CROSS_FRAME=0
    // restores the old per-image combining.
    if std::env::var("BREP_HORIZON_CROSS_FRAME").as_deref() == Ok("0") {
        let mut best: Option<PolygonClass> = None;
        for shift in [-u_period, 0.0, u_period] {
            let image = Vec2 {
                x: point.x + shift,
                y: point.y,
            };
            let mut crossings = 0usize;
            for polygon in &loops {
                match point_in_polygon(image, polygon, tolerance) {
                    PolygonClass::Boundary => return Ok(Some(PolygonClass::Boundary)),
                    PolygonClass::Inside => crossings += 1,
                    PolygonClass::Outside => {}
                }
            }
            if crossings % 2 == 1 {
                best = Some(PolygonClass::Inside);
            } else if best.is_none() {
                best = Some(PolygonClass::Outside);
            }
        }
        if debug {
            eprintln!(
                "horizon: face {} loops={} straddling (legacy per-image) -> {:?}",
                face.id,
                loops.len(),
                best
            );
        }
        return Ok(best);
    }
    let mut crossings = 0usize;
    for polygon in &loops {
        let mut image_hits = 0usize;
        for shift in [-u_period, 0.0, u_period] {
            let image = Vec2 {
                x: point.x + shift,
                y: point.y,
            };
            match point_in_polygon(image, polygon, tolerance) {
                PolygonClass::Boundary => return Ok(Some(PolygonClass::Boundary)),
                PolygonClass::Inside => image_hits += 1,
                PolygonClass::Outside => {}
            }
        }
        crossings += image_hits % 2;
    }
    let class = if crossings % 2 == 1 {
        PolygonClass::Inside
    } else {
        PolygonClass::Outside
    };
    if debug {
        eprintln!(
            "horizon: face {} loops={} straddling -> {:?}",
            face.id,
            loops.len(),
            class
        );
    }
    Ok(Some(class))
}

/// Classify a spherical cap represented by a varying full-wrap contact rim and
/// the oppositely wound collapsed rim at one sphere pole.  This is the natural
/// topology when a circle about an oblique axis encloses a pole of the sphere's
/// stored parameter frame.
fn winding_sphere_cap_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    tolerance: f64,
) -> Result<Option<PolygonClass>, String> {
    if !matches!(
        face.surface.analytic(),
        Some(crate::AnalyticSurface::Sphere { .. })
    ) || face.surface.closed_directions()? != (true, false)
        || face.loops.len() != 2
    {
        return Ok(None);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let period = u1 - u0;
    let v_span = v1 - v0;
    if !(period > 0.0 && v_span > 0.0) {
        return Ok(None);
    }
    struct WindingLoop {
        points: Vec<Vec2>,
        winding: f64,
        vmin: f64,
        vmax: f64,
    }
    let mut loops = Vec::with_capacity(2);
    for loop_record in &face.loops {
        let mut points = Vec::new();
        for coedge in &loop_record.coedges {
            let [start, end] = coedge.pcurve.domain()?;
            let samples = 2usize
                .max((interior_knot_count(&coedge.pcurve.knots, coedge.pcurve.degree) + 1) * (coedge.pcurve.degree + 1) * 4);
            for index in 0..samples {
                let parameter = start + (end - start) * index as f64 / samples as f64;
                let p = coedge.pcurve.evaluate(parameter)?;
                points.push(Vec2 { x: p.x, y: p.y });
            }
        }
        if points.len() < 2 {
            return Ok(None);
        }
        let first = points[0];
        let mut cursor = first;
        let mut unwrapped = vec![cursor];
        for next in points.iter().skip(1) {
            let du = next.x - cursor.x;
            let folded = du - period * (du / period).round();
            cursor = Vec2 {
                x: cursor.x + folded,
                y: next.y,
            };
            unwrapped.push(cursor);
        }
        let last_raw = *points.last().unwrap();
        let closing_du = first.x - last_raw.x;
        let closing_folded = closing_du - period * (closing_du / period).round();
        let winding = cursor.x + closing_folded - first.x;
        let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &unwrapped {
            vmin = vmin.min(p.y);
            vmax = vmax.max(p.y);
        }
        loops.push(WindingLoop {
            points: unwrapped,
            winding,
            vmin,
            vmax,
        });
    }
    if loops
        .iter()
        .any(|loop_data| (loop_data.winding.abs() - period).abs() > 0.05 * period)
        || loops[0].winding * loops[1].winding >= 0.0
    {
        return Ok(None);
    }
    let flat = |loop_data: &WindingLoop| loop_data.vmax - loop_data.vmin <= 1e-6 * v_span;
    let (pole, rim) = match (flat(&loops[0]), flat(&loops[1])) {
        (true, false) => (&loops[0], &loops[1]),
        (false, true) => (&loops[1], &loops[0]),
        _ => return Ok(None),
    };
    let pole_v = 0.5 * (pole.vmin + pole.vmax);
    if (pole_v - v0).abs() > 1e-6 * v_span && (pole_v - v1).abs() > 1e-6 * v_span {
        return Ok(None);
    }

    let window_start = rim.points[0].x.min(rim.points[rim.points.len() - 1].x);
    let query_u = window_start + (point.x - window_start).rem_euclid(period);
    let mut crossings = Vec::new();
    for pair in rim.points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if (a.x > query_u) != (b.x > query_u) {
            crossings.push(a.y + (query_u - a.x) / (b.x - a.x) * (b.y - a.y));
        }
        for image in [-period, 0.0, period] {
            if point_segment_distance(
                Vec2 {
                    x: point.x + image,
                    y: point.y,
                },
                a,
                b,
            ) <= tolerance
            {
                return Ok(Some(PolygonClass::Boundary));
            }
        }
    }
    let Some(rim_v) = crossings
        .into_iter()
        .min_by(|a, b| (a - point.y).abs().total_cmp(&(b - point.y).abs()))
    else {
        return Ok(None);
    };
    let inside = if pole_v < rim_v {
        point.y <= rim_v + tolerance
    } else {
        point.y >= rim_v - tolerance
    };
    Ok(Some(if inside {
        PolygonClass::Inside
    } else {
        PolygonClass::Outside
    }))
}

/// Point-in-face for a SINGLY-PERIODIC (closed-u) FULL-BAND STRIP whose two
/// full-wrap rim loops are drawn as COVERING-PLANE pcurves running OUT OF the
/// u-domain by up to a whole period (case 08:10-genus face 1513, the 4th seam
/// representation after helmet-unwrapped / band-complement / 675-hops): the
/// raw even-odd sees u-winding loop polygons whose parity is meaningless, so
/// nearly the whole material strip reads Outside, the marched clip crumbles
/// the section curves to sub-mm bits, and the neighbours strand one-use.
///
/// The material region is the strip BETWEEN the two rims: on an open-v
/// surface a 2-loop face whose loops both wrap the full period has no other
/// representable region. The verdict is a sampled per-u hull — each rim is a
/// single periodic polyline v = rim(u) in the covering plane, and the query
/// point is between the rims iff an upward (+v) vertical ray at the query's
/// u (folded modulo the period into each rim's own one-period window) crosses
/// the two rim polylines an odd number of times in total. NO flat-level
/// approximation is used (the rims are scalloped: 1513's upper rim is flat at
/// v=1 on parts of u but dips to v=0.745 elsewhere — a v-between-extremes
/// rule is provably wrong on it).
///
/// STRUCTURAL gates (all must hold; anything else returns None and keeps the
/// previous classification bit-identically):
///   - surface closed in exactly u; exactly 2 loops;
///   - at least one loop's samples run OUT of the u-domain by >1e-3 period
///     (the covering-plane discriminator — in-domain 2-rim bands keep their
///     current path);
///   - each loop's unwrapped net u-travel is exactly ±one period and its v
///     closes (a true full-wrap rim, not a partial arc);
///   - opposite travel directions, disjoint v-hulls, and the material-left
///     winding rule confirming the BETWEEN strip (a complement verdict cannot
///     be represented on an open-v surface — decline, never guess).
/// Escape hatch: BREP_COVERING_RIM_STRIP=0.
fn covering_rim_strip_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    tolerance: f64,
) -> Result<Option<PolygonClass>, String> {
    if face.surface.closed_directions()? != (true, false) || face.loops.len() != 2 {
        return Ok(None);
    }
    if std::env::var("BREP_COVERING_RIM_STRIP").as_deref() == Ok("0") {
        return Ok(None);
    }
    let [u0, u1] = face.surface.domain_u()?;
    let [v0, v1] = face.surface.domain_v()?;
    let period = u1 - u0;
    if !(period > 0.0) {
        return Ok(None);
    }
    let v_span = (v1 - v0).abs().max(1e-30);
    struct Rim {
        points: Vec<Vec2>,
        vmin: f64,
        vmax: f64,
        ascending: bool,
    }
    let mut rims: Vec<Rim> = Vec::with_capacity(2);
    let mut out_of_domain = false;
    for loop_record in &face.loops {
        let mut points: Vec<Vec2> = Vec::new();
        for coedge in &loop_record.coedges {
            let curve = &coedge.pcurve;
            let [start, end] = curve.domain()?;
            let sample_count =
                2usize.max((interior_knot_count(&curve.knots, curve.degree) + 1) * (curve.degree + 1) * 4);
            for index in 0..=sample_count {
                let parameter = start + (end - start) * index as f64 / sample_count as f64;
                let evaluated = curve.evaluate(parameter)?;
                points.push(Vec2 {
                    x: evaluated.x,
                    y: evaluated.y,
                });
            }
        }
        if points.len() < 3 {
            return Ok(None);
        }
        // Covering-plane discriminator: this lane exists for loops drawn PAST
        // the domain; everything in-domain keeps the existing classifiers.
        for p in &points {
            if p.x < u0 - 1e-3 * period || p.x > u1 + 1e-3 * period {
                out_of_domain = true;
            }
        }
        // Unwrap into the covering plane (period-folded deltas — a no-op for
        // an already-continuous covering-plane representation, and the same
        // fold the horizon lane uses for in-domain seam hops).
        let mut unwrapped: Vec<Vec2> = Vec::with_capacity(points.len());
        let mut cursor = points[0];
        unwrapped.push(cursor);
        for pair in points.windows(2) {
            let du = pair[1].x - pair[0].x;
            let folded = du - period * (du / period).round();
            cursor = Vec2 {
                x: cursor.x + folded,
                y: pair[1].y,
            };
            unwrapped.push(cursor);
        }
        let net = unwrapped[unwrapped.len() - 1].x - unwrapped[0].x;
        if (net.abs() - period).abs() > 2e-2 * period {
            return Ok(None); // not a clean single full wrap
        }
        if (unwrapped[unwrapped.len() - 1].y - unwrapped[0].y).abs() > 1e-3 * v_span {
            return Ok(None); // rim does not close in v
        }
        let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &unwrapped {
            vmin = vmin.min(p.y);
            vmax = vmax.max(p.y);
        }
        rims.push(Rim {
            points: unwrapped,
            vmin,
            vmax,
            ascending: net > 0.0,
        });
    }
    if !out_of_domain {
        return Ok(None);
    }
    if rims[0].ascending == rims[1].ascending {
        return Ok(None); // rims must traverse opposite u directions
    }
    let (lower, upper) = if rims[0].vmax <= rims[1].vmin {
        (&rims[0], &rims[1])
    } else if rims[1].vmax <= rims[0].vmin {
        (&rims[1], &rims[0])
    } else {
        return Ok(None); // interleaved v-hulls: not a clean strip
    };
    if upper.vmin - lower.vmax <= 1e-6 * v_span {
        return Ok(None);
    }
    // Material-left rule (same convention as the doubly-periodic band lanes):
    // the between strip is CCW in uv iff the LOWER rim travels +u; that must
    // match the face sense, else the material would be the complement — which
    // an open-v surface cannot represent. Decline rather than guess.
    if (lower.ascending) != face.same_sense {
        return Ok(None);
    }
    // Boundary: proximity to either rim polyline at the query's period images.
    for rim in [lower, upper] {
        for shift in [-period, 0.0, period] {
            let image = Vec2 {
                x: point.x + shift,
                y: point.y,
            };
            for pair in rim.points.windows(2) {
                if point_segment_distance(image, pair[0], pair[1]) <= tolerance {
                    return Ok(Some(PolygonClass::Boundary));
                }
            }
        }
    }
    // Sampled per-u hull: fold the query into each rim's own one-period
    // window and count upward-ray crossings; odd total = between the rims.
    let mut crossings = 0usize;
    for rim in [lower, upper] {
        let window_base = rim.points[0].x.min(rim.points[rim.points.len() - 1].x);
        let x = window_base + (point.x - window_base).rem_euclid(period);
        for pair in rim.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if (a.x > x) != (b.x > x) {
                let v_cross = a.y + (x - a.x) / (b.x - a.x) * (b.y - a.y);
                if v_cross > point.y {
                    crossings += 1;
                }
            }
        }
    }
    Ok(Some(if crossings % 2 == 1 {
        PolygonClass::Inside
    } else {
        PolygonClass::Outside
    }))
}

/// The last few spherical regions built by [`sphere_chart_point_in_face`], keyed
/// by the exact inputs that determine them.
///
/// A region depends on the face's sphere, its orientation and its trim samples —
/// never on the query point — but the trim query is called once PER POINT from
/// the innermost loops of the imprint and the fragment builder, and building one
/// costs a surface evaluation per boundary sample, a 27-cell canonicalization
/// pass, two hash maps and a seed search that can run two dozen crossing counts.
/// Rebuilding that for every point of the same face is the whole cost of this
/// lane; the query itself is one crossing count.
///
/// The key is compared EXACTLY, not hashed: a signature is a few hundred `f64`s
/// and walking it costs a fraction of one surface evaluation, so there is no
/// reason to accept even a remote chance of answering from another face's region.
/// Four entries, most-recent-first — a boolean alternates between a handful of
/// faces at a time, and a miss only costs what this lane used to cost always.
struct SphereRegionCache {
    scratch: Vec<f64>,
    entries: Vec<(Vec<f64>, crate::sphere_chart::SphericalRegion)>,
}

const SPHERE_REGION_CACHE_ENTRIES: usize = 4;

thread_local! {
    static SPHERE_REGIONS: std::cell::RefCell<SphereRegionCache> = const {
        std::cell::RefCell::new(SphereRegionCache {
            scratch: Vec::new(),
            entries: Vec::new(),
        })
    };
}

/// One trim loop as [`parameter_point_in_face`] already sampled it: the closed
/// polygon it counts parity against, plus `(edge id, first sample, sample count)`
/// per coedge in loop order, which is all the spherical lane needs to drop a slit
/// and to rebuild the boundary as SEGMENTS rather than as one ring.
struct LoopSamples {
    polygon: Vec<Vec2>,
    coedges: Vec<(u64, usize, usize)>,
}

/// Point-in-face for a SPHERICAL carrier, decided on the ball itself instead of
/// in its polar parameter domain — but ONLY in the far field, where the generic
/// path has nothing better than a parity count.
///
/// The generic classifier below does not really answer by parity.  Whenever the
/// query has a nearest trim segment whose foot is interior to that pcurve, it
/// answers by which SIDE of that curve the point is on, which is locally exact
/// and is what the imprint and the fragment builder are tuned against.  Parity is
/// its fallback for points further from every trim segment than that segment is
/// long — and parity is exactly what the polar domain breaks: a loop enclosing a
/// pole does not enclose it in `uv` (it wraps the domain instead), and a region
/// straddling the seam is not one polygon there at all.  Three separate special
/// cases in this file exist because of that.
///
/// So this lane engages on the complement of the refinement's condition: it
/// replaces the parity count and nothing else.  Material is the intersection of
/// the regions to the LEFT of each trim loop, and the side a point falls on is
/// the sign of the signed solid angle that loop subtends there — a question with
/// no parameter domain in it, so pole and seam need no mention.  A collapsed pole
/// loop has no area and a seam is traversed once each way; both cancel exactly.
///
/// [`crate::sphere_chart::SphericalRegion`] is the same classifier the chart
/// tessellation uses, so the trim query and the mesh cannot disagree about where
/// the material is.
fn sphere_chart_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    sampled: &[LoopSamples],
) -> Result<Option<PolygonClass>, String> {
    if std::env::var("BREP_NO_SPHERE_CHARTS").is_ok()
        || std::env::var("BREP_NO_SPHERE_CHART_TRIM").is_ok()
    {
        return Ok(None);
    }
    let Some(atlas) = crate::sphere_chart::SphereAtlas::of_surface(&face.surface) else {
        return Ok(None);
    };
    // An edge used TWICE by one face is a slit — the parametric seam of a ball
    // the trim never cut. It bounds no material, so it is left out of the region
    // rather than cancelled numerically afterwards.
    let mut uses: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for coedge in face.loops.iter().flat_map(|record| record.coedges.iter()) {
        *uses.entry(coedge.edge_id).or_insert(0) += 1;
    }
    // Boundary SEGMENTS, not one polyline per loop.
    //
    // A ball drilled through carries BOTH rims in a single loop, joined by the
    // seam traversed once each way — a keyhole. Concatenating that loop's
    // surviving coedges and closing the ring would bridge rim to rim with two
    // chords that are not reverses of each other, and the winding of that figure
    // is not the winding of the two rims: on a through-drilled ball it comes out
    // exactly inverted. Handing the classifier the segments themselves lets it
    // cancel the seam and recover the two real rims.
    //
    // The samples are the CALLER's, taken once for its own parity count and its
    // own nearest-segment scan. Sampling the pcurves again here doubled the cost
    // of every query on a spherical face, and the caller only reaches this lane
    // when it would otherwise answer by parity — so that second pass was paid on
    // nearly every query and used on almost none of them.
    //
    // They are sampled in each coedge's own traversal direction, which is the
    // direction its pcurve is parameterized in: `forward` selects which end of
    // the EDGE's samples a traversal starts from, it does not reverse the
    // pcurve. A loop walked backwards puts the material on the wrong side of
    // every boundary.
    // The material-left convention is stated against the FACE normal, which the
    // generic path encodes as `(cross > 0) == same_sense` in uv: material is left
    // of the boundary seen from `same_sense ? Su x Sv : -(Su x Sv)`.
    let outward_face_normal =
        face.same_sense == atlas.parameterization_is_outward(&face.surface)?;
    SPHERE_REGIONS.with(|cache| {
        let SphereRegionCache { scratch, entries } = &mut *cache.borrow_mut();
        // The signature: everything the region is built from, and nothing that
        // varies with the query point.
        scratch.clear();
        scratch.extend_from_slice(&[
            atlas.centre.x,
            atlas.centre.y,
            atlas.centre.z,
            atlas.radius,
            if outward_face_normal { 1.0 } else { 0.0 },
        ]);
        for axis in atlas.basis {
            scratch.extend_from_slice(&[axis.x, axis.y, axis.z]);
        }
        // Where the per-coedge runs start, so the miss path below can read the
        // samples back out of the signature instead of re-deriving which coedges
        // survived.
        let header = scratch.len();
        for samples in sampled.iter() {
            let count = samples.polygon.len();
            if count < 2 {
                continue;
            }
            for &(edge_id, first, span) in &samples.coedges {
                if span == 0 || uses.get(&edge_id).copied().unwrap_or(0) >= 2 {
                    continue;
                }
                scratch.push(span as f64);
                // `span + 1` points: the coedge's own samples plus the first
                // sample of the next coedge, which is this one's END point. The
                // loop is closed, so the last coedge wraps back to `polygon[0]`.
                for offset in 0..=span {
                    let uv = samples.polygon[(first + offset) % count];
                    scratch.push(uv.x);
                    scratch.push(uv.y);
                }
            }
        }
        if let Some(index) = entries.iter().position(|(signature, _)| signature == scratch) {
            if index != 0 {
                entries.swap(0, index);
            }
        } else {
            let mut points: Vec<Vec3> = Vec::new();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            let mut cursor = header;
            while cursor < scratch.len() {
                let span = scratch[cursor] as usize;
                cursor += 1;
                let start = points.len();
                for index in 0..=span {
                    let uv = (scratch[cursor + 2 * index], scratch[cursor + 2 * index + 1]);
                    points.push(face.surface.evaluate(uv.0, uv.1)?);
                }
                cursor += 2 * (span + 1);
                spans.push((start, points.len()));
            }
            // One canonicalization over ALL the face's points: a vertex named by
            // two coedges is evaluated twice and the two answers agree only to
            // rounding, which the exact-reverse cancellation and the loop walk
            // both need collapsed.
            crate::sphere_chart::canonicalize_points(&mut points, 1e-9);
            let mut boundary: Vec<(Vec3, Vec3)> = Vec::new();
            for (first, last) in spans {
                for index in first..last.saturating_sub(1) {
                    boundary.push((points[index], points[index + 1]));
                }
            }
            let region = crate::sphere_chart::SphericalRegion::from_segments(
                atlas.centre,
                &boundary,
                outward_face_normal,
            );
            entries.insert(0, (scratch.clone(), region));
            entries.truncate(SPHERE_REGION_CACHE_ENTRIES);
        }
        let region = &entries[0].1;
        if region.is_whole_sphere() {
            return Ok(Some(PolygonClass::Inside));
        }
        if !region.is_decidable() {
            // No seed: decline to the generic path rather than answer Inside for
            // the whole ball.
            return Ok(None);
        }
        let probe = face.surface.evaluate(point.x, point.y)?;
        Ok(Some(if region.contains(atlas.centre, probe) {
            PolygonClass::Inside
        } else {
            PolygonClass::Outside
        }))
    })
}

pub fn parameter_point_in_face(
    face: &FaceRecord,
    point: Vec2,
    tolerance: f64,
) -> Result<PolygonClass, String> {
    if let Some(class) = seam_band_point_in_face(face, point, tolerance)? {
        return Ok(class);
    }
    if let Some(class) = winding_sphere_cap_point_in_face(face, point, tolerance)? {
        return Ok(class);
    }
    if let Some(class) = wrapped_horizon_point_in_face(face, point, tolerance)? {
        return Ok(class);
    }
    if let Some(class) = covering_rim_strip_point_in_face(face, point, tolerance)? {
        return Ok(class);
    }
    let mut crossings = 0;
    let mut boundary = false;
    let mut nearest: Option<(SegmentReference<'_>, f64)> = None;
    // Retained, not dropped per loop: the spherical lane below reads exactly
    // these samples instead of taking a second set of its own.
    let mut sampled: Vec<LoopSamples> = Vec::with_capacity(face.loops.len());
    for loop_record in &face.loops {
        let mut polygon = Vec::new();
        let mut segments = Vec::new();
        let mut coedges = Vec::with_capacity(loop_record.coedges.len());
        for coedge in &loop_record.coedges {
            let first = polygon.len();
            let curve = &coedge.pcurve;
            let [start, end] = curve.domain()?;
            let sample_count =
                2usize.max((interior_knot_count(&curve.knots, curve.degree) + 1) * (curve.degree + 1) * 4);
            for index in 0..sample_count {
                let parameter = start + (end - start) * index as f64 / sample_count as f64;
                let evaluated = curve.evaluate(parameter)?;
                polygon.push(Vec2 {
                    x: evaluated.x,
                    y: evaluated.y,
                });
                let parameter_end = if index + 1 < sample_count {
                    start + (end - start) * (index + 1) as f64 / sample_count as f64
                } else {
                    end
                };
                segments.push(SegmentReference {
                    start: Vec2 {
                        x: evaluated.x,
                        y: evaluated.y,
                    },
                    end: Vec2 { x: 0.0, y: 0.0 },
                    curve,
                    parameter_start: parameter,
                    parameter_end,
                });
            }
            coedges.push((coedge.edge_id, first, polygon.len() - first));
        }
        for index in 0..polygon.len() {
            segments[index].end = polygon[(index + 1) % polygon.len()];
        }
        match point_in_polygon(point, &polygon, tolerance) {
            PolygonClass::Boundary => boundary = true,
            PolygonClass::Inside => crossings += 1,
            PolygonClass::Outside => {}
        }
        for segment in segments {
            let distance = point_segment_distance(point, segment.start, segment.end);
            if nearest
                .as_ref()
                .is_none_or(|(_, nearest_distance)| distance < *nearest_distance)
            {
                nearest = Some((segment, distance));
            }
        }
        sampled.push(LoopSamples { polygon, coedges });
    }
    if boundary {
        return Ok(PolygonClass::Boundary);
    }
    let parity = if crossings % 2 == 1 {
        PolygonClass::Inside
    } else {
        PolygonClass::Outside
    };
    // The LAST of the special lanes, and the only one that engages here rather
    // than ahead of the scan: it substitutes for the parity count and for nothing
    // else. The four lanes above were each written for a shape the parity gets
    // wrong and are calibrated against real documents; pre-empting them would move
    // decisions this change has no business moving. Whenever the nearest-curve
    // refinement below can answer, it is more accurate than any global rule and is
    // what the imprint and the fragment builder are tuned against — so this asks
    // the sphere only on the complement of the refinement's own condition.
    let parity_decides = match nearest.as_ref() {
        None => true,
        Some((segment, distance)) => {
            let length = segment.end.sub(segment.start).length();
            *distance > length || length <= 0.0
        }
    };
    if parity_decides {
        if let Some(class) = sphere_chart_point_in_face(face, point, &sampled)? {
            return Ok(class);
        }
    }
    let Some((segment, distance)) = nearest else {
        return Ok(parity);
    };
    let segment_length = segment.end.sub(segment.start).length();
    if distance > segment_length || segment_length <= 0.0 {
        return Ok(parity);
    }
    let chord_parameter = segment.parameter_start
        + (segment.parameter_end - segment.parameter_start)
            * (point.sub(segment.start).dot(segment.end.sub(segment.start))
                / (segment_length * segment_length))
                .clamp(0.0, 1.0);
    let [domain_start, domain_end] = segment.curve.domain()?;
    let mut parameter = chord_parameter;
    for _ in 0..12 {
        let derivatives = segment.curve.derivatives(parameter, 2)?;
        let on_curve = Vec2 {
            x: derivatives[0].x,
            y: derivatives[0].y,
        };
        let tangent = Vec2 {
            x: derivatives[1].x,
            y: derivatives[1].y,
        };
        let second = Vec2 {
            x: derivatives[2].x,
            y: derivatives[2].y,
        };
        let residual = on_curve.sub(point);
        let denominator = tangent.dot(tangent) + residual.dot(second);
        if denominator.abs() < 1e-30 {
            break;
        }
        let step = -residual.dot(tangent) / denominator;
        parameter = (parameter + step).clamp(domain_start, domain_end);
        if step.abs() < 1e-14 * (domain_end - domain_start + 1.0) {
            break;
        }
    }
    let margin = 1e-9 * (domain_end - domain_start);
    if parameter > domain_start + margin && parameter < domain_end - margin {
        let derivatives = segment.curve.derivatives(parameter, 1)?;
        let on_curve = Vec2 {
            x: derivatives[0].x,
            y: derivatives[0].y,
        };
        let tangent = Vec2 {
            x: derivatives[1].x,
            y: derivatives[1].y,
        };
        let offset = point.sub(on_curve);
        if offset.length() <= tolerance {
            return Ok(PolygonClass::Boundary);
        }
        let cross = tangent.x * offset.y - tangent.y * offset.x;
        if cross.abs() > 1e-30 {
            return Ok(if (cross > 0.0) == face.same_sense {
                PolygonClass::Inside
            } else {
                PolygonClass::Outside
            });
        }
    }
    Ok(parity)
}
