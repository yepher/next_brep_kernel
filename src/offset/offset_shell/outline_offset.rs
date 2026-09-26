//! The OUTWARD wall of a planar opening: the opening's plane trimmed by its
//! outer outline OFFSET outward by the pad, in the plane.
//!
//! An outward shell holds each opening face in place and extends it until it
//! meets the grown offsets of the retained faces around it. The wall used to be
//! built by padding the plane's 2 × 2 net under cloned pcurves, which SCALES the
//! outline about the patch centre. That is the offset outline only for a
//! rectangle filling its patch, or a circle centred on it. On an L-shaped
//! opening the two notch edges pass through the centre and do not move at all,
//! so the notch faces' grown offsets cross the plane outside the wall and the
//! shell reads "completion produced non-integral genus"
//! (`l-step-open-bottom`, d = −1.5, in the membrane census).
//!
//! Here each outline element is offset along its own outward normal: a line to
//! the parallel line, a circular arc to the concentric arc at `r ± pad`. Two
//! neighbours meet where their offsets intersect, nearest the vertex's own
//! offset, which mitres a convex corner and trims a reflex one; at a tangent
//! junction the offsets already meet at `v + n·pad`. Every edge and pcurve is
//! exact: a line or arc in 3D, and its image under the plane's affine
//! parameterization in uv, which is exact for a rational curve because an
//! affine map commutes with the rational combination.
//!
//! What this declines, with the reason: an outline element that is neither a
//! line nor a circular arc, a concave arc whose offset radius reaches zero, an
//! offset element that reverses or collapses, a junction the two offsets never
//! reach, and an offset outline that crosses itself.

use super::*;
use crate::{make_arc, make_line, NurbsCurve, NurbsSurface, Vec4};
use std::f64::consts::{PI, TAU};

type P2 = [f64; 2];

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}
fn scale(a: P2, k: f64) -> P2 {
    [a[0] * k, a[1] * k]
}
fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn length(a: P2) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: P2) -> P2 {
    scale(a, 1.0 / length(a))
}

/// One outline element in the plane's orthonormal frame, in traversal order.
#[derive(Clone, Copy, Debug)]
enum Element {
    Line { start: P2, end: P2 },
    /// `sweep` is the signed angle travelled from `start` to `end`: positive
    /// counter-clockwise in the frame. A whole circle is ±2π.
    Arc { centre: P2, radius: f64, start: P2, end: P2, sweep: f64 },
}

impl Element {
    fn start(&self) -> P2 {
        match self {
            Element::Line { start, .. } | Element::Arc { start, .. } => *start,
        }
    }
    fn end(&self) -> P2 {
        match self {
            Element::Line { end, .. } | Element::Arc { end, .. } => *end,
        }
    }
    fn tangent_at(&self, point: P2) -> P2 {
        match self {
            Element::Line { start, end } => unit(sub(*end, *start)),
            Element::Arc { centre, sweep, .. } => {
                let radial = unit(sub(point, *centre));
                if *sweep > 0.0 {
                    [-radial[1], radial[0]]
                } else {
                    [radial[1], -radial[0]]
                }
            }
        }
    }
    fn sample(&self, fraction: f64) -> P2 {
        match self {
            Element::Line { start, end } => add(*start, scale(sub(*end, *start), fraction)),
            Element::Arc { centre, radius, start, sweep, .. } => {
                let angle = (start[1] - centre[1]).atan2(start[0] - centre[0]) + sweep * fraction;
                [centre[0] + radius * angle.cos(), centre[1] + radius * angle.sin()]
            }
        }
    }
}

/// The plane's frame: an origin, two orthonormal in-plane axes, and the affine
/// map back to the opening surface's own (u, v).
struct PlaneFrame {
    origin: Vec3,
    e1: Vec3,
    e2: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    domain: [f64; 4],
}

impl PlaneFrame {
    fn new(surface: &NurbsSurface) -> Result<Self, String> {
        let [u0, u1] = surface.domain_u()?;
        let [v0, v1] = surface.domain_v()?;
        let origin = surface.evaluate(u0, v0)?;
        let axis_u = surface.evaluate(u1, v0)?.sub(origin);
        let axis_v = surface.evaluate(u0, v1)?.sub(origin);
        let normal = axis_u.cross(axis_v).normalized()?;
        let e1 = axis_u.normalized()?;
        let e2 = normal.cross(e1);
        Ok(Self { origin, e1, e2, axis_u, axis_v, domain: [u0, u1, v0, v1] })
    }
    fn flat(&self, point: Vec3) -> P2 {
        let relative = point.sub(self.origin);
        [relative.dot(self.e1), relative.dot(self.e2)]
    }
    fn world(&self, point: P2) -> Vec3 {
        self.origin.add(self.e1.scale(point[0])).add(self.e2.scale(point[1]))
    }
    fn uv(&self, point: Vec3) -> Vec3 {
        let relative = point.sub(self.origin);
        let (a, b) = (relative.dot(self.axis_u), relative.dot(self.axis_v));
        let (guu, guv, gvv) = (
            self.axis_u.dot(self.axis_u),
            self.axis_u.dot(self.axis_v),
            self.axis_v.dot(self.axis_v),
        );
        let determinant = guu * gvv - guv * guv;
        let s = (a * gvv - b * guv) / determinant;
        let t = (b * guu - a * guv) / determinant;
        let [u0, u1, v0, v1] = self.domain;
        Vec3::new(u0 + s * (u1 - u0), v0 + t * (v1 - v0), 0.0)
    }
    fn at_uv(&self, u: f64, v: f64) -> Vec3 {
        let [u0, u1, v0, v1] = self.domain;
        self.origin
            .add(self.axis_u.scale((u - u0) / (u1 - u0)))
            .add(self.axis_v.scale((v - v0) / (v1 - v0)))
    }
}

/// The circle through three points, or `None` when they are collinear.
fn circle_through(a: P2, b: P2, c: P2) -> Option<(P2, f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() <= 1e-300 {
        return None;
    }
    let (aa, bb, cc) = (dot(a, a), dot(b, b), dot(c, c));
    let centre = [
        (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d,
        (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d,
    ];
    Some((centre, length(sub(a, centre))))
}

/// Read one coedge's edge as a line or a circular arc in the frame, traversed
/// in the coedge's direction.
fn classify_edge(
    frame: &PlaneFrame,
    edge: &EdgeRecord,
    forward: bool,
    start: Vec3,
    end: Vec3,
    tolerance: f64,
) -> Result<Result<Element, String>, String> {
    // Samples in the COEDGE's direction.
    let sample = |fraction: f64| -> Result<P2, String> {
        let along = if forward { fraction } else { 1.0 - fraction };
        Ok(frame.flat(edge.curve.evaluate(edge.t0 + (edge.t1 - edge.t0) * along)?))
    };
    let (a, b) = (frame.flat(start), frame.flat(end));
    let samples = (1..16)
        .map(|index| sample(index as f64 / 16.0))
        .collect::<Result<Vec<_>, _>>()?;
    let chord = sub(b, a);
    if length(chord) > tolerance
        && samples
            .iter()
            .all(|p| (cross(sub(*p, a), chord) / length(chord)).abs() <= tolerance)
    {
        return Ok(Ok(Element::Line { start: a, end: b }));
    }
    let middle = samples[7];
    // A closed edge (a whole circle) has start == end: fit through three
    // interior samples instead.
    let fit = if length(chord) > tolerance {
        circle_through(a, middle, b)
    } else {
        circle_through(samples[2], samples[7], samples[12])
    };
    let Some((centre, radius)) = fit else {
        return Ok(Err(format!("outline edge {} is neither a line nor an arc", edge.id)));
    };
    if samples
        .iter()
        .chain([&a, &b])
        .any(|p| (length(sub(*p, centre)) - radius).abs() > tolerance)
    {
        return Ok(Err(format!("outline edge {} is neither a line nor a circular arc", edge.id)));
    }
    // The signed sweep: accumulate the turning angle about the centre over the
    // samples, which also reads a whole circle as ±2π.
    let mut sweep = 0.0;
    let mut previous = a;
    for point in samples.iter().chain([&b]) {
        sweep += cross(sub(previous, centre), sub(*point, centre))
            .atan2(dot(sub(previous, centre), sub(*point, centre)));
        previous = *point;
    }
    Ok(Ok(Element::Arc { centre, radius, start: a, end: b, sweep }))
}

/// Intersections of two offset elements treated as whole lines and circles.
fn intersections(first: &Element, second: &Element) -> Vec<P2> {
    let line_circle = |p: P2, d: P2, c: P2, r: f64| -> Vec<P2> {
        let f = sub(p, c);
        let (b, cc) = (dot(f, d), dot(f, f) - r * r);
        let disc = b * b - cc;
        if disc < 0.0 {
            return Vec::new();
        }
        let root = disc.sqrt();
        vec![add(p, scale(d, -b - root)), add(p, scale(d, -b + root))]
    };
    match (first, second) {
        (Element::Line { start: p, end: q }, Element::Line { start: r, end: s }) => {
            let (d, e) = (unit(sub(*q, *p)), unit(sub(*s, *r)));
            let denominator = cross(d, e);
            if denominator.abs() <= 1e-12 {
                return Vec::new();
            }
            let t = cross(sub(*r, *p), e) / denominator;
            vec![add(*p, scale(d, t))]
        }
        (Element::Line { start: p, end: q }, Element::Arc { centre, radius, .. })
        | (Element::Arc { centre, radius, .. }, Element::Line { start: p, end: q }) => {
            line_circle(*p, unit(sub(*q, *p)), *centre, *radius)
        }
        (
            Element::Arc { centre: c1, radius: r1, .. },
            Element::Arc { centre: c2, radius: r2, .. },
        ) => {
            let between = sub(*c2, *c1);
            let distance = length(between);
            if distance <= 1e-12 || distance > r1 + r2 || distance < (r1 - r2).abs() {
                return Vec::new();
            }
            let a = (r1 * r1 - r2 * r2 + distance * distance) / (2.0 * distance);
            let h = (r1 * r1 - a * a).max(0.0).sqrt();
            let base = add(*c1, scale(between, a / distance));
            let normal = [-between[1] / distance, between[0] / distance];
            vec![add(base, scale(normal, h)), add(base, scale(normal, -h))]
        }
    }
}

fn polar(point: P2, centre: P2) -> f64 {
    (point[1] - centre[1]).atan2(point[0] - centre[0])
}

/// The planar opening's outward wall over its outline offset by `pad`, or the
/// reason it is not built.
pub(super) fn offset_outline_wall(
    source: &BrepSolid,
    face: &FaceRecord,
    pad: f64,
) -> Result<Result<BrepSolid, String>, String> {
    if !face.surface.is_affine()? {
        return Ok(Err("the opening is not planar".into()));
    }
    let tolerance = 1e-6 * crate::solid_scale(source).max(1.0);
    let frame = PlaneFrame::new(&face.surface)?;
    let edge_by_id = source.edges.iter().map(|edge| (edge.id, edge)).collect::<HashMap<_, _>>();
    let vertex_by_id = source
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.point))
        .collect::<HashMap<_, _>>();
    // The outer loop: the largest |uv area|, as the scaled wall keeps.
    let mut outer = None;
    let mut outer_area = f64::NEG_INFINITY;
    for (index, record) in face.loops.iter().enumerate() {
        let area = parameter_space_area(&FaceRecord {
            id: 0,
            surface: face.surface.clone(),
            same_sense: true,
            loops: vec![record.clone()],
            name: None,
        })?
        .abs();
        if area > outer_area {
            outer_area = area;
            outer = Some(index);
        }
    }
    let Some(outer) = outer else {
        return Ok(Err("the opening has no loop".into()));
    };
    let mut elements = Vec::new();
    for coedge in &face.loops[outer].coedges {
        let edge = edge_by_id
            .get(&coedge.edge_id)
            .ok_or_else(|| format!("offset_shell: missing edge {}", coedge.edge_id))?;
        if edge.degenerate {
            continue;
        }
        let (start_id, end_id) = if coedge.forward {
            (edge.start_vertex_id, edge.end_vertex_id)
        } else {
            (edge.end_vertex_id, edge.start_vertex_id)
        };
        let (start, end) = (vertex_by_id[&start_id], vertex_by_id[&end_id]);
        match classify_edge(&frame, edge, coedge.forward, start, end, tolerance)? {
            Ok(element) => elements.push(element),
            Err(reason) => return Ok(Err(reason)),
        }
    }
    let count = elements.len();
    if count == 0 {
        return Ok(Err("the outline has no edge".into()));
    }
    // Orientation from a dense sample of the whole loop.
    let mut ring = Vec::new();
    for element in &elements {
        for index in 0..16 {
            ring.push(element.sample(index as f64 / 16.0));
        }
    }
    let signed_area = (0..ring.len())
        .map(|index| cross(ring[index], ring[(index + 1) % ring.len()]))
        .sum::<f64>()
        * 0.5;
    if signed_area.abs() <= tolerance * tolerance {
        return Ok(Err("the outline encloses no area".into()));
    }
    let outward = |tangent: P2| {
        if signed_area > 0.0 {
            [tangent[1], -tangent[0]]
        } else {
            [-tangent[1], tangent[0]]
        }
    };
    // Each element's offset, as a whole line or circle.
    let mut offsets = Vec::with_capacity(count);
    for element in &elements {
        match *element {
            Element::Line { start, end } => {
                let n = outward(element.tangent_at(start));
                offsets.push(Element::Line { start: add(start, scale(n, pad)), end: add(end, scale(n, pad)) });
            }
            Element::Arc { centre, radius, start, end, sweep } => {
                let middle = element.sample(0.5);
                let sign = dot(outward(element.tangent_at(middle)), unit(sub(middle, centre))).signum();
                let grown = radius + sign * pad;
                if grown <= tolerance {
                    return Ok(Err(format!(
                        "a concave outline arc of radius {radius:.6} has no offset at {pad:.6}"
                    )));
                }
                let k = grown / radius;
                offsets.push(Element::Arc {
                    centre,
                    radius: grown,
                    start: add(centre, scale(sub(start, centre), k)),
                    end: add(centre, scale(sub(end, centre), k)),
                    sweep,
                });
            }
        }
    }
    // The junction at each element's START, between its predecessor and it.
    let mut junctions = Vec::with_capacity(count);
    for index in 0..count {
        let previous = &elements[(index + count - 1) % count];
        let current = &elements[index];
        let vertex = current.start();
        let n_in = outward(previous.tangent_at(previous.end()));
        let n_out = outward(current.tangent_at(vertex));
        let naive_in = add(vertex, scale(n_in, pad));
        let naive_out = add(vertex, scale(n_out, pad));
        if length(sub(n_in, n_out)) <= 1e-9 {
            junctions.push(naive_out);
            continue;
        }
        let target = scale(add(naive_in, naive_out), 0.5);
        let best = intersections(&offsets[(index + count - 1) % count], &offsets[index])
            .into_iter()
            .min_by(|a, b| length(sub(*a, target)).total_cmp(&length(sub(*b, target))));
        let Some(junction) = best else {
            return Ok(Err(format!("the offsets meeting at outline corner {index} never meet")));
        };
        // A corner so sharp that its mitre runs far past the pad is not a
        // wall this construction should build.
        if length(sub(junction, vertex)) > 12.0 * pad + tolerance {
            return Ok(Err(format!("outline corner {index} mitres past 12 pads")));
        }
        junctions.push(junction);
    }
    // The grown elements, between consecutive junctions.
    let mut grown = Vec::with_capacity(count);
    for index in 0..count {
        let (from, to) = (junctions[index], junctions[(index + 1) % count]);
        match (elements[index], offsets[index]) {
            (Element::Line { start, end }, _) => {
                if dot(sub(to, from), sub(end, start)) <= tolerance {
                    return Ok(Err(format!("offset outline edge {index} collapses or reverses")));
                }
                grown.push(Element::Line { start: from, end: to });
            }
            (Element::Arc { sweep, .. }, Element::Arc { centre, radius, .. }) => {
                let (a, b) = (polar(from, centre), polar(to, centre));
                let mut travelled = b - a;
                if sweep > 0.0 {
                    travelled = travelled.rem_euclid(TAU);
                    if travelled <= 1e-9 && sweep > PI {
                        travelled = TAU;
                    }
                } else {
                    travelled = -(-travelled).rem_euclid(TAU);
                    if travelled >= -1e-9 && sweep < -PI {
                        travelled = -TAU;
                    }
                }
                if travelled.abs() <= 1e-9 || (travelled - sweep).abs() > PI {
                    return Ok(Err(format!("offset outline arc {index} collapses or reverses")));
                }
                grown.push(Element::Arc { centre, radius, start: from, end: to, sweep: travelled });
            }
            _ => unreachable!("an offset keeps its element's kind"),
        }
    }
    // Simple: no two non-adjacent pieces of the sampled outline cross.
    let mut polyline = Vec::new();
    for element in &grown {
        let steps = if matches!(element, Element::Line { .. }) { 1 } else { 24 };
        for index in 0..steps {
            polyline.push(element.sample(index as f64 / steps as f64));
        }
    }
    let n = polyline.len();
    let proper = |a: P2, b: P2, c: P2, d: P2| {
        let (o1, o2) = (cross(sub(b, a), sub(c, a)), cross(sub(b, a), sub(d, a)));
        let (o3, o4) = (cross(sub(d, c), sub(a, c)), cross(sub(d, c), sub(b, c)));
        o1 * o2 < 0.0 && o3 * o4 < 0.0
    };
    for i in 0..n {
        for j in i + 2..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if proper(polyline[i], polyline[(i + 1) % n], polyline[j], polyline[(j + 1) % n]) {
                return Ok(Err("the offset outline crosses itself".into()));
            }
        }
    }

    // Build the wall: the same plane, its net enlarged over the grown outline.
    let uv_ring = polyline.iter().map(|p| frame.uv(frame.world(*p))).collect::<Vec<_>>();
    let margin = 1e-9;
    let ua = uv_ring.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - margin;
    let ub = uv_ring.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + margin;
    let va = uv_ring.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - margin;
    let vb = uv_ring.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + margin;
    let net = vec![
        vec![Vec4::from_point(frame.at_uv(ua, va), 1.0), Vec4::from_point(frame.at_uv(ua, vb), 1.0)],
        vec![Vec4::from_point(frame.at_uv(ub, va), 1.0), Vec4::from_point(frame.at_uv(ub, vb), 1.0)],
    ];
    let surface = NurbsSurface::new(1, 1, vec![ua, ua, ub, ub], vec![va, va, vb, vb], net)?;
    let closed = count == 1;
    let vertices = (0..count)
        .map(|index| VertexRecord { id: index as u64 + 1, point: frame.world(junctions[index]) })
        .collect::<Vec<_>>();
    let mut edges = Vec::with_capacity(count);
    let mut coedges = Vec::with_capacity(count);
    for (index, element) in grown.iter().enumerate() {
        let curve = match *element {
            Element::Line { start, end } => make_line(frame.world(start), frame.world(end))?,
            Element::Arc { centre, radius, start, sweep, .. } => {
                let y_axis = if sweep > 0.0 { frame.e2 } else { frame.e2.scale(-1.0) };
                let local = sub(start, centre);
                let start_angle = if sweep > 0.0 { local[1].atan2(local[0]) } else { (-local[1]).atan2(local[0]) };
                make_arc(frame.world(centre), frame.e1, y_axis, radius, start_angle, start_angle + sweep.abs())?
            }
        };
        let pcurve = NurbsCurve::new(
            curve.degree,
            curve.knots.clone(),
            curve
                .control_points
                .iter()
                .map(|control| Ok(Vec4::from_point(frame.uv(control.point()?), control.w)))
                .collect::<Result<Vec<_>, String>>()?,
        )?;
        let [t0, t1] = curve.domain()?;
        let next = if closed { 0 } else { (index + 1) % count };
        edges.push(EdgeRecord {
            id: index as u64 + 1,
            curve,
            t0,
            t1,
            start_vertex_id: index as u64 + 1,
            end_vertex_id: next as u64 + 1,
            degenerate: false,
            name: None,
        });
        coedges.push(CoedgeRecord { id: index as u64 + 1, edge_id: index as u64 + 1, forward: true, pcurve });
    }
    Ok(Ok(BrepSolid {
        id: source.id,
        vertices,
        edges,
        shells: vec![ShellRecord {
            id: 1,
            faces: vec![FaceRecord {
                id: face.id,
                surface,
                same_sense: face.same_sense,
                loops: vec![LoopRecord { id: 1, coedges }],
                name: face.name.clone(),
            }],
        }],
        genus: 0,
    }))
}

/// Whether two wall carriers trim the same region: every sampled point of each
/// outline lies within `tolerance` of the other's TRIMMED BOUNDARY CURVES.
///
/// The scaled wall IS the offset outline for a rectangle filling its patch and
/// for a circle centred on it, which is most openings. Where the two agree, the
/// scaled wall is kept: it is the construction every existing shell was built
/// and named by, and rebuilding an identical region on a different patch would
/// renumber the fragments it later cuts into (measured: the outward
/// sphere-under-frustum case row's worst vector-area face moved from
/// `P.CO2_S_1` to `P.CO2_S`, with its volume, area and counts unchanged).
///
/// The comparison is point-to-CURVE, not polyline-to-polyline, because the two
/// walls do not share a parameterization: the scaled wall's outline is the
/// source's own pcurves, and this module's is `make_arc`'s. On a rim a boolean
/// has split, a 48-gon of one is a chord of the other, and the two read
/// `R(1 − cos(θ/2))` apart — 1e-4 of the radius, four decades over the band —
/// so a sampled comparison sends an already-correct wall down the rebuild.
/// A split rim also shows what the scaled wall does there, which is why it takes
/// this lane either way: on a cylinder whose top rim is split at t = 0.37, the
/// scaled wall's outline runs from radius 2.499869769 to 2.501280935 where the
/// offset outline is 2.5 exactly. Scaling two arcs about a sampled trim box is
/// not a circle.
pub(super) fn walls_trim_the_same_region(
    first: &BrepSolid,
    second: &BrepSolid,
    tolerance: f64,
) -> Result<bool, String> {
    let stations = |solid: &BrepSolid| -> Result<Vec<Vec3>, String> {
        let face = &solid.shells[0].faces[0];
        let mut points = Vec::new();
        for record in &face.loops {
            for coedge in &record.coedges {
                let [t0, t1] = coedge.pcurve.domain()?;
                for index in 0..24 {
                    let uv = coedge.pcurve.evaluate(t0 + (t1 - t0) * index as f64 / 24.0)?;
                    points.push(face.surface.evaluate(uv.x, uv.y)?);
                }
            }
        }
        Ok(points)
    };
    // The distance from a point to a solid's trimmed boundary curves, exactly
    // for the lines and rational arcs both walls are made of.
    let to_boundary = |point: Vec3, solid: &BrepSolid| -> Result<f64, String> {
        let mut nearest = f64::INFINITY;
        for edge in &solid.edges {
            let projection = crate::project_point_to_curve(&edge.curve, point)?;
            let (low, high) = (edge.t0.min(edge.t1), edge.t0.max(edge.t1));
            let clamped = projection.u.clamp(low, high);
            nearest = nearest.min(edge.curve.evaluate(clamped)?.sub(point).length());
        }
        Ok(nearest)
    };
    let (a, b) = (stations(first)?, stations(second)?);
    if a.is_empty() || b.is_empty() || first.edges.is_empty() || second.edges.is_empty() {
        return Ok(false);
    }
    for point in &a {
        if to_boundary(*point, second)? > tolerance {
            return Ok(false);
        }
    }
    for point in &b {
        if to_boundary(*point, first)? > tolerance {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the SCALED outward wall still covers the opening's offset outline
/// wherever a retained neighbour's grown offset must cut it: `Ok(Ok(()))`, or
/// the worst uncovered distance.
///
/// The scaled wall is the fallback for an outline this module does not offset.
/// It does not have to BE the offset outline, only to contain it along every
/// edge a retained face meets, so each neighbour's section lies inside the
/// wall's trim. That holds when the outline is star-shaped about the scale
/// centre with the pad to spare, and fails exactly where it did on the L-step:
/// an edge through the centre does not move.
pub(super) fn scaled_wall_covers_offset(
    source: &BrepSolid,
    face: &FaceRecord,
    wall: &BrepSolid,
    pad: f64,
    source_faces: &[&FaceRecord],
    opening_set: &HashSet<u64>,
) -> Result<Result<(), f64>, String> {
    let tolerance = 1e-6 * crate::solid_scale(source).max(1.0);
    let frame = PlaneFrame::new(&face.surface)?;
    let edge_by_id = source.edges.iter().map(|edge| (edge.id, edge)).collect::<HashMap<_, _>>();
    // The scaled wall's outline, sampled, in the source plane's frame.
    let wall_face = &wall.shells[0].faces[0];
    let mut hull = Vec::new();
    for record in &wall_face.loops {
        for coedge in &record.coedges {
            let [t0, t1] = coedge.pcurve.domain()?;
            for index in 0..32 {
                let uv = coedge.pcurve.evaluate(t0 + (t1 - t0) * index as f64 / 32.0)?;
                hull.push(frame.flat(wall_face.surface.evaluate(uv.x, uv.y)?));
            }
        }
    }
    let inside = |point: P2| -> f64 {
        // 0 when inside or on the hull, else the distance to it.
        let mut winding = 0i32;
        let mut nearest = f64::INFINITY;
        for index in 0..hull.len() {
            let (a, b) = (hull[index], hull[(index + 1) % hull.len()]);
            let along = sub(b, a);
            let t = (dot(sub(point, a), along) / dot(along, along).max(1e-300)).clamp(0.0, 1.0);
            nearest = nearest.min(length(sub(point, add(a, scale(along, t)))));
            if a[1] <= point[1] {
                if b[1] > point[1] && cross(along, sub(point, a)) > 0.0 {
                    winding += 1;
                }
            } else if b[1] <= point[1] && cross(along, sub(point, a)) < 0.0 {
                winding -= 1;
            }
        }
        if winding != 0 { 0.0 } else { nearest }
    };
    // The source outline's orientation and its retained-edge offsets.
    let outer = face
        .loops
        .iter()
        .max_by(|a, b| {
            let area = |record: &LoopRecord| {
                parameter_space_area(&FaceRecord {
                    id: 0,
                    surface: face.surface.clone(),
                    same_sense: true,
                    loops: vec![record.clone()],
                    name: None,
                })
                .map(f64::abs)
                .unwrap_or(0.0)
            };
            area(a).total_cmp(&area(b))
        })
        .ok_or_else(|| "offset_shell: opening has no loop".to_string())?;
    let mut samples = Vec::new();
    for coedge in &outer.coedges {
        let edge = edge_by_id[&coedge.edge_id];
        for index in 0..=32 {
            let along = index as f64 / 32.0;
            let along = if coedge.forward { along } else { 1.0 - along };
            samples.push((coedge.edge_id, frame.flat(edge.curve.evaluate(edge.t0 + (edge.t1 - edge.t0) * along)?)));
        }
    }
    let signed_area = (0..samples.len())
        .map(|index| cross(samples[index].1, samples[(index + 1) % samples.len()].1))
        .sum::<f64>()
        * 0.5;
    let mut worst: f64 = 0.0;
    for index in 0..samples.len() {
        let (edge_id, point) = samples[index];
        let Some((mate, _)) = junction_mate(source_faces, face, edge_id) else {
            continue;
        };
        if opening_set.contains(&mate.id) {
            continue;
        }
        let (previous, next) = (samples[(index + samples.len() - 1) % samples.len()].1, samples[(index + 1) % samples.len()].1);
        let chord = sub(next, previous);
        if length(chord) <= tolerance {
            continue;
        }
        let tangent = unit(chord);
        let n = if signed_area > 0.0 { [tangent[1], -tangent[0]] } else { [-tangent[1], tangent[0]] };
        worst = worst.max(inside(add(point, scale(n, pad))));
    }
    Ok(if worst <= tolerance * 10.0 { Ok(()) } else { Err(worst) })
}
