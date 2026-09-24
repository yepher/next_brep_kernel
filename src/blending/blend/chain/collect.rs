use super::*;

pub(super) struct ChainSegment<'a> {
    pub(super) edge: &'a EdgeRecord,
    /// Chain traversal: true when the chain runs t0 -> t1.
    pub(super) forward: bool,
    pub(super) first: BlendMate<'a>,
    pub(super) second: BlendMate<'a>,
}

impl<'a> ChainSegment<'a> {
    pub(super) fn mate(&self, side: usize) -> &BlendMate<'a> {
        if side == 0 {
            &self.first
        } else {
            &self.second
        }
    }
}

/// A tangent-continuous edge run, closed or terminated at two free ends.
pub(super) struct SmoothChain<'a> {
    pub(super) segments: Vec<ChainSegment<'a>>,
    pub(super) closed: bool,
    /// Entry and exit vertices; unused for closed chains.
    pub(super) start_vertex: u64,
    pub(super) end_vertex: u64,
}

type ChainContinuation<'a> = (&'a EdgeRecord, bool, u64, BlendMate<'a>, BlendMate<'a>);

/// The single conjugated (G1-continuous) continuation of a chain at
/// `current_vertex`, or `None` at a free end.  `extend_forward` chooses
/// whether the new segment ENTERS the chain here (forward walk) or EXITS
/// here (backward walk); `arrive_tangent` is the chain-forward tangent the
/// existing chain has at this vertex and `reference_first_normal` its
/// first-side outward normal, so the new segment keeps the same first side.
fn find_conjugate<'a>(
    solid: &'a BrepSolid,
    current_vertex: u64,
    arrive_tangent: Vec3,
    reference_first_normal: Vec3,
    exclude_edge: u64,
    extend_forward: bool,
) -> Result<Option<ChainContinuation<'a>>, String> {
    let mut next: Option<ChainContinuation<'a>> = None;
    for candidate in &solid.edges {
        if candidate.id == exclude_edge || candidate.start_vertex_id == candidate.end_vertex_id {
            continue;
        }
        // The endpoint touching `current_vertex`, the chain-forward flag,
        // the edge parameter there, and the far vertex.
        let (forward, join_t, other_vertex) = if extend_forward {
            // The candidate ENTERS the chain at `current_vertex`.
            if candidate.start_vertex_id == current_vertex {
                (true, candidate.t0, candidate.end_vertex_id)
            } else if candidate.end_vertex_id == current_vertex {
                (false, candidate.t1, candidate.start_vertex_id)
            } else {
                continue;
            }
        } else {
            // The candidate EXITS the chain at `current_vertex`.
            if candidate.end_vertex_id == current_vertex {
                (true, candidate.t1, candidate.start_vertex_id)
            } else if candidate.start_vertex_id == current_vertex {
                (false, candidate.t0, candidate.end_vertex_id)
            } else {
                continue;
            }
        };
        let derivatives = candidate.curve.derivatives(join_t, 1)?;
        let tangent = derivatives[1].normalized()?;
        let chain_tangent = if forward {
            tangent
        } else {
            tangent.scale(-1.0)
        };
        if chain_tangent.dot(arrive_tangent) < 1.0 - 1e-6 {
            continue;
        }
        if next.is_some() {
            return Err("blend: ambiguous conjugated continuation at a chain vertex".into());
        }
        let (face_a, loop_a, coedge_a) = locate_mate(solid, candidate.id, None)?;
        let (face_b, loop_b, coedge_b) =
            locate_mate(solid, candidate.id, Some((face_a.id, loop_a)))?;
        let normal_a = outward_normal_at(face_a, coedge_a, candidate, join_t)?;
        let side_a_first = normal_a.dot(reference_first_normal) > 0.9;
        let normal_b = outward_normal_at(face_b, coedge_b, candidate, join_t)?;
        let side_b_first = normal_b.dot(reference_first_normal) > 0.9;
        if side_a_first == side_b_first {
            return Err("blend: cannot assign chain sides (normals ambiguous)".into());
        }
        let (first, second) = if side_a_first {
            (
                BlendMate {
                    face: face_a,
                    coedge: coedge_a,
                    loop_index: loop_a,
                    rho: 0.0,
                },
                BlendMate {
                    face: face_b,
                    coedge: coedge_b,
                    loop_index: loop_b,
                    rho: 0.0,
                },
            )
        } else {
            (
                BlendMate {
                    face: face_b,
                    coedge: coedge_b,
                    loop_index: loop_b,
                    rho: 0.0,
                },
                BlendMate {
                    face: face_a,
                    coedge: coedge_a,
                    loop_index: loop_a,
                    rho: 0.0,
                },
            )
        };
        next = Some((candidate, forward, other_vertex, first, second));
    }
    Ok(next)
}

fn outward_normal_at(
    face: &FaceRecord,
    coedge: &CoedgeRecord,
    edge: &EdgeRecord,
    t: f64,
) -> Result<Vec3, String> {
    let uv = edge_uv_on_face(coedge, edge, t)?;
    let normal = raw_normal(&face.surface, uv[0], uv[1])?;
    Ok(if face.same_sense {
        normal
    } else {
        normal.scale(-1.0)
    })
}

/// Collect the chain of conjugated (G1-continuous) edges through
/// `seed_edge_id`, assigning each segment's mates so the "first" side is
/// continuous along the chain.  Walks FORWARD from the seed until it either
/// closes back to the seed start (a CLOSED chain) or reaches a free end
/// (no conjugated continuation); an unclosed forward walk then also walks
/// BACKWARD from the seed start, so the returned OPEN chain runs free end to
/// free end in order.
pub(super) fn collect_smooth_chain(solid: &BrepSolid, seed_edge_id: u64) -> Result<SmoothChain<'_>, String> {
    let seed = solid
        .edges
        .iter()
        .find(|edge| edge.id == seed_edge_id)
        .ok_or_else(|| format!("blend: edge {seed_edge_id} not found"))?;
    if seed.start_vertex_id == seed.end_vertex_id {
        return Err("blend: chain seed must be an open edge".into());
    }
    let (first_face, first_loop, first_coedge) = locate_mate(solid, seed_edge_id, None)?;
    let (second_face, second_loop, second_coedge) =
        locate_mate(solid, seed_edge_id, Some((first_face.id, first_loop)))?;
    let seed_segment = ChainSegment {
        edge: seed,
        forward: true,
        first: BlendMate {
            face: first_face,
            coedge: first_coedge,
            loop_index: first_loop,
            rho: 0.0,
        },
        second: BlendMate {
            face: second_face,
            coedge: second_coedge,
            loop_index: second_loop,
            rho: 0.0,
        },
    };
    let chain_start_vertex = seed.start_vertex_id;

    // The chain-forward tangent and first-side outward normal a segment has
    // at one of its endpoints, used to require G1 continuity at the vertex.
    let endpoint_tangent = |segment: &ChainSegment, at_exit: bool| -> Result<Vec3, String> {
        let t = if segment.forward == at_exit {
            segment.edge.t1
        } else {
            segment.edge.t0
        };
        let tangent = segment.edge.curve.derivatives(t, 1)?[1].normalized()?;
        Ok(if segment.forward {
            tangent
        } else {
            tangent.scale(-1.0)
        })
    };
    let endpoint_normal = |segment: &ChainSegment, at_exit: bool| -> Result<Vec3, String> {
        let t = if segment.forward == at_exit {
            segment.edge.t1
        } else {
            segment.edge.t0
        };
        outward_normal_at(segment.first.face, segment.first.coedge, segment.edge, t)
    };

    // ---- forward walk (append) --------------------------------------
    let mut forward_segments = vec![seed_segment];
    let mut current_vertex = seed.end_vertex_id;
    let mut closed = false;
    let mut guard = 0;
    loop {
        if current_vertex == chain_start_vertex {
            closed = true;
            break;
        }
        guard += 1;
        if guard > 128 {
            return Err("blend: chain walk did not terminate after 128 segments".into());
        }
        let previous = forward_segments.last().unwrap();
        let arrive_tangent = endpoint_tangent(previous, true)?;
        let reference_first_normal = endpoint_normal(previous, true)?;
        let exclude = previous.edge.id;
        match find_conjugate(
            solid,
            current_vertex,
            arrive_tangent,
            reference_first_normal,
            exclude,
            true,
        )? {
            Some((candidate, forward, other_vertex, first, second)) => {
                forward_segments.push(ChainSegment {
                    edge: candidate,
                    forward,
                    first,
                    second,
                });
                current_vertex = other_vertex;
            }
            None => break,
        }
    }
    let forward_end_vertex = current_vertex;
    if closed {
        return Ok(SmoothChain {
            segments: forward_segments,
            closed: true,
            start_vertex: chain_start_vertex,
            end_vertex: chain_start_vertex,
        });
    }

    // ---- backward walk (prepend) ------------------------------------
    let mut prefix: Vec<ChainSegment> = Vec::new();
    let mut current_vertex = chain_start_vertex;
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 128 {
            return Err("blend: chain walk did not terminate after 128 segments".into());
        }
        let front = prefix.last().unwrap_or(&forward_segments[0]);
        let arrive_tangent = endpoint_tangent(front, false)?;
        let reference_first_normal = endpoint_normal(front, false)?;
        let exclude = front.edge.id;
        match find_conjugate(
            solid,
            current_vertex,
            arrive_tangent,
            reference_first_normal,
            exclude,
            false,
        )? {
            Some((candidate, forward, other_vertex, first, second)) => {
                prefix.push(ChainSegment {
                    edge: candidate,
                    forward,
                    first,
                    second,
                });
                current_vertex = other_vertex;
            }
            None => break,
        }
    }
    let start_vertex = current_vertex;

    // Assemble free-end -> free-end order: reversed prefix, then the seed
    // forward run.
    let mut segments = Vec::with_capacity(prefix.len() + forward_segments.len());
    segments.extend(prefix.into_iter().rev());
    segments.extend(forward_segments);
    Ok(SmoothChain {
        segments,
        closed: false,
        start_vertex,
        end_vertex: forward_end_vertex,
    })
}
