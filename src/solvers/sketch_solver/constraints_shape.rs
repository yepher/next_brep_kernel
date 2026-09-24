use super::*;

impl Engine {
    // -------------------------------------------------------------------
    // New point-based constraints (Tangent / Concentric / Equal-radius /
    // Collinear / Symmetric). All geometry math lives here in Rust; the app
    // only builds the `{type, points}` record and renders the result.
    // -------------------------------------------------------------------

    /// True when a `line` geometry exists whose points reference both `a` and
    /// `b`. Used to tell a line/circle tangent apart from a circle/circle one
    /// without extra constraint fields.
    pub(super) fn is_line_pair(&self, a: usize, b: usize) -> bool {
        let key_a = point_key(&self.points[a].id);
        let key_b = point_key(&self.points[b].id);
        for geometry in &self.geometries {
            let Some(obj) = geometry.as_object() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("line") {
                continue;
            }
            let Some(points) = obj.get("points").and_then(Value::as_array) else {
                continue;
            };
            let has_a = points.iter().any(|p| point_key(p) == key_a);
            let has_b = points.iter().any(|p| point_key(p) == key_b);
            if has_a && has_b {
                return true;
            }
        }
        false
    }

    /// If `(a, b)` is an end (anchor, handle) pair of a spline geometry — in
    /// either slot order — return the canonical `(anchor, handle)` indices.
    ///
    /// A spline (`"bezier"`/`"spline"` geometry) with `3n + 1` control points
    /// has exactly two such pairs: `(ids[0], ids[1])` at the start and
    /// `(ids[3n], ids[3n - 1])` at the end. The cubic Bézier hodograph gives
    /// the END TANGENT convention: `B'(0) = 3(P1 − P0)` and
    /// `B'(1) = 3(P_last − P_{last-1})`, so the tangent direction at a spline
    /// endpoint is exactly anchor→adjacent-handle. Tangency constraints on
    /// splines therefore constrain that segment's DIRECTION only (G1);
    /// curvature/G2 continuity is intentionally not modeled.
    pub(super) fn spline_end_info(&self, a: usize, b: usize) -> Option<(usize, usize)> {
        let key_a = point_key(&self.points[a].id);
        let key_b = point_key(&self.points[b].id);
        for geometry in &self.geometries {
            let Some(obj) = geometry.as_object() else {
                continue;
            };
            let geometry_type = obj.get("type").and_then(Value::as_str).unwrap_or("");
            if !is_spline_geometry_type(geometry_type) {
                continue;
            }
            let Some(points) = obj.get("points").and_then(Value::as_array) else {
                continue;
            };
            if points.len() < 4 {
                continue;
            }
            let seg_count = (points.len() - 1) / 3;
            let last_anchor = seg_count * 3;
            for (anchor_slot, handle_slot) in [(0usize, 1usize), (last_anchor, last_anchor - 1)] {
                let anchor_key = point_key(&points[anchor_slot]);
                let handle_key = point_key(&points[handle_slot]);
                if anchor_key == key_a && handle_key == key_b {
                    return Some((a, b));
                }
                if anchor_key == key_b && handle_key == key_a {
                    return Some((b, a));
                }
            }
        }
        None
    }

    /// Rotate a spline's end tangent segment by `angle_deg`. The handle is the
    /// natural orientation DOF: rotate it about the anchor whenever it is
    /// free, so the endpoint (typically pinned by a coincident/ground) stays
    /// put; otherwise swing the free anchor about the fixed handle. No-op when
    /// both are fixed.
    fn rotate_spline_end_segment(&mut self, anchor: usize, handle: usize, angle_deg: f64) {
        if !self.points[handle].fixed {
            let (cx, cy) = (self.points[anchor].x, self.points[anchor].y);
            self.rotate_point(cx, cy, handle, angle_deg);
        } else if !self.points[anchor].fixed {
            let (cx, cy) = (self.points[handle].x, self.points[handle].y);
            self.rotate_point(cx, cy, anchor, angle_deg);
        }
    }

    /// Move `boundary` along the current center→boundary direction so that
    /// distance(center, boundary) == target (target clamped to >= 0).
    fn snap_radius(&mut self, center: usize, boundary: usize, target: f64) {
        let target = target.max(0.0);
        let dx = self.points[boundary].x - self.points[center].x;
        let dy = self.points[boundary].y - self.points[center].y;
        let r = (dx * dx + dy * dy).sqrt();
        if r <= 1e-12 {
            self.points[boundary].x = self.points[center].x + target;
            self.points[boundary].y = self.points[center].y;
            return;
        }
        let scale = target / r;
        self.points[boundary].x = self.points[center].x + dx * scale;
        self.points[boundary].y = self.points[center].y + dy * scale;
    }

    fn translate_point(&mut self, index: usize, dx: f64, dy: f64) {
        self.points[index].x += dx;
        self.points[index].y += dy;
    }

    /// Nudge a circle/arc so distance(center, boundary) moves toward `target`,
    /// distributing the correction over whichever of {center, boundary} is free.
    /// When the boundary is pinned (fixed, or held by another constraint such as
    /// a coincidence), the center absorbs the change over iterations — so equal
    /// radius can still resize a circle whose rim point is locked to geometry.
    /// `damp` in (0,1] controls the step size. Returns true if it could move.
    fn adjust_circle_radius(
        &mut self,
        center: usize,
        boundary: usize,
        target: f64,
        damp: f64,
    ) -> bool {
        let target = target.max(0.0);
        let dx = self.points[boundary].x - self.points[center].x;
        let dy = self.points[boundary].y - self.points[center].y;
        let r = (dx * dx + dy * dy).sqrt();
        let free_b = !self.points[boundary].fixed;
        let free_c = !self.points[center].fixed;
        let n = (free_b as i32 + free_c as i32) as f64;
        if n == 0.0 {
            return false;
        }
        if r <= 1e-12 {
            // Degenerate: push the boundary (or center) out along +x.
            if free_b {
                self.points[boundary].x = self.points[center].x + target;
            } else {
                self.points[center].x = self.points[boundary].x - target;
            }
            return true;
        }
        let ux = dx / r;
        let uy = dy / r;
        let step = (target - r) * damp / n;
        if free_b {
            self.points[boundary].x += ux * step;
            self.points[boundary].y += uy * step;
        }
        if free_c {
            self.points[center].x -= ux * step;
            self.points[center].y -= uy * step;
        }
        true
    }

    /// Concentric: pull the two circle/arc centers together. Unlike Coincident,
    /// this does NOT lock the points as fixed (a concentric circle center should
    /// stay free to move as a pair).
    pub(super) fn c_concentric(&mut self, constraint: &mut HotConstraint, a: usize, b: usize) -> CResult {
        let tolerance = self.tolerance();
        let dx = self.points[a].x - self.points[b].x;
        let dy = self.points[a].y - self.points[b].y;
        if dx.hypot(dy) <= tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        let fixed_a = self.points[a].fixed;
        let fixed_b = self.points[b].fixed;
        if !fixed_a && !fixed_b {
            let cx = (self.points[a].x + self.points[b].x) / 2.0;
            let cy = (self.points[a].y + self.points[b].y) / 2.0;
            self.points[a].x = cx;
            self.points[a].y = cy;
            self.points[b].x = cx;
            self.points[b].y = cy;
            constraint.set_error(Value::Null);
        } else if !fixed_a {
            self.points[a].x = self.points[b].x;
            self.points[a].y = self.points[b].y;
            constraint.set_error(Value::Null);
        } else if !fixed_b {
            self.points[b].x = self.points[a].x;
            self.points[b].y = self.points[a].y;
            constraint.set_error(Value::Null);
        } else {
            constraint.set_error(Value::String(format!(
                "Concentric constraint not satisfied: centers {} and {} are both fixed",
                fmt_id(&self.points[a].id),
                fmt_id(&self.points[b].id)
            )));
        }
        Ok(Value::Null)
    }

    /// Pick which circle "leads" an equal-radius pair when neither is
    /// dimensioned or pinned, so interactive dragging works bidirectionally.
    ///
    /// The app moves a dragged boundary point and then re-solves without
    /// pinning it, so at solve time one radius has just jumped. We remember the
    /// radii from the previous solve (`_eqRadLast{1,2}`) and, on the first pass
    /// of each new solve (`_eqRadCycle`), treat whichever radius changed more as
    /// the reference the user is dragging; the other follows it. Returns 1 (use
    /// circle 1), 2 (use circle 2), or 0 (average — nothing is being dragged).
    fn equal_radius_leader(
        &self,
        constraint: &mut HotConstraint,
        r1: f64,
        r2: f64,
        tolerance: f64,
    ) -> i64 {
        let cycle = self.pass_token.split(':').next().unwrap_or("").to_string();
        let stored_cycle = constraint
            .raw
            .get("_eqRadCycle")
            .and_then(Value::as_str)
            .map(str::to_string);
        if stored_cycle.as_deref() == Some(cycle.as_str()) {
            return constraint
                .raw
                .get("_eqRadRef")
                .and_then(Value::as_f64)
                .map(|v| v as i64)
                .unwrap_or(0);
        }
        // First pass of a new solve: detect the just-dragged radius.
        let last1 = constraint
            .raw
            .get("_eqRadLast1")
            .and_then(Value::as_f64)
            .unwrap_or(r1);
        let last2 = constraint
            .raw
            .get("_eqRadLast2")
            .and_then(Value::as_f64)
            .unwrap_or(r2);
        let d1 = (r1 - last1).abs();
        let d2 = (r2 - last2).abs();
        let bias = tolerance.max(1e-6);
        let leader = if d1 > d2 + bias {
            1
        } else if d2 > d1 + bias {
            2
        } else {
            0
        };
        constraint
            .raw
            .insert("_eqRadRef".into(), json_num(leader as f64));
        constraint
            .raw
            .insert("_eqRadCycle".into(), Value::String(cycle));
        leader
    }

    /// Equal radius: dist(c1,b1) == dist(c2,b2). Reference selection order:
    /// explicit radius (Distance) dimension > pinned/fixed boundary >
    /// drag-leader (the boundary the user just moved) > average of both.
    pub(super) fn c_equal_radius(
        &mut self,
        constraint: &mut HotConstraint,
        c1: usize,
        b1: usize,
        c2: usize,
        b2: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        let r1_dim = self.find_distance_constraint_on_pair(c1, b1);
        let r2_dim = self.find_distance_constraint_on_pair(c2, b2);
        let r1 = self.distance(c1, b1);
        let r2 = self.distance(c2, b2);
        // A circle's radius is immovable only when BOTH its center and boundary
        // are fixed (otherwise the free one can absorb the change).
        let c1_locked = self.points[c1].fixed && self.points[b1].fixed;
        let c2_locked = self.points[c2].fixed && self.points[b2].fixed;

        // (target radius, move circle1?, move circle2?)
        let (target, move1, move2) = match (r1_dim, r2_dim) {
            (Some(_), Some(_)) => {
                let message = "Both circles have a radius dimension applied to them".to_string();
                constraint.set_error(Value::String(message.clone()));
                return Ok(Value::String(message));
            }
            (Some(value), None) => (value.abs(), false, true),
            (None, Some(value)) => (value.abs(), true, false),
            (None, None) => {
                if c1_locked && c2_locked {
                    if (r1 - r2).abs() <= tolerance {
                        constraint.set_error(Value::Null);
                    } else {
                        constraint.set_error(Value::String(format!(
                            "Equal radius: circles {} and {} are both fully fixed",
                            fmt_id(&self.points[c1].id),
                            fmt_id(&self.points[c2].id)
                        )));
                    }
                    return Ok(Value::Null);
                } else if c1_locked {
                    (r1, false, true)
                } else if c2_locked {
                    (r2, true, false)
                } else {
                    match self.equal_radius_leader(constraint, r1, r2, tolerance) {
                        1 => (r1, false, true),
                        2 => (r2, true, false),
                        _ => ((r1 + r2) / 2.0, true, true),
                    }
                }
            }
        };

        let mut dev = 0.0f64;
        if move1 {
            dev = dev.max((r1 - target).abs());
        }
        if move2 {
            dev = dev.max((r2 - target).abs());
        }
        if dev <= tolerance {
            constraint.set_error(Value::Null);
        } else {
            constraint.set_error(Value::String(format!(
                "Equal radius constraint not satisfied\n        {} != {}",
                fmt_number(r1),
                fmt_number(r2)
            )));
            let mut moved_any = false;
            if move1 {
                moved_any |= self.adjust_circle_radius(c1, b1, target, 0.5);
            }
            if move2 {
                moved_any |= self.adjust_circle_radius(c2, b2, target, 0.5);
            }
            if !moved_any {
                constraint.set_error(Value::String(format!(
                    "Equal radius: circles {} and {} cannot move",
                    fmt_id(&self.points[c1].id),
                    fmt_id(&self.points[c2].id)
                )));
            }
        }

        // Remember the resulting radii so the next solve can tell which one the
        // user drags.
        let out1 = self.distance(c1, b1);
        let out2 = self.distance(c2, b2);
        constraint.raw.insert("_eqRadLast1".into(), json_num(out1));
        constraint.raw.insert("_eqRadLast2".into(), json_num(out2));
        Ok(Value::Null)
    }

    /// Collinear: every point lies on the line through the first two. Reuses the
    /// tested Point-on-Line relaxation for each extra point.
    pub(super) fn c_collinear(
        &mut self,
        constraint: &mut HotConstraint,
        indices: &[Option<usize>],
    ) -> CResult {
        let resolved: Vec<usize> = indices.iter().filter_map(|slot| *slot).collect();
        if resolved.len() < 3 {
            constraint.set_error(Value::String(
                "Collinear constraint requires at least 3 points".into(),
            ));
            return Ok(Value::Null);
        }
        let a = resolved[0];
        let b = resolved[1];
        for &pk in &resolved[2..] {
            self.c_point_on_line(constraint, a, b, pk)?;
        }
        // Recompute overall satisfaction (overrides the per-point error set by
        // the last Point-on-Line correction).
        let tolerance = self.tolerance();
        let dx = self.points[b].x - self.points[a].x;
        let dy = self.points[b].y - self.points[a].y;
        let len = dx.hypot(dy);
        let mut max_dev = 0.0f64;
        if len > tolerance {
            for &pk in &resolved[2..] {
                let cx = self.points[pk].x - self.points[a].x;
                let cy = self.points[pk].y - self.points[a].y;
                max_dev = max_dev.max(((cx * dy - cy * dx) / len).abs());
            }
        }
        if len > tolerance && max_dev <= tolerance {
            constraint.set_error(Value::Null);
        } else {
            constraint.set_error(Value::String(format!(
                "Collinear constraint not satisfied. Max dist: {:.4}",
                max_dev
            )));
        }
        Ok(Value::Null)
    }

    /// Symmetric: points p and q are mirror images across the axis line a→b.
    pub(super) fn c_symmetric(
        &mut self,
        constraint: &mut HotConstraint,
        indices: &[Option<usize>],
    ) -> CResult {
        let a = self.req(indices, 0)?;
        let b = self.req(indices, 1)?;
        let p = self.req(indices, 2)?;
        let q = self.req(indices, 3)?;
        let tolerance = self.tolerance();
        let dx = self.points[b].x - self.points[a].x;
        let dy = self.points[b].y - self.points[a].y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < tolerance * tolerance {
            constraint.set_error(Value::String(
                "Symmetric constraint axis is degenerate".into(),
            ));
            return Ok(Value::Null);
        }
        // Reflection of (px,py) across the infinite line a→b.
        let ax = self.points[a].x;
        let ay = self.points[a].y;
        let reflect = |px: f64, py: f64| -> (f64, f64) {
            let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
            let proj_x = ax + t * dx;
            let proj_y = ay + t * dy;
            (2.0 * proj_x - px, 2.0 * proj_y - py)
        };
        let (target_q_x, target_q_y) = reflect(self.points[p].x, self.points[p].y);
        let (target_p_x, target_p_y) = reflect(self.points[q].x, self.points[q].y);
        let err_p = (self.points[p].x - target_p_x).hypot(self.points[p].y - target_p_y);
        let err_q = (self.points[q].x - target_q_x).hypot(self.points[q].y - target_q_y);
        if err_p.max(err_q) <= tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Symmetric constraint not satisfied. Error: {:.4}",
            err_p.max(err_q)
        )));
        if !self.points[p].fixed {
            self.points[p].x += (target_p_x - self.points[p].x) * 0.5;
            self.points[p].y += (target_p_y - self.points[p].y) * 0.5;
        }
        if !self.points[q].fixed {
            self.points[q].x += (target_q_x - self.points[q].x) * 0.5;
            self.points[q].y += (target_q_y - self.points[q].y) * 0.5;
        }
        Ok(Value::Null)
    }

    /// Tangent dispatch. Spline end pairs (anchor, handle) are recognized
    /// first, in either pair slot: spline/spline joins the two end tangents,
    /// spline/line aligns the end tangent with the line, spline/circle makes
    /// the end tangent perpendicular to the radial direction at the endpoint.
    /// Otherwise the legacy dispatch applies: line/circle when the first pair
    /// is a line geometry, circle/circle in all remaining cases. (A pair that
    /// is simultaneously a spline end and a line is read as the spline.)
    pub(super) fn c_tangent(&mut self, constraint: &mut HotConstraint, indices: &[Option<usize>]) -> CResult {
        let p0 = self.req(indices, 0)?;
        let p1 = self.req(indices, 1)?;
        let p2 = self.req(indices, 2)?;
        let p3 = self.req(indices, 3)?;
        match (self.spline_end_info(p0, p1), self.spline_end_info(p2, p3)) {
            (Some((a1, h1)), Some((a2, h2))) => {
                self.c_tangent_spline_spline(constraint, a1, h1, a2, h2)
            }
            (Some((anchor, handle)), None) => {
                if self.is_line_pair(p2, p3) {
                    self.c_tangent_spline_line(constraint, anchor, handle, p2, p3)
                } else {
                    self.c_tangent_spline_circle(constraint, anchor, handle, p2, p3)
                }
            }
            (None, Some((anchor, handle))) => {
                if self.is_line_pair(p0, p1) {
                    self.c_tangent_spline_line(constraint, anchor, handle, p0, p1)
                } else {
                    self.c_tangent_spline_circle(constraint, anchor, handle, p0, p1)
                }
            }
            (None, None) => {
                if self.is_line_pair(p0, p1) {
                    self.c_tangent_line_circle(constraint, p0, p1, p2, p3)
                } else {
                    self.c_tangent_circle_circle(constraint, p0, p1, p2, p3)
                }
            }
        }
    }

    /// A tangent side (two-point segment) may rotate only when it is not
    /// fully fixed and not pinned to an axis by a horizontal/vertical
    /// constraint (mirrors the ∠ constraint's movement guards).
    fn tangent_side_movable(&self, a: usize, b: usize) -> bool {
        let pinned = self.points[a].fixed && self.points[b].fixed;
        let axis_locked = self.participate_in_constraint(CType::Horizontal, &[a, b])
            || self.participate_in_constraint(CType::Vertical, &[a, b]);
        !(pinned || axis_locked)
    }

    /// Shared angular relaxation step for the spline tangent variants: rotate
    /// toward `target_angle` (degrees, relative to `current_angle`) with the
    /// same 1.5°-per-pass clamp the ∠ constraint uses. Returns the clamped
    /// rotation to apply to side 1 (side 2 gets the negative), split in half
    /// when both sides can move.
    fn tangent_rotation_split(
        delta_raw: f64,
        side1_moving: bool,
        side2_moving: bool,
    ) -> (f64, f64) {
        let max_step = 1.5;
        let mut delta = delta_raw;
        if delta.abs() > max_step {
            delta = js_sign(delta) * max_step;
        }
        if side1_moving && side2_moving {
            (delta / 2.0, -delta / 2.0)
        } else if side1_moving {
            (delta, 0.0)
        } else if side2_moving {
            (0.0, -delta)
        } else {
            (0.0, 0.0)
        }
    }

    /// Spline-end / line tangent: the spline's end tangent direction
    /// (anchor→handle) must be parallel to line `a→b`. Direction only (G1) —
    /// there is no curvature to match against a straight line anyway.
    fn c_tangent_spline_line(
        &mut self,
        constraint: &mut HotConstraint,
        anchor: usize,
        handle: usize,
        a: usize,
        b: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        if self.distance(anchor, handle) < tolerance {
            constraint.set_error(Value::String(
                "Tangent: spline end tangent is degenerate".into(),
            ));
            return Ok(Value::Null);
        }
        if self.distance(a, b) < tolerance {
            constraint.set_error(Value::String("Tangent: line is degenerate".into()));
            return Ok(Value::Null);
        }
        let seg_angle = self.calculate_angle(anchor, handle);
        let line_angle = self.calculate_angle(a, b);
        let current = normalize_angle(seg_angle - line_angle);
        // Parallel modulo 180: pick the closer of 0°/180° (∥ constraint rule).
        let target = if current > 90.0 && current <= 270.0 {
            180.0
        } else if current > 270.0 {
            360.0
        } else {
            0.0
        };
        let delta_raw = shortest_angle_delta(target, current);
        if delta_raw.abs() < tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Tangent constraint not satisfied\n        spline end tangent {} not parallel to line {}",
            fmt_number(seg_angle),
            fmt_number(line_angle)
        )));

        let seg_moving = self.tangent_side_movable(anchor, handle);
        let line_moving = self.tangent_side_movable(a, b);
        if !seg_moving && !line_moving {
            constraint.set_error(Value::String(
                "Tangent: spline end and line are both fully fixed".into(),
            ));
            return Ok(Value::Null);
        }
        let (rot_seg, rot_line) = Self::tangent_rotation_split(delta_raw, seg_moving, line_moving);
        if rot_seg != 0.0 && !rot_seg.is_nan() {
            self.rotate_spline_end_segment(anchor, handle, rot_seg);
        }
        if rot_line != 0.0 && !rot_line.is_nan() {
            // Same pivot policy as the ∠ constraint applies to a line.
            if self.points[a].fixed {
                let (cx, cy) = (self.points[a].x, self.points[a].y);
                self.rotate_point(cx, cy, b, rot_line);
            } else if self.points[b].fixed {
                let (cx, cy) = (self.points[b].x, self.points[b].y);
                self.rotate_point(cx, cy, a, rot_line);
            } else {
                let mid_x = (self.points[a].x + self.points[b].x) / 2.0;
                let mid_y = (self.points[a].y + self.points[b].y) / 2.0;
                self.rotate_point(mid_x, mid_y, a, rot_line);
                self.rotate_point(mid_x, mid_y, b, rot_line);
            }
        }
        Ok(Value::Null)
    }

    /// Spline-end / circle (or arc) tangent: the spline's end tangent
    /// (anchor→handle) must be perpendicular to the radial direction
    /// center→anchor, i.e. the curve leaves its endpoint along the circle's
    /// tangent there. Pair it with a coincident (spline endpoint ≡ rim/arc
    /// endpoint) so the anchor actually lies on the circle. G1 only — the
    /// spline's curvature at the endpoint is NOT matched to 1/r (no G2).
    fn c_tangent_spline_circle(
        &mut self,
        constraint: &mut HotConstraint,
        anchor: usize,
        handle: usize,
        center: usize,
        boundary: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        if self.distance(anchor, handle) < tolerance {
            constraint.set_error(Value::String(
                "Tangent: spline end tangent is degenerate".into(),
            ));
            return Ok(Value::Null);
        }
        if self.distance(center, anchor) < tolerance {
            constraint.set_error(Value::String(
                "Tangent: spline endpoint coincides with the circle center".into(),
            ));
            return Ok(Value::Null);
        }
        let seg_angle = self.calculate_angle(anchor, handle);
        let radial_angle = self.calculate_angle(center, anchor);
        let current = normalize_angle(seg_angle - radial_angle);
        // Perpendicular modulo 180: pick the closer of 90°/270° (⟂ rule).
        let target = if current <= 180.0 { 90.0 } else { 270.0 };
        let delta_raw = shortest_angle_delta(target, current);
        if delta_raw.abs() < tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Tangent constraint not satisfied\n        spline end tangent {} not perpendicular to radius {}",
            fmt_number(seg_angle),
            fmt_number(radial_angle)
        )));

        let seg_moving = self.tangent_side_movable(anchor, handle);
        // The circle side moves rigidly (center + boundary rotate together
        // about the tangency point) so its radius is preserved.
        let circle_moving = !self.points[center].fixed && !self.points[boundary].fixed;
        if !seg_moving && !circle_moving {
            constraint.set_error(Value::String(
                "Tangent: spline end and circle are both fully fixed".into(),
            ));
            return Ok(Value::Null);
        }
        let (rot_seg, rot_circle) =
            Self::tangent_rotation_split(delta_raw, seg_moving, circle_moving);
        if rot_seg != 0.0 && !rot_seg.is_nan() {
            self.rotate_spline_end_segment(anchor, handle, rot_seg);
        }
        if rot_circle != 0.0 && !rot_circle.is_nan() {
            let (cx, cy) = (self.points[anchor].x, self.points[anchor].y);
            self.rotate_point(cx, cy, center, rot_circle);
            self.rotate_point(cx, cy, boundary, rot_circle);
        }
        Ok(Value::Null)
    }

    /// Spline-end / spline-end tangent (G1 join): the two end tangent
    /// directions must be parallel. Typically combined with a coincident on
    /// the two anchors so the curves share the endpoint.
    fn c_tangent_spline_spline(
        &mut self,
        constraint: &mut HotConstraint,
        a1: usize,
        h1: usize,
        a2: usize,
        h2: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        if self.distance(a1, h1) < tolerance || self.distance(a2, h2) < tolerance {
            constraint.set_error(Value::String(
                "Tangent: spline end tangent is degenerate".into(),
            ));
            return Ok(Value::Null);
        }
        let angle1 = self.calculate_angle(a1, h1);
        let angle2 = self.calculate_angle(a2, h2);
        let current = normalize_angle(angle1 - angle2);
        let target = if current > 90.0 && current <= 270.0 {
            180.0
        } else if current > 270.0 {
            360.0
        } else {
            0.0
        };
        let delta_raw = shortest_angle_delta(target, current);
        if delta_raw.abs() < tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Tangent constraint not satisfied\n        spline end tangents {} and {} not parallel",
            fmt_number(angle1),
            fmt_number(angle2)
        )));

        let m1 = self.tangent_side_movable(a1, h1);
        let m2 = self.tangent_side_movable(a2, h2);
        if !m1 && !m2 {
            constraint.set_error(Value::String(
                "Tangent: both spline ends are fully fixed".into(),
            ));
            return Ok(Value::Null);
        }
        let (rot1, rot2) = Self::tangent_rotation_split(delta_raw, m1, m2);
        if rot1 != 0.0 && !rot1.is_nan() {
            self.rotate_spline_end_segment(a1, h1, rot1);
        }
        if rot2 != 0.0 && !rot2.is_nan() {
            self.rotate_spline_end_segment(a2, h2, rot2);
        }
        Ok(Value::Null)
    }

    /// Line/circle tangent: perpendicular distance(center, line a→b) == radius,
    /// where radius = dist(center, boundary).
    fn c_tangent_line_circle(
        &mut self,
        constraint: &mut HotConstraint,
        a: usize,
        b: usize,
        center: usize,
        boundary: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        let dx = self.points[b].x - self.points[a].x;
        let dy = self.points[b].y - self.points[a].y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < tolerance {
            constraint.set_error(Value::String("Tangent: line is degenerate".into()));
            return Ok(Value::Null);
        }
        let nx = -dy / len;
        let ny = dx / len;
        let signed = (self.points[center].x - self.points[a].x) * nx
            + (self.points[center].y - self.points[a].y) * ny;
        let side = if signed >= 0.0 { 1.0 } else { -1.0 };
        let d = signed.abs();
        let radius = self.distance(center, boundary);
        let err = d - radius;
        if err.abs() <= tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Tangent constraint not satisfied\n        distance {} != radius {}",
            fmt_number(d),
            fmt_number(radius)
        )));

        let center_free = !self.points[center].fixed;
        let boundary_free = !self.points[boundary].fixed;
        let handles = (center_free as i32) + (boundary_free as i32);
        if handles == 0 {
            constraint.set_error(Value::String(
                "Tangent: circle center and boundary are both fixed".into(),
            ));
            return Ok(Value::Null);
        }
        let per = err / handles as f64 * 0.5;
        // Increase radius toward the line (uses the original center position).
        if boundary_free {
            self.snap_radius(center, boundary, radius + per);
        }
        // Move the center toward the line to reduce the perpendicular distance.
        if center_free {
            self.points[center].x -= nx * side * per;
            self.points[center].y -= ny * side * per;
        }
        Ok(Value::Null)
    }

    /// Circle/circle tangent: dist(centers) == r1 + r2 (external) or |r1 - r2|
    /// (internal), chosen by the initial configuration. Circles translate
    /// rigidly (center + boundary together) so radii are preserved.
    fn c_tangent_circle_circle(
        &mut self,
        constraint: &mut HotConstraint,
        c1: usize,
        b1: usize,
        c2: usize,
        b2: usize,
    ) -> CResult {
        let tolerance = self.tolerance();
        let r1 = self.distance(c1, b1);
        let r2 = self.distance(c2, b2);
        let cdx = self.points[c2].x - self.points[c1].x;
        let cdy = self.points[c2].y - self.points[c1].y;
        let d = (cdx * cdx + cdy * cdy).sqrt();
        let external = r1 + r2;
        let internal = (r1 - r2).abs();
        let target = if (d - external).abs() <= (d - internal).abs() {
            external
        } else {
            internal
        };
        let err = d - target;
        if err.abs() <= tolerance {
            constraint.set_error(Value::Null);
            return Ok(Value::Null);
        }
        constraint.set_error(Value::String(format!(
            "Tangent constraint not satisfied\n        center distance {} != {}",
            fmt_number(d),
            fmt_number(target)
        )));

        let (ux, uy) = if d > 1e-12 {
            (cdx / d, cdy / d)
        } else {
            (1.0, 0.0)
        };
        // A circle can be translated only when both its center and boundary are
        // free, so its radius is preserved.
        let c1_movable = !self.points[c1].fixed && !self.points[b1].fixed;
        let c2_movable = !self.points[c2].fixed && !self.points[b2].fixed;
        let handles = (c1_movable as i32) + (c2_movable as i32);
        if handles == 0 {
            constraint.set_error(Value::String(
                "Tangent: both circles are fully constrained".into(),
            ));
            return Ok(Value::Null);
        }
        let per = err / handles as f64 * 0.5;
        if c1_movable {
            self.translate_point(c1, ux * per, uy * per);
            self.translate_point(b1, ux * per, uy * per);
        }
        if c2_movable {
            self.translate_point(c2, -ux * per, -uy * per);
            self.translate_point(b2, -ux * per, -uy * per);
        }
        Ok(Value::Null)
    }
}
