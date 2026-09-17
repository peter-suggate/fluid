//! Component-wise adaptive MAC interpolation for the 2-D level-set solver.
//!
//! Ando--Batty 2020, section 5: affine MLS with tensor linear weights, using
//! each face's own spacing. On regular stencils this is bilinear interpolation.
//! No cell-centred velocity participates in either interpolation or extension.
use crate::{Fields, Graph, RowKind, ValidationError};
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Sample {
    point: [f64; 2],
    width: f64,
    value: f64,
}

#[derive(Clone, Debug)]
struct Component {
    samples: Vec<Sample>,
    bins: HashMap<(i32, i32), Vec<usize>>,
}

impl Component {
    fn new(samples: Vec<Sample>) -> Self {
        let mut bins: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (id, s) in samples.iter().enumerate() {
            // Include a one-ring apron for rank-deficient transition stencils.
            let radius = 1.5 * s.width;
            for y in (s.point[1] - radius).floor() as i32..=(s.point[1] + radius).floor() as i32 {
                for x in (s.point[0] - radius).floor() as i32..=(s.point[0] + radius).floor() as i32
                {
                    bins.entry((x, y)).or_default().push(id);
                }
            }
        }
        Self { samples, bins }
    }

    fn sample(&self, p: [f64; 2]) -> f64 {
        let Some(ids) = self.bins.get(&(p[0].floor() as i32, p[1].floor() as i32)) else {
            return 0.0;
        };
        let candidates: Vec<_> = ids
            .iter()
            .copied()
            .filter(|&id| {
                let s = &self.samples[id];
                (s.point[0] - p[0]).abs() <= 1.5 * s.width
                    && (s.point[1] - p[1]).abs() <= 1.5 * s.width
            })
            .collect();
        let ids = &candidates;
        let mut points = Vec::with_capacity(16);
        for &id in ids {
            let s = &self.samples[id];
            let d = [s.point[0] - p[0], s.point[1] - p[1]];
            let w = (1.0 - d[0].abs() / s.width).max(0.0) * (1.0 - d[1].abs() / s.width).max(0.0);
            if w > 1e-12 {
                points.push((d, s.value, w));
            }
        }
        // At a sparse edge or T junction, complete deficient support from the
        // nearest sample's ring, with the paper's epsilon tensor kernel. Do
        // not blend zeros for unallocated cells into a supported velocity.
        if needs_ring(&points) {
            let distance = |s: &Sample| (s.point[0] - p[0]).powi(2) + (s.point[1] - p[1]).powi(2);
            let nearest = ids
                .iter()
                .map(|&i| distance(&self.samples[i]))
                .fold(f64::INFINITY, f64::min);
            let anchors: Vec<_> = ids
                .iter()
                .copied()
                .filter(|&i| distance(&self.samples[i]) <= nearest + 1e-10)
                .collect();
            points.clear();
            for &id in ids {
                let s = &self.samples[id];
                if !anchors.iter().any(|&i| {
                    let a = &self.samples[i];
                    let h = s.width.max(a.width) * 1.5;
                    (s.point[0] - a.point[0]).abs() <= h && (s.point[1] - a.point[1]).abs() <= h
                }) {
                    continue;
                }
                let d = [s.point[0] - p[0], s.point[1] - p[1]];
                let w =
                    (1.0 - d[0].abs() / s.width).max(0.01) * (1.0 - d[1].abs() / s.width).max(0.01);
                points.push((d, s.value, w));
            }
        }
        if points.is_empty() {
            0.0
        } else {
            affine_value(&points)
        }
    }
}

fn needs_ring(points: &[([f64; 2], f64, f64)]) -> bool {
    if points.is_empty() {
        return true;
    }
    let anchor = points[0].0;
    let direction = points
        .iter()
        .map(|p| [p.0[0] - anchor[0], p.0[1] - anchor[1]])
        .max_by(|a, b| (a[0] * a[0] + a[1] * a[1]).total_cmp(&(b[0] * b[0] + b[1] * b[1])))
        .unwrap();
    let length = direction[0] * direction[0] + direction[1] * direction[1];
    if length < 1e-20 {
        return anchor[0] * anchor[0] + anchor[1] * anchor[1] > 1e-16;
    }
    if points.iter().any(|p| {
        ((p.0[0] - anchor[0]) * direction[1] - (p.0[1] - anchor[1]) * direction[0]).abs()
            > 1e-8 * length.sqrt()
    }) {
        return false;
    }
    (anchor[0] * direction[1] - anchor[1] * direction[0]).abs() > 1e-8 * length.sqrt()
}

fn affine_value(points: &[([f64; 2], f64, f64)]) -> f64 {
    let mut w = 0.0;
    let mut mean = [0.0; 3];
    for &(d, v, weight) in points {
        w += weight;
        mean[0] += weight * d[0];
        mean[1] += weight * d[1];
        mean[2] += weight * v;
    }
    for x in &mut mean {
        *x /= w;
    }
    let (mut xx, mut xy, mut yy, mut xv, mut yv) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(d, v, weight) in points {
        let x = d[0] - mean[0];
        let y = d[1] - mean[1];
        let q = v - mean[2];
        xx += weight * x * x;
        xy += weight * x * y;
        yy += weight * y * y;
        xv += weight * x * q;
        yv += weight * y * q;
    }
    let det = xx * yy - xy * xy;
    let (gx, gy) = if det > 1e-12 * (xx + yy).powi(2) {
        ((yy * xv - xy * yv) / det, (xx * yv - xy * xv) / det)
    } else {
        // Moore--Penrose inverse on a line: exact grid-line interpolation is
        // deliberately rank one. A diagonal regularizer would damp affine data.
        let norm = xx * xx + 2.0 * xy * xy + yy * yy;
        if norm > 1e-24 {
            ((xx * xv + xy * yv) / norm, (xy * xv + yy * yv) / norm)
        } else {
            (0.0, 0.0)
        }
    };
    mean[2] - gx * mean[0] - gy * mean[1]
}

#[derive(Clone, Debug)]
pub struct StaggeredVelocity2d {
    components: [Component; 2],
    dimensions: [f32; 2],
}

impl StaggeredVelocity2d {
    pub fn new(graph: &Graph, fields: &Fields) -> Result<Self, ValidationError> {
        if graph.dimension != 2
            || fields.face_velocity.len() != graph.rows.len()
            || fields.face_velocity.iter().any(|v| !v.is_finite())
        {
            return Err(ValidationError(
                "staggered 2-D sampling needs finite face velocities".into(),
            ));
        }
        let dimensions = [graph.dimensions[0], graph.dimensions[1]];
        let mut walls: [Vec<&crate::Row>; 4] = Default::default();
        for row in &graph.rows {
            if row.kind != RowKind::ClosedWorld || row.axis >= 2 {
                continue;
            }
            let a = row.axis as usize;
            if row.center[a] == 0.0 {
                walls[2 * a].push(row);
            }
            if row.center[a] == dimensions[a] {
                walls[2 * a + 1].push(row);
            }
        }
        let mut samples: [Vec<Sample>; 2] = Default::default();
        for row in &graph.rows {
            let axis = row.axis as usize;
            if axis >= 2 {
                continue;
            }
            let value = if row.kind == RowKind::ClosedWorld {
                fields.face_velocity[row.id as usize]
            } else if row.open_fraction > 1e-8 {
                (fields.face_velocity[row.id as usize]
                    - (1.0 - row.open_fraction) * row.solid_velocity)
                    / row.open_fraction
            } else {
                row.solid_velocity
            };
            let s = Sample {
                point: [row.center[0] as f64, row.center[1] as f64],
                width: row.static_measure.unwrap_or(row.measure).max(1.0) as f64,
                value: value as f64,
            };
            samples[axis].push(s.clone());
            // Mirror samples at the exterior domain. Tangential velocity is
            // even (free slip). Normal velocity reflects about the actual
            // boundary value, including a separating wall's accepted velocity.
            let mut copies = vec![s];
            for dim in 0..2 {
                let prior = copies.clone();
                for q in prior {
                    for (side, boundary) in [0.0, dimensions[dim] as f64].into_iter().enumerate() {
                        let distance = (q.point[dim] - boundary).abs();
                        if distance <= 1e-10 || distance > q.width * 1.5 {
                            continue;
                        }
                        let mut ghost = q.clone();
                        ghost.point[dim] = 2.0 * boundary - q.point[dim];
                        if axis == dim {
                            let wall = walls[2 * dim + side].iter().find(|r| {
                                (r.center[1 - dim] - row.center[1 - dim]).abs()
                                    <= r.measure * 0.5 + 1e-8
                            });
                            if let Some(wall) = wall {
                                ghost.value =
                                    2.0 * fields.face_velocity[wall.id as usize] as f64 - q.value;
                            }
                        }
                        samples[axis].push(ghost.clone());
                        copies.push(ghost);
                    }
                }
            }
        }
        // Canonical geometric order makes row storage order irrelevant to MLS.
        for values in &mut samples {
            values.sort_by(|a, b| {
                a.point[0]
                    .total_cmp(&b.point[0])
                    .then(a.point[1].total_cmp(&b.point[1]))
                    .then(a.width.total_cmp(&b.width))
            });
        }
        Ok(Self {
            components: samples.map(Component::new),
            dimensions,
        })
    }
    pub fn sample(&self, point: [f32; 2]) -> [f32; 2] {
        let p = std::array::from_fn(|a| point[a].clamp(0.0, self.dimensions[a]) as f64);
        self.components.each_ref().map(|c| c.sample(p) as f32)
    }
    pub fn trace(&self, start: [f32; 2], dt: f32) -> [f32; 2] {
        let first = self.sample(start);
        let mid = std::array::from_fn(|a| {
            (start[a] - 0.5 * dt * first[a]).clamp(0.0, self.dimensions[a])
        });
        let velocity = self.sample(mid);
        std::array::from_fn(|a| (start[a] - dt * velocity[a]).clamp(0.0, self.dimensions[a]))
    }
}

/// Synchronous same-component face extension. Accepted liquid faces and solid
/// boundary values are immutable. Dry values are rebuilt rather than re-seeded
/// from last frame's extension or the conservative volume field.
pub fn extend_faces(
    graph: &Graph,
    fields: &mut Fields,
    phi: &[f32],
    depth_count: u8,
) -> Result<(), ValidationError> {
    if graph.dimension != 2
        || phi.len() != graph.cells.len()
        || fields.capacity.len() != graph.cells.len()
        || fields.face_velocity.len() != graph.rows.len()
        || phi.iter().any(|v| !v.is_finite())
        || fields.face_velocity.iter().any(|v| !v.is_finite())
    {
        return Err(ValidationError(
            "face extension requires finite 2-D face velocities and cell phi".into(),
        ));
    }
    let mut known = vec![false; graph.rows.len()];
    let mut fixed = vec![false; graph.rows.len()];
    let mut values = fields.face_velocity.clone();
    for r in &graph.rows {
        let id = r.id as usize;
        let liquid = r
            .terms
            .iter()
            .any(|t| phi[t.cell_id as usize] <= 0.0 && fields.capacity[t.cell_id as usize] > 0.0);
        known[id] = liquid;
        fixed[id] = liquid || r.kind == RowKind::ClosedWorld || r.open_fraction <= 1e-8;
        if r.kind != RowKind::ClosedWorld && r.open_fraction > 1e-8 {
            values[id] =
                (values[id] - (1.0 - r.open_fraction) * r.solid_velocity) / r.open_fraction;
        }
        if !fixed[id] {
            values[id] = 0.0;
        }
    }
    let mut neighbours = vec![Vec::new(); graph.rows.len()];
    for r in &graph.rows {
        if fixed[r.id as usize] {
            continue;
        }
        let mut cells: Vec<usize> = r.terms.iter().map(|t| t.cell_id as usize).collect();
        let own = cells.clone();
        for c in own {
            for &edge in &graph.incidences[c] {
                let edge = &graph.rows[edge as usize];
                if edge.open_fraction <= 1e-8 || edge.kind == RowKind::ClosedWorld {
                    continue;
                }
                cells.extend(edge.terms.iter().map(|t| t.cell_id as usize));
            }
        }
        cells.sort_unstable();
        cells.dedup();
        let ids = &mut neighbours[r.id as usize];
        for c in cells {
            for &j in &graph.incidences[c] {
                let other = &graph.rows[j as usize];
                if other.axis != r.axis || j == r.id {
                    continue;
                }
                let dx = (other.center[0] - r.center[0]).abs();
                let dy = (other.center[1] - r.center[1]).abs();
                let span = 0.5 * (r.measure + other.measure);
                if dx <= span + 1e-6
                    && dy <= span + 1e-6
                    && (dx < 1e-6 || dy < 1e-6 || dx < 0.75 * span || dy < 0.75 * span)
                {
                    ids.push(j as usize);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids.sort_by(|&a, &b| {
            graph.rows[a].center[0]
                .total_cmp(&graph.rows[b].center[0])
                .then(graph.rows[a].center[1].total_cmp(&graph.rows[b].center[1]))
        });
    }
    for _ in 0..depth_count {
        let mut next = values.clone();
        let mut next_known = known.clone();
        let mut changed = false;
        for r in &graph.rows {
            let id = r.id as usize;
            if fixed[id] || known[id] {
                continue;
            }
            let (mut sum, mut weight) = (0.0_f64, 0.0_f64);
            for &j in &neighbours[id] {
                if !known[j] {
                    continue;
                }
                let other = &graph.rows[j];
                let d = (other.center[0] - r.center[0]).hypot(other.center[1] - r.center[1]) as f64;
                let w = other.measure as f64 / d.max(1e-8);
                sum += w * values[j] as f64;
                weight += w;
            }
            if weight > 0.0 {
                next[id] = (sum / weight) as f32;
                next_known[id] = true;
                changed = true;
            }
        }
        values = next;
        known = next_known;
        if !changed {
            break;
        }
    }
    for r in &graph.rows {
        let id = r.id as usize;
        if !fixed[id] {
            fields.face_velocity[id] =
                r.open_fraction * values[id] + (1.0 - r.open_fraction) * r.solid_velocity;
        }
    }
    Ok(())
}
