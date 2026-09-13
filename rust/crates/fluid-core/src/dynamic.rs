//! Exact 2-D centre-slab moving capacity, face geometry and inflow source plan.
use crate::rigid::RigidBodyState;
use crate::scene_model::{FluidInflow, PhysicalScene, SceneModelError, SliceFrame, Vec3};
use crate::sources::SourceLedger;
use crate::types::{Graph, RowKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct DynamicGeometryInput<'a> {
    pub scene: &'a PhysicalScene,
    pub frame: SliceFrame,
    pub graph: &'a Graph,
    pub time_s: f64,
    pub dt_s: f64,
    pub bodies: &'a [RigidBodyState],
    pub previous_bodies: &'a [RigidBodyState],
    pub density: Option<&'a [f32]>,
    pub pending_source_area_fine: f32,
    pub pending_source_compensation: f32,
    pub pressure_member: Option<&'a [u8]>,
    pub source_reduction_groups: Option<&'a [Vec<u32>]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicGeometry {
    pub capacity: Vec<f32>,
    pub capacity_before: Vec<f32>,
    pub capacity_after: Vec<f32>,
    pub capacity_rate: Vec<f32>,
    pub source_rate: Vec<f32>,
    pub open_fraction: Vec<f32>,
    pub open_fraction_before: Vec<f32>,
    pub open_fraction_after: Vec<f32>,
    pub mean_open_fraction: Vec<f32>,
    pub solid_velocity: Vec<f32>,
    pub inflow_coverage: Vec<f32>,
    pub inflow_velocity_fine: [f32; 2],
    pub requested_source_area_fine: f32,
    pub source_rate_area_fine: f32,
    pub source_available_area_fine: f32,
    pub source_factor: f32,
    pub source_component_count: u32,
    pub source_anchored_component_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRemapReceipt {
    pub volume_before: f64,
    pub volume_after_geometry: f64,
    pub capacity_before: f64,
    pub capacity_after: f64,
    pub closing_capacity: f64,
    pub opening_capacity: f64,
    pub excess_after_geometry: f64,
    pub requested_area: f32,
    pub planned_area: f64,
    pub balance_residual: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRemapPlan {
    pub density: Vec<f32>,
    pub capacity: Vec<f32>,
    pub capacity_rate: Vec<f32>,
    pub source_rate: Vec<f32>,
    pub pressure_capacity_rate: Vec<f32>,
    pub pressure_source_rate: Vec<f32>,
    pub ledger: SourceLedger,
    pub receipt: DynamicRemapReceipt,
}

#[inline]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
fn q_inverse_rotate(q: crate::scene_model::Quaternion, p: Vec3) -> Vec3 {
    let u = Vec3 {
        x: -q.x,
        y: -q.y,
        z: -q.z,
    };
    let uv = Vec3 {
        x: u.y * p.z - u.z * p.y,
        y: u.z * p.x - u.x * p.z,
        z: u.x * p.y - u.y * p.x,
    };
    let uuv = Vec3 {
        x: u.y * uv.z - u.z * uv.y,
        y: u.z * uv.x - u.x * uv.z,
        z: u.x * uv.y - u.y * uv.x,
    };
    Vec3 {
        x: p.x + 2.0 * (q.w * uv.x + uuv.x),
        y: p.y + 2.0 * (q.w * uv.y + uuv.y),
        z: p.z + 2.0 * (q.w * uv.z + uuv.z),
    }
}
fn contains(body: &RigidBodyState, world: Vec3) -> Result<bool, SceneModelError> {
    let p = q_inverse_rotate(
        body.orientation,
        Vec3 {
            x: world.x - body.position_m.x,
            y: world.y - body.position_m.y,
            z: world.z - body.position_m.z,
        },
    );
    let d = body.description.dimensions_m;
    Ok(match body.description.shape {
        crate::scene_model::RigidShape::Sphere => p.x.hypot(p.y).hypot(p.z) <= d.x,
        crate::scene_model::RigidShape::Box => {
            p.x.abs() <= 0.5 * d.x && p.y.abs() <= 0.5 * d.y && p.z.abs() <= 0.5 * d.z
        }
        crate::scene_model::RigidShape::Cylinder => {
            p.x * p.x + p.z * p.z <= d.x * d.x && p.y.abs() <= 0.5 * d.y
        }
        crate::scene_model::RigidShape::Capsule => {
            let cy = p.y.clamp(-0.5 * d.y, 0.5 * d.y);
            p.x.hypot(p.y - cy).hypot(p.z) <= d.x
        }
        crate::scene_model::RigidShape::Cup => {
            let t = d.z.clamp(1e-4, (0.95 * d.x).min(0.95 * d.y));
            let ox = p.x.hypot(p.z) - d.x;
            let oy = p.y.abs() - 0.5 * d.y;
            let outer = ox.max(0.0).hypot(oy.max(0.0)) + ox.max(oy).min(0.0);
            let cavity = (p.x.hypot(p.z) - (d.x - t)).max((-0.5 * d.y + t) - p.y);
            outer.max(-cavity) <= 0.0
        }
    })
}
fn velocity_at(body: &RigidBodyState, p: Vec3) -> Vec3 {
    let r = Vec3 {
        x: p.x - body.position_m.x,
        y: p.y - body.position_m.y,
        z: p.z - body.position_m.z,
    };
    let w = body.angular_velocity_rad_s;
    Vec3 {
        x: body.linear_velocity_m_s.x + w.y * r.z - w.z * r.y,
        y: body.linear_velocity_m_s.y + w.z * r.x - w.x * r.z,
        z: body.linear_velocity_m_s.z + w.x * r.y - w.y * r.x,
    }
}
fn world_point(frame: SliceFrame, p: [f32; 3]) -> Vec3 {
    Vec3 {
        x: frame.origin_x + p[0] as f64 * frame.source_cell_size,
        y: frame.origin_y + p[1] as f64 * frame.source_cell_size,
        z: frame.center_z,
    }
}
fn static_fraction(graph: &Graph, frame: SliceFrame, x: i32, y: i32) -> f32 {
    let [nx, ny, nz] = frame.source_dimensions;
    if x < 0
        || y < 0
        || frame.center_cell_z < 0
        || x >= nx as i32
        || y >= ny as i32
        || frame.center_cell_z >= nz as i32
    {
        return 0.0;
    }
    // A reduced graph owns exactly its centre slab.  The full source-z index
    // is retained only by the 3-D pressure graph.
    let z = if graph.dimension == 2 {
        0
    } else {
        frame.center_cell_z as usize
    };
    let i = x as usize + nx as usize * (y as usize + ny as usize * z);
    graph.solid_voxel_fraction.get(i).copied().unwrap_or(0.0)
}
fn compact_capacity(
    graph: &Graph,
    frame: SliceFrame,
    bodies: &[RigidBodyState],
) -> Result<Vec<f32>, SceneModelError> {
    let mut out = vec![0.0; graph.cells.len()];
    let h = frame.source_cell_size;
    for c in &graph.cells {
        let mut sum = 0.0;
        let mut count = 0;
        for y in (c.minimum[1].floor() as i32)..(c.maximum[1].ceil() as i32) {
            for x in (c.minimum[0].floor() as i32)..(c.maximum[0].ceil() as i32) {
                let center = Vec3 {
                    x: frame.origin_x + (x as f64 + 0.5) * h,
                    y: frame.origin_y + (y as f64 + 0.5) * h,
                    z: frame.center_z,
                };
                let mut maximum: f64 = 0.0;
                for b in bodies {
                    let mut covered = 0.0;
                    for dx in [-0.4, 0.4] {
                        for dy in [-0.4, 0.4] {
                            if contains(
                                b,
                                Vec3 {
                                    x: center.x + dx * h,
                                    y: center.y + dy * h,
                                    z: center.z,
                                },
                            )? {
                                covered += 0.25
                            }
                        }
                    }
                    maximum = maximum.max(covered)
                }
                sum += (1.0 - static_fraction(graph, frame, x, y) as f64) * (1.0 - maximum);
                count += 1
            }
        }
        out[c.id as usize] = (if count > 0 { sum / count as f64 } else { 0.0 }) as f32
    }
    Ok(out)
}
fn static_row_open(
    graph: &Graph,
    frame: SliceFrame,
    axis: usize,
    center: [f32; 3],
    area: f32,
    initial_open: f32,
) -> f64 {
    if graph.solid_voxel_fraction.is_empty() {
        return 1.0;
    }
    let plane = center[axis].round() as i32;
    if plane == 0 || plane == graph.dimensions[axis] as i32 {
        // The reduced plane does not store exterior voxels. Preserve the
        // constructor's exact SolidWorld face sample at domain boundaries.
        return initial_open as f64;
    }
    let tangent_axis = 1 - axis;
    let start = (center[tangent_axis] as f64 - 0.5 * area as f64).round() as i32;
    let n = (area as f64).round() as i32;
    let mut open = 0.0;
    for tangent in start..start + n {
        let (mut x, mut y) = if axis == 0 {
            (plane - 1, tangent)
        } else {
            (tangent, plane - 1)
        };
        let a = static_fraction(graph, frame, x, y);
        if axis == 0 {
            x = plane
        } else {
            y = plane
        }
        let b = static_fraction(graph, frame, x, y);
        open += 1.0 - a.max(b) as f64
    }
    if n > 0 {
        open / n as f64
    } else {
        0.0
    }
}
fn dynamic_rows(
    graph: &Graph,
    frame: SliceFrame,
    bodies: &[RigidBodyState],
) -> Result<(Vec<f32>, Vec<f32>), SceneModelError> {
    let mut open = vec![0.0; graph.rows.len()];
    let mut velocity = vec![0.0; graph.rows.len()];
    let h = frame.source_cell_size;
    for row in &graph.rows {
        if row.kind == RowKind::ClosedWorld {
            continue;
        }
        let center = world_point(frame, row.center);
        let tangent = row.measure as f64 * h;
        let (mut covered, mut speed) = (0.0, 0.0);
        for sign in [-0.35, 0.35] {
            let p = if row.axis == 0 {
                Vec3 {
                    x: center.x,
                    y: center.y + sign * tangent,
                    z: center.z,
                }
            } else {
                Vec3 {
                    x: center.x + sign * tangent,
                    y: center.y,
                    z: center.z,
                }
            };
            if let Some(b) = bodies.iter().find(|b| contains(b, p).unwrap_or(false)) {
                covered += 0.5;
                let v = velocity_at(b, p);
                speed += 0.5 * (if row.axis == 0 { v.x } else { v.y }) / h
            }
        }
        let mut fallback = None;
        if covered == 0.0 {
            'outer: for ns in [-0.4, 0.4] {
                for ts in [-0.4, 0.4] {
                    let normal = row.distance.max(1.0) as f64 * h;
                    let p = if row.axis == 0 {
                        Vec3 {
                            x: center.x + ns * normal,
                            y: center.y + ts * tangent,
                            z: center.z,
                        }
                    } else {
                        Vec3 {
                            x: center.x + ts * tangent,
                            y: center.y + ns * normal,
                            z: center.z,
                        }
                    };
                    if let Some(b) = bodies.iter().find(|b| contains(b, p).unwrap_or(false)) {
                        fallback = Some(b);
                        break 'outer;
                    }
                }
            }
        }
        let initial_open = row.static_open_fraction.unwrap_or(row.open_fraction);
        open[row.id as usize] = (static_row_open(
            graph,
            frame,
            row.axis as usize,
            row.center,
            row.measure,
            initial_open,
        ) * (1.0 - covered)) as f32;
        velocity[row.id as usize] = (if covered > 0.0 {
            speed / covered
        } else if let Some(b) = fallback {
            let v = velocity_at(b, center);
            (if row.axis == 0 { v.x } else { v.y }) / h
        } else {
            0.0
        }) as f32
    }
    Ok((open, velocity))
}
fn inflow_strength(i: &FluidInflow, t: f64) -> f64 {
    if t < i.start_s || t >= i.end_s {
        0.0
    } else if i.ramp_s <= 0.0 {
        1.0
    } else {
        1.0f64
            .min((t - i.start_s) / i.ramp_s)
            .min((i.end_s - t) / i.ramp_s)
    }
}
fn average_strength(i: &FluidInflow, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    let mut p = vec![
        a,
        b,
        i.start_s,
        i.end_s,
        i.start_s + i.ramp_s,
        i.end_s - i.ramp_s,
        0.5 * (i.start_s + i.end_s),
    ];
    p.retain(|x| *x >= a && *x <= b);
    p.sort_by(|x, y| x.total_cmp(y));
    p.dedup();
    let mut sum = 0.0;
    for w in p.windows(2) {
        sum += 0.5 * (inflow_strength(i, w[0]) + inflow_strength(i, w[1])) * (w[1] - w[0])
    }
    sum / (b - a)
}
fn compensated<I: IntoIterator<Item = f32>>(values: I) -> f32 {
    let (mut total, mut correction) = (0.0f32, 0.0f32);
    for source in values {
        let value = source - correction;
        let next = total + value;
        correction = (next - total) - value;
        total = next
    }
    total
}
fn grouped(values: &[f32], groups: Option<&[Vec<u32>]>) -> f32 {
    match groups {
        Some(g) if !g.is_empty() => compensated(g.iter().map(|ids| {
            compensated(
                ids.iter()
                    .map(|&id| values.get(id as usize).copied().unwrap_or(0.0)),
            )
        })),
        _ => compensated(values.iter().copied()),
    }
}

pub fn compute_dynamic_geometry(
    input: &DynamicGeometryInput<'_>,
) -> Result<DynamicGeometry, SceneModelError> {
    if !(input.dt_s > 0.0 && input.dt_s.is_finite()) {
        return Err(SceneModelError(
            "slice dynamic geometry dt must be finite and positive".into(),
        ));
    }
    for b in input.bodies.iter().chain(input.previous_bodies) {
        crate::scene_model::validate_supported_shape(b.description.shape)?
    }
    let after = compact_capacity(input.graph, input.frame, input.bodies)?;
    let before = compact_capacity(input.graph, input.frame, input.previous_bodies)?;
    let rate = after
        .iter()
        .zip(&before)
        .map(|(&a, &b)| ((a as f64 - b as f64) / input.dt_s) as f32)
        .collect();
    let (rows_after, solid_velocity) = dynamic_rows(input.graph, input.frame, input.bodies)?;
    let (rows_before, _) = dynamic_rows(input.graph, input.frame, input.previous_bodies)?;
    let mean: Vec<f32> = rows_after
        .iter()
        .zip(&rows_before)
        .map(|(&a, &b)| (0.5f32 * (b + a)) as f32)
        .collect();
    let mut out = DynamicGeometry {
        capacity: after.clone(),
        capacity_before: before,
        capacity_after: after,
        capacity_rate: rate,
        open_fraction: mean.clone(),
        mean_open_fraction: mean.clone(),
        open_fraction_before: rows_before,
        open_fraction_after: rows_after,
        solid_velocity,
        source_rate: vec![0.0; input.graph.cells.len()],
        inflow_coverage: vec![0.0; input.graph.rows.len()],
        ..Default::default()
    };
    let Some(inflow) = input.scene.fluid.inflow else {
        return Ok(out);
    };
    let strength = average_strength(&inflow, input.time_s, input.time_s + input.dt_s) as f32;
    let speed = inflow
        .velocity_m_s
        .x
        .hypot(inflow.velocity_m_s.y)
        .hypot(inflow.velocity_m_s.z);
    if !(strength > 0.0 && speed > 0.0) {
        return Ok(out);
    }
    let h = input.frame.source_cell_size;
    let direction = Vec3 {
        x: inflow.velocity_m_s.x / speed,
        y: inflow.velocity_m_s.y / speed,
        z: inflow.velocity_m_s.z / speed,
    };
    let half = 0.5 * inflow.length_m / speed;
    let outlet = Vec3 {
        x: inflow.center_m.x + inflow.velocity_m_s.x * half,
        y: inflow.center_m.y + inflow.velocity_m_s.y * half,
        z: inflow.center_m.z + inflow.velocity_m_s.z * half,
    };
    out.inflow_velocity_fine = [
        (inflow.velocity_m_s.x * strength as f64 / h) as f32,
        (inflow.velocity_m_s.y * strength as f64 / h) as f32,
    ];
    let plane = (input.frame.center_z - outlet.z).abs();
    let chord = if plane < inflow.radius_m {
        2.0 * (inflow.radius_m * inflow.radius_m - plane * plane).sqrt()
    } else {
        0.0
    };
    out.requested_source_area_fine =
        (inflow.velocity_m_s.x.hypot(inflow.velocity_m_s.y) * chord * strength as f64 / (h * h))
            as f32;
    let axis = if direction.y.abs() > direction.x.abs() && direction.y.abs() >= direction.z.abs() {
        1
    } else if direction.z.abs() > direction.x.abs() {
        2
    } else {
        0
    };
    for row in &input.graph.rows {
        if axis == 2 || row.axis as usize != axis || mean[row.id as usize] <= 0.0 {
            continue;
        }
        let p = world_point(input.frame, row.center);
        let r = Vec3 {
            x: p.x - outlet.x,
            y: p.y - outlet.y,
            z: p.z - outlet.z,
        };
        let axial = r.x * direction.x + r.y * direction.y + r.z * direction.z;
        let radial = (r.x - axial * direction.x)
            .hypot(r.y - axial * direction.y)
            .hypot(r.z - axial * direction.z);
        if (axial / h).abs() > 0.51 * row.distance.max(1.0) as f64 {
            continue;
        }
        let edge = (0.5 * row.distance.max(1.0) as f64) * h;
        out.inflow_coverage[row.id as usize] =
            clamp01(0.5 - (radial - inflow.radius_m) / edge) as f32
    }
    let mut weights = vec![0.0; input.graph.cells.len()];
    let mut eligible = vec![false; weights.len()];
    let effective_speed = (speed * strength as f64) as f32;
    for c in &input.graph.cells {
        let p = world_point(input.frame, c.center);
        let r = Vec3 {
            x: p.x - outlet.x,
            y: p.y - outlet.y,
            z: p.z - outlet.z,
        };
        let axial = (r.x * direction.x + r.y * direction.y + r.z * direction.z) / h;
        let radial = (r.x - axial * h * direction.x)
            .hypot(r.y - axial * h * direction.y)
            .hypot(r.z - axial * h * direction.z)
            / h;
        let minw = c.widths[0].min(c.widths[1]) as f64;
        let length = (effective_speed as f64 * input.dt_s / h).max(minw);
        let edge = (0.5 * minw).max(0.5);
        let coverage = clamp01(0.5 - (radial - inflow.radius_m / h) / edge)
            * clamp01(0.5 + axial / edge)
            * clamp01(0.5 + (length - axial) / edge);
        let id = c.id as usize;
        weights[id] = (coverage * out.capacity_after[id] as f64 * c.measure as f64) as f32;
        let fill = input
            .density
            .and_then(|d| d.get(id))
            .copied()
            .unwrap_or(0.0)
            / out.capacity_after[id].max(1e-6);
        eligible[id] = out.capacity_after[id] > 1e-8
            && (fill >= 0.5
                || input
                    .pressure_member
                    .and_then(|m| m.get(id))
                    .copied()
                    .unwrap_or(0)
                    != 0
                || weights[id] > 0.0)
    }
    let mut parent: Vec<i32> = eligible
        .iter()
        .enumerate()
        .map(|(i, &yes)| if yes { i as i32 } else { -1 })
        .collect();
    fn root(parent: &[i32], mut i: usize) -> usize {
        for _ in 0..64 {
            let p = parent[i] as usize;
            if p == i {
                break;
            }
            i = p
        }
        i
    }
    for row in &input.graph.rows {
        if mean[row.id as usize] <= 0.0 {
            continue;
        }
        let members: Vec<usize> = row
            .terms
            .iter()
            .filter(|t| eligible[t.cell_id as usize])
            .map(|t| root(&parent, t.cell_id as usize))
            .collect();
        if let Some(&minimum) = members.iter().min() {
            for member in members {
                let r = root(&parent, member);
                parent[r] = minimum as i32
            }
        }
    }
    for i in 0..parent.len() {
        if parent[i] >= 0 {
            parent[i] = root(&parent, i) as i32
        }
    }
    let components: BTreeSet<i32> = parent.iter().copied().filter(|&x| x >= 0).collect();
    let mut anchored = BTreeSet::new();
    for row in &input.graph.rows {
        if mean[row.id as usize] <= 0.0 {
            continue;
        }
        let (mut row_root, mut sum, mut scale) = (-1, 0.0f32, 0.0f32);
        for term in &row.terms {
            let id = term.cell_id as usize;
            if !eligible[id] {
                continue;
            }
            let candidate = parent[id];
            if row_root < 0 {
                row_root = candidate
            } else if row_root != candidate {
                continue;
            }
            sum += term.coefficient;
            scale += term.coefficient.abs()
        }
        if row_root >= 0 && sum.abs() > 9.5367431640625e-7 * scale {
            anchored.insert(row_root);
        }
    }
    for (id, w) in weights.iter_mut().enumerate() {
        if !anchored.contains(&parent[id]) {
            *w = 0.0
        }
    }
    let total = grouped(&weights, input.source_reduction_groups);
    let requested_area = (out.requested_source_area_fine as f64 * input.dt_s) as f32;
    let pending_increment = requested_area - input.pending_source_compensation;
    let pending = input.pending_source_area_fine + pending_increment;
    let factor = if total > 0.0 && pending > 0.0 {
        (pending.min(total) / total / input.dt_s as f32) as f32
    } else {
        0.0
    };
    for (i, w) in weights.iter().enumerate() {
        out.source_rate[i] = *w * factor
    }
    out.source_rate_area_fine = grouped(&out.source_rate, input.source_reduction_groups);
    out.source_available_area_fine = total;
    out.source_factor = factor;
    out.source_component_count = components.len() as u32;
    out.source_anchored_component_count = anchored.len() as u32;
    Ok(out)
}

pub fn plan_dynamic_remap(
    graph: &Graph,
    density: &[f32],
    geometry: &DynamicGeometry,
    dt: f64,
    mut ledger: SourceLedger,
) -> Result<DynamicRemapPlan, SceneModelError> {
    let n = graph.cells.len();
    if !(dt > 0.0 && dt.is_finite())
        || [
            density.len(),
            geometry.capacity_before.len(),
            geometry.capacity_after.len(),
            geometry.capacity_rate.len(),
            geometry.source_rate.len(),
        ]
        .iter()
        .any(|&x| x != n)
    {
        return Err(SceneModelError(
            "dynamic remap fields must align and dt must be positive".into(),
        ));
    }
    let (mut closing, mut opening, mut excess, mut planned) = (0.0, 0.0, 0.0, 0.0);
    let mut pressure_capacity = vec![0.0; n];
    for c in &graph.cells {
        let i = c.id as usize;
        let delta =
            (geometry.capacity_after[i] - geometry.capacity_before[i]) as f64 * c.measure as f64;
        if delta < 0.0 {
            closing -= delta
        } else {
            opening += delta
        }
        excess += (density[i] - geometry.capacity_after[i]).max(0.0) as f64 * c.measure as f64;
        pressure_capacity[i] = geometry.capacity_rate[i] * c.measure;
        planned += geometry.source_rate[i] as f64 * dt
    }
    ledger
        .plan(
            dt,
            geometry.requested_source_area_fine as f64,
            geometry.source_available_area_fine,
            geometry.source_factor,
            geometry.source_rate_area_fine,
        )
        .map_err(|e| SceneModelError(e.into()))?;
    let sum = |v: &[f32]| {
        graph
            .cells
            .iter()
            .fold((0.0, 0.0), |(total, correction), c| {
                let value = v[c.id as usize] as f64 * c.measure as f64 - correction;
                let next = total + value;
                (next, (next - total) - value)
            })
            .0
    };
    let volume = sum(density);
    Ok(DynamicRemapPlan {
        density: density.to_vec(),
        capacity: geometry.capacity_after.clone(),
        capacity_rate: geometry.capacity_rate.clone(),
        source_rate: geometry.source_rate.clone(),
        pressure_capacity_rate: pressure_capacity,
        pressure_source_rate: geometry.source_rate.clone(),
        ledger,
        receipt: DynamicRemapReceipt {
            volume_before: volume,
            volume_after_geometry: volume,
            capacity_before: sum(&geometry.capacity_before),
            capacity_after: sum(&geometry.capacity_after),
            closing_capacity: closing,
            opening_capacity: opening,
            excess_after_geometry: excess,
            requested_area: (geometry.requested_source_area_fine as f64 * dt) as f32,
            planned_area: planned,
            balance_residual: 0.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interval_average_spans_both_ramps() {
        let i = FluidInflow {
            center_m: Vec3::default(),
            radius_m: 1.0,
            length_m: 1.0,
            velocity_m_s: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            start_s: 0.0,
            end_s: 4.0,
            ramp_s: 1.0,
        };
        assert_eq!(average_strength(&i, 0.0, 4.0), 0.75)
    }

    /// Expected values were emitted by sliceDynamicGeometry for this one-cell
    /// continuous-source fixture.
    #[test]
    fn continuous_source_matches_typescript_reference() {
        let graph: Graph = serde_json::from_str(
            r#"{
          "dimension":2,"dimensions":[4,1],
          "cells":[{"id":0,"minimum":[0,0],"maximum":[1,1],"center":[0.75,0.5],
            "widths":[1,1],"area":1}],
          "rows":[{"id":0,"kind":"intra-brick","axis":0,"center":[0.5,0.5],
            "area":1,"distance":1,"dualWeight":1,
            "terms":[{"cellId":0,"coefficient":1}]}],"incidences":[[0]]
        }"#,
        )
        .unwrap();
        let scene: PhysicalScene = serde_json::from_str(
            r#"{
          "schemaVersion":"2.0.0","sceneId":"source",
          "container":{"width_m":4,"height_m":1,"depth_m":1},
          "fluid":{"density_kg_m3":1000,"dynamicViscosity_Pa_s":0.001,
            "surfaceTension_N_m":0.07,"gravity_m_s2":{"x":0,"y":-9.81,"z":0},
            "inflow":{"center_m":{"x":0,"y":0.5,"z":0},"radius_m":1,"length_m":1,
              "velocity_m_s":{"x":1,"y":0,"z":0},"start_s":0,"end_s":4,"ramp_s":0}},
          "rigidBodies":[]
        }"#,
        )
        .unwrap();
        let result = compute_dynamic_geometry(&DynamicGeometryInput {
            scene: &scene,
            frame: SliceFrame {
                origin_x: 0.0,
                origin_y: 0.0,
                center_z: 0.0,
                source_cell_size: 1.0,
                center_cell_z: 0,
                source_dimensions: [4, 1, 1],
            },
            graph: &graph,
            time_s: 1.0,
            dt_s: 0.25,
            bodies: &[],
            previous_bodies: &[],
            density: Some(&[0.0]),
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: None,
            source_reduction_groups: None,
        })
        .unwrap();
        assert_eq!(result.capacity, vec![1.0]);
        assert_eq!(result.source_rate, vec![2.0]);
        assert_eq!(result.inflow_coverage, vec![1.0]);
        assert_eq!(result.inflow_velocity_fine, [1.0, 0.0]);
        assert_eq!(
            (
                result.requested_source_area_fine,
                result.source_rate_area_fine,
                result.source_available_area_fine,
                result.source_factor
            ),
            (2.0, 2.0, 1.0, 2.0)
        );
        assert_eq!(
            (
                result.source_component_count,
                result.source_anchored_component_count
            ),
            (1, 1)
        );
    }
}
