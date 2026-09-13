//! Dimension-aware velocity, pressure-graph, and PLIC stage kernels.

use crate::kernels::{add, div, mul};
use crate::pressure::{solve_pressure_pcg, PressureError};
use crate::types::{Fields, Graph, Row, RowKind, SpatialOwnerCache, ValidationError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const LIQUID_ISOVALUE: f32 = 0.5;
pub const GHOST_FLUID_THETA_MIN: f32 = 0.05;
const VOLUME_ROUNDOFF_RATIO: f32 = 9.536_743e-7;

#[inline]
pub fn physical_row_measure(row: &Row) -> f32 {
    mul(row.static_measure.unwrap_or(row.measure), row.open_fraction)
}
#[inline]
fn pressure_dual_weight(row: &Row) -> f32 {
    let open = if row.kind == RowKind::ClosedWorld {
        if row.separating {
            1.0
        } else {
            0.0
        }
    } else {
        row.open_fraction
    };
    mul(row.static_dual_weight.unwrap_or(row.dual_weight), open)
}
#[inline]
fn fill(fields: &Fields, cell: usize) -> f32 {
    div(fields.density[cell], fields.capacity[cell].max(1e-8))
}

pub fn owner_at(graph: &Graph, point: [f32; 3]) -> Option<usize> {
    let d = graph.dimension as usize;
    match graph.spatial_owner_index.get_or_build(graph) {
        SpatialOwnerCache::Dense2d { dimensions, owners } => {
            if !(point[0] >= 0.0
                && point[0] < graph.dimensions[0]
                && point[1] >= 0.0
                && point[1] < graph.dimensions[1])
            {
                return None;
            }
            let x = point[0].floor() as usize;
            let y = point[1].floor() as usize;
            let id = owners[x + dimensions[0] * y];
            (id >= 0).then_some(id as usize)
        }
        SpatialOwnerCache::Sparse3d { widths, owners } => {
            if !(0..3).all(|a| point[a] >= 0.0 && point[a] < graph.dimensions[a]) {
                return None;
            }
            let q = point.map(|v| v.floor() as i32);
            widths
                .iter()
                .filter_map(|&width| {
                    let key = q.map(|v| v / (width as i32) * (width as i32));
                    let &id = owners.get(&(width, key))?;
                    let cell = &graph.cells[id as usize];
                    (0..3)
                        .all(|a| point[a] >= cell.minimum[a] && point[a] < cell.maximum[a])
                        .then_some(id as usize)
                })
                .min()
        }
        SpatialOwnerCache::LinearFallback => graph
            .cells
            .iter()
            .position(|c| (0..d).all(|a| point[a] >= c.minimum[a] && point[a] < c.maximum[a])),
    }
}

pub fn solid_voxel_fraction_at(graph: &Graph, x: i32, y: i32, z: i32) -> Option<f32> {
    if graph.solid_voxel_fraction.is_empty() {
        return None;
    }
    let nx = graph.dimensions[0] as i32;
    let ny = graph.dimensions[1] as i32;
    let nz = if graph.dimension == 2 {
        1
    } else {
        graph.dimensions[2] as i32
    };
    if x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz {
        return Some(1.0);
    }
    graph
        .solid_voxel_fraction
        .get((x + nx * (y + ny * z)) as usize)
        .copied()
}

pub fn solid_voxel_at(graph: &Graph, x: i32, y: i32, z: i32) -> bool {
    solid_voxel_fraction_at(graph, x, y, z).is_some_and(|fraction| fraction >= 128.0 / 255.0)
}

fn own_term(row: &Row, cell: usize) -> Option<crate::types::RowTerm> {
    row.terms
        .iter()
        .find(|t| t.cell_id as usize == cell)
        .copied()
}

fn velocity_extension_neighbor_weight(
    row: &Row,
    own: crate::types::RowTerm,
    other: crate::types::RowTerm,
) -> f32 {
    if own.coefficient * other.coefficient >= 0.0 {
        return 0.0;
    }
    let distance = row.distance;
    let own_fraction = mul(own.coefficient.abs(), distance);
    let other_fraction = mul(other.coefficient.abs(), distance);
    div(
        mul(mul(physical_row_measure(row), own_fraction), other_fraction),
        distance.max(1e-9),
    )
}

/// VEX2: one f32 seed publication followed by synchronous frozen-front transforms.
pub fn extend_velocity(
    graph: &Graph,
    fields: &mut Fields,
    depth_count: u8,
) -> Result<(), ValidationError> {
    fields.validate_for(graph)?;
    let n = graph.cells.len();
    let d = graph.dimension as usize;
    fields.extension_depth.fill(255);
    let mut velocity = fields.cell_velocity.clone();
    let mut known = vec![0u8; n];
    for cell in 0..n {
        if fields.density[cell] > LIQUID_ISOVALUE {
            known[cell] = 1;
            fields.extension_depth[cell] = 0;
        } else {
            for axis in 0..d {
                velocity[d * cell + axis] = 0.0;
            }
        }
    }
    for depth in 1..=depth_count {
        let mut next_velocity = velocity.clone();
        let mut next_known = known.clone();
        for cell in 0..n {
            if known[cell] != 0 {
                continue;
            }
            let mut side_weight = vec![0.0f32; 2 * d];
            let mut side_velocity = vec![vec![0.0f32; 2 * d]; d];
            for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                if physical_row_measure(row) <= 1e-8 {
                    continue;
                }
                let Some(own) = own_term(row, cell) else {
                    continue;
                };
                let side = 2 * row.axis as usize + usize::from(own.coefficient < 0.0);
                for &other in &row.terms {
                    let j = other.cell_id as usize;
                    if j == cell
                        || own.coefficient * other.coefficient >= 0.0
                        || known[j] == 0
                        || fields.extension_depth[j] >= depth
                    {
                        continue;
                    }
                    let weight = velocity_extension_neighbor_weight(row, own, other);
                    side_weight[side] = add(side_weight[side], weight);
                    for axis in 0..d {
                        side_velocity[axis][side] = add(
                            side_velocity[axis][side],
                            mul(weight, velocity[d * j + axis]),
                        );
                    }
                }
            }
            let pair_sum = |values: &[f32], axis: usize| {
                add(
                    values[2 * axis].min(values[2 * axis + 1]),
                    values[2 * axis].max(values[2 * axis + 1]),
                )
            };
            let mut weight = 0.0;
            for axis in 0..d {
                weight = add(weight, pair_sum(&side_weight, axis));
            }
            if weight > 0.0 {
                for component in 0..d {
                    let mut value = 0.0;
                    for axis in 0..d {
                        value = add(value, pair_sum(&side_velocity[component], axis));
                    }
                    next_velocity[d * cell + component] = div(value, weight);
                }
                next_known[cell] = 1;
                fields.extension_depth[cell] = depth;
            }
        }
        velocity = next_velocity;
        known = next_known;
    }
    fields.cell_velocity = velocity;
    Ok(())
}

/// Production body-force update, including unilateral closed-world release.
pub fn force_faces(
    graph: &mut Graph,
    fields: &mut Fields,
    dt: f32,
    acceleration: [f32; 3],
    inflow_velocity: [f32; 3],
) {
    let d = graph.dimension as usize;
    for row in &mut graph.rows {
        let i = row.id as usize;
        row.separating = false;
        if row.kind == RowKind::ClosedWorld {
            if let Some(term) = row.terms.first() {
                let cell = term.cell_id as usize;
                if fields.capacity[cell] > 1e-8 && fields.density[cell] > LIQUID_ISOVALUE {
                    let length = (0..d)
                        .map(|a| mul(acceleration[a], acceleration[a]))
                        .fold(0.0, add)
                        .sqrt();
                    let orientation = if term.coefficient >= 0.0 { 1.0 } else { -1.0 };
                    let predicted = add(
                        fields.cell_velocity[d * cell + row.axis as usize],
                        mul(dt, acceleration[row.axis as usize]),
                    );
                    let outward = mul(dt, mul(orientation, predicted - row.solid_velocity))
                        / row.distance.max(1e-6);
                    let deadband = if fields.pressure_row_member.get(i).copied().unwrap_or(0) != 0 {
                        5e-5
                    } else {
                        1e-4
                    };
                    row.separating = length > 1e-6
                        && mul(orientation, acceleration[row.axis as usize]) > mul(0.5, length)
                        && outward > deadband
                }
                fields.face_velocity[i] = if row.separating {
                    add(
                        fields.cell_velocity[d * cell + row.axis as usize],
                        mul(dt, acceleration[row.axis as usize]),
                    )
                } else {
                    row.solid_velocity
                }
            }
        } else if physical_row_measure(row) <= 1e-8 {
            fields.face_velocity[i] = row.solid_velocity
        } else {
            let forced = add(
                fields.face_velocity[i],
                mul(row.open_fraction, mul(dt, acceleration[row.axis as usize])),
            );
            let inflow = fields.inflow_coverage.get(i).copied().unwrap_or(0.0);
            fields.face_velocity[i] = add(
                mul(1.0 - inflow, forced),
                mul(inflow, inflow_velocity[row.axis as usize]),
            )
        }
    }
}

pub fn enforce_inflow_faces(graph: &Graph, fields: &mut Fields, inflow_velocity: [f32; 3]) {
    for row in &graph.rows {
        let i = row.id as usize;
        if physical_row_measure(row) <= 1e-8 {
            continue;
        }
        let coverage = fields.inflow_coverage.get(i).copied().unwrap_or(0.0);
        if coverage > 0.0 {
            fields.face_velocity[i] = add(
                mul(1.0 - coverage, fields.face_velocity[i]),
                mul(coverage, inflow_velocity[row.axis as usize]),
            )
        }
    }
}

fn face_support(graph: &Graph, fields: &Fields, x: f32, y: f32) -> ([f32; 2], f32, bool) {
    let Some(cell) = owner_at(graph, [x.floor() + 0.5, y.floor() + 0.5, 0.0]) else {
        return ([0.0; 2], 1.0, false);
    };
    let record = &graph.cells[cell];
    (
        [
            fields.cell_velocity[2 * cell],
            fields.cell_velocity[2 * cell + 1],
        ],
        record.widths[0].min(record.widths[1]).max(1.0),
        true,
    )
}
fn sample_support(graph: &Graph, fields: &Fields, x: f32, y: f32, span: f32) -> [f32; 2] {
    let bx = x.clamp(0.5 * span, graph.dimensions[0] - 0.5 * span);
    let by = y.clamp(0.5 * span, graph.dimensions[1] - 0.5 * span);
    let sx = bx / span - 0.5;
    let sy = by / span - 0.5;
    let lx = sx.floor();
    let ly = sy.floor();
    let tx = sx - lx;
    let ty = sy - ly;
    let mut result = [0.0; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            let weight = mul(
                if dx == 1 { tx } else { 1.0 - tx },
                if dy == 1 { ty } else { 1.0 - ty },
            );
            if weight == 0.0 {
                continue;
            }
            let (q, _, owned) = face_support(
                graph,
                fields,
                span * (lx + dx as f32 + 0.5),
                span * (ly + dy as f32 + 0.5),
            );
            if owned {
                for a in 0..2 {
                    result[a] = add(result[a], mul(weight, q[a]))
                }
            }
        }
    }
    result
}
fn inside_solid(graph: &Graph, fields: &Fields, point: [f32; 2]) -> bool {
    if solid_voxel_at(graph, point[0].floor() as i32, point[1].floor() as i32, 0) {
        return true;
    }
    owner_at(graph, [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0])
        .map(|i| fields.capacity[i] <= 1e-8)
        .unwrap_or(false)
}
fn clip_segment(graph: &Graph, fields: &Fields, start: [f32; 2], candidate: [f32; 2]) -> [f32; 2] {
    let (mut low, mut high, mut found) = (0.0, 1.0, false);
    for probe in 1..=8 {
        let t = div(probe as f32, 8.0);
        let q = [
            add(start[0], mul(t, candidate[0] - start[0])),
            add(start[1], mul(t, candidate[1] - start[1])),
        ];
        if inside_solid(graph, fields, q) {
            low = (probe - 1) as f32 / 8.0;
            high = t;
            found = true;
            break;
        }
    }
    if !found {
        return candidate;
    }
    for _ in 0..8 {
        let mid = mul(0.5, add(low, high));
        let q = [
            add(start[0], mul(mid, candidate[0] - start[0])),
            add(start[1], mul(mid, candidate[1] - start[1])),
        ];
        if inside_solid(graph, fields, q) {
            high = mid
        } else {
            low = mid
        }
    }
    let t = (low - 1e-4).max(0.0);
    [
        add(start[0], mul(t, candidate[0] - start[0])),
        add(start[1], mul(t, candidate[1] - start[1])),
    ]
}
fn trace_characteristic(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    span: f32,
    dt: f32,
    direction: f32,
) -> [f32; 2] {
    let initial = sample_support(graph, fields, position[0], position[1], span);
    let length = add(
        mul(div(initial[0], span), div(initial[0], span)),
        mul(div(initial[1], span), div(initial[1], span)),
    )
    .sqrt();
    let substeps = mul(length, dt).ceil().clamp(1.0, 16.0) as usize;
    let sub_dt = div(dt, substeps as f32);
    let mut traced = position;
    for step in 0..substeps {
        let first = if step == 0 {
            initial
        } else {
            sample_support(graph, fields, traced[0], traced[1], span)
        };
        let raw_mid = [
            (traced[0] + mul(direction, mul(mul(0.5, sub_dt), first[0])))
                .clamp(0.5 * span, graph.dimensions[0] - 0.5 * span),
            (traced[1] + mul(direction, mul(mul(0.5, sub_dt), first[1])))
                .clamp(0.5 * span, graph.dimensions[1] - 0.5 * span),
        ];
        let midpoint = clip_segment(graph, fields, traced, raw_mid);
        let middle = sample_support(graph, fields, midpoint[0], midpoint[1], span);
        let raw = [
            (traced[0] + mul(direction, mul(sub_dt, middle[0])))
                .clamp(0.5 * span, graph.dimensions[0] - 0.5 * span),
            (traced[1] + mul(direction, mul(sub_dt, middle[1])))
                .clamp(0.5 * span, graph.dimensions[1] - 0.5 * span),
        ];
        traced = clip_segment(graph, fields, traced, raw)
    }
    traced
}

fn row_touches_liquid(row: &Row, fields: &Fields) -> bool {
    row.terms
        .iter()
        .any(|t| fields.density[t.cell_id as usize] > LIQUID_ISOVALUE)
}
fn row_source_fluid_velocity(row: &Row, fields: &Fields) -> f32 {
    let velocity = fields.face_velocity[row.id as usize];
    if row.kind == RowKind::ClosedWorld {
        return velocity;
    }
    if row.open_fraction > 1e-6 {
        div(
            velocity - mul(1.0 - row.open_fraction, row.solid_velocity),
            row.open_fraction,
        )
    } else {
        row.solid_velocity
    }
}
fn source_staggered_cell_sample(
    graph: &Graph,
    fields: &Fields,
    point: [f32; 2],
    axis: usize,
) -> (f32, bool) {
    let mut query = [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0];
    let mut cell = owner_at(graph, query);
    if cell.is_none() {
        query[axis] -= 1.0;
        cell = owner_at(graph, query)
    }
    let Some(cell) = cell else {
        return (0.0, false);
    };
    if fields.capacity[cell] <= 1e-8 {
        return (0.0, false);
    }
    let own_cell = &graph.cells[cell];
    let tangent = 1 - axis;
    let mut values = [0.0; 2];
    let mut weights = [0.0; 2];
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis
            || !row_touches_liquid(row, fields)
            || fields.solid_motion_active && row.open_fraction < 1.0
        {
            continue;
        }
        let Some(own) = own_term(row, cell) else {
            continue;
        };
        let side = usize::from(own.coefficient < 0.0);
        let mut patch = 0.0;
        if row.terms.len() == 1 {
            let length = mul(
                own.coefficient.abs(),
                row.static_dual_weight.unwrap_or(row.dual_weight),
            );
            let expected = if side == 1 {
                own_cell.maximum[axis]
            } else {
                own_cell.minimum[axis]
            };
            if length != own_cell.widths[tangent]
                || row.center[tangent] != own_cell.center[tangent]
                || row.center[axis] != expected
            {
                return (0.0, false);
            }
            if point[tangent] >= own_cell.minimum[tangent]
                && point[tangent] < own_cell.maximum[tangent]
            {
                patch = length
            }
        } else {
            for term in &row.terms {
                if own.coefficient * term.coefficient >= 0.0 {
                    continue;
                }
                let other = &graph.cells[term.cell_id as usize];
                let lower = own_cell.minimum[tangent].max(other.minimum[tangent]);
                let upper = own_cell.maximum[tangent].min(other.maximum[tangent]);
                if point[tangent] >= lower && point[tangent] < upper {
                    patch = add(patch, upper - lower)
                }
            }
        }
        if patch > 0.0 {
            values[side] = add(
                values[side],
                mul(patch, row_source_fluid_velocity(row, fields)),
            );
            weights[side] = add(weights[side], patch)
        }
    }
    let fraction = ((point[axis] - own_cell.minimum[axis]) / own_cell.widths[axis]).clamp(0.0, 1.0);
    if fraction == 0.0 && weights[0] > 0.0 {
        return (div(values[0], weights[0]), true);
    }
    if fraction == 1.0 && weights[1] > 0.0 {
        return (div(values[1], weights[1]), true);
    }
    if weights[0] <= 0.0 || weights[1] <= 0.0 {
        return (0.0, false);
    }
    let low = div(values[0], weights[0]);
    let high = div(values[1], weights[1]);
    (add(mul(1.0 - fraction, low), mul(fraction, high)), true)
}
fn staggered_coordinates(
    graph: &Graph,
    position: [f32; 2],
    axis: usize,
    span: f32,
) -> ([f32; 2], [f32; 2], [f32; 2]) {
    let mut offset = [0.5, 0.5];
    offset[axis] = 0.0;
    let bounded = [
        position[0].clamp(offset[0] * span, graph.dimensions[0] - offset[0] * span),
        position[1].clamp(offset[1] * span, graph.dimensions[1] - offset[1] * span),
    ];
    let shifted = [bounded[0] / span - offset[0], bounded[1] / span - offset[1]];
    let lower = [shifted[0].floor(), shifted[1].floor()];
    (
        offset,
        lower,
        [shifted[0] - lower[0], shifted[1] - lower[1]],
    )
}
fn sample_source_linear(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    axis: usize,
    span: f32,
) -> f32 {
    let (offset, lower, fraction) = staggered_coordinates(graph, position, axis, span);
    let mut velocity = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let weight = mul(
                if dx == 1 {
                    fraction[0]
                } else {
                    1.0 - fraction[0]
                },
                if dy == 1 {
                    fraction[1]
                } else {
                    1.0 - fraction[1]
                },
            );
            if weight == 0.0 {
                continue;
            }
            let point = [
                span * (lower[0] + dx as f32 + offset[0]),
                span * (lower[1] + dy as f32 + offset[1]),
            ];
            let (value, valid) = source_staggered_cell_sample(graph, fields, point, axis);
            let fallback = sample_support(graph, fields, point[0], point[1], span)[axis];
            velocity = add(velocity, mul(weight, if valid { value } else { fallback }))
        }
    }
    velocity
}
fn uniform_staggered_node(
    graph: &Graph,
    fields: &Fields,
    point: [f32; 2],
    axis: usize,
    span: f32,
) -> (f32, bool) {
    let Some(cell) = owner_at(graph, [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0]) else {
        return (0.0, false);
    };
    if graph.cells[cell].widths[..2].iter().any(|&w| w != span) || fields.capacity[cell] != 1.0 {
        return (0.0, false);
    }
    let mut found = false;
    let mut value = 0.0;
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis || row.center[0] != point[0] || row.center[1] != point[1] {
            continue;
        }
        if row.terms.len() != 2
            || row.open_fraction != 1.0
            || row.static_measure.unwrap_or(row.measure) != span
            || row.distance != span
            || !row_touches_liquid(row, fields)
        {
            return (0.0, false);
        }
        let (mut negative, mut positive) = (false, false);
        for term in &row.terms {
            let endpoint = &graph.cells[term.cell_id as usize];
            if endpoint.widths[..2].iter().any(|&w| w != span)
                || fields.capacity[term.cell_id as usize] != 1.0
            {
                return (0.0, false);
            }
            let mut expected = point;
            if term.coefficient < 0.0 {
                expected[axis] -= 0.5 * span;
                negative = true
            } else if term.coefficient > 0.0 {
                expected[axis] += 0.5 * span;
                positive = true
            } else {
                return (0.0, false);
            }
            if endpoint.center[0] != expected[0] || endpoint.center[1] != expected[1] {
                return (0.0, false);
            }
        }
        if !negative || !positive || found {
            return (0.0, false);
        }
        found = true;
        value = fields.face_velocity[row.id as usize]
    }
    (value, found)
}
fn cubic_line(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    if t == 0.0 {
        return b;
    }
    if t == 1.0 {
        return c;
    }
    let inner = add(add(a - b, mul(2.0, c - b)), mul(-0.5, d - b));
    let cubic = add(mul(1.5, b - c), mul(0.5, d - a));
    add(
        b,
        mul(t, add(mul(0.5, c - a), mul(t, add(inner, mul(t, cubic))))),
    )
}
fn sample_source(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    axis: usize,
    span: f32,
) -> f32 {
    let (offset, lower, fraction) = staggered_coordinates(graph, position, axis, span);
    let interpolated = [fraction[0] != 0.0, fraction[1] != 0.0];
    let first = [
        span * (lower[0] - 1.0 + offset[0]),
        span * (lower[1] - 1.0 + offset[1]),
    ];
    let last = [
        span * (lower[0] + 2.0 + offset[0]),
        span * (lower[1] + 2.0 + offset[1]),
    ];
    let lower_bound = [offset[0] * span, offset[1] * span];
    let upper_bound = [
        graph.dimensions[0] - lower_bound[0],
        graph.dimensions[1] - lower_bound[1],
    ];
    if (!interpolated[0] && !interpolated[1])
        || (0..2)
            .any(|a| interpolated[a] && (first[a] < lower_bound[a] || last[a] > upper_bound[a]))
        || interpolated[axis] && (first[axis] <= 0.0 || last[axis] >= graph.dimensions[axis])
    {
        return sample_source_linear(graph, fields, position, axis, span);
    }
    let mut values = [[0.0; 4]; 4];
    let (mut core_min, mut core_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let x_count = if interpolated[0] { 4 } else { 1 };
    let y_count = if interpolated[1] { 4 } else { 1 };
    for yi in 0..y_count {
        let y = if interpolated[1] { yi } else { 1 };
        for xi in 0..x_count {
            let x = if interpolated[0] { xi } else { 1 };
            let point = [
                span * (lower[0] + x as f32 - 1.0 + offset[0]),
                span * (lower[1] + y as f32 - 1.0 + offset[1]),
            ];
            let (node, valid) = uniform_staggered_node(graph, fields, point, axis, span);
            if !valid {
                return sample_source_linear(graph, fields, position, axis, span);
            }
            values[y][x] = node;
            if (x == 1 || interpolated[0] && x == 2) && (y == 1 || interpolated[1] && y == 2) {
                core_min = core_min.min(node);
                core_max = core_max.max(node)
            }
        }
    }
    let mut lines = [0.0; 4];
    for yi in 0..y_count {
        let y = if interpolated[1] { yi } else { 1 };
        lines[y] = if interpolated[0] {
            cubic_line(
                values[y][0],
                values[y][1],
                values[y][2],
                values[y][3],
                fraction[0],
            )
        } else {
            values[y][1]
        }
    }
    let cubic = if interpolated[1] {
        cubic_line(lines[0], lines[1], lines[2], lines[3], fraction[1])
    } else {
        lines[1]
    };
    cubic.clamp(core_min, core_max)
}

/// Exact 2-D accepted-face semi-Lagrangian preparation.
pub fn prepare_faces(graph: &Graph, fields: &mut Fields, dt: f32) -> Result<(), ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::prepare_faces_3d(graph, fields, dt);
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "3-D face preparation requires 3-D staggered sampling".into(),
        ));
    }
    for row in &graph.rows {
        let i = row.id as usize;
        if physical_row_measure(row) <= 1e-8 {
            fields.face_velocity[i] = row.solid_velocity;
            continue;
        }
        let (mut touches_extended, mut sampling_width) = (false, f32::INFINITY);
        for term in &row.terms {
            let cell = term.cell_id as usize;
            touches_extended |= fields.extension_depth[cell] != 255;
            sampling_width = sampling_width.min(
                graph.cells[cell].widths[0]
                    .min(graph.cells[cell].widths[1])
                    .max(1.0),
            )
        }
        if !touches_extended {
            fields.face_velocity[i] = row.solid_velocity;
            continue;
        }
        let span = sampling_width.max(1.0);
        let mut region_width = span;
        for term in &row.terms {
            let cell = &graph.cells[term.cell_id as usize];
            if cell.refinement_region_scale.unwrap_or(1.0) > 1.0 {
                region_width = region_width.max(cell.widths[0].min(cell.widths[1]))
            }
        }
        let departure = trace_characteristic(
            graph,
            fields,
            [row.center[0], row.center[1]],
            region_width,
            dt,
            -1.0,
        );
        let characteristic = sample_source(graph, fields, departure, row.axis as usize, span);
        fields.face_velocity[i] = add(
            mul(row.open_fraction, characteristic),
            mul(1.0 - row.open_fraction, row.solid_velocity),
        )
    }
    Ok(())
}

pub fn trace_effective_transport_arrival(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 3],
    dt: f32,
) -> [f32; 3] {
    if graph.dimension == 3 {
        return crate::numerics3d::trace_characteristic_3d(graph, fields, position, dt, 1.0);
    }
    let span = owner_at(
        graph,
        [position[0].floor() + 0.5, position[1].floor() + 0.5, 0.0],
    )
    .map(|i| graph.cells[i].widths[0])
    .unwrap_or(1.0);
    let xy = trace_characteristic(graph, fields, [position[0], position[1]], span, dt, 1.0);
    [xy[0], xy[1], position[2]]
}

pub fn publish_transport_characteristic_clearance<'a>(
    graph: &Graph,
    fields: &'a mut Fields,
    dt: f32,
    source_density: Option<&[f32]>,
    source_gamma: Option<&[f32]>,
) -> Result<&'a [f32], ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::publish_characteristic_clearance_3d(
            graph,
            fields,
            dt,
            source_density,
            source_gamma,
        );
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "characteristic clearance is a 2-D stage".into(),
        ));
    }
    let density = source_density.unwrap_or(&fields.density);
    let gamma = source_gamma.unwrap_or(&fields.gamma);
    let mut result = vec![0.0; graph.cells.len()];
    for receiver in &graph.cells {
        let id = receiver.id as usize;
        if fields.capacity[id] <= 1e-8 {
            continue;
        }
        let span = receiver.widths[0];
        let departure = trace_characteristic(
            graph,
            fields,
            [receiver.center[0], receiver.center[1]],
            span,
            dt,
            -1.0,
        );
        let bx = departure[0].clamp(0.5 * span, graph.dimensions[0] - 0.5 * span);
        let by = departure[1].clamp(0.5 * span, graph.dimensions[1] - 0.5 * span);
        let sx = bx / span - 0.5;
        let sy = by / span - 0.5;
        let lx = sx.floor();
        let ly = sy.floor();
        let tx = sx - lx;
        let ty = sy - ly;
        let (mut visible, mut minimum_width, mut valid) = (0.0, f32::INFINITY, true);
        let (mut minx, mut miny, mut maxx, mut maxy) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for dy in 0..2 {
            for dx in 0..2 {
                let weight = mul(
                    if dx == 1 { tx } else { 1.0 - tx },
                    if dy == 1 { ty } else { 1.0 - ty },
                );
                visible = add(visible, weight);
                let Some(cell) = owner_at(
                    graph,
                    [
                        span * (lx + dx as f32 + 0.5),
                        span * (ly + dy as f32 + 0.5),
                        0.0,
                    ],
                ) else {
                    valid = false;
                    continue;
                };
                if fields.capacity[cell] <= 1e-8
                    || fields.capacity[cell] < 0.999999
                    || (density[cell] - 1.0).abs() > 0.005
                    || (gamma[cell] - 1.0).abs() > 0.005
                {
                    valid = false;
                    continue;
                }
                minimum_width = minimum_width
                    .min(graph.cells[cell].widths[0])
                    .min(graph.cells[cell].widths[1]);
                let vx = fields.cell_velocity[2 * cell];
                let vy = fields.cell_velocity[2 * cell + 1];
                minx = minx.min(vx);
                miny = miny.min(vy);
                maxx = maxx.max(vx);
                maxy = maxy.max(vy)
            }
        }
        if !valid || visible < 0.999999 {
            continue;
        }
        let dx = maxx - minx;
        let dy = maxy - miny;
        let maximum_delta = add(mul(dx, dx), mul(dy, dy)).sqrt();
        let clearance = mul(0.02, minimum_width);
        if mul(dt, maximum_delta) <= mul(0.002, minimum_width) {
            result[id] = clearance
        }
    }
    fields.characteristic_clearance = result;
    Ok(&fields.characteristic_clearance)
}

fn pressure_density(graph: &Graph, fields: &Fields, cell: usize) -> f32 {
    let mut density = fill(fields, cell);
    let final_capacity = mul(fields.capacity[cell], graph.cells[cell].measure);
    let before = mul(
        Fields::optional_cell(&fields.capacity_before, cell, fields.capacity[cell]),
        graph.cells[cell].measure,
    );
    let source = Fields::optional_cell(&fields.source_rate, cell, 0.0);
    if fields.solid_motion_active && final_capacity < before && fields.density[cell] > 0.0 {
        let rate = if fields.frame_dt > 0.0 {
            div(final_capacity - before, fields.frame_dt)
        } else {
            0.0
        };
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(-rate, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
        ));
    }
    if source > 0.0 {
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(source, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
        ));
    }
    density
}

fn volume_roundoff(capacity: f32) -> f32 {
    mul(VOLUME_ROUNDOFF_RATIO, capacity)
}

fn moving_pressure_predicted_fill(graph: &Graph, fields: &Fields, cell: usize) -> bool {
    if !fields.solid_motion_active {
        return false;
    }
    let mut equation = 0.0;
    let mut correction = 0.0;
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        let Some(own) = own_term(row, cell) else {
            continue;
        };
        let velocity = fields.face_velocity[row.id as usize]
            - mul(1.0 - row.open_fraction, row.solid_velocity);
        let value = mul(
            own.coefficient,
            mul(row.static_dual_weight.unwrap_or(row.dual_weight), velocity),
        );
        if value > 0.0 {
            let supported = row
                .terms
                .iter()
                .filter(|t| t.cell_id as usize != cell && t.coefficient * own.coefficient < 0.0)
                .any(|t| {
                    let j = t.cell_id as usize;
                    mul(fields.density[j], graph.cells[j].measure)
                        > volume_roundoff(mul(fields.capacity[j], graph.cells[j].measure))
                        || Fields::optional_cell(&fields.source_rate, j, 0.0) > 0.0
                });
            if !supported {
                continue;
            }
        }
        let adjusted = value - correction;
        let next = add(equation, adjusted);
        correction = (next - equation) - adjusted;
        equation = next;
    }
    let predicted = add(
        mul(fields.density[cell], graph.cells[cell].measure),
        mul(fields.frame_dt, equation),
    );
    let cap = mul(fields.capacity[cell], graph.cells[cell].measure);
    predicted >= cap - volume_roundoff(cap)
}

pub fn prepare_pressure_membership(graph: &Graph, fields: &mut Fields) {
    let prior = fields.pressure_member.clone();
    fields.pressure_member.fill(0);
    for cell in 0..graph.cells.len() {
        let mut submerged = prior[cell] != 0;
        let mut neighbors = 0;
        if submerged {
            'rows: for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                if row.terms.len() < 2 {
                    submerged = false;
                    break;
                }
                for term in &row.terms {
                    if term.cell_id as usize != cell {
                        neighbors += 1;
                        if prior[term.cell_id as usize] == 0 {
                            submerged = false;
                            break 'rows;
                        }
                    }
                }
            }
            submerged &= neighbors > 0;
        }
        let member = (pressure_density(graph, fields, cell) >= LIQUID_ISOVALUE
            || submerged
            || moving_pressure_predicted_fill(graph, fields, cell)
            || Fields::optional_cell(&fields.source_rate, cell, 0.0) > 0.0)
            && mul(fields.capacity[cell], graph.cells[cell].measure) > 1e-8;
        fields.pressure_member[cell] = u8::from(member);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureRows {
    pub active: Vec<u8>,
    pub theta: Vec<f32>,
}

fn ghost_theta(liquid_phi: f32, air_phi: f32) -> f32 {
    ((liquid_phi.abs() as f64 / ((liquid_phi.abs() as f64 + air_phi.abs() as f64).max(1e-12)))
        .clamp(GHOST_FLUID_THETA_MIN as f64, 1.0)) as f32
}

/// Builds PCM/PCF row membership and the Jacobi diagonal in canonical row order.
pub fn prepare_pressure_topology(graph: &Graph, fields: &mut Fields) -> PressureRows {
    prepare_pressure_membership(graph, fields);
    let mut state = PressureRows {
        active: vec![0; graph.rows.len()],
        theta: vec![0.0; graph.rows.len()],
    };
    for row in &graph.rows {
        let d = graph.dimension as usize;
        let mut geometric = [0.0f32; 3];
        let mut geometric_offset = 0.0;
        let mut geometric_weight = 0.0;
        if row.kind != RowKind::ClosedWorld {
            for term in &row.terms {
                let i = term.cell_id as usize;
                let base = d * i;
                let length = (0..d)
                    .map(|a| {
                        mul(
                            fields.interface_normal[base + a],
                            fields.interface_normal[base + a],
                        )
                    })
                    .fold(0.0, add);
                if length <= 0.5 {
                    continue;
                }
                let weight = term.coefficient.abs();
                for a in 0..d {
                    geometric[a] = add(geometric[a], mul(weight, fields.interface_normal[base + a]))
                }
                let shifted = if d == 2 {
                    add(
                        fields.interface_offset[i],
                        add(
                            mul(
                                fields.interface_normal[base],
                                graph.cells[i].center[0] - row.center[0],
                            ),
                            mul(
                                fields.interface_normal[base + 1],
                                graph.cells[i].center[1] - row.center[1],
                            ),
                        ),
                    )
                } else {
                    (0..d)
                        .map(|a| {
                            mul(
                                fields.interface_normal[base + a],
                                graph.cells[i].center[a] - row.center[a],
                            )
                        })
                        .fold(fields.interface_offset[i], add)
                };
                geometric_offset = add(geometric_offset, mul(weight, shifted));
                geometric_weight = add(geometric_weight, weight)
            }
        }
        let geometric_length = (0..d)
            .map(|a| mul(geometric[a], geometric[a]))
            .fold(0.0, add)
            .sqrt();
        let mut geometry_valid =
            geometric_weight > 1e-8 && geometric_length > mul(1e-6, geometric_weight);
        if geometry_valid {
            for term in &row.terms {
                let i = term.cell_id as usize;
                let numerator = if d == 2 {
                    add(
                        mul(geometric[0], graph.cells[i].center[0] - row.center[0]),
                        mul(geometric[1], graph.cells[i].center[1] - row.center[1]),
                    ) - geometric_offset
                } else {
                    (0..d)
                        .map(|a| mul(geometric[a], graph.cells[i].center[a] - row.center[a]))
                        .fold(-geometric_offset, add)
                };
                let phi = div(numerator, geometric_length);
                geometry_valid &= fields.capacity[i] >= 0.999999
                    && if fields.pressure_member[i] != 0 {
                        phi <= 0.0
                    } else {
                        phi >= 0.0
                    };
            }
        }
        let mut liquid_phi = 0.0;
        let mut liquid_weight = 0.0;
        let mut air_phi = 0.0;
        let mut air_weight = 0.0;
        let mut liquid_count = 0;
        let mut air_count = 0;
        let mut full_gradient = 0.0;
        let mut liquid_gradient = 0.0;
        for term in &row.terms {
            let i = term.cell_id as usize;
            let width = if row.kind == RowKind::SparseAir {
                1.0
            } else {
                graph.cells[i].widths[row.axis as usize]
            };
            let old_phi = mul(LIQUID_ISOVALUE - pressure_density(graph, fields, i), width);
            let phi = if geometry_valid {
                let numerator = if d == 2 {
                    add(
                        mul(geometric[0], graph.cells[i].center[0] - row.center[0]),
                        mul(geometric[1], graph.cells[i].center[1] - row.center[1]),
                    ) - geometric_offset
                } else {
                    (0..d)
                        .map(|a| mul(geometric[a], graph.cells[i].center[a] - row.center[a]))
                        .fold(-geometric_offset, add)
                };
                div(numerator, geometric_length)
            } else {
                old_phi
            };
            let weight = term.coefficient.abs();
            full_gradient = add(full_gradient, mul(term.coefficient, phi));
            if fields.pressure_member[i] != 0 {
                liquid_count += 1;
                liquid_phi = add(liquid_phi, mul(weight, phi));
                liquid_weight = add(liquid_weight, weight);
                liquid_gradient = add(liquid_gradient, mul(term.coefficient, phi));
            } else {
                air_count += 1;
                air_phi = add(air_phi, mul(weight, phi));
                air_weight = add(air_weight, weight);
            }
        }
        if liquid_count == 0 || pressure_dual_weight(row) <= 1e-8 {
            continue;
        }
        if row.kind == RowKind::SparseAir {
            air_phi = add(air_phi, mul(liquid_weight, 0.5));
            air_weight = add(air_weight, liquid_weight);
        }
        let cut = air_count > 0 || row.kind == RowKind::SparseAir;
        let mut theta = if cut {
            ghost_theta(
                div(liquid_phi, liquid_weight.max(1e-9)),
                div(air_phi, air_weight.max(1e-9)),
            )
        } else {
            1.0
        };
        let gravity_length = (0..d)
            .map(|a| mul(fields.acceleration_fine[a], fields.acceleration_fine[a]))
            .fold(0.0, add)
            .sqrt();
        let partial_region = graph
            .cells
            .iter()
            .any(|c| c.refinement_region_scale.unwrap_or(1.0) > 1.0)
            && graph
                .cells
                .iter()
                .any(|c| c.refinement_region_scale.unwrap_or(1.0) == 1.0);
        if cut
            && graph.dimension == 2
            && row.axis == 1
            && gravity_length > 1e-6
            && partial_region
            && fields.acceleration_fine[1] < 0.0
            && fields.acceleration_fine[0].abs() <= mul(1e-6, gravity_length)
        {
            let (height, height_valid) = pressure_planar_column_height(graph, fields, row);
            let liquid_y = div(
                row.terms
                    .iter()
                    .filter(|t| fields.pressure_member[t.cell_id as usize] != 0)
                    .map(|t| {
                        mul(
                            t.coefficient.abs(),
                            graph.cells[t.cell_id as usize].center[1],
                        )
                    })
                    .fold(0.0, add),
                liquid_weight.max(1e-9),
            );
            let mut air_y_sum = 0.0;
            let mut hydro_air_weight = 0.0;
            for term in &row.terms {
                if fields.pressure_member[term.cell_id as usize] == 0 {
                    air_y_sum = add(
                        air_y_sum,
                        mul(
                            term.coefficient.abs(),
                            graph.cells[term.cell_id as usize].center[1],
                        ),
                    );
                    hydro_air_weight = add(hydro_air_weight, term.coefficient.abs())
                }
            }
            if row.kind == RowKind::SparseAir {
                let direction = if row.center[1] >= liquid_y { 1.0 } else { -1.0 };
                air_y_sum = add(
                    air_y_sum,
                    mul(liquid_weight, add(liquid_y, mul(direction, row.distance))),
                );
                hydro_air_weight = add(hydro_air_weight, liquid_weight)
            }
            let air_y = div(air_y_sum, hydro_air_weight.max(1e-9));
            if height_valid && air_y > liquid_y + 1e-6 && height > liquid_y && height < air_y {
                theta = div(height - liquid_y, air_y - liquid_y).clamp(GHOST_FLUID_THETA_MIN, 1.0)
            }
        }
        if cut && row.kind == RowKind::MixedSeam {
            let factor = if full_gradient == 0.0 {
                0.0
            } else if liquid_gradient == 0.0 {
                div(1.0, GHOST_FLUID_THETA_MIN)
            } else {
                div(full_gradient, liquid_gradient).clamp(0.0, div(1.0, GHOST_FLUID_THETA_MIN))
            };
            theta = if factor > 0.0 { div(1.0, factor) } else { 0.0 };
        }
        state.theta[row.id as usize] = theta;
        state.active[row.id as usize] = 1;
    }
    fields.pressure_diagonal.fill(0.0);
    let d = graph.dimension as usize;
    for cell in 0..graph.cells.len() {
        if fields.pressure_member[cell] == 0 {
            continue;
        }
        let mut axes = vec![0.0; d];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            if state.active[row.id as usize] == 0 || state.theta[row.id as usize] <= 0.0 {
                continue;
            }
            if let Some(own) = own_term(row, cell) {
                let contribution = div(
                    mul(
                        pressure_dual_weight(row),
                        mul(own.coefficient, own.coefficient),
                    ),
                    state.theta[row.id as usize],
                );
                axes[row.axis as usize] = add(axes[row.axis as usize], contribution);
            }
        }
        fields.pressure_diagonal[cell] = axes.into_iter().fold(0.0, add);
    }
    fields.pressure_row_member = state.active.clone();
    state
}

fn pressure_integrated_column_height(graph: &Graph, fields: &Fields, x: i32) -> (f32, bool) {
    let ny = graph.dimensions[1] as i32;
    let mut y = 0;
    let mut mass = 0.0;
    let mut previous = 1.0;
    let mut column_open = -1.0;
    let (mut saw_open, mut saw_liquid, mut saw_air) = (false, false, false);
    while y < ny {
        if solid_voxel_fraction_at(graph, x, y, 0).unwrap_or(0.0) >= 1.0 {
            return (0.0, false);
        }
        let owner = owner_at(graph, [x as f32 + 0.5, y as f32 + 0.5, 0.0]);
        let (fill_value, width) = if let Some(cell) = owner {
            let open = fields.capacity[cell];
            if open <= 1e-6 {
                return (0.0, false);
            }
            saw_open = true;
            if column_open < 0.0 {
                column_open = open
            }
            if (open - column_open).abs() > 1e-3 {
                return (0.0, false);
            }
            let value = pressure_density(graph, fields, cell).clamp(0.0, 1.0);
            let width = (graph.cells[cell].widths[1] as i32
                - y % (graph.cells[cell].widths[1] as i32))
                .max(1)
                .min(ny - y);
            (value, width)
        } else {
            (0.0, (8 - y % 8).max(1).min(ny - y))
        };
        if fill_value > previous + 0.01 {
            return (0.0, false);
        }
        previous = fill_value;
        saw_liquid |= fill_value > 1e-3;
        saw_air |= fill_value < 1.0 - 1e-3;
        mass = add(mass, mul(fill_value, width as f32));
        y += width
    }
    (mass, saw_open && saw_liquid && saw_air)
}
fn pressure_planar_column_height(graph: &Graph, fields: &Fields, row: &Row) -> (f32, bool) {
    let nx = graph.dimensions[0] as i32;
    let centre = (row.center[0].floor() as i32).clamp(0, nx - 1);
    let mut height = 0.0;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut valid = true;
    for offset in [0, -1, 1] {
        let (value, ok) =
            pressure_integrated_column_height(graph, fields, (centre + offset).clamp(0, nx - 1));
        if offset == 0 {
            height = value
        }
        valid &= ok;
        minimum = minimum.min(value);
        maximum = maximum.max(value)
    }
    valid &= maximum - minimum <= 0.01;
    (height, valid)
}

fn row_gradient(row: &Row, member: &[u8], input: &[f32]) -> f32 {
    if row.terms.len() == 2 && row.terms[0].coefficient == -row.terms[1].coefficient {
        let a = row.terms[0];
        let b = row.terms[1];
        return mul(
            b.coefficient,
            (if member[b.cell_id as usize] != 0 {
                input[b.cell_id as usize]
            } else {
                0.0
            }) - (if member[a.cell_id as usize] != 0 {
                input[a.cell_id as usize]
            } else {
                0.0
            }),
        );
    }
    let mut negative = 0.0;
    let mut positive = 0.0;
    for term in &row.terms {
        let i = term.cell_id as usize;
        if member[i] != 0 {
            let v = mul(term.coefficient, input[i]);
            if term.coefficient < 0.0 {
                negative = add(negative, v)
            } else {
                positive = add(positive, v)
            }
        }
    }
    add(negative, positive)
}

pub fn apply_pressure_operator(
    graph: &Graph,
    member: &[u8],
    rows: &PressureRows,
    input: &[f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    let d = graph.dimension as usize;
    let compute = |cell: usize| {
        if member[cell] == 0 {
            return 0.0;
        }
        let mut neg = [0.0; 3];
        let mut pos = [0.0; 3];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            let ri = row.id as usize;
            if rows.active[ri] == 0 || rows.theta[ri] <= 0.0 {
                continue;
            }
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            let contribution = div(
                mul(
                    pressure_dual_weight(row),
                    mul(own.coefficient, row_gradient(row, member, input)),
                ),
                rows.theta[ri],
            );
            let a = row.axis as usize;
            if own.coefficient > 0.0 {
                neg[a] = add(neg[a], contribution)
            } else {
                pos[a] = add(pos[a], contribution)
            }
        }
        let mut total = 0.0;
        for a in 0..d {
            total = add(total, add(neg[a].min(pos[a]), neg[a].max(pos[a])))
        }
        total
    };
    #[cfg(feature = "parallel")]
    if output.len() >= crate::kernels::PARALLEL_POINTWISE_THRESHOLD {
        const CHUNK: usize = 1024;
        output
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let base = chunk_index * CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = compute(base + offset)
                }
            });
        return;
    }
    for (cell, value) in output.iter_mut().enumerate() {
        *value = compute(cell)
    }
}

pub fn assemble_pressure_rhs(graph: &Graph, fields: &mut Fields, rows: &PressureRows) {
    fields.pressure_rhs.fill(0.0);
    let d = graph.dimension as usize;
    for cell in 0..graph.cells.len() {
        if fields.pressure_member[cell] == 0 {
            continue;
        }
        let mut neg = [0.0; 3];
        let mut pos = [0.0; 3];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            if rows.active[row.id as usize] == 0 {
                continue;
            }
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            let fluid = fields.face_velocity[row.id as usize]
                - mul(1.0 - row.open_fraction, row.solid_velocity);
            let value = mul(
                own.coefficient,
                mul(row.static_dual_weight.unwrap_or(row.dual_weight), fluid),
            );
            let a = row.axis as usize;
            if own.coefficient > 0.0 {
                neg[a] = add(neg[a], value)
            } else {
                pos[a] = add(pos[a], value)
            }
        }
        let mut rhs = 0.0;
        for a in 0..d {
            rhs = add(rhs, add(neg[a].min(pos[a]), neg[a].max(pos[a])))
        }
        rhs = add(
            rhs,
            -mul(
                Fields::optional_cell(&fields.capacity_rate, cell, 0.0),
                graph.cells[cell].measure,
            ),
        );
        rhs = add(rhs, Fields::optional_cell(&fields.source_rate, cell, 0.0));
        fields.pressure_rhs[cell] = rhs;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureReceipt {
    pub iterations: u32,
    pub initial_residual: f32,
    pub residual: f32,
    pub converged: bool,
}

pub fn solve_pressure(
    graph: &Graph,
    fields: &mut Fields,
    rows: &PressureRows,
    maximum_iterations: u32,
    relative_tolerance: f32,
    execution_order: Option<&[u32]>,
) -> Result<PressureReceipt, PressureError> {
    let result = solve_pressure_pcg(
        &fields.pressure_diagonal,
        &fields.pressure_rhs,
        &fields.pressure,
        &fields.pressure_member,
        execution_order,
        maximum_iterations,
        relative_tolerance,
        |input, output| {
            apply_pressure_operator(graph, &fields.pressure_member, rows, input, output)
        },
    )?;
    fields.pressure = result.pressure;
    Ok(PressureReceipt {
        iterations: result.iterations,
        initial_residual: result.initial_true_residual_squared.max(0.0).sqrt(),
        residual: result.final_true_residual_squared.max(0.0).sqrt(),
        converged: result.converged,
    })
}

pub fn collocate_velocity(graph: &Graph, fields: &mut Fields) {
    let d = graph.dimension as usize;
    fields.cell_velocity.fill(0.0);
    let mut weights = vec![0.0; fields.cell_velocity.len()];
    for row in &graph.rows {
        for term in &row.terms {
            let at = d * term.cell_id as usize + row.axis as usize;
            let weight = mul(
                term.coefficient.abs(),
                row.static_dual_weight.unwrap_or(row.dual_weight),
            );
            let fluid = if row.open_fraction > 1e-6 {
                div(
                    fields.face_velocity[row.id as usize]
                        - mul(1.0 - row.open_fraction, row.solid_velocity),
                    row.open_fraction,
                )
            } else {
                row.solid_velocity
            };
            fields.cell_velocity[at] = add(fields.cell_velocity[at], mul(weight, fluid));
            weights[at] = add(weights[at], weight)
        }
    }
    for (i, v) in fields.cell_velocity.iter_mut().enumerate() {
        if weights[i] > 0.0 {
            *v = div(*v, weights[i])
        }
    }
}

pub fn project_pressure_velocity(graph: &Graph, fields: &mut Fields, rows: &PressureRows) {
    for row in &graph.rows {
        let i = row.id as usize;
        if rows.active[i] == 0 || rows.theta[i] <= 0.0 {
            continue;
        }
        let jump = row_gradient(row, &fields.pressure_member, &fields.pressure);
        let open = if row.kind == RowKind::ClosedWorld {
            if row.separating {
                1.0
            } else {
                0.0
            }
        } else {
            row.open_fraction
        };
        fields.face_velocity[i] = add(
            fields.face_velocity[i],
            -div(mul(open, jump), rows.theta[i]),
        );
    }
    collocate_velocity(graph, fields)
}

pub fn project_velocity(
    graph: &Graph,
    fields: &mut Fields,
    maximum_iterations: u32,
    relative_tolerance: f32,
) -> Result<PressureReceipt, PressureError> {
    let rows = prepare_pressure_topology(graph, fields);
    assemble_pressure_rhs(graph, fields, &rows);
    let receipt = solve_pressure(
        graph,
        fields,
        &rows,
        maximum_iterations,
        relative_tolerance,
        None,
    )?;
    project_pressure_velocity(graph, fields, &rows);
    Ok(receipt)
}

fn box_fraction_projected(px: f32, py: f32, offset: f32) -> f32 {
    let dominant = px.max(py);
    if dominant <= 1e-20 {
        return if offset >= 0.0 { 1.0 } else { 0.0 };
    }
    let mut spans = Vec::with_capacity(2);
    for v in [div(px, dominant), div(py, dominant)] {
        if v >= 1e-6 {
            spans.push(v)
        }
    }
    let total = spans.iter().copied().fold(0.0, add);
    let shifted = add(div(offset, dominant), mul(0.5, total));
    if shifted <= 0.0 {
        return 0.0;
    }
    if shifted >= total {
        return 1.0;
    }
    let complement = shifted > mul(0.5, total);
    let x = if complement { total - shifted } else { shifted };
    let fraction = if spans.len() == 1 {
        div(x, spans[0])
    } else {
        let aa = spans[0].min(spans[1]);
        let bb = spans[0].max(spans[1]);
        if x < aa {
            mul(mul(0.5, div(x, aa)), div(x, bb))
        } else {
            div(x - mul(0.5, aa), bb)
        }
    }
    .clamp(0.0, 1.0);
    if complement {
        1.0 - fraction
    } else {
        fraction
    }
}
fn box_fraction(nx: f32, ny: f32, offset: f32, wx: f32, wy: f32) -> f32 {
    box_fraction_projected(mul(nx.abs(), wx), mul(ny.abs(), wy), offset)
}
fn offset_for_fill(fill: f32, nx: f32, ny: f32, widths: [f32; 2]) -> f32 {
    let projected = [mul(nx.abs(), widths[0]), mul(ny.abs(), widths[1])];
    let dominant = projected[0].max(projected[1]);
    let radius = mul(0.5, add(projected[0], projected[1]));
    if fill <= 0.0 {
        return -radius;
    }
    if fill >= 1.0 {
        return radius;
    }
    if fill == 0.5 || dominant <= 1e-20 {
        return 0.0;
    }
    let mut spans = Vec::new();
    for value in projected {
        let q = div(value, dominant);
        if q >= 1e-6 {
            spans.push(q)
        }
    }
    if spans.len() == 1 {
        return mul(mul(fill - 0.5, spans[0]), dominant);
    }
    let complement = fill > 0.5;
    let target = if complement { 1.0 - fill } else { fill };
    let aa = spans[0].min(spans[1]);
    let bb = spans[0].max(spans[1]);
    let mut shifted = add(mul(target, bb), mul(0.5, aa));
    if target < div(mul(0.5, aa), bb) {
        shifted = mul(mul(2.0, target), mul(aa, bb)).sqrt()
    }
    let result = mul(shifted - mul(0.5, add(aa, bb)), dominant);
    if complement {
        -result
    } else {
        result
    }
}

#[derive(Clone, Copy)]
struct Plane {
    nx: f32,
    ny: f32,
    offset: f32,
}
fn plane_from_fill(fill: f32, gx: f32, gy: f32, widths: [f32; 2]) -> Plane {
    let maximum = gx.abs().max(gy.abs());
    if maximum <= 1e-20 {
        return Plane {
            nx: 0.0,
            ny: 0.0,
            offset: 0.0,
        };
    }
    let sx = div(gx, maximum);
    let sy = div(gy, maximum);
    let length = add(mul(sx, sx), mul(sy, sy)).sqrt();
    let nx = div(sx, length);
    let ny = div(sy, length);
    Plane {
        nx,
        ny,
        offset: offset_for_fill(fill, nx, ny, widths),
    }
}
fn certificate_sample(value: f32) -> Option<f32> {
    (value.is_finite() && (-VOLUME_ROUNDOFF_RATIO..=1.0 + VOLUME_ROUNDOFF_RATIO).contains(&value))
        .then(|| value.clamp(0.0, 1.0))
}
fn fit_score(plane: Plane, samples: &[f32; 9]) -> (f32, f32) {
    let mut full = 0.0;
    let mut sides = [0.0; 4];
    for j in 0..3 {
        for i in 0..3 {
            let displaced =
                plane.offset - add(mul(plane.nx, i as f32 - 1.0), mul(plane.ny, j as f32 - 1.0));
            let difference =
                box_fraction(plane.nx, plane.ny, displaced, 1.0, 1.0) - samples[3 * j + i];
            let error = mul(difference, difference);
            full = add(full, error);
            if i <= 1 {
                sides[0] = add(sides[0], error)
            }
            if i >= 1 {
                sides[1] = add(sides[1], error)
            }
            if j <= 1 {
                sides[2] = add(sides[2], error)
            }
            if j >= 1 {
                sides[3] = add(sides[3], error)
            }
        }
    }
    (
        div(full, 9.0),
        div(sides.into_iter().fold(f32::INFINITY, f32::min), 6.0),
    )
}

/// Production least-squares PLIC fallback. 2-D graphs use the analytic CM12
/// box inverse; true 3-D PLIC is deliberately rejected until polyhedron clipping lands.
pub fn reconstruct_interfaces(graph: &Graph, fields: &mut Fields) -> Result<(), ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::reconstruct_interfaces_3d(graph, fields);
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "3-D PLIC requires plane/polyhedron clipping".into(),
        ));
    }
    fields.interface_normal.fill(0.0);
    fields.interface_offset.fill(0.0);
    for cell in 0..graph.cells.len() {
        let rho = fill(fields, cell);
        if fields.capacity[cell] < 0.999999 || !(rho > 0.0 && rho < 1.0) {
            continue;
        }
        let c = &graph.cells[cell];
        let (mut mxx, mut mxy, mut myy, mut bx, mut by) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            if physical_row_measure(row) <= 1e-8 {
                continue;
            }
            for term in &row.terms {
                if own.coefficient * term.coefficient >= 0.0 {
                    continue;
                }
                let j = term.cell_id as usize;
                if fields.capacity[j] < 0.999999 {
                    continue;
                }
                let other = &graph.cells[j];
                let dx = other.center[0] - c.center[0];
                let dy = other.center[1] - c.center[1];
                let tangent = 1 - row.axis as usize;
                let overlap = (c.maximum[tangent].min(other.maximum[tangent])
                    - c.minimum[tangent].max(other.minimum[tangent]))
                .max(0.0);
                let weight = div(overlap, add(mul(dx, dx), mul(dy, dy)).max(1e-12));
                let delta = fill(fields, j) - rho;
                mxx = add(mxx, mul(weight, mul(dx, dx)));
                mxy = add(mxy, mul(weight, mul(dx, dy)));
                myy = add(myy, mul(weight, mul(dy, dy)));
                bx = add(bx, mul(weight, mul(dx, delta)));
                by = add(by, mul(weight, mul(dy, delta)));
            }
        }
        let determinant = mul(mxx, myy) - mul(mxy, mxy);
        let scale = mxx.max(myy);
        if !(scale > 1e-12 && determinant.abs() > 1e-7 * scale * scale) {
            continue;
        }
        let gx = div(mul(myy, bx) - mul(mxy, by), determinant);
        let gy = div(-mul(mxy, bx) + mul(mxx, by), determinant);
        let mut best = plane_from_fill(rho, -gx, -gy, [c.widths[0], c.widths[1]]);
        if best.nx == 0.0 && best.ny == 0.0 {
            continue;
        }
        let mut samples = [0.0; 9];
        let mut certified = c.widths[0] == 1.0 && c.widths[1] == 1.0;
        for j in 0..3 {
            for i in 0..3 {
                if !certified {
                    break;
                }
                let point = [
                    c.center[0] + i as f32 - 1.0,
                    c.center[1] + j as f32 - 1.0,
                    0.0,
                ];
                let Some(other_id) = owner_at(graph, point) else {
                    certified = false;
                    break;
                };
                let other = &graph.cells[other_id];
                let Some(sample) = certificate_sample(fill(fields, other_id)) else {
                    certified = false;
                    break;
                };
                if other.widths[0] != 1.0
                    || other.widths[1] != 1.0
                    || other.center[0] != point[0]
                    || other.center[1] != point[1]
                    || fields.capacity[other_id] < 0.999999
                {
                    certified = false;
                    break;
                }
                samples[3 * j + i] = sample
            }
        }
        if certified {
            let mut best_score = fit_score(best, &samples);
            let mut corner = best;
            let mut corner_score = best_score;
            if best_score.0 > 9.094947e-13 {
                for direction in 0..2 {
                    let mut heights = [0.0; 3];
                    for column in 0..3 {
                        for at in 0..3 {
                            let index = if direction == 1 {
                                3 * at + column
                            } else {
                                3 * column + at
                            };
                            heights[column] = add(heights[column], samples[index])
                        }
                    }
                    for difference in 0..3 {
                        let slope = match difference {
                            0 => heights[1] - heights[0],
                            1 => mul(0.5, heights[2] - heights[0]),
                            _ => heights[2] - heights[1],
                        };
                        let integration = if direction == 1 { 1 } else { 0 };
                        let orientation_component =
                            if integration == 0 { best.nx } else { best.ny };
                        if orientation_component == 0.0 {
                            continue;
                        }
                        let orientation = if orientation_component > 0.0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let candidate = if integration == 0 {
                            plane_from_fill(rho, orientation, -slope, [1.0, 1.0])
                        } else {
                            plane_from_fill(rho, -slope, orientation, [1.0, 1.0])
                        };
                        let score = fit_score(candidate, &samples);
                        if score.0 < best_score.0 {
                            best = candidate;
                            best_score = score
                        }
                        if score.1 < corner_score.1
                            || (score.1 == corner_score.1 && score.0 < corner_score.0)
                        {
                            corner = candidate;
                            corner_score = score
                        }
                    }
                }
                if best_score.0 > 9.094947e-13 && corner_score.1 <= 9.094947e-13 {
                    best = corner
                }
            }
        }
        fields.interface_normal[2 * cell] = best.nx;
        fields.interface_normal[2 * cell + 1] = best.ny;
        fields.interface_offset[cell] = best.offset;
    }
    Ok(())
}

pub(crate) fn plic_box_fraction(nx: f32, ny: f32, offset: f32, wx: f32, wy: f32) -> f32 {
    box_fraction(nx, ny, offset, wx, wy)
}
pub(crate) fn plic_box_fraction_rect(nx: f32, ny: f32, offset: f32, wx: f64, wy: f64) -> f32 {
    box_fraction_projected(
        (nx.abs() as f64 * wx) as f32,
        (ny.abs() as f64 * wy) as f32,
        offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Cell;
    #[test]
    fn box_fraction_is_symmetric() {
        assert_eq!(box_fraction(1.0, 0.0, 0.0, 2.0, 1.0), 0.5);
        assert_eq!(box_fraction(-1.0, 0.0, 0.0, 2.0, 1.0), 0.5)
    }
    fn cell(id: u32, minimum: [f32; 3], maximum: [f32; 3]) -> Cell {
        Cell {
            id,
            minimum,
            maximum,
            center: [
                (minimum[0] + maximum[0]) * 0.5,
                (minimum[1] + maximum[1]) * 0.5,
                (minimum[2] + maximum[2]) * 0.5,
            ],
            widths: [
                maximum[0] - minimum[0],
                maximum[1] - minimum[1],
                maximum[2] - minimum[2],
            ],
            measure: 1.0,
            ..Default::default()
        }
    }
    #[test]
    fn indexed_owner_preserves_half_open_boundaries() {
        let graph = Graph {
            dimension: 2,
            dimensions: [2.0, 1.0, 1.0],
            cells: vec![
                cell(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                cell(1, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            ],
            ..Default::default()
        };
        graph.initialize_spatial_owner_cache();
        assert_eq!(owner_at(&graph, [0.0, 0.0, -999.0]), Some(0));
        assert_eq!(owner_at(&graph, [1.0, 0.5, 999.0]), Some(1));
        assert_eq!(owner_at(&graph, [2.0, 0.5, 0.0]), None);
        assert_eq!(owner_at(&graph, [f32::NAN, 0.5, 0.0]), None)
    }
    #[test]
    fn fractional_graph_uses_exact_linear_fallback() {
        let graph = Graph {
            dimension: 2,
            dimensions: [2.0, 1.0, 1.0],
            cells: vec![cell(0, [0.25, 0.0, 0.0], [1.25, 1.0, 1.0])],
            ..Default::default()
        };
        assert_eq!(owner_at(&graph, [0.1, 0.5, 0.0]), None);
        assert_eq!(owner_at(&graph, [0.3, 0.5, 0.0]), Some(0));
        assert_eq!(owner_at(&graph, [1.25, 0.5, 0.0]), None)
    }
    #[test]
    fn sparse_three_dimensional_owner_keeps_depth() {
        let graph = Graph {
            dimension: 3,
            dimensions: [64.0, 64.0, 64.0],
            cells: vec![
                cell(0, [8.0, 8.0, 8.0], [16.0, 16.0, 16.0]),
                cell(1, [8.0, 8.0, 24.0], [16.0, 16.0, 32.0]),
            ],
            ..Default::default()
        };
        graph.initialize_spatial_owner_cache();
        assert_eq!(owner_at(&graph, [12.0, 12.0, 12.0]), Some(0));
        assert_eq!(owner_at(&graph, [12.0, 12.0, 28.0]), Some(1));
        assert_eq!(owner_at(&graph, [12.0, 12.0, 20.0]), None);
    }
}
