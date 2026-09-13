//! True 3-D CM12 least-squares interface reconstruction on compiled rows.
use crate::geometry3d::{interface_from_fill, plane_box_fraction, Plane3};
use crate::numerics::{owner_at, physical_row_measure, solid_voxel_at};
use crate::types::{Fields, Graph, ValidationError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const PARALLEL_3D_STAGE_THRESHOLD: usize = 512;
#[inline]
fn add(a: f32, b: f32) -> f32 {
    a + b
}
#[inline]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}
#[inline]
fn div(a: f32, b: f32) -> f32 {
    a / b
}
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    add(add(mul(a[0], b[0]), mul(a[1], b[1])), mul(a[2], b[2]))
}
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        mul(a[1], b[2]) - mul(a[2], b[1]),
        mul(a[2], b[0]) - mul(a[0], b[2]),
        mul(a[0], b[1]) - mul(a[1], b[0]),
    ]
}
fn fill(fields: &Fields, id: usize) -> f32 {
    div(fields.density[id], fields.capacity[id].max(1e-8))
}
fn owner(graph: &Graph, p: [f32; 3]) -> Option<usize> {
    owner_at(graph, p)
}
fn cell_velocity_sample(graph: &Graph, fields: &Fields, p: [f32; 3], spans: [f32; 3]) -> [f32; 3] {
    let q = [
        p[0].clamp(0.5 * spans[0], graph.dimensions[0] - 0.5 * spans[0]),
        p[1].clamp(0.5 * spans[1], graph.dimensions[1] - 0.5 * spans[1]),
        p[2].clamp(0.5 * spans[2], graph.dimensions[2] - 0.5 * spans[2]),
    ];
    let lower = [
        (q[0] / spans[0] - 0.5).floor(),
        (q[1] / spans[1] - 0.5).floor(),
        (q[2] / spans[2] - 0.5).floor(),
    ];
    let t = [
        q[0] / spans[0] - 0.5 - lower[0],
        q[1] / spans[1] - 0.5 - lower[1],
        q[2] / spans[2] - 0.5 - lower[2],
    ];
    let mut out = [0.0; 3];
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let w = mul(
                    mul(
                        if x == 1 { t[0] } else { 1.0 - t[0] },
                        if y == 1 { t[1] } else { 1.0 - t[1] },
                    ),
                    if z == 1 { t[2] } else { 1.0 - t[2] },
                );
                if w == 0.0 {
                    continue;
                }
                let point = [
                    spans[0] * (lower[0] + x as f32 + 0.5),
                    spans[1] * (lower[1] + y as f32 + 0.5),
                    spans[2] * (lower[2] + z as f32 + 0.5),
                ];
                if let Some(id) = owner(graph, point) {
                    for a in 0..3 {
                        out[a] = add(out[a], mul(w, fields.cell_velocity[3 * id + a]))
                    }
                }
            }
        }
    }
    out
}
fn clip_segment(graph: &Graph, fields: &Fields, start: [f32; 3], candidate: [f32; 3]) -> [f32; 3] {
    let solid = |p: [f32; 3]| {
        let lattice = [
            p[0].floor() as i32,
            p[1].floor() as i32,
            p[2].floor() as i32,
        ];
        if solid_voxel_at(graph, lattice[0], lattice[1], lattice[2]) {
            return true;
        }
        owner(graph, p)
            .map(|id| fields.capacity[id] <= 1e-8)
            .unwrap_or(false)
    };
    if solid(start) {
        return start;
    }
    let (mut lo, mut hi, mut found) = (0.0, 1.0, false);
    for i in 1..=8 {
        let t = i as f32 / 8.0;
        let p = [
            start[0] + t * (candidate[0] - start[0]),
            start[1] + t * (candidate[1] - start[1]),
            start[2] + t * (candidate[2] - start[2]),
        ];
        if solid(p) {
            lo = (i - 1) as f32 / 8.0;
            hi = t;
            found = true;
            break;
        }
    }
    if !found {
        return candidate;
    }
    for _ in 0..8 {
        let t = 0.5 * (lo + hi);
        let p = [
            start[0] + t * (candidate[0] - start[0]),
            start[1] + t * (candidate[1] - start[1]),
            start[2] + t * (candidate[2] - start[2]),
        ];
        if solid(p) {
            hi = t
        } else {
            lo = t
        }
    }
    let t = (lo - 1e-4).max(0.0);
    [
        start[0] + t * (candidate[0] - start[0]),
        start[1] + t * (candidate[1] - start[1]),
        start[2] + t * (candidate[2] - start[2]),
    ]
}
pub fn trace_characteristic_3d(
    graph: &Graph,
    fields: &Fields,
    p: [f32; 3],
    dt: f32,
    direction: f32,
) -> [f32; 3] {
    trace_characteristic_at_spans(
        graph,
        fields,
        p,
        dt,
        direction,
        owner(graph, p)
            .map(|id| graph.cells[id].widths)
            .unwrap_or([1.0; 3]),
        false,
    )
}

fn trace_characteristic_at_spans(
    graph: &Graph,
    fields: &Fields,
    p: [f32; 3],
    dt: f32,
    direction: f32,
    spans: [f32; 3],
    span_bounds: bool,
) -> [f32; 3] {
    let initial = cell_velocity_sample(graph, fields, p, spans);
    let speed = add(
        add(mul(initial[0], initial[0]), mul(initial[1], initial[1])),
        mul(initial[2], initial[2]),
    )
    .sqrt();
    let steps = mul(speed, dt).ceil().clamp(1.0, 16.0) as usize;
    let step_dt = dt / steps as f32;
    let mut traced = p;
    let half = if span_bounds {
        [0.5 * spans[0], 0.5 * spans[1], 0.5 * spans[2]]
    } else {
        [0.5; 3]
    };
    for step in 0..steps {
        let first = if step == 0 {
            initial
        } else {
            cell_velocity_sample(graph, fields, traced, spans)
        };
        let raw_mid = [
            (traced[0] + direction * 0.5 * step_dt * first[0])
                .clamp(half[0], graph.dimensions[0] - half[0]),
            (traced[1] + direction * 0.5 * step_dt * first[1])
                .clamp(half[1], graph.dimensions[1] - half[1]),
            (traced[2] + direction * 0.5 * step_dt * first[2])
                .clamp(half[2], graph.dimensions[2] - half[2]),
        ];
        let midpoint = clip_segment(graph, fields, traced, raw_mid);
        let middle = cell_velocity_sample(graph, fields, midpoint, spans);
        let raw = [
            (traced[0] + direction * step_dt * middle[0])
                .clamp(half[0], graph.dimensions[0] - half[0]),
            (traced[1] + direction * step_dt * middle[1])
                .clamp(half[1], graph.dimensions[1] - half[1]),
            (traced[2] + direction * step_dt * middle[2])
                .clamp(half[2], graph.dimensions[2] - half[2]),
        ];
        traced = clip_segment(graph, fields, traced, raw)
    }
    traced
}

fn staggered_liquid_row(fields: &Fields, row: &crate::types::Row) -> bool {
    row.terms
        .iter()
        .any(|term| fields.density[term.cell_id as usize] > 0.5)
}

fn staggered_fluid_row(fields: &Fields, row: &crate::types::Row, source: &[f32]) -> f32 {
    let velocity = source[row.id as usize];
    if row.open_fraction > 1e-6 {
        div(
            velocity - mul(1.0 - row.open_fraction, row.solid_velocity),
            row.open_fraction,
        )
    } else {
        fields
            .face_velocity
            .get(row.id as usize)
            .map(|_| row.solid_velocity)
            .unwrap_or(0.0)
    }
}

fn staggered_cell_sample(
    graph: &Graph,
    fields: &Fields,
    source: &[f32],
    point: [f32; 3],
    axis: usize,
) -> Option<f32> {
    let mut query = point;
    let mut cell = owner(graph, query);
    if cell.is_none() {
        query[axis] -= 1.0;
        cell = owner(graph, query);
    }
    let cell = cell?;
    let c = &graph.cells[cell];
    let lower = c.minimum;
    let upper = c.maximum;
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    let mut values = [0.0; 2];
    let mut weights = [0.0; 2];
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis || !staggered_liquid_row(fields, row) {
            continue;
        }
        if fields.solid_motion_active && row.open_fraction < 1.0 {
            continue;
        }
        let Some(own) = row.terms.iter().find(|term| term.cell_id as usize == cell) else {
            continue;
        };
        let side = usize::from(own.coefficient < 0.0);
        let mut area = 0.0;
        if row.terms.len() == 1 {
            let patch_area = mul(
                own.coefficient.abs(),
                row.static_dual_weight.unwrap_or(row.dual_weight),
            );
            let whole = mul(c.widths[tangents[0]], c.widths[tangents[1]]);
            if patch_area != whole
                || row.center[tangents[0]] != c.center[tangents[0]]
                || row.center[tangents[1]] != c.center[tangents[1]]
                || row.center[axis] != if side == 1 { upper[axis] } else { lower[axis] }
            {
                return None;
            }
            if point[tangents[0]] >= lower[tangents[0]]
                && point[tangents[0]] < upper[tangents[0]]
                && point[tangents[1]] >= lower[tangents[1]]
                && point[tangents[1]] < upper[tangents[1]]
            {
                area = patch_area;
            }
        } else {
            for term in &row.terms {
                if mul(own.coefficient, term.coefficient) >= 0.0 {
                    continue;
                }
                let other = &graph.cells[term.cell_id as usize];
                let patch_lower = [
                    lower[0].max(other.minimum[0]),
                    lower[1].max(other.minimum[1]),
                    lower[2].max(other.minimum[2]),
                ];
                let patch_upper = [
                    upper[0].min(other.maximum[0]),
                    upper[1].min(other.maximum[1]),
                    upper[2].min(other.maximum[2]),
                ];
                if point[tangents[0]] >= patch_lower[tangents[0]]
                    && point[tangents[0]] < patch_upper[tangents[0]]
                    && point[tangents[1]] >= patch_lower[tangents[1]]
                    && point[tangents[1]] < patch_upper[tangents[1]]
                {
                    area = add(
                        area,
                        mul(
                            patch_upper[tangents[0]] - patch_lower[tangents[0]],
                            patch_upper[tangents[1]] - patch_lower[tangents[1]],
                        ),
                    );
                }
            }
        }
        if area > 0.0 {
            values[side] = add(
                values[side],
                mul(area, staggered_fluid_row(fields, row, source)),
            );
            weights[side] = add(weights[side], area);
        }
    }
    let fraction = div(point[axis] - lower[axis], c.widths[axis]).clamp(0.0, 1.0);
    if fraction == 0.0 && weights[0] > 0.0 {
        return Some(div(values[0], weights[0]));
    }
    if fraction == 1.0 && weights[1] > 0.0 {
        return Some(div(values[1], weights[1]));
    }
    if weights[0] <= 0.0 || weights[1] <= 0.0 {
        return None;
    }
    Some(
        mul(1.0 - fraction, div(values[0], weights[0])) + mul(fraction, div(values[1], weights[1])),
    )
}

fn staggered_sample_linear(
    graph: &Graph,
    fields: &Fields,
    source_face_velocity: &[f32],
    p: [f32; 3],
    axis: usize,
    spans: [f32; 3],
) -> f32 {
    let mut offset = [0.5; 3];
    offset[axis] = 0.0;
    let bounded = [
        p[0].clamp(
            offset[0] * spans[0],
            graph.dimensions[0] - offset[0] * spans[0],
        ),
        p[1].clamp(
            offset[1] * spans[1],
            graph.dimensions[1] - offset[1] * spans[1],
        ),
        p[2].clamp(
            offset[2] * spans[2],
            graph.dimensions[2] - offset[2] * spans[2],
        ),
    ];
    let shifted = [
        bounded[0] / spans[0] - offset[0],
        bounded[1] / spans[1] - offset[1],
        bounded[2] / spans[2] - offset[2],
    ];
    let lower = [shifted[0].floor(), shifted[1].floor(), shifted[2].floor()];
    let fraction = [
        shifted[0] - lower[0],
        shifted[1] - lower[1],
        shifted[2] - lower[2],
    ];
    let mut result = 0.0;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let w = mul(
                    mul(
                        if x == 1 {
                            fraction[0]
                        } else {
                            1.0 - fraction[0]
                        },
                        if y == 1 {
                            fraction[1]
                        } else {
                            1.0 - fraction[1]
                        },
                    ),
                    if z == 1 {
                        fraction[2]
                    } else {
                        1.0 - fraction[2]
                    },
                );
                if w == 0.0 {
                    continue;
                }
                let point = [
                    spans[0] * (lower[0] + x as f32 + offset[0]),
                    spans[1] * (lower[1] + y as f32 + offset[1]),
                    spans[2] * (lower[2] + z as f32 + offset[2]),
                ];
                let value = staggered_cell_sample(graph, fields, source_face_velocity, point, axis)
                    .unwrap_or_else(|| cell_velocity_sample(graph, fields, point, spans)[axis]);
                result = add(result, mul(w, value))
            }
        }
    }
    result
}

fn staggered_uniform_node(
    graph: &Graph,
    fields: &Fields,
    source_face_velocity: &[f32],
    point: [f32; 3],
    axis: usize,
    spans: [f32; 3],
) -> Option<f32> {
    let cell = owner(graph, point)?;
    if graph.cells[cell].widths != spans || fields.capacity[cell] != 1.0 {
        return None;
    }
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let mut found = None;
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis || row.center != point {
            continue;
        }
        if row.terms.len() != 2
            || row.open_fraction != 1.0
            || row.static_measure.unwrap_or(row.measure) != mul(spans[u], spans[v])
            || row.distance != spans[axis]
            || !row
                .terms
                .iter()
                .any(|term| fields.density[term.cell_id as usize] > 0.5)
        {
            return None;
        }
        let mut negative = false;
        let mut positive = false;
        for term in &row.terms {
            let endpoint = term.cell_id as usize;
            if graph.cells[endpoint].widths != spans || fields.capacity[endpoint] != 1.0 {
                return None;
            }
            let mut expected = point;
            if term.coefficient < 0.0 {
                expected[axis] -= 0.5 * spans[axis];
                negative = true;
            } else if term.coefficient > 0.0 {
                expected[axis] += 0.5 * spans[axis];
                positive = true;
            } else {
                return None;
            }
            if graph.cells[endpoint].center != expected {
                return None;
            }
        }
        if !negative || !positive || found.is_some() {
            return None;
        }
        found = Some(source_face_velocity[row.id as usize]);
    }
    found
}

fn staggered_cubic_line(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    if t == 0.0 {
        return b;
    }
    if t == 1.0 {
        return c;
    }
    b + t
        * (0.5 * (c - a)
            + t * ((a - b) + 2.0 * (c - b) - 0.5 * (d - b) + t * (1.5 * (b - c) + 0.5 * (d - a))))
}

fn staggered_sample(
    graph: &Graph,
    fields: &Fields,
    source_face_velocity: &[f32],
    p: [f32; 3],
    axis: usize,
    spans: [f32; 3],
) -> f32 {
    let mut offset = [0.5; 3];
    offset[axis] = 0.0;
    let lower_bound = [
        offset[0] * spans[0],
        offset[1] * spans[1],
        offset[2] * spans[2],
    ];
    let upper_bound = [
        graph.dimensions[0] - offset[0] * spans[0],
        graph.dimensions[1] - offset[1] * spans[1],
        graph.dimensions[2] - offset[2] * spans[2],
    ];
    let bounded = [
        p[0].clamp(lower_bound[0], upper_bound[0]),
        p[1].clamp(lower_bound[1], upper_bound[1]),
        p[2].clamp(lower_bound[2], upper_bound[2]),
    ];
    let shifted = [
        bounded[0] / spans[0] - offset[0],
        bounded[1] / spans[1] - offset[1],
        bounded[2] / spans[2] - offset[2],
    ];
    let lower = shifted.map(f32::floor);
    let fraction = [
        shifted[0] - lower[0],
        shifted[1] - lower[1],
        shifted[2] - lower[2],
    ];
    let first = [
        spans[0] * (lower[0] - 1.0 + offset[0]),
        spans[1] * (lower[1] - 1.0 + offset[1]),
        spans[2] * (lower[2] - 1.0 + offset[2]),
    ];
    let last = [
        spans[0] * (lower[0] + 2.0 + offset[0]),
        spans[1] * (lower[1] + 2.0 + offset[1]),
        spans[2] * (lower[2] + 2.0 + offset[2]),
    ];
    let interpolated = fraction.map(|v| v != 0.0);
    if fraction == [0.0; 3]
        || (0..3)
            .any(|a| interpolated[a] && (first[a] < lower_bound[a] || last[a] > upper_bound[a]))
        || (interpolated[axis] && (first[axis] <= 0.0 || last[axis] >= graph.dimensions[axis]))
    {
        return staggered_sample_linear(graph, fields, source_face_velocity, p, axis, spans);
    }
    let counts = interpolated.map(|v| if v { 4 } else { 1 });
    let mut values = [[[0.0; 4]; 4]; 4];
    let mut core_minimum = f32::MAX;
    let mut core_maximum = -f32::MAX;
    for zi in 0..counts[2] {
        let z = if interpolated[2] { zi } else { 1 };
        for yi in 0..counts[1] {
            let y = if interpolated[1] { yi } else { 1 };
            for xi in 0..counts[0] {
                let x = if interpolated[0] { xi } else { 1 };
                let point = [
                    spans[0] * (lower[0] + x as f32 - 1.0 + offset[0]),
                    spans[1] * (lower[1] + y as f32 - 1.0 + offset[1]),
                    spans[2] * (lower[2] + z as f32 - 1.0 + offset[2]),
                ];
                let Some(node) =
                    staggered_uniform_node(graph, fields, source_face_velocity, point, axis, spans)
                else {
                    return staggered_sample_linear(
                        graph,
                        fields,
                        source_face_velocity,
                        p,
                        axis,
                        spans,
                    );
                };
                values[z][y][x] = node;
                let bracket = (x == 1 || (interpolated[0] && x == 2))
                    && (y == 1 || (interpolated[1] && y == 2))
                    && (z == 1 || (interpolated[2] && z == 2));
                if bracket {
                    core_minimum = core_minimum.min(node);
                    core_maximum = core_maximum.max(node);
                }
            }
        }
    }
    let mut lines = [[0.0; 4]; 4];
    for zi in 0..counts[2] {
        let z = if interpolated[2] { zi } else { 1 };
        for yi in 0..counts[1] {
            let y = if interpolated[1] { yi } else { 1 };
            lines[z][y] = staggered_cubic_line(
                values[z][y][0],
                values[z][y][1],
                values[z][y][2],
                values[z][y][3],
                fraction[0],
            );
        }
    }
    let mut planes = [0.0; 4];
    for zi in 0..counts[2] {
        let z = if interpolated[2] { zi } else { 1 };
        planes[z] = staggered_cubic_line(
            lines[z][0],
            lines[z][1],
            lines[z][2],
            lines[z][3],
            fraction[1],
        );
    }
    staggered_cubic_line(planes[0], planes[1], planes[2], planes[3], fraction[2])
        .clamp(core_minimum, core_maximum)
}
pub fn prepare_faces_3d(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
) -> Result<(), ValidationError> {
    if graph.dimension != 3 || fields.cell_velocity.len() != 3 * graph.cells.len() {
        return Err(ValidationError(
            "3-D face preparation fields do not align".into(),
        ));
    }
    let source_face_velocity = fields.face_velocity.clone();
    let prepare = |row: &crate::types::Row| -> f32 {
        if physical_row_measure(row) <= 1e-8 {
            return row.solid_velocity;
        }
        if !row
            .terms
            .iter()
            .any(|t| fields.extension_depth[t.cell_id as usize] != 255)
        {
            return row.solid_velocity;
        }
        let sampling_width = row
            .terms
            .iter()
            .map(|term| {
                graph.cells[term.cell_id as usize]
                    .widths
                    .into_iter()
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(f32::INFINITY, f32::min)
            .max(1.0);
        let mut region_width = sampling_width;
        for term in &row.terms {
            let cell = &graph.cells[term.cell_id as usize];
            if cell.refinement_region_scale.unwrap_or(1.0) > 1.0 {
                region_width =
                    region_width.max(cell.widths.into_iter().fold(f32::INFINITY, f32::min));
            }
        }
        // Hold this row's finest incident width for its complete RK2 trace,
        // matching prepareTransportFaceRow's resident support selection.
        let trace_spans = [region_width; 3];
        let sample_spans = [sampling_width; 3];
        let departure =
            trace_characteristic_at_spans(graph, fields, row.center, dt, -1.0, trace_spans, true);
        let characteristic = staggered_sample(
            graph,
            fields,
            &source_face_velocity,
            departure,
            row.axis as usize,
            sample_spans,
        );
        add(
            mul(row.open_fraction, characteristic),
            mul(1.0 - row.open_fraction, row.solid_velocity),
        )
    };
    #[cfg(feature = "parallel")]
    let prepared: Vec<f32> = if graph.rows.len() >= PARALLEL_3D_STAGE_THRESHOLD {
        graph.rows.par_iter().map(prepare).collect()
    } else {
        graph.rows.iter().map(prepare).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let prepared: Vec<f32> = graph.rows.iter().map(prepare).collect();
    for row in &graph.rows {
        fields.face_velocity[row.id as usize] = prepared[row.id as usize];
    }
    Ok(())
}

pub fn publish_characteristic_clearance_3d<'a>(
    graph: &Graph,
    fields: &'a mut Fields,
    dt: f32,
    source_density: Option<&[f32]>,
    source_gamma: Option<&[f32]>,
) -> Result<&'a [f32], ValidationError> {
    if graph.dimension != 3 {
        return Err(ValidationError("3-D clearance requires dimension=3".into()));
    }
    let density = source_density.unwrap_or(&fields.density);
    let gamma = source_gamma.unwrap_or(&fields.gamma);
    let mut result = vec![0.0; graph.cells.len()];
    for receiver in &graph.cells {
        let id = receiver.id as usize;
        if fields.capacity[id] <= 1e-8 {
            continue;
        }
        let span = receiver.widths.into_iter().fold(f32::INFINITY, f32::min);
        let departure = trace_characteristic_3d(graph, fields, receiver.center, dt, -1.0);
        let bounded = [
            departure[0].clamp(0.5 * span, graph.dimensions[0] - 0.5 * span),
            departure[1].clamp(0.5 * span, graph.dimensions[1] - 0.5 * span),
            departure[2].clamp(0.5 * span, graph.dimensions[2] - 0.5 * span),
        ];
        let shifted = [
            bounded[0] / span - 0.5,
            bounded[1] / span - 0.5,
            bounded[2] / span - 0.5,
        ];
        let lower = [shifted[0].floor(), shifted[1].floor(), shifted[2].floor()];
        let mut valid = true;
        let mut visible = 0.0;
        let mut minimum = f32::INFINITY;
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let t = [
                        shifted[0] - lower[0],
                        shifted[1] - lower[1],
                        shifted[2] - lower[2],
                    ];
                    let weight = mul(
                        mul(
                            if x == 1 { t[0] } else { 1.0 - t[0] },
                            if y == 1 { t[1] } else { 1.0 - t[1] },
                        ),
                        if z == 1 { t[2] } else { 1.0 - t[2] },
                    );
                    visible = add(visible, weight);
                    let point = [
                        span * (lower[0] + x as f32 + 0.5),
                        span * (lower[1] + y as f32 + 0.5),
                        span * (lower[2] + z as f32 + 0.5),
                    ];
                    let Some(cell) = owner(graph, point) else {
                        valid = false;
                        continue;
                    };
                    if fields.capacity[cell] < 0.999999
                        || (density[cell] - 1.0).abs() > 0.005
                        || (gamma[cell] - 1.0).abs() > 0.005
                    {
                        valid = false;
                        continue;
                    }
                    minimum = minimum.min(
                        graph.cells[cell]
                            .widths
                            .into_iter()
                            .fold(f32::INFINITY, f32::min),
                    );
                    for a in 0..3 {
                        low[a] = low[a].min(fields.cell_velocity[3 * cell + a]);
                        high[a] = high[a].max(fields.cell_velocity[3 * cell + a])
                    }
                }
            }
        }
        if !valid || visible < 0.999999 {
            continue;
        }
        let delta = add(
            add(
                mul(high[0] - low[0], high[0] - low[0]),
                mul(high[1] - low[1], high[1] - low[1]),
            ),
            mul(high[2] - low[2], high[2] - low[2]),
        )
        .sqrt();
        if mul(dt, delta) <= mul(0.002, minimum) {
            result[id] = mul(0.02, minimum)
        }
    }
    fields.characteristic_clearance = result;
    Ok(&fields.characteristic_clearance)
}

const FIT_ROUNDOFF: f32 = 9.094_947e-13;

fn certified_uniform_sample(graph: &Graph, fields: &Fields, position: [f32; 3]) -> Option<f32> {
    let id = owner(graph, position)?;
    let cell = &graph.cells[id];
    if cell.widths != [1.0; 3] || cell.center != position || fields.capacity[id] < 0.999999 {
        return None;
    }
    let observed = fill(fields, id);
    (observed >= -9.536_743e-7 && observed <= 1.0 + 9.536_743e-7).then(|| observed.clamp(0.0, 1.0))
}

fn uniform_extrusion_fit_score(
    plane: Plane3,
    samples: &[f32; 9],
    axis_u: usize,
    axis_v: usize,
) -> [f32; 2] {
    let mut full = 0.0;
    let mut sides = [0.0; 4];
    for j in 0..3 {
        for i in 0..3 {
            let mut displacement = [0.0; 3];
            displacement[axis_u] = i as f32 - 1.0;
            displacement[axis_v] = j as f32 - 1.0;
            let difference = shifted_fraction(plane, displacement) - samples[3 * j + i];
            let error = mul(difference, difference);
            full = add(full, error);
            if i <= 1 {
                sides[0] = add(sides[0], error);
            }
            if i >= 1 {
                sides[1] = add(sides[1], error);
            }
            if j <= 1 {
                sides[2] = add(sides[2], error);
            }
            if j >= 1 {
                sides[3] = add(sides[3], error);
            }
        }
    }
    [
        div(full, 9.0),
        div(sides[0].min(sides[1].min(sides[2].min(sides[3]))), 6.0),
    ]
}

/// Literal uniform-stencil branch of geometricResidentFitUniformExtrusion.
fn fit_uniform_extrusion(
    graph: &Graph,
    fields: &Fields,
    id: usize,
    fallback: Plane3,
) -> Option<Plane3> {
    if graph.cells[id].widths != [1.0; 3] {
        return None;
    }
    let magnitude = fallback.normal.map(f32::abs);
    let mut extrusion = 0;
    if magnitude[1] < magnitude[extrusion] {
        extrusion = 1;
    }
    if magnitude[2] < magnitude[extrusion] {
        extrusion = 2;
    }
    if magnitude[extrusion] > 1e-6 {
        return None;
    }
    let axis_u = (extrusion + 1) % 3;
    let axis_v = (extrusion + 2) % 3;
    let centre = graph.cells[id].center;
    let mut samples = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            let mut position = centre;
            position[axis_u] += i as f32 - 1.0;
            position[axis_v] += j as f32 - 1.0;
            let observed = certified_uniform_sample(graph, fields, position)?;
            samples[3 * j + i] = observed;
            for side in 0..2 {
                let mut adjacent = position;
                adjacent[extrusion] += if side == 1 { 1.0 } else { -1.0 };
                let next = certified_uniform_sample(graph, fields, adjacent)?;
                if (next - observed).abs() > 9.536_743e-7 {
                    return None;
                }
            }
        }
    }
    let fill = samples[4];
    let mut best = fallback;
    let mut best_score = uniform_extrusion_fit_score(best, &samples, axis_u, axis_v);
    if best_score[0] <= FIT_ROUNDOFF {
        return Some(best);
    }
    let mut corner = best;
    let mut corner_score = best_score;
    for direction in 0..2 {
        let mut heights = [0.0; 3];
        for (column, height) in heights.iter_mut().enumerate() {
            for at in 0..3 {
                let index = if direction == 1 {
                    3 * at + column
                } else {
                    3 * column + at
                };
                *height = add(*height, samples[index]);
            }
        }
        let integration = if direction == 1 { axis_v } else { axis_u };
        let transverse = if direction == 1 { axis_u } else { axis_v };
        if fallback.normal[integration] == 0.0 {
            continue;
        }
        for difference in 0..3 {
            let slope = match difference {
                0 => heights[1] - heights[0],
                1 => mul(0.5, heights[2] - heights[0]),
                _ => heights[2] - heights[1],
            };
            let mut candidate = [0.0; 3];
            candidate[integration] = if fallback.normal[integration] >= 0.0 {
                1.0
            } else {
                -1.0
            };
            candidate[transverse] = -slope;
            let plane = interface_from_fill(fill, candidate, [1.0; 3]);
            let score = uniform_extrusion_fit_score(plane, &samples, axis_u, axis_v);
            if score[0] < best_score[0] {
                best = plane;
                best_score = score;
            }
            if score[1] < corner_score[1]
                || (score[1] == corner_score[1] && score[0] < corner_score[0])
            {
                corner = plane;
                corner_score = score;
            }
        }
    }
    if best_score[0] <= FIT_ROUNDOFF {
        Some(best)
    } else if corner_score[1] <= FIT_ROUNDOFF {
        Some(corner)
    } else {
        Some(best)
    }
}

fn uniform_samples_3d(graph: &Graph, fields: &Fields, id: usize) -> Option<[f32; 27]> {
    if graph.cells[id].widths != [1.0; 3] {
        return None;
    }
    let centre = graph.cells[id].center;
    let mut samples = [0.0; 27];
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let p = [
                    centre[0] + x as f32 - 1.0,
                    centre[1] + y as f32 - 1.0,
                    centre[2] + z as f32 - 1.0,
                ];
                samples[9 * z + 3 * y + x] = certified_uniform_sample(graph, fields, p)?;
            }
        }
    }
    Some(samples)
}

fn shifted_fraction(plane: Plane3, displacement: [f32; 3]) -> f32 {
    plane_box_fraction(
        plane.normal,
        plane.offset - dot(plane.normal, displacement),
        [1.0; 3],
    )
}

fn volume_fit_score_3d(plane: Plane3, samples: &[f32; 27]) -> f32 {
    let mut error = 0.0;
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let d = [x as f32 - 1.0, y as f32 - 1.0, z as f32 - 1.0];
                let difference = shifted_fraction(plane, d) - samples[9 * z + 3 * y + x];
                error = add(error, mul(difference, difference));
            }
        }
    }
    div(error, 27.0)
}

fn slope_plane(fill: f32, slopes: [f32; 2], axis: usize, orientation: f32) -> Plane3 {
    let mut normal = [0.0; 3];
    normal[axis] = orientation;
    normal[(axis + 1) % 3] = slopes[0];
    normal[(axis + 2) % 3] = slopes[1];
    interface_from_fill(fill, normal, [1.0; 3])
}

fn refine_uniform_plane_3d(samples: &[f32; 27], initial: Plane3, initial_error: f32) -> Plane3 {
    let mut best = initial;
    let mut error = initial_error;
    let magnitude = initial.normal.map(f32::abs);
    let mut axis = 0;
    if magnitude[1] > magnitude[axis] {
        axis = 1
    }
    if magnitude[2] > magnitude[axis] {
        axis = 2
    }
    if magnitude[axis] <= 1e-20 {
        return best;
    }
    let orientation = if initial.normal[axis] >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut slopes = [
        div(initial.normal[(axis + 1) % 3], magnitude[axis]),
        div(initial.normal[(axis + 2) % 3], magnitude[axis]),
    ];
    for _ in 0..8 {
        if error <= FIT_ROUNDOFF {
            break;
        }
        let h = 0.002;
        let up = slope_plane(samples[13], [slopes[0] + h, slopes[1]], axis, orientation);
        let um = slope_plane(samples[13], [slopes[0] - h, slopes[1]], axis, orientation);
        let vp = slope_plane(samples[13], [slopes[0], slopes[1] + h], axis, orientation);
        let vm = slope_plane(samples[13], [slopes[0], slopes[1] - h], axis, orientation);
        let mut hessian = [0.0; 3];
        let mut rhs = [0.0; 2];
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let i = 9 * z + 3 * y + x;
                    let d = [x as f32 - 1.0, y as f32 - 1.0, z as f32 - 1.0];
                    let residual = shifted_fraction(best, d) - samples[i];
                    let derivative = [
                        div(shifted_fraction(up, d) - shifted_fraction(um, d), 2.0 * h),
                        div(shifted_fraction(vp, d) - shifted_fraction(vm, d), 2.0 * h),
                    ];
                    hessian[0] = add(hessian[0], mul(derivative[0], derivative[0]));
                    hessian[1] = add(hessian[1], mul(derivative[0], derivative[1]));
                    hessian[2] = add(hessian[2], mul(derivative[1], derivative[1]));
                    rhs[0] = add(rhs[0], mul(derivative[0], residual));
                    rhs[1] = add(rhs[1], mul(derivative[1], residual));
                }
            }
        }
        let determinant = mul(hessian[0], hessian[2]) - mul(hessian[1], hessian[1]);
        if !(determinant > 1e-12 * mul(hessian[0], hessian[2]).max(1.0)) {
            break;
        }
        let mut step = [
            div(
                mul(hessian[2], rhs[0]) - mul(hessian[1], rhs[1]),
                determinant,
            ),
            div(
                mul(hessian[0], rhs[1]) - mul(hessian[1], rhs[0]),
                determinant,
            ),
        ];
        let length = add(mul(step[0], step[0]), mul(step[1], step[1])).sqrt();
        let scale = div(0.5, length.max(1e-20)).min(1.0);
        step[0] = mul(step[0], scale);
        step[1] = mul(step[1], scale);
        let mut improved = false;
        for _ in 0..4 {
            let proposed = [slopes[0] - step[0], slopes[1] - step[1]];
            let plane = slope_plane(samples[13], proposed, axis, orientation);
            let next_error = volume_fit_score_3d(plane, samples);
            if next_error < error {
                best = plane;
                error = next_error;
                slopes = proposed;
                improved = true;
                break;
            }
            step[0] = mul(step[0], 0.5);
            step[1] = mul(step[1], 0.5);
        }
        if !improved {
            break;
        }
    }
    best
}

fn fit_uniform_plane_3d(graph: &Graph, fields: &Fields, id: usize, fallback: Plane3) -> Plane3 {
    let Some(samples) = uniform_samples_3d(graph, fields, id) else {
        return fallback;
    };
    let mut best = fallback;
    let mut best_error = volume_fit_score_3d(best, &samples);
    if best_error <= FIT_ROUNDOFF {
        return best;
    }
    let centre = graph.cells[id].center;
    for integration in 0..3 {
        if fallback.normal[integration] == 0.0 {
            continue;
        }
        let axis_u = (integration + 1) % 3;
        let axis_v = (integration + 2) % 3;
        let mut heights = [0.0; 4];
        let mut supported = true;
        for (column, height) in heights.iter_mut().enumerate() {
            let transverse = if column >= 2 { axis_v } else { axis_u };
            let mut position = centre;
            position[transverse] += if column & 1 != 0 { 1.0 } else { -1.0 };
            for along in 0..7 {
                let mut query = position;
                query[integration] += along as f32 - 3.0;
                let Some(observed) = certified_uniform_sample(graph, fields, query) else {
                    supported = false;
                    break;
                };
                *height = add(*height, observed);
            }
            if !supported {
                break;
            }
        }
        if !supported {
            continue;
        }
        let mut candidate = [0.0; 3];
        candidate[integration] = if fallback.normal[integration] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        candidate[axis_u] = mul(-0.5, heights[1] - heights[0]);
        candidate[axis_v] = mul(-0.5, heights[3] - heights[2]);
        let plane = interface_from_fill(samples[13], candidate, [1.0; 3]);
        let error = volume_fit_score_3d(plane, &samples);
        if error < best_error {
            best = plane;
            best_error = error;
        }
        if best_error <= FIT_ROUNDOFF {
            return best;
        }
    }
    refine_uniform_plane_3d(&samples, best, best_error)
}

fn opposite_subface_cell(graph: &Graph, cell: usize, at: usize) -> Option<usize> {
    let incidence = graph.subface_incidences.get(cell)?.get(at)?;
    let face = graph.subfaces.get(incidence.subface_id as usize)?;
    let other = if incidence.orientation < 0 {
        face.positive_cell
    } else {
        face.negative_cell
    };
    (other >= 0 && other as usize != cell).then_some(other as usize)
}

fn same_projection(graph: &Graph, a: usize, b: usize, extrusion: usize) -> bool {
    (0..3).all(|axis| {
        axis == extrusion
            || (graph.cells[a].center[axis] == graph.cells[b].center[axis]
                && graph.cells[a].widths[axis] == graph.cells[b].widths[axis])
    })
}

fn adaptive_extrusion_certified(
    graph: &Graph,
    fields: &Fields,
    cell: usize,
    extrusion: usize,
) -> bool {
    let centre_fill = fill(fields, cell);
    let centre = &graph.cells[cell];
    let Some(incidences) = graph.subface_incidences.get(cell) else {
        return false;
    };
    for (at, incidence) in incidences.iter().enumerate() {
        let face = &graph.subfaces[incidence.subface_id as usize];
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 1e-8 {
            continue;
        }
        let Some(other) = opposite_subface_cell(graph, cell, at) else {
            continue;
        };
        if fields.capacity[other] < 0.999999 {
            return false;
        }
        let observed = fill(fields, other);
        if row.axis as usize == extrusion {
            if (observed - centre_fill).abs() > 9.536_743e-7 {
                return false;
            }
            continue;
        }
        let leader = !(0..at).any(|prior| {
            let prior_face = &graph.subfaces[incidences[prior].subface_id as usize];
            prior_face.row_id == face.row_id
                && opposite_subface_cell(graph, cell, prior)
                    .is_some_and(|candidate| same_projection(graph, candidate, other, extrusion))
        });
        if !leader {
            continue;
        }
        let lower = centre.center[extrusion] - 0.5 * centre.widths[extrusion];
        let upper = centre.center[extrusion] + 0.5 * centre.widths[extrusion];
        let mut coverage = 0.0;
        for (candidate_at, candidate_incidence) in incidences.iter().enumerate() {
            let candidate_face = &graph.subfaces[candidate_incidence.subface_id as usize];
            if candidate_face.row_id != face.row_id {
                continue;
            }
            let Some(member) = opposite_subface_cell(graph, cell, candidate_at) else {
                continue;
            };
            if !same_projection(graph, member, other, extrusion) {
                continue;
            }
            if fields.capacity[member] < 0.999999
                || (fill(fields, member) - observed).abs() > 9.536_743e-7
            {
                return false;
            }
            let mc = graph.cells[member].center[extrusion];
            let mw = graph.cells[member].widths[extrusion];
            coverage = add(
                coverage,
                (upper.min(mc + 0.5 * mw) - lower.max(mc - 0.5 * mw)).max(0.0),
            );
        }
        if coverage < centre.widths[extrusion] - 9.536_743e-7 {
            return false;
        }
    }
    true
}

fn projected_gradient(graph: &Graph, fields: &Fields, cell: usize, extrusion: usize) -> [f32; 3] {
    let axis_u = (extrusion + 1) % 3;
    let axis_v = (extrusion + 2) % 3;
    let centre = &graph.cells[cell];
    let centre_fill = fill(fields, cell);
    let incidences = &graph.subface_incidences[cell];
    let (mut m00, mut m01, mut m11, mut b0, mut b1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (at, incidence) in incidences.iter().enumerate() {
        let face = &graph.subfaces[incidence.subface_id as usize];
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 1e-8 || row.axis as usize == extrusion {
            continue;
        }
        let Some(other) = opposite_subface_cell(graph, cell, at) else {
            continue;
        };
        if fields.capacity[other] < 0.999999 {
            continue;
        }
        let leader = !(0..at).any(|prior| {
            let prior_face = &graph.subfaces[incidences[prior].subface_id as usize];
            prior_face.row_id == face.row_id
                && opposite_subface_cell(graph, cell, prior)
                    .is_some_and(|candidate| same_projection(graph, candidate, other, extrusion))
        });
        if !leader {
            continue;
        }
        let mut delta = [
            graph.cells[other].center[0] - centre.center[0],
            graph.cells[other].center[1] - centre.center[1],
            graph.cells[other].center[2] - centre.center[2],
        ];
        delta[extrusion] = 0.0;
        let remaining = 3 - row.axis as usize - extrusion;
        let lower = centre.center[remaining] - 0.5 * centre.widths[remaining];
        let upper = centre.center[remaining] + 0.5 * centre.widths[remaining];
        let oc = graph.cells[other].center[remaining];
        let ow = graph.cells[other].widths[remaining];
        let overlap = (upper.min(oc + 0.5 * ow) - lower.max(oc - 0.5 * ow)).max(0.0);
        let weight = div(overlap, dot(delta, delta).max(1e-12));
        let difference = fill(fields, other) - centre_fill;
        let du = delta[axis_u];
        let dv = delta[axis_v];
        m00 = add(m00, mul(mul(weight, du), du));
        m01 = add(m01, mul(mul(weight, du), dv));
        m11 = add(m11, mul(mul(weight, dv), dv));
        b0 = add(b0, mul(mul(weight, du), difference));
        b1 = add(b1, mul(mul(weight, dv), difference));
    }
    let determinant = mul(m00, m11) - mul(m01, m01);
    let scale = m00.max(m11);
    let mut result = [0.0; 3];
    if scale > 1e-12 && determinant.abs() > 1e-7 * scale * scale {
        result[axis_u] = div(mul(m11, b0) - mul(m01, b1), determinant);
        result[axis_v] = div(mul(-m01, b0) + mul(m00, b1), determinant);
    }
    result
}

fn reconstruct_interface_cell_3d(graph: &Graph, fields: &Fields, id: usize) -> Plane3 {
    let rho = fill(fields, id);
    if fields.capacity[id] < 0.999999 || !(rho > 0.0 && rho < 1.0) {
        return Plane3::default();
    }
    let cell = &graph.cells[id];
    let mut neighbours: Vec<(usize, f32)> = Vec::new();
    if let Some(incidences) = graph.subface_incidences.get(id) {
        for incidence in incidences {
            let face = &graph.subfaces[incidence.subface_id as usize];
            let row = &graph.rows[face.row_id as usize];
            if row.open_fraction <= 1e-8 {
                continue;
            }
            let other = if incidence.orientation < 0 {
                face.positive_cell
            } else {
                face.negative_cell
            };
            if other >= 0 && other as usize != id {
                neighbours.push((other as usize, face.measure));
            }
        }
    }
    if neighbours.is_empty() {
        for &row_id in &graph.incidences[id] {
            let row = &graph.rows[row_id as usize];
            let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == id) else {
                continue;
            };
            if physical_row_measure(row) <= 1e-8 {
                continue;
            }
            for term in &row.terms {
                if own.coefficient * term.coefficient < 0.0 {
                    neighbours.push((term.cell_id as usize, row.measure));
                }
            }
        }
    }
    let mut m0 = [0.0; 3];
    let mut m1 = [0.0; 3];
    let mut m2 = [0.0; 3];
    let mut rhs = [0.0; 3];
    for (other_id, area) in neighbours {
        if fields.capacity[other_id] < 0.999999 {
            continue;
        }
        let other = &graph.cells[other_id];
        let delta = [
            other.center[0] - cell.center[0],
            other.center[1] - cell.center[1],
            other.center[2] - cell.center[2],
        ];
        let weight = div(area, dot(delta, delta).max(1e-12));
        let difference = fill(fields, other_id) - rho;
        for a in 0..3 {
            m0[a] = add(m0[a], mul(mul(weight, delta[0]), delta[a]));
            m1[a] = add(m1[a], mul(mul(weight, delta[1]), delta[a]));
            m2[a] = add(m2[a], mul(mul(weight, delta[2]), delta[a]));
            rhs[a] = add(rhs[a], mul(mul(weight, delta[a]), difference));
        }
    }
    let determinant = dot(m0, cross(m1, m2));
    let scale = m0[0].max(m1[1].max(m2[2]));
    if !(scale > 1e-12 && determinant.abs() > 1e-7 * scale * scale * scale) {
        return Plane3::default();
    }
    let c12 = cross(m1, m2);
    let c20 = cross(m2, m0);
    let c01 = cross(m0, m1);
    let mut gradient = [
        div(
            add(
                add(mul(rhs[0], c12[0]), mul(rhs[1], c20[0])),
                mul(rhs[2], c01[0]),
            ),
            determinant,
        ),
        div(
            add(
                add(mul(rhs[0], c12[1]), mul(rhs[1], c20[1])),
                mul(rhs[2], c01[1]),
            ),
            determinant,
        ),
        div(
            add(
                add(mul(rhs[0], c12[2]), mul(rhs[1], c20[2])),
                mul(rhs[2], c01[2]),
            ),
            determinant,
        ),
    ];
    let magnitude = gradient.map(f32::abs);
    let mut extrusion = 0;
    if magnitude[1] < magnitude[extrusion] {
        extrusion = 1;
    }
    if magnitude[2] < magnitude[extrusion] {
        extrusion = 2;
    }
    if adaptive_extrusion_certified(graph, fields, id, extrusion) {
        let projected = projected_gradient(graph, fields, id, extrusion);
        if dot(projected, projected) > 0.0 {
            gradient = projected;
        }
    }
    if mul(dot(gradient, gradient), dot(cell.widths, cell.widths)) <= 1e-12 {
        return Plane3::default();
    }
    let fallback = interface_from_fill(rho, gradient.map(|v| -v), cell.widths);
    fit_uniform_extrusion(graph, fields, id, fallback)
        .unwrap_or_else(|| fit_uniform_plane_3d(graph, fields, id, fallback))
}

/// Resident CM12 reconstruction. The initial compact-face least-squares
/// accumulation and uniform 3-D volume fit preserve the WGSL loop order.
pub fn reconstruct_interfaces_3d(
    graph: &Graph,
    fields: &mut Fields,
) -> Result<(), ValidationError> {
    if graph.dimension != 3 {
        return Err(ValidationError(
            "3-D reconstruction requires dimension=3".into(),
        ));
    }
    let n = graph.cells.len();
    #[cfg(feature = "parallel")]
    let planes: Vec<Plane3> = if n >= PARALLEL_3D_STAGE_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|id| reconstruct_interface_cell_3d(graph, fields, id))
            .collect()
    } else {
        (0..n)
            .map(|id| reconstruct_interface_cell_3d(graph, fields, id))
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let planes: Vec<Plane3> = (0..n)
        .map(|id| reconstruct_interface_cell_3d(graph, fields, id))
        .collect();
    fields.interface_normal.resize(3 * n, 0.0);
    fields.interface_offset.resize(n, 0.0);
    for (id, plane) in planes.into_iter().enumerate() {
        fields.interface_normal[3 * id..3 * id + 3].copy_from_slice(&plane.normal);
        fields.interface_offset[id] = plane.offset;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::numerics::{prepare_faces, project_velocity, reconstruct_interfaces};
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};
    use crate::transport::transport_volume;

    fn planar_fixture() -> (Graph, Fields, [f32; 3]) {
        let compiled = compile_topology::<3>(TopologySeed {
            dimensions: [8, 8, 8],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0; 3],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            }],
        })
        .unwrap();
        let mut graph = compiled.graph;
        for row in &mut graph.rows {
            if row.kind == crate::types::RowKind::ClosedWorld {
                row.open_fraction = 0.0;
            }
        }
        for face in &mut graph.subfaces {
            face.aperture = graph.rows[face.row_id as usize].open_fraction;
        }
        graph.initialize_spatial_owner_cache();
        let normal = [0.36, 0.48, 0.8];
        let global_offset = dot(normal, [4.0; 3]);
        let n = graph.cells.len();
        let r = graph.rows.len();
        let f = graph.subfaces.len();
        let density = graph
            .cells
            .iter()
            .map(|cell| {
                plane_box_fraction(
                    normal,
                    global_offset - dot(normal, cell.center),
                    cell.widths,
                )
            })
            .collect();
        let mut cell_velocity = vec![0.0; 3 * n];
        for cell in &graph.cells {
            let i = cell.id as usize;
            // A smooth, nontrivial solenoidal field. Projection enforces the
            // six closed vessel walls before the geometric transport step.
            cell_velocity[3 * i] = 0.025 * (cell.center[1] - 4.0);
            cell_velocity[3 * i + 1] = -0.025 * (cell.center[0] - 4.0);
            cell_velocity[3 * i + 2] = 0.01 * (cell.center[0] - cell.center[1]);
        }
        let fields = Fields {
            density,
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            capacity_before: vec![1.0; n],
            capacity_after: vec![1.0; n],
            frame_dt: 0.2,
            cell_velocity,
            face_velocity: vec![0.0; r],
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![0.0; n],
            pressure_member: vec![0; n],
            pressure_row_member: vec![0; r],
            extension_depth: vec![0; n],
            interface_normal: vec![0.0; 3 * n],
            interface_offset: vec![0.0; n],
            low_flux: vec![0.0; f],
            high_flux: vec![0.0; f],
            limited_flux: vec![0.0; f],
            characteristic_clearance: vec![0.0; n],
            ..Default::default()
        };
        (graph, fields, normal)
    }

    #[test]
    fn wgsl_uniform_volume_fit_recovers_rotated_plane() {
        let (graph, mut fields, expected) = planar_fixture();
        reconstruct_interfaces(&graph, &mut fields).unwrap();
        let mut checked = 0;
        for cell in &graph.cells {
            let i = cell.id as usize;
            if cell.center.iter().all(|&v| v >= 3.5 && v <= 4.5)
                && fields.density[i] > 0.0
                && fields.density[i] < 1.0
            {
                let actual = [
                    fields.interface_normal[3 * i],
                    fields.interface_normal[3 * i + 1],
                    fields.interface_normal[3 * i + 2],
                ];
                assert!(dot(actual, expected) > 0.99999, "cell {i}: {actual:?}");
                let reconstructed =
                    plane_box_fraction(actual, fields.interface_offset[i], cell.widths);
                assert!((reconstructed - fields.density[i]).abs() <= 2.0 * f32::EPSILON);
                checked += 1;
            }
        }
        assert!(checked >= 2);
    }

    #[test]
    fn wgsl_elvira_extrusion_recovers_reflection_symmetric_plane() {
        let (graph, mut fields, _) = planar_fixture();
        let expected = [0.6, 0.8, 0.0];
        let global_offset = dot(expected, [4.0; 3]);
        for cell in &graph.cells {
            fields.density[cell.id as usize] = plane_box_fraction(
                expected,
                global_offset - dot(expected, cell.center),
                cell.widths,
            );
        }
        reconstruct_interfaces(&graph, &mut fields).unwrap();
        let mut checked = 0;
        for cell in &graph.cells {
            let i = cell.id as usize;
            if cell.center.iter().all(|&v| v >= 2.5 && v <= 5.5)
                && fields.density[i] > 0.0
                && fields.density[i] < 1.0
            {
                let actual = [
                    fields.interface_normal[3 * i],
                    fields.interface_normal[3 * i + 1],
                    fields.interface_normal[3 * i + 2],
                ];
                assert!(dot(actual, expected) > 0.99999, "cell {i}: {actual:?}");
                assert_eq!(actual[2].abs().to_bits(), 0.0f32.to_bits());
                checked += 1;
            }
        }
        assert!(checked >= 3);
    }

    #[test]
    fn resident_characteristic_preserves_uniform_flow_and_clips_q8_solid() {
        let (mut graph, mut fields, _) = planar_fixture();
        for velocity in fields.cell_velocity.chunks_exact_mut(3) {
            velocity.copy_from_slice(&[0.25, -0.125, 0.5]);
        }
        let traced = trace_characteristic_3d(&graph, &fields, [4.0; 3], 0.4, 1.0);
        assert_eq!(traced, [4.1, 3.95, 4.2]);

        fields
            .cell_velocity
            .chunks_exact_mut(3)
            .for_each(|velocity| {
                velocity.copy_from_slice(&[2.0, 0.0, 0.0]);
            });
        graph.solid_voxel_fraction = vec![0.0; 8 * 8 * 8];
        graph.solid_voxel_fraction[4 + 8 * (3 + 8 * 3)] = 1.0;
        let clipped = trace_characteristic_3d(&graph, &fields, [3.5; 3], 1.0, 1.0);
        assert!(clipped[0] >= 3.99 && clipped[0] < 4.0, "{clipped:?}");
        assert_eq!(clipped[1], 3.5);
        assert_eq!(clipped[2], 3.5);
    }

    #[test]
    fn sphere_vof_fixture_reconstructs_outward_normals() {
        let (graph, mut fields, _) = planar_fixture();
        let centre = [4.0; 3];
        let radius2 = 2.35_f32 * 2.35;
        const Q: usize = 16;
        for cell in &graph.cells {
            let mut inside = 0_u32;
            for z in 0..Q {
                for y in 0..Q {
                    for x in 0..Q {
                        let p = [
                            cell.minimum[0] + (x as f32 + 0.5) * cell.widths[0] / Q as f32,
                            cell.minimum[1] + (y as f32 + 0.5) * cell.widths[1] / Q as f32,
                            cell.minimum[2] + (z as f32 + 0.5) * cell.widths[2] / Q as f32,
                        ];
                        let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                        inside += u32::from(dot(d, d) <= radius2);
                    }
                }
            }
            fields.density[cell.id as usize] = inside as f32 / (Q * Q * Q) as f32;
        }
        reconstruct_interfaces(&graph, &mut fields).unwrap();
        let mut minimum_alignment = 1.0_f32;
        let mut sum_alignment = 0.0_f32;
        let mut count = 0;
        for cell in &graph.cells {
            let i = cell.id as usize;
            if fields.density[i] <= 0.05 || fields.density[i] >= 0.95 {
                continue;
            }
            let radial = [
                cell.center[0] - centre[0],
                cell.center[1] - centre[1],
                cell.center[2] - centre[2],
            ];
            let length = dot(radial, radial).sqrt();
            let expected = radial.map(|v| v / length);
            let actual = [
                fields.interface_normal[3 * i],
                fields.interface_normal[3 * i + 1],
                fields.interface_normal[3 * i + 2],
            ];
            let alignment = dot(actual, expected);
            minimum_alignment = minimum_alignment.min(alignment);
            sum_alignment += alignment;
            count += 1;
        }
        assert!(count >= 24);
        assert!(minimum_alignment > 0.9, "minimum {minimum_alignment}");
        assert!(sum_alignment / count as f32 > 0.98);
    }

    #[test]
    fn compiled_3d_pressure_characteristic_transport_frame_is_conservative() {
        let (graph, mut fields, _) = planar_fixture();
        let before: f64 = graph
            .cells
            .iter()
            .map(|cell| (fields.density[cell.id as usize] * cell.measure) as f64)
            .sum();
        prepare_faces(&graph, &mut fields, 0.2).unwrap();
        reconstruct_interfaces(&graph, &mut fields).unwrap();
        let pressure = project_velocity(&graph, &mut fields, 160, 1e-6).unwrap();
        assert!(pressure.converged, "{pressure:?}");
        assert!(pressure.residual.is_finite());
        let (steps, _) = transport_volume(&graph, &mut fields, 0.2).unwrap();
        assert!(steps >= 1);
        assert!(fields.fault.is_none(), "{:?}", fields.fault);
        let after: f64 = graph
            .cells
            .iter()
            .map(|cell| (fields.density[cell.id as usize] * cell.measure) as f64)
            .sum();
        assert!((after - before).abs() <= 2e-7, "{before} -> {after}");
        assert!(fields
            .density
            .iter()
            .all(|&value| value >= -2e-6 && value <= 1.0 + 2e-6));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_reconstruction_and_face_preparation_are_bitwise_stable() {
        let (graph, seed, _) = planar_fixture();
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut fields = seed.clone();
            pool.install(|| {
                reconstruct_interfaces_3d(&graph, &mut fields).unwrap();
                prepare_faces_3d(&graph, &mut fields, 0.2).unwrap();
            });
            (
                fields
                    .interface_normal
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                fields
                    .interface_offset
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                fields
                    .face_velocity
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            )
        };
        let reference = run(1);
        for threads in [2, 4, 8] {
            assert_eq!(run(threads), reference, "thread count {threads}");
        }
    }
}
