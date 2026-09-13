//! Conservative physical-subface transport and synchronized low/FCT limiting.

use crate::kernels::{add, div, mul};
use crate::numerics::{plic_box_fraction_rect, reconstruct_interfaces};
use crate::types::{Fields, Graph, NumericalFault, SubfaceIncidence, ValidationError};
use serde::{Deserialize, Serialize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const VOLUME_ROUNDOFF_RATIO: f32 = 9.536_743e-7;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransportMicrostepReceipt {
    pub interface_normal: Vec<f32>,
    pub interface_offset: Vec<f32>,
    pub sweep: Vec<f32>,
    pub initial_low_flux: Vec<f32>,
    pub high_flux: Vec<f32>,
    pub low_flux: Vec<f32>,
    pub low_volume: Vec<f32>,
    pub positive_budget: Vec<f32>,
    pub negative_budget: Vec<f32>,
    pub increase: Vec<f32>,
    pub decrease: Vec<f32>,
    pub limited_flux: Vec<f32>,
    pub next_volume: Vec<f32>,
}

struct TransportIncidences<'a> {
    compiled: Option<&'a [Vec<SubfaceIncidence>]>,
    fallback: Vec<Vec<SubfaceIncidence>>,
}

impl<'a> TransportIncidences<'a> {
    fn new(graph: &'a Graph) -> Self {
        if graph.subface_incidences.len() == graph.cells.len() {
            let mut fallback = vec![Vec::new(); graph.cells.len()];
            for face in &graph.subfaces {
                if face.negative_cell >= 0
                    && graph.subface_incidences[face.negative_cell as usize].is_empty()
                {
                    fallback[face.negative_cell as usize].push(SubfaceIncidence {
                        subface_id: face.id,
                        orientation: -1,
                    });
                }
                if face.positive_cell >= 0
                    && graph.subface_incidences[face.positive_cell as usize].is_empty()
                {
                    fallback[face.positive_cell as usize].push(SubfaceIncidence {
                        subface_id: face.id,
                        orientation: 1,
                    });
                }
            }
            return Self {
                compiled: Some(&graph.subface_incidences),
                fallback,
            };
        }
        let mut fallback = vec![Vec::new(); graph.cells.len()];
        for face in &graph.subfaces {
            if face.negative_cell >= 0 {
                fallback[face.negative_cell as usize].push(SubfaceIncidence {
                    subface_id: face.id,
                    orientation: -1,
                });
            }
            if face.positive_cell >= 0 {
                fallback[face.positive_cell as usize].push(SubfaceIncidence {
                    subface_id: face.id,
                    orientation: 1,
                });
            }
        }
        Self {
            compiled: None,
            fallback,
        }
    }

    #[inline]
    fn for_cell(&self, cell: usize) -> &[SubfaceIncidence] {
        if let Some(compiled) = self.compiled {
            if compiled[cell].is_empty() {
                &self.fallback[cell]
            } else {
                &compiled[cell]
            }
        } else {
            &self.fallback[cell]
        }
    }
}
#[inline]
fn delta(flux: f32, orientation: i8) -> f32 {
    if orientation < 0 {
        -flux
    } else {
        flux
    }
}

#[inline]
pub(crate) fn physical_subface_rate(graph: &Graph, fields: &Fields, face: usize) -> f32 {
    if fields.subface_compatibility_rate.len() == graph.subfaces.len() {
        return fields.subface_compatibility_rate[face] as f32;
    }
    let subface = &graph.subfaces[face];
    let row = &graph.rows[subface.row_id as usize];
    let base_velocity =
        fields.face_velocity[row.id as usize] - mul(1.0 - row.open_fraction, row.solid_velocity);
    let correction = fields
        .subface_velocity_correction
        .get(face)
        .copied()
        .unwrap_or(0.0);
    mul(subface.measure, base_velocity - correction)
}

#[inline]
fn physical_subface_sweep(graph: &Graph, fields: &Fields, face: usize, dt: f32) -> f32 {
    if fields.subface_compatibility_rate.len() == graph.subfaces.len() {
        return (fields.subface_compatibility_rate[face] * dt as f64) as f32;
    }
    mul(physical_subface_rate(graph, fields, face), dt)
}
#[inline]
fn roundoff(capacity: f32) -> f32 {
    mul(VOLUME_ROUNDOFF_RATIO, capacity)
}
#[inline]
fn valid(volume: f32, capacity: f32) -> bool {
    let margin = roundoff(capacity);
    capacity >= 0.0 && capacity.is_finite() && volume >= -margin && volume <= add(capacity, margin)
}
fn capacity_at(
    graph: &Graph,
    fields: &Fields,
    cell: usize,
    numerator: usize,
    denominator: usize,
) -> f32 {
    let measure = graph.cells[cell].measure;
    if !fields.solid_motion_active {
        return mul(fields.capacity[cell], measure);
    }
    let before = mul(
        fields
            .capacity_before
            .get(cell)
            .copied()
            .unwrap_or(fields.capacity[cell]),
        measure,
    );
    let after = mul(
        fields
            .capacity_after
            .get(cell)
            .copied()
            .unwrap_or(fields.capacity[cell]),
        measure,
    );
    if numerator == 0 {
        before
    } else if numerator >= denominator {
        after
    } else {
        add(
            before,
            mul(
                after - before,
                div(numerator as f32, denominator.max(1) as f32),
            ),
        )
    }
}
fn starting_volume(volumes: &[f32], fields: &Fields, cell: usize, dt: f32) -> f32 {
    add(
        volumes[cell],
        mul(dt, fields.source_rate.get(cell).copied().unwrap_or(0.0)),
    )
}
#[inline]
fn receiver(face: &crate::types::Subface, flux: f32) -> i32 {
    if flux >= 0.0 {
        face.positive_cell
    } else {
        face.negative_cell
    }
}
#[inline]
fn donor(face: &crate::types::Subface, sweep: f32) -> i32 {
    if sweep >= 0.0 {
        face.negative_cell
    } else {
        face.positive_cell
    }
}
fn lower_positive(value: f32) -> f32 {
    f32::from_bits(value.to_bits().saturating_sub(1))
}
fn fault(fields: &mut Fields, stage: &str, index: usize, observed: f32, expected: f32) {
    fields.fault = Some(NumericalFault {
        stage: stage.into(),
        index: index as u32,
        observed,
        expected,
    })
}

fn high_flux(
    graph: &Graph,
    fields: &Fields,
    face: &crate::types::Subface,
    sweep: f32,
    low: f32,
) -> f32 {
    if graph.dimension == 3 {
        return crate::transport3d::geometric_high_flux_3d(graph, fields, face, sweep, low)
            .unwrap_or(f32::NAN);
    }
    if sweep == 0.0 {
        return 0.0;
    }
    let donor_id = donor(face, sweep);
    if donor_id < 0 {
        return 0.0;
    }
    let id = donor_id as usize;
    let cell = &graph.cells[id];
    let capacity = mul(fields.capacity[id], cell.measure);
    let observed = mul(fields.density[id], cell.measure).clamp(0.0, capacity.max(0.0));
    if capacity <= 0.0 || observed == 0.0 {
        return 0.0;
    }
    if observed == capacity {
        return sweep;
    }
    if graph.dimension != 2 {
        return low;
    }
    let nx = fields.interface_normal[2 * id];
    let ny = fields.interface_normal[2 * id + 1];
    if nx == 0.0 && ny == 0.0 {
        return low;
    }
    if face.aperture <= 1e-8 {
        return low;
    }
    let travel = sweep.abs() as f64 / (face.measure as f64 * face.aperture as f64).max(1e-20);
    if travel > cell.widths[face.axis as usize] as f64 {
        return f32::NAN;
    }
    let mut minimum = [-0.5 * cell.widths[0] as f64, -0.5 * cell.widths[1] as f64];
    let mut maximum = [0.5 * cell.widths[0] as f64, 0.5 * cell.widths[1] as f64];
    if face.negative_cell >= 0 && face.positive_cell >= 0 {
        let other_id = if donor_id == face.negative_cell {
            face.positive_cell
        } else {
            face.negative_cell
        } as usize;
        let other = &graph.cells[other_id];
        for a in 0..2 {
            minimum[a] = minimum[a].max(other.minimum[a] as f64 - cell.center[a] as f64);
            maximum[a] = maximum[a].min(other.maximum[a] as f64 - cell.center[a] as f64);
        }
    }
    let axis = face.axis as usize;
    let boundary = face.center[axis] as f64 - cell.center[axis] as f64;
    if sweep > 0.0 {
        minimum[axis] = boundary - travel;
        maximum[axis] = boundary
    } else {
        minimum[axis] = boundary;
        maximum[axis] = boundary + travel
    }
    let widths = [maximum[0] - minimum[0], maximum[1] - minimum[1]];
    if widths[0] < 0.0 || widths[1] < 0.0 {
        return f32::NAN;
    }
    let centre = [
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
    ];
    let offset = (fields.interface_offset[id] as f64
        - add(mul(nx, centre[0] as f32), mul(ny, centre[1] as f32)) as f64) as f32;
    mul(
        sweep,
        plic_box_fraction_rect(nx, ny, offset, widths[0], widths[1]),
    )
}

#[derive(Clone, Copy, Default)]
struct FaceFluxRecord {
    sweep: f32,
    low: f32,
    high: f32,
}

fn face_flux_record(
    graph: &Graph,
    fields: &Fields,
    volumes: &[f32],
    face_index: usize,
    dt: f32,
    numerator: usize,
    denominator: usize,
) -> FaceFluxRecord {
    let face = &graph.subfaces[face_index];
    let sweep = physical_subface_sweep(graph, fields, face_index, dt);
    let n = face.negative_cell;
    let p = face.positive_cell;
    let vn = if n >= 0 { volumes[n as usize] } else { 0.0 };
    let vp = if p >= 0 { volumes[p as usize] } else { 0.0 };
    let cn = if n >= 0 {
        capacity_at(graph, fields, n as usize, numerator, denominator)
    } else {
        0.0
    };
    let cp = if p >= 0 {
        capacity_at(graph, fields, p as usize, numerator, denominator)
    } else {
        0.0
    };
    let low = if sweep > 0.0 && cn > 0.0 {
        mul(sweep, div(vn.clamp(0.0, cn.max(0.0)), cn))
    } else if sweep < 0.0 && cp > 0.0 {
        mul(sweep, div(vp.clamp(0.0, cp.max(0.0)), cp))
    } else {
        0.0
    };
    FaceFluxRecord {
        sweep,
        low,
        high: high_flux(graph, fields, face, sweep, low),
    }
}

fn compute_face_flux_records(
    graph: &Graph,
    fields: &Fields,
    volumes: &[f32],
    dt: f32,
    numerator: usize,
    denominator: usize,
    records: &mut [FaceFluxRecord],
) {
    debug_assert_eq!(records.len(), graph.subfaces.len());
    #[cfg(feature = "parallel")]
    if records.len() >= crate::kernels::PARALLEL_POINTWISE_THRESHOLD {
        const CHUNK: usize = 1024;
        records
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let base = chunk_index * CHUNK;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    *record = face_flux_record(
                        graph,
                        fields,
                        volumes,
                        base + offset,
                        dt,
                        numerator,
                        denominator,
                    );
                }
            });
        return;
    }
    for (face_index, record) in records.iter_mut().enumerate() {
        *record = face_flux_record(
            graph,
            fields,
            volumes,
            face_index,
            dt,
            numerator,
            denominator,
        );
    }
}

/// Literal frozen-bank receiver limiter. Moving cells use the production dual
/// potential/FISTA recurrence; static cells use monotone receiver factors.
fn limit_low(
    graph: &Graph,
    incidences: &TransportIncidences<'_>,
    fields: &mut Fields,
    volumes: &[f32],
    sweeps: &[f32],
    dt: f32,
    numerator: usize,
    denominator: usize,
) -> bool {
    let n = graph.cells.len();
    let mut current = vec![0.0; n];
    let mut prior = vec![0.0; n];
    let mut proposed = vec![0.0; n];
    let mut incoming = vec![0.0; n];
    if !fields.solid_motion_active {
        current.fill(1.0);
        prior.fill(1.0);
        for cell in 0..n {
            let mut total = 0.0;
            for entry in incidences.for_cell(cell) {
                total = add(
                    total,
                    delta(
                        fields.low_flux[entry.subface_id as usize],
                        entry.orientation,
                    )
                    .max(0.0),
                )
            }
            incoming[cell] = total
        }
    }
    for pass in 0..1024 {
        let mut invalid = 0usize;
        let mut first = 0usize;
        for cell in 0..n {
            let volume = starting_volume(volumes, fields, cell, dt);
            let capacity = capacity_at(graph, fields, cell, numerator, denominator);
            let (mut signed, mut outgoing, mut sweep_weight) = (0.0, 0.0, 0.0);
            for entry in incidences.for_cell(cell) {
                let face = &graph.subfaces[entry.subface_id as usize];
                let original = fields.low_flux[face.id as usize];
                let flux;
                if fields.solid_motion_active {
                    let sweep = sweeps[face.id as usize];
                    let weight = sweep.abs();
                    if weight == 0.0 {
                        flux = 0.0
                    } else {
                        let donor = donor(face, sweep);
                        let receiver = receiver(face, sweep);
                        if donor < 0 {
                            flux = 0.0
                        } else {
                            let base = if sweep >= 0.0 { original } else { -original };
                            let magnitude = add(
                                base,
                                mul(
                                    weight,
                                    current[donor as usize]
                                        - if receiver >= 0 {
                                            current[receiver as usize]
                                        } else {
                                            0.0
                                        },
                                ),
                            )
                            .clamp(0.0, weight);
                            flux = if sweep >= 0.0 { magnitude } else { -magnitude }
                        }
                    }
                    sweep_weight = add(sweep_weight, sweep.abs())
                } else {
                    let receiver = receiver(face, original);
                    let factor = if receiver >= 0 {
                        current[receiver as usize]
                    } else {
                        1.0
                    };
                    flux = mul(original, factor);
                    let change = delta(original, entry.orientation);
                    if change < 0.0 {
                        outgoing = add(outgoing, mul(-change, factor))
                    }
                }
                signed = add(signed, delta(flux, entry.orientation))
            }
            let candidate = add(volume, signed);
            if fields.solid_motion_active {
                let ready = if capacity != 0.0 {
                    valid(candidate, capacity)
                } else {
                    candidate.abs() <= roundoff(capacity_at(graph, fields, cell, 0, denominator))
                };
                let previous = current[cell];
                let diagonal = mul(2.0, sweep_weight);
                let next = if diagonal > 0.0 {
                    let tau = div(1.0, diagonal);
                    add(
                        add(previous, mul(tau, candidate - capacity)).max(0.0),
                        add(previous, mul(tau, candidate)).min(0.0),
                    )
                } else if !ready {
                    fault(fields, "transport-low", cell, candidate, capacity);
                    return false;
                } else {
                    previous
                };
                if !next.is_finite() {
                    fault(fields, "transport-low", cell, next, capacity);
                    return false;
                }
                proposed[cell] = next;
                if !ready {
                    if invalid == 0 {
                        first = cell
                    }
                    invalid += 1
                }
            } else {
                let previous = current[cell];
                let allowed = if valid(volume, capacity) {
                    capacity.max(volume)
                } else {
                    capacity
                };
                let mut next = previous;
                if candidate > allowed && incoming[cell] > 0.0 {
                    next = next.min(div(
                        add(allowed - volume, outgoing).max(0.0),
                        incoming[cell],
                    ));
                    next = next.min((previous - div(candidate - allowed, incoming[cell])).max(0.0));
                    if next == previous && previous > 0.0 {
                        next = lower_positive(previous)
                    }
                }
                proposed[cell] = next;
                if !valid(candidate, capacity) {
                    if invalid == 0 {
                        first = cell
                    }
                    invalid += 1
                }
            }
        }
        if invalid == 0 {
            break;
        }
        if pass == 1023 {
            fault(fields, "transport-low", first, 1024.0, invalid as f32);
            return false;
        }
        if fields.solid_motion_active {
            let k = pass as f32;
            for cell in 0..n {
                let x = proposed[cell];
                let old = prior[cell];
                prior[cell] = x;
                current[cell] = add(x, mul(div(k, k + 3.0), x - old))
            }
        } else {
            current.copy_from_slice(&proposed)
        }
    }
    for face in &graph.subfaces {
        let i = face.id as usize;
        let original = fields.low_flux[i];
        if fields.solid_motion_active {
            let sweep = sweeps[i];
            let weight = sweep.abs();
            if weight == 0.0 {
                fields.low_flux[i] = 0.0
            } else {
                let donor = donor(face, sweep);
                let receiver = receiver(face, sweep);
                if donor < 0 {
                    fields.low_flux[i] = 0.0
                } else {
                    let base = if sweep >= 0.0 { original } else { -original };
                    let magnitude = add(
                        base,
                        mul(
                            weight,
                            current[donor as usize]
                                - if receiver >= 0 {
                                    current[receiver as usize]
                                } else {
                                    0.0
                                },
                        ),
                    )
                    .clamp(0.0, weight);
                    fields.low_flux[i] = if sweep >= 0.0 { magnitude } else { -magnitude }
                }
            }
        } else {
            let receiver = receiver(face, original);
            fields.low_flux[i] = mul(
                original,
                if receiver >= 0 {
                    current[receiver as usize]
                } else {
                    1.0
                },
            )
        }
    }
    true
}

fn coverage_missing(graph: &Graph, incidences: &TransportIncidences<'_>, cell: usize) -> bool {
    let d = graph.dimension as usize;
    let mut negative = vec![0.0; d];
    let mut positive = vec![0.0; d];
    for e in incidences.for_cell(cell) {
        let f = &graph.subfaces[e.subface_id as usize];
        if e.orientation < 0 {
            positive[f.axis as usize] = add(positive[f.axis as usize], f.measure)
        } else {
            negative[f.axis as usize] = add(negative[f.axis as usize], f.measure)
        }
    }
    for a in 0..d {
        let expected = div(graph.cells[cell].measure, graph.cells[cell].widths[a]);
        let tolerance = mul(VOLUME_ROUNDOFF_RATIO, expected);
        if (negative[a] - expected).abs() > tolerance || (positive[a] - expected).abs() > tolerance
        {
            return true;
        }
    }
    false
}
fn other(face: &crate::types::Subface, orientation: i8) -> i32 {
    if orientation < 0 {
        face.positive_cell
    } else {
        face.negative_cell
    }
}
fn other_amount(
    incidences: &TransportIncidences<'_>,
    fields: &Fields,
    volumes: &[f32],
    flux: &[f32],
    cell: usize,
    parent: i32,
    dt: f32,
) -> f32 {
    let mut change = 0.0;
    for e in incidences.for_cell(cell) {
        if e.subface_id as i32 != parent {
            change = add(change, delta(flux[e.subface_id as usize], e.orientation))
        }
    }
    add(starting_volume(volumes, fields, cell, dt), change)
}
fn ordered_amount(
    graph: &Graph,
    incidences: &TransportIncidences<'_>,
    fields: &Fields,
    volumes: &[f32],
    flux: &[f32],
    cell: usize,
    parent: i32,
    dt: f32,
) -> f32 {
    let base = other_amount(incidences, fields, volumes, flux, cell, parent, dt);
    if parent < 0 {
        return base;
    }
    let face = &graph.subfaces[parent as usize];
    add(
        base,
        delta(
            flux[parent as usize],
            if face.negative_cell == cell as i32 {
                -1
            } else {
                1
            },
        ),
    )
}
fn cell_before(graph: &Graph, a: usize, b: usize) -> bool {
    if b == usize::MAX {
        return true;
    }
    let x = graph.cells[a].center;
    let y = graph.cells[b].center;
    for axis in 0..graph.dimension as usize {
        if x[axis] != y[axis] {
            return x[axis] < y[axis];
        }
    }
    false
}
fn face_budget(face: &crate::types::Subface, sweep: f32, main: f32) -> (f32, f32, f32) {
    let maximum = if donor(face, sweep) < 0 {
        0.0
    } else {
        sweep.abs()
    };
    let magnitude = if sweep >= 0.0 { main } else { -main };
    (magnitude, maximum - magnitude, maximum)
}

/// Deterministic zero-capacity component tree and leaf-to-root replacement.
fn allocate_closing(
    graph: &Graph,
    incidences: &TransportIncidences<'_>,
    fields: &mut Fields,
    volumes: &[f32],
    sweeps: &[f32],
    dt: f32,
    numerator: usize,
    denominator: usize,
) -> Option<Vec<i32>> {
    let n = graph.cells.len();
    let mut parents = vec![-1; n];
    if !fields.solid_motion_active {
        return Some(parents);
    }
    let closing: Vec<usize> = (0..n)
        .filter(|&i| capacity_at(graph, fields, i, numerator, denominator) == 0.0)
        .collect();
    let mut seen = vec![false; n];
    for seed in closing {
        if seen[seed] {
            continue;
        }
        let mut members = vec![seed];
        seen[seed] = true;
        let mut cursor = 0;
        while cursor < members.len() {
            let current = members[cursor];
            for e in incidences.for_cell(current) {
                let face = &graph.subfaces[e.subface_id as usize];
                let q = other(face, e.orientation);
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                if seen[q] || capacity_at(graph, fields, q, numerator, denominator) != 0.0 {
                    continue;
                }
                let b = face_budget(
                    face,
                    sweeps[face.id as usize],
                    fields.low_flux[face.id as usize],
                );
                if b.0.min(b.1) <= 0.0 {
                    continue;
                }
                if members.len() == 128 {
                    fault(fields, "transport-closing", seed, 128.0, 0.0);
                    return None;
                }
                seen[q] = true;
                members.push(q)
            }
            cursor += 1
        }
        let mut leader = usize::MAX;
        for &m in &members {
            if cell_before(graph, m, leader) {
                leader = m
            }
        }
        if seed != leader {
            continue;
        }
        let (mut component_budget, mut total, mut any) = (0.0, 0.0, false);
        for &m in &members {
            let residual = other_amount(
                incidences,
                fields,
                volumes,
                &fields.low_flux,
                m,
                -1,
                dt,
            );
            let tolerance = roundoff(capacity_at(graph, fields, m, 0, denominator));
            if residual.abs() > tolerance {
                fault(fields, "transport-closing", m, residual, tolerance);
                return None;
            }
            component_budget = add(component_budget, tolerance);
            total = add(total, residual);
            any |= residual != 0.0
        }
        if !any {
            continue;
        }
        let mut depth = vec![-1i32; members.len()];
        let mut root = None;
        let mut root_budget = -1.0;
        for (index, &m) in members.iter().enumerate() {
            for e in incidences.for_cell(m) {
                let face = &graph.subfaces[e.subface_id as usize];
                let q = other(face, e.orientation);
                if q >= 0
                    && (capacity_at(graph, fields, q as usize, numerator, denominator) <= 0.0
                        || coverage_missing(graph, incidences, q as usize))
                {
                    continue;
                }
                let b = face_budget(
                    face,
                    sweeps[face.id as usize],
                    fields.low_flux[face.id as usize],
                );
                let signed = if e.orientation < 0 { total } else { -total };
                let correction = if sweeps[face.id as usize] >= 0.0 {
                    signed
                } else {
                    -signed
                };
                let available = if correction == 0.0 {
                    b.0.min(b.1)
                } else if correction >= 0.0 {
                    b.1
                } else {
                    b.0
                };
                if available >= component_budget
                    && (available > root_budget
                        || available == root_budget
                            && root
                                .map(|r: (usize, u32)| cell_before(graph, m, members[r.0]))
                                .unwrap_or(true))
                {
                    root = Some((index, face.id));
                    root_budget = available
                }
            }
        }
        let Some((ri, rf)) = root else {
            fault(fields, "transport-closing", leader, total, -1.0);
            return None;
        };
        parents[members[ri]] = rf as i32;
        depth[ri] = 0;
        for _ in 1..members.len() {
            let mut chosen: Option<(usize, u32, i32, f32)> = None;
            for (index, &m) in members.iter().enumerate() {
                if depth[index] >= 0 {
                    continue;
                }
                for e in incidences.for_cell(m) {
                    let q = other(&graph.subfaces[e.subface_id as usize], e.orientation);
                    let Some(pi) = members.iter().position(|&v| v as i32 == q) else {
                        continue;
                    };
                    if depth[pi] < 0 {
                        continue;
                    }
                    let b = face_budget(
                        &graph.subfaces[e.subface_id as usize],
                        sweeps[e.subface_id as usize],
                        fields.low_flux[e.subface_id as usize],
                    );
                    let available = b.0.min(b.1);
                    if available < component_budget {
                        continue;
                    }
                    if chosen
                        .map(|(ci, _, _, best)| {
                            available > best
                                || available == best && cell_before(graph, m, members[ci])
                        })
                        .unwrap_or(true)
                    {
                        chosen = Some((index, e.subface_id, depth[pi] + 1, available))
                    }
                }
            }
            let Some((ci, face, dep, _)) = chosen else {
                fault(fields, "transport-closing", leader, total, -2.0);
                return None;
            };
            parents[members[ci]] = face as i32;
            depth[ci] = dep
        }
        let mut processed = vec![false; members.len()];
        for _ in 0..members.len() {
            let mut chosen = None;
            for i in 0..members.len() {
                if processed[i] {
                    continue;
                }
                if chosen
                    .map(|j| {
                        depth[i] > depth[j]
                            || depth[i] == depth[j] && cell_before(graph, members[i], members[j])
                    })
                    .unwrap_or(true)
                {
                    chosen = Some(i)
                }
            }
            let i = chosen.unwrap();
            let m = members[i];
            let face_id = parents[m];
            let face = &graph.subfaces[face_id as usize];
            let base = other_amount(
                incidences,
                fields,
                volumes,
                &fields.low_flux,
                m,
                face_id,
                dt,
            );
            let replacement = if face.negative_cell == m as i32 {
                base
            } else {
                -base
            };
            let original = fields.low_flux[face_id as usize];
            let sweep = sweeps[face_id as usize];
            let magnitude = if sweep >= 0.0 {
                replacement
            } else {
                -replacement
            };
            let maximum = face_budget(face, sweep, original).2;
            if !(magnitude >= 0.0
                && magnitude <= maximum
                && (replacement - original).abs() <= component_budget)
            {
                fault(fields, "transport-closing", m, replacement, maximum);
                return None;
            }
            fields.low_flux[face_id as usize] = replacement;
            if ordered_amount(
                graph,
                incidences,
                fields,
                volumes,
                &fields.low_flux,
                m,
                face_id,
                dt,
            ) != 0.0
            {
                let observed = ordered_amount(
                    graph,
                    incidences,
                    fields,
                    volumes,
                    &fields.low_flux,
                    m,
                    face_id,
                    dt,
                );
                fault(fields, "transport-closing", m, observed, 0.0);
                return None;
            }
            processed[i] = true
        }
    }
    Some(parents)
}

pub fn transport_volume(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
) -> Result<(usize, Vec<TransportMicrostepReceipt>), ValidationError> {
    transport_volume_with_commit(graph, fields, dt, true, |_, _, _| Ok(()))
}

/// Production transport commits external ledgers at the same successful
/// microstep boundary as density. Diagnostic field copies are opt-in.
pub fn transport_volume_with_commit(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
    capture_receipts: bool,
    mut commit: impl FnMut(usize, f32, &mut Fields) -> Result<(), ValidationError>,
) -> Result<(usize, Vec<TransportMicrostepReceipt>), ValidationError> {
    fields.validate_for(graph)?;
    if graph.dimension != 2 && graph.dimension != 3 {
        return Err(ValidationError("transport dimension must be 2 or 3".into()));
    }
    let incidences = TransportIncidences::new(graph);
    let n = graph.cells.len();
    let mut volumes: Vec<f32> = graph
        .cells
        .iter()
        .map(|c| mul(fields.density[c.id as usize], c.measure))
        .collect();
    let mut cfl = 0.0f32;
    for cell in 0..n {
        let old = capacity_at(graph, fields, cell, 0, 1);
        let final_cap = capacity_at(graph, fields, cell, 1, 1);
        if !valid(volumes[cell], old) {
            fault(fields, "transport-initialize", cell, volumes[cell], old);
            return Ok((1, vec![]));
        }
        let mut cfl_capacity = old.max(final_cap);
        if old > 0.0 && final_cap > 0.0 {
            cfl_capacity = old.min(final_cap)
        }
        let (mut outgoing, mut prism, mut total_rate) = (0.0f32, 0.0f32, 0.0f32);
        for e in incidences.for_cell(cell) {
            let face = &graph.subfaces[e.subface_id as usize];
            let row = &graph.rows[face.row_id as usize];
            let rate = physical_subface_rate(graph, fields, e.subface_id as usize);
            let flow = rate.abs();
            total_rate = add(total_rate, flow);
            // `delta` is the signed change of this cell. Outgoing volume is
            // its opposite (positive flux leaves the negative-side cell).
            outgoing = add(outgoing, (-delta(rate, e.orientation)).max(0.0));
            if row.open_fraction > 1e-8 {
                prism = prism.max(div(
                    flow,
                    mul(
                        mul(face.measure, row.open_fraction),
                        graph.cells[cell].widths[row.axis as usize],
                    ),
                ))
            }
        }
        let source = fields.source_rate.get(cell).copied().unwrap_or(0.0);
        if cfl_capacity == 0.0 {
            if add(total_rate, source) > 0.0 {
                fault(fields, "transport-plan", cell, add(total_rate, source), 0.0)
            }
            continue;
        }
        cfl = cfl.max(mul(dt, div(add(outgoing, source), cfl_capacity).max(prism)))
    }
    let steps = (2.0 * cfl).ceil().max(1.0) as usize;
    if steps > 128 {
        fault(fields, "transport-plan", 0, steps as f32, 128.0);
        return Ok((steps, vec![]));
    }
    let dtm = div(dt, steps as f32);
    let mut sweeps = vec![0.0; graph.subfaces.len()];
    fields.low_flux = vec![0.0; graph.subfaces.len()];
    fields.high_flux = vec![0.0; graph.subfaces.len()];
    fields.limited_flux = vec![0.0; graph.subfaces.len()];
    let mut face_flux_records = vec![FaceFluxRecord::default(); graph.subfaces.len()];
    let mut receipts = Vec::with_capacity(steps);
    for step in 0..steps {
        reconstruct_interfaces(graph, fields)?;
        compute_face_flux_records(
            graph,
            fields,
            &volumes,
            dtm,
            step,
            steps,
            &mut face_flux_records,
        );
        for (i, record) in face_flux_records.iter().copied().enumerate() {
            if !record.high.is_finite() {
                fault(fields, "transport-flux", i, record.high, record.low);
                return Ok((steps, receipts));
            }
            sweeps[i] = record.sweep;
            fields.low_flux[i] = record.low;
            fields.high_flux[i] = record.high
        }
        let initial = if capture_receipts {
            fields.low_flux.clone()
        } else {
            Vec::new()
        };
        if !limit_low(
            graph,
            &incidences,
            fields,
            &volumes,
            &sweeps,
            dtm,
            step + 1,
            steps,
        ) {
            return Ok((steps, receipts));
        }
        let Some(parents) =
            allocate_closing(
                graph,
                &incidences,
                fields,
                &volumes,
                &sweeps,
                dtm,
                step + 1,
                steps,
            )
        else {
            return Ok((steps, receipts));
        };
        let mut low_volume = vec![0.0; n];
        let mut increase = vec![1.0; n];
        let mut decrease = vec![1.0; n];
        let mut positive = vec![0.0; n];
        let mut negative = vec![0.0; n];
        for cell in 0..n {
            let (mut low_delta, mut pb, mut nb) = (0.0, 0.0, 0.0);
            for e in incidences.for_cell(cell) {
                let i = e.subface_id as usize;
                low_delta = add(low_delta, delta(fields.low_flux[i], e.orientation));
                let anti = fields.high_flux[i] - fields.low_flux[i];
                let q = delta(anti, e.orientation);
                pb = add(pb, q.max(0.0));
                nb = add(nb, (-q).max(0.0))
            }
            let capacity = capacity_at(graph, fields, cell, step + 1, steps);
            let mut low = add(starting_volume(&volumes, fields, cell, dtm), low_delta);
            if fields.solid_motion_active && capacity == 0.0 {
                low = ordered_amount(
                    graph,
                    &incidences,
                    fields,
                    &volumes,
                    &fields.low_flux,
                    cell,
                    parents[cell],
                    dtm,
                )
            }
            low_volume[cell] = low;
            positive[cell] = pb;
            negative[cell] = nb;
            if !valid(low, capacity) {
                fault(fields, "transport-low", cell, low, capacity);
                return Ok((steps, receipts));
            }
            let observed = low.clamp(0.0, capacity.max(0.0));
            if pb > 0.0 {
                increase[cell] = div(capacity - observed, pb).min(1.0)
            }
            if nb > 0.0 {
                decrease[cell] = div(observed, nb).min(1.0)
            }
        }
        for face in &graph.subfaces {
            let i = face.id as usize;
            let low = fields.low_flux[i];
            let anti = fields.high_flux[i] - low;
            let n = face.negative_cell;
            let p = face.positive_cell;
            let (mut ni, mut nd, mut pi, mut pd) = (1.0, 1.0, 1.0, 1.0);
            if n >= 0 {
                ni = increase[n as usize];
                nd = decrease[n as usize]
            }
            if p >= 0 {
                pi = increase[p as usize];
                pd = decrease[p as usize]
            }
            let mut factor = if anti >= 0.0 { nd.min(pi) } else { ni.min(pd) };
            if fields.solid_motion_active
                && ((n >= 0 && capacity_at(graph, fields, n as usize, step + 1, steps) == 0.0)
                    || (p >= 0 && capacity_at(graph, fields, p as usize, step + 1, steps) == 0.0))
            {
                factor = 0.0
            }
            fields.limited_flux[i] = add(low, mul(factor, fields.high_flux[i] - low))
        }
        let mut next = vec![0.0; n];
        for cell in 0..n {
            let mut change = 0.0;
            for e in incidences.for_cell(cell) {
                change = add(
                    change,
                    delta(fields.limited_flux[e.subface_id as usize], e.orientation),
                )
            }
            let capacity = capacity_at(graph, fields, cell, step + 1, steps);
            next[cell] = add(starting_volume(&volumes, fields, cell, dtm), change);
            if fields.solid_motion_active && capacity == 0.0 {
                next[cell] = ordered_amount(
                    graph,
                    &incidences,
                    fields,
                    &volumes,
                    &fields.limited_flux,
                    cell,
                    parents[cell],
                    dtm,
                )
            }
            if !valid(next[cell], capacity) {
                fault(fields, "transport-commit", cell, next[cell], capacity);
                return Ok((steps, receipts));
            }
        }
        if capture_receipts {
            receipts.push(TransportMicrostepReceipt {
                interface_normal: fields.interface_normal.clone(),
                interface_offset: fields.interface_offset.clone(),
                sweep: sweeps.clone(),
                initial_low_flux: initial,
                high_flux: fields.high_flux.clone(),
                low_flux: fields.low_flux.clone(),
                low_volume,
                positive_budget: positive,
                negative_budget: negative,
                increase,
                decrease,
                limited_flux: fields.limited_flux.clone(),
                next_volume: next.clone(),
            });
        }
        volumes = next;
        for cell in 0..n {
            fields.density[cell] = div(volumes[cell], graph.cells[cell].measure);
            fields.gamma[cell] = 1.0
        }
        commit(step, dtm, fields)?;
    }
    Ok((steps, receipts))
}

pub fn transport_volume_microstep(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
) -> Result<TransportMicrostepReceipt, ValidationError> {
    let (count, mut receipts) = transport_volume(graph, fields, dt)?;
    if count != 1 || receipts.len() != 1 {
        return Err(ValidationError(format!(
            "validation microstep requires one CFL packet, observed {count}"
        )));
    }
    Ok(receipts.remove(0))
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::types::{Cell, Row, RowTerm, Subface};

    #[test]
    fn face_flux_plane_is_bitwise_across_pool_sizes() {
        let count = crate::kernels::PARALLEL_POINTWISE_THRESHOLD + 17;
        let graph = Graph {
            dimension: 2,
            dimensions: [1.0, 1.0, 0.0],
            cells: vec![Cell {
                id: 0,
                minimum: [0.0, 0.0, 0.0],
                maximum: [1.0, 1.0, 0.0],
                center: [0.5, 0.5, 0.0],
                widths: [1.0, 1.0, 0.0],
                measure: 1.0,
                ..Cell::default()
            }],
            rows: vec![Row {
                id: 0,
                axis: 0,
                center: [1.0, 0.5, 0.0],
                measure: 1.0,
                distance: 1.0,
                dual_weight: 1.0,
                terms: vec![RowTerm {
                    cell_id: 0,
                    coefficient: -1.0,
                }],
                open_fraction: 1.0,
                ..Row::default()
            }],
            subfaces: (0..count)
                .map(|id| Subface {
                    id: id as u32,
                    row_id: 0,
                    axis: 0,
                    center: [1.0, 0.5, 0.0],
                    measure: 1.0,
                    negative_cell: 0,
                    positive_cell: -1,
                    aperture: 1.0,
                    ..Subface::default()
                })
                .collect(),
            ..Graph::default()
        };
        let fields = Fields {
            density: vec![0.375],
            capacity: vec![1.0],
            face_velocity: vec![0.125],
            interface_normal: vec![1.0, 0.0],
            interface_offset: vec![0.125],
            ..Fields::default()
        };
        let volumes = [0.375];
        let expected: Vec<_> = (0..count)
            .map(|face| face_flux_record(&graph, &fields, &volumes, face, 0.1, 0, 1))
            .collect();
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut actual = vec![FaceFluxRecord::default(); count];
            pool.install(|| {
                compute_face_flux_records(&graph, &fields, &volumes, 0.1, 0, 1, &mut actual)
            });
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_eq!(actual.sweep.to_bits(), expected.sweep.to_bits());
                assert_eq!(actual.low.to_bits(), expected.low.to_bits());
                assert_eq!(actual.high.to_bits(), expected.high.to_bits());
            }
        }
    }
}
