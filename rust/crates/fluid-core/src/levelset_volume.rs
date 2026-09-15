//! Conservative volume plus signed-distance transport experiment.

use crate::numerics::{owner_at, sample_support};
use crate::levelset_redistance::{sample_scalar, RedistanceField};
use crate::levelset_surface;
use crate::presentation::{RdfSupport, RdfSurface, RdfTopology};
use crate::types::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
struct StageClock {
    #[cfg(not(target_arch = "wasm32"))]
    started: std::time::Instant,
}
impl StageClock {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            started: std::time::Instant::now(),
        }
    }
    fn elapsed(self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        { return self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64; }
        #[cfg(target_arch = "wasm32")]
        { 0 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LevelSetVolumeReceipt {
    pub initial_liquid_volume: f64,
    pub final_liquid_volume: f64,
    pub signed_volume_drift: f64,
    pub absolute_volume_drift: f64,
    pub volume_roundoff_bound: f64,
    pub over_capacity_cell_count: usize,
    /// Maximum physical volume above the cell's open capacity.
    pub maximum_volume_over_capacity: f64,
    pub total_volume_over_capacity: f64,
    pub maximum_over_capacity_ratio: f64,
    /// Area enclosed by phi after transport, weighted by accepted open capacity.
    pub phi_implied_liquid_volume: f64,
    /// Conservative V minus the capacity-weighted volume implied by phi.
    pub signed_phi_volume_mismatch: f64,
    pub absolute_phi_volume_mismatch: f64,
    pub inside_band_absolute_phi_volume_mismatch: f64,
    pub outside_band_absolute_phi_volume_mismatch: f64,
    pub maximum_absolute_phi_volume_mismatch: f64,
    /// Maximum |V - C H(phi)| / C over cells with positive integrated capacity.
    pub maximum_normalized_phi_volume_mismatch: f64,
    pub sharpening: crate::levelset_sharpening::SharpeningReceipt,
    pub maximum_normalized_row_residual: f64,
    pub maximum_donor_residual: f64,
    pub zero_weight_donors: usize,
    pub invalid_phi_samples: usize,
    pub maximum_trace_distance: f64,
    pub maximum_trace_courant: f64,
    pub trace_nanoseconds: u64,
    pub volume_gather_nanoseconds: u64,
    pub phi_gather_nanoseconds: u64,
    pub plane_fit_nanoseconds: u64,
    pub rdf_nanoseconds: u64,
    pub redistance_nanoseconds: u64,
    pub redistanced_samples: usize,
    pub redistance_fallback_samples: usize,
    pub redistance_segment_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterfaceSeamReceipt {
    pub comparison_count: usize,
    pub skipped_invalid_plane_count: usize,
    pub mean_absolute_offset_difference: f64,
    pub rms_offset_difference: f64,
    pub maximum_absolute_offset_difference: f64,
}

/// Per-cell S0 diagnostic for the disagreement between conservative volume and phi.
///
/// `phi_implied_fill` is the average linear-cut fill over the adaptive cell's
/// finest children. `phi_implied_volume` follows the sharpening formulation's
/// C H(phi) product; it is deliberately not an exact terrain/liquid intersection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhiVolumeMismatchCell {
    pub cell_id: u32,
    pub phi_implied_fill: f64,
    pub accepted_volume: f64,
    pub integrated_open_capacity: f64,
    pub phi_implied_volume: f64,
    pub signed_volume_mismatch: f64,
    pub normalized_volume_mismatch: f64,
    pub inside_interface_band: bool,
}

/// Publish phi-only planes for pressure embedding. These planes never feed the
/// direct level-set surface; density and volume do not position their zero set.
pub fn publish_pressure_geometry_from_phi(
    graph: &Graph,
    fields: &mut Fields,
    phi: &[f32],
) -> Result<(), ValidationError> {
    if graph.dimension != 2 || phi.len() != graph.cells.len() {
        return Err(ValidationError("level-set pressure geometry requires one phi value per 2-D cell".into()));
    }
    fields.interface_normal.fill(0.0);
    fields.interface_offset.fill(0.0);
    for i in 0..graph.cells.len() {
        let c = &graph.cells[i];
        let (mut mxx, mut mxy, mut myy, mut bx, mut by) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut directional_fallback = ([0.0_f64; 2], 0.0_f64);
        for &row_id in &graph.incidences[i] {
            let row = &graph.rows[row_id as usize];
            let Some(own) = row.terms.iter().find(|term| term.cell_id as usize == i) else { continue };
            for term in &row.terms {
                let j = term.cell_id as usize;
                if own.coefficient * term.coefficient >= 0.0 || !phi[j].is_finite() { continue; }
                let dx = (graph.cells[j].center[0] - c.center[0]) as f64;
                let dy = (graph.cells[j].center[1] - c.center[1]) as f64;
                let tangent = 1 - row.axis as usize;
                let overlap = (c.maximum[tangent].min(graph.cells[j].maximum[tangent])
                    - c.minimum[tangent].max(graph.cells[j].minimum[tangent])).max(0.0) as f64;
                let weight = overlap / (dx * dx + dy * dy).max(1e-20);
                let delta = (phi[j] - phi[i]) as f64;
                mxx += weight * dx * dx; mxy += weight * dx * dy; myy += weight * dy * dy;
                bx += weight * dx * delta; by += weight * dy * delta;
                let support = weight * (dx * dx + dy * dy);
                if support > directional_fallback.1 {
                    let inverse_length_squared = 1.0 / (dx * dx + dy * dy).max(1e-20);
                    directional_fallback = ([
                        delta * dx * inverse_length_squared,
                        delta * dy * inverse_length_squared,
                    ], support);
                }
            }
        }
        let determinant = mxx * myy - mxy * mxy;
        let scale = mxx.max(myy);
        let mut gradient = if scale > 1e-20 && determinant.abs() > 1e-7 * scale * scale {
            [(myy * bx - mxy * by) / determinant, (-mxy * bx + mxx * by) / determinant]
        } else {
            directional_fallback.0
        };
        let length = gradient[0].hypot(gradient[1]);
        if !(length > 1e-20 && length.is_finite()) { continue; }
        gradient[0] /= length; gradient[1] /= length;
        let offset = -(phi[i] as f64) / length;
        if !offset.is_finite() { continue; }
        fields.interface_normal[2 * i] = gradient[0] as f32;
        fields.interface_normal[2 * i + 1] = gradient[1] as f32;
        fields.interface_offset[i] = offset as f32;
    }
    Ok(())
}

fn trace_point(graph: &Graph, fields: &Fields, start: [f32; 2], span: f32, dt: f32) -> [f32; 2] {
    let first = sample_support(graph, fields, start[0], start[1], span);
    let midpoint = [
        (start[0] - 0.5 * dt * first[0]).clamp(0.0, graph.dimensions[0]),
        (start[1] - 0.5 * dt * first[1]).clamp(0.0, graph.dimensions[1]),
    ];
    let velocity = sample_support(graph, fields, midpoint[0], midpoint[1], span);
    [
        (start[0] - dt * velocity[0]).clamp(0.0, graph.dimensions[0]),
        (start[1] - dt * velocity[1]).clamp(0.0, graph.dimensions[1]),
    ]
}

#[derive(Clone, Copy)]
struct ReleasedWall {
    axis: usize,
    boundary: f32,
    tangent_minimum: f32,
    tangent_maximum: f32,
    inward: f32,
    displacement: f32,
}

fn released_walls(graph: &Graph, fields: &Fields, dt: f32) -> Vec<ReleasedWall> {
    let mut result = Vec::new();
    for row in graph.rows.iter().filter(|row| {
        row.kind == crate::types::RowKind::ClosedWorld && row.separating
    }) {
        for term in &row.terms {
            let inward = if term.coefficient >= 0.0 { 1.0 } else { -1.0 };
            let away_speed = inward
                * (fields.face_velocity[row.id as usize] - row.solid_velocity);
            if away_speed <= 1.0e-6 {
                continue;
            }
            let cell = &graph.cells[term.cell_id as usize];
            let axis = row.axis as usize;
            let tangent = 1 - axis;
            result.push(ReleasedWall {
                axis,
                boundary: row.center[axis],
                tangent_minimum: cell.minimum[tangent],
                tangent_maximum: cell.maximum[tangent],
                inward,
                displacement: dt * away_speed,
            });
        }
    }
    result
}

/// Signed distance to released solid faces, positive in the exterior solid.
/// Taking the maximum with transported phi supplies the air continuation on
/// both sides of the wall. The interior half is essential: it changes an old,
/// deeply negative contact value into `-distance_to_wall`, so a wall-normal
/// translation moves the zero set by the actual characteristic distance.
fn released_wall_phi(walls: &[ReleasedWall], point: [f32; 2]) -> Option<f32> {
    walls
        .iter()
        .filter(|wall| {
            let tangent = 1 - wall.axis;
            point[tangent] >= wall.tangent_minimum - 1.0e-6
                && point[tangent] <= wall.tangent_maximum + 1.0e-6
        })
        .map(|wall| {
            wall.displacement - wall.inward * (point[wall.axis] - wall.boundary)
        })
        .reduce(f32::max)
}

fn trace_rk2(
    graph: &Graph,
    fields: &Fields,
    dt: f32,
) -> (Vec<[[f64; 2]; 5]>, f64, f64) {
    let mut footprints = Vec::with_capacity(graph.cells.len());
    let (mut max_distance, mut max_courant) = (0.0_f64, 0.0_f64);
    for cell in &graph.cells {
        let start = [cell.center[0], cell.center[1]];
        let span = cell.widths[0].min(cell.widths[1]).max(1.0);
        let starts = [
            [cell.minimum[0], cell.minimum[1]],
            [cell.maximum[0], cell.minimum[1]],
            [cell.maximum[0], cell.maximum[1]],
            [cell.minimum[0], cell.maximum[1]],
            start,
        ];
        let traced = starts.map(|point| trace_point(graph, fields, point, span, dt));
        for (before, after) in starts.iter().zip(traced) {
            let distance = ((after[0] - before[0]).powi(2)
                + (after[1] - before[1]).powi(2)).sqrt() as f64;
            max_distance = max_distance.max(distance);
            max_courant = max_courant.max(distance / span as f64);
        }
        footprints.push(traced.map(|point| point.map(|value| value as f64)));
    }
    (footprints, max_distance, max_courant)
}

fn advect_shared_phi(
    graph: &Graph,
    fields: &Fields,
    previous: &RdfSurface,
    dt: f32,
    _receipt: &mut LevelSetVolumeReceipt,
) -> Result<Vec<f32>, ValidationError> {
    let [nx, ny] = previous.dimensions.map(|value| value as usize);
    if previous.vertex_phi_fine.len() != (nx + 1) * (ny + 1) {
        return Err(ValidationError("direct level-set vertex count does not match dimensions".into()));
    }
    let released_walls = if dt > 0.0 {
        released_walls(graph, fields, dt)
    } else {
        Vec::new()
    };
    let mut result = Vec::with_capacity(previous.vertex_phi_fine.len());
    for y in 0..=ny {
        for x in 0..=nx {
            let start = [x as f32, y as f32];
            let owner_point = [
                start[0].clamp(0.5, graph.dimensions[0] - 0.5),
                start[1].clamp(0.5, graph.dimensions[1] - 0.5), 0.0,
            ];
            let span = owner_at(graph, owner_point).map(|id| {
                graph.cells[id].widths[0].min(graph.cells[id].widths[1]).max(1.0)
            }).unwrap_or(1.0);
            let first = sample_support(graph, fields, start[0], start[1], span);
            let midpoint = [
                (start[0] - 0.5 * dt * first[0]).clamp(0.0, graph.dimensions[0]),
                (start[1] - 0.5 * dt * first[1]).clamp(0.0, graph.dimensions[1]),
            ];
            let velocity = sample_support(graph, fields, midpoint[0], midpoint[1], span);
            let raw_departure = [start[0] - dt * velocity[0], start[1] - dt * velocity[1]];
            let departure = [
                raw_departure[0].clamp(0.0, graph.dimensions[0]),
                raw_departure[1].clamp(0.0, graph.dimensions[1]),
            ];
            let sampled = sample_scalar(previous, departure)
                .ok_or_else(|| ValidationError("direct level-set departure has no finite scalar".into()))?;
            result.push(released_wall_phi(&released_walls, start)
                .map_or(sampled, |wall_phi| sampled.max(wall_phi)));
        }
    }
    Ok(result)
}

fn redistance_vertices(
    dimensions: [u32; 2],
    vertices: Vec<f32>,
    diagnostic_volume: f64,
    receipt: &mut LevelSetVolumeReceipt,
) -> Result<Vec<f32>, ValidationError> {
    let raw = levelset_surface::publish(dimensions, vertices, diagnostic_volume)?;
    if raw.segments_fine.is_empty() {
        receipt.redistance_fallback_samples += raw.vertex_phi_fine.len();
        return Ok(raw.vertex_phi_fine);
    }
    let distance = RedistanceField::new(&raw)?;
    let [nx, ny] = dimensions.map(|value| value as usize);
    let mut result = Vec::with_capacity(raw.vertex_phi_fine.len());
    for y in 0..=ny {
        for x in 0..=nx {
            let index = x + (nx + 1) * y;
            if let Some(value) = distance.sample([x as f32, y as f32]) {
                let prior = raw.vertex_phi_fine[index];
                // Reinitialization is a narrow-band operation. Preserve the
                // authored far field, and keep already-metric samples bitwise
                // stable so a resting interface remains an exact fixed point.
                if value.abs() <= 2.0 && (value - prior).abs() > 1.0e-6 {
                    receipt.redistanced_samples += 1;
                    result.push(value);
                } else {
                    result.push(prior);
                }
            } else {
                receipt.redistance_fallback_samples += 1;
                result.push(raw.vertex_phi_fine[index]);
            }
        }
    }
    Ok(result)
}

fn raw_weights_from_footprints(
    graph: &Graph,
    footprints: &[[[f64; 2]; 5]],
    capacity: &[f64],
) -> (Vec<Vec<(usize, f64)>>, usize) {
    // A bin is at least as wide as the largest leaf, so each leaf occupies at
    // most four bins. The map stays proportional to represented sparse cells,
    // rather than to the possibly much larger finest domain.
    let bin_width = graph.cells.iter()
        .map(|cell| cell.widths[0].max(cell.widths[1]) as f64)
        .fold(1.0_f64, f64::max);
    let bin = |value: f64| (value / bin_width).floor() as i64;
    let mut bins: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (donor, cell) in graph.cells.iter().enumerate() {
        if capacity.get(donor).copied().unwrap_or(0.0) <= 1e-30 { continue; }
        for by in bin(cell.minimum[1] as f64)..=bin(cell.maximum[1] as f64) {
            for bx in bin(cell.minimum[0] as f64)..=bin(cell.maximum[0] as f64) {
                bins.entry((bx, by)).or_default().push(donor);
            }
        }
    }
    let mut rows = Vec::with_capacity(footprints.len());
    for (receiver, footprint) in footprints.iter().enumerate() {
        let cell = &graph.cells[receiver];
        if capacity.get(receiver).copied().unwrap_or(0.0) <= 1e-30 {
            rows.push(Vec::new());
            continue;
        }
        let triangles = footprint_triangles(*footprint, cell);
        let minimum = [0, 1].map(|axis| triangles.iter().flatten()
            .map(|point| point[axis]).fold(f64::INFINITY, f64::min));
        let maximum = [0, 1].map(|axis| triangles.iter().flatten()
            .map(|point| point[axis]).fold(f64::NEG_INFINITY, f64::max));
        let mut candidates = Vec::new();
        if maximum[0] > minimum[0] && maximum[1] > minimum[1] {
            for by in bin(minimum[1])..=bin(maximum[1]) {
                for bx in bin(minimum[0])..=bin(maximum[0]) {
                    if let Some(ids) = bins.get(&(bx, by)) { candidates.extend_from_slice(ids); }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        let mut row = Vec::with_capacity(candidates.len());
        for donor in candidates {
            let source = &graph.cells[donor];
            let overlap: f64 = triangles.iter().map(|triangle| {
                polygon_area(&clip_rectangle(
                    triangle,
                    [source.minimum[0] as f64, source.minimum[1] as f64],
                    [source.maximum[0] as f64, source.maximum[1] as f64],
                ))
            }).sum();
            if overlap > 0.0 { row.push((donor, overlap)); }
        }
        if row.is_empty() && capacity.get(receiver).copied().unwrap_or(0.0) > 1e-30 {
            row.push((receiver, cell.measure as f64));
        }
        rows.push(row);
    }
    // A backward stencil need not touch every donor at high Courant number.
    // Add a deterministic self edge so final column normalisation can still
    // export that donor's exact amount.
    let mut covered = vec![false; graph.cells.len()];
    for row in &rows { for &(donor, _) in row { covered[donor] = true; } }
    let missing = covered.iter().enumerate()
        .filter(|(donor, value)| capacity[*donor] > 1e-30 && !**value).count();
    for donor in 0..covered.len() {
        if capacity[donor] > 1e-30 && !covered[donor] {
            rows[donor].push((donor, graph.cells[donor].measure as f64));
        }
    }
    (rows, missing)
}

#[cfg(test)]
fn raw_weights(
    graph: &Graph,
    landings: &[[f32; 2]],
    capacity: &[f64],
) -> (Vec<Vec<(usize, f64)>>, usize) {
    let footprints: Vec<_> = graph.cells.iter().zip(landings).map(|(cell, landing)| {
        let displacement = [
            landing[0] as f64 - cell.center[0] as f64,
            landing[1] as f64 - cell.center[1] as f64,
        ];
        [
            [cell.minimum[0] as f64 + displacement[0], cell.minimum[1] as f64 + displacement[1]],
            [cell.maximum[0] as f64 + displacement[0], cell.minimum[1] as f64 + displacement[1]],
            [cell.maximum[0] as f64 + displacement[0], cell.maximum[1] as f64 + displacement[1]],
            [cell.minimum[0] as f64 + displacement[0], cell.maximum[1] as f64 + displacement[1]],
            [landing[0] as f64, landing[1] as f64],
        ]
    }).collect();
    raw_weights_from_footprints(graph, &footprints, capacity)
}

fn signed_polygon_area(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() < 3 { return 0.0; }
    0.5 * (0..polygon.len()).map(|i| {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f64>()
}

fn polygon_area(polygon: &[[f64; 2]]) -> f64 {
    signed_polygon_area(polygon).abs()
}

fn clip_axis(
    polygon: &[[f64; 2]],
    axis: usize,
    bound: f64,
    keep_greater: bool,
) -> Vec<[f64; 2]> {
    let mut result = Vec::new();
    if polygon.is_empty() { return result; }
    let signed = |point: [f64; 2]| if keep_greater {
        point[axis] - bound
    } else {
        bound - point[axis]
    };
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let da = signed(a);
        let db = signed(b);
        if da >= 0.0 { result.push(a); }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            result.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    result
}

fn clip_rectangle(
    polygon: &[[f64; 2]],
    minimum: [f64; 2],
    maximum: [f64; 2],
) -> Vec<[f64; 2]> {
    let mut result = polygon.to_vec();
    result = clip_axis(&result, 0, minimum[0], true);
    result = clip_axis(&result, 0, maximum[0], false);
    result = clip_axis(&result, 1, minimum[1], true);
    clip_axis(&result, 1, maximum[1], false)
}

fn footprint_triangles(footprint: [[f64; 2]; 5], cell: &crate::Cell) -> [Vec<[f64; 2]>; 4] {
    let triangles = std::array::from_fn::<_, 4, _>(|i| vec![
        footprint[i], footprint[(i + 1) % 4], footprint[4],
    ]);
    let signs: Vec<_> = triangles.iter().map(|triangle| signed_polygon_area(triangle)).collect();
    let valid = signs.iter().all(|area| area.is_finite() && area.abs() > 1e-12)
        && signs.iter().all(|area| area.signum() == signs[0].signum());
    if valid { return triangles; }
    let displacement = [
        footprint[4][0] - cell.center[0] as f64,
        footprint[4][1] - cell.center[1] as f64,
    ];
    let corners = [
        [cell.minimum[0] as f64 + displacement[0], cell.minimum[1] as f64 + displacement[1]],
        [cell.maximum[0] as f64 + displacement[0], cell.minimum[1] as f64 + displacement[1]],
        [cell.maximum[0] as f64 + displacement[0], cell.maximum[1] as f64 + displacement[1]],
        [cell.minimum[0] as f64 + displacement[0], cell.maximum[1] as f64 + displacement[1]],
    ];
    std::array::from_fn(|i| vec![corners[i], corners[(i + 1) % 4], footprint[4]])
}

fn balance_capacity_marginals(
    rows: &mut [Vec<(usize, f64)>],
    capacity: &[f64],
) -> (f64, f64, usize) {
    const EPS: f64 = 1e-30;
    let mut zero_weight_donors = 0;
    for _ in 0..3 {
        for (receiver, row) in rows.iter_mut().enumerate() {
            let sum: f64 = row.iter().map(|entry| entry.1).sum();
            let scale = if capacity[receiver] > 0.0 && sum > EPS { capacity[receiver] / sum } else { 0.0 };
            for entry in row { entry.1 *= scale; }
        }
        let mut columns = vec![0.0; capacity.len()];
        for row in rows.iter() { for &(donor, value) in row { columns[donor] += value; } }
        zero_weight_donors = columns.iter().zip(capacity).filter(|(sum, q)| **q > 0.0 && **sum <= EPS).count();
        for row in rows.iter_mut() {
            for (donor, value) in row {
                *value *= if capacity[*donor] > 0.0 && columns[*donor] > EPS {
                    capacity[*donor] / columns[*donor]
                } else { 0.0 };
            }
        }
    }
    let max_row = rows.iter().enumerate().map(|(i, row)| {
        let sum: f64 = row.iter().map(|entry| entry.1).sum();
        if capacity[i] > EPS { ((sum - capacity[i]) / capacity[i]).abs() } else { sum.abs() }
    }).fold(0.0_f64, f64::max);
    let mut columns = vec![0.0; capacity.len()];
    for row in rows.iter() { for &(donor, value) in row { columns[donor] += value; } }
    let max_column = columns.iter().zip(capacity).map(|(sum, q)| {
        if *q > EPS { ((sum - q) / q).abs() } else { sum.abs() }
    }).fold(0.0_f64, f64::max);
    (max_row, max_column, zero_weight_donors)
}

fn maximum_excess_ratio(density: f32, capacity: f32) -> Option<f64> {
    (capacity > 0.0).then(|| density as f64 / capacity as f64 - 1.0)
}

fn commit_volume_amounts(graph: &Graph, fields: &mut Fields, amounts: &[f64]) -> f64 {
    let target: f64 = amounts.iter().sum();
    let mut cast_error_bound = 0.0;
    for (i, cell) in graph.cells.iter().enumerate() {
        fields.density[i] = (amounts[i] / cell.measure as f64) as f32;
        cast_error_bound +=
            (amounts[i] - fields.density[i] as f64 * cell.measure as f64).abs();
    }
    for _ in 0..4 {
        let committed: f64 = graph.cells.iter().enumerate()
            .map(|(i, cell)| fields.density[i] as f64 * cell.measure as f64).sum();
        let residual = target - committed;
        if residual.abs() > cast_error_bound + f64::EPSILON * target.abs().max(1.0) { break; }
        let best = graph.cells.iter().enumerate()
            .filter(|(i, _)| fields.capacity[*i] > 0.0 && fields.density[*i] > 0.0)
            .filter_map(|(id, cell)| {
                let previous = fields.density[id];
                let candidate = (previous as f64 + residual / cell.measure as f64).max(0.0) as f32;
                let changed = (candidate as f64 - previous as f64) * cell.measure as f64;
                let remaining = (residual - changed).abs();
                (changed != 0.0 && remaining < residual.abs()).then_some((remaining, id, candidate))
            }).min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((_, id, candidate)) = best else { break };
        fields.density[id] = candidate;
    }
    graph.cells.iter().enumerate()
        .map(|(i, cell)| fields.density[i] as f64 * cell.measure as f64).sum()
}

pub fn interface_seam_receipt(graph: &Graph, fields: &Fields) -> InterfaceSeamReceipt {
    let mut result = InterfaceSeamReceipt::default();
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for subface in &graph.subfaces {
        if subface.negative_cell < 0 || subface.positive_cell < 0 { continue; }
        let a = subface.negative_cell as usize;
        let b = subface.positive_cell as usize;
        if graph.cells[a].widths[..2] == graph.cells[b].widths[..2] { continue; }
        let valid = |i: usize| {
            let capacity = fields.capacity[i];
            let fraction = if capacity > 1e-8 { fields.density[i] / capacity } else { 0.0 };
            let nx = fields.interface_normal[2 * i];
            let ny = fields.interface_normal[2 * i + 1];
            let length = (nx * nx + ny * ny).sqrt();
            fraction > 0.0 && fraction < 1.0 && length.is_finite() && (length - 1.0).abs() < 1e-3
        };
        if !valid(a) || !valid(b) {
            result.skipped_invalid_plane_count += 1;
            continue;
        }
        let evaluation = |i: usize| {
            let c = &graph.cells[i];
            fields.interface_normal[2 * i] as f64 * (subface.center[0] - c.center[0]) as f64
                + fields.interface_normal[2 * i + 1] as f64 * (subface.center[1] - c.center[1]) as f64
                - fields.interface_offset[i] as f64
        };
        let difference = (evaluation(a) - evaluation(b)).abs();
        sum += difference;
        sum_squares += difference * difference;
        result.maximum_absolute_offset_difference = result.maximum_absolute_offset_difference.max(difference);
        result.comparison_count += 1;
    }
    if result.comparison_count > 0 {
        result.mean_absolute_offset_difference = sum / result.comparison_count as f64;
        result.rms_offset_difference = (sum_squares / result.comparison_count as f64).sqrt();
    }
    result
}

/// Compare the signed-distance reconstruction seen from both sides of every
/// mixed-resolution seam. Unlike the legacy PLIC metric, eligibility follows
/// phi and its fitted gradient rather than conservative volume fraction.
pub fn interface_seam_receipt_from_phi(
    graph: &Graph,
    fields: &Fields,
    phi: &[f32],
) -> InterfaceSeamReceipt {
    let mut result = InterfaceSeamReceipt::default();
    if phi.len() != graph.cells.len() || fields.interface_normal.len() != 2 * graph.cells.len() {
        return result;
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for subface in &graph.subfaces {
        if subface.negative_cell < 0 || subface.positive_cell < 0 { continue; }
        let a = subface.negative_cell as usize;
        let b = subface.positive_cell as usize;
        if graph.cells[a].widths[..2] == graph.cells[b].widths[..2] { continue; }
        let touches_interface = |i: usize| {
            let cell = &graph.cells[i];
            let nx = fields.interface_normal[2 * i].abs();
            let ny = fields.interface_normal[2 * i + 1].abs();
            let half_span = 0.5 * (nx * cell.widths[0] + ny * cell.widths[1]);
            phi[i].abs() <= half_span
        };
        let valid = |i: usize| {
            let nx = fields.interface_normal[2 * i];
            let ny = fields.interface_normal[2 * i + 1];
            let length = nx.hypot(ny);
            phi[i].is_finite() && length.is_finite() && (length - 1.0).abs() < 1e-3
        };
        if !valid(a) || !valid(b) {
            result.skipped_invalid_plane_count += 1;
            continue;
        }
        if !touches_interface(a) && !touches_interface(b) { continue; }
        let evaluation = |i: usize| {
            let cell = &graph.cells[i];
            phi[i] as f64
                + fields.interface_normal[2 * i] as f64
                    * (subface.center[0] - cell.center[0]) as f64
                + fields.interface_normal[2 * i + 1] as f64
                    * (subface.center[1] - cell.center[1]) as f64
        };
        let difference = (evaluation(a) - evaluation(b)).abs();
        sum += difference;
        sum_squares += difference * difference;
        result.maximum_absolute_offset_difference =
            result.maximum_absolute_offset_difference.max(difference);
        result.comparison_count += 1;
    }
    if result.comparison_count > 0 {
        result.mean_absolute_offset_difference = sum / result.comparison_count as f64;
        result.rms_offset_difference = (sum_squares / result.comparison_count as f64).sqrt();
    }
    result
}

pub fn phi_volume_mismatch_cells(
    graph: &Graph,
    fields: &Fields,
    phi: &[f32],
    surface: &RdfSurface,
) -> Result<Vec<PhiVolumeMismatchCell>, ValidationError> {
    if graph.dimension != 2
        || phi.len() != graph.cells.len()
        || fields.capacity.len() != graph.cells.len()
        || fields.density.len() != graph.cells.len()
    {
        return Err(ValidationError(
            "phi volume mismatch requires one scalar per 2-D cell".into(),
        ));
    }
    let fill = levelset_surface::implied_fill_fine_cells(surface)?;
    let nx = surface.dimensions[0] as usize;
    let mut cells = Vec::with_capacity(graph.cells.len());
    for cell in &graph.cells {
        let id = cell.id as usize;
        let mut implied_area = 0.0_f64;
        for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
            for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                implied_area += fill[x + nx * y] as f64;
            }
        }
        let average_fill = implied_area / cell.measure as f64;
        let integrated_capacity = fields.capacity[id] as f64 * cell.measure as f64;
        let implied = integrated_capacity * average_fill;
        let volume = fields.density[id] as f64 * cell.measure as f64;
        let mismatch = volume - implied;
        cells.push(PhiVolumeMismatchCell {
            cell_id: cell.id,
            phi_implied_fill: average_fill,
            accepted_volume: volume,
            integrated_open_capacity: integrated_capacity,
            phi_implied_volume: implied,
            signed_volume_mismatch: mismatch,
            normalized_volume_mismatch: if integrated_capacity > 0.0 {
                mismatch.abs() / integrated_capacity
            } else {
                0.0
            },
            inside_interface_band: phi[id].abs()
                <= 2.0 * cell.widths[0].max(cell.widths[1]),
        });
    }
    Ok(cells)
}

fn publish_phi_volume_mismatch(
    graph: &Graph,
    fields: &Fields,
    phi: &[f32],
    surface: &RdfSurface,
    receipt: &mut LevelSetVolumeReceipt,
) -> Result<(), ValidationError> {
    for cell in phi_volume_mismatch_cells(graph, fields, phi, surface)? {
        let implied = cell.phi_implied_volume;
        let mismatch = cell.signed_volume_mismatch;
        let absolute = mismatch.abs();
        receipt.phi_implied_liquid_volume += implied;
        receipt.signed_phi_volume_mismatch += mismatch;
        receipt.absolute_phi_volume_mismatch += absolute;
        receipt.maximum_absolute_phi_volume_mismatch =
            receipt.maximum_absolute_phi_volume_mismatch.max(absolute);
        receipt.maximum_normalized_phi_volume_mismatch = receipt
            .maximum_normalized_phi_volume_mismatch
            .max(cell.normalized_volume_mismatch);
        if cell.inside_interface_band {
            receipt.inside_band_absolute_phi_volume_mismatch += absolute;
        } else {
            receipt.outside_band_absolute_phi_volume_mismatch += absolute;
        }
    }
    Ok(())
}

pub fn advance(
    graph: &Graph,
    fields: &mut Fields,
    previous_surface: &RdfSurface,
    rdf_topology: &RdfTopology,
    rdf_support: &RdfSupport,
    phi: &mut Vec<f32>,
    dt: f32,
) -> Result<(RdfSurface, LevelSetVolumeReceipt), ValidationError> {
    let [nx, ny] = previous_surface.dimensions.map(|v| v as usize);
    let mut fine_capacity = vec![0.0; nx * ny];
    for y in 0..ny { for x in 0..nx {
        fine_capacity[x + nx * y] = crate::numerics::owner_at(
            graph, [x as f32 + 0.5, y as f32 + 0.5, 0.0],
        ).map_or(0.0, |i| fields.capacity[i]);
    }}
    advance_with_fine_capacity(
        graph, fields, previous_surface, rdf_topology, rdf_support, phi, &fine_capacity, dt,
    )
}

pub fn advance_with_fine_capacity(
    graph: &Graph,
    fields: &mut Fields,
    previous_surface: &RdfSurface,
    _rdf_topology: &RdfTopology,
    _rdf_support: &RdfSupport,
    phi: &mut Vec<f32>,
    fine_capacity: &[f32],
    dt: f32,
) -> Result<(RdfSurface, LevelSetVolumeReceipt), ValidationError> {
    let expected_dimensions = [graph.dimensions[0] as u32, graph.dimensions[1] as u32];
    if previous_surface.dimensions != expected_dimensions
        || previous_surface.vertex_phi_fine.iter().any(|value| !value.is_finite())
    {
        return Err(ValidationError("direct level-set source does not match the accepted graph".into()));
    }
    // Validate explicit contour storage before any conservative field mutation.
    let _ = RedistanceField::new(previous_surface)?;
    let mut receipt = LevelSetVolumeReceipt::default();
    let clock = StageClock::start();
    let (footprints, distance, courant) = trace_rk2(graph, fields, dt);
    receipt.trace_nanoseconds = clock.elapsed();
    receipt.maximum_trace_distance = distance;
    receipt.maximum_trace_courant = courant;

    let clock = StageClock::start();
    let capacity: Vec<f64> = graph.cells.iter().enumerate()
        .map(|(i, c)| fields.capacity[i] as f64 * c.measure as f64).collect();
    let volume: Vec<f64> = graph.cells.iter().enumerate()
        .map(|(i, c)| fields.density[i] as f64 * c.measure as f64).collect();
    for i in 0..graph.cells.len() {
        if !capacity[i].is_finite() || capacity[i] < 0.0 || !volume[i].is_finite() || volume[i] < 0.0 {
            return Err(ValidationError("invalid level-set-volume mass or capacity".into()));
        }
        if volume[i] > 0.0 && capacity[i] <= 1e-30 {
            return Err(ValidationError("positive level-set volume has zero open capacity".into()));
        }
    }
    receipt.initial_liquid_volume = volume.iter().sum();
    let (mut weights, zero_support_donors) =
        raw_weights_from_footprints(graph, &footprints, &capacity);
    let (row_residual, donor_residual, zero_weight) = balance_capacity_marginals(&mut weights, &capacity);
    receipt.maximum_normalized_row_residual = row_residual;
    receipt.maximum_donor_residual = donor_residual;
    receipt.zero_weight_donors = zero_support_donors.max(zero_weight);
    let mut next_volume = vec![0.0_f64; graph.cells.len()];
    for (receiver, row) in weights.iter().enumerate() {
        for &(donor, coupling) in row {
            if capacity[donor] > 1e-30 {
                next_volume[receiver] += coupling * volume[donor] / capacity[donor];
            }
        }
    }
    let conservative_target: f64 = next_volume.iter().sum();
    let mut cast_error_bound = 0.0;
    for (i, cell) in graph.cells.iter().enumerate() {
        fields.density[i] = (next_volume[i] / cell.measure as f64) as f32;
        cast_error_bound +=
            (next_volume[i] - fields.density[i] as f64 * cell.measure as f64).abs();
    }
    // Project the f64 conservative target onto the nearest reachable f32 mass.
    // Only pre-existing liquid in positive-capacity cells may carry a change,
    // and the total correction cannot exceed the measured error introduced by
    // the casts above. A transport residual therefore cannot be hidden here.
    for _ in 0..4 {
        let committed: f64 = graph.cells.iter().enumerate()
            .map(|(i, cell)| fields.density[i] as f64 * cell.measure as f64)
            .sum();
        let residual = conservative_target - committed;
        if residual.abs() > cast_error_bound + f64::EPSILON * conservative_target.abs().max(1.0) {
            break;
        }
        let best = graph.cells.iter().enumerate()
            .filter(|(i, _)| fields.capacity[*i] > 0.0 && fields.density[*i] > 0.0)
            .filter_map(|(id, cell)| {
                let previous = fields.density[id];
                let candidate = (previous as f64 + residual / cell.measure as f64).max(0.0) as f32;
                let changed = (candidate as f64 - previous as f64) * cell.measure as f64;
                let remaining = (residual - changed).abs();
                (changed != 0.0 && remaining < residual.abs())
                    .then_some((remaining, id, candidate))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((_, id, candidate)) = best else { break };
        fields.density[id] = candidate;
    }
    receipt.final_liquid_volume = graph.cells.iter().enumerate()
        .map(|(i, c)| fields.density[i] as f64 * c.measure as f64).sum();
    receipt.signed_volume_drift = receipt.final_liquid_volume - receipt.initial_liquid_volume;
    receipt.absolute_volume_drift = receipt.signed_volume_drift.abs();
    receipt.volume_roundoff_bound = 2.0 * f32::EPSILON as f64 * receipt.initial_liquid_volume.max(1.0);
    for i in 0..graph.cells.len() {
        let excess = (fields.density[i] as f64 - fields.capacity[i] as f64).max(0.0)
            * graph.cells[i].measure as f64;
        if excess > 0.0 {
            receipt.over_capacity_cell_count += 1;
            receipt.maximum_volume_over_capacity = receipt.maximum_volume_over_capacity.max(excess);
            receipt.total_volume_over_capacity += excess;
            if let Some(ratio) = maximum_excess_ratio(fields.density[i], fields.capacity[i]) {
                receipt.maximum_over_capacity_ratio = receipt.maximum_over_capacity_ratio.max(ratio);
            }
        }
    }
    receipt.volume_gather_nanoseconds = clock.elapsed();

    let clock = StageClock::start();
    let vertices = advect_shared_phi(graph, fields, previous_surface, dt, &mut receipt)?;
    receipt.phi_gather_nanoseconds = clock.elapsed();

    let clock = StageClock::start();
    let vertices = redistance_vertices(
        previous_surface.dimensions,
        vertices,
        receipt.final_liquid_volume,
        &mut receipt,
    )?;
    let surface = levelset_surface::publish(
        previous_surface.dimensions,
        vertices,
        receipt.final_liquid_volume,
    )?;
    receipt.rdf_nanoseconds = clock.elapsed();
    let clock = StageClock::start();
    let accepted_redistance = RedistanceField::new(&surface)?;
    receipt.redistance_segment_count = accepted_redistance.segment_count();
    *phi = graph.cells.iter().map(|cell| {
        let point = [cell.center[0], cell.center[1]];
        if let Some(value) = accepted_redistance.sample(point) {
            receipt.redistanced_samples += 1;
            value
        } else {
            receipt.redistance_fallback_samples += 1;
            sample_scalar(&surface, point).unwrap_or(f32::NAN)
        }
    }).collect();
    if phi.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError("direct level-set has no finite cell-centre scalar".into()));
    }
    let mut sharpened_volume: Vec<f64> = graph.cells.iter().enumerate()
        .map(|(i, cell)| fields.density[i] as f64 * cell.measure as f64).collect();
    receipt.sharpening = crate::levelset_sharpening::sharpen_volume(
        graph, fields, fine_capacity, &surface, phi, &mut sharpened_volume,
    )?;
    receipt.final_liquid_volume = commit_volume_amounts(graph, fields, &sharpened_volume);
    receipt.signed_volume_drift = receipt.final_liquid_volume - receipt.initial_liquid_volume;
    receipt.absolute_volume_drift = receipt.signed_volume_drift.abs();
    receipt.over_capacity_cell_count = 0;
    receipt.maximum_volume_over_capacity = 0.0;
    receipt.total_volume_over_capacity = 0.0;
    receipt.maximum_over_capacity_ratio = 0.0;
    for i in 0..graph.cells.len() {
        let excess = (fields.density[i] as f64 - fields.capacity[i] as f64).max(0.0)
            * graph.cells[i].measure as f64;
        if excess > 0.0 {
            receipt.over_capacity_cell_count += 1;
            receipt.maximum_volume_over_capacity = receipt.maximum_volume_over_capacity.max(excess);
            receipt.total_volume_over_capacity += excess;
            if let Some(ratio) = maximum_excess_ratio(fields.density[i], fields.capacity[i]) {
                receipt.maximum_over_capacity_ratio = receipt.maximum_over_capacity_ratio.max(ratio);
            }
        }
    }
    publish_phi_volume_mismatch(graph, fields, phi, &surface, &mut receipt)?;
    receipt.redistance_nanoseconds += clock.elapsed();
    Ok((surface, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};

    fn brick(key: u32, coordinate: [i32; 3], resolution: u8) -> BrickSeed {
        BrickSeed {
            id: key,
            key,
            coordinate,
            span_bricks: 1,
            resolution,
            active: true,
            density: Vec::new(),
            gamma: Vec::new(),
            refinement_region_scale: None,
        }
    }

    fn graph(dimensions: [u32; 3], bricks: Vec<BrickSeed>) -> Graph {
        compile_topology::<2>(TopologySeed {
            dimensions,
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks,
        }).unwrap().graph
    }

    fn cell_at(graph: &Graph, minimum: [f32; 2]) -> usize {
        graph.cells.iter().position(|cell| {
            cell.minimum[0] == minimum[0] && cell.minimum[1] == minimum[1]
        }).unwrap()
    }

    fn velocity_fields(graph: &Graph, velocity: [f32; 2]) -> Fields {
        let mut fields = Fields::default();
        fields.cell_velocity = graph.cells.iter()
            .flat_map(|_| velocity).collect();
        fields
    }

    fn circle_surface(dimensions: [u32; 2], centre: [f32; 2], radius: f32) -> RdfSurface {
        let vertices = (0..=dimensions[1]).flat_map(|y| (0..=dimensions[0]).map(move |x| {
            (x as f32 - centre[0]).hypot(y as f32 - centre[1]) - radius
        })).collect();
        levelset_surface::publish(dimensions, vertices, std::f64::consts::PI * (radius as f64).powi(2)).unwrap()
    }

    fn flat_non_distance_surface(dimensions: [u32; 2]) -> RdfSurface {
        let vertices = (0..=dimensions[1]).flat_map(|y| (0..=dimensions[0]).map(move |_| {
            if y < 8 { -6.5 } else if y == 8 { -2.75 } else { 1.0 }
        })).collect();
        levelset_surface::publish(dimensions, vertices, 8.25 * dimensions[0] as f64).unwrap()
    }

    #[test]
    fn separating_ceiling_carves_the_exact_wall_normal_phi_gap() {
        let mut graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let previous = levelset_surface::publish([8, 8], vec![-4.0; 81], 64.0).unwrap();
        // Deliberately disagree with the MAC wall face: the carve follows the
        // authoritative boundary flux, not this collocated trace velocity.
        let mut fields = velocity_fields(&graph, [0.0, -0.5]);
        fields.face_velocity = vec![0.0; graph.rows.len()];
        for row in &mut graph.rows {
            if row.kind == crate::types::RowKind::ClosedWorld
                && row.axis == 1
                && row.center[1] == 8.0
            {
                row.separating = true;
                fields.face_velocity[row.id as usize] = -2.0;
            }
        }

        let mut receipt = LevelSetVolumeReceipt::default();
        let full = advect_shared_phi(&graph, &fields, &previous, 0.25, &mut receipt).unwrap();
        let half = advect_shared_phi(&graph, &fields, &previous, 0.125, &mut receipt).unwrap();
        for x in 0..=8 {
            assert!((full[x + 9 * 8] - 0.5).abs() <= 1.0e-6);
            assert!((full[x + 9 * 7] + 0.5).abs() <= 1.0e-6);
            assert!((half[x + 9 * 8] - 0.25).abs() <= 1.0e-6);
            assert!((half[x + 9 * 7] + 0.75).abs() <= 1.0e-6);
        }

        for row in &mut graph.rows {
            row.separating = false;
        }
        let unchanged = advect_shared_phi(&graph, &fields, &previous, 0.25, &mut receipt).unwrap();
        assert!(unchanged.iter().all(|&phi| phi == -4.0));

        for row in &mut graph.rows {
            if row.kind == crate::types::RowKind::ClosedWorld
                && row.axis == 1
                && row.center[1] == 8.0
            {
                row.separating = true;
            }
        }
        let zero_dt = advect_shared_phi(&graph, &fields, &previous, 0.0, &mut receipt).unwrap();
        assert_eq!(zero_dt, previous.vertex_phi_fine);
    }

    #[test]
    fn pressure_geometry_keeps_phi_gradient_when_zero_is_outside_cell() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let phi: Vec<_> = graph.cells.iter().map(|cell| cell.center[1] - 20.0).collect();
        let mut fields = Fields::default();
        fields.interface_normal = vec![0.0; 2 * graph.cells.len()];
        fields.interface_offset = vec![0.0; graph.cells.len()];
        publish_pressure_geometry_from_phi(&graph, &mut fields, &phi).unwrap();
        let interior = cell_at(&graph, [3.0, 3.0]);
        assert!(fields.interface_normal[2 * interior].abs() < 1e-6);
        assert!((fields.interface_normal[2 * interior + 1] - 1.0).abs() < 1e-6);
        assert!(fields.interface_offset[interior] > graph.cells[interior].widths[1]);
    }

    #[test]
    fn pressure_geometry_uses_directional_gradient_in_one_cell_high_support() {
        let graph = graph([8, 1, 1], vec![brick(0, [0, 0, 0], 8)]);
        let phi: Vec<_> = graph.cells.iter().map(|cell| cell.center[0] - 4.0).collect();
        let mut fields = Fields::default();
        fields.interface_normal = vec![0.0; 2 * graph.cells.len()];
        fields.interface_offset = vec![0.0; graph.cells.len()];
        publish_pressure_geometry_from_phi(&graph, &mut fields, &phi).unwrap();
        assert!(graph.cells.iter().all(|cell| {
            let id = cell.id as usize;
            (fields.interface_normal[2 * id] - 1.0).abs() < 1e-6
                && fields.interface_normal[2 * id + 1].abs() < 1e-6
        }));
    }

    #[test]
    fn excess_ratio_reports_positive_near_solid_capacity() {
        let capacity = 1.0e-12;
        let density = 7.0e-12;
        let ratio = maximum_excess_ratio(density, capacity).unwrap();
        assert!((ratio - 6.0).abs() < 1.0e-5, "{ratio}");
        assert_eq!(maximum_excess_ratio(1.0, 0.0), None);
    }

    #[test]
    fn capacity_balancing_ends_with_exact_donor_marginals() {
        let mut rows = vec![vec![(0, 0.8), (1, 0.2)], vec![(0, 0.1), (1, 0.9)]];
        let capacity = [1.0, 3.0];
        let (_, donor_residual, missing) = balance_capacity_marginals(&mut rows, &capacity);
        assert_eq!(missing, 0);
        assert!(donor_residual < 1e-14, "{donor_residual}");
        let columns = [
            rows.iter().flat_map(|r| r.iter()).filter(|e| e.0 == 0).map(|e| e.1).sum::<f64>(),
            rows.iter().flat_map(|r| r.iter()).filter(|e| e.0 == 1).map(|e| e.1).sum::<f64>(),
        ];
        assert!((columns[0] - capacity[0]).abs() < 1e-14);
        assert!((columns[1] - capacity[1]).abs() < 1e-14);
    }

    #[test]
    fn phi_volume_mismatch_receipt_splits_interface_band_from_bulk() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let surface = levelset_surface::publish(
            [8, 8],
            (0..=8).flat_map(|y| (0..=8).map(move |_| y as f32 - 4.0)).collect(),
            32.0,
        ).unwrap();
        let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
        let mut fields = Fields::default();
        fields.capacity = vec![1.0; graph.cells.len()];
        fields.density = graph.cells.iter()
            .map(|cell| if cell.center[1] < 4.0 { 1.0 } else { 0.0 }).collect();
        let mut receipt = LevelSetVolumeReceipt::default();
        publish_phi_volume_mismatch(&graph, &fields, &phi, &surface, &mut receipt).unwrap();
        assert!(receipt.absolute_phi_volume_mismatch < 1e-6);
        assert!((receipt.phi_implied_liquid_volume - 32.0).abs() < 1e-6);

        let deep = cell_at(&graph, [0.0, 0.0]);
        fields.density[deep] += 0.25;
        let cells = phi_volume_mismatch_cells(&graph, &fields, &phi, &surface).unwrap();
        assert_eq!(cells.len(), graph.cells.len());
        assert_eq!(cells[deep].cell_id as usize, deep);
        assert!((cells[deep].phi_implied_fill - 1.0).abs() < 1e-6);
        assert!((cells[deep].accepted_volume - 1.25).abs() < 1e-6);
        assert!((cells[deep].normalized_volume_mismatch - 0.25).abs() < 1e-6);
        let mut changed = LevelSetVolumeReceipt::default();
        publish_phi_volume_mismatch(&graph, &fields, &phi, &surface, &mut changed).unwrap();
        assert!((changed.signed_phi_volume_mismatch - 0.25).abs() < 1e-6);
        assert!((changed.absolute_phi_volume_mismatch - 0.25).abs() < 1e-6);
        assert!((changed.outside_band_absolute_phi_volume_mismatch - 0.25).abs() < 1e-6);
        assert!((changed.maximum_normalized_phi_volume_mismatch - 0.25).abs() < 1e-6);
        assert_eq!(changed.inside_band_absolute_phi_volume_mismatch, 0.0);

        fields.capacity[deep] = 1.0e-12;
        fields.density[deep] = 1.25e-12;
        let near_solid = phi_volume_mismatch_cells(&graph, &fields, &phi, &surface).unwrap();
        assert!((near_solid[deep].normalized_volume_mismatch - 0.25).abs() < 1e-5,
            "{}", near_solid[deep].normalized_volume_mismatch);
    }

    #[test]
    fn phi_seam_metric_compares_signed_distance_on_both_rungs() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 4), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 4), brick(3, [1, 1, 0], 8),
        ]);
        let phi: Vec<_> = graph.cells.iter().map(|cell| cell.center[0] - 8.25).collect();
        let mut fields = Fields::default();
        fields.interface_normal = vec![0.0; 2 * graph.cells.len()];
        fields.interface_offset = vec![0.0; graph.cells.len()];
        publish_pressure_geometry_from_phi(&graph, &mut fields, &phi).unwrap();
        let seam = interface_seam_receipt_from_phi(&graph, &fields, &phi);
        assert!(seam.comparison_count > 0);
        assert!(seam.maximum_absolute_offset_difference < 1e-5,
            "{}", seam.maximum_absolute_offset_difference);
    }

    #[test]
    fn rectangle_support_is_identity_and_matches_equal_grid_translation() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let capacity: Vec<_> = graph.cells.iter().map(|cell| cell.measure as f64).collect();
        let centres: Vec<_> = graph.cells.iter().map(|cell| [cell.center[0], cell.center[1]]).collect();
        let (identity, missing) = raw_weights(&graph, &centres, &capacity);
        assert_eq!(missing, 0);
        for (receiver, row) in identity.iter().enumerate() {
            assert_eq!(row, &vec![(receiver, 1.0)]);
        }

        let receiver = cell_at(&graph, [3.0, 3.0]);
        let left = cell_at(&graph, [2.0, 3.0]);
        let mut landings = centres;
        landings[receiver][0] -= 0.25;
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        assert_eq!(rows[receiver], vec![(left, 0.25), (receiver, 0.75)]);
    }

    #[test]
    fn arbitrarily_small_motion_crosses_every_adaptive_rung_boundary() {
        let graph = graph([32, 8, 1], vec![
            brick(0, [0, 0, 0], 1),
            brick(1, [1, 0, 0], 2),
            brick(2, [2, 0, 0], 4),
            brick(3, [3, 0, 0], 8),
        ]);
        let capacity: Vec<_> = graph.cells.iter().map(|cell| cell.measure as f64).collect();
        let mut landings: Vec<_> = graph.cells.iter().map(|cell| [cell.center[0], cell.center[1]]).collect();
        let epsilon = 1.0e-4_f32;
        let receivers = [[8.0, 0.0], [16.0, 0.0], [24.0, 0.0]];
        for minimum in receivers {
            let receiver = cell_at(&graph, minimum);
            landings[receiver][0] -= epsilon;
        }
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        for minimum in receivers {
            let receiver = cell_at(&graph, minimum);
            let cell = &graph.cells[receiver];
            let upstream = rows[receiver].iter().find(|(donor, _)| {
                graph.cells[*donor].maximum[0] == minimum[0]
            }).map(|entry| entry.1).unwrap();
            assert!(upstream > 0.0, "width {} did not cross at x={}", cell.widths[0], minimum[0]);
            let sum: f64 = rows[receiver].iter().map(|entry| entry.1).sum();
            assert!((sum - cell.measure as f64).abs() < 1e-5, "{sum} != {}", cell.measure);

            let coarse_minimum = [minimum[0] - cell.widths[0] * 2.0, 0.0];
            let coarse = cell_at(&graph, coarse_minimum);
            let mut reverse: Vec<_> = graph.cells.iter()
                .map(|candidate| [candidate.center[0], candidate.center[1]]).collect();
            reverse[coarse][0] += epsilon;
            let (reverse_rows, _) = raw_weights(&graph, &reverse, &capacity);
            assert!(reverse_rows[coarse].iter().any(|(donor, overlap)| {
                graph.cells[*donor].minimum[0] == minimum[0] && *overlap > 0.0
            }), "coarse width {} did not cross into finer cells at x={}",
                graph.cells[coarse].widths[0], minimum[0]);
        }
    }

    #[test]
    fn diagonal_subcell_motion_reaches_corner_and_zero_capacity_is_excluded() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let mut capacity: Vec<_> = graph.cells.iter().map(|cell| cell.measure as f64).collect();
        let mut landings: Vec<_> = graph.cells.iter().map(|cell| [cell.center[0], cell.center[1]]).collect();
        let receiver = cell_at(&graph, [3.0, 3.0]);
        let corner = cell_at(&graph, [2.0, 2.0]);
        landings[receiver][0] -= 0.125;
        landings[receiver][1] -= 0.25;
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        assert_eq!(rows[receiver].len(), 4);
        assert!((rows[receiver].iter().find(|entry| entry.0 == corner).unwrap().1 - 0.03125).abs() < 1e-12);
        assert!((rows[receiver].iter().map(|entry| entry.1).sum::<f64>() - 1.0).abs() < 1e-12);

        capacity[corner] = 0.0;
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        assert!(rows[receiver].iter().all(|entry| entry.0 != corner));

        let boundary = cell_at(&graph, [0.0, 3.0]);
        landings[boundary][0] = 0.25;
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        assert!((rows[boundary].iter().map(|entry| entry.1).sum::<f64>() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn diagonal_tiny_motion_from_width_two_reaches_fine_corner_cell() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 4), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 4), brick(3, [1, 1, 0], 8),
        ]);
        let capacity: Vec<_> = graph.cells.iter().map(|cell| cell.measure as f64).collect();
        let mut landings: Vec<_> = graph.cells.iter().map(|cell| [cell.center[0], cell.center[1]]).collect();
        let receiver = cell_at(&graph, [6.0, 2.0]);
        let fine_corner = cell_at(&graph, [8.0, 4.0]);
        landings[receiver][0] += 1.0e-3;
        landings[receiver][1] += 1.0e-3;
        let (rows, _) = raw_weights(&graph, &landings, &capacity);
        let corner_overlap = rows[receiver].iter()
            .find(|entry| entry.0 == fine_corner).map(|entry| entry.1).unwrap_or(0.0);
        assert!(corner_overlap > 0.0 && corner_overlap < 2.0e-6, "{corner_overlap}");
        assert!((rows[receiver].iter().map(|entry| entry.1).sum::<f64>() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn donor_supported_only_by_closed_receiver_gets_conservative_fallback() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let donor = cell_at(&graph, [3.0, 3.0]);
        let closed_receiver = cell_at(&graph, [4.0, 3.0]);
        let mut capacity: Vec<_> = graph.cells.iter().map(|cell| cell.measure as f64).collect();
        capacity[closed_receiver] = 0.0;
        let mut landings: Vec<_> = graph.cells.iter().map(|cell| [cell.center[0], cell.center[1]]).collect();
        // Move every open receiver away from the donor; the geometrically
        // overlapping closed row must not count as usable donor support.
        for (receiver, landing) in landings.iter_mut().enumerate() {
            if receiver != closed_receiver { landing[0] = 7.5; }
        }
        landings[closed_receiver] = [3.5, 3.5];
        let (rows, missing) = raw_weights(&graph, &landings, &capacity);
        assert!(rows[closed_receiver].is_empty());
        assert!(missing > 0);
        assert!(rows[donor].iter().any(|entry| entry.0 == donor && entry.1 > 0.0));
        let mut balanced = rows;
        let (_, donor_residual, _) = balance_capacity_marginals(&mut balanced, &capacity);
        assert!(donor_residual < 1e-12, "{donor_residual}");
    }

    #[test]
    fn stationary_direct_level_set_preserves_vertices_and_segments_exactly() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [0.0, 0.0]);
        let source = circle_surface([16, 16], [7.0, 8.0], 3.0);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0 / 30.0, &mut receipt).unwrap();
        let accepted = levelset_surface::publish(source.dimensions, vertices, source.receipt.exact_area_fine).unwrap();
        assert_eq!(accepted.vertex_phi_fine, source.vertex_phi_fine);
        assert_eq!(accepted.segments_fine, source.segments_fine);
        assert_eq!(receipt.redistanced_samples, 0);
    }

    #[test]
    fn zero_dt_with_nonzero_velocity_preserves_authoritative_scalar_exactly() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [2.0, -3.0]);
        let source = circle_surface([16, 16], [7.0, 8.0], 3.0);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 0.0, &mut receipt).unwrap();
        assert_eq!(vertices, source.vertex_phi_fine);
    }

    #[test]
    fn microscopic_tangential_motion_does_not_mix_metrics_in_a_flat_field() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [3.0e-6, 0.0]);
        let source = flat_non_distance_surface([16, 16]);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0 / 30.0, &mut receipt).unwrap();
        assert_eq!(vertices, source.vertex_phi_fine);
        let accepted = levelset_surface::publish(source.dimensions, vertices,
            source.receipt.exact_area_fine).unwrap();
        assert_eq!(accepted.segments_fine, source.segments_fine);
    }

    #[test]
    fn normal_motion_changes_the_stored_scalar_continuously_from_zero_dt() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [0.0, 0.25]);
        let source = flat_non_distance_surface([16, 16]);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0e-4, &mut receipt).unwrap();
        let maximum = vertices.iter().zip(&source.vertex_phi_fine)
            .map(|(next, prior)| (next - prior).abs()).fold(0.0_f32, f32::max);
        assert!(maximum > 0.0 && maximum < 1.0e-3, "{maximum}");
    }

    #[test]
    fn uniformly_translated_circle_stays_single_signed_region() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [0.2, 0.0]);
        let source = circle_surface([16, 16], [7.0, 8.0], 3.0);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0, &mut receipt).unwrap();
        let centre_index = 8 + 8 * 17;
        let expected = sample_scalar(&source, [7.8, 8.0]).unwrap();
        assert!((vertices[centre_index] - expected).abs() < 1e-6);
        let accepted = levelset_surface::publish(source.dimensions, vertices, source.receipt.exact_area_fine).unwrap();
        assert!(sample_scalar(&accepted, [7.2, 8.0]).unwrap() < 0.0);
        assert!(sample_scalar(&accepted, [7.2, 5.5]).unwrap() < 0.0);
        assert!(sample_scalar(&accepted, [7.2, 4.5]).unwrap() > 0.0);
        assert_eq!(accepted.receipt.unresolved_fine_cells, 0);
        assert!((accepted.receipt.represented_area_fine - source.receipt.represented_area_fine).abs() < 1.0);
    }

    #[test]
    fn contourless_phase_advects_its_finite_scalar_without_redistance() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let fields = velocity_fields(&graph, [0.25, 0.0]);
        let source = levelset_surface::publish([8, 8], vec![-2.0; 81], 64.0).unwrap();
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0, &mut receipt).unwrap();
        assert!(vertices.iter().all(|value| *value == -2.0));
        assert_eq!(receipt.redistanced_samples, 0);
        assert_eq!(receipt.redistance_fallback_samples, 0);
    }

    #[test]
    fn corner_traces_measure_affine_expansion_instead_of_rigid_translation() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let mut fields = velocity_fields(&graph, [0.0, 0.0]);
        fields.cell_velocity = graph.cells.iter().flat_map(|cell| [
            0.5 * (cell.center[0] - 4.0),
            0.5 * (cell.center[1] - 4.0),
        ]).collect();
        fields.face_velocity = graph.rows.iter().map(|row| {
            0.5 * (row.center[row.axis as usize] - 4.0)
        }).collect();
        let (footprints, _, _) = trace_rk2(&graph, &fields, 0.1);
        let id = cell_at(&graph, [3.0, 3.0]);
        let triangles = footprint_triangles(footprints[id], &graph.cells[id]);
        let area: f64 = triangles.iter().map(|triangle| polygon_area(triangle)).sum();
        assert!(area < graph.cells[id].measure as f64, "{area}");
        assert!(area > 0.8 * graph.cells[id].measure as f64, "{area}");
    }

    #[test]
    fn folded_corner_footprint_falls_back_to_finite_rigid_box() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let id = cell_at(&graph, [3.0, 3.0]);
        let folded = [
            [3.0, 3.0], [4.0, 4.0], [4.0, 3.0], [3.0, 4.0], [3.5, 3.5],
        ];
        let triangles = footprint_triangles(folded, &graph.cells[id]);
        let area: f64 = triangles.iter().map(|triangle| polygon_area(triangle)).sum();
        assert!(area.is_finite());
        assert!((area - graph.cells[id].measure as f64).abs() < 1e-12, "{area}");
    }

    #[test]
    fn vertex_redistance_preserves_the_raw_zero_contour() {
        let source = flat_non_distance_surface([16, 16]);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = redistance_vertices(
            source.dimensions,
            source.vertex_phi_fine.clone(),
            source.receipt.exact_area_fine,
            &mut receipt,
        ).unwrap();
        let accepted = levelset_surface::publish(
            source.dimensions,
            vertices,
            source.receipt.exact_area_fine,
        ).unwrap();
        let source_y = source.segments_fine.chunks_exact(4)
            .map(|segment| segment[1]).sum::<f32>() / (source.segments_fine.len() / 4) as f32;
        assert!(accepted.segments_fine.chunks_exact(4).all(|segment| {
            (segment[1] - source_y).abs() < 1.0e-5
                && (segment[3] - source_y).abs() < 1.0e-5
        }));
        assert!(receipt.redistanced_samples > 0);
    }
}
