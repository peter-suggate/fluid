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
            }
        }
        let determinant = mxx * myy - mxy * mxy;
        let scale = mxx.max(myy);
        let mut gradient = if scale > 1e-20 && determinant.abs() > 1e-7 * scale * scale {
            [(myy * bx - mxy * by) / determinant, (-mxy * bx + mxx * by) / determinant]
        } else { [0.0, 0.0] };
        let length = gradient[0].hypot(gradient[1]);
        if !(length > 1e-20 && length.is_finite()) { continue; }
        gradient[0] /= length; gradient[1] /= length;
        let extent = 0.5 * (gradient[0].abs() * c.widths[0] as f64
            + gradient[1].abs() * c.widths[1] as f64);
        let offset = -(phi[i] as f64) / length;
        if !offset.is_finite() || offset.abs() > extent { continue; }
        fields.interface_normal[2 * i] = gradient[0] as f32;
        fields.interface_normal[2 * i + 1] = gradient[1] as f32;
        fields.interface_offset[i] = offset as f32;
    }
    Ok(())
}

fn trace_rk2(graph: &Graph, fields: &Fields, dt: f32) -> (Vec<[f32; 2]>, f64, f64) {
    let mut landings = Vec::with_capacity(graph.cells.len());
    let (mut max_distance, mut max_courant) = (0.0_f64, 0.0_f64);
    for cell in &graph.cells {
        let start = [cell.center[0], cell.center[1]];
        let span = cell.widths[0].min(cell.widths[1]).max(1.0);
        let first = sample_support(graph, fields, start[0], start[1], span);
        let midpoint = [start[0] - 0.5 * dt * first[0], start[1] - 0.5 * dt * first[1]];
        let velocity = sample_support(graph, fields, midpoint[0], midpoint[1], span);
        let landing = [
            (start[0] - dt * velocity[0]).clamp(0.5, graph.dimensions[0] - 0.5),
            (start[1] - dt * velocity[1]).clamp(0.5, graph.dimensions[1] - 0.5),
        ];
        let distance = ((landing[0] - start[0]).powi(2) + (landing[1] - start[1]).powi(2)).sqrt() as f64;
        max_distance = max_distance.max(distance);
        max_courant = max_courant.max(distance / span as f64);
        landings.push(landing);
    }
    (landings, max_distance, max_courant)
}

fn advect_shared_phi(
    graph: &Graph,
    fields: &Fields,
    previous: &RdfSurface,
    dt: f32,
    receipt: &mut LevelSetVolumeReceipt,
) -> Result<Vec<f32>, ValidationError> {
    let [nx, ny] = previous.dimensions.map(|value| value as usize);
    if previous.vertex_phi_fine.len() != (nx + 1) * (ny + 1) {
        return Err(ValidationError("direct level-set vertex count does not match dimensions".into()));
    }
    let redistance = RedistanceField::new(previous)?;
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
            let departure = [
                (start[0] - dt * velocity[0]).clamp(0.0, graph.dimensions[0]),
                (start[1] - dt * velocity[1]).clamp(0.0, graph.dimensions[1]),
            ];
            let index = x + (nx + 1) * y;
            if departure == start {
                result.push(previous.vertex_phi_fine[index]);
            } else if let Some(value) = redistance.sample(departure) {
                receipt.redistanced_samples += 1;
                result.push(value);
            } else if let Some(value) = sample_scalar(previous, departure) {
                receipt.redistance_fallback_samples += 1;
                result.push(value);
            } else {
                return Err(ValidationError("direct level-set departure has no finite scalar".into()));
            }
        }
    }
    Ok(result)
}

fn raw_weights(
    graph: &Graph,
    landings: &[[f32; 2]],
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
    let mut rows = Vec::with_capacity(landings.len());
    for (receiver, &landing) in landings.iter().enumerate() {
        let cell = &graph.cells[receiver];
        if capacity.get(receiver).copied().unwrap_or(0.0) <= 1e-30 {
            rows.push(Vec::new());
            continue;
        }
        let displacement = [
            landing[0] as f64 - cell.center[0] as f64,
            landing[1] as f64 - cell.center[1] as f64,
        ];
        let minimum = [
            (cell.minimum[0] as f64 + displacement[0]).max(0.0),
            (cell.minimum[1] as f64 + displacement[1]).max(0.0),
        ];
        let maximum = [
            (cell.maximum[0] as f64 + displacement[0]).min(graph.dimensions[0] as f64),
            (cell.maximum[1] as f64 + displacement[1]).min(graph.dimensions[1] as f64),
        ];
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
            let width = maximum[0].min(source.maximum[0] as f64)
                - minimum[0].max(source.minimum[0] as f64);
            let height = maximum[1].min(source.maximum[1] as f64)
                - minimum[1].max(source.minimum[1] as f64);
            let overlap = width.max(0.0) * height.max(0.0);
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

pub fn advance(
    graph: &Graph,
    fields: &mut Fields,
    previous_surface: &RdfSurface,
    _rdf_topology: &RdfTopology,
    _rdf_support: &RdfSupport,
    phi: &mut Vec<f32>,
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
    let (landings, distance, courant) = trace_rk2(graph, fields, dt);
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
    let (mut weights, zero_support_donors) = raw_weights(graph, &landings, &capacity);
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
    for (i, cell) in graph.cells.iter().enumerate() {
        fields.density[i] = (next_volume[i] / cell.measure as f64) as f32;
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
            if fields.capacity[i] > 1e-8 {
                receipt.maximum_over_capacity_ratio = receipt.maximum_over_capacity_ratio
                    .max(fields.density[i] as f64 / fields.capacity[i] as f64 - 1.0);
            }
        }
    }
    receipt.volume_gather_nanoseconds = clock.elapsed();

    let clock = StageClock::start();
    let vertices = advect_shared_phi(graph, fields, previous_surface, dt, &mut receipt)?;
    receipt.phi_gather_nanoseconds = clock.elapsed();

    let clock = StageClock::start();
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
    fn uniformly_translated_circle_stays_single_signed_region() {
        let graph = graph([16, 16, 1], vec![
            brick(0, [0, 0, 0], 8), brick(1, [1, 0, 0], 8),
            brick(2, [0, 1, 0], 8), brick(3, [1, 1, 0], 8),
        ]);
        let fields = velocity_fields(&graph, [0.2, 0.0]);
        let source = circle_surface([16, 16], [7.0, 8.0], 3.0);
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0, &mut receipt).unwrap();
        let accepted = levelset_surface::publish(source.dimensions, vertices, source.receipt.exact_area_fine).unwrap();
        assert!(sample_scalar(&accepted, [7.2, 8.0]).unwrap() < 0.0);
        assert!(sample_scalar(&accepted, [7.2, 5.5]).unwrap() < 0.0);
        assert!(sample_scalar(&accepted, [7.2, 4.5]).unwrap() > 0.0);
        assert_eq!(accepted.receipt.unresolved_fine_cells, 0);
        assert!((accepted.receipt.represented_area_fine - source.receipt.represented_area_fine).abs() < 1.0);
    }

    #[test]
    fn contourless_phase_uses_only_finite_scalar_fallback_at_boundaries() {
        let graph = graph([8, 8, 1], vec![brick(0, [0, 0, 0], 8)]);
        let fields = velocity_fields(&graph, [0.25, 0.0]);
        let source = levelset_surface::publish([8, 8], vec![-2.0; 81], 64.0).unwrap();
        let mut receipt = LevelSetVolumeReceipt::default();
        let vertices = advect_shared_phi(&graph, &fields, &source, 1.0, &mut receipt).unwrap();
        assert!(vertices.iter().all(|value| *value == -2.0));
        assert_eq!(receipt.redistanced_samples, 0);
        assert!(receipt.redistance_fallback_samples > 0);
    }
}
