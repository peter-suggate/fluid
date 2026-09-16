//! Conservative volume sharpening toward an immutable direct level set.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::levelset_adaptive_distance::{AdaptiveDistance, RETURN_REACH};
use crate::levelset_redistance::sample_scalar;
use crate::levelset_surface;
use crate::presentation::RdfSurface;
use crate::{Fields, Graph, ValidationError};

const NONE: usize = usize::MAX;
const AMBIGUOUS: usize = usize::MAX - 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SharpeningReceipt {
    pub component_count: usize,
    #[serde(default)]
    pub adaptive_distance_cells: usize,
    #[serde(default)]
    pub far_relocated_volume: f64,
    pub ambiguous_cell_count: usize,
    /// Accepted V in cells without one unique nearby phi-liquid component.
    pub unassigned_volume: f64,
    pub initial_band_absolute_mismatch: f64,
    pub final_band_absolute_mismatch: f64,
    pub initial_distance_weighted_mismatch: f64,
    pub final_distance_weighted_mismatch: f64,
    pub initial_over_capacity_volume: f64,
    pub final_over_capacity_volume: f64,
    pub initial_over_capacity_count: usize,
    pub final_over_capacity_count: usize,
    pub relocated_volume: f64,
    pub donor_count: usize,
    pub receiver_count: usize,
    pub unresolved_eligible_residual: f64,
    pub maximum_relocation_distance: f64,
    pub cross_component_pair_count: usize,
    pub bound_violation_count: usize,
    pub global_conservation_residual: f64,
    pub maximum_component_conservation_residual: f64,
}

struct Dsu(Vec<usize>);
impl Dsu {
    fn new(n: usize) -> Self { Self((0..n).collect()) }
    fn find(&mut self, i: usize) -> usize {
        if self.0[i] != i { self.0[i] = self.find(self.0[i]); }
        self.0[i]
    }
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b { self.0[b] = a; }
    }
}

fn owner_raster(graph: &Graph) -> Vec<usize> {
    let nx = graph.dimensions[0] as usize;
    let ny = graph.dimensions[1] as usize;
    let mut owner = vec![NONE; nx * ny];
    for cell in &graph.cells {
        for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
            for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                owner[x + nx * y] = cell.id as usize;
            }
        }
    }
    owner
}

/// Label phi-liquid fine components and extend each label two finest cells
/// into phi-air. Equidistant collisions remain ambiguous rather than joining
/// distinct liquid regions.
fn region_labels(capacity: &[f32], fill: &[f32], nx: usize, ny: usize) -> (Vec<usize>, usize) {
    let mut dsu = Dsu::new(nx * ny);
    let liquid = |i: usize| capacity[i] > 0.0 && fill[i] > 0.0;
    for y in 0..ny { for x in 0..nx {
        let i = x + nx * y;
        if !liquid(i) { continue; }
        if x > 0 && liquid(i - 1) { dsu.join(i, i - 1); }
        if y > 0 && liquid(i - nx) { dsu.join(i, i - nx); }
    }}
    let mut roots = HashMap::new();
    let mut labels = vec![NONE; nx * ny];
    for i in 0..labels.len() { if liquid(i) {
        let root = dsu.find(i); let next = roots.len();
        labels[i] = *roots.entry(root).or_insert(next);
    }}
    let mut depth = vec![usize::MAX; labels.len()];
    let mut queue = VecDeque::new();
    for i in 0..labels.len() { if labels[i] != NONE { depth[i] = 0; queue.push_back(i); } }
    while let Some(i) = queue.pop_front() {
        if depth[i] >= 2 || labels[i] == AMBIGUOUS { continue; }
        let (x, y) = (i % nx, i / nx);
        for j in [x.checked_sub(1).map(|_| i - 1), (x + 1 < nx).then_some(i + 1),
            y.checked_sub(1).map(|_| i - nx), (y + 1 < ny).then_some(i + nx)].into_iter().flatten() {
            if capacity[j] <= 0.0 { continue; }
            let next = depth[i] + 1;
            if next < depth[j] { depth[j] = next; labels[j] = labels[i]; queue.push_back(j); }
            else if next == depth[j] && labels[j] != labels[i] { labels[j] = AMBIGUOUS; }
        }
    }
    (labels, roots.len())
}

fn cell_labels(graph: &Graph, fine: &[usize], nx: usize) -> (Vec<usize>, usize) {
    let mut result = vec![NONE; graph.cells.len()]; let mut ambiguous_count = 0;
    for cell in &graph.cells {
        let mut labels = HashSet::new(); let mut ambiguous = false;
        for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
            for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                let label = fine[x + nx * y];
                if label == AMBIGUOUS { ambiguous = true; }
                else if label != NONE { labels.insert(label); }
            }
        }
        if labels.len() > 1 { ambiguous = true; }
        result[cell.id as usize] = if ambiguous { AMBIGUOUS } else { labels.into_iter().next().unwrap_or(NONE) };
        if ambiguous { ambiguous_count += 1; }
    }
    (result, ambiguous_count)
}

fn physical_targets(graph: &Graph, fine_capacity: &[f32], fill: &[f32], owner: &[usize]) -> Vec<f64> {
    let mut target = vec![0.0; graph.cells.len()];
    for i in 0..fill.len() { if owner[i] != NONE {
        target[owner[i]] += fine_capacity[i] as f64 * fill[i] as f64;
    }}
    target
}

/// Move conservative cell volume toward the existing phi surface. `surface`
/// and `phi` are immutable by construction; callers publish the same contour.
pub fn sharpen_volume(
    graph: &Graph,
    fields: &Fields,
    fine_capacity: &[f32],
    surface: &RdfSurface,
    phi: &[f32],
    volume: &mut [f64],
) -> Result<SharpeningReceipt, ValidationError> {
    let [nx, ny] = surface.dimensions.map(|v| v as usize);
    if graph.dimension != 2 || fine_capacity.len() != nx * ny || volume.len() != graph.cells.len()
        || fields.capacity.len() != graph.cells.len()
        || phi.iter().any(|p| !p.is_finite())
        || phi.len() != graph.cells.len() || fine_capacity.iter().any(|v| !v.is_finite() || *v < 0.0)
        || volume.iter().any(|v| !v.is_finite() || *v < 0.0)
    { return Err(ValidationError("volume sharpening requires finite 2-D capacity, phi, and volume".into())); }
    let before = volume.to_vec();
    let owner = owner_raster(graph);
    let fill = levelset_surface::implied_fill_fine_cells(surface)?;
    let (fine_region, component_count) = region_labels(fine_capacity, &fill, nx, ny);
    let (seed_region, _) = cell_labels(graph, &fine_region, nx);
    let target = physical_targets(graph, fine_capacity, &fill, &owner);
    let capacity: Vec<f64> = graph.cells.iter().enumerate()
        .map(|(i, cell)| fields.capacity[i] as f64 * cell.measure as f64).collect();
    let mut distance = AdaptiveDistance::build(graph, fields, surface, fine_capacity,
        &owner, &seed_region, component_count, &target);
    let near = |i: usize| seed_region[i] < component_count && phi[i].abs() as f64
        <= 2.0 * graph.cells[i].widths[0].max(graph.cells[i].widths[1]) as f64;
    let region: Vec<_> = (0..volume.len()).map(|i|
        if seed_region[i] != NONE { seed_region[i] } else { distance.component[i] }).collect();
    distance.component.clone_from(&region);
    let ambiguous_cell_count = region.iter().filter(|&&r| r == AMBIGUOUS).count();
    let mut residual: Vec<f64> = volume.iter().zip(&target).map(|(v, h)| v - h).collect();
    let in_band = |i: usize| region[i] < component_count
        && (near(i) || distance.signed_distance[i].is_finite());
    let mut receipt = SharpeningReceipt { component_count, ambiguous_cell_count, ..Default::default() };
    receipt.adaptive_distance_cells = distance.distance.iter().filter(|d| d.is_finite()).count();
    for i in 0..volume.len() {
        if region[i] >= component_count && volume[i] > 0.0 { receipt.unassigned_volume += volume[i]; }
        if in_band(i) { receipt.initial_band_absolute_mismatch += residual[i].abs(); }
        receipt.initial_distance_weighted_mismatch += residual[i].abs() * phi[i].abs() as f64;
        let excess = (volume[i] - capacity[i]).max(0.0);
        receipt.initial_over_capacity_volume += excess;
        if excess > 0.0 { receipt.initial_over_capacity_count += 1; }
    }
    // Gate diffuse donor islands, rather than individual cells, at half of the
    // smallest cell in that island so thin isolated material is not erased.
    let mut donors: Vec<_> = (0..volume.len()).filter(|&i| in_band(i) && residual[i] > 0.0).collect();
    // Complete established near-surface sharpening before the new far return.
    donors.sort_by_key(|&i| (!near(i), i));
    let maximum_width = graph.cells.iter().map(|c| c.widths[0].max(c.widths[1]) as f64)
        .fold(1.0_f64, f64::max);
    let mut donor_dsu = Dsu::new(volume.len());
    let donor_set: HashSet<_> = donors.iter().copied().collect();
    for &i in &donors { for &(j, _) in &distance.edges[i] {
        if donor_set.contains(&j) && region[j] == region[i] { donor_dsu.join(i, j); }
    }}
    let mut island_amount: HashMap<usize, f64> = HashMap::new();
    for &i in &donors {
        let root = donor_dsu.find(i);
        *island_amount.entry(root).or_insert(0.0) += residual[i];
    }
    let mut donor_used = HashSet::new(); let mut receiver_used = HashSet::new();
    let before_component: Vec<f64> = (0..component_count).map(|r| (0..volume.len())
        .filter(|&i| region[i] == r).map(|i| volume[i]).sum()).collect();
    for donor in donors {
        let root = donor_dsu.find(donor);
        // Coordinates are measured in finest-cell units, so this gate is
        // independent of the adaptive rung containing the diffuse island.
        if island_amount[&root] < 0.5 { continue; }
        let r = region[donor];
        let center = graph.cells[donor].center;
        let width = graph.cells[donor].widths[0].max(graph.cells[donor].widths[1]) as f64;
        let (landing, normal) = if near(donor) {
            let sample = |dx: f32, dy: f32| sample_scalar(surface, [
                (center[0]+dx).clamp(0.0,graph.dimensions[0]),
                (center[1]+dy).clamp(0.0,graph.dimensions[1]),
            ]).unwrap_or(phi[donor]);
            let gradient = [(sample(0.5,0.0)-sample(-0.5,0.0)) as f64,
                (sample(0.0,0.5)-sample(0.0,-0.5)) as f64];
            let length = gradient[0].hypot(gradient[1]);
            if length <= 1e-12 || !length.is_finite() { continue; }
            let sign = if phi[donor] >= 0.0 {1.0} else {-1.0};
            let inward = gradient.map(|v| -sign*v/length);
            let travel = (phi[donor].abs() as f64).min(2.0*width);
            ([center[0] as f64+inward[0]*travel, center[1] as f64+inward[1]*travel], inward)
        } else {
            let landing = distance.closest[donor];
            let vector = [landing[0]-center[0] as f64, landing[1]-center[1] as f64];
            let length = vector[0].hypot(vector[1]).max(1e-12);
            (landing, vector.map(|v| v/length))
        };
        let reach = if near(donor) {4.0*maximum_width} else {RETURN_REACH};
        let paths = distance.paths(donor, reach);
        let mut receivers: Vec<_> = (0..volume.len()).filter(|&j| j != donor && in_band(j)
            && (!near(donor) || near(j)) && region[j] == r && residual[j] < 0.0 && target[j] > 0.0)
            .filter_map(|j| {
                let dx = graph.cells[j].center[0] as f64 - graph.cells[donor].center[0] as f64;
                let dy = graph.cells[j].center[1] as f64 - graph.cells[donor].center[1] as f64;
                if !paths[j].is_finite() { return None; }
                let actual = if near(donor) { dx.hypot(dy) } else { paths[j] };
                let landing_distance = (graph.cells[j].center[0] as f64 - landing[0])
                    .hypot(graph.cells[j].center[1] as f64 - landing[1]);
                let radius = 2.0 * width.max(
                    graph.cells[j].widths[0].max(graph.cells[j].widths[1]) as f64);
                if (near(donor) && actual > radius) || landing_distance >= radius { return None; }
                let alignment = if actual > 1e-12 {
                    ((dx * normal[0] + dy * normal[1]) / actual).max(0.0)
                } else { 1.0 };
                let kernel = (1.0 - landing_distance / radius).powi(2) * (0.25 + 0.75 * alignment);
                Some((j, actual, kernel))
            }).collect();
        receivers.sort_by_key(|entry| entry.0);
        for _ in 0..4 {
            let weighted: Vec<_> = receivers.iter().filter_map(|&(receiver, distance, kernel)| {
                let room = (-residual[receiver])
                    .min((capacity[receiver] - volume[receiver]).max(0.0));
                (room > 0.0 && kernel > 0.0).then_some((receiver, distance, room, kernel * room))
            }).collect();
            let weight_sum: f64 = weighted.iter().map(|entry| entry.3).sum();
            let budget = residual[donor];
            if budget <= 1e-12 || weight_sum <= 0.0 { break; }
            for (receiver, distance, room, weight) in weighted {
                let moved = room.min(budget * weight / weight_sum).min(residual[donor]);
                if moved <= 0.0 { continue; }
                volume[donor] -= moved; volume[receiver] += moved;
                residual[donor] -= moved; residual[receiver] += moved;
                receipt.relocated_volume += moved;
                if !near(donor) { receipt.far_relocated_volume += moved; }
                receipt.maximum_relocation_distance = receipt.maximum_relocation_distance.max(distance);
                donor_used.insert(donor); receiver_used.insert(receiver);
            }
        }
    }
    receipt.donor_count = donor_used.len(); receipt.receiver_count = receiver_used.len();
    for i in 0..volume.len() {
        if in_band(i) {
            receipt.final_band_absolute_mismatch += residual[i].abs();
            if residual[i] > 0.0 { receipt.unresolved_eligible_residual += residual[i]; }
        }
        receipt.final_distance_weighted_mismatch += residual[i].abs() * phi[i].abs() as f64;
        let excess = (volume[i] - capacity[i]).max(0.0);
        receipt.final_over_capacity_volume += excess;
        if excess > 0.0 { receipt.final_over_capacity_count += 1; }
        if volume[i] < -1e-12 || volume[i] > before[i].max(capacity[i]) + 1e-12 {
            receipt.bound_violation_count += 1;
        }
    }
    let after_mass: f64 = volume.iter().sum(); let before_mass: f64 = before.iter().sum();
    receipt.global_conservation_residual = after_mass - before_mass;
    for r in 0..component_count {
        let after: f64 = (0..volume.len()).filter(|&i| region[i] == r).map(|i| volume[i]).sum();
        receipt.maximum_component_conservation_residual = receipt.maximum_component_conservation_residual
            .max((after - before_component[r]).abs());
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};

    fn fixture() -> (Graph, Fields, RdfSurface, Vec<f32>) {
        let brick = BrickSeed { id: 0, key: 0, coordinate: [0, 0, 0], span_bricks: 1,
            resolution: 8, active: true, density: Vec::new(), gamma: Vec::new(),
            refinement_region_scale: None };
        let graph = compile_topology::<2>(TopologySeed { dimensions: [8, 8, 1], generation: 1,
            sparse_air_phi: 0.5, boundaries: [BoundaryMode::Closed; 6], bricks: vec![brick] })
            .unwrap().graph;
        let vertices = (0..=8).flat_map(|y| (0..=8).map(move |_| y as f32 - 4.0)).collect();
        let surface = levelset_surface::publish([8, 8], vertices, 32.0).unwrap();
        let phi = graph.cells.iter().map(|cell| cell.center[1] - 4.0).collect();
        let mut fields = Fields::default();
        fields.capacity = vec![1.0; graph.cells.len()];
        fields.interface_normal = (0..graph.cells.len()).flat_map(|_| [0.0, 1.0]).collect();
        (graph, fields, surface, phi)
    }

    #[test]
    fn matching_volume_is_an_exact_no_op_for_volume_and_surface() {
        let (graph, fields, surface, phi) = fixture();
        let before_surface = surface.clone();
        let fill = levelset_surface::implied_fill_fine_cells(&surface).unwrap();
        let mut volume: Vec<f64> = fill.iter().map(|v| *v as f64).collect();
        let before = volume.clone();
        let receipt = sharpen_volume(&graph, &fields, &vec![1.0; 64], &surface, &phi, &mut volume).unwrap();
        assert_eq!(volume, before);
        assert_eq!(surface.vertex_phi_fine, before_surface.vertex_phi_fine);
        assert_eq!(surface.segments_fine, before_surface.segments_fine);
        assert_eq!(receipt.relocated_volume, 0.0);
        assert_eq!(receipt.global_conservation_residual, 0.0);
    }

    #[test]
    fn diffuse_under_capacity_air_volume_moves_to_local_phi_receiver() {
        let (graph, fields, surface, phi) = fixture();
        let fill = levelset_surface::implied_fill_fine_cells(&surface).unwrap();
        let mut volume: Vec<f64> = fill.iter().map(|v| *v as f64).collect();
        let donor = graph.cells.iter().position(|c| c.center[0] == 3.5 && c.center[1] == 5.5).unwrap();
        let receiver = graph.cells.iter().position(|c| c.center[0] == 3.5 && c.center[1] == 3.5).unwrap();
        volume[donor] = 0.75;
        volume[receiver] = 0.25;
        let before_mass: f64 = volume.iter().sum();
        let receipt = sharpen_volume(&graph, &fields, &vec![1.0; 64], &surface, &phi, &mut volume).unwrap();
        assert!(receipt.relocated_volume > 0.7, "{}", receipt.relocated_volume);
        assert!(volume[donor] < 0.05 && volume[receiver] > 0.95);
        assert!(receipt.final_band_absolute_mismatch < receipt.initial_band_absolute_mismatch);
        assert!(receipt.final_distance_weighted_mismatch < receipt.initial_distance_weighted_mismatch);
        assert_eq!(receipt.cross_component_pair_count, 0);
        assert_eq!(receipt.bound_violation_count, 0);
        assert!((volume.iter().sum::<f64>() - before_mass).abs() < 1e-12);
        assert!(receipt.maximum_relocation_distance <= 2.0);
    }
}
