//! Exact 3-D atomic accepted/candidate generation transactions and bounded leaf reuse.
use crate::scene::SceneState;
use crate::topology::CompiledTopology;
use crate::transfer3d::{plan_transfer_3d, transfer_fields_3d, NewAirCoverage3d, TransferError};
use crate::{collocate_velocity, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetirementReceipt {
    pub generation: u32,
    pub topology_changed_brick_ids: Vec<u32>,
    pub retired_brick_ids: Vec<u32>,
    pub reshaped_brick_ids: Vec<u32>,
    pub retired_residue_mass_fine_cells: f32,
    pub pending_dynamic_release_ids: Vec<u32>,
}
#[derive(Clone, Debug)]
pub struct LeafArena {
    pub capacity: u32,
    pub maximum_volume_leaves: usize,
    pub free_leaf_ids: Vec<u32>,
    pub authored_leaf_ids: BTreeSet<u32>,
    pub retirement: RetirementReceipt,
}
impl LeafArena {
    pub fn new_3d(
        topology: &CompiledTopology<3>,
        page_budget: Option<u32>,
    ) -> Result<Self, ValidationError> {
        let [nx, ny, nz] = topology.graph.dimensions.map(|v| (v as u32).div_ceil(8));
        let default_budget = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| ValidationError("leaf budget overflow".into()))?
            .saturating_sub(topology.bricks.len() as u32);
        let budget = page_budget.unwrap_or(default_budget);
        let high = topology
            .bricks
            .iter()
            .map(|b| b.seed.id)
            .max()
            .map_or(Some(0), |id| id.checked_add(1))
            .ok_or_else(|| ValidationError("leaf id overflow".into()))?;
        let capacity = high
            .checked_add(budget)
            .ok_or_else(|| ValidationError("leaf budget overflow".into()))?
            .max(1);
        Ok(Self {
            capacity,
            maximum_volume_leaves: (topology.bricks.len() + budget as usize).max(1),
            free_leaf_ids: Vec::new(),
            authored_leaf_ids: topology.bricks.iter().map(|b| b.seed.id).collect(),
            retirement: RetirementReceipt {
                generation: topology.graph.topology_generation,
                ..RetirementReceipt::default()
            },
        })
    }
    /// A released leaf may still be described as inactive in the current cold
    /// directory. The next planner removes its old entry before claiming it.
    pub fn release_after_publication(&mut self) -> Result<(), ValidationError> {
        if self.retirement.retired_residue_mass_fine_cells != 0.0 {
            return Err(ValidationError(
                "cannot release a leaf containing numerical residue".into(),
            ));
        }
        for &id in &self.retirement.pending_dynamic_release_ids {
            if id >= self.capacity
                || self.authored_leaf_ids.contains(&id)
                || self.free_leaf_ids.contains(&id)
            {
                return Err(ValidationError(
                    "invalid or repeated dynamic leaf release".into(),
                ));
            }
        }
        self.free_leaf_ids
            .extend(self.retirement.pending_dynamic_release_ids.drain(..));
        Ok(())
    }
}

fn failure(message: &str) -> TransferError {
    ValidationError(message.into()).into()
}

/// Transfer an accepted per-volume cell rate through the already-certified
/// conservative overlap plan. The outer-frame source budget is frozen before
/// a projected-support regrid, so rebuilding candidate geometry must not
/// replace this plane with construction zeros.
fn transfer_frozen_cell_rate(
    source: &crate::Graph,
    target: &crate::Graph,
    values: &[f32],
    plan: &crate::transfer3d::TransferPlan,
) -> Result<Vec<f32>, TransferError> {
    if values.len() != source.cells.len() {
        return Err(failure(
            "frozen cell-rate length differs from accepted topology",
        ));
    }
    let mut result = vec![0.0; target.cells.len()];
    for cell in &target.cells {
        let id = cell.id as usize;
        let mut total = 0.0f32;
        let mut correction = 0.0f32;
        for entry in plan.cell_offsets[id] as usize..plan.cell_offsets[id + 1] as usize {
            let source_id = plan.cell_sources[entry];
            if source_id == u32::MAX {
                continue;
            }
            let value = values[source_id as usize] * plan.cell_areas[entry] - correction;
            let next = total + value;
            correction = (next - total) - value;
            total = next;
        }
        result[id] = total / cell.measure;
    }
    Ok(result)
}

/// Return the capacity authority that owns the accepted scalar image.
///
/// During geometric solid motion the scalar image still belongs to the
/// beginning-of-frame geometry until transport commits it onto the final
/// aperture.  A projected-support topology edit occurs inside that interval,
/// so its transfer must certify the accepted volume against the historical
/// capacity plane.  The candidate retains its final geometry below; this
/// temporary plane is only the generation-transfer authority.
fn transfer_epoch_capacity_3d(
    state: &SceneState<3>,
    candidate: &SceneState<3>,
    new_air: &[NewAirCoverage3d],
) -> Result<Option<Vec<f32>>, TransferError> {
    if !state.fields.solid_motion_active
        || state.fields.capacity_before.len() != state.topology.graph.cells.len()
    {
        return Ok(None);
    }
    let needs_historical_capacity = state.topology.graph.cells.iter().any(|cell| {
        let id = cell.id as usize;
        let amount = state.fields.density[id] * cell.measure;
        let current = state.fields.capacity[id] * cell.measure;
        let tolerance = 90.536_743e-7 * current;
        amount > current + tolerance
    });
    if !needs_historical_capacity {
        return Ok(None);
    }
    let plan = plan_transfer_3d(&state.topology.graph, &candidate.topology.graph, new_air)?;
    let mut result = vec![0.0; candidate.topology.graph.cells.len()];
    for target in &candidate.topology.graph.cells {
        let id = target.id as usize;
        let mut total = 0.0f32;
        let mut correction = 0.0f32;
        for entry in plan.cell_offsets[id] as usize..plan.cell_offsets[id + 1] as usize {
            let source = plan.cell_sources[entry];
            if source == u32::MAX {
                continue;
            }
            let value =
                state.fields.capacity_before[source as usize] * plan.cell_areas[entry] - correction;
            let next = total + value;
            correction = (next - total) - value;
            total = next;
        }
        result[id] = total / target.measure;
    }
    Ok(Some(result))
}

/// Build and validate all candidate fields before replacing the accepted
/// state. Any capacity or geometry error leaves both state and arena intact.
pub fn commit_candidate_3d(
    state: &mut SceneState<3>,
    candidate: SceneState<3>,
    arena: &mut LeafArena,
) -> Result<(), TransferError> {
    let (next, next_arena) = prepare_candidate_3d(state, candidate, arena)?;
    *state = next;
    *arena = next_arena;
    Ok(())
}

pub fn prepare_candidate_3d(
    state: &SceneState<3>,
    mut candidate: SceneState<3>,
    arena: &LeafArena,
) -> Result<(SceneState<3>, LeafArena), TransferError> {
    if candidate.topology.graph.topology_generation != state.topology.graph.topology_generation + 1
    {
        return Err(failure("candidate generation is not consecutive"));
    }
    if candidate
        .topology
        .bricks
        .iter()
        .any(|b| b.seed.id >= arena.capacity)
        || candidate.topology.graph.cells.len() > arena.capacity as usize * 512
        || candidate.topology.graph.rows.len() > arena.capacity as usize * 1728
    {
        return Err(failure("candidate exceeds the fixed physical leaf arena"));
    }
    let accepted: HashMap<_, _> = state
        .topology
        .bricks
        .iter()
        .map(|b| (b.seed.key, b))
        .collect();
    let new_air: Vec<_> = candidate
        .topology
        .bricks
        .iter()
        .filter(|b| b.seed.active && !accepted.get(&b.seed.key).is_some_and(|old| old.seed.active))
        .map(|b| {
            let lo = b.seed.coordinate.map(|v| v as f32 * 8.0);
            let span = b.seed.span_bricks as f32 * 8.0;
            NewAirCoverage3d {
                minimum_fine: lo,
                maximum_exclusive_fine: lo.map(|v| v + span),
            }
        })
        .collect();
    let final_capacity = candidate.fields.capacity.clone();
    let transfer_capacity = transfer_epoch_capacity_3d(state, &candidate, &new_air)?;
    let mut transfer_fields;
    let source_fields = if transfer_capacity.is_some() {
        transfer_fields = state.fields.clone();
        transfer_fields
            .capacity
            .clone_from(&state.fields.capacity_before);
        &transfer_fields
    } else {
        &state.fields
    };
    let moved = transfer_fields_3d(
        &state.topology.graph,
        &candidate.topology.graph,
        source_fields,
        transfer_capacity.as_deref().unwrap_or(&final_capacity),
        &new_air,
    )?;
    let source_rate = transfer_frozen_cell_rate(
        &state.topology.graph,
        &candidate.topology.graph,
        &state.fields.source_rate,
        &moved.plan,
    )?;
    let f = &mut candidate.fields;
    f.density = moved.density;
    f.gamma = moved.gamma;
    f.pressure = moved.pressure;
    f.cell_velocity = moved.cell_velocity;
    f.face_velocity = moved.face_velocity;
    f.capacity = final_capacity;
    if let Some(capacity_before) = transfer_capacity {
        f.capacity_before = capacity_before;
        f.capacity_after.clone_from(&f.capacity);
        let inverse_dt = if state.fields.frame_dt > 0.0 {
            1.0 / state.fields.frame_dt
        } else {
            0.0
        };
        for id in 0..f.capacity.len() {
            f.capacity_rate[id] = (f.capacity_after[id] - f.capacity_before[id]) * inverse_dt;
        }
        f.solid_motion_active = true;
    }
    f.interface_normal = moved.interface_normal;
    f.interface_offset = moved.interface_offset;
    f.source_rate = source_rate;
    // A corrected physical face-rate image belongs to one accepted topology
    // and one frozen outer-frame domain. Face ids and pressure membership may
    // both change across this transaction, so the next frame must rebuild it.
    f.subface_compatibility_rate.clear();
    // Candidate publication classifies the transferred image, exactly like
    // publishCandidateTopologyDeltaWork. Stable IDs describe ownership, not
    // phase membership: refinement creates wet children with new stable IDs.
    f.pressure_member = candidate
        .topology
        .graph
        .cells
        .iter()
        .map(|cell| {
            u8::from(
                f.density[cell.id as usize] / f.capacity[cell.id as usize].max(1e-6)
                    >= crate::numerics::LIQUID_ISOVALUE
                    || f.source_rate[cell.id as usize] > 0.0,
            )
        })
        .collect();
    f.pressure_row_member.fill(0);
    f.pressure_rhs.fill(0.0);
    f.pressure_diagonal.fill(0.0);
    f.extension_depth.fill(255);
    f.frame_dt = state.fields.frame_dt;
    f.acceleration_fine = state.fields.acceleration_fine;
    f.fault = None;
    crate::numerics3d::reconstruct_interfaces_3d(&candidate.topology.graph, f)?;
    collocate_velocity(&candidate.topology.graph, f);
    let next_by_key: HashMap<_, _> = candidate
        .topology
        .bricks
        .iter()
        .map(|b| (b.seed.key, b))
        .collect();
    let mut retirement = RetirementReceipt {
        generation: candidate.topology.graph.topology_generation,
        ..RetirementReceipt::default()
    };
    for brick in &state.topology.bricks {
        let next = next_by_key.get(&brick.seed.key);
        let fine_box = |seed: &crate::topology::BrickSeed| {
            let lo: [i64; 3] = std::array::from_fn(|axis| {
                (seed.coordinate[axis] as i64 * 8)
                    .clamp(0, state.description.dimensions[axis] as i64)
            });
            let hi: [i64; 3] = std::array::from_fn(|axis| {
                ((seed.coordinate[axis] as i64 + seed.span_bricks as i64) * 8)
                    .clamp(0, state.description.dimensions[axis] as i64)
            });
            (lo, hi)
        };
        let (brick_lo, brick_hi) = fine_box(&brick.seed);
        let brick_volume = (0..3)
            .map(|axis| (brick_hi[axis] - brick_lo[axis]).max(0) as u64)
            .product::<u64>();
        let covered_by_reshape = next.is_none()
            && candidate
                .topology
                .bricks
                .iter()
                .filter(|candidate| candidate.seed.active)
                .map(|candidate| {
                    let (lo, hi) = fine_box(&candidate.seed);
                    (0..3)
                        .map(|axis| {
                            (brick_hi[axis].min(hi[axis]) - brick_lo[axis].max(lo[axis])).max(0)
                                as u64
                        })
                        .product::<u64>()
                })
                .sum::<u64>()
                == brick_volume;
        let is_active = next.is_some_and(|b| b.seed.active) || covered_by_reshape;
        if next.is_none_or(|b| b.seed.resolution != brick.seed.resolution)
            || brick.seed.active != is_active
        {
            retirement.topology_changed_brick_ids.push(brick.seed.id)
        }
        if covered_by_reshape {
            if !arena.authored_leaf_ids.contains(&brick.seed.id) {
                retirement.pending_dynamic_release_ids.push(brick.seed.id);
            }
        } else if brick.seed.active && !is_active {
            retirement.retired_brick_ids.push(brick.seed.id);
            for id in brick.cell_range.clone() {
                retirement.retired_residue_mass_fine_cells += state.fields.density[id as usize]
                    * state.topology.graph.cells[id as usize].measure;
            }
            if !arena.authored_leaf_ids.contains(&brick.seed.id) {
                retirement.pending_dynamic_release_ids.push(brick.seed.id);
            }
        } else if next.is_some_and(|b| b.seed.resolution != brick.seed.resolution) {
            retirement.reshaped_brick_ids.push(brick.seed.id)
        }
    }
    if retirement.retired_residue_mass_fine_cells != 0.0 {
        return Err(failure("candidate retires nonzero fluid"));
    }
    let active_ids: BTreeSet<_> = candidate
        .topology
        .bricks
        .iter()
        .filter(|b| b.seed.active)
        .map(|b| b.seed.id)
        .collect();
    let mut next_arena = arena.clone();
    next_arena
        .free_leaf_ids
        .retain(|id| !active_ids.contains(id));
    next_arena.retirement = retirement;
    Ok((candidate, next_arena))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{compile_scene_3d, SceneDescription};

    fn state(resolution: u8, active: bool, density: Vec<f32>) -> SceneState<3> {
        let description: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":3,"dimensions":[8,8,8],"cellSizeM":0.05,
            "dtS":1.0/30.0,"densityKgM3":998.2,
            "boundaries":["closed","closed","closed","closed","closed","closed"],
            "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"resolution":resolution,
                "active":active,"density":density}]
        }))
        .unwrap();
        compile_scene_3d(description).unwrap()
    }

    fn parent_or_children(children: bool, generation: u32) -> SceneState<3> {
        parent_or_children_in(children, generation, [16, 16, 16])
    }

    fn parent_or_children_in(
        children: bool,
        generation: u32,
        dimensions: [u32; 3],
    ) -> SceneState<3> {
        let bricks = if children {
            let mut bricks = Vec::new();
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        bricks.push(serde_json::json!({
                            "id":1+x+2*y+4*z,"key":1+x+2*y+4*z,
                            "coordinate":[x,y,z],"spanBricks":1,
                            "resolution":1,"active":true,"density":[0.25]
                        }));
                    }
                }
            }
            bricks
        } else {
            vec![serde_json::json!({
                "id":0,"key":0,"coordinate":[0,0,0],"spanBricks":2,
                "resolution":1,"active":true,"density":[0.25]
            })]
        };
        let description: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":3,"dimensions":dimensions,"cellSizeM":0.05,
            "dtS":1.0/30.0,"densityKgM3":998.2,
            "boundaries":["closed","closed","closed","closed","closed","closed"],
            "bricks":bricks
        }))
        .unwrap();
        let mut state = compile_scene_3d(description).unwrap();
        state.topology.graph.topology_generation = generation;
        state
    }

    #[test]
    fn rerung_commits_one_atomic_conservative_generation() {
        let mut accepted = state(1, true, vec![0.25]);
        accepted.fields.subface_compatibility_rate =
            vec![0.125; accepted.topology.graph.subfaces.len()];
        vec![f32::EPSILON; accepted.topology.graph.subfaces.len()];
        let arena = LeafArena::new_3d(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(2, true, vec![0.0; 8]);
        candidate.topology.graph.topology_generation = 2;
        let (next, next_arena) = prepare_candidate_3d(&accepted, candidate, &arena).unwrap();
        let before = accepted.state_volume();
        let after = next.state_volume();
        assert_eq!(before.to_bits(), after.to_bits());
        assert_eq!(next.fields.density, vec![0.25; 8]);
        assert!(next.fields.subface_compatibility_rate.is_empty());
        assert_eq!(next_arena.retirement.reshaped_brick_ids, vec![0]);
        assert_eq!(next_arena.retirement.generation, 2);
    }

    #[test]
    fn refinement_reclassifies_every_transferred_wet_child_for_pressure() {
        let mut accepted = state(1, true, vec![1.0]);
        accepted.fields.pressure_member[0] = 1;
        let arena = LeafArena::new_3d(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(2, true, vec![0.0; 8]);
        candidate.topology.graph.topology_generation = 2;
        candidate.fields.pressure_member.fill(0);
        let (next, _) = prepare_candidate_3d(&accepted, candidate, &arena).unwrap();
        assert_eq!(next.fields.density, vec![1.0; 8]);
        assert_eq!(next.fields.pressure_member, vec![1; 8]);
        assert!(next
            .fields
            .pressure_row_member
            .iter()
            .all(|&member| member == 0));
        assert!(next.fields.pressure_rhs.iter().all(|&value| value == 0.0));
        assert!(next
            .fields
            .pressure_diagonal
            .iter()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn moving_cut_refinement_uses_the_scalar_epoch_capacity_without_losing_volume() {
        let mut accepted = state(1, true, vec![0.953125]);
        accepted.fields.capacity = vec![0.90625];
        accepted.fields.capacity_before = vec![1.0];
        accepted.fields.capacity_after = accepted.fields.capacity.clone();
        accepted.fields.capacity_rate = vec![-2.8125];
        accepted.fields.solid_motion_active = true;
        accepted.fields.frame_dt = 1.0 / 30.0;
        let before = accepted.state_volume();
        let arena = LeafArena::new_3d(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(2, true, vec![0.0; 8]);
        candidate.topology.graph.topology_generation = 2;
        candidate.fields.capacity.fill(0.90625);
        candidate.fields.capacity_before.fill(0.90625);
        candidate.fields.capacity_after.fill(0.90625);
        let (next, _) = prepare_candidate_3d(&accepted, candidate, &arena).unwrap();
        assert_eq!(next.state_volume().to_bits(), before.to_bits());
        assert_eq!(next.fields.capacity, vec![0.90625; 8]);
        assert_eq!(next.fields.capacity_before, vec![1.0; 8]);
        assert!(next
            .fields
            .density
            .iter()
            .all(|&density| density == 0.953125));
        assert!(next.fields.solid_motion_active);
    }

    #[test]
    fn refinement_preserves_the_frozen_source_rate_and_dry_membership() {
        let mut accepted = state(1, true, vec![0.0]);
        accepted.fields.source_rate[0] = 2.0;
        accepted.fields.pressure_member[0] = 1;
        let arena = LeafArena::new_3d(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(2, true, vec![0.0; 8]);
        candidate.topology.graph.topology_generation = 2;
        let (next, _) = prepare_candidate_3d(&accepted, candidate, &arena).unwrap();
        assert_eq!(next.fields.source_rate, vec![2.0; 8]);
        assert_eq!(next.fields.pressure_member, vec![1; 8]);
        let integrated = next
            .topology
            .graph
            .cells
            .iter()
            .map(|cell| next.fields.source_rate[cell.id as usize] * cell.measure)
            .sum::<f32>();
        assert_eq!(integrated, 2.0 * accepted.topology.graph.cells[0].measure);
    }

    #[test]
    fn nonempty_leaf_retirement_is_rejected_without_mutation() {
        let accepted = state(1, true, vec![0.25]);
        let arena = LeafArena::new_3d(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(1, false, vec![0.0]);
        candidate.topology.graph.topology_generation = 2;
        assert!(prepare_candidate_3d(&accepted, candidate, &arena).is_err());
        assert_eq!(accepted.topology.graph.topology_generation, 1);
        assert!(arena.retirement.retired_brick_ids.is_empty());
    }

    #[test]
    fn wet_split_merge_split_reuses_dynamic_ids_and_reserves_authored_parent() {
        let parent = parent_or_children(false, 1);
        let mut arena = LeafArena::new_3d(&parent.topology, Some(8)).unwrap();
        arena.free_leaf_ids = (1..=8).collect();

        let split = parent_or_children(true, 2);
        let (split, arena) = prepare_candidate_3d(&parent, split, &arena).unwrap();
        assert!(arena.free_leaf_ids.is_empty());
        assert_eq!(
            split.state_volume().to_bits(),
            parent.state_volume().to_bits()
        );

        let merged = parent_or_children(false, 3);
        let (merged, mut arena) = prepare_candidate_3d(&split, merged, &arena).unwrap();
        assert_eq!(
            merged.state_volume().to_bits(),
            parent.state_volume().to_bits()
        );
        assert!(arena.retirement.retired_brick_ids.is_empty());
        assert_eq!(
            arena.retirement.pending_dynamic_release_ids,
            (1..=8).collect::<Vec<_>>()
        );
        arena.release_after_publication().unwrap();
        assert_eq!(arena.free_leaf_ids, (1..=8).collect::<Vec<_>>());
        assert!(!arena.free_leaf_ids.contains(&0));

        let split_again = parent_or_children(true, 4);
        let (split_again, arena) = prepare_candidate_3d(&merged, split_again, &arena).unwrap();
        assert!(arena.free_leaf_ids.is_empty());
        assert_eq!(
            split_again.state_volume().to_bits(),
            parent.state_volume().to_bits()
        );
        assert!(arena.authored_leaf_ids.contains(&0));
    }

    #[test]
    fn clipped_boundary_children_fully_cover_and_release_after_merge() {
        let parent = parent_or_children_in(false, 1, [12, 16, 16]);
        let mut arena = LeafArena::new_3d(&parent.topology, Some(8)).unwrap();
        arena.free_leaf_ids = (1..=8).collect();
        let split = parent_or_children_in(true, 2, [12, 16, 16]);
        let (split, arena) = prepare_candidate_3d(&parent, split, &arena).unwrap();
        let merged = parent_or_children_in(false, 3, [12, 16, 16]);
        let (merged, mut arena) = prepare_candidate_3d(&split, merged, &arena).unwrap();
        assert_eq!(
            merged.state_volume().to_bits(),
            parent.state_volume().to_bits()
        );
        assert_eq!(
            arena.retirement.pending_dynamic_release_ids,
            (1..=8).collect::<Vec<_>>()
        );
        arena.release_after_publication().unwrap();
        assert_eq!(arena.free_leaf_ids, (1..=8).collect::<Vec<_>>());
    }

    trait Volume {
        fn state_volume(&self) -> f32;
    }
    impl Volume for SceneState<3> {
        fn state_volume(&self) -> f32 {
            self.topology
                .graph
                .cells
                .iter()
                .map(|c| self.fields.density[c.id as usize] * c.measure)
                .sum()
        }
    }
}
