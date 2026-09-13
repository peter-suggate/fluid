//! Atomic accepted/candidate generation transactions and bounded leaf reuse.
use crate::scene::SceneState;
use crate::topology::CompiledTopology;
use crate::transfer::{transfer_fields, NewAirCoverage, TransferError};
use crate::{collocate_velocity, reconstruct_interfaces, ValidationError};
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
    pub maximum_slice_leaves: usize,
    pub free_leaf_ids: Vec<u32>,
    pub authored_leaf_ids: BTreeSet<u32>,
    pub retirement: RetirementReceipt,
}
impl LeafArena {
    pub fn new(
        topology: &CompiledTopology<2>,
        page_budget: Option<u32>,
    ) -> Result<Self, ValidationError> {
        let nx = (topology.graph.dimensions[0] as u32).div_ceil(8);
        let ny = (topology.graph.dimensions[1] as u32).div_ceil(8);
        let default_budget = nx
            .checked_mul(ny)
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
            maximum_slice_leaves: (topology.bricks.len() + budget as usize).max(1),
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

/// Build and validate all candidate fields before replacing the accepted
/// state. Any capacity or geometry error leaves both state and arena intact.
pub fn commit_candidate(
    state: &mut SceneState<2>,
    candidate: SceneState<2>,
    arena: &mut LeafArena,
) -> Result<(), TransferError> {
    let (next, next_arena) = prepare_candidate(state, candidate, arena)?;
    *state = next;
    *arena = next_arena;
    Ok(())
}

pub fn prepare_candidate(
    state: &SceneState<2>,
    mut candidate: SceneState<2>,
    arena: &LeafArena,
) -> Result<(SceneState<2>, LeafArena), TransferError> {
    if candidate.topology.graph.topology_generation != state.topology.graph.topology_generation + 1
    {
        return Err(failure("candidate generation is not consecutive"));
    }
    if candidate
        .topology
        .bricks
        .iter()
        .any(|b| b.seed.id >= arena.capacity)
        || candidate.topology.graph.cells.len() > arena.capacity as usize * 64
        || candidate.topology.graph.rows.len() > arena.capacity as usize * 144
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
            let lo = [
                b.seed.coordinate[0] as f32 * 8.0,
                b.seed.coordinate[1] as f32 * 8.0,
            ];
            let span = b.seed.span_bricks as f32 * 8.0;
            NewAirCoverage {
                minimum_fine: lo,
                maximum_exclusive_fine: [lo[0] + span, lo[1] + span],
            }
        })
        .collect();
    let moved = transfer_fields(
        &state.topology.graph,
        &candidate.topology.graph,
        &state.fields,
        &candidate.fields.capacity,
        &new_air,
    )?;
    let prior_member: HashMap<_, _> = state
        .topology
        .graph
        .cells
        .iter()
        .filter_map(|c| {
            c.stable_id
                .map(|stable| (stable, state.fields.pressure_member[c.id as usize]))
        })
        .collect();
    let f = &mut candidate.fields;
    f.density = moved.density;
    f.gamma = moved.gamma;
    f.pressure = moved.pressure;
    f.cell_velocity = moved.cell_velocity;
    f.face_velocity = moved.face_velocity;
    f.capacity = moved.capacity;
    f.interface_normal = moved.interface_normal;
    f.interface_offset = moved.interface_offset;
    f.pressure_member = candidate
        .topology
        .graph
        .cells
        .iter()
        .map(|c| {
            c.stable_id
                .and_then(|id| prior_member.get(&id).copied())
                .unwrap_or(0)
        })
        .collect();
    f.pressure_row_member.fill(0);
    f.pressure_rhs.fill(0.0);
    f.pressure_diagonal.fill(0.0);
    f.extension_depth.fill(255);
    f.frame_dt = state.fields.frame_dt;
    f.acceleration_fine = state.fields.acceleration_fine;
    f.fault = None;
    reconstruct_interfaces(&candidate.topology.graph, f)?;
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
        let is_active = next.is_none_or(|b| b.seed.active);
        if next.is_none_or(|b| b.seed.resolution != brick.seed.resolution)
            || brick.seed.active != is_active
        {
            retirement.topology_changed_brick_ids.push(brick.seed.id)
        }
        if brick.seed.active && !is_active {
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
    use crate::scene::{compile_scene_2d, SceneDescription};
    fn state(bricks: serde_json::Value) -> SceneState<2> {
        let description:SceneDescription=serde_json::from_value(serde_json::json!({
        "schemaVersion":1,"dimension":2,"dimensions":[16,8,1],"cellSizeM":0.05,"dtS":1.0/30.0,"densityKgM3":998.2,
        "boundaries":["closed","closed","closed","closed","closed","closed"],"bricks":bricks})).unwrap();
        compile_scene_2d(description).unwrap()
    }
    fn brick(id: u32, key: u32, x: i32, active: bool) -> serde_json::Value {
        serde_json::json!({"id":id,"key":key,"coordinate":[x,0,0],"resolution":2,"active":active})
    }
    #[test]
    fn dynamic_leaf_reuse_waits_for_publication() {
        let mut accepted = state(serde_json::json!([brick(0, 0, 0, true)]));
        let mut arena = LeafArena::new(&accepted.topology, Some(1)).unwrap();
        let mut grow = state(serde_json::json!([
            brick(0, 0, 0, true),
            brick(1, 1, 1, true)
        ]));
        grow.topology.graph.topology_generation = 2;
        commit_candidate(&mut accepted, grow, &mut arena).unwrap();
        let mut retire = state(serde_json::json!([
            brick(0, 0, 0, true),
            brick(1, 1, 1, false)
        ]));
        retire.topology.graph.topology_generation = 3;
        commit_candidate(&mut accepted, retire, &mut arena).unwrap();
        assert!(arena.free_leaf_ids.is_empty());
        assert_eq!(arena.retirement.pending_dynamic_release_ids, vec![1]);
        arena.release_after_publication().unwrap();
        assert_eq!(arena.free_leaf_ids, vec![1]);
        let mut reuse = state(serde_json::json!([
            brick(0, 0, 0, true),
            brick(1, 19, 1, true)
        ]));
        reuse.topology.graph.topology_generation = 4;
        commit_candidate(&mut accepted, reuse, &mut arena).unwrap();
        assert!(arena.free_leaf_ids.is_empty());
        assert_eq!(accepted.topology.bricks[1].seed.key, 19);
    }
    #[test]
    fn failed_capacity_transfer_preserves_accepted_generation_and_fields() {
        let mut accepted = state(serde_json::json!([brick(0, 0, 0, true)]));
        accepted.fields.density.fill(1.0);
        let before = accepted.fields.clone();
        let mut arena = LeafArena::new(&accepted.topology, Some(0)).unwrap();
        let mut candidate = state(serde_json::json!([brick(0, 0, 0, true)]));
        candidate.topology.graph.topology_generation = 2;
        candidate.fields.capacity.fill(0.0);
        assert!(commit_candidate(&mut accepted, candidate, &mut arena).is_err());
        assert_eq!(accepted.topology.graph.topology_generation, 1);
        assert_eq!(accepted.fields, before);
        assert_eq!(arena.retirement.generation, 1);
    }
}
