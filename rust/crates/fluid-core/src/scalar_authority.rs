//! Final scalar packet publication (FSM1) for the reduced two-dimensional world.

use crate::kernels::{add, div, mul};
use crate::numerics::{reconstruct_interfaces, LIQUID_ISOVALUE};
use crate::topology::CompiledTopology;
use crate::types::{Fields, Graph, NumericalFault, RowKind, ValidationError};
use serde::Serialize;

pub const INVALID_SCALAR_PACKET: u32 = u32::MAX;
const VOLUME_ROUNDOFF_RATIO: f32 = 9.536_743_164_062_5e-7;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalarCellAddress {
    pub packet: u32,
    pub lane: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScalarPublicationReceipt {
    pub generation: u32,
    pub topology_generation: u32,
    pub topology_slot: u8,
    pub changed_cell_count: u32,
    pub nonexact_cell_count: u32,
    pub bulk_cell_count: u32,
    pub flip_cell_count: u32,
    pub fault: u32,
    pub first_fault_packet: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarAuthority {
    pub packet_capacity: usize,
    pub changed_low: Vec<u32>,
    pub changed_high: Vec<u32>,
    pub nonexact_low: Vec<u32>,
    pub nonexact_high: Vec<u32>,
    pub bulk_low: Vec<u32>,
    pub bulk_high: Vec<u32>,
    pub flip_low: Vec<u32>,
    pub flip_high: Vec<u32>,
    pub source_density: Vec<f32>,
    pub source_gamma: Vec<f32>,
    pub receipt: ScalarPublicationReceipt,
}

impl ScalarAuthority {
    pub fn new(
        leaf_capacity: usize,
        cell_count: usize,
        topology_generation: u32,
        topology_slot: u8,
    ) -> Self {
        let packet_capacity = leaf_capacity.max(1) * 64;
        let plane = || vec![0u32; packet_capacity];
        Self {
            packet_capacity,
            changed_low: plane(),
            changed_high: plane(),
            nonexact_low: plane(),
            nonexact_high: plane(),
            bulk_low: plane(),
            bulk_high: plane(),
            flip_low: plane(),
            flip_high: plane(),
            source_density: vec![0.0; cell_count],
            source_gamma: vec![0.0; cell_count],
            receipt: ScalarPublicationReceipt {
                topology_generation,
                topology_slot,
                first_fault_packet: INVALID_SCALAR_PACKET,
                ..Default::default()
            },
        }
    }

    pub fn publish(
        &mut self,
        graph: &Graph,
        addresses: &[ScalarCellAddress],
        fields: &Fields,
        source_density: &[f32],
        source_gamma: &[f32],
        generation: u32,
        topology_generation: u32,
        topology_slot: u8,
        has_rigid_bodies: bool,
    ) -> Result<&ScalarPublicationReceipt, ValidationError> {
        let n = graph.cells.len();
        if source_density.len() != n || source_gamma.len() != n || addresses.len() != n {
            return Err(ValidationError(
                "scalar source banks/addresses do not match accepted cells".into(),
            ));
        }
        for plane in [
            &mut self.changed_low,
            &mut self.changed_high,
            &mut self.nonexact_low,
            &mut self.nonexact_high,
            &mut self.bulk_low,
            &mut self.bulk_high,
            &mut self.flip_low,
            &mut self.flip_high,
        ] {
            plane.fill(0);
        }
        self.source_density = source_density.to_vec();
        self.source_gamma = source_gamma.to_vec();
        let (mut changed_count, mut nonexact_count, mut bulk_count, mut flip_count) = (0, 0, 0, 0);
        let (mut fault, mut first) = (0, INVALID_SCALAR_PACKET);
        for cell in &graph.cells {
            let id = cell.id as usize;
            let address = addresses[id];
            let packet = address.packet as usize;
            if packet >= self.packet_capacity {
                fault = 1;
                first = address.packet;
                continue;
            }
            let lane = address.lane as usize;
            let mask = 1u32 << (lane & 31);
            let changed = fields.density[id].to_bits() != self.source_density[id].to_bits()
                || fields.gamma[id].to_bits() != self.source_gamma[id].to_bits();
            let exact = fields.capacity[id].to_bits() == 0x3f80_0000
                && self.source_gamma[id].to_bits() == 0x3f80_0000
                && fields.gamma[id].to_bits() == 0x3f80_0000
                && self.source_density[id].to_bits() == fields.density[id].to_bits()
                && matches!(fields.density[id].to_bits(), 0 | 0x3f80_0000);
            let bulk = !has_rigid_bodies
                && fields.capacity[id] as f64 >= 1.0_f64 - 1e-6_f64
                && Fields::optional_cell(&fields.characteristic_clearance, id, 0.0) > 0.0
                && (fields.density[id] as f64 - 1.0).abs() <= 0.005
                && (fields.gamma[id] as f64 - 1.0).abs() <= 0.005;
            let flip = pressure_membership_predicate(graph, fields, id)
                != (fields.pressure_member[id] != 0);
            if changed {
                if lane < 32 {
                    self.changed_low[packet] |= mask
                } else {
                    self.changed_high[packet] |= mask
                }
                changed_count += 1
            }
            if !exact {
                if lane < 32 {
                    self.nonexact_low[packet] |= mask
                } else {
                    self.nonexact_high[packet] |= mask
                }
                nonexact_count += 1
            }
            if bulk {
                if lane < 32 {
                    self.bulk_low[packet] |= mask
                } else {
                    self.bulk_high[packet] |= mask
                }
                bulk_count += 1;
                self.source_density[id] = fields.density[id];
                self.source_gamma[id] = fields.gamma[id];
            }
            if flip {
                if lane < 32 {
                    self.flip_low[packet] |= mask
                } else {
                    self.flip_high[packet] |= mask
                }
                flip_count += 1
            }
        }
        self.receipt = ScalarPublicationReceipt {
            generation: generation.max(1),
            topology_generation,
            topology_slot,
            changed_cell_count: changed_count,
            nonexact_cell_count: nonexact_count,
            bulk_cell_count: bulk_count,
            flip_cell_count: flip_count,
            fault,
            first_fault_packet: first,
        };
        Ok(&self.receipt)
    }
}

pub fn scalar_cell_addresses<const D: usize>(
    topology: &CompiledTopology<D>,
) -> Result<Vec<ScalarCellAddress>, ValidationError> {
    let mut result = vec![ScalarCellAddress::default(); topology.graph.cells.len()];
    for brick in &topology.bricks {
        let r = brick.seed.resolution as u32;
        if r == 0 {
            return Err(ValidationError("zero scalar brick resolution".into()));
        }
        for dense in brick.cell_range.clone() {
            let stable = topology.graph.cells[dense as usize]
                .stable_id
                .unwrap_or(dense);
            let local = stable - brick.seed.key * if D == 3 { 512 } else { 64 };
            let x = local % r;
            let y = (local / r) % r;
            let z = if D == 3 { (local / (r * r)) % r } else { 0 };
            let packet_axis = ((r + 3) / 4).max(1);
            result[dense as usize] = ScalarCellAddress {
                packet: 64 * brick.seed.id
                    + (x >> 2)
                    + packet_axis * (y >> 2)
                    + packet_axis * packet_axis * (z >> 2),
                lane: ((x & 3) + 4 * (y & 3) + 16 * (z & 3)) as u8,
            };
        }
    }
    Ok(result)
}

pub fn publish_scalar_interface_state_from_geometric_density<const D: usize>(
    authority: &mut ScalarAuthority,
    topology: &CompiledTopology<D>,
    fields: &mut Fields,
    source_density: &[f32],
    source_gamma: &[f32],
    generation: u32,
    topology_slot: u8,
    has_rigid_bodies: bool,
) -> Result<ScalarPublicationReceipt, ValidationError> {
    publish_scalar_state(authority, topology, fields, source_density, source_gamma,
        generation, topology_slot, has_rigid_bodies, true)
}

/// Publish scalar authority without replacing a separately transported surface.
pub fn publish_scalar_state_preserving_interface<const D: usize>(
    authority: &mut ScalarAuthority,
    topology: &CompiledTopology<D>,
    fields: &mut Fields,
    source_density: &[f32],
    source_gamma: &[f32],
    generation: u32,
    topology_slot: u8,
    has_rigid_bodies: bool,
) -> Result<ScalarPublicationReceipt, ValidationError> {
    publish_scalar_state(authority, topology, fields, source_density, source_gamma,
        generation, topology_slot, has_rigid_bodies, false)
}

fn publish_scalar_state<const D: usize>(
    authority: &mut ScalarAuthority,
    topology: &CompiledTopology<D>,
    fields: &mut Fields,
    source_density: &[f32],
    source_gamma: &[f32],
    generation: u32,
    topology_slot: u8,
    has_rigid_bodies: bool,
    reconstruct_volume_interface: bool,
) -> Result<ScalarPublicationReceipt, ValidationError> {
    let addresses = scalar_cell_addresses(topology)?;
    let receipt = authority
        .publish(
            &topology.graph,
            &addresses,
            fields,
            source_density,
            source_gamma,
            generation,
            topology.graph.topology_generation,
            topology_slot,
            has_rigid_bodies,
        )?
        .clone();
    if receipt.fault != 0 {
        fields.fault = Some(NumericalFault {
            stage: "scalar-publication".into(),
            index: receipt.first_fault_packet,
            observed: receipt.fault as f32,
            expected: 0.0,
        });
    }
    if reconstruct_volume_interface {
        reconstruct_interfaces(&topology.graph, fields)?;
    }
    Ok(receipt)
}

fn pressure_density(graph: &Graph, fields: &Fields, cell: usize) -> f32 {
    let mut density = div(fields.density[cell], fields.capacity[cell].max(1e-8));
    let dt = fields.frame_dt;
    let area = graph.cells[cell].measure;
    let final_capacity = mul(fields.capacity[cell], area);
    let before = mul(
        Fields::optional_cell(&fields.capacity_before, cell, fields.capacity[cell]),
        area,
    );
    let source = Fields::optional_cell(&fields.source_rate, cell, 0.0);
    if fields.solid_motion_active && final_capacity < before && fields.density[cell] > 0.0 {
        let rate = if dt > 0.0 {
            div(final_capacity - before, dt)
        } else {
            0.0
        };
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(-rate, dt), final_capacity.max(1e-8)).min(0.5),
        ))
    }
    if source > 0.0 {
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(source, dt), final_capacity.max(1e-8)).min(0.5),
        ))
    }
    density
}
fn predicted_fill(graph: &Graph, fields: &Fields, cell: usize) -> bool {
    if !fields.solid_motion_active {
        return false;
    }
    let mut equation = 0.0;
    let mut correction = 0.0;
    for &rid in &graph.incidences[cell] {
        let row = &graph.rows[rid as usize];
        let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == cell) else {
            continue;
        };
        let geometry = !graph.solid_voxel_fraction.is_empty() || fields.solid_motion_active;
        let velocity = if geometry {
            fields.face_velocity[row.id as usize] - mul(1.0 - row.open_fraction, row.solid_velocity)
        } else {
            fields.face_velocity[row.id as usize]
        };
        let weight = if geometry {
            row.static_dual_weight.unwrap_or(row.dual_weight)
        } else {
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
        };
        let value = mul(own.coefficient, mul(weight, velocity));
        if value > 0.0 {
            let supported = row
                .terms
                .iter()
                .filter(|t| t.cell_id as usize != cell && t.coefficient * own.coefficient < 0.0)
                .any(|t| {
                    let j = t.cell_id as usize;
                    let cap = mul(fields.capacity[j], graph.cells[j].measure);
                    mul(fields.density[j], graph.cells[j].measure) > mul(VOLUME_ROUNDOFF_RATIO, cap)
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
    let capacity = mul(fields.capacity[cell], graph.cells[cell].measure);
    add(
        mul(fields.density[cell], graph.cells[cell].measure),
        mul(fields.frame_dt, equation),
    ) >= capacity - mul(VOLUME_ROUNDOFF_RATIO, capacity)
}
fn pressure_membership_predicate(graph: &Graph, fields: &Fields, cell: usize) -> bool {
    let mut submerged = fields.pressure_member[cell] != 0;
    let mut neighbors = 0;
    if submerged {
        'rows: for &rid in &graph.incidences[cell] {
            let row = &graph.rows[rid as usize];
            if row.terms.len() < 2 {
                submerged = false;
                break;
            }
            for t in &row.terms {
                if t.cell_id as usize != cell {
                    neighbors += 1;
                    if fields.pressure_member[t.cell_id as usize] == 0 {
                        submerged = false;
                        break 'rows;
                    }
                }
            }
        }
        submerged &= neighbors > 0
    }
    (pressure_density(graph, fields, cell) >= LIQUID_ISOVALUE
        || submerged
        || predicted_fill(graph, fields, cell)
        || Fields::optional_cell(&fields.source_rate, cell, 0.0) > 0.0)
        && fields.capacity[cell] as f64 * graph.cells[cell].measure as f64 > 1e-8_f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn address_is_packet_major_and_lane_xy() {
        let a = ScalarCellAddress {
            packet: 65,
            lane: 15,
        };
        assert_eq!(a.packet, 65);
        assert_eq!(1u32 << (a.lane & 31), 0x8000)
    }
}
