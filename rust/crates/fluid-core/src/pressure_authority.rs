//! CPU pressure membership/cache/execution authority used by the symmetry embedding.
//!
//! The GPU implementation carries PCM1/PCF1/PEI1 headers as well.  Rust only
//! needs their observable state: stable membership bits, 64-row dirty tiles,
//! cached field/diagonal words, and the canonical dense execution stream.

use crate::types::{Fields, Graph};
use serde::Serialize;

pub const INVALID_AUTHORITY_ID: u32 = u32::MAX;
const MEMBERSHIP_LEAF_BITS: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureAuthorityReceipt {
    pub topology_generation: u32,
    pub cell_generation: u32,
    pub row_generation: u32,
    pub coefficient_generation: u32,
    pub execution_generation: u32,
    pub dirty_cell_leaves: Vec<u32>,
    pub dirty_row_tiles: Vec<u32>,
    pub changed_diagonal_count: u32,
    pub pressure_cell_count: u32,
    pub pressure_row_count: u32,
    pub fault: u32,
    pub first_fault_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PressureAuthority {
    pub cell_capacity: usize,
    pub row_capacity: usize,
    pub accepted_cell_bits: Vec<u32>,
    pub accepted_row_bits: Vec<u32>,
    pub density_bits: Vec<u32>,
    pub capacity_bits: Vec<u32>,
    pub normal_x_bits: Vec<u32>,
    pub normal_y_bits: Vec<u32>,
    pub diagonal_bits: Vec<u32>,
    pub row_theta: Vec<f32>,
    /// Dense cell ids in the topology's canonical (stable-leaf bit) order.
    pub execution_order: Vec<u32>,
    pub receipt: PressureAuthorityReceipt,
}

impl PressureAuthority {
    pub fn new(graph: &Graph, cell_capacity: Option<usize>, row_capacity: Option<usize>) -> Self {
        let high = graph
            .cells
            .iter()
            .map(|c| c.stable_id.unwrap_or(c.id) as usize + 1)
            .max()
            .unwrap_or(1);
        let cells = cell_capacity.unwrap_or(high).max(1);
        let rows = row_capacity.unwrap_or(graph.rows.len()).max(1);
        Self {
            cell_capacity: cells,
            row_capacity: rows,
            accepted_cell_bits: vec![0; cells.div_ceil(32)],
            accepted_row_bits: vec![0; rows.div_ceil(32)],
            density_bits: vec![0; cells],
            capacity_bits: vec![0; cells],
            normal_x_bits: vec![0; cells],
            normal_y_bits: vec![0; cells],
            diagonal_bits: vec![0; cells],
            row_theta: vec![0.0; rows],
            execution_order: Vec::new(),
            receipt: PressureAuthorityReceipt {
                first_fault_id: INVALID_AUTHORITY_ID,
                ..Default::default()
            },
        }
    }

    /// Grow stable-cell/row storage without discarding the accepted cache or
    /// its generation counters. Shrinking requests are deliberately ignored.
    pub fn resize_preserving_cache(
        &mut self,
        graph: &Graph,
        cell_capacity: Option<usize>,
        row_capacity: Option<usize>,
    ) {
        let stable_high = graph
            .cells
            .iter()
            .map(|c| c.stable_id.unwrap_or(c.id) as usize + 1)
            .max()
            .unwrap_or(1);
        let cells = cell_capacity
            .unwrap_or(stable_high)
            .max(stable_high)
            .max(self.cell_capacity);
        let rows = row_capacity
            .unwrap_or(graph.rows.len())
            .max(graph.rows.len())
            .max(self.row_capacity);
        if cells == self.cell_capacity && rows == self.row_capacity {
            return;
        }
        self.cell_capacity = cells;
        self.row_capacity = rows;
        self.accepted_cell_bits.resize(cells.div_ceil(32), 0);
        self.accepted_row_bits.resize(rows.div_ceil(32), 0);
        self.density_bits.resize(cells, 0);
        self.capacity_bits.resize(cells, 0);
        self.normal_x_bits.resize(cells, 0);
        self.normal_y_bits.resize(cells, 0);
        self.diagonal_bits.resize(cells, 0);
        self.row_theta.resize(rows, 0.0);
    }

    pub fn publish(
        &mut self,
        graph: &Graph,
        fields: &Fields,
        active_rows: &[u8],
        theta: &[f32],
        topology_generation: u32,
        global_row_invalidation: bool,
    ) {
        let mut cells = vec![0u32; self.accepted_cell_bits.len()];
        let mut rows = vec![0u32; self.accepted_row_bits.len()];
        let mut density = self.density_bits.clone();
        let mut capacity = self.capacity_bits.clone();
        let mut normal_x = self.normal_x_bits.clone();
        let mut normal_y = self.normal_y_bits.clone();
        let mut diagonal = self.diagonal_bits.clone();
        let mut row_theta = self.row_theta.clone();
        let mut stable_order = Vec::new();
        let mut fault = 0u32;
        let mut first = INVALID_AUTHORITY_ID;
        for cell in &graph.cells {
            let dense = cell.id as usize;
            let stable = cell.stable_id.unwrap_or(cell.id) as usize;
            if stable >= self.cell_capacity {
                fault = 3;
                first = stable as u32;
                break;
            }
            density[stable] = fields.density[dense].to_bits();
            capacity[stable] = fields.capacity[dense].to_bits();
            normal_x[stable] = fields.interface_normal[2 * dense].to_bits();
            normal_y[stable] = fields.interface_normal[2 * dense + 1].to_bits();
            diagonal[stable] = fields.pressure_diagonal[dense].to_bits();
            if fields.pressure_member[dense] != 0 {
                cells[stable >> 5] |= 1u32 << (stable & 31);
                stable_order.push((stable, cell.id));
            }
        }
        for row in &graph.rows {
            let id = row.id as usize;
            if id >= self.row_capacity {
                if fault == 0 {
                    fault = 3;
                    first = row.id;
                }
                break;
            }
            if active_rows[id] != 0 {
                rows[id >> 5] |= 1u32 << (id & 31);
            }
            row_theta[id] = theta[id];
        }
        if fault != 0 {
            self.receipt.topology_generation = topology_generation;
            self.receipt.fault = fault;
            self.receipt.first_fault_id = first;
            return;
        }
        let cell_generation = self.receipt.cell_generation + 1;
        let row_generation = self.receipt.row_generation + 1;
        if cell_generation >= 0x7fff_ffff || row_generation >= 0x7fff_ffff {
            self.receipt.topology_generation = topology_generation;
            self.receipt.fault = 2;
            self.receipt.first_fault_id = INVALID_AUTHORITY_ID;
            return;
        }
        let dirty_cell_leaves = (0..cells.len().div_ceil(MEMBERSHIP_LEAF_BITS / 32))
            .filter(|leaf| {
                let begin = leaf * (MEMBERSHIP_LEAF_BITS / 32);
                let end = (begin + MEMBERSHIP_LEAF_BITS / 32).min(cells.len());
                cells[begin..end] != self.accepted_cell_bits[begin..end]
            })
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        let mut changed_stable = vec![false; self.cell_capacity];
        for cell in &graph.cells {
            let dense = cell.id as usize;
            let stable = cell.stable_id.unwrap_or(cell.id) as usize;
            let mask = 1u32 << (stable & 31);
            changed_stable[stable] = density[stable] != self.density_bits[stable]
                || capacity[stable] != self.capacity_bits[stable]
                || normal_x[stable] != self.normal_x_bits[stable]
                || normal_y[stable] != self.normal_y_bits[stable]
                || ((cells[stable >> 5] ^ self.accepted_cell_bits[stable >> 5]) & mask) != 0;
            let _ = dense;
        }
        let topology_changed = self.receipt.topology_generation != topology_generation;
        let mut dirty_row_tiles = Vec::new();
        for tile in 0..rows.len().div_ceil(2) {
            let mut changed = topology_changed
                || global_row_invalidation
                || self.accepted_row_bits.get(2 * tile) != rows.get(2 * tile)
                || self.accepted_row_bits.get(2 * tile + 1) != rows.get(2 * tile + 1);
            if !changed {
                for local in 0..64 {
                    if let Some(row) = graph.rows.get(64 * tile + local) {
                        if row.terms.iter().any(|t| {
                            changed_stable[graph.cells[t.cell_id as usize]
                                .stable_id
                                .unwrap_or(t.cell_id)
                                as usize]
                        }) {
                            changed = true;
                            break;
                        }
                    }
                }
            }
            if changed {
                dirty_row_tiles.push(tile as u32);
            }
        }
        let changed_diagonal_count = diagonal
            .iter()
            .zip(&self.diagonal_bits)
            .filter(|(a, b)| a != b)
            .count() as u32;
        let pressure_row_count = rows.iter().map(|w| w.count_ones()).sum();
        self.accepted_cell_bits = cells;
        self.accepted_row_bits = rows;
        self.density_bits = density;
        self.capacity_bits = capacity;
        self.normal_x_bits = normal_x;
        self.normal_y_bits = normal_y;
        self.diagonal_bits = diagonal;
        self.row_theta = row_theta;
        stable_order.sort_unstable_by_key(|&(stable, _)| stable);
        self.execution_order = stable_order.into_iter().map(|(_, dense)| dense).collect();
        self.receipt = PressureAuthorityReceipt {
            topology_generation,
            cell_generation,
            row_generation,
            coefficient_generation: self.receipt.coefficient_generation + 1,
            execution_generation: self.receipt.execution_generation + 1,
            dirty_cell_leaves,
            dirty_row_tiles,
            changed_diagonal_count,
            pressure_cell_count: self.execution_order.len() as u32,
            pressure_row_count,
            fault: 0,
            first_fault_id: INVALID_AUTHORITY_ID,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cell, Row, RowKind, RowTerm};

    fn cell(id: u32, stable: u32) -> Cell {
        Cell {
            id,
            stable_id: Some(stable),
            minimum: [id as f32, 0.0, 0.0],
            maximum: [id as f32 + 1.0, 1.0, 1.0],
            center: [id as f32 + 0.5, 0.5, 0.5],
            widths: [1.0; 3],
            measure: 1.0,
            brick_key: None,
            refinement_region_scale: None,
        }
    }
    fn row(id: u32, cell_id: u32) -> Row {
        Row {
            id,
            kind: RowKind::IntraBrick,
            axis: 0,
            center: [0.0; 3],
            measure: 1.0,
            static_measure: None,
            distance: 1.0,
            dual_weight: 1.0,
            static_dual_weight: None,
            static_open_fraction: None,
            terms: vec![RowTerm {
                cell_id,
                coefficient: 1.0,
            }],
            open_fraction: 1.0,
            open_fraction_before: None,
            open_fraction_after: None,
            solid_velocity: 0.0,
            separating: false,
        }
    }
    #[test]
    fn publishes_stable_bit_order_and_sixty_four_row_tiles() {
        let graph = Graph {
            dimension: 3,
            cells: vec![cell(0, 65), cell(1, 2)],
            rows: (0..65).map(|id| row(id, (id == 64) as u32)).collect(),
            incidences: vec![vec![], vec![]],
            ..Default::default()
        };
        let mut fields = Fields {
            density: vec![0.25, 0.75],
            capacity: vec![1.0; 2],
            pressure_diagonal: vec![1.0, 2.0],
            pressure_member: vec![1, 1],
            interface_normal: vec![0.0; 4],
            ..Default::default()
        };
        let mut authority = PressureAuthority::new(&graph, None, None);
        authority.publish(&graph, &fields, &vec![1; 65], &vec![1.0; 65], 7, false);
        assert_eq!(authority.execution_order, vec![1, 0]);
        assert_eq!(authority.receipt.dirty_row_tiles, vec![0, 1]);
        fields.density[1] = 0.5;
        authority.publish(&graph, &fields, &vec![1; 65], &vec![1.0; 65], 7, false);
        assert_eq!(authority.receipt.dirty_row_tiles, vec![1]);
        let receipt = authority.receipt.clone();
        let order = authority.execution_order.clone();
        let member = authority.accepted_cell_bits.clone();
        authority.resize_preserving_cache(&graph, Some(130), Some(192));
        assert_eq!(authority.receipt, receipt);
        assert_eq!(authority.execution_order, order);
        assert_eq!(&authority.accepted_cell_bits[..member.len()], member);
        assert_eq!(
            (authority.cell_capacity, authority.row_capacity),
            (130, 192)
        );
    }
}
