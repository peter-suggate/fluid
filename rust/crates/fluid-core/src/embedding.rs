//! Exact retained-3D pressure embedding for the two-dimensional symmetry world.

use std::collections::HashMap;

use crate::kernels::{add, div, mul};
use crate::numerics::{GHOST_FLUID_THETA_MIN, LIQUID_ISOVALUE};
use crate::pressure::{solve_pressure_pcg, PressureError, PressurePcgReceipt};
use crate::pressure_authority::PressureAuthority;
use crate::types::{Fields, Graph, NumericalFault, Row, RowKind};

const VIRTUAL_ROUNDOFF_RATIO: f32 = 9.536_743_164_062_5e-7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingFaultKind {
    Cell,
    Row,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappingFault {
    pub kind: MappingFaultKind,
    pub id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingOptions {
    pub center_cell_z: i32,
    pub z_boundary_omitted: bool,
    pub sparse_air_phi: f32,
    pub source_generation: u32,
    pub solid_world: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            center_cell_z: 0,
            z_boundary_omitted: false,
            sparse_air_phi: 0.5,
            source_generation: 1,
            solid_world: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PressureEmbedding {
    pub source: Graph,
    pub reduced_cell: Vec<i32>,
    pub reduced_owner: Vec<i32>,
    pub centre_cell: Vec<i32>,
    pub projected_row: Vec<i32>,
    pub centre_row: Vec<i32>,
    pub mapping_fault: Option<MappingFault>,
    pub pressure_authority: PressureAuthority,
    pub pressure: Vec<f32>,
    pub pressure_member: Vec<u8>,
    pub row_active: Vec<u8>,
    pub row_theta: Vec<f32>,
    density_bits: Vec<u32>,
    capacity_bits: Vec<u32>,
    normal_x_bits: Vec<u32>,
    normal_y_bits: Vec<u32>,
    row_open_bits: Vec<u32>,
    cache_initialized: bool,
    options: EmbeddingOptions,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEmbedding {
    pub diagonal: Vec<f32>,
    pub rhs: Vec<f32>,
    pub active_rows: Vec<u8>,
    pub theta: Vec<f32>,
    pub pressure_weight: Vec<f32>,
    pub virtual_diagonal: Vec<f32>,
    pub virtual_rhs: Vec<f32>,
    pub virtual_member: Vec<u8>,
    pub dirty_row_count: u32,
    pub mapping_fault: Option<MappingFault>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingSolveReceipt {
    pub prepared: PreparedEmbedding,
    pub solve: PressurePcgReceipt,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RowKey {
    axis: u8,
    x: u32,
    y: u32,
}
fn row_key(row: &Row) -> RowKey {
    RowKey {
        axis: row.axis,
        x: row.center[0].to_bits(),
        y: row.center[1].to_bits(),
    }
}

impl PressureEmbedding {
    pub fn new(
        source: Graph,
        reduced: &Graph,
        options: EmbeddingOptions,
        initial_fields: Option<&Fields>,
        previous: Option<&PressureEmbedding>,
    ) -> Self {
        assert_eq!(
            source.dimension, 3,
            "pressure embedding source graph must be 3-D"
        );
        assert_eq!(
            reduced.dimension, 2,
            "pressure embedding reduced graph must be 2-D"
        );
        let nx = reduced.dimensions[0] as usize;
        let ny = reduced.dimensions[1] as usize;
        let mut owners = vec![-1; nx * ny];
        for cell in &reduced.cells {
            for y in cell.minimum[1].floor() as i32..cell.maximum[1].floor() as i32 {
                for x in cell.minimum[0].floor() as i32..cell.maximum[0].floor() as i32 {
                    if x >= 0 && y >= 0 && (x as usize) < nx && (y as usize) < ny {
                        owners[x as usize + nx * y as usize] = cell.id as i32;
                    }
                }
            }
        }
        let reduced_cell = source
            .cells
            .iter()
            .map(|cell| {
                let x = (cell.center[0].floor() as i32).clamp(0, nx as i32 - 1) as usize;
                let y = (cell.center[1].floor() as i32).clamp(0, ny as i32 - 1) as usize;
                owners[x + nx * y]
            })
            .collect::<Vec<_>>();
        let mut centre_cell = vec![-1; reduced.cells.len()];
        let z = options.center_cell_z as f32 + 0.5;
        for cell in &source.cells {
            let rid = reduced_cell[cell.id as usize];
            if rid < 0 {
                continue;
            }
            let target = &reduced.cells[rid as usize];
            if cell.minimum[0] == target.minimum[0]
                && cell.maximum[0] == target.maximum[0]
                && cell.minimum[1] == target.minimum[1]
                && cell.maximum[1] == target.maximum[1]
                && z >= cell.minimum[2]
                && z < cell.maximum[2]
            {
                centre_cell[rid as usize] = cell.id as i32;
            }
        }
        let mut candidates: HashMap<RowKey, Vec<u32>> = HashMap::new();
        for row in &reduced.rows {
            candidates.entry(row_key(row)).or_default().push(row.id);
        }
        let mut projected_row = vec![-1; source.rows.len()];
        for row in &source.rows {
            if row.axis == 2 {
                continue;
            }
            if let Some(ids) = candidates.get(&row_key(row)) {
                if ids.len() == 1 {
                    projected_row[row.id as usize] = ids[0] as i32;
                }
            }
        }
        let mut centre_row = vec![-1; reduced.rows.len()];
        for row in &source.rows {
            let projected = projected_row[row.id as usize];
            if projected < 0 {
                continue;
            }
            let touches = row.terms.iter().any(|term| {
                let rid = reduced_cell[term.cell_id as usize];
                rid >= 0 && centre_cell[rid as usize] == term.cell_id as i32
            });
            if touches {
                let slot = &mut centre_row[projected as usize];
                *slot = if *slot == -1 { row.id as i32 } else { -2 };
            }
        }
        let mapping_fault = centre_cell
            .iter()
            .position(|&x| x < 0)
            .map(|id| MappingFault {
                kind: MappingFaultKind::Cell,
                id: id as u32,
            })
            .or_else(|| {
                centre_row
                    .iter()
                    .position(|&x| x < 0)
                    .map(|id| MappingFault {
                        kind: MappingFaultKind::Row,
                        id: id as u32,
                    })
            });
        let pressure = source
            .cells
            .iter()
            .map(|c| {
                let r = reduced_cell[c.id as usize];
                initial_fields
                    .filter(|_| r >= 0)
                    .map_or(0.0, |f| f.pressure[r as usize])
            })
            .collect();
        let pressure_member = source
            .cells
            .iter()
            .map(|c| {
                let r = reduced_cell[c.id as usize];
                initial_fields
                    .filter(|_| r >= 0)
                    .map_or(0, |f| f.pressure_member[r as usize])
            })
            .collect::<Vec<_>>();
        let high = source
            .cells
            .iter()
            .map(|c| c.stable_id.unwrap_or(c.id) as usize + 1)
            .max()
            .unwrap_or(1);
        let pressure_authority = previous
            .filter(|p| {
                p.pressure_authority.cell_capacity >= high
                    && p.pressure_authority.row_capacity >= source.rows.len()
            })
            .map(|p| p.pressure_authority.clone())
            .unwrap_or_else(|| {
                let mut authority =
                    PressureAuthority::new(&source, Some(high), Some(source.rows.len()));
                if let Some(previous) = previous {
                    // The new banks are populated on the next pressure prepare,
                    // but publication continues to expose the last sealed
                    // receipt across the topology swap.
                    authority.receipt = previous.pressure_authority.receipt.clone();
                }
                authority
            });
        let nc = source.cells.len();
        let nr = source.rows.len();
        Self {
            source,
            reduced_cell,
            reduced_owner: owners,
            centre_cell,
            projected_row,
            centre_row,
            mapping_fault,
            pressure_authority,
            pressure,
            pressure_member,
            row_active: vec![0; nr],
            row_theta: vec![0.0; nr],
            density_bits: vec![0; nc],
            capacity_bits: vec![0; nc],
            normal_x_bits: vec![0; nc],
            normal_y_bits: vec![0; nc],
            row_open_bits: vec![0; nr],
            cache_initialized: false,
            options,
        }
    }

    fn row2<'a>(&self, reduced: &'a Graph, row: &Row) -> Option<&'a Row> {
        let id = self.projected_row[row.id as usize];
        (id >= 0).then(|| &reduced.rows[id as usize])
    }
    fn source_rate(&self, reduced: &Graph, fields: &Fields, source: usize) -> f32 {
        let r = self.reduced_cell[source];
        if r < 0 {
            return 0.0;
        }
        mul(
            Fields::optional_cell(&fields.source_rate, r as usize, 0.0),
            div(
                self.source.cells[source].measure,
                reduced.cells[r as usize].measure,
            ),
        )
    }
    fn capacity(&self, fields: &Fields, source: usize, before: bool) -> f32 {
        let r = self.reduced_cell[source];
        if r < 0 {
            return 0.0;
        }
        let v = if before {
            Fields::optional_cell(
                &fields.capacity_before,
                r as usize,
                fields.capacity[r as usize],
            )
        } else {
            fields.capacity[r as usize]
        };
        mul(v, self.source.cells[source].measure)
    }
    fn pressure_density(&self, reduced: &Graph, fields: &Fields, source: usize) -> f32 {
        let r = self.reduced_cell[source];
        if r < 0 {
            return 0.0;
        }
        let r = r as usize;
        let mut density = div(fields.density[r], fields.capacity[r].max(1e-6));
        let final_capacity = self.capacity(fields, source, false);
        let before_capacity = self.capacity(fields, source, true);
        let source_rate = self.source_rate(reduced, fields, source);
        if fields.solid_motion_active && final_capacity < before_capacity && fields.density[r] > 0.0
        {
            let rate = if fields.frame_dt > 0.0 {
                div(final_capacity - before_capacity, fields.frame_dt)
            } else {
                0.0
            };
            density = density.max(add(
                LIQUID_ISOVALUE,
                div(mul(-rate, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
            ));
        }
        if source_rate > 0.0 {
            density = density.max(add(
                LIQUID_ISOVALUE,
                div(mul(source_rate, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
            ));
        }
        density
    }
    fn open_fraction(&self, reduced: &Graph, fields: &Fields, row: &Row) -> f32 {
        if row.axis == 2 {
            let mut open = 1.0f32;
            let mut seen = false;
            for term in &row.terms {
                let r = self.reduced_cell[term.cell_id as usize];
                if r < 0 {
                    continue;
                }
                let r = r as usize;
                let before = Fields::optional_cell(&fields.capacity_before, r, fields.capacity[r]);
                let after = Fields::optional_cell(&fields.capacity_after, r, fields.capacity[r]);
                open = open.min(mul(0.5, add(before, after)));
                seen = true;
            }
            return if seen { open } else { 1.0 };
        }
        match self.row2(reduced, row) {
            None => 1.0,
            Some(r) if r.kind == RowKind::ClosedWorld => {
                if r.separating {
                    1.0
                } else {
                    0.0
                }
            }
            Some(r) => r.open_fraction,
        }
    }
    fn fluid_velocity(&self, reduced: &Graph, fields: &Fields, row: &Row) -> f32 {
        if row.axis == 2 {
            return 0.0;
        }
        self.row2(reduced, row).map_or(0.0, |r| {
            fields.face_velocity[r.id as usize] - mul(1.0 - r.open_fraction, r.solid_velocity)
        })
    }
    fn skip_z_air(&self, row: &Row) -> bool {
        row.axis == 2 && row.kind == RowKind::SparseAir && !self.options.z_boundary_omitted
    }
    fn moving_predicted_fill(&self, reduced: &Graph, fields: &Fields, source: usize) -> bool {
        if !fields.solid_motion_active {
            return false;
        }
        let mut equation = 0.0;
        let mut correction = 0.0;
        for &row_id in &self.source.incidences[source] {
            let row = &self.source.rows[row_id as usize];
            if self.skip_z_air(row) {
                continue;
            }
            let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == source) else {
                continue;
            };
            let value = mul(
                own.coefficient,
                mul(row.dual_weight, self.fluid_velocity(reduced, fields, row)),
            );
            if value > 0.0 {
                let supported = row
                    .terms
                    .iter()
                    .filter(|t| {
                        t.cell_id as usize != source && t.coefficient * own.coefficient < 0.0
                    })
                    .any(|t| {
                        let s = t.cell_id as usize;
                        let r = self.reduced_cell[s];
                        r >= 0
                            && (mul(fields.density[r as usize], self.source.cells[s].measure)
                                > mul(VIRTUAL_ROUNDOFF_RATIO, self.capacity(fields, s, false))
                                || self.source_rate(reduced, fields, s) > 0.0)
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
        let r = self.reduced_cell[source];
        if r < 0 {
            return false;
        }
        let cap = self.capacity(fields, source, false);
        add(
            mul(
                fields.density[r as usize],
                self.source.cells[source].measure,
            ),
            mul(fields.frame_dt, equation),
        ) >= cap - mul(VIRTUAL_ROUNDOFF_RATIO, cap)
    }
    fn integrated_height(&self, reduced: &Graph, fields: &Fields, x: i32) -> (f32, bool) {
        let nx = reduced.dimensions[0] as i32;
        let ny = reduced.dimensions[1] as i32;
        let sx = x.clamp(0, nx - 1);
        let (mut y, mut mass, mut previous, mut column_open) = (0, 0.0, 1.0, -1.0);
        let (mut saw_open, mut saw_liquid, mut saw_air) = (false, false, false);
        while y < ny {
            if reduced
                .solid_voxel_fraction
                .get((sx + nx * y) as usize)
                .copied()
                .unwrap_or(0.0)
                >= 1.0
            {
                return (0.0, false);
            }
            let owner = self.reduced_owner[(sx + nx * y) as usize];
            let (fill, width) = if owner < 0 {
                (0.0, (8 - y % 8).max(1).min(ny - y))
            } else {
                let r = owner as usize;
                let open = fields.capacity[r];
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
                let source = self.centre_cell[r];
                let fill = if source < 0 {
                    0.0
                } else {
                    self.pressure_density(reduced, fields, source as usize)
                        .clamp(0.0, 1.0)
                };
                let w = (reduced.cells[r].widths[1] as i32
                    - y % (reduced.cells[r].widths[1] as i32))
                    .max(1)
                    .min(ny - y);
                (fill, w)
            };
            if fill > previous + 0.01 {
                return (0.0, false);
            }
            previous = fill;
            saw_liquid |= fill > 1e-3;
            saw_air |= fill < 1.0 - 1e-3;
            mass = add(mass, mul(fill, width as f32));
            y += width;
        }
        (mass, saw_open && saw_liquid && saw_air)
    }
    fn planar_height(&self, reduced: &Graph, fields: &Fields, row: &Row) -> (f32, bool) {
        let nx = reduced.dimensions[0] as i32;
        let centre = (row.center[0].floor() as i32).clamp(0, nx - 1);
        let (mut height, mut minimum, mut maximum, mut valid) = (0.0, f32::MAX, -f32::MAX, true);
        for offset in [0, -1, 1] {
            let (v, ok) =
                self.integrated_height(reduced, fields, (centre + offset).clamp(0, nx - 1));
            if offset == 0 {
                height = v
            }
            valid &= ok;
            minimum = minimum.min(v);
            maximum = maximum.max(v)
        }
        (height, valid && maximum - minimum <= 0.01)
    }

    pub fn prepare(&mut self, reduced: &Graph, fields: &mut Fields) -> PreparedEmbedding {
        let nc = self.source.cells.len();
        let nr = self.source.rows.len();
        let mut result = PreparedEmbedding {
            diagonal: vec![0.0; reduced.cells.len()],
            rhs: vec![0.0; reduced.cells.len()],
            active_rows: vec![0; nr],
            theta: vec![0.0; nr],
            pressure_weight: vec![0.0; nr],
            virtual_diagonal: vec![0.0; nc],
            virtual_rhs: vec![0.0; nc],
            virtual_member: vec![0; nc],
            dirty_row_count: 0,
            mapping_fault: self.mapping_fault,
        };
        if let Some(fault) = self.mapping_fault {
            fields.fault = Some(NumericalFault {
                stage: "pressure-embedding-map".into(),
                index: fault.id,
                observed: if fault.kind == MappingFaultKind::Cell {
                    1.0
                } else {
                    2.0
                },
                expected: 0.0,
            });
            return result;
        }
        let prior = self.pressure_member.clone();
        let mut direct_dirty = vec![0u8; nc];
        let mut dirty_cell = vec![0u8; nc];
        for cell in &self.source.cells {
            let s = cell.id as usize;
            let r = self.reduced_cell[s];
            if r < 0 {
                continue;
            }
            let r = r as usize;
            let bits = [
                fields.density[r].to_bits(),
                fields.capacity[r].to_bits(),
                fields.interface_normal[2 * r].to_bits(),
                fields.interface_normal[2 * r + 1].to_bits(),
            ];
            direct_dirty[s] = u8::from(
                !self.cache_initialized
                    || bits[0] != self.density_bits[s]
                    || bits[1] != self.capacity_bits[s]
                    || bits[2] != self.normal_x_bits[s]
                    || bits[3] != self.normal_y_bits[s]
                    || Fields::optional_cell(&fields.source_rate, r, 0.0) != 0.0,
            );
            self.density_bits[s] = bits[0];
            self.capacity_bits[s] = bits[1];
            self.normal_x_bits[s] = bits[2];
            self.normal_y_bits[s] = bits[3];
        }
        for cell in &self.source.cells {
            let s = cell.id as usize;
            let r = self.reduced_cell[s];
            if r < 0 {
                continue;
            }
            let mut submerged = prior[s] != 0;
            let mut neighbors = 0;
            if submerged {
                'outer: for &rid in &self.source.incidences[s] {
                    let row = &self.source.rows[rid as usize];
                    if self.skip_z_air(row) {
                        continue;
                    }
                    if row.terms.len() < 2 {
                        submerged = false;
                        break;
                    }
                    for t in &row.terms {
                        if t.cell_id as usize != s {
                            neighbors += 1;
                            if prior[t.cell_id as usize] == 0 {
                                submerged = false;
                                break 'outer;
                            }
                        }
                    }
                }
                submerged &= neighbors > 0;
            }
            result.virtual_member[s] = u8::from(
                (self.pressure_density(reduced, fields, s) >= LIQUID_ISOVALUE
                    || submerged
                    || self.moving_predicted_fill(reduced, fields, s)
                    || self.source_rate(reduced, fields, s) > 0.0)
                    && self.capacity(fields, s, false) > 1e-8,
            );
            if result.virtual_member[s] != prior[s] {
                direct_dirty[s] = 1
            }
        }
        dirty_cell.copy_from_slice(&direct_dirty);
        for cell in &self.source.cells {
            let s = cell.id as usize;
            let r = self.reduced_cell[s];
            if r < 0 || fields.capacity[r as usize] < 0.999999 {
                continue;
            }
            let fill = div(
                fields.density[r as usize],
                fields.capacity[r as usize].max(1e-8),
            );
            if fill <= 0.0 || fill >= 1.0 {
                continue;
            }
            for &rid in &self.source.incidences[s] {
                let row = &self.source.rows[rid as usize];
                let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == s) else {
                    continue;
                };
                if row.terms.iter().any(|t| {
                    t.cell_id as usize != s
                        && own.coefficient * t.coefficient < 0.0
                        && direct_dirty[t.cell_id as usize] != 0
                }) {
                    dirty_cell[s] = 1;
                    break;
                }
            }
        }
        let global =
            !self.cache_initialized || self.options.solid_world || fields.solid_motion_active;
        let mut dirty_tiles = vec![0u8; nr.div_ceil(64)];
        if global {
            dirty_tiles.fill(1)
        } else {
            for row in &self.source.rows {
                let open = self.open_fraction(reduced, fields, row);
                if open.to_bits() != self.row_open_bits[row.id as usize]
                    || row
                        .terms
                        .iter()
                        .any(|t| dirty_cell[t.cell_id as usize] != 0)
                {
                    dirty_tiles[row.id as usize >> 6] = 1
                }
            }
        }
        let partial = reduced
            .cells
            .iter()
            .any(|c| c.refinement_region_scale.unwrap_or(1.0) > 1.0)
            && reduced
                .cells
                .iter()
                .any(|c| c.refinement_region_scale.unwrap_or(1.0) == 1.0);
        for row in &self.source.rows {
            let ri = row.id as usize;
            let open = self.open_fraction(reduced, fields, row);
            self.row_open_bits[ri] = open.to_bits();
            if dirty_tiles[ri >> 6] == 0 {
                result.active_rows[ri] = self.row_active[ri];
                result.theta[ri] = self.row_theta[ri];
                result.pressure_weight[ri] = mul(row.dual_weight, open);
                continue;
            }
            result.dirty_row_count += 1;
            if self.skip_z_air(row) {
                continue;
            }
            let row2 = self.row2(reduced, row);
            let closed = row2.is_some_and(|r| r.kind == RowKind::ClosedWorld);
            let exterior = closed || row.kind == RowKind::SparseAir;
            let (mut gx, mut gy, mut go, mut gw) = (0.0, 0.0, 0.0, 0.0);
            if !closed {
                for term in &row.terms {
                    let r = self.reduced_cell[term.cell_id as usize];
                    if r < 0 {
                        continue;
                    }
                    let r = r as usize;
                    let cell = &reduced.cells[r];
                    let nx = fields.interface_normal[2 * r];
                    let ny = fields.interface_normal[2 * r + 1];
                    let w = term.coefficient.abs();
                    if add(mul(nx, nx), mul(ny, ny)) <= 0.5 {
                        continue;
                    }
                    gx = add(gx, mul(w, nx));
                    gy = add(gy, mul(w, ny));
                    go = add(
                        go,
                        mul(
                            w,
                            add(
                                fields.interface_offset[r],
                                add(
                                    mul(nx, cell.center[0] - row.center[0]),
                                    mul(ny, cell.center[1] - row.center[1]),
                                ),
                            ),
                        ),
                    );
                    gw = add(gw, w)
                }
            }
            let gl = add(mul(gx, gx), mul(gy, gy)).sqrt();
            let mut valid = gw > 1e-8 && gl > mul(1e-6, gw);
            if valid {
                for term in &row.terms {
                    let r = self.reduced_cell[term.cell_id as usize];
                    if r < 0 {
                        valid = false;
                        break;
                    }
                    let cell = &self.source.cells[term.cell_id as usize];
                    let phi = div(
                        add(
                            mul(gx, cell.center[0] - row.center[0]),
                            mul(gy, cell.center[1] - row.center[1]),
                        ) - go,
                        gl,
                    );
                    valid &= fields.capacity[r as usize] >= 0.999999
                        && if result.virtual_member[term.cell_id as usize] != 0 {
                            phi <= 0.0
                        } else {
                            phi >= 0.0
                        }
                }
            }
            let (mut liquid, mut air) = (0, 0);
            let (mut lp, mut lw, mut ap, mut aw, mut ly_sum, mut ay_sum, mut full, mut liquid_grad) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for term in &row.terms {
                let s = term.cell_id as usize;
                let r = self.reduced_cell[s];
                if r < 0 {
                    continue;
                }
                let cell = &self.source.cells[s];
                let old = mul(
                    LIQUID_ISOVALUE - self.pressure_density(reduced, fields, s),
                    if exterior {
                        1.0
                    } else {
                        cell.widths[row.axis as usize]
                    },
                );
                let phi = if valid {
                    div(
                        add(
                            mul(gx, cell.center[0] - row.center[0]),
                            mul(gy, cell.center[1] - row.center[1]),
                        ) - go,
                        gl,
                    )
                } else {
                    old
                };
                let w = term.coefficient.abs();
                let signed = mul(term.coefficient, phi);
                full = add(full, signed);
                if result.virtual_member[s] != 0 {
                    liquid += 1;
                    lp = add(lp, mul(w, phi));
                    lw = add(lw, w);
                    liquid_grad = add(liquid_grad, signed);
                    ly_sum = add(ly_sum, mul(w, cell.center[1]))
                } else {
                    air += 1;
                    ap = add(ap, mul(w, phi));
                    aw = add(aw, w);
                    ay_sum = add(ay_sum, mul(w, cell.center[1]))
                }
            }
            if liquid == 0 {
                continue;
            }
            if exterior {
                ap = add(ap, mul(lw, self.options.sparse_air_phi));
                let ly = div(ly_sum, lw.max(1e-9));
                let direction = if row.center[1] >= ly { 1.0 } else { -1.0 };
                ay_sum = add(ay_sum, mul(lw, add(ly, mul(direction, row.distance))));
                aw = add(aw, lw)
            }
            let cut = air > 0 || exterior;
            let mut value = if cut {
                ghost_theta(div(lp, lw.max(1e-9)), div(ap, aw.max(1e-9)))
            } else {
                1.0
            };
            let acceleration = fields.acceleration_fine;
            let gravity = add(add_sq(acceleration[0]), add_sq(acceleration[1])).sqrt();
            if cut
                && row.axis == 1
                && gravity > 1e-6
                && partial
                && acceleration[1] < 0.0
                && acceleration[0].abs() <= mul(1e-6, gravity)
            {
                let (height, ok) = self.planar_height(reduced, fields, row);
                let ly = div(ly_sum, lw.max(1e-9));
                let ay = div(ay_sum, aw.max(1e-9));
                if ok && ay > ly + 1e-6 && height > ly && height < ay {
                    value = div(height - ly, ay - ly).clamp(GHOST_FLUID_THETA_MIN, 1.0)
                }
            }
            if cut && row.kind == RowKind::MixedSeam {
                let factor = if full == 0.0 {
                    0.0
                } else if liquid_grad == 0.0 {
                    div(1.0, GHOST_FLUID_THETA_MIN)
                } else {
                    div(full, liquid_grad).clamp(0.0, div(1.0, GHOST_FLUID_THETA_MIN))
                };
                value = if factor > 0.0 { div(1.0, factor) } else { 0.0 }
            }
            let weighted = mul(row.dual_weight, open);
            if !(weighted > 1e-8) {
                continue;
            }
            result.pressure_weight[ri] = weighted;
            result.theta[ri] = value;
            result.active_rows[ri] = 1;
        }
        for cell in &self.source.cells {
            let s = cell.id as usize;
            if result.virtual_member[s] == 0 {
                continue;
            }
            let (mut axes, mut flux) = ([0.0; 3], [0.0; 6]);
            for &rid in &self.source.incidences[s] {
                let row = &self.source.rows[rid as usize];
                let ri = row.id as usize;
                if result.active_rows[ri] == 0 || result.theta[ri] <= 0.0 {
                    continue;
                }
                let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == s) else {
                    continue;
                };
                axes[row.axis as usize] = add(
                    axes[row.axis as usize],
                    div(
                        mul(
                            result.pressure_weight[ri],
                            mul(own.coefficient, own.coefficient),
                        ),
                        result.theta[ri],
                    ),
                );
                let value = mul(
                    own.coefficient,
                    mul(row.dual_weight, self.fluid_velocity(reduced, fields, row)),
                );
                let side = 2 * row.axis as usize + usize::from(own.coefficient <= 0.0);
                flux[side] = add(flux[side], value)
            }
            result.virtual_diagonal[s] = sum_axes(axes);
            result.virtual_rhs[s] = sum_sides(flux);
            let r = self.reduced_cell[s];
            if r >= 0 {
                result.virtual_rhs[s] = add(
                    result.virtual_rhs[s],
                    -mul(
                        Fields::optional_cell(&fields.capacity_rate, r as usize, 0.0),
                        cell.measure,
                    ),
                );
                result.virtual_rhs[s] =
                    add(result.virtual_rhs[s], self.source_rate(reduced, fields, s))
            }
        }
        for cell in &reduced.cells {
            let r = cell.id as usize;
            let s = self.centre_cell[r];
            if s < 0 || result.virtual_member[s as usize] == 0 {
                continue;
            }
            let s = s as usize;
            let (mut axes, mut flux) = ([0.0; 3], [0.0; 6]);
            for &rid in &self.source.incidences[s] {
                let row = &self.source.rows[rid as usize];
                let ri = row.id as usize;
                if result.active_rows[ri] == 0 || result.theta[ri] <= 0.0 {
                    continue;
                }
                let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == s) else {
                    continue;
                };
                axes[row.axis as usize] = add(
                    axes[row.axis as usize],
                    div(
                        mul(
                            result.pressure_weight[ri],
                            mul(own.coefficient, own.coefficient),
                        ),
                        result.theta[ri],
                    ),
                );
                let value = mul(
                    own.coefficient,
                    mul(row.dual_weight, self.fluid_velocity(reduced, fields, row)),
                );
                let side = 2 * row.axis as usize + usize::from(own.coefficient <= 0.0);
                flux[side] = add(flux[side], value)
            }
            let depth = self.source.cells[s].widths[2];
            result.diagonal[r] = div(sum_axes(axes), depth);
            result.rhs[r] = div(sum_sides(flux), depth);
            result.rhs[r] = add(
                result.rhs[r],
                -mul(
                    Fields::optional_cell(&fields.capacity_rate, r, 0.0),
                    cell.measure,
                ),
            );
            result.rhs[r] = add(
                result.rhs[r],
                Fields::optional_cell(&fields.source_rate, r, 0.0),
            )
        }
        fields.pressure_member.fill(0);
        for c in &reduced.cells {
            let s = self.centre_cell[c.id as usize];
            if s >= 0 {
                fields.pressure_member[c.id as usize] = result.virtual_member[s as usize]
            }
        }
        fields.pressure_row_member = reduced
            .rows
            .iter()
            .map(|r| {
                let s = self.centre_row[r.id as usize];
                if s >= 0 {
                    result.active_rows[s as usize]
                } else {
                    0
                }
            })
            .collect();
        self.pressure_member = result.virtual_member.clone();
        self.row_active.copy_from_slice(&result.active_rows);
        self.row_theta.copy_from_slice(&result.theta);
        self.cache_initialized = true;
        let mut virtual_fields = Fields::default();
        virtual_fields.density = self
            .source
            .cells
            .iter()
            .map(|c| {
                let r = self.reduced_cell[c.id as usize];
                if r < 0 {
                    0.0
                } else {
                    fields.density[r as usize]
                }
            })
            .collect();
        virtual_fields.capacity = self
            .source
            .cells
            .iter()
            .map(|c| {
                let r = self.reduced_cell[c.id as usize];
                if r < 0 {
                    0.0
                } else {
                    fields.capacity[r as usize]
                }
            })
            .collect();
        virtual_fields.interface_normal = (0..2 * nc)
            .map(|at| {
                let r = self.reduced_cell[at >> 1];
                if r < 0 {
                    0.0
                } else {
                    fields.interface_normal[2 * r as usize + (at & 1)]
                }
            })
            .collect();
        virtual_fields.pressure_diagonal = result.virtual_diagonal.clone();
        virtual_fields.pressure_member = result.virtual_member.clone();
        self.pressure_authority.publish(
            &self.source,
            &virtual_fields,
            &result.active_rows,
            &result.theta,
            self.options.source_generation,
            global,
        );
        result
    }

    pub fn solve(
        &mut self,
        reduced: &Graph,
        fields: &mut Fields,
        maximum_iterations: u32,
        relative_tolerance: f32,
        prepared: Option<PreparedEmbedding>,
    ) -> Result<EmbeddingSolveReceipt, PressureError> {
        let prepared = prepared.unwrap_or_else(|| self.prepare(reduced, fields));
        let solve = solve_pressure_pcg(
            &prepared.virtual_diagonal,
            &prepared.virtual_rhs,
            &self.pressure,
            &prepared.virtual_member,
            Some(&self.pressure_authority.execution_order),
            maximum_iterations,
            relative_tolerance,
            |input, output| apply_virtual_operator(&self.source, &prepared, input, output),
        )?;
        self.pressure.clone_from(&solve.pressure);
        for c in &reduced.cells {
            let r = c.id as usize;
            let s = self.centre_cell[r];
            if s >= 0 {
                fields.pressure[r] = self.pressure[s as usize];
                fields.pressure_diagonal[r] = prepared.diagonal[r];
                fields.pressure_rhs[r] = prepared.rhs[r]
            }
        }
        Ok(EmbeddingSolveReceipt { prepared, solve })
    }
    pub fn project(
        &self,
        reduced: &Graph,
        fields: &mut Fields,
        prepared: &PreparedEmbedding,
    ) -> Vec<f32> {
        let mut velocities = vec![0.0; self.source.rows.len()];
        for row in &self.source.rows {
            let row2 = self.row2(reduced, row);
            let base = if row.axis == 2 {
                0.0
            } else {
                row2.map_or(0.0, |r| fields.face_velocity[r.id as usize])
            };
            let ri = row.id as usize;
            if prepared.active_rows[ri] == 0 || prepared.theta[ri] <= 0.0 {
                velocities[ri] = base;
                continue;
            }
            let mut jump = 0.0;
            for t in &row.terms {
                if prepared.virtual_member[t.cell_id as usize] != 0 {
                    jump = add(jump, mul(t.coefficient, self.pressure[t.cell_id as usize]))
                }
            }
            velocities[ri] = base
                - mul(
                    self.open_fraction(reduced, fields, row),
                    div(jump, prepared.theta[ri]),
                )
        }
        for row in &reduced.rows {
            let s = self.centre_row[row.id as usize];
            if s >= 0 {
                fields.face_velocity[row.id as usize] = velocities[s as usize]
            }
        }
        velocities
    }
}

#[inline]
fn add_sq(v: f32) -> f32 {
    mul(v, v)
}
#[inline]
fn ghost_theta(liquid: f32, air: f32) -> f32 {
    ((liquid as f64).abs() / ((liquid as f64).abs() + (air as f64).abs()).max(1e-12_f64))
        .clamp(GHOST_FLUID_THETA_MIN as f64, 1.0) as f32
}
#[inline]
fn sum_axes(a: [f32; 3]) -> f32 {
    add(add(a[0], a[1]), a[2])
}
#[inline]
fn sum_sides(v: [f32; 6]) -> f32 {
    add(
        add(
            add(v[0].min(v[1]), v[0].max(v[1])),
            add(v[2].min(v[3]), v[2].max(v[3])),
        ),
        add(v[4].min(v[5]), v[4].max(v[5])),
    )
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
    let mut jump = 0.0;
    for t in &row.terms {
        if member[t.cell_id as usize] != 0 {
            jump = add(jump, mul(t.coefficient, input[t.cell_id as usize]))
        }
    }
    jump
}
fn apply_virtual_operator(
    graph: &Graph,
    prepared: &PreparedEmbedding,
    input: &[f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    for cell in &graph.cells {
        let s = cell.id as usize;
        if prepared.virtual_member[s] == 0 {
            continue;
        }
        let (mut neg, mut pos) = ([0.0; 3], [0.0; 3]);
        for &rid in &graph.incidences[s] {
            let row = &graph.rows[rid as usize];
            let ri = row.id as usize;
            if prepared.active_rows[ri] == 0 || prepared.theta[ri] <= 0.0 {
                continue;
            }
            let Some(own) = row.terms.iter().find(|t| t.cell_id as usize == s) else {
                continue;
            };
            let contribution = div(
                mul(
                    prepared.pressure_weight[ri],
                    mul(
                        own.coefficient,
                        row_gradient(row, &prepared.virtual_member, input),
                    ),
                ),
                prepared.theta[ri],
            );
            let target = if own.coefficient > 0.0 {
                &mut neg
            } else {
                &mut pos
            };
            target[row.axis as usize] = add(target[row.axis as usize], contribution)
        }
        output[s] = sum_axes([
            add(neg[0].min(pos[0]), neg[0].max(pos[0])),
            add(neg[1].min(pos[1]), neg[1].max(pos[1])),
            add(neg[2].min(pos[2]), neg[2].max(pos[2])),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cell, RowTerm};
    fn cell(id: u32, stable: u32, min: [f32; 3], max: [f32; 3]) -> Cell {
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let widths = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        Cell {
            id,
            stable_id: Some(stable),
            minimum: min,
            maximum: max,
            center,
            widths,
            measure: widths[0] * widths[1] * widths[2],
            brick_key: Some(0),
            refinement_region_scale: None,
        }
    }
    fn row(id: u32, axis: u8, center: [f32; 3], terms: Vec<RowTerm>) -> Row {
        Row {
            id,
            kind: RowKind::IntraBrick,
            axis,
            center,
            measure: 1.0,
            static_measure: None,
            distance: 1.0,
            dual_weight: 1.0,
            static_dual_weight: None,
            static_open_fraction: None,
            terms,
            open_fraction: 1.0,
            open_fraction_before: None,
            open_fraction_after: None,
            solid_velocity: 0.0,
            separating: false,
        }
    }
    #[test]
    fn maps_centre_and_preserves_z_diagonal() {
        let reduced = Graph {
            dimension: 2,
            dimensions: [1.0, 1.0, 1.0],
            cells: vec![cell(0, 0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])],
            rows: vec![row(
                0,
                0,
                [0.0, 0.5, 0.0],
                vec![RowTerm {
                    cell_id: 0,
                    coefficient: -1.0,
                }],
            )],
            incidences: vec![vec![0]],
            ..Default::default()
        };
        let source = Graph {
            dimension: 3,
            dimensions: [1.0, 1.0, 2.0],
            cells: vec![
                cell(0, 10, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                cell(1, 11, [0.0, 0.0, 1.0], [1.0, 1.0, 2.0]),
            ],
            rows: vec![
                row(
                    0,
                    0,
                    [0.0, 0.5, 0.5],
                    vec![RowTerm {
                        cell_id: 0,
                        coefficient: -1.0,
                    }],
                ),
                row(
                    1,
                    2,
                    [0.5, 0.5, 1.0],
                    vec![
                        RowTerm {
                            cell_id: 0,
                            coefficient: 1.0,
                        },
                        RowTerm {
                            cell_id: 1,
                            coefficient: -1.0,
                        },
                    ],
                ),
            ],
            incidences: vec![vec![0, 1], vec![1]],
            ..Default::default()
        };
        let mut fields = Fields {
            density: vec![1.0],
            capacity: vec![1.0],
            pressure: vec![0.0],
            pressure_rhs: vec![0.0],
            pressure_diagonal: vec![0.0],
            pressure_member: vec![0],
            face_velocity: vec![0.0],
            interface_normal: vec![0.0; 2],
            ..Default::default()
        };
        let mut e = PressureEmbedding::new(
            source,
            &reduced,
            EmbeddingOptions::default(),
            Some(&fields),
            None,
        );
        assert_eq!(e.centre_cell, vec![0]);
        assert_eq!(e.centre_row, vec![0]);
        let p = e.prepare(&reduced, &mut fields);
        assert_eq!(p.diagonal[0].to_bits(), 2.0f32.to_bits());
        assert_eq!(e.pressure_authority.execution_order, vec![0, 1]);
    }
    #[test]
    fn ambiguous_projected_row_faults() {
        let c = cell(0, 0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let r = row(
            0,
            0,
            [0.5, 0.0, 0.0],
            vec![RowTerm {
                cell_id: 0,
                coefficient: 1.0,
            }],
        );
        let reduced = Graph {
            dimension: 2,
            dimensions: [1.0, 1.0, 1.0],
            cells: vec![c.clone()],
            rows: vec![r.clone(), Row { id: 1, ..r.clone() }],
            incidences: vec![vec![0, 1]],
            ..Default::default()
        };
        let source = Graph {
            dimension: 3,
            dimensions: [1.0, 1.0, 1.0],
            cells: vec![c],
            rows: vec![r],
            incidences: vec![vec![0]],
            ..Default::default()
        };
        let e = PressureEmbedding::new(source, &reduced, EmbeddingOptions::default(), None, None);
        assert_eq!(
            e.mapping_fault,
            Some(MappingFault {
                kind: MappingFaultKind::Row,
                id: 0
            })
        );
    }
}
