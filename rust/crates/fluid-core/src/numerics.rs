//! Dimension-aware velocity, pressure-graph, and PLIC stage kernels.

use crate::kernels::{add, div, mul};
use crate::pressure::{solve_pressure_pcg, PressureError};
use crate::types::{Fields, Graph, Row, RowKind, SpatialOwnerCache, ValidationError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub use crate::band_projection::{
    project_receiver_band_rates_2d, BandProjectionReceipt2d, BAND_PROJECTION_NORMALIZED_TARGET,
};

pub const LIQUID_ISOVALUE: f32 = 0.5;
pub const GHOST_FLUID_THETA_MIN: f32 = 0.05;
const VOLUME_ROUNDOFF_RATIO: f32 = 9.536_743e-7;

#[inline]
pub fn physical_row_measure(row: &Row) -> f32 {
    mul(row.static_measure.unwrap_or(row.measure), row.open_fraction)
}
#[inline]
fn pressure_dual_weight(row: &Row) -> f32 {
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
}
#[inline]
fn fill(fields: &Fields, cell: usize) -> f32 {
    div(fields.density[cell], fields.capacity[cell].max(1e-8))
}

pub fn owner_at(graph: &Graph, point: [f32; 3]) -> Option<usize> {
    let d = graph.dimension as usize;
    match graph.spatial_owner_index.get_or_build(graph) {
        SpatialOwnerCache::Dense2d { dimensions, owners } => {
            if !(point[0] >= 0.0
                && point[0] < graph.dimensions[0]
                && point[1] >= 0.0
                && point[1] < graph.dimensions[1])
            {
                return None;
            }
            let x = point[0].floor() as usize;
            let y = point[1].floor() as usize;
            let id = owners[x + dimensions[0] * y];
            (id >= 0).then_some(id as usize)
        }
        SpatialOwnerCache::Sparse3d { widths, owners } => {
            if !(0..3).all(|a| point[a] >= 0.0 && point[a] < graph.dimensions[a]) {
                return None;
            }
            let q = point.map(|v| v.floor() as i32);
            widths
                .iter()
                .filter_map(|&width| {
                    let key = q.map(|v| v / (width as i32) * (width as i32));
                    let &id = owners.get(&(width, key))?;
                    let cell = &graph.cells[id as usize];
                    (0..3)
                        .all(|a| point[a] >= cell.minimum[a] && point[a] < cell.maximum[a])
                        .then_some(id as usize)
                })
                .min()
        }
        SpatialOwnerCache::LinearFallback => graph
            .cells
            .iter()
            .position(|c| (0..d).all(|a| point[a] >= c.minimum[a] && point[a] < c.maximum[a])),
    }
}

pub fn solid_voxel_fraction_at(graph: &Graph, x: i32, y: i32, z: i32) -> Option<f32> {
    if graph.solid_voxel_fraction.is_empty() {
        return None;
    }
    let nx = graph.dimensions[0] as i32;
    let ny = graph.dimensions[1] as i32;
    let nz = if graph.dimension == 2 {
        1
    } else {
        graph.dimensions[2] as i32
    };
    if x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz {
        return Some(1.0);
    }
    graph
        .solid_voxel_fraction
        .get((x + nx * (y + ny * z)) as usize)
        .copied()
}

pub fn solid_voxel_at(graph: &Graph, x: i32, y: i32, z: i32) -> bool {
    solid_voxel_fraction_at(graph, x, y, z).is_some_and(|fraction| fraction >= 128.0 / 255.0)
}

fn own_term(row: &Row, cell: usize) -> Option<crate::types::RowTerm> {
    row.terms
        .iter()
        .find(|t| t.cell_id as usize == cell)
        .copied()
}

fn velocity_extension_neighbor_weight(
    row: &Row,
    own: crate::types::RowTerm,
    other: crate::types::RowTerm,
) -> f32 {
    if own.coefficient * other.coefficient >= 0.0 {
        return 0.0;
    }
    let distance = row.distance;
    let own_fraction = mul(own.coefficient.abs(), distance);
    let other_fraction = mul(other.coefficient.abs(), distance);
    div(
        mul(mul(physical_row_measure(row), own_fraction), other_fraction),
        distance.max(1e-9),
    )
}

/// VEX2: one f32 seed publication followed by synchronous frozen-front transforms.
pub fn extend_velocity(
    graph: &Graph,
    fields: &mut Fields,
    depth_count: u8,
) -> Result<(), ValidationError> {
    fields.validate_for(graph)?;
    let n = graph.cells.len();
    let d = graph.dimension as usize;
    fields.extension_depth.fill(255);
    let mut velocity = fields.cell_velocity.clone();
    let mut known = vec![0u8; n];
    for cell in 0..n {
        if fields.density[cell] > LIQUID_ISOVALUE {
            known[cell] = 1;
            fields.extension_depth[cell] = 0;
        } else {
            for axis in 0..d {
                velocity[d * cell + axis] = 0.0;
            }
        }
    }
    for depth in 1..=depth_count {
        let mut next_velocity = velocity.clone();
        let mut next_known = known.clone();
        for cell in 0..n {
            if known[cell] != 0 {
                continue;
            }
            let mut side_weight = vec![0.0f32; 2 * d];
            let mut side_velocity = vec![vec![0.0f32; 2 * d]; d];
            for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                if physical_row_measure(row) <= 1e-8 {
                    continue;
                }
                let Some(own) = own_term(row, cell) else {
                    continue;
                };
                let side = 2 * row.axis as usize + usize::from(own.coefficient < 0.0);
                for &other in &row.terms {
                    let j = other.cell_id as usize;
                    if j == cell
                        || own.coefficient * other.coefficient >= 0.0
                        || known[j] == 0
                        || fields.extension_depth[j] >= depth
                    {
                        continue;
                    }
                    let weight = velocity_extension_neighbor_weight(row, own, other);
                    side_weight[side] = add(side_weight[side], weight);
                    for axis in 0..d {
                        side_velocity[axis][side] = add(
                            side_velocity[axis][side],
                            mul(weight, velocity[d * j + axis]),
                        );
                    }
                }
            }
            let pair_sum = |values: &[f32], axis: usize| {
                add(
                    values[2 * axis].min(values[2 * axis + 1]),
                    values[2 * axis].max(values[2 * axis + 1]),
                )
            };
            let mut weight = 0.0;
            for axis in 0..d {
                weight = add(weight, pair_sum(&side_weight, axis));
            }
            if weight > 0.0 {
                for component in 0..d {
                    let mut value = 0.0;
                    for axis in 0..d {
                        value = add(value, pair_sum(&side_velocity[component], axis));
                    }
                    next_velocity[d * cell + component] = div(value, weight);
                }
                next_known[cell] = 1;
                fields.extension_depth[cell] = depth;
            }
        }
        velocity = next_velocity;
        known = next_known;
    }
    fields.cell_velocity = velocity;
    Ok(())
}

/// Production body-force update, including unilateral closed-world release.
pub fn force_faces(
    graph: &mut Graph,
    fields: &mut Fields,
    dt: f32,
    acceleration: [f32; 3],
    inflow_velocity: [f32; 3],
) {
    let d = graph.dimension as usize;
    for row in &mut graph.rows {
        let i = row.id as usize;
        row.separating = false;
        if row.kind == RowKind::ClosedWorld {
            if let Some(term) = row.terms.first() {
                let cell = term.cell_id as usize;
                if fields.capacity[cell] > 1e-8 && fields.density[cell] > LIQUID_ISOVALUE {
                    let length = (0..d)
                        .map(|a| mul(acceleration[a], acceleration[a]))
                        .fold(0.0, add)
                        .sqrt();
                    let orientation = if term.coefficient >= 0.0 { 1.0 } else { -1.0 };
                    let predicted = add(
                        fields.cell_velocity[d * cell + row.axis as usize],
                        mul(dt, acceleration[row.axis as usize]),
                    );
                    let outward = mul(dt, mul(orientation, predicted - row.solid_velocity))
                        / row.distance.max(1e-6);
                    let deadband = if fields.pressure_row_member.get(i).copied().unwrap_or(0) != 0 {
                        5e-5
                    } else {
                        1e-4
                    };
                    row.separating = length > 1e-6
                        && mul(orientation, acceleration[row.axis as usize]) > mul(0.5, length)
                        && outward > deadband
                }
                fields.face_velocity[i] = if row.separating {
                    add(
                        fields.cell_velocity[d * cell + row.axis as usize],
                        mul(dt, acceleration[row.axis as usize]),
                    )
                } else {
                    row.solid_velocity
                }
            }
        } else if physical_row_measure(row) <= 1e-8 {
            fields.face_velocity[i] = row.solid_velocity
        } else {
            let forced = add(
                fields.face_velocity[i],
                mul(row.open_fraction, mul(dt, acceleration[row.axis as usize])),
            );
            let inflow = fields.inflow_coverage.get(i).copied().unwrap_or(0.0);
            fields.face_velocity[i] = add(
                mul(1.0 - inflow, forced),
                mul(inflow, inflow_velocity[row.axis as usize]),
            )
        }
    }
}

pub fn enforce_inflow_faces(graph: &Graph, fields: &mut Fields, inflow_velocity: [f32; 3]) {
    for row in &graph.rows {
        let i = row.id as usize;
        if physical_row_measure(row) <= 1e-8 {
            continue;
        }
        let coverage = fields.inflow_coverage.get(i).copied().unwrap_or(0.0);
        if coverage > 0.0 {
            fields.face_velocity[i] = add(
                mul(1.0 - coverage, fields.face_velocity[i]),
                mul(coverage, inflow_velocity[row.axis as usize]),
            )
        }
    }
}

fn face_support(
    graph: &Graph,
    fields: &Fields,
    x: f32,
    y: f32,
    require_extended: bool,
) -> ([f32; 2], f32, bool) {
    let Some(cell) = owner_at(graph, [x.floor() + 0.5, y.floor() + 0.5, 0.0]) else {
        return ([0.0; 2], 1.0, false);
    };
    let record = &graph.cells[cell];
    (
        [
            fields.cell_velocity[2 * cell],
            fields.cell_velocity[2 * cell + 1],
        ],
        record.widths[0].min(record.widths[1]).max(1.0),
        !require_extended || fields.extension_depth.get(cell).copied().unwrap_or(255) != 255,
    )
}
pub(crate) fn sample_support(
    graph: &Graph,
    fields: &Fields,
    x: f32,
    y: f32,
    span: f32,
) -> [f32; 2] {
    sample_support_with_weight(graph, fields, x, y, span, false).0
}

fn sample_support_with_weight(
    graph: &Graph,
    fields: &Fields,
    x: f32,
    y: f32,
    span: f32,
    renormalize_sparse_support: bool,
) -> ([f32; 2], f32) {
    let bx = x.clamp(0.5 * span, graph.dimensions[0] - 0.5 * span);
    let by = y.clamp(0.5 * span, graph.dimensions[1] - 0.5 * span);
    let sx = bx / span - 0.5;
    let sy = by / span - 0.5;
    let lx = sx.floor();
    let ly = sy.floor();
    let tx = sx - lx;
    let ty = sy - ly;
    let mut result = [0.0; 2];
    let mut represented_weight = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let weight = mul(
                if dx == 1 { tx } else { 1.0 - tx },
                if dy == 1 { ty } else { 1.0 - ty },
            );
            if weight == 0.0 {
                continue;
            }
            let (q, _, owned) = face_support(
                graph,
                fields,
                span * (lx + dx as f32 + 0.5),
                span * (ly + dy as f32 + 0.5),
                renormalize_sparse_support,
            );
            if owned {
                represented_weight = add(represented_weight, weight);
                for a in 0..2 {
                    result[a] = add(result[a], mul(weight, q[a]))
                }
            }
        }
    }
    if renormalize_sparse_support && represented_weight > 0.0 && represented_weight != 1.0 {
        for value in &mut result {
            *value = div(*value, represented_weight);
        }
    }
    (result, represented_weight)
}

#[derive(Clone, Copy, Debug)]
struct FaceVelocityPiece2d {
    tangent_minimum: f64,
    tangent_maximum: f64,
    velocity: f64,
}

#[derive(Clone, Debug, Default)]
struct CellFaceVelocity2d {
    /// x-minimum, x-maximum, y-minimum, y-maximum.
    sides: [Vec<FaceVelocityPiece2d>; 4],
}

/// Experimental face-consistent velocity field for the cellwise remap lab.
///
/// Every physical subface retains its own normal rate. Within a cell, the
/// x component is linear in x between the y-selected side pieces, and the y
/// component is linear in y between the x-selected pieces. A coarse seam side
/// therefore uses the same subface strips as its fine neighbours. The field is
/// RT0 within each matching strip, while mismatched opposite-side partitions
/// may introduce an internal tangential discontinuity.
#[derive(Clone, Debug)]
pub struct FaceConsistentVelocity2d {
    cells: Vec<CellFaceVelocity2d>,
    streamfunction: Option<StreamfunctionVelocity2d>,
    subface_rates: Vec<f64>,
    projected_subface_rates_preserved: usize,
    original_subface_rates_preserved: usize,
    extension_subface_rates_republished: usize,
    extension_depths: Vec<u8>,
    extension_generations: u8,
    extension_cell_visits: usize,
    dimensions: [f64; 2],
    lattice_dimensions: [usize; 2],
    nearest_owner: Vec<usize>,
    represented: Vec<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceVelocitySample2d {
    pub velocity: [f64; 2],
    pub support_fallback: bool,
    /// True only when sampling fell back from the compiled global
    /// streamfunction to a cell-local RT0 profile.
    pub velocity_model_fallback: bool,
    /// Exact piecewise-RT0 region selected for this point. A changed signature
    /// marks a sampled region transition where tangential velocity may jump.
    pub region: FaceVelocityRegion2d,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaceVelocityRegion2d {
    pub owner: usize,
    /// Selected x-minimum, x-maximum, y-minimum, y-maximum side pieces.
    pub side_pieces: [usize; 4],
}

#[derive(Clone, Debug)]
struct StreamfunctionVelocity2d {
    psi: Vec<f64>,
    dx: Vec<f64>,
    dy: Vec<f64>,
    active_lattice: Vec<bool>,
    dimensions: [usize; 2],
    max_integrated_flux_residual: f64,
}

/// Continue the constrained streamfunction through sparse topology holes.
/// The constrained graph supplies Dirichlet data; the remaining finest-grid
/// vertices minimize discrete Dirichlet energy.  Sampling this continuation
/// keeps pathlines in the same global curl field when impact-driven lateral
/// motion leaves the represented sparse cells.
fn fill_streamfunction_holes_harmonic(
    psi: &mut [f64],
    nx: usize,
    ny: usize,
) -> Result<(), ValidationError> {
    let stride = nx + 1;
    let mut unknown_of = vec![usize::MAX; psi.len()];
    let mut vertices = Vec::new();
    for (vertex, value) in psi.iter().enumerate() {
        if !value.is_finite() {
            unknown_of[vertex] = vertices.len();
            vertices.push(vertex);
        }
    }
    if vertices.is_empty() {
        return Ok(());
    }
    if vertices.len() == psi.len() {
        return Err(ValidationError(
            "streamfunction harmonic continuation has no constrained vertex".into(),
        ));
    }

    let neighbours = |vertex: usize| {
        let x = vertex % stride;
        let y = vertex / stride;
        [
            (x > 0).then(|| vertex - 1),
            (x < nx).then(|| vertex + 1),
            (y > 0).then(|| vertex - stride),
            (y < ny).then(|| vertex + stride),
        ]
    };
    let mut diagonal = vec![0.0; vertices.len()];
    let mut rhs = vec![0.0; vertices.len()];
    for (i, &vertex) in vertices.iter().enumerate() {
        for neighbour in neighbours(vertex).into_iter().flatten() {
            diagonal[i] += 1.0;
            if unknown_of[neighbour] == usize::MAX {
                rhs[i] += psi[neighbour];
            }
        }
    }
    let apply = |values: &[f64], result: &mut [f64]| {
        for (i, &vertex) in vertices.iter().enumerate() {
            let mut value = diagonal[i] * values[i];
            for neighbour in neighbours(vertex).into_iter().flatten() {
                let j = unknown_of[neighbour];
                if j != usize::MAX {
                    value -= values[j];
                }
            }
            result[i] = value;
        }
    };

    let mut solution = vec![0.0; vertices.len()];
    let mut residual = rhs.clone();
    let mut preconditioned: Vec<f64> = residual
        .iter()
        .zip(&diagonal)
        .map(|(&value, &scale)| value / scale)
        .collect();
    let mut direction = preconditioned.clone();
    let mut residual_dot_preconditioned = residual
        .iter()
        .zip(&preconditioned)
        .map(|(&a, &b)| a * b)
        .sum::<f64>();
    let rhs_scale = rhs.iter().map(|value| value.abs()).fold(1.0_f64, f64::max);
    let mut product = vec![0.0; vertices.len()];
    let mut converged = false;
    for _ in 0..4096 {
        let max_residual = residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if max_residual <= 1.0e-11 * rhs_scale {
            converged = true;
            break;
        }
        apply(&direction, &mut product);
        let denominator = direction
            .iter()
            .zip(&product)
            .map(|(&a, &b)| a * b)
            .sum::<f64>();
        if !denominator.is_finite() || denominator <= 0.0 {
            break;
        }
        let alpha = residual_dot_preconditioned / denominator;
        for i in 0..solution.len() {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * product[i];
            preconditioned[i] = residual[i] / diagonal[i];
        }
        let next = residual
            .iter()
            .zip(&preconditioned)
            .map(|(&a, &b)| a * b)
            .sum::<f64>();
        if !next.is_finite() {
            break;
        }
        let beta = next / residual_dot_preconditioned;
        for i in 0..direction.len() {
            direction[i] = preconditioned[i] + beta * direction[i];
        }
        residual_dot_preconditioned = next;
    }
    if !converged
        && residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            > 1.0e-9 * rhs_scale
    {
        return Err(ValidationError(
            "streamfunction harmonic continuation did not converge".into(),
        ));
    }
    for (i, &vertex) in vertices.iter().enumerate() {
        psi[vertex] = solution[i];
    }
    Ok(())
}

impl FaceConsistentVelocity2d {
    /// Compile an isolated post-projection tracing field. Pressure-owned rates
    /// stay exact; non-pressure rates are republished from a private velocity
    /// extension whose depth exceeds the measured liquid travel in fine units.
    pub fn new(graph: &Graph, fields: &Fields, dt: f32) -> Result<Self, ValidationError> {
        fields.validate_for(graph)?;
        if graph.dimension != 2 {
            return Err(ValidationError(
                "face-consistent velocity requires a 2-D graph".into(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ValidationError(
                "face-consistent velocity dt must be finite and positive".into(),
            ));
        }

        let original_rates: Vec<f64> = (0..graph.subfaces.len())
            .map(|face| crate::transport::physical_subface_rate(graph, fields, face) as f64)
            .collect();
        let maximum_liquid_travel = graph
            .subfaces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                (face.negative_cell >= 0 && fields.density[face.negative_cell as usize] > 0.0)
                    || (face.positive_cell >= 0
                        && fields.density[face.positive_cell as usize] > 0.0)
            })
            .map(|(i, face)| {
                dt as f64 * (original_rates[i] / face.measure.max(1.0e-8) as f64).abs()
            })
            .fold(0.0_f64, f64::max);
        if !maximum_liquid_travel.is_finite() || maximum_liquid_travel >= 253.0 {
            return Err(ValidationError(
                "face-consistent velocity extension depth exceeds the u8 receipt range".into(),
            ));
        }
        let extension_generations = (maximum_liquid_travel.ceil() as u8 + 1).max(8);
        let mut extended = fields.clone();
        // The remap defines liquid support as every cell with nonzero material,
        // while production velocity extension seeds only cells above the
        // interface isovalue. Promote densities only in this private clone so
        // partial liquid cells retain their projected velocity as seed data.
        for density in &mut extended.density {
            if *density > 0.0 {
                *density = 1.0;
            }
        }
        extend_velocity(graph, &mut extended, extension_generations)?;
        let extension_cell_visits = graph
            .cells
            .len()
            .checked_mul(extension_generations as usize)
            .ok_or_else(|| {
                ValidationError("face-consistent extension work count overflow".into())
            })?;

        let preserve_all_rates = fields.subface_compatibility_rate.len() == graph.subfaces.len();
        let mut subface_rates = original_rates;
        let projected_subface_rates_preserved = graph
            .subfaces
            .iter()
            .filter(|face| face_is_pressure_owned(fields, face))
            .count();
        let mut extension_subface_rates_republished = 0;
        if !preserve_all_rates {
            for face in &graph.subfaces {
                let index = face.id as usize;
                let row = &graph.rows[face.row_id as usize];
                if row.kind == RowKind::ClosedWorld || face_is_pressure_owned(fields, face) {
                    continue;
                }
                let mut velocity = 0.0_f64;
                let mut count = 0_u32;
                for cell in [face.negative_cell, face.positive_cell] {
                    if cell < 0 || extended.extension_depth[cell as usize] == 255 {
                        continue;
                    }
                    velocity +=
                        extended.cell_velocity[2 * cell as usize + face.axis as usize] as f64;
                    count += 1;
                }
                if count == 0 {
                    continue;
                }
                velocity /= count as f64;
                let correction = fields
                    .subface_velocity_correction
                    .get(index)
                    .copied()
                    .unwrap_or(0.0) as f64;
                subface_rates[index] =
                    face.measure as f64 * (row.open_fraction as f64 * velocity - correction);
                extension_subface_rates_republished += 1;
            }
        }
        let original_subface_rates_preserved =
            graph.subfaces.len() - extension_subface_rates_republished;

        let cells = compile_face_velocity_cells(graph, &subface_rates)?;

        let lattice_dimensions = [graph.dimensions[0] as usize, graph.dimensions[1] as usize];
        let lattice_count = lattice_dimensions[0]
            .checked_mul(lattice_dimensions[1])
            .ok_or_else(|| ValidationError("face-consistent owner lattice overflow".into()))?;
        let mut nearest_owner = vec![usize::MAX; lattice_count];
        let mut represented = vec![false; lattice_count];
        let mut frontier = VecDeque::new();
        for cell in &graph.cells {
            for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
                for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                    let at = x + lattice_dimensions[0] * y;
                    if represented[at] {
                        return Err(ValidationError(
                            "face-consistent velocity topology owners overlap".into(),
                        ));
                    }
                    represented[at] = true;
                    nearest_owner[at] = cell.id as usize;
                    frontier.push_back(at);
                }
            }
        }
        if frontier.is_empty() {
            return Err(ValidationError(
                "face-consistent velocity has no represented support".into(),
            ));
        }
        while let Some(at) = frontier.pop_front() {
            let x = at % lattice_dimensions[0];
            let y = at / lattice_dimensions[0];
            let owner = nearest_owner[at];
            let mut visit = |next: usize| {
                if nearest_owner[next] == usize::MAX {
                    nearest_owner[next] = owner;
                    frontier.push_back(next);
                }
            };
            if x > 0 {
                visit(at - 1);
            }
            if x + 1 < lattice_dimensions[0] {
                visit(at + 1);
            }
            if y > 0 {
                visit(at - lattice_dimensions[0]);
            }
            if y + 1 < lattice_dimensions[1] {
                visit(at + lattice_dimensions[0]);
            }
        }

        Ok(Self {
            cells,
            streamfunction: None,
            subface_rates,
            projected_subface_rates_preserved,
            original_subface_rates_preserved,
            extension_subface_rates_republished,
            extension_depths: extended.extension_depth,
            extension_generations,
            extension_cell_visits,
            dimensions: [graph.dimensions[0] as f64, graph.dimensions[1] as f64],
            lattice_dimensions,
            nearest_owner,
            represented,
        })
    }

    pub fn sample(&self, graph: &Graph, point: [f64; 2]) -> Result<[f64; 2], ValidationError> {
        Ok(self.sample_with_diagnostic(graph, point)?.velocity)
    }

    /// Rebuilds only the piecewise face profiles for a projected private rate
    /// field. Extension depths, sparse ownership, and work counters are
    /// retained exactly from the original tracing context.
    pub fn with_subface_rates(
        &self,
        graph: &Graph,
        rates: &[f64],
        receiver_band: &[bool],
    ) -> Result<Self, ValidationError> {
        if rates.len() != graph.subfaces.len()
            || receiver_band.len() != graph.cells.len()
            || rates.iter().any(|rate| !rate.is_finite())
        {
            return Err(ValidationError(
                "face-consistent projected rate size or value is invalid".into(),
            ));
        }
        let mut projected = self.clone();
        projected.cells = compile_face_velocity_cells(graph, rates)?;
        projected.streamfunction = Some(compile_streamfunction_velocity(
            graph,
            rates,
            receiver_band,
        )?);
        projected.subface_rates = rates.to_vec();
        Ok(projected)
    }

    /// Rebuild piecewise face profiles without imposing a global
    /// streamfunction. Geometry probes and the local closure intentionally
    /// admit compressible analytic fields, so their tracer must not require
    /// the all-face circulation compatibility used by band projection.
    pub fn with_piecewise_subface_rates(
        &self,
        graph: &Graph,
        rates: &[f64],
    ) -> Result<Self, ValidationError> {
        if rates.len() != graph.subfaces.len() || rates.iter().any(|rate| !rate.is_finite()) {
            return Err(ValidationError(
                "face-consistent local rate size or value is invalid".into(),
            ));
        }
        let mut projected = self.clone();
        projected.cells = compile_face_velocity_cells(graph, rates)?;
        projected.streamfunction = None;
        projected.subface_rates = rates.to_vec();
        Ok(projected)
    }

    /// Samples the represented field or, in a sparse support hole, the nearest
    /// represented cell's boundary-clamped profile. The nearest-owner fill is
    /// deterministic; callers must retain `support_fallback` in their receipt
    /// because extrapolation does not establish a smooth flow-map contract.
    pub fn sample_with_diagnostic(
        &self,
        graph: &Graph,
        point: [f64; 2],
    ) -> Result<FaceVelocitySample2d, ValidationError> {
        let bounded = [
            point[0].clamp(0.0, self.dimensions[0]),
            point[1].clamp(0.0, self.dimensions[1]),
        ];
        let lookup = [
            bounded[0].min(f32::from_bits((self.dimensions[0] as f32).to_bits() - 1) as f64),
            bounded[1].min(f32::from_bits((self.dimensions[1] as f32).to_bits() - 1) as f64),
        ];
        let lattice_x = lookup[0].floor() as usize;
        let lattice_y = lookup[1].floor() as usize;
        let lattice_at = lattice_x + self.lattice_dimensions[0] * lattice_y;
        let owner = self.nearest_owner[lattice_at];
        let support_fallback = !self.represented[lattice_at];
        let cell = &graph.cells[owner];
        if let Some(streamfunction) = &self.streamfunction {
            if streamfunction.active_lattice[lattice_at] {
                let (velocity, region) = sample_streamfunction_velocity(streamfunction, bounded)?;
                return Ok(FaceVelocitySample2d {
                    velocity,
                    support_fallback: false,
                    velocity_model_fallback: false,
                    region: FaceVelocityRegion2d {
                        owner,
                        side_pieces: region,
                    },
                });
            }
        }
        let profile = &self.cells[owner];
        let x = ((bounded[0] - cell.minimum[0] as f64) / cell.widths[0] as f64).clamp(0.0, 1.0);
        let y = ((bounded[1] - cell.minimum[1] as f64) / cell.widths[1] as f64).clamp(0.0, 1.0);
        let (left, left_piece) = piece_velocity(&profile.sides[0], bounded[1])?;
        let (right, right_piece) = piece_velocity(&profile.sides[1], bounded[1])?;
        let (bottom, bottom_piece) = piece_velocity(&profile.sides[2], bounded[0])?;
        let (top, top_piece) = piece_velocity(&profile.sides[3], bounded[0])?;
        Ok(FaceVelocitySample2d {
            velocity: [left + x * (right - left), bottom + y * (top - bottom)],
            support_fallback,
            velocity_model_fallback: true,
            region: FaceVelocityRegion2d {
                owner,
                side_pieces: [left_piece, right_piece, bottom_piece, top_piece],
            },
        })
    }

    pub fn subface_rates(&self) -> &[f64] {
        &self.subface_rates
    }

    pub fn projected_subface_rates_preserved(&self) -> usize {
        self.projected_subface_rates_preserved
    }

    pub fn original_subface_rates_preserved(&self) -> usize {
        self.original_subface_rates_preserved
    }

    pub fn extension_subface_rates_republished(&self) -> usize {
        self.extension_subface_rates_republished
    }

    pub fn extension_depths(&self) -> &[u8] {
        &self.extension_depths
    }

    pub fn extension_generations(&self) -> u8 {
        self.extension_generations
    }

    /// Number of represented-cell iterations performed by the synchronous
    /// extension loop (`cells * generations`), including already-known cells.
    pub fn extension_cell_visits(&self) -> usize {
        self.extension_cell_visits
    }

    /// Number of finest-unit lattice slots allocated and visited to compile
    /// deterministic sparse-support ownership, including unrepresented holes.
    pub fn support_lattice_entries(&self) -> usize {
        self.nearest_owner.len()
    }

    pub fn streamfunction_max_integrated_flux_residual(&self) -> f64 {
        self.streamfunction
            .as_ref()
            .map_or(0.0, |field| field.max_integrated_flux_residual)
    }

    pub(crate) fn streamfunction_scalar(
        &self,
        point: [f64; 2],
    ) -> Result<Option<f64>, ValidationError> {
        self.streamfunction
            .as_ref()
            .map(|field| sample_streamfunction_scalar(field, point))
            .transpose()
    }

    pub(crate) fn streamfunction_range(&self) -> f64 {
        self.streamfunction.as_ref().map_or(0.0, |field| {
            let (minimum, maximum) = field.psi.iter().copied().fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
            if minimum.is_finite() && maximum.is_finite() {
                maximum - minimum
            } else {
                0.0
            }
        })
    }
}

fn compile_streamfunction_velocity(
    graph: &Graph,
    rates: &[f64],
    receiver_band: &[bool],
) -> Result<StreamfunctionVelocity2d, ValidationError> {
    let nx = graph.dimensions[0] as usize;
    let ny = graph.dimensions[1] as usize;
    let stride = nx + 1;
    let count = stride * (ny + 1);
    let mut active_lattice = vec![false; nx * ny];
    for (cell, geometry) in graph.cells.iter().enumerate() {
        for y in geometry.minimum[1] as usize..geometry.maximum[1] as usize {
            for x in geometry.minimum[0] as usize..geometry.maximum[0] as usize {
                let at = x + nx * y;
                if receiver_band[cell] {
                    active_lattice[at] = true;
                }
            }
        }
    }
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); count];
    let mut constrain = |a: usize, b: usize, delta: f64| {
        adjacency[a].push((b, delta));
        adjacency[b].push((a, -delta));
    };
    for face in &graph.subfaces {
        let unit_rate = rates[face.id as usize] / face.measure.max(1.0e-8) as f64;
        if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            for y in lo..hi {
                constrain(x + stride * y, x + stride * (y + 1), unit_rate);
            }
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            for x in lo..hi {
                constrain(x + stride * y, x + 1 + stride * y, -unit_rate);
            }
        }
    }
    for edges in &mut adjacency {
        edges.sort_by_key(|&(other, _)| other);
    }
    let mut psi = vec![f64::NAN; count];
    let mut psi_path_magnitude = vec![0.0_f64; count];
    let mut psi_path_operations = vec![0_usize; count];
    for seed in 0..count {
        if adjacency[seed].is_empty() || psi[seed].is_finite() {
            continue;
        }
        psi[seed] = 0.0;
        let mut queue = VecDeque::from([seed]);
        while let Some(at) = queue.pop_front() {
            for &(next, delta) in &adjacency[at] {
                let candidate = psi[at] + delta;
                if !psi[next].is_finite() {
                    psi[next] = candidate;
                    psi_path_magnitude[next] = psi_path_magnitude[at] + delta.abs();
                    psi_path_operations[next] = psi_path_operations[at] + 1;
                    queue.push_back(next);
                } else {
                    let operations = psi_path_operations[next]
                        .saturating_add(psi_path_operations[at])
                        .saturating_add(2) as f64;
                    let gamma = operations * f64::EPSILON
                        / (1.0 - operations * f64::EPSILON);
                    let magnitude = psi_path_magnitude[next]
                        + psi_path_magnitude[at]
                        + delta.abs();
                    if (psi[next] - candidate).abs() > gamma * magnitude {
                        return Err(ValidationError(
                            "streamfunction face integrals are not discretely compatible".into(),
                        ));
                    }
                }
            }
        }
    }
    // Adaptive cells have face constraints only on their perimeter. Populate
    // their finest-lattice interior with the Coons interpolant of those four
    // boundary traces. It agrees with every constrained subface endpoint and
    // supplies one global scalar field across 2:1 hanging nodes.
    for (cell, geometry) in graph.cells.iter().enumerate() {
        if !receiver_band[cell] {
            continue;
        }
        let x0 = geometry.minimum[0] as usize;
        let x1 = geometry.maximum[0] as usize;
        let y0 = geometry.minimum[1] as usize;
        let y1 = geometry.maximum[1] as usize;
        let width = (x1 - x0) as f64;
        let height = (y1 - y0) as f64;
        for y in y0 + 1..y1 {
            let ty = (y - y0) as f64 / height;
            for x in x0 + 1..x1 {
                let tx = (x - x0) as f64 / width;
                let left = psi[x0 + stride * y];
                let right = psi[x1 + stride * y];
                let bottom = psi[x + stride * y0];
                let top = psi[x + stride * y1];
                let c00 = psi[x0 + stride * y0];
                let c10 = psi[x1 + stride * y0];
                let c01 = psi[x0 + stride * y1];
                let c11 = psi[x1 + stride * y1];
                if [left, right, bottom, top, c00, c10, c01, c11]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(ValidationError(
                        "streamfunction adaptive-cell boundary is incomplete".into(),
                    ));
                }
                let boundary = (1.0 - tx) * left + tx * right + (1.0 - ty) * bottom + ty * top;
                let corners = (1.0 - tx) * (1.0 - ty) * c00
                    + tx * (1.0 - ty) * c10
                    + (1.0 - tx) * ty * c01
                    + tx * ty * c11;
                psi[x + stride * y] = boundary - corners;
            }
        }
    }
    // Retain derivatives defined solely by receiver-cell samples. The global
    // continuation shares these vertex derivatives, so its first patch is C1
    // with the accepted receiver field without feeding exterior values back
    // into that field's Hermite degrees of freedom.
    let mut receiver_psi = vec![f64::NAN; count];
    for (cell, geometry) in graph.cells.iter().enumerate() {
        if !receiver_band[cell] {
            continue;
        }
        let x0 = geometry.minimum[0] as usize;
        let x1 = geometry.maximum[0] as usize;
        let y0 = geometry.minimum[1] as usize;
        let y1 = geometry.maximum[1] as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                receiver_psi[x + stride * y] = psi[x + stride * y];
            }
        }
    }
    fill_streamfunction_holes_harmonic(&mut psi, nx, ny)?;
    active_lattice.fill(true);
    let mut max_integrated_flux_residual = 0.0_f64;
    for face in &graph.subfaces {
        let (flux, endpoint_magnitude) = if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            let a = psi[x + stride * lo];
            let b = psi[x + stride * hi];
            (b - a, a.abs() + b.abs())
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            let a = psi[lo + stride * y];
            let b = psi[hi + stride * y];
            (-(b - a), a.abs() + b.abs())
        };
        if !flux.is_finite() {
            return Err(ValidationError(format!(
                "streamfunction face {} has incomplete endpoint support",
                face.id
            )));
        }
        let residual = (flux - rates[face.id as usize]).abs();
        max_integrated_flux_residual = max_integrated_flux_residual.max(residual);
        // The integral is a subtraction of two accumulated potential values.
        // Bound its floating-point cancellation by those operands rather than
        // by the (possibly near-zero) flux alone.
        let endpoint_ids = if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            [x + stride * lo, x + stride * hi]
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            [lo + stride * y, hi + stride * y]
        };
        let operations = psi_path_operations[endpoint_ids[0]]
            .saturating_add(psi_path_operations[endpoint_ids[1]])
            .saturating_add(4) as f64;
        let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
        let unit_rate = rates[face.id as usize] / face.measure.max(1.0e-8) as f64;
        let subdivision_discrepancy =
            (face.measure as f64 * unit_rate - rates[face.id as usize]).abs();
        let tolerance = subdivision_discrepancy
            + gamma
            * (psi_path_magnitude[endpoint_ids[0]]
                + psi_path_magnitude[endpoint_ids[1]]
                + endpoint_magnitude
                + flux.abs()
                + rates[face.id as usize].abs());
        if residual > tolerance {
            return Err(ValidationError(format!(
                "streamfunction face {} integrated flux residual {residual} exceeds {tolerance}",
                face.id
            )));
        }
    }
    let derivative = |values: &[f64], x: usize, y: usize, axis: usize| -> f64 {
        let at = |qx: usize, qy: usize| values[qx + stride * qy];
        let center = at(x, y);
        let (low, high, distance) = if axis == 0 {
            let low = x.saturating_sub(1);
            let high = (x + 1).min(nx);
            (at(low, y), at(high, y), (high - low) as f64)
        } else {
            let low = y.saturating_sub(1);
            let high = (y + 1).min(ny);
            (at(x, low), at(x, high), (high - low) as f64)
        };
        if low.is_finite() && high.is_finite() && distance > 0.0 {
            (high - low) / distance
        } else if center.is_finite() && high.is_finite() {
            high - center
        } else if center.is_finite() && low.is_finite() {
            center - low
        } else {
            0.0
        }
    };
    let mut dx = vec![0.0; count];
    let mut dy = vec![0.0; count];
    for y in 0..=ny {
        for x in 0..=nx {
            let at = x + stride * y;
            if psi[at].is_finite() {
                let derivative_field = if receiver_psi[at].is_finite() {
                    &receiver_psi
                } else {
                    &psi
                };
                dx[at] = derivative(derivative_field, x, y, 0);
                dy[at] = derivative(derivative_field, x, y, 1);
            }
        }
    }
    // Endpoint equality on a closed wall proves only its integrated normal
    // flux.  Hermite interpolation also needs the tangential derivative of
    // psi to vanish at both endpoints in order to preserve no-penetration
    // pointwise, including the first and last face of a constrained run.
    for face in &graph.subfaces {
        if graph.rows[face.row_id as usize].kind != RowKind::ClosedWorld {
            continue;
        }
        if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            for y in lo..=hi {
                dy[x + stride * y] = 0.0;
            }
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            for x in lo..=hi {
                dx[x + stride * y] = 0.0;
            }
        }
    }
    Ok(StreamfunctionVelocity2d {
        psi,
        dx,
        dy,
        active_lattice,
        dimensions: [nx, ny],
        max_integrated_flux_residual,
    })
}

fn hermite(t: f64) -> ([f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        [2.0 * t3 - 3.0 * t2 + 1.0, -2.0 * t3 + 3.0 * t2],
        [t3 - 2.0 * t2 + t, t3 - t2],
        [6.0 * t2 - 6.0 * t, -6.0 * t2 + 6.0 * t],
        [3.0 * t2 - 4.0 * t + 1.0, 3.0 * t2 - 2.0 * t],
    )
}

fn sample_streamfunction_velocity(
    field: &StreamfunctionVelocity2d,
    point: [f64; 2],
) -> Result<([f64; 2], [usize; 4]), ValidationError> {
    let [nx, ny] = field.dimensions;
    let x0 = (point[0].floor() as usize).min(nx.saturating_sub(1));
    let y0 = (point[1].floor() as usize).min(ny.saturating_sub(1));
    let tx = (point[0] - x0 as f64).clamp(0.0, 1.0);
    let ty = (point[1] - y0 as f64).clamp(0.0, 1.0);
    let stride = nx + 1;
    let ids = [
        x0 + stride * y0,
        x0 + 1 + stride * y0,
        x0 + stride * (y0 + 1),
        x0 + 1 + stride * (y0 + 1),
    ];
    if ids.iter().any(|&id| !field.psi[id].is_finite()) {
        return Err(ValidationError(
            "streamfunction sample is outside its compatible support".into(),
        ));
    }
    let (hx, kx, dhx, dkx) = hermite(tx);
    let (hy, ky, dhy, dky) = hermite(ty);
    let mut psi_x = 0.0;
    let mut psi_y = 0.0;
    for j in 0..2 {
        for i in 0..2 {
            let id = ids[2 * j + i];
            psi_x += dhx[i] * hy[j] * field.psi[id]
                + dkx[i] * hy[j] * field.dx[id]
                + dhx[i] * ky[j] * field.dy[id];
            psi_y += hx[i] * dhy[j] * field.psi[id]
                + kx[i] * dhy[j] * field.dx[id]
                + hx[i] * dky[j] * field.dy[id];
        }
    }
    Ok(([psi_y, -psi_x], ids))
}

fn sample_streamfunction_scalar(
    field: &StreamfunctionVelocity2d,
    point: [f64; 2],
) -> Result<f64, ValidationError> {
    let [nx, ny] = field.dimensions;
    let x0 = (point[0].floor() as usize).min(nx.saturating_sub(1));
    let y0 = (point[1].floor() as usize).min(ny.saturating_sub(1));
    let tx = (point[0] - x0 as f64).clamp(0.0, 1.0);
    let ty = (point[1] - y0 as f64).clamp(0.0, 1.0);
    let stride = nx + 1;
    let ids = [
        x0 + stride * y0,
        x0 + 1 + stride * y0,
        x0 + stride * (y0 + 1),
        x0 + 1 + stride * (y0 + 1),
    ];
    if ids.iter().any(|&id| !field.psi[id].is_finite()) {
        return Err(ValidationError(
            "streamfunction scalar sample is outside its compatible support".into(),
        ));
    }
    let (hx, kx, _, _) = hermite(tx);
    let (hy, ky, _, _) = hermite(ty);
    let mut psi = 0.0;
    for j in 0..2 {
        for i in 0..2 {
            let id = ids[2 * j + i];
            psi += hx[i] * hy[j] * field.psi[id]
                + kx[i] * hy[j] * field.dx[id]
                + hx[i] * ky[j] * field.dy[id];
        }
    }
    Ok(psi)
}

fn face_is_pressure_owned(fields: &Fields, face: &crate::Subface) -> bool {
    [face.negative_cell, face.positive_cell]
        .into_iter()
        .filter(|&cell| cell >= 0)
        .any(|cell| {
            fields
                .pressure_member
                .get(cell as usize)
                .copied()
                .unwrap_or(0)
                != 0
        })
}

/// Replace the adjustable receiver-band rates by the minimum-energy
/// streamfunction extension of the authoritative pressure-owned fluxes.
/// Closed-wall fluxes remain hard constraints. Pressure-owned constraints are
/// added in canonical order; only a constraint that closes an incompatible
/// f32 circulation cycle becomes adjustable, and its eventual change must fit
/// the derived physical-rate roundoff bound. Rates republished from the
/// resulting scalar field are exactly divergence-free by telescoping around
/// every receiver cell.
pub(crate) fn streamfunction_extension_rates_2d(
    graph: &Graph,
    fields: &Fields,
    rates: &mut [f64],
    receiver_band: &[bool],
    globalize: bool,
) -> Result<(), ValidationError> {
    #[derive(Clone, Copy)]
    struct Edge {
        a: usize,
        b: usize,
        delta: f64,
        fixed: bool,
        candidate: bool,
        face: usize,
    }

    let supplied_rates = rates.to_vec();
    let nx = graph.dimensions[0] as usize;
    let ny = graph.dimensions[1] as usize;
    let stride = nx + 1;
    let count = stride * (ny + 1);
    let mut soft_pressure_boundary = vec![false; graph.subfaces.len()];
    let mut edges = Vec::<Edge>::new();
    let mut face_edges = vec![Vec::<usize>::new(); graph.subfaces.len()];
    for face in &graph.subfaces {
        let touches_receiver = [face.negative_cell, face.positive_cell]
            .into_iter()
            .filter(|&cell| cell >= 0)
            .any(|cell| receiver_band[cell as usize]);
        let closed = graph.rows[face.row_id as usize].kind == RowKind::ClosedWorld;
        let pressure_owned = face_is_pressure_owned(fields, face);
        if !touches_receiver && !(globalize && (closed || pressure_owned)) {
            continue;
        }
        let fixed = closed;
        let candidate = pressure_owned && !closed;
        let unit_delta = if face.axis == 0 {
            rates[face.id as usize] / face.measure.max(1.0e-8) as f64
        } else {
            -rates[face.id as usize] / face.measure.max(1.0e-8) as f64
        };
        if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            for y in lo..hi {
                face_edges[face.id as usize].push(edges.len());
                edges.push(Edge {
                    a: x + stride * y,
                    b: x + stride * (y + 1),
                    delta: unit_delta,
                    fixed,
                    candidate,
                    face: face.id as usize,
                });
            }
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            for x in lo..hi {
                face_edges[face.id as usize].push(edges.len());
                edges.push(Edge {
                    a: x + stride * y,
                    b: x + 1 + stride * y,
                    delta: unit_delta,
                    fixed,
                    candidate,
                    face: face.id as usize,
                });
            }
        }
    }
    #[derive(Clone)]
    struct PotentialUnion {
        parent: Vec<usize>,
        potential: Vec<f64>,
    }
    impl PotentialUnion {
        fn new(count: usize) -> Self {
            Self {
                parent: (0..count).collect(),
                potential: vec![0.0; count],
            }
        }
        fn find(&mut self, at: usize) -> (usize, f64) {
            let parent = self.parent[at];
            if parent == at {
                return (at, 0.0);
            }
            let (root, above) = self.find(parent);
            self.potential[at] += above;
            self.parent[at] = root;
            (root, self.potential[at])
        }
        /// Add psi[b]-psi[a]=delta. Return the existing-cycle residual when
        /// the endpoints were already connected.
        fn constrain(&mut self, a: usize, b: usize, delta: f64) -> Option<(f64, f64)> {
            let (ra, pa) = self.find(a);
            let (rb, pb) = self.find(b);
            if ra == rb {
                return Some(((pb - pa) - delta, pa.abs().max(pb.abs()).max(delta.abs()).max(1.0)));
            }
            if ra < rb {
                self.parent[rb] = ra;
                self.potential[rb] = delta + pa - pb;
            } else {
                self.parent[ra] = rb;
                self.potential[ra] = pb - pa - delta;
            }
            None
        }
    }
    let cycle_tolerance = |scale: f64| 4096.0 * f64::EPSILON * scale;
    let mut constraints = PotentialUnion::new(count);
    for edge in edges.iter().filter(|edge| edge.fixed) {
        if let Some((residual, scale)) = constraints.constrain(edge.a, edge.b, edge.delta) {
            if residual.abs() > cycle_tolerance(scale) {
                return Err(ValidationError(format!(
                    "hard pressure/closed streamfunction cycle disagrees by {} on face {}",
                    residual.abs(), edge.face
                )));
            }
        }
    }
    for face in 0..graph.subfaces.len() {
        let candidates: Vec<_> = face_edges[face]
            .iter()
            .copied()
            .filter(|&edge| edges[edge].candidate)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let mut trial = constraints.clone();
        let compatible = candidates.iter().all(|&edge| {
            let edge = edges[edge];
            match trial.constrain(edge.a, edge.b, edge.delta) {
                Some((residual, scale)) => residual.abs() <= cycle_tolerance(scale),
                None => true,
            }
        });
        if compatible {
            constraints = trial;
            for edge in candidates {
                edges[edge].fixed = true;
            }
        } else {
            soft_pressure_boundary[face] = true;
        }
    }
    let mut fixed_adjacency = vec![Vec::<(usize, f64, usize)>::new(); count];
    for edge in edges.iter().filter(|edge| edge.fixed) {
        fixed_adjacency[edge.a].push((edge.b, edge.delta, edge.face));
        fixed_adjacency[edge.b].push((edge.a, -edge.delta, edge.face));
    }
    for adjacency in &mut fixed_adjacency {
        adjacency.sort_by_key(|&(other, _, face)| (other, face));
    }
    let mut rate_divergence = vec![0.0_f64; graph.cells.len()];
    for (subface, &rate) in graph.subfaces.iter().zip(rates.iter()) {
        if subface.negative_cell >= 0 {
            rate_divergence[subface.negative_cell as usize] += rate;
        }
        if subface.positive_cell >= 0 {
            rate_divergence[subface.positive_cell as usize] -= rate;
        }
    }
    let mut psi = vec![f64::NAN; count];
    for seed in 0..count {
        if fixed_adjacency[seed].is_empty() || psi[seed].is_finite() {
            continue;
        }
        psi[seed] = 0.0;
        let mut queue = VecDeque::from([seed]);
        while let Some(at) = queue.pop_front() {
            for &(next, delta, face) in &fixed_adjacency[at] {
                let candidate = psi[at] + delta;
                if !psi[next].is_finite() {
                    psi[next] = candidate;
                    queue.push_back(next);
                } else {
                    let residual = (psi[next] - candidate).abs();
                    let tolerance =
                        4096.0 * f64::EPSILON * psi[next].abs().max(candidate.abs()).max(1.0);
                    if residual > tolerance {
                        let subface = &graph.subfaces[face];
                        return Err(ValidationError(format!(
                            "pressure-owned streamfunction face {face} constraint {at}->{next} disagrees by {residual}; axis={} center={:?} measure={} cells=({}, {}) pressure=({}, {}) divergence=({}, {}) density=({}, {}) capacity=({}, {}) soft={:?}",
                            subface.axis,
                            subface.center,
                            subface.measure,
                            subface.negative_cell,
                            subface.positive_cell,
                            if subface.negative_cell >= 0 { fields.pressure_member[subface.negative_cell as usize] } else { 0 },
                            if subface.positive_cell >= 0 { fields.pressure_member[subface.positive_cell as usize] } else { 0 },
                            if subface.negative_cell >= 0 { rate_divergence[subface.negative_cell as usize] } else { 0.0 },
                            if subface.positive_cell >= 0 { rate_divergence[subface.positive_cell as usize] } else { 0.0 },
                            if subface.negative_cell >= 0 { fields.density[subface.negative_cell as usize] } else { 0.0 },
                            if subface.positive_cell >= 0 { fields.density[subface.positive_cell as usize] } else { 0.0 },
                            if subface.negative_cell >= 0 { fields.capacity[subface.negative_cell as usize] } else { 0.0 },
                            if subface.positive_cell >= 0 { fields.capacity[subface.positive_cell as usize] } else { 0.0 },
                            soft_pressure_boundary.iter().enumerate().filter_map(|(i, &soft)| soft.then_some(i)).collect::<Vec<_>>(),
                        )));
                    }
                }
            }
        }
    }
    let mut active_vertex = vec![false; count];
    for edge in &edges {
        active_vertex[edge.a] = true;
        active_vertex[edge.b] = true;
    }
    if psi.iter().all(|value| !value.is_finite()) {
        let seed = active_vertex
            .iter()
            .position(|&active| active)
            .ok_or_else(|| ValidationError("streamfunction receiver band is empty".into()))?;
        psi[seed] = 0.0;
    }
    let mut variable = vec![usize::MAX; count];
    let mut vertices = Vec::new();
    for vertex in 0..count {
        if active_vertex[vertex] && !psi[vertex].is_finite() {
            variable[vertex] = vertices.len();
            vertices.push(vertex);
        }
    }
    let unknowns = vertices.len();
    let mut diagonal = vec![0.0_f64; unknowns];
    let mut rhs = vec![0.0_f64; unknowns];
    let mut neighbours = vec![Vec::<usize>::new(); unknowns];
    for edge in &edges {
        let ia = variable[edge.a];
        let ib = variable[edge.b];
        if ia != usize::MAX {
            diagonal[ia] += 1.0;
            rhs[ia] -= edge.delta;
            if ib != usize::MAX {
                neighbours[ia].push(ib);
            } else {
                rhs[ia] += psi[edge.b];
            }
        }
        if ib != usize::MAX {
            diagonal[ib] += 1.0;
            rhs[ib] += edge.delta;
            if ia != usize::MAX {
                neighbours[ib].push(ia);
            } else {
                rhs[ib] += psi[edge.a];
            }
        }
    }
    let apply = |input: &[f64], output: &mut [f64]| {
        for i in 0..unknowns {
            output[i] = neighbours[i]
                .iter()
                .fold(diagonal[i] * input[i], |value, &other| value - input[other]);
        }
    };
    let mut solution = vec![0.0_f64; unknowns];
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut image = vec![0.0_f64; unknowns];
    let mut residual_squared = residual.iter().map(|value| value * value).sum::<f64>();
    let rhs_scale = rhs.iter().map(|value| value.abs()).fold(1.0_f64, f64::max);
    for _ in 0..8192 {
        if residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            <= 256.0 * f64::EPSILON * rhs_scale
        {
            break;
        }
        apply(&direction, &mut image);
        let curvature = direction
            .iter()
            .zip(&image)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        if !(curvature > 0.0 && curvature.is_finite()) {
            return Err(ValidationError(
                "streamfunction extension solve lost positive curvature".into(),
            ));
        }
        let alpha = residual_squared / curvature;
        for i in 0..unknowns {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * image[i];
        }
        let next = residual.iter().map(|value| value * value).sum::<f64>();
        if next == 0.0 {
            residual_squared = 0.0;
            break;
        }
        let beta = next / residual_squared;
        for i in 0..unknowns {
            direction[i] = residual[i] + beta * direction[i];
        }
        residual_squared = next;
    }
    if residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        > 1.0e-10 * rhs_scale
    {
        return Err(ValidationError(
            "streamfunction extension solve did not converge".into(),
        ));
    }
    for (i, &vertex) in vertices.iter().enumerate() {
        psi[vertex] = solution[i];
    }
    if globalize {
        // Continue the accepted receiver solution without adding remote
        // dry-air targets to its variational problem. This happens only after
        // the conservative receiver band has stabilized, so the private
        // continuation cannot feed back into band construction.
        fill_streamfunction_holes_harmonic(&mut psi, nx, ny)?;
    }
    for face in &graph.subfaces {
        let fixed = (face_is_pressure_owned(fields, face)
            && !soft_pressure_boundary[face.id as usize])
            || graph.rows[face.row_id as usize].kind == RowKind::ClosedWorld;
        if fixed || (!globalize && face_edges[face.id as usize].is_empty()) {
            continue;
        }
        let delta = if globalize && face_edges[face.id as usize].is_empty() {
            if face.axis == 0 {
                let x = face.center[0].round() as usize;
                let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
                let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
                psi[x + stride * hi] - psi[x + stride * lo]
            } else {
                let y = face.center[1].round() as usize;
                let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
                let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
                psi[hi + stride * y] - psi[lo + stride * y]
            }
        } else if face.axis == 0 {
            let x = face.center[0].round() as usize;
            let lo = (face.center[1] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[1] + 0.5 * face.measure).round() as usize;
            psi[x + stride * hi] - psi[x + stride * lo]
        } else {
            let y = face.center[1].round() as usize;
            let lo = (face.center[0] - 0.5 * face.measure).round() as usize;
            let hi = (face.center[0] + 0.5 * face.measure).round() as usize;
            psi[hi + stride * y] - psi[lo + stride * y]
        };
        rates[face.id as usize] = if face.axis == 0 { delta } else { -delta };
    }
    for face in &graph.subfaces {
        if !soft_pressure_boundary[face.id as usize] {
            continue;
        }
        let change = (rates[face.id as usize] - supplied_rates[face.id as usize]).abs();
        let mut bound = 0.0_f64;
        for cell in [face.negative_cell, face.positive_cell]
            .into_iter()
            .filter(|&cell| cell >= 0 && fields.pressure_member[cell as usize] != 0)
        {
            let cell = cell as usize;
            let incident_faces = graph
                .subfaces
                .iter()
                .filter(|subface| {
                    subface.negative_cell == cell as i32
                        || subface.positive_cell == cell as i32
                })
                .collect::<Vec<_>>();
            let magnitude = incident_faces
                .iter()
                .map(|subface| supplied_rates[subface.id as usize].abs())
                .sum::<f64>();
            let operations = (8 * incident_faces.len() + 8) as f64;
            let gamma = operations * f32::EPSILON as f64
                / (1.0 - operations * f32::EPSILON as f64);
            let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
            let residual_floor = if fields.frame_dt > 0.0 {
                BAND_PROJECTION_NORMALIZED_TARGET * capacity
                    / (64.0 * fields.frame_dt as f64)
            } else {
                0.0
            };
            bound = bound.max(3.0 * gamma * magnitude + residual_floor);
        }
        if change > bound {
            return Err(ValidationError(format!(
                "pressure free-surface streamfunction reconciliation changed face {} by {change}, above derived f32 roundoff bound {bound}",
                face.id
            )));
        }
    }
    let mut divergence = vec![0.0_f64; graph.cells.len()];
    for (face, &rate) in graph.subfaces.iter().zip(rates.iter()) {
        if face.negative_cell >= 0 {
            divergence[face.negative_cell as usize] += rate;
        }
        if face.positive_cell >= 0 {
            divergence[face.positive_cell as usize] -= rate;
        }
    }
    if let Some((cell, value)) = divergence
        .iter()
        .copied()
        .enumerate()
        .filter(|&(cell, _)| receiver_band[cell])
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
    {
        let scale = graph.incidences[cell]
            .iter()
            .map(|&row| graph.rows[row as usize].measure as f64)
            .sum::<f64>()
            .max(1.0);
        if value.abs() > 4096.0 * f64::EPSILON * scale {
            return Err(ValidationError(format!(
                "streamfunction receiver cell {cell} has non-telescoping divergence {value}"
            )));
        }
    }
    Ok(())
}

fn compile_face_velocity_cells(
    graph: &Graph,
    subface_rates: &[f64],
) -> Result<Vec<CellFaceVelocity2d>, ValidationError> {
    let mut cells = vec![CellFaceVelocity2d::default(); graph.cells.len()];
    for face in &graph.subfaces {
        let axis = face.axis as usize;
        let tangent = 1 - axis;
        let piece = FaceVelocityPiece2d {
            tangent_minimum: face.center[tangent] as f64 - 0.5 * face.measure as f64,
            tangent_maximum: face.center[tangent] as f64 + 0.5 * face.measure as f64,
            velocity: subface_rates[face.id as usize] / face.measure as f64,
        };
        if face.negative_cell >= 0 {
            cells[face.negative_cell as usize].sides[2 * axis + 1].push(piece);
        }
        if face.positive_cell >= 0 {
            cells[face.positive_cell as usize].sides[2 * axis].push(piece);
        }
    }
    for (cell_index, profile) in cells.iter_mut().enumerate() {
        let cell = &graph.cells[cell_index];
        for side in 0..4 {
            let tangent = 1 - side / 2;
            let pieces = &mut profile.sides[side];
            pieces.sort_by(|a, b| {
                a.tangent_minimum
                    .total_cmp(&b.tangent_minimum)
                    .then(a.tangent_maximum.total_cmp(&b.tangent_maximum))
            });
            let expected_minimum = cell.minimum[tangent] as f64;
            let expected_maximum = cell.maximum[tangent] as f64;
            let tolerance = 32.0 * f32::EPSILON as f64 * cell.widths[tangent] as f64;
            if pieces.is_empty()
                || (pieces[0].tangent_minimum - expected_minimum).abs() > tolerance
                || (pieces.last().unwrap().tangent_maximum - expected_maximum).abs() > tolerance
                || pieces.windows(2).any(|pair| {
                    (pair[0].tangent_maximum - pair[1].tangent_minimum).abs() > tolerance
                })
            {
                return Err(ValidationError(format!(
                    "face-consistent velocity has incomplete side {side} for cell {cell_index}"
                )));
            }
        }
    }
    Ok(cells)
}

fn piece_velocity(
    pieces: &[FaceVelocityPiece2d],
    tangent: f64,
) -> Result<(f64, usize), ValidationError> {
    let index = pieces
        .partition_point(|piece| piece.tangent_maximum <= tangent)
        .min(pieces.len().saturating_sub(1));
    pieces
        .get(index)
        .map(|piece| (piece.velocity, index))
        .ok_or_else(|| ValidationError("face-consistent velocity side is empty".into()))
}

fn inside_solid(graph: &Graph, fields: &Fields, point: [f32; 2]) -> bool {
    let voxel = [point[0].floor() as i32, point[1].floor() as i32];
    if solid_voxel_at(graph, voxel[0], voxel[1], 0) {
        // A point exactly on the lower face of a solid voxel belongs to the
        // fluid/solid surface when the negative-side voxel is represented
        // fluid. Treating the integer coordinate as solid freezes tangential
        // characteristics at upper world walls and internal solid faces,
        // while arbitrarily nearby fluid points continue to slide. A segment
        // that actually penetrates is still clipped at its first non-surface
        // probe inside the solid voxel.
        for axis in 0..2 {
            if point[axis].fract() != 0.0 {
                continue;
            }
            let mut fluid_voxel = voxel;
            fluid_voxel[axis] -= 1;
            if solid_voxel_at(graph, fluid_voxel[0], fluid_voxel[1], 0) {
                continue;
            }
            let center = [
                fluid_voxel[0] as f32 + 0.5,
                fluid_voxel[1] as f32 + 0.5,
                0.0,
            ];
            if owner_at(graph, center)
                .is_some_and(|cell| fields.capacity[cell] > 1.0e-8)
            {
                return false;
            }
        }
        return true;
    }
    owner_at(graph, [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0])
        .map(|i| fields.capacity[i] <= 1e-8)
        .unwrap_or(false)
}
pub(crate) fn clip_segment(
    graph: &Graph,
    fields: &Fields,
    start: [f32; 2],
    candidate: [f32; 2],
) -> [f32; 2] {
    let (mut low, mut high, mut found) = (0.0, 1.0, false);
    for probe in 1..=8 {
        let t = div(probe as f32, 8.0);
        let q = [
            add(start[0], mul(t, candidate[0] - start[0])),
            add(start[1], mul(t, candidate[1] - start[1])),
        ];
        if inside_solid(graph, fields, q) {
            low = (probe - 1) as f32 / 8.0;
            high = t;
            found = true;
            break;
        }
    }
    if !found {
        return candidate;
    }
    for _ in 0..8 {
        let mid = mul(0.5, add(low, high));
        let q = [
            add(start[0], mul(mid, candidate[0] - start[0])),
            add(start[1], mul(mid, candidate[1] - start[1])),
        ];
        if inside_solid(graph, fields, q) {
            high = mid
        } else {
            low = mid
        }
    }
    let t = (low - 1e-4).max(0.0);
    [
        add(start[0], mul(t, candidate[0] - start[0])),
        add(start[1], mul(t, candidate[1] - start[1])),
    ]
}
fn trace_characteristic(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    span: f32,
    dt: f32,
    direction: f32,
    renormalize_sparse_support: bool,
) -> [f32; 2] {
    let support = |point: [f32; 2]| {
        sample_support_with_weight(
            graph,
            fields,
            point[0],
            point[1],
            span,
            renormalize_sparse_support,
        )
        .0
    };
    let initial = support(position);
    let length = add(
        mul(div(initial[0], span), div(initial[0], span)),
        mul(div(initial[1], span), div(initial[1], span)),
    )
    .sqrt();
    let substeps = mul(length, dt).ceil().clamp(1.0, 16.0) as usize;
    let sub_dt = div(dt, substeps as f32);
    let mut traced = position;
    for step in 0..substeps {
        let first = if step == 0 {
            initial
        } else {
            support(traced)
        };
        let raw_mid = [
            (traced[0] + mul(direction, mul(mul(0.5, sub_dt), first[0])))
                .clamp(0.5 * span, graph.dimensions[0] - 0.5 * span),
            (traced[1] + mul(direction, mul(mul(0.5, sub_dt), first[1])))
                .clamp(0.5 * span, graph.dimensions[1] - 0.5 * span),
        ];
        let midpoint = clip_segment(graph, fields, traced, raw_mid);
        let middle = support(midpoint);
        let raw = [
            (traced[0] + mul(direction, mul(sub_dt, middle[0])))
                .clamp(0.5 * span, graph.dimensions[0] - 0.5 * span),
            (traced[1] + mul(direction, mul(sub_dt, middle[1])))
                .clamp(0.5 * span, graph.dimensions[1] - 0.5 * span),
        ];
        traced = clip_segment(graph, fields, traced, raw)
    }
    traced
}

fn row_touches_liquid(row: &Row, fields: &Fields) -> bool {
    row.terms
        .iter()
        .any(|t| fields.density[t.cell_id as usize] > LIQUID_ISOVALUE)
}
fn row_source_fluid_velocity(row: &Row, fields: &Fields) -> f32 {
    let velocity = fields.face_velocity[row.id as usize];
    if row.kind == RowKind::ClosedWorld {
        return velocity;
    }
    if row.open_fraction > 1e-6 {
        div(
            velocity - mul(1.0 - row.open_fraction, row.solid_velocity),
            row.open_fraction,
        )
    } else {
        row.solid_velocity
    }
}
fn source_staggered_cell_sample(
    graph: &Graph,
    fields: &Fields,
    point: [f32; 2],
    axis: usize,
) -> (f32, bool) {
    let mut query = [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0];
    let mut cell = owner_at(graph, query);
    if cell.is_none() {
        query[axis] -= 1.0;
        cell = owner_at(graph, query)
    }
    let Some(cell) = cell else {
        return (0.0, false);
    };
    if fields.capacity[cell] <= 1e-8 {
        return (0.0, false);
    }
    let own_cell = &graph.cells[cell];
    let tangent = 1 - axis;
    let mut values = [0.0; 2];
    let mut weights = [0.0; 2];
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis
            || !row_touches_liquid(row, fields)
            || fields.solid_motion_active && row.open_fraction < 1.0
        {
            continue;
        }
        let Some(own) = own_term(row, cell) else {
            continue;
        };
        let side = usize::from(own.coefficient < 0.0);
        let mut patch = 0.0;
        if row.terms.len() == 1 {
            let length = mul(
                own.coefficient.abs(),
                row.static_dual_weight.unwrap_or(row.dual_weight),
            );
            let expected = if side == 1 {
                own_cell.maximum[axis]
            } else {
                own_cell.minimum[axis]
            };
            if length != own_cell.widths[tangent]
                || row.center[tangent] != own_cell.center[tangent]
                || row.center[axis] != expected
            {
                return (0.0, false);
            }
            if point[tangent] >= own_cell.minimum[tangent]
                && point[tangent] < own_cell.maximum[tangent]
            {
                patch = length
            }
        } else {
            for term in &row.terms {
                if own.coefficient * term.coefficient >= 0.0 {
                    continue;
                }
                let other = &graph.cells[term.cell_id as usize];
                let lower = own_cell.minimum[tangent].max(other.minimum[tangent]);
                let upper = own_cell.maximum[tangent].min(other.maximum[tangent]);
                if point[tangent] >= lower && point[tangent] < upper {
                    patch = add(patch, upper - lower)
                }
            }
        }
        if patch > 0.0 {
            values[side] = add(
                values[side],
                mul(patch, row_source_fluid_velocity(row, fields)),
            );
            weights[side] = add(weights[side], patch)
        }
    }
    let fraction = ((point[axis] - own_cell.minimum[axis]) / own_cell.widths[axis]).clamp(0.0, 1.0);
    if fraction == 0.0 && weights[0] > 0.0 {
        return (div(values[0], weights[0]), true);
    }
    if fraction == 1.0 && weights[1] > 0.0 {
        return (div(values[1], weights[1]), true);
    }
    if weights[0] <= 0.0 || weights[1] <= 0.0 {
        return (0.0, false);
    }
    let low = div(values[0], weights[0]);
    let high = div(values[1], weights[1]);
    (add(mul(1.0 - fraction, low), mul(fraction, high)), true)
}
fn staggered_coordinates(
    graph: &Graph,
    position: [f32; 2],
    axis: usize,
    span: f32,
) -> ([f32; 2], [f32; 2], [f32; 2]) {
    let mut offset = [0.5, 0.5];
    offset[axis] = 0.0;
    let bounded = [
        position[0].clamp(offset[0] * span, graph.dimensions[0] - offset[0] * span),
        position[1].clamp(offset[1] * span, graph.dimensions[1] - offset[1] * span),
    ];
    let shifted = [bounded[0] / span - offset[0], bounded[1] / span - offset[1]];
    let lower = [shifted[0].floor(), shifted[1].floor()];
    (
        offset,
        lower,
        [shifted[0] - lower[0], shifted[1] - lower[1]],
    )
}
fn sample_source_linear(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    axis: usize,
    span: f32,
    renormalize_sparse_support: bool,
) -> f32 {
    let (offset, lower, fraction) = staggered_coordinates(graph, position, axis, span);
    let mut velocity = 0.0;
    let mut represented_weight = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let weight = mul(
                if dx == 1 {
                    fraction[0]
                } else {
                    1.0 - fraction[0]
                },
                if dy == 1 {
                    fraction[1]
                } else {
                    1.0 - fraction[1]
                },
            );
            if weight == 0.0 {
                continue;
            }
            let point = [
                span * (lower[0] + dx as f32 + offset[0]),
                span * (lower[1] + dy as f32 + offset[1]),
            ];
            let (value, valid) = source_staggered_cell_sample(graph, fields, point, axis);
            if !renormalize_sparse_support {
                let fallback = sample_support(graph, fields, point[0], point[1], span)[axis];
                velocity = add(velocity, mul(weight, if valid { value } else { fallback }));
                continue;
            }
            if valid {
                velocity = add(velocity, mul(weight, value));
                represented_weight = add(represented_weight, weight);
            } else {
                let (fallback, support_weight) =
                    sample_support_with_weight(graph, fields, point[0], point[1], span, true);
                if support_weight > 0.0 {
                    velocity = add(velocity, mul(weight, fallback[axis]));
                    represented_weight = add(represented_weight, weight);
                }
            }
        }
    }
    if !renormalize_sparse_support {
        velocity
    } else if represented_weight > 0.0 {
        div(velocity, represented_weight)
    } else {
        0.0
    }
}
fn uniform_staggered_node(
    graph: &Graph,
    fields: &Fields,
    point: [f32; 2],
    axis: usize,
    span: f32,
) -> (f32, bool) {
    let Some(cell) = owner_at(graph, [point[0].floor() + 0.5, point[1].floor() + 0.5, 0.0]) else {
        return (0.0, false);
    };
    if graph.cells[cell].widths[..2].iter().any(|&w| w != span) || fields.capacity[cell] != 1.0 {
        return (0.0, false);
    }
    let mut found = false;
    let mut value = 0.0;
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        if row.axis as usize != axis || row.center[0] != point[0] || row.center[1] != point[1] {
            continue;
        }
        if row.terms.len() != 2
            || row.open_fraction != 1.0
            || row.static_measure.unwrap_or(row.measure) != span
            || row.distance != span
            || !row_touches_liquid(row, fields)
        {
            return (0.0, false);
        }
        let (mut negative, mut positive) = (false, false);
        for term in &row.terms {
            let endpoint = &graph.cells[term.cell_id as usize];
            if endpoint.widths[..2].iter().any(|&w| w != span)
                || fields.capacity[term.cell_id as usize] != 1.0
            {
                return (0.0, false);
            }
            let mut expected = point;
            if term.coefficient < 0.0 {
                expected[axis] -= 0.5 * span;
                negative = true
            } else if term.coefficient > 0.0 {
                expected[axis] += 0.5 * span;
                positive = true
            } else {
                return (0.0, false);
            }
            if endpoint.center[0] != expected[0] || endpoint.center[1] != expected[1] {
                return (0.0, false);
            }
        }
        if !negative || !positive || found {
            return (0.0, false);
        }
        found = true;
        value = fields.face_velocity[row.id as usize]
    }
    (value, found)
}
fn cubic_line(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    if t == 0.0 {
        return b;
    }
    if t == 1.0 {
        return c;
    }
    let inner = add(add(a - b, mul(2.0, c - b)), mul(-0.5, d - b));
    let cubic = add(mul(1.5, b - c), mul(0.5, d - a));
    add(
        b,
        mul(t, add(mul(0.5, c - a), mul(t, add(inner, mul(t, cubic))))),
    )
}
fn sample_source(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 2],
    axis: usize,
    span: f32,
    renormalize_sparse_support: bool,
) -> f32 {
    let (offset, lower, fraction) = staggered_coordinates(graph, position, axis, span);
    let interpolated = [fraction[0] != 0.0, fraction[1] != 0.0];
    let first = [
        span * (lower[0] - 1.0 + offset[0]),
        span * (lower[1] - 1.0 + offset[1]),
    ];
    let last = [
        span * (lower[0] + 2.0 + offset[0]),
        span * (lower[1] + 2.0 + offset[1]),
    ];
    let lower_bound = [offset[0] * span, offset[1] * span];
    let upper_bound = [
        graph.dimensions[0] - lower_bound[0],
        graph.dimensions[1] - lower_bound[1],
    ];
    if (!interpolated[0] && !interpolated[1])
        || (0..2)
            .any(|a| interpolated[a] && (first[a] < lower_bound[a] || last[a] > upper_bound[a]))
        || interpolated[axis] && (first[axis] <= 0.0 || last[axis] >= graph.dimensions[axis])
    {
        return sample_source_linear(
            graph,
            fields,
            position,
            axis,
            span,
            renormalize_sparse_support,
        );
    }
    let mut values = [[0.0; 4]; 4];
    let (mut core_min, mut core_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let x_count = if interpolated[0] { 4 } else { 1 };
    let y_count = if interpolated[1] { 4 } else { 1 };
    for yi in 0..y_count {
        let y = if interpolated[1] { yi } else { 1 };
        for xi in 0..x_count {
            let x = if interpolated[0] { xi } else { 1 };
            let point = [
                span * (lower[0] + x as f32 - 1.0 + offset[0]),
                span * (lower[1] + y as f32 - 1.0 + offset[1]),
            ];
            let (node, valid) = uniform_staggered_node(graph, fields, point, axis, span);
            if !valid {
                return sample_source_linear(
                    graph,
                    fields,
                    position,
                    axis,
                    span,
                    renormalize_sparse_support,
                );
            }
            values[y][x] = node;
            if (x == 1 || interpolated[0] && x == 2) && (y == 1 || interpolated[1] && y == 2) {
                core_min = core_min.min(node);
                core_max = core_max.max(node)
            }
        }
    }
    let mut lines = [0.0; 4];
    for yi in 0..y_count {
        let y = if interpolated[1] { yi } else { 1 };
        lines[y] = if interpolated[0] {
            cubic_line(
                values[y][0],
                values[y][1],
                values[y][2],
                values[y][3],
                fraction[0],
            )
        } else {
            values[y][1]
        }
    }
    let cubic = if interpolated[1] {
        cubic_line(lines[0], lines[1], lines[2], lines[3], fraction[1])
    } else {
        lines[1]
    };
    cubic.clamp(core_min, core_max)
}

/// Exact 2-D accepted-face semi-Lagrangian preparation.
pub fn prepare_faces(graph: &Graph, fields: &mut Fields, dt: f32) -> Result<(), ValidationError> {
    prepare_faces_impl(graph, fields, dt, false)
}

/// Face preparation for geometric cellwise transport. Sparse interpolation
/// renormalizes the represented stencil so a uniform translating liquid does
/// not blend toward zero beside an unallocated air page.
pub fn prepare_faces_for_cellwise_remap(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
) -> Result<(), ValidationError> {
    prepare_faces_impl(graph, fields, dt, true)
}

fn prepare_faces_impl(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
    renormalize_sparse_support: bool,
) -> Result<(), ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::prepare_faces_3d(graph, fields, dt);
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "3-D face preparation requires 3-D staggered sampling".into(),
        ));
    }
    for row in &graph.rows {
        let i = row.id as usize;
        if physical_row_measure(row) <= 1e-8 {
            fields.face_velocity[i] = row.solid_velocity;
            continue;
        }
        let (mut touches_extended, mut sampling_width) = (false, f32::INFINITY);
        for term in &row.terms {
            let cell = term.cell_id as usize;
            touches_extended |= fields.extension_depth[cell] != 255;
            sampling_width = sampling_width.min(
                graph.cells[cell].widths[0]
                    .min(graph.cells[cell].widths[1])
                    .max(1.0),
            )
        }
        if !touches_extended {
            fields.face_velocity[i] = row.solid_velocity;
            continue;
        }
        let span = sampling_width.max(1.0);
        let mut region_width = span;
        for term in &row.terms {
            let cell = &graph.cells[term.cell_id as usize];
            if cell.refinement_region_scale.unwrap_or(1.0) > 1.0 {
                region_width = region_width.max(cell.widths[0].min(cell.widths[1]))
            }
        }
        let departure = trace_characteristic(
            graph,
            fields,
            [row.center[0], row.center[1]],
            region_width,
            dt,
            -1.0,
            renormalize_sparse_support,
        );
        let characteristic = sample_source(
            graph,
            fields,
            departure,
            row.axis as usize,
            span,
            renormalize_sparse_support,
        );
        fields.face_velocity[i] = add(
            mul(row.open_fraction, characteristic),
            mul(1.0 - row.open_fraction, row.solid_velocity),
        )
    }
    Ok(())
}

pub fn trace_effective_transport_arrival(
    graph: &Graph,
    fields: &Fields,
    position: [f32; 3],
    dt: f32,
) -> [f32; 3] {
    if graph.dimension == 3 {
        return crate::numerics3d::trace_characteristic_3d(graph, fields, position, dt, 1.0);
    }
    let span = owner_at(
        graph,
        [position[0].floor() + 0.5, position[1].floor() + 0.5, 0.0],
    )
    .map(|i| graph.cells[i].widths[0])
    .unwrap_or(1.0);
    let xy = trace_characteristic(
        graph,
        fields,
        [position[0], position[1]],
        span,
        dt,
        1.0,
        false,
    );
    [xy[0], xy[1], position[2]]
}

pub fn publish_transport_characteristic_clearance<'a>(
    graph: &Graph,
    fields: &'a mut Fields,
    dt: f32,
    source_density: Option<&[f32]>,
    source_gamma: Option<&[f32]>,
) -> Result<&'a [f32], ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::publish_characteristic_clearance_3d(
            graph,
            fields,
            dt,
            source_density,
            source_gamma,
        );
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "characteristic clearance is a 2-D stage".into(),
        ));
    }
    let density = source_density.unwrap_or(&fields.density);
    let gamma = source_gamma.unwrap_or(&fields.gamma);
    let mut result = vec![0.0; graph.cells.len()];
    for receiver in &graph.cells {
        let id = receiver.id as usize;
        if fields.capacity[id] <= 1e-8 {
            continue;
        }
        let span = receiver.widths[0];
        let departure = trace_characteristic(
            graph,
            fields,
            [receiver.center[0], receiver.center[1]],
            span,
            dt,
            -1.0,
            false,
        );
        let bx = departure[0].clamp(0.5 * span, graph.dimensions[0] - 0.5 * span);
        let by = departure[1].clamp(0.5 * span, graph.dimensions[1] - 0.5 * span);
        let sx = bx / span - 0.5;
        let sy = by / span - 0.5;
        let lx = sx.floor();
        let ly = sy.floor();
        let tx = sx - lx;
        let ty = sy - ly;
        let (mut visible, mut minimum_width, mut valid) = (0.0, f32::INFINITY, true);
        let (mut minx, mut miny, mut maxx, mut maxy) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for dy in 0..2 {
            for dx in 0..2 {
                let weight = mul(
                    if dx == 1 { tx } else { 1.0 - tx },
                    if dy == 1 { ty } else { 1.0 - ty },
                );
                visible = add(visible, weight);
                let Some(cell) = owner_at(
                    graph,
                    [
                        span * (lx + dx as f32 + 0.5),
                        span * (ly + dy as f32 + 0.5),
                        0.0,
                    ],
                ) else {
                    valid = false;
                    continue;
                };
                if fields.capacity[cell] <= 1e-8
                    || fields.capacity[cell] < 0.999999
                    || (density[cell] - 1.0).abs() > 0.005
                    || (gamma[cell] - 1.0).abs() > 0.005
                {
                    valid = false;
                    continue;
                }
                minimum_width = minimum_width
                    .min(graph.cells[cell].widths[0])
                    .min(graph.cells[cell].widths[1]);
                let vx = fields.cell_velocity[2 * cell];
                let vy = fields.cell_velocity[2 * cell + 1];
                minx = minx.min(vx);
                miny = miny.min(vy);
                maxx = maxx.max(vx);
                maxy = maxy.max(vy)
            }
        }
        if !valid || visible < 0.999999 {
            continue;
        }
        let dx = maxx - minx;
        let dy = maxy - miny;
        let maximum_delta = add(mul(dx, dx), mul(dy, dy)).sqrt();
        let clearance = mul(0.02, minimum_width);
        if mul(dt, maximum_delta) <= mul(0.002, minimum_width) {
            result[id] = clearance
        }
    }
    fields.characteristic_clearance = result;
    Ok(&fields.characteristic_clearance)
}

fn pressure_density(graph: &Graph, fields: &Fields, cell: usize) -> f32 {
    let mut density = fill(fields, cell);
    let final_capacity = mul(fields.capacity[cell], graph.cells[cell].measure);
    let before = mul(
        Fields::optional_cell(&fields.capacity_before, cell, fields.capacity[cell]),
        graph.cells[cell].measure,
    );
    let source = Fields::optional_cell(&fields.source_rate, cell, 0.0);
    if fields.solid_motion_active && final_capacity < before && fields.density[cell] > 0.0 {
        let rate = if fields.frame_dt > 0.0 {
            div(final_capacity - before, fields.frame_dt)
        } else {
            0.0
        };
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(-rate, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
        ));
    }
    if source > 0.0 {
        density = density.max(add(
            LIQUID_ISOVALUE,
            div(mul(source, fields.frame_dt), final_capacity.max(1e-8)).min(0.5),
        ));
    }
    density
}

fn volume_roundoff(capacity: f32) -> f32 {
    mul(VOLUME_ROUNDOFF_RATIO, capacity)
}

fn moving_pressure_predicted_fill(graph: &Graph, fields: &Fields, cell: usize) -> bool {
    if !fields.solid_motion_active {
        return false;
    }
    let mut equation = 0.0;
    let mut correction = 0.0;
    for &row_id in &graph.incidences[cell] {
        let row = &graph.rows[row_id as usize];
        let Some(own) = own_term(row, cell) else {
            continue;
        };
        let velocity = fields.face_velocity[row.id as usize]
            - mul(1.0 - row.open_fraction, row.solid_velocity);
        let value = mul(
            own.coefficient,
            mul(row.static_dual_weight.unwrap_or(row.dual_weight), velocity),
        );
        if value > 0.0 {
            let supported = row
                .terms
                .iter()
                .filter(|t| t.cell_id as usize != cell && t.coefficient * own.coefficient < 0.0)
                .any(|t| {
                    let j = t.cell_id as usize;
                    mul(fields.density[j], graph.cells[j].measure)
                        > volume_roundoff(mul(fields.capacity[j], graph.cells[j].measure))
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
    let predicted = add(
        mul(fields.density[cell], graph.cells[cell].measure),
        mul(fields.frame_dt, equation),
    );
    let cap = mul(fields.capacity[cell], graph.cells[cell].measure);
    predicted >= cap - volume_roundoff(cap)
}

pub fn prepare_pressure_membership(graph: &Graph, fields: &mut Fields) {
    let prior = fields.pressure_member.clone();
    let count = graph.cells.len();
    let current_liquid: Vec<bool> = (0..count)
        .map(|cell| {
            mul(fields.capacity[cell], graph.cells[cell].measure) > 1e-8
                && (pressure_density(graph, fields, cell) >= LIQUID_ISOVALUE
                    || moving_pressure_predicted_fill(graph, fields, cell)
                    || Fields::optional_cell(&fields.source_rate, cell, 0.0) > 0.0)
        })
        .collect();
    let current_air: Vec<bool> = (0..count)
        .map(|cell| {
            !current_liquid[cell] && mul(fields.capacity[cell], graph.cells[cell].measure) > 1e-8
        })
        .collect();
    let mut visited = vec![false; count];
    let mut enclosed_air = vec![false; count];
    for seed in 0..count {
        if !current_air[seed] || visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = VecDeque::from([seed]);
        let mut component = Vec::new();
        let mut touches_exterior = false;
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                let open = physical_row_measure(row) > 1e-8;
                // Sparse-air support is an atmospheric connection. Closed
                // tank walls are not: sealed air stays P0 so the coupled
                // closed-air pressure authority can represent it as one
                // thermodynamic component.
                if open && row.kind == RowKind::SparseAir {
                    touches_exterior = true;
                }
                if !open {
                    continue;
                }
                for term in &row.terms {
                    let other = term.cell_id as usize;
                    if other != cell && current_air[other] && !visited[other] {
                        visited[other] = true;
                        queue.push_back(other);
                    }
                }
            }
        }
        if !touches_exterior {
            for cell in component {
                enclosed_air[cell] = true;
            }
        }
    }
    for cell in 0..count {
        let mut retained_submerged = prior[cell] != 0 && enclosed_air[cell];
        let mut neighbors = 0;
        if retained_submerged {
            'rows: for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                if physical_row_measure(row) <= 1e-8 {
                    continue;
                }
                for term in &row.terms {
                    if term.cell_id as usize != cell {
                        neighbors += 1;
                        if prior[term.cell_id as usize] == 0 {
                            retained_submerged = false;
                            break 'rows;
                        }
                    }
                }
            }
            retained_submerged &= neighbors > 0;
        }
        fields.pressure_member[cell] = u8::from(current_liquid[cell] || retained_submerged);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureRows {
    pub active: Vec<u8>,
    pub theta: Vec<f32>,
}

fn promote_swept_static_wall_pressure_support(graph: &Graph, fields: &mut Fields, dt: f32) {
    if graph.dimension != 2 || !(dt > 0.0) {
        return;
    }
    let mut minimum_velocity = vec![[0.0_f64; 2]; graph.cells.len()];
    let mut maximum_velocity = vec![[0.0_f64; 2]; graph.cells.len()];
    for face in &graph.subfaces {
        let axis = face.axis as usize;
        let velocity = crate::transport::physical_subface_rate(graph, fields, face.id as usize)
            as f64
            / face.measure.max(1.0e-8) as f64;
        for cell in [face.negative_cell, face.positive_cell] {
            if cell < 0 {
                continue;
            }
            let cell = cell as usize;
            minimum_velocity[cell][axis] = minimum_velocity[cell][axis].min(velocity);
            maximum_velocity[cell][axis] = maximum_velocity[cell][axis].max(velocity);
        }
    }
    let mut promoted = vec![false; graph.cells.len()];
    let liquid: Vec<bool> = graph
        .cells
        .iter()
        .enumerate()
        .map(|(cell, geometry)| {
            let capacity = mul(fields.capacity[cell], geometry.measure);
            capacity > 1.0e-8
                && mul(fields.density[cell], geometry.measure) > volume_roundoff(capacity)
        })
        .collect();
    let mut visited = vec![false; graph.cells.len()];
    for seed in 0..graph.cells.len() {
        if !liquid[seed] || visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = VecDeque::from([seed]);
        let mut component = Vec::new();
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            for &row_id in &graph.incidences[cell] {
                let row = &graph.rows[row_id as usize];
                if physical_row_measure(row) <= 1.0e-8 {
                    continue;
                }
                for term in &row.terms {
                    let other = term.cell_id as usize;
                    if other != cell && liquid[other] && !visited[other] {
                        visited[other] = true;
                        queue.push_back(other);
                    }
                }
            }
        }
        let sweeps: Vec<([f64; 2], [f64; 2])> = component
            .iter()
            .map(|&source| {
                let geometry = &graph.cells[source];
                (
                    [
                        geometry.minimum[0] as f64
                            + dt as f64 * minimum_velocity[source][0],
                        geometry.minimum[1] as f64
                            + dt as f64 * minimum_velocity[source][1],
                    ],
                    [
                        geometry.maximum[0] as f64
                            + dt as f64 * maximum_velocity[source][0],
                        geometry.maximum[1] as f64
                            + dt as f64 * maximum_velocity[source][1],
                    ],
                )
            })
            .collect();
        let hits_wall = sweeps.iter().any(|&(minimum, maximum)| {
            graph.subfaces.iter().any(|face| {
                    if graph.rows[face.row_id as usize].kind != RowKind::ClosedWorld {
                        return false;
                    }
                    let row = &graph.rows[face.row_id as usize];
                    if row.solid_velocity.abs() > 1.0e-8
                        || (row.open_fraction_after.unwrap_or(row.open_fraction)
                            - row.open_fraction_before.unwrap_or(row.open_fraction))
                            .abs()
                            > 1.0e-8
                    {
                        return false;
                    }
                    let normal = face.axis as usize;
                    let tangent = 1 - normal;
                    let normal_at = face.center[normal] as f64;
                    let tangent_min =
                        face.center[tangent] as f64 - 0.5 * face.measure as f64;
                    let tangent_max =
                        face.center[tangent] as f64 + 0.5 * face.measure as f64;
                    normal_at >= minimum[normal]
                        && normal_at <= maximum[normal]
                        && tangent_max > minimum[tangent]
                        && tangent_min < maximum[tangent]
                })
        });
        if !hits_wall {
            continue;
        }
        for (target, candidate) in graph.cells.iter().enumerate().filter(|(target, candidate)| {
            mul(fields.capacity[*target], candidate.measure) > 1.0e-8
        }) {
            if sweeps.iter().any(|&(minimum, maximum)| {
                candidate.maximum[0] as f64 > minimum[0]
                    && (candidate.minimum[0] as f64) < maximum[0]
                    && candidate.maximum[1] as f64 > minimum[1]
                    && (candidate.minimum[1] as f64) < maximum[1]
            }) {
                promoted[target] = true;
            }
        }
    }
    for (member, promote) in fields.pressure_member.iter_mut().zip(promoted) {
        *member |= u8::from(promote);
    }
}

fn ghost_theta(liquid_phi: f32, air_phi: f32) -> f32 {
    ((liquid_phi.abs() as f64 / ((liquid_phi.abs() as f64 + air_phi.abs() as f64).max(1e-12)))
        .clamp(GHOST_FLUID_THETA_MIN as f64, 1.0)) as f32
}

/// Builds PCM/PCF row membership and the Jacobi diagonal in canonical row order.
pub fn prepare_pressure_topology(graph: &Graph, fields: &mut Fields) -> PressureRows {
    prepare_pressure_topology_impl(graph, fields, false, true)
}

/// Reconstruct the row coefficients used by the completed pressure projection
/// without advancing the history-sensitive pressure membership a second time.
pub(crate) fn pressure_projection_velocity_roundoff_scale(
    graph: &Graph,
    fields: &Fields,
) -> Vec<f64> {
    let mut snapshot = fields.clone();
    let rows = prepare_pressure_topology_impl(graph, &mut snapshot, false, false);
    graph
        .rows
        .iter()
        .map(|row| {
            let at = row.id as usize;
            if rows.active[at] == 0 || rows.theta[at] <= 0.0 {
                return fields.face_velocity[at].abs() as f64;
            }
            let jump = row_gradient(row, &fields.pressure_member, &fields.pressure);
            let open = if row.kind == RowKind::ClosedWorld {
                if row.separating { 1.0 } else { 0.0 }
            } else {
                row.open_fraction
            };
            let pressure_change = div(mul(open, jump), rows.theta[at]);
            let solid_term = mul(1.0 - row.open_fraction, row.solid_velocity);
            // project_pressure_velocity forms post = pre - pressure_change in
            // f32. Since |pre| <= |post| + |pressure_change|, this bounds both
            // cancellation operands and the later solid-velocity subtraction.
            fields.face_velocity[at].abs() as f64
                + 2.0 * pressure_change.abs() as f64
                + solid_term.abs() as f64
        })
        .collect()
}

pub fn prepare_pressure_topology_with_swept_static_wall_support(
    graph: &Graph,
    fields: &mut Fields,
) -> PressureRows {
    // The caller first prepares and snapshots the physical topology. Avoid a
    // second membership refresh here: retained submerged membership has
    // history and is intentionally not an idempotent operation.
    prepare_pressure_topology_impl(graph, fields, true, false)
}

fn prepare_pressure_topology_impl(
    graph: &Graph,
    fields: &mut Fields,
    swept_static_wall_support: bool,
    refresh_membership: bool,
) -> PressureRows {
    if refresh_membership {
        prepare_pressure_membership(graph, fields);
    }
    if swept_static_wall_support {
        promote_swept_static_wall_pressure_support(graph, fields, fields.frame_dt);
    }
    let mut state = PressureRows {
        active: vec![0; graph.rows.len()],
        theta: vec![0.0; graph.rows.len()],
    };
    for row in &graph.rows {
        let d = graph.dimension as usize;
        let mut geometric = [0.0f32; 3];
        let mut geometric_offset = 0.0;
        let mut geometric_weight = 0.0;
        if row.kind != RowKind::ClosedWorld {
            for term in &row.terms {
                let i = term.cell_id as usize;
                let base = d * i;
                let length = (0..d)
                    .map(|a| {
                        mul(
                            fields.interface_normal[base + a],
                            fields.interface_normal[base + a],
                        )
                    })
                    .fold(0.0, add);
                if length <= 0.5 {
                    continue;
                }
                let weight = term.coefficient.abs();
                for a in 0..d {
                    geometric[a] = add(geometric[a], mul(weight, fields.interface_normal[base + a]))
                }
                let shifted = if d == 2 {
                    add(
                        fields.interface_offset[i],
                        add(
                            mul(
                                fields.interface_normal[base],
                                graph.cells[i].center[0] - row.center[0],
                            ),
                            mul(
                                fields.interface_normal[base + 1],
                                graph.cells[i].center[1] - row.center[1],
                            ),
                        ),
                    )
                } else {
                    (0..d)
                        .map(|a| {
                            mul(
                                fields.interface_normal[base + a],
                                graph.cells[i].center[a] - row.center[a],
                            )
                        })
                        .fold(fields.interface_offset[i], add)
                };
                geometric_offset = add(geometric_offset, mul(weight, shifted));
                geometric_weight = add(geometric_weight, weight)
            }
        }
        let geometric_length = (0..d)
            .map(|a| mul(geometric[a], geometric[a]))
            .fold(0.0, add)
            .sqrt();
        let mut geometry_valid =
            geometric_weight > 1e-8 && geometric_length > mul(1e-6, geometric_weight);
        if geometry_valid {
            for term in &row.terms {
                let i = term.cell_id as usize;
                let numerator = if d == 2 {
                    add(
                        mul(geometric[0], graph.cells[i].center[0] - row.center[0]),
                        mul(geometric[1], graph.cells[i].center[1] - row.center[1]),
                    ) - geometric_offset
                } else {
                    (0..d)
                        .map(|a| mul(geometric[a], graph.cells[i].center[a] - row.center[a]))
                        .fold(-geometric_offset, add)
                };
                let phi = div(numerator, geometric_length);
                geometry_valid &= fields.capacity[i] >= 0.999999
                    && if fields.pressure_member[i] != 0 {
                        phi <= 0.0
                    } else {
                        phi >= 0.0
                    };
            }
        }
        let mut liquid_phi = 0.0;
        let mut liquid_weight = 0.0;
        let mut air_phi = 0.0;
        let mut air_weight = 0.0;
        let mut liquid_count = 0;
        let mut air_count = 0;
        let mut full_gradient = 0.0;
        let mut liquid_gradient = 0.0;
        for term in &row.terms {
            let i = term.cell_id as usize;
            let width = if row.kind == RowKind::SparseAir {
                1.0
            } else {
                graph.cells[i].widths[row.axis as usize]
            };
            let old_phi = mul(LIQUID_ISOVALUE - pressure_density(graph, fields, i), width);
            let phi = if geometry_valid {
                let numerator = if d == 2 {
                    add(
                        mul(geometric[0], graph.cells[i].center[0] - row.center[0]),
                        mul(geometric[1], graph.cells[i].center[1] - row.center[1]),
                    ) - geometric_offset
                } else {
                    (0..d)
                        .map(|a| mul(geometric[a], graph.cells[i].center[a] - row.center[a]))
                        .fold(-geometric_offset, add)
                };
                div(numerator, geometric_length)
            } else {
                old_phi
            };
            let weight = term.coefficient.abs();
            full_gradient = add(full_gradient, mul(term.coefficient, phi));
            if fields.pressure_member[i] != 0 {
                liquid_count += 1;
                liquid_phi = add(liquid_phi, mul(weight, phi));
                liquid_weight = add(liquid_weight, weight);
                liquid_gradient = add(liquid_gradient, mul(term.coefficient, phi));
            } else {
                air_count += 1;
                air_phi = add(air_phi, mul(weight, phi));
                air_weight = add(air_weight, weight);
            }
        }
        if liquid_count == 0 || pressure_dual_weight(row) <= 1e-8 {
            continue;
        }
        if row.kind == RowKind::SparseAir {
            air_phi = add(air_phi, mul(liquid_weight, 0.5));
            air_weight = add(air_weight, liquid_weight);
        }
        let cut = air_count > 0 || row.kind == RowKind::SparseAir;
        let mut theta = if cut {
            ghost_theta(
                div(liquid_phi, liquid_weight.max(1e-9)),
                div(air_phi, air_weight.max(1e-9)),
            )
        } else {
            1.0
        };
        let gravity_length = (0..d)
            .map(|a| mul(fields.acceleration_fine[a], fields.acceleration_fine[a]))
            .fold(0.0, add)
            .sqrt();
        let partial_region = graph
            .cells
            .iter()
            .any(|c| c.refinement_region_scale.unwrap_or(1.0) > 1.0)
            && graph
                .cells
                .iter()
                .any(|c| c.refinement_region_scale.unwrap_or(1.0) == 1.0);
        if cut
            && graph.dimension == 2
            && row.axis == 1
            && gravity_length > 1e-6
            && partial_region
            && fields.acceleration_fine[1] < 0.0
            && fields.acceleration_fine[0].abs() <= mul(1e-6, gravity_length)
        {
            let (height, height_valid) = pressure_planar_column_height(graph, fields, row);
            let liquid_y = div(
                row.terms
                    .iter()
                    .filter(|t| fields.pressure_member[t.cell_id as usize] != 0)
                    .map(|t| {
                        mul(
                            t.coefficient.abs(),
                            graph.cells[t.cell_id as usize].center[1],
                        )
                    })
                    .fold(0.0, add),
                liquid_weight.max(1e-9),
            );
            let mut air_y_sum = 0.0;
            let mut hydro_air_weight = 0.0;
            for term in &row.terms {
                if fields.pressure_member[term.cell_id as usize] == 0 {
                    air_y_sum = add(
                        air_y_sum,
                        mul(
                            term.coefficient.abs(),
                            graph.cells[term.cell_id as usize].center[1],
                        ),
                    );
                    hydro_air_weight = add(hydro_air_weight, term.coefficient.abs())
                }
            }
            if row.kind == RowKind::SparseAir {
                let direction = if row.center[1] >= liquid_y { 1.0 } else { -1.0 };
                air_y_sum = add(
                    air_y_sum,
                    mul(liquid_weight, add(liquid_y, mul(direction, row.distance))),
                );
                hydro_air_weight = add(hydro_air_weight, liquid_weight)
            }
            let air_y = div(air_y_sum, hydro_air_weight.max(1e-9));
            if height_valid && air_y > liquid_y + 1e-6 && height > liquid_y && height < air_y {
                theta = div(height - liquid_y, air_y - liquid_y).clamp(GHOST_FLUID_THETA_MIN, 1.0)
            }
        }
        if cut && row.kind == RowKind::MixedSeam {
            let factor = if full_gradient == 0.0 {
                0.0
            } else if liquid_gradient == 0.0 {
                div(1.0, GHOST_FLUID_THETA_MIN)
            } else {
                div(full_gradient, liquid_gradient).clamp(0.0, div(1.0, GHOST_FLUID_THETA_MIN))
            };
            theta = if factor > 0.0 { div(1.0, factor) } else { 0.0 };
        }
        state.theta[row.id as usize] = theta;
        state.active[row.id as usize] = 1;
    }
    fields.pressure_diagonal.fill(0.0);
    let d = graph.dimension as usize;
    for cell in 0..graph.cells.len() {
        if fields.pressure_member[cell] == 0 {
            continue;
        }
        let mut axes = vec![0.0; d];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            if state.active[row.id as usize] == 0 || state.theta[row.id as usize] <= 0.0 {
                continue;
            }
            if let Some(own) = own_term(row, cell) {
                let contribution = div(
                    mul(
                        pressure_dual_weight(row),
                        mul(own.coefficient, own.coefficient),
                    ),
                    state.theta[row.id as usize],
                );
                axes[row.axis as usize] = add(axes[row.axis as usize], contribution);
            }
        }
        fields.pressure_diagonal[cell] = axes.into_iter().fold(0.0, add);
    }
    fields.pressure_row_member = state.active.clone();
    state
}

fn pressure_integrated_column_height(graph: &Graph, fields: &Fields, x: i32) -> (f32, bool) {
    let ny = graph.dimensions[1] as i32;
    let mut y = 0;
    let mut mass = 0.0;
    let mut previous = 1.0;
    let mut column_open = -1.0;
    let (mut saw_open, mut saw_liquid, mut saw_air) = (false, false, false);
    while y < ny {
        if solid_voxel_fraction_at(graph, x, y, 0).unwrap_or(0.0) >= 1.0 {
            return (0.0, false);
        }
        let owner = owner_at(graph, [x as f32 + 0.5, y as f32 + 0.5, 0.0]);
        let (fill_value, width) = if let Some(cell) = owner {
            let open = fields.capacity[cell];
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
            let value = pressure_density(graph, fields, cell).clamp(0.0, 1.0);
            let width = (graph.cells[cell].widths[1] as i32
                - y % (graph.cells[cell].widths[1] as i32))
                .max(1)
                .min(ny - y);
            (value, width)
        } else {
            (0.0, (8 - y % 8).max(1).min(ny - y))
        };
        if fill_value > previous + 0.01 {
            return (0.0, false);
        }
        previous = fill_value;
        saw_liquid |= fill_value > 1e-3;
        saw_air |= fill_value < 1.0 - 1e-3;
        mass = add(mass, mul(fill_value, width as f32));
        y += width
    }
    (mass, saw_open && saw_liquid && saw_air)
}
fn pressure_planar_column_height(graph: &Graph, fields: &Fields, row: &Row) -> (f32, bool) {
    let nx = graph.dimensions[0] as i32;
    let centre = (row.center[0].floor() as i32).clamp(0, nx - 1);
    let mut height = 0.0;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut valid = true;
    for offset in [0, -1, 1] {
        let (value, ok) =
            pressure_integrated_column_height(graph, fields, (centre + offset).clamp(0, nx - 1));
        if offset == 0 {
            height = value
        }
        valid &= ok;
        minimum = minimum.min(value);
        maximum = maximum.max(value)
    }
    valid &= maximum - minimum <= 0.01;
    (height, valid)
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
    let mut negative = 0.0;
    let mut positive = 0.0;
    for term in &row.terms {
        let i = term.cell_id as usize;
        if member[i] != 0 {
            let v = mul(term.coefficient, input[i]);
            if term.coefficient < 0.0 {
                negative = add(negative, v)
            } else {
                positive = add(positive, v)
            }
        }
    }
    add(negative, positive)
}

pub fn apply_pressure_operator(
    graph: &Graph,
    member: &[u8],
    rows: &PressureRows,
    input: &[f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    let d = graph.dimension as usize;
    let compute = |cell: usize| {
        if member[cell] == 0 {
            return 0.0;
        }
        let mut neg = [0.0; 3];
        let mut pos = [0.0; 3];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            let ri = row.id as usize;
            if rows.active[ri] == 0 || rows.theta[ri] <= 0.0 {
                continue;
            }
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            let contribution = div(
                mul(
                    pressure_dual_weight(row),
                    mul(own.coefficient, row_gradient(row, member, input)),
                ),
                rows.theta[ri],
            );
            let a = row.axis as usize;
            if own.coefficient > 0.0 {
                neg[a] = add(neg[a], contribution)
            } else {
                pos[a] = add(pos[a], contribution)
            }
        }
        let mut total = 0.0;
        for a in 0..d {
            total = add(total, add(neg[a].min(pos[a]), neg[a].max(pos[a])))
        }
        total
    };
    #[cfg(feature = "parallel")]
    if output.len() >= crate::kernels::PARALLEL_POINTWISE_THRESHOLD {
        const CHUNK: usize = 1024;
        output
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let base = chunk_index * CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = compute(base + offset)
                }
            });
        return;
    }
    for (cell, value) in output.iter_mut().enumerate() {
        *value = compute(cell)
    }
}

pub fn assemble_pressure_rhs(graph: &Graph, fields: &mut Fields, rows: &PressureRows) {
    fields.pressure_rhs.fill(0.0);
    let d = graph.dimension as usize;
    for cell in 0..graph.cells.len() {
        if fields.pressure_member[cell] == 0 {
            continue;
        }
        let mut neg = [0.0; 3];
        let mut pos = [0.0; 3];
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            if rows.active[row.id as usize] == 0 {
                continue;
            }
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            let fluid = fields.face_velocity[row.id as usize]
                - mul(1.0 - row.open_fraction, row.solid_velocity);
            let value = mul(
                own.coefficient,
                mul(row.static_dual_weight.unwrap_or(row.dual_weight), fluid),
            );
            let a = row.axis as usize;
            if own.coefficient > 0.0 {
                neg[a] = add(neg[a], value)
            } else {
                pos[a] = add(pos[a], value)
            }
        }
        let mut rhs = 0.0;
        for a in 0..d {
            rhs = add(rhs, add(neg[a].min(pos[a]), neg[a].max(pos[a])))
        }
        rhs = add(
            rhs,
            -mul(
                Fields::optional_cell(&fields.capacity_rate, cell, 0.0),
                graph.cells[cell].measure,
            ),
        );
        rhs = add(rhs, Fields::optional_cell(&fields.source_rate, cell, 0.0));
        fields.pressure_rhs[cell] = rhs;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureReceipt {
    pub iterations: u32,
    pub initial_residual: f32,
    pub residual: f32,
    pub converged: bool,
}

pub fn solve_pressure(
    graph: &Graph,
    fields: &mut Fields,
    rows: &PressureRows,
    maximum_iterations: u32,
    relative_tolerance: f32,
    execution_order: Option<&[u32]>,
) -> Result<PressureReceipt, PressureError> {
    let result = solve_pressure_pcg(
        &fields.pressure_diagonal,
        &fields.pressure_rhs,
        &fields.pressure,
        &fields.pressure_member,
        execution_order,
        maximum_iterations,
        relative_tolerance,
        |input, output| {
            apply_pressure_operator(graph, &fields.pressure_member, rows, input, output)
        },
    )?;
    fields.pressure = result.pressure;
    Ok(PressureReceipt {
        iterations: result.iterations,
        initial_residual: result.initial_true_residual_squared.max(0.0).sqrt(),
        residual: result.final_true_residual_squared.max(0.0).sqrt(),
        converged: result.converged,
    })
}

pub fn collocate_velocity(graph: &Graph, fields: &mut Fields) {
    let d = graph.dimension as usize;
    fields.cell_velocity.fill(0.0);
    let mut weights = vec![0.0; fields.cell_velocity.len()];
    for row in &graph.rows {
        for term in &row.terms {
            let at = d * term.cell_id as usize + row.axis as usize;
            let weight = mul(
                term.coefficient.abs(),
                row.static_dual_weight.unwrap_or(row.dual_weight),
            );
            let fluid = if row.open_fraction > 1e-6 {
                div(
                    fields.face_velocity[row.id as usize]
                        - mul(1.0 - row.open_fraction, row.solid_velocity),
                    row.open_fraction,
                )
            } else {
                row.solid_velocity
            };
            fields.cell_velocity[at] = add(fields.cell_velocity[at], mul(weight, fluid));
            weights[at] = add(weights[at], weight)
        }
    }
    for (i, v) in fields.cell_velocity.iter_mut().enumerate() {
        if weights[i] > 0.0 {
            *v = div(*v, weights[i])
        }
    }
}

pub fn project_pressure_velocity(graph: &Graph, fields: &mut Fields, rows: &PressureRows) {
    for row in &graph.rows {
        let i = row.id as usize;
        if rows.active[i] == 0 || rows.theta[i] <= 0.0 {
            continue;
        }
        let jump = row_gradient(row, &fields.pressure_member, &fields.pressure);
        let open = if row.kind == RowKind::ClosedWorld {
            if row.separating {
                1.0
            } else {
                0.0
            }
        } else {
            row.open_fraction
        };
        fields.face_velocity[i] = add(
            fields.face_velocity[i],
            -div(mul(open, jump), rows.theta[i]),
        );
    }
    collocate_velocity(graph, fields)
}

pub fn project_velocity(
    graph: &Graph,
    fields: &mut Fields,
    maximum_iterations: u32,
    relative_tolerance: f32,
) -> Result<PressureReceipt, PressureError> {
    let rows = prepare_pressure_topology(graph, fields);
    assemble_pressure_rhs(graph, fields, &rows);
    let receipt = solve_pressure(
        graph,
        fields,
        &rows,
        maximum_iterations,
        relative_tolerance,
        None,
    )?;
    project_pressure_velocity(graph, fields, &rows);
    Ok(receipt)
}

fn box_fraction_projected(px: f32, py: f32, offset: f32) -> f32 {
    let dominant = px.max(py);
    if dominant <= 1e-20 {
        return if offset >= 0.0 { 1.0 } else { 0.0 };
    }
    let mut spans = Vec::with_capacity(2);
    for v in [div(px, dominant), div(py, dominant)] {
        if v >= 1e-6 {
            spans.push(v)
        }
    }
    let total = spans.iter().copied().fold(0.0, add);
    let shifted = add(div(offset, dominant), mul(0.5, total));
    if shifted <= 0.0 {
        return 0.0;
    }
    if shifted >= total {
        return 1.0;
    }
    let complement = shifted > mul(0.5, total);
    let x = if complement { total - shifted } else { shifted };
    let fraction = if spans.len() == 1 {
        div(x, spans[0])
    } else {
        let aa = spans[0].min(spans[1]);
        let bb = spans[0].max(spans[1]);
        if x < aa {
            mul(mul(0.5, div(x, aa)), div(x, bb))
        } else {
            div(x - mul(0.5, aa), bb)
        }
    }
    .clamp(0.0, 1.0);
    if complement {
        1.0 - fraction
    } else {
        fraction
    }
}
fn box_fraction(nx: f32, ny: f32, offset: f32, wx: f32, wy: f32) -> f32 {
    box_fraction_projected(mul(nx.abs(), wx), mul(ny.abs(), wy), offset)
}
fn offset_for_fill(fill: f32, nx: f32, ny: f32, widths: [f32; 2]) -> f32 {
    let projected = [mul(nx.abs(), widths[0]), mul(ny.abs(), widths[1])];
    let dominant = projected[0].max(projected[1]);
    let radius = mul(0.5, add(projected[0], projected[1]));
    if fill <= 0.0 {
        return -radius;
    }
    if fill >= 1.0 {
        return radius;
    }
    if fill == 0.5 || dominant <= 1e-20 {
        return 0.0;
    }
    let mut spans = Vec::new();
    for value in projected {
        let q = div(value, dominant);
        if q >= 1e-6 {
            spans.push(q)
        }
    }
    if spans.len() == 1 {
        return mul(mul(fill - 0.5, spans[0]), dominant);
    }
    let complement = fill > 0.5;
    let target = if complement { 1.0 - fill } else { fill };
    let aa = spans[0].min(spans[1]);
    let bb = spans[0].max(spans[1]);
    let mut shifted = add(mul(target, bb), mul(0.5, aa));
    if target < div(mul(0.5, aa), bb) {
        shifted = mul(mul(2.0, target), mul(aa, bb)).sqrt()
    }
    let result = mul(shifted - mul(0.5, add(aa, bb)), dominant);
    if complement {
        -result
    } else {
        result
    }
}

#[derive(Clone, Copy)]
struct Plane {
    nx: f32,
    ny: f32,
    offset: f32,
}
fn plane_from_fill(fill: f32, gx: f32, gy: f32, widths: [f32; 2]) -> Plane {
    let maximum = gx.abs().max(gy.abs());
    if maximum <= 1e-20 {
        return Plane {
            nx: 0.0,
            ny: 0.0,
            offset: 0.0,
        };
    }
    let sx = div(gx, maximum);
    let sy = div(gy, maximum);
    let length = add(mul(sx, sx), mul(sy, sy)).sqrt();
    let nx = div(sx, length);
    let ny = div(sy, length);
    Plane {
        nx,
        ny,
        offset: offset_for_fill(fill, nx, ny, widths),
    }
}
fn certificate_sample(value: f32) -> Option<f32> {
    (value.is_finite() && (-VOLUME_ROUNDOFF_RATIO..=1.0 + VOLUME_ROUNDOFF_RATIO).contains(&value))
        .then(|| value.clamp(0.0, 1.0))
}
fn fit_score(plane: Plane, samples: &[f32; 9]) -> (f32, f32) {
    let mut full = 0.0;
    let mut sides = [0.0; 4];
    for j in 0..3 {
        for i in 0..3 {
            let displaced =
                plane.offset - add(mul(plane.nx, i as f32 - 1.0), mul(plane.ny, j as f32 - 1.0));
            let difference =
                box_fraction(plane.nx, plane.ny, displaced, 1.0, 1.0) - samples[3 * j + i];
            let error = mul(difference, difference);
            full = add(full, error);
            if i <= 1 {
                sides[0] = add(sides[0], error)
            }
            if i >= 1 {
                sides[1] = add(sides[1], error)
            }
            if j <= 1 {
                sides[2] = add(sides[2], error)
            }
            if j >= 1 {
                sides[3] = add(sides[3], error)
            }
        }
    }
    (
        div(full, 9.0),
        div(sides.into_iter().fold(f32::INFINITY, f32::min), 6.0),
    )
}

/// Production least-squares PLIC fallback. 2-D graphs use the analytic CM12
/// box inverse; true 3-D PLIC is deliberately rejected until polyhedron clipping lands.
pub fn reconstruct_interfaces(graph: &Graph, fields: &mut Fields) -> Result<(), ValidationError> {
    reconstruct_interfaces_impl(graph, fields, false)
}

/// Reconstruct interfaces after a committed geometric remap. Isolated
/// sub-cell packets have no unique least-squares gradient, so retain their
/// previous orientation or use a deterministic axis fallback.
pub fn reconstruct_interfaces_for_cellwise_remap(
    graph: &Graph,
    fields: &mut Fields,
) -> Result<(), ValidationError> {
    reconstruct_interfaces_impl(graph, fields, true)
}

fn reconstruct_interfaces_impl(
    graph: &Graph,
    fields: &mut Fields,
    close_isolated_packets: bool,
) -> Result<(), ValidationError> {
    if graph.dimension == 3 {
        return crate::numerics3d::reconstruct_interfaces_3d(graph, fields);
    }
    if graph.dimension != 2 {
        return Err(ValidationError(
            "3-D PLIC requires plane/polyhedron clipping".into(),
        ));
    }
    // A transported sub-cell packet can be locally isolated, so the density
    // stencil has no unique gradient even though its PLIC volume is perfectly
    // well-defined. Retain the last orientation as the deterministic closure
    // for that case; a newly transferred/seeded packet falls back to +X.
    let previous_normals = fields.interface_normal.clone();
    fields.interface_normal.fill(0.0);
    fields.interface_offset.fill(0.0);
    for cell in 0..graph.cells.len() {
        let rho = fill(fields, cell);
        if fields.capacity[cell] < 0.999999 || !(rho > 0.0 && rho < 1.0) {
            continue;
        }
        let c = &graph.cells[cell];
        let (mut mxx, mut mxy, mut myy, mut bx, mut by) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            let Some(own) = own_term(row, cell) else {
                continue;
            };
            if physical_row_measure(row) <= 1e-8 {
                continue;
            }
            for term in &row.terms {
                if own.coefficient * term.coefficient >= 0.0 {
                    continue;
                }
                let j = term.cell_id as usize;
                if fields.capacity[j] < 0.999999 {
                    continue;
                }
                let other = &graph.cells[j];
                let dx = other.center[0] - c.center[0];
                let dy = other.center[1] - c.center[1];
                let tangent = 1 - row.axis as usize;
                let overlap = (c.maximum[tangent].min(other.maximum[tangent])
                    - c.minimum[tangent].max(other.minimum[tangent]))
                .max(0.0);
                let weight = div(overlap, add(mul(dx, dx), mul(dy, dy)).max(1e-12));
                let delta = fill(fields, j) - rho;
                mxx = add(mxx, mul(weight, mul(dx, dx)));
                mxy = add(mxy, mul(weight, mul(dx, dy)));
                myy = add(myy, mul(weight, mul(dy, dy)));
                bx = add(bx, mul(weight, mul(dx, delta)));
                by = add(by, mul(weight, mul(dy, delta)));
            }
        }
        let determinant = mul(mxx, myy) - mul(mxy, mxy);
        let scale = mxx.max(myy);
        if !(scale > 1e-12 && determinant.abs() > 1e-7 * scale * scale) {
            if !close_isolated_packets {
                continue;
            }
            let previous = [previous_normals[2 * cell], previous_normals[2 * cell + 1]];
            let direction = if previous[0].abs().max(previous[1].abs()) > 1e-20 {
                previous
            } else {
                [1.0, 0.0]
            };
            let fallback =
                plane_from_fill(rho, direction[0], direction[1], [c.widths[0], c.widths[1]]);
            fields.interface_normal[2 * cell] = fallback.nx;
            fields.interface_normal[2 * cell + 1] = fallback.ny;
            fields.interface_offset[cell] = fallback.offset;
            continue;
        }
        let gx = div(mul(myy, bx) - mul(mxy, by), determinant);
        let gy = div(-mul(mxy, bx) + mul(mxx, by), determinant);
        let mut best = plane_from_fill(rho, -gx, -gy, [c.widths[0], c.widths[1]]);
        if close_isolated_packets && best.nx == 0.0 && best.ny == 0.0 {
            let previous = [previous_normals[2 * cell], previous_normals[2 * cell + 1]];
            let direction = if previous[0].abs().max(previous[1].abs()) > 1e-20 {
                previous
            } else {
                [1.0, 0.0]
            };
            best = plane_from_fill(rho, direction[0], direction[1], [c.widths[0], c.widths[1]]);
        }
        if best.nx == 0.0 && best.ny == 0.0 {
            continue;
        }
        let mut samples = [0.0; 9];
        let mut certified = c.widths[0] == 1.0 && c.widths[1] == 1.0;
        for j in 0..3 {
            for i in 0..3 {
                if !certified {
                    break;
                }
                let point = [
                    c.center[0] + i as f32 - 1.0,
                    c.center[1] + j as f32 - 1.0,
                    0.0,
                ];
                let Some(other_id) = owner_at(graph, point) else {
                    certified = false;
                    break;
                };
                let other = &graph.cells[other_id];
                let Some(sample) = certificate_sample(fill(fields, other_id)) else {
                    certified = false;
                    break;
                };
                if other.widths[0] != 1.0
                    || other.widths[1] != 1.0
                    || other.center[0] != point[0]
                    || other.center[1] != point[1]
                    || fields.capacity[other_id] < 0.999999
                {
                    certified = false;
                    break;
                }
                samples[3 * j + i] = sample
            }
        }
        if certified {
            let mut best_score = fit_score(best, &samples);
            let mut corner = best;
            let mut corner_score = best_score;
            if best_score.0 > 9.094947e-13 {
                for direction in 0..2 {
                    let mut heights = [0.0; 3];
                    for column in 0..3 {
                        for at in 0..3 {
                            let index = if direction == 1 {
                                3 * at + column
                            } else {
                                3 * column + at
                            };
                            heights[column] = add(heights[column], samples[index])
                        }
                    }
                    for difference in 0..3 {
                        let slope = match difference {
                            0 => heights[1] - heights[0],
                            1 => mul(0.5, heights[2] - heights[0]),
                            _ => heights[2] - heights[1],
                        };
                        let integration = if direction == 1 { 1 } else { 0 };
                        let orientation_component =
                            if integration == 0 { best.nx } else { best.ny };
                        if orientation_component == 0.0 {
                            continue;
                        }
                        let orientation = if orientation_component > 0.0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let candidate = if integration == 0 {
                            plane_from_fill(rho, orientation, -slope, [1.0, 1.0])
                        } else {
                            plane_from_fill(rho, -slope, orientation, [1.0, 1.0])
                        };
                        let score = fit_score(candidate, &samples);
                        if score.0 < best_score.0 {
                            best = candidate;
                            best_score = score
                        }
                        if score.1 < corner_score.1
                            || (score.1 == corner_score.1 && score.0 < corner_score.0)
                        {
                            corner = candidate;
                            corner_score = score
                        }
                    }
                }
                if best_score.0 > 9.094947e-13 && corner_score.1 <= 9.094947e-13 {
                    best = corner
                }
            }
        }
        fields.interface_normal[2 * cell] = best.nx;
        fields.interface_normal[2 * cell + 1] = best.ny;
        fields.interface_offset[cell] = best.offset;
    }
    Ok(())
}

pub(crate) fn plic_box_fraction(nx: f32, ny: f32, offset: f32, wx: f32, wy: f32) -> f32 {
    box_fraction(nx, ny, offset, wx, wy)
}
pub(crate) fn plic_box_fraction_rect(nx: f32, ny: f32, offset: f32, wx: f64, wy: f64) -> f32 {
    box_fraction_projected(
        (nx.abs() as f64 * wx) as f32,
        (ny.abs() as f64 * wy) as f32,
        offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};
    use crate::types::Cell;

    fn test_brick(key: u32, coordinate: [i32; 3], resolution: u8) -> BrickSeed {
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

    fn uniform_test_graph() -> Graph {
        compile_topology::<2>(TopologySeed {
            dimensions: [8, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![test_brick(0, [0, 0, 0], 8)],
        })
        .unwrap()
        .graph
    }

    fn seam_test_graph() -> Graph {
        compile_topology::<2>(TopologySeed {
            dimensions: [16, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![
                test_brick(0, [0, 0, 0], 4),
                test_brick(1, [1, 0, 0], 8),
            ],
        })
        .unwrap()
        .graph
    }

    fn streamfunction_test_fields(graph: &Graph) -> Fields {
        let n = graph.cells.len();
        Fields {
            density: vec![1.0; n],
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            capacity_before: vec![1.0; n],
            capacity_after: vec![1.0; n],
            frame_dt: 1.0 / 30.0,
            source_rate: vec![0.0; n],
            cell_velocity: vec![0.0; 2 * n],
            face_velocity: vec![0.0; graph.rows.len()],
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![1.0; n],
            pressure_member: vec![1; n],
            pressure_row_member: vec![1; graph.rows.len()],
            extension_depth: vec![0; n],
            interface_normal: vec![0.0; 2 * n],
            interface_offset: vec![0.0; n],
            ..Fields::default()
        }
    }

    fn compatible_rates(graph: &Graph) -> Vec<f64> {
        let lx = graph.dimensions[0] as f64;
        let ly = graph.dimensions[1] as f64;
        let psi = |x: f64, y: f64| {
            (std::f64::consts::PI * x / lx).sin()
                * (std::f64::consts::PI * y / ly).sin()
        };
        graph
            .subfaces
            .iter()
            .map(|face| {
                let half = 0.5 * face.measure as f64;
                if face.axis == 0 {
                    let x = face.center[0] as f64;
                    psi(x, face.center[1] as f64 + half)
                        - psi(x, face.center[1] as f64 - half)
                } else {
                    let y = face.center[1] as f64;
                    -(psi(face.center[0] as f64 + half, y)
                        - psi(face.center[0] as f64 - half, y))
                }
            })
            .collect()
    }

    fn maximum_rate_divergence(graph: &Graph, rates: &[f64]) -> f64 {
        let mut divergence = vec![0.0_f64; graph.cells.len()];
        for (face, &rate) in graph.subfaces.iter().zip(rates) {
            if face.negative_cell >= 0 {
                divergence[face.negative_cell as usize] += rate;
            }
            if face.positive_cell >= 0 {
                divergence[face.positive_cell as usize] -= rate;
            }
        }
        divergence.into_iter().map(f64::abs).fold(0.0, f64::max)
    }

    #[test]
    fn streamfunction_reconciles_only_f32_scale_pressure_cycles() {
        let graph = uniform_test_graph();
        let mut fields = streamfunction_test_fields(&graph);
        let pocket = graph
            .cells
            .iter()
            .find(|cell| cell.center[0] == 4.5 && cell.center[1] == 4.5)
            .unwrap()
            .id as usize;
        fields.pressure_member[pocket] = 0;
        let face = graph
            .subfaces
            .iter()
            .find(|face| {
                graph.rows[face.row_id as usize].kind != RowKind::ClosedWorld
                    && [face.negative_cell, face.positive_cell].contains(&(pocket as i32))
            })
            .unwrap()
            .id as usize;
        let receiver = vec![true; graph.cells.len()];
        let compatible = compatible_rates(&graph);
        fields.subface_compatibility_rate = compatible.clone();
        let base = FaceConsistentVelocity2d::new(&graph, &fields, fields.frame_dt).unwrap();

        let mut roundoff = compatible.clone();
        roundoff[face] += f32::EPSILON as f64;
        streamfunction_extension_rates_2d(&graph, &fields, &mut roundoff, &receiver, true)
            .unwrap();
        assert!(maximum_rate_divergence(&graph, &roundoff) <= 1.0e-11);
        base.with_subface_rates(&graph, &roundoff, &receiver)
            .unwrap();

        let mut physical_mismatch = compatible;
        physical_mismatch[face] += 0.1;
        let error = streamfunction_extension_rates_2d(
            &graph,
            &fields,
            &mut physical_mismatch,
            &receiver,
            true,
        )
        .unwrap_err();
        assert!(
            error.0.contains("above derived f32 roundoff bound"),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn streamfunction_commits_adaptive_subface_constraints_atomically() {
        let graph = seam_test_graph();
        let mut fields = streamfunction_test_fields(&graph);
        let compatible = compatible_rates(&graph);
        fields.subface_compatibility_rate = compatible.clone();
        let face = graph
            .subfaces
            .iter()
            .find(|face| {
                face.measure > 1.0
                    && graph.rows[face.row_id as usize].kind != RowKind::ClosedWorld
            })
            .expect("adaptive fixture has no multi-unit physical subface")
            .id as usize;
        let receiver = vec![true; graph.cells.len()];
        let base = FaceConsistentVelocity2d::new(&graph, &fields, fields.frame_dt).unwrap();
        let mut rates = compatible;
        rates[face] += f32::EPSILON as f64;

        streamfunction_extension_rates_2d(&graph, &fields, &mut rates, &receiver, true).unwrap();
        assert!(maximum_rate_divergence(&graph, &rates) <= 1.0e-11);
        base.with_subface_rates(&graph, &rates, &receiver)
            .unwrap();
    }

    #[test]
    fn global_streamfunction_preserves_remote_closed_world_rates_bit_exact() {
        let graph = uniform_test_graph();
        let mut fields = streamfunction_test_fields(&graph);
        fields.pressure_member.fill(0);
        fields.pressure_row_member.fill(0);
        let mut rates = compatible_rates(&graph);
        fields.subface_compatibility_rate = rates.clone();
        let base = FaceConsistentVelocity2d::new(&graph, &fields, fields.frame_dt).unwrap();
        let mut receiver = vec![false; graph.cells.len()];
        let centre = graph
            .cells
            .iter()
            .find(|cell| cell.center[0] == 4.5 && cell.center[1] == 4.5)
            .unwrap()
            .id as usize;
        receiver[centre] = true;
        let closed: Vec<_> = graph
            .subfaces
            .iter()
            .filter(|face| graph.rows[face.row_id as usize].kind == RowKind::ClosedWorld)
            .map(|face| (face.id as usize, rates[face.id as usize].to_bits()))
            .collect();
        assert!(!closed.is_empty());

        streamfunction_extension_rates_2d(&graph, &fields, &mut rates, &receiver, true).unwrap();
        for (face, bits) in closed {
            assert_eq!(rates[face].to_bits(), bits, "ClosedWorld face {face} changed");
        }
        base.with_subface_rates(&graph, &rates, &receiver)
            .unwrap();
    }
    #[test]
    fn box_fraction_is_symmetric() {
        assert_eq!(box_fraction(1.0, 0.0, 0.0, 2.0, 1.0), 0.5);
        assert_eq!(box_fraction(-1.0, 0.0, 0.0, 2.0, 1.0), 0.5)
    }
    fn cell(id: u32, minimum: [f32; 3], maximum: [f32; 3]) -> Cell {
        Cell {
            id,
            minimum,
            maximum,
            center: [
                (minimum[0] + maximum[0]) * 0.5,
                (minimum[1] + maximum[1]) * 0.5,
                (minimum[2] + maximum[2]) * 0.5,
            ],
            widths: [
                maximum[0] - minimum[0],
                maximum[1] - minimum[1],
                maximum[2] - minimum[2],
            ],
            measure: 1.0,
            ..Default::default()
        }
    }
    #[test]
    fn indexed_owner_preserves_half_open_boundaries() {
        let graph = Graph {
            dimension: 2,
            dimensions: [2.0, 1.0, 1.0],
            cells: vec![
                cell(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                cell(1, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            ],
            ..Default::default()
        };
        graph.initialize_spatial_owner_cache();
        assert_eq!(owner_at(&graph, [0.0, 0.0, -999.0]), Some(0));
        assert_eq!(owner_at(&graph, [1.0, 0.5, 999.0]), Some(1));
        assert_eq!(owner_at(&graph, [2.0, 0.5, 0.0]), None);
        assert_eq!(owner_at(&graph, [f32::NAN, 0.5, 0.0]), None)
    }
    #[test]
    fn fractional_graph_uses_exact_linear_fallback() {
        let graph = Graph {
            dimension: 2,
            dimensions: [2.0, 1.0, 1.0],
            cells: vec![cell(0, [0.25, 0.0, 0.0], [1.25, 1.0, 1.0])],
            ..Default::default()
        };
        assert_eq!(owner_at(&graph, [0.1, 0.5, 0.0]), None);
        assert_eq!(owner_at(&graph, [0.3, 0.5, 0.0]), Some(0));
        assert_eq!(owner_at(&graph, [1.25, 0.5, 0.0]), None)
    }
    #[test]
    fn sparse_three_dimensional_owner_keeps_depth() {
        let graph = Graph {
            dimension: 3,
            dimensions: [64.0, 64.0, 64.0],
            cells: vec![
                cell(0, [8.0, 8.0, 8.0], [16.0, 16.0, 16.0]),
                cell(1, [8.0, 8.0, 24.0], [16.0, 16.0, 32.0]),
            ],
            ..Default::default()
        };
        graph.initialize_spatial_owner_cache();
        assert_eq!(owner_at(&graph, [12.0, 12.0, 12.0]), Some(0));
        assert_eq!(owner_at(&graph, [12.0, 12.0, 28.0]), Some(1));
        assert_eq!(owner_at(&graph, [12.0, 12.0, 20.0]), None);
    }
}
