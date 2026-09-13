//! Experimental two-dimensional compatibility projection for the cellwise
//! remap receiver band.
//!
//! The primary pressure operator remains authoritative. Only rates whose
//! incident represented cells are all outside that operator may change.
//! Receiver cells outside the operator are unknowns; adjacent nonreceiver air
//! and sparse-air exterior faces provide zero-potential Dirichlet vents.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::numerics::pressure_projection_velocity_roundoff_scale;
use crate::{Fields, Graph, RowKind, ValidationError};

pub const BAND_PROJECTION_NORMALIZED_TARGET: f64 = 2.0 * f32::EPSILON as f64;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BandProjectionReceipt2d {
    pub accepted: bool,
    pub unknowns: usize,
    pub components: usize,
    pub open_components: usize,
    pub enclosed_components: usize,
    pub infeasible_enclosed_components: usize,
    pub max_enclosed_component_normalized_defect: f64,
    pub worst_infeasible_component_root: Option<usize>,
    pub worst_infeasible_component_cells: usize,
    pub worst_infeasible_component_liquid_cells: usize,
    pub worst_infeasible_component_capacity: f64,
    pub worst_infeasible_component_signed_defect: f64,
    pub iterations: usize,
    pub initial_max_normalized_residual: f64,
    pub pre_postconditioning_max_normalized_residual: f64,
    pub measured_max_normalized_residual: f64,
    pub max_receiver_normalized_divergence: f64,
    pub fixed_projected_subfaces: usize,
    pub changed_subfaces: usize,
    pub zero_original_rate_change_count: usize,
    pub max_absolute_rate_change: f64,
    pub max_relative_rate_change: f64,
    pub postconditioning_rounds: usize,
    pub postconditioning_max_absolute_rate_change: f64,
    pub max_courant_before: f64,
    pub max_courant_after: f64,
}

/// Receipt for the private f64 cleanup of primary-pressure rates consumed by
/// whole-step geometric transport. The primary f32 velocity field is not
/// changed; this removes only a residual that fits a derived f32 evaluation
/// bound before the outside-operator band projection runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureRateCleanupReceipt2d {
    pub cells: usize,
    pub components: usize,
    pub open_components: usize,
    pub changed_subfaces: usize,
    pub max_normalized_divergence_before: f64,
    pub max_normalized_divergence_after: f64,
    pub max_normalized_roundoff_bound: f64,
    pub max_absolute_rate_change: f64,
    pub max_relative_rate_change: f64,
    pub zero_original_rate_change_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProjectionFace {
    correctable: bool,
    weight: f64,
}

/// Remove representable f32 projection residue from pressure-owned receiver
/// cells in the private f64 rate plane used by cellwise remap.
///
/// The cleanup is deliberately bounded before it changes a rate. A residual
/// must fit Higham's forward-error bound for the incident f32 physical-rate
/// construction and ordered divergence gather. This keeps the operation from
/// masking a materially unconverged primary solve. Closed-world rates remain
/// exact; component residue leaves through a pressure/non-pressure interface
/// and is subsequently handled by the ordinary outside-operator projection.
pub fn cleanup_pressure_receiver_rates_2d(
    graph: &Graph,
    fields: &Fields,
    rates: &mut [f64],
    receiver_mask: &[bool],
    dt: f32,
) -> Result<PressureRateCleanupReceipt2d, ValidationError> {
    graph.validate()?;
    fields.validate_for(graph)?;
    if graph.dimension != 2
        || rates.len() != graph.subfaces.len()
        || receiver_mask.len() != graph.cells.len()
        || !dt.is_finite()
        || dt <= 0.0
        || rates.iter().any(|rate| !rate.is_finite())
    {
        return Err(ValidationError(
            "pressure-rate cleanup input differs from the 2-D topology or is invalid".into(),
        ));
    }

    let n = graph.cells.len();
    let active: Vec<bool> = (0..n)
        .map(|cell| {
            receiver_mask[cell]
                && fields.pressure_member.get(cell).copied().unwrap_or(0) != 0
                && cell_capacity(graph, fields, cell) > 0.0
        })
        .collect();
    let cells = active.iter().filter(|&&value| value).count();
    if cells == 0 {
        return Ok(PressureRateCleanupReceipt2d::default());
    }

    let incidences = subface_incidences(graph);
    let projection_velocity_scale = pressure_projection_velocity_roundoff_scale(graph, fields);
    let correctable: Vec<bool> = graph
        .subfaces
        .iter()
        .map(|face| {
            let row = &graph.rows[face.row_id as usize];
            row.kind != RowKind::ClosedWorld
                && row.open_fraction > 0.0
                && face.aperture > 0.0
                && [face.negative_cell, face.positive_cell]
                    .into_iter()
                    .filter_map(represented_cell)
                    .any(|cell| active[cell])
        })
        .collect();

    let divergence_before = cell_divergence(graph, rates);
    let mut receipt = PressureRateCleanupReceipt2d {
        cells,
        ..Default::default()
    };
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        let capacity = cell_capacity(graph, fields, cell);
        let normalized = normalized_divergence(graph, fields, cell, divergence_before[cell], dt);
        receipt.max_normalized_divergence_before =
            receipt.max_normalized_divergence_before.max(normalized);
        let magnitude: f64 = incidences[cell]
            .iter()
            .map(|&face_index| {
                let face = &graph.subfaces[face_index];
                let correction = fields
                    .subface_velocity_correction
                    .get(face_index)
                    .copied()
                    .unwrap_or(0.0) as f64;
                let projected_scale = face.measure as f64
                    * (projection_velocity_scale[face.row_id as usize] + correction.abs());
                // The stored rate may be a small remainder of two large f32
                // operands: the predicted face velocity and its pressure
                // correction. Include both operand scales rather than using
                // only the cancellation result.
                rates[face_index].abs() + projected_scale
            })
            .sum();
        // Each physical rate reaches this plane through rounded row velocity,
        // open-fraction/correction arithmetic and an ordered signed gather.
        // Match the established 3-D compatibility cleanup's conservative
        // operation census instead of accepting an arbitrary residual limit.
        let operations = (8 * incidences[cell].len() + 8) as f64;
        let gamma_n = operations * f32::EPSILON as f64
            / (1.0 - operations * f32::EPSILON as f64);
        let absolute_bound = 3.0 * gamma_n * magnitude
            + BAND_PROJECTION_NORMALIZED_TARGET * capacity / (64.0 * dt as f64);
        receipt.max_normalized_roundoff_bound = receipt
            .max_normalized_roundoff_bound
            .max(dt as f64 * absolute_bound / capacity.max(f64::MIN_POSITIVE));
        if divergence_before[cell].abs() > absolute_bound {
            return Err(ValidationError(format!(
                "pressure-rate cleanup residual cell {cell} {} exceeds derived roundoff bound {absolute_bound}",
                divergence_before[cell].abs()
            )));
        }
    }

    let original = rates.to_vec();
    let mut parent: Vec<usize> = (0..n).collect();
    for (face_index, face) in graph.subfaces.iter().enumerate() {
        if !correctable[face_index] || face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let a = face.negative_cell as usize;
        let b = face.positive_cell as usize;
        if active[a] && active[b] {
            union_minimum(&mut parent, a, b);
        }
    }
    for cell in 0..n {
        if active[cell] {
            parent[cell] = find_root(&parent, cell);
        }
    }
    let roots: Vec<_> = (0..n)
        .filter(|&cell| active[cell] && parent[cell] == cell)
        .collect();
    receipt.components = roots.len();
    let mut component_vent = vec![None::<(usize, usize)>; n];
    for (face_index, face) in graph.subfaces.iter().enumerate() {
        if !correctable[face_index] {
            continue;
        }
        let negative = represented_cell(face.negative_cell);
        let positive = represented_cell(face.positive_cell);
        let active_cell = match (negative, positive) {
            (Some(a), Some(b)) if active[a] != active[b] => Some(if active[a] { a } else { b }),
            (Some(a), None) if active[a] => Some(a),
            (None, Some(b)) if active[b] => Some(b),
            _ => None,
        };
        if let Some(cell) = active_cell {
            component_vent[parent[cell]].get_or_insert((face_index, cell));
        }
    }
    receipt.open_components = roots
        .iter()
        .filter(|&&root| component_vent[root].is_some())
        .count();

    for &component in &roots {
        let members: Vec<_> = (0..n)
            .filter(|&cell| active[cell] && parent[cell] == component)
            .collect();
        let start = component_vent[component].map_or(members[0], |(_, cell)| cell);
        let mut tree = vec![None::<(usize, usize)>; n];
        let mut order = Vec::with_capacity(members.len());
        let mut seen = vec![false; n];
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            order.push(cell);
            for &face_index in &incidences[cell] {
                if !correctable[face_index] {
                    continue;
                }
                let Some(other) = represented_cell(other_cell(&graph.subfaces[face_index], cell))
                else {
                    continue;
                };
                if active[other] && !seen[other] {
                    seen[other] = true;
                    tree[other] = Some((cell, face_index));
                    queue.push_back(other);
                }
            }
        }
        if order.len() != members.len() {
            return Err(ValidationError(format!(
                "pressure-rate cleanup component {component} is disconnected"
            )));
        }

        let mut divergence = cell_divergence(graph, rates);
        for &cell in order.iter().rev() {
            let Some((parent_cell, face_index)) = tree[cell] else {
                continue;
            };
            let sign = outward_rate_sign(&graph.subfaces[face_index], cell);
            let change = -divergence[cell] / sign;
            rates[face_index] += change;
            divergence[parent_cell] += divergence[cell];
            divergence[cell] = 0.0;
        }
        let root_residual = divergence[start];
        if let Some((face_index, cell)) = component_vent[component] {
            rates[face_index] +=
                -root_residual / outward_rate_sign(&graph.subfaces[face_index], cell);
        } else if normalized_divergence(graph, fields, start, root_residual, dt)
            > BAND_PROJECTION_NORMALIZED_TARGET
        {
            return Err(ValidationError(format!(
                "pressure-rate cleanup closed component {component} residual cannot fit the physical tolerance"
            )));
        }
    }

    let divergence_after = cell_divergence(graph, rates);
    receipt.max_normalized_divergence_after = (0..n)
        .filter(|&cell| active[cell])
        .map(|cell| normalized_divergence(graph, fields, cell, divergence_after[cell], dt))
        .fold(0.0, f64::max);
    if receipt.max_normalized_divergence_after > BAND_PROJECTION_NORMALIZED_TARGET {
        return Err(ValidationError(format!(
            "pressure-rate cleanup residual {} exceeds {}",
            receipt.max_normalized_divergence_after, BAND_PROJECTION_NORMALIZED_TARGET
        )));
    }
    for (&before, &after) in original.iter().zip(rates.iter()) {
        let change = (after - before).abs();
        if change == 0.0 {
            continue;
        }
        receipt.changed_subfaces += 1;
        receipt.max_absolute_rate_change = receipt.max_absolute_rate_change.max(change);
        if before == 0.0 {
            receipt.zero_original_rate_change_count += 1;
        } else {
            receipt.max_relative_rate_change =
                receipt.max_relative_rate_change.max(change / before.abs());
        }
    }
    Ok(receipt)
}

/// Projects a private physical-subface rate field onto two-sided discrete
/// continuity over the supplied receiver band.
///
/// This function mutates only `rates`. Every face incident to a pressure
/// member and every closed-world face remains bit-exact. An enclosed component
/// with a nonrepresentable component sum returns `accepted = false` without
/// changing rates rather than routing its defect through a fixed face.
pub fn project_receiver_band_rates_2d(
    graph: &Graph,
    fields: &Fields,
    rates: &mut [f64],
    receiver_mask: &[bool],
    dt: f32,
    maximum_iterations: usize,
) -> Result<BandProjectionReceipt2d, ValidationError> {
    graph.validate()?;
    fields.validate_for(graph)?;
    if graph.dimension != 2 {
        return Err(ValidationError(
            "receiver-band projection requires a 2-D graph".into(),
        ));
    }
    if rates.len() != graph.subfaces.len() || receiver_mask.len() != graph.cells.len() {
        return Err(ValidationError(
            "receiver-band projection input size differs from topology".into(),
        ));
    }
    if !dt.is_finite() || dt <= 0.0 || rates.iter().any(|rate| !rate.is_finite()) {
        return Err(ValidationError(
            "receiver-band projection input is non-finite or has invalid dt".into(),
        ));
    }

    let n = graph.cells.len();
    let original = rates.to_vec();
    let active: Vec<bool> = (0..n)
        .map(|cell| {
            receiver_mask[cell]
                && fields.pressure_member.get(cell).copied().unwrap_or(0) == 0
                && cell_capacity(graph, fields, cell) > 0.0
        })
        .collect();
    let unknowns = active.iter().filter(|&&value| value).count();
    let fixed_projected_subfaces = graph
        .subfaces
        .iter()
        .filter(|face| face_incident_to_pressure(fields, face))
        .count();
    let max_courant_before = max_receiver_courant(graph, fields, &original, receiver_mask, dt);

    if unknowns == 0 {
        return Ok(BandProjectionReceipt2d {
            accepted: true,
            max_receiver_normalized_divergence: max_receiver_normalized_divergence(
                graph,
                fields,
                rates,
                receiver_mask,
                dt,
            ),
            fixed_projected_subfaces,
            max_courant_before,
            max_courant_after: max_courant_before,
            ..Default::default()
        });
    }

    let incidences = subface_incidences(graph);
    let faces = compile_projection_faces(graph, fields, &active)?;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut projected_rates = original.clone();
    for (face_index, face) in graph.subfaces.iter().enumerate() {
        if !faces[face_index].correctable || face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let a = face.negative_cell as usize;
        let b = face.positive_cell as usize;
        if active[a] && active[b] {
            union_minimum(&mut parent, a, b);
        }
    }
    for cell in 0..n {
        if active[cell] {
            parent[cell] = find_root(&parent, cell);
        }
    }

    let roots: Vec<_> = (0..n)
        .filter(|&cell| active[cell] && parent[cell] == cell)
        .collect();
    let mut component_open = vec![false; n];
    let mut component_vent = vec![None::<(usize, usize)>; n];
    let mut component_capacity = vec![0.0_f64; n];
    let divergence_before = cell_divergence(graph, &original);
    let mut component_divergence = vec![0.0_f64; n];
    for cell in 0..n {
        if active[cell] {
            let root = parent[cell];
            component_capacity[root] += cell_capacity(graph, fields, cell);
            component_divergence[root] += divergence_before[cell];
        }
    }
    for (face_index, face) in graph.subfaces.iter().enumerate() {
        if !faces[face_index].correctable {
            continue;
        }
        let negative = represented_cell(face.negative_cell);
        let positive = represented_cell(face.positive_cell);
        match (negative, positive) {
            (Some(a), Some(b)) if active[a] != active[b] => {
                let cell = if active[a] { a } else { b };
                let root = parent[cell];
                component_open[root] = true;
                component_vent[root].get_or_insert((face_index, cell));
            }
            (Some(a), None) if active[a] => {
                component_open[parent[a]] = true;
                component_vent[parent[a]].get_or_insert((face_index, a));
            }
            (None, Some(b)) if active[b] => {
                component_open[parent[b]] = true;
                component_vent[parent[b]].get_or_insert((face_index, b));
            }
            _ => {}
        }
    }

    let mut infeasible_enclosed_components = 0;
    let mut max_enclosed_component_normalized_defect = 0.0_f64;
    let mut worst_infeasible_component_root = None;
    let mut worst_infeasible_component_cells = 0;
    let mut worst_infeasible_component_liquid_cells = 0;
    let mut worst_infeasible_component_capacity = 0.0;
    let mut worst_infeasible_component_signed_defect = 0.0;
    for &root in &roots {
        if component_open[root] {
            continue;
        }
        let normalized = dt as f64 * component_divergence[root].abs()
            / component_capacity[root].max(f64::MIN_POSITIVE);
        if normalized > BAND_PROJECTION_NORMALIZED_TARGET {
            infeasible_enclosed_components += 1;
            if normalized > max_enclosed_component_normalized_defect {
                max_enclosed_component_normalized_defect = normalized;
                worst_infeasible_component_root = Some(root);
                worst_infeasible_component_cells = (0..n)
                    .filter(|&cell| active[cell] && parent[cell] == root)
                    .count();
                worst_infeasible_component_liquid_cells = (0..n)
                    .filter(|&cell| {
                        active[cell] && parent[cell] == root && fields.density[cell] > 0.0
                    })
                    .count();
                worst_infeasible_component_capacity = component_capacity[root];
                worst_infeasible_component_signed_defect = component_divergence[root];
            }
        }
    }
    if infeasible_enclosed_components != 0 {
        return Ok(BandProjectionReceipt2d {
            accepted: false,
            unknowns,
            components: roots.len(),
            open_components: roots.iter().filter(|&&root| component_open[root]).count(),
            enclosed_components: roots.iter().filter(|&&root| !component_open[root]).count(),
            infeasible_enclosed_components,
            max_enclosed_component_normalized_defect,
            worst_infeasible_component_root,
            worst_infeasible_component_cells,
            worst_infeasible_component_liquid_cells,
            worst_infeasible_component_capacity,
            worst_infeasible_component_signed_defect,
            initial_max_normalized_residual: max_receiver_normalized_divergence(
                graph, fields, rates, &active, dt,
            ),
            measured_max_normalized_residual: max_receiver_normalized_divergence(
                graph, fields, rates, &active, dt,
            ),
            max_receiver_normalized_divergence: max_receiver_normalized_divergence(
                graph,
                fields,
                rates,
                receiver_mask,
                dt,
            ),
            fixed_projected_subfaces,
            max_courant_before,
            max_courant_after: max_courant_before,
            ..Default::default()
        });
    }

    let mut rhs = vec![0.0_f64; n];
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        let root = parent[cell];
        rhs[cell] = -divergence_before[cell];
        if !component_open[root] {
            rhs[cell] += component_divergence[root] * cell_capacity(graph, fields, cell)
                / component_capacity[root];
        }
    }

    let mut diagonal = vec![0.0_f64; n];
    let mut neighbors = vec![Vec::<(usize, f64)>::new(); n];
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        for &face_index in &incidences[cell] {
            let compiled = faces[face_index];
            if !compiled.correctable {
                continue;
            }
            diagonal[cell] += compiled.weight;
            let other = other_cell(&graph.subfaces[face_index], cell);
            if let Some(other) = represented_cell(other) {
                if active[other] {
                    neighbors[cell].push((other, compiled.weight));
                }
            }
        }
        neighbors[cell].sort_by_key(|&(other, _)| other);
        if diagonal[cell] <= 0.0 || !diagonal[cell].is_finite() {
            return Err(ValidationError(format!(
                "receiver-band projection unknown cell {cell} has no correction degree of freedom"
            )));
        }
    }

    let apply = |input: &[f64], output: &mut [f64]| {
        for cell in 0..n {
            output[cell] = if active[cell] {
                neighbors[cell]
                    .iter()
                    .fold(diagonal[cell] * input[cell], |value, &(other, weight)| {
                        value - weight * input[other]
                    })
            } else {
                0.0
            };
        }
    };
    let precondition = |residual: &[f64], output: &mut [f64]| {
        for cell in 0..n {
            if !active[cell] {
                output[cell] = 0.0;
                continue;
            }
            let value = neighbors[cell]
                .iter()
                .filter(|&&(other, _)| other < cell)
                .fold(residual[cell], |value, &(other, weight)| {
                    value + weight * output[other]
                });
            output[cell] = value / diagonal[cell];
        }
        for cell in (0..n).rev() {
            if !active[cell] {
                continue;
            }
            let value = neighbors[cell]
                .iter()
                .filter(|&&(other, _)| other > cell)
                .fold(output[cell], |value, &(other, weight)| {
                    value + weight * output[other] / diagonal[cell]
                });
            output[cell] = value;
        }
    };

    let initial_max_normalized_residual = max_normalized_vector(graph, fields, &rhs, &active, dt);
    let mut potential = vec![0.0_f64; n];
    let mut residual = rhs.clone();
    let mut preconditioned = vec![0.0_f64; n];
    precondition(&residual, &mut preconditioned);
    let mut direction = preconditioned.clone();
    let mut image = vec![0.0_f64; n];
    let mut gamma = dot_active(&residual, &preconditioned, &active);
    let mut iterations = 0;
    let solve_target = BAND_PROJECTION_NORMALIZED_TARGET / 64.0;
    while max_normalized_vector(graph, fields, &residual, &active, dt) > solve_target
        && iterations < maximum_iterations
    {
        apply(&direction, &mut image);
        let curvature = dot_active(&direction, &image, &active);
        if !(curvature > 0.0 && curvature.is_finite() && gamma.is_finite()) {
            return Err(ValidationError(format!(
                "receiver-band projection invalid Krylov curvature {curvature}"
            )));
        }
        let alpha = gamma / curvature;
        for cell in 0..n {
            if active[cell] {
                potential[cell] += alpha * direction[cell];
                residual[cell] -= alpha * image[cell];
            }
        }
        iterations += 1;
        if iterations % 64 == 0 {
            apply(&potential, &mut image);
            for cell in 0..n {
                if active[cell] {
                    residual[cell] = rhs[cell] - image[cell];
                }
            }
        }
        precondition(&residual, &mut preconditioned);
        let next_gamma = dot_active(&residual, &preconditioned, &active);
        if max_normalized_vector(graph, fields, &residual, &active, dt) <= solve_target {
            break;
        }
        if !(gamma > 0.0) {
            return Err(ValidationError(
                "receiver-band projection non-positive Krylov gamma".into(),
            ));
        }
        let beta = next_gamma / gamma;
        for cell in 0..n {
            if active[cell] {
                direction[cell] = preconditioned[cell] + beta * direction[cell];
            }
        }
        gamma = next_gamma;
    }
    apply(&potential, &mut image);
    for cell in 0..n {
        if active[cell] {
            residual[cell] = rhs[cell] - image[cell];
        }
    }
    if max_normalized_vector(graph, fields, &residual, &active, dt) > solve_target {
        return Err(ValidationError(format!(
            "receiver-band projection did not reach its roundoff precondition in {maximum_iterations} iterations"
        )));
    }

    for (face_index, face) in graph.subfaces.iter().enumerate() {
        let compiled = faces[face_index];
        if !compiled.correctable {
            continue;
        }
        let negative = represented_cell(face.negative_cell)
            .filter(|&cell| active[cell])
            .map_or(0.0, |cell| potential[cell]);
        let positive = represented_cell(face.positive_cell)
            .filter(|&cell| active[cell])
            .map_or(0.0, |cell| potential[cell]);
        projected_rates[face_index] -= compiled.weight * (positive - negative);
    }

    let pre_postconditioning_max_normalized_residual = {
        let divergence = cell_divergence(graph, &projected_rates);
        (0..n)
            .filter(|&cell| active[cell])
            .map(|cell| normalized_divergence(graph, fields, cell, divergence[cell], dt))
            .fold(0.0, f64::max)
    };
    let rates_before_postconditioning = projected_rates.clone();
    let postconditioning_rounds = usize::from(
        pre_postconditioning_max_normalized_residual > BAND_PROJECTION_NORMALIZED_TARGET,
    );
    if postconditioning_rounds != 0 {
        postcondition_components(
            graph,
            fields,
            &mut projected_rates,
            &active,
            &faces,
            &incidences,
            &parent,
            &component_open,
            &component_vent,
            dt,
        )?;
    }
    let postconditioning_max_absolute_rate_change = projected_rates
        .iter()
        .zip(&rates_before_postconditioning)
        .map(|(after, before)| (after - before).abs())
        .fold(0.0, f64::max);

    let divergence_after = cell_divergence(graph, &projected_rates);
    let measured_max_normalized_residual = (0..n)
        .filter(|&cell| active[cell])
        .map(|cell| normalized_divergence(graph, fields, cell, divergence_after[cell], dt))
        .fold(0.0, f64::max);
    if measured_max_normalized_residual > BAND_PROJECTION_NORMALIZED_TARGET {
        return Err(ValidationError(format!(
            "receiver-band projection residual {measured_max_normalized_residual} exceeds {}",
            BAND_PROJECTION_NORMALIZED_TARGET
        )));
    }

    let mut changed_subfaces = 0;
    let mut zero_original_rate_change_count = 0;
    let mut max_absolute_rate_change = 0.0_f64;
    let mut max_relative_rate_change = 0.0_f64;
    for (before, after) in original.iter().zip(projected_rates.iter()) {
        let change = (after - before).abs();
        if change == 0.0 {
            continue;
        }
        changed_subfaces += 1;
        max_absolute_rate_change = max_absolute_rate_change.max(change);
        if *before == 0.0 {
            zero_original_rate_change_count += 1;
        } else {
            max_relative_rate_change = max_relative_rate_change.max(change / before.abs());
        }
    }

    rates.copy_from_slice(&projected_rates);
    Ok(BandProjectionReceipt2d {
        accepted: true,
        unknowns,
        components: roots.len(),
        open_components: roots.iter().filter(|&&root| component_open[root]).count(),
        enclosed_components: roots.iter().filter(|&&root| !component_open[root]).count(),
        infeasible_enclosed_components,
        max_enclosed_component_normalized_defect,
        worst_infeasible_component_root,
        worst_infeasible_component_cells,
        worst_infeasible_component_liquid_cells,
        worst_infeasible_component_capacity,
        worst_infeasible_component_signed_defect,
        iterations,
        initial_max_normalized_residual,
        pre_postconditioning_max_normalized_residual,
        measured_max_normalized_residual,
        max_receiver_normalized_divergence: max_receiver_normalized_divergence(
            graph,
            fields,
            &projected_rates,
            receiver_mask,
            dt,
        ),
        fixed_projected_subfaces,
        changed_subfaces,
        zero_original_rate_change_count,
        max_absolute_rate_change,
        max_relative_rate_change,
        postconditioning_rounds,
        postconditioning_max_absolute_rate_change,
        max_courant_before,
        max_courant_after: max_receiver_courant(graph, fields, &projected_rates, receiver_mask, dt),
    })
}

fn compile_projection_faces(
    graph: &Graph,
    fields: &Fields,
    active: &[bool],
) -> Result<Vec<ProjectionFace>, ValidationError> {
    graph
        .subfaces
        .iter()
        .map(|face| {
            let row = &graph.rows[face.row_id as usize];
            let negative = represented_cell(face.negative_cell);
            let positive = represented_cell(face.positive_cell);
            let active_count = negative.is_some_and(|cell| active[cell]) as usize
                + positive.is_some_and(|cell| active[cell]) as usize;
            let pressure_fixed = face_incident_to_pressure(fields, face);
            let other_is_vent = match (negative, positive) {
                (Some(a), Some(b)) if active[a] != active[b] => {
                    let other = if active[a] { b } else { a };
                    fields.pressure_member.get(other).copied().unwrap_or(0) == 0
                        && cell_capacity(graph, fields, other) > 0.0
                }
                (Some(a), None) if active[a] => true,
                (None, Some(b)) if active[b] => true,
                _ => false,
            };
            let correctable = row.kind != RowKind::ClosedWorld
                && row.open_fraction > 0.0
                && !pressure_fixed
                && (active_count == 2 || active_count == 1 && other_is_vent);
            let distance = if correctable {
                match (negative, positive) {
                    (Some(a), Some(b)) => (graph.cells[b].center[face.axis as usize]
                        - graph.cells[a].center[face.axis as usize])
                        .abs() as f64,
                    (Some(a), None) | (None, Some(a)) => (face.center[face.axis as usize]
                        - graph.cells[a].center[face.axis as usize])
                        .abs() as f64,
                    _ => 0.0,
                }
            } else {
                0.0
            };
            if correctable && !(distance > 0.0 && distance.is_finite()) {
                return Err(ValidationError(format!(
                    "receiver-band projection face {} has invalid distance",
                    face.id
                )));
            }
            Ok(ProjectionFace {
                correctable,
                weight: if correctable {
                    face.measure as f64 * row.open_fraction as f64 / distance
                } else {
                    0.0
                },
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn postcondition_components(
    graph: &Graph,
    fields: &Fields,
    rates: &mut [f64],
    active: &[bool],
    faces: &[ProjectionFace],
    incidences: &[Vec<usize>],
    parent: &[usize],
    component_open: &[bool],
    component_vent: &[Option<(usize, usize)>],
    dt: f32,
) -> Result<(), ValidationError> {
    let n = graph.cells.len();
    let roots: Vec<_> = (0..n)
        .filter(|&cell| active[cell] && parent[cell] == cell)
        .collect();
    for root in roots {
        let members: Vec<_> = (0..n)
            .filter(|&cell| active[cell] && parent[cell] == root)
            .collect();
        let start = component_vent[root].map_or(members[0], |(_, cell)| cell);
        let mut tree = vec![None::<(usize, usize)>; n];
        let mut order = Vec::with_capacity(members.len());
        let mut seen = vec![false; n];
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            order.push(cell);
            for &face_index in &incidences[cell] {
                if !faces[face_index].correctable {
                    continue;
                }
                let other = other_cell(&graph.subfaces[face_index], cell);
                let Some(other) = represented_cell(other) else {
                    continue;
                };
                if active[other] && !seen[other] {
                    seen[other] = true;
                    tree[other] = Some((cell, face_index));
                    queue.push_back(other);
                }
            }
        }
        let mut divergence = cell_divergence(graph, rates);
        for &cell in order.iter().rev() {
            let Some((parent_cell, face_index)) = tree[cell] else {
                continue;
            };
            let sign = outward_rate_sign(&graph.subfaces[face_index], cell);
            let change = -divergence[cell] / sign;
            rates[face_index] += change;
            divergence[parent_cell] += divergence[cell];
            divergence[cell] = 0.0;
        }
        let root_residual = divergence[start];
        if component_open[root] {
            let (vent, active_cell) = component_vent[root].ok_or_else(|| {
                ValidationError(format!(
                    "receiver-band projection open component {root} has no vent face"
                ))
            })?;
            rates[vent] += -root_residual / outward_rate_sign(&graph.subfaces[vent], active_cell);
        } else {
            let normalized = normalized_divergence(graph, fields, start, root_residual, dt);
            if normalized > BAND_PROJECTION_NORMALIZED_TARGET {
                return Err(ValidationError(format!(
                    "receiver-band projection enclosed component {root} postcondition {normalized} exceeds {}",
                    BAND_PROJECTION_NORMALIZED_TARGET
                )));
            }
        }
    }
    Ok(())
}

fn subface_incidences(graph: &Graph) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); graph.cells.len()];
    for face in &graph.subfaces {
        if let Some(cell) = represented_cell(face.negative_cell) {
            result[cell].push(face.id as usize);
        }
        if let Some(cell) = represented_cell(face.positive_cell) {
            result[cell].push(face.id as usize);
        }
    }
    for entries in &mut result {
        entries.sort_unstable();
    }
    result
}

fn cell_divergence(graph: &Graph, rates: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; graph.cells.len()];
    for (face, &rate) in graph.subfaces.iter().zip(rates) {
        if let Some(cell) = represented_cell(face.negative_cell) {
            result[cell] += rate;
        }
        if let Some(cell) = represented_cell(face.positive_cell) {
            result[cell] -= rate;
        }
    }
    result
}

fn max_receiver_normalized_divergence(
    graph: &Graph,
    fields: &Fields,
    rates: &[f64],
    receiver_mask: &[bool],
    dt: f32,
) -> f64 {
    let divergence = cell_divergence(graph, rates);
    (0..graph.cells.len())
        .filter(|&cell| receiver_mask[cell])
        .map(|cell| normalized_divergence(graph, fields, cell, divergence[cell], dt))
        .fold(0.0, f64::max)
}

fn max_receiver_courant(
    graph: &Graph,
    fields: &Fields,
    rates: &[f64],
    receiver_mask: &[bool],
    dt: f32,
) -> f64 {
    let mut outgoing = vec![0.0_f64; graph.cells.len()];
    for (face, &rate) in graph.subfaces.iter().zip(rates) {
        if let Some(cell) = represented_cell(face.negative_cell) {
            outgoing[cell] += rate.max(0.0);
        }
        if let Some(cell) = represented_cell(face.positive_cell) {
            outgoing[cell] += (-rate).max(0.0);
        }
    }
    (0..graph.cells.len())
        .filter(|&cell| receiver_mask[cell])
        .map(|cell| dt as f64 * outgoing[cell] / cell_capacity(graph, fields, cell).max(1.0e-30))
        .fold(0.0, f64::max)
}

fn max_normalized_vector(
    graph: &Graph,
    fields: &Fields,
    values: &[f64],
    active: &[bool],
    dt: f32,
) -> f64 {
    (0..graph.cells.len())
        .filter(|&cell| active[cell])
        .map(|cell| normalized_divergence(graph, fields, cell, values[cell], dt))
        .fold(0.0, f64::max)
}

fn normalized_divergence(
    graph: &Graph,
    fields: &Fields,
    cell: usize,
    divergence: f64,
    dt: f32,
) -> f64 {
    dt as f64 * divergence.abs() / cell_capacity(graph, fields, cell).max(1.0e-30)
}

fn cell_capacity(graph: &Graph, fields: &Fields, cell: usize) -> f64 {
    graph.cells[cell].measure as f64 * fields.capacity[cell] as f64
}

fn face_incident_to_pressure(fields: &Fields, face: &crate::Subface) -> bool {
    [face.negative_cell, face.positive_cell]
        .into_iter()
        .filter_map(represented_cell)
        .any(|cell| fields.pressure_member.get(cell).copied().unwrap_or(0) != 0)
}

fn represented_cell(cell: i32) -> Option<usize> {
    (cell >= 0).then_some(cell as usize)
}

fn other_cell(face: &crate::Subface, cell: usize) -> i32 {
    if face.negative_cell == cell as i32 {
        face.positive_cell
    } else {
        face.negative_cell
    }
}

fn outward_rate_sign(face: &crate::Subface, cell: usize) -> f64 {
    if face.negative_cell == cell as i32 {
        1.0
    } else {
        -1.0
    }
}

fn find_root(parent: &[usize], mut cell: usize) -> usize {
    while parent[cell] != cell {
        cell = parent[cell];
    }
    cell
}

fn union_minimum(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    let root = ra.min(rb);
    parent[ra] = root;
    parent[rb] = root;
}

fn dot_active(a: &[f64], b: &[f64], active: &[bool]) -> f64 {
    a.iter()
        .zip(b)
        .zip(active)
        .filter(|&(_, &is_active)| is_active)
        .map(|((&a, &b), _)| a * b)
        .sum()
}
