//! Primary pressure correction coupled to sealed represented-air components.
//!
//! A closed air component contributes one uniform pressure unknown. This lets
//! the primary solve choose its aggregate liquid/air interface flux while the
//! later secondary solve remains confined to air/air faces.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    numerics::{PressureReceipt, PressureRows},
    pressure::{solve_pressure_pcg, PressureError},
    types::{Fields, Graph, RowKind, SubfaceIncidence},
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClosedAirPressureReceipt {
    pub pressure: PressureReceipt,
    pub sealed_component_count: usize,
    pub f64_refinement_iterations: u32,
    pub pre_refinement_normalized_residual: f64,
    pub maximum_normalized_physical_residual: f64,
    pub normalized_target: f64,
    pub skipped_for_solid_motion: bool,
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    let mut root = i;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[i] != i {
        let next = parent[i];
        parent[i] = root;
        i = next;
    }
    root
}

fn physical_velocity(fields: &Fields, graph: &Graph, row: usize) -> f32 {
    let r = &graph.rows[row];
    fields.face_velocity[row] - (1.0 - r.open_fraction) * r.solid_velocity
}

/// Refine the current primary projection while enforcing aggregate volume
/// compatibility for each sealed represented-air component.
pub fn solve_closed_air_coupled_pressure(
    graph: &Graph,
    fields: &mut Fields,
    rows: &PressureRows,
    maximum_iterations: u32,
    relative_tolerance: f32,
    execution_order: Option<&[u32]>,
) -> Result<ClosedAirPressureReceipt, PressureError> {
    if graph.dimension != 3
        || fields.pressure_member.len() != graph.cells.len()
        || rows.active.len() != graph.rows.len()
        || rows.theta.len() != graph.rows.len()
    {
        return Err(PressureError(
            "closed-air pressure plane lengths differ".into(),
        ));
    }
    let n = graph.cells.len();
    let air: Vec<bool> = (0..n)
        .map(|i| {
            fields.pressure_member[i] == 0 && graph.cells[i].measure * fields.capacity[i] > 0.0
        })
        .collect();
    let mut air_parent: Vec<usize> = (0..n).collect();
    for face in &graph.subfaces {
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 0.0 || face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let a = face.negative_cell as usize;
        let b = face.positive_cell as usize;
        if air[a] && air[b] {
            let ra = find(&mut air_parent, a);
            let rb = find(&mut air_parent, b);
            let root = ra.min(rb);
            air_parent[ra] = root;
            air_parent[rb] = root;
        }
    }
    for i in 0..n {
        if air[i] {
            air_parent[i] = find(&mut air_parent, i);
        }
    }
    let mut open_air = BTreeSet::new();
    for face in &graph.subfaces {
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 0.0 {
            continue;
        }
        if face.negative_cell < 0 && face.positive_cell >= 0 {
            let cell = face.positive_cell as usize;
            if air[cell] {
                open_air.insert(air_parent[cell]);
            }
        } else if face.positive_cell < 0 && face.negative_cell >= 0 {
            let cell = face.negative_cell as usize;
            if air[cell] {
                open_air.insert(air_parent[cell]);
            }
        }
    }

    // Only sealed components incident to a primary row need a supernode.
    let mut coupled_roots = BTreeSet::new();
    for row in &graph.rows {
        if rows.active[row.id as usize] == 0 || rows.theta[row.id as usize] <= 0.0 {
            continue;
        }
        let touches_pressure = row
            .terms
            .iter()
            .any(|t| fields.pressure_member[t.cell_id as usize] != 0);
        if !touches_pressure {
            continue;
        }
        for term in &row.terms {
            let cell = term.cell_id as usize;
            if air[cell] && !open_air.contains(&air_parent[cell]) {
                coupled_roots.insert(air_parent[cell]);
            }
        }
    }

    let mut pressure_unknown = vec![None; n];
    let pressure_order: Vec<usize> = execution_order
        .map(|order| order.iter().map(|&v| v as usize).collect())
        .unwrap_or_else(|| (0..n).collect());
    let mut unknown_count = 0;
    for cell in pressure_order {
        if cell < n && fields.pressure_member[cell] != 0 && pressure_unknown[cell].is_none() {
            pressure_unknown[cell] = Some(unknown_count);
            unknown_count += 1;
        }
    }
    for (cell, slot) in pressure_unknown.iter_mut().enumerate() {
        if fields.pressure_member[cell] != 0 && slot.is_none() {
            *slot = Some(unknown_count);
            unknown_count += 1;
        }
    }
    let mut air_unknown = BTreeMap::new();
    for component in coupled_roots {
        air_unknown.insert(component, unknown_count);
        unknown_count += 1;
    }
    if unknown_count == 0 {
        return Ok(ClosedAirPressureReceipt {
            normalized_target: 2.0 * f32::EPSILON as f64,
            ..Default::default()
        });
    }

    let mapped = |cell: usize| -> Option<usize> {
        pressure_unknown[cell].or_else(|| {
            air[cell]
                .then(|| air_unknown.get(&air_parent[cell]).copied())
                .flatten()
        })
    };
    let mut row_terms = vec![Vec::<(usize, f32)>::new(); graph.rows.len()];
    let mut row_weight = vec![0.0f32; graph.rows.len()];
    for row in &graph.rows {
        let ri = row.id as usize;
        if rows.active[ri] == 0 || rows.theta[ri] <= 0.0 {
            continue;
        }
        let open = if row.kind == RowKind::ClosedWorld {
            f32::from(row.separating)
        } else {
            row.open_fraction
        };
        row_weight[ri] = row.static_dual_weight.unwrap_or(row.dual_weight) * open / rows.theta[ri];
        let mut terms = BTreeMap::<usize, f32>::new();
        for term in &row.terms {
            if let Some(u) = mapped(term.cell_id as usize) {
                *terms.entry(u).or_default() += term.coefficient;
            }
        }
        row_terms[ri] = terms.into_iter().collect();
    }

    let mut diagonal = vec![0.0f32; unknown_count];
    for (ri, terms) in row_terms.iter().enumerate() {
        for &(u, coefficient) in terms {
            diagonal[u] += row_weight[ri] * coefficient * coefficient;
        }
    }
    let mut rhs = vec![0.0f32; unknown_count];
    let mut rhs64 = vec![0.0f64; unknown_count];
    let mut unknown_capacity = vec![0.0f64; unknown_count];
    for cell in 0..n {
        let Some(u) = pressure_unknown[cell] else {
            continue;
        };
        unknown_capacity[u] = graph.cells[cell].measure as f64 * fields.capacity[cell] as f64;
        let mut value = 0.0f64;
        for &row_id in &graph.incidences[cell] {
            let row = &graph.rows[row_id as usize];
            if rows.active[row_id as usize] == 0 {
                continue;
            }
            if let Some(term) = row.terms.iter().find(|t| t.cell_id as usize == cell) {
                value += term.coefficient as f64
                    * row.static_dual_weight.unwrap_or(row.dual_weight) as f64
                    * physical_velocity(fields, graph, row_id as usize) as f64;
            }
        }
        value -= fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
            * graph.cells[cell].measure as f64;
        value += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
        rhs64[u] = value;
        rhs[u] = value as f32;
    }
    let mut fallback = vec![Vec::<SubfaceIncidence>::new(); n];
    for face in &graph.subfaces {
        if face.negative_cell >= 0 {
            fallback[face.negative_cell as usize].push(SubfaceIncidence {
                subface_id: face.id,
                orientation: -1,
            });
        }
        if face.positive_cell >= 0 {
            fallback[face.positive_cell as usize].push(SubfaceIncidence {
                subface_id: face.id,
                orientation: 1,
            });
        }
    }
    let incidences = if graph.subface_incidences.len() == n {
        &graph.subface_incidences
    } else {
        &fallback
    };
    for (&component, &u) in &air_unknown {
        let mut value = 0.0f64;
        for cell in 0..n {
            if !air[cell] || air_parent[cell] != component {
                continue;
            }
            for entry in &incidences[cell] {
                let face = &graph.subfaces[entry.subface_id as usize];
                value += entry.orientation as f64
                    * face.measure as f64
                    * physical_velocity(fields, graph, face.row_id as usize) as f64;
            }
            value -= fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
                * graph.cells[cell].measure as f64;
            value += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
            unknown_capacity[u] += graph.cells[cell].measure as f64 * fields.capacity[cell] as f64;
        }
        rhs64[u] = value;
        rhs[u] = value as f32;
    }

    // Pin one gauge unknown in every operator component without a Dirichlet
    // endpoint. An omitted row term is a zero-potential anchor.
    let mut system_parent: Vec<usize> = (0..unknown_count).collect();
    let mut anchored = vec![false; unknown_count];
    for (ri, terms) in row_terms.iter().enumerate() {
        if row_weight[ri] <= 0.0 || terms.is_empty() {
            continue;
        }
        let first = terms[0].0;
        for &(u, _) in &terms[1..] {
            let a = find(&mut system_parent, first);
            let b = find(&mut system_parent, u);
            let root = a.min(b);
            system_parent[a] = root;
            system_parent[b] = root;
        }
        let sum: f32 = terms.iter().map(|t| t.1).sum();
        let scale: f32 = terms.iter().map(|t| t.1.abs()).sum();
        if sum.abs() > 1e-6 * scale.max(1.0) {
            anchored[first] = true;
        }
    }
    for i in 0..unknown_count {
        system_parent[i] = find(&mut system_parent, i);
    }
    let mut root_anchored = vec![false; unknown_count];
    for i in 0..unknown_count {
        root_anchored[system_parent[i]] |= anchored[i];
    }
    let mut member = vec![1u8; unknown_count];
    let mut pinned = BTreeSet::new();
    let mut pinned_unknowns = Vec::new();
    for i in 0..unknown_count {
        let r = system_parent[i];
        if !root_anchored[r] && pinned.insert(r) {
            let component_rhs: f64 = (0..unknown_count)
                .filter(|&u| system_parent[u] == r)
                .map(|u| rhs64[u])
                .sum();
            let component_capacity: f64 = (0..unknown_count)
                .filter(|&u| system_parent[u] == r)
                .map(|u| unknown_capacity[u])
                .sum();
            let defect = component_rhs.abs() * fields.frame_dt as f64;
            let limit = 2.0 * f32::EPSILON as f64 * component_capacity;
            if !component_rhs.is_finite()
                || !component_capacity.is_finite()
                || component_capacity <= 0.0
                || !defect.is_finite()
                || !limit.is_finite()
                || defect > limit
            {
                return Err(PressureError(format!(
                    "closed pressure/air component {r} defect {defect} exceeds {limit}"
                )));
            }
            // The sealed system admits only a zero-sum operator image. Retain
            // its already-certified physical roundoff as a uniform normalized
            // target instead of concentrating the whole remainder on the
            // equation selected as the gauge.
            if component_capacity > 0.0 {
                for u in 0..unknown_count {
                    if system_parent[u] == r {
                        let target = component_rhs * unknown_capacity[u] / component_capacity;
                        rhs64[u] -= target;
                        rhs[u] = rhs64[u] as f32;
                    }
                }
            }
            member[i] = 0;
            rhs[i] = 0.0;
            pinned_unknowns.push(i);
        }
    }
    let apply = |input: &[f32], output: &mut [f32]| {
        output.fill(0.0);
        for (ri, terms) in row_terms.iter().enumerate() {
            if row_weight[ri] <= 0.0 {
                continue;
            }
            let gradient: f32 = terms.iter().map(|&(u, c)| c * input[u]).sum();
            for &(u, c) in terms {
                output[u] += row_weight[ri] * c * gradient;
            }
        }
    };
    let solved = solve_pressure_pcg(
        &diagonal,
        &rhs,
        &vec![0.0; unknown_count],
        &member,
        None,
        maximum_iterations,
        relative_tolerance,
        apply,
    )?;
    // Continue in f64 with the exact operator used for rate publication. The
    // f32 solve supplies a deterministic warm start; it is not the acceptance
    // authority for the strict physical-volume gate.
    let row_weight64: Vec<f64> = graph
        .rows
        .iter()
        .map(|row| {
            let ri = row.id as usize;
            let open = if row.kind == RowKind::ClosedWorld {
                f64::from(row.separating)
            } else {
                row.open_fraction as f64
            };
            if rows.active[ri] == 0 || rows.theta[ri] <= 0.0 {
                0.0
            } else {
                row.static_dual_weight.unwrap_or(row.dual_weight) as f64 * open
                    / rows.theta[ri] as f64
            }
        })
        .collect();
    let mut diagonal64 = vec![0.0f64; unknown_count];
    for (ri, terms) in row_terms.iter().enumerate() {
        for &(u, coefficient) in terms {
            diagonal64[u] += row_weight64[ri] * (coefficient as f64).powi(2);
        }
    }
    let apply64 = |input: &[f64], output: &mut [f64]| {
        output.fill(0.0);
        for (ri, terms) in row_terms.iter().enumerate() {
            if row_weight64[ri] <= 0.0 {
                continue;
            }
            let gradient: f64 = terms.iter().map(|&(u, c)| c as f64 * input[u]).sum();
            for &(u, c) in terms {
                output[u] += row_weight64[ri] * c as f64 * gradient;
            }
        }
    };
    let mut pressure64: Vec<f64> = solved.pressure.iter().map(|&v| v as f64).collect();
    for &u in &pinned_unknowns {
        pressure64[u] = 0.0;
    }
    let mut image64 = vec![0.0; unknown_count];
    apply64(&pressure64, &mut image64);
    let mut residual64: Vec<f64> = (0..unknown_count)
        .map(|u| {
            if member[u] != 0 {
                rhs64[u] - image64[u]
            } else {
                0.0
            }
        })
        .collect();
    let normalized = |u: usize, residual: f64| {
        if unknown_capacity[u] > 0.0 {
            residual.abs() * fields.frame_dt as f64 / unknown_capacity[u]
        } else if residual == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    };
    let pre_refinement_normalized_residual = (0..unknown_count)
        .map(|u| normalized(u, rhs64[u] - image64[u]))
        .fold(0.0f64, f64::max);
    let mut z64: Vec<f64> = (0..unknown_count)
        .map(|u| {
            if member[u] != 0 && diagonal64[u] > 0.0 {
                residual64[u] / diagonal64[u]
            } else {
                0.0
            }
        })
        .collect();
    let mut direction64 = z64.clone();
    let mut gamma64: f64 = (0..unknown_count).map(|u| residual64[u] * z64[u]).sum();
    let mut f64_refinement_iterations = 0;
    let mut true_image64 = vec![0.0; unknown_count];
    for _ in 0..maximum_iterations {
        apply64(&pressure64, &mut true_image64);
        let within = (0..unknown_count).all(|u| {
            (rhs64[u] - true_image64[u]).abs() * fields.frame_dt as f64
                <= 0.25 * 2.0 * f32::EPSILON as f64 * unknown_capacity[u]
        });
        if within || !(gamma64 > 0.0 && gamma64.is_finite()) {
            break;
        }
        apply64(&direction64, &mut image64);
        let curvature: f64 = (0..unknown_count)
            .map(|u| direction64[u] * image64[u])
            .sum();
        if !(curvature > 0.0 && curvature.is_finite()) {
            break;
        }
        let alpha = gamma64 / curvature;
        for u in 0..unknown_count {
            if member[u] != 0 {
                pressure64[u] += alpha * direction64[u];
                residual64[u] -= alpha * image64[u];
                z64[u] = if diagonal64[u] > 0.0 {
                    residual64[u] / diagonal64[u]
                } else {
                    0.0
                };
            }
        }
        f64_refinement_iterations += 1;
        let next_gamma: f64 = (0..unknown_count).map(|u| residual64[u] * z64[u]).sum();
        let beta = next_gamma / gamma64;
        gamma64 = next_gamma;
        for u in 0..unknown_count {
            if member[u] != 0 {
                direction64[u] = z64[u] + beta * direction64[u];
            }
        }
    }
    let base_subface_rate: Vec<f64> = graph
        .subfaces
        .iter()
        .map(|face| {
            face.measure as f64 * physical_velocity(fields, graph, face.row_id as usize) as f64
        })
        .collect();
    let mut row_correction = vec![0.0f64; graph.rows.len()];
    for row in &graph.rows {
        let ri = row.id as usize;
        if row_weight[ri] <= 0.0 {
            continue;
        }
        let gradient: f64 = row_terms[ri]
            .iter()
            .map(|&(u, c)| c as f64 * pressure64[u])
            .sum();
        let open = if row.kind == RowKind::ClosedWorld {
            f64::from(row.separating)
        } else {
            row.open_fraction as f64
        };
        row_correction[ri] = open * gradient / rows.theta[ri] as f64;
        fields.face_velocity[ri] -= row_correction[ri] as f32;
    }
    for cell in 0..n {
        if let Some(u) = pressure_unknown[cell] {
            fields.pressure[cell] += pressure64[u] as f32;
        }
    }
    for cell in 0..n {
        if air[cell] {
            if let Some(&u) = air_unknown.get(&air_parent[cell]) {
                fields.pressure[cell] = pressure64[u] as f32;
            }
        }
    }
    // Freeze the corrected physical bulk rates in f64. The complementary air
    // solve may replace A-A entries, but must preserve every primary row.
    fields.subface_compatibility_rate = graph
        .subfaces
        .iter()
        .enumerate()
        .map(|(i, face)| {
            base_subface_rate[i] - face.measure as f64 * row_correction[face.row_id as usize]
        })
        .collect();
    // Certify every primary liquid equation and every sealed-air aggregate
    // against the physical rates transport will consume. This includes the
    // equation omitted to fix each gauge; pinning must never hide a defect.
    let audit_unknown = |u: usize| -> f64 {
        let mut residual = 0.0f64;
        if let Some(cell) = pressure_unknown.iter().position(|&slot| slot == Some(u)) {
            for entry in &incidences[cell] {
                let rate = fields.subface_compatibility_rate[entry.subface_id as usize];
                residual += if entry.orientation < 0 { -rate } else { rate };
            }
            residual -= fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
                * graph.cells[cell].measure as f64;
            residual += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
        } else if let Some((&component, _)) = air_unknown.iter().find(|(_, slot)| **slot == u) {
            for cell in 0..n {
                if !air[cell] || air_parent[cell] != component {
                    continue;
                }
                for entry in &incidences[cell] {
                    let rate = fields.subface_compatibility_rate[entry.subface_id as usize];
                    residual += if entry.orientation < 0 { -rate } else { rate };
                }
                residual -= fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
                    * graph.cells[cell].measure as f64;
                residual += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
            }
        }
        residual
    };
    let pinned_set: BTreeSet<_> = pinned_unknowns.into_iter().collect();
    let mut maximum_normalized_physical_residual = 0.0f64;
    for u in 0..unknown_count {
        if pressure_unknown.iter().all(|&slot| slot != Some(u))
            && air_unknown.values().all(|&slot| slot != u)
        {
            continue;
        }
        let residual = audit_unknown(u);
        maximum_normalized_physical_residual =
            maximum_normalized_physical_residual.max(normalized(u, residual));
        let defect = residual.abs() * fields.frame_dt as f64;
        let limit = 2.0 * f32::EPSILON as f64 * unknown_capacity[u];
        if !residual.is_finite()
            || !unknown_capacity[u].is_finite()
            || unknown_capacity[u] <= 0.0
            || !defect.is_finite()
            || !limit.is_finite()
            || defect > limit
        {
            let kind = if air_unknown.values().any(|&slot| slot == u) {
                "sealed-air aggregate"
            } else {
                "primary liquid"
            };
            let gauge = if pinned_set.contains(&u) {
                " gauge"
            } else {
                ""
            };
            return Err(PressureError(format!(
                "closed pressure/air {kind}{gauge} equation {u} defect {defect} exceeds {limit}"
            )));
        }
    }
    Ok(ClosedAirPressureReceipt {
        pressure: PressureReceipt {
            iterations: solved.iterations,
            initial_residual: solved.initial_true_residual_squared.max(0.0).sqrt(),
            residual: solved.final_true_residual_squared.max(0.0).sqrt(),
            converged: solved.converged,
        },
        sealed_component_count: air_unknown.len(),
        f64_refinement_iterations,
        pre_refinement_normalized_residual,
        maximum_normalized_physical_residual,
        normalized_target: 2.0 * f32::EPSILON as f64,
        skipped_for_solid_motion: false,
    })
}
