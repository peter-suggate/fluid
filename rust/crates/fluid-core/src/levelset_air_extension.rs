//! Close divergence in the air band used to advect the direct level set.
//!
//! Averaging each velocity component independently preserves neither air-cell
//! divergence nor liquid area. Correct only air/air faces; the liquid pressure
//! solution and prescribed solid boundary velocities remain immutable.
use crate::{Fields, Graph, RowKind, ValidationError};

#[derive(Clone, Debug, Default)]
pub struct AirExtensionReceipt {
    pub constrained_cells: usize,
    pub corrected_faces: usize,
    pub iterations: usize,
    pub maximum_initial_divergence: f64,
    pub maximum_final_divergence_error: f64,
    /// Nonzero flux imposed on closed air components cannot be removed while
    /// preserving their liquid/solid boundary faces. Retain its uniform rate.
    pub maximum_compatible_divergence: f64,
    pub isolated_cells: usize,
}

fn root(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = root(parent, parent[i]);
    }
    parent[i]
}

pub fn project_air_extension(
    graph: &Graph,
    fields: &mut Fields,
    phi: &[f32],
) -> Result<AirExtensionReceipt, ValidationError> {
    let n = graph.cells.len();
    if graph.dimension != 2
        || phi.len() != n
        || fields.capacity.len() != n
        || fields.face_velocity.len() != graph.rows.len()
        || fields.capacity.iter().any(|c| !c.is_finite() || *c < 0.0)
        || phi.iter().any(|p| !p.is_finite())
        || fields.face_velocity.iter().any(|v| !v.is_finite())
    {
        return Err(ValidationError(
            "air extension requires finite 2-D phi and face fields".into(),
        ));
    }
    // Cover the interpolation stencil around the zero set. This is a support
    // requirement, not a material relocation radius or a density threshold.
    let mut active: Vec<bool> = graph
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            fields.capacity[i] > 0.0 && phi[i] > 0.0 && phi[i] <= 2.0 * c.widths[0].max(c.widths[1])
        })
        .collect();
    let free: Vec<bool> = graph
        .rows
        .iter()
        .map(|r| {
            r.kind != RowKind::ClosedWorld
                && r.open_fraction > 1e-8
                && r.terms.iter().all(|t| {
                    phi[t.cell_id as usize] > 0.0 && fields.capacity[t.cell_id as usize] > 0.0
                })
        })
        .collect();
    let mut rhs = vec![0.0_f64; n];
    let mut diagonal = vec![0.0; n];
    for r in &graph.rows {
        for t in &r.terms {
            let i = t.cell_id as usize;
            if active[i] {
                rhs[i] += t.coefficient as f64
                    * r.dual_weight as f64
                    * fields.face_velocity[r.id as usize] as f64;
                if free[r.id as usize] {
                    diagonal[i] += (t.coefficient as f64).powi(2)
                        * r.dual_weight as f64
                        * r.open_fraction as f64;
                }
            }
        }
    }
    let mut receipt = AirExtensionReceipt::default();
    for i in 0..n {
        if active[i] && diagonal[i] <= 1e-30 {
            active[i] = false;
            receipt.isolated_cells += 1;
        }
        if !active[i] {
            rhs[i] = 0.0;
        }
    }
    receipt.constrained_cells = active.iter().filter(|&&a| a).count();
    if receipt.constrained_cells == 0 {
        return Ok(receipt);
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for r in &graph.rows {
        if free[r.id as usize] {
            let ids: Vec<_> = r
                .terms
                .iter()
                .map(|t| t.cell_id as usize)
                .filter(|&i| active[i])
                .collect();
            if let Some(&first) = ids.first() {
                for &i in &ids[1..] {
                    let a = root(&mut parent, first);
                    let b = root(&mut parent, i);
                    parent[b] = a;
                }
            }
        }
    }
    for i in 0..n {
        parent[i] = root(&mut parent, i);
    }
    let mut anchored = vec![false; n];
    for r in &graph.rows {
        if free[r.id as usize] {
            let outside = r.terms.iter().any(|t| !active[t.cell_id as usize])
                || r.terms
                    .iter()
                    .map(|t| t.coefficient as f64)
                    .sum::<f64>()
                    .abs()
                    > 1e-12;
            if outside {
                for t in &r.terms {
                    if active[t.cell_id as usize] {
                        anchored[parent[t.cell_id as usize]] = true;
                    }
                }
            }
        }
    }
    let mut component_flux = vec![0.0; n];
    let mut component_measure = vec![0.0; n];
    for i in 0..n {
        if active[i] {
            component_flux[parent[i]] += rhs[i];
            component_measure[parent[i]] += graph.cells[i].measure as f64;
        }
    }
    let mut compatible = vec![0.0; n];
    for i in 0..n {
        if active[i] {
            let measure = graph.cells[i].measure as f64;
            receipt.maximum_initial_divergence = receipt
                .maximum_initial_divergence
                .max((rhs[i] / measure).abs());
            if !anchored[parent[i]] {
                compatible[i] = component_flux[parent[i]] / component_measure[parent[i]];
                rhs[i] -= compatible[i] * measure;
                receipt.maximum_compatible_divergence = receipt
                    .maximum_compatible_divergence
                    .max(compatible[i].abs());
            }
        }
    }
    let multiply = |p: &[f64]| {
        let mut out = vec![0.0; n];
        for r in &graph.rows {
            if free[r.id as usize] {
                let gradient: f64 = r
                    .terms
                    .iter()
                    .filter(|t| active[t.cell_id as usize])
                    .map(|t| t.coefficient as f64 * p[t.cell_id as usize])
                    .sum();
                for t in &r.terms {
                    let i = t.cell_id as usize;
                    if active[i] {
                        out[i] += t.coefficient as f64
                            * r.dual_weight as f64
                            * r.open_fraction as f64
                            * gradient;
                    }
                }
            }
        }
        out
    };
    let mut pressure = vec![0.0; n];
    let mut residual = rhs;
    let mut z: Vec<_> = (0..n)
        .map(|i| {
            if active[i] {
                residual[i] / diagonal[i]
            } else {
                0.0
            }
        })
        .collect();
    let mut direction = z.clone();
    let mut rz: f64 = residual.iter().zip(&z).map(|(a, b)| a * b).sum();
    let tolerance = (rz * 1e-16).max(1e-24);
    for step in 0..512 {
        if rz <= tolerance {
            break;
        }
        let product = multiply(&direction);
        let denominator: f64 = direction.iter().zip(&product).map(|(a, b)| a * b).sum();
        if denominator <= 1e-30 {
            break;
        }
        let alpha = rz / denominator;
        for i in 0..n {
            pressure[i] += alpha * direction[i];
            residual[i] -= alpha * product[i];
            z[i] = if active[i] {
                residual[i] / diagonal[i]
            } else {
                0.0
            };
        }
        let next: f64 = residual.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = next / rz;
        rz = next;
        for i in 0..n {
            direction[i] = z[i] + beta * direction[i];
        }
        receipt.iterations = step + 1;
    }
    if pressure.iter().any(|p| !p.is_finite()) {
        return Err(ValidationError(
            "air-extension pressure is non-finite".into(),
        ));
    }
    let mut corrected = fields.face_velocity.clone();
    for r in &graph.rows {
        if free[r.id as usize] {
            let gradient: f64 = r
                .terms
                .iter()
                .filter(|t| active[t.cell_id as usize])
                .map(|t| t.coefficient as f64 * pressure[t.cell_id as usize])
                .sum();
            if gradient != 0.0 {
                corrected[r.id as usize] -= (r.open_fraction as f64 * gradient) as f32;
                receipt.corrected_faces += 1;
            }
        }
    }
    if corrected.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError(
            "air-extension velocity is non-finite".into(),
        ));
    }
    fields.face_velocity = corrected;
    let mut final_flux = vec![0.0_f64; n];
    for r in &graph.rows {
        for t in &r.terms {
            let i = t.cell_id as usize;
            if active[i] {
                final_flux[i] += t.coefficient as f64
                    * r.dual_weight as f64
                    * fields.face_velocity[r.id as usize] as f64;
            }
        }
    }
    for i in 0..n {
        if active[i] {
            receipt.maximum_final_divergence_error = receipt
                .maximum_final_divergence_error
                .max((final_flux[i] / graph.cells[i].measure as f64 - compatible[i]).abs());
        }
    }
    Ok(receipt)
}
