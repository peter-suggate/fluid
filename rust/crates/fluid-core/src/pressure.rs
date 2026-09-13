//! Production Sparse CM12 Chronopoulos-Gear PCG recurrence.

use crate::kernels::{
    div, is_dense_order, mul, pressure_xrz_active, pressure_xrz_dense, reduce_dot,
    reduce_production, scaled_add_active, scaled_add_dense,
};
use serde::{Deserialize, Serialize};

pub const TRUE_RESIDUAL_CADENCE: usize = 8;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressurePcgIteration {
    pub encoded_iteration: u32,
    pub gate_open: bool,
    pub gamma: f32,
    pub alpha: f32,
    pub beta: f32,
    pub recursive_residual_squared: f32,
    pub guarded_true_residual_squared: f32,
    pub executed_iterations: u32,
    pub curvature_breakdown: bool,
    pub curvature_recoveries: u32,
    pub first_tolerance_iteration: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PressurePcgReceipt {
    pub pressure: Vec<f32>,
    pub residual: Vec<f32>,
    pub iterations: u32,
    pub encoded_iterations: u32,
    pub initial_true_residual_squared: f32,
    pub final_true_residual_squared: f32,
    pub final_true_residual_maximum: f32,
    pub rhs_squared: f32,
    pub recursive_residual_squared: f32,
    pub first_tolerance_iteration: Option<u32>,
    pub curvature_recoveries: u32,
    pub residual_drift: bool,
    pub converged: bool,
    pub records: Vec<PressurePcgIteration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PressureError(pub String);
impl std::fmt::Display for PressureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PressureError {}

fn measure_true_residual<F>(
    rhs: &[f32],
    pressure: &[f32],
    residual: &mut [f32],
    active: &[u32],
    image: &mut [f32],
    squares: &mut [f32],
    apply: &mut F,
) -> (f32, f32)
where
    F: FnMut(&[f32], &mut [f32]),
{
    apply(pressure, image);
    squares.fill(0.0);
    let mut maximum = 0.0f32;
    for &id in active {
        let i = id as usize;
        let value = rhs[i] - image[i];
        residual[i] = value;
        squares[i] = mul(value, value);
        maximum = maximum.max(value.abs());
    }
    (reduce_production(squares, active), maximum)
}

/// Executes the fixed-budget device-gated recurrence. `apply` must write the
/// exact composite operator image in the graph's canonical accumulation order.
#[allow(unused_assignments)] // Mirrors the resident initializePCG publication before the true-residual refresh.
pub fn solve_pressure_pcg<F>(
    diagonal: &[f32],
    rhs: &[f32],
    initial_pressure: &[f32],
    member: &[u8],
    execution_order: Option<&[u32]>,
    maximum_iterations: u32,
    relative_tolerance: f32,
    mut apply: F,
) -> Result<PressurePcgReceipt, PressureError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let count = rhs.len();
    if diagonal.len() != count || initial_pressure.len() != count || member.len() != count {
        return Err(PressureError("pressure PCG plane lengths differ".into()));
    }
    if !(relative_tolerance.is_finite() && relative_tolerance >= 0.0) {
        return Err(PressureError(
            "relativeTolerance must be finite and nonnegative".into(),
        ));
    }
    let active: Vec<u32> = execution_order.map(ToOwned::to_owned).unwrap_or_else(|| {
        member
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v != 0).then_some(i as u32))
            .collect()
    });
    let mut seen = vec![false; count];
    for &id in &active {
        let i = id as usize;
        if i >= count || member[i] == 0 || seen[i] {
            return Err(PressureError(
                "executionOrder must contain unique pressure members".into(),
            ));
        }
        seen[i] = true;
    }
    let dense_order = is_dense_order(&active, count);

    let mut pressure = initial_pressure.to_vec();
    let mut residual = vec![0.0; count];
    let mut z = vec![0.0; count];
    let mut direction = vec![0.0; count];
    let mut image_direction = vec![0.0; count];
    let mut image_z = vec![0.0; count];
    let mut guard_residual = vec![0.0; count];
    let mut applied = vec![0.0; count];
    let mut products = vec![0.0; count];
    let mut squares = vec![0.0; count];

    apply(&pressure, &mut applied);
    for &id in &active {
        let i = id as usize;
        residual[i] = rhs[i] - applied[i];
        z[i] = if diagonal[i] > 0.0 {
            div(residual[i], diagonal[i])
        } else {
            0.0
        };
        direction[i] = z[i];
    }
    let mut gamma = reduce_dot(&residual, &z, &active, &mut products);
    let rhs_squared = reduce_dot(rhs, rhs, &active, &mut products);
    let mut live = true;
    let mut curvature = false;
    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut recursive_residual_squared = 0.0;
    let mut iterations = 0u32;
    let mut first_tolerance = None;
    let mut curvature_recoveries = 0u32;
    let initial = measure_true_residual(
        rhs,
        &pressure,
        &mut residual,
        &active,
        &mut applied,
        &mut squares,
        &mut apply,
    );
    for &id in &active {
        let i = id as usize;
        z[i] = if diagonal[i] > 0.0 {
            div(residual[i], diagonal[i])
        } else {
            0.0
        };
    }
    gamma = reduce_dot(&residual, &z, &active, &mut products);
    let tolerance_squared = mul(mul(relative_tolerance, relative_tolerance), rhs_squared);
    if relative_tolerance > 0.0 && initial.0 <= tolerance_squared {
        live = false;
        first_tolerance = Some(0);
    }
    if live {
        apply(&z, &mut image_z);
        image_direction.copy_from_slice(&image_z);
        let delta = reduce_dot(&z, &image_z, &active, &mut products);
        if delta > 1e-20 {
            alpha = div(gamma, delta);
        } else {
            curvature = true;
            curvature_recoveries += 1;
        }
    }

    let mut guarded_true_residual_squared = 0.0;
    let mut records = Vec::with_capacity(maximum_iterations as usize + 1);
    macro_rules! record {
        ($encoded:expr) => {
            records.push(PressurePcgIteration {
                encoded_iteration: $encoded,
                gate_open: live && !curvature,
                gamma,
                alpha,
                beta,
                recursive_residual_squared,
                guarded_true_residual_squared,
                executed_iterations: iterations,
                curvature_breakdown: curvature,
                curvature_recoveries,
                first_tolerance_iteration: first_tolerance,
            });
        };
    }
    record!(0);
    for iteration in 0..maximum_iterations {
        if live && !curvature {
            if iterations > 0 {
                if dense_order {
                    scaled_add_dense(&z, beta, &mut direction);
                    scaled_add_dense(&image_z, beta, &mut image_direction)
                } else {
                    scaled_add_active(&z, beta, &mut direction, &seen);
                    scaled_add_active(&image_z, beta, &mut image_direction, &seen);
                }
            }
            if dense_order {
                pressure_xrz_dense(
                    &mut pressure,
                    &mut residual,
                    &mut z,
                    alpha,
                    &direction,
                    &image_direction,
                    diagonal,
                )
            } else {
                pressure_xrz_active(
                    &mut pressure,
                    &mut residual,
                    &mut z,
                    alpha,
                    &direction,
                    &image_direction,
                    diagonal,
                    &seen,
                );
            }
            apply(&z, &mut image_z);
            let next_gamma = reduce_dot(&residual, &z, &active, &mut products);
            let delta = reduce_dot(&image_z, &z, &active, &mut products);
            recursive_residual_squared = reduce_dot(&residual, &residual, &active, &mut products);
            let previous_gamma = gamma;
            let previous_alpha = alpha;
            beta = if previous_gamma > 1e-20 {
                div(next_gamma, previous_gamma)
            } else {
                0.0
            };
            let denominator = delta - mul(beta, div(next_gamma, previous_alpha.max(1e-20)));
            gamma = next_gamma;
            iterations += 1;
            if denominator > 1e-20 {
                alpha = div(next_gamma, denominator);
            } else {
                alpha = 0.0;
                curvature = true;
                curvature_recoveries += 1;
            }
        }
        if (iteration + 1) as usize % TRUE_RESIDUAL_CADENCE == 0
            && iteration + 1 < maximum_iterations
        {
            if live {
                let measured = measure_true_residual(
                    rhs,
                    &pressure,
                    &mut guard_residual,
                    &active,
                    &mut applied,
                    &mut squares,
                    &mut apply,
                );
                guarded_true_residual_squared = measured.0;
                if relative_tolerance > 0.0 && measured.0 <= tolerance_squared {
                    if first_tolerance.is_none() {
                        first_tolerance = Some(iterations);
                    }
                    live = false;
                } else if curvature || measured.0 > mul(16.0, recursive_residual_squared.max(1e-30))
                {
                    if !curvature {
                        curvature_recoveries += 1;
                    }
                    curvature = true;
                }
            }
            if live && curvature {
                residual.copy_from_slice(&guard_residual);
                for &id in &active {
                    let i = id as usize;
                    z[i] = if diagonal[i] > 0.0 {
                        div(residual[i], diagonal[i])
                    } else {
                        0.0
                    };
                    direction[i] = z[i];
                }
                gamma = reduce_dot(&residual, &z, &active, &mut products);
                beta = 0.0;
                apply(&z, &mut image_z);
                image_direction.copy_from_slice(&image_z);
                let delta = reduce_dot(&z, &image_z, &active, &mut products);
                if delta > 1e-20 {
                    alpha = div(gamma, delta);
                    curvature = false;
                } else {
                    live = false;
                }
            }
        }
        record!(iteration + 1);
    }
    let final_true = measure_true_residual(
        rhs,
        &pressure,
        &mut residual,
        &active,
        &mut applied,
        &mut squares,
        &mut apply,
    );
    if relative_tolerance > 0.0 && final_true.0 <= tolerance_squared && first_tolerance.is_none() {
        first_tolerance = Some(iterations);
    }
    Ok(PressurePcgReceipt {
        pressure,
        residual,
        iterations,
        encoded_iterations: maximum_iterations,
        initial_true_residual_squared: initial.0,
        final_true_residual_squared: final_true.0,
        final_true_residual_maximum: final_true.1,
        rhs_squared,
        recursive_residual_squared,
        first_tolerance_iteration: first_tolerance,
        curvature_recoveries,
        residual_drift: mul(16.0, recursive_residual_squared.max(0.0)) < final_true.0,
        converged: relative_tolerance > 0.0 && final_true.0 <= tolerance_squared,
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagonal_system_reaches_solution_and_keeps_fixed_budget() {
        let receipt = solve_pressure_pcg(
            &[2.0, 4.0],
            &[2.0, 8.0],
            &[0.0; 2],
            &[1, 1],
            None,
            16,
            1e-6,
            |input, out| {
                out[0] = 2.0 * input[0];
                out[1] = 4.0 * input[1];
            },
        )
        .unwrap();
        assert_eq!(receipt.encoded_iterations, 16);
        assert_eq!(receipt.records.len(), 17);
        assert!((receipt.pressure[0] - 1.0).abs() < 1e-5);
        assert!((receipt.pressure[1] - 2.0).abs() < 1e-5);
        assert!(receipt.converged);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn pcg_receipt_is_bitwise_across_pool_sizes() {
        let n = 1024;
        let diagonal: Vec<f32> = (0..n).map(|i| 1.0 + (i % 13) as f32 * 0.125).collect();
        let rhs: Vec<f32> = (0..n).map(|i| ((i % 29) as f32 - 14.0) * 0.03125).collect();
        let member = vec![1u8; n];
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    solve_pressure_pcg(
                        &diagonal,
                        &rhs,
                        &vec![0.0; n],
                        &member,
                        None,
                        24,
                        1e-6,
                        |input, out| {
                            for i in 0..n {
                                out[i] = diagonal[i] * input[i]
                            }
                        },
                    )
                    .unwrap()
                })
        };
        let baseline = run(1);
        for threads in [2, 4, 8] {
            let observed = run(threads);
            assert_eq!(
                observed
                    .pressure
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                baseline
                    .pressure
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                observed.final_true_residual_squared.to_bits(),
                baseline.final_true_residual_squared.to_bits()
            );
            assert_eq!(observed.iterations, baseline.iterations);
        }
    }
}
