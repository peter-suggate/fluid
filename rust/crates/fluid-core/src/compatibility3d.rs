//! Geometric transport compatibility projection for the non-pressure domain.
//!
//! This is the CPU image of `geometric-air-projection.wgsl.ts`: physical
//! subfaces define a singular component Laplacian, the compatible f32 RHS is
//! solved by Jacobi-preconditioned CG, and corrections remain per subface.

use crate::kernels::{add, div, mul};
use crate::types::{Fields, Graph, SubfaceIncidence, ValidationError, PRESSURE_REDUCTION_LANES};
use serde::{Deserialize, Serialize};

const NORMALIZED_TARGET: f32 = 2.0 * f32::EPSILON;
const VOLUME_ROUNDOFF_RATIO: f32 = 8.0 * f32::EPSILON;
const F64_KRYLOV_TARGET_FRACTION: f32 = 1.0 / 1024.0;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometricCompatibilityReceipt3d {
    pub accepted: bool,
    pub skipped_for_solid_motion: bool,
    pub iterations: u32,
    pub initial_normalized_residual: f32,
    pub recursive_normalized_residual: f32,
    pub measured_normalized_residual: f32,
    pub maximum_projected_volume_error: f32,
    pub pre_postconditioning_maximum: f32,
    pub pre_postconditioning_bound_maximum: f32,
    pub residual_correction_maximum: f32,
    pub postconditioning_rounds: u32,
}

#[inline]
fn cell_open_volume(graph: &Graph, fields: &Fields, cell: usize) -> f32 {
    mul(graph.cells[cell].measure, fields.capacity[cell])
}

#[inline]
fn cell_is_air(graph: &Graph, fields: &Fields, cell: usize) -> bool {
    cell < graph.cells.len()
        && cell_open_volume(graph, fields, cell) > 0.0
        && fields.pressure_member.get(cell).copied().unwrap_or(0) == 0
}

fn incidences<'a>(
    graph: &'a Graph,
    cell: usize,
    fallback: &'a [Vec<SubfaceIncidence>],
) -> &'a [SubfaceIncidence] {
    graph
        .subface_incidences
        .get(cell)
        .filter(|v| !v.is_empty())
        .unwrap_or(&fallback[cell])
}

fn fallback_incidences(graph: &Graph) -> Vec<Vec<SubfaceIncidence>> {
    let mut result = vec![Vec::new(); graph.cells.len()];
    for face in &graph.subfaces {
        if face.negative_cell >= 0 {
            result[face.negative_cell as usize].push(SubfaceIncidence {
                subface_id: face.id,
                orientation: -1,
            });
        }
        if face.positive_cell >= 0 {
            result[face.positive_cell as usize].push(SubfaceIncidence {
                subface_id: face.id,
                orientation: 1,
            });
        }
    }
    result
}

#[inline]
fn cell_delta(rate: f32, orientation: i8) -> f32 {
    if orientation < 0 {
        -rate
    } else {
        rate
    }
}

#[inline]
fn base_rate(graph: &Graph, fields: &Fields, face: usize) -> f32 {
    let subface = &graph.subfaces[face];
    let row = &graph.rows[subface.row_id as usize];
    let mut velocity = fields.face_velocity[row.id as usize];
    if row.open_fraction < 1.0 || row.solid_velocity != 0.0 {
        velocity = velocity - mul(1.0 - row.open_fraction, row.solid_velocity);
    }
    mul(subface.measure, velocity)
}

fn reduce_sum_max(values: &[(f32, f32)]) -> (f32, f32) {
    let mut sum = 0.0;
    let mut maximum: f32 = 0.0;
    for group in values.chunks(PRESSURE_REDUCTION_LANES) {
        let mut sums = [0.0; PRESSURE_REDUCTION_LANES];
        let mut maxima = [0.0; PRESSURE_REDUCTION_LANES];
        for (lane, &(s, m)) in group.iter().enumerate() {
            sums[lane] = s;
            maxima[lane] = m;
        }
        let mut width = PRESSURE_REDUCTION_LANES / 2;
        while width != 0 {
            for lane in 0..width {
                sums[lane] = add(sums[lane], sums[lane + width]);
                maxima[lane] = maxima[lane].max(maxima[lane + width]);
            }
            width /= 2;
        }
        sum = add(sum, sums[0]);
        maximum = maximum.max(maxima[0]);
    }
    (sum, maximum)
}

fn normalized(graph: &Graph, fields: &Fields, cell: usize, residual: f32) -> f32 {
    let capacity = cell_open_volume(graph, fields, cell);
    let value = div(mul(residual.abs(), fields.frame_dt), capacity);
    if capacity > 0.0 && value.is_finite() && value >= 0.0 {
        value
    } else {
        f32::MAX
    }
}

/// Projects physical subface rates in the complete accepted non-pressure
/// domain. The returned correction is also published into `fields` for the
/// subsequent conservative transport stage.
pub fn solve_geometric_transport_compatibility_3d(
    graph: &Graph,
    fields: &mut Fields,
    maximum_iterations: usize,
) -> Result<GeometricCompatibilityReceipt3d, ValidationError> {
    if graph.dimension != 3 {
        return Err(ValidationError(
            "geometric compatibility requires dimension 3".into(),
        ));
    }
    if fields.capacity.len() != graph.cells.len() || fields.face_velocity.len() != graph.rows.len()
    {
        return Err(ValidationError(
            "geometric compatibility field size differs".into(),
        ));
    }
    let n = graph.cells.len();
    let nf = graph.subfaces.len();
    if !fields.subface_compatibility_rate.is_empty()
        && fields.subface_compatibility_rate.len() != nf
    {
        return Err(ValidationError(
            "geometric compatibility incoming rate size differs".into(),
        ));
    }
    let incoming_rate = (!fields.subface_compatibility_rate.is_empty())
        .then(|| fields.subface_compatibility_rate.clone());
    let starting_rate: Vec<f64> = incoming_rate.unwrap_or_else(|| {
        (0..nf)
            .map(|face| base_rate(graph, fields, face) as f64)
            .collect()
    });
    fields.subface_velocity_correction = vec![0.0; nf];
    if fields.solid_motion_active {
        fields.subface_compatibility_rate.clear();
        return Ok(GeometricCompatibilityReceipt3d {
            accepted: true,
            skipped_for_solid_motion: true,
            ..Default::default()
        });
    }
    let fallback = fallback_incidences(graph);
    let mut active: Vec<bool> = (0..n).map(|i| cell_is_air(graph, fields, i)).collect();
    let mut correctable = vec![false; nf];
    let mut face_distance = vec![0.0; nf];
    for face in &graph.subfaces {
        let i = face.id as usize;
        let row = &graph.rows[face.row_id as usize];
        if row.open_fraction <= 0.0 {
            continue;
        }
        let negative = (face.negative_cell >= 0).then_some(face.negative_cell as usize);
        let positive = (face.positive_cell >= 0).then_some(face.positive_cell as usize);
        let eligible = match (negative, positive) {
            // Preserve every primary-owned liquid-adjacent face. The
            // secondary compatibility solve owns only air/air faces.
            (Some(a), Some(b)) => active[a] && active[b],
            (Some(a), None) | (None, Some(a)) => active[a],
            (None, None) => false,
        };
        if !eligible {
            continue;
        }
        let distance = match (negative, positive) {
            (Some(a), Some(b)) => (graph.cells[b].center[row.axis as usize]
                - graph.cells[a].center[row.axis as usize])
                .abs(),
            (Some(a), None) | (None, Some(a)) => {
                (face.center[row.axis as usize] - graph.cells[a].center[row.axis as usize]).abs()
            }
            _ => 0.0,
        };
        face_distance[i] = distance;
        correctable[i] = distance > 0.0 && distance.is_finite();
    }
    // Compatibility owns only cells incident to its non-pressure operator.
    // A newly exposed air cell surrounded entirely by pressure or closed
    // faces has no geometric correction degree of freedom; its boundary flux
    // remains owned by the primary pressure projection.
    for cell in 0..n {
        if active[cell]
            && !incidences(graph, cell, &fallback)
                .iter()
                .any(|entry| correctable[entry.subface_id as usize])
        {
            active[cell] = false;
        }
    }

    // Deterministic union by smallest root mirrors atomicMin component labels.
    let mut parent: Vec<usize> = (0..n).collect();
    for face in &graph.subfaces {
        if !correctable[face.id as usize] {
            continue;
        }
        if face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let a = face.negative_cell as usize;
        let b = face.positive_cell as usize;
        if !active[a] || !active[b] {
            continue;
        }
        let mut ra = a;
        while parent[ra] != ra {
            ra = parent[ra];
        }
        let mut rb = b;
        while parent[rb] != rb {
            rb = parent[rb];
        }
        let root = ra.min(rb);
        parent[ra] = root;
        parent[rb] = root;
    }
    for i in 0..n {
        if !active[i] {
            continue;
        }
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        parent[i] = root;
    }

    let physical_residual = |cell: usize| {
        let mut rate = 0.0f64;
        for entry in incidences(graph, cell, &fallback) {
            let value = starting_rate[entry.subface_id as usize];
            rate += if entry.orientation < 0 { -value } else { value };
        }
        let capacity_rate = fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
            * graph.cells[cell].measure as f64;
        (rate - capacity_rate + fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64) as f32
    };
    let raw_rhs: Vec<f32> = (0..n)
        .map(|i| if active[i] { physical_residual(i) } else { 0.0 })
        .collect();

    // A component has only internal correctable faces, so its Laplacian range
    // is the zero-sum subspace. Certify the physical defect at the same fixed
    // normalized threshold, then remove only that f32 accumulation residue.
    let mut component_rhs = vec![0.0; n];
    let mut component_capacity = vec![0.0; n];
    // Match the former production component gather: each 64-invocation group
    // forms one compensated partial per root before the globally ordered add.
    // Summing already-diverged cells naively can otherwise turn cancellation
    // of large paired face rates into a false compatibility defect.
    for first in (0..n).step_by(PRESSURE_REDUCTION_LANES) {
        let end = (first + PRESSURE_REDUCTION_LANES).min(n);
        for lane in first..end {
            if !active[lane]
                || (first..lane).any(|prior| active[prior] && parent[prior] == parent[lane])
            {
                continue;
            }
            let root = parent[lane];
            let mut rhs_sum = 0.0;
            let mut rhs_correction = 0.0;
            let mut capacity_sum = 0.0;
            let mut capacity_correction = 0.0;
            for other in lane..end {
                if !active[other] || parent[other] != root {
                    continue;
                }
                let rhs_value = raw_rhs[other] - rhs_correction;
                let rhs_next = add(rhs_sum, rhs_value);
                rhs_correction = (rhs_next - rhs_sum) - rhs_value;
                rhs_sum = rhs_next;
                let capacity_value = cell_open_volume(graph, fields, other) - capacity_correction;
                let capacity_next = add(capacity_sum, capacity_value);
                capacity_correction = (capacity_next - capacity_sum) - capacity_value;
                capacity_sum = capacity_next;
            }
            component_rhs[root] = add(component_rhs[root], rhs_sum);
            component_capacity[root] = add(component_capacity[root], capacity_sum);
        }
    }
    let mut component_open = vec![false; n];
    for face in &graph.subfaces {
        if !correctable[face.id as usize] {
            continue;
        }
        let negative = (face.negative_cell >= 0).then_some(face.negative_cell as usize);
        let positive = (face.positive_cell >= 0).then_some(face.positive_cell as usize);
        let active_cell = match (negative, positive) {
            (Some(a), Some(b)) if active[a] != active[b] => Some(if active[a] { a } else { b }),
            (Some(a), None) if active[a] => Some(a),
            (None, Some(b)) if active[b] => Some(b),
            _ => None,
        };
        if let Some(cell) = active_cell {
            component_open[parent[cell]] = true;
        }
    }
    for root in 0..n {
        if parent[root] != root || component_capacity[root] == 0.0 || component_open[root] {
            continue;
        }
        let defect = mul(component_rhs[root].abs(), fields.frame_dt);
        let limit = mul(NORMALIZED_TARGET, component_capacity[root]);
        if !(defect <= limit) {
            let mut interface_rate = 0.0f64;
            let mut exterior_rate = 0.0f64;
            let mut capacity_rate = 0.0f64;
            let mut source_rate = 0.0f64;
            let mut interface_faces = 0usize;
            for cell in 0..n {
                if !active[cell] || parent[cell] != root {
                    continue;
                }
                capacity_rate += fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
                    * graph.cells[cell].measure as f64;
                source_rate += fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64;
                for entry in incidences(graph, cell, &fallback) {
                    let face = &graph.subfaces[entry.subface_id as usize];
                    let other = if face.negative_cell == cell as i32 {
                        face.positive_cell
                    } else {
                        face.negative_cell
                    };
                    let contribution = cell_delta(
                        base_rate(graph, fields, face.id as usize),
                        entry.orientation,
                    ) as f64;
                    if other < 0 {
                        exterior_rate += contribution;
                    } else if fields.pressure_member[other as usize] != 0 {
                        interface_rate += contribution;
                        interface_faces += 1;
                    }
                }
            }
            return Err(ValidationError(format!(
                "geometric compatibility component {root} defect {defect} exceeds {limit}; rhs {}, capacity {}, members {}, interfaceRate {interface_rate}, interfaceFaces {interface_faces}, exteriorRate {exterior_rate}, capacityRate {capacity_rate}, sourceRate {source_rate}",
                component_rhs[root],
                component_capacity[root],
                (0..n).filter(|&i| active[i] && parent[i] == root).count()
            )));
        }
    }
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        if active[i] {
            let root = parent[i];
            rhs[i] = if component_open[root] {
                raw_rhs[i]
            } else {
                raw_rhs[i]
                    - mul(
                        div(component_rhs[root], component_capacity[root]),
                        cell_open_volume(graph, fields, i),
                    )
            };
        }
    }

    let mut diagonal = vec![0.0; n];
    let mut neighbors = vec![Vec::<(usize, f32)>::new(); n];
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        for entry in incidences(graph, cell, &fallback) {
            let face = entry.subface_id as usize;
            if !correctable[face] {
                continue;
            }
            let sf = &graph.subfaces[face];
            let row = &graph.rows[sf.row_id as usize];
            let distance = face_distance[face];
            let weight = div(mul(sf.measure, row.open_fraction), distance);
            diagonal[cell] = add(diagonal[cell], weight);
            let other = if sf.negative_cell == cell as i32 {
                sf.positive_cell
            } else {
                sf.negative_cell
            };
            if other >= 0 {
                neighbors[cell].push((other as usize, weight));
            }
        }
    }
    let apply = |input: &[f32], output: &mut [f32]| {
        for cell in 0..n {
            let mut value = 0.0;
            if active[cell] {
                for entry in incidences(graph, cell, &fallback) {
                    let face = entry.subface_id as usize;
                    if !correctable[face] {
                        continue;
                    }
                    let sf = &graph.subfaces[face];
                    let row = &graph.rows[sf.row_id as usize];
                    let negative = if sf.negative_cell >= 0 {
                        input[sf.negative_cell as usize]
                    } else {
                        0.0
                    };
                    let positive = if sf.positive_cell >= 0 {
                        input[sf.positive_cell as usize]
                    } else {
                        0.0
                    };
                    let correction = mul(
                        row.open_fraction,
                        div(positive - negative, face_distance[face]),
                    );
                    value = add(
                        value,
                        cell_delta(mul(sf.measure, correction), entry.orientation),
                    );
                }
            }
            output[cell] = value;
        }
    };

    let initial_pairs: Vec<_> = (0..n)
        .map(|i| {
            (
                0.0,
                if active[i] {
                    normalized(graph, fields, i, rhs[i])
                } else {
                    0.0
                },
            )
        })
        .collect();
    let initial_max = reduce_sum_max(&initial_pairs).1;
    // Symmetric Gauss-Seidel is an SPD incomplete factorization of the
    // component Laplacian. It preserves the exact operator and acceptance
    // criterion while resolving the low modes that Jacobi leaves after 64
    // iterations on long adaptive components.
    let precondition = |residual: &[f32], z: &mut [f32]| {
        for i in 0..n {
            if !active[i] || diagonal[i] <= 0.0 {
                z[i] = 0.0;
                continue;
            }
            let mut value = residual[i];
            for &(j, weight) in &neighbors[i] {
                if j < i {
                    value = add(value, mul(weight, z[j]));
                }
            }
            z[i] = div(value, diagonal[i]);
        }
        for i in (0..n).rev() {
            if !active[i] || diagonal[i] <= 0.0 {
                continue;
            }
            let mut value = z[i];
            for &(j, weight) in &neighbors[i] {
                if j > i {
                    value = add(value, div(mul(weight, z[j]), diagonal[i]));
                }
            }
            z[i] = value;
        }
    };
    let mut x = vec![0.0; n];
    let mut residual = rhs.clone();
    let mut z = vec![0.0; n];
    precondition(&residual, &mut z);
    let mut direction = z.clone();
    let mut image = vec![0.0; n];
    let mut gamma_values = vec![(0.0, 0.0); n];
    for i in 0..n {
        if active[i] {
            gamma_values[i].0 = mul(residual[i], z[i]);
        }
    }
    let mut gamma = reduce_sum_max(&gamma_values).0;
    let mut recursive_max = initial_max;
    let mut iterations = 0u32;
    let f32_iteration_budget = maximum_iterations.min(64) as u32;
    while recursive_max > NORMALIZED_TARGET && iterations < f32_iteration_budget {
        apply(&direction, &mut image);
        let curvature_values: Vec<_> = (0..n)
            .map(|i| {
                (
                    if active[i] {
                        mul(direction[i], image[i])
                    } else {
                        0.0
                    },
                    0.0,
                )
            })
            .collect();
        let curvature = reduce_sum_max(&curvature_values).0;
        if !(curvature > 0.0 && curvature.is_finite()) {
            return Err(ValidationError(format!(
                "geometric compatibility curvature {curvature}"
            )));
        }
        let alpha = div(gamma, curvature);
        let previous_gamma = gamma;
        for i in 0..n {
            gamma_values[i] = (0.0, 0.0);
            if active[i] {
                x[i] = add(x[i], mul(alpha, direction[i]));
                residual[i] = residual[i] - mul(alpha, image[i]);
                gamma_values[i].1 = normalized(graph, fields, i, residual[i]);
            }
        }
        precondition(&residual, &mut z);
        for i in 0..n {
            if active[i] {
                gamma_values[i].0 = mul(residual[i], z[i]);
            }
        }
        let reduced = reduce_sum_max(&gamma_values);
        gamma = reduced.0;
        recursive_max = reduced.1;
        iterations += 1;
        // Long, low-frequency air components reach the f32 recursive-residual
        // floor. Every production-sized 64-step chunk therefore replaces the
        // recurrence with the explicitly applied residual and restarts PCG.
        // Acceptance still uses the unmodified physical residual below.
        if iterations % 64 == 0 || recursive_max <= NORMALIZED_TARGET {
            apply(&x, &mut image);
            let mut measured = 0.0f32;
            for i in 0..n {
                gamma_values[i] = (0.0, 0.0);
                if active[i] {
                    residual[i] = rhs[i] - image[i];
                    measured = measured.max(normalized(graph, fields, i, raw_rhs[i] - image[i]));
                    gamma_values[i].1 = normalized(graph, fields, i, residual[i]);
                }
            }
            precondition(&residual, &mut z);
            for i in 0..n {
                direction[i] = z[i];
                if active[i] {
                    gamma_values[i].0 = mul(residual[i], z[i]);
                }
            }
            let restarted = reduce_sum_max(&gamma_values);
            gamma = restarted.0;
            recursive_max = restarted.1;
            if measured <= NORMALIZED_TARGET {
                break;
            }
            continue;
        }
        if !(previous_gamma > 0.0) {
            return Err(ValidationError(
                "geometric compatibility non-positive gamma".into(),
            ));
        }
        let beta = div(gamma, previous_gamma);
        for i in 0..n {
            if active[i] {
                direction[i] = add(z[i], mul(beta, direction[i]));
            }
        }
    }
    apply(&x, &mut image);
    let mut high_precision_correction: Option<Vec<f32>> = None;
    let mut publication_rounding = vec![0.0f32; nf];
    if initial_max > NORMALIZED_TARGET {
        // Native CPU execution can retain a higher precision Krylov image and
        // round only the published correction. This resolves the recursive
        // f32 floor without changing the physical operator or acceptance test.
        let mut rhs64: Vec<f64> = rhs.iter().map(|&v| v as f64).collect();
        for component in 0..n {
            if !active[component] || parent[component] != component || component_open[component] {
                continue;
            }
            let members: Vec<_> = (0..n)
                .filter(|&i| active[i] && parent[i] == component)
                .collect();
            let sum: f64 = members.iter().map(|&i| rhs64[i]).sum();
            let capacity: f64 = members
                .iter()
                .map(|&i| cell_open_volume(graph, fields, i) as f64)
                .sum();
            for &i in &members {
                rhs64[i] -= sum * cell_open_volume(graph, fields, i) as f64 / capacity;
            }
        }
        let mut diagonal64 = vec![0.0f64; n];
        for cell in 0..n {
            if !active[cell] {
                continue;
            }
            for entry in incidences(graph, cell, &fallback) {
                let face = entry.subface_id as usize;
                if correctable[face] {
                    let sf = &graph.subfaces[face];
                    let row = &graph.rows[sf.row_id as usize];
                    diagonal64[cell] +=
                        sf.measure as f64 * row.open_fraction as f64 / face_distance[face] as f64;
                }
            }
        }
        let mut x64 = vec![0.0f64; n];
        let mut r64 = rhs64.clone();
        let mut z64 = vec![0.0f64; n];
        let precondition64 = |residual: &[f64], z: &mut [f64]| {
            for i in 0..n {
                if !active[i] || diagonal64[i] <= 0.0 {
                    z[i] = 0.0;
                    continue;
                }
                let mut value = residual[i];
                for entry in incidences(graph, i, &fallback) {
                    let face = entry.subface_id as usize;
                    if !correctable[face] {
                        continue;
                    }
                    let sf = &graph.subfaces[face];
                    let other = if sf.negative_cell == i as i32 {
                        sf.positive_cell
                    } else {
                        sf.negative_cell
                    };
                    if other >= 0 && (other as usize) < i {
                        let row = &graph.rows[sf.row_id as usize];
                        let weight = sf.measure as f64 * row.open_fraction as f64
                            / face_distance[face] as f64;
                        value += weight * z[other as usize];
                    }
                }
                z[i] = value / diagonal64[i];
            }
            for i in (0..n).rev() {
                if !active[i] || diagonal64[i] <= 0.0 {
                    continue;
                }
                for entry in incidences(graph, i, &fallback) {
                    let face = entry.subface_id as usize;
                    if !correctable[face] {
                        continue;
                    }
                    let sf = &graph.subfaces[face];
                    let other = if sf.negative_cell == i as i32 {
                        sf.positive_cell
                    } else {
                        sf.negative_cell
                    };
                    if other > i as i32 {
                        let row = &graph.rows[sf.row_id as usize];
                        let weight = sf.measure as f64 * row.open_fraction as f64
                            / face_distance[face] as f64;
                        z[i] += weight * z[other as usize] / diagonal64[i];
                    }
                }
            }
        };
        let apply64 = |input: &[f64], output: &mut [f64]| {
            for cell in 0..n {
                let mut value = 0.0f64;
                if active[cell] {
                    for entry in incidences(graph, cell, &fallback) {
                        let face = entry.subface_id as usize;
                        if !correctable[face] {
                            continue;
                        }
                        let sf = &graph.subfaces[face];
                        let row = &graph.rows[sf.row_id as usize];
                        let negative = if sf.negative_cell >= 0 {
                            input[sf.negative_cell as usize]
                        } else {
                            0.0
                        };
                        let positive = if sf.positive_cell >= 0 {
                            input[sf.positive_cell as usize]
                        } else {
                            0.0
                        };
                        let rate =
                            sf.measure as f64 * row.open_fraction as f64 * (positive - negative)
                                / face_distance[face] as f64;
                        value += if entry.orientation < 0 { -rate } else { rate };
                    }
                }
                output[cell] = value;
            }
        };
        precondition64(&r64, &mut z64);
        let mut p64 = z64.clone();
        let mut ap64 = vec![0.0f64; n];
        let mut gamma64: f64 = (0..n).filter(|&i| active[i]).map(|i| r64[i] * z64[i]).sum();
        let remaining = maximum_iterations.saturating_sub(iterations as usize);
        for _ in 0..remaining {
            apply64(&p64, &mut ap64);
            let curvature: f64 = (0..n)
                .filter(|&i| active[i])
                .map(|i| p64[i] * ap64[i])
                .sum();
            if !(curvature > 0.0 && curvature.is_finite()) {
                break;
            }
            let alpha = gamma64 / curvature;
            for i in 0..n {
                if active[i] {
                    x64[i] += alpha * p64[i];
                    r64[i] -= alpha * ap64[i];
                }
            }
            iterations += 1;
            let maximum = (0..n)
                .filter(|&i| active[i])
                .map(|i| {
                    r64[i].abs() * fields.frame_dt as f64
                        / cell_open_volume(graph, fields, i) as f64
                })
                .fold(0.0f64, f64::max);
            if maximum <= NORMALIZED_TARGET as f64 * F64_KRYLOV_TARGET_FRACTION as f64 {
                break;
            }
            precondition64(&r64, &mut z64);
            let next_gamma: f64 = (0..n).filter(|&i| active[i]).map(|i| r64[i] * z64[i]).sum();
            let beta = next_gamma / gamma64;
            gamma64 = next_gamma;
            for i in 0..n {
                if active[i] {
                    p64[i] = z64[i] + beta * p64[i];
                }
            }
        }
        for i in 0..n {
            x[i] = x64[i] as f32;
        }
        let mut published = vec![0.0f32; nf];
        for face in &graph.subfaces {
            let i = face.id as usize;
            if !correctable[i] {
                continue;
            }
            let row = &graph.rows[face.row_id as usize];
            let negative = if face.negative_cell >= 0 {
                x64[face.negative_cell as usize]
            } else {
                0.0
            };
            let positive = if face.positive_cell >= 0 {
                x64[face.positive_cell as usize]
            } else {
                0.0
            };
            let exact = row.open_fraction as f64 * (positive - negative) / face_distance[i] as f64;
            published[i] = exact as f32;
            publication_rounding[i] =
                (face.measure as f64 * (exact - published[i] as f64).abs()) as f32;
        }
        high_precision_correction = Some(published);
        apply(&x, &mut image);
    }
    // Round-to-f32 can reintroduce a few ulps on high dynamic range cells.
    // Symmetric residual refinement keeps the correction in the gradient
    // space before the separately bounded conservative publication cleanup.
    let polish_budget = maximum_iterations
        .saturating_sub(iterations as usize)
        .min(32);
    for _ in 0..polish_budget {
        apply(&x, &mut image);
        let mut maximum = 0.0f32;
        for i in 0..n {
            residual[i] = if active[i] { rhs[i] - image[i] } else { 0.0 };
            if active[i] {
                maximum = maximum.max(normalized(graph, fields, i, residual[i]));
            }
        }
        if maximum <= 0.5 * VOLUME_ROUNDOFF_RATIO {
            break;
        }
        precondition(&residual, &mut z);
        for i in 0..n {
            if active[i] {
                x[i] = add(x[i], z[i]);
            }
        }
        iterations += 1;
    }
    let mut maximum_volume_error = 0.0f32;
    for face in &graph.subfaces {
        let i = face.id as usize;
        if !correctable[i] {
            continue;
        }
        let row = &graph.rows[face.row_id as usize];
        let negative = if face.negative_cell >= 0 {
            x[face.negative_cell as usize]
        } else {
            0.0
        };
        let positive = if face.positive_cell >= 0 {
            x[face.positive_cell as usize]
        } else {
            0.0
        };
        fields.subface_velocity_correction[i] = high_precision_correction
            .as_ref()
            .map(|values| values[i])
            .unwrap_or_else(|| {
                mul(
                    row.open_fraction,
                    div(positive - negative, face_distance[i]),
                )
            });
        if high_precision_correction.is_none() {
            let exact = row.open_fraction as f64 * (positive as f64 - negative as f64)
                / face_distance[i] as f64;
            publication_rounding[i] = (face.measure as f64
                * (exact - fields.subface_velocity_correction[i] as f64).abs())
                as f32;
        }
    }
    fields.subface_compatibility_rate.clone_from(&starting_rate);
    for face in &graph.subfaces {
        let i = face.id as usize;
        if correctable[i] {
            fields.subface_compatibility_rate[i] -=
                face.measure as f64 * fields.subface_velocity_correction[i] as f64;
        }
    }
    let evaluate_cell = |fields: &Fields, cell: usize| {
        if !active[cell] {
            return 0.0;
        }
        let mut value = 0.0f64;
        for entry in incidences(graph, cell, &fallback) {
            let face = entry.subface_id as usize;
            let rate = fields.subface_compatibility_rate[face];
            value += if entry.orientation < 0 { -rate } else { rate };
        }
        let capacity_rate = fields.capacity_rate.get(cell).copied().unwrap_or(0.0) as f64
            * graph.cells[cell].measure as f64;
        (value - capacity_rate + fields.source_rate.get(cell).copied().unwrap_or(0.0) as f64) as f32
    };
    let evaluate =
        |fields: &Fields| -> Vec<f32> { (0..n).map(|cell| evaluate_cell(fields, cell)).collect() };
    let mut tree = vec![None::<(usize, usize, i8)>; n];
    let mut order = Vec::new();
    let mut seen = vec![false; n];
    let mut open_face = vec![None::<(usize, i8)>; n];
    for face in &graph.subfaces {
        if !correctable[face.id as usize] {
            continue;
        }
        let negative = (face.negative_cell >= 0).then_some(face.negative_cell as usize);
        let positive = (face.positive_cell >= 0).then_some(face.positive_cell as usize);
        let (cell, orientation) = match (negative, positive) {
            (Some(a), Some(b)) if active[a] != active[b] => {
                if active[a] {
                    (a, -1)
                } else {
                    (b, 1)
                }
            }
            (Some(a), None) if active[a] => (a, -1),
            (None, Some(b)) if active[b] => (b, 1),
            _ => continue,
        };
        let component = parent[cell];
        let candidate = (face.id as usize, orientation);
        if open_face[component].is_none() {
            open_face[component] = Some(candidate);
        }
    }
    for component in 0..n {
        if !active[component] || parent[component] != component {
            continue;
        }
        let root = if let Some((face, _)) = open_face[component] {
            let sf = &graph.subfaces[face];
            if sf.negative_cell >= 0 {
                sf.negative_cell as usize
            } else {
                sf.positive_cell as usize
            }
        } else {
            (0..n)
                .filter(|&i| active[i] && parent[i] == component)
                .max_by(|&a, &b| {
                    cell_open_volume(graph, fields, a)
                        .total_cmp(&cell_open_volume(graph, fields, b))
                })
                .unwrap()
        };
        seen[root] = true;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(cell) = queue.pop_front() {
            order.push(cell);
            for entry in incidences(graph, cell, &fallback) {
                let face = entry.subface_id as usize;
                if !correctable[face] {
                    continue;
                }
                let sf = &graph.subfaces[face];
                let other = if sf.negative_cell == cell as i32 {
                    sf.positive_cell
                } else {
                    sf.negative_cell
                };
                if other >= 0 && active[other as usize] && !seen[other as usize] {
                    seen[other as usize] = true;
                    let orientation = if sf.negative_cell == other { -1 } else { 1 };
                    tree[other as usize] = Some((cell, face, orientation));
                    queue.push_back(other as usize);
                }
            }
        }
    }
    let mut projected = evaluate(fields);
    let pre_postconditioning_maximum = (0..n)
        .filter(|&i| active[i])
        .map(|i| normalized(graph, fields, i, projected[i]))
        .fold(0.0f32, f32::max);
    let mut pre_postconditioning_bound_maximum = 0.0f32;
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        let mut magnitude = 0.0f32;
        let entries = incidences(graph, cell, &fallback);
        for entry in entries {
            let face = entry.subface_id as usize;
            let correction_rate = mul(
                graph.subfaces[face].measure,
                fields.subface_velocity_correction[face],
            );
            magnitude = add(magnitude, base_rate(graph, fields, face).abs());
            magnitude = add(magnitude, correction_rate.abs());
        }
        let capacity_rate = mul(
            fields.capacity_rate.get(cell).copied().unwrap_or(0.0),
            graph.cells[cell].measure,
        );
        magnitude = add(magnitude, capacity_rate.abs());
        magnitude = add(
            magnitude,
            fields.source_rate.get(cell).copied().unwrap_or(0.0).abs(),
        );
        // Higham's gamma_n forward-error bound for the exact ordered f32
        // expression: base/correction construction, signed incidence gather,
        // capacity subtraction, and source addition. The retained f64 Krylov
        // termination allowance is added separately.
        // Per incidence: wall blend (two), base area product, correction area
        // product, subtraction, orientation, and ordered accumulation. Include
        // one guard operation for each input rounding plus the capacity/source
        // tail; this is conservative for both static and cut rows.
        let operations = (8 * entries.len() + 8) as f32;
        let gamma_n = div(
            mul(operations, f32::EPSILON),
            1.0 - mul(operations, f32::EPSILON),
        );
        let capacity = cell_open_volume(graph, fields, cell);
        let krylov_allowance = if fields.frame_dt > 0.0 {
            div(
                mul(F64_KRYLOV_TARGET_FRACTION * NORMALIZED_TARGET, capacity),
                fields.frame_dt,
            )
        } else {
            0.0
        };
        let publication_bound = entries.iter().fold(0.0f32, |sum, entry| {
            add(sum, publication_rounding[entry.subface_id as usize])
        });
        let absolute_bound = add(
            // RHS construction, closed-component compatibility projection,
            // and the final projected audit are three independently rounded
            // ordered evaluations of the same physical-rate expression.
            add(mul(3.0 * gamma_n, magnitude), publication_bound),
            krylov_allowance,
        );
        let normalized_bound = if fields.frame_dt > 0.0 {
            div(mul(absolute_bound, fields.frame_dt), capacity)
        } else {
            absolute_bound
        };
        pre_postconditioning_bound_maximum =
            pre_postconditioning_bound_maximum.max(normalized_bound);
        if projected[cell].abs() > absolute_bound {
            return Err(ValidationError(format!(
                "geometric compatibility solver residual cell {cell} {} exceeds derived roundoff bound {absolute_bound}",
                projected[cell].abs()
            )));
        }
    }
    let mut residual_correction_maximum = 0.0f32;
    let mut postconditioning_rounds = 0u32;
    for _ in 0..8 {
        let maximum = (0..n)
            .filter(|&i| active[i])
            .map(|i| normalized(graph, fields, i, projected[i]))
            .fold(0.0f32, f32::max);
        if maximum <= NORMALIZED_TARGET {
            break;
        }
        // Internal face corrections conserve each component's summed rate.
        // The minimax feasible target distributes a retained component residue
        // in proportion to open volume, giving every member the same normalized
        // error. Open components first discharge their sum through representable
        // boundary corrections; a sub-ulp remainder is then retained this way.
        let mut component_residual = vec![0.0f64; n];
        let mut component_open_volume = vec![0.0f64; n];
        for cell in 0..n {
            if active[cell] {
                let component = parent[cell];
                component_residual[component] += projected[cell] as f64;
                component_open_volume[component] += cell_open_volume(graph, fields, cell) as f64;
            }
        }
        let mut target = vec![0.0f32; n];
        let mut retain_component_residual = vec![false; n];
        for component in 0..n {
            if !active[component] || parent[component] != component {
                continue;
            }
            let normalized_component = if component_open_volume[component] > 0.0 {
                (component_residual[component].abs() * fields.frame_dt as f64
                    / component_open_volume[component]) as f32
            } else {
                f32::MAX
            };
            if !component_open[component] && normalized_component > NORMALIZED_TARGET {
                return Err(ValidationError(format!(
                    "geometric compatibility closed component {component} residual cannot fit the physical tolerance"
                )));
            }
            // Once an open boundary correction reaches its nearest f32 value,
            // its remaining component sum may be smaller than one correction
            // ulp. It is already within the physical per-volume tolerance and
            // can be represented by conservative internal tree transfers.
            retain_component_residual[component] =
                !component_open[component] || normalized_component <= NORMALIZED_TARGET;
        }
        let mut target_sum = vec![0.0f64; n];
        for cell in 0..n {
            if !active[cell] {
                continue;
            }
            let component = parent[cell];
            if retain_component_residual[component] {
                target[cell] = (component_residual[component]
                    * cell_open_volume(graph, fields, cell) as f64
                    / component_open_volume[component]) as f32;
                target_sum[component] += target[cell] as f64;
            }
        }
        // Restore the f32 target sum after individually rounding the minimax
        // targets. The largest-capacity cell has the most normalized slack;
        // stable cell order resolves equal capacities.
        for component in 0..n {
            if !retain_component_residual[component] {
                continue;
            }
            let remainder = (component_residual[component] - target_sum[component]) as f32;
            let owner = (0..n)
                .filter(|&cell| active[cell] && parent[cell] == component)
                .max_by(|&a, &b| {
                    cell_open_volume(graph, fields, a)
                        .total_cmp(&cell_open_volume(graph, fields, b))
                        .then_with(|| b.cmp(&a))
                })
                .unwrap();
            target[owner] = add(target[owner], remainder);
            if normalized(graph, fields, owner, target[owner]) > NORMALIZED_TARGET {
                return Err(ValidationError(format!(
                    "geometric compatibility component {component} rounded target exceeds the physical tolerance"
                )));
            }
        }
        for &cell in order.iter().rev() {
            let Some((parent_cell, face, orientation)) = tree[cell] else {
                continue;
            };
            let excess = projected[cell] - target[cell];
            let rate_delta = if orientation < 0 { excess } else { -excess };
            residual_correction_maximum = residual_correction_maximum.max(rate_delta.abs());
            fields.subface_compatibility_rate[face] += rate_delta as f64;
            projected[parent_cell] = add(projected[parent_cell], excess);
            projected[cell] = target[cell];
        }
        projected = evaluate(fields);
        for component in 0..n {
            if retain_component_residual[component] {
                continue;
            }
            if let Some((face, orientation)) = open_face[component] {
                let sf = &graph.subfaces[face];
                let cell = if sf.negative_cell >= 0 {
                    sf.negative_cell as usize
                } else {
                    sf.positive_cell as usize
                };
                let rate_delta = if orientation < 0 {
                    projected[cell]
                } else {
                    -projected[cell]
                };
                residual_correction_maximum = residual_correction_maximum.max(rate_delta.abs());
                fields.subface_compatibility_rate[face] += rate_delta as f64;
            }
        }
        projected = evaluate(fields);
        iterations += 1;
        postconditioning_rounds += 1;
    }
    let (measured_cell, measured_max) = (0..n)
        .filter(|&i| active[i])
        .map(|i| (i, normalized(graph, fields, i, projected[i])))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    if measured_max > NORMALIZED_TARGET {
        return Err(ValidationError(format!(
            "geometric compatibility normalized residual {measured_max} exceeds {NORMALIZED_TARGET} at cell {measured_cell}, component {}, open {}, rate {}",
            parent[measured_cell], component_open[parent[measured_cell]], projected[measured_cell]
        )));
    }
    for cell in 0..n {
        if !active[cell] {
            continue;
        }
        let error = mul(projected[cell].abs(), fields.frame_dt);
        let limit = mul(VOLUME_ROUNDOFF_RATIO, cell_open_volume(graph, fields, cell));
        if !(error <= limit) {
            return Err(ValidationError(format!(
                "geometric compatibility cell {cell} volume error {error} exceeds {limit}"
            )));
        }
        maximum_volume_error = maximum_volume_error.max(error);
    }
    // Preserve the correction plane for diagnostics. Transport consumes the
    // directly published compatible rate above, so this derived f32 velocity
    // need not reproduce a small rate by subtracting two large operands.
    for face in &graph.subfaces {
        let i = face.id as usize;
        let row = &graph.rows[face.row_id as usize];
        let mut base_velocity = fields.face_velocity[row.id as usize];
        if row.open_fraction < 1.0 || row.solid_velocity != 0.0 {
            base_velocity = base_velocity - mul(1.0 - row.open_fraction, row.solid_velocity);
        }
        fields.subface_velocity_correction[i] =
            base_velocity - div(fields.subface_compatibility_rate[i] as f32, face.measure);
    }
    Ok(GeometricCompatibilityReceipt3d {
        accepted: true,
        skipped_for_solid_motion: false,
        iterations,
        initial_normalized_residual: initial_max,
        recursive_normalized_residual: recursive_max,
        measured_normalized_residual: measured_max,
        maximum_projected_volume_error: maximum_volume_error,
        pre_postconditioning_maximum,
        pre_postconditioning_bound_maximum,
        residual_correction_maximum,
        postconditioning_rounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Cell, Row, RowKind, Subface};

    fn run_default_cpu_water_box(frames: u32) {
        use crate::initial_scene::SceneDocument;
        use crate::production_scene::ProductionSceneOptions;
        use crate::world::WorldOptions;
        use crate::world3d::World3d;

        let scene: SceneDocument = serde_json::from_str(include_str!(
            "../../../core/testdata/water-box-body-free-world3d-scene.json"
        ))
        .unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../core/testdata/water-box-ui-default-cpu-world3d-options.json"
        ))
        .unwrap();
        let mut production = ProductionSceneOptions::default();
        let mut options: WorldOptions = serde_json::from_value(fixture["options"].clone()).unwrap();
        crate::runtime_options3d::apply_initial_values(
            &mut production,
            &mut options,
            &fixture["options"]["methodValues"],
        )
        .unwrap();
        let mut world = World3d::from_document(scene, production, options).unwrap();
        for sequence in 1..=frames {
            if let Err(error) = world.advance(sequence, 1.0 / 30.0) {
                panic!(
                    "frame {sequence}: {error}; primary {:?}; refinement {:?}",
                    world.receipt().pressure,
                    world.receipt().primary_refinement
                );
            }
            assert!(
                world.receipt().compatibility.measured_normalized_residual <= NORMALIZED_TARGET
            );
        }
    }

    #[test]
    fn default_cpu_water_box_remains_compatible_across_twelve_frames() {
        run_default_cpu_water_box(12);
    }

    #[test]
    fn default_cpu_water_box_remains_compatible_for_thirty_frames() {
        run_default_cpu_water_box(30);
    }

    fn line_fixture() -> (Graph, Fields) {
        let cells = (0..3)
            .map(|i| Cell {
                id: i,
                minimum: [i as f32, 0.0, 0.0],
                maximum: [i as f32 + 1.0, 1.0, 1.0],
                center: [i as f32 + 0.5, 0.5, 0.5],
                widths: [1.0; 3],
                measure: 1.0,
                ..Default::default()
            })
            .collect();
        let rows = (0..2)
            .map(|i| Row {
                id: i,
                kind: RowKind::IntraBrick,
                axis: 0,
                center: [i as f32 + 1.0, 0.5, 0.5],
                measure: 1.0,
                distance: 1.0,
                dual_weight: 1.0,
                open_fraction: 1.0,
                ..Default::default()
            })
            .collect();
        let subfaces = (0..2)
            .map(|i| Subface {
                id: i,
                row_id: i,
                axis: 0,
                center: [i as f32 + 1.0, 0.5, 0.5],
                measure: 1.0,
                negative_cell: i as i32,
                positive_cell: i as i32 + 1,
                aperture: 1.0,
                ..Default::default()
            })
            .collect();
        let graph = Graph {
            dimension: 3,
            dimensions: [3.0, 1.0, 1.0],
            cells,
            rows,
            subfaces,
            incidences: vec![vec![0], vec![0, 1], vec![1]],
            subface_incidences: vec![
                vec![SubfaceIncidence {
                    subface_id: 0,
                    orientation: -1,
                }],
                vec![
                    SubfaceIncidence {
                        subface_id: 0,
                        orientation: 1,
                    },
                    SubfaceIncidence {
                        subface_id: 1,
                        orientation: -1,
                    },
                ],
                vec![SubfaceIncidence {
                    subface_id: 1,
                    orientation: 1,
                }],
            ],
            ..Default::default()
        };
        let fields = Fields {
            density: vec![0.0; 3],
            capacity: vec![1.0; 3],
            frame_dt: 0.01,
            face_velocity: vec![1.0, 1.0],
            pressure_member: vec![0; 3],
            ..Default::default()
        };
        (graph, fields)
    }

    #[test]
    fn compatible_air_component_projects_each_physical_subface() {
        let (graph, mut fields) = line_fixture();
        let receipt = solve_geometric_transport_compatibility_3d(&graph, &mut fields, 64).unwrap();
        assert!(receipt.accepted);
        assert!(receipt.iterations <= 64);
        assert!(receipt.measured_normalized_residual <= NORMALIZED_TARGET);
        assert!(crate::transport::physical_subface_rate(&graph, &fields, 0).abs() <= f32::EPSILON);
        assert!(crate::transport::physical_subface_rate(&graph, &fields, 1).abs() <= f32::EPSILON);
    }

    #[test]
    fn open_boundary_anchors_nonzero_component_rate() {
        let (mut graph, mut fields) = line_fixture();
        graph.subfaces.push(Subface {
            id: 2,
            row_id: 0,
            axis: 0,
            center: [0.0, 0.5, 0.5],
            measure: 1.0,
            negative_cell: -1,
            positive_cell: 0,
            aperture: 1.0,
            ..Default::default()
        });
        graph.subface_incidences[0].insert(
            0,
            SubfaceIncidence {
                subface_id: 2,
                orientation: 1,
            },
        );
        let receipt = solve_geometric_transport_compatibility_3d(&graph, &mut fields, 256).unwrap();
        assert!(receipt.accepted);
        assert!(fields.subface_velocity_correction[2].is_finite());
    }

    #[test]
    fn secondary_projection_preserves_primary_liquid_adjacent_rates() {
        let (mut graph, mut fields) = line_fixture();
        fields.pressure_member[0] = 1;
        fields.face_velocity = vec![0.75, 0.0, 0.0];
        graph.rows.push(Row {
            id: 2,
            kind: RowKind::SparseAir,
            axis: 0,
            center: [3.0, 0.5, 0.5],
            measure: 1.0,
            distance: 0.5,
            dual_weight: 0.5,
            open_fraction: 1.0,
            ..Default::default()
        });
        graph.subfaces.push(Subface {
            id: 2,
            row_id: 2,
            axis: 0,
            center: [3.0, 0.5, 0.5],
            measure: 1.0,
            negative_cell: 2,
            positive_cell: -1,
            aperture: 1.0,
            ..Default::default()
        });
        graph.subface_incidences[2].push(SubfaceIncidence {
            subface_id: 2,
            orientation: -1,
        });
        let primary_rate = base_rate(&graph, &fields, 0);

        let receipt = solve_geometric_transport_compatibility_3d(&graph, &mut fields, 256).unwrap();

        assert!(receipt.accepted);
        assert_eq!(
            fields.subface_compatibility_rate[0].to_bits(),
            (primary_rate as f64).to_bits()
        );
        assert_eq!(
            fields.subface_velocity_correction[0].to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn postconditioner_rejects_an_unconverged_large_residual() {
        let (graph, mut fields) = line_fixture();
        let error = solve_geometric_transport_compatibility_3d(&graph, &mut fields, 0).unwrap_err();
        assert!(error.0.contains("exceeds derived roundoff bound"));
    }
}
