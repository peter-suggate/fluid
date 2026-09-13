//! Measured gates for the opt-in whole-step cellwise remap lab.
//!
//! These cases deliberately exercise the native solver topology and receipt;
//! they are not a second implementation of the remap.

use fluid_core::adaptive_remap::{
    transport_volume_cellwise_with_commit, CellwiseClosure, CellwiseRemapOptions,
};
use fluid_core::band_projection::{
    cleanup_pressure_receiver_rates_2d, project_receiver_band_rates_2d,
    BAND_PROJECTION_NORMALIZED_TARGET,
};
use fluid_core::geometry::BoundaryMode;
use fluid_core::initial_scene::SceneDocument;
use fluid_core::numerics::{
    prepare_pressure_topology, prepare_pressure_topology_with_swept_static_wall_support,
    FaceConsistentVelocity2d,
};
use fluid_core::production_scene::ProductionSceneOptions;
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::world::{CellwiseTransportMode, TransportExperiment, World, WorldOptions};
use fluid_core::world3d::World3d;
use fluid_core::{Fields, Graph};

fn brick(key: u32, coordinate: [i32; 3], resolution: u8) -> BrickSeed {
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

fn uniform_topology(bricks_x: i32, bricks_y: i32) -> Graph {
    let mut bricks = Vec::new();
    for y in 0..bricks_y {
        for x in 0..bricks_x {
            let key = (x + bricks_x * y) as u32;
            bricks.push(brick(key, [x, y, 0], 8));
        }
    }
    compile_topology::<2>(TopologySeed {
        dimensions: [(8 * bricks_x) as u32, (8 * bricks_y) as u32, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks,
    })
    .unwrap()
    .graph
}

fn seam_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 4), brick(1, [1, 0, 0], 8)],
    })
    .unwrap()
    .graph
}

fn sparse_left_brick_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 8)],
    })
    .unwrap()
    .graph
}

fn one_brick_topology(boundaries: [BoundaryMode; 6]) -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries,
        bricks: vec![brick(0, [0, 0, 0], 8)],
    })
    .unwrap()
    .graph
}

fn single_open_cell_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Open; 6],
        bricks: vec![brick(0, [0, 0, 0], 1)],
    })
    .unwrap()
    .graph
}

fn single_closed_cell_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 1)],
    })
    .unwrap()
    .graph
}

fn analytic_fields(graph: &Graph, velocity: impl Fn(f64, f64) -> [f64; 2]) -> Fields {
    let n = graph.cells.len();
    let r = graph.rows.len();
    let mut cell_velocity = Vec::with_capacity(2 * n);
    for cell in &graph.cells {
        let [u, v] = velocity(cell.center[0] as f64, cell.center[1] as f64);
        cell_velocity.extend([u as f32, v as f32]);
    }
    let face_velocity = graph
        .rows
        .iter()
        .map(|row| {
            if row.kind == fluid_core::RowKind::ClosedWorld {
                return 0.0;
            }
            let value = velocity(row.center[0] as f64, row.center[1] as f64);
            value[row.axis as usize] as f32
        })
        .collect();
    Fields {
        density: graph
            .cells
            .iter()
            .map(|cell| {
                let margin = cell.widths[0].max(cell.widths[1]);
                u8::from(
                    cell.center[0] >= margin
                        && cell.center[0] <= graph.dimensions[0] - margin
                        && cell.center[1] >= margin
                        && cell.center[1] <= graph.dimensions[1] - margin,
                ) as f32
            })
            .collect(),
        gamma: vec![1.0; n],
        capacity: vec![1.0; n],
        capacity_before: vec![1.0; n],
        capacity_after: vec![1.0; n],
        source_rate: vec![0.0; n],
        cell_velocity,
        face_velocity,
        pressure: vec![0.0; n],
        pressure_rhs: vec![0.0; n],
        pressure_diagonal: vec![1.0; n],
        pressure_member: vec![1; n],
        pressure_row_member: vec![1; r],
        extension_depth: vec![0; n],
        interface_normal: vec![0.0; 2 * n],
        interface_offset: vec![0.0; n],
        ..Fields::default()
    }
}

#[test]
fn pressure_rate_cleanup_uses_projection_operands_without_admitting_physical_residuals() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    fields.frame_dt = 1.0 / 30.0;
    let face = graph
        .subfaces
        .iter()
        .find(|face| face.negative_cell >= 0 && face.positive_cell >= 0)
        .expect("fixture has no internal face")
        .id as usize;
    let subface = &graph.subfaces[face];
    fields.pressure[subface.negative_cell as usize] = -50.0;
    fields.pressure[subface.positive_cell as usize] = 50.0;
    let receiver = vec![true; graph.cells.len()];

    let mut roundoff = vec![0.0; graph.subfaces.len()];
    roundoff[face] = 2.0e-5;
    let receipt = cleanup_pressure_receiver_rates_2d(
        &graph,
        &fields,
        &mut roundoff,
        &receiver,
        fields.frame_dt,
    )
    .expect("projection-scale roundoff should be cleanable");
    assert!(receipt.max_normalized_divergence_after <= BAND_PROJECTION_NORMALIZED_TARGET);

    let mut physical = vec![0.0; graph.subfaces.len()];
    physical[face] = 0.1;
    let error = cleanup_pressure_receiver_rates_2d(
        &graph,
        &fields,
        &mut physical,
        &receiver,
        fields.frame_dt,
    )
    .unwrap_err();
    assert!(
        error.0.contains("exceeds derived roundoff bound"),
        "unexpected rejection: {error}"
    );
}

fn probe(
    graph: &Graph,
    fields: &mut Fields,
    trace_segments: usize,
    edge_samples: usize,
) -> fluid_core::adaptive_remap::CellwiseRemapReceipt {
    transport_volume_cellwise_with_commit(
        graph,
        fields,
        1.0,
        CellwiseRemapOptions {
            edge_samples,
            trace_segments,
            closure: CellwiseClosure::None,
            commit_material: false,
        },
        |_, _| panic!("geometry-only probe must not invoke the material commit"),
    )
    .unwrap()
}

fn assert_geometry_certificate(receipt: &fluid_core::adaptive_remap::CellwiseRemapReceipt) {
    assert!(receipt.traces > 0);
    assert_eq!(
        receipt.rk_evaluations,
        4 * receipt.trace_segments * receipt.traces
    );
    assert!(receipt.area_balance_relative_error <= 1e-12, "{receipt:?}");
    assert!(receipt.max_area_identity_error <= 1e-10, "{receipt:?}");
    assert_eq!(
        receipt.pre_correction_liquid_receiver_folds, 0,
        "{receipt:?}"
    );
    assert_eq!(
        receipt.corrected_liquid_receiver_folds, 0,
        "{receipt:?}"
    );
}

#[test]
fn whole_step_uniform_translation_has_flat_work_across_courant() {
    let graph = uniform_topology(8, 2);
    let mut trace_work = None;
    for courant in [0.5, 2.0, 8.0, 20.0] {
        let mut fields = analytic_fields(&graph, |_, _| [courant, 0.0]);
        // Keep the transported liquid and its C=20 receiver band clear of the
        // closed exterior walls. Boundary-clipped dry polygons are diagnosed
        // separately by the amended M2b gate.
        for (density, cell) in fields.density.iter_mut().zip(&graph.cells) {
            *density = f32::from(
                cell.center[0] >= 22.0
                    && cell.center[0] <= 40.0
                    && cell.center[1] >= 4.0
                    && cell.center[1] <= 12.0,
            );
        }
        let density = fields.density.clone();
        let receipt = probe(&graph, &mut fields, 1, 1);
        assert_geometry_certificate(&receipt);
        assert_eq!(
            fields.density, density,
            "probe changed material at C={courant}"
        );
        assert!(receipt.max_courant_all >= courant - 1e-5, "{receipt:?}");
        assert!(receipt.max_courant_liquid >= courant - 1e-5, "{receipt:?}");
        assert!(receipt.receivers > 0, "{receipt:?}");
        assert_eq!(
            receipt.traces,
            *trace_work.get_or_insert(receipt.traces),
            "trace work changed with Courant"
        );
    }
}

#[test]
fn edge_samples_are_intervals_per_finest_unit() {
    let graph = seam_topology();
    let mut previous_traces = 0;
    let mut previous_chain_points = 0;
    for samples in [1, 2, 4] {
        let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
        let receipt = probe(&graph, &mut fields, 1, samples);
        assert_geometry_certificate(&receipt);
        assert!(receipt.traces > previous_traces);
        assert!(receipt.chain_points > previous_chain_points);
        previous_traces = receipt.traces;
        previous_chain_points = receipt.chain_points;
    }
}

#[test]
fn face_consistent_sampler_preserves_fields_and_matches_subface_fluxes() {
    let graph = seam_topology();
    let fields = analytic_fields(&graph, |x, y| {
        [0.12 * x - 0.03 * y, -0.07 * y + 0.02 * x]
    });
    let original = fields.clone();
    let sampler = FaceConsistentVelocity2d::new(&graph, &fields, 1.0).unwrap();
    assert_eq!(fields, original, "diagnostic extension mutated solver fields");
    assert!(sampler.extension_generations() >= 8);

    let mut checked_seam = false;
    for face in &graph.subfaces {
        if face.negative_cell < 0 || face.positive_cell < 0 {
            continue;
        }
        let axis = face.axis as usize;
        let expected = sampler.subface_rates()[face.id as usize] / face.measure as f64;
        let epsilon = 1.0e-6;
        for sign in [-1.0, 1.0] {
            let mut point = [face.center[0] as f64, face.center[1] as f64];
            point[axis] += sign * epsilon;
            let sampled = sampler.sample(&graph, point).unwrap()[axis];
            assert!(
                (sampled - expected).abs() <= 2.0e-5,
                "face {} side {sign}: sampled {sampled}, expected {expected}",
                face.id
            );
        }
        checked_seam |= graph.rows[face.row_id as usize].kind == fluid_core::RowKind::MixedSeam;
    }
    assert!(checked_seam, "fixture did not exercise a mixed seam");

    let mut checked_hanging_point = false;
    for lower in &graph.subfaces {
        if graph.rows[lower.row_id as usize].kind != fluid_core::RowKind::MixedSeam {
            continue;
        }
        let axis = lower.axis as usize;
        let tangent = 1 - axis;
        for upper in &graph.subfaces {
            if upper.axis != lower.axis
                || graph.rows[upper.row_id as usize].kind != fluid_core::RowKind::MixedSeam
                || (upper.center[axis] - lower.center[axis]).abs() > 1.0e-6
            {
                continue;
            }
            let boundary = lower.center[tangent] as f64 + 0.5 * lower.measure as f64;
            let upper_minimum = upper.center[tangent] as f64 - 0.5 * upper.measure as f64;
            if (boundary - upper_minimum).abs() > 1.0e-6 {
                continue;
            }
            let expected = sampler.subface_rates()[upper.id as usize] / upper.measure as f64;
            for sign in [-1.0, 1.0] {
                let mut point = [lower.center[0] as f64, lower.center[1] as f64];
                point[axis] += sign * 1.0e-6;
                point[tangent] = boundary;
                let sampled = sampler.sample(&graph, point).unwrap()[axis];
                assert!(
                    (sampled - expected).abs() <= 2.0e-5,
                    "hanging point selected the wrong half-open strip: {sampled} vs {expected}"
                );
            }
            checked_hanging_point = true;
        }
    }
    assert!(checked_hanging_point, "fixture had no mixed-seam hanging point");
}

#[test]
fn face_consistent_sampler_preserves_nonoperator_faces_incident_to_pressure_cells() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |x, y| [0.11 * x, -0.07 * y]);
    fields.pressure_member.fill(0);
    fields.pressure_row_member.fill(0);
    fields.subface_velocity_correction = vec![0.0; graph.subfaces.len()];

    let face = graph
        .subfaces
        .iter()
        .find(|face| {
            face.negative_cell >= 0
                && face.positive_cell >= 0
                && graph.rows[face.row_id as usize].kind != fluid_core::RowKind::ClosedWorld
        })
        .unwrap();
    let pressure_cell = face.negative_cell as usize;
    fields.pressure_member[pressure_cell] = 1;
    fields.face_velocity[face.row_id as usize] = 1.375;
    fields.subface_velocity_correction[face.id as usize] = 0.125;
    let expected = (face.measure * (1.375_f32 - 0.125_f32)) as f64;
    let original = fields.clone();

    let sampler = FaceConsistentVelocity2d::new(&graph, &fields, 1.0 / 30.0).unwrap();
    assert_eq!(fields, original, "private extension mutated baseline fields");
    assert_eq!(fields.pressure_row_member[face.row_id as usize], 0);
    assert_eq!(sampler.subface_rates()[face.id as usize], expected);
}

#[test]
fn diagnostic_extension_seeds_every_nonzero_liquid_cell_without_mutation() {
    let graph = uniform_topology(2, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.25, 0.0]);
    fields.density.fill(0.0);
    fields.extension_depth.fill(255);
    let donor = graph.cells.len() / 2;
    fields.density[donor] = 0.01;
    fields.extension_depth[donor] = 0;
    let original = fields.clone();
    let sampler = FaceConsistentVelocity2d::new(&graph, &fields, 1.0).unwrap();
    assert_eq!(fields, original, "private extension mutated baseline fields");
    assert_eq!(sampler.extension_depths()[donor], 0);
    assert!(
        sampler
            .extension_depths()
            .iter()
            .enumerate()
            .any(|(cell, &depth)| cell != donor && depth != 255),
        "the thin nonzero liquid seed did not extend to any neighbour"
    );
}

#[test]
fn projected_streamfunction_remains_continuous_across_sparse_support() {
    let graph = sparse_left_brick_topology();
    assert!(graph
        .rows
        .iter()
        .any(|row| row.kind == fluid_core::RowKind::SparseAir));
    let fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    let base = FaceConsistentVelocity2d::new(&graph, &fields, 1.0 / 30.0).unwrap();
    let psi = |x: f64, y: f64| {
        (std::f64::consts::PI * x / 8.0).sin()
            * (std::f64::consts::PI * y / 8.0).sin()
    };
    let rates: Vec<_> = graph
        .subfaces
        .iter()
        .map(|face| {
            if face.axis == 0 {
                let x = face.center[0] as f64;
                let half = 0.5 * face.measure as f64;
                psi(x, face.center[1] as f64 + half)
                    - psi(x, face.center[1] as f64 - half)
            } else {
                let y = face.center[1] as f64;
                let half = 0.5 * face.measure as f64;
                -(psi(face.center[0] as f64 + half, y)
                    - psi(face.center[0] as f64 - half, y))
            }
        })
        .collect();
    let receiver_band = vec![true; graph.cells.len()];
    let projected = base
        .with_subface_rates(&graph, &rates, &receiver_band)
        .unwrap();

    let represented = projected
        .sample_with_diagnostic(&graph, [8.0 - 1.0e-4, 4.5])
        .unwrap();
    let sparse_hole = projected
        .sample_with_diagnostic(&graph, [8.0 + 1.0e-4, 4.5])
        .unwrap();
    assert!(!represented.support_fallback);
    assert!(!represented.velocity_model_fallback);
    assert!(!sparse_hole.support_fallback);
    assert!(!sparse_hole.velocity_model_fallback);
    for axis in 0..2 {
        assert!(
            (represented.velocity[axis] - sparse_hole.velocity[axis]).abs() <= 1.0e-3,
            "axis {axis}: represented={represented:?}, sparse={sparse_hole:?}"
        );
    }
}

fn swept_contact_fields(graph: &Graph, velocity: [f64; 2]) -> Fields {
    let mut fields = analytic_fields(graph, |_, _| velocity);
    fields.density.fill(0.0);
    fields.pressure_member.fill(0);
    fields.pressure_row_member.fill(0);
    fields.frame_dt = 1.0;
    for cell in &graph.cells {
        if (4.0..6.0).contains(&cell.center[0]) && (3.0..5.0).contains(&cell.center[1]) {
            fields.density[cell.id as usize] = 1.0;
        }
    }
    prepare_pressure_topology(graph, &mut fields);
    fields
}

#[test]
fn swept_static_wall_support_obeys_boundary_kind_and_relative_motion() {
    let closed = one_brick_topology([BoundaryMode::Closed; 6]);
    let mut approaching = swept_contact_fields(&closed, [3.0, 0.0]);
    let physical = approaching.pressure_member.clone();
    let density = approaching.density.clone();
    prepare_pressure_topology_with_swept_static_wall_support(&closed, &mut approaching);
    assert!(
        approaching
            .pressure_member
            .iter()
            .zip(&physical)
            .any(|(&after, &before)| after != 0 && before == 0),
        "an impacting component did not gain swept pressure support"
    );
    assert_eq!(approaching.density, density, "pressure support created material");

    for velocity in [[-1.0, 0.0], [0.0, 1.0]] {
        let mut nonimpacting = swept_contact_fields(&closed, velocity);
        let physical = nonimpacting.pressure_member.clone();
        let density = nonimpacting.density.clone();
        prepare_pressure_topology_with_swept_static_wall_support(&closed, &mut nonimpacting);
        assert_eq!(nonimpacting.pressure_member, physical, "velocity={velocity:?}");
        assert_eq!(nonimpacting.density, density, "velocity={velocity:?}");
    }

    let open = one_brick_topology([BoundaryMode::Open; 6]);
    let mut exiting = swept_contact_fields(&open, [3.0, 0.0]);
    let physical = exiting.pressure_member.clone();
    let density = exiting.density.clone();
    prepare_pressure_topology_with_swept_static_wall_support(&open, &mut exiting);
    assert_eq!(exiting.pressure_member, physical, "open boundary triggered wall contact");
    assert_eq!(exiting.density, density);

    let mut unrelated_motion_graph = closed.clone();
    for row in &mut unrelated_motion_graph.rows {
        if row.kind == fluid_core::RowKind::ClosedWorld
            && row.axis == 0
            && row.center[0] == 0.0
        {
            row.solid_velocity = 1.0;
        }
    }
    let mut unrelated_motion = swept_contact_fields(&unrelated_motion_graph, [3.0, 0.0]);
    unrelated_motion.solid_motion_active = true;
    let physical = unrelated_motion.pressure_member.clone();
    prepare_pressure_topology_with_swept_static_wall_support(
        &unrelated_motion_graph,
        &mut unrelated_motion,
    );
    assert!(
        unrelated_motion
            .pressure_member
            .iter()
            .zip(physical)
            .any(|(&after, before)| after != 0 && before == 0),
        "unrelated moving geometry disabled contact with the static wall"
    );

    let mut moving_wall_graph = closed.clone();
    for row in &mut moving_wall_graph.rows {
        if row.kind == fluid_core::RowKind::ClosedWorld
            && row.axis == 0
            && row.center[0] == moving_wall_graph.dimensions[0]
        {
            row.solid_velocity = 3.0;
        }
    }
    let mut moving_wall = swept_contact_fields(&moving_wall_graph, [3.0, 0.0]);
    moving_wall.solid_motion_active = true;
    let physical = moving_wall.pressure_member.clone();
    prepare_pressure_topology_with_swept_static_wall_support(&moving_wall_graph, &mut moving_wall);
    assert_eq!(
        moving_wall.pressure_member, physical,
        "a moving ClosedWorld row was treated as a static wall"
    );
}

#[test]
fn zero_trace_segments_is_rejected_before_material_mutation() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    let density = fields.density.clone();
    let result = transport_volume_cellwise_with_commit(
        &graph,
        &mut fields,
        1.0,
        CellwiseRemapOptions {
            trace_segments: 0,
            commit_material: false,
            ..CellwiseRemapOptions::default()
        },
        |_, _| Ok(()),
    );
    assert_eq!(
        result.unwrap_err().0,
        "cellwise remap traceSegments must be in 1..=128"
    );
    assert_eq!(fields.density, density);
}

#[test]
fn compressible_affine_pathline_sweep_is_not_the_frozen_face_flux() {
    // For u=a*x, the backward pathline from a vertical edge at x is
    // x(t)=x*exp(-a*t). Its exact swept width is therefore nonlinear in dt,
    // even though this field is smooth and exactly RT0 representable.
    let a: f64 = 0.4;
    let x: f64 = 5.0;
    let dt: f64 = 1.0;
    let exact_swept_width = x * (1.0 - (-a * dt).exp());
    let frozen_flux_width = a * x * dt;
    assert!((exact_swept_width - 1.6483997698218034).abs() < 1.0e-14);
    assert!((frozen_flux_width - exact_swept_width).abs() > 0.3);
}

#[test]
fn compressible_affine_full_chain_converges_to_nonzero_correction() {
    let graph = single_open_cell_topology();
    let a = 0.4;
    let mut deltas = Vec::new();
    for segments in [1, 2, 4, 8] {
        let mut fields = analytic_fields(&graph, |x, _| [a * x, 0.0]);
        fields.density.fill(1.0);
        let receipt = probe(&graph, &mut fields, segments, 4);
        assert!(receipt.area_balance_relative_error <= 1.0e-12);
        assert!(receipt.max_area_identity_error <= 1.0e-10);
        deltas.push(receipt.max_correction_delta_over_h);
    }
    assert!(deltas[3] > 0.1, "correction unexpectedly vanished: {deltas:?}");
    assert!(
        (deltas[3] - deltas[2]).abs() < 1.0e-5,
        "RK refinement did not converge to a stable residual: {deltas:?}"
    );
}

#[test]
fn band_projection_is_two_sided_and_preserves_every_operator_rate() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    fields.pressure_member.fill(0);
    fields.pressure_row_member.fill(0);

    let shared = graph
        .subfaces
        .iter()
        .find(|face| face.negative_cell >= 0 && face.positive_cell >= 0)
        .unwrap();
    let a = shared.negative_cell as usize;
    let b = shared.positive_cell as usize;
    let mut receivers = vec![false; graph.cells.len()];
    receivers[a] = true;
    receivers[b] = true;
    let mut rates = vec![0.0; graph.subfaces.len()];
    rates[shared.id as usize] = 1.0;

    let pressure_face = graph
        .subfaces
        .iter()
        .find(|face| {
            let cells = [face.negative_cell, face.positive_cell];
            cells.contains(&(a as i32))
                && cells
                    .iter()
                    .any(|&cell| cell >= 0 && cell as usize != a && cell as usize != b)
        })
        .unwrap();
    let pressure_cell = if pressure_face.negative_cell >= 0
        && pressure_face.negative_cell as usize != a
    {
        pressure_face.negative_cell as usize
    } else {
        pressure_face.positive_cell as usize
    };
    fields.pressure_member[pressure_cell] = 1;
    let original_fields = fields.clone();
    rates[pressure_face.id as usize] = 0.375;
    let before = rates.clone();
    let receipt = project_receiver_band_rates_2d(
        &graph,
        &fields,
        &mut rates,
        &receivers,
        1.0 / 30.0,
        256,
    )
    .unwrap();
    assert!(receipt.accepted, "{receipt:?}");
    assert_eq!(receipt.unknowns, 2);
    assert!(receipt.open_components > 0);
    assert!(receipt.changed_subfaces > 0);
    assert!(receipt.max_receiver_normalized_divergence <= BAND_PROJECTION_NORMALIZED_TARGET);
    for face in &graph.subfaces {
        if [face.negative_cell, face.positive_cell]
            .into_iter()
            .any(|cell| cell == pressure_cell as i32)
        {
            assert_eq!(rates[face.id as usize], before[face.id as usize]);
        }
    }
    assert_eq!(fields, original_fields);
}

#[test]
fn band_projection_uses_open_vents_and_rejects_a_closed_defect() {
    let open = single_open_cell_topology();
    let mut open_fields = analytic_fields(&open, |_, _| [0.0, 0.0]);
    open_fields.pressure_member.fill(0);
    open_fields.pressure_row_member.fill(0);
    let mut open_rates = vec![0.0; open.subfaces.len()];
    open_rates[0] = 1.0;
    let open_receipt = project_receiver_band_rates_2d(
        &open,
        &open_fields,
        &mut open_rates,
        &[true],
        1.0 / 30.0,
        256,
    )
    .unwrap();
    assert!(open_receipt.accepted, "{open_receipt:?}");
    assert_eq!(open_receipt.open_components, 1);
    assert!(open_receipt.max_receiver_normalized_divergence <= BAND_PROJECTION_NORMALIZED_TARGET);

    let closed = single_closed_cell_topology();
    let mut closed_fields = analytic_fields(&closed, |_, _| [0.0, 0.0]);
    closed_fields.pressure_member.fill(0);
    closed_fields.pressure_row_member.fill(0);
    let mut closed_rates = vec![0.0; closed.subfaces.len()];
    closed_rates[0] = 1.0;
    let before = closed_rates.clone();
    let closed_receipt = project_receiver_band_rates_2d(
        &closed,
        &closed_fields,
        &mut closed_rates,
        &[true],
        1.0 / 30.0,
        256,
    )
    .unwrap();
    assert!(!closed_receipt.accepted, "{closed_receipt:?}");
    assert_eq!(closed_receipt.enclosed_components, 1);
    assert_eq!(closed_receipt.infeasible_enclosed_components, 1);
    assert_eq!(closed_rates, before);
}

#[test]
fn pregeometry_certificate_stops_an_unresolved_local_band_before_tracing() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    fields.density.fill(1.0);
    fields.pressure_member.fill(0);
    fields.pressure_row_member.fill(0);
    fields.subface_compatibility_rate = vec![0.0; graph.subfaces.len()];
    let interior = graph
        .subfaces
        .iter()
        .find(|face| face.negative_cell >= 0 && face.positive_cell >= 0)
        .unwrap();
    fields.subface_compatibility_rate[interior.id as usize] = 1.0;
    let original = fields.clone();
    let receipt = transport_volume_cellwise_with_commit(
        &graph,
        &mut fields,
        1.0,
        CellwiseRemapOptions {
            closure: CellwiseClosure::Local,
            commit_material: false,
            ..CellwiseRemapOptions::default()
        },
        |_, _| panic!("certificate failure must not commit"),
    )
    .unwrap();
    assert!(receipt.pregeometry_receiver_band > 0);
    assert!(receipt.closure_unresolved > 0);
    assert!(receipt.pregeometry_certificate_violations > 0);
    assert!(receipt.pregeometry_max_abs_normalized_divergence > 0.1);
    assert_eq!(receipt.traces, 0);
    assert_eq!(receipt.rk_evaluations, 0);
    assert_eq!(fields, original);
}

#[test]
fn partial_donor_without_plic_faults_instead_of_gathering_a_full_cell() {
    let graph = uniform_topology(1, 1);
    let mut fields = analytic_fields(&graph, |_, _| [0.0, 0.0]);
    fields.density.fill(0.0);
    let donor = graph.cells.len() / 2;
    fields.density[donor] = 0.5;
    fields.interface_normal.fill(0.0);
    let original = fields.clone();
    let mut committed = false;
    let result = transport_volume_cellwise_with_commit(
        &graph,
        &mut fields,
        1.0,
        CellwiseRemapOptions {
            edge_samples: 1,
            trace_segments: 1,
            closure: CellwiseClosure::None,
            commit_material: true,
        },
        |_, _| {
            committed = true;
            Ok(())
        },
    );
    assert_eq!(
        result.unwrap_err().0,
        format!("cellwise remap partial donor {donor} has no PLIC plane")
    );
    assert!(!committed);
    assert_eq!(fields, original);
}

#[test]
fn rotation_and_single_vortex_reversal_preserve_geometry_contract() {
    let graph = uniform_topology(4, 4);
    let cx = graph.dimensions[0] as f64 * 0.5;
    let cy = graph.dimensions[1] as f64 * 0.5;
    let mut rotation = analytic_fields(&graph, |x, y| [-0.04 * (y - cy), 0.04 * (x - cx)]);
    assert_geometry_certificate(&probe(&graph, &mut rotation, 4, 2));

    let lx = graph.dimensions[0] as f64;
    let ly = graph.dimensions[1] as f64;
    let vortex = |sign: f64, x: f64, y: f64| {
        let px = std::f64::consts::PI * x / lx;
        let py = std::f64::consts::PI * y / ly;
        [
            sign * px.sin().powi(2) * (2.0 * py).sin(),
            -sign * py.sin().powi(2) * (2.0 * px).sin(),
        ]
    };
    for sign in [1.0, -1.0] {
        let mut fields = analytic_fields(&graph, |x, y| vortex(sign, x, y));
        assert_geometry_certificate(&probe(&graph, &mut fields, 4, 2));
    }
}

#[test]
fn shared_chains_close_the_area_identity_across_a_two_to_one_seam() {
    let graph = seam_topology();
    assert!(graph
        .rows
        .iter()
        .any(|row| row.kind == fluid_core::RowKind::MixedSeam));
    let mut fields = analytic_fields(&graph, |x, y| {
        [0.35 + 0.02 * y.sin(), -0.15 + 0.01 * x.cos()]
    });
    assert_geometry_certificate(&probe(&graph, &mut fields, 4, 2));
}

fn water_box(experiment: TransportExperiment) -> World {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/production-scene-golden.json"
    ))
    .unwrap();
    let document: SceneDocument =
        serde_json::from_value(fixture["cases"][0]["scene"].clone()).unwrap();
    World::from_document(
        document,
        ProductionSceneOptions::default(),
        WorldOptions {
            transport_experiment: experiment,
            ..WorldOptions::default()
        },
    )
    .unwrap()
}

fn moving_blob(experiment: TransportExperiment) -> World {
    let mut scene: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/coarse-surface-translation-scene.json"
    ))
    .unwrap();
    scene["sceneId"] = serde_json::json!("cellwise-moving-blob");
    scene["duration_s"] = serde_json::json!(6.0);
    scene["container"]["width_m"] = serde_json::json!(1.6);
    scene["container"]["height_m"] = serde_json::json!(1.2);
    scene["container"]["depth_m"] = serde_json::json!(0.4);
    scene["container"]["fillFraction"] = serde_json::json!(0.0);
    scene["solidVoxels"] = serde_json::json!([]);
    scene["rigidBodies"] = serde_json::json!([]);
    scene["fluid"]["initialCondition"] = serde_json::json!("tank-fill");
    scene["fluid"]["initialLiquidVolumes"] = serde_json::json!([{
        "shape": "sphere",
        "center_m": { "x": -0.25, "y": 0.60, "z": 0.0 },
        "radius_m": 0.30
    }]);
    scene["fluid"]["initialVelocity_m_s"] =
        serde_json::json!({ "x": 0.08, "y": 0.0, "z": 0.0 });
    scene["fluid"]["gravity_m_s2"] =
        serde_json::json!({ "x": 0.0, "y": 0.0, "z": 0.0 });
    scene["fluid"]["surfaceTension_N_m"] = serde_json::json!(0.0);
    scene["fluid"]["refinementRegions"] = serde_json::json!([]);

    let document: SceneDocument = serde_json::from_value(scene).unwrap();
    World::from_document(
        document,
        ProductionSceneOptions::default(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1.0e-6,
            transport_experiment: experiment,
            ..WorldOptions::default()
        },
    )
    .unwrap()
}

fn liquid_centroid_x(world: &World) -> f64 {
    let (weighted, measure) = world
        .state
        .topology
        .graph
        .cells
        .iter()
        .fold((0.0, 0.0), |(weighted, measure), cell| {
            let volume = world.state.fields.density[cell.id as usize] as f64
                * cell.measure as f64;
            (weighted + volume * cell.center[0] as f64, measure + volume)
        });
    weighted / measure
}

#[test]
fn world_selector_defaults_to_legacy_receipt_shape() {
    let world = water_box(TransportExperiment::Baseline);
    let options = serde_json::to_value(&world.options).unwrap();
    assert!(options.get("transportExperiment").is_none());
    let receipt = serde_json::to_value(world.receipt()).unwrap();
    assert!(receipt.get("cellwiseRemap").is_none());
}

#[test]
fn structured_selector_round_trips_the_fixed_trace_segment_count() {
    let configured =
        TransportExperiment::configured_with_samples(CellwiseTransportMode::Probe, 8, 4);
    let encoded = serde_json::to_value(configured).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({"mode":"cellwise-probe","traceSegments":8,"edgeSamples":4})
    );
    assert_eq!(
        serde_json::from_value::<TransportExperiment>(encoded).unwrap(),
        configured
    );
}

#[test]
fn structured_selector_rejects_samples_outside_the_m2b_matrix() {
    let error = serde_json::from_value::<TransportExperiment>(serde_json::json!({
        "mode": "cellwise-probe",
        "traceSegments": 2,
        "edgeSamples": 3
    }))
    .unwrap_err();
    assert!(error.to_string().contains("edgeSamples must be 1, 2, or 4"));
}

#[test]
fn world_probe_measures_geometry_then_runs_legacy_transport() {
    let mut probe = water_box(TransportExperiment::CellwiseProbe);
    let mut baseline = water_box(TransportExperiment::Baseline);
    probe.advance(1, 1.0 / 30.0).unwrap();
    baseline.advance(1, 1.0 / 30.0).unwrap();
    assert!(
        probe.state.fields.fault.is_none(),
        "{:?}",
        probe.state.fields.fault
    );
    assert!(probe.microsteps > 0);
    let receipt = probe.cellwise_remap_receipt.as_ref().unwrap();
    assert_eq!(receipt.closure, CellwiseClosure::BandProjection);
    assert!(
        !receipt.closure_accepted || receipt.pregeometry_certificate_violations > 0,
        "{receipt:?}"
    );
    assert_eq!(receipt.traces, 0);
    assert_eq!(probe.state.topology.graph, baseline.state.topology.graph);
    assert_eq!(probe.state.fields, baseline.state.fields);
    assert_eq!(probe.source_ledger, baseline.source_ledger);
    assert_eq!(probe.microsteps, baseline.microsteps);
}

#[test]
fn world_remap_reports_failed_pregeometry_gate_without_legacy_microsteps() {
    let mut world = water_box(TransportExperiment::CellwiseRemap);
    world.advance(1, 1.0 / 30.0).unwrap();
    assert_eq!(world.microsteps, 0);
    let receipt = world.cellwise_remap_receipt.as_ref().unwrap();
    assert_eq!(receipt.closure, CellwiseClosure::BandProjection);
    assert_eq!(receipt.traces, 0);
    assert_eq!(
        world
            .state
            .fields
            .fault
            .as_ref()
            .map(|fault| fault.stage.as_str()),
        Some("cellwise-remap-pregeometry-continuity")
    );
}

#[test]
fn moving_blob_commits_through_the_sparse_adaptive_world() {
    let mut world = moving_blob(TransportExperiment::configured_with_samples(
        CellwiseTransportMode::Remap,
        1,
        1,
    ));
    let initial_measure = world.receipt().liquid_measure;
    let initial_centroid = liquid_centroid_x(&world);
    let initial_generation = world.state.topology.graph.topology_generation;
    let mut saw_mixed_cell_widths = false;
    let mut saw_mixed_seam = false;
    let mut saw_coarse_liquid_surface = false;
    let mut saw_wet_mixed_seam = false;

    for sequence in 1..=150 {
        world.advance(sequence, 1.0 / 30.0).unwrap();
        let receipt = world.cellwise_remap_receipt.as_ref().unwrap();
        assert!(receipt.traces > 0, "frame {sequence}: {receipt:?}");
        assert!(receipt.closure_accepted, "frame {sequence}: {receipt:?}");
        assert_eq!(
            receipt.pregeometry_certificate_violations, 0,
            "frame {sequence}: {receipt:?}"
        );
        assert!(receipt.gather_clips > 0, "frame {sequence}: {receipt:?}");
        assert!(
            world.pressure.converged,
            "frame {sequence}: {:?}",
            world.pressure
        );
        assert!(
            world.state.fields.fault.is_none(),
            "frame {sequence}: {:?}",
            world.state.fields.fault
        );
        assert_eq!(world.microsteps, 0);
        let mut widths = world
            .state
            .topology
            .graph
            .cells
            .iter()
            .map(|cell| cell.widths[0] as u8)
            .collect::<Vec<_>>();
        widths.sort_unstable();
        widths.dedup();
        saw_mixed_cell_widths |= widths.len() > 1;
        saw_coarse_liquid_surface |= world
            .state
            .topology
            .graph
            .cells
            .iter()
            .any(|cell| {
                let density = world.state.fields.density[cell.id as usize];
                cell.widths[0] > 1.0
                    && density > 1.0e-5
                    && density < world.state.fields.capacity[cell.id as usize] - 1.0e-5
            });
        saw_mixed_seam |= world
            .state
            .topology
            .graph
            .rows
            .iter()
            .any(|row| row.kind == fluid_core::RowKind::MixedSeam);
        saw_wet_mixed_seam |= world
            .state
            .topology
            .graph
            .rows
            .iter()
            .filter(|row| row.kind == fluid_core::RowKind::MixedSeam)
            .any(|row| {
                row.terms
                    .iter()
                    .any(|term| world.state.fields.density[term.cell_id as usize] > 1.0e-5)
            });
    }

    let measure = world.receipt().liquid_measure;
    assert!(
        (measure - initial_measure).abs() <= 2.0e-5 * initial_measure.max(1.0)
    );
    assert!(liquid_centroid_x(&world) > initial_centroid + 4.0);
    assert!(saw_mixed_cell_widths);
    assert!(saw_mixed_seam);
    assert!(saw_coarse_liquid_surface);
    assert!(saw_wet_mixed_seam);
    assert!(world.state.topology.graph.topology_generation > initial_generation);
    assert!(world
        .state
        .fields
        .density
        .iter()
        .all(|&rho| (-1.0e-6..=1.000001).contains(&rho)));
}

#[test]
fn three_dimensional_world_rejects_the_two_dimensional_experiment() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/production-scene-golden.json"
    ))
    .unwrap();
    let document: SceneDocument =
        serde_json::from_value(fixture["cases"][0]["scene"].clone()).unwrap();
    let result = World3d::from_document(
        document,
        ProductionSceneOptions::default(),
        WorldOptions {
            transport_experiment: TransportExperiment::CellwiseProbe,
            ..WorldOptions::default()
        },
    );
    match result {
        Ok(_) => panic!("3D accepted a 2D-only transport experiment"),
        Err(error) => assert_eq!(
            error.0,
            "cellwise transport experiment is available only in 2D"
        ),
    }
}
