use fluid_core::geometry::BoundaryMode;
use fluid_core::numerics::{collocate_velocity, force_faces, prepare_faces_for_level_set_volume};
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::{Fields, Graph, RowKind};

const DT: f32 = 1.0 / 30.0;

fn graph() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![
            BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0, 0, 0],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: Vec::new(),
                gamma: Vec::new(),
                refinement_region_scale: None,
            },
            BrickSeed {
                id: 1,
                key: 1,
                coordinate: [1, 0, 0],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: Vec::new(),
                gamma: Vec::new(),
                refinement_region_scale: None,
            },
        ],
    })
    .unwrap()
    .graph
}

fn fields(graph: &Graph) -> Fields {
    let cells = graph.cells.len();
    Fields {
        density: vec![1.0; cells],
        gamma: vec![1.0; cells],
        capacity: vec![1.0; cells],
        capacity_before: vec![1.0; cells],
        capacity_after: vec![1.0; cells],
        cell_velocity: vec![0.0; 2 * cells],
        face_velocity: vec![0.0; graph.rows.len()],
        pressure: vec![0.0; cells],
        pressure_rhs: vec![0.0; cells],
        pressure_diagonal: vec![1.0; cells],
        pressure_member: vec![1; cells],
        pressure_row_member: vec![1; graph.rows.len()],
        extension_depth: vec![0; cells],
        interface_normal: vec![0.0; 2 * cells],
        interface_offset: vec![0.0; cells],
        ..Fields::default()
    }
}

fn seed_streamfunction_velocity(graph: &Graph, fields: &mut Fields, amplitude: f32) {
    let width = graph.dimensions[0];
    let height = graph.dimensions[1];
    for row in &graph.rows {
        let velocity = if row.kind == RowKind::ClosedWorld {
            0.0
        } else if row.axis == 0 {
            amplitude
                * std::f32::consts::PI
                / height
                * (std::f32::consts::PI * row.center[0] / width).sin()
                * (std::f32::consts::PI * row.center[1] / height).cos()
        } else {
            -amplitude
                * std::f32::consts::PI
                / width
                * (std::f32::consts::PI * row.center[0] / width).cos()
                * (std::f32::consts::PI * row.center[1] / height).sin()
        };
        fields.face_velocity[row.id as usize] = velocity;
    }
}

fn face_energy(graph: &Graph, fields: &Fields) -> f64 {
    graph
        .rows
        .iter()
        .map(|row| {
            let velocity = fields.face_velocity[row.id as usize] as f64;
            0.5 * row.static_dual_weight.unwrap_or(row.dual_weight) as f64 * velocity * velocity
        })
        .sum()
}

fn reverse_rows(graph: &Graph, fields: &Fields) -> (Graph, Fields) {
    let count = graph.rows.len();
    let remap = |old: u32| count as u32 - 1 - old;
    let mut reversed_graph = graph.clone();
    reversed_graph.rows.reverse();
    for (id, row) in reversed_graph.rows.iter_mut().enumerate() {
        row.id = id as u32;
    }
    for incidences in &mut reversed_graph.incidences {
        for row in incidences.iter_mut() {
            *row = remap(*row);
        }
        incidences.sort_unstable();
    }
    for subface in &mut reversed_graph.subfaces {
        subface.row_id = remap(subface.row_id);
    }
    let mut reversed_fields = fields.clone();
    reversed_fields.face_velocity.reverse();
    reversed_fields.pressure_row_member.reverse();
    (reversed_graph, reversed_fields)
}

#[test]
fn uniform_tangent_flow_survives_collocation_and_lsv_face_preparation_at_a_wall() {
    let graph = graph();
    let mut fields = fields(&graph);
    for row in &graph.rows {
        if row.axis == 0 && row.kind != RowKind::ClosedWorld {
            fields.face_velocity[row.id as usize] = 3.0;
        }
    }

    collocate_velocity(&graph, &mut fields);
    prepare_faces_for_level_set_volume(&graph, &mut fields, DT).unwrap();

    let samples: Vec<f32> = graph
        .rows
        .iter()
        .filter(|row| {
            row.axis == 0
                && row.kind != RowKind::ClosedWorld
                && row.center[1] == 0.5
                && (4.0..12.0).contains(&row.center[0])
        })
        .map(|row| fields.face_velocity[row.id as usize])
        .collect();
    assert!(!samples.is_empty());
    assert!(
        samples.iter().all(|velocity| (*velocity - 3.0).abs() <= 1.0e-6),
        "free-slip tangential flow was damped near the wall: {samples:?}"
    );
}

#[test]
fn tiny_divergence_free_vortex_has_no_material_face_energy_filtering() {
    let mut graph = graph();
    let mut fields = fields(&graph);
    seed_streamfunction_velocity(&graph, &mut fields, 1.0e-4);
    let before = face_energy(&graph, &fields);

    collocate_velocity(&graph, &mut fields);
    prepare_faces_for_level_set_volume(&graph, &mut fields, DT).unwrap();
    force_faces(&mut graph, &mut fields, DT, [0.0; 3], [0.0; 3]);

    let retention = face_energy(&graph, &fields) / before;
    assert!(
        (0.9999..=1.0001).contains(&retention),
        "tiny divergence-free flow changed face energy at first order: retention={retention}"
    );
}

#[test]
fn lsv_face_preparation_is_invariant_to_row_execution_order() {
    let graph = graph();
    let mut fields = fields(&graph);
    seed_streamfunction_velocity(&graph, &mut fields, 3.0);
    collocate_velocity(&graph, &mut fields);
    let (reversed_graph, mut reversed_fields) = reverse_rows(&graph, &fields);
    let mut forward_fields = fields;

    prepare_faces_for_level_set_volume(&graph, &mut forward_fields, DT).unwrap();
    prepare_faces_for_level_set_volume(&reversed_graph, &mut reversed_fields, DT).unwrap();

    let count = graph.rows.len();
    let mut maximum_difference = 0.0_f32;
    for old in 0..count {
        maximum_difference = maximum_difference.max(
            (forward_fields.face_velocity[old]
                - reversed_fields.face_velocity[count - 1 - old])
                .abs(),
        );
    }
    assert!(
        maximum_difference <= 2.0e-6,
        "LSV face advection depends on row execution order: maximum difference={maximum_difference}"
    );
}

fn assert_closed_wall_rejects_normal_field(sign: f32) {
    let mut graph = graph();
    let mut fields = fields(&graph);
    for row in &graph.rows {
        if row.axis == 1 && row.kind != RowKind::ClosedWorld {
            fields.face_velocity[row.id as usize] = sign * row.center[1];
        }
    }
    collocate_velocity(&graph, &mut fields);
    prepare_faces_for_level_set_volume(&graph, &mut fields, DT).unwrap();
    force_faces(&mut graph, &mut fields, DT, [0.0; 3], [0.0; 3]);

    let leaked: Vec<(u32, f32, f32)> = graph
        .rows
        .iter()
        .filter(|row| row.kind == RowKind::ClosedWorld)
        .filter_map(|row| {
            let velocity = fields.face_velocity[row.id as usize];
            (velocity.abs() > 1.0e-7).then_some((row.id, row.center[1], velocity))
        })
        .collect();
    assert!(
        leaked.is_empty(),
        "body-force enforcement left normal fluid velocity on stationary walls: {leaked:?}"
    );
}

#[test]
fn approaching_normal_field_does_not_leak_onto_stationary_wall_faces() {
    assert_closed_wall_rejects_normal_field(-1.0);
}

#[test]
fn rebounding_normal_field_does_not_leak_onto_stationary_wall_faces() {
    assert_closed_wall_rejects_normal_field(1.0);
}
