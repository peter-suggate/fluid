use fluid_core::{
    closed_air_pressure::solve_closed_air_coupled_pressure,
    compatibility3d::solve_geometric_transport_compatibility_3d,
    numerics::prepare_pressure_topology,
    scene::{compile_scene_3d, SceneDescription, SceneState},
    types::RowKind,
};

fn sealed_line(density: &[f32]) -> SceneState<3> {
    let bricks: Vec<_> = density
        .iter()
        .enumerate()
        .map(|(i, &rho)| {
            serde_json::json!({
                "id":i,"key":i,"coordinate":[i as i32,0,0],
                "resolution":1,"active":true,"density":[rho],"gamma":[1.0]
            })
        })
        .collect();
    let description: SceneDescription = serde_json::from_value(serde_json::json!({
        "schemaVersion":1,"dimension":3,"dimensions":[8*density.len(),8,8],
        "cellSizeM":1.0,"originM":[0,0,0],"dtS":1.0,
        "densityKgM3":1.0,"gravityM_s2":[0,0,0],
        "boundaries":["closed","closed","closed","closed","closed","closed"],
        "bricks":bricks
    }))
    .unwrap();
    let mut state = compile_scene_3d(description).unwrap();
    for row in &mut state.topology.graph.rows {
        if row.kind == RowKind::ClosedWorld {
            row.open_fraction = 0.0;
            row.open_fraction_before = Some(0.0);
            row.open_fraction_after = Some(0.0);
        }
    }
    state.fields.frame_dt = 1.0;
    state
}

fn open_line(density: &[f32]) -> SceneState<3> {
    let mut state = sealed_line(density);
    for row in &mut state.topology.graph.rows {
        if row.kind == RowKind::ClosedWorld && row.axis == 0 {
            row.kind = RowKind::SparseAir;
            row.open_fraction = 1.0;
        }
    }
    state
}

fn interface_faces(state: &SceneState<3>) -> Vec<usize> {
    state
        .topology
        .graph
        .subfaces
        .iter()
        .filter(|face| {
            face.negative_cell >= 0
                && face.positive_cell >= 0
                && (state.fields.pressure_member[face.negative_cell as usize] != 0)
                    != (state.fields.pressure_member[face.positive_cell as usize] != 0)
        })
        .map(|face| face.id as usize)
        .collect()
}

fn exercise(mut state: SceneState<3>) {
    let rows = prepare_pressure_topology(&state.topology.graph, &mut state.fields);
    let interfaces = interface_faces(&state);
    let interface_rows: std::collections::BTreeSet<_> = interfaces
        .iter()
        .map(|&face| state.topology.graph.subfaces[face].row_id as usize)
        .collect();
    assert_eq!(interface_rows.len(), 2);
    let graph = &mut state.topology.graph;
    let fields = &mut state.fields;
    // Equal +X velocities exchange volume between the two sealed air
    // components while leaving the liquid component's aggregate unchanged.
    for &row in &interface_rows {
        fields.face_velocity[row] = 1.0;
    }
    let receipt = solve_closed_air_coupled_pressure(graph, fields, &rows, 128, 1e-7, None).unwrap();
    assert!(receipt.pressure.residual <= 1e-4, "{receipt:?}");
    assert!(
        receipt.maximum_normalized_physical_residual <= receipt.normalized_target,
        "{receipt:?}"
    );
    let primary_rates: Vec<_> = interfaces
        .iter()
        .map(|&face| fields.subface_compatibility_rate[face].to_bits())
        .collect();
    for &face in &interfaces {
        assert!(fields.subface_compatibility_rate[face].abs() <= 1e-4);
    }
    solve_geometric_transport_compatibility_3d(graph, fields, 256).unwrap();
    assert_eq!(
        interfaces
            .iter()
            .map(|&face| fields.subface_compatibility_rate[face].to_bits())
            .collect::<Vec<_>>(),
        primary_rates,
        "secondary projection changed a primary-owned interface rate"
    );
}

#[test]
fn sealed_air_air_liquid_air_air_components_are_coupled_by_primary_pressure() {
    exercise(sealed_line(&[0.0, 0.0, 1.0, 0.0, 0.0]));
}

#[test]
fn sealed_singleton_air_components_are_not_dropped_as_zero_degree_cells() {
    exercise(sealed_line(&[0.0, 1.0, 0.0]));
}

#[test]
fn exterior_anchored_air_keeps_the_legacy_primary_interface_flux() {
    let mut state = open_line(&[0.0, 1.0, 0.0]);
    let rows = prepare_pressure_topology(&state.topology.graph, &mut state.fields);
    let interfaces = interface_faces(&state);
    let interface_rows: std::collections::BTreeSet<_> = interfaces
        .iter()
        .map(|&face| state.topology.graph.subfaces[face].row_id as usize)
        .collect();
    for &row in &interface_rows {
        state.fields.face_velocity[row] = 1.0;
    }
    let before: Vec<_> = interface_rows
        .iter()
        .map(|&row| state.fields.face_velocity[row].to_bits())
        .collect();
    solve_closed_air_coupled_pressure(
        &state.topology.graph,
        &mut state.fields,
        &rows,
        128,
        1e-7,
        None,
    )
    .unwrap();
    assert_eq!(
        interface_rows
            .iter()
            .map(|&row| state.fields.face_velocity[row].to_bits())
            .collect::<Vec<_>>(),
        before,
        "an exterior-anchored air component acquired a sealed-air constraint"
    );
}

#[test]
fn globally_unbalanced_sealed_source_is_rejected_before_gauge_pinning() {
    let mut state = sealed_line(&[0.0, 1.0, 0.0]);
    state.fields.source_rate = vec![0.0; state.topology.graph.cells.len()];
    state.fields.source_rate[1] = 1.0;
    let rows = prepare_pressure_topology(&state.topology.graph, &mut state.fields);
    let error = solve_closed_air_coupled_pressure(
        &state.topology.graph,
        &mut state.fields,
        &rows,
        128,
        1e-7,
        None,
    )
    .unwrap_err();
    assert!(error.0.contains("defect"), "{error}");
}

#[test]
fn nonfinite_primary_face_rate_cannot_produce_an_accepted_receipt() {
    let mut state = sealed_line(&[0.0, 1.0, 0.0]);
    let rows = prepare_pressure_topology(&state.topology.graph, &mut state.fields);
    let interface = interface_faces(&state)[0];
    let row = state.topology.graph.subfaces[interface].row_id as usize;
    state.fields.face_velocity[row] = f32::NAN;
    assert!(solve_closed_air_coupled_pressure(
        &state.topology.graph,
        &mut state.fields,
        &rows,
        128,
        1e-7,
        None,
    )
    .is_err());
}
