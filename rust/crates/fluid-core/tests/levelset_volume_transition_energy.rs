use fluid_core::scene::{compile_scene_2d, SceneDescription, SceneState};
use fluid_core::transfer::transfer_fields_allow_overcapacity;
use fluid_core::world::{TransportExperiment, World, WorldOptions};
use fluid_core::{collocate_velocity, Fields, Graph, RowKind};

fn grid(resolution: u8) -> SceneState<2> {
    let desc: SceneDescription = serde_json::from_value(serde_json::json!({
        "schemaVersion":1,"dimension":2,"dimensions":[8,8,1],"cellSizeM":0.2,
        "dtS":1.0/30.0,"densityKgM3":1.0,
        "boundaries":["open","open","open","open","closed","closed"],
        "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"resolution":resolution,"active":true}]
    }))
    .unwrap();
    compile_scene_2d(desc).unwrap()
}
fn kinetic(g: &Graph, f: &Fields) -> f64 {
    let mut f = f.clone();
    collocate_velocity(g, &mut f);
    g.cells
        .iter()
        .map(|c| {
            let i = c.id as usize;
            0.5 * c.measure as f64
                * f.density[i] as f64
                * ((f.cell_velocity[2 * i] as f64).powi(2)
                    + (f.cell_velocity[2 * i + 1] as f64).powi(2))
        })
        .sum()
}
#[test]
fn refinement_prolongs_mac_field_without_filtering_and_roundtrips_flux() {
    let mut coarse = grid(2);
    let fine = grid(4);
    coarse.fields.density.fill(1.0);
    for r in &coarse.topology.graph.rows {
        coarse.fields.face_velocity[r.id as usize] = (r.center[r.axis as usize] * 0.3).sin();
    }
    collocate_velocity(&coarse.topology.graph, &mut coarse.fields);
    let moved = transfer_fields_allow_overcapacity(
        &coarse.topology.graph,
        &fine.topology.graph,
        &coarse.fields,
        &fine.fields.capacity,
        &[],
    )
    .unwrap();
    for r in &fine.topology.graph.rows {
        let q = r.center[r.axis as usize];
        let lo = (q / 4.0).floor() * 4.0;
        let expected =
            (lo * 0.3).sin() * (1.0 - (q - lo) / 4.0) + ((lo + 4.0) * 0.3).sin() * ((q - lo) / 4.0);
        assert!((moved.face_velocity[r.id as usize] - expected).abs() < 2e-7);
    }
    let mut refined = fine.fields.clone();
    refined.density = moved.density;
    refined.cell_velocity = moved.cell_velocity;
    refined.face_velocity = moved.face_velocity;
    collocate_velocity(&fine.topology.graph, &mut refined);
    let back = transfer_fields_allow_overcapacity(
        &fine.topology.graph,
        &coarse.topology.graph,
        &refined,
        &coarse.fields.capacity,
        &[],
    )
    .unwrap();
    for (a, b) in back.face_velocity.iter().zip(&coarse.fields.face_velocity) {
        assert!(
            (a - b).abs() < 2e-7,
            "round trip filters accepted flux: {a} vs {b}"
        );
    }
}
#[test]
fn quarter_pool_coarsens_with_physical_open_top() {
    let input: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/quarter-pool-transition-seed.json"
    ))
    .unwrap();
    let mut w = World::from_document(
        serde_json::from_value(input["scene"].clone()).unwrap(),
        serde_json::from_value(serde_json::json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    let check_boundaries = |g: &Graph| {
        for r in g.rows.iter().filter(|r| r.axis == 1) {
            if r.center[1] == 0.0 {
                assert_eq!(r.kind, RowKind::ClosedWorld);
            }
            if r.center[1] == g.dimensions[1] {
                assert_ne!(r.kind, RowKind::ClosedWorld);
            }
        }
    };
    check_boundaries(&w.state.topology.graph);
    let mut transitions = 0;
    for frame in 1..=7 {
        w.advance_with_observer(frame, 1.0 / 30.0, |stage, g, _f| {
            if stage.ends_with("transfer-after") {
                transitions += 1;
                check_boundaries(g);
            }
        })
        .unwrap();
        let mass: f64 = w
            .state
            .topology
            .graph
            .cells
            .iter()
            .map(|c| c.measure as f64 * w.state.fields.density[c.id as usize] as f64)
            .sum();
        assert!((mass - 332.5).abs() < 1e-4);
        if frame == 5 {
            // Fixed-dt snapshot after staggered characteristic transport and
            // conservative receiver balancing and symmetric volume allocation.
            // Still-water pages reach B1;
            // the moving ball follows the updated, less-damped velocity field.
            assert_eq!(w.state.topology.graph.cells.len(), 22);
            assert!((kinetic(&w.state.topology.graph, &w.state.fields) - 2530.0701).abs() < 0.01);
        }
    }
    assert!(transitions >= 3);
    assert!(
        w.state.topology.graph.cells.len() < 352,
        "retain useful adaptivity"
    );
}

fn grid3(resolution: u8) -> SceneState<3> {
    let desc: SceneDescription = serde_json::from_value(serde_json::json!({
        "schemaVersion":1,"dimension":3,"dimensions":[8,8,8],"cellSizeM":0.2,
        "dtS":1.0/30.0,"densityKgM3":1.0,
        "boundaries":["open","open","open","open","open","open"],
        "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"resolution":resolution,"active":true}]
    }))
    .unwrap();
    fluid_core::scene::compile_scene::<3>(desc).unwrap()
}

#[test]
fn native_3d_face_transfer_matches_2d_on_every_axis() {
    use fluid_core::transfer3d::transfer_fields_3d;
    let mut coarse = grid3(2);
    let fine = grid3(4);
    coarse.fields.density.fill(1.0);
    for row in &coarse.topology.graph.rows {
        coarse.fields.face_velocity[row.id as usize] = (row.center[row.axis as usize] * 0.3).sin();
    }
    collocate_velocity(&coarse.topology.graph, &mut coarse.fields);
    let moved = transfer_fields_3d(
        &coarse.topology.graph,
        &fine.topology.graph,
        &coarse.fields,
        &fine.fields.capacity,
        &[],
    )
    .unwrap();
    for row in &fine.topology.graph.rows {
        let q = row.center[row.axis as usize];
        let lo = (q / 4.0).floor() * 4.0;
        let expected =
            (lo * 0.3).sin() * (1.0 - (q - lo) / 4.0) + ((lo + 4.0) * 0.3).sin() * ((q - lo) / 4.0);
        assert!((moved.face_velocity[row.id as usize] - expected).abs() < 2e-7);
    }
    let mut refined = fine.fields.clone();
    refined.density = moved.density;
    refined.face_velocity = moved.face_velocity;
    collocate_velocity(&fine.topology.graph, &mut refined);
    let back = transfer_fields_3d(
        &fine.topology.graph,
        &coarse.topology.graph,
        &refined,
        &coarse.fields.capacity,
        &[],
    )
    .unwrap();
    for (a, b) in back.face_velocity.iter().zip(&coarse.fields.face_velocity) {
        assert!((a - b).abs() < 2e-7);
    }
}
