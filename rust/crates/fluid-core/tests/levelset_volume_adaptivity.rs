use fluid_core::{
    initial_scene::SceneDocument,
    world::{TransportExperiment, World, WorldOptions},
};

#[test]
fn dam_break_publishes_coarsening_proofs_and_retains_adaptivity() {
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-dam-break-advance-seed.json"
    )).unwrap();
    let scene: SceneDocument = serde_json::from_value(seed["scene"].clone()).unwrap();
    let dt = 1.0 / 30.0;
    let mut world = World::from_document(
        scene,
        serde_json::from_value(serde_json::json!({"dtS": dt, "timeStep": "paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    ).unwrap();
    let mut saw_certificate = false;
    for frame in 1..=10 {
        world.advance(frame, dt).unwrap();
        saw_certificate |= world.resolution_policy.history.values()
            .any(|history| history.surface_proof.as_ref().is_some_and(|proof|
                proof.generation_by_target_resolution.values().any(|&generation|
                    generation == world.state.topology.graph.topology_generation)));
        let volume: f64 = world.state.topology.graph.cells.iter().map(|cell|
            world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64).sum();
        assert!((volume - 168.0).abs() < 1e-4, "frame {frame}: volume {volume}");
        assert_eq!(world.resolution_receipt.as_ref().unwrap().fault_bits, 0);
    }
    assert!(saw_certificate, "production publication must generate real demotion certificates");
    assert!(world.state.topology.graph.cells.len() < 24 * 16,
        "frame 10 must retain coarse cells instead of saturating the entire domain");
}
