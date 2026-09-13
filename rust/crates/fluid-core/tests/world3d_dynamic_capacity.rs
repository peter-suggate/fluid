use fluid_core::{
    initial_scene::SceneDocument, production_scene::ProductionSceneOptions,
    runtime_options3d::apply_initial_values, world::WorldOptions, world3d::World3d,
};

/// Frozen from `sceneDocument(findSceneDefinition("minimal-power-dam-break"))`
/// with the production Node host-flow wet sphere. This keeps the adaptive
/// cut-cell regression independent of a previously built Wasm artifact.
#[test]
fn wet_sphere_first_frame_transfers_the_moving_capacity_epoch_conservatively() {
    let scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-scene.json"
    ))
    .unwrap();
    let host: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-world-options.json"
    ))
    .unwrap();
    let dt = host["dt_s"].as_f64().unwrap();
    let mut production = ProductionSceneOptions::default();
    let mut options = WorldOptions {
        run_epoch: host["options"]["runEpoch"].as_u64().unwrap() as u32,
        command_sequence: host["options"]["commandSequence"].as_u64().unwrap() as u32,
        tracer_budget: host["options"]["tracerBudget"].as_u64().unwrap() as usize,
        ..WorldOptions::default()
    };
    apply_initial_values(
        &mut production,
        &mut options,
        &host["options"]["methodValues"],
    )
    .unwrap();
    let mut world = World3d::from_document(scene, production, options).unwrap();
    let before = world.receipt().liquid_measure;
    world.advance(1, dt).unwrap();
    let receipt = world.receipt();
    assert_eq!(receipt.revision.frame, 1);
    assert!(receipt.fault.is_none());
    assert!((receipt.liquid_measure - before).abs() <= 1e-5 * before.max(1.0));
    for cell in &world.state.topology.graph.cells {
        let id = cell.id as usize;
        let amount = world.state.fields.density[id] * cell.measure;
        let capacity = world.state.fields.capacity[id] * cell.measure;
        assert!(amount <= capacity + 90.536_743e-7 * capacity.max(1.0));
    }
}
