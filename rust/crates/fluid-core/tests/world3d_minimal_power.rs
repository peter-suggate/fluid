use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::{ProductionSceneOptions, ProductionTimeStep},
    world::WorldOptions,
    world3d::World3d,
};

fn minimal_power_options() -> (ProductionSceneOptions, WorldOptions) {
    let mut production = ProductionSceneOptions::default();
    production.time_step = ProductionTimeStep::Scene;
    production.atlas.brick_fine_resolution = 8;
    production.atlas.surface_fine_rings = 1;
    let options = WorldOptions {
        run_epoch: 37,
        command_sequence: 0,
        pressure_iterations: 12,
        pressure_relative_tolerance: 1e-6,
        tracer_budget: 96,
        topology_page_budget: None,
    };
    (production, options)
}

#[test]
fn wet_sphere_uses_moving_geometry_pressure_seam() {
    let scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-scene.json"
    ))
    .unwrap();
    let (production, options) = minimal_power_options();
    let mut world = World3d::from_document(scene, production, options).unwrap();

    world.advance(1, 1.0 / 30.0).unwrap();

    let receipt = world.receipt();
    assert!(receipt.coupled_pressure.skipped_for_solid_motion);
    assert!(receipt.compatibility.skipped_for_solid_motion);
    assert!(receipt.fault.is_none(), "{:?}", receipt.fault);
}

#[test]
fn body_free_minimal_power_pressure12_reprojects_after_support_transition() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-scene.json"
    ))
    .unwrap();
    value["rigidBodies"] = serde_json::json!([]);
    let scene: SceneDocument = serde_json::from_value(value).unwrap();
    let (production, options) = minimal_power_options();
    let mut world = World3d::from_document(scene, production, options).unwrap();
    world.advance(1, 1.0 / 30.0).unwrap();
    let receipt = world.receipt();
    assert_eq!(receipt.pressure.iterations, 12);
    assert!(receipt.primary_refinement.iterations <= 64);
    assert_eq!(
        receipt.pressure_total_iterations,
        receipt.pressure.iterations
            + receipt.primary_refinement.iterations
            + receipt.coupled_pressure.f64_refinement_iterations
    );
    assert!(receipt.compatibility.accepted);
    assert!(receipt.compatibility.measured_normalized_residual <= 2.0 * f32::EPSILON);
}

#[test]
fn body_free_water_box_pressure12_has_certified_compatibility() {
    let scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-body-free-world3d-scene.json"
    ))
    .unwrap();
    let mut production = ProductionSceneOptions::default();
    production.time_step = ProductionTimeStep::Scene;
    production.atlas.brick_fine_resolution = 8;
    production.atlas.surface_fine_rings = 1;
    let options = WorldOptions {
        run_epoch: 39,
        command_sequence: 0,
        pressure_iterations: 12,
        pressure_relative_tolerance: 1e-6,
        tracer_budget: 0,
        topology_page_budget: None,
    };
    let mut world = World3d::from_document(scene, production, options).unwrap();
    world.advance(1, 1.0 / 30.0).unwrap();
    let receipt = world.receipt();
    assert!(receipt.compatibility.accepted);
    assert!(receipt.compatibility.measured_normalized_residual <= 2.0 * f32::EPSILON);
    assert!(
        receipt.compatibility.pre_postconditioning_maximum
            <= receipt.compatibility.pre_postconditioning_bound_maximum
    );
}
