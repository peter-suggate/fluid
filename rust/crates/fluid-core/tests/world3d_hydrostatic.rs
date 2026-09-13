use fluid_core::{
    initial_scene::SceneDocument, production_scene::ProductionSceneOptions, scene_model::Vec3,
    world::WorldOptions, world3d::World3d,
};
#[test]
fn flat_three_dimensional_pool_remains_hydrostatic() {
    let mut scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/coarse-surface-translation-scene.json"
    ))
    .unwrap();
    scene.container.width_m = 1.;
    scene.container.height_m = 1.;
    scene.container.depth_m = 1.;
    scene.container.fill_fraction = 0.5;
    scene.voxel_domain.finest_cell_size_m = 0.125;
    scene.fluid.initial_condition = "tank-fill".into();
    scene.fluid.initial_velocity_m_s = None;
    scene.fluid.initial_liquid_volumes.clear();
    scene.fluid.initial_height_field = None;
    scene.fluid.initial_brick_seeds_m = None;
    scene.fluid.inflow = None;
    scene.fluid.refinement_regions.clear();
    scene.fluid.gravity_m_s2 = Vec3 {
        x: 0.,
        y: -9.81,
        z: 0.,
    };
    scene.rigid_bodies.clear();
    scene.terrain = None;
    scene.solid_voxels.clear();
    scene.scenery = None;
    let mut production = ProductionSceneOptions::default();
    production.atlas.fixed_resolution = Some(8);
    production.resolution.policy.freeze_topology = true;
    let mut world = World3d::from_document(
        scene,
        production,
        WorldOptions {
            pressure_iterations: 128,
            pressure_relative_tolerance: 1e-7,
            ..Default::default()
        },
    )
    .unwrap();
    let initial = world.state.fields.density.clone();
    for frame in 1..=3 {
        world.advance(frame, 1. / 30.).unwrap();
        assert!(world.state.fields.fault.is_none());
        assert!(world.receipt().drift.abs() < 1e-6);
        // Air velocities are an extension field; hydrostatic equilibrium
        // constrains the liquid's motion, not that auxiliary field.
        let speed = world
            .state
            .fields
            .cell_velocity
            .chunks_exact(3)
            .enumerate()
            .filter(|(cell, _)| initial[*cell] >= 0.5)
            .flat_map(|(_, v)| v)
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        assert!(speed < 1e-4, "hydrostatic speed {speed}");
        let error = world
            .state
            .fields
            .density
            .iter()
            .zip(&initial)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(error < 1e-5, "hydrostatic density change {error}");
    }
}
