use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    scene_model::{FluidInflow, Vec3},
    world::WorldOptions,
    world3d::World3d,
};

fn dry_hose_scene() -> SceneDocument {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../core/testdata/world-golden.json")).unwrap();
    let mut scene: SceneDocument =
        serde_json::from_value(fixture["cases"][0]["scene"].clone()).unwrap();
    scene.container.fill_fraction = 0.0;
    scene.fluid.initial_dam_break_dimensions_m = None;
    scene.fluid.initial_liquid_volumes.clear();
    scene.fluid.initial_brick_seeds_m = None;
    scene.fluid.initial_height_field = None;
    scene.fluid.gravity_m_s2 = Vec3::default();
    scene.fluid.inflow = Some(FluidInflow {
        center_m: Vec3 {
            x: -0.42,
            y: 0.32,
            z: -0.12,
        },
        radius_m: 0.075,
        length_m: 0.1,
        velocity_m_s: Vec3 {
            x: 0.30,
            y: 0.20,
            z: 0.10,
        },
        start_s: -1.0,
        end_s: 10.0,
        ramp_s: 0.0,
    });
    scene
}

#[test]
fn dry_xyz_hose_allocates_support_and_commits_each_frozen_frame_budget() {
    let mut world = World3d::from_document(
        dry_hose_scene(),
        ProductionSceneOptions::default(),
        WorldOptions::default(),
    )
    .unwrap();
    let initial_generation = world.revision.topology_generation;
    assert_eq!(world.receipt().liquid_measure, 0.0);

    let dt = 1.0 / 120.0;
    for sequence in 1..=2 {
        world.advance(sequence, dt).unwrap();
        let receipt = world.receipt();
        assert!(receipt.source_ledger.event_requested > 0.0);
        assert!(receipt.source_ledger.event_emitted > 0.0);
        assert_eq!(receipt.source_ledger.fault, 0);
        assert!(receipt.source_ledger.pending >= -f32::EPSILON);
        assert!(receipt.fault.is_none());
        assert!(
            (receipt.liquid_measure - receipt.source_ledger.emitted as f64).abs()
                <= 32.0 * f32::EPSILON as f64 * receipt.liquid_measure.max(1.0)
        );
        assert!(receipt.drift.abs() <= 32.0 * f32::EPSILON as f64);
    }
    assert!(world.revision.topology_generation > initial_generation);
    assert!(world
        .state
        .fields
        .cell_velocity
        .chunks_exact(3)
        .any(|v| v[0] > 0.0 && v[1] > 0.0 && v[2] > 0.0));
}
