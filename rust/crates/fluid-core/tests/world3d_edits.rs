use fluid_core::{
    initial_scene::SceneDocument,
    injection3d::LiquidDrop3d,
    production_scene::ProductionSceneOptions,
    scene_model::Vec3,
    world::WorldOptions,
    world3d::{Command3d, World3d},
};
fn empty_scene() -> SceneDocument {
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
    scene
}
#[test]
fn live_drop_allocates_dry_pages_and_counts_only_admitted_volume() {
    let scene = empty_scene();
    let mut world = World3d::from_document(
        scene,
        ProductionSceneOptions::default(),
        WorldOptions::default(),
    )
    .unwrap();
    assert_eq!(world.receipt().liquid_measure, 0.0);
    let drop = LiquidDrop3d {
        centre_m: Vec3 {
            x: 0.15,
            y: 0.35,
            z: 0.0,
        },
        radius_m: 0.15,
        half_height_m: None,
    };
    world
        .apply_command(1, 1, Command3d::InjectLiquid { drop })
        .unwrap();
    let volume = world.receipt().liquid_measure;
    assert!(volume > 0.0);
    assert!(world.last_injection.as_ref().unwrap().accepted);
    assert!((world.last_injection.as_ref().unwrap().volume_admitted_fine - volume).abs() < 1e-9);
    assert!(world.receipt().drift.abs() < 1e-12);
    world
        .apply_command(2, 1, Command3d::InjectLiquid { drop })
        .unwrap();
    assert_eq!(world.receipt().liquid_measure, volume);
    assert_eq!(
        world.last_injection.as_ref().unwrap().volume_admitted_fine,
        0.0
    );
    assert!(world.snapshot(u32::MAX).unwrap().len() > 32);
}
#[test]
fn invalid_live_control_preserves_accepted_revision() {
    let mut world = World3d::from_document(
        empty_scene(),
        ProductionSceneOptions::default(),
        WorldOptions::default(),
    )
    .unwrap();
    let revision = world.revision.field_revision;
    assert!(world
        .apply_command(
            1,
            1,
            Command3d::InjectLiquid {
                drop: LiquidDrop3d {
                    centre_m: Vec3::default(),
                    radius_m: -1.,
                    half_height_m: None
                }
            }
        )
        .is_err());
    assert_eq!(world.revision.command_sequence, 0);
    assert_eq!(world.revision.field_revision, revision);
    assert_eq!(world.receipt().liquid_measure, 0.0);
}
