use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    runtime_options3d::apply_initial_values,
    scene_model::{Quaternion, Vec3},
    world::WorldOptions,
    world3d::{Command3d, RigidConstraint3d, World3d},
};

fn world() -> World3d {
    let scene: SceneDocument = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-scene.json"
    ))
    .unwrap();
    let host: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/minimal-power-wet-sphere-world-options.json"
    ))
    .unwrap();
    let mut production = ProductionSceneOptions::default();
    let mut options = WorldOptions {
        run_epoch: 37,
        command_sequence: 0,
        tracer_budget: 0,
        ..WorldOptions::default()
    };
    apply_initial_values(
        &mut production,
        &mut options,
        &host["options"]["methodValues"],
    )
    .unwrap();
    World3d::from_document(scene, production, options).unwrap()
}

fn constraint(id: &str, held: bool, position_m: Vec3, velocity: Vec3) -> RigidConstraint3d {
    RigidConstraint3d {
        id: id.into(),
        held,
        position_m,
        orientation: Quaternion::default(),
        linear_velocity_m_s: velocity,
        angular_velocity_rad_s: Vec3::default(),
    }
}

#[test]
fn held_pose_publishes_while_paused_then_release_resumes_live_integration_atomically() {
    let mut world = world();
    let id = world.physical.bodies[0].description.id.clone();
    let original = world.physical.bodies[0].clone();
    let field_revision = world.revision.field_revision;
    let bad = vec![
        constraint(
            &id,
            true,
            Vec3 {
                x: 0.0,
                y: 0.62,
                z: 0.0,
            },
            Vec3::default(),
        ),
        constraint("missing", true, Vec3::default(), Vec3::default()),
    ];
    assert!(world
        .apply_command(1, 37, Command3d::SetRigidConstraints { constraints: bad })
        .is_err());
    assert_eq!(world.physical.bodies[0], original);
    assert_eq!(world.revision.command_sequence, 0);
    assert_eq!(world.revision.field_revision, field_revision);

    let held_position = Vec3 {
        x: 0.0,
        y: 0.62,
        z: 0.0,
    };
    world
        .apply_command(
            1,
            37,
            Command3d::SetRigidConstraints {
                constraints: vec![constraint(&id, true, held_position, Vec3::default())],
            },
        )
        .unwrap();
    let published = world.physical.bodies[0].position_m;
    assert_eq!(published.x, held_position.x as f32 as f64);
    assert_eq!(published.y, held_position.y as f32 as f64);
    assert!(world.physical.bodies[0].held);
    assert_eq!(world.revision.field_revision, field_revision + 1);
    assert_eq!(world.revision.surface_revision, field_revision + 1);

    world.advance(2, 1.0 / 30.0).unwrap();
    assert_eq!(world.physical.bodies[0].position_m, published);
    assert!(world.physical.bodies[0].held);
    assert!(
        world.receipt().fault.is_none(),
        "{:?}",
        world.receipt().fault
    );

    world
        .apply_command(
            3,
            37,
            Command3d::SetRigidConstraints {
                constraints: vec![constraint(
                    &id,
                    false,
                    published,
                    Vec3 {
                        x: 0.1,
                        y: 0.0,
                        z: 0.0,
                    },
                )],
            },
        )
        .unwrap();
    assert!(!world.physical.bodies[0].held);
    let released_x = world.physical.bodies[0].position_m.x;
    world.advance(4, 1.0 / 30.0).unwrap();
    assert!(world.physical.bodies[0].position_m.x > released_x);
    assert!(world.receipt().fault.is_none());
}
