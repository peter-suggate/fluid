//! Regression cases migrated from advance-slice.test.ts at the CPU cutover.
use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    world::{World, WorldOptions},
};
fn waterbox() -> World {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../core/testdata/world-golden.json")).unwrap();
    let case = &fixture["cases"][0];
    let document: SceneDocument = serde_json::from_value(case["scene"].clone()).unwrap();
    let production: ProductionSceneOptions =
        serde_json::from_value(case["productionOptions"].clone()).unwrap();
    World::from_document(
        document,
        production,
        WorldOptions {
            pressure_iterations: 8,
            ..Default::default()
        },
    )
    .unwrap()
}
#[test]
fn sub_isovalue_fluid_is_not_a_velocity_extension_seed() {
    let document = serde_json::from_str(include_str!(
        "../../../core/testdata/coarse-surface-translation-scene.json"
    ))
    .unwrap();
    let mut world = World::from_document(
        document,
        ProductionSceneOptions::default(),
        WorldOptions::default(),
    )
    .unwrap();
    world.state.fields.density.fill(0.49);
    world.state.fields.cell_velocity.fill(3.0);
    fluid_core::extend_velocity(&world.state.topology.graph, &mut world.state.fields, 8).unwrap();
    assert!(world.state.fields.extension_depth.iter().all(|&v| v == 255));
    assert!(world.state.fields.cell_velocity.iter().all(|&v| v == 0.0));
}
#[test]
fn waterbox_pressure8_matches_frozen_legacy_frames() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/legacy-water-box-pressure8-frame2.json"
    ))
    .unwrap();
    let mut world = waterbox();
    for (frame, checkpoint) in expected["frames"].as_array().unwrap().iter().enumerate() {
        world.advance(frame as u32 + 1, world.timestep_s).unwrap();
        assert!(world.state.fields.fault.is_none());
        assert_eq!(
            world.revision.topology_generation as u64,
            checkpoint["generation"].as_u64().unwrap()
        );
        let actual = serde_json::to_value(&world.state.fields).unwrap();
        for (name, field) in checkpoint["fields"].as_object().unwrap() {
            let values = actual[name].as_array().unwrap();
            let words = field.get("words").unwrap_or(&field["values"]).as_array().unwrap();
            assert_eq!(values.len(), words.len(), "frame {frame} {name} length");
            for (i, (actual, expected)) in values.iter().zip(words).enumerate() {
                if field["type"] == "f32" {
                    let expected = u32::from_str_radix(expected.as_str().unwrap(), 16).unwrap();
                    assert_eq!(
                        (actual.as_f64().unwrap() as f32).to_bits(),
                        expected,
                        "frame {frame} {name}[{i}]"
                    );
                } else {
                    assert_eq!(actual, expected, "frame {frame} {name}[{i}]");
                }
            }
        }
        let receipt = world.resolution_receipt.as_ref().unwrap();
        assert_eq!(receipt.promoted_brick_count, 0);
        assert_eq!(receipt.demoted_brick_count, 0);
        assert_eq!(world.topology_slot, 0);
    }
}
