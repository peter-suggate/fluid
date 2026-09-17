use fluid_core::world::{TransportExperiment, World, WorldOptions};

fn phi(world: &mut World) -> Vec<f32> {
    let bytes = world.snapshot(2).unwrap();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at+4].try_into().unwrap()) as usize;
    let entry = (0..word(16)).map(|i| 32+16*i).find(|&i| word(i)==31).unwrap();
    let start = word(entry+8);
    bytes[start..start+4*word(entry+12)].chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
fn pool_splash_releases_both_closed_and_ambient_open_tops() {
    for top in ["closed", "open"] {
        let mut seed: serde_json::Value = serde_json::from_str(include_str!(
            "../../../core/testdata/half-pool-ceiling-seed.json")).unwrap();
        seed["scene"]["container"]["top"] = top.into();
        if top == "open" {
            seed["scene"]["solidVoxels"].as_array_mut().unwrap()
                .retain(|p| p["minimum"][1] != 48);
        }
        let mut world = World::from_document(
            serde_json::from_value(seed["scene"].clone()).unwrap(),
            serde_json::from_value(serde_json::json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
            WorldOptions { pressure_iterations:256, pressure_relative_tolerance:1e-6,
                transport_experiment:TransportExperiment::LevelSetVolume, ..Default::default() }).unwrap();
        let mass = |w: &World| w.state.topology.graph.cells.iter().map(|c|
            c.measure as f64*w.state.fields.density[c.id as usize] as f64).sum::<f64>();
        let initial = mass(&world);
        let mut contacted = false;
        for frame in 1..=90 {
            world.advance(frame,1.0/30.0).unwrap();
            assert!((mass(&world)-initial).abs()<1e-5*initial, "{top}, frame {frame}: mass loss");
            let vertices = phi(&mut world);
            assert!(vertices.iter().all(|v| v.is_finite()));
            let ceiling = &vertices[48*65..];
            contacted |= ceiling.iter().any(|&v| v<0.0);
            if frame == 61 {
                assert!(contacted, "{top}: test did not exercise contact");
                assert!(ceiling.iter().all(|&v| v>0.0),
                    "{top}: ceiling remained attached after the splash fell: {ceiling:?}");
            }
        }
    }
}
