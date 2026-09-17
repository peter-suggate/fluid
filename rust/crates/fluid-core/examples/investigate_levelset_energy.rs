//! Measure mechanical energy without changing the solver or stopping at a test envelope.
//! Arguments: {scene: SceneDocument} JSON, frame count (default 120).
use fluid_core::world::{TransportExperiment, World, WorldOptions};
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let frames: u32 = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(120);
    let scene = &seed["scene"];
    let h = scene["voxelDomain"]["finestCellSize_m"].as_f64().unwrap();
    let gravity = ["x", "y"].map(|a| scene["fluid"]["gravity_m_s2"][a].as_f64().unwrap() / h);
    let mut world = World::from_document(
        serde_json::from_value(scene.clone()).unwrap(),
        serde_json::from_value(json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    for frame in 0..=frames {
        if frame > 0 {
            world.advance(frame, 1.0 / 30.0).unwrap();
        }
        let (mut mass, mut kinetic, mut potential) = (0.0, 0.0, 0.0);
        for c in &world.state.topology.graph.cells {
            let i = c.id as usize;
            let m = c.measure as f64 * world.state.fields.density[i] as f64;
            mass += m;
            for a in 0..2 {
                kinetic += 0.5 * m * (world.state.fields.cell_velocity[2 * i + a] as f64).powi(2);
                potential -= m * gravity[a] * c.center[a] as f64;
            }
        }
        println!(
            "{}",
            json!({"frame":frame,"mass":mass,"kinetic":kinetic,"potential":potential,
            "total":kinetic+potential,"cells":world.state.topology.graph.cells.len(),
            "maxFaceSpeed":world.state.fields.face_velocity.iter().fold(0.0_f32,|a,v|a.max(v.abs())),
            "transport":world.level_set_volume_receipt,"policy":world.resolution_policy})
        );
    }
}
