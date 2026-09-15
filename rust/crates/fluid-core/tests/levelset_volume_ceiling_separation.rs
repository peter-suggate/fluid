use fluid_core::{
    initial_scene::SceneDocument,
    publication::{PlaneId, HEADER_BYTES, DIRECTORY_ENTRY_BYTES},
    world::{TransportExperiment, World, WorldOptions},
    RowKind,
};

const DT: f64 = 1.0 / 30.0;

fn published_phi(world: &mut World) -> Vec<f32> {
    let bytes = world.snapshot(2).unwrap();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    for entry in 0..word(16) {
        let at = HEADER_BYTES + entry * DIRECTORY_ENTRY_BYTES;
        if word(at) == PlaneId::RdfVertices as usize {
            let start = word(at + 8);
            return bytes[start..start + 4 * word(at + 12)].chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect();
        }
    }
    panic!("surface publication omitted phi vertices");
}

fn water_box() -> World {
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-dam-break-advance-seed.json"
    ))
    .unwrap();
    let scene: SceneDocument = serde_json::from_value(seed["scene"].clone()).unwrap();
    World::from_document(
        scene,
        serde_json::from_value(serde_json::json!({"dtS":DT, "timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1.0e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn ceiling_contact_releases_under_downward_gravity() {
    let mut world = water_box();
    let ceiling = world.state.topology.graph.dimensions[1];
    let volume = |world: &World| world.state.topology.graph.cells.iter().map(|cell| {
        world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64
    }).sum::<f64>();
    let initial_volume = volume(&world);
    let initial_generation = world.state.topology.graph.topology_generation;
    let mut projections = 0;
    for frame in 1..=90 {
        world.advance_with_observer(frame, DT, |stage, graph, fields| {
            if stage != "velocity-projection" && stage != "projected-support-velocity-projection" {
                return;
            }
            projections += 1;
            for row in graph.rows.iter().filter(|row| row.kind == RowKind::ClosedWorld) {
                let term = row.terms.first().unwrap();
                let inward = if term.coefficient >= 0.0 { 1.0 } else { -1.0 };
                let away = inward * (fields.face_velocity[row.id as usize] - row.solid_velocity);
                assert!(away >= -1.0e-6,
                    "frame {frame}, {stage}, wall {}: pressure projected into solid at {away}", row.id);
            }
        }).unwrap();
        assert!((volume(&world) - initial_volume).abs() <= 1.0e-6 * initial_volume,
            "frame {frame}: wall separation changed conservative volume");
        if [30, 60, 90].contains(&frame) {
            let nx = world.state.topology.graph.dimensions[0] as usize;
            let ny = world.state.topology.graph.dimensions[1] as usize;
            let vertices = published_phi(&mut world);
            let top = &vertices[ny * (nx + 1)..];
            assert!(top.iter().all(|&phi| phi > 0.0),
                "frame {frame}: the ceiling surface stayed attached: {top:?}");
        }
    }
    assert!(projections >= 90);
    assert!(world.state.topology.graph.topology_generation > initial_generation,
        "regression must exercise adaptive topology changes");
    let graph = &world.state.topology.graph;
    let top: Vec<f32> = graph
        .rows
        .iter()
        .filter(|row| row.kind == RowKind::ClosedWorld && row.axis == 1 && row.center[1] == ceiling)
        .flat_map(|row| row.terms.iter().map(|term| world.level_set_phi[term.cell_id as usize]))
        .collect();
    assert!(
        top.iter().all(|&phi| phi > 0.0),
        "released ceiling retained phi-liquid adjacent cells after 90 frames: {top:?}"
    );
}
