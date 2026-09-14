use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    types::RowKind,
    world::{TransportExperiment, World, WorldOptions},
};

fn physical_volume(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .map(|cell| {
            world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64
        })
        .sum()
}

fn total_excess(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .map(|cell| {
            let id = cell.id as usize;
            ((world.state.fields.density[id] - world.state.fields.capacity[id]).max(0.0)
                as f64)
                * cell.measure as f64
        })
        .sum()
}

#[test]
fn production_embedding_releases_excess_without_persisting_pressure_source() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../core/testdata/world-golden.json")).unwrap();
    let scene: SceneDocument = serde_json::from_value(
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "cm12-figure-3")
            .unwrap()["scene"]
            .clone(),
    )
    .unwrap();
    let mut production = ProductionSceneOptions::default();
    // Keep the initial atlas tight so the first interface-support query must
    // allocate a receiver and exercise the post-transition pressure solve.
    production.atlas.surface_fine_rings = 0;
    let mut world = World::from_document(
        scene,
        production,
        WorldOptions {
            pressure_iterations: 128,
            pressure_relative_tolerance: 1.0e-7,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(world.embedding.is_some(), "fixture must exercise pressure embedding");
    assert!(world.state.fields.source_rate.iter().all(|&rate| rate == 0.0));

    let graph = &world.state.topology.graph;
    let overfull = graph
        .cells
        .iter()
        .filter(|cell| world.state.fields.capacity[cell.id as usize] > 0.99)
        .filter(|cell| {
            graph.incidences[cell.id as usize]
                .iter()
                .any(|&row| graph.rows[row as usize].kind == RowKind::SparseAir)
        })
        .max_by(|a, b| {
            world.state.fields.density[a.id as usize]
                .total_cmp(&world.state.fields.density[b.id as usize])
        })
        .unwrap();
    let overfull_id = overfull.id as usize;
    world.state.fields.density[overfull_id] = world.state.fields.capacity[overfull_id] + 0.75;
    world.state.fields.gamma[overfull_id] = 1.0;
    world.level_set_phi[overfull_id] = -0.5;
    let initial_volume = physical_volume(&world);
    let initial_excess = total_excess(&world);
    assert!(initial_excess > 0.0);

    let dt = 1.0 / 30.0;
    let expected_rate = 0.5 * initial_excess / dt;
    let generation_before = world.state.topology.graph.topology_generation;
    let mut observed_rate = 0.0_f64;
    let mut observed_rhs = None;
    let mut post_support_projection = false;
    world
        .advance_with_observer(1, dt, |stage, graph, fields| {
            if stage == "pressure-rhs" {
                observed_rate = fields.source_rate.iter().map(|&rate| rate as f64).sum();
                observed_rhs = Some(
                    graph
                        .cells
                        .iter()
                        .map(|cell| fields.pressure_rhs[cell.id as usize] as f64)
                        .sum::<f64>(),
                );
            } else if stage == "projected-support-velocity-projection" {
                post_support_projection = true;
            }
        })
        .unwrap();

    assert!(
        (observed_rate - expected_rate).abs() <= 2.0e-5 * expected_rate.max(1.0),
        "observed pressure source {observed_rate}, expected {expected_rate}"
    );
    assert!(observed_rhs.is_some_and(f64::is_finite));
    assert!(post_support_projection, "fixture must exercise the second pressure solve");
    assert!(
        world.state.topology.graph.topology_generation > generation_before,
        "fixture must exercise a projected-support topology transition"
    );
    assert!(world.state.fields.source_rate.iter().all(|&rate| rate == 0.0));
    assert!(
        (physical_volume(&world) - initial_volume).abs() <= 2.0e-5,
        "pressure feedback must not create or delete conservative volume"
    );

    let excess_after_one = total_excess(&world);
    assert!(
        excess_after_one < initial_excess,
        "excess did not decay: initial={initial_excess}, frame1={excess_after_one}"
    );

    for sequence in 2..=4 {
        world.advance(sequence, dt).unwrap();
        assert!(world.state.fields.source_rate.iter().all(|&rate| rate == 0.0));
        assert!(
            (physical_volume(&world) - initial_volume).abs() <= 2.0e-5,
            "frame {sequence} changed conservative volume"
        );
    }
    let final_excess = total_excess(&world);
    assert!(
        final_excess < excess_after_one,
        "repeated pressure feedback did not continue releasing excess: frame1={excess_after_one}, frame4={final_excess}"
    );
}
