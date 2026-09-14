//! Hydrostatic invariants for the direct level-set plus conservative-volume path.

use fluid_core::{
    initial_scene::SceneDocument,
    numerics::{
        level_set_volume_excess_pressure_source, prepare_pressure_topology_with_level_set,
    },
    production_scene::ProductionSceneOptions,
    resolution::ResolutionRegion,
    world::{Command, TransportExperiment, World, WorldOptions},
};

const DT: f64 = 1.0 / 30.0;
const SURFACE_Y_FINE: f64 = 15.25;

fn scene() -> SceneDocument {
    serde_json::from_value(serde_json::json!({
        "schemaVersion": "2.0.0",
        "sceneId": "level-set-volume-hydrostatic-offset",
        "container": {
            "width_m": 1.6,
            "height_m": 1.2,
            "depth_m": 0.8,
            "fillFraction": 61.0 / 96.0,
            "top": "closed"
        },
        "voxelDomain": {
            "finestCellSize_m": 0.05,
            "brickSize_cells": 8
        },
        "fluid": {
            "density_kg_m3": 998.2,
            "dynamicViscosity_Pa_s": 0.001002,
            "surfaceTension_N_m": 0.0,
            "gravity_m_s2": { "x": 0.0, "y": -9.80665, "z": 0.0 },
            "initialCondition": "tank-fill"
        },
        "numerics": {
            "fixedDt_s": DT,
            "maxDt_s": DT
        },
        "rigidBodies": []
    }))
    .unwrap()
}

fn world(fixed_resolution: Option<u32>) -> World {
    let mut production = ProductionSceneOptions::default();
    production.atlas.fixed_resolution = fixed_resolution;
    World::from_document(
        scene(),
        production,
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1.0e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap()
}

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

fn assert_flat_surface(world: &World) {
    assert_eq!(
        world.level_set_phi.len(),
        world.state.topology.graph.cells.len()
    );
    for cell in &world.state.topology.graph.cells {
        let phi = world.level_set_phi[cell.id as usize] as f64;
        assert!(
            (cell.center[1] as f64 - phi - SURFACE_Y_FINE).abs() <= 2.0e-4,
            "cell {} sees surface height {}",
            cell.id,
            cell.center[1] as f64 - phi
        );
    }
}

fn maximum_surface_height_error(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .map(|cell| {
            let phi = world.level_set_phi[cell.id as usize] as f64;
            (cell.center[1] as f64 - phi - SURFACE_Y_FINE).abs()
        })
        .fold(0.0_f64, f64::max)
}

fn maximum_liquid_face_speed(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .rows
        .iter()
        .filter(|row| {
            row.terms.iter().any(|term| {
                world.state.fields.pressure_member[term.cell_id as usize] != 0
            })
        })
        .fold(0.0_f64, |maximum, row| {
            maximum.max(world.state.fields.face_velocity[row.id as usize].abs() as f64)
        })
}

#[test]
fn authored_height_seeds_the_same_zero_set_on_coarse_and_fine_atlases() {
    let coarse = world(None);
    let fine = world(Some(8));
    assert!(coarse
        .state
        .topology
        .graph
        .cells
        .iter()
        .any(|cell| cell.widths[0] == 8.0));
    assert!(fine
        .state
        .topology
        .graph
        .cells
        .iter()
        .all(|cell| cell.widths[0] == 1.0));
    assert_flat_surface(&coarse);
    assert_flat_surface(&fine);
}

#[test]
fn pressure_phase_uses_phi_while_excess_source_uses_conservative_volume() {
    let world = world(None);
    let graph = &world.state.topology.graph;
    let mut fields = world.state.fields.clone();
    let overfull = 0;
    let phi_liquid = 1;
    fields.density[overfull] = fields.capacity[overfull] + 0.25;
    fields.density[phi_liquid] = 0.0;
    let source = level_set_volume_excess_pressure_source(graph, &fields, DT as f32).unwrap();
    assert!(source[overfull] > 0.0);
    assert_eq!(source[phi_liquid], 0.0);
    fields.source_rate = source;

    let mut phi = vec![1.0; graph.cells.len()];
    phi[phi_liquid] = -1.0;
    prepare_pressure_topology_with_level_set(graph, &mut fields, &phi).unwrap();
    assert_eq!(
        fields.pressure_member[overfull], 0,
        "positive phi stays pressure-air even when conservative volume is over capacity"
    );
    assert_eq!(
        fields.pressure_member[phi_liquid], 1,
        "negative phi stays pressure-liquid even when conservative volume is zero"
    );
}

#[test]
fn flat_pool_remains_hydrostatic_through_asymmetric_refinement() {
    let mut world = world(None);
    let initial_volume = physical_volume(&world);
    assert_flat_surface(&world);
    let mut sequence = 0;
    let mut saw_mixed_rungs = false;
    let mut maximum_speed = 0.0_f64;
    let mut maximum_height_error = 0.0_f64;
    let mut maximum_volume_drift = 0.0_f64;

    for frame in 1..=30 {
        if frame == 4 {
            sequence += 1;
            world
                .apply_command(
                    sequence,
                    1,
                    Command::SetRefinementRegions {
                        regions: vec![ResolutionRegion {
                            minimum_fine: [0.0, 0.0],
                            maximum_fine: [16.0, 24.0],
                            minimum_cell_width: 1,
                            maximum_cell_width: Some(1),
                        }],
                    },
                )
                .unwrap();
        }
        sequence += 1;
        world.advance(sequence, DT).unwrap();
        let has_fine = world
            .state
            .topology
            .graph
            .cells
            .iter()
            .any(|cell| cell.widths[0] == 1.0);
        let has_coarse = world
            .state
            .topology
            .graph
            .cells
            .iter()
            .any(|cell| cell.widths[0] > 1.0);
        saw_mixed_rungs |= has_fine && has_coarse;
        maximum_speed = maximum_speed.max(maximum_liquid_face_speed(&world));
        maximum_height_error = maximum_height_error.max(maximum_surface_height_error(&world));
        maximum_volume_drift = maximum_volume_drift.max((physical_volume(&world) - initial_volume).abs());
        assert!(
            maximum_speed <= 1.0e-3,
            "frame {frame} hydrostatic speed {}",
            maximum_speed
        );
        assert!(
            maximum_volume_drift <= 2.0e-5,
            "frame {frame} changed conservative volume"
        );
        assert_flat_surface(&world);
    }
    assert!(saw_mixed_rungs, "asymmetric edit never produced a mixed-rung topology");
    println!(
        "30-frame hydrostatic maxima: liquid face speed={maximum_speed:.12} fine/s, surface-height error={maximum_height_error:.12} fine, volume drift={maximum_volume_drift:.12} fine^2"
    );
}
