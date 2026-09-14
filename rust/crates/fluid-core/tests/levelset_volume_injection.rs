//! Reader-dropped balls in the direct level-set plus conservative-volume lane.
//!
//! The lab's drop is one gesture, so it has to land in both of this method's
//! authorities at once: the conservative volume the transport moves, and the
//! shared vertex scalar that positions the interface. A ball in only one of
//! them is either a surface with no water behind it or water with no surface.

use fluid_core::{
    initial_scene::SceneDocument,
    injection::LiquidDrop,
    production_scene::ProductionSceneOptions,
    world::{Command, TransportExperiment, World, WorldOptions},
};

const DT: f64 = 1.0 / 30.0;
/// The authored tank fill, in finest cells. Air is everything above it.
const SURFACE_Y_FINE: f64 = 15.25;
/// Centre of the dropped ball, clear of both the pool and the closed lid.
const DROP: LiquidDrop = LiquidDrop {
    centre_fine: [16.0, 20.0],
    radius_fine: 3.0,
};

fn scene() -> SceneDocument {
    serde_json::from_value(serde_json::json!({
        "schemaVersion": "2.0.0",
        "sceneId": "level-set-volume-injection",
        "container": {
            "width_m": 1.6,
            "height_m": 1.2,
            "depth_m": 0.8,
            "fillFraction": 61.0 / 96.0,
            "top": "closed"
        },
        "voxelDomain": { "finestCellSize_m": 0.05, "brickSize_cells": 8 },
        "fluid": {
            "density_kg_m3": 998.2,
            "dynamicViscosity_Pa_s": 0.001002,
            "surfaceTension_N_m": 0.0,
            "gravity_m_s2": { "x": 0.0, "y": -9.80665, "z": 0.0 },
            "initialCondition": "tank-fill"
        },
        "numerics": { "fixedDt_s": DT, "maxDt_s": DT },
        "rigidBodies": []
    }))
    .unwrap()
}

fn world(transport: TransportExperiment) -> World {
    World::from_document(
        scene(),
        ProductionSceneOptions::default(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1.0e-6,
            transport_experiment: transport,
            tracer_budget: 0,
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
        .map(|cell| world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64)
        .sum()
}

/// Volume sitting above the authored pool — the ball, and nothing else.
fn airborne_volume(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .filter(|cell| cell.minimum[1] as f64 >= SURFACE_Y_FINE)
        .map(|cell| world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64)
        .sum()
}

/// Volume-weighted height of the liquid above the pool, in finest cells. Read
/// off the volume rather than off the contour's topmost cell: how sharp the
/// ball's upper rim stays is the transport's business, where its mass sits is
/// the drop's.
fn airborne_centroid(world: &World) -> f64 {
    let (mass, moment) = world
        .state
        .topology
        .graph
        .cells
        .iter()
        .filter(|cell| cell.minimum[1] as f64 >= SURFACE_Y_FINE)
        .fold((0.0_f64, 0.0_f64), |(mass, moment), cell| {
            let volume = world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64;
            (mass + volume, moment + volume * cell.center[1] as f64)
        });
    if mass <= 0.0 {
        return SURFACE_Y_FINE;
    }
    moment / mass
}

fn drop_command(world: &mut World, sequence: u32) {
    world
        .apply_command(sequence, 1, Command::InjectLiquid { drop: DROP })
        .unwrap();
}

#[test]
fn a_dropped_ball_lands_in_the_shared_level_set_as_well_as_the_volume() {
    let mut world = world(TransportExperiment::LevelSetVolume);
    let pool = physical_volume(&world);
    assert_eq!(airborne_volume(&world), 0.0, "the authored tank starts flat");
    assert!(
        world
            .state
            .topology
            .graph
            .cells
            .iter()
            .filter(|cell| cell.minimum[1] as f64 >= SURFACE_Y_FINE)
            .all(|cell| world.level_set_phi[cell.id as usize] > 0.0),
        "the authored zero set already calls some of the air liquid",
    );

    drop_command(&mut world, 1);
    let receipt = world.last_injection.clone().unwrap();
    assert!(receipt.accepted && receipt.fault.is_none());
    assert!(receipt.bricks_demanded > 0 && receipt.cells_wetted > 0);
    // The dose takes max against a smoothed rim, so it admits the requested
    // disk to within that rim's width rather than exactly.
    let requested = std::f64::consts::PI * DROP.radius_fine.powi(2);
    assert!(
        (receipt.area_admitted_fine - requested).abs() < 0.2 * requested,
        "admitted {} against a disk of {requested}",
        receipt.area_admitted_fine,
    );
    assert!((physical_volume(&world) - pool - receipt.area_admitted_fine).abs() < 1.0e-4);
    assert_eq!(world.revision.injections, 1);

    // Both authorities now hold the ball, and every derived cell scalar is one
    // the published contour agrees with.
    let interior: Vec<_> = world
        .state
        .topology
        .graph
        .cells
        .iter()
        .filter(|cell| {
            (cell.center[0] as f64 - DROP.centre_fine[0])
                .hypot(cell.center[1] as f64 - DROP.centre_fine[1])
                < DROP.radius_fine - 1.0
        })
        .map(|cell| cell.id as usize)
        .collect();
    assert!(!interior.is_empty());
    for id in interior {
        assert!(world.state.fields.density[id] > 0.0, "cell {id} took no dose");
        assert!(
            world.level_set_phi[id] < 0.0,
            "cell {id} is inside the ball but its level set reads {}",
            world.level_set_phi[id],
        );
    }
    assert_eq!(
        world.level_set_phi.len(),
        world.state.topology.graph.cells.len()
    );
    assert!(world.level_set_phi.iter().all(|value| value.is_finite()));
}

#[test]
fn a_dropped_ball_falls_and_the_transport_keeps_its_volume() {
    let mut world = world(TransportExperiment::LevelSetVolume);
    drop_command(&mut world, 1);
    let injected = physical_volume(&world);
    let ball = airborne_volume(&world);
    assert!(ball > 0.0);
    let centroid = airborne_centroid(&world);

    let mut maximum_drift = 0.0_f64;
    for frame in 1..=12 {
        world.advance(1 + frame, DT).unwrap();
        assert!(
            world.state.fields.fault.is_none(),
            "frame {frame} faulted at {:?}",
            world.state.fields.fault,
        );
        maximum_drift = maximum_drift.max((physical_volume(&world) - injected).abs());
    }
    assert!(
        maximum_drift <= 1.0e-3 * injected,
        "the dropped ball's volume drifted by {maximum_drift} of {injected}",
    );
    println!(
        "12-frame drop: volume drift={maximum_drift:.9} fine^2 of {injected:.6}, \
         centroid {centroid:.3} -> {:.3} fine, airborne {ball:.3} -> {:.3} fine^2",
        airborne_centroid(&world),
        airborne_volume(&world),
    );
    // Where the ball's mass sits, not how much of it is still above the pool:
    // how fast an airborne ball spreads is the transport's business and is
    // being tuned, but a dropped ball that does not fall is not a drop.
    assert!(
        airborne_centroid(&world) < centroid - 0.5,
        "the ball never left the height it was dropped at ({centroid})",
    );
    assert!(
        airborne_volume(&world) > 0.0,
        "the ball vanished instead of falling",
    );
}

#[test]
fn the_drop_is_the_same_gesture_the_baseline_lane_answers() {
    for transport in [
        TransportExperiment::Baseline,
        TransportExperiment::LevelSetVolume,
    ] {
        let mut world = world(transport);
        let before = physical_volume(&world);
        drop_command(&mut world, 1);
        let receipt = world.last_injection.clone().unwrap();
        assert!(receipt.accepted, "{transport:?} refused the drop");
        assert!(receipt.cells_wetted > 0, "{transport:?} wetted nothing");
        assert!(
            (physical_volume(&world) - before - receipt.area_admitted_fine).abs() < 1.0e-4,
            "{transport:?} did not account for the added mass",
        );
        assert_eq!(world.revision.injections, 1);
        assert_eq!(world.revision.frame, 0, "{transport:?} spent a frame on it");

        // Off the lattice, both lanes refuse the whole ball and change nothing.
        let accepted = world.state.fields.density.clone();
        world
            .apply_command(
                2,
                1,
                Command::InjectLiquid {
                    drop: LiquidDrop {
                        centre_fine: [-64.0, -64.0],
                        radius_fine: 2.0,
                    },
                },
            )
            .unwrap();
        assert!(!world.last_injection.as_ref().unwrap().accepted);
        assert_eq!(world.state.fields.density, accepted);
        assert_eq!(world.revision.injections, 1);

        // And in both lanes the ball then behaves like water: it falls, and
        // the step it falls through does not fault on it.
        let centroid = airborne_centroid(&world);
        for frame in 1..=12 {
            world.advance(2 + frame, DT).unwrap();
            assert!(
                world.state.fields.fault.is_none(),
                "{transport:?} faulted on frame {frame} after the drop: {:?}",
                world.state.fields.fault,
            );
        }
        let fell = centroid - airborne_centroid(&world);
        assert!(fell > 0.5, "{transport:?} let the ball hang at {centroid}");
        println!("{transport:?}: dropped ball fell {fell:.3} fine in 12 frames");
    }
}
