//! Emit per-frame receipts for the 2D/3D dam-break discrepancy investigation.
use fluid_core::{
    initial_scene::SceneDocument,
    world::{TransportExperiment, World, WorldOptions},
};

fn main() {
    let seed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../core/testdata/water-box-dam-break-advance-seed.json"
    )).unwrap();
    let scene: SceneDocument = serde_json::from_value(seed["scene"].clone()).unwrap();
    let mut world = World::from_document(
        scene,
        serde_json::from_value(serde_json::json!({"dtS": 1.0 / 30.0, "timeStep": "paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    ).unwrap();
    for frame in 0..=10 {
        if frame > 0 {
            world.advance(frame, 1.0 / 30.0).unwrap();
        }
        println!("{}", serde_json::json!({
            "frame": frame,
            "cells": world.state.topology.graph.cells.len(),
            "resolution": world.resolution_receipt,
            "policy": world.resolution_policy,
            "options": world.resolution_options,
            "transport": world.level_set_volume_receipt,
        }));
    }
}
