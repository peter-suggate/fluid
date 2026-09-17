//! Emit per-frame receipts for the 2D/3D dam-break discrepancy investigation.
use fluid_core::{
    initial_scene::SceneDocument,
    world::{TransportExperiment, World, WorldOptions},
};

fn main() {
    // Optional scene-document JSON and frame count let this probe follow the
    // live scene catalog rather than only the historical dam fixture.
    let args: Vec<String> = std::env::args().collect();
    let source = args.get(1).map(|path| std::fs::read_to_string(path).unwrap());
    let seed: serde_json::Value = serde_json::from_str(source.as_deref().unwrap_or(include_str!(
        "../../../core/testdata/water-box-dam-break-advance-seed.json"
    ))).unwrap();
    let frames: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10);
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
    for frame in 0..=frames {
        let mut stage_cells = std::collections::BTreeMap::new();
        if frame > 0 {
            world.advance_with_observer(frame, 1.0 / 30.0, |stage, graph, _| {
                stage_cells.insert(stage.to_string(), graph.cells.len());
            }).unwrap();
        }
        println!("{}", serde_json::json!({
            "frame": frame,
            "cells": world.state.topology.graph.cells.len(),
            "bricks": world.state.topology.bricks.iter().map(|b| &b.seed).collect::<Vec<_>>(),
            "stageCells": stage_cells,
            "resolution": world.resolution_receipt,
            "policy": world.resolution_policy,
            "options": world.resolution_options,
            "transport": world.level_set_volume_receipt,
        }));
    }
}
