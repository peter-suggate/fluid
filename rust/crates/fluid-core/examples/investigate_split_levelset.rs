//! Reproduce the Advance Lab symmetric dam with independently held half-tanks.
//! Arguments: scene JSON, left width, right width, frames, pressure tolerance, dt.
//! Use widths 0 0 to leave adaptivity unconstrained.
use fluid_core::{
    world::{Command, TransportExperiment, World, WorldOptions},
    Fields, Graph,
};
use serde_json::{json, Value};

fn fronts(segments: &[f32]) -> [f32; 2] {
    segments
        .chunks_exact(2)
        .fold([f32::INFINITY, f32::NEG_INFINITY], |r, p| {
            [r[0].min(p[0]), r[1].max(p[0])]
        })
}

fn replay(g: &Graph, f: &Fields, previous: &fluid_core::presentation::RdfSurface) -> Value {
    let topology = fluid_core::presentation::RdfTopology::compile(g).unwrap();
    let mut results = Vec::new();
    for speed in [None, Some(0.0_f32), Some(10.0_f32)] {
        let mut changed = f.clone();
        if let Some(speed) = speed {
            for row in &g.rows {
                changed.face_velocity[row.id as usize] = if row.axis == 1 {
                    0.0
                } else if row.center[0] < g.dimensions[0] / 2.0 {
                    -speed
                } else {
                    speed
                };
            }
        }
        let mut phi = fluid_core::levelset_surface::cell_phi(g, previous).unwrap();
        let (surface, _) = fluid_core::levelset_volume::advance(
            g,
            &mut changed,
            previous,
            &topology,
            &Default::default(),
            &mut phi,
            1.0 / 30.0,
        )
        .unwrap();
        results
            .push(json!({"prescribedOutwardSpeed":speed,"front":fronts(&surface.segments_fine)}));
    }
    json!(results)
}

fn fields(g: &Graph, f: &Fields) -> Value {
    json!({"cells":g.cells,"rows":g.rows,"density":f.density,"capacity":f.capacity,
        "velocity":f.cell_velocity,"faceVelocity":f.face_velocity,"pressure":f.pressure,
        "rhs":f.pressure_rhs,"member":f.pressure_member,"extensionDepth":f.extension_depth})
}
fn plane(bytes: &[u8], id: u32) -> Vec<f32> {
    let word = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    for i in 0..word(16) {
        let p = 32 + i * 16;
        if word(p) == id as usize {
            let start = word(p + 8);
            return (0..word(p + 12))
                .map(|j| {
                    f32::from_le_bytes(bytes[start + 4 * j..start + 4 * j + 4].try_into().unwrap())
                })
                .collect();
        }
    }
    panic!("missing plane {id}")
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seed: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let left: u8 = a[2].parse().unwrap();
    let right: u8 = a[3].parse().unwrap();
    let frames: u32 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(7);
    let tolerance: f32 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(1e-6);
    let dt: f64 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(1.0/30.0);
    let mut w = World::from_document(
        serde_json::from_value(seed["scene"].clone()).unwrap(),
        serde_json::from_value(json!({"dtS":dt,"timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: tolerance,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    let nx = w.state.topology.graph.dimensions[0];
    let ny = w.state.topology.graph.dimensions[1];
    if left != 0 || right != 0 { w.apply_command(1, 1, serde_json::from_value::<Command>(json!({"type":"set-refinement-regions","regions":[
        {"minimumFine":[0,0],"maximumFine":[nx/2.0,ny],"minimumCellWidth":left,"maximumCellWidth":left},
        {"minimumFine":[nx/2.0,0],"maximumFine":[nx,ny],"minimumCellWidth":right,"maximumCellWidth":right}
    ]})).unwrap()).unwrap(); }
    let full_trace = std::env::var_os("FLUID_SPLIT_FULL_TRACE").is_some();
    let fields_trace = full_trace || std::env::var_os("FLUID_SPLIT_FIELDS_TRACE").is_some();
    let stage_frame: Option<u32> = std::env::var("FLUID_SPLIT_STAGE_FRAME").ok()
        .map(|s| s.parse().expect("FLUID_SPLIT_STAGE_FRAME must be a frame number"));
    for frame in 0..=frames {
        let mut stages = Vec::new();
        let mut captured = None;
        let previous = if frame == 1 {
            let phi = plane(w.snapshot(2).unwrap(), 31);
            Some(
                fluid_core::levelset_surface::publish([nx as u32, ny as u32], phi, w.seeded_volume)
                    .unwrap(),
            )
        } else {
            None
        };
        if frame > 0 {
            w.advance_with_observer(frame + 1, dt, |name, g, f| {
                if full_trace && stage_frame.is_none_or(|selected| frame == selected) {
                    stages.push(json!({"name":name,"fields":fields(g,f)}));
                }
                if frame == 1 && name == "level-set-volume-velocity-extension" {
                    captured = Some((g.clone(), f.clone()));
                }
            })
            .unwrap();
        }
        let replays = captured.map(|(g, f)| replay(&g, &f, previous.as_ref().unwrap()));
        let snapshot = w.snapshot(2).unwrap();
        let phi = plane(snapshot, 31);
        let segments = plane(snapshot, 32);
        let fine_fill = if fields_trace {
            let surface = fluid_core::levelset_surface::publish(
                [nx as u32, ny as u32], phi.clone(), w.seeded_volume).unwrap();
            Some(fluid_core::levelset_surface::implied_fill_fine_cells(&surface).unwrap())
        } else { None };
        // Counterfactual extension on cloned fields: production state and
        // subsequent frames are unchanged. Distances are finest-cell units.
        let extension_replay = if std::env::var_os("FLUID_SPLIT_EXTENSION_TRACE").is_some() {
            Some([8, 16, 32].map(|depth| {
                let mut f = w.state.fields.clone();
                let g = &w.state.topology.graph;
                fluid_core::staggered_velocity::extend_faces(g, &mut f, &w.level_set_phi, depth).unwrap();
                let velocity = fluid_core::staggered_velocity::StaggeredVelocity2d::new(g, &f).unwrap();
                json!({"depth":depth,"left":velocity.sample([0.5,12.5]),"right":velocity.sample([nx-0.5,12.5])})
            }))
        } else { None };
        println!(
            "{}",
            json!({"frame":frame,"left":left,"right":right,"dtS":dt,
            "pressureTolerance":tolerance,"pressureIterations":256,
            "front":fronts(&segments),"replays":replays,
            "phi":if fields_trace {Some(phi)} else {None},"segments":if fields_trace {Some(segments)} else {None},
            "cellPhi":if fields_trace {Some(&w.level_set_phi)} else {None},"fineFill":fine_fill,
            "extensionReplay":extension_replay,
            "fields":if fields_trace {Some(fields(&w.state.topology.graph,&w.state.fields))} else {None},
            "maxFaceSpeed":w.state.fields.face_velocity.iter().fold(0.0_f32,|a,v|a.max(v.abs())),
            "cells":w.state.topology.graph.cells.len(),"timings":w.stage_timings,
            "liquidVolume":w.state.topology.graph.cells.iter().map(|c|c.measure as f64*w.state.fields.density[c.id as usize] as f64).sum::<f64>(),
            "policy":w.resolution_policy,"resolution":w.resolution_receipt,
            "stages":stages,"pressure":w.pressure,"transport":w.level_set_volume_receipt,
            "bricks":w.state.topology.bricks.iter().map(|b|&b.seed).collect::<Vec<_>>() })
        );
    }
}
