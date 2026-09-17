//! Isolate generation transfer from evolution in the native 2D LSV world.
//! Input: catalog scene JSON ({scene: ...}), frames (default 30), optional freeze.
use fluid_core::initial_scene::SceneDocument;
use fluid_core::transfer::{transfer_fields_allow_overcapacity, NewAirCoverage};
use fluid_core::world::{TransportExperiment, World, WorldOptions};
use fluid_core::{collocate_velocity, Fields, Graph};
use serde_json::{json, Value};

// Fine-grid units, unit liquid density; no dilute-cell threshold. The face
// second moment uses the same per-cell weights as collocation, but averages
// u^2 rather than squaring the average. It is a diagnostic, not the exact
// free-surface pressure-operator energy norm.
fn energy(g: &Graph, f: &Fields, gravity: f64) -> Value {
    let mut fresh = f.clone();
    collocate_velocity(g, &mut fresh);
    let mut face_second = vec![0.0; 2 * g.cells.len()];
    let mut weights = vec![0.0; face_second.len()];
    for r in &g.rows {
        let u = if r.kind == fluid_core::RowKind::ClosedWorld && r.separating {
            f.face_velocity[r.id as usize] as f64
        } else if r.open_fraction > 1e-6 {
            (f.face_velocity[r.id as usize] as f64
                - (1.0 - r.open_fraction as f64) * r.solid_velocity as f64)
                / r.open_fraction as f64
        } else {
            r.solid_velocity as f64
        };
        for t in &r.terms {
            let at = 2 * t.cell_id as usize + r.axis as usize;
            let w =
                t.coefficient.abs() as f64 * r.static_dual_weight.unwrap_or(r.dual_weight) as f64;
            face_second[at] += w * u * u;
            weights[at] += w;
        }
    }
    let (mut mass, mut k, mut kfresh, mut kface, mut pe) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut p, mut pfresh) = ([0.0; 2], [0.0; 2]);
    for c in &g.cells {
        let i = c.id as usize;
        let m = f.density[i] as f64 * c.measure as f64;
        mass += m;
        pe += m * gravity * c.center[1] as f64;
        for a in 0..2 {
            let u = f.cell_velocity[2 * i + a] as f64;
            let v = fresh.cell_velocity[2 * i + a] as f64;
            k += 0.5 * m * u * u;
            kfresh += 0.5 * m * v * v;
            p[a] += m * u;
            pfresh[a] += m * v;
            if weights[2 * i + a] > 0.0 {
                kface += 0.5 * m * face_second[2 * i + a] / weights[2 * i + a];
            }
        }
    }
    json!({"cells":g.cells.len(),"generation":g.topology_generation,"mass":mass,
        "storedK":k,"collocatedK":kfresh,"faceSecondMomentK":kface,"potential":pe,
        "momentum":p,"collocatedMomentum":pfresh})
}

fn replay(source: &Graph, old: &Fields, target: &Graph, next: &Fields, gravity: f64) -> Value {
    // Coverage is used only for target portions absent in the source. All such
    // portions in this no-edit scene are newly represented air.
    let air = [NewAirCoverage {
        minimum_fine: [0.0; 2],
        maximum_exclusive_fine: [target.dimensions[0] as f32, target.dimensions[1] as f32],
    }];
    let moved =
        transfer_fields_allow_overcapacity(source, target, old, &next.capacity, &air).unwrap();
    let mut raw = next.clone();
    raw.density = moved.density;
    raw.cell_velocity = moved.cell_velocity;
    raw.face_velocity = moved.face_velocity;
    let raw_energy = energy(target, &raw, gravity);
    let gathered_velocity = raw.cell_velocity.clone();
    let density_error = raw
        .density
        .iter()
        .zip(&next.density)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let face_error = raw
        .face_velocity
        .iter()
        .zip(&next.face_velocity)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    collocate_velocity(target, &mut raw);
    let mut changes:Vec<Value>=target.cells.iter().filter_map(|c| {
        let i=c.id as usize; let m=raw.density[i] as f64*c.measure as f64;
        let a=[gathered_velocity[2*i],gathered_velocity[2*i+1]];
        let b=[raw.cell_velocity[2*i],raw.cell_velocity[2*i+1]];
        let dk=0.5*m*((b[0] as f64).powi(2)+(b[1] as f64).powi(2)
            -(a[0] as f64).powi(2)-(a[1] as f64).powi(2));
        (dk.abs()>1e-4).then(||json!({"center":c.center,"widths":c.widths,"mass":m,
            "gatherVelocity":a,"publishedVelocity":b,"deltaK":dk,
            "faces":target.incidences[i].iter().map(|&id| {
                let r=&target.rows[id as usize];
                json!({"center":r.center,"axis":r.axis,"kind":r.kind,
                    "openFraction":r.open_fraction,"velocity":raw.face_velocity[id as usize],
                    "oldFaces":(moved.plan.face_offsets[id as usize] as usize..moved.plan.face_offsets[id as usize+1] as usize)
                        .map(|j| {let old_id=moved.plan.face_sources[j] as usize;
                            json!({"center":source.rows[old_id].center,"velocity":old.face_velocity[old_id],
                                "area":moved.plan.face_areas[j]})}).collect::<Vec<_>>()})
            }).collect::<Vec<_>>() }))
    }).collect();
    changes.sort_by(|a, b| {
        b["deltaK"]
            .as_f64()
            .unwrap()
            .abs()
            .total_cmp(&a["deltaK"].as_f64().unwrap().abs())
    });
    changes.truncate(12);
    let cell_error = raw
        .cell_velocity
        .iter()
        .zip(&next.cell_velocity)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(
        (density_error, face_error, cell_error),
        (0.0, 0.0, 0.0),
        "transfer replay must match publication exactly"
    );
    // No evolution: transfer back to the original graph, then collocate.
    let back =
        transfer_fields_allow_overcapacity(target, source, &raw, &old.capacity, &air).unwrap();
    let mut roundtrip = old.clone();
    roundtrip.density = back.density;
    roundtrip.cell_velocity = back.cell_velocity;
    roundtrip.face_velocity = back.face_velocity;
    collocate_velocity(source, &mut roundtrip);
    json!({"rawGather":raw_energy,"roundtrip":energy(source,&roundtrip,gravity),"largestCollocationChanges":changes,
        "replayMaximumErrors":{"density":density_error,"faceVelocity":face_error,"cellVelocity":cell_error}})
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let scene: SceneDocument = serde_json::from_value(input["scene"].clone()).unwrap();
    let frames: u32 = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(30);
    let frozen = args.get(3).is_some_and(|v| v == "freeze");
    let h = input["scene"]["voxelDomain"]["finestCellSize_m"]
        .as_f64()
        .unwrap();
    let gravity = 9.80665 / h;
    let mut world = World::from_document(
        scene,
        serde_json::from_value(json!({"dtS":1.0/30.0,"timeStep":"paper"})).unwrap(),
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    world.resolution_options.policy.freeze_topology = frozen;
    println!(
        "{}",
        json!({"frame":0,"frozen":frozen,"accepted":energy(&world.state.topology.graph,&world.state.fields,gravity)})
    );
    for frame in 1..=frames {
        let mut stages = Vec::new();
        let mut transitions = Vec::new();
        let mut before: Option<(String, Graph, Fields)> = None;
        world
            .advance_with_observer(frame, 1.0 / 30.0, |stage, g, f| {
                let e = energy(g, f, gravity);
                if stage.ends_with("transfer-before") {
                    before = Some((stage.to_string(), g.clone(), f.clone()));
                }
                if stage.ends_with("transfer-after") {
                    let (label, old_graph, old_fields) = before.take().unwrap();
                    transitions.push(
                        json!({"stage":label,"before":energy(&old_graph,&old_fields,gravity),
                    "after":e,"isolation":replay(&old_graph,&old_fields,g,f,gravity)}),
                    );
                }
                stages.push(json!({"stage":stage,"energy":e}));
            })
            .unwrap();
        println!(
            "{}",
            json!({"frame":frame,"frozen":frozen,"stages":stages,"transitions":transitions,
            "accepted":energy(&world.state.topology.graph,&world.state.fields,gravity)})
        );
    }
}
