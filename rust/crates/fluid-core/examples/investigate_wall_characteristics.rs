//! Replay phi advection using an immutable captured production face field.
//! Usage: investigate_wall_characteristics FULL_TRACE_JSONL [max_steps=32]
//! This diagnostic never mutates production state or runs pressure/remapping.
use fluid_core::{
    levelset_redistance::sample_scalar, levelset_surface, staggered_velocity::StaggeredVelocity2d,
    Fields, Graph,
};
use serde_json::{json, Value};
fn areas(phi: Vec<f32>) -> [f64; 2] {
    let surface = levelset_surface::publish([32, 16], phi, 128.0).unwrap();
    let fill = levelset_surface::implied_fill_fine_cells(&surface).unwrap();
    let mut area = [0.0; 2];
    for (i, v) in fill.iter().enumerate() {
        area[usize::from(i % 32 >= 16)] += *v as f64;
    }
    area
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read_to_string(&args[1]).unwrap();
    let frames: Vec<Value> = input
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let maximum: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(32);
    for pair in frames.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        let Some(stage) = b["stages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == "level-set-volume-velocity-extension")
        else {
            continue;
        };
        let fs = &stage["fields"];
        let graph = Graph {
            dimension: 2,
            dimensions: [32.0, 16.0, 1.0],
            cells: serde_json::from_value(fs["cells"].clone()).unwrap(),
            rows: serde_json::from_value(fs["rows"].clone()).unwrap(),
            ..Default::default()
        };
        let fields = Fields {
            face_velocity: serde_json::from_value(fs["faceVelocity"].clone()).unwrap(),
            ..Default::default()
        };
        let velocity = StaggeredVelocity2d::new(&graph, &fields).unwrap();
        let phi: Vec<f32> = serde_json::from_value(a["phi"].clone()).unwrap();
        let surface = levelset_surface::publish([32, 16], phi.clone(), 128.0).unwrap();
        let mut traces = Vec::new();
        let mut steps = 1;
        while steps <= maximum {
            let mut next = Vec::new();
            for y in 0..=16 {
                for x in 0..=32 {
                    let mut p = [x as f32, y as f32];
                    for _ in 0..steps {
                        p = velocity.trace(p, (1.0 / 30.0) / (steps as f32));
                    }
                    next.push(sample_scalar(&surface, p).unwrap());
                }
            }
            traces.push(json!({"steps":steps,"area":areas(next)}));
            steps *= 2;
        }
        let mut wall = Vec::new();
        for right in [false, true] {
            let x = if right { 31.0 } else { 0.5 };
            let h = if right { 2.0 } else { 1.0 };
            for y in [0.5 * h, 1.5 * h] {
                let center = [x, y];
                let eps = 0.01;
                let jac: [[f32; 2]; 2] = std::array::from_fn(|axis| {
                    let mut lo = center;
                    lo[axis] -= eps;
                    let mut hi = center;
                    hi[axis] += eps;
                    let a = velocity.sample(lo);
                    let b = velocity.sample(hi);
                    std::array::from_fn(|c| (b[c] - a[c]) / (2.0 * eps))
                });
                let dt = 1.0 / 30.0;
                let departure = velocity.trace(center, dt);
                let derivative: [[f32; 2]; 2] = std::array::from_fn(|axis| {
                    let mut lo = center;
                    lo[axis] -= eps;
                    let mut hi = center;
                    hi[axis] += eps;
                    let a = velocity.trace(lo, dt);
                    let b = velocity.trace(hi, dt);
                    std::array::from_fn(|c| (b[c] - a[c]) / (2.0 * eps))
                });
                let det = derivative[0][0] * derivative[1][1] - derivative[0][1] * derivative[1][0];
                wall.push(json!({"point":center,"velocity":velocity.sample(center),"velocityGradient":jac,
                    "divergence":jac[0][0]+jac[1][1],"departure":departure,"backtraceDeterminant":det}));
            }
        }
        println!(
            "{}",
            json!({"frame":b["frame"],"beforeArea":areas(phi),"traceRefinement":traces,"wall":wall})
        );
    }
}
