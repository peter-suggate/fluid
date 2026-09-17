//! Replay boundary characteristics on a captured split-probe stage.
//! Arguments: full trace JSONL, frame number.
use fluid_core::{levelset_redistance::sample_scalar, levelset_surface,
    staggered_velocity::StaggeredVelocity2d, Fields, Graph};
use serde_json::{json, Value};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let frame: u64 = args[2].parse().unwrap();
    let rows: Vec<Value> = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str(s).unwrap()).collect();
    let prev = rows.iter().find(|r| r["frame"] == frame - 1).unwrap();
    let row = rows.iter().find(|r| r["frame"] == frame).unwrap();
    let f = &row["stages"].as_array().unwrap().iter()
        .find(|s| s["name"] == "level-set-volume-velocity-extension").unwrap()["fields"];
    let cells: Vec<fluid_core::Cell> = serde_json::from_value(f["cells"].clone()).unwrap();
    let dimensions: [f32; 3] = std::array::from_fn(|a|
        cells.iter().map(|c| c.maximum[a]).fold(0.0, f32::max));
    let graph = Graph { dimension: 2, dimensions, cells,
        rows: serde_json::from_value(f["rows"].clone()).unwrap(), ..Default::default() };
    let fields = Fields { face_velocity: serde_json::from_value(f["faceVelocity"].clone()).unwrap(),
        capacity: serde_json::from_value(f["capacity"].clone()).unwrap(), ..Default::default() };
    let surface = levelset_surface::publish([dimensions[0] as u32, dimensions[1] as u32],
        serde_json::from_value(prev["phi"].clone()).unwrap(), prev["liquidVolume"].as_f64().unwrap()).unwrap();
    let velocity = StaggeredVelocity2d::new(&graph, &fields).unwrap();
    let dt = row["dtS"].as_f64().unwrap() as f32;
    for x in 0..=dimensions[0] as usize {
        let p = [x as f32, dimensions[1]];
        let old = sample_scalar(&surface, p).unwrap();
        if old > 0.0 { continue; }
        let v = velocity.sample(p);
        let mid = std::array::from_fn(|a| (p[a]-0.5*dt*v[a]).clamp(0.0,dimensions[a]));
        let vm = velocity.sample(mid);
        let raw = std::array::from_fn::<_,2,_>(|a| p[a]-dt*vm[a]);
        let departure = velocity.trace(p,dt);
        println!("{}", json!({"frame":frame,"point":p,"oldPhi":old,"velocity":v,
            "rawDeparture":raw,"clampedDeparture":departure,
            "sampledPhi":sample_scalar(&surface,departure)}));
    }
}
