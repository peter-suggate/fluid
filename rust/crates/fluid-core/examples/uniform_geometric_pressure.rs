//! Replay a captured pre-projection pressure system without evolving the scene.
use fluid_core::uniform_geometric::{pressure::Pressure, UniformGeometricOptions};
use serde::Deserialize;
use std::io::{self, Read};
#[derive(Deserialize)]
struct Snapshot {
    dims: [usize; 2],
    h: [f32; 2],
    phi: Vec<f32>,
    topology: Vec<[f32; 3]>,
    minimum: Vec<f32>,
    rhs: Vec<f32>,
    options: UniformGeometricOptions,
    dt: f32,
    rho: f32,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let r: Snapshot = serde_json::from_str(&input)?;
    r.options.validate()?;
    let mut solver = Pressure::new(r.dims, r.h)?;
    let fine = &mut solver.levels[0];
    for n in [r.phi.len(), r.topology.len(), r.minimum.len(), r.rhs.len()] {
        if n != fine.p.len() {
            return Err("pressure snapshot dimensions do not match".into());
        }
    }
    fine.phi = r.phi;
    fine.topology = r.topology;
    fine.minimum = r.minimum;
    fine.rhs = r.rhs;
    let budget = (r.options.pressure_full_cycles + r.options.pressure_v_cycles) as usize;
    solver.solve(&r.options, r.dt, r.rho, false, budget);
    println!(
        "{}",
        serde_json::json!({"receipt":solver.receipt,"pressure":solver.levels[0].p})
    );
    Ok(())
}
