//! Whole-scene runner. JSON stdin, fields and receipts stdout.
use fluid_core::uniform_geometric::scene_runner;
use std::io::{self, Read};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request = serde_json::from_str(&input)?;
    let start = std::time::Instant::now();
    let mut output = scene_runner::run(request)?;
    output["elapsedMs"] = serde_json::json!(start.elapsed().as_secs_f64() * 1000.0);
    println!("{}", output);
    Ok(())
}
