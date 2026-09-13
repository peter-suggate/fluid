//! Compare the former linear owner query with Graph's generation-local index.

use fluid_core::{owner_at, Graph};
use std::io::{self, Read};
use std::time::Instant;

fn linear(graph: &Graph, p: [f32; 3]) -> Option<usize> {
    let d = graph.dimension as usize;
    graph
        .cells
        .iter()
        .position(|c| (0..d).all(|a| p[a] >= c.minimum[a] && p[a] < c.maximum[a]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let graph: Graph = serde_json::from_str(&input)?;
    let count = 200_000usize;
    let queries = (0..count)
        .map(|i| {
            let x = ((i * 2654435761usize) % graph.dimensions[0] as usize) as f32 + 0.5;
            let y = ((i * 2246822519usize) % graph.dimensions[1] as usize) as f32 + 0.5;
            [x, y, 0.5]
        })
        .collect::<Vec<_>>();
    let start = Instant::now();
    let linear_sum = queries
        .iter()
        .filter_map(|&p| linear(&graph, p))
        .fold(0usize, usize::wrapping_add);
    let linear_ns = start.elapsed().as_nanos();
    graph.initialize_spatial_owner_cache();
    let start = Instant::now();
    let indexed_sum = queries
        .iter()
        .filter_map(|&p| owner_at(&graph, p))
        .fold(0usize, usize::wrapping_add);
    let indexed_ns = start.elapsed().as_nanos();
    assert_eq!(indexed_sum, linear_sum);
    println!(
        "{}",
        serde_json::json!({"queries":count,"cells":graph.cells.len(),"linearNs":linear_ns,"indexedNs":indexed_ns,"speedup":linear_ns as f64/indexed_ns as f64,"checksum":indexed_sum})
    );
    Ok(())
}
