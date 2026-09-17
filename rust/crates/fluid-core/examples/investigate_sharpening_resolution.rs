use fluid_core::geometry::BoundaryMode;
use fluid_core::levelset_sharpening::sharpen_volume;
use fluid_core::levelset_surface::{self, implied_fill_fine_cells};
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::{Fields, Graph};
use serde_json::json;

fn graph(left: u8, right: u8) -> Graph {
    let bricks = (0..2)
        .flat_map(|y| {
            (0..4).map(move |x| {
                let key = x + 4 * y;
                BrickSeed {
                    id: key,
                    key,
                    coordinate: [x as i32, y as i32, 0],
                    span_bricks: 1,
                    resolution: 8 / if x < 2 { left } else { right },
                    active: true,
                    density: vec![],
                    gamma: vec![],
                    refinement_region_scale: None,
                }
            })
        })
        .collect();
    compile_topology::<2>(TopologySeed {
        dimensions: [32, 16, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks,
    })
    .unwrap()
    .graph
}
fn main() {
    for (left, right) in [(1, 1), (2, 2), (4, 4), (1, 2), (2, 1)] {
        let graph = graph(left, right);
        for case in ["balanced", "surplus", "far-balanced"] {
            let vertices = (0..=16)
                .flat_map(|y| (0..=32).map(move |_| y as f32 - 4.0))
                .collect();
            let surface = levelset_surface::publish([32, 16], vertices, 128.0).unwrap();
            let fill = implied_fill_fine_cells(&surface).unwrap();
            let fine_capacity = vec![1.0; 512];
            let target: Vec<f64> = graph
                .cells
                .iter()
                .map(|c| {
                    (c.minimum[1] as usize..c.maximum[1] as usize)
                        .flat_map(|y| {
                            (c.minimum[0] as usize..c.maximum[0] as usize).map(move |x| x + 32 * y)
                        })
                        .map(|i| fill[i] as f64)
                        .sum()
                })
                .collect();
            let mut volume: Vec<f64> = graph
                .cells
                .iter()
                .map(|c| {
                    (c.minimum[1] as usize..c.maximum[1] as usize)
                        .flat_map(|y| {
                            (c.minimum[0] as usize..c.maximum[0] as usize).map(move |_| y)
                        })
                        .map(|y| match case {
                            "balanced" => {
                                if y < 4 {
                                    0.5
                                } else if y < 8 {
                                    0.5
                                } else {
                                    0.0
                                }
                            }
                            "surplus" => {
                                if y < 4 {
                                    0.8
                                } else if y < 8 {
                                    0.5
                                } else {
                                    0.0
                                }
                            }
                            _ => {
                                if y < 4 {
                                    0.5
                                } else if y >= 8 && y < 12 {
                                    0.5
                                } else {
                                    0.0
                                }
                            }
                        })
                        .sum()
                })
                .collect();
            let fields = Fields {
                capacity: vec![1.0; graph.cells.len()],
                ..Default::default()
            };
            let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
            for pass in 0..=5 {
                let halves:Vec<_>=[false,true].map(|r|{
      let ids:Vec<_>=graph.cells.iter().filter(|c|(c.center[0]>=16.0)==r).map(|c|c.id as usize).collect();
      json!({"positive":ids.iter().map(|&i|(volume[i]-target[i]).max(0.0)).sum::<f64>(),
        "negative":ids.iter().map(|&i|(target[i]-volume[i]).max(0.0)).sum::<f64>()})
    }).to_vec();
                println!(
                    "{}",
                    json!({"case":case,"left":left,"right":right,"pass":pass,"halves":halves})
                );
                if pass < 5 {
                    sharpen_volume(&graph, &fields, &fine_capacity, &surface, &phi, &mut volume)
                        .unwrap();
                }
            }
        }
    }
}
