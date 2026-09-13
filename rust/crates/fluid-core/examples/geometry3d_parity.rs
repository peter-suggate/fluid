use std::io::{self, Read};

use fluid_core::geometry::BoundaryMode;
use fluid_core::numerics::reconstruct_interfaces;
use fluid_core::presentation3d::Rdf3d;
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::Fields;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    dimensions: [u32; 3],
    samples: Vec<CellSample>,
}

#[derive(Deserialize)]
struct CellSample {
    center: [f32; 3],
    density: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CellPlane {
    center: [f32; 3],
    normal: [f32; 3],
    offset: f32,
    rdf_center: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = String::new();
    io::stdin().read_to_string(&mut json)?;
    let input: Input = serde_json::from_str(&json)?;
    if input.dimensions != [16, 8, 8] {
        return Err("the production parity fixture requires dimensions [16,8,8]".into());
    }
    let mut compiled = compile_topology::<3>(TopologySeed {
        dimensions: input.dimensions,
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![
            BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0, 0, 0],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            },
            BrickSeed {
                id: 1,
                key: 1,
                coordinate: [1, 0, 0],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: vec![],
                gamma: vec![],
                refinement_region_scale: None,
            },
        ],
    })?;
    let graph = &mut compiled.graph;
    if input.samples.len() != graph.cells.len() {
        return Err(format!(
            "samples has {} entries; expected {}",
            input.samples.len(),
            graph.cells.len()
        )
        .into());
    }
    for row in &mut graph.rows {
        if row.kind == fluid_core::RowKind::ClosedWorld {
            row.open_fraction = 0.0;
        }
    }
    let n = graph.cells.len();
    let density: Vec<f32> = graph
        .cells
        .iter()
        .map(|cell| {
            input
                .samples
                .iter()
                .find(|sample| sample.center == cell.center)
                .map(|sample| sample.density)
                .ok_or_else(|| format!("missing density at {:?}", cell.center))
        })
        .collect::<Result<_, _>>()?;
    let mut fields = Fields {
        density,
        gamma: vec![1.0; n],
        capacity: vec![1.0; n],
        pressure: vec![0.0; n],
        pressure_rhs: vec![0.0; n],
        pressure_diagonal: vec![0.0; n],
        pressure_member: vec![0; n],
        extension_depth: vec![0; n],
        cell_velocity: vec![0.0; 3 * n],
        face_velocity: vec![0.0; graph.rows.len()],
        interface_normal: vec![0.0; 3 * n],
        interface_offset: vec![0.0; n],
        ..Default::default()
    };
    reconstruct_interfaces(&graph, &mut fields)?;
    let rdf = Rdf3d::reconstruct(&compiled, &fields)?;
    let graph = &compiled.graph;
    let cells: Vec<_> = graph
        .cells
        .iter()
        .map(|cell| {
            let id = cell.id as usize;
            CellPlane {
                center: cell.center,
                normal: [
                    fields.interface_normal[3 * id],
                    fields.interface_normal[3 * id + 1],
                    fields.interface_normal[3 * id + 2],
                ],
                offset: fields.interface_offset[id],
                rdf_center: rdf.center_values[id],
            }
        })
        .collect();
    serde_json::to_writer(io::stdout(), &cells)?;
    Ok(())
}
