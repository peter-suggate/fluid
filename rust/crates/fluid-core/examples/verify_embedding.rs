//! Non-UI retained-3D pressure embedding fixture runner.

use std::io::{self, Read};

use fluid_core::embedding::{EmbeddingOptions, MappingFaultKind, PressureEmbedding};
use fluid_core::{Fields, Graph};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    source: Graph,
    reduced: Graph,
    fields: Fields,
    options: Options,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Options {
    center_cell_z: i32,
    z_boundary_omitted: bool,
    sparse_air_phi: f32,
    source_generation: u32,
    solid_world: bool,
    maximum_iterations: u32,
    relative_tolerance: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut fixture: Fixture = serde_json::from_str(&input)?;
    let options = EmbeddingOptions {
        center_cell_z: fixture.options.center_cell_z,
        z_boundary_omitted: fixture.options.z_boundary_omitted,
        sparse_air_phi: fixture.options.sparse_air_phi,
        source_generation: fixture.options.source_generation,
        solid_world: fixture.options.solid_world,
    };
    let mut embedding = PressureEmbedding::new(
        fixture.source,
        &fixture.reduced,
        options,
        Some(&fixture.fields),
        None,
    );
    let prepared = embedding.prepare(&fixture.reduced, &mut fixture.fields);
    let solved = embedding.solve(
        &fixture.reduced,
        &mut fixture.fields,
        fixture.options.maximum_iterations,
        fixture.options.relative_tolerance,
        Some(prepared.clone()),
    )?;
    let projected = embedding.project(&fixture.reduced, &mut fixture.fields, &prepared);
    let second = embedding.prepare(&fixture.reduced, &mut fixture.fields);
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "mappingFault": embedding.mapping_fault.map(|f| (match f.kind {
                MappingFaultKind::Cell => "cell", MappingFaultKind::Row => "row" }, f.id)),
            "reducedCell": embedding.reduced_cell, "centreCell": embedding.centre_cell,
            "projectedRow": embedding.projected_row, "centreRow": embedding.centre_row,
            "prepared": prepared_json(&prepared), "pressure": embedding.pressure,
            "solveResidual": solved.solve.residual, "projected": projected,
            "faceVelocity": fixture.fields.face_velocity,
            "authorityOrder": embedding.pressure_authority.execution_order,
            "authorityDirtyTiles": embedding.pressure_authority.receipt.dirty_row_tiles,
            "second": prepared_json(&second),
        }))?
    );
    Ok(())
}

fn prepared_json(p: &fluid_core::embedding::PreparedEmbedding) -> serde_json::Value {
    serde_json::json!({
        "diagonal": p.diagonal, "rhs": p.rhs, "activeRows": p.active_rows,
        "theta": p.theta, "pressureWeight": p.pressure_weight,
        "virtualDiagonal": p.virtual_diagonal, "virtualRhs": p.virtual_rhs,
        "virtualMember": p.virtual_member, "dirtyRowCount": p.dirty_row_count,
    })
}
