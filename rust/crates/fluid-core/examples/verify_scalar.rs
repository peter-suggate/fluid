//! Non-UI FSM1 scalar publication fixture runner.

use fluid_core::scalar_authority::{ScalarAuthority, ScalarCellAddress};
use fluid_core::{reconstruct_interfaces, Fields, Graph};
use serde::Deserialize;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    graph: Graph,
    fields: Fields,
    addresses: Vec<ScalarCellAddressInput>,
    source_density: Vec<f32>,
    source_gamma: Vec<f32>,
    leaf_capacity: usize,
    generation: u32,
    topology_slot: u8,
    has_rigid_bodies: bool,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScalarCellAddressInput {
    packet: u32,
    lane: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut f: Fixture = serde_json::from_str(&input)?;
    let addresses = f
        .addresses
        .into_iter()
        .map(|a| ScalarCellAddress {
            packet: a.packet,
            lane: a.lane,
        })
        .collect::<Vec<_>>();
    let mut authority = ScalarAuthority::new(
        f.leaf_capacity,
        f.graph.cells.len(),
        f.graph.topology_generation,
        f.topology_slot,
    );
    authority.publish(
        &f.graph,
        &addresses,
        &f.fields,
        &f.source_density,
        &f.source_gamma,
        f.generation,
        f.graph.topology_generation,
        f.topology_slot,
        f.has_rigid_bodies,
    )?;
    reconstruct_interfaces(&f.graph, &mut f.fields)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "receipt":{"generation":authority.receipt.generation,"topologyGeneration":authority.receipt.topology_generation,
              "topologySlot":authority.receipt.topology_slot,"changedCellCount":authority.receipt.changed_cell_count,
              "nonexactCellCount":authority.receipt.nonexact_cell_count,"bulkCellCount":authority.receipt.bulk_cell_count,
              "flipCellCount":authority.receipt.flip_cell_count,"fault":authority.receipt.fault,
              "firstFaultPacket":authority.receipt.first_fault_packet},
            "changedLow":authority.changed_low,"changedHigh":authority.changed_high,
            "nonexactLow":authority.nonexact_low,"nonexactHigh":authority.nonexact_high,
            "bulkLow":authority.bulk_low,"bulkHigh":authority.bulk_high,
            "flipLow":authority.flip_low,"flipHigh":authority.flip_high,
            "sourceDensity":authority.source_density,"sourceGamma":authority.source_gamma,
            "density":f.fields.density,"gamma":f.fields.gamma,
            "interfaceNormal":f.fields.interface_normal,"interfaceOffset":f.fields.interface_offset
        }))?
    );
    Ok(())
}
