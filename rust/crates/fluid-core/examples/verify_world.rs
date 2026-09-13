//! Native production-world runner used by tools/wasm/world-parity.ts.
use fluid_core::initial_scene::SceneDocument;
use fluid_core::production_scene::ProductionSceneOptions;
use fluid_core::world::{World, WorldOptions};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    scene: SceneDocument,
    #[serde(default)]
    production_options: ProductionSceneOptions,
    #[serde(default)]
    world_options: WorldOptions,
    #[serde(default = "frames")]
    frames: u32,
}
fn frames() -> u32 {
    3
}
fn capture(world: &World) -> serde_json::Value {
    let embedding_fault = world
        .embedding
        .as_ref()
        .and_then(|e| e.mapping_fault)
        .map(|f| json!({"kind":format!("{:?}",f.kind).to_lowercase(),"id":f.id}));
    let embedding = world.embedding.as_ref().map(|e| {
        json!({
            "centreCell": e.centre_cell,
            "centreRow": e.centre_row,
            "pressureMember": e.pressure_member,
            "rowActive": e.row_active,
            "rowTheta": e.row_theta,
        })
    });
    json!({"graph":world.state.topology.graph,"fields":world.state.fields,
        "receipt":world.receipt(),"resolution":world.resolution_receipt,
        "rigid":world.physical.as_ref().map(|p|&p.coupling_receipts),
        "pressureAuthority":world.embedding.as_ref().map(|e|&e.pressure_authority.receipt).unwrap_or(&world.pressure_authority.receipt),
        "scalarAuthority":world.scalar_authority.receipt,
        "embeddingFault":embedding_fault,"embedding":embedding})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let input: Input = serde_json::from_str(&text)?;
    let mut world =
        World::from_document(input.scene, input.production_options, input.world_options)?;
    let mut output = vec![capture(&world)];
    let mut stages = Vec::new();
    for frame in 0..input.frames {
        let mut observed = Vec::new();
        world.advance_with_observer(frame + 1, world.timestep_s, |name, graph, fields| {
            observed.push(json!({"name":name,"graph":graph,"fields":fields}));
        })?;
        stages.push(observed);
        output.push(capture(&world));
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"frames":output,"stages":stages}))?
    );
    Ok(())
}
