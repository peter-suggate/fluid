//! Non-UI fixture runner for field/receipt comparisons against frozen goldens.
use fluid_core::{Fields, Graph};
use serde::Deserialize;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    graph: Graph,
    fields: Fields,
    #[serde(default)]
    support: fluid_core::presentation::RdfSupport,
    target: Option<Graph>,
    #[serde(default)]
    target_capacity: Vec<f32>,
    #[serde(default)]
    new_air: Vec<fluid_core::transfer::NewAirCoverage>,
    #[serde(default)]
    dt: f32,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let action = std::env::args().nth(1).ok_or("missing fixture stage")?;
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut fixture: Fixture = serde_json::from_str(&input)?;
    let output = match action.as_str() {
        "rdf" => {
            let cache = fluid_core::presentation::RdfTopology::compile(&fixture.graph)?;
            serde_json::to_string(&fluid_core::presentation::reconstruct_shared_rdf(
                &fixture.graph,
                &fixture.fields,
                &cache,
                &fixture.support,
            )?)?
        }
        "transfer" => serde_json::to_string(&fluid_core::transfer::transfer_fields(
            &fixture.graph,
            fixture.target.as_ref().ok_or("missing target")?,
            &fixture.fields,
            &fixture.target_capacity,
            &fixture.new_air,
        )?)?,
        "faces" | "extension" | "pressure" | "transport" | "clearance" => {
            let receipt = match action.as_str() {
                "faces" => {
                    fluid_core::prepare_faces(&fixture.graph, &mut fixture.fields, fixture.dt)?;
                    serde_json::Value::Null
                }
                "extension" => {
                    fluid_core::extend_velocity(&fixture.graph, &mut fixture.fields, 8)?;
                    serde_json::Value::Null
                }
                "pressure" => serde_json::to_value(fluid_core::project_velocity(
                    &fixture.graph,
                    &mut fixture.fields,
                    28,
                    1e-6,
                )?)?,
                "transport" => serde_json::to_value(fluid_core::transport_volume(
                    &fixture.graph,
                    &mut fixture.fields,
                    fixture.dt,
                )?)?,
                _ => {
                    fluid_core::publish_transport_characteristic_clearance(
                        &fixture.graph,
                        &mut fixture.fields,
                        fixture.dt,
                        None,
                        None,
                    )?;
                    serde_json::Value::Null
                }
            };
            serde_json::to_string(&serde_json::json!({"fields":fixture.fields,"receipt":receipt}))?
        }
        _ => return Err("unknown fixture stage".into()),
    };
    println!("{output}");
    Ok(())
}
