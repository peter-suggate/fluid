//! Browser boundary. Numerical algorithms live exclusively in fluid-core;
//! this crate validates serialized inputs and publishes owned snapshots.

use fluid_core::{Fields, Graph};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "threaded", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

fn error(value: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&value.to_string())
}
fn decode<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, JsValue> {
    serde_json::from_str(json).map_err(error)
}
fn encode<T: Serialize>(value: &T) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(error)
}

#[wasm_bindgen]
pub struct FluidWorld {
    inner: OwnedWorld,
}
enum OwnedWorld {
    Slice(fluid_core::world::World),
    Uniform(fluid_core::uniform_geometric::session::Session),
    Volume(fluid_core::world3d::World3d),
}
#[wasm_bindgen]
impl FluidWorld {
    pub fn from_scene(scene_json: &str, options_json: &str) -> Result<FluidWorld, JsValue> {
        let scene: serde_json::Value = decode(scene_json)?;
        let options: serde_json::Value = decode(options_json)?;
        if options.get("method").and_then(serde_json::Value::as_str) == Some("uniform-volume") {
            if options.get("dimension").and_then(serde_json::Value::as_u64) != Some(2) { return Err(error("Uniform Rust currently requires dimension 2")); }
            let seed=serde_json::from_value(options.get("uniformSeed").cloned().ok_or_else(||error("uniform initial fields are required"))?).map_err(error)?;
            let values=serde_json::from_value(options.get("methodValues").cloned().ok_or_else(||error("resolved uniform defaults are required"))?).map_err(error)?;
            let epoch=options.get("runEpoch").and_then(serde_json::Value::as_u64).and_then(|v|u32::try_from(v).ok()).ok_or_else(||error("invalid run epoch"))?;
            let sequence=options.get("commandSequence").and_then(serde_json::Value::as_u64).and_then(|v|u32::try_from(v).ok()).ok_or_else(||error("invalid command sequence"))?;
            return Ok(Self{inner:OwnedWorld::Uniform(fluid_core::uniform_geometric::session::Session::new(seed,serde_json::from_value(scene).map_err(error)?,values,epoch,sequence).map_err(error)?)});
        }
        if options.get("dimension").and_then(serde_json::Value::as_u64) == Some(3) {
            let mut world_options: fluid_core::world::WorldOptions =
                serde_json::from_value(options.clone()).map_err(error)?;
            let mut production: fluid_core::production_scene::ProductionSceneOptions =
                serde_json::from_value(
                    options
                        .get("production")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({})),
                )
                .map_err(error)?;
            if let Some(values) = options.get("methodValues") {
                fluid_core::runtime_options3d::apply_initial_values(
                    &mut production,
                    &mut world_options,
                    values,
                )
                .map_err(error)?;
            }
            let world = fluid_core::world3d::World3d::from_document(
                serde_json::from_value(scene).map_err(error)?,
                production,
                world_options,
            )
            .map_err(error)?;
            return Ok(Self {
                inner: OwnedWorld::Volume(world),
            });
        }
        let inner = if scene
            .get("schemaVersion")
            .is_some_and(serde_json::Value::is_string)
        {
            fluid_core::world::World::from_document(
                serde_json::from_value(scene).map_err(error)?,
                serde_json::from_value(
                    options
                        .get("production")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({})),
                )
                .map_err(error)?,
                serde_json::from_value(options).map_err(error)?,
            )
        } else {
            fluid_core::world::World::from_scene(
                serde_json::from_value(scene).map_err(error)?,
                serde_json::from_value(options).map_err(error)?,
            )
        }
        .map_err(error)?;
        Ok(Self {
            inner: OwnedWorld::Slice(inner),
        })
    }
    pub fn advance(&mut self, command_sequence: u32, dt_s: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            OwnedWorld::Slice(world) => world.advance(command_sequence, dt_s),
            OwnedWorld::Uniform(world) => world.advance(command_sequence, dt_s),
            OwnedWorld::Volume(world) => world.advance(command_sequence, dt_s),
        }
        .map_err(error)?;
        self.receipt()
    }
    pub fn receipt(&self) -> Result<String, JsValue> {
        match &self.inner {
            OwnedWorld::Slice(world) => encode(&world.receipt()),
            OwnedWorld::Uniform(world) => encode(&world.receipt()),
            OwnedWorld::Volume(world) => encode(&world.receipt()),
        }
    }
    pub fn apply_command(&mut self, command_json: &str) -> Result<String, JsValue> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Envelope {
            command_sequence: u32,
            run_epoch: u32,
        }
        let envelope: Envelope = decode(command_json)?;
        match &mut self.inner {
            OwnedWorld::Uniform(world) => world.apply_command(envelope.command_sequence, envelope.run_epoch, decode(command_json)?),
            OwnedWorld::Slice(world) => world.apply_command(
                envelope.command_sequence,
                envelope.run_epoch,
                decode(command_json)?,
            ),
            OwnedWorld::Volume(world) => world.apply_command(
                envelope.command_sequence,
                envelope.run_epoch,
                decode(command_json)?,
            ),
        }
        .map_err(error)?;
        self.receipt()
    }
    pub fn snapshot(&mut self, view_mask: u32) -> Result<Vec<u8>, JsValue> {
        let publication = match &mut self.inner {
            OwnedWorld::Slice(world) => world.snapshot(view_mask),
            OwnedWorld::Uniform(world) => world.snapshot(view_mask),
            OwnedWorld::Volume(world) => world.snapshot(view_mask),
        }
        .map_err(error)?;
        Ok(publication.to_vec())
    }
}

/// Whole-scene verification boundary for the uniform migration. Production
/// selection remains gated on the complete dimensional contract.
#[wasm_bindgen]
pub fn run_uniform_geometric_scene(request_json: &str) -> Result<String, JsValue> {
    let output=fluid_core::uniform_geometric::scene_runner::run(decode(request_json)?).map_err(error)?;
    encode(&output)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct StageOptions {
    maximum_iterations: u32,
    relative_tolerance: f32,
    extension_depth: u8,
    dt: f32,
    acceleration: [f32; 3],
    inflow_velocity: [f32; 3],
    transport_experiment: fluid_core::world::TransportExperiment,
}
impl Default for StageOptions {
    fn default() -> Self {
        Self {
            maximum_iterations: 28,
            relative_tolerance: 1e-5,
            extension_depth: 8,
            dt: 1.0 / 30.0,
            acceleration: [0.0; 3],
            inflow_velocity: [0.0; 3],
            transport_experiment: fluid_core::world::TransportExperiment::Baseline,
        }
    }
}

/// Field-level verification boundary. Production advances use the owned world;
/// this export makes native/Wasm numerical equivalence independently testable.
#[wasm_bindgen]
pub fn run_stage(
    stage: &str,
    graph_json: &str,
    fields_json: &str,
    options_json: &str,
) -> Result<String, JsValue> {
    let mut graph: Graph = decode(graph_json)?;
    let mut fields: Fields = decode(fields_json)?;
    let options: StageOptions = decode(options_json)?;
    fields.validate_for(&graph).map_err(error)?;
    let receipt = match stage {
        "pressure" => serde_json::to_value(
            fluid_core::project_velocity(
                &graph,
                &mut fields,
                options.maximum_iterations,
                options.relative_tolerance,
            )
            .map_err(error)?,
        )
        .map_err(error)?,
        "extension" => {
            fluid_core::extend_velocity(&graph, &mut fields, options.extension_depth)
                .map_err(error)?;
            serde_json::Value::Null
        }
        "reconstruct" => {
            fluid_core::reconstruct_interfaces(&graph, &mut fields).map_err(error)?;
            serde_json::Value::Null
        }
        "faces" => {
            fluid_core::prepare_faces(&graph, &mut fields, options.dt).map_err(error)?;
            serde_json::Value::Null
        }
        "force" => {
            fluid_core::force_faces(
                &mut graph,
                &mut fields,
                options.dt,
                options.acceleration,
                options.inflow_velocity,
            );
            serde_json::Value::Null
        }
        "clearance" => {
            fluid_core::publish_transport_characteristic_clearance(
                &graph,
                &mut fields,
                options.dt,
                None,
                None,
            )
            .map_err(error)?;
            serde_json::Value::Null
        }
        "transport"
            if options.transport_experiment
                == fluid_core::world::TransportExperiment::LevelSetVolume =>
        {
            fluid_core::extend_velocity(&graph, &mut fields, 8).map_err(error)?;
            let topology = fluid_core::presentation::RdfTopology::compile(&graph).map_err(error)?;
            let support = fluid_core::presentation::RdfSupport {
                solid_fraction: (!graph.solid_voxel_fraction.is_empty())
                    .then(|| graph.solid_voxel_fraction.clone()),
                ..Default::default()
            };
            // The stage protocol carries fields but no persistent surface, so this
            // one-shot entry point bootstraps a shared fine-grid level set directly
            // from occupancy. It must not introduce a volume-fitted PLIC surface.
            let surface = fluid_core::levelset_surface::initialize_from_volume(&graph, &fields)
                .map_err(error)?;
            let mut phi = fluid_core::levelset_surface::cell_phi(&graph, &surface)
                .map_err(error)?;
            let (_, receipt) = fluid_core::levelset_volume::advance(
                &graph, &mut fields, &surface, &topology, &support, &mut phi, options.dt,
            ).map_err(error)?;
            serde_json::to_value(receipt).map_err(error)?
        }
        "transport" => match options.transport_experiment.cellwise_mode() {
            None => serde_json::to_value(
                fluid_core::transport_volume(&graph, &mut fields, options.dt).map_err(error)?,
            )
            .map_err(error)?,
            Some(fluid_core::world::CellwiseTransportMode::Probe) => {
                let remap_options = fluid_core::adaptive_remap::CellwiseRemapOptions {
                    closure: options.transport_experiment.closure(),
                    commit_material: false,
                    trace_segments: options.transport_experiment.trace_segments(),
                    edge_samples: options.transport_experiment.edge_samples(),
                    ..Default::default()
                };
                serde_json::to_value(
                    fluid_core::adaptive_remap::transport_volume_cellwise_with_commit(
                        &graph,
                        &mut fields,
                        options.dt,
                        remap_options,
                        |_, _| Ok(()),
                    )
                    .map_err(error)?,
                )
                .map_err(error)?
            }
            Some(fluid_core::world::CellwiseTransportMode::Remap) => {
                let remap_options = fluid_core::adaptive_remap::CellwiseRemapOptions {
                    closure: options.transport_experiment.closure(),
                    trace_segments: options.transport_experiment.trace_segments(),
                    edge_samples: options.transport_experiment.edge_samples(),
                    ..Default::default()
                };
                serde_json::to_value(
                    fluid_core::adaptive_remap::transport_volume_cellwise_with_commit(
                        &graph,
                        &mut fields,
                        options.dt,
                        remap_options,
                        |_, _| Ok(()),
                    )
                    .map_err(error)?,
                )
                .map_err(error)?
            }
        },
        _ => return Err(error(format!("unknown numerical stage {stage}"))),
    };
    encode(&serde_json::json!({ "fields": fields, "receipt": receipt, "graph":graph }))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PressureCsr {
    row_offsets: Vec<u32>,
    row_cells: Vec<u32>,
    row_coefficients: Vec<f32>,
    row_weights: Vec<f32>,
    execution_order: Option<Vec<u32>>,
}

#[wasm_bindgen]
pub fn run_rdf(graph_json: &str, fields_json: &str, support_json: &str) -> Result<String, JsValue> {
    let graph: Graph = decode(graph_json)?;
    let fields: Fields = decode(fields_json)?;
    let support: fluid_core::presentation::RdfSupport = decode(support_json)?;
    let topology = fluid_core::presentation::RdfTopology::compile(&graph).map_err(error)?;
    encode(
        &fluid_core::presentation::reconstruct_shared_rdf(&graph, &fields, &topology, &support)
            .map_err(error)?,
    )
}

/// Executes the literal production pressure recurrence on a serialized GᵀWG
/// operator. Used by the non-browser parity suite, including mixed seams.
#[wasm_bindgen]
pub fn run_pressure(
    diagonal: &[f32],
    rhs: &[f32],
    pressure: &[f32],
    member: &[u8],
    graph_json: &str,
    options_json: &str,
) -> Result<String, JsValue> {
    let graph: PressureCsr = decode(graph_json)?;
    let options: StageOptions = decode(options_json)?;
    if graph.row_offsets.len() != graph.row_weights.len() + 1
        || graph.row_offsets.first() != Some(&0)
        || graph.row_offsets.last().copied() != Some(graph.row_cells.len() as u32)
        || graph.row_coefficients.len() != graph.row_cells.len()
        || graph.row_offsets.windows(2).any(|w| w[0] > w[1])
        || graph.row_cells.iter().any(|&i| i as usize >= rhs.len())
        || graph
            .row_weights
            .iter()
            .chain(&graph.row_coefficients)
            .any(|v| !v.is_finite())
    {
        return Err(error("invalid pressure CSR operator"));
    }
    let result = fluid_core::solve_pressure_pcg(
        diagonal,
        rhs,
        pressure,
        member,
        graph.execution_order.as_deref(),
        options.maximum_iterations,
        options.relative_tolerance,
        |input, output| {
            output.fill(0.0);
            for (row, &weight) in graph.row_weights.iter().enumerate() {
                let range = graph.row_offsets[row] as usize..graph.row_offsets[row + 1] as usize;
                let mut gradient = 0.0_f32;
                for at in range.clone() {
                    gradient += graph.row_coefficients[at] * input[graph.row_cells[at] as usize];
                }
                let weighted = weight * gradient;
                for at in range {
                    let cell = graph.row_cells[at] as usize;
                    output[cell] += graph.row_coefficients[at] * weighted;
                }
            }
        },
    )
    .map_err(error)?;
    encode(&result)
}
