use std::io::{self, Read};

use fluid_core::initial_scene::SceneDocument;
use fluid_core::production_scene::{ProductionSceneOptions, ProductionTimeStep};
use fluid_core::world::WorldOptions;
use fluid_core::world3d::World3d;
use fluid_core::{
    assemble_pressure_rhs, collocate_velocity, enforce_inflow_faces, extend_velocity, force_faces,
    prepare_faces, prepare_pressure_topology, project_pressure_velocity, reconstruct_interfaces,
    solve_pressure,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    scene: SceneDocument,
    dt_s: f64,
    pressure_iterations: u32,
    pressure_relative_tolerance: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CellOutput {
    center: [f32; 3],
    density: f32,
    pressure: f32,
    velocity_fine_s: [f32; 3],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Output {
    pressure_iterations: u32,
    pressure_residual: f32,
    pressure_converged: bool,
    microsteps: usize,
    cells: Vec<CellOutput>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = String::new();
    io::stdin().read_to_string(&mut json)?;
    let input: Input = serde_json::from_str(&json)?;
    let mut world = World3d::from_document(
        input.scene,
        ProductionSceneOptions {
            dt_s: Some(input.dt_s),
            time_step: ProductionTimeStep::Scene,
            ..ProductionSceneOptions::default()
        },
        WorldOptions {
            pressure_iterations: input.pressure_iterations,
            pressure_relative_tolerance: input.pressure_relative_tolerance,
            tracer_budget: 0,
            ..WorldOptions::default()
        },
    )?;
    world.topology_frozen = true;
    world.state.topology.graph.initialize_spatial_owner_cache();
    let dt = input.dt_s as f32;
    let inflow = world.physical.begin_frame(
        &mut world.state,
        world.revision.time,
        input.dt_s,
        &mut world.source_ledger,
    )?;
    let graph = &mut world.state.topology.graph;
    let fields = &mut world.state.fields;
    fields.frame_dt = dt;
    extend_velocity(graph, fields, 8)?;
    prepare_faces(graph, fields, dt)?;
    force_faces(graph, fields, dt, fields.acceleration_fine, inflow);
    reconstruct_interfaces(graph, fields)?;
    let rows = prepare_pressure_topology(graph, fields);
    world.pressure_authority.publish(
        graph,
        fields,
        &rows.active,
        &rows.theta,
        graph.topology_generation,
        true,
    );
    assemble_pressure_rhs(graph, fields, &rows);
    world.pressure = solve_pressure(
        graph,
        fields,
        &rows,
        input.pressure_iterations,
        input.pressure_relative_tolerance,
        Some(&world.pressure_authority.execution_order),
    )?;
    project_pressure_velocity(graph, fields, &rows);
    enforce_inflow_faces(graph, fields, inflow);
    collocate_velocity(graph, fields);
    let graph = &world.state.topology.graph;
    let fields = &world.state.fields;
    let cells = graph
        .cells
        .iter()
        .map(|cell| {
            let id = cell.id as usize;
            CellOutput {
                center: cell.center,
                density: fields.density[id],
                pressure: fields.pressure[id],
                velocity_fine_s: [
                    fields.cell_velocity[3 * id],
                    fields.cell_velocity[3 * id + 1],
                    fields.cell_velocity[3 * id + 2],
                ],
            }
        })
        .collect();
    serde_json::to_writer(
        io::stdout(),
        &Output {
            pressure_iterations: world.pressure.iterations,
            pressure_residual: world.pressure.residual,
            pressure_converged: world.pressure.converged,
            microsteps: 0,
            cells,
        },
    )?;
    Ok(())
}
