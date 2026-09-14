use fluid_core::{
    initial_scene::SceneDocument,
    production_scene::ProductionSceneOptions,
    world::{TransportExperiment, World, WorldOptions},
};

const DT: f64 = 1.0 / 30.0;
const GRAVITY_FINE: f64 = 9.80665 / 0.025;

fn mini32_scene() -> SceneDocument {
    serde_json::from_value(serde_json::json!({
        "schemaVersion": "2.0.0",
        "sceneId": "minimal-power-dam-break-32",
        "container": {
            "width_m": 0.8,
            "height_m": 0.8,
            "depth_m": 0.8,
            "fillFraction": 0.359375,
            "top": "closed"
        },
        "solidVoxels": [
            {"operation":"fill", "minimum":[0,-1,0], "maximumExclusive":[32,0,32], "materialId":1},
            {"operation":"fill", "minimum":[-1,0,0], "maximumExclusive":[0,32,32], "materialId":1},
            {"operation":"fill", "minimum":[32,0,0], "maximumExclusive":[33,32,32], "materialId":1},
            {"operation":"fill", "minimum":[0,0,-1], "maximumExclusive":[32,32,0], "materialId":1},
            {"operation":"fill", "minimum":[0,0,32], "maximumExclusive":[32,32,33], "materialId":1},
            {"operation":"fill", "minimum":[0,32,0], "maximumExclusive":[32,33,32], "materialId":1}
        ],
        "voxelDomain": {"finestCellSize_m":0.025, "brickSize_cells":8},
        "fluid": {
            "density_kg_m3": 998.2,
            "dynamicViscosity_Pa_s": 0.001002,
            "surfaceTension_N_m": 0.0,
            "gravity_m_s2": {"x":0.0, "y":-9.80665, "z":0.0},
            "initialCondition": "dam-break"
        },
        "numerics": {"fixedDt_s":DT, "maxDt_s":DT},
        "rigidBodies": []
    }))
    .unwrap()
}

fn energy(world: &World) -> (f64, f64) {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .fold((0.0, 0.0), |(kinetic, potential), cell| {
            let id = cell.id as usize;
            let measure = world.state.fields.density[id] as f64 * cell.measure as f64;
            let velocity = &world.state.fields.cell_velocity[2 * id..2 * id + 2];
            (
                kinetic
                    + 0.5
                        * measure
                        * ((velocity[0] as f64).powi(2) + (velocity[1] as f64).powi(2)),
                potential + measure * GRAVITY_FINE * cell.center[1] as f64,
            )
        })
}

fn liquid_measure(world: &World) -> f64 {
    world
        .state
        .topology
        .graph
        .cells
        .iter()
        .map(|cell| {
            world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64
        })
        .sum()
}

#[test]
fn mini32_level_set_volume_retains_impact_window_energy_without_substeps() {
    let production: ProductionSceneOptions = serde_json::from_value(serde_json::json!({
        "dtS": DT,
        "timeStep": "paper"
    }))
    .unwrap();
    let mut world = World::from_document(
        mini32_scene(),
        production,
        WorldOptions {
            pressure_iterations: 256,
            pressure_relative_tolerance: 1.0e-6,
            transport_experiment: TransportExperiment::LevelSetVolume,
            ..Default::default()
        },
    )
    .unwrap();
    let initial_measure = liquid_measure(&world);
    let initial_energy = {
        let (kinetic, potential) = energy(&world);
        kinetic + potential
    };
    assert!((initial_measure - 588.8).abs() <= 1.0e-4, "unexpected mini32 seed");

    let mut impact_kinetic = 0.0;
    let mut impact_samples = 0;
    for frame in 1..=20 {
        world.advance(frame, DT).unwrap();
        assert_eq!(world.receipt().microsteps, 0, "frame {frame} used transport substeps");
        assert!(world.state.fields.fault.is_none(), "frame {frame} numerical fault");
        assert!(world.state.fields.density.iter().all(|value| value.is_finite()));
        assert!(world.state.fields.cell_velocity.iter().all(|value| value.is_finite()));
        let measure = liquid_measure(&world);
        assert!(
            ((measure - initial_measure) / initial_measure).abs() <= 2.0e-8,
            "frame {frame} mass drifted: {measure} vs {initial_measure}"
        );
        let (kinetic, potential) = energy(&world);
        let total = kinetic + potential;
        assert!(
            total.is_finite() && (0.4 * initial_energy..=1.1 * initial_energy).contains(&total),
            "frame {frame} energy left the bounded diagnostic envelope: {total}"
        );
        if frame >= 10 {
            impact_kinetic += kinetic;
            impact_samples += 1;
        }
    }
    let mean_impact_kinetic = impact_kinetic / impact_samples as f64;
    assert!(
        mean_impact_kinetic >= 90_000.0,
        "frames 10-20 mean kinetic energy regressed: {mean_impact_kinetic}"
    );
}
