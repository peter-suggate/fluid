//! Production authored scene -> native full-volume sparse scene.
use crate::atlas::{
    initialize_sparse_brick_atlas_from_scene, sparse_cm12_initial_active_brick_keys,
    SparseAdaptiveMassAtlas, CM12_PAPER_DT_S,
};
use crate::dynamic3d::VolumeFrame;
use crate::initial_scene::{lattice_dimensions, SceneDocument};
use crate::numerics3d::reconstruct_interfaces_3d;
use crate::physical3d::PhysicalContext3d;
use crate::production_scene::{
    atlas_brick_seed, cell_sizes, physical_scene, source_boundaries, ProductionSceneError,
    ProductionSceneOptions, ProductionTimeStep,
};
use crate::resolution::ResolutionPolicyOptions;
use crate::resolution3d::{static_boundary_floors_3d, ResolutionPolicyOptions3d};
use crate::scene::{compile_scene, SceneDescription, SceneState};
use crate::solid_world::fluid_solid_world_for_scene;

pub struct ProductionScene3dBundle {
    pub state: SceneState<3>,
    pub physical: PhysicalContext3d,
    pub resolution_options: ResolutionPolicyOptions,
    /// Generation-zero 3-D policy cache, including static SolidWorld evidence.
    pub resolution_options_3d: ResolutionPolicyOptions3d,
    pub source_atlas: SparseAdaptiveMassAtlas,
    pub document: SceneDocument,
    pub options: ProductionSceneOptions,
    pub dt_s: f64,
}

pub fn production_scene_3d(
    scene: SceneDocument,
    mut options: ProductionSceneOptions,
) -> Result<ProductionScene3dBundle, ProductionSceneError> {
    let dimensions = lattice_dimensions(&scene);
    let sizes = cell_sizes(&scene, dimensions);
    let h = scene.voxel_domain.finest_cell_size_m;
    if !h.is_finite() || h <= 0.0 || sizes.iter().any(|v| (*v - h).abs() > 1e-9 * h.max(1.0)) {
        return Err(ProductionSceneError::Invalid(
            "3-D production lattice must use cubic finest cells",
        ));
    }
    options.atlas.finest_dimensions = dimensions;
    if options.atlas.maximum_dt_s == 0.0 {
        options.atlas.maximum_dt_s = scene.numerics.max_dt_s;
    }
    if options.atlas.fixed_dt_s.is_none() {
        options.atlas.fixed_dt_s = Some(scene.numerics.fixed_dt_s);
    }
    let curved = scene
        .fluid
        .initial_liquid_volumes
        .iter()
        .any(|v| !matches!(v, crate::initial_scene::InitialLiquidVolume::Box { .. }));
    options.atlas.coarse_first_curvature_tolerance = if options.resolution.policy.coarse_first {
        Some(options.resolution.policy.curvature_tolerance)
    } else {
        None
    };
    options.atlas.initial_surface_coarsening_bias_rings =
        if options.resolution.policy.activity_signals && !curved {
            1
        } else {
            0
        };
    let atlas = initialize_sparse_brick_atlas_from_scene(&scene, &options.atlas)?;
    let active = sparse_cm12_initial_active_brick_keys(
        &scene,
        &atlas,
        if options.resolution.policy.coarse_first && scene.fluid.refinement_regions.is_empty() {
            2
        } else {
            1
        },
    );
    let bricks = atlas
        .bricks
        .iter()
        .map(|b| atlas_brick_seed(&atlas, b, active.contains(&b.key), None))
        .collect();
    let dt = options.dt_s.unwrap_or(match options.time_step {
        ProductionTimeStep::Paper => CM12_PAPER_DT_S,
        ProductionTimeStep::Scene => scene.numerics.fixed_dt_s,
    });
    let origin = [
        -0.5 * scene.container.width_m,
        0.0,
        -0.5 * scene.container.depth_m,
    ];
    let description = SceneDescription {
        schema_version: 1,
        dimension: 3,
        dimensions,
        cell_size_m: h as f32,
        origin_m: origin.map(|v| v as f32),
        dt_s: dt as f32,
        density_kg_m3: scene.fluid.density_kg_m3 as f32,
        gravity_m_s2: scene.fluid.gravity_m_s2.array().map(|v| v as f32),
        boundaries: source_boundaries(&scene),
        bricks,
        solids: vec![],
        liquids: vec![],
        raster_capacity: vec![],
        raster_density: vec![],
        raster_gamma: vec![],
        raster_velocity: vec![],
        velocity_x: vec![],
        velocity_y: vec![],
        aperture_x: vec![],
        aperture_y: vec![],
        solid_velocity_x: vec![],
        solid_velocity_y: vec![],
    };
    let mut state = compile_scene::<3>(description)?;
    let world = fluid_solid_world_for_scene(&scene);
    let [nx, ny, nz] = dimensions;
    state.topology.graph.solid_voxel_fraction = (0..nz)
        .flat_map(|z| {
            (0..ny).flat_map({
                let world = &world;
                move |y| {
                    (0..nx)
                        .map(move |x| world.sample([x as i32, y as i32, z as i32]).solid_fraction)
                }
            })
        })
        .collect();
    let initial = scene.fluid.initial_velocity_m_s.unwrap_or_default();
    for cell in &state.topology.graph.cells {
        let i = cell.id as usize;
        state.fields.cell_velocity[3 * i] = (initial.x / h) as f32;
        state.fields.cell_velocity[3 * i + 1] = (initial.y / h) as f32;
        state.fields.cell_velocity[3 * i + 2] = (initial.z / h) as f32;
    }
    for row in &state.topology.graph.rows {
        state.fields.face_velocity[row.id as usize] =
            [initial.x, initial.y, initial.z][row.axis as usize] as f32 / h as f32;
    }
    let frame = VolumeFrame {
        origin_m: origin,
        cell_size_m: h,
        dimensions,
    };
    let physical = PhysicalContext3d::new(physical_scene(&scene), frame, Some(world))?;
    physical.candidate_geometry(&mut state, 0.0, dt)?;
    // GPU generation zero applies moving-solid capacity before the first
    // conservative field publication. A body authored inside liquid cannot
    // leave more liquid than the receiving open volume.
    for (density, &capacity) in state.fields.density.iter_mut().zip(&state.fields.capacity) {
        *density = density.min(capacity);
    }
    reconstruct_interfaces_3d(&state.topology.graph, &mut state.fields)?;
    options.resolution.moving_rigid_bodies = !scene.rigid_bodies.is_empty();
    let mut resolution_options_3d = ResolutionPolicyOptions3d::production();
    resolution_options_3d.policy = options.resolution.policy.clone();
    resolution_options_3d.moving_rigid_bodies = !scene.rigid_bodies.is_empty();
    resolution_options_3d.static_boundary_floor_by_brick = static_boundary_floors_3d(
        &state.topology,
        resolution_options_3d.policy.detail_tolerance as f32,
    )
    .map_err(|_| ProductionSceneError::Invalid("invalid 3-D static solid evidence"))?;
    Ok(ProductionScene3dBundle {
        state,
        physical,
        resolution_options: options.resolution.clone(),
        resolution_options_3d,
        source_atlas: atlas,
        document: scene,
        options,
        dt_s: dt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_model::{FluidInflow, Vec3};
    use crate::sources::SourceLedger;

    fn water_box() -> SceneDocument {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../core/testdata/production-scene-golden.json"
        ))
        .unwrap();
        serde_json::from_value(fixture["cases"][0]["scene"].clone()).unwrap()
    }

    #[test]
    fn authored_scene_constructs_a_physical_three_dimensional_world() {
        let mut bundle =
            production_scene_3d(water_box(), ProductionSceneOptions::default()).unwrap();
        let graph = &bundle.state.topology.graph;
        assert_eq!(graph.dimension, 3);
        assert!(!graph.cells.is_empty());
        assert_eq!(
            bundle.state.fields.cell_velocity.len(),
            3 * graph.cells.len()
        );
        assert_eq!(
            graph.solid_voxel_fraction.len(),
            bundle
                .state
                .description
                .dimensions
                .into_iter()
                .map(|v| v as usize)
                .product::<usize>()
        );
        assert!(bundle.state.fields.density.iter().any(|&v| v > 0.0));
        assert!(bundle
            .state
            .fields
            .capacity
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
        let mut ledger = SourceLedger::default();
        let inflow = bundle
            .physical
            .begin_frame(&mut bundle.state, 0.0, bundle.dt_s, &mut ledger)
            .unwrap();
        assert!(inflow.into_iter().all(f32::is_finite));
        bundle.physical.finish_frame(&bundle.state).unwrap();
    }

    #[test]
    fn initial_rigid_intersection_never_publishes_density_above_capacity() {
        let mut scene = water_box();
        scene.rigid_bodies.push(
            serde_json::from_value(serde_json::json!({
              "id":"immersed","name":"Immersed sphere","shape":"sphere",
              "dimensions_m":{"x":0.4,"y":0.4,"z":0.4},"density_kg_m3":500,
              "position_m":{"x":-0.5,"y":0.4,"z":0},"orientation":{"w":1,"x":0,"y":0,"z":0},
              "linearVelocity_m_s":{"x":0,"y":0,"z":0},
              "angularVelocity_rad_s":{"x":0,"y":0,"z":0},"restitution":0.2,"friction":0.4
            }))
            .unwrap(),
        );
        let mut bundle = production_scene_3d(scene, ProductionSceneOptions::default()).unwrap();
        assert!(bundle
            .state
            .fields
            .density
            .iter()
            .zip(&bundle.state.fields.capacity)
            .all(|(&density, &capacity)| density <= capacity));
        assert!(bundle
            .state
            .fields
            .capacity
            .iter()
            .any(|&capacity| capacity < 1.0));
        bundle.physical.finish_frame(&bundle.state).unwrap();
        assert_eq!(bundle.physical.coupling_receipts.len(), 1);
        let exchange = bundle.physical.exchange[0].lanes;
        let receipt = &bundle.physical.coupling_receipts[0];
        assert!(receipt.displaced_volume_m3 > 0.0);
        assert_eq!(
            receipt.displaced_volume_m3,
            exchange[6] as f64 / 65536.0 * bundle.physical.frame.cell_size_m.powi(3)
        );
        let weight = exchange[11] as f64 / 65536.0;
        assert_eq!(
            receipt.mean_fluid_velocity_m_s.x,
            exchange[7] as f64 * 1e-4 / weight
        );
        assert_eq!(
            receipt.mean_fluid_velocity_m_s.y,
            exchange[8] as f64 * 1e-4 / weight
        );
        assert_eq!(
            receipt.mean_fluid_velocity_m_s.z,
            exchange[9] as f64 * 1e-4 / weight
        );
    }

    #[test]
    fn authored_three_dimensional_inflow_uses_the_gpu_volume_rate() {
        let mut bundle =
            production_scene_3d(water_box(), ProductionSceneOptions::default()).unwrap();
        bundle.physical.scene.fluid.inflow = Some(FluidInflow {
            center_m: Vec3 {
                x: 0.0,
                y: 0.4,
                z: 0.0,
            },
            radius_m: 0.1,
            length_m: 0.2,
            velocity_m_s: Vec3 {
                x: 0.2,
                y: 0.0,
                z: 0.0,
            },
            start_s: 0.0,
            end_s: 10.0,
            ramp_s: 0.0,
        });
        let mut ledger = SourceLedger::default();
        bundle
            .physical
            .begin_frame(&mut bundle.state, 0.0, bundle.dt_s, &mut ledger)
            .unwrap();
        let h = bundle.physical.frame.cell_size_m;
        let requested_rate = (0.2 * std::f64::consts::PI * 0.1f64.powi(2) / h.powi(3)) as f32;
        assert_eq!(
            ledger.event_requested,
            (requested_rate as f64 * bundle.dt_s) as f32
        );
    }
}
