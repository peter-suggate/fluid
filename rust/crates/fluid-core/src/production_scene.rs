//! Production authored scene -> exact native centre-Z sparse scene.

use crate::atlas::{
    initialize_sparse_brick_atlas_from_scene, sparse_brick_key,
    sparse_cm12_initial_active_brick_keys, AtlasError, SparseAdaptiveMassAtlas,
    SparseBrickAtlasInitializationOptions, CM12_PAPER_DT_S,
};
use crate::embedding::{EmbeddingOptions, PressureEmbedding};
use crate::geometry::BoundaryMode;
use crate::initial_liquid::{
    base_initial_liquid_fraction_at_cell, initial_liquid_fraction_at_cell,
};
use crate::initial_scene::{lattice_dimensions, rasterize_slice_rigid_geometry, SceneDocument};
use crate::numerics::{collocate_velocity, reconstruct_interfaces};
use crate::physical::PhysicalContext;
use crate::resolution::{ResolutionPolicyOptions, ResolutionRegion};
use crate::scene::{compile_scene, SceneDescription, SceneError, SceneState};
use crate::scene_model::{Container, FluidPhysics, PhysicalScene, SliceFrame};
use crate::solid_world::fluid_solid_world_for_scene;
use crate::topology::{BrickSeed, CompiledTopology, TopologyError};
use crate::types::{Fields, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProductionTimeStep {
    #[default]
    Paper,
    Scene,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProductionSceneOptions {
    pub dt_s: Option<f64>,
    pub time_step: ProductionTimeStep,
    pub atlas: SparseBrickAtlasInitializationOptions,
    pub resolution: ResolutionPolicyOptions,
}
impl Default for ProductionSceneOptions {
    fn default() -> Self {
        Self {
            dt_s: None,
            time_step: ProductionTimeStep::Paper,
            atlas: SparseBrickAtlasInitializationOptions::default(),
            resolution: ResolutionPolicyOptions::production(),
        }
    }
}

pub struct ProductionSceneBundle {
    pub state: SceneState<2>,
    pub physical: PhysicalContext,
    pub embedding: Option<PressureEmbedding>,
    pub resolution_options: ResolutionPolicyOptions,
    pub source_atlas: SparseAdaptiveMassAtlas,
    pub source_topology: CompiledTopology<3>,
    pub source_fields: Fields,
    pub document: SceneDocument,
    pub options: ProductionSceneOptions,
    /// Resolved step before numerical kernels convert it to f32.
    pub dt_s: f64,
    /// Finest centre-plane material image in the production canvas layout.
    pub material_id: Vec<u16>,
    /// Authored out-of-plane initial velocity, in metres/second.
    pub velocity_z: Vec<f32>,
}

#[derive(Debug)]
pub enum ProductionSceneError {
    Atlas(AtlasError),
    Scene(SceneError),
    Topology(TopologyError),
    Validation(ValidationError),
    Invalid(&'static str),
}
impl std::fmt::Display for ProductionSceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ProductionSceneError {}
impl From<AtlasError> for ProductionSceneError {
    fn from(v: AtlasError) -> Self {
        Self::Atlas(v)
    }
}
impl From<SceneError> for ProductionSceneError {
    fn from(v: SceneError) -> Self {
        Self::Scene(v)
    }
}
impl From<TopologyError> for ProductionSceneError {
    fn from(v: TopologyError) -> Self {
        Self::Topology(v)
    }
}
impl From<ValidationError> for ProductionSceneError {
    fn from(v: ValidationError) -> Self {
        Self::Validation(v)
    }
}

pub(crate) fn cell_sizes(scene: &SceneDocument, d: [u32; 3]) -> [f64; 3] {
    [
        scene.container.width_m / d[0] as f64,
        scene.container.height_m / d[1] as f64,
        scene.container.depth_m / d[2] as f64,
    ]
}
pub(crate) fn physical_scene(scene: &SceneDocument) -> PhysicalScene {
    PhysicalScene {
        schema_version: scene.schema_version.clone(),
        scene_id: scene.scene_id.clone(),
        container: Container {
            width_m: scene.container.width_m,
            height_m: scene.container.height_m,
            depth_m: scene.container.depth_m,
        },
        fluid: FluidPhysics {
            density_kg_m3: scene.fluid.density_kg_m3,
            dynamic_viscosity_pa_s: scene.fluid.dynamic_viscosity_pa_s,
            surface_tension_n_m: scene.fluid.surface_tension_n_m,
            gravity_m_s2: scene.fluid.gravity_m_s2,
            inflow: scene.fluid.inflow,
        },
        rigid_bodies: scene.rigid_bodies.clone(),
    }
}
pub(crate) fn source_boundaries(scene: &SceneDocument) -> [BoundaryMode; 6] {
    [
        BoundaryMode::Closed,
        BoundaryMode::Closed,
        BoundaryMode::Closed,
        if scene.container.top == "open" {
            BoundaryMode::Open
        } else {
            BoundaryMode::Closed
        },
        if scene.container.depth_boundary.as_deref() == Some("symmetry") {
            BoundaryMode::Symmetry
        } else {
            BoundaryMode::Closed
        },
        if scene.container.depth_boundary.as_deref() == Some("symmetry") {
            BoundaryMode::Symmetry
        } else {
            BoundaryMode::Closed
        },
    ]
}
fn reduced_boundaries(scene: &SceneDocument) -> [BoundaryMode; 6] {
    // Source +Y is reflected into canvas/topology +Y exactly once.
    [
        BoundaryMode::Closed,
        BoundaryMode::Closed,
        if scene.container.top == "open" {
            BoundaryMode::Open
        } else {
            BoundaryMode::Closed
        },
        BoundaryMode::Closed,
        BoundaryMode::Closed,
        BoundaryMode::Closed,
    ]
}
pub(crate) fn atlas_brick_seed(
    atlas: &SparseAdaptiveMassAtlas,
    b: &crate::atlas::SparseAdaptiveMassBrick,
    active: bool,
    plane_z: Option<u32>,
) -> BrickSeed {
    let (density, gamma) = if let Some(z) = plane_z {
        let scale = atlas.brick_fine_resolution * b.span_bricks / b.resolution;
        let z0 = b.coordinate[2] as u32 * atlas.brick_fine_resolution;
        let lz = ((z - z0) / scale).min(b.resolution - 1);
        let mut d = vec![0.; b.resolution.pow(2) as usize];
        let mut g = vec![1.; d.len()];
        for y in 0..b.resolution {
            for x in 0..b.resolution {
                let a = (x + b.resolution * (y + b.resolution * lz)) as usize;
                let q = (x + b.resolution * y) as usize;
                d[q] = b.density[a] as f32;
                g[q] = b.gamma[a] as f32;
            }
        }
        (d, g)
    } else {
        (
            b.density.iter().map(|&v| v as f32).collect(),
            b.gamma.iter().map(|&v| v as f32).collect(),
        )
    };
    BrickSeed {
        id: b.key,
        key: b.key,
        coordinate: b.coordinate,
        span_bricks: b.span_bricks,
        resolution: b.resolution as u8,
        active,
        density,
        gamma,
        refinement_region_scale: None,
    }
}

/// Extrude accepted XY brick activity/rungs through the immutable source-Z layout.
/// This is the native counterpart of `createSlicePressureEmbedding`'s candidate rebuild.
pub fn rebuild_source_topology(
    source: &SparseAdaptiveMassAtlas,
    reduced: &CompiledTopology<2>,
    boundaries: [BoundaryMode; 6],
) -> Result<CompiledTopology<3>, ProductionSceneError> {
    let mut bricks = Vec::new();
    for xy in &reduced.bricks {
        let xy = &xy.seed;
        if !xy.active {
            continue;
        }
        let existing: Vec<_> = source
            .bricks
            .iter()
            .filter(|b| {
                b.coordinate[0] == xy.coordinate[0]
                    && b.coordinate[1] == xy.coordinate[1]
                    && b.span_bricks == xy.span_bricks
            })
            .collect();
        if existing.is_empty() {
            for z in (0..source.brick_dimensions[2]).step_by(xy.span_bricks as usize) {
                let coordinate = [xy.coordinate[0], xy.coordinate[1], z as i32];
                let key = sparse_brick_key(coordinate, source.brick_dimensions);
                let n = (xy.resolution as usize).pow(3);
                bricks.push(BrickSeed {
                    id: key,
                    key,
                    coordinate,
                    span_bricks: xy.span_bricks,
                    resolution: xy.resolution,
                    active: true,
                    density: vec![0.; n],
                    gamma: vec![1.; n],
                    refinement_region_scale: None,
                });
            }
        } else {
            for b in existing {
                let mut seed = atlas_brick_seed(source, b, true, None);
                if seed.resolution != xy.resolution {
                    let n = (xy.resolution as usize).pow(3);
                    seed.resolution = xy.resolution;
                    seed.density = vec![0.; n];
                    seed.gamma = vec![1.; n];
                }
                bricks.push(seed);
            }
        }
    }
    Ok(crate::topology::compile_topology::<3>(
        crate::topology::TopologySeed {
            dimensions: source.dimensions,
            generation: reduced.graph.topology_generation,
            sparse_air_phi: 0.5,
            boundaries,
            bricks,
        },
    )?)
}

/// Rebuild the retained three-dimensional graph and its centre-plane mapping
/// after an accepted 2-D topology transition. Source Z pages and stable keys
/// remain those of the generation-zero atlas.
pub fn rebuild_source_embedding(
    document: &SceneDocument,
    source: &SparseAdaptiveMassAtlas,
    reduced: &SceneState<2>,
    previous: Option<&PressureEmbedding>,
) -> Result<(CompiledTopology<3>, PressureEmbedding), ProductionSceneError> {
    let mut topology =
        rebuild_source_topology(source, &reduced.topology, source_boundaries(document))?;
    let world = fluid_solid_world_for_scene(document);
    let [nx, ny, nz] = source.dimensions;
    topology.graph.solid_voxel_fraction = (0..nz)
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
    let center_cell_z = (nz / 2) as i32;
    let embedding = PressureEmbedding::new(
        topology.graph.clone(),
        &reduced.topology.graph,
        EmbeddingOptions {
            center_cell_z,
            z_boundary_omitted: false,
            sparse_air_phi: 0.5,
            source_generation: reduced.topology.graph.topology_generation,
            solid_world: true,
        },
        Some(&reduced.fields),
        previous,
    );
    Ok((topology, embedding))
}

fn refinement_regions(scene: &SceneDocument, h: f64, center_z: f64) -> Vec<ResolutionRegion> {
    scene
        .fluid
        .refinement_regions
        .iter()
        .filter_map(|v| {
            let min = v.get("min_m")?;
            let max = v.get("max_m")?;
            let n = |p: &serde_json::Value, k: &str| p.get(k)?.as_f64();
            if center_z < n(min, "z")? || center_z >= n(max, "z")? {
                return None;
            }
            let floor = v.get("minimumCellSize_cells")?.as_u64()? as u8;
            let ceiling = v
                .get("maximumCellSize_cells")
                .and_then(|v| v.as_u64())
                .map(|v| v as u8);
            Some(ResolutionRegion {
                minimum_fine: [
                    (n(min, "x")? + 0.5 * scene.container.width_m) / h,
                    n(min, "y")? / h,
                ],
                maximum_fine: [
                    (n(max, "x")? + 0.5 * scene.container.width_m) / h,
                    n(max, "y")? / h,
                ],
                minimum_cell_width: floor,
                maximum_cell_width: ceiling,
            })
        })
        .collect()
}

pub fn production_scene(
    scene: SceneDocument,
    mut options: ProductionSceneOptions,
) -> Result<ProductionSceneBundle, ProductionSceneError> {
    let dims = lattice_dimensions(&scene);
    let [nx, ny, nz] = dims;
    let z = nz / 2;
    let sizes = cell_sizes(&scene, dims);
    let h = scene.voxel_domain.finest_cell_size_m;
    if !h.is_finite() || h <= 0.0 {
        return Err(ProductionSceneError::Invalid("source cell size"));
    }
    let world = fluid_solid_world_for_scene(&scene);
    options.atlas.finest_dimensions = dims;
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
    let regions = refinement_regions(&scene, h, 0.0);
    let reduced_bricks = atlas
        .bricks
        .iter()
        .filter(|b| {
            let z0 = b.coordinate[2] * atlas.brick_fine_resolution as i32;
            z as i32 >= z0 && (z as i32) < z0 + (b.span_bricks * atlas.brick_fine_resolution) as i32
        })
        .map(|b| {
            let mut seed = atlas_brick_seed(&atlas, b, active.contains(&b.key), Some(z));
            let edge = (atlas.brick_fine_resolution * b.span_bricks) as f64;
            let lo = [
                b.coordinate[0] as f64 * atlas.brick_fine_resolution as f64,
                b.coordinate[1] as f64 * atlas.brick_fine_resolution as f64,
            ];
            seed.refinement_region_scale = Some(
                regions
                    .iter()
                    .filter(|r| {
                        lo[0] < r.maximum_fine[0]
                            && lo[0] + edge > r.minimum_fine[0]
                            && lo[1] < r.maximum_fine[1]
                            && lo[1] + edge > r.minimum_fine[1]
                    })
                    .fold(1u8, |v, r| v.max(r.minimum_cell_width)) as f32,
            );
            seed
        })
        .collect::<Vec<_>>();

    let mut capacity = vec![0.; (nx * ny) as usize];
    let mut density = vec![0.; capacity.len()];
    let mut material_id = vec![0; capacity.len()];
    let mut solid_plane = vec![0.; capacity.len()];
    for y in 0..ny {
        for x in 0..nx {
            let i = (x + nx * y) as usize;
            let s = world.sample([x as i32, y as i32, z as i32]);
            let open = 1. - s.solid_fraction;
            solid_plane[i] = s.solid_fraction;
            material_id[(x + nx * (ny - 1 - y)) as usize] = s.material_id;
            let base =
                base_initial_liquid_fraction_at_cell(&scene, [x as i32, y as i32, z as i32], dims);
            density[i] = open
                * (initial_liquid_fraction_at_cell(
                    &scene,
                    [x as i32, y as i32, z as i32],
                    dims,
                    base,
                ) as f32);
            capacity[i] = open;
        }
    }
    let phys = physical_scene(&scene);
    let rigid = rasterize_slice_rigid_geometry(&phys, [nx, ny], sizes);
    // Rigid raster is canvas-ordered; apply its multiplier to source-ordered scalar capacity.
    for y in 0..ny as usize {
        for x in 0..nx as usize {
            capacity[x + nx as usize * y] *=
                rigid.capacity_multiplier[(ny as usize - 1 - y) * nx as usize + x];
        }
    }
    let initial = scene.fluid.initial_velocity_m_s.unwrap_or_default();
    let mut vx = vec![initial.x as f32; ((nx + 1) * ny) as usize];
    let mut vy = vec![-initial.y as f32; (nx * (ny + 1)) as usize];
    let mut ax = vec![0.; vx.len()];
    let mut ay = vec![0.; vy.len()];
    for cy in 0..ny as usize {
        let sy = ny as usize - 1 - cy;
        for x in 0..=nx as usize {
            let i = cy * (nx as usize + 1) + x;
            let open = if x == 0 || x == nx as usize {
                0.
            } else {
                1. - world
                    .sample([(x - 1) as i32, sy as i32, z as i32])
                    .solid_fraction
                    .max(world.sample([x as i32, sy as i32, z as i32]).solid_fraction)
            };
            ax[i] = open * rigid.aperture_x_multiplier[i];
        }
    }
    for syf in 0..=ny as usize {
        let cy = ny as usize - syf;
        for x in 0..nx as usize {
            let i = cy * nx as usize + x;
            let open = if syf == 0 || (syf == ny as usize && scene.container.top != "open") {
                0.
            } else if syf == ny as usize {
                1.
            } else {
                1. - world
                    .sample([x as i32, syf as i32 - 1, z as i32])
                    .solid_fraction
                    .max(
                        world
                            .sample([x as i32, syf as i32, z as i32])
                            .solid_fraction,
                    )
            };
            ay[i] = open * rigid.aperture_y_multiplier[i];
        }
    }
    let dt = options.dt_s.unwrap_or(match options.time_step {
        ProductionTimeStep::Paper => CM12_PAPER_DT_S,
        ProductionTimeStep::Scene => scene.numerics.fixed_dt_s,
    });
    let desc = SceneDescription {
        schema_version: 1,
        dimension: 2,
        dimensions: [nx, ny, 1],
        cell_size_m: h as f32,
        origin_m: [(-0.5 * scene.container.width_m) as f32, 0., 0.],
        dt_s: dt as f32,
        density_kg_m3: scene.fluid.density_kg_m3 as f32,
        gravity_m_s2: [
            scene.fluid.gravity_m_s2.x as f32,
            scene.fluid.gravity_m_s2.y as f32,
            0.,
        ],
        boundaries: reduced_boundaries(&scene),
        bricks: reduced_bricks,
        solids: vec![],
        liquids: vec![],
        raster_capacity: capacity,
        raster_density: density,
        raster_gamma: vec![],
        raster_velocity: vec![],
        velocity_x: std::mem::take(&mut vx),
        velocity_y: std::mem::take(&mut vy),
        aperture_x: ax,
        aperture_y: ay,
        solid_velocity_x: rigid
            .solid_velocity_x_fine
            .iter()
            .map(|v| v * h as f32)
            .collect(),
        solid_velocity_y: rigid
            .solid_velocity_y_fine
            .iter()
            .map(|v| v * h as f32)
            .collect(),
    };
    let mut state = compile_scene::<2>(desc)?;
    state.fields.acceleration_fine = [
        (scene.fluid.gravity_m_s2.x / h) as f32,
        (scene.fluid.gravity_m_s2.y / h) as f32,
        0.0,
    ];
    state.topology.graph.solid_voxel_fraction = solid_plane;
    collocate_velocity(&state.topology.graph, &mut state.fields);
    reconstruct_interfaces(&state.topology.graph, &mut state.fields)?;
    options.resolution.refinement_regions = regions;
    options.resolution.moving_rigid_bodies = !scene.rigid_bodies.is_empty();
    options.resolution.static_boundary_floor_by_brick = state
        .topology
        .bricks
        .iter()
        .filter_map(|record| {
            let brick = &record.seed;
            let edge = atlas.brick_fine_resolution * brick.span_bricks;
            let x0 = (brick.coordinate[0].max(0) as u32) * atlas.brick_fine_resolution;
            let y0 = (brick.coordinate[1].max(0) as u32) * atlas.brick_fine_resolution;
            let touches_static = (y0..(y0 + edge).min(ny)).any(|y| {
                (x0..(x0 + edge).min(nx))
                    .any(|x| state.topology.graph.solid_voxel_fraction[(x + nx * y) as usize] > 0.0)
            });
            touches_static.then_some((brick.key, (edge / brick.resolution as u32) as u8))
        })
        .collect();
    let source_topology =
        rebuild_source_topology(&atlas, &state.topology, source_boundaries(&scene))?;
    let source_desc = SceneDescription {
        schema_version: 1,
        dimension: 3,
        dimensions: dims,
        cell_size_m: h as f32,
        origin_m: [
            (-0.5 * scene.container.width_m) as f32,
            0.,
            (-0.5 * scene.container.depth_m) as f32,
        ],
        dt_s: dt as f32,
        density_kg_m3: scene.fluid.density_kg_m3 as f32,
        gravity_m_s2: scene.fluid.gravity_m_s2.array().map(|v| v as f32),
        boundaries: source_boundaries(&scene),
        bricks: source_topology
            .bricks
            .iter()
            .map(|r| r.seed.clone())
            .collect(),
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
    let mut source_state = compile_scene::<3>(source_desc)?;
    source_state.topology.graph.solid_voxel_fraction = (0..nz)
        .flat_map(|zz| {
            (0..ny).flat_map({
                let world = &world;
                move |yy| {
                    (0..nx).map(move |xx| {
                        world
                            .sample([xx as i32, yy as i32, zz as i32])
                            .solid_fraction
                    })
                }
            })
        })
        .collect();
    let frame = SliceFrame {
        origin_x: -0.5 * scene.container.width_m,
        origin_y: 0.,
        center_z: 0.,
        source_cell_size: h,
        center_cell_z: z as i32,
        source_dimensions: dims,
    };
    let embedding = if scene.container.depth_boundary.as_deref() == Some("symmetry") {
        Some(PressureEmbedding::new(
            source_state.topology.graph.clone(),
            &state.topology.graph,
            EmbeddingOptions {
                center_cell_z: z as i32,
                z_boundary_omitted: false,
                sparse_air_phi: 0.5,
                source_generation: atlas.generation,
                solid_world: true,
            },
            Some(&state.fields),
            None,
        ))
    } else {
        None
    };
    let physical = PhysicalContext::new(phys, frame, Some(world), sizes[0] * sizes[1] * sizes[2])?;
    let velocity_z = vec![initial.z as f32; (nx * ny) as usize];
    Ok(ProductionSceneBundle {
        state,
        physical,
        embedding,
        resolution_options: options.resolution.clone(),
        source_atlas: atlas,
        source_topology: source_state.topology,
        source_fields: source_state.fields,
        document: scene,
        options,
        dt_s: dt,
        material_id,
        velocity_z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Fixture {
        cases: Vec<Case>,
        transition: Transition,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Case {
        id: String,
        scene: SceneDocument,
        initialization: Init,
        expected: Expected,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Init {
        dt: f64,
        brick_fine_resolution: u32,
        maximum_macro_span_bricks: Option<u32>,
        surface_fine_rings: u32,
        activity_signals: bool,
        coarse_first: bool,
        curvature_tolerance: f64,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Expected {
        dimensions: [u32; 2],
        source_dimensions: [u32; 3],
        center_cell_z: i32,
        source_cell_size: f64,
        bricks: Vec<ExpectedBrick>,
        cells: usize,
        rows: usize,
        subfaces: usize,
        density_hash: String,
        capacity_hash: String,
        face_velocity_hash: String,
        normal_hash: String,
        offset_hash: String,
        material_hash: String,
        solid_hash: String,
        cell_graph_hash: String,
        row_float_hash: String,
        row_word_hash: String,
        subface_float_hash: String,
        subface_word_hash: String,
        incidence_hash: String,
        subface_incidence_hash: String,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct ExpectedBrick {
        id: u32,
        key: u32,
        coordinate: [i32; 2],
        span_bricks: u32,
        resolution: u8,
        active: bool,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Transition {
        scene_id: String,
        target_key: u32,
        target_resolution: u8,
        generation: u32,
        cells: usize,
        rows: usize,
        bricks: Vec<SourceBrick>,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct SourceBrick {
        key: u32,
        coordinate: [i32; 3],
        span_bricks: u32,
        resolution: u8,
    }
    fn hash(bytes: &[u8]) -> String {
        let mut h = 14_695_981_039_346_656_037u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(1_099_511_628_211)
        }
        format!("{h:016x}")
    }
    fn f32_hash(v: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend(x.to_bits().to_le_bytes())
        }
        hash(&bytes)
    }
    fn u16_hash(v: &[u16]) -> String {
        let mut bytes = Vec::with_capacity(v.len() * 2);
        for x in v {
            bytes.extend(x.to_le_bytes())
        }
        hash(&bytes)
    }
    fn u32_hash(v: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend(x.to_le_bytes())
        }
        hash(&bytes)
    }

    #[test]
    fn production_slices_match_frozen_typescript_constructor() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../core/testdata/production-scene-golden.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let mut options = ProductionSceneOptions::default();
            options.dt_s = Some(case.initialization.dt);
            options.atlas.brick_fine_resolution = case.initialization.brick_fine_resolution;
            options.atlas.maximum_macro_span_bricks = case.initialization.maximum_macro_span_bricks;
            options.atlas.surface_fine_rings = case.initialization.surface_fine_rings;
            options.resolution.policy.activity_signals = case.initialization.activity_signals;
            options.resolution.policy.coarse_first = case.initialization.coarse_first;
            options.resolution.policy.curvature_tolerance = case.initialization.curvature_tolerance;
            let actual = production_scene(case.scene, options)
                .unwrap_or_else(|e| panic!("{}: {e}", case.id));
            assert_eq!(
                [
                    actual.state.topology.graph.dimensions[0] as u32,
                    actual.state.topology.graph.dimensions[1] as u32
                ],
                case.expected.dimensions,
                "{} dimensions",
                case.id
            );
            assert_eq!(
                actual.source_atlas.dimensions, case.expected.source_dimensions,
                "{} source dimensions",
                case.id
            );
            assert_eq!(
                actual.physical.frame.center_cell_z, case.expected.center_cell_z,
                "{} center z",
                case.id
            );
            assert_eq!(
                actual.physical.frame.source_cell_size, case.expected.source_cell_size,
                "{} h",
                case.id
            );
            let bricks = actual
                .state
                .topology
                .bricks
                .iter()
                .map(|r| ExpectedBrick {
                    id: r.seed.id,
                    key: r.seed.key,
                    coordinate: [r.seed.coordinate[0], r.seed.coordinate[1]],
                    span_bricks: r.seed.span_bricks,
                    resolution: r.seed.resolution,
                    active: r.seed.active,
                })
                .collect::<Vec<_>>();
            assert_eq!(bricks, case.expected.bricks, "{} bricks", case.id);
            assert_eq!(
                actual.state.topology.graph.cells.len(),
                case.expected.cells,
                "{} cells",
                case.id
            );
            assert_eq!(
                actual.state.topology.graph.rows.len(),
                case.expected.rows,
                "{} rows",
                case.id
            );
            assert_eq!(
                actual.state.topology.graph.subfaces.len(),
                case.expected.subfaces,
                "{} subfaces",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.fields.density),
                case.expected.density_hash,
                "{} density",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.fields.capacity),
                case.expected.capacity_hash,
                "{} capacity",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.fields.face_velocity),
                case.expected.face_velocity_hash,
                "{} face velocity",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.fields.interface_normal),
                case.expected.normal_hash,
                "{} normals",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.fields.interface_offset),
                case.expected.offset_hash,
                "{} offsets",
                case.id
            );
            assert_eq!(
                u16_hash(&actual.material_id),
                case.expected.material_hash,
                "{} materials",
                case.id
            );
            assert_eq!(
                f32_hash(&actual.state.topology.graph.solid_voxel_fraction),
                case.expected.solid_hash,
                "{} solid plane",
                case.id
            );
            let mut cells = Vec::new();
            for c in &actual.state.topology.graph.cells {
                cells.extend([
                    c.id as f32,
                    c.stable_id.unwrap() as f32,
                    c.brick_key.unwrap() as f32,
                    c.minimum[0],
                    c.minimum[1],
                    c.maximum[0],
                    c.maximum[1],
                    c.center[0],
                    c.center[1],
                    c.widths[0],
                    c.widths[1],
                    c.measure,
                    c.refinement_region_scale.unwrap_or(1.0),
                ]);
            }
            assert_eq!(
                f32_hash(&cells),
                case.expected.cell_graph_hash,
                "{} cell graph",
                case.id
            );
            let mut floats = Vec::new();
            let mut words = Vec::new();
            for r in &actual.state.topology.graph.rows {
                floats.extend([
                    r.center[0],
                    r.center[1],
                    r.measure,
                    r.static_measure.unwrap(),
                    r.distance,
                    r.static_dual_weight.unwrap(),
                    r.dual_weight,
                    r.static_open_fraction.unwrap(),
                    r.open_fraction,
                    r.solid_velocity,
                ]);
                let kind = match r.kind {
                    crate::types::RowKind::IntraBrick => 0,
                    crate::types::RowKind::BrickFace => 1,
                    crate::types::RowKind::MixedSeam => 2,
                    crate::types::RowKind::SparseAir => 3,
                    crate::types::RowKind::ClosedWorld => 4,
                };
                words.extend([r.id, r.axis as u32, kind, r.terms.len() as u32]);
                for t in &r.terms {
                    words.extend([t.cell_id, t.coefficient.to_bits()]);
                }
            }
            assert_eq!(
                f32_hash(&floats),
                case.expected.row_float_hash,
                "{} row floats",
                case.id
            );
            assert_eq!(
                u32_hash(&words),
                case.expected.row_word_hash,
                "{} row words",
                case.id
            );
            let mut sf = Vec::new();
            let mut sw = Vec::new();
            for f in &actual.state.topology.graph.subfaces {
                sf.extend([
                    f.center[0],
                    f.center[1],
                    f.measure,
                    f.aperture,
                    f.solid_velocity,
                ]);
                sw.extend([
                    f.id,
                    f.row_id,
                    f.axis as u32,
                    f.negative_cell as u32,
                    f.positive_cell as u32,
                ]);
            }
            assert_eq!(
                f32_hash(&sf),
                case.expected.subface_float_hash,
                "{} subface floats",
                case.id
            );
            assert_eq!(
                u32_hash(&sw),
                case.expected.subface_word_hash,
                "{} subface words",
                case.id
            );
            let mut incidences = Vec::new();
            for v in &actual.state.topology.graph.incidences {
                incidences.push(v.len() as u32);
                incidences.extend(v);
            }
            assert_eq!(
                u32_hash(&incidences),
                case.expected.incidence_hash,
                "{} incidences",
                case.id
            );
            let mut si = Vec::new();
            for v in &actual.state.topology.graph.subface_incidences {
                si.push(v.len() as u32);
                for e in v {
                    si.extend([e.subface_id, e.orientation as i32 as u32]);
                }
            }
            assert_eq!(
                u32_hash(&si),
                case.expected.subface_incidence_hash,
                "{} subface incidences",
                case.id
            );
        }
    }

    #[test]
    fn candidate_xy_rungs_rebuild_the_frozen_typescript_source_graph() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../core/testdata/production-scene-golden.json"
        ))
        .unwrap();
        let case = fixture
            .cases
            .into_iter()
            .find(|c| c.id == fixture.transition.scene_id)
            .unwrap();
        let mut options = ProductionSceneOptions::default();
        options.atlas.brick_fine_resolution = case.initialization.brick_fine_resolution;
        options.resolution.policy.coarse_first = case.initialization.coarse_first;
        let bundle = production_scene(case.scene, options).unwrap();
        let mut bricks = bundle
            .state
            .topology
            .bricks
            .iter()
            .map(|r| r.seed.clone())
            .collect::<Vec<_>>();
        let changed = bricks
            .iter_mut()
            .find(|b| b.key == fixture.transition.target_key)
            .unwrap();
        changed.resolution = fixture.transition.target_resolution;
        // Transition preparation deliberately drops stale source payload when
        // the rung changes; transfer fills the candidate after compilation.
        changed.density.clear();
        changed.gamma.clear();
        let reduced = crate::topology::compile_topology::<2>(crate::topology::TopologySeed {
            dimensions: [
                bundle.source_atlas.dimensions[0],
                bundle.source_atlas.dimensions[1],
                1,
            ],
            generation: fixture.transition.generation,
            sparse_air_phi: 0.5,
            boundaries: bundle.state.topology.boundaries,
            bricks,
        })
        .unwrap();
        let source = rebuild_source_topology(
            &bundle.source_atlas,
            &reduced,
            bundle.source_topology.boundaries,
        )
        .unwrap();
        assert_eq!(
            source.graph.topology_generation,
            fixture.transition.generation
        );
        assert_eq!(source.graph.cells.len(), fixture.transition.cells);
        assert_eq!(source.graph.rows.len(), fixture.transition.rows);
        let actual = source
            .bricks
            .iter()
            .map(|r| SourceBrick {
                key: r.seed.key,
                coordinate: r.seed.coordinate,
                span_bricks: r.seed.span_bricks,
                resolution: r.seed.resolution,
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, fixture.transition.bricks);
    }
}
