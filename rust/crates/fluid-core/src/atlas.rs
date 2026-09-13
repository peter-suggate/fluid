//! Generation-zero Sparse Geometric (CM12) atlas construction.
//!
//! This is the Rust physical authority corresponding to
//! `lib/methods/adaptive-volume/sparse-brick-atlas.ts`.  Payloads are sampled
//! from the scene and the canonical static-solid world; callers do not provide
//! a precomputed JavaScript atlas.

use crate::initial_liquid::{
    base_initial_liquid_fraction_at_cell, initial_height_field_range,
    initial_liquid_fraction_at_cell, scene_dam_break_box,
};
use crate::initial_scene::{InitialLiquidVolume, SceneDocument};
use crate::scene_model::Vec3;
use crate::solid_world::{fluid_solid_world_for_scene, CompiledStaticSolidWorld};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

pub const DEFAULT_BRICK_FINE_RESOLUTION: u32 = 8;
pub const SPARSE_CM12_SOLID_RESTRICTION_TOLERANCE: f64 = 0.08;
pub const CM12_PAPER_DT_S: f64 = 1.0 / 30.0;
const STRUCTURAL_TO_FLUID_PAGE_RATIO: usize = 3;

pub type BrickCoordinate = [i32; 3];
pub type SparseBrickResolution = u32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SparseBrickAtlasInitializationOptions {
    pub finest_dimensions: [u32; 3],
    #[serde(default = "default_fine")]
    pub brick_fine_resolution: u32,
    #[serde(default)]
    pub maximum_finest_cells: Option<u64>,
    #[serde(default)]
    pub maximum_macro_span_bricks: Option<u32>,
    #[serde(default = "default_epsilon")]
    pub empty_epsilon: f64,
    #[serde(default = "default_surface_rings")]
    pub surface_fine_rings: u32,
    #[serde(default)]
    pub initial_surface_coarsening_bias_rings: u32,
    #[serde(default)]
    pub coarse_first_curvature_tolerance: Option<f64>,
    /// Serializable fixed-policy counterpart of TS's test-only callback.
    #[serde(default)]
    pub fixed_resolution: Option<u32>,
    /// Scene numerics used solely to bound generation-zero inflow support.
    #[serde(default)]
    #[serde(rename = "maximumDt_s")]
    pub maximum_dt_s: f64,
    #[serde(default)]
    #[serde(rename = "fixedDt_s")]
    pub fixed_dt_s: Option<f64>,
}

fn default_fine() -> u32 {
    8
}
fn default_epsilon() -> f64 {
    1e-12
}
fn default_surface_rings() -> u32 {
    1
}

impl Default for SparseBrickAtlasInitializationOptions {
    fn default() -> Self {
        Self {
            finest_dimensions: [1; 3],
            brick_fine_resolution: 8,
            maximum_finest_cells: None,
            maximum_macro_span_bricks: None,
            empty_epsilon: 1e-12,
            surface_fine_rings: 1,
            initial_surface_coarsening_bias_rings: 0,
            coarse_first_curvature_tolerance: None,
            fixed_resolution: None,
            maximum_dt_s: 0.0,
            fixed_dt_s: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SparseAdaptiveMassBrick {
    pub key: u32,
    pub coordinate: BrickCoordinate,
    pub span_bricks: u32,
    pub unclipped: bool,
    pub resolution: SparseBrickResolution,
    pub density: Vec<f64>,
    pub gamma: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SparseAdaptiveMassAtlas {
    pub dimensions: [u32; 3],
    pub brick_fine_resolution: u32,
    pub brick_cell_capacity: u32,
    pub brick_dimensions: [u32; 3],
    /// Stable key-ordered resident roster.
    pub bricks: Vec<SparseAdaptiveMassBrick>,
    /// Key to roster index.
    pub directory: BTreeMap<u32, usize>,
    /// Exact-origin key to roster index, grouped by dyadic span.
    pub directories_by_span: BTreeMap<u32, BTreeMap<u32, usize>>,
    pub maximum_span_bricks: u32,
    pub generation: u32,
    /// Effective construction step used to bound source support.
    pub initial_inflow_dt_s: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasError(pub String);
impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for AtlasError {}

#[derive(Clone, Copy, Debug)]
struct RefinementRegion {
    min: [f32; 3],
    max: [f32; 3],
    floor: f32,
    ceiling: f32,
}

fn ladder(fine: u32) -> Result<&'static [u32], AtlasError> {
    match fine {
        4 => Ok(&[1, 2, 4]),
        8 => Ok(&[1, 2, 4, 8]),
        16 => Ok(&[1, 2, 4, 8, 16]),
        _ => Err(AtlasError("brickFineResolution must be 4, 8, or 16".into())),
    }
}
fn valid_rung(r: u32, fine: u32) -> bool {
    ladder(fine).is_ok_and(|v| v.contains(&r))
}
fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
pub fn sparse_brick_key(c: BrickCoordinate, dims: [u32; 3]) -> u32 {
    (c[0] as u32) + dims[0] * ((c[1] as u32) + dims[1] * c[2] as u32)
}

fn create_atlas(
    dimensions: [u32; 3],
    fine: u32,
    mut bricks: Vec<SparseAdaptiveMassBrick>,
) -> Result<SparseAdaptiveMassAtlas, AtlasError> {
    bricks.sort_by_key(|b| b.key);
    let brick_dimensions = dimensions.map(|v| ceil_div(v, fine));
    let mut directory = BTreeMap::new();
    let mut by_span: BTreeMap<u32, BTreeMap<u32, usize>> = BTreeMap::new();
    let mut maximum_span_bricks = 1;
    for (i, b) in bricks.iter().enumerate() {
        if !b.span_bricks.is_power_of_two()
            || b.coordinate
                .iter()
                .any(|&v| v.rem_euclid(b.span_bricks as i32) != 0)
        {
            return Err(AtlasError(format!(
                "brick {} has invalid span/alignment",
                b.key
            )));
        }
        if !valid_rung(b.resolution, fine)
            || b.density.len() != b.resolution.pow(3) as usize
            || b.gamma.len() != b.density.len()
        {
            return Err(AtlasError(format!(
                "brick {} has invalid resolution/payload",
                b.key
            )));
        }
        if sparse_brick_key(b.coordinate, brick_dimensions) != b.key
            || directory.insert(b.key, i).is_some()
        {
            return Err(AtlasError(format!(
                "brick {} has duplicate or mismatched key",
                b.key
            )));
        }
        by_span.entry(b.span_bricks).or_default().insert(b.key, i);
        maximum_span_bricks = maximum_span_bricks.max(b.span_bricks);
    }
    Ok(SparseAdaptiveMassAtlas {
        dimensions,
        brick_fine_resolution: fine,
        brick_cell_capacity: fine.pow(3),
        brick_dimensions,
        bricks,
        directory,
        directories_by_span: by_span,
        maximum_span_bricks,
        generation: 1,
        initial_inflow_dt_s: CM12_PAPER_DT_S,
    })
}

pub fn sparse_brick_containing_coordinate<'a>(
    atlas: &'a SparseAdaptiveMassAtlas,
    q: BrickCoordinate,
) -> Option<&'a SparseAdaptiveMassBrick> {
    if q.iter()
        .enumerate()
        .any(|(a, &v)| v < 0 || v >= atlas.brick_dimensions[a] as i32)
    {
        return None;
    }
    let mut span = 1;
    while span <= atlas.maximum_span_bricks {
        let origin = q.map(|v| v.div_euclid(span as i32) * span as i32);
        let key = sparse_brick_key(origin, atlas.brick_dimensions);
        if let Some(&i) = atlas
            .directories_by_span
            .get(&span)
            .and_then(|m| m.get(&key))
        {
            return Some(&atlas.bricks[i]);
        }
        span *= 2;
    }
    None
}

fn initial_density(
    scene: &SceneDocument,
    dims: [u32; 3],
    q: [i32; 3],
    world: &CompiledStaticSolidWorld,
) -> f64 {
    if q.iter()
        .enumerate()
        .any(|(a, &v)| v < 0 || v >= dims[a] as i32)
        || !scene.systems.fluid
    {
        return 0.0;
    }
    let base = base_initial_liquid_fraction_at_cell(scene, q, dims);
    (1.0 - world.sample(q).solid_fraction as f64)
        * initial_liquid_fraction_at_cell(scene, q, dims, base)
}

fn initial_brick(
    scene: &SceneDocument,
    dims: [u32; 3],
    c: BrickCoordinate,
    resolution: u32,
    fine: u32,
    span: u32,
    world: &CompiledStaticSolidWorld,
) -> SparseAdaptiveMassBrick {
    let factor = fine * span / resolution;
    let mut density = vec![0.0; resolution.pow(3) as usize];
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let mut rho = 0.0;
                let mut count = 0;
                for dz in 0..factor {
                    for dy in 0..factor {
                        for dx in 0..factor {
                            let q = [
                                c[0] * fine as i32 + (x * factor + dx) as i32,
                                c[1] * fine as i32 + (y * factor + dy) as i32,
                                c[2] * fine as i32 + (z * factor + dz) as i32,
                            ];
                            if q.iter().enumerate().any(|(a, &v)| v >= dims[a] as i32) {
                                continue;
                            }
                            rho += initial_density(scene, dims, q, world);
                            count += 1;
                        }
                    }
                }
                density[(x + resolution * (y + resolution * z)) as usize] =
                    if count > 0 { rho / count as f64 } else { 0.0 };
            }
        }
    }
    let bd = dims.map(|v| ceil_div(v, fine));
    SparseAdaptiveMassBrick {
        key: sparse_brick_key(c, bd),
        coordinate: c,
        span_bricks: span,
        unclipped: false,
        resolution,
        density,
        gamma: vec![1.0; resolution.pow(3) as usize],
    }
}
fn uniform_brick(
    c: BrickCoordinate,
    span: u32,
    resolution: u32,
    bd: [u32; 3],
) -> SparseAdaptiveMassBrick {
    SparseAdaptiveMassBrick {
        key: sparse_brick_key(c, bd),
        coordinate: c,
        span_bricks: span,
        unclipped: false,
        resolution,
        density: vec![1.0; resolution.pow(3) as usize],
        gamma: vec![1.0; resolution.pow(3) as usize],
    }
}

fn prolong(mut b: SparseAdaptiveMassBrick, r: u32) -> SparseAdaptiveMassBrick {
    if r == b.resolution {
        return b;
    }
    let old = b.resolution;
    let factor = r / old;
    let sample = |v: &[f64], x: u32, y: u32, z: u32| {
        v[((x / factor) + old * ((y / factor) + old * (z / factor))) as usize]
    };
    b.density = (0..r)
        .flat_map(|z| (0..r).flat_map(move |y| (0..r).map(move |x| (x, y, z))))
        .map(|(x, y, z)| sample(&b.density, x, y, z))
        .collect();
    b.gamma = (0..r)
        .flat_map(|z| (0..r).flat_map(move |y| (0..r).map(move |x| (x, y, z))))
        .map(|(x, y, z)| sample(&b.gamma, x, y, z))
        .collect();
    b.resolution = r;
    b
}
fn restrict(mut b: SparseAdaptiveMassBrick, r: u32) -> SparseAdaptiveMassBrick {
    if r == b.resolution {
        return b;
    }
    let old = b.resolution;
    let f = old / r;
    let reduce = |v: &[f64]| {
        let mut out = vec![0.; r.pow(3) as usize];
        for z in 0..r {
            for y in 0..r {
                for x in 0..r {
                    let mut s = 0.;
                    for dz in 0..f {
                        for dy in 0..f {
                            for dx in 0..f {
                                s += v[((x * f + dx) + old * ((y * f + dy) + old * (z * f + dz)))
                                    as usize];
                            }
                        }
                    }
                    out[(x + r * (y + r * z)) as usize] = s / (f.pow(3) as f64);
                }
            }
        }
        out
    };
    b.density = reduce(&b.density);
    b.gamma = reduce(&b.gamma);
    b.resolution = r;
    b
}

fn parse_regions(scene: &SceneDocument, dims: [u32; 3]) -> Vec<RefinementRegion> {
    scene
        .fluid
        .refinement_regions
        .iter()
        .take(8)
        .filter_map(|v| {
            if v.get("rule").and_then(|x| x.as_str()) != Some("minimum-cell-size") {
                return None;
            }
            let vec3 =
                |name: &str| -> Option<Vec3> { serde_json::from_value(v.get(name)?.clone()).ok() };
            let lo = vec3("min_m")?;
            let hi = vec3("max_m")?;
            let c = &scene.container;
            let origin = [-c.width_m / 2., 0., -c.depth_m / 2.];
            let cell = [
                c.width_m / dims[0] as f64,
                c.height_m / dims[1] as f64,
                c.depth_m / dims[2] as f64,
            ];
            let mut min = [0f32; 3];
            let mut max = [0f32; 3];
            let lv = [lo.x, lo.y, lo.z];
            let hv = [hi.x, hi.y, hi.z];
            for a in 0..3 {
                min[a] = ((lv[a] - origin[a]) / cell[a]).clamp(0., dims[a] as f64) as f32;
                max[a] = ((hv[a] - origin[a]) / cell[a]).clamp(0., dims[a] as f64) as f32;
            }
            if (0..3).any(|a| max[a] <= min[a]) {
                return None;
            }
            let clamp = |n: f64| -> [u32; 1] {
                let mut c = 1;
                for x in [1, 2, 4, 8, 16, 32] {
                    if x as f64 <= n + 1e-9 {
                        c = x
                    }
                }
                [c]
            };
            let floor = clamp(v.get("minimumCellSize_cells")?.as_f64()?)[0] as f32;
            let ceiling = v
                .get("maximumCellSize_cells")
                .and_then(|x| x.as_f64())
                .map(|x| clamp(x)[0] as f32)
                .unwrap_or(0.);
            Some(RefinementRegion {
                min,
                max,
                floor,
                ceiling,
            })
        })
        .collect()
}
fn bounds_for(
    regions: &[RefinementRegion],
    origin: [u32; 3],
    extent: [u32; 3],
    fine: u32,
    nominal: u32,
) -> (u32, u32) {
    let mut floor = 1f32;
    let mut ceiling = 0f32;
    for r in regions {
        let intersects = (0..3)
            .all(|a| (origin[a] as f32) < r.max[a] && (origin[a] + extent[a]) as f32 > r.min[a]);
        if intersects {
            floor = floor.max(r.floor)
        }
        let contained = (0..3)
            .all(|a| origin[a] as f32 >= r.min[a] && (origin[a] + extent[a]) as f32 <= r.max[a]);
        if contained && r.ceiling > 0. {
            ceiling = if ceiling == 0. {
                r.ceiling
            } else {
                ceiling.min(r.ceiling)
            }
        }
    }
    let maxr = (nominal as f32 / floor).floor().max(1.).min(fine as f32) as u32;
    let minr = if ceiling == 0. {
        1
    } else {
        (nominal as f32 / ceiling).ceil().max(1.).min(fine as f32) as u32
    };
    (maxr, minr)
}
fn region_resolution(
    regions: &[RefinementRegion],
    dims: [u32; 3],
    c: BrickCoordinate,
    span: u32,
    requested: u32,
    fine: u32,
) -> u32 {
    let o = c.map(|v| v as u32 * fine);
    let e = [0, 1, 2].map(|a| (span * fine).min(dims[a].saturating_sub(o[a])));
    let (maxr, minr) = bounds_for(regions, o, e, fine, span * fine);
    requested.max(minr).min(maxr)
}

fn seed_cell(scene: &SceneDocument, p: Vec3, dims: [u32; 3]) -> [i32; 3] {
    let c = &scene.container;
    [
        ((p.x / c.width_m + 0.5) * dims[0] as f64)
            .floor()
            .clamp(0., (dims[0] - 1) as f64) as i32,
        (p.y / c.height_m * dims[1] as f64)
            .floor()
            .clamp(0., (dims[1] - 1) as f64) as i32,
        ((p.z / c.depth_m + 0.5) * dims[2] as f64)
            .floor()
            .clamp(0., (dims[2] - 1) as f64) as i32,
    ]
}
fn structural_seeds(
    scene: &SceneDocument,
    dims: [u32; 3],
    bd: [u32; 3],
    fine: u32,
    world: &CompiledStaticSolidWorld,
    budget: usize,
) -> Vec<BrickCoordinate> {
    let authored = scene.voxel_domain.brick_size_cells as i32;
    let mut out = BTreeSet::new();
    if let Some(seeds) = &scene.fluid.initial_brick_seeds_m {
        for &p in seeds {
            let a = seed_cell(scene, p, dims).map(|v| v.div_euclid(authored));
            let lower = a.map(|v| (v * authored).div_euclid(fine as i32));
            let upper = [0, 1, 2]
                .map(|i| ceil_div(dims[i].min(((a[i] + 1) * authored) as u32), fine) as i32);
            for z in lower[2]..upper[2] {
                for y in lower[1]..upper[1] {
                    for x in lower[0]..upper[0] {
                        out.insert([x, y, z]);
                    }
                }
            }
        }
    }
    if budget > 0 && world.pages.len() <= budget {
        let mut mixed = BTreeSet::new();
        'pages: for p in &world.pages {
            let solid = p.solid_fraction.iter().any(|&x| x > 0);
            let open = p.solid_fraction.iter().any(|&x| x < 255);
            if !solid || !open {
                continue;
            }
            for d in [
                [0, 0, 0],
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ] {
                let lo = [0, 1, 2].map(|a| ((p.coordinate[a] + d[a]) * 8).div_euclid(fine as i32));
                let hi = [0, 1, 2].map(|a| {
                    ceil_div(
                        dims[a].min((((p.coordinate[a] + d[a]) * 8 + 8).max(0)) as u32),
                        fine,
                    ) as i32
                });
                for z in lo[2].max(0)..hi[2].min(bd[2] as i32) {
                    for y in lo[1].max(0)..hi[1].min(bd[1] as i32) {
                        for x in lo[0].max(0)..hi[0].min(bd[0] as i32) {
                            mixed.insert([x, y, z]);
                            if mixed.len() > budget {
                                mixed.clear();
                                break 'pages;
                            }
                        }
                    }
                }
            }
        }
        out.extend(mixed);
    }
    out.into_iter().collect()
}

fn region_coordinates(
    regions: &[RefinementRegion],
    bd: [u32; 3],
    fine: u32,
) -> Vec<BrickCoordinate> {
    let mut out = BTreeSet::new();
    for r in regions {
        let own = (fine as f32 / r.floor).floor().max(1.).min(fine as f32) as u32;
        let halo = ((fine / own).ilog2() as i32 - 1).max(0);
        let lo = [0, 1, 2].map(|a| ((r.min[a] / fine as f32).floor() as i32 - halo).max(0));
        let hi =
            [0, 1, 2].map(|a| ((r.max[a] / fine as f32).ceil() as i32 + halo).min(bd[a] as i32));
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    out.insert([x, y, z]);
                }
            }
        }
    }
    out.into_iter().collect()
}

fn add_bounds(
    set: &mut BTreeSet<BrickCoordinate>,
    lo: [i32; 3],
    hi: [i32; 3],
    bd: [u32; 3],
    fine: u32,
) {
    let l = lo.map(|v| v.div_euclid(fine as i32).max(0));
    let h =
        [0, 1, 2].map(|a| ((hi[a] + fine as i32 - 1).div_euclid(fine as i32)).min(bd[a] as i32));
    for z in l[2]..h[2] {
        for y in l[1]..h[1] {
            for x in l[0]..h[0] {
                set.insert([x, y, z]);
            }
        }
    }
}
fn candidate_coordinates(
    scene: &SceneDocument,
    dims: [u32; 3],
    bd: [u32; 3],
    fine: u32,
    regions: &[RefinementRegion],
    world: &CompiledStaticSolidWorld,
) -> Vec<BrickCoordinate> {
    if !scene.systems.fluid {
        return vec![];
    }
    let mut set = BTreeSet::new();
    for c in structural_seeds(scene, dims, bd, fine, world, 0) {
        set.insert(c);
    }
    let replacement = scene
        .fluid
        .initial_brick_seeds_m
        .as_ref()
        .is_some_and(|s| !s.is_empty())
        && !scene.fluid.initial_brick_seeds_additive;
    let normalized = |p: [f64; 3]| {
        [
            (p[0] + scene.container.width_m * 0.5) / scene.container.width_m,
            p[1] / scene.container.height_m,
            (p[2] + scene.container.depth_m * 0.5) / scene.container.depth_m,
        ]
    };
    let mut add_norm = |lo: [f64; 3], hi: [f64; 3]| {
        if (0..3).any(|a| hi[a] <= lo[a]) {
            return;
        }
        add_bounds(
            &mut set,
            [0, 1, 2].map(|a| (lo[a] * dims[a] as f64).floor() as i32 - 1),
            [0, 1, 2].map(|a| (hi[a] * dims[a] as f64).ceil() as i32 + 1),
            bd,
            fine,
        )
    };
    if !replacement {
        if scene.fluid.initial_condition == "tank-fill" {
            let top = scene
                .fluid
                .initial_height_field
                .as_ref()
                .map(|f| {
                    initial_height_field_range(
                        f,
                        -scene.container.width_m / 2.,
                        scene.container.width_m / 2.,
                        -scene.container.depth_m / 2.,
                        scene.container.depth_m / 2.,
                    )[1] / scene.container.height_m
                })
                .unwrap_or(scene.container.fill_fraction);
            add_norm([0., 0., 0.], [1., top.min(1.), 1.]);
        } else {
            let d = scene_dam_break_box(scene);
            add_norm([d.min.x, d.min.y, d.min.z], [d.max.x, d.max.y, d.max.z]);
        }
    }
    for v in &scene.fluid.initial_liquid_volumes {
        let (lo, hi) = match *v {
            InitialLiquidVolume::Box { min_m, max_m } => {
                ([min_m.x, min_m.y, min_m.z], [max_m.x, max_m.y, max_m.z])
            }
            InitialLiquidVolume::Torus {
                center_m,
                radius_m,
                tube_radius_m,
            } => (
                [
                    center_m.x - radius_m - tube_radius_m,
                    center_m.y - tube_radius_m,
                    center_m.z - radius_m - tube_radius_m,
                ],
                [
                    center_m.x + radius_m + tube_radius_m,
                    center_m.y + tube_radius_m,
                    center_m.z + radius_m + tube_radius_m,
                ],
            ),
            InitialLiquidVolume::Cylinder {
                center_m,
                radius_m,
                half_height_m,
            } => (
                [
                    center_m.x - radius_m,
                    center_m.y - radius_m,
                    center_m.z - half_height_m,
                ],
                [
                    center_m.x + radius_m,
                    center_m.y + radius_m,
                    center_m.z + half_height_m,
                ],
            ),
            InitialLiquidVolume::Sphere { center_m, radius_m }
            | InitialLiquidVolume::Hemisphere {
                center_m, radius_m, ..
            } => (
                [
                    center_m.x - radius_m,
                    center_m.y - radius_m,
                    center_m.z - radius_m,
                ],
                [
                    center_m.x + radius_m,
                    center_m.y + radius_m,
                    center_m.z + radius_m,
                ],
            ),
        };
        add_norm(normalized(lo), normalized(hi));
    }
    for c in region_coordinates(regions, bd, fine) {
        set.insert(c);
    }
    let n = set.len();
    for c in structural_seeds(
        scene,
        dims,
        bd,
        fine,
        world,
        STRUCTURAL_TO_FLUID_PAGE_RATIO * n,
    ) {
        set.insert(c);
    }
    set.into_iter().collect()
}

fn brick_interface(
    scene: &SceneDocument,
    dims: [u32; 3],
    c: BrickCoordinate,
    eps: f64,
    fine: u32,
    world: &CompiledStaticSolidWorld,
) -> bool {
    let o = c.map(|v| v * fine as i32);
    for z in 0..fine as i32 {
        for y in 0..fine as i32 {
            for x in 0..fine as i32 {
                let q = [o[0] + x, o[1] + y, o[2] + z];
                if q.iter().enumerate().any(|(a, &v)| v >= dims[a] as i32) {
                    continue;
                }
                let own = initial_density(scene, dims, q, world);
                if own > eps && own < 1. - eps {
                    return true;
                }
                for d in [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ] {
                    let n = [q[0] + d[0], q[1] + d[1], q[2] + d[2]];
                    if n.iter()
                        .enumerate()
                        .any(|(a, &v)| v < 0 || v >= dims[a] as i32)
                    {
                        continue;
                    }
                    if (own > eps) != (initial_density(scene, dims, n, world) > eps) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn solid_floor(c: BrickCoordinate, fine: u32, world: &CompiledStaticSolidWorld) -> u32 {
    let o = c.map(|v| v * fine as i32);
    let w = (fine + 2) as usize;
    let mut samples = vec![0f64; w.pow(3)];
    let mut affected = false;
    for z in -1..=fine as i32 {
        for y in -1..=fine as i32 {
            for x in -1..=fine as i32 {
                let v = world.sample([o[0] + x, o[1] + y, o[2] + z]).solid_fraction as f64;
                affected |= v != 0.;
                samples[((x + 1) as usize) + w * ((y + 1) as usize + w * (z + 1) as usize)] = v;
            }
        }
    }
    if !affected {
        return 1;
    }
    let at = |x: i32, y: i32, z: i32| {
        samples[((x + 1) as usize) + w * ((y + 1) as usize + w * (z + 1) as usize)]
    };
    let rms = |s: f64, ss: f64, n: u32| (ss / n as f64 - (s / n as f64).powi(2)).max(0.).sqrt();
    for &r in ladder(fine).unwrap() {
        if r == fine {
            break;
        }
        let span = fine / r;
        let mut err: f64 = 0.;
        for mz in 0..r {
            for my in 0..r {
                for mx in 0..r {
                    let (mut s, mut ss, mut n) = (0., 0., 0);
                    for z in 0..span {
                        for y in 0..span {
                            for x in 0..span {
                                let v = at(
                                    (mx * span + x) as i32,
                                    (my * span + y) as i32,
                                    (mz * span + z) as i32,
                                );
                                s += v;
                                ss += v * v;
                                n += 1
                            }
                        }
                    }
                    err = err.max(rms(s, ss, n));
                }
            }
        }
        for axis in 0..3 {
            for face in 0..=r {
                for mv in 0..r {
                    for mu in 0..r {
                        let (mut s, mut ss, mut n) = (0., 0., 0);
                        for v in 0..span {
                            for u in 0..span {
                                let plane = (face * span) as i32;
                                let (uu, vv) = ((mu * span + u) as i32, (mv * span + v) as i32);
                                let (mut x, mut y, mut z) = (uu, vv, plane);
                                if axis == 0 {
                                    x = plane;
                                    y = uu;
                                    z = vv
                                } else if axis == 1 {
                                    x = vv;
                                    y = plane;
                                    z = uu
                                }
                                let val = 1.
                                    - at(
                                        x - (axis == 0) as i32,
                                        y - (axis == 1) as i32,
                                        z - (axis == 2) as i32,
                                    )
                                    .max(at(x, y, z));
                                s += val;
                                ss += val * val;
                                n += 1
                            }
                        }
                        err = err.max(rms(s, ss, n));
                    }
                }
            }
        }
        if err <= SPARSE_CM12_SOLID_RESTRICTION_TOLERANCE {
            return r;
        }
    }
    fine
}

fn curvature(
    scene: &SceneDocument,
    dims: [u32; 3],
    c: BrickCoordinate,
    fine: u32,
    tol: f64,
    world: &CompiledStaticSolidWorld,
) -> u32 {
    let (mut lo, mut hi): ([f64; 3], [f64; 3]) = ([1.; 3], [-1.; 3]);
    let mut count = 0;
    for z in -1..=fine as i32 {
        for y in -1..=fine as i32 {
            for x in -1..=fine as i32 {
                let q = [
                    c[0] * fine as i32 + x,
                    c[1] * fine as i32 + y,
                    c[2] * fine as i32 + z,
                ];
                let mut g = [0f64; 3];
                for a in 0..3 {
                    let mut b = q;
                    let mut f = q;
                    b[a] -= 1;
                    f[a] += 1;
                    b = [0, 1, 2].map(|i| b[i].clamp(0, dims[i] as i32 - 1));
                    f = [0, 1, 2].map(|i| f[i].clamp(0, dims[i] as i32 - 1));
                    g[a] = (initial_density(scene, dims, f, world)
                        - initial_density(scene, dims, b, world))
                        / 2.;
                }
                let len = g[0].hypot(g[1]).hypot(g[2]);
                if len < 1e-6 {
                    continue;
                }
                count += 1;
                for a in 0..3 {
                    lo[a] = lo[a].min(g[a] / len);
                    hi[a] = hi[a].max(g[a] / len)
                }
            }
        }
    }
    if count == 0 {
        return 1;
    }
    let variation = (hi[0] - lo[0]).hypot(hi[1] - lo[1]).hypot(hi[2] - lo[2]);
    let mut r = 1;
    while r < fine && variation / r as f64 > tol {
        r *= 2
    }
    r
}

fn face_neighbors<'a>(
    atlas: &'a SparseAdaptiveMassAtlas,
    b: &SparseAdaptiveMassBrick,
) -> Vec<&'a SparseAdaptiveMassBrick> {
    atlas
        .bricks
        .iter()
        .filter(|n| {
            n.key != b.key
                && (0..3).any(|axis| {
                    let touch = b.coordinate[axis] + b.span_bricks as i32 == n.coordinate[axis]
                        || n.coordinate[axis] + n.span_bricks as i32 == b.coordinate[axis];
                    touch
                        && (0..3).filter(|&a| a != axis).all(|a| {
                            b.coordinate[a] < (n.coordinate[a] + n.span_bricks as i32)
                                && n.coordinate[a] < (b.coordinate[a] + b.span_bricks as i32)
                        })
                })
        })
        .collect()
}

fn inflow_support(
    scene: &SceneDocument,
    dims: [u32; 3],
    fine: u32,
    opt: &SparseBrickAtlasInitializationOptions,
) -> Vec<BrickCoordinate> {
    let Some(i) = scene.fluid.inflow else {
        return vec![];
    };
    let speed = i
        .velocity_m_s
        .x
        .hypot(i.velocity_m_s.y)
        .hypot(i.velocity_m_s.z);
    let dt = CM12_PAPER_DT_S
        .max(opt.maximum_dt_s)
        .max(opt.fixed_dt_s.unwrap_or(0.));
    if i.radius_m <= 0. || speed <= 0. || dt <= 0. {
        return vec![];
    }
    let cs = (scene.container.width_m / dims[0] as f64)
        .min(scene.container.height_m / dims[1] as f64)
        .min(scene.container.depth_m / dims[2] as f64);
    let half = i.length_m / (2. * speed);
    let outlet = [
        i.center_m.x + i.velocity_m_s.x * half,
        i.center_m.y + i.velocity_m_s.y * half,
        i.center_m.z + i.velocity_m_s.z * half,
    ];
    let origin = [
        -scene.container.width_m / 2.,
        0.,
        -scene.container.depth_m / 2.,
    ];
    let velocity = [
        i.velocity_m_s.x / cs,
        i.velocity_m_s.y / cs,
        i.velocity_m_s.z / cs,
    ];
    let radius = i.radius_m / cs + 1.;
    let of = [0, 1, 2].map(|a| (outlet[a] - origin[a]) / cs);
    let lo = [0, 1, 2].map(|a| {
        ((of[a].min(of[a] + 2. * velocity[a] * dt) - radius) / fine as f64).floor() as i32
    });
    let hi = [0, 1, 2].map(|a| {
        ((of[a].max(of[a] + 2. * velocity[a] * dt) + radius) / fine as f64).floor() as i32
    });
    let bd = dims.map(|v| ceil_div(v, fine));
    let mut out = vec![];
    for z in lo[2].max(0)..=hi[2].min(bd[2] as i32 - 1) {
        for y in lo[1].max(0)..=hi[1].min(bd[1] as i32 - 1) {
            for x in lo[0].max(0)..=hi[0].min(bd[0] as i32 - 1) {
                out.push([x, y, z]);
            }
        }
    }
    out
}

fn exterior_coordinates(b: &SparseAdaptiveMassBrick, bd: [u32; 3]) -> Vec<BrickCoordinate> {
    let s = b.span_bricks as i32;
    let mut out = vec![];
    for z in -1..=s {
        for y in -1..=s {
            for x in -1..=s {
                if (0..s).contains(&x) && (0..s).contains(&y) && (0..s).contains(&z) {
                    continue;
                }
                let q = [
                    b.coordinate[0] + x,
                    b.coordinate[1] + y,
                    b.coordinate[2] + z,
                ];
                if q.iter()
                    .enumerate()
                    .all(|(a, &v)| v >= 0 && v < bd[a] as i32)
                {
                    out.push(q)
                }
            }
        }
    }
    out
}
fn has_open_voxel(
    c: BrickCoordinate,
    dims: [u32; 3],
    fine: u32,
    world: &CompiledStaticSolidWorld,
) -> bool {
    for z in 0..fine {
        for y in 0..fine {
            for x in 0..fine {
                let q = [
                    c[0] * fine as i32 + x as i32,
                    c[1] * fine as i32 + y as i32,
                    c[2] * fine as i32 + z as i32,
                ];
                if q.iter().enumerate().any(|(a, &v)| v >= dims[a] as i32) {
                    continue;
                }
                if world.sample(q).solid_fraction < 1. {
                    return true;
                }
            }
        }
    }
    false
}

fn strong_grade_by_coarsening(
    dim: [u32; 3],
    mut bricks: Vec<SparseAdaptiveMassBrick>,
    fine: u32,
) -> Result<Vec<SparseAdaptiveMassBrick>, AtlasError> {
    loop {
        let atlas = create_atlas(dim, fine, bricks.clone())?;
        let mut res: Vec<u32> = atlas.bricks.iter().map(|b| b.resolution).collect();
        let mut changed = false;
        for (i, b) in atlas.bricks.iter().enumerate() {
            for n in face_neighbors(&atlas, b) {
                let j = atlas.directory[&n.key];
                let wi = fine * b.span_bricks / res[i];
                let wj = fine * n.span_bricks / res[j];
                if wi.max(wj) <= 2 * wi.min(wj) {
                    continue;
                }
                let finer = if wi < wj { i } else { j };
                let coarse = if finer == i { wj } else { wi };
                let mut next = res[finer];
                while next > 1 && coarse > 2 * fine * atlas.bricks[finer].span_bricks / next {
                    next /= 2
                }
                if next == res[finer] {
                    return Err(AtlasError("cannot strongly grade by coarsening".into()));
                }
                res[finer] = next;
                changed = true;
            }
        }
        if !changed {
            return Ok(atlas.bricks);
        }
        bricks = atlas
            .bricks
            .into_iter()
            .enumerate()
            .map(|(i, b)| restrict(b, res[i]))
            .collect();
    }
}

fn atlas_with_air_support(
    scene: &SceneDocument,
    dims: [u32; 3],
    bricks: Vec<SparseAdaptiveMassBrick>,
    fine: u32,
    regions: &[RefinementRegion],
    opt: &SparseBrickAtlasInitializationOptions,
    world: &CompiledStaticSolidWorld,
) -> Result<SparseAdaptiveMassAtlas, AtlasError> {
    let mut atlas = create_atlas(dims, fine, bricks)?;
    let inflow: Vec<_> = inflow_support(scene, dims, fine, opt)
        .into_iter()
        .filter(|&c| has_open_voxel(c, dims, fine, world))
        .collect();
    let roots: BTreeSet<_> = inflow
        .iter()
        .map(|&c| sparse_brick_key(c, atlas.brick_dimensions))
        .collect();
    if !inflow.is_empty() {
        let mut all: BTreeMap<u32, SparseAdaptiveMassBrick> =
            atlas.bricks.into_iter().map(|b| (b.key, b)).collect();
        for c in inflow {
            let key = sparse_brick_key(c, atlas.brick_dimensions);
            let r = region_resolution(regions, dims, c, 1, fine, fine);
            let b = match all.remove(&key) {
                Some(b) if b.resolution <= r => prolong(b, r),
                Some(b) => restrict(b, r),
                None => initial_brick(scene, dims, c, r, fine, 1, world),
            };
            all.insert(key, b);
        }
        atlas = create_atlas(dims, fine, all.into_values().collect())?;
    }
    let support_roots: Vec<_> = atlas
        .bricks
        .iter()
        .filter(|b| roots.contains(&b.key) || b.density.iter().any(|&d| d > 0.))
        .cloned()
        .collect();
    let mut requests: BTreeMap<u32, (BrickCoordinate, u32)> = BTreeMap::new();
    for b in &support_roots {
        let face_r = (b.resolution / b.span_bricks).max(1) / 2;
        for c in exterior_coordinates(b, atlas.brick_dimensions) {
            let exterior = (0..3)
                .filter(|&a| {
                    c[a] < b.coordinate[a] || c[a] >= b.coordinate[a] + b.span_bricks as i32
                })
                .count();
            let r = if exterior == 1 { face_r.max(1) } else { 1 };
            if sparse_brick_containing_coordinate(&atlas, c).is_some()
                || !has_open_voxel(c, dims, fine, world)
            {
                continue;
            }
            let key = sparse_brick_key(c, atlas.brick_dimensions);
            requests
                .entry(key)
                .and_modify(|x| x.1 = x.1.max(r))
                .or_insert((c, r));
        }
    }
    if !requests.is_empty() {
        let support_keys: Vec<_> = requests.keys().copied().collect();
        let mut all: BTreeMap<u32, SparseAdaptiveMassBrick> =
            atlas.bricks.into_iter().map(|b| (b.key, b)).collect();
        for (_, (c, r)) in requests {
            let rr = region_resolution(regions, dims, c, 1, r, fine);
            let b = initial_brick(scene, dims, c, rr, fine, 1, world);
            all.insert(b.key, b);
        }
        if regions.is_empty() {
            let mut queue: VecDeque<u32> = support_keys.into();
            let mut queued: BTreeSet<u32> = queue.iter().copied().collect();
            loop {
                let temp = create_atlas(dims, fine, all.values().cloned().collect())?;
                let Some(key) = queue.pop_front() else { break };
                queued.remove(&key);
                let Some(&idx) = temp.directory.get(&key) else {
                    continue;
                };
                let b = &temp.bricks[idx];
                for n in face_neighbors(&temp, b) {
                    let own = fine * b.span_bricks / all[&b.key].resolution;
                    let other = fine * n.span_bricks / all[&n.key].resolution;
                    if own.max(other) <= 2 * own.min(other) {
                        continue;
                    }
                    let coarse = if own > other { b.key } else { n.key };
                    let current = all[&coarse].resolution;
                    let promoted = current * 2;
                    if !valid_rung(promoted, fine) {
                        return Err(AtlasError("cannot grade initial air support".into()));
                    }
                    let old = all.remove(&coarse).unwrap();
                    all.insert(coarse, prolong(old, promoted));
                    if queued.insert(coarse) {
                        queue.push_back(coarse)
                    }
                }
            }
        } else {
            all = strong_grade_by_coarsening(dims, all.into_values().collect(), fine)?
                .into_iter()
                .map(|b| (b.key, b))
                .collect()
        }
        atlas = create_atlas(dims, fine, all.into_values().collect())?;
    }
    enforce_physical_floors(scene, atlas, regions, world)
}

fn enforce_physical_floors(
    scene: &SceneDocument,
    mut atlas: SparseAdaptiveMassAtlas,
    regions: &[RefinementRegion],
    world: &CompiledStaticSolidWorld,
) -> Result<SparseAdaptiveMassAtlas, AtlasError> {
    let fine = atlas.brick_fine_resolution;
    if !regions.iter().any(|r| r.floor > fine as f32) {
        return Ok(atlas);
    }
    let dims = atlas.dimensions;
    let floor_for = |b: &SparseAdaptiveMassBrick| {
        regions
            .iter()
            .filter(|r| {
                (0..3).all(|a| {
                    let lo = (b.coordinate[a] * fine as i32) as f32;
                    let hi = ((b.coordinate[a] + b.span_bricks as i32) * fine as i32)
                        .min(dims[a] as i32) as f32;
                    lo < r.max[a] && hi > r.min[a]
                })
            })
            .fold(1u32, |v, r| v.max(r.floor as u32))
    };
    let width = |b: &SparseAdaptiveMassBrick| fine * b.span_bricks / b.resolution;
    let mut bricks = atlas.bricks.clone();
    let mut samples: u64 = 0;
    loop {
        let Some(v) = bricks.iter().find(|b| width(b) < floor_for(b)).cloned() else {
            break;
        };
        let target = floor_for(&v);
        let span = v.span_bricks.max(target / fine);
        let c = v
            .coordinate
            .map(|q| q.div_euclid(span as i32) * span as i32);
        let r = (fine * span / target).max(1);
        samples += (span * fine).pow(3) as u64;
        if samples > 16_777_216 {
            return Err(AtlasError(
                "initial minimum-cell-size constraint exceeds grouping capacity".into(),
            ));
        }
        bricks.retain(|b| {
            !(0..3).all(|a| {
                b.coordinate[a] >= c[a]
                    && b.coordinate[a] + b.span_bricks as i32 <= c[a] + span as i32
            })
        });
        bricks.push(initial_brick(
            scene,
            atlas.dimensions,
            c,
            r,
            fine,
            span,
            world,
        ));
    }
    loop {
        atlas = create_atlas(atlas.dimensions, fine, bricks.clone())?;
        let mut choice = None;
        for b in &atlas.bricks {
            for n in face_neighbors(&atlas, b) {
                if width(b) * 2 < width(n) {
                    choice = Some((b.clone(), width(n) / 2));
                    break;
                }
            }
            if choice.is_some() {
                break;
            }
        }
        let Some((b, target)) = choice else {
            return create_atlas(atlas.dimensions, fine, bricks);
        };
        let span = b.span_bricks.max(target / fine);
        let c = b
            .coordinate
            .map(|q| q.div_euclid(span as i32) * span as i32);
        let r = (fine * span / target).max(1);
        bricks.retain(|x| {
            !(0..3).all(|a| {
                x.coordinate[a] >= c[a]
                    && x.coordinate[a] + x.span_bricks as i32 <= c[a] + span as i32
            })
        });
        bricks.push(initial_brick(
            scene,
            atlas.dimensions,
            c,
            r,
            fine,
            span,
            world,
        ));
    }
}

fn coarse_bulk_cover(
    mut atlas: SparseAdaptiveMassAtlas,
    max_span: u32,
) -> Result<SparseAdaptiveMassAtlas, AtlasError> {
    let fine = atlas.brick_fine_resolution;
    let full = |b: &SparseAdaptiveMassBrick| {
        b.resolution == 1 && b.density.iter().all(|&x| x == 1.) && b.gamma.iter().all(|&x| x == 1.)
    };
    let mut span = 2;
    while span <= max_span {
        let mut groups: BTreeMap<BrickCoordinate, Vec<SparseAdaptiveMassBrick>> = BTreeMap::new();
        for b in &atlas.bricks {
            if b.span_bricks == span / 2 && full(b) {
                let o = b
                    .coordinate
                    .map(|q| q.div_euclid(span as i32) * span as i32);
                groups.entry(o).or_default().push(b.clone());
            }
        }
        let mut removed = BTreeSet::new();
        let mut parents = vec![];
        for (o, g) in groups {
            if g.len() != 8
                || (0..3).any(|a| (o[a] + span as i32) * fine as i32 > atlas.dimensions[a] as i32)
            {
                continue;
            }
            let mut eligible = true;
            for axis in 0..3 {
                for sign in [-1, 1] {
                    let t: Vec<_> = (0..3).filter(|&a| a != axis).collect();
                    for u in 0..span as i32 {
                        for v in 0..span as i32 {
                            let mut q = o;
                            q[axis] += if sign < 0 { -1 } else { span as i32 };
                            q[t[0]] += u;
                            q[t[1]] += v;
                            if q.iter()
                                .enumerate()
                                .any(|(a, &x)| x < 0 || x >= atlas.brick_dimensions[a] as i32)
                            {
                                continue;
                            }
                            if sparse_brick_containing_coordinate(&atlas, q)
                                .is_none_or(|n| !full(n) || n.span_bricks < span / 2)
                            {
                                eligible = false;
                            }
                        }
                    }
                }
            }
            if eligible {
                removed.extend(g.iter().map(|b| b.key));
                parents.push(uniform_brick(o, span, 1, atlas.brick_dimensions));
            }
        }
        if parents.is_empty() {
            break;
        }
        let mut next: Vec<_> = atlas
            .bricks
            .into_iter()
            .filter(|b| !removed.contains(&b.key))
            .collect();
        next.extend(parents);
        atlas = create_atlas(atlas.dimensions, fine, next)?;
        span *= 2;
    }
    Ok(atlas)
}

fn hierarchical_tank(
    scene: &SceneDocument,
    dims: [u32; 3],
    bd: [u32; 3],
    fine: u32,
    surface_rings: u32,
    max_span: u32,
    regions: &[RefinementRegion],
    opt: &SparseBrickAtlasInitializationOptions,
    world: &CompiledStaticSolidWorld,
) -> Result<Option<Vec<SparseAdaptiveMassBrick>>, AtlasError> {
    if !scene.systems.fluid {
        return Ok(Some(vec![]));
    }
    let replacement = scene
        .fluid
        .initial_brick_seeds_m
        .as_ref()
        .is_some_and(|s| !s.is_empty())
        && !scene.fluid.initial_brick_seeds_additive;
    if scene.fluid.initial_condition != "tank-fill"
        || scene.fluid.initial_height_field.is_some()
        || replacement
        || !scene.fluid.initial_liquid_volumes.is_empty()
    {
        return Ok(None);
    }
    let full_y = (scene.container.fill_fraction * dims[1] as f64 + 0.5)
        .floor()
        .clamp(0., dims[1] as f64) as i32;
    for p in &world.pages {
        for (local, &v) in p.solid_fraction.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let x = p.coordinate[0] * 8 + (local % 8) as i32;
            let y = p.coordinate[1] * 8 + ((local / 8) % 8) as i32;
            let z = p.coordinate[2] * 8 + (local / 64) as i32;
            if x >= 0 && x < dims[0] as i32 && y >= 0 && y < full_y && z >= 0 && z < dims[2] as i32
            {
                return Ok(None);
            }
        }
    }
    let full_by = full_y / fine as i32;
    let fractional = (full_y % fine as i32 != 0) as i32;
    let wet = [bd[0] as i32, full_by, bd[2] as i32];
    let inflow = inflow_support(scene, dims, fine, opt);
    let mut root = 1;
    while root < *bd.iter().max().unwrap() {
        root *= 2
    }
    let mut bricks = vec![];
    struct Visit<'a> {
        dims: [u32; 3],
        bd: [u32; 3],
        fine: u32,
        surface: u32,
        max_span: u32,
        regions: &'a [RefinementRegion],
        wet: [i32; 3],
        fractional: i32,
        inflow: &'a [BrickCoordinate],
        out: &'a mut Vec<SparseAdaptiveMassBrick>,
    }
    impl Visit<'_> {
        fn go(&mut self, o: BrickCoordinate, span: u32) {
            if (0..3).any(|a| o[a] < 0 || o[a] >= self.wet[a]) {
                return;
            }
            let inside = (0..3).all(|a| o[a] + span as i32 <= self.wet[a]);
            if inside {
                let edge = span * self.fine;
                let free = self.wet[1] < self.dims[1] as i32;
                let clearance = if free {
                    (self.wet[1] - o[1] - span as i32 - (self.surface as i32 - 1) + self.fractional)
                        .max(0)
                } else {
                    i32::MAX
                };
                let allowed_surface = if !free {
                    edge
                } else if clearance < (self.fine.ilog2() as i32) {
                    edge.min(1u32 << clearance)
                } else {
                    edge.min(
                        1u32 << ((self.fine as f64
                            * (clearance - self.fine.ilog2() as i32 + 1) as f64)
                            .log2()
                            .floor() as u32),
                    )
                };
                let distance = self
                    .inflow
                    .iter()
                    .map(|q| {
                        (0..3)
                            .map(|a| {
                                if q[a] < o[a] {
                                    o[a] - q[a]
                                } else if q[a] >= o[a] + span as i32 {
                                    q[a] - (o[a] + span as i32 - 1)
                                } else {
                                    0
                                }
                            })
                            .sum::<i32>()
                    })
                    .min()
                    .unwrap_or(i32::MAX);
                let allowed = allowed_surface.min(if distance >= 31 {
                    u32::MAX
                } else {
                    1u32 << distance
                });
                let deep = if free {
                    (self.wet[1] - o[1] - 1 - (self.surface as i32 - 1) + self.fractional).max(0)
                } else {
                    i32::MAX
                };
                let deep_allowed = if !free {
                    edge
                } else if deep < self.fine.ilog2() as i32 {
                    edge.min(1u32 << deep)
                } else {
                    edge.min(
                        1u32 << ((self.fine as f64 * (deep - self.fine.ilog2() as i32 + 1) as f64)
                            .log2()
                            .floor() as u32),
                    )
                };
                let crosses = span > 1 && allowed < self.fine && deep_allowed != allowed;
                let requested = edge / allowed.max(1);
                let required = if requested <= self.fine {
                    region_resolution(self.regions, self.dims, o, span, requested, self.fine)
                } else {
                    requested
                };
                if !crosses
                    && span <= self.max_span
                    && required <= if span > 1 { self.fine / 2 } else { self.fine }
                {
                    self.out.push(uniform_brick(o, span, required, self.bd));
                    return;
                }
            }
            if span == 1 {
                if inside {
                    let r = region_resolution(self.regions, self.dims, o, 1, self.fine, self.fine);
                    self.out.push(uniform_brick(o, 1, r, self.bd));
                }
                return;
            }
            let h = span / 2;
            for dz in [0, h] {
                for dy in [0, h] {
                    for dx in [0, h] {
                        self.go([o[0] + dx as i32, o[1] + dy as i32, o[2] + dz as i32], h)
                    }
                }
            }
        }
    }
    if full_by > 0 {
        Visit {
            dims,
            bd,
            fine,
            surface: surface_rings,
            max_span,
            regions,
            wet,
            fractional,
            inflow: &inflow,
            out: &mut bricks,
        }
        .go([0, 0, 0], root)
    }
    if full_y % fine as i32 != 0 && full_by < bd[1] as i32 {
        let sr = if surface_rings > 0 { fine } else { fine / 2 };
        for z in 0..bd[2] as i32 {
            for x in 0..bd[0] as i32 {
                let c = [x, full_by, z];
                let r = region_resolution(regions, dims, c, 1, sr, fine);
                bricks.push(initial_brick(scene, dims, c, r, fine, 1, world));
            }
        }
    }
    if !regions.is_empty() {
        bricks = strong_grade_by_coarsening(dims, bricks, fine)?
    }
    let provisional = create_atlas(dims, fine, bricks.clone())?;
    for c in structural_seeds(
        scene,
        dims,
        bd,
        fine,
        world,
        STRUCTURAL_TO_FLUID_PAGE_RATIO * bricks.len(),
    ) {
        if sparse_brick_containing_coordinate(&provisional, c).is_none() {
            let r = region_resolution(regions, dims, c, 1, fine, fine);
            bricks.push(initial_brick(scene, dims, c, r, fine, 1, world));
        }
    }
    if !regions.is_empty() {
        bricks = strong_grade_by_coarsening(dims, bricks, fine)?
    }
    bricks.sort_by_key(|b| b.key);
    Ok(Some(bricks))
}

#[derive(Clone)]
struct Candidate {
    coordinate: BrickCoordinate,
    key: u32,
    interface: bool,
}
fn tile(
    c: BrickCoordinate,
    dims: [u32; 3],
    fine: u32,
    regions: &[RefinementRegion],
) -> (u32, BrickCoordinate) {
    let o = c.map(|v| v as u32 * fine);
    let e = [0, 1, 2].map(|a| fine.min(dims[a].saturating_sub(o[a])));
    let (maxr, _) = bounds_for(regions, o, e, fine, fine);
    let scale = (fine as f64 / maxr as f64).round().max(1.) as u32;
    (scale, c.map(|v| v.div_euclid(scale as i32) * scale as i32))
}

pub fn initialize_sparse_brick_atlas_from_scene(
    scene: &SceneDocument,
    opt: &SparseBrickAtlasInitializationOptions,
) -> Result<SparseAdaptiveMassAtlas, AtlasError> {
    let dims = opt.finest_dimensions;
    if dims.contains(&0) {
        return Err(AtlasError(
            "finestDimensions must contain positive integers".into(),
        ));
    }
    let fine = opt.brick_fine_resolution;
    ladder(fine)?;
    let cells = dims.iter().fold(1u64, |p, &v| p * v as u64);
    if opt.maximum_finest_cells.is_some_and(|m| cells > m) {
        return Err(AtlasError(format!(
            "bounded finest lattice has {cells} cells"
        )));
    }
    let max_span = opt.maximum_macro_span_bricks.unwrap_or(u32::MAX);
    if opt
        .maximum_macro_span_bricks
        .is_some_and(|span| !span.is_power_of_two())
    {
        return Err(AtlasError(
            "maximumMacroSpanBricks must be a positive power of two".into(),
        ));
    }
    if let Some(r) = opt.fixed_resolution {
        if !valid_rung(r, fine) {
            return Err(AtlasError("fixedResolution is not a supported rung".into()));
        }
    }
    let world = fluid_solid_world_for_scene(scene);
    let bd = dims.map(|v| ceil_div(v, fine));
    let regions = parse_regions(scene, dims);
    let authored = opt.surface_fine_rings.clamp(1, 8);
    let surface = authored - opt.initial_surface_coarsening_bias_rings.min(authored);
    let hierarchical_compatible = regions.iter().all(|r| {
        r.ceiling == 0. || (0..3).all(|a| r.min[a] <= 1e-4 && r.max[a] >= dims[a] as f32 - 1e-4)
    });
    if opt.fixed_resolution.is_none()
        && opt.coarse_first_curvature_tolerance.is_none()
        && hierarchical_compatible
    {
        if let Some(bricks) = hierarchical_tank(
            scene, dims, bd, fine, surface, max_span, &regions, opt, &world,
        )? {
            let mut atlas =
                atlas_with_air_support(scene, dims, bricks, fine, &regions, opt, &world)?;
            atlas.initial_inflow_dt_s = CM12_PAPER_DT_S
                .max(opt.maximum_dt_s)
                .max(opt.fixed_dt_s.unwrap_or(0.0));
            return Ok(atlas);
        }
    }
    let candidate_coords = candidate_coordinates(scene, dims, bd, fine, &regions, &world);
    let structural: BTreeSet<_> = structural_seeds(
        scene,
        dims,
        bd,
        fine,
        &world,
        STRUCTURAL_TO_FLUID_PAGE_RATIO * candidate_coords.len(),
    )
    .into_iter()
    .chain(region_coordinates(&regions, bd, fine))
    .map(|c| sparse_brick_key(c, bd))
    .collect();
    let mut candidates = vec![];
    for c in candidate_coords {
        let nonempty = (0..fine).any(|z| {
            (0..fine).any(|y| {
                (0..fine).any(|x| {
                    initial_density(
                        scene,
                        dims,
                        [
                            c[0] * fine as i32 + x as i32,
                            c[1] * fine as i32 + y as i32,
                            c[2] * fine as i32 + z as i32,
                        ],
                        &world,
                    ) > opt.empty_epsilon
                })
            })
        });
        let key = sparse_brick_key(c, bd);
        if nonempty || structural.contains(&key) {
            candidates.push(Candidate {
                coordinate: c,
                key,
                interface: brick_interface(scene, dims, c, opt.empty_epsilon, fine, &world),
            })
        }
    }
    candidates.sort_by_key(|c| c.key);
    if !regions.is_empty() {
        let mut known: BTreeSet<_> = candidates.iter().map(|c| c.key).collect();
        let original = candidates.clone();
        for c in original {
            let (scale, o) = tile(c.coordinate, dims, fine, &regions);
            if scale <= 1 {
                continue;
            }
            for z in 0..scale as i32 {
                for y in 0..scale as i32 {
                    for x in 0..scale as i32 {
                        let q = [o[0] + x, o[1] + y, o[2] + z];
                        if q.iter()
                            .enumerate()
                            .any(|(a, &v)| v < 0 || v >= bd[a] as i32)
                            || tile(q, dims, fine, &regions) != (scale, o)
                        {
                            continue;
                        }
                        let key = sparse_brick_key(q, bd);
                        if known.insert(key) {
                            candidates.push(Candidate {
                                coordinate: q,
                                key,
                                interface: brick_interface(
                                    scene,
                                    dims,
                                    q,
                                    opt.empty_epsilon,
                                    fine,
                                    &world,
                                ),
                            })
                        }
                    }
                }
            }
        }
        candidates.sort_by_key(|c| c.key)
    }
    let by_key: BTreeMap<_, _> = candidates.iter().map(|c| (c.key, c.clone())).collect();
    let mut tile_for = BTreeMap::new();
    let mut tiles: BTreeMap<(u32, BrickCoordinate), (bool, BTreeSet<(u32, BrickCoordinate)>)> =
        BTreeMap::new();
    for c in &candidates {
        let t = tile(c.coordinate, dims, fine, &regions);
        tile_for.insert(c.key, t);
        tiles.entry(t).or_default().0 |= c.interface;
    }
    for c in &candidates {
        for d in [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ] {
            let q = [
                c.coordinate[0] + d[0],
                c.coordinate[1] + d[1],
                c.coordinate[2] + d[2],
            ];
            if q.iter()
                .enumerate()
                .any(|(a, &v)| v < 0 || v >= bd[a] as i32)
            {
                continue;
            }
            let nk = sparse_brick_key(q, bd);
            if !by_key.contains_key(&nk) {
                continue;
            }
            let a = tile_for[&c.key];
            let b = tile_for[&nk];
            if a != b {
                tiles.get_mut(&a).unwrap().1.insert(b);
            }
        }
    }
    let mut distance = BTreeMap::new();
    let mut queue = VecDeque::new();
    for (&t, (interface, _)) in &tiles {
        if *interface {
            distance.insert(t, 0u32);
            queue.push_back(t)
        }
    }
    while let Some(t) = queue.pop_front() {
        let d = distance[&t];
        for &n in &tiles[&t].1 {
            if let std::collections::btree_map::Entry::Vacant(e) = distance.entry(n) {
                e.insert(d + 1);
                queue.push_back(n)
            }
        }
    }
    let mut resolutions = BTreeMap::new();
    let mut maximums = BTreeMap::new();
    for c in &candidates {
        let (scale, _) = tile(c.coordinate, dims, fine, &regions);
        let t = tile_for[&c.key];
        let policy_fine = fine / scale;
        let adaptive = match distance.get(&t) {
            None => 1,
            Some(&d) => {
                let rung = policy_fine
                    .ilog2()
                    .saturating_sub(d.saturating_sub(surface).saturating_add(1));
                if d < surface {
                    policy_fine
                } else {
                    1 << rung
                }
            }
        };
        let selected = if let Some(r) = opt.fixed_resolution {
            r
        } else if let Some(tol) = opt.coarse_first_curvature_tolerance {
            if c.interface {
                curvature(scene, dims, c.coordinate, fine, tol, &world)
            } else {
                1
            }
        } else {
            adaptive
        };
        let evidence = selected.max(solid_floor(c.coordinate, fine, &world));
        let chosen = region_resolution(&regions, dims, c.coordinate, 1, evidence, fine);
        let o = c.coordinate.map(|v| v as u32 * fine);
        let e = [0, 1, 2].map(|a| fine.min(dims[a].saturating_sub(o[a])));
        let (maxr, _) = bounds_for(&regions, o, e, fine, fine);
        if !valid_rung(chosen, fine) {
            return Err(AtlasError("resolution policy returned invalid rung".into()));
        }
        resolutions.insert(c.key, chosen);
        maximums.insert(c.key, maxr);
    }
    loop {
        let mut changed = false;
        for c in &candidates {
            for d in [
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ] {
                let q = [
                    c.coordinate[0] + d[0],
                    c.coordinate[1] + d[1],
                    c.coordinate[2] + d[2],
                ];
                if q.iter()
                    .enumerate()
                    .any(|(a, &v)| v < 0 || v >= bd[a] as i32)
                {
                    continue;
                }
                let nk = sparse_brick_key(q, bd);
                if !by_key.contains_key(&nk) {
                    continue;
                }
                let mapped = maximums[&c.key].min(2 * maximums[&nk]);
                if mapped < maximums[&c.key] {
                    maximums.insert(c.key, mapped);
                    changed = true
                }
            }
        }
        if !changed {
            break;
        }
    }
    for c in &candidates {
        resolutions.insert(c.key, resolutions[&c.key].min(maximums[&c.key]));
    }
    loop {
        let mut changed = false;
        for c in &candidates {
            for d in [
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ] {
                let q = [
                    c.coordinate[0] + d[0],
                    c.coordinate[1] + d[1],
                    c.coordinate[2] + d[2],
                ];
                if q.iter()
                    .enumerate()
                    .any(|(a, &v)| v < 0 || v >= bd[a] as i32)
                {
                    continue;
                }
                let nk = sparse_brick_key(q, bd);
                if !by_key.contains_key(&nk) {
                    continue;
                }
                let own = resolutions[&c.key];
                let other = resolutions[&nk];
                if own > 2 * other {
                    let promoted = (own / 2).min(maximums[&nk]);
                    if promoted > other {
                        resolutions.insert(nk, promoted);
                        changed = true
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let bricks = candidates
        .into_iter()
        .map(|c| {
            initial_brick(
                scene,
                dims,
                c.coordinate,
                resolutions[&c.key],
                fine,
                1,
                &world,
            )
        })
        .collect();
    let atlas = atlas_with_air_support(scene, dims, bricks, fine, &regions, opt, &world)?;
    let mut atlas = if opt.coarse_first_curvature_tolerance.is_some() && regions.is_empty() {
        coarse_bulk_cover(atlas, max_span)?
    } else {
        atlas
    };
    atlas.initial_inflow_dt_s = CM12_PAPER_DT_S
        .max(opt.maximum_dt_s)
        .max(opt.fixed_dt_s.unwrap_or(0.0));
    Ok(atlas)
}

pub fn sparse_cm12_initial_active_brick_keys(
    scene: &SceneDocument,
    atlas: &SparseAdaptiveMassAtlas,
    _minimum_air_resolution: u32,
) -> BTreeSet<u32> {
    let mut active: BTreeSet<_> = atlas
        .bricks
        .iter()
        .filter(|b| b.density.iter().any(|&d| d > 0.))
        .map(|b| b.key)
        .collect();
    let opt = SparseBrickAtlasInitializationOptions {
        finest_dimensions: atlas.dimensions,
        brick_fine_resolution: atlas.brick_fine_resolution,
        maximum_dt_s: atlas.initial_inflow_dt_s,
        fixed_dt_s: Some(atlas.initial_inflow_dt_s),
        ..Default::default()
    };
    for c in inflow_support(scene, atlas.dimensions, atlas.brick_fine_resolution, &opt) {
        if let Some(b) = sparse_brick_containing_coordinate(atlas, c) {
            active.insert(b.key);
        }
    }
    let regions = parse_regions(scene, atlas.dimensions);
    if regions.is_empty() {
        return active;
    }
    let current: Vec<_> = active.iter().copied().collect();
    for key in current {
        let Some(&i) = atlas.directory.get(&key) else {
            continue;
        };
        let b = &atlas.bricks[i];
        if b.span_bricks != 1 {
            continue;
        }
        let (scale, o) = tile(
            b.coordinate,
            atlas.dimensions,
            atlas.brick_fine_resolution,
            &regions,
        );
        if scale <= 1 {
            continue;
        }
        for z in 0..scale as i32 {
            for y in 0..scale as i32 {
                for x in 0..scale as i32 {
                    let q = [o[0] + x, o[1] + y, o[2] + z];
                    if q.iter()
                        .enumerate()
                        .any(|(a, &v)| v >= atlas.brick_dimensions[a] as i32)
                    {
                        continue;
                    }
                    let k = sparse_brick_key(q, atlas.brick_dimensions);
                    if atlas.directory.contains_key(&k) {
                        active.insert(k);
                    }
                }
            }
        }
    }
    active
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Golden {
        schema_version: u32,
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Case {
        scene: SceneDocument,
        options: SparseBrickAtlasInitializationOptions,
        atlas: ExpectedAtlas,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ExpectedAtlas {
        dimensions: [u32; 3],
        brick_dimensions: [u32; 3],
        active_keys: Vec<u32>,
        bricks: Vec<ExpectedBrick>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ExpectedBrick {
        key: u32,
        coordinate: BrickCoordinate,
        span_bricks: u32,
        resolution: u32,
        density_bits: Vec<String>,
        gamma_bits: Vec<String>,
    }
    #[test]
    fn matches_typescript_source_atlas_goldens_exactly() {
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../core/testdata/sparse-atlas-golden.json"
        ))
        .unwrap();
        assert_eq!(golden.schema_version, 1);
        for case in golden.cases {
            let actual =
                initialize_sparse_brick_atlas_from_scene(&case.scene, &case.options).unwrap();
            assert_eq!(
                actual.dimensions, case.atlas.dimensions,
                "{} dimensions",
                case.scene.scene_id
            );
            assert_eq!(
                actual.brick_dimensions, case.atlas.brick_dimensions,
                "{} brick dimensions",
                case.scene.scene_id
            );
            assert_eq!(
                actual.bricks.len(),
                case.atlas.bricks.len(),
                "{} brick count",
                case.scene.scene_id
            );
            for (a, e) in actual.bricks.iter().zip(case.atlas.bricks) {
                assert_eq!(
                    (a.key, a.coordinate, a.span_bricks, a.resolution),
                    (e.key, e.coordinate, e.span_bricks, e.resolution),
                    "{} brick {} topology",
                    case.scene.scene_id,
                    e.key
                );
                let expected_density: Vec<_> = e
                    .density_bits
                    .iter()
                    .map(|v| u64::from_str_radix(v, 16).unwrap())
                    .collect();
                let expected_gamma: Vec<_> = e
                    .gamma_bits
                    .iter()
                    .map(|v| u64::from_str_radix(v, 16).unwrap())
                    .collect();
                assert_eq!(
                    a.density.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected_density,
                    "{} brick {} density words",
                    case.scene.scene_id,
                    e.key
                );
                assert_eq!(
                    a.gamma.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected_gamma,
                    "{} brick {} gamma words",
                    case.scene.scene_id,
                    e.key
                );
            }
            let active: Vec<_> = sparse_cm12_initial_active_brick_keys(&case.scene, &actual, 2)
                .into_iter()
                .collect();
            assert_eq!(
                active, case.atlas.active_keys,
                "{} active",
                case.scene.scene_id
            );
        }
    }
}
