//! Exact CPU form of the static `SolidWorld` authored by a scene document.

use crate::initial_scene::{
    lattice_dimensions, SceneDocument, SolidVoxelPatch, TerrainDescription, TerrainFeatureKind,
    TerrainGrid,
};
use crate::rigid::{StaticContactProvider, StaticContactVoxel};
use crate::scene_model::{Quaternion, Vec3};
use std::collections::BTreeMap;

pub const SOLID_WORLD_BRICK_CELLS: i32 = 8;
pub const SOLID_WORLD_TERRAIN_MATERIAL_ID: u16 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticSolidSample {
    pub solid_fraction: f32,
    pub source_fraction_u8: u8,
    pub signed_distance_cells: f32,
    pub signed_distance_q8: i16,
    pub material_id: u16,
}

impl Default for StaticSolidSample {
    fn default() -> Self {
        Self {
            solid_fraction: 0.0,
            source_fraction_u8: 0,
            signed_distance_cells: f32::INFINITY,
            signed_distance_q8: i16::MAX,
            material_id: 0,
        }
    }
}

impl StaticSolidSample {
    pub fn resident_solid(self) -> bool {
        self.source_fraction_u8 >= 128
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolidWorldPage {
    pub coordinate: [i32; 3],
    pub solid_fraction: Vec<u8>,
    pub signed_distance_q8: Vec<i16>,
    pub material_id: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledStaticSolidWorld {
    pub dimensions: [u32; 3],
    pub cell_size_m: [f64; 3],
    pub pages: Vec<SolidWorldPage>,
    pub regions: Vec<SolidVoxelPatch>,
    page_by_coordinate: BTreeMap<[i32; 3], usize>,
}

fn address(q: [i32; 3]) -> ([i32; 3], usize) {
    let page = q.map(|v| v.div_euclid(SOLID_WORLD_BRICK_CELLS));
    let local = q.map(|v| v.rem_euclid(SOLID_WORLD_BRICK_CELLS) as usize);
    (page, local[0] + 8 * (local[1] + 8 * local[2]))
}

fn empty_page(coordinate: [i32; 3]) -> SolidWorldPage {
    SolidWorldPage {
        coordinate,
        solid_fraction: vec![0; 512],
        signed_distance_q8: vec![i16::MAX; 512],
        material_id: vec![0; 512],
    }
}

fn js_round(value: f64) -> f64 {
    (value + 0.5).floor()
}

pub fn sample_terrain_grid(grid: &TerrainGrid, x: f64, z: f64) -> f64 {
    let (nx, nz) = (grid.size.nx, grid.size.nz);
    let fx = ((x - grid.origin_m.x) / grid.spacing_m).clamp(0.0, (nx - 1) as f64);
    let fz = ((z - grid.origin_m.z) / grid.spacing_m).clamp(0.0, (nz - 1) as f64);
    let x0 = (fx.floor() as usize).min(nx - 2);
    let z0 = (fz.floor() as usize).min(nz - 2);
    let (tx, tz) = (fx - x0 as f64, fz - z0 as f64);
    let at = |i: usize, j: usize| grid.heights_m.get(i + nx * j).copied().unwrap_or(0.0);
    let lower = at(x0, z0) * (1.0 - tx) + at(x0 + 1, z0) * tx;
    let upper = at(x0, z0 + 1) * (1.0 - tx) + at(x0 + 1, z0 + 1) * tx;
    (lower * (1.0 - tz) + upper * tz).max(0.0)
}

fn feature_weight(feature: &crate::initial_scene::TerrainFeature, x: f64, z: f64) -> f64 {
    let rotation = feature.rotation_rad.unwrap_or(0.0);
    let cos = rotation.cos();
    let sin = rotation.sin();
    let dx = x - feature.center_m.x;
    let dz = z - feature.center_m.z;
    let local_x = (cos * dx + sin * dz) / feature.radius_m.x;
    let local_z = (-sin * dx + cos * dz) / feature.radius_m.z;
    let distance = local_x.hypot(local_z);
    let flat = feature.flat.unwrap_or(0.45);
    if distance <= flat {
        1.0
    } else if distance >= 1.0 {
        0.0
    } else {
        let s = 1.0 - (distance - flat) / (1.0 - flat);
        s * s * (3.0 - 2.0 * s)
    }
}

pub fn terrain_height_at(terrain: Option<&TerrainDescription>, x: f64, z: f64) -> f64 {
    let Some(t) = terrain else { return 0.0 };
    if let Some(grid) = &t.grid {
        return sample_terrain_grid(grid, x, z);
    }
    // Procedural pond terrain is handled below from its authored JSON.
    if let Some(procedural) = &t.procedural {
        return pond::sample(procedural, x, z);
    }
    let mut mounds = 0.0;
    let mut carve_power = 0.0;
    for f in &t.features {
        let w = feature_weight(f, x, z);
        match f.kind {
            TerrainFeatureKind::Mound => mounds += f.amount_m * w,
            TerrainFeatureKind::Basin => carve_power += (f.amount_m * w).powf(8.0),
        }
    }
    let carve = if carve_power > 0.0 {
        carve_power.powf(1.0 / 8.0)
    } else {
        0.0
    };
    (t.base_height_m + mounds - carve).max(0.0)
}

fn scene_has_terrain(t: Option<&TerrainDescription>) -> bool {
    t.is_some_and(|v| {
        v.base_height_m > 0.0
            || !v.features.is_empty()
            || v.grid.is_some()
            || v.procedural.is_some()
    })
}

fn apply_patch(pages: &mut BTreeMap<[i32; 3], SolidWorldPage>, patch: &SolidVoxelPatch) {
    let fill = patch.operation == "fill";
    let material = patch.material_id.unwrap_or(1);
    for z in patch.minimum[2]..patch.maximum_exclusive[2] {
        for y in patch.minimum[1]..patch.maximum_exclusive[1] {
            for x in patch.minimum[0]..patch.maximum_exclusive[0] {
                let (pc, index) = address([x, y, z]);
                if !fill && !pages.contains_key(&pc) {
                    continue;
                }
                let page = pages.entry(pc).or_insert_with(|| empty_page(pc));
                page.solid_fraction[index] = if fill { 255 } else { 0 };
                page.signed_distance_q8[index] = if fill { -128 } else { i16::MAX };
                page.material_id[index] = if fill { material } else { 0 };
            }
        }
    }
}

pub fn fluid_solid_world_for_scene(scene: &SceneDocument) -> CompiledStaticSolidWorld {
    let dims = lattice_dimensions(scene);
    let c = &scene.container;
    let cell = [
        c.width_m / dims[0] as f64,
        c.height_m / dims[1] as f64,
        c.depth_m / dims[2] as f64,
    ];
    let mut pages: BTreeMap<[i32; 3], SolidWorldPage> = BTreeMap::new();
    if scene_has_terrain(scene.terrain.as_ref()) {
        for z in 0..dims[2] {
            for x in 0..dims[0] {
                // TypeScript stores the column bake in Float32Array before voxelizing.
                let wx = -0.5 * c.width_m + (x as f64 + 0.5) * cell[0];
                let wz = -0.5 * c.depth_m + (z as f64 + 0.5) * cell[2];
                let height = (terrain_height_at(scene.terrain.as_ref(), wx, wz).min(c.height_m)
                    as f32) as f64;
                let covered = (height / cell[1]).ceil().min(dims[1] as f64) as u32;
                for y in 0..covered {
                    let fraction = ((height - y as f64 * cell[1]) / cell[1]).clamp(0.0, 1.0);
                    let quantized = js_round(255.0 * fraction) as u8;
                    if quantized == 0 {
                        continue;
                    }
                    let (pc, index) = address([x as i32, y as i32, z as i32]);
                    let page = pages.entry(pc).or_insert_with(|| empty_page(pc));
                    page.solid_fraction[index] = quantized;
                    page.signed_distance_q8[index] =
                        js_round((((y as f64 + 0.5) * cell[1] - height) / cell[1]) * 256.0)
                            .clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                    page.material_id[index] = SOLID_WORLD_TERRAIN_MATERIAL_ID;
                }
            }
        }
    }
    for patch in &scene.solid_voxels {
        apply_patch(&mut pages, patch)
    }
    pages.retain(|_, p| p.solid_fraction.iter().any(|&v| v > 0));
    let mut regions = scenery_collider_regions(scene, cell);
    let mut pages: Vec<_> = pages.into_values().collect();
    pages.sort_by_key(|page| (page.coordinate[2], page.coordinate[1], page.coordinate[0]));
    let page_by_coordinate = pages
        .iter()
        .enumerate()
        .map(|(i, p)| (p.coordinate, i))
        .collect();
    CompiledStaticSolidWorld {
        dimensions: dims,
        cell_size_m: cell,
        pages,
        regions: std::mem::take(&mut regions),
        page_by_coordinate,
    }
}

impl CompiledStaticSolidWorld {
    pub fn sample(&self, q: [i32; 3]) -> StaticSolidSample {
        let (pc, index) = address(q);
        let mut result = self
            .page_by_coordinate
            .get(&pc)
            .map(|&i| StaticSolidSample {
                solid_fraction: self.pages[i].solid_fraction[index] as f32 / 255.0,
                source_fraction_u8: self.pages[i].solid_fraction[index],
                signed_distance_cells: self.pages[i].signed_distance_q8[index] as f32 / 256.0,
                signed_distance_q8: self.pages[i].signed_distance_q8[index],
                material_id: self.pages[i].material_id[index],
            })
            .unwrap_or_default();
        for region in &self.regions {
            if (0..3).all(|a| q[a] >= region.minimum[a] && q[a] < region.maximum_exclusive[a]) {
                let fill = region.operation == "fill";
                result = if fill {
                    StaticSolidSample {
                        solid_fraction: 1.0,
                        source_fraction_u8: 255,
                        signed_distance_cells: -0.5,
                        signed_distance_q8: -128,
                        material_id: region.material_id.unwrap_or(1),
                    }
                } else {
                    StaticSolidSample {
                        solid_fraction: 0.0,
                        source_fraction_u8: 0,
                        signed_distance_cells: i16::MAX as f32 / 256.0,
                        signed_distance_q8: i16::MAX,
                        material_id: 0,
                    }
                };
            }
        }
        result
    }
}

impl StaticContactProvider for CompiledStaticSolidWorld {
    fn cell_size_m(&self) -> [f32; 3] {
        self.cell_size_m.map(|v| v as f32)
    }
    fn visit_candidates(
        &self,
        centre: [f32; 3],
        radius: f32,
        visitor: &mut dyn FnMut(StaticContactVoxel),
    ) {
        let origin = [
            -0.5 * self.dimensions[0] as f64 * self.cell_size_m[0],
            0.0,
            -0.5 * self.dimensions[2] as f64 * self.cell_size_m[2],
        ];
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        for a in 0..3 {
            lo[a] =
                (((centre[a] - radius) as f64 - origin[a]) / self.cell_size_m[a]).floor() as i32;
            hi[a] =
                (((centre[a] + radius) as f64 - origin[a]) / self.cell_size_m[a]).floor() as i32;
        }
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let q = [x, y, z];
                    let s = self.sample(q);
                    if s.source_fraction_u8 == 0 {
                        continue;
                    }
                    let min = [
                        origin[0] + x as f64 * self.cell_size_m[0],
                        origin[1] + y as f64 * self.cell_size_m[1],
                        origin[2] + z as f64 * self.cell_size_m[2],
                    ];
                    visitor(StaticContactVoxel {
                        coordinate: q,
                        fraction: s.solid_fraction,
                        minimum_m: min.map(|v| v as f32),
                        maximum_m: [
                            min[0] + self.cell_size_m[0],
                            min[1] + self.cell_size_m[1],
                            min[2] + self.cell_size_m[2],
                        ]
                        .map(|v| v as f32),
                    });
                }
            }
        }
    }
    fn signed_distance_cells(&self, q: [i32; 3]) -> f32 {
        self.sample(q).signed_distance_cells
    }
}

#[derive(Clone, Copy)]
struct Frame {
    origin: Vec3,
    scale: f64,
    orientation: Option<Quaternion>,
    units_metres: bool,
}
fn qmul(a: Quaternion, b: Quaternion) -> Quaternion {
    Quaternion {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    }
}
fn rotate(q: Option<Quaternion>, v: Vec3) -> Vec3 {
    let Some(q) = q else { return v };
    let tx = 2.0 * (q.y * v.z - q.z * v.y);
    let ty = 2.0 * (q.z * v.x - q.x * v.z);
    let tz = 2.0 * (q.x * v.y - q.y * v.x);
    Vec3 {
        x: v.x + q.w * tx + (q.y * tz - q.z * ty),
        y: v.y + q.w * ty + (q.z * tx - q.x * tz),
        z: v.z + q.w * tz + (q.x * ty - q.y * tx),
    }
}
fn scenery_collider_regions(scene: &SceneDocument, cell: [f64; 3]) -> Vec<SolidVoxelPatch> {
    let Some(graph) = scene.scenery.as_ref() else {
        return vec![];
    };
    let Some(nodes) = graph.get("nodes").and_then(|v| v.as_array()) else {
        return vec![];
    };
    let root = Frame {
        origin: Vec3::default(),
        scale: 1.0,
        orientation: None,
        units_metres: false,
    };
    let mut boxes = vec![];
    fn visit(
        scene: &SceneDocument,
        nodes: &[serde_json::Value],
        parent: Frame,
        out: &mut Vec<(Vec3, Vec3)>,
    ) {
        for node in nodes {
            let place = node.get("place");
            let metres = place
                .and_then(|p| p.get("units"))
                .and_then(|v| v.as_str())
                .map(|v| v == "metres")
                .unwrap_or(parent.units_metres);
            let unit = if metres {
                1.0
            } else {
                scene
                    .container
                    .width_m
                    .max(scene.container.height_m)
                    .max(scene.container.depth_m)
            };
            let scale = parent.scale
                * place
                    .and_then(|p| p.get("scale"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
            let local: Vec3 = place
                .and_then(|p| p.get("position"))
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            let offset = rotate(
                parent.orientation,
                Vec3 {
                    x: local.x * unit * parent.scale,
                    y: local.y * unit * parent.scale,
                    z: local.z * unit * parent.scale,
                },
            );
            let mut origin = Vec3 {
                x: parent.origin.x + offset.x,
                y: parent.origin.y + offset.y,
                z: parent.origin.z + offset.z,
            };
            let anchor = place.and_then(|p| p.get("anchor")).and_then(|v| v.as_str());
            if matches!(anchor, Some("floor") | Some("terrain")) {
                let world_scale = scene
                    .container
                    .width_m
                    .max(scene.container.height_m)
                    .max(scene.container.depth_m);
                let floor = match scene.environment.as_deref() {
                    Some("night-lab") => -0.72 * world_scale,
                    Some("garden") => scene
                        .terrain
                        .as_ref()
                        .map(|t| t.base_height_m)
                        .unwrap_or(0.0),
                    _ => -0.025,
                };
                let ground = place
                    .and_then(|p| p.get("ground"))
                    .and_then(|v| v.as_array());
                let gx = ground
                    .and_then(|g| g.first())
                    .and_then(|v| v.as_f64())
                    .unwrap_or(local.x)
                    * unit;
                let gz = ground
                    .and_then(|g| g.get(1))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(local.z)
                    * unit;
                let datum = if anchor == Some("terrain") && scene.terrain.is_some() {
                    terrain_height_at(scene.terrain.as_ref(), gx, gz)
                } else {
                    floor
                };
                origin.y = datum + offset.y;
            }
            let own_q = place
                .and_then(|p| p.get("orientation"))
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok());
            let orientation = own_q
                .map(|q| qmul(parent.orientation.unwrap_or_default(), q))
                .or(parent.orientation);
            let frame = Frame {
                origin,
                scale,
                orientation,
                units_metres: metres,
            };
            let kind = node.get("kind").and_then(|v| v.as_str());
            if matches!(kind, Some("group") | Some("recursive-shape")) {
                if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
                    visit(scene, children, frame, out)
                }
                continue;
            }
            let collider = node
                .get("tags")
                .and_then(|v| v.as_array())
                .is_some_and(|a| a.iter().any(|v| v.as_str() == Some("fluid-collider")));
            if kind == Some("box") && collider && orientation.is_none() {
                if let Some(h) = node
                    .get("halfSize")
                    .cloned()
                    .and_then(|v| serde_json::from_value::<Vec3>(v).ok())
                {
                    let factor = unit * scale;
                    out.push((
                        origin,
                        Vec3 {
                            x: h.x * factor,
                            y: h.y * factor,
                            z: h.z * factor,
                        },
                    ))
                }
            }
        }
    }
    visit(scene, nodes, root, &mut boxes);
    let origin = [
        -0.5 * scene.container.width_m,
        0.0,
        -0.5 * scene.container.depth_m,
    ];
    let eps = 1e-9;
    boxes
        .into_iter()
        .filter_map(|(p, h)| {
            let min = [p.x - h.x, p.y - h.y, p.z - h.z];
            let max = [p.x + h.x, p.y + h.y, p.z + h.z];
            let minimum = [0, 1, 2].map(|a| ((min[a] - origin[a]) / cell[a] + eps).floor() as i32);
            let maximum_exclusive = [0, 1, 2].map(|a| {
                let v = ((max[a] - origin[a]) / cell[a] - eps).ceil();
                if v == 0.0 {
                    0
                } else {
                    v as i32
                }
            });
            if (0..3).any(|a| minimum[a] >= maximum_exclusive[a]) {
                None
            } else {
                Some(SolidVoxelPatch {
                    operation: "fill".into(),
                    minimum,
                    maximum_exclusive,
                    material_id: None,
                })
            }
        })
        .collect()
}

mod pond {
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct P {
        spec: S,
        container: C,
    }
    #[derive(Deserialize)]
    struct C {
        #[serde(rename = "height_m")]
        h: f64,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct S {
        #[serde(rename = "center_m")]
        center: [f64; 2],
        #[serde(rename = "radius_m")]
        radius: [f64; 2],
        #[serde(rename = "groundHeight_m")]
        ground: f64,
        #[serde(rename = "basinDepth_m")]
        depth: f64,
        #[serde(rename = "rimHeight_m")]
        rim_h: f64,
        #[serde(rename = "rimHalfWidth_m")]
        rim_w: f64,
        crest: Option<String>,
        crest_wall: Option<W>,
        #[serde(rename = "innerFace_m")]
        inner: f64,
        floor_dish: Option<f64>,
        beach: Option<B>,
        lobes: usize,
        wobble: f64,
        section_height_variation: f64,
        section_width_variation: f64,
        #[serde(rename = "relief_m")]
        relief: f64,
        seed: i32,
        #[serde(default)]
        terraces: Vec<T>,
    }
    #[derive(Clone, Copy, Deserialize)]
    struct W {
        #[serde(rename = "crestRadius_m")]
        crest: f64,
        #[serde(rename = "footRadius_m")]
        foot: f64,
        batter_rad: f64,
    }
    #[derive(Deserialize)]
    struct B {
        turn: f64,
        width: f64,
        #[serde(rename = "innerFace_m")]
        inner: f64,
    }
    #[derive(Deserialize)]
    struct T {
        #[serde(rename = "center_m")]
        center: [f64; 2],
        #[serde(rename = "radius_m")]
        radius: [f64; 2],
        #[serde(rename = "height_m")]
        height: f64,
        rotation_rad: Option<f64>,
        flat: Option<f64>,
        #[serde(rename = "faceRun_m")]
        face_run: Option<f64>,
        wall: Option<W>,
        wobble: Option<f64>,
        lobes: Option<i32>,
    }
    fn hash(n: i32) -> f64 {
        let mut h = (n ^ 0x9e3779b9_u32 as i32).wrapping_mul(0x85ebca6b_u32 as i32) as u32;
        h = (h ^ (h >> 13)).wrapping_mul(0xc2b2ae35);
        ((h ^ (h >> 16)) as f64) / 4294967296.0
    }
    fn hs(n: i32) -> f64 {
        2.0 * hash(n) - 1.0
    }
    fn ramp(v: f64) -> f64 {
        let t = v.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
    fn bw(b: Option<&B>, turn: f64) -> f64 {
        let Some(b) = b else { return 0.0 };
        if b.width <= 0.0 {
            return 0.0;
        }
        let d = ((((turn - b.turn + 0.5) % 1.0) + 1.0) % 1.0 - 0.5).abs();
        ramp(1.0 - d / b.width)
    }
    fn sway(t: f64, seed: i32) -> f64 {
        0.58 * (std::f64::consts::TAU * (2.0 * t + hash(seed.wrapping_add(0x51470000)))).sin()
            + 0.42 * (std::f64::consts::TAU * (3.0 * t + hash(seed.wrapping_add(0x51470001)))).sin()
    }
    fn curve(s: &S) -> Vec<[f64; 2]> {
        let g: Vec<_> = (0..s.lobes)
            .map(|i| hs(s.seed.wrapping_add(31 * i as i32)))
            .collect();
        let at =
            |i: isize| g[((i % s.lobes as isize + s.lobes as isize) % s.lobes as isize) as usize];
        let mut out = vec![];
        for i in 0..s.lobes {
            let (a, b, c, d) = (
                at(i as isize - 1),
                at(i as isize),
                at(i as isize + 1),
                at(i as isize + 2),
            );
            for k in 0..16 {
                let t = k as f64 / 16.0;
                let t2 = t * t;
                let t3 = t2 * t;
                let grain = 0.5
                    * (2.0 * b
                        + (c - a) * t
                        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                        + (3.0 * b - 3.0 * c + d - a) * t3);
                let turn = (i as f64 + t) / s.lobes as f64;
                let a = std::f64::consts::TAU * turn;
                let (co, si) = (a.cos(), a.sin());
                let bearing = (s.radius[1] * si).atan2(s.radius[0] * co) / std::f64::consts::TAU;
                let scale = 1.0 + s.wobble * (0.38 * grain + 0.62 * sway(turn, s.seed))
                    - 0.12 * bw(s.beach.as_ref(), bearing);
                out.push([
                    s.center[0] + s.radius[0] * scale * co,
                    s.center[1] + s.radius[1] * scale * si,
                ]);
            }
        }
        out
    }
    fn distance(p: &[[f64; 2]], x: f64, z: f64) -> f64 {
        let (mut nearest, mut inside) = (f64::INFINITY, false);
        for i in 0..p.len() {
            let [a, b] = p[i];
            let [c, d] = p[(i + 1) % p.len()];
            let (dx, dz) = (c - a, d - b);
            let l = dx * dx + dz * dz;
            let t = if l > 0.0 {
                (((x - a) * dx + (z - b) * dz) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (ox, oz) = (x - a - t * dx, z - b - t * dz);
            nearest = nearest.min(ox * ox + oz * oz);
            if (b > z) != (d > z) && x < a + ((z - b) / (d - b)) * (c - a) {
                inside = !inside
            }
        }
        nearest.sqrt() * if inside { -1.0 } else { 1.0 }
    }
    fn spline(v: &[f64], turn: f64) -> f64 {
        let n = v.len();
        let p = ((turn % 1.0) + 1.0) % 1.0 * n as f64;
        let i = p.floor() as isize;
        let t = p - i as f64;
        let (t2, t3) = (t * t, t * t * t);
        let at = |o: isize| v[((i + o) % n as isize + n as isize) as usize % n];
        let (a, b, c, d) = (at(-1), at(0), at(1), at(2));
        0.5 * (2.0 * b
            + (c - a) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (3.0 * b - 3.0 * c + d - a) * t3)
    }
    fn mods(s: &S, salt: i32) -> Vec<f64> {
        (0..s.lobes + 4)
            .map(|i| hs(s.seed.wrapping_add(salt).wrapping_add(61 * i as i32)))
            .collect()
    }
    fn wall(w: W, rise: f64, d: f64) -> f64 {
        let b = w.batter_rad.max(0.02);
        let (co, si) = (b.cos(), b.sin());
        let appetite = (w.crest + w.foot) * (1.0 - si);
        let fit = if appetite > rise {
            rise / appetite
        } else {
            1.0
        };
        let (c, f) = (w.crest * fit, w.foot * fit);
        let cy = rise - c;
        let td = c * co;
        let ty = cy + c * si;
        let fty = f * (1.0 - si);
        let ftd = td + (ty - fty) * b.tan();
        let fd = ftd + f * co;
        if d <= 0.0 {
            rise
        } else if d >= fd {
            0.0
        } else if d <= td {
            cy + (c * c - d * d).max(0.0).sqrt()
        } else if d >= ftd {
            f - (f * f - (d - fd) * (d - fd)).max(0.0).sqrt()
        } else {
            ty - (d - td) / b.tan()
        }
    }
    fn wall_run(w: W, r: f64) -> f64 {
        let mut d = 0.0;
        while d < 4.0 * r + w.crest + w.foot {
            if wall(w, r, d) <= 0.0 {
                return d;
            }
            d += 0.0005
        }
        d
    }
    fn inner(u: f64) -> f64 {
        let t = u.clamp(0.0, 1.0);
        let slope = 1.0 / (1.0 - 0.35 / 2.0);
        if t <= 0.65 {
            slope * t
        } else {
            let v = (t - 0.65) / 0.35;
            slope * (0.65 + 0.35 * (v - 0.5 * v * v))
        }
    }
    fn relief(x: f64, z: f64, seed: i32) -> f64 {
        let (mut total, mut amp, mut freq, mut weight) = (0.0, 1.0, 26.0, 0.0);
        for octave in 0_i32..2 {
            let (px, pz) = (x * freq, z * freq);
            let (cx, cz) = (px.floor() as i32, pz.floor() as i32);
            let (fx, fz) = (px - cx as f64, pz - cz as f64);
            let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
            let q = |i: i32, j: i32| {
                hs(seed
                    .wrapping_add(0x9e37 * octave)
                    .wrapping_add(73_856_093_i32.wrapping_mul(cx + i))
                    .wrapping_add(19_349_663_i32.wrapping_mul(cz + j)))
            };
            let lo = q(0, 0) * (1.0 - sx) + q(1, 0) * sx;
            let hi = q(0, 1) * (1.0 - sx) + q(1, 1) * sx;
            total += amp * (lo * (1.0 - sz) + hi * sz);
            weight += amp;
            amp *= 0.45;
            freq *= 2.7
        }
        total / weight
    }
    fn tw(t: &T, x: f64, z: f64) -> f64 {
        let r = t.rotation_rad.unwrap_or(0.0);
        let (si, co) = r.sin_cos();
        let (dx, dz) = (x - t.center[0], z - t.center[1]);
        let (lx, lz) = (co * dx + si * dz, -si * dx + co * dz);
        let d = (lx / t.radius[0]).hypot(lz / t.radius[1]);
        let i = t
            .face_run
            .map(|f| (1.0 - (f / (t.radius[0] * t.radius[1]).sqrt()).clamp(1e-3, 0.95)).max(0.0))
            .unwrap_or(t.flat.unwrap_or(0.45));
        if d <= i {
            1.0
        } else if d >= 1.0 {
            0.0
        } else {
            ramp(1.0 - (d - i) / (1.0 - i))
        }
    }
    fn tl(t: &T, x: f64, z: f64) -> f64 {
        let Some(w) = t.wall else {
            return t.height * tw(t, x, z);
        };
        let r = t.rotation_rad.unwrap_or(0.0);
        let (si, co) = r.sin_cos();
        let (dx, dz) = (x - t.center[0], z - t.center[1]);
        let (lx, lz) = (co * dx + si * dz, -si * dx + co * dz);
        let wander = 1.0
            + t.wobble.unwrap_or(0.0) * ((t.lobes.unwrap_or(5) as f64) * lz.atan2(lx) + 0.7).sin();
        let (rx, rz) = (
            (t.radius[0] * wander).max(1e-4),
            (t.radius[1] * wander).max(1e-4),
        );
        let q = (lx / rx).hypot(lz / rz);
        let g = (lx / (rx * rx)).hypot(lz / (rz * rz));
        wall(
            w,
            t.height,
            if g > 1e-9 {
                (q - 1.0) * q / g
            } else {
                (q - 1.0) * (rx * rz).sqrt()
            },
        )
    }
    pub fn sample(value: &serde_json::Value, x: f64, z: f64) -> f64 {
        let p: P = serde_json::from_value(value.clone()).expect("valid pond terrain");
        let s = &p.spec;
        let c = curve(s);
        let d = distance(&c, x, z);
        let turn = (z - s.center[1]).atan2(x - s.center[0]) / std::f64::consts::TAU;
        let rh = s.rim_h * (1.0 + s.section_height_variation * spline(&mods(s, 0), turn));
        let rw = s.rim_w * (1.0 + s.section_width_variation * spline(&mods(s, 977), turn));
        let inward = -d - rw;
        let dish = s.floor_dish.unwrap_or(0.45);
        let face = s
            .beach
            .as_ref()
            .map(|b| s.inner + (b.inner - s.inner) * bw(Some(b), turn))
            .unwrap_or(s.inner);
        let ir = c
            .iter()
            .map(|q| (q[0] - s.center[0]).hypot(q[1] - s.center[1]))
            .fold(f64::INFINITY, f64::min);
        let reach = (0.9 * (ir - s.rim_w)).max(1e-4);
        let fall = if d >= 0.0 {
            0.0
        } else {
            s.depth * ((1.0 - dish) * inner(inward / face) + dish * ramp(inward / reach))
        };
        let crest = if s.crest.as_deref() == Some("flat") {
            0.0
        } else if s.crest.as_deref() == Some("wall") && s.crest_wall.is_some() {
            let w = s.crest_wall.unwrap();
            wall(w, rh, d.abs() - (rw - wall_run(w, rh)).max(0.0))
        } else {
            let t = (d.abs() / rw).clamp(0.0, 1.0);
            rh * (1.0 - t * t).powf(1.5)
        };
        let outside = ramp((d - rw) / rw);
        let lift = s
            .terraces
            .iter()
            .map(|t| tl(t, x, z) * outside)
            .fold(0.0, f64::max);
        (s.ground - fall + crest + lift + s.relief * relief(x, z, s.seed ^ 0x51ed2701))
            .clamp(0.0, p.container.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Golden {
        schema_version: u32,
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        scene: SceneDocument,
        dimensions: [u32; 3],
        pages: Vec<Page>,
        regions: Vec<SolidVoxelPatch>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Page {
        coordinate: [i32; 3],
        solid_fraction: Vec<u8>,
        signed_distance_q8: Vec<i16>,
        material_id: Vec<u16>,
    }

    #[test]
    fn compiled_world_matches_typescript_pages_and_regions_exactly() {
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../core/testdata/solid-world-golden.json"
        ))
        .unwrap();
        assert_eq!(golden.schema_version, 1);
        for case in golden.cases {
            let world = fluid_solid_world_for_scene(&case.scene);
            assert_eq!(
                world.dimensions, case.dimensions,
                "{} dims",
                case.scene.scene_id
            );
            assert_eq!(
                world.regions, case.regions,
                "{} regions",
                case.scene.scene_id
            );
            assert_eq!(
                world.pages.len(),
                case.pages.len(),
                "{} page count",
                case.scene.scene_id
            );
            for (actual, expected) in world.pages.iter().zip(case.pages) {
                assert_eq!(
                    actual.coordinate, expected.coordinate,
                    "{} coordinate",
                    case.scene.scene_id
                );
                assert_eq!(
                    actual.solid_fraction, expected.solid_fraction,
                    "{} fraction page {:?}",
                    case.scene.scene_id, actual.coordinate
                );
                assert_eq!(
                    actual.signed_distance_q8, expected.signed_distance_q8,
                    "{} sdf page {:?}",
                    case.scene.scene_id, actual.coordinate
                );
                assert_eq!(
                    actual.material_id, expected.material_id,
                    "{} material page {:?}",
                    case.scene.scene_id, actual.coordinate
                );
            }
        }
    }

    #[test]
    fn samples_signed_pages_regions_and_empty_space() {
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../core/testdata/solid-world-golden.json"
        ))
        .unwrap();
        let case = &golden.cases[3];
        let world = fluid_solid_world_for_scene(&case.scene);
        assert!(world.sample([0, -1, 0]).resident_solid());
        assert_eq!(world.sample([0, 0, 0]).signed_distance_cells, f32::INFINITY);
        let region = &world.regions[0];
        let q = region.minimum;
        let sample = world.sample(q);
        assert_eq!(sample.source_fraction_u8, 255);
        assert_eq!(sample.signed_distance_q8, -128);
        assert_eq!(sample.material_id, 1);
    }
}
