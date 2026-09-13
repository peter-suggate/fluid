//! Production-scene physical sampling helpers.
//!
//! This module owns the dimensional reduction at the Rust boundary: source Y
//! is reflected into canvas Y exactly once, while rigid containment stays in
//! authored metre space.

use crate::scene_model::{PhysicalScene, Quaternion, RigidBodyDescription, RigidShape, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneDocument {
    pub schema_version: String,
    pub scene_id: String,
    #[serde(default)]
    pub environment: Option<String>,
    #[serde(default)]
    pub systems: SceneSystems,
    pub container: ContainerInput,
    pub voxel_domain: VoxelDomainInput,
    #[serde(default)]
    pub numerics: NumericsInput,
    #[serde(default)]
    pub solid_voxels: Vec<SolidVoxelPatch>,
    pub fluid: FluidInput,
    #[serde(default)]
    pub rigid_bodies: Vec<RigidBodyDescription>,
    #[serde(default)]
    pub terrain: Option<TerrainDescription>,
    #[serde(default)]
    pub scenery: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericsInput {
    #[serde(rename = "fixedDt_s", default = "default_fixed_dt")]
    pub fixed_dt_s: f64,
    #[serde(rename = "maxDt_s", default)]
    pub max_dt_s: f64,
}
impl Default for NumericsInput {
    fn default() -> Self {
        Self {
            fixed_dt_s: default_fixed_dt(),
            max_dt_s: 0.0,
        }
    }
}
fn default_fixed_dt() -> f64 {
    1.0 / 60.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainDescription {
    #[serde(rename = "baseHeight_m")]
    pub base_height_m: f64,
    #[serde(default)]
    pub features: Vec<TerrainFeature>,
    #[serde(default)]
    pub grid: Option<TerrainGrid>,
    #[serde(default)]
    pub procedural: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainFeature {
    pub kind: TerrainFeatureKind,
    #[serde(rename = "center_m")]
    pub center_m: HeightFieldCenter,
    #[serde(rename = "radius_m")]
    pub radius_m: HeightFieldCenter,
    #[serde(rename = "amount_m")]
    pub amount_m: f64,
    #[serde(default)]
    #[serde(rename = "rotation_rad")]
    pub rotation_rad: Option<f64>,
    #[serde(default)]
    pub flat: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TerrainFeatureKind {
    Basin,
    Mound,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainGrid {
    pub kind: String,
    #[serde(rename = "origin_m")]
    pub origin_m: HeightFieldCenter,
    #[serde(rename = "spacing_m")]
    pub spacing_m: f64,
    pub size: TerrainGridSize,
    #[serde(rename = "heights_m")]
    pub heights_m: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainGridSize {
    pub nx: usize,
    pub nz: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSystems {
    #[serde(default = "yes")]
    pub fluid: bool,
}
impl Default for SceneSystems {
    fn default() -> Self {
        Self { fluid: true }
    }
}
fn yes() -> bool {
    true
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerInput {
    #[serde(rename = "width_m")]
    pub width_m: f64,
    #[serde(rename = "height_m")]
    pub height_m: f64,
    #[serde(rename = "depth_m")]
    pub depth_m: f64,
    pub fill_fraction: f64,
    pub top: String,
    #[serde(default)]
    pub depth_boundary: Option<String>,
    #[serde(default)]
    pub shape: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoxelDomainInput {
    #[serde(rename = "finestCellSize_m")]
    pub finest_cell_size_m: f64,
    #[serde(rename = "brickSize_cells")]
    pub brick_size_cells: u8,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolidVoxelPatch {
    pub operation: String,
    pub minimum: [i32; 3],
    pub maximum_exclusive: [i32; 3],
    #[serde(default)]
    pub material_id: Option<u16>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "kebab-case")]
pub enum InitialLiquidVolume {
    Box {
        #[serde(rename = "min_m")]
        min_m: Vec3,
        #[serde(rename = "max_m")]
        max_m: Vec3,
    },
    Sphere {
        #[serde(rename = "center_m")]
        center_m: Vec3,
        #[serde(rename = "radius_m")]
        radius_m: f64,
    },
    Hemisphere {
        #[serde(rename = "center_m")]
        center_m: Vec3,
        #[serde(rename = "radius_m")]
        radius_m: f64,
        #[serde(rename = "outwardNormal")]
        outward_normal: Vec3,
    },
    Cylinder {
        #[serde(rename = "center_m")]
        center_m: Vec3,
        #[serde(rename = "radius_m")]
        radius_m: f64,
        #[serde(rename = "halfHeight_m")]
        half_height_m: f64,
    },
    Torus {
        #[serde(rename = "center_m")]
        center_m: Vec3,
        #[serde(rename = "radius_m")]
        radius_m: f64,
        #[serde(rename = "tubeRadius_m")]
        tube_radius_m: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum InitialLiquidHeightField {
    Quadratic {
        #[serde(rename = "baseHeight_m")]
        base_height_m: f64,
        #[serde(rename = "center_m")]
        center_m: HeightFieldCenter,
        #[serde(rename = "curvatureX_mInv")]
        curvature_x_m_inv: f64,
        #[serde(rename = "curvatureZ_mInv")]
        curvature_z_m_inv: f64,
    },
    Cosine {
        #[serde(rename = "baseHeight_m")]
        base_height_m: f64,
        #[serde(rename = "amplitude_m")]
        amplitude_m: f64,
        #[serde(rename = "wavelength_m")]
        wavelength_m: f64,
        #[serde(rename = "originX_m")]
        origin_x_m: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeightFieldCenter {
    pub x: f64,
    pub z: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FluidInput {
    #[serde(rename = "density_kg_m3")]
    pub density_kg_m3: f64,
    #[serde(rename = "dynamicViscosity_Pa_s")]
    pub dynamic_viscosity_pa_s: f64,
    #[serde(rename = "surfaceTension_N_m")]
    pub surface_tension_n_m: f64,
    #[serde(rename = "gravity_m_s2")]
    pub gravity_m_s2: Vec3,
    #[serde(default)]
    #[serde(rename = "initialVelocity_m_s")]
    pub initial_velocity_m_s: Option<Vec3>,
    pub initial_condition: String,
    #[serde(default)]
    #[serde(rename = "initialDamBreakDimensions_m")]
    pub initial_dam_break_dimensions_m: Option<Vec3>,
    #[serde(default)]
    #[serde(rename = "initialDamBreakOrigin_m")]
    pub initial_dam_break_origin_m: Option<Vec3>,
    #[serde(default)]
    #[serde(rename = "initialBrickSeeds_m")]
    pub initial_brick_seeds_m: Option<Vec<Vec3>>,
    #[serde(default)]
    pub initial_brick_seeds_additive: bool,
    #[serde(default)]
    pub initial_liquid_volumes: Vec<InitialLiquidVolume>,
    #[serde(default)]
    pub inflow: Option<crate::scene_model::FluidInflow>,
    #[serde(default)]
    pub initial_height_field: Option<InitialLiquidHeightField>,
    #[serde(default)]
    pub refinement_regions: Vec<serde_json::Value>,
}

pub fn lattice_dimensions(scene: &SceneDocument) -> [u32; 3] {
    let axis = |extent: f64| {
        ((extent / scene.voxel_domain.finest_cell_size_m + 0.5).floor() as u32).clamp(8, 2048)
    };
    [
        axis(scene.container.width_m),
        axis(scene.container.height_m),
        axis(scene.container.depth_m),
    ]
}
fn overlap(cell: u32, count: u32, minimum: f64, maximum: f64) -> f64 {
    let lo = cell as f64 / count as f64;
    let hi = (cell + 1) as f64 / count as f64;
    let v = (hi.min(maximum) - lo.max(minimum)).max(0.0) * count as f64;
    if v <= 1e-12 {
        0.0
    } else if v >= 1.0 - 1e-12 {
        1.0
    } else {
        v
    }
}
/// Exact procedural tank-fill/dam-break fraction before painted bricks,
/// height fields, and analytic-volume union.
pub fn base_initial_liquid_fraction(scene: &SceneDocument, q: [u32; 3], dims: [u32; 3]) -> f64 {
    if !scene.systems.fluid || q.iter().zip(dims).any(|(&v, n)| v >= n) {
        return 0.0;
    }
    if scene.fluid.initial_condition == "tank-fill" {
        return overlap(q[1], dims[1], 0.0, scene.container.fill_fraction);
    }
    let size = scene
        .fluid
        .initial_dam_break_dimensions_m
        .map(|d| {
            [
                d.x / scene.container.width_m,
                d.y / scene.container.height_m,
                d.z / scene.container.depth_m,
            ]
        })
        .unwrap_or_else(|| {
            let fill = scene.container.fill_fraction.clamp(0.0, 1.0);
            if fill == 0.0 {
                [0.0; 3]
            } else {
                let h = 0.92_f64.max(fill);
                let foot = (fill / h).sqrt();
                [foot, h, foot]
            }
        });
    let origin = scene
        .fluid
        .initial_dam_break_origin_m
        .map(|p| {
            [
                p.x / scene.container.width_m,
                p.y / scene.container.height_m,
                p.z / scene.container.depth_m,
            ]
        })
        .unwrap_or([0.0; 3]);
    overlap(q[0], dims[0], origin[0], origin[0] + size[0])
        * overlap(q[1], dims[1], origin[1], origin[1] + size[1])
        * overlap(q[2], dims[2], origin[2], origin[2] + size[2])
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceRigidRaster {
    pub capacity_multiplier: Vec<f32>,
    pub aperture_x_multiplier: Vec<f32>,
    pub aperture_y_multiplier: Vec<f32>,
    pub solid_velocity_x_fine: Vec<f32>,
    pub solid_velocity_y_fine: Vec<f32>,
}

fn inverse_rotate(q: Quaternion, v: Vec3) -> Vec3 {
    let norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    let (w, x, y, z) = if norm > 0.0 {
        (
            q.w / norm.sqrt(),
            -q.x / norm.sqrt(),
            -q.y / norm.sqrt(),
            -q.z / norm.sqrt(),
        )
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let tx = 2.0 * (y * v.z - z * v.y);
    let ty = 2.0 * (z * v.x - x * v.z);
    let tz = 2.0 * (x * v.y - y * v.x);
    Vec3 {
        x: v.x + w * tx + (y * tz - z * ty),
        y: v.y + w * ty + (z * tx - x * tz),
        z: v.z + w * tz + (x * ty - y * tx),
    }
}

pub fn rigid_shape_contains(shape: RigidShape, d: Vec3, p: Vec3) -> bool {
    match shape {
        RigidShape::Sphere => p.x.hypot(p.y).hypot(p.z) <= d.x,
        RigidShape::Box => {
            p.x.abs() <= 0.5 * d.x && p.y.abs() <= 0.5 * d.y && p.z.abs() <= 0.5 * d.z
        }
        RigidShape::Capsule => {
            let cy = p.y.clamp(-0.5 * d.y, 0.5 * d.y);
            p.x.hypot(p.y - cy).hypot(p.z) <= d.x
        }
        RigidShape::Cylinder => p.x * p.x + p.z * p.z <= d.x * d.x && p.y.abs() <= 0.5 * d.y,
        RigidShape::Cup => {
            let t = d.z.clamp(1e-4, (0.95 * d.x).min(0.95 * d.y));
            let qx = p.x.hypot(p.z) - d.x;
            let qy = p.y.abs() - 0.5 * d.y;
            let outer = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0);
            let cavity = (p.x.hypot(p.z) - (d.x - t)).max((-0.5 * d.y + t) - p.y);
            outer.max(-cavity) <= 0.0
        }
    }
}

fn contains(body: &RigidBodyDescription, p: Vec3) -> bool {
    rigid_shape_contains(
        body.shape,
        body.dimensions_m,
        inverse_rotate(
            body.orientation,
            Vec3 {
                x: p.x - body.position_m.x,
                y: p.y - body.position_m.y,
                z: p.z - body.position_m.z,
            },
        ),
    )
}
fn velocity(body: &RigidBodyDescription, p: Vec3) -> Vec3 {
    let r = Vec3 {
        x: p.x - body.position_m.x,
        y: p.y - body.position_m.y,
        z: p.z - body.position_m.z,
    };
    let a = body.angular_velocity_rad_s;
    Vec3 {
        x: body.linear_velocity_m_s.x + a.y * r.z - a.z * r.y,
        y: body.linear_velocity_m_s.y + a.z * r.x - a.x * r.z,
        z: body.linear_velocity_m_s.z + a.x * r.y - a.y * r.x,
    }
}
fn owner<'a>(scene: &'a PhysicalScene, p: Vec3) -> Option<&'a RigidBodyDescription> {
    scene.rigid_bodies.iter().find(|b| contains(b, p))
}

/// Exact four-point centre-plane counterpart of production's eight-point
/// moving-rigid voxel quadrature and two-point face quadrature.
pub fn rasterize_slice_rigid_geometry(
    scene: &PhysicalScene,
    dimensions: [u32; 2],
    source_cell_size: [f64; 3],
) -> SliceRigidRaster {
    let (nx, ny) = (dimensions[0] as usize, dimensions[1] as usize);
    let mut result = SliceRigidRaster {
        capacity_multiplier: vec![1.0; nx * ny],
        aperture_x_multiplier: vec![1.0; (nx + 1) * ny],
        aperture_y_multiplier: vec![1.0; nx * (ny + 1)],
        solid_velocity_x_fine: vec![0.0; (nx + 1) * ny],
        solid_velocity_y_fine: vec![0.0; nx * (ny + 1)],
    };
    for sy in 0..ny {
        let cy = ny - 1 - sy;
        for x in 0..nx {
            let c = Vec3 {
                x: -0.5 * scene.container.width_m + (x as f64 + 0.5) * source_cell_size[0],
                y: (sy as f64 + 0.5) * source_cell_size[1],
                z: 0.0,
            };
            let mut maximum = 0.0_f64;
            for body in &scene.rigid_bodies {
                let mut inside = 0;
                for corner in 0..4 {
                    let p = Vec3 {
                        x: c.x + if corner & 1 != 0 { 0.4 } else { -0.4 } * source_cell_size[0],
                        y: c.y + if corner & 2 != 0 { 0.4 } else { -0.4 } * source_cell_size[1],
                        z: 0.0,
                    };
                    if contains(body, p) {
                        inside += 1;
                    }
                }
                maximum = maximum.max(inside as f64 / 4.0);
            }
            result.capacity_multiplier[cy * nx + x] = (1.0 - maximum) as f32;
        }
    }
    for cy in 0..ny {
        let sy = ny - 1 - cy;
        for x in 0..=nx {
            let at = cy * (nx + 1) + x;
            let face = Vec3 {
                x: -0.5 * scene.container.width_m + x as f64 * source_cell_size[0],
                y: (sy as f64 + 0.5) * source_cell_size[1],
                z: 0.0,
            };
            let (mut covered, mut wall) = (0.0, 0.0);
            for sign in [-0.35, 0.35] {
                let p = Vec3 {
                    y: face.y + sign * source_cell_size[1],
                    ..face
                };
                if let Some(b) = owner(scene, p) {
                    covered += 0.5;
                    wall += 0.5 * velocity(b, p).x;
                }
            }
            result.aperture_x_multiplier[at] = (1.0 - covered) as f32;
            result.solid_velocity_x_fine[at] = if covered > 0.0 {
                (wall / covered / source_cell_size[0]) as f32
            } else {
                let mut fallback = None;
                for axis_sign in [-0.4, 0.4] {
                    for tangent_sign in [-0.4, 0.4] {
                        fallback = fallback.or_else(|| {
                            owner(
                                scene,
                                Vec3 {
                                    x: face.x + axis_sign * source_cell_size[0],
                                    y: face.y + tangent_sign * source_cell_size[1],
                                    z: 0.0,
                                },
                            )
                        });
                    }
                }
                fallback.map_or(0.0, |body| {
                    (velocity(body, face).x / source_cell_size[0]) as f32
                })
            };
        }
    }
    for syf in 0..=ny {
        let cyf = ny - syf;
        for x in 0..nx {
            let at = cyf * nx + x;
            let face = Vec3 {
                x: -0.5 * scene.container.width_m + (x as f64 + 0.5) * source_cell_size[0],
                y: syf as f64 * source_cell_size[1],
                z: 0.0,
            };
            let (mut covered, mut wall) = (0.0, 0.0);
            for sign in [-0.35, 0.35] {
                let p = Vec3 {
                    x: face.x + sign * source_cell_size[0],
                    ..face
                };
                if let Some(b) = owner(scene, p) {
                    covered += 0.5;
                    wall += 0.5 * -velocity(b, p).y;
                }
            }
            result.aperture_y_multiplier[at] = (1.0 - covered) as f32;
            result.solid_velocity_y_fine[at] = if covered > 0.0 {
                (wall / covered / source_cell_size[1]) as f32
            } else {
                let mut fallback = None;
                for axis_sign in [-0.4, 0.4] {
                    for tangent_sign in [-0.4, 0.4] {
                        fallback = fallback.or_else(|| {
                            owner(
                                scene,
                                Vec3 {
                                    x: face.x + tangent_sign * source_cell_size[0],
                                    y: face.y + axis_sign * source_cell_size[1],
                                    z: 0.0,
                                },
                            )
                        });
                    }
                }
                fallback.map_or(0.0, |body| {
                    (-velocity(body, face).y / source_cell_size[1]) as f32
                })
            };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_authored_shape_membership_arms_are_present() {
        let d = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 0.1,
        };
        let center = Vec3::default();
        for shape in [
            RigidShape::Sphere,
            RigidShape::Box,
            RigidShape::Capsule,
            RigidShape::Cylinder,
        ] {
            assert!(rigid_shape_contains(shape, d, center));
        }
        assert!(rigid_shape_contains(
            RigidShape::Cup,
            d,
            Vec3 {
                x: 0.95,
                y: 0.0,
                z: 0.0
            }
        ));
    }
}
