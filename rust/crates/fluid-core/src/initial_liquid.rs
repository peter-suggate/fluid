//! Exact Rust counterpart of `lib/core/initial-fluid.ts` and
//! `lib/core/initial-height-field.ts` for generation-zero liquid occupancy.

use crate::initial_scene::{InitialLiquidHeightField, InitialLiquidVolume, SceneDocument};
use crate::scene_model::Vec3;
use std::collections::BTreeSet;

pub const INITIAL_FLUID_BRICK_SIZE: i32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamBreakFractions {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamBreakBox {
    pub min: Vec3,
    pub max: Vec3,
}

pub fn dam_break_fractions(fill_fraction: f64) -> DamBreakFractions {
    let fill = fill_fraction.clamp(0.0, 1.0);
    if fill == 0.0 {
        return DamBreakFractions {
            width: 0.0,
            height: 0.0,
            depth: 0.0,
        };
    }
    let height = 0.92_f64.max(fill);
    let footprint = (fill / height).sqrt();
    DamBreakFractions {
        width: footprint,
        height,
        depth: footprint,
    }
}

pub fn scene_dam_break_fractions(scene: &SceneDocument) -> DamBreakFractions {
    match scene.fluid.initial_dam_break_dimensions_m {
        Some(d) => DamBreakFractions {
            width: d.x / scene.container.width_m,
            height: d.y / scene.container.height_m,
            depth: d.z / scene.container.depth_m,
        },
        None => dam_break_fractions(scene.container.fill_fraction),
    }
}

pub fn scene_dam_break_box(scene: &SceneDocument) -> DamBreakBox {
    let size = scene_dam_break_fractions(scene);
    let min = scene
        .fluid
        .initial_dam_break_origin_m
        .map(|p| Vec3 {
            x: p.x / scene.container.width_m,
            y: p.y / scene.container.height_m,
            z: p.z / scene.container.depth_m,
        })
        .unwrap_or_default();
    DamBreakBox {
        min,
        max: Vec3 {
            x: min.x + size.width,
            y: min.y + size.height,
            z: min.z + size.depth,
        },
    }
}

fn interval_cell_overlap_fraction(cell: i32, count: u32, minimum: f64, maximum: f64) -> f64 {
    let lower = cell as f64 / count as f64;
    let upper = (cell + 1) as f64 / count as f64;
    let overlap = (upper.min(maximum) - lower.max(minimum)).max(0.0) * count as f64;
    if overlap <= 1e-12 {
        0.0
    } else if overlap >= 1.0 - 1e-12 {
        1.0
    } else {
        overlap
    }
}

pub fn base_initial_liquid_fraction_at_cell(
    scene: &SceneDocument,
    q: [i32; 3],
    dims: [u32; 3],
) -> f64 {
    if !scene.systems.fluid || q.iter().zip(dims).any(|(&v, n)| v < 0 || v >= n as i32) {
        return 0.0;
    }
    if scene.fluid.initial_condition == "tank-fill" {
        return interval_cell_overlap_fraction(q[1], dims[1], 0.0, scene.container.fill_fraction);
    }
    let dam = scene_dam_break_box(scene);
    interval_cell_overlap_fraction(q[0], dims[0], dam.min.x, dam.max.x)
        * interval_cell_overlap_fraction(q[1], dims[1], dam.min.y, dam.max.y)
        * interval_cell_overlap_fraction(q[2], dims[2], dam.min.z, dam.max.z)
}

pub fn initial_height_field_height(field: &InitialLiquidHeightField, x: f64, z: f64) -> f64 {
    match *field {
        InitialLiquidHeightField::Cosine {
            base_height_m,
            amplitude_m,
            wavelength_m,
            origin_x_m,
        } => {
            base_height_m
                + amplitude_m * (2.0 * std::f64::consts::PI * (x - origin_x_m) / wavelength_m).cos()
        }
        InitialLiquidHeightField::Quadratic {
            base_height_m,
            center_m,
            curvature_x_m_inv,
            curvature_z_m_inv,
        } => {
            base_height_m
                + curvature_x_m_inv * (x - center_m.x).powi(2)
                + curvature_z_m_inv * (z - center_m.z).powi(2)
        }
    }
}

pub fn initial_height_field_range(
    field: &InitialLiquidHeightField,
    x0: f64,
    x1: f64,
    z0: f64,
    z1: f64,
) -> [f64; 2] {
    match *field {
        InitialLiquidHeightField::Cosine {
            base_height_m,
            amplitude_m,
            wavelength_m,
            origin_x_m,
        } => {
            let mut values = vec![
                initial_height_field_height(field, x0, z0),
                initial_height_field_height(field, x1, z0),
            ];
            let half_wave = wavelength_m / 2.0;
            let first = ((x0 - origin_x_m) / half_wave).ceil() as i64;
            let last = ((x1 - origin_x_m) / half_wave).floor() as i64;
            if last >= first {
                values.push(base_height_m + amplitude_m * if first % 2 == 0 { 1.0 } else { -1.0 });
            }
            if last > first {
                values.push(base_height_m - amplitude_m * if first % 2 == 0 { 1.0 } else { -1.0 });
            }
            [
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]
        }
        InitialLiquidHeightField::Quadratic { center_m, .. } => {
            let cx = center_m.x.clamp(x0, x1);
            let cz = center_m.z.clamp(z0, z1);
            [
                initial_height_field_height(field, cx, cz),
                [
                    initial_height_field_height(field, x0, z0),
                    initial_height_field_height(field, x0, z1),
                    initial_height_field_height(field, x1, z0),
                    initial_height_field_height(field, x1, z1),
                ]
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max),
            ]
        }
    }
}

pub fn initial_height_field_fraction_at_cell(
    scene: &SceneDocument,
    q: [i32; 3],
    dims: [u32; 3],
) -> Option<f64> {
    let field = scene.fluid.initial_height_field.as_ref()?;
    if !scene.systems.fluid || q.iter().zip(dims).any(|(&v, n)| v < 0 || v >= n as i32) {
        return Some(0.0);
    }
    let c = &scene.container;
    let (hx, hy, hz) = (
        c.width_m / dims[0] as f64,
        c.height_m / dims[1] as f64,
        c.depth_m / dims[2] as f64,
    );
    let (x0, y0, z0) = (
        -c.width_m / 2.0 + q[0] as f64 * hx,
        q[1] as f64 * hy,
        -c.depth_m / 2.0 + q[2] as f64 * hz,
    );
    let [low, high] = initial_height_field_range(field, x0, x0 + hx, z0, z0 + hz);
    if y0 + hy <= low {
        return Some(1.0);
    }
    if y0 >= high {
        return Some(0.0);
    }
    let mut fraction = 0.0;
    for iz in 0..8 {
        for ix in 0..8 {
            let height = initial_height_field_height(
                field,
                x0 + (ix as f64 + 0.5) * hx / 8.0,
                z0 + (iz as f64 + 0.5) * hz / 8.0,
            );
            fraction += ((height - y0) / hy).clamp(0.0, 1.0) / 64.0;
        }
    }
    Some(fraction)
}

fn seed_cell(scene: &SceneDocument, seed: Vec3, dims: [u32; 3]) -> [i32; 3] {
    let c = &scene.container;
    [
        ((seed.x / c.width_m + 0.5) * dims[0] as f64)
            .floor()
            .clamp(0.0, (dims[0] - 1) as f64) as i32,
        (seed.y / c.height_m * dims[1] as f64)
            .floor()
            .clamp(0.0, (dims[1] - 1) as f64) as i32,
        ((seed.z / c.depth_m + 0.5) * dims[2] as f64)
            .floor()
            .clamp(0.0, (dims[2] - 1) as f64) as i32,
    ]
}

pub fn initial_fluid_brick_contains_cell(
    scene: &SceneDocument,
    q: [i32; 3],
    dims: [u32; 3],
    brick_size: i32,
) -> Option<bool> {
    let seeds = scene.fluid.initial_brick_seeds_m.as_ref()?;
    let brick = [
        q[0].div_euclid(brick_size),
        q[1].div_euclid(brick_size),
        q[2].div_euclid(brick_size),
    ];
    Some(seeds.iter().any(|&seed| {
        let cell = seed_cell(scene, seed, dims);
        (0..3).all(|axis| cell[axis].div_euclid(brick_size) == brick[axis])
    }))
}

fn normalized_normal(normal: Vec3) -> Vec3 {
    let length = normal.x.hypot(normal.y).hypot(normal.z);
    if length > 1e-12 {
        Vec3 {
            x: normal.x / length,
            y: normal.y / length,
            z: normal.z / length,
        }
    } else {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

pub fn initial_liquid_volume_contains_point(volume: &InitialLiquidVolume, point: Vec3) -> bool {
    match *volume {
        InitialLiquidVolume::Box { min_m, max_m } => {
            point.x >= min_m.x
                && point.x <= max_m.x
                && point.y >= min_m.y
                && point.y <= max_m.y
                && point.z >= min_m.z
                && point.z <= max_m.z
        }
        InitialLiquidVolume::Torus {
            center_m,
            radius_m,
            tube_radius_m,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            (d.x.hypot(d.z) - radius_m).hypot(d.y) <= tube_radius_m
        }
        InitialLiquidVolume::Cylinder {
            center_m,
            radius_m,
            half_height_m,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            d.x.hypot(d.y) <= radius_m && d.z.abs() <= half_height_m
        }
        InitialLiquidVolume::Sphere { center_m, radius_m } => {
            (point.x - center_m.x)
                .hypot(point.y - center_m.y)
                .hypot(point.z - center_m.z)
                <= radius_m
        }
        InitialLiquidVolume::Hemisphere {
            center_m,
            radius_m,
            outward_normal,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            let n = normalized_normal(outward_normal);
            d.x.hypot(d.y).hypot(d.z) <= radius_m && d.x * n.x + d.y * n.y + d.z * n.z <= 0.0
        }
    }
}

fn analytic_box_signed_distance(point: Vec3, min: Vec3, max: Vec3) -> f64 {
    let center = Vec3 {
        x: 0.5 * (min.x + max.x),
        y: 0.5 * (min.y + max.y),
        z: 0.5 * (min.z + max.z),
    };
    let half = Vec3 {
        x: 0.5 * (max.x - min.x),
        y: 0.5 * (max.y - min.y),
        z: 0.5 * (max.z - min.z),
    };
    let q = Vec3 {
        x: (point.x - center.x).abs() - half.x,
        y: (point.y - center.y).abs() - half.y,
        z: (point.z - center.z).abs() - half.z,
    };
    q.x.max(0.0).hypot(q.y.max(0.0)).hypot(q.z.max(0.0)) + q.x.max(q.y).max(q.z).min(0.0)
}

pub fn initial_liquid_volume_signed_distance(volume: &InitialLiquidVolume, point: Vec3) -> f64 {
    match *volume {
        InitialLiquidVolume::Box { min_m, max_m } => {
            analytic_box_signed_distance(point, min_m, max_m)
        }
        InitialLiquidVolume::Cylinder {
            center_m,
            radius_m,
            half_height_m,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            let radial = d.x.hypot(d.y) - radius_m;
            let axial = d.z.abs() - half_height_m;
            radial.max(0.0).hypot(axial.max(0.0)) + radial.max(axial).min(0.0)
        }
        InitialLiquidVolume::Torus {
            center_m,
            radius_m,
            tube_radius_m,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            (d.x.hypot(d.z) - radius_m).hypot(d.y) - tube_radius_m
        }
        InitialLiquidVolume::Sphere { center_m, radius_m } => {
            (point.x - center_m.x)
                .hypot(point.y - center_m.y)
                .hypot(point.z - center_m.z)
                - radius_m
        }
        InitialLiquidVolume::Hemisphere {
            center_m,
            radius_m,
            outward_normal,
        } => {
            let d = Vec3 {
                x: point.x - center_m.x,
                y: point.y - center_m.y,
                z: point.z - center_m.z,
            };
            let n = normalized_normal(outward_normal);
            (d.x.hypot(d.y).hypot(d.z) - radius_m).max(d.x * n.x + d.y * n.y + d.z * n.z)
        }
    }
}

pub fn initial_liquid_volumes_signed_distance(scene: &SceneDocument, point: Vec3) -> Option<f64> {
    if scene.fluid.initial_liquid_volumes.is_empty() {
        None
    } else {
        Some(
            scene
                .fluid
                .initial_liquid_volumes
                .iter()
                .map(|v| initial_liquid_volume_signed_distance(v, point))
                .fold(f64::INFINITY, f64::min),
        )
    }
}

/// Initial liquid geometry before adaptive averaging. Container faces are not
/// free surfaces; extend a box through any container face that it touches.
fn liquid_box_distance(scene: &SceneDocument, point: Vec3, min: Vec3, max: Vec3) -> f64 {
    let domain_min = [-0.5 * scene.container.width_m, 0.0, -0.5 * scene.container.depth_m];
    let domain_max = [0.5 * scene.container.width_m, scene.container.height_m, 0.5 * scene.container.depth_m];
    let p = [point.x, point.y, point.z];
    let lo = [min.x, min.y, min.z];
    let hi = [max.x, max.y, max.z];
    let far = scene.container.width_m + scene.container.height_m + scene.container.depth_m;
    let q: [f64; 3] = std::array::from_fn(|axis| {
        let low = if lo[axis] <= domain_min[axis] { -far } else { lo[axis] - p[axis] };
        let high = if hi[axis] >= domain_max[axis] { -far } else { p[axis] - hi[axis] };
        low.max(high)
    });
    q[0].max(0.0).hypot(q[1].max(0.0)).hypot(q[2].max(0.0))
        + q[0].max(q[1]).max(q[2]).min(0.0)
}

fn painted_brick_union_distance(
    scene: &SceneDocument,
    point: Vec3,
    seeds: &[Vec3],
    dims: [u32; 3],
) -> f64 {
    let c = &scene.container;
    let domain_min = [-0.5 * c.width_m, 0.0, -0.5 * c.depth_m];
    let domain_max = [0.5 * c.width_m, c.height_m, 0.5 * c.depth_m];
    let h = [c.width_m / dims[0] as f64, c.height_m / dims[1] as f64,
        c.depth_m / dims[2] as f64];
    let unique: BTreeSet<_> = seeds.iter().map(|&seed| {
        seed_cell(scene, seed, dims)
            .map(|value| value.div_euclid(INITIAL_FLUID_BRICK_SIZE) * INITIAL_FLUID_BRICK_SIZE)
    }).collect();
    let boxes: Vec<_> = unique.into_iter().map(|lo| {
        let minimum = [-0.5 * c.width_m + lo[0] as f64 * h[0], lo[1] as f64 * h[1],
            -0.5 * c.depth_m + lo[2] as f64 * h[2]];
        let maximum = [-0.5 * c.width_m + (lo[0] + 8) as f64 * h[0],
            (lo[1] + 8) as f64 * h[1], -0.5 * c.depth_m + (lo[2] + 8) as f64 * h[2]];
        (minimum, maximum)
    }).collect();
    let p = [point.x, point.y, point.z];
    let contains = |minimum: [f64; 3], maximum: [f64; 3]| {
        (0..3).all(|axis| p[axis] >= minimum[axis] && p[axis] <= maximum[axis])
    };
    if !boxes.iter().any(|&(minimum, maximum)| contains(minimum, maximum)) {
        return boxes.iter().map(|&(minimum, maximum)| liquid_box_distance(
            scene,
            point,
            Vec3 { x: minimum[0], y: minimum[1], z: minimum[2] },
            Vec3 { x: maximum[0], y: maximum[1], z: maximum[2] },
        )).fold(f64::INFINITY, f64::min);
    }

    // Inside the union, min(box SDF) is zero on a shared brick face.  Merge
    // the intervals cut by each coordinate ray instead, then measure to the
    // connected union's exterior.  Container faces extend beyond the domain
    // because they are walls, not liquid surfaces.
    let far = c.width_m + c.height_m + c.depth_m;
    let mut distance = f64::INFINITY;
    for axis in 0..3 {
        let mut intervals: Vec<_> = boxes.iter().filter_map(|&(minimum, maximum)| {
            (0..3).filter(|&other| other != axis)
                .all(|other| p[other] >= minimum[other] && p[other] <= maximum[other])
                .then(|| {
                    let lo = if minimum[axis] <= domain_min[axis] { -far } else { minimum[axis] };
                    let hi = if maximum[axis] >= domain_max[axis] { far } else { maximum[axis] };
                    [lo, hi]
                })
        }).collect();
        intervals.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        let mut merged: Vec<[f64; 2]> = Vec::new();
        for interval in intervals {
            if let Some(last) = merged.last_mut().filter(|last| interval[0] <= last[1]) {
                last[1] = last[1].max(interval[1]);
            } else {
                merged.push(interval);
            }
        }
        if let Some(&[lo, hi]) = merged.iter()
            .find(|interval| p[axis] >= interval[0] && p[axis] <= interval[1]) {
            distance = distance.min((p[axis] - lo).min(hi - p[axis]));
        }
    }
    -distance
}

/// A continuous implicit surface in metres for the authored liquid union.
/// Its zero set is independent of the sparse atlas. Height fields and unions
/// need not be exact distances; geometric redistancing is derived from their
/// published zero contour, rather than mixed into the transported scalar.
pub fn initial_liquid_surface_scalar(scene: &SceneDocument, point: Vec3, dims: [u32; 3]) -> f64 {
    let c = &scene.container;
    let empty = c.width_m + c.height_m + c.depth_m;
    if !scene.systems.fluid { return empty; }
    let mut scalar = if let Some(field) = &scene.fluid.initial_height_field {
        point.y - initial_height_field_height(field, point.x, point.z)
    } else if scene.fluid.initial_condition == "tank-fill" {
        if c.fill_fraction > 0.0 { point.y - c.height_m * c.fill_fraction.clamp(0.0, 1.0) }
        else { empty }
    } else {
        let bounds = scene_dam_break_box(scene);
        if bounds.max.x <= bounds.min.x || bounds.max.y <= bounds.min.y || bounds.max.z <= bounds.min.z {
            empty
        } else {
            liquid_box_distance(scene, point,
                Vec3 { x: (bounds.min.x - 0.5) * c.width_m, y: bounds.min.y * c.height_m, z: (bounds.min.z - 0.5) * c.depth_m },
                Vec3 { x: (bounds.max.x - 0.5) * c.width_m, y: bounds.max.y * c.height_m, z: (bounds.max.z - 0.5) * c.depth_m })
        }
    };
    if let Some(seeds) = &scene.fluid.initial_brick_seeds_m {
        if !scene.fluid.initial_brick_seeds_additive { scalar = empty; }
        scalar = scalar.min(painted_brick_union_distance(scene, point, seeds, dims));
    }
    for volume in &scene.fluid.initial_liquid_volumes {
        let distance = match *volume {
            InitialLiquidVolume::Box { min_m, max_m } => liquid_box_distance(scene, point, min_m, max_m),
            _ => initial_liquid_volume_signed_distance(volume, point),
        };
        scalar = scalar.min(distance);
    }
    scalar
}

pub fn initial_liquid_volume_contains_cell(
    scene: &SceneDocument,
    q: [i32; 3],
    dims: [u32; 3],
) -> bool {
    if scene.fluid.initial_liquid_volumes.is_empty() {
        return false;
    }
    let c = &scene.container;
    let point = Vec3 {
        x: -0.5 * c.width_m + (q[0] as f64 + 0.5) * c.width_m / dims[0] as f64,
        y: (q[1] as f64 + 0.5) * c.height_m / dims[1] as f64,
        z: -0.5 * c.depth_m + (q[2] as f64 + 0.5) * c.depth_m / dims[2] as f64,
    };
    scene
        .fluid
        .initial_liquid_volumes
        .iter()
        .any(|v| initial_liquid_volume_contains_point(v, point))
}

pub fn initial_liquid_fraction_at_cell(
    scene: &SceneDocument,
    q: [i32; 3],
    dims: [u32; 3],
    base_fraction: f64,
) -> f64 {
    let brick = initial_fluid_brick_contains_cell(scene, q, dims, INITIAL_FLUID_BRICK_SIZE);
    let base = initial_height_field_fraction_at_cell(scene, q, dims)
        .unwrap_or(base_fraction.clamp(0.0, 1.0));
    let resolved = match brick {
        None => base,
        Some(wet) if scene.fluid.initial_brick_seeds_additive => {
            if wet {
                1.0
            } else {
                base
            }
        }
        Some(wet) => {
            if wet {
                1.0
            } else {
                0.0
            }
        }
    };
    if resolved >= 1.0 || scene.fluid.initial_liquid_volumes.is_empty() {
        return resolved;
    }
    let c = &scene.container;
    let h = [
        c.width_m / dims[0] as f64,
        c.height_m / dims[1] as f64,
        c.depth_m / dims[2] as f64,
    ];
    let center = Vec3 {
        x: -0.5 * c.width_m + (q[0] as f64 + 0.5) * h[0],
        y: (q[1] as f64 + 0.5) * h[1],
        z: -0.5 * c.depth_m + (q[2] as f64 + 0.5) * h[2],
    };
    let mut wet = 0;
    for corner in 0..8 {
        let point = Vec3 {
            x: center.x + if corner & 1 != 0 { 0.4 } else { -0.4 } * h[0],
            y: center.y + if corner & 2 != 0 { 0.4 } else { -0.4 } * h[1],
            z: center.z + if corner & 4 != 0 { 0.4 } else { -0.4 } * h[2],
        };
        if scene
            .fluid
            .initial_liquid_volumes
            .iter()
            .any(|v| initial_liquid_volume_contains_point(v, point))
        {
            wet += 1;
        }
    }
    resolved.max(wet as f64 / 8.0)
}

/// Centre-Z slice in canvas row order. Source +Y is reflected exactly once.
pub fn rasterize_center_slice(scene: &SceneDocument, dims: [u32; 3]) -> Vec<f32> {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let z = (nz / 2) as i32;
    let mut density = vec![0.0; (nx * ny) as usize];
    for source_y in 0..ny {
        for x in 0..nx {
            let q = [x as i32, source_y as i32, z];
            let base = base_initial_liquid_fraction_at_cell(scene, q, dims);
            let canvas_y = ny - 1 - source_y;
            density[(canvas_y * nx + x) as usize] =
                initial_liquid_fraction_at_cell(scene, q, dims, base) as f32;
        }
    }
    density
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Golden {
        schema_version: u32,
        cases: Vec<GoldenCase>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct GoldenCase {
        scene: SceneDocument,
        dimensions: [u32; 3],
        center_slice: Vec<f32>,
    }

    #[test]
    fn matches_typescript_generation_zero_liquid_goldens_exactly() {
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../core/testdata/initial-liquid-golden.json"
        ))
        .unwrap();
        assert_eq!(golden.schema_version, 1);
        let ids: Vec<_> = golden
            .cases
            .iter()
            .map(|case| case.scene.scene_id.as_str())
            .collect();
        for required in [
            "initial-liquid-all-volume-arms",
            "initial-liquid-replacement-seeds",
            "initial-liquid-additive-seeds",
            "initial-liquid-quadratic-height",
            "initial-liquid-cosine-height",
            "interactive-water-box",
            "coarse-first-pool-impact-quarter",
            "twin-dam-collision",
            "falling-water-torus",
        ] {
            assert!(ids.contains(&required), "missing fixture {required}");
        }
        for case in golden.cases {
            let actual = rasterize_center_slice(&case.scene, case.dimensions);
            assert_eq!(
                actual.len(),
                case.center_slice.len(),
                "{} length",
                case.scene.scene_id
            );
            for (index, (&left, &right)) in actual.iter().zip(&case.center_slice).enumerate() {
                assert_eq!(
                    left.to_bits(),
                    right.to_bits(),
                    "{} cell {index}: Rust {left:?}, TS {right:?}",
                    case.scene.scene_id
                );
            }
        }
    }

    #[test]
    fn out_of_bounds_and_disabled_fluid_match_typescript() {
        let golden: Golden = serde_json::from_str(include_str!(
            "../../../core/testdata/initial-liquid-golden.json"
        ))
        .unwrap();
        let mut scene = golden.cases.into_iter().next().unwrap().scene;
        let dims = [24, 16, 16];
        assert_eq!(
            base_initial_liquid_fraction_at_cell(&scene, [-1, 0, 0], dims),
            0.0
        );
        scene.systems.fluid = false;
        assert_eq!(
            base_initial_liquid_fraction_at_cell(&scene, [0, 0, 0], dims),
            0.0
        );
        assert_eq!(
            initial_height_field_fraction_at_cell(&scene, [0, 0, 0], dims),
            None
        );
    }
}
