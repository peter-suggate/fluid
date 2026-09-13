//! Dimension-aware physical geometry used while a scene is compiled.
//!
//! Coordinates are expressed in finest-cell units.  World/unit conversion is
//! deliberately kept at the scene boundary so topology coefficients retain
//! the same dyadic arithmetic as the source implementation.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BoundaryMode {
    Open,
    Closed,
    Symmetry,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SolidShape {
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    Plane {
        normal: [f32; 3],
        offset: f32,
    },
}

impl SolidShape {
    pub fn contains(&self, point: [f32; 3]) -> bool {
        match *self {
            Self::Box {
                center,
                half_extents,
            } => (0..3).all(|a| (point[a] - center[a]).abs() <= half_extents[a]),
            Self::Sphere { center, radius } => {
                let d = (0..3)
                    .map(|a| {
                        let q = point[a] - center[a];
                        q * q
                    })
                    .sum::<f32>();
                d <= radius * radius
            }
            Self::Plane { normal, offset } => dot(normal, point) <= offset,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Solid {
    pub shape: SolidShape,
    #[serde(default)]
    pub linear_velocity: [f32; 3],
    #[serde(default)]
    pub angular_velocity: [f32; 3],
    #[serde(default)]
    pub center: [f32; 3],
}

impl Solid {
    pub fn velocity_at(&self, point: [f32; 3]) -> [f32; 3] {
        let r = sub(point, self.center);
        add(self.linear_velocity, cross(self.angular_velocity, r))
    }
}

#[inline]
pub fn overlap_1d(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

pub fn box_measure<const D: usize>(minimum: [f32; 3], maximum: [f32; 3]) -> f32 {
    (0..D).fold(1.0_f32, |v, a| v * (maximum[a] - minimum[a]).max(0.0))
}

pub fn box_overlap_measure<const D: usize>(
    a_min: [f32; 3],
    a_max: [f32; 3],
    b_min: [f32; 3],
    b_max: [f32; 3],
) -> f32 {
    (0..D).fold(1.0_f32, |v, axis| {
        v * overlap_1d(a_min[axis], a_max[axis], b_min[axis], b_max[axis])
    })
}

/// Production-compatible regular subsampling of the open part of a cell.
/// Four samples per active axis exactly mirrors the 2-D rigid cut sampling.
pub fn open_fraction<const D: usize>(
    minimum: [f32; 3],
    maximum: [f32; 3],
    solids: &[Solid],
    samples: usize,
) -> f32 {
    if solids.is_empty() {
        return 1.0;
    }
    let samples = samples.max(1);
    let count = samples.pow(D as u32);
    let mut open = 0usize;
    for linear in 0..count {
        let mut q = linear;
        let mut p = [0.0; 3];
        for axis in 0..D {
            let lane = q % samples;
            q /= samples;
            let t = (lane as f32 + 0.5) / samples as f32;
            p[axis] = minimum[axis] + t * (maximum[axis] - minimum[axis]);
        }
        if D == 2 {
            p[2] = 0.5 * (minimum[2] + maximum[2]);
        }
        if !solids.iter().any(|solid| solid.shape.contains(p)) {
            open += 1;
        }
    }
    open as f32 / count as f32
}

/// Samples the open measure and solid normal velocity of a physical face.
pub fn face_geometry<const D: usize>(
    axis: usize,
    center: [f32; 3],
    widths: [f32; 3],
    solids: &[Solid],
    samples: usize,
) -> (f32, f32) {
    if solids.is_empty() {
        return (1.0, 0.0);
    }
    let tangents: Vec<usize> = (0..D).filter(|&a| a != axis).collect();
    let samples = samples.max(1);
    let count = samples.pow(tangents.len() as u32);
    let mut open = 0usize;
    let mut velocity = 0.0_f32;
    let mut covered = 0usize;
    for linear in 0..count {
        let mut q = linear;
        let mut point = center;
        for &tangent in &tangents {
            let lane = q % samples;
            q /= samples;
            point[tangent] += ((lane as f32 + 0.5) / samples as f32 - 0.5) * widths[tangent];
        }
        if let Some(solid) = solids.iter().find(|solid| solid.shape.contains(point)) {
            velocity += solid.velocity_at(point)[axis];
            covered += 1;
        } else {
            open += 1;
        }
    }
    (
        open as f32 / count as f32,
        if covered == 0 {
            0.0
        } else {
            velocity / covered as f32
        },
    )
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_half_cell_box() {
        let solid = Solid {
            shape: SolidShape::Box {
                center: [0.25, 0.5, 0.0],
                half_extents: [0.25, 1.0, 1.0],
            },
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            center: [0.0; 3],
        };
        assert_eq!(
            open_fraction::<2>([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], &[solid], 4),
            0.5
        );
    }
}
