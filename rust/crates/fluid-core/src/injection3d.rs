//! Resident editor drop arithmetic (`injectionCoverageAt` and `injectLiquid`).
//! The caller must first commit topology covering the conservative drop bounds.
use crate::dynamic3d::VolumeFrame;
use crate::scene_model::Vec3;
use crate::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LiquidDrop3d {
    pub centre_m: Vec3,
    pub radius_m: f64,
    #[serde(rename = "halfHeight_m", default)]
    pub half_height_m: Option<f64>,
}
#[derive(Clone, Copy, Debug)]
pub struct FineDrop3d {
    pub centre: [f32; 3],
    pub radius: f32,
    pub half_depth: Option<f32>,
}
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InjectionReceipt3d {
    pub accepted: bool,
    pub cells_wetted: usize,
    pub volume_admitted_fine: f64,
    pub accepted_generation: u32,
    pub candidate_generation: u32,
}
impl LiquidDrop3d {
    pub fn in_frame(self, frame: VolumeFrame) -> Result<FineDrop3d, ValidationError> {
        if !self.centre_m.array().into_iter().all(f64::is_finite)
            || !self.radius_m.is_finite()
            || self.radius_m <= 0.0
            || self
                .half_height_m
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(ValidationError("invalid liquid drop".into()));
        }
        let result = FineDrop3d {
            centre: std::array::from_fn(|a| {
                ((self.centre_m.array()[a] - frame.origin_m[a]) / frame.cell_size_m) as f32
            }),
            radius: (self.radius_m / frame.cell_size_m) as f32,
            half_depth: self.half_height_m.map(|v| (v / frame.cell_size_m) as f32),
        };
        if !result.centre.into_iter().all(f32::is_finite)
            || !result.radius.is_finite()
            || result.half_depth.is_some_and(|v| !v.is_finite())
        {
            return Err(ValidationError(
                "liquid drop exceeds f32 coordinate range".into(),
            ));
        }
        Ok(result)
    }
}
impl FineDrop3d {
    pub fn bounds(self) -> ([f64; 3], [f64; 3]) {
        let radius = [
            self.radius,
            self.radius,
            self.half_depth.unwrap_or(self.radius),
        ];
        (
            std::array::from_fn(|a| (self.centre[a] - radius[a]) as f64),
            std::array::from_fn(|a| (self.centre[a] + radius[a]) as f64),
        )
    }
    pub fn coverage(self, point: [f32; 3], width: f32) -> f32 {
        let relative: [f32; 3] = std::array::from_fn(|a| point[a] - self.centre[a]);
        let signed = if let Some(half) = self.half_depth {
            let radial =
                (relative[0] * relative[0] + relative[1] * relative[1]).sqrt() - self.radius;
            let axial = relative[2].abs() - half;
            (radial.max(0.0) * radial.max(0.0) + axial.max(0.0) * axial.max(0.0)).sqrt()
                + radial.max(axial).min(0.0)
        } else {
            let q = relative.map(|v| v / self.radius.max(1e-6));
            (((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2]).sqrt() - 1.0) * self.radius
        };
        (0.5 - signed / width.max(1e-6)).clamp(0.0, 1.0)
    }
}
pub fn apply_dose_3d(
    graph: &Graph,
    fields: &mut Fields,
    drop: FineDrop3d,
) -> Result<InjectionReceipt3d, ValidationError> {
    fields.validate_for(graph)?;
    if graph.dimension != 3 {
        return Err(ValidationError("3D drop requires dimension3".into()));
    }
    let mut receipt = InjectionReceipt3d {
        accepted: true,
        accepted_generation: graph.topology_generation,
        candidate_generation: graph.topology_generation,
        ..Default::default()
    };
    for c in &graph.cells {
        let i = c.id as usize;
        if fields.capacity[i] * c.measure <= 1e-8 {
            continue;
        }
        let coverage = drop.coverage(c.center, c.widths[0].min(c.widths[1].min(c.widths[2])));
        let previous = fields.density[i];
        let next = previous.max(coverage * fields.capacity[i]);
        fields.density[i] = next;
        if coverage > 0.0 {
            fields.gamma[i] = 1.0;
        }
        if next > previous {
            receipt.cells_wetted += 1;
            receipt.volume_admitted_fine += (next as f64 - previous as f64) * c.measure as f64;
        }
    }
    Ok(receipt)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sphere_and_xy_disk_keep_source_frame_axes() {
        let sphere = FineDrop3d {
            centre: [4.0; 3],
            radius: 2.0,
            half_depth: None,
        };
        assert_eq!(sphere.coverage([4.0; 3], 1.0), 1.0);
        assert_eq!(sphere.coverage([6.0, 4.0, 4.0], 1.0), 0.5);
        let disk = FineDrop3d {
            half_depth: Some(0.5),
            ..sphere
        };
        assert_eq!(disk.coverage([4.0, 4.0, 4.5], 1.0), 0.5);
        assert_eq!(disk.coverage([4.0, 4.0, 5.0], 1.0), 0.0);
        assert_eq!(disk.coverage([5.0, 4.0, 4.0], 1.0), 1.0);
    }
}
