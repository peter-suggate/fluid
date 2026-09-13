//! Reader intervention arithmetic. Topology must accept the demanded support
//! before `apply_dose` is called; a dropped ball takes max, never adds density.
use crate::topology::BrickSeed;
use crate::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiquidDrop {
    pub centre_fine: [f64; 2],
    pub radius_fine: f64,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DoseReceipt {
    pub cells_wetted: usize,
    pub area_admitted_fine: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InjectionReceipt {
    pub accepted: bool,
    pub bricks_demanded: usize,
    pub bricks_activated: usize,
    pub bricks_promoted: usize,
    pub cells_wetted: usize,
    pub area_requested_fine: f64,
    pub area_admitted_fine: f64,
    pub accepted_generation: u32,
    pub candidate_generation: u32,
    pub fault: Option<crate::NumericalFault>,
}
pub fn coverage(drop: LiquidDrop, centre: [f64; 2], width: f64) -> f64 {
    let r = drop.radius_fine.max(1e-6);
    let x = (centre[0] - drop.centre_fine[0]) / r;
    let y = (centre[1] - drop.centre_fine[1]) / r;
    let signed = x.hypot(y) - 1.0;
    (0.5 - signed * r / width.max(1e-6)).clamp(0.0, 1.0)
}
pub fn addressable(drop: LiquidDrop, dimensions: [u32; 2]) -> bool {
    let [x, y] = drop.centre_fine;
    let r = drop.radius_fine;
    x.is_finite()
        && y.is_finite()
        && r.is_finite()
        && r > 0.0
        && x + r >= 0.0
        && x - r <= dimensions[0] as f64
        && y + r >= 0.0
        && y - r <= dimensions[1] as f64
}
pub fn from_canvas(dimensions: [u32; 2], canvas: [f64; 2], radius: f64) -> LiquidDrop {
    LiquidDrop {
        centre_fine: [canvas[0], dimensions[1] as f64 - canvas[1]],
        radius_fine: radius,
    }
}
pub fn demanded_bricks(bricks: &[BrickSeed], drop: LiquidDrop) -> BTreeSet<u32> {
    let [x, y] = drop.centre_fine;
    let r = drop.radius_fine;
    bricks
        .iter()
        .filter(|b| {
            let span = 8.0 * b.span_bricks as f64;
            let ox = 8.0 * b.coordinate[0] as f64;
            let oy = 8.0 * b.coordinate[1] as f64;
            x + r >= ox && x - r <= ox + span && y + r >= oy && y - r <= oy + span
        })
        .map(|b| b.key)
        .collect()
}
pub fn requested_area(drop: LiquidDrop, dimensions: [u32; 2]) -> f64 {
    let [x, y] = drop.centre_fine;
    let r = drop.radius_fine;
    let x0 = (x - r).floor().max(0.0) as i32;
    let x1 = (x + r).ceil().min(dimensions[0] as f64) as i32;
    let y0 = (y - r).floor().max(0.0) as i32;
    let y1 = (y + r).ceil().min(dimensions[1] as f64) as i32;
    let mut area = 0.0;
    for iy in y0..y1 {
        for ix in x0..x1 {
            for sy in 0..4 {
                for sx in 0..4 {
                    let px = ix as f64 + (sx as f64 + 0.5) / 4.0 - x;
                    let py = iy as f64 + (sy as f64 + 0.5) / 4.0 - y;
                    if px * px + py * py <= r * r {
                        area += 1.0 / 16.0
                    }
                }
            }
        }
    }
    area
}
pub fn apply_dose(
    graph: &Graph,
    fields: &mut Fields,
    drop: LiquidDrop,
) -> Result<DoseReceipt, ValidationError> {
    fields.validate_for(graph)?;
    if graph.dimension != 2 {
        return Err(ValidationError(
            "disk injection requires dimension 2".into(),
        ));
    }
    let mut receipt = DoseReceipt::default();
    for c in &graph.cells {
        let id = c.id as usize;
        let open = fields.capacity[id];
        if open <= 1e-8 {
            continue;
        }
        let value = coverage(
            drop,
            [c.center[0] as f64, c.center[1] as f64],
            c.widths[0].min(c.widths[1]) as f64,
        );
        if value <= 0.0 {
            continue;
        }
        fields.gamma[id] = 1.0;
        let previous = fields.density[id];
        let next = previous.max((value * open as f64) as f32);
        if next == previous {
            continue;
        }
        fields.density[id] = next;
        receipt.cells_wetted += 1;
        receipt.area_admitted_fine += (next as f64 - previous as f64) * c.measure as f64;
    }
    Ok(receipt)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rim_uses_resident_smoothed_indicator() {
        let drop = LiquidDrop {
            centre_fine: [4.0, 4.0],
            radius_fine: 2.0,
        };
        assert_eq!(coverage(drop, [6.0, 4.0], 1.0), 0.5);
        assert_eq!(coverage(drop, [4.0, 4.0], 1.0), 1.0);
        assert_eq!(coverage(drop, [7.0, 4.0], 1.0), 0.0);
    }
    #[test]
    fn pointer_reflection_and_off_domain_refusal() {
        let drop = from_canvas([16, 24], [3.0, 7.0], 2.0);
        assert_eq!(drop.centre_fine, [3.0, 17.0]);
        assert!(addressable(drop, [16, 24]));
        assert!(!addressable(
            LiquidDrop {
                centre_fine: [-3.0, 0.0],
                radius_fine: 2.0
            },
            [16, 24]
        ));
        assert_eq!(
            requested_area(
                LiquidDrop {
                    centre_fine: [4.0, 4.0],
                    radius_fine: 1.0
                },
                [8, 8]
            ),
            3.25
        );
    }
}
