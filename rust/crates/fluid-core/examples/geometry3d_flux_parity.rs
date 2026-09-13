use std::io::{self, Read};

use fluid_core::geometry3d::{plane_box_fraction, plane_box_offset};
use fluid_core::transport3d::geometric_prism_flux_3d;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Case {
    normal: [f32; 3],
    offset: f32,
    widths: [f32; 3],
    fill: f32,
    cell_center: [f32; 3],
    cell_widths: [f32; 3],
    other_minimum: [f32; 3],
    other_maximum: [f32; 3],
    axis: usize,
    boundary: f32,
    area: f32,
    aperture: f32,
    sweep: f32,
    low: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultRow {
    fraction: f32,
    inverse_offset: f32,
    high_flux: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = String::new();
    io::stdin().read_to_string(&mut json)?;
    let cases: Vec<Case> = serde_json::from_str(&json)?;
    let output = cases
        .into_iter()
        .map(|case| {
            Ok(ResultRow {
                fraction: plane_box_fraction(case.normal, case.offset, case.widths),
                inverse_offset: plane_box_offset(case.normal, case.widths, case.fill),
                high_flux: geometric_prism_flux_3d(
                    case.normal,
                    case.offset,
                    case.cell_center,
                    case.cell_widths,
                    Some((case.other_minimum, case.other_maximum)),
                    case.axis,
                    case.boundary,
                    case.area,
                    case.aperture,
                    case.sweep,
                    case.low,
                )?,
            })
        })
        .collect::<Result<Vec<_>, fluid_core::ValidationError>>()?;
    serde_json::to_writer(io::stdout(), &output)?;
    Ok(())
}
