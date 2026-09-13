//! True 3-D planar swept-prism flux on compiled subfaces.
use crate::geometry3d::plane_box_fraction;
use crate::types::{Fields, Graph, ValidationError};
#[inline]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}

/// Geometry-only image of the partial-donor branch in production `gvHighFlux`.
/// The caller is responsible for the empty/full-capacity endpoint branches.
#[allow(clippy::too_many_arguments)]
pub fn geometric_prism_flux_3d(
    normal: [f32; 3],
    offset: f32,
    cell_center: [f32; 3],
    cell_widths: [f32; 3],
    other_box: Option<([f32; 3], [f32; 3])>,
    axis: usize,
    boundary: f32,
    area: f32,
    aperture: f32,
    sweep: f32,
    low: f32,
) -> Result<f32, ValidationError> {
    if sweep == 0.0 {
        return Ok(0.0);
    }
    if normal == [0.0; 3] || aperture <= 1e-8 {
        return Ok(low);
    }
    if axis >= 3 || area <= 0.0 {
        return Err(ValidationError("invalid 3-D flux face geometry".into()));
    }
    let travel = sweep.abs() / (area * aperture);
    if travel > cell_widths[axis] {
        return Err(ValidationError("3-D face sweep exceeds donor width".into()));
    }
    let mut minimum = [
        -0.5 * cell_widths[0],
        -0.5 * cell_widths[1],
        -0.5 * cell_widths[2],
    ];
    let mut maximum = [
        0.5 * cell_widths[0],
        0.5 * cell_widths[1],
        0.5 * cell_widths[2],
    ];
    if let Some((other_minimum, other_maximum)) = other_box {
        for a in 0..3 {
            minimum[a] = minimum[a].max(other_minimum[a] - cell_center[a]);
            maximum[a] = maximum[a].min(other_maximum[a] - cell_center[a]);
        }
    }
    let local_boundary = boundary - cell_center[axis];
    if sweep > 0.0 {
        minimum[axis] = local_boundary - travel;
        maximum[axis] = local_boundary;
    } else {
        minimum[axis] = local_boundary;
        maximum[axis] = local_boundary + travel;
    }
    let widths = [
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2],
    ];
    if widths.iter().any(|&width| width < 0.0) {
        return Err(ValidationError("3-D face has negative prism width".into()));
    }
    let centre = [
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.5 * (minimum[2] + maximum[2]),
    ];
    let projected = normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2];
    Ok(sweep * plane_box_fraction(normal, offset - projected, widths))
}

pub(crate) fn geometric_high_flux_3d(
    graph: &Graph,
    fields: &Fields,
    face: &crate::types::Subface,
    sweep: f32,
    low: f32,
) -> Result<f32, ValidationError> {
    if graph.dimension != 3 || fields.interface_normal.len() != 3 * graph.cells.len() {
        return Err(ValidationError(
            "3-D flux fields do not align with topology".into(),
        ));
    }
    if sweep == 0.0 {
        return Ok(0.0);
    }
    let donor = if sweep > 0.0 {
        face.negative_cell
    } else {
        face.positive_cell
    };
    if donor < 0 {
        return Ok(0.0);
    }
    let id = donor as usize;
    let cell = &graph.cells[id];
    let capacity = mul(fields.capacity[id], cell.measure);
    let observed = mul(fields.density[id], cell.measure).clamp(0.0, capacity.max(0.0));
    if capacity <= 0.0 || observed == 0.0 {
        return Ok(0.0);
    }
    if observed == capacity {
        return Ok(sweep);
    }
    let normal = [
        fields.interface_normal[3 * id],
        fields.interface_normal[3 * id + 1],
        fields.interface_normal[3 * id + 2],
    ];
    if normal == [0.0; 3] || face.aperture <= 1e-8 {
        return Ok(low);
    }
    let other_box = if face.negative_cell >= 0 && face.positive_cell >= 0 {
        let other_id = if donor == face.negative_cell {
            face.positive_cell
        } else {
            face.negative_cell
        } as usize;
        let other = &graph.cells[other_id];
        Some((other.minimum, other.maximum))
    } else {
        None
    };
    geometric_prism_flux_3d(
        normal,
        fields.interface_offset[id],
        cell.center,
        cell.widths,
        other_box,
        face.axis as usize,
        face.center[face.axis as usize],
        face.measure,
        face.aperture,
        sweep,
        low,
    )
}

pub fn geometric_high_fluxes_3d(
    graph: &Graph,
    fields: &Fields,
    sweeps: &[f32],
    low_flux: &[f32],
) -> Result<Vec<f32>, ValidationError> {
    if graph.dimension != 3
        || sweeps.len() != graph.subfaces.len()
        || low_flux.len() != graph.subfaces.len()
        || fields.interface_normal.len() != 3 * graph.cells.len()
    {
        return Err(ValidationError(
            "3-D flux fields do not align with topology".into(),
        ));
    }
    let mut result = vec![0.0; graph.subfaces.len()];
    for face in &graph.subfaces {
        let i = face.id as usize;
        result[i] = geometric_high_flux_3d(graph, fields, face, sweeps[i], low_flux[i])?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    #[test]
    fn half_filled_axis_plane_sweeps_exact_half() {
        let graph = Graph {
            schema_version: 1,
            dimension: 3,
            dimensions: [1.0; 3],
            cells: vec![Cell {
                id: 0,
                minimum: [0.0; 3],
                maximum: [1.0; 3],
                center: [0.5; 3],
                widths: [1.0; 3],
                measure: 1.0,
                ..Default::default()
            }],
            rows: vec![Row {
                id: 0,
                axis: 0,
                center: [1.0, 0.5, 0.5],
                measure: 1.0,
                distance: 1.0,
                dual_weight: 1.0,
                terms: vec![RowTerm {
                    cell_id: 0,
                    coefficient: 1.0,
                }],
                ..Default::default()
            }],
            subfaces: vec![Subface {
                id: 0,
                row_id: 0,
                axis: 0,
                center: [1.0, 0.5, 0.5],
                measure: 1.0,
                negative_cell: 0,
                positive_cell: -1,
                aperture: 1.0,
                ..Default::default()
            }],
            incidences: vec![vec![0]],
            ..Default::default()
        };
        let fields = Fields {
            density: vec![0.5],
            capacity: vec![1.0],
            interface_normal: vec![0.0, 1.0, 0.0],
            interface_offset: vec![0.0],
            ..Default::default()
        };
        assert_eq!(
            geometric_high_fluxes_3d(&graph, &fields, &[0.2], &[0.1]).unwrap(),
            vec![0.1]
        )
    }
}
