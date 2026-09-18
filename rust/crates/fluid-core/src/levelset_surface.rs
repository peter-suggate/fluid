//! Direct publication of the level-set experiment's shared vertex field.

use crate::levelset_redistance::{sample_scalar, RedistanceField};
use crate::numerics::owner_at;
use crate::presentation::{RdfReceipt, RdfSurface};
use crate::types::{Fields, Graph, ValidationError};

pub(crate) type Vertex = [f64; 3];

pub(crate) fn triangles(x: f64, y: f64, phi: [f64; 4]) -> [[Vertex; 3]; 4] {
    let corners = [
        [x, y, phi[0]], [x + 1.0, y, phi[1]],
        [x + 1.0, y + 1.0, phi[2]], [x, y + 1.0, phi[3]],
    ];
    let centre = [x + 0.5, y + 0.5, phi.iter().sum::<f64>() * 0.25];
    std::array::from_fn(|i| [corners[i], corners[(i + 1) % 4], centre])
}

fn clipped_area(triangle: &[Vertex; 3]) -> f64 {
    let mut polygon = Vec::with_capacity(4);
    for i in 0..3 {
        let a = triangle[i];
        let b = triangle[(i + 1) % 3];
        if a[2] <= 0.0 { polygon.push([a[0], a[1]]); }
        if (a[2] < 0.0) != (b[2] < 0.0) {
            let t = a[2] / (a[2] - b[2]);
            polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    let mut twice = 0.0;
    for i in 0..polygon.len() {
        let a = polygon[i]; let b = polygon[(i + 1) % polygon.len()];
        twice += a[0] * b[1] - b[0] * a[1];
    }
    twice.abs() * 0.5
}

/// Liquid fraction enclosed by the authoritative vertex scalar in each finest cell.
/// The order is x-fastest and matches the production slice's finest raster.
pub fn implied_fill_fine_cells(surface: &RdfSurface) -> Result<Vec<f32>, ValidationError> {
    let [nx, ny] = surface.dimensions.map(|value| value as usize);
    let stride = nx + 1;
    if nx == 0 || ny == 0
        || surface.vertex_phi_fine.len() != stride.saturating_mul(ny + 1)
        || surface.vertex_phi_fine.iter().any(|value| !value.is_finite())
    {
        return Err(ValidationError(
            "direct level-set surface cannot publish implied cell fill".into(),
        ));
    }
    let mut result = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            let phi = [
                surface.vertex_phi_fine[x + stride * y] as f64,
                surface.vertex_phi_fine[x + 1 + stride * y] as f64,
                surface.vertex_phi_fine[x + 1 + stride * (y + 1)] as f64,
                surface.vertex_phi_fine[x + stride * (y + 1)] as f64,
            ];
            let fill = triangles(x as f64, y as f64, phi)
                .iter().map(clipped_area).sum::<f64>();
            result.push(fill.clamp(0.0, 1.0) as f32);
        }
    }
    Ok(result)
}

/// Publish a surface directly from its authoritative fine-vertex scalar.
/// `diagnostic_volume` is used only for the receipt's comparison fields.
pub fn publish(
    dimensions: [u32; 2],
    vertex_phi_fine: Vec<f32>,
    diagnostic_volume: f64,
) -> Result<RdfSurface, ValidationError> {
    let [nx, ny] = dimensions.map(|value| value as usize);
    if nx == 0 || ny == 0 || !diagnostic_volume.is_finite() || diagnostic_volume < 0.0 {
        return Err(ValidationError("direct level-set dimensions or diagnostic volume are invalid".into()));
    }
    if vertex_phi_fine.len() != (nx + 1).saturating_mul(ny + 1) {
        return Err(ValidationError("direct level-set vertex count does not match dimensions".into()));
    }
    if vertex_phi_fine.iter().any(|value| !value.is_finite()) {
        return Err(ValidationError("direct level-set contains a non-finite vertex".into()));
    }
    let stride = nx + 1;
    let mut receipt = RdfReceipt::default();
    receipt.exact_area_fine = diagnostic_volume;
    let mut segments = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let phi = [
                vertex_phi_fine[x + stride * y] as f64,
                vertex_phi_fine[x + 1 + stride * y] as f64,
                vertex_phi_fine[x + 1 + stride * (y + 1)] as f64,
                vertex_phi_fine[x + stride * (y + 1)] as f64,
            ];
            let mut cell_segments = 0;
            for triangle in triangles(x as f64, y as f64, phi) {
                receipt.represented_area_fine += clipped_area(&triangle);
                let mut points: Vec<[f64; 2]> = Vec::with_capacity(2);
                for i in 0..3 {
                    let a = triangle[i]; let b = triangle[(i + 1) % 3];
                    if (a[2] < 0.0) == (b[2] < 0.0) || a[2] == b[2] { continue; }
                    let t = a[2] / (a[2] - b[2]);
                    let point = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    if !points.iter().any(|other| (other[0] - point[0]).hypot(other[1] - point[1]) < 1e-10) {
                        points.push(point);
                    }
                }
                if points.len() == 2 && (points[0][0] - points[1][0]).hypot(points[0][1] - points[1][1]) > 1e-10 {
                    segments.extend(points.into_iter().flat_map(|point| point.map(|v| v as f32)));
                    cell_segments += 1;
                }
            }
            if cell_segments > 0 { receipt.interface_cells += 1; }
        }
    }
    receipt.signed_area_error_fine = receipt.represented_area_fine - diagnostic_volume;
    Ok(RdfSurface { adaptive_sdf: None, dimensions, vertex_phi_fine, segments_fine: segments, receipt })
}

/// Seed the shared field from retained authored geometry, before any adaptive
/// volume averaging can discard the initial subcell surface location.
pub fn initialize_from_document(
    scene: &crate::initial_scene::SceneDocument,
    diagnostic_volume: f64,
) -> Result<RdfSurface, ValidationError> {
    let dims = crate::initial_scene::lattice_dimensions(scene);
    let c = &scene.container;
    let h = scene.voxel_domain.finest_cell_size_m;
    let mut vertices = Vec::with_capacity((dims[0] as usize + 1) * (dims[1] as usize + 1));
    for y in 0..=dims[1] {
        for x in 0..=dims[0] {
            let point = crate::scene_model::Vec3 {
                x: -0.5 * c.width_m + x as f64 * c.width_m / dims[0] as f64,
                y: y as f64 * c.height_m / dims[1] as f64,
                z: -0.5 * c.depth_m + (dims[2] / 2) as f64 * c.depth_m / dims[2] as f64
                    + 0.5 * c.depth_m / dims[2] as f64,
            };
            let scalar = crate::initial_liquid::initial_liquid_surface_scalar(scene, point, dims);
            vertices.push((scalar / h) as f32);
        }
    }
    publish([dims[0], dims[1]], vertices, diagnostic_volume)
}

/// Fallback for raw state callers with no retained scene geometry.
pub fn initialize_from_volume(graph: &Graph, fields: &Fields) -> Result<RdfSurface, ValidationError> {
    let dimensions = [graph.dimensions[0] as u32, graph.dimensions[1] as u32];
    let mut vertices = Vec::with_capacity((dimensions[0] as usize + 1) * (dimensions[1] as usize + 1));
    for y in 0..=dimensions[1] {
        for x in 0..=dimensions[0] {
            let mut sum = 0.0_f64;
            for dy in [-0.25_f32, 0.25] { for dx in [-0.25_f32, 0.25] {
                let point = [
                    (x as f32 + dx).clamp(0.25, graph.dimensions[0] - 0.25),
                    (y as f32 + dy).clamp(0.25, graph.dimensions[1] - 0.25), 0.0,
                ];
                let scalar = owner_at(graph, point).and_then(|id| {
                    let capacity = fields.capacity[id];
                    (capacity > 1e-8).then(|| {
                        let fill = (fields.density[id] / capacity).clamp(0.0, 1.0);
                        let width = graph.cells[id].widths[0].min(graph.cells[id].widths[1]);
                        (0.5 - fill) * 2.0 * width
                    })
                }).unwrap_or(1.0);
                sum += scalar as f64;
            }}
            vertices.push((sum * 0.25) as f32);
        }
    }
    let volume = graph.cells.iter().enumerate()
        .map(|(i, cell)| fields.density[i] as f64 * cell.measure as f64).sum();
    publish(dimensions, vertices, volume)
}

/// Union a dropped ball into the authoritative vertex scalar.
///
/// The shared field is a signed distance in finest cells and the drop is a
/// disk in that same frame, so the ball carries its own exact scalar and the
/// union is the pointwise minimum — the set operation the dose performs on
/// volume, performed on the surface that positions it. Republished rather
/// than patched: segments and represented area are derived from the vertices,
/// and a surface whose contour disagreed with its own field would be read by
/// redistancing, pressure geometry and the resolution plan alike.
pub fn union_drop(
    surface: &RdfSurface,
    drop: crate::injection::LiquidDrop,
    diagnostic_volume: f64,
) -> Result<RdfSurface, ValidationError> {
    let [centre_x, centre_y] = drop.centre_fine;
    if !(centre_x.is_finite() && centre_y.is_finite() && drop.radius_fine.is_finite()) {
        return Err(ValidationError("dropped ball is not addressable in the shared field".into()));
    }
    let [nx, ny] = surface.dimensions.map(|value| value as usize);
    let stride = nx + 1;
    let mut vertices = surface.vertex_phi_fine.clone();
    if vertices.len() != stride.saturating_mul(ny + 1) {
        return Err(ValidationError("direct level-set vertex count does not match dimensions".into()));
    }
    for y in 0..=ny {
        for x in 0..=nx {
            let ball = ((x as f64 - centre_x).hypot(y as f64 - centre_y) - drop.radius_fine) as f32;
            let at = x + stride * y;
            if ball < vertices[at] {
                vertices[at] = ball;
            }
        }
    }
    publish(surface.dimensions, vertices, diagnostic_volume)
}

pub fn refresh(surface: &RdfSurface, diagnostic_volume: f64) -> Result<RdfSurface, ValidationError> {
    if !diagnostic_volume.is_finite() || diagnostic_volume < 0.0 {
        return Err(ValidationError("direct level-set diagnostic volume is invalid".into()));
    }
    let mut refreshed = surface.clone();
    refreshed.receipt.exact_area_fine = diagnostic_volume;
    refreshed.receipt.signed_area_error_fine =
        refreshed.receipt.represented_area_fine - diagnostic_volume;
    Ok(refreshed)
}

pub fn cell_phi(graph: &Graph, surface: &RdfSurface) -> Result<Vec<f32>, ValidationError> {
    if let Some(sdf) = &surface.adaptive_sdf {
        return graph.cells.iter().map(|cell| sdf.sample([cell.center[0], cell.center[1]])
            .map(|s| s.phi).ok_or_else(|| ValidationError("adaptive SDF has no cell-centre scalar".into()))).collect();
    }
    let distance = RedistanceField::new(surface)?;
    graph.cells.iter().map(|cell| {
        let point = [cell.center[0], cell.center[1]];
        distance.sample(point).or_else(|| sample_scalar(surface, point))
            .ok_or_else(|| ValidationError("direct level-set has no finite cell-centre scalar".into()))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_surface_is_independent_of_volume_geometry() {
        let vertices = (0..=4).flat_map(|_| (0..=4).map(|x| x as f32 - 2.0)).collect();
        let a = publish([4, 4], vertices, 3.0).unwrap();
        let b = refresh(&a, 99.0).unwrap();
        assert_eq!(a.vertex_phi_fine, b.vertex_phi_fine);
        assert_eq!(a.segments_fine, b.segments_fine);
        assert_eq!(a.receipt.represented_area_fine, b.receipt.represented_area_fine);
        assert_eq!(b.receipt.exact_area_fine, 99.0);
        assert_eq!(b.receipt.unresolved_fine_cells, 0);
    }

    #[test]
    fn implied_fill_is_retained_per_fine_cell() {
        let surface = publish([2, 1], vec![-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], 1.0)
            .unwrap();
        let fill = implied_fill_fine_cells(&surface).unwrap();
        assert_eq!(fill.len(), 2);
        assert!(fill.iter().all(|value| (*value - 0.5).abs() < 1e-6));
        assert!((fill.iter().map(|&value| value as f64).sum::<f64>()
            - surface.receipt.represented_area_fine).abs() < 1e-6);
    }

    #[test]
    fn centre_fan_segments_preserve_shared_vertex_signs() {
        let vertices: Vec<f32> = (0..=4).flat_map(|y| (0..=4)
            .map(move |x| ((x as f32 - 2.0).hypot(y as f32 - 2.0)) - 1.4)).collect();
        let surface = publish([4, 4], vertices.clone(), 0.0).unwrap();
        assert!(!surface.segments_fine.is_empty());
        assert_eq!(surface.vertex_phi_fine, vertices);
        for segment in surface.segments_fine.chunks_exact(4) {
            let midpoint = [0.5 * (segment[0] + segment[2]), 0.5 * (segment[1] + segment[3])];
            assert!(RedistanceField::new(&surface).unwrap().sample(midpoint).unwrap().abs() < 1e-5);
        }
    }
}

/// A compatibility raster and contour derived only from adaptive corner values.
pub fn publish_adaptive(sdf: crate::adaptive_sdf::AdaptiveSdf, dimensions: [u32; 2], volume: f64) -> Result<RdfSurface, ValidationError> {
    let mut surface = publish(dimensions, sdf.raster()?, volume)?;
    surface.adaptive_sdf = Some(sdf);
    Ok(surface)
}

pub fn adapt_to_graph(surface: &RdfSurface, graph: &Graph) -> Result<RdfSurface, ValidationError> {
    let sdf = if let Some(old) = &surface.adaptive_sdf { old.remap(graph)? } else {
        crate::adaptive_sdf::AdaptiveSdf::from_graph(graph, |p| sample_scalar(surface, p).map(crate::adaptive_sdf::Sample::new))?
    };
    publish_adaptive(sdf, surface.dimensions, surface.receipt.exact_area_fine)
}
