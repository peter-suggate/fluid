//! Shared 2-D RDF publication. Geometry is evaluated in f64 and rounded only
//! at the published f32 vertices/segments, as in the production slice surface.
use crate::types::{Fields, Graph, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

type Point = [f64; 2];
type Vertex = [f64; 3];
#[derive(Clone, Debug)]
struct Plane {
    normal: Point,
    centre: Point,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RdfReceipt {
    pub interface_cells: usize,
    pub unsupported_cut_partial_cells: usize,
    pub inactive_air_ghost_samples: usize,
    pub reflected_air_ghost_samples: usize,
    pub ambiguous_fine_cells: usize,
    pub unresolved_fine_cells: usize,
    pub exact_area_fine: f64,
    pub represented_area_fine: f64,
    pub signed_area_error_fine: f64,
    pub mean_absolute_partial_cell_error_fine: f64,
    pub maximum_absolute_partial_cell_error_fine: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RdfSurface {
    /// Runtime scalar authority; the fine raster is only its publication.
    #[serde(skip)]
    pub adaptive_sdf: Option<crate::adaptive_sdf::AdaptiveSdf>,
    pub dimensions: [u32; 2],
    /// Non-finite vertices serialize as null, never as a zero level set.
    pub vertex_phi_fine: Vec<f32>,
    pub segments_fine: Vec<f32>,
    pub receipt: RdfReceipt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AirCellGeometry {
    pub minimum: Point,
    pub maximum: Point,
    pub center: Point,
}

/// Cold geometry needed to prove that missing topology is open air. A missing
/// solid plane does not constitute proof. This prevents inventing surface
/// continuation through a solid or outside the world.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RdfSupport {
    pub solid_fraction: Option<Vec<f32>>,
    pub inactive_cells: Vec<AirCellGeometry>,
}

/// Cached once per accepted generation. Order is canonical cell order, even
/// where a coarse edge contributes several finest-lattice vertices.
#[derive(Clone, Debug)]
pub struct RdfTopology {
    generation: u32,
    dimensions: [usize; 2],
    cell_count: usize,
    cells_at_vertex: Vec<Vec<usize>>,
    actual_vertex: Vec<bool>,
    owner_at_voxel: Vec<Option<usize>>,
}
impl RdfTopology {
    pub fn compile(graph: &Graph) -> Result<Self, ValidationError> {
        graph.validate()?;
        if graph.dimension != 2 {
            return Err(ValidationError("RDF slice requires dimension 2".into()));
        }
        let [nx, ny] = [graph.dimensions[0] as usize, graph.dimensions[1] as usize];
        if nx == 0
            || ny == 0
            || graph.dimensions[0] != nx as f32
            || graph.dimensions[1] != ny as f32
        {
            return Err(ValidationError(
                "RDF requires integral fine dimensions".into(),
            ));
        }
        let count = (nx + 1)
            .checked_mul(ny + 1)
            .ok_or_else(|| ValidationError("RDF dimensions overflow".into()))?;
        let mut result = Self {
            generation: graph.topology_generation,
            dimensions: [nx, ny],
            cell_count: graph.cells.len(),
            cells_at_vertex: vec![Vec::new(); count],
            actual_vertex: vec![false; count],
            owner_at_voxel: vec![None; nx * ny],
        };
        for cell in &graph.cells {
            let lo = [cell.minimum[0] as usize, cell.minimum[1] as usize];
            let hi = [cell.maximum[0] as usize, cell.maximum[1] as usize];
            if hi[0] > nx
                || hi[1] > ny
                || (0..2)
                    .any(|a| cell.minimum[a] != lo[a] as f32 || cell.maximum[a] != hi[a] as f32)
            {
                return Err(ValidationError(
                    "RDF cell is not on the fine lattice".into(),
                ));
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    result.cells_at_vertex[x + (nx + 1) * y].push(cell.id as usize);
                }
            }
            for x in [lo[0], hi[0]] {
                for y in [lo[1], hi[1]] {
                    result.actual_vertex[x + (nx + 1) * y] = true;
                }
            }
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    result.owner_at_voxel[x + nx * y] = Some(cell.id as usize);
                }
            }
        }
        Ok(result)
    }
    fn owner(&self, p: Point) -> Option<usize> {
        if p[0] < 0.0
            || p[1] < 0.0
            || p[0] >= self.dimensions[0] as f64
            || p[1] >= self.dimensions[1] as f64
        {
            return None;
        }
        self.owner_at_voxel[p[0].floor() as usize + self.dimensions[0] * p[1].floor() as usize]
    }
    fn open_voxel(&self, support: &RdfSupport, p: Point) -> bool {
        let [nx, ny] = self.dimensions;
        if p[0] < 0.0 || p[1] < 0.0 || p[0] >= nx as f64 || p[1] >= ny as f64 {
            return false;
        }
        support
            .solid_fraction
            .as_ref()
            .and_then(|s| s.get(p[0].floor() as usize + nx * p[1].floor() as usize))
            .is_some_and(|v| *v <= 0.0)
    }
    fn open_box(&self, support: &RdfSupport, geometry: &AirCellGeometry) -> bool {
        if (0..2)
            .any(|a| geometry.minimum[a] < 0.0 || geometry.maximum[a] > self.dimensions[a] as f64)
        {
            return false;
        }
        for y in geometry.minimum[1].floor() as usize..geometry.maximum[1].ceil() as usize {
            for x in geometry.minimum[0].floor() as usize..geometry.maximum[0].ceil() as usize {
                if !self.open_voxel(support, [x as f64 + 0.5, y as f64 + 0.5]) {
                    return false;
                }
            }
        }
        true
    }
}

fn partial_open(fields: &Fields, cell: usize) -> bool {
    let cap = fields.capacity[cell] as f64;
    let fill = if cap > 1e-8 {
        fields.density[cell] as f64 / cap
    } else {
        0.0
    };
    cap >= 0.999999 && fill > 1e-6 && fill < 1.0 - 1e-6
}
fn segment(graph: &Graph, fields: &Fields, cell: usize) -> Vec<Point> {
    let c = &graph.cells[cell];
    let n = [
        fields.interface_normal[2 * cell] as f64,
        fields.interface_normal[2 * cell + 1] as f64,
    ];
    let d = fields.interface_offset[cell] as f64;
    let mut points: Vec<Point> = Vec::with_capacity(4);
    let mut insert = |p: Point| {
        if (0..2).any(|a| p[a] < c.minimum[a] as f64 - 1e-8 || p[a] > c.maximum[a] as f64 + 1e-8)
            || points
                .iter()
                .any(|q| (p[0] - q[0]).hypot(p[1] - q[1]) < 1e-7)
        {
            return;
        }
        points.push(p);
    };
    if n[1].abs() > 1e-12 {
        for x in [c.minimum[0] as f64, c.maximum[0] as f64] {
            insert([
                x,
                c.center[1] as f64 + (d - n[0] * (x - c.center[0] as f64)) / n[1],
            ]);
        }
    }
    if n[0].abs() > 1e-12 {
        for y in [c.minimum[1] as f64, c.maximum[1] as f64] {
            insert([
                c.center[0] as f64 + (d - n[1] * (y - c.center[1] as f64)) / n[0],
                y,
            ]);
        }
    }
    points.truncate(2);
    points
}
fn weighted_phi(planes: &[&Plane], p: Point) -> Option<f64> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for plane in planes {
        let dx = p[0] - plane.centre[0];
        let dy = p[1] - plane.centre[1];
        let distance = plane.normal[0] * dx + plane.normal[1] * dy;
        let weight = distance * distance / (dx * dx + dy * dy).max(1e-12);
        weighted += weight * distance;
        total += weight;
    }
    (total > 1e-8).then_some(weighted / total)
}
fn moments(samples: &[Vertex]) -> (Vertex, [f64; 5]) {
    let count = samples.len() as f64;
    let mut mean = [0.0; 3];
    for sample in samples {
        for a in 0..3 {
            mean[a] += sample[a];
        }
    }
    for value in &mut mean {
        *value *= 1.0 / count;
    }
    let (mut xx, mut xy, mut yy, mut bx, mut by) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for s in samples {
        let x = s[0] - mean[0];
        let y = s[1] - mean[1];
        let value = s[2] - mean[2];
        xx += x * x;
        xy += x * y;
        yy += y * y;
        bx += x * value;
        by += y * value;
    }
    (mean, [xx, xy, yy, bx, by])
}
fn rank(samples: &[Vertex]) -> u8 {
    if samples.len() < 2 {
        return 0;
    }
    let (_, [xx, xy, yy, _, _]) = moments(samples);
    let trace = xx + yy;
    if trace <= 1e-12 {
        0
    } else if (xx * yy - xy * xy).abs() > 64.0 * f32::EPSILON as f64 * trace * trace {
        2
    } else {
        1
    }
}
fn affine(samples: &[Vertex], p: Point, preserve_slope: bool) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let (mean, [xx, xy, yy, bx, by]) = moments(samples);
    let det = xx * yy - xy * xy;
    let trace = xx + yy;
    let (gx, gy) = if det.abs() > 64.0 * f32::EPSILON as f64 * trace * trace {
        ((yy * bx - xy * by) / det, (xx * by - xy * bx) / det)
    } else if preserve_slope && trace > 1e-12 {
        (bx / trace, by / trace)
    } else {
        (0.0, 0.0)
    };
    Some(mean[2] + gx * (p[0] - mean[0]) + gy * (p[1] - mean[1]))
}
fn reflected(graph: &Graph, ids: &[usize], p: Point, sx: f64, sy: f64) -> Option<AirCellGeometry> {
    let mut candidates: Vec<_> = ids
        .iter()
        .copied()
        .filter(|&i| {
            let c = &graph.cells[i];
            usize::from((c.center[0] as f64 >= p[0]) != (sx > 0.0))
                + usize::from((c.center[1] as f64 >= p[1]) != (sy > 0.0))
                == 1
        })
        .collect();
    candidates.sort_by(|&a, &b| {
        let a = &graph.cells[a];
        let b = &graph.cells[b];
        ((a.center[0] as f64 - p[0]).hypot(a.center[1] as f64 - p[1]))
            .total_cmp(&((b.center[0] as f64 - p[0]).hypot(b.center[1] as f64 - p[1])))
            .then(a.id.cmp(&b.id))
    });
    candidates.first().map(|&i| {
        let c = &graph.cells[i];
        let center = [
            p[0] + sx * (c.center[0] as f64 - p[0]).abs(),
            p[1] + sy * (c.center[1] as f64 - p[1]).abs(),
        ];
        AirCellGeometry {
            minimum: [
                center[0] - c.widths[0] as f64 * 0.5,
                center[1] - c.widths[1] as f64 * 0.5,
            ],
            maximum: [
                center[0] + c.widths[0] as f64 * 0.5,
                center[1] + c.widths[1] as f64 * 0.5,
            ],
            center,
        }
    })
}
fn triangles(x: f64, y: f64, phi: [f64; 4]) -> [[Vertex; 3]; 4] {
    let corners = [
        [x, y, phi[0]],
        [x + 1.0, y, phi[1]],
        [x + 1.0, y + 1.0, phi[2]],
        [x, y + 1.0, phi[3]],
    ];
    let centre = [
        x + 0.5,
        y + 0.5,
        ((phi[0] + phi[2]) + (phi[1] + phi[3])) * 0.25,
    ];
    std::array::from_fn(|i| [corners[i], corners[(i + 1) % 4], centre])
}
fn clipped_area(triangle: &[Vertex; 3]) -> f64 {
    let mut polygon: Vec<Point> = Vec::with_capacity(4);
    for i in 0..3 {
        let a = triangle[i];
        let b = triangle[(i + 1) % 3];
        if a[2] <= 0.0 {
            polygon.push([a[0], a[1]])
        }
        if (a[2] < 0.0) != (b[2] < 0.0) {
            let t = a[2] / (a[2] - b[2]);
            polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    let mut sum = 0.0;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        sum += a[0] * b[1] - b[0] * a[1]
    }
    sum.abs() * 0.5
}

pub fn reconstruct_shared_rdf(
    graph: &Graph,
    fields: &Fields,
    cache: &RdfTopology,
    support: &RdfSupport,
) -> Result<RdfSurface, ValidationError> {
    fields.validate_for(graph)?;
    if graph.dimension != 2
        || cache.generation != graph.topology_generation
        || cache.cell_count != graph.cells.len()
    {
        return Err(ValidationError(
            "RDF cache belongs to another topology generation".into(),
        ));
    }
    let [nx, ny] = cache.dimensions;
    let stride = nx + 1;
    let mut receipt = RdfReceipt::default();
    let mut planes = Vec::new();
    let mut by_cell = vec![None; graph.cells.len()];
    for cell in &graph.cells {
        let id = cell.id as usize;
        let cap = fields.capacity[id] as f64;
        let rho = if cap > 1e-8 {
            fields.density[id] as f64 / cap
        } else {
            0.0
        };
        if cap < 0.999999 && rho > 1e-6 && rho < 1.0 - 1e-6 {
            receipt.unsupported_cut_partial_cells += 1;
        }
        if !partial_open(fields, id) {
            continue;
        }
        let normal = [
            fields.interface_normal[2 * id] as f64,
            fields.interface_normal[2 * id + 1] as f64,
        ];
        if normal[0].hypot(normal[1]) <= 0.5 {
            continue;
        }
        let points = segment(graph, fields, id);
        if points.len() != 2 {
            continue;
        }
        by_cell[id] = Some(planes.len());
        planes.push(Plane {
            normal,
            centre: [
                (points[0][0] + points[1][0]) * 0.5,
                (points[0][1] + points[1][1]) * 0.5,
            ],
        });
    }
    receipt.interface_cells = planes.len();
    let mut cell_phi = vec![f64::NAN; graph.cells.len()];
    for cell in &graph.cells {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for x in [cell.minimum[0] as usize, cell.maximum[0] as usize] {
            for y in [cell.minimum[1] as usize, cell.maximum[1] as usize] {
                for &id in &cache.cells_at_vertex[x + stride * y] {
                    if seen.insert(id) {
                        ids.push(id)
                    }
                }
            }
        }
        let neighbors: Vec<_> = ids
            .iter()
            .filter_map(|&i| by_cell[i].map(|j| &planes[j]))
            .collect();
        // Cell centres use any positive orientation weight; the ghost helper's
        // 1e-8 support threshold is intentionally stronger.
        let p = [cell.center[0] as f64, cell.center[1] as f64];
        let (mut sum, mut total) = (0.0, 0.0);
        for plane in neighbors {
            let dx = p[0] - plane.centre[0];
            let dy = p[1] - plane.centre[1];
            let distance = plane.normal[0] * dx + plane.normal[1] * dy;
            let weight = distance * distance / (dx * dx + dy * dy).max(1e-12);
            sum += weight * distance;
            total += weight;
        }
        if total > 0.0 {
            cell_phi[cell.id as usize] = sum / total;
        }
    }
    let mut vertices = vec![f32::NAN; stride * (ny + 1)];
    for y in 0..=ny {
        for x in 0..=nx {
            let at = x + stride * y;
            let ids = &cache.cells_at_vertex[at];
            let p = [x as f64, y as f64];
            if !cache.actual_vertex[at] {
                let local: Vec<_> = ids
                    .iter()
                    .filter_map(|&i| by_cell[i].map(|j| &planes[j]))
                    .collect();
                if !local.is_empty() {
                    let sum: f64 = local
                        .iter()
                        .map(|plane| {
                            plane.normal[0] * (p[0] - plane.centre[0])
                                + plane.normal[1] * (p[1] - plane.centre[1])
                        })
                        .sum();
                    vertices[at] = (sum / local.len() as f64) as f32;
                    continue;
                }
            }
            let mut samples: Vec<Vertex> = ids
                .iter()
                .copied()
                .filter(|&i| cell_phi[i].is_finite())
                .map(|i| {
                    [
                        graph.cells[i].center[0] as f64,
                        graph.cells[i].center[1] as f64,
                        cell_phi[i],
                    ]
                })
                .collect();
            if !ids.is_empty() && rank(&samples) < 2 && !fields.solid_motion_active {
                let local: Vec<_> = ids
                    .iter()
                    .filter_map(|&i| by_cell[i].map(|j| &planes[j]))
                    .collect();
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        let probe = [p[0] + sx * 0.25, p[1] + sy * 0.25];
                        if probe[0] < 0.0
                            || probe[1] < 0.0
                            || probe[0] >= nx as f64
                            || probe[1] >= ny as f64
                            || cache.owner(probe).is_some()
                        {
                            continue;
                        }
                        let inactive = support.inactive_cells.iter().find(|c| {
                            (0..2).all(|a| probe[a] >= c.minimum[a] && probe[a] < c.maximum[a])
                        });
                        let ghost = inactive
                            .cloned()
                            .or_else(|| reflected(graph, ids, p, sx, sy));
                        let Some(ghost) = ghost else { continue };
                        if !cache.open_box(support, &ghost) {
                            continue;
                        }
                        let Some(value) = weighted_phi(&local, ghost.center) else {
                            continue;
                        };
                        if samples
                            .iter()
                            .any(|s| (s[0] - ghost.center[0]).hypot(s[1] - ghost.center[1]) < 1e-7)
                        {
                            continue;
                        }
                        samples.push([ghost.center[0], ghost.center[1], value]);
                        if inactive.is_some() {
                            receipt.inactive_air_ghost_samples += 1
                        } else {
                            receipt.reflected_air_ghost_samples += 1
                        }
                    }
                }
            }
            if samples.is_empty() {
                for &i in ids {
                    let cap = fields.capacity[i] as f64;
                    if cap <= 1e-8 {
                        continue;
                    }
                    let c = &graph.cells[i];
                    let fill = (fields.density[i] as f64 / cap).clamp(0.0, 1.0);
                    samples.push([
                        c.center[0] as f64,
                        c.center[1] as f64,
                        (0.5 - fill) * 4.0 * (c.widths[0].max(c.widths[1]) as f64),
                    ]);
                }
            }
            if samples.is_empty() {
                continue;
            }
            let fit = if samples.len() >= 3 {
                affine(&samples, p, true)
            } else {
                None
            };
            vertices[at] = fit
                .unwrap_or_else(|| samples.iter().map(|v| v[2]).sum::<f64>() / samples.len() as f64)
                as f32;
        }
    }
    // Pure-phase corner certificates precede exact full/empty aligned faces.
    for y in 0..=ny {
        for x in 0..=nx {
            let at = x + stride * y;
            let ids = &cache.cells_at_vertex[at];
            if ids.iter().any(|&i| by_cell[i].is_some()) {
                continue;
            }
            let p = [x as f64, y as f64];
            let (mut liquid, mut air) = (false, false);
            let mut width = 1.0_f64;
            for &i in ids {
                let c = &graph.cells[i];
                width = width.min(c.widths[0].min(c.widths[1]) as f64);
                let cap = fields.capacity[i] as f64;
                if cap <= 1e-8 {
                    continue;
                }
                let fill = fields.density[i] as f64 / cap;
                if fill >= 1.0 - 1e-6 {
                    liquid = true
                } else if fill <= 1e-6 {
                    air = true
                }
            }
            if !fields.solid_motion_active {
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        let probe = [p[0] + sx * 0.25, p[1] + sy * 0.25];
                        if cache.owner(probe).is_none() && cache.open_voxel(support, probe) {
                            air = true
                        }
                    }
                }
            }
            if vertices[at].is_finite() {
                if liquid && !air && vertices[at] >= 0.0 {
                    vertices[at] = (-0.25 * width) as f32
                } else if air && !liquid && vertices[at] <= 0.0 {
                    vertices[at] = (0.25 * width) as f32
                }
            }
        }
    }
    let phase_at = |p: Point| -> i8 {
        if let Some(i) = cache.owner(p) {
            let cap = fields.capacity[i] as f64;
            if cap <= 1e-8 {
                return 0;
            }
            let fill = fields.density[i] as f64 / cap;
            if fill >= 1.0 - 1e-6 {
                -1
            } else if fill <= 1e-6 {
                1
            } else {
                0
            }
        } else if !fields.solid_motion_active && cache.open_voxel(support, p) {
            1
        } else {
            0
        }
    };
    for y in 0..=ny {
        for x in 0..=nx {
            let at = x + stride * y;
            if cache.cells_at_vertex[at]
                .iter()
                .any(|&i| by_cell[i].is_some())
            {
                continue;
            }
            for tangent in [-0.25, 0.25] {
                if phase_at([x as f64 - 1e-4, y as f64 + tangent])
                    * phase_at([x as f64 + 1e-4, y as f64 + tangent])
                    == -1
                    || phase_at([x as f64 + tangent, y as f64 - 1e-4])
                        * phase_at([x as f64 + tangent, y as f64 + 1e-4])
                        == -1
                {
                    vertices[at] = 0.0;
                    break;
                }
            }
        }
    }
    let corners = |x: usize, y: usize| {
        [
            vertices[x + stride * y] as f64,
            vertices[x + 1 + stride * y] as f64,
            vertices[x + 1 + stride * (y + 1)] as f64,
            vertices[x + stride * (y + 1)] as f64,
        ]
    };
    let mut segments = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let phi = corners(x, y);
            let cuts = (0..4)
                .filter(|&i| {
                    phi[i].is_finite()
                        && phi[(i + 1) % 4].is_finite()
                        && (phi[i] < 0.0) != (phi[(i + 1) % 4] < 0.0)
                })
                .count();
            if cuts == 4 {
                receipt.ambiguous_fine_cells += 1
            } else if cuts != 0 && cuts != 2 {
                receipt.unresolved_fine_cells += 1;
            }
            for tri in triangles(x as f64, y as f64, phi) {
                let mut points = Vec::new();
                for i in 0..3 {
                    let a = tri[i];
                    let b = tri[(i + 1) % 3];
                    if !a[2].is_finite()
                        || !b[2].is_finite()
                        || (a[2] < 0.0) == (b[2] < 0.0)
                        || a[2] == b[2]
                    {
                        continue;
                    }
                    let t = a[2] / (a[2] - b[2]);
                    points.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
                if points.len() == 2 {
                    for p in points {
                        segments.extend(p.map(|v| v as f32));
                    }
                } else if !points.is_empty() {
                    receipt.unresolved_fine_cells += 1
                }
            }
        }
    }
    let mut partial_count = 0;
    for c in &graph.cells {
        let i = c.id as usize;
        let target = fields.density[i] as f64 * c.measure as f64;
        receipt.exact_area_fine += target;
        if !partial_open(fields, i) {
            receipt.represented_area_fine += target;
            continue;
        }
        let (mut area, mut valid) = (0.0, true);
        for y in c.minimum[1] as usize..c.maximum[1] as usize {
            for x in c.minimum[0] as usize..c.maximum[0] as usize {
                let phi = corners(x, y);
                if phi.iter().any(|v| !v.is_finite()) {
                    valid = false;
                    continue;
                }
                for tri in triangles(x as f64, y as f64, phi) {
                    area += clipped_area(&tri);
                }
            }
        }
        if !valid {
            continue;
        }
        receipt.represented_area_fine += area;
        let error = (area - target).abs();
        receipt.mean_absolute_partial_cell_error_fine += error;
        receipt.maximum_absolute_partial_cell_error_fine =
            receipt.maximum_absolute_partial_cell_error_fine.max(error);
        partial_count += 1;
    }
    receipt.mean_absolute_partial_cell_error_fine /= partial_count.max(1) as f64;
    receipt.signed_area_error_fine = receipt.represented_area_fine - receipt.exact_area_fine;
    Ok(RdfSurface {
        adaptive_sdf: None,
        dimensions: [nx as u32, ny as u32],
        vertex_phi_fine: vertices,
        segments_fine: segments,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn centre_fan_is_reflection_invariant() {
        let a: f64 = triangles(0.0, 0.0, [-0.8, 0.4, 0.7, -0.2])
            .iter()
            .map(clipped_area)
            .sum();
        let b: f64 = triangles(0.0, 0.0, [0.4, -0.8, -0.2, 0.7])
            .iter()
            .map(clipped_area)
            .sum();
        assert!((a - b).abs() < 1e-14);
    }
    #[test]
    fn missing_solid_plane_does_not_prove_air() {
        let cache = RdfTopology {
            generation: 1,
            dimensions: [1, 1],
            cell_count: 0,
            cells_at_vertex: vec![vec![]; 4],
            actual_vertex: vec![false; 4],
            owner_at_voxel: vec![None],
        };
        assert!(!cache.open_voxel(&RdfSupport::default(), [0.5, 0.5]));
        assert!(cache.open_voxel(
            &RdfSupport {
                solid_fraction: Some(vec![0.0]),
                inactive_cells: vec![]
            },
            [0.5, 0.5]
        ));
    }
}
