//! The 2-D counterpart of the resident 3-D shared-corner level set.
//!
//! Only accepted-cell corners carry scalar authority. Hanging vertices are
//! constrained to the coarser edge, samples are bilinear, and redistance uses
//! the same edge seeds, affine-plane certificate and 16 Jacobi rounds as WGSL.
//! The fine raster is a derived publication for the existing lab ABI.

use crate::types::{Graph, ValidationError};
use std::collections::BTreeMap;

const METRIC: u8 = 3;
const AIR: u8 = 1;
const LIQUID: u8 = 2;
const BAND: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub phi: f32,
    pub support: u8,
}
impl Sample {
    pub fn new(phi: f32) -> Self {
        Self {
            phi,
            support: if phi.abs() <= BAND {
                METRIC
            } else if phi < 0.0 {
                LIQUID
            } else {
                AIR
            },
        }
    }
}
#[derive(Clone, Debug)]
struct Cell {
    lower: [usize; 2],
    upper: [usize; 2],
    corners: [usize; 4],
}
#[derive(Clone, Debug)]
struct Vertex {
    position: [usize; 2],
    incident: Vec<usize>,
    constraint: Option<(usize, usize, f32, usize)>,
}
#[derive(Clone, Debug)]
pub struct AdaptiveSdf {
    dimensions: [usize; 2],
    generation: u32,
    cells: Vec<Cell>,
    vertices: Vec<Vertex>,
    samples: Vec<Sample>,
    // As in the lab's Graph, this is an owner index, never scalar authority.
    owners: Vec<Option<usize>>,
    constraint_order: Vec<usize>,
}
#[derive(Default, Debug)]
pub struct RedistanceReceipt {
    pub updated: usize,
    pub fallback: usize,
    pub seeds: usize,
}

impl AdaptiveSdf {
    pub fn from_graph(
        graph: &Graph,
        mut seed: impl FnMut([f32; 2]) -> Option<Sample>,
    ) -> Result<Self, ValidationError> {
        if graph.dimension != 2 {
            return Err(ValidationError("adaptive SDF requires a 2-D graph".into()));
        }
        let dimensions = [graph.dimensions[0] as usize, graph.dimensions[1] as usize];
        let mut ids = BTreeMap::new();
        for c in &graph.cells {
            for k in 0..4 {
                ids.insert(
                    [
                        if k & 1 == 0 {
                            c.minimum[0]
                        } else {
                            c.maximum[0]
                        } as usize,
                        if k & 2 == 0 {
                            c.minimum[1]
                        } else {
                            c.maximum[1]
                        } as usize,
                    ],
                    0,
                );
            }
        }
        let mut vertices = Vec::with_capacity(ids.len());
        for (p, id) in &mut ids {
            *id = vertices.len();
            vertices.push(Vertex {
                position: *p,
                incident: Vec::new(),
                constraint: None,
            });
        }
        let mut cells = Vec::with_capacity(graph.cells.len());
        let mut owners = vec![None; dimensions[0] * dimensions[1]];
        for c in &graph.cells {
            let lower = [c.minimum[0] as usize, c.minimum[1] as usize];
            let upper = [c.maximum[0] as usize, c.maximum[1] as usize];
            let corners = std::array::from_fn(|k| {
                ids[&[
                    if k & 1 == 0 { lower[0] } else { upper[0] },
                    if k & 2 == 0 { lower[1] } else { upper[1] },
                ]]
            });
            let id = cells.len();
            for y in lower[1]..upper[1] {
                for x in lower[0]..upper[0] {
                    owners[x + dimensions[0] * y] = Some(id);
                }
            }
            cells.push(Cell {
                lower,
                upper,
                corners,
            });
        }
        for vertex in &mut vertices {
            let p = vertex.position;
            for dy in [0, 1] {
                for dx in [0, 1] {
                    if p[0] < dx || p[1] < dy {
                        continue;
                    }
                    let q = [p[0] - dx, p[1] - dy];
                    if q[0] >= dimensions[0] || q[1] >= dimensions[1] {
                        continue;
                    }
                    if let Some(c) = owners[q[0] + dimensions[0] * q[1]] {
                        if !vertex.incident.contains(&c) {
                            vertex.incident.push(c);
                        }
                    }
                }
            }
            vertex.incident.sort_unstable();
            for &id in &vertex.incident {
                let cell = &cells[id];
                for axis in 0..2 {
                    let tangent = 1 - axis;
                    if (p[axis] == cell.lower[axis] || p[axis] == cell.upper[axis])
                        && p[tangent] > cell.lower[tangent]
                        && p[tangent] < cell.upper[tangent]
                    {
                        let side = usize::from(p[axis] == cell.upper[axis]) << axis;
                        let width = cell.upper[tangent] - cell.lower[tangent];
                        if vertex.constraint.is_none_or(|old| width > old.3) {
                            vertex.constraint = Some((
                                cell.corners[side],
                                cell.corners[side | (1 << tangent)],
                                (p[tangent] - cell.lower[tangent]) as f32 / width as f32,
                                width,
                            ));
                        }
                    }
                }
            }
        }
        let samples = vertices
            .iter()
            .map(|v| {
                seed(v.position.map(|x| x as f32))
                    .filter(|s| s.phi.is_finite())
                    .ok_or_else(|| {
                        ValidationError(format!("adaptive SDF lacks source at {:?}", v.position))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut constraint_order: Vec<_> = (0..vertices.len())
            .filter(|&v| vertices[v].constraint.is_some())
            .collect();
        constraint_order.sort_by_key(|&v| std::cmp::Reverse(vertices[v].constraint.unwrap().3));
        let mut result = Self {
            dimensions,
            generation: graph.topology_generation,
            cells,
            vertices,
            samples,
            owners,
            constraint_order,
        };
        result.constrain();
        Ok(result)
    }
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn constrained_count(&self) -> usize {
        self.constraint_order.len()
    }
    pub fn generation(&self) -> u32 {
        self.generation
    }
    pub fn positions(&self) -> impl Iterator<Item = [f32; 2]> + '_ {
        self.vertices.iter().map(|v| v.position.map(|x| x as f32))
    }
    pub fn independent(&self, v: usize) -> bool {
        self.vertices[v].constraint.is_none()
    }
    pub fn value(&self, v: usize) -> Sample {
        self.samples[v]
    }
    pub fn set(&mut self, v: usize, sample: Sample) {
        self.samples[v] = sample;
    }
    pub fn constrain(&mut self) {
        for &v in &self.constraint_order {
            let (a, b, t, _) = self.vertices[v].constraint.unwrap();
            self.samples[v] = Sample {
                phi: self.samples[a].phi * (1.0 - t) + self.samples[b].phi * t,
                support: self.samples[a].support.min(self.samples[b].support),
            };
        }
    }
    fn owner(&self, p: [f32; 2]) -> Option<usize> {
        if p.iter().any(|v| !v.is_finite())
            || (0..2).any(|a| p[a] < 0.0 || p[a] > self.dimensions[a] as f32)
        {
            return None;
        }
        let q = p.map(|v| v.floor() as usize);
        for k in 0..4 {
            let dx = k & 1;
            let dy = (k >> 1) & 1;
            if q[0] < dx
                || q[1] < dy
                || (dx > 0 && p[0] != q[0] as f32)
                || (dy > 0 && p[1] != q[1] as f32)
            {
                continue;
            }
            let x = q[0] - dx;
            let y = q[1] - dy;
            if x < self.dimensions[0] && y < self.dimensions[1] {
                if let Some(id) = self.owners[x + self.dimensions[0] * y] {
                    return Some(id);
                }
            }
        }
        None
    }
    fn sample_cell(&self, id: usize, p: [f32; 2]) -> Sample {
        let c = &self.cells[id];
        let t: [f32; 2] = std::array::from_fn(|a| {
            ((p[a] - c.lower[a] as f32) / (c.upper[a] - c.lower[a]) as f32).clamp(0.0, 1.0)
        });
        let mut result = Sample {
            phi: 0.0,
            support: METRIC,
        };
        for k in 0..4 {
            let w = if k & 1 == 0 { 1.0 - t[0] } else { t[0] }
                * if k & 2 == 0 { 1.0 - t[1] } else { t[1] };
            if w == 0.0 {
                continue;
            }
            result.phi += w * self.samples[c.corners[k]].phi;
            result.support = result.support.min(self.samples[c.corners[k]].support);
        }
        result
    }
    pub fn sample(&self, p: [f32; 2]) -> Option<Sample> {
        self.owner(p).map(|id| self.sample_cell(id, p))
    }
    /// Missing sparse pages are air, but only extend an existing air certificate.
    /// This is used on growth/retirement and by the derived full-domain raster.
    pub fn sample_extended(&self, p: [f32; 2]) -> Option<Sample> {
        if let Some(s) = self.sample(p) {
            return Some(s);
        }
        if self.cells.is_empty() {
            return Some(Sample {
                phi: length(self.dimensions.map(|v| v as f32)) + BAND,
                support: AIR,
            });
        }
        let mut best = None;
        let mut distance = f32::INFINITY;
        for (id, c) in self.cells.iter().enumerate() {
            let q: [f32; 2] =
                std::array::from_fn(|a| p[a].clamp(c.lower[a] as f32, c.upper[a] as f32));
            let d = length(sub(p, q));
            if d < distance {
                let s = self.sample_cell(id, q);
                if s.phi >= 0.0 {
                    best = Some(Sample {
                        phi: s.phi + d,
                        support: AIR,
                    });
                    distance = d;
                }
            }
        }
        best
    }
    pub fn raster(&self) -> Result<Vec<f32>, ValidationError> {
        let [nx, ny] = self.dimensions;
        (0..(nx + 1) * (ny + 1))
            .map(|i| {
                self.sample_extended([(i % (nx + 1)) as f32, (i / (nx + 1)) as f32])
                    .map(|s| s.phi)
                    .ok_or_else(|| {
                        ValidationError("adaptive SDF cannot publish unsupported phase".into())
                    })
            })
            .collect()
    }
    pub fn remap(&self, graph: &Graph) -> Result<Self, ValidationError> {
        Self::from_graph(graph, |p| self.sample_extended(p))
    }
    fn affine_plane(&self, c: usize) -> Option<[f32; 2]> {
        let cell = &self.cells[c];
        let s = cell.corners.map(|v| self.samples[v]);
        if s.iter().any(|s| s.support != METRIC) {
            return None;
        }
        let w = [
            (cell.upper[0] - cell.lower[0]) as f32,
            (cell.upper[1] - cell.lower[1]) as f32,
        ];
        let g = [(s[1].phi - s[0].phi) / w[0], (s[2].phi - s[0].phi) / w[1]];
        let scale = (s[0].phi.abs() + length(w)).max(1.0);
        if (s[3].phi - s[0].phi - g[0] * w[0] - g[1] * w[1]).abs() > 1e-5 * scale
            || (length(g) - 1.0).abs() > 2e-4
        {
            return None;
        }
        Some(g)
    }
    pub fn redistance(&mut self) -> RedistanceReceipt {
        let n = self.vertices.len();
        let original = self.samples.clone();
        let mut bands = vec![BAND; n];
        let mut fixed = vec![false; n];
        let mut seeds = vec![[0.0; 2]; n];
        let mut refs = vec![None; n];
        let mut values = vec![0.0; n];
        let mut receipt = RedistanceReceipt::default();
        for v in 0..n {
            let p = self.vertices[v].position.map(|x| x as f32);
            for &c in &self.vertices[v].incident {
                let corners = self.cells[c].corners;
                let lo = corners
                    .iter()
                    .map(|&i| original[i].phi)
                    .fold(f32::INFINITY, f32::min);
                let hi = corners
                    .iter()
                    .map(|&i| original[i].phi)
                    .fold(f32::NEG_INFINITY, f32::max);
                if lo <= 0.0 && hi >= 0.0 {
                    let cell = &self.cells[c];
                    bands[v] = bands[v].max(
                        length([
                            (cell.upper[0] - cell.lower[0]) as f32,
                            (cell.upper[1] - cell.lower[1]) as f32,
                        ])
                        .ceil(),
                    );
                }
            }
            if original[v].support == METRIC {
                for &c in &self.vertices[v].incident {
                    if let Some(g) = self.affine_plane(c) {
                        seeds[v] = [p[0] - original[v].phi * g[0], p[1] - original[v].phi * g[1]];
                        fixed[v] = true;
                        refs[v] = Some(v);
                        break;
                    }
                }
                if refs[v].is_none() {
                    if original[v].phi == 0.0 {
                        seeds[v] = p;
                        refs[v] = Some(v);
                    } else {
                        let mut best = f32::INFINITY;
                        for &c in &self.vertices[v].incident {
                            for &other in &self.cells[c].corners {
                                if other == v
                                    || (original[other].support != METRIC
                                        && original[other].phi.abs() > bands[v])
                                {
                                    continue;
                                }
                                let q = self.vertices[other].position.map(|x| x as f32);
                                if (q[0] != p[0]) == (q[1] != p[1])
                                    || original[v].phi * original[other].phi > 0.0
                                {
                                    continue;
                                }
                                let t = original[v].phi.abs()
                                    / (original[v].phi.abs() + original[other].phi.abs());
                                let point = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
                                let d = length(sub(point, p));
                                if d < best {
                                    best = d;
                                    seeds[v] = point;
                                    refs[v] = Some(v);
                                }
                            }
                        }
                    }
                }
            }
            if refs[v].is_some() {
                receipt.seeds += 1;
            }
            values[v] = sign(original[v].phi)
                * if refs[v].is_some() {
                    length(sub(seeds[v], p))
                } else {
                    bands[v] + 1.0
                };
        }
        for _ in 0..16 {
            let mut next_refs = refs.clone();
            let mut next_values = values.clone();
            for v in 0..n {
                if fixed[v] {
                    next_refs[v] = Some(v);
                    next_values[v] = original[v].phi;
                    continue;
                }
                let p = self.vertices[v].position.map(|x| x as f32);
                let mut best = f32::INFINITY;
                let mut winner = None;
                for &c in &self.vertices[v].incident {
                    for &other in &self.cells[c].corners {
                        if let Some(id) = refs[other] {
                            let d = length(sub(seeds[id], p));
                            if d < best || (d == best && winner.is_none_or(|old| id < old)) {
                                best = d;
                                winner = Some(id);
                            }
                        }
                    }
                }
                if best > bands[v] {
                    winner = None;
                    best = bands[v] + 1.0;
                }
                next_refs[v] = winner;
                next_values[v] = sign(original[v].phi) * best;
            }
            let stable = next_refs == refs && next_values == values;
            refs = next_refs;
            values = next_values;
            if stable {
                break;
            }
        }
        for v in 0..n {
            if refs[v].is_some() && values[v].abs() <= bands[v] {
                self.samples[v] = Sample {
                    phi: values[v],
                    support: METRIC,
                };
                receipt.updated += 1;
            } else {
                self.samples[v] = Sample {
                    phi: original[v].phi,
                    support: if original[v].support == METRIC {
                        if original[v].phi < 0.0 {
                            LIQUID
                        } else {
                            AIR
                        }
                    } else {
                        original[v].support
                    },
                };
                receipt.fallback += 1;
            }
        }
        self.constrain();
        receipt
    }
}
fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}
fn sign(v: f32) -> f32 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph(boxes: &[[usize; 4]], dims: [f32; 3], generation: u32) -> Graph {
        Graph {
            dimension: 2,
            dimensions: dims,
            topology_generation: generation,
            cells: boxes
                .iter()
                .enumerate()
                .map(|(id, b)| crate::Cell {
                    id: id as u32,
                    minimum: [b[0] as f32, b[1] as f32, 0.0],
                    maximum: [b[2] as f32, b[3] as f32, 1.0],
                    widths: [(b[2] - b[0]) as f32, (b[3] - b[1]) as f32, 1.0],
                    center: [(b[0] + b[2]) as f32 * 0.5, (b[1] + b[3]) as f32 * 0.5, 0.5],
                    measure: ((b[2] - b[0]) * (b[3] - b[1])) as f32,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }
    #[test]
    fn shared_hanging_vertex_reads_coarse_edge_and_is_continuous() {
        let g = graph(
            &[[0, 0, 4, 4], [4, 0, 6, 2], [4, 2, 6, 4]],
            [6.0, 4.0, 1.0],
            1,
        );
        let sdf = AdaptiveSdf::from_graph(&g, |p| Some(Sample::new(p[1] * p[1]))).unwrap();
        assert_eq!(sdf.vertex_count(), 8);
        assert_eq!(sdf.constrained_count(), 1);
        // The authored value at (4,2) is 4. The coarse master edge requires 8.
        for x in [4.0 - 1e-5, 4.0, 4.0 + 1e-5] {
            assert!((sdf.sample([x, 2.0]).unwrap().phi - 8.0).abs() < 1e-4);
        }
    }
    #[test]
    fn bilinear_field_is_authoritative_even_if_publication_raster_is_modified() {
        let g = graph(&[[0, 0, 4, 4]], [4.0, 4.0, 1.0], 1);
        let sdf = AdaptiveSdf::from_graph(&g, |p| Some(Sample::new(p[0] * p[1] - 1.0))).unwrap();
        assert_eq!(sdf.vertex_count(), 4);
        let mut surface = crate::levelset_surface::publish_adaptive(sdf, [4, 4], 0.0).unwrap();
        surface.vertex_phi_fine.fill(999.0);
        assert!(
            (crate::levelset_redistance::sample_scalar(&surface, [0.25, 0.75]).unwrap() + 0.8125)
                .abs()
                < 1e-6
        );
    }
    #[test]
    fn affine_distance_survives_redistance_and_split_merge_transfer() {
        let coarse = graph(&[[0, 0, 4, 4]], [4.0, 4.0, 1.0], 1);
        let fine = graph(
            &[[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4], [2, 2, 4, 4]],
            [4.0, 4.0, 1.0],
            2,
        );
        let mut sdf = AdaptiveSdf::from_graph(&coarse, |p| {
            Some(Sample::new(0.6 * p[0] + 0.8 * p[1] - 2.8))
        })
        .unwrap();
        for _ in 0..3 {
            sdf.redistance();
        }
        let mut sdf = sdf.remap(&fine).unwrap();
        sdf.redistance();
        let sdf = sdf.remap(&coarse).unwrap();
        for y in 0..=16 {
            for x in 0..=16 {
                let p = [x as f32 * 0.25, y as f32 * 0.25];
                assert!(
                    (sdf.sample(p).unwrap().phi - (0.6 * p[0] + 0.8 * p[1] - 2.8)).abs() < 2e-6
                );
            }
        }
    }
    #[test]
    fn coarse_gap_matches_the_resident_edge_seed_amplification() {
        let g = graph(
            &[[0, 0, 8, 8], [0, 8, 8, 16], [0, 16, 8, 24]],
            [8.0, 24.0, 1.0],
            1,
        );
        let mut sdf =
            AdaptiveSdf::from_graph(&g, |p| Some(Sample::new((p[1] - 8.01).min(18.5 - p[1]))))
                .unwrap();
        let low = sdf.sample([0.0, 8.0]).unwrap().phi;
        let high = sdf.sample([0.0, 16.0]).unwrap().phi;
        let crossing_distance = 8.0 * (-low) / (high - low);
        sdf.redistance();
        // Same 2.5-cell distance to the other surface, same edge-crossing seed
        // as WGSL: this arm deliberately reproduces the representation defect.
        assert!((sdf.sample([0.0, 8.0]).unwrap().phi + crossing_distance).abs() < 1e-5);
        assert!((sdf.sample([0.0, 16.0]).unwrap().phi - 2.5).abs() < 1e-5);
        assert!(crossing_distance > 3.0 * (-low));
    }
    #[test]
    fn clipped_domain_corners_and_sparse_growth_keep_finite_support() {
        let g = graph(&[[0, 0, 4, 3]], [8.0, 3.0, 1.0], 1);
        let sdf = AdaptiveSdf::from_graph(&g, |p| Some(Sample::new(p[0] - 1.0))).unwrap();
        assert_eq!(sdf.sample([4.0, 3.0]).unwrap().phi, 3.0);
        assert!(sdf.sample([5.0, 2.0]).is_none());
        let grown = graph(&[[0, 0, 4, 3], [4, 0, 8, 3]], [8.0, 3.0, 1.0], 2);
        let next = sdf.remap(&grown).unwrap();
        assert_eq!(next.sample([8.0, 3.0]).unwrap().phi, 7.0);
        assert!(next.raster().unwrap().iter().all(|p| p.is_finite()));
    }
}
