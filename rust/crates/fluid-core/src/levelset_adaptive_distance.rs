//! Auxiliary 2-D distance/ownership field on accepted adaptive cells.
//!
//! Contour segments seed exact local distances. Dijkstra extension supplies an
//! obstacle-aware graph distance (not an exact Euclidean SDF far from the
//! contour), a closest-point witness and component ownership. The dense public
//! phi remains the surface authority; no raster distance workspace is created.
use crate::presentation::RdfSurface;
use crate::{Fields, Graph};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub(crate) const NONE: usize = usize::MAX;
pub(crate) const AMBIGUOUS: usize = usize::MAX - 1;
pub(crate) const RETURN_REACH: f64 = 8.0;

#[derive(Clone, Copy, Debug)]
struct Visit {
    distance: f64,
    cell: usize,
    component: usize,
}
impl PartialEq for Visit {
    fn eq(&self, b: &Self) -> bool {
        self.distance == b.distance && self.cell == b.cell && self.component == b.component
    }
}
impl Eq for Visit {}
impl Ord for Visit {
    fn cmp(&self, b: &Self) -> Ordering {
        b.distance
            .total_cmp(&self.distance)
            .then_with(|| b.cell.cmp(&self.cell))
            .then_with(|| b.component.cmp(&self.component))
    }
}
impl PartialOrd for Visit {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}

pub(crate) struct AdaptiveDistance {
    /// Signed graph distance in finest-cell units; negative in phi-liquid.
    pub signed_distance: Vec<f64>,
    pub distance: Vec<f64>,
    pub component: Vec<usize>,
    pub closest: Vec<[f64; 2]>,
    pub edges: Vec<Vec<(usize, f64)>>,
}

fn open_line(a: [f32; 3], b: [f32; 3], capacity: &[f32], nx: usize, ny: usize) -> bool {
    // Check every interval cut by a finest voxel boundary, including partial
    // solids conservatively. Row apertures alone cannot locate a cut opening.
    let mut cuts = vec![0.0_f64, 1.0];
    for axis in 0..2 {
        let delta = (b[axis] - a[axis]) as f64;
        if delta.abs() < 1e-12 {
            continue;
        }
        for k in a[axis].min(b[axis]).floor() as i32..=a[axis].max(b[axis]).ceil() as i32 {
            let t = (k as f64 - a[axis] as f64) / delta;
            if t > 0.0 && t < 1.0 {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.windows(2).all(|ts| {
        let t = 0.5 * (ts[0] + ts[1]);
        let x = (a[0] as f64 + t * (b[0] - a[0]) as f64).floor() as usize;
        let y = (a[1] as f64 + t * (b[1] - a[1]) as f64).floor() as usize;
        x < nx && y < ny && capacity[x + nx * y] >= 1.0 - 1e-6
    })
}

impl AdaptiveDistance {
    pub fn build(
        graph: &Graph,
        fields: &Fields,
        surface: &RdfSurface,
        fine_capacity: &[f32],
        owner: &[usize],
        seed_component: &[usize],
        components: usize,
        target: &[f64],
    ) -> Self {
        let n = graph.cells.len();
        let [nx, ny] = surface.dimensions.map(|v| v as usize);
        let mut result = Self {
            signed_distance: vec![f64::INFINITY; n],
            distance: vec![f64::INFINITY; n],
            component: vec![NONE; n],
            closest: vec![[0.0; 2]; n],
            edges: vec![Vec::new(); n],
        };
        for row in &graph.rows {
            if row.open_fraction < 1.0 - 1e-6 || row.measure <= 0.0 {
                continue;
            }
            for a in &row.terms {
                for b in &row.terms {
                    let (i, j) = (a.cell_id as usize, b.cell_id as usize);
                    if i >= j
                        || a.coefficient * b.coefficient >= 0.0
                        || fields.capacity[i] < 1.0 - 1e-6
                        || fields.capacity[j] < 1.0 - 1e-6
                    {
                        continue;
                    }
                    let (p, q) = (graph.cells[i].center, graph.cells[j].center);
                    if !open_line(p, q, fine_capacity, nx, ny) {
                        continue;
                    }
                    let length = ((p[0] - q[0]) as f64).hypot((p[1] - q[1]) as f64);
                    result.edges[i].push((j, length));
                    result.edges[j].push((i, length));
                }
            }
        }
        for edges in &mut result.edges {
            edges.sort_by_key(|&(j, _)| j);
            edges.dedup_by_key(|e| e.0);
        }
        let mut heap = BinaryHeap::new();
        for segment in surface.segments_fine.chunks_exact(4) {
            let a = [segment[0] as f64, segment[1] as f64];
            let d = [
                (segment[2] - segment[0]) as f64,
                (segment[3] - segment[1]) as f64,
            ];
            let norm = d[0] * d[0] + d[1] * d[1];
            if norm <= 1e-20 {
                continue;
            }
            // A published segment lies inside one finest triangle. Include
            // incident cells on either side of exact lattice-aligned contours.
            let x0 = (a[0].min(segment[2] as f64) - 1e-5).floor().max(0.0) as usize;
            let y0 = (a[1].min(segment[3] as f64) - 1e-5).floor().max(0.0) as usize;
            let x1 = (a[0].max(segment[2] as f64) + 1e-5)
                .floor()
                .min((nx - 1) as f64) as usize;
            let y1 = (a[1].max(segment[3] as f64) + 1e-5)
                .floor()
                .min((ny - 1) as f64) as usize;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = owner[x + nx * y];
                    if i == NONE
                        || seed_component[i] >= components
                        || fields.capacity[i] < 1.0 - 1e-6
                        || fine_capacity[x + nx * y] < 1.0 - 1e-6
                    {
                        continue;
                    }
                    let p = graph.cells[i].center;
                    let t = (((p[0] as f64 - a[0]) * d[0] + (p[1] as f64 - a[1]) * d[1]) / norm)
                        .clamp(0.0, 1.0);
                    let closest = [a[0] + t * d[0], a[1] + t * d[1]];
                    let distance = (p[0] as f64 - closest[0]).hypot(p[1] as f64 - closest[1]);
                    if distance < result.distance[i] {
                        result.distance[i] = distance;
                        result.closest[i] = closest;
                        result.component[i] = seed_component[i];
                        heap.push(Visit {
                            distance,
                            cell: i,
                            component: seed_component[i],
                        });
                    }
                }
            }
        }
        while let Some(v) = heap.pop() {
            if v.distance > result.distance[v.cell] + 1e-9
                || v.component != result.component[v.cell]
            {
                continue;
            }
            for &(j, length) in &result.edges[v.cell] {
                // Never enter a different represented liquid component, or an
                // unresolved cell holding several components.
                if target[j] > 0.0 && seed_component[j] != v.component {
                    continue;
                }
                let next = v.distance + length;
                if next > RETURN_REACH {
                    continue;
                }
                if next < result.distance[j] - 1e-9 {
                    result.distance[j] = next;
                    result.component[j] = v.component;
                    result.closest[j] = result.closest[v.cell];
                    heap.push(Visit {
                        distance: next,
                        cell: j,
                        component: v.component,
                    });
                } else if (next - result.distance[j]).abs() <= 1e-9
                    && result.component[j] != v.component
                    && result.component[j] != AMBIGUOUS
                {
                    result.component[j] = AMBIGUOUS;
                    heap.push(Visit {
                        distance: next,
                        cell: j,
                        component: AMBIGUOUS,
                    });
                }
            }
        }
        for (i, cell) in graph.cells.iter().enumerate() {
            let phi = crate::levelset_redistance::sample_scalar(
                surface,
                [cell.center[0], cell.center[1]],
            )
            .unwrap_or(1.0);
            result.signed_distance[i] = if phi < 0.0 {
                -result.distance[i]
            } else {
                result.distance[i]
            };
        }
        result
    }

    /// Bounded physical paths certify every relocation, including around walls.
    pub fn paths(&self, start: usize, reach: f64) -> Vec<f64> {
        let mut distance = vec![f64::INFINITY; self.edges.len()];
        distance[start] = 0.0;
        let component = self.component[start];
        let mut heap = BinaryHeap::from([Visit {
            distance: 0.0,
            cell: start,
            component,
        }]);
        while let Some(v) = heap.pop() {
            if v.distance > distance[v.cell] {
                continue;
            }
            for &(j, length) in &self.edges[v.cell] {
                let next = v.distance + length;
                if self.component[j] == component && next <= reach && next < distance[j] {
                    distance[j] = next;
                    heap.push(Visit {
                        distance: next,
                        cell: j,
                        component,
                    });
                }
            }
        }
        distance
    }
}
