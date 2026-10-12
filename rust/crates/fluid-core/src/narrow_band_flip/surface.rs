//! The samples' surface: the bounded crossing search and nodal metric distance
//! (uniform-narrow-band-membership.wgsl.ts, -redistance.wgsl.ts) and the Eq. 4
//! sphere union (-surface.wgsl.ts), on the plane's h vertex lattice. Distances
//! are in h cells; phi is stored in metres.
use super::{layout::Layout, particles::Particle};
use crate::uniform_geometric::grid::Grid;

pub const SURFACE_RADIUS: f32 = 0.875;
/// A quarter-cell lattice has one tangential offset of h/4 at a vertex of the
/// plane: sqrt(0.875^2 - 0.25^2). Samples this deep rebuild a resting surface.
pub const SEED_DEPTH: f32 = 0.838_525_5;
/// Cells either side of a vertex searched for a crossing.
const REACH: i32 = 5;
const NONE: u32 = u32::MAX;

pub struct Surface {
    dims: [usize; 2],
    /// Cells whose corners straddle zero, as a mask and a list.
    crossing: Vec<bool>,
    crossings: Vec<u32>,
    /// Per vertex, the nearest crossing cell and the squared distance to its box.
    nearest: Vec<u32>,
    gap2: Vec<f32>,
    /// The nodal metric distance (bank A), negative in liquid.
    pub depth: Vec<f32>,
    /// Per vertex, the squared distance to the nearest sample within 2h; 4 with none.
    sample2: Vec<f32>,
    /// The 4h tiles a sample's sphere support reaches.
    reached: Vec<bool>,
}

fn bilinear(v: [f32; 4], q: [f32; 2]) -> [f32; 3] {
    let x0 = v[0] + (v[1] - v[0]) * q[0];
    let x1 = v[2] + (v[3] - v[2]) * q[0];
    let dx = (v[1] - v[0]) + ((v[3] - v[2]) - (v[1] - v[0])) * q[1];
    [dx, x1 - x0, x0 + (x1 - x0) * q[1]]
}

/// nbSurfaceDistance: from `p` to the zero set of one cell's interpolant.
/// Edge crossings are the fallback where the clamped Newton solve cannot
/// reach zero inside the cell.
fn cell_distance(phi: &[f32], dims: [usize; 2], h: f32, p: [f32; 2], cell: [i32; 2]) -> f32 {
    let nxv = dims[0] + 1;
    let j = cell[0] as usize + nxv * cell[1] as usize;
    let v = [phi[j], phi[j + 1], phi[j + nxv], phi[j + nxv + 1]].map(|value| value / h);
    let local = [p[0] - cell[0] as f32, p[1] - cell[1] as f32];
    let mut distance = 1e20_f32;
    for [k, other] in [[0, 1], [2, 3], [0, 2], [1, 3]] {
        if v[k] * v[other] > 0.0 || v[k] == v[other] {
            continue;
        }
        let t = (v[k] / (v[k] - v[other])).clamp(0.0, 1.0);
        let q: [f32; 2] = std::array::from_fn(|a| {
            let from = ((k >> a) & 1) as f32;
            from + (((other >> a) & 1) as f32 - from) * t
        });
        distance = distance.min((local[0] - q[0]).hypot(local[1] - q[1]));
    }
    let mut q = local.map(|x| x.clamp(0.0, 1.0));
    for _ in 0..6 {
        let [gx, gy, f] = bilinear(v, q);
        let g2 = gx * gx + gy * gy;
        if g2 < 1e-12 {
            break;
        }
        q = [
            (q[0] - f * gx / g2).clamp(0.0, 1.0),
            (q[1] - f * gy / g2).clamp(0.0, 1.0),
        ];
    }
    if bilinear(v, q)[2].abs() < 1e-4 {
        distance = distance.min((local[0] - q[0]).hypot(local[1] - q[1]));
    }
    distance
}

impl Surface {
    pub fn new(dims: [usize; 2]) -> Self {
        let cells = dims[0] * dims[1];
        let vertices = (dims[0] + 1) * (dims[1] + 1);
        Self {
            dims,
            crossing: vec![false; cells],
            crossings: Vec::new(),
            nearest: vec![NONE; vertices],
            gap2: vec![f32::INFINITY; vertices],
            depth: vec![6.0; vertices],
            sample2: vec![4.0; vertices],
            reached: vec![false; dims[0].div_ceil(4) * dims[1].div_ceil(4)],
        }
    }

    /// depthSeeds and the nearest search. Exact zero belongs to either
    /// incident cell; a flat zero plateau alone is not an interface, and
    /// neither is a domain or solid wall.
    fn search(&mut self, g: &Grid, phi: &[f32]) {
        let [nx, ny] = self.dims;
        let nxv = nx + 1;
        self.crossings.clear();
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * y;
                let j = x + nxv * y;
                let v = [phi[j], phi[j + 1], phi[j + nxv], phi[j + nxv + 1]];
                let low = v[0].min(v[1]).min(v[2].min(v[3]));
                let high = v[0].max(v[1]).max(v[2].max(v[3]));
                let hit = low <= 0.0 && high >= 0.0 && low < high && g.capacity[i] >= 0.5;
                self.crossing[i] = hit;
                if hit {
                    self.crossings.push(i as u32);
                }
            }
        }
        self.nearest.fill(NONE);
        self.gap2.fill(f32::INFINITY);
        for &c in &self.crossings {
            let cell = [(c as usize % nx) as i32, (c as usize / nx) as i32];
            for vy in (cell[1] - REACH).max(0)..=(cell[1] + 1 + REACH).min(ny as i32) {
                let dy = (cell[1] - vy).max(vy - cell[1] - 1).max(0);
                for vx in (cell[0] - REACH).max(0)..=(cell[0] + 1 + REACH).min(nx as i32) {
                    let dx = (cell[0] - vx).max(vx - cell[0] - 1).max(0);
                    let d2 = (dx * dx + dy * dy) as f32;
                    let v = vx as usize + nxv * vy as usize;
                    if d2 < self.gap2[v] {
                        self.gap2[v] = d2;
                        self.nearest[v] = c;
                    }
                }
            }
        }
    }

    /// bulkDepth: stale interior phi is no shallower than the nearest crossing
    /// cell's box, so the union's one-cell shrink cannot open the bulk.
    fn bulk(&self, v: usize, phi: f32) -> f32 {
        if phi >= 0.0 {
            phi
        } else if self.nearest[v] == NONE {
            phi.min(-8.0)
        } else {
            phi.min(-self.gap2[v].sqrt())
        }
    }

    /// Ferstl et al. 2016, Eq. 4: phi <- min(phi' + h, phi_p), from the
    /// advected level set into `out`. A tile no sample reaches keeps phi';
    /// deeper than two cells the shrunk level set wins whatever the samples are.
    pub fn reconstruct(
        &mut self,
        g: &Grid,
        advected: &[f32],
        samples: &[Particle],
        out: &mut [f32],
    ) {
        self.search(g, advected);
        let [nx, ny] = self.dims;
        let nxv = nx + 1;
        let tiles = [nx.div_ceil(4), ny.div_ceil(4)];
        self.sample2.fill(4.0);
        self.reached.fill(false);
        for sample in samples {
            let q = sample.position;
            let lo = q.map(|x| (x - 2.0).ceil().max(0.0) as usize);
            let hi = [
                ((q[0] + 2.0).floor() as usize).min(nx),
                ((q[1] + 2.0).floor() as usize).min(ny),
            ];
            for vy in lo[1]..=hi[1] {
                for vx in lo[0]..=hi[0] {
                    let d = [vx as f32 - q[0], vy as f32 - q[1]];
                    let nearest = &mut self.sample2[vx + nxv * vy];
                    *nearest = nearest.min(d[0] * d[0] + d[1] * d[1]);
                }
            }
            let lo = q.map(|x| ((x - 2.0) / 4.0).floor().max(0.0) as usize);
            let hi: [usize; 2] =
                std::array::from_fn(|a| (((q[a] + 2.0) / 4.0).floor() as usize).min(tiles[a] - 1));
            for ty in lo[1]..=hi[1] {
                for tx in lo[0]..=hi[0] {
                    self.reached[tx + tiles[0] * ty] = true;
                }
            }
        }
        let h = g.h[0].min(g.h[1]);
        let solids = g.capacity.iter().any(|&c| c <= 1e-5);
        for vy in 0..=ny {
            for vx in 0..=nx {
                let v = vx + nxv * vy;
                if solids && g.buried([vx as i32, vy as i32]) {
                    out[v] = advected[v];
                    continue;
                }
                let bulk = self.bulk(v, advected[v] / h);
                let tile = (vx / 4).min(tiles[0] - 1) + tiles[0] * (vy / 4).min(tiles[1] - 1);
                let value = if self.reached[tile] && bulk > -2.0 {
                    (bulk + 1.0).min(self.sample2[v].sqrt() - SURFACE_RADIUS)
                } else {
                    bulk
                };
                out[v] = value * h;
            }
        }
    }

    /// The nearest crossing cell's distance, and that of any crossing cell
    /// beside it whose box is nearer than the best so far.
    fn gap(&self, g: &Grid, h: f32, p: [f32; 2], c: u32) -> f32 {
        let nx = self.dims[0];
        let cell = [(c as usize % nx) as i32, (c as usize / nx) as i32];
        let mut best = cell_distance(&g.phi, self.dims, h, p, cell);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let q = [cell[0] + dx, cell[1] + dy];
                if (dx == 0 && dy == 0) || !g.index(q).is_some_and(|i| self.crossing[i]) {
                    continue;
                }
                let b: [f32; 2] = std::array::from_fn(|a| {
                    (q[a] as f32 - p[a]).max(p[a] - (q[a] + 1) as f32).max(0.0)
                });
                if b[0] * b[0] + b[1] * b[1] < best * best {
                    best = best.min(cell_distance(&g.phi, self.dims, h, p, q));
                }
            }
        }
        best
    }

    /// nbBuildDistance over every vertex: the metric distance to the surface
    /// phi holds, at least six cells where no crossing is within reach.
    pub fn measure(&mut self, g: &Grid) {
        self.search(g, &g.phi);
        let [nx, ny] = self.dims;
        let nxv = nx + 1;
        let h = g.h[0].min(g.h[1]);
        for vy in 0..=ny {
            for vx in 0..=nx {
                let v = vx + nxv * vy;
                let initial = g.phi[v];
                let mut value = initial / h;
                if initial.abs() > 1e-8 {
                    let mut distance = value.abs().max(6.0);
                    if self.nearest[v] != NONE {
                        let found = self.gap(g, h, [vx as f32, vy as f32], self.nearest[v]);
                        if found < 1e19 {
                            distance = found;
                        }
                    }
                    value = initial.signum() * distance.max(1e-8 / h);
                }
                self.depth[v] = value;
            }
        }
    }

    /// The crossing cell nearest the vertex nearest `p`, within reach.
    pub fn nearest_cell(&self, p: [f32; 2]) -> Option<[i32; 2]> {
        let v: [usize; 2] =
            std::array::from_fn(|a| (p[a].round().max(0.0) as usize).min(self.dims[a]));
        let c = self.nearest[v[0] + (self.dims[0] + 1) * v[1]];
        (c != NONE).then(|| {
            [
                (c as usize % self.dims[0]) as i32,
                (c as usize / self.dims[0]) as i32,
            ]
        })
    }

    /// nbRedistanceVertex: an h owner's vertex touching no crossing cell takes
    /// its metric distance. Interface vertices keep their reconstructed
    /// values: changing a magnitude there would move the surface. A 4h tile's
    /// corners keep theirs too, and its other vertices hang on them.
    pub fn redistance(&mut self, g: &mut Grid, layout: &Layout) {
        self.measure(g);
        let [nx, ny] = self.dims;
        let nxv = nx + 1;
        let h = g.h[0].min(g.h[1]);
        let solids = g.capacity.iter().any(|&c| c <= 1e-5);
        for vy in 0..=ny {
            for vx in 0..=nx {
                let vertex = [vx as i32, vy as i32];
                let interface = (0..4).any(|k| {
                    g.index([vertex[0] - (k & 1), vertex[1] - (k >> 1)])
                        .is_some_and(|i| self.crossing[i])
                });
                if interface || !layout.fine_vertex([vx, vy]) || (solids && g.buried(vertex)) {
                    continue;
                }
                g.phi[vx + nxv * vy] = self.depth[vx + nxv * vy] * h;
            }
        }
        layout.resolve_phi(&mut g.phi);
    }

    /// particleDepth: the metric field at a point, in cells.
    pub fn depth_at(&self, g: &Grid, p: [f32; 2]) -> f32 {
        g.scalar(&self.depth, p)
    }
}
