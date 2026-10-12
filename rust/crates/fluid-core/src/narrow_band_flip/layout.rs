//! The 3D method's mixed 4h/h layout in the plane (uniform-mixed-layout-
//! builder.ts, -ownership.ts, -remap.ts, -phi-resolve.ts): a tile of 4x4 h
//! cells is one 4h owner or sixteen h owners, ungraded. The fields stay dense.
//! A 4h tile's state is its four corner phi and its positive faces; its other
//! texels hold what the samplers read there.
use super::{
    particles::Particle,
    surface::{Surface, SURFACE_RADIUS},
};
use crate::uniform_geometric::grid::Grid;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Face {
    /// An h face: a fine tile's, or a 4h tile's patch beside a fine tile.
    Unit,
    /// One 4h face, held in the tile's cache.
    Coarse,
    /// Inside a 4h tile: interpolated, never state.
    Derived,
}

pub struct Layout {
    dims: [usize; 2],
    pub tiles: [usize; 2],
    /// Tiles this step resolves at h.
    pub fine: Vec<bool>,
    was: Vec<bool>,
    /// fineGridPadding: h cells a sample's support grows by before it joins.
    pub padding: f32,
    wet: Vec<bool>,
    coupled: Vec<bool>,
}

fn kind(fine: &[bool], tiles: [usize; 2], p: [i32; 2], a: usize) -> Face {
    let t = [p[0] as usize / 4, p[1] as usize / 4];
    if fine[t[0] + tiles[0] * t[1]] {
        return Face::Unit;
    }
    if p[a] % 4 != 3 {
        return Face::Derived;
    }
    let mut n = t;
    n[a] += 1;
    if n[a] < tiles[a] && fine[n[0] + tiles[0] * n[1]] {
        Face::Unit
    } else {
        Face::Coarse
    }
}

impl Layout {
    /// Every tile starts at h: the first step's decision coarsens.
    pub fn new(dims: [usize; 2], padding: f32) -> Self {
        let tiles = dims.map(|n| n / 4);
        let count = tiles[0] * tiles[1];
        Self {
            dims,
            tiles,
            fine: vec![true; count],
            was: vec![true; count],
            padding,
            wet: vec![false; count],
            coupled: vec![false; count],
        }
    }
    pub fn fine_tile(&self, t: [i32; 2]) -> bool {
        t[0] >= 0
            && t[1] >= 0
            && (t[0] as usize) < self.tiles[0]
            && (t[1] as usize) < self.tiles[1]
            && self.fine[t[0] as usize + self.tiles[0] * t[1] as usize]
    }
    /// The cell is an h owner. False outside the lattice.
    pub fn fine_cell(&self, p: [i32; 2]) -> bool {
        self.fine_tile([p[0].div_euclid(4), p[1].div_euclid(4)])
    }
    pub fn tile_of(&self, p: [i32; 2]) -> usize {
        p[0] as usize / 4 + self.tiles[0] * (p[1] as usize / 4)
    }
    /// The positive face of cell `p` along `a`.
    pub fn face(&self, p: [i32; 2], a: usize) -> Face {
        kind(&self.fine, self.tiles, p, a)
    }
    pub fn fine_count(&self) -> u32 {
        self.fine.iter().filter(|&&f| f).count() as u32
    }
    /// Tiles a vertex touches, per axis: one inside a tile, two on its edge.
    fn incident(&self, v: usize, a: usize) -> std::ops::Range<usize> {
        if v % 4 != 0 {
            v / 4..v / 4 + 1
        } else {
            (v / 4).saturating_sub(1)..(v / 4 + 1).min(self.tiles[a])
        }
    }
    /// The vertex is an h owner's.
    pub fn fine_vertex(&self, v: [usize; 2]) -> bool {
        self.incident(v[1], 1).any(|ty| {
            self.incident(v[0], 0)
                .any(|tx| self.fine[tx + self.tiles[0] * ty])
        })
    }
    /// umResolvePhi: a vertex off the 4h lattice with a 4h tile beside it
    /// hangs on that tile's corners (the lowest tile of several).
    pub fn resolve_phi(&self, phi: &mut [f32]) {
        let nxv = self.dims[0] + 1;
        for y in 0..=self.dims[1] {
            for x in 0..=self.dims[0] {
                if x % 4 == 0 && y % 4 == 0 {
                    continue;
                }
                let Some((tx, ty)) = self
                    .incident(y, 1)
                    .flat_map(|ty| self.incident(x, 0).map(move |tx| (tx, ty)))
                    .find(|&(tx, ty)| !self.fine[tx + self.tiles[0] * ty])
                else {
                    continue;
                };
                let corner = |dx: usize, dy: usize| phi[4 * (tx + dx) + nxv * 4 * (ty + dy)];
                let f = [(x - 4 * tx) as f32 / 4.0, (y - 4 * ty) as f32 / 4.0];
                let low = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * f[0];
                let high = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * f[0];
                phi[x + nxv * y] = low + (high - low) * f[1];
            }
        }
    }

    /// The step's layout, from its starting state: every tile holding a phi
    /// crossing of its owners' corners, a sample's swept support or source
    /// liquid, and the tiles a cut cell couples to them. This is the method's
    /// fixed particle band: every crossing tile is required.
    pub fn decide(
        &mut self,
        g: &Grid,
        samples: &[Particle],
        surface: &Surface,
        dt: f32,
        source: Option<&dyn Fn([f32; 2]) -> f32>,
    ) {
        self.was.clone_from(&self.fine);
        let [tw, th] = self.tiles;
        let nxv = self.dims[0] + 1;
        let epsilon = 1e-6 * g.h[0].min(g.h[1]);
        let crosses = |x: usize, y: usize, width: usize| {
            let j = x + nxv * y;
            let v = [
                g.phi[j],
                g.phi[j + width],
                g.phi[j + nxv * width],
                g.phi[j + nxv * width + width],
            ];
            let low = v[0].min(v[1]).min(v[2].min(v[3]));
            let high = v[0].max(v[1]).max(v[2].max(v[3]));
            low <= epsilon && high >= -epsilon && low < high
        };
        for ty in 0..th {
            for tx in 0..tw {
                let t = tx + tw * ty;
                self.fine[t] = if self.was[t] {
                    (0..16).any(|k| crosses(4 * tx + k % 4, 4 * ty + k / 4, 1))
                } else {
                    crosses(4 * tx, 4 * ty, 4)
                };
                self.wet[t] = (0..25).any(|k| g.phi[4 * tx + k % 5 + nxv * (4 * ty + k / 5)] < 0.0);
            }
        }
        if let Some(source) = source {
            let reach = 0.5 * (4.0 * g.h[0]).hypot(4.0 * g.h[1]) + 2.0 * g.h[0].max(g.h[1]);
            for ty in 0..th {
                for tx in 0..tw {
                    if source([4.0 * tx as f32 + 2.0, 4.0 * ty as f32 + 2.0]) <= reach {
                        self.fine[tx + tw * ty] = true;
                    }
                }
            }
        }
        // nbJoin: a sample within a cell of the surface joins the tiles its
        // support, or the crossing cell it sits under, sweeps this step.
        for p in samples {
            if p.ballistic || p.depth < -1.0 {
                continue;
            }
            let cell = (p.depth <= 0.0)
                .then(|| surface.nearest_cell(p.position))
                .flatten();
            let range: [[usize; 2]; 2] = std::array::from_fn(|a| {
                let (low, high) = match cell {
                    Some(c) => (c[a] as f32, c[a] as f32 + 1.0),
                    None => (
                        p.position[a] - SURFACE_RADIUS,
                        p.position[a] + SURFACE_RADIUS,
                    ),
                };
                let travel = dt * p.before[a] / g.h[a];
                let limit = self.tiles[a] as f32 - 1.0;
                [
                    ((low + travel.min(0.0) - self.padding) / 4.0)
                        .floor()
                        .clamp(0.0, limit) as usize,
                    ((high + travel.max(0.0) + self.padding - 1e-5) / 4.0)
                        .floor()
                        .clamp(0.0, limit) as usize,
                ]
            });
            for ty in range[1][0]..=range[1][1] {
                for tx in range[0][0]..=range[0][1] {
                    self.fine[tx + tw * ty] = true;
                }
            }
        }
        // Solid promotion: a cut cell couples the tiles within a cell of it.
        // A coupled tile that is wet or in the band brings its 3x3 with it.
        self.coupled.fill(false);
        let mut any = false;
        for (i, &capacity) in g.capacity.iter().enumerate() {
            if capacity >= 1.0 - 1e-5 {
                continue;
            }
            any = true;
            let p = g.point(i);
            for k in 0..9 {
                let q = [p[0] - 1 + k % 3, p[1] - 1 + k / 3];
                if g.index(q).is_some() {
                    let tile = self.tile_of(q);
                    self.coupled[tile] = true;
                }
            }
        }
        if !any {
            return;
        }
        for t in 0..self.coupled.len() {
            self.coupled[t] &= self.wet[t] || self.fine[t];
        }
        for ty in 0..th {
            for tx in 0..tw {
                if !self.coupled[tx + tw * ty] {
                    continue;
                }
                for y in ty.saturating_sub(1)..(ty + 2).min(th) {
                    for x in tx.saturating_sub(1)..(tx + 2).min(tw) {
                        self.fine[x + tw * y] = true;
                    }
                }
            }
        }
    }

    /// Cell `j` of the tile's layer under its positive face along `a`.
    pub fn top(&self, t: [usize; 2], a: usize, j: usize) -> [i32; 2] {
        let mut p = [4 * t[0] as i32, 4 * t[1] as i32];
        p[a] += 3;
        p[1 - a] += j as i32;
        p
    }

    /// The fields under the new layout. A tile turned fine takes each h face
    /// from its 4h faces, linear along the face normal and constant across; a
    /// 4h face newly split starts every patch on its value; a tile turned
    /// coarse keeps its corners and the mean of each face's patches.
    pub fn remap(&self, g: &mut Grid, cache: &mut [[f32; 2]]) {
        let [tw, th] = self.tiles;
        let nx = self.dims[0];
        let at = |p: [i32; 2]| p[0] as usize + nx * p[1] as usize;
        for ty in 0..th {
            for tx in 0..tw {
                let t = tx + tw * ty;
                for a in 0..2 {
                    let first = self.top([tx, ty], a, 0);
                    let split = kind(&self.was, self.tiles, first, a) == Face::Coarse
                        && self.face(first, a) == Face::Unit;
                    if split {
                        for j in 0..4 {
                            g.velocity[at(self.top([tx, ty], a, j))][a] = cache[t][a];
                        }
                    }
                    if self.was[t] || !self.fine[t] {
                        continue;
                    }
                    for j in 0..4 {
                        let high = self.top([tx, ty], a, j);
                        let mut below = high;
                        below[a] -= 4;
                        let low = if below[a] < 0 {
                            if a == 0 {
                                g.low_x[high[1] as usize]
                            } else {
                                g.low_y[high[0] as usize]
                            }
                        } else if kind(&self.was, self.tiles, below, a) == Face::Coarse {
                            cache[self.tile_of(below)][a]
                        } else {
                            g.velocity[at(below)][a]
                        };
                        let value = g.velocity[at(high)][a];
                        for i in 0..3 {
                            let mut p = high;
                            p[a] -= 3 - i;
                            g.velocity[at(p)][a] = low + (value - low) * (i + 1) as f32 / 4.0;
                        }
                    }
                }
            }
        }
        self.resolve(&mut g.velocity, cache);
        self.resolve_phi(&mut g.phi);
    }

    fn coarse(&self, cache: &[[f32; 2]], t: [i32; 2], a: usize) -> f32 {
        if t[0] < 0 || t[1] < 0 || t[0] >= self.tiles[0] as i32 || t[1] >= self.tiles[1] as i32 {
            return 0.0;
        }
        cache[t[0] as usize + self.tiles[0] * t[1] as usize][a]
    }
    /// The 4h field at an h site: bilinear over the tiles' positive faces.
    pub fn sample(&self, cache: &[[f32; 2]], a: usize, site: [f32; 2]) -> f32 {
        let q: [f32; 2] = std::array::from_fn(|b| {
            let (lower, offset) = if b == a { (-1.0, 1.0) } else { (0.0, 0.5) };
            (site[b] / 4.0 - offset).clamp(lower, self.tiles[b] as f32 - 1.0)
        });
        let base = q.map(|v| v.floor() as i32);
        let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
        let mut value = 0.0;
        for k in 0..4 {
            let w = (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] });
            if w > 0.0 {
                value += w * self.coarse(cache, [base[0] + (k & 1), base[1] + (k >> 1)], a);
            }
        }
        value
    }
    /// After a stage wrote the faces that are state (h texels, and the cache
    /// for a 4h face): the cache takes the mean of every split layer, and a
    /// 4h tile's other texels take the 4h field at their own sites.
    pub fn resolve(&self, field: &mut [[f32; 2]], cache: &mut [[f32; 2]]) {
        let [tw, th] = self.tiles;
        let nx = self.dims[0];
        let at = |p: [i32; 2]| p[0] as usize + nx * p[1] as usize;
        for ty in 0..th {
            for tx in 0..tw {
                for a in 0..2 {
                    if self.face(self.top([tx, ty], a, 0), a) == Face::Unit {
                        let sum: f32 = (0..4).map(|j| field[at(self.top([tx, ty], a, j))][a]).sum();
                        cache[tx + tw * ty][a] = 0.25 * sum;
                    }
                }
            }
        }
        for ty in 0..th {
            for tx in 0..tw {
                if self.fine[tx + tw * ty] {
                    continue;
                }
                for k in 0..16 {
                    let p = [(4 * tx + k % 4) as i32, (4 * ty + k / 4) as i32];
                    for a in 0..2 {
                        if self.face(p, a) == Face::Unit {
                            continue;
                        }
                        let mut site = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
                        site[a] += 0.5;
                        field[at(p)][a] = self.sample(cache, a, site);
                    }
                }
            }
        }
    }
}
