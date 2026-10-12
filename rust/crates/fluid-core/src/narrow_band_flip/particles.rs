//! Persistent surface samples (uniform-narrow-band-flip.ts): the forward
//! trace, the quadratic MAC transfer, the FLIP/PIC update, resampling and band
//! seeding. Positions are in h cells, velocities in m/s.
use super::{
    layout::{Face, Layout},
    surface::{Surface, SEED_DEPTH, SURFACE_RADIUS},
};
use crate::{
    types::ValidationError,
    uniform_geometric::{extension::Extension, grid::Grid},
};

pub const FLIP_RATIO: f32 = 0.95;
/// Quarter-cell lattice sites of one cell in the plane.
pub const SITES: usize = 4;
/// Deep overlap samples a cell keeps: twice its sites.
const CROWD: u32 = 8;
/// uniform-narrow-band-advection.wgsl.ts NARROW_BAND_TRACE_BEND and _LIMIT.
const TRACE_BEND: f32 = 0.125;
const TRACE_LIMIT: u32 = 128;

#[derive(Clone, Copy, Debug, Default)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    /// The grid velocity at the sample before forces and pressure.
    pub before: [f32; 2],
    /// Metric distance to the surface, negative inside.
    pub depth: f32,
    /// No liquid cell supports the sample: it falls under gravity alone.
    pub ballistic: bool,
}

/// classify's census of the samples against the surface they made.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Census {
    pub outside: u32,
    pub unsupported: u32,
    pub deep: u32,
}

pub struct Particles {
    pub list: Vec<Particle>,
    capacity: usize,
    counts: Vec<u32>,
    mass: Vec<[f32; 2]>,
    momentum: Vec<[f32; 2]>,
    /// The same sums for each tile's 4h faces.
    coarse_mass: Vec<[f32; 2]>,
    coarse_momentum: Vec<[f32; 2]>,
    /// New samples the last seeding could not store.
    pub clipped: u32,
}

fn open(g: &Grid, p: [f32; 2]) -> bool {
    g.index([p[0].floor() as i32, p[1].floor() as i32])
        .is_some_and(|i| g.capacity[i] >= 0.5)
}

/// gridSupported: one of the cells whose velocity the sample reads is liquid.
/// Negative phi at a sample is not enough: a sub-cell droplet may have no
/// resolved cell, so the grid cannot evolve its momentum.
fn supported(g: &Grid, liquid: &[bool], p: [f32; 2]) -> bool {
    let q = [p[0] - 0.5, p[1] - 0.5];
    let base = q.map(f32::floor);
    let f = [q[0] - base[0], q[1] - base[1]];
    (0..4).any(|k| {
        let bit = [k & 1, k >> 1];
        let w = (if bit[0] == 0 { 1.0 - f[0] } else { f[0] })
            * (if bit[1] == 0 { 1.0 - f[1] } else { f[1] });
        let cell = g.clamp_cell([base[0] as i32 + bit[0], base[1] as i32 + bit[1]]);
        w > 1e-6 && liquid[cell[0] as usize + g.dims[0] * cell[1] as usize]
    })
}

/// nbTraceSpan: one step from `p`, where `k1` is the field in cells a second.
/// The field one base substep ahead is the midpoint stage of Kutta's rule over
/// two substeps and the end stage of Shu and Osher's over one; two need the
/// step to stay within two cells and the field to be straight along it.
fn trace_span(
    g: &Grid,
    e: &Extension,
    p: [f32; 2],
    base: f32,
    k1: [f32; 2],
    most: u32,
) -> ([f32; 2], u32) {
    let sample = |q: [f32; 2]| {
        let v = e.sample(g.clamp(q));
        [v[0] / g.h[0], v[1] / g.h[1]]
    };
    let k2 = sample([p[0] + base * k1[0], p[1] + base * k1[1]]);
    let turn = (k2[0] - k1[0]).abs().max((k2[1] - k1[1]).abs());
    let two = most >= 2
        && 2.0 * base.abs() * k1[0].abs().max(k1[1].abs()) <= 2.001
        && 2.0 * base.abs() * turn <= TRACE_BEND;
    let k3 = sample(std::array::from_fn(|a| {
        if two {
            p[a] + 2.0 * base * (2.0 * k2[a] - k1[a])
        } else {
            p[a] + 0.25 * base * (k1[a] + k2[a])
        }
    }));
    if two {
        (
            std::array::from_fn(|a| p[a] + base * (k1[a] + 4.0 * k2[a] + k3[a]) / 3.0),
            2,
        )
    } else {
        (
            std::array::from_fn(|a| p[a] + base * (k1[a] + k2[a] + 4.0 * k3[a]) / 6.0),
            1,
        )
    }
}

fn weight(x: f32) -> f32 {
    let a = x.abs();
    if a < 0.5 {
        0.75 - a * a
    } else {
        let b = (1.5 - a).max(0.0);
        0.5 * b * b
    }
}

/// seedShell: fit an outer lattice site to the sphere reconstruction's nodal
/// interface, along the depth gradient. The tangential quarter-cell offset
/// stays, so a resting plane is preserved without any volume correction.
fn shell(g: &Grid, surface: &Surface, lattice: [f32; 2]) -> ([f32; 2], f32) {
    let mut q = lattice;
    let mut d = surface.depth_at(g, q);
    for _ in 0..4 {
        let rise = d + SEED_DEPTH;
        if rise <= 1e-3 {
            break;
        }
        let gradient: [f32; 2] = std::array::from_fn(|a| {
            let mut lo = q;
            let mut hi = q;
            lo[a] -= 0.5;
            hi[a] += 0.5;
            surface.depth_at(g, hi) - surface.depth_at(g, lo)
        });
        let g2 = gradient[0] * gradient[0] + gradient[1] * gradient[1];
        if g2 < 1e-8 {
            break;
        }
        let step = rise.min(0.5) / g2.sqrt();
        let next: [f32; 2] = std::array::from_fn(|a| {
            (q[a] - step * gradient[a]).clamp(0.01, g.dims[a] as f32 - 0.01)
        });
        let depth = surface.depth_at(g, next);
        if depth >= d {
            break;
        }
        q = next;
        d = depth;
    }
    (q, d)
}

/// seedSite. Paper Section 3.3: only the inner [-R, -h] band is replenished;
/// the outer layer keeps its transported samples. Section 3.2: liquid made
/// from a description gets its whole band, at initialization and where a
/// source has just made it. `source` is the sources' own distance in cells.
fn site(
    g: &Grid,
    surface: &Surface,
    cell: [usize; 2],
    k: usize,
    initial: bool,
    source: &dyn Fn([f32; 2]) -> f32,
) -> Option<([f32; 2], f32)> {
    let q = [
        cell[0] as f32 + 0.25 + 0.5 * (k & 1) as f32,
        cell[1] as f32 + 0.25 + 0.5 * (k >> 1) as f32,
    ];
    let distance = surface.depth_at(g, q);
    if distance > 0.0 || distance < -4.0 {
        return None;
    }
    if !initial && distance > -1.0 && source(q) >= 0.0 {
        return None;
    }
    if distance <= -SURFACE_RADIUS {
        return Some((q, distance));
    }
    let (p, depth) = shell(g, surface, q);
    if depth > 0.0 || !open(g, p) {
        return None;
    }
    Some((p, surface.depth_at(g, p)))
}

impl Particles {
    /// The whole pool is paid for up front, as the GPU method's is.
    pub fn new(dims: [usize; 2]) -> Self {
        let cells = dims[0] * dims[1];
        let capacity = (2 * SITES * cells).min(cells.max(2_097_152));
        Self {
            list: Vec::with_capacity(capacity),
            capacity,
            counts: vec![0; cells],
            mass: vec![[0.0; 2]; cells],
            momentum: vec![[0.0; 2]; cells],
            coarse_mass: vec![[0.0; 2]; cells / 16],
            coarse_momentum: vec![[0.0; 2]; cells / 16],
            clipped: 0,
        }
    }

    /// advectParticles: solid collision, rather than grid ownership, limits a
    /// sample's motion. A supported sample follows the extended grid velocity;
    /// a ballistic one drifts at its own.
    pub fn advect(
        &mut self,
        g: &Grid,
        e: &Extension,
        liquid: &[bool],
        dt: f32,
    ) -> Result<(), ValidationError> {
        let limit = g.dims.map(|n| n as f32);
        let solid_free = g.capacity.iter().all(|&c| c >= 0.5);
        let cells = |v: [f32; 2]| [v[0] / g.h[0], v[1] / g.h[1]];
        let mut kept = 0;
        for i in 0..self.list.len() {
            let mut p = self.list[i];
            let mut q = p.position;
            if !open(g, q) {
                continue;
            }
            p.ballistic = p.ballistic || !supported(g, liquid, q);
            let mut speed = cells(if p.ballistic { p.velocity } else { e.sample(q) });
            let steps = ((dt * speed[0].abs().max(speed[1].abs())).ceil() as u32).max(1);
            if steps > TRACE_LIMIT {
                return Err(ValidationError(format!(
                    "narrow-band FLIP sample trajectory exceeded {TRACE_LIMIT} cells in one step"
                )));
            }
            let base = dt / steps as f32;
            let mut s = 0;
            while s < steps {
                let mut units = 1;
                let end = if p.ballistic {
                    let v = cells(p.velocity);
                    [q[0] + base * v[0], q[1] + base * v[1]]
                } else {
                    // The first subdivision starts where the speed was sampled.
                    if s > 0 {
                        speed = cells(e.sample(q));
                    }
                    let (end, span) = trace_span(g, e, q, base, speed, steps - s);
                    units = span;
                    end
                };
                if g.open_top && end[1] >= limit[1] {
                    q = end;
                    break;
                }
                let endpoint = [
                    end[0].clamp(0.01, limit[0] - 0.01),
                    end[1].clamp(0.01, limit[1] - 0.01),
                ];
                if solid_free {
                    q = endpoint;
                } else {
                    let travel = [endpoint[0] - q[0], endpoint[1] - q[1]];
                    let walk = ((2.0 * travel[0].abs().max(travel[1].abs())).ceil() as u32).max(1);
                    if walk > 256 {
                        return Err(ValidationError(
                            "narrow-band FLIP sample collision walk exceeded 256 half cells".into(),
                        ));
                    }
                    // A voxel face stops a sample as a container wall does: the
                    // crossing axis ends at the face and loses its velocity.
                    let stride = travel.map(|t| t / walk as f32);
                    for _ in 0..walk {
                        for a in 0..2 {
                            let mut point = q;
                            point[a] += stride[a];
                            if !open(g, point) {
                                let cell = q[a].floor();
                                point[a] = cell + if stride[a] > 0.0 { 0.99 } else { 0.01 };
                                p.velocity[a] = 0.0;
                            }
                            q = point;
                        }
                    }
                }
                for a in 0..2 {
                    if endpoint[a] != end[a] {
                        p.velocity[a] = 0.0;
                    }
                }
                s += units;
            }
            if !(0..2).all(|a| q[a] >= 0.0 && q[a] < limit[a]) {
                continue;
            }
            p.position = q;
            self.list[kept] = p;
            kept += 1;
        }
        self.list.truncate(kept);
        Ok(())
    }

    /// classify: a sample is coupled to the grid this step if the surface it
    /// has just rebuilt gives it a liquid cell.
    pub fn classify(&mut self, g: &Grid, liquid: &[bool], surface: &Surface) -> Census {
        let h = g.h[0].min(g.h[1]);
        let mut census = Census::default();
        for p in &mut self.list {
            let supported = supported(g, liquid, p.position);
            census.outside += (g.phi_at(p.position) / h > 0.5) as u32;
            census.unsupported += !supported as u32;
            census.deep += (surface.depth_at(g, p.position) < -4.0) as u32;
            p.ballistic = !supported;
        }
        census
    }

    /// transferFace: within two cells of the surface a positive face takes the
    /// coupled samples' velocity through the quadratic kernel. Deeper faces,
    /// faces no sample weighs on and the walls keep the grid's own. A 4h face
    /// point-samples the same h-wide kernel at its centre, into `cache`.
    pub fn transfer(
        &mut self,
        g: &mut Grid,
        surface: &Surface,
        layout: &Layout,
        cache: &mut [[f32; 2]],
    ) {
        let dims = g.dims.map(|n| n as i32);
        let tiles = layout.tiles.map(|n| n as i32);
        self.mass.fill([0.0; 2]);
        self.momentum.fill([0.0; 2]);
        self.coarse_mass.fill([0.0; 2]);
        self.coarse_momentum.fill([0.0; 2]);
        for p in self.list.iter().filter(|p| !p.ballistic) {
            for a in 0..2 {
                let b = 1 - a;
                // Face nodes sit on integers along their axis, half way across.
                let along = (p.position[a] + 0.5).floor() as i32;
                let across = p.position[b].floor() as i32;
                for da in -1..=1 {
                    let i = along + da - 1;
                    if i < 0 || i + 1 >= dims[a] {
                        continue;
                    }
                    let wa = weight(p.position[a] - (i + 1) as f32);
                    for db in -1..=1 {
                        let j = across + db;
                        if j < 0 || j >= dims[b] {
                            continue;
                        }
                        let w = wa * weight(p.position[b] - (j as f32 + 0.5));
                        let cell = if a == 0 { [i, j] } else { [j, i] };
                        let at = cell[0] as usize + g.dims[0] * cell[1] as usize;
                        self.mass[at][a] += w;
                        self.momentum[at][a] += w * p.velocity[a];
                    }
                }
                // The one 4h face whose centre the kernel can reach.
                let k = (p.position[a] / 4.0).round() as i32 - 1;
                let j = (p.position[b] / 4.0).floor() as i32;
                if k < 0 || k + 1 >= tiles[a] || j < 0 || j >= tiles[b] {
                    continue;
                }
                let w = weight(p.position[a] - (4 * k + 4) as f32)
                    * weight(p.position[b] - (4 * j + 2) as f32);
                if w > 0.0 {
                    let tile = if a == 0 { [k, j] } else { [j, k] };
                    let at = tile[0] as usize + layout.tiles[0] * tile[1] as usize;
                    self.coarse_mass[at][a] += w;
                    self.coarse_momentum[at][a] += w * p.velocity[a];
                }
            }
        }
        for at in 0..self.mass.len() {
            let cell = g.point(at);
            for a in 0..2 {
                let mass = self.mass[at][a];
                if mass < 1e-5 || cell[a] + 1 >= dims[a] || layout.face(cell, a) != Face::Unit {
                    continue;
                }
                let mut q = [cell[0] as f32 + 0.5, cell[1] as f32 + 0.5];
                q[a] += 0.5;
                let depth = surface.depth_at(g, q);
                if (-2.0..=1.5).contains(&depth) {
                    g.velocity[at][a] = self.momentum[at][a] / mass;
                }
            }
        }
        for at in 0..self.coarse_mass.len() {
            let tile = [(at % layout.tiles[0]) as i32, (at / layout.tiles[0]) as i32];
            for a in 0..2 {
                let mass = self.coarse_mass[at][a];
                let mut first = [4 * tile[0], 4 * tile[1]];
                first[a] += 3;
                if mass < 1e-5 || tile[a] + 1 >= tiles[a] || layout.face(first, a) != Face::Coarse {
                    continue;
                }
                let mut q = [4.0 * tile[0] as f32 + 2.0, 4.0 * tile[1] as f32 + 2.0];
                q[a] += 2.0;
                let depth = surface.depth_at(g, q);
                if (-2.0..=1.5).contains(&depth) {
                    cache[at][a] = self.coarse_momentum[at][a] / mass;
                }
            }
        }
    }

    /// snapshot: the extended grid velocity each coupled sample sees before
    /// forces and pressure.
    pub fn snapshot(&mut self, e: &Extension) {
        for p in &mut self.list {
            p.before = if p.ballistic {
                [0.0; 2]
            } else {
                e.sample(p.position)
            };
        }
    }

    /// update: 95% FLIP, 5% PIC from the projected, extended velocity. A
    /// ballistic sample takes the gravity kick instead and stops at a wall.
    pub fn update(
        &mut self,
        g: &Grid,
        e: &Extension,
        surface: &Surface,
        gravity: f32,
        dt: f32,
    ) -> Result<(), ValidationError> {
        let limit = g.dims.map(|n| n as f32);
        for p in &mut self.list {
            let mut pic = [0.0; 2];
            let v = if p.ballistic {
                let mut v = [p.velocity[0], p.velocity[1] + gravity * dt];
                for a in 0..2 {
                    let q = p.position[a];
                    let ambient = a == 1 && g.open_top;
                    if (q <= 0.011 && v[a] < 0.0)
                        || (q >= limit[a] - 0.011 && v[a] > 0.0 && !ambient)
                    {
                        v[a] = 0.0;
                    }
                }
                v
            } else {
                pic = e.sample(p.position);
                std::array::from_fn(|a| {
                    let flip = p.velocity[a] + pic[a] - p.before[a];
                    pic[a] + (flip - pic[a]) * FLIP_RATIO
                })
            };
            if !(v[0].abs() < 1e10 && v[1].abs() < 1e10) {
                return Err(ValidationError(
                    "narrow-band FLIP sample has a nonfinite velocity".into(),
                ));
            }
            p.velocity = v;
            p.depth = surface.depth_at(g, p.position);
            p.before = pic;
        }
        Ok(())
    }

    /// nbKept: retire the deep interior and the crowded overlap. Surface
    /// samples survive even when compressed: a thin structure cannot outlive
    /// an arbitrary per-cell cap.
    pub fn resample(&mut self, g: &Grid) {
        let nx = g.dims[0];
        let cell = |p: &Particle| p.position[0] as usize + nx * p.position[1] as usize;
        let counts = &mut self.counts;
        counts.fill(0);
        self.list.retain(|p| {
            let rank = &mut counts[cell(p)];
            *rank += 1;
            p.depth >= -4.0 && (p.depth >= -1.0 || *rank <= CROWD)
        });
        counts.fill(0);
        for p in &self.list {
            counts[cell(p)] += 1;
        }
    }

    /// seedCells: top each band cell up to its lattice sites. Source cells go
    /// first, so a full pool cannot punch holes in a continuous jet. `source`
    /// is the sources' own distance in cells, with `sourced` false for none.
    pub fn seed(
        &mut self,
        g: &Grid,
        e: &Extension,
        surface: &Surface,
        initial: bool,
        sourced: bool,
        source: &dyn Fn([f32; 2]) -> f32,
    ) {
        self.clipped = 0;
        let passes: &[Option<bool>] = if sourced {
            &[Some(true), Some(false)]
        } else {
            &[None]
        };
        for pass in passes {
            for y in 0..g.dims[1] {
                for x in 0..g.dims[0] {
                    let at = x + g.dims[0] * y;
                    let count = self.counts[at] as usize;
                    if count >= SITES || g.capacity[at] < 0.5 {
                        continue;
                    }
                    let centre = [x as f32 + 0.5, y as f32 + 0.5];
                    let depth = surface.depth_at(g, centre);
                    if depth > 0.9 || depth < -4.5 {
                        continue;
                    }
                    if pass.is_some_and(|wanted| (source(centre) < 1.0) != wanted) {
                        continue;
                    }
                    for k in count..SITES {
                        let Some((position, depth)) = site(g, surface, [x, y], k, initial, source)
                        else {
                            continue;
                        };
                        if self.list.len() >= self.capacity {
                            self.clipped += 1;
                            continue;
                        }
                        let velocity = e.sample(position);
                        self.list.push(Particle {
                            position,
                            velocity,
                            before: velocity,
                            depth,
                            ballistic: false,
                        });
                    }
                }
            }
        }
    }

    /// Six floats a sample: position in cells, velocity in m/s, depth, ballistic.
    pub fn publish(&self, out: &mut Vec<f32>) {
        out.clear();
        for p in &self.list {
            out.extend([
                p.position[0],
                p.position[1],
                p.velocity[0],
                p.velocity[1],
                p.depth,
                p.ballistic as u8 as f32,
            ]);
        }
    }
}
