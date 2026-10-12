//! The h pressure band of the 3D method's two-stage solve
//! (uniform-pressure-band.ts), in the plane. Every h tile holding a liquid
//! row is a band tile, and each of its sixteen cells a row. The rows solve the
//! full h pressure from the forced velocity, ghost fluid to air, with the
//! projected 4h flux on faces into liquid no band row resolves. They start
//! from the 4h pressure and iterate the correction to it: V-cycles
//! h -> 2h (2x2 aggregates per tile) -> 4h (one aggregate per tile).
use super::layout::Layout;
use crate::uniform_geometric::{grid::Grid, world::World as Fluid};

const GHOST: u8 = 0;
const BAND: u8 = 1;
const WALL: u8 = 2;
const OPEN: u8 = 3;
const NEUMANN: u8 = 4;
const CLOSED: u8 = 5;
const LIQUID: u8 = 1;
/// A row inside a solid: the CM11a p_min = 0 row.
const INSIDE: u8 = 2;
/// A row of a cell with no capacity: its phi continues its open neighbours.
const SHUT: u8 = 4;
const NONE: u32 = u32::MAX;
/// UNIFORM_MIXED_THETA_MIN.
const THETA_MIN: f32 = 1e-3;
/// UNIFORM_PRESSURE_BAND_SCHEDULE.
const CYCLES: usize = 4;
const FINE_SWEEPS: usize = 3;
const MIDDLE_SWEEPS: usize = 2;
const COARSE_SWEEPS: usize = 8;
const COARSE_RELAXATION: f32 = 1.4;
const COARSE_SCALE: f32 = 0.6;

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BandReceipt {
    pub tiles: u32,
    pub rows: u32,
    pub cycles: u32,
    /// The largest liquid row residual, as a divergence in 1/s.
    pub residual: f32,
    pub converged: bool,
}

#[derive(Clone, Copy, Default)]
struct Aggregate {
    correction: f32,
    residual: f32,
    diagonal: f32,
    couple: [f32; 4],
    near: [u32; 4],
}

#[derive(Default)]
pub struct Band {
    dims: [usize; 2],
    tiles: [usize; 2],
    slots: Vec<u32>,
    /// Slot + 1 of a band tile, 0 outside the band.
    slot_of: Vec<u32>,
    /// The simulation's pressure phi of each h cell of an h tile.
    phi: Vec<f32>,
    /// Per tile, the 4h solve: its phi and pressure, and whether the owner
    /// is an open liquid one or an open air one.
    root_phi: Vec<f32>,
    root_pressure: Vec<f32>,
    root_liquid: Vec<bool>,
    root_air: Vec<bool>,
    rhs: Vec<f32>,
    diagonal: Vec<f32>,
    /// The start pressure of a row: the iterate is the correction to it.
    base: Vec<f32>,
    solve: Vec<f32>,
    flags: Vec<u8>,
    kinds: Vec<[u8; 4]>,
    coefficient: Vec<[f32; 4]>,
    /// The forced velocity of each face; a Neumann face's kept 4h velocity.
    forced: Vec<[f32; 4]>,
    volume: Vec<[f32; 4]>,
    /// The wall halo of a row's face, less the row's start.
    halo: Vec<[f32; 4]>,
    near: Vec<[u32; 4]>,
    middle: Vec<Aggregate>,
    coarse: Vec<Aggregate>,
    /// The pressure of every h cell: the band's rows, the 4h solve elsewhere.
    pub pressure: Vec<f32>,
    pub receipt: BandReceipt,
}

fn sign(f: usize) -> i32 {
    if f & 1 == 1 {
        1
    } else {
        -1
    }
}
// umPressureSurfaceTheta.
fn theta(liquid: f32, air: f32) -> f32 {
    (-liquid / (air - liquid).max(1e-30)).clamp(THETA_MIN, 1.0 / THETA_MIN)
}
fn pair_theta(a: f32, wet_a: bool, b: f32, wet_b: bool) -> f32 {
    if wet_a == wet_b {
        1.0
    } else if wet_a {
        theta(a, b)
    } else {
        theta(b, a)
    }
}
fn face(field: &[[f32; 2]], low: [&[f32]; 2], g: &Grid, p: [i32; 2], a: usize) -> f32 {
    if p[a] == -1 {
        let b = p[1 - a];
        return if b >= 0 && (b as usize) < g.dims[1 - a] {
            low[a][b as usize]
        } else {
            0.0
        };
    }
    g.index(p).map_or(0.0, |i| field[i][a])
}

impl Band {
    pub fn new(dims: [usize; 2]) -> Self {
        let tiles = dims.map(|n| n / 4);
        let count = tiles[0] * tiles[1];
        Self {
            dims,
            tiles,
            slot_of: vec![0; count],
            phi: vec![0.0; dims[0] * dims[1]],
            root_phi: vec![0.0; count],
            root_pressure: vec![0.0; count],
            root_liquid: vec![false; count],
            root_air: vec![false; count],
            pressure: vec![0.0; dims[0] * dims[1]],
            ..Default::default()
        }
    }
    fn origin(&self, s: usize) -> [i32; 2] {
        let t = self.slots[s] as usize;
        [
            4 * (t % self.tiles[0]) as i32,
            4 * (t / self.tiles[0]) as i32,
        ]
    }
    /// The band row of an h cell.
    fn cell(&self, q: [i32; 2]) -> Option<usize> {
        if q[0] < 0 || q[1] < 0 || q[0] >= self.dims[0] as i32 || q[1] >= self.dims[1] as i32 {
            return None;
        }
        let q = q.map(|v| v as usize);
        let slot = self.slot_of[q[0] / 4 + self.tiles[0] * (q[1] / 4)];
        (slot != 0).then(|| 16 * (slot as usize - 1) + q[0] % 4 + 4 * (q[1] % 4))
    }
    /// The cell's tile has pressure rows.
    pub fn rows_at(&self, p: [i32; 2]) -> bool {
        self.cell(p).is_some()
    }
    fn wet(&self, phi: f32, width: f32, h: f32) -> bool {
        phi < -THETA_MIN * width * h
    }
    /// bNeighbour: (phi, liquid, Neumann) of the cell across a face. Inside a
    /// 4h owner the neighbour is liquid no band row resolves where that owner
    /// is liquid or the h centre there is; an air h centre in an air owner is
    /// the free surface.
    fn neighbour(&self, g: &Grid, layout: &Layout, q: [i32; 2], h: f32) -> (f32, bool, bool) {
        if layout.fine_cell(q) {
            let phi = self.phi[g.index(q).unwrap()];
            let wet = self.wet(phi, 1.0, h);
            return (phi, wet, wet);
        }
        let owner = self.root_phi[layout.tile_of(q)];
        if self.wet(owner, 4.0, h) {
            return (owner, true, true);
        }
        let phi = g.phi_centre(q);
        let wet = self.wet(phi, 1.0, h);
        (phi, wet, wet)
    }
    // bCoarseKind: a 4h owner as (pressure, liquid, air).
    fn owner(&self, c: [i32; 2]) -> [f32; 3] {
        let t = c[0].clamp(0, self.tiles[0] as i32 - 1) as usize
            + self.tiles[0] * c[1].clamp(0, self.tiles[1] as i32 - 1) as usize;
        if self.root_liquid[t] {
            [self.root_pressure[t], 1.0, 0.0]
        } else {
            [0.0, 0.0, self.root_air[t] as u8 as f32]
        }
    }
    /// The 4h pressure at an h centre, carried to the row's own depth where
    /// an air or a closed owner is in its stencil (the init launch).
    fn start(&self, p: [i32; 2], own: f32) -> f32 {
        let y = [
            (p[0] as f32 + 0.5) / 4.0 - 0.5,
            (p[1] as f32 + 0.5) / 4.0 - 0.5,
        ];
        let b = y.map(|v| v.floor() as i32);
        let w = [y[0] - y[0].floor(), y[1] - y[1].floor()];
        let top = self.tiles[1] as i32 - 1;
        let mut sum = [0.0_f32; 3];
        for k in 0..4 {
            let c = [b[0] + (k & 1), b[1] + (k >> 1)];
            let t = c[0].clamp(0, self.tiles[0] as i32 - 1) as usize
                + self.tiles[0] * c[1].clamp(0, top) as usize;
            if !self.root_liquid[t] {
                continue;
            }
            let weight = (if k & 1 == 1 { w[0] } else { 1.0 - w[0] })
                * (if k >> 1 == 1 { w[1] } else { 1.0 - w[1] });
            sum[0] += weight * self.root_pressure[t];
            sum[1] += weight * self.root_phi[t];
            sum[2] += weight;
        }
        if sum[2] >= 0.99999 {
            return sum[0] / sum[2];
        }
        if sum[2] <= 0.0 {
            return 0.0;
        }
        // bCoarseCarry: per column, the 4h pressure's vertical difference per
        // 4h spacing, the liquid owners' height above b, and the air weight.
        let mut rise = [0.0_f32; 2];
        let mut height = [0.0_f32; 2];
        let mut air = 0.0;
        for bit in 0..2 {
            let side = if bit == 1 { w[0] } else { 1.0 - w[0] };
            let x = b[0] + bit;
            let y0 = b[1].clamp(0, top);
            let y1 = (b[1] + 1).clamp(0, top);
            let low = self.owner([x, y0]);
            let high = if y1 != y0 { self.owner([x, y1]) } else { low };
            let lower = side * (1.0 - w[1]);
            let upper = side * w[1];
            let liquid = lower * low[1] + upper * high[1];
            air += lower * low[2] + upper * high[2];
            height[0] += lower * low[1] * (y0 - b[1]) as f32 + upper * high[1] * (y1 - b[1]) as f32;
            height[1] += liquid;
            if y1 != y0 && low[1] > 0.0 && high[1] > 0.0 {
                rise[0] += liquid * (high[0] - low[0]);
                rise[1] += liquid;
                continue;
            }
            let mut own = low;
            let mut at = y0;
            let mut beyond = if y0 == 0 { y0 + 1 } else { y0 - 1 };
            if y1 != y0 {
                if high[1] > 0.0 {
                    own = high;
                    at = y1;
                    beyond = y1 + 1;
                } else {
                    beyond = y0 - 1;
                }
            }
            if own[1] == 0.0 || beyond < 0 || beyond > top {
                continue;
            }
            let other = self.owner([x, beyond]);
            if other[1] > 0.0 {
                rise[0] += liquid * (other[0] - own[0]) * (beyond - at) as f32;
                rise[1] += liquid;
            }
        }
        let mean = sum[0] / sum[2];
        if air > 0.0 || rise[1] == 0.0 {
            return mean * (own * sum[2] / sum[1].min(-1e-30)).clamp(0.0, 1.0 / THETA_MIN);
        }
        let bound = mean / THETA_MIN;
        (mean + (w[1] - height[0] / height[1]) * rise[0] / rise[1])
            .clamp(bound.min(0.0), bound.max(0.0))
    }
    // bFollowHalo: the wall halo follows its row, the Neumann ghost clamped
    // at 0 of the total pressure.
    fn follow(&mut self, cell: usize, value: f32, scale: [f32; 2]) {
        for f in 0..4 {
            if self.kinds[cell][f] == WALL {
                self.halo[cell][f] = (value + sign(f) as f32 * self.forced[cell][f] * scale[f / 2])
                    .max(-self.base[cell]);
            }
        }
    }
    fn live(&self, cell: usize) -> bool {
        self.flags[cell] & LIQUID != 0 && self.diagonal[cell] > 0.0
    }
    fn off(&self, cell: usize) -> f32 {
        let mut off = 0.0;
        for f in 0..4 {
            match self.kinds[cell][f] {
                BAND => off += self.coefficient[cell][f] * self.solve[self.near[cell][f] as usize],
                WALL => off += self.coefficient[cell][f] * self.halo[cell][f],
                _ => {}
            }
        }
        off
    }
    // A liquid row's residual; a p_min = 0 row at its bound counts only a
    // positive one.
    fn residual(&self, cell: usize) -> f32 {
        let r = self.rhs[cell] + self.off(cell) - self.diagonal[cell] * self.solve[cell];
        if self.flags[cell] & INSIDE != 0 && self.solve[cell] + self.base[cell] <= 0.0 {
            r.max(0.0)
        } else {
            r
        }
    }
    /// One red-black Gauss-Seidel half sweep at h; `settle` relaxes only the
    /// shut rows.
    fn sweep(&mut self, colour: usize, settle: bool, scale: [f32; 2]) {
        for cell in 0..self.solve.len() {
            let l = cell % 16;
            if (l % 4 + l / 4) & 1 != colour
                || !self.live(cell)
                || (settle && self.flags[cell] & SHUT == 0)
            {
                continue;
            }
            let mut next = (self.rhs[cell] + self.off(cell)) / self.diagonal[cell];
            if self.flags[cell] & INSIDE != 0 {
                next = next.max(-self.base[cell]);
            }
            self.solve[cell] = next;
            self.follow(cell, next, scale);
        }
    }
    fn middle_colour(&self, c: usize) -> usize {
        let t = self.slots[c / 4] as usize;
        let a = c % 4;
        (2 * (t % self.tiles[0]) + (a & 1) + 2 * (t / self.tiles[0]) + (a >> 1)) & 1
    }
    /// One half sweep of the 2h aggregates. With `prolong` the 4h correction
    /// is prolonged on the fly: a red update never reads its own value.
    fn middle_sweep(&mut self, colour: usize, prolong: bool) {
        for c in 0..self.middle.len() {
            let m = self.middle[c];
            if self.middle_colour(c) != colour || m.diagonal <= 0.0 {
                continue;
            }
            let mut off = 0.0;
            for f in 0..4 {
                if m.near[f] == NONE {
                    continue;
                }
                let n = m.near[f] as usize;
                let mut v = self.middle[n].correction;
                if prolong && self.middle[n].diagonal > 0.0 {
                    v += self.coarse[n / 4].correction;
                }
                off += m.couple[f] * v;
            }
            self.middle[c].correction = (m.residual + off) / m.diagonal;
        }
    }

    /// The band solve and its projection. `g` holds the projected 4h velocity
    /// in simulation ownership; `forced` is the field before any pressure.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &mut self,
        g: &mut Grid,
        layout: &Layout,
        root: &Fluid,
        forced: &[[f32; 2]],
        forced_low: [&[f32]; 2],
        rate: f32,
        rho: f32,
        dt: f32,
        tolerance: f32,
    ) {
        let [tw, th] = self.tiles;
        let h = g.h[0].min(g.h[1]);
        // The 4h solve, per tile.
        let level = &root.pressure.levels[0];
        for ty in 0..th {
            for tx in 0..tw {
                let t = tx + tw * ty;
                let c = [tx as i32, ty as i32];
                let phi = root.grid.pressure_phi(c);
                let open = root.grid.capacity[t] > 1e-5;
                let wet = self.wet(phi, 4.0, h);
                self.root_phi[t] = phi;
                self.root_liquid[t] = open && wet;
                self.root_air[t] = open && !wet;
                self.root_pressure[t] = if phi < 0.0 {
                    level.p[level.index([c[0] + 1, c[1] + 1]).unwrap()]
                } else {
                    0.0
                };
            }
        }
        for i in 0..self.pressure.len() {
            let p = g.point(i);
            self.pressure[i] = self.root_pressure[layout.tile_of(p)];
            if layout.fine_cell(p) {
                self.phi[i] = g.pressure_phi(p);
            }
        }
        // list: every h tile with a liquid pressure row.
        self.slots.clear();
        self.slot_of.fill(0);
        for t in 0..tw * th {
            if !layout.fine[t] {
                continue;
            }
            let origin = [4 * (t % tw), 4 * (t / tw)];
            let wet = (0..16).any(|l| {
                self.wet(
                    self.phi[origin[0] + l % 4 + self.dims[0] * (origin[1] + l / 4)],
                    1.0,
                    h,
                )
            });
            if wet {
                self.slots.push(t as u32);
                self.slot_of[t] = self.slots.len() as u32;
            }
        }
        let count = self.slots.len();
        let rows = 16 * count;
        self.rhs.clear();
        self.rhs.resize(rows, 0.0);
        self.diagonal.clear();
        self.diagonal.resize(rows, 0.0);
        self.base.clear();
        self.base.resize(rows, 0.0);
        self.solve.clear();
        self.solve.resize(rows, 0.0);
        self.flags.clear();
        self.flags.resize(rows, 0);
        self.kinds.clear();
        self.kinds.resize(rows, [GHOST; 4]);
        self.coefficient.clear();
        self.coefficient.resize(rows, [0.0; 4]);
        self.forced.clear();
        self.forced.resize(rows, [0.0; 4]);
        self.volume.clear();
        self.volume.resize(rows, [0.0; 4]);
        self.halo.clear();
        self.halo.resize(rows, [0.0; 4]);
        self.near.clear();
        self.near.resize(rows, [NONE; 4]);
        self.middle.clear();
        self.middle.resize(4 * count, Aggregate::default());
        self.coarse.clear();
        self.coarse.resize(count, Aggregate::default());
        self.receipt = BandReceipt {
            tiles: count as u32,
            converged: true,
            ..Default::default()
        };
        if count == 0 {
            return;
        }
        let halo_scale = [rho * g.h[0] / dt, rho * g.h[1] / dt];
        let mut shut_rows = false;

        // prep: row assembly, baked once per solve.
        for s in 0..count {
            let origin = self.origin(s);
            for l in 0..16 {
                let p = [origin[0] + (l % 4) as i32, origin[1] + (l / 4) as i32];
                let cell = 16 * s + l;
                let i = g.index(p).unwrap();
                let own = self.phi[i];
                let liquid = self.wet(own, 1.0, h);
                let shut = g.capacity[i] <= 1e-5;
                let inside = shut || g.rigid_centres.as_ref().is_some_and(|v| v[i]);
                // Native divergenceAtWithCapacity: V u + (V_i - V) u_s.
                let bodies = liquid && g.rigid_speeds.is_some();
                let capacity = if bodies { g.capacity[i] } else { 0.0 };
                let mut diagonal = 0.0;
                let mut divergence = 0.0;
                for f in 0..4 {
                    let axis = f / 2;
                    let spacing = g.h[axis];
                    let mut q = p;
                    q[axis] += sign(f);
                    let mut low = p;
                    if sign(f) < 0 {
                        low[axis] -= 1;
                    }
                    let velocity = face(forced, forced_low, g, low, axis);
                    let volume = g.pressure_face(low, axis);
                    let mut kind = GHOST;
                    let mut coefficient = 0.0;
                    let mut sealed = false;
                    if g.index(q).is_none() {
                        if axis == 1 && sign(f) > 0 && g.open_top {
                            kind = OPEN;
                            coefficient = 1.0 / (spacing * spacing * theta(own, 0.5 * h));
                        } else {
                            kind = WALL;
                        }
                    } else {
                        let (phi, wet, neumann) = self.neighbour(g, layout, q, h);
                        let gradient =
                            1.0 / (spacing * spacing * pair_theta(own, liquid, phi, wet));
                        if let Some(n) = self.cell(q) {
                            kind = BAND;
                            coefficient = gradient;
                            self.near[cell][f] = n as u32;
                        } else if neumann {
                            kind = NEUMANN;
                        } else {
                            coefficient = gradient;
                        }
                        // A shut row has no surface of its own: its face to
                        // an air row is closed.
                        if kind != NEUMANN {
                            sealed =
                                (liquid && shut && !wet) || (!liquid && wet && g.open(q) <= 1e-5);
                        }
                    }
                    // CM11a: V scales the coefficient and the flux.
                    let mut fraction = volume;
                    if volume <= 1e-6 || sealed {
                        kind = CLOSED;
                        coefficient = 0.0;
                        fraction = 0.0;
                    } else if kind == WALL {
                        coefficient = volume / (spacing * spacing);
                    } else if kind != NEUMANN {
                        coefficient *= volume;
                    }
                    self.kinds[cell][f] = kind;
                    self.coefficient[cell][f] = coefficient;
                    self.forced[cell][f] = velocity;
                    self.volume[cell][f] = volume;
                    if kind != NEUMANN {
                        diagonal += coefficient;
                        divergence += sign(f) as f32 * fraction * velocity / spacing;
                    }
                    if bodies && kind != WALL && kind != OPEN {
                        divergence +=
                            sign(f) as f32 * (capacity - fraction) * g.solid_speed(low, axis)
                                / spacing;
                    }
                }
                self.rhs[cell] = if liquid {
                    -rho * (divergence - rate * g.capacity[i]) / dt
                } else {
                    0.0
                };
                self.diagonal[cell] = diagonal;
                self.flags[cell] =
                    (liquid as u8) * LIQUID + (inside as u8) * INSIDE + (shut as u8) * SHUT;
                shut_rows |= liquid && shut;
            }
        }
        // middleBake, coarseBake: Galerkin aggregation over liquid rows.
        // Faces inside an aggregate cancel, wall halos follow their row and
        // Neumann faces carry no correction flux. Dirichlet terms stay exact;
        // only couplings between aggregates scale.
        for c in 0..4 * count {
            let s = c / 4;
            let a = [c % 4 & 1, c % 4 >> 1];
            let tile = self.slots[s] as usize;
            let mut diagonal = 0.0;
            let mut couple = [0.0_f32; 4];
            for k in 0..4 {
                let l = [2 * a[0] + (k & 1), 2 * a[1] + (k >> 1)];
                let cell = 16 * s + l[0] + 4 * l[1];
                if !self.live(cell) {
                    continue;
                }
                for f in 0..4 {
                    let coefficient = self.coefficient[cell][f];
                    match self.kinds[cell][f] {
                        GHOST | OPEN => diagonal += coefficient,
                        BAND => {
                            let n = self.near[cell][f] as usize;
                            let mut m = [l[0] as i32, l[1] as i32];
                            m[f / 2] += sign(f);
                            if !self.live(n) {
                                diagonal += coefficient;
                            } else if !(m[f / 2] >= 0
                                && m[f / 2] < 4
                                && m[f / 2] as usize / 2 == a[f / 2])
                            {
                                couple[f] += coefficient;
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut near = [NONE; 4];
            let mut coupled = 0.0;
            for f in 0..4 {
                couple[f] *= COARSE_SCALE;
                coupled += couple[f];
                let mut m = [
                    2 * (tile % tw) as i32 + a[0] as i32,
                    2 * (tile / tw) as i32 + a[1] as i32,
                ];
                m[f / 2] += sign(f);
                if m[0] < 0 || m[1] < 0 || m[0] >= 2 * tw as i32 || m[1] >= 2 * th as i32 {
                    continue;
                }
                let slot = self.slot_of[m[0] as usize / 2 + tw * (m[1] as usize / 2)];
                if slot != 0 {
                    near[f] = (4 * (slot - 1)) + (m[0] as u32 % 2) + 2 * (m[1] as u32 % 2);
                }
            }
            self.middle[c] = Aggregate {
                diagonal: diagonal + coupled,
                couple,
                near,
                ..Default::default()
            };
        }
        for s in 0..count {
            let tile = self.slots[s] as usize;
            let mut diagonal = 0.0;
            let mut couple = [0.0_f32; 4];
            for a in 0..4 {
                let m = self.middle[4 * s + a];
                let at = [a & 1, a >> 1];
                diagonal += m.diagonal;
                for f in 0..4 {
                    let inside = at[f / 2] == if sign(f) > 0 { 0 } else { 1 };
                    diagonal -= m.couple[f];
                    if !inside {
                        couple[f] += m.couple[f];
                    }
                }
            }
            // diagonal is now the aggregate's Dirichlet part.
            let mut near = [NONE; 4];
            let mut coupled = 0.0;
            for f in 0..4 {
                couple[f] *= COARSE_SCALE;
                coupled += couple[f];
                let mut c = [(tile % tw) as i32, (tile / tw) as i32];
                c[f / 2] += sign(f);
                if c[0] < 0 || c[1] < 0 || c[0] >= tw as i32 || c[1] >= th as i32 {
                    continue;
                }
                let slot = self.slot_of[c[0] as usize + tw * c[1] as usize];
                if slot != 0 {
                    near[f] = slot - 1;
                }
            }
            self.coarse[s] = Aggregate {
                diagonal: (diagonal + coupled).max(0.0),
                couple,
                near,
                ..Default::default()
            };
        }

        // init: start from the 4h pressure; Neumann faces carry the projected
        // 4h flux into the row.
        for s in 0..count {
            let origin = self.origin(s);
            for l in 0..16 {
                let p = [origin[0] + (l % 4) as i32, origin[1] + (l / 4) as i32];
                let cell = 16 * s + l;
                let liquid = self.flags[cell] & LIQUID != 0;
                let mut rhs = self.rhs[cell];
                for f in 0..4 {
                    if self.kinds[cell][f] != NEUMANN {
                        continue;
                    }
                    let mut low = p;
                    if sign(f) < 0 {
                        low[f / 2] -= 1;
                    }
                    let kept = g.face(low, f / 2);
                    self.forced[cell][f] = kept;
                    rhs -= rho * sign(f) as f32 * self.volume[cell][f] * kept / (g.h[f / 2] * dt);
                }
                if !liquid {
                    continue;
                }
                let mut start = self.start(p, self.phi[g.index(p).unwrap()]);
                if self.flags[cell] & INSIDE != 0 {
                    start = start.max(0.0);
                }
                self.base[cell] = start;
                self.rhs[cell] = rhs;
                self.follow(cell, 0.0, halo_scale);
            }
        }
        // rebase: the rows against their start. The differences of the start
        // are exact; the rows then solve for the correction to it.
        for cell in 0..rows {
            if self.flags[cell] & LIQUID == 0 {
                continue;
            }
            let own = self.base[cell];
            let mut open = 0.0;
            let mut sums = [0.0_f32; 2];
            for axis in 0..2 {
                let mut w = [0.0_f32; 2];
                let mut d = [0.0_f32; 2];
                for side in 0..2 {
                    let f = 2 * axis + side;
                    match self.kinds[cell][f] {
                        BAND => {
                            w[side] = self.coefficient[cell][f];
                            d[side] = self.base[self.near[cell][f] as usize] - own;
                        }
                        GHOST | OPEN => open += self.coefficient[cell][f],
                        _ => {}
                    }
                }
                sums[axis] = w[0] * (d[0] + d[1]) + (w[1] - w[0]) * d[1];
            }
            self.rhs[cell] += (sums[0] + sums[1]) - open * own;
        }
        // Shut rows exist only beside a solid: they settle on their open
        // neighbours' start before any open row relaxes.
        if shut_rows {
            self.sweep(0, true, halo_scale);
            self.sweep(1, true, halo_scale);
        }
        let scale = dt / rho;
        // The 3D band's residual target is 0: every cycle runs.
        for _ in 0..CYCLES {
            for _ in 0..FINE_SWEEPS {
                self.sweep(0, false, halo_scale);
                self.sweep(1, false, halo_scale);
            }
            // restrict: the h residual into the 2h aggregates. With every 2h
            // correction zero, the first red half sweep is local.
            for c in 0..4 * count {
                let a = [c % 4 & 1, c % 4 >> 1];
                let mut sum = 0.0;
                for k in 0..4 {
                    let cell = 16 * (c / 4) + 2 * a[0] + (k & 1) + 4 * (2 * a[1] + (k >> 1));
                    if self.live(cell) {
                        sum += self.residual(cell);
                    }
                }
                let red = self.middle_colour(c) == 0 && self.middle[c].diagonal > 0.0;
                self.middle[c].residual = sum;
                self.middle[c].correction = if red {
                    sum / self.middle[c].diagonal
                } else {
                    0.0
                };
            }
            self.middle_sweep(1, false);
            for _ in 1..MIDDLE_SWEEPS {
                self.middle_sweep(0, false);
                self.middle_sweep(1, false);
            }
            // middleRestrict: the 2h residual into the tile aggregate.
            for s in 0..count {
                let mut sum = 0.0;
                for c in 4 * s..4 * s + 4 {
                    let m = self.middle[c];
                    if m.diagonal <= 0.0 {
                        continue;
                    }
                    let mut off = 0.0;
                    for f in 0..4 {
                        if m.near[f] != NONE {
                            off += m.couple[f] * self.middle[m.near[f] as usize].correction;
                        }
                    }
                    sum += m.residual + off - m.diagonal * m.correction;
                }
                self.coarse[s].residual = sum;
                self.coarse[s].correction = 0.0;
            }
            for _ in 0..COARSE_SWEEPS {
                for colour in 0..2 {
                    for s in 0..count {
                        let t = self.slots[s] as usize;
                        let c = self.coarse[s];
                        if (t % tw + t / tw) & 1 != colour || c.diagonal <= 0.0 {
                            continue;
                        }
                        let mut off = 0.0;
                        for f in 0..4 {
                            if c.near[f] != NONE {
                                off += c.couple[f] * self.coarse[c.near[f] as usize].correction;
                            }
                        }
                        self.coarse[s].correction = c.correction
                            + COARSE_RELAXATION * ((c.residual + off) / c.diagonal - c.correction);
                    }
                }
            }
            self.middle_sweep(0, true);
            self.middle_sweep(1, false);
            for _ in 1..MIDDLE_SWEEPS {
                self.middle_sweep(0, false);
                self.middle_sweep(1, false);
            }
            // middleProlong.
            for cell in 0..rows {
                if !self.live(cell) {
                    continue;
                }
                let l = cell % 16;
                let mut next = self.solve[cell]
                    + self.middle[4 * (cell / 16) + (l % 4) / 2 + 2 * (l / 8)].correction;
                if self.flags[cell] & INSIDE != 0 {
                    next = next.max(-self.base[cell]);
                }
                self.solve[cell] = next;
                self.follow(cell, next, halo_scale);
            }
            for _ in 0..FINE_SWEEPS {
                self.sweep(0, false, halo_scale);
                self.sweep(1, false, halo_scale);
            }
        }
        // measure.
        let largest = (0..rows)
            .filter(|&cell| self.live(cell))
            .map(|cell| self.residual(cell).abs())
            .fold(0.0_f32, f32::max);
        self.receipt.residual = largest * scale;
        self.receipt.cycles = CYCLES as u32;
        self.receipt.converged = self.receipt.residual <= tolerance;
        self.receipt.rows = (0..rows).filter(|&cell| self.live(cell)).count() as u32;

        // project: each band cell's positive faces, the low walls, and the
        // band faces a non-band texel below a band tile holds.
        for s in 0..count {
            let origin = self.origin(s);
            for l in 0..16 {
                let p = [origin[0] + (l % 4) as i32, origin[1] + (l / 4) as i32];
                let cell = 16 * s + l;
                let i = g.index(p).unwrap();
                self.pressure[i] = self.base[cell] + self.solve[cell];
                let mut released = 0;
                for axis in 0..2 {
                    let value = self.projected(g, cell, p, 2 * axis + 1, scale);
                    g.velocity[i][axis] = value;
                    if self.released(g, cell, p, 2 * axis + 1, value, dt) {
                        released |= 1 << axis;
                    }
                    if p[axis] == 0 {
                        let low = self.projected(g, cell, p, 2 * axis, scale);
                        if axis == 0 {
                            g.low_x[p[1] as usize] = low;
                        } else {
                            g.low_y[p[0] as usize] = low;
                        }
                        if self.released(g, cell, p, 2 * axis, low, dt) {
                            released |= 1 << (axis + 2);
                        }
                    } else if l % 4 == 0 && axis == 0 || l / 4 == 0 && axis == 1 {
                        let mut q = p;
                        q[axis] -= 1;
                        if self.cell(q).is_none() {
                            let j = g.index(q).unwrap();
                            g.velocity[j][axis] = self.projected(g, cell, p, 2 * axis, scale);
                        }
                    }
                }
                g.released[i] = released;
            }
        }
    }
    // bFace: umProject on one face of a band cell.
    fn projected(&self, g: &Grid, cell: usize, p: [i32; 2], f: usize, scale: f32) -> f32 {
        let kind = self.kinds[cell][f];
        let axis = f / 2;
        let forced = self.forced[cell][f];
        if kind == NEUMANN {
            return forced;
        }
        if kind == CLOSED {
            // A closed face moves with its wall.
            let mut low = p;
            if sign(f) < 0 {
                low[axis] -= 1;
            }
            return g.solid_speed(low, axis);
        }
        let liquid = self.flags[cell] & LIQUID != 0;
        let h = g.h[axis];
        let own = self.solve[cell];
        let base = self.base[cell];
        let s = sign(f) as f32;
        // The gradient coefficient: the row coefficient without its V.
        let coefficient = self.coefficient[cell][f] / self.volume[cell][f];
        if kind == WALL {
            return if liquid {
                forced - scale * s * (self.halo[cell][f] - own) / h
            } else {
                forced
            };
        }
        if kind == OPEN {
            return if liquid {
                forced + scale * s * (base + own) * coefficient * h
            } else {
                forced
            };
        }
        let (other, other_base, other_liquid) = if kind == BAND {
            let n = self.near[cell][f] as usize;
            (self.solve[n], self.base[n], self.flags[n] & LIQUID != 0)
        } else {
            (0.0, 0.0, false)
        };
        if !liquid && !other_liquid {
            return 0.0;
        }
        // The start's difference and the correction's, each exact.
        forced - scale * s * ((other_base - base) + (other - own)) * coefficient * h
    }
    // bReleased: wall and embedded contact on the separating side's pressure.
    fn released(&self, g: &Grid, cell: usize, p: [i32; 2], f: usize, value: f32, dt: f32) -> bool {
        let axis = f / 2;
        let liquid = self.flags[cell] & LIQUID != 0;
        let total = |c: usize| self.solve[c] + self.base[c];
        let open = |q: [i32; 2]| g.open(q) > 1e-5;
        if sign(f) > 0 {
            let mut q = p;
            q[axis] += 1;
            let own = open(p);
            if own == open(q) || self.volume[cell][f] <= 1e-6 {
                return false;
            }
            let mut pressure = 0.0;
            if g.index(q).is_none() {
                if liquid && !(axis == 1 && g.open_top) {
                    pressure = self.halo[cell][f] + self.base[cell];
                }
            } else if own {
                if let Some(b) = self.cell(q).filter(|&b| self.flags[b] & LIQUID != 0) {
                    pressure = total(b);
                }
            } else if liquid {
                pressure = total(cell);
            }
            // Separation is relative to the wall's own motion.
            return pressure <= 0.0
                && (if own { -1.0 } else { 1.0 }) * (value - g.solid_speed(p, axis)) * dt
                    > 1e-4 * g.h[axis];
        }
        if !open(p) || self.volume[cell][f] <= 1e-6 || self.kinds[cell][f] != WALL {
            return false;
        }
        let pressure = if liquid {
            self.halo[cell][f] + self.base[cell]
        } else {
            0.0
        };
        pressure <= 0.0 && value * dt > 1e-4 * g.h[axis]
    }
}
