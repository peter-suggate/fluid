//! The narrow-band FLIP advance on the 3D method's mixed frame
//! (uniform-mixed-frame.ts with the method's particle stages): a 4h Eulerian
//! base with h tiles where the surface and its samples are, an all-4h pressure
//! solve and an h band solve inside it. The fields are uniform_geometric's
//! dense h lattice, which the layout resolves after every stage.
use super::{
    band::{Band, BandReceipt},
    layout::{Face, Layout},
    options::NarrowBandFlipOptions,
    particles::{Census, Particles},
    surface::Surface,
};
use crate::{
    types::ValidationError,
    uniform_geometric::{
        extension::Extension,
        grid::{plane_box_fraction, Grid},
        inflow::{Plug, Velocity},
        pages::Region,
        pressure::PressureReceipt,
        surface::advect_window,
        transport::Sources,
        velocity::departure,
        world::World as Fluid,
        UniformGeometricOptions,
    },
};

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub frame: u32,
    pub time: f64,
    /// The all-4h solve, its residuals h-equivalent.
    pub pressure: PressureReceipt,
    /// The h band solve inside it.
    pub band: BandReceipt,
    pub tiles: u32,
    /// Tiles resolved at h this step.
    pub fine_tiles: u32,
    /// Liquid the surface holds, in cells.
    pub volume: f64,
    /// Liquid the scene should hold: the start, plus sources, less outflow.
    pub budget: f64,
    /// Divergence asked of every liquid cell this step, in 1/s.
    pub control_rate: f32,
    pub injected_volume: f64,
    pub outflow_volume: f64,
    pub particles: u32,
    pub census: Census,
    pub reseed_clipped: u32,
    pub max_speed: f32,
}

pub struct World {
    /// The h lattice and its stages. Its own advance never runs here: V is
    /// the surface's occupancy, not a transported quantity.
    pub fluid: Fluid,
    /// The all-4h pressure grid: one owner a tile, under h tiles too.
    root: Fluid,
    root_extension: Extension,
    pub layout: Layout,
    pub band: Band,
    /// Each tile's positive 4h faces: the state of an unsplit one, the mean
    /// of a split one's patches.
    cache: Vec<[f32; 2]>,
    /// The same for the extended field the samplers read.
    sampled: Vec<[f32; 2]>,
    /// The velocity before any pressure, for the band's rows.
    forced: Vec<[f32; 2]>,
    forced_low: [Vec<f32>; 2],
    next: Vec<[f32; 2]>,
    next_cache: Vec<[f32; 2]>,
    /// Per tile: 1 at h, 2 at h with pressure rows, 0 at 4h.
    pub tiles: Vec<u8>,
    pub options: NarrowBandFlipOptions,
    pub particles: Particles,
    pub surface: Surface,
    pub receipt: Receipt,
    solver: UniformGeometricOptions,
    liquid: Vec<bool>,
    /// The Eq. 4 union's target, swapped with phi.
    union: Vec<f32>,
    budget: f64,
}

/// Liquid a step sweeps out through the open top: per column, the swept box's
/// liquid fraction under a plane fit of its corner phi.
fn outflow(g: &Grid, dt: f32) -> f64 {
    if !g.open_top {
        return 0.0;
    }
    let [nx, ny] = g.dims;
    let h = g.h[0].min(g.h[1]);
    let top = ny as f32;
    let mut total = 0.0;
    for x in 0..nx {
        let i = x + nx * (ny - 1);
        let speed = g.velocity[i][1].max(0.0);
        let travel = (speed * dt / g.h[1]).min(top);
        if travel <= 0.0 || g.capacity[i] <= 1e-5 {
            continue;
        }
        let corner = |dx: f32, y: f32| g.phi_at([x as f32 + dx, y]) / h;
        let [a, b, c, d] = [
            corner(0.0, top - travel),
            corner(1.0, top - travel),
            corner(0.0, top),
            corner(1.0, top),
        ];
        if a.min(b).min(c.min(d)) >= 0.0 {
            continue;
        }
        let normal = [0.5 * ((b - a) + (d - c)), 0.5 * ((c - a) + (d - b)), 0.0];
        let fraction = plane_box_fraction(normal, -0.25 * ((a + b) + (c + d)));
        total += (fraction * travel * g.capacity[i]) as f64;
    }
    total
}

impl World {
    pub fn new(
        grid: Grid,
        options: NarrowBandFlipOptions,
        gravity: [f32; 2],
        rho: f32,
        viscosity: f32,
        sigma: f32,
    ) -> Result<Self, ValidationError> {
        options.validate()?;
        let solver = options.solver();
        let dims = grid.dims;
        if dims.iter().any(|&n| n % 4 != 0 || n < 8) {
            return Err(ValidationError(
                "narrow-band FLIP needs a lattice of whole 4h tiles, at least two a side".into(),
            ));
        }
        let tiles = dims.map(|n| n / 4);
        let coarse = Grid::new(tiles, grid.h.map(|h| 4.0 * h), grid.open_top)?;
        // The 3D norm is h-equivalent, an owner's divergence times its width:
        // a 4h owner is held to a quarter of the tolerance.
        let mut quarter = solver.clone();
        quarter.pressure_residual_tolerance *= 0.25;
        let mut root = Fluid::from_grid(coarse, quarter, gravity, rho, viscosity, sigma)?;
        // A divergence the solve leaves is volume this method loses.
        root.pressure.keep_surface = true;
        root.pressure.warm = true;
        let fluid = Fluid::from_grid(grid, solver.clone(), gravity, rho, viscosity, sigma)?;
        let count = tiles[0] * tiles[1];
        Ok(Self {
            root_extension: Extension::dense(&root.grid),
            root,
            layout: Layout::new(dims, options.fine_grid_padding),
            band: Band::new(dims),
            cache: vec![[0.0; 2]; count],
            sampled: vec![[0.0; 2]; count],
            forced: fluid.grid.velocity.clone(),
            forced_low: [fluid.grid.low_x.clone(), fluid.grid.low_y.clone()],
            next: fluid.grid.velocity.clone(),
            next_cache: vec![[0.0; 2]; count],
            tiles: vec![2; count],
            union: fluid.grid.phi.clone(),
            liquid: vec![false; dims[0] * dims[1]],
            particles: Particles::new(dims),
            surface: Surface::new(dims),
            receipt: Receipt::default(),
            budget: 0.0,
            fluid,
            options,
            solver,
        })
    }

    /// The surface's cell state: V is its occupancy, and a cell is liquid
    /// where the pressure solve gives it a row.
    fn refresh(&mut self) {
        let g = &mut self.fluid.grid;
        let mut total = 0.0;
        for i in 0..g.volume.len() {
            let p = g.point(i);
            let volume = g.target(p);
            g.volume[i] = volume;
            self.liquid[i] = g.capacity[i] > 1e-5 && g.phi_centre(p) < 0.0;
            total += volume as f64;
        }
        self.receipt.volume = total;
    }

    /// Initialization (paper Section 3.2): the described liquid gets its whole
    /// band, and the budget is what its surface holds. Every tile is at h
    /// until the first step decides. Call once the solids are in the grid.
    pub fn bootstrap(&mut self) {
        self.surface.redistance(&mut self.fluid.grid, &self.layout);
        self.refresh();
        self.budget = self.receipt.volume;
        self.receipt.budget = self.budget;
        let e = Extension::build(&self.fluid.grid, &self.solver, 0.0);
        self.particles.resample(&self.fluid.grid);
        self.particles
            .seed(&self.fluid.grid, &e, &self.surface, true, false, &|_| 1.0);
        self.receipt.tiles = self.tiles.len() as u32;
        self.receipt.fine_tiles = self.layout.fine_count();
        self.receipt.particles = self.particles.list.len() as u32;
        self.receipt.reseed_clipped = self.particles.clipped;
    }

    /// Kim et al. 2007 against the scalar budget: the divergence that closes
    /// the gap over the control time, outside a 1% dead band.
    fn control_rate(&self, dt: f32) -> f32 {
        let seconds = self.options.volume_control_seconds;
        if seconds <= 0.0 {
            return 0.0;
        }
        let filled = self.receipt.volume as f32;
        let budget = self.budget as f32;
        let gap = budget - filled;
        let gain = (1.0 / seconds).min(1.0 / (3.0 * dt.max(1e-6)));
        let excess = gap.signum() * (gap.abs() - 0.01 * budget).max(0.0);
        (excess * gain / filled.max(0.5 * budget).max(1e-6)).clamp(-4.0, 4.0)
    }

    /// The extended field under the layout: a 4h face the extension left
    /// alone keeps its state, one it filled takes the mean of its texels, and
    /// a 4h tile's other texels follow.
    fn mix(&mut self, e: &mut Extension) {
        let g = &self.fluid.grid;
        let [tw, th] = self.layout.tiles;
        for ty in 0..th {
            for tx in 0..tw {
                for a in 0..2 {
                    if self.layout.face(self.layout.top([tx, ty], a, 0), a) != Face::Coarse {
                        continue;
                    }
                    let mut kept = true;
                    let mut sum = 0.0;
                    for j in 0..4 {
                        let i = g.index(self.layout.top([tx, ty], a, j)).unwrap();
                        kept &= e.values[i][a] == g.velocity[i][a];
                        sum += e.values[i][a];
                    }
                    self.sampled[tx + tw * ty][a] = if kept {
                        self.cache[tx + tw * ty][a]
                    } else {
                        0.25 * sum
                    };
                }
            }
        }
        self.layout.resolve(&mut e.values, &mut self.sampled);
    }

    /// Semi-Lagrangian momentum: every face that is state traces from its own
    /// centre, its substeps bounded by its own width.
    fn transport(&mut self, e: &Extension, dt: f32) {
        let g = &self.fluid.grid;
        self.next.clone_from(&g.velocity);
        self.next_cache.clone_from(&self.cache);
        for i in 0..g.velocity.len() {
            let p = g.point(i);
            for a in 0..2 {
                let kind = self.layout.face(p, a);
                if kind == Face::Derived || (kind == Face::Coarse && p[1 - a] % 4 != 0) {
                    continue;
                }
                let mut start = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
                start[a] += 0.5;
                let (width, old) = if kind == Face::Coarse {
                    start[1 - a] += 1.5;
                    (4.0, self.cache[self.layout.tile_of(p)][a])
                } else {
                    (1.0, g.velocity[i][a])
                };
                // E2b: a face outside the extension's fine tiles carries none.
                let mut v = if e.fine_at(start) {
                    e.sample(departure(g, e, start, dt, width))[a]
                } else {
                    0.0
                };
                // Keep an old velocity directed away from a positive wall.
                if p[a] == g.dims[a] as i32 - 1 && !(a == 1 && g.open_top) {
                    v = v.min(old);
                }
                if kind == Face::Coarse {
                    self.next_cache[self.layout.tile_of(p)][a] = v;
                } else {
                    self.next[i][a] = v;
                }
            }
        }
        std::mem::swap(&mut self.fluid.grid.velocity, &mut self.next);
        std::mem::swap(&mut self.cache, &mut self.next_cache);
    }

    /// Body forces. The h faces take uniform_geometric's; a 4h face takes
    /// gravity where either of its owners holds liquid and the viscous
    /// Laplacian at its own width.
    fn force(&mut self, e: &Extension, nozzle: Option<&Velocity>, dt: f32) {
        let g = &self.fluid.grid;
        let [tw, th] = self.layout.tiles;
        let nxv = g.dims[0] + 1;
        let occupancy = |t: [i32; 2]| {
            if t[0] < 0 || t[1] < 0 || t[0] >= tw as i32 || t[1] >= th as i32 {
                return 0.0;
            }
            let j = 4 * t[0] as usize + nxv * 4 * t[1] as usize;
            let centre =
                0.25 * ((g.phi[j] + g.phi[j + 4]) + (g.phi[j + 4 * nxv] + g.phi[j + 4 * nxv + 4]));
            (0.5 - centre / (16.0 * g.h[1])).clamp(0.0, 1.0)
        };
        let molecular = self.fluid.viscosity / self.fluid.rho;
        let stored = |t: [i32; 2], a: usize| {
            self.cache
                [t[0].clamp(0, tw as i32 - 1) as usize + tw * t[1].clamp(0, th as i32 - 1) as usize]
                [a]
        };
        for ty in 0..th {
            for tx in 0..tw {
                let t = [tx as i32, ty as i32];
                let own = occupancy(t);
                for a in 0..2 {
                    let mut v = self.cache[tx + tw * ty][a];
                    if own > 0.0 && molecular > 0.0 {
                        for b in 0..2 {
                            let mut low = t;
                            let mut high = t;
                            low[b] -= 1;
                            high[b] += 1;
                            let width = 4.0 * g.h[b];
                            v += dt
                                * molecular
                                * ((stored(high, a) - 2.0 * stored(t, a)) + stored(low, a))
                                / (width * width);
                        }
                    }
                    if a == 1 && (own > 1e-5 || occupancy([t[0], t[1] + 1]) > 1e-5) {
                        v += self.fluid.gravity[1] * dt;
                    }
                    self.next_cache[tx + tw * ty][a] = v;
                }
            }
        }
        self.fluid.force_velocity(e, nozzle, dt);
        for ty in 0..th {
            for tx in 0..tw {
                for a in 0..2 {
                    if self.layout.face(self.layout.top([tx, ty], a, 0), a) == Face::Coarse {
                        self.cache[tx + tw * ty][a] = self.next_cache[tx + tw * ty][a];
                    }
                }
            }
        }
        self.layout
            .resolve(&mut self.fluid.grid.velocity, &mut self.cache);
    }

    /// The two-stage projection: the all-4h solve of the forced field with
    /// the volume-control divergence, its projected faces back in simulation
    /// ownership, then the h band from the forced field inside them.
    fn project(&mut self, nozzle: Option<&Velocity>, rate: f32, dt: f32) {
        let [tw, th] = self.layout.tiles;
        let layout = &self.layout;
        {
            let g = &self.fluid.grid;
            self.forced.clone_from(&g.velocity);
            self.forced_low[0].clone_from(&g.low_x);
            self.forced_low[1].clone_from(&g.low_y);
            let r = &mut self.root.grid;
            let nxv = g.dims[0] + 1;
            for y in 0..=th {
                for x in 0..=tw {
                    r.phi[x + (tw + 1) * y] = g.phi[4 * x + nxv * 4 * y];
                }
            }
            // A 4h owner's open fraction and face V are the means of its h ones.
            let cut = !g.solid_free();
            let mut faces = r.rigid_faces.take().unwrap_or_default();
            let mut speeds = r.rigid_speeds.take().unwrap_or_default();
            let mut centres = r.rigid_centres.take().unwrap_or_default();
            faces.resize(tw * th, [0.0; 2]);
            speeds.resize(tw * th, [0.0; 2]);
            centres.resize(tw * th, false);
            for ty in 0..th {
                for tx in 0..tw {
                    let t = tx + tw * ty;
                    let mut open = 0.0;
                    let mut inside = 0;
                    for k in 0..16 {
                        let i = 4 * tx + k % 4 + g.dims[0] * (4 * ty + k / 4);
                        open += g.capacity[i];
                        inside += g.rigid_centres.as_ref().is_some_and(|v| v[i]) as u32;
                    }
                    r.capacity[t] = open / 16.0;
                    r.velocity[t] = self.cache[t];
                    centres[t] = inside >= 8;
                    for a in 0..2 {
                        let layer = |f: &dyn Fn([i32; 2]) -> f32| {
                            0.25 * (0..4).map(|j| f(layout.top([tx, ty], a, j))).sum::<f32>()
                        };
                        faces[t][a] = layer(&|p| g.pressure_face(p, a));
                        speeds[t][a] = layer(&|p| g.solid_speed(p, a));
                    }
                }
            }
            for ty in 0..th {
                r.low_x[ty] = 0.25 * g.low_x[4 * ty..4 * ty + 4].iter().sum::<f32>();
            }
            for tx in 0..tw {
                r.low_y[tx] = 0.25 * g.low_y[4 * tx..4 * tx + 4].iter().sum::<f32>();
            }
            r.rigid_faces = cut.then_some(faces);
            r.rigid_speeds = g.rigid_speeds.is_some().then_some(speeds);
            r.rigid_centres = g.rigid_centres.is_some().then_some(centres);
        }
        self.root.volume_rate = rate;
        self.root.project(&self.root_extension, None, dt);
        {
            // Back to simulation: a 4h face takes its value, every patch of a
            // split one takes it too, and an h tile's faces run linearly
            // between its two.
            let r = &self.root.grid;
            let g = &mut self.fluid.grid;
            for ty in 0..th {
                for tx in 0..tw {
                    let t = tx + tw * ty;
                    for a in 0..2 {
                        let high = r.velocity[t][a];
                        let mut below = [tx as i32, ty as i32];
                        below[a] -= 1;
                        let low = r.face(below, a);
                        self.cache[t][a] = high;
                        for j in 0..4 {
                            let top = layout.top([tx, ty], a, j);
                            g.velocity[top[0] as usize + g.dims[0] * top[1] as usize][a] = high;
                            if !layout.fine[t] {
                                continue;
                            }
                            for i in 0..3 {
                                let mut p = top;
                                p[a] -= 3 - i;
                                g.velocity[p[0] as usize + g.dims[0] * p[1] as usize][a] =
                                    low + (high - low) * (i + 1) as f32 / 4.0;
                            }
                        }
                    }
                }
            }
            for y in 0..g.dims[1] {
                g.low_x[y] = r.low_x[y / 4];
            }
            for x in 0..g.dims[0] {
                g.low_y[x] = r.low_y[x / 4];
            }
            for i in 0..g.released.len() {
                let p = g.point(i);
                let bits = r.released[layout.tile_of(p)];
                let mut released = 0;
                for a in 0..2 {
                    if p[a] % 4 == 3 {
                        released |= bits & (1 << a);
                    }
                    if p[a] == 0 {
                        released |= bits & (1 << (a + 2));
                    }
                }
                g.released[i] = released;
            }
        }
        self.band.solve(
            &mut self.fluid.grid,
            layout,
            &self.root,
            &self.forced,
            [&self.forced_low[0], &self.forced_low[1]],
            rate,
            self.fluid.rho,
            dt,
            self.options.pressure_residual_tolerance,
        );
        let g = &mut self.fluid.grid;
        for i in 0..g.velocity.len() {
            let p = g.point(i);
            let tile = layout.tile_of(p);
            self.tiles[tile] = match (layout.fine[tile], self.band.rows_at(p)) {
                (true, true) => 1,
                (true, false) => 2,
                _ => 4,
            };
            if let Some(nozzle) = nozzle.filter(|_| layout.fine[tile]) {
                g.velocity[i] = nozzle.apply(p, g.velocity[i]);
            }
        }
        layout.resolve(&mut g.velocity, &mut self.cache);
    }

    pub fn advance(&mut self, dt: f32) -> Result<(), ValidationError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ValidationError("invalid narrow-band FLIP timestep".into()));
        }
        let time = self.receipt.time;
        let plug = self
            .fluid
            .inflow
            .as_ref()
            .map(|source| Plug::new(source, time, dt, &self.fluid.grid))
            .filter(Plug::active);
        let n = self.liquid.len();
        let mut sources = Sources::default();
        {
            let g = &self.fluid.grid;
            if !g.drops.is_empty() {
                sources.drop = (0..n).map(|i| g.drop_fraction(g.point(i))).collect();
            }
            if let Some(plug) = &plug {
                sources.plug = (0..n).map(|i| plug.amount(g.point(i))).collect();
            }
        }
        let sourced = sources.any();
        let left = outflow(&self.fluid.grid, dt).min(self.budget);
        self.budget -= left;
        let rate = self.control_rate(dt);

        // Head of step: the layout from the starting state, the fields under
        // it, and the extended velocity the samples and the level set both
        // move through.
        {
            let g = &self.fluid.grid;
            let source = |p: [f32; 2]| {
                let phi = g.source_phi(p, f32::MAX);
                plug.as_ref().map_or(phi, |plug| plug.stream_phi(p, phi))
            };
            self.layout.decide(
                g,
                &self.particles.list,
                &self.surface,
                dt,
                sourced.then_some(&source as &dyn Fn([f32; 2]) -> f32),
            );
        }
        self.layout.remap(&mut self.fluid.grid, &mut self.cache);
        self.refresh();
        let mut e = Extension::build_with(
            &self.fluid.grid,
            &self.solver,
            dt,
            &sources,
            std::mem::take(&mut self.fluid.packed),
        );
        self.mix(&mut e);
        self.particles
            .advect(&self.fluid.grid, &e, &self.liquid, dt)?;
        // The level set moves without its sources: they join after the union,
        // which would otherwise shrink liquid no sample has reached yet.
        let drops = std::mem::take(&mut self.fluid.grid.drops);
        advect_window(
            &self.fluid.grid,
            &e,
            dt,
            Region::whole(self.fluid.grid.dims),
            None,
            &self.solver,
            &mut self.fluid.advected,
        );
        self.fluid.grid.drops = drops;
        self.layout.resolve_phi(&mut self.fluid.advected);
        self.surface.reconstruct(
            &self.fluid.grid,
            &self.fluid.advected,
            &self.particles.list,
            &mut self.union,
        );
        std::mem::swap(&mut self.fluid.grid.phi, &mut self.union);
        self.layout.resolve_phi(&mut self.fluid.grid.phi);
        let mut added = 0.0;
        if sourced {
            let g = &mut self.fluid.grid;
            // The budget takes what the sources describe (the swept plug and
            // the drops' cover), as the 3D budget takes flow rate and drop
            // volume: the surface they leave is held to it by volume control.
            for i in 0..n {
                let cover = sources.drop.get(i).copied().unwrap_or(0.0)
                    + sources.plug.get(i).copied().unwrap_or(0.0);
                added += (cover * g.capacity[i]) as f64;
            }
            self.budget += added;
            let nxv = g.dims[0] + 1;
            for y in 0..=g.dims[1] {
                for x in 0..=g.dims[0] {
                    if g.buried([x as i32, y as i32]) {
                        continue;
                    }
                    let p = [x as f32, y as f32];
                    let phi = g.source_phi(p, g.phi[x + nxv * y]);
                    g.phi[x + nxv * y] = plug.as_ref().map_or(phi, |plug| plug.stream_phi(p, phi));
                }
            }
        }
        self.surface.redistance(&mut self.fluid.grid, &self.layout);
        self.refresh();

        self.transport(&e, dt);
        self.receipt.census =
            self.particles
                .classify(&self.fluid.grid, &self.liquid, &self.surface);
        self.particles.transfer(
            &mut self.fluid.grid,
            &self.surface,
            &self.layout,
            &mut self.cache,
        );
        self.layout
            .resolve(&mut self.fluid.grid.velocity, &mut self.cache);
        let mut e = Extension::build_with(
            &self.fluid.grid,
            &self.solver,
            dt,
            &Sources::default(),
            e.values,
        );
        self.mix(&mut e);
        self.particles.snapshot(&e);
        let nozzle = self
            .fluid
            .inflow
            .as_ref()
            .and_then(|source| Velocity::new(source, time, dt, &self.fluid.grid));
        self.force(&e, nozzle.as_ref(), dt);
        self.project(nozzle.as_ref(), rate, dt);
        let mut e = Extension::build_with(
            &self.fluid.grid,
            &self.solver,
            dt,
            &Sources::default(),
            e.values,
        );
        self.mix(&mut e);
        self.particles.update(
            &self.fluid.grid,
            &e,
            &self.surface,
            self.fluid.gravity[1],
            dt,
        )?;
        self.particles.resample(&self.fluid.grid);
        {
            let g = &self.fluid.grid;
            let h = g.h[0].min(g.h[1]);
            let source = |p: [f32; 2]| {
                let phi = g.source_phi(p, h);
                plug.as_ref().map_or(phi, |plug| plug.stream_phi(p, phi)) / h
            };
            self.particles
                .seed(g, &e, &self.surface, false, sourced, &source);
        }
        self.fluid.grid.drops.clear();
        self.fluid.packed = e.values;

        let g = &self.fluid.grid;
        let finite = |s: &[f32]| s.iter().fold(true, |ok, v| ok & v.is_finite());
        if !(finite(&g.phi) && finite(g.velocity.as_flattened())) {
            return Err(ValidationError("non-finite narrow-band FLIP state".into()));
        }
        self.receipt.frame += 1;
        self.receipt.time += dt as f64;
        self.receipt.pressure = self.root.pressure.receipt.clone();
        self.receipt.pressure.residual *= 4.0;
        self.receipt.pressure.initial_residual *= 4.0;
        self.receipt.band = self.band.receipt.clone();
        self.receipt.fine_tiles = self.layout.fine_count();
        self.receipt.budget = self.budget;
        self.receipt.control_rate = rate;
        self.receipt.injected_volume = added;
        self.receipt.outflow_volume = left;
        self.receipt.particles = self.particles.list.len() as u32;
        self.receipt.reseed_clipped = self.particles.clipped;
        self.receipt.max_speed = g
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0_f32, f32::max);
        Ok(())
    }
}
