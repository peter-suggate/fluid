//! Uniform numerical advance. No adaptive graph participates in the solve.
use super::{
    extension::{Extension, TwoLevel},
    grid::Grid,
    inflow::Plug,
    options::UniformGeometricOptions,
    pages::Window,
    pressure::{cycle_budget, theta, Pressure, Scratch, FREE},
    surface, surface_volume,
    transport::{Sources, Transport, TransportReceipt},
};
use crate::{
    initial_liquid::{
        base_initial_liquid_fraction_at_cell, initial_liquid_fraction_at_cell,
        initial_liquid_surface_scalar,
    },
    initial_scene::{lattice_dimensions, SceneDocument},
    scene_model::Vec3,
    solid_world::fluid_solid_world_for_scene,
    types::ValidationError,
};
#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub frame: u32,
    pub time: f64,
    pub transport: TransportReceipt,
    pub sharpening_dust: f64,
    pub pressure: super::pressure::PressureReceipt,
    pub volume: f64,
    pub phi_area: f64,
    pub contour_area: f64,
    pub contour_l1: f64,
    pub max_speed: f32,
    pub injected_volume: f64,
    /// uv.wgsl uvBalanceReduce surface-deficit rate for this step.
    pub balance_rate: f32,
    /// Total surface volume state: [shift, half-range, target V, prior surface V].
    pub surface_volume: [f32; 4],
}
pub struct World {
    pub grid: Grid,
    pub options: UniformGeometricOptions,
    pub pressure: Pressure,
    pub receipt: Receipt,
    /// Allocated only by diagnostic scene runners.
    pub advected_phi_audit: Option<Vec<f32>>,
    pub tile_classes: Vec<u8>,
    /// Lagged pressure demand (executed cycles, converged) of the last solve.
    pressure_demand: Option<(usize, bool)>,
    /// Diagnostic replay of a GPU-encoded cycle budget for the next solve only.
    pub pressure_budget_override: Option<usize>,
    transport: Transport,
    pub inflow: Option<crate::scene_model::FluidInflow>,
    /// The phi solve window and its dense-census schedule.
    pub window: Window,
    /// A published phi region header for the next step only (a GPU oracle's).
    pub phi_region_override: Option<Vec<u32>>,
    /// The post-step surface targets (gammaA).
    pub gamma: Vec<f32>,
    /// Persistent GPU intermediates: the packed extension field (stale outside
    /// SHELL) and the advected-phi scratch (stale outside the window).
    packed: Vec<[f32; 2]>,
    advected: Vec<f32>,
    /// Transport scratch: live receivers, their departures, the gathered V.
    cells: Vec<u32>,
    departures: Vec<[f32; 2]>,
    gathered: Vec<f32>,
    pub(super) gravity: [f32; 2],
    pub(super) rho: f32,
    viscosity: f32,
    sigma: f32,
}
impl World {
    pub fn from_grid(
        grid: Grid,
        options: UniformGeometricOptions,
        gravity: [f32; 2],
        rho: f32,
        viscosity: f32,
        sigma: f32,
    ) -> Result<Self, ValidationError> {
        options.validate()?;
        super::validate_supported(&options)?;
        if grid.h.iter().any(|h| !h.is_finite() || *h <= 0.0) || !rho.is_finite() || rho <= 0.0 {
            return Err(ValidationError("invalid uniform physical scale".into()));
        }
        let count = grid.dims[0]
            .checked_mul(grid.dims[1])
            .ok_or_else(|| ValidationError("uniform grid size overflow".into()))?;
        let vertices = grid.dims[0]
            .checked_add(1)
            .and_then(|x| grid.dims[1].checked_add(1).and_then(|y| x.checked_mul(y)))
            .ok_or_else(|| ValidationError("uniform vertex size overflow".into()))?;
        if grid
            .dims
            .iter()
            .any(|&n| n < 2 || n > i32::MAX as usize - 2)
            || grid.volume.len() != count
            || grid.capacity.len() != count
            || grid.velocity.len() != count
            || grid.released.len() != count
            || grid.phi.len() != vertices
            || grid.low_x.len() != grid.dims[1]
            || grid.low_y.len() != grid.dims[0]
        {
            return Err(ValidationError(
                "uniform field dimensions do not match the grid".into(),
            ));
        }
        if grid
            .volume
            .iter()
            .chain(&grid.phi)
            .chain(grid.velocity.iter().flatten())
            .chain(&grid.low_x)
            .chain(&grid.low_y)
            .chain(&gravity)
            .any(|v| !v.is_finite())
            || grid
                .capacity
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !viscosity.is_finite()
            || viscosity < 0.0
            || !sigma.is_finite()
            || sigma < 0.0
        {
            return Err(ValidationError(
                "invalid uniform field or material value".into(),
            ));
        }
        let pressure = Pressure::new(grid.dims, grid.h)?;
        let transport = Transport::new(grid.volume.len());
        Ok(Self {
            window: Window::new(grid.dims, &options),
            phi_region_override: None,
            gamma: vec![0.0; count],
            packed: vec![[0.0; 2]; count],
            advected: grid.phi.clone(),
            cells: Vec::new(),
            departures: Vec::new(),
            gathered: vec![0.0; count],
            grid,
            options,
            pressure,
            transport,
            inflow: None,
            gravity,
            rho,
            viscosity,
            sigma,
            receipt: Receipt::default(),
            advected_phi_audit: None,
            pressure_demand: None,
            pressure_budget_override: None,
            tile_classes: Vec::new(),
        })
    }
    pub fn from_document(
        scene: &SceneDocument,
        options: UniformGeometricOptions,
    ) -> Result<Self, ValidationError> {
        if !scene.rigid_bodies.is_empty() {
            return Err(ValidationError(
                "uniform 2D scene adapter: rigid bodies are not yet ported".into(),
            ));
        }
        let dims = lattice_dimensions(scene);
        let h = [
            scene.container.width_m as f32 / dims[0] as f32,
            scene.container.height_m as f32 / dims[1] as f32,
        ];
        let mut grid = Grid::new(
            [dims[0] as usize, dims[1] as usize],
            h,
            scene.container.top == "open",
        )?;
        let solid = fluid_solid_world_for_scene(scene);
        let z = (dims[2] / 2) as i32;
        for i in 0..grid.volume.len() {
            let p = grid.point(i);
            let q = [p[0], p[1], z];
            grid.capacity[i] = if solid.sample(q).resident_solid() {
                0.0
            } else {
                1.0
            };
            let base = base_initial_liquid_fraction_at_cell(scene, q, dims);
            grid.volume[i] =
                initial_liquid_fraction_at_cell(scene, q, dims, base) as f32 * grid.capacity[i];
            if let Some(v) = scene.fluid.initial_velocity_m_s {
                grid.velocity[i] = [v.x as f32, v.y as f32];
            }
        }
        // uniform-volume-initial.ts buriedVertices: a vertex whose every incident
        // cell is closed (any solid fraction) is air by construction.
        let closed: Vec<bool> = (0..grid.volume.len())
            .map(|i| {
                let p = grid.point(i);
                solid.sample([p[0], p[1], z]).solid_fraction > 0.0
            })
            .collect();
        let empty = scene
            .container
            .width_m
            .max(scene.container.height_m)
            .max(scene.container.depth_m) as f32;
        for y in 0..=grid.dims[1] {
            for x in 0..=grid.dims[0] {
                let buried = (0..4).all(|k| {
                    grid.index([x as i32 - 1 + (k & 1), y as i32 - 1 + (k >> 1)])
                        .is_none_or(|i| closed[i])
                });
                if buried {
                    grid.phi[x + (grid.dims[0] + 1) * y] = empty;
                    continue;
                }
                let point = Vec3 {
                    x: (x as f64 / dims[0] as f64 - 0.5) * scene.container.width_m,
                    y: y as f64 / dims[1] as f64 * scene.container.height_m,
                    z: 0.0,
                };
                grid.phi[x + (grid.dims[0] + 1) * y] =
                    initial_liquid_surface_scalar(scene, point, dims) as f32;
            }
        }
        let mut world = Self::from_grid(
            grid,
            options,
            [
                scene.fluid.gravity_m_s2.x as f32,
                scene.fluid.gravity_m_s2.y as f32,
            ],
            scene.fluid.density_kg_m3 as f32,
            scene.fluid.dynamic_viscosity_pa_s as f32,
            scene.fluid.surface_tension_n_m as f32,
        )?;
        world.inflow = scene.fluid.inflow;
        Ok(world)
    }
    pub fn advance(&mut self, dt: f32) -> Result<(), ValidationError> {
        self.advance_observed(dt, 8, |_, _| {})
    }

    /// Diagnostic stage observer. Observations are read-only and absent from the
    /// ordinary advance; iteration overrides are not shared numerical defaults.
    pub fn advance_observed(
        &mut self,
        dt: f32,
        sharpening_rounds: usize,
        mut observe: impl FnMut(&str, &Grid),
    ) -> Result<(), ValidationError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ValidationError("invalid uniform timestep".into()));
        }
        let plug = self
            .inflow
            .as_ref()
            .map(|source| Plug::new(source, self.receipt.time, dt, &self.grid));
        observe("start", &self.grid);
        let n = self.grid.volume.len();
        let mut sources = Sources::default();
        if !self.grid.drops.is_empty() {
            sources.drop = (0..n)
                .map(|i| self.grid.drop_fraction(self.grid.point(i)))
                .collect();
        }
        if let Some(source) = plug.as_ref().filter(|s| s.active()) {
            sources.plug = (0..n).map(|i| source.amount(self.grid.point(i))).collect();
        }
        // Head of step: phi census, then the tile classes and extension.
        let two_level = TwoLevel::new(self.grid.dims, &self.options);
        let published = self.phi_region_override.take();
        let window = self.window.census(
            &self.grid,
            &self.options,
            dt,
            self.gravity[1],
            two_level.shell_reach,
            two_level.enabled.then_some(two_level.fine_reach),
            &sources,
            plug.as_ref(),
            published.as_deref(),
        );
        let extension = Extension::build_with(
            &self.grid,
            &self.options,
            dt,
            &sources,
            std::mem::take(&mut self.packed),
            None,
        );
        self.packed.clone_from(&extension.values);
        self.tile_classes.clone_from(&extension.classes);
        extension.transport_cells(&mut self.cells);
        self.departures.clear();
        self.departures.extend(self.cells.iter().map(|&i| {
            let p = self.grid.point(i as usize);
            extension.trace(&self.grid, [p[0] as f32 + 0.5, p[1] as f32 + 0.5], dt)
        }));
        surface::advect_window(
            &self.grid,
            &extension,
            dt,
            window,
            plug.as_ref(),
            &self.options,
            &mut self.advected,
        );
        std::mem::swap(&mut self.grid.phi, &mut self.advected);
        observe("advected", &self.grid);
        if let Some(audit) = &mut self.advected_phi_audit {
            audit.clone_from(&self.grid.phi);
        }
        std::mem::swap(&mut self.grid.phi, &mut self.advected);
        if self.options.redistance == "on" {
            surface::redistance_window(&mut self.grid, &self.advected, window);
        } else {
            self.grid.phi.copy_from_slice(&self.advected);
        }
        observe("redistanced", &self.grid);
        self.receipt.transport = self.transport.advance(
            self.grid.dims,
            &self.departures,
            &self.grid.capacity,
            &self.grid.volume,
            self.options.volume_dust_threshold,
            &self.cells,
            &sources,
            &mut self.gathered,
        )?;
        self.receipt.injected_volume = self.receipt.transport.injected_volume;
        self.grid.drops.clear();
        std::mem::swap(&mut self.grid.volume, &mut self.gathered);
        let corrected = self.options.total_surface_volume == "on";
        // uvGather's gamma: zero outside the live set and in sealed cells.
        // TSV rewrites every gammaA before any read.
        if !corrected {
            self.gamma.fill(0.0);
            for &i in &self.cells {
                let i = i as usize;
                if self.grid.capacity[i] > 0.0 {
                    self.gamma[i] = self.grid.target(self.grid.point(i));
                }
            }
        }
        observe("transported", &self.grid);
        if corrected {
            self.receipt.surface_volume = surface_volume::correct(&mut self.grid);
            for (y, row) in self.gamma.chunks_mut(self.grid.dims[0]).enumerate() {
                for (x, gamma) in row.iter_mut().enumerate() {
                    *gamma = self.grid.target([x as i32, y as i32]);
                }
            }
            observe("corrected", &self.grid);
        }
        self.receipt.sharpening_dust = surface::sharpen(
            &mut self.grid,
            &self.options,
            &self.gamma,
            sharpening_rounds,
        );
        observe("sharpened", &self.grid);
        let nozzle = self.inflow.as_ref().and_then(|source| {
            super::inflow::Velocity::new(source, self.receipt.time, dt, &self.grid)
        });
        self.advect_velocity(&extension, nozzle.as_ref(), dt, &mut observe);
        observe("forced", &self.grid);
        self.project(&extension, nozzle.as_ref(), dt);
        observe("projected", &self.grid);
        self.receipt.frame += 1;
        self.receipt.time += dt as f64;
        self.receipt.volume = self.grid.volume.iter().map(|&v| v as f64).sum();
        // Nothing after TSV moves phi or capacity: gammaA is this state's target.
        self.receipt.phi_area = if corrected {
            self.gamma.iter().map(|&v| v as f64).sum()
        } else {
            (0..self.grid.volume.len())
                .map(|i| self.grid.target(self.grid.point(i)) as f64)
                .sum()
        };
        self.receipt.contour_area = 0.0;
        self.receipt.contour_l1 = 0.0;
        // A cell with no V and every corner outside adds exactly zero to both.
        let (dims, phi) = (self.grid.dims, &self.grid.phi);
        for (y, row) in self.grid.volume.chunks(dims[0]).enumerate() {
            for (x, &volume) in row.iter().enumerate() {
                let j = x + (dims[0] + 1) * y;
                let corners = [j, j + 1, j + dims[0] + 1, j + dims[0] + 2];
                if volume == 0.0 && corners.iter().all(|&k| phi[k] > 0.0) {
                    continue;
                }
                let i = x + dims[0] * y;
                let area = super::diagnostics::contour_fill(&self.grid, i, 0.0) as f64;
                self.receipt.contour_area += area;
                self.receipt.contour_l1 += (area - volume as f64).abs();
            }
        }
        self.receipt.max_speed = self
            .grid
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0_f32, f32::max);
        self.receipt.pressure = self.pressure.receipt.clone();
        let finite = |s: &[f32]| s.iter().fold(true, |ok, v| ok & v.is_finite());
        let g = &self.grid;
        if !(finite(&g.volume) && finite(&g.phi) && finite(g.velocity.as_flattened())) {
            return Err(ValidationError("non-finite uniform state".into()));
        }
        Ok(())
    }
    // reference.wgsl surfaceOccupancy.
    fn occupancy(&self, p: [i32; 2]) -> f32 {
        self.grid.index(p).map_or(0.0, |_| {
            (0.5 - self.grid.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5])
                / (4.0 * self.grid.h[1]))
                .clamp(0.0, 1.0)
        })
    }
    // reference.wgsl normalSurfaceOccupancy.
    fn normal_occupancy(&self, p: [i32; 2]) -> f32 {
        let g = &self.grid;
        if g.index(p).is_some() {
            self.occupancy(p)
        } else if super::velocity::solid_voxel(g, p) {
            self.occupancy(g.clamp_cell(p))
        } else {
            0.0
        }
    }
    // reference.wgsl interfaceNormal.
    fn normal(&self, p: [i32; 2]) -> [f32; 2] {
        let gradient: [f32; 2] = std::array::from_fn(|a| {
            let mut lo = p;
            let mut hi = p;
            lo[a] -= 1;
            hi[a] += 1;
            (self.normal_occupancy(hi) - self.normal_occupancy(lo)) / (2.0 * self.grid.h[a])
        });
        let length = (gradient[0] * gradient[0] + gradient[1] * gradient[1])
            .sqrt()
            .max(1e-6);
        gradient.map(|v| v / length)
    }
    // reference.wgsl curvatureAt.
    fn curvature(&self, p: [i32; 2]) -> f32 {
        let terms: [f32; 2] = std::array::from_fn(|a| {
            let mut lo = p;
            let mut hi = p;
            lo[a] -= 1;
            hi[a] += 1;
            (self.normal(hi)[a] - self.normal(lo)[a]) / (2.0 * self.grid.h[a])
        });
        -(terms[0] + terms[1])
    }
    fn advect_velocity(
        &mut self,
        e: &Extension,
        nozzle: Option<&super::inflow::Velocity>,
        dt: f32,
        observe: &mut impl FnMut(&str, &Grid),
    ) {
        let mut next = std::mem::take(&mut self.pressure.scratch.velocity);
        super::velocity::advect(&self.grid, e, &self.options, dt, &mut next);
        std::mem::swap(&mut self.grid.velocity, &mut next);
        observe("velocityAdvected", &self.grid);
        std::mem::swap(&mut self.grid.velocity, &mut next);
        let g = &self.grid;
        let molecular = self.viscosity / self.rho;
        let sigma_over_rho = self.sigma / self.rho;
        super::velocity::for_each_fine(g, e, &self.options, |i, p| {
            let v = &mut next[i];
            // reference.wgsl applyVelocityForces. The walls are free-slip.
            let occupancy = self.occupancy(p);
            if occupancy > 0.0 && molecular > 0.0 {
                let sample = |q: [i32; 2]| g.velocity[g.index(g.clamp_cell(q)).unwrap()];
                let centre = g.velocity[i];
                let terms: [[f32; 2]; 2] = std::array::from_fn(|a| {
                    let mut lo = p;
                    let mut hi = p;
                    lo[a] -= 1;
                    hi[a] += 1;
                    std::array::from_fn(|c| {
                        ((sample(hi)[c] - 2.0 * centre[c]) + sample(lo)[c]) / (g.h[a] * g.h[a])
                    })
                });
                for c in 0..2 {
                    v[c] += dt * molecular * (terms[0][c] + terms[1][c]);
                }
            }
            let above = [p[0], p[1] + 1];
            if occupancy > 1e-5
                || self.occupancy(above) > 1e-5
                || g.airborne(
                    p,
                    self.options.airborne_momentum == "on",
                    self.options.volume_dust_threshold,
                )
                || g.airborne(
                    above,
                    self.options.airborne_momentum == "on",
                    self.options.volume_dust_threshold,
                )
            {
                v[1] += self.gravity[1] * dt;
            }
            if sigma_over_rho > 0.0 {
                let difference: [f32; 2] = std::array::from_fn(|a| {
                    let mut q = p;
                    q[a] += 1;
                    if g.index(q).is_some() {
                        self.occupancy(q) - occupancy
                    } else {
                        0.0
                    }
                });
                if difference[0] != 0.0 || difference[1] != 0.0 {
                    let centre = self.curvature(p);
                    for a in 0..2 {
                        if difference[a] != 0.0 {
                            let mut q = p;
                            q[a] += 1;
                            v[a] += dt
                                * sigma_over_rho
                                * 0.5
                                * (centre + self.curvature(q))
                                * difference[a]
                                / g.h[a];
                        }
                    }
                }
            }
            if let Some(nozzle) = nozzle {
                *v = nozzle.apply_swept(p, *v);
            }
        });
        std::mem::swap(&mut self.grid.velocity, &mut next);
        self.pressure.scratch.velocity = next;
    }
    pub(super) fn project(
        &mut self,
        e: &Extension,
        nozzle: Option<&super::inflow::Velocity>,
        dt: f32,
    ) {
        let mut s = std::mem::take(&mut self.pressure.scratch);
        let g = &self.grid;
        let rows = self.options.volume_pressure_rows.as_str();
        let dims = self.pressure.levels[0].dims;
        let halo = |p: [i32; 2]| (p[0] + 1) as usize + dims[0] * (p[1] + 1) as usize;
        // Pressure phi once per haloed cell, and uvTarget once per open
        // liquid cell for the balance and the volume correction.
        s.pressure_phi.clear();
        for y in 0..dims[1] as i32 {
            for x in 0..dims[0] as i32 {
                s.pressure_phi.push(g.pressure_phi([x - 1, y - 1], rows));
            }
        }
        s.target.clear();
        s.target.resize(g.volume.len(), 0.0);
        for (i, target) in s.target.iter_mut().enumerate() {
            let p = g.point(i);
            if g.capacity[i] > 1e-5 && s.pressure_phi[halo(p)] < 0.0 {
                *target = g.target(p);
            }
        }
        let rate = if self.options.surface_deficit_balancing == "on" {
            balance_rate(g, &mut s)
        } else {
            0.0
        };
        self.receipt.balance_rate = rate;
        let check_solid = g.rigid_speeds.is_some();
        let fine = &mut self.pressure.levels[0];
        let interior = |a: usize, v: usize| v > 0 && v < dims[a] - 1;
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                // mg.wgsl mgBuildFinestTopology and mgBuildFinestRhs. The coarse
                // pyramid downsamples this raw phi before any continuation.
                let i = x + dims[0] * y;
                let p = [x as i32 - 1, y as i32 - 1];
                let mut rhs = 0.0;
                if let Some(j) = g.index(p) {
                    let open = g.capacity[j];
                    let phi = s.pressure_phi[i];
                    fine.phi[i] = phi;
                    fine.topology[i] = [open, g.pressure_face(p, 0), g.pressure_face(p, 1)];
                    // Cell-centre solid proxy for cellInsideSolid.
                    let inside_solid =
                        open <= 1e-5 || g.rigid_centres.as_ref().is_some_and(|v| v[j]);
                    fine.minimum[i] = if inside_solid { 0.0 } else { FREE };
                    if phi < 0.0 {
                        rhs = self.rho
                            * (divergence(g, p, check_solid, open)
                                - volume_correction(g, &s.target, j, open, phi, rate, dt))
                            / dt;
                    }
                } else {
                    let open_top = g.open_top && y == dims[1] - 1 && interior(0, x);
                    let mut topology = [if open_top { 1.0 } else { 0.0 }; 3];
                    // Low-side halo cells own the missing negative dual faces.
                    if x == 0 && interior(1, y) {
                        topology[1] = g.pressure_face(p, 0);
                    }
                    if y == 0 && interior(0, x) {
                        topology[2] = g.pressure_face(p, 1);
                    }
                    fine.phi[i] = 0.5 * g.h[0].min(g.h[1]);
                    fine.topology[i] = topology;
                    fine.minimum[i] = if open_top { FREE } else { 0.0 };
                    // The solid halo is a constrained row: its wall flux enters b.
                    if !open_top && s.pressure_phi[i] < 0.0 {
                        rhs = self.rho * divergence(g, p, false, g.open(p)) / dt;
                    }
                }
                fine.rhs[i] = -rhs;
                fine.p[i] = 0.0;
            }
        }
        let budget = self
            .pressure_budget_override
            .take()
            .unwrap_or_else(|| cycle_budget(&self.options, self.pressure_demand));
        self.pressure
            .solve(&self.options, dt, self.rho, g.open_top, budget);
        self.pressure_demand = Some((
            self.pressure.receipt.cycles,
            self.pressure.receipt.converged,
        ));
        let fine = &self.pressure.levels[0];
        let pressure_phi = &s.pressure_phi;
        let pressure = |p: [i32; 2]| -> f32 {
            let i = halo(p);
            if pressure_phi[i] >= 0.0 {
                0.0
            } else {
                fine.p[i]
            }
        };
        // reference.wgsl geometricProjectedFace.
        let project = |p: [i32; 2], a: usize, v: f32| {
            let mut q = p;
            q[a] += 1;
            if g.pressure_face(p, a) <= 1e-6 {
                return g.solid_speed(p, a);
            }
            let pa = pressure_phi[halo(p)];
            let pb = pressure_phi[halo(q)];
            if pa >= 0.0 && pb >= 0.0 {
                return if g.airborne(
                    p,
                    self.options.airborne_momentum == "on",
                    self.options.volume_dust_threshold,
                ) || g.airborne(
                    q,
                    self.options.airborne_momentum == "on",
                    self.options.volume_dust_threshold,
                ) {
                    v
                } else {
                    0.0
                };
            }
            v - dt / self.rho * (pressure(q) - pressure(p)) / (g.h[a] * theta(pa, pb))
        };
        // E2b: a cell outside the fine tiles has no row and no liquid.
        let n = g.volume.len();
        let velocities = &mut s.velocity;
        let low_x = &mut s.low_x;
        let low_y = &mut s.low_y;
        let released = &mut s.released;
        velocities.clear();
        velocities.resize(n, [0.0; 2]);
        low_x.clear();
        low_x.resize(g.dims[1], 0.0);
        low_y.clear();
        low_y.resize(g.dims[0], 0.0);
        released.clear();
        released.resize(n, 0);
        super::velocity::for_each_fine(g, e, &self.options, |i, p| {
            for a in 0..2 {
                let v = project(p, a, g.velocity[i][a]);
                velocities[i][a] = v;
                let mut q = p;
                q[a] += 1;
                let own = g.open(p) > 1e-5;
                let other = g.open(q) > 1e-5;
                if own != other {
                    let solid = if own { q } else { p };
                    let inward = if own { -1.0 } else { 1.0 };
                    if g.pressure_face(p, a) > 1e-6
                        && pressure(solid) <= 0.0
                        && inward
                            * (if g.rigid_speeds.is_some() {
                                v - g.solid_speed(p, a)
                            } else {
                                v
                            })
                            * dt
                            > 1e-4 * g.h[a]
                    {
                        released[i] |= 1 << a;
                    }
                }
                if p[a] == 0 {
                    let mut halo = p;
                    halo[a] -= 1;
                    let v = project(halo, a, g.face(halo, a));
                    if a == 0 {
                        low_x[p[1] as usize] = v;
                    } else {
                        low_y[p[0] as usize] = v;
                    }
                    if own
                        && g.pressure_face(halo, a) > 1e-6
                        && pressure(halo) <= 0.0
                        && v * dt > 1e-4 * g.h[a]
                    {
                        released[i] |= 1 << (a + 2);
                    }
                }
            }
            if let Some(nozzle) = nozzle {
                velocities[i] = nozzle.apply(p, velocities[i]);
            }
        });
        std::mem::swap(&mut self.grid.velocity, &mut s.velocity);
        std::mem::swap(&mut self.grid.low_x, &mut s.low_x);
        std::mem::swap(&mut self.grid.low_y, &mut s.low_y);
        std::mem::swap(&mut self.grid.released, &mut s.released);
        self.pressure.scratch = s;
    }
}

// reference.wgsl divergenceAtWithCapacity: wall-relative flux plus
// vi * div(u_s), both in s^-1.
fn divergence(g: &Grid, p: [i32; 2], check_solid: bool, vi: f32) -> f32 {
    let solid = |q: [i32; 2], a: usize| {
        let mut r = q;
        r[a] += 1;
        if check_solid && g.index(q).is_some() && g.index(r).is_some() {
            g.solid_speed(q, a)
        } else {
            0.0
        }
    };
    let mut terms = [0.0; 4];
    for a in 0..2 {
        let mut m = p;
        m[a] -= 1;
        let vp = g.pressure_face(p, a);
        let vm = g.pressure_face(m, a);
        terms[2 * a] = (vp * g.face(p, a) + (vi - vp) * solid(p, a)) / g.h[a];
        terms[2 * a + 1] = -(vm * g.face(m, a) + (vi - vm) * solid(m, a)) / g.h[a];
    }
    (terms[0] + terms[1]) + (terms[2] + terms[3])
}

// reference.wgsl volumeCorrectionDivergenceFromAuthority. The deficit target
// is the sharpened surface fraction gammaA.
fn volume_correction(
    g: &Grid,
    target: &[f32],
    i: usize,
    cap: f32,
    phi: f32,
    rate: f32,
    dt: f32,
) -> f32 {
    let v = g.volume[i];
    let positive = (0.5 * (v - cap).max(0.0)).min(cap);
    let deficit = if cap > 1e-5 && v <= cap && phi < 0.0 {
        (target[i] - v).max(0.0)
    } else {
        0.0
    };
    (positive - rate * deficit) / dt.max(1e-12)
}

// uv.wgsl uvBalanceSum: a 64-lane workgroup tree.
fn workgroup_sum(mut lanes: [[f32; 2]; 64]) -> [f32; 2] {
    let mut stride = 32;
    while stride > 0 {
        for l in 0..stride {
            let upper = lanes[l + stride];
            for (value, add) in lanes[l].iter_mut().zip(upper) {
                *value += add;
            }
        }
        stride /= 2;
    }
    lanes[0]
}

// uv.wgsl uvBalanceReduce: lane l accumulates records l, l+64, ... in order.
fn lane_reduce(records: &[[f32; 2]]) -> [f32; 2] {
    let mut lanes = [[0.0_f32; 2]; 64];
    for (i, record) in records.iter().enumerate() {
        for c in 0..2 {
            lanes[i % 64][c] += record[c];
        }
    }
    workgroup_sum(lanes)
}

// uv.wgsl uvBalanceMeasure with its reductions. One record per 4x4 tile, lane
// x + 4y; above 1024 records the chunk pass reduces each 1024 first. A tile
// with no liquid lane sums zeros.
fn balance_rate(g: &Grid, s: &mut Scratch) -> f32 {
    let tiles = g.dims.map(|n| n.div_ceil(4));
    let halo = g.dims[0] + 2;
    s.records.clear();
    for ty in 0..tiles[1] {
        for tx in 0..tiles[0] {
            let mut lanes = [[0.0_f32; 2]; 64];
            let mut live = false;
            for (l, lane) in lanes.iter_mut().enumerate().take(16) {
                let p = [(4 * tx + l % 4) as i32, (4 * ty + l / 4) as i32];
                let Some(i) = g.index(p) else { continue };
                let cap = g.capacity[i];
                if cap > 1e-5
                    && s.pressure_phi[(p[0] + 1) as usize + halo * (p[1] + 1) as usize] < 0.0
                {
                    let v = g.volume[i];
                    // uv.wgsl uvSurfaceDeficit.
                    let deficit = if v > cap {
                        0.0
                    } else {
                        (s.target[i] - v).max(0.0)
                    };
                    *lane = [(0.5 * (v - cap).max(0.0)).min(cap), deficit];
                    live = true;
                }
            }
            s.records
                .push(if live { workgroup_sum(lanes) } else { [0.0; 2] });
        }
    }
    let total = if s.records.len() > 1024 {
        s.chunks.clear();
        s.chunks.extend(s.records.chunks(1024).map(lane_reduce));
        lane_reduce(&s.chunks)
    } else {
        lane_reduce(&s.records)
    };
    // The Metal build divides by multiplying with the reciprocal.
    if total[1] > 0.0 {
        (total[0] * (1.0 / total[1])).min(1.0)
    } else {
        0.0
    }
}
