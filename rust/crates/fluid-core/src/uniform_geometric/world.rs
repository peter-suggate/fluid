//! Uniform numerical advance. No adaptive graph participates in the solve.
use super::{
    extension::Extension,
    grid::Grid,
    options::UniformGeometricOptions,
    pressure::{theta, Pressure},
    surface,
    transport::{Transport, TransportReceipt},
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
    pub swept_extension: super::swept_extension::Receipt,
}
pub struct World {
    pub swept_extension: super::swept_extension::Config,
    pub grid: Grid,
    pub options: UniformGeometricOptions,
    pub pressure: Pressure,
    pub receipt: Receipt,
    /// Allocated only by diagnostic scene runners.
    pub advected_phi_audit: Option<Vec<f32>>,
    pub tile_classes: Vec<u8>,
    /// Opt-in pressure-source ablation for the diagnostic runner; never a lab default.
    pub diagnostic_disable_overfill_correction: bool,
    pub energy_experiment: super::energy_experiment::Config,
    pub energy_receipt: super::energy_experiment::Receipt,
    transport: Transport,
    pub inflow: Option<crate::scene_model::FluidInflow>,
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
        if options.active_region == "on" {
            return Err(ValidationError("uniform 2D solve-window scheduling is not yet ported; use an explicit whole-domain reference profile".into()));
        }
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
            swept_extension: if options.total_surface_volume == "on" {
                super::swept_extension::Config::lab_profile("area-only")?
            } else {
                super::swept_extension::Config::default()
            },
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
            diagnostic_disable_overfill_correction: false,
            energy_experiment: Default::default(),
            energy_receipt: Default::default(),
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
        for y in 0..=grid.dims[1] {
            for x in 0..=grid.dims[0] {
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
        let inflow = self
            .inflow
            .as_ref()
            .and_then(|source| super::inflow::Step::new(source, self.receipt.time, dt, &self.grid));
        observe("start", &self.grid);
        let previous_energy = if self.energy_experiment.mode == "energy-cap" {
            super::energy_experiment::energy(&self.grid, self.rho, self.gravity)
        } else {
            (0.0, 0.0)
        };
        let phi_reference = (self.swept_extension.mode
            == super::swept_extension::Mode::TransportAgreement)
            .then(|| {
                (0..self.grid.volume.len())
                    .map(|i| super::swept_extension::contour_fill(&self.grid, i, 0.0))
                    .collect::<Vec<_>>()
            });
        let prior_phase = super::velocity::phase(&self.grid, &self.options);
        let (extension, correction) =
            super::swept_extension::build(&self.grid, &self.options, dt, &self.swept_extension);
        self.receipt.swept_extension = correction;
        self.tile_classes.clone_from(&extension.classes);
        let active: Vec<_> = (0..self.grid.volume.len())
            .map(|i| extension.transport_at(self.grid.point(i), &self.options))
            .collect();
        let departures: Vec<_> = (0..self.grid.volume.len())
            .map(|i| {
                if !active[i] {
                    return [0.0; 2];
                }
                let p = self.grid.point(i);
                extension.trace(&self.grid, [p[0] as f32 + 0.5, p[1] as f32 + 0.5], dt)
            })
            .collect();
        surface::advect(&mut self.grid, &extension, &self.options, dt);
        observe("advected", &self.grid);
        if let Some(audit) = &mut self.advected_phi_audit {
            audit.clone_from(&self.grid.phi);
        }
        if self.options.redistance == "on" {
            surface::redistance(&mut self.grid);
        }
        observe("redistanced", &self.grid);
        let mut next = vec![0.0; self.grid.volume.len()];
        self.receipt.transport = self.transport.advance(
            self.grid.dims,
            &departures,
            &self.grid.capacity,
            &self.grid.volume,
            self.options.volume_dust_threshold,
            &active,
            &mut next,
        )?;
        self.receipt.injected_volume = 0.0;
        if !self.grid.drops.is_empty() {
            for i in 0..next.len() {
                let added = self
                    .grid
                    .drop_fraction(self.grid.point(i))
                    .min((self.grid.capacity[i] - next[i]).max(0.0));
                next[i] += added;
                self.receipt.injected_volume += added as f64;
            }
            self.grid.drops.clear();
        }
        self.grid.volume = next;
        if let Some(source) = &inflow {
            self.receipt.injected_volume += source.inject(&mut self.grid);
        }
        observe("transported", &self.grid);
        if let Some(reference) = phi_reference.filter(|_| self.receipt.injected_volume == 0.0) {
            let transported: Vec<f32> = self
                .transport
                .edges
                .iter()
                .enumerate()
                .map(|(i, edge)| {
                    if !active[i] {
                        return 0.0;
                    }
                    (0..5)
                        .map(|k| edge.weights[k] * reference[edge.donors[k]])
                        .sum()
                })
                .collect();
            self.receipt.swept_extension.surface = Some(super::swept_extension::correct_surface(
                &mut self.grid,
                &transported,
                &self.swept_extension,
            ));
            self.receipt.swept_extension.accepted = true;
        }
        if self.swept_extension.mode == super::swept_extension::Mode::RegionalVolume
            && self.receipt.injected_volume == 0.0
        {
            let reference = self.grid.volume.clone();
            self.receipt.swept_extension.surface = Some(super::swept_extension::correct_surface(
                &mut self.grid,
                &reference,
                &self.swept_extension,
            ));
            self.receipt.swept_extension.accepted = true;
        }
        if self.receipt.swept_extension.surface.is_some() {
            observe("corrected", &self.grid);
        }
        self.receipt.sharpening_dust =
            surface::sharpen_rounds(&mut self.grid, &self.options, sharpening_rounds);
        observe("sharpened", &self.grid);
        self.advect_velocity(&extension, &prior_phase, dt, &mut observe);
        if let Some(source) = &inflow {
            source.enforce_velocity(&mut self.grid);
        }
        observe("forced", &self.grid);
        if self.energy_experiment.mode == "off" {
            self.project(dt);
        } else {
            super::energy_experiment::project(self, dt, previous_energy);
        }
        if let Some(source) = &inflow {
            source.enforce_velocity(&mut self.grid);
        }
        observe("projected", &self.grid);
        self.receipt.frame += 1;
        self.receipt.time += dt as f64;
        self.receipt.volume = self.grid.volume.iter().map(|&v| v as f64).sum();
        self.receipt.phi_area = (0..self.grid.volume.len())
            .map(|i| self.grid.target(self.grid.point(i)) as f64)
            .sum();
        self.receipt.contour_area = 0.0;
        self.receipt.contour_l1 = 0.0;
        for i in 0..self.grid.volume.len() {
            let area = super::swept_extension::contour_fill(&self.grid, i, 0.0) as f64;
            self.receipt.contour_area += area;
            self.receipt.contour_l1 += (area - self.grid.volume[i] as f64).abs();
        }
        self.receipt.max_speed = self
            .grid
            .velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0_f32, f32::max);
        self.receipt.pressure = self.pressure.receipt.clone();
        if self
            .grid
            .volume
            .iter()
            .chain(&self.grid.phi)
            .chain(self.grid.velocity.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(ValidationError("non-finite uniform state".into()));
        }
        Ok(())
    }
    fn occupancy(&self, p: [i32; 2]) -> f32 {
        self.grid.index(p).map_or(0.0, |_| {
            (0.5 - self.grid.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5])
                / (4.0 * self.grid.h[1]))
                .clamp(0.0, 1.0)
        })
    }
    fn normal(&self, p: [i32; 2]) -> [f32; 2] {
        let g = &self.grid;
        let mut n = [0.0; 2];
        for a in 0..2 {
            let mut lo = p;
            let mut hi = p;
            lo[a] -= 1;
            hi[a] += 1;
            let sample = |q| {
                if g.index(q).is_some() {
                    self.occupancy(q)
                } else {
                    self.occupancy(g.clamp_cell(q))
                }
            };
            n[a] = (sample(hi) - sample(lo)) / (2.0 * g.h[a]);
        }
        let norm = (n[0] * n[0] + n[1] * n[1]).sqrt();
        if norm > 1e-9 {
            [n[0] / norm, n[1] / norm]
        } else {
            [0.0; 2]
        }
    }
    fn curvature(&self, p: [i32; 2]) -> f32 {
        let mut value = 0.0;
        for a in 0..2 {
            let mut lo = p;
            let mut hi = p;
            lo[a] -= 1;
            hi[a] += 1;
            value -= (self.normal(hi)[a] - self.normal(lo)[a]) / (2.0 * self.grid.h[a]);
        }
        value
    }
    fn advect_velocity(
        &mut self,
        e: &Extension,
        prior_phase: &[bool],
        dt: f32,
        observe: &mut impl FnMut(&str, &Grid),
    ) {
        let g = &self.grid;
        let mut next = super::velocity::advect(g, e, prior_phase, &self.options, dt);
        std::mem::swap(&mut self.grid.velocity, &mut next);
        observe("velocityAdvected", &self.grid);
        std::mem::swap(&mut self.grid.velocity, &mut next);
        let g = &self.grid;
        for i in 0..next.len() {
            let p = g.point(i);
            if self.options.velocity_transport != "maccormack"
                && self.options.two_level_advection == "tiles"
                && !e.fine_at(p.map(|v| v as f32 + 0.5))
            {
                continue;
            }
            let own = self.occupancy(p);
            for a in 0..2 {
                let mut v = next[i][a];
                let mut neighbor = p;
                neighbor[a] += 1;
                let other = self.occupancy(neighbor);
                if a == 1 && (own > 1e-5 || other > 1e-5) {
                    v += self.gravity[a] * dt;
                }
                if own > 0.0 && self.viscosity > 0.0 {
                    let mut lap = 0.0;
                    for axis in 0..2 {
                        let mut lo = p;
                        let mut hi = p;
                        lo[axis] -= 1;
                        hi[axis] += 1;
                        let sample = |r| g.velocity[g.index(g.clamp_cell(r)).unwrap()][a];
                        lap += (sample(lo) - 2.0 * g.velocity[i][a] + sample(hi))
                            / (g.h[axis] * g.h[axis]);
                    }
                    v += dt * self.viscosity / self.rho * lap;
                }
                if self.sigma > 0.0 && g.index(neighbor).is_some() && own != other {
                    v += dt * self.sigma / self.rho
                        * 0.5
                        * (self.curvature(p) + self.curvature(neighbor))
                        * (other - own)
                        / g.h[a];
                }
                next[i][a] = v;
            }
        }
        self.grid.velocity = next;
    }
    pub(super) fn project(&mut self, dt: f32) {
        let g = &self.grid;
        let balance = self
            .energy_experiment
            .mode
            .starts_with("balance-")
            .then(|| {
                super::energy_experiment::balance(
                    g,
                    &self.options.volume_pressure_rows,
                    dt,
                    &self.energy_experiment,
                )
            });
        let fine = &mut self.pressure.levels[0];
        for y in 0..fine.dims[1] {
            for x in 0..fine.dims[0] {
                let i = x + fine.dims[0] * y;
                let p = [x as i32 - 1, y as i32 - 1];
                let open = g.open(p);
                let ambient =
                    g.open_top && p[1] == g.dims[1] as i32 && p[0] >= 0 && p[0] < g.dims[0] as i32;
                fine.phi[i] = g.pressure_phi(p, &self.options.volume_pressure_rows);
                fine.topology[i] = [
                    if ambient { 1.0 } else { open },
                    g.pressure_face(p, 0),
                    g.pressure_face(p, 1),
                ];
                fine.minimum[i] = if (open <= 1e-5
                    || g.index(p)
                        .is_some_and(|j| g.rigid_centres.as_ref().is_some_and(|v| v[j])))
                    && !ambient
                {
                    0.0
                } else {
                    -3.402823e38
                };
                let mut divergence = 0.0;
                for a in 0..2 {
                    let mut q = p;
                    q[a] -= 1;
                    divergence += (g.pressure_face(p, a) * g.face(p, a)
                        - g.pressure_face(q, a) * g.face(q, a))
                        / g.h[a];
                    if g.rigid_speeds.is_some() {
                        divergence += (g.pressure_face(p, a) - open) * g.solid_speed(p, a)
                            - (g.pressure_face(q, a) - open) * g.solid_speed(q, a);
                    }
                }
                let correction = if self.diagnostic_disable_overfill_correction {
                    0.0
                } else {
                    g.index(p).map_or(0.0, |j| {
                        (0.5 * (g.volume[j] - open).max(0.0)).min(open) / dt.max(1e-12)
                            + balance.as_ref().map_or(0.0, |v| v[j])
                    })
                };
                fine.rhs[i] = if fine.phi[i] < 0.0 {
                    -self.rho * (divergence - correction) / dt
                } else {
                    0.0
                };
            }
        }
        self.pressure.solve(&self.options, dt, self.rho, g.open_top);
        let fine = &self.pressure.levels[0];
        let pressure = |p: [i32; 2]| -> f32 {
            if g.pressure_phi(p, &self.options.volume_pressure_rows) >= 0.0 {
                return 0.0;
            }
            fine.index([p[0] + 1, p[1] + 1]).map_or(0.0, |i| fine.p[i])
        };
        let project = |p: [i32; 2], a: usize, v: f32| {
            let mut q = p;
            q[a] += 1;
            if g.pressure_face(p, a) <= 1e-6 {
                return g.solid_speed(p, a);
            }
            let pa = g.pressure_phi(p, &self.options.volume_pressure_rows);
            let pb = g.pressure_phi(q, &self.options.volume_pressure_rows);
            if pa >= 0.0 && pb >= 0.0 {
                return 0.0;
            }
            v - dt / self.rho * (pressure(q) - pressure(p)) / (g.h[a] * theta(pa, pb))
        };
        let mut velocities = g.velocity.clone();
        let mut low_x = g.low_x.clone();
        let mut low_y = g.low_y.clone();
        let mut released = vec![0_u8; g.volume.len()];
        for i in 0..velocities.len() {
            let p = g.point(i);
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
        }
        self.grid.velocity = velocities;
        self.grid.low_x = low_x;
        self.grid.low_y = low_y;
        self.grid.released = released;
    }
}
