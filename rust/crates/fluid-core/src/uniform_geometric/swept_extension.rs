//! Opt-in 2D lab experiment. A bounded air-only correction of extended MAC
//! velocities; projected liquid faces and solid faces remain immutable.
use super::{extension::Extension, grid::Grid, UniformGeometricOptions};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    #[default]
    Off,
    Support,
    Local,
    Coarse,
    Combined,
    Conforming,
    ConformingOnly,
    TransportAgreement,
    RegionalVolume,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct Config {
    pub mode: Mode,
    pub local_sweeps: usize,
    pub coarse_sweeps: usize,
    pub cycles: usize,
    pub trace_steps: usize,
    pub area_guard: bool,
    pub agreement_gain: f32,
    pub agreement_clamp: f32,
    pub agreement_iterations: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Off,
            local_sweeps: 4,
            coarse_sweeps: 24,
            cycles: 1,
            trace_steps: 1,
            area_guard: false,
            agreement_gain: 0.7,
            agreement_clamp: 0.5,
            agreement_iterations: 2,
        }
    }
}
impl Config {
    /// UI exposes only the useful surface experiments. Velocity trials remain
    /// native-runner diagnostics; none alters the shared 3D defaults.
    pub fn lab_profile(profile: &str) -> Result<Self, crate::types::ValidationError> {
        let mut config = Self::default();
        match profile {
            "off" => {}
            "regional" | "regional-area" | "area-only" => {
                config.mode = Mode::RegionalVolume;
                config.agreement_gain = 0.3;
                config.area_guard = profile != "regional";
                if profile == "area-only" {
                    config.agreement_iterations = 0;
                }
            }
            _ => {
                return Err(crate::types::ValidationError(
                    "unknown 2D surface experiment".into(),
                ))
            }
        }
        Ok(config)
    }
    pub fn validate(&self) -> Result<(), crate::types::ValidationError> {
        if self.local_sweeps > 64
            || self.coarse_sweeps > 128
            || self.cycles > 8
            || self.trace_steps == 0
            || self.trace_steps > 16
        {
            return Err(crate::types::ValidationError(
                "invalid swept extension experiment budget".into(),
            ));
        }
        if !self.agreement_gain.is_finite()
            || !(0.0..=1.0).contains(&self.agreement_gain)
            || !self.agreement_clamp.is_finite()
            || !(0.0..=1.0).contains(&self.agreement_clamp)
            || self.agreement_iterations > 8
        {
            return Err(crate::types::ValidationError(
                "invalid transport agreement experiment budget".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub mode: Mode,
    pub active_cells: usize,
    pub coarse_cells: usize,
    pub band_cells: f32,
    pub divergence_before: f64,
    pub divergence_after: f64,
    pub max_correction: f32,
    pub accepted: bool,
    pub surface: Option<SurfaceReceipt>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceReceipt {
    pub max_normal_shift: f32,
    pub area_shift: f32,
    pub area_before: f64,
    pub area_after: f64,
    pub area_target: f64,
}

struct System {
    dims: [usize; 2],
    active: Vec<bool>,
    diagonal: Vec<f32>,
    edge: Vec<[f32; 2]>,
    rhs: Vec<f32>,
    p: Vec<f32>,
}
impl System {
    fn new(dims: [usize; 2]) -> Self {
        let n = dims[0] * dims[1];
        Self {
            dims,
            active: vec![false; n],
            diagonal: vec![0.0; n],
            edge: vec![[0.0; 2]; n],
            rhs: vec![0.0; n],
            p: vec![0.0; n],
        }
    }
    fn neighbor_sum(&self, i: usize) -> f32 {
        let [nx, ny] = self.dims;
        let x = i % nx;
        let y = i / nx;
        let mut sum = 0.0;
        if x + 1 < nx {
            sum += self.edge[i][0] * self.p[i + 1];
        }
        if y + 1 < ny {
            sum += self.edge[i][1] * self.p[i + nx];
        }
        if x > 0 {
            sum += self.edge[i - 1][0] * self.p[i - 1];
        }
        if y > 0 {
            sum += self.edge[i - nx][1] * self.p[i - nx];
        }
        sum
    }
    fn residual(&self, i: usize) -> f32 {
        self.rhs[i] - self.diagonal[i] * self.p[i] + self.neighbor_sum(i)
    }
    fn smooth(&mut self, sweeps: usize) {
        for _ in 0..sweeps {
            for color in 0..2 {
                for i in 0..self.p.len() {
                    if self.active[i]
                        && self.diagonal[i] > 0.0
                        && ((i % self.dims[0] + i / self.dims[0]) & 1) == color
                    {
                        self.p[i] = (self.rhs[i] + self.neighbor_sum(i)) / self.diagonal[i];
                    }
                }
            }
        }
    }
}

pub fn build(
    g: &Grid,
    o: &UniformGeometricOptions,
    dt: f32,
    config: &Config,
) -> (Extension, Receipt) {
    if config.mode == Mode::Off {
        return (Extension::build(g, o, dt), Receipt::default());
    }
    if matches!(config.mode, Mode::TransportAgreement | Mode::RegionalVolume) {
        return (
            Extension::build(g, o, dt),
            Receipt {
                mode: config.mode,
                ..Receipt::default()
            },
        );
    }
    let h = g.h[0].min(g.h[1]);
    let travel = g
        .velocity
        .iter()
        .flatten()
        .fold(0.0_f32, |a, &v| a.max(v.abs() * dt / h));
    let band = travel + 4.0;
    // The correction's footprint must also have actual fine extension support.
    let mut options = o.clone();
    options.two_level_fine_reach = options.two_level_fine_reach.max((band / 4.0).ceil());
    let mut extension = Extension::build(g, &options, dt);
    extension.trace_steps = config.trace_steps;
    let mut receipt = Receipt {
        mode: config.mode,
        band_cells: band,
        ..Receipt::default()
    };
    if config.mode == Mode::Support {
        return (extension, receipt);
    }
    if config.mode == Mode::ConformingOnly {
        extension.enable_conforming(g);
        return (extension, receipt);
    }
    let n = g.volume.len();
    let mut fine = System::new(g.dims);
    let mut air = vec![false; n];
    for i in 0..n {
        let p = g.point(i);
        let phi = g.pressure_phi(p, &o.volume_pressure_rows);
        air[i] = g.capacity[i] > 0.99999 && phi >= 0.0;
        // Far-air phi is not globally redistanced. Select the swept footprint
        // geometrically through dilated work tiles, rather than |phi| as distance.
        fine.active[i] = air[i] && extension.fine_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]);
    }
    let scale = [h / g.h[0], h / g.h[1]];
    for i in 0..n {
        let p = g.point(i);
        if fine.active[i] {
            for a in 0..2 {
                let mut lo = p;
                lo[a] -= 1;
                fine.rhs[i] -= (extension.fine_face(p, a) - extension.fine_face(lo, a)) * scale[a];
            }
        }
        for a in 0..2 {
            let mut q = p;
            q[a] += 1;
            let Some(j) = g.index(q) else {
                continue;
            };
            if !air[i]
                || !air[j]
                || !(fine.active[i] || fine.active[j])
                || g.pressure_face(p, a) <= 1e-5
            {
                continue;
            }
            let w = scale[a] * scale[a];
            // Inactive air is a zero-potential outer anchor, never a liquid face.
            fine.edge[i][a] = w;
            if fine.active[i] {
                fine.diagonal[i] += w;
            }
            if fine.active[j] {
                fine.diagonal[j] += w;
            }
        }
    }
    receipt.active_cells = fine.active.iter().filter(|&&v| v).count();
    receipt.divergence_before = fine
        .rhs
        .iter()
        .map(|&v| (v as f64 / h as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cd = g.dims.map(|v| v.div_ceil(4));
    let parent = |i: usize| (i % g.dims[0]) / 4 + cd[0] * ((i / g.dims[0]) / 4);
    let mut coarse = System::new(cd);
    // Galerkin aggregation with piecewise-constant prolongation. Fine smoothing
    // below distributes the correction within blocks; no post-hoc face mask.
    for i in 0..n {
        if fine.active[i] {
            coarse.active[parent(i)] = true;
        }
        for a in 0..2 {
            let w = fine.edge[i][a];
            if w == 0.0 {
                continue;
            }
            let j = i + if a == 0 { 1 } else { g.dims[0] };
            let ci = parent(i);
            let cj = parent(j);
            if fine.active[i] && fine.active[j] && ci == cj {
                continue;
            }
            if fine.active[i] {
                coarse.diagonal[ci] += w;
            }
            if fine.active[j] {
                coarse.diagonal[cj] += w;
            }
            if fine.active[i] && fine.active[j] {
                coarse.edge[ci][a] += w;
            }
        }
    }
    receipt.coarse_cells = coarse.active.iter().filter(|&&v| v).count();
    for _ in 0..config.cycles {
        if config.mode == Mode::Local {
            fine.smooth(config.local_sweeps);
            continue;
        }
        if matches!(config.mode, Mode::Combined | Mode::Conforming) {
            fine.smooth(config.local_sweeps);
        }
        coarse.rhs.fill(0.0);
        coarse.p.fill(0.0);
        for i in 0..n {
            if fine.active[i] {
                coarse.rhs[parent(i)] += fine.residual(i);
            }
        }
        coarse.smooth(config.coarse_sweeps);
        for i in 0..n {
            if fine.active[i] {
                fine.p[i] += coarse.p[parent(i)];
            }
        }
        if matches!(config.mode, Mode::Combined | Mode::Conforming) {
            fine.smooth(config.local_sweeps);
        }
    }
    receipt.divergence_after = (0..n)
        .filter(|&i| fine.active[i])
        .map(|i| (fine.residual(i) as f64 / h as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    receipt.accepted = fine.p.iter().all(|v| v.is_finite())
        && receipt.divergence_after <= receipt.divergence_before * 1.00001;
    if receipt.accepted {
        for i in 0..n {
            for a in 0..2 {
                if fine.edge[i][a] == 0.0 {
                    continue;
                }
                let j = i + if a == 0 { 1 } else { g.dims[0] };
                let correction = (fine.p[i] - fine.p[j]) * scale[a];
                extension.values[i][a] += correction;
                receipt.max_correction = receipt.max_correction.max(correction.abs());
            }
        }
    }
    if config.mode == Mode::Conforming {
        extension.enable_conforming(g);
    }
    (extension, receipt)
}

pub fn contour_fill(g: &Grid, i: usize, shift: f32) -> f32 {
    let nxv = g.dims[0] + 1;
    let j = i % g.dims[0] + nxv * (i / g.dims[0]);
    super::diagnostics::bilinear_fill(
        [j, j + 1, j + nxv, j + nxv + 1].map(|k| (g.phi[k] + shift) as f64),
        8,
    ) as f32
        * g.capacity[i]
}

/// Correct the existing contour by smooth normal displacements. Overlapping
/// patches have +/-4-cell support. Native trials can use transported old-phi
/// occupancy; the UI candidate uses V. Neither reconstructs phi from cell fills.
/// The optional scalar area constraint is global, including across components.
pub fn correct_surface(g: &mut Grid, reference: &[f32], config: &Config) -> SurfaceReceipt {
    let n = g.volume.len();
    let [nx, ny] = g.dims;
    let h = g.h[0].min(g.h[1]);
    let mut receipt = SurfaceReceipt {
        area_before: (0..n).map(|i| contour_fill(g, i, 0.0) as f64).sum(),
        area_target: reference.iter().map(|&v| v as f64).sum(),
        ..SurfaceReceipt::default()
    };
    let mut max_shift = 0.0_f32;
    // Select by geometric proximity, not |phi|: fast motion can leave phi far
    // from a distance function even on vertices incident to the zero contour.
    let mut band = vec![0.0_f32; g.phi.len()];
    for i in 0..n {
        if g.capacity[i] == 0.0 {
            continue;
        }
        let j = i % nx + (nx + 1) * (i / nx);
        let corners = [j, j + 1, j + nx + 1, j + nx + 2];
        if corners.iter().any(|&k| g.phi[k] <= 0.0) && corners.iter().any(|&k| g.phi[k] >= 0.0) {
            for k in corners {
                band[k] = 1.0;
            }
        }
    }
    for _ in 0..4 {
        let old = band.clone();
        for y in 0..=ny {
            for x in 0..=nx {
                let j = x + (nx + 1) * y;
                for (a, b) in [
                    (x.saturating_sub(1), y),
                    ((x + 1).min(nx), y),
                    (x, y.saturating_sub(1)),
                    (x, (y + 1).min(ny)),
                ] {
                    band[j] = band[j].max((old[a + (nx + 1) * b] - 0.2).max(0.0));
                }
            }
        }
    }
    let metric: Vec<f32> = band
        .iter()
        .enumerate()
        .map(|(j, &w)| {
            let grad = g.gradient(&g.phi, [(j % (nx + 1)) as f32, (j / (nx + 1)) as f32]);
            w * ((grad[0] / g.h[0]).hypot(grad[1] / g.h[1])).max(0.1)
        })
        .collect();
    let fill = |g: &Grid, i: usize, shift: f32| -> f32 {
        let j = i % nx + (nx + 1) * (i / nx);
        super::diagnostics::bilinear_fill(
            [j, j + 1, j + nx + 1, j + nx + 2].map(|k| (g.phi[k] + shift * metric[k]) as f64),
            8,
        ) as f32
            * g.capacity[i]
    };
    for _ in 0..config.agreement_iterations {
        let mut fields = vec![[0.0; 2]; n];
        for i in 0..n {
            if g.capacity[i] < 0.99999 {
                continue;
            }
            let target = fill(g, i, 0.0);
            let area = (fill(g, i, -0.25 * h) - fill(g, i, 0.25 * h)) / 0.5;
            fields[i] = [reference[i] - target, area.max(0.0)];
        }
        let mut scratch = fields.clone();
        // Two binomial filters per axis, total support +/-4 cells. Each tap
        // stops at a solid so opposing sides of a wall cannot exchange error.
        for a in [0, 1, 0, 1] {
            for i in 0..n {
                scratch[i] = [0.0; 2];
                if g.capacity[i] < 0.99999 {
                    continue;
                }
                let p = g.point(i);
                for (d, w) in [(-2, 1.0), (-1, 4.0), (0, 6.0), (1, 4.0), (2, 1.0)] {
                    let mut q = p;
                    q[a] += d;
                    let Some(j) = g.index(q) else {
                        continue;
                    };
                    let mut mid = p;
                    mid[a] += d.signum();
                    if g.capacity[j] < 0.99999 || g.open(mid) < 0.99999 {
                        continue;
                    }
                    for b in 0..2 {
                        scratch[i][b] += w / 16.0 * fields[j][b];
                    }
                }
            }
            std::mem::swap(&mut fields, &mut scratch);
        }
        let mut next = g.phi.clone();
        for y in 0..=ny {
            for x in 0..=nx {
                let j = x + (nx + 1) * y;
                if band[j] == 0.0 {
                    continue;
                }
                let mut sum = [0.0; 2];
                for dy in [-1, 0] {
                    for dx in [-1, 0] {
                        if let Some(i) = g.index([x as i32 + dx, y as i32 + dy]) {
                            for b in 0..2 {
                                sum[b] += fields[i][b];
                            }
                        }
                    }
                }
                if sum[1] < 0.02 {
                    continue;
                }
                let shift = config.agreement_gain * sum[0] / sum[1];
                if shift.abs() < 1e-4 {
                    continue;
                }
                let shift = shift.clamp(-config.agreement_clamp, config.agreement_clamp) * h;
                next[j] -= shift * metric[j];
                max_shift = max_shift.max(shift.abs());
            }
        }
        g.phi = next;
    }
    if config.area_guard {
        let desired: f64 = reference.iter().map(|&v| v as f64).sum();
        let weights = metric;
        let area = |shift: f32| -> f64 {
            (0..n)
                .map(|i| {
                    let j = i % nx + (nx + 1) * (i / nx);
                    super::diagnostics::bilinear_fill(
                        [j, j + 1, j + nx + 1, j + nx + 2]
                            .map(|k| (g.phi[k] - shift * weights[k]) as f64),
                        8,
                    ) * g.capacity[i] as f64
                })
                .sum()
        };
        let mut lo = -h;
        let mut hi = h;
        if (area(0.0) - desired).abs() > 1e-4 {
            for _ in 0..14 {
                let mid = (lo + hi) * 0.5;
                if area(mid) < desired {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let shift = (lo + hi) * 0.5;
            receipt.area_shift = shift;
            for (v, w) in g.phi.iter_mut().zip(weights) {
                *v -= shift * w;
            }
            max_shift = max_shift.max(shift.abs());
        }
    }
    receipt.max_normal_shift = max_shift;
    receipt.area_after = (0..n).map(|i| contour_fill(g, i, 0.0) as f64).sum();
    receipt
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Grid, UniformGeometricOptions) {
        let mut g = Grid::new([32, 24], [0.05; 2], false).unwrap();
        for y in 0..=24 {
            for x in 0..=32 {
                g.phi[x + 33 * y] = ((x as f32 - 10.0).hypot(y as f32 - 10.0) - 6.0) * 0.05;
            }
        }
        for i in 0..g.volume.len() {
            let p = g.point(i);
            g.volume[i] = g.target(p);
            g.velocity[i] = [0.02 * (p[0] as f32 + 1.0), -0.02 * (p[1] as f32 + 1.0)];
        }
        let mut o = UniformGeometricOptions::default();
        o.active_region = "off".into();
        (g, o)
    }
    #[test]
    fn normal_correction_preserves_matching_surface_and_stored_volume() {
        let (mut g, _) = fixture();
        let reference: Vec<_> = (0..g.volume.len())
            .map(|i| contour_fill(&g, i, 0.0))
            .collect();
        g.volume = reference.clone();
        let before = g.phi.clone();
        let config = Config::lab_profile("regional-area").unwrap();
        let r = correct_surface(&mut g, &reference, &config);
        assert_eq!(g.phi, before);
        assert_eq!(g.volume, reference);
        assert_eq!(r.max_normal_shift, 0.0);
    }
    #[test]
    fn normal_correction_handles_stretched_phi_and_does_not_seed_remote_cells() {
        let (mut g, _) = fixture();
        let reference: Vec<_> = (0..g.volume.len())
            .map(|i| contour_fill(&g, i, 0.0))
            .collect();
        g.volume = reference.clone();
        for v in &mut g.phi {
            *v = (*v + 0.015) * 8.0;
        }
        let far = g.phi[32 + 33 * 24];
        let before: f64 = (0..g.volume.len())
            .map(|i| (contour_fill(&g, i, 0.0) - reference[i]).abs() as f64)
            .sum();
        let r = correct_surface(
            &mut g,
            &reference,
            &Config::lab_profile("regional-area").unwrap(),
        );
        let after: f64 = (0..g.volume.len())
            .map(|i| (contour_fill(&g, i, 0.0) - reference[i]).abs() as f64)
            .sum();
        assert!((r.area_after - r.area_target).abs() < 0.01);
        assert!(after < before * 0.2, "{after} vs {before}");
        assert_eq!(g.volume, reference);
        assert_eq!(g.phi[32 + 33 * 24], far);
    }
    #[test]
    fn zero_motion_and_off_are_unchanged() {
        let (mut g, o) = fixture();
        g.velocity.fill([0.0; 2]);
        let original = Extension::build(&g, &o, 1.0 / 30.0);
        for mode in [
            Mode::Off,
            Mode::Local,
            Mode::Coarse,
            Mode::Combined,
            Mode::Conforming,
        ] {
            let (e, r) = build(
                &g,
                &o,
                1.0 / 30.0,
                &Config {
                    mode,
                    ..Config::default()
                },
            );
            assert_eq!(e.values, original.values);
            assert_eq!(r.max_correction, 0.0);
        }
    }
    #[test]
    fn correction_preserves_liquid_faces_and_reduces_air_residual() {
        let (g, o) = fixture();
        let dt = 1.0 / 30.0;
        let (base, _) = build(
            &g,
            &o,
            dt,
            &Config {
                mode: Mode::Support,
                ..Config::default()
            },
        );
        let (e, r) = build(
            &g,
            &o,
            dt,
            &Config {
                mode: Mode::Combined,
                ..Config::default()
            },
        );
        assert!(r.accepted && r.divergence_after < r.divergence_before);
        let mut norm = 0.0_f64;
        for i in 0..g.volume.len() {
            let p = g.point(i);
            let phi = g.pressure_phi(p, "off");
            if phi >= 0.0 && e.fine_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]) {
                let mut div = 0.0;
                for a in 0..2 {
                    let mut lo = p;
                    lo[a] -= 1;
                    div += (e.fine_face(p, a) - e.fine_face(lo, a)) / g.h[a];
                }
                norm += (div as f64).powi(2);
            }
            for a in 0..2 {
                let mut q = p;
                q[a] += 1;
                if phi < 0.0 || g.pressure_phi(q, "off") < 0.0 || g.index(q).is_none() {
                    assert_eq!(e.values[i][a], base.values[i][a]);
                }
            }
        }
        assert!((norm.sqrt() - r.divergence_after).abs() < 1e-3);
    }
}
