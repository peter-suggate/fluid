//! Diagnostic-only global compensation experiments. No shared defaults/UI.
use super::{grid::Grid, world::World};
use crate::types::ValidationError;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct Config {
    pub mode: String,
    pub gain: f64,
    pub carry_debt: bool,
    pub balance_scope: String,
    pub audit_receipt: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            mode: "off".into(),
            gain: 1.0,
            carry_debt: false,
            balance_scope: "global".into(),
            audit_receipt: true,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if ![
            "off",
            "balance-deficit",
            "balance-surface-deficit",
            "balance-uniform",
            "source-work",
            "source-rate",
            "energy-cap",
        ]
        .contains(&self.mode.as_str())
            || !["global", "liquid-components"].contains(&self.balance_scope.as_str())
            || !self.gain.is_finite()
            || !(0.0..=if self.mode == "source-rate" { 8.0 } else { 1.0 }).contains(&self.gain)
        {
            return Err(ValidationError(
                "invalid diagnostic energy compensation".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub kinetic_before: f64,
    pub kinetic_after: f64,
    pub reference_kinetic: f64,
    pub requested_removal: f64,
    pub removed: f64,
    pub debt: f64,
    pub alpha: f64,
    pub minimum_estimate: f64,
    pub budget_miss: f64,
    pub solves: usize,
    pub positive_source: f64,
    pub negative_source: f64,
    pub max_liquid_divergence_error: f64,
}

pub fn energy(g: &Grid, rho: f32, gravity: [f32; 2]) -> (f64, f64) {
    let mut k = 0.0;
    let mut p = 0.0;
    for i in 0..g.volume.len() {
        let cell = g.point(i);
        let mass = rho as f64
            * g.h[0] as f64
            * g.h[1] as f64
            * super::swept_extension::contour_fill(g, i, 0.0) as f64;
        for a in 0..2 {
            let mut q = cell;
            q[a] -= 1;
            k += 0.25 * mass * ((g.face(cell, a) as f64).powi(2) + (g.face(q, a) as f64).powi(2));
            p -= mass * gravity[a] as f64 * (cell[a] as f64 + 0.5) * g.h[a] as f64;
        }
    }
    (k, p)
}

/// Negative compensation only in non-overfilled pressure-liquid cells.
/// Every original positive overfill source is left exactly unchanged.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BalanceFields {
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
    pub deficit: Vec<f32>,
    pub region: Vec<usize>,
    pub region_count: usize,
}

pub fn balance(g: &Grid, rows: &str, dt: f32, config: &Config) -> Vec<f32> {
    if config.balance_scope != "global" {
        return balance_fields(g, rows, dt, config).negative;
    }
    // The practical candidate needs one scratch field and two scalar sums;
    // detailed per-cell diagnostics and connectivity are opt-in only.
    let mut negative = vec![0.0; g.volume.len()];
    if !config.mode.starts_with("balance-") {
        return negative;
    }
    let mut positive = 0.0_f64;
    let mut total = 0.0_f64;
    for i in 0..g.volume.len() {
        if g.capacity[i] <= 1e-5 || g.pressure_phi(g.point(i), rows) >= 0.0 {
            continue;
        }
        let excess = (g.volume[i] - g.capacity[i]).max(0.0);
        positive += (0.5 * excess).min(g.capacity[i]) as f64 / dt as f64;
        if excess == 0.0 {
            negative[i] = if config.mode == "balance-surface-deficit" {
                (super::swept_extension::contour_fill(g, i, 0.0) - g.volume[i]).max(0.0)
            } else if config.mode == "balance-deficit" {
                (g.capacity[i] - g.volume[i]).max(0.0)
            } else {
                g.capacity[i]
            };
            total += negative[i] as f64;
        }
    }
    if total > 0.0 {
        let rate = (config.gain * positive / total).min(1.0 / dt as f64) as f32;
        for v in &mut negative {
            *v *= -rate;
        }
    }
    negative
}

pub fn balance_fields(g: &Grid, rows: &str, dt: f32, config: &Config) -> BalanceFields {
    let n = g.volume.len();
    let mut out = BalanceFields {
        positive: vec![0.0; n],
        negative: vec![0.0; n],
        deficit: vec![0.0; n],
        region: vec![n; n],
        region_count: 0,
    };
    if !config.mode.starts_with("balance-") {
        return out;
    }
    let wet: Vec<_> = (0..n)
        .map(|i| g.capacity[i] > 1e-5 && g.pressure_phi(g.point(i), rows) < 0.0)
        .collect();
    if config.balance_scope == "liquid-components" {
        // Physical liquid connectivity, not solid-halo pressure-unknown connectivity.
        // Rebuilt each step so splitting and merging do not retain stale budgets.
        let mut queue = Vec::new();
        for seed in 0..n {
            if !wet[seed] || out.region[seed] != n {
                continue;
            }
            let id = out.region_count;
            out.region_count += 1;
            out.region[seed] = id;
            queue.clear();
            queue.push(seed);
            let mut head = 0;
            while head < queue.len() {
                let i = queue[head];
                head += 1;
                let p = g.point(i);
                for a in 0..2 {
                    for direction in [-1, 1] {
                        let mut q = p;
                        q[a] += direction;
                        let Some(j) = g.index(q) else {
                            continue;
                        };
                        let face = if direction > 0 { p } else { q };
                        if wet[j] && out.region[j] == n && g.pressure_face(face, a) > 1e-5 {
                            out.region[j] = id;
                            queue.push(j);
                        }
                    }
                }
            }
        }
    } else {
        out.region_count = 1;
        for i in 0..n {
            if wet[i] {
                out.region[i] = 0;
            }
        }
    }
    let mut positive = vec![0.0_f64; out.region_count];
    let mut total = vec![0.0_f64; out.region_count];
    for i in 0..n {
        if !wet[i] {
            continue;
        }
        let id = out.region[i];
        let excess = (g.volume[i] - g.capacity[i]).max(0.0);
        let amount = (0.5 * excess).min(g.capacity[i]);
        out.positive[i] = amount / dt;
        // Preserve the original global experiment's accumulation exactly.
        positive[id] += amount as f64 / dt as f64;
        if excess == 0.0 {
            out.deficit[i] = if config.mode == "balance-surface-deficit" {
                (super::swept_extension::contour_fill(g, i, 0.0) - g.volume[i]).max(0.0)
            } else if config.mode == "balance-deficit" {
                (g.capacity[i] - g.volume[i]).max(0.0)
            } else {
                g.capacity[i]
            };
            total[id] += out.deficit[i] as f64;
        }
    }
    let rates: Vec<_> = (0..out.region_count)
        .map(|id| {
            if total[id] > 0.0 {
                (config.gain * positive[id] / total[id]).min(1.0 / dt as f64) as f32
            } else {
                0.0
            }
        })
        .collect();
    for i in 0..n {
        if wet[i] {
            out.negative[i] = -out.deficit[i] * rates[out.region[i]];
        }
    }
    out
}

fn scale(g: &mut Grid, a: f64) {
    for v in g
        .velocity
        .iter_mut()
        .flatten()
        .chain(&mut g.low_x)
        .chain(&mut g.low_y)
    {
        *v *= a as f32;
    }
}

/// Fit K(u_expansion + alpha*(u_full-u_expansion)), then repeat the ORIGINAL
/// constrained pressure solve using alpha*u_predicted. The final solve retains
/// overfill RHS, contact constraints and release publication; no post-scale.
pub fn project(w: &mut World, dt: f32, previous_energy: (f64, f64)) {
    let config = w.energy_experiment.clone();
    if !config.audit_receipt && config.mode.starts_with("balance-") {
        w.project(dt);
        return;
    }
    let previous_debt = if config.carry_debt {
        w.energy_receipt.debt
    } else {
        0.0
    };
    let mut r = Receipt {
        alpha: 1.0,
        solves: 1,
        ..Receipt::default()
    };
    if config.mode == "off" || config.mode.starts_with("balance-") {
        w.project(dt);
        r.kinetic_after = energy(&w.grid, w.rho, w.gravity).0;
        r.kinetic_before = r.kinetic_after;
    } else if config.mode == "source-rate" {
        let mut source = 0.0_f64;
        let mut liquid = 0.0_f64;
        for i in 0..w.grid.volume.len() {
            if w.grid.capacity[i] > 1e-5
                && w.grid
                    .pressure_phi(w.grid.point(i), &w.options.volume_pressure_rows)
                    < 0.0
            {
                source += (0.5 * (w.grid.volume[i] - w.grid.capacity[i]).max(0.0))
                    .min(w.grid.capacity[i]) as f64;
                liquid += w.grid.capacity[i] as f64;
            }
        }
        r.alpha = (-config.gain * source / liquid.max(1e-30)).exp();
        // A cheap heuristic: damping depends on repaired volume per step, not time alone.
        scale(&mut w.grid, r.alpha);
        w.project(dt);
        r.kinetic_after = energy(&w.grid, w.rho, w.gravity).0;
    } else {
        let input = w.grid.clone();
        let pressure = w.pressure.clone();
        w.project(dt);
        let full = w.grid.clone();
        let full_pressure = w.pressure.clone();
        let (full_k, potential) = energy(&full, w.rho, w.gravity);
        r.kinetic_before = full_k;
        let target = if config.mode == "source-work" {
            w.grid = input.clone();
            w.pressure = pressure.clone();
            w.diagnostic_disable_overfill_correction = true;
            w.project(dt);
            w.diagnostic_disable_overfill_correction = false;
            r.solves += 1;
            r.reference_kinetic = energy(&w.grid, w.rho, w.gravity).0;
            r.requested_removal =
                config.gain * (full_k - r.reference_kinetic).max(0.0) + previous_debt;
            (full_k - r.requested_removal).max(0.0)
        } else {
            let budget = (previous_energy.0 + previous_energy.1 - potential).max(0.0);
            r.requested_removal = config.gain * (full_k - budget).max(0.0);
            full_k - r.requested_removal
        };
        w.grid = full.clone();
        w.pressure = full_pressure.clone();
        if full_k > target + 1e-8 {
            w.grid = input.clone();
            scale(&mut w.grid, 0.0);
            w.pressure = pressure.clone();
            w.project(dt);
            r.solves += 1;
            let floor = w.grid.clone();
            let c = energy(&floor, w.rho, w.gravity).0;
            let mut mid = floor.clone();
            for i in 0..mid.velocity.len() {
                for a in 0..2 {
                    mid.velocity[i][a] = 0.5 * (floor.velocity[i][a] + full.velocity[i][a]);
                }
            }
            for i in 0..mid.low_x.len() {
                mid.low_x[i] = 0.5 * (floor.low_x[i] + full.low_x[i]);
            }
            for i in 0..mid.low_y.len() {
                mid.low_y[i] = 0.5 * (floor.low_y[i] + full.low_y[i]);
            }
            let middle = energy(&mid, w.rho, w.gravity).0;
            let a = (2.0 * (full_k + c - 2.0 * middle)).max(0.0);
            let b = full_k - c - a;
            let min_alpha = if a > 1e-20 {
                (-b / (2.0 * a)).clamp(0.0, 1.0)
            } else if b >= 0.0 {
                0.0
            } else {
                1.0
            };
            let eval = |x: f64| (a * x + b) * x + c;
            r.minimum_estimate = eval(min_alpha);
            let mut lo = min_alpha;
            let mut hi = 1.0;
            if r.minimum_estimate <= target {
                for _ in 0..40 {
                    let x = 0.5 * (lo + hi);
                    if eval(x) <= target {
                        lo = x;
                    } else {
                        hi = x;
                    }
                }
            }
            r.alpha = lo;
            w.grid = input;
            scale(&mut w.grid, lo);
            w.pressure = pressure;
            w.project(dt);
            r.solves += 1;
            // Wall active-set changes can invalidate the quadratic prediction.
            // Never accept more kinetic energy than the uncompensated step.
            if energy(&w.grid, w.rho, w.gravity).0 > full_k {
                w.grid = full;
                w.pressure = full_pressure;
                r.alpha = 1.0;
            }
        }
        r.kinetic_after = energy(&w.grid, w.rho, w.gravity).0;
        r.removed = full_k - r.kinetic_after;
        r.budget_miss = (r.kinetic_after - target).max(0.0);
        r.debt = if config.carry_debt {
            (r.requested_removal - r.removed).max(0.0)
        } else {
            0.0
        };
    }
    let g = &w.grid;
    let negative = balance(g, &w.options.volume_pressure_rows, dt, &config);
    for i in 0..g.volume.len() {
        let p = g.point(i);
        if g.capacity[i] <= 1e-5 || g.pressure_phi(p, &w.options.volume_pressure_rows) >= 0.0 {
            continue;
        }
        let source = (0.5 * (g.volume[i] - g.capacity[i]).max(0.0)).min(g.capacity[i]) / dt;
        r.positive_source += source as f64;
        r.negative_source += negative[i] as f64;
        let mut div = 0.0_f64;
        for a in 0..2 {
            let mut q = p;
            q[a] -= 1;
            div += (g.pressure_face(p, a) * g.face(p, a) - g.pressure_face(q, a) * g.face(q, a))
                as f64
                / g.h[a] as f64;
        }
        r.max_liquid_divergence_error = r
            .max_liquid_divergence_error
            .max((div - source as f64 - negative[i] as f64).abs());
    }
    w.energy_receipt = r;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_geometric::UniformGeometricOptions;
    fn pool() -> Grid {
        let mut g = Grid::new([32, 24], [0.1, 0.1], false).unwrap();
        for y in 0..=24 {
            for x in 0..=32 {
                g.phi[x + 33 * y] = y as f32 * 0.1 - 1.2;
            }
        }
        for i in 0..g.volume.len() {
            g.volume[i] = if i < 32 * 12 { 1.05 } else { 0.0 };
        }
        g
    }
    #[test]
    fn mandatory_expansion_survives_every_compensator() {
        let mut reference = None;
        for mode in [
            "off",
            "balance-deficit",
            "balance-surface-deficit",
            "balance-uniform",
            "source-rate",
            "source-work",
            "energy-cap",
        ] {
            let mut options = UniformGeometricOptions::default();
            options.active_region = "off".into();
            options.total_surface_volume = "off".into();
            let mut w = World::from_grid(pool(), options, [0.0, 0.0], 1000.0, 0.0, 0.0).unwrap();
            w.energy_experiment.mode = mode.into();
            project(&mut w, 1.0 / 30.0, (0.0, 0.0));
            let k = energy(&w.grid, 1000.0, [0.0, 0.0]).0;
            let expected = *reference.get_or_insert(k);
            assert!(
                (k - expected).abs() < 1e-6,
                "{mode} weakened mandatory expansion: {k} != {expected}"
            );
            assert!(k > 500.0);
            assert!(
                w.energy_receipt.max_liquid_divergence_error < 0.005,
                "{mode}"
            );
        }
    }
    #[test]
    fn balancing_changes_only_non_overfilled_liquid_and_cancels_net_source() {
        let mut g = pool();
        for i in 0..32 * 12 {
            if i % 32 >= 16 {
                g.volume[i] = 0.95;
            }
        }
        for mode in [
            "balance-deficit",
            "balance-surface-deficit",
            "balance-uniform",
        ] {
            let config = Config {
                mode: mode.into(),
                ..Config::default()
            };
            let negative = balance(&g, "off", 1.0 / 30.0, &config);
            let mut net = 0.0_f64;
            for i in 0..g.volume.len() {
                if g.volume[i] > 1.0 || i >= 32 * 12 {
                    assert_eq!(negative[i], 0.0);
                }
                net += negative[i] as f64;
                if i < 32 * 12 {
                    net += (0.5 * (g.volume[i] - 1.0).max(0.0) / (1.0 / 30.0)) as f64;
                }
            }
            assert!(net.abs() < 1e-4, "{mode}: {net}");
        }
    }

    #[test]
    fn disconnected_liquid_does_not_share_component_budgets() {
        let mut g = pool();
        for i in 0..g.volume.len() {
            if i % 32 == 16 {
                g.capacity[i] = 0.0;
                g.volume[i] = 0.0;
            } else if i % 32 > 16 && i < 32 * 12 {
                g.volume[i] = 0.95;
            }
        }
        let mut config = Config {
            mode: "balance-surface-deficit".into(),
            ..Config::default()
        };
        let global = balance_fields(&g, "off", 1.0 / 30.0, &config);
        assert!(global.negative.iter().sum::<f32>() < -100.0);
        config.balance_scope = "liquid-components".into();
        let local = balance_fields(&g, "off", 1.0 / 30.0, &config);
        assert_eq!(local.region_count, 2);
        assert_eq!(global.positive, local.positive);
        assert!(local.negative.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn insufficient_deficit_leaves_uncompensated_expansion() {
        let mut g = pool();
        for i in 0..32 * 12 {
            if i % 32 >= 16 {
                g.volume[i] = 0.99999;
            }
        }
        let config = Config {
            mode: "balance-surface-deficit".into(),
            ..Config::default()
        };
        let fields = balance_fields(&g, "off", 1.0 / 30.0, &config);
        for i in 0..g.volume.len() {
            assert!(fields.negative[i].abs() * (1.0 / 30.0) <= fields.deficit[i] + 1e-7);
        }
        assert!(fields.positive.iter().sum::<f32>() + fields.negative.iter().sum::<f32>() > 100.0);
    }
}
