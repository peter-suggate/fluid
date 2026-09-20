//! Two-dimensional CM11a hierarchy. Coordinates include a persistent one-cell
//! halo. Port of webgpu-uniform-pressure-multigrid.{ts,wgsl.ts}.
use super::options::{
    UniformGeometricOptions, UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,
    UNIFORM_CM11A_COARSE_SWEEP_CAP, UNIFORM_CM11A_PHI_PRESERVATION_LEVELS,
    UNIFORM_CM11A_RECOVERY_BATCHES, UNIFORM_CM11A_RECOVERY_REDUCTION,
    UNIFORM_CM11A_RECOVERY_SWEEPS,
};
use crate::types::ValidationError;
const FREE: f32 = -3.402823e38;
const OFFSETS: [[i32; 2]; 4] = [[-1, 0], [1, 0], [0, -1], [0, 1]];
// Compensated f32 pair, matching mgTwoSum/mgDS* in the WGSL oracle.
#[derive(Clone, Copy, Default)]
struct Pair(f32, f32);
impl Pair {
    fn sum(a: f32, b: f32) -> Self {
        let s = a + b;
        let bb = s - a;
        Self(s, (a - (s - bb)) + (b - bb))
    }
    fn add(self, b: Self) -> Self {
        let s = Self::sum(self.0, b.0);
        let t = Self::sum(self.1, b.1);
        let u = Self::sum(s.1, t.0);
        let v = Self::sum(s.0, u.0);
        Self(v.0, (v.1 + u.1) + t.1)
    }
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
    fn scale(self, b: f32) -> Self {
        let product = self.0 * b;
        let error = self.0.mul_add(b, -product) + self.1 * b;
        Self::sum(product, error)
    }
    fn divide(self, b: f32) -> Self {
        let q = self.0 / b;
        let remainder = self.add(Self(q, 0.0).scale(b).neg());
        Self::sum(q, (remainder.0 + remainder.1) / b)
    }
    fn value(self) -> f32 {
        self.0 + self.1
    }
}
fn pair_sum(v: [Pair; 4]) -> Pair {
    v[0].add(v[1]).add(v[2].add(v[3]))
}
#[derive(Clone)]
pub struct Level {
    pub dims: [usize; 2],
    pub h: [f32; 2],
    pub phi: Vec<f32>,
    /// Cell capacity, positive x and positive y pressure dual volumes.
    pub topology: Vec<[f32; 3]>,
    pub p: Vec<f32>,
    pub rhs: Vec<f32>,
    pub minimum: Vec<f32>,
    coefficients: Vec<[f32; 4]>,
}
impl Level {
    pub fn new(physical: [usize; 2], h: [f32; 2]) -> Self {
        let dims = physical.map(|n| n + 2);
        let n = dims[0] * dims[1];
        Self {
            dims,
            h,
            phi: vec![1.0; n],
            topology: vec![[0.0; 3]; n],
            p: vec![0.0; n],
            rhs: vec![0.0; n],
            minimum: vec![FREE; n],
            coefficients: vec![[0.0; 4]; n],
        }
    }
    pub fn index(&self, q: [i32; 2]) -> Option<usize> {
        (q[0] >= 0 && q[1] >= 0 && q[0] < self.dims[0] as i32 && q[1] < self.dims[1] as i32)
            .then(|| q[0] as usize + self.dims[0] * q[1] as usize)
    }
    fn point(&self, i: usize) -> [i32; 2] {
        [(i % self.dims[0]) as i32, (i / self.dims[0]) as i32]
    }
    fn neighbor(&self, i: usize, k: usize) -> Option<usize> {
        let q = self.point(i);
        self.index([q[0] + OFFSETS[k][0], q[1] + OFFSETS[k][1]])
    }
    fn children(&self, coarse: &Self, i: usize) -> [usize; 4] {
        let q = coarse.point(i);
        let stride = [
            (self.dims[0] - 2) / (coarse.dims[0] - 2),
            (self.dims[1] - 2) / (coarse.dims[1] - 2),
        ];
        std::array::from_fn(|k| {
            let p: [i32; 2] = std::array::from_fn(|a| {
                ((q[a] - 1) * stride[a] as i32 + ((k >> a) & 1) as i32 * (stride[a] as i32 - 1) + 1)
                    .clamp(0, self.dims[a] as i32 - 1)
            });
            self.index(p).unwrap()
        })
    }
    fn continue_phi(&mut self) {
        let mut result = self.phi.clone();
        for (i, value) in result.iter_mut().enumerate() {
            if self.topology[i][0] > 1e-5 {
                continue;
            }
            let mut sum = 0.0;
            let mut weight = 0.0;
            for k in 0..4 {
                if let Some(j) = self.neighbor(i, k) {
                    let v = self.topology[j][0];
                    if v > 1e-5 && self.phi[j] < 0.0 {
                        sum += v * self.phi[j];
                        weight += v;
                    }
                }
            }
            if weight > 0.0 {
                *value = sum / weight.max(1e-9);
            }
        }
        self.phi = result;
    }
    fn bake(&mut self, open_top: bool) {
        for i in 0..self.p.len() {
            for k in 0..4 {
                let axis = k / 2;
                let mut coefficient = 0.0;
                if let Some(j) = self.neighbor(i, k) {
                    let vf = if k % 2 == 1 {
                        self.topology[i][axis + 1]
                    } else {
                        self.topology[j][axis + 1]
                    };
                    if vf > 1e-6 {
                        coefficient =
                            vf / (self.h[axis] * self.h[axis] * theta(self.phi[i], self.phi[j]));
                    }
                } else if open_top && k == 3 {
                    coefficient = self.topology[i][2]
                        / (self.h[1] * self.h[1] * theta(self.phi[i], 0.5 * self.h[1]));
                }
                self.coefficients[i][k] = coefficient;
            }
        }
    }
    fn diagonal(&self, i: usize) -> f32 {
        (self.coefficients[i][0] + self.coefficients[i][1])
            + (self.coefficients[i][2] + self.coefficients[i][3])
    }
    fn apply(&self, i: usize) -> f32 {
        let mut terms = [0.0; 4];
        for (k, t) in terms.iter_mut().enumerate() {
            let other = self
                .neighbor(i, k)
                .filter(|&j| self.phi[j] < 0.0)
                .map_or(0.0, |j| self.p[j]);
            *t = self.coefficients[i][k] * (self.p[i] - other);
        }
        (terms[0] + terms[1]) + (terms[2] + terms[3])
    }
    fn residual(&self, rhs: &[f32]) -> Vec<f32> {
        rhs.iter()
            .enumerate()
            .map(|(i, &b)| {
                if self.phi[i] < 0.0 {
                    b - self.apply(i)
                } else {
                    0.0
                }
            })
            .collect()
    }
    fn smooth(&mut self, rhs: &[f32]) {
        for colour in 0..2 {
            for i in 0..self.p.len() {
                let q = self.point(i);
                if self.phi[i] >= 0.0 || ((q[0] + q[1]) & 1) != colour {
                    self.p[i] = self.p[i].max(self.minimum[i]);
                    continue;
                }
                let mut sums = [0.0; 4];
                for (k, s) in sums.iter_mut().enumerate() {
                    if let Some(j) = self.neighbor(i, k).filter(|&j| self.phi[j] < 0.0) {
                        *s = self.coefficients[i][k] * self.p[j];
                    }
                }
                let sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                let diagonal = self.diagonal(i);
                self.p[i] = (if diagonal > 0.0 {
                    (rhs[i] + sum) / diagonal
                } else {
                    0.0
                })
                .max(self.minimum[i]);
            }
        }
    }
    fn norm(&self, rhs: &[f32], scale: f32) -> f32 {
        let mut maximum = 0.0_f32;
        for i in 0..self.p.len() {
            // Also inspect inactive rows: prolongation can read their values.
            if !self.p[i].is_finite() {
                return f32::INFINITY;
            }
            if self.phi[i] >= 0.0 {
                continue;
            }
            let diagonal = self.diagonal(i);
            if diagonal <= 0.0 {
                continue;
            }
            let r = rhs[i] - self.apply(i);
            if !r.is_finite() {
                return f32::INFINITY;
            }
            let gap = (self.p[i] - self.minimum[i]).max(0.0);
            let residual = if r < 0.0 && -r >= gap * diagonal {
                gap * diagonal
            } else {
                r.abs()
            };
            let violation = (self.minimum[i] - self.p[i]).max(0.0) * diagonal;
            let value = residual.max(violation) * scale;
            if !value.is_finite() {
                return f32::INFINITY;
            }
            maximum = maximum.max(value);
        }
        maximum
    }
    fn prolong(&self, fine: &Self) -> Vec<f32> {
        (0..fine.p.len())
            .map(|i| {
                let p = fine.point(i);
                let q: [f32; 2] = std::array::from_fn(|a| {
                    (p[a] as f32 - 0.5) * (self.dims[a] - 2) as f32 / (fine.dims[a] - 2) as f32
                        + 0.5
                });
                let base = q.map(|v| v.floor() as i32);
                let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
                let mut sum = 0.0;
                let mut total = 0.0;
                for k in 0..4 {
                    let r = [base[0] + (k & 1), base[1] + ((k >> 1) & 1)];
                    if r[0] < 1
                        || r[1] < 1
                        || r[0] >= self.dims[0] as i32 - 1
                        || r[1] >= self.dims[1] as i32 - 1
                    {
                        continue;
                    }
                    let w = (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                        * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] });
                    sum += w * self.p[self.index(r).unwrap()];
                    total += w;
                }
                if total > 0.0 {
                    sum / total
                } else {
                    0.0
                }
            })
            .collect()
    }
}
pub fn theta(a: f32, b: f32) -> f32 {
    if (a < 0.0) == (b < 0.0) {
        1.0
    } else {
        let liquid = a.min(b);
        let air = a.max(b);
        (-liquid / (air - liquid).max(1e-9)).clamp(0.05, 1.0)
    }
}
#[derive(Default, Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PressureReceipt {
    pub cycles: usize,
    pub budget: usize,
    pub residual: f32,
    pub converged: bool,
    pub coarse_iterations: usize,
    pub coarse_cap_fail: bool,
    pub initial_residual: f32,
    pub rejected_cycles: usize,
    pub recovery_sweeps: usize,
    pub recovery_exhausted: bool,
}
#[derive(Clone)]
pub struct Pressure {
    pub levels: Vec<Level>,
    pub receipt: PressureReceipt,
}
impl Pressure {
    pub fn new(dims: [usize; 2], h: [f32; 2]) -> Result<Self, ValidationError> {
        if dims.iter().any(|&v| v < 2) {
            return Err(ValidationError(
                "CM11a requires at least two cells per axis".into(),
            ));
        }
        let minimum = *dims.iter().min().unwrap();
        let count = if minimum.is_power_of_two() {
            minimum.ilog2() as usize
        } else {
            0
        };
        let lockstep = count > 0
            && dims.iter().all(|n| n % (1 << (count - 1)) == 0)
            && ((dims[0] >> (count - 1)) + 2) * ((dims[1] >> (count - 1)) + 2) <= 256;
        let mut levels = vec![Level::new(dims, h)];
        let mut current = dims;
        loop {
            let next = if lockstep {
                if levels.len() >= count {
                    break;
                }
                current.map(|n| n / 2)
            } else {
                current.map(|n| if n > 2 && n % 2 == 0 { n / 2 } else { n })
            };
            if next == current {
                break;
            }
            levels.push(Level::new(
                next,
                std::array::from_fn(|a| h[a] * dims[a] as f32 / next[a] as f32),
            ));
            current = next;
        }
        if levels.last().unwrap().p.len() > 256 {
            return Err(ValidationError(
                "CM11a coarsest lattice exceeds 256 cells".into(),
            ));
        }
        Ok(Self {
            levels,
            receipt: PressureReceipt::default(),
        })
    }
    fn restrict(&self, level: usize, field: &[f32]) -> Vec<f32> {
        let fine = &self.levels[level];
        let coarse = &self.levels[level + 1];
        (0..coarse.p.len())
            .map(|i| {
                let c = fine.children(coarse, i);
                ((field[c[0]] + field[c[1]]) + (field[c[2]] + field[c[3]])) * 0.25
            })
            .collect()
    }
    fn restrict_min(&mut self, level: usize, subtract: bool) {
        let fine = &self.levels[level];
        let coarse = &self.levels[level + 1];
        let values = (0..coarse.p.len())
            .map(|i| {
                // A bound-active row may have a valid nonzero equation
                // residual. Every coarse correction must inherit its bound;
                // dropping it makes the coarse solve chase forbidden suction.
                fine.children(coarse, i).iter().fold(FREE, |m, &j| {
                    m.max(fine.minimum[j] - if subtract { fine.p[j] } else { 0.0 })
                })
            })
            .collect();
        self.levels[level + 1].minimum = values;
    }
    fn coarsest(&mut self, rhs: &[f32], scale: f32) {
        let level = self.levels.last_mut().unwrap();
        let mut p: Vec<Pair> = level.p.iter().map(|&v| Pair(v, 0.0)).collect();
        let mut converged = false;
        for iteration in 0..UNIFORM_CM11A_COARSE_SWEEP_CAP {
            for colour in 0..2 {
                for i in 0..p.len() {
                    let q = level.point(i);
                    if level.phi[i] >= 0.0 || (q[0] + q[1] + 1) & 1 != colour {
                        continue;
                    }
                    let terms = std::array::from_fn(|k| {
                        level
                            .neighbor(i, k)
                            .filter(|&j| level.phi[j] < 0.0)
                            .map_or(Pair::default(), |j| p[j].scale(level.coefficients[i][k]))
                    });
                    let d = level.diagonal(i);
                    if d > 0.0 {
                        let next = pair_sum(terms).add(Pair(rhs[i], 0.0)).divide(d);
                        p[i] = if next.value() < level.minimum[i] {
                            Pair(level.minimum[i], 0.0)
                        } else {
                            next
                        };
                    }
                }
            }
            for i in 0..p.len() {
                if p[i].value() < level.minimum[i] {
                    p[i] = Pair(level.minimum[i], 0.0);
                }
            }
            let mut maximum = 0.0_f32;
            for i in 0..p.len() {
                if level.phi[i] >= 0.0 {
                    continue;
                }
                let d = level.diagonal(i);
                if d <= 0.0 {
                    continue;
                }
                let terms = std::array::from_fn(|k| {
                    let other = level
                        .neighbor(i, k)
                        .filter(|&j| level.phi[j] < 0.0)
                        .map_or(Pair::default(), |j| p[j]);
                    p[i].add(other.neg()).scale(level.coefficients[i][k])
                });
                let r = Pair(rhs[i], 0.0).add(pair_sum(terms).neg()).value();
                let gap = (p[i].value() - level.minimum[i]).max(0.0);
                let residual = if r < 0.0 && -r >= gap * d {
                    gap * d
                } else {
                    r.abs()
                };
                maximum = maximum.max(residual * scale);
            }
            self.receipt.coarse_iterations = self.receipt.coarse_iterations.max(iteration + 1);
            if maximum <= UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE {
                converged = true;
                break;
            }
        }
        self.receipt.coarse_cap_fail |= !converged;
        for (out, p) in level.p.iter_mut().zip(p) {
            *out = p.value();
        }
    }
    fn v_cycle(&mut self, l: usize, rhs: &[f32], o: &UniformGeometricOptions, scale: f32) {
        if l + 1 == self.levels.len() {
            self.coarsest(rhs, scale);
            return;
        }
        for _ in 0..o.pressure_sweeps as usize {
            self.levels[l].smooth(rhs);
        }
        let residual = self.levels[l].residual(rhs);
        let coarse_rhs = self.restrict(l, &residual);
        self.levels[l + 1].p.fill(0.0);
        self.restrict_min(l, true);
        self.v_cycle(l + 1, &coarse_rhs, o, scale);
        let correction = self.levels[l + 1].prolong(&self.levels[l]);
        for (p, c) in self.levels[l].p.iter_mut().zip(correction) {
            *p += c;
        }
        for _ in 0..o.pressure_sweeps as usize {
            self.levels[l].smooth(rhs);
        }
    }
    /// A rejected candidate never reaches velocity projection. Keep the best
    /// finite feasible iterate even when recovery cannot reach the tolerance.
    fn accept_candidate(
        &mut self,
        residual: f32,
        best: &mut Vec<f32>,
        best_residual: &mut f32,
    ) -> bool {
        if residual.is_finite() && residual <= *best_residual {
            best.clone_from(&self.levels[0].p);
            *best_residual = residual;
            true
        } else {
            self.levels[0].p.clone_from(best);
            false
        }
    }
    fn recover(
        &mut self,
        rhs: &[f32],
        scale: f32,
        tolerance: f32,
        best: &mut Vec<f32>,
        best_residual: &mut f32,
    ) {
        for _ in 0..UNIFORM_CM11A_RECOVERY_BATCHES {
            for _ in 0..UNIFORM_CM11A_RECOVERY_SWEEPS {
                self.levels[0].smooth(rhs);
            }
            self.receipt.recovery_sweeps += UNIFORM_CM11A_RECOVERY_SWEEPS;
            let candidate = self.levels[0].norm(rhs, scale);
            // Red-black smoothing can transiently increase the infinity norm.
            // Keep its finite working iterate between batches, but publish only
            // the best field. A non-finite batch restarts from that safe field.
            if !candidate.is_finite() {
                self.levels[0].p.clone_from(best);
            } else if candidate <= *best_residual {
                best.clone_from(&self.levels[0].p);
                *best_residual = candidate;
            }
            if tolerance > 0.0 && *best_residual <= tolerance {
                self.receipt.converged = true;
                break;
            }
        }
        self.levels[0].p.clone_from(best);
        self.receipt.residual = *best_residual;
        self.receipt.recovery_exhausted = !self.receipt.converged;
    }
    pub fn solve(&mut self, o: &UniformGeometricOptions, dt: f32, rho: f32, open_top: bool) {
        let maximum = (o.pressure_full_cycles + o.pressure_v_cycles) as usize;
        // Same demand rule as pressure-policy.ts. The CPU observes its previous
        // completed solve synchronously; GPU callers observe an asynchronous sample.
        let budget = if o.pressure_cycle_budget == "lagged"
            && o.pressure_residual_tolerance > 0.0
            && self.receipt.cycles > 0
        {
            let observed = self.receipt.cycles;
            let demand = if self.receipt.converged {
                observed + o.pressure_budget_headroom as usize
            } else {
                (2 * observed).max(observed + 2)
            };
            demand.max(1).min(maximum)
        } else {
            maximum
        };
        self.receipt = PressureReceipt {
            budget,
            ..Default::default()
        };
        let count = self.levels.len();
        for l in 0..count - 1 {
            let fine = &self.levels[l];
            let coarse = &self.levels[l + 1];
            let mut phis = vec![0.0; coarse.p.len()];
            let mut topology = vec![[0.0; 3]; coarse.p.len()];
            for i in 0..phis.len() {
                let children = fine.children(coarse, i);
                let mut sum = 0.0;
                let mut positive = 0.0;
                let mut positives = 0;
                for j in children {
                    sum += fine.phi[j];
                    if fine.phi[j] >= 0.0 {
                        positive += fine.phi[j];
                        positives += 1;
                    }
                    for a in 0..3 {
                        topology[i][a] += fine.topology[j][a] * 0.25;
                    }
                }
                phis[i] = if positives > 0
                    && positives < 4
                    && l + 1 <= UNIFORM_CM11A_PHI_PRESERVATION_LEVELS
                {
                    positive / positives as f32
                } else {
                    sum * 0.25
                };
            }
            self.levels[l + 1].phi = phis;
            self.levels[l + 1].topology = topology;
        }
        for level in &mut self.levels {
            level.continue_phi();
            level.bake(open_top);
            level.p.fill(0.0);
        }
        let rhs = self.levels[0].rhs.clone();
        let original_min = self.levels[0].minimum.clone();
        let scale = dt / rho;
        let mut best = self.levels[0].p.clone();
        let mut best_residual = self.levels[0].norm(&rhs, scale);
        self.receipt.initial_residual = best_residual;
        self.receipt.residual = best_residual;
        for cycle in 0..budget {
            if cycle < o.pressure_full_cycles as usize {
                let backup = self.levels[0].p.clone();
                for (m, p) in self.levels[0].minimum.iter_mut().zip(&backup) {
                    *m -= p;
                }
                let mut correction = vec![self.levels[0].residual(&rhs)];
                for l in 0..count - 1 {
                    correction.push(self.restrict(l, &correction[l]));
                    self.restrict_min(l, false);
                }
                self.levels[count - 1].p.fill(0.0);
                self.coarsest(&correction[count - 1], scale);
                for l in (0..count - 1).rev() {
                    self.levels[l].p = self.levels[l + 1].prolong(&self.levels[l]);
                    self.v_cycle(l, &correction[l], o, scale);
                }
                for (p, b) in self.levels[0].p.iter_mut().zip(backup) {
                    *p += b;
                }
                self.levels[0].minimum.clone_from(&original_min);
            } else {
                self.v_cycle(0, &rhs, o, scale);
            }
            self.receipt.cycles = cycle + 1;
            let candidate = self.levels[0].norm(&rhs, scale);
            if !self.accept_candidate(candidate, &mut best, &mut best_residual) {
                self.receipt.rejected_cycles += 1;
                // Once multigrid diverges, do not feed its correction back into
                // another cycle. Recover using only projected fine-grid sweeps.
                self.recover(
                    &rhs,
                    scale,
                    o.pressure_residual_tolerance
                        .min(self.receipt.initial_residual * UNIFORM_CM11A_RECOVERY_REDUCTION),
                    &mut best,
                    &mut best_residual,
                );
                break;
            }
            self.receipt.residual = best_residual;
            if o.pressure_residual_tolerance > 0.0
                && self.receipt.residual <= o.pressure_residual_tolerance
            {
                self.receipt.converged = true;
                break;
            }
        }
    }
}

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn rejected_and_nonfinite_candidates_restore_the_accepted_field() {
        let mut solver = Pressure::new([8, 8], [0.05; 2]).unwrap();
        solver.levels[0].p.fill(12.0);
        let mut best = solver.levels[0].p.clone();
        let mut norm = 3.0;
        for candidate in [4.0, f32::INFINITY, f32::NAN] {
            solver.levels[0].p.fill(1e20);
            assert!(!solver.accept_candidate(candidate, &mut best, &mut norm));
            assert_eq!(solver.levels[0].p, best);
            assert_eq!(norm, 3.0);
        }
        solver.levels[0].p.fill(8.0);
        assert!(solver.accept_candidate(2.0, &mut best, &mut norm));
        assert_eq!(best, solver.levels[0].p);
        assert_eq!(norm, 2.0);
    }

    #[test]
    fn wall_bounds_survive_all_six_transfers() {
        let mut solver = Pressure::new([128, 128], [0.05; 2]).unwrap();
        for y in 0..solver.levels[0].dims[1] {
            let i = y * solver.levels[0].dims[0];
            solver.levels[0].minimum[i] = 0.0;
            solver.levels[0].p[i] = 7.0;
        }
        for l in 0..solver.levels.len() - 1 {
            solver.restrict_min(l, l == 0);
            assert_eq!(solver.levels[l + 1].minimum[0], -7.0);
            assert!(solver.levels[l + 1].minimum.iter().any(|&v| v == FREE));
        }
    }

    #[test]
    fn rejected_pressure_recovers_with_bounded_projected_sweeps() {
        let mut solver = Pressure::new([4, 4], [1.0; 2]).unwrap();
        let l = &mut solver.levels[0];
        for y in 1..=4 {
            for x in 1..=4 {
                let i = x + l.dims[0] * y;
                l.phi[i] = -1.0;
                l.topology[i] = [1.0; 3];
                l.minimum[i] = 0.0;
                l.rhs[i] = 1.0;
            }
        }
        l.bake(false);
        let rhs = l.rhs.clone();
        let mut best = l.p.clone();
        let mut norm = l.norm(&rhs, 1.0);
        let initial = norm;
        l.p.fill(1e20);
        let bad = l.norm(&rhs, 1.0);
        assert!(!solver.accept_candidate(bad, &mut best, &mut norm));
        solver.recover(&rhs, 1.0, 0.01, &mut best, &mut norm);
        assert!(norm < initial && norm <= 0.01, "residual {norm}");
        assert!(solver.receipt.converged);
        assert!(solver.receipt.recovery_sweeps > 0 && solver.receipt.recovery_sweeps <= 64);
        assert!(solver.levels[0]
            .p
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0));
        // Disabled early stopping exhausts the cap honestly, retaining the best.
        solver.receipt = PressureReceipt::default();
        solver.recover(&rhs, 1.0, 0.0, &mut best, &mut norm);
        assert!(solver.receipt.recovery_exhausted);
        assert_eq!(solver.receipt.recovery_sweeps, 64);
        assert_eq!(solver.levels[0].p, best);
    }

    #[test]
    fn inactive_nan_and_bound_violations_are_not_convergence() {
        let mut level = Level::new([2, 2], [1.0; 2]);
        let rhs = vec![0.0; level.p.len()];
        level.p[0] = f32::NAN;
        assert!(level.norm(&rhs, 1.0).is_infinite());
        level.p[0] = -1.0;
        level.phi[0] = -1.0;
        level.minimum[0] = 0.0;
        level.coefficients[0] = [1.0; 4];
        assert!(level.norm(&rhs, 1.0) >= 4.0);
    }
}
