//! Two-dimensional CM11a hierarchy. Coordinates include a persistent one-cell
//! halo. Port of webgpu-uniform-pressure-multigrid.{ts,wgsl.ts}, pressure-plan.ts
//! and pressure-policy.ts, reduced to one depth layer with symmetric depth.
//! The cycle operators visit liquid row lists (audit-paging §5.6); the few
//! dense passes are the ones whose dead-row values stay observable.
use super::options::{
    UniformGeometricOptions, UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,
    UNIFORM_CM11A_COARSE_SWEEP_CAP, UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET,
    UNIFORM_CM11A_PHI_PRESERVATION_LEVELS, UNIFORM_CM11A_RECOVERY_BATCHES,
    UNIFORM_CM11A_RECOVERY_REDUCTION, UNIFORM_CM11A_RECOVERY_SWEEPS,
};
use crate::types::ValidationError;
pub const FREE: f32 = -3.402823e38;
// pressure-plan.ts UNIFORM_CM11A_COARSEST_TARGET_CELLS.
const COARSEST_TARGET_CELLS: usize = 256;
// mgD4Sum6/mgD4Sum8 over four distinct in-plane terms.
fn sum4(v: [f32; 4]) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3])
}
// The same sums as Metal compiles them into the per-cycle operators (smoother,
// residual, trilinear transfer and coarse solve): the terms added in order.
fn seq4(v: [f32; 4]) -> f32 {
    ((v[0] + v[1]) + v[2]) + v[3]
}
// mgD4Sum8 as Metal compiles it into mgDownsampleTopology: the eight corners
// summed in order, corner k+4 being plane child k again. Verified bitwise
// against the GPU pyramid; the source's pairwise tree is not.
fn corner_sum(v: [f32; 4]) -> f32 {
    (1..8).fold(v[0], |s, k| s + v[k & 3])
}
/// cm12-numerics.ts cm12GhostFluidTheta with the geometric 1e-9 epsilon.
pub fn cm12_theta(liquid: f32, air: f32) -> f32 {
    (liquid.abs() / (liquid.abs() + air.abs()).max(1e-9)).clamp(0.05, 1.0)
}
// The -x, +x, -y, +y neighbours of (x, y) on a haloed level.
fn neighbours(dims: [usize; 2], x: usize, y: usize) -> [Option<usize>; 4] {
    let i = x + dims[0] * y;
    [
        (x > 0).then(|| i - 1),
        (x + 1 < dims[0]).then(|| i + 1),
        (y > 0).then(|| i - dims[0]),
        (y + 1 < dims[1]).then(|| i + dims[0]),
    ]
}
/// A liquid row as bake leaves it: its neighbours (itself where absent),
/// which of them are liquid, the face coefficients and the diagonal.
#[derive(Clone, Copy)]
struct Row {
    i: u32,
    n: [u32; 4],
    liquid: [bool; 4],
    c: [f32; 4],
    d: f32,
    inverse: f32,
}
impl Row {
    // mg.wgsl mgApply as Metal compiles it: the products folded in order into
    // fused multiply-adds. Verified bitwise against the GPU's restricted
    // residuals; the source's separately rounded products are not.
    fn apply(&self, p: &[f32]) -> f32 {
        let centre = p[self.i as usize];
        let difference = |k: usize| {
            centre
                - if self.liquid[k] {
                    p[self.n[k] as usize]
                } else {
                    0.0
                }
        };
        (1..4).fold(self.c[0] * difference(0), |sum, k| {
            self.c[k].mul_add(difference(k), sum)
        })
    }
}
/// One fine coordinate's mgTrilinearPressure footprint on a coarse axis.
#[derive(Clone, Copy)]
struct Tap {
    cell: [u32; 2],
    interior: [bool; 2],
    weight: [f32; 2],
}
#[derive(Clone)]
pub struct Level {
    pub dims: [usize; 2],
    pub h: [f32; 2],
    pub phi: Vec<f32>,
    /// The raw phi of the last solve, before mgExtrapolatePhiOneCell.
    pub raw_phi: Vec<f32>,
    /// Cell capacity, positive x and positive y pressure dual volumes.
    pub topology: Vec<[f32; 3]>,
    pub p: Vec<f32>,
    pub rhs: Vec<f32>,
    pub minimum: Vec<f32>,
    /// -x, +x, -y, +y face coefficients.
    pub coefficients: Vec<[f32; 4]>,
    /// Liquid rows by colour (x+y)&1 in raster order, rebuilt by bake.
    rows: [Vec<Row>; 2],
    /// mg.wgsl mgFineChild of every row, into the next finer level.
    children: Vec<[u32; 4]>,
    /// Per axis, each row's taps on the next coarser level.
    taps: [Vec<Tap>; 2],
    residual: Vec<f32>,
}
impl Level {
    pub fn new(physical: [usize; 2], h: [f32; 2]) -> Self {
        let dims = physical.map(|n| n + 2);
        let n = dims[0] * dims[1];
        Self {
            dims,
            h,
            phi: vec![1.0; n],
            raw_phi: Vec::new(),
            topology: vec![[0.0; 3]; n],
            p: vec![0.0; n],
            rhs: vec![0.0; n],
            minimum: vec![FREE; n],
            coefficients: vec![[0.0; 4]; n],
            rows: [Vec::new(), Vec::new()],
            children: Vec::new(),
            taps: [Vec::new(), Vec::new()],
            residual: vec![0.0; n],
        }
    }
    pub fn index(&self, q: [i32; 2]) -> Option<usize> {
        (q[0] >= 0 && q[1] >= 0 && q[0] < self.dims[0] as i32 && q[1] < self.dims[1] as i32)
            .then(|| q[0] as usize + self.dims[0] * q[1] as usize)
    }
    // mg.wgsl mgFineChild, child k = ox + 2 oy, for every row of `self`.
    fn link(&mut self, fine: &mut Self) {
        let stride: [i32; 2] =
            std::array::from_fn(|a| ((fine.dims[a] - 2) / (self.dims[a] - 2)) as i32);
        self.children = (0..self.dims[0] * self.dims[1])
            .map(|i| {
                let q = [(i % self.dims[0]) as i32, (i / self.dims[0]) as i32];
                std::array::from_fn(|k| {
                    let p: [i32; 2] = std::array::from_fn(|a| {
                        ((q[a] - 1) * stride[a] + ((k >> a) & 1) as i32 * (stride[a] - 1) + 1)
                            .clamp(0, fine.dims[a] as i32 - 1)
                    });
                    (p[0] + fine.dims[0] as i32 * p[1]) as u32
                })
            })
            .collect();
        // mg.wgsl mgTrilinearPressure, whose weights separate by axis.
        fine.taps = std::array::from_fn(|a| {
            let scale = (self.dims[a] - 2) as f32 / ((fine.dims[a] - 2) as f32).max(1.0);
            (0..fine.dims[a])
                .map(|p| {
                    let q = (p as f32 - 0.5) * scale + 0.5;
                    let base = q.floor() as i32;
                    let f = q - q.floor();
                    let r = [base, base + 1];
                    Tap {
                        cell: r.map(|v| v.max(0) as u32),
                        interior: r.map(|v| v >= 1 && v < self.dims[a] as i32 - 1),
                        weight: [1.0 - f, f],
                    }
                })
                .collect()
        });
        // prolong fills cells off the support with +0, which needs every
        // positive weight total to have a finite reciprocal.
        for ty in &fine.taps[1] {
            for tx in &fine.taps[0] {
                let weights: [f32; 4] = std::array::from_fn(|k| {
                    let (a, b) = (k & 1, k >> 1);
                    if tx.interior[a] && ty.interior[b] {
                        tx.weight[a] * ty.weight[b]
                    } else {
                        0.0
                    }
                });
                let total = seq4(weights);
                if total > 0.0 {
                    assert!(
                        (1.0 / total).is_finite(),
                        "prolongation weight total {total} has no finite reciprocal"
                    );
                }
            }
        }
    }
    // mg.wgsl mgDownsampleTopology: raw coarse phi and dual volumes from this
    // level's raw fields. `destination` is the coarse level index. Counts are
    // over eight depth-doubled corners, and Metal divides by reciprocal.
    fn downsample(&self, coarse: &mut Self, destination: usize) {
        for i in 0..coarse.p.len() {
            let c = coarse.children[i].map(|j| j as usize);
            let t = c.map(|j| self.topology[j]);
            let phi = c.map(|j| self.phi[j]);
            let open = t.map(|v| v[0] > 1e-5);
            let pick = |keep: [bool; 4]| -> [f32; 4] {
                std::array::from_fn(|k| if keep[k] { phi[k] } else { 0.0 })
            };
            let count = |keep: [bool; 4]| 2.0 * keep.iter().filter(|&&k| k).count() as f32;
            let open_count = count(open);
            let phi_sum = if open_count > 0.0 {
                corner_sum(pick(open)) * 8.0 * (1.0 / open_count)
            } else {
                corner_sum(phi)
            };
            let positive: [bool; 4] = std::array::from_fn(|k| open[k] && phi[k] >= 0.0);
            let positives = count(positive);
            let mixed =
                positives > 0.0 && count(std::array::from_fn(|k| open[k] && phi[k] < 0.0)) > 0.0;
            coarse.phi[i] = if mixed && destination <= UNIFORM_CM11A_PHI_PRESERVATION_LEVELS {
                corner_sum(pick(positive)) * (1.0 / positives)
            } else {
                phi_sum / 8.0
            };
            // Only the fine faces on the coarse face plane restrict to it.
            coarse.topology[i] = [
                sum4(t.map(|v| v[0])) * 0.25,
                (t[1][1] + t[3][1]) * 0.5,
                (t[2][2] + t[3][2]) * 0.5,
            ];
        }
    }
    // mg.wgsl mgExtrapolatePhiOneCell.
    fn continue_phi(&mut self) {
        self.raw_phi.clone_from(&self.phi);
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let i = x + self.dims[0] * y;
                if self.topology[i][0] > 1e-5 {
                    continue;
                }
                let mut terms = [0.0; 4];
                let mut weights = [0.0; 4];
                for (k, j) in neighbours(self.dims, x, y).into_iter().enumerate() {
                    if let Some(j) = j {
                        let v = self.topology[j][0];
                        if v > 1e-5 && self.raw_phi[j] < 0.0 {
                            terms[k] = v * self.raw_phi[j];
                            weights[k] = v;
                        }
                    }
                }
                let weight = sum4(weights);
                if weight > 0.0 {
                    self.phi[i] = sum4(terms) / weight.max(1e-9);
                }
            }
        }
    }
    // mg.wgsl mgBakeCoefficients / mgCoefficientRaw, then the row lists.
    fn bake(&mut self, open_top: bool) {
        for rows in &mut self.rows {
            rows.clear();
        }
        for y in 0..self.dims[1] {
            for x in 0..self.dims[0] {
                let i = x + self.dims[0] * y;
                let n = neighbours(self.dims, x, y);
                let mut c = [0.0; 4];
                for (k, coefficient) in c.iter_mut().enumerate() {
                    let axis = k / 2;
                    if let Some(j) = n[k] {
                        let vf = if k % 2 == 1 {
                            self.topology[i][axis + 1]
                        } else {
                            self.topology[j][axis + 1]
                        };
                        if vf > 1e-6 {
                            *coefficient = vf
                                / (self.h[axis] * self.h[axis] * theta(self.phi[i], self.phi[j]));
                        }
                    } else if open_top && k == 3 {
                        let h = self.h[1];
                        *coefficient =
                            self.topology[i][2] / (h * h * cm12_theta(self.phi[i], 0.5 * h));
                    }
                }
                self.coefficients[i] = c;
                if self.phi[i] < 0.0 {
                    let d = seq4(c);
                    self.rows[(x + y) & 1].push(Row {
                        i: i as u32,
                        n: n.map(|j| j.unwrap_or(i) as u32),
                        liquid: n.map(|j| j.is_some_and(|j| self.phi[j] < 0.0)),
                        c,
                        d,
                        inverse: 1.0 / d,
                    });
                }
            }
        }
    }
    fn residual_into(&self, rhs: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for row in self.rows.iter().flatten() {
            out[row.i as usize] = rhs[row.i as usize] - row.apply(&self.p);
        }
    }
    // mg.wgsl mgSmoothColour, one ping-pong pass per colour. A colour's liquid
    // updates read only the other colour, so updating them in place reproduces
    // the snapshot exactly. Every other row is projected with max(p, minimum);
    // that is idempotent and the other colour's liquid rows are rewritten next,
    // so only the first sweep after p last changed (`first`) has any to do.
    // Metal divides by reciprocal multiply; its hardware reciprocal is not
    // always correctly rounded, so about one row in seventy still differs.
    fn smooth(&mut self, rhs: &[f32], first: bool) {
        let Self {
            p, rows, minimum, ..
        } = self;
        for (colour, rows) in rows.iter().enumerate() {
            for row in rows {
                let i = row.i as usize;
                let sums = std::array::from_fn(|k| {
                    if row.liquid[k] {
                        row.c[k] * p[row.n[k] as usize]
                    } else {
                        0.0
                    }
                });
                let value = if row.d > 0.0 {
                    (seq4(sums) + rhs[i]) * row.inverse
                } else {
                    0.0
                };
                p[i] = value.max(minimum[i]);
            }
            if colour == 0 && first {
                for (v, &m) in p.iter_mut().zip(minimum.iter()) {
                    *v = v.max(m);
                }
            }
        }
    }
    // mg.wgsl mgMeasureFineResidual, in divergence units.
    fn norm(&self, rhs: &[f32], dt: f32, rho: f32) -> f32 {
        // Also inspect inactive rows: prolongation can read their values.
        if !self.p.iter().all(|v| v.is_finite()) {
            return f32::INFINITY;
        }
        let mut maximum = 0.0_f32;
        for row in self.rows.iter().flatten() {
            let (i, diagonal) = (row.i as usize, row.d);
            if diagonal <= 0.0 {
                continue;
            }
            let r = rhs[i] - row.apply(&self.p);
            if !r.is_finite() {
                return f32::INFINITY;
            }
            let gap = (self.p[i] - self.minimum[i]).max(0.0);
            let residual = if (r < 0.0) & (-r >= gap * diagonal) {
                gap * diagonal
            } else {
                r.abs()
            };
            let violation = (self.minimum[i] - self.p[i]).max(0.0) * diagonal;
            // Every value is finite and nonnegative, so the order is free.
            let value = (residual.max(violation) * dt) / rho;
            if !value.is_finite() {
                return f32::INFINITY;
            }
            maximum = maximum.max(value);
        }
        maximum
    }
    // The box of coarse cells holding any value other than +0.
    fn support(&self) -> Option<[[usize; 2]; 2]> {
        let mut hull: Option<[[usize; 2]; 2]> = None;
        for (y, row) in self.p.chunks_exact(self.dims[0]).enumerate() {
            let Some(first) = row.iter().position(|v| v.to_bits() != 0) else {
                continue;
            };
            let last = row.iter().rposition(|v| v.to_bits() != 0).unwrap_or(first);
            hull = Some(match hull {
                None => [[first, last], [y, y]],
                Some([x, ys]) => [[x[0].min(first), x[1].max(last)], [ys[0], y]],
            });
        }
        hull
    }
    // mg.wgsl mgTrilinearPressure: coarse (self) into, or onto, the fine p.
    // A fine cell whose interior taps all miss the support reads only +0 and
    // so receives +0 (link checks every positive weight total inverts).
    fn prolong(&self, taps: &[Vec<Tap>; 2], out: &mut [f32], add: bool) {
        let width = taps[0].len();
        let hull = self.support();
        let span = |a: usize| {
            let Some(hull) = hull else { return 0..0 };
            let reads = |t: &Tap| {
                (0..2).any(|k| {
                    t.interior[k] && (hull[a][0]..=hull[a][1]).contains(&(t.cell[k] as usize))
                })
            };
            match (
                taps[a].iter().position(reads),
                taps[a].iter().rposition(reads),
            ) {
                (Some(first), Some(last)) => first..last + 1,
                _ => 0..0,
            }
        };
        let (xs, ys) = (span(0), span(1));
        let fill = |out: &mut [f32]| {
            if add {
                // -0 + +0 is +0, exactly as evaluating the cell would give.
                for o in out {
                    *o += 0.0;
                }
            } else {
                out.fill(0.0);
            }
        };
        for (y, ty) in taps[1].iter().enumerate() {
            let out = &mut out[width * y..width * (y + 1)];
            let xs = if ys.contains(&y) { xs.clone() } else { 0..0 };
            let (head, rest) = out.split_at_mut(xs.start);
            let (live, tail) = rest.split_at_mut(xs.len());
            fill(head);
            fill(tail);
            for (o, tx) in live.iter_mut().zip(&taps[0][xs]) {
                let mut values = [0.0; 4];
                let mut weights = [0.0; 4];
                for k in 0..4 {
                    let (a, b) = (k & 1, k >> 1);
                    if !(tx.interior[a] && ty.interior[b]) {
                        continue;
                    }
                    let w = tx.weight[a] * ty.weight[b];
                    values[k] =
                        w * self.p[tx.cell[a] as usize + self.dims[0] * ty.cell[b] as usize];
                    weights[k] = w;
                }
                let total = seq4(weights);
                let value = if total > 0.0 {
                    seq4(values) * (1.0 / total)
                } else {
                    0.0
                };
                if add {
                    *o += value;
                } else {
                    *o = value;
                }
            }
        }
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
/// pressure-policy.ts uniformCM11aCycleBudget with the page-domain host inputs:
/// minimum 1, initial 1, headroom 0. `last` is the previous solve's executed
/// cycle count and whether it met tolerance. A zero tolerance disables the lag.
pub fn cycle_budget(o: &UniformGeometricOptions, last: Option<(usize, bool)>) -> usize {
    let maximum = (o.pressure_full_cycles + o.pressure_v_cycles) as usize;
    if o.pressure_residual_tolerance <= 0.0 {
        return maximum;
    }
    let minimum = maximum.min(UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET);
    let demand = match last {
        None => UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET,
        Some((executed, true)) => executed,
        Some((executed, false)) => (2 * executed).max(executed + 2),
    };
    maximum.min(minimum.max(demand))
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
/// World's velocity and projection buffers, kept between steps.
#[derive(Clone, Default)]
pub(super) struct Scratch {
    /// Pressure phi on level 0's haloed lattice.
    pub pressure_phi: Vec<f32>,
    /// uvTarget where the deficit reads it: open liquid rows with V <= C.
    pub target: Vec<f32>,
    pub velocity: Vec<[f32; 2]>,
    pub low_x: Vec<f32>,
    pub low_y: Vec<f32>,
    pub released: Vec<u8>,
    pub records: Vec<[f32; 2]>,
    pub chunks: Vec<[f32; 2]>,
}
#[derive(Clone)]
pub struct Pressure {
    pub levels: Vec<Level>,
    pub receipt: PressureReceipt,
    pub(super) scratch: Scratch,
    // Per-solve fields kept between steps: the finest Full-Cycle correction,
    // the pre-cycle iterate, the best iterate and the finest bounds.
    correction: Vec<f32>,
    backup: Vec<f32>,
    best: Vec<f32>,
    original_min: Vec<f32>,
}
// pressure-plan.ts lockstepLevels for dimension 2: the depth layer counts in the
// coarsest haloed size.
fn lockstep(dims: [usize; 2]) -> Option<Vec<[usize; 2]>> {
    let minimum = dims[0].min(dims[1]);
    if !minimum.is_power_of_two() {
        return None;
    }
    let count = minimum.ilog2() as usize;
    let coarsening = 1 << (count - 1);
    if dims.iter().any(|n| n % coarsening != 0) {
        return None;
    }
    let levels: Vec<_> = (0..count).map(|k| dims.map(|n| n >> k)).collect();
    let last = levels[count - 1];
    ((last[0] + 2) * (last[1] + 2) * 3 <= COARSEST_TARGET_CELLS).then_some(levels)
}
// pressure-plan.ts semiCoarsenedLevels.
fn semi_coarsened(dims: [usize; 2]) -> Vec<[usize; 2]> {
    let mut levels = vec![dims];
    loop {
        let previous = levels[levels.len() - 1];
        let next = previous.map(|n| if n % 2 == 0 && n > 2 { n / 2 } else { n });
        if next == previous {
            return levels;
        }
        levels.push(next);
    }
}
// mg.wgsl mgRestrictResidual, its eight corners summed in order.
fn restrict(children: &[[u32; 4]], field: &[f32], out: &mut [f32]) {
    for (o, c) in out.iter_mut().zip(children) {
        *o = corner_sum(c.map(|j| field[j as usize])) / 8.0;
    }
}
impl Pressure {
    pub fn new(dims: [usize; 2], h: [f32; 2]) -> Result<Self, ValidationError> {
        if dims.iter().any(|&v| v < 2) {
            return Err(ValidationError(
                "CM11a requires at least two cells per axis".into(),
            ));
        }
        let plan = lockstep(dims).unwrap_or_else(|| semi_coarsened(dims));
        let mut levels: Vec<Level> = plan
            .into_iter()
            .map(|size| {
                Level::new(
                    size,
                    std::array::from_fn(|a| h[a] * (dims[a] / size[a]) as f32),
                )
            })
            .collect();
        for l in 0..levels.len() - 1 {
            let (fine, coarse) = levels.split_at_mut(l + 1);
            coarse[0].link(&mut fine[l]);
        }
        Ok(Self {
            levels,
            receipt: PressureReceipt::default(),
            scratch: Scratch::default(),
            correction: Vec::new(),
            backup: Vec::new(),
            best: Vec::new(),
            original_min: Vec::new(),
        })
    }
    fn restrict_min(&mut self, level: usize, subtract: bool) {
        let (fine, coarse) = self.levels.split_at_mut(level + 1);
        let (fine, coarse) = (&fine[level], &mut coarse[0]);
        for (m, children) in coarse.minimum.iter_mut().zip(&coarse.children) {
            // A bound-active row may have a valid nonzero equation residual.
            // Every coarse correction must inherit its bound; dropping it
            // makes the coarse solve chase forbidden suction.
            *m = children.iter().fold(FREE, |m, &j| {
                let j = j as usize;
                m.max(fine.minimum[j] - if subtract { fine.p[j] } else { 0.0 })
            });
        }
    }
    // Level l's p from, or plus, the prolonged level l + 1.
    fn prolong(&mut self, l: usize, add: bool) {
        let (fine, coarse) = self.levels.split_at_mut(l + 1);
        let fine = &mut fine[l];
        coarse[0].prolong(&fine.taps, &mut fine.p, add);
    }
    // uniform-coarse-solver.wgsl mgSolveCoarsest. Metal's fast math folds every
    // mgTwoSum error term to zero, so the compiled solve carries no low word:
    // plain f32 reproduces the GPU's pyramids bitwise and the pair form does
    // not. The level's middle depth layer has colour (x+y+1)&1, so the odd
    // rows go first. After the first iteration every row already satisfies
    // its bound, so the projection pass has nothing to do. Only the iteration
    // count leaves the solve, so the residual pass stops at the first row over
    // tolerance, starting from the last iteration's.
    fn coarsest(&mut self, rhs: &[f32], dt: f32, rho: f32) {
        let Level {
            p, rows, minimum, ..
        } = self.levels.last_mut().unwrap();
        let update = |p: &mut [f32], row: &Row| {
            let i = row.i as usize;
            let terms = std::array::from_fn(|k| {
                if row.liquid[k] {
                    p[row.n[k] as usize] * row.c[k]
                } else {
                    0.0
                }
            });
            // mgDSDivide: one remainder correction of the quotient.
            let n = seq4(terms) + rhs[i];
            let quotient = n / row.d;
            let next = quotient + (n - quotient * row.d) / row.d;
            p[i] = if next < minimum[i] { minimum[i] } else { next };
        };
        // The source's max ignores NaN, as the comparison does.
        let over = |p: &[f32], row: &Row| {
            let i = row.i as usize;
            let terms = std::array::from_fn(|k| {
                let other = if row.liquid[k] {
                    p[row.n[k] as usize]
                } else {
                    0.0
                };
                (p[i] - other) * row.c[k]
            });
            let r = rhs[i] - seq4(terms);
            let gap = (p[i] - minimum[i]).max(0.0);
            let residual = if (r < 0.0) & (-r >= gap * row.d) {
                gap * row.d
            } else {
                r.abs()
            };
            (residual * dt) / rho > UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE
        };
        let [even, odd] = &*rows;
        let mut witness: Option<&Row> = None;
        let mut converged = false;
        let mut executed = 0;
        for iteration in 0..UNIFORM_CM11A_COARSE_SWEEP_CAP {
            for row in odd.iter().chain(even) {
                if row.d > 0.0 {
                    update(p, row);
                }
            }
            if iteration == 0 {
                for (v, &m) in p.iter_mut().zip(minimum.iter()) {
                    *v = if *v < m { m } else { *v };
                }
            }
            executed = iteration + 1;
            if witness.is_some_and(|row| over(p, row)) {
                continue;
            }
            // Rows with d <= 0 are neither updated nor measured.
            witness = odd
                .iter()
                .chain(even)
                .find(|row| (row.d > 0.0 || row.d.is_nan()) && over(p, row));
            if witness.is_none() {
                converged = true;
                break;
            }
        }
        self.receipt.coarse_iterations = self.receipt.coarse_iterations.max(executed);
        self.receipt.coarse_cap_fail |= !converged;
    }
    // The coarsest solve or V-cycle on level l's rhs, which the caller keeps
    // in `rhs[l]` for l > 0.
    fn descend(&mut self, l: usize, o: &UniformGeometricOptions, dt: f32, rho: f32) {
        let rhs = std::mem::take(&mut self.levels[l].rhs);
        if l + 1 == self.levels.len() {
            self.coarsest(&rhs, dt, rho);
        } else {
            self.v_cycle(l, &rhs, o, dt, rho);
        }
        self.levels[l].rhs = rhs;
    }
    fn v_cycle(&mut self, l: usize, rhs: &[f32], o: &UniformGeometricOptions, dt: f32, rho: f32) {
        if l + 1 == self.levels.len() {
            self.coarsest(rhs, dt, rho);
            return;
        }
        let sweeps = o.pressure_sweeps as usize;
        for s in 0..sweeps {
            self.levels[l].smooth(rhs, s == 0);
        }
        let mut residual = std::mem::take(&mut self.levels[l].residual);
        self.levels[l].residual_into(rhs, &mut residual);
        let coarse = &mut self.levels[l + 1];
        // The GPU's coarse rhs slot A holds the last restricted residual.
        restrict(&coarse.children, &residual, &mut coarse.rhs);
        coarse.p.fill(0.0);
        self.levels[l].residual = residual;
        self.restrict_min(l, true);
        self.descend(l + 1, o, dt, rho);
        self.prolong(l, true);
        for s in 0..sweeps {
            self.levels[l].smooth(rhs, s == 0);
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
    #[allow(clippy::too_many_arguments)]
    fn recover(
        &mut self,
        rhs: &[f32],
        dt: f32,
        rho: f32,
        tolerance: f32,
        best: &mut Vec<f32>,
        best_residual: &mut f32,
    ) {
        for _ in 0..UNIFORM_CM11A_RECOVERY_BATCHES {
            for s in 0..UNIFORM_CM11A_RECOVERY_SWEEPS {
                self.levels[0].smooth(rhs, s == 0);
            }
            self.receipt.recovery_sweeps += UNIFORM_CM11A_RECOVERY_SWEEPS;
            let candidate = self.levels[0].norm(rhs, dt, rho);
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
    /// Solves the finest level's phi, topology, rhs and minimum, which the
    /// caller builds raw: the coarse pyramid is downsampled before any level's
    /// phi is continued. `budget` truncates the Full-Cycle then V-cycle plan.
    pub fn solve(
        &mut self,
        o: &UniformGeometricOptions,
        dt: f32,
        rho: f32,
        open_top: bool,
        budget: usize,
    ) {
        self.receipt = PressureReceipt {
            budget,
            ..Default::default()
        };
        let count = self.levels.len();
        for l in 0..count - 1 {
            let (fine, coarse) = self.levels.split_at_mut(l + 1);
            fine[l].downsample(&mut coarse[0], l + 1);
        }
        for level in &mut self.levels {
            level.continue_phi();
            level.bake(open_top);
            level.p.fill(0.0);
        }
        let rhs = std::mem::take(&mut self.levels[0].rhs);
        let mut correction = std::mem::take(&mut self.correction);
        let mut backup = std::mem::take(&mut self.backup);
        let mut best = std::mem::take(&mut self.best);
        let mut original_min = std::mem::take(&mut self.original_min);
        correction.resize(rhs.len(), 0.0);
        original_min.clone_from(&self.levels[0].minimum);
        best.clone_from(&self.levels[0].p);
        let mut best_residual = self.levels[0].norm(&rhs, dt, rho);
        self.receipt.initial_residual = best_residual;
        self.receipt.residual = best_residual;
        for cycle in 0..budget {
            if cycle < o.pressure_full_cycles as usize {
                backup.clone_from(&self.levels[0].p);
                for (m, p) in self.levels[0].minimum.iter_mut().zip(&backup) {
                    *m -= p;
                }
                // Level l > 0 keeps its correction rhs in `rhs`.
                self.levels[0].residual_into(&rhs, &mut correction);
                for l in 0..count - 1 {
                    let (fine, coarse) = self.levels.split_at_mut(l + 1);
                    let field = if l == 0 { &correction } else { &fine[l].rhs };
                    restrict(&coarse[0].children, field, &mut coarse[0].rhs);
                    self.restrict_min(l, false);
                }
                self.levels[count - 1].p.fill(0.0);
                if count == 1 {
                    self.coarsest(&correction, dt, rho);
                } else {
                    self.descend(count - 1, o, dt, rho);
                }
                for l in (0..count - 1).rev() {
                    self.prolong(l, false);
                    if l == 0 {
                        self.v_cycle(0, &correction, o, dt, rho);
                    } else {
                        self.descend(l, o, dt, rho);
                    }
                }
                for (p, b) in self.levels[0].p.iter_mut().zip(&backup) {
                    *p += b;
                }
                self.levels[0].minimum.clone_from(&original_min);
            } else {
                self.v_cycle(0, &rhs, o, dt, rho);
            }
            self.receipt.cycles = cycle + 1;
            let candidate = self.levels[0].norm(&rhs, dt, rho);
            if !self.accept_candidate(candidate, &mut best, &mut best_residual) {
                self.receipt.rejected_cycles += 1;
                // Once multigrid diverges, do not feed its correction back into
                // another cycle. Recover using only projected fine-grid sweeps.
                self.recover(
                    &rhs,
                    dt,
                    rho,
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
        self.levels[0].rhs = rhs;
        self.correction = correction;
        self.backup = backup;
        self.best = best;
        self.original_min = original_min;
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
            assert!(solver.levels[l + 1].minimum.contains(&FREE));
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
        let mut norm = l.norm(&rhs, 1.0, 1.0);
        let initial = norm;
        l.p.fill(1e20);
        let bad = l.norm(&rhs, 1.0, 1.0);
        assert!(!solver.accept_candidate(bad, &mut best, &mut norm));
        solver.recover(&rhs, 1.0, 1.0, 0.01, &mut best, &mut norm);
        assert!(norm < initial && norm <= 0.01, "residual {norm}");
        assert!(solver.receipt.converged);
        assert!(solver.receipt.recovery_sweeps > 0 && solver.receipt.recovery_sweeps <= 64);
        assert!(solver.levels[0]
            .p
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0));
        // Disabled early stopping exhausts the cap honestly, retaining the best.
        solver.receipt = PressureReceipt::default();
        solver.recover(&rhs, 1.0, 1.0, 0.0, &mut best, &mut norm);
        assert!(solver.receipt.recovery_exhausted);
        assert_eq!(solver.receipt.recovery_sweeps, 64);
        assert_eq!(solver.levels[0].p, best);
    }
}
