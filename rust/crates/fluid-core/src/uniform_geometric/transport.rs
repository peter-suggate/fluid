//! The four-donor + identity reduction of uniform-volume.wgsl.ts.
//! Accumulation follows uniform-volume-donor-sum.wgsl.ts: exact nonnegative
//! f32 sums followed by one ties-to-even rounding, independent of visit order.
use crate::types::ValidationError;

/// The 192-bit integer of the GPU's six u32 words, held as three u64 limbs.
#[derive(Clone, Copy, Debug, Default)]
struct ExactSum([u64; 3]);
impl ExactSum {
    fn add(&mut self, value: f32) {
        let bits = value.to_bits();
        if bits == 0 {
            return;
        }
        let exponent = bits >> 23;
        let mantissa = if exponent == 0 {
            bits & 0x7fffff
        } else {
            (bits & 0x7fffff) | 0x800000
        } as u128;
        let shift = exponent.saturating_sub(1);
        // The GPU's six u32 words: an addend starting past the last word is
        // out of range, one starting at word six adds nothing.
        assert!(shift < 224, "exact sum addend {value} out of range");
        // The addend's two limbs at limb shift / 64, as a branch-free
        // 192-bit add; bits past the top limb drop.
        let wide = mantissa << (shift % 64);
        let (low, high) = (wide as u64, (wide >> 64) as u64);
        let limb = shift / 64;
        let pick = |at: u32, v: u64| if limb == at { v } else { 0 };
        let addend = [
            pick(0, low),
            pick(0, high) | pick(1, low),
            pick(1, high) | pick(2, low),
        ];
        let w = &mut self.0;
        let (s0, c0) = w[0].overflowing_add(addend[0]);
        let (s1, c1) = w[1].overflowing_add(addend[1]);
        let (s1, c2) = s1.overflowing_add(c0 as u64);
        w[2] = w[2].wrapping_add(addend[2]).wrapping_add((c1 | c2) as u64);
        w[0] = s0;
        w[1] = s1;
    }
    fn value(self) -> f32 {
        let [w0, w1, w2] = self.0;
        // The 128 bits holding the leading one, their base bit, and whether
        // any bit below them is set.
        let (x, base, below) = if w2 != 0 {
            (((w2 as u128) << 64) | w1 as u128, 64, w0 != 0)
        } else {
            (((w1 as u128) << 64) | w0 as u128, 0, false)
        };
        if x < 0x800000 {
            return f32::from_bits(x as u32);
        }
        let shift = 104 - x.leading_zeros();
        let mut mantissa = (x >> shift) as u32 & 0xffffff;
        let mut exponent = base + 105 - x.leading_zeros();
        let guard = ((x << 1) >> shift) as u32 & 1;
        let sticky = (x & (((1_u128 << shift) - 1) >> 1)) != 0 || below;
        mantissa += guard & (sticky as u32 | mantissa & 1);
        let carry = mantissa >> 24;
        mantissa >>= carry;
        exponent += carry;
        f32::from_bits((exponent << 23) | (mantissa & 0x7fffff))
    }
}

/// This step's external volume sources per cell (uniform-volume.wgsl.ts
/// dropSource and inflowSweptPlugSource). Empty vectors mean no source.
#[derive(Clone, Debug, Default)]
pub struct Sources {
    pub drop: Vec<f32>,
    pub plug: Vec<f32>,
}
impl Sources {
    pub fn at(&self, i: usize) -> bool {
        self.drop.get(i).is_some_and(|&v| v > 0.0) || self.plug.get(i).is_some_and(|&v| v > 0.0)
    }
    pub fn any(&self) -> bool {
        !self.drop.is_empty() || !self.plug.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Edges {
    pub donors: [usize; 5],
    pub weights: [f32; 5],
}
#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransportReceipt {
    pub initial_volume: f64,
    pub final_volume: f64,
    pub dust_volume: f64,
    pub dust_cells: usize,
    pub fallback_donors: usize,
    pub max_donor_error: f32,
    pub skipped_cells: usize,
    pub injected_volume: f64,
}

/// Scratch is owned by the world and reused on every step.
pub struct Transport {
    pub edges: Vec<Edges>,
    exact: Vec<ExactSum>,
    sums: Vec<f32>,
    /// uvDonorTiles at cell resolution: every donor of a live receiver this
    /// step (receivers are their own identity donor). Donor sums are cleared,
    /// accumulated and finished on this set only; no other word is read.
    donors: Vec<u32>,
    stamp: Vec<u32>,
    step: u32,
}
impl Transport {
    pub fn new(cells: usize) -> Self {
        Self {
            edges: vec![Edges::default(); cells],
            exact: vec![ExactSum::default(); cells],
            sums: vec![0.0; cells],
            donors: Vec::new(),
            stamp: vec![0; cells],
            step: 0,
        }
    }
    fn clear_sums(&mut self) {
        for &d in &self.donors {
            self.exact[d as usize] = ExactSum::default();
        }
    }
    fn finish_sums(&mut self) {
        for &d in &self.donors {
            self.sums[d as usize] = self.exact[d as usize].value();
        }
    }
    /// `cells` are the live receivers in ascending raster order and
    /// `departures[k]` is cell k's centre after the exact RK2 trace and solid
    /// clipping, in lattice units. Capacity is in cell areas (unit depth).
    /// Every other cell of `out` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &mut self,
        dims: [usize; 2],
        departures: &[[f32; 2]],
        capacity: &[f32],
        volume: &[f32],
        dust: f32,
        cells: &[u32],
        sources: &Sources,
        out: &mut [f32],
    ) -> Result<TransportReceipt, ValidationError> {
        let n = dims[0]
            .checked_mul(dims[1])
            .ok_or_else(|| ValidationError("uniform dimensions overflow".into()))?;
        let invalid_capacity = capacity
            .iter()
            .fold(false, |bad, v| bad | !(*v >= 0.0 && *v <= 1.0));
        let invalid_volume = volume.iter().fold(false, |bad, v| bad | !v.is_finite());
        if n == 0
            || departures.len() != cells.len()
            || [capacity.len(), volume.len(), out.len(), self.edges.len()]
                .iter()
                .any(|&len| len != n)
            || !dust.is_finite()
            || dust < 0.0
            || cells.last().is_some_and(|&i| i as usize >= n)
            || cells.windows(2).any(|w| w[0] >= w[1])
            || departures.iter().flatten().any(|v| !v.is_finite())
            || invalid_capacity
            || invalid_volume
        {
            return Err(ValidationError("invalid uniform transport inputs".into()));
        }
        let mut receipt = TransportReceipt {
            initial_volume: volume.iter().map(|&v| v as f64).sum(),
            skipped_cells: n - cells.len(),
            ..Default::default()
        };
        self.step = self.step.wrapping_add(1);
        if self.step == 0 {
            self.stamp.fill(0);
            self.step = 1;
        }
        self.donors.clear();
        for (&cell, &p) in cells.iter().zip(departures) {
            let i = cell as usize;
            let q = [p[0] - 0.5, p[1] - 0.5];
            let base = [q[0].floor() as i32, q[1].floor() as i32];
            let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
            let edge = &mut self.edges[i];
            edge.donors.fill(i);
            edge.weights.fill(0.0);
            if self.stamp[i] != self.step {
                self.stamp[i] = self.step;
                self.donors.push(cell);
                self.exact[i] = ExactSum::default();
            }
            for k in 0..4 {
                let x = base[0] + (k & 1) as i32;
                let y = base[1] + ((k >> 1) & 1) as i32;
                if x < 0
                    || y < 0
                    || x >= dims[0] as i32
                    || y >= dims[1] as i32
                    || capacity[i] <= 0.0
                {
                    continue;
                }
                let donor = x as usize + dims[0] * y as usize;
                let weight = (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                    * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] })
                    * capacity[i].min(capacity[donor]);
                edge.donors[k] = donor;
                edge.weights[k] = weight;
                if self.stamp[donor] != self.step {
                    self.stamp[donor] = self.step;
                    self.donors.push(donor as u32);
                    self.exact[donor] = ExactSum::default();
                }
                self.exact[donor].add(weight);
            }
        }
        self.finish_sums();
        for &cell in cells {
            let i = cell as usize;
            if self.sums[i] == 0.0 {
                self.edges[i].weights[4] = capacity[i].max(1e-6);
                if capacity[i] > 0.0 {
                    receipt.fallback_donors += 1;
                }
            }
        }
        for _ in 0..3 {
            self.clear_sums();
            let (edges, exact) = (&mut self.edges[..], &mut self.exact[..]);
            for &cell in cells {
                let edge = &mut edges[cell as usize];
                let sum = edge.weights.iter().fold(0.0_f32, |a, &b| a + b);
                let scale = capacity[cell as usize] / sum.max(1e-20);
                for k in 0..5 {
                    edge.weights[k] *= scale;
                    exact[edge.donors[k]].add(edge.weights[k]);
                }
            }
            self.finish_sums();
            let (edges, sums) = (&mut self.edges[..], &self.sums[..]);
            for &cell in cells {
                let edge = &mut edges[cell as usize];
                for k in 0..5 {
                    edge.weights[k] /= sums[edge.donors[k]].max(1e-20);
                }
            }
        }
        // uvGather: outside the live set V is zero by the dust-floor predicate;
        // a sealed-solid reservoir (open <= 0) keeps its V.
        self.clear_sums();
        out.fill(0.0);
        let (edges, exact) = (&self.edges[..], &mut self.exact[..]);
        for &cell in cells {
            let i = cell as usize;
            if capacity[i] <= 0.0 {
                out[i] = volume[i];
                continue;
            }
            let edge = &edges[i];
            let mut value = 0.0;
            for k in 0..5 {
                value += edge.weights[k] * volume[edge.donors[k]];
                exact[edge.donors[k]].add(edge.weights[k]);
            }
            let before = value;
            if let Some(&amount) = sources.drop.get(i) {
                value += amount.min((capacity[i] - value).max(0.0));
            }
            if let Some(&amount) = sources.plug.get(i) {
                value += amount;
            }
            receipt.injected_volume += (value - before) as f64;
            // uvDustFloor: a zero threshold (or NaN) never compares true.
            if value != 0.0 && value.abs() < dust {
                receipt.dust_cells += 1;
                receipt.dust_volume += value as f64;
                value = 0.0;
            }
            out[i] = value;
        }
        self.finish_sums();
        for &cell in cells {
            let i = cell as usize;
            if capacity[i] > 0.0 {
                receipt.max_donor_error = receipt.max_donor_error.max((self.sums[i] - 1.0).abs());
            }
        }
        receipt.final_volume = out.iter().map(|&v| v as f64).sum();
        Ok(receipt)
    }
}
