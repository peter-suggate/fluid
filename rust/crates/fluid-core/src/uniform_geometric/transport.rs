//! The four-donor + identity reduction of uniform-volume.wgsl.ts.
//! Accumulation follows uniform-volume-donor-sum.wgsl.ts: exact nonnegative
//! f32 sums followed by one ties-to-even rounding, independent of visit order.
use crate::types::ValidationError;

#[derive(Clone, Copy, Debug, Default)]
struct ExactSum([u32; 6]);
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
        };
        let shift = exponent.saturating_sub(1);
        let limb = (shift / 32) as usize;
        let offset = shift % 32;
        let mut carry = (mantissa as u64) << offset;
        for word in &mut self.0[limb..] {
            let sum = *word as u64 + (carry & 0xffffffff);
            *word = sum as u32;
            carry = (carry >> 32) + (sum >> 32);
            if carry == 0 {
                break;
            }
        }
    }
    fn value(self) -> f32 {
        let w = self.0;
        let top = (0..6).rev().find(|&i| w[i] != 0).unwrap_or(0);
        if top == 0 && w[0] < 0x800000 {
            return f32::from_bits(w[0]);
        }
        let highest = top as u32 * 32 + 31 - w[top].leading_zeros();
        let shift = highest - 23;
        let limb = (shift / 32) as usize;
        let offset = shift % 32;
        let mut mantissa = w[limb] >> offset;
        if offset != 0 && limb + 1 < 6 {
            mantissa |= w[limb + 1] << (32 - offset);
        }
        let mut exponent = highest - 22;
        if shift > 0 {
            let guard_index = ((shift - 1) / 32) as usize;
            let guard_bit = (shift - 1) % 32;
            let guard = (w[guard_index] >> guard_bit) & 1;
            let sticky = w[..guard_index]
                .iter()
                .fold(w[guard_index] & ((1 << guard_bit) - 1), |a, b| a | b);
            if guard != 0 && (sticky != 0 || mantissa & 1 != 0) {
                mantissa += 1;
                if mantissa == 0x1000000 {
                    mantissa >>= 1;
                    exponent += 1;
                }
            }
        }
        f32::from_bits((exponent << 23) | (mantissa & 0x7fffff))
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
}

/// Scratch is owned by the world and reused on every step.
pub struct Transport {
    pub edges: Vec<Edges>,
    exact: Vec<ExactSum>,
    sums: Vec<f32>,
}
impl Transport {
    pub fn new(cells: usize) -> Self {
        Self {
            edges: vec![Edges::default(); cells],
            exact: vec![ExactSum::default(); cells],
            sums: vec![0.0; cells],
        }
    }
    fn finish_sums(&mut self) {
        for (sum, exact) in self.sums.iter_mut().zip(&self.exact) {
            *sum = exact.value();
        }
    }
    /// Departure points are receiver centres in lattice units, after the exact
    /// RK2 trace and solid clipping. Capacity is in cell areas (unit depth).
    pub fn advance(
        &mut self,
        dims: [usize; 2],
        departures: &[[f32; 2]],
        capacity: &[f32],
        volume: &[f32],
        dust: f32,
        active: &[bool],
        out: &mut [f32],
    ) -> Result<TransportReceipt, ValidationError> {
        let n = dims[0]
            .checked_mul(dims[1])
            .ok_or_else(|| ValidationError("uniform dimensions overflow".into()))?;
        if n == 0
            || [
                departures.len(),
                active.len(),
                capacity.len(),
                volume.len(),
                out.len(),
                self.edges.len(),
            ]
            .iter()
            .any(|&len| len != n)
            || !dust.is_finite()
            || dust < 0.0
            || departures.iter().flatten().any(|v| !v.is_finite())
            || capacity
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || volume.iter().any(|v| !v.is_finite())
        {
            return Err(ValidationError("invalid uniform transport inputs".into()));
        }
        let mut receipt = TransportReceipt {
            initial_volume: volume.iter().map(|&v| v as f64).sum(),
            ..Default::default()
        };
        self.exact.fill(ExactSum::default());
        for (i, p) in departures.iter().enumerate() {
            if !active[i] {
                continue;
            }
            let q = [p[0] - 0.5, p[1] - 0.5];
            let base = [q[0].floor() as i32, q[1].floor() as i32];
            let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
            let edge = &mut self.edges[i];
            edge.donors.fill(i);
            edge.weights.fill(0.0);
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
                self.exact[donor].add(weight);
            }
        }
        self.finish_sums();
        for i in 0..n {
            if active[i] && self.sums[i] == 0.0 {
                self.edges[i].weights[4] = capacity[i].max(1e-6);
                if capacity[i] > 0.0 {
                    receipt.fallback_donors += 1;
                }
            }
        }
        for _ in 0..3 {
            self.exact.fill(ExactSum::default());
            for (i, edge) in self.edges.iter_mut().enumerate() {
                if !active[i] {
                    continue;
                }
                let sum = edge.weights.iter().fold(0.0_f32, |a, &b| a + b);
                let scale = capacity[i] / sum.max(1e-20);
                for k in 0..5 {
                    edge.weights[k] *= scale;
                    self.exact[edge.donors[k]].add(edge.weights[k]);
                }
            }
            self.finish_sums();
            for (i, edge) in self.edges.iter_mut().enumerate() {
                if !active[i] {
                    continue;
                }
                for k in 0..5 {
                    edge.weights[k] /= self.sums[edge.donors[k]].max(1e-20);
                }
            }
        }
        self.exact.fill(ExactSum::default());
        for (i, edge) in self.edges.iter().enumerate() {
            if !active[i] {
                out[i] = 0.0;
                receipt.skipped_cells += 1;
                continue;
            }
            let mut value = 0.0;
            for k in 0..5 {
                value += edge.weights[k] * volume[edge.donors[k]];
                self.exact[edge.donors[k]].add(edge.weights[k]);
            }
            if value != 0.0 && value.abs() < dust {
                receipt.dust_cells += 1;
                receipt.dust_volume += value as f64;
                value = 0.0;
            }
            out[i] = value;
        }
        self.finish_sums();
        for i in 0..n {
            if capacity[i] > 0.0 && active[i] {
                receipt.max_donor_error = receipt.max_donor_error.max((self.sums[i] - 1.0).abs());
            }
        }
        receipt.final_volume = out.iter().map(|&v| v as f64).sum();
        Ok(receipt)
    }
}
