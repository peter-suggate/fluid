//! Dense cell, vertex and positive-MAC lattices in physical +Y coordinates.
#[derive(Clone)]
pub struct Grid {
    pub dims: [usize; 2],
    pub h: [f32; 2],
    pub open_top: bool,
    pub volume: Vec<f32>,
    pub capacity: Vec<f32>,
    pub phi: Vec<f32>,
    pub velocity: Vec<[f32; 2]>,
    pub low_x: Vec<f32>,
    pub low_y: Vec<f32>,
    pub released: Vec<u8>,
    pub drops: Vec<LiquidDrop>,
    pub rigid_faces: Option<Vec<[f32; 2]>>,
    pub rigid_speeds: Option<Vec<[f32; 2]>>,
    pub rigid_centres: Option<Vec<bool>>,
}
impl Grid {
    /// Bound allocations before accepting an external scene or diagnostic request.
    /// This is the backend's address-space budget, not a numerical option.
    pub fn new(
        dims: [usize; 2],
        h: [f32; 2],
        open_top: bool,
    ) -> Result<Self, crate::types::ValidationError> {
        let invalid = || {
            crate::types::ValidationError("uniform grid must have at least two cells per axis, positive finite spacing, and at most 4194304 cells".into())
        };
        let n = dims[0].checked_mul(dims[1]).ok_or_else(invalid)?;
        if dims.iter().any(|&n| n < 2 || n > i32::MAX as usize - 2)
            || n > 4_194_304
            || h.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid());
        }
        Ok(Self {
            dims,
            h,
            open_top,
            volume: vec![0.0; n],
            capacity: vec![1.0; n],
            phi: vec![1.0; (dims[0] + 1) * (dims[1] + 1)],
            velocity: vec![[0.0; 2]; n],
            low_x: vec![0.0; dims[1]],
            low_y: vec![0.0; dims[0]],
            released: vec![0; n],
            rigid_faces: None,
            rigid_speeds: None,
            rigid_centres: None,
            drops: Vec::new(),
        })
    }
    pub fn drop_fraction(&self, p: [i32; 2]) -> f32 {
        let mut value = 0.0_f32;
        for drop in &self.drops {
            let mut covered = 0.0;
            for k in 0..4 {
                let x = (p[0] as f32 + 0.25 + 0.5 * (k & 1) as f32) * self.h[0] - drop.centre_m[0];
                let y = (p[1] as f32 + 0.25 + 0.5 * (k >> 1) as f32) * self.h[1] - drop.centre_m[1];
                if x.hypot(y) <= drop.radius_m {
                    covered += 0.25;
                }
            }
            value = (value + covered).min(1.0);
        }
        value
    }
    pub fn source_phi(&self, p: [f32; 2], mut phi: f32) -> f32 {
        for drop in &self.drops {
            phi = phi.min(
                (p[0] * self.h[0] - drop.centre_m[0]).hypot(p[1] * self.h[1] - drop.centre_m[1])
                    - drop.radius_m,
            );
        }
        phi
    }
    pub fn index(&self, p: [i32; 2]) -> Option<usize> {
        (p[0] >= 0 && p[1] >= 0 && p[0] < self.dims[0] as i32 && p[1] < self.dims[1] as i32)
            .then(|| p[0] as usize + self.dims[0] * p[1] as usize)
    }
    pub fn point(&self, i: usize) -> [i32; 2] {
        [(i % self.dims[0]) as i32, (i / self.dims[0]) as i32]
    }
    pub fn clamp(&self, p: [f32; 2]) -> [f32; 2] {
        [
            p[0].clamp(0.0, self.dims[0] as f32),
            p[1].clamp(0.0, self.dims[1] as f32),
        ]
    }
    pub fn clamp_cell(&self, p: [i32; 2]) -> [i32; 2] {
        [
            p[0].clamp(0, self.dims[0] as i32 - 1),
            p[1].clamp(0, self.dims[1] as i32 - 1),
        ]
    }
    pub fn open(&self, p: [i32; 2]) -> f32 {
        self.index(p).map_or(0.0, |i| self.capacity[i])
    }
    pub fn scalar(&self, field: &[f32], p: [f32; 2]) -> f32 {
        let p = self.clamp(p);
        let base = [
            (p[0].floor() as usize).min(self.dims[0] - 1),
            (p[1].floor() as usize).min(self.dims[1] - 1),
        ];
        let f = [p[0] - base[0] as f32, p[1] - base[1] as f32];
        let mut terms = [0.0; 4];
        for k in 0..4 {
            let x = k & 1;
            let y = k >> 1;
            terms[k] = field[base[0] + x + (self.dims[0] + 1) * (base[1] + y)]
                * (if x == 0 { 1.0 - f[0] } else { f[0] })
                * (if y == 0 { 1.0 - f[1] } else { f[1] });
        }
        (terms[0] + terms[1]) + (terms[2] + terms[3])
    }
    pub fn phi_at(&self, p: [f32; 2]) -> f32 {
        self.scalar(&self.phi, p)
    }
    pub fn gradient(&self, field: &[f32], p: [f32; 2]) -> [f32; 2] {
        std::array::from_fn(|a| {
            let mut lo = p;
            let mut hi = p;
            lo[a] -= 0.25;
            hi[a] += 0.25;
            lo = self.clamp(lo);
            hi = self.clamp(hi);
            (self.scalar(field, hi) - self.scalar(field, lo)) / (hi[a] - lo[a]).max(1e-6)
        })
    }
    pub fn pressure_phi(&self, p: [i32; 2], rows: &str) -> f32 {
        let h = self.h[0].min(self.h[1]);
        if let Some(i) = self.index(p).filter(|&i| self.capacity[i] > 1e-5) {
            let phi = self.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]);
            if rows == "off" {
                return phi;
            }
            let volume_phi = h * (0.5 - self.volume[i] / self.capacity[i]);
            if rows == "all" {
                return phi.min(volume_phi);
            }
            if phi < 0.0 || volume_phi >= 0.0 {
                return phi;
            }
            let isolated = OFFSETS.iter().all(|e| {
                let q = [p[0] + e[0], p[1] + e[1]];
                self.index(q).is_none()
                    || self.phi_at([q[0] as f32 + 0.5, q[1] as f32 + 0.5]) >= 0.0
            });
            return if isolated {
                volume_phi.max(-0.5 * h)
            } else {
                phi
            };
        }
        if self.open_top && p[1] == self.dims[1] as i32 && p[0] >= 0 && p[0] < self.dims[0] as i32 {
            return 0.5 * h;
        }
        let mut terms = [0.0; 4];
        let mut weights = [0.0; 4];
        for (i, e) in OFFSETS.iter().enumerate() {
            let q = [p[0] + e[0], p[1] + e[1]];
            let open = self.open(q);
            if open <= 1e-5 {
                continue;
            }
            // q is open, so this takes only the surface branch above. Solid
            // continuation must include any enabled volume-supported row.
            let phi = self.pressure_phi(q, rows);
            if phi < 0.0 {
                terms[i] = open * phi;
                weights[i] = open;
            }
        }
        let sum = (terms[0] + terms[1]) + (terms[2] + terms[3]);
        let weight = (weights[0] + weights[1]) + (weights[2] + weights[3]);
        if weight > 0.0 {
            sum / weight.max(1e-9)
        } else {
            0.5 * h
        }
    }
    pub fn solid_speed(&self, p: [i32; 2], a: usize) -> f32 {
        self.index(p)
            .and_then(|i| self.rigid_speeds.as_ref().map(|v| v[i][a]))
            .unwrap_or(0.0)
    }
    pub fn pressure_face(&self, p: [i32; 2], a: usize) -> f32 {
        self.index(p)
            .and_then(|i| self.rigid_faces.as_ref().map(|v| v[i][a]))
            .unwrap_or_else(|| self.base_pressure_face(p, a))
    }
    pub fn base_pressure_face(&self, p: [i32; 2], a: usize) -> f32 {
        let mut q = p;
        q[a] += 1;
        let x = self.open(p);
        let y = self.open(q);
        if self.index(p).is_some() != self.index(q).is_some() {
            let inside = if self.index(p).is_some() { x } else { y };
            if a == 1 && q[1] == self.dims[1] as i32 && self.open_top {
                0.5 * (1.0 + inside)
            } else {
                0.5 * inside
            }
        } else if self.index(p).is_none() {
            0.0
        } else {
            0.5 * (x + y)
        }
    }
    pub fn face(&self, p: [i32; 2], a: usize) -> f32 {
        if a == 0 && p[0] == -1 && p[1] >= 0 && p[1] < self.dims[1] as i32 {
            return self.low_x[p[1] as usize];
        }
        if a == 1 && p[1] == -1 && p[0] >= 0 && p[0] < self.dims[0] as i32 {
            return self.low_y[p[0] as usize];
        }
        self.index(p).map_or(0.0, |i| self.velocity[i][a])
    }
    pub fn relative_face(&self, p: [i32; 2], a: usize) -> f32 {
        let velocity = self.face(p, a);
        if self.rigid_speeds.is_some() {
            velocity - self.solid_speed(p, a)
        } else {
            velocity
        }
    }
    pub fn is_released(&self, p: [i32; 2], a: usize) -> bool {
        let mut q = p;
        let mut bit = a;
        if q[a] < 0 {
            q[a] = 0;
            bit += 2;
        }
        self.index(q)
            .is_some_and(|i| self.released[i] & (1 << bit) != 0)
    }
    pub fn target(&self, p: [i32; 2]) -> f32 {
        let mut samples = [0.0; 4];
        let mut centre = 0.0;
        let mut fill = 0.0;
        let mut magnitude = 0.0_f32;
        let mut gradient = [0.0; 2];
        for k in 0..4 {
            let c = [(k & 1) as f32, ((k >> 1) & 1) as f32];
            let v = self.phi_at([
                p[0] as f32 + 0.25 + 0.5 * c[0],
                p[1] as f32 + 0.25 + 0.5 * c[1],
            ]);
            samples[k] = v;
            centre += 0.25 * v;
            magnitude = magnitude.max(v.abs());
            fill += if v < 0.0 {
                1.0
            } else if v == 0.0 {
                0.5
            } else {
                0.0
            };
            for a in 0..2 {
                gradient[a] += (2.0 * c[a] - 1.0) * v;
            }
        }
        let mut residual = 0.0_f32;
        for k in 0..4 {
            let signs = [
                2.0 * (k & 1) as f32 - 1.0,
                2.0 * ((k >> 1) & 1) as f32 - 1.0,
            ];
            residual = residual.max(
                (samples[k] - (centre + 0.25 * (gradient[0] * signs[0] + gradient[1] * signs[1])))
                    .abs(),
            );
        }
        let fraction = if residual <= 1e-4 * (1.0 + magnitude) {
            plane_fraction(gradient, -centre)
        } else {
            fill * 0.25
        };
        fraction * self.open(p)
    }
}
pub const OFFSETS: [[i32; 2]; 4] = [[-1, 0], [1, 0], [0, -1], [0, 1]];
/// Exact area of n·x <= offset in the centred unit square.
pub fn plane_fraction(n: [f32; 2], offset: f32) -> f32 {
    let dominant = n[0].abs().max(n[1].abs());
    if dominant <= 1e-20 {
        return if offset >= 0.0 { 1.0 } else { 0.0 };
    }
    let mut spans = [0.0; 2];
    let mut dimensions = 0;
    for value in n {
        let span = value.abs() / dominant;
        if span >= 1e-6 {
            spans[dimensions] = span;
            dimensions += 1;
        }
    }
    let total = spans[0] + spans[1];
    let shifted = offset / dominant + 0.5 * total;
    if shifted <= 0.0 {
        return 0.0;
    }
    if shifted >= total {
        return 1.0;
    }
    let complement = shifted > 0.5 * total;
    let x = if complement { total - shifted } else { shifted };
    let f = if dimensions == 1 {
        x / spans[0]
    } else {
        let a = spans[0].min(spans[1]);
        let b = spans[0].max(spans[1]);
        if x < a {
            0.5 * (x / a) * (x / b)
        } else {
            (x - 0.5 * a) / b
        }
    }
    .clamp(0.0, 1.0);
    if complement {
        1.0 - f
    } else {
        f
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiquidDrop {
    #[serde(rename = "centre_m")]
    pub centre_m: [f32; 2],
    #[serde(rename = "radius_m")]
    pub radius_m: f32,
}
