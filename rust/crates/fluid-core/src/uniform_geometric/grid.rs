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
        // uvSourcePhi's drop arm; WGSL length, not hypot.
        for drop in &self.drops {
            let d = [
                p[0] * self.h[0] - drop.centre_m[0],
                p[1] * self.h[1] - drop.centre_m[1],
            ];
            phi = phi.min((d[0] * d[0] + d[1] * d[1]).sqrt() - drop.radius_m);
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
    /// Catmull-Rom phi sample with the enclosing four vertices as a monotonic
    /// bound, matching uvPhiCubic on the physical 2D slice.
    pub fn phi_cubic(&self, p: [f32; 2]) -> f32 {
        let p = self.clamp(p);
        let base = [
            (p[0].floor() as usize).min(self.dims[0] - 1),
            (p[1].floor() as usize).min(self.dims[1] - 1),
        ];
        let weights = base.map(|_| [0.0_f32; 4]);
        let mut weights = weights;
        for a in 0..2 {
            let t = p[a] - base[a] as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            weights[a] = [
                0.5 * (2.0 * t2 - t3 - t),
                0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
                0.5 * (4.0 * t2 - 3.0 * t3 + t),
                0.5 * (t3 - t2),
            ];
        }
        let mut value = 0.0;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for dy in -1..3 {
            let mut row = 0.0;
            for dx in -1..3 {
                let x = (base[0] as i32 + dx).clamp(0, self.dims[0] as i32) as usize;
                let y = (base[1] as i32 + dy).clamp(0, self.dims[1] as i32) as usize;
                let sample = self.phi[x + (self.dims[0] + 1) * y];
                row += weights[0][(dx + 1) as usize] * sample;
                if (0..=1).contains(&dx) && (0..=1).contains(&dy) {
                    lo = lo.min(sample);
                    hi = hi.max(sample);
                }
            }
            value += weights[1][(dy + 1) as usize] * row;
        }
        value.clamp(lo, hi)
    }
    /// Ballistic V has neither a phi pressure row nor a nearby solid or wall.
    pub fn airborne(&self, p: [i32; 2], enabled: bool, dust: f32) -> bool {
        if !enabled
            || self
                .index(p)
                .is_none_or(|i| self.volume[i] <= dust.max(0.05))
        {
            return false;
        }
        let h = self.h[0].min(self.h[1]);
        if self.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]) <= 1.5 * h {
            return false;
        }
        for y in p[1] - 2..=p[1] + 2 {
            for x in p[0] - 2..=p[0] + 2 {
                if self
                    .index([x, y])
                    .is_none_or(|i| self.capacity[i] < 0.99999)
                {
                    return false;
                }
            }
        }
        true
    }
    /// uvPhi(cell + 0.5) as Metal compiles it into the pressure build: the
    /// depth-symmetric corners k and k+4 are one plane vertex, and each
    /// x row's four eighths are summed in sequence. Verified bitwise against
    /// the GPU CM11a finest level; the source's pairwise d4Sum8 is not.
    pub fn phi_centre(&self, cell: [i32; 2]) -> f32 {
        let row = self.dims[0] + 1;
        let base = cell[0] as usize + row * cell[1] as usize;
        let v = |k: usize| self.phi[k] * 0.125;
        let [a, b, c, d] = [v(base), v(base + 1), v(base + row), v(base + row + 1)];
        (((a + b) + a) + b) + (((c + d) + c) + d)
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
            let phi = self.phi_centre(p);
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
                self.index(q).is_none() || self.phi_centre(q) >= 0.0
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
    /// No cut cell anywhere (uvSolidFree): the host's static-solid, terrain and body certificate.
    pub fn solid_free(&self) -> bool {
        self.capacity.iter().all(|&c| c == 1.0)
            && self.rigid_faces.is_none()
            && self.rigid_centres.is_none()
    }
    /// uvBuried: no incident cell of the vertex is open. Its phi is not state.
    pub fn buried(&self, v: [i32; 2]) -> bool {
        (0..4).all(|k| {
            let c = [v[0] - 1 + (k & 1), v[1] - 1 + (k >> 1)];
            self.index(c).is_none_or(|i| self.capacity[i] <= 1e-5)
        })
    }
    /// uvTarget, cached arm (uniform-volume.wgsl.ts): eight quarter-cell probes
    /// of the z-duplicated vertex octet, each a d4Sum8 of ((v*wx)*wy)*wz.
    pub fn target(&self, p: [i32; 2]) -> f32 {
        target_of(&self.phi, self.dims, p) * self.open(p)
    }
}
fn corner3(k: usize) -> [usize; 3] {
    [k & 1, (k >> 1) & 1, (k >> 2) & 1]
}
/// uvTarget of an arbitrary vertex field, before the open-fraction factor.
pub fn target_of(phi: &[f32], dims: [usize; 2], p: [i32; 2]) -> f32 {
    let nxv = dims[0] + 1;
    let base = p[0] as usize + nxv * p[1] as usize;
    if let Some(settled) = settled_target([
        phi[base],
        phi[base + 1],
        phi[base + nxv],
        phi[base + nxv + 1],
    ]) {
        return settled;
    }
    let vertices: [f32; 8] = std::array::from_fn(|j| {
        let c = corner3(j);
        phi[base + c[0] + nxv * c[1]]
    });
    let mut samples = [0.0_f32; 8];
    let mut centre = 0.0_f32;
    let mut fill = 0.0_f32;
    let mut magnitude = 0.0_f32;
    for (k, sample) in samples.iter_mut().enumerate() {
        let f = corner3(k).map(|c| 0.25 + 0.5 * c as f32);
        let weighted: [f32; 8] = std::array::from_fn(|j| {
            let c = corner3(j);
            let w: [f32; 3] = std::array::from_fn(|a| if c[a] == 1 { f[a] } else { 1.0 - f[a] });
            ((vertices[j] * w[0]) * w[1]) * w[2]
        });
        let value = d4_sum8(weighted);
        *sample = value;
        centre += 0.125 * value;
        magnitude = magnitude.max(value.abs());
        fill += if value < 0.0 {
            1.0
        } else if value == 0.0 {
            0.5
        } else {
            0.0
        };
    }
    let mut gradient = [0.0_f32; 3];
    for (k, &s) in samples.iter().enumerate() {
        let c = corner3(k);
        for a in 0..3 {
            gradient[a] += ((2.0 * c[a] as f32 - 1.0) * s) / 2.0;
        }
    }
    let mut residual = 0.0_f32;
    for (k, &s) in samples.iter().enumerate() {
        let sign = corner3(k).map(|c| 2.0 * c as f32 - 1.0);
        let dot = (gradient[0] * (0.25 * sign[0]) + gradient[1] * (0.25 * sign[1]))
            + gradient[2] * (0.25 * sign[2]);
        residual = residual.max((s - (centre + dot)).abs());
    }
    if residual <= 1e-4 * (1.0 + magnitude) {
        plane_box_fraction(gradient, -centre)
    } else {
        fill / 8.0
    }
}
/// uvTarget's value, without evaluating it, for a cell whose four vertices
/// share one sign and lie within a factor 1.9 of each other. Every sample then
/// has that sign, and the fitted plane's half-cell swing (at most max - min)
/// stays at least a tenth of |centre| short of the cell's corners, far beyond
/// f32 rounding: both arms of uvTarget return exactly 0 (air) or 1 (liquid).
fn settled_target(v: [f32; 4]) -> Option<f32> {
    let lo = v[0].min(v[1]).min(v[2].min(v[3]));
    let hi = v[0].max(v[1]).max(v[2].max(v[3]));
    if !v.iter().all(|x| x.abs() <= 1e30) {
        return None;
    }
    if lo > 1e-30 && hi <= 1.9 * lo {
        Some(0.0)
    } else if hi < -1e-30 && lo >= 1.9 * hi {
        Some(1.0)
    } else {
        None
    }
}
/// d4Sum8 (webgpu-uniform-reference.wgsl.ts).
pub fn d4_sum8(v: [f32; 8]) -> f32 {
    ((v[0] + v[5]) + (v[1] + v[4])) + ((v[2] + v[7]) + (v[3] + v[6]))
}
pub const OFFSETS: [[i32; 2]; 4] = [[-1, 0], [1, 0], [0, -1], [0, 1]];
/// geometricPlaneBoxFraction with unit widths (geometric-plane-box.wgsl.ts).
pub fn plane_box_fraction(normal: [f32; 3], offset: f32) -> f32 {
    let projected = normal.map(f32::abs);
    let dominant = projected[0].max(projected[1].max(projected[2]));
    if dominant <= 1e-20 {
        return if offset >= 0.0 { 1.0 } else { 0.0 };
    }
    let mut spans = [0.0_f32; 3];
    let mut dimensions = 0;
    for value in projected {
        let span = value / dominant;
        if span >= 1e-6 {
            spans[dimensions] = span;
            dimensions += 1;
        }
    }
    let total = spans[0] + spans[1] + spans[2];
    let shifted = offset / dominant + 0.5 * total;
    if shifted <= 0.0 {
        return 0.0;
    }
    if shifted >= total {
        return 1.0;
    }
    let complement = shifted > 0.5 * total;
    let x = if complement { total - shifted } else { shifted };
    let fraction = if dimensions == 1 {
        x / spans[0]
    } else if dimensions == 2 {
        let a = spans[0].min(spans[1]);
        let b = spans[0].max(spans[1]);
        if x < a {
            (0.5 * (x / a)) * (x / b)
        } else {
            (x - 0.5 * a) / b
        }
    } else {
        let a = spans[0].min(spans[1].min(spans[2]));
        let c = spans[0].max(spans[1].max(spans[2]));
        let b = spans[0]
            .min(spans[1])
            .max(spans[0].max(spans[1]).min(spans[2]));
        (uniform_sum_primitive(x, a, b) - uniform_sum_primitive(x - c, a, b)) / c
    }
    .clamp(0.0, 1.0);
    if complement {
        1.0 - fraction
    } else {
        fraction
    }
}
/// geometricUniformSumPrimitive.
fn uniform_sum_primitive(x: f32, a: f32, b: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a {
        return ((x * (x / a)) * (x / b)) / 6.0;
    }
    if x <= b {
        return ((0.5 * x) * (x - a) + (a * a) / 6.0) / b;
    }
    if x < a + b {
        let tail = a + b - x;
        return (x - 0.5 * (a + b)) + ((tail * (tail / a)) * (tail / b)) / 6.0;
    }
    x - 0.5 * (a + b)
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiquidDrop {
    #[serde(rename = "centre_m")]
    pub centre_m: [f32; 2],
    #[serde(rename = "radius_m")]
    pub radius_m: f32,
}
