//! JRW07 active front and CM11b known-value hierarchy, reduced to two axes.
use super::{
    grid::{Grid, OFFSETS},
    options::UniformGeometricOptions,
};
const INF: f32 = 65504.0;
#[derive(Clone, Copy, Default)]
struct Face {
    value: [f32; 2],
    distance: [f32; 2],
    known: u8,
    active: u8,
}

#[cfg(test)]
mod conforming_tests {
    use super::*;
    #[test]
    fn reconstruction_has_mac_divergence_and_shared_normal_flux() {
        let mut g = Grid::new([12, 12], [0.05, 0.08], false).unwrap();
        g.phi.fill(-1.0);
        for i in 0..g.velocity.len() {
            let p = g.point(i);
            g.velocity[i] = [
                (0.3 * p[0] as f32).sin() + 0.02 * (p[1] as f32).powi(2),
                (0.2 * p[1] as f32).cos() + 0.01 * (p[0] as f32).powi(2),
            ];
        }
        let mut o = UniformGeometricOptions::default();
        o.two_level_velocity = "off".into();
        let mut e = Extension::build(&g, &o, 0.03);
        e.enable_conforming(&g);
        for cell in [[3, 4], [5, 6], [7, 3]] {
            let expected = (e.fine_face(cell, 0) - e.fine_face([cell[0] - 1, cell[1]], 0)) / g.h[0]
                + (e.fine_face(cell, 1) - e.fine_face([cell[0], cell[1] - 1], 1)) / g.h[1];
            for f in [[0.2, 0.3], [0.7, 0.8], [0.4, 0.6]] {
                let p = [cell[0] as f32 + f[0], cell[1] as f32 + f[1]];
                let mut div = 0.0;
                for a in 0..2 {
                    let mut lo = p;
                    let mut hi = p;
                    lo[a] -= 0.001;
                    hi[a] += 0.001;
                    div += (e.sample(hi)[a] - e.sample(lo)[a]) / ((hi[a] - lo[a]) * g.h[a]);
                }
                assert!((div - expected).abs() < 0.005, "{div} != {expected}");
            }
            let p = [cell[0] as f32 + 1.0, cell[1] as f32 + 0.37];
            assert!(
                (e.sample([p[0] - 1e-5, p[1]])[0] - e.sample([p[0] + 1e-5, p[1]])[0]).abs() < 1e-4
            );
            let flux = (e.sample([p[0], cell[1] as f32 + 0.25])[0]
                + e.sample([p[0], cell[1] as f32 + 0.75])[0])
                * 0.5;
            assert!((flux - e.fine_face(cell, 0)).abs() < 1e-6);
        }
    }
}
#[derive(Clone)]
pub struct Extension {
    dims: [usize; 2],
    pub values: Vec<[f32; 2]>,
    pub classes: Vec<u8>,
    coarse: Vec<[f32; 2]>,
    coarse_dims: [usize; 2],
    two_level: bool,
    conforming_slopes: Option<Vec<[f32; 2]>>,
    h: [f32; 2],
    pub trace_steps: usize,
}
fn index(d: [usize; 2], p: [i32; 2]) -> Option<usize> {
    (p[0] >= 0 && p[1] >= 0 && p[0] < d[0] as i32 && p[1] < d[1] as i32)
        .then(|| p[0] as usize + d[0] * p[1] as usize)
}
fn hierarchy_sample(
    source: &[Face],
    sd: [usize; 2],
    p: [usize; 2],
    td: [usize; 2],
    a: usize,
) -> (f32, f32) {
    let q: [f32; 2] = std::array::from_fn(|b| {
        if b == a {
            (p[b] + 1) as f32 * sd[b] as f32 / td[b] as f32 - 1.0
        } else {
            (p[b] as f32 + 0.5) * sd[b] as f32 / td[b] as f32 - 0.5
        }
    });
    let base = q.map(|v| v.floor() as i32);
    let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
    let mut sum = 0.0;
    let mut weight = 0.0;
    for k in 0..4 {
        let r = [
            (base[0] + (k & 1)).clamp(0, sd[0] as i32 - 1),
            (base[1] + ((k >> 1) & 1)).clamp(0, sd[1] as i32 - 1),
        ];
        let v = source[index(sd, r).unwrap()];
        let w = (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
            * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] });
        if v.known & (1 << a) != 0 && w > 0.0 {
            sum += w * v.value[a];
            weight += w;
        }
    }
    (if weight > 0.0 { sum / weight } else { 0.0 }, weight)
}
impl Extension {
    pub fn build(g: &Grid, o: &UniformGeometricOptions, dt: f32) -> Self {
        Self::build_inner(g, o, dt, None)
    }
    pub fn build_prediction(
        g: &Grid,
        o: &UniformGeometricOptions,
        dt: f32,
        original: &Self,
    ) -> Self {
        Self::build_inner(g, o, dt, Some(original))
    }
    fn build_inner(g: &Grid, o: &UniformGeometricOptions, dt: f32, workmap: Option<&Self>) -> Self {
        let dims = g.dims;
        let cd = dims.map(|n| n.div_ceil(4));
        let two = o.two_level_velocity == "on" && dims.iter().all(|n| n % 4 == 0);
        let mut seed = vec![false; cd[0] * cd[1]];
        let mut displacement = 0.0_f32;
        let band = 4.0 * g.h[0].max(g.h[1]);
        for i in 0..g.volume.len() {
            let p = g.point(i);
            let t = p[0] as usize / 4 + cd[0] * (p[1] as usize / 4);
            seed[t] |= g.volume[i].abs() > o.volume_dust_threshold.max(1e-6)
                || g.capacity[i] < 0.99999
                || g.drop_fraction(g.point(i)) > 0.0;
            for a in 0..2 {
                displacement = displacement.max(g.velocity[i][a].abs() * dt / g.h[a]);
            }
        }
        for y in 0..=dims[1] {
            for x in 0..=dims[0] {
                if g.phi[x + (dims[0] + 1) * y] < band {
                    for oy in 0..=1 {
                        for ox in 0..=1 {
                            let tx = (x as i32 - ox).max(0) as usize / 4;
                            let ty = (y as i32 - oy).max(0) as usize / 4;
                            if tx < cd[0] && ty < cd[1] {
                                seed[tx + cd[0] * ty] = true;
                            }
                        }
                    }
                }
            }
        }
        let reaches = [
            o.two_level_fine_reach as i32,
            (o.two_level_fine_reach + o.two_level_shell_reach.max(1.0)) as i32,
            if o.transport_work_map == "tiles" && o.volume_dust_threshold > 0.0 {
                (((displacement.ceil() + 1.0) / 4.0).ceil() as i32 + o.transport_reach as i32)
                    .clamp(0, 16)
            } else {
                0
            },
        ];
        let mut classes = vec![if two { 0 } else { 7 }; seed.len()];
        if two {
            for y in 0..cd[1] {
                for x in 0..cd[0] {
                    for (bit, &r) in reaches.iter().enumerate() {
                        'scan: for sy in
                            (y as i32 - r).max(0)..=(y as i32 + r).min(cd[1] as i32 - 1)
                        {
                            for sx in (x as i32 - r).max(0)..=(x as i32 + r).min(cd[0] as i32 - 1) {
                                if seed[sx as usize + cd[0] * sy as usize] {
                                    classes[x + cd[0] * y] |= 1 << bit;
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(original) = workmap {
            classes.clone_from(&original.classes);
        }
        let shell = |p: [i32; 2]| -> bool {
            !two || o.two_level_extension == "dense"
                || classes[p[0] as usize / 4 + cd[0] * (p[1] as usize / 4)] & 2 != 0
        };
        let mut open = vec![[false; 2]; g.volume.len()];
        let mut source = vec![[false; 2]; g.volume.len()];
        for i in 0..source.len() {
            let p = g.point(i);
            for a in 0..2 {
                let mut q = p;
                q[a] += 1;
                open[i][a] = g.pressure_face(p, a) > 1e-5 && shell(p);
                source[i][a] = open[i][a]
                    && (g.pressure_phi(p, &o.volume_pressure_rows) < 0.0
                        || g.index(q).is_some()
                            && g.pressure_phi(q, &o.volume_pressure_rows) < 0.0);
            }
        }
        let mut faces = vec![
            Face {
                distance: [INF; 2],
                ..Default::default()
            };
            source.len()
        ];
        for i in 0..faces.len() {
            let p = g.point(i);
            for a in 0..2 {
                if source[i][a] {
                    faces[i].known |= 1 << a;
                    faces[i].value[a] = g.velocity[i][a];
                    faces[i].distance[a] = 0.0;
                } else if open[i][a]
                    && OFFSETS.iter().any(|e| {
                        index(dims, [p[0] + e[0], p[1] + e[1]]).is_some_and(|j| source[j][a])
                    })
                {
                    faces[i].active |= 1 << a;
                }
            }
        }
        let epsilon = |d: f32| {
            d.abs().clamp(
                2.0 * g.h[0].min(g.h[1]),
                ((dims[0] as f32 * g.h[0]).powi(2) + (dims[1] as f32 * g.h[1]).powi(2)).sqrt(),
            ) * 1.1920929e-7
        };
        let distance = |faces: &[Face], p: [i32; 2], a: usize| {
            index(dims, p)
                .filter(|&i| open[i][a])
                .map_or(INF, |i| faces[i].distance[a])
        };
        let candidate = |faces: &[Face], p: [i32; 2], a: usize| {
            let mut minima = [0.0; 2];
            let mut h = g.h;
            for axis in 0..2 {
                let mut lo = p;
                let mut hi = p;
                lo[axis] -= 1;
                hi[axis] += 1;
                minima[axis] = distance(faces, lo, a).min(distance(faces, hi, a));
            }
            if minima[1] < minima[0] {
                minima.swap(0, 1);
                h.swap(0, 1);
            }
            if minima[0] >= 0.5 * INF {
                return INF;
            }
            let root = minima[0] + h[0];
            if minima[1] >= 0.5 * INF || root <= minima[1] {
                return root;
            }
            let x = 1.0 / (h[0] * h[0]);
            let y = 1.0 / (h[1] * h[1]);
            let aa = x + y;
            let b = minima[0] * x + minima[1] * y;
            let c = -1.0 + minima[0] * minima[0] * x + minima[1] * minima[1] * y;
            (b + (b * b - aa * c).max(0.0).sqrt()) / aa
        };
        for _ in 0..o.extension_front_sweeps as usize {
            if faces.iter().all(|f| f.active == 0) {
                break;
            }
            let mut next = faces.clone();
            let band = 2.0 * g.h[0].max(g.h[1]);
            for i in 0..faces.len() {
                let p = g.point(i);
                for a in 0..2 {
                    if !open[i][a] || source[i][a] {
                        continue;
                    }
                    let old = faces[i].distance[a];
                    let updated = old.min(candidate(&faces, p, a));
                    if faces[i].active & (1 << a) != 0 {
                        if updated <= band {
                            next[i].distance[a] = updated;
                            let mut sum = 0.0;
                            let mut weight = 0.0;
                            for axis in 0..2 {
                                let mut lo = p;
                                let mut hi = p;
                                lo[axis] -= 1;
                                hi[axis] += 1;
                                let dl = distance(&faces, lo, a);
                                let dh = distance(&faces, hi, a);
                                let minimum = dl.min(dh);
                                if minimum >= updated - epsilon(updated) {
                                    continue;
                                }
                                let mut v = 0.0;
                                let mut count = 0.0;
                                for (q, d) in [(lo, dl), (hi, dh)] {
                                    if (d - minimum).abs() <= epsilon(updated) {
                                        v += index(dims, q).map_or(0.0, |j| {
                                            if faces[j].known & (1 << a) != 0 {
                                                faces[j].value[a]
                                            } else {
                                                0.0
                                            }
                                        });
                                        count += 1.0;
                                    }
                                }
                                if count > 0.0 {
                                    let w = (updated - minimum) / (g.h[axis] * g.h[axis]);
                                    sum += w * v / count;
                                    weight += w;
                                }
                            }
                            next[i].value[a] = if weight > 0.0 { sum / weight } else { 0.0 };
                            next[i].known |= 1 << a;
                        }
                        if (old < 0.5 * INF && (updated - old).abs() <= epsilon(old))
                            || updated > band
                        {
                            next[i].active &= !(1 << a);
                        }
                    } else if updated <= band {
                        for e in OFFSETS {
                            let q = [p[0] + e[0], p[1] + e[1]];
                            if let Some(j) = index(dims, q) {
                                let nd = faces[j].distance[a];
                                let nu = nd.min(candidate(&faces, q, a));
                                if open[j][a]
                                    && !source[j][a]
                                    && faces[j].active & (1 << a) != 0
                                    && nd < 0.5 * INF
                                    && nu <= band
                                    && (nu - nd).abs() <= epsilon(nd)
                                    && old > nu + epsilon(old)
                                {
                                    next[i].active |= 1 << a;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            faces = next;
        }
        let mut levels = vec![(dims, faces)];
        while levels.last().unwrap().0.iter().any(|&n| n > 1) {
            let (sd, s) = levels.last().unwrap();
            let td = sd.map(|n| n.div_ceil(2));
            let mut target = vec![Face::default(); td[0] * td[1]];
            for y in 0..td[1] {
                for x in 0..td[0] {
                    for a in 0..2 {
                        let (mut value, mut weight) = hierarchy_sample(s, *sd, [x, y], td, a);
                        if weight <= 0.0 && a == 1 {
                            let mut sum = 0.0;
                            for k in 0..4 {
                                if let Some(j) = index(
                                    *sd,
                                    [(2 * x + (k & 1)) as i32, (2 * y + (k >> 1)) as i32],
                                ) {
                                    if s[j].known & (1 << a) != 0 {
                                        sum += s[j].value[a];
                                        weight += 1.0;
                                    }
                                }
                            }
                            if weight > 0.0 {
                                value = sum / weight;
                            }
                        }
                        if weight > 0.0 {
                            let v = &mut target[x + td[0] * y];
                            v.value[a] = value;
                            v.known |= 1 << a;
                        }
                    }
                }
            }
            levels.push((td, target));
        }
        for l in (0..levels.len() - 1).rev() {
            let td = levels[l].0;
            for y in 0..td[1] {
                for x in 0..td[0] {
                    for a in 0..2 {
                        let i = x + td[0] * y;
                        if levels[l].1[i].known & (1 << a) == 0 {
                            let (v, w) =
                                hierarchy_sample(&levels[l + 1].1, levels[l + 1].0, [x, y], td, a);
                            if w > 0.0 {
                                levels[l].1[i].value[a] = v;
                                levels[l].1[i].known |= 1 << a;
                            }
                        }
                    }
                }
            }
        }
        let coarse = levels.iter().find(|(d, _)| *d == cd).map_or_else(
            || vec![[0.0; 2]; cd[0] * cd[1]],
            |(_, f)| f.iter().map(|v| v.value).collect(),
        );
        let mut values: Vec<_> = levels[0].1.iter().map(|v| v.value).collect();
        for i in 0..values.len() {
            for a in 0..2 {
                if !open[i][a] {
                    values[i][a] = 0.0;
                }
            }
        }
        Self {
            dims,
            values,
            classes,
            coarse,
            coarse_dims: cd,
            two_level: two,
            conforming_slopes: None,
            h: g.h,
            trace_steps: 1,
        }
    }
    pub fn with_coarse_from(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.coarse.clone_from(&other.coarse);
        result
    }
    pub fn transport_at(&self, p: [i32; 2], o: &UniformGeometricOptions) -> bool {
        !self.two_level
            || o.transport_work_map != "tiles"
            || o.volume_dust_threshold <= 0.0
            || self.classes[p[0] as usize / 4 + self.coarse_dims[0] * (p[1] as usize / 4)] & 4 != 0
    }
    pub fn fine_at(&self, p: [f32; 2]) -> bool {
        let tile: [usize; 2] = std::array::from_fn(|a| {
            (p[a].floor() as i32).clamp(0, self.dims[a] as i32 - 1) as usize / 4
        });
        !self.two_level || self.classes[tile[0] + self.coarse_dims[0] * tile[1]] & 1 != 0
    }
    pub fn fine_face(&self, mut p: [i32; 2], a: usize) -> f32 {
        if p[a] < 0 || p[a] >= self.dims[a] as i32 {
            return 0.0;
        }
        p[1 - a] = p[1 - a].clamp(0, self.dims[1 - a] as i32 - 1);
        self.values[index(self.dims, p).unwrap()][a]
    }
    pub fn sample(&self, p: [f32; 2]) -> [f32; 2] {
        if self.conforming_slopes.is_some() && self.fine_at(p) {
            return self.conforming_sample(p);
        }
        let tile = [
            ((p[0].floor() as i32).clamp(0, self.dims[0] as i32 - 1) as usize) / 4,
            ((p[1].floor() as i32).clamp(0, self.dims[1] as i32 - 1) as usize) / 4,
        ];
        let coarse =
            self.two_level && self.classes[tile[0] + self.coarse_dims[0] * tile[1]] & 1 == 0;
        std::array::from_fn(|a| {
            let mut offset = [0.5; 2];
            offset[a] = 1.0;
            let d = if coarse { self.coarse_dims } else { self.dims };
            let scale = if coarse { 4.0 } else { 1.0 };
            let q: [f32; 2] = std::array::from_fn(|b| {
                (p[b] / scale - offset[b]).clamp(if b == a { -1.0 } else { 0.0 }, d[b] as f32 - 1.0)
            });
            let base = q.map(|v| v.floor() as i32);
            let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
            let mut terms = [0.0; 4];
            for k in 0..4 {
                let r = [base[0] + (k & 1) as i32, base[1] + (k >> 1) as i32];
                let v = if coarse {
                    index(d, r).map_or(0.0, |i| self.coarse[i][a])
                } else {
                    index(d, r).map_or(0.0, |i| self.values[i][a])
                };
                terms[k] = v
                    * (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                    * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] });
            }
            (terms[0] + terms[1]) + (terms[2] + terms[3])
        })
    }
    pub fn trace(&self, g: &Grid, p: [f32; 2], dt: f32) -> [f32; 2] {
        let mut q = p;
        for _ in 0..self.trace_steps {
            q = self.trace_single(g, q, dt / self.trace_steps as f32);
        }
        q
    }
    fn trace_single(&self, g: &Grid, p: [f32; 2], dt: f32) -> [f32; 2] {
        let v = self.sample(p);
        let mid = g.clamp([
            p[0] - 0.5 * dt * v[0] / g.h[0],
            p[1] - 0.5 * dt * v[1] / g.h[1],
        ]);
        let v = self.sample(mid);
        let end = g.clamp([p[0] - dt * v[0] / g.h[0], p[1] - dt * v[1] / g.h[1]]);
        let steps = (2.0 * (end[0] - p[0]).abs().max((end[1] - p[1]).abs()))
            .ceil()
            .max(1.0) as usize;
        let mut previous = p;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let q = [p[0] + t * (end[0] - p[0]), p[1] + t * (end[1] - p[1])];
            if g.open(g.clamp_cell(q.map(|v| v.floor() as i32))) <= 1e-5 {
                return previous;
            }
            previous = q;
        }
        end
    }

    /// Experimental H(div) reconstruction: shared linear normal face profiles
    /// plus interior bubbles cancel the tangential-slope divergence. The
    /// pointwise divergence equals the cell's MAC flux divergence everywhere.
    pub fn enable_conforming(&mut self, g: &Grid) {
        let mut slopes = vec![[0.0; 2]; self.values.len()];
        for i in 0..slopes.len() {
            let p = g.point(i);
            for a in 0..2 {
                if g.pressure_face(p, a) <= 1e-5 {
                    continue;
                }
                let mut lo = p;
                lo[1 - a] -= 1;
                let mut hi = p;
                hi[1 - a] += 1;
                slopes[i][a] = 0.5 * (self.fine_face(hi, a) - self.fine_face(lo, a));
            }
        }
        self.conforming_slopes = Some(slopes);
    }
    fn conforming_sample(&self, p: [f32; 2]) -> [f32; 2] {
        let cell: [i32; 2] =
            std::array::from_fn(|a| (p[a].floor() as i32).clamp(0, self.dims[a] as i32 - 1));
        let x = (p[0] - cell[0] as f32).clamp(0.0, 1.0);
        let y = (p[1] - cell[1] as f32).clamp(0.0, 1.0);
        let left = [cell[0] - 1, cell[1]];
        let bottom = [cell[0], cell[1] - 1];
        let slopes = self.conforming_slopes.as_ref().unwrap();
        let slope = |q, a| index(self.dims, q).map_or(0.0, |i| slopes[i][a]);
        let sl = slope(left, 0);
        let sr = slope(cell, 0);
        let tb = slope(bottom, 1);
        let tt = slope(cell, 1);
        [
            (1.0 - x) * self.fine_face(left, 0)
                + x * self.fine_face(cell, 0)
                + ((1.0 - x) * sl + x * sr) * (y - 0.5)
                + 0.5 * self.h[0] / self.h[1] * (tt - tb) * x * (1.0 - x),
            (1.0 - y) * self.fine_face(bottom, 1)
                + y * self.fine_face(cell, 1)
                + ((1.0 - y) * tb + y * tt) * (x - 0.5)
                + 0.5 * self.h[1] / self.h[0] * (sr - sl) * y * (1.0 - y),
        ]
    }
}
