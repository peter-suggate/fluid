//! Vertex phi transport (uvAdvectPhi), closest-point redistance
//! (uvRedistancePhi) and Sec. 3.5 volume sharpening, uniform-volume.wgsl.ts.
use super::{
    extension::Extension, grid::Grid, inflow::Plug, options::UniformGeometricOptions, pages::Region,
};

struct Advect<'a> {
    g: &'a Grid,
    e: &'a Extension,
    dt: f32,
    options: &'a UniformGeometricOptions,
    plug: Option<&'a Plug>,
    window: Region,
    solid_free: bool,
    external: bool,
    /// Per wall [axis][upper]: some face on it can sweep air in this step.
    walls: [[bool; 2]; 2],
}

impl Advect<'_> {
    fn trace(&self, p: [f32; 2]) -> [f32; 2] {
        self.e.trace(self.g, p, self.dt)
    }

    /// uvPhiFarAir: outside SHELL, away from every domain plane, on a step
    /// with no external source, the four corrections are identity.
    fn far_air(&self, vertex: [i32; 2]) -> bool {
        self.e.two_level.enabled
            && !self.external
            && (0..2).all(|a| vertex[a] >= 4 && vertex[a] <= self.g.dims[a] as i32 - 4)
            && !self.e.shell_at(vertex)
    }

    /// uvClosedWallPhi.
    fn closed_wall(&self, p: [f32; 2], advected: f32) -> f32 {
        let g = self.g;
        if self.dt <= 0.0 {
            return advected;
        }
        let mut interior = p;
        let mut contact = false;
        for a in 0..2 {
            for upper in [false, true] {
                let inward = if upper { -1.0 } else { 1.0 };
                let plane = if upper { g.dims[a] as f32 } else { 0.0 };
                let ambient = a == 1 && upper && g.open_top;
                if (p[a] - plane).abs() > 1e-5 || ambient {
                    continue;
                }
                let mut probe = p;
                probe[a] += inward;
                if advected >= 0.0 && inward * self.e.sample(probe)[a] >= -1e-6 {
                    continue;
                }
                interior[a] += inward;
                contact = true;
            }
        }
        if !contact || g.open(g.clamp_cell(interior.map(|v| v.floor() as i32))) <= 1e-5 {
            return advected;
        }
        let continued = g.phi_at(self.trace(interior));
        if continued < 0.0 {
            advected.min(continued)
        } else {
            advected
        }
    }

    /// uvEmbeddedContact.
    fn embedded_contact(&self, p: [f32; 2], vertex: [i32; 2], advected: f32) -> f32 {
        let g = self.g;
        if self.solid_free {
            return advected;
        }
        let mut arriving = 1e20_f32;
        let mut continued = 1e20_f32;
        let mut air = -1e20_f32;
        for k in 0..4 {
            let fluid = [vertex[0] - 1 + (k & 1), vertex[1] - 1 + (k >> 1)];
            if g.open(fluid) <= 1e-5 {
                continue;
            }
            for a in 0..2 {
                let side = if fluid[a] < vertex[a] { 1 } else { -1 };
                let mut solid = fluid;
                solid[a] += side;
                if g.index(solid).is_none() || g.open(solid) > 1e-5 {
                    continue;
                }
                let mut interior = p;
                interior[a] -= side as f32;
                let into = side as f32 * self.e.sample(interior)[a] > 1e-6;
                if advected < 0.0 || into {
                    let value = g.phi_at(self.trace(interior));
                    continued = continued.min(value);
                    if into {
                        arriving = arriving.min(value);
                    }
                }
                let face = if side > 0 { fluid } else { solid };
                let travel = self.dt * (-(side as f32) * g.relative_face(face, a));
                if g.is_released(face, a) && travel > 1e-4 * g.h[a] {
                    air = air.max(travel);
                }
            }
        }
        let result = if arriving < 1e20 {
            arriving
        } else if continued < 1e20 {
            continued
        } else {
            advected
        };
        result.max(air)
    }

    /// uvEmbeddedAir: the first embedded solid the characteristic from `p`
    /// to its RK2 endpoint `end` crosses.
    fn embedded_air(&self, p: [f32; 2], end: [f32; 2], advected: f32) -> f32 {
        let (g, dt) = (self.g, self.dt);
        if self.solid_free {
            return advected;
        }
        let steps = ((2.0 * (end[0] - p[0]).abs().max((end[1] - p[1]).abs())).ceil() as u32).max(1);
        let mut previous = p;
        let mut result = advected;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let q = [p[0] + (end[0] - p[0]) * t, p[1] + (end[1] - p[1]) * t];
            let solid = q.map(|v| v.floor() as i32);
            if g.index(solid).is_some() && g.open(solid) <= 1e-5 {
                for a in 0..2 {
                    for side in [-1, 1] {
                        let mut fluid = solid;
                        fluid[a] += side;
                        if g.open(fluid) <= 1e-5 {
                            continue;
                        }
                        let inward = side as f32;
                        let plane = solid[a] as f32 + if side > 0 { 1.0 } else { 0.0 };
                        let distance = inward * (p[a] - plane);
                        if distance < -1e-5 || inward * (end[a] - p[a]) >= 0.0 {
                            continue;
                        }
                        let before = inward * (previous[a] - plane);
                        let after = inward * (q[a] - plane);
                        if before < -1e-5 || after > 1e-5 {
                            continue;
                        }
                        let face = if side > 0 { solid } else { fluid };
                        let away = inward * g.relative_face(face, a);
                        if g.is_released(face, a) && dt * away > 1e-4 * g.h[a] {
                            result = result.max(dt * away - distance * g.h[a]);
                        }
                    }
                }
                return result;
            }
            previous = q;
        }
        result
    }

    /// uvReleasedWalls: ambient air swept in by separating wall velocities. The
    /// z perturbations of the 3D probe collapse onto the one cell layer.
    fn released_walls(&self, p: [f32; 2], advected: f32) -> f32 {
        let g = self.g;
        let mut result = advected;
        for a in 0..2 {
            for upper in [false, true] {
                if !self.walls[a][upper as usize] {
                    continue;
                }
                let inward = if upper { -1.0 } else { 1.0 };
                let ambient = a == 1 && upper && g.open_top;
                let plane = if upper { g.dims[a] as f32 } else { 0.0 };
                for side in [-1e-4, 1e-4] {
                    let mut probe = p;
                    probe[1 - a] += side;
                    probe[a] = plane + inward * 1e-4;
                    let cell = g.clamp_cell(probe.map(|v| v.floor() as i32));
                    if !self.window.contains(cell) {
                        continue;
                    }
                    let mut face = cell;
                    if !upper {
                        face[a] -= 1;
                    }
                    let away = inward * g.face(face, a);
                    if !ambient && !g.is_released(face, a) {
                        continue;
                    }
                    if self.dt * away > 1e-4 * g.h[a] {
                        result = result.max(self.dt * away - (inward * (p[a] - plane)) * g.h[a]);
                    }
                }
            }
        }
        result
    }

    /// uvSourcePhi.
    fn source(&self, p: [f32; 2], phi: f32) -> f32 {
        let phi = self.g.source_phi(p, phi);
        self.plug.map_or(phi, |s| s.phi(p, phi))
    }

    fn drain(&self, q: [f32; 2], phi: f32) -> f32 {
        let h = self.g.h[0].min(self.g.h[1]);
        if phi >= 0.5 * h {
            return phi;
        }
        let centre = q.map(|v| (v + 0.5).floor() as i32);
        for y in centre[1] - 2..centre[1] + 2 {
            for x in centre[0] - 2..centre[0] + 2 {
                if self
                    .g
                    .index([x, y])
                    .is_some_and(|i| self.g.volume[i] > 0.05)
                {
                    return phi;
                }
            }
        }
        (phi + 0.5 * h).min(0.5 * h)
    }

    fn seed_cells(&self, q: [f32; 2], phi: f32) -> f32 {
        let g = self.g;
        let h = g.h[0].min(g.h[1]);
        let base = q.map(|v| (v - 0.5).floor() as i32);
        let mut seed = phi;
        for dy in 0..2 {
            for dx in 0..2 {
                if let Some(i) = g.index([base[0] + dx, base[1] + dy]) {
                    if g.capacity[i] >= 0.99999 {
                        seed = seed.min(h * (0.5 - g.volume[i] / g.capacity[i]));
                    }
                }
            }
        }
        if seed >= phi || seed >= 0.0 {
            return phi;
        }
        let centre = q.map(|v| (v + 0.5).floor() as i32);
        for y in centre[1] - 2..centre[1] + 2 {
            for x in centre[0] - 2..centre[0] + 2 {
                let c = [x, y];
                if g.index(c).is_some() && g.phi_at([x as f32 + 0.5, y as f32 + 0.5]) < 0.0 {
                    return phi;
                }
            }
        }
        seed
    }

    fn isolated_shift(&self, q: [f32; 2]) -> f32 {
        let g = self.g;
        let base = q.map(|v| (v + 0.5).floor() as i32);
        let empty = self.options.volume_dust_threshold.max(1e-5);
        for dy in -8..8 {
            for dx in -8..8 {
                if dx != -8 && dx != 7 && dy != -8 && dy != 7 {
                    continue;
                }
                let c = [base[0] + dx, base[1] + dy];
                if let Some(i) = g.index(c) {
                    if g.volume[i] > empty || g.target(c) > 0.0 {
                        return 0.0;
                    }
                }
            }
        }
        let mut residual = 0.0;
        let mut area = 0.0_f32;
        for dy in -7..7 {
            for dx in -7..7 {
                let c = [base[0] + dx, base[1] + dy];
                if let Some(i) = g.index(c) {
                    let gamma = g.target(c);
                    residual += g.volume[i] - gamma;
                    if gamma > 0.0 && gamma < g.capacity[i] {
                        area += 1.0;
                    }
                }
            }
        }
        if area < 1.0 {
            return 0.0;
        }
        g.h[0].min(g.h[1]) * (residual / area).clamp(-0.25, 0.25)
    }

    fn vertex(&self, x: usize, y: usize) -> f32 {
        let g = self.g;
        let vertex = [x as i32, y as i32];
        let p = [x as f32, y as f32];
        let end = self.e.rk2(g, p, self.dt);
        let departure = self.e.walk(g, p, end);
        let mut advected = g.phi_at(departure);
        let splash = self.options.phi_cubic_advection == "on"
            || self.options.phi_drain == "on"
            || self.options.phi_seed_cells == "on"
            || self.options.isolated_body_volume == "on";
        if self.far_air(vertex) && !(splash && self.e.shell_at(departure.map(|v| v.floor() as i32)))
        {
            return advected;
        }
        if !self.solid_free && g.buried(vertex) {
            return g.phi[x + (g.dims[0] + 1) * y];
        }
        let h = g.h[0].min(g.h[1]);
        if self.options.phi_cubic_advection == "on" && advected.abs() < 2.0 * h {
            advected = g.phi_cubic(departure);
        }
        let contact = self.embedded_contact(p, vertex, self.closed_wall(p, advected));
        let mut released = self.released_walls(p, self.embedded_air(p, end, contact));
        if self.options.phi_drain == "on" {
            released = self.drain(departure, released);
        }
        let mut value = self.source(p, released);
        if self.options.isolated_body_volume == "on" && value.abs() < 2.0 * h {
            value -= self.isolated_shift(departure);
        }
        if self.options.phi_seed_cells == "on" {
            value = self.seed_cells(departure, value);
        }
        value
    }
}

/// uvAdvectPhi over the window's vertex range into `scratch`. Vertices outside
/// it keep whatever the scratch last held, as the GPU ping-pong target does.
pub fn advect_window(
    g: &Grid,
    e: &Extension,
    dt: f32,
    window: Region,
    plug: Option<&Plug>,
    options: &UniformGeometricOptions,
    scratch: &mut [f32],
) {
    let pass = Advect {
        g,
        e,
        dt,
        options,
        plug,
        window,
        solid_free: g.solid_free(),
        external: !g.drops.is_empty() || plug.is_some_and(Plug::active),
        // Every face uvReleasedWalls can probe on each wall (its cell clamps
        // onto the wall layer), with the probe's own admission test.
        walls: std::array::from_fn(|a| {
            std::array::from_fn(|upper| {
                let inward = if upper == 1 { -1.0 } else { 1.0 };
                let ambient = a == 1 && upper == 1 && g.open_top;
                (0..g.dims[1 - a] as i32).any(|j| {
                    let mut face = [j; 2];
                    face[a] = if upper == 1 { g.dims[a] as i32 - 1 } else { -1 };
                    let away = inward * g.face(face, a);
                    (ambient || g.is_released(face, a)) && dt * away > 1e-4 * g.h[a]
                })
            })
        }),
    };
    let [lo, hi] = window.vertices(g.dims);
    let nxv = g.dims[0] + 1;
    for y in lo[1]..=hi[1] {
        for x in lo[0]..=hi[0] {
            scratch[x + nxv * y] = pass.vertex(x, y);
        }
    }
}

/// uvRedistancePhi over the window's vertex range, reading the advected
/// scratch and writing `g.phi`; vertices outside the range keep their phi.
pub fn redistance_window(g: &mut Grid, scratch: &[f32], window: Region) {
    let h = g.h;
    let minh = h[0].min(h[1]);
    let band = 4.0 * h[0].max(h[1]);
    let limit = g.dims.map(|n| n as f32);
    let [lo, hi] = window.vertices(g.dims);
    let nxv = g.dims[0] + 1;
    for y in lo[1]..=hi[1] {
        for x in lo[0]..=hi[0] {
            let i = x + nxv * y;
            let initial = scratch[i];
            let mut value = initial;
            if initial.abs() > 1e-8 && initial.abs() < band && !g.buried([x as i32, y as i32]) {
                let p = [x as f32, y as f32];
                let mut q = p;
                for _ in 0..8 {
                    let gradient = g.gradient(scratch, q);
                    let s = [gradient[0] / h[0], gradient[1] / h[1]];
                    let norm = s[0] * s[0] + s[1] * s[1];
                    if norm < 1e-16 {
                        break;
                    }
                    let phi = g.scalar(scratch, q);
                    for a in 0..2 {
                        q[a] = (q[a]
                            - ((phi * gradient[a]) / ((h[a] * h[a]) * norm)).clamp(-2.0, 2.0))
                        .clamp((p[a] - 4.0).max(0.0), (p[a] + 4.0).min(limit[a]));
                    }
                }
                if g.scalar(scratch, q).abs() < 0.005 * minh {
                    let d = [(p[0] - q[0]) * h[0], (p[1] - q[1]) * h[1]];
                    value = initial.signum() * (d[0] * d[0] + d[1] * d[1]).sqrt();
                }
            }
            g.phi[i] = value;
        }
    }
}

/// Whole-domain advect for diagnostics: no inflow plug, result in `g.phi`.
pub fn advect(g: &mut Grid, e: &Extension, dt: f32) {
    let mut scratch = g.phi.clone();
    advect_window(
        g,
        e,
        dt,
        Region::whole(g.dims),
        None,
        &UniformGeometricOptions::default(),
        &mut scratch,
    );
    g.phi = scratch;
}

/// Whole-domain redistance of `g.phi`.
pub fn redistance(g: &mut Grid) {
    let scratch = g.phi.clone();
    redistance_window(g, &scratch, Region::whole(g.dims));
}

/// Diagnostic iteration control over the current surface targets.
pub fn sharpen_rounds(g: &mut Grid, o: &UniformGeometricOptions, rounds: usize) -> f64 {
    let gamma: Vec<f32> = (0..g.volume.len()).map(|i| g.target(g.point(i))).collect();
    sharpen(g, o, &gamma, rounds)
}

/// Sec. 3.5 sharpening: uvClassifySharpenTiles, uvCacheSharpenCells/Faces,
/// then `rounds` of uvPrepare/Propose/Limit/CommitSharpen over the active
/// tiles. Phi and the target `gamma` are fixed throughout. Returns the volume
/// the dust floor removed.
pub fn sharpen(g: &mut Grid, o: &UniformGeometricOptions, gamma: &[f32], rounds: usize) -> f64 {
    if o.density_sharpening == "off" {
        return 0.0;
    }
    let n = g.volume.len();
    let [nx, ny] = g.dims;
    let nxv = nx + 1;
    let minh = g.h[0].min(g.h[1]);
    let band = o.sharpening_distance * minh;
    let dose = o.sharpening_strength.clamp(0.0, 1.0);
    // phi_at at each cell centre: every bilinear weight is exactly one half.
    let mut phi = Vec::with_capacity(n);
    for y in 0..ny {
        let (lo, hi) = (&g.phi[nxv * y..], &g.phi[nxv * (y + 1)..]);
        phi.extend((0..nx).map(|x| {
            let t = [lo[x], lo[x + 1], hi[x], hi[x + 1]].map(|v| (v * 0.5) * 0.5);
            (t[0] + t[1]) + (t[2] + t[3])
        }));
    }
    let cd = [nx.div_ceil(4), ny.div_ceil(4)];
    let mut tiles = vec![o.sharpening_work_map != "on"; cd[0] * cd[1]];
    if o.sharpening_work_map == "on" {
        for y in 0..ny {
            for x in 0..nx {
                // Negated: non-finite phi keeps its tile active.
                let far = phi[x + nx * y].abs() >= band;
                if !far {
                    tiles[(x >> 2) + cd[0] * (y >> 2)] = true;
                }
            }
        }
    }
    let active = |x: usize, y: usize| tiles[(x >> 2) + cd[0] * (y >> 2)];
    // Active row spans [y, x0, x1) in ascending raster order.
    let mut spans: Vec<[usize; 3]> = Vec::new();
    for y in 0..ny {
        for tx in (0..cd[0]).filter(|&tx| tiles[tx + cd[0] * (y >> 2)]) {
            let (x0, x1) = (4 * tx, (4 * tx + 4).min(nx));
            match spans.last_mut() {
                Some(last) if last[0] == y && last[2] == x0 => last[2] = x1,
                _ => spans.push([y, x0, x1]),
            }
        }
    }
    // Face flags: bit 0 the face exchanges, bit 1 a->b admitted, bit 2 b->a.
    let mut faces = vec![[0_u8; 2]; n];
    for &[y, x0, x1] in &spans {
        for x in x0..x1 {
            let i = x + nx * y;
            for a in 0..2 {
                let q = if a == 0 { [x + 1, y] } else { [x, y + 1] };
                if q[0] >= nx || q[1] >= ny {
                    continue;
                }
                let j = q[0] + nx * q[1];
                let face_open = g.rigid_faces.as_ref().map_or(1.0, |f| f[i][a]);
                if !active(q[0], q[1])
                    || g.capacity[i] < 0.99999
                    || g.capacity[j] < 0.99999
                    || face_open < 0.99999
                {
                    continue;
                }
                let mut centre = [x as f32 + 0.5, y as f32 + 0.5];
                centre[a] += 0.5;
                let middle = g.phi_at(centre);
                let epsilon = 1e-6;
                let (pa, pb) = (phi[i], phi[j]);
                let inward_a = pa >= 0.0
                    && pb < pa - epsilon
                    && middle <= pa + epsilon
                    && middle >= pb - epsilon;
                let inward_b = pb >= 0.0
                    && pa < pb - epsilon
                    && middle <= pb + epsilon
                    && middle >= pa - epsilon;
                let relay_a = pa > 0.0 && gamma[i] <= 1e-6;
                let relay_b = pb > 0.0 && gamma[j] <= 1e-6;
                faces[i][a] =
                    1 | if (middle <= epsilon && !relay_b) || inward_a {
                        2
                    } else {
                        0
                    } | if (middle <= epsilon && !relay_a) || inward_b {
                        4
                    } else {
                        0
                    };
            }
        }
    }
    let mut dust = 0.0;
    let dust_floor = |value: f32, dust: &mut f64| {
        if value != 0.0 && value.abs() < o.volume_dust_threshold {
            *dust += value as f64;
            0.0
        } else {
            value
        }
    };
    let mut surplus = vec![0.0_f32; n];
    let mut need = vec![0.0_f32; n];
    let mut flux = vec![[0.0_f32; 2]; n];
    let mut limits = vec![[0.0_f32; 2]; n];
    for round in 0..rounds {
        for &[y, x0, x1] in &spans {
            for i in x0 + nx * y..x1 + nx * y {
                let admitted = g.capacity[i] > 0.99999 && phi[i].abs() < band;
                let relay = phi[i] > 0.0 && gamma[i] <= 1e-6;
                let own = g.volume[i];
                surplus[i] = if admitted {
                    dose * (own - gamma[i]).max(0.0)
                } else {
                    0.0
                };
                need[i] = if admitted {
                    dose * ((if relay { 1.0 } else { gamma[i] }) - own).max(0.0)
                } else {
                    0.0
                };
            }
        }
        for &[y, x0, x1] in &spans {
            for i in x0 + nx * y..x1 + nx * y {
                for a in 0..2 {
                    flux[i][a] = 0.0;
                    let flags = faces[i][a];
                    if flags & 1 == 0 {
                        continue;
                    }
                    let j = i + if a == 0 { 1 } else { nx };
                    let ab = if flags & 2 != 0 {
                        surplus[i].min(need[j])
                    } else {
                        0.0
                    };
                    let ba = if flags & 4 != 0 {
                        surplus[j].min(need[i])
                    } else {
                        0.0
                    };
                    flux[i][a] = ab - ba;
                }
            }
        }
        for &[y, x0, x1] in &spans {
            for x in x0..x1 {
                let i = x + nx * y;
                let mut outgoing = 0.0_f32;
                let mut incoming = 0.0_f32;
                for a in 0..2 {
                    let positive = flux[i][a];
                    let negative = match a {
                        0 if x > 0 && active(x - 1, y) => flux[i - 1][0],
                        1 if y > 0 && active(x, y - 1) => flux[i - nx][1],
                        _ => 0.0,
                    };
                    outgoing += positive.max(0.0) + (-negative).max(0.0);
                    incoming += (-positive).max(0.0) + negative.max(0.0);
                }
                limits[i] = [
                    (surplus[i] / outgoing.max(1e-20)).min(1.0),
                    (need[i] / incoming.max(1e-20)).min(1.0),
                ];
            }
        }
        // Inactive cells meet only the dust floor, which is idempotent: after
        // the first round, in raster order with the commits, it is a no-op.
        let mut next = 0;
        for &[y, x0, x1] in &spans {
            if round == 0 {
                for v in &mut g.volume[next..x0 + nx * y] {
                    *v = dust_floor(*v, &mut dust);
                }
                next = x1 + nx * y;
            }
            for x in x0..x1 {
                let i = x + nx * y;
                let limited = |i: usize, j: usize, a: usize| {
                    let raw = flux[i][a];
                    raw * if raw >= 0.0 {
                        limits[i][0].min(limits[j][1])
                    } else {
                        limits[i][1].min(limits[j][0])
                    }
                };
                let mut terms = [0.0_f32; 4];
                if x + 1 < nx {
                    terms[0] = -if active(x + 1, y) {
                        limited(i, i + 1, 0)
                    } else {
                        0.0
                    };
                }
                if x > 0 {
                    terms[1] = if active(x - 1, y) {
                        limited(i - 1, i, 0)
                    } else {
                        0.0
                    };
                }
                if y + 1 < ny {
                    terms[2] = -if active(x, y + 1) {
                        limited(i, i + nx, 1)
                    } else {
                        0.0
                    };
                }
                if y > 0 {
                    terms[3] = if active(x, y - 1) {
                        limited(i - nx, i, 1)
                    } else {
                        0.0
                    };
                }
                // d4Sum6 with the two z terms zero.
                let sum = ((terms[0] + terms[1]) + 0.0) + (terms[2] + terms[3]);
                g.volume[i] = dust_floor(g.volume[i] + sum, &mut dust);
            }
        }
        if round == 0 {
            for v in &mut g.volume[next..] {
                *v = dust_floor(*v, &mut dust);
            }
        }
    }
    dust
}
