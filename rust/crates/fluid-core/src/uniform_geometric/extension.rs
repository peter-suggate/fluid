//! Sec. 3.3 velocity extension and the two-level 4h tile classes, reduced to
//! two axes. Tile classes mirror uniform-volume.wgsl.ts uvTwoLevelSeedCooperative,
//! uvTwoLevelDilate{X,Y,Z} and the E7 uvTransportReach passes; the extension
//! mirrors webgpu-uniform-velocity-extrapolation.wgsl.ts (JRW07 FIM, the CM11b
//! source-aware hierarchy, the shell pack and publishCoarseVelocityTable).
use super::{grid::Grid, options::UniformGeometricOptions, transport::Sources};

const INF: f32 = 65504.0;
pub const FINE: u8 = 1;
pub const SHELL: u8 = 2;
pub const TRANSPORT: u8 = 4;
const TILE_FAR: i32 = 63;

/// Host gates and reaches (webgpu-uniform-reference.ts writeParams).
#[derive(Clone, Copy, Debug)]
pub struct TwoLevel {
    /// physical.z >= 0: the sampler reads the 4h table outside FINE.
    pub enabled: bool,
    pub fine_reach: i32,
    pub shell_reach: i32,
    pub tiled_extension: bool,
    pub advection_tiles: bool,
    pub transport_tiles: bool,
    /// twoLevel.w: transport reach biased by eight, or -1 with E3 off.
    pub transport_w: i32,
}
impl TwoLevel {
    pub fn new(dims: [usize; 2], o: &UniformGeometricOptions) -> Self {
        let tiles = dims[0].div_ceil(4) * dims[1].div_ceil(4);
        let enabled = o.two_level_velocity == "on"
            && dims.iter().all(|n| n % 4 == 0)
            && 6 * tiles + 3 <= dims[0] * dims[1];
        let round = |v: f32, lo: f32, hi: f32| v.clamp(lo, hi).round() as i32;
        let transport_tiles =
            enabled && o.transport_work_map != "dense" && o.volume_dust_threshold > 0.0;
        Self {
            enabled,
            fine_reach: round(o.two_level_fine_reach, 0.0, 8.0),
            shell_reach: round(o.two_level_shell_reach, 0.0, 8.0),
            tiled_extension: enabled && o.two_level_extension != "dense",
            advection_tiles: enabled && o.two_level_advection != "dense",
            transport_tiles,
            transport_w: if transport_tiles {
                round(o.transport_reach, -8.0, 8.0) + 8
            } else {
                -1
            },
        }
    }
    fn k(&self) -> i32 {
        if self.enabled {
            self.fine_reach
        } else {
            0
        }
    }
    fn s(&self) -> i32 {
        self.k() + self.shell_reach.max(1)
    }
    /// uvTileRequiredReach / uvTwoLevelTransportReach.
    fn required_reach(&self, displacement: f32) -> i32 {
        if self.transport_w < 0 {
            return 0;
        }
        (m0(displacement) + self.transport_w - 8).clamp(0, 16)
    }
}
/// E3's predicate reach, ceil((ceil(D)+1)/4) tiles.
fn m0(d: f32) -> i32 {
    ((d.max(0.0).ceil() + 1.0) / 4.0).ceil() as i32
}
fn pack(cls: i32, dist: i32) -> i32 {
    (cls & 15) | (dist.clamp(0, 63) << 4)
}
fn tile_dist(value: i32) -> i32 {
    (value >> 4) & 63
}

#[derive(Clone)]
pub struct Extension {
    dims: [usize; 2],
    cd: [usize; 2],
    h: [f32; 2],
    pub two_level: TwoLevel,
    /// The packed transport field: this step's values inside SHELL, the
    /// persistent (stale) field elsewhere.
    pub values: Vec<[f32; 2]>,
    /// Tile table class word: FINE, SHELL and TRANSPORT bits.
    pub classes: Vec<u8>,
    coarse: Vec<[f32; 2]>,
    /// lean.x: no cut cell anywhere, so the trace walk cannot stop early.
    pub solid_free: bool,
    /// E3's domain maximum displacement D, in cells.
    pub displacement: f32,
}

struct Tiles {
    classes: Vec<u8>,
    distance: Vec<u8>,
    displacement: f32,
}

/// The largest |v| per axis. Max skips NaN and every |v| is +0 or more, so
/// the lane order cannot change the bits.
fn max_speed(v: &[[f32; 2]]) -> [f32; 2] {
    let mut lanes = [0.0_f32; 16];
    let mut chunks = v.chunks_exact(8);
    for chunk in &mut chunks {
        for (lane, v) in lanes.iter_mut().zip(chunk.as_flattened()) {
            *lane = lane.max(v.abs());
        }
    }
    let mut m = [0.0_f32; 2];
    for (k, &lane) in lanes.iter().enumerate() {
        m[k & 1] = m[k & 1].max(lane);
    }
    for v in chunks.remainder() {
        m = [m[0].max(v[0].abs()), m[1].max(v[1].abs())];
    }
    m
}

fn seed_and_dilate(
    g: &Grid,
    o: &UniformGeometricOptions,
    t: &TwoLevel,
    dt: f32,
    sources: &Sources,
) -> Tiles {
    let d = g.dims;
    let cd = d.map(|n| n.div_ceil(4));
    let dust = if o.volume_dust_threshold <= 0.0 {
        1e-6
    } else {
        o.volume_dust_threshold
    };
    let band = 4.0 * g.h[0].max(g.h[1]);
    let nxv = d[0] + 1;
    let mut flags = vec![0_i32; cd[0] * cd[1]];
    let with_sources = sources.any();
    for y in 0..d[1] {
        let row = &mut flags[cd[0] * (y / 4)..][..cd[0]];
        let volume = g.volume[d[0] * y..][..d[0]].chunks(4);
        let capacity = g.capacity[d[0] * y..][..d[0]].chunks(4);
        for (tx, (volume, capacity)) in volume.zip(capacity).enumerate() {
            let mut f = 0;
            for (k, (&v, &c)) in volume.iter().zip(capacity).enumerate() {
                if v.abs() >= dust || (with_sources && sources.at(d[0] * y + 4 * tx + k)) {
                    f |= 7;
                }
                if c < 0.99999 {
                    f |= 3;
                }
            }
            row[tx] |= f;
        }
    }
    let speed = max_speed(&g.velocity);
    let displacement = ((speed[0] * dt) / g.h[0]).max(((speed[1] * dt) / g.h[1]).max(0.0));
    // A vertex row y lies on tile rows (y-1)/4 ..= y/4; a tile spans vertices 4t..=4t+4.
    let mut near = vec![false; cd[0]];
    for y in 0..=d[1] {
        let row = &g.phi[nxv * y..][..nxv];
        for (tx, near) in near.iter_mut().enumerate() {
            *near = row[4 * tx..=(4 * tx + 4).min(d[0])]
                .iter()
                .any(|&v| v < band);
        }
        let rows = (y.saturating_sub(1) / 4)..=(y / 4).min(cd[1] - 1);
        for ty in rows {
            for (tx, &near) in near.iter().enumerate() {
                if near {
                    flags[tx + cd[0] * ty] |= 7;
                }
            }
        }
    }
    let seed: Vec<i32> = flags
        .iter()
        .map(|&f| pack(f, if f & 4 != 0 { 0 } else { TILE_FAR }))
        .collect();
    let k = t.k();
    let s = t.s();
    let required = m0(displacement);
    let m = t.required_reach(displacement);
    let reach_g = if t.transport_w < 0 {
        0
    } else if required > 16 {
        1 << 20
    } else {
        m + required + 1
    };
    let r = s.max(m).max(reach_g).min(cd[0].max(cd[1]) as i32);
    let scan = |source: &[i32], axis: usize, first: bool| -> Vec<i32> {
        let mut out = vec![0; source.len()];
        let len = cd[axis];
        let (step, line_step) = if axis == 0 { (1, cd[0]) } else { (cd[0], 1) };
        let masks = [1, 2, 4, if first { 4 } else { 8 }];
        let reach = [k, s, m, reach_g].map(|v| v.min(r));
        let ring = r.min(TILE_FAR) as usize;
        let mut nearest = vec![[i32::MAX; 4]; len];
        for line in 0..cd[1 - axis] {
            let at = |x: usize| source[line * line_step + x * step];
            let mut last = [None; 4];
            for (x, nearest) in nearest.iter_mut().enumerate() {
                for b in 0..4 {
                    if at(x) & masks[b] != 0 {
                        last[b] = Some(x);
                    }
                    nearest[b] = last[b].map_or(i32::MAX, |q| (x - q) as i32);
                }
            }
            let mut next = [None; 4];
            for (x, nearest) in nearest.iter_mut().enumerate().rev() {
                for b in 0..4 {
                    if at(x) & masks[b] != 0 {
                        next[b] = Some(x);
                    }
                    if let Some(q) = next[b] {
                        nearest[b] = nearest[b].min((q - x) as i32);
                    }
                }
            }
            for (x, nearest) in nearest.iter().enumerate() {
                let hit = (0..4).fold(0, |hit, b| hit | (((nearest[b] <= reach[b]) as i32) << b));
                let mut dist = TILE_FAR;
                let mut a = 0;
                while a <= ring && (a as i32) < dist {
                    for q in [x.checked_sub(a), Some(x + a).filter(|&q| q < len)]
                        .into_iter()
                        .flatten()
                    {
                        dist = dist.min(tile_dist(at(q)).max(a as i32));
                    }
                    a += 1;
                }
                out[line * line_step + x * step] = pack(hit, dist);
            }
        }
        out
    };
    // The z scan of a one-layer lattice is the identity.
    let scanned = scan(&scan(&seed, 0, true), 1, false);
    Tiles {
        classes: scanned.iter().map(|&w| (w & 7) as u8).collect(),
        distance: scanned.iter().map(|&w| tile_dist(w) as u8).collect(),
        displacement,
    }
}

#[derive(Clone)]
struct Level {
    dims: [usize; 2],
    value: Vec<[f32; 2]>,
    known: Vec<u8>,
    /// Per component, the base face location of the value's origin.
    origin: Vec<[[f32; 2]; 2]>,
}
impl Level {
    fn new(dims: [usize; 2]) -> Self {
        let n = dims[0] * dims[1];
        Self {
            dims,
            value: vec![[0.0; 2]; n],
            known: vec![0; n],
            origin: vec![[[0.0; 2]; 2]; n],
        }
    }
}

/// One (source, target) level pair of nearestHierarchySample, tabulated per
/// axis b and per `b == c`: the target face location in base cells and the
/// lower source corner of the 2x2 sample.
struct Stencil {
    sd: [usize; 2],
    location: [[Vec<f32>; 2]; 2],
    lower: [[Vec<i32>; 2]; 2],
}
impl Stencil {
    /// Per axis and `b == c`, the targets [first, last] whose restriction (the
    /// nearest sample, then the footprint) can read each source coordinate.
    fn readers(&self) -> [[Vec<[usize; 2]>; 2]; 2] {
        std::array::from_fn(|b| {
            std::array::from_fn(|same| {
                let top = self.sd[b] as i32 - 1;
                let mut readers = vec![[usize::MAX, 0]; self.sd[b]];
                for (p, &lower) in self.lower[b][same].iter().enumerate() {
                    let first = lower.min(2 * p as i32).clamp(0, top);
                    let last = (lower.max(2 * p as i32) + 1).clamp(0, top);
                    for x in first..=last {
                        let r = &mut readers[x as usize];
                        r[0] = r[0].min(p);
                        r[1] = r[1].max(p);
                    }
                }
                readers
            })
        })
    }
    /// The source cells nearestHierarchySample reads for target `p`.
    fn reads(&self, p: [usize; 2], c: usize) -> [usize; 4] {
        std::array::from_fn(|k| {
            let q: [usize; 2] = std::array::from_fn(|b| {
                (self.lower[b][(b == c) as usize][p[b]] + ((k >> b) & 1) as i32)
                    .clamp(0, self.sd[b] as i32 - 1) as usize
            });
            q[0] + self.sd[0] * q[1]
        })
    }
}

struct Hierarchy {
    base: [usize; 2],
    h: [f32; 2],
    epsilon: f32,
    /// Base face locations, per axis b and `b == c`.
    origin: [[Vec<f32>; 2]; 2],
}
impl Hierarchy {
    fn new(base: [usize; 2], h: [f32; 2], epsilon: f32) -> Self {
        let mut hierarchy = Self {
            base,
            h,
            epsilon,
            origin: Default::default(),
        };
        hierarchy.origin = hierarchy.stencil(base, base).location;
        hierarchy
    }
    fn stencil(&self, sd: [usize; 2], td: [usize; 2]) -> Stencil {
        let axis = |b: usize, f: &dyn Fn(usize, bool) -> f32| -> [Vec<f32>; 2] {
            [false, true].map(|same| (0..td[b]).map(|p| f(p, same)).collect())
        };
        let location = std::array::from_fn(|b| {
            axis(b, &|p, same| {
                if same {
                    ((p + 1) as f32 * self.base[b] as f32) / td[b] as f32
                } else {
                    ((p as f32 + 0.5) * self.base[b] as f32) / td[b] as f32
                }
            })
        });
        let lower = std::array::from_fn(|b| {
            axis(b, &|p, same| {
                if same {
                    ((p + 1) as f32 * sd[b] as f32) / td[b] as f32 - 1.0
                } else {
                    ((p as f32 + 0.5) * sd[b] as f32) / td[b] as f32 - 0.5
                }
            })
            .map(|v| v.into_iter().map(|x| x.floor() as i32).collect())
        });
        Stencil {
            sd,
            location,
            lower,
        }
    }
    /// restrictKnownVelocity: the nearest samples, else the 2x2 footprint.
    fn restrict(
        &self,
        p: [usize; 2],
        st: &Stencil,
        c: usize,
        source: &impl Fn([usize; 2], usize) -> Option<(f32, [f32; 2])>,
    ) -> (f32, f32, [f32; 2]) {
        let r = self.nearest(p, st, c, false, source);
        if r.1 > 0.0 {
            r
        } else {
            self.nearest(p, st, c, true, source)
        }
    }
    /// nearestHierarchySample. Each 2D corner appears twice in the 3D
    /// eight-corner stencil (z clamps to the single layer), which leaves
    /// the d4Sum8Vec2 value unchanged.
    fn nearest(
        &self,
        p: [usize; 2],
        st: &Stencil,
        c: usize,
        footprint: bool,
        source: &impl Fn([usize; 2], usize) -> Option<(f32, [f32; 2])>,
    ) -> (f32, f32, [f32; 2]) {
        let sd = st.sd;
        let location: [f32; 2] = std::array::from_fn(|b| st.location[b][(b == c) as usize][p[b]]);
        let lower: [i32; 2] = std::array::from_fn(|b| {
            if footprint {
                2 * p[b] as i32
            } else {
                st.lower[b][(b == c) as usize][p[b]]
            }
        });
        let mut best = 1e30_f32;
        let mut distances = [1e30_f32; 4];
        let mut found = [false; 4];
        let mut origins = [[0.0_f32; 2]; 4];
        let mut values = [0.0_f32; 4];
        for k in 0..4 {
            let q = [
                (lower[0] + (k & 1) as i32).clamp(0, sd[0] as i32 - 1) as usize,
                (lower[1] + (k >> 1) as i32).clamp(0, sd[1] as i32 - 1) as usize,
            ];
            let Some((value, at)) = source(q, q[0] + sd[0] * q[1]) else {
                continue;
            };
            let delta = [
                (at[0] - location[0]) * self.h[0],
                (at[1] - location[1]) * self.h[1],
            ];
            // dot(delta,delta) as Metal lowers it, fma(dy,dy,dx*dx). Geometric
            // ties beyond a few cells sit within one rounding of each other,
            // so the unfused sum breaks them the other way.
            let distance = delta[1].mul_add(delta[1], delta[0] * delta[0]);
            distances[k] = distance;
            found[k] = true;
            origins[k] = at;
            values[k] = value;
            best = best.min(distance);
        }
        let mut sum = [[0.0_f32; 2]; 4];
        let mut origin = None;
        for k in 0..4 {
            if found[k] && (distances[k] - best).abs() <= self.epsilon {
                sum[k] = [values[k], 1.0];
                if origin.is_none() {
                    origin = Some(origins[k]);
                }
            }
        }
        let value = (sum[0][0] + sum[1][0]) + (sum[2][0] + sum[3][0]);
        let weight = (sum[0][1] + sum[1][1]) + (sum[2][1] + sum[3][1]);
        (
            if weight > 0.0 { value / weight } else { 0.0 },
            weight,
            origin.unwrap_or_default(),
        )
    }
}

impl Extension {
    /// Fresh-state convenience: the persistent transport field starts at zero.
    pub fn build(g: &Grid, o: &UniformGeometricOptions, dt: f32) -> Self {
        Self::build_with(
            g,
            o,
            dt,
            &Sources::default(),
            vec![[0.0; 2]; g.volume.len()],
            None,
        )
    }
    /// MacCormack prediction: the same work map, rebuilt on the predicted field.
    pub fn build_prediction(
        g: &Grid,
        o: &UniformGeometricOptions,
        dt: f32,
        original: &Self,
    ) -> Self {
        Self::build_with(
            g,
            o,
            dt,
            &Sources::default(),
            original.values.clone(),
            Some(original),
        )
    }
    pub fn with_coarse_from(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.coarse.clone_from(&other.coarse);
        result
    }
    pub fn build_with(
        g: &Grid,
        o: &UniformGeometricOptions,
        dt: f32,
        sources: &Sources,
        mut packed: Vec<[f32; 2]>,
        workmap: Option<&Self>,
    ) -> Self {
        let dims = g.dims;
        let cd = dims.map(|n| n.div_ceil(4));
        let t = TwoLevel::new(dims, o);
        let tiles = match workmap {
            Some(w) => Tiles {
                classes: w.classes.clone(),
                distance: Vec::new(),
                displacement: w.displacement,
            },
            None if t.enabled => seed_and_dilate(g, o, &t, dt, sources),
            None => Tiles {
                classes: vec![7; cd[0] * cd[1]],
                distance: Vec::new(),
                displacement: 0.0,
            },
        };
        let n = g.volume.len();
        let spread = |tile: &[bool]| -> Vec<bool> {
            let mut cells = Vec::with_capacity(n);
            for y in 0..dims[1] {
                let row = &tile[cd[0] * (y / 4)..][..cd[0]];
                cells.extend((0..dims[0]).map(|x| row[x / 4]));
            }
            cells
        };
        let shell = spread(
            &tiles
                .classes
                .iter()
                .map(|&c| !t.tiled_extension || c & SHELL != 0)
                .collect::<Vec<_>>(),
        );
        let minh = g.h[0].min(g.h[1]);
        // Face openness and liquid density are read at SHELL cells and their
        // axis neighbours; evaluate them on SHELL tiles grown by one tile.
        let near: Vec<bool> = if t.tiled_extension {
            let mut grown = vec![false; cd[0] * cd[1]];
            for ty in 0..cd[1] {
                for tx in 0..cd[0] {
                    if tiles.classes[tx + cd[0] * ty] & SHELL != 0 {
                        for y in ty.saturating_sub(1)..(ty + 2).min(cd[1]) {
                            for x in tx.saturating_sub(1)..(tx + 2).min(cd[0]) {
                                grown[x + cd[0] * y] = true;
                            }
                        }
                    }
                }
            }
            spread(&grown)
        } else {
            vec![true; n]
        };
        let mut open = vec![0_u8; n];
        let mut liquid = vec![false; n];
        for i in 0..n {
            if !near[i] {
                continue;
            }
            let p = g.point(i);
            for a in 0..2 {
                if g.pressure_face(p, a) > 1e-5 {
                    open[i] |= 1 << a;
                }
            }
            liquid[i] = 0.5 - g.pressure_phi(p, &o.volume_pressure_rows) / minh > 0.5
                || g.airborne(p, o.airborne_momentum == "on", o.volume_dust_threshold);
        }
        let index = |p: [i32; 2]| -> Option<usize> {
            (p[0] >= 0 && p[1] >= 0 && p[0] < dims[0] as i32 && p[1] < dims[1] as i32)
                .then(|| p[0] as usize + dims[0] * p[1] as usize)
        };
        let is_open = |p: [i32; 2], c: usize| index(p).is_some_and(|i| open[i] & (1 << c) != 0);
        let source_face = |p: [i32; 2], c: usize| {
            is_open(p, c) && {
                let mut q = p;
                q[c] += 1;
                liquid[index(p).unwrap()] || index(q).is_some_and(|j| liquid[j])
            }
        };
        let shell_cells: Vec<usize> = (0..n).filter(|&i| shell[i]).collect();
        let mut source = vec![0_u8; n];
        let mut value = vec![[0.0_f32; 2]; n];
        let mut distance = vec![[INF; 2]; n];
        let mut known = vec![0_u8; n];
        let mut active = vec![0_u8; n];
        let mut active_count = 0;
        // sourceFace is read at SHELL cells and their axis neighbours.
        for i in (0..n).filter(|&i| near[i]) {
            let p = g.point(i);
            for c in 0..2 {
                if source_face(p, c) {
                    source[i] |= 1 << c;
                }
            }
        }
        for &i in &shell_cells {
            let p = g.point(i);
            for c in 0..2 {
                if open[i] & (1 << c) == 0 {
                    continue;
                }
                if source[i] & (1 << c) != 0 {
                    value[i][c] = g.velocity[i][c];
                    distance[i][c] = 0.0;
                    known[i] |= 1 << c;
                } else if (0..2).any(|axis| {
                    [-1, 1].iter().any(|&side| {
                        let mut q = p;
                        q[axis] += side;
                        index(q).is_some_and(|j| source[j] & (1 << c) != 0)
                    })
                }) {
                    active[i] |= 1 << c;
                    active_count += 1;
                }
            }
        }
        let h = g.h;
        let band = 2.0 * h[0].max(h[1]);
        let diagonal = ((dims[0] as f32 * h[0]).powi(2) + (dims[1] as f32 * h[1]).powi(2)).sqrt();
        let epsilon = |d: f32| d.abs().clamp(2.0 * minh, diagonal) * 1.1920929e-7;
        let passes =
            (o.extension_front_sweeps.round().max(1.0) as usize).min(dims[0].max(dims[1]).min(16));
        // Passes write SHELL cells only, so later copies refresh just those.
        let (mut old_distance, mut old_value) = (Vec::new(), Vec::new());
        let (mut old_known, mut old_active) = (Vec::new(), Vec::new());
        for pass in 0..passes {
            if active_count == 0 {
                break;
            }
            if pass == 0 {
                old_distance.clone_from(&distance);
                old_value.clone_from(&value);
                old_known.clone_from(&known);
                old_active.clone_from(&active);
            } else {
                for &i in &shell_cells {
                    old_distance[i] = distance[i];
                    old_value[i] = value[i];
                    old_known[i] = known[i];
                    old_active[i] = active[i];
                }
            }
            let nd = |q: [i32; 2], c: usize| -> f32 {
                match index(q) {
                    Some(j) if open[j] & (1 << c) != 0 && shell[j] => old_distance[j][c],
                    _ => INF,
                }
            };
            let nv = |q: [i32; 2], c: usize| -> f32 {
                match index(q) {
                    Some(j)
                        if open[j] & (1 << c) != 0 && shell[j] && old_known[j] & (1 << c) != 0 =>
                    {
                        old_value[j][c]
                    }
                    _ => 0.0,
                }
            };
            let godunov = |p: [i32; 2], c: usize| -> f32 {
                let mut minima = [INF; 2];
                let mut spacing = h;
                for axis in 0..2 {
                    let mut lo = p;
                    let mut hi = p;
                    lo[axis] -= 1;
                    hi[axis] += 1;
                    minima[axis] = nd(lo, c).min(nd(hi, c));
                }
                if minima[1] < minima[0] {
                    minima.swap(0, 1);
                    spacing.swap(0, 1);
                }
                if minima[0] >= 0.5 * INF {
                    return INF;
                }
                let root = minima[0] + spacing[0];
                if minima[1] >= 0.5 * INF || root <= minima[1] {
                    return root;
                }
                let (mut a, mut b, mut cc) = (0.0_f32, 0.0_f32, -1.0_f32);
                for k in 0..2 {
                    let inverse = 1.0 / (spacing[k] * spacing[k]);
                    a += inverse;
                    b += minima[k] * inverse;
                    cc += (minima[k] * minima[k]) * inverse;
                }
                (b + (b * b - a * cc).max(0.0).sqrt()) / a
            };
            let upwind = |p: [i32; 2], c: usize, solved: f32| -> f32 {
                let eps = epsilon(solved);
                let mut weighted = [0.0_f32; 2];
                let mut weights = [0.0_f32; 2];
                for axis in 0..2 {
                    let mut lo = p;
                    let mut hi = p;
                    lo[axis] -= 1;
                    hi[axis] += 1;
                    let low = nd(lo, c);
                    let high = nd(hi, c);
                    let minimum = low.min(high);
                    if minimum >= solved - eps {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    if (low - minimum).abs() <= eps {
                        sum += nv(lo, c);
                        count += 1.0;
                    }
                    if (high - minimum).abs() <= eps {
                        sum += nv(hi, c);
                        count += 1.0;
                    }
                    if count <= 0.0 {
                        continue;
                    }
                    let weight = (solved - minimum) / (h[axis] * h[axis]);
                    weighted[axis] = (weight * sum) / count;
                    weights[axis] = weight;
                }
                let total = weights[0] + weights[1];
                if total > 0.0 {
                    (weighted[0] + weighted[1]) / total
                } else {
                    0.0
                }
            };
            let converged_distance = |q: [i32; 2], c: usize| -> f32 {
                let Some(j) = index(q) else {
                    return INF;
                };
                if open[j] & (1 << c) == 0 || source[j] & (1 << c) != 0 || !shell[j] {
                    return INF;
                }
                let old = old_distance[j][c];
                if old_active[j] & (1 << c) == 0 || old >= 0.5 * INF {
                    return INF;
                }
                let updated = old.min(godunov(q, c));
                if updated > band || (updated - old).abs() > epsilon(old) {
                    return INF;
                }
                updated
            };
            active_count = 0;
            for &i in &shell_cells {
                let p = g.point(i);
                for c in 0..2 {
                    let bit = 1 << c;
                    if open[i] & bit == 0 || source[i] & bit != 0 {
                        continue;
                    }
                    if old_active[i] & bit != 0 {
                        let old = old_distance[i][c];
                        let eps = epsilon(old);
                        let updated = old.min(godunov(p, c));
                        if updated <= band {
                            distance[i][c] = updated;
                            value[i][c] = upwind(p, c, updated);
                            known[i] |= bit;
                        }
                        let converged = old < 0.5 * INF && (updated - old).abs() <= eps;
                        if !converged && updated <= band {
                            active[i] |= bit;
                        } else {
                            active[i] &= !bit;
                        }
                    } else if godunov(p, c) <= band && {
                        let own = old_distance[i][c];
                        let eps = epsilon(own);
                        (0..2).any(|axis| {
                            let mut lo = p;
                            let mut hi = p;
                            lo[axis] -= 1;
                            hi[axis] += 1;
                            own > converged_distance(lo, c) + eps
                                || own > converged_distance(hi, c) + eps
                        })
                    } {
                        active[i] |= bit;
                    }
                    if active[i] & bit != 0 {
                        active_count += 1;
                    }
                }
            }
        }
        // Source-aware hierarchy: ceil(n/2) levels down to one cell. A level
        // restricts only where its stencil can read a known face; elsewhere
        // both samples are empty and the face stays unknown.
        let hierarchy = Hierarchy::new(dims, h, (1e-6 * minh) * minh);
        let mut levels: Vec<Level> = Vec::new();
        let mut current = dims;
        let mut known_cells: Vec<usize> = shell_cells
            .iter()
            .copied()
            .filter(|&i| known[i] != 0)
            .collect();
        while current[0].max(current[1]) > 1 {
            let td = current.map(|v| v.div_ceil(2));
            let sd = current;
            let st = hierarchy.stencil(sd, td);
            let readers = st.readers();
            let mut want = vec![0_u8; td[0] * td[1]];
            for &j in &known_cells {
                let q = [j % sd[0], j / sd[0]];
                let bits = levels.last().map_or(known[j], |l| l.known[j]);
                for c in (0..2).filter(|&c| bits & (1 << c) != 0) {
                    let [rx, ry] = [0, 1].map(|b| readers[b][(b == c) as usize][q[b]]);
                    for y in ry[0]..=ry[1].min(td[1] - 1) {
                        for x in rx[0]..=rx[1].min(td[0] - 1) {
                            want[x + td[0] * y] |= 1 << c;
                        }
                    }
                }
            }
            let mut level = Level::new(td);
            known_cells.clear();
            for (i, &want) in want.iter().enumerate().filter(|(_, &w)| w != 0) {
                let p = [i % td[0], i / td[0]];
                for c in (0..2).filter(|&c| want & (1 << c) != 0) {
                    let result = match levels.last() {
                        None => hierarchy.restrict(p, &st, c, &|q: [usize; 2], j: usize| {
                            (known[j] & (1 << c) != 0 && shell[j]).then(|| {
                                let at = std::array::from_fn(|b| {
                                    hierarchy.origin[b][(b == c) as usize][q[b]]
                                });
                                (value[j][c], at)
                            })
                        }),
                        Some(previous) => hierarchy.restrict(p, &st, c, &|_, j: usize| {
                            (previous.known[j] & (1 << c) != 0)
                                .then(|| (previous.value[j][c], previous.origin[j][c]))
                        }),
                    };
                    if result.1 > 0.0 {
                        level.value[i][c] = result.0;
                        level.origin[i][c] = result.2;
                        level.known[i] |= 1 << c;
                    }
                }
                if level.known[i] != 0 {
                    known_cells.push(i);
                }
            }
            levels.push(level);
            current = td;
        }
        // Prolong unknown faces from the coarsest level back to level 0, in
        // place. Only the SHELL pack reads level 0: prolong the faces it reads.
        let pack = levels.first().map(|l| hierarchy.stencil(l.dims, dims));
        let mut need = Vec::new();
        if let (Some(st), true) = (&pack, levels.len() > 1) {
            need = vec![0_u8; levels[0].dims[0] * levels[0].dims[1]];
            for &i in &shell_cells {
                let p = [i % dims[0], i / dims[0]];
                for c in (0..2).filter(|&c| open[i] & !known[i] & (1 << c) != 0) {
                    for q in st.reads(p, c) {
                        need[q] |= 1 << c;
                    }
                }
            }
        }
        for li in (0..levels.len().saturating_sub(1)).rev() {
            let (finer, coarser) = levels.split_at_mut(li + 1);
            let (up, coarser) = (&mut finer[li], &coarser[0]);
            let td = up.dims;
            let st = hierarchy.stencil(coarser.dims, td);
            for y in 0..td[1] {
                for x in 0..td[0] {
                    let i = x + td[0] * y;
                    for c in 0..2 {
                        if up.known[i] & (1 << c) != 0 || (li == 0 && need[i] & (1 << c) == 0) {
                            continue;
                        }
                        let r = hierarchy.nearest([x, y], &st, c, false, &|_, j: usize| {
                            (coarser.known[j] & (1 << c) != 0)
                                .then(|| (coarser.value[j][c], coarser.origin[j][c]))
                        });
                        if r.1 > 0.0 {
                            up.value[i][c] = r.0;
                            up.origin[i][c] = r.2;
                            up.known[i] |= 1 << c;
                        }
                    }
                }
            }
        }
        // Fused prolongation and pack on SHELL faces only; a closed face packs zero.
        if let (Some(coarser), Some(st)) = (levels.first(), &pack) {
            for &i in &shell_cells {
                let p = [i % dims[0], i / dims[0]];
                for c in 0..2 {
                    packed[i][c] = if open[i] & (1 << c) == 0 {
                        0.0
                    } else if known[i] & (1 << c) != 0 {
                        value[i][c]
                    } else {
                        let r = hierarchy.nearest(p, st, c, false, &|_, j: usize| {
                            (coarser.known[j] & (1 << c) != 0)
                                .then(|| (coarser.value[j][c], coarser.origin[j][c]))
                        });
                        if r.1 > 0.0 {
                            r.0
                        } else {
                            0.0
                        }
                    };
                }
            }
        }
        // publishCoarseVelocityTable from the ceil(n/4) level.
        let mut coarse = vec![[0.0_f32; 2]; cd[0] * cd[1]];
        if t.enabled && levels.len() > 1 {
            let level = &levels[1];
            debug_assert_eq!(level.dims, cd);
            for ty in 0..cd[1] {
                for tx in 0..cd[0] {
                    let i = tx + cd[0] * ty;
                    for c in 0..2 {
                        let mut v = if level.known[i] & (1 << c) != 0 {
                            level.value[i][c]
                        } else {
                            0.0
                        };
                        let tt = [tx, ty];
                        if tt[c] + 1 == cd[c]
                            && !(0..4).any(|a| {
                                let mut p = [4 * tx as i32, 4 * ty as i32];
                                p[1 - c] += a;
                                p[c] += 3;
                                index(p).is_some() && g.pressure_face(p, c) > 1e-5
                            })
                        {
                            v = 0.0;
                        }
                        coarse[i][c] = v;
                    }
                }
            }
        }
        let mut classes = tiles.classes;
        // E7: narrow TRANSPORT by each tile's own post-extension reach.
        if t.transport_tiles && workmap.is_none() {
            let cap = tiles.displacement.max(0.0).ceil().clamp(0.0, 63.0) as i32;
            let mut speed = vec![[0.0_f32; 2]; cd[0] * cd[1]];
            for y in 0..dims[1] {
                let row = &mut speed[cd[0] * (y / 4)..][..cd[0]];
                for (m, v) in row
                    .iter_mut()
                    .zip(packed[dims[0] * y..][..dims[0]].chunks(4))
                {
                    let v = max_speed(v);
                    *m = [m[0].max(v[0]), m[1].max(v[1])];
                }
            }
            let mut measure = vec![0_i32; cd[0] * cd[1]];
            for (i, speed) in speed.iter().enumerate() {
                let mut travel = 0_i32;
                let mut lane = |x: f32| travel = travel.max(x.ceil().clamp(0.0, 63.0) as i32);
                lane(((speed[0] * dt) / h[0]).max(((speed[1] * dt) / h[1]).max(0.0)));
                for c in 0..2 {
                    lane((coarse[i][c].abs() * dt) / h[c]);
                }
                measure[i] = travel.min(cap);
            }
            let r = (m0(tiles.displacement) + 1).min(cd[0].max(cd[1]) as i32);
            let scan = |source: &[i32], axis: usize| -> Vec<i32> {
                let mut out = vec![0; source.len()];
                for ty in 0..cd[1] as i32 {
                    for tx in 0..cd[0] as i32 {
                        let mut travel = 0;
                        for delta in -r..=r {
                            let mut q = [tx, ty];
                            q[axis] += delta;
                            if q[axis] < 0 || q[axis] >= cd[axis] as i32 {
                                continue;
                            }
                            travel = travel.max(source[q[0] as usize + cd[0] * q[1] as usize]);
                        }
                        out[tx as usize + cd[0] * ty as usize] = travel.clamp(0, 63);
                    }
                }
                out
            };
            let reach = scan(&scan(&measure, 0), 1);
            if m0(tiles.displacement) <= 16 {
                for i in 0..classes.len() {
                    if classes[i] & TRANSPORT != 0
                        && tiles.distance[i] as i32 > t.required_reach(reach[i] as f32)
                    {
                        classes[i] &= 3;
                    }
                }
            }
        }
        let solid_free = g.capacity.iter().all(|&c| c == 1.0) && g.rigid_faces.is_none();
        Self {
            dims,
            cd,
            h,
            two_level: t,
            values: packed,
            classes,
            coarse,
            solid_free,
            displacement: tiles.displacement,
        }
    }
    fn tile_of(&self, p: [f32; 2]) -> usize {
        let cell: [usize; 2] = std::array::from_fn(|a| {
            (p[a].floor() as i32).clamp(0, self.dims[a] as i32 - 1) as usize
        });
        cell[0] / 4 + self.cd[0] * (cell[1] / 4)
    }
    /// E3: the gather builds rows only on TRANSPORT tiles. The live receivers,
    /// in ascending raster order; every cell with the map off.
    pub fn transport_cells(&self, cells: &mut Vec<u32>) {
        cells.clear();
        let [nx, ny] = self.dims;
        if !self.two_level.transport_tiles {
            cells.extend(0..(nx * ny) as u32);
            return;
        }
        for y in 0..ny {
            let row = &self.classes[self.cd[0] * (y / 4)..][..self.cd[0]];
            for (tx, &class) in row.iter().enumerate() {
                if class & TRANSPORT != 0 {
                    let start = nx * y + 4 * tx;
                    cells.extend(start as u32..(start + 4.min(nx - 4 * tx)) as u32);
                }
            }
        }
    }
    /// uvTwoLevelFineAt, true everywhere with the sampler off.
    pub fn fine_at(&self, p: [f32; 2]) -> bool {
        !self.two_level.enabled || self.classes[self.tile_of(p)] & FINE != 0
    }
    pub fn shell_at(&self, p: [i32; 2]) -> bool {
        !self.two_level.enabled || self.classes[self.tile_of(p.map(|v| v as f32))] & SHELL != 0
    }
    /// The packed field's MAC face, zero outside the lattice.
    pub fn fine_face(&self, p: [i32; 2], a: usize) -> f32 {
        if p[0] < 0 || p[1] < 0 || p[0] >= self.dims[0] as i32 || p[1] >= self.dims[1] as i32 {
            return 0.0;
        }
        self.values[p[0] as usize + self.dims[0] * p[1] as usize][a]
    }
    fn coarse_face(&self, t: [i32; 2], a: usize) -> f32 {
        if t[0] < 0 || t[1] < 0 || t[0] >= self.cd[0] as i32 || t[1] >= self.cd[1] as i32 {
            return 0.0;
        }
        self.coarse[t[0] as usize + self.cd[0] * t[1] as usize][a]
    }
    /// sampleVelocityComponent: ((wx*wy)*wz)*v terms summed as d4Sum8.
    pub fn sample(&self, p: [f32; 2]) -> [f32; 2] {
        let coarse = self.two_level.enabled && !self.fine_at(p);
        std::array::from_fn(|a| {
            let (d, scale) = if coarse {
                (self.cd, 0.25)
            } else {
                (self.dims, 1.0)
            };
            let q: [f32; 2] = std::array::from_fn(|b| {
                let (lower, offset) = if b == a { (-1.0, 1.0) } else { (0.0, 0.5) };
                let v = if coarse {
                    scale * p[b] - offset
                } else {
                    p[b] - offset
                };
                v.clamp(lower, d[b] as f32 - 1.0)
            });
            let base = q.map(|v| v.floor() as i32);
            let f = [q[0] - q[0].floor(), q[1] - q[1].floor()];
            let mut terms = [0.0_f32; 4];
            for (k, term) in terms.iter_mut().enumerate() {
                let r = [base[0] + (k & 1) as i32, base[1] + (k >> 1) as i32];
                let w = (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                    * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] });
                let v = if coarse {
                    self.coarse_face(r, a)
                } else {
                    self.fine_face(r, a)
                };
                *term = w * v;
            }
            (terms[0] + terms[1]) + (terms[2] + terms[3])
        })
    }
    /// The clamped RK2 endpoint uvTrace and uvEmbeddedAir share.
    pub fn rk2(&self, g: &Grid, p: [f32; 2], dt: f32) -> [f32; 2] {
        let v = self.sample(p);
        let mid = g.clamp(std::array::from_fn(|a| {
            p[a] - ((0.5 * dt) * v[a]) / self.h[a]
        }));
        let v = self.sample(mid);
        g.clamp(std::array::from_fn(|a| p[a] - (dt * v[a]) / self.h[a]))
    }
    /// uvTrace: RK2 then the half-cell solid walk (E4 skips it when no cell is cut).
    pub fn trace(&self, g: &Grid, p: [f32; 2], dt: f32) -> [f32; 2] {
        self.walk(g, p, self.rk2(g, p, dt))
    }
    /// uvTrace's walk from `p` toward its RK2 endpoint `end`.
    pub fn walk(&self, g: &Grid, p: [f32; 2], end: [f32; 2]) -> [f32; 2] {
        if self.solid_free {
            return end;
        }
        let steps = ((2.0 * (end[0] - p[0]).abs().max((end[1] - p[1]).abs().max(0.0))).ceil()
            as u32)
            .max(1);
        let mut previous = p;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let q = [p[0] + (end[0] - p[0]) * t, p[1] + (end[1] - p[1]) * t];
            if g.open(g.clamp_cell(q.map(|v| v.floor() as i32))) <= 1e-5 {
                return previous;
            }
            previous = q;
        }
        end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remote_pool_does_not_dilute_a_falling_drops_air_extension() {
        for pool in [false, true] {
            let mut g = Grid::new([64, 64], [0.05; 2], false).unwrap();
            for y in 0..=64 {
                for x in 0..=64 {
                    let ball = ((x as f32 - 32.0).hypot(y as f32 - 45.0) - 6.0) * 0.05;
                    g.phi[x + 65 * y] = if pool {
                        ball.min((y as f32 - 20.0) * 0.05)
                    } else {
                        ball
                    };
                }
            }
            for i in 0..g.volume.len() {
                let p = g.point(i);
                g.volume[i] = g.target(p);
                if p[1] > 25
                    && (g.pressure_phi(p, "off") < 0.0
                        || g.pressure_phi([p[0], p[1] + 1], "off") < 0.0)
                {
                    g.velocity[i][1] = -4.0;
                }
            }
            let mut o = UniformGeometricOptions::default();
            o.extension_front_sweeps = 2.0;
            let e = Extension::build(&g, &o, 1.0 / 30.0);
            // Previously the pool changed the second air layer from -4 to -3,
            // and the next layer to -2, despite being far from these samples.
            for y in 34..40 {
                assert!(
                    (e.fine_face([32, y], 1) + 4.0).abs() < 1e-5,
                    "pool={pool}, y={y}"
                );
            }
            if pool {
                assert!(e.fine_face([32, 20], 1).abs() < 1e-5);
            }
            for i in 0..g.velocity.len() {
                let p = g.point(i);
                if !e.shell_at(p) {
                    continue;
                }
                for a in 0..2 {
                    let mut q = p;
                    q[a] += 1;
                    if g.pressure_face(p, a) > 1e-5
                        && (g.pressure_phi(p, "off") < 0.0 || g.pressure_phi(q, "off") < 0.0)
                    {
                        assert!(
                            (e.values[i][a] - g.velocity[i][a]).abs() < 1e-5,
                            "liquid face changed"
                        );
                    }
                }
            }
        }
    }
}
