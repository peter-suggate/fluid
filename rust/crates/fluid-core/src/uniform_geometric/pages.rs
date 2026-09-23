//! The phi solve window, reduced to two axes: webgpu-uniform-reference.wgsl.ts
//! scanExternalActiveSources, reduceExternalActiveRegionSummaries and
//! finalizeActiveRegion, plus the host's phiCensusDenseSteps schedule
//! (webgpu-uniform-reference.ts writeParams). The window bounds only the two
//! vertex phi passes and uvReleasedWalls; every other stage stays whole-domain.
use super::{grid::Grid, inflow::Plug, options::UniformGeometricOptions, transport::Sources};

/// UNIFORM_ACTIVE_HEADER_WORDS: the published header is compared word for word.
pub const HEADER_WORDS: usize = 256;
const TRAVEL_PLUS_WORD: usize = 237;
const TRAVEL_MINUS_WORD: usize = 238;
const TRAVEL_TOTAL_WORD: usize = 239;
/// VERTEX_PHI_REACH: vertices this far past the cell window belong to it.
const VERTEX_REACH: u32 = 6;

/// A half-open cell box [min, max).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub min: [u32; 2],
    pub max: [u32; 2],
}
impl Region {
    pub fn whole(dims: [usize; 2]) -> Self {
        Self {
            min: [0; 2],
            max: dims.map(|n| n as u32),
        }
    }
    pub fn contains(&self, cell: [i32; 2]) -> bool {
        (0..2).all(|a| cell[a] >= self.min[a] as i32 && cell[a] < self.max[a] as i32)
    }
    /// The inclusive vertex range the windowed vertex passes run on.
    pub fn vertices(&self, dims: [usize; 2]) -> [[usize; 2]; 2] {
        [
            std::array::from_fn(|a| (self.min[a] - self.min[a].min(VERTEX_REACH)) as usize),
            std::array::from_fn(|a| (self.max[a] + VERTEX_REACH).min(dims[a] as u32) as usize),
        ]
    }
}

fn pack(value: [u32; 3]) -> u32 {
    let c = value.map(|v| v.min(1023));
    (c[0] | (c[1] << 10)) | (c[2] << 20)
}

/// Persistent window state: the last published header and the dense-census
/// countdown. Construction encodes the initial writeParams, which consumes one
/// of the two dense census steps the rescan flag requests.
#[derive(Clone, Debug)]
pub struct Window {
    /// The phi region buffer exists (a multi-page lattice with a dust floor).
    pub enabled: bool,
    current: Region,
    union: Region,
    dense_steps: u32,
    /// The header published by the most recent census, as the GPU stores it.
    pub header: Vec<u32>,
}

impl Window {
    pub fn new(dims: [usize; 2], o: &UniformGeometricOptions) -> Self {
        let whole = Region::whole(dims);
        Self {
            enabled: dims.iter().any(|&n| n > 32) && o.volume_dust_threshold > 0.0,
            current: whole,
            union: whole,
            dense_steps: 1,
            header: header(whole, whole, 0),
        }
    }

    /// Runs this step's census and returns the window the vertex passes use.
    /// `publish` replaces the published header (a matched GPU oracle step)
    /// after the census has computed and recorded Rust's own.
    #[allow(clippy::too_many_arguments)]
    pub fn census(
        &mut self,
        g: &Grid,
        o: &UniformGeometricOptions,
        dt: f32,
        gravity: f32,
        shell_reach: i32,
        fine_reach: Option<i32>,
        sources: &Sources,
        plug: Option<&Plug>,
        publish: Option<&[u32]>,
    ) -> Region {
        let strength = plug.is_some_and(|p| p.strength() > 0.0);
        if strength || !g.drops.is_empty() {
            self.dense_steps = 2;
        }
        let windowed = self.enabled && o.volume_dust_threshold > 0.0 && self.dense_steps == 0;
        self.dense_steps = self.dense_steps.saturating_sub(1);
        if !self.enabled {
            return Region::whole(g.dims);
        }
        let d = g.dims.map(|n| n as u32);
        let dust = if o.volume_dust_threshold <= 0.0 {
            1e-6
        } else {
            o.volume_dust_threshold
        };
        let band = 4.0 * g.h[0].max(g.h[1]);
        let nxv = g.dims[0] + 1;
        let previous = self.union;
        let mut observed_min = d;
        let mut observed_max = [0_u32; 2];
        let mut speed = 0.0_f32;
        let mut plus = [0_u32; 3];
        let mut minus = [0_u32; 3];
        // Window induction: with no external seed, only the previous union
        // grown by eight cells can seed.
        let induced = windowed && plug.is_none() && sources.drop.is_empty();
        let span = |a: usize| {
            if induced {
                previous.min[a].saturating_sub(8) as usize
                    ..(previous.max[a] as usize + 8).min(g.dims[a])
            } else {
                0..g.dims[a]
            }
        };
        for (x, y) in span(1).flat_map(|y| span(0).map(move |x| (x, y))) {
            let (p, i) = ([x as i32, y as i32], x + g.dims[0] * y);
            let source = plug.is_some_and(|s| s.window_seed(p))
                || sources.drop.get(i).is_some_and(|&v| v > 0.0);
            let in_window = !windowed
                || (0..2).all(|a| {
                    p[a] >= previous.min[a] as i32 - 8 && p[a] < previous.max[a] as i32 + 8
                });
            // geometricActiveSeed
            let seed = in_window
                && (g.volume[i].abs() > dust
                    || (0..4).any(|k| {
                        g.phi[p[0] as usize + (k & 1) + nxv * (p[1] as usize + (k >> 1))] < band
                    }));
            if !(source || seed) {
                continue;
            }
            for a in 0..2 {
                observed_min[a] = observed_min[a].min(p[a] as u32);
                observed_max[a] = observed_max[a].max(p[a] as u32 + 1);
            }
            let v = g.velocity[i];
            speed = speed.max((v[0] * v[0] + v[1] * v[1]).sqrt());
            for a in 0..2 {
                plus[a] = plus[a].max(((v[a].max(0.0) * dt) / g.h[a]).ceil() as u32);
                minus[a] = minus[a].max((((-v[a]).max(0.0) * dt) / g.h[a]).ceil() as u32);
            }
        }
        let plus = pack(plus);
        let minus = pack(minus);
        // finalizeActiveRegion.
        let inflow = plug.map_or([0.0; 2], |s| s.velocity);
        let limit = speed.max((inflow[0] * inflow[0] + inflow[1] * inflow[1]).sqrt());
        let travel: [u32; 2] = std::array::from_fn(|a| ((limit * dt) / g.h[a]).ceil() as u32);
        let gravity_cells = ((gravity.abs() * dt) * dt) / g.h[1];
        let mut acceleration_plus = [0.0_f32; 2];
        let mut acceleration_minus = [0.0_f32; 2];
        if gravity > 0.0 {
            acceleration_plus[1] = gravity_cells;
        }
        if gravity < 0.0 {
            acceleration_minus[1] = gravity_cells;
        }
        for a in 0..2 {
            let cells = (inflow[a].abs() * dt) / g.h[a];
            if inflow[a] > 0.0 {
                acceleration_plus[a] += cells;
            }
            if inflow[a] < 0.0 {
                acceleration_minus[a] += cells;
            }
        }
        let unpack = |w: u32, a: usize| (w >> (10 * a)) & 1023;
        let travel_plus: [u32; 2] =
            std::array::from_fn(|a| unpack(plus, a) + acceleration_plus[a].ceil() as u32);
        let travel_minus: [u32; 2] =
            std::array::from_fn(|a| unpack(minus, a) + acceleration_minus[a].ceil() as u32);
        let fine_tiles = fine_reach.unwrap_or(0).max(0) as u32;
        let shell_tiles = fine_tiles + shell_reach.max(1) as u32;
        let standing = (4 * shell_tiles).max(8);
        let stencil = 6.max(o.sharpening_distance.ceil() as u32 + 1);
        let mut current = self.current;
        let observed = (0..2).all(|a| observed_max[a] > observed_min[a]);
        for a in 0..2 {
            let redirect = travel[a] + travel[a].div_ceil(2);
            let low = standing.max(travel_minus[a].max(redirect) + stencil);
            let high = standing.max(travel_plus[a].max(redirect) + stencil);
            if observed {
                current.min[a] = observed_min[a] - observed_min[a].min(low);
                current.max[a] = d[a].min(observed_max[a] + high);
            }
            current.min[a] -= current.min[a] % 4;
            current.max[a] = d[a].min(current.max[a] + (4 - current.max[a] % 4) % 4);
            if current.min[a] <= low {
                current.min[a] = 0;
            }
            if current.max[a] + high >= d[a] {
                current.max[a] = d[a];
            }
        }
        let union = Region {
            min: std::array::from_fn(|a| self.current.min[a].min(current.min[a])),
            max: std::array::from_fn(|a| self.current.max[a].max(current.max[a])),
        };
        let mut words = header(current, union, speed.to_bits());
        words[TRAVEL_PLUS_WORD] = pack([travel_plus[0], travel_plus[1], 0]);
        words[TRAVEL_MINUS_WORD] = pack([travel_minus[0], travel_minus[1], 0]);
        words[TRAVEL_TOTAL_WORD] = pack([
            travel_plus[0] + travel_minus[0],
            travel_plus[1] + travel_minus[1],
            0,
        ]);
        self.header = words;
        (self.current, self.union) = (current, union);
        if let Some(w) = publish {
            self.current = Region {
                min: [w[0], w[1]],
                max: [w[3], w[4]],
            };
            self.union = Region {
                min: [w[7], w[8]],
                max: [w[10], w[11]],
            };
        }
        self.union
    }
}

fn header(current: Region, union: Region, speed_bits: u32) -> Vec<u32> {
    let mut w = vec![0_u32; HEADER_WORDS];
    w[..6].copy_from_slice(&[
        current.min[0],
        current.min[1],
        0,
        current.max[0],
        current.max[1],
        1,
    ]);
    w[6] = speed_bits;
    w[7..13].copy_from_slice(&[union.min[0], union.min[1], 0, union.max[0], union.max[1], 1]);
    w[13] = (union.max[0] - union.min[0]).div_ceil(4);
    w[14] = (union.max[1] - union.min[1]).div_ceil(4);
    w[15] = 1;
    w
}
