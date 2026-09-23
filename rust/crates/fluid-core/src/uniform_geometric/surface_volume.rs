//! Total surface volume (uniform-surface-volume.wgsl.ts with the compact work
//! box and the lean measure; webgpu-uniform-surface-volume.ts encode). Two
//! 17-sample tetrahedral volume curves refine one bounded global normal shift of
//! phi so its fill matches the gathered V. The 2D reference is one cell layer
//! with z-duplicated vertices, so every eight-corner stencil repeats k & 3.
use super::grid::Grid;

type Record = [f32; 20];

/// sumGroup: the 64-lane workgroup tree, stride 32 down to 1.
fn tree(lanes: &mut [Record; 64]) -> Record {
    let mut stride = 32;
    while stride > 0 {
        let (low, high) = lanes.split_at_mut(stride);
        for (a, b) in low.iter_mut().zip(high.iter()) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        stride /= 2;
    }
    lanes[0]
}

fn tetra(v: [f32; 4]) -> f32 {
    let mut n = [0.0_f32; 4];
    let mut o = [0.0_f32; 4];
    let (mut count, mut outside) = (0, 0);
    for x in v {
        if x < 0.0 {
            n[count] = x;
            count += 1;
        } else {
            o[outside] = x;
            outside += 1;
        }
    }
    match count {
        0 => 0.0,
        4 => 1.0,
        1 => ((-n[0] / (o[0] - n[0])) * (-n[0] / (o[1] - n[0]))) * (-n[0] / (o[2] - n[0])),
        3 => 1.0 - ((o[0] / (o[0] - n[0])) * (o[0] / (o[0] - n[1]))) * (o[0] / (o[0] - n[2])),
        _ => {
            let a = -n[0] / (o[0] - n[0]);
            let b = -n[0] / (o[1] - n[0]);
            let c = -n[1] / (o[0] - n[1]);
            let d = -n[1] / (o[1] - n[1]);
            (((a * b) + ((b * c) * (1.0 - a))) + ((c * d) * (1.0 - b))).clamp(0.0, 1.0)
        }
    }
}

fn fill(v: [f32; 8]) -> f32 {
    // Six whole (or six empty) tetrahedra sum to exactly 6/6 (or 0/6).
    if v.iter().all(|&x| x < 0.0) {
        return 1.0;
    }
    if !v.iter().any(|&x| x < 0.0) {
        return 0.0;
    }
    (((((tetra([v[0], v[1], v[3], v[7]]) + tetra([v[0], v[1], v[5], v[7]]))
        + tetra([v[0], v[2], v[3], v[7]]))
        + tetra([v[0], v[2], v[6], v[7]]))
        + tetra([v[0], v[4], v[5], v[7]]))
        + tetra([v[0], v[4], v[6], v[7]]))
        / 6.0
}

/// Shifts `g.phi` in place and returns the state vector
/// [shift, search half-range, target V, surface V before correction].
pub fn correct(g: &mut Grid) -> [f32; 4] {
    let dims = g.dims;
    let nxv = dims[0] + 1;
    let n = dims[0] * dims[1];
    let minh = g.h[0].min(g.h[1]);
    let vertex = |q: [usize; 2], k: usize| (q[0] + (k & 1)) + nxv * (q[1] + ((k >> 1) & 1));
    // seed + finishWork.
    let mut band = vec![0_u32; n];
    let mut minimum = dims;
    let mut maximum = [0_usize; 2];
    for (y, row) in band.chunks_mut(dims[0]).enumerate() {
        for (x, b) in row.iter_mut().enumerate() {
            let (q, i) = ([x, y], x + dims[0] * y);
            let mut live = g.volume[i] != 0.0;
            if g.capacity[i] > 0.0 {
                let mut lo = 1e30_f32;
                let mut hi = -1e30_f32;
                for k in 0..4 {
                    let v = g.phi[vertex(q, k)];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                live = live || lo <= 0.0;
                if lo <= 0.0 && hi >= 0.0 {
                    *b = 5;
                }
            }
            if live {
                for a in 0..2 {
                    minimum[a] = minimum[a].min(q[a]);
                    maximum[a] = maximum[a].max(q[a] + 1);
                }
            }
        }
    }
    let (lo, d) = if maximum[0] > 0 {
        let lo: [usize; 2] = std::array::from_fn(|a| minimum[a].saturating_sub(5));
        (
            lo,
            std::array::from_fn(|a| dims[a].min(maximum[a] + 5) - lo[a]),
        )
    } else {
        ([0; 2], [0; 2])
    };
    let cells = d[0] * d[1];
    // Four dilations over the box; both parity buffers are zero outside it.
    let mut next = vec![0_u32; n];
    for _ in 0..4 {
        for y in lo[1]..lo[1] + d[1] {
            for x in lo[0]..lo[0] + d[0] {
                let i = x + dims[0] * y;
                let mut b = 0;
                if g.capacity[i] > 0.0 {
                    b = band[i];
                    let q = [x as i32, y as i32];
                    for a in 0..2 {
                        for s in [-1, 1] {
                            let mut m = q;
                            m[a] += s;
                            if let Some(j) = g.index(m) {
                                b = b.max(band[j].saturating_sub(1));
                            }
                        }
                    }
                }
                next[i] = b;
            }
        }
        std::mem::swap(&mut band, &mut next);
    }
    // metric: per-vertex shift scale, zero outside the box.
    let mut scale = vec![0.0_f32; g.phi.len()];
    let rows = if cells == 0 {
        0..0
    } else {
        lo[1]..lo[1] + d[1] + 1
    };
    for y in rows {
        for x in lo[0]..=lo[0] + d[0] {
            let q = [x as i32, y as i32];
            let mut b = 0;
            let mut open = [false; 4];
            for k in 0..4 {
                let c = [q[0] - 1 + (k & 1), q[1] - 1 + (k >> 1)];
                if let Some(j) = g.index(c) {
                    b = b.max(band[j]);
                    if g.capacity[j] > 0.0 {
                        for a in 0..2 {
                            open[2 * a + ((k as usize >> a) & 1)] = true;
                        }
                    }
                }
            }
            let mut gradient = [0.0_f32; 2];
            for a in 0..2 {
                let mut low = [x, y];
                let mut high = [x, y];
                low[a] = low[a].saturating_sub(1);
                high[a] = dims[a].min(high[a] + 1);
                if open[2 * a] && open[2 * a + 1] {
                    gradient[a] = (g.phi[high[0] + nxv * high[1]] - g.phi[low[0] + nxv * low[1]])
                        / ((high[a] - low[a]) as f32 * g.h[a]);
                }
            }
            let length = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt();
            scale[x + nxv * y] = (b as f32 * 0.2) * 0.1_f32.max(length);
        }
    }
    let mut state = [0.0, minh, 0.0, 0.0_f32];
    let groups = cells.div_ceil(64);
    for iteration in 0..2 {
        // measure: one record per box cell in logical order, 64 to a group.
        let mut partial = vec![[0.0_f32; 20]; groups];
        let mut lanes = [[0.0_f32; 20]; 64];
        for (group, out) in partial.iter_mut().enumerate() {
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = [0.0; 20];
                let logical = group * 64 + l;
                if logical >= cells {
                    continue;
                }
                let q = [lo[0] + logical % d[0], lo[1] + logical / d[0]];
                let i = q[0] + dims[0] * q[1];
                let cap = g.capacity[i];
                lane[17] = g.volume[i];
                if cap <= 0.0 {
                    continue;
                }
                let corner: [usize; 8] = std::array::from_fn(|k| vertex(q, k & 3));
                let bits = corner.map(|v| scale[v].to_bits());
                if bits.iter().fold(0, |a, b| a | b) == 0 {
                    let filled = if g.phi[corner[0]] < 0.0 { cap } else { 0.0 };
                    lane[..17].fill(filled);
                    continue;
                }
                let (centre, width) = (state[0], state[1]);
                let raw = corner.map(|v| g.phi[v]);
                let factor = corner.map(|v| scale[v]);
                let mut low = 1e30_f32;
                let mut high = -1e30_f32;
                for k in 0..8 {
                    low = low.min(raw[k] - (centre + width) * factor[k]);
                    high = high.max(raw[k] - (centre - width) * factor[k]);
                }
                for (sample, value) in lane[..17].iter_mut().enumerate() {
                    let fraction = if high < 0.0 {
                        1.0
                    } else if low < 0.0 {
                        let shift = centre + ((sample as f32 / 8.0) - 1.0) * width;
                        fill(std::array::from_fn(|k| raw[k] - shift * factor[k]))
                    } else {
                        0.0
                    };
                    *value = fraction * cap;
                }
            }
            *out = tree(&mut lanes);
        }
        // reduce: 64 partials to a group.
        let reduced: Vec<Record> = (0..cells.div_ceil(4096))
            .map(|group| {
                for (l, lane) in lanes.iter_mut().enumerate() {
                    let i = group * 64 + l;
                    *lane = if i < groups { partial[i] } else { [0.0; 20] };
                }
                tree(&mut lanes)
            })
            .collect();
        // solve: lane l accumulates every 64th reduced record, then the tree.
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane = [0.0; 20];
            for record in reduced.iter().skip(l).step_by(64) {
                for k in 0..20 {
                    lane[k] += record[k];
                }
            }
        }
        let sums = tree(&mut lanes);
        let desired = sums[17];
        let mut shift = state[0];
        let width = state[1];
        if iteration == 0 {
            state[3] = sums[8];
        }
        if sums[16] - sums[0] > 1e-6
            && (sums[8] - desired).abs() > 1e-5_f32.max(1e-7 * desired.abs())
        {
            shift = state[0] + if desired > sums[16] { width } else { -width };
            for k in 0..16 {
                let (a, b) = (sums[k], sums[k + 1]);
                if desired >= a && desired <= b && b > a {
                    shift = state[0]
                        + (((k as f32 + ((desired - a) / (b - a)).clamp(0.0, 1.0)) - 8.0) * width)
                            / 8.0;
                    break;
                }
            }
        }
        state[0] = shift.clamp(-minh, minh);
        state[1] = width / 8.0;
        state[2] = desired;
    }
    // apply.
    for (phi, s) in g.phi.iter_mut().zip(&scale) {
        *phi -= state[0] * s;
    }
    state
}
