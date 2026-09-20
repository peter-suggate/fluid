//! CM11b velocity transport, including bounded MacCormack and liquid-only gathers.
use super::{extension::Extension, grid::Grid, options::UniformGeometricOptions};

pub fn phase(g: &Grid, o: &UniformGeometricOptions) -> Vec<bool> {
    (0..g.volume.len())
        .map(|i| g.capacity[i] > 1e-5 && g.pressure_phi(g.point(i), &o.volume_pressure_rows) < 0.0)
        .collect()
}

fn clamp(g: &Grid, mut p: [f32; 2]) -> [f32; 2] {
    p[0] = p[0].clamp(0.0, g.dims[0] as f32);
    p[1] = p[1].max(0.0);
    if !g.open_top {
        p[1] = p[1].min(g.dims[1] as f32);
    }
    p
}

fn departure(g: &Grid, e: &Extension, start: [f32; 2], dt: f32) -> [f32; 2] {
    let mut q = start;
    let mut remaining = dt.abs();
    let direction = dt.signum();
    for _ in 0..32 {
        if remaining <= 1e-7 {
            break;
        }
        let v = e.sample(q);
        let rate = (v[0].abs() / g.h[0]).max(v[1].abs() / g.h[1]);
        let step = remaining.min(1.5 / rate.max(1e-6));
        let signed_step = direction * step;
        let mid = clamp(
            g,
            std::array::from_fn(|a| q[a] - 0.5 * v[a] * signed_step / g.h[a]),
        );
        let v = e.sample(mid);
        q = clamp(
            g,
            std::array::from_fn(|a| q[a] - v[a] * signed_step / g.h[a]),
        );
        remaining -= step;
    }
    // Velocity uses the source's endpoint test and eight bisections. The phi
    // characteristic deliberately has a different, half-cell chord walk.
    let solid = |p: [f32; 2]| {
        let cell = p.map(|v| v.floor() as i32);
        g.index(cell).is_some_and(|i| g.capacity[i] <= 1e-5)
    };
    if solid(q) {
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..8 {
            let mid = 0.5 * (lo + hi);
            if solid(std::array::from_fn(|a| start[a] + mid * (q[a] - start[a]))) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        q = std::array::from_fn(|a| start[a] + lo * (q[a] - start[a]));
    }
    q
}

fn stencil(g: &Grid, p: [f32; 2], a: usize) -> [([i32; 2], f32); 4] {
    let q: [f32; 2] = std::array::from_fn(|b| {
        (p[b] - if a == b { 1.0 } else { 0.5 })
            .clamp(if a == b { -1.0 } else { 0.0 }, g.dims[b] as f32 - 1.0)
    });
    let base = q.map(|v| v.floor() as i32);
    let f = q.map(|v| v - v.floor());
    std::array::from_fn(|k| {
        (
            [base[0] + (k & 1) as i32, base[1] + (k >> 1) as i32],
            (if k & 1 == 0 { 1.0 - f[0] } else { f[0] })
                * (if k & 2 == 0 { 1.0 - f[1] } else { f[1] }),
        )
    })
}

fn physical(g: &Grid, phase: &[bool], p: [f32; 2], a: usize) -> Option<f32> {
    let mut terms = [[0.0; 2]; 4];
    for (k, (donor, w)) in stencil(g, p, a).into_iter().enumerate() {
        let mut neighbor = donor;
        neighbor[a] += 1;
        if g.index(donor).is_some_and(|i| phase[i]) || g.index(neighbor).is_some_and(|i| phase[i]) {
            terms[k] = [w * g.face(donor, a), w];
        }
    }
    let sum: [f32; 2] =
        std::array::from_fn(|a| (terms[0][a] + terms[1][a]) + (terms[2][a] + terms[3][a]));
    (sum[1] > 0.0).then(|| sum[0] / sum[1])
}

fn component(
    g: &Grid,
    e: &Extension,
    phase: &[bool],
    o: &UniformGeometricOptions,
    start: [f32; 2],
    p: [i32; 2],
    a: usize,
    dt: f32,
) -> f32 {
    let q = departure(g, e, start, dt);
    let mut neighbor = p;
    neighbor[a] += 1;
    let wet = |p| g.index(p).is_some_and(|i| g.volume[i] > 1e-5);
    if o.liquid_only_velocity_advection != "on" || !(wet(p) || wet(neighbor)) {
        return e.sample(q)[a];
    }
    if let Some(v) = physical(g, phase, q, a) {
        return v;
    }
    for probe in 1..=16 {
        let t = probe as f32 / 16.0;
        if let Some(v) = physical(
            g,
            phase,
            std::array::from_fn(|b| q[b] + t * (start[b] - q[b])),
            a,
        ) {
            return v;
        }
    }
    0.0
}

fn pass(
    g: &Grid,
    e: &Extension,
    phase: &[bool],
    o: &UniformGeometricOptions,
    dt: f32,
    maccormack: bool,
) -> Vec<[f32; 2]> {
    (0..g.volume.len())
        .map(|i| {
            let p = g.point(i);
            if !maccormack
                && o.two_level_advection == "tiles"
                && !e.fine_at(p.map(|v| v as f32 + 0.5))
            {
                return [0.0; 2];
            }
            std::array::from_fn(|a| {
                let mut start = p.map(|v| v as f32 + 0.5);
                start[a] += 0.5;
                let mut v = component(g, e, phase, o, start, p, a, dt);
                if p[a] == g.dims[a] as i32 - 1 && !(a == 1 && g.open_top) {
                    v = if maccormack {
                        g.velocity[i][a]
                    } else {
                        v.min(g.velocity[i][a])
                    };
                }
                v
            })
        })
        .collect()
}

/// Returns the advected field. The caller applies body forces exactly once.
pub fn advect(
    g: &Grid,
    e: &Extension,
    prior_phase: &[bool],
    o: &UniformGeometricOptions,
    dt: f32,
) -> Vec<[f32; 2]> {
    if o.velocity_transport != "maccormack" {
        return pass(g, e, prior_phase, o, dt, false);
    }
    let predicted = pass(g, e, prior_phase, o, dt, true);
    let mut prediction = g.clone();
    prediction.velocity = predicted.clone();
    let predicted_extension = Extension::build_prediction(&prediction, o, dt, e);
    let reversed = pass(
        &prediction,
        &predicted_extension,
        &phase(g, o),
        o,
        -dt,
        true,
    );
    // The GPU's shared coarse table is republished by prediction extension;
    // correction still binds the original fine transport field.
    let correction_extension = e.with_coarse_from(&predicted_extension);
    (0..g.volume.len())
        .map(|i| {
            let p = g.point(i);
            std::array::from_fn(|a| {
                let mut start = p.map(|v| v as f32 + 0.5);
                start[a] += 0.5;
                let q = departure(g, &correction_extension, start, dt);
                let mut lower = f32::INFINITY;
                let mut upper = f32::NEG_INFINITY;
                for (donor, w) in stencil(g, q, a) {
                    if w > 0.0 {
                        let v = e.fine_face(donor, a);
                        lower = lower.min(v);
                        upper = upper.max(v);
                    }
                }
                let corrected = predicted[i][a] + 0.5 * (g.velocity[i][a] - reversed[i][a]);
                if corrected < lower || corrected > upper {
                    predicted[i][a]
                } else {
                    corrected
                }
            })
        })
        .collect()
}
