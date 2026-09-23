//! CM11b semi-Lagrangian velocity transport: reference.wgsl semiLagrangianAdvection.
use super::{
    extension::{Extension, FINE},
    grid::Grid,
    options::UniformGeometricOptions,
};

/// Static solid voxel occupancy with the one-cell halo the GPU packs. The 2D
/// domain walls are the voxel shell the parity scenes author
/// (solid-world.ts boxSolidVoxelShell with the container's top); cells inside
/// the domain use the capacity field as their voxel proxy.
pub(super) fn solid_voxel(g: &Grid, c: [i32; 2]) -> bool {
    if let Some(i) = g.index(c) {
        return g.capacity[i] <= 1e-5;
    }
    let n = g.dims.map(|v| v as i32);
    let inside = |a: usize| c[a] >= 0 && c[a] < n[a];
    (inside(1) && (c[0] == -1 || c[0] == n[0]))
        || (inside(0) && (c[1] == -1 || (c[1] == n[1] && !g.open_top)))
}

// reference.wgsl clampVelocityTraceToDomain.
fn clamp(g: &Grid, mut p: [f32; 2]) -> [f32; 2] {
    p[0] = p[0].clamp(0.0, g.dims[0] as f32);
    p[1] = p[1].max(0.0);
    if !g.open_top {
        p[1] = p[1].min(g.dims[1] as f32);
    }
    p
}

// reference.wgsl staticSolidVoxelAtWorld(traceWorld(p)), including the world
// round trip that decides which voxel an endpoint on a cell edge lands in.
fn trace_solid(g: &Grid, p: [f32; 2]) -> bool {
    let width = g.dims[0] as f32 * g.h[0];
    let world = [-0.5 * width + p[0] * g.h[0], p[1] * g.h[1]];
    let cell = [
        ((world[0] + 0.5 * width) / g.h[0]).floor() as i32,
        (world[1] / g.h[1]).floor() as i32,
    ];
    solid_voxel(g, cell)
}

// reference.wgsl departurePoint then clipDepartureAtSolid.
fn departure(g: &Grid, e: &Extension, start: [f32; 2], dt: f32) -> [f32; 2] {
    let mut q = start;
    let mut remaining = dt.abs();
    let direction = if dt >= 0.0 { 1.0 } else { -1.0 };
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
    if !trace_solid(g, q) {
        return q;
    }
    // MSL mix(x, y, t) is x + (y - x) * t.
    let mix = |t: f32| -> [f32; 2] { std::array::from_fn(|a| start[a] + (q[a] - start[a]) * t) };
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..8 {
        let mid = 0.5 * (lo + hi);
        if trace_solid(g, mix(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    mix(lo)
}

/// E2b's work set: every cell, or with tiled advection only the cells of FINE
/// tiles (uvTwoLevelFineAt), in raster order within each row.
pub(super) fn for_each_fine(
    g: &Grid,
    e: &Extension,
    o: &UniformGeometricOptions,
    mut f: impl FnMut(usize, [i32; 2]),
) {
    let [nx, ny] = g.dims;
    if !(o.two_level_advection == "tiles" && e.two_level.enabled) {
        for i in 0..nx * ny {
            f(i, [(i % nx) as i32, (i / nx) as i32]);
        }
        return;
    }
    let tiles = nx.div_ceil(4);
    for y in 0..ny {
        for tx in 0..tiles {
            if e.classes[tx + tiles * (y / 4)] & FINE == 0 {
                continue;
            }
            for x in 4 * tx..(4 * tx + 4).min(nx) {
                f(x + nx * y, [x as i32, y as i32]);
            }
        }
    }
}

/// Writes the advected field into `out`. The caller applies body forces
/// exactly once. E2b: a cell outside the fine tiles carries zero velocity and
/// no forces.
pub fn advect(
    g: &Grid,
    e: &Extension,
    o: &UniformGeometricOptions,
    dt: f32,
    out: &mut Vec<[f32; 2]>,
) {
    out.clear();
    out.resize(g.volume.len(), [0.0; 2]);
    for_each_fine(g, e, o, |i, p| {
        out[i] = std::array::from_fn(|a| {
            let mut start = p.map(|v| v as f32 + 0.5);
            start[a] += 0.5;
            let v = e.sample(departure(g, e, start, dt))[a];
            // Keep an old velocity directed away from a positive wall.
            if p[a] == g.dims[a] as i32 - 1 && !(a == 1 && g.open_top) {
                v.min(g.velocity[i][a])
            } else {
                v
            }
        });
    });
}
