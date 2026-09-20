//! Read-only measurements for native scene experiments. Areas are in cell units.
use super::{extension::Extension, grid::Grid, UniformGeometricOptions};

/// Integrate the negative part of a bilinear cell. Each horizontal slice is
/// linear and integrated exactly; split at edge roots before midpoint quadrature.
/// Corner order: bottom-left, bottom-right, top-left, top-right.
pub fn bilinear_fill(c: [f64; 4], slices: usize) -> f64 {
    if c.iter().all(|&v| v < 0.0) {
        return 1.0;
    }
    if c.iter().all(|&v| v > 0.0) {
        return 0.0;
    }
    let mut cuts = vec![0.0, 1.0];
    for (a, b) in [(c[0], c[2]), (c[1], c[3])] {
        if a != b {
            let root = a / (a - b);
            if root > 0.0 && root < 1.0 {
                cuts.push(root);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    let mut fill = 0.0;
    for interval in cuts.windows(2) {
        let dy = (interval[1] - interval[0]) / slices as f64;
        for s in 0..slices {
            let y = interval[0] + (s as f64 + 0.5) * dy;
            let left = c[0] + y * (c[2] - c[0]);
            let right = c[1] + y * (c[3] - c[1]);
            let width = if left == right {
                if left < 0.0 {
                    1.0
                } else if left == 0.0 {
                    0.5
                } else {
                    0.0
                }
            } else {
                let root = (left / (left - right)).clamp(0.0, 1.0);
                if left < right {
                    root
                } else {
                    1.0 - root
                }
            };
            fill += dy * width;
        }
    }
    fill
}

#[derive(Default, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub volume: f64,
    pub target_area: f64,
    pub contour_area: f64,
    pub target_l1: f64,
    pub contour_l1: f64,
    pub interior_deficit: f64,
    pub band_residual: f64,
    pub far_air_volume: f64,
    pub rowless_volume: f64,
    pub excess_over_capacity: f64,
    pub max_volume: f32,
    pub max_courant: f32,
    pub liquid_cells: usize,
    pub front_cell: usize,
}
pub fn measure(g: &Grid, dt: f32, slices: usize) -> Metrics {
    let mut m = Metrics::default();
    let h = g.h[0].min(g.h[1]);
    let nxv = g.dims[0] + 1;
    for i in 0..g.volume.len() {
        let p = g.point(i);
        let v = g.volume[i] as f64;
        let target = g.target(p) as f64;
        let corner = p[0] as usize + nxv * p[1] as usize;
        let c = [corner, corner + 1, corner + nxv, corner + nxv + 1].map(|j| g.phi[j] as f64);
        let area = bilinear_fill(c, slices) * g.capacity[i] as f64;
        let phi = g.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]);
        m.volume += v;
        m.target_area += target;
        m.contour_area += area;
        m.target_l1 += (v - target).abs();
        m.contour_l1 += (v - area).abs();
        m.max_volume = m.max_volume.max(g.volume[i]);
        m.excess_over_capacity += (v - g.capacity[i] as f64).max(0.0);
        if phi < -2.1 * h {
            m.interior_deficit += (g.capacity[i] as f64 - v).max(0.0);
        } else if phi > 2.1 * h {
            m.far_air_volume += v;
        } else {
            m.band_residual += v - target;
        }
        if phi >= 0.0 {
            m.rowless_volume += v;
        } else if g.capacity[i] > 1e-5 {
            m.liquid_cells += 1;
        }
        if area > 0.01 {
            m.front_cell = m.front_cell.max(p[0] as usize);
        }
        for a in 0..2 {
            m.max_courant = m.max_courant.max(dt * g.velocity[i][a].abs() / g.h[a]);
        }
    }
    m
}

/// Keep the starting state and extended velocity fixed. Refine only the RK2
/// characteristic integration, with ONE phi interpolation at the final foot.
/// Raw pullbacks deliberately omit surface::advect's contact/source treatment;
/// the production one-step replay is included to expose that difference.
pub fn trace_replays(g: &Grid, options: &UniformGeometricOptions, dt: f32) -> serde_json::Value {
    let mut result = Vec::new();
    for variant in ["default", "front16", "fine", "fine-front16"] {
        let mut o = options.clone();
        if variant.contains("front16") {
            o.extension_front_sweeps = 16.0;
        }
        if variant.contains("fine") {
            o.two_level_velocity = "off".into();
        }
        let e = Extension::build(g, &o, dt);
        let flux = liquid_divergence(g, &e);
        let mut production = g.clone();
        super::surface::advect(&mut production, &e, &o, dt);
        let production_area = measure(&production, dt, 32).contour_area;
        for steps in [1, 4, 16, 64] {
            let mut replay = g.clone();
            let mut feet = vec![[0.0; 2]; g.phi.len()];
            for y in 0..=g.dims[1] {
                for x in 0..=g.dims[0] {
                    let i = x + (g.dims[0] + 1) * y;
                    let mut foot = [x as f32, y as f32];
                    for _ in 0..steps {
                        foot = e.trace(g, foot, dt / steps as f32);
                    }
                    replay.phi[i] = g.phi_at(foot);
                    feet[i] = foot;
                }
            }
            let mut inverted_band_triangles = 0;
            let mut minimum_band_jacobian = 1.0_f32;
            for i in 0..g.volume.len() {
                let p = g.point(i);
                if g.phi_at([p[0] as f32 + 0.5, p[1] as f32 + 0.5]).abs() > 2.1 * g.h[0] {
                    continue;
                }
                let j = p[0] as usize + (g.dims[0] + 1) * p[1] as usize;
                let [a, b, c, d] =
                    [j, j + 1, j + g.dims[0] + 1, j + g.dims[0] + 2].map(|k| feet[k]);
                for [origin, u, v] in [[a, b, c], [d, c, b]] {
                    let det = (u[0] - origin[0]) * (v[1] - origin[1])
                        - (u[1] - origin[1]) * (v[0] - origin[0]);
                    minimum_band_jacobian = minimum_band_jacobian.min(det);
                    if det < -1e-5 {
                        inverted_band_triangles += 1;
                    }
                }
            }
            result.push(serde_json::json!({"variant":variant,"steps":steps,
                "contourArea":measure(&replay,dt,32).contour_area,
                "productionArea":production_area,
                "liquidDivergence":flux,
                "invertedBandTriangles":inverted_band_triangles,
                "minimumBandJacobian":minimum_band_jacobian}));
        }
    }
    let e = Extension::build(g, options, dt);
    let resolved_pullbacks: Vec<_> = [1, 2, 4, 8]
        .into_iter()
        .map(|resolution| {
            serde_json::json!({"resolution":resolution,"traceSteps":16,
            "contourArea":resolved_pullback_area(g,&e,dt,resolution,16).0})
        })
        .collect();
    let durations: Vec<_> = [0.0, 1.0 / 16.0, 0.25, 0.5, 1.0]
        .into_iter()
        .map(|fraction| {
            let (area, divergence, air_divergence, air_area) =
                resolved_pullback_area(g, &e, dt * fraction, 4, 16);
            serde_json::json!({"fraction":fraction,"resolution":4,"traceSteps":16,
            "contourArea":area,"integratedDivergence":divergence,
            "divergenceInOriginalAir":air_divergence,"areaInOriginalAir":air_area})
        })
        .collect();
    serde_json::json!({"traces":result,"resolvedPullbacks":resolved_pullbacks,"durations":durations})
}

/// Integral of div(u) over phi-liquid, using the actual MAC interpolation seen
/// by phi. Units: cell areas/second. Pressure constrains cell rows, so split
/// partially covered cells with and without a liquid centre.
fn liquid_divergence(g: &Grid, e: &Extension) -> serde_json::Value {
    let mut interior = 0.0_f64;
    let mut liquid_centre_band = 0.0_f64;
    let mut air_centre_band = 0.0_f64;
    let mut row_divergence = 0.0_f64;
    const R: usize = 8;
    for i in 0..g.volume.len() {
        if g.capacity[i] < 0.99999 {
            continue;
        }
        let cell = g.point(i);
        let phi = g.phi_at(cell.map(|v| v as f32 + 0.5));
        if phi < 0.0 {
            for a in 0..2 {
                let mut lo = cell;
                lo[a] -= 1;
                row_divergence += ((g.face(cell, a) - g.face(lo, a)) / g.h[a]) as f64;
            }
        }
        for y in 0..R {
            for x in 0..R {
                let p = [
                    cell[0] as f32 + (x as f32 + 0.5) / R as f32,
                    cell[1] as f32 + (y as f32 + 0.5) / R as f32,
                ];
                let c = [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
                    .map(|d| g.phi_at([p[0] + d[0] / R as f32, p[1] + d[1] / R as f32]) as f64);
                if c.iter().all(|&v| v >= 0.0) {
                    continue;
                }
                let fill = bilinear_fill(c, 8) / (R * R) as f64;
                let mut div = 0.0_f64;
                for a in 0..2 {
                    let mut lo = p;
                    let mut hi = p;
                    lo[a] -= 0.001;
                    hi[a] += 0.001;
                    div +=
                        ((e.sample(hi)[a] - e.sample(lo)[a]) / ((hi[a] - lo[a]) * g.h[a])) as f64;
                }
                if phi < -2.1 * g.h[0] {
                    interior += fill * div;
                } else if phi < 0.0 {
                    liquid_centre_band += fill * div;
                } else {
                    air_centre_band += fill * div;
                }
            }
        }
    }
    serde_json::json!({"interior":interior,"liquidCentreBand":liquid_centre_band,
        "airCentreBand":air_centre_band,"total":interior+liquid_centre_band+air_centre_band,
        "projectedLiquidRowSum":row_divergence})
}

/// Reconstruct the SAME frozen-velocity pullback on successively finer observer
/// lattices. This changes only measurement, not the simulated grid or velocity.
fn resolved_pullback_area(
    g: &Grid,
    e: &Extension,
    dt: f32,
    resolution: usize,
    steps: usize,
) -> (f64, f64, f64, f64) {
    let nx = g.dims[0] * resolution;
    let ny = g.dims[1] * resolution;
    let mut previous = vec![0.0_f64; nx + 1];
    let mut next = previous.clone();
    let mut area = 0.0;
    let mut divergence = 0.0;
    let mut air_divergence = 0.0;
    let mut air_area = 0.0;
    for y in 0..=ny {
        for x in 0..=nx {
            let mut p = [x as f32 / resolution as f32, y as f32 / resolution as f32];
            for _ in 0..steps {
                p = e.trace(g, p, dt / steps as f32);
            }
            next[x] = g.phi_at(p) as f64;
        }
        if y > 0 {
            for x in 0..nx {
                let i = x / resolution + g.dims[0] * ((y - 1) / resolution);
                let fill = bilinear_fill([previous[x], previous[x + 1], next[x], next[x + 1]], 8)
                    * g.capacity[i] as f64
                    / (resolution * resolution) as f64;
                area += fill;
                if fill > 0.0 {
                    let p = [
                        (x as f32 + 0.5) / resolution as f32,
                        (y as f32 - 0.5) / resolution as f32,
                    ];
                    let was_air = g.phi_at(p) >= 0.0;
                    if was_air {
                        air_area += fill;
                    }
                    for a in 0..2 {
                        let mut lo = p;
                        let mut hi = p;
                        lo[a] -= 0.001;
                        hi[a] += 0.001;
                        let contribution = fill
                            * ((e.sample(hi)[a] - e.sample(lo)[a]) / ((hi[a] - lo[a]) * g.h[a]))
                                as f64;
                        divergence += contribution;
                        if was_air {
                            air_divergence += contribution;
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut previous, &mut next);
    }
    (area, divergence, air_divergence, air_area)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_contour_area() {
        for n in [8, 32, 128] {
            assert!((bilinear_fill([-0.3, 0.7, -0.3, 0.7], n) - 0.3).abs() < 1e-12);
            assert!((bilinear_fill([-1.0, 0.0, 0.0, 1.0], n) - 0.5).abs() < 1e-12);
            assert!((bilinear_fill([-1.0, 1.0, 1.0, -1.0], n) - 0.5).abs() < 1e-12);
            // phi = xy - 1/4: area = a - a ln(a).
            let a = 0.25_f64;
            assert!((bilinear_fill([-a, -a, -a, 1.0 - a], n) - (a - a * a.ln())).abs() < 0.002);
        }
    }

    #[test]
    fn observations_and_frozen_replays_do_not_change_the_advance() {
        use super::super::world::World;
        let mut grid = Grid::new([16, 16], [0.05; 2], false).unwrap();
        for y in 0..=16 {
            for x in 0..=16 {
                grid.phi[x + 17 * y] = 0.05 * ((x as f32 - 5.0).hypot(y as f32 - 6.0) - 3.5);
            }
        }
        for i in 0..grid.volume.len() {
            grid.volume[i] = grid.target(grid.point(i));
        }
        let mut options = UniformGeometricOptions::default();
        options.active_region = "off".into();
        let mut normal = World::from_grid(
            grid.clone(),
            options.clone(),
            [0.0, -9.81],
            1000.0,
            0.0,
            0.0,
        )
        .unwrap();
        let mut observed =
            World::from_grid(grid, options.clone(), [0.0, -9.81], 1000.0, 0.0, 0.0).unwrap();
        for _ in 0..4 {
            normal.advance(1.0 / 30.0).unwrap();
            let mut names = Vec::new();
            observed
                .advance_observed(1.0 / 30.0, 8, |name, g| {
                    names.push(name.to_string());
                    assert!(measure(g, 1.0 / 30.0, 32).contour_area.is_finite());
                    if name == "transported" {
                        let mut copy = g.clone();
                        super::super::surface::sharpen_rounds(&mut copy, &options, 32);
                    }
                })
                .unwrap();
            assert_eq!(
                names,
                [
                    "start",
                    "advected",
                    "redistanced",
                    "transported",
                    "corrected",
                    "sharpened",
                    "projected"
                ]
            );
            assert_eq!(normal.grid.phi, observed.grid.phi);
            assert_eq!(normal.grid.volume, observed.grid.volume);
            assert_eq!(normal.grid.velocity, observed.grid.velocity);
            assert_eq!(normal.grid.released, observed.grid.released);
            assert_eq!(normal.grid.low_x, observed.grid.low_x);
            assert_eq!(normal.grid.low_y, observed.grid.low_y);
            assert_eq!(
                serde_json::to_value(&normal.receipt).unwrap(),
                serde_json::to_value(&observed.receipt).unwrap()
            );
        }
    }
}
