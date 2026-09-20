//! Planar projection of the authored nozzle. Like planar rigid bodies, sources
//! retain XY position and velocity regardless of their authored Z position.
//! The emitted area is 2 * radius * planar speed * integrated strength.
use super::grid::Grid;
use crate::scene_model::FluidInflow;

pub struct Step {
    outlet: [f32; 2],
    direction: [f32; 2],
    velocity: [f32; 2],
    length: f32,
    nozzle_length: f32,
    radius: f32,
}

fn strength(i: &FluidInflow, t: f64) -> f64 {
    if t < i.start_s || t > i.end_s {
        return 0.0;
    }
    if i.ramp_s == 0.0 {
        return 1.0;
    }
    1.0_f64
        .min((t - i.start_s) / i.ramp_s)
        .min((i.end_s - t) / i.ramp_s)
        .max(0.0)
}

impl Step {
    pub fn new(i: &FluidInflow, time: f64, dt: f32, g: &Grid) -> Option<Self> {
        let end = time + dt as f64;
        let mut cuts = vec![
            time,
            end,
            i.start_s,
            i.end_s,
            i.start_s + i.ramp_s,
            i.end_s - i.ramp_s,
            0.5 * (i.start_s + i.end_s),
        ];
        cuts.retain(|&t| t >= time && t <= end);
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let integral: f64 = cuts
            .windows(2)
            .map(|w| {
                // Midpoints also handle discontinuous start/end when ramp is zero.
                strength(i, 0.5 * (w[0] + w[1])) * (w[1] - w[0])
            })
            .sum();
        let speed = i.velocity_m_s.x.hypot(i.velocity_m_s.y);
        if integral <= 0.0 || speed <= 1e-9 {
            return None;
        }
        let direction = [
            (i.velocity_m_s.x / speed) as f32,
            (i.velocity_m_s.y / speed) as f32,
        ];
        let full_speed = speed.hypot(i.velocity_m_s.z);
        Some(Self {
            outlet: [
                i.center_m.x as f32
                    + 0.5 * g.dims[0] as f32 * g.h[0]
                    + (0.5 * i.length_m * i.velocity_m_s.x / full_speed) as f32,
                i.center_m.y as f32 + (0.5 * i.length_m * i.velocity_m_s.y / full_speed) as f32,
            ],
            direction,
            velocity: [
                (i.velocity_m_s.x * integral / dt as f64) as f32,
                (i.velocity_m_s.y * integral / dt as f64) as f32,
            ],
            length: (speed * integral) as f32,
            nozzle_length: (i.length_m * speed / full_speed) as f32,
            radius: i.radius_m as f32,
        })
    }

    fn coordinates(&self, p: [f32; 2]) -> [f32; 2] {
        let q = [p[0] - self.outlet[0], p[1] - self.outlet[1]];
        [
            q[0] * self.direction[0] + q[1] * self.direction[1],
            -q[0] * self.direction[1] + q[1] * self.direction[0],
        ]
    }

    // Clip a cell against the swept rectangular jet. Exact area keeps small
    // nozzles alive even when no cell centre lies inside the source.
    fn fraction(&self, g: &Grid, p: [i32; 2]) -> f32 {
        let center = self.coordinates([(p[0] as f32 + 0.5) * g.h[0], (p[1] as f32 + 0.5) * g.h[1]]);
        let axial_half =
            0.5 * (self.direction[0].abs() * g.h[0] + self.direction[1].abs() * g.h[1]);
        let radial_half =
            0.5 * (self.direction[1].abs() * g.h[0] + self.direction[0].abs() * g.h[1]);
        if center[0] + axial_half <= 0.
            || center[0] - axial_half >= self.length
            || center[1].abs() - radial_half >= self.radius
        {
            return 0.;
        }
        let mut polygon: Vec<_> = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .map(|v| {
                self.coordinates([(p[0] as f32 + v[0]) * g.h[0], (p[1] as f32 + v[1]) * g.h[1]])
            })
            .to_vec();
        for (axis, sign, bound) in [
            (0, -1., 0.),
            (0, 1., self.length),
            (1, -1., self.radius),
            (1, 1., self.radius),
        ] {
            let mut clipped = Vec::new();
            for j in 0..polygon.len() {
                let a = polygon[j];
                let b = polygon[(j + 1) % polygon.len()];
                let da = sign * a[axis] - bound;
                let db = sign * b[axis] - bound;
                if da <= 0. {
                    clipped.push(a);
                }
                if (da <= 0.) != (db <= 0.) {
                    let t = da / (da - db);
                    clipped.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            }
            polygon = clipped;
        }
        let area: f32 = (0..polygon.len())
            .map(|j| {
                let a = polygon[j];
                let b = polygon[(j + 1) % polygon.len()];
                a[0] * b[1] - a[1] * b[0]
            })
            .sum();
        (0.5 * area.abs() / (g.h[0] * g.h[1])).clamp(0., 1.)
    }

    pub fn inject(&self, g: &mut Grid) -> f64 {
        let mut added = 0.0;
        for i in 0..g.volume.len() {
            let amount = self
                .fraction(g, g.point(i))
                .min((g.capacity[i] - g.volume[i]).max(0.));
            g.volume[i] += amount;
            added += amount as f64;
        }
        if added > 0.0 {
            for y in 0..=g.dims[1] {
                for x in 0..=g.dims[0] {
                    let p = self.coordinates([x as f32 * g.h[0], y as f32 * g.h[1]]);
                    let q = [
                        // Include the wetted nozzle in the surface boundary.
                        // A short per-frame plug alone can miss every vertex,
                        // leaving newly emitted liquid without a signed surface.
                        (p[0] - 0.5 * (self.length - self.nozzle_length)).abs()
                            - 0.5 * (self.length + self.nozzle_length),
                        p[1].abs() - self.radius,
                    ];
                    let phi = q[0].max(0.).hypot(q[1].max(0.)) + q[0].max(q[1]).min(0.);
                    let j = x + (g.dims[0] + 1) * y;
                    g.phi[j] = g.phi[j].min(phi);
                }
            }
        }
        added
    }

    pub fn enforce_velocity(&self, g: &mut Grid) {
        for i in 0..g.volume.len() {
            let p = g.point(i);
            if g.capacity[i] <= 1e-5 || self.fraction(g, p) <= 0. {
                continue;
            }
            for axis in 0..2 {
                let mut neighbor = p;
                neighbor[axis] += 1;
                if g.open(neighbor) > 1e-5 {
                    g.velocity[i][axis] = self.velocity[axis];
                }
                neighbor[axis] -= 2;
                if let Some(j) = g.index(neighbor).filter(|&j| g.capacity[j] > 1e-5) {
                    g.velocity[j][axis] = self.velocity[axis];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_model::Vec3;
    fn source() -> FluidInflow {
        FluidInflow {
            center_m: Vec3 {
                x: 0.,
                y: 0.5,
                z: 0.2,
            },
            velocity_m_s: Vec3 {
                x: 0.8,
                y: -0.6,
                z: 0.,
            },
            radius_m: 0.013,
            length_m: 0.05,
            start_s: 0.,
            end_s: 1.,
            ramp_s: 0.,
        }
    }
    #[test]
    fn subcell_diagonal_jet_has_exact_planar_area_and_velocity() {
        let mut g = Grid::new([20, 20], [0.05, 0.05], true).unwrap();
        let s = source();
        let step = Step::new(&s, 0., 0.01, &g).unwrap();
        let added = step.inject(&mut g);
        assert!((added * 0.0025 - 2. * s.radius_m * 0.01).abs() < 1e-8);
        step.enforce_velocity(&mut g);
        assert!(g.velocity.iter().any(|v| v[0] == 0.8));
        assert!(g.velocity.iter().any(|v| v[1] == -0.6));
        assert!(g.phi.iter().any(|&v| v < 0.05));
    }
    #[test]
    fn timing_integrates_ramps_and_start_stop_crossings() {
        let g = Grid::new([20, 20], [0.05, 0.05], true).unwrap();
        let mut s = source();
        s.start_s = 1.;
        s.end_s = 3.;
        s.ramp_s = 0.5;
        assert!(Step::new(&s, 0., 0.5, &g).is_none());
        assert!(Step::new(&s, 3., 0.5, &g).is_none());
        assert!((Step::new(&s, 0.5, 1., &g).unwrap().length - 0.25).abs() < 1e-6);
        assert!((Step::new(&s, 2.5, 1., &g).unwrap().length - 0.25).abs() < 1e-6);
        s.ramp_s = 0.;
        assert!((Step::new(&s, 0.5, 1., &g).unwrap().length - 0.5).abs() < 1e-6);
    }
    #[test]
    fn solids_and_full_cells_do_not_accept_water() {
        let mut g = Grid::new([20, 20], [0.05, 0.05], true).unwrap();
        let step = Step::new(&source(), 0., 0.1, &g).unwrap();
        g.capacity.fill(0.);
        assert_eq!(step.inject(&mut g), 0.);
        step.enforce_velocity(&mut g);
        assert!(g.velocity.iter().all(|v| *v == [0., 0.]));
        assert!(g.phi.iter().all(|&v| v == 1.));
        g.capacity.fill(1.);
        g.volume.fill(1.);
        assert_eq!(step.inject(&mut g), 0.);
    }
}
