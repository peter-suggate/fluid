//! Planar projection of the authored nozzle. Like planar rigid bodies, sources
//! retain XY position and velocity regardless of their authored Z position.
use super::grid::Grid;
use crate::scene_model::FluidInflow;

// initial-fluid.ts inflowStrength.
fn boundary_strength(i: &FluidInflow, t: f64) -> f64 {
    if t < i.start_s || t >= i.end_s {
        return 0.0;
    }
    if i.ramp_s <= 0.0 {
        return 1.0;
    }
    1.0_f64
        .min((t - i.start_s) / i.ramp_s)
        .min((i.end_s - t) / i.ramp_s)
}

// inflow-boundary.ts averageInflowStrength.
fn average_strength(i: &FluidInflow, from: f64, to: f64) -> f64 {
    if to.partial_cmp(&from) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    let mut cuts = vec![
        from,
        to,
        i.start_s,
        i.end_s,
        i.start_s + i.ramp_s,
        i.end_s - i.ramp_s,
        0.5 * (i.start_s + i.end_s),
    ];
    cuts.retain(|&t| t >= from && t <= to);
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let integral: f64 = cuts
        .windows(2)
        .map(|w| 0.5 * (boundary_strength(i, w[0]) + boundary_strength(i, w[1])) * (w[1] - w[0]))
        .sum();
    integral / (to - from)
}

// WGSL length.
fn length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn smooth_coverage(radius: f32, radial: f32, edge: f32) -> f32 {
    let c = (0.5 + 0.5 * (radius - radial) / edge).clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// The nozzle's axis-face velocity boundary: inflow-boundary.ts
/// applyInflowVelocity and applyInflowSweptVelocity. The 2D reduction keeps
/// the planar velocity; the aperture is a segment of width 2r across the
/// dominant-axis face instead of a disk, so its edge width is half a tangent
/// cell and its target projected width is 2r / |direction[axis]|.
pub struct Velocity {
    axis: usize,
    receiver: i32,
    donor: i32,
    face_coordinate: f32,
    minimum: [f32; 2],
    h: [f32; 2],
    outlet: [f32; 2],
    velocity: [f32; 2],
    radius: f32,
    edge: f32,
    strength: f32,
    aperture_scale: f32,
    desired: [f32; 2],
    dt: f32,
}

impl Velocity {
    pub fn new(i: &FluidInflow, time: f64, dt: f32, g: &Grid) -> Option<Self> {
        let strength = (average_strength(i, time, time + dt as f64) as f32).clamp(0.0, 1.0);
        let planar = [i.velocity_m_s.x, i.velocity_m_s.y];
        let speed = planar[0].hypot(planar[1]);
        if strength <= 0.0 || speed <= 1e-6 {
            return None;
        }
        // inflow-boundary.ts inflowOutletCenter uses the authored 3D speed.
        let half_length_over_speed = 0.5 * i.length_m / speed.hypot(i.velocity_m_s.z);
        let outlet64 = [
            i.center_m.x + planar[0] * half_length_over_speed,
            i.center_m.y + planar[1] * half_length_over_speed,
        ];
        let direction64 = planar.map(|v| v / speed);
        let axis = if direction64[1].abs() > direction64[0].abs() {
            1
        } else {
            0
        };
        let tangent = 1 - axis;
        let width = g.dims[0] as f64 * g.h[0] as f64;
        let minimum64 = [-0.5 * width, 0.0];
        let cell = g.h.map(|v| v as f64);
        // Host createInflowGridBoundary: f64 aperture normalization.
        let host_face = ((outlet64[axis] - minimum64[axis]) / cell[axis]).round() - 1.0;
        let host_face = host_face.min(g.dims[axis] as f64 - 2.0).max(0.0);
        let edge64 = 0.5 * cell[tangent];
        let mut weight = 0.0;
        for k in 0..g.dims[tangent] {
            let mut point = [0.0; 2];
            point[axis] = minimum64[axis] + (host_face + 1.0) * cell[axis];
            point[tangent] = minimum64[tangent] + (k as f64 + 0.5) * cell[tangent];
            let relative = [point[0] - outlet64[0], point[1] - outlet64[1]];
            let axial = relative[0] * direction64[0] + relative[1] * direction64[1];
            let radial =
                (relative[0] - axial * direction64[0]).hypot(relative[1] - axial * direction64[1]);
            let c = (0.5 + 0.5 * (i.radius_m - radial) / edge64.max(1e-6)).clamp(0.0, 1.0);
            weight += c * c * (3.0 - 2.0 * c);
        }
        let raw = weight * cell[tangent];
        let target = 2.0 * i.radius_m / direction64[axis].abs().max(1e-6);
        let aperture_scale = if raw > 0.0 {
            (target / raw) as f32
        } else {
            0.0
        };
        // Shader inflowFaceIndex, in f32 with round-half-even.
        let minimum = [-0.5 * (g.dims[0] as f32 * g.h[0]), 0.0];
        let outlet = outlet64.map(|v| v as f32);
        let velocity = planar.map(|v| v as f32);
        let face = (((outlet[axis] - minimum[axis]) / g.h[axis]).round_ties_even() as i32 - 1)
            .clamp(0, g.dims[axis] as i32 - 2);
        let receiver = if velocity[axis] >= 0.0 {
            face + 1
        } else {
            face
        };
        let donor = if velocity[axis] >= 0.0 {
            receiver - 1
        } else {
            receiver + 1
        };
        let mut desired = velocity.map(|v| v * strength);
        desired[axis] *= aperture_scale;
        Some(Self {
            axis,
            receiver,
            donor,
            face_coordinate: minimum[axis] + (face + 1) as f32 * g.h[axis],
            minimum,
            h: g.h,
            outlet,
            velocity,
            radius: i.radius_m as f32,
            edge: (0.5 * g.h[tangent]).max(1e-6),
            strength,
            aperture_scale,
            desired,
            dt,
        })
    }

    fn speed(&self) -> f32 {
        length(self.velocity)
    }

    // inflow-boundary.ts inflowApertureFraction.
    fn aperture(&self, q: [i32; 2]) -> f32 {
        let speed = self.speed();
        let direction = self.velocity.map(|v| v / speed);
        let tangent = 1 - self.axis;
        let mut point = [0.0; 2];
        point[self.axis] = self.face_coordinate;
        point[tangent] = self.minimum[tangent] + (q[tangent] as f32 + 0.5) * self.h[tangent];
        let relative = [point[0] - self.outlet[0], point[1] - self.outlet[1]];
        let axial = relative[0] * direction[0] + relative[1] * direction[1];
        let radial = length([
            relative[0] - axial * direction[0],
            relative[1] - axial * direction[1],
        ]);
        smooth_coverage(self.radius, radial, self.edge)
    }

    // inflow-boundary.ts inflowSweptPlugSource, positivity only: the injected
    // amount belongs to the volume source.
    fn swept_plug(&self, q: [i32; 2]) -> bool {
        let magnitude = self.speed();
        let speed = magnitude * self.strength;
        if speed <= 1e-6 || self.dt <= 0.0 {
            return false;
        }
        let direction = self.velocity.map(|v| v / magnitude.max(1e-6));
        let centre: [f32; 2] =
            std::array::from_fn(|a| self.minimum[a] + (q[a] as f32 + 0.5) * self.h[a]);
        let relative = [centre[0] - self.outlet[0], centre[1] - self.outlet[1]];
        let axial = relative[0] * direction[0] + relative[1] * direction[1];
        let half_axial = 0.5 * (direction[0].abs() * self.h[0] + direction[1].abs() * self.h[1]);
        let overlap =
            ((axial + half_axial).min(speed * self.dt) - (axial - half_axial).max(0.0)).max(0.0);
        if overlap <= 0.0 {
            return false;
        }
        let radial = length([
            relative[0] - axial * direction[0],
            relative[1] - axial * direction[1],
        ]);
        smooth_coverage(self.radius, radial, self.edge) > 0.0 && self.aperture_scale > 0.0
    }

    /// applyInflowVelocity, after projection.
    pub fn apply(&self, q: [i32; 2], mut v: [f32; 2]) -> [f32; 2] {
        let a = self.axis;
        if (q[a] != self.receiver && q[a] != self.donor) || self.aperture(q) <= 0.0 {
            return v;
        }
        if q[a] == self.receiver {
            v = self.desired;
        }
        v[a] = self.desired[a];
        v
    }

    /// applyInflowSweptVelocity, the last body force.
    pub fn apply_swept(&self, q: [i32; 2], v: [f32; 2]) -> [f32; 2] {
        let upstream = q[self.axis] == self.donor && self.aperture(q) > 0.0;
        if upstream || self.swept_plug(q) {
            self.desired
        } else {
            v
        }
    }
}

/// The nozzle as a volume source: inflow-boundary.ts inflowSweptPlugSource,
/// the plug arm of uvSourcePhi, uniformInflowWindowSeed, and the authored
/// velocity finalizeActiveRegion pads with. Planar reduction: the aperture is a
/// segment of width 2r across the dominant-axis face, so its edge width is half
/// a tangent cell and its target projected width is 2r / |direction[axis]|.
/// Positions are world metres with the container centred on x = 0.
pub struct Plug {
    minimum: [f32; 2],
    h: [f32; 2],
    outlet: [f32; 2],
    /// inflowVelocityLength.xyz: the authored velocity, whatever the strength.
    pub velocity: [f32; 2],
    radius: f32,
    edge: f32,
    strength: f32,
    aperture_scale: f32,
    dt: f32,
}

impl Plug {
    pub fn new(i: &FluidInflow, time: f64, dt: f32, g: &Grid) -> Self {
        let strength = (average_strength(i, time, time + dt as f64) as f32).clamp(0.0, 1.0);
        let planar = [i.velocity_m_s.x, i.velocity_m_s.y];
        let speed = planar[0].hypot(planar[1]);
        let full = speed.hypot(i.velocity_m_s.z);
        // inflow-boundary.ts inflowOutletCenter uses the authored 3D speed.
        let half_length_over_speed = if full > 0.0 {
            0.5 * i.length_m / full
        } else {
            0.0
        };
        let outlet64 = [
            i.center_m.x + planar[0] * half_length_over_speed,
            i.center_m.y + planar[1] * half_length_over_speed,
        ];
        let direction64 = if speed > 0.0 {
            planar.map(|v| v / speed)
        } else {
            [1.0, 0.0]
        };
        let axis = if direction64[1].abs() > direction64[0].abs() {
            1
        } else {
            0
        };
        let tangent = 1 - axis;
        let width = g.dims[0] as f64 * g.h[0] as f64;
        let minimum64 = [-0.5 * width, 0.0];
        let cell = g.h.map(|v| v as f64);
        // Host createInflowGridBoundary: f64 aperture normalization.
        let host_face = ((outlet64[axis] - minimum64[axis]) / cell[axis]).round() - 1.0;
        let host_face = host_face.min(g.dims[axis] as f64 - 2.0).max(0.0);
        let edge64 = 0.5 * cell[tangent];
        let mut weight = 0.0;
        for k in 0..g.dims[tangent] {
            let mut point = [0.0; 2];
            point[axis] = minimum64[axis] + (host_face + 1.0) * cell[axis];
            point[tangent] = minimum64[tangent] + (k as f64 + 0.5) * cell[tangent];
            let relative = [point[0] - outlet64[0], point[1] - outlet64[1]];
            let axial = relative[0] * direction64[0] + relative[1] * direction64[1];
            let radial =
                (relative[0] - axial * direction64[0]).hypot(relative[1] - axial * direction64[1]);
            let c = (0.5 + 0.5 * (i.radius_m - radial) / edge64.max(1e-6)).clamp(0.0, 1.0);
            weight += c * c * (3.0 - 2.0 * c);
        }
        let raw = weight * cell[tangent];
        let target = if speed > 0.0 {
            2.0 * i.radius_m / direction64[axis].abs().max(1e-6)
        } else {
            0.0
        };
        Self {
            minimum: [-0.5 * (g.dims[0] as f32 * g.h[0]), 0.0],
            h: g.h,
            outlet: outlet64.map(|v| v as f32),
            velocity: planar.map(|v| v as f32),
            radius: i.radius_m as f32,
            edge: (0.5 * g.h[tangent]).max(1e-6),
            strength,
            aperture_scale: if raw > 0.0 {
                (target / raw) as f32
            } else {
                0.0
            },
            dt,
        }
    }

    /// The plug's advance speed this step, |u| * strength.
    fn speed(&self) -> f32 {
        length(self.velocity) * self.strength
    }

    /// The time-averaged strength over this step.
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// The inflow arm of uvStepHasExternalSource.
    pub fn active(&self) -> bool {
        self.speed() > 1e-6
    }

    /// inflowSweptPlugSource: the cell's share of the plug swept this step.
    pub fn amount(&self, q: [i32; 2]) -> f32 {
        let magnitude = length(self.velocity);
        let speed = magnitude * self.strength;
        if speed <= 1e-6 || self.dt <= 0.0 {
            return 0.0;
        }
        let direction = self.velocity.map(|v| v / magnitude.max(1e-6));
        let centre: [f32; 2] =
            std::array::from_fn(|a| self.minimum[a] + (q[a] as f32 + 0.5) * self.h[a]);
        let relative = [centre[0] - self.outlet[0], centre[1] - self.outlet[1]];
        let axial = relative[0] * direction[0] + relative[1] * direction[1];
        let half_axial = 0.5 * (direction[0].abs() * self.h[0] + direction[1].abs() * self.h[1]);
        let overlap =
            ((axial + half_axial).min(speed * self.dt) - (axial - half_axial).max(0.0)).max(0.0);
        if overlap <= 0.0 {
            return 0.0;
        }
        let radial = length([
            relative[0] - axial * direction[0],
            relative[1] - axial * direction[1],
        ]);
        overlap / (2.0 * half_axial).max(1e-6)
            * smooth_coverage(self.radius, radial, self.edge)
            * self.aperture_scale
    }

    /// The plug arm of uvSourcePhi at trace position p (lattice units).
    pub fn phi(&self, p: [f32; 2], phi: f32) -> f32 {
        let speed = self.speed();
        if speed <= 1e-6 {
            return phi;
        }
        let magnitude = length(self.velocity);
        let direction = self.velocity.map(|v| v / magnitude);
        let delta: [f32; 2] =
            std::array::from_fn(|a| self.minimum[a] + p[a] * self.h[a] - self.outlet[a]);
        let axial = delta[0] * direction[0] + delta[1] * direction[1];
        let plug = (length([
            delta[0] - axial * direction[0],
            delta[1] - axial * direction[1],
        ]) - self.radius)
            .max((-axial).max(axial - speed * self.dt));
        phi.min(plug)
    }

    /// uniformInflowWindowSeed: the conservative swept-inlet footprint.
    pub fn window_seed(&self, q: [i32; 2]) -> bool {
        if self.strength <= 0.0 {
            return false;
        }
        let end: [f32; 2] = std::array::from_fn(|a| self.outlet[a] + self.velocity[a] * self.dt);
        let pad = self.radius + length(self.h);
        (0..2).all(|a| {
            let world = self.minimum[a] + (q[a] as f32 + 0.5) * self.h[a];
            world >= self.outlet[a].min(end[a]) - pad && world <= self.outlet[a].max(end[a]) + pad
        })
    }
}
