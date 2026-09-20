//! Uniform-grid adapter for the shared packed rigid integrator.
use super::grid::Grid;
use crate::{
    initial_scene::SceneDocument,
    rigid::{self, FluidExchange, RigidBodyState},
    scene_model::{RigidBodyDescription, Vec3},
    solid_world::{fluid_solid_world_for_scene, CompiledStaticSolidWorld},
    types::ValidationError,
};

pub struct Physical {
    pub bodies: Vec<RigidBodyState>,
    base_capacity: Vec<f32>,
    contacts: CompiledStaticSolidWorld,
    gravity: Vec3,
    density: f32,
    depth: f32,
}
fn point(g: &Grid, p: [f32; 2]) -> Vec3 {
    Vec3 {
        x: ((p[0] - g.dims[0] as f32 * 0.5) * g.h[0]) as f64,
        y: (p[1] * g.h[1]) as f64,
        z: 0.0,
    }
}
fn velocity(b: &RigidBodyState, p: Vec3) -> [f32; 2] {
    let w = b.angular_velocity_rad_s.z as f32;
    [
        b.linear_velocity_m_s.x as f32 - w * (p.y - b.position_m.y) as f32,
        b.linear_velocity_m_s.y as f32 + w * (p.x - b.position_m.x) as f32,
    ]
}
impl Physical {
    pub fn new(scene: &SceneDocument, grid: &Grid) -> Result<Self, ValidationError> {
        let mut result = Self {
            bodies: Vec::new(),
            base_capacity: grid.capacity.clone(),
            contacts: fluid_solid_world_for_scene(scene),
            gravity: scene.fluid.gravity_m_s2,
            density: scene.fluid.density_kg_m3 as f32,
            depth: grid.h[0].min(grid.h[1]),
        };
        for body in &scene.rigid_bodies {
            result.add(body.clone())?;
        }
        Ok(result)
    }
    pub fn add(&mut self, mut body: RigidBodyDescription) -> Result<(), ValidationError> {
        if self.bodies.len() >= 12 {
            return Err(ValidationError(
                "At most 12 rigid bodies are supported".into(),
            ));
        }
        let values = [
            body.dimensions_m.x,
            body.dimensions_m.y,
            body.dimensions_m.z,
            body.density_kg_m3,
        ];
        if values.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || body.position_m.array().iter().any(|x| !x.is_finite())
        {
            return Err(ValidationError(
                "Invalid rigid body dimensions or position".into(),
            ));
        }
        body.position_m.z = 0.0;
        body.linear_velocity_m_s.z = 0.0;
        body.angular_velocity_rad_s.x = 0.0;
        body.angular_velocity_rad_s.y = 0.0;
        self.bodies
            .push(rigid::initialize_body_2d(body, self.depth as f64)?);
        Ok(())
    }
    fn fraction(&self, g: &Grid, b: &RigidBodyState, p: [f32; 2]) -> f32 {
        let mut inside = 0.0;
        for k in 0..4 {
            let q = point(
                g,
                [
                    p[0] + if k & 1 == 0 { -0.4 } else { 0.4 },
                    p[1] + if k & 2 == 0 { -0.4 } else { 0.4 },
                ],
            );
            if rigid::body_contains(b, q).unwrap_or(false) {
                inside += 0.25;
            }
        }
        inside
    }
    pub fn geometry(&self, g: &mut Grid) {
        if self.bodies.is_empty() {
            if g.rigid_faces.is_some() {
                g.capacity.clone_from(&self.base_capacity);
                g.rigid_faces = None;
                g.rigid_speeds = None;
                g.rigid_centres = None;
            }
            return;
        }
        let n = g.volume.len();
        let mut faces = vec![[0.0; 2]; n];
        let mut speeds = vec![[0.0; 2]; n];
        let mut centres = vec![false; n];
        for i in 0..n {
            let p = g.point(i);
            let centre = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
            let fraction = self
                .bodies
                .iter()
                .map(|b| self.fraction(g, b, centre))
                .fold(0.0, f32::max);
            g.capacity[i] = self.base_capacity[i] * (1.0 - fraction);
            centres[i] = self
                .bodies
                .iter()
                .any(|b| rigid::body_contains(b, point(g, centre)).unwrap_or(false));
        }
        for i in 0..n {
            let p = g.point(i);
            for a in 0..2 {
                let mut q = p;
                q[a] += 1;
                // Keep domain/static boundary rules identical to the body-free world.
                if g.index(q).is_none()
                    || self.base_capacity[i] < 0.99999
                    || g.index(q).is_some_and(|j| self.base_capacity[j] < 0.99999)
                {
                    faces[i][a] = g.base_pressure_face(p, a);
                    continue;
                }
                let mut at = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
                at[a] += 0.5;
                let mut solid = 0.0;
                let mut speed = 0.0;
                for k in 0..4 {
                    let sample = point(
                        g,
                        [
                            at[0] + if k & 1 == 0 { -0.4 } else { 0.4 },
                            at[1] + if k & 2 == 0 { -0.4 } else { 0.4 },
                        ],
                    );
                    if let Some(b) = self
                        .bodies
                        .iter()
                        .find(|b| rigid::body_contains(b, sample).unwrap_or(false))
                    {
                        solid += 1.0;
                        speed += velocity(b, sample)[a];
                    }
                }
                faces[i][a] = 1.0 - solid * 0.25;
                if solid > 0.0 {
                    speeds[i][a] = speed / solid;
                } else {
                    let world = point(g, at);
                    let mut nearest = g.h[0].max(g.h[1]);
                    for b in &self.bodies {
                        let distance = rigid::body_signed_distance(b, world).abs();
                        if distance <= nearest {
                            nearest = distance;
                            speeds[i][a] = velocity(b, world)[a];
                        }
                    }
                }
            }
        }
        g.rigid_faces = Some(faces);
        g.rigid_speeds = Some(speeds);
        g.rigid_centres = Some(centres);
    }
    pub fn advance(&mut self, g: &Grid, dt: f32) -> Result<(), ValidationError> {
        if self.bodies.is_empty() {
            return Ok(());
        }
        let mut exchange = vec![FluidExchange::default(); self.bodies.len()];
        let cell_volume = g.h[0] * g.h[1] * self.depth;
        for i in 0..g.volume.len() {
            let p = g.point(i);
            let at = [p[0] as f32 + 0.5, p[1] as f32 + 0.5];
            let mut owner = None;
            let mut fraction = 0.0;
            for (j, b) in self.bodies.iter().enumerate() {
                let f = self.fraction(g, b, at);
                if f > fraction {
                    fraction = f;
                    owner = Some(j);
                }
            }
            let Some(j) = owner else { continue };
            let b = &self.bodies[j];
            let world = point(g, at);
            let wet = (0.5 - g.phi_at(at) / (4.0 * g.h[1])).clamp(0.0, 1.0);
            let mass = self.density * cell_volume * wet;
            let solid_v = velocity(b, world);
            let fluid_v = g.velocity[i];
            let blend = (45.0 * dt).clamp(0.0, 1.0);
            let reaction = [
                -mass * fraction * (solid_v[0] - fluid_v[0]) * blend,
                -mass * fraction * (solid_v[1] - fluid_v[1]) * blend,
            ];
            let torque = (world.x - b.position_m.x) as f32 * reaction[1]
                - (world.y - b.position_m.y) as f32 * reaction[0];
            let displaced = wet * fraction;
            let lanes = &mut exchange[j].lanes;
            let values = [
                reaction[0] * 1e6,
                reaction[1] * 1e6,
                0.0,
                0.0,
                0.0,
                torque * 1e6,
                displaced * 65536.0,
                displaced * fluid_v[0] * 1e4,
                displaced * fluid_v[1] * 1e4,
                0.0,
                0.0,
                0.0,
            ];
            for k in 0..12 {
                lanes[k] = lanes[k].wrapping_add(rigid::round_ties_even(values[k] as f64));
            }
        }
        rigid::advance_rigid_bodies_with_contacts(
            &mut self.bodies,
            &exchange,
            dt,
            self.density,
            self.gravity,
            cell_volume,
            1,
            Some(&self.contacts),
        )?;
        for b in &mut self.bodies {
            b.position_m.z = 0.0;
            b.linear_velocity_m_s.z = 0.0;
        }
        Ok(())
    }
}
