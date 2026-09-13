//! Owned full-volume moving-body and source state for the Rust 3-D world.
use crate::dynamic::{plan_dynamic_remap, DynamicGeometry, DynamicRemapReceipt};
use crate::dynamic3d::{compute_dynamic_geometry_3d, DynamicGeometry3dInput, VolumeFrame};
use crate::rigid::{
    advance_rigid_bodies_with_contacts, body_contains, bounding_radius, initialize_body,
    primitive_volume, round_ties_even, FluidExchange, RigidBodyState, RigidCouplingReceipt,
};
use crate::scene::SceneState;
use crate::scene_model::{PhysicalScene, SceneModelError, Vec3};
use crate::solid_world::CompiledStaticSolidWorld;
use crate::sources::SourceLedger;
use crate::{Fields, Graph, ValidationError};

pub struct PhysicalContext3d {
    pub scene: PhysicalScene,
    pub frame: VolumeFrame,
    pub solid_world: Option<CompiledStaticSolidWorld>,
    pub cell_volume_m3: f64,
    pub bodies: Vec<RigidBodyState>,
    pub previous_bodies: Vec<RigidBodyState>,
    pub exchange: Vec<FluidExchange>,
    pub coupling_receipts: Vec<RigidCouplingReceipt>,
    pub remap_receipt: DynamicRemapReceipt,
    pub constraint_motion_pending: bool,
}

impl PhysicalContext3d {
    pub fn new(
        scene: PhysicalScene,
        frame: VolumeFrame,
        solid_world: Option<CompiledStaticSolidWorld>,
    ) -> Result<Self, ValidationError> {
        let bodies = scene
            .rigid_bodies
            .iter()
            .take(12)
            .cloned()
            .map(initialize_body)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            scene,
            frame,
            solid_world,
            cell_volume_m3: frame.cell_size_m.powi(3),
            previous_bodies: bodies.clone(),
            exchange: vec![FluidExchange::default(); bodies.len()],
            bodies,
            coupling_receipts: vec![],
            remap_receipt: DynamicRemapReceipt::default(),
            constraint_motion_pending: false,
        })
    }

    pub fn begin_frame(
        &mut self,
        state: &mut SceneState<3>,
        time: f64,
        dt: f64,
        ledger: &mut SourceLedger,
    ) -> Result<[f32; 3], ValidationError> {
        state.fields.acceleration_fine = self
            .scene
            .fluid
            .gravity_m_s2
            .array()
            .map(|v| (v / self.frame.cell_size_m) as f32);
        let constrained_previous = self
            .constraint_motion_pending
            .then(|| self.previous_bodies.clone());
        self.previous_bodies = advance_rigid_bodies_with_contacts(
            &mut self.bodies,
            &self.exchange,
            dt as f32,
            self.scene.fluid.density_kg_m3 as f32,
            self.scene.fluid.gravity_m_s2,
            self.cell_volume_m3 as f32,
            1,
            self.solid_world
                .as_ref()
                .map(|w| w as &dyn crate::rigid::StaticContactProvider),
        )?;
        if let Some(previous) = constrained_previous {
            self.previous_bodies = previous;
        }
        self.constraint_motion_pending = false;
        self.exchange.fill(FluidExchange::default());
        let groups = state
            .topology
            .bricks
            .iter()
            .map(|b| b.cell_range.clone().collect::<Vec<u32>>())
            .collect::<Vec<_>>();
        let geometry3d = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &self.scene,
            frame: self.frame,
            graph: &state.topology.graph,
            time_s: time,
            dt_s: dt,
            bodies: &self.bodies,
            previous_bodies: &self.previous_bodies,
            density: Some(&state.fields.density),
            pending_source_area_fine: ledger.pending,
            pending_source_compensation: ledger.pending_compensation,
            pressure_member: Some(&state.fields.pressure_member),
            source_reduction_groups: Some(&groups),
        })?;
        let inflow = geometry3d.inflow_velocity_fine;
        let geometry: DynamicGeometry = geometry3d.into();
        let remap = plan_dynamic_remap(
            &state.topology.graph,
            &state.fields.density,
            &geometry,
            dt,
            ledger.clone(),
        )?;
        state.fields.density = remap.density;
        self.remap_receipt = remap.receipt;
        *ledger = remap.ledger;
        apply_geometry(
            &mut state.topology.graph,
            &mut state.fields,
            &geometry,
            !self.scene.rigid_bodies.is_empty(),
        );
        Ok(inflow)
    }

    pub fn candidate_geometry(
        &self,
        state: &mut SceneState<3>,
        time: f64,
        dt: f64,
    ) -> Result<(), ValidationError> {
        let geometry: DynamicGeometry = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &self.scene,
            frame: self.frame,
            graph: &state.topology.graph,
            time_s: time,
            dt_s: dt,
            bodies: &self.bodies,
            previous_bodies: &self.bodies,
            density: None,
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: None,
            source_reduction_groups: None,
        })?
        .into();
        apply_geometry(
            &mut state.topology.graph,
            &mut state.fields,
            &geometry,
            !self.scene.rigid_bodies.is_empty(),
        );
        Ok(())
    }

    pub fn finish_frame(&mut self, state: &SceneState<3>) -> Result<(), ValidationError> {
        let (exchange, receipts) = coupling_loads_3d(
            self.frame,
            &state.topology.graph,
            &state.fields,
            &self.bodies,
            self.scene.fluid.density_kg_m3,
            self.scene.fluid.gravity_m_s2,
        )?;
        self.exchange = exchange;
        self.coupling_receipts = receipts;
        Ok(())
    }
}

pub(crate) fn apply_geometry(
    graph: &mut Graph,
    fields: &mut Fields,
    geometry: &DynamicGeometry,
    moving: bool,
) {
    fields.capacity.clone_from(&geometry.capacity);
    fields.capacity_before.clone_from(&geometry.capacity_before);
    fields.capacity_after.clone_from(&geometry.capacity_after);
    fields.capacity_rate.clone_from(&geometry.capacity_rate);
    fields.source_rate.clone_from(&geometry.source_rate);
    fields.inflow_coverage.clone_from(&geometry.inflow_coverage);
    fields.solid_motion_active = moving;
    for row in &mut graph.rows {
        let i = row.id as usize;
        row.open_fraction = geometry.open_fraction[i];
        row.open_fraction_before = Some(geometry.open_fraction_before[i]);
        row.open_fraction_after = Some(geometry.open_fraction_after[i]);
        row.solid_velocity = geometry.solid_velocity[i];
    }
    for face in &mut graph.subfaces {
        let row = &graph.rows[face.row_id as usize];
        face.aperture = row.open_fraction;
        face.solid_velocity = row.solid_velocity;
    }
}

fn coupling_loads_3d(
    frame: VolumeFrame,
    graph: &Graph,
    fields: &Fields,
    bodies: &[RigidBodyState],
    density: f64,
    gravity: Vec3,
) -> Result<(Vec<FluidExchange>, Vec<RigidCouplingReceipt>), SceneModelError> {
    let count = bodies.len().min(12);
    let h = frame.cell_size_m;
    let mut exchange = vec![FluidExchange::default(); count];
    for cell in &graph.cells {
        let mut best = 0.0;
        let mut owner = None;
        for (bi, body) in bodies.iter().take(count).enumerate() {
            let mut covered = 0u32;
            let mut samples = 0u32;
            for z in cell.minimum[2] as i32..cell.maximum[2] as i32 {
                for y in cell.minimum[1] as i32..cell.maximum[1] as i32 {
                    for x in cell.minimum[0] as i32..cell.maximum[0] as i32 {
                        for dz in [-0.4, 0.4] {
                            for dy in [-0.4, 0.4] {
                                for dx in [-0.4, 0.4] {
                                    let p = Vec3 {
                                        x: frame.origin_m[0] + (x as f64 + 0.5 + dx) * h,
                                        y: frame.origin_m[1] + (y as f64 + 0.5 + dy) * h,
                                        z: frame.origin_m[2] + (z as f64 + 0.5 + dz) * h,
                                    };
                                    covered += body_contains(body, p)? as u32;
                                    samples += 1;
                                }
                            }
                        }
                    }
                }
            }
            let coverage = if samples > 0 {
                covered as f64 / samples as f64
            } else {
                0.0
            };
            if coverage > best {
                best = coverage;
                owner = Some(bi)
            }
        }
        let Some(owner) = owner else { continue };
        if best <= 0.0 {
            continue;
        }
        let id = cell.id as usize;
        let open = fields.capacity[id] as f64;
        let wet = (fields.density[id] as f64 / open.max(0.125)).clamp(0.0, 1.0);
        if wet <= 0.0 {
            continue;
        }
        let weight = wet * best * cell.measure as f64;
        let e = &mut exchange[owner].lanes;
        let q = round_ties_even(weight * 65536.0);
        e[6] += q;
        e[11] += q;
        for axis in 0..3 {
            e[7 + axis] +=
                round_ties_even(weight * fields.cell_velocity[3 * id + axis] as f64 * h * 10000.0);
        }
    }
    let mut receipts = Vec::with_capacity(count);
    for (i, body) in bodies.iter().take(count).enumerate() {
        let e = exchange[i].lanes;
        let displaced = (e[6] as f64 / 65536.0 * h.powi(3)).clamp(
            0.0,
            primitive_volume(body.description.shape, body.description.dimensions_m)?,
        );
        let weight = e[11] as f64 / 65536.0;
        let mean = if weight > 1e-8 {
            Vec3 {
                x: e[7] as f64 * 1e-4 / weight,
                y: e[8] as f64 * 1e-4 / weight,
                z: e[9] as f64 * 1e-4 / weight,
            }
        } else {
            Vec3::default()
        };
        let relative = Vec3 {
            x: body.linear_velocity_m_s.x - mean.x,
            y: body.linear_velocity_m_s.y - mean.y,
            z: body.linear_velocity_m_s.z - mean.z,
        };
        let speed = relative.x.hypot(relative.y).hypot(relative.z);
        let volume = primitive_volume(body.description.shape, body.description.dimensions_m)?;
        let immersed = (displaced / volume.max(1e-9)).clamp(0.0, 1.0);
        let radius = bounding_radius(&body.description)? as f64;
        let drag_scale =
            -0.5 * density * 0.9 * std::f64::consts::PI * radius * radius * immersed * speed;
        let drag = Vec3 {
            x: drag_scale * relative.x,
            y: drag_scale * relative.y,
            z: drag_scale * relative.z,
        };
        let buoyancy = Vec3 {
            x: -density * displaced * gravity.x,
            y: -density * displaced * gravity.y,
            z: -density * displaced * gravity.z,
        };
        let denominator = (body.mass_kg + 0.5 * density * displaced).max(1e-8);
        let desired = Vec3 {
            x: (body.mass_kg * gravity.x + drag.x + buoyancy.x) / denominator,
            y: (body.mass_kg * gravity.y + drag.y + buoyancy.y) / denominator,
            z: (body.mass_kg * gravity.z + drag.z + buoyancy.z) / denominator,
        };
        let force = Vec3 {
            x: body.mass_kg * (desired.x - gravity.x),
            y: body.mass_kg * (desired.y - gravity.y),
            z: body.mass_kg * (desired.z - gravity.z),
        };
        receipts.push(RigidCouplingReceipt {
            body_id: body.description.id.clone(),
            displaced_volume_m3: displaced,
            mean_fluid_velocity_m_s: mean,
            force_n: force,
        });
    }
    Ok((exchange, receipts))
}
