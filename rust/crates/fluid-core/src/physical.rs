//! Owned moving-body and source state. Geometry and source accounting are
//! advanced before projection; reaction loads are consumed one frame later.
use crate::dynamic::{
    compute_dynamic_geometry, plan_dynamic_remap, DynamicGeometry, DynamicGeometryInput,
    DynamicRemapReceipt,
};
use crate::rigid::{
    advance_rigid_bodies_with_contacts, initialize_body, rigid_coupling_loads, FluidExchange,
    RigidBodyState, RigidCouplingReceipt, StaticContactProvider,
};
use crate::scene::SceneState;
use crate::scene_model::{PhysicalScene, SceneModelError, SliceFrame};
use crate::solid_world::CompiledStaticSolidWorld;
use crate::sources::SourceLedger;
use crate::{Fields, Graph, ValidationError};

pub struct PhysicalContext {
    pub scene: PhysicalScene,
    pub frame: SliceFrame,
    pub solid_world: Option<CompiledStaticSolidWorld>,
    pub cell_volume_m3: f64,
    pub bodies: Vec<RigidBodyState>,
    pub previous_bodies: Vec<RigidBodyState>,
    pub exchange: Vec<FluidExchange>,
    pub coupling_receipts: Vec<RigidCouplingReceipt>,
    pub remap_receipt: DynamicRemapReceipt,
}
impl PhysicalContext {
    pub fn new(
        scene: PhysicalScene,
        frame: SliceFrame,
        solid_world: Option<CompiledStaticSolidWorld>,
        cell_volume_m3: f64,
    ) -> Result<Self, ValidationError> {
        let bodies = scene
            .rigid_bodies
            .iter()
            .take(12)
            .cloned()
            .map(initialize_body)
            .collect::<Result<Vec<_>, _>>()?;
        let exchange = vec![FluidExchange::default(); bodies.len()];
        Ok(Self {
            scene,
            frame,
            solid_world,
            cell_volume_m3,
            previous_bodies: bodies.clone(),
            bodies,
            exchange,
            coupling_receipts: Vec::new(),
            remap_receipt: DynamicRemapReceipt::default(),
        })
    }
    pub fn begin_frame(
        &mut self,
        state: &mut SceneState<2>,
        time: f64,
        dt: f64,
        ledger: &mut SourceLedger,
    ) -> Result<[f32; 3], ValidationError> {
        state.fields.acceleration_fine = [
            (self.scene.fluid.gravity_m_s2.x / self.frame.source_cell_size) as f32,
            (self.scene.fluid.gravity_m_s2.y / self.frame.source_cell_size) as f32,
            0.0,
        ];
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
                .map(|w| w as &dyn StaticContactProvider),
        )?;
        self.exchange.fill(FluidExchange::default());
        let groups: Vec<Vec<u32>> = state
            .topology
            .bricks
            .iter()
            .map(|b| b.cell_range.clone().collect())
            .collect();
        let geometry = compute_dynamic_geometry(&DynamicGeometryInput {
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
        Ok([
            geometry.inflow_velocity_fine[0],
            geometry.inflow_velocity_fine[1],
            0.0,
        ])
    }
    pub fn candidate_geometry(
        &self,
        state: &mut SceneState<2>,
        time: f64,
        dt: f64,
    ) -> Result<(), ValidationError> {
        let geometry = compute_dynamic_geometry(&DynamicGeometryInput {
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
        })?;
        apply_geometry(
            &mut state.topology.graph,
            &mut state.fields,
            &geometry,
            !self.scene.rigid_bodies.is_empty(),
        );
        Ok(())
    }
    pub fn finish_frame(&mut self, state: &SceneState<2>) -> Result<(), ValidationError> {
        let result = rigid_coupling_loads(
            self.frame,
            &state.topology.graph,
            &state.fields,
            &self.bodies,
            self.scene.fluid.density_kg_m3,
            self.scene.fluid.gravity_m_s2,
        )?;
        self.exchange = result.exchange;
        self.coupling_receipts = result.receipts;
        Ok(())
    }
}
fn apply_geometry(
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
pub fn publish_final_apertures(graph: &mut Graph, fields: &mut Fields) {
    if !fields.solid_motion_active {
        return;
    }
    for row in &mut graph.rows {
        let old = row.open_fraction;
        let new = row.open_fraction_after.unwrap_or(old);
        let wall = row.solid_velocity;
        let stored = fields.face_velocity[row.id as usize];
        let fluid = if old > 0.0 {
            (stored - ((1.0 - old) * wall)) / old
        } else {
            0.0
        };
        fields.face_velocity[row.id as usize] = (new * fluid) + ((1.0 - new) * wall);
        row.open_fraction = new;
    }
    for face in &mut graph.subfaces {
        face.aperture = graph.rows[face.row_id as usize].open_fraction;
    }
}
impl From<SceneModelError> for ValidationError {
    fn from(e: SceneModelError) -> Self {
        Self(e.to_string())
    }
}
