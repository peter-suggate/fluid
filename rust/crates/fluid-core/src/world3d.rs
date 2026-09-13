//! Owned 3D numerical world. Rendering receives immutable publication planes;
//! it never supplies integrated rigid poses or mutable physics state.
use crate::dynamic::{plan_dynamic_remap, DynamicGeometry};
use crate::dynamic3d::{compute_dynamic_geometry_3d, DynamicGeometry3dInput};
use crate::initial_scene::SceneDocument;
use crate::lifecycle3d::{prepare_candidate_3d, LeafArena};
use crate::physical3d::{apply_geometry, PhysicalContext3d};
use crate::presentation3d::Rdf3d;
use crate::pressure_authority::PressureAuthority;
use crate::production_scene::ProductionSceneOptions;
use crate::production_scene3d::production_scene_3d;
use crate::publication::{encode_publication, Plane, PlaneId};
use crate::resolution3d::{
    initialize_resolution_policy_3d, plan_projected_transport_support_3d, plan_resolution_3d,
    ResolutionPolicyOptions3d, ResolutionPolicyReceipt3d, ResolutionPolicyState3d,
    ResolutionRegion3d,
};
use crate::rigid::{initialize_body, set_rigid_body_kinematics, RigidBodyState};
use crate::scalar_authority::{
    publish_scalar_interface_state_from_geometric_density, ScalarAuthority,
};
use crate::scene::SceneState;
use crate::scene_model::{Quaternion, RigidBodyDescription, Vec3};
use crate::sources::SourceLedger;
use crate::topology::BrickSeed;
use crate::tracers::{TracerReceipt, Tracers};
use crate::world::{Revision, WorldOptions};
use crate::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Command3d {
    Snapshot,
    SetTimeStep {
        #[serde(rename = "dt_s")]
        dt_s: f64,
    },
    SetPressureBudget {
        iterations: u32,
        #[serde(rename = "relativeTolerance")]
        relative_tolerance: f32,
    },
    SetTracers {
        enabled: bool,
    },
    ReseedTracers,
    SetTopologyFrozen {
        frozen: bool,
    },
    SetScene {
        scene: SceneDocument,
    },
    SetRigidBodies {
        bodies: Vec<RigidBodyDescription>,
    },
    SetRigidConstraints {
        constraints: Vec<RigidConstraint3d>,
    },
    InjectLiquid {
        drop: crate::injection3d::LiquidDrop3d,
    },
    SetRuntimeValues {
        values: serde_json::Value,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RigidConstraint3d {
    pub id: String,
    pub held: bool,
    #[serde(rename = "position_m")]
    pub position_m: Vec3,
    pub orientation: Quaternion,
    #[serde(rename = "linearVelocity_m_s")]
    pub linear_velocity_m_s: Vec3,
    #[serde(rename = "angularVelocity_rad_s")]
    pub angular_velocity_rad_s: Vec3,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt3d<'a> {
    #[serde(flatten)]
    pub revision: &'a Revision,
    pub pressure: &'a PressureReceipt,
    pub primary_refinement: &'a PressureReceipt,
    pub coupled_pressure: &'a crate::closed_air_pressure::ClosedAirPressureReceipt,
    pub pressure_total_iterations: u32,
    pub microsteps: usize,
    pub compatibility: &'a crate::compatibility3d::GeometricCompatibilityReceipt3d,
    pub source_ledger: &'a SourceLedger,
    pub tracers: TracerReceipt,
    pub liquid_measure: f64,
    pub seeded_volume: f64,
    pub drift: f64,
    pub fault: &'a Option<NumericalFault>,
    pub last_injection: &'a Option<crate::injection3d::InjectionReceipt3d>,
}

pub struct World3d {
    pub state: SceneState<3>,
    pub physical: PhysicalContext3d,
    pub document: SceneDocument,
    pub production_options: ProductionSceneOptions,
    pub revision: Revision,
    pub options: WorldOptions,
    pub pressure: PressureReceipt,
    pub primary_refinement: PressureReceipt,
    pub coupled_pressure: crate::closed_air_pressure::ClosedAirPressureReceipt,
    pub pressure_authority: PressureAuthority,
    pub scalar_authority: ScalarAuthority,
    pub source_ledger: SourceLedger,
    pub tracers: Tracers,
    pub tracer_receipt: TracerReceipt,
    pub microsteps: usize,
    pub compatibility: crate::compatibility3d::GeometricCompatibilityReceipt3d,
    pub seeded_volume: f64,
    pub timestep_s: f64,
    pub topology_frozen: bool,
    pub topology_slot: u8,
    pub arena: LeafArena,
    pub resolution_options: ResolutionPolicyOptions3d,
    pub resolution_policy: ResolutionPolicyState3d,
    pub resolution_receipt: Option<ResolutionPolicyReceipt3d>,
    pub last_injection: Option<crate::injection3d::InjectionReceipt3d>,
    rdf: Rdf3d,
    density_view: Vec<f32>,
    phi_view: Vec<f32>,
    velocity_view: Vec<f32>,
    publication: Vec<u8>,
}
impl World3d {
    pub fn from_document(
        document: SceneDocument,
        production: ProductionSceneOptions,
        options: WorldOptions,
    ) -> Result<Self, ValidationError> {
        if options.run_epoch == 0 {
            return Err(ValidationError("runEpoch must be positive".into()));
        }
        validate_pressure(
            options.pressure_iterations,
            options.pressure_relative_tolerance,
        )?;
        let bundle = production_scene_3d(document, production)
            .map_err(|e| ValidationError(e.to_string()))?;
        let state = bundle.state;
        let graph = &state.topology.graph;
        let pressure_authority = PressureAuthority::new(graph, None, None);
        let arena = LeafArena::new_3d(&state.topology, options.topology_page_budget)?;
        let capacity = arena.capacity as usize;
        let resolution_policy = initialize_resolution_policy_3d(&state.topology);
        let mut resolution_options =
            resolution_options_for_document(&bundle.document, &bundle.options)?;
        resolution_options.static_boundary_floor_by_brick =
            bundle.resolution_options_3d.static_boundary_floor_by_brick;
        let scalar_authority =
            ScalarAuthority::new(capacity, graph.cells.len(), graph.topology_generation, 0);
        let revision = Revision {
            schema_version: 1,
            dimension: 3,
            run_epoch: options.run_epoch,
            command_sequence: options.command_sequence,
            frame: 0,
            time: 0.0,
            injections: 0,
            topology_generation: graph.topology_generation,
            field_revision: 1,
            surface_revision: 1,
            memory_epoch: 1,
        };
        let rdf = Rdf3d::reconstruct(&state.topology, &state.fields)?;
        let tracers = Tracers::new(state.description.dimensions, options.tracer_budget);
        let seeded_volume = measure(&state);
        let dimensions = state.description.dimensions;
        let count = dimensions
            .into_iter()
            .try_fold(1usize, |n, d| n.checked_mul(d as usize))
            .ok_or_else(|| ValidationError("3D presentation dimensions overflow".into()))?;
        if count > usize::MAX / 24 {
            return Err(ValidationError(
                "3D presentation allocation overflow".into(),
            ));
        }
        Ok(Self {
            state,
            physical: bundle.physical,
            document: bundle.document,
            production_options: bundle.options,
            revision,
            options,
            pressure: PressureReceipt::default(),
            primary_refinement: PressureReceipt::default(),
            coupled_pressure: Default::default(),
            pressure_authority,
            scalar_authority,
            source_ledger: SourceLedger::default(),
            tracers,
            tracer_receipt: TracerReceipt::default(),
            microsteps: 1,
            compatibility: Default::default(),
            seeded_volume,
            timestep_s: bundle.dt_s,
            topology_frozen: false,
            topology_slot: 0,
            arena,
            resolution_options,
            resolution_policy,
            resolution_receipt: None,
            last_injection: None,
            rdf,
            density_view: vec![0.0; count],
            phi_view: vec![0.0; count],
            velocity_view: vec![0.0; 4 * count],
            publication: vec![],
        })
    }
    pub(crate) fn transition(
        &mut self,
        mut bricks: Vec<BrickSeed>,
        dt: f64,
    ) -> Result<(), ValidationError> {
        let mut description = self.state.description.clone();
        // Accepted source planes, rather than authored seed payloads, own all
        // material after construction. The transfer fills candidate fields.
        for b in &mut bricks {
            b.density.clear();
            b.gamma.clear();
        }
        description.bricks = bricks;
        let mut candidate = crate::scene::compile_scene_3d(description)
            .map_err(|e| ValidationError(e.to_string()))?;
        candidate.topology.graph.topology_generation =
            self.state.topology.graph.topology_generation + 1;
        candidate
            .topology
            .graph
            .solid_voxel_fraction
            .clone_from(&self.state.topology.graph.solid_voxel_fraction);
        self.physical
            .candidate_geometry(&mut candidate, self.revision.time, dt)?;
        let (candidate, arena) = prepare_candidate_3d(&self.state, candidate, &self.arena)
            .map_err(|e| ValidationError(e.to_string()))?;
        let rdf = Rdf3d::reconstruct(&candidate.topology, &candidate.fields)?;
        let static_floors = crate::resolution3d::static_boundary_floors_3d(
            &candidate.topology,
            self.resolution_options.policy.detail_tolerance as f32,
        )
        .map_err(|e| ValidationError(format!("3D static boundary policy: {e:?}")))?;
        self.pressure_authority
            .resize_preserving_cache(&candidate.topology.graph, None, None);
        self.scalar_authority = ScalarAuthority::new(
            arena.capacity as usize,
            candidate.topology.graph.cells.len(),
            candidate.topology.graph.topology_generation,
            self.topology_slot ^ 1,
        );
        self.revision.topology_generation = candidate.topology.graph.topology_generation;
        self.topology_slot ^= 1;
        self.rdf = rdf;
        self.state = candidate;
        self.resolution_options.static_boundary_floor_by_brick = static_floors;
        self.arena = arena;
        Ok(())
    }
    fn check_sequence(&self, sequence: u32) -> Result<(), ValidationError> {
        if sequence <= self.revision.command_sequence {
            return Err(ValidationError("commandSequence must increase".into()));
        }
        Ok(())
    }
    pub fn receipt(&self) -> Receipt3d<'_> {
        let liquid = measure(&self.state);
        let expected = self.seeded_volume + self.source_ledger.emitted as f64;
        Receipt3d {
            revision: &self.revision,
            pressure: &self.pressure,
            primary_refinement: &self.primary_refinement,
            coupled_pressure: &self.coupled_pressure,
            pressure_total_iterations: self.pressure.iterations
                + self.primary_refinement.iterations
                + self.coupled_pressure.f64_refinement_iterations,
            microsteps: self.microsteps,
            compatibility: &self.compatibility,
            source_ledger: &self.source_ledger,
            tracers: self.tracer_receipt,
            liquid_measure: liquid,
            seeded_volume: self.seeded_volume,
            drift: if expected > 0.0 {
                (liquid - expected) / expected
            } else {
                0.0
            },
            fault: &self.state.fields.fault,
            last_injection: &self.last_injection,
        }
    }
    pub fn apply_command(
        &mut self,
        sequence: u32,
        epoch: u32,
        command: Command3d,
    ) -> Result<(), ValidationError> {
        self.check_sequence(sequence)?;
        if epoch != self.revision.run_epoch {
            return Err(ValidationError("stale runEpoch".into()));
        }
        match command {
            Command3d::Snapshot => {}
            Command3d::SetTimeStep { dt_s } => {
                validate_dt(dt_s)?;
                self.timestep_s = dt_s;
                self.state.fields.subface_compatibility_rate.clear();
            }
            Command3d::SetPressureBudget {
                iterations,
                relative_tolerance,
            } => {
                validate_pressure(iterations, relative_tolerance)?;
                self.options.pressure_iterations = iterations;
                self.options.pressure_relative_tolerance = relative_tolerance;
            }
            Command3d::SetTracers { enabled } => self.tracers.set_enabled(enabled),
            Command3d::ReseedTracers => self.tracers.reseed(),
            Command3d::SetTopologyFrozen { frozen } => self.topology_frozen = frozen,
            Command3d::SetScene { scene } => self.replace_document(scene)?,
            Command3d::SetRigidBodies { bodies } => {
                let mut scene = self.document.clone();
                scene.rigid_bodies = bodies;
                self.replace_document(scene)?;
            }
            Command3d::SetRigidConstraints { constraints } => {
                self.apply_rigid_constraints(&constraints)?
            }
            Command3d::InjectLiquid { drop } => self.inject_liquid(drop)?,
            Command3d::SetRuntimeValues { values } => self.apply_runtime_values(&values)?,
        }
        self.revision.command_sequence = sequence;
        Ok(())
    }
    fn apply_rigid_constraints(
        &mut self,
        constraints: &[RigidConstraint3d],
    ) -> Result<(), ValidationError> {
        if constraints.is_empty() {
            return Ok(());
        }
        let finite_vec = |v: Vec3| v.array().into_iter().all(f64::is_finite);
        let mut seen = std::collections::BTreeSet::new();
        for constraint in constraints {
            if constraint.id.is_empty() || !seen.insert(constraint.id.as_str()) {
                return Err(ValidationError(
                    "rigid constraints require unique nonempty ids".into(),
                ));
            }
            if !self
                .physical
                .bodies
                .iter()
                .any(|body| body.description.id == constraint.id)
            {
                return Err(ValidationError(format!(
                    "rigid constraint references unknown body {}",
                    constraint.id
                )));
            }
            let q = constraint.orientation;
            let q2 = (q.w * q.w + q.x * q.x) + q.y * q.y + q.z * q.z;
            if !finite_vec(constraint.position_m)
                || !finite_vec(constraint.linear_velocity_m_s)
                || !finite_vec(constraint.angular_velocity_rad_s)
                || !q2.is_finite()
                || q2 <= f32::MIN_POSITIVE as f64
            {
                return Err(ValidationError(
                    "rigid constraint pose and velocities must be finite".into(),
                ));
            }
        }

        let old_bodies = self.physical.bodies.clone();
        let mut next_bodies = old_bodies.clone();
        for constraint in constraints {
            let body = next_bodies
                .iter_mut()
                .find(|body| body.description.id == constraint.id)
                .expect("validated rigid constraint id");
            set_rigid_body_kinematics(
                body,
                constraint.position_m,
                constraint.orientation,
                constraint.linear_velocity_m_s,
                constraint.angular_velocity_rad_s,
                constraint.held,
            );
        }
        let graph = &self.state.topology.graph;
        let old_geometry: DynamicGeometry = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &self.physical.scene,
            frame: self.physical.frame,
            graph,
            time_s: self.revision.time,
            dt_s: self.timestep_s,
            bodies: &old_bodies,
            previous_bodies: &old_bodies,
            density: Some(&self.state.fields.density),
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: Some(&self.state.fields.pressure_member),
            source_reduction_groups: None,
        })?
        .into();
        let new_geometry: DynamicGeometry = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &self.physical.scene,
            frame: self.physical.frame,
            graph,
            time_s: self.revision.time,
            dt_s: self.timestep_s,
            bodies: &next_bodies,
            previous_bodies: &next_bodies,
            density: Some(&self.state.fields.density),
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: Some(&self.state.fields.pressure_member),
            source_reduction_groups: None,
        })?
        .into();
        let mut geometry = edited_geometry(old_geometry, new_geometry, self.timestep_s);
        // A paused pose edit cannot admit a source event. The next frame owns
        // source planning and freezes that budget in the normal begin-frame path.
        geometry.source_rate.fill(0.0);
        geometry.requested_source_area_fine = 0.0;
        geometry.source_rate_area_fine = 0.0;
        geometry.source_available_area_fine = 0.0;
        geometry.source_factor = 0.0;
        let remap = plan_dynamic_remap(
            graph,
            &self.state.fields.density,
            &geometry,
            self.timestep_s,
            self.source_ledger.clone(),
        )?;
        let mut next_graph = graph.clone();
        let mut next_fields = self.state.fields.clone();
        next_fields.density = remap.density;
        next_fields.subface_compatibility_rate.clear();
        apply_geometry(&mut next_graph, &mut next_fields, &geometry, true);
        reconstruct_interfaces(&mut next_graph, &mut next_fields)?;
        let mut next_topology = self.state.topology.clone();
        next_topology.graph = next_graph;
        let rdf = Rdf3d::reconstruct(&next_topology, &next_fields)?;

        self.state.topology = next_topology;
        self.state.fields = next_fields;
        self.physical.previous_bodies = old_bodies;
        self.physical.bodies = next_bodies;
        self.physical.remap_receipt = remap.receipt;
        self.physical.constraint_motion_pending = true;
        self.rdf = rdf;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        Ok(())
    }
    fn replace_document(&mut self, document: SceneDocument) -> Result<(), ValidationError> {
        let candidate = production_scene_3d(document.clone(), self.production_options.clone())
            .map_err(|e| ValidationError(e.to_string()))?;
        if candidate.state.description.dimensions != self.state.description.dimensions
            || candidate.physical.frame != self.physical.frame
        {
            return Err(ValidationError(
                "live scene edit changes the 3D simulation lattice; rebuild the solver".into(),
            ));
        }
        let next_production_options = candidate.options.clone();
        let next_resolution_options =
            resolution_options_for_document(&document, &next_production_options)?;

        let old_graph = self.state.topology.graph.clone();
        let old_bodies = self.physical.bodies.clone();
        let old_exchange = self.physical.exchange.clone();
        let mut next_graph = old_graph.clone();
        next_graph.solid_voxel_fraction =
            candidate.state.topology.graph.solid_voxel_fraction.clone();
        let mut next_physical = candidate.physical;
        next_physical.bodies = reconcile_bodies(
            &self.document.rigid_bodies,
            &old_bodies,
            &document.rigid_bodies,
        )?;
        next_physical.previous_bodies = old_bodies.clone();
        next_physical.exchange = next_physical
            .bodies
            .iter()
            .map(|body| {
                old_bodies
                    .iter()
                    .position(|old| old.description.id == body.description.id)
                    .and_then(|i| old_exchange.get(i).copied())
                    .unwrap_or_default()
            })
            .collect();

        let old_geometry: DynamicGeometry = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &self.physical.scene,
            frame: self.physical.frame,
            graph: &old_graph,
            time_s: self.revision.time,
            dt_s: self.timestep_s,
            bodies: &old_bodies,
            previous_bodies: &old_bodies,
            density: Some(&self.state.fields.density),
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: Some(&self.state.fields.pressure_member),
            source_reduction_groups: None,
        })?
        .into();
        let new_geometry: DynamicGeometry = compute_dynamic_geometry_3d(&DynamicGeometry3dInput {
            scene: &next_physical.scene,
            frame: next_physical.frame,
            graph: &next_graph,
            time_s: self.revision.time,
            dt_s: self.timestep_s,
            bodies: &next_physical.bodies,
            previous_bodies: &next_physical.bodies,
            density: Some(&self.state.fields.density),
            pending_source_area_fine: 0.0,
            pending_source_compensation: 0.0,
            pressure_member: Some(&self.state.fields.pressure_member),
            source_reduction_groups: None,
        })?
        .into();
        let geometry = edited_geometry(old_geometry, new_geometry, self.timestep_s);
        let remap = plan_dynamic_remap(
            &next_graph,
            &self.state.fields.density,
            &geometry,
            self.timestep_s,
            self.source_ledger.clone(),
        )?;

        self.state.topology.graph = next_graph;
        self.state.fields.density = remap.density;
        self.source_ledger = remap.ledger;
        next_physical.remap_receipt = remap.receipt;
        apply_geometry(
            &mut self.state.topology.graph,
            &mut self.state.fields,
            &geometry,
            !next_physical.bodies.is_empty(),
        );
        self.state.fields.subface_compatibility_rate.clear();
        self.physical = next_physical;
        self.document = document;
        self.production_options = next_production_options;
        self.resolution_options = next_resolution_options;
        self.resolution_options.static_boundary_floor_by_brick =
            crate::resolution3d::static_boundary_floors_3d(
                &self.state.topology,
                self.resolution_options.policy.detail_tolerance as f32,
            )
            .map_err(|e| ValidationError(format!("3D edited static policy: {e:?}")))?;
        reconstruct_interfaces(&mut self.state.topology.graph, &mut self.state.fields)?;
        self.rdf = Rdf3d::reconstruct(&self.state.topology, &self.state.fields)?;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        Ok(())
    }
    pub fn advance(&mut self, sequence: u32, dt_s: f64) -> Result<(), ValidationError> {
        self.check_sequence(sequence)?;
        validate_dt(dt_s)?;
        if self.state.fields.fault.is_some() {
            return Err(ValidationError(
                "3D world has a numerical fault; reset before advancing".into(),
            ));
        }
        let result = self.advance_frame(sequence, dt_s);
        if let Err(error) = &result {
            // A failed physical stage may have written scratch or live fields.
            // Latch the failure so callers cannot continue from a partial frame.
            self.state
                .fields
                .fault
                .get_or_insert_with(|| NumericalFault {
                    stage: format!("advance3d: {error}"),
                    index: 0,
                    observed: 1.0,
                    expected: 0.0,
                });
        }
        result
    }
    fn advance_frame(&mut self, sequence: u32, dt_s: f64) -> Result<(), ValidationError> {
        self.check_sequence(sequence)?;
        validate_dt(dt_s)?;
        if self.state.fields.fault.is_some() {
            return Err(ValidationError("3D world has a numerical fault".into()));
        }
        // This rate plane is authority only for the frozen topology, source,
        // pressure membership and geometry of one outer frame.
        self.state.fields.subface_compatibility_rate.clear();
        let dt = dt_s as f32;
        // Source receivers must exist before admission freezes the frame ledger.
        let source_policy = self.current_resolution_options(dt_s)?;
        if !source_policy.source_demand_bounds_fine.is_empty() {
            let support = plan_projected_transport_support_3d(
                &self.state.topology,
                &self.state.fields,
                dt_s,
                self.physical.frame.cell_size_m,
                &source_policy,
            )
            .map_err(|e| ValidationError(format!("3D source support: {e:?}")))?;
            if support.fault_bits != 0 {
                return Err(ValidationError(format!(
                    "3D source support fault {}",
                    support.fault_bits
                )));
            }
            if self.bricks_changed(&support.candidate_bricks) {
                self.transition(support.candidate_bricks, dt_s)?;
            }
        }
        let inflow = self.physical.begin_frame(
            &mut self.state,
            self.revision.time,
            dt_s,
            &mut self.source_ledger,
        )?;
        let graph = &mut self.state.topology.graph;
        let fields = &mut self.state.fields;
        fields.frame_dt = dt;
        extend_velocity(graph, fields, 8)?;
        prepare_faces(graph, fields, dt)?;
        force_faces(graph, fields, dt, fields.acceleration_fine, inflow);
        reconstruct_interfaces(graph, fields)?;
        let rows = prepare_pressure_topology(graph, fields);
        self.pressure_authority.publish(
            graph,
            fields,
            &rows.active,
            &rows.theta,
            graph.topology_generation,
            true,
        );
        assemble_pressure_rhs(graph, fields, &rows);
        self.pressure = solve_pressure(
            graph,
            fields,
            &rows,
            self.options.pressure_iterations,
            self.options.pressure_relative_tolerance,
            Some(&self.pressure_authority.execution_order),
        )?;
        project_pressure_velocity(graph, fields, &rows);
        enforce_inflow_faces(graph, fields, inflow);
        collocate_velocity(graph, fields);
        let policy = self.current_resolution_options(dt_s)?;
        let support = plan_projected_transport_support_3d(
            &self.state.topology,
            &self.state.fields,
            dt_s,
            self.physical.frame.cell_size_m,
            &policy,
        )
        .map_err(|e| ValidationError(format!("3D projected support: {e:?}")))?;
        if support.fault_bits != 0 {
            return Err(ValidationError(format!(
                "3D projected support fault {}",
                support.fault_bits
            )));
        }
        if self.bricks_changed(&support.candidate_bricks) {
            self.transition(support.candidate_bricks, dt_s)?;
            extend_velocity(&self.state.topology.graph, &mut self.state.fields, 8)?;
        }
        // The accepted transport topology may differ from the image on which
        // the user-budgeted solve ran. Rebuild and refine the primary liquid
        // projection before solving the complementary air compatibility
        // domain, matching the production post-transaction refinement seam.
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            let rows = prepare_pressure_topology(graph, fields);
            self.pressure_authority.publish(
                graph,
                fields,
                &rows.active,
                &rows.theta,
                graph.topology_generation,
                true,
            );
            if fields.solid_motion_active {
                // Moving geometry owns the capacity-change seam. Keep its
                // established primary refinement and leave the complementary
                // air solve disabled, just as the secondary compatibility
                // stage is disabled for this frame.
                assemble_pressure_rhs(graph, fields, &rows);
                self.primary_refinement = solve_pressure(
                    graph,
                    fields,
                    &rows,
                    64,
                    self.options.pressure_relative_tolerance,
                    Some(&self.pressure_authority.execution_order),
                )?;
                project_pressure_velocity(graph, fields, &rows);
                fields.subface_compatibility_rate.clear();
                self.coupled_pressure = crate::closed_air_pressure::ClosedAirPressureReceipt {
                    pressure: self.primary_refinement.clone(),
                    normalized_target: 2.0 * f32::EPSILON as f64,
                    skipped_for_solid_motion: true,
                    ..Default::default()
                };
            } else {
                self.coupled_pressure =
                    crate::closed_air_pressure::solve_closed_air_coupled_pressure(
                        graph,
                        fields,
                        &rows,
                        4096,
                        self.options.pressure_relative_tolerance,
                        Some(&self.pressure_authority.execution_order),
                    )?;
                self.primary_refinement = self.coupled_pressure.pressure.clone();
            }
            enforce_inflow_faces(graph, fields, inflow);
            if fields.subface_compatibility_rate.len() == graph.subfaces.len() {
                for face in &graph.subfaces {
                    let row = &graph.rows[face.row_id as usize];
                    if fields
                        .inflow_coverage
                        .get(row.id as usize)
                        .copied()
                        .unwrap_or(0.0)
                        > 0.0
                    {
                        let fluid_velocity = fields.face_velocity[row.id as usize]
                            - (1.0 - row.open_fraction) * row.solid_velocity;
                        fields.subface_compatibility_rate[face.id as usize] =
                            face.measure as f64 * fluid_velocity as f64;
                    }
                }
            }
            collocate_velocity(graph, fields);
        }
        let graph = &mut self.state.topology.graph;
        let fields = &mut self.state.fields;
        self.compatibility = crate::compatibility3d::solve_geometric_transport_compatibility_3d(
            graph, fields, 4096,
        )?;
        let source_density = fields.density.clone();
        let source_gamma = fields.gamma.clone();
        publish_transport_characteristic_clearance(
            graph,
            fields,
            dt,
            Some(&source_density),
            Some(&source_gamma),
        )?;
        let ledger = &mut self.source_ledger;
        let (steps, _) =
            transport_volume_with_commit(graph, fields, dt, false, |_, step, fields| {
                let receipt = ledger
                    .commit_microstep(step as f64)
                    .map_err(|e| ValidationError(e.into()))?;
                if !receipt.accepted {
                    fields.fault = Some(NumericalFault {
                        stage: "source-commit".into(),
                        index: 0,
                        observed: ledger.pending,
                        expected: 0.0,
                    });
                }
                Ok(())
            })?;
        self.microsteps = steps;
        crate::physical::publish_final_apertures(graph, fields);
        self.tracer_receipt = self.tracers.advance(graph, fields, dt)?;
        publish_scalar_interface_state_from_geometric_density(
            &mut self.scalar_authority,
            &self.state.topology,
            &mut self.state.fields,
            &source_density,
            &source_gamma,
            self.revision.frame + 1,
            self.topology_slot,
            !self.physical.bodies.is_empty(),
        )?;
        self.physical.finish_frame(&self.state)?;
        let policy = self.current_resolution_options(dt_s)?;
        let decision = plan_resolution_3d(
            &self.state.topology,
            &self.state.fields,
            &self.resolution_policy,
            dt_s,
            self.physical.frame.cell_size_m,
            &policy,
        )
        .map_err(|e| ValidationError(format!("3D resolution: {e:?}")))?;
        if decision.receipt.fault_bits != 0 {
            return Err(ValidationError(format!(
                "3D resolution fault {}",
                decision.receipt.fault_bits
            )));
        }
        if self.bricks_changed(&decision.candidate_bricks) {
            self.transition(decision.candidate_bricks, dt_s)?;
        } else {
            self.rdf = Rdf3d::reconstruct(&self.state.topology, &self.state.fields)?;
        }
        self.resolution_policy = decision.state;
        self.resolution_receipt = Some(decision.receipt);
        self.arena.release_after_publication()?;
        self.revision.frame += 1;
        self.revision.time += dt_s;
        self.revision.command_sequence = sequence;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        Ok(())
    }
    fn current_resolution_options(
        &self,
        dt_s: f64,
    ) -> Result<ResolutionPolicyOptions3d, ValidationError> {
        let mut policy = self.resolution_options.clone();
        policy.maximum_leaves = Some(self.arena.maximum_volume_leaves);
        policy.maximum_cells = Some(self.arena.capacity as usize * 512);
        policy.free_leaf_ids.clone_from(&self.arena.free_leaf_ids);
        policy.moving_rigid_bodies = !self.physical.bodies.is_empty();
        policy.policy.freeze_topology |= self.topology_frozen;
        if let Some(inflow) = &self.physical.scene.fluid.inflow {
            let strength = crate::dynamic3d::average_strength(
                inflow,
                self.revision.time,
                self.revision.time + dt_s,
            ) as f32;
            let velocity = inflow.velocity_m_s.array();
            let speed = velocity[0].hypot(velocity[1]).hypot(velocity[2]);
            if strength > 0.0 && speed > 0.0 {
                let h = self.physical.frame.cell_size_m;
                let outlet = std::array::from_fn(|a| {
                    ((inflow.center_m.array()[a] + velocity[a] * (0.5 * inflow.length_m / speed)
                        - self.physical.frame.origin_m[a])
                        / h) as f32
                });
                let velocity = velocity.map(|v| (v * strength as f64 / h) as f32);
                if let Some(bounds) = crate::resolution3d::continuous_inflow_demand_bounds_3d(
                    outlet,
                    velocity,
                    (inflow.radius_m / h) as f32,
                    dt_s as f32,
                )
                .map_err(|e| ValidationError(format!("3D source bounds: {e:?}")))?
                {
                    policy.source_demand_bounds_fine.push(bounds);
                }
            }
        }
        Ok(policy)
    }
    fn bricks_changed(&self, bricks: &[BrickSeed]) -> bool {
        let old = &self.state.topology.bricks;
        old.len() != bricks.len()
            || old.iter().zip(bricks).any(|(a, b)| {
                a.seed.id != b.id
                    || a.seed.key != b.key
                    || a.seed.coordinate != b.coordinate
                    || a.seed.span_bricks != b.span_bricks
                    || a.seed.resolution != b.resolution
                    || a.seed.active != b.active
            })
    }
    fn allocated_bytes(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            core::arch::wasm32::memory_size::<0>() * 65536
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native diagnostic lower bound; browser reports actual linear memory,
            // including the shared Rayon stacks and solver scratch allocations.
            let f = &self.state.fields;
            let numeric: usize = [
                &f.density,
                &f.gamma,
                &f.capacity,
                &f.capacity_before,
                &f.capacity_after,
                &f.capacity_rate,
                &f.source_rate,
                &f.inflow_coverage,
                &f.cell_velocity,
                &f.face_velocity,
                &f.pressure,
                &f.pressure_rhs,
                &f.pressure_diagonal,
                &f.interface_normal,
                &f.interface_offset,
                &f.low_flux,
                &f.high_flux,
                &f.limited_flux,
                &f.characteristic_clearance,
                &f.subface_velocity_correction,
                &self.density_view,
                &self.phi_view,
                &self.velocity_view,
                &self.tracers.state,
            ]
            .iter()
            .map(|v| v.capacity() * 4)
            .sum();
            let g = &self.state.topology.graph;
            numeric
                + g.cells.capacity() * std::mem::size_of::<crate::types::Cell>()
                + g.rows.capacity() * std::mem::size_of::<crate::types::Row>()
                + g.subfaces.capacity() * std::mem::size_of::<crate::types::Subface>()
                + g.solid_voxel_fraction.capacity() * 4
                + self.publication.capacity()
        }
    }
    pub fn snapshot(&mut self, _mask: u32) -> Result<&[u8], ValidationError> {
        let [nx, ny, nz] = self.state.description.dimensions.map(|n| n as usize);
        let h = self.physical.frame.cell_size_m as f32;
        let g = &self.state.topology.graph;
        let f = &self.state.fields;
        self.density_view.fill(0.0);
        self.phi_view.fill(4.0 * h);
        self.velocity_view.fill(0.0);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let q = x + nx * (y + ny * z);
                    let p = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    if let Some(i) = owner_at(g, p) {
                        let fill = (f.density[i] / f.capacity[i].max(1e-6)).clamp(0.0, 1.0);
                        self.density_view[q] = fill;
                        self.phi_view[q] = self.rdf.sample(i, p, g, f).unwrap_or(
                            (LIQUID_ISOVALUE - fill) * 4.0 * min_width(g.cells[i].widths),
                        ) * h;
                        for a in 0..3 {
                            self.velocity_view[4 * q + a] = f.cell_velocity[3 * i + a] * h;
                        }
                    }
                }
            }
        }
        let metadata=serde_json::to_vec(&serde_json::json!({"revision":self.revision,"receipt":self.receipt(),
            "scene":{"dimensions":self.state.description.dimensions,"originM":self.physical.frame.origin_m,
                "cellSizeM":[h,h,h],"dtS":self.timestep_s},
            "stats":{"pressureIterations":self.pressure.iterations,"activeCells":g.cells.len(),
                "activeBricks":self.state.topology.bricks.iter().filter(|b|b.seed.active).count(),
                "allocatedBytes":self.allocated_bytes(),
                "presentationBytes":(self.density_view.capacity()+self.phi_view.capacity()+self.velocity_view.capacity())*4,
                "pressureSolver":"Rust PCG", "gridKind":"octree"},
            "rigidBodies":self.physical.bodies,"rigidCoupling":self.physical.coupling_receipts,
            "geometricCompatibility":self.compatibility,
            "primaryRefinement":self.primary_refinement,
            "coupledPressure":self.coupled_pressure,
            "pressureTotalIterations":self.pressure.iterations+self.primary_refinement.iterations
                +self.coupled_pressure.f64_refinement_iterations,
            "pressureAuthority":self.pressure_authority.receipt,"scalarAuthority":self.scalar_authority.receipt,
            "tracerLattice":self.tracers.lattice,"tracersEnabled":self.tracers.enabled
        })).map_err(|e|ValidationError(e.to_string()))?;
        let mut planes = vec![
            Plane::F32(PlaneId::Density3D, &self.density_view),
            Plane::F32(PlaneId::SurfacePhi3D, &self.phi_view),
            Plane::F32(PlaneId::Velocity3D, &self.velocity_view),
        ];
        if self.tracers.enabled {
            planes.push(Plane::F32(PlaneId::Tracers, &self.tracers.state));
        }
        encode_publication(&metadata, &planes, &mut self.publication)
            .map_err(|e| ValidationError(e.into()))?;
        Ok(&self.publication)
    }
}
fn measure(state: &SceneState<3>) -> f64 {
    state
        .topology
        .graph
        .cells
        .iter()
        .map(|c| state.fields.density[c.id as usize] as f64 * c.measure as f64)
        .sum()
}
fn edited_geometry(old: DynamicGeometry, mut new: DynamicGeometry, dt: f64) -> DynamicGeometry {
    new.capacity_before = old.capacity_after;
    new.open_fraction_before = old.open_fraction_after;
    new.capacity = new.capacity_after.clone();
    new.capacity_rate = new
        .capacity_after
        .iter()
        .zip(&new.capacity_before)
        .map(|(&after, &before)| ((after as f64 - before as f64) / dt) as f32)
        .collect();
    new.mean_open_fraction = new
        .open_fraction_after
        .iter()
        .zip(&new.open_fraction_before)
        .map(|(&after, &before)| 0.5 * (before + after))
        .collect();
    new.open_fraction = new.mean_open_fraction.clone();
    new
}
fn reconcile_bodies(
    old_authored: &[RigidBodyDescription],
    live: &[RigidBodyState],
    next_authored: &[RigidBodyDescription],
) -> Result<Vec<RigidBodyState>, ValidationError> {
    let mut ids = std::collections::BTreeSet::new();
    let mut result = Vec::with_capacity(next_authored.len().min(12));
    for next in next_authored.iter().take(12) {
        if !ids.insert(next.id.as_str()) {
            return Err(ValidationError(format!(
                "duplicate rigid body id {}",
                next.id
            )));
        }
        let mut body = initialize_body(next.clone())?;
        let old = old_authored.iter().find(|old| old.id == next.id);
        let current = live
            .iter()
            .find(|current| current.description.id == next.id);
        if let (Some(old), Some(current)) = (old, current) {
            let authored_pose_unchanged = old.position_m == next.position_m
                && old.orientation == next.orientation
                && old.linear_velocity_m_s == next.linear_velocity_m_s
                && old.angular_velocity_rad_s == next.angular_velocity_rad_s
                && old.motion == next.motion;
            if authored_pose_unchanged {
                body.position_m = current.position_m;
                body.orientation = current.orientation;
                body.linear_velocity_m_s = current.linear_velocity_m_s;
                body.angular_velocity_rad_s = current.angular_velocity_rad_s;
                body.angular_momentum_kg_m2_s.x = if body.inverse_inertia_body_kg_m2.x > 0.0 {
                    current.angular_velocity_rad_s.x / body.inverse_inertia_body_kg_m2.x
                } else {
                    0.0
                };
                body.angular_momentum_kg_m2_s.y = if body.inverse_inertia_body_kg_m2.y > 0.0 {
                    current.angular_velocity_rad_s.y / body.inverse_inertia_body_kg_m2.y
                } else {
                    0.0
                };
                body.angular_momentum_kg_m2_s.z = if body.inverse_inertia_body_kg_m2.z > 0.0 {
                    current.angular_velocity_rad_s.z / body.inverse_inertia_body_kg_m2.z
                } else {
                    0.0
                };
                body.held = current.held;
            }
        }
        result.push(body);
    }
    Ok(result)
}
fn min_width(v: [f32; 3]) -> f32 {
    v[0].min(v[1].min(v[2]))
}
fn validate_dt(dt: f64) -> Result<(), ValidationError> {
    if !(dt.is_finite() && dt > 0.0 && (dt as f32).is_finite()) {
        Err(ValidationError(
            "timestep must be finite and positive".into(),
        ))
    } else {
        Ok(())
    }
}
fn validate_pressure(iterations: u32, tolerance: f32) -> Result<(), ValidationError> {
    if iterations == 0 || iterations > 4096 || !tolerance.is_finite() || tolerance < 0.0 {
        Err(ValidationError("invalid pressure budget".into()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> SceneDocument {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../core/testdata/production-scene-golden.json"
        ))
        .unwrap();
        serde_json::from_value(fixture["cases"][0]["scene"].clone()).unwrap()
    }
    #[test]
    fn closed_three_dimensional_world_advances_and_publishes_coherently() {
        let mut world = World3d::from_document(
            document(),
            ProductionSceneOptions::default(),
            WorldOptions::default(),
        )
        .unwrap();
        let volume = world.receipt().liquid_measure;
        for sequence in 1..=3 {
            world.advance(sequence, 1.0 / 60.0).unwrap();
            assert!(
                world.state.fields.fault.is_none(),
                "{:?}",
                world.state.fields.fault
            );
            assert!((world.receipt().liquid_measure - volume).abs() < volume * 2e-6);
            let publication = world.snapshot(u32::MAX).unwrap();
            assert_eq!(&publication[..8], b"FLUIDCPU");
            let offset = u32::from_le_bytes(publication[20..24].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(publication[24..28].try_into().unwrap()) as usize;
            let metadata: serde_json::Value =
                serde_json::from_slice(&publication[offset..offset + len]).unwrap();
            assert_eq!(metadata["revision"]["dimension"], 3);
            assert_eq!(metadata["revision"]["frame"], sequence);
            assert!(world.phi_view.iter().all(|v| v.is_finite()));
            assert!(world
                .density_view
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
    }
    #[test]
    fn invalid_command_does_not_change_three_dimensional_world() {
        let mut world = World3d::from_document(
            document(),
            ProductionSceneOptions::default(),
            WorldOptions::default(),
        )
        .unwrap();
        assert!(world
            .apply_command(1, 2, Command3d::SetTimeStep { dt_s: 0.1 })
            .is_err());
        assert_eq!(world.revision.command_sequence, 0);
        assert!(world.advance(1, f64::NAN).is_err());
        assert_eq!(world.revision.frame, 0);
    }
    #[test]
    fn timestep_change_invalidates_the_frozen_compatibility_rate() {
        let mut world = World3d::from_document(
            document(),
            ProductionSceneOptions::default(),
            WorldOptions::default(),
        )
        .unwrap();
        world.state.fields.subface_compatibility_rate =
            vec![0.125; world.state.topology.graph.subfaces.len()];
        vec![f32::EPSILON; world.state.topology.graph.subfaces.len()];
        world
            .apply_command(1, 1, Command3d::SetTimeStep { dt_s: 1.0 / 60.0 })
            .unwrap();
        assert!(world.state.fields.subface_compatibility_rate.is_empty());
    }
    #[test]
    fn authored_edits_preserve_live_pose_until_the_pose_is_explicitly_changed() {
        let mut scene = document();
        scene.rigid_bodies.push(
            serde_json::from_value(serde_json::json!({
              "id":"edit-body","name":"Edit body","shape":"sphere",
              "dimensions_m":{"x":0.05,"y":0.05,"z":0.05},"density_kg_m3":500,
              "position_m":{"x":0,"y":0.7,"z":0},"orientation":{"w":1,"x":0,"y":0,"z":0},
              "linearVelocity_m_s":{"x":0,"y":0,"z":0},
              "angularVelocity_rad_s":{"x":0,"y":0,"z":0},"restitution":0.2,"friction":0.4
            }))
            .unwrap(),
        );
        let mut world = World3d::from_document(
            scene.clone(),
            ProductionSceneOptions::default(),
            WorldOptions::default(),
        )
        .unwrap();
        let liquid = world.receipt().liquid_measure;
        world.physical.bodies[0].position_m.x = 0.125;
        scene.fluid.dynamic_viscosity_pa_s *= 2.0;
        world
            .apply_command(
                1,
                1,
                Command3d::SetScene {
                    scene: scene.clone(),
                },
            )
            .unwrap();
        assert_eq!(world.physical.bodies[0].position_m.x, 0.125);
        assert!((world.receipt().liquid_measure - liquid).abs() < liquid * 1e-6);

        let mut bodies = scene.rigid_bodies.clone();
        bodies[0].position_m.x = -0.2;
        world
            .apply_command(2, 1, Command3d::SetRigidBodies { bodies })
            .unwrap();
        assert_eq!(world.physical.bodies[0].position_m.x, -0.2);
        assert_eq!(world.revision.command_sequence, 2);
        assert!((world.receipt().liquid_measure - liquid).abs() < liquid * 1e-6);
    }
}

fn resolution_options_for_document(
    document: &SceneDocument,
    production: &ProductionSceneOptions,
) -> Result<ResolutionPolicyOptions3d, ValidationError> {
    let mut options = ResolutionPolicyOptions3d::production();
    options.policy = production.resolution.policy.clone();
    let origin = [
        -0.5 * document.container.width_m,
        0.0,
        -0.5 * document.container.depth_m,
    ];
    let h = document.voxel_domain.finest_cell_size_m;
    for raw in &document.fluid.refinement_regions {
        let min: crate::scene_model::Vec3 = serde_json::from_value(
            raw.get("min_m")
                .cloned()
                .ok_or_else(|| ValidationError("refinement region missing min_m".into()))?,
        )
        .map_err(|e| ValidationError(e.to_string()))?;
        let max: crate::scene_model::Vec3 = serde_json::from_value(
            raw.get("max_m")
                .cloned()
                .ok_or_else(|| ValidationError("refinement region missing max_m".into()))?,
        )
        .map_err(|e| ValidationError(e.to_string()))?;
        let minimum = min.array();
        let maximum = max.array();
        if (0..3)
            .any(|a| !minimum[a].is_finite() || !maximum[a].is_finite() || minimum[a] > maximum[a])
        {
            return Err(ValidationError("invalid refinement region bounds".into()));
        }
        let parse_width = |key: &str| -> Result<Option<u8>, ValidationError> {
            raw.get(key)
                .map(|v| {
                    let n = v
                        .as_u64()
                        .ok_or_else(|| ValidationError(format!("invalid {key}")))?;
                    if n == 0 || n > 128 || !n.is_power_of_two() {
                        return Err(ValidationError(format!(
                            "{key} must be a representable dyadic width"
                        )));
                    }
                    Ok(n as u8)
                })
                .transpose()
        };
        let minimum_cell_width = parse_width("minimumCellSize_cells")?.unwrap_or(1);
        let maximum_cell_width = parse_width("maximumCellSize_cells")?;
        if maximum_cell_width.is_some_and(|v| v < minimum_cell_width) {
            return Err(ValidationError("refinement minimum exceeds maximum".into()));
        }
        options.refinement_regions.push(ResolutionRegion3d {
            minimum_fine: std::array::from_fn(|a| (minimum[a] - origin[a]) / h),
            maximum_fine: std::array::from_fn(|a| (maximum[a] - origin[a]) / h),
            minimum_cell_width,
            maximum_cell_width,
        });
    }
    Ok(options)
}

impl World3d {
    pub fn inject_liquid(
        &mut self,
        drop: crate::injection3d::LiquidDrop3d,
    ) -> Result<(), ValidationError> {
        let drop = drop.in_frame(self.physical.frame)?;
        let bounds = drop.bounds();
        let dims = self.state.description.dimensions;
        if (0..3).any(|a| bounds.1[a] < 0.0 || bounds.0[a] > dims[a] as f64) {
            self.last_injection = Some(crate::injection3d::InjectionReceipt3d::default());
            return Ok(());
        }
        let accepted = self.revision.topology_generation;
        let mut policy = self.current_resolution_options(self.timestep_s)?;
        policy.injection_bounds_fine = Some(bounds);
        // An explicit live edit is allowed to demand pages even when normal
        // temporal adaptivity has been frozen by the reader.
        policy.policy.freeze_topology = false;
        let decision = plan_projected_transport_support_3d(
            &self.state.topology,
            &self.state.fields,
            self.timestep_s,
            self.physical.frame.cell_size_m,
            &policy,
        )
        .map_err(|e| ValidationError(format!("3D injection support: {e:?}")))?;
        if decision.fault_bits != 0 {
            return Err(ValidationError(format!(
                "3D injection support fault {}",
                decision.fault_bits
            )));
        }
        if self.bricks_changed(&decision.candidate_bricks) {
            self.transition(decision.candidate_bricks, self.timestep_s)?;
        }
        let mut receipt = crate::injection3d::apply_dose_3d(
            &self.state.topology.graph,
            &mut self.state.fields,
            drop,
        )?;
        self.state.fields.subface_compatibility_rate.clear();
        receipt.accepted_generation = accepted;
        reconstruct_interfaces(&self.state.topology.graph, &mut self.state.fields)?;
        self.rdf = Rdf3d::reconstruct(&self.state.topology, &self.state.fields)?;
        self.seeded_volume += receipt.volume_admitted_fine;
        self.last_injection = Some(receipt);
        self.revision.injections += 1;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        self.arena.release_after_publication()?;
        Ok(())
    }
    /// Adopt authored method values within Rust. The host sends the control
    /// values themselves and performs no numerical policy normalization.
    pub fn apply_runtime_values(
        &mut self,
        values: &serde_json::Value,
    ) -> Result<(), ValidationError> {
        let values = values
            .as_object()
            .ok_or_else(|| ValidationError("method values must be an object".into()))?;
        let mut iterations = self.options.pressure_iterations;
        let mut tolerance = self.options.pressure_relative_tolerance;
        if let Some(v) = values.get("pressureIterations") {
            iterations =
                u32::try_from(v.as_u64().ok_or_else(|| {
                    ValidationError("pressureIterations must be an integer".into())
                })?)
                .map_err(|e| ValidationError(e.to_string()))?;
        }
        if let Some(v) = values.get("pressureRelativeTolerance") {
            tolerance = v.as_f64().ok_or_else(|| {
                ValidationError("pressureRelativeTolerance must be numeric".into())
            })? as f32;
        }
        validate_pressure(iterations, tolerance)?;
        let mut dt = self.timestep_s;
        if let Some(v) = values.get("timeStep") {
            dt = match v.as_str() {
                Some("scene") => self.document.numerics.fixed_dt_s,
                Some("paper") => crate::atlas::CM12_PAPER_DT_S,
                _ => return Err(ValidationError("unknown timeStep mode".into())),
            };
        }
        validate_dt(dt)?;
        let policy =
            crate::runtime_options3d::activity_policy(&self.resolution_options.policy, values);
        let static_floors = crate::resolution3d::static_boundary_floors_3d(
            &self.state.topology,
            policy.detail_tolerance as f32,
        )
        .map_err(|e| ValidationError(format!("3D static policy: {e:?}")))?;
        self.resolution_options.static_boundary_floor_by_brick = static_floors;
        self.options.pressure_iterations = iterations;
        self.options.pressure_relative_tolerance = tolerance;
        self.timestep_s = dt;
        self.state.fields.subface_compatibility_rate.clear();
        self.resolution_options.policy = policy.clone();
        self.production_options.resolution.policy = policy;
        Ok(())
    }
}
