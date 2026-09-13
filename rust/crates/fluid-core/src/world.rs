//! Owned numerical world and revision-coherent publication boundary.
use crate::embedding::PressureEmbedding;
use crate::lifecycle::{prepare_candidate, LeafArena};
use crate::numerics::{
    assemble_pressure_rhs, enforce_inflow_faces, prepare_pressure_topology,
    project_pressure_velocity, solve_pressure,
};
use crate::physical::{publish_final_apertures, PhysicalContext};
use crate::presentation::{
    reconstruct_shared_rdf, AirCellGeometry, RdfSupport, RdfSurface, RdfTopology,
};
use crate::pressure_authority::PressureAuthority;
use crate::publication::{encode_publication, Plane, PlaneId};
use crate::resolution::{
    initialize_resolution_policy, plan_projected_transport_support, plan_resolution,
    ResolutionPolicyOptions, ResolutionPolicyReceipt, ResolutionPolicyState,
};
use crate::scalar_authority::{
    publish_scalar_interface_state_from_geometric_density, ScalarAuthority,
};
use crate::scene::{compile_scene_2d, SceneDescription, SceneState};
use crate::sources::SourceLedger;
use crate::topology::BrickSeed;
use crate::tracers::{TracerReceipt, Tracers, TRACER_BUDGET};
use crate::{
    collocate_velocity, extend_velocity, force_faces, prepare_faces,
    publish_transport_characteristic_clearance, reconstruct_interfaces,
    transport_volume_with_commit, Fields, PressureReceipt, ValidationError,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WorldOptions {
    pub run_epoch: u32,
    pub command_sequence: u32,
    pub pressure_iterations: u32,
    pub pressure_relative_tolerance: f32,
    pub tracer_budget: usize,
    pub topology_page_budget: Option<u32>,
}
impl Default for WorldOptions {
    fn default() -> Self {
        Self {
            run_epoch: 1,
            command_sequence: 0,
            pressure_iterations: 28,
            pressure_relative_tolerance: 1e-6,
            tracer_budget: TRACER_BUDGET,
            topology_page_budget: None,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Revision {
    pub schema_version: u32,
    pub dimension: u8,
    pub run_epoch: u32,
    pub command_sequence: u32,
    pub frame: u32,
    pub time: f64,
    pub injections: u32,
    pub topology_generation: u32,
    pub field_revision: u32,
    pub surface_revision: u32,
    pub memory_epoch: u32,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldReceipt<'a> {
    #[serde(flatten)]
    pub revision: &'a Revision,
    pub pressure: &'a PressureReceipt,
    pub microsteps: usize,
    pub source_ledger: &'a SourceLedger,
    pub tracers: TracerReceipt,
    pub liquid_measure: f64,
    pub fault: &'a Option<crate::NumericalFault>,
    pub seeded_volume: f64,
    pub drift: f64,
    pub max_velocity: f64,
    pub last_injection: &'a Option<crate::injection::InjectionReceipt>,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Command {
    Snapshot,
    SetTimeStep {
        #[serde(rename = "dt_s")]
        dt_s: f64,
    },
    InjectLiquid {
        drop: crate::injection::LiquidDrop,
    },
    SetRefinementRegions {
        regions: Vec<crate::resolution::ResolutionRegion>,
    },
    SetTracers {
        enabled: bool,
    },
    ReseedTracers,
    SetPressureBudget {
        iterations: u32,
        #[serde(rename = "relativeTolerance")]
        relative_tolerance: f32,
    },
}

pub type CandidateBuilder =
    Box<dyn Fn(&SceneState<2>, Vec<BrickSeed>) -> Result<SceneState<2>, ValidationError>>;
pub type EmbeddingBuilder = Box<
    dyn Fn(
        &SceneState<2>,
        Option<&PressureEmbedding>,
    ) -> Result<Option<PressureEmbedding>, ValidationError>,
>;

pub struct World {
    pub state: SceneState<2>,
    pub revision: Revision,
    pub options: WorldOptions,
    pub pressure: PressureReceipt,
    pub source_ledger: SourceLedger,
    pub microsteps: usize,
    pub tracers: Tracers,
    pub tracer_receipt: TracerReceipt,
    pub physical: Option<PhysicalContext>,
    pub embedding: Option<PressureEmbedding>,
    pub candidate_builder: Option<CandidateBuilder>,
    pub embedding_builder: Option<EmbeddingBuilder>,
    pub arena: LeafArena,
    pub resolution_options: ResolutionPolicyOptions,
    pub resolution_policy: ResolutionPolicyState,
    pub resolution_receipt: Option<ResolutionPolicyReceipt>,
    pub pressure_authority: PressureAuthority,
    pub scalar_authority: ScalarAuthority,
    /// Accepted sparse-topology bank. The runtime starts in bank zero and
    /// flips banks for every committed topology transition.
    pub topology_slot: u8,
    pub seeded_volume: f64,
    pub last_injection: Option<crate::injection::InjectionReceipt>,
    pub timestep_s: f64,
    pub scene_document: Option<serde_json::Value>,
    pub material_id: Vec<u16>,
    pub view_history: crate::view_history::ViewHistory,
    material_values: Vec<f32>,
    capacity_fine: Vec<f32>,
    rdf_topology: RdfTopology,
    rdf_support: RdfSupport,
    surface: RdfSurface,
    graph_json: Vec<u8>,
    last_graph_publication: Option<u32>,
    publication: Vec<u8>,
}
impl World {
    pub fn from_document(
        document: crate::initial_scene::SceneDocument,
        production_options: crate::production_scene::ProductionSceneOptions,
        options: WorldOptions,
    ) -> Result<Self, ValidationError> {
        let bundle = crate::production_scene::production_scene(document, production_options)
            .map_err(|e| ValidationError(e.to_string()))?;
        let scene_document =
            serde_json::to_value(&bundle.document).map_err(|e| ValidationError(e.to_string()))?;
        let mut world = Self::from_state(bundle.state, options)?;
        world.material_id = bundle.material_id;
        world.material_values = world.material_id.iter().map(|&v| v as f32).collect();
        world.scene_document = Some(scene_document);
        world.physical = Some(bundle.physical);
        world.embedding = bundle.embedding;
        world.resolution_options = bundle.resolution_options;
        world.timestep_s = bundle.dt_s;
        if world.embedding.is_some() {
            let document = bundle.document;
            let atlas = bundle.source_atlas;
            world.embedding_builder = Some(Box::new(move |reduced, previous| {
                let (_, embedding) = crate::production_scene::rebuild_source_embedding(
                    &document, &atlas, reduced, previous,
                )
                .map_err(|e| ValidationError(e.to_string()))?;
                Ok(Some(embedding))
            }));
        }
        Ok(world)
    }
    pub fn from_scene(
        description: SceneDescription,
        options: WorldOptions,
    ) -> Result<Self, ValidationError> {
        if description.dimension != 2 {
            return Err(ValidationError(
                "3D world requires the 3D transport implementation".into(),
            ));
        }
        let state = compile_scene_2d(description).map_err(|e| ValidationError(e.to_string()))?;
        Self::from_state(state, options)
    }
    pub fn from_state(
        mut state: SceneState<2>,
        options: WorldOptions,
    ) -> Result<Self, ValidationError> {
        if options.run_epoch == 0 {
            return Err(ValidationError("runEpoch must be positive".into()));
        }
        validate_pressure_options(
            options.pressure_iterations,
            options.pressure_relative_tolerance,
        )?;
        let arena = LeafArena::new(&state.topology, options.topology_page_budget)?;
        let resolution_policy = initialize_resolution_policy(&state.topology);
        let pressure_authority = PressureAuthority::new(
            &state.topology.graph,
            None,
            Some((arena.capacity as usize * 144).max(state.topology.graph.rows.len())),
        );
        let scalar_authority = ScalarAuthority::new(
            arena.capacity as usize,
            state.topology.graph.cells.len(),
            state.topology.graph.topology_generation,
            0,
        );
        reconstruct_interfaces(&state.topology.graph, &mut state.fields)?;
        let rdf_topology = RdfTopology::compile(&state.topology.graph)?;
        let rdf_support = Self::support_for(&state);
        let surface = reconstruct_shared_rdf(
            &state.topology.graph,
            &state.fields,
            &rdf_topology,
            &rdf_support,
        )?;
        let revision = Revision {
            schema_version: 1,
            dimension: 2,
            run_epoch: options.run_epoch,
            command_sequence: options.command_sequence,
            frame: 0,
            time: 0.0,
            injections: 0,
            topology_generation: state.topology.graph.topology_generation,
            field_revision: 1,
            surface_revision: 1,
            memory_epoch: 1,
        };
        let tracers = Tracers::new(state.description.dimensions, options.tracer_budget);
        let graph_json = Self::encode_graph(&state)?;
        let seeded_volume = state
            .topology
            .graph
            .cells
            .iter()
            .map(|c| state.fields.density[c.id as usize] as f64 * c.measure as f64)
            .sum();
        let timestep_s = state.description.dt_s as f64;
        let view_history = crate::view_history::ViewHistory::new(&state);
        let [nx, ny, _] = state.description.dimensions.map(|v| v as usize);
        let mut capacity_fine = vec![0.0; nx * ny];
        for cy in 0..ny {
            for x in 0..nx {
                let index = x + nx * (ny - 1 - cy);
                capacity_fine[x + nx * cy] = if !state.description.raster_capacity.is_empty() {
                    state.description.raster_capacity[index]
                } else {
                    crate::numerics::owner_at(
                        &state.topology.graph,
                        [x as f32 + 0.5, (ny - cy) as f32 - 0.5, 0.0],
                    )
                    .map_or(0.0, |i| state.fields.capacity[i])
                };
            }
        }
        Ok(Self {
            state,
            revision,
            options,
            pressure: PressureReceipt::default(),
            source_ledger: SourceLedger::default(),
            microsteps: 1,
            tracers,
            tracer_receipt: TracerReceipt::default(),
            physical: None,
            embedding: None,
            candidate_builder: None,
            embedding_builder: None,
            arena,
            resolution_policy,
            pressure_authority,
            scalar_authority,
            topology_slot: 0,
            seeded_volume,
            last_injection: None,
            timestep_s,
            scene_document: None,
            material_id: Vec::new(),
            view_history,
            material_values: vec![0.0; nx * ny],
            capacity_fine,
            resolution_options: ResolutionPolicyOptions::default(),
            resolution_receipt: None,
            rdf_topology,
            rdf_support,
            surface,
            graph_json,
            last_graph_publication: None,
            publication: Vec::new(),
        })
    }
    fn check_sequence(&self, sequence: u32) -> Result<(), ValidationError> {
        if sequence <= self.revision.command_sequence {
            return Err(ValidationError("commandSequence must increase".into()));
        }
        Ok(())
    }
    pub fn receipt(&self) -> WorldReceipt<'_> {
        let liquid_measure: f64 = self
            .state
            .topology
            .graph
            .cells
            .iter()
            .map(|c| self.state.fields.density[c.id as usize] as f64 * c.measure as f64)
            .sum();
        WorldReceipt {
            revision: &self.revision,
            pressure: &self.pressure,
            microsteps: self.microsteps,
            source_ledger: &self.source_ledger,
            tracers: self.tracer_receipt,
            liquid_measure: self
                .state
                .topology
                .graph
                .cells
                .iter()
                .map(|c| self.state.fields.density[c.id as usize] as f64 * c.measure as f64)
                .sum(),
            fault: &self.state.fields.fault,
            seeded_volume: self.seeded_volume,
            drift: if self.seeded_volume != 0.0 {
                (liquid_measure - self.seeded_volume) / self.seeded_volume
            } else {
                0.0
            },
            max_velocity: self
                .state
                .fields
                .face_velocity
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs() as f64)),
            last_injection: &self.last_injection,
        }
    }
    pub fn apply_command(
        &mut self,
        sequence: u32,
        run_epoch: u32,
        command: Command,
    ) -> Result<(), ValidationError> {
        self.check_sequence(sequence)?;
        if run_epoch != self.revision.run_epoch {
            return Err(ValidationError("stale runEpoch".into()));
        }
        match command {
            Command::Snapshot => {}
            Command::SetTimeStep { dt_s } => {
                if !dt_s.is_finite() || !(dt_s as f32).is_finite() || (dt_s as f32) <= 0.0 {
                    return Err(ValidationError("invalid timestep".into()));
                }
                self.timestep_s = dt_s;
                self.state.description.dt_s = dt_s as f32;
            }
            Command::InjectLiquid { drop } => {
                self.inject_liquid(drop)?;
            }
            Command::SetRefinementRegions { regions } => {
                for r in &regions {
                    if ![1, 2, 4, 8].contains(&r.minimum_cell_width)
                        || r.maximum_cell_width
                            .is_some_and(|w| ![1, 2, 4, 8].contains(&w) || w < r.minimum_cell_width)
                        || (0..2).any(|a| {
                            !r.minimum_fine[a].is_finite()
                                || !r.maximum_fine[a].is_finite()
                                || r.maximum_fine[a] <= r.minimum_fine[a]
                        })
                    {
                        return Err(ValidationError("invalid refinement region".into()));
                    }
                }
                self.resolution_options.refinement_regions = regions;
            }
            Command::SetTracers { enabled } => self.tracers.set_enabled(enabled),
            Command::ReseedTracers => self.tracers.reseed(),
            Command::SetPressureBudget {
                iterations,
                relative_tolerance,
            } => {
                validate_pressure_options(iterations, relative_tolerance)?;
                self.options.pressure_iterations = iterations;
                self.options.pressure_relative_tolerance = relative_tolerance
            }
        }
        self.revision.command_sequence = sequence;
        Ok(())
    }
    pub fn advance(&mut self, sequence: u32, dt_s: f64) -> Result<(), ValidationError> {
        self.advance_with_observer(sequence, dt_s, |_, _, _| {})
    }
    pub fn advance_with_observer(
        &mut self,
        sequence: u32,
        dt_s: f64,
        mut observe: impl FnMut(&str, &crate::Graph, &Fields),
    ) -> Result<(), ValidationError> {
        self.check_sequence(sequence)?;
        let dt = dt_s as f32;
        if !dt_s.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(ValidationError(
                "dt must be finite, positive and representable as f32".into(),
            ));
        }
        if self.physical.is_none() && self.state.fields.solid_motion_active {
            return Err(ValidationError(
                "moving geometry requires a physical scene context".into(),
            ));
        }
        self.view_history.capture_bricks(&self.state);
        self.state.fields.fault = None;
        self.state.fields.frame_dt = dt;
        let inflow = if let Some(physical) = &mut self.physical {
            physical.begin_frame(
                &mut self.state,
                self.revision.time,
                dt_s,
                &mut self.source_ledger,
            )?
        } else {
            self.state
                .fields
                .capacity_before
                .clone_from(&self.state.fields.capacity);
            self.state
                .fields
                .capacity_after
                .clone_from(&self.state.fields.capacity);
            [0.0; 3]
        };
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            observe("dynamic-geometry", graph, fields);
            extend_velocity(graph, fields, 8)?;
            observe("transport-velocity-extension", graph, fields);
            prepare_faces(graph, fields, dt)?;
            observe("face-preparation", graph, fields);
        }
        self.view_history
            .capture_faces(&self.state, self.cell_size());
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            force_faces(graph, fields, dt, fields.acceleration_fine, inflow);
            observe("body-forces", graph, fields);
            reconstruct_interfaces(graph, fields)?;
            observe("interface-reconstruction", graph, fields);
            if let Some(embedding) = &mut self.embedding {
                let prepared = embedding.prepare(graph, fields);
                fields.pressure_diagonal.clone_from(&prepared.diagonal);
                fields.pressure_rhs.clone_from(&prepared.rhs);
                observe("pressure-rhs", graph, fields);
                let solved = embedding.solve(
                    graph,
                    fields,
                    self.options.pressure_iterations,
                    self.options.pressure_relative_tolerance,
                    Some(prepared),
                )?;
                self.pressure = PressureReceipt {
                    iterations: solved.solve.iterations,
                    initial_residual: solved.solve.initial_true_residual_squared.max(0.0).sqrt(),
                    residual: solved.solve.final_true_residual_squared.max(0.0).sqrt(),
                    converged: solved.solve.converged,
                };
                observe("pressure-solve", graph, fields);
                embedding.project(graph, fields, &solved.prepared);
            } else {
                let rows = prepare_pressure_topology(graph, fields);
                self.pressure_authority.publish(
                    graph,
                    fields,
                    &rows.active,
                    &rows.theta,
                    graph.topology_generation,
                    self.physical
                        .as_ref()
                        .is_some_and(|p| p.solid_world.is_some()),
                );
                observe("pressure-topology", graph, fields);
                assemble_pressure_rhs(graph, fields, &rows);
                observe("pressure-rhs", graph, fields);
                self.pressure = solve_pressure(
                    graph,
                    fields,
                    &rows,
                    self.options.pressure_iterations,
                    self.options.pressure_relative_tolerance,
                    Some(&self.pressure_authority.execution_order),
                )?;
                observe("pressure-solve", graph, fields);
                project_pressure_velocity(graph, fields, &rows);
            }
            enforce_inflow_faces(graph, fields, inflow);
            collocate_velocity(graph, fields);
            observe("velocity-projection", graph, fields);
        }
        let support = plan_projected_transport_support(
            &self.state.topology,
            &self.state.fields,
            dt_s,
            self.cell_size(),
            &self.resolution_options.policy,
            Some(self.arena.maximum_slice_leaves),
            Some(self.arena.capacity as usize * 64),
            &self.arena.free_leaf_ids,
        )
        .map_err(|e| ValidationError(format!("projected support: {e:?}")))?;
        if support.fault_bits != 0 {
            return Err(ValidationError(format!(
                "projected support fault {}",
                support.fault_bits
            )));
        }
        if self.bricks_changed(&support.candidate_bricks) {
            self.transition(support.candidate_bricks, dt_s)?;
            extend_velocity(&self.state.topology.graph, &mut self.state.fields, 8)?;
        }
        self.view_history.capture_density(&self.state);
        let source_density = self.state.fields.density.clone();
        let source_gamma = self.state.fields.gamma.clone();
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            publish_transport_characteristic_clearance(
                graph,
                fields,
                dt,
                Some(&source_density),
                Some(&source_gamma),
            )?;
            let ledger = &mut self.source_ledger;
            let (steps, _) =
                transport_volume_with_commit(graph, fields, dt, false, |_, dtm, fields| {
                    let receipt = ledger
                        .commit_microstep(dtm as f64)
                        .map_err(|e| ValidationError(e.into()))?;
                    if !receipt.accepted {
                        fields.fault = Some(crate::NumericalFault {
                            stage: "source-commit".into(),
                            index: 0,
                            observed: ledger.pending,
                            expected: 0.0,
                        });
                    }
                    Ok(())
                })?;
            self.microsteps = steps;
            publish_final_apertures(graph, fields);
            observe("conservative-transport", graph, fields);
            self.tracer_receipt = self.tracers.advance(graph, fields, dt)?;
        }
        publish_scalar_interface_state_from_geometric_density(
            &mut self.scalar_authority,
            &self.state.topology,
            &mut self.state.fields,
            &source_density,
            &source_gamma,
            self.revision.frame + 1,
            self.topology_slot,
            self.physical
                .as_ref()
                .is_some_and(|p| !p.scene.rigid_bodies.is_empty()),
        )?;
        observe(
            "scalar-publication",
            &self.state.topology.graph,
            &self.state.fields,
        );
        if let Some(physical) = &mut self.physical {
            physical.finish_frame(&self.state)?;
        }
        let mut policy = self.resolution_options.clone();
        policy.maximum_leaves = Some(self.arena.maximum_slice_leaves);
        policy.maximum_cells = Some(self.arena.capacity as usize * 64);
        policy.free_leaf_ids.clone_from(&self.arena.free_leaf_ids);
        policy.moving_rigid_bodies = self
            .physical
            .as_ref()
            .is_some_and(|p| !p.scene.rigid_bodies.is_empty());
        policy.injection_demanded_brick_keys = self
            .state
            .topology
            .graph
            .cells
            .iter()
            .filter(|c| self.state.fields.source_rate[c.id as usize] > 0.0)
            .filter_map(|c| c.brick_key)
            .collect();
        let decision = plan_resolution(
            &self.state.topology,
            &self.state.fields,
            &self.resolution_policy,
            dt_s,
            self.cell_size(),
            &policy,
        )
        .map_err(|e| ValidationError(format!("resolution: {e:?}")))?;
        self.view_history
            .capture_activity(&self.state, &decision.receipt);
        self.resolution_policy = decision.state;
        let changed =
            decision.receipt.candidate_generation > self.state.topology.graph.topology_generation;
        self.resolution_receipt = Some(decision.receipt);
        if changed {
            self.transition(decision.candidate_bricks, dt_s)?;
        }
        self.refresh_surface()?;
        self.arena.release_after_publication()?;
        self.revision.frame += 1;
        self.revision.time += dt_s;
        self.revision.command_sequence = sequence;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        Ok(())
    }
    pub fn inject_liquid(
        &mut self,
        drop: crate::injection::LiquidDrop,
    ) -> Result<(), ValidationError> {
        use crate::injection::{
            addressable, apply_dose, demanded_bricks, requested_area, InjectionReceipt,
        };
        let dimensions = [
            self.state.description.dimensions[0],
            self.state.description.dimensions[1],
        ];
        let generation = self.state.topology.graph.topology_generation;
        let mut receipt = InjectionReceipt {
            accepted_generation: generation,
            candidate_generation: generation,
            ..InjectionReceipt::default()
        };
        if !addressable(drop, dimensions) {
            self.last_injection = Some(receipt);
            return Ok(());
        }
        let bricks: Vec<_> = self
            .state
            .topology
            .bricks
            .iter()
            .map(|b| b.seed.clone())
            .collect();
        let demand = demanded_bricks(&bricks, drop);
        if demand.is_empty() {
            self.last_injection = Some(receipt);
            return Ok(());
        }
        receipt.bricks_demanded = demand.len();
        receipt.area_requested_fine = requested_area(drop, dimensions);
        let mut options = self.resolution_options.clone();
        options.injection_demanded_brick_keys = demand;
        options.maximum_leaves = Some(self.arena.maximum_slice_leaves);
        options.maximum_cells = Some(self.arena.capacity as usize * 64);
        options.free_leaf_ids.clone_from(&self.arena.free_leaf_ids);
        options.moving_rigid_bodies = self
            .physical
            .as_ref()
            .is_some_and(|p| !p.scene.rigid_bodies.is_empty());
        // The intervention planner observes collocated motion; projected face
        // demand belongs to the advance's pre-transport transaction.
        let mut planning = self.state.fields.clone();
        planning.face_velocity.clear();
        planning.acceleration_fine = [0.0; 3];
        let decision = plan_resolution(
            &self.state.topology,
            &planning,
            &self.resolution_policy,
            self.timestep_s,
            self.cell_size(),
            &options,
        )
        .map_err(|e| ValidationError(format!("injection resolution: {e:?}")))?;
        receipt.bricks_activated = decision.receipt.activated_brick_count;
        receipt.bricks_promoted = decision.receipt.promoted_brick_count;
        self.view_history
            .capture_activity(&self.state, &decision.receipt);
        self.resolution_policy = decision.state;
        self.resolution_receipt = Some(decision.receipt.clone());
        if decision.receipt.candidate_generation > generation {
            if let Err(error) = self.transition(decision.candidate_bricks, self.timestep_s) {
                receipt.fault = Some(crate::NumericalFault {
                    stage: format!("injection-transfer: {error}"),
                    index: 0,
                    observed: 1.0,
                    expected: 0.0,
                });
                self.last_injection = Some(receipt);
                return Ok(());
            }
        }
        let dose = apply_dose(&self.state.topology.graph, &mut self.state.fields, drop)?;
        reconstruct_interfaces(&self.state.topology.graph, &mut self.state.fields)?;
        self.refresh_surface()?;
        self.seeded_volume += dose.area_admitted_fine;
        self.revision.injections += 1;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        receipt.accepted = true;
        receipt.candidate_generation = self.state.topology.graph.topology_generation;
        receipt.cells_wetted = dose.cells_wetted;
        receipt.area_admitted_fine = dose.area_admitted_fine;
        self.last_injection = Some(receipt);
        Ok(())
    }
    fn cell_size(&self) -> f64 {
        self.physical
            .as_ref()
            .map_or(self.state.description.cell_size_m as f64, |p| {
                p.frame.source_cell_size
            })
    }
    fn bricks_changed(&self, bricks: &[BrickSeed]) -> bool {
        bricks.len() != self.state.topology.bricks.len()
            || bricks.iter().any(|b| {
                self.state
                    .topology
                    .bricks
                    .iter()
                    .find(|old| old.seed.key == b.key)
                    .is_none_or(|old| {
                        old.seed.active != b.active
                            || old.seed.resolution != b.resolution
                            || old.seed.span_bricks != b.span_bricks
                    })
            })
    }
    pub fn transition(
        &mut self,
        mut bricks: Vec<BrickSeed>,
        dt: f64,
    ) -> Result<(), ValidationError> {
        for b in &mut bricks {
            let lo = [b.coordinate[0] as f64 * 8.0, b.coordinate[1] as f64 * 8.0];
            let span = b.span_bricks as f64 * 8.0;
            let scale = self
                .resolution_options
                .refinement_regions
                .iter()
                .filter(|r| {
                    (0..2).all(|a| lo[a] < r.maximum_fine[a] && lo[a] + span > r.minimum_fine[a])
                })
                .fold(1u8, |v, r| v.max(r.minimum_cell_width));
            b.refinement_region_scale = Some(scale as f32);
            if self
                .state
                .topology
                .bricks
                .iter()
                .find(|old| old.seed.key == b.key)
                .is_some_and(|old| old.seed.resolution != b.resolution)
            {
                b.density.clear();
                b.gamma.clear();
            }
        }
        let mut candidate = if let Some(build) = &self.candidate_builder {
            build(&self.state, bricks)?
        } else {
            let mut description = self.state.description.clone();
            description.bricks = bricks;
            let mut candidate =
                compile_scene_2d(description).map_err(|e| ValidationError(e.to_string()))?;
            candidate
                .topology
                .graph
                .solid_voxel_fraction
                .clone_from(&self.state.topology.graph.solid_voxel_fraction);
            candidate
        };
        candidate.topology.graph.topology_generation =
            self.state.topology.graph.topology_generation + 1;
        if let Some(physical) = &self.physical {
            physical.candidate_geometry(&mut candidate, self.revision.time, dt)?;
        }
        // Stage all candidate-dependent authorities before swapping accepted state.
        let (staged, arena) = prepare_candidate(&self.state, candidate, &self.arena)
            .map_err(|e| ValidationError(e.to_string()))?;
        let embedding = if let Some(build) = &self.embedding_builder {
            build(&staged, self.embedding.as_ref())?
        } else if self.embedding.is_some() {
            return Err(ValidationError(
                "symmetry topology transition requires its retained source builder".into(),
            ));
        } else {
            None
        };
        let rdf_topology = RdfTopology::compile(&staged.topology.graph)?;
        let graph_json = Self::encode_graph(&staged)?;
        self.state = staged;
        self.arena = arena;
        self.embedding = embedding;
        self.rdf_topology = rdf_topology;
        self.graph_json = graph_json;
        self.revision.topology_generation = self.state.topology.graph.topology_generation;
        self.topology_slot ^= 1;
        self.pressure_authority.resize_preserving_cache(
            &self.state.topology.graph,
            None,
            Some((self.arena.capacity as usize * 144).max(self.state.topology.graph.rows.len())),
        );
        self.rdf_support = Self::support_for(&self.state);
        Ok(())
    }
    fn encode_graph(state: &SceneState<2>) -> Result<Vec<u8>, ValidationError> {
        let mut value = serde_json::to_value(&state.topology.graph)
            .map_err(|e| ValidationError(e.to_string()))?;
        value["bricks"] = serde_json::to_value(
            state
                .topology
                .bricks
                .iter()
                .map(|b| &b.seed)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ValidationError(e.to_string()))?;
        value["generation"] = serde_json::json!(state.topology.graph.topology_generation);
        serde_json::to_vec(&value).map_err(|e| ValidationError(e.to_string()))
    }
    fn support_for(state: &SceneState<2>) -> RdfSupport {
        let mut inactive_cells = Vec::new();
        for brick in state.topology.bricks.iter().filter(|b| !b.seed.active) {
            let width = 8.0 * brick.seed.span_bricks as f64 / brick.seed.resolution as f64;
            for y in 0..brick.seed.resolution {
                for x in 0..brick.seed.resolution {
                    let minimum = [
                        8.0 * brick.seed.coordinate[0] as f64 + x as f64 * width,
                        8.0 * brick.seed.coordinate[1] as f64 + y as f64 * width,
                    ];
                    let maximum = [
                        (minimum[0] + width).min(state.description.dimensions[0] as f64),
                        (minimum[1] + width).min(state.description.dimensions[1] as f64),
                    ];
                    if (0..2).all(|a| maximum[a] > minimum[a]) {
                        inactive_cells.push(AirCellGeometry {
                            minimum,
                            maximum,
                            center: [
                                (minimum[0] + maximum[0]) * 0.5,
                                (minimum[1] + maximum[1]) * 0.5,
                            ],
                        });
                    }
                }
            }
        }
        RdfSupport {
            solid_fraction: (!state.topology.graph.solid_voxel_fraction.is_empty())
                .then(|| state.topology.graph.solid_voxel_fraction.clone()),
            inactive_cells,
        }
    }
    fn refresh_surface(&mut self) -> Result<(), ValidationError> {
        self.surface = reconstruct_shared_rdf(
            &self.state.topology.graph,
            &self.state.fields,
            &self.rdf_topology,
            &self.rdf_support,
        )?;
        Ok(())
    }
    /// Bits 0/1/2 select fields/surface/tracers. Bit 3 requests the cold graph;
    /// a changed graph is always sent once regardless of the view mask.
    pub fn snapshot(&mut self, view_mask: u32) -> Result<&[u8], ValidationError> {
        let include_graph = self.last_graph_publication != Some(self.revision.topology_generation)
            || view_mask & 8 != 0;
        let scene = serde_json::json!({"dimensions":self.state.description.dimensions,"cellSizeM":self.cell_size(),"dtS":self.timestep_s,"originM":self.state.description.origin_m,
            "frame":self.physical.as_ref().map(|p|p.frame),"hasStaticWorld":self.physical.as_ref().is_some_and(|p|p.solid_world.as_ref().is_some_and(|w|!w.pages.is_empty()||!w.regions.is_empty())),
            "hasInflow":self.physical.as_ref().is_some_and(|p|p.scene.fluid.inflow.is_some()),"hasRigidBodies":self.physical.as_ref().is_some_and(|p|!p.bodies.is_empty())});
        let metadata=serde_json::to_vec(&serde_json::json!({"revision":self.revision,"receipt":self.receipt(),"surface":self.surface.receipt,
            "graphIncluded":include_graph,"tracerLattice":self.tracers.lattice,"tracersEnabled":self.tracers.enabled,"scene":scene,
            "resolution":self.resolution_receipt,"retirement":self.arena.retirement,
            "pressureAuthority":self.embedding.as_ref().map_or(&self.pressure_authority.receipt,|e|&e.pressure_authority.receipt),
            "scalarAuthority":self.scalar_authority.receipt,"rigidBodies":self.physical.as_ref().map(|p|&p.bodies),
            "rigidCoupling":self.physical.as_ref().map(|p|&p.coupling_receipts)})).map_err(|e|ValidationError(e.to_string()))?;
        let mut planes = Vec::with_capacity(27);
        let f: &Fields = &self.state.fields;
        if view_mask & 1 != 0 {
            macro_rules! plane {
                ($id:ident,$field:ident) => {
                    planes.push(Plane::F32(PlaneId::$id, &f.$field))
                };
            }
            plane!(Density, density);
            plane!(Gamma, gamma);
            plane!(Capacity, capacity);
            plane!(CellVelocity, cell_velocity);
            plane!(FaceVelocity, face_velocity);
            plane!(Pressure, pressure);
            plane!(PressureRhs, pressure_rhs);
            plane!(PressureDiagonal, pressure_diagonal);
            planes.push(Plane::U8(PlaneId::PressureMember, &f.pressure_member));
            planes.push(Plane::U8(
                PlaneId::PressureRowMember,
                &f.pressure_row_member,
            ));
            planes.push(Plane::U8(PlaneId::ExtensionDepth, &f.extension_depth));
            plane!(InterfaceNormal, interface_normal);
            plane!(InterfaceOffset, interface_offset);
            plane!(LowFlux, low_flux);
            plane!(HighFlux, high_flux);
            plane!(LimitedFlux, limited_flux);
            plane!(CapacityBefore, capacity_before);
            plane!(CapacityAfter, capacity_after);
            plane!(CapacityRate, capacity_rate);
            plane!(SourceRate, source_rate);
            plane!(InflowCoverage, inflow_coverage);
            plane!(CharacteristicClearance, characteristic_clearance);
            planes.push(Plane::F32(
                PlaneId::DensityBefore,
                &self.view_history.density_before,
            ));
            planes.push(Plane::F32(
                PlaneId::VelocityXBeforePressure,
                &self.view_history.velocity_x_before,
            ));
            planes.push(Plane::F32(
                PlaneId::VelocityYBeforePressure,
                &self.view_history.velocity_y_before,
            ));
            planes.push(Plane::U8(
                PlaneId::BrickResolutionBefore,
                &self.view_history.brick_resolution_before,
            ));
            planes.push(Plane::F32(
                PlaneId::BrickActivity,
                &self.view_history.brick_activity,
            ));
            planes.push(Plane::F32(PlaneId::MaterialId, &self.material_values));
            planes.push(Plane::F32(PlaneId::CapacityFine, &self.capacity_fine));
        }
        if view_mask & 2 != 0 {
            planes.push(Plane::F32(
                PlaneId::RdfVertices,
                &self.surface.vertex_phi_fine,
            ));
            planes.push(Plane::F32(
                PlaneId::RdfSegments,
                &self.surface.segments_fine,
            ));
        }
        if view_mask & 4 != 0 && self.tracers.enabled {
            planes.push(Plane::F32(PlaneId::Tracers, &self.tracers.state));
        }
        if include_graph {
            planes.push(Plane::Json(PlaneId::GraphJson, &self.graph_json));
        }
        encode_publication(&metadata, &planes, &mut self.publication)
            .map_err(|e| ValidationError(e.into()))?;
        self.last_graph_publication = Some(self.revision.topology_generation);
        Ok(&self.publication)
    }
}
fn validate_pressure_options(iterations: u32, tolerance: f32) -> Result<(), ValidationError> {
    if iterations > 4096 || !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ValidationError(
            "invalid pressure budget or relative tolerance".into(),
        ));
    }
    Ok(())
}

impl From<crate::pressure::PressureError> for ValidationError {
    fn from(e: crate::pressure::PressureError) -> Self {
        Self(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn world() -> World {
        let scene:SceneDescription=serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":2,"dimensions":[8,8,1],"cellSizeM":0.05,
            "dtS":1.0/60.0,"densityKgM3":998.2,"gravityMS2":[0.0,-9.81,0.0],
            "boundaries":["closed","closed","closed","closed","closed","closed"],
            "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"spanBricks":1,"resolution":8,"active":true}],
            "rasterDensity":(0..64).map(|i|if i%8<4{1.0}else{0.0}).collect::<Vec<_>>()
        })).unwrap();
        World::from_scene(
            scene,
            WorldOptions {
                tracer_budget: 128,
                ..WorldOptions::default()
            },
        )
        .unwrap()
    }
    #[test]
    fn invalid_commands_do_not_mutate_owned_state() {
        let mut world = world();
        let before = world.state.fields.density.clone();
        assert!(world.advance(1, f64::NAN).is_err());
        assert!(world.advance(1, -1.0).is_err());
        assert!(world.apply_command(1, 2, Command::ReseedTracers).is_err());
        assert_eq!(world.revision.command_sequence, 0);
        assert_eq!(world.state.fields.density, before);
        world.apply_command(1, 1, Command::Snapshot).unwrap();
        assert!(world.apply_command(1, 1, Command::Snapshot).is_err());
    }
    #[test]
    fn binary_snapshot_has_one_revision_and_only_initial_cold_graph() {
        let mut world = world();
        let parse = |bytes: &[u8]| {
            let start = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
            serde_json::from_slice::<serde_json::Value>(&bytes[start..start + len]).unwrap()
        };
        let first = parse(world.snapshot(7).unwrap());
        assert_eq!(first["graphIncluded"], true);
        assert_eq!(
            first["revision"]["fieldRevision"],
            first["revision"]["surfaceRevision"]
        );
        world.apply_command(1, 1, Command::Snapshot).unwrap();
        let second = parse(world.snapshot(7).unwrap());
        assert_eq!(second["graphIncluded"], false);
        assert_eq!(second["revision"]["commandSequence"], 1);
        assert_eq!(second["receipt"]["commandSequence"], 1);
    }
    #[test]
    fn fixed_topology_advance_conserves_closed_box_volume() {
        let mut world = world();
        let initial = world.receipt().liquid_measure;
        for sequence in 1..=3 {
            world.advance(sequence, 1.0 / 60.0).unwrap();
            assert!(
                world.state.fields.fault.is_none(),
                "{:?}",
                world.state.fields.fault
            );
        }
        assert_eq!(world.revision.frame, 3);
        assert!((world.receipt().liquid_measure - initial).abs() < 1e-4);
        assert!(world
            .state
            .fields
            .face_velocity
            .iter()
            .any(|v| v.abs() > 1e-4));
        assert_eq!(
            world.revision.field_revision,
            world.revision.surface_revision
        );
    }
    #[test]
    fn liquid_command_publishes_once_and_accounts_for_reader_added_mass() {
        let mut world = world();
        let before = world.receipt().liquid_measure;
        world
            .apply_command(
                1,
                1,
                Command::InjectLiquid {
                    drop: crate::injection::LiquidDrop {
                        centre_fine: [6.0, 4.0],
                        radius_fine: 1.5,
                    },
                },
            )
            .unwrap();
        let receipt = world.last_injection.as_ref().unwrap();
        assert!(receipt.accepted);
        assert!(receipt.area_admitted_fine > 0.0);
        assert_eq!(world.revision.injections, 1);
        assert_eq!(world.revision.frame, 0);
        assert_eq!(
            world.revision.surface_revision,
            world.revision.field_revision
        );
        assert!(
            (world.receipt().liquid_measure - before - receipt.area_admitted_fine).abs() < 1e-5
        );
        assert!(world.receipt().drift.abs() < 1e-7);
        let accepted = world.state.fields.density.clone();
        world
            .apply_command(
                2,
                1,
                Command::InjectLiquid {
                    drop: crate::injection::LiquidDrop {
                        centre_fine: [-8.0, -8.0],
                        radius_fine: 1.0,
                    },
                },
            )
            .unwrap();
        assert!(!world.last_injection.as_ref().unwrap().accepted);
        assert_eq!(world.state.fields.density, accepted);
        assert_eq!(world.revision.injections, 1);
    }
    #[test]
    fn production_transport_commits_without_copying_diagnostic_planes() {
        let mut world = world();
        let mut commits = 0;
        let (steps, receipts) = transport_volume_with_commit(
            &world.state.topology.graph,
            &mut world.state.fields,
            1.0 / 60.0,
            false,
            |index, dt, fields| {
                assert_eq!(index, commits);
                assert!(dt > 0.0);
                assert!(fields.density.iter().all(|v| v.is_finite()));
                commits += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(commits, steps);
        assert!(receipts.is_empty());
    }
}
