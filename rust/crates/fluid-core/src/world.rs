//! Owned numerical world and revision-coherent publication boundary.
use crate::embedding::PressureEmbedding;
use crate::lifecycle::{prepare_candidate, LeafArena};
use crate::numerics::{
    assemble_pressure_rhs, enforce_inflow_faces, prepare_pressure_topology,
    prepare_pressure_topology_with_level_set,
    prepare_pressure_topology_with_swept_static_wall_support, project_pressure_velocity,
    solve_pressure,
};
use crate::physical::{publish_final_apertures, PhysicalContext};
use crate::presentation::{
    reconstruct_shared_rdf, AirCellGeometry, RdfSupport, RdfSurface, RdfTopology,
};
use crate::pressure_authority::PressureAuthority;
use crate::publication::{encode_publication, Plane, PlaneId};
use crate::resolution::{
    initialize_resolution_policy, plan_projected_transport_support,
    plan_projected_transport_support_with_surface, plan_resolution,
    plan_resolution_with_surface, ResolutionPolicyOptions, ResolutionPolicyReceipt,
    ResolutionPolicyState,
};
use crate::scalar_authority::{
    publish_scalar_interface_state_from_geometric_density,
    publish_scalar_state_preserving_interface, ScalarAuthority,
};
use crate::scene::{compile_scene_2d, SceneDescription, SceneState};
use crate::sources::SourceLedger;
use crate::topology::BrickSeed;
use crate::tracers::{TracerReceipt, Tracers, TRACER_BUDGET};
use crate::{
    collocate_velocity, extend_velocity, extend_velocity_with_level_set, force_faces,
    force_faces_with_level_set, prepare_faces,
    prepare_faces_for_level_set_volume,
    prepare_faces_for_cellwise_remap, publish_transport_characteristic_clearance,
    refresh_level_set_separating_faces,
    reconstruct_interfaces, reconstruct_interfaces_for_cellwise_remap,
    transport_volume_with_commit, Fields, PressureReceipt, ValidationError,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Use the existing source term only while enforcing the pressure constraint.
/// Restoring it even on failure prevents expansion from becoming injected mass
/// or a refinement demand. Every existing projection must enforce this target;
/// the post-support projection would otherwise erase the primary expansion.
fn with_level_set_volume_pressure_source<T>(
    graph: &mut crate::Graph,
    fields: &mut Fields,
    dt: f32,
    enabled: bool,
    project: impl FnOnce(&mut crate::Graph, &mut Fields) -> Result<T, ValidationError>,
) -> Result<T, ValidationError> {
    if !enabled {
        return project(graph, fields);
    }
    let mut source = crate::numerics::level_set_volume_excess_pressure_source(graph, fields, dt)?;
    for (id, rate) in source.iter_mut().enumerate() {
        *rate += Fields::optional_cell(&fields.source_rate, id, 0.0);
    }
    let physical_source = std::mem::replace(&mut fields.source_rate, source);
    let result = project(graph, fields);
    fields.source_rate = physical_source;
    result
}

fn rebind_violating_separating_walls(
    graph: &mut crate::Graph,
    fields: &mut Fields,
    preprojection_velocity: &[f32],
    rebound: &mut [bool],
) -> usize {
    let mut added = 0;
    for row in &graph.rows {
        if row.kind != crate::RowKind::ClosedWorld || !row.separating {
            continue;
        }
        let Some(term) = row.terms.first() else { continue };
        let inward = if term.coefficient >= 0.0 { 1.0 } else { -1.0 };
        let relative = inward
            * (fields.face_velocity[row.id as usize] - row.solid_velocity);
        if relative < -1.0e-6 && !rebound[row.id as usize] {
            rebound[row.id as usize] = true;
            added += 1;
        }
    }
    if added == 0 {
        return 0;
    }
    fields.face_velocity.clone_from_slice(preprojection_velocity);
    for row in &mut graph.rows {
        if rebound[row.id as usize] {
            row.separating = false;
            fields.face_velocity[row.id as usize] = row.solid_velocity;
        }
    }
    added
}

#[derive(Clone, Copy, Debug)]
struct NativeStageClock {
    #[cfg(not(target_arch = "wasm32"))]
    started: std::time::Instant,
}

impl NativeStageClock {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            started: std::time::Instant::now(),
        }
    }

    fn elapsed_nanoseconds(self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            return self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldStageTimings {
    /// Timings use the native monotonic clock. Wasm receipts explicitly mark
    /// them unavailable rather than presenting zeros as measured work.
    pub available: bool,
    pub field_build: u64,
    pub primary_pressure: u64,
    pub support_planning_transfer: u64,
    pub post_support_pressure: u64,
    pub transport: u64,
    pub post_transport: u64,
    pub resolution_publication: u64,
    pub other: u64,
    pub total_advance: u64,
}

impl WorldStageTimings {
    fn finish(&mut self, total_clock: NativeStageClock) {
        self.total_advance = total_clock.elapsed_nanoseconds();
        let attributed = self
            .field_build
            .saturating_add(self.primary_pressure)
            .saturating_add(self.support_planning_transfer)
            .saturating_add(self.post_support_pressure)
            .saturating_add(self.transport)
            .saturating_add(self.post_transport)
            .saturating_add(self.resolution_publication);
        self.other = self.total_advance.saturating_sub(attributed);
    }
}

/// Selects the experimental 2D volume-transport path. The baseline remains
/// the production transport unless a lab caller opts in explicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TransportExperiment {
    #[default]
    Baseline,
    /// Measure the cellwise construction, then advance material with the
    /// baseline transport so multi-frame diagnostics follow the current path.
    CellwiseProbe,
    /// Commit material with one cellwise whole-step remap.
    CellwiseRemap,
    /// Conservative volume coupled to an advected signed-distance surface.
    LevelSetVolume,
    /// Configured cellwise run used to sweep fixed RK4 segment counts after
    /// the M2 trajectory-crossing stop. This remains one WorldOptions field.
    Configured {
        mode: CellwiseTransportMode,
        trace_segments: usize,
        edge_samples: usize,
        closure: crate::adaptive_remap::CellwiseClosure,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CellwiseTransportMode {
    Probe,
    Remap,
}

impl TransportExperiment {
    pub fn configured(mode: CellwiseTransportMode, trace_segments: usize) -> Self {
        Self::configured_with_samples(mode, trace_segments, 1)
    }

    pub fn configured_with_samples(
        mode: CellwiseTransportMode,
        trace_segments: usize,
        edge_samples: usize,
    ) -> Self {
        Self::configured_with_closure(
            mode,
            trace_segments,
            edge_samples,
            crate::adaptive_remap::CellwiseClosure::BandProjection,
        )
    }

    pub fn configured_with_closure(
        mode: CellwiseTransportMode,
        trace_segments: usize,
        edge_samples: usize,
        closure: crate::adaptive_remap::CellwiseClosure,
    ) -> Self {
        Self::Configured {
            mode,
            trace_segments,
            edge_samples,
            closure,
        }
    }

    pub fn cellwise_mode(self) -> Option<CellwiseTransportMode> {
        match self {
            Self::Baseline | Self::LevelSetVolume => None,
            Self::CellwiseProbe => Some(CellwiseTransportMode::Probe),
            Self::CellwiseRemap => Some(CellwiseTransportMode::Remap),
            Self::Configured { mode, .. } => Some(mode),
        }
    }
    pub fn is_levelset_volume(self) -> bool {
        matches!(self, Self::LevelSetVolume)
    }

    pub fn trace_segments(self) -> usize {
        match self {
            Self::Configured { trace_segments, .. } => trace_segments,
            _ => 1,
        }
    }

    pub fn edge_samples(self) -> usize {
        match self {
            Self::Configured { edge_samples, .. } => edge_samples,
            _ => 1,
        }
    }

    pub fn closure(self) -> crate::adaptive_remap::CellwiseClosure {
        match self {
            Self::Configured { closure, .. } => closure,
            _ => crate::adaptive_remap::CellwiseClosure::BandProjection,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum TransportExperimentName {
    Baseline,
    CellwiseProbe,
    CellwiseRemap,
    LevelSetVolume,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfiguredTransportExperiment {
    mode: TransportExperimentName,
    trace_segments: usize,
    #[serde(default = "one_usize", skip_serializing_if = "usize_is_one")]
    edge_samples: usize,
    #[serde(default, skip_serializing_if = "band_projection_closure")]
    closure: crate::adaptive_remap::CellwiseClosure,
}

fn one_usize() -> usize {
    1
}

fn usize_is_one(value: &usize) -> bool {
    *value == 1
}

fn band_projection_closure(value: &crate::adaptive_remap::CellwiseClosure) -> bool {
    *value == crate::adaptive_remap::CellwiseClosure::BandProjection
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TransportExperimentRepresentation {
    Name(TransportExperimentName),
    Configured(ConfiguredTransportExperiment),
}

impl Serialize for TransportExperiment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let name = match self {
            Self::Baseline => TransportExperimentName::Baseline,
            Self::LevelSetVolume => TransportExperimentName::LevelSetVolume,
            Self::CellwiseProbe => TransportExperimentName::CellwiseProbe,
            Self::CellwiseRemap => TransportExperimentName::CellwiseRemap,
            Self::Configured { mode: CellwiseTransportMode::Probe, .. } => TransportExperimentName::CellwiseProbe,
            Self::Configured { mode: CellwiseTransportMode::Remap, .. } => TransportExperimentName::CellwiseRemap,
        };
        if matches!(self, Self::Configured { .. }) {
            ConfiguredTransportExperiment {
                mode: name,
                trace_segments: self.trace_segments(),
                edge_samples: self.edge_samples(),
                closure: self.closure(),
            }
            .serialize(serializer)
        } else {
            name.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for TransportExperiment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let representation = TransportExperimentRepresentation::deserialize(deserializer)?;
        match representation {
            TransportExperimentRepresentation::Name(TransportExperimentName::Baseline) => {
                Ok(Self::Baseline)
            }
            TransportExperimentRepresentation::Name(TransportExperimentName::CellwiseProbe) => {
                Ok(Self::CellwiseProbe)
            }
            TransportExperimentRepresentation::Name(TransportExperimentName::CellwiseRemap) => {
                Ok(Self::CellwiseRemap)
            }
            TransportExperimentRepresentation::Name(TransportExperimentName::LevelSetVolume) => Ok(Self::LevelSetVolume),
            TransportExperimentRepresentation::Configured(config) => {
                let mode = match config.mode {
                    TransportExperimentName::Baseline => {
                        return Err(serde::de::Error::custom(
                            "baseline transport does not accept traceSegments",
                        ))
                    }
                    TransportExperimentName::LevelSetVolume => return Err(serde::de::Error::custom(
                        "level-set-volume transport does not accept cellwise options",
                    )),
                    TransportExperimentName::CellwiseProbe => CellwiseTransportMode::Probe,
                    TransportExperimentName::CellwiseRemap => CellwiseTransportMode::Remap,
                };
                if !(1..=128).contains(&config.trace_segments) {
                    return Err(serde::de::Error::custom(
                        "cellwise traceSegments must be between 1 and 128",
                    ));
                }
                if !matches!(config.edge_samples, 1 | 2 | 4) {
                    return Err(serde::de::Error::custom(
                        "cellwise edgeSamples must be 1, 2, or 4",
                    ));
                }
                Ok(Self::configured_with_closure(
                    mode,
                    config.trace_segments,
                    config.edge_samples,
                    config.closure,
                ))
            }
        }
    }
}

fn baseline_transport_experiment(value: &TransportExperiment) -> bool {
    *value == TransportExperiment::Baseline
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WorldOptions {
    pub run_epoch: u32,
    pub command_sequence: u32,
    pub pressure_iterations: u32,
    pub pressure_relative_tolerance: f32,
    pub tracer_budget: usize,
    pub topology_page_budget: Option<u32>,
    #[serde(skip_serializing_if = "baseline_transport_experiment")]
    pub transport_experiment: TransportExperiment,
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
            transport_experiment: TransportExperiment::Baseline,
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
    pub stage_timings: &'a WorldStageTimings,
    pub microsteps: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cellwise_remap: Option<&'a crate::adaptive_remap::CellwiseRemapReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_set_volume: Option<&'a crate::levelset_volume::LevelSetVolumeReceipt>,
    pub interface_seams: crate::levelset_volume::InterfaceSeamReceipt,
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

fn commit_source_ledger(
    ledger: &mut SourceLedger,
    dt: f32,
    fields: &mut Fields,
) -> Result<(), ValidationError> {
    let receipt = ledger
        .commit_microstep(dt as f64)
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
}

pub struct World {
    pub state: SceneState<2>,
    pub revision: Revision,
    pub options: WorldOptions,
    pub pressure: PressureReceipt,
    pub stage_timings: WorldStageTimings,
    pub source_ledger: SourceLedger,
    pub microsteps: usize,
    pub cellwise_remap_receipt: Option<crate::adaptive_remap::CellwiseRemapReceipt>,
    pub level_set_volume_receipt: Option<crate::levelset_volume::LevelSetVolumeReceipt>,
    pub level_set_phi: Vec<f32>,
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
        let initial_surface = if options.transport_experiment.is_levelset_volume() {
            let volume = bundle.state.topology.graph.cells.iter()
                .map(|cell| {
                    bundle.state.fields.density[cell.id as usize] as f64 * cell.measure as f64
                })
                .sum();
            Some(crate::levelset_surface::initialize_from_document(
                &bundle.document, volume,
            )?)
        } else {
            None
        };
        let mut world = Self::from_state_with_surface(bundle.state, options, initial_surface)?;
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
        state: SceneState<2>,
        options: WorldOptions,
    ) -> Result<Self, ValidationError> {
        Self::from_state_with_surface(state, options, None)
    }
    fn from_state_with_surface(
        mut state: SceneState<2>,
        options: WorldOptions,
        initial_surface: Option<RdfSurface>,
    ) -> Result<Self, ValidationError> {
        if options.run_epoch == 0 {
            return Err(ValidationError("runEpoch must be positive".into()));
        }
        validate_pressure_options(
            options.pressure_iterations,
            options.pressure_relative_tolerance,
        )?;
        if options.transport_experiment.cellwise_mode().is_some() {
            if !(1..=128).contains(&options.transport_experiment.trace_segments()) {
                return Err(ValidationError(
                    "cellwise traceSegments must be between 1 and 128".into(),
                ));
            }
            if !matches!(options.transport_experiment.edge_samples(), 1 | 2 | 4) {
                return Err(ValidationError(
                    "cellwise edgeSamples must be 1, 2, or 4".into(),
                ));
            }
        }
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
        if options.transport_experiment.is_levelset_volume() {
            state.fields.interface_normal.fill(0.0);
            state.fields.interface_offset.fill(0.0);
        } else {
            reconstruct_interfaces(&state.topology.graph, &mut state.fields)?;
        }
        let rdf_topology = RdfTopology::compile(&state.topology.graph)?;
        let rdf_support = Self::support_for(&state);
        let surface = if options.transport_experiment.is_levelset_volume() {
            match initial_surface {
                Some(surface) => surface,
                None => crate::levelset_surface::initialize_from_volume(
                    &state.topology.graph,
                    &state.fields,
                )?,
            }
        } else {
            reconstruct_shared_rdf(
                &state.topology.graph,
                &state.fields,
                &rdf_topology,
                &rdf_support,
            )?
        };
        let level_set_phi = if options.transport_experiment.is_levelset_volume() {
            crate::levelset_surface::cell_phi(&state.topology.graph, &surface)?
        } else {
            Vec::new()
        };
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
            stage_timings: WorldStageTimings {
                available: !cfg!(target_arch = "wasm32"),
                ..Default::default()
            },
            source_ledger: SourceLedger::default(),
            microsteps: 1,
            cellwise_remap_receipt: None,
            level_set_volume_receipt: None,
            level_set_phi,
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
            stage_timings: &self.stage_timings,
            microsteps: self.microsteps,
            cellwise_remap: self.cellwise_remap_receipt.as_ref(),
            level_set_volume: self.level_set_volume_receipt.as_ref(),
            interface_seams: if self.options.transport_experiment.is_levelset_volume() {
                crate::levelset_volume::interface_seam_receipt_from_phi(
                    &self.state.topology.graph,
                    &self.state.fields,
                    &self.level_set_phi,
                )
            } else {
                crate::levelset_volume::interface_seam_receipt(
                    &self.state.topology.graph,
                    &self.state.fields,
                )
            },
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
        let total_clock = NativeStageClock::start();
        self.stage_timings = WorldStageTimings {
            available: !cfg!(target_arch = "wasm32"),
            ..Default::default()
        };
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
        let level_set_volume = self.options.transport_experiment.is_levelset_volume();
        if level_set_volume && self.physical.as_ref().is_some_and(|physical| {
            !physical.scene.rigid_bodies.is_empty() || physical.scene.fluid.inflow.is_some()
        }) {
            return Err(ValidationError(
                "level-set-volume does not support rigid bodies or inflow sources".into(),
            ));
        }
        let field_build_clock = NativeStageClock::start();
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
        let cellwise_remap = matches!(
            self.options.transport_experiment.cellwise_mode(),
            Some(CellwiseTransportMode::Remap)
        );
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            observe("dynamic-geometry", graph, fields);
            if level_set_volume {
                extend_velocity_with_level_set(graph, fields, &self.level_set_phi, 8)?;
            } else {
                extend_velocity(graph, fields, 8)?;
            }
            observe("transport-velocity-extension", graph, fields);
            if level_set_volume {
                prepare_faces_for_level_set_volume(graph, fields, dt)?;
            } else if cellwise_remap {
                prepare_faces_for_cellwise_remap(graph, fields, dt)?;
            } else {
                prepare_faces(graph, fields, dt)?;
            }
            observe("face-preparation", graph, fields);
        }
        self.view_history
            .capture_faces(&self.state, self.cell_size());
        let swept_static_wall_pressure = cellwise_remap;
        {
            let graph = &mut self.state.topology.graph;
            let fields = &mut self.state.fields;
            if level_set_volume {
                force_faces_with_level_set(
                    graph,
                    fields,
                    &self.level_set_phi,
                    dt,
                    fields.acceleration_fine,
                    inflow,
                );
            } else {
                force_faces(graph, fields, dt, fields.acceleration_fine, inflow);
            }
            observe("body-forces", graph, fields);
            if level_set_volume {
                crate::levelset_volume::publish_pressure_geometry_from_phi(
                    graph,
                    fields,
                    &self.level_set_phi,
                )?;
            } else if cellwise_remap {
                reconstruct_interfaces_for_cellwise_remap(graph, fields)?;
            } else {
                reconstruct_interfaces(graph, fields)?;
            }
            observe("interface-reconstruction", graph, fields);
            self.stage_timings.field_build = field_build_clock.elapsed_nanoseconds();
            let primary_pressure_clock = NativeStageClock::start();
            with_level_set_volume_pressure_source(graph, fields, dt, level_set_volume, |graph, fields| {
                let preprojection_velocity = if level_set_volume {
                    fields.face_velocity.clone()
                } else {
                    Vec::new()
                };
                let mut rebound = if level_set_volume {
                    vec![false; graph.rows.len()]
                } else {
                    Vec::new()
                };
                let mut accumulated_iterations = 0;
                loop {
                if let Some(embedding) = &mut self.embedding {
                    let prepared = if level_set_volume {
                        embedding.prepare_with_level_set(graph, fields, &self.level_set_phi)?
                    } else {
                        embedding.prepare(graph, fields)
                    };
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
                        iterations: accumulated_iterations + solved.solve.iterations,
                        initial_residual: solved.solve.initial_true_residual_squared.max(0.0).sqrt(),
                        residual: solved.solve.final_true_residual_squared.max(0.0).sqrt(),
                        converged: solved.solve.converged,
                    };
                    accumulated_iterations = self.pressure.iterations;
                    observe("pressure-solve", graph, fields);
                    embedding.project(graph, fields, &solved.prepared);
                } else {
                    // Swept wall-contact support is a single transient pressure
                    // solve.  Restore the physical membership before support
                    // planning so promoted dry cells cannot survive a topology
                    // transfer or apply the contact impulse a second time.
                    let physical_rows = if level_set_volume {
                        prepare_pressure_topology_with_level_set(graph, fields, &self.level_set_phi)?
                    } else {
                        prepare_pressure_topology(graph, fields)
                    };
                    let physical_pressure_member = fields.pressure_member.clone();
                    let physical_pressure_row_member = fields.pressure_row_member.clone();
                    let physical_pressure_diagonal = fields.pressure_diagonal.clone();
                    let rows = if swept_static_wall_pressure {
                        prepare_pressure_topology_with_swept_static_wall_support(graph, fields)
                    } else {
                        physical_rows
                    };
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
                    let solved = solve_pressure(
                        graph,
                        fields,
                        &rows,
                        self.options.pressure_iterations,
                        self.options.pressure_relative_tolerance,
                        Some(&self.pressure_authority.execution_order),
                    )?;
                    self.pressure = PressureReceipt {
                        iterations: accumulated_iterations + solved.iterations,
                        ..solved
                    };
                    accumulated_iterations = self.pressure.iterations;
                    observe("pressure-solve", graph, fields);
                    project_pressure_velocity(graph, fields, &rows);
                    if swept_static_wall_pressure {
                        fields
                            .pressure_member
                            .clone_from(&physical_pressure_member);
                        fields
                            .pressure_row_member
                            .clone_from(&physical_pressure_row_member);
                        fields
                            .pressure_diagonal
                            .clone_from(&physical_pressure_diagonal);
                    }
                }
                if !level_set_volume
                    || rebind_violating_separating_walls(
                        graph,
                        fields,
                        &preprojection_velocity,
                        &mut rebound,
                    ) == 0
                {
                    break;
                }
                }
                Ok(())
            })?;
            enforce_inflow_faces(graph, fields, inflow);
            collocate_velocity(graph, fields);
            observe("velocity-projection", graph, fields);
            self.stage_timings.primary_pressure = primary_pressure_clock.elapsed_nanoseconds();
        }
        let support_clock = NativeStageClock::start();
        let support = if level_set_volume {
            let mut options = self.resolution_options.clone();
            options.coarsest_demanded_pages = true;
            options.coarsen_inactive_pages = true;
            options.maximum_leaves = Some(self.arena.maximum_slice_leaves);
            options.maximum_cells = Some(self.arena.capacity as usize * 64);
            options.free_leaf_ids.clone_from(&self.arena.free_leaf_ids);
            plan_projected_transport_support_with_surface(
                &self.state.topology,
                &self.state.fields,
                dt_s,
                self.cell_size(),
                &options,
                true,
                &self.surface,
            )
        } else {
            plan_projected_transport_support(
                &self.state.topology,
                &self.state.fields,
                dt_s,
                self.cell_size(),
                &self.resolution_options.policy,
                Some(self.arena.maximum_slice_leaves),
                Some(self.arena.capacity as usize * 64),
                &self.arena.free_leaf_ids,
                cellwise_remap,
            )
        }
        .map_err(|e| ValidationError(format!("projected support: {e:?}")))?;
        if support.fault_bits != 0 {
            return Err(ValidationError(format!(
                "projected support fault {}",
                support.fault_bits
            )));
        }
        if self.bricks_changed(&support.candidate_bricks) {
            self.transition(support.candidate_bricks, dt_s)?;
            if level_set_volume {
                extend_velocity_with_level_set(
                    &self.state.topology.graph,
                    &mut self.state.fields,
                    &self.level_set_phi,
                    8,
                )?;
            } else {
                extend_velocity(&self.state.topology.graph, &mut self.state.fields, 8)?;
            }
            if cellwise_remap {
                reconstruct_interfaces_for_cellwise_remap(
                    &self.state.topology.graph,
                    &mut self.state.fields,
                )?;
            }
            if level_set_volume {
                crate::levelset_volume::publish_pressure_geometry_from_phi(
                    &self.state.topology.graph,
                    &mut self.state.fields,
                    &self.level_set_phi,
                )?;
            }
            if cellwise_remap || level_set_volume {
                let post_support_pressure_clock = NativeStageClock::start();
                // Projected-support transfer conserves material and momentum, but
                // the interpolated candidate face field is not discretely
                // divergence-free on its new rows. Geometric whole-step transport
                // consumes this generation immediately, so close the candidate
                // field with the same pressure operator before tracing it. Body
                // forces and face advection are already present in the transferred
                // field and must not be applied a second time.
                let graph = &mut self.state.topology.graph;
                let fields = &mut self.state.fields;
                if level_set_volume {
                    // Topology compilation clears this transient contact state.
                    // Reclassify from transferred velocity without adding force.
                    refresh_level_set_separating_faces(
                        graph,
                        fields,
                        &self.level_set_phi,
                        dt,
                        fields.acceleration_fine,
                    );
                }
                with_level_set_volume_pressure_source(graph, fields, dt, level_set_volume, |graph, fields| {
                    let preprojection_velocity = if level_set_volume {
                        fields.face_velocity.clone()
                    } else {
                        Vec::new()
                    };
                    let mut rebound = if level_set_volume {
                        vec![false; graph.rows.len()]
                    } else {
                        Vec::new()
                    };
                    let mut accumulated_iterations = 0;
                    loop {
                    if let Some(embedding) = &mut self.embedding {
                        let prepared = if level_set_volume {
                            embedding.prepare_with_level_set(graph, fields, &self.level_set_phi)?
                        } else {
                            embedding.prepare(graph, fields)
                        };
                        fields.pressure_diagonal.clone_from(&prepared.diagonal);
                        fields.pressure_rhs.clone_from(&prepared.rhs);
                        let solved = embedding.solve(
                            graph,
                            fields,
                            self.options.pressure_iterations,
                            self.options.pressure_relative_tolerance,
                            Some(prepared),
                        )?;
                        self.pressure = PressureReceipt {
                            iterations: accumulated_iterations + solved.solve.iterations,
                            initial_residual: solved
                                .solve
                                .initial_true_residual_squared
                                .max(0.0)
                                .sqrt(),
                            residual: solved.solve.final_true_residual_squared.max(0.0).sqrt(),
                            converged: solved.solve.converged,
                        };
                        accumulated_iterations = self.pressure.iterations;
                        embedding.project(graph, fields, &solved.prepared);
                    } else {
                        let rows = if level_set_volume {
                            prepare_pressure_topology_with_level_set(graph, fields, &self.level_set_phi)?
                        } else {
                            prepare_pressure_topology(graph, fields)
                        };
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
                        assemble_pressure_rhs(graph, fields, &rows);
                        let solved = solve_pressure(
                            graph,
                            fields,
                            &rows,
                            self.options.pressure_iterations,
                            self.options.pressure_relative_tolerance,
                            Some(&self.pressure_authority.execution_order),
                        )?;
                        self.pressure = PressureReceipt {
                            iterations: accumulated_iterations + solved.iterations,
                            ..solved
                        };
                        accumulated_iterations = self.pressure.iterations;
                        project_pressure_velocity(graph, fields, &rows);
                    }
                    if !level_set_volume
                        || rebind_violating_separating_walls(
                            graph,
                            fields,
                            &preprojection_velocity,
                            &mut rebound,
                        ) == 0
                    {
                        break;
                    }
                    }
                    Ok(())
                })?;
                enforce_inflow_faces(graph, fields, inflow);
                collocate_velocity(graph, fields);
                observe("projected-support-velocity-projection", graph, fields);
                self.stage_timings.post_support_pressure =
                    post_support_pressure_clock.elapsed_nanoseconds();
            }
        }
        self.stage_timings.support_planning_transfer = support_clock
            .elapsed_nanoseconds()
            .saturating_sub(self.stage_timings.post_support_pressure);
        let transport_clock = NativeStageClock::start();
        if level_set_volume {
            extend_velocity_with_level_set(
                &self.state.topology.graph,
                &mut self.state.fields,
                &self.level_set_phi,
                8,
            )?;
            observe(
                "level-set-volume-velocity-extension",
                &self.state.topology.graph,
                &self.state.fields,
            );
        }
        self.view_history.capture_density(&self.state);
        let source_density = self.state.fields.density.clone();
        let source_gamma = self.state.fields.gamma.clone();
        let transport_fault = {
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
            self.cellwise_remap_receipt = None;
            self.level_set_volume_receipt = None;
            let experiment = self.options.transport_experiment;
            if experiment.is_levelset_volume() {
                let (surface, receipt) = crate::levelset_volume::advance_with_fine_capacity(
                    graph,
                    fields,
                    &self.surface,
                    &self.rdf_topology,
                    &self.rdf_support,
                    &mut self.level_set_phi,
                    &self.capacity_fine,
                    dt,
                )?;
                self.surface = surface;
                self.level_set_volume_receipt = Some(receipt);
                self.microsteps = 0;
            } else { match experiment.cellwise_mode() {
                None => {
                    let (steps, _) = transport_volume_with_commit(
                        graph,
                        fields,
                        dt,
                        false,
                        |_, dtm, fields| commit_source_ledger(ledger, dtm, fields),
                    )?;
                    self.microsteps = steps;
                }
                Some(CellwiseTransportMode::Probe) => {
                    let options = crate::adaptive_remap::CellwiseRemapOptions {
                        closure: experiment.closure(),
                        commit_material: false,
                        trace_segments: experiment.trace_segments(),
                        edge_samples: experiment.edge_samples(),
                        ..Default::default()
                    };
                    self.cellwise_remap_receipt = Some(
                        crate::adaptive_remap::transport_volume_cellwise_with_commit(
                            graph,
                            fields,
                            dt,
                            options,
                            |_, _| Ok(()),
                        )?,
                    );
                    let (steps, _) = transport_volume_with_commit(
                        graph,
                        fields,
                        dt,
                        false,
                        |_, dtm, fields| commit_source_ledger(ledger, dtm, fields),
                    )?;
                    self.microsteps = steps;
                }
                Some(CellwiseTransportMode::Remap) => {
                    let options = crate::adaptive_remap::CellwiseRemapOptions {
                        closure: experiment.closure(),
                        trace_segments: experiment.trace_segments(),
                        edge_samples: experiment.edge_samples(),
                        ..Default::default()
                    };
                    self.cellwise_remap_receipt = Some(
                        crate::adaptive_remap::transport_volume_cellwise_with_commit(
                            graph,
                            fields,
                            dt,
                            options,
                            |dtm, fields| commit_source_ledger(ledger, dtm, fields),
                        )?,
                    );
                    self.microsteps = 0;
                }
            }}
            publish_final_apertures(graph, fields);
            observe("conservative-transport", graph, fields);
            self.tracer_receipt = self.tracers.advance(graph, fields, dt)?;
            fields.fault.clone()
        };
        self.stage_timings.transport = transport_clock.elapsed_nanoseconds();
        let post_transport_clock = NativeStageClock::start();
        let has_rigid_bodies = self.physical
            .as_ref()
            .is_some_and(|p| !p.scene.rigid_bodies.is_empty());
        if level_set_volume {
            publish_scalar_state_preserving_interface(
                &mut self.scalar_authority,
                &self.state.topology,
                &mut self.state.fields,
                &source_density,
                &source_gamma,
                self.revision.frame + 1,
                self.topology_slot,
                has_rigid_bodies,
            )?;
            crate::levelset_volume::publish_pressure_geometry_from_phi(
                &self.state.topology.graph,
                &mut self.state.fields,
                &self.level_set_phi,
            )?;
        } else {
            publish_scalar_interface_state_from_geometric_density(
                &mut self.scalar_authority,
                &self.state.topology,
                &mut self.state.fields,
                &source_density,
                &source_gamma,
                self.revision.frame + 1,
                self.topology_slot,
                has_rigid_bodies,
            )?;
        }
        if cellwise_remap {
            reconstruct_interfaces_for_cellwise_remap(
                &self.state.topology.graph,
                &mut self.state.fields,
            )?;
        }
        observe(
            "scalar-publication",
            &self.state.topology.graph,
            &self.state.fields,
        );
        if let Some(physical) = &mut self.physical {
            physical.finish_frame(&self.state)?;
        }
        self.stage_timings.post_transport = post_transport_clock.elapsed_nanoseconds();
        let resolution_clock = NativeStageClock::start();
        let mut policy = self.resolution_options.clone();
        // Geometric support closes both sides of a coarse cell. The remap
        // experiment uses relative motion; direct phi uses the GPU policy's
        // absolute liquid-speed floor inside the shared measurement routine.
        policy.translation_invariant_motion_sizing = cellwise_remap || level_set_volume;
        policy.coarsen_inactive_pages = level_set_volume;
        policy.coarsest_demanded_pages = level_set_volume;
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
        let decision = if level_set_volume {
            plan_resolution_with_surface(
                &self.state.topology,
                &self.state.fields,
                &self.resolution_policy,
                dt_s,
                self.cell_size(),
                &policy,
                &self.surface,
            )
        } else {
            plan_resolution(
                &self.state.topology,
                &self.state.fields,
                &self.resolution_policy,
                dt_s,
                self.cell_size(),
                &policy,
            )
        }
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
        self.stage_timings.resolution_publication = resolution_clock.elapsed_nanoseconds();
        // Candidate topology construction initializes its fault plane. Keep a
        // rejected transport visible through the completed World advance even
        // when the resolution policy accepts a later generation in the same
        // frame.
        if transport_fault.is_some() {
            self.state.fields.fault = transport_fault;
        }
        self.revision.frame += 1;
        self.revision.time += dt_s;
        self.revision.command_sequence = sequence;
        self.revision.field_revision += 1;
        self.revision.surface_revision = self.revision.field_revision;
        self.stage_timings.finish(total_clock);
        Ok(())
    }
    pub fn inject_liquid(
        &mut self,
        drop: crate::injection::LiquidDrop,
    ) -> Result<(), ValidationError> {
        use crate::injection::{
            addressable, apply_dose, demanded_bricks, requested_area, InjectionReceipt,
        };
        let level_set_volume = self.options.transport_experiment.is_levelset_volume();
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
        // The level-set experiment positions its interface with a shared vertex
        // scalar rather than with per-cell volume, so the ball belongs in that
        // scalar before the plan measures which pages the new interface crosses.
        // Published here and committed only once the transfer has been accepted:
        // a refused drop must not leave behind an interface the volume never saw.
        let injected_surface = if level_set_volume {
            Some(crate::levelset_surface::union_drop(
                &self.surface,
                drop,
                self.surface.receipt.exact_area_fine,
            )?)
        } else {
            None
        };
        let mut options = self.resolution_options.clone();
        options.injection_demanded_brick_keys = demand;
        options.maximum_leaves = Some(self.arena.maximum_slice_leaves);
        options.maximum_cells = Some(self.arena.capacity as usize * 64);
        options.free_leaf_ids.clone_from(&self.arena.free_leaf_ids);
        options.moving_rigid_bodies = self
            .physical
            .as_ref()
            .is_some_and(|p| !p.scene.rigid_bodies.is_empty());
        if level_set_volume {
            // The page policy the level-set advance plans under, so the drop's
            // generation is not one the next step immediately undoes.
            options.translation_invariant_motion_sizing = true;
            options.coarsen_inactive_pages = true;
            options.coarsest_demanded_pages = true;
        }
        // The intervention planner observes collocated motion; projected face
        // demand belongs to the advance's pre-transport transaction.
        let mut planning = self.state.fields.clone();
        planning.face_velocity.clear();
        planning.acceleration_fine = [0.0; 3];
        let decision = match injected_surface.as_ref() {
            Some(surface) => plan_resolution_with_surface(
                &self.state.topology,
                &planning,
                &self.resolution_policy,
                self.timestep_s,
                self.cell_size(),
                &options,
                surface,
            ),
            None => plan_resolution(
                &self.state.topology,
                &planning,
                &self.resolution_policy,
                self.timestep_s,
                self.cell_size(),
                &options,
            ),
        }
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
        if let Some(surface) = injected_surface {
            // Accepted, so the ball now stands in both authorities. The derived
            // cell scalar and the pressure planes are rebuilt from the surface
            // for the same reason the advance rebuilds them: nothing downstream
            // may read a phi the published contour no longer agrees with.
            self.surface = surface;
            self.level_set_phi = crate::levelset_surface::cell_phi(
                &self.state.topology.graph,
                &self.surface,
            )?;
            crate::levelset_volume::publish_pressure_geometry_from_phi(
                &self.state.topology.graph,
                &mut self.state.fields,
                &self.level_set_phi,
            )?;
        } else {
            reconstruct_interfaces(&self.state.topology.graph, &mut self.state.fields)?;
        }
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
        let (staged, arena) = if self.options.transport_experiment.is_levelset_volume() {
            crate::lifecycle::prepare_candidate_allow_overcapacity(&self.state, candidate, &self.arena)
        } else {
            prepare_candidate(&self.state, candidate, &self.arena)
        }
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
        if self.options.transport_experiment.is_levelset_volume() {
            self.level_set_phi = crate::levelset_surface::cell_phi(
                &self.state.topology.graph,
                &self.surface,
            )?;
        }
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
        if self.options.transport_experiment.is_levelset_volume() {
            let diagnostic_volume = self.state.topology.graph.cells.iter().map(|cell| {
                self.state.fields.density[cell.id as usize] as f64 * cell.measure as f64
            }).sum();
            self.surface = crate::levelset_surface::refresh(&self.surface, diagnostic_volume)?;
            let cell_size = self.cell_size();
            crate::resolution::publish_direct_surface_proofs(
                &self.state.topology, &self.state.fields, &self.surface,
                &self.resolution_options, &mut self.resolution_policy,
                self.timestep_s, cell_size,
            ).map_err(|error| ValidationError(format!("surface proof: {error:?}")))?;
        } else {
            self.surface = reconstruct_shared_rdf(
                &self.state.topology.graph,
                &self.state.fields,
                &self.rdf_topology,
                &self.rdf_support,
            )?;
        }
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
    fn level_set_world() -> World {
        let scene: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":2,"dimensions":[8,8,1],"cellSizeM":0.05,
            "dtS":1.0/60.0,"densityKgM3":998.2,"gravityMS2":[0.0,0.0,0.0],
            "boundaries":["closed","closed","closed","closed","closed","closed"],
            "bricks":[{"id":0,"key":0,"coordinate":[0,0,0],"spanBricks":1,"resolution":8,"active":true}],
            "rasterDensity":(0..64).map(|i|if i%8<4{1.0}else{0.0}).collect::<Vec<_>>()
        })).unwrap();
        World::from_scene(
            scene,
            WorldOptions {
                transport_experiment: TransportExperiment::LevelSetVolume,
                tracer_budget: 0,
                ..WorldOptions::default()
            },
        )
        .unwrap()
    }
    fn level_set_uniform_bulk_world(speed_y: f32) -> World {
        let bricks: Vec<_> = (0..3).flat_map(|y| (0..3).map(move |x| {
            let key = (x + 3 * y) as u32;
            serde_json::json!({
                "id":key,"key":key,"coordinate":[x,y,0],"spanBricks":1,
                "resolution":if key == 4 { 2 } else { 4 },"active":true
            })
        })).collect();
        let raster_velocity: Vec<_> = (0..24 * 24)
            .flat_map(|_| [0.0, speed_y]).collect();
        let scene: SceneDescription = serde_json::from_value(serde_json::json!({
            "schemaVersion":1,"dimension":2,"dimensions":[24,24,1],"cellSizeM":0.05,
            "dtS":1.0/30.0,"densityKgM3":998.2,"gravityMS2":[0.0,0.0,0.0],
            "boundaries":["open","open","open","open","open","open"],
            "bricks":bricks,"rasterDensity":vec![1.0;24*24],
            "rasterVelocity":raster_velocity
        })).unwrap();
        World::from_scene(
            scene,
            WorldOptions {
                transport_experiment: TransportExperiment::LevelSetVolume,
                tracer_budget: 0,
                ..WorldOptions::default()
            },
        ).unwrap()
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
    fn level_set_zero_motion_preserves_contour_and_reinitializes_narrow_band() {
        let mut world = level_set_world();
        let zero_vertices: Vec<_> = world
            .surface
            .vertex_phi_fine
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value == 0.0).then_some(index))
            .collect();
        let segments = world.surface.segments_fine.clone();
        world.advance(1, 1.0 / 60.0).unwrap();
        assert_eq!(world.surface.segments_fine, segments);
        for index in zero_vertices {
            assert_eq!(world.surface.vertex_phi_fine[index], 0.0);
            assert_eq!(world.surface.vertex_phi_fine[index - 1], -1.0);
            assert_eq!(world.surface.vertex_phi_fine[index - 2], -2.0);
            assert_eq!(world.surface.vertex_phi_fine[index + 1], 1.0);
            assert_eq!(world.surface.vertex_phi_fine[index + 2], 2.0);
        }
    }
    #[test]
    fn resting_level_set_consumes_published_surface_proofs() {
        let mut world = level_set_world();
        let original_contour = world.surface.segments_fine.clone();
        let original_cells = world.state.topology.graph.cells.len();
        let mut demoted = false;
        for frame in 1..=6 {
            world.advance(frame, 1.0 / 60.0).unwrap();
            demoted |= world.resolution_receipt.as_ref().unwrap().demoted_brick_count > 0;
            assert_eq!(world.surface.segments_fine, original_contour);
        }
        assert!(demoted, "accepted surface certificates must reach the planner");
        assert!(world.state.topology.graph.cells.len() < original_cells);
    }

    #[test]
    fn level_set_topology_transition_preserves_direct_surface_exactly() {
        let mut world = level_set_world();
        let vertices = world.surface.vertex_phi_fine.clone();
        let segments = world.surface.segments_fine.clone();
        let mut bricks: Vec<_> = world.state.topology.bricks.iter().map(|brick| brick.seed.clone()).collect();
        bricks[0].resolution = 4;
        world.transition(bricks, 1.0 / 60.0).unwrap();
        assert_eq!(world.state.topology.graph.topology_generation, 2);
        assert_eq!(world.surface.vertex_phi_fine, vertices);
        assert_eq!(world.surface.segments_fine, segments);
        assert_eq!(world.level_set_phi.len(), world.state.topology.graph.cells.len());
    }
    #[test]
    fn level_set_uniform_translation_preserves_coarse_bulk_resolution() {
        let mut stationary = level_set_uniform_bulk_world(0.0);
        let mut falling = level_set_uniform_bulk_world(-200.0);
        stationary.advance(1, 1.0 / 30.0).unwrap();
        falling.advance(1, 1.0 / 30.0).unwrap();

        let centre = |world: &World| {
            world.resolution_receipt.as_ref().unwrap().bricks.iter()
                .find(|record| record.brick_key == 4).cloned().unwrap()
        };
        let still = centre(&stationary);
        let moving = centre(&falling);
        let moving_history = falling.resolution_policy.history.get(&4).unwrap();
        assert!(
            moving_history.velocity_travel.abs() <= 1.0e-5,
            "uniform bulk translation recorded travel {}",
            moving_history.velocity_travel,
        );
        assert_eq!(moving.requested_resolution, still.requested_resolution);
        assert_eq!(moving.scheduled_resolution, still.scheduled_resolution);
        assert_eq!(moving.scheduled_resolution, 2);
        assert_eq!(
            moving.reasons & crate::resolution::activity_reason::VELOCITY_FLOOR,
            0,
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
    fn level_set_drop_lands_in_both_the_volume_and_the_shared_surface() {
        let mut world = level_set_world();
        let drop = crate::injection::LiquidDrop {
            centre_fine: [6.0, 4.0],
            radius_fine: 1.5,
        };
        let inside = |world: &World| {
            world
                .state
                .topology
                .graph
                .cells
                .iter()
                .position(|cell| {
                    (cell.center[0] as f64 - drop.centre_fine[0]).hypot(
                        cell.center[1] as f64 - drop.centre_fine[1],
                    ) < 0.75
                })
                .unwrap()
        };
        // The ball is dropped into dry air the seeded column never reached, so
        // both authorities have to change for it to exist at all.
        let at = inside(&world);
        assert_eq!(world.state.fields.density[at], 0.0);
        assert!(world.level_set_phi[at] > 0.0);
        let before = world.receipt().liquid_measure;
        world
            .apply_command(1, 1, Command::InjectLiquid { drop })
            .unwrap();
        let receipt = world.last_injection.as_ref().unwrap().clone();
        assert!(receipt.accepted && receipt.fault.is_none());
        assert!(receipt.area_admitted_fine > 0.0);
        assert!(receipt.cells_wetted > 0);
        assert_eq!(world.revision.injections, 1);
        let at = inside(&world);
        assert!(world.state.fields.density[at] > 0.0);
        assert!(
            world.level_set_phi[at] < 0.0,
            "the shared level set still calls the ball's interior air: {}",
            world.level_set_phi[at],
        );
        assert!(
            (world.receipt().liquid_measure - before - receipt.area_admitted_fine).abs() < 1e-5
        );
        // Every cell carries a scalar the accepted contour agrees with, which
        // is what the next advance's redistancing and pressure geometry read.
        assert_eq!(
            world.level_set_phi.len(),
            world.state.topology.graph.cells.len()
        );
        assert!(world.level_set_phi.iter().all(|value| value.is_finite()));
        world.advance(2, 1.0 / 60.0).unwrap();
        assert!(world.state.fields.fault.is_none());
        assert!(world.level_set_volume_receipt.is_some());
    }
    #[test]
    fn level_set_drop_outside_the_lattice_is_refused_whole() {
        let mut world = level_set_world();
        let density = world.state.fields.density.clone();
        let phi = world.level_set_phi.clone();
        world
            .apply_command(
                1,
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
        assert_eq!(world.state.fields.density, density);
        assert_eq!(world.level_set_phi, phi);
        assert_eq!(world.revision.injections, 0);
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
