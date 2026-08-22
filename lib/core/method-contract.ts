import type { FluidPipelineGraph } from "./fluid-pipeline";
import type { SceneDescription } from "./model";
import type { GPUQuality } from "./gpu-quality";
import type { GPUEulerianInfo, GPURigidLoad } from "./webgpu-eulerian";
import type { RigidBodyState } from "./rigid-body";
import type { GPURigidBodyPick, GPURigidBodyPose } from "./webgpu-rigid-body";
import type { Vec3 } from "./model";
import type { GPUSecondaryParticleSource } from "./webgpu-secondary-particles";
import type { GPUFluidFaceVelocitySource } from "./webgpu-face-velocity-overlay";
import type { GPUPressureJournalSource } from "./webgpu-pressure-journal-overlay";
import type { AnyStageLens, StageLensSource } from "./stage-lens";
import type { SparseCM12PressureJournal } from
  "../methods/adaptive-mass/sparse-cm12-pressure-journal";
import type { GPUFluidTracerSource } from "./webgpu-tracer-overlay";
import type { SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import type {
  SparseAdaptiveGridConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "./levelset-consumer-abi";
import type { GPUInitializationPhase } from "./gpu-initialization";
import type { OctreeTechniqueDebugSource } from "./levelset-consumer-abi";
import type { CoarseLevelSetConsumerSource } from "./levelset-consumer-abi";
import type { ResourcePluginDefinition } from "./resource-plugin";
import type { SparseScenePrimitiveUpdate } from "./webgpu-sparse-scene-proxies";
import type { ConfigurationReadout, DiagnosticRow } from "./method-diagnostics";
import type { WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import type { ViewportFailureIndicator } from "./viewport-failure-diagnostics";
import type { MethodHarnessPlugin } from "./method-harness-contract";

/**
 * Method plugin contract.
 *
 * A simulation method owns three things:
 *  1. its identity and descriptive copy (picker labels, viewport badge, the
 *     technical summary shown in diagnostics),
 *  2. a declarative parameter schema — common numerics the method consumes
 *     plus any method-specific knobs — which the UI renders generically, and
 *  3. a solver factory that receives the scene and the resolved parameter
 *     values and returns a GPU solver honouring the shared solver interface.
 *
 * Adding a new method means adding one module under lib/methods/ and listing
 * it in the registry; no UI or renderer changes are required.
 */

export type MethodParamValue = number | string | boolean;
export type MethodParamValues = Record<string, MethodParamValue>;

/** An authored, reproducible solver configuration attached to validation scenes. */
export interface MethodProfile {
  methodId: string;
  quality: GPUQuality;
  overrides: MethodParamValues;
}

interface ParamBase {
  key: string;
  label: string;
  /** One-line explanation shown under the control. */
  hint?: string;
  /** Coarse controls are always visible; fine controls sit behind "Advanced". */
  tier: "coarse" | "fine";
  /** Runtime parameters can be applied to a live solver. All others are
   * structural and start a transactional solver rebuild. */
  update?: "runtime" | "solver";
}

export interface NumberParamSpec extends ParamBase {
  kind: "number";
  unit: string;
  min: number;
  max: number;
  step: number;
  digits?: number;
  default: number;
}

export interface SelectParamSpec extends ParamBase {
  kind: "select";
  options: ReadonlyArray<{ value: string; label: string }>;
  default: string;
}

export type MethodParamSpec = NumberParamSpec | SelectParamSpec;

/**
 * One knob on the live accuracy-for-frame-time strip.
 *
 * Declared apart from `MethodParamSpec` because a dial is rendered by exactly
 * one panel in exactly one row, and that row prints two captions the generic
 * parameter schema has nowhere to put. The panel renders this shape without
 * knowing what any of the keys mean; which dials exist, and what each one is
 * worth, is the owning method's answer.
 */
export interface RuntimeDialSpec {
  readonly key: string;
  readonly label: string;
  /** Upper-case strip label; the panel has one line to say what this is. */
  readonly short: string;
  readonly unit: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly digits: number;
  readonly default: number;
  /** Value meaning "construction decides"; rendered as AUTO. Absent = none. */
  readonly auto?: number;
  /**
   * What the knob controls, in one line, in terms of the simulation rather
   * than the machinery. This is the line that has to answer "what am I even
   * moving" without the reader knowing what a V-cycle is.
   */
  readonly does: string;
  /**
   * What the cheap direction costs, naming the direction explicitly — cheaper
   * is *down* for three of these and *up* for the cadence, so "lower" is not a
   * safe shorthand. Concrete failure, not "reduced accuracy": the reason to
   * print this beside the slider is so a strange-looking frame is recognisable
   * as a dial you moved rather than a bug you introduced.
   */
  readonly cost: string;
  readonly hint: string;
}

/** What the strip prints once the method has resolved its own semantics. */
export interface RuntimeDialReadouts {
  /**
   * Slider position per dial key, after the method has clamped it to the
   * compiled envelope and snapped it to the step the GPU can express.
   */
  readonly positions: Readonly<Record<string, number>>;
  /**
   * Value a dial will actually take, beside dials whose slider position is
   * only a request. Keyed by dial; a key with no entry has a position that is
   * already the value. "6 sweeps" that the ping-pong smoother rounds to 6 and
   * "48 iterations" that the compiled envelope clamps to 33 are different
   * claims, and only the effective one explains the measurement.
   */
  readonly effective: Readonly<Record<string, string>>;
  /** Live work counter for the strip header, against whatever bounds it. */
  readonly executed?: { readonly text: string; readonly hint: string };
}

/** A method's contribution to the live dial strip: the knobs and their readouts. */
export interface RuntimeDials {
  /** Eyebrow naming whose machinery these dials drive. */
  readonly label: string;
  readonly specs: readonly RuntimeDialSpec[];
  /**
   * Derive the strip's readouts from everything the panel already holds — the
   * resolved method values, the scene's authored numerics, and the last stats
   * poll. The panel cannot do this itself: every one of these derivations is a
   * solver invariant (an envelope clamp, a protection width, a residual
   * target) whose rule lives with the method that compiled it.
   */
  readouts(
    values: MethodParamValues,
    scene: Pick<SceneDescription, "numerics">,
    info: GPUEulerianInfo | undefined,
  ): RuntimeDialReadouts;
}

/**
 * A method-owned render pass the renderer composes onto the finished frame.
 *
 * The renderer drives overlays but does not own them: these bake a method's
 * own generated constants into their WGSL and read its compact debug source,
 * so relocating them into core would move a method's shaders there with them.
 * Instead the method registers a factory under the renderer's optional
 * pipeline key and the renderer constructs through this shape, which is
 * exactly what it drives and nothing more.
 */
export interface OverlayPipeline {
  initialize(): Promise<void>;
  setSource(source: OctreeTechniqueDebugSource | undefined): void;
  setOwnerRows(ownerRows: GPUTexture | undefined): void;
  /** Draws when the mode code selects one of its programs; false when it has nothing to draw. */
  encode(encoder: GPUCommandEncoder, target: GPUTextureView, modeCode: number): boolean;
  /**
   * Optional because the renderer retires these overlays by dropping the
   * reference with the device they were built on; a pipeline holding host
   * allocations of its own declares this and gets cleaned up on device loss.
   */
  destroy?(): void;
}

export type OverlayPipelineFactory = (
  device: GPUDevice,
  format: GPUTextureFormat,
  uniformBuffer: GPUBuffer,
) => OverlayPipeline;

/**
 * A ball of liquid added to a solve that is already running.
 *
 * Authoring one into the document would re-seed from t = 0, which throws away
 * the run the user is adding water to. A solver that implements the injection
 * below can take the ball where the clock already is instead.
 */
export interface InjectedLiquidBall {
  readonly centre_m: Vec3;
  readonly radius_m: number;
  /**
   * Set when the drop is a disk rather than a ball: the half-depth it spans
   * along z, which is how a 2D case's liquid is shaped. Absent is a ball.
   */
  readonly halfHeight_m?: number;
}

/** Minimal interface the renderer needs from a GPU solver. */
export interface GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  /** Field the renderer contours; a smooth level set when the solver keeps one separate from volumeTexture. */
  readonly surfaceFieldTexture?: GPUTexture;
  /** Dense tall-column metadata; compact octree methods have no such field. */
  readonly columnBaseTexture?: GPUTexture;
  /** Adaptive pressure-cell ownership for scientific grid slices. */
  readonly gridCellTexture?: GPUTexture;
  /** Live velocity field for scientific slice modes (CFL/speed heatmaps). */
  readonly velocityTexture?: GPUTexture;
  /** Optional one-way escaped spray droplets rendered above the liquid surface. */
  readonly secondaryParticles?: GPUSecondaryParticleSource;
  /** Optional presentation-only fluid markers, for the seed-spectrum view. */
  readonly tracerSource?: GPUFluidTracerSource;
  /**
   * Turn marker advection on or off. Absent means the method has no markers.
   *
   * The renderer calls this from the view it is asked to draw, because the view
   * is the only thing that knows whether anyone is looking: markers are pure
   * presentation and a solver that advected them unwatched would be charging
   * every frame for a picture nobody asked for.
   */
  setTracersEnabled?(enabled: boolean): void;
  /**
   * Optional MAC face velocities, for the face-arrow view.
   *
   * There is no enable to pair with this, and that asymmetry with the markers
   * above is the point: face velocities exist because the solve needs them, so
   * a view that reads them adds a draw and no simulation work at all.
   */
  readonly faceVelocitySource?: GPUFluidFaceVelocitySource;
  /**
   * Optional captured pressure-solve journal, for the pressure-lab film.
   *
   * Unlike face velocities this one *is* paired with an enable, and for the
   * same reason the markers are: the numbers it draws do not otherwise exist.
   * The snapshots are written by dispatches the encoder only emits for an
   * armed frame, so an unwatched solve encodes none of them.
   */
  readonly pressureJournalSource?: GPUPressureJournalSource;
  /** Arm or disarm journal capture; false when the method reserved no journal. */
  armPressureJournal?(armed: boolean): boolean;
  /**
   * Read back the captured film's iteration records.
   *
   * Maps a buffer, so it belongs to the same paused-ownership boundary as
   * `readStats` and never to a live frame. Resolves undefined when nothing has
   * been captured. The whole-field snapshots stay on the device — only these
   * few kilobytes of scalars ever cross.
   */
  readPressureJournal?(): Promise<SparseCM12PressureJournal | undefined>;
  /**
   * Per-stage lenses and the buffers they draw. Absent on methods with none.
   *
   * A lens is a reading of one *stage*, so a method whose advance is not
   * partitioned into named resident stages has nothing to offer here. Declaring
   * one costs nothing: no snapshot is copied, no header read back and no
   * pipeline compiled until a view arms a lens, and only ever one at a time.
   */
  readonly stageLensSource?: StageLensSource;
  /** Always-resident structural sparse scene used by production SVO rendering. */
  readonly sparseVoxelSceneSource?: SparseVoxelSceneRenderSource;
  /** Exact compact topology/geometry buffers for paper-technique overlays. */
  readonly octreeTechniqueDebugSource?: OctreeTechniqueDebugSource;
  /** True only after the complete t=0 sparse authority has passed its queue fence. */
  readonly initialSparseAuthorityReady?: boolean;
  /** Row-independent, globally indexed sparse fine level-set bricks. */
  readonly globalFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  /** Live sparse represented-cell fields for scientific grid overlays. */
  readonly sparseAdaptiveGridSource?: SparseAdaptiveGridConsumerSource;
  /** Compact moving surface used when coarse-1 deliberately has no fine band. */
  readonly coarseLevelSetSource?: CoarseLevelSetConsumerSource;
  /**
   * Whatever the owning method publishes for quality assurance, unnamed here.
   *
   * A method's diagnostics are producer headers, forensic arenas and control
   * buffers whose meaning belongs entirely to its own solve — the compact
   * topology deltas of an adaptive lane say nothing to a uniform one, and no
   * two lanes agree on which of them even exist. Naming them in this contract
   * would put one lane's vocabulary in front of every consumer of every other,
   * and would make the renderer the place a lane goes to declare a debug
   * buffer. So the contract carries the bag and forgets its contents.
   *
   * Reading it is the job of a typed decoder shipped with the method that
   * fills it (for the octree lane, `octreeDebugSources` in
   * `lib/octree-debug-sources.ts`), which is the only code entitled to know
   * what the keys mean. Nothing the renderer draws may depend on this: a
   * publication a frame needs is a named member above, not an entry here.
   */
  readonly debug?: Record<string, unknown>;
  /** GPU-authored rigid records matching the renderer's four-vec4 body ABI. */
  readonly rigidRenderBuffer?: GPUBuffer;
  /** GPU-authored 128-byte primitive-motion sidecars, including conservative swept bounds. */
  readonly rigidMotionBuffer?: GPUBuffer;
  /** Updates selection metadata without mirroring dynamic poses through CPU memory. */
  setSelectedRigidBody?(index: number): void;
  /**
   * Add a ball of liquid to the running field, without re-seeding.
   *
   * Optional: a method that does not implement it simply keeps taking the
   * re-seed, so dropping a ball still works — it just costs the run.
   */
  injectLiquidBall?(ball: InjectedLiquidBall): void;
  /**
   * Stage the latest authoritative scene revision for GPU consumers.
   *
   * Scene geometry, materials, lighting, and eventually fluid-domain inputs
   * all travel through this one live update seam. Implementations retain no
   * separate baked/fluid-free scene identity; they may reuse generation-matched
   * acceleration data, but the supplied scene is always the authority.
   */
  stageSceneUpdate?(scene: SceneDescription): void;
  /** Stage allocation-free keyed primitive motion into the shared sparse scene. */
  stageLivePrimitiveUpdates?(updates: readonly SparseScenePrimitiveUpdate[]): boolean;
  /**
   * Encode bounded maintenance for the scene revision staged above.
   * Presentation calls this even while simulation is paused so sparse
   * accelerators can converge without a solver step or a queue fence.
   */
  encodeSceneMaintenance?(encoder: GPUCommandEncoder): void;
  /** User-triggered ray query against authoritative GPU rigid poses. */
  pickRigidBody?(origin: Vec3, direction: Vec3): Promise<GPURigidBodyPick | undefined>;
  /**
   * Authoritative poses for the bodies the solver owns, in roster order.
   *
   * The host roster is a command channel — it is never written back from the
   * run — so anything that must agree with the drawn frame asks here.
   */
  readRigidBodyPoses?(): Promise<GPURigidBodyPose[] | undefined>;
  /** Adaptive pressure-DOF ownership used by the representation alarm. */
  readonly gridPressureSamplesTexture?: GPUTexture;
  /** Fine MLS pressure materialized by the latest adaptive solve. */
  readonly gridPressureTexture?: GPUTexture;
  /** Post-projection fine-cell divergence diagnostic. */
  readonly gridDivergenceTexture?: GPUTexture;
  /** Lazily allocate dense adaptive fields when a scientific grid slice needs them. */
  ensureGridDiagnosticTextures?(): void;
  /**
   * Adopt scene scalars that no lattice or seed depends on — density,
   * viscosity, surface tension, gravity. Solvers read these from the retained
   * scene when they write per-step params, so adopting a new scene is a
   * uniform write rather than a rebuild. Implementing this is what lets
   * `gpuSceneUniformKey` stay out of the rebuild trigger.
   */
  applySceneUniforms?(scene: SceneDescription): void;
  /**
   * Re-seed t=0 in place for a scene that differs only in the seed tier,
   * reusing every allocation, arena, and compiled pipeline. Resolves false
   * when the seed cannot be honoured, in which case the caller must take the
   * full rebuild — the solver is left usable but not re-seeded.
   */
  reseed?(scene: SceneDescription): Promise<boolean>;
  /** Apply configuration explicitly classified as runtime-safe by the method. */
  applyRuntimeValues?(values: MethodParamValues): void;
  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean;
  /** Host-only timing snapshot; never fences or maps simulation buffers. */
  readPerformanceTraceSnapshot?(): Pick<GPUEulerianInfo,
    "physicsTrace" | "physicsCPUTrace" | "physicsCaptureIdentity">;
  readStats(): Promise<GPUEulerianInfo>;
  destroy(): void;
}

export interface SimulationMethod {
  id: string;
  /** Full name shown in the method picker. */
  label: string;
  /** Short name for segmented controls and the top bar. */
  shortLabel: string;
  /** Upper-case badge shown over the viewport. */
  badge: string;
  /** One-sentence summary for the picker. */
  description: string;
  /** Technical summary for the diagnostics panel. */
  detail: string;
  /** Where the authoritative fluid state lives. */
  backend: "webgpu";
  /** Resource lifecycle declaration, colocated with the method that owns its
   * initialization. Readiness attributes every status the bring-up publishes
   * through it, so a method without one has no lane to appear in. */
  resource: ResourcePluginDefinition;
  /** Per-quality flavour text for the quality selector. */
  qualityLabels: Record<GPUQuality, string>;
  /** Hide the generic quality selector when the method exposes its relevant
   * scientific trade-offs directly. */
  showQualityControl?: boolean;
  /**
   * Field-visualization modes (`GridOverlayMode` strings) whose data this
   * method's solver actually publishes. Pickers narrow the visualization
   * catalog to these, so a method never offers a view of a publication it
   * does not produce — the uniform reference has no power topology. Declared
   * as strings because the modes belong to the catalog entries, not to the
   * method module.
   */
  supportedFieldModes?: readonly string[];
  /**
   * Lenses on this method's resident stages, in stage order.
   *
   * Separate from `supportedFieldModes` because a lens is not a field view: a
   * field view samples what survived to the end of the frame, and a lens
   * reports what one stage did in the middle of it. Declared on the descriptor
   * rather than fetched from the solver so a picker can offer them before a
   * device exists, and so a method with none simply has none.
   */
  stageLenses?: readonly AnyStageLens[];
  /**
   * Method-specific parameters. Common parameters (resolution, time step,
   * pressure solve effort) live in the scene numerics and are declared once
   * in the common schema, not here.
   */
  params: ReadonlyArray<MethodParamSpec>;
  /**
   * How this method interprets the common "pressure solve effort" iteration
   * budget (scene.numerics.pressureMaxIterations). Purely descriptive; the
   * mapping itself happens in createSolver.
   */
  pressureMapping: string;
  /**
   * Effective parameter values implied by a quality preset. The UI shows
   * these as the baseline; user overrides are stored sparsely on top and
   * merged via resolveMethodValues before reaching createSolver.
   */
  presetFor(quality: GPUQuality): MethodParamValues;
  /**
   * Resolve method-owned invariants after defaults, the quality preset, and
   * sparse user overrides have been merged. This is the right seam for a
   * compound scientific choice whose dependent controls must survive URL
   * hydration and scene-profile application as one valid configuration.
   */
  normalizeValues?(values: MethodParamValues): MethodParamValues;
  /** Keys omitted from the structural solver fingerprint and applied directly
   * to the active/candidate solver instead. */
  runtimeParamKeys?: readonly string[];
  /**
   * What this method can do, asked as a question about the method rather than
   * a test of its id.
   *
   * Every flag here replaced a literal `methodId === "octree"` in a component
   * or in core. That test was wrong the moment a second octree-family method
   * existed, and it was invisible to the type system, so a new method silently
   * inherited the dense answer instead of declaring its own.
   */
  capabilities?: {
    /** Publishes a volume the renderer can ray-march, not just a slice field. */
    volumeRendering?: boolean;
    /**
     * The initial sparse authority and adaptive raster source are published
     * separately and must cross a presentation fence before transport unlocks.
     * A method that attaches its field textures atomically with the warmed
     * solver leaves this off.
     */
    fencedInitialPresentation?: boolean;
    /**
     * A change in the *shape* of the rigid roster — gaining its first body, or
     * its first moving one — is adopted by the running solver rather than
     * requiring one to be built.
     *
     * The test is exactly one thing: the method allocates nothing from the
     * roster. Two different arrangements pass it.
     *
     *  - The method owns rigid state whose arenas are sized at
     *    `GPU_RIGID_BODY_CAPACITY` whether or not the scene fills them, with
     *    the whole roster re-uploaded through `syncBodies` every advance and
     *    coupling gated at encode time on the live body count rather than
     *    compiled in. Uniform CM12. A body dropped into running water there
     *    both keeps the clock and interacts with it.
     *  - The method has no rigid system at all, so there is nothing for the
     *    roster to size. Sparse CM12. A body there is inert — drawn from the
     *    CPU roster, movable by a carry, and invisible to the water — but
     *    restarting the solve does not make it any less so, which is the whole
     *    argument for declaring this: the reset costs the scene and buys
     *    nothing.
     *
     * A method with a dense solid field sized on solid *presence* — the octree
     * family, through `octreeSparseWorldRequired` — must leave this off, and
     * `rigidAllocationKey` then keeps that first body in the solver key.
     */
    adoptsRigidRosterShape?: boolean;
  };
  /**
   * The SIM panel's advance diagram, loaded on demand.
   *
   * Graphs are declared beside their encoders, so loading one pulls in the
   * solver module; the thunk is what keeps a panel from paying that cost just
   * to ask whether a diagram exists.
   */
  pipelineGraph?(): Promise<FluidPipelineGraph>;
  /**
   * The live accuracy-for-frame-time strip this method offers.
   *
   * The panel imported one method's dial table and three of its derivation
   * helpers directly, which made the observatory's most prominent strip an
   * import of a Losasso V-cycle module — and left a second method with nine
   * greyed-out sliders belonging to machinery it does not run. A method
   * without live, no-rebuild knobs simply declares none and gets no strip.
   */
  dials?: RuntimeDials;
  /**
   * The cards this method contributes to the live diagnostics grid.
   *
   * Every one of them was a panel branch on `gridKind === "octree"` or
   * `methodId === "uniform"` wrapping a publication a single method makes — a
   * frontier list capacity, a global-fine generation, an active-region work
   * box. A grid kind does not produce any of those: two methods in one family
   * share it, so the second silently inherits the first one's cards, and a
   * method that publishes none of them still has to be excluded by name
   * somewhere. Handing over the finished card puts the copy, the tone
   * thresholds, and the test for whether the publication exists yet in the one
   * place that can answer all three.
   *
   * `info` is undefined until the first stats poll lands, so the method also
   * owns what its cards say while there is nothing to report. `water` is the
   * renderer's presentation receipt rather than anything a solver publishes: a
   * method that judges its own generation against the mesh actually drawn
   * needs both sides, and no other consumer can pair them.
   */
  diagnosticRows?(
    info: GPUEulerianInfo | undefined,
    values: MethodParamValues,
    water: WaterSurfacePresentationDiagnostics | undefined,
  ): readonly DiagnosticRow[];
  /**
   * What the resolved configuration actually built, printed beside the
   * controls that chose it.
   *
   * The method panel prints the allocated grid itself because every method has
   * one. This is for the readouts a method only has because of the machinery
   * it selected — an executor's convergence against the cap it was compiled
   * with. That one was gated on a method id *and* a substring of the solver's
   * published label, which made a UI file the second place in the repo that
   * had to recognise a §4.3 executor from its label.
   */
  configurationReadouts?(
    info: GPUEulerianInfo | undefined,
    values: MethodParamValues,
  ): readonly ConfigurationReadout[];
  /**
   * The one failure worth drawing over the frame, from this method's own
   * published authority.
   *
   * Core keeps the triage that is true whatever is running — the scene's
   * runtime plan decides whether there is a fluid publication to judge at all,
   * and the projection is the raster water shaders' own camera — but every
   * verdict inside the alert reads a publication: which stage rejected, whether
   * the drawn mesh is the admitted generation, and where in the domain the
   * first invalid sample was. Those are statements about one method's
   * transaction, in its own vocabulary, and a method that publishes no such
   * receipts declares nothing and draws no marker.
   */
  viewportFailure?(
    info: GPUEulerianInfo,
    water: WaterSurfacePresentationDiagnostics | undefined,
    scene: SceneDescription,
  ): ViewportFailureIndicator | undefined;
  /**
   * Diagnostic render passes this method contributes, keyed by the renderer's
   * optional pipeline key.
   *
   * The renderer resolves a key to a factory through the registry; it never
   * names an overlay module. That direction is the whole point: an overlay
   * bakes its method's generated catalog constants into its WGSL, so an
   * import from core would drag one method's shader constants into the
   * renderer for every method's frame. A key nothing registers compiles
   * nothing, which is also how a method that publishes no such source avoids
   * building a pass that could only draw nothing.
   */
  overlayPipelines?: Readonly<Record<string, OverlayPipelineFactory>>;
  /**
   * The clock step this method actually advances, when its own numerics pin
   * one. Returning `undefined` accepts the scene's fixed step.
   *
   * A method whose step is a numerical contract rather than a maximum must say
   * so here: the scene author is free to write a smaller `fixedDt_s`, and
   * without this hook the app would drive a paper-calibrated solver at a rate
   * the paper never validated. It is a hook rather than a constant because
   * whether the contract binds can depend on the resolved parameter values.
   */
  effectiveStep_s?(
    scene: Pick<SceneDescription, "numerics">,
    values: MethodParamValues,
  ): number | undefined;
  /**
   * This method's node-only harness half, loaded on demand.
   *
   * The Dawn smoke executor asked fifty-three questions of the form
   * `method.id !== "uniform"`, imported twenty-two modules out of three method
   * packages, and carried a thousand lines of one lane's terminal oracle. None
   * of that belongs to a generic run loop, and none of it can be answered
   * generically. The thunk is what keeps it out of the browser: a plugin is a
   * node-only module reaching freely into its own method's ABI, and nothing in
   * the app ever calls this.
   */
  harness?(): Promise<MethodHarnessPlugin>;
  /** WebGPU methods create a solver; the CPU reference method does not. */
  createSolver?(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    onRigidLoads?: (loads: GPURigidLoad[]) => void
  ): GPUSolverInstance;
  /** Browser-safe construction path. Long shader compilation must use the
   * asynchronous WebGPU pipeline APIs so the main thread can keep painting. */
  createSolverAsync?(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    onProgress: GPUInitializationReporter,
    signal?: AbortSignal,
  ): Promise<GPUSolverInstance>;
}

export interface GPUInitializationProgress {
  phase: GPUInitializationPhase;
  taskId?: string;
  label: string;
  completed: number;
  total: number;
}

export type GPUInitializationReporter = (progress: GPUInitializationProgress) => void;

export function resolveMethodValues(method: SimulationMethod, quality: GPUQuality, overrides: MethodParamValues): MethodParamValues {
  const defaults = Object.fromEntries(method.params.map((spec) => [spec.key, spec.default]));
  const merged = { ...defaults, ...method.presetFor(quality), ...overrides };
  return method.normalizeValues?.(merged) ?? merged;
}

export function numberValue(values: MethodParamValues, spec: ReadonlyArray<MethodParamSpec>, key: string): number {
  const declared = spec.find((candidate) => candidate.key === key);
  const raw = values[key];
  if (typeof raw === "number" && Number.isFinite(raw)) {
    if (declared?.kind === "number") return Math.min(declared.max, Math.max(declared.min, raw));
    return raw;
  }
  return declared?.kind === "number" ? declared.default : 0;
}
