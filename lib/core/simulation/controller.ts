import { BUILD_ID, canonicalScene, cloneScene, parseScene, type SceneDescription } from "../model";
import { adoptRigidBodyRoster, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBodies, initializeRigidBody, rigidDiagnostics, type RigidBodyState, type RigidStepDiagnostics } from "../rigid-body";
import type { RigidBodyDescription } from "../model";
import type { RigidShape } from "../model";
import { sceneEditRequiresReset } from "../webgpu-renderer";
import type { RendererFrameMetrics } from "../webgpu-renderer";
import { getMethod } from "../method-registry";
import { requiresFencedInitialRasterPresentation } from "../gpu-t0-presentation";
import { findSceneDefinition, getSceneDefinition } from "../scenes";
import {
  sceneCardForDefinition,
  sceneDefinitionCamera,
  sceneDefinitionTakesLattice,
  sceneDocumentAtLattice,
  type SceneCard,
} from "../scene-definition";
import { svoSceneryDetailCellSize_m, svoSceneryRefinementDepth, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM, SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM } from "../../svo/svo-render-tuning";
import { terrainSampleShape } from "../terrain";
import { sceneStoneQuery, withSceneStoneQuery } from "../stone-look-controls";
import { sceneCanopyQuery, withSceneCanopyQuery } from "../tree-canopy-controls";
import { sceneRimQuery, withSceneRimQuery } from "../vessel-rim-controls";
import { savedSceneCard } from "../scene-cards";
import { sceneOverrideClearPlan } from "../scene-overrides";
import { SCENE_STARTERS } from "../empty-scene";
import { useShellStore } from "../stores/shell-store";
import { useSceneStore } from "../stores/scene-store";
import { useEditorHistoryStore, type EditorHistorySnapshot } from "../stores/history-store";
import { useMethodStore } from "../stores/method-store";
import type { MethodParamValue } from "../method-contract";
import { useRuntimeStore } from "../stores/runtime-store";
import { useDiagnosticsStore, emptyPerformanceReport } from "../stores/diagnostics-store";
import {
  gpuPhysicsPerformanceActivityFrameId,
  synthesizePerformanceActivityFrame,
  type ActivityFrameIdentity,
} from "../performance-activity";
import { usePerformanceActivityStore } from "../stores/performance-activity-store";
import { usePerformanceInstrumentationStore } from "../stores/performance-instrumentation-store";
import { useUIStore } from "../stores/ui-store";
import { commitGPUCompletion, gpuCanAcceptNextStep } from "./gpu-clock";
import { safeBrowserGPUBringupEnabled } from "../gpu-startup";
import { planSceneRuntime } from "../scene-runtime";
import { resourceInteractionGates } from "../resource-readiness";
import { addSceneryNode, createSceneryNodeAt, scenerySelectionId } from "../editor-scenery";
import { findSceneryNode, withoutSceneryNode } from "../scenery-edit";
import { scaleScene as scaleSceneBy, sceneScaleOption, sceneScaleSummary, type SceneScaleAxis, type SceneScaleFactor } from "../scene-scale";
import { sceneLatticeDimensions } from "../scene-lattice";
import { fluidBodyBox, fluidBodyBoxPatch, scaleFluidBodyVolume, type FluidBodyBox } from "../editor-fluid-body";
import { tankResizeIsStructural, tankResizePatch } from "../editor-tank";
import { useSceneDraftStore, type SceneDraftSubject } from "../stores/scene-draft-store";
import type { SceneryPropKind } from "../stores/ui-store";
import {
  browserSceneLibraryStorage,
  loadSceneFromLibrary,
  saveSceneToLibrary,
  type SceneLibraryEntry,
} from "../scene-library";
import {
  combineMainThreadPerformanceTraces,
  CPUPerformanceTrace,
  performanceTraceMatchesLane,
  type PerformanceTrace,
} from "../performance-trace";
import { effectiveSimulationStep_s, methodPinsSimulationStep } from "../simulation-step";

export type BodyDragPhase = "start" | "move" | "end";

const MAX_BODIES = 12;

/** Only CPU work recorded while encoding this exact GPU advance may share its
 * retained frame. Reporting cadence is not a join key: the renderer and
 * controller traces observed when readback completes can belong to later
 * animation frames. */
export function matchingPhysicsCPUTrace(
  physics: PerformanceTrace | undefined,
  cpu: PerformanceTrace | undefined,
  capture: Readonly<{ sampleId: number; context: string; frameId: string }> | undefined,
): PerformanceTrace | undefined {
  if (!physics || !cpu
    || !capture
    || !performanceTraceMatchesLane(physics, "gpu", "physics")
    || !performanceTraceMatchesLane(cpu, "cpu", "main-thread")
    || physics.sampleId !== cpu.sampleId
    || capture.sampleId !== physics.sampleId) return undefined;
  const fallbackSuffix = ":queue-wall-fallback";
  const physicsContext = physics.context.endsWith(fallbackSuffix)
    ? physics.context.slice(0, -fallbackSuffix.length)
    : physics.context;
  if (capture.context !== physicsContext || cpu.context !== physicsContext) return undefined;
  const expectedFrameId = gpuPhysicsPerformanceActivityFrameId({
    sampleId: physics.sampleId,
    context: physicsContext,
  });
  return capture.frameId === expectedFrameId ? cpu : undefined;
}

/** A completed GPU sample may only use its encoding trace. Latest controller
 * and renderer callbacks remain useful for CPU-only reports, but are never a
 * fallback join for an asynchronously completed physics frame. */
export function performanceReportCPUTrace(input: Readonly<{
  physics?: PerformanceTrace;
  physicsCPU?: PerformanceTrace;
  physicsCaptureIdentity?: Readonly<{ sampleId: number; context: string; frameId: string }>;
  controllerCPU?: PerformanceTrace;
  rendererCPU?: PerformanceTrace;
  context: string;
}>): PerformanceTrace | undefined {
  if (input.physics) {
    return matchingPhysicsCPUTrace(
      input.physics,
      input.physicsCPU,
      input.physicsCaptureIdentity,
    );
  }
  return combineMainThreadPerformanceTraces(
    [input.controllerCPU, input.rendererCPU]
      .filter((trace): trace is PerformanceTrace => trace !== undefined),
    input.context,
  );
}

/** O(1) host-clock collapse for GPU authority; no intermediate host state exists. */
export function collapseGPUFixedSteps(accumulator_s: number, dt_s: number) {
  if (!Number.isFinite(accumulator_s) || accumulator_s < 0 || !Number.isFinite(dt_s) || dt_s <= 0) {
    return { steps: 0, remainder_s: Math.max(0, Number.isFinite(accumulator_s) ? accumulator_s : 0) };
  }
  const steps = Math.floor((accumulator_s + 1e-12) / dt_s);
  return { steps, remainder_s: Math.max(0, accumulator_s - steps * dt_s) };
}

export const MIN_SHARED_STEP_S = 0.001;
export const MAX_SHARED_STEP_S = 0.05;

export function clampSharedStepSize(step_s: number): number {
  if (!Number.isFinite(step_s)) return 0.004;
  const wholeMilliseconds = Math.round(step_s * 1000) / 1000;
  return Math.min(MAX_SHARED_STEP_S, Math.max(MIN_SHARED_STEP_S, wholeMilliseconds));
}

export function sharedStepNumerics(
  numerics: SceneDescription["numerics"],
  step_s: number,
): SceneDescription["numerics"] {
  const sharedStep_s = clampSharedStepSize(step_s);
  return { ...numerics, fixedDt_s: sharedStep_s, maxDt_s: sharedStep_s };
}

/**
 * Owns the mutable runtime the render loop needs at 60 Hz — rigid-body
 * states, accumulators, pending GPU impulse loads — and publishes
 * serializable snapshots into the zustand stores. UI actions that must
 * rebuild runtime state (scene loads, body edits, resets) are methods here so
 * stores stay pure data.
 */
class SimulationController {
  private bodies: RigidBodyState[] = [];
  private accumulator = 0;
  private simulationTime = 0;
  private gpuCompletedTime = 0;
  private lastClock: number | null = null;
  private kinematicDrag: { bodyId: string; position: RigidBodyState["position_m"]; velocity: RigidBodyState["linearVelocity_m_s"] } | null = null;
  private rateWallClock = 0;
  private rateSimTime = 0;
  private safeBrowserStepConsumed = false;
  private cpuTickTraceSampleId = 0;
  private pendingCpuTickTrace?: PerformanceTrace;
  private lastPerformanceReportAt_ms = 0;
  private lastPerformanceReportContext = "";
  /** Document captured when a direct-manipulation gesture opened. */
  private pendingEdit?: { label: string; snapshot: EditorHistorySnapshot };

  private safeBrowserBringup(): boolean {
    return typeof location !== "undefined" && safeBrowserGPUBringupEnabled(location.search);
  }

  private webgpuTransportReady(): boolean {
    const diagnostics = useDiagnosticsStore.getState();
    if (!resourceInteractionGates(diagnostics.resourceReadiness, true).transportInteractive) return false;
    // A method that publishes its sparse authority and raster separately has
    // to cross that fence before transport unlocks; one that attaches its
    // field textures atomically with the warmed solver does not. The method
    // declares which it is, so this never has to list ids.
    return !requiresFencedInitialRasterPresentation(useMethodStore.getState().methodId)
      || (diagnostics.gpuInfo?.initialSparseAuthorityReady === true
        && diagnostics.gpuInfo?.initialRasterSurfaceReady === true);
  }

  constructor() {
    const scene = useSceneStore.getState().scene;
    // A reset lands on a different scene, so a proposal made against the old
    // one describes nothing. Drop it before anything can draw from it.
    useSceneDraftStore.getState().clearDraft();
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    // Open on an empty selection: with no gizmo on screen the first drag
    // orbits the scene, which is what someone arriving at a viewport expects.
    // Selecting is the user's move to make.
    useUIStore.getState().select(undefined);
  }

  time(): number { return this.simulationTime; }
  currentBodies(): RigidBodyState[] { return this.bodies; }

  private publishBodies(diagnostics?: RigidStepDiagnostics) {
    const scene = useSceneStore.getState().scene;
    useDiagnosticsStore.getState().set({ bodies: cloneRigidBodies(this.bodies), rigidState: diagnostics ?? rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2) });
  }

  /** Prepare every fixed step owed by the wall clock. GPU admission is renderer-budgeted. */
  tick(now: number) {
    const method = useMethodStore.getState();
    const instrumentationEnabled = usePerformanceInstrumentationStore.getState().enabled;
    const sampleId = instrumentationEnabled ? ++this.cpuTickTraceSampleId : this.cpuTickTraceSampleId;
    const cpuTrace = instrumentationEnabled
      ? new CPUPerformanceTrace(
        sampleId,
        `${method.methodId}:${method.quality}`,
        { id: "frame-control", label: "Simulation clock + admission" },
      )
      : undefined;
    try {
    if (this.lastClock === null) this.lastClock = now;
    const elapsed = Math.max(0, (now - this.lastClock) / 1000);
    this.lastClock = now;
    const runtime = useRuntimeStore.getState();
    if (!planSceneRuntime(useSceneStore.getState().scene).fluidSolver) {
      if (runtime.runState !== "paused") runtime.setRunState("paused");
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.accumulator = 0;
      this.rateWallClock = 0;
      return;
    }
    if (this.safeBrowserBringup() && runtime.runState === "running") {
      runtime.setRunState("paused");
      runtime.setSimRate(null);
      this.accumulator = 0;
      this.rateWallClock = 0;
      return;
    }
    if (runtime.runState !== "running") {
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    const scene = useSceneStore.getState().scene;
    if (!this.webgpuTransportReady()) {
      this.accumulator = 0;
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    this.accumulator += elapsed;
    const dt = effectiveSimulationStep_s(scene, useMethodStore.getState());
    cpuTrace?.transition({ id: "frame-control", label: "GPU target-clock control" });
    // The renderer admits GPU work against this target clock. Collapsing
    // accumulated fixed ticks is exact because no host fluid/body evolution
    // occurs at the intermediate ticks; solver.advanceTo owns subdivision.
    const collapsed = collapseGPUFixedSteps(this.accumulator, dt);
    if (collapsed.steps > 0) {
      this.applyDragConstraint();
      this.accumulator = collapsed.remainder_s;
      this.simulationTime += collapsed.steps * dt;
    }
    if (this.rateWallClock === 0) { this.rateWallClock = now; this.rateSimTime = this.gpuCompletedTime; }
    else if (now - this.rateWallClock > 500) {
      const committedTime = this.gpuCompletedTime;
      runtime.setSimRate((committedTime - this.rateSimTime) / ((now - this.rateWallClock) / 1000));
      this.rateWallClock = now; this.rateSimTime = committedTime;
    }
    } finally {
      this.pendingCpuTickTrace = cpuTrace?.finish();
    }
  }

  private applyDragConstraint() {
    const drag = this.kinematicDrag;
    if (!drag) return;
    const body = this.bodies.find((candidate) => candidate.description.id === drag.bodyId);
    if (body) { body.held = true; body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
  }

  singleStep() {
    const runtime = useRuntimeStore.getState();
    runtime.setRunState("paused");
    const scene = useSceneStore.getState().scene;
    if (!planSceneRuntime(scene).fluidSolver) return;
    const dt = effectiveSimulationStep_s(scene, useMethodStore.getState());
    if (!this.webgpuTransportReady()) return;
    if (!gpuCanAcceptNextStep(this.simulationTime, this.gpuCompletedTime)) return;
    if (this.safeBrowserBringup() && this.safeBrowserStepConsumed) return;
    if (this.safeBrowserBringup()) this.safeBrowserStepConsumed = true;
    this.simulationTime += dt;
  }

  /** Apply one shared fixed step to rigid and fluid work without resetting time. */
  setStepSize(step_s: number) {
    const sceneStore = useSceneStore.getState();
    const numerics = sharedStepNumerics(sceneStore.scene.numerics, step_s);
    // Asking for a step is how you leave the paper step. The uniform paper
    // profile pins the clock at 1/30 s, but the control stays live while it
    // does: an edit releases the profile to the scene-authored dt rather than
    // presenting a number that silently refuses to move.
    const releasedPaperStep = this.releaseUniformPaperStep();
    if (!releasedPaperStep
      && numerics.fixedDt_s === sceneStore.scene.numerics.fixedDt_s
      && numerics.maxDt_s === sceneStore.scene.numerics.maxDt_s) return;
    sceneStore.patchNumerics(numerics);
    this.accumulator = Math.min(this.accumulator, numerics.fixedDt_s);
    useRuntimeStore.getState().setNotice(releasedPaperStep
      ? `Uniform paper 1/30 s step released · shared rigid + fluid step ${(numerics.fixedDt_s * 1000).toFixed(2)} ms`
      : `Shared rigid + fluid step · ${(numerics.fixedDt_s * 1000).toFixed(2)} ms`);
  }

  /** Switch the uniform method off its paper step; true when it was on one. */
  private releaseUniformPaperStep(): boolean {
    const { methodId } = useMethodStore.getState();
    if (!methodPinsSimulationStep(useSceneStore.getState().scene, useMethodStore.getState())) return false;
    this.setMethodParam(methodId, "timeStep", "scene");
    return true;
  }

  reset(source?: SceneDescription, presetId?: string) {
    const sceneStore = useSceneStore.getState();
    const scene = source ?? sceneStore.scene;
    if (source) sceneStore.setScene(source, presetId);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    const runtimePlan = planSceneRuntime(scene);
    this.simulationTime = 0; this.gpuCompletedTime = 0; this.accumulator = 0; this.lastClock = null;
    this.rateWallClock = 0; this.rateSimTime = 0;
    this.pendingCpuTickTrace = undefined;
    // A carry outlives a rebuild. This used to drop the kinematic constraint
    // unconditionally, so any edit that re-seeded the solver let go of a body
    // the user was still holding: the hand stayed on the cursor and the solid
    // fell out of it with no click anywhere. Kept while the body it names
    // survives into the new roster, and re-applied before the publish below so
    // the body stands where the hand is rather than snapping back to the pose
    // it was authored at.
    const heldId = this.kinematicDrag?.bodyId;
    if (heldId !== undefined && !this.bodies.some((body) => body.description.id === heldId)) {
      this.kinematicDrag = null;
    }
    // The constraint is the only thing that may hold a body, so anything still
    // flagged without one is a release the roster outlived: a gesture whose
    // body was replaced by an edit, or a drag ended while it was off-roster.
    for (const body of this.bodies) {
      if (body.held && body.description.id !== this.kinematicDrag?.bodyId) body.held = false;
    }
    this.applyDragConstraint();
    if (!this.safeBrowserBringup()) this.safeBrowserStepConsumed = false;
    const diagnosticsStore = useDiagnosticsStore.getState();
    if (runtimePlan.fluidSolver
      && diagnosticsStore.resourceReadiness.fluid.state !== "preparing") {
      diagnosticsStore.set({ gpuStatus: {
        state: "initializing",
        label: "Preparing fenced t=0 fluid resources",
        phase: "planning",
        completed: 0,
        total: 0,
        startedAt_ms: performance.now(),
        kind: "rebuild",
        retainingPrevious: false,
        resource: getMethod(useMethodStore.getState().methodId).resource,
      } });
    }
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({ gpuInfo: null, waterSurfacePresentation: null, couplingState: { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 }, samples: [], performanceReport: emptyPerformanceReport, performanceReports: [] });
    const runtime = useRuntimeStore.getState();
    // Publish the new t=0 identity before pausing. Renderer subscribers use
    // this synchronous epoch edge to reject completions from the old queue.
    runtime.resetSimulationTime();
    runtime.setSimRate(null);
    runtime.setRunState("paused");
    // A reset lands on a fresh scene, so nothing carries a selection into it.
    // commitEdit({ reseed: true }) restores the user's selection afterwards.
    useUIStore.getState().select(undefined);
    runtime.setNotice(!runtimePlan.fluidSolver
      ? "Live renderer scene reset · fluid solver disabled"
      : `${scene.fluid.inflow ? "Inflow scene" : scene.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
  }

  /**
   * Open a library card: the one path a scene reaches the product by.
   *
   * Authored presets, saved documents and empty starters used to load through
   * three different functions with three different behaviours — only the preset
   * path applied a camera or a solver profile, and nothing at all could create
   * an empty scene. A library that shows all three in one grid cannot afford
   * that, so the differences live in the card and this stays one function.
   *
   * A card's `open` may throw: a stored document is validated on load, not on
   * save, so an entry written by an older schema surfaces here as a notice
   * rather than as a corrupted live scene.
   */
  openSceneCard(card: SceneCard): boolean {
    let opening;
    try {
      opening = card.open();
    } catch (error) {
      useRuntimeStore.getState().setNotice(
        error instanceof Error ? error.message : `${card.name} failed validation`, "warn");
      return false;
    }
    this.recordHistory(`open ${card.name}`);
    // A card is a clean scene entry, not a continuation of whichever hidden
    // studio state happened to be retained behind the library. Reset both
    // serializable stores before applying the scene's authored contracts so a
    // prior panel, diagnostic arm, quality experiment, or method override can
    // never leak into another scene or its URL.
    const previousCamera = useUIStore.getState().camera;
    useMethodStore.setState(useMethodStore.getInitialState());
    if (opening.methodProfile) useMethodStore.getState().seedProfile(opening.methodProfile);
    this.reset(opening.scene, opening.presetId);
    useUIStore.setState({
      ...useUIStore.getInitialState(),
      // Absent only for a saved document whose origin scene this build no
      // longer has; there the camera the reader already had is the better answer.
      camera: opening.camera ?? previousCamera,
    });
    const runtimePlan = planSceneRuntime(opening.scene);
    const effectiveStep_s = effectiveSimulationStep_s(
      opening.scene, useMethodStore.getState());
    useRuntimeStore.getState().setNotice(!runtimePlan.fluidSolver
      ? `${card.name} opened · no fluid solver, so nothing waits on one`
      : `${card.name} opened · dt ${effectiveStep_s.toFixed(4)} s`);
    useRuntimeStore.getState().setRunState(runtimePlan.fluidSolver ? "running" : "paused");
    useShellStore.getState().enterStudio();
    return true;
  }

  loadPreset(presetId: string) {
    this.openSceneCard(sceneCardForDefinition(getSceneDefinition(presetId)));
  }

  /**
   * Create a fresh document and enter the studio.
   *
   * The scene is fluid-free, which is the whole point: `planSceneRuntime` sees
   * `systems.fluid === false`, the renderer attaches the live sparse scene
   * instead of a solver, and there is no t=0 fence, no fluid pipeline family and
   * no transport gate between the click and the first frame. Water is added
   * later, deliberately, and that is the one expensive transition in the flow.
   */
  newScene(starterId?: string): boolean {
    const starter = SCENE_STARTERS.find((candidate) => candidate.id === starterId) ?? SCENE_STARTERS[0];
    return this.openSceneCard({
      id: `starter:${starter.id}`,
      source: "starter",
      name: starter.name,
      blurb: starter.blurb,
      shelf: "Start from empty",
      open: () => ({ scene: starter.create(), presetId: `starter:${starter.id}` }),
    });
  }

  /**
   * Hand the document to the fluid solver, or take it back.
   *
   * The one control that moves a scene across `planSceneRuntime`'s fluid gate,
   * and so between two different things being attached to the device: a solver,
   * or the live sparse scene. It is a document edit rather than a runtime knob —
   * `systems.fluid` is in the renderer's own solver key — so it goes on the undo
   * stack and through `reset`, exactly as opening a scene card does.
   *
   * Turning water *off* is how a scene whose solver will not come up is still
   * worth opening: the set renders, and nothing waits on a fluid authority. The
   * fill, the initial condition and the jet are left untouched in both
   * directions, so turning it back on restores the pond that was authored rather
   * than an empty tank.
   */
  setFluidSystem(enabled: boolean) {
    const scene = useSceneStore.getState().scene;
    if ((scene.systems?.fluid !== false) === enabled) return;
    const label = enabled ? "Enable water" : "Disable water";
    this.announceGPURebuild(label);
    this.recordHistory(label.toLowerCase());
    const next = cloneScene(scene);
    next.systems = { ...next.systems, fluid: enabled };
    this.reset(next);
    const runtime = useRuntimeStore.getState();
    runtime.setRunState("paused");
    runtime.setNotice(enabled
      ? "Water enabled · solver rebuilding, paused at t = 0"
      : "Water off · live SVO scene, so nothing waits on a solver");
  }

  /**
   * Re-author the open document at a lattice, rather than patch one onto it.
   *
   * The asymmetry this closes. `tools/run-svo-dry-render-smoke.ts` has been able
   * to ask for the hero garden at another lattice since
   * `createHeroGardenHoseSceneWithSet` took options: it calls the factory with
   * `cellSize_m` and `detailCellSize_m` as *inputs*, so the heightfield is baked
   * at that spacing and every generator resolves its legibility floors against
   * that detail voxel. The browser had no equivalent. Its two lattice controls
   * both wrote onto a finished document — "Finest cell" patches
   * `voxelDomain.finestCellSize_m`, and `environmentRefinementDepth` was a
   * render tuning that never reached the document at all — so the tree got finer
   * while the thing it voxelizes stayed exactly as coarse as it was built.
   *
   * **This is now the only way the depth moves.** It used to be *silently* two
   * numbers: `webgpu-octree-sparse-bricks.ts` divided the tree's cell by
   * `2^depth` from `SvoRenderTuning`, while `buildEnvironmentProxyCatalog` read
   * `voxelDomain.detailCellSize_m` off the document and got whatever was
   * authored. The tuning field is gone; the renderer derives the depth from the
   * document through `svoSceneryRefinementDepth`, and the document is the only
   * carrier that reaches the render worker. Two settings that can disagree is
   * the bug, not the drift between them.
   *
   * A rebuild is a reload: the document comes from the preset's factory, so
   * scenery edits, container edits and inflow edits made since it was opened do
   * not survive it. It goes on the undo stack for exactly that reason.
   *
   * Refused while water is on, and that is the engine's own rule rather than a
   * caution: `svoSceneryDetailCellSize_m` returns depth zero for a fluid scene
   * because a solver brick pins its node and leaves nowhere to descend, so there
   * is no finer document for a wet scene to be authored at.
   */
  rebuildSceneAtLattice(request: {
    readonly cellSize_m?: number;
    readonly environmentRefinementDepth?: number;
  }): boolean {
    const runtime = useRuntimeStore.getState();
    const sceneStore = useSceneStore.getState();
    const { scene, presetId } = sceneStore;
    const definition = findSceneDefinition(presetId);
    if (!definition || !sceneDefinitionTakesLattice(definition)) {
      runtime.setNotice(`${scene.sceneId} cannot be re-authored at another lattice`
        + " · its factory takes no cell size, so the tree is all a lattice can move here", "warn");
      return false;
    }
    if (scene.systems?.fluid !== false) {
      runtime.setNotice("Turn water off to re-author this scene at another environment lattice"
        + " · a solver brick pins its node, so a wet document cannot move on this ladder", "warn");
      return false;
    }
    const currentDepth = svoSceneryRefinementDepth(scene.voxelDomain, { fluid: false });
    const zeroCellSize_m = request.cellSize_m
      ?? scene.voxelDomain.environmentRefinementBaseCellSize_m
      ?? scene.voxelDomain.finestCellSize_m;
    if (!(zeroCellSize_m > 0) || !Number.isFinite(zeroCellSize_m)) return false;
    // The document's own depth is the fallback, because the document is where
    // the depth lives. A request that names only a cell size keeps the rung the
    // set was last expanded at rather than silently flattening it to zero.
    const depth = Math.min(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
      Math.max(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM, Math.trunc(
        request.environmentRefinementDepth ?? currentDepth)));
    // A negative rung is a coarser *tree*, not just a coarser authoring hint.
    // Enlarge the actual scene lattice and leave its internal refinement at
    // zero; positive rungs keep the zero cell and descend beneath it as before.
    const cellSize_m = depth < 0 ? zeroCellSize_m * 2 ** -depth : zeroCellSize_m;
    const detailCellSize_m = depth < 0 ? cellSize_m : svoSceneryDetailCellSize_m(zeroCellSize_m, {
      environmentRefinementDepth: depth,
      fluid: false,
    });
    const already = scene.voxelDomain.finestCellSize_m === cellSize_m
      && (scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m) === detailCellSize_m
      && (scene.voxelDomain.environmentRefinementBaseCellSize_m ?? cellSize_m)
        === (depth < 0 ? zeroCellSize_m : cellSize_m);
    if (already) return false;
    const label = `Re-author at ${(cellSize_m * 1000).toFixed(4).replace(/\.?0+$/, "")} mm`
      + (depth > 0 ? ` / ${(detailCellSize_m * 1000).toFixed(5).replace(/\.?0+$/, "")} mm set` : "");
    let rebuilt;
    try {
      rebuilt = sceneDocumentAtLattice(definition, { cellSize_m, detailCellSize_m });
      if (depth < 0) {
        rebuilt.scene.voxelDomain.environmentRefinementBaseCellSize_m = zeroCellSize_m;
      }
    } catch (error) {
      runtime.setNotice(error instanceof Error ? error.message : `${definition.name} refused that lattice`, "warn");
      return false;
    }
    // A rebuild is a reload, but the canopy dials and stone looks are the
    // user's art direction, not the factory's: carry them onto the re-authored
    // document, the same values the URL round-trips, so changing the
    // environment lattice does not silently regrow the tree the user just
    // thinned or reshape a boulder they just squashed.
    const canopy = sceneCanopyQuery(scene);
    const withCanopy = canopy === sceneCanopyQuery(rebuilt.scene)
      ? rebuilt.scene
      : withSceneCanopyQuery(rebuilt.scene, canopy);
    const stones = sceneStoneQuery(scene);
    const withStones = stones === sceneStoneQuery(withCanopy)
      ? withCanopy
      : withSceneStoneQuery(withCanopy, stones);
    const rim = sceneRimQuery(scene);
    const rebuiltScene = rim === sceneRimQuery(withStones)
      ? withStones
      : withSceneRimQuery(withStones, rim);
    this.announceGPURebuild(label);
    this.recordHistory(label.toLowerCase());
    // Nothing to synchronize: the rebuilt document carries the depth, and the
    // renderer reads it from there. That is the whole point of there being one.
    this.reset(rebuiltScene, presetId);
    useUIStore.getState().setCamera(sceneDefinitionCamera(definition));
    useRuntimeStore.getState().setRunState("paused");
    useRuntimeStore.getState().setNotice(`${definition.name} re-authored`
      + ` · lattice ${(cellSize_m * 1000).toFixed(4).replace(/\.?0+$/, "")} mm, set drawn at`
      + ` ${(detailCellSize_m * 1000).toFixed(5).replace(/\.?0+$/, "")} mm`
      + ` · terrain at ${((terrainSampleShape(rebuilt.scene.terrain)?.spacing_m ?? 0) * 1000).toFixed(4).replace(/\.?0+$/, "")} mm`);
    return true;
  }

  /**
   * Move the one refinement depth there is, by whichever route this scene has.
   *
   * The depth is `voxelDomain.detailCellSize_m` and nothing else — see
   * `svoSceneryRefinementDepth`. What differs between scenes is only how far a
   * change to it can propagate, and that turns out to be a much smaller
   * difference than `rebuildSceneAtLattice` alone implied:
   *
   *  - **Every** scenery generator reads the leaf off the document at
   *    *expansion* time, not at `create()` time — `growGenerator` takes it from
   *    `EnvironmentSceneryContext.detailCellSize_m`, which the render worker
   *    builds per world. So a plain patch already re-resolves every legibility
   *    floor in the set. That is the bulk of what a finer lattice buys.
   *  - A **factory** additionally re-bakes whatever it stored into the document
   *    at construction. For the hero garden that is `scene.terrain.spacing_m`,
   *    which `heroGardenTerrainSample_m` derives from the detail cell. A patch
   *    leaves that at the pitch it was authored with.
   *
   * So a factory rebuild is preferred and a patch is the fallback, rather than
   * the fallback being *nothing*. Only two of the catalogue's presets declare
   * `buildAt`, and refusing the other thirty-seven outright removed a control
   * that did real work on them: the proxies are exact analytic SDFs, so a finer
   * leaf resolves them genuinely better even with no new authored detail.
   *
   * Wet scenes are the one true refusal, and it is the engine's rule rather than
   * a caution — a solver brick pins its node, so there is nowhere to descend.
   */
  setEnvironmentRefinementDepth(depth: number): boolean {
    const runtime = useRuntimeStore.getState();
    const sceneStore = useSceneStore.getState();
    const { scene, presetId } = sceneStore;
    if (scene.systems?.fluid !== false) {
      runtime.setNotice("Turn water off to change environment refinement depth"
        + " · a solver brick pins its node, so a wet scene cannot move on this ladder", "warn");
      return false;
    }
    const requested = Math.min(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
      Math.max(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM, Math.trunc(depth)));
    if (requested === svoSceneryRefinementDepth(scene.voxelDomain, { fluid: false })) return false;
    const definition = findSceneDefinition(presetId);
    if (definition && sceneDefinitionTakesLattice(definition)) {
      return this.rebuildSceneAtLattice({ environmentRefinementDepth: requested });
    }
    const zeroCellSize_m = scene.voxelDomain.environmentRefinementBaseCellSize_m
      ?? scene.voxelDomain.finestCellSize_m;
    const finestCellSize_m = requested < 0 ? zeroCellSize_m * 2 ** -requested : zeroCellSize_m;
    const detailCellSize_m = requested < 0 ? finestCellSize_m : svoSceneryDetailCellSize_m(zeroCellSize_m, {
      environmentRefinementDepth: requested, fluid: false,
    });
    const label = `Draw the set at ${(detailCellSize_m * 1000).toFixed(5).replace(/\.?0+$/, "")} mm`;
    this.announceGPURebuild(label);
    this.recordHistory(label.toLowerCase());
    sceneStore.patchScene({
      voxelDomain: {
        ...scene.voxelDomain,
        finestCellSize_m,
        // Omitted rather than equal, so an unrefined document keeps saying
        // nothing about its detail cell and `svoSceneryRefinementDepth` reads
        // the same zero from either spelling.
        ...(detailCellSize_m < finestCellSize_m ? { detailCellSize_m } : { detailCellSize_m: undefined }),
        ...(requested < 0
          ? { environmentRefinementBaseCellSize_m: zeroCellSize_m }
          : { environmentRefinementBaseCellSize_m: undefined }),
      },
    });
    runtime.setNotice(`Set drawn at ${(detailCellSize_m * 1000).toFixed(5).replace(/\.?0+$/, "")} mm`
      + ` · every generator re-resolves its legibility floors at that leaf`
      + (terrainSampleShape(scene.terrain)
        ? `, but ${scene.sceneId}'s factory takes no lattice, so its terrain keeps the`
          + ` ${((terrainSampleShape(scene.terrain)?.spacing_m ?? 0) * 1000).toFixed(4).replace(/\.?0+$/, "")} mm pitch it was authored at`
        : ""));
    return true;
  }

  setQuality(quality: Parameters<ReturnType<typeof useMethodStore.getState>["setQuality"]>[0]) {
    this.announceGPURebuild(`Apply ${quality} quality`);
    useMethodStore.getState().setQuality(quality);
    this.reset();
    useRuntimeStore.getState().setNotice(`Quality ${quality} · simulation reset`);
  }

  setMethod(methodId: string) {
    this.announceGPURebuild(`Switch to ${getMethod(methodId).label}`);
    useMethodStore.getState().setMethodId(methodId);
    this.reset();
    useRuntimeStore.getState().setNotice(`${getMethod(methodId).label} selected`);
  }

  /** Structural settings start from a defined t=0 state. Runtime-safe settings
   * are applied to the live GPU solver without changing the simulation clock. */
  private announceGPURebuild(operation: string) {
    const current = useDiagnosticsStore.getState().gpuStatus;
    useDiagnosticsStore.getState().set({ gpuStatus: {
      state: "initializing",
      label: "Preparing GPU work plan",
      phase: "planning",
      completed: 0,
      total: 0,
      startedAt_ms: performance.now(),
      kind: "rebuild",
      operation,
      retainingPrevious: current.state === "ready" || (current.state === "initializing" && Boolean(current.retainingPrevious)),
      resource: getMethod(useMethodStore.getState().methodId).resource,
    } });
  }

  setMethodParam(methodId: string, key: string, value: MethodParamValue) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === useMethodStore.getState().methodId && spec?.update !== "runtime";
    if (structural) this.announceGPURebuild(`Apply ${spec?.label ?? key}: ${String(value)}`);
    useMethodStore.getState().setParam(methodId, key, value);
    if (structural) this.reset();
  }

  resetMethodParam(methodId: string, key: string) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === useMethodStore.getState().methodId && spec?.update !== "runtime";
    if (structural) this.announceGPURebuild(`Restore ${spec?.label ?? key} default`);
    useMethodStore.getState().resetParam(methodId, key);
    if (structural) this.reset();
  }

  // ---- authoring: draft/commit, history ----------------------------------

  /** The live document, cloned so history never aliases store state. */
  private documentSnapshot(label: string): EditorHistorySnapshot {
    const sceneStore = useSceneStore.getState();
    return { label, scene: cloneScene(sceneStore.scene), presetId: sceneStore.presetId };
  }

  /**
   * Push the pre-edit document onto the undo stack. `coalesceKey` groups a
   * continuing gesture — a held slider, a repeated nudge — into one entry.
   * An open `beginEdit` gesture already holds the pre-gesture snapshot, so
   * document writes made during one record nothing until it commits.
   */
  private recordHistory(label: string, coalesceKey?: string) {
    if (this.pendingEdit) return;
    useEditorHistoryStore.getState().record(this.documentSnapshot(label), { coalesceKey });
  }

  /**
   * Direct manipulation is a live-preview / commit-on-release split: the drag
   * touches only runtime state the renderer already owns (`manipulateBody`),
   * and the document changes once, here, when the pointer is released. That
   * keeps one undo entry per gesture and one solver invalidation per edit.
   *
   * Committing currently takes the renderer's existing solver-key rebuild
   * path; Phase 1 of docs/WYSIWYG_EDITOR_PLAN.md replaces it with a warm
   * re-seed so the edit is simulating again in ~100 ms.
   */
  beginEdit(label: string) {
    this.pendingEdit = { label, snapshot: this.documentSnapshot(label) };
  }

  /**
   * Close the gesture; a patch that changed nothing records no undo entry.
   *
   * `reseed` asks for the solver to be started again from the edited document.
   * Scene fields the solver cannot adopt live reach it no other way, and an edit
   * that the renderer shows but the physics ignores is the worst possible
   * outcome for a WYSIWYG editor.
   *
   * It is a request, not an instruction: `sceneEditRequiresReset` decides. An
   * edit confined to the uniform tier — aiming the hose, changing density — is
   * adopted by the running solver through `applySceneUniforms`, so resetting for
   * it would throw away the simulation the user is editing in order to apply a
   * buffer write. Call sites keep asking; only the ones that need it now pay.
   */
  commitEdit(patch?: Partial<SceneDescription>, options: { reseed?: boolean } = {}) {
    const pending = this.pendingEdit;
    this.pendingEdit = undefined;
    const sceneStore = useSceneStore.getState();
    if (patch) sceneStore.patchScene(patch);
    if (!pending) return false;
    const committed = useSceneStore.getState().scene;
    const changed = canonicalScene(committed) !== canonicalScene(pending.snapshot.scene);
    if (!changed) return false;
    useEditorHistoryStore.getState().record(pending.snapshot);
    if (options.reseed && sceneEditRequiresReset(pending.snapshot.scene, committed, useMethodStore.getState().methodId)) {
      // reset() re-selects the first body; an editor gesture must keep the
      // thing the user is still holding selected.
      const selection = useUIStore.getState().selection;
      this.reset();
      useUIStore.getState().select(selection);
    } else this.adoptRigidBodies(committed);
    useRuntimeStore.getState().setNotice(pending.label);
    return true;
  }

  /**
   * Reconcile the running roster with an edit that did not take a reset.
   *
   * `reset` is the only other place `this.bodies` is built, so without this a
   * body added while the clock runs would be in the document, on screen, and
   * absent from `advanceTo` — the solver would never hear about it.
   */
  private adoptRigidBodies(scene: SceneDescription) {
    const before = this.bodies;
    this.bodies = adoptRigidBodyRoster(before, scene.rigidBodies);
    if (before.length === this.bodies.length && before.every((body, index) => body === this.bodies[index])) return;
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
  }

  /** Abandon a gesture without recording it; runtime preview state is kept. */
  cancelEdit() { this.pendingEdit = undefined; }

  private applyHistorySnapshot(entry: EditorHistorySnapshot, verb: string) {
    this.reset(cloneScene(entry.scene), entry.presetId);
    useRuntimeStore.getState().setNotice(entry.label ? `${verb} ${entry.label}` : `${verb} last edit`);
  }

  undo(): boolean {
    this.pendingEdit = undefined;
    const entry = useEditorHistoryStore.getState().undo(this.documentSnapshot(""));
    if (!entry) { useRuntimeStore.getState().setNotice("Nothing to undo"); return false; }
    this.applyHistorySnapshot(entry, "Undid");
    return true;
  }

  redo(): boolean {
    this.pendingEdit = undefined;
    const entry = useEditorHistoryStore.getState().redo(this.documentSnapshot(""));
    if (!entry) { useRuntimeStore.getState().setNotice("Nothing to redo"); return false; }
    this.applyHistorySnapshot(entry, "Redid");
    return true;
  }

  // ---- rigid-body roster ------------------------------------------------

  addBody(shape: RigidShape) {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { useRuntimeStore.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return; }
    this.recordHistory(`add ${shape}`);
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${shape}-${bodyIndex}`)) bodyIndex += 1;
    const description = createBodyDescription(shape, bodyIndex, scene.container.height_m);
    sceneStore.patchScene({ rigidBodies: [...scene.rigidBodies, description] });
    this.bodies = [...this.bodies, initializeRigidBody(description)];
    this.publishBodies();
    useUIStore.getState().selectBody(description.id);
    useRuntimeStore.getState().setNotice(`${description.name} added above the container`);
  }

  /**
   * Spawn a body at a specific point, e.g. dropped from the viewport tray.
   * Tray drops start the clock so the body visibly falls; editor placement
   * passes `autoRun: false` because authoring geometry is an edit, not a throw.
   */
  /**
   * Returns the body it created, so a caller that places one in order to act on
   * it immediately — the DRAG tool grabbing what an empty click just dropped —
   * does not have to re-find it by guessing the generated id.
   */
  addBodyAt(shape: RigidShape, position: RigidBodyState["position_m"], options: { autoRun?: boolean } = {}): RigidBodyDescription | undefined {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { useRuntimeStore.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return undefined; }
    this.recordHistory(`place ${shape}`);
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${shape}-${bodyIndex}`)) bodyIndex += 1;
    const template = createBodyDescription(shape, bodyIndex, scene.container.height_m);
    const radius = boundingRadius(template);
    const description = { ...template, position_m: {
      x: Math.min(scene.container.width_m / 2 - radius, Math.max(-scene.container.width_m / 2 + radius, position.x)),
      y: Math.min(scene.container.height_m + 0.8, Math.max(radius, position.y)),
      z: Math.min(scene.container.depth_m / 2 - radius, Math.max(-scene.container.depth_m / 2 + radius, position.z))
    }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } };
    sceneStore.patchScene({ rigidBodies: [...scene.rigidBodies, description] });
    this.bodies = [...this.bodies, initializeRigidBody(description)];
    this.publishBodies();
    useUIStore.getState().selectBody(description.id);
    if (options.autoRun !== false) useRuntimeStore.getState().setRunState("running");
    useRuntimeStore.getState().setNotice(`${description.name} ${options.autoRun === false ? "placed" : "dropped into the scene"}`);
    return description;
  }

  removeBody(bodyId: string) {
    const sceneStore = useSceneStore.getState();
    if (!sceneStore.scene.rigidBodies.some((body) => body.id === bodyId)) return;
    this.recordHistory("remove body");
    const descriptions = sceneStore.scene.rigidBodies.filter((body) => body.id !== bodyId);
    sceneStore.patchScene({ rigidBodies: descriptions });
    this.bodies = this.bodies.filter((body) => body.description.id !== bodyId);
    this.publishBodies();
    // Deleting is not a way of selecting something else: land on empty rather
    // than jumping the gizmo onto whichever body happens to be first.
    useUIStore.getState().select(undefined);
    useRuntimeStore.getState().setRunState("paused");
    useRuntimeStore.getState().setNotice("Body removed");
  }

  /**
   * Held numeric controls fire per input event, so the default coalesce key is
   * the edited field: one undo entry per property the user swept, not one per
   * pixel of slider travel.
   */
  updateBody(bodyId: string, patch: Partial<RigidBodyDescription>, coalesceKey = `body:${bodyId}:${Object.keys(patch).sort().join(",")}`) {
    const sceneStore = useSceneStore.getState();
    this.recordHistory("body edit", coalesceKey);
    const descriptions = sceneStore.scene.rigidBodies.map((body) => body.id === bodyId ? { ...body, ...patch } : body);
    sceneStore.patchScene({ rigidBodies: descriptions });
    const description = descriptions.find((item) => item.id === bodyId);
    if (description) this.bodies = this.bodies.map((body) => {
      if (body.description.id !== bodyId) return body;
      const updated = initializeRigidBody(description);
      updated.position_m = { ...(patch.position_m ?? body.position_m) };
      return updated;
    });
    this.publishBodies();
    useRuntimeStore.getState().setNotice("Body parameters updated");
  }

  resetBody(bodyId: string) {
    this.bodies = this.bodies.map((body) => body.description.id === bodyId ? initializeRigidBody(body.description) : body);
    this.publishBodies();
    useRuntimeStore.getState().setRunState("paused");
  }

  dropBody(bodyId: string) {
    const scene = useSceneStore.getState().scene;
    this.bodies = this.bodies.map((body) => {
      if (body.description.id !== bodyId) return body;
      return initializeRigidBody({ ...body.description, position_m: { x: body.position_m.x, y: scene.container.height_m + boundingRadius(body.description) + 0.08, z: body.position_m.z }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
    });
    this.publishBodies();
    useRuntimeStore.getState().setRunState("running");
    useRuntimeStore.getState().setNotice("Body released with buoyancy, drag, torque, and fluid reaction enabled");
  }

  /** Kinematic constraint shared by the physics drag and the editor gizmo. */
  private applyBodyManipulation(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"]) {
    if (phase === "end") this.kinematicDrag = null;
    else this.kinematicDrag = { bodyId, position: { ...position }, velocity: { ...velocity } };
    const body = this.bodies.find((candidate) => candidate.description.id === bodyId);
    if (!body) return;
    // Held is the whole of "no gravity": the pose below is a command, and a
    // body that also integrated would sink out of the hand between the frames
    // where the pointer happens not to move.
    body.held = phase !== "end";
    body.position_m = { ...position };
    if (orientation) body.orientation = { ...orientation };
    body.linearVelocity_m_s = phase === "end" ? { x: 0, y: 0, z: 0 } : { ...velocity };
    body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 };
    this.publishBodies();
  }

  dragBody(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"]) {
    this.applyBodyManipulation(bodyId, position, velocity, phase, orientation);
    const runtime = useRuntimeStore.getState();
    if (phase === "start") { runtime.setRunState("running"); runtime.setNotice("Kinematic drag active · GPU immersed boundary coupling"); }
    if (phase === "end") runtime.setNotice("Body released to buoyancy, drag, and collision response");
  }

  /**
   * Gizmo manipulation. Same kinematic constraint as `dragBody`, but authoring
   * never starts the clock: placing geometry is an edit, not a throw. The pose
   * lives in runtime state only until `commitEdit` writes the document.
   */
  manipulateBody(bodyId: string, position: RigidBodyState["position_m"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"]) {
    this.applyBodyManipulation(bodyId, position, { x: 0, y: 0, z: 0 }, phase, orientation);
  }

  // ---- renderer callbacks ------------------------------------------------

  /** Publish transport time only after the corresponding GPU work completes. */
  gpuAdvanceCompleted(time_s: number) {
    const completed = commitGPUCompletion(this.simulationTime, this.gpuCompletedTime, time_s);
    if (completed === this.gpuCompletedTime) return;
    this.gpuCompletedTime = completed;
    useRuntimeStore.getState().setSimulationTime(completed);
  }

  /** Drop host-side debt when paused, retaining only work already admitted to the GPU queue. */
  gpuSchedulingPaused(submittedTime_s?: number) {
    const submitted = submittedTime_s !== undefined && Number.isFinite(submittedTime_s)
      ? Math.max(this.gpuCompletedTime, submittedTime_s)
      : this.gpuCompletedTime;
    // A reset can pause while the renderer still owns the previous solver.
    if (submitted > this.simulationTime + 1e-9) return;
    this.simulationTime = submitted;
    this.accumulator = 0;
  }

  recordFrame(metrics: RendererFrameMetrics, resolution: string) {
    const diagnostics = useDiagnosticsStore.getState();
    const methodId = metrics.methodId ?? useMethodStore.getState().methodId;
    const context = metrics.context ?? methodId;
    diagnostics.set({
      frameMs: metrics.cpu?.total_ms ?? 0,
      resolution,
      waterSurfacePresentation: metrics.waterSurfacePresentation ?? null,
    });
    const instrumentation = usePerformanceInstrumentationStore.getState();
    if (!instrumentation.enabled) {
      this.pendingCpuTickTrace = undefined;
      return;
    }
    const capturedAt_ms = performance.now();
    const reportDue = context !== this.lastPerformanceReportContext
      || capturedAt_ms - this.lastPerformanceReportAt_ms >= 100;
    if (!reportDue) return;
    this.lastPerformanceReportAt_ms = capturedAt_ms;
    this.lastPerformanceReportContext = context;
    const physicsTrace = diagnostics.gpuInfo?.physicsTrace;
    const rendererCPU = metrics.cpu && metrics.cpu.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(metrics.cpu, "cpu", "main-thread") ? metrics.cpu : undefined;
    const controllerCPU = this.pendingCpuTickTrace
      && this.pendingCpuTickTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(this.pendingCpuTickTrace, "cpu", "main-thread")
      ? this.pendingCpuTickTrace
      : undefined;
    this.pendingCpuTickTrace = undefined;
    const physics = physicsTrace && physicsTrace.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(physicsTrace, "gpu", "physics") ? physicsTrace : undefined;
    const cpu = performanceReportCPUTrace({
      physics,
      physicsCPU: diagnostics.gpuInfo?.physicsCPUTrace,
      physicsCaptureIdentity: diagnostics.gpuInfo?.physicsCaptureIdentity,
      controllerCPU,
      rendererCPU,
      context,
    });
    const report = {
      methodId,
      context,
      capturedAt_ms,
      cpu,
      physics,
      presentation: metrics.presentation && metrics.presentation.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(metrics.presentation, "gpu", "presentation") ? metrics.presentation : undefined,
      presentationStages: metrics.presentationStages
        && metrics.presentationStages.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(metrics.presentationStages, "gpu", "presentation")
        ? metrics.presentationStages : undefined,
      presentationBands: metrics.presentationBands
        && metrics.presentationBands.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(metrics.presentationBands, "gpu", "presentation")
        ? metrics.presentationBands : undefined,
      // No lane or timestamp guard: the manifest is a statement about the
      // encode, not a measurement of it, and it is republished every recorded
      // frame. Instrumentation being off is the only reason it can be absent.
      presentationStageManifest: metrics.presentationStageManifest,
    };
    diagnostics.pushPerformanceReport(report);
    const activityStore = usePerformanceActivityStore.getState();
    if (activityStore.enabled && (cpu || report.physics || report.presentation)) {
      const identity: ActivityFrameIdentity = report.physics
        ? {
          frameId: gpuPhysicsPerformanceActivityFrameId(report.physics),
          generation: activityStore.generation,
        }
        : {
          frameId: `${methodId}:${context}:${capturedAt_ms.toFixed(3)}`,
          generation: activityStore.generation,
        };
      const baseFrame = synthesizePerformanceActivityFrame({
        identity,
        context,
        capturedAt_cpu_ms: capturedAt_ms,
        cpu,
        physics: report.physics,
        presentation: report.presentation,
      });
      activityStore.publish(baseFrame);
    }
  }

  // ---- draft manipulation --------------------------------------------------

  /**
   * Open a direct-manipulation gesture: the draft store takes the proposals,
   * and the document is not touched until `commitDraft`.
   */
  beginDraft(subject: SceneDraftSubject, label: string) {
    useSceneDraftStore.getState().beginDraft(subject, label);
    this.beginEdit(label);
  }

  /**
   * Land the open draft on the document — one write, one re-seed, one undo
   * entry — or drop it if the gesture was interrupted.
   *
   * `announceRebuild` is for the gestures that move the lattice (the tank), so
   * the status line names the edit that is costing the pause instead of
   * reporting an anonymous rebuild.
   */
  commitDraft(options: { announceRebuild?: string } = {}) {
    const draftStore = useSceneDraftStore.getState();
    const draft = draftStore.draft;
    draftStore.clearDraft();
    if (!draft) { this.cancelEdit(); return false; }
    if (options.announceRebuild) this.announceGPURebuild(options.announceRebuild);
    useSceneStore.getState().patchScene(draft.patch);
    return this.commitEdit(undefined, { reseed: true });
  }

  /** Abandon the open gesture, leaving the committed scene untouched. */
  cancelDraft() {
    useSceneDraftStore.getState().clearDraft();
    this.cancelEdit();
  }

  // ---- scene scale ---------------------------------------------------------

  /**
   * Scale the world or the detail by a factor of two.
   *
   * A world scale keeps the lattice dimensions and warm re-seeds its updated
   * metre mapping. A detail scale moves the lattice and rebuilds. Compiled
   * pipelines remain cached.
   */
  scaleScene(axis: SceneScaleAxis, factor: SceneScaleFactor) {
    const sceneStore = useSceneStore.getState();
    const next = scaleSceneBy(sceneStore.scene, axis, factor);
    if (!next) {
      const blocked = sceneScaleOption(sceneScaleSummary(sceneStore.scene), axis, factor).blocked;
      useRuntimeStore.getState().setNotice(`Scale refused · ${blocked ?? "unavailable at this size"}`);
      return false;
    }
    const label = `${factor > 1 ? "Doubled" : "Halved"} the ${axis === "world" ? "world size" : "detail"}`;
    if (axis === "detail") this.announceGPURebuild(label);
    this.beginEdit(label);
    sceneStore.setScene(next, sceneStore.presetId);
    this.commitEdit(undefined, { reseed: true });
    const [nx, ny, nz] = sceneLatticeDimensions(next);
    useRuntimeStore.getState().setNotice(`${label} · ${nx}×${ny}×${nz} cells at ${next.voxelDomain.finestCellSize_m.toFixed(4)} m`);
    return true;
  }

  // ---- initial water body --------------------------------------------------

  /**
   * Reshape the initial water body.
   *
   * Called once, when a shape gesture is released. There is deliberately no
   * per-move variant: the water body lives in the solver's seed tier, so every
   * document write asks the renderer to re-seed, and writing on each pointer
   * move turned a smooth drag into dozens of re-seeds a second. The viewport
   * previews the box itself and spends the one re-seed here.
   */
  shapeFluidBody(box: FluidBodyBox) {
    const sceneStore = useSceneStore.getState();
    sceneStore.patchScene(fluidBodyBoxPatch(sceneStore.scene, box));
    this.commitEdit(undefined, { reseed: true });
  }

  /**
   * Resize the tank to a dragged box. Structural — the lattice moves — so this
   * is announced and, like `shapeFluidBody`, only ever called on release.
   */
  resizeTank(box: FluidBodyBox) {
    const sceneStore = useSceneStore.getState();
    const extents = {
      width_m: box.max.x - box.min.x,
      height_m: box.max.y - box.min.y,
      depth_m: box.max.z - box.min.z,
    };
    if (tankResizeIsStructural(sceneStore.scene, extents)) this.announceGPURebuild("Resize the tank");
    sceneStore.patchScene(tankResizePatch(sceneStore.scene, extents));
    this.commitEdit(undefined, { reseed: true });
  }

  /**
   * Grow or shrink the water body about its own centre, without a drag.
   * `volumeFactor` is a volume, so ×2 is twice the water.
   */
  scaleFluidBody(volumeFactor: number) {
    const sceneStore = useSceneStore.getState();
    const box = fluidBodyBox(sceneStore.scene);
    if (!box) { useRuntimeStore.getState().setNotice("No water body to resize"); return false; }
    this.beginEdit(volumeFactor > 1 ? "Grew the water body" : "Shrank the water body");
    this.shapeFluidBody(scaleFluidBodyVolume(box, volumeFactor, sceneStore.scene));
    return true;
  }

  // ---- persistence -------------------------------------------------------

  // ---- render-only scenery -------------------------------------------------

  /**
   * Scenery extends the sparse render domain but never enters the solve, so it
   * is the one authoring action that does not need a re-seed.
   */
  addScenery(kind: SceneryPropKind, point_m: RigidBodyState["position_m"], normal: RigidBodyState["position_m"]) {
    const sceneStore = useSceneStore.getState();
    const node = createSceneryNodeAt(sceneStore.scene, kind, point_m, normal);
    this.recordHistory(`place ${kind}`);
    // A whole scene rather than a patch: the node joins the same list the
    // preset's own scenery lives in, and a merge patch cannot append.
    sceneStore.setScene(addSceneryNode(sceneStore.scene, node), sceneStore.presetId);
    useUIStore.getState().select({ kind: "scenery", id: scenerySelectionId(node.id) });
    useRuntimeStore.getState().setNotice(`${node.id} placed`);
    return node;
  }

  /**
   * Remove whatever is selected, from the scene the entity handed back.
   *
   * A whole scene rather than a patch because removing the last prop drops the
   * `props` key, and a merge patch cannot express an absence — so the entity
   * composes the document it wants and this only lands it.
   */
  removeEntity(label: string, next: SceneDescription) {
    const sceneStore = useSceneStore.getState();
    this.beginEdit(label);
    sceneStore.setScene(next, sceneStore.presetId);
    useUIStore.getState().select(undefined);
    this.commitEdit(undefined, { reseed: true });
    useRuntimeStore.getState().setNotice(label);
  }

  removeScenery(id: string) {
    const sceneStore = useSceneStore.getState();
    if (!findSceneryNode(sceneStore.scene, id)) return;
    this.recordHistory("remove scenery");
    sceneStore.setScene(withoutSceneryNode(sceneStore.scene, id), sceneStore.presetId);
    useUIStore.getState().select(undefined);
    useRuntimeStore.getState().setNotice(`${id} removed`);
  }

  // ---- named scene library ------------------------------------------------

  /** Save the live document under a name, replacing an entry of the same name. */
  saveNamedScene(name: string) {
    const sceneStore = useSceneStore.getState();
    const { methodId, quality, overrides } = useMethodStore.getState();
    const { entries, entry } = saveSceneToLibrary(
      browserSceneLibraryStorage(), name, sceneStore.scene, sceneStore.presetId, {
        savedAt_ms: Date.now(),
        methodProfile: { methodId, quality, overrides: { ...(overrides[methodId] ?? {}) } },
      },
    );
    useRuntimeStore.getState().setNotice(`Saved “${entry.name}”`);
    return entries;
  }

  /**
   * Load a library entry as a scene edit, so it is undoable like any other.
   *
   * Routed through the same card as the library grid, which is how a saved
   * validation scene regained the solver profile it was authored under — this
   * path used to drop it and run whatever method happened to be selected.
   */
  loadNamedScene(entry: SceneLibraryEntry): boolean {
    return this.openSceneCard(savedSceneCard(entry));
  }

  loadLocalScene(): boolean {
    const stored = localStorage.getItem("fluid-lab.scene.v1");
    if (!stored) return false;
    try { const loaded = parseScene(stored); this.reset(loaded); useRuntimeStore.getState().setNotice("Loaded the last local scene"); }
    catch { useRuntimeStore.getState().setNotice("Stored scene failed validation", "warn"); }
    return true;
  }

  importScene(name: string, contents: string) {
    try { const loaded = parseScene(contents); this.recordHistory(`import ${name}`); this.reset(loaded); useRuntimeStore.getState().setNotice(`Loaded ${name}`); }
    catch (error) { useRuntimeStore.getState().setNotice(error instanceof Error ? error.message : "Scene import failed", "warn"); }
  }

  applyAndResetFluid() {
    this.reset(cloneScene(useSceneStore.getState().scene));
  }

  /**
   * Put the named URL overrides back to what this scene was authored with.
   *
   * The one way a reader gets out of a link someone else tuned. Each key is
   * restored individually — see `sceneOverrideClearPlan` for why this is not a
   * re-parse of the reduced query — and the whole set is one undo entry, so
   * clearing a scene by accident costs a single Ctrl-Z.
   *
   * Startup flags come back as `plan.reload`: `gpu=off` was consumed before any
   * store existed, so the only honest way to retire it is to leave the address
   * without it and load the page again. That is a navigation, and it is the
   * caller's to perform — this reports the keys it could not clear.
   */
  clearOverrides(keys: readonly string[]): { readonly reload: readonly string[] } {
    const sceneStore = useSceneStore.getState();
    const plan = sceneOverrideClearPlan(keys, { scene: sceneStore.scene, presetId: sceneStore.presetId });
    const label = keys.length === 1 ? `clear ${keys[0]}` : `clear ${keys.length} overrides`;

    if (plan.scene) {
      // Through the editor gesture rather than `reset`, so that clearing an
      // override the running solver can already adopt — a density, the hose —
      // does not throw away the simulation being watched to apply it.
      this.beginEdit(label);
      sceneStore.setScene(plan.scene);
      this.commitEdit(undefined, { reseed: true });
    }
    const ui = useUIStore.getState();
    if (Object.keys(plan.ui).length > 0) useUIStore.setState(plan.ui);
    if (Object.keys(plan.svoRenderTuning).length > 0) {
      ui.setSvoRenderTuning({ ...ui.svoRenderTuning, ...plan.svoRenderTuning });
    }
    // Parameters before the method, so a restored value lands on the method it
    // belongs to rather than triggering a rebuild of the one being left.
    for (const param of plan.methodParams) {
      if (param.value === undefined) this.resetMethodParam(param.methodId, param.key);
      else this.setMethodParam(param.methodId, param.key, param.value);
    }
    if (plan.quality) this.setQuality(plan.quality);
    if (plan.methodId) this.setMethod(plan.methodId);

    if (!plan.scene && plan.reload.length === keys.length) return { reload: plan.reload };
    useRuntimeStore.getState().setNotice(keys.length === 1
      ? `Restored ${keys[0]} to the authored scene`
      : `Restored ${keys.length - plan.reload.length} overrides to the authored scene`);
    return { reload: plan.reload };
  }
}

export { BUILD_ID };

type SimulationControllerWindow = Window & {
  /**
   * The WebGPU viewport deliberately survives React Fast Refresh so compiled
   * pipelines and the live device are not destroyed. Its controller must have
   * the same lifetime: replacing only this singleton leaves the retained draw
   * loop reading an old clock while refreshed transport controls write a new
   * one, making Play and Step appear inert until a full page reload.
   */
  __fluidLabSimulationController?: SimulationController;
};

const simulationWindow = typeof window === "undefined"
  ? undefined
  : window as SimulationControllerWindow;
const retainedSimulation = simulationWindow?.__fluidLabSimulationController;
if (retainedSimulation) {
  // Pick up edited methods while preserving the live clock, bodies, and GPU
  // completion state owned by the retained WebGPU session.
  Object.setPrototypeOf(retainedSimulation, SimulationController.prototype);
}
export const simulation = retainedSimulation ?? new SimulationController();
if (simulationWindow) simulationWindow.__fluidLabSimulationController = simulation;
