import { BUILD_ID, canonicalScene, cloneScene, parseScene, type RunState, type SceneDescription } from "../model";
import { adoptRigidBodyRoster, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBodies, initializeRigidBody, rigidDiagnostics, type RigidBodyState, type RigidStepDiagnostics } from "../rigid-body";
import type { RigidBodyDescription } from "../model";
import type { RigidShape, Vec3 } from "../model";
import { BROWSER_GPU_THROUGHPUT_DEPTH, sceneEditRequiresReset } from "../webgpu-renderer";
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
import { defaultSession, type PaneId, type PaneSession } from "../session/session";
import type { EditorHistorySnapshot } from "../stores/history-store";
import type { GPUQuality } from "../gpu-quality";
import type { MethodParamValue } from "../method-contract";
import { emptyPerformanceReport } from "../stores/diagnostics-store";
import {
  gpuPhysicsPerformanceActivityFrameId,
  synthesizePerformanceActivityFrame,
  type ActivityFrameIdentity,
} from "../performance-activity";
import { usePerformanceActivityStore } from "../stores/performance-activity-store";
import { usePerformanceInstrumentationStore } from "../stores/performance-instrumentation-store";
import { collapseGPUFixedSteps } from "./gpu-clock";
import { LOCKSTEP_IN_FLIGHT_DEPTH, PaneClockHost, PRIMARY_PANE_ID, type PaneClockReport } from "./pane-clock";
import { applyHostRunState, hostResetPlan, paneResetPlan } from "./pane-transport";
import { safeBrowserGPUBringupEnabled } from "../gpu-startup";
import { planSceneRuntime } from "../scene-runtime";
import { resourceInteractionGates } from "../resource-readiness";
import { addSceneryNode, createSceneryNodeAt, scenerySelectionId } from "../editor-scenery";
import { findSceneryNode, withoutSceneryNode } from "../scenery-edit";
import { scaleScene as scaleSceneBy, sceneScaleOption, sceneScaleSummary, type SceneScaleAxis, type SceneScaleFactor } from "../scene-scale";
import { sceneLatticeDimensions } from "../scene-lattice";
import { fluidBodyBox, fluidBodyBoxPatch, scaleFluidBodyVolume, type FluidBodyBox } from "../editor-fluid-body";
import { tankResizeIsStructural, tankResizePatch } from "../editor-tank";
import type { SceneDraftSubject } from "../stores/scene-draft-store";
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

/**
 * The mutable runtime one pane's document owns.
 *
 * Everything here is a property of *a document being edited and advanced*, not
 * of the product's transport: the rigid roster the solver is stepping, the
 * kinematic constraint a hand is holding a body with, the gesture waiting to be
 * committed, and how recently this pane published a performance report. A
 * second pane is a second document, so a second one of these — which is the
 * whole reason editing geometry in B used to edit A: one roster, one pending
 * edit and one carry served both panes.
 *
 * What stays on the controller instead, because it is the *host's*: the clock,
 * the run state, the rate readout, and the CPU trace of the host tick.
 */
/**
 * How much of the studio a scene open is allowed to take with it.
 *
 * The library front door is a clean entry: arriving at a card means arriving at
 * the scene its author intended, so it resets the solver, the quality, the
 * tuning and every view setting. That is the default and it stays the default.
 *
 * `retainConfiguration` is the other gesture, and the one the studio's scene
 * selector makes: swapping the document *under a standing experiment*. Here the
 * solver, its parameters, the raised instrument and the field slice are the
 * experiment — the reader set them up in order to look at something — and a
 * scene swap that silently re-chose them would end the experiment rather than
 * move it. It matters most in compare mode, where those keys are shared with
 * the other pane: a reset would be recorded as a diff nobody asked for, or
 * pushed onto pane A through a padlocked group. The camera is the deliberate
 * exception; see `docs/ab-compare-handoff.md`.
 */
export interface OpenSceneCardOptions {
  readonly retainConfiguration?: boolean;
}

interface PaneRuntime {
  /** Rigid states the renderer advances; rebuilt by `reset`, adopted by edits. */
  bodies: RigidBodyState[];
  /** The carry / gizmo constraint, re-applied every step it survives. */
  kinematicDrag: { bodyId: string; position: RigidBodyState["position_m"]; velocity: RigidBodyState["linearVelocity_m_s"] } | null;
  /** Document captured when a direct-manipulation gesture opened. */
  pendingEdit?: { label: string; snapshot: EditorHistorySnapshot };
  /** This pane's report cadence — two panes must not throttle each other. */
  lastPerformanceReportAt_ms: number;
  lastPerformanceReportContext: string;
}

function createPaneRuntime(): PaneRuntime {
  return {
    bodies: [],
    kinematicDrag: null,
    lastPerformanceReportAt_ms: 0,
    lastPerformanceReportContext: "",
  };
}

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

export { collapseGPUFixedSteps };
export { PRIMARY_PANE_ID } from "./pane-clock";

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
  /**
   * The host clock. One target, N pane completions: single-pane mode is the
   * one-pane case of the same barrier, not a separate path.
   */
  private clock = new PaneClockHost();
  /**
   * The realm each pane authors, runs and reports in.
   *
   * Pane A is the module-level default store set, so a single-pane product is
   * this map with one entry and every `this.session(...)` below resolves to
   * exactly the store objects this controller always read. That equality is
   * what lets the migration land as a pure refactor.
   */
  private paneSessions = new Map<PaneId, PaneSession>([[PRIMARY_PANE_ID, defaultSession]]);
  /**
   * The document runtime each pane is editing and advancing, keyed the same way
   * as its realm. Pane A's entry exists from construction, so a single-pane
   * product is this map with one entry and every `this.runtime(...)` below
   * resolves to exactly the fields this controller used to hold directly.
   */
  private paneRuntimes = new Map<PaneId, PaneRuntime>([[PRIMARY_PANE_ID, createPaneRuntime()]]);
  private lastClock: number | null = null;
  private rateWallClock = 0;
  private rateSimTime = 0;
  private safeBrowserStepConsumed = false;
  private cpuTickTraceSampleId = 0;
  private pendingCpuTickTrace?: PerformanceTrace;

  private safeBrowserBringup(): boolean {
    return typeof location !== "undefined" && safeBrowserGPUBringupEnabled(location.search);
  }

  private webgpuTransportReady(paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const diagnostics = this.session(paneId).diagnostics.getState();
    if (!resourceInteractionGates(diagnostics.resourceReadiness, true).transportInteractive) return false;
    // A method that publishes its sparse authority and raster separately has
    // to cross that fence before transport unlocks; one that attaches its
    // field textures atomically with the warmed solver does not. The method
    // declares which it is, so this never has to list ids.
    return !requiresFencedInitialRasterPresentation(this.session(paneId).method.getState().methodId)
      || (diagnostics.gpuInfo?.initialSparseAuthorityReady === true
        && diagnostics.gpuInfo?.initialRasterSurfaceReady === true);
  }

  constructor() {
    // Pane A, and only pane A: the controller is constructed once, with the
    // module-level realm, before any compare pane can exist.
    const scene = this.session(PRIMARY_PANE_ID).scene.getState().scene;
    // A reset lands on a different scene, so a proposal made against the old
    // one describes nothing. Drop it before anything can draw from it.
    this.session(PRIMARY_PANE_ID).sceneDraft.getState().clearDraft();
    const runtime = this.runtime(PRIMARY_PANE_ID);
    runtime.bodies = initializeRigidBodies(scene.rigidBodies);
    this.publishBodies(rigidDiagnostics(runtime.bodies, scene.fluid.gravity_m_s2));
    // Open on an empty selection: with no gizmo on screen the first drag
    // orbits the scene, which is what someone arriving at a viewport expects.
    // Selecting is the user's move to make.
    this.session(PRIMARY_PANE_ID).ui.getState().select(undefined);
  }

  /** The target every pane's transport advances toward. */
  time(): number { return this.clock.targetTime(); }
  /** The slowest pane's completed state: the time the product is actually at. */
  completedTime(): number { return this.clock.completedTime(); }
  currentBodies(paneId: PaneId = PRIMARY_PANE_ID): RigidBodyState[] { return this.runtime(paneId).bodies; }

  // ---- panes -------------------------------------------------------------

  /**
   * Adopt a controller retained across a Fast Refresh. Field initializers run
   * in the constructor, which a retained instance never re-enters, so the host
   * clock of a session that started before this module was edited has to be
   * rebuilt from whatever clock fields that instance still carries. Losing it
   * would strand the live draw loop on a target that never advances again.
   */
  adoptRetainedRuntime() {
    const retained = this as unknown as {
      clock?: PaneClockHost;
      paneSessions?: Map<PaneId, PaneSession>;
      paneRuntimes?: Map<PaneId, PaneRuntime>;
      simulationTime?: number;
      gpuCompletedTime?: number;
      bodies?: RigidBodyState[];
      kinematicDrag?: PaneRuntime["kinematicDrag"];
      pendingEdit?: PaneRuntime["pendingEdit"];
      lastPerformanceReportAt_ms?: number;
      lastPerformanceReportContext?: string;
    };
    // A controller retained from before the realm ledger existed has no map at
    // all, and `this.session()` would throw on the very first draw.
    if (!(retained.paneSessions instanceof Map) || retained.paneSessions.size === 0) {
      this.paneSessions = new Map<PaneId, PaneSession>([[PRIMARY_PANE_ID, defaultSession]]);
    }
    // Same for the document runtime, but this one carries live state a refresh
    // must not drop: the rigid roster the GPU is mid-advance on, and the carry
    // whose hand is still on the cursor. A controller from before the runtime
    // was per pane held them as three fields; migrate the single slot into
    // pane A's rather than starting the roster empty, which would strand the
    // draw loop on a scene with no bodies until the next reset.
    if (!(retained.paneRuntimes instanceof Map) || retained.paneRuntimes.size === 0) {
      this.paneRuntimes = new Map<PaneId, PaneRuntime>([[PRIMARY_PANE_ID, {
        ...createPaneRuntime(),
        bodies: Array.isArray(retained.bodies) ? retained.bodies : [],
        kinematicDrag: retained.kinematicDrag ?? null,
        pendingEdit: retained.pendingEdit,
        lastPerformanceReportAt_ms: retained.lastPerformanceReportAt_ms ?? 0,
        lastPerformanceReportContext: retained.lastPerformanceReportContext ?? "",
      }]]);
    }
    if (retained.clock && typeof retained.clock.targetTime === "function") {
      // A clock built by the previous evaluation of this module holds the live
      // pane registrations and completions as its own properties but answers to
      // an old prototype. Re-point it exactly as the controller itself is
      // re-pointed; `instanceof` cannot be used here for the same reason.
      Object.setPrototypeOf(retained.clock, PaneClockHost.prototype);
      return;
    }
    // Older sessions kept the clock as three fields on the controller.
    const target = Number.isFinite(retained.simulationTime) ? retained.simulationTime ?? 0 : 0;
    const completed = Number.isFinite(retained.gpuCompletedTime) ? retained.gpuCompletedTime ?? 0 : 0;
    this.clock = new PaneClockHost();
    if (target > 0) this.clock.step(target);
    if (completed > 0) this.clock.completeAdvance(completed);
  }

  /**
   * The realm a call belongs to.
   *
   * Unnamed callers are pane A, which is why every method below takes its pane
   * as a defaulted trailing argument: the product's existing call sites keep
   * meaning exactly what they meant, and a compare host names the pane it is
   * acting for.
   */
  session(paneId: PaneId = PRIMARY_PANE_ID): PaneSession {
    return this.paneSessions.get(paneId) ?? defaultSession;
  }

  /**
   * The document runtime a call acts on, created on first reach.
   *
   * Lazily rather than only on attach, because a renderer callback can arrive
   * for a pane in the window between its viewport mounting and its session
   * being handed over; an absent entry there would silently write pane A's
   * roster, which is exactly the bug the map exists to end.
   */
  private runtime(paneId: PaneId = PRIMARY_PANE_ID): PaneRuntime {
    const existing = this.paneRuntimes.get(paneId);
    if (existing) return existing;
    const created = createPaneRuntime();
    this.paneRuntimes.set(paneId, created);
    return created;
  }

  /**
   * Hand a pane its realm. Compare mode does this before registering the pane
   * on the clock, so the first thing the new pane's transport reports already
   * lands in its own stores rather than in pane A's.
   */
  attachPaneSession(paneId: PaneId, session: PaneSession) {
    // A fresh document runtime with a *new* realm, and only then: re-attaching
    // the realm a pane already holds is what an effect re-running does, and
    // that must not drop the roster mid-carry. A pane handed a different realm
    // is a different document and starts with an empty one.
    if (this.paneSessions.get(paneId) !== session) {
      this.paneRuntimes.set(paneId, createPaneRuntime());
    }
    this.paneSessions.set(paneId, session);
  }

  /** Leaving compare mode. Pane A's realm is the session and never detaches. */
  detachPaneSession(paneId: PaneId): boolean {
    if (paneId === PRIMARY_PANE_ID) return false;
    this.paneRuntimes.delete(paneId);
    return this.paneSessions.delete(paneId);
  }

  /**
   * Publish the host clock into every pane.
   *
   * Host-level on purpose: the published time is the *minimum* over the panes,
   * so both transports must read the same t or the product would claim two
   * different states of one experiment. It is the one runtime field that is not
   * a per-pane measurement.
   */
  private publishSimulationTime() {
    const completed = this.clock.completedTime();
    for (const session of this.paneSessions.values()) {
      session.runtime.getState().setSimulationTime(completed);
    }
  }

  /**
   * Admit a pane to the host clock. Pane A is registered at construction; a
   * second registration puts the host in lockstep, where the target advances
   * one step at a time and only once every pane has reached it.
   */
  registerPane(paneId: PaneId) {
    if (!this.clock.registerPane(paneId)) return;
    this.publishSimulationTime();
  }

  /** Leaving compare mode. Pane A is the session and is never unregistered. */
  unregisterPane(paneId: PaneId) {
    if (!this.clock.unregisterPane(paneId)) return;
    this.publishSimulationTime();
  }

  paneIds(): readonly string[] { return this.clock.paneIds(); }
  paneClocks(): readonly PaneClockReport[] { return this.clock.reports(); }

  // ---- transport ---------------------------------------------------------

  /**
   * Play/pause. The transport is the *host's*, so this reaches every pane.
   *
   * Not a pane's, even though the run state is stored per session: there is one
   * target clock, and a pause that flipped only pane A left pane B's renderer
   * believing it was still running. Two consequences, both bugs a reader sees:
   * pane B never took the pause boundary its `readStats` readback rides on, so
   * the diff strip's B column stayed frozen at its t=0 volume and reported a
   * divergence between two identical panes; and pane B never reported the work
   * it had already admitted to its GPU queue, so `gpuSchedulingPaused` rewound
   * the host target using pane A's drain alone and could orphan an advance pane
   * B had encoded — a completion no solver will re-encode, and therefore a
   * barrier that never opens again.
   *
   * With one pane attached this is exactly the single-pane store write it
   * replaced.
   */
  setRunState(runState: RunState) {
    applyHostRunState(this.paneSessions.values(), runState);
  }

  /** The run state the transport shows: pane A's, which is the host's. */
  runState(): RunState { return this.session(PRIMARY_PANE_ID).runtime.getState().runState; }

  /**
   * Transport queue depth a pane's renderer may run at. Two panes each two
   * deep would let one drift a frame ahead of the other inside the barrier's
   * own tolerance, so lockstep pins the window to a single advance.
   */
  inFlightDepth(): number {
    return this.clock.lockstep() ? LOCKSTEP_IN_FLIGHT_DEPTH : BROWSER_GPU_THROUGHPUT_DEPTH;
  }

  /**
   * A pane whose diff changes `fixedDt_s` or pins a method step declares it
   * here; `undefined` puts the pane back on pane A's step.
   */
  setPaneDt(paneId: PaneId, dt_s: number | undefined) { this.clock.setPaneDt(paneId, dt_s); }

  /**
   * True when a registered pane is not on pane A's step. The host then runs at
   * the smaller step and the larger-dt pane skips the steps it does not need —
   * a comparison of rates rather than of paired steps, which the diff strip
   * has to say out loud.
   */
  panesDtDiffer(): boolean { return this.clock.panesDtDiffer(); }

  private publishBodies(diagnostics?: RigidStepDiagnostics, paneId: PaneId = PRIMARY_PANE_ID) {
    const scene = this.session(paneId).scene.getState().scene;
    const bodies = this.runtime(paneId).bodies;
    this.session(paneId).diagnostics.getState().set({ bodies: cloneRigidBodies(bodies), rigidState: diagnostics ?? rigidDiagnostics(bodies, scene.fluid.gravity_m_s2) });
  }

  /**
   * Prepare every fixed step owed by the wall clock. GPU admission is
   * renderer-budgeted.
   *
   * Host-level, so every store read here is pane A's on purpose. There is one
   * target clock and one transport: the run state, the rate readout and the
   * step size the whole product advances at are the host's, and `dt` is pane
   * A's effective step by the design's own rule. A pane whose diff pins a
   * different step declares it through `setPaneDt`, and the host then runs at
   * the smaller one.
   */
  tick(now: number) {
    const method = this.session(PRIMARY_PANE_ID).method.getState();
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
    const runtime = this.session(PRIMARY_PANE_ID).runtime.getState();
    // Whether there is anything to advance is the *host's* question, not pane
    // A's. Panes may hold different scenes, so a scenery-only document in A
    // must not stop a pane B that has water in it; the clock stops only when
    // no attached pane plans a solver.
    let hostFluidSolver = false;
    for (const pane of this.paneSessions.values()) {
      if (planSceneRuntime(pane.scene.getState().scene).fluidSolver) { hostFluidSolver = true; break; }
    }
    if (!hostFluidSolver) {
      this.setRunState("paused");
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.clock.dropPendingTime();
      this.rateWallClock = 0;
      return;
    }
    if (this.safeBrowserBringup() && runtime.runState === "running") {
      this.setRunState("paused");
      runtime.setSimRate(null);
      this.clock.dropPendingTime();
      this.rateWallClock = 0;
      return;
    }
    if (runtime.runState !== "running") {
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    const scene = this.session(PRIMARY_PANE_ID).scene.getState().scene;
    if (!this.webgpuTransportReady()) {
      this.clock.dropPendingTime();
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    const dt = effectiveSimulationStep_s(scene, this.session(PRIMARY_PANE_ID).method.getState());
    cpuTrace?.transition({ id: "frame-control", label: "GPU target-clock control" });
    // Every pane's renderer admits GPU work against this one target clock.
    // With a single pane, collapsing accumulated fixed ticks is exact because
    // no host fluid/body evolution occurs at the intermediate ticks;
    // solver.advanceTo owns subdivision. With two, the host takes one step per
    // barrier opening instead, so neither pane runs ahead of the other.
    // Every pane's carry, not pane A's: the step the barrier just opened is a
    // step for both documents, and a body held in B has to be held through it.
    if (this.clock.advance(elapsed, dt) > 0) {
      for (const paneId of this.paneRuntimes.keys()) this.applyDragConstraint(paneId);
    }
    if (this.rateWallClock === 0) { this.rateWallClock = now; this.rateSimTime = this.clock.completedTime(); }
    else if (now - this.rateWallClock > 500) {
      const committedTime = this.clock.completedTime();
      runtime.setSimRate((committedTime - this.rateSimTime) / ((now - this.rateWallClock) / 1000));
      this.rateWallClock = now; this.rateSimTime = committedTime;
    }
    } finally {
      this.pendingCpuTickTrace = cpuTrace?.finish();
    }
  }

  private applyDragConstraint(paneId: PaneId = PRIMARY_PANE_ID) {
    const runtime = this.runtime(paneId);
    const drag = runtime.kinematicDrag;
    if (!drag) return;
    const body = runtime.bodies.find((candidate) => candidate.description.id === drag.bodyId);
    if (body) { body.held = true; body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
  }

  /**
   * STEP: one paired step for every pane.
   *
   * Host-level like `tick`, and pane A for the same reasons — the transport is
   * the host's, and the barrier below refuses the step until the slowest pane
   * has landed the previous one.
   */
  singleStep() {
    // Every pane, not pane A: STEP is a transport gesture, and a pane still
    // marked running while the clock stands still is the state that orphans
    // queued work at the next pause. See `setRunState`.
    this.setRunState("paused");
    const scene = this.session(PRIMARY_PANE_ID).scene.getState().scene;
    if (!planSceneRuntime(scene).fluidSolver) return;
    const dt = effectiveSimulationStep_s(scene, this.session(PRIMARY_PANE_ID).method.getState());
    if (!this.webgpuTransportReady()) return;
    // One STEP is one step for every pane: the host refuses it until the
    // slowest pane has landed the previous one.
    if (!this.clock.canAcceptNextStep()) return;
    if (this.safeBrowserBringup() && this.safeBrowserStepConsumed) return;
    if (this.safeBrowserBringup()) this.safeBrowserStepConsumed = true;
    this.clock.step(dt);
  }

  /** Apply one shared fixed step to rigid and fluid work without resetting time. */
  setStepSize(step_s: number, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const numerics = sharedStepNumerics(sceneStore.scene.numerics, step_s);
    // Asking for a step is how you leave the paper step. The uniform paper
    // profile pins the clock at 1/30 s, but the control stays live while it
    // does: an edit releases the profile to the scene-authored dt rather than
    // presenting a number that silently refuses to move.
    const releasedPaperStep = this.releaseUniformPaperStep(paneId);
    if (!releasedPaperStep
      && numerics.fixedDt_s === sceneStore.scene.numerics.fixedDt_s
      && numerics.maxDt_s === sceneStore.scene.numerics.maxDt_s) return;
    sceneStore.patchNumerics(numerics);
    this.clock.clampPendingTime(numerics.fixedDt_s);
    this.session(paneId).runtime.getState().setNotice(releasedPaperStep
      ? `Uniform paper 1/30 s step released · shared rigid + fluid step ${(numerics.fixedDt_s * 1000).toFixed(2)} ms`
      : `Shared rigid + fluid step · ${(numerics.fixedDt_s * 1000).toFixed(2)} ms`);
  }

  /** Switch the uniform method off its paper step; true when it was on one. */
  private releaseUniformPaperStep(paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const { methodId } = this.session(paneId).method.getState();
    if (!methodPinsSimulationStep(this.session(paneId).scene.getState().scene, this.session(paneId).method.getState())) return false;
    this.setMethodParam(methodId, "timeStep", "scene", paneId);
    return true;
  }

  /**
   * Reset one pane's document — and, with it, every other pane's clock.
   *
   * `PaneClockHost.reset` below returns *every* registered pane to zero,
   * because there is one clock. A pane whose document runtime and renderer
   * timeline are not re-seeded with it is then stranded above a target that has
   * rewound past it: its solver refuses to re-encode a time it has already
   * submitted, so it never reports the completion the min-completion barrier is
   * waiting for and the transport dies for *both* panes. That was compare
   * mode's most visible failure — one RESET and the pair froze, pane B still
   * showing the water it had reached while both clocks read t = 0.
   *
   * The pane that was asked pauses, as a reset always has. The others keep the
   * run state they were in, so a scene chosen in pane B does not stop pane A.
   * With one pane attached the fan-out is empty and this is the reset that
   * shipped.
   */
  reset(source?: SceneDescription, presetId?: string, paneId: PaneId = PRIMARY_PANE_ID) {
    for (const step of paneResetPlan(this.paneSessions.keys(), paneId)) {
      this.resetPane(step.id, step.id === paneId ? source : undefined,
        step.id === paneId ? presetId : undefined, step.resynchronizing);
    }
  }

  /**
   * RESET on the transport: every pane back to its own t = 0, and stopped.
   *
   * Distinct from `reset` because the subject is different — a transport
   * gesture is about the experiment, not about a document someone edited — and
   * because the outcome is different: nobody's run state is retained.
   */
  resetAll() {
    for (const step of hostResetPlan(this.paneSessions.keys())) {
      this.resetPane(step.id, undefined, undefined, step.resynchronizing);
    }
  }

  /** Everything a reset does to one pane. */
  private resetPane(
    paneId: PaneId,
    source?: SceneDescription,
    presetId?: string,
    resynchronizing = false,
  ) {
    const sceneStore = this.session(paneId).scene.getState();
    const scene = source ?? sceneStore.scene;
    if (source) sceneStore.setScene(source, presetId);
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = initializeRigidBodies(scene.rigidBodies);
    const runtimePlan = planSceneRuntime(scene);
    // Reset is a host operation: every pane returns to its own t = 0.
    this.clock.reset(); this.lastClock = null;
    this.rateWallClock = 0; this.rateSimTime = 0;
    this.pendingCpuTickTrace = undefined;
    // A carry outlives a rebuild. This used to drop the kinematic constraint
    // unconditionally, so any edit that re-seeded the solver let go of a body
    // the user was still holding: the hand stayed on the cursor and the solid
    // fell out of it with no click anywhere. Kept while the body it names
    // survives into the new roster, and re-applied before the publish below so
    // the body stands where the hand is rather than snapping back to the pose
    // it was authored at.
    const heldId = paneRuntime.kinematicDrag?.bodyId;
    if (heldId !== undefined && !paneRuntime.bodies.some((body) => body.description.id === heldId)) {
      paneRuntime.kinematicDrag = null;
    }
    // The constraint is the only thing that may hold a body, so anything still
    // flagged without one is a release the roster outlived: a gesture whose
    // body was replaced by an edit, or a drag ended while it was off-roster.
    for (const body of paneRuntime.bodies) {
      if (body.held && body.description.id !== paneRuntime.kinematicDrag?.bodyId) body.held = false;
    }
    this.applyDragConstraint(paneId);
    if (!this.safeBrowserBringup()) this.safeBrowserStepConsumed = false;
    const diagnosticsStore = this.session(paneId).diagnostics.getState();
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
        resource: getMethod(this.session(paneId).method.getState().methodId).resource,
      } });
    }
    this.publishBodies(rigidDiagnostics(paneRuntime.bodies, scene.fluid.gravity_m_s2), paneId);
    this.session(paneId).diagnostics.getState().set({ gpuInfo: null, waterSurfacePresentation: null, couplingState: { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 }, samples: [], performanceReport: emptyPerformanceReport, performanceReports: [] });
    const runtime = this.session(paneId).runtime.getState();
    // Publish the new t=0 identity before pausing. Renderer subscribers use
    // this synchronous epoch edge to reject completions from the old queue.
    runtime.resetSimulationTime();
    runtime.setSimRate(null);
    // A pane re-seeded because *another* pane reset keeps running if it was
    // running: its clock had to return to zero with the host's, but nobody
    // asked it to stop.
    if (!resynchronizing) runtime.setRunState("paused");
    // A reset lands on a fresh scene, so nothing carries a selection into it.
    // commitEdit({ reseed: true }) restores the user's selection afterwards.
    // A re-seeded pane's document did not change, so its selection still names
    // something and is left alone.
    if (!resynchronizing) this.session(paneId).ui.getState().select(undefined);
    // Only the pane that was asked says so. The others were re-seeded to keep
    // the one clock honest, and a notice on each would read as two resets.
    if (!resynchronizing) {
      runtime.setNotice(!runtimePlan.fluidSolver
        ? "Live renderer scene reset · fluid solver disabled"
        : `${scene.fluid.inflow ? "Inflow scene" : scene.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
    }
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
  openSceneCard(
    card: SceneCard,
    paneId: PaneId = PRIMARY_PANE_ID,
    options: OpenSceneCardOptions = {},
  ): boolean {
    let opening;
    try {
      opening = card.open();
    } catch (error) {
      this.session(paneId).runtime.getState().setNotice(
        error instanceof Error ? error.message : `${card.name} failed validation`, "warn");
      return false;
    }
    this.recordHistory(`open ${card.name}`, undefined, paneId);
    // A card is a clean scene entry, not a continuation of whichever hidden
    // studio state happened to be retained behind the library. Reset both
    // serializable stores before applying the scene's authored contracts so a
    // prior panel, diagnostic arm, quality experiment, or method override can
    // never leak into another scene or its URL.
    const previousCamera = this.session(paneId).ui.getState().camera;
    if (!options.retainConfiguration) {
      this.session(paneId).method.setState(this.session(paneId).method.getInitialState());
    }
    if (opening.methodProfile) this.session(paneId).method.getState().seedProfile(opening.methodProfile);
    this.reset(opening.scene, opening.presetId, paneId);
    this.session(paneId).ui.setState(options.retainConfiguration ? {
      camera: opening.camera ?? previousCamera,
      // The document changed under them: an armed gesture and an axis lock are
      // about the thing that is no longer there. `reset` has already dropped
      // the selection they would have acted on.
      armedGesture: undefined,
      axisConstraint: undefined,
      selectionControlsOpen: false,
    } : {
      ...this.session(paneId).ui.getInitialState(),
      // Absent only for a saved document whose origin scene this build no
      // longer has; there the camera the reader already had is the better answer.
      camera: opening.camera ?? previousCamera,
    });
    const runtimePlan = planSceneRuntime(opening.scene);
    const effectiveStep_s = effectiveSimulationStep_s(
      opening.scene, this.session(paneId).method.getState());
    this.session(paneId).runtime.getState().setNotice(!runtimePlan.fluidSolver
      ? `${card.name} opened · no fluid solver, so nothing waits on one`
      : `${card.name} opened · dt ${effectiveStep_s.toFixed(4)} s`);
    // The clock a scene opens on is the host's: with two panes attached,
    // starting only the pane that was opened would leave the other holding the
    // transport paused, and neither would move.
    this.setRunState(runtimePlan.fluidSolver ? "running" : "paused");
    useShellStore.getState().enterStudio();
    return true;
  }

  loadPreset(presetId: string, paneId: PaneId = PRIMARY_PANE_ID) {
    this.openSceneCard(sceneCardForDefinition(getSceneDefinition(presetId)), paneId);
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
  newScene(starterId?: string, paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const starter = SCENE_STARTERS.find((candidate) => candidate.id === starterId) ?? SCENE_STARTERS[0];
    return this.openSceneCard({
      id: `starter:${starter.id}`,
      source: "starter",
      name: starter.name,
      blurb: starter.blurb,
      shelf: "Start from empty",
      open: () => ({ scene: starter.create(), presetId: `starter:${starter.id}` }),
    }, paneId);
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
  setFluidSystem(enabled: boolean, paneId: PaneId = PRIMARY_PANE_ID) {
    const scene = this.session(paneId).scene.getState().scene;
    if ((scene.systems?.fluid !== false) === enabled) return;
    const label = enabled ? "Enable water" : "Disable water";
    this.announceGPURebuild(label, paneId);
    this.recordHistory(label.toLowerCase(), undefined, paneId);
    const next = cloneScene(scene);
    next.systems = { ...next.systems, fluid: enabled };
    this.reset(next, undefined, paneId);
    const runtime = this.session(paneId).runtime.getState();
    // The clock is the host's: a rebuild stops the experiment, not one pane.
    this.setRunState("paused");
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
  }, paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const runtime = this.session(paneId).runtime.getState();
    const sceneStore = this.session(paneId).scene.getState();
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
    this.announceGPURebuild(label, paneId);
    this.recordHistory(label.toLowerCase(), undefined, paneId);
    // Nothing to synchronize: the rebuilt document carries the depth, and the
    // renderer reads it from there. That is the whole point of there being one.
    this.reset(rebuiltScene, presetId, paneId);
    this.session(paneId).ui.getState().setCamera(sceneDefinitionCamera(definition));
    // The clock is the host's: a rebuild stops the experiment, not one pane.
    this.setRunState("paused");
    this.session(paneId).runtime.getState().setNotice(`${definition.name} re-authored`
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
  setEnvironmentRefinementDepth(depth: number, paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const runtime = this.session(paneId).runtime.getState();
    const sceneStore = this.session(paneId).scene.getState();
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
      return this.rebuildSceneAtLattice({ environmentRefinementDepth: requested }, paneId);
    }
    const zeroCellSize_m = scene.voxelDomain.environmentRefinementBaseCellSize_m
      ?? scene.voxelDomain.finestCellSize_m;
    const finestCellSize_m = requested < 0 ? zeroCellSize_m * 2 ** -requested : zeroCellSize_m;
    const detailCellSize_m = requested < 0 ? finestCellSize_m : svoSceneryDetailCellSize_m(zeroCellSize_m, {
      environmentRefinementDepth: requested, fluid: false,
    });
    const label = `Draw the set at ${(detailCellSize_m * 1000).toFixed(5).replace(/\.?0+$/, "")} mm`;
    this.announceGPURebuild(label, paneId);
    this.recordHistory(label.toLowerCase(), undefined, paneId);
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

  setQuality(quality: GPUQuality, paneId: PaneId = PRIMARY_PANE_ID) {
    this.announceGPURebuild(`Apply ${quality} quality`, paneId);
    this.session(paneId).method.getState().setQuality(quality);
    this.reset(undefined, undefined, paneId);
    this.session(paneId).runtime.getState().setNotice(`Quality ${quality} · simulation reset`);
  }

  setMethod(methodId: string, paneId: PaneId = PRIMARY_PANE_ID) {
    this.announceGPURebuild(`Switch to ${getMethod(methodId).label}`, paneId);
    this.session(paneId).method.getState().setMethodId(methodId);
    this.reset(undefined, undefined, paneId);
    this.session(paneId).runtime.getState().setNotice(`${getMethod(methodId).label} selected`);
  }

  /** Structural settings start from a defined t=0 state. Runtime-safe settings
   * are applied to the live GPU solver without changing the simulation clock. */
  private announceGPURebuild(operation: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const current = this.session(paneId).diagnostics.getState().gpuStatus;
    this.session(paneId).diagnostics.getState().set({ gpuStatus: {
      state: "initializing",
      label: "Preparing GPU work plan",
      phase: "planning",
      completed: 0,
      total: 0,
      startedAt_ms: performance.now(),
      kind: "rebuild",
      operation,
      retainingPrevious: current.state === "ready" || (current.state === "initializing" && Boolean(current.retainingPrevious)),
      resource: getMethod(this.session(paneId).method.getState().methodId).resource,
    } });
  }

  setMethodParam(methodId: string, key: string, value: MethodParamValue, paneId: PaneId = PRIMARY_PANE_ID) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === this.session(paneId).method.getState().methodId && spec?.update !== "runtime";
    if (structural) this.announceGPURebuild(`Apply ${spec?.label ?? key}: ${String(value)}`, paneId);
    this.session(paneId).method.getState().setParam(methodId, key, value);
    if (structural) this.reset(undefined, undefined, paneId);
  }

  resetMethodParam(methodId: string, key: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === this.session(paneId).method.getState().methodId && spec?.update !== "runtime";
    if (structural) this.announceGPURebuild(`Restore ${spec?.label ?? key} default`, paneId);
    this.session(paneId).method.getState().resetParam(methodId, key);
    if (structural) this.reset(undefined, undefined, paneId);
  }

  // ---- authoring: draft/commit, history ----------------------------------

  /** The live document, cloned so history never aliases store state. */
  private documentSnapshot(label: string, paneId: PaneId = PRIMARY_PANE_ID): EditorHistorySnapshot {
    const sceneStore = this.session(paneId).scene.getState();
    return { label, scene: cloneScene(sceneStore.scene), presetId: sceneStore.presetId };
  }

  /**
   * Push the pre-edit document onto the undo stack. `coalesceKey` groups a
   * continuing gesture — a held slider, a repeated nudge — into one entry.
   * An open `beginEdit` gesture already holds the pre-gesture snapshot, so
   * document writes made during one record nothing until it commits.
   */
  private recordHistory(label: string, coalesceKey?: string, paneId: PaneId = PRIMARY_PANE_ID) {
    if (this.runtime(paneId).pendingEdit) return;
    this.session(paneId).history.getState().record(this.documentSnapshot(label, paneId), { coalesceKey });
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
  beginEdit(label: string, paneId: PaneId = PRIMARY_PANE_ID) {
    this.runtime(paneId).pendingEdit = { label, snapshot: this.documentSnapshot(label, paneId) };
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
  commitEdit(patch?: Partial<SceneDescription>, options: { reseed?: boolean } = {}, paneId: PaneId = PRIMARY_PANE_ID) {
    const paneRuntime = this.runtime(paneId);
    const pending = paneRuntime.pendingEdit;
    paneRuntime.pendingEdit = undefined;
    const sceneStore = this.session(paneId).scene.getState();
    if (patch) sceneStore.patchScene(patch);
    if (!pending) return false;
    const committed = this.session(paneId).scene.getState().scene;
    const changed = canonicalScene(committed) !== canonicalScene(pending.snapshot.scene);
    if (!changed) return false;
    this.session(paneId).history.getState().record(pending.snapshot);
    if (options.reseed && sceneEditRequiresReset(pending.snapshot.scene, committed, this.session(paneId).method.getState().methodId)) {
      // reset() re-selects the first body; an editor gesture must keep the
      // thing the user is still holding selected.
      const selection = this.session(paneId).ui.getState().selection;
      this.reset(undefined, undefined, paneId);
      this.session(paneId).ui.getState().select(selection);
    } else this.adoptRigidBodies(committed, paneId);
    this.session(paneId).runtime.getState().setNotice(pending.label);
    return true;
  }

  /**
   * Reconcile the running roster with an edit that did not take a reset.
   *
   * `reset` is the only other place a pane's roster is built, so without this a
   * body added while the clock runs would be in the document, on screen, and
   * absent from `advanceTo` — the solver would never hear about it.
   */
  private adoptRigidBodies(scene: SceneDescription, paneId: PaneId = PRIMARY_PANE_ID) {
    const paneRuntime = this.runtime(paneId);
    const before = paneRuntime.bodies;
    const after = adoptRigidBodyRoster(before, scene.rigidBodies);
    paneRuntime.bodies = after;
    if (before.length === after.length && before.every((body, index) => body === after[index])) return;
    this.publishBodies(rigidDiagnostics(after, scene.fluid.gravity_m_s2), paneId);
  }

  /**
   * Adopt a document some other authority wrote straight into a pane's store.
   *
   * `commitEdit`'s tail without the gesture. The compare mirror is the one such
   * authority: pane B's document is pane A's plus a diff, so an edit in A is
   * mirrored by *writing B's scene store* — which moves B's drawing and moves
   * neither B's running rigid roster nor B's solver. A sphere placed in A then
   * exists in B's document, on B's screen, and nowhere in B's `advanceTo`.
   *
   * No history entry and no notice, because the edit happened in the other
   * pane: the undo stack belongs to the pane the reader actually edited, and
   * one gesture must not read back as two.
   *
   * The reset is `sceneEditRequiresReset`'s decision, exactly as it is for the
   * pane that was edited — so a uniform-tier field the live solver adopts
   * through `applySceneUniforms` costs this pane no more than it cost that one,
   * and a structural or seed-tier change re-seeds both.
   */
  adoptSceneEdit(previous: SceneDescription, paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const committed = this.session(paneId).scene.getState().scene;
    // Documents are immutable at this boundary, so identity is the whole test —
    // and much cheaper than `canonicalScene` over a sculpted terrain grid.
    if (committed === previous) return false;
    if (sceneEditRequiresReset(previous, committed, this.session(paneId).method.getState().methodId)) {
      // Same care `commitEdit` takes: reset() re-selects the first body, and a
      // mirrored edit must not steal the selection out of the receiving pane.
      const selection = this.session(paneId).ui.getState().selection;
      // The transport is the host's, not the document's. `reset` pauses the pane
      // it re-seeds, which is right for the pane the reader edited and wrong for
      // the pane merely keeping up with it: under a lockstep clock a pane that
      // stops submitting stalls *both*, so a mirror that paused the receiver
      // would freeze the comparison it exists to keep running.
      const runState = this.session(paneId).runtime.getState().runState;
      this.reset(undefined, undefined, paneId);
      this.session(paneId).ui.getState().select(selection);
      this.session(paneId).runtime.getState().setRunState(runState);
    } else this.adoptRigidBodies(committed, paneId);
    return true;
  }

  /** Abandon a gesture without recording it; runtime preview state is kept. */
  cancelEdit(paneId: PaneId = PRIMARY_PANE_ID) { this.runtime(paneId).pendingEdit = undefined; }

  private applyHistorySnapshot(entry: EditorHistorySnapshot, verb: string, paneId: PaneId = PRIMARY_PANE_ID) {
    this.reset(cloneScene(entry.scene), entry.presetId, paneId);
    this.session(paneId).runtime.getState().setNotice(entry.label ? `${verb} ${entry.label}` : `${verb} last edit`);
  }

  undo(paneId: PaneId = PRIMARY_PANE_ID): boolean {
    this.runtime(paneId).pendingEdit = undefined;
    const entry = this.session(paneId).history.getState().undo(this.documentSnapshot("", paneId));
    if (!entry) { this.session(paneId).runtime.getState().setNotice("Nothing to undo"); return false; }
    this.applyHistorySnapshot(entry, "Undid", paneId);
    return true;
  }

  redo(paneId: PaneId = PRIMARY_PANE_ID): boolean {
    this.runtime(paneId).pendingEdit = undefined;
    const entry = this.session(paneId).history.getState().redo(this.documentSnapshot("", paneId));
    if (!entry) { this.session(paneId).runtime.getState().setNotice("Nothing to redo"); return false; }
    this.applyHistorySnapshot(entry, "Redid", paneId);
    return true;
  }

  // ---- rigid-body roster ------------------------------------------------

  addBody(shape: RigidShape, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { this.session(paneId).runtime.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return; }
    this.recordHistory(`add ${shape}`, undefined, paneId);
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${shape}-${bodyIndex}`)) bodyIndex += 1;
    const description = createBodyDescription(shape, bodyIndex, scene.container.height_m);
    sceneStore.patchScene({ rigidBodies: [...scene.rigidBodies, description] });
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = [...paneRuntime.bodies, initializeRigidBody(description)];
    this.publishBodies(undefined, paneId);
    this.session(paneId).ui.getState().selectBody(description.id);
    this.session(paneId).runtime.getState().setNotice(`${description.name} added above the container`);
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
  addBodyAt(shape: RigidShape, position: RigidBodyState["position_m"], options: { autoRun?: boolean; dimensions_m?: Vec3 } = {}, paneId: PaneId = PRIMARY_PANE_ID): RigidBodyDescription | undefined {
    const sceneStore = this.session(paneId).scene.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { this.session(paneId).runtime.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return undefined; }
    this.recordHistory(`place ${shape}`, undefined, paneId);
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${shape}-${bodyIndex}`)) bodyIndex += 1;
    // The size the placement row is showing, when it is showing one. Applied
    // before the radius is taken, or a body sized up in the strip would be
    // rested against the footprint of the one the table ships.
    const template = options.dimensions_m === undefined
      ? createBodyDescription(shape, bodyIndex, scene.container.height_m)
      : { ...createBodyDescription(shape, bodyIndex, scene.container.height_m), dimensions_m: options.dimensions_m };
    const radius = boundingRadius(template);
    const description = { ...template, position_m: {
      x: Math.min(scene.container.width_m / 2 - radius, Math.max(-scene.container.width_m / 2 + radius, position.x)),
      y: Math.min(scene.container.height_m + 0.8, Math.max(radius, position.y)),
      z: Math.min(scene.container.depth_m / 2 - radius, Math.max(-scene.container.depth_m / 2 + radius, position.z))
    }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } };
    sceneStore.patchScene({ rigidBodies: [...scene.rigidBodies, description] });
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = [...paneRuntime.bodies, initializeRigidBody(description)];
    this.publishBodies(undefined, paneId);
    this.session(paneId).ui.getState().selectBody(description.id);
    // Starting the clock is a transport act, and the transport is the host's:
    // a body dropped into pane B has to start the pair, not pane B alone.
    if (options.autoRun !== false) this.setRunState("running");
    this.session(paneId).runtime.getState().setNotice(`${description.name} ${options.autoRun === false ? "placed" : "dropped into the scene"}`);
    return description;
  }

  removeBody(bodyId: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    if (!sceneStore.scene.rigidBodies.some((body) => body.id === bodyId)) return;
    this.recordHistory("remove body", undefined, paneId);
    const descriptions = sceneStore.scene.rigidBodies.filter((body) => body.id !== bodyId);
    sceneStore.patchScene({ rigidBodies: descriptions });
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = paneRuntime.bodies.filter((body) => body.description.id !== bodyId);
    this.publishBodies(undefined, paneId);
    // Deleting is not a way of selecting something else: land on empty rather
    // than jumping the gizmo onto whichever body happens to be first.
    this.session(paneId).ui.getState().select(undefined);
    // One clock, so an edit that stops it stops it for every pane.
    this.setRunState("paused");
    this.session(paneId).runtime.getState().setNotice("Body removed");
  }

  /**
   * Held numeric controls fire per input event, so the default coalesce key is
   * the edited field: one undo entry per property the user swept, not one per
   * pixel of slider travel.
   */
  updateBody(bodyId: string, patch: Partial<RigidBodyDescription>, coalesceKey = `body:${bodyId}:${Object.keys(patch).sort().join(",")}`, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    this.recordHistory("body edit", coalesceKey, paneId);
    const descriptions = sceneStore.scene.rigidBodies.map((body) => body.id === bodyId ? { ...body, ...patch } : body);
    sceneStore.patchScene({ rigidBodies: descriptions });
    const description = descriptions.find((item) => item.id === bodyId);
    const paneRuntime = this.runtime(paneId);
    if (description) paneRuntime.bodies = paneRuntime.bodies.map((body) => {
      if (body.description.id !== bodyId) return body;
      const updated = initializeRigidBody(description);
      updated.position_m = { ...(patch.position_m ?? body.position_m) };
      return updated;
    });
    this.publishBodies(undefined, paneId);
    this.session(paneId).runtime.getState().setNotice("Body parameters updated");
  }

  resetBody(bodyId: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = paneRuntime.bodies.map((body) => body.description.id === bodyId ? initializeRigidBody(body.description) : body);
    this.publishBodies(undefined, paneId);
    // One clock, so an edit that stops it stops it for every pane.
    this.setRunState("paused");
  }

  dropBody(bodyId: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const scene = this.session(paneId).scene.getState().scene;
    const paneRuntime = this.runtime(paneId);
    paneRuntime.bodies = paneRuntime.bodies.map((body) => {
      if (body.description.id !== bodyId) return body;
      return initializeRigidBody({ ...body.description, position_m: { x: body.position_m.x, y: scene.container.height_m + boundingRadius(body.description) + 0.08, z: body.position_m.z }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
    });
    this.publishBodies(undefined, paneId);
    // One clock, so a gesture that starts it starts it for every pane.
    this.setRunState("running");
    this.session(paneId).runtime.getState().setNotice("Body released with buoyancy, drag, torque, and fluid reaction enabled");
  }

  /** Kinematic constraint shared by the physics drag and the editor gizmo. */
  private applyBodyManipulation(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"], paneId: PaneId = PRIMARY_PANE_ID) {
    const paneRuntime = this.runtime(paneId);
    if (phase === "end") paneRuntime.kinematicDrag = null;
    else paneRuntime.kinematicDrag = { bodyId, position: { ...position }, velocity: { ...velocity } };
    const body = paneRuntime.bodies.find((candidate) => candidate.description.id === bodyId);
    if (!body) return;
    // Held is the whole of "no gravity": the pose below is a command, and a
    // body that also integrated would sink out of the hand between the frames
    // where the pointer happens not to move.
    body.held = phase !== "end";
    body.position_m = { ...position };
    if (orientation) body.orientation = { ...orientation };
    body.linearVelocity_m_s = phase === "end" ? { x: 0, y: 0, z: 0 } : { ...velocity };
    body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 };
    this.publishBodies(undefined, paneId);
  }

  dragBody(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"], paneId: PaneId = PRIMARY_PANE_ID) {
    this.applyBodyManipulation(bodyId, position, velocity, phase, orientation, paneId);
    const runtime = this.session(paneId).runtime.getState();
    // The carry starts the host clock, not this pane's: see `addBodyAt`.
    if (phase === "start") { this.setRunState("running"); runtime.setNotice("Kinematic drag active · GPU immersed boundary coupling"); }
    if (phase === "end") runtime.setNotice("Body released to buoyancy, drag, and collision response");
  }

  /**
   * Gizmo manipulation. Same kinematic constraint as `dragBody`, but authoring
   * never starts the clock: placing geometry is an edit, not a throw. The pose
   * lives in runtime state only until `commitEdit` writes the document.
   */
  manipulateBody(bodyId: string, position: RigidBodyState["position_m"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"], paneId: PaneId = PRIMARY_PANE_ID) {
    this.applyBodyManipulation(bodyId, position, { x: 0, y: 0, z: 0 }, phase, orientation, paneId);
  }

  // ---- renderer callbacks ------------------------------------------------

  /**
   * Publish transport time only after the corresponding GPU work completes.
   * A pane reports its own completions; the published time is the slowest
   * pane's, so the product never claims a state one of its panes is not in.
   */
  gpuAdvanceCompleted(time_s: number, paneId: PaneId = PRIMARY_PANE_ID) {
    if (!this.clock.completeAdvance(time_s, paneId)) return;
    this.publishSimulationTime();
  }

  /** Drop host-side debt when paused, retaining only work already admitted to a pane's GPU queue. */
  gpuSchedulingPaused(submittedTime_s?: number, paneId: PaneId = PRIMARY_PANE_ID) {
    this.clock.schedulingPaused(submittedTime_s, paneId);
  }

  recordFrame(metrics: RendererFrameMetrics, resolution: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const diagnostics = this.session(paneId).diagnostics.getState();
    const methodId = metrics.methodId ?? this.session(paneId).method.getState().methodId;
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
    // The cadence is this pane's own: two panes reporting into one gate would
    // each suppress the other's report and neither would publish every 100 ms.
    const paneRuntime = this.runtime(paneId);
    const reportDue = context !== paneRuntime.lastPerformanceReportContext
      || capturedAt_ms - paneRuntime.lastPerformanceReportAt_ms >= 100;
    if (!reportDue) return;
    paneRuntime.lastPerformanceReportAt_ms = capturedAt_ms;
    paneRuntime.lastPerformanceReportContext = context;
    const physicsTrace = diagnostics.gpuInfo?.physicsTrace;
    const rendererCPU = metrics.cpu && metrics.cpu.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(metrics.cpu, "cpu", "main-thread") ? metrics.cpu : undefined;
    // The tick trace is the *host's* — one clock, measured once — so only the
    // pane that clock runs for may claim it. Letting a second pane's frame
    // consume it would leave pane A's report with no controller CPU at all.
    const hostTick = paneId === PRIMARY_PANE_ID ? this.pendingCpuTickTrace : undefined;
    const controllerCPU = hostTick
      && hostTick.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(hostTick, "cpu", "main-thread")
      ? hostTick
      : undefined;
    if (paneId === PRIMARY_PANE_ID) this.pendingCpuTickTrace = undefined;
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
    const presentation = metrics.presentation && metrics.presentation.capturedAt_ms >= instrumentation.enabledAt_ms
      && performanceTraceMatchesLane(metrics.presentation, "gpu", "presentation") ? metrics.presentation : undefined;
    const report = {
      methodId,
      context,
      capturedAt_ms,
      cpu,
      physics,
      presentation,
      presentationStages: metrics.presentationStages
        && metrics.presentationStages.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(metrics.presentationStages, "gpu", "presentation")
        ? metrics.presentationStages : undefined,
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
  beginDraft(subject: SceneDraftSubject, label: string, paneId: PaneId = PRIMARY_PANE_ID) {
    this.session(paneId).sceneDraft.getState().beginDraft(subject, label);
    this.beginEdit(label, paneId);
  }

  /**
   * Land the open draft on the document — one write, one re-seed, one undo
   * entry — or drop it if the gesture was interrupted.
   *
   * `announceRebuild` is for the gestures that move the lattice (the tank), so
   * the status line names the edit that is costing the pause instead of
   * reporting an anonymous rebuild.
   */
  commitDraft(options: { announceRebuild?: string } = {}, paneId: PaneId = PRIMARY_PANE_ID) {
    const draftStore = this.session(paneId).sceneDraft.getState();
    const draft = draftStore.draft;
    draftStore.clearDraft();
    if (!draft) { this.cancelEdit(paneId); return false; }
    if (options.announceRebuild) this.announceGPURebuild(options.announceRebuild, paneId);
    this.session(paneId).scene.getState().patchScene(draft.patch);
    return this.commitEdit(undefined, { reseed: true }, paneId);
  }

  /** Abandon the open gesture, leaving the committed scene untouched. */
  cancelDraft(paneId: PaneId = PRIMARY_PANE_ID) {
    this.session(paneId).sceneDraft.getState().clearDraft();
    this.cancelEdit(paneId);
  }

  // ---- scene scale ---------------------------------------------------------

  /**
   * Scale the world or the detail by a factor of two.
   *
   * A world scale keeps the lattice dimensions and warm re-seeds its updated
   * metre mapping. A detail scale moves the lattice and rebuilds. Compiled
   * pipelines remain cached.
   */
  scaleScene(axis: SceneScaleAxis, factor: SceneScaleFactor, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const next = scaleSceneBy(sceneStore.scene, axis, factor);
    if (!next) {
      const blocked = sceneScaleOption(sceneScaleSummary(sceneStore.scene), axis, factor).blocked;
      this.session(paneId).runtime.getState().setNotice(`Scale refused · ${blocked ?? "unavailable at this size"}`);
      return false;
    }
    const label = `${factor > 1 ? "Doubled" : "Halved"} the ${axis === "world" ? "world size" : "detail"}`;
    if (axis === "detail") this.announceGPURebuild(label, paneId);
    this.beginEdit(label, paneId);
    sceneStore.setScene(next, sceneStore.presetId);
    this.commitEdit(undefined, { reseed: true }, paneId);
    const [nx, ny, nz] = sceneLatticeDimensions(next);
    this.session(paneId).runtime.getState().setNotice(`${label} · ${nx}×${ny}×${nz} cells at ${next.voxelDomain.finestCellSize_m.toFixed(4)} m`);
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
  shapeFluidBody(box: FluidBodyBox, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    sceneStore.patchScene(fluidBodyBoxPatch(sceneStore.scene, box));
    this.commitEdit(undefined, { reseed: true }, paneId);
  }

  /**
   * Resize the tank to a dragged box. Structural — the lattice moves — so this
   * is announced and, like `shapeFluidBody`, only ever called on release.
   */
  resizeTank(box: FluidBodyBox, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const extents = {
      width_m: box.max.x - box.min.x,
      height_m: box.max.y - box.min.y,
      depth_m: box.max.z - box.min.z,
    };
    if (tankResizeIsStructural(sceneStore.scene, extents)) this.announceGPURebuild("Resize the tank", paneId);
    sceneStore.patchScene(tankResizePatch(sceneStore.scene, extents));
    this.commitEdit(undefined, { reseed: true }, paneId);
  }

  /**
   * Grow or shrink the water body about its own centre, without a drag.
   * `volumeFactor` is a volume, so ×2 is twice the water.
   */
  scaleFluidBody(volumeFactor: number, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const box = fluidBodyBox(sceneStore.scene);
    if (!box) { this.session(paneId).runtime.getState().setNotice("No water body to resize"); return false; }
    this.beginEdit(volumeFactor > 1 ? "Grew the water body" : "Shrank the water body", paneId);
    this.shapeFluidBody(scaleFluidBodyVolume(box, volumeFactor, sceneStore.scene), paneId);
    return true;
  }

  // ---- persistence -------------------------------------------------------

  // ---- render-only scenery -------------------------------------------------

  /**
   * Scenery extends the sparse render domain but never enters the solve, so it
   * is the one authoring action that does not need a re-seed.
   */
  addScenery(kind: SceneryPropKind, point_m: RigidBodyState["position_m"], normal: RigidBodyState["position_m"], paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const node = createSceneryNodeAt(sceneStore.scene, kind, point_m, normal);
    this.recordHistory(`place ${kind}`, undefined, paneId);
    // A whole scene rather than a patch: the node joins the same list the
    // preset's own scenery lives in, and a merge patch cannot append.
    sceneStore.setScene(addSceneryNode(sceneStore.scene, node), sceneStore.presetId);
    this.session(paneId).ui.getState().select({ kind: "scenery", id: scenerySelectionId(node.id) });
    this.session(paneId).runtime.getState().setNotice(`${node.id} placed`);
    return node;
  }

  /**
   * Remove whatever is selected, from the scene the entity handed back.
   *
   * A whole scene rather than a patch because removing the last prop drops the
   * `props` key, and a merge patch cannot express an absence — so the entity
   * composes the document it wants and this only lands it.
   */
  removeEntity(label: string, next: SceneDescription, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    this.beginEdit(label, paneId);
    sceneStore.setScene(next, sceneStore.presetId);
    this.session(paneId).ui.getState().select(undefined);
    this.commitEdit(undefined, { reseed: true }, paneId);
    this.session(paneId).runtime.getState().setNotice(label);
  }

  removeScenery(id: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    if (!findSceneryNode(sceneStore.scene, id)) return;
    this.recordHistory("remove scenery", undefined, paneId);
    sceneStore.setScene(withoutSceneryNode(sceneStore.scene, id), sceneStore.presetId);
    this.session(paneId).ui.getState().select(undefined);
    this.session(paneId).runtime.getState().setNotice(`${id} removed`);
  }

  // ---- named scene library ------------------------------------------------

  /** Save the live document under a name, replacing an entry of the same name. */
  saveNamedScene(name: string, paneId: PaneId = PRIMARY_PANE_ID) {
    const sceneStore = this.session(paneId).scene.getState();
    const { methodId, quality, overrides } = this.session(paneId).method.getState();
    const { entries, entry } = saveSceneToLibrary(
      browserSceneLibraryStorage(), name, sceneStore.scene, sceneStore.presetId, {
        savedAt_ms: Date.now(),
        methodProfile: { methodId, quality, overrides: { ...(overrides[methodId] ?? {}) } },
      },
    );
    this.session(paneId).runtime.getState().setNotice(`Saved “${entry.name}”`);
    return entries;
  }

  /**
   * Load a library entry as a scene edit, so it is undoable like any other.
   *
   * Routed through the same card as the library grid, which is how a saved
   * validation scene regained the solver profile it was authored under — this
   * path used to drop it and run whatever method happened to be selected.
   */
  loadNamedScene(entry: SceneLibraryEntry, paneId: PaneId = PRIMARY_PANE_ID): boolean {
    return this.openSceneCard(savedSceneCard(entry), paneId);
  }

  loadLocalScene(paneId: PaneId = PRIMARY_PANE_ID): boolean {
    const stored = localStorage.getItem("fluid-lab.scene.v1");
    if (!stored) return false;
    try { const loaded = parseScene(stored); this.reset(loaded, undefined, paneId); this.session(paneId).runtime.getState().setNotice("Loaded the last local scene"); }
    catch { this.session(paneId).runtime.getState().setNotice("Stored scene failed validation", "warn"); }
    return true;
  }

  importScene(name: string, contents: string, paneId: PaneId = PRIMARY_PANE_ID) {
    try { const loaded = parseScene(contents); this.recordHistory(`import ${name}`, undefined, paneId); this.reset(loaded, undefined, paneId); this.session(paneId).runtime.getState().setNotice(`Loaded ${name}`); }
    catch (error) { this.session(paneId).runtime.getState().setNotice(error instanceof Error ? error.message : "Scene import failed", "warn"); }
  }

  applyAndResetFluid(paneId: PaneId = PRIMARY_PANE_ID) {
    this.reset(cloneScene(this.session(paneId).scene.getState().scene), undefined, paneId);
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
  clearOverrides(keys: readonly string[], paneId: PaneId = PRIMARY_PANE_ID): { readonly reload: readonly string[] } {
    const sceneStore = this.session(paneId).scene.getState();
    const plan = sceneOverrideClearPlan(keys, { scene: sceneStore.scene, presetId: sceneStore.presetId });
    const label = keys.length === 1 ? `clear ${keys[0]}` : `clear ${keys.length} overrides`;

    if (plan.scene) {
      // Through the editor gesture rather than `reset`, so that clearing an
      // override the running solver can already adopt — a density, the hose —
      // does not throw away the simulation being watched to apply it.
      this.beginEdit(label, paneId);
      sceneStore.setScene(plan.scene);
      this.commitEdit(undefined, { reseed: true }, paneId);
    }
    const ui = this.session(paneId).ui.getState();
    if (Object.keys(plan.ui).length > 0) this.session(paneId).ui.setState(plan.ui);
    if (Object.keys(plan.svoRenderTuning).length > 0) {
      ui.setSvoRenderTuning({ ...ui.svoRenderTuning, ...plan.svoRenderTuning });
    }
    // Parameters before the method, so a restored value lands on the method it
    // belongs to rather than triggering a rebuild of the one being left.
    for (const param of plan.methodParams) {
      if (param.value === undefined) this.resetMethodParam(param.methodId, param.key, paneId);
      else this.setMethodParam(param.methodId, param.key, param.value, paneId);
    }
    if (plan.quality) this.setQuality(plan.quality, paneId);
    if (plan.methodId) this.setMethod(plan.methodId, paneId);

    if (!plan.scene && plan.reload.length === keys.length) return { reload: plan.reload };
    this.session(paneId).runtime.getState().setNotice(keys.length === 1
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
  retainedSimulation.adoptRetainedRuntime();
}
export const simulation = retainedSimulation ?? new SimulationController();
if (simulationWindow) simulationWindow.__fluidLabSimulationController = simulation;
