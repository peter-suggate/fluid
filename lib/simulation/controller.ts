import { BUILD_ID, canonicalScene, cloneScene, parseScene, type SceneDescription } from "../model";
import { EulerianFluidSolver } from "../eulerian-solver";
import { advanceRigidBodies, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBodies, initializeRigidBody, rigidDiagnostics, type RigidBodyState, type RigidStepDiagnostics } from "../rigid-body";
import type { RigidBodyDescription } from "../model";
import { applyFluidReactions, computeFluidLoads, type CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { RigidShape } from "../model";
import type { RendererFrameMetrics, SimulationBackend } from "../webgpu-renderer";
import { getMethod } from "../methods";
import { cameraForPreset, getScenePreset } from "../scenes";
import { useSceneStore } from "../stores/scene-store";
import { useEditorHistoryStore, type EditorHistorySnapshot } from "../stores/history-store";
import { useMethodStore, resolvedMethodValues } from "../stores/method-store";
import type { MethodParamValue } from "../methods";
import { useRuntimeStore } from "../stores/runtime-store";
import { useDiagnosticsStore, emptyPerformanceReport } from "../stores/diagnostics-store";
import { usePerformanceInstrumentationStore } from "../stores/performance-instrumentation-store";
import { useUIStore } from "../stores/ui-store";
import { commitGPUCompletion, gpuCanAcceptNextStep } from "./gpu-clock";
import { safeBrowserGPUBringupEnabled } from "../gpu-startup";
import { planSceneRuntime } from "../scene-runtime";
import {
  combineMainThreadPerformanceTraces,
  CPUPerformanceTrace,
  performanceTraceMatchesLane,
  type PerformanceTrace,
} from "../performance-trace";

export type BodyDragPhase = "start" | "move" | "end";

const MAX_BODIES = 12;

/** O(1) host-clock collapse for GPU authority; no intermediate host state exists. */
export function collapseGPUFixedSteps(accumulator_s: number, dt_s: number) {
  if (!Number.isFinite(accumulator_s) || accumulator_s < 0 || !Number.isFinite(dt_s) || dt_s <= 0) {
    return { steps: 0, remainder_s: Math.max(0, Number.isFinite(accumulator_s) ? accumulator_s : 0) };
  }
  const steps = Math.floor((accumulator_s + 1e-12) / dt_s);
  return { steps, remainder_s: Math.max(0, accumulator_s - steps * dt_s) };
}

/**
 * Owns the mutable runtime the render loop needs at 60 Hz — rigid-body
 * states, the CPU oracle solver, accumulators, pending GPU impulse loads —
 * and publishes serializable snapshots into the zustand stores. UI actions
 * that must rebuild runtime state (scene loads, body edits, resets) are
 * methods here so stores stay pure data.
 */
class SimulationController {
  /** Allocated only for the CPU reference method; GPU methods own no host fluid field. */
  private fluidSolver?: EulerianFluidSolver;
  private bodies: RigidBodyState[] = [];
  private accumulator = 0;
  private simulationTime = 0;
  private gpuCompletedTime = 0;
  private lastClock: number | null = null;
  private cpuOracleStep = 0;
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
    if (diagnostics.gpuStatus.state !== "ready") return false;
    return useMethodStore.getState().methodId !== "octree"
      || (diagnostics.gpuInfo?.initialSparseAuthorityReady === true
        && diagnostics.gpuInfo?.initialRasterSurfaceReady === true);
  }

  /**
   * The CPU solver doubles as the reference method and the background
   * validation oracle. When it is the active method its resolution comes
   * from the method's cell-size parameter (comparable to the GPU quality
   * presets); as a background oracle it stays at the scene's cheap nominal
   * resolution.
   */
  private buildFluidSolver(scene: SceneDescription): EulerianFluidSolver {
    if (!planSceneRuntime(scene).fluidSolver) throw new Error("This scene does not enable a CPU fluid solver");
    const methodState = useMethodStore.getState();
    if (getMethod(methodState.methodId).backend !== "cpu") return new EulerianFluidSolver(scene);
    const cellSize = Number(resolvedMethodValues(methodState).cellSize_m);
    const reference = Number.isFinite(cellSize) && cellSize > 0 ? { ...scene, nominalResolution: { length_m: cellSize } } : scene;
    // As the active method the CPU solve runs at its full requested
    // resolution; the 1 800-cell default cap is only for the cheap
    // background oracle that accompanies the GPU methods.
    return new EulerianFluidSolver(reference, 2_000_000);
  }

  constructor() {
    const scene = useSceneStore.getState().scene;
    if (this.backend === "cpu-reference") this.fluidSolver = this.buildFluidSolver(scene);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({
      fluidState: this.fluidSolver?.diagnostics ?? null,
      fluidRenderState: this.fluidSolver?.getRenderState() ?? null,
    });
    useUIStore.getState().selectBody(scene.rigidBodies[0]?.id);
  }

  get backend(): SimulationBackend {
    return getMethod(useMethodStore.getState().methodId).backend === "cpu" ? "cpu-reference" : "webgpu";
  }

  time(): number { return this.simulationTime; }
  currentBodies(): RigidBodyState[] { return this.bodies; }

  private cpuFluid(scene: SceneDescription): EulerianFluidSolver {
    return this.fluidSolver ??= this.buildFluidSolver(scene);
  }

  private publishBodies(diagnostics?: RigidStepDiagnostics) {
    const scene = useSceneStore.getState().scene;
    useDiagnosticsStore.getState().set({ bodies: cloneRigidBodies(this.bodies), rigidState: diagnostics ?? rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2) });
  }

  /** Prepare every fixed step owed by the wall clock. GPU admission is renderer-budgeted. */
  tick(now: number) {
    const method = useMethodStore.getState();
    const cpuTrace = usePerformanceInstrumentationStore.getState().enabled
      ? new CPUPerformanceTrace(
        ++this.cpuTickTraceSampleId,
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
    const backend = this.backend;
    if (backend === "webgpu" && !this.webgpuTransportReady()) {
      this.accumulator = 0;
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    this.accumulator += elapsed;
    const dt = scene.numerics.fixedDt_s;
    let steps = 0;
    let diagnostics: RigidStepDiagnostics | undefined;
    let fluidDiagnostics: ReturnType<EulerianFluidSolver["step"]> | undefined;
    let latestCoupling: CouplingDiagnostics | undefined;
    cpuTrace?.transition({
      id: backend === "cpu-reference" ? "velocity-advection" : "frame-control",
      label: backend === "cpu-reference" ? "CPU reference simulation + coupling" : "GPU target-clock control",
    });
    if (backend === "webgpu") {
      // The renderer admits GPU work against this target clock. Collapsing
      // accumulated fixed ticks is exact because no host fluid/body evolution
      // occurs at the intermediate ticks; solver.advanceTo owns subdivision.
      const collapsed = collapseGPUFixedSteps(this.accumulator, dt);
      steps = collapsed.steps;
      if (steps > 0) {
        this.applyDragConstraint();
        this.accumulator = collapsed.remainder_s;
        this.simulationTime += steps * dt;
      }
    } else {
      const fluid = this.cpuFluid(scene);
      while (this.accumulator + 1e-12 >= dt) {
        this.applyDragConstraint();
        const coupling = computeFluidLoads(scene, fluid, this.bodies);
        latestCoupling = applyFluidReactions(fluid, this.bodies, coupling.loads, dt);
        diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, coupling.loads);
        this.applyDragConstraint();
        this.cpuOracleStep += 1;
        fluidDiagnostics = fluid.step(dt);
        this.accumulator -= dt;
        this.simulationTime += dt;
        steps += 1;
      }
    }
    if (steps > 0) {
      if (backend === "cpu-reference") this.publishBodies(diagnostics);
      const patch: Parameters<ReturnType<typeof useDiagnosticsStore.getState>["set"]>[0] = {};
      if (fluidDiagnostics) {
        patch.fluidState = fluidDiagnostics;
        if (backend === "cpu-reference") patch.fluidRenderState = this.cpuFluid(scene).getRenderState();
      }
      if (latestCoupling) patch.couplingState = latestCoupling;
      useDiagnosticsStore.getState().set(patch);
      if (backend === "cpu-reference") runtime.setSimulationTime(this.simulationTime);
    }
    if (this.rateWallClock === 0) { this.rateWallClock = now; this.rateSimTime = backend === "webgpu" ? this.gpuCompletedTime : this.simulationTime; }
    else if (now - this.rateWallClock > 500) {
      const committedTime = backend === "webgpu" ? this.gpuCompletedTime : this.simulationTime;
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
    if (body) { body.position_m = { ...drag.position }; body.linearVelocity_m_s = { ...drag.velocity }; body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 }; }
  }

  singleStep() {
    const runtime = useRuntimeStore.getState();
    runtime.setRunState("paused");
    const scene = useSceneStore.getState().scene;
    if (!planSceneRuntime(scene).fluidSolver) return;
    const dt = scene.numerics.fixedDt_s;
    const backend = this.backend;
    if (backend === "webgpu" && !this.webgpuTransportReady()) return;
    if (backend === "webgpu" && !gpuCanAcceptNextStep(this.simulationTime, this.gpuCompletedTime)) return;
    if (this.safeBrowserBringup() && this.safeBrowserStepConsumed) return;
    if (this.safeBrowserBringup()) this.safeBrowserStepConsumed = true;
    let couplingDiagnostics: CouplingDiagnostics | undefined;
    let diagnostics: RigidStepDiagnostics | undefined;
    let fluidDiagnostics: ReturnType<EulerianFluidSolver["step"]> | undefined;
    if (backend === "cpu-reference") {
      const fluid = this.cpuFluid(scene);
      const coupling = computeFluidLoads(scene, fluid, this.bodies);
      couplingDiagnostics = applyFluidReactions(fluid, this.bodies, coupling.loads, dt);
      diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, coupling.loads);
      fluidDiagnostics = fluid.step(dt);
    }
    this.simulationTime += dt;
    if (backend === "cpu-reference") runtime.setSimulationTime(this.simulationTime);
    if (backend === "cpu-reference") {
      this.publishBodies(diagnostics);
      useDiagnosticsStore.getState().set({ fluidState: fluidDiagnostics, fluidRenderState: this.cpuFluid(scene).getRenderState(), couplingState: couplingDiagnostics });
    }
  }

  reset(source?: SceneDescription, presetId?: string) {
    const sceneStore = useSceneStore.getState();
    const scene = source ?? sceneStore.scene;
    if (source) sceneStore.setScene(source, presetId);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    const runtimePlan = planSceneRuntime(scene, { methodId: useMethodStore.getState().methodId });
    this.fluidSolver = this.backend === "cpu-reference" && runtimePlan.fluidSolver ? this.buildFluidSolver(scene) : undefined;
    this.simulationTime = 0; this.gpuCompletedTime = 0; this.accumulator = 0; this.lastClock = null;
    this.rateWallClock = 0; this.rateSimTime = 0;
    this.cpuOracleStep = 0;
    this.kinematicDrag = null;
    if (!this.safeBrowserBringup()) this.safeBrowserStepConsumed = false;
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({ fluidState: this.fluidSolver?.diagnostics ?? null, fluidRenderState: this.fluidSolver?.getRenderState() ?? null, gpuInfo: null, waterSurfacePresentation: null, couplingState: { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 }, samples: [], performanceReport: emptyPerformanceReport, performanceReports: [] });
    const runtime = useRuntimeStore.getState();
    // Publish the new t=0 identity before pausing. Renderer subscribers use
    // this synchronous epoch edge to reject completions from the old queue.
    runtime.resetSimulationTime();
    runtime.setSimRate(null);
    runtime.setRunState("paused");
    useUIStore.getState().selectBody(scene.rigidBodies[0]?.id);
    runtime.setNotice(!runtimePlan.fluidSolver
      ? "Static renderer scene reset · fluid solver disabled"
      : `${scene.fluid.inflow ? "Inflow scene" : scene.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
  }

  loadPreset(presetId: string) {
    const preset = getScenePreset(presetId);
    this.recordHistory(`load ${preset.name}`);
    if (preset.methodProfile) useMethodStore.getState().applyProfile(preset.methodProfile);
    const scene = preset.create();
    this.reset(scene, preset.id);
    useUIStore.getState().setCamera(cameraForPreset(preset));
    const runtimePlan = planSceneRuntime(scene, { methodId: useMethodStore.getState().methodId });
    useRuntimeStore.getState().setNotice(!runtimePlan.fluidSolver
      ? `${preset.name} loaded · fluid solver disabled`
      : `${preset.name} loaded · dt ${scene.numerics.fixedDt_s.toFixed(4)} s`);
    useRuntimeStore.getState().setRunState(runtimePlan.fluidSolver ? "running" : "paused");
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
    if (this.backend !== "webgpu") return;
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
    } });
  }

  setMethodParam(methodId: string, key: string, value: MethodParamValue) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === useMethodStore.getState().methodId && (method.backend === "cpu" || spec?.update !== "runtime");
    if (structural && method.backend === "webgpu") this.announceGPURebuild(`Apply ${spec?.label ?? key}: ${String(value)}`);
    useMethodStore.getState().setParam(methodId, key, value);
    if (structural) this.reset();
  }

  resetMethodParam(methodId: string, key: string) {
    const method = getMethod(methodId), spec = method.params.find((candidate) => candidate.key === key);
    const structural = methodId === useMethodStore.getState().methodId && (method.backend === "cpu" || spec?.update !== "runtime");
    if (structural && method.backend === "webgpu") this.announceGPURebuild(`Restore ${spec?.label ?? key} default`);
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

  /** Close the gesture; a patch that changed nothing records no undo entry. */
  commitEdit(patch?: Partial<SceneDescription>) {
    const pending = this.pendingEdit;
    this.pendingEdit = undefined;
    const sceneStore = useSceneStore.getState();
    if (patch) sceneStore.patchScene(patch);
    if (!pending) return false;
    const changed = canonicalScene(useSceneStore.getState().scene) !== canonicalScene(pending.snapshot.scene);
    if (!changed) return false;
    useEditorHistoryStore.getState().record(pending.snapshot);
    useRuntimeStore.getState().setNotice(pending.label);
    return true;
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
  addBodyAt(shape: RigidShape, position: RigidBodyState["position_m"], options: { autoRun?: boolean } = {}) {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { useRuntimeStore.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return; }
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
  }

  removeBody(bodyId: string) {
    const sceneStore = useSceneStore.getState();
    if (!sceneStore.scene.rigidBodies.some((body) => body.id === bodyId)) return;
    this.recordHistory("remove body");
    const descriptions = sceneStore.scene.rigidBodies.filter((body) => body.id !== bodyId);
    sceneStore.patchScene({ rigidBodies: descriptions });
    this.bodies = this.bodies.filter((body) => body.description.id !== bodyId);
    this.publishBodies();
    useUIStore.getState().selectBody(descriptions[0]?.id);
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
    if (this.backend !== "webgpu") return;
    const completed = commitGPUCompletion(this.simulationTime, this.gpuCompletedTime, time_s);
    if (completed === this.gpuCompletedTime) return;
    this.gpuCompletedTime = completed;
    useRuntimeStore.getState().setSimulationTime(completed);
  }

  /** Drop host-side debt when paused, retaining only work already admitted to the GPU queue. */
  gpuSchedulingPaused(submittedTime_s?: number) {
    if (this.backend !== "webgpu") return;
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
    const cpu = combineMainThreadPerformanceTraces(
      [controllerCPU, rendererCPU].filter((trace): trace is PerformanceTrace => trace !== undefined),
      context,
    );
    const report = {
      methodId,
      context,
      capturedAt_ms,
      cpu,
      physics: physicsTrace && physicsTrace.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(physicsTrace, "gpu", "physics") ? physicsTrace : undefined,
      presentation: metrics.presentation && metrics.presentation.capturedAt_ms >= instrumentation.enabledAt_ms
        && performanceTraceMatchesLane(metrics.presentation, "gpu", "presentation") ? metrics.presentation : undefined,
    };
    diagnostics.pushPerformanceReport(report);
  }

  // ---- persistence -------------------------------------------------------

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
}

export { BUILD_ID };
export const simulation = new SimulationController();
