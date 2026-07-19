import { BUILD_ID, cloneScene, parseScene, type SceneDescription } from "../model";
import { EulerianFluidSolver } from "../eulerian-solver";
import { advanceRigidBodies, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBodies, initializeRigidBody, rigidDiagnostics, type RigidBodyState, type RigidStepDiagnostics } from "../rigid-body";
import type { RigidBodyDescription } from "../model";
import { applyFluidReactions, computeFluidLoads, type CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { RigidShape } from "../model";
import type { RendererFrameMetrics, SimulationBackend } from "../webgpu-renderer";
import { getMethod } from "../methods";
import { cameraForPreset, getScenePreset } from "../scenes";
import { useSceneStore } from "../stores/scene-store";
import { useMethodStore, resolvedMethodValues } from "../stores/method-store";
import type { MethodParamValue } from "../methods";
import { useRuntimeStore } from "../stores/runtime-store";
import { useDiagnosticsStore, emptyPerformance, type PerformanceSnapshot } from "../stores/diagnostics-store";
import { useUIStore } from "../stores/ui-store";
import { commitGPUCompletion, gpuCanAcceptNextStep } from "./gpu-clock";

export type BodyDragPhase = "start" | "move" | "end";

const MAX_BODIES = 12;

/**
 * Owns the mutable runtime the render loop needs at 60 Hz — rigid-body
 * states, the CPU oracle solver, accumulators, pending GPU impulse loads —
 * and publishes serializable snapshots into the zustand stores. UI actions
 * that must rebuild runtime state (scene loads, body edits, resets) are
 * methods here so stores stay pure data.
 */
class SimulationController {
  private fluidSolver: EulerianFluidSolver;
  private bodies: RigidBodyState[] = [];
  private accumulator = 0;
  private simulationTime = 0;
  private gpuCompletedTime = 0;
  private lastClock: number | null = null;
  private cpuOracleStep = 0;
  private kinematicDrag: { bodyId: string; position: RigidBodyState["position_m"]; velocity: RigidBodyState["linearVelocity_m_s"] } | null = null;
  private sampleClock = 0;
  private cpuSimulationMs = 0;
  private performance: PerformanceSnapshot = emptyPerformance;
  private rateWallClock = 0;
  private rateSimTime = 0;

  /**
   * The CPU solver doubles as the reference method and the background
   * validation oracle. When it is the active method its resolution comes
   * from the method's cell-size parameter (comparable to the GPU quality
   * presets); as a background oracle it stays at the scene's cheap nominal
   * resolution.
   */
  private buildFluidSolver(scene: SceneDescription): EulerianFluidSolver {
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
    this.fluidSolver = this.buildFluidSolver(scene);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({ fluidState: this.fluidSolver.diagnostics, fluidRenderState: this.fluidSolver.getRenderState() });
    useUIStore.getState().selectBody(scene.rigidBodies[0]?.id);
  }

  get backend(): SimulationBackend {
    return getMethod(useMethodStore.getState().methodId).backend === "cpu" ? "cpu-reference" : "webgpu";
  }

  time(): number { return this.simulationTime; }
  currentBodies(): RigidBodyState[] { return this.bodies; }

  private publishBodies(diagnostics?: RigidStepDiagnostics) {
    const scene = useSceneStore.getState().scene;
    useDiagnosticsStore.getState().set({ bodies: cloneRigidBodies(this.bodies), rigidState: diagnostics ?? rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2) });
  }

  /** Prepare every fixed step owed by the wall clock. GPU admission is renderer-budgeted. */
  tick(now: number) {
    const tickStart = performance.now();
    if (this.lastClock === null) this.lastClock = now;
    const elapsed = Math.max(0, (now - this.lastClock) / 1000);
    this.lastClock = now;
    const runtime = useRuntimeStore.getState();
    if (runtime.runState !== "running") {
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    const scene = useSceneStore.getState().scene;
    const backend = this.backend;
    this.accumulator += elapsed;
    const dt = scene.numerics.fixedDt_s;
    let steps = 0;
    let diagnostics: RigidStepDiagnostics | undefined;
    let fluidDiagnostics: ReturnType<EulerianFluidSolver["step"]> | undefined;
    let latestCoupling: CouplingDiagnostics | undefined;
    while (this.accumulator + 1e-12 >= dt) {
      this.applyDragConstraint();
      if (backend === "webgpu") {
        // WebGPU owns the canonical body state and advances coupling,
        // integration, and contacts in the submitted command stream.
      } else {
        const coupling = computeFluidLoads(scene, this.fluidSolver, this.bodies);
        latestCoupling = applyFluidReactions(this.fluidSolver, this.bodies, coupling.loads, dt);
        diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, coupling.loads);
        this.applyDragConstraint();
        this.cpuOracleStep += 1;
        fluidDiagnostics = this.fluidSolver.step(dt);
      }
      this.accumulator -= dt;
      this.simulationTime += dt;
      steps += 1;
    }
    if (steps > 0) {
      if (backend === "cpu-reference") this.publishBodies(diagnostics);
      const patch: Parameters<ReturnType<typeof useDiagnosticsStore.getState>["set"]>[0] = {};
      if (fluidDiagnostics) {
        patch.fluidState = fluidDiagnostics;
        if (backend === "cpu-reference") patch.fluidRenderState = this.fluidSolver.getRenderState();
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
    this.cpuSimulationMs = performance.now() - tickStart;
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
    const dt = scene.numerics.fixedDt_s;
    const backend = this.backend;
    if (backend === "webgpu" && !gpuCanAcceptNextStep(this.simulationTime, this.gpuCompletedTime)) return;
    let couplingDiagnostics: CouplingDiagnostics | undefined;
    let diagnostics: RigidStepDiagnostics | undefined;
    let fluidDiagnostics: ReturnType<EulerianFluidSolver["step"]> | undefined;
    if (backend === "cpu-reference") {
      const coupling = computeFluidLoads(scene, this.fluidSolver, this.bodies);
      couplingDiagnostics = applyFluidReactions(this.fluidSolver, this.bodies, coupling.loads, dt);
      diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, coupling.loads);
      fluidDiagnostics = this.fluidSolver.step(dt);
    }
    this.simulationTime += dt;
    if (backend === "cpu-reference") runtime.setSimulationTime(this.simulationTime);
    if (backend === "cpu-reference") {
      this.publishBodies(diagnostics);
      useDiagnosticsStore.getState().set({ fluidState: fluidDiagnostics, fluidRenderState: this.fluidSolver.getRenderState(), couplingState: couplingDiagnostics });
    }
  }

  reset(source?: SceneDescription) {
    const sceneStore = useSceneStore.getState();
    const scene = source ?? sceneStore.scene;
    if (source) sceneStore.setScene(source);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    this.fluidSolver = this.buildFluidSolver(scene);
    this.simulationTime = 0; this.gpuCompletedTime = 0; this.accumulator = 0; this.lastClock = null;
    this.rateWallClock = 0; this.rateSimTime = 0;
    this.cpuOracleStep = 0; this.cpuSimulationMs = 0;
    this.kinematicDrag = null;
    this.performance = emptyPerformance;
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({ fluidState: this.fluidSolver.diagnostics, fluidRenderState: this.fluidSolver.getRenderState(), gpuInfo: null, couplingState: { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 }, samples: [], performanceSnapshot: emptyPerformance, performanceHistory: [] });
    const runtime = useRuntimeStore.getState();
    runtime.setSimulationTime(0);
    runtime.setSimRate(null);
    runtime.setRunState("paused");
    useUIStore.getState().selectBody(scene.rigidBodies[0]?.id);
    runtime.setNotice(`${scene.fluid.inflow ? "Inflow scene" : scene.fluid.initialCondition === "dam-break" ? "Dam-break" : "Tank fill"} reset at t = 0`);
  }

  loadPreset(presetId: string) {
    const preset = getScenePreset(presetId);
    const scene = preset.create();
    useSceneStore.getState().setScene(scene, preset.id);
    this.reset(scene);
    useUIStore.getState().setCamera(cameraForPreset(preset));
    useRuntimeStore.getState().setNotice(`${preset.name} loaded · dt ${scene.numerics.fixedDt_s.toFixed(4)} s`);
    useRuntimeStore.getState().setRunState("running");
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

  // ---- rigid-body roster ------------------------------------------------

  addBody(shape: RigidShape) {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { useRuntimeStore.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return; }
    let bodyIndex = 1;
    while (scene.rigidBodies.some((body) => body.id === `body-${shape}-${bodyIndex}`)) bodyIndex += 1;
    const description = createBodyDescription(shape, bodyIndex, scene.container.height_m);
    sceneStore.patchScene({ rigidBodies: [...scene.rigidBodies, description] });
    this.bodies = [...this.bodies, initializeRigidBody(description)];
    this.publishBodies();
    useUIStore.getState().selectBody(description.id);
    useRuntimeStore.getState().setNotice(`${description.name} added above the container`);
  }

  /** Spawn a body at a specific point, e.g. dropped from the viewport tray. */
  addBodyAt(shape: RigidShape, position: RigidBodyState["position_m"]) {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    if (scene.rigidBodies.length >= MAX_BODIES) { useRuntimeStore.getState().setNotice(`Renderer limit is ${MAX_BODIES} bodies in this verified increment`, "warn"); return; }
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
    useRuntimeStore.getState().setRunState("running");
    useRuntimeStore.getState().setNotice(`${description.name} dropped into the scene`);
  }

  removeBody(bodyId: string) {
    const sceneStore = useSceneStore.getState();
    const descriptions = sceneStore.scene.rigidBodies.filter((body) => body.id !== bodyId);
    sceneStore.patchScene({ rigidBodies: descriptions });
    this.bodies = this.bodies.filter((body) => body.description.id !== bodyId);
    this.publishBodies();
    useUIStore.getState().selectBody(descriptions[0]?.id);
    useRuntimeStore.getState().setRunState("paused");
    useRuntimeStore.getState().setNotice("Body removed");
  }

  updateBody(bodyId: string, patch: Partial<RigidBodyDescription>) {
    const sceneStore = useSceneStore.getState();
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

  dragBody(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase, orientation?: RigidBodyState["orientation"]) {
    if (phase === "end") this.kinematicDrag = null;
    else this.kinematicDrag = { bodyId, position: { ...position }, velocity: { ...velocity } };
    const body = this.bodies.find((candidate) => candidate.description.id === bodyId);
    if (body) {
      body.position_m = { ...position };
      if (orientation) body.orientation = { ...orientation };
      body.linearVelocity_m_s = phase === "end" ? { x: 0, y: 0, z: 0 } : { ...velocity };
      body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 };
      this.publishBodies();
    }
    const runtime = useRuntimeStore.getState();
    if (phase === "start") { runtime.setRunState("running"); runtime.setNotice("Kinematic drag active · GPU immersed boundary coupling"); }
    if (phase === "end") runtime.setNotice("Body released to buoyancy, drag, and collision response");
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
    const now = performance.now();
    const metricSampleDue = now - this.sampleClock > 250;
    if (metricSampleDue) this.sampleClock = now;
    const gpu = diagnostics.gpuInfo?.gpuTimings, previous = this.performance;
    // Timestamp-query wraparound can produce wildly negative or multi-hour
    // stage times; keep the previous sample rather than displaying garbage.
    const sane = (value: number | undefined, fallback: number) =>
      value !== undefined && Number.isFinite(value) && value >= 0 && value < 10_000 ? value : fallback;
    const methodId = metrics.methodId ?? useMethodStore.getState().methodId;
    const renderTimingContext = metrics.renderTimingContext ?? `${methodId}:legacy`;
    const samePhysicsMethod = previous.methodId === methodId;
    const sameRenderContext = previous.renderTimingContext === renderTimingContext;
    const physicsFallback = samePhysicsMethod ? previous : emptyPerformance;
    const renderFallback = sameRenderContext ? previous : emptyPerformance;
    const snapshot: PerformanceSnapshot = {
      methodId,
      renderTimingContext,
      renderTimingEpoch: metrics.renderTimingEpoch ?? renderFallback.renderTimingEpoch,
      renderTimingSampleId: metrics.renderTimingSampleId ?? renderFallback.renderTimingSampleId,
      gpuPhysicsTimingAvailable: Boolean(gpu),
      gpuRenderTimestampSupported: Boolean(metrics.gpuRenderTimestampAvailable),
      gpuRenderTimingAvailable: Boolean(metrics.gpuRenderTimestampAvailable && metrics.gpuRender_ms !== undefined),
      cpuSimulation_ms: this.cpuSimulationMs,
      cpuFrame_ms: metrics.cpuFrame_ms,
      cpuPhysicsSubmit_ms: metrics.cpuPhysicsSubmit_ms,
      cpuDataUpload_ms: metrics.cpuDataUpload_ms,
      cpuRenderEncode_ms: metrics.cpuRenderEncode_ms,
      adaptiveRebuildWall_ms: sane(diagnostics.gpuInfo?.quadtreeRebuildWall_ms, physicsFallback.adaptiveRebuildWall_ms),
      adaptiveRebuildPending: Boolean(diagnostics.gpuInfo?.quadtreeRebuildPending),
      adaptiveInlineTopology: Boolean(diagnostics.gpuInfo?.quadtreeInlineRebuild),
      adaptiveRebuildBlockedFrames: diagnostics.gpuInfo?.quadtreeRebuildBlockedFrames ?? physicsFallback.adaptiveRebuildBlockedFrames,
      adaptiveRebuildCompletedCount: diagnostics.gpuInfo?.quadtreeRebuildCompletedCount ?? physicsFallback.adaptiveRebuildCompletedCount,
      adaptiveGPUConstructionKernel_ms: sane(diagnostics.gpuInfo?.quadtreeGPUConstructionKernel_ms, physicsFallback.adaptiveGPUConstructionKernel_ms),
      adaptiveGPUSparsePack_ms: sane(diagnostics.gpuInfo?.quadtreeGPUSparsePack_ms, physicsFallback.adaptiveGPUSparsePack_ms),
      adaptiveCPUTopologyPack_ms: sane(diagnostics.gpuInfo?.quadtreeCPUTopologyPack_ms, physicsFallback.adaptiveCPUTopologyPack_ms),
      adaptiveCPURedistance_ms: sane(diagnostics.gpuInfo?.quadtreeCPURedistance_ms, physicsFallback.adaptiveCPURedistance_ms),
      adaptiveCPUQuadtreeDecode_ms: sane(diagnostics.gpuInfo?.quadtreeCPUQuadtreeDecode_ms, physicsFallback.adaptiveCPUQuadtreeDecode_ms),
      adaptiveCPUTallGrid_ms: sane(diagnostics.gpuInfo?.quadtreeCPUTallGrid_ms, physicsFallback.adaptiveCPUTallGrid_ms),
      adaptiveCPUVariationalAssembly_ms: sane(diagnostics.gpuInfo?.quadtreeCPUVariationalAssembly_ms, physicsFallback.adaptiveCPUVariationalAssembly_ms),
      adaptiveCPUSystemPack_ms: sane(diagnostics.gpuInfo?.quadtreeCPUSystemPack_ms, physicsFallback.adaptiveCPUSystemPack_ms),
      adaptiveCPUICFactorization_ms: sane(diagnostics.gpuInfo?.quadtreeCPUICFactorization_ms, physicsFallback.adaptiveCPUICFactorization_ms),
      adaptiveCPUResourceUpload_ms: sane(diagnostics.gpuInfo?.quadtreeCPUResourceUpload_ms, physicsFallback.adaptiveCPUResourceUpload_ms),
      gpuActiveStages: gpu?.activeStages ?? physicsFallback.gpuActiveStages,
      gpuPreparation_ms: sane(gpu?.preparation_ms, physicsFallback.gpuPreparation_ms),
      gpuLayerConstruction_ms: sane(gpu?.layerConstruction_ms, physicsFallback.gpuLayerConstruction_ms),
      gpuAdvection_ms: sane(gpu?.advection_ms, physicsFallback.gpuAdvection_ms),
      gpuConditioning_ms: sane(gpu?.conditioning_ms, physicsFallback.gpuConditioning_ms),
      gpuRemeshing_ms: sane(gpu?.remeshing_ms, physicsFallback.gpuRemeshing_ms),
      gpuPressure_ms: sane(gpu?.pressure_ms, physicsFallback.gpuPressure_ms),
      gpuProjection_ms: sane(gpu?.projection_ms, physicsFallback.gpuProjection_ms),
      gpuExtrapolation_ms: sane(gpu?.extrapolation_ms, physicsFallback.gpuExtrapolation_ms),
      gpuMaterialization_ms: sane(gpu?.materialization_ms, physicsFallback.gpuMaterialization_ms),
      gpuSurfaceUpdate_ms: sane(gpu?.surfaceUpdate_ms, physicsFallback.gpuSurfaceUpdate_ms),
      gpuRigid_ms: sane(gpu?.rigidCoupling_ms, physicsFallback.gpuRigid_ms),
      gpuSpraySimulation_ms: sane(gpu?.spray_ms, physicsFallback.gpuSpraySimulation_ms),
      gpuFluidResidency_ms: sane(gpu?.fluidResidency_ms, physicsFallback.gpuFluidResidency_ms),
      gpuSparsePublication_ms: sane(gpu?.sparsePublication_ms, physicsFallback.gpuSparsePublication_ms),
      gpuDiagnostics_ms: sane(gpu?.diagnostics_ms, physicsFallback.gpuDiagnostics_ms),
      gpuOverhead_ms: sane(gpu?.overhead_ms, physicsFallback.gpuOverhead_ms),
      gpuRender_ms: sane(metrics.gpuRender_ms, renderFallback.gpuRender_ms),
      gpuSurfaceExtraction_ms: sane(metrics.gpuSurfaceExtraction_ms, 0),
      gpuDryScene_ms: sane(metrics.gpuDryScene_ms, renderFallback.gpuDryScene_ms),
      gpuSvoTemporal_ms: sane(metrics.gpuSvoTemporal_ms, renderFallback.gpuSvoTemporal_ms),
      gpuInterfaces_ms: sane(metrics.gpuInterfaces_ms, renderFallback.gpuInterfaces_ms),
      gpuSprayFront_ms: sane(metrics.gpuSprayFront_ms, renderFallback.gpuSprayFront_ms),
      gpuSprayBack_ms: sane(metrics.gpuSprayBack_ms, renderFallback.gpuSprayBack_ms),
      gpuSprayRender_ms: sane(metrics.gpuSprayRender_ms, renderFallback.gpuSprayRender_ms),
      gpuOpticalComposite_ms: sane(metrics.gpuOpticalComposite_ms, renderFallback.gpuOpticalComposite_ms),
      gpuUpscale_ms: sane(metrics.gpuUpscale_ms, renderFallback.gpuUpscale_ms)
    };
    this.performance = snapshot;
    diagnostics.set({ frameMs: metrics.cpuFrame_ms, resolution });
    diagnostics.pushPerformance(snapshot, metricSampleDue ? { t: now / 1000, frame_ms: metrics.cpuFrame_ms, volume_drift_pct: this.fluidSolver.diagnostics.markerVolumeDrift * 100, constraint_error: this.fluidSolver.diagnostics.divergenceAfter_s, kinetic_energy_J: this.fluidSolver.diagnostics.kineticEnergy_J } : undefined);
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
    try { const loaded = parseScene(contents); this.reset(loaded); useRuntimeStore.getState().setNotice(`Loaded ${name}`); }
    catch (error) { useRuntimeStore.getState().setNotice(error instanceof Error ? error.message : "Scene import failed", "warn"); }
  }

  applyAndResetFluid() {
    this.reset(cloneScene(useSceneStore.getState().scene));
  }
}

export { BUILD_ID };
export const simulation = new SimulationController();
