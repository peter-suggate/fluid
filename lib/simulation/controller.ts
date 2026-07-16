import { BUILD_ID, cloneScene, createRunManifest, parseScene, serializeScene, type SceneDescription } from "../model";
import { EulerianFluidSolver } from "../eulerian-solver";
import { advanceRigidBodies, boundingRadius, cloneRigidBodies, createBodyDescription, initializeRigidBodies, initializeRigidBody, rigidDiagnostics, type RigidBodyState, type RigidExternalLoad, type RigidStepDiagnostics } from "../rigid-body";
import type { RigidBodyDescription } from "../model";
import { applyFluidReactions, computeFluidLoads, type CouplingDiagnostics } from "../fluid-rigid-coupling";
import { mergeGPURigidLoads, type GPURigidLoad } from "../webgpu-eulerian";
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
import { commitGPUCompletion, gpuBatchDepth, gpuCanAcceptNextStep } from "./gpu-clock";
import { externalLoadsFromGPU } from "./gpu-loads";

export type BodyDragPhase = "start" | "move" | "end";

const MAX_BODIES = 12;

function downloadText(name: string, text: string, mime = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

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
  private gpuRigidLoads: GPURigidLoad[] = [];
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

  /** Fixed-step advance driven by the shell's requestAnimationFrame. */
  tick(now: number) {
    const tickStart = performance.now();
    if (this.lastClock === null) this.lastClock = now;
    const elapsed = Math.min((now - this.lastClock) / 1000, 0.05);
    this.lastClock = now;
    const runtime = useRuntimeStore.getState();
    if (runtime.runState !== "running") {
      if (runtime.simRate !== null) runtime.setSimRate(null);
      this.rateWallClock = 0;
      return;
    }
    const scene = useSceneStore.getState().scene;
    const backend = this.backend;
    const methodId = useMethodStore.getState().methodId;
    this.accumulator += elapsed;
    const dt = scene.numerics.fixedDt_s;
    // Restricted tall cells use a frame-sized partitioned-coupling batch: all
    // GPU impulses are accumulated over the batch and consumed over the next
    // fixed rigid substeps. Other coupled methods retain a one-step handshake.
    const batchDepth = backend === "webgpu" ? gpuBatchDepth(methodId, dt, this.bodies.length > 0) : 1;
    const gpuCanQueue = () => backend !== "webgpu" || this.simulationTime < this.gpuCompletedTime + batchDepth * dt - 1e-9;
    let steps = 0;
    let diagnostics: RigidStepDiagnostics | undefined;
    let fluidDiagnostics: ReturnType<EulerianFluidSolver["step"]> | undefined;
    let latestCoupling: CouplingDiagnostics | undefined;
    while (this.accumulator >= dt && steps < Math.max(2, batchDepth) && gpuCanQueue()) {
      this.applyDragConstraint();
      let loads: ReadonlyMap<string, RigidExternalLoad>;
      if (backend === "webgpu") {
        const gpuCoupling = externalLoadsFromGPU(scene, this.gpuRigidLoads, dt, this.bodies);
        loads = gpuCoupling.loads; latestCoupling = gpuCoupling.diagnostics;
      } else {
        const coupling = computeFluidLoads(scene, this.fluidSolver, this.bodies);
        latestCoupling = applyFluidReactions(this.fluidSolver, this.bodies, coupling.loads, dt); loads = coupling.loads;
      }
      diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, loads);
      this.applyDragConstraint();
      this.cpuOracleStep += 1;
      // The adaptive GPU method can enqueue several independent fluid steps;
      // running the coarse CPU oracle every four of those steps becomes the
      // main-thread bottleneck. It remains a low-rate diagnostic here and the
      // explicit Validation workflow still runs its dedicated comparisons.
      const oracleStride = backend === "webgpu" ? (methodId === "quadtree-tall-cell" ? 32 : 4) : 1;
      if (this.cpuOracleStep % oracleStride === 0) fluidDiagnostics = this.fluidSolver.step(dt * oracleStride);
      this.accumulator -= dt;
      this.simulationTime += dt;
      steps += 1;
    }
    if (backend === "webgpu" && !gpuCanQueue()) this.accumulator = Math.min(this.accumulator, dt);
    else if (steps === 2 && this.accumulator > dt * 2) this.accumulator = dt * 2;
    if (steps > 0) {
      this.publishBodies(diagnostics);
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
    const gpuCoupling = backend === "webgpu" ? externalLoadsFromGPU(scene, this.gpuRigidLoads, dt, this.bodies) : undefined;
    const coupling = gpuCoupling ? undefined : computeFluidLoads(scene, this.fluidSolver, this.bodies);
    const couplingDiagnostics = gpuCoupling?.diagnostics ?? applyFluidReactions(this.fluidSolver, this.bodies, coupling!.loads, dt);
    const diagnostics = advanceRigidBodies(this.bodies, scene, dt, 6, gpuCoupling?.loads ?? coupling!.loads);
    const fluidDiagnostics = this.fluidSolver.step(dt);
    this.simulationTime += dt;
    if (backend === "cpu-reference") runtime.setSimulationTime(this.simulationTime);
    this.publishBodies(diagnostics);
    useDiagnosticsStore.getState().set({ fluidState: fluidDiagnostics, fluidRenderState: this.fluidSolver.getRenderState(), couplingState: couplingDiagnostics });
  }

  reset(source?: SceneDescription) {
    const sceneStore = useSceneStore.getState();
    const scene = source ?? sceneStore.scene;
    if (source) sceneStore.setScene(source);
    this.bodies = initializeRigidBodies(scene.rigidBodies);
    this.fluidSolver = this.buildFluidSolver(scene);
    this.simulationTime = 0; this.gpuCompletedTime = 0; this.accumulator = 0; this.lastClock = null;
    this.cpuOracleStep = 0; this.cpuSimulationMs = 0;
    this.gpuRigidLoads = []; this.kinematicDrag = null;
    this.performance = emptyPerformance;
    this.publishBodies(rigidDiagnostics(this.bodies, scene.fluid.gravity_m_s2));
    useDiagnosticsStore.getState().set({ fluidState: this.fluidSolver.diagnostics, fluidRenderState: this.fluidSolver.getRenderState(), gpuInfo: null, couplingState: { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 }, samples: [], performanceSnapshot: emptyPerformance, performanceHistory: [] });
    const runtime = useRuntimeStore.getState();
    runtime.setSimulationTime(0);
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
    useMethodStore.getState().setQuality(quality);
    this.reset();
    useRuntimeStore.getState().setNotice(`Quality ${quality} · simulation reset`);
  }

  setMethod(methodId: string) {
    useMethodStore.getState().setMethodId(methodId);
    this.reset();
    useRuntimeStore.getState().setNotice(`${getMethod(methodId).label} selected`);
  }

  /** GPU methods rebuild from the config key; the CPU solver needs an explicit reset. */
  setMethodParam(methodId: string, key: string, value: MethodParamValue) {
    useMethodStore.getState().setParam(methodId, key, value);
    if (methodId === useMethodStore.getState().methodId && getMethod(methodId).backend === "cpu") this.reset();
  }

  resetMethodParam(methodId: string, key: string) {
    useMethodStore.getState().resetParam(methodId, key);
    if (methodId === useMethodStore.getState().methodId && getMethod(methodId).backend === "cpu") this.reset();
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
    if (description) this.bodies = this.bodies.map((body) => body.description.id === bodyId ? initializeRigidBody(description) : body);
    this.publishBodies();
    useRuntimeStore.getState().setRunState("paused");
    useRuntimeStore.getState().setNotice("Body parameters updated; simulation paused");
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

  dragBody(bodyId: string, position: RigidBodyState["position_m"], velocity: RigidBodyState["linearVelocity_m_s"], phase: BodyDragPhase) {
    if (phase === "end") this.kinematicDrag = null;
    else this.kinematicDrag = { bodyId, position: { ...position }, velocity: { ...velocity } };
    const body = this.bodies.find((candidate) => candidate.description.id === bodyId);
    if (body) {
      body.position_m = { ...position };
      body.linearVelocity_m_s = phase === "end" ? { x: 0, y: 0, z: 0 } : { ...velocity };
      body.angularVelocity_rad_s = { x: 0, y: 0, z: 0 }; body.angularMomentum_kg_m2_s = { x: 0, y: 0, z: 0 };
      this.publishBodies();
    }
    const runtime = useRuntimeStore.getState();
    if (phase === "start") { runtime.setRunState("running"); runtime.setNotice("Kinematic drag active · GPU immersed boundary coupling"); }
    if (phase === "end") runtime.setNotice("Body released to buoyancy, drag, and collision response");
  }

  // ---- renderer callbacks ------------------------------------------------

  mergeGPULoads(loads: GPURigidLoad[]) {
    this.gpuRigidLoads = mergeGPURigidLoads(this.gpuRigidLoads, loads);
  }

  /** Publish transport time only after the corresponding GPU work completes. */
  gpuAdvanceCompleted(time_s: number) {
    if (this.backend !== "webgpu") return;
    const completed = commitGPUCompletion(this.simulationTime, this.gpuCompletedTime, time_s);
    if (completed === this.gpuCompletedTime) return;
    this.gpuCompletedTime = completed;
    useRuntimeStore.getState().setSimulationTime(completed);
  }

  recordFrame(metrics: RendererFrameMetrics, resolution: string) {
    const diagnostics = useDiagnosticsStore.getState();
    const now = performance.now();
    if (now - this.sampleClock <= 250) {
      if (metrics.cpuFrame_ms !== diagnostics.frameMs || resolution !== diagnostics.resolution) diagnostics.set({ frameMs: metrics.cpuFrame_ms, resolution });
      return;
    }
    this.sampleClock = now;
    const gpu = diagnostics.gpuInfo?.gpuTimings, previous = this.performance;
    // Timestamp-query wraparound can produce wildly negative or multi-hour
    // stage times; keep the previous sample rather than displaying garbage.
    const sane = (value: number | undefined, fallback: number) =>
      value !== undefined && Number.isFinite(value) && value >= 0 && value < 10_000 ? value : fallback;
    const methodId = metrics.methodId ?? useMethodStore.getState().methodId;
    const waterRenderMode = metrics.waterRenderMode ?? useUIStore.getState().waterRenderMode;
    const samePhysicsMethod = previous.methodId === methodId;
    const sameRenderMethod = samePhysicsMethod && previous.waterRenderMode === waterRenderMode;
    const physicsFallback = samePhysicsMethod ? previous : emptyPerformance;
    const renderFallback = sameRenderMethod ? previous : emptyPerformance;
    const snapshot: PerformanceSnapshot = {
      methodId,
      waterRenderMode,
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
      adaptiveRebuildBlockedFrames: diagnostics.gpuInfo?.quadtreeRebuildBlockedFrames ?? physicsFallback.adaptiveRebuildBlockedFrames,
      gpuLayerConstruction_ms: sane(gpu?.layerConstruction_ms, physicsFallback.gpuLayerConstruction_ms),
      gpuAdvection_ms: sane(gpu?.advection_ms, physicsFallback.gpuAdvection_ms),
      gpuPressure_ms: sane(gpu?.pressure_ms, physicsFallback.gpuPressure_ms),
      gpuProjection_ms: sane(gpu?.projection_ms, physicsFallback.gpuProjection_ms),
      gpuRigid_ms: sane(gpu?.rigidCoupling_ms, physicsFallback.gpuRigid_ms),
      gpuDiagnostics_ms: sane(gpu?.diagnostics_ms, physicsFallback.gpuDiagnostics_ms),
      gpuOverhead_ms: sane(gpu?.overhead_ms, physicsFallback.gpuOverhead_ms),
      gpuRender_ms: sane(metrics.gpuRender_ms, renderFallback.gpuRender_ms),
      gpuSurfaceExtraction_ms: sane(metrics.gpuSurfaceExtraction_ms, 0),
      gpuDryScene_ms: sane(metrics.gpuDryScene_ms, renderFallback.gpuDryScene_ms),
      gpuInterfaces_ms: sane(metrics.gpuInterfaces_ms, renderFallback.gpuInterfaces_ms),
      gpuOpticalComposite_ms: sane(metrics.gpuOpticalComposite_ms, renderFallback.gpuOpticalComposite_ms),
      gpuUpscale_ms: sane(metrics.gpuUpscale_ms, renderFallback.gpuUpscale_ms)
    };
    this.performance = snapshot;
    diagnostics.set({ frameMs: metrics.cpuFrame_ms, resolution });
    diagnostics.pushPerformance(snapshot, { t: now / 1000, frame_ms: metrics.cpuFrame_ms, volume_drift_pct: this.fluidSolver.diagnostics.markerVolumeDrift * 100, constraint_error: this.fluidSolver.diagnostics.divergenceAfter_s, kinetic_energy_J: this.fluidSolver.diagnostics.kineticEnergy_J });
  }

  // ---- persistence -------------------------------------------------------

  saveScene() {
    const scene = useSceneStore.getState().scene;
    localStorage.setItem("fluid-lab.scene.v1", serializeScene(scene));
    downloadText(`${scene.sceneId}.fluid.json`, serializeScene(scene));
    useRuntimeStore.getState().setNotice("Scene saved locally and exported");
  }

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

  exportMetrics() {
    const scene = useSceneStore.getState().scene;
    const method = useMethodStore.getState();
    const diagnostics = useDiagnosticsStore.getState();
    const adapter = diagnostics.gpuStatus.state === "ready" ? diagnostics.gpuStatus.adapter : diagnostics.gpuStatus.label;
    const payload = {
      manifest: createRunManifest(scene, adapter),
      backend: this.backend,
      method: method.methodId,
      gpuQuality: method.quality,
      methodOverrides: method.overrides[method.methodId] ?? {},
      gpuInfo: diagnostics.gpuInfo,
      shellMetrics: { presentationFrame_ms: diagnostics.frameMs, canvasResolution: diagnostics.resolution, samples: diagnostics.samples, performance: diagnostics.performanceSnapshot, performanceHistory: diagnostics.performanceHistory },
      rigidBodyState: diagnostics.bodies,
      rigidBodyDiagnostics: diagnostics.rigidState,
      eulerianMetrics: diagnostics.fluidState,
      couplingMetrics: diagnostics.couplingState
    };
    downloadText(`fluid-lab-run-${Date.now()}.json`, JSON.stringify(payload, null, 2) + "\n");
    useRuntimeStore.getState().setNotice("Run manifest and shell metrics exported");
  }

  applyAndResetFluid() {
    this.reset(cloneScene(useSceneStore.getState().scene));
  }
}

export { BUILD_ID };
export const simulation = new SimulationController();
