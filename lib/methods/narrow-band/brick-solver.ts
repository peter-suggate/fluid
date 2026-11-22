/**
 * The brick engine as an app solver (GPUSolverInstance): the host and
 * contract adapter of design §5, lane E.
 *
 * - Construction: the scene adapter's setup (brick-scene.ts), the stages
 *   and pressure solver the method passes in, the engine, then the interim
 *   presentation.
 * - advanceTo: one fixed step of dt a call, submitted with no fence (no
 *   substeps), and `info.submittedTime_s` advanced by it. The app's queue
 *   ledger counts and bounds advances by that clock (webgpu-renderer
 *   submitPreparedGPUFluid); without it every frame queued another step
 *   unbounded and the app froze at t = 0. The presentation copy is encoded
 *   after the step as its own submission, outside the step's.
 * - Health: receipts are lagged. assertSimulationHealthy awaits the caller's
 *   completion (never a second fence behind later work) and raises a latched
 *   FAIL as a SimulationFailureError, which the app shows and stops on.
 * - Observers read `brickEngine` and `brickPresentation` after a fence
 *   (tools/nbflip-observe-brick.ts).
 * - Live voxel edits (M3 stage 2c): a scene that differs from the running one
 *   in its solid voxels alone is taken in place. The setup's own mask folds
 *   the edit (the same SolidWorld the editor's hit test reads), its rewritten
 *   words go up, and the engine restamps the bricks the changed cells reach
 *   (BrickEngine.restampStatics) between two steps. Any other scene change is
 *   refused here (applySceneUniforms answers false), and the renderer rebuilds,
 *   as it did before this solver adopted anything live.
 */
import type { GPUSolverInstance, InjectedLiquidBall } from "../../core/method-contract";
import { liveFluidEditRefusal, type LiveFluidEdit, type LiveFluidEditResult } from "../../core/live-fluid-edit";
import { brickLiveDrop } from "./brick-source";
import { brickDropRefusal } from "./brick-scene";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";
import { narrowBandSiteAxis, type NarrowBandSurfaceSamples } from "../uniform/uniform-narrow-band-sites";
import { NBF_PARTICLE_SETTINGS, nbfParticleStride } from "./particles-formulas.wgsl";
import { P_ENERGY_BUFFER } from "./particles-kernels.wgsl";
import type { GPUQuality } from "../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import type { RigidBodyState } from "../../core/rigid-body";
import { FAIL, H, STATE } from "./brick-abi";
import { SimulationFailureError, type SimulationFailure } from "../../core/simulation-failure";
import { BrickEngine } from "./brick-engine";
import type { BrickCensusHold } from "./brick-census.wgsl";
import { BrickPresentation } from "./brick-present";
import type { BrickDetailSurface } from "./brick-detail-surface";
import { BrickDetail, type BrickDetailSpec } from "./brick-detail";
import type { BrickSceneSetup } from "./brick-scene";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import type { BrickStagesFactory } from "./brick-stages";
import type { SceneDescription } from "../../core/model";
import type { GPURigidBodyPick, GPURigidBodyPose } from "../../core/webgpu-rigid-body";
import { BrickRigidBodies } from "./bodies-host";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { brickStaticCutBricks } from "./brick-static.wgsl";
import { brickBoundaryChangedCells, brickBoundaryImage } from "./brick-boundary.wgsl";
import { GPUStageTimestampRecorder } from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { gpuPhysicsPerformanceActivityFrameId } from "../../core/gpu-performance-activity";

/** The sampled advances' cadence (ms, wall), the uniform engine's: one timed step at most this often. */
const BRICK_PHYSICS_TRACE_CADENCE_MS = 100;

export interface BrickSolverOptions {
  readonly setup: BrickSceneSetup;
  readonly stages: BrickStagesFactory;
  readonly solver: PressureSolverFactory;
  readonly values: Readonly<Record<string, unknown>>;
  readonly quality: GPUQuality;
  /** The scene's far phi (m), where nothing is stored (fluidLatticeFarPhi). */
  readonly farPhi_m: number;
  /** Encode the presentation copy after each advance (default true; benches may turn it off). */
  readonly present?: boolean;
  /** Test and diagnostic only: the engine's census hold (BrickEngineOptions.censusHold). */
  readonly censusHold?: BrickCensusHold;
  /** The scene whose rigid bodies the solver hosts (lane P, M5: bodies-host.ts); absent, or no bodies: none. */
  readonly rigid?: SceneDescription;
  /** The scene's detail engines (brick-detail.ts brickDetailSpecs): each steps after the parent, driven by it. */
  readonly detail?: readonly BrickDetailSpec[];
}

/** Whether `scene` differs from `running` in its solid voxels alone (each other field the same object, or the same
 * content: a document store may rebuild a field it did not change). */
export function sceneDiffersInSolidsAlone(scene: SceneDescription, running: SceneDescription): boolean {
  const fields = Object.keys(scene) as (keyof SceneDescription)[];
  if (fields.length !== Object.keys(running).length) return false;
  for (const field of fields) {
    if (field === "solidVoxels") continue;
    if (!Object.hasOwn(running, field)) return false;
    if (scene[field] !== running[field] && JSON.stringify(scene[field]) !== JSON.stringify(running[field])) return false;
  }
  return true;
}

export class BrickNBFlipSolver implements GPUSolverInstance {
  readonly brickEngine: BrickEngine;
  readonly brickPresentation: BrickPresentation;
  /** Lane P's rigid bodies (M5), when the scene has any. */
  readonly rigidBodies?: BrickRigidBodies;
  /** The detail engines (brick-detail.ts), in the scene's box order. */
  readonly details: readonly BrickDetail[];
  readonly fluidDomain: GPUSolverInstance["fluidDomain"];
  readonly denseLevelSetVolumeSource: DenseLevelSetVolumeConsumerSource;
  readonly detailSurfaces: readonly BrickDetailSurface[];
  private presentEnabled: boolean;
  /** Presentation copies submitted, and the last copy's GPU completion time (ms, wall), apart from the steps. */
  presentations = 0;
  lastPresentationMs?: number;
  info: GPUEulerianInfo;
  /** The document the running state follows: the scene it was built from, then each live voxel edit taken. */
  private scene?: SceneDescription;
  /** The procedural boundary's image the engine holds (a live edit moves its validity words). */
  private boundary?: Uint32Array<ArrayBuffer>;
  /** The particle layer's frame (ParticleFrame: lattice cells, h, the lattice corner in world metres), written once. */
  private readonly particleFrame: GPUBuffer;
  /** The particle source of each bank (a record a bank: the source is rebuilt by no frame). */
  private readonly particleBanks: readonly [GPUFluidParticleSource, GPUFluidParticleSource];
  /** The stage timing (SIM panel): a timed step's readback in flight, the last sample's wall time and its id. */
  private tracePending = false;
  private lastTraceAt_ms = -Infinity;
  private traceSampleId = 0;
  private destroyed = false;

  private constructor(private readonly options: BrickSolverOptions, engine: BrickEngine, presentation: BrickPresentation, rigid: BrickRigidBodies | undefined, details: readonly BrickDetail[]) {
    this.brickEngine = engine; this.brickPresentation = presentation; this.presentEnabled = options.present ?? true; this.details = details; this.detailSurfaces = details.map(d => d.surface);
    if (rigid) this.rigidBodies = rigid;
    if (options.rigid) this.scene = options.rigid;
    if (options.setup.boundary) this.boundary = options.setup.boundary;
    const s = options.setup, h = s.h_m;
    this.fluidDomain = { origin_m: s.origin_m, cellSize_m: [h, h, h], dimensions: s.lattice };
    this.denseLevelSetVolumeSource = { vertexPhi: presentation.vertexPhi, contourVertexPhi: true, openFraction: presentation.openFraction, cellSize_m: [h, h, h] };
    this.info = { ...this.makeInfo(), submittedTime_s: 0, completedTime_s: 0, encodedSteps: 0, pipelineGraph: "brick" };
    const device = engine.device, o = s.origin_m;
    this.particleFrame = device.createBuffer({ label: "nb-brick.particleFrame", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.particleFrame, 0, new Float32Array([...s.lattice, 0, h, h, h, 0, o[0], o[1], o[2], 0]));
    this.particleBanks = [0, 1].map(bank => this.particleBankSource(bank)) as [GPUFluidParticleSource, GPUFluidParticleSource];
  }

  /**
   * The particle layer's records of `bank` (brick-abi.md §5.6 Banks): `parts` holds both banks at NB-FLIP's record
   * (positions in h cells from the lattice corner, velocity in 4..6, the depth in 7, the grid's velocity in 8..10 until
   * pDelta, the mode in the sign of 11), bank b from record b * PCAP, PCOUNT[b] live. pDelta overwrites 8..10 with the
   * FLIP delta, so Motion reads the energy it stored first (P_ENERGY, indexed in the bank). Eight surface samples a
   * cell (NBF_PARTICLE_SETTINGS), so a sphere is a little under a quarter cell in radius, as the old engine's.
   */
  private particleBankSource(bank: number): GPUFluidParticleSource {
    const e = this.brickEngine, L = e.layout, h = this.options.setup.h_m;
    // An APIC record (24 floats) keeps the FLIP record's 12 first (its affine rows follow), so the layer reads both.
    if (L.particleStride !== nbfParticleStride(L.apic)) throw new Error(`Brick engine: the particle layer reads NB-FLIP records (12 or 24 floats), not ${L.particleStride}`);
    return {
      buffer: e.buffer("parts"), strideFloats: L.particleStride, capacity: L.cap.particles, firstRecord: bank * L.cap.particles,
      positionScale_m: [h, h, h], frame: { buffer: this.particleFrame, size: 48 },
      radius_m: 0.22 * h * 2 / narrowBandSiteAxis(NBF_PARTICLE_SETTINGS.surfaceSamples as NarrowBandSurfaceSamples),
      liveCount: { buffer: e.buffer("set"), byteOffset: 4 * (H.PCOUNT0 + bank) }, motionEnergy: e.buffer(P_ENERGY_BUFFER),
      grid: { velocityFloat: 8, depthFloat: 7, ballisticFloat: 11, ballisticSignBit: true, surfaceDepth: 2 },
    };
  }

  /** The completed step's records: step n's rebin files bank (n & 1) ^ 1, so after `stepsSubmitted` steps the queue's
   * last step left them in bank stepsSubmitted & 1 (the boot's seeds are bank 0's). The renderer reads this every frame. */
  get particleSource(): GPUFluidParticleSource { return this.particleBanks[this.brickEngine.stepsSubmitted & 1]!; }

  static async create(device: GPUDevice, options: BrickSolverOptions): Promise<BrickNBFlipSolver> {
    const s = options.setup;
    const rigid = options.rigid?.rigidBodies?.length
      ? await BrickRigidBodies.create(device, options.rigid, { frame_m: s.origin_m, h_m: s.h_m, dt_s: s.dt_s, density_kg_m3: s.density_kg_m3, coupling: options.values.rigidCoupling !== "off", ...(s.terrain ? { terrain: s.terrain } : {}), ...(s.boundary ? { boundary: s.boundary } : {}) })
      : undefined;
    const engine = await BrickEngine.create(device, {
      lattice: s.lattice, ...(s.box ? { box: s.box } : {}), ...(s.openTop ? { openTop: true } : {}), ...(s.openSides ? { openSides: true } : {}), farPhi: s.farPhi, h_m: s.h_m, dt_s: s.dt_s, gravity_mps2: s.gravity_mps2, capacities: s.capacities, liquid: s.liquid, initial: s.initial, initialVelocity: s.initialVelocity, ...(s.sources ? { sources: s.sources } : {}), liveDrops: true, ...(options.censusHold ? { censusHold: options.censusHold } : {}),
      solver: options.solver, stages: options.stages, values: options.values, label: s.sceneId,
      fluid: { density_kg_m3: s.density_kg_m3, viscosity_Pa_s: s.viscosity_Pa_s, surfaceTension_N_m: s.surfaceTension_N_m },
      particles: { bandDepth: s.particleBandDepth },
      ...(rigid ? { bodies: rigid } : {}),
      ...(s.staticMask ? { staticMask: s.staticMask } : {}), ...(s.staticsDeferred ? { staticsDeferred: true } : {}), ...(s.terrain ? { terrain: s.terrain } : {}), ...(s.boundary ? { boundary: s.boundary } : {}),
    });
    const presentation = await BrickPresentation.create(engine, options.farPhi_m, options.setup.presentCells);
    const details: BrickDetail[] = [];
    try { for (const spec of options.detail ?? []) details.push(await BrickDetail.create(device, engine, spec, { solver: options.solver, values: options.values })); }
    catch (error) { details.forEach(d => d.destroy()); presentation.destroy(); engine.destroy(); rigid?.destroy(); throw error; }
    // The stage timing's closing marker, compiled before any sampled step asks for it (markersReady).
    await GPUStageTimestampRecorder.prepare(device);
    const solver = new BrickNBFlipSolver(options, engine, presentation, rigid, details);
    solver.present();
    return solver;
  }

  get volumeTexture(): GPUTexture { return this.brickPresentation.volume; }

  /**
   * A ball (or a 2D case's disk) dropped into the running scene, with no re-seed: a live drop (brick-source.ts
   * brickLiveDrop) the next submitted step adds, as a scheduled drop is added (its box claimed by that step's head, its
   * volume joining the budget). False when the fixed pools may not take it (brickDropRefusal): the app then re-seeds
   * the scene with the ball, whose pools the pool rule sizes.
   */
  injectLiquidBall(ball: InjectedLiquidBall): boolean {
    return this.liveDropRefusal(ball) === undefined;
  }

  /** The editor's water shapes, as far as a drop reaches (the old engine's editFluid): a ball, added. */
  async editFluid(edit: LiveFluidEdit): Promise<LiveFluidEditResult> {
    if (edit.operation !== "add" || edit.shape !== "ball") return { accepted: false, reason: "The brick engine adds water as balls only." };
    const s = this.options.setup, cells = s.presentCells ?? s.lattice;
    const refusal = liveFluidEditRefusal(edit, { origin_m: s.origin_m, cellSize_m: [s.h_m, s.h_m, s.h_m], dimensions: cells });
    if (refusal) return { accepted: false, reason: refusal };
    const reason = this.liveDropRefusal({ centre_m: edit.center_m, radius_m: edit.radius_m });
    return reason === undefined ? { accepted: true } : { accepted: false, reason: `The ball does not fit this run's pools (${reason}); drop it while paused to re-seed.` };
  }

  /** Queue the drop, or say why not (nothing is queued then). */
  private liveDropRefusal(ball: InjectedLiquidBall): string | undefined {
    if (!(ball.radius_m > 0)) return "a ball needs a positive radius";
    const e = this.brickEngine, s = this.options.setup;
    if (e.failure) return "the run has failed";
    const drop = brickLiveDrop(ball, e.nextDropTime(), { lattice: s.lattice, ...(s.box ? { box: s.box } : {}), h_m: s.h_m, origin_m: s.origin_m });
    if (drop.box[0].some((v, a) => v >= drop.box[1][a]!)) return "the ball lies outside the fluid box";
    const reason = brickDropRefusal(drop, e.receipt?.header, e.layout.cap, e.stages.tiered === true, s.particleBandDepth, e.layout.window);
    if (reason !== undefined) return reason;
    e.queueDrop(drop);
    return undefined;
  }
  get surfaceFieldTexture(): GPUTexture { return this.brickPresentation.volume; }
  /** False only while a voxel stroke waits on the live-edit pipelines (BrickEngine.editHeld): the app holds the clock. */
  get simulationReady(): boolean { return !this.brickEngine.editHeld; }

  /** The live-edit pipelines, compiled in the background after creation (harnesses await them before timing a step). */
  async waitForSimulationReady(): Promise<void> {
    await this.brickEngine.editsReady;
    if (this.brickEngine.editsError) throw new Error(`Brick engine: ${this.brickEngine.editsError}`);
  }
  get framePending(): boolean { return false; }
  get presentationPending(): boolean { return false; }

  /** Turn the per-advance presentation copy on or off (benches time the step alone). */
  setPresentationEnabled(enabled: boolean) { this.presentEnabled = enabled; }

  private present() {
    const d = this.brickEngine.device, enc = d.createCommandEncoder({ label: "nb-brick.present" });
    this.brickPresentation.encode(enc);
    for (const d of this.details) d.present(enc);
    const t0 = performance.now();
    d.queue.submit([enc.finish()]);
    this.presentations++;
    void d.queue.onSubmittedWorkDone().then(() => { this.lastPresentationMs = performance.now() - t0; });
  }

  /** One step toward `time_s` a call (the app's ledger asks again while the clock owes more). */
  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean {
    if (this.info.simulationFailure) return false;
    const dt = this.options.setup.dt_s, submitted = this.info.submittedTime_s ?? 0;
    if (time_s + 1e-9 < submitted + dt) return false;
    // A held stroke: no step until its pipelines land (simulationReady is false; the app does not ask).
    if (this.brickEngine.editHeld) return false;
    this.rigidBodies?.offer(bodies);
    try {
      if (!this.advanceSampled()) this.brickEngine.advance(1);
      for (const d of this.details) d.advance();
    } catch (error) {
      // A latched FAIL (lagged receipt) refuses further steps: report it, never step on.
      this.refresh({ simulationFailure: this.failure(error instanceof Error ? error.message : String(error)) });
      return false;
    }
    if (this.presentEnabled) this.present();
    this.refresh({ submittedTime_s: submitted + dt, encodedSteps: (this.info.encodedSteps ?? 0) + 1 });
    return true;
  }

  /**
   * One step timed, when the instrumentation is on and the cadence is due (BrickEngine.advanceTimed): its phases
   * (brick-timing.ts) land in info.physicsTrace once read back, an exact partition of the step's GPU time (part 1,
   * part 2 and the rigid integrator; the presentation copy is the renderer's, not the step's). False: nothing was
   * submitted, the caller steps untimed. Hardware timestamps only: no substitute timing is reported.
   */
  private advanceSampled(): boolean {
    const instrumentation = usePerformanceInstrumentationStore.getState();
    if (!instrumentation.enabled || this.tracePending) return false;
    const requested = performance.now();
    if (requested - this.lastTraceAt_ms < BRICK_PHYSICS_TRACE_CADENCE_MS) return false;
    const device = this.brickEngine.device;
    if (!GPUStageTimestampRecorder.supported(device) || !GPUStageTimestampRecorder.markersReady(device)) {
      this.info.physicsTrace = undefined;
      this.info.physicsTraceUnavailable = "GPU hardware timing is unavailable. No substitute timing is reported.";
      return false;
    }
    const sampleId = ++this.traceSampleId, context = `brick:sim-${(this.info.submittedTime_s ?? 0).toFixed(6)}`;
    this.tracePending = true; this.lastTraceAt_ms = requested;
    let segments: GPUStageTimestampRecorder[];
    try { segments = this.brickEngine.advanceTimed(() => new GPUStageTimestampRecorder(device, sampleId, "physics", context)); }
    catch (error) { this.tracePending = false; throw error; }
    this.info.physicsCaptureIdentity = { sampleId, context, frameId: gpuPhysicsPerformanceActivityFrameId({ sampleId, context }) };
    // A segment may close at zero under the browser's timestamp quantization (the rigid integrator's): its boundaries
    // are still sampled and ordered, so it reads as a zero, not as an invalid sample.
    void Promise.all(segments.map(segment => segment.read(true))).then(traces => {
      const current = usePerformanceInstrumentationStore.getState();
      if (this.destroyed || !current.enabled || current.enabledAt_ms > requested) return;
      if (traces.length === 0 || traces.some(trace => !trace)) {
        this.info.physicsTrace = undefined;
        this.info.physicsTraceUnavailable = "GPU hardware timing was invalid. No substitute timing is reported.";
        return;
      }
      const complete = traces.map(trace => trace!);
      this.info.physicsTrace = { ...complete.at(-1)!, total_ms: complete.reduce((sum, trace) => sum + trace.total_ms, 0), phases: complete.flatMap(trace => trace.phases) };
      this.info.physicsTraceUnavailable = undefined;
    }).catch(error => {
      if (!this.destroyed) { this.info.physicsTrace = undefined; this.info.physicsTraceUnavailable = `GPU hardware timing failed: ${String(error)}`; }
    }).finally(() => { this.tracePending = false; });
    return true;
  }

  async awaitFrameCompletion() {
    const t = this.info.submittedTime_s;
    await this.brickEngine.fence();
    this.info.completedTime_s = Math.max(this.info.completedTime_s ?? 0, t ?? 0);
  }

  /** The caller's completion, then the lagged receipt's latch: no second fence behind later submitted work. */
  async assertSimulationHealthy(completion?: Promise<void>) {
    if (completion) await completion; else await this.brickEngine.fence();
    const failure = this.info.simulationFailure
      ?? (this.brickEngine.failure ? this.failure(this.brickEngine.failure) : undefined);
    if (failure) { this.info.simulationFailure = failure; throw new SimulationFailureError(failure); }
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.brickEngine.fence();
    const header = await this.brickEngine.read("set", 0, 256);
    this.refresh({}, header);
    return this.info;
  }

  /** New engine words into `info`, keeping the clock and the fields the app's ledger writes. */
  private refresh(extra: Partial<GPUEulerianInfo>, header?: Uint32Array) {
    this.info = { ...this.info, ...this.makeInfo(header), ...extra };
  }

  private failure(message: string): SimulationFailure {
    const h = this.brickEngine.receipt?.header, fail = h?.[H.FAIL] ?? 0, info = h?.[H.FAIL_INFO] ?? 0;
    const names = Object.entries(FAIL).filter(([, b]) => (fail & b) !== 0).map(([n]) => n);
    return {
      method: "uniform-narrow-band-flip", code: `BRICK_${names.join("_") || "FAIL"}`, message, kernel: "nb-brick",
      frame: this.brickEngine.receipt?.step ?? 0, generation: 0, ownerId: info, operands: [fail, info],
      operandNames: ["FAIL", "FAIL_INFO"], rawWords: h ? Array.from(h.subarray(0, 16)) : [], scene: this.options.setup.sceneId,
      time_s: this.info.completedTime_s,
    };
  }

  private makeInfo(header = this.brickEngine.receipt?.header): GPUEulerianInfo {
    const s = this.options.setup, [nx, ny, nz] = s.lattice, cells = nx * ny * nz;
    const liveF = header?.[H.LIVE_F] ?? 0, stored = 512 * liveF;
    const word = (at?: number) => at === undefined || !header ? undefined : header[at];
    return {
      nx, ny, nz, ...(s.presentCells ? { tankCells: s.presentCells } : {}), storedNy: ny, cellCount: stored, equivalentUniformCells: cells, compressionRatio: stored > 0 ? cells / stored : 0,
      regularLayers: 0, maximumNeighborDelta: 0, gridKind: "uniform", cellSize_m: s.h_m,
      pressureIterations: word(this.brickEngine.solver.receipt?.iterations) ?? 0,
      allocatedBytes: this.brickEngine.bytes + this.brickPresentation.bytes, quality: this.options.quality,
      uniformPressureCappedFrames: word(this.brickEngine.solver.receipt?.capped),
      brickEngine: { liveF, liveC: header?.[H.LIVE_C] ?? 0, peakF: header?.[H.PEAK_F] ?? 0, pressureCapNow: header ? (header[H.STATE]! & STATE.PRESSURE_CAP) !== 0 : undefined,
        step: header?.[H.STEP] ?? 0, presentations: this.presentations, lastPresentationMs: this.lastPresentationMs,
        // After step k the records are bank k & 1's (brick-abi.md §5.6 Banks: step n's rebin writes bank (n & 1) ^ 1).
        particles: header ? Math.min(header[H.PCOUNT0 + (header[H.STEP]! & 1)]!, this.brickEngine.layout.cap.particles) : 0 },
    };
  }

  /**
   * Preflight a live voxel edit without changing anything: the solver is running, it holds a static mask, the scene
   * differs from the running one in its solid voxels alone, and its voxels compose (an unknown patch kind or a region
   * the mask does not read fails here, explicitly), and it fits what the engine was built with: a trial fold on a copy
   * of the mask (its sealed page slots and child tables) and the bricks it cuts against the K pool. A refused edit
   * changes nothing; the run goes on.
   */
  validateLiveSolidEdit(scene: SceneDescription): void {
    if (this.info.simulationFailure) throw new Error("The brick engine has halted; reset the scene.");
    const mask = this.options.setup.staticMask;
    if (!mask || !this.scene) throw new Error("This brick engine was built with no static mask (a hand-built setup); reset the scene to apply a voxel edit.");
    if (!sceneDiffersInSolidsAlone(scene, this.scene)) throw new Error("The brick engine takes a voxel stroke live only when nothing else in the scene changes; reset the scene to apply this edit.");
    // An open world's mask is its widened box scene's (brick-scene.ts brickOpenWorldSetup), and an open world takes no voxels.
    if (this.options.setup.openSides) throw new Error("An open world (container.walls \"none\") takes no solid voxels; the brick engine refuses the edit.");
    const world = solidWorldForFluidLattice(scene), trial = mask.clone();
    try { trial.update(world); } catch (error) {
      throw new Error(`This voxel edit needs more of the solid mask than the brick engine was built with (${error instanceof Error ? error.message : String(error)}); reset the scene to apply it.`);
    }
    // A sculpt cut of the ground is not read by the brick engine (brick-terrain.wgsl.ts): refused, never run as open ground.
    if (this.options.setup.terrain && (scene.solidVoxels ?? []).some(p => p.operation === "clear" && p.affectsScenery === true)) throw new Error("The brick engine does not read a sculpt cut of the terrain; reset the scene to rebuild it on the uniform engine.");
    const s = this.options.setup;
    const field = s.boundary ? brickBoundaryImage(scene) : undefined;
    if (s.boundary && field?.length !== s.boundary.length) throw new Error("This edit changes the procedural boundary's bricks; reset the scene to apply it.");
    const cut = brickStaticCutBricks(trial, s.terrain || field ? { ...(s.terrain ? { terrain: s.terrain } : {}), ...(field ? { boundary: field } : {}), box: s.box ?? s.lattice } : undefined), k = this.brickEngine.layout.cap.k;
    if (cut > k) throw new Error(`This voxel edit cuts ${cut} bricks, over the ${k} K slots the brick engine was built with; reset the scene to apply it.`);
    // The detail engines hold the same patches, cut to their boxes and refined (brick-detail-scene.ts).
    for (const d of this.details) {
      try { d.validateSolids(scene); } catch (error) { throw new Error(`${error instanceof Error ? error.message : String(error)}; reset the scene to apply it.`); }
    }
  }

  /**
   * Adopt `scene` live, or answer false and leave everything as it was: the renderer then rebuilds. Taken live: a scene
   * whose solid voxels alone differ (validateLiveSolidEdit). The mask folds the edit; a fold the mask refuses (its slots,
   * sealed at construction, all held: the preflight refuses it first) poisons the mask, so it latches a simulation
   * failure and throws.
   */
  applySceneUniforms(scene: SceneDescription): void | false {
    if (scene === this.scene) return;
    const mask = this.options.setup.staticMask;
    if (!mask || !this.scene || this.info.simulationFailure || !sceneDiffersInSolidsAlone(scene, this.scene)) return false;
    // An open world's mask is its widened box scene's: never folded live (the rebuild refuses the voxels, validateScene).
    if (this.options.setup.openSides) return false;
    // A sculpt cut of the ground: the rebuild takes it (the brick engine refuses it, brickSceneSetup's gap).
    if (this.options.setup.terrain && (scene.solidVoxels ?? []).some(p => p.operation === "clear" && p.affectsScenery === true)) return false;
    const world = solidWorldForFluidLattice(scene);
    const lo: [number, number, number] = [Infinity, Infinity, Infinity], hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    let ranges: ReturnType<typeof mask.update>;
    try {
      ranges = mask.update(world, (x, y, z) => {
        lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
        hi[0] = Math.max(hi[0], x + 1); hi[1] = Math.max(hi[1], y + 1); hi[2] = Math.max(hi[2], z + 1);
      });
    } catch (error) {
      const message = `live voxel edit refused: ${error instanceof Error ? error.message : String(error)}`;
      this.refresh({ simulationFailure: this.failure(message) });
      throw error;
    }
    // The boundary's validity (a later voxel edit owns each cell it touches): the cells whose bit moved join the box,
    // since a cut of a record cell that is no mask voxel moves no mask bit.
    const boundary = this.boundary ? brickBoundaryImage(scene) : undefined;
    if (this.boundary && boundary) {
      if (boundary.length !== this.boundary.length) { this.refresh({ simulationFailure: this.failure("live voxel edit refused: the procedural boundary's bricks changed") }); throw new Error("the procedural boundary's bricks changed"); }
      brickBoundaryChangedCells(this.boundary, boundary, (x, y, z) => {
        lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
        hi[0] = Math.max(hi[0], x + 1); hi[1] = Math.max(hi[1], y + 1); hi[2] = Math.max(hi[2], z + 1);
      });
      this.boundary = boundary;
      this.rigidBodies?.updateBoundary(boundary);
    }
    this.scene = scene;
    this.rigidBodies?.system.setScene(scene);
    // A paint or a fill over solid moves no bit: nothing to upload or restamp.
    if (ranges && hi[0] > lo[0]) this.brickEngine.restampStatics(mask.words, ranges, lo, hi, boundary);
    // The same edit opens every detail engine's own cells (its mask is the box's cut of the same patches).
    for (const d of this.details) {
      try { d.foldSolids(scene); } catch (error) {
        const message = `live voxel edit refused by the detail engine: ${error instanceof Error ? error.message : String(error)}`;
        this.refresh({ simulationFailure: this.failure(message) });
        throw error;
      }
    }
  }

  // The rigid contract (method-contract.ts), the shared system's.
  get rigidRenderBuffer(): GPUBuffer | undefined { return this.rigidBodies?.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer | undefined { return this.rigidBodies?.motionBuffer; }
  /** The roster for the next step's sync (the old host's syncRigidBodies; prescribed bodies are sampled there). */
  syncRigidBodies(bodies: readonly RigidBodyState[]): void { this.rigidBodies?.offer(bodies); }
  setSelectedRigidBody(index: number): void { this.rigidBodies?.system.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]): Promise<GPURigidBodyPick | undefined> { return this.rigidBodies ? this.rigidBodies.system.pick(origin, direction) : Promise.resolve(undefined); }
  readRigidBodyPoses(): Promise<GPURigidBodyPose[] | undefined> { return this.rigidBodies ? this.rigidBodies.readPoses() : Promise.resolve(undefined); }

  destroy() { this.destroyed = true; this.details.forEach(d => d.destroy()); this.brickPresentation.destroy(); this.brickEngine.destroy(); this.rigidBodies?.destroy(); this.particleFrame.destroy(); }
}
