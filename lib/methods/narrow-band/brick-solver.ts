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
 */
import type { GPUSolverInstance } from "../../core/method-contract";
import type { GPUQuality } from "../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import type { RigidBodyState } from "../../core/rigid-body";
import { FAIL, H, STATE } from "./brick-abi";
import { SimulationFailureError, type SimulationFailure } from "../../core/simulation-failure";
import { BrickEngine } from "./brick-engine";
import type { BrickCensusHold } from "./brick-census.wgsl";
import { BrickPresentation } from "./brick-present";
import type { BrickSceneSetup } from "./brick-scene";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import type { BrickStagesFactory } from "./brick-stages";
import type { SceneDescription } from "../../core/model";
import type { GPURigidBodyPick, GPURigidBodyPose } from "../../core/webgpu-rigid-body";
import { BrickRigidBodies } from "./bodies-host";

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
}

export class BrickNBFlipSolver implements GPUSolverInstance {
  readonly brickEngine: BrickEngine;
  readonly brickPresentation: BrickPresentation;
  /** Lane P's rigid bodies (M5), when the scene has any. */
  readonly rigidBodies?: BrickRigidBodies;
  readonly fluidDomain: GPUSolverInstance["fluidDomain"];
  readonly denseLevelSetVolumeSource: DenseLevelSetVolumeConsumerSource;
  private presentEnabled: boolean;
  /** Presentation copies submitted, and the last copy's GPU completion time (ms, wall), apart from the steps. */
  presentations = 0;
  lastPresentationMs?: number;
  info: GPUEulerianInfo;

  private constructor(private readonly options: BrickSolverOptions, engine: BrickEngine, presentation: BrickPresentation, rigid?: BrickRigidBodies) {
    this.brickEngine = engine; this.brickPresentation = presentation; this.presentEnabled = options.present ?? true;
    if (rigid) this.rigidBodies = rigid;
    const s = options.setup, h = s.h_m;
    this.fluidDomain = { origin_m: s.origin_m, cellSize_m: [h, h, h], dimensions: s.lattice };
    this.denseLevelSetVolumeSource = { vertexPhi: presentation.vertexPhi, contourVertexPhi: true, openFraction: presentation.openFraction, cellSize_m: [h, h, h] };
    this.info = { ...this.makeInfo(), submittedTime_s: 0, completedTime_s: 0, encodedSteps: 0 };
  }

  static async create(device: GPUDevice, options: BrickSolverOptions): Promise<BrickNBFlipSolver> {
    const s = options.setup;
    const rigid = options.rigid?.rigidBodies?.length
      ? await BrickRigidBodies.create(device, options.rigid, { frame_m: s.origin_m, h_m: s.h_m, dt_s: s.dt_s, density_kg_m3: s.density_kg_m3, coupling: options.values.rigidCoupling !== "off" })
      : undefined;
    const engine = await BrickEngine.create(device, {
      lattice: s.lattice, ...(s.box ? { box: s.box } : {}), ...(s.openTop ? { openTop: true } : {}), farPhi: s.farPhi, h_m: s.h_m, dt_s: s.dt_s, gravity_mps2: s.gravity_mps2, capacities: s.capacities, liquid: s.liquid, initial: s.initial, initialVelocity: s.initialVelocity, ...(s.sources ? { sources: s.sources } : {}), ...(options.censusHold ? { censusHold: options.censusHold } : {}),
      solver: options.solver, stages: options.stages, values: options.values, label: s.sceneId,
      fluid: { density_kg_m3: s.density_kg_m3, viscosity_Pa_s: s.viscosity_Pa_s, surfaceTension_N_m: s.surfaceTension_N_m },
      particles: { bandDepth: s.particleBandDepth },
      ...(rigid ? { bodies: rigid } : {}),
    });
    const presentation = await BrickPresentation.create(engine, options.farPhi_m);
    const solver = new BrickNBFlipSolver(options, engine, presentation, rigid);
    solver.present();
    return solver;
  }

  get volumeTexture(): GPUTexture { return this.brickPresentation.volume; }
  get surfaceFieldTexture(): GPUTexture { return this.brickPresentation.volume; }
  get simulationReady(): boolean { return true; }
  get framePending(): boolean { return false; }
  get presentationPending(): boolean { return false; }

  /** Turn the per-advance presentation copy on or off (benches time the step alone). */
  setPresentationEnabled(enabled: boolean) { this.presentEnabled = enabled; }

  private present() {
    const d = this.brickEngine.device, enc = d.createCommandEncoder({ label: "nb-brick.present" });
    this.brickPresentation.encode(enc);
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
    this.rigidBodies?.offer(bodies);
    try { this.brickEngine.advance(1); } catch (error) {
      // A latched FAIL (lagged receipt) refuses further steps: report it, never step on.
      this.refresh({ simulationFailure: this.failure(error instanceof Error ? error.message : String(error)) });
      return false;
    }
    if (this.presentEnabled) this.present();
    this.refresh({ submittedTime_s: submitted + dt, encodedSteps: (this.info.encodedSteps ?? 0) + 1 });
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
      nx, ny, nz, storedNy: ny, cellCount: stored, equivalentUniformCells: cells, compressionRatio: stored > 0 ? cells / stored : 0,
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

  // The rigid contract (method-contract.ts), the shared system's.
  get rigidRenderBuffer(): GPUBuffer | undefined { return this.rigidBodies?.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer | undefined { return this.rigidBodies?.motionBuffer; }
  /** The roster for the next step's sync (the old host's syncRigidBodies; prescribed bodies are sampled there). */
  syncRigidBodies(bodies: readonly RigidBodyState[]): void { this.rigidBodies?.offer(bodies); }
  setSelectedRigidBody(index: number): void { this.rigidBodies?.system.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]): Promise<GPURigidBodyPick | undefined> { return this.rigidBodies ? this.rigidBodies.system.pick(origin, direction) : Promise.resolve(undefined); }
  readRigidBodyPoses(): Promise<GPURigidBodyPose[] | undefined> { return this.rigidBodies ? this.rigidBodies.readPoses() : Promise.resolve(undefined); }

  destroy() { this.brickPresentation.destroy(); this.brickEngine.destroy(); this.rigidBodies?.destroy(); }
}
