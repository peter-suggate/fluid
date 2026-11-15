/**
 * The brick engine as an app solver (GPUSolverInstance): the host and
 * contract adapter of design §5, lane E.
 *
 * - Construction: the scene adapter's setup (brick-scene.ts), the stages
 *   and pressure solver the method passes in, the engine, then the interim
 *   presentation.
 * - advanceTo: whole fixed steps of dt, submitted with no fence (no
 *   substeps); the presentation copy is encoded after the last step as its
 *   own submission, outside the step's.
 * - Health: receipts are lagged; assertSimulationHealthy fences and reads
 *   the header (FAIL is latched).
 * - Observers read `brickEngine` and `brickPresentation` after a fence
 *   (tools/nbflip-observe-brick.ts).
 */
import type { GPUSolverInstance } from "../../core/method-contract";
import type { GPUQuality } from "../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import type { RigidBodyState } from "../../core/rigid-body";
import { H, STATE } from "./brick-abi";
import { BrickEngine } from "./brick-engine";
import { BrickPresentation } from "./brick-present";
import type { BrickSceneSetup } from "./brick-scene";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import type { BrickStagesFactory } from "./brick-stages";

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
}

export class BrickNBFlipSolver implements GPUSolverInstance {
  readonly brickEngine: BrickEngine;
  readonly brickPresentation: BrickPresentation;
  readonly fluidDomain: GPUSolverInstance["fluidDomain"];
  readonly denseLevelSetVolumeSource: DenseLevelSetVolumeConsumerSource;
  private time_s = 0;
  private presentEnabled: boolean;
  /** Presentation copies submitted, and the last copy's GPU completion time (ms, wall), apart from the steps. */
  presentations = 0;
  lastPresentationMs?: number;
  info: GPUEulerianInfo;

  private constructor(private readonly options: BrickSolverOptions, engine: BrickEngine, presentation: BrickPresentation) {
    this.brickEngine = engine; this.brickPresentation = presentation; this.presentEnabled = options.present ?? true;
    const s = options.setup, h = s.h_m;
    this.fluidDomain = { origin_m: s.origin_m, cellSize_m: [h, h, h], dimensions: s.lattice };
    this.denseLevelSetVolumeSource = { vertexPhi: presentation.vertexPhi, contourVertexPhi: true, openFraction: presentation.openFraction, cellSize_m: [h, h, h] };
    this.info = this.makeInfo();
  }

  static async create(device: GPUDevice, options: BrickSolverOptions): Promise<BrickNBFlipSolver> {
    const s = options.setup;
    const engine = await BrickEngine.create(device, {
      lattice: s.lattice, h_m: s.h_m, dt_s: s.dt_s, gravity_mps2: s.gravity_mps2, capacities: s.capacities, liquid: s.liquid,
      solver: options.solver, stages: options.stages, values: options.values, label: s.sceneId,
      fluid: { density_kg_m3: s.density_kg_m3, viscosity_Pa_s: s.viscosity_Pa_s, surfaceTension_N_m: s.surfaceTension_N_m },
    });
    const presentation = await BrickPresentation.create(engine, options.farPhi_m);
    const solver = new BrickNBFlipSolver(options, engine, presentation);
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

  advanceTo(time_s: number, _bodies: RigidBodyState[]): boolean {
    const dt = this.options.setup.dt_s, steps = Math.floor((time_s - this.time_s) / dt + 1e-9);
    if (steps < 1) return false;
    this.brickEngine.advance(steps);
    this.time_s += steps * dt;
    if (this.presentEnabled) this.present();
    this.info = this.makeInfo();
    return true;
  }

  async awaitFrameCompletion() { await this.brickEngine.fence(); }

  async assertSimulationHealthy(completion?: Promise<void>) {
    if (completion) await completion;
    await this.brickEngine.assertHealthy();
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.brickEngine.fence();
    const header = await this.brickEngine.read("set", 0, 256);
    this.info = this.makeInfo(header);
    return this.info;
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
        step: header?.[H.STEP] ?? 0, presentations: this.presentations, lastPresentationMs: this.lastPresentationMs },
    };
  }

  destroy() { this.brickPresentation.destroy(); this.brickEngine.destroy(); }
}
