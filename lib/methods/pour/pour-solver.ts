/**
 * Pour solver: the GPUSolverInstance (docs/plans/pour-rewrite-2026-11-20.md §2), forked from the seep solver
 * (lib/methods/seep/seep-solver.ts).
 *
 * One step a 1/60 s, every dispatch over GPU-compacted brick lists (indirect) or an early-exit particle range;
 * no per-step readback. The nozzle's per-step words come from a schedule uploaded once (G2P publishes the next
 * step's). Statistics and health are read only on demand.
 */
import type { GPUSolverInstance } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import { integratedInflowVolume } from "../../core/inflow-boundary";
import { brickStaticMask } from "../narrow-band/brick-static.wgsl";
import { BRICKS, CELLS, CELL_M, DT, NX, NY, NZ, PHI_MAX, assertPourScene, bakePour, type PourBake } from "./pour-layout";
import { pourKernels, type KernelSource, type PourKernels } from "./pour-kernels";
import { ARGS_WORDS, B, BULK_WG, C, COARSE_CELLS, COARSEST_CELLS, FAIL, MAX_CYCLES, MG_STRIDE, NBULK, PARTICLE_CAPACITY, bindGroupLayoutEntries } from "./pour-wgsl";

export const POUR_METHOD_ID = "pour-m1max";
/** Mass-fit passes over the band (even: they ping-pong phiT -> phi -> phiT; the last unions the nozzle's plug). */
/** The surface is the brick engine's union (pour.combine); no mass fit runs. */
const MASS_FIT_PASSES = 0;

export interface PourSolverOptions { readonly onProgress?: (progress: { stage: string; fraction: number }) => void; readonly signal?: AbortSignal }

/** On-demand solver diagnostics (one readback). */
export interface PourDiagnostics {
  readonly step: number;
  readonly volumeCells: number;
  /** The nozzle's delivered volume to this step (analytic, cells) and the volume's drift against it. */
  readonly deliveredCells: number;
  readonly volumeDrift: number;
  readonly failures: string[];
  readonly particles: number;
  /** Particles seeded in the band (cumulative). */
  readonly seeded: number;
  readonly deleted: number;
  readonly deletedBy: { readonly deep: number; readonly cap: number; readonly out: number };
  readonly cycles: number;
  readonly cyclesTotal: number;
  readonly capHits: number;
  readonly firstResidual: number;
  readonly lastResidual: number;
  readonly maxSegments: number;
  readonly residualHistory: number[];
  readonly lists: { readonly active: number; readonly surf: number; readonly part: number; readonly clear: number; readonly liquid: number };
}

interface Dispatch { readonly label: string; readonly pipeline: GPUComputePipeline; readonly group: GPUBindGroup; readonly direct?: readonly [number, number?]; readonly indirect?: number }

/** Where a step's dispatch list is cut into separately submitted command buffers (see buildStep). Dawn translates
 *  a command buffer into Metal at its submit: with one cut at 4 the GPU idled through the translation of the other
 *  246 dispatches, and every step paid it. Four cuts (after the surface prefix, then through the PCG cycles, about 7
 *  of which run) overlap translation with execution: worst 25-step window (301-325, fenced) 18.2 to 16.6 ms; nine
 *  cuts did no better (17.0). */
const SUBMIT_CUTS = [4, 30, 80, 140];

export class PourSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  private readonly uTextures: GPUTexture[];
  private readonly uSampler: GPUSampler;
  readonly surfaceFieldTexture: GPUTexture;
  readonly framePending = false;
  private readonly buffers: Record<string, GPUBuffer> = {};
  private readonly uniforms = new Map<string, GPUBuffer>();
  private readonly layouts = new Map<KernelSource, GPUBindGroupLayout>();
  private readonly pipelines = new Map<KernelSource, GPUComputePipeline>();
  private readonly kernels: PourKernels;
  private steps: [Dispatch[], Dispatch[]] = [[], []];
  private volumeInit?: Dispatch;
  private volumeMeasure?: Dispatch;
  private volumeMeasureT?: Dispatch;
  private readonly readback: GPUBuffer;
  private readbackQueue: Promise<unknown> = Promise.resolve();
  private stepIndex = 0;
  private prebuilt: { parity: number; cbs: GPUCommandBuffer[] } | null = null;
  private destroyed = false;
  private failureText?: string;

  private constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly bake: PourBake) {
    this.kernels = pourKernels(bake.inflow);
    const buffer = (name: string, size: number, usage: number) => {
      this.buffers[name] = device.createBuffer({ label: `pour.${name}`, size, usage });
    };
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    buffer("ctrl", C.WORDS * 4, S);
    buffer("args", ARGS_WORDS * 4, S | GPUBufferUsage.INDIRECT);
    buffer("bricks", B.COUNT * BRICKS * 4, S);
    buffer("phis", 3 * CELLS * 4, S);
    buffer("vel", 9 * CELLS * 4, S);
    buffer("op", 2 * CELLS * 4, S);
    buffer("bits", 2 * CELLS * 4, S);
    buffer("pres", 6 * CELLS * 4, S);
    buffer("coarse", (7 * COARSE_CELLS + COARSEST_CELLS * COARSEST_CELLS) * 4, S);
    // NBULK = 0: the composite's buffer is bound but never read.
    if (NBULK !== 0) throw new Error("pour: a BULK region needs its 6 x COARSE_LATTICE buffer");
    buffer("bulk", 16, S);
    buffer("parts", 2 * PARTICLE_CAPACITY * 2 * 16, S);
    buffer("pcell", PARTICLE_CAPACITY * 4, S);
    buffer("pgrid", 3 * CELLS * 4, S);
    buffer("keys", CELLS * 4, S);
    buffer("acc", 6 * CELLS * 4, S);
    buffer("vsum", BRICKS * 4, S);
    this.readback = device.createBuffer({ label: "pour.readback", size: BRICKS * 4 + C.WORDS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const tex = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
    this.volumeTexture = device.createTexture({ label: "pour.fill", size: [NX, NY, NZ], dimension: "3d", format: "r32float", usage: tex });
    this.uTextures = [[NX + 2, NY, NZ], [NX, NY + 2, NZ], [NX, NY, NZ + 2]].map((size, a) =>
      device.createTexture({ label: `pour.u${a}`, size, dimension: "2d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    this.uSampler = device.createSampler({ label: "pour.uSampler", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.surfaceFieldTexture = device.createTexture({ label: "pour.surface", size: [NX + 1, NY + 1, NZ + 1], dimension: "3d", format: "r32float", usage: tex });
    const allocatedBytes = Object.values(this.buffers).reduce((s, b) => s + b.size, 0) + 4 * (NX * NY * NZ + (NX + 1) * (NY + 1) * (NZ + 1))
      + 4 * ((NX + 2) * NY * NZ + NX * (NY + 2) * NZ + NX * NY * (NZ + 2));
    this.info = {
      nx: NX, ny: NY, nz: NZ, storedNy: NY, cellCount: CELLS, equivalentUniformCells: CELLS, compressionRatio: 1,
      regularLayers: 0, maximumNeighborDelta: 0, gridKind: "uniform", cellSize_m: CELL_M, pressureIterations: MAX_CYCLES,
      allocatedBytes, quality: "balanced", submittedTime_s: 0, completedTime_s: 0, encodedSteps: 0, simulationPipelinesReady: false,
    };
  }

  static async create(device: GPUDevice, scene: SceneDescription, options: PourSolverOptions = {}): Promise<PourSolver> {
    options.onProgress?.({ stage: "bake", fraction: 0 });
    const mask = brickStaticMask(scene);
    if (!mask) throw new Error(`${POUR_METHOD_ID}: the scene has no static voxel mask (the glass)`);
    assertPourScene(scene, (x, y, z) => mask.solid(x, y, z));
    const bake = bakePour(scene);
    const solver = new PourSolver(device, scene, bake);
    try {
      options.onProgress?.({ stage: "compile", fraction: 0.2 });
      await solver.compile();
      options.signal?.throwIfAborted();
      options.onProgress?.({ stage: "upload", fraction: 0.8 });
      solver.upload();
      solver.record();
      await solver.measure(true);
      solver.info.initialVolumeCellSum = 0;
      solver.info.volumeCellSum = 0;
      solver.info.volumeDrift = 0;
      solver.info.simulationPipelinesReady = true;
      options.onProgress?.({ stage: "ready", fraction: 1 });
      return solver;
    } catch (error) {
      solver.destroy();
      throw error;
    }
  }

  private async compile(): Promise<void> {
    const all = Object.values(this.kernels) as KernelSource[];
    const modules = all.map(k => this.device.createShaderModule({ label: k.label, code: k.code }));
    const errors = (await Promise.all(modules.map(async (m, i) => (await m.getCompilationInfo()).messages
      .filter(x => x.type === "error").map(x => `${all[i]!.label} ${x.lineNum}:${x.linePos} ${x.message}`)))).flat();
    if (errors.length) throw new Error(`Pour WGSL:\n${errors.join("\n")}`);
    await Promise.all(all.map(async (k, i) => {
      const layout = this.device.createBindGroupLayout({ label: k.label, entries: bindGroupLayoutEntries(k.bindings) });
      this.layouts.set(k, layout);
      const pipeline = await this.device.createComputePipelineAsync({
        label: k.label, layout: this.device.createPipelineLayout({ label: k.label, bindGroupLayouts: [layout] }), compute: { module: modules[i]!, entryPoint: k.entry },
      });
      this.pipelines.set(k, pipeline);
    }));
  }

  private group(k: KernelSource, cfg: readonly number[] = [0, 0, 0, 0]): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = k.bindings.map((spec, binding) => {
      if (spec.kind === "uniform") {
        const key = cfg.join(",");
        let u = this.uniforms.get(key);
        if (!u) {
          u = this.device.createBuffer({ label: `pour.cfg(${key})`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.device.queue.writeBuffer(u, 0, new Uint32Array([cfg[0] ?? 0, cfg[1] ?? 0, cfg[2] ?? 0, cfg[3] ?? 0]));
          this.uniforms.set(key, u);
        }
        return { binding, resource: { buffer: u } };
      }
      if (spec.kind === "layers" || spec.kind === "layersW") return { binding, resource: this.uTextures[Number(spec.name[2])]!.createView({ dimension: "2d-array" }) };
      if (spec.kind === "sampler") return { binding, resource: this.uSampler };
      if (spec.kind === "volume") return { binding, resource: (spec.name === "surf" ? this.surfaceFieldTexture : this.volumeTexture).createView() };
      const buffer = this.buffers[spec.name];
      if (!buffer) throw new Error(`pour: no buffer ${spec.name}`);
      return { binding, resource: { buffer } };
    });
    return this.device.createBindGroup({ label: k.label, layout: this.layouts.get(k)!, entries });
  }

  /** The empty box: air everywhere (phi = +PHI_MAX), at rest, zero pressure; step 1's nozzle words published. */
  private upload(): void {
    const q = this.device.queue, b = this.buffers, { inflow } = this.bake;
    const bricks = new Uint32Array(B.COUNT * BRICKS);
    bricks.set(this.bake.brickClass, B.CLASS * BRICKS);
    new Float32Array(bricks.buffer, B.MINPHI * BRICKS * 4, BRICKS).fill(PHI_MAX);
    bricks.set(inflow.scheduleWords, B.INFLOW * BRICKS);
    bricks.set(inflow.scheduleWords.subarray(0, 3), B.INFLOW_NOW * BRICKS);
    q.writeBuffer(b.bricks, 0, bricks);
    const args = new Uint32Array(ARGS_WORDS);
    args.set([BULK_WG, 1, 1], C.ARG_ACTB); args.set([Math.ceil(NBULK / 8), 1, 1], C.ARG_ACTBB);
    args.set([0, 1, 1], C.ARG_LIQ); args.set([BULK_WG, 1, 1], C.ARG_LIQB); args.set([NBULK, 1, 1], C.ARG_LIQBB);
    q.writeBuffer(b.args, 0, args);
    // Step 1 reads schedule row 0; G2P advances the counter (stamps start at 1, so the zeroed flag tables list
    // every block on step 1).
    const ctrl = new Uint32Array(C.WORDS);
    ctrl[C.STEP] = 1;
    q.writeBuffer(b.ctrl, 0, ctrl);
    const air = new Float32Array(3 * CELLS).fill(PHI_MAX);
    q.writeBuffer(b.phis, 0, air);
  }

  private record(): void {
    const N = CELLS, K = this.kernels;
    const P = this.pipelines;
    const d = (k: KernelSource, cfg: number[] | undefined, launch: { direct?: [number, number?]; indirect?: number }, label = k.label): Dispatch =>
      ({ label, pipeline: P.get(k)!, group: this.group(k, cfg), ...launch });
    const capGroups = Math.ceil(PARTICLE_CAPACITY / 256);
    const W = (word: number) => word * 4;
    for (const parity of [0, 1] as const) {
      const p = [parity, 0, 0, 0];
      const list: Dispatch[] = [
        d(K.classify, p, { direct: [Math.ceil(BRICKS / 64)] }),
        d(K.clear, undefined, { indirect: W(C.ARG_CLEAR) }),
        d(K.publishU, [1, 0, 0, 0], { indirect: W(C.ARG_CLEAR) }, "pour.publishU.clear"),
        d(K.advectPhi, undefined, { indirect: W(C.ARG_SURF) }),
        d(K.count, p, { direct: [capGroups] }),
        d(K.seed, p, { indirect: W(C.ARG_PART) }),
        d(K.scan, p, { indirect: W(C.ARG_PART) }),
        d(K.scatter, p, { direct: [capGroups] }),
        d(K.splat, p, { indirect: W(C.ARG_PART) }),
        d(K.combine, undefined, { indirect: W(C.ARG_PART) }),
        ...Array.from({ length: MASS_FIT_PASSES }, (_, k) => k).map(k =>
          d(K.massFit, [k % 2 === 0 ? N : 0, k % 2 === 0 ? 0 : N, k === MASS_FIT_PASSES - 1 ? 1 : 0, 0], { indirect: W(C.ARG_PART) }, `pour.massFit.${k}`)),
        d(K.redistInit, undefined, { indirect: W(C.ARG_SURF) }),
        d(K.redistSweep, [0, N, 0, 0], { indirect: W(C.ARG_SURF) }, "pour.redistSweep.1"),
        d(K.redistSweep, [N, 0, 1, 0], { indirect: W(C.ARG_SURF) }, "pour.redistSweep.2"),
        d(K.separate, p, { direct: [capGroups] }),
        d(K.p2g, p, { indirect: W(C.ARG_PART) }),
        d(K.velocity, undefined, { indirect: W(C.ARG_ACTIVE) }),
        d(K.velocityTrace, undefined, { indirect: W(C.ARG_TRACE) }),
        d(K.operator, undefined, { indirect: W(C.ARG_ACTB) }),
        d(K.coarsenBrick, undefined, { indirect: W(C.ARG_ACTBB) }),
        d(K.coarsenTop, undefined, { direct: [1] }),
      ];
      list.push(d(K.pcgGuess, undefined, { indirect: W(C.ARG_LIQB) }));
      for (let k = 0; k <= MAX_CYCLES; k++) {
        const mg = (row: number, slot: number) => W(C.MG_ARGS + MG_STRIDE * row + 4 * slot);
        const kc = [k, 0, 0, 0];
        const prev = (slot: 0 | 1 | 2 | 3 | 4 | 5) => k === 0 ? W([C.ARG_LIQ, C.ARG_LIQBB, C.ARG_LIQB, C.ARG_L4, C.ARG_L4U, C.ARG_SHIFT][slot]) : mg(k - 1, slot);
        list.push(d(K.pcgUpdate, kc, { indirect: prev(2) }, `pour.pcgUpdate.${k}`));
        list.push(d(K.pcgCheck, kc, { direct: [1] }, `pour.pcgCheck.${k}`));
        if (k < MAX_CYCLES) {
          list.push(
            d(K.pcgShiftDown, kc, { indirect: mg(k, 5) }, `pour.pcgShiftDown.${k}`),
            d(K.mgDown0, undefined, { indirect: mg(k, 0) }, `pour.mgDown0.${k}`),
            d(K.mgDown1, undefined, { indirect: mg(k, 1) }, `pour.mgDown1.${k}`),
            d(K.mgDown3, undefined, { indirect: mg(k, 4) }, `pour.mgDown3.${k}`),
            d(K.mgRestrict3, undefined, { indirect: mg(k, 3) }, `pour.mgRestrict3.${k}`),
            d(K.coarseSolve, undefined, { direct: [1] }, `pour.coarseSolve.${k}`),
            d(K.mgUp3, undefined, { indirect: mg(k, 3) }, `pour.mgUp3.${k}`),
            d(K.mgUp2, undefined, { indirect: mg(k, 1) }, `pour.mgUp2.${k}`),
            d(K.mgUp1, undefined, { indirect: mg(k, 0) }, `pour.mgUp1.${k}`),
            d(K.pcgShiftUp, kc, { indirect: mg(k, 5) }, `pour.pcgShiftUp.${k}`),
            d(K.pcgPrecond, kc, { indirect: mg(k, 2) }, `pour.pcgPrecond.${k}`),
            d(K.pcgDirection, kc, { indirect: mg(k, 2) }, `pour.pcgDirection.${k}`));
        }
      }
      list.push(
        d(K.project, undefined, { indirect: W(C.ARG_ACTB) }),
        d(K.extend, [0, 6 * N, 0, 3], { indirect: W(C.ARG_SURF) }, "pour.extend.1"),
        d(K.extend, [6 * N, 0, 3, 0], { indirect: W(C.ARG_SURF) }, "pour.extend.2"),
        d(K.publishU, [0, 0, 0, 0], { indirect: W(C.ARG_ACTIVE) }),
        d(K.g2p, p, { direct: [capGroups] }),
      );
      this.steps[parity] = list;
    }
    this.volumeInit = d(K.volume, [1, 0, 0, 0], { direct: [128, BRICKS / 128] }, "pour.volume.init");
    this.volumeMeasure = d(K.volume, [0, 0, 0, 0], { direct: [128, BRICKS / 128] });
    this.volumeMeasureT = d(K.volume, [0, 1, 0, 0], { direct: [128, BRICKS / 128] }, "pour.volume.phiT");
  }

  private encode(pass: GPUComputePassEncoder, x: Dispatch): void {
    pass.setPipeline(x.pipeline);
    pass.setBindGroup(0, x.group);
    if (x.indirect !== undefined) pass.dispatchWorkgroupsIndirect(this.buffers.args!, x.indirect);
    else pass.dispatchWorkgroups(x.direct![0], x.direct![1] ?? 1);
  }

  /** Encode one step into `encoder` (one compute pass). */
  encodeStep(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "pour.step" });
    for (const x of this.steps[this.stepIndex & 1]) this.encode(pass, x);
    pass.end();
    this.stepIndex++;
  }

  /** A step's command buffers depend only on its parity; split at SUBMIT_CUTS so the GPU runs each part while
   *  Dawn translates the next one at its submit (seep-solver.ts buildStep). */
  private buildStep(parity: number): GPUCommandBuffer[] {
    const list = this.steps[parity & 1];
    const parts: Dispatch[][] = []; let at = 0;
    for (const c of [...SUBMIT_CUTS, list.length]) { parts.push(list.slice(at, c)); at = c; }
    return parts.map(part => {
      const encoder = this.device.createCommandEncoder({ label: "pour.step" });
      const pass = encoder.beginComputePass({ label: "pour.step" });
      for (const x of part) this.encode(pass, x);
      pass.end();
      return encoder.finish();
    });
  }

  advanceTo(time_s: number): boolean {
    if (this.destroyed) return false;
    const submitted = this.info.submittedTime_s ?? 0;
    if (time_s + 1e-9 < submitted + DT) return false;
    const parity = this.stepIndex & 1;
    const cbs = this.prebuilt?.parity === parity ? this.prebuilt.cbs : this.buildStep(parity);
    this.prebuilt = null;
    for (const cb of cbs) this.device.queue.submit([cb]);
    this.stepIndex++;
    this.info.submittedTime_s = submitted + DT;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.prebuilt = { parity: this.stepIndex & 1, cbs: this.buildStep(this.stepIndex & 1) };
    return true;
  }

  async awaitFrameCompletion(): Promise<void> {
    const t = this.info.submittedTime_s;
    await this.device.queue.onSubmittedWorkDone();
    this.info.completedTime_s = t;
  }

  private measure(publish: boolean, phiT = false): Promise<{ volume: number; ctrl: Uint32Array; sums: Float32Array }> {
    const run = this.readbackQueue.then(() => this.measureNow(publish, phiT));
    this.readbackQueue = run.catch(() => undefined);
    return run;
  }

  private async measureNow(publish: boolean, phiT: boolean): Promise<{ volume: number; ctrl: Uint32Array; sums: Float32Array }> {
    const encoder = this.device.createCommandEncoder({ label: "pour.measure" });
    const pass = encoder.beginComputePass({ label: "pour.measure" });
    this.encode(pass, publish ? this.volumeInit! : phiT ? this.volumeMeasureT! : this.volumeMeasure!);
    pass.end();
    encoder.copyBufferToBuffer(this.buffers.vsum!, 0, this.readback, 0, BRICKS * 4);
    encoder.copyBufferToBuffer(this.buffers.ctrl!, 0, this.readback, BRICKS * 4, C.WORDS * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const raw = this.readback.getMappedRange().slice(0);
    this.readback.unmap();
    const sums = new Float32Array(raw, 0, BRICKS);
    let volume = 0;
    for (let i = 0; i < BRICKS; i++) volume += sums[i]!;
    return { volume, ctrl: new Uint32Array(raw, BRICKS * 4, C.WORDS), sums };
  }

  async readDiagnostics(): Promise<PourDiagnostics> {
    const { volume, ctrl } = await this.measure(false);
    const f = new Float32Array(ctrl.buffer, ctrl.byteOffset, ctrl.length);
    const failBits = ctrl[C.FAIL]!;
    const failures = Object.entries(FAIL).filter(([, bit]) => (failBits & bit) !== 0).map(([name]) => name);
    const step = ctrl[C.STEP]! - 1;
    const delivered = integratedInflowVolume(this.scene.fluid.inflow!, 0, step * DT) / CELL_M ** 3;
    const drift = delivered > 0 ? volume / delivered - 1 : 0;
    this.info.volumeCellSum = volume;
    this.info.initialVolumeCellSum = delivered;
    this.info.volumeDrift = drift;
    this.info.pressureIterationsExecuted = ctrl[C.CYCLES];
    if (failures.length && !this.failureText) {
      this.failureText = `Pour solver latched ${failures.join(", ")} at step ${step} (max backtrace segments ${ctrl[C.MAX_SEG]}, pool overflow ${ctrl[C.OVERFLOW]}, capped steps ${ctrl[C.CAP_HITS]})`;
      this.info.simulationPipelineError = this.failureText;
    }
    return {
      step, volumeCells: volume, deliveredCells: delivered, volumeDrift: drift, failures,
      particles: ctrl[C.PCOUNT]! + ctrl[C.PCOUNT + 1]!, seeded: ctrl[C.EMITTED]!, deleted: ctrl[C.DELETED]!,
      deletedBy: { deep: ctrl[C.DEL_DEEP]!, cap: ctrl[C.DEL_CAP]!, out: ctrl[C.DEL_OUT]! },
      cycles: ctrl[C.CYCLES]!, cyclesTotal: ctrl[C.CYCLES_TOTAL]!, capHits: ctrl[C.CAP_HITS]!,
      firstResidual: f[C.FIRST_RES]!, lastResidual: f[C.LAST_RES]!, maxSegments: ctrl[C.MAX_SEG]!,
      residualHistory: Array.from(f.subarray(C.RES_HIST, C.RES_HIST + MAX_CYCLES + 1)),
      lists: { active: ctrl[C.N_ACTIVE]!, surf: ctrl[C.N_ACTIVE + 1]!, part: ctrl[C.N_ACTIVE + 2]!, clear: ctrl[C.N_ACTIVE + 3]!, liquid: ctrl[C.N_LIQ]! },
    };
  }

  /** Development aid: run the next step in segments, measuring after each the fill volume of the field that stage
   *  wrote (phiT after advection and combination, phi otherwise) and the particle census. Completes the step. */
  async debugLedgerStep(
    stops: readonly (readonly [string, boolean])[] = [["pour.advectPhi", true], ["pour.scatter", true], ["pour.combine", true], ...(MASS_FIT_PASSES ? [[`pour.massFit.${MASS_FIT_PASSES - 1}`, true] as const] : []), ["pour.redistSweep.2", false], ["pour.g2p", false]],
    onStop?: (label: string) => Promise<void>,
  ): Promise<{ stage: string; total: number; particles: number; seeded: number; deleted: { deep: number; cap: number; out: number }; bricks: Float32Array }[]> {
    const parity = this.stepIndex & 1;
    const list = this.steps[parity];
    if (stops[stops.length - 1]?.[0] !== "pour.g2p") throw new Error("pour ledger: the last stop must be pour.g2p");
    const rows = [];
    let begin = 0;
    for (const [label, phiT] of stops) {
      const end = list.findIndex(x => x.label === label);
      if (end < begin) throw new Error(`pour ledger: no dispatch ${label} after ${begin}`);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const x of list.slice(begin, end + 1)) this.encode(pass, x);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      begin = end + 1;
      if (onStop) await onStop(label);
      const m = await this.measure(false, phiT);
      rows.push({ stage: label.replace("pour.", ""), total: m.volume, particles: m.ctrl[C.PCOUNT + 1 - parity]!, seeded: m.ctrl[C.EMITTED]!,
        deleted: { deep: m.ctrl[C.DEL_DEEP]!, cap: m.ctrl[C.DEL_CAP]!, out: m.ctrl[C.DEL_OUT]! }, bricks: m.sums });
    }
    this.stepIndex++;
    this.info.submittedTime_s = (this.info.submittedTime_s ?? 0) + DT;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    return rows;
  }

  /** Development aid (tests and tools only): copy a slice of an internal buffer. */
  async debugRead(name: string, byteOffset = 0, byteSize?: number): Promise<ArrayBuffer> {
    const src = this.buffers[name];
    if (!src) throw new Error(`pour: no buffer ${name}`);
    const size = byteSize ?? src.size - byteOffset;
    const dst = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, byteOffset, dst, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = dst.getMappedRange().slice(0);
    dst.unmap(); dst.destroy();
    return out;
  }

  private pendingStats?: Promise<GPUEulerianInfo>;

  readStats(): Promise<GPUEulerianInfo> {
    this.pendingStats ??= this.readStatsNow().finally(() => { this.pendingStats = undefined; });
    return this.pendingStats;
  }

  private async readStatsNow(): Promise<GPUEulerianInfo> {
    const d = await this.readDiagnostics();
    return Object.assign({ ...this.info, volumeCellSum: d.volumeCells, initialVolumeCellSum: d.deliveredCells, volumeDrift: d.volumeDrift, maxDivergence_s: d.lastResidual / CELL_M }, { particles: d.particles, pour: d });
  }

  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    if (completion) await completion;
    const d = await this.readDiagnostics();
    if (d.failures.length) throw new Error(this.failureText);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.prebuilt = null;
    for (const b of Object.values(this.buffers)) b.destroy();
    for (const u of this.uniforms.values()) u.destroy();
    this.readback.destroy();
    this.volumeTexture.destroy();
    this.surfaceFieldTexture.destroy();
  }
}

export async function createPourSolver(device: GPUDevice, scene: SceneDescription, options: PourSolverOptions = {}): Promise<PourSolver> {
  return PourSolver.create(device, scene, options);
}
