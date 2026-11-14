/**
 * Seep solver: the GPUSolverInstance (lane S, docs/plans/seep-rewrite-2026-11-14.md §2).
 *
 * One compute pass per 1/60 s step, every dispatch over GPU-compacted brick
 * lists (indirect) or an early-exit particle range; no per-step readback.
 * Statistics and health are read only on demand.
 */
import type { GPUSolverInstance } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import {
  BRICKS, CELLS, CELL_M, DT, NX, NY, NZ, PARTICLE_CELLS, RESERVOIR_TOP, assertSeepScene, bakeSeep, cellIndex, cellOpen,
} from "./seep-layout";
import {
  ALL_KERNELS, advectPhiKernel, classifyKernel, clearKernel, coarseSolveKernel, coarsenBrickKernel, coarsenTopKernel, combineKernel,
  countKernel, extendKernel, g2pKernel, operatorKernel, p2gKernel, mgDown0Kernel, mgDown1Kernel, mgDown2Kernel, mgUp1Kernel, mgUp2Kernel, pcgDirectionKernel, pcgPrecondKernel, pcgUpdateKernel, projectKernel,
  redistInitKernel, redistSweepKernel, scanKernel, scatterKernel, seedKernel, velocityKernel, volumeKernel, type KernelSource,
} from "./seep-kernels";
import type { SeepObservationSource } from "./seep-observation";
import { ARGS_WORDS, B, C, COARSE_CELLS, FAIL, MAX_CYCLES, MG_STRIDE, PARTICLE_CAPACITY, bindGroupLayoutEntries } from "./seep-wgsl";

export const SEEP_METHOD_ID = "seep-m1max";

export interface SeepSolverOptions { readonly onProgress?: (progress: { stage: string; fraction: number }) => void; readonly signal?: AbortSignal }

/** On-demand solver diagnostics (one readback). */
export interface SeepDiagnostics {
  readonly step: number;
  readonly volumeCells: number;
  readonly initialVolumeCells: number;
  readonly volumeDrift: number;
  readonly failures: string[];
  readonly particles: number;
  readonly seeded: number;
  readonly deleted: number;
  readonly cycles: number;
  readonly cyclesTotal: number;
  readonly capHits: number;
  readonly firstResidual: number;
  readonly lastResidual: number;
  readonly maxSegments: number;
  readonly residualHistory: number[];
  readonly lists: { readonly active: number; readonly surf: number; readonly part: number; readonly clear: number };
}

interface Dispatch { readonly label: string; readonly pipeline: GPUComputePipeline; readonly group: GPUBindGroup; readonly direct?: readonly [number, number?]; readonly indirect?: number }

export class SeepSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly observation: SeepObservationSource;
  readonly observationSource: SeepObservationSource;
  readonly framePending = false;
  private readonly buffers: Record<string, GPUBuffer> = {};
  private readonly uniforms = new Map<string, GPUBuffer>();
  private readonly layouts = new Map<KernelSource, GPUBindGroupLayout>();
  private readonly pipelines = new Map<KernelSource, GPUComputePipeline>();
  private steps: [Dispatch[], Dispatch[]] = [[], []];
  private volumeInit?: Dispatch;
  private volumeMeasure?: Dispatch;
  private readonly readback: GPUBuffer;
  private initialVolumeCells = 0;
  private stepIndex = 0;
  private destroyed = false;
  private failureText?: string;

  private constructor(private readonly device: GPUDevice) {
    const buffer = (name: string, size: number, usage: number) => {
      this.buffers[name] = device.createBuffer({ label: `seep.${name}`, size, usage });
    };
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    buffer("ctrl", C.WORDS * 4, S | GPUBufferUsage.COPY_SRC);
    buffer("args", ARGS_WORDS * 4, S | GPUBufferUsage.INDIRECT);
    buffer("bricks", B.COUNT * BRICKS * 4, S | GPUBufferUsage.COPY_SRC);
    buffer("phis", 3 * CELLS * 4, S);
    buffer("vel", 9 * CELLS * 4, S);
    buffer("op", 2 * CELLS * 4, S);
    buffer("bits", 2 * CELLS * 4, S);
    buffer("pres", 6 * CELLS * 4, S);
    buffer("coarse", 7 * COARSE_CELLS * 4, S);
    buffer("parts", 2 * PARTICLE_CAPACITY * 2 * 16, S);
    buffer("pcell", PARTICLE_CAPACITY * 4, S);
    buffer("pgrid", 2 * PARTICLE_CELLS * 4, S);
    buffer("acc", 6 * PARTICLE_CELLS * 4, S);
    buffer("vsum", BRICKS * 4, S | GPUBufferUsage.COPY_SRC);
    this.readback = device.createBuffer({ label: "seep.readback", size: BRICKS * 4 + C.WORDS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const tex = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
    this.volumeTexture = device.createTexture({ label: "seep.fill", size: [NX, NY, NZ], dimension: "3d", format: "r32float", usage: tex });
    this.surfaceFieldTexture = device.createTexture({ label: "seep.surface", size: [NX + 1, NY + 1, NZ + 1], dimension: "3d", format: "r32float", usage: tex });
    const allocatedBytes = Object.values(this.buffers).reduce((s, b) => s + b.size, 0) + 4 * (NX * NY * NZ + (NX + 1) * (NY + 1) * (NZ + 1));
    this.info = {
      nx: NX, ny: NY, nz: NZ, storedNy: NY, cellCount: CELLS, equivalentUniformCells: CELLS, compressionRatio: 1,
      regularLayers: 0, maximumNeighborDelta: 0, gridKind: "uniform", cellSize_m: CELL_M, pressureIterations: MAX_CYCLES,
      allocatedBytes, quality: "balanced", submittedTime_s: 0, completedTime_s: 0, encodedSteps: 0, simulationPipelinesReady: false,
    };
    const b = this.buffers;
    const source: SeepObservationSource = {
      dimensions: [NX, NY, NZ],
      cellSize_m: CELL_M,
      buffers: [
        { name: "seepObs_phis", wgslType: "array<f32>", buffer: b.phis },
        { name: "seepObs_vel", wgslType: "array<f32>", buffer: b.vel },
        { name: "seepObs_ctrl", wgslType: "array<u32>", buffer: b.ctrl },
      ],
      declarations: (group: number) => source.buffers.map((x, i) => `@group(${group}) @binding(${i}) var<storage, read> ${x.name}: ${x.wgslType};`).join("\n"),
      wgsl: OBSERVATION_WGSL,
    };
    this.observation = source;
    this.observationSource = source;
  }

  static async create(device: GPUDevice, scene: SceneDescription, options: SeepSolverOptions = {}): Promise<SeepSolver> {
    assertSeepScene(scene);
    options.onProgress?.({ stage: "bake", fraction: 0 });
    const bake = bakeSeep();
    const solver = new SeepSolver(device);
    try {
      options.onProgress?.({ stage: "compile", fraction: 0.2 });
      await solver.compile();
      options.signal?.throwIfAborted();
      options.onProgress?.({ stage: "upload", fraction: 0.8 });
      solver.upload(bake);
      solver.record();
      solver.initialVolumeCells = await solver.measureVolume(true);
      solver.info.initialVolumeCellSum = solver.initialVolumeCells;
      solver.info.volumeCellSum = solver.initialVolumeCells;
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
    const modules = ALL_KERNELS.map(k => this.device.createShaderModule({ label: k.label, code: k.code }));
    const errors = (await Promise.all(modules.map(async (m, i) => (await m.getCompilationInfo()).messages
      .filter(x => x.type === "error").map(x => `${ALL_KERNELS[i].label} ${x.lineNum}:${x.linePos} ${x.message}`)))).flat();
    if (errors.length) throw new Error(`Seep WGSL:\n${errors.join("\n")}`);
    await Promise.all(ALL_KERNELS.map(async (k, i) => {
      const layout = this.device.createBindGroupLayout({ label: k.label, entries: bindGroupLayoutEntries(k.bindings) });
      this.layouts.set(k, layout);
      const pipeline = await this.device.createComputePipelineAsync({
        label: k.label, layout: this.device.createPipelineLayout({ label: k.label, bindGroupLayouts: [layout] }), compute: { module: modules[i], entryPoint: k.entry },
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
          u = this.device.createBuffer({ label: `seep.cfg(${key})`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.device.queue.writeBuffer(u, 0, new Uint32Array([cfg[0] ?? 0, cfg[1] ?? 0, cfg[2] ?? 0, cfg[3] ?? 0]));
          this.uniforms.set(key, u);
        }
        return { binding, resource: { buffer: u } };
      }
      if (spec.kind === "volume") return { binding, resource: (spec.name === "surf" ? this.surfaceFieldTexture : this.volumeTexture).createView() };
      const buffer = this.buffers[spec.name];
      if (!buffer) throw new Error(`seep: no buffer ${spec.name}`);
      return { binding, resource: { buffer } };
    });
    return this.device.createBindGroup({ label: k.label, layout: this.layouts.get(k)!, entries });
  }

  private upload(bake: ReturnType<typeof bakeSeep>): void {
    const q = this.device.queue, b = this.buffers;
    const bricks = new Uint32Array(B.COUNT * BRICKS);
    bricks.set(bake.brickClass, B.CLASS * BRICKS);
    const minPhi = new Float32Array(bricks.buffer, B.MINPHI * BRICKS * 4, BRICKS);
    for (let k = 0; k < BRICKS; k++) {
      let m = Infinity;
      for (let i = 0; i < 512; i++) m = Math.min(m, bake.phi[(k << 9) | i]);
      minPhi[k] = m;
    }
    q.writeBuffer(b.bricks, 0, bricks);
    q.writeBuffer(b.phis, 0, bake.phi);
    q.writeBuffer(b.phis, CELLS * 4, bake.phi);
    q.writeBuffer(b.phis, 2 * CELLS * 4, bake.phiCell);
    // Hydrostatic start: P = g dt (y_s - y_c) in velocity units, so that the
    // first step's gravity on resting liquid is balanced exactly (ghost fluid
    // at the flat surface y_s = 192).
    const pres = new Float32Array(CELLS);
    const gdt = 9.80665 * DT;
    for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
      const i = cellIndex(x, y, z);
      if (cellOpen(x, y, z) && bake.phiCell[i] < 0) pres[i] = gdt * Math.max(0, RESERVOIR_TOP - y - 0.5);
    }
    q.writeBuffer(b.pres, 0, pres);
  }

  private record(): void {
    const N = CELLS;
    const P = this.pipelines;
    const d = (k: KernelSource, cfg: number[] | undefined, launch: { direct?: [number, number?]; indirect?: number }, label = k.label): Dispatch =>
      ({ label, pipeline: P.get(k)!, group: this.group(k, cfg), ...launch });
    const capGroups = Math.ceil(PARTICLE_CAPACITY / 256);
    const W = (word: number) => word * 4;
    for (const parity of [0, 1] as const) {
      const p = [parity, 0, 0, 0];
      const list: Dispatch[] = [
        d(classifyKernel, undefined, { direct: [Math.ceil(BRICKS / 64)] }),
        d(clearKernel, undefined, { indirect: W(C.ARG_CLEAR) }),
        d(countKernel, p, { direct: [capGroups] }),
        d(seedKernel, p, { indirect: W(C.ARG_PART) }),
        d(scanKernel, p, { indirect: W(C.ARG_PART) }),
        d(scatterKernel, p, { direct: [capGroups] }),
        d(advectPhiKernel, undefined, { indirect: W(C.ARG_SURF) }),
        d(combineKernel, p, { indirect: W(C.ARG_PART) }),
        d(redistInitKernel, undefined, { indirect: W(C.ARG_SURF) }),
        d(redistSweepKernel, [0, N, 0, 0], { indirect: W(C.ARG_SURF) }, "seep.redistSweep.1"),
        d(redistSweepKernel, [N, 0, 1, 0], { indirect: W(C.ARG_SURF) }, "seep.redistSweep.2"),
        d(p2gKernel, p, { indirect: W(C.ARG_PART) }),
        d(velocityKernel, undefined, { indirect: W(C.ARG_ACTIVE) }),
        d(operatorKernel, undefined, { indirect: W(C.ARG_ACTIVE) }),
        d(coarsenBrickKernel, undefined, { indirect: W(C.ARG_ACTIVE) }),
        d(coarsenTopKernel, undefined, { direct: [1] }),
      ];
      for (let k = 0; k <= MAX_CYCLES; k++) {
        const mg = (slot: number) => W(C.MG_ARGS + MG_STRIDE * k + 4 * slot);
        const kc = [k, 0, 0, 0];
        // X_k and D0..D2_k run under C_(k-1)'s gate (every slot of a row carries the same count).
        const prev = k === 0 ? W(C.ARG_ACTIVE) : W(C.MG_ARGS + MG_STRIDE * (k - 1) + 28);
        list.push(d(pcgUpdateKernel, kc, { indirect: prev }, `seep.pcgUpdate.${k}`));
        list.push(d(coarseSolveKernel, kc, { direct: [1] }, `seep.coarseSolve.${k}`));
        if (k < MAX_CYCLES) {
          list.splice(list.length - 1, 0,
            d(mgDown0Kernel, undefined, { indirect: prev }, `seep.mgDown0.${k}`),
            d(mgDown1Kernel, undefined, { indirect: prev }, `seep.mgDown1.${k}`),
            d(mgDown2Kernel, undefined, { indirect: prev }, `seep.mgDown2.${k}`));
          list.push(
            d(mgUp2Kernel, undefined, { indirect: mg(3) }, `seep.mgUp2.${k}`),
            d(mgUp1Kernel, undefined, { indirect: mg(4) }, `seep.mgUp1.${k}`),
            d(pcgPrecondKernel, kc, { indirect: mg(5) }, `seep.pcgPrecond.${k}`),
            d(pcgDirectionKernel, kc, { indirect: mg(6) }, `seep.pcgDirection.${k}`));
        }
      }
      list.push(
        d(projectKernel, undefined, { indirect: W(C.ARG_ACTIVE) }),
        d(extendKernel, [0, 6 * N, 0, 3], { indirect: W(C.ARG_SURF) }, "seep.extend.1"),
        d(extendKernel, [6 * N, 0, 3, 0], { indirect: W(C.ARG_SURF) }, "seep.extend.2"),
        d(g2pKernel, p, { direct: [capGroups] }),
      );
      this.steps[parity] = list;
    }
    this.volumeInit = d(volumeKernel, [1, 0, 0, 0], { direct: [128, BRICKS / 128] }, "seep.volume.init");
    this.volumeMeasure = d(volumeKernel, [0, 0, 0, 0], { direct: [128, BRICKS / 128] });
  }

  private encode(pass: GPUComputePassEncoder, x: Dispatch): void {
    pass.setPipeline(x.pipeline);
    pass.setBindGroup(0, x.group);
    if (x.indirect !== undefined) pass.dispatchWorkgroupsIndirect(this.buffers.args, x.indirect);
    else pass.dispatchWorkgroups(x.direct![0], x.direct![1] ?? 1);
  }

  /** Encode one step into `encoder` (one compute pass). */
  encodeStep(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "seep.step" });
    for (const x of this.steps[this.stepIndex & 1]) this.encode(pass, x);
    pass.end();
    this.stepIndex++;
  }

  advanceTo(time_s: number): boolean {
    if (this.destroyed) return false;
    const submitted = this.info.submittedTime_s ?? 0;
    if (time_s + 1e-9 < submitted + DT) return false;
    const encoder = this.device.createCommandEncoder({ label: "seep.step" });
    this.encodeStep(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.info.submittedTime_s = submitted + DT;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    return true;
  }

  async awaitFrameCompletion(): Promise<void> {
    const t = this.info.submittedTime_s;
    await this.device.queue.onSubmittedWorkDone();
    this.info.completedTime_s = t;
  }

  /**
   * Time one step dispatch by dispatch (timestamp queries, one pass each).
   * Diagnostic only: it does advance the simulation by one step.
   */
  async profileStep(): Promise<{ label: string; ms: number }[]> {
    if (!this.device.features.has("timestamp-query")) throw new Error("seep.profileStep needs the timestamp-query feature");
    const list = this.steps[this.stepIndex & 1];
    const querySet = this.device.createQuerySet({ type: "timestamp", count: 2 * list.length });
    const resolve = this.device.createBuffer({ size: 16 * list.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.device.createBuffer({ size: 16 * list.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "seep.profile" });
    list.forEach((x, i) => {
      const pass = encoder.beginComputePass({ label: x.label, timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      this.encode(pass, x);
      pass.end();
    });
    encoder.resolveQuerySet(querySet, 0, 2 * list.length, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, read, 0, 16 * list.length);
    this.device.queue.submit([encoder.finish()]);
    this.stepIndex++;
    this.info.submittedTime_s = (this.info.submittedTime_s ?? 0) + DT;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    querySet.destroy(); resolve.destroy(); read.destroy();
    return list.map((x, i) => ({ label: x.label, ms: Number(t[2 * i + 1] - t[2 * i]) / 1e6 }));
  }

  private async measure(publish: boolean): Promise<{ volume: number; ctrl: Uint32Array }> {
    const encoder = this.device.createCommandEncoder({ label: "seep.measure" });
    const pass = encoder.beginComputePass({ label: "seep.measure" });
    this.encode(pass, publish ? this.volumeInit! : this.volumeMeasure!);
    pass.end();
    encoder.copyBufferToBuffer(this.buffers.vsum, 0, this.readback, 0, BRICKS * 4);
    encoder.copyBufferToBuffer(this.buffers.ctrl, 0, this.readback, BRICKS * 4, C.WORDS * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const raw = this.readback.getMappedRange().slice(0);
    this.readback.unmap();
    const sums = new Float32Array(raw, 0, BRICKS);
    let volume = 0;
    for (let i = 0; i < BRICKS; i++) volume += sums[i];
    return { volume, ctrl: new Uint32Array(raw, BRICKS * 4, C.WORDS) };
  }

  private async measureVolume(publish: boolean): Promise<number> { return (await this.measure(publish)).volume; }

  async readDiagnostics(): Promise<SeepDiagnostics> {
    const { volume, ctrl } = await this.measure(false);
    const f = new Float32Array(ctrl.buffer, ctrl.byteOffset, ctrl.length);
    const failBits = ctrl[C.FAIL];
    const failures = Object.entries(FAIL).filter(([, bit]) => (failBits & bit) !== 0).map(([name]) => name);
    const drift = (volume - this.initialVolumeCells) / this.initialVolumeCells;
    this.info.volumeCellSum = volume;
    this.info.volumeDrift = drift;
    this.info.pressureIterationsExecuted = ctrl[C.CYCLES];
    if (failures.length && !this.failureText) {
      this.failureText = `Seep solver latched ${failures.join(", ")} (max backtrace segments ${ctrl[C.MAX_SEG]}, pool overflow ${ctrl[C.OVERFLOW]})`;
      this.info.simulationPipelineError = this.failureText;
    }
    return {
      step: ctrl[C.STEP], volumeCells: volume, initialVolumeCells: this.initialVolumeCells, volumeDrift: drift, failures,
      particles: ctrl[C.PCOUNT] + ctrl[C.PCOUNT + 1], seeded: ctrl[C.SEEDED], deleted: ctrl[C.DELETED],
      cycles: ctrl[C.CYCLES], cyclesTotal: ctrl[C.CYCLES_TOTAL], capHits: ctrl[C.CAP_HITS],
      firstResidual: f[C.FIRST_RES], lastResidual: f[C.LAST_RES], maxSegments: ctrl[C.MAX_SEG],
      residualHistory: Array.from(f.subarray(C.RES_HIST, C.RES_HIST + MAX_CYCLES + 1)),
      lists: { active: ctrl[C.N_ACTIVE], surf: ctrl[C.N_ACTIVE + 1], part: ctrl[C.N_ACTIVE + 2], clear: ctrl[C.N_ACTIVE + 3] },
    };
  }

  /** Development aid: encode the next step only up to and including dispatch `label`, then stop (the solver is then mid-step). */
  debugPartialStep(label: string): void {
    const list = this.steps[this.stepIndex & 1];
    const end = list.findIndex(x => x.label === label);
    if (end < 0) throw new Error(`seep: no dispatch ${label}`);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const x of list.slice(0, end + 1)) this.encode(pass, x);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Development aid (tests and lane S tools only): copy a slice of an internal buffer. */
  async debugRead(name: string, byteOffset = 0, byteSize?: number): Promise<ArrayBuffer> {
    const src = this.buffers[name];
    if (!src) throw new Error(`seep: no buffer ${name}`);
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

  async readStats(): Promise<GPUEulerianInfo> {
    const d = await this.readDiagnostics();
    return { ...this.info, volumeCellSum: d.volumeCells, initialVolumeCellSum: d.initialVolumeCells, volumeDrift: d.volumeDrift, maxDivergence_s: d.lastResidual / CELL_M };
  }

  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    if (completion) await completion;
    const d = await this.readDiagnostics();
    if (d.failures.length) throw new Error(this.failureText);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of Object.values(this.buffers)) b.destroy();
    for (const u of this.uniforms.values()) u.destroy();
    this.readback.destroy();
    this.volumeTexture.destroy();
    this.surfaceFieldTexture.destroy();
  }
}

/** Observation accessors over the solver's own buffers (see seep-observation.ts). */
const OBSERVATION_WGSL = /* wgsl */ `
fn seepObs_cidx(c: vec3i) -> u32 {
  let b = u32(c.x >> 3u) | (u32(c.y >> 3u) << 6u) | (u32(c.z >> 3u) << 11u);
  return (b << 9u) | u32((c.x & 7) | ((c.y & 7) << 3u) | ((c.z & 7) << 6u));
}
fn seepVertexPhi(v: vec3i) -> f32 { return seepObs_phis[seepObs_cidx(clamp(v, vec3i(0), vec3i(${NX - 1}, ${NY - 1}, ${NZ - 1})))]; }
fn seepCellOpen(c: vec3i) -> bool {
  if (any(c < vec3i(0)) || any(c >= vec3i(${NX}, ${NY}, ${NZ}))) { return false; }
  if (c.x >= 240 && c.x < 256) { return c.y >= 64 && c.y < 66 && c.z >= 32 && c.z < 34; }
  return true;
}
fn seepFaceU(c: vec3i, axis: u32) -> f32 {
  let e = vec3i(select(0, 1, axis == 0u), select(0, 1, axis == 1u), select(0, 1, axis == 2u));
  let n = c + e;
  if (any(c < vec3i(0)) || any(n >= vec3i(${NX}, ${NY}, ${NZ}))) { return 0.0; }
  return seepObs_vel[axis * ${CELLS}u + seepObs_cidx(n)];
}
fn seepParticleCount() -> u32 { return seepObs_ctrl[${C.PCOUNT}] + seepObs_ctrl[${C.PCOUNT + 1}]; }
`;

export async function createSeepSolver(device: GPUDevice, scene: SceneDescription, options: SeepSolverOptions = {}): Promise<SeepSolver> {
  return SeepSolver.create(device, scene, options);
}
