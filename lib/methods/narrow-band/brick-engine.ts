/**
 * The brick engine's host (lane E, M1). This file covers:
 * - the pools and buffers, allocated from the scene's capacities;
 * - the boot;
 * - the recorded schedule (record(): every ring slot's dispatch list, with
 *   bind groups and cfg uniforms built once);
 * - the param ring (one writeBuffer a step);
 * - the two-part submit with the next step encoded ahead;
 * - lagged receipts, which raise a latched FAIL without a fence.
 *
 * Memory and work follow the fluid:
 * - Every buffer is sized from the capacities (scene.fluid.brickPools), apart
 *   from the directory's page window (one word a 32^3 page of the lattice).
 * - Every dispatch is either indirect from a list or sized from a capacity.
 * - Nothing is sized from the lattice.
 */
import { ARGS, BrickLayout, FAIL, H, LIST, NONE, RING, RING_SLOTS, RING_WORDS, bricksPerAxis, brickKey, PHI_AIR, REACH_MARGIN, type BrickCapacities } from "./brick-abi";
import { bindGroupLayoutEntries, type BrickDispatch, type BrickKernel, type BrickPlacement, ENGINE_BUFFERS } from "./brick-kernel";
import { type PressureSolver, type PressureSolverFactory, validatePressureSolver } from "./brick-pressure-slot";
import { brickSetKernels, churnGroups } from "./brick-set.wgsl";
import { brickPressureKernels } from "./brick-pressure.wgsl";
import { type BrickStages, type BrickStagesFactory, validateBrickStages } from "./brick-stages";

export type Vec3 = readonly [number, number, number];
/** A box of initial liquid, in cells (vertex coordinates of the lattice). */
export interface BrickLiquidBox { readonly lo: Vec3; readonly hi: Vec3 }

export interface BrickEngineOptions {
  readonly lattice: Vec3;
  readonly h_m: number;
  readonly dt_s: number;
  readonly gravity_mps2: Vec3;
  readonly capacities: BrickCapacities;
  readonly apic?: boolean;
  readonly liquid: readonly BrickLiquidBox[];
  /** The pressure solver at the slot (brick-abi.md §6.1). */
  readonly solver: PressureSolverFactory;
  /** F's stages (brick-abi.md §6.2; brick-stages.ts). */
  readonly stages: BrickStagesFactory;
  /** The method's resolved values and the fluid's constants, passed to the stage factory. */
  readonly values?: Readonly<Record<string, unknown>>;
  readonly fluid?: { readonly density_kg_m3: number; readonly viscosity_Pa_s: number; readonly surfaceTension_N_m: number };
  readonly label?: string;
}

export interface BrickStageSchedule { readonly part1: readonly BrickPlacement[]; readonly part2: readonly BrickPlacement[]; readonly boot: readonly BrickPlacement[] }

const RECEIPTS = 4;
const FAIL_NAMES = Object.entries(FAIL);
export const describeFail = (bits: number, info: number) =>
  `Brick engine failure ${FAIL_NAMES.filter(([, b]) => (bits & b) !== 0).map(([n]) => n).join("|") || `0x${bits.toString(16)}`} (info 0x${(info >>> 0).toString(16)})`;

/** The signed distance (h) of a box at vertex p, as bootPhi computes it. */
export function boxDistance(p: Vec3, b: BrickLiquidBox): number {
  const q = [0, 1, 2].map(a => Math.abs(p[a]! - 0.5 * (b.lo[a]! + b.hi[a]!)) - 0.5 * (b.hi[a]! - b.lo[a]!));
  return Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(q[0]!, q[1]!, q[2]!), 0);
}

/**
 * The reach rule (brick-abi.md §5.4) on the CPU, for the boot and the oracle.
 * `liquid(b)` gives a brick's liquid-vertex box in local coordinates (or
 * undefined), `bound(b)` its velocity bound (m/s), and `particles(b)` whether
 * its bin is non-empty. Returns the keys of every brick to keep or enter.
 */
export function reachSet(lattice: Vec3, bricks: Iterable<Vec3>, liquid: (b: Vec3) => { lo: Vec3; hi: Vec3 } | undefined,
  bound: (b: Vec3) => number, particles: (b: Vec3) => boolean, dt: number, h: number, gravity: Vec3): Set<number> {
  const out = new Set<number>(), nb = bricksPerAxis(lattice), g = Math.hypot(...gravity);
  for (const b of bricks) {
    const parts = particles(b), box = liquid(b);
    if (!parts && !box) continue;
    const lo = parts ? [0, 0, 0] : box!.lo, hi = parts ? [7, 7, 7] : box!.hi;
    const r = Math.fround(Math.fround((bound(b) + g * dt) * dt) / h) + REACH_MARGIN;
    const b0 = [0, 1, 2].map(a => Math.max(Math.floor((8 * b[a]! + lo[a]! - r) / 8), 0));
    const b1 = [0, 1, 2].map(a => Math.min(Math.floor((8 * b[a]! + hi[a]! + r) / 8), nb[a]! - 1));
    for (let z = b0[2]!; z <= b1[2]!; z++) for (let y = b0[1]!; y <= b1[1]!; y++) for (let x = b0[0]!; x <= b1[0]!; x++) out.add(brickKey(x, y, z));
  }
  return out;
}

/** The initial bricks: the reach set of the boxes' liquid vertices, at rest. */
export function bootBricks(o: Pick<BrickEngineOptions, "lattice" | "liquid" | "dt_s" | "h_m" | "gravity_mps2">): number[] {
  const nb = bricksPerAxis(o.lattice), f32 = (b: BrickLiquidBox): BrickLiquidBox => ({ lo: o.lattice.map((_, a) => Math.fround(b.lo[a]!)) as unknown as Vec3, hi: o.lattice.map((_, a) => Math.fround(b.hi[a]!)) as unknown as Vec3 });
  const boxes = o.liquid.map(f32), candidates = new Map<number, Vec3>();
  for (const box of boxes) {
    const lo = [0, 1, 2].map(a => Math.max(Math.floor(box.lo[a]! / 8) - 1, 0)), hi = [0, 1, 2].map(a => Math.min(Math.floor(box.hi[a]! / 8) + 1, nb[a]! - 1));
    for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) candidates.set(brickKey(x, y, z), [x, y, z]);
  }
  const liquid = (b: Vec3) => {
    let lo = [8, 8, 8], hi = [-1, -1, -1];
    for (let l = 0; l < 512; l++) {
      const c = [l & 7, (l >> 3) & 7, l >> 6], v = [0, 1, 2].map(a => 8 * b[a]! + c[a]!) as unknown as Vec3;
      if (v.some((x, a) => x > o.lattice[a]!)) continue;
      const d = Math.min(PHI_AIR, ...boxes.map(box => boxDistance(v, box)));
      if (Math.fround(d) < 0) { lo = lo.map((m, a) => Math.min(m, c[a]!)); hi = hi.map((m, a) => Math.max(m, c[a]!)); }
    }
    return hi[0]! < 0 ? undefined : { lo: lo as unknown as Vec3, hi: hi as unknown as Vec3 };
  };
  return [...reachSet(o.lattice, candidates.values(), liquid, () => 0, () => false, o.dt_s, o.h_m, o.gravity_mps2)].sort((a, b) => a - b);
}

interface Recorded { readonly pipeline: GPUComputePipeline; readonly group: GPUBindGroup; readonly dispatch: BrickDispatch }

export class BrickEngine {
  readonly layout: BrickLayout;
  readonly solver: PressureSolver;
  readonly stages: BrickStages;
  readonly schedule: BrickStageSchedule;
  /** Pipeline compile times (ms), by kernel label, from this engine's creation. */
  readonly compileMs: Readonly<Record<string, number>>;
  private readonly buffers = new Map<string, GPUBuffer>();
  private readonly cfgs = new Map<string, GPUBuffer>();
  private readonly recorded: { part1: Recorded[]; part2: Recorded[] }[] = [];
  private readonly receipts: { buffer: GPUBuffer; busy: boolean }[] = [];
  private ahead?: { step: number; part1: GPUCommandBuffer; part2: GPUCommandBuffer; receipt: number };
  private steps = 0;
  private time = 0;
  /** The newest receipt (the set buffer's control header) and the step it closed. */
  receipt?: { step: number; header: Uint32Array };
  failure?: string;

  private constructor(readonly device: GPUDevice, readonly options: BrickEngineOptions, layout: BrickLayout, solver: PressureSolver, stages: BrickStages,
    schedule: BrickStageSchedule, private readonly pipelines: Map<BrickKernel, GPUComputePipeline>, compileMs: Record<string, number>) {
    this.layout = layout; this.solver = solver; this.stages = stages; this.schedule = schedule; this.compileMs = compileMs;
  }

  /** Every placement of one step and the boot, in order (record()'s input). */
  static scheduleFor(layout: BrickLayout, solver: PressureSolver, stages: BrickStages, boxes: number): BrickStageSchedule {
    const set = brickSetKernels(layout), pres = brickPressureKernels(layout), f = layout.cap.f, c = layout.cap.c;
    const list = (k: number, g: "G1" | "G8" | "G64"): BrickDispatch => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
    const direct = (n: number): BrickDispatch => ({ direct: [n, 1, 1] });
    const update: BrickPlacement[] = [
      { kernel: set.setClaimPages, dispatch: direct(churnGroups(f, 64)) },
      { kernel: set.setClaimBricks, dispatch: direct(churnGroups(f, 64)) },
    ];
    const tail: BrickPlacement[] = [
      { kernel: set.setRecords, dispatch: direct(churnGroups(f, 1)) },
      { kernel: set.setInit, dispatch: direct(churnGroups(f, 1)) },
    ];
    const lists: BrickPlacement[] = [
      { kernel: set.setLists, dispatch: direct(Math.ceil(Math.max(f, c) / 64)) },
      { kernel: set.setArgs, dispatch: direct(1) },
    ];
    const part2: BrickPlacement[] = [
      { kernel: set.setMark, dispatch: list(LIST.F_ALL, "G1") },
      ...update,
      { kernel: set.setLeave, dispatch: list(LIST.F_ALL, "G64") },
      { kernel: set.setLeavePages, dispatch: list(LIST.C_ALL, "G64") },
      ...tail, ...lists,
      ...stages.levelSet,
      ...stages.velocity,
      { kernel: pres.operator, dispatch: list(LIST.F_ALL, "G1") },
      ...solver.setup,
      ...Array.from({ length: solver.slots }, (_, k) => solver.slot(k)).flat(),
      ...solver.finish,
      { kernel: pres.project, dispatch: list(LIST.F_LIQ, "G1") },
      ...stages.tail,
      { kernel: set.stepEnd, dispatch: direct(1), z: 0 },
    ];
    const boot: BrickPlacement[] = [...update, ...tail, { kernel: set.bootPhi, dispatch: direct(churnGroups(f, 1)), z: boxes }, ...lists, ...stages.boot, { kernel: set.stepEnd, dispatch: direct(1), z: 1 }];
    // Part 1: the head (P's bodyStamp, E's solidStamp: M3 and M5), then F's move and rebin.
    return { part1: [...stages.move, ...stages.rebin], part2, boot };
  }

  static async create(device: GPUDevice, options: BrickEngineOptions): Promise<BrickEngine> {
    const layout = new BrickLayout({ lattice: options.lattice, ...options.capacities, apic: options.apic });
    const solver = options.solver(layout);
    validatePressureSolver(solver);
    const stages = options.stages({ layout, dt_s: options.dt_s, h_m: options.h_m, gravity_mps2: options.gravity_mps2, values: options.values ?? {}, fluid: options.fluid });
    validateBrickStages(stages);
    const initial = bootBricks(options);
    if (initial.length > layout.cap.f) throw new Error(`Brick engine: the initial liquid needs ${initial.length} F bricks, over scene.fluid.brickPools.f = ${layout.cap.f}`);
    const schedule = BrickEngine.scheduleFor(layout, solver, stages, options.liquid.length);
    // One pipeline a kernel, compiled concurrently; each module once.
    const kernels = [...new Set([...schedule.boot, ...schedule.part1, ...schedule.part2].map(p => p.kernel))];
    const modules = new Map<string, GPUShaderModule>(), pipelines = new Map<BrickKernel, GPUComputePipeline>(), compileMs: Record<string, number> = {};
    await Promise.all(kernels.map(async k => {
      const t0 = performance.now();
      let module = modules.get(k.code);
      if (!module) {
        module = device.createShaderModule({ label: `nb-brick.${k.label}`, code: k.code }); modules.set(k.code, module);
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
        if (errors.length) {
          const lines = k.code.split("\n");
          throw new Error(`Brick engine: ${k.label} does not compile:\n${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}\n  ${lines[m.lineNum - 1]?.slice(Math.max(0, m.linePos - 80), m.linePos + 80) ?? ""}`).join("\n")}`);
        }
      }
      const layout_ = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: bindGroupLayoutEntries(k.bindings) })] });
      pipelines.set(k, await device.createComputePipelineAsync({ label: `nb-brick.${k.label}`, layout: layout_, compute: { module, entryPoint: k.entry } }));
      compileMs[k.label] = performance.now() - t0;
    }));
    const engine = new BrickEngine(device, options, layout, solver, stages, schedule, pipelines, compileMs);
    engine.allocate(initial);
    engine.record();
    engine.boot();
    engine.encodeAhead();
    return engine;
  }

  /** Bytes of every buffer the engine allocated. */
  get bytes(): number { let n = 0; for (const b of this.buffers.values()) n += b.size; return n; }
  get stepsSubmitted(): number { return this.steps; }

  private allocate(initial: readonly number[]) {
    const L = this.layout, d = this.device, U = GPUBufferUsage;
    const make = (name: string, words: number, usage = 0) => {
      const b = d.createBuffer({ label: `nb-brick.${name}`, size: 4 * Math.max(words, 4), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | usage });
      this.buffers.set(name, b); return b;
    };
    const set = make("set", L.set.words), dir = make("dir", L.dir.words);
    make("args", ARGS.WORDS, U.INDIRECT);
    for (const pool of ["vel", "phi", "pres", "cut"] as const) make(pool, L.poolWords[pool]);
    make("parts", L.partsWords);
    for (const [name, words] of [...Object.entries(this.solver.buffers), ...Object.entries(this.stages.buffers)]) {
      if ((ENGINE_BUFFERS as readonly string[]).includes(name) || name === "boot" || this.buffers.has(name)) throw new Error(`Brick engine: lane buffer ${name} shadows another buffer`);
      make(name, words);
    }
    const boxes = new Float32Array(Math.max(6 * this.options.liquid.length, 4));
    this.options.liquid.forEach((b, i) => boxes.set([...b.lo, ...b.hi], 6 * i));
    const boot = make("boot", boxes.length); d.queue.writeBuffer(boot, 0, boxes);
    // The set buffer: free stacks full (FREE[i] = CAP - 1 - i), the initial candidates.
    const s = new Uint32Array(L.set.words), cap = L.cap;
    s[H.TOP_F] = cap.f; s[H.TOP_C] = cap.c; s[H.TOP_B] = cap.b; s[H.TOP_K] = cap.k;
    const fill = (at: number, n: number) => { for (let i = 0; i < n; i++) s[at + i] = n - 1 - i; };
    fill(L.set.freeF, cap.f); fill(L.set.freeC, cap.c); fill(L.set.freeB, cap.b); fill(L.set.freeK, cap.k);
    s.set(initial, L.set.cand); s[H.N_CAND] = initial.length;
    d.queue.writeBuffer(set, 0, s);
    // The directory: every page, child and record word NONE, the tile masks clear.
    const w = new Uint32Array(L.dir.words).fill(NONE); w.fill(0, L.dir.cmask);
    d.queue.writeBuffer(dir, 0, w);
    for (let i = 0; i < RECEIPTS; i++) this.receipts.push({ buffer: d.createBuffer({ label: `nb-brick.receipt${i}`, size: 4 * 256, usage: U.MAP_READ | U.COPY_DST }), busy: false });
  }

  private cfg(x: number, y: number, z: number, w: number): GPUBuffer {
    const key = `${x},${y},${z},${w}`;
    let b = this.cfgs.get(key);
    if (!b) {
      b = this.device.createBuffer({ label: `nb-brick.cfg.${key}`, size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set([x, y, z, w]); b.unmap(); this.cfgs.set(key, b);
    }
    return b;
  }

  private bind(p: BrickPlacement, slot: number): Recorded {
    const pipeline = this.pipelines.get(p.kernel)!;
    const entries = p.kernel.bindings.map((b, i): GPUBindGroupEntry => {
      if (b.name === "cfg") return { binding: i, resource: { buffer: this.cfg(slot, slot & 1, p.z ?? 0, p.w ?? 0) } };
      const buffer = this.buffers.get(b.name);
      if (!buffer) throw new Error(`Brick engine: ${p.kernel.label} binds unknown buffer ${b.name}`);
      return { binding: i, resource: { buffer } };
    });
    return { pipeline, group: this.device.createBindGroup({ label: `nb-brick.${p.kernel.label}.${slot}`, layout: pipeline.getBindGroupLayout(0), entries }), dispatch: p.dispatch };
  }

  /** record(): each ring slot's two parts as pipelines, bind groups and dispatches, built once. */
  private record() {
    for (let slot = 0; slot < RING_SLOTS; slot++)
      this.recorded.push({ part1: this.schedule.part1.map(p => this.bind(p, slot)), part2: this.schedule.part2.map(p => this.bind(p, slot)) });
  }

  private encode(list: readonly Recorded[], label: string, receipt?: number): GPUCommandBuffer {
    const enc = this.device.createCommandEncoder({ label });
    if (list.length) {
      const pass = enc.beginComputePass({ label });
      const args = this.buffers.get("args")!;
      for (const r of list) {
        pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
        if ("direct" in r.dispatch) pass.dispatchWorkgroups(...r.dispatch.direct); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
      }
      pass.end();
    }
    if (receipt !== undefined) enc.copyBufferToBuffer(this.buffers.get("set")!, 0, this.receipts[receipt]!.buffer, 0, 4 * 256);
    return enc.finish();
  }

  private ringWords(step: number): ArrayBuffer {
    const o = this.options, buf = new ArrayBuffer(4 * RING_WORDS), u = new Uint32Array(buf), f = new Float32Array(buf);
    u[RING.STEP] = step; f[RING.DT] = o.dt_s; f[RING.TIME] = this.time; u[RING.FLAGS] = 0;
    f.set(o.gravity_mps2, RING.GRAVITY); f[RING.H] = o.h_m;
    return buf;
  }

  private boot() {
    const rec = this.schedule.boot.map(p => this.bind(p, 0));
    this.device.queue.writeBuffer(this.buffers.get("set")!, 4 * this.layout.set.ring, this.ringWords(0));
    this.device.queue.submit([this.encode(rec, "nb-brick.boot")]);
  }

  private encodeAhead() {
    const step = this.steps, rec = this.recorded[step % RING_SLOTS]!;
    const free = this.receipts.findIndex((r, i) => !r.busy && i === step % RECEIPTS);
    const receipt = free < 0 ? undefined : free;
    if (receipt !== undefined) this.receipts[receipt]!.busy = true;
    this.ahead = { step, part1: this.encode(rec.part1, `nb-brick.step${step}.part1`), part2: this.encode(rec.part2, `nb-brick.step${step}.part2`, receipt), receipt: receipt ?? -1 };
  }

  /** Submit `steps` steps: per step one ring write, part 1, part 2, then the next step encoded ahead. */
  advance(steps = 1) {
    for (let i = 0; i < steps; i++) {
      if (this.failure) throw new Error(this.failure);
      const a = this.ahead!, slot = a.step % RING_SLOTS;
      this.device.queue.writeBuffer(this.buffers.get("set")!, 4 * (this.layout.set.ring + RING_WORDS * slot), this.ringWords(a.step));
      this.device.queue.submit([a.part1]);
      this.device.queue.submit([a.part2]);
      if (a.receipt >= 0) this.collect(a.receipt, a.step);
      this.steps++; this.time += this.options.dt_s;
      this.encodeAhead();
    }
  }

  private collect(index: number, step: number) {
    const r = this.receipts[index]!;
    r.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const header = new Uint32Array(r.buffer.getMappedRange().slice(0));
      r.buffer.unmap(); r.busy = false;
      if (!this.receipt || this.receipt.step < step) this.receipt = { step, header };
      if (header[H.FAIL] !== 0 && !this.failure) this.failure = describeFail(header[H.FAIL]!, header[H.FAIL_INFO]!);
    }, () => { r.busy = false; });
  }

  /** Wait for every submitted step (tests and harnesses; never on the step cadence). */
  async fence() { await this.device.queue.onSubmittedWorkDone(); }

  /** Read words of an engine buffer (tests and harnesses). */
  async read(name: string, offsetWords = 0, words?: number): Promise<Uint32Array> {
    const src = this.buffers.get(name);
    if (!src) throw new Error(`Brick engine: no buffer ${name}`);
    const n = words ?? src.size / 4 - offsetWords;
    const staging = this.device.createBuffer({ size: 4 * Math.max(n, 1), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(src, 4 * offsetWords, staging, 0, 4 * n);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy();
    return out;
  }

  /** An engine buffer, for adapters that bind it outside the recorded step (presentation, observers). */
  buffer(name: string): GPUBuffer {
    const b = this.buffers.get(name);
    if (!b) throw new Error(`Brick engine: no buffer ${name}`);
    return b;
  }

  /** Write words into an engine buffer (tests: perturb a field between steps). */
  write(name: string, offsetWords: number, data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>) {
    this.device.queue.writeBuffer(this.buffers.get(name)!, 4 * offsetWords, data);
  }

  /** Throws the latched failure, if any, after fencing (tests). */
  async assertHealthy() {
    await this.fence();
    const h = await this.read("set", 0, 4);
    if (h[H.FAIL] !== 0) throw new Error(describeFail(h[H.FAIL]!, h[H.FAIL_INFO]!));
  }

  destroy() {
    for (const b of this.buffers.values()) b.destroy();
    for (const b of this.cfgs.values()) b.destroy();
    for (const r of this.receipts) r.buffer.destroy();
  }
}
