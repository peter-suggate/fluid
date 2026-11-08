/**
 * Tier A GPU cloth solver: one implicit step per `encodeStep`, at the scene dt,
 * with K VBD iterations over vertex colours. There are no substeps. Every
 * per-step stage runs on the GPU and nothing is read back. The step clock,
 * the BDF2 bootstrap and attachment scripts are evaluated in WGSL. Status
 * and diagnostics land in a ring that harnesses read asynchronously.
 *
 * Five storage buffers (the M1 Max grants 10 per stage; C2's contact block adds one):
 *   positions  vec4f[N]  the iterate x, cloth-local metres (xyz; w free). Shared with render and C2.
 *   state      vec4f     per-vertex segments (stride Np = N padded to 16 records, 256 B):
 *                          0 velocity at the step start (C2 binds `state` as its velocity input)
 *                          1 x_t, 2 v_{n-1}, 3 dx_{n-1}, 4 y, 5 x^, 6 Chebyshev x(n-1), 7 x(n-2),
 *                          8 pin offset p_{n+1} + alpha (x_t - p_n), 9 pin dual (lambda xyz, k),
 *                          10 vertex normal (render)
 *                        then per edge (lambda, k, stabilized limit), the damping tangent at x_t
 *                        (per face f0, f1 with D11, D22; per hinge grad theta x 4; damped scenes
 *                        only), diagnostics partials, and the clock record (step, previous step, h~, BDF2).
 *   status     u32       sticky status bits, steps completed, the diagnostics ring
 *   topo       u32       CSR vertex-face/hinge/edge adjacency, colour-sorted vertices, hinge vertices,
 *                        triangles (v0, v1, v2, material) and canonical edges (v0 < v1, f0, f1), each
 *                        segment 256 B aligned so render and C2 bind slices of it
 *   rest       vec4f     face Dm^-1, (area, damping), hinge (theta0, k, kd), edge (L, eps),
 *                        vertex (m, elastic, length, group), anchor, materials, scripts
 */
import { writeBufferChunked } from '../methods/uniform/uniform-chunked-upload';
import type { ClothIntegrator, ClothSolverStateData, ClothVector } from './types';
import { clothChebyshevOmega, type ClothSolverModel } from './oracle';
import {
  CLOTH_BINDING, CLOTH_DISPATCH_STRIDE, CLOTH_SCRIPT_KIND, CLOTH_SEGMENT_ALIGN, CLOTH_SLOT, CLOTH_STATE, CLOTH_STATE_SEGMENTS, CLOTH_STATUS,
  CLOTH_STATUS_RING_BASE, CLOTH_STATUS_RING_SLOTS, CLOTH_STATUS_SLOT_WORDS, CLOTH_STATUS_WORDS, CLOTH_WORKGROUP, clothSolverWGSL,
} from './gpu-solver-wgsl';

/** Storage buffers per stage the solver binds; C2's contact block adds one. */
export const CLOTH_SOLVER_STORAGE_BUFFERS = 5;

export interface ClothSolverContact {
  /** ogcVertexBlockWGSL({group: 0, binding: CLOTH_BINDING.contact}). */
  readonly wgsl: string;
  /** OgcContactDetector.solverBuffer. */
  readonly buffer: GPUBuffer;
}

/** Points inside one step where another lane encodes work into the solver's
 * compute pass. They never advance time: C2's detection and truncation
 * rounds run here, between iterations of the one step. */
export interface ClothStepHooks {
  /** After the step frame (inertia target and initial guess are in place), before iteration 1. */
  afterFrame?(pass: GPUComputePassEncoder): void;
  /** After colour `colour`'s sweep of iteration `iteration` (1-based); C2 truncates here (DAT section 5.1). */
  afterColour?(pass: GPUComputePassEncoder, iteration: number, colour: number): void;
  /** After iteration `iteration` (1-based): all colours, then Chebyshev (skipped for vertices with
   * contacts) and the dual updates. C2 truncates the Chebyshev move and re-detects here. */
  afterIteration?(pass: GPUComputePassEncoder, iteration: number): void;
  /** After the last iteration, before velocities are formed. */
  beforeFinish?(pass: GPUComputePassEncoder): void;
}

export interface ClothGPUSolverOptions {
  /** World position of the cloth-local origin. */
  readonly origin_m: ClothVector;
  readonly dt_s: number;
  readonly gravity_m_s2: ClothVector;
  readonly integrator: ClothIntegrator;
  readonly iterations: number;
  readonly chebyshevRho: number;
  /** Most iterations `setIterations` may ask for (dispatch records are authored once); default `iterations`. */
  readonly maxIterations?: number;
  /** World positions and velocities at the start (default: rest, zero). */
  readonly positions_m?: ArrayLike<number>;
  readonly velocities_m_s?: ArrayLike<number>;
  readonly solverState?: ClothSolverStateData;
  /** Write the diagnostics slot of every step (residual, strain, energies). */
  readonly diagnostics?: boolean;
  readonly contact?: ClothSolverContact;
  readonly hooks?: ClothStepHooks;
  readonly label?: string;
}

export interface ClothStatusSlot {
  readonly step: number;
  readonly bdf2: boolean;
  /** |grad E| / |m g| of the step's variational problem at its last iterate. */
  readonly residual: number;
  readonly maxStrain: number;
  readonly kinetic_J: number;
  /** -sum m g . x in the cloth-local frame. */
  readonly potential_J: number;
  readonly elastic_J: number;
}

export interface ClothStatusReport {
  readonly status: number;
  readonly steps: number;
  /** Ring slots in step order (only steps run with diagnostics on; the last CLOTH_STATUS_RING_SLOTS). */
  readonly slots: readonly ClothStatusSlot[];
}

export interface ClothGPUState {
  readonly positions_m: Float64Array;
  readonly velocities_m_s: Float64Array;
  readonly solverState: ClothSolverStateData;
}

/** Packed setup-time tables of a model (CPU, edit time only). */
export interface ClothGPUTables {
  readonly topo: Uint32Array<ArrayBuffer>;
  readonly rest: Float32Array<ArrayBuffer>;
  /** Word offsets of the triangle and edge segments in `topo` (256 B aligned). */
  readonly trianglesWord: number;
  readonly edgesWord: number;
  /** Colour ranges (start, count) in the colour-sorted vertex list. */
  readonly colourRanges: readonly (readonly [number, number])[];
  readonly params: Uint32Array<ArrayBuffer>;
  readonly strainLimited: boolean;
  readonly damped: boolean;
  /** State layout, in vec4 records. */
  readonly stride: number;
  readonly edgeDualRecord: number;
  readonly dampFaceRecord: number;
  readonly dampHingeRecord: number;
  readonly partialRecord: number;
  readonly clockRecord: number;
  readonly stateRecords: number;
}

const PARAM_WORDS = 44;
/** Float words of the params uniform: gravity xyz + dt, then the clock origin and rho. */
const PARAM_GRAVITY = 36, PARAM_TIME = 40;
const align = (n: number, a: number) => Math.ceil(n / a) * a;

export function packClothGPUTables(model: ClothSolverModel, origin: ClothVector, options: Pick<ClothGPUSolverOptions, 'dt_s' | 'gravity_m_s2' | 'integrator' | 'chebyshevRho'>, time_s = 0): ClothGPUTables {
  const t = model.topology, N = t.vertexCount, T = t.faceCount, E = t.edgeCount, H = t.hingeCount, M = model.materials.length, S = model.scripts.length;
  const byColour: number[][] = Array.from({ length: model.colourCount }, () => []);
  for (let v = 0; v < N; v++) byColour[model.colours[v]!]!.push(v);
  const order = byColour.flat(), colourRanges: [number, number][] = [];
  let start = 0;
  for (const list of byColour) { colourRanges.push([start, list.length]); start += list.length; }
  const triangles = new Uint32Array(4 * T);
  for (let f = 0; f < T; f++) triangles.set([t.faces[3 * f]!, t.faces[3 * f + 1]!, t.faces[3 * f + 2]!, t.faceMaterial[f]!], 4 * f);
  const parts = [t.vertexFaceOffsets, t.vertexFaces, t.vertexHingeOffsets, t.vertexHinges, t.vertexEdgeOffsets, t.vertexEdges, Uint32Array.from(order), t.hinges, triangles, t.edges];
  const topoBase: number[] = [];
  let words = 0;
  for (const p of parts) { words = align(words, 64); topoBase.push(words); words += p.length; }
  const topo = new Uint32Array(Math.max(align(words, 4), 4));
  parts.forEach((p, i) => topo.set(p, topoBase[i]));
  const restCounts = [T, T, H, E, N, N, 2 * M, 4 * S], restBase: number[] = [];
  let records = 0;
  for (const c of restCounts) { restBase.push(records); records += c; }
  const rest = new Float32Array(4 * Math.max(records, 1));
  for (let f = 0; f < T; f++) {
    rest.set(t.faceDmInv.subarray(4 * f, 4 * f + 4), 4 * (restBase[0]! + f));
    rest.set([t.faceArea_m2[f]!, model.faceDamping_s[f]!], 4 * (restBase[1]! + f));
  }
  for (let h = 0; h < H; h++) rest.set([t.hingeRestAngle_rad[h]!, t.hingeStiffness_N_m[h]!, model.hingeDamping_s[h]!], 4 * (restBase[2]! + h));
  let strainLimited = false;
  for (let e = 0; e < E; e++) { rest.set([t.edgeRestLength_m[e]!, model.edgeStrainLimit[e]!], 4 * (restBase[3]! + e)); strainLimited ||= model.edgeStrainLimit[e]! > 0; }
  const restPos = t.data.restPositions_m;
  for (let v = 0; v < N; v++) {
    if (t.groups[v]! >= 2 ** 24) throw new Error('Cloth attachment groups must be below 2^24');
    rest.set([t.mass_kg[v]!, model.elasticStiffness_N_m[v]!, model.vertexLength_m[v]!, t.groups[v]!], 4 * (restBase[4]! + v));
    rest.set([restPos[3 * v]! - origin[0], restPos[3 * v + 1]! - origin[1], restPos[3 * v + 2]! - origin[2]], 4 * (restBase[5]! + v));
  }
  model.membrane.forEach((c, m) => rest.set([c.c11, c.c22, c.c12, c.g, c.softWarp, c.softWeft, c.ratio, 0], 4 * (restBase[6]! + 2 * m)));
  model.scripts.forEach((s, i) => {
    const kind = CLOTH_SCRIPT_KIND[s.kind];
    if (kind === undefined) throw new Error(`Unknown cloth attachment script kind ${String(s.kind)}`);
    if (!(s.end_s > s.start_s)) throw new Error(`Cloth attachment script ${i} needs end_s > start_s`);
    if (s.kind === 'oscillate' && !(s.period_s! > 0)) throw new Error(`Cloth oscillate script ${i} needs a positive period`);
    if (s.kind === 'rotate' && !(Math.hypot(...s.vector) > 0 && Number.isFinite(s.angle_rad))) throw new Error(`Cloth rotate script ${i} needs an axis and an angle`);
    const o = s.origin_m ?? [0, 0, 0];
    rest.set([kind, s.group, s.start_s, s.end_s, ...s.vector, s.angle_rad ?? 0, o[0] - origin[0], o[1] - origin[1], o[2] - origin[2], s.period_s ?? 0,
      s.easing === 'cosine' ? 1 : 0, 0, 0, 0], 4 * (restBase[7]! + 4 * i));
  });
  let damped = false;
  for (let f = 0; f < T; f++) damped ||= model.faceDamping_s[f]! > 0;
  for (let h = 0; h < H; h++) damped ||= model.hingeDamping_s[h]! > 0;
  const stride = align(N, CLOTH_SEGMENT_ALIGN), workgroups = Math.ceil(N / CLOTH_WORKGROUP);
  const edgeDualRecord = CLOTH_STATE_SEGMENTS * stride, dampFaceRecord = edgeDualRecord + align(E, CLOTH_SEGMENT_ALIGN);
  const dampHingeRecord = dampFaceRecord + (damped ? align(2 * T, CLOTH_SEGMENT_ALIGN) : 0);
  const partialRecord = dampHingeRecord + (damped ? align(3 * H, CLOTH_SEGMENT_ALIGN) : 0);
  const clockRecord = partialRecord + align(2 * workgroups, CLOTH_SEGMENT_ALIGN), stateRecords = clockRecord + CLOTH_SEGMENT_ALIGN;
  const params = new Uint32Array(PARAM_WORDS), paramsF = new Float32Array(params.buffer);
  params.set([N, T, E, H, ...topoBase, 0, 0, ...restBase, S, options.integrator === 'bdf2' ? 1 : 0, strainLimited ? 1 : 0, workgroups,
    stride, edgeDualRecord, partialRecord, clockRecord, dampFaceRecord, dampHingeRecord, damped ? 1 : 0, 0]);
  paramsF.set([...options.gravity_m_s2, options.dt_s, time_s, options.chebyshevRho, 0, 0], PARAM_GRAVITY);
  return { topo, rest, trianglesWord: topoBase[8]!, edgesWord: topoBase[9]!, colourRanges, params, strainLimited, damped, stride, edgeDualRecord,
    dampFaceRecord, dampHingeRecord, partialRecord, clockRecord, stateRecords };
}

type Kernel = 'beginStep' | 'frameElements' | 'frame' | 'solveColour' | 'postIteration' | 'edgeDualUpdate' | 'diagnose' | 'reduceDiagnostics' | 'finishStep';
const KERNELS: readonly Kernel[] = ['beginStep', 'frameElements', 'frame', 'solveColour', 'postIteration', 'edgeDualUpdate', 'diagnose', 'reduceDiagnostics', 'finishStep'];

export class ClothGPUSolver {
  readonly N: number; readonly T: number; readonly E: number;
  readonly positions: GPUBuffer; readonly state: GPUBuffer; readonly status: GPUBuffer; readonly topo: GPUBuffer;
  private readonly rest: GPUBuffer; private readonly params: GPUBuffer; private readonly dispatch: GPUBuffer;
  readonly tables: ClothGPUTables;
  private pipelines!: Record<Kernel, GPUComputePipeline>;
  private bindGroup!: GPUBindGroup;
  private iterations: number;
  private readonly maxIterations: number;
  /** Scene time at clock step 0 (load and undo restart the step count). */
  private timeBase = 0;
  diagnostics: boolean;
  hooks: ClothStepHooks;

  private constructor(private readonly device: GPUDevice, readonly model: ClothSolverModel, readonly options: ClothGPUSolverOptions) {
    const t = model.topology;
    this.N = t.vertexCount; this.T = t.faceCount; this.E = t.edgeCount;
    if (!(options.dt_s > 0)) throw new Error('Cloth solver needs a positive step');
    if (!(Number.isInteger(options.iterations) && options.iterations >= 1)) throw new Error('Cloth solver needs at least one iteration per step');
    if (!(options.chebyshevRho >= 0 && options.chebyshevRho < 1)) throw new Error('Cloth Chebyshev rho must be in [0, 1)');
    const need = CLOTH_SOLVER_STORAGE_BUFFERS + (options.contact ? 1 : 0);
    if (device.limits.maxStorageBuffersPerShaderStage < need)
      throw new Error(`Cloth solver needs ${need} storage buffers per stage; the device grants ${device.limits.maxStorageBuffersPerShaderStage}`);
    this.iterations = options.iterations; this.maxIterations = Math.max(options.maxIterations ?? options.iterations, options.iterations);
    this.diagnostics = !!options.diagnostics; this.hooks = options.hooks ?? {};
    this.tables = packClothGPUTables(model, options.origin_m, options, options.solverState?.time_s ?? 0);
    const label = options.label ?? 'Cloth', storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (name: string, bytes: number, usage = storage) => device.createBuffer({ label: `${label} ${name}`, size: Math.max(256, align(bytes, 256)), usage });
    this.positions = make('positions', 16 * this.tables.stride);
    this.state = make('state', 16 * this.tables.stateRecords);
    this.status = make('status', 4 * CLOTH_STATUS_WORDS);
    this.topo = make('topology', 4 * this.tables.topo.length);
    this.rest = make('rest data', 4 * this.tables.rest.length);
    this.params = make('params', 4 * PARAM_WORDS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.dispatch = make('dispatch records', CLOTH_DISPATCH_STRIDE * this.recordCount(this.maxIterations), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    writeBufferChunked(device.queue, this.topo, 0, this.tables.topo);
    writeBufferChunked(device.queue, this.rest, 0, this.tables.rest);
    this.writeDispatchRecords();
    this.uploadState(options.positions_m ?? t.data.restPositions_m, options.velocities_m_s, options.solverState);
  }

  static async create(device: GPUDevice, model: ClothSolverModel, options: ClothGPUSolverOptions): Promise<ClothGPUSolver> {
    const solver = new ClothGPUSolver(device, model, options);
    await solver.compile();
    return solver;
  }

  get iterationCount(): number { return this.iterations; }
  get colourCount(): number { return this.tables.colourRanges.length; }
  get strainLimited(): boolean { return this.tables.strainLimited; }
  /** Slices other consumers bind: velocities (C2), normals and triangles (render), canonical edges (C2). */
  get velocities(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.velocity * this.tables.stride, size: 16 * this.tables.stride }; }
  get normals(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.normal * this.tables.stride, size: 16 * this.tables.stride }; }
  get triangles(): GPUBufferBinding { return { buffer: this.topo, offset: 4 * this.tables.trianglesWord, size: 16 * Math.max(this.T, 1) }; }
  get edges(): GPUBufferBinding { return { buffer: this.topo, offset: 4 * this.tables.edgesWord, size: 16 * Math.max(this.E, 1) }; }
  get positionsBinding(): GPUBufferBinding { return { buffer: this.positions, offset: 0, size: 16 * this.tables.stride }; }

  /** Change K (a per-scene authoring value, up to `maxIterations`); rewrites the dispatch records, never per step. */
  setIterations(k: number): void {
    if (!(Number.isInteger(k) && k >= 1)) throw new Error('Cloth solver needs at least one iteration per step');
    if (k > this.maxIterations) throw new Error(`Cloth dispatch records were authored for at most ${this.maxIterations} iterations`);
    this.iterations = k; this.writeDispatchRecords();
  }

  private recordCount(k: number): number { return k * (this.tables.colourRanges.length + 1); }

  private writeDispatchRecords(): void {
    const C = this.tables.colourRanges.length, records = new ArrayBuffer(CLOTH_DISPATCH_STRIDE * this.recordCount(this.iterations));
    let omega = 1;
    for (let it = 1; it <= this.iterations; it++) {
      omega = clothChebyshevOmega(it, this.options.chebyshevRho, omega);
      for (let c = 0; c <= C; c++) {
        const offset = CLOTH_DISPATCH_STRIDE * ((it - 1) * (C + 1) + c), u = new Uint32Array(records, offset, 8), f = new Float32Array(records, offset, 8);
        const [start, count] = c < C ? this.tables.colourRanges[c]! : [0, this.N];
        u.set([start, count, it, 0]); f[4] = omega;
      }
    }
    writeBufferChunked(this.device.queue, this.dispatch, 0, new Uint8Array(records));
  }

  private async compile(): Promise<void> {
    const d = this.device, B = CLOTH_BINDING, C = GPUShaderStage.COMPUTE;
    const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } });
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: B.params, visibility: C, buffer: { type: 'uniform' } },
      { binding: B.dispatch, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 } },
      storage(B.positions), storage(B.state), storage(B.status), storage(B.topo, true), storage(B.rest, true)];
    if (this.options.contact) entries.push(storage(B.contact, true));
    const layout = d.createBindGroupLayout({ label: 'Cloth solver bindings', entries });
    const buffer = (binding: number, b: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding, resource: { buffer: b, size } });
    this.bindGroup = d.createBindGroup({ label: 'Cloth solver', layout, entries: [
      buffer(B.params, this.params), buffer(B.dispatch, this.dispatch, 32), buffer(B.positions, this.positions), buffer(B.state, this.state),
      buffer(B.status, this.status), buffer(B.topo, this.topo), buffer(B.rest, this.rest),
      ...(this.options.contact ? [buffer(B.contact, this.options.contact.buffer)] : [])] });
    const module = d.createShaderModule({ label: 'Cloth solver', code: clothSolverWGSL({ contactBlock: this.options.contact?.wgsl }) });
    const pipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const built = await Promise.all(KERNELS.map(entryPoint => d.createComputePipelineAsync({ label: `Cloth ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(KERNELS.map((n, i) => [n, built[i]!])) as Record<Kernel, GPUComputePipeline>;
  }

  private run(pass: GPUComputePassEncoder, name: Kernel, workgroups: number, record = 0): void {
    if (workgroups <= 0) return;
    pass.setPipeline(this.pipelines[name]);
    pass.setBindGroup(0, this.bindGroup, [CLOTH_DISPATCH_STRIDE * record]);
    pass.dispatchWorkgroups(workgroups);
  }
  private threads(n: number): number { return Math.ceil(n / CLOTH_WORKGROUP); }

  /** Encode exactly one implicit step of length dt into `pass`. */
  encodeStep(pass: GPUComputePassEncoder): void {
    const C = this.tables.colourRanges.length, limited = this.tables.strainLimited;
    this.run(pass, 'beginStep', 1);
    if (limited || this.tables.damped) this.run(pass, 'frameElements', this.threads(Math.max(limited ? this.E : 0, this.tables.damped ? Math.max(this.T, this.model.topology.hingeCount) : 0)));
    this.run(pass, 'frame', this.threads(this.N));
    this.hooks.afterFrame?.(pass);
    for (let it = 1; it <= this.iterations; it++) {
      const base = (it - 1) * (C + 1);
      for (let c = 0; c < C; c++) {
        this.run(pass, 'solveColour', this.threads(this.tables.colourRanges[c]![1]), base + c);
        this.hooks.afterColour?.(pass, it, c);
      }
      this.run(pass, 'postIteration', this.threads(this.N), base + C);
      if (limited) this.run(pass, 'edgeDualUpdate', this.threads(this.E), base + C);
      this.hooks.afterIteration?.(pass, it);
    }
    this.hooks.beforeFinish?.(pass);
    if (this.diagnostics) {
      this.run(pass, 'diagnose', this.threads(this.N));
      this.run(pass, 'reduceDiagnostics', 1);
    }
    this.run(pass, 'finishStep', this.threads(this.N));
  }

  /** Encode `steps` steps in one compute pass of `encoder`. */
  encodeSteps(encoder: GPUCommandEncoder, steps = 1): void {
    const pass = encoder.beginComputePass({ label: 'Cloth step' });
    for (let s = 0; s < steps; s++) this.encodeStep(pass);
    pass.end();
  }

  step(steps = 1): void {
    const encoder = this.device.createCommandEncoder({ label: 'Cloth step' });
    this.encodeSteps(encoder, steps);
    this.device.queue.submit([encoder.finish()]);
  }

  /** Replace the dynamic state (load, undo, edit remap). World metres. Resets status and the step count. */
  uploadState(positions_m: ArrayLike<number>, velocities_m_s?: ArrayLike<number>, solverState?: ClothSolverStateData): void {
    const N = this.N, o = this.options.origin_m, q = this.device.queue, L = this.tables;
    if (positions_m.length !== 3 * N) throw new Error('Cloth state needs 3 position values per vertex');
    if (velocities_m_s && velocities_m_s.length !== 3 * N) throw new Error('Cloth state needs 3 velocity values per vertex');
    const x = new Float32Array(4 * L.stride), state = new Float32Array(4 * L.stateRecords);
    const at = (s: number, i: number) => 4 * (s * L.stride + i);
    for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) {
      x[4 * i + c] = positions_m[3 * i + c]! - o[c]!;
      state[at(CLOTH_STATE.velocity, i) + c] = velocities_m_s?.[3 * i + c] ?? 0;
    }
    if (solverState) {
      if (solverState.previousVelocities_m_s.length !== 3 * N || solverState.previousDisplacements_m.length !== 3 * N) throw new Error('Cloth solver state must cover every vertex');
      if (solverState.edgeDuals.length !== 2 * this.E) throw new Error('Cloth solver state needs (lambda, k) per edge');
      let p = 0;
      for (let i = 0; i < N; i++) {
        for (let c = 0; c < 3; c++) {
          state[at(CLOTH_STATE.vPrev, i) + c] = solverState.previousVelocities_m_s[3 * i + c]!;
          state[at(CLOTH_STATE.dxPrev, i) + c] = solverState.previousDisplacements_m[3 * i + c]!;
        }
        if (this.model.topology.groups[i]! > 0) {
          if (4 * p + 4 > solverState.pinDuals.length) throw new Error('Cloth solver state is missing pin duals');
          state.set(solverState.pinDuals.slice(4 * p, 4 * p + 4), at(CLOTH_STATE.pinDual, i)); p++;
        }
      }
      if (4 * p !== solverState.pinDuals.length) throw new Error('Cloth solver state has pin duals for a different pin set');
      for (let e = 0; e < this.E; e++) state.set([solverState.edgeDuals[2 * e]!, solverState.edgeDuals[2 * e + 1]!], 4 * (L.edgeDualRecord + e));
    }
    state.set([0, solverState?.previousStep_s ?? 0, this.options.dt_s, 0], 4 * L.clockRecord);
    writeBufferChunked(q, this.positions, 0, x); writeBufferChunked(q, this.state, 0, state);
    this.timeBase = solverState?.time_s ?? 0;
    const params = L.params.slice();
    new Float32Array(params.buffer)[PARAM_TIME] = this.timeBase;
    q.writeBuffer(this.params, 0, params);
    q.writeBuffer(this.status, 0, new Uint32Array(CLOTH_STATUS_WORDS));
  }

  private async read(buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const size = align(Math.max(bytes, 4), 4);
    const staging = this.device.createBuffer({ label: 'Cloth readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap(); staging.destroy();
    return out;
  }

  /** World positions (harness, picking, save; never a step path). */
  async readPositions(): Promise<Float64Array> {
    const x = new Float32Array(await this.read(this.positions, 16 * this.N)), o = this.options.origin_m, out = new Float64Array(3 * this.N);
    for (let i = 0; i < this.N; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = x[4 * i + c]! + o[c]!;
    return out;
  }

  /** The full dynamic state for save, load and undo. */
  async readState(): Promise<ClothGPUState> {
    const N = this.N, L = this.tables;
    const [xb, sb] = await Promise.all([this.read(this.positions, 16 * N), this.read(this.state, 16 * L.stateRecords)]);
    const x = new Float32Array(xb), s = new Float32Array(sb), o = this.options.origin_m, at = (seg: number, i: number) => 4 * (seg * L.stride + i);
    const positions = new Float64Array(3 * N), velocities = new Float64Array(3 * N), vPrev: number[] = [], dxPrev: number[] = [], pinDuals: number[] = [];
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        positions[3 * i + c] = x[4 * i + c]! + o[c]!; velocities[3 * i + c] = s[at(CLOTH_STATE.velocity, i) + c]!;
        vPrev.push(s[at(CLOTH_STATE.vPrev, i) + c]!); dxPrev.push(s[at(CLOTH_STATE.dxPrev, i) + c]!);
      }
      if (this.model.topology.groups[i]! > 0) pinDuals.push(...s.subarray(at(CLOTH_STATE.pinDual, i), at(CLOTH_STATE.pinDual, i) + 4));
    }
    const edgeDuals: number[] = [];
    for (let k = 0; k < this.E; k++) edgeDuals.push(s[4 * (L.edgeDualRecord + k)]!, s[4 * (L.edgeDualRecord + k) + 1]!);
    const clock = s.subarray(4 * L.clockRecord, 4 * L.clockRecord + 4);
    return { positions_m: positions, velocities_m_s: velocities, solverState: { previousVelocities_m_s: vPrev, previousDisplacements_m: dxPrev,
      previousStep_s: clock[1]!, time_s: Math.fround(this.timeBase) + clock[0]! * this.options.dt_s, pinDuals, edgeDuals } };
  }

  async readStatus(): Promise<ClothStatusReport> {
    const words = new Uint32Array(await this.read(this.status, 4 * CLOTH_STATUS_WORDS)), floats = new Float32Array(words.buffer), slots: ClothStatusSlot[] = [];
    for (let s = 0; s < CLOTH_STATUS_RING_SLOTS; s++) {
      const b = CLOTH_STATUS_RING_BASE + s * CLOTH_STATUS_SLOT_WORDS;
      if (!words[b + CLOTH_SLOT.step]) continue;
      slots.push({ step: words[b + CLOTH_SLOT.step]! - 1, bdf2: words[b + CLOTH_SLOT.bdf2] === 1, residual: floats[b + CLOTH_SLOT.residual]!,
        maxStrain: floats[b + CLOTH_SLOT.maxStrain]!, kinetic_J: floats[b + CLOTH_SLOT.kinetic]!, potential_J: floats[b + CLOTH_SLOT.potential]!, elastic_J: floats[b + CLOTH_SLOT.elastic]! });
    }
    slots.sort((a, b) => a.step - b.step);
    return { status: words[0]!, steps: words[1]!, slots };
  }

  static describeStatus(status: number): string[] {
    return status & CLOTH_STATUS.nonFinite ? ['A cloth position or velocity became non-finite'] : [];
  }

  destroy(): void { for (const b of [this.positions, this.state, this.status, this.topo, this.rest, this.params, this.dispatch]) b.destroy(); }
}
