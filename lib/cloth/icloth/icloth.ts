/**
 * I-Cloth collision handling (Tang et al. 2018, "I-Cloth: Incremental Collision Handling for GPU-Based Interactive
 * Cloth Simulation", ACM TOG 37(6) 204) around the Newton-PCG solver's one implicit step:
 *
 *   afterFrame    proximity detection at x_t: vertex-face and edge-edge pairs within the proximity range and a plane
 *                 per solid in reach become records the solve reads as penalty repulsion (kc / 2 (r - C)^2) and
 *                 lagged Coulomb friction (iclothProximityBlockWGSL; the paper's step 1 and its terms in step 2).
 *   beforeFinish  rounds of: continuous collision detection of the linear path x_t -> x (vertex-face and edge-edge
 *                 cubics, vertices marched through the solids' fields), then the impact-zone solve over every vertex
 *                 of an impact (augmented-Lagrangian gradient descent, Algorithm 3), until a round finds no impact.
 *                 Round 0 tests every pair; later rounds only pairs with a vertex the zones moved (Algorithm 1).
 *                 A last full detection counts what is left.
 *
 * Everything is decided on the GPU: the dispatches of `rounds` x `impactIterations` are encoded up front and return at
 * their first read once the step is clean or the solve has converged. Nothing is carried between steps (records,
 * multipliers and flags are rebuilt each step), so a saved pose needs no state of this stage.
 *
 * Buffers: `cons` (u32, atomic here; the solver binds it read-only for the proximity records) holds the control
 * words, vertex flags, the proximity lists and records, the impact lists and records and the spatial hash; `work`
 * (vec4f) the solve's iterates and partial sums; `solid` the field images and the frame's pose table; `mesh` the
 * triangles with their ownership masks and the vertices' masses and pins. Pools are sized from the mesh at build; an
 * overflow sets a failing status bit, never a silent truncation.
 */
import { writeBufferChunked } from '../../methods/uniform/uniform-chunked-upload';
import type { ClothSolidField } from '../contact/solid-field';
import type { ClothSolverProximity, ClothStepHooks } from '../gpu-solver';
import type { PreparedClothTopology } from '../topology';
import {
  ICLOTH_BINDING, ICLOTH_CTL, ICLOTH_CTL_WORDS, ICLOTH_MAX_ROUNDS, ICLOTH_MODE, ICLOTH_MODE_COUNT_ONLY, ICLOTH_MODE_LAST, ICLOTH_PARAM_WORDS, ICLOTH_PARTIALS, ICLOTH_RECORD_WORDS,
  ICLOTH_SOLVER, ICLOTH_STATUS, ICLOTH_WG, ICLOTH_WGSL, ICLOTH_WORK, iclothProximityBlockWGSL,
} from './icloth-wgsl';

export { ICLOTH_FAILING, ICLOTH_STATUS } from './icloth-wgsl';

export interface IClothBodyPose {
  position: readonly [number, number, number];
  /** Unit quaternion (x, y, z, w). */
  orientation: readonly [number, number, number, number];
}

export interface IClothConfig {
  /** The cloth's contact radius r (m): proximity repulsion acts within it. */
  readonly r: number;
  /** The repulsion stiffness kc (N/m). */
  readonly kc: number;
  /** Coulomb friction of cloth on cloth (0: frictionless). */
  readonly selfFriction: number;
  readonly dt_s: number;
  /** Friction's regularisation speed (m/s): below it a sticking contact creeps. */
  readonly epsV_m_s: number;
  /** Detection + impact-zone rounds per step (cap). */
  readonly rounds: number;
  /** Gradient-descent iterations per impact-zone solve (cap). */
  readonly impactIterations: number;
  /** Most steps one frame encodes (the moving solid's pose table holds one more pose). */
  readonly stepsPerFrame: number;
  /** Whether the scene has a moving solid (its field and poses must then be set). */
  readonly body: boolean;
  /** A/B and tests only: false drops the proximity records (no repulsion, no friction). Default on. */
  readonly proximity?: boolean;
  /** A/B and tests only: false drops the collision rounds; the closing detection still counts what a step leaves. Default on. */
  readonly collisions?: boolean;
  /** Pool sizes per vertex / triangle; the defaults are sized for layered folds. */
  readonly proximityEntriesPerVertex?: number;
  readonly proximityRecordsPerVertex?: number;
  readonly impactEntriesPerVertex?: number;
  readonly impactRecordsPerVertex?: number;
  readonly hashEntriesPerTriangle?: number;
  readonly label?: string;
}

/** The impacts' target separation, the end-distance tolerance of the tests and the root test's distance, from r. */
export function iclothTolerances(r: number): { delta: number; dTol: number; eta: number; stepThreshold: number } {
  // The solve separates an impact to delta; the tests report a pair that ends within delta / 2 (so a solve's result,
  // at 0.75 delta or more, is clear of them by 2500 ulps of a 1 m coordinate at r = 1 mm), and a coplanarity root
  // where the features are within delta / 20. A step below delta / 100 ends the solve (the paper's step threshold).
  const delta = 0.1 * r;
  return { delta, dTol: 0.5 * delta, eta: 0.05 * delta, stepThreshold: 0.01 * delta };
}

/** Chebyshev's omega of iteration k (the paper: 1 before iteration 10, then the recurrence at rho); 1 with rho 0 (ICLOTH_SOLVER). */
export function iclothOmega(k: number): number {
  const { rho, chebyshevStart: s } = ICLOTH_SOLVER as { rho: number; chebyshevStart: number };
  if (!(rho > 0) || k < s) return 1;
  let w = 2 / (2 - rho * rho);
  for (let i = s + 1; i <= k; i++) w = 4 / (4 - rho * rho * w);
  return w;
}

/** Triangle ownership masks: bits 0-2 the corners this triangle tests (a vertex's first triangle), bits 3-5 the edges
 * (corner i to i + 1; an edge's lowest triangle). */
export function iclothOwnershipMasks(faces: Uint32Array, vertexCount: number): Uint32Array {
  const T = faces.length / 3, masks = new Uint32Array(T), seen = new Uint8Array(vertexCount), edges = new Set<number>();
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < 3; i++) {
      const v = faces[3 * t + i]!, u = faces[3 * t + (i + 1) % 3]!;
      if (!seen[v]) { seen[v] = 1; masks[t]! |= 1 << i; }
      const key = Math.min(u, v) * vertexCount + Math.max(u, v);
      if (!edges.has(key)) { edges.add(key); masks[t]! |= 8 << i; }
    }
  }
  return masks;
}

export interface IClothCounters {
  readonly status: number;
  /** The guard's latch: a step failed and every later dispatch of the stage returned at once. */
  readonly halted: boolean;
  readonly proximityPairs: number;
  readonly proximitySolid: number;
  /** Rounds of this step that found impacts. */
  readonly rounds: number;
  /** Impacts found per round of this step. */
  readonly impactsPerRound: number[];
  readonly zoneVerticesPerRound: number[];
  /** Gradient-descent iterations of this step, over its rounds. */
  readonly impactIterations: number;
  /** Impacts the closing verification found (0: the step ended penetration-free by these tests). */
  readonly remainingPairs: number;
  readonly remainingSolid: number;
  readonly impactRecords: number;
  readonly maxViolation_m: number;
  readonly steps: number;
  readonly cappedSolves: number;
  readonly stalledInfeasible: number;
  readonly degenerate: number;
}

type Kernel = 'icStepBegin' | 'icCcdBegin' | 'icHashClear' | 'icHashInsert' | 'icPairs' | 'icSolid' | 'icZoneInit' | 'icGrad' | 'icTrial' | 'icCommit' | 'icContactState';
const KERNELS: readonly Kernel[] = ['icStepBegin', 'icCcdBegin', 'icHashClear', 'icHashInsert', 'icPairs', 'icSolid', 'icZoneInit', 'icGrad', 'icTrial', 'icCommit', 'icContactState'];
const PROXIMITY_KERNELS: readonly Kernel[] = ['icStepBegin', 'icHashClear', 'icHashInsert', 'icPairs', 'icSolid'];
const RECORD_STRIDE = 256, SOLID_HEAD_WORDS = 64, POSE_WORDS = 16;
const align = (n: number, a: number) => Math.ceil(n / a) * a;

export class IClothCollision {
  readonly N: number; readonly T: number;
  /** The constraint buffer (control words first: the status word is at byte 0). */
  readonly cons: GPUBuffer;
  /** Per vertex: bit 0 in contact at the last step's end, bit 1 an onset in it (icContactState; the solver's BDF2 restart). */
  readonly contactState: GPUBuffer;
  private readonly work: GPUBuffer; private readonly mesh: GPUBuffer; private readonly params: GPUBuffer; private readonly records: GPUBuffer;
  private solid: GPUBuffer;
  private readonly paramWords = new Uint32Array(ICLOTH_PARAM_WORDS);
  private pipelines!: Record<Kernel, GPUComputePipeline>;
  private proximityPipelines: Partial<Record<Kernel, GPUComputePipeline>> = {};
  private layout!: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private bound?: { positions: GPUBuffer; state: GPUBuffer };
  private staticField?: ClothSolidField; private bodyField?: ClothSolidField;
  private posesPending = false;
  /** The step of the frame being encoded (the pose table's index). */
  private frameStep = 0;
  private destroyed = false;
  readonly layoutWords: { flags: number; proxLists: number; proxRecords: number; impactLists: number; impactRecords: number; hashHeads: number; hashPool: number; total: number };
  readonly capacity: { proxEntries: number; proxRecords: number; impactEntries: number; impactRecords: number; hashHeads: number; hashEntries: number };
  readonly tolerances: ReturnType<typeof iclothTolerances>;
  private readonly workgroups: number;
  private readonly label: string;

  private constructor(private readonly device: GPUDevice, topology: PreparedClothTopology, readonly config: IClothConfig) {
    const N = this.N = topology.vertexCount, T = this.T = topology.faceCount;
    this.label = config.label ?? 'icloth';
    if (!(config.r > 0 && config.kc > 0 && config.dt_s > 0 && config.epsV_m_s > 0 && config.selfFriction >= 0)) throw new Error('I-Cloth needs positive r, kc, dt and friction regularisation, and a non-negative friction');
    if (!(Number.isInteger(config.rounds) && config.rounds >= 1 && config.rounds <= ICLOTH_MAX_ROUNDS)) throw new Error(`I-Cloth rounds must be an integer in 1..${ICLOTH_MAX_ROUNDS}, got ${String(config.rounds)}`);
    if (!(Number.isInteger(config.impactIterations) && config.impactIterations >= 1 && config.impactIterations <= 1000)) throw new Error(`I-Cloth impactIterations must be an integer in 1..1000, got ${String(config.impactIterations)}`);
    if (!(Number.isInteger(config.stepsPerFrame) && config.stepsPerFrame >= 1)) throw new Error('I-Cloth needs the most steps a frame encodes');
    let maxEdge = 0, mass = 0, free = 0;
    for (const l of topology.edgeRestLength_m) maxEdge = Math.max(maxEdge, l);
    // The least rest altitude: how close a vertex and a face next to it, or two edges of adjacent faces, already are.
    let minAltitude = Infinity;
    for (let t = 0; t < T; t++) {
      let longest = 0;
      for (let i = 0; i < 3; i++) {
        const a = topology.faces[3 * t + i]!, b = topology.faces[3 * t + (i + 1) % 3]!, X = topology.data.restPositions_m;
        longest = Math.max(longest, Math.hypot(X[3 * a]! - X[3 * b]!, X[3 * a + 1]! - X[3 * b + 1]!, X[3 * a + 2]! - X[3 * b + 2]!));
      }
      minAltitude = Math.min(minAltitude, 2 * topology.faceArea_m2[t]! / longest);
    }
    for (let v = 0; v < N; v++) if (!(topology.groups[v]! > 0)) { mass += topology.mass_kg[v]!; free++; }
    this.tolerances = iclothTolerances(config.r);
    const range = 2 * config.r;
    // Proximity pairs are every non-incident pair within the range, so the range must stay below the mesh's own
    // feature distances (a vertex and a face it is not part of are at least about an edge apart at rest).
    if (!(range < minAltitude)) throw new Error(`I-Cloth's proximity range ${range} m (2 r) must be below the mesh's least rest altitude ${minAltitude} m: the mesh is finer than the cloth is thick`);
    // Two layers lying within the range give a vertex about 50 records (its face tests both ways and every pair of
    // its ring's edges with the ring under it), three layers 100; the 150^2 rotating sphere's hem, folded on the
    // floor, reaches 124 records and over 64 impacts at one vertex (measured): lists of 256 and 128, and 40 and 16
    // records per vertex (each record is in up to four lists).
    const KP = config.proximityEntriesPerVertex ?? 256, capP = (config.proximityRecordsPerVertex ?? 40) * N;
    const KI = config.impactEntriesPerVertex ?? 128, capI = (config.impactRecordsPerVertex ?? 16) * N;
    let heads = 1024; while (heads < 4 * T) heads *= 2;
    const capH = (config.hashEntriesPerTriangle ?? 16) * T;
    const flags = ICLOTH_CTL_WORDS, proxLists = flags + N, proxRecords = proxLists + N * (KP + 1), impactLists = proxRecords + ICLOTH_RECORD_WORDS * capP;
    const impactRecords = impactLists + N * (KI + 1), hashHeads = impactRecords + ICLOTH_RECORD_WORDS * capI, hashPool = hashHeads + heads, total = hashPool + 3 * capH;
    this.layoutWords = { flags, proxLists, proxRecords, impactLists, impactRecords, hashHeads, hashPool, total };
    this.capacity = { proxEntries: KP, proxRecords: capP, impactEntries: KI, impactRecords: capI, hashHeads: heads, hashEntries: capH };
    if (4 * total > device.limits.maxStorageBufferBindingSize) throw new Error(`I-Cloth's constraint buffer needs ${4 * total} bytes; the device binds ${device.limits.maxStorageBufferBindingSize}`);
    this.workgroups = Math.ceil(N / ICLOTH_WG);
    const stride = align(N, 16), partials = ICLOTH_WORK.segments * stride;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (name: string, bytes: number, usage = storage) => device.createBuffer({ label: `${this.label} ${name}`, size: Math.max(256, align(bytes, 256)), usage });
    this.cons = make('constraints', 4 * total);
    this.work = make('work', 16 * (partials + ICLOTH_PARTIALS * this.workgroups));
    this.mesh = make('mesh', 4 * (4 * T + 2 * N));
    this.contactState = make('contact state', 4 * N);
    this.params = make('params', 4 * ICLOTH_PARAM_WORDS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.solid = make('solids', 4 * (SOLID_HEAD_WORDS + POSE_WORDS * (config.stepsPerFrame + 1)));
    // Mesh: triangles (v0, v1, v2, ownership mask), then vertices (mass as f32 bits, flags: bit 0 pinned).
    const meshWords = new Uint32Array(4 * T + 2 * N), meshF = new Float32Array(meshWords.buffer), masks = iclothOwnershipMasks(topology.faces, N);
    for (let t = 0; t < T; t++) meshWords.set([topology.faces[3 * t]!, topology.faces[3 * t + 1]!, topology.faces[3 * t + 2]!, masks[t]!], 4 * t);
    for (let v = 0; v < N; v++) { meshF[4 * T + 2 * v] = topology.mass_kg[v]!; meshWords[4 * T + 2 * v + 1] = topology.groups[v]! > 0 ? 1 : 0; }
    writeBufferChunked(device.queue, this.mesh, 0, meshWords);
    // Dispatch records: per step of a frame, per round (the proximity pass, the rounds, the verification), per iteration.
    const K = config.impactIterations, perRound = K + 1, perStep = (config.rounds + 2) * perRound;
    const data = new ArrayBuffer(RECORD_STRIDE * config.stepsPerFrame * perStep);
    for (let s = 0; s < config.stepsPerFrame; s++) for (let r = 0; r < config.rounds + 2; r++) for (let k = 0; k <= K; k++) {
      const at = RECORD_STRIDE * ((s * (config.rounds + 2) + r) * perRound + k);
      // r = 0: proximity; 1..rounds: CCD round r - 1; rounds + 1: the verification (cell sizes of round index `rounds`).
      const mode = r === 0 ? ICLOTH_MODE.proximity : r <= config.rounds ? ICLOTH_MODE.ccd : ICLOTH_MODE.verify;
      // The verification's cells are sized by the last round's result (round index `rounds`; 0 without rounds).
      const round = r === 0 ? 0 : mode === ICLOTH_MODE.verify && config.collisions === false ? 0 : r - 1;
      // With the collision rounds switched off (a stage switch of the tests) the closing detection only counts.
      const countOnly = mode === ICLOTH_MODE.verify && config.collisions === false ? ICLOTH_MODE_COUNT_ONLY : 0;
      new Uint32Array(data, at, 4).set([round, k, s, mode | (k === K ? ICLOTH_MODE_LAST : 0) | countOnly]);
      new Float32Array(data, at + 16, 4).set([iclothOmega(k), 0, 0, 0]);
    }
    this.records = make('dispatch records', data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    writeBufferChunked(device.queue, this.records, 0, new Uint8Array(data));
    const w = this.paramWords, f = new Float32Array(w.buffer);
    w.set([N, T, heads - 1, this.workgroups, KP, capP, KI, capI, flags, proxLists, proxRecords, impactLists, impactRecords, hashHeads, hashPool, capH]);
    // st (stride, x_t, y) is written at bind; sol at the fields' upload.
    w.set([0, 0, 0, config.rounds, 0, 0, SOLID_HEAD_WORDS, K, stride, partials, 4 * T, 0], 16);
    f.set([config.r, config.kc, this.tolerances.delta, this.tolerances.dTol,
      this.tolerances.eta, range, 1.5 * maxEdge, config.selfFriction,
      config.dt_s, config.epsV_m_s * config.dt_s, ICLOTH_SOLVER.mu, free ? mass / free : 1,
      ICLOTH_SOLVER.armijo, this.tolerances.stepThreshold, ICLOTH_SOLVER.feasible * this.tolerances.delta, 0], 28);
    device.queue.writeBuffer(this.cons, 0, new Uint32Array(ICLOTH_CTL_WORDS));
  }

  static async create(device: GPUDevice, topology: PreparedClothTopology, config: IClothConfig): Promise<IClothCollision> {
    const c = new IClothCollision(device, topology, config);
    await c.compile();
    return c;
  }

  private async compile(): Promise<void> {
    const d = this.device, B = ICLOTH_BINDING, C = GPUShaderStage.COMPUTE;
    const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } });
    this.layout = d.createBindGroupLayout({ label: `${this.label} bindings`, entries: [
      { binding: B.params, visibility: C, buffer: { type: 'uniform' } },
      { binding: B.record, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 } },
      storage(B.positions), storage(B.state, true), storage(B.mesh, true), storage(B.solid, true), storage(B.cons), storage(B.work), storage(B.contactState)] });
    const module = d.createShaderModule({ label: this.label, code: ICLOTH_WGSL });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const built = await Promise.all(KERNELS.map(entryPoint => d.createComputePipelineAsync({ label: `icloth ${entryPoint}`, layout, compute: { module, entryPoint } }).catch(async error => {
      const info = await module.getCompilationInfo();
      throw new Error(`icloth ${entryPoint}: ${info.messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n') || String(error)}`);
    })));
    this.pipelines = Object.fromEntries(KERNELS.map((n, i) => [n, built[i]!])) as Record<Kernel, GPUComputePipeline>;
    // The proximity pass's own pipeline objects, so stage timing tells it from the collision rounds (clothStageOf).
    const prox = await Promise.all(PROXIMITY_KERNELS.map(entryPoint => d.createComputePipelineAsync({ label: `icloth prox ${entryPoint}`, layout, compute: { module, entryPoint } })));
    this.proximityPipelines = Object.fromEntries(PROXIMITY_KERNELS.map((n, i) => [n, prox[i]!]));
  }

  /** The solver's proximity terms over this stage's records (ClothGPUSolverOptions.proximity). */
  solverProximity(binding: number): ClothSolverProximity {
    return { buffer: this.cons, state: this.contactState, wgsl: iclothProximityBlockWGSL({ binding, lists: this.layoutWords.proxLists, records: this.layoutWords.proxRecords,
      entriesPerVertex: this.capacity.proxEntries, r: this.config.r, kc: this.config.kc, epsU: this.config.epsV_m_s * this.config.dt_s }) };
  }

  /** The solver's buffers: positions (vec4 per vertex) and its state segments (x_t and the inertial target y). */
  bind(inputs: { positions: GPUBuffer; state: GPUBuffer; stride: number; xtSegment: number; ySegment: number }): void {
    this.bound = { positions: inputs.positions, state: inputs.state };
    this.paramWords.set([inputs.stride, inputs.xtSegment, inputs.ySegment], 16);
    this.writeParams();
    this.rebind();
  }
  private writeParams(): void { this.device.queue.writeBuffer(this.params, 0, this.paramWords); }
  private rebind(): void {
    if (!this.bound) return;
    const B = ICLOTH_BINDING, buffer = (binding: number, b: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding, resource: { buffer: b, size } });
    this.bindGroup = this.device.createBindGroup({ label: this.label, layout: this.layout, entries: [
      buffer(B.params, this.params), buffer(B.record, this.records, 32), buffer(B.positions, this.bound.positions), buffer(B.state, this.bound.state),
      buffer(B.mesh, this.mesh), buffer(B.solid, this.solid), buffer(B.cons, this.cons), buffer(B.work, this.work), buffer(B.contactState, this.contactState)] });
  }

  /** Lay the field images out behind the pose table (setup and solid edits only; never in a step path). */
  private uploadFields(): void {
    const table = SOLID_HEAD_WORDS + POSE_WORDS * (this.config.stepsPerFrame + 1);
    const staticBase = this.staticField ? table : 0, bodyBase = this.bodyField ? table + (this.staticField?.words.length ?? 0) : 0;
    const words = new Uint32Array(table + (this.staticField?.words.length ?? 0) + (this.bodyField?.words.length ?? 0));
    if (this.staticField) words.set(this.staticField.words, staticBase);
    if (this.bodyField) words.set(this.bodyField.words, bodyBase);
    if (4 * words.length > this.device.limits.maxStorageBufferBindingSize) throw new Error(`I-Cloth's solid fields need ${4 * words.length} bytes; the device binds ${this.device.limits.maxStorageBufferBindingSize}`);
    const old = this.solid;
    this.solid = this.device.createBuffer({ label: `${this.label} solids`, size: Math.max(256, align(4 * words.length, 256)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    writeBufferChunked(this.device.queue, this.solid, 0, words);
    old.destroy();
    this.paramWords[20] = staticBase; this.paramWords[21] = bodyBase;
    this.writeParams();
    this.rebind();
    this.posesPending = false;
  }
  private checkField(field: ClothSolidField, what: string): void {
    // The march reads the field out to the proximity reach; a missing brick must lie beyond it.
    if (!(field.band >= 4 * this.config.r)) throw new Error(`I-Cloth: ${what}'s band ${field.band} m is below 4 r = ${4 * this.config.r} m`);
  }
  /** The static solid field, in the cloth frame. */
  setSolidField(field: ClothSolidField): void { this.checkField(field, 'the solid field'); this.staticField = field; this.uploadFields(); }
  /** The moving solid's field, in its body frame. */
  setBodyField(field: ClothSolidField): void {
    if (!this.config.body) throw new Error('I-Cloth: built without a moving solid (config.body)');
    this.checkField(field, 'the moving solid'); this.bodyField = field; this.uploadFields();
  }
  /** The moving solid's poses at the step boundaries of the frame about to be encoded (steps + 1 of them). */
  setBodyPoses(poses: readonly IClothBodyPose[]): void {
    if (!this.config.body) throw new Error('I-Cloth: built without a moving solid (config.body)');
    if (poses.length < 2 || poses.length > this.config.stepsPerFrame + 1) throw new Error(`I-Cloth: ${poses.length} moving-solid poses for a frame; it takes 2 to ${this.config.stepsPerFrame + 1}`);
    const f = new Float32Array(POSE_WORDS * poses.length);
    poses.forEach((p, k) => {
      const [x, y, z, w] = p.orientation, n = Math.hypot(x, y, z, w);
      if (!(n > 0) || Math.abs(n - 1) > 1e-6 || !p.position.every(Number.isFinite)) throw new Error(`I-Cloth: moving-solid pose ${k} needs a finite position and a unit quaternion`);
      const b = POSE_WORDS * k;
      f.set(p.position, b);
      f.set([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], b + 4);
      f.set([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], b + 8);
      f.set([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], b + 12);
    });
    this.device.queue.writeBuffer(this.solid, 4 * SOLID_HEAD_WORDS, f);
    this.posesPending = true; this.posesCount = poses.length;
  }
  private posesCount = 0;

  /** Start of a frame's encoding: its steps index the pose table from 0. */
  encodeFrameBegin(): void {
    if (this.config.body) {
      if (!this.bodyField) throw new Error('I-Cloth: a scene with a moving solid needs its field (setBodyField) before a frame');
      if (!this.posesPending) throw new Error('I-Cloth: a frame with a moving solid needs its poses (setBodyPoses) before it begins');
    }
    this.posesPending = false; this.frameStep = 0;
  }

  private recordIndex(round: number, iteration: number): number {
    return ((this.frameStep * (this.config.rounds + 2) + round) * (this.config.impactIterations + 1) + iteration);
  }
  private run(pass: GPUComputePassEncoder, name: Kernel, workgroups: number, record: number, proximity = false): void {
    if (workgroups <= 0) return;
    if (!this.bindGroup) throw new Error('I-Cloth: not bound to a solver (bind)');
    pass.setPipeline(proximity ? this.proximityPipelines[name]! : this.pipelines[name]);
    pass.setBindGroup(0, this.bindGroup, [RECORD_STRIDE * record]);
    pass.dispatchWorkgroups(workgroups);
  }
  private encodeDetect(pass: GPUComputePassEncoder, record: number, proximity = false): void {
    const WT = Math.ceil(this.T / ICLOTH_WG);
    this.run(pass, 'icHashClear', Math.ceil(this.capacity.hashHeads / ICLOTH_WG), record, proximity);
    this.run(pass, 'icHashInsert', WT, record, proximity);
    this.run(pass, 'icPairs', WT, record, proximity);
    this.run(pass, 'icSolid', this.workgroups, record, proximity);
  }

  /** The step hooks the solver encodes: proximity detection after the frame, the collision rounds before the finish. */
  get stepHooks(): Required<Pick<ClothStepHooks, 'afterFrame' | 'beforeFinish'>> {
    return {
      afterFrame: pass => {
        if (this.frameStep >= this.config.stepsPerFrame) throw new Error(`I-Cloth: a frame encodes at most ${this.config.stepsPerFrame} steps`);
        if (this.config.body && this.frameStep + 2 > this.posesCount) throw new Error('I-Cloth: the frame has more steps than moving-solid poses');
        const record = this.recordIndex(0, 0);
        this.run(pass, 'icStepBegin', this.workgroups, record, true);
        if (this.config.proximity !== false) this.encodeDetect(pass, record, true);
        return pass;
      },
      beforeFinish: pass => {
        const { impactIterations: K } = this.config, W = this.workgroups, rounds = this.config.collisions === false ? 0 : this.config.rounds;
        this.run(pass, 'icCcdBegin', W, this.recordIndex(1, 0));
        for (let r = 0; r < rounds; r++) {
          const first = this.recordIndex(r + 1, 0);
          this.encodeDetect(pass, first);
          this.run(pass, 'icZoneInit', W, first);
          for (let k = 0; k < K; k++) { this.run(pass, 'icGrad', W, first + k); this.run(pass, 'icTrial', W, first + k); }
          this.run(pass, 'icGrad', W, first + K);
          this.run(pass, 'icCommit', W, first);
        }
        this.encodeDetect(pass, this.recordIndex(this.config.rounds + 1, 0));
        this.run(pass, 'icContactState', W, this.recordIndex(1, 0));
        this.frameStep++;
      },
    };
  }

  /** Tests only: the first round's detection alone (no solve), from the bound positions and x_t. */
  encodeDetectOnly(pass: GPUComputePassEncoder): void {
    const record = this.recordIndex(1, 0);
    this.run(pass, 'icStepBegin', this.workgroups, record);
    this.run(pass, 'icCcdBegin', this.workgroups, record);
    this.encodeDetect(pass, record);
  }
  private async readWords(offset: number, count: number): Promise<Uint32Array> {
    if (!count) return new Uint32Array(0);
    const staging = this.device.createBuffer({ size: 4 * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.cons, 4 * offset, staging, 0, 4 * count);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return w;
  }
  /** The step's impact records (harness reads): kind, vertices (-1: unused), plane and weights, and the multiplier. */
  async readImpacts(): Promise<{ kind: number; ids: [number, number, number, number]; n: [number, number, number]; d0: number; w: [number, number, number, number]; lambda: number }[]> {
    const count = Math.min((await this.readWords(ICLOTH_CTL.impactCount, 1))[0]!, this.capacity.impactRecords);
    const w = await this.readWords(this.layoutWords.impactRecords, ICLOTH_RECORD_WORDS * count), f = new Float32Array(w.buffer);
    return Array.from({ length: count }, (_, r) => {
      const b = ICLOTH_RECORD_WORDS * r, id = (i: number) => w[b + i] === 0xffffffff ? -1 : w[b + i]!;
      return { kind: w[b + 14]!, ids: [id(0), id(1), id(2), id(3)], n: [f[b + 4]!, f[b + 5]!, f[b + 6]!], d0: f[b + 7]!, w: [f[b + 8]!, f[b + 9]!, f[b + 10]!, f[b + 11]!], lambda: f[b + 12]! };
    });
  }
  /** The pose the time integration left before the step's first round with impacts (harness reads; xyz per vertex;
   * stale from an earlier step when this step had no impacts). */
  async readZoneStart(): Promise<Float32Array> {
    const stride = align(this.N, 16), staging = this.device.createBuffer({ size: 16 * this.N, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.work, 16 * ICLOTH_WORK.x0 * stride, staging, 0, 16 * this.N);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(staging.getMappedRange().slice(0)), out = new Float32Array(3 * this.N);
    staging.destroy();
    for (let v = 0; v < this.N; v++) out.set(f.subarray(4 * v, 4 * v + 3), 3 * v);
    return out;
  }
  /** The contact state at the last step's end in the detector's carried form (OgcCarriedState: its contact state is
   * bits 8-9 of `held`; I-Cloth keeps no reference positions), read with a pose so a run continued from it restarts
   * its BDF2 history as the straight run does. The copy is submitted now. */
  async readCarriedState(): Promise<{ reference: Float32Array; held: Uint32Array }> {
    const staging = this.device.createBuffer({ size: 4 * this.N, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.contactState, 0, staging, 0, 4 * this.N);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const held = new Uint32Array(staging.getMappedRange().slice(0)).map(s => (s & 3) << 8), reference = new Float32Array(4 * this.N);
    staging.destroy();
    for (let v = 0; v < this.N; v++) reference[4 * v + 3] = -1;
    return { reference, held };
  }
  /** A saved contact state (readCarriedState's `held`); one for another vertex count fails explicitly. */
  restoreContactState(held: ArrayLike<number>): void {
    if (held.length !== this.N) throw new Error(`I-Cloth contact state covers ${held.length} vertices, the document has ${this.N}`);
    this.device.queue.writeBuffer(this.contactState, 0, Uint32Array.from(held, f => (f >> 8) & 3));
  }

  async readProximity(): Promise<{ ids: [number, number, number, number]; n: [number, number, number]; d0: number; w: [number, number, number, number]; muN: number }[]> {
    const count = Math.min((await this.readWords(ICLOTH_CTL.proxCount, 1))[0]!, this.capacity.proxRecords);
    const w = await this.readWords(this.layoutWords.proxRecords, ICLOTH_RECORD_WORDS * count), f = new Float32Array(w.buffer);
    return Array.from({ length: count }, (_, r) => {
      const b = ICLOTH_RECORD_WORDS * r, id = (i: number) => w[b + i] === 0xffffffff ? -1 : w[b + i]!;
      return { ids: [id(0), id(1), id(2), id(3)], n: [f[b + 4]!, f[b + 5]!, f[b + 6]!], d0: f[b + 7]!, w: [f[b + 8]!, f[b + 9]!, f[b + 10]!, f[b + 11]!], muN: f[b + 15]! };
    });
  }

  /** The control words (harness and benchmark reads; never a step path). */
  async readCounters(): Promise<IClothCounters> {
    const staging = this.device.createBuffer({ size: 4 * ICLOTH_CTL_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.cons, 0, staging, 0, 4 * ICLOTH_CTL_WORDS);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(staging.getMappedRange().slice(0)), f = new Float32Array(w.buffer), C = ICLOTH_CTL;
    staging.destroy();
    const rounds = this.config.rounds;
    return { status: w[C.status]!, halted: w[C.halt]! !== 0, proximityPairs: w[C.proxPairs]!, proximitySolid: w[C.proxSolid]!, rounds: w[C.roundsUsed]!,
      impactsPerRound: Array.from(w.subarray(C.impactsRound, C.impactsRound + rounds)), zoneVerticesPerRound: Array.from(w.subarray(C.zoneVertices, C.zoneVertices + rounds)),
      impactIterations: w[C.gdIterations]!, remainingPairs: w[C.remainingPairs]!, remainingSolid: w[C.remainingSolid]!, impactRecords: w[C.impactCount]!,
      maxViolation_m: f[C.maxViolation]!, steps: w[C.steps]!, cappedSolves: w[C.cappedSolves]!, stalledInfeasible: w[C.stalledInfeasible]!, degenerate: w[C.degenerate]! };
  }

  /** Notes for a status word's bits. */
  static describe(status: number): string[] {
    const S = ICLOTH_STATUS, out: string[] = [];
    if (status & S.hashOverflow) out.push('I-Cloth: the spatial hash pool overflowed');
    if (status & S.cellSpan) out.push('I-Cloth: a triangle spans more than 4 hash cells on an axis');
    if (status & S.proxOverflow) out.push('I-Cloth: the proximity record pool overflowed');
    if (status & S.proxListOverflow) out.push('I-Cloth: a vertex has more proximity records than its list holds');
    if (status & S.impactOverflow) out.push('I-Cloth: the impact record pool overflowed');
    if (status & S.impactListOverflow) out.push('I-Cloth: a vertex has more impacts than its list holds');
    if (status & S.roundCap) out.push('I-Cloth: impacts remained after the last collision round of a step (the stage halted at that step)');
    if (status & S.iterationCap) out.push('I-Cloth: an impact-zone solve ran to its iteration cap');
    if (status & S.solidBounds) out.push('I-Cloth: a vertex left the solid field\'s lattice');
    if (status & S.nonFinite) out.push('I-Cloth: a non-finite position reached the collision stage');
    if (status & S.degenerate) out.push('I-Cloth: a colliding pair gave no normal (coincident features)');
    return out;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.contactState, this.cons, this.work, this.mesh, this.params, this.records, this.solid]) b.destroy();
  }
}
