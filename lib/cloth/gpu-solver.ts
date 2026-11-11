/**
 * GPU cloth solver: one implicit step per `encodeStep`, at the scene dt, by Newton-PCG on the step's objective
 * (gpu-newton-wgsl.ts; caps and tolerances in `newton`). There are no substeps. Every per-step stage runs on the GPU
 * and nothing is read back: the step clock, the BDF2 bootstrap and its contact restart, attachment scripts and the
 * Newton and PCG stopping tests are evaluated in WGSL. Status and diagnostics land in a ring that harnesses read
 * asynchronously.
 *
 * Storage buffers (the M1 Max grants 10 per stage; C2's contact adds three):
 *   positions  vec4f[N]  the iterate x, cloth-local metres (xyz; w free). Shared with render and C2.
 *   state      vec4f     per-vertex segments (stride Np = N padded to 16 records, 256 B):
 *                          0 velocity at the step start (C2 binds `state` as its velocity input)
 *                          1 x_t, 2 v_{n-1}, 3 dx_{n-1}, 4 y, 5 x^, 6 pin offset p_{n+1} + alpha (x_t - p_n),
 *                          7 pin dual (lambda xyz, k), 8 vertex normal (render), 9 mouse grab
 *                        then per edge (lambda, k, stabilized limit), the damping tangent at x_t
 *                        (per face f0, f1 with D11, D22; damped scenes only), diagnostics partials, and the clock record (step, previous step, h~, BDF2).
 *   status     u32       sticky status bits, steps completed, the contact restart's words, the diagnostics ring
 *   topo       u32       CSR vertex-face/edge adjacency, bending rows (j, w_j), triangles (v0, v1, v2, material),
 *                        canonical edges (v0 < v1, f0, f1) and the bending stencil, each segment 256 B aligned so
 *                        render and C2 bind slices of it (bound twice: as u32 and as vec4u records, `topo4`)
 *   rest       vec4f     face Dm^-1, (area, damping), bending row (k, kd), edge (L, eps),
 *                        vertex (m, elastic, length, group), anchor, materials, scripts, bending diagonal
 *   nw         vec4f     the Newton-PCG work buffer (gpu-newton-wgsl.ts)
 */
import { writeBufferChunked } from '../methods/uniform/uniform-chunked-upload';
import type { ClothIntegrator, ClothNewtonSettings, ClothSolverStateData, ClothVector } from './types';
import { CLOTH_INITIAL_GUESS, type ClothInitialGuess, type ClothSolverModel } from './oracle';
import { clothNewtonSegmentCount, CLOTH_CONTACT_LIN_RECORDS, CLOTH_NEWTON_SEGMENTS, CLOTH_NEWTON_TRIAL_RECORDS, CLOTH_PCG_PARTIALS, CLOTH_PCG_SCALARS, type ClothPCGLayout } from './gpu-newton-wgsl';
import {
  CLOTH_BINDING, CLOTH_DISPATCH_STRIDE, CLOTH_SCRIPT_KIND, CLOTH_SEGMENT_ALIGN, CLOTH_SLOT, CLOTH_STATE, CLOTH_STATE_SEGMENTS, CLOTH_STATUS,
  CLOTH_STATUS_RING_BASE, CLOTH_STATUS_RING_SLOTS, CLOTH_STATUS_WORD, CLOTH_STATUS_SLOT_WORDS, CLOTH_STATUS_WORDS, CLOTH_WORKGROUP, clothSolverWGSL,
} from './gpu-solver-wgsl';

/** Storage buffers per stage the solver binds (topo counts twice, as topo and topo4); C2's contact adds three (its
 * block, the line-search filter output and the contact state). */
export const CLOTH_SOLVER_STORAGE_BUFFERS = 7;

export interface ClothSolverContact {
  /** ogcVertexBlockWGSL({group: 0, binding: CLOTH_BINDING.contact}). */
  readonly wgsl: string;
  /** OgcContactDetector.solverBuffer. */
  readonly buffer: GPUBuffer;
  /** Unread since the projected arc (Pi_v is ogcProject); kept only while callers still pass it. */
  readonly alpha?: GPUBuffer;
  /** The detector's per-vertex contact state (OgcContactDetector.contactState; bit 1, the contact switched on in the
   * last step), which restarts the cloth's BDF2 history (finishStep, and the step after). */
  readonly state: GPUBuffer;
}

/** Points inside one step where another lane encodes work into the solver's
 * compute pass. They never advance time: C2's detection and truncation
 * rounds run here, between iterations of the one step. */
export interface ClothStepHooks {
  /** After the step frame (inertia target and initial guess are in place), before Newton iteration 1. */
  afterFrame?(pass: GPUComputePassEncoder): void;
  /** After the last Newton iteration, before the diagnostics and the step's end. */
  beforeFinish?(pass: GPUComputePassEncoder): void;
  /** After the PCG solve wrote the direction d (`newtonDirection`), before the line search; C2 writes the filter's alpha here. */
  filter?(pass: GPUComputePassEncoder): void;
  /** After each Newton iteration moved the iterate; C2 decides and runs re-detection rounds here. */
  afterNewtonStep?(pass: GPUComputePassEncoder): void;
  /** With contact: once per step after the last position update (and the diagnostics), before finishStep forms the
   * velocities and the BDF2 history; C2 writes the step's contact state here. */
  stepEnd?(pass: GPUComputePassEncoder): void;
}

export interface ClothGPUSolverOptions {
  /** World position of the cloth-local origin. */
  readonly origin_m: ClothVector;
  readonly dt_s: number;
  readonly gravity_m_s2: ClothVector;
  readonly integrator: ClothIntegrator;
  /** Newton-PCG caps and tolerances (tolerances default to CLOTH_NEWTON_DEFAULTS). */
  readonly newton: ClothNewtonOptions;
  /** Default 'inertial' (see ClothInitialGuess): the inertial target y, the centre of contact's "target" certificates. */
  readonly initialGuess?: ClothInitialGuess;
  /** World positions and velocities at the start (default: rest, zero). */
  readonly positions_m?: ArrayLike<number>;
  readonly velocities_m_s?: ArrayLike<number>;
  readonly solverState?: ClothSolverStateData;
  /** Write the diagnostics slot of every step (residual, strain, energies). */
  readonly diagnostics?: boolean;
  readonly contact?: ClothSolverContact;
  readonly hooks?: ClothStepHooks;
  readonly label?: string;
  /** 'newton', the only method (VBD and its two-level correction were removed in M4); any other value fails. */
  readonly method?: ClothSolverMethod;
}

export type ClothSolverMethod = 'newton';
export type ClothNewtonOptions = ClothNewtonSettings;
export const CLOTH_NEWTON_DEFAULTS = Object.freeze({ pcgTolerance: 1e-3, residualTolerance: 1e-4, velocityTolerance_m_s: 0, forcingMax: 0, trustRegion: 'off' as 'off' | 'hybrid', activeSet: 'on' as 'on' | 'off',
  contactLinearization: 'off' as 'off' | 'factorized' | 'blocks', contactLinearizationRecords: 32 });

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
  /** Steps restarted as implicit Euler after a contact onset since the state was uploaded (Newton with contact). */
  readonly restarts: number;
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
  readonly params: Uint32Array<ArrayBuffer>;
  readonly strainLimited: boolean;
  readonly damped: boolean;
  /** State layout, in vec4 records. */
  readonly stride: number;
  readonly edgeDualRecord: number;
  readonly dampFaceRecord: number;
  readonly partialRecord: number;
  readonly clockRecord: number;
  readonly stateRecords: number;
  /** vec4 records of the Newton-PCG `nw` buffer. */
  readonly newtonRecords: number;
  /** The compiled PCG layout (compileClothPCGPatches): the shaders' constants, the patch count (pcgStep workgroups). */
  readonly pcg: ClothPCGLayout & { readonly patches: number; readonly maxHalo: number };
}

const PARAM_WORDS = 72;
/** u32 words of the bending block (Params.bend: its diagonal's rest record in w). */
const PARAM_BEND = 44;
/** Float words of the params uniform: gravity xyz + dt, then the clock origin and the initial guess. */
const PARAM_GRAVITY = 36, PARAM_TIME = 40;
/** Float words of the mouse grab (Params.grab): displacement xyz, held flag. */
const PARAM_GRAB = 48;
/** u32 words of the Newton block (Params.newton0 .. newton4). */
const PARAM_NEWTON = 52;
const align = (n: number, a: number) => Math.ceil(n / a) * a;

/**
 * The bending stencil (Kirchhoff plate rows, topology `clothBendingRows`): bending is (k/2)|R_r x|^2 a row and its damping
 * (kd k / 2h~)|R_r (x - x^)|^2, so a vertex's bending gradient is sum_j Q_vj (x_j - x_v) (+ the damping counterpart
 * in u = x - x^ over h~) with Q = sum_r k w w^T and Qd = sum_r kd k w w^T over v's stencil neighbours j != v
 * (sum_j Q_vj = 0 a row, as the weights of a bending row sum to zero), and its Hessian (Q_vv + Qd_vv / h~) I. Per
 * vertex: CSR neighbours (j, Q_vj, Qd_vj) and the diagonal (Q_vv, Qd_vv), in Float64 at setup.
 */
export function clothBendingStencil(model: ClothSolverModel): { offsets: Uint32Array; columns: Uint32Array; q: Float64Array; qd: Float64Array; diagonal: Float64Array } {
  const t = model.topology, N = t.vertexCount;
  const rows: Map<number, [number, number]>[] = Array.from({ length: N }, () => new Map());
  const diagonal = new Float64Array(2 * N);
  for (let r = 0; r < t.bendRowCount; r++) {
    const k = t.bendStiffness_N_m3[r]!, kd = model.bendDamping_s[r]! * k, o0 = t.bendOffsets[r]!, o1 = t.bendOffsets[r + 1]!;
    for (let a = o0; a < o1; a++) {
      const va = t.bendVertices[a]!, wa = t.bendWeights[a]!;
      diagonal[2 * va] += k * wa * wa; diagonal[2 * va + 1] += kd * wa * wa;
      for (let b = o0; b < o1; b++) {
        const vb = t.bendVertices[b]!;
        if (b === a || vb === va) continue;
        const wb = t.bendWeights[b]!, entry = rows[va]!.get(vb) ?? [0, 0];
        entry[0] += k * wa * wb; entry[1] += kd * wa * wb; rows[va]!.set(vb, entry);
      }
    }
  }
  const offsets = new Uint32Array(N + 1);
  for (let v = 0; v < N; v++) offsets[v + 1] = offsets[v]! + rows[v]!.size;
  const columns = new Uint32Array(offsets[N]!), q = new Float64Array(offsets[N]!), qd = new Float64Array(offsets[N]!);
  rows.forEach((row, v) => {
    let o = offsets[v]!;
    for (const [j, [a, b]] of [...row].sort((x, y) => x[0] - y[0])) { columns[o] = j; q[o] = a; qd[o] = b; o++; }
  });
  return { offsets, columns, q, qd, diagonal };
}

/** Rows per PCG patch: one 256-thread workgroup per patch, one row per thread (gpu-newton-wgsl.ts, pcgStep). */
export const CLOTH_PCG_PATCH = 256;
/** Workgroup-memory vertices (vec3 as three f32 arrays) a patch can stage under a device's workgroup storage limit
 * (pcgStep also holds a 256-record reduction and two scalars): its rows and as much of its halo as fits; the rest of
 * a halo is read through global memory. */
export function clothPCGSharedCap(workgroupStorageBytes: number): number {
  const cap = Math.floor((workgroupStorageBytes - 16 * CLOTH_PCG_PATCH - 64) / 12 / 32) * 32;
  if (cap < CLOTH_PCG_PATCH) throw new Error(`Cloth PCG needs ${12 * CLOTH_PCG_PATCH + 16 * CLOTH_PCG_PATCH + 64} B of workgroup storage; the device grants ${workgroupStorageBytes}`);
  return cap;
}
/** u32 words per patch in the patch table: K1, K2, halo count, halo word, block record, bending record, column word, rows. */
export const CLOTH_PCG_PATCH_WORDS = 8;
/** A compiled column at or above this bit is a global vertex index (outside the staged halo). */
export const CLOTH_PCG_GLOBAL = 0x80000000;

/**
 * The compiled topology of the assembled H (built once per topology revision, CPU, edit time): the vertices in the
 * Morton order of their rest positions fall into patches of CLOTH_PCG_PATCH rows (compact tiles, so a patch's halo
 * is its border; the vertex numbering itself is unchanged, `rows` and `vertexSlot` map between the two). Row v holds, in sliced-ELL order (slot s of lane l at s * PATCH + l),
 * K1 3x3 blocks (slot 0 the diagonal, then v's 1-ring neighbours by index: membrane, damping, strain edges and the
 * bending entries of the 1-ring) and K2 scalar entries (the bending stencil's 2-ring entries beyond the 1-ring), K1, K2
 * the patch's largest. A column is a patch-local index (rows 0..PATCH-1, then the patch's halo of outside columns, as
 * far as the staged workgroup memory holds) or CLOTH_PCG_GLOBAL | v. Per incidence the assembly reads the slot that
 * each corner of a face (faceSlots: three bytes), the other end of an edge (edgeSlots) and each bending stencil record
 * (stencilDest: a block slot, or bit 30 and the 2-ring index) lands in.
 */
export interface ClothPCGPatches {
  readonly patches: number;
  readonly shared: number;
  /** The vertex of each patch slot (p * PATCH + lane), and the slot of each vertex. */
  readonly rows: Uint32Array;
  readonly vertexSlot: Uint32Array;
  /** Per patch: K1, K2, halo count, halo offset (into halo), rows; block and bending records relative to their regions. */
  readonly table: Uint32Array;
  readonly halo: Uint32Array;
  /** Per patch, K1 * PATCH block columns then K2 * PATCH bending columns, local-encoded; relative word offsets in table. */
  readonly columns: Uint32Array;
  readonly faceSlots: Uint32Array;
  readonly edgeSlots: Uint32Array;
  readonly stencilDest: Uint32Array;
  /** vec4 records of the assembled blocks (3 per slot) and of the bending entries (one per two slots). */
  readonly blockRecords: number;
  readonly bendRecords: number;
  /** Largest K1 and K2 over the patches, and the largest halo (diagnostics). */
  readonly maxK1: number; readonly maxK2: number; readonly maxHalo: number;
}

export function compileClothPCGPatches(model: ClothSolverModel, stencil: ReturnType<typeof clothBendingStencil>, sharedCap = clothPCGSharedCap(16384)): ClothPCGPatches {
  const t = model.topology, N = t.vertexCount, PATCH = CLOTH_PCG_PATCH, patches = Math.max(1, Math.ceil(N / PATCH));
  // 1-ring by index (faces and edges share it: every edge is a face edge).
  const ring: number[][] = Array.from({ length: N }, () => []);
  for (let f = 0; f < t.faceCount; f++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    const u = t.faces[3 * f + a]!, w = t.faces[3 * f + b]!;
    if (u !== w) ring[u]!.push(w);
  }
  for (let e = 0; e < t.edgeCount; e++) { const u = t.edges[4 * e]!, w = t.edges[4 * e + 1]!; ring[u]!.push(w); ring[w]!.push(u); }
  const slotOf: Map<number, number>[] = ring.map(r => new Map([...new Set(r)].sort((x, y) => x - y).map((j, i) => [j, i + 1])));
  const far: number[][] = Array.from({ length: N }, (_, v) => {
    const out: number[] = [];
    for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) { const j = stencil.columns[o]!; if (j !== v && !slotOf[v]!.has(j)) out.push(j); }
    return out;
  });
  const farIndex = far.map(list => new Map(list.map((j, i) => [j, i])));
  // Morton order (10 bits an axis) of the rest positions in their bounding box, ties by index.
  const x = t.data.restPositions_m, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < N; v++) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a]!, x[3 * v + a]!); hi[a] = Math.max(hi[a]!, x[3 * v + a]!); }
  const spread = (q: number) => { let r = 0; for (let b = 0; b < 10; b++) r |= ((q >> b) & 1) << (3 * b); return r >>> 0; };
  const code = new Uint32Array(N);
  for (let v = 0; v < N; v++) {
    let c = 0;
    for (let a = 0; a < 3; a++) {
      const span = hi[a]! - lo[a]!, q = span > 0 ? Math.min(1023, Math.floor(1024 * (x[3 * v + a]! - lo[a]!) / span)) : 0;
      c |= spread(q) << a;
    }
    code[v] = c >>> 0;
  }
  const rows = Uint32Array.from({ length: N }, (_, v) => v).sort((a, b) => code[a]! - code[b]! || a - b);
  const vertexSlot = new Uint32Array(N);
  rows.forEach((v, slot) => { vertexSlot[v] = slot; });
  const patchOf = (v: number) => Math.floor(vertexSlot[v]! / PATCH);
  const table = new Uint32Array(CLOTH_PCG_PATCH_WORDS * patches), halos: number[][] = [], local: Map<number, number>[] = [];
  let maxK1 = 1, maxK2 = 0, maxHalo = 0;
  for (let p = 0; p < patches; p++) {
    const r0 = p * PATCH, r1 = Math.min(N, r0 + PATCH);
    let K1 = 1, K2 = 0;
    const outside = new Set<number>();
    for (let s = r0; s < r1; s++) {
      const v = rows[s]!;
      K1 = Math.max(K1, 1 + slotOf[v]!.size); K2 = Math.max(K2, far[v]!.length);
      for (const j of [...slotOf[v]!.keys(), ...far[v]!]) if (patchOf(j) !== p) outside.add(j);
    }
    K2 += K2 & 1; // bending entries pack two to a record
    const halo = [...outside].sort((x, y) => x - y);
    halos.push(halo); maxK1 = Math.max(maxK1, K1); maxK2 = Math.max(maxK2, K2); maxHalo = Math.max(maxHalo, halo.length);
    table.set([K1, K2, halo.length, 0, 0, 0, 0, r1 - r0], CLOTH_PCG_PATCH_WORDS * p);
  }
  const shared = Math.min(sharedCap, align(PATCH + maxHalo, 32));
  let haloWords = 0, columnWords = 0, blockRecords = 0, bendRecords = 0;
  for (let p = 0; p < patches; p++) {
    const w = CLOTH_PCG_PATCH_WORDS * p, K1 = table[w]!, K2 = table[w + 1]!;
    table[w + 3] = haloWords; haloWords += halos[p]!.length;
    table[w + 4] = blockRecords; blockRecords += 3 * K1 * PATCH;
    table[w + 5] = bendRecords; bendRecords += (K2 / 2) * PATCH;
    table[w + 6] = columnWords; columnWords += (K1 + K2) * PATCH;
    const map = new Map<number, number>();
    halos[p]!.forEach((j, h) => { if (PATCH + h < shared) map.set(j, PATCH + h); });
    local.push(map);
  }
  const halo = new Uint32Array(Math.max(haloWords, 1)), columns = new Uint32Array(Math.max(columnWords, 1));
  for (let p = 0; p < patches; p++) {
    const w = CLOTH_PCG_PATCH_WORDS * p, K1 = table[w]!, K2 = table[w + 1]!, r0 = p * PATCH, count = table[w + 7]!, c0 = table[w + 6]!;
    halo.set(halos[p]!, table[w + 3]);
    const enc = (j: number) => patchOf(j) === p ? vertexSlot[j]! - r0 : local[p]!.get(j) ?? (CLOTH_PCG_GLOBAL | j) >>> 0;
    for (let l = 0; l < PATCH; l++) {
      // Padding slots point at the row itself (lane l, or lane 0 past the last row) with zero blocks and weights.
      const self = l < count ? l : 0;
      for (let k = 0; k < K1; k++) columns[c0 + k * PATCH + l] = self;
      for (let k = 0; k < K2; k++) columns[c0 + (K1 + k) * PATCH + l] = self;
      if (l >= count) continue;
      const v = rows[r0 + l]!;
      for (const [j, k] of slotOf[v]!) columns[c0 + k * PATCH + l] = enc(j);
      far[v]!.forEach((j, k) => { columns[c0 + (K1 + k) * PATCH + l] = enc(j); });
    }
  }
  const faceSlots = new Uint32Array(Math.max(t.vertexFaces.length, 1));
  for (let v = 0; v < N; v++) for (let j = t.vertexFaceOffsets[v]!; j < t.vertexFaceOffsets[v + 1]!; j++) {
    const f = t.vertexFaces[j]! >>> 2;
    let word = 0;
    for (let c = 0; c < 3; c++) { const u = t.faces[3 * f + c]!; word |= (u === v ? 0 : slotOf[v]!.get(u)!) << (8 * c); }
    faceSlots[j] = word;
  }
  const edgeSlots = new Uint32Array(Math.max(t.vertexEdges.length, 1));
  for (let v = 0; v < N; v++) for (let j = t.vertexEdgeOffsets[v]!; j < t.vertexEdgeOffsets[v + 1]!; j++) {
    const e = t.vertexEdges[j]! >>> 2, u = t.edges[4 * e]!, w = t.edges[4 * e + 1]!;
    edgeSlots[j] = slotOf[v]!.get(u === v ? w : u)!;
  }
  const stencilDest = new Uint32Array(Math.max(stencil.columns.length, 1));
  for (let v = 0; v < N; v++) for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) {
    const j = stencil.columns[o]!, k = slotOf[v]!.get(j);
    stencilDest[o] = k !== undefined ? k : (0x40000000 | farIndex[v]!.get(j)!) >>> 0;
  }
  if (maxK1 > 255) throw new Error(`Cloth vertex valence ${maxK1 - 1} exceeds the compiled stencil's 255 slots`);
  return { patches, shared, rows, vertexSlot, table, halo, columns, faceSlots, edgeSlots, stencilDest, blockRecords, bendRecords, maxK1, maxK2, maxHalo };
}

export function packClothGPUTables(model: ClothSolverModel, origin: ClothVector, options: Pick<ClothGPUSolverOptions, 'dt_s' | 'gravity_m_s2' | 'integrator' | 'initialGuess'> & { readonly contact?: unknown; readonly newton?: Pick<ClothNewtonSettings, 'contactLinearization' | 'contactLinearizationRecords'> }, time_s = 0, workgroupStorageBytes = 16384): ClothGPUTables {
  const t = model.topology, N = t.vertexCount, T = t.faceCount, E = t.edgeCount, M = model.materials.length, S = model.scripts.length;
  const triangles = new Uint32Array(4 * T);
  for (let f = 0; f < T; f++) triangles.set([t.faces[3 * f]!, t.faces[3 * f + 1]!, t.faces[3 * f + 2]!, t.faceMaterial[f]!], 4 * f);
  const stencil = clothBendingStencil(model);
  // The bending stencil as CSR records (j, Q_vj, Qd_vj, 0) in vertex order (Newton's H p).
  const stencilRecords = new Uint32Array(4 * Math.max(stencil.columns.length, 1)), stencilF = new Float32Array(stencilRecords.buffer);
  for (let o = 0; o < stencil.columns.length; o++) { stencilRecords[4 * o] = stencil.columns[o]!; stencilF[4 * o + 1] = stencil.q[o]!; stencilF[4 * o + 2] = stencil.qd[o]!; }
  // Bending rows relative to their own vertex: per row the offsets of its other entries, each a pair (j, w_j bits).
  const R = t.bendRowCount;
  const bendOffsets = Uint32Array.from(t.bendOffsets, (o, r) => o - Math.min(r, R)), bendEntries = new Uint32Array(2 * Math.max(t.bendVertices.length - R, 1)), bendF = new Float32Array(bendEntries.buffer);
  for (let r = 0, q = 0; r < R; r++) for (let o = t.bendOffsets[r]! + 1; o < t.bendOffsets[r + 1]!; o++, q++) { bendEntries[2 * q] = t.bendVertices[o]!; bendF[2 * q + 1] = t.bendWeights[o]!; }
  const pcg = compileClothPCGPatches(model, stencil, clothPCGSharedCap(workgroupStorageBytes));
  const parts = [t.vertexFaceOffsets, t.vertexFaces, bendOffsets, bendEntries, t.vertexEdgeOffsets, t.vertexEdges, triangles, t.edges, stencil.offsets, stencilRecords,
    pcg.table, pcg.halo, pcg.columns, pcg.faceSlots, pcg.edgeSlots, pcg.stencilDest, pcg.rows, pcg.vertexSlot];
  const topoBase: number[] = [];
  let words = 0;
  for (const p of parts) { words = align(words, 64); topoBase.push(words); words += p.length; }
  const topo = new Uint32Array(Math.max(align(words, 4), 4));
  parts.forEach((p, i) => topo.set(p, topoBase[i]));
  const restCounts = [T, T, R, E, N, N, 2 * M, 4 * S, N], restBase: number[] = [];
  let records = 0;
  for (const c of restCounts) { restBase.push(records); records += c; }
  const rest = new Float32Array(4 * Math.max(records, 1));
  for (let f = 0; f < T; f++) {
    rest.set(t.faceDmInv.subarray(4 * f, 4 * f + 4), 4 * (restBase[0]! + f));
    rest.set([t.faceArea_m2[f]!, model.faceDamping_s[f]!], 4 * (restBase[1]! + f));
  }
  for (let r = 0; r < R; r++) rest.set([t.bendStiffness_N_m3[r]!, model.bendDamping_s[r]!], 4 * (restBase[2]! + r));
  const diagonalRecord = restBase[8]!;
  for (let v = 0; v < N; v++) rest.set(stencil.diagonal.subarray(2 * v, 2 * v + 2), 4 * (diagonalRecord + v));
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
  for (let r = 0; r < R; r++) damped ||= model.bendDamping_s[r]! > 0;
  const stride = align(N, CLOTH_SEGMENT_ALIGN), workgroups = Math.ceil(N / CLOTH_WORKGROUP);
  const edgeDualRecord = CLOTH_STATE_SEGMENTS * stride, dampFaceRecord = edgeDualRecord + align(E, CLOTH_SEGMENT_ALIGN);
  const partialRecord = dampFaceRecord + (damped ? align(2 * T, CLOTH_SEGMENT_ALIGN) : 0);
  const clockRecord = partialRecord + align(2 * workgroups, CLOTH_SEGMENT_ALIGN), stateRecords = clockRecord + CLOTH_SEGMENT_ALIGN;
  const params = new Uint32Array(PARAM_WORDS), paramsF = new Float32Array(params.buffer);
  // Params: n, topo0 (vertex faces, bending rows), topo1 (vertex edges), topo2 (triangles, edges, stencil offsets), rest0, rest1, flags, state0, state1.
  params.set([N, T, E, N, ...topoBase.slice(0, 4), topoBase[4]!, topoBase[5]!, 0, 0, topoBase[6]!, topoBase[7]!, topoBase[8]!, 0, ...restBase.slice(0, 8),
    S, options.integrator === 'bdf2' ? 1 : 0, strainLimited ? 1 : 0, workgroups, stride, edgeDualRecord, partialRecord, clockRecord, dampFaceRecord, 0, damped ? 1 : 0, 0]);
  const guess = CLOTH_INITIAL_GUESS[options.initialGuess ?? 'inertial'];
  if (guess === undefined) throw new Error(`Unknown cloth initial guess ${String(options.initialGuess)}`);
  paramsF.set([...options.gravity_m_s2, options.dt_s, time_s, 0, guess, 0], PARAM_GRAVITY);
  params.set([0, 0, 0, diagonalRecord], PARAM_BEND);
  // nw: segments, face records, assembled blocks and bending entries, line-search partials, PCG partials and scalars, control.
  const newtonFaces = clothNewtonSegmentCount(!!options.contact) * stride, newtonBlocks = align(newtonFaces + 3 * T, CLOTH_SEGMENT_ALIGN);
  const newtonBends = newtonBlocks + pcg.blockRecords, newtonPartials = align(newtonBends + pcg.bendRecords, CLOTH_SEGMENT_ALIGN);
  const pcgPartials = align(newtonPartials + CLOTH_NEWTON_TRIAL_RECORDS * workgroups, CLOTH_SEGMENT_ALIGN);
  const pcgScalars = pcgPartials + CLOTH_PCG_PARTIALS * pcg.patches;
  // The compiled contact linearization (newton.contactLinearization, with contact): a header per vertex, a total per
  // patch, the pool; before the control block, which stays last.
  const linMode = options.contact ? options.newton?.contactLinearization ?? 'off' : 'off';
  if (linMode !== 'off' && CLOTH_CONTACT_LIN_RECORDS[linMode] === undefined) throw new Error(`Unknown cloth Newton contactLinearization ${String(linMode)}`);
  const linPerVertex = options.newton?.contactLinearizationRecords ?? CLOTH_NEWTON_DEFAULTS.contactLinearizationRecords;
  if (!(Number.isInteger(linPerVertex) && linPerVertex >= 1)) throw new Error(`Cloth Newton contactLinearizationRecords must be a positive integer, got ${String(linPerVertex)}`);
  const linHead = align(pcgScalars + CLOTH_PCG_SCALARS, CLOTH_SEGMENT_ALIGN), linPatch = linHead + stride, linPool = align(linPatch + pcg.patches, CLOTH_SEGMENT_ALIGN);
  const contactLin = linMode === 'off' ? undefined : { mode: linMode, head: linHead, patch: linPatch, pool: linPool, records: linPerVertex * N };
  const newtonControl = align(contactLin ? linPool + contactLin.records : pcgScalars + CLOTH_PCG_SCALARS, CLOTH_SEGMENT_ALIGN);
  // The patch table's relative offsets become absolute: halo and columns (topo words), blocks and bending entries (nw records).
  for (let p = 0; p < pcg.patches; p++) {
    const w = topoBase[10]! + CLOTH_PCG_PATCH_WORDS * p;
    topo[w + 3]! += topoBase[11]!; topo[w + 4]! += newtonBlocks; topo[w + 5]! += newtonBends; topo[w + 6]! += topoBase[12]!;
  }
  params.set([newtonFaces, newtonPartials, workgroups, newtonControl, topoBase[9]! / 4], PARAM_NEWTON);
  params.set([topoBase[10]!, topoBase[13]!, topoBase[14]!, topoBase[15]!, pcgPartials, pcg.patches, pcgScalars, 0, topoBase[16]!, topoBase[17]!, 0, 0], PARAM_NEWTON + 8);
  return { topo, rest, trianglesWord: topoBase[6]!, edgesWord: topoBase[7]!, params, strainLimited, damped, stride, edgeDualRecord,
    dampFaceRecord, partialRecord, clockRecord, stateRecords, newtonRecords: newtonControl + CLOTH_SEGMENT_ALIGN,
    pcg: { shared: pcg.shared, maxK1: pcg.maxK1, maxK2: pcg.maxK2, patches: pcg.patches, maxHalo: pcg.maxHalo, ...(contactLin ? { contactLin } : {}) } };
}

const NEWTON_KERNELS = ['nwBeginStep', 'nwFaces', 'nwGradient', 'pcgStep', 'pcgSolve', 'nwTrial', 'nwModel', 'nwChoose', 'nwApply', 'nwEdgeDuals'] as const;
/** With contact: the projected arc's trial moves (after the filter hook, before nwTrial). */
const CONTACT_NEWTON_KERNELS = ['nwProject'] as const;
/** With newton.contactLinearization: the compiled contact records' count and build (after nwGradient). */
const CONTACT_LIN_KERNELS = ['nwLinCount', 'nwLinBuild'] as const;
/** With contact: the BDF2 restart's scans of C2's contact state. */
const RESTART_KERNELS = ['restartScan', 'onsetScan'] as const;
type Kernel = 'grabBegin' | 'beginStep' | 'frameElements' | 'frame' | 'diagnose' | 'reduceDiagnostics' | 'finishStep' | typeof NEWTON_KERNELS[number] | typeof RESTART_KERNELS[number] | typeof CONTACT_NEWTON_KERNELS[number] | typeof CONTACT_LIN_KERNELS[number];
const KERNELS: readonly Kernel[] = ['grabBegin', 'beginStep', 'frameElements', 'frame', 'diagnose', 'reduceDiagnostics', 'finishStep', ...NEWTON_KERNELS];

export class ClothGPUSolver {
  readonly N: number; readonly T: number; readonly E: number;
  readonly positions: GPUBuffer; readonly state: GPUBuffer; readonly status: GPUBuffer; readonly topo: GPUBuffer;
  private readonly rest: GPUBuffer; private readonly params: GPUBuffer; private readonly dispatch: GPUBuffer;
  /** The Newton-PCG work buffer. */
  private readonly newtonWork: GPUBuffer;
  readonly method: ClothSolverMethod = 'newton';
  private readonly newton: Required<ClothNewtonOptions>;
  /** With contact: the BDF2 history restarts after a contact onset (restartScan, onsetScan). */
  private readonly restarts: boolean;
  readonly tables: ClothGPUTables;
  private pipelines!: Record<Kernel, GPUComputePipeline>;
  private bindGroup!: GPUBindGroup;
  /** Scene time at clock step 0 (load and undo restart the step count). */
  private timeBase = 0;
  diagnostics: boolean;
  hooks: ClothStepHooks;
  /** A grab set since the last step: the next step first takes its anchors from the live pose (grabBegin). */
  private grabAnchorsPending = false;
  private grabHeld = false;

  private constructor(private readonly device: GPUDevice, readonly model: ClothSolverModel, readonly options: ClothGPUSolverOptions) {
    const t = model.topology;
    this.N = t.vertexCount; this.T = t.faceCount; this.E = t.edgeCount;
    if (!(options.dt_s > 0)) throw new Error('Cloth solver needs a positive step');
    if (options.method !== undefined && options.method !== 'newton') throw new Error(`Unknown cloth solver method ${String(options.method)}`);
    const o = options.newton;
    if (!o || !(Number.isInteger(o.iterations) && o.iterations >= 1) || !(Number.isInteger(o.pcgIterations) && o.pcgIterations >= 1))
      throw new Error('Cloth Newton solver needs integer caps newton.iterations >= 1 and newton.pcgIterations >= 1');
    this.newton = { ...CLOTH_NEWTON_DEFAULTS, ...o };
    if (!(this.newton.pcgTolerance >= 0 && this.newton.residualTolerance >= 0 && this.newton.velocityTolerance_m_s >= 0)) throw new Error('Cloth Newton tolerances must be non-negative');
    if (!(this.newton.forcingMax >= 0 && this.newton.forcingMax < 1)) throw new Error(`Cloth Newton forcingMax must be in [0, 1), got ${this.newton.forcingMax}`);
    if (this.newton.trustRegion !== 'off' && this.newton.trustRegion !== 'hybrid') throw new Error(`Unknown cloth Newton trustRegion ${String(this.newton.trustRegion)}`);
    if (this.newton.activeSet !== 'on' && this.newton.activeSet !== 'off') throw new Error(`Unknown cloth Newton activeSet ${String(this.newton.activeSet)}`);
    if (this.newton.contactLinearization !== 'off' && this.newton.contactLinearization !== 'factorized' && this.newton.contactLinearization !== 'blocks')
      throw new Error(`Unknown cloth Newton contactLinearization ${String(this.newton.contactLinearization)}`);
    // Contact enters Newton as global terms: its line-search filter bounds every direction and re-detection runs between iterations.
    if (options.contact && !(options.contact.state && options.hooks?.filter && options.hooks.afterNewtonStep && options.hooks.stepEnd))
      throw new Error('Cloth Newton solver with contact needs the contact state (contact.state) and hooks.filter, hooks.afterNewtonStep and hooks.stepEnd');
    // With contact: the contact records and the contact state (the per-stage limit is 10 storage buffers).
    this.restarts = !!options.contact;
    const need = CLOTH_SOLVER_STORAGE_BUFFERS + (options.contact ? 3 : 0);
    if (device.limits.maxStorageBuffersPerShaderStage < need)
      throw new Error(`Cloth solver needs ${need} storage buffers per stage; the device grants ${device.limits.maxStorageBuffersPerShaderStage}`);
    this.diagnostics = !!options.diagnostics; this.hooks = options.hooks ?? {};
    this.tables = packClothGPUTables(model, options.origin_m, options, options.solverState?.time_s ?? 0, device.limits.maxComputeWorkgroupStorageSize);
    const label = options.label ?? 'Cloth', storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (name: string, bytes: number, usage = storage) => device.createBuffer({ label: `${label} ${name}`, size: Math.max(256, align(bytes, 256)), usage });
    this.positions = make('positions', 16 * this.tables.stride);
    this.state = make('state', 16 * this.tables.stateRecords);
    this.status = make('status', 4 * CLOTH_STATUS_WORDS);
    this.topo = make('topology', 4 * this.tables.topo.length);
    this.rest = make('rest data', 4 * this.tables.rest.length);
    this.params = make('params', 4 * PARAM_WORDS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.newtonWork = make('newton work', 16 * this.tables.newtonRecords);
    {
      const P = new Uint32Array(this.tables.params.buffer, 4 * PARAM_NEWTON, 20), F = new Float32Array(P.buffer, P.byteOffset, 20);
      const g = options.gravity_m_s2, gg = g[0] ** 2 + g[1] ** 2 + g[2] ** 2;
      let m2 = 0; for (let v = 0; v < this.N; v++) if (!(t.groups[v]! > 0)) m2 += t.mass_kg[v]! ** 2;
      F[5] = this.newton.pcgTolerance ** 2; F[6] = this.newton.residualTolerance ** 2 * m2 * gg; F[15] = this.newton.velocityTolerance_m_s;
      F[18] = this.newton.forcingMax ** 2;
    }
    // One dispatch record per PCG dispatch i = 0 .. pcgIterations: (0, N, i, 0).
    const dispatches = this.newton.pcgIterations + 1;
    this.dispatch = make('dispatch records', CLOTH_DISPATCH_STRIDE * dispatches, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    writeBufferChunked(device.queue, this.topo, 0, this.tables.topo);
    writeBufferChunked(device.queue, this.rest, 0, this.tables.rest);
    const records = new ArrayBuffer(CLOTH_DISPATCH_STRIDE * dispatches);
    for (let i = 0; i < dispatches; i++) new Uint32Array(records, CLOTH_DISPATCH_STRIDE * i, 4).set([0, this.N, i, 0]);
    writeBufferChunked(device.queue, this.dispatch, 0, new Uint8Array(records));
    this.uploadState(options.positions_m ?? t.data.restPositions_m, options.velocities_m_s, options.solverState);
  }

  static async create(device: GPUDevice, model: ClothSolverModel, options: ClothGPUSolverOptions): Promise<ClothGPUSolver> {
    const solver = new ClothGPUSolver(device, model, options);
    await solver.compile();
    return solver;
  }

  get strainLimited(): boolean { return this.tables.strainLimited; }
  /** Slices other consumers bind: velocities (C2), normals and triangles (render), canonical edges (C2). */
  get velocities(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.velocity * this.tables.stride, size: 16 * this.tables.stride }; }
  get normals(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.normal * this.tables.stride, size: 16 * this.tables.stride }; }
  get triangles(): GPUBufferBinding { return { buffer: this.topo, offset: 4 * this.tables.trianglesWord, size: 16 * Math.max(this.T, 1) }; }
  get edges(): GPUBufferBinding { return { buffer: this.topo, offset: 4 * this.tables.edgesWord, size: 16 * Math.max(this.E, 1) }; }
  get positionsBinding(): GPUBufferBinding { return { buffer: this.positions, offset: 0, size: 16 * this.tables.stride }; }
  /** Scene time of clock step 0 (the uploaded state's clock). */
  get startTime_s(): number { return this.timeBase; }
  /** x_t of the step in flight (the positions at the start of the step); C2's detector binds it. */
  get stepStart(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.xt * this.tables.stride, size: 16 * this.tables.stride }; }
  /** The step's inertial target y (vec4 per vertex, cloth-local metres; a pinned vertex's is its attachment target), set by the frame pass. */
  get inertialTarget(): GPUBufferBinding { return { buffer: this.state, offset: 16 * CLOTH_STATE.y * this.tables.stride, size: 16 * this.tables.stride }; }
  /** Newton: the direction d of the iteration in flight (vec4 per vertex), which C2's line-search filter reads. */
  get newtonDirection(): GPUBufferBinding {
    return { buffer: this.newtonWork, offset: 16 * CLOTH_NEWTON_SEGMENTS.d * this.tables.stride, size: 16 * this.tables.stride };
  }

  /** Newton: the control records C2's re-detection reads after each step: the first (vec4: done, PCG done, the accepted
   * step alpha, Newton iterations of the step) for its level, and control + 2 (the active set, and in .w the vertices a
   * travel bound holds at the accepted trial: the cap-aware done rule's signal for a new round). */
  get newtonControl(): GPUBufferBinding {
    return { buffer: this.newtonWork, offset: 16 * (this.tables.newtonRecords - CLOTH_SEGMENT_ALIGN), size: 16 * 3 };
  }

  private async compile(): Promise<void> {
    const d = this.device, B = CLOTH_BINDING, C = GPUShaderStage.COMPUTE;
    const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } });
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: B.params, visibility: C, buffer: { type: 'uniform' } },
      { binding: B.dispatch, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 } },
      storage(B.positions), storage(B.state), storage(B.status), storage(B.topo, true), storage(B.rest, true), storage(B.topo4, true)];
    const contact = this.options.contact;
    entries.push(storage(B.newton));
    if (contact) entries.push(storage(B.contact, true), storage(B.contactState, true));
    const layout = d.createBindGroupLayout({ label: 'Cloth solver bindings', entries });
    const buffer = (binding: number, b: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding, resource: { buffer: b, size } });
    this.bindGroup = d.createBindGroup({ label: 'Cloth solver', layout, entries: [
      buffer(B.params, this.params), buffer(B.dispatch, this.dispatch, 32), buffer(B.positions, this.positions), buffer(B.state, this.state),
      buffer(B.status, this.status), buffer(B.topo, this.topo), buffer(B.rest, this.rest), buffer(B.topo4, this.topo),
      buffer(B.newton, this.newtonWork),
      ...(contact ? [buffer(B.contact, contact.buffer), buffer(B.contactState, contact.state)] : [])] });
    // The PCG reductions take two subgroupAdds where the device has subgroups of at least 16 lanes (256 / 16 partials).
    const subgroups = d.features.has('subgroups') && (d.adapterInfo?.subgroupMinSize ?? 0) >= 16;
    const module = d.createShaderModule({ label: 'Cloth solver', code: clothSolverWGSL({ contactBlock: contact?.wgsl, pcg: this.tables.pcg, subgroups, trustRegion: this.newton.trustRegion === 'hybrid', activeSet: this.newton.activeSet === 'on' }) });
    const pipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const kernels: readonly Kernel[] = [...KERNELS, ...(contact ? [...RESTART_KERNELS, ...CONTACT_NEWTON_KERNELS] : []), ...(this.tables.pcg.contactLin ? CONTACT_LIN_KERNELS : [])];
    const built = await Promise.all(kernels.map(entryPoint => d.createComputePipelineAsync({ label: `Cloth ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(kernels.map((n, i) => [n, built[i]!])) as Record<Kernel, GPUComputePipeline>;
  }

  private run(pass: GPUComputePassEncoder, name: Kernel, workgroups: number, record = 0): void {
    if (workgroups <= 0) return;
    pass.setPipeline(this.pipelines[name]);
    pass.setBindGroup(0, this.bindGroup, [CLOTH_DISPATCH_STRIDE * record]);
    // One workgroup per vertex can exceed a dimension's limit; the kernel folds (x, y) back into one index.
    const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
  }
  private threads(n: number): number { return Math.ceil(n / CLOTH_WORKGROUP); }

  /** Whether a mouse grab is held (`setGrab`). */
  get grabbing(): boolean { return this.grabHeld; }

  /**
   * Hold a mouse grab: each listed vertex is pulled by the spring (k/2)|x - (a + D)|^2, where k is its weight times its
   * reference stiffness (m/h^2 + elastic, the scale an AVBD pin starts from), a is where the vertex is at the first
   * step that holds the grab (taken on the GPU, so the grab starts from the live pose) and D is the displacement
   * (`moveGrab`, starting at zero). An implicit energy term like the pins: no substeps, no clamps, stable at any
   * weight. Transient input: never saved or published; `uploadState` and `releaseGrab` drop it. Writes the grab
   * segment and the params uniform (input uploads, no readback); edit time per press, never per step.
   */
  setGrab(vertices: ArrayLike<number>, weights: ArrayLike<number>): void {
    if (vertices.length !== weights.length) throw new Error('Cloth grab needs one weight per vertex');
    if (!vertices.length) throw new Error('Cloth grab needs at least one vertex');
    const segment = new Float32Array(4 * this.tables.stride);
    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i]!, w = weights[i]!;
      if (!(Number.isInteger(v) && v >= 0 && v < this.N)) throw new Error(`Cloth grab vertex ${v} is not a vertex of this cloth (${this.N})`);
      if (!(w > 0 && Number.isFinite(w))) throw new Error(`Cloth grab weight ${w} must be positive and finite`);
      segment[4 * v + 3] = w;
    }
    writeBufferChunked(this.device.queue, this.state, 16 * CLOTH_STATE.grab * this.tables.stride, segment);
    this.grabAnchorsPending = true; this.grabHeld = true;
    this.moveGrab([0, 0, 0]);
  }

  /** The held grab's displacement D from its anchors (metres; world and cloth frames differ by a translation only). */
  moveGrab(displacement_m: ClothVector): void {
    if (!this.grabHeld) return;
    if (!displacement_m.every(Number.isFinite)) throw new Error('Cloth grab displacement must be finite');
    this.device.queue.writeBuffer(this.params, 4 * PARAM_GRAB, new Float32Array([...displacement_m, 1]));
  }

  /** Let go: the cloth carries on with the velocity the grab gave it. */
  releaseGrab(): void {
    if (!this.grabHeld) return;
    this.grabHeld = false; this.grabAnchorsPending = false;
    this.device.queue.writeBuffer(this.params, 4 * PARAM_GRAB, new Float32Array(4));
  }

  /** Encode exactly one implicit step of length dt into `pass`. */
  encodeStep(pass: GPUComputePassEncoder): void {
    const limited = this.tables.strainLimited, o = this.newton, W = this.threads(this.N);
    if (this.grabAnchorsPending) { this.run(pass, 'grabBegin', W); this.grabAnchorsPending = false; }
    if (this.restarts) this.run(pass, 'restartScan', W);
    this.run(pass, 'beginStep', 1);
    if (limited || this.tables.damped) this.run(pass, 'frameElements', this.threads(Math.max(limited ? this.E : 0, this.tables.damped ? this.T : 0)));
    this.run(pass, 'frame', W);
    this.hooks.afterFrame?.(pass);
    this.run(pass, 'nwBeginStep', 1);
    for (let n = 0; n < o.iterations; n++) {
      this.run(pass, 'nwFaces', this.threads(this.T));
      this.run(pass, 'nwGradient', W);
      // The contact linearization at this iterate, compiled for every product of the iteration (pcgStep, nwModel).
      if (this.tables.pcg.contactLin) for (const name of CONTACT_LIN_KERNELS) this.run(pass, name, this.tables.pcg.patches);
      // PCG: dispatch 0 starts the solve, dispatch i >= 1 takes iteration i (one dispatch each, Chronopoulos-Gear).
      // One patch (every vertex a row of one workgroup): the whole solve in one dispatch, its stop a loop break.
      if (this.tables.pcg.patches === 1) this.run(pass, 'pcgSolve', 1, o.pcgIterations);
      else for (let i = 0; i <= o.pcgIterations; i++) this.run(pass, 'pcgStep', this.tables.pcg.patches, i);
      this.hooks.filter?.(pass);
      if (this.options.contact) this.run(pass, 'nwProject', W);
      this.run(pass, 'nwTrial', W);
      // Trust-region iterations (newton.trustRegion 'hybrid'): s.H s of the projected step for the model's decrease.
      if (o.trustRegion === 'hybrid') this.run(pass, 'nwModel', this.tables.pcg.patches);
      this.run(pass, 'nwChoose', 1);
      this.run(pass, 'nwApply', W);
      this.hooks.afterNewtonStep?.(pass);
      if (limited) this.run(pass, 'nwEdgeDuals', this.threads(this.E));
    }
    this.hooks.beforeFinish?.(pass);
    if (this.diagnostics) {
      this.run(pass, 'diagnose', W);
      this.run(pass, 'reduceDiagnostics', 1);
    }
    this.hooks.stepEnd?.(pass);
    if (this.restarts) this.run(pass, 'onsetScan', W);
    this.run(pass, 'finishStep', W);
  }

  /** Newton-PCG control records (harness reads; never a step path): done flag, PCG done, last step, Newton iterations
   * of the last step; then |g|^2 over free DOFs at its first and latest Newton iterate, |m g|^2 for scale (overwritten
   * here), and the step's PCG iterations; then with contact the last Newton iteration's active set (vertices with an
   * active normal, normals kept, normals released) and the vertices a travel bound held at its accepted trial; then the
   * trust region (control + 3: trust-region mode, radius), the step's globalisation counts (control + 4: consecutive
   * full line-search steps, trust-region iterations, of them rejected, the last ratio rho), the last direction (control
   * + 5: its squared preconditioner norm, whether its solve ended on the boundary) and the last trust-region iteration's
   * model (control + 6: g.s, s.H s, contact's share of s.H s, the energy change at s). */
  async readNewtonControl(): Promise<Float32Array> {
    const L = this.tables.newtonRecords - CLOTH_SEGMENT_ALIGN;
    const out = new Float32Array(await this.readRange(this.newtonWork, 16 * L, 112));
    out[6] = new Float32Array(this.tables.params.buffer, 4 * PARAM_NEWTON, 8)[6]! / this.newton.residualTolerance ** 2;
    return out;
  }

  /** The compiled contact linearization as the last Newton iteration that ran left it (harness reads; never a step
   * path): the pool and what the rows asked of it, the items built (entries, or partner terms for 'blocks') and the
   * rows left on the matrix-free product because their records did not fit. Undefined with the switch off. */
  async readContactLinearization(): Promise<{ mode: 'factorized' | 'blocks'; poolRecords: number; demandRecords: number; usedRecords: number; items: number; entries: number; rows: number; fallbackRows: number; poolBytes: number; usedBytes: number } | undefined> {
    const L = this.tables.pcg.contactLin;
    if (!L) return undefined;
    const per = CLOTH_CONTACT_LIN_RECORDS[L.mode];
    const [head, patch] = await Promise.all([this.readRange(this.newtonWork, 16 * L.head, 16 * this.N), this.readRange(this.newtonWork, 16 * L.patch, 16 * this.tables.pcg.patches)]);
    const h = new Uint32Array(head), t = new Uint32Array(patch);
    let demandRecords = 0, items = 0, entries = 0, rows = 0, fallbackRows = 0;
    for (let p = 0; p < this.tables.pcg.patches; p++) demandRecords += t[4 * p]!;
    for (let v = 0; v < this.N; v++) {
      if (h[4 * v + 2] === 0) continue;
      rows++;
      // A row left on the matrix-free product keeps its entry count in y; a compiled one holds its items there.
      if (h[4 * v] === 0x7f000000) fallbackRows++; else items += h[4 * v + 1]!;
      entries += L.mode === 'factorized' || h[4 * v] === 0x7f000000 ? h[4 * v + 1]! : 0;
    }
    return { mode: L.mode, poolRecords: L.records, demandRecords, usedRecords: per * items, items, entries, rows, fallbackRows, poolBytes: 16 * L.records, usedBytes: 16 * per * items };
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
    // Render normals of the uploaded pose; every step rewrites them at its end.
    const tri = this.model.topology.data.triangles, nrm = at(CLOTH_STATE.normal, 0);
    for (let f = 0; f < tri.length; f += 3) {
      const [a, b, c] = [tri[f]!, tri[f + 1]!, tri[f + 2]!], p = (v: number, k: number) => x[4 * v + k]!;
      const u = [0, 1, 2].map(k => p(b, k) - p(a, k)), w = [0, 1, 2].map(k => p(c, k) - p(a, k));
      const n = [u[1]! * w[2]! - u[2]! * w[1]!, u[2]! * w[0]! - u[0]! * w[2]!, u[0]! * w[1]! - u[1]! * w[0]!];
      for (const v of [a, b, c]) for (let k = 0; k < 3; k++) state[nrm + 4 * v + k]! += n[k]!;
    }
    for (let i = 0; i < N; i++) {
      const o4 = nrm + 4 * i, l = Math.hypot(state[o4]!, state[o4 + 1]!, state[o4 + 2]!);
      if (l > 0) for (let k = 0; k < 3; k++) state[o4 + k]! /= l; else state[o4 + 1] = 1;
    }
    state.set([0, solverState?.previousStep_s ?? 0, this.options.dt_s, 0], 4 * L.clockRecord);
    writeBufferChunked(q, this.positions, 0, x); writeBufferChunked(q, this.state, 0, state);
    this.timeBase = solverState?.time_s ?? 0;
    // A new state drops any grab (the params below hold none, the grab segment above is zero).
    this.grabHeld = false; this.grabAnchorsPending = false;
    const params = L.params.slice();
    new Float32Array(params.buffer)[PARAM_TIME] = this.timeBase;
    q.writeBuffer(this.params, 0, params);
    q.writeBuffer(this.status, 0, new Uint32Array(CLOTH_STATUS_WORDS));
  }

  private read(buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> { return this.readRange(buffer, 0, bytes); }
  private async readRange(buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
    const size = align(Math.max(bytes, 4), 4);
    const staging = this.device.createBuffer({ label: 'Cloth readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
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

  /**
   * The buffer ranges that hold the full dynamic state, in `decodeState`'s order: positions, then the velocity, v_{n-1},
   * dx_{n-1} and pin-dual segments, the edge duals and the clock record. A pose mirror copies these to read the state
   * without a step-path readback.
   */
  stateRanges(): readonly { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number }[] {
    const L = this.tables, seg = (k: number) => ({ buffer: this.state, offset: 16 * k * L.stride, size: 16 * this.N });
    return [{ buffer: this.positions, offset: 0, size: 16 * this.N }, seg(CLOTH_STATE.velocity), seg(CLOTH_STATE.vPrev), seg(CLOTH_STATE.dxPrev),
      seg(CLOTH_STATE.pinDual), { buffer: this.state, offset: 16 * L.edgeDualRecord, size: 16 * this.E }, { buffer: this.state, offset: 16 * L.clockRecord, size: 16 }];
  }

  /** The state from the bytes of `stateRanges`, one buffer per range (save, load, undo and the edit path's pose). */
  decodeState(parts: readonly ArrayBuffer[]): ClothGPUState {
    const N = this.N, o = this.options.origin_m;
    if (parts.length !== 7) throw new Error('Cloth state decode needs the 7 stateRanges');
    const [x, v, vp, dx, pin, edge, clock] = parts.map(b => new Float32Array(b)) as [Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array];
    const positions = new Float64Array(3 * N), velocities = new Float64Array(3 * N), vPrev: number[] = [], dxPrev: number[] = [], pinDuals: number[] = [];
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        positions[3 * i + c] = x[4 * i + c]! + o[c]!; velocities[3 * i + c] = v[4 * i + c]!;
        vPrev.push(vp[4 * i + c]!); dxPrev.push(dx[4 * i + c]!);
      }
      if (this.model.topology.groups[i]! > 0) pinDuals.push(...pin.subarray(4 * i, 4 * i + 4));
    }
    const edgeDuals: number[] = [];
    for (let k = 0; k < this.E; k++) edgeDuals.push(edge[4 * k]!, edge[4 * k + 1]!);
    return { positions_m: positions, velocities_m_s: velocities, solverState: { previousVelocities_m_s: vPrev, previousDisplacements_m: dxPrev,
      previousStep_s: clock[1]!, time_s: Math.fround(this.timeBase) + clock[0]! * this.options.dt_s, pinDuals, edgeDuals } };
  }

  /** The full dynamic state for save, load and undo. */
  async readState(): Promise<ClothGPUState> {
    return this.decodeState(await Promise.all(this.stateRanges().map(r => this.readRange(r.buffer, r.offset, r.size))));
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
    return { status: words[0]!, steps: words[1]!, restarts: words[CLOTH_STATUS_WORD.restarts]!, slots };
  }

  static describeStatus(status: number): string[] {
    return status & CLOTH_STATUS.nonFinite ? ['A cloth position or velocity became non-finite'] : [];
  }

  destroy(): void { for (const b of [this.positions, this.state, this.status, this.topo, this.rest, this.params, this.dispatch, this.newtonWork]) b.destroy(); }
}
