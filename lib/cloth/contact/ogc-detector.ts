/**
 * GPU OGC self-contact detector for triangle cloth (Chen et al. 2025).
 *
 * Owns the detection positions X = x_det, the per-vertex contact lists, the
 * DAT division pairs, the broad-phase hash grid, the truncation and the step's
 * re-detection decision. Per step, all on the GPU and with no readback:
 *
 *   encodeStepBegin + encodeDetection        mandatory round at x = x_t
 *   (solver initial guess) encodeTruncate
 *   per colour sweep: solver (ogcAccumulateVertexContact), encodeTruncate
 *   per iteration:    encodeDecide + encodeDetection   indirect: zero workgroups
 *                                                      unless gamma_e N truncated
 *
 * The solver's positions are X + dX: dX accumulates across iterations, every
 * truncation cuts the accumulated dX, and a detection round commits it (DAT
 * Alg. 3). Truncation is Planar-DAT by default, or OGC's isotropic bound.
 *
 * Each vertex's query radius is its velocity envelope
 *   r_q,v = clamp(max(rqBase, r + s |v dt + g dt^2|), rqMax).
 * A division pair is recorded whenever the two primitives are within the larger
 * of their envelopes (a face's or edge's envelope is its vertices' widest), and
 * every vertex's displacement is capped at 0.5 gamma_r r_q,v, so two unpaired
 * primitives close by at most gamma_r max(r_q,a, r_q,b), less than their distance.
 * Capacities are authored per scene; overflow, a round cap or an envelope cap
 * set the status word (OGC_STATUS), which the frame's owner reads off the step path.
 */
import { buildOgcTopology, type OgcMeshTopology, type OgcTopology } from "./ogc-topology";
import {
  OGC_APPLY_WGSL, OGC_ARGS_WORDS, OGC_BOUND_WGSL, OGC_CELL_WGSL, OGC_DECIDE_WGSL, OGC_INTERSECT_WGSL, OGC_PARAM,
  OGC_PARAM_WORDS, OGC_PLANES_WGSL, OGC_PREP_WGSL, OGC_QUERY_WGSL, OGC_STEP_BEGIN_WGSL,
  ogcFrameBeginWGSL, ogcGridCountWGSL, ogcGridInsertWGSL, ogcScanWGSL,
} from "./ogc-detector-wgsl";
import {
  OGC_COUNTER, OGC_COUNTER_WORDS, OGC_FRAME_COUNTERS, OGC_STATUS, OGC_TRUNCATION, ogcVertexBlockWGSL,
  type OgcTruncationMode,
} from "./ogc-wgsl";
import { writeBufferChunked } from "../../methods/uniform/uniform-chunked-upload";
import { SOLID_HEADER_WORDS, type ClothSolidField } from "./solid-field";
import { OGC_NONE } from "./ogc-topology";

export interface OgcContactConfig {
  /** Offset radius: the physical cloth thickness (m). */
  r: number;
  /** Contact stiffness k_c of the activation (N/m). */
  kc: number;
  /** Friction coefficient and the static-dynamic transition velocity eps_v (m/s). */
  mu: number;
  epsV: number;
  /** The step (s). */
  dt: number;
  gravity: readonly [number, number, number];
  /** Query radius floor; with envelopeScale 0 it is the fixed r_q. */
  rqBase: number;
  /** Authored envelope cap; reaching it raises OGC_STATUS.envelopeCap. */
  rqMax: number;
  /** Safety factor s on the inertial displacement; 0 gives the fixed envelope. */
  envelopeScale: number;
  /** OGC's gamma_p (0 < gamma_p < 0.5) and gamma_e. */
  gammaP?: number;
  gammaE?: number;
  /** Truncation scheme (default planar) and DAT's relaxation gamma_r in (0, 1) (default 0.9). */
  truncation?: OgcTruncationMode;
  gammaR?: number;
  /** Half-width of the empty band DAT keeps on each side of a plane (m); default from the coordinate scale. */
  separationEps?: number;
  /** Authored capacities: entries per vertex list, DAT pair slots, grid entries, hash buckets (power of two). */
  contactCapacity: number;
  pairCapacity: number;
  gridCapacity: number;
  bucketCount: number;
  /** Re-detection rounds allowed per step after the mandatory one. */
  roundCap: number;
  /** Words reserved for the static solid field image (solid-field.ts); 0 or absent: the cloth sees no solid. */
  solidCapacityWords?: number;
}

export interface OgcSolverInputs {
  /** vec4<f32> per vertex, cloth-local metres (w is the solver's). */
  positions: GPUBuffer | GPUBufferBinding;
  /** vec4<f32> per vertex, m/s: the step's start velocity, for the envelope. */
  velocities: GPUBuffer | GPUBufferBinding;
}

export interface OgcFrameCounters {
  status: number;
  steps: number;
  detections: number;
  redetections: number;
  decides: number;
  truncated: number;
  truncatedMax: number;
  pairEntries: number;
  pairMax: number;
  candidates: number;
  vfPairs: number;
  eePairs: number;
  envelopeCapHits: number;
  peakEnvelope: number;
  /** DAT division pairs recorded over the frame's rounds, the widest round's slots, and the persistent peak. */
  datPairs: number;
  framePeakPairSlots: number;
  peakPairSlots: number;
  /** Vertices cut by the 0.5 gamma_r r_q,v cap (or b_v), and pairs held for want of a separating plane. */
  capped: number;
  uncertified: number;
  /** Vertices with a static-solid entry over the frame's rounds, and truncations the solid bound made. */
  solidContacts: number;
  solidCut: number;
  roundCapHits: number;
  peakContacts: number;
  peakGrid: number;
  intersections: number;
  cell: number;
}

interface Kernel { pipeline: GPUComputePipeline; bindGroup?: GPUBindGroup; bindings: number[]; override?: Record<number, "positions"> }

const WG = 64;
const DISPATCH = { prep: 0, cell: 1, count: 2, scan1: 3, scan2: 4, scan3: 5, insert: 6, query: 7, bound: 8 } as const;
const FLOAT32_EPS = 2 ** -23;

export class OgcContactDetector {
  readonly N: number; readonly T: number; readonly E: number;
  readonly topology: OgcTopology;
  readonly params: GPUBuffer; readonly topo: GPUBuffer;
  /** x_det (xyz) and b_v (w), vec4<f32> per vertex. */
  readonly det: GPUBuffer;
  readonly work: GPUBuffer; readonly counts: GPUBuffer; readonly grid: GPUBuffer;
  /** Header, per-vertex counts and contact entries (see ogc-wgsl.ts). */
  readonly contacts: GPUBuffer;
  /** DAT division pairs, two vec4<u32> each (see ogc-detector-wgsl.ts). */
  readonly pairs: GPUBuffer;
  readonly mode: OgcTruncationMode;
  readonly counters: GPUBuffer; readonly args: GPUBuffer;
  /** The planes dispatch's indirect args, written by the round's bound pass. */
  readonly planesArgs: GPUBuffer;
  private kernels = new Map<string, Kernel>();
  private readonly paramWords = new Uint32Array(OGC_PARAM_WORDS);

  private inputs?: OgcSolverInputs;

  private constructor(private readonly device: GPUDevice, topology: OgcTopology, readonly config: OgcContactConfig, coordinateScale: number) {
    const L = topology.layout;
    this.topology = topology;
    this.N = L.vertexCount; this.T = L.triangleCount; this.E = L.edgeCount;
    const { N, T, E } = this;
    const H = config.bucketCount;
    if (!(H > 0 && (H & (H - 1)) === 0)) throw new Error("OGC: bucketCount must be a power of two");
    const B1 = Math.ceil(H / 1024), B2 = Math.ceil(B1 / 1024);
    if (B2 > 1024) throw new Error("OGC: bucketCount above 2^30 needs a fourth scan level");
    for (const [name, wg] of [["vertices", Math.ceil(N / WG)], ["vertices+edges", Math.ceil((N + E) / WG)], ["faces", Math.ceil(T / WG)], ["buckets", B1]] as const)
      if (wg > 65535) throw new Error(`OGC: ${name} exceed one dispatch dimension`);
    const gammaP = config.gammaP ?? 0.45, gammaE = config.gammaE ?? 0.01;
    if (!(gammaP > 0 && gammaP < 0.5)) throw new Error("OGC: gamma_p must lie in (0, 0.5)");
    if (!(config.r > 0 && config.rqBase >= config.r && config.rqMax >= config.rqBase)) throw new Error("OGC: need 0 < r <= rqBase <= rqMax");
    if (!(config.contactCapacity > 0 && config.pairCapacity > 0 && config.gridCapacity > 0 && config.roundCap >= 0)) throw new Error("OGC: capacities must be authored");
    const mode = config.truncation ?? "planar";
    if (!(mode in OGC_TRUNCATION)) throw new Error(`OGC: unknown truncation "${String(mode)}"`);
    this.mode = mode;
    const gammaR = config.gammaR ?? 0.9;
    if (!(gammaR > 0 && gammaR < 1)) throw new Error("OGC: gamma_r must lie in (0, 1)");
    const sepEps = config.separationEps ?? Math.max(1e-6, 4 * FLOAT32_EPS * coordinateScale);
    if (!(sepEps > 0 && sepEps < 0.5 * config.r)) throw new Error("OGC: separationEps must lie in (0, r/2)");
    if (2 * config.pairCapacity * 16 > device.limits.maxStorageBufferBindingSize) throw new Error("OGC: pairCapacity exceeds one storage binding");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (label: string, bytes: number, u = usage) => device.createBuffer({ label: `ogc ${label}`, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: u });
    this.params = mk("params", 4 * OGC_PARAM_WORDS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.topo = mk("topology", 4 * topology.words.length);
    this.det = mk("x_det", 16 * N);
    this.work = mk("work", 4 * (7 * N + T + E));
    this.counts = mk("grid counts", 4 * H);
    const local1Off = 0, sums1Off = H, local2Off = H + B1, sums2Off = H + 2 * B1, local3Off = H + 2 * B1 + B2, sums3Off = H + 2 * B1 + 2 * B2;
    const entriesOff = Math.ceil((sums3Off + 1) / 2) * 2;
    this.grid = mk("grid", 4 * (entriesOff + 2 * config.gridCapacity));
    const countsBase = 3, entriesBase = 3 + Math.ceil(N / 4);
    const solidWords = config.solidCapacityWords ?? 0;
    if (!(Number.isSafeInteger(solidWords) && solidWords >= 0)) throw new Error("OGC: solidCapacityWords must be a whole number");
    if (solidWords > 0 && solidWords < SOLID_HEADER_WORDS + 4) throw new Error("OGC: solidCapacityWords cannot hold an empty solid image");
    this.solidBase = solidWords > 0 ? 4 * (entriesBase + N * config.contactCapacity) : 0;
    this.solidCapacityWords = solidWords;
    this.contacts = mk("contacts", 16 * (entriesBase + N * config.contactCapacity) + 4 * solidWords);
    if (this.contacts.size > device.limits.maxStorageBufferBindingSize) throw new Error("OGC: the contact buffer exceeds one storage binding");
    this.pairs = mk("dat pairs", 32 * config.pairCapacity);
    this.counters = mk("counters", 4 * OGC_COUNTER_WORDS);
    this.args = mk("args", 4 * OGC_ARGS_WORDS, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.planesArgs = mk("planes args", 12, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const w = this.paramWords, f = new Float32Array(w.buffer), P = OGC_PARAM;
    w[P.N] = N; w[P.T] = T; w[P.E] = E; w[P.H] = H;
    w[P.gridCap] = config.gridCapacity; w[P.contactCap] = config.contactCapacity; w[P.roundCap] = config.roundCap;
    w[P.exceedThreshold] = Math.max(1, Math.ceil(gammaE * N));
    f[P.r] = config.r; f[P.rqBase] = config.rqBase; f[P.rqMax] = config.rqMax; f[P.envScale] = config.envelopeScale;
    f[P.gammaP] = gammaP; f[P.cellFloor] = Math.max(L.meanEdgeLength, config.rqBase); f[P.dt] = config.dt;
    f[P.intersectCell] = Math.max(L.meanEdgeLength, 1e-6);
    f.set(config.gravity, P.gravity);
    w[P.trisOff] = L.trisOffset; w[P.edgesOff] = L.edgesOffset; w[P.triEdgesOff] = L.triEdgesOffset;
    w[P.vtOffOff] = L.vtOffOffset; w[P.vtListOff] = L.vtListOffset; w[P.veOffOff] = L.veOffOffset; w[P.veListOff] = L.veListOffset;
    w[P.B1] = B1; w[P.B2] = B2;
    w[P.local1Off] = local1Off; w[P.sums1Off] = sums1Off; w[P.local2Off] = local2Off; w[P.sums2Off] = sums2Off;
    w[P.local3Off] = local3Off; w[P.sums3Off] = sums3Off; w[P.entriesOff] = entriesOff;
    w[P.countsOff] = 4 * countsBase; w[P.entriesBase] = 4 * entriesBase;
    w[P.dminTOff] = N; w[P.dminEOff] = N + T; w[P.envOff] = N + T + E;
    w[P.wgN] = Math.ceil(N / WG); w[P.wgT] = Math.ceil(T / WG); w[P.wgNE] = Math.ceil((N + E) / WG);
    w[P.wgS1] = B1; w[P.wgS2] = B2; f[P.kc] = config.kc;
    w[P.mode] = OGC_TRUNCATION[mode]; f[P.gammaR] = gammaR; w[P.pairCap] = config.pairCapacity;
    w[P.tvOff] = 2 * N + T + E; f[P.sepEps] = sepEps;
    w[P.solidBase] = this.solidBase; w[P.solidOff] = 3 * N + T + E;
    device.queue.writeBuffer(this.params, 0, w);
    writeBufferChunked(device.queue, this.topo, 0, topology.words);
    const hdr = new Uint32Array(12), hf = new Float32Array(hdr.buffer);
    hdr[0] = N; hdr[1] = config.contactCapacity; hdr[2] = countsBase; hdr[3] = entriesBase;
    hf[4] = config.r; hf[5] = config.kc; hf[6] = config.mu; hf[7] = config.epsV * config.dt;
    hdr[8] = this.solidBase;
    device.queue.writeBuffer(this.contacts, 0, hdr);
    if (this.solidBase) {
      // No solid until one is set: one empty slot, no bricks, an unbounded band.
      const empty = new Uint32Array(SOLID_HEADER_WORDS + 4), ef = new Float32Array(empty.buffer);
      ef[2] = 1; ef[3] = 3.0e38; empty[SOLID_HEADER_WORDS + 3] = OGC_NONE;
      device.queue.writeBuffer(this.contacts, 4 * this.solidBase, empty);
    }
  }

  /** Word offset of the solid image in the contact buffer (0: no solid) and its authored capacity. */
  readonly solidBase: number;
  readonly solidCapacityWords: number;

  /**
   * Upload the static solid field (setup and solid edits only; never in a step
   * path). Refuses an image beyond the authored capacity, and a band too thin
   * for the envelope: every missing brick must lie further than
   * r + 1.5 L_max rqMax from where a vertex's round can reach.
   */
  setSolidField(field: ClothSolidField) {
    if (!this.solidBase) throw new Error("OGC: the detector was built without solidCapacityWords");
    if (field.words.length > this.solidCapacityWords)
      throw new Error(`OGC: the solid field needs ${field.words.length} words, the scene authored ${this.solidCapacityWords}`);
    const need = this.config.r + 1.5 * field.maxLipschitz * this.config.rqMax;
    if (!(field.band >= need)) throw new Error(`OGC: the solid field's band ${field.band} m is below r + 1.5 L_max rqMax = ${need} m`);
    writeBufferChunked(this.device.queue, this.contacts, 4 * this.solidBase, field.words);
  }

  /**
   * Build the detector and compile its kernels (asynchronously, outside any
   * constructor). Without inputs, `bind` attaches the solver's buffers later:
   * the solver itself is built around `solverBuffer`.
   */
  static async create(device: GPUDevice, mesh: OgcMeshTopology, restPositions: Float32Array, config: OgcContactConfig, inputs?: OgcSolverInputs): Promise<OgcContactDetector> {
    let scale = 1;
    for (const c of restPositions) scale = Math.max(scale, Math.abs(c));
    const d = new OgcContactDetector(device, buildOgcTopology(mesh, restPositions), config, scale);
    await d.compile();
    if (inputs) d.bind(inputs);
    return d;
  }

  /** The buffer the solver kernel binds for ogcVertexBlockWGSL. */
  get solverBuffer(): GPUBuffer { return this.contacts; }

  static vertexBlockWGSL(group: number, binding: number): string { return ogcVertexBlockWGSL({ group, binding }); }

  /** What the cloth solver's vertex kernel takes: the block at (group 0, binding) and the buffer it reads. */
  solverContact(binding: number): { wgsl: string; buffer: GPUBuffer } {
    return { wgsl: ogcVertexBlockWGSL({ group: 0, binding }), buffer: this.contacts };
  }

  /**
   * The step hooks for one implicit step of the cloth solver (its ClothStepHooks):
   * the mandatory round and a truncation of the initial guess after the step
   * frame, a truncation after every colour sweep, and after every iteration a
   * truncation of the iteration's last moves, the decision and the (indirect,
   * usually empty) re-detection round.
   */
  get stepHooks() {
    return {
      afterFrame: (pass: GPUComputePassEncoder) => { this.encodeStepBegin(pass); this.encodeDetection(pass); this.encodeTruncate(pass); },
      afterColour: (pass: GPUComputePassEncoder) => { this.encodeTruncate(pass); },
      afterIteration: (pass: GPUComputePassEncoder) => { this.encodeTruncate(pass); this.encodeDecide(pass); this.encodeDetection(pass); },
    };
  }

  private async compile() {
    const specs: [string, string, number[], Record<number, "positions">?][] = [
      ["prep", OGC_PREP_WGSL, [0, 2, 3, 4, 5, 8, 9]],
      ["cell", OGC_CELL_WGSL, [0, 9]],
      ["count", ogcGridCountWGSL(false), [0, 1, 4, 5, 6, 9]],
      ["scan1", ogcScanWGSL(1), [0, 6, 7]],
      ["scan2", ogcScanWGSL(2), [0, 7]],
      ["scan3", ogcScanWGSL(3), [0, 7]],
      ["insert", ogcGridInsertWGSL(false), [0, 1, 4, 5, 6, 7, 9]],
      ["query", OGC_QUERY_WGSL, [0, 1, 4, 5, 7, 8, 9, 11]],
      ["bound", OGC_BOUND_WGSL, [0, 1, 4, 5, 8, 9, 12]],
      ["stepBegin", OGC_STEP_BEGIN_WGSL, [0, 9, 10]],
      ["decide", OGC_DECIDE_WGSL, [0, 9, 10]],
      ["frameBegin", ogcFrameBeginWGSL(OGC_FRAME_COUNTERS), [9]],
      ["planes", OGC_PLANES_WGSL, [0, 2, 4, 5, 9, 11]],
      ["apply", OGC_APPLY_WGSL, [0, 2, 4, 5, 9]],
      // Intersection counting reads the solver's current positions where detection reads x_det.
      ["xCount", ogcGridCountWGSL(true), [0, 1, 4, 6, 9], { 4: "positions" }],
      ["xInsert", ogcGridInsertWGSL(true), [0, 1, 4, 6, 7, 9], { 4: "positions" }],
      ["xTest", OGC_INTERSECT_WGSL, [0, 1, 4, 7, 9], { 4: "positions" }],
    ];
    await Promise.all(specs.map(async ([name, code, bindings, override]) => {
      const module = this.device.createShaderModule({ label: `ogc ${name}`, code });
      const pipeline = await this.device.createComputePipelineAsync({ label: `ogc ${name}`, layout: "auto", compute: { module, entryPoint: "main" } });
      this.kernels.set(name, { pipeline, bindings, override });
    }));
  }

  /** Attach the solver's positions and step-start velocities (once, at setup). */
  bind(inputs: OgcSolverInputs) {
    if (this.inputs) throw new Error("OGC: the detector is already bound to a solver");
    this.inputs = inputs;
    const res = (b: GPUBuffer | GPUBufferBinding): GPUBufferBinding => ("buffer" in b ? b : { buffer: b });
    const all: Record<number, GPUBufferBinding> = {
      0: res(this.params), 1: res(this.topo), 2: res(inputs.positions), 3: res(inputs.velocities), 4: res(this.det), 5: res(this.work),
      6: res(this.counts), 7: res(this.grid), 8: res(this.contacts), 9: res(this.counters), 10: res(this.args), 11: res(this.pairs), 12: res(this.planesArgs),
    };
    for (const [name, k] of this.kernels) {
      k.bindGroup = this.device.createBindGroup({
        label: `ogc ${name}`, layout: k.pipeline.getBindGroupLayout(0),
        entries: k.bindings.map(i => ({ binding: i, resource: k.override?.[i] === "positions" ? all[2]! : all[i]! })),
      });
    }
  }

  private kernel(name: string): Kernel & { bindGroup: GPUBindGroup } {
    const k = this.kernels.get(name)!;
    if (!k.bindGroup) throw new Error("OGC: bind the solver's buffers before encoding");
    return k as Kernel & { bindGroup: GPUBindGroup };
  }

  private run(pass: GPUComputePassEncoder, name: string, workgroups: number) {
    const k = this.kernel(name);
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroups(workgroups);
  }
  private runIndirect(pass: GPUComputePassEncoder, name: keyof typeof DISPATCH) {
    const k = this.kernel(name);
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.args, 12 * DISPATCH[name]);
  }

  /** Reset the per-frame statistics (status and peaks persist). */
  encodeFrameBegin(pass: GPUComputePassEncoder) { this.run(pass, "frameBegin", 1); }
  /** Start of a step: arm the mandatory detection round. */
  encodeStepBegin(pass: GPUComputePassEncoder) { this.run(pass, "stepBegin", 1); }
  /** One detection round, dispatched indirectly: zero workgroups unless armed. */
  encodeDetection(pass: GPUComputePassEncoder) {
    for (const name of Object.keys(DISPATCH) as (keyof typeof DISPATCH)[]) this.runIndirect(pass, name);
  }
  /** Between iterations: count the iteration's truncations and arm a round when gamma_e N were truncated. */
  encodeDecide(pass: GPUComputePassEncoder) { this.run(pass, "decide", 1); }
  /** Truncate the accumulated displacement x - x_det of every vertex: division planes (indirect), then apply. */
  encodeTruncate(pass: GPUComputePassEncoder) {
    const k = this.kernel("planes");
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.planesArgs, 0);
    this.run(pass, "apply", Math.ceil(this.N / WG));
  }
  /** Count intersecting non-adjacent face pairs of the current positions (counter `intersections`). */
  encodeIntersectionCount(pass: GPUComputePassEncoder) {
    const H = this.config.bucketCount, B1 = Math.ceil(H / 1024), B2 = Math.ceil(B1 / 1024), wgT = Math.ceil(this.T / WG);
    this.run(pass, "xCount", Math.max(1, wgT));
    this.run(pass, "scan1", B1); this.run(pass, "scan2", B2); this.run(pass, "scan3", 1);
    this.run(pass, "xInsert", Math.max(1, wgT)); this.run(pass, "xTest", Math.max(1, wgT));
  }

  /** Read the counters. Harness and gate use only: never in a step path. */
  async readCounters(): Promise<OgcFrameCounters> {
    const read = this.device.createBuffer({ size: 4 * OGC_COUNTER_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.counters, 0, read, 0, 4 * OGC_COUNTER_WORDS);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(read.getMappedRange().slice(0)), fl = new Float32Array(u.buffer);
    read.destroy();
    const C = OGC_COUNTER;
    return {
      status: u[C.status], steps: u[C.frameSteps], detections: u[C.frameDetections], redetections: u[C.frameRedetections],
      decides: u[C.frameDecides], truncated: u[C.frameTruncated], truncatedMax: u[C.frameTruncatedMax],
      pairEntries: u[C.framePairEntries], pairMax: u[C.framePairMax], candidates: u[C.frameCandidates],
      vfPairs: u[C.frameVfPairs], eePairs: u[C.frameEePairs], envelopeCapHits: u[C.frameEnvelopeCapHits],
      peakEnvelope: fl[C.framePeakEnvelope], roundCapHits: u[C.frameRoundCapHits], peakContacts: u[C.peakContacts],
      datPairs: u[C.frameDatPairs], framePeakPairSlots: u[C.framePeakPairSlots], peakPairSlots: u[C.peakPairSlots],
      capped: u[C.frameCapped], uncertified: u[C.frameUncertified],
      solidContacts: u[C.frameSolidContacts], solidCut: u[C.frameSolidCut],
      peakGrid: u[C.peakGrid], intersections: u[C.intersections], cell: fl[C.cell],
    };
  }

  /** Explain a status word. */
  static describeStatus(status: number): string[] {
    return (Object.entries(OGC_STATUS) as [string, number][]).filter(([, bit]) => status & bit).map(([name]) => name);
  }

  destroy() {
    for (const buffer of [this.params, this.topo, this.det, this.work, this.counts, this.grid, this.contacts, this.pairs, this.counters, this.args, this.planesArgs]) buffer.destroy();
  }
}
