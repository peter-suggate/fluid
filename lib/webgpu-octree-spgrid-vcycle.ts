import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-mgpcg";

/** Native sparse-level cell roles from Setaluri et al., section 5. */
export const SPGRID_CELL_FLAG = Object.freeze({
  active: 1 << 0,
  ghost: 1 << 1,
  multigridOnly: 1 << 2,
} as const);

export interface SPGridLeafOracle {
  readonly origin: readonly [number, number, number];
  /** Power-of-two width in finest cells. */
  readonly size: number;
}

export interface SPGridTransferRecord {
  readonly fine: number;
  readonly coarse: number;
  readonly weight: number;
}

export interface SPGridOracleLevel {
  readonly scale: number;
  readonly coordinates: readonly (readonly [number, number, number])[];
  readonly flags: readonly number[];
}

export interface SPGridPyramidOracle {
  readonly levels: readonly SPGridOracleLevel[];
  /** P(fine,coarse), stored once and used as P and P^T. */
  readonly transfers: readonly (readonly SPGridTransferRecord[])[];
}

const coordinateKey = (value: readonly [number, number, number]) => `${value[0]},${value[1]},${value[2]}`;

function assertPowerOfTwo(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive power of two`);
  }
}

/**
 * Small CPU oracle for topology/transfer tests. Active octree cells live at
 * their native level; coarser leaves appear as ghosts on finer MG levels and
 * generated interpolation targets are explicitly marked multigrid-only.
 */
export function buildSPGridPyramidOracle(leaves: readonly SPGridLeafOracle[], levelCount: number): SPGridPyramidOracle {
  if (!Number.isSafeInteger(levelCount) || levelCount < 2) throw new RangeError("SPGrid level count must be at least two");
  const maps = Array.from({ length: levelCount }, () => new Map<string, { coordinate: [number, number, number]; flags: number }>());
  const add = (level: number, coordinate: [number, number, number], flags: number) => {
    const key = coordinateKey(coordinate), old = maps[level].get(key);
    if (old) old.flags = (old.flags & SPGRID_CELL_FLAG.active) !== 0 || (flags & SPGRID_CELL_FLAG.active) !== 0
      ? SPGRID_CELL_FLAG.active
      : (old.flags & SPGRID_CELL_FLAG.ghost) !== 0 || (flags & SPGRID_CELL_FLAG.ghost) !== 0
        ? SPGRID_CELL_FLAG.ghost : SPGRID_CELL_FLAG.multigridOnly;
    else maps[level].set(key, { coordinate, flags });
  };
  for (const leaf of leaves) {
    assertPowerOfTwo(leaf.size, "SPGrid leaf size");
    if (leaf.origin.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      throw new RangeError("SPGrid leaf origins must be non-negative integers");
    }
    const nativeLevel = Math.min(levelCount - 1, Math.round(Math.log2(leaf.size)));
    for (let level = 0; level < levelCount; level += 1) {
      const scale = 2 ** level;
      const coordinate: [number, number, number] = [
        Math.floor(leaf.origin[0] / scale), Math.floor(leaf.origin[1] / scale), Math.floor(leaf.origin[2] / scale),
      ];
      add(level, coordinate, level === nativeLevel ? SPGRID_CELL_FLAG.active
        : level < nativeLevel ? SPGRID_CELL_FLAG.ghost : SPGRID_CELL_FLAG.multigridOnly);
    }
  }
  const transfers: SPGridTransferRecord[][] = [];
  for (let level = 0; level < levelCount - 1; level += 1) {
    const fine = [...maps[level].values()];
    const records: SPGridTransferRecord[] = [];
    for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
      const cell = fine[fineIndex];
      const targets: Array<{ coordinate: [number, number, number]; weight: number }> = [];
      if ((cell.flags & SPGRID_CELL_FLAG.ghost) !== 0) {
        targets.push({ coordinate: cell.coordinate.map((v) => Math.floor(v / 2)) as [number, number, number], weight: 1 });
      } else {
        // Cell-centred trilinear interpolation. At a boundary, clamped targets
        // deliberately remain duplicate records: accumulation then retains a
        // unit row sum and prolongation consumes the identical records.
        const axis = cell.coordinate.map((v) => {
          const base = Math.floor(v / 2), neighbour = Math.max(0, base + ((v & 1) === 0 ? -1 : 1));
          return [{ value: base, weight: 0.75 }, { value: neighbour, weight: 0.25 }] as const;
        });
        for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
          targets.push({ coordinate: [axis[0][x].value, axis[1][y].value, axis[2][z].value],
            weight: axis[0][x].weight * axis[1][y].weight * axis[2][z].weight });
        }
      }
      for (const target of targets) add(level + 1, target.coordinate, SPGRID_CELL_FLAG.multigridOnly);
    }
    const coarse = [...maps[level + 1].values()], coarseIndex = new Map(coarse.map((cell, index) => [coordinateKey(cell.coordinate), index]));
    for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
      const cell = fine[fineIndex];
      if ((cell.flags & SPGRID_CELL_FLAG.ghost) !== 0) {
        const key = coordinateKey(cell.coordinate.map((v) => Math.floor(v / 2)) as [number, number, number]);
        records.push({ fine: fineIndex, coarse: coarseIndex.get(key)!, weight: 1 });
        continue;
      }
      const axis = cell.coordinate.map((v) => {
        const base = Math.floor(v / 2), neighbour = Math.max(0, base + ((v & 1) === 0 ? -1 : 1));
        return [{ value: base, weight: 0.75 }, { value: neighbour, weight: 0.25 }] as const;
      });
      for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
        const key = coordinateKey([axis[0][x].value, axis[1][y].value, axis[2][z].value]);
        records.push({ fine: fineIndex, coarse: coarseIndex.get(key)!,
          weight: axis[0][x].weight * axis[1][y].weight * axis[2][z].weight });
      }
    }
    transfers.push(records);
  }
  return { levels: maps.map((map, level) => ({ scale: 2 ** level,
    coordinates: [...map.values()].map((entry) => entry.coordinate), flags: [...map.values()].map((entry) => entry.flags) })), transfers };
}

export function restrictSPGrid(fine: readonly number[], records: readonly SPGridTransferRecord[], coarseCount: number): number[] {
  const result = new Array<number>(coarseCount).fill(0);
  for (const record of records) result[record.coarse] += record.weight * fine[record.fine];
  return result;
}

/** Exact transpose of restrictSPGrid because it consumes the same immutable records. */
export function prolongSPGrid(coarse: readonly number[], records: readonly SPGridTransferRecord[], fineCount: number): number[] {
  const result = new Array<number>(fineCount).fill(0);
  for (const record of records) result[record.fine] += record.weight * coarse[record.coarse];
  return result;
}

/** Deterministic direct bottom oracle. No residual-dependent stopping is used. */
export function solveSPGridBottomLDLT(operator: readonly (readonly number[])[], rhs: readonly number[]): number[] {
  const n = operator.length;
  if (rhs.length !== n || operator.some((row) => row.length !== n)) throw new RangeError("bottom matrix dimensions disagree");
  const l = Array.from({ length: n }, () => new Array<number>(n).fill(0)), d = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < i; j += 1) {
      let value = operator[i][j];
      for (let k = 0; k < j; k += 1) value -= l[i][k] * d[k] * l[j][k];
      l[i][j] = value / d[j];
    }
    let diagonal = operator[i][i];
    for (let k = 0; k < i; k += 1) diagonal -= l[i][k] * l[i][k] * d[k];
    if (!(diagonal > 1e-14) || !Number.isFinite(diagonal)) throw new RangeError("bottom matrix is not SPD");
    d[i] = diagonal; l[i][i] = 1;
  }
  const y = new Array<number>(n), z = new Array<number>(n), x = new Array<number>(n);
  for (let i = 0; i < n; i += 1) { let value = rhs[i]; for (let j = 0; j < i; j += 1) value -= l[i][j] * y[j]; y[i] = value; }
  for (let i = 0; i < n; i += 1) z[i] = y[i] / d[i];
  for (let i = n - 1; i >= 0; i -= 1) { let value = z[i]; for (let j = i + 1; j < n; j += 1) value -= l[j][i] * x[j]; x[i] = value; }
  return x;
}

export interface OctreeSPGridVCyclePlan {
  readonly rowCapacity: number;
  readonly levelCount: number;
  readonly levelStride: number;
  readonly transferStride: number;
  readonly topologyBytes: number;
  readonly stateBytes: number;
  readonly allocatedBytes: number;
  readonly rowDispatch: readonly [number, number, number];
  readonly slotDispatch: readonly [number, number, number];
  readonly transferDispatch: readonly [number, number, number];
}

export interface OctreeSPGridVCycleOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  readonly finestCellWidth: number;
  readonly maximumLevels?: number;
  readonly preSmoothingIterations?: number;
  readonly postSmoothingIterations?: number;
  readonly bottomIterations?: number;
  readonly damping?: number;
}

export interface OctreeSPGridVCycleSource { readonly leafHeaders: GPUBuffer; readonly leafEntries: GPUBuffer }

const STATE_CHANNELS = 14;
const TOPOLOGY_HEADER_WORDS = 16;
/** Bounded so the worst-case 12-level state remains below 128 MiB. */
export const SPGRID_MAXIMUM_ROW_CAPACITY = 16_384;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}
function nextPowerOfTwo(value: number): number { let result = 1; while (result < value) result *= 2; return result; }
function dispatchFor(capacity: number): readonly [number, number, number] {
  const blocks = Math.ceil(capacity / 64), x = Math.min(65_535, Math.max(1, blocks));
  return [x, Math.max(1, Math.ceil(blocks / x)), 1];
}

export function planOctreeSPGridVCycle(options: Pick<OctreeSPGridVCycleOptions, "dimensions" | "rowCapacity" | "maximumLevels">): OctreeSPGridVCyclePlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "SPGrid row capacity");
  if (rowCapacity > SPGRID_MAXIMUM_ROW_CAPACITY) {
    throw new RangeError(`SPGrid row capacity exceeds the bounded ${SPGRID_MAXIMUM_ROW_CAPACITY}-row implementation`);
  }
  const width = Math.max(...options.dimensions.map((value) => positiveInteger(value, "SPGrid dimension")));
  const requiredLevels = Math.ceil(Math.log2(width)) + 1;
  if (requiredLevels > 12) throw new RangeError("SPGrid dimensions exceed the bounded 12-level exact-bottom hierarchy");
  if (options.maximumLevels !== undefined && options.maximumLevels < requiredLevels) {
    throw new RangeError("SPGrid maximumLevels would truncate the exact one-cell bottom level");
  }
  const levelCount = Math.max(2, requiredLevels);
  // Up to four sparse native/MG-only cells per live row at a level, held below
  // 50% hash occupancy. Overflow is fail-closed in solverControl.
  const levelStride = nextPowerOfTwo(rowCapacity * 8), transferStride = rowCapacity * 8;
  const rowMapWords = levelCount * rowCapacity, worklistWords = levelCount * levelStride;
  const transferWords = (levelCount - 1) * transferStride * 3;
  const topologyBytes = (TOPOLOGY_HEADER_WORDS + rowMapWords + worklistWords + transferWords) * 4;
  const stateBytes = STATE_CHANNELS * levelCount * levelStride * 4;
  return { rowCapacity, levelCount, levelStride, transferStride, topologyBytes, stateBytes,
    allocatedBytes: topologyBytes + stateBytes + levelCount * 32 + levelCount * 64,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(levelStride), transferDispatch: dispatchFor(transferStride) };
}

type PipelineName = "emitCells" | "buildTransfers" | "buildStencil" | "ensureDiagonal" | "finalizeIndirect"
  | "clearTopology" | "clearState" | "clearDispatch" | "clearCorrection"
  | "zeroVectors" | "seedRhs" | "applyA" | "applyB" | "jacobiAtoB" | "jacobiBtoA"
  | "formResidual" | "restrictResidual" | "exactBottom" | "prolongCorrection" | "publish";

const BINDINGS: Readonly<Record<PipelineName, readonly number[]>> = Object.freeze({
  clearTopology: [4], clearState: [5], clearDispatch: [6], clearCorrection: [0, 3, 9],
  emitCells: [0, 1, 3, 4, 5, 6, 7], buildTransfers: [0, 4, 5, 6, 7],
  buildStencil: [0, 1, 2, 3, 4, 5, 7], ensureDiagonal: [0, 4, 5, 6], finalizeIndirect: [0, 6],
  zeroVectors: [0, 4, 5, 6], seedRhs: [0, 3, 4, 5, 7, 8], applyA: [0, 4, 5, 6, 7],
  applyB: [0, 4, 5, 6, 7], jacobiAtoB: [0, 4, 5, 6, 7], jacobiBtoA: [0, 4, 5, 6, 7],
  formResidual: [0, 4, 5, 6, 7], restrictResidual: [0, 4, 5, 6, 7],
  exactBottom: [0, 4, 5, 6, 7],
  prolongCorrection: [0, 4, 5, 6, 7], publish: [0, 3, 4, 5, 7, 9],
});

type CachedGroup = { rowCount: GPUBuffer; control: GPUBuffer; rhs?: GPUBuffer; correction?: GPUBuffer; group: GPUBindGroup };

/**
 * A/B implementation of the paper-style native sparse pyramid. Setup is
 * intentionally GPU-idempotent and rebuilds from the captured authoritative
 * L1 rows whenever MGPCG invokes encodeSetup; no readback or fence is needed.
 */
export class WebGPUOctreeSPGridVCycle implements OctreeFirstOrderSPDVCycle {
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly plan: OctreeSPGridVCyclePlan;
  readonly allocatedBytes: number;
  /** Legacy MGPCG accounting name: ordered dispatch count, not transitions. */
  readonly encodedCorrectionPassCount: number;
  readonly encodedPassTransitionCount = 1;
  readonly encodedCorrectionPassTransitionCount = 1;
  readonly encodedSetupDispatchCount: number;
  readonly encodedCorrectionDispatchCount: number;
  readonly diagnostics: Readonly<{ levelCount: number; coarsestCapacity: number; maximumTransferRecordsPerLevel: number;
    correctionDispatchCount: number; correctionPassTransitions: number; bottomOperation: "exact-single-cell"; coarsestDegreesOfFreedom: 1 }>;
  private readonly capturedHeaders: GPUBuffer;
  private readonly capturedEntries: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatchMeta: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<PipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<string, CachedGroup>();
  private readonly pre: number;
  private readonly post: number;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly source: OctreeSPGridVCycleSource,
    options: OctreeSPGridVCycleOptions) {
    this.plan = planOctreeSPGridVCycle(options);
    if (!(options.finestCellWidth > 0) || !Number.isFinite(options.finestCellWidth)) throw new RangeError("SPGrid finest cell width must be positive");
    if (source.leafHeaders.size < this.plan.rowCapacity * 48 || source.leafEntries.size < 8) throw new RangeError("SPGrid L1 source capacity is too small");
    if ((source.leafHeaders.usage & GPUBufferUsage.COPY_SRC) === 0 || (source.leafEntries.usage & GPUBufferUsage.COPY_SRC) === 0) {
      throw new RangeError("SPGrid L1 source buffers require COPY_SRC capture usage");
    }
    const limits = device.limits;
    const storageLimit = limits?.maxStorageBufferBindingSize ?? Number.POSITIVE_INFINITY;
    const bufferLimit = limits?.maxBufferSize ?? Number.POSITIVE_INFINITY;
    if (this.plan.stateBytes > storageLimit || this.plan.topologyBytes > storageLimit
      || Math.max(this.plan.stateBytes, this.plan.topologyBytes) > bufferLimit) {
      throw new RangeError("SPGrid pyramid exceeds this device's storage-buffer limits");
    }
    this.pre = Math.max(1, Math.min(8, Math.round(options.preSmoothingIterations ?? 2)));
    this.post = Math.max(1, Math.min(8, Math.round(options.postSmoothingIterations ?? this.pre)));
    if (this.pre !== this.post) throw new RangeError("SPGrid pre/post smoothing must match to retain symmetry");
    if ((this.pre & 1) !== 0) throw new RangeError("SPGrid smoothing count must be even");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.capturedHeaders = device.createBuffer({ label: "SPGrid captured L1 headers", size: this.plan.rowCapacity * 48, usage: storage });
    this.capturedEntries = device.createBuffer({ label: "SPGrid captured L1 entries", size: source.leafEntries.size, usage: storage });
    this.topology = device.createBuffer({ label: "SPGrid native sparse topology/worklists/transfers", size: this.plan.topologyBytes, usage: storage });
    this.state = device.createBuffer({ label: "SPGrid six-face stencils and vectors", size: this.plan.stateBytes, usage: storage });
    this.dispatchMeta = device.createBuffer({ label: "SPGrid worklist counts and indirect dispatches", size: this.plan.levelCount * 32,
      usage: storage });
    const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
    this.params = Object.freeze(Array.from({ length: this.plan.levelCount }, (_, level) => {
      const buffer = device.createBuffer({ label: `SPGrid level ${level} parameters`, size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(16), floats = new Float32Array(words.buffer);
      words.set([options.dimensions[0], options.dimensions[1], options.dimensions[2], level,
        this.plan.rowCapacity, this.plan.levelCount, this.plan.levelStride, this.plan.transferStride,
        this.plan.rowDispatch[0], this.plan.slotDispatch[0], this.plan.transferDispatch[0], this.pre]);
      words[12] = this.post; words[13] = 1; floats[14] = damping; floats[15] = options.finestCellWidth;
      device.queue.writeBuffer(buffer, 0, words); return buffer;
    }));
    const shaderModule = device.createShaderModule({ label: "Paper native sparse SPGrid V-cycle", code: octreeSPGridVCycleShader });
    const make = (entryPoint: PipelineName) => device.createComputePipeline({ label: `SPGrid V-cycle · ${entryPoint}`,
      layout: "auto", compute: { module: shaderModule, entryPoint } });
    this.pipelines = Object.freeze(Object.fromEntries((Object.keys(BINDINGS) as PipelineName[]).map((name) => [name, make(name)])) as Record<PipelineName, GPUComputePipeline>);
    const l = this.plan.levelCount;
    // Three clears, finest-cell emission, one transfer build per adjacent
    // level, the finest stencil build, then diagonal/finalization per level.
    this.encodedSetupDispatchCount = 3 * l + 4;
    this.encodedCorrectionDispatchCount = l + 4 + (l - 1) * (2 * this.pre + 2 * this.post + 4);
    this.encodedCorrectionPassCount = this.encodedCorrectionDispatchCount;
    this.diagnostics = Object.freeze({ levelCount: l, coarsestCapacity: this.plan.levelStride,
      maximumTransferRecordsPerLevel: this.plan.transferStride, correctionDispatchCount: this.encodedCorrectionDispatchCount,
      correctionPassTransitions: 1, bottomOperation: "exact-single-cell" as const, coarsestDegreesOfFreedom: 1 as const });
    this.allocatedBytes = this.plan.allocatedBytes + this.capturedHeaders.size + this.capturedEntries.size;
  }

  encodeCapture(encoder: GPUCommandEncoder): void {
    this.assertLive();
    encoder.copyBufferToBuffer(this.source.leafHeaders, 0, this.capturedHeaders, 0, this.capturedHeaders.size);
    encoder.copyBufferToBuffer(this.source.leafEntries, 0, this.capturedEntries, 0, this.capturedEntries.size);
  }

  encodeSetup(encoder: GPUCommandEncoder, input: { solverControl: GPUBuffer; rowCount: GPUBuffer },
    sharedPass?: GPUComputePassEncoder): void {
    this.assertLive();
    const pass = sharedPass ?? encoder.beginComputePass({ label: "SPGrid V-cycle · rebuild native pyramid" });
    this.run(pass, "clearTopology", 0, input, dispatchFor(this.plan.topologyBytes / 4));
    this.run(pass, "clearState", 0, input, dispatchFor(this.plan.stateBytes / 4));
    this.run(pass, "clearDispatch", 0, input, dispatchFor(this.plan.levelCount * 8));
    this.run(pass, "emitCells", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) this.run(pass, "buildTransfers", level, input, this.plan.slotDispatch);
    this.run(pass, "buildStencil", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount; level += 1) {
      this.run(pass, "ensureDiagonal", level, input, this.plan.slotDispatch);
      this.run(pass, "finalizeIndirect", level, input, [1, 1, 1]);
    }
    if (!sharedPass) pass.end();
  }

  encodeCorrection(encoder: GPUCommandEncoder, input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer },
    sharedPass?: GPUComputePassEncoder): void {
    this.assertLive();
    const pass = sharedPass ?? encoder.beginComputePass({ label: "SPGrid V-cycle · one-pass symmetric correction" });
    this.run(pass, "clearCorrection", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount; level += 1) this.runIndirect(pass, "zeroVectors", level, input, false);
    this.run(pass, "seedRhs", 0, input, this.plan.rowDispatch);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) {
      this.smooth(pass, level, this.pre, input); this.runIndirect(pass, "applyA", level, input, false);
      this.runIndirect(pass, "formResidual", level, input, false); this.runIndirect(pass, "restrictResidual", level, input, true);
    }
    this.runIndirect(pass, "exactBottom", this.plan.levelCount - 1, input, false);
    for (let level = this.plan.levelCount - 2; level >= 0; level -= 1) {
      this.runIndirect(pass, "prolongCorrection", level, input, true); this.smooth(pass, level, this.post, input);
    }
    this.run(pass, "publish", 0, input, this.plan.rowDispatch); if (!sharedPass) pass.end();
  }

  private smooth(pass: GPUComputePassEncoder, level: number, iterations: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const b = (iteration & 1) !== 0; this.runIndirect(pass, b ? "applyB" : "applyA", level, input, false);
      this.runIndirect(pass, b ? "jacobiBtoA" : "jacobiAtoB", level, input, false);
    }
  }

  private runIndirect(pass: GPUComputePassEncoder, name: PipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }, transfer: boolean): void {
    // WebGPU usage scopes cover an entire compute pass: a buffer cannot be
    // writable storage while also supplying indirect arguments in that pass.
    // Keep the one-pass schedule and dispatch the bounded sparse capacities;
    // every kernel exits against the GPU-authored live count.
    this.bind(pass, name, level, input);
    pass.dispatchWorkgroups(...(transfer ? this.plan.transferDispatch : this.plan.slotDispatch));
  }
  private run(pass: GPUComputePassEncoder, name: PipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer },
    dispatch: readonly [number, number, number]): void {
    this.bind(pass, name, level, input); pass.dispatchWorkgroups(...dispatch);
  }
  private bind(pass: GPUComputePassEncoder, name: PipelineName, level: number,
    input: { rhs?: GPUBuffer; correction?: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    const pipeline = this.pipelines[name], key = `${name}:${level}`, cached = this.groups.get(key);
    let group = cached?.group;
    if (!cached || cached.rowCount !== input.rowCount || cached.control !== input.solverControl
      || cached.rhs !== input.rhs || cached.correction !== input.correction) {
      const buffers = [this.params[level], this.capturedHeaders, this.capturedEntries, input.rowCount, this.topology,
        this.state, this.dispatchMeta, input.solverControl, input.rhs, input.correction];
      group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: BINDINGS[name].map((binding) => ({
        binding, resource: { buffer: buffers[binding]! },
      })) });
      this.groups.set(key, { rowCount: input.rowCount, control: input.solverControl, rhs: input.rhs, correction: input.correction, group });
    }
    pass.setPipeline(pipeline); pass.setBindGroup(0, group!);
  }
  private assertLive(): void { if (this.destroyed) throw new Error("SPGrid V-cycle is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.groups.clear();
    this.capturedHeaders.destroy(); this.capturedEntries.destroy(); this.topology.destroy(); this.state.destroy(); this.dispatchMeta.destroy();
    for (const buffer of this.params) buffer.destroy();
  }
}

export const octreeSPGridVCycleShader = /* wgsl */ `
struct Params{dimsLevel:vec4u,capacity:vec4u,dispatchSmooth:vec4u,solve:vec2u,weights:vec2f}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct LeafEntry{row:u32,coefficient:f32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read> rowCounts:array<u32>;
@group(0) @binding(4) var<storage,read_write> topology:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> dispatchMeta:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> inputRhs:array<f32>;
@group(0) @binding(9) var<storage,read_write> outputCorrection:array<f32>;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const INVALID=0xffffffffu;
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;
const KEY=0u;const FLAGS=1u;const DIAG=2u;const XP=3u;const XM=4u;const YP=5u;const YM=6u;const ZP=7u;const ZM=8u;
const RHS=9u;const A=10u;const B=11u;const AX=12u;const RESIDUAL=13u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn report(flag:u32){atomicOr(&control[0],flag);}fn rows()->u32{return min(rowCounts[0],p.capacity.x);}fn level()->u32{return p.dimsLevel.w;}
fn stride()->u32{return p.capacity.z;}fn levels()->u32{return p.capacity.y;}fn transferStride()->u32{return p.capacity.w;}
fn rowIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.x*64u;}fn slotIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.y*64u;}
fn transferIndex(g:vec3u)->u32{return g.x+g.y*p.dispatchSmooth.z*64u;}fn at(c:u32,l:u32,s:u32)->u32{return(c*levels()+l)*stride()+s;}
fn loadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(atomicLoad(&state[at(c,l,s)]));}fn storef(c:u32,l:u32,s:u32,v:f32){atomicStore(&state[at(c,l,s)],bitcast<u32>(v));}
fn atomicAddF(index:u32,value:f32){if(!finite(value)){report(NONFINITE);return;}var old=atomicLoad(&state[index]);loop{let v=bitcast<f32>(old);
 let claim=atomicCompareExchangeWeak(&state[index],old,bitcast<u32>(v+value));if(claim.exchanged){return;}old=claim.old_value;}}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*p.capacity.x;}
fn transferBase()->u32{return workBase()+levels()*stride();}fn rowMap(l:u32,r:u32)->u32{return atomicLoad(&topology[rowMapBase()+l*p.capacity.x+r]);}
fn workSlot(l:u32,i:u32)->u32{return atomicLoad(&topology[workBase()+l*stride()+i]);}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+(l*transferStride()+i)*3u+w;}
fn count(l:u32)->u32{return atomicLoad(&dispatchMeta[l*8u]);}fn transferCount(l:u32)->u32{return atomicLoad(&dispatchMeta[l*8u+1u]);}
fn genericIndex(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dimsLevel.xyz+vec3u(s-1u))/s;}fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn hash(key:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return(h^(h>>15u))&(stride()-1u);}
fn mergeClass(index:u32,incoming:u32){var old=atomicLoad(&state[index]);loop{var merged=MG_ONLY;if((old&ACTIVE)!=0u||(incoming&ACTIVE)!=0u){merged=ACTIVE;}
 else if((old&GHOST)!=0u||(incoming&GHOST)!=0u){merged=GHOST;}let claim=atomicCompareExchangeWeak(&state[index],old,merged);if(claim.exchanged){return;}old=claim.old_value;}}
fn insert(l:u32,q:vec3u,flags:u32)->u32{let key=coordKey(min(q,dims(l)-vec3u(1u)),l);var slot=hash(key);for(var probe=0u;probe<256u;probe+=1u){
 let index=at(KEY,l,slot);var old=atomicLoad(&state[index]);if(old==key){mergeClass(at(FLAGS,l,slot),flags);return slot;}if(old==0u){let c=atomicCompareExchangeWeak(&state[index],0u,key);
  if(c.exchanged){mergeClass(at(FLAGS,l,slot),flags);let w=atomicAdd(&dispatchMeta[l*8u],1u);if(w>=stride()){report(OVERFLOW);return INVALID;}
   atomicStore(&topology[workBase()+l*stride()+w],slot);return slot;}old=c.old_value;if(old==key){mergeClass(at(FLAGS,l,slot),flags);return slot;}}
 slot=(slot+1u)&(stride()-1u);}report(OVERFLOW);return INVALID;}
fn find(l:u32,q:vec3u)->u32{let key=coordKey(q,l);var slot=hash(key);for(var probe=0u;probe<256u;probe+=1u){let old=atomicLoad(&state[at(KEY,l,slot)]);
 if(old==key){return slot;}if(old==0u){return INVALID;}slot=(slot+1u)&(stride()-1u);}return INVALID;}
fn appendTransfer(l:u32,fine:u32,coarse:u32,weight:f32){let i=atomicAdd(&dispatchMeta[l*8u+1u],1u);if(i>=transferStride()){report(OVERFLOW);return;}
 atomicStore(&topology[transferWord(l,i,0u)],fine);atomicStore(&topology[transferWord(l,i,1u)],coarse);atomicStore(&topology[transferWord(l,i,2u)],bitcast<u32>(weight));}
@compute @workgroup_size(64) fn clearTopology(@builtin(global_invocation_id) g:vec3u){let i=genericIndex(g);if(i<arrayLength(&topology)){atomicStore(&topology[i],0u);}}
@compute @workgroup_size(64) fn clearState(@builtin(global_invocation_id) g:vec3u){let i=genericIndex(g);if(i<arrayLength(&state)){atomicStore(&state[i],0u);}}
@compute @workgroup_size(64) fn clearDispatch(@builtin(global_invocation_id) g:vec3u){let i=genericIndex(g);if(i<arrayLength(&dispatchMeta)){atomicStore(&dispatchMeta[i],0u);}}
@compute @workgroup_size(64) fn clearCorrection(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()){outputCorrection[r]=0.0;}}
@compute @workgroup_size(64) fn emitCells(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r>=rows()||stopped()){return;}let h=headers[r];
 for(var l=0u;l<levels();l+=1u){let scale=1u<<l;let native=firstTrailingBit(h.size);let flag=select(select(MG_ONLY,ACTIVE,l==native),GHOST,l<native);
  let slot=insert(l,decode(coordKey(vec3u(h.cell%p.dimsLevel.x,(h.cell/p.dimsLevel.x)%p.dimsLevel.y,h.cell/(p.dimsLevel.x*p.dimsLevel.y))/scale,l),l),flag);
  if(slot!=INVALID){atomicStore(&topology[rowMapBase()+l*p.capacity.x+r],slot);}}}
@compute @workgroup_size(64) fn buildTransfers(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)||l+1u>=levels()||stopped()){return;}
 let fine=workSlot(l,i);let q=decode(atomicLoad(&state[at(KEY,l,fine)]),l);let flags=atomicLoad(&state[at(FLAGS,l,fine)]);if((flags&GHOST)!=0u){let c=insert(l+1u,q/2u,MG_ONLY);
  if(c!=INVALID){appendTransfer(l,fine,c,1.0);}return;}let base=q/2u;let side=vec3i(select(-1,1,(q.x&1u)!=0u),select(-1,1,(q.y&1u)!=0u),select(-1,1,(q.z&1u)!=0u));
 for(var corner=0u;corner<8u;corner+=1u){var targetCoord=vec3i(base);var weight=1.0;for(var axis=0u;axis<3u;axis+=1u){if((corner&(1u<<axis))!=0u){targetCoord[axis]+=side[axis];weight*=0.25;}else{weight*=0.75;}}
  let cq=vec3u(max(targetCoord,vec3i(0)));let c=insert(l+1u,cq,MG_ONLY);if(c!=INVALID){appendTransfer(l,fine,c,weight);}}}
fn addFace(l:u32,own:u32,a:vec3u,b:vec3u,c:f32){let d=vec3i(b)-vec3i(a);var ch=0u;if(all(d==vec3i(1,0,0))){ch=XP;}else if(all(d==vec3i(-1,0,0))){ch=XM;}
 else if(all(d==vec3i(0,1,0))){ch=YP;}else if(all(d==vec3i(0,-1,0))){ch=YM;}else if(all(d==vec3i(0,0,1))){ch=ZP;}else if(all(d==vec3i(0,0,-1))){ch=ZM;}else{return;}atomicAddF(at(ch,l,own),c);}
@compute @workgroup_size(64) fn buildStencil(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r>=rows()||stopped()){return;}let h=headers[r];var sum=0.0;
 for(var j=0u;j<h.entryCount;j+=1u){sum+=entries[h.entryStart+j].coefficient;}let anchor=max(0.0,h.diagonal-sum);for(var l=0u;l<levels();l+=1u){let own=rowMap(l,r);
  atomicAddF(at(DIAG,l,own),anchor);let ownQ=decode(atomicLoad(&state[at(KEY,l,own)]),l);for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=rows()){continue;}
   let other=rowMap(l,e.row);if(other!=own){atomicAddF(at(DIAG,l,own),e.coefficient);addFace(l,own,ownQ,decode(atomicLoad(&state[at(KEY,l,other)]),l),e.coefficient);}}}}
@compute @workgroup_size(64) fn ensureDiagonal(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)){return;}let s=workSlot(l,i);
 if(loadf(DIAG,l,s)<=1e-20){storef(DIAG,l,s,1.0);}}
@compute @workgroup_size(1) fn finalizeIndirect(){let l=level();let n=count(l);let t=transferCount(l);let nb=(n+63u)/64u;let tb=(t+63u)/64u;
 atomicStore(&dispatchMeta[l*8u+2u],min(65535u,max(1u,nb)));atomicStore(&dispatchMeta[l*8u+3u],max(1u,(nb+65534u)/65535u));atomicStore(&dispatchMeta[l*8u+4u],1u);
 atomicStore(&dispatchMeta[l*8u+5u],min(65535u,max(1u,tb)));atomicStore(&dispatchMeta[l*8u+6u],max(1u,(tb+65534u)/65535u));atomicStore(&dispatchMeta[l*8u+7u],1u);}
@compute @workgroup_size(64) fn zeroVectors(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>=count(l)){return;}let s=workSlot(l,i);
 for(var c=RHS;c<=RESIDUAL;c+=1u){storef(c,l,s,0.0);}}
@compute @workgroup_size(64) fn seedRhs(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let v=inputRhs[r];if(!finite(v)){report(NONFINITE);}else{atomicAddF(at(RHS,0u,rowMap(0u,r)),v);}}}
fn neighbour(l:u32,q:vec3u,axis:u32,positive:bool)->u32{var v=vec3i(q);v[axis]+=select(-1,1,positive);if(any(v<vec3i(0))||any(vec3u(v)>=dims(l))){return INVALID;}return find(l,vec3u(v));}
fn apply(slot:u32,source:u32){let l=level();let q=decode(atomicLoad(&state[at(KEY,l,slot)]),l);var value=loadf(DIAG,l,slot)*loadf(source,l,slot);
 let channels=array<u32,6>(XP,XM,YP,YM,ZP,ZM);for(var k=0u;k<6u;k+=1u){let c=loadf(channels[k],l,slot);let other=neighbour(l,q,k/2u,(k&1u)==0u);
  if(other!=INVALID){value-=c*loadf(source,l,other);}}storef(AX,l,slot,value);}
fn smoothable(l:u32,s:u32)->bool{return(atomicLoad(&state[at(FLAGS,l,s)])&GHOST)==0u;}
@compute @workgroup_size(64) fn applyA(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);if(smoothable(l,s)){apply(s,A);}}}
@compute @workgroup_size(64) fn applyB(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);if(smoothable(l,s)){apply(s,B);}}}
fn jacobi(slot:u32,src:u32,dst:u32){let l=level();let d=loadf(DIAG,l,slot);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(src,l,slot)+p.weights.x*(loadf(RHS,l,slot)-loadf(AX,l,slot))/d;
 if(!finite(x)){report(NONFINITE);}else{storef(dst,l,slot,x);}}
@compute @workgroup_size(64) fn jacobiAtoB(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);if(smoothable(l,s)){jacobi(s,A,B);}}}
@compute @workgroup_size(64) fn jacobiBtoA(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);if(smoothable(l,s)){jacobi(s,B,A);}}}
@compute @workgroup_size(64) fn formResidual(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i<count(l)&&!stopped()){let s=workSlot(l,i);storef(RESIDUAL,l,s,select(loadf(RHS,l,s)-loadf(AX,l,s),loadf(RHS,l,s),!smoothable(l,s)));}}
@compute @workgroup_size(64) fn restrictResidual(@builtin(global_invocation_id) g:vec3u){let i=transferIndex(g);let l=level();if(i>=transferCount(l)||stopped()){return;}
 let f=atomicLoad(&topology[transferWord(l,i,0u)]);let c=atomicLoad(&topology[transferWord(l,i,1u)]);let w=bitcast<f32>(atomicLoad(&topology[transferWord(l,i,2u)]));atomicAddF(at(RHS,l+1u,c),w*loadf(RESIDUAL,l,f));}
@compute @workgroup_size(64) fn exactBottom(@builtin(global_invocation_id) g:vec3u){let i=slotIndex(g);let l=level();if(i>0u||stopped()){return;}if(count(l)!=1u){report(NONPOSITIVE);return;}
 let s=workSlot(l,0u);let d=loadf(DIAG,l,s);if(!(d>0.0)){report(NONPOSITIVE);return;}let x=loadf(RHS,l,s)/d;if(!finite(x)){report(NONFINITE);}else{storef(A,l,s,x);}}
@compute @workgroup_size(64) fn prolongCorrection(@builtin(global_invocation_id) g:vec3u){let i=transferIndex(g);let l=level();if(i>=transferCount(l)||stopped()){return;}
 let f=atomicLoad(&topology[transferWord(l,i,0u)]);let c=atomicLoad(&topology[transferWord(l,i,1u)]);let w=bitcast<f32>(atomicLoad(&topology[transferWord(l,i,2u)]));atomicAddF(at(A,l,f),w*loadf(A,l+1u,c));}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) g:vec3u){let r=rowIndex(g);if(r<rows()&&!stopped()){let v=loadf(A,0u,rowMap(0u,r));if(!finite(v)){report(NONFINITE);}else{outputCorrection[r]=v;}}}
`;
