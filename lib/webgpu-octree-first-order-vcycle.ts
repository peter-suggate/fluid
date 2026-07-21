import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-mgpcg";

export interface OctreeFirstOrderVCyclePlan {
  readonly rowCapacity: number;
  readonly hierarchyStride: number;
  readonly levelCount: number;
  readonly stateBytes: number;
  readonly allocatedBytes: number;
  readonly rowDispatch: readonly [number, number, number];
  readonly slotDispatch: readonly [number, number, number];
}

export interface OctreeFirstOrderVCycleOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  /** Isotropic physical width of one finest-grid cell. */
  readonly finestCellWidth: number;
  readonly maximumLevels?: number;
  readonly preSmoothingIterations?: number;
  readonly postSmoothingIterations?: number;
  readonly coarsestIterations?: number;
  readonly damping?: number;
}

export interface OctreeFirstOrderVCycleSource {
  readonly leafHeaders: GPUBuffer;
  readonly leafEntries: GPUBuffer;
}

export function firstOrderOctreeAxisCoefficient(aOrigin: readonly [number, number, number], aSize: number,
  bOrigin: readonly [number, number, number], bSize: number, finestCellWidth: number): number {
  const overlap = (axis: number) => Math.max(0, Math.min(aOrigin[axis] + aSize, bOrigin[axis] + bSize)
    - Math.max(aOrigin[axis], bOrigin[axis]));
  let touchingAxis = -1;
  for (let axis = 0; axis < 3; axis += 1) {
    const touches = aOrigin[axis] + aSize === bOrigin[axis] || bOrigin[axis] + bSize === aOrigin[axis];
    if (!touches) continue;
    if (touchingAxis !== -1) return 0;
    touchingAxis = axis;
  }
  if (touchingAxis < 0) return 0;
  const transverse = [0, 1, 2].filter((axis) => axis !== touchingAxis);
  const sharedAreaCells = overlap(transverse[0]) * overlap(transverse[1]);
  if (!(sharedAreaCells > 0)) return 0;
  const centerDistanceCells = 0.5 * (aSize + bSize);
  return finestCellWidth * sharedAreaCells / centerDistanceCells;
}

/** CPU oracle for the first-order ghost-fluid free-surface diagonal anchor. */
export function firstOrderGhostFluidBoundaryCoefficient(area: number, fullCenterDistance: number,
  liquidPhi: number, airPhi: number, openFraction = 1): number {
  if (!(area >= 0) || !(fullCenterDistance > 0) || !(openFraction >= 0 && openFraction <= 1)
    || !(liquidPhi < 0) || !(airPhi >= 0)
    || ![area, fullCenterDistance, liquidPhi, airPhi, openFraction].every(Number.isFinite)) {
    throw new RangeError("first-order ghost-fluid boundary inputs are invalid");
  }
  const theta = Math.min(1, Math.max(0.01,
    Math.abs(liquidPhi) / Math.max(Math.abs(liquidPhi) + Math.abs(airPhi), 1e-12)));
  return openFraction * area / (theta * fullCenterDistance);
}

/** CPU oracle for the adjacent-level transfer used by focused SPD tests. */
export function ghostValuePropagate(coarse: readonly number[], fineToCoarse: readonly number[]): number[] {
  return fineToCoarse.map((parent) => coarse[parent] ?? 0);
}

/** Exact transpose of ghostValuePropagate. */
export function ghostValueAccumulate(fine: readonly number[], fineToCoarse: readonly number[], coarseCount: number): number[] {
  const coarse = new Array<number>(coarseCount).fill(0);
  for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) coarse[fineToCoarse[fineIndex]] += fine[fineIndex];
  return coarse;
}

export type DenseSymmetricOperator = readonly (readonly number[])[];

export function galerkinCoarsen(operator: DenseSymmetricOperator, fineToCoarse: readonly number[], coarseCount: number): number[][] {
  const coarse = Array.from({ length: coarseCount }, () => new Array<number>(coarseCount).fill(0));
  for (let i = 0; i < operator.length; i += 1) for (let j = 0; j < operator.length; j += 1) {
    coarse[fineToCoarse[i]][fineToCoarse[j]] += operator[i][j];
  }
  return coarse;
}

function multiplyDense(operator: DenseSymmetricOperator, value: readonly number[]): number[] {
  return operator.map((row) => row.reduce((sum, coefficient, column) => sum + coefficient * value[column], 0));
}

function jacobiDense(operator: DenseSymmetricOperator, rhs: readonly number[], initial: readonly number[],
  iterations: number, damping: number): number[] {
  let value = [...initial];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const product = multiplyDense(operator, value);
    value = value.map((entry, row) => entry + damping * (rhs[row] - product[row]) / operator[row][row]);
  }
  return value;
}

/** Symmetric Galerkin V-cycle oracle matching the GPU schedule. */
export function applyFirstOrderVCycleOracle(operator: DenseSymmetricOperator,
  adjacentMaps: readonly (readonly number[])[], rhs: readonly number[], smoothingIterations = 2,
  coarsestIterations = 16, damping = 2 / 3): number[] {
  const recurse = (levelOperator: DenseSymmetricOperator, level: number, levelRhs: readonly number[]): number[] => {
    if (level >= adjacentMaps.length) {
      return jacobiDense(levelOperator, levelRhs, new Array(levelRhs.length).fill(0), coarsestIterations, damping);
    }
    const map = adjacentMaps[level], coarseCount = Math.max(...map) + 1;
    let value = jacobiDense(levelOperator, levelRhs, new Array(levelRhs.length).fill(0), smoothingIterations, damping);
    const residual = multiplyDense(levelOperator, value).map((product, row) => levelRhs[row] - product);
    const coarseOperator = galerkinCoarsen(levelOperator, map, coarseCount);
    const correction = recurse(coarseOperator, level + 1, ghostValueAccumulate(residual, map, coarseCount));
    const prolonged = ghostValuePropagate(correction, map); value = value.map((entry, row) => entry + prolonged[row]);
    return jacobiDense(levelOperator, levelRhs, value, smoothingIterations, damping);
  };
  return recurse(operator, 0, rhs);
}

const STATE_CHANNELS = 8;
const MAP = 0, KEYS = 1, COUNTS = 2, DIAGONAL = 3;
const RHS = 4, SOLUTION_A = 5, SOLUTION_B = 6, WORK = 7;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function dispatchFor(capacity: number): readonly [number, number, number] {
  const blocks = Math.ceil(capacity / 64), x = Math.min(65_535, Math.max(1, blocks));
  return [x, Math.max(1, Math.ceil(blocks / x)), 1];
}

export function planOctreeFirstOrderVCycle(options: Pick<OctreeFirstOrderVCycleOptions,
  "dimensions" | "rowCapacity" | "maximumLevels">): OctreeFirstOrderVCyclePlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "L1 V-cycle row capacity");
  const width = Math.max(...options.dimensions.map((value) => positiveInteger(value, "L1 V-cycle dimension")));
  const levelCount = Math.min(Math.max(2, options.maximumLevels ?? 12), Math.ceil(Math.log2(width)) + 1);
  // Open-addressed level maps remain below 50% occupancy even when a level
  // cannot aggregate any live rows. This is bounded by live-row capacity, not
  // by the logical finest-domain volume.
  const hierarchyStride = nextPowerOfTwo(rowCapacity * 2);
  const stateBytes = STATE_CHANNELS * levelCount * hierarchyStride * 4;
  return {
    rowCapacity, hierarchyStride, levelCount, stateBytes,
    allocatedBytes: stateBytes + levelCount * 64,
    rowDispatch: dispatchFor(rowCapacity), slotDispatch: dispatchFor(hierarchyStride),
  };
}

type PipelineName = "setupMaps" | "setupDiagonal" | "copyRhs" | "applyA" | "applyB"
  | "jacobiAtoB" | "jacobiBtoA" | "formResidual" | "ghostAccumulate"
  | "ghostPropagate" | "publish";
const PIPELINE_BINDINGS: Readonly<Record<PipelineName, readonly number[]>> = Object.freeze({
  setupMaps: [0, 1, 3, 4, 5], setupDiagonal: [0, 1, 2, 3, 4, 5],
  copyRhs: [0, 3, 4, 5, 6], applyA: [0, 1, 2, 3, 4, 5], applyB: [0, 1, 2, 3, 4, 5],
  jacobiAtoB: [0, 4, 5], jacobiBtoA: [0, 4, 5], formResidual: [0, 4, 5],
  ghostAccumulate: [0, 3, 4, 5], ghostPropagate: [0, 3, 4, 5], publish: [0, 3, 4, 5, 7],
});

/**
 * Sparse first-order Galerkin V-cycle for the middle operation of Section 4.3.
 *
 * Each hierarchy level is a sparse uniform dyadic grid keyed only by live
 * octree rows. Piecewise-constant GhostValuePropagate is P; the adjacent-level
 * GhostValueAccumulate operation is exactly P^T. Coarse operators are applied
 * matrix-free as P^T L1 P, preserving symmetry without storing coarse edges.
 */
export class WebGPUOctreeFirstOrderVCycle implements OctreeFirstOrderSPDVCycle {
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly plan: OctreeFirstOrderVCyclePlan;
  readonly allocatedBytes: number;
  private readonly state: GPUBuffer;
  private readonly firstOrderHeaders: GPUBuffer;
  private readonly firstOrderEntries: GPUBuffer;
  private readonly levelParams: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<PipelineName, GPUComputePipeline>>;
  private readonly preIterations: number;
  private readonly postIterations: number;
  private readonly coarsestIterations: number;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly source: OctreeFirstOrderVCycleSource,
    options: OctreeFirstOrderVCycleOptions) {
    this.plan = planOctreeFirstOrderVCycle(options);
    this.allocatedBytes = this.plan.allocatedBytes;
    if (source.leafHeaders.size < this.plan.rowCapacity * 48 || source.leafEntries.size < 8) {
      throw new RangeError("L1 V-cycle source row capacity is too small");
    }
    if ((source.leafHeaders.usage & GPUBufferUsage.COPY_SRC) === 0
      || (source.leafEntries.usage & GPUBufferUsage.COPY_SRC) === 0) {
      throw new RangeError("L1 V-cycle source rows must support ordered COPY_SRC capture");
    }
    this.preIterations = Math.max(1, Math.min(8, Math.round(options.preSmoothingIterations ?? 2)));
    this.postIterations = Math.max(1, Math.min(8, Math.round(options.postSmoothingIterations ?? this.preIterations)));
    if (this.preIterations !== this.postIterations) {
      throw new RangeError("L1 V-cycle requires matching pre/post smoothing counts");
    }
    this.coarsestIterations = Math.max(2, Math.min(64, Math.round(options.coarsestIterations ?? 16)));
    if ((this.preIterations & 1) !== 0 || (this.coarsestIterations & 1) !== 0) {
      throw new RangeError("L1 V-cycle smoothing counts must be even so canonical solutions end in A");
    }
    if (!(options.finestCellWidth > 0) || !Number.isFinite(options.finestCellWidth)) {
      throw new RangeError("L1 V-cycle finest cell width must be finite and positive");
    }
    const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const snapshotUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.firstOrderHeaders = device.createBuffer({ label: "Octree Section 4.3 captured L1 headers",
      size: this.plan.rowCapacity * 48, usage: snapshotUsage });
    this.firstOrderEntries = device.createBuffer({ label: "Octree Section 4.3 captured L1 entries",
      size: source.leafEntries.size, usage: snapshotUsage });
    this.allocatedBytes += this.firstOrderHeaders.size + this.firstOrderEntries.size;
    this.state = device.createBuffer({ label: "Octree Section 4.3 sparse L1 V-cycle state",
      size: Math.max(4, this.plan.stateBytes), usage: storage });
    this.levelParams = Object.freeze(Array.from({ length: this.plan.levelCount }, (_, level) => {
      const buffer = device.createBuffer({ label: `Octree L1 V-cycle level ${level} params`, size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(16), floats = new Float32Array(words.buffer);
      words[0] = options.dimensions[0]; words[1] = options.dimensions[1]; words[2] = options.dimensions[2]; words[3] = level;
      words[4] = this.plan.rowCapacity; words[5] = this.plan.hierarchyStride; words[6] = this.plan.levelCount;
      words[7] = this.plan.rowDispatch[0]; words[8] = this.plan.slotDispatch[0];
      words[9] = this.preIterations; words[10] = this.postIterations; words[11] = this.coarsestIterations;
      floats[12] = damping; floats[13] = 1e-30; floats[14] = options.finestCellWidth;
      device.queue.writeBuffer(buffer, 0, words); return buffer;
    }));
    const module = device.createShaderModule({ label: "Octree sparse first-order V-cycle", code: octreeFirstOrderVCycleShader });
    const pipeline = (entryPoint: PipelineName) => device.createComputePipeline({
      label: `Octree L1 V-cycle · ${entryPoint}`, layout: "auto", compute: { module, entryPoint },
    });
    this.pipelines = Object.freeze({
      setupMaps: pipeline("setupMaps"), setupDiagonal: pipeline("setupDiagonal"),
      copyRhs: pipeline("copyRhs"), applyA: pipeline("applyA"), applyB: pipeline("applyB"),
      jacobiAtoB: pipeline("jacobiAtoB"), jacobiBtoA: pipeline("jacobiBtoA"),
      formResidual: pipeline("formResidual"), ghostAccumulate: pipeline("ghostAccumulate"),
      ghostPropagate: pipeline("ghostPropagate"), publish: pipeline("publish"),
    });
  }

  /**
   * Capture the Cartesian Losasso/GFM rows after compact assembly and before
   * authoritative power publication replaces the shared LeafHeader/Entry ABI
   * with L2 coefficients. The copy is ordered in the same command encoder.
   */
  encodeCapture(encoder: GPUCommandEncoder): void {
    this.assertLive();
    encoder.copyBufferToBuffer(this.source.leafHeaders, 0, this.firstOrderHeaders, 0,
      this.plan.rowCapacity * 48);
    encoder.copyBufferToBuffer(this.source.leafEntries, 0, this.firstOrderEntries, 0,
      this.source.leafEntries.size);
  }

  encodeSetup(encoder: GPUCommandEncoder, input: { solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    this.assertLive(); encoder.clearBuffer(this.state);
    this.runRows(encoder, "setupMaps", 0, input.rowCount, input.solverControl);
    this.runRows(encoder, "setupDiagonal", 0, input.rowCount, input.solverControl);
  }

  encodeCorrection(encoder: GPUCommandEncoder, input: {
    rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer;
  }): void {
    this.assertLive();
    // Preserve maps/keys/member counts/diagonals; reset all per-application
    // vectors. Contiguous channel storage makes this one bounded clear.
    const vectorOffset = this.channelOffset(RHS);
    encoder.clearBuffer(this.state, vectorOffset, this.plan.stateBytes - vectorOffset);
    this.runRows(encoder, "copyRhs", 0, input.rowCount, input.solverControl, input.rhs, input.correction);
    for (let level = 0; level < this.plan.levelCount - 1; level += 1) {
      this.smooth(encoder, level, this.preIterations, input);
      this.applyOperator(encoder, level, false, input);
      this.runSlots(encoder, "formResidual", level, input.rowCount, input.solverControl, input.rhs, input.correction);
      this.runRows(encoder, "ghostAccumulate", level, input.rowCount, input.solverControl, input.rhs, input.correction);
    }
    this.smooth(encoder, this.plan.levelCount - 1, this.coarsestIterations, input);
    for (let level = this.plan.levelCount - 2; level >= 0; level -= 1) {
      this.runRows(encoder, "ghostPropagate", level, input.rowCount, input.solverControl, input.rhs, input.correction);
      this.smooth(encoder, level, this.postIterations, input);
    }
    this.runRows(encoder, "publish", 0, input.rowCount, input.solverControl, input.rhs, input.correction);
  }

  private smooth(encoder: GPUCommandEncoder, level: number, iterations: number,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const useB = (iteration & 1) !== 0;
      this.applyOperator(encoder, level, useB, input);
      this.runSlots(encoder, useB ? "jacobiBtoA" : "jacobiAtoB", level,
        input.rowCount, input.solverControl, input.rhs, input.correction);
    }
  }

  private applyOperator(encoder: GPUCommandEncoder, level: number, useB: boolean,
    input: { rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer; rowCount: GPUBuffer }): void {
    encoder.clearBuffer(this.state, this.levelOffset(WORK, level), this.plan.hierarchyStride * 4);
    this.runRows(encoder, useB ? "applyB" : "applyA", level,
      input.rowCount, input.solverControl, input.rhs, input.correction);
  }

  private runRows(encoder: GPUCommandEncoder, name: PipelineName, level: number,
    rowCount: GPUBuffer, solverControl: GPUBuffer, rhs?: GPUBuffer, correction?: GPUBuffer): void {
    this.run(encoder, name, level, this.plan.rowDispatch, rowCount, solverControl, rhs, correction);
  }
  private runSlots(encoder: GPUCommandEncoder, name: PipelineName, level: number,
    rowCount: GPUBuffer, solverControl: GPUBuffer, rhs?: GPUBuffer, correction?: GPUBuffer): void {
    this.run(encoder, name, level, this.plan.slotDispatch, rowCount, solverControl, rhs, correction);
  }
  private run(encoder: GPUCommandEncoder, name: PipelineName, level: number,
    dispatch: readonly [number, number, number], rowCount: GPUBuffer, solverControl: GPUBuffer,
    rhs?: GPUBuffer, correction?: GPUBuffer): void {
    const pipeline = this.pipelines[name];
    const candidates = [this.levelParams[level], this.firstOrderHeaders, this.firstOrderEntries,
      rowCount, this.state, solverControl, rhs, correction];
    const entries: GPUBindGroupEntry[] = PIPELINE_BINDINGS[name].map((binding) => ({
      binding, resource: { buffer: candidates[binding]! },
    }));
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const pass = encoder.beginComputePass({ label: pipeline.label }); pass.setPipeline(pipeline);
    pass.setBindGroup(0, group); pass.dispatchWorkgroups(...dispatch); pass.end();
  }

  private channelOffset(channel: number): number {
    return channel * this.plan.levelCount * this.plan.hierarchyStride * 4;
  }
  private levelOffset(channel: number, level: number): number {
    return this.channelOffset(channel) + level * this.plan.hierarchyStride * 4;
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Octree L1 V-cycle is destroyed"); }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.state.destroy();
    this.firstOrderHeaders.destroy(); this.firstOrderEntries.destroy(); for (const buffer of this.levelParams) buffer.destroy(); }
}

export const octreeFirstOrderVCycleShader = /* wgsl */ `
struct Params { dimsLevel:vec4u, capacity:vec4u, solve:vec4u, weights:vec4f }
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct LeafEntry { row:u32,coefficient:f32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read> rowCounts:array<u32>;
@group(0) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> solverControl:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read> inputRhs:array<f32>;
@group(0) @binding(7) var<storage,read_write> outputCorrection:array<f32>;
const MAP:u32=0u;const KEYS:u32=1u;const COUNTS:u32=2u;const DIAGONAL:u32=3u;
const RHS:u32=4u;const SOLUTION_A:u32=5u;const SOLUTION_B:u32=6u;const WORK:u32=7u;
const INVALID:u32=0xffffffffu;const HIERARCHY_OVERFLOW:u32=2u;const NONFINITE:u32=4u;const NONPOSITIVE:u32=8u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn report(flag:u32){atomicOr(&solverControl[0],flag);}
fn stopped()->bool{return atomicLoad(&solverControl[0])!=0u||atomicLoad(&solverControl[1])!=0u;}
fn rows()->u32{return min(select(0u,rowCounts[0],arrayLength(&rowCounts)>0u),params.capacity.x);}
fn stride()->u32{return params.capacity.y;}fn levels()->u32{return params.capacity.z;}fn level()->u32{return params.dimsLevel.w;}
fn rowIndex(gid:vec3u)->u32{return gid.x+gid.y*params.capacity.w*64u;}fn slotIndex(gid:vec3u)->u32{return gid.x+gid.y*params.solve.x*64u;}
fn at(channel:u32,l:u32,slot:u32)->u32{return (channel*levels()+l)*stride()+slot;}
fn map(row:u32,l:u32)->u32{return atomicLoad(&state[at(MAP,l,row)]);}
fn loadFloat(channel:u32,l:u32,slot:u32)->f32{return bitcast<f32>(atomicLoad(&state[at(channel,l,slot)]));}
fn storeFloat(channel:u32,l:u32,slot:u32,value:f32){atomicStore(&state[at(channel,l,slot)],bitcast<u32>(value));}
fn atomicAddFloat(index:u32,value:f32){if(!finite(value)){report(NONFINITE);return;}var old=atomicLoad(&state[index]);loop{
  let current=bitcast<f32>(old);if(!finite(current)){report(NONFINITE);return;}let result=atomicCompareExchangeWeak(&state[index],old,bitcast<u32>(current+value));
  if(result.exchanged){return;}old=result.old_value;}}
fn coord(cell:u32)->vec3u{let nx=params.dimsLevel.x;let ny=params.dimsLevel.y;return vec3u(cell%nx,(cell/nx)%ny,cell/(nx*ny));}
fn aggregateKey(row:u32,l:u32)->u32{let block=1u<<l;let q=coord(headers[row].cell)/block;let d=(params.dimsLevel.xyz+vec3u(block-1u))/block;
  return q.x+d.x*(q.y+d.y*q.z);}
fn hash(key:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return (h^(h>>15u))&(stride()-1u);}
fn findAggregate(row:u32,l:u32)->u32{let key=aggregateKey(row,l)+1u;var slot=hash(key);for(var probe=0u;probe<128u;probe+=1u){
  let index=at(KEYS,l,slot);let old=atomicLoad(&state[index]);if(old==key){return slot;}if(old==0u){let claim=atomicCompareExchangeWeak(&state[index],0u,key);
    if(claim.exchanged||claim.old_value==key){return slot;}}slot=(slot+1u)&(stride()-1u);}report(HIERARCHY_OVERFLOW);return INVALID;}
fn overlap(a0:u32,a1:u32,b0:u32,b1:u32)->u32{let lo=max(a0,b0);let hi=min(a1,b1);return select(0u,hi-lo,hi>lo);}
fn axisCoefficient(row:u32,other:u32)->f32{let a=coord(headers[row].cell);let b=coord(headers[other].cell);let sa=headers[row].size;let sb=headers[other].size;
  let ox=overlap(a.x,a.x+sa,b.x,b.x+sb);let oy=overlap(a.y,a.y+sa,b.y,b.y+sb);let oz=overlap(a.z,a.z+sa,b.z,b.z+sb);
  let tx=(a.x+sa==b.x||b.x+sb==a.x)&&oy>0u&&oz>0u;let ty=(a.y+sa==b.y||b.y+sb==a.y)&&ox>0u&&oz>0u;
  let tz=(a.z+sa==b.z||b.z+sb==a.z)&&ox>0u&&oy>0u;if(select(0u,1u,tx)+select(0u,1u,ty)+select(0u,1u,tz)!=1u){return 0.0;}
  var area=0u;if(tx){area=oy*oz;}else if(ty){area=ox*oz;}else{area=ox*oy;}
  return params.weights.z*f32(area)/(0.5*f32(sa+sb));}
fn firstOrderCoefficient(row:u32,e:LeafEntry)->f32{if(e.row>=rows()||!finite(e.coefficient)||e.coefficient<0.0){report(NONFINITE);return 0.0;}
  let geometric=axisCoefficient(row,e.row);if(e.coefficient>0.0&&geometric<=0.0){report(NONPOSITIVE);return 0.0;}return e.coefficient;}
// These are captured Cartesian rows, ordered before L2 power publication in
// the same command encoder. Their unmatched diagonal is therefore the exact
// Losasso/GFM L1 free-surface anchor, never an inherited power coefficient.
fn firstOrderDiagonal(row:u32)->f32{let value=headers[row].diagonal;if(!finite(value)||value<=params.weights.y){report(NONPOSITIVE);return 0.0;}return value;}
fn boundaryAnchor(row:u32)->f32{let h=headers[row];var coupled=0.0;for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];
  if(e.row>=rows()||!finite(e.coefficient)||e.coefficient<0.0){report(NONFINITE);continue;}coupled+=e.coefficient;}
  let value=firstOrderDiagonal(row)-coupled;let scale=max(1.0,max(abs(firstOrderDiagonal(row)),abs(coupled)));
  if(!finite(value)||value < -2e-5*scale){report(NONPOSITIVE);return 0.0;}return max(0.0,value);}
fn solution(channel:u32,l:u32,slot:u32)->f32{return loadFloat(channel,l,slot);}
fn activeSlot(slot:u32,l:u32)->bool{return slot<stride()&&atomicLoad(&state[at(COUNTS,l,slot)])>0u;}

@compute @workgroup_size(64) fn setupMaps(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=rows()||stopped()){return;}
  atomicStore(&state[at(MAP,0u,row)],row);atomicStore(&state[at(COUNTS,0u,row)],1u);for(var l=1u;l<levels();l+=1u){let slot=findAggregate(row,l);
    if(slot==INVALID){return;}atomicStore(&state[at(MAP,l,row)],slot);atomicAdd(&state[at(COUNTS,l,slot)],1u);}}
@compute @workgroup_size(64) fn setupDiagonal(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=rows()||stopped()){return;}
  let h=headers[row];let anchor=boundaryAnchor(row);for(var l=0u;l<levels();l+=1u){let own=map(row,l);var contribution=anchor;
    for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=rows()){report(NONFINITE);continue;}let coefficient=firstOrderCoefficient(row,e);
      if(coefficient>0.0&&map(e.row,l)!=own){contribution+=coefficient;}}
    if(l==0u){let expected=firstOrderDiagonal(row);let scale=max(1.0,max(abs(expected),abs(contribution)));
      if(abs(expected-contribution)>2e-5*scale){report(NONPOSITIVE);return;}}
    atomicAddFloat(at(DIAGONAL,l,own),contribution);}}
@compute @workgroup_size(64) fn copyRhs(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<rows()&&!stopped()){
  let value=inputRhs[row];if(!finite(value)){report(NONFINITE);}else{storeFloat(RHS,0u,row,value);}}}
fn applyRow(row:u32,channel:u32){let l=level();let own=map(row,l);let x=solution(channel,l,own);var value=boundaryAnchor(row)*x;let h=headers[row];
  for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=rows()){report(NONFINITE);continue;}let coefficient=firstOrderCoefficient(row,e);
    let neighbor=map(e.row,l);if(coefficient>0.0&&neighbor!=own){value+=coefficient*(x-solution(channel,l,neighbor));}}
  atomicAddFloat(at(WORK,l,own),value);}
@compute @workgroup_size(64) fn applyA(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<rows()&&!stopped()){applyRow(row,SOLUTION_A);}}
@compute @workgroup_size(64) fn applyB(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<rows()&&!stopped()){applyRow(row,SOLUTION_B);}}
fn jacobi(slot:u32,source:u32,destination:u32){let l=level();if(!activeSlot(slot,l)||stopped()){return;}let diagonal=loadFloat(DIAGONAL,l,slot);
  if(!finite(diagonal)||diagonal<=params.weights.y){report(NONPOSITIVE);return;}let old=solution(source,l,slot);
  let next=old+params.weights.x*(loadFloat(RHS,l,slot)-loadFloat(WORK,l,slot))/diagonal;
  if(!finite(next)){report(NONFINITE);}else{storeFloat(destination,l,slot,next);}}
@compute @workgroup_size(64) fn jacobiAtoB(@builtin(global_invocation_id) gid:vec3u){jacobi(slotIndex(gid),SOLUTION_A,SOLUTION_B);}
@compute @workgroup_size(64) fn jacobiBtoA(@builtin(global_invocation_id) gid:vec3u){jacobi(slotIndex(gid),SOLUTION_B,SOLUTION_A);}
@compute @workgroup_size(64) fn formResidual(@builtin(global_invocation_id) gid:vec3u){let slot=slotIndex(gid);let l=level();if(activeSlot(slot,l)&&!stopped()){
  storeFloat(WORK,l,slot,loadFloat(RHS,l,slot)-loadFloat(WORK,l,slot));}}
// GhostValueAccumulate = P^T. Dividing by the child membership count makes
// each child residual contribute exactly once although this kernel visits rows.
@compute @workgroup_size(64) fn ghostAccumulate(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);let l=level();if(row>=rows()||l+1u>=levels()||stopped()){return;}
  let child=map(row,l);let parent=map(row,l+1u);let members=atomicLoad(&state[at(COUNTS,l,child)]);
  if(members==0u){report(NONPOSITIVE);return;}atomicAddFloat(at(RHS,l+1u,parent),loadFloat(WORK,l,child)/f32(members));}
// GhostValuePropagate = P, implemented as an identical-value member average
// so concurrent rows use defined atomic accumulation into each child slot.
@compute @workgroup_size(64) fn ghostPropagate(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);let l=level();if(row>=rows()||l+1u>=levels()||stopped()){return;}
  let child=map(row,l);let parent=map(row,l+1u);let members=atomicLoad(&state[at(COUNTS,l,child)]);
  if(members==0u){report(NONPOSITIVE);return;}atomicAddFloat(at(SOLUTION_A,l,child),loadFloat(SOLUTION_A,l+1u,parent)/f32(members));}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<rows()&&!stopped()){
  let value=loadFloat(SOLUTION_A,0u,row);if(!finite(value)){report(NONFINITE);}else{outputCorrection[row]=value;}}}
`;
