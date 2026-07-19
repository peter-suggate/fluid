/** Isolated two-level address space for scene-scale 8-cubed fluid bricks. */

export const SCENE_FLUID_BRICK_SIZE = 8 as const;
export const SCENE_FLUID_BLOCK_BRICKS = 8 as const;
export const SCENE_FLUID_BLOCK_ENTRIES = SCENE_FLUID_BLOCK_BRICKS ** 3;
export const SCENE_FLUID_MISSING_AIR_PHI = 1_000_000;
const CONTROL_WORDS = 16;
const ROOT_WORDS = 4;
const TOMBSTONE = 0xffff_ffff;

export type SignedCoordinate = readonly [number, number, number];

export interface SceneFluidBrickAddress {
  brick: readonly [number, number, number];
  block: readonly [number, number, number];
  localBrick: readonly [number, number, number];
  localBrickIndex: number;
}

export interface SceneFluidCellAddress extends SceneFluidBrickAddress {
  localCell: readonly [number, number, number];
  residentSlot?: number;
  missing: boolean;
  airPhi: number;
}

export interface SceneFluidAddressPlan {
  maximumBlocks: number;
  maximumResidentBricks: number;
  rootCapacity: number;
  allocatedWords: number;
  allocatedBytes: number;
  rootOffsetWords: number;
  blockFreeOffsetWords: number;
  brickFreeOffsetWords: number;
  blockCountOffsetWords: number;
  blockPageOffsetWords: number;
}

export interface SceneFluidLifecycleStats {
  residentBlocks: number;
  residentBricks: number;
  peakBlocks: number;
  peakBricks: number;
  requiredBlocks: number;
  requiredBricks: number;
  blockOverflow: number;
  brickOverflow: number;
  activated: number;
  retired: number;
  generation: number;
  maximumBlocks: number;
  maximumResidentBricks: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function signedCoordinate(value: SignedCoordinate, label: string): readonly [number, number, number] {
  value.forEach((component, axis) => {
    if (!Number.isSafeInteger(component)) throw new RangeError(`${label} axis ${axis} must be a safe integer`);
  });
  return [...value] as [number, number, number];
}

function floorDiv(value: number, divisor: number): number { return Math.floor(value / divisor); }
function floorMod(value: number, divisor: number): number { return value - floorDiv(value, divisor) * divisor; }
function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

export function planSceneScaleFluidBricks(maximumBlocks: number, maximumResidentBricks: number): SceneFluidAddressPlan {
  positiveInteger(maximumBlocks, "Scene fluid block capacity");
  positiveInteger(maximumResidentBricks, "Scene fluid brick capacity");
  if (maximumBlocks > 0x7fff_fffe || maximumResidentBricks > 0x7fff_fffe) throw new RangeError("Scene fluid capacity exceeds 32-bit addressing");
  const rootCapacity = nextPowerOfTwo(maximumBlocks * 2);
  const rootOffsetWords = CONTROL_WORDS;
  const blockFreeOffsetWords = rootOffsetWords + rootCapacity * ROOT_WORDS;
  const brickFreeOffsetWords = blockFreeOffsetWords + maximumBlocks;
  const blockCountOffsetWords = brickFreeOffsetWords + maximumResidentBricks;
  const blockPageOffsetWords = blockCountOffsetWords + maximumBlocks;
  const allocatedWords = blockPageOffsetWords + maximumBlocks * SCENE_FLUID_BLOCK_ENTRIES;
  return {
    maximumBlocks, maximumResidentBricks, rootCapacity, allocatedWords,
    allocatedBytes: allocatedWords * 4,
    rootOffsetWords, blockFreeOffsetWords, brickFreeOffsetWords, blockCountOffsetWords, blockPageOffsetWords,
  };
}

export function addressSceneFluidBrick(brickValue: SignedCoordinate): SceneFluidBrickAddress {
  const brick = signedCoordinate(brickValue, "Scene fluid brick");
  const block = brick.map((value) => floorDiv(value, SCENE_FLUID_BLOCK_BRICKS)) as [number, number, number];
  const localBrick = brick.map((value) => floorMod(value, SCENE_FLUID_BLOCK_BRICKS)) as [number, number, number];
  return {
    brick, block, localBrick,
    localBrickIndex: localBrick[0] + 8 * (localBrick[1] + 8 * localBrick[2]),
  };
}

export function splitSceneFluidCell(cellValue: SignedCoordinate): { brick: readonly [number, number, number]; localCell: readonly [number, number, number] } {
  const cell = signedCoordinate(cellValue, "Scene fluid cell");
  return {
    brick: cell.map((value) => floorDiv(value, SCENE_FLUID_BRICK_SIZE)) as [number, number, number],
    localCell: cell.map((value) => floorMod(value, SCENE_FLUID_BRICK_SIZE)) as [number, number, number],
  };
}

function key(value: SignedCoordinate): string { return `${value[0]},${value[1]},${value[2]}`; }

interface CPUBlock { slot: number; entries: Uint32Array<ArrayBuffer>; resident: number }

/** Deterministic CPU oracle for the bounded root -> block page -> brick-slot lifecycle. */
export class SceneScaleFluidBrickLifecycle {
  readonly plan: SceneFluidAddressPlan;
  private readonly blocks = new Map<string, CPUBlock>();
  private readonly freeBlocks: number[];
  private readonly freeBricks: number[];
  private statsValue: SceneFluidLifecycleStats;

  constructor(maximumBlocks: number, maximumResidentBricks: number) {
    this.plan = planSceneScaleFluidBricks(maximumBlocks, maximumResidentBricks);
    this.freeBlocks = Array.from({ length: maximumBlocks }, (_, index) => maximumBlocks - 1 - index);
    this.freeBricks = Array.from({ length: maximumResidentBricks }, (_, index) => maximumResidentBricks - 1 - index);
    this.statsValue = this.emptyStats();
  }

  private emptyStats(): SceneFluidLifecycleStats {
    return {
      residentBlocks: this.blocks.size, residentBricks: this.plan.maximumResidentBricks - this.freeBricks.length,
      peakBlocks: this.statsValue?.peakBlocks ?? 0, peakBricks: this.statsValue?.peakBricks ?? 0,
      requiredBlocks: 0, requiredBricks: 0, blockOverflow: 0, brickOverflow: 0,
      activated: 0, retired: 0, generation: this.statsValue?.generation ?? 0,
      maximumBlocks: this.plan.maximumBlocks, maximumResidentBricks: this.plan.maximumResidentBricks,
    };
  }

  private unique(values: readonly SignedCoordinate[], label: string): SignedCoordinate[] {
    const result = new Map<string, SignedCoordinate>();
    for (const value of values) { const checked = signedCoordinate(value, label); result.set(key(checked), checked); }
    return [...result.values()];
  }

  publish(activeValues: readonly SignedCoordinate[], retiredValues: readonly SignedCoordinate[]): SceneFluidLifecycleStats {
    const active = this.unique(activeValues, "Active scene fluid brick");
    const retired = this.unique(retiredValues, "Retired scene fluid brick");
    const activeKeys = new Set(active.map(key));
    if (retired.some((value) => activeKeys.has(key(value)))) throw new RangeError("A scene fluid brick cannot be active and retired in one publication");
    const stats = this.emptyStats(); stats.generation += 1;
    for (const brick of active) {
      const address = addressSceneFluidBrick(brick); const blockKey = key(address.block);
      let block = this.blocks.get(blockKey);
      if (!block) {
        stats.requiredBlocks += 1;
        if (this.freeBlocks.length === 0) { stats.blockOverflow += 1; continue; }
      }
      if (block?.entries[address.localBrickIndex] !== undefined && block.entries[address.localBrickIndex] !== 0) continue;
      stats.requiredBricks += 1;
      if (this.freeBricks.length === 0) { stats.brickOverflow += 1; continue; }
      if (!block) {
        const blockSlot = this.freeBlocks.pop()!;
        block = { slot: blockSlot, entries: new Uint32Array(SCENE_FLUID_BLOCK_ENTRIES), resident: 0 };
        this.blocks.set(blockKey, block);
      }
      const slot = this.freeBricks.pop()!;
      block.entries[address.localBrickIndex] = slot + 1; block.resident += 1; stats.activated += 1;
    }
    for (const brick of retired) {
      const address = addressSceneFluidBrick(brick); const blockKey = key(address.block); const block = this.blocks.get(blockKey);
      if (!block) continue;
      const encoded = block.entries[address.localBrickIndex]; if (encoded === 0) continue;
      block.entries[address.localBrickIndex] = 0; block.resident -= 1; this.freeBricks.push(encoded - 1); stats.retired += 1;
      if (block.resident === 0) { this.blocks.delete(blockKey); this.freeBlocks.push(block.slot); }
    }
    stats.residentBlocks = this.blocks.size;
    stats.residentBricks = this.plan.maximumResidentBricks - this.freeBricks.length;
    stats.peakBlocks = Math.max(stats.peakBlocks, stats.residentBlocks);
    stats.peakBricks = Math.max(stats.peakBricks, stats.residentBricks);
    this.statsValue = stats;
    return { ...stats };
  }

  slot(brickValue: SignedCoordinate): number | undefined {
    const address = addressSceneFluidBrick(brickValue);
    const encoded = this.blocks.get(key(address.block))?.entries[address.localBrickIndex] ?? 0;
    return encoded === 0 ? undefined : encoded - 1;
  }

  resolveCell(cellValue: SignedCoordinate): SceneFluidCellAddress {
    const split = splitSceneFluidCell(cellValue); const address = addressSceneFluidBrick(split.brick); const residentSlot = this.slot(split.brick);
    return { ...address, localCell: split.localCell, residentSlot, missing: residentSlot === undefined, airPhi: SCENE_FLUID_MISSING_AIR_PHI };
  }

  stats(): SceneFluidLifecycleStats { return { ...this.statsValue }; }
}

/** GPU mirror uses one invocation so allocation order exactly matches the CPU oracle. */
export const sceneScaleFluidBrickLifecycleShader = /* wgsl */ `
struct Params { counts: vec4u, offsets: vec4u, capacities: vec4u }
@group(0) @binding(0) var<storage, read_write> arena: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> changes: array<vec4i>;
@group(0) @binding(2) var<uniform> params: Params;
const EMPTY: u32 = 0u; const TOMBSTONE: u32 = 0xffffffffu; const BLOCK_ENTRIES: u32 = 512u;
fn hash(p: vec3i) -> u32 {
  var h = bitcast<u32>(p.x) * 0x9e3779b1u;
  h = (h ^ bitcast<u32>(p.y)) * 0x85ebca6bu;
  return (h ^ (bitcast<u32>(p.z) * 0xc2b2ae35u)) & (params.capacities.z - 1u);
}
fn floorDiv8(v: i32) -> i32 {
  let quotient = v / 8; let remainder = v % 8;
  return quotient - select(0i, 1i, remainder < 0);
}
fn mod8(v: i32, q: i32) -> u32 { return u32(v - q * 8); }
fn findRoot(p: vec3i) -> u32 {
  let start = hash(p);
  for (var probe = 0u; probe < params.capacities.z; probe += 1u) {
    let root = params.offsets.x + ((start + probe) & (params.capacities.z - 1u)) * 4u;
    let encoded = atomicLoad(&arena[root + 3u]);
    if (encoded == EMPTY) { return TOMBSTONE; }
    if (encoded != TOMBSTONE && bitcast<i32>(atomicLoad(&arena[root])) == p.x && bitcast<i32>(atomicLoad(&arena[root + 1u])) == p.y && bitcast<i32>(atomicLoad(&arena[root + 2u])) == p.z) { return root; }
  }
  return TOMBSTONE;
}
fn pop(control: u32, base: u32) -> u32 {
  let available = atomicLoad(&arena[control]); if (available == 0u) { return TOMBSTONE; }
  atomicStore(&arena[control], available - 1u); return atomicLoad(&arena[base + available - 1u]);
}
fn push(control: u32, base: u32, capacity: u32, slot: u32) {
  let count = atomicLoad(&arena[control]); if (count < capacity) { atomicStore(&arena[base + count], slot); atomicStore(&arena[control], count + 1u); }
}
fn ensureRoot(p: vec3i) -> u32 {
  let found = findRoot(p); if (found != TOMBSTONE) { return found; }
  let block = pop(0u, params.offsets.y); if (block == TOMBSTONE) { atomicAdd(&arena[8], 1u); return TOMBSTONE; }
  let start = hash(p); var reusable = TOMBSTONE;
  for (var probe = 0u; probe < params.capacities.z; probe += 1u) {
    let root = params.offsets.x + ((start + probe) & (params.capacities.z - 1u)) * 4u;
    let encoded = atomicLoad(&arena[root + 3u]);
    if (encoded == TOMBSTONE && reusable == TOMBSTONE) { reusable = root; }
    if (encoded == EMPTY) { if (reusable == TOMBSTONE) { reusable = root; } break; }
  }
  if (reusable == TOMBSTONE) { push(0u, params.offsets.y, params.capacities.x, block); atomicAdd(&arena[8], 1u); return TOMBSTONE; }
  atomicStore(&arena[reusable], bitcast<u32>(p.x)); atomicStore(&arena[reusable + 1u], bitcast<u32>(p.y)); atomicStore(&arena[reusable + 2u], bitcast<u32>(p.z));
  atomicStore(&arena[reusable + 3u], block + 1u); atomicStore(&arena[params.offsets.w + block], 0u);
  for (var local = 0u; local < BLOCK_ENTRIES; local += 1u) { atomicStore(&arena[params.capacities.w + block * BLOCK_ENTRIES + local], 0u); }
  let resident = atomicAdd(&arena[2], 1u) + 1u; atomicMax(&arena[4], resident); atomicAdd(&arena[6], 1u); return reusable;
}
fn address(brick: vec3i) -> vec4u {
  let block = vec3i(floorDiv8(brick.x), floorDiv8(brick.y), floorDiv8(brick.z));
  return vec4u(bitcast<u32>(block.x), bitcast<u32>(block.y), bitcast<u32>(block.z), mod8(brick.x, block.x) + 8u * (mod8(brick.y, block.y) + 8u * mod8(brick.z, block.z)));
}
fn activate(brick: vec3i) {
  let a = address(brick); let root = ensureRoot(bitcast<vec3i>(a.xyz)); if (root == TOMBSTONE) { return; }
  let block = atomicLoad(&arena[root + 3u]) - 1u; let entry = params.capacities.w + block * BLOCK_ENTRIES + a.w;
  if (atomicLoad(&arena[entry]) != 0u) { return; }
  atomicAdd(&arena[7], 1u); let slot = pop(1u, params.offsets.z);
  if (slot == TOMBSTONE) { atomicAdd(&arena[9], 1u); return; }
  atomicStore(&arena[entry], slot + 1u); atomicAdd(&arena[params.offsets.w + block], 1u);
  let resident = atomicAdd(&arena[3], 1u) + 1u; atomicMax(&arena[5], resident); atomicAdd(&arena[10], 1u);
}
fn retire(brick: vec3i) {
  let a = address(brick); let root = findRoot(bitcast<vec3i>(a.xyz)); if (root == TOMBSTONE) { return; }
  let block = atomicLoad(&arena[root + 3u]) - 1u; let entry = params.capacities.w + block * BLOCK_ENTRIES + a.w;
  let encoded = atomicLoad(&arena[entry]); if (encoded == 0u) { return; }
  atomicStore(&arena[entry], 0u); push(1u, params.offsets.z, params.capacities.y, encoded - 1u); atomicSub(&arena[3], 1u); atomicAdd(&arena[11], 1u);
  let remaining = atomicSub(&arena[params.offsets.w + block], 1u) - 1u;
  if (remaining == 0u) { atomicStore(&arena[root + 3u], TOMBSTONE); push(0u, params.offsets.y, params.capacities.x, block); atomicSub(&arena[2], 1u); }
}
@compute @workgroup_size(1)
fn publish() {
  atomicStore(&arena[6], 0u); atomicStore(&arena[7], 0u); atomicStore(&arena[8], 0u); atomicStore(&arena[9], 0u); atomicStore(&arena[10], 0u); atomicStore(&arena[11], 0u); atomicAdd(&arena[12], 1u);
  for (var i = 0u; i < params.counts.x; i += 1u) { activate(changes[i].xyz); }
  for (var i = 0u; i < params.counts.y; i += 1u) { retire(changes[params.counts.z + i].xyz); }
}
`;

export class GPUSceneScaleFluidBrickArena {
  readonly plan: SceneFluidAddressPlan;
  readonly arena: GPUBuffer;
  private readonly changes: GPUBuffer; private readonly params: GPUBuffer; private readonly pipeline: GPUComputePipeline; private readonly group: GPUBindGroup;
  private readonly changeData: Int32Array<ArrayBuffer>; private destroyed = false;
  constructor(private readonly device: GPUDevice, maximumBlocks: number, maximumResidentBricks: number) {
    this.plan = planSceneScaleFluidBricks(maximumBlocks, maximumResidentBricks);
    this.arena = device.createBuffer({ size: this.plan.allocatedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    this.changes = device.createBuffer({ size: maximumResidentBricks * 2 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const initial = new Uint32Array(this.plan.allocatedWords); initial[0] = maximumBlocks; initial[1] = maximumResidentBricks;
    for (let i = 0; i < maximumBlocks; i += 1) initial[this.plan.blockFreeOffsetWords + i] = maximumBlocks - 1 - i;
    for (let i = 0; i < maximumResidentBricks; i += 1) initial[this.plan.brickFreeOffsetWords + i] = maximumResidentBricks - 1 - i;
    new Uint32Array(this.arena.getMappedRange()).set(initial);
    this.arena.unmap();
    this.changeData = new Int32Array(maximumResidentBricks * 2 * 4);
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const module = device.createShaderModule({ code: sceneScaleFluidBrickLifecycleShader });
    this.pipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "publish" } });
    this.group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: this.arena } }, { binding: 1, resource: { buffer: this.changes } }, { binding: 2, resource: { buffer: this.params } }] });
  }
  encodeLifecycle(encoder: GPUCommandEncoder, active: readonly SignedCoordinate[], retired: readonly SignedCoordinate[]): void {
    if (this.destroyed) return;
    if (active.length > this.plan.maximumResidentBricks || retired.length > this.plan.maximumResidentBricks) throw new RangeError("Scene fluid lifecycle change list exceeds capacity");
    this.changeData.fill(0);
    active.forEach((value, i) => this.changeData.set(signedCoordinate(value, "Active scene fluid brick"), i * 4));
    retired.forEach((value, i) => this.changeData.set(signedCoordinate(value, "Retired scene fluid brick"), (this.plan.maximumResidentBricks + i) * 4));
    this.device.queue.writeBuffer(this.changes, 0, this.changeData);
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      active.length, retired.length, this.plan.maximumResidentBricks, 0,
      this.plan.rootOffsetWords, this.plan.blockFreeOffsetWords, this.plan.brickFreeOffsetWords, this.plan.blockCountOffsetWords,
      this.plan.maximumBlocks, this.plan.maximumResidentBricks, this.plan.rootCapacity, this.plan.blockPageOffsetWords,
    ]));
    const pass = encoder.beginComputePass(); pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.group); pass.dispatchWorkgroups(1); pass.end();
  }
  async readStats(): Promise<SceneFluidLifecycleStats> {
    const readback = this.device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.arena, 0, readback, 0, 52); this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const d = new Uint32Array(readback.getMappedRange()); const result: SceneFluidLifecycleStats = {
      residentBlocks: d[2], residentBricks: d[3], peakBlocks: d[4], peakBricks: d[5], requiredBlocks: d[6], requiredBricks: d[7], blockOverflow: d[8], brickOverflow: d[9], activated: d[10], retired: d[11], generation: d[12], maximumBlocks: this.plan.maximumBlocks, maximumResidentBricks: this.plan.maximumResidentBricks,
    }; readback.unmap(); readback.destroy(); return result;
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.arena.destroy(); this.changes.destroy(); this.params.destroy(); }
}
