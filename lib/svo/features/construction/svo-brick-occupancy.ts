/**
 * Lossless empty-space metadata for one render-facing 8x8x8 SVO brick.
 *
 * The packed word is stored in the otherwise-unused `links.w` word of the
 * brick's terminal canonical node. No extra resident buffer or render binding
 * is required. Bits 0..27 contain occupancy. Bits 28..31 contain the live
 * brick lifecycle and are intentionally orthogonal: rebuilding occupancy must
 * not change whether a leaf is active, dirty, queued, or being relocated.
 */

export const SVO_BRICK_OCCUPANCY = Object.freeze({
  brickSize: 8,
  macroSize: 4,
  macroCount: 8,
  macroMask: 0xff,
  minShifts: [8, 11, 14] as const,
  maxShifts: [17, 20, 23] as const,
  coordinateMask: 0x7,
  occupiedBit: 1 << 26,
  readyBit: 1 << 27,
  metadataMask: 0x0fffffff,
  /** One existing node word is reused, so the incremental allocation is zero. */
  incrementalStorageBytesPerBrick: 0,
  metadataBytesPerBrick: Uint32Array.BYTES_PER_ELEMENT,
});

export const SVO_BRICK_LIFECYCLE = Object.freeze({
  activeBit: 0x10000000,
  dirtyBit: 0x20000000,
  queuedBit: 0x40000000,
  relocatingBit: 0x80000000,
  mask: 0xf0000000,
});

export type SvoBrickLocalCoordinate = readonly [number, number, number];

export interface SvoBrickOccupancySummary {
  ready: boolean;
  occupied: boolean;
  /** xyz octant order: x | y << 1 | z << 2; each bit covers one 4^3 macrocell. */
  macroMask: number;
  minInclusive: SvoBrickLocalCoordinate;
  maxInclusive: SvoBrickLocalCoordinate;
}

export interface BuiltSvoBrickOccupancySummary extends SvoBrickOccupancySummary {
  packed: number;
  occupiedCellCount: number;
}

export interface SvoBrickLifecycle {
  active: boolean;
  dirty: boolean;
  queued: boolean;
  relocating: boolean;
}

/** Encode only the lifecycle bits in a terminal-node flags word. */
export function encodeSvoBrickLifecycle(lifecycle: SvoBrickLifecycle): number {
  let packed = 0;
  if (lifecycle.active) packed |= SVO_BRICK_LIFECYCLE.activeBit;
  if (lifecycle.dirty) packed |= SVO_BRICK_LIFECYCLE.dirtyBit;
  if (lifecycle.queued) packed |= SVO_BRICK_LIFECYCLE.queuedBit;
  if (lifecycle.relocating) packed |= SVO_BRICK_LIFECYCLE.relocatingBit;
  return packed >>> 0;
}

/** Decode lifecycle bits while ignoring the independently rebuilt occupancy summary. */
export function decodeSvoBrickLifecycle(packedValue: number): SvoBrickLifecycle {
  const packed = packedValue >>> 0;
  return {
    active: (packed & SVO_BRICK_LIFECYCLE.activeBit) !== 0,
    dirty: (packed & SVO_BRICK_LIFECYCLE.dirtyBit) !== 0,
    queued: (packed & SVO_BRICK_LIFECYCLE.queuedBit) !== 0,
    relocating: (packed & SVO_BRICK_LIFECYCLE.relocatingBit) !== 0,
  };
}

/** Replace lifecycle state without disturbing current occupancy metadata. */
export function replaceSvoBrickLifecycle(packedValue: number, lifecycle: SvoBrickLifecycle): number {
  return (((packedValue >>> 0) & SVO_BRICK_OCCUPANCY.metadataMask) | encodeSvoBrickLifecycle(lifecycle)) >>> 0;
}

/** Replace occupancy metadata without disturbing the live leaf lifecycle. */
export function replaceSvoBrickOccupancy(packedValue: number, summary: SvoBrickOccupancySummary): number {
  return (((packedValue >>> 0) & SVO_BRICK_LIFECYCLE.mask) | encodeSvoBrickOccupancy(summary)) >>> 0;
}

/** Only current active leaves may be consumed by traversal or derived-data passes. */
export function isSvoBrickLifecycleCurrent(lifecycle: SvoBrickLifecycle): boolean {
  return lifecycle.active && !lifecycle.dirty && !lifecycle.relocating;
}

function localCoordinate(value: SvoBrickLocalCoordinate, name: string): void {
  for (const [axis, component] of value.entries()) {
    if (!Number.isSafeInteger(component) || component < 0 || component >= SVO_BRICK_OCCUPANCY.brickSize) {
      throw new RangeError(`${name}[${axis}] must be an integer in an 8-cell brick`);
    }
  }
}

/** Encode a summary into the terminal node's existing flags word. */
export function encodeSvoBrickOccupancy(summary: SvoBrickOccupancySummary): number {
  if (!summary.ready) {
    if (summary.occupied || summary.macroMask !== 0) {
      throw new RangeError("Unavailable brick occupancy cannot claim occupied cells");
    }
    return 0;
  }
  if (!Number.isSafeInteger(summary.macroMask) || summary.macroMask < 0 || summary.macroMask > 0xff) {
    throw new RangeError("Brick macrocell mask must fit eight bits");
  }
  if (summary.occupied !== (summary.macroMask !== 0)) {
    throw new RangeError("Occupied brick metadata must have at least one occupied macrocell, and vice versa");
  }
  localCoordinate(summary.minInclusive, "minInclusive");
  localCoordinate(summary.maxInclusive, "maxInclusive");
  if (summary.occupied && summary.minInclusive.some((value, axis) => value > summary.maxInclusive[axis])) {
    throw new RangeError("Occupied brick bounds must be ordered");
  }
  let packed = SVO_BRICK_OCCUPANCY.readyBit | summary.macroMask;
  if (summary.occupied) packed |= SVO_BRICK_OCCUPANCY.occupiedBit;
  for (let axis = 0; axis < 3; axis += 1) {
    packed |= summary.minInclusive[axis] << SVO_BRICK_OCCUPANCY.minShifts[axis];
    packed |= summary.maxInclusive[axis] << SVO_BRICK_OCCUPANCY.maxShifts[axis];
  }
  return packed >>> 0;
}

/** Decode a terminal node flags word. Unknown/reserved bits are ignored. */
export function decodeSvoBrickOccupancy(packedValue: number): SvoBrickOccupancySummary {
  const packed = (packedValue >>> 0) & SVO_BRICK_OCCUPANCY.metadataMask;
  const coordinate = (shifts: readonly [number, number, number]) => shifts.map(
    (shift) => (packed >>> shift) & SVO_BRICK_OCCUPANCY.coordinateMask,
  ) as unknown as [number, number, number];
  return {
    ready: (packed & SVO_BRICK_OCCUPANCY.readyBit) !== 0,
    occupied: (packed & SVO_BRICK_OCCUPANCY.occupiedBit) !== 0,
    macroMask: packed & SVO_BRICK_OCCUPANCY.macroMask,
    minInclusive: coordinate(SVO_BRICK_OCCUPANCY.minShifts),
    maxInclusive: coordinate(SVO_BRICK_OCCUPANCY.maxShifts),
  };
}

/**
 * Build exact conservative metadata from the authoritative packed material /
 * owner payload. Every non-zero material is retained; invalid owners can only
 * add false-positive work, never hide a possible exact hit.
 */
export function buildSvoBrickOccupancy(
  materialOwners: ArrayLike<number>,
  voxelOffset = 0,
  previousTerminalFlags = 0,
): BuiltSvoBrickOccupancySummary {
  if (!Number.isSafeInteger(voxelOffset) || voxelOffset < 0) throw new RangeError("Voxel offset must be a non-negative integer");
  const cellCount = SVO_BRICK_OCCUPANCY.brickSize ** 3;
  if (voxelOffset + cellCount > materialOwners.length) throw new RangeError("Material-owner payload does not contain a complete 8^3 brick");
  let macroMask = 0;
  let occupiedCellCount = 0;
  const minimum = [7, 7, 7];
  const maximum = [0, 0, 0];
  for (let localIndex = 0; localIndex < cellCount; localIndex += 1) {
    if (((materialOwners[voxelOffset + localIndex] ?? 0) & 0xffff) === 0) continue;
    const x = localIndex & 7;
    const y = (localIndex >>> 3) & 7;
    const z = localIndex >>> 6;
    minimum[0] = Math.min(minimum[0], x); minimum[1] = Math.min(minimum[1], y); minimum[2] = Math.min(minimum[2], z);
    maximum[0] = Math.max(maximum[0], x); maximum[1] = Math.max(maximum[1], y); maximum[2] = Math.max(maximum[2], z);
    macroMask |= 1 << ((x >>> 2) | ((y >>> 2) << 1) | ((z >>> 2) << 2));
    occupiedCellCount += 1;
  }
  const occupied = occupiedCellCount !== 0;
  const summary: SvoBrickOccupancySummary = {
    ready: true,
    occupied,
    macroMask,
    minInclusive: occupied ? minimum as [number, number, number] : [0, 0, 0],
    maxInclusive: occupied ? maximum as [number, number, number] : [0, 0, 0],
  };
  return { ...summary, packed: replaceSvoBrickOccupancy(previousTerminalFlags, summary), occupiedCellCount };
}

/** Conservative count of cells retained by the 4^3 macrocell mask. */
export function svoBrickOccupancyRetainedCellCount(summary: Pick<SvoBrickOccupancySummary, "ready" | "macroMask">): number {
  if (!summary.ready) return SVO_BRICK_OCCUPANCY.brickSize ** 3;
  let occupiedMacrocells = 0;
  for (let bits = summary.macroMask & 0xff; bits !== 0; bits >>>= 1) occupiedMacrocells += bits & 1;
  return occupiedMacrocells * SVO_BRICK_OCCUPANCY.macroSize ** 3;
}

/**
 * WGSL helpers intentionally depend only on a packed word supplied by the
 * caller. The dry renderer can therefore compose them after its canonical
 * SvoNode declaration without adding bindings or changing traversal ABI.
 */
export const svoBrickOccupancyWGSL = /* wgsl */ `
const SVO_BRICK_OCCUPANCY_READY:u32=${SVO_BRICK_OCCUPANCY.readyBit}u;
const SVO_BRICK_OCCUPANCY_OCCUPIED:u32=${SVO_BRICK_OCCUPANCY.occupiedBit}u;
const SVO_BRICK_LIFECYCLE_ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const SVO_BRICK_LIFECYCLE_DIRTY:u32=${SVO_BRICK_LIFECYCLE.dirtyBit}u;
const SVO_BRICK_LIFECYCLE_QUEUED:u32=${SVO_BRICK_LIFECYCLE.queuedBit}u;
const SVO_BRICK_LIFECYCLE_RELOCATING:u32=${SVO_BRICK_LIFECYCLE.relocatingBit}u;
struct SvoBrickOccupancy{ready:u32,occupied:u32,macroMask:u32,minInclusive:vec3u,maxInclusive:vec3u}
struct SvoBrickLifecycle{activeFlag:u32,dirtyFlag:u32,queuedFlag:u32,relocatingFlag:u32}
fn svoBrickOccupancyDecode(packed:u32)->SvoBrickOccupancy{
  return SvoBrickOccupancy(
    select(0u,1u,(packed&SVO_BRICK_OCCUPANCY_READY)!=0u),
    select(0u,1u,(packed&SVO_BRICK_OCCUPANCY_OCCUPIED)!=0u),
    packed&0xffu,
    vec3u((packed>>8u)&7u,(packed>>11u)&7u,(packed>>14u)&7u),
    vec3u((packed>>17u)&7u,(packed>>20u)&7u,(packed>>23u)&7u));
}
fn svoBrickLifecycleDecode(packed:u32)->SvoBrickLifecycle{
  return SvoBrickLifecycle(
    select(0u,1u,(packed&SVO_BRICK_LIFECYCLE_ACTIVE)!=0u),
    select(0u,1u,(packed&SVO_BRICK_LIFECYCLE_DIRTY)!=0u),
    select(0u,1u,(packed&SVO_BRICK_LIFECYCLE_QUEUED)!=0u),
    select(0u,1u,(packed&SVO_BRICK_LIFECYCLE_RELOCATING)!=0u));
}
fn svoBrickLifecycleCurrent(state:SvoBrickLifecycle)->bool{
  return state.activeFlag!=0u&&state.dirtyFlag==0u&&state.relocatingFlag==0u;
}
fn svoBrickMacroOccupied(summary:SvoBrickOccupancy,local:vec3u)->bool{
  if(summary.ready==0u){return true;}
  let macroCoord=local>>vec3u(2u);
  let bit=macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u);
  return (summary.macroMask&(1u<<bit))!=0u;
}
fn svoBrickOccupiedBounds(summary:SvoBrickOccupancy,brickMinimum:vec3f,cellSize:vec3f)->mat2x3f{
  let minimum=brickMinimum+vec3f(summary.minInclusive)*cellSize;
  let maximum=brickMinimum+vec3f(summary.maxInclusive+vec3u(1u))*cellSize;
  return mat2x3f(minimum,maximum);
}
`;
