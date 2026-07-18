/**
 * Brick-pooled 3D atlas storage for the bulk fluid fields (phi + velocity).
 *
 * Each resident 8^3 fluid brick owns one (brickSize+2)^3 atlas tile: the
 * payload plus a one-voxel apron on every side so hardware trilinear
 * filtering never reads across tile boundaries. Slot lifecycle follows the
 * sparse surface band pattern: a u32 page table per logical brick with an
 * INVALID sentinel, a free-list stack, control-word counters with an
 * overflow flag, and an indirect active-tile worklist. In mirror mode the
 * dense textures stay authoritative; the atlas is populated from them each
 * frame and continuously validated against them, so later kernels can flip
 * to brick-resident sampling with a proven storage substrate.
 */

import type { GPUFluidBrickResidency } from "./webgpu-fluid-brick-residency";

export const BRICK_ATLAS_INVALID_SLOT = 0xffff_ffff;
export const BRICK_ATLAS_ACTIVE_DISPATCH_OFFSET_BYTES = 4;

export type FluidBrickAtlasMode = "mirror";

export interface FluidBrickAtlasOptions {
  brickSize?: 4 | 8;
  /** Fraction of the logical brick lattice backed by physical tiles. */
  maximumResidentFraction?: number;
  /** Optional hard physical-tile ceiling. */
  maximumTiles?: number;
  /** Compare atlas sampling against the dense fields every frame. */
  validate?: boolean;
  mode?: FluidBrickAtlasMode;
}

export interface FluidBrickAtlasPlan {
  dimensions: readonly [number, number, number];
  brickDimensions: readonly [number, number, number];
  logicalBrickCount: number;
  brickSize: 4 | 8;
  /** Payload plus one apron voxel on each side. */
  tileSize: number;
  /** Physical tile slots actually backed by atlas texels. */
  capacity: number;
  /** Tiles per atlas axis. */
  tileGridDimensions: readonly [number, number, number];
  /** Atlas texture extent in texels. */
  atlasDimensions: readonly [number, number, number];
  /** True when device texture limits forced the capacity below the request. */
  degraded: boolean;
  bytesPerTile: number;
  allocatedTextureBytes: number;
}

function positiveInteger(value: number, label: string) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

/** Pure atlas geometry planning, clamped to the device's 3D texture limit. */
export function planFluidBrickAtlas(
  dimensions: readonly [number, number, number],
  options: FluidBrickAtlasOptions & { maxTextureDimension3D?: number } = {},
): FluidBrickAtlasPlan {
  dimensions.forEach((value, axis) => positiveInteger(value, `Brick atlas dimension ${axis}`));
  const brickSize = options.brickSize ?? 8;
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Brick atlas brick size must be 4 or 8");
  const tileSize = brickSize + 2;
  const brickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const logicalBrickCount = brickDimensions[0] * brickDimensions[1] * brickDimensions[2];
  // In-box mirror mode backs the whole logical lattice by default: a small
  // box's residency band can cover most bricks, and the payoff of a lower
  // fraction only arrives with scene-scale addressing. Overflow degrades to
  // dense fallback sampling rather than failing.
  const fraction = Number.isFinite(options.maximumResidentFraction)
    ? Math.max(1 / logicalBrickCount, Math.min(1, options.maximumResidentFraction!))
    : 1;
  const hardCapacity = options.maximumTiles === undefined
    ? logicalBrickCount
    : Math.max(1, Math.min(logicalBrickCount, Math.floor(options.maximumTiles)));
  const requested = Math.min(hardCapacity, Math.max(1, Math.ceil(logicalBrickCount * fraction)));
  const maxTexels = Math.max(tileSize, Math.floor(options.maxTextureDimension3D ?? 2048));
  const maxTilesPerAxis = Math.max(1, Math.floor(maxTexels / tileSize));
  // Near-cubic tile packing keeps every atlas axis under the device limit.
  const gx = Math.min(maxTilesPerAxis, Math.max(1, Math.ceil(Math.cbrt(requested))));
  const gy = Math.min(maxTilesPerAxis, Math.max(1, Math.ceil(Math.sqrt(requested / gx))));
  const gz = Math.min(maxTilesPerAxis, Math.max(1, Math.ceil(requested / (gx * gy))));
  const capacity = Math.min(requested, gx * gy * gz);
  const tileGridDimensions = [gx, gy, gz] as const;
  const atlasDimensions = [gx * tileSize, gy * tileSize, gz * tileSize] as const;
  // phi r32float (4 B) + velocity rgba32float (16 B) per texel.
  const bytesPerTile = tileSize ** 3 * 20;
  return {
    dimensions: [...dimensions] as [number, number, number], brickDimensions, logicalBrickCount, brickSize,
    tileSize, capacity, tileGridDimensions, atlasDimensions,
    degraded: capacity < requested,
    bytesPerTile,
    allocatedTextureBytes: atlasDimensions[0] * atlasDimensions[1] * atlasDimensions[2] * 20,
  };
}

export interface FluidBrickAtlasStats {
  residentTiles: number;
  activated: number;
  retired: number;
  free: number;
  overflow: number;
  peakResidentTiles: number;
  generation: number;
  capacity: number;
  /** Hardware-filtered helper error vs dense hardware sampling (filterable devices). */
  maxAbsPhiError: number;
  maxAbsVelocityError: number;
  /** FP32 manual-trilinear round-trip error; expected to be ~0 in mirror mode. */
  maxAbsPhiErrorManual: number;
  maxAbsVelocityErrorManual: number;
  comparedSamples: number;
}

const atlasParamsWGSL = /* wgsl */ `
struct AtlasParams {
  dims: vec4u,
  brickDims: vec4u,
  tileGrid: vec4u,
  capacitySeed: vec4u,
  cell: vec4f,
}
`;

/**
 * Reusable atlas sampling snippet.
 *
 * Requires these declarations in the including module:
 *   var<uniform> atlasParams: AtlasParams  (dims.xyz = dense cells, dims.w = brickSize,
 *     tileGrid.xyz = tiles per axis, tileGrid.w = tileSize, capacitySeed.x = tile capacity)
 *   var<storage, read> brickAtlasPageTable: array<u32>
 *   var brickAtlasPhi: texture_3d<f32>
 *   var brickAtlasVelocity: texture_3d<f32>
 *   var denseLevelSet: texture_3d<f32>
 *   var denseVelocity: texture_3d<f32>
 *   var brickAtlasSampler: sampler          (filterable variant only)
 *
 * Positions are dense cell-center coordinates (integer position = cell center),
 * matching the dense trilinear convention used throughout the solver. When the
 * containing brick has no atlas slot the helpers fall back to the dense
 * textures, so callers see a total function over the whole domain.
 */
export function fluidBrickAtlasSamplingWGSL(filterable: boolean): string {
  const hardware = /* wgsl */ `
fn brickAtlasSamplePhi(position: vec3f) -> f32 {
  let slotBrick = brickAtlasSlotFor(position);
  if (slotBrick.x == BRICK_ATLAS_INVALID) { return brickAtlasDensePhi(position); }
  let uvw = brickAtlasTileUVW(slotBrick.x, slotBrick.yzw, position);
  return textureSampleLevel(brickAtlasPhi, brickAtlasSampler, uvw, 0.0).x;
}
fn brickAtlasSampleVelocity(position: vec3f) -> vec3f {
  let slotBrick = brickAtlasSlotFor(position);
  if (slotBrick.x == BRICK_ATLAS_INVALID) { return brickAtlasDenseVelocity(position); }
  let uvw = brickAtlasTileUVW(slotBrick.x, slotBrick.yzw, position);
  return textureSampleLevel(brickAtlasVelocity, brickAtlasSampler, uvw, 0.0).xyz;
}
fn brickAtlasDensePhi(position: vec3f) -> f32 {
  let p = brickAtlasClampPosition(position);
  return textureSampleLevel(denseLevelSet, brickAtlasSampler, (p + vec3f(0.5)) / vec3f(atlasParams.dims.xyz), 0.0).x;
}
fn brickAtlasDenseVelocity(position: vec3f) -> vec3f {
  let p = brickAtlasClampPosition(position);
  return textureSampleLevel(denseVelocity, brickAtlasSampler, (p + vec3f(0.5)) / vec3f(atlasParams.dims.xyz), 0.0).xyz;
}
`;
  const manualAlias = /* wgsl */ `
fn brickAtlasSamplePhi(position: vec3f) -> f32 { return brickAtlasSamplePhiManual(position); }
fn brickAtlasSampleVelocity(position: vec3f) -> vec3f { return brickAtlasSampleVelocityManual(position); }
fn brickAtlasDensePhi(position: vec3f) -> f32 { return brickAtlasDensePhiManual(position); }
fn brickAtlasDenseVelocity(position: vec3f) -> vec3f { return brickAtlasDenseVelocityManual(position); }
`;
  return /* wgsl */ `
const BRICK_ATLAS_INVALID: u32 = 0xffffffffu;

fn brickAtlasClampPosition(position: vec3f) -> vec3f {
  return clamp(position, vec3f(0.0), vec3f(atlasParams.dims.xyz - vec3u(1u)));
}
/** x = slot (or INVALID), yzw = containing brick coordinate. */
fn brickAtlasSlotFor(position: vec3f) -> vec4u {
  let p = brickAtlasClampPosition(position);
  let brick = min(vec3u(floor(p / f32(atlasParams.dims.w))), atlasParams.brickDims.xyz - vec3u(1u));
  let brickIndex = brick.x + atlasParams.brickDims.x * (brick.y + atlasParams.brickDims.y * brick.z);
  if (brickIndex >= arrayLength(&brickAtlasPageTable)) { return vec4u(BRICK_ATLAS_INVALID, brick); }
  let slot = brickAtlasPageTable[brickIndex];
  if (slot >= atlasParams.capacitySeed.x) { return vec4u(BRICK_ATLAS_INVALID, brick); }
  return vec4u(slot, brick);
}
fn brickAtlasTileOrigin(slot: u32) -> vec3u {
  let tile = vec3u(slot % atlasParams.tileGrid.x, (slot / atlasParams.tileGrid.x) % atlasParams.tileGrid.y,
    slot / (atlasParams.tileGrid.x * atlasParams.tileGrid.y));
  return tile * atlasParams.tileGrid.w;
}
fn brickAtlasTileUVW(slot: u32, brick: vec3u, position: vec3f) -> vec3f {
  let p = brickAtlasClampPosition(position);
  let local = p - vec3f(brick * atlasParams.dims.w);
  // Cell-center c of local cell l lives at tile texel l+1; +0.5 centers the texel.
  let texel = vec3f(brickAtlasTileOrigin(slot)) + local + vec3f(1.5);
  return texel / vec3f(atlasParams.tileGrid.xyz * atlasParams.tileGrid.w);
}
/** Atlas texel holding dense cell \`cell\` through brick \`brick\`'s tile (apron included). */
fn brickAtlasTexelFor(slot: u32, brick: vec3u, cell: vec3i) -> vec3i {
  let local = cell - vec3i(brick * atlasParams.dims.w) + vec3i(1);
  return vec3i(brickAtlasTileOrigin(slot)) + local;
}
fn brickAtlasDensePhiManual(position: vec3f) -> f32 {
  let p = brickAtlasClampPosition(position);
  let a = vec3i(floor(p));
  let b = min(a + vec3i(1), vec3i(atlasParams.dims.xyz) - vec3i(1));
  let t = fract(p);
  let p000 = textureLoad(denseLevelSet, vec3i(a.x,a.y,a.z), 0).x; let p100 = textureLoad(denseLevelSet, vec3i(b.x,a.y,a.z), 0).x;
  let p010 = textureLoad(denseLevelSet, vec3i(a.x,b.y,a.z), 0).x; let p110 = textureLoad(denseLevelSet, vec3i(b.x,b.y,a.z), 0).x;
  let p001 = textureLoad(denseLevelSet, vec3i(a.x,a.y,b.z), 0).x; let p101 = textureLoad(denseLevelSet, vec3i(b.x,a.y,b.z), 0).x;
  let p011 = textureLoad(denseLevelSet, vec3i(a.x,b.y,b.z), 0).x; let p111 = textureLoad(denseLevelSet, vec3i(b.x,b.y,b.z), 0).x;
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn brickAtlasDenseVelocityManual(position: vec3f) -> vec3f {
  let p = brickAtlasClampPosition(position);
  let a = vec3i(floor(p));
  let b = min(a + vec3i(1), vec3i(atlasParams.dims.xyz) - vec3i(1));
  let t = fract(p);
  let v000 = textureLoad(denseVelocity, vec3i(a.x,a.y,a.z), 0).xyz; let v100 = textureLoad(denseVelocity, vec3i(b.x,a.y,a.z), 0).xyz;
  let v010 = textureLoad(denseVelocity, vec3i(a.x,b.y,a.z), 0).xyz; let v110 = textureLoad(denseVelocity, vec3i(b.x,b.y,a.z), 0).xyz;
  let v001 = textureLoad(denseVelocity, vec3i(a.x,a.y,b.z), 0).xyz; let v101 = textureLoad(denseVelocity, vec3i(b.x,a.y,b.z), 0).xyz;
  let v011 = textureLoad(denseVelocity, vec3i(a.x,b.y,b.z), 0).xyz; let v111 = textureLoad(denseVelocity, vec3i(b.x,b.y,b.z), 0).xyz;
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y), mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y), t.z);
}
fn brickAtlasSamplePhiManual(position: vec3f) -> f32 {
  let slotBrick = brickAtlasSlotFor(position);
  if (slotBrick.x == BRICK_ATLAS_INVALID) { return brickAtlasDensePhiManual(position); }
  let p = brickAtlasClampPosition(position);
  let a = vec3i(floor(p));
  let t = fract(p);
  let slot = slotBrick.x; let brick = slotBrick.yzw;
  let p000 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a), 0).x;
  let p100 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(1,0,0)), 0).x;
  let p010 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(0,1,0)), 0).x;
  let p110 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(1,1,0)), 0).x;
  let p001 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(0,0,1)), 0).x;
  let p101 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(1,0,1)), 0).x;
  let p011 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(0,1,1)), 0).x;
  let p111 = textureLoad(brickAtlasPhi, brickAtlasTexelFor(slot, brick, a + vec3i(1,1,1)), 0).x;
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn brickAtlasSampleVelocityManual(position: vec3f) -> vec3f {
  let slotBrick = brickAtlasSlotFor(position);
  if (slotBrick.x == BRICK_ATLAS_INVALID) { return brickAtlasDenseVelocityManual(position); }
  let p = brickAtlasClampPosition(position);
  let a = vec3i(floor(p));
  let t = fract(p);
  let slot = slotBrick.x; let brick = slotBrick.yzw;
  let v000 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a), 0).xyz;
  let v100 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(1,0,0)), 0).xyz;
  let v010 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(0,1,0)), 0).xyz;
  let v110 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(1,1,0)), 0).xyz;
  let v001 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(0,0,1)), 0).xyz;
  let v101 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(1,0,1)), 0).xyz;
  let v011 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(0,1,1)), 0).xyz;
  let v111 = textureLoad(brickAtlasVelocity, brickAtlasTexelFor(slot, brick, a + vec3i(1,1,1)), 0).xyz;
  return mix(mix(mix(v000,v100,t.x),mix(v010,v110,t.x),t.y), mix(mix(v001,v101,t.x),mix(v011,v111,t.x),t.y), t.z);
}
${filterable ? hardware : manualAlias}
`;
}

// Control words mirror the sparse surface band: free, generation, overflow,
// activated, retired, resident, -, -, peak resident.
export const brickAtlasLifecycleShader = /* wgsl */ `
${atlasParamsWGSL}
@group(0) @binding(0) var<storage, read> residencyStates: array<u32>;
@group(0) @binding(1) var<storage, read_write> pageTable: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> activeTiles: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> atlasParams: AtlasParams;

const INVALID: u32 = 0xffffffffu;
const RESIDENT: u32 = 1u;

@compute @workgroup_size(1)
fn resetTileCounters() {
  atomicAdd(&control[1], 1u);
  for (var i = 2u; i <= 7u; i += 1u) { atomicStore(&control[i], 0u); }
  atomicStore(&activeTiles[0], 0u);
  atomicStore(&activeTiles[1], 0u);
  atomicStore(&activeTiles[2], 1u);
  atomicStore(&activeTiles[3], 1u);
}

@compute @workgroup_size(64)
fn retireTiles(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  if (brickIndex >= atlasParams.brickDims.w || brickIndex >= arrayLength(&residencyStates)) { return; }
  if ((residencyStates[brickIndex] & RESIDENT) != 0u) { return; }
  let slot = atomicExchange(&pageTable[brickIndex], INVALID);
  if (slot == INVALID || slot >= atlasParams.capacitySeed.x) { return; }
  let freeSlot = atomicAdd(&control[0], 1u);
  if (freeSlot < arrayLength(&freeList)) { freeList[freeSlot] = slot; }
  else { atomicStore(&control[2], 1u); }
  atomicAdd(&control[4], 1u);
}

@compute @workgroup_size(64)
fn activateAndList(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  if (brickIndex >= atlasParams.brickDims.w || brickIndex >= arrayLength(&residencyStates)) { return; }
  if ((residencyStates[brickIndex] & RESIDENT) == 0u) { return; }
  var slot = atomicLoad(&pageTable[brickIndex]);
  if (slot == INVALID) {
    let oldFree = atomicSub(&control[0], 1u);
    if (oldFree == 0u) {
      // Exhausted: residency stays authoritative and sampling falls back to
      // the dense fields for this brick, so overflow degrades rather than fails.
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      return;
    }
    let freeIndex = oldFree - 1u;
    if (freeIndex >= arrayLength(&freeList)) {
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      return;
    }
    slot = freeList[freeIndex];
    if (slot >= atlasParams.capacitySeed.x) {
      atomicAdd(&control[0], 1u);
      atomicStore(&control[2], 1u);
      return;
    }
    atomicStore(&pageTable[brickIndex], slot);
    atomicAdd(&control[3], 1u);
  }
  let activeIndex = atomicAdd(&activeTiles[0], 1u);
  if (4u + activeIndex < arrayLength(&activeTiles)) {
    atomicStore(&activeTiles[4u + activeIndex], brickIndex);
  } else {
    atomicStore(&control[2], 1u);
    return;
  }
  let resident = atomicAdd(&control[5], 1u) + 1u;
  atomicMax(&control[8], resident);
}

@compute @workgroup_size(1)
fn finalizeTileDispatch() {
  let resident = min(atomicLoad(&activeTiles[0]), atlasParams.capacitySeed.x);
  let tileVoxels = atlasParams.tileGrid.w * atlasParams.tileGrid.w * atlasParams.tileGrid.w;
  let blocks = (resident * tileVoxels + 255u) / 256u;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  atomicStore(&activeTiles[1], x);
  atomicStore(&activeTiles[2], y);
}
`;

export const brickAtlasMirrorShader = /* wgsl */ `
${atlasParamsWGSL}
@group(0) @binding(0) var denseLevelSet: texture_3d<f32>;
@group(0) @binding(1) var denseVelocity: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> pageTable: array<u32>;
@group(0) @binding(3) var<storage, read> activeTiles: array<u32>;
@group(0) @binding(4) var phiAtlas: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var velocityAtlas: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var<uniform> atlasParams: AtlasParams;

const INVALID: u32 = 0xffffffffu;

fn brickCoordinate(index: u32) -> vec3u {
  let b = atlasParams.brickDims;
  return vec3u(index % b.x, (index / b.x) % b.y, index / (b.x * b.y));
}
fn tileOrigin(slot: u32) -> vec3u {
  let g = atlasParams.tileGrid;
  return vec3u(slot % g.x, (slot / g.x) % g.y, slot / (g.x * g.y)) * g.w;
}

// The apron reads dense neighbor cells directly with clamp-to-edge semantics,
// matching the dense sampler exactly so mirror-mode parity holds through the
// container walls without a dedicated boundary-condition pass.
@compute @workgroup_size(256)
fn mirrorResident(@builtin(global_invocation_id) gid: vec3u) {
  let stream = gid.x + gid.y * activeTiles[1] * 256u;
  let tileSize = atlasParams.tileGrid.w;
  let tileVoxels = tileSize * tileSize * tileSize;
  let activeIndex = stream / tileVoxels;
  if (activeIndex >= activeTiles[0] || 4u + activeIndex >= arrayLength(&activeTiles)) { return; }
  let brickIndex = activeTiles[4u + activeIndex];
  if (brickIndex >= atlasParams.brickDims.w || brickIndex >= arrayLength(&pageTable)) { return; }
  let slot = pageTable[brickIndex];
  if (slot == INVALID || slot >= atlasParams.capacitySeed.x) { return; }
  let localIndex = stream - activeIndex * tileVoxels;
  let local = vec3u(localIndex % tileSize, (localIndex / tileSize) % tileSize, localIndex / (tileSize * tileSize));
  let cell = vec3i(brickCoordinate(brickIndex) * atlasParams.dims.w) + vec3i(local) - vec3i(1);
  let clamped = clamp(cell, vec3i(0), vec3i(atlasParams.dims.xyz) - vec3i(1));
  let texel = vec3i(tileOrigin(slot)) + vec3i(local);
  textureStore(phiAtlas, texel, vec4f(textureLoad(denseLevelSet, clamped, 0).x, 0.0, 0.0, 0.0));
  textureStore(velocityAtlas, texel, vec4f(textureLoad(denseVelocity, clamped, 0).xyz, 0.0));
}
`;

export function brickAtlasValidateShader(filterable: boolean): string {
  return /* wgsl */ `
${atlasParamsWGSL}
@group(0) @binding(0) var denseLevelSet: texture_3d<f32>;
@group(0) @binding(1) var denseVelocity: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> brickAtlasPageTable: array<u32>;
@group(0) @binding(3) var<storage, read> activeTiles: array<u32>;
@group(0) @binding(4) var brickAtlasPhi: texture_3d<f32>;
@group(0) @binding(5) var brickAtlasVelocity: texture_3d<f32>;
${filterable ? "@group(0) @binding(6) var brickAtlasSampler: sampler;" : ""}
@group(0) @binding(7) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> atlasParams: AtlasParams;

${fluidBrickAtlasSamplingWGSL(filterable)}

fn hash(seed: u32) -> u32 {
  var h = seed * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn jitter(seed: u32) -> f32 { return f32(hash(seed)) / 4294967296.0; }
fn accumulate(index: u32, error: f32) {
  if (index < arrayLength(&stats)) { atomicMax(&stats[index], bitcast<u32>(max(error, 0.0))); }
}

// Round-trip and seam-continuity check: jittered positions inside and across
// resident bricks must sample identically from the atlas tile (via its apron)
// and from the dense fields. The manual FP32 channel is exact; the hardware
// channel additionally covers the filtered sampler path within its fixed-point
// weight precision.
@compute @workgroup_size(256)
fn compareAtlasToDense(@builtin(global_invocation_id) gid: vec3u) {
  let stream = gid.x + gid.y * activeTiles[1] * 256u;
  let tileSize = atlasParams.tileGrid.w;
  let tileVoxels = tileSize * tileSize * tileSize;
  let activeIndex = stream / tileVoxels;
  if (activeIndex >= activeTiles[0] || 4u + activeIndex >= arrayLength(&activeTiles)) { return; }
  let brickIndex = activeTiles[4u + activeIndex];
  if (brickIndex >= atlasParams.brickDims.w || brickIndex >= arrayLength(&brickAtlasPageTable)) { return; }
  if (brickAtlasPageTable[brickIndex] >= atlasParams.capacitySeed.x) { return; }
  let localIndex = stream - activeIndex * tileVoxels;
  let local = vec3u(localIndex % tileSize, (localIndex / tileSize) % tileSize, localIndex / (tileSize * tileSize));
  let b = atlasParams.brickDims;
  let brick = vec3u(brickIndex % b.x, (brickIndex / b.x) % b.y, brickIndex / (b.x * b.y));
  let base = vec3f(brick * atlasParams.dims.w) + vec3f(local) - vec3f(1.0);
  let seed = stream * 3u + atlasParams.capacitySeed.y * 65599u;
  let position = brickAtlasClampPosition(base + vec3f(jitter(seed), jitter(seed + 1u), jitter(seed + 2u)));
  let manualPhi = abs(brickAtlasSamplePhiManual(position) - brickAtlasDensePhiManual(position));
  let manualVelocity = length(brickAtlasSampleVelocityManual(position) - brickAtlasDenseVelocityManual(position));
  accumulate(2u, manualPhi);
  accumulate(3u, manualVelocity);
  ${filterable ? /* wgsl */ `
  let hardwarePhi = abs(brickAtlasSamplePhi(position) - brickAtlasDensePhi(position));
  let hardwareVelocity = length(brickAtlasSampleVelocity(position) - brickAtlasDenseVelocity(position));
  accumulate(0u, hardwarePhi);
  accumulate(1u, hardwareVelocity);
  ` : ""}
  if (4u < arrayLength(&stats)) { atomicAdd(&stats[4], 1u); }
}
`;
}

function buffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBufferView<ArrayBuffer>,
) {
  const result = device.createBuffer({ label, size: Math.max(4, size), usage });
  if (data?.byteLength) device.queue.writeBuffer(result, 0, data);
  return result;
}

export class WebGPUFluidBrickAtlas {
  readonly plan: FluidBrickAtlasPlan;
  readonly mode: FluidBrickAtlasMode;
  readonly filterable: boolean;
  readonly allocatedBytes: number;
  readonly phiAtlas: GPUTexture;
  readonly velocityAtlas: GPUTexture;
  readonly pageTable: GPUBuffer;
  readonly activeTiles: GPUBuffer;
  readonly params: GPUBuffer;

  private readonly freeList: GPUBuffer;
  private readonly control: GPUBuffer;
  private readonly stats: GPUBuffer;
  private readonly lifecycleGroup: GPUBindGroup;
  private readonly lifecyclePipelines: Record<"reset" | "retire" | "activate" | "finalize", GPUComputePipeline>;
  private readonly mirrorLayout: GPUBindGroupLayout;
  private readonly mirrorPipeline: GPUComputePipeline;
  private readonly validateLayout: GPUBindGroupLayout;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly sampler?: GPUSampler;
  private readonly validate: boolean;
  private frame = 0;
  private destroyed = false;
  private statsReadback?: GPUBuffer;
  private statsReadbackBusy = false;

  constructor(
    private readonly device: GPUDevice,
    dimensions: readonly [number, number, number],
    residency: GPUFluidBrickResidency,
    options: FluidBrickAtlasOptions = {},
  ) {
    this.plan = planFluidBrickAtlas(dimensions, {
      ...options,
      brickSize: options.brickSize ?? residency.brickSize,
      maxTextureDimension3D: Number(device.limits.maxTextureDimension3D),
    });
    this.mode = options.mode ?? "mirror";
    this.validate = options.validate ?? true;
    this.filterable = device.features.has("float32-filterable" as GPUFeatureName);
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    this.phiAtlas = device.createTexture({ label: "Fluid brick atlas phi", size: [...this.plan.atlasDimensions], dimension: "3d", format: "r32float", usage: textureUsage });
    this.velocityAtlas = device.createTexture({ label: "Fluid brick atlas velocity", size: [...this.plan.atlasDimensions], dimension: "3d", format: "rgba32float", usage: textureUsage });
    const pageTableData = new Uint32Array(this.plan.logicalBrickCount); pageTableData.fill(BRICK_ATLAS_INVALID_SLOT);
    const freeData = Uint32Array.from({ length: this.plan.capacity }, (_, index) => index);
    const controlData = new Uint32Array(16); controlData[0] = this.plan.capacity;
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.pageTable = buffer(device, "Fluid brick atlas page table", pageTableData.byteLength, storageCopy, pageTableData);
    this.freeList = buffer(device, "Fluid brick atlas free tiles", freeData.byteLength, storageCopy, freeData);
    this.control = buffer(device, "Fluid brick atlas control", 64, storageCopy, controlData);
    this.activeTiles = buffer(device, "Fluid brick atlas active tiles and dispatch", (4 + this.plan.capacity) * 4, storageCopy | GPUBufferUsage.INDIRECT);
    this.stats = buffer(device, "Fluid brick atlas validation stats", 32, storageCopy);
    this.params = buffer(device, "Fluid brick atlas parameters", 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.writeParams();

    const lifecycleLayout = device.createBindGroupLayout({ label: "Fluid brick atlas lifecycle layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ...[1, 2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const lifecycleModule = device.createShaderModule({ label: "Fluid brick atlas lifecycle", code: brickAtlasLifecycleShader });
    const mirrorModule = device.createShaderModule({ label: "Fluid brick atlas mirror", code: brickAtlasMirrorShader });
    const validateModule = device.createShaderModule({ label: "Fluid brick atlas validation", code: brickAtlasValidateShader(this.filterable) });
    void Promise.all([lifecycleModule.getCompilationInfo(), mirrorModule.getCompilationInfo(), validateModule.getCompilationInfo()]).then((reports) => {
      for (const report of reports) for (const message of report.messages) if (message.type === "error") {
        console.error(`Fluid brick atlas WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
      }
    }).catch(() => { /* device loss is handled by the owning renderer */ });
    const lifecyclePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [lifecycleLayout] });
    const lifecycle = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: lifecyclePipelineLayout, compute: { module: lifecycleModule, entryPoint } });
    this.lifecyclePipelines = {
      reset: lifecycle("Reset fluid brick atlas counters", "resetTileCounters"),
      retire: lifecycle("Retire fluid brick atlas tiles", "retireTiles"),
      activate: lifecycle("Activate and list fluid brick atlas tiles", "activateAndList"),
      finalize: lifecycle("Finalize fluid brick atlas dispatch", "finalizeTileDispatch"),
    };
    this.lifecycleGroup = device.createBindGroup({ label: "Fluid brick atlas lifecycle bindings", layout: lifecycleLayout, entries: [
      { binding: 0, resource: { buffer: residency.stateBuffer } },
      { binding: 1, resource: { buffer: this.pageTable } },
      { binding: 2, resource: { buffer: this.freeList } },
      { binding: 3, resource: { buffer: this.control } },
      { binding: 4, resource: { buffer: this.activeTiles } },
      { binding: 5, resource: { buffer: this.params } },
    ] });
    this.mirrorLayout = device.createBindGroupLayout({ label: "Fluid brick atlas mirror layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    this.mirrorPipeline = device.createComputePipeline({
      label: "Mirror dense fields into fluid brick atlas",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.mirrorLayout] }),
      compute: { module: mirrorModule, entryPoint: "mirrorResident" },
    });
    const sampleType = this.filterable ? "float" as const : "unfilterable-float" as const;
    this.validateLayout = device.createBindGroupLayout({ label: "Fluid brick atlas validation layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType, viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType, viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType, viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType, viewDimension: "3d" } },
      ...(this.filterable ? [{ binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" as const } }] : []),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    this.validatePipeline = device.createComputePipeline({
      label: "Compare fluid brick atlas to dense fields",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.validateLayout] }),
      compute: { module: validateModule, entryPoint: "compareAtlasToDense" },
    });
    if (this.filterable) this.sampler = device.createSampler({ label: "Fluid brick atlas trilinear sampler", magFilter: "linear", minFilter: "linear" });
    this.allocatedBytes = this.plan.allocatedTextureBytes + pageTableData.byteLength + freeData.byteLength
      + 64 + (4 + this.plan.capacity) * 4 + 32 + 80;
  }

  private writeParams() {
    const data = new ArrayBuffer(80), u = new Uint32Array(data), f = new Float32Array(data);
    const p = this.plan;
    u.set([p.dimensions[0], p.dimensions[1], p.dimensions[2], p.brickSize], 0);
    u.set([p.brickDimensions[0], p.brickDimensions[1], p.brickDimensions[2], p.logicalBrickCount], 4);
    u.set([p.tileGridDimensions[0], p.tileGridDimensions[1], p.tileGridDimensions[2], p.tileSize], 8);
    u.set([p.capacity, this.frame >>> 0, 0, 0], 12);
    f.set([0, 0, 0, 0], 16);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  /** Encode slot lifecycle + dense mirror (+ validation) after residency classification. */
  encode(encoder: GPUCommandEncoder, denseLevelSet: GPUTexture, denseVelocity: GPUTexture): void {
    if (this.destroyed) return;
    this.frame += 1;
    // Refresh the jitter seed so validation coverage moves between frames.
    this.device.queue.writeBuffer(this.params, 52, new Uint32Array([this.frame >>> 0]));
    const bricks = Math.ceil(this.plan.logicalBrickCount / 64);
    const lifecycle = encoder.beginComputePass({ label: "Fluid brick atlas tile lifecycle" });
    lifecycle.setBindGroup(0, this.lifecycleGroup);
    lifecycle.setPipeline(this.lifecyclePipelines.reset); lifecycle.dispatchWorkgroups(1);
    lifecycle.setPipeline(this.lifecyclePipelines.retire); lifecycle.dispatchWorkgroups(bricks);
    lifecycle.setPipeline(this.lifecyclePipelines.activate); lifecycle.dispatchWorkgroups(bricks);
    lifecycle.setPipeline(this.lifecyclePipelines.finalize); lifecycle.dispatchWorkgroups(1);
    lifecycle.end();
    const mirrorGroup = this.device.createBindGroup({ label: "Fluid brick atlas mirror bindings", layout: this.mirrorLayout, entries: [
      { binding: 0, resource: denseLevelSet.createView() },
      { binding: 1, resource: denseVelocity.createView() },
      { binding: 2, resource: { buffer: this.pageTable } },
      { binding: 3, resource: { buffer: this.activeTiles } },
      { binding: 4, resource: this.phiAtlas.createView() },
      { binding: 5, resource: this.velocityAtlas.createView() },
      { binding: 6, resource: { buffer: this.params } },
    ] });
    const mirror = encoder.beginComputePass({ label: "Mirror dense fields into fluid brick atlas" });
    mirror.setPipeline(this.mirrorPipeline);
    mirror.setBindGroup(0, mirrorGroup);
    mirror.dispatchWorkgroupsIndirect(this.activeTiles, BRICK_ATLAS_ACTIVE_DISPATCH_OFFSET_BYTES);
    mirror.end();
    if (!this.validate) return;
    encoder.clearBuffer(this.stats);
    const validateGroup = this.device.createBindGroup({ label: "Fluid brick atlas validation bindings", layout: this.validateLayout, entries: [
      { binding: 0, resource: denseLevelSet.createView() },
      { binding: 1, resource: denseVelocity.createView() },
      { binding: 2, resource: { buffer: this.pageTable } },
      { binding: 3, resource: { buffer: this.activeTiles } },
      { binding: 4, resource: this.phiAtlas.createView() },
      { binding: 5, resource: this.velocityAtlas.createView() },
      ...(this.filterable && this.sampler ? [{ binding: 6, resource: this.sampler }] : []),
      { binding: 7, resource: { buffer: this.stats } },
      { binding: 8, resource: { buffer: this.params } },
    ] });
    const compare = encoder.beginComputePass({ label: "Compare fluid brick atlas to dense fields" });
    compare.setPipeline(this.validatePipeline);
    compare.setBindGroup(0, validateGroup);
    compare.dispatchWorkgroupsIndirect(this.activeTiles, BRICK_ATLAS_ACTIVE_DISPATCH_OFFSET_BYTES);
    compare.end();
  }

  async readStats(): Promise<FluidBrickAtlasStats> {
    const empty: FluidBrickAtlasStats = {
      residentTiles: 0, activated: 0, retired: 0, free: this.plan.capacity, overflow: 0, peakResidentTiles: 0,
      generation: 0, capacity: this.plan.capacity, maxAbsPhiError: 0, maxAbsVelocityError: 0,
      maxAbsPhiErrorManual: 0, maxAbsVelocityErrorManual: 0, comparedSamples: 0,
    };
    if (this.destroyed) return empty;
    const pooled = !this.statsReadbackBusy;
    const readback = pooled
      ? (this.statsReadback ??= this.device.createBuffer({ label: "Fluid brick atlas stats readback", size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }))
      : this.device.createBuffer({ label: "Fluid brick atlas stats readback (transient)", size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (pooled) this.statsReadbackBusy = true;
    const encoder = this.device.createCommandEncoder({ label: "Read fluid brick atlas stats" });
    encoder.copyBufferToBuffer(this.control, 0, readback, 0, 64);
    encoder.copyBufferToBuffer(this.stats, 0, readback, 64, 32);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, 96));
      const errors = new Float32Array(words.buffer, words.byteOffset + 64, 4);
      return {
        free: words[0], generation: words[1], overflow: words[2], activated: words[3], retired: words[4],
        residentTiles: words[5], peakResidentTiles: words[8], capacity: this.plan.capacity,
        maxAbsPhiError: errors[0], maxAbsVelocityError: errors[1],
        maxAbsPhiErrorManual: errors[2], maxAbsVelocityErrorManual: errors[3],
        comparedSamples: words[16 + 4],
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      if (pooled) this.statsReadbackBusy = false;
      else readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.phiAtlas.destroy();
    this.velocityAtlas.destroy();
    this.statsReadback?.destroy();
    for (const resource of [this.pageTable, this.freeList, this.control, this.activeTiles, this.stats, this.params]) resource.destroy();
  }
}
