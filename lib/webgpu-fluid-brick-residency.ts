/**
 * GPU-owned logical residency for finest-level fluid bricks.
 *
 * The dense textures remain a compatibility backing store while kernels are
 * migrated to brick payloads.  This page table is nevertheless authoritative
 * for sparse publication and octree work scheduling: only resident core/halo
 * bricks are emitted to the worklist, and dry bricks retire after a short
 * hysteresis window.
 */

export const FLUID_BRICK_RESIDENT = 1;
export const FLUID_BRICK_CORE = 2;
export const FLUID_BRICK_HALO = 4;
export const FLUID_BRICK_ACTIVATED = 8;
/** Scratch bit carried between the split classify/expand/emit dispatches. */
export const FLUID_BRICK_WAS_RESIDENT = 32;

export const FLUID_BRICK_WORKLIST_HEADER_WORDS = 16;
export const FLUID_BRICK_STATE_STRIDE_BYTES = 4;
export const FLUID_BRICK_WORKLIST_ENTRY_STRIDE_BYTES = 8;
export const FLUID_BRICK_WORKLIST_WORDS = Object.freeze({
  activeCount: 0,
  retiredCount: 4,
  coreCount: 8,
  haloCount: 9,
  activatedCount: 10,
  retiredStatsCount: 11,
  generation: 15,
} as const);
export const FLUID_BRICK_ACTIVE_DISPATCH_OFFSET_BYTES = 4;
export const FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES = 20;
/** Generic 4x4x4 cell kernels consume 64 cells/workgroup. */
export const FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES = 48;
/** Backward-compatible surface scheduler name for the same generic stream. */
export const FLUID_BRICK_ACTIVE_SURFACE_DISPATCH_OFFSET_BYTES = FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES;
/**
 * Topology-tile worklist header layout (mirrors the brick worklist): word 0 is
 * the active tile count with its 4x4x4-workgroup indirect dispatch in words
 * 1..3, word 4 the retired tile count with its dispatch in words 5..7. Tile
 * indices follow the 16-word header, actives first then retireds.
 */
export const FLUID_TILE_WORKLIST_HEADER_WORDS = 16;
export const FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES = 4;
export const FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES = 20;
/** Candidate-origin dispatches used by refinement/balance (header words 8..14). */
export const FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES = 32;
export const FLUID_TILE_RETIRED_CANDIDATE_DISPATCH_OFFSET_BYTES = 48;

export interface FluidBrickResidencyOptions {
  brickSize?: 4 | 8;
  /** Signed-distance air band retained for advection/interpolation stencils. */
  haloCells?: number;
  /** Consecutive dry publications before a formerly resident brick is freed. */
  retireAfterFrames?: number;
  /**
   * Retain every brick containing liquid, not only the two-sided interface
   * band. Bulk-field atlases use this independent residency domain so deep
   * velocity remains available without making surface-only kernels visit the
   * whole wet volume.
   */
  includeLiquidInterior?: boolean;
  /** Tree leaf index for every x-major solver brick. */
  leafIndices?: Uint32Array<ArrayBuffer>;
  leafCapacity?: number;
  /**
   * Power-of-two bricks per topology-tile axis. A tile is the atomic unit of
   * octree topology rebuilds: max(brickSize, maximumLeafSize) cells, so every
   * dyadic pressure leaf lies inside exactly one tile and a partial rebuild
   * can never split a leaf. 1 keeps tiles congruent with bricks.
   */
  topologyTileBricks?: number;
  /**
   * Direct-paged topology mode. Residency is derived exclusively from compact
   * surface candidates, so legacy leaf publication buffers collapse to
   * one-word bindable fallbacks. Bootstrap texture classification remains
   * valid: worklists/states are complete while bounded leaf publication is a
   * deliberate no-op for every nonzero identity leaf.
   */
  surfaceCandidatesOnly?: boolean;
  /**
   * Physical sparse-key slots retained by direct surface-candidate authority.
   * Logical brick coordinates remain unchanged; exhaustion rejects the whole
   * candidate generation and preserves the previous publication.
   */
  surfaceCandidateBrickCapacity?: number;
  /** Sparse topology-tile key slots. See `surfaceCandidateBrickCapacity`. */
  surfaceCandidateTileCapacity?: number;
}

export interface FluidBrickResidencyAllocationPlan {
  readonly brickCapacity: number;
  readonly tileCapacity: number;
  readonly leafCapacity: number;
  readonly identityMapping: "implicit" | "explicit";
  readonly surfaceCandidatesOnly: boolean;
  readonly sparseKeyPools: boolean;
  readonly brickStateCapacity: number;
  readonly tileStateCapacity: number;
  readonly stateBytes: number;
  readonly worklistBytes: number;
  readonly tileWorklistBytes: number;
  readonly tileStateBytes: number;
  readonly leafIndexBytes: number;
  readonly leafStateBytes: number;
  readonly parameterBytes: number;
  /** A/B scratch state, worklists, and the GPU commit predicate. */
  readonly transactionalBytes: number;
  readonly allocatedBytes: number;
  /** Bytes avoided by the sentinel-backed implicit identity mapping. */
  readonly savedIdentityBytes: number;
  /** Bytes avoided by omitting the legacy per-leaf state mirror. */
  readonly savedLeafStateBytes: number;
  /** Box-indexed scheduler bytes not allocated by sparse candidate authority. */
  readonly savedSchedulerBytes: number;
  /** Sparse scheduler bytes minus the dense logical-key scheduler bytes. */
  readonly schedulerByteDelta: number;
}

/**
 * Exact persistent allocation accounting for the compatibility residency ABI.
 *
 * The common direct-paged path has identity brick/leaf ownership. It stores a
 * single sentinel word and reconstructs the leaf index in WGSL, instead of
 * retaining a box-sized `0..capacity-1` identity array. Explicit non-identity
 * mappings remain available for compatibility consumers.
 */
export function planFluidBrickResidencyAllocation(
  brickDimensions: readonly [number, number, number],
  tileDimensions: readonly [number, number, number],
  leafCapacity: number,
  explicitLeafMapping = false,
  surfaceCandidatesOnly = false,
  surfaceCandidateBrickCapacity?: number,
  surfaceCandidateTileCapacity?: number,
): FluidBrickResidencyAllocationPlan {
  const volume = (value: readonly [number, number, number], label: string) => {
    value.forEach((component) => {
      if (!Number.isSafeInteger(component) || component < 1) throw new RangeError(`${label} dimensions must be positive integers`);
    });
    return value[0] * value[1] * value[2];
  };
  const brickCapacity = volume(brickDimensions, "Brick");
  const tileCapacity = volume(tileDimensions, "Topology tile");
  if (!Number.isSafeInteger(leafCapacity) || leafCapacity < 1) throw new RangeError("Leaf capacity must be a positive integer");
  if (surfaceCandidatesOnly && explicitLeafMapping) {
    throw new RangeError("Surface-candidate-only residency requires implicit brick/leaf identity");
  }
  const sparseKeyPools = surfaceCandidatesOnly && surfaceCandidateBrickCapacity !== undefined;
  const sparseCapacity = (value: number | undefined, logical: number, label: string) => {
    if (value === undefined) return logical;
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} capacity must be a positive integer`);
    return Math.min(logical, value);
  };
  if (surfaceCandidateTileCapacity !== undefined && !sparseKeyPools) {
    throw new RangeError("Sparse tile capacity requires a sparse surface-candidate brick capacity");
  }
  const brickStateCapacity = sparseKeyPools
    ? sparseCapacity(surfaceCandidateBrickCapacity, brickCapacity, "Surface-candidate brick") : brickCapacity;
  const tileStateCapacity = sparseKeyPools
    ? sparseCapacity(surfaceCandidateTileCapacity ?? brickStateCapacity * 27, tileCapacity,
      "Surface-candidate topology tile") : tileCapacity;
  // Sparse records store key-plus-one and lifecycle state. Dense compatibility
  // stores only the state because its array index is the logical key.
  const stateBytes = brickStateCapacity * (sparseKeyPools ? 8 : FLUID_BRICK_STATE_STRIDE_BYTES);
  const worklistBytes = (FLUID_BRICK_WORKLIST_HEADER_WORDS + brickStateCapacity * 4) * 4;
  const tileWorklistBytes = (FLUID_TILE_WORKLIST_HEADER_WORDS + tileStateCapacity * 2) * 4;
  const tileStateBytes = tileStateCapacity * (sparseKeyPools ? 8 : 4);
  const leafIndexBytes = explicitLeafMapping ? brickCapacity * 4 : 4;
  const leafStateBytes = surfaceCandidatesOnly ? 4 : leafCapacity * 4;
  const parameterBytes = 64;
  const transactionalBytes = stateBytes + worklistBytes + tileWorklistBytes + tileStateBytes + 16;
  const allocatedBytes = stateBytes + worklistBytes + tileWorklistBytes + tileStateBytes
    + leafIndexBytes + leafStateBytes + parameterBytes + transactionalBytes;
  const denseSchedulerBytes = 2 * ((brickCapacity * 4
    + (FLUID_BRICK_WORKLIST_HEADER_WORDS + brickCapacity * 4) * 4)
    + (tileCapacity * 4 + (FLUID_TILE_WORKLIST_HEADER_WORDS + tileCapacity * 2) * 4));
  const allocatedSchedulerBytes = 2 * (stateBytes + worklistBytes + tileStateBytes + tileWorklistBytes);
  const schedulerByteDelta = sparseKeyPools ? allocatedSchedulerBytes - denseSchedulerBytes : 0;
  return {
    brickCapacity,
    tileCapacity,
    leafCapacity,
    identityMapping: explicitLeafMapping ? "explicit" : "implicit",
    surfaceCandidatesOnly,
    sparseKeyPools,
    brickStateCapacity,
    tileStateCapacity,
    stateBytes,
    worklistBytes,
    tileWorklistBytes,
    tileStateBytes,
    leafIndexBytes,
    leafStateBytes,
    parameterBytes,
    transactionalBytes,
    allocatedBytes,
    savedIdentityBytes: explicitLeafMapping ? 0 : brickCapacity * 4 - 4,
    savedLeafStateBytes: surfaceCandidatesOnly ? leafCapacity * 4 - 4 : 0,
    savedSchedulerBytes: Math.max(0, -schedulerByteDelta),
    schedulerByteDelta,
  };
}

export interface FluidBrickResidencyStats {
  resident: number;
  core: number;
  halo: number;
  activated: number;
  retired: number;
  generation: number;
  capacity: number;
}

export interface SurfaceCandidateResidencyPoolPlan {
  readonly brickCapacity: number;
  readonly tileCapacity: number;
  readonly logicalBrickCount: number;
  readonly logicalTileCount: number;
  readonly bandBrickLayers: number;
  readonly bandTileLayers: number;
}

/**
 * Capacity bound for the compact surface producer's sparse scheduler.
 *
 * This is intentionally an interface-area budget rather than a logical-box
 * allocation. Highly folded surfaces may exhaust it; the transactional GPU
 * publisher then retains the last complete generation and reports overflow.
 * The active analytic t=0 tile count is an explicit lower bound so cold-start
 * authority cannot be truncated by the steady-state estimate.
 */
export function planSurfaceCandidateResidencyPools(
  brickDimensions: readonly [number, number, number],
  tileDimensions: readonly [number, number, number],
  brickSize: number,
  haloCells: number,
  producerRowCapacity: number,
  minimumTileCapacity = 1,
): SurfaceCandidateResidencyPoolPlan {
  const checkedVolume = (dims: readonly [number, number, number], label: string) => {
    if (!dims.every((value) => Number.isSafeInteger(value) && value > 0)) {
      throw new RangeError(`${label} dimensions must be positive integers`);
    }
    return dims[0] * dims[1] * dims[2];
  };
  if (!Number.isSafeInteger(brickSize) || brickSize < 1 || !Number.isFinite(haloCells) || haloCells < 0
    || !Number.isSafeInteger(producerRowCapacity) || producerRowCapacity < 1
    || !Number.isSafeInteger(minimumTileCapacity) || minimumTileCapacity < 0) {
    throw new RangeError("Surface-candidate residency pool inputs are invalid");
  }
  const logicalBrickCount = checkedVolume(brickDimensions, "Brick");
  const logicalTileCount = checkedVolume(tileDimensions, "Topology tile");
  const area = (dims: readonly [number, number, number]) =>
    dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2];
  // Two-sided swept support plus one complete insertion/retirement generation.
  const bandBrickLayers = Math.max(2, Math.ceil(haloCells / brickSize) * 2 + 1);
  const topologyTileBricks = Math.max(1, Math.ceil(brickDimensions[0] / tileDimensions[0]));
  const bandTileLayers = Math.max(3, Math.ceil(bandBrickLayers / topologyTileBricks) + 2);
  const brickCapacity = Math.max(1, Math.min(logicalBrickCount, producerRowCapacity,
    Math.ceil(area(brickDimensions) * bandBrickLayers)));
  const tileCapacity = Math.max(1, Math.min(logicalTileCount,
    Math.max(minimumTileCapacity, Math.ceil(area(tileDimensions) * bandTileLayers))));
  return { brickCapacity, tileCapacity, logicalBrickCount, logicalTileCount, bandBrickLayers, bandTileLayers };
}

export interface CPUFluidBrickState {
  flags: number;
  dryFrames: number;
}

export interface CPUFluidBrickClassificationOptions {
  haloPhi: number;
  retireAfterFrames: number;
  includeLiquidInterior?: boolean;
}

/** Deterministic CPU mirror of the per-brick lifecycle used by unit tests. */
export function classifyCPUFluidBrick(
  minimumPhi: number,
  previous: CPUFluidBrickState = { flags: 0, dryFrames: 0 },
  options: CPUFluidBrickClassificationOptions,
  maximumPhi = minimumPhi,
): CPUFluidBrickState {
  if (!Number.isFinite(minimumPhi) || !Number.isFinite(maximumPhi) || maximumPhi < minimumPhi) {
    throw new RangeError("Brick signed-distance range must be finite and ordered");
  }
  if (!(options.haloPhi >= 0) || !Number.isFinite(options.haloPhi)) throw new RangeError("Brick halo must be finite and non-negative");
  if (!Number.isInteger(options.retireAfterFrames) || options.retireAfterFrames < 0 || options.retireAfterFrames > 0xffff) {
    throw new RangeError("Brick retirement window must be a uint16");
  }
  // A sparse surface band must not retain every negative (deep-liquid)
  // brick. Core pages actually straddle phi=0; halo pages have at least one
  // sample within the requested absolute-distance support.
  const core = minimumPhi <= 0 && maximumPhi >= 0;
  const minimumAbsolutePhi = core ? 0 : Math.min(Math.abs(minimumPhi), Math.abs(maximumPhi));
  const desired = minimumAbsolutePhi < options.haloPhi
    || (options.includeLiquidInterior === true && minimumPhi < 0);
  const wasResident = (previous.flags & FLUID_BRICK_RESIDENT) !== 0;
  const dryFrames = desired ? 0 : Math.min(0xffff, previous.dryFrames + 1);
  const resident = desired || (wasResident && dryFrames <= options.retireAfterFrames);
  const flags = (resident ? FLUID_BRICK_RESIDENT : 0)
    | (core ? FLUID_BRICK_CORE : 0)
    | (resident && !core ? FLUID_BRICK_HALO : 0)
    | (resident && !wasResident ? FLUID_BRICK_ACTIVATED : 0);
  return { flags, dryFrames };
}

export const fluidBrickResidencyShader = /* wgsl */ `
struct Params {
  dimsBrick: vec4u,
  brickDimsCapacity: vec4u,
  settings: vec4f,
  // Bricks per topology-tile axis, then the tile lattice dimensions.
  tiling: vec4u,
}
@group(0) @binding(0) var levelSet: texture_3d<f32>;
// state: low 8 bits flags, high 16 bits consecutive dry publications. Atomic
// so the expansion dispatch can read neighbor flags while writing its own.
@group(0) @binding(1) var<storage, read_write> states: array<atomic<u32>>;
// Header words 0..15, active (solver index, leaf index) pairs, then retired pairs.
@group(0) @binding(2) var<storage, read_write> worklist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> leafIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> leafStates: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var velocity: texture_3d<f32>;
// Header words 0..15, active tile indices, then retired tile indices. Tiles
// are max(brick, maximumLeaf)-sized brick groups; the octree topology rebuild
// consumes this list so a partial rebuild can never split a pressure leaf.
@group(0) @binding(7) var<storage, read_write> tileWorklist: array<atomic<u32>>;
// Persistent topology-tile activity. A tile can be active solely because it
// grades a neighboring surface tile, so brick WAS_RESIDENT bits cannot retire
// it reliably after that neighbor moves away.
@group(0) @binding(8) var<storage, read_write> tileStates: array<atomic<u32>>;

// A single INVALID word denotes the production identity mapping. This avoids
// persisting one redundant leaf index for every logical brick. Non-identity
// compatibility mappings retain the original full array.
const IMPLICIT_IDENTITY: u32 = 0xffffffffu;

const RESIDENT: u32 = 1u;
const CORE: u32 = 2u;
const HALO: u32 = 4u;
const ACTIVATED: u32 = 8u;
const WAS_RESIDENT: u32 = 32u;
const HEADER_WORDS: u32 = 16u;

fn brickCoordinate(index: u32) -> vec3u {
  let bx = params.brickDimsCapacity.x;
  let by = params.brickDimsCapacity.y;
  return vec3u(index % bx, (index / bx) % by, index / (bx * by));
}

fn leafIndexFor(brickIndex: u32) -> u32 {
  if (arrayLength(&leafIndices) == 1u && leafIndices[0] == IMPLICIT_IDENTITY) { return brickIndex; }
  return leafIndices[brickIndex];
}

fn tiledDispatch(blocks: u32) -> vec2u {
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  return vec2u(x, y);
}

fn brickPhiRange(origin: vec3u) -> vec2f {
  let brickSize = params.dimsBrick.w;
  var minimumPhi = 3.402823e38;
  var maximumPhi = -3.402823e38;
  for (var z = 0u; z < brickSize; z += 1u) {
    for (var y = 0u; y < brickSize; y += 1u) {
      for (var x = 0u; x < brickSize; x += 1u) {
        let cell = origin + vec3u(x, y, z);
        if (all(cell < params.dimsBrick.xyz)) {
          let samplePhi = textureLoad(levelSet, vec3i(cell), 0).x;
          minimumPhi = min(minimumPhi, samplePhi);
          maximumPhi = max(maximumPhi, samplePhi);
        }
      }
    }
  }
  return vec2f(minimumPhi, maximumPhi);
}

fn brickMaximumSpeed(origin: vec3u) -> f32 {
  let brickSize = params.dimsBrick.w;
  var maximumSpeed = 0.0;
  for (var z = 0u; z < brickSize; z += 1u) {
    for (var y = 0u; y < brickSize; y += 1u) {
      for (var x = 0u; x < brickSize; x += 1u) {
        let cell = origin + vec3u(x, y, z);
        if (all(cell < params.dimsBrick.xyz)) {
          maximumSpeed = max(maximumSpeed, length(textureLoad(velocity, vec3i(cell), 0).xyz));
        }
      }
    }
  }
  return maximumSpeed;
}

fn emitWorklistFor(brickIndex: u32, flags: u32, wasResident: bool) {
  let capacity = params.brickDimsCapacity.w;
  let leafIndex = leafIndexFor(brickIndex);
  if (leafIndex < arrayLength(&leafStates)) { leafStates[leafIndex] = flags & 15u; }
  if ((flags & RESIDENT) != 0u) {
    let slot = atomicAdd(&worklist[0], 1u);
    if (slot < capacity) {
      let base = HEADER_WORDS + slot * 2u;
      atomicStore(&worklist[base], brickIndex);
      atomicStore(&worklist[base + 1u], leafIndex);
    }
    if ((flags & CORE) != 0u) { atomicAdd(&worklist[8], 1u); }
    else { atomicAdd(&worklist[9], 1u); }
    if ((flags & ACTIVATED) != 0u) { atomicAdd(&worklist[10], 1u); }
  } else if (wasResident) {
    let slot = atomicAdd(&worklist[4], 1u);
    if (slot < capacity) {
      let base = HEADER_WORDS + capacity * 2u + slot * 2u;
      atomicStore(&worklist[base], brickIndex);
      atomicStore(&worklist[base + 1u], leafIndex);
    }
    atomicAdd(&worklist[11], 1u);
  }
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  let capacity = params.brickDimsCapacity.w;
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states)) { return; }
  let brickSize = params.dimsBrick.w;
  let origin = brickCoordinate(brickIndex) * brickSize;
  let range = brickPhiRange(origin);
  let minimumPhi = range.x;
  let maximumPhi = range.y;
  let previous = atomicLoad(&states[brickIndex]);
  let previousFlags = previous & 0xffu;
  let wasResident = (previousFlags & RESIDENT) != 0u;
  let core = minimumPhi <= 0.0 && maximumPhi >= 0.0;
  let minimumAbsolutePhi = select(min(abs(minimumPhi), abs(maximumPhi)), 0.0, core);
  let desired = minimumAbsolutePhi < params.settings.x
    || (params.settings.w > 0.5 && minimumPhi < 0.0);
  var dryFrames = select(min(0xffffu, (previous >> 16u) + 1u), 0u, desired);
  let retireAfter = u32(params.settings.y);
  let resident = desired || (wasResident && dryFrames <= retireAfter);
  var flags = select(0u, RESIDENT, resident)
    | select(0u, CORE, core)
    | select(0u, HALO, resident && !core)
    | select(0u, ACTIVATED, resident && !wasResident)
    | select(0u, WAS_RESIDENT, wasResident);
  atomicStore(&states[brickIndex], flags | (dryFrames << 16u));
  emitWorklistFor(brickIndex, flags, wasResident);
}

// Split lifecycle used when velocity-swept pre-activation is on. The phi
// support band widens to cover the material swept this step, and worklist
// emission moves behind the downstream-expansion dispatch.
@compute @workgroup_size(64)
fn classifySwept(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  let capacity = params.brickDimsCapacity.w;
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states)) { return; }
  let brickSize = params.dimsBrick.w;
  let origin = brickCoordinate(brickIndex) * brickSize;
  let range = brickPhiRange(origin);
  let core = range.x <= 0.0 && range.y >= 0.0;
  let minimumAbsolutePhi = select(min(abs(range.x), abs(range.y)), 0.0, core);
  // Support = max(phi halo band, |v| dt with a 1.5 safety factor) so a fast
  // front pre-activates the bricks it will sweep before phi arrives.
  let sweptSupport = brickMaximumSpeed(origin) * params.settings.z * 1.5;
  let desired = minimumAbsolutePhi < max(params.settings.x, sweptSupport)
    || (params.settings.w > 0.5 && range.x < 0.0);
  let previous = atomicLoad(&states[brickIndex]);
  let wasResident = ((previous & 0xffu) & RESIDENT) != 0u;
  var dryFrames = select(min(0xffffu, (previous >> 16u) + 1u), 0u, desired);
  let resident = desired || (wasResident && dryFrames <= u32(params.settings.y));
  var flags = select(0u, RESIDENT, resident)
    | select(0u, CORE, core)
    | select(0u, HALO, resident && !core)
    | select(0u, ACTIVATED, resident && !wasResident)
    | select(0u, WAS_RESIDENT, wasResident);
  atomicStore(&states[brickIndex], flags | (dryFrames << 16u));
}

// Pull-model neighbor activation: a brick outside the band becomes resident
// when a face-adjacent core brick's velocity at the shared face points into
// it, so a moving front never advects into an unscheduled brick. Reads the
// flags written by classifySwept in the previous dispatch of the same pass.
@compute @workgroup_size(64)
fn expandDownstream(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  let capacity = params.brickDimsCapacity.w;
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states)) { return; }
  let state = atomicLoad(&states[brickIndex]);
  if ((state & RESIDENT) != 0u) { return; }
  let brickSize = params.dimsBrick.w;
  let brick = vec3i(brickCoordinate(brickIndex));
  let brickDims = vec3i(params.brickDimsCapacity.xyz);
  var forced = false;
  for (var axis = 0; axis < 3 && !forced; axis += 1) {
    for (var sign = -1; sign <= 1 && !forced; sign += 2) {
      var direction = vec3i(0);
      direction[axis] = sign;
      let neighbor = brick + direction;
      if (any(neighbor < vec3i(0)) || any(neighbor >= brickDims)) { continue; }
      let neighborIndex = u32(neighbor.x) + params.brickDimsCapacity.x * (u32(neighbor.y) + params.brickDimsCapacity.y * u32(neighbor.z));
      if (neighborIndex >= arrayLength(&states)) { continue; }
      if ((atomicLoad(&states[neighborIndex]) & CORE) == 0u) { continue; }
      // Scan the complete shared face. A single centre texel misses localized
      // jets and would activate the downstream brick one substep too late once
      // velocity output is dispatched only over resident bricks.
      var neighborBase = vec3u(neighbor) * brickSize;
      let tangentA = (axis + 1) % 3;
      let tangentB = (axis + 2) % 3;
      var maximumInwardSpeed = 0.0;
      for (var a = 0u; a < brickSize; a += 1u) {
        for (var b = 0u; b < brickSize; b += 1u) {
          var faceCell = neighborBase;
          faceCell[axis] = select(neighborBase[axis] + brickSize - 1u, neighborBase[axis], sign > 0);
          faceCell[tangentA] = neighborBase[tangentA] + a;
          faceCell[tangentB] = neighborBase[tangentB] + b;
          if (any(faceCell >= params.dimsBrick.xyz)) { continue; }
          let flow = textureLoad(velocity, vec3i(faceCell), 0).xyz;
          maximumInwardSpeed = max(maximumInwardSpeed, flow[axis] * f32(-sign));
        }
      }
      forced = maximumInwardSpeed > 1e-4;
    }
  }
  if (!forced) { return; }
  let wasResident = (state & WAS_RESIDENT) != 0u;
  let flags = (state & (CORE | WAS_RESIDENT)) | RESIDENT | HALO
    | select(0u, ACTIVATED, !wasResident);
  atomicStore(&states[brickIndex], flags);
}

@compute @workgroup_size(64)
fn emitWorklist(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  let capacity = params.brickDimsCapacity.w;
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states)) { return; }
  let state = atomicLoad(&states[brickIndex]);
  emitWorklistFor(brickIndex, state & 0xffu, (state & WAS_RESIDENT) != 0u);
}

fn tileHasResident(tile: vec3i) -> bool {
  let tileDims = vec3i(params.tiling.yzw);
  if (any(tile < vec3i(0)) || any(tile >= tileDims)) { return false; }
  let factor = params.tiling.x;
  for (var z = 0u; z < factor; z += 1u) {
    for (var y = 0u; y < factor; y += 1u) {
      for (var x = 0u; x < factor; x += 1u) {
        let brick = vec3u(tile) * factor + vec3u(x, y, z);
        if (any(brick >= params.brickDimsCapacity.xyz)) { continue; }
        let brickIndex = brick.x + params.brickDimsCapacity.x * (brick.y + params.brickDimsCapacity.y * brick.z);
        if (brickIndex < arrayLength(&states) && (atomicLoad(&states[brickIndex]) & RESIDENT) != 0u) { return true; }
      }
    }
  }
  return false;
}

// One thread per topology tile scans the 3x3x3 tile neighborhood. The full
// 2:1 grading chain travels less than one maximum-leaf tile, so this dilation
// replaces a maxLeaf-1 phi-residency halo without making the atlas retain that
// much dense air/liquid support. Persistent tile state emits a retired rebuild
// even for tiles that were active only because of this grading dilation.
@compute @workgroup_size(64)
fn emitTopologyTiles(@builtin(global_invocation_id) gid: vec3u) {
  let tileCapacity = params.tiling.y * params.tiling.z * params.tiling.w;
  let tileIndex = gid.x;
  if (tileIndex >= tileCapacity) { return; }
  let factor = params.tiling.x;
  let tile = vec3u(
    tileIndex % params.tiling.y,
    (tileIndex / params.tiling.y) % params.tiling.z,
    tileIndex / (params.tiling.y * params.tiling.z)
  );
  var anyResident = false;
  for (var dz = -1; dz <= 1 && !anyResident; dz += 1) {
    for (var dy = -1; dy <= 1 && !anyResident; dy += 1) {
      for (var dx = -1; dx <= 1 && !anyResident; dx += 1) {
        anyResident = tileHasResident(vec3i(tile) + vec3i(dx, dy, dz));
      }
    }
  }
  let wasActive = atomicExchange(&tileStates[tileIndex], select(0u, 1u, anyResident)) != 0u;
  if (anyResident) {
    let slot = atomicAdd(&tileWorklist[0], 1u);
    if (slot < tileCapacity) { atomicStore(&tileWorklist[HEADER_WORDS + slot], tileIndex); }
  } else if (wasActive) {
    let slot = atomicAdd(&tileWorklist[4], 1u);
    if (slot < tileCapacity) { atomicStore(&tileWorklist[HEADER_WORDS + tileCapacity + slot], tileIndex); }
  }
}

@compute @workgroup_size(1)
fn finalize() {
  let capacity = params.brickDimsCapacity.w;
  let resident = min(atomicLoad(&worklist[0]), capacity);
  let retired = min(atomicLoad(&worklist[4]), capacity);
  let voxelsPerBrick = params.dimsBrick.w * params.dimsBrick.w * params.dimsBrick.w;
  let activeDispatch = tiledDispatch((resident * voxelsPerBrick + 255u) / 256u);
  atomicStore(&worklist[1], activeDispatch.x);
  atomicStore(&worklist[2], activeDispatch.y);
  atomicStore(&worklist[3], 1u);
  let surfaceDispatch = tiledDispatch((resident * voxelsPerBrick + 63u) / 64u);
  atomicStore(&worklist[12], surfaceDispatch.x);
  atomicStore(&worklist[13], surfaceDispatch.y);
  atomicStore(&worklist[14], 1u);
  let retiredDispatch = tiledDispatch((retired * voxelsPerBrick + 255u) / 256u);
  atomicStore(&worklist[5], retiredDispatch.x);
  atomicStore(&worklist[6], retiredDispatch.y);
  atomicStore(&worklist[7], 1u);
  atomicAdd(&worklist[15], 1u);
  // Topology work dispatches 4x4x4-cell workgroups over tiles: an 8-cubed
  // brick is eight blocks (a 4-cell brick is one) and a tile is factor-cubed
  // bricks, so one tile spans factor^3 times the per-brick block count.
  let tileCapacity = params.tiling.y * params.tiling.z * params.tiling.w;
  let activeTiles = min(atomicLoad(&tileWorklist[0]), tileCapacity);
  let retiredTiles = min(atomicLoad(&tileWorklist[4]), tileCapacity);
  let brickGroups = select(1u, 8u, params.dimsBrick.w == 8u);
  let groupsPerTile = params.tiling.x * params.tiling.x * params.tiling.x * brickGroups;
  let tileDispatch = tiledDispatch(activeTiles * groupsPerTile);
  atomicStore(&tileWorklist[1], tileDispatch.x);
  atomicStore(&tileWorklist[2], tileDispatch.y);
  atomicStore(&tileWorklist[3], 1u);
  let retiredTileDispatch = tiledDispatch(retiredTiles * groupsPerTile);
  atomicStore(&tileWorklist[5], retiredTileDispatch.x);
  atomicStore(&tileWorklist[6], retiredTileDispatch.y);
  atomicStore(&tileWorklist[7], 1u);
  // Refinement and balancing only visit possible origins of splittable leaves.
  // Every such dyadic origin is even-aligned, so one 4^3 workgroup spans an
  // 8^3 cell region and needs one eighth as many invocations as cell passes.
  let candidateGroupsPerTile = max(1u, groupsPerTile / 8u);
  let activeCandidateDispatch = tiledDispatch(activeTiles * candidateGroupsPerTile);
  atomicStore(&tileWorklist[8], activeCandidateDispatch.x);
  atomicStore(&tileWorklist[9], activeCandidateDispatch.y);
  atomicStore(&tileWorklist[10], 1u);
  let retiredCandidateDispatch = tiledDispatch(retiredTiles * candidateGroupsPerTile);
  atomicStore(&tileWorklist[12], retiredCandidateDispatch.x);
  atomicStore(&tileWorklist[13], retiredCandidateDispatch.y);
  atomicStore(&tileWorklist[14], 1u);
  atomicStore(&tileWorklist[15], atomicLoad(&worklist[15]));
}
`;

/**
 * Compact surface candidates -> legacy brick/tile scheduler ABI.
 *
 * All mutation is staged in candidate buffers. `commitSurfaceCandidates`
 * copies them into the stable publication only when the producer generation
 * is complete, newer, in-bounds, and fault-free. A valid zero-count producer
 * consequently commits an empty/retired generation, while failed zero-count
 * producers leave the previous worklists and both state tables untouched.
 */
export const surfaceCandidateResidencyShader = /* wgsl */ `
struct Params { dimsBrick:vec4u, brickDimsCapacity:vec4u, settings:vec4f, tiling:vec4u }
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
@group(0) @binding(0) var<storage,read> publishedStates:array<u32>;
@group(0) @binding(1) var<storage,read_write> states:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> worklist:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> publishedTileStates:array<u32>;
@group(0) @binding(4) var<storage,read_write> tileStates:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> tileWorklist:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read> leaves:array<SurfaceLeaf>;
@group(0) @binding(7) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(8) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(9) var<storage,read_write> transaction:array<atomic<u32>>;
@group(0) @binding(10) var<uniform> params:Params;
const RESIDENT=1u;const CORE=2u;const HALO=4u;const ACTIVATED=8u;const LIVE=32u;const WAS_RESIDENT=32u;const HEADER=16u;const INVALID=0xffffffffu;
fn origin(p:u32)->vec3u{return vec3u(p&1023u,(p>>10u)&1023u,(p>>20u)&1023u);}
fn dispatch2(n:u32)->vec2u{let x=min(n,65535u);return vec2u(x,select(1u,(n+x-1u)/x,x>0u));}
fn producerAccepted()->bool{return candidateControl[5]==1u&&candidateControl[6]==0u&&candidateControl[7]==arrayLength(&candidates)&&candidateControl[4]>atomicLoad(&tileWorklist[15]);}
@compute @workgroup_size(64) fn beginSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(!producerAccepted()||i>=params.brickDimsCapacity.w){return;}let old=publishedStates[i];let was=(old&RESIDENT)!=0u;atomicStore(&states[i],select(0u,WAS_RESIDENT,was)|(min(0xffffu,(old>>16u)+1u)<<16u));}
@compute @workgroup_size(64) fn beginPressureTiles(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let cap=params.tiling.y*params.tiling.z*params.tiling.w;if(!producerAccepted()||i>=cap){return;}atomicStore(&tileStates[i],0u);}
// Aanjaneya et al. (2017), Section 4.2: power faces can join octree edge
// neighbours, so the active/ghost pressure pyramid needs the complete local
// face-and-edge 1-ring. This pressure support is deliberately independent of
// Section 5's narrower fine-phi brick set.
@compute @workgroup_size(64) fn markPressureTiles(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(!producerAccepted()||row>=arrayLength(&leaves)){return;}let leaf=leaves[row];if((leaf.flags&LIVE)==0u||leaf.size==0u){return;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);if(any(a>=params.dimsBrick.xyz)){atomicStore(&transaction[0],9u);return;}let tileCells=params.dimsBrick.w*params.tiling.x;let td=vec3i(params.tiling.yzw);let first=vec3i(a/tileCells);let last=vec3i(min(params.dimsBrick.xyz-vec3u(1u),a+vec3u(leaf.size-1u))/tileCells);for(var tz=first.z-1;tz<=last.z+1;tz++){for(var ty=first.y-1;ty<=last.y+1;ty++){for(var tx=first.x-1;tx<=last.x+1;tx++){let q=vec3i(tx,ty,tz);if(any(q<vec3i(0))||any(q>=td)){continue;}let key=u32(q.x)+params.tiling.y*(u32(q.y)+params.tiling.z*u32(q.z));atomicStore(&tileStates[key],1u);}}}}
@compute @workgroup_size(64) fn markSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(!producerAccepted()||i>=candidateControl[0]){return;}if(i>=arrayLength(&candidates)){atomicStore(&transaction[0],1u);return;}let c=candidates[i];if(c.row>=arrayLength(&leaves)){atomicStore(&transaction[0],2u);return;}let leaf=leaves[c.row];if(leaf.size==0u){atomicStore(&transaction[0],3u);return;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);if(any(a>=params.dimsBrick.xyz)){atomicStore(&transaction[0],4u);return;}let first=a/params.dimsBrick.w;let last=min(params.brickDimsCapacity.xyz-vec3u(1u),(a+vec3u(leaf.size-1u))/params.dimsBrick.w);let bits=select(HALO,CORE,(c.flags&CORE)!=0u);for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){let b=x+params.brickDimsCapacity.x*(y+params.brickDimsCapacity.y*z);atomicOr(&states[b],bits);}}}}
@compute @workgroup_size(64) fn resolveSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let cap=params.brickDimsCapacity.w;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||i>=cap){return;}let marked=atomicLoad(&states[i]);let desired=(marked&(CORE|HALO))!=0u;let was=(marked&WAS_RESIDENT)!=0u;let dry=select(marked>>16u,0u,desired);let persistent=params.settings.w>0.5;let resident=select(desired||(was&&dry<=u32(params.settings.y)),was||desired,persistent);let core=(marked&CORE)!=0u;let flags=select(0u,RESIDENT,resident)|select(0u,CORE,core)|select(0u,HALO,resident&&!core)|select(0u,ACTIVATED,resident&&!was)|select(0u,WAS_RESIDENT,was);atomicStore(&states[i],flags|(dry<<16u));if(resident){let slot=atomicAdd(&worklist[0],1u);if(slot<cap){atomicStore(&worklist[HEADER+slot*2u],i);atomicStore(&worklist[HEADER+slot*2u+1u],i);}if(core){atomicAdd(&worklist[8],1u);}else{atomicAdd(&worklist[9],1u);}if(!was){atomicAdd(&worklist[10],1u);}}else if(was){let slot=atomicAdd(&worklist[4],1u);if(slot<cap){atomicStore(&worklist[HEADER+cap*2u+slot*2u],i);atomicStore(&worklist[HEADER+cap*2u+slot*2u+1u],i);}atomicAdd(&worklist[11],1u);}}
fn tileResident(q:vec3i)->bool{let td=vec3i(params.tiling.yzw);if(any(q<vec3i(0))||any(q>=td)){return false;}let f=params.tiling.x;for(var z=0u;z<f;z++){for(var y=0u;y<f;y++){for(var x=0u;x<f;x++){let b=vec3u(q)*f+vec3u(x,y,z);if(any(b>=params.brickDimsCapacity.xyz)){continue;}let i=b.x+params.brickDimsCapacity.x*(b.y+params.brickDimsCapacity.y*b.z);if((atomicLoad(&states[i])&RESIDENT)!=0u){return true;}}}}return false;}
@compute @workgroup_size(64) fn emitTiles(@builtin(global_invocation_id) gid:vec3u){let cap=params.tiling.y*params.tiling.z*params.tiling.w;let i=gid.x;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||i>=cap){return;}let q=vec3i(vec3u(i%params.tiling.y,(i/params.tiling.y)%params.tiling.z,i/(params.tiling.y*params.tiling.z)));var live=atomicLoad(&tileStates[i])!=0u;for(var z=-1;z<=1&&!live;z++){for(var y=-1;y<=1&&!live;y++){for(var x=-1;x<=1&&!live;x++){live=tileResident(q+vec3i(x,y,z));}}}let was=publishedTileStates[i]!=0u;atomicStore(&tileStates[i],select(0u,1u,live));if(live){let slot=atomicAdd(&tileWorklist[0],1u);if(slot<cap){atomicStore(&tileWorklist[HEADER+slot],i);}}else if(was){let slot=atomicAdd(&tileWorklist[4],1u);if(slot<cap){atomicStore(&tileWorklist[HEADER+cap+slot],i);}}}
@compute @workgroup_size(1) fn finalize(){if(!producerAccepted()||atomicLoad(&transaction[0])!=0u){return;}let cap=params.brickDimsCapacity.w;let tc=params.tiling.y*params.tiling.z*params.tiling.w;let rawResident=atomicLoad(&worklist[0]);let rawRetired=atomicLoad(&worklist[4]);let rawActiveTiles=atomicLoad(&tileWorklist[0]);let rawRetiredTiles=atomicLoad(&tileWorklist[4]);if(rawResident>cap||rawRetired>cap||rawActiveTiles>tc||rawRetiredTiles>tc){atomicStore(&transaction[0],5u);return;}let voxels=params.dimsBrick.w*params.dimsBrick.w*params.dimsBrick.w;let a=dispatch2((rawResident*voxels+255u)/256u);atomicStore(&worklist[1],a.x);atomicStore(&worklist[2],a.y);atomicStore(&worklist[3],1u);let s=dispatch2((rawResident*voxels+63u)/64u);atomicStore(&worklist[12],s.x);atomicStore(&worklist[13],s.y);atomicStore(&worklist[14],1u);let r=dispatch2((rawRetired*voxels+255u)/256u);atomicStore(&worklist[5],r.x);atomicStore(&worklist[6],r.y);atomicStore(&worklist[7],1u);atomicStore(&worklist[15],candidateControl[4]);let bg=select(1u,8u,params.dimsBrick.w==8u);let g=params.tiling.x*params.tiling.x*params.tiling.x*bg;let ad=dispatch2(rawActiveTiles*g);atomicStore(&tileWorklist[1],ad.x);atomicStore(&tileWorklist[2],ad.y);atomicStore(&tileWorklist[3],1u);let rd=dispatch2(rawRetiredTiles*g);atomicStore(&tileWorklist[5],rd.x);atomicStore(&tileWorklist[6],rd.y);atomicStore(&tileWorklist[7],1u);let cg=max(1u,g/8u);let ac=dispatch2(rawActiveTiles*cg);atomicStore(&tileWorklist[8],ac.x);atomicStore(&tileWorklist[9],ac.y);atomicStore(&tileWorklist[10],1u);let rc=dispatch2(rawRetiredTiles*cg);atomicStore(&tileWorklist[12],rc.x);atomicStore(&tileWorklist[13],rc.y);atomicStore(&tileWorklist[14],1u);atomicStore(&tileWorklist[15],candidateControl[4]);atomicStore(&transaction[1],1u);}
`;

/**
 * Producer-bounded counterpart of `surfaceCandidateResidencyShader`.
 *
 * Candidate brick and topology-tile keys live in open-addressed physical
 * pools. A key is stored as logical+1, leaving zero as the empty sentinel.
 * Every stage writes generation B and the final commit copies it over A only
 * after all capacity/status checks pass. Thus a saturated pool cannot publish
 * a partial topology or silently drop a retirement.
 */
export const sparseSurfaceCandidateResidencyShader = /* wgsl */ `
struct Params { dimsBrick:vec4u, brickDimsCapacity:vec4u, settings:vec4f, tiling:vec4u }
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
@group(0) @binding(0) var<storage,read> publishedStates:array<u32>;
@group(0) @binding(1) var<storage,read_write> states:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> worklist:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> publishedTileStates:array<u32>;
@group(0) @binding(4) var<storage,read_write> tileStates:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> tileWorklist:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read> leaves:array<SurfaceLeaf>;
@group(0) @binding(7) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(8) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(9) var<storage,read_write> transaction:array<atomic<u32>>;
@group(0) @binding(10) var<uniform> params:Params;
const RESIDENT=1u;const CORE=2u;const HALO=4u;const ACTIVATED=8u;const LIVE=32u;const WAS_RESIDENT=32u;const HEADER=16u;const INVALID=0xffffffffu;
fn origin(p:u32)->vec3u{return vec3u(p&1023u,(p>>10u)&1023u,(p>>20u)&1023u);}
fn dispatch2(n:u32)->vec2u{let x=min(n,65535u);return vec2u(x,select(1u,(n+x-1u)/x,x>0u));}
fn producerAccepted()->bool{return candidateControl[5]==1u&&candidateControl[6]==0u&&candidateControl[7]==arrayLength(&candidates)&&candidateControl[4]>atomicLoad(&tileWorklist[15]);}
fn hashKey(key:u32)->u32{var x=key*747796405u+2891336453u;x=((x>>((x>>28u)+4u))^x)*277803737u;return (x>>22u)^x;}
fn brickSlots()->u32{return arrayLength(&states)/2u;}
fn tileSlots()->u32{return arrayLength(&tileStates)/2u;}
fn worklistCapacity()->u32{return (arrayLength(&worklist)-HEADER)/4u;}
fn tileWorklistCapacity()->u32{return (arrayLength(&tileWorklist)-HEADER)/2u;}
fn claimBrick(logical:u32)->u32{let encoded=logical+1u;let cap=brickSlots();if(encoded==0u||encoded==INVALID||cap==0u){return INVALID;}let start=hashKey(logical)%cap;for(var attempt=0u;attempt<cap;attempt++){var claimSlot=INVALID;var expected=0u;for(var probe=0u;probe<cap;probe++){let slot=(start+probe)%cap;let key=atomicLoad(&states[slot*2u]);if(key==encoded){return slot;}if(key==INVALID&&claimSlot==INVALID){claimSlot=slot;expected=INVALID;}if(key==0u){if(claimSlot==INVALID){claimSlot=slot;expected=0u;}break;}}if(claimSlot==INVALID){break;}loop{let result=atomicCompareExchangeWeak(&states[claimSlot*2u],expected,encoded);if(result.exchanged){return claimSlot;}if(result.old_value!=expected){break;}}}atomicStore(&transaction[0],6u);return INVALID;}
fn claimTile(logical:u32)->u32{let encoded=logical+1u;let cap=tileSlots();if(encoded==0u||encoded==INVALID||cap==0u){return INVALID;}let start=hashKey(logical)%cap;for(var attempt=0u;attempt<cap;attempt++){var claimSlot=INVALID;var expected=0u;for(var probe=0u;probe<cap;probe++){let slot=(start+probe)%cap;let key=atomicLoad(&tileStates[slot*2u]);if(key==encoded){return slot;}if(key==INVALID&&claimSlot==INVALID){claimSlot=slot;expected=INVALID;}if(key==0u){if(claimSlot==INVALID){claimSlot=slot;expected=0u;}break;}}if(claimSlot==INVALID){break;}loop{let result=atomicCompareExchangeWeak(&tileStates[claimSlot*2u],expected,encoded);if(result.exchanged){return claimSlot;}if(result.old_value!=expected){break;}}}atomicStore(&transaction[0],7u);return INVALID;}
@compute @workgroup_size(64) fn beginSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let slot=gid.x;if(!producerAccepted()||slot>=brickSlots()){return;}let encoded=publishedStates[slot*2u];let old=publishedStates[slot*2u+1u];atomicStore(&states[slot*2u],encoded);if(encoded==0u||encoded==INVALID){atomicStore(&states[slot*2u+1u],0u);return;}let was=(old&RESIDENT)!=0u;atomicStore(&states[slot*2u+1u],select(0u,WAS_RESIDENT,was)|(min(0xffffu,(old>>16u)+1u)<<16u));}
@compute @workgroup_size(64) fn beginSparseTiles(@builtin(global_invocation_id) gid:vec3u){let slot=gid.x;if(!producerAccepted()||slot>=tileSlots()){return;}atomicStore(&tileStates[slot*2u],publishedTileStates[slot*2u]);atomicStore(&tileStates[slot*2u+1u],0u);}
// Same Section 4.2 pressure-support transaction as the dense scheduler above;
// hashing here is only sparse storage addressing, not a numerical method.
@compute @workgroup_size(64) fn markPressureTiles(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||row>=arrayLength(&leaves)){return;}let leaf=leaves[row];if((leaf.flags&LIVE)==0u||leaf.size==0u){return;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);if(any(a>=params.dimsBrick.xyz)){atomicStore(&transaction[0],9u);return;}let tileCells=params.dimsBrick.w*params.tiling.x;let td=vec3i(params.tiling.yzw);let first=vec3i(a/tileCells);let last=vec3i(min(params.dimsBrick.xyz-vec3u(1u),a+vec3u(leaf.size-1u))/tileCells);for(var tz=first.z-1;tz<=last.z+1;tz++){for(var ty=first.y-1;ty<=last.y+1;ty++){for(var tx=first.x-1;tx<=last.x+1;tx++){let q=vec3i(tx,ty,tz);if(any(q<vec3i(0))||any(q>=td)){continue;}let key=u32(q.x)+params.tiling.y*(u32(q.y)+params.tiling.z*u32(q.z));let slot=claimTile(key);if(slot!=INVALID){atomicStore(&tileStates[slot*2u+1u],1u);}}}}}
@compute @workgroup_size(64) fn markSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(!producerAccepted()||i>=candidateControl[0]){return;}if(i>=arrayLength(&candidates)){atomicStore(&transaction[0],1u);return;}let c=candidates[i];if(c.row>=arrayLength(&leaves)){atomicStore(&transaction[0],2u);return;}let leaf=leaves[c.row];if(leaf.size==0u){atomicStore(&transaction[0],3u);return;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);if(any(a>=params.dimsBrick.xyz)){atomicStore(&transaction[0],4u);return;}let first=a/params.dimsBrick.w;let last=min(params.brickDimsCapacity.xyz-vec3u(1u),(a+vec3u(leaf.size-1u))/params.dimsBrick.w);let bits=select(HALO,CORE,(c.flags&CORE)!=0u);for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){let logical=x+params.brickDimsCapacity.x*(y+params.brickDimsCapacity.y*z);let slot=claimBrick(logical);if(slot!=0xffffffffu){atomicOr(&states[slot*2u+1u],bits);}}}}}
@compute @workgroup_size(64) fn resolveSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){let slot=gid.x;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||slot>=brickSlots()){return;}let encoded=atomicLoad(&states[slot*2u]);if(encoded==0u||encoded==INVALID){return;}let logical=encoded-1u;if(logical>=params.brickDimsCapacity.w){atomicStore(&transaction[0],8u);return;}let marked=atomicLoad(&states[slot*2u+1u]);let desired=(marked&(CORE|HALO))!=0u;let was=(marked&WAS_RESIDENT)!=0u;let dry=select(marked>>16u,0u,desired);let persistent=params.settings.w>0.5;let resident=select(desired||(was&&dry<=u32(params.settings.y)),was||desired,persistent);let core=(marked&CORE)!=0u;let flags=select(0u,RESIDENT,resident)|select(0u,CORE,core)|select(0u,HALO,resident&&!core)|select(0u,ACTIVATED,resident&&!was)|select(0u,WAS_RESIDENT,was);atomicStore(&states[slot*2u+1u],flags|(dry<<16u));let cap=worklistCapacity();if(resident){let output=atomicAdd(&worklist[0],1u);if(output<cap){atomicStore(&worklist[HEADER+output*2u],logical);atomicStore(&worklist[HEADER+output*2u+1u],logical);}if(core){atomicAdd(&worklist[8],1u);}else{atomicAdd(&worklist[9],1u);}if(!was){atomicAdd(&worklist[10],1u);}}else if(was){let output=atomicAdd(&worklist[4],1u);if(output<cap){atomicStore(&worklist[HEADER+cap*2u+output*2u],logical);atomicStore(&worklist[HEADER+cap*2u+output*2u+1u],logical);}atomicAdd(&worklist[11],1u);atomicStore(&states[slot*2u+1u],0u);atomicStore(&states[slot*2u],INVALID);}else{atomicStore(&states[slot*2u+1u],0u);atomicStore(&states[slot*2u],INVALID);}}
@compute @workgroup_size(64) fn markSparseTiles(@builtin(global_invocation_id) gid:vec3u){let item=gid.x;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||item>=atomicLoad(&worklist[0])||item>=worklistCapacity()){return;}let logical=atomicLoad(&worklist[HEADER+item*2u]);let b=vec3u(logical%params.brickDimsCapacity.x,(logical/params.brickDimsCapacity.x)%params.brickDimsCapacity.y,logical/(params.brickDimsCapacity.x*params.brickDimsCapacity.y));let q=vec3i(b/params.tiling.x);let td=vec3i(params.tiling.yzw);for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let n=q+vec3i(x,y,z);if(any(n<vec3i(0))||any(n>=td)){continue;}let key=u32(n.x)+params.tiling.y*(u32(n.y)+params.tiling.z*u32(n.z));let slot=claimTile(key);if(slot!=0xffffffffu){atomicStore(&tileStates[slot*2u+1u],1u);}}}}}
@compute @workgroup_size(64) fn emitTiles(@builtin(global_invocation_id) gid:vec3u){let slot=gid.x;if(!producerAccepted()||atomicLoad(&transaction[0])!=0u||slot>=tileSlots()){return;}let encoded=atomicLoad(&tileStates[slot*2u]);if(encoded==0u||encoded==INVALID){return;}let key=encoded-1u;let live=atomicLoad(&tileStates[slot*2u+1u])!=0u;let oldEncoded=publishedTileStates[slot*2u];let was=oldEncoded==encoded;let cap=tileWorklistCapacity();if(live){let output=atomicAdd(&tileWorklist[0],1u);if(output<cap){atomicStore(&tileWorklist[HEADER+output],key);}}else if(was){let output=atomicAdd(&tileWorklist[4],1u);if(output<cap){atomicStore(&tileWorklist[HEADER+cap+output],key);}atomicStore(&tileStates[slot*2u+1u],0u);atomicStore(&tileStates[slot*2u],INVALID);}else{atomicStore(&tileStates[slot*2u+1u],0u);atomicStore(&tileStates[slot*2u],INVALID);}}
@compute @workgroup_size(1) fn finalize(){if(!producerAccepted()||atomicLoad(&transaction[0])!=0u){return;}let cap=worklistCapacity();let tc=tileWorklistCapacity();let rawResident=atomicLoad(&worklist[0]);let rawRetired=atomicLoad(&worklist[4]);let rawActiveTiles=atomicLoad(&tileWorklist[0]);let rawRetiredTiles=atomicLoad(&tileWorklist[4]);if(rawResident>cap||rawRetired>cap||rawActiveTiles>tc||rawRetiredTiles>tc){atomicStore(&transaction[0],5u);return;}let voxels=params.dimsBrick.w*params.dimsBrick.w*params.dimsBrick.w;let a=dispatch2((rawResident*voxels+255u)/256u);atomicStore(&worklist[1],a.x);atomicStore(&worklist[2],a.y);atomicStore(&worklist[3],1u);let s=dispatch2((rawResident*voxels+63u)/64u);atomicStore(&worklist[12],s.x);atomicStore(&worklist[13],s.y);atomicStore(&worklist[14],1u);let r=dispatch2((rawRetired*voxels+255u)/256u);atomicStore(&worklist[5],r.x);atomicStore(&worklist[6],r.y);atomicStore(&worklist[7],1u);atomicStore(&worklist[15],candidateControl[4]);let bg=select(1u,8u,params.dimsBrick.w==8u);let g=params.tiling.x*params.tiling.x*params.tiling.x*bg;let ad=dispatch2(rawActiveTiles*g);atomicStore(&tileWorklist[1],ad.x);atomicStore(&tileWorklist[2],ad.y);atomicStore(&tileWorklist[3],1u);let rd=dispatch2(rawRetiredTiles*g);atomicStore(&tileWorklist[5],rd.x);atomicStore(&tileWorklist[6],rd.y);atomicStore(&tileWorklist[7],1u);let cg=max(1u,g/8u);let ac=dispatch2(rawActiveTiles*cg);atomicStore(&tileWorklist[8],ac.x);atomicStore(&tileWorklist[9],ac.y);atomicStore(&tileWorklist[10],1u);let rc=dispatch2(rawRetiredTiles*cg);atomicStore(&tileWorklist[12],rc.x);atomicStore(&tileWorklist[13],rc.y);atomicStore(&tileWorklist[14],1u);atomicStore(&tileWorklist[15],candidateControl[4]);atomicStore(&transaction[1],1u);}
`;

export const surfaceCandidateCommitShader = /* wgsl */ `
struct Params { dimsBrick:vec4u, brickDimsCapacity:vec4u, settings:vec4f, tiling:vec4u }
@group(0) @binding(0) var<storage,read_write> publishedStates:array<u32>;
@group(0) @binding(1) var<storage,read> candidateStates:array<u32>;
@group(0) @binding(2) var<storage,read_write> publishedWorklist:array<u32>;
@group(0) @binding(3) var<storage,read> candidateWorklist:array<u32>;
@group(0) @binding(4) var<storage,read_write> publishedTileStates:array<u32>;
@group(0) @binding(5) var<storage,read> candidateTileStates:array<u32>;
@group(0) @binding(6) var<storage,read_write> publishedTileWorklist:array<u32>;
@group(0) @binding(7) var<storage,read> candidateTileWorklist:array<u32>;
@group(0) @binding(8) var<storage,read> transaction:array<u32>;
@group(0) @binding(9) var<uniform> params:Params;
@compute @workgroup_size(64) fn commitSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){if(transaction[1]!=1u){return;}let i=gid.x;if(i<arrayLength(&publishedStates)){publishedStates[i]=candidateStates[i];}if(i<arrayLength(&publishedWorklist)){publishedWorklist[i]=candidateWorklist[i];}if(i<arrayLength(&publishedTileStates)){publishedTileStates[i]=candidateTileStates[i];}if(i<arrayLength(&publishedTileWorklist)){publishedTileWorklist[i]=candidateTileWorklist[i];}}
`;

export class GPUFluidBrickResidency {
  readonly brickSize: 4 | 8;
  readonly brickDimensions: readonly [number, number, number];
  readonly capacity: number;
  /** Physical scheduler slots; equals `capacity` in compatibility mode. */
  readonly publicationCapacity: number;
  readonly worklist: GPUBuffer;
  readonly worklistByteLength: number;
  /** Bricks per topology-tile axis (power of two, 1 = tile congruent with brick). */
  readonly topologyTileBricks: number;
  readonly tileDimensions: readonly [number, number, number];
  readonly tileCapacity: number;
  /** Physical topology-tile scheduler slots. */
  readonly tilePublicationCapacity: number;
  /** Active/retired topology-tile worklist consumed by octree rebuilds. */
  readonly tileWorklist: GPUBuffer;
  readonly tileWorklistByteLength: number;

  private readonly device: GPUDevice;
  private readonly states: GPUBuffer;
  private readonly tileStates: GPUBuffer;
  private readonly candidateStates: GPUBuffer;
  private readonly candidateWorklist: GPUBuffer;
  private readonly candidateTileStates: GPUBuffer;
  private readonly candidateTileWorklist: GPUBuffer;
  private readonly candidateTransaction: GPUBuffer;
  private leafIndices: GPUBuffer;
  private leafStatesBuffer: GPUBuffer;
  private currentAllocationPlan: FluidBrickResidencyAllocationPlan;
  private candidateOnly: boolean;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly surfaceCandidateLayout: GPUBindGroupLayout;
  private readonly surfaceCandidateCommitLayout: GPUBindGroupLayout;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly classifySweptPipeline: GPUComputePipeline;
  private readonly expandDownstreamPipeline: GPUComputePipeline;
  private readonly emitWorklistPipeline: GPUComputePipeline;
  private readonly emitTopologyTilesPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly beginSurfaceCandidatesPipeline: GPUComputePipeline;
  private readonly markPressureTopologyTilesPipeline: GPUComputePipeline;
  private readonly markSurfaceCandidatesPipeline: GPUComputePipeline;
  private readonly resolveSurfaceCandidatesPipeline: GPUComputePipeline;
  private readonly emitSurfaceCandidateTilesPipeline: GPUComputePipeline;
  private readonly beginSurfaceCandidateTilesPipeline?: GPUComputePipeline;
  private readonly markSurfaceCandidateTilesPipeline?: GPUComputePipeline;
  private readonly finalizeSurfaceCandidatesPipeline: GPUComputePipeline;
  private readonly commitSurfaceCandidatesPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    dimensions: readonly [number, number, number],
    cellSize: readonly [number, number, number],
    options: FluidBrickResidencyOptions = {},
  ) {
    this.device = device;
    this.brickSize = options.brickSize ?? 8;
    if (this.brickSize !== 4 && this.brickSize !== 8) throw new RangeError("Fluid brick size must be 4 or 8");
    for (const [axis, value] of dimensions.entries()) if (!Number.isInteger(value) || value < 1) throw new RangeError(`Fluid dimension ${axis} must be positive`);
    for (const value of cellSize) if (!(value > 0) || !Number.isFinite(value)) throw new RangeError("Fluid cell size must be positive and finite");
    this.brickDimensions = dimensions.map((value) => Math.ceil(value / this.brickSize)) as [number, number, number];
    this.capacity = this.brickDimensions[0] * this.brickDimensions[1] * this.brickDimensions[2];
    this.candidateOnly = options.surfaceCandidatesOnly === true;
    const explicitMapping = options.leafIndices;
    if (this.candidateOnly && explicitMapping) {
      throw new RangeError("Surface-candidate-only residency requires implicit brick/leaf identity");
    }
    if (explicitMapping && explicitMapping.length !== this.capacity) throw new RangeError("Fluid brick leaf mapping must cover every solver brick");
    let maximumMappedLeaf = 0;
    if (explicitMapping) for (const leaf of explicitMapping) maximumMappedLeaf = Math.max(maximumMappedLeaf, leaf);
    const leafCapacity = options.leafCapacity ?? (explicitMapping ? maximumMappedLeaf + 1 : this.capacity);
    if (!Number.isInteger(leafCapacity) || leafCapacity < 1 || (explicitMapping?.some((leaf) => leaf >= leafCapacity) ?? false)) throw new RangeError("Fluid brick leaf capacity is invalid");
    const mapping = explicitMapping ?? new Uint32Array([0xffff_ffff]);
    const buffer = (label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBufferView<ArrayBuffer>) => {
      const result = device.createBuffer({ label, size: Math.max(4, size), usage });
      if (data && data.byteLength > 0) device.queue.writeBuffer(result, 0, data);
      return result;
    };
    const tileBricks = options.topologyTileBricks ?? 1;
    if (!Number.isInteger(tileBricks) || tileBricks < 1 || (tileBricks & (tileBricks - 1)) !== 0) {
      throw new RangeError("Topology tile size must be a positive power-of-two brick count");
    }
    this.topologyTileBricks = tileBricks;
    this.tileDimensions = this.brickDimensions.map((value) => Math.ceil(value / tileBricks)) as [number, number, number];
    this.tileCapacity = this.tileDimensions[0] * this.tileDimensions[1] * this.tileDimensions[2];
    this.currentAllocationPlan = planFluidBrickResidencyAllocation(
      this.brickDimensions,
      this.tileDimensions,
      leafCapacity,
      explicitMapping !== undefined,
      this.candidateOnly,
      options.surfaceCandidateBrickCapacity,
      options.surfaceCandidateTileCapacity,
    );
    this.publicationCapacity = this.currentAllocationPlan.brickStateCapacity;
    this.tilePublicationCapacity = this.currentAllocationPlan.tileStateCapacity;
    this.states = buffer("Fluid brick page states", this.currentAllocationPlan.stateBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const worklistWords = FLUID_BRICK_WORKLIST_HEADER_WORDS + this.publicationCapacity * 4;
    this.worklistByteLength = worklistWords * 4;
    this.worklist = buffer("Fluid brick active and retired worklists", worklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const tileWorklistWords = FLUID_TILE_WORKLIST_HEADER_WORDS + this.tilePublicationCapacity * 2;
    this.tileWorklistByteLength = tileWorklistWords * 4;
    // Words 8..10 and 12..14 are the candidate-page dispatches consumed by
    // the Section 4.2 owner-support transaction.
    this.tileWorklist = buffer("Topology tile active and retired worklists", tileWorklistWords * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.tileStates = buffer("Persistent topology tile activity", this.currentAllocationPlan.tileStateBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateStates = buffer("Candidate fluid brick page states", this.currentAllocationPlan.stateBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateWorklist = buffer("Candidate fluid brick worklists", worklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateTileStates = buffer("Candidate topology tile activity", this.currentAllocationPlan.tileStateBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateTileWorklist = buffer("Candidate topology tile worklists", tileWorklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.candidateTransaction = buffer("Surface candidate publication transaction", 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.leafIndices = buffer("Fluid brick to sparse leaf mapping", mapping.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mapping);
    this.leafStatesBuffer = buffer(
      this.candidateOnly ? "Unused sparse leaf residency fallback" : "Sparse leaf fluid residency",
      this.currentAllocationPlan.leafStateBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.params = buffer("Fluid brick residency parameters", 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const parameterData = new ArrayBuffer(64), uints = new Uint32Array(parameterData), floats = new Float32Array(parameterData);
    uints.set([dimensions[0], dimensions[1], dimensions[2], this.brickSize], 0);
    uints.set([this.brickDimensions[0], this.brickDimensions[1], this.brickDimensions[2], this.capacity], 4);
    uints.set([tileBricks, this.tileDimensions[0], this.tileDimensions[1], this.tileDimensions[2]], 12);
    const haloCells = options.haloCells ?? 2;
    const retireAfterFrames = options.retireAfterFrames ?? 3;
    if (!(haloCells >= 0) || !Number.isFinite(haloCells)) throw new RangeError("Fluid brick halo must be finite and non-negative");
    if (!Number.isInteger(retireAfterFrames) || retireAfterFrames < 0 || retireAfterFrames > 0xffff) throw new RangeError("Fluid brick retirement window must be a uint16");
    floats.set([haloCells * Math.max(...cellSize), retireAfterFrames, 0, options.includeLiquidInterior ? 1 : 0], 8);
    device.queue.writeBuffer(this.params, 0, parameterData);
    const shaderModule = device.createShaderModule({ label: "Fluid brick residency shader", code: fluidBrickResidencyShader });
    this.layout = device.createBindGroupLayout({ label: "Fluid brick residency layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.classifyPipeline = device.createComputePipeline({ label: "Classify fluid brick residency", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "classify" } });
    this.classifySweptPipeline = device.createComputePipeline({ label: "Classify fluid brick residency with swept support", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "classifySwept" } });
    this.expandDownstreamPipeline = device.createComputePipeline({ label: "Expand fluid brick residency downstream", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "expandDownstream" } });
    this.emitWorklistPipeline = device.createComputePipeline({ label: "Emit fluid brick worklists", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "emitWorklist" } });
    this.emitTopologyTilesPipeline = device.createComputePipeline({ label: "Emit topology tile worklists", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "emitTopologyTiles" } });
    this.finalizePipeline = device.createComputePipeline({ label: "Finalize fluid brick worklists", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "finalize" } });
    this.surfaceCandidateLayout=device.createBindGroupLayout({label:"Adaptive surface brick residency candidate layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
    ]});
    const surfaceModule=device.createShaderModule({label:"Adaptive surface brick residency shader",code:this.currentAllocationPlan.sparseKeyPools?sparseSurfaceCandidateResidencyShader:surfaceCandidateResidencyShader});
    const surfaceLayout=device.createPipelineLayout({bindGroupLayouts:[this.surfaceCandidateLayout]});
    const surfacePipeline=(label:string,entryPoint:string)=>device.createComputePipeline({label,layout:surfaceLayout,compute:{module:surfaceModule,entryPoint}});
    this.beginSurfaceCandidatesPipeline=surfacePipeline("Begin surface-candidate brick residency","beginSurfaceCandidates");
    this.markPressureTopologyTilesPipeline=surfacePipeline("Mark live pressure topology tiles","markPressureTiles");
    this.markSurfaceCandidatesPipeline=surfacePipeline("Mark surface-candidate brick residency","markSurfaceCandidates");
    this.resolveSurfaceCandidatesPipeline=surfacePipeline("Resolve surface-candidate brick residency","resolveSurfaceCandidates");
    this.emitSurfaceCandidateTilesPipeline=surfacePipeline("Emit surface-candidate topology tiles","emitTiles");
    if(this.currentAllocationPlan.sparseKeyPools){
      this.beginSurfaceCandidateTilesPipeline=surfacePipeline("Begin sparse surface-candidate topology tiles","beginSparseTiles");
      this.markSurfaceCandidateTilesPipeline=surfacePipeline("Mark sparse surface-candidate topology tiles","markSparseTiles");
    }else{
      this.beginSurfaceCandidateTilesPipeline=surfacePipeline("Begin dense pressure topology tiles","beginPressureTiles");
    }
    this.finalizeSurfaceCandidatesPipeline=surfacePipeline("Finalize surface-candidate residency","finalize");
    this.surfaceCandidateCommitLayout=device.createBindGroupLayout({label:"Adaptive surface brick residency commit layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
    ]});
    const commitModule=device.createShaderModule({label:"Adaptive surface brick residency commit shader",code:surfaceCandidateCommitShader});
    this.commitSurfaceCandidatesPipeline=device.createComputePipeline({label:"Commit surface-candidate residency",layout:device.createPipelineLayout({bindGroupLayouts:[this.surfaceCandidateCommitLayout]}),compute:{module:commitModule,entryPoint:"commitSurfaceCandidates"}});
    // The texture binding changes with the projection's ping-pong surface and
    // is therefore created in encode(). Keep the common resources resident.
  }

  /** GPU-owned per-brick state words, consumable by sibling schedulers (atlas). */
  get stateBuffer(): GPUBuffer { return this.states; }
  /** Persistent topology-tile state, shared with the analytic cold publisher. */
  get topologyTileStateBuffer(): GPUBuffer { return this.tileStates; }
  get leafStates(): GPUBuffer { return this.leafStatesBuffer; }
  get surfaceCandidatesOnly(): boolean { return this.candidateOnly; }
  get allocationPlan(): FluidBrickResidencyAllocationPlan { return this.currentAllocationPlan; }
  get allocatedBytes(): number { return this.currentAllocationPlan.allocatedBytes; }

  /**
   * Permanently retire legacy leaf mirrors once compact surface candidates are
   * authoritative. Existing submitted commands retain WebGPU resource
   * lifetime. Dense classification remains usable for bootstrap worklists.
   */
  cutoverToSurfaceCandidatesOnly(): number {
    if (this.destroyed || this.candidateOnly) return 0;
    if (this.currentAllocationPlan.identityMapping !== "implicit") {
      throw new Error("Explicit brick/leaf mappings cannot cut over to surface-candidate-only residency");
    }
    const previousBytes = this.currentAllocationPlan.allocatedBytes;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const leafIndexFallback = this.device.createBuffer({ label: "Unused fluid brick leaf-index fallback", size: 4, usage });
    const leafStateFallback = this.device.createBuffer({ label: "Unused sparse leaf residency fallback", size: 4, usage });
    this.device.queue.writeBuffer(leafIndexFallback, 0, new Uint32Array([0xffff_ffff]));
    this.leafIndices.destroy();
    this.leafStatesBuffer.destroy();
    this.leafIndices = leafIndexFallback;
    this.leafStatesBuffer = leafStateFallback;
    this.candidateOnly = true;
    this.currentAllocationPlan = planFluidBrickResidencyAllocation(
      this.brickDimensions,
      this.tileDimensions,
      this.currentAllocationPlan.leafCapacity,
      false,
      true,
    );
    return previousBytes - this.currentAllocationPlan.allocatedBytes;
  }

  encode(
    encoder: GPUCommandEncoder,
    levelSet: GPUTexture,
    velocity?: GPUTexture,
    options: { dt_s?: number; preActivation?: boolean } = {},
  ): void {
    if (this.destroyed) return;
    const preActivation = (options.preActivation ?? false) && !!velocity;
    // settings.z is the per-publication swept dt. settings.w is the immutable
    // includeLiquidInterior bit written by the constructor; overwriting both
    // words here made every pre-activated narrow surface scheduler retain the
    // complete deep-liquid volume.
    this.device.queue.writeBuffer(this.params, 40, new Float32Array([Math.max(0, options.dt_s ?? 0)]));
    // Preserve word 15 as a monotonically increasing GPU generation counter.
    encoder.clearBuffer(this.worklist, 0, (FLUID_BRICK_WORKLIST_HEADER_WORDS - 1) * 4);
    encoder.clearBuffer(this.tileWorklist, 0, FLUID_TILE_WORKLIST_HEADER_WORDS * 4);
    const bindGroup = this.device.createBindGroup({ label: "Fluid brick residency bindings", layout: this.layout, entries: [
      { binding: 0, resource: levelSet.createView() },
      { binding: 1, resource: { buffer: this.states } },
      { binding: 2, resource: { buffer: this.worklist } },
      { binding: 3, resource: { buffer: this.leafIndices } },
      { binding: 4, resource: { buffer: this.leafStatesBuffer } },
      { binding: 5, resource: { buffer: this.params } },
      // The velocity texture only feeds swept support / downstream expansion;
      // the level set doubles as a typed placeholder when no velocity exists.
      { binding: 6, resource: (velocity ?? levelSet).createView() },
      { binding: 7, resource: { buffer: this.tileWorklist } },
      { binding: 8, resource: { buffer: this.tileStates } },
    ] });
    const classify = encoder.beginComputePass({ label: "Classify evolving fluid bricks" });
    classify.setBindGroup(0, bindGroup);
    const bricks = Math.ceil(this.capacity / 64);
    if (preActivation) {
      classify.setPipeline(this.classifySweptPipeline); classify.dispatchWorkgroups(bricks);
      classify.setPipeline(this.expandDownstreamPipeline); classify.dispatchWorkgroups(bricks);
      classify.setPipeline(this.emitWorklistPipeline); classify.dispatchWorkgroups(bricks);
    } else {
      classify.setPipeline(this.classifyPipeline); classify.dispatchWorkgroups(bricks);
    }
    // Tiles derive from the final brick states of this pass, so the emission
    // runs after classification (and after downstream expansion when on).
    classify.setPipeline(this.emitTopologyTilesPipeline);
    classify.dispatchWorkgroups(Math.ceil(this.tileCapacity / 64));
    classify.end();
    const finalize = encoder.beginComputePass({ label: "Finalize evolving fluid brick worklists" });
    finalize.setPipeline(this.finalizePipeline);
    finalize.setBindGroup(0, bindGroup);
    finalize.dispatchWorkgroups(1);
    finalize.end();
  }

  /** Derive the legacy brick/tile scheduler ABI from compact surface leaves. */
  encodeSurfaceCandidates(
    encoder: GPUCommandEncoder,
    leaves: GPUBuffer,
    candidates: GPUBuffer,
    candidateControl: GPUBuffer,
  ): void {
    if (this.destroyed) return;
    // Preserve the stable publication and stage generation B independently.
    // Header word 15 carries generation and is copied before clearing all
    // other candidate counters/dispatches. No candidate producer buffer is
    // used as INDIRECT while it is storage-bound in this encoding scope.
    encoder.copyBufferToBuffer(this.worklist, 15 * 4, this.candidateWorklist, 15 * 4, 4);
    encoder.copyBufferToBuffer(this.tileWorklist, 15 * 4, this.candidateTileWorklist, 15 * 4, 4);
    encoder.clearBuffer(this.candidateWorklist, 0, (FLUID_BRICK_WORKLIST_HEADER_WORDS - 1) * 4);
    encoder.clearBuffer(this.candidateTileWorklist, 0, (FLUID_TILE_WORKLIST_HEADER_WORDS - 1) * 4);
    encoder.clearBuffer(this.candidateTransaction);
    const bindGroup=this.device.createBindGroup({label:"Adaptive surface brick residency candidate bindings",layout:this.surfaceCandidateLayout,entries:[
      {binding:0,resource:{buffer:this.states}},{binding:1,resource:{buffer:this.candidateStates}},
      {binding:2,resource:{buffer:this.candidateWorklist}},{binding:3,resource:{buffer:this.tileStates}},
      {binding:4,resource:{buffer:this.candidateTileStates}},{binding:5,resource:{buffer:this.candidateTileWorklist}},
      {binding:6,resource:{buffer:leaves}},{binding:7,resource:{buffer:candidates}},
      {binding:8,resource:{buffer:candidateControl}},{binding:9,resource:{buffer:this.candidateTransaction}},
      {binding:10,resource:{buffer:this.params}},
    ]});
    const bricks=Math.ceil(this.publicationCapacity/64);
    const tiles=Math.ceil(this.tilePublicationCapacity/64);
    const stage=(label:string,pipeline:GPUComputePipeline,workgroups:number)=>{const pass=encoder.beginComputePass({label});pass.setBindGroup(0,bindGroup);pass.setPipeline(pipeline);pass.dispatchWorkgroups(workgroups);pass.end();};
    stage("Begin adaptive surface brick residency",this.beginSurfaceCandidatesPipeline,bricks);
    if(this.beginSurfaceCandidateTilesPipeline) stage("Begin adaptive sparse topology tiles",this.beginSurfaceCandidateTilesPipeline,tiles);
    stage("Mark live pressure topology tiles",this.markPressureTopologyTilesPipeline,
      Math.ceil(Math.max(1,leaves.size/64)/64));
    stage("Mark adaptive surface brick residency",this.markSurfaceCandidatesPipeline,Math.ceil(Math.max(1,candidates.size/8)/64));
    stage("Resolve adaptive surface brick residency",this.resolveSurfaceCandidatesPipeline,bricks);
    if(this.markSurfaceCandidateTilesPipeline) stage("Mark adaptive sparse topology tiles",this.markSurfaceCandidateTilesPipeline,bricks);
    stage("Emit adaptive surface topology tiles",this.emitSurfaceCandidateTilesPipeline,tiles);
    const finalize=encoder.beginComputePass({label:"Finalize adaptive surface brick worklists"});
    finalize.setPipeline(this.finalizeSurfaceCandidatesPipeline);finalize.setBindGroup(0,bindGroup);finalize.dispatchWorkgroups(1);finalize.end();
    const commitGroup=this.device.createBindGroup({label:"Adaptive surface brick residency commit bindings",layout:this.surfaceCandidateCommitLayout,entries:[
      {binding:0,resource:{buffer:this.states}},{binding:1,resource:{buffer:this.candidateStates}},
      {binding:2,resource:{buffer:this.worklist}},{binding:3,resource:{buffer:this.candidateWorklist}},
      {binding:4,resource:{buffer:this.tileStates}},{binding:5,resource:{buffer:this.candidateTileStates}},
      {binding:6,resource:{buffer:this.tileWorklist}},{binding:7,resource:{buffer:this.candidateTileWorklist}},
      {binding:8,resource:{buffer:this.candidateTransaction}},{binding:9,resource:{buffer:this.params}},
    ]});
    const commit=encoder.beginComputePass({label:"Commit adaptive surface brick residency"});
    commit.setPipeline(this.commitSurfaceCandidatesPipeline);commit.setBindGroup(0,commitGroup);
    commit.dispatchWorkgroups(Math.ceil(Math.max(this.worklistByteLength,this.tileWorklistByteLength,
      this.currentAllocationPlan.stateBytes,this.currentAllocationPlan.tileStateBytes)/4/64));commit.end();
  }

  async readStats(): Promise<FluidBrickResidencyStats> {
    if (this.destroyed) return { resident: 0, core: 0, halo: 0, activated: 0, retired: 0, generation: 0, capacity: this.capacity };
    const readback = this.device.createBuffer({ label: "Fluid brick residency readback", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read fluid brick residency" });
    encoder.copyBufferToBuffer(this.worklist, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, 64));
      return { resident: words[0], retired: words[11], core: words[8], halo: words[9], activated: words[10], generation: words[15], capacity: this.capacity };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.states.destroy();
    this.worklist.destroy();
    this.tileWorklist.destroy();
    this.tileStates.destroy();
    this.candidateStates.destroy();
    this.candidateWorklist.destroy();
    this.candidateTileStates.destroy();
    this.candidateTileWorklist.destroy();
    this.candidateTransaction.destroy();
    this.leafIndices.destroy();
    this.leafStatesBuffer.destroy();
    this.params.destroy();
  }
}
