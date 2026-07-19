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
  let leafIndex = leafIndices[brickIndex];
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
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states) || brickIndex >= arrayLength(&leafIndices)) { return; }
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
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states) || brickIndex >= arrayLength(&leafIndices)) { return; }
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
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states) || brickIndex >= arrayLength(&leafIndices)) { return; }
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
}
`;

export class GPUFluidBrickResidency {
  readonly brickSize: 4 | 8;
  readonly brickDimensions: readonly [number, number, number];
  readonly capacity: number;
  readonly leafStates: GPUBuffer;
  readonly worklist: GPUBuffer;
  readonly worklistByteLength: number;
  /** Bricks per topology-tile axis (power of two, 1 = tile congruent with brick). */
  readonly topologyTileBricks: number;
  readonly tileDimensions: readonly [number, number, number];
  readonly tileCapacity: number;
  /** Active/retired topology-tile worklist consumed by octree rebuilds. */
  readonly tileWorklist: GPUBuffer;
  readonly tileWorklistByteLength: number;
  readonly allocatedBytes: number;

  private readonly device: GPUDevice;
  private readonly states: GPUBuffer;
  private readonly tileStates: GPUBuffer;
  private readonly leafIndices: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly classifySweptPipeline: GPUComputePipeline;
  private readonly expandDownstreamPipeline: GPUComputePipeline;
  private readonly emitWorklistPipeline: GPUComputePipeline;
  private readonly emitTopologyTilesPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
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
    const mapping = options.leafIndices ?? Uint32Array.from({ length: this.capacity }, (_, index) => index);
    if (mapping.length !== this.capacity) throw new RangeError("Fluid brick leaf mapping must cover every solver brick");
    const leafCapacity = options.leafCapacity ?? (mapping.length === 0 ? 1 : Math.max(...mapping) + 1);
    if (!Number.isInteger(leafCapacity) || leafCapacity < 1 || mapping.some((leaf) => leaf >= leafCapacity)) throw new RangeError("Fluid brick leaf capacity is invalid");
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
    this.states = buffer("Fluid brick page states", this.capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const worklistWords = FLUID_BRICK_WORKLIST_HEADER_WORDS + this.capacity * 4;
    this.worklistByteLength = worklistWords * 4;
    this.worklist = buffer("Fluid brick active and retired worklists", worklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const tileWorklistWords = FLUID_TILE_WORKLIST_HEADER_WORDS + this.tileCapacity * 2;
    this.tileWorklistByteLength = tileWorklistWords * 4;
    this.tileWorklist = buffer("Topology tile active and retired worklists", tileWorklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.tileStates = buffer("Persistent topology tile activity", this.tileCapacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.leafIndices = buffer("Fluid brick to sparse leaf mapping", mapping.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mapping);
    this.leafStates = buffer("Sparse leaf fluid residency", leafCapacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
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
    // The texture binding changes with the projection's ping-pong surface and
    // is therefore created in encode(). Keep the common resources resident.
    this.allocatedBytes = this.capacity * 4 + worklistWords * 4 + tileWorklistWords * 4 + this.tileCapacity * 4 + mapping.byteLength + leafCapacity * 4 + 64;
  }

  /** GPU-owned per-brick state words, consumable by sibling schedulers (atlas). */
  get stateBuffer(): GPUBuffer { return this.states; }

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
      { binding: 4, resource: { buffer: this.leafStates } },
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
    this.leafIndices.destroy();
    this.leafStates.destroy();
    this.params.destroy();
  }
}
