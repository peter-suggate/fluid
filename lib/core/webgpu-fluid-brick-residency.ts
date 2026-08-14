/**
 * GPU-owned logical residency for finest-level fluid bricks.
 *
 * The dense textures remain a compatibility backing store while kernels are
 * migrated to brick payloads.  This page table is nevertheless authoritative
 * for sparse publication and octree work scheduling: only resident core/halo
 * bricks are emitted to the worklist, and dry bricks retire after a short
 * hysteresis window.
 */
import { PassBroker } from "./webgpu-pass-broker";
import type { GPUInitializationTask } from "./gpu-initialization";

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
  /**
   * Keep the closed side/floor (and optionally ceiling) pressure-wall tiles,
   * plus their one-tile grading support, resident even when they are dry.
   * The power topology refines those strips independently of liquid phi.
   */
  includePressureBoundarySupport?: boolean;
  pressureBoundaryTopClosed?: boolean;
  /**
   * Require liquid within reach before a dry pressure-wall tile is retained.
   *
   * The sentence above — "the power topology refines those strips
   * independently of liquid phi" — stopped being true when
   * `fluidGatedBoundaryRefinement` became the default: `refineLeaf` gates a
   * closed-wall crossing on `minimumPhi <= band`, so a dry wall tile far from
   * any liquid never splits. Retaining it anyway published topology for leaves
   * that do not exist, and the cost is not the tile — it is the 64 owner pages
   * per tile that the single-workgroup candidate builder then sorts, dedups and
   * copies every advance.
   *
   * That is the whole domain tax on a droplet in a vast container. The tile
   * lattice is a uniform occupancy grid; this flag is what stops the container
   * walls from marking all of it occupied. Measured on `power-droplet-256`
   * (100 liquid cells in a 256-cubed box): 448 of 512 tiles retained, 28,672 of
   * 30,400 candidate pages, and 52.9 ms of a 71 ms domain step.
   *
   * Wall tiles still search one tile further than liquid tiles, preserving the
   * 2:1 grading margin the unconditional retention was standing in for.
   */
  fluidGatedBoundarySupport?: boolean;
  /**
   * Retain the whole pressure-owner tile lattice for authored inflows. Inflow
   * protection refines dry cells around the nozzle before transported phi
   * arrives, so phi-derived residency alone cannot bound that support.
   */
  includeWholeDomainPressureSupport?: boolean;
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
  fineSeedCandidatesOnly?: boolean;
  /**
   * Physical sparse-key slots retained by direct fine-seed-candidate authority.
   * Logical brick coordinates remain unchanged; exhaustion rejects the whole
   * candidate generation and preserves the previous publication.
   */
  fineSeedCandidateBrickCapacity?: number;
  /** Sparse topology-tile key slots. See `fineSeedCandidateBrickCapacity`. */
  fineSeedCandidateTileCapacity?: number;
  /** Move shader/pipeline creation into the shared staged startup runner. */
  deferPipelineCompilation?: boolean;
}

export interface FluidBrickResidencyAllocationPlan {
  readonly brickCapacity: number;
  readonly tileCapacity: number;
  readonly leafCapacity: number;
  readonly identityMapping: "implicit" | "explicit";
  readonly fineSeedCandidatesOnly: boolean;
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
  /** A/B scratch state and worklists. The tile header owns the commit predicate. */
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
  fineSeedCandidatesOnly = false,
  fineSeedCandidateBrickCapacity?: number,
  fineSeedCandidateTileCapacity?: number,
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
  if (fineSeedCandidatesOnly && explicitLeafMapping) {
    throw new RangeError("Fine-seed-candidate-only residency requires implicit brick/leaf identity");
  }
  const sparseCapacity = (value: number | undefined, logical: number, label: string) => {
    if (value === undefined) return logical;
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} capacity must be a positive integer`);
    return Math.min(logical, value);
  };
  if (fineSeedCandidateTileCapacity !== undefined
    && !(fineSeedCandidatesOnly && fineSeedCandidateBrickCapacity !== undefined)) {
    throw new RangeError("Sparse tile capacity requires a sparse fine-seed-candidate brick capacity");
  }
  const requestedBrickStateCapacity = fineSeedCandidatesOnly && fineSeedCandidateBrickCapacity !== undefined
    ? sparseCapacity(fineSeedCandidateBrickCapacity, brickCapacity, "Fine-seed-candidate brick")
    : brickCapacity;
  const requestedTileStateCapacity = fineSeedCandidatesOnly && fineSeedCandidateBrickCapacity !== undefined
    ? sparseCapacity(fineSeedCandidateTileCapacity ?? requestedBrickStateCapacity * 27, tileCapacity,
      "Fine-seed-candidate topology tile")
    : tileCapacity;
  // A key pool which covers the complete logical domain has no sparse-memory
  // benefit. Use the dense logical-key representation in that case: its
  // device-wide mark/resolve dispatches replace the sparse publisher's
  // single-lane hash insertion and active-brick tile-ring scan. This is common
  // for small domains (including mini dam) where the surface budget clamps to
  // the domain volume.
  const sparseKeyPools = fineSeedCandidatesOnly
    && (requestedBrickStateCapacity < brickCapacity || requestedTileStateCapacity < tileCapacity);
  const brickStateCapacity = sparseKeyPools ? requestedBrickStateCapacity : brickCapacity;
  const tileStateCapacity = sparseKeyPools ? requestedTileStateCapacity : tileCapacity;
  // Sparse records store key-plus-one and lifecycle state. Dense compatibility
  // stores only the state because its array index is the logical key.
  const stateBytes = brickStateCapacity * (sparseKeyPools ? 8 : FLUID_BRICK_STATE_STRIDE_BYTES);
  const worklistBytes = (FLUID_BRICK_WORKLIST_HEADER_WORDS + brickStateCapacity * 4) * 4;
  const tileWorklistBytes = (FLUID_TILE_WORKLIST_HEADER_WORDS + tileStateCapacity * 2) * 4;
  const tileStateBytes = tileStateCapacity * (sparseKeyPools ? 8 : 4);
  const leafIndexBytes = explicitLeafMapping ? brickCapacity * 4 : 4;
  const leafStateBytes = fineSeedCandidatesOnly ? 4 : leafCapacity * 4;
  const parameterBytes = 64;
  // Dense multi-dispatch publication uses one trailing atomic word in its
  // candidate tile-state arena to carry fail-closed validation across dispatch
  // boundaries. The published arena retains the exact public ABI.
  const candidateControlBytes = sparseKeyPools ? 0 : 4;
  const transactionalBytes = stateBytes + worklistBytes + tileWorklistBytes + tileStateBytes
    + candidateControlBytes;
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
    fineSeedCandidatesOnly,
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
    savedLeafStateBytes: fineSeedCandidatesOnly ? leafCapacity * 4 - 4 : 0,
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

export interface FineSeedCandidateResidencyPoolPlan {
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
 *
 * `includeLiquidInterior` switches the budget from a surface band to the wet
 * volume. Pressure topology needs an owner page for every deep-liquid coarse
 * row, not just the refined interface sheet, so a scheduler that retains the
 * interior is bounded by its logical lattice rather than by interface area.
 * Keeping the area estimate there silently truncated the deep bulk: the tile
 * ring dilates one tile out of a *resident brick*, so once the tank floor sat
 * more than one topology tile below the free surface it published no tile, and
 * every cell in it decoded as an unmapped owner page.
 */
export function planFineSeedCandidateResidencyPools(
  brickDimensions: readonly [number, number, number],
  tileDimensions: readonly [number, number, number],
  brickSize: number,
  haloCells: number,
  producerRowCapacity: number,
  minimumTileCapacity = 1,
  includeLiquidInterior = false,
): FineSeedCandidateResidencyPoolPlan {
  const checkedVolume = (dims: readonly [number, number, number], label: string) => {
    if (!dims.every((value) => Number.isSafeInteger(value) && value > 0)) {
      throw new RangeError(`${label} dimensions must be positive integers`);
    }
    return dims[0] * dims[1] * dims[2];
  };
  if (!Number.isSafeInteger(brickSize) || brickSize < 1 || !Number.isFinite(haloCells) || haloCells < 0
    || !Number.isSafeInteger(producerRowCapacity) || producerRowCapacity < 1
    || !Number.isSafeInteger(minimumTileCapacity) || minimumTileCapacity < 0) {
    throw new RangeError("Fine-seed-candidate residency pool inputs are invalid");
  }
  const logicalBrickCount = checkedVolume(brickDimensions, "Brick");
  const logicalTileCount = checkedVolume(tileDimensions, "Topology tile");
  const area = (dims: readonly [number, number, number]) =>
    dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2];
  // Two-sided swept support plus one complete insertion/retirement generation.
  const bandBrickLayers = Math.max(2, Math.ceil(haloCells / brickSize) * 2 + 1);
  const topologyTileBricks = Math.max(1, Math.ceil(brickDimensions[0] / tileDimensions[0]));
  const bandTileLayers = Math.max(3, Math.ceil(bandBrickLayers / topologyTileBricks) + 2);
  const brickCapacity = includeLiquidInterior
    ? logicalBrickCount
    : Math.max(1, Math.min(logicalBrickCount, producerRowCapacity,
      Math.ceil(area(brickDimensions) * bandBrickLayers)));
  const tileCapacity = includeLiquidInterior
    ? logicalTileCount
    : Math.max(1, Math.min(logicalTileCount,
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
fn schedulerFlags() -> u32 { return u32(params.settings.w); }
fn includeLiquidInterior() -> bool { return (schedulerFlags() & 1u) != 0u; }
fn includePressureBoundarySupport() -> bool { return (schedulerFlags() & 2u) != 0u; }
fn pressureBoundaryTopClosed() -> bool { return (schedulerFlags() & 4u) != 0u; }
fn includeWholeDomainPressureSupport() -> bool { return (schedulerFlags() & 8u) != 0u; }
fn fluidGatedBoundarySupport() -> bool { return (schedulerFlags() & 16u) != 0u; }

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
    || (includeLiquidInterior() && minimumPhi < 0.0);
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
    || (includeLiquidInterior() && range.x < 0.0);
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

fn tileHasPressureBoundarySupport(tile: vec3u) -> bool {
  if (includeWholeDomainPressureSupport()) { return true; }
  if (!includePressureBoundarySupport()) { return false; }
  let d = params.tiling.yzw;
  // The topology's unit wall strip can force refinement in every boundary
  // tile even when phi is dry. Retain one additional maximum-leaf tile for
  // the same 2:1 grading closure used around resident liquid tiles.
  return tile.x < min(2u, d.x) || tile.x + 2u >= d.x
    || tile.z < min(2u, d.z) || tile.z + 2u >= d.z
    || tile.y < min(2u, d.y)
    || (pressureBoundaryTopClosed() && tile.y + 2u >= d.y);
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
  let boundary = tileHasPressureBoundarySupport(tile);
  // A dry wall tile is retained only when liquid is within reach. See the
  // fluidGatedBoundarySupport option for why unconditional retention was
  // over-conservative once boundary refinement itself became fluid-gated, and
  // for what it cost. Wall tiles search one tile wider than liquid tiles, which
  // is the 2:1 grading margin the unconditional form was standing in for.
  var anyResident = boundary && !fluidGatedBoundarySupport();
  let reach = select(1, 2, boundary);
  for (var dz = -reach; dz <= reach && !anyResident; dz += 1) {
    for (var dy = -reach; dy <= reach && !anyResident; dy += 1) {
      for (var dx = -reach; dx <= reach && !anyResident; dx += 1) {
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

const makeFineSeedCandidateShaderPrelude = (atomicCandidateState = false) => /* wgsl */ `
struct Params { dimsBrick:vec4u, brickDimsCapacity:vec4u, settings:vec4f, tiling:vec4u }
struct FineSeedLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
@group(0) @binding(0) var<storage,read_write> publishedStates:array<u32>;
@group(0) @binding(1) var<storage,read_write> states:array<${atomicCandidateState ? "atomic<u32>" : "u32"}>;
@group(0) @binding(2) var<storage,read_write> worklist:array<u32>;
@group(0) @binding(3) var<storage,read_write> publishedTileStates:array<u32>;
@group(0) @binding(4) var<storage,read_write> tileStates:array<${atomicCandidateState ? "atomic<u32>" : "u32"}>;
@group(0) @binding(5) var<storage,read_write> tileWorklist:array<u32>;
@group(0) @binding(6) var<storage,read> leaves:array<FineSeedLeaf>;
@group(0) @binding(7) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(8) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(9) var<storage,read_write> publishedTileWorklist:array<u32>;
@group(0) @binding(10) var<uniform> params:Params;
@group(0) @binding(11) var<storage,read_write> publishedWorklist:array<u32>;
const RESIDENT=1u;const CORE=2u;const HALO=4u;const ACTIVATED=8u;const LIVE=32u;
const WAS_RESIDENT=32u;const HEADER=16u;const INVALID=0xffffffffu;const COMMIT=0xc01117edu;
fn schedulerFlags()->u32{return u32(params.settings.w);}
fn persistentLiquid()->bool{return (schedulerFlags()&1u)!=0u;}
fn pressureBoundarySupport()->bool{return (schedulerFlags()&2u)!=0u;}
fn pressureBoundaryTopClosed()->bool{return (schedulerFlags()&4u)!=0u;}
fn wholeDomainPressureSupport()->bool{return (schedulerFlags()&8u)!=0u;}
fn gatedBoundarySupport()->bool{return (schedulerFlags()&16u)!=0u;}
fn pressureBoundaryTile(key:u32)->bool{
  if(wholeDomainPressureSupport()){return true;}
  // Gated: a dry wall tile is not claimed for its own sake. The live-leaf ring
  // marking below reaches every wall tile liquid can actually touch, and
  // boundary refinement is itself fluid-gated. See fluidGatedBoundarySupport.
  if(gatedBoundarySupport()){return false;}
  if(!pressureBoundarySupport()){return false;}
  let d=params.tiling.yzw;
  let q=vec3u(key%d.x,(key/d.x)%d.y,key/(d.x*d.y));
  return q.x<min(2u,d.x)||q.x+2u>=d.x
    ||q.z<min(2u,d.z)||q.z+2u>=d.z
    ||q.y<min(2u,d.y)||(pressureBoundaryTopClosed()&&q.y+2u>=d.y);
}
fn dispatch2(n:u32)->vec2u{let x=min(n,65535u);return vec2u(x,select(1u,(n+x-1u)/x,x>0u));}
fn producerAccepted()->bool{return candidateControl[5]==1u&&candidateControl[6]==0u
  &&candidateControl[7]==arrayLength(&candidates)&&candidateControl[4]>publishedTileWorklist[15];}
fn clearScratch(){for(var i=0u;i<arrayLength(&worklist);i++){worklist[i]=0u;}
  for(var i=0u;i<arrayLength(&tileWorklist);i++){tileWorklist[i]=0u;}}
fn finishHeaders(){
  let resident=worklist[0];let retired=worklist[4];let voxels=params.dimsBrick.w*params.dimsBrick.w*params.dimsBrick.w;
  let a=dispatch2((resident*voxels+255u)/256u);worklist[1]=a.x;worklist[2]=a.y;worklist[3]=1u;
  let s=dispatch2((resident*voxels+63u)/64u);worklist[12]=s.x;worklist[13]=s.y;worklist[14]=1u;
  let r=dispatch2((retired*voxels+255u)/256u);worklist[5]=r.x;worklist[6]=r.y;worklist[7]=1u;
  worklist[15]=candidateControl[4];
  let bg=select(1u,8u,params.dimsBrick.w==8u);let groups=params.tiling.x*params.tiling.x*params.tiling.x*bg;
  let ad=dispatch2(tileWorklist[0]*groups);tileWorklist[1]=ad.x;tileWorklist[2]=ad.y;tileWorklist[3]=1u;
  let rd=dispatch2(tileWorklist[4]*groups);tileWorklist[5]=rd.x;tileWorklist[6]=rd.y;tileWorklist[7]=1u;
  let candidatesPerTile=max(1u,groups/8u);let ac=dispatch2(tileWorklist[0]*candidatesPerTile);
  tileWorklist[8]=ac.x;tileWorklist[9]=ac.y;tileWorklist[10]=1u;
  let rc=dispatch2(tileWorklist[4]*candidatesPerTile);tileWorklist[12]=rc.x;tileWorklist[13]=rc.y;tileWorklist[14]=1u;
  tileWorklist[15]=candidateControl[4];tileWorklist[11]=COMMIT;
}
fn commitCandidateGeneration(lane:u32,stride:u32){
  for(var i=lane;i<arrayLength(&publishedStates);i+=stride){
    publishedStates[i]=${atomicCandidateState ? "atomicLoad(&states[i])" : "states[i]"};
  }
  for(var i=lane;i<arrayLength(&publishedWorklist);i+=stride){publishedWorklist[i]=worklist[i];}
  for(var i=lane;i<arrayLength(&publishedTileStates);i+=stride){
    publishedTileStates[i]=${atomicCandidateState ? "atomicLoad(&tileStates[i])" : "tileStates[i]"};
  }
  for(var i=lane;i<arrayLength(&publishedTileWorklist);i+=stride){
    publishedTileWorklist[i]=select(tileWorklist[i],0u,i==11u);
  }
}
`;

/**
 * Dense compact-candidate publisher. Expensive initialization, marking, and
 * lifecycle resolution occupy the full device. A final 256-lane stable-prefix
 * publication preserves deterministic logical-key order without a serial
 * capacity walk.
 */
export const fineSeedCandidateResidencyShader = makeFineSeedCandidateShaderPrelude(true) + /* wgsl */ `
fn tileCapacity()->u32{return params.tiling.y*params.tiling.z*params.tiling.w;}
fn recordDenseError(code:u32){atomicOr(&tileStates[tileCapacity()],code);}
fn markTileRing(key:u32){
  let td=vec3i(params.tiling.yzw);let q=vec3i(i32(key%params.tiling.y),
    i32((key/params.tiling.y)%params.tiling.z),i32(key/(params.tiling.y*params.tiling.z)));
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
    let n=q+vec3i(x,y,z);if(any(n<vec3i(0))||any(n>=td)){continue;}
    let logical=u32(n.x)+params.tiling.y*(u32(n.y)+params.tiling.z*u32(n.z));
    atomicOr(&tileStates[logical],1u);
  }}}
}
@compute @workgroup_size(256) fn prepareFineSeedCandidateResidency(@builtin(global_invocation_id)gid:vec3u){
  let i=gid.x;let brickCap=params.brickDimsCapacity.w;let tileCap=tileCapacity();
  if(i<arrayLength(&worklist)){worklist[i]=0u;}
  if(i<arrayLength(&tileWorklist)){tileWorklist[i]=0u;}
  if(i<brickCap){let old=publishedStates[i];let was=(old&RESIDENT)!=0u;
    atomicStore(&states[i],select(0u,WAS_RESIDENT,was)|(min(0xffffu,(old>>16u)+1u)<<16u));}
  if(i<tileCap){atomicStore(&tileStates[i],select(0u,1u,pressureBoundaryTile(i)));}
  if(i==tileCap){atomicStore(&tileStates[i],0u);}
}
@compute @workgroup_size(256) fn markFineSeedCandidateResidency(@builtin(global_invocation_id)gid:vec3u){
  if(!producerAccepted()){return;}let i=gid.x;
  // Aanjaneya et al. (2017), Section 4.2 requires the complete local
  // face-and-edge pressure-support ring, independently of the fine-phi band.
  let tileCells=params.dimsBrick.w*params.tiling.x;let td=vec3i(params.tiling.yzw);
  if(i<arrayLength(&leaves)){let leaf=leaves[i];
    if((leaf.flags&LIVE)!=0u&&leaf.size!=0u){let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);
      if(any(a>=params.dimsBrick.xyz)){recordDenseError(9u);}else{let first=vec3i(a/tileCells);
      let last=vec3i(min(params.dimsBrick.xyz-vec3u(1u),a+vec3u(leaf.size-1u))/tileCells);
      for(var z=first.z-1;z<=last.z+1;z++){for(var y=first.y-1;y<=last.y+1;y++){
        for(var x=first.x-1;x<=last.x+1;x++){let q=vec3i(x,y,z);
          if(any(q<vec3i(0))||any(q>=td)){continue;}
          atomicOr(&tileStates[u32(q.x)+params.tiling.y*(u32(q.y)+params.tiling.z*u32(q.z))],1u);
        }}}}
    }
  }
  if(i<candidateControl[0]){
    if(i>=arrayLength(&candidates)){recordDenseError(1u);return;}let c=candidates[i];
    if(c.row>=arrayLength(&leaves)){recordDenseError(2u);return;}let leaf=leaves[c.row];
    if(leaf.size==0u){recordDenseError(3u);return;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);
    if(any(a>=params.dimsBrick.xyz)){recordDenseError(4u);return;}let first=a/params.dimsBrick.w;
    let last=min(params.brickDimsCapacity.xyz-vec3u(1u),(a+vec3u(leaf.size-1u))/params.dimsBrick.w);
    let bits=select(HALO,CORE,(c.flags&CORE)!=0u);
    for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
      let logical=x+params.brickDimsCapacity.x*(y+params.brickDimsCapacity.y*z);
      atomicOr(&states[logical],bits);
    }}}
  }
}
@compute @workgroup_size(256) fn resolveFineSeedCandidateResidency(@builtin(global_invocation_id)gid:vec3u){
  if(!producerAccepted()){return;}let logical=gid.x;let brickCap=params.brickDimsCapacity.w;
  if(logical<brickCap){let marked=atomicLoad(&states[logical]);
    let desired=(marked&(CORE|HALO))!=0u;let was=(marked&WAS_RESIDENT)!=0u;
    let dry=select(marked>>16u,0u,desired);let persistent=persistentLiquid();
    let resident=select(desired||(was&&dry<=u32(params.settings.y)),was||desired,persistent);
    let core=(marked&CORE)!=0u;let flags=select(0u,RESIDENT,resident)|select(0u,CORE,core)
      |select(0u,HALO,resident&&!core)|select(0u,ACTIVATED,resident&&!was)|select(0u,WAS_RESIDENT,was);
    atomicStore(&states[logical],flags|(dry<<16u));
    if(resident){let b=vec3u(logical%params.brickDimsCapacity.x,
      (logical/params.brickDimsCapacity.x)%params.brickDimsCapacity.y,
      logical/(params.brickDimsCapacity.x*params.brickDimsCapacity.y));
      let key=(b.x/params.tiling.x)+params.tiling.y*((b.y/params.tiling.x)+params.tiling.z*(b.z/params.tiling.x));
      markTileRing(key);}
  }
}
var<workgroup> activePrefix:array<u32,256>;var<workgroup> retiredPrefix:array<u32,256>;
var<workgroup> corePrefix:array<u32,256>;var<workgroup> haloPrefix:array<u32,256>;
var<workgroup> activatedPrefix:array<u32,256>;var<workgroup> tileActivePrefix:array<u32,256>;
var<workgroup> tileRetiredPrefix:array<u32,256>;
var<workgroup> densePublishAccepted:u32;
@compute @workgroup_size(256) fn publishFineSeedCandidateResidency(@builtin(local_invocation_index)lid:u32){
  let brickCap=params.brickDimsCapacity.w;let tileCap=tileCapacity();
  if(lid==0u){densePublishAccepted=select(0u,1u,
    producerAccepted()&&atomicLoad(&tileStates[tileCap])==0u);}
  workgroupBarrier();if(workgroupUniformLoad(&densePublishAccepted)==0u){return;}
  let brickChunk=(brickCap+255u)/256u;let brickFirst=min(brickCap,lid*brickChunk);
  let brickEnd=min(brickCap,brickFirst+brickChunk);var activeCount=0u;var retired=0u;
  var core=0u;var halo=0u;var activated=0u;
  for(var logical=brickFirst;logical<brickEnd;logical+=1u){let marked=atomicLoad(&states[logical]);
    let resident=(marked&RESIDENT)!=0u;let was=(marked&WAS_RESIDENT)!=0u;
    activeCount+=select(0u,1u,resident);retired+=select(0u,1u,!resident&&was);
    core+=select(0u,1u,resident&&(marked&CORE)!=0u);
    halo+=select(0u,1u,resident&&(marked&CORE)==0u);
    activated+=select(0u,1u,resident&&!was);
  }
  let tileChunk=(tileCap+255u)/256u;let tileFirst=min(tileCap,lid*tileChunk);
  let tileEnd=min(tileCap,tileFirst+tileChunk);var tileActive=0u;var tileRetired=0u;
  for(var key=tileFirst;key<tileEnd;key+=1u){let live=atomicLoad(&tileStates[key])!=0u;
    let was=publishedTileStates[key]!=0u;tileActive+=select(0u,1u,live);
    tileRetired+=select(0u,1u,!live&&was);
  }
  activePrefix[lid]=activeCount;retiredPrefix[lid]=retired;corePrefix[lid]=core;haloPrefix[lid]=halo;
  activatedPrefix[lid]=activated;tileActivePrefix[lid]=tileActive;tileRetiredPrefix[lid]=tileRetired;
  workgroupBarrier();
  if(lid==0u){var a=0u;var r=0u;var c=0u;var h=0u;var n=0u;var ta=0u;var tr=0u;
    for(var lane=0u;lane<256u;lane+=1u){let ac=activePrefix[lane];let rc=retiredPrefix[lane];
      let cc=corePrefix[lane];let hc=haloPrefix[lane];let nc=activatedPrefix[lane];
      let tac=tileActivePrefix[lane];let trc=tileRetiredPrefix[lane];
      activePrefix[lane]=a;retiredPrefix[lane]=r;corePrefix[lane]=c;haloPrefix[lane]=h;
      activatedPrefix[lane]=n;tileActivePrefix[lane]=ta;tileRetiredPrefix[lane]=tr;
      a+=ac;r+=rc;c+=cc;h+=hc;n+=nc;ta+=tac;tr+=trc;}
    worklist[0]=a;worklist[4]=r;worklist[8]=c;worklist[9]=h;worklist[10]=n;worklist[11]=r;
    tileWorklist[0]=ta;tileWorklist[4]=tr;
  }
  workgroupBarrier();var activeAt=activePrefix[lid];var retiredAt=retiredPrefix[lid];
  for(var logical=brickFirst;logical<brickEnd;logical+=1u){let marked=atomicLoad(&states[logical]);
    let resident=(marked&RESIDENT)!=0u;let was=(marked&WAS_RESIDENT)!=0u;
    if(resident){worklist[HEADER+activeAt*2u]=logical;worklist[HEADER+activeAt*2u+1u]=logical;activeAt+=1u;}
    else if(was){worklist[HEADER+brickCap*2u+retiredAt*2u]=logical;
      worklist[HEADER+brickCap*2u+retiredAt*2u+1u]=logical;retiredAt+=1u;}
  }
  var tileActiveAt=tileActivePrefix[lid];var tileRetiredAt=tileRetiredPrefix[lid];
  for(var key=tileFirst;key<tileEnd;key+=1u){let live=atomicLoad(&tileStates[key])!=0u;
    let was=publishedTileStates[key]!=0u;if(live){tileWorklist[HEADER+tileActiveAt]=key;tileActiveAt+=1u;}
    else if(was){tileWorklist[HEADER+tileCap+tileRetiredAt]=key;tileRetiredAt+=1u;}}
  workgroupBarrier();if(lid==0u){finishHeaders();}
  storageBarrier();workgroupBarrier();commitCandidateGeneration(lid,256u);
}
`;

/**
 * Sparse-key counterpart. The same single-owner schedule rebuilds the
 * open-addressed records deterministically and leaves tombstones intact, so a
 * saturated brick or tile pool rejects the entire candidate generation.
 */
export const sparseFineSeedCandidateResidencyShader = makeFineSeedCandidateShaderPrelude() + /* wgsl */ `
fn hashKey(key:u32)->u32{var x=key*747796405u+2891336453u;x=((x>>((x>>28u)+4u))^x)*277803737u;return (x>>22u)^x;}
fn brickSlots()->u32{return arrayLength(&states)/2u;}fn tileSlots()->u32{return arrayLength(&tileStates)/2u;}
fn worklistCapacity()->u32{return (arrayLength(&worklist)-HEADER)/4u;}
fn tileWorklistCapacity()->u32{return (arrayLength(&tileWorklist)-HEADER)/2u;}
// The sparse publisher deliberately retains a single owner for insertion:
// first occurrence still determines every open-addressed slot and tombstone
// choice. Compact fine leaves repeatedly name the same brick/tile, though, so
// memoize successful probes for this publication. A collision merely evicts a
// memo entry and falls back to the canonical probe below; it cannot alter the
// table or publication order.
const CLAIM_CACHE_SIZE=128u;
var<workgroup> brickClaimKeys:array<u32,128>;var<workgroup> brickClaimSlots:array<u32,128>;
var<workgroup> tileClaimKeys:array<u32,128>;var<workgroup> tileClaimSlots:array<u32,128>;
var<workgroup> tileRingKeys:array<u32,128>;
fn claimBrick(logical:u32)->u32{let encoded=logical+1u;let cap=brickSlots();if(encoded==0u||encoded==INVALID||cap==0u){return INVALID;}
  let start=hashKey(logical)%cap;var tombstone=INVALID;
  for(var probe=0u;probe<cap;probe++){let slot=(start+probe)%cap;let key=states[slot*2u];
    if(key==encoded){return slot;}if(key==INVALID&&tombstone==INVALID){tombstone=slot;}
    if(key==0u){let destination=select(slot,tombstone,tombstone!=INVALID);states[destination*2u]=encoded;return destination;}}
  if(tombstone!=INVALID){states[tombstone*2u]=encoded;return tombstone;}return INVALID;
}
fn claimTile(logical:u32)->u32{let encoded=logical+1u;let cap=tileSlots();if(encoded==0u||encoded==INVALID||cap==0u){return INVALID;}
  let start=hashKey(logical)%cap;var tombstone=INVALID;
  for(var probe=0u;probe<cap;probe++){let slot=(start+probe)%cap;let key=tileStates[slot*2u];
    if(key==encoded){return slot;}if(key==INVALID&&tombstone==INVALID){tombstone=slot;}
    if(key==0u){let destination=select(slot,tombstone,tombstone!=INVALID);tileStates[destination*2u]=encoded;return destination;}}
  if(tombstone!=INVALID){tileStates[tombstone*2u]=encoded;return tombstone;}return INVALID;
}
fn claimBrickCached(logical:u32)->u32{
  let encoded=logical+1u;let cache=hashKey(logical)&(CLAIM_CACHE_SIZE-1u);
  if(brickClaimKeys[cache]==encoded){let slot=brickClaimSlots[cache];
    if(slot<brickSlots()&&states[slot*2u]==encoded){return slot;}}
  let slot=claimBrick(logical);if(slot!=INVALID){brickClaimKeys[cache]=encoded;brickClaimSlots[cache]=slot;}return slot;
}
fn claimTileCached(logical:u32)->u32{
  let encoded=logical+1u;let cache=hashKey(logical)&(CLAIM_CACHE_SIZE-1u);
  if(tileClaimKeys[cache]==encoded){let slot=tileClaimSlots[cache];
    if(slot<tileSlots()&&tileStates[slot*2u]==encoded){return slot;}}
  let slot=claimTile(logical);if(slot!=INVALID){tileClaimKeys[cache]=encoded;tileClaimSlots[cache]=slot;}return slot;
}
fn markTileRing(key:u32)->bool{
  let encodedCenter=key+1u;let ringCache=hashKey(key)&(CLAIM_CACHE_SIZE-1u);
  if(tileRingKeys[ringCache]==encodedCenter){return true;}
  let td=vec3i(params.tiling.yzw);let q=vec3i(i32(key%params.tiling.y),
    i32((key/params.tiling.y)%params.tiling.z),i32(key/(params.tiling.y*params.tiling.z)));
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
    let n=q+vec3i(x,y,z);if(any(n<vec3i(0))||any(n>=td)){continue;}
    let logical=u32(n.x)+params.tiling.y*(u32(n.y)+params.tiling.z*u32(n.z));let slot=claimTileCached(logical);
    if(slot==INVALID){return false;}tileStates[slot*2u+1u]=1u;
  }}}tileRingKeys[ringCache]=encodedCenter;return true;
}
var<workgroup> sparsePrefix0:array<u32,256>;var<workgroup> sparsePrefix1:array<u32,256>;
var<workgroup> sparsePrefix2:array<u32,256>;var<workgroup> sparsePrefix3:array<u32,256>;
var<workgroup> sparsePrefix4:array<u32,256>;var<workgroup> sparseErrors:array<u32,256>;
var<workgroup> sparseError:u32;var<workgroup> sparseAccepted:u32;
@compute @workgroup_size(256) fn publishFineSeedCandidateResidency(@builtin(local_invocation_index)lid:u32){
  for(var i=lid;i<arrayLength(&worklist);i+=256u){worklist[i]=0u;}
  for(var i=lid;i<arrayLength(&tileWorklist);i+=256u){tileWorklist[i]=0u;}
  if(lid<CLAIM_CACHE_SIZE){brickClaimKeys[lid]=0u;brickClaimSlots[lid]=INVALID;
    tileClaimKeys[lid]=0u;tileClaimSlots[lid]=INVALID;tileRingKeys[lid]=0u;}
  if(lid==0u){sparseError=0u;sparseAccepted=select(0u,1u,producerAccepted());}
  workgroupBarrier();if(workgroupUniformLoad(&sparseAccepted)==0u){return;}
  for(var slot=lid;slot<brickSlots();slot+=256u){let encoded=publishedStates[slot*2u];
    let old=publishedStates[slot*2u+1u];states[slot*2u]=encoded;
    let was=encoded!=0u&&encoded!=INVALID&&(old&RESIDENT)!=0u;
    states[slot*2u+1u]=select(0u,WAS_RESIDENT,was)|(min(0xffffu,(old>>16u)+1u)<<16u);}
  for(var slot=lid;slot<tileSlots();slot+=256u){tileStates[slot*2u]=publishedTileStates[slot*2u];
    tileStates[slot*2u+1u]=0u;}storageBarrier();workgroupBarrier();
  if(lid==0u){let tileCells=params.dimsBrick.w*params.tiling.x;let td=vec3i(params.tiling.yzw);
    let logicalTileCount=params.tiling.y*params.tiling.z*params.tiling.w;
    for(var key=0u;key<logicalTileCount&&sparseError==0u;key++){
      if(!pressureBoundaryTile(key)){continue;}
      let slot=claimTile(key);if(slot==INVALID){sparseError=7u;break;}
      tileStates[slot*2u+1u]=1u;
    }
    for(var row=0u;row<arrayLength(&leaves)&&sparseError==0u;row++){let leaf=leaves[row];
      if((leaf.flags&LIVE)==0u||leaf.size==0u){continue;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);
      if(any(a>=params.dimsBrick.xyz)){sparseError=9u;break;}let first=vec3i(a/tileCells);
      let last=vec3i(min(params.dimsBrick.xyz-vec3u(1u),a+vec3u(leaf.size-1u))/tileCells);
      for(var z=first.z-1;z<=last.z+1&&sparseError==0u;z++){for(var y=first.y-1;y<=last.y+1&&sparseError==0u;y++){
        for(var x=first.x-1;x<=last.x+1;x++){let q=vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=td)){continue;}
          let key=u32(q.x)+params.tiling.y*(u32(q.y)+params.tiling.z*u32(q.z));let slot=claimTile(key);
          if(slot==INVALID){sparseError=7u;break;}tileStates[slot*2u+1u]=1u;
        }}}
    }
    for(var i=0u;i<candidateControl[0]&&sparseError==0u;i++){
      if(i>=arrayLength(&candidates)){sparseError=1u;break;}let c=candidates[i];
      if(c.row>=arrayLength(&leaves)){sparseError=2u;break;}let leaf=leaves[c.row];
      if(leaf.size==0u){sparseError=3u;break;}let a=vec3u(leaf.originX,leaf.originY,leaf.originZ);
      if(any(a>=params.dimsBrick.xyz)){sparseError=4u;break;}let first=a/params.dimsBrick.w;
      let last=min(params.brickDimsCapacity.xyz-vec3u(1u),(a+vec3u(leaf.size-1u))/params.dimsBrick.w);
      let bits=select(HALO,CORE,(c.flags&CORE)!=0u);
      for(var z=first.z;z<=last.z&&sparseError==0u;z++){for(var y=first.y;y<=last.y&&sparseError==0u;y++){
        for(var x=first.x;x<=last.x;x++){let logical=x+params.brickDimsCapacity.x*(y+params.brickDimsCapacity.y*z);
          let slot=claimBrickCached(logical);if(slot==INVALID){sparseError=6u;break;}states[slot*2u+1u]|=bits;
        }}}
    }
  }storageBarrier();workgroupBarrier();if(workgroupUniformLoad(&sparseError)!=0u){return;}
  let brickChunk=(brickSlots()+255u)/256u;let brickFirst=min(brickSlots(),lid*brickChunk);
  let brickEnd=min(brickSlots(),brickFirst+brickChunk);var activeCount=0u;var retired=0u;
  var coreCount=0u;var haloCount=0u;var activated=0u;var localError=0u;
  for(var slot=brickFirst;slot<brickEnd;slot++){let encoded=states[slot*2u];
    if(encoded==0u||encoded==INVALID){continue;}let logical=encoded-1u;
    if(logical>=params.brickDimsCapacity.w){localError=8u;continue;}let marked=states[slot*2u+1u];
    let desired=(marked&(CORE|HALO))!=0u;let was=(marked&WAS_RESIDENT)!=0u;
    let dry=select(marked>>16u,0u,desired);let persistent=persistentLiquid();
    let resident=select(desired||(was&&dry<=u32(params.settings.y)),was||desired,persistent);
    let core=(marked&CORE)!=0u;let flags=select(0u,RESIDENT,resident)|select(0u,CORE,core)
      |select(0u,HALO,resident&&!core)|select(0u,ACTIVATED,resident&&!was)|select(0u,WAS_RESIDENT,was);
    states[slot*2u+1u]=flags|(dry<<16u);activeCount+=select(0u,1u,resident);
    retired+=select(0u,1u,!resident&&was);coreCount+=select(0u,1u,resident&&core);
    haloCount+=select(0u,1u,resident&&!core);activated+=select(0u,1u,resident&&!was);
  }
  sparsePrefix0[lid]=activeCount;sparsePrefix1[lid]=retired;sparsePrefix2[lid]=coreCount;
  sparsePrefix3[lid]=haloCount;sparsePrefix4[lid]=activated;sparseErrors[lid]=localError;
  storageBarrier();workgroupBarrier();
  if(lid==0u){var a=0u;var r=0u;var c=0u;var h=0u;var n=0u;
    for(var lane=0u;lane<256u;lane++){let ac=sparsePrefix0[lane];let rc=sparsePrefix1[lane];
      let cc=sparsePrefix2[lane];let hc=sparsePrefix3[lane];let nc=sparsePrefix4[lane];
      sparsePrefix0[lane]=a;sparsePrefix1[lane]=r;a+=ac;r+=rc;c+=cc;h+=hc;n+=nc;
      if(sparseErrors[lane]!=0u&&sparseError==0u){sparseError=sparseErrors[lane];}}
    if(a>worklistCapacity()||r>worklistCapacity()){sparseError=5u;}
    worklist[0]=a;worklist[4]=r;worklist[8]=c;worklist[9]=h;worklist[10]=n;worklist[11]=r;
  }workgroupBarrier();if(workgroupUniformLoad(&sparseError)!=0u){return;}
  var activeAt=sparsePrefix0[lid];var retiredAt=sparsePrefix1[lid];let cap=worklistCapacity();
  for(var slot=brickFirst;slot<brickEnd;slot++){let encoded=states[slot*2u];
    if(encoded==0u||encoded==INVALID){continue;}let logical=encoded-1u;let marked=states[slot*2u+1u];
    if((marked&RESIDENT)!=0u){worklist[HEADER+activeAt*2u]=logical;
      worklist[HEADER+activeAt*2u+1u]=logical;activeAt+=1u;}
    else{if((marked&WAS_RESIDENT)!=0u){worklist[HEADER+cap*2u+retiredAt*2u]=logical;
      worklist[HEADER+cap*2u+retiredAt*2u+1u]=logical;retiredAt+=1u;}
      states[slot*2u+1u]=0u;states[slot*2u]=INVALID;}
  }storageBarrier();workgroupBarrier();
  if(lid==0u){for(var item=0u;item<worklist[0]&&sparseError==0u;item++){
      let logical=worklist[HEADER+item*2u];
      let b=vec3u(logical%params.brickDimsCapacity.x,(logical/params.brickDimsCapacity.x)%params.brickDimsCapacity.y,
        logical/(params.brickDimsCapacity.x*params.brickDimsCapacity.y));
      let key=(b.x/params.tiling.x)+params.tiling.y*((b.y/params.tiling.x)+params.tiling.z*(b.z/params.tiling.x));
      if(!markTileRing(key)){sparseError=7u;}}
  }storageBarrier();workgroupBarrier();if(workgroupUniformLoad(&sparseError)!=0u){return;}
  let tileChunk=(tileSlots()+255u)/256u;let tileFirst=min(tileSlots(),lid*tileChunk);
  let tileEnd=min(tileSlots(),tileFirst+tileChunk);var tileActive=0u;var tileRetired=0u;
  for(var slot=tileFirst;slot<tileEnd;slot++){let encoded=tileStates[slot*2u];
    if(encoded==0u||encoded==INVALID){continue;}let live=tileStates[slot*2u+1u]!=0u;
    let was=publishedTileStates[slot*2u]==encoded;tileActive+=select(0u,1u,live);
    tileRetired+=select(0u,1u,!live&&was);}
  sparsePrefix0[lid]=tileActive;sparsePrefix1[lid]=tileRetired;workgroupBarrier();
  if(lid==0u){var a=0u;var r=0u;for(var lane=0u;lane<256u;lane++){
      let ac=sparsePrefix0[lane];let rc=sparsePrefix1[lane];sparsePrefix0[lane]=a;sparsePrefix1[lane]=r;
      a+=ac;r+=rc;}if(a>tileWorklistCapacity()||r>tileWorklistCapacity()){sparseError=5u;}
    tileWorklist[0]=a;tileWorklist[4]=r;
  }workgroupBarrier();if(workgroupUniformLoad(&sparseError)!=0u){return;}
  var tileActiveAt=sparsePrefix0[lid];var tileRetiredAt=sparsePrefix1[lid];let tc=tileWorklistCapacity();
  for(var slot=tileFirst;slot<tileEnd;slot++){let encoded=tileStates[slot*2u];
    if(encoded==0u||encoded==INVALID){continue;}let key=encoded-1u;let live=tileStates[slot*2u+1u]!=0u;
    let was=publishedTileStates[slot*2u]==encoded;if(live){tileWorklist[HEADER+tileActiveAt]=key;tileActiveAt+=1u;}
    else{if(was){tileWorklist[HEADER+tc+tileRetiredAt]=key;tileRetiredAt+=1u;}
      tileStates[slot*2u+1u]=0u;tileStates[slot*2u]=INVALID;}}
  storageBarrier();workgroupBarrier();if(lid==0u){finishHeaders();}
}
`;

export const fineSeedCandidateCommitShader = /* wgsl */ `
struct Params { dimsBrick:vec4u, brickDimsCapacity:vec4u, settings:vec4f, tiling:vec4u }
@group(0) @binding(0) var<storage,read_write> publishedStates:array<u32>;
@group(0) @binding(1) var<storage,read> candidateStates:array<u32>;
@group(0) @binding(2) var<storage,read_write> publishedWorklist:array<u32>;
@group(0) @binding(3) var<storage,read> candidateWorklist:array<u32>;
@group(0) @binding(4) var<storage,read_write> publishedTileStates:array<u32>;
@group(0) @binding(5) var<storage,read> candidateTileStates:array<u32>;
@group(0) @binding(6) var<storage,read_write> publishedTileWorklist:array<u32>;
@group(0) @binding(7) var<storage,read> candidateTileWorklist:array<u32>;
const COMMIT=0xc01117edu;
@compute @workgroup_size(64) fn commitFineSeedCandidates(@builtin(global_invocation_id) gid:vec3u){
  if(candidateTileWorklist[11]!=COMMIT){return;}let i=gid.x;
  if(i<arrayLength(&publishedStates)){publishedStates[i]=candidateStates[i];}
  if(i<arrayLength(&publishedWorklist)){publishedWorklist[i]=candidateWorklist[i];}
  if(i<arrayLength(&publishedTileStates)){publishedTileStates[i]=candidateTileStates[i];}
  if(i<arrayLength(&publishedTileWorklist)){publishedTileWorklist[i]=select(candidateTileWorklist[i],0u,i==11u);}
}
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
  private readonly leafIndices: GPUBuffer;
  private readonly leafStatesBuffer: GPUBuffer;
  private readonly currentAllocationPlan: FluidBrickResidencyAllocationPlan;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly fineSeedCandidateLayout: GPUBindGroupLayout;
  private readonly fineSeedCandidatePublishLayout: GPUBindGroupLayout;
  private readonly fineSeedCandidateCommitLayout: GPUBindGroupLayout;
  private classifyPipeline!: GPUComputePipeline;
  private classifySweptPipeline!: GPUComputePipeline;
  private expandDownstreamPipeline!: GPUComputePipeline;
  private emitWorklistPipeline!: GPUComputePipeline;
  private emitTopologyTilesPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private fineSeedCandidatePipelines: GPUComputePipeline[] = [];
  private commitFineSeedCandidatesPipeline!: GPUComputePipeline;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly surfaceLayout: GPUPipelineLayout;
  private readonly surfacePublishLayout: GPUPipelineLayout;
  private readonly commitPipelineLayout: GPUPipelineLayout;
  private shaderModule?: GPUShaderModule;
  private surfaceModule?: GPUShaderModule;
  private commitModule?: GPUShaderModule;
  private readonly fineSeedCandidateEntryPoints: readonly string[];
  private readonly pipelinesDeferred: boolean;
  private readonly fineSeedBindGroupCache: {
    readonly leaves: GPUBuffer;
    readonly candidates: GPUBuffer;
    readonly candidateControl: GPUBuffer;
    readonly candidate: GPUBindGroup;
    readonly finalPublish: GPUBindGroup;
    readonly commit: GPUBindGroup;
  }[] = [];
  private destroyed = false;

  constructor(
    device: GPUDevice,
    dimensions: readonly [number, number, number],
    cellSize: readonly [number, number, number],
    options: FluidBrickResidencyOptions = {},
  ) {
    this.device = device;
    this.pipelinesDeferred = true;
    this.brickSize = options.brickSize ?? 8;
    if (this.brickSize !== 4 && this.brickSize !== 8) throw new RangeError("Fluid brick size must be 4 or 8");
    for (const [axis, value] of dimensions.entries()) if (!Number.isInteger(value) || value < 1) throw new RangeError(`Fluid dimension ${axis} must be positive`);
    for (const value of cellSize) if (!(value > 0) || !Number.isFinite(value)) throw new RangeError("Fluid cell size must be positive and finite");
    this.brickDimensions = dimensions.map((value) => Math.ceil(value / this.brickSize)) as [number, number, number];
    this.capacity = this.brickDimensions[0] * this.brickDimensions[1] * this.brickDimensions[2];
    const candidateOnly = options.fineSeedCandidatesOnly === true;
    const explicitMapping = options.leafIndices;
    if (candidateOnly && explicitMapping) {
      throw new RangeError("Fine-seed-candidate-only residency requires implicit brick/leaf identity");
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
      candidateOnly,
      options.fineSeedCandidateBrickCapacity,
      options.fineSeedCandidateTileCapacity,
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
    // COPY_SRC is failure-only QA: recurring topology rejection must be able
    // to prove that compact dirty-tile membership and the published worklist
    // name the same generation without perturbing either authority.
    this.tileStates = buffer("Persistent topology tile activity", this.currentAllocationPlan.tileStateBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.candidateStates = buffer("Candidate fluid brick page states", this.currentAllocationPlan.stateBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateWorklist = buffer("Candidate fluid brick worklists", worklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateTileStates = buffer("Candidate topology tile activity",
      this.currentAllocationPlan.tileStateBytes + (this.currentAllocationPlan.sparseKeyPools ? 0 : 4),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.candidateTileWorklist = buffer("Candidate topology tile worklists", tileWorklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.leafIndices = buffer("Fluid brick to sparse leaf mapping", mapping.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mapping);
    this.leafStatesBuffer = buffer(
      candidateOnly ? "Unused sparse leaf residency fallback" : "Sparse leaf fluid residency",
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
    const schedulerFlags = (options.includeLiquidInterior ? 1 : 0)
      | (options.includePressureBoundarySupport ? 2 : 0)
      | (options.pressureBoundaryTopClosed ? 4 : 0)
      | (options.includeWholeDomainPressureSupport ? 8 : 0)
      | (options.fluidGatedBoundarySupport ? 16 : 0);
    floats.set([haloCells * Math.max(...cellSize), retireAfterFrames, 0, schedulerFlags], 8);
    device.queue.writeBuffer(this.params, 0, parameterData);
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
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.fineSeedCandidateLayout=device.createBindGroupLayout({label:"Fine-seed brick residency candidate layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
    ]});
    this.fineSeedCandidatePublishLayout=device.createBindGroupLayout({label:"Fine-seed brick residency final publication layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:11,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
    this.surfaceLayout=device.createPipelineLayout({bindGroupLayouts:[this.fineSeedCandidateLayout]});
    this.surfacePublishLayout=device.createPipelineLayout({bindGroupLayouts:[this.fineSeedCandidatePublishLayout]});
    this.fineSeedCandidateEntryPoints = this.currentAllocationPlan.sparseKeyPools
      ? ["publishFineSeedCandidateResidency"]
      : ["prepareFineSeedCandidateResidency", "markFineSeedCandidateResidency",
        "resolveFineSeedCandidateResidency", "publishFineSeedCandidateResidency"];
    this.fineSeedCandidateCommitLayout=device.createBindGroupLayout({label:"Fine-seed brick residency commit layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    ]});
    this.commitPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.fineSeedCandidateCommitLayout]});
    // The texture binding changes with the projection's ping-pong surface and
    // is therefore created in encode(). Keep the common resources resident.
  }

  private mainDescriptor(entryPoint: string, label: string): GPUComputePipelineDescriptor {
    this.shaderModule ??= this.device.createShaderModule({
      label: "Fluid brick residency shader", code: fluidBrickResidencyShader,
    });
    return { label, layout: this.pipelineLayout, compute: { module: this.shaderModule, entryPoint } };
  }

  private surfaceDescriptor(entryPoint: string, index: number): GPUComputePipelineDescriptor {
    this.surfaceModule ??= this.device.createShaderModule({ label: "Fine-seed brick residency shader",
      code: this.currentAllocationPlan.sparseKeyPools
        ? sparseFineSeedCandidateResidencyShader : fineSeedCandidateResidencyShader });
    return { label: `${entryPoint} · deterministic fine-seed-candidate residency`,
      layout: !this.currentAllocationPlan.sparseKeyPools && index === 3
        ? this.surfacePublishLayout : this.surfaceLayout,
      compute: { module: this.surfaceModule, entryPoint } };
  }

  private commitDescriptor(): GPUComputePipelineDescriptor {
    this.commitModule ??= this.device.createShaderModule({
      label: "Fine-seed brick residency commit shader", code: fineSeedCandidateCommitShader,
    });
    return { label: "Commit fine-seed-candidate residency", layout: this.commitPipelineLayout,
      compute: { module: this.commitModule, entryPoint: "commitFineSeedCandidates" } };
  }

  private readonly mainPipelineDefinitions = [
    ["classify", "Classify fluid brick residency"],
    ["classifySwept", "Classify fluid brick residency with swept support"],
    ["expandDownstream", "Expand fluid brick residency downstream"],
    ["emitWorklist", "Emit fluid brick worklists"],
    ["emitTopologyTiles", "Emit topology tile worklists"],
    ["finalize", "Finalize fluid brick worklists"],
  ] as const;

  private assignMainPipeline(index: number, pipeline: GPUComputePipeline): void {
    if (index === 0) this.classifyPipeline = pipeline;
    else if (index === 1) this.classifySweptPipeline = pipeline;
    else if (index === 2) this.expandDownstreamPipeline = pipeline;
    else if (index === 3) this.emitWorklistPipeline = pipeline;
    else if (index === 4) this.emitTopologyTilesPipeline = pipeline;
    else if (index === 5) this.finalizePipeline = pipeline;
    else throw new RangeError(`Unknown fluid-residency pipeline index ${index}`);
  }

  initializationTasks(): GPUInitializationTask[] {
    if (!this.pipelinesDeferred) return [];
    const main = this.mainPipelineDefinitions.map(([entryPoint, label], index) => ({
      id: `octree.residency.pipeline.${entryPoint}`, phase: "adaptive-topology" as const,
      label: `Compile residency · ${label}`,
      run: async () => this.assignMainPipeline(index,
        await this.device.createComputePipelineAsync(this.mainDescriptor(entryPoint, label))),
    }));
    const surface = this.fineSeedCandidateEntryPoints.map((entryPoint, index) => ({
      id: `octree.residency.pipeline.${entryPoint}`, phase: "adaptive-topology" as const,
      label: `Compile residency · ${entryPoint}`,
      run: async () => { this.fineSeedCandidatePipelines[index] =
        await this.device.createComputePipelineAsync(this.surfaceDescriptor(entryPoint, index)); },
    }));
    return [...main, ...surface, {
      id: "octree.residency.pipeline.commitFineSeedCandidates",
      phase: "adaptive-topology" as const,
      label: "Compile fine-seed residency commit",
      run: async () => { this.commitFineSeedCandidatesPipeline =
        await this.device.createComputePipelineAsync(this.commitDescriptor()); },
    }];
  }

  /** GPU-owned per-brick state words, consumable by sibling schedulers (atlas). */
  get stateBuffer(): GPUBuffer { return this.states; }
  /** Persistent topology-tile state, shared with the analytic cold publisher. */
  get topologyTileStateBuffer(): GPUBuffer { return this.tileStates; }
  get leafStates(): GPUBuffer { return this.leafStatesBuffer; }
  get allocationPlan(): FluidBrickResidencyAllocationPlan { return this.currentAllocationPlan; }
  get allocatedBytes(): number { return this.currentAllocationPlan.allocatedBytes; }

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
    const broker = new PassBroker(encoder);
    broker.clearBuffer(this.worklist, 0, (FLUID_BRICK_WORKLIST_HEADER_WORDS - 1) * 4);
    broker.clearBuffer(this.tileWorklist, 0, FLUID_TILE_WORKLIST_HEADER_WORDS * 4);
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
    const classify = broker.compute({ label: "Classify evolving fluid bricks" });
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
    const finalize = broker.compute({ label: "Finalize evolving fluid brick worklists" });
    finalize.setPipeline(this.finalizePipeline);
    finalize.setBindGroup(0, bindGroup);
    finalize.dispatchWorkgroups(1);
    broker.fence("fluid brick worklists published");
  }

  /** Derive the brick/tile scheduler ABI from compact fine-seed leaves. */
  encodeFineSeedCandidates(
    encoder: GPUCommandEncoder,
    leaves: GPUBuffer,
    candidates: GPUBuffer,
    candidateControl: GPUBuffer,
  ): void {
    if (this.destroyed) return;
    // The deterministic schedule overwrites generation B completely. Failed,
    // stale, malformed, or capacity-exhausted generations never publish the
    // commit marker and therefore leave generation A byte-for-byte untouched.
    const broker = new PassBroker(encoder);
    let groups = this.fineSeedBindGroupCache.find((entry) => entry.leaves === leaves
      && entry.candidates === candidates && entry.candidateControl === candidateControl);
    if (!groups) {
      groups = {
        leaves, candidates, candidateControl,
        candidate: this.device.createBindGroup({
          label: "Fine-seed brick residency candidate bindings",
          layout: this.fineSeedCandidateLayout, entries: [
            {binding:0,resource:{buffer:this.states}},{binding:1,resource:{buffer:this.candidateStates}},
            {binding:2,resource:{buffer:this.candidateWorklist}},{binding:3,resource:{buffer:this.tileStates}},
            {binding:4,resource:{buffer:this.candidateTileStates}},{binding:5,resource:{buffer:this.candidateTileWorklist}},
            {binding:6,resource:{buffer:leaves}},{binding:7,resource:{buffer:candidates}},
            {binding:8,resource:{buffer:candidateControl}},{binding:9,resource:{buffer:this.tileWorklist}},
            {binding:10,resource:{buffer:this.params}},
          ],
        }),
        finalPublish: this.device.createBindGroup({
          label: "Fine-seed brick residency final publication bindings",
          layout: this.fineSeedCandidatePublishLayout, entries: [
            {binding:0,resource:{buffer:this.states}},{binding:1,resource:{buffer:this.candidateStates}},
            {binding:2,resource:{buffer:this.candidateWorklist}},{binding:3,resource:{buffer:this.tileStates}},
            {binding:4,resource:{buffer:this.candidateTileStates}},{binding:5,resource:{buffer:this.candidateTileWorklist}},
            {binding:7,resource:{buffer:candidates}},{binding:8,resource:{buffer:candidateControl}},
            {binding:9,resource:{buffer:this.tileWorklist}},
            {binding:10,resource:{buffer:this.params}},{binding:11,resource:{buffer:this.worklist}},
          ],
        }),
        commit: this.device.createBindGroup({
          label: "Fine-seed brick residency commit bindings",
          layout: this.fineSeedCandidateCommitLayout, entries: [
            {binding:0,resource:{buffer:this.states}},{binding:1,resource:{buffer:this.candidateStates}},
            {binding:2,resource:{buffer:this.worklist}},{binding:3,resource:{buffer:this.candidateWorklist}},
            {binding:4,resource:{buffer:this.tileStates}},{binding:5,resource:{buffer:this.candidateTileStates}},
            {binding:6,resource:{buffer:this.tileWorklist}},{binding:7,resource:{buffer:this.candidateTileWorklist}},
          ],
        }),
      };
      this.fineSeedBindGroupCache.push(groups);
    }
    const bindGroup = groups.candidate;
    const publish=broker.compute({label:"Publish deterministic fine-seed brick residency"});
    publish.setBindGroup(0,bindGroup);
    if (this.currentAllocationPlan.sparseKeyPools) {
      publish.setPipeline(this.fineSeedCandidatePipelines[0]!);publish.dispatchWorkgroups(1);
    } else {
      const prepareWords = Math.max(this.worklistByteLength, this.tileWorklistByteLength,
        this.currentAllocationPlan.stateBytes, this.currentAllocationPlan.tileStateBytes + 4) / 4;
      const markItems = Math.max(leaves.size / 48, candidates.size / 8);
      const dispatches = [
        Math.max(1, Math.ceil(prepareWords / 256)),
        Math.max(1, Math.ceil(markItems / 256)),
        Math.max(1, Math.ceil(this.publicationCapacity / 256)),
      ];
      this.fineSeedCandidatePipelines.slice(0,3).forEach((pipeline, index) => {
        publish.setPipeline(pipeline);publish.dispatchWorkgroups(dispatches[index]!);
      });
      publish.setPipeline(this.fineSeedCandidatePipelines[3]!);
      publish.setBindGroup(0,groups.finalPublish);
      publish.dispatchWorkgroups(1);
    }
    if (this.currentAllocationPlan.sparseKeyPools) {
      const commit=broker.compute({label:"Commit fine-seed brick residency"});
      commit.setPipeline(this.commitFineSeedCandidatesPipeline);commit.setBindGroup(0,groups.commit);
      commit.dispatchWorkgroups(Math.ceil(Math.max(this.worklistByteLength,this.tileWorklistByteLength,
        this.currentAllocationPlan.stateBytes,this.currentAllocationPlan.tileStateBytes)/4/64));
    }
    broker.fence("fine-seed brick residency committed");
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
    this.leafIndices.destroy();
    this.leafStatesBuffer.destroy();
    this.params.destroy();
  }
}
