import {
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_NO_OWNER,
  SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII,
  sparseBrickBandedLeafCodecWGSL,
  sparseBrickSceneGeometryCodecWGSL,
  sparseBrickSceneIdentityCodecWGSL,
  type SparseBrickLeafPayloadMode,
  type SparseBrickOctreeGPU,
  type SparseBrickPayloadProfileName,
  type SparseBrickPlan,
  type SparseBrickSceneGeometryFormat,
  type SparseBrickSize,
} from "../construction/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "../construction/svo-brick-occupancy";
import { completeCooperativeBuild } from "../../../core/cooperative-build";
import { svoEnvironmentLightingWGSL } from "./svo-environment-lighting";
import { svoLightWGSL } from "../../contracts/svo-light-abi";
import { svoMaterialWGSL } from "../../contracts/svo-material-abi";
import {
  SVO_NODE_MIP_LAYOUT,
  SVO_NODE_MIP_OPACITY_STORAGE,
  raiseSvoNodeMipSeedToFloor,
  svoNodeMipOpacityChannels,
  svoNodeMipSeedKey,
  type SvoNodeMipCoordinate,
  type SvoNodeMipSeedPage,
} from "./svo-node-mip-pyramid";
import { svoTetrahedralRadianceWGSL } from "./svo-tetrahedral-radiance";
import { SVO_NODE_MIP_MAXIMUM_LEVELS, svoNodeMipPageHashWGSL } from "./svo-node-mip-page-hash";
import {
  svoBandedReconstructionEnabled,
  svoSolidDirectOcclusionEnabled,
  svoSolidMarchOffsetFixEnabled,
} from "../construction/svo-banded-leaf-payload";
import { liveSvoDerivedPageValidityWGSL } from "./webgpu-svo-live-derived-cache";
import type { WebGpuLiveSvoNodeMipGpuTarget } from "./webgpu-svo-node-mip-pyramid";
import {
  LIVE_SVO_RADIANCE_STORAGE,
  liveSvoRadianceBytesPerTexel,
  type WebGpuLiveSvoTetrahedralRadianceGpuTarget,
} from "./webgpu-svo-tetrahedral-radiance";

export const LIVE_SVO_DERIVED_WORKLIST = Object.freeze({
  countWord: 0,
  generationWord: 1,
  levelWord: 2,
  recordOffsetWord: 3,
  /** Words 4-7 name the queue ring the batch was taken from; see `prepareRound`. */
  ringRecordsWord: 4,
  ringHeadWord: 5,
  ringCapacityWord: 6,
  ringKindWord: 7,
  headerWords: 8,
  recordWords: 12,
  destinationSlotWord: 0,
  sourcePageCoordinateWord: 1,
  sourceLeafWord: 4,
  /**
   * One record layout at every level: slot, page coordinate, eight children.
   *
   * The page coordinate used to be carried only by the base level and by the
   * radiance floor, because only those two needed to locate the cells they
   * cover. Every parent needs it now: a child page that is absent is no longer
   * proof of empty space — it can be the interior of a coarse leaf that
   * publishes at its own level — and reading that leaf means turning a texel
   * back into a world cell. Twelve words hold exactly one of each.
   */
  childSlotWord: 4,
  /** Historic alias; the radiance floor's layout is now every level's layout. */
  radianceBaseChildSlotWord: 4,
  invalidIndex: 0xffff_ffff,
  /**
   * A child slot that names a *leaf* rather than a page.
   *
   * `writeRecord` resolves an absent child page once, at plan time, into the
   * deepest active leaf covering it — which, because a page is emitted for
   * every leaf at that leaf's own level, is always a leaf whose extent contains
   * the whole child page. The reduction then samples that leaf's voxels
   * directly instead of averaging in a zero it would have no way to distinguish
   * from air. Slots are physical atlas indices and never reach the top bit.
   */
  leafSlotTag: 0x8000_0000,
} as const);

export const LIVE_SVO_RADIANCE_FEEDBACK = Object.freeze({
  /**
   * On, now that the radiance floor makes it affordable.
   *
   * The historic objection was cost and in-place mutation: the base pass ran
   * once per finest voxel, so a garden paid a full-resolution diffuse solve to
   * move a low-frequency field. With radiance capped three levels above the
   * leaf the base pass runs on ~500x fewer texels, and a phase that lands
   * mid-rotation now perturbs a 5 cm irradiance sample rather than a voxel.
   */
  enabledByDefault: true,
  /**
   * At the 0.85 transport ceiling, 24 rotations bound the unpropagated fixed-
   * point residual below 2.1% (0.85^24). Static scenes then go exactly idle.
   */
  settleCycleCount: 24,
  /** Frames a window stays open however few rotations the budget let it run. */
  settleFrameCount: 96,
  directionCount: 4,
  distanceCells: [2.5, 6, 18, 54] as const,
  /** Pure-white display materials remain contractive in the transport solve. */
  maximumTransportAlbedo: 0.85,
} as const);

/**
 * The fixed work of one lighting frame, and of one round inside it.
 *
 * Lighting is a queue the GPU owns: the planner appends pages to per-level
 * rings, and every frame runs the same dispatches over whatever the rings hold.
 * Nothing here is sized by an edit, no dispatch is indirect, and the host never
 * reads a count back — it only knows upper bounds, and plans frames from them.
 *
 * `pagesPerFrame` is the voxelizer's brick budget, deliberately: a frame of
 * voxelization repairs at most that many bricks, a brick owns at most one seed
 * page, and so the seeds of the chunk that has just landed are always built in
 * the frame they land in. It is also the size of one batch, so a round either
 * spends the frame's allowance or empties the ring it is on.
 */
export const LIVE_SVO_DERIVED_BUDGET = Object.freeze({
  pagesPerFrame: 2_048,
  /** `populate`, `selectFeedback`: 32 workgroups of 64, grid-striding their list. */
  plannerWorkgroups: 32,
  plannerThreads: 32 * 64,
  /** Build, gather and copy: 64 workgroups of 256, grid-striding the batch's texels. */
  buildWorkgroups: 64,
  buildThreads: 64 * 256,
  /**
   * Floor pages a rebuilt page's light can reach in one bounce: the gather's
   * farthest step (`distanceCells`, in floor texels) over a page's interior.
   */
  feedbackReachPages: Math.ceil(54 / SVO_NODE_MIP_LAYOUT.interiorSize),
} as const);

/**
 * The one batch a round builds: a header, then `capacity` records.
 *
 * Records may repeat no destination slot. Level zero supplies its virtual page
 * coordinate and resolves the deepest active leaf per texel; parent records
 * supply eight child slots in xyz-bit order. A record whose slot is `invalidIndex`
 * is one the planner declined to resolve, and builds nothing.
 */
export interface LiveSvoDerivedGpuBatch {
  buffer: GPUBuffer;
  capacity: number;
}

export interface WebGpuLiveSvoDerivedBuilderOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  radiance: WebGpuLiveSvoTetrahedralRadianceGpuTarget;
  /** vec4f per material ID: linear emissive RGB plus unused alpha. */
  materialEmission: GPUBuffer;
  /** Canonical SVO material table used for diffuse transport albedo. */
  materialPbr?: GPUBuffer;
  /** Selected image-free environment record used as the open-cone source. */
  environmentLighting?: GPUBuffer;
  /** Bounded authored light table; the first four records feed diffuse transport. */
  lights?: GPUBuffer;
  lightCount?: number;
  /** The planner's batch; every round builds what it holds. */
  batch: LiveSvoDerivedGpuBatch;
  /** Virtual mip levels the page table covers. */
  levelCount: number;
  /** GPU publication generation used to certify newly allocated empty pages. */
  generationSource: LiveSvoDerivedGenerationSource;
  /** Number of address-plan slots resident in the fixed atlas. */
  plannedPageCount: number;
  /** Finest canonical tree level used for cell-to-leaf traversal. */
  finestLevel: number;
  worldOrigin_m?: readonly [number, number, number];
  cellSize_m?: readonly [number, number, number];
  /** Previous-frame radiance feedback. Enabled when both optional lighting buffers are supplied. */
  radianceFeedback?: boolean;
  /** Optional compact scratch page grid; defaults to the smallest grid that fits one batch. */
  scratchAtlasPages?: readonly [number, number, number];
  /** Finest level that owns a radiance page. See `SVO_RADIANCE_LEVEL_FLOOR`. */
  radianceFloorLevel?: number;
  /**
   * Pages the deepest *radiance* level can hold.
   *
   * A batch of radiance pages can be no deeper than this, so where it is
   * smaller than a batch the radiance scratch is sized by it instead.
   */
  radianceScratchCapacity?: number;
  label?: string;
}

export interface LiveSvoDirtyLeafSource {
  buffer: GPUBuffer;
  countOffsetBytes: number;
  recordOffsetBytes: number;
  capacity: number;
  /** Source records contain leafIndex in word zero and use this stride. */
  recordStrideWords: number;
}

/**
 * A ring of every leaf whose voxels have been rewritten, appended as each lands.
 *
 * Unlike a {@link LiveSvoDirtyLeafSource}, which is a list its owner rewrites,
 * a journal is consumed by position: the planner keeps its own cursor on the
 * GPU and takes `[cursor, write)` once, so a leaf is lit when its chunk
 * finishes and never again for the chunks that follow it.
 */
export interface LiveSvoDirtyLeafJournal {
  buffer: GPUBuffer;
  /** Appends so far, modulo 2^32; never reset. */
  writeCursorOffsetBytes: number;
  /** Entry `k` is the leaf index at `recordOffsetBytes + (k % capacity) * recordStrideWords * 4`. */
  recordOffsetBytes: number;
  recordStrideWords: number;
  capacity: number;
  /**
   * The revision being voxelized and the last one finished. While they differ
   * more leaves are still to come, and pages above the ones just written wait.
   */
  requestedRevisionOffsetBytes: number;
  completedRevisionOffsetBytes: number;
}

/** One nonzero, monotonically increasing generation shared by every dirty lane. */
export interface LiveSvoDerivedGenerationSource {
  buffer: GPUBuffer;
  offsetBytes: number;
}

export interface WebGpuLiveSvoDerivedWorklistPlannerOptions {
  tree: SparseBrickOctreeGPU;
  nodeMips: WebGpuLiveSvoNodeMipGpuTarget;
  /** Leaves the scene voxelizer has rewritten; consumed once, in the frame they land. */
  sceneJournal: LiveSvoDirtyLeafJournal;
  /** Lists a solver step rewrites whole; consumed on the frames that step. */
  stepLeafSources: readonly LiveSvoDirtyLeafSource[];
  /** Supplies the radiance validity words and the first slot of the radiance pool. */
  radiance: WebGpuLiveSvoTetrahedralRadianceGpuTarget;
  /** Whether the builder gathers; without it no feedback rotation is ever started. */
  radianceFeedback: boolean;
  generationSource: LiveSvoDerivedGenerationSource;
  levelCount: number;
  finestLevel: number;
  /**
   * Worklist depth each level may reach, as a vector or as one figure for all.
   *
   * A vector is what the address plan supplies (`pageCapacityByLevel`), and the
   * arena lays its sections out by prefix sum over it. A scalar is the old
   * uniform shape, kept because it is the natural thing for a fixture to say.
   */
  pageCapacityPerLevel: number | readonly number[];
  /** Finest level that owns a radiance page; its records carry a page coordinate. */
  radianceFloorLevel?: number;
  label?: string;
}

/**
 * Where a coarse leaf's data lives in the pyramid, and the A/B that measures it.
 *
 * `own-level` (the default) gives every leaf exactly one page, at the level
 * whose texels are its voxels. `extent` is the shape that shipped: a coarse
 * leaf publishes a base page for every finest brick it covers, holding the same
 * `brickSize^3` values `8^p` times over. The two differ in what the *base*
 * level holds inside a coarse leaf and in nothing else — every parent level is
 * exact under both, because a reduction whose child page is absent reads the
 * leaf directly (`LIVE_SVO_DERIVED_WORKLIST.leafSlotTag`).
 *
 * Read once at module load, like `FLUID_SVO_NODE_MIP_APRON`, so the WGSL and
 * the CPU plan cannot disagree about which shape this process is building.
 */
export const LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE = "FLUID_SVO_COARSE_LEAF_PAGES";

export function liveSvoCoarseLeafPagePolicy(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): "own-level" | "extent" {
  const raw = environment?.[LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "") return "own-level";
  if (raw !== "own-level" && raw !== "extent") {
    throw new RangeError(`${LIVE_SVO_COARSE_LEAF_PAGES_ENVIRONMENT_VARIABLE} must be "own-level" or "extent"`);
  }
  return raw;
}

const COARSE_LEAF_PAGES = liveSvoCoarseLeafPagePolicy();

/** CPU mirror of the GPU dirty-leaf coverage used by generic planning tests. */
export function liveSvoLeafBasePages(options: {
  coordinate: readonly [number, number, number];
  leafLevel: number;
  finestLevel: number;
  brickSize: SparseBrickSize;
}): readonly (readonly [number, number, number])[] {
  const { coordinate, leafLevel, finestLevel, brickSize } = options;
  if (!Number.isInteger(leafLevel) || !Number.isInteger(finestLevel) || leafLevel < 0 || leafLevel > finestLevel || finestLevel > 21) {
    throw new RangeError("Live derived leaf levels must be ordered integers in [0, 21]");
  }
  if (coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Live derived leaf coordinate must be nonnegative safe integers");
  }
  const scale = 2 ** (finestLevel - leafLevel);
  const first = coordinate.map((value) => Math.floor(value * scale * brickSize / SVO_NODE_MIP_LAYOUT.interiorSize));
  const last = coordinate.map((value) => Math.floor(((value + 1) * scale * brickSize - 1) / SVO_NODE_MIP_LAYOUT.interiorSize));
  const pages: [number, number, number][] = [];
  for (let z = first[2]; z <= last[2]; z += 1) for (let y = first[1]; y <= last[1]; y += 1) for (let x = first[0]; x <= last[0]; x += 1) {
    pages.push([x, y, z]);
  }
  return pages;
}

/**
 * The one pyramid page a leaf owns, at the level whose texels are its voxels.
 *
 * A leaf spans `2^(finest - level) * brickSize` cells per axis and holds
 * `brickSize^3` voxels. A virtual page at mip level `p` spans `8 * 2^p` cells
 * over `8^3` texels, so the level where one texel is exactly one voxel is
 * `p = log2(cells / 8)` — and at that level the leaf *is* one page, because a
 * leaf's origin is aligned to its own extent.
 *
 * {@link liveSvoLeafBasePages} instead expanded every leaf to every base page
 * in its extent, which made the page count a function of claim *volume*: a
 * coarse leaf published `8^p` pages holding `brickSize^3` distinct values
 * between them, i.e. the same data `8^p` times. On `hero-garden-hose` at
 * environment refinement depth 3 that was 58 % of all pages from 0.59 % of the
 * leaves, and it is the entire reason the pyramid grew ~7.9x per halving of
 * the leaf while the ground's surface grew ~3.6x.
 *
 * Nothing is lost by publishing at the leaf's own level, because nothing finer
 * exists to lose: the levels beneath hold no page, and a parent whose child
 * page is absent now reads the leaf directly (`childSample`'s tagged fallback
 * in the build shader). What a *cone* loses is a level-0 sample in the interior
 * of a coarse region, which is by construction a region the topology declined
 * to resolve.
 *
 * A brick smaller than a page (`brickSize` 4 at the finest level) has no level
 * of its own and still lands in the base page containing it, exactly as before.
 */
export function liveSvoLeafPage(options: {
  coordinate: readonly [number, number, number];
  leafLevel: number;
  finestLevel: number;
  brickSize: SparseBrickSize;
}): SvoNodeMipSeedPage {
  const { coordinate, leafLevel, finestLevel, brickSize } = options;
  if (!Number.isInteger(leafLevel) || !Number.isInteger(finestLevel) || leafLevel < 0 || leafLevel > finestLevel || finestLevel > 21) {
    throw new RangeError("Live derived leaf levels must be ordered integers in [0, 21]");
  }
  if (coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new RangeError("Live derived leaf coordinate must be nonnegative safe integers");
  }
  const interior = SVO_NODE_MIP_LAYOUT.interiorSize;
  const cells = 2 ** (finestLevel - leafLevel) * brickSize;
  if (cells < interior) {
    const cellOrigin = coordinate.map((value) => value * cells);
    return { level: 0, coordinate: cellOrigin.map((value) => Math.floor(value / interior)) as unknown as SvoNodeMipCoordinate };
  }
  const level = Math.log2(cells / interior);
  if (!Number.isInteger(level)) throw new RangeError("Live derived leaf extent must be a power-of-two multiple of the page interior");
  return { level, coordinate: [...coordinate] as unknown as SvoNodeMipCoordinate };
}

/**
 * Virtual pages the canonical sparse topology claims, each at its own level.
 *
 * One page per leaf — see {@link liveSvoLeafPage}. Empty space with no leaf
 * still never consumes a physical opacity/radiance slot.
 *
 * `floorLevel` raises every seed below it to the floor
 * ({@link SVO_OPACITY_LEVEL_FLOOR}). Because a raised seed lands on a page the
 * plan's ancestor walk would have produced anyway, the result at every level at
 * or above the floor is *identical* to the unfloored plan — the only difference
 * is that the levels beneath it are gone. Zero is the shipped behaviour.
 *
 * The GPU worklist needs no matching parameter: `seedPage` asks the direct page
 * table for a slot, a floored plan gives levels below the floor a zero-depth
 * slab, and the lookup answers `INVALID`. The page coordinate it carries is
 * halved per level from the seed either way, so the first level that *does*
 * resolve is the floor, holding the same coordinate this function computes.
 */
export function liveSvoPlanBasePages(
  plan: SparseBrickPlan,
  policy: "own-level" | "extent" = COARSE_LEAF_PAGES,
  floorLevel = 0,
): readonly SvoNodeMipSeedPage[] {
  return completeCooperativeBuild(liveSvoPlanBasePagesSteps(plan, policy, floorLevel));
}

/**
 * The same seeds, offered as slices.
 *
 * One pass over every planned leaf, and the curvature refinement now plans
 * 441k of them at environment refinement depth 3 — 45 ms per 112k measured, so
 * roughly 175 ms held whole. The opacity floor collapses many of those leaves
 * onto one seed, which shrinks what the *address* plan then costs but not what
 * this pass does: it still visits every leaf to find that out.
 */
export function* liveSvoPlanBasePagesSteps(
  plan: SparseBrickPlan,
  policy: "own-level" | "extent" = COARSE_LEAF_PAGES,
  floorLevel = 0,
): Generator<unknown, readonly SvoNodeMipSeedPage[], undefined> {
  const unique = new Map<string, SvoNodeMipSeedPage>();
  const add = (seed: SvoNodeMipSeedPage): void => {
    const page = floorLevel > 0 ? raiseSvoNodeMipSeedToFloor(seed, floorLevel) : seed;
    unique.set(svoNodeMipSeedKey(page), page);
  };
  let visited = 0;
  for (const leaf of plan.leaves) {
    if ((visited += 1) % 4096 === 0) yield;
    const node = plan.nodes[leaf.nodeIndex];
    if (!node) throw new RangeError(`Live derived leaf ${leaf.index} references a missing node`);
    const request = {
      coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z] as const,
      leafLevel: node.level,
      finestLevel: plan.maximumDepth,
      brickSize: plan.brickSize,
    };
    if (policy === "extent") {
      for (const at of liveSvoLeafBasePages(request)) {
        add({ level: 0, coordinate: at as unknown as SvoNodeMipCoordinate });
      }
      continue;
    }
    add(liveSvoLeafPage(request));
  }
  return [...unique.values()];
}

/**
 * `Params` below: 5 vec4u, then per-level section offsets and capacities, each
 * `SVO_NODE_MIP_MAXIMUM_LEVELS` words packed four to a vec4u. Stated once so the
 * host write and the struct cannot drift apart.
 */
const LIVE_SVO_DERIVED_PLANNER_LEVEL_VECTORS = Math.ceil(SVO_NODE_MIP_MAXIMUM_LEVELS / 4);
export const LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES = (4 * 5 + 2 * 4 * LIVE_SVO_DERIVED_PLANNER_LEVEL_VECTORS) * 4;

/**
 * The lighting queue's words. One buffer: a state block, a header per ring,
 * then each ring's records — one slot a record.
 *
 * Every level owns two rings. Its *seed* ring holds pages whose own voxels
 * changed, and is drained in the frame they are appended. Its *ancestor* ring
 * holds pages one of whose children was rebuilt, and waits while a revision is
 * still being voxelized, so a page above a 27-chunk edit is reduced once, after
 * the last chunk, instead of 27 times.
 */
export const LIVE_SVO_DERIVED_QUEUE = Object.freeze({
  /** Lighting frames encoded since the queue was reset. */
  clockWord: 0,
  /** Feedback rotations started. */
  cycleWord: 1,
  /** `[begin, end)` of the journal this frame consumes; `cursor` is where the next begins. */
  journalBeginWord: 2,
  journalEndWord: 3,
  journalCursorWord: 4,
  /** Nonzero while no revision is being voxelized. */
  settledWord: 5,
  /** The value a seed page is stamped with: one marking as "awaiting build" per revision. */
  revisionWord: 6,
  /** The ring the sweep is on, and how much of it this visit may still take. */
  sweepRingWord: 7,
  sweepRemainingWord: 8,
  /** Pages this frame may still build. */
  allowanceWord: 9,
  /** Nonzero on a frame that starts a feedback rotation. */
  fireWord: 10,
  /** Latest clock any page's feedback window ends at. */
  windowClockWord: 11,
  /** The scene-wide window a relight opens: the clock and the rotation it ends at. */
  globalClockWord: 12,
  globalCycleWord: 13,
  /**
   * Faults, as bits: a page was dropped because both of its level's rings were
   * full; the journal lapped the cursor; a page went to its level's other ring.
   * Never read on the frame path.
   */
  overflowWord: 14,
  ringOverflowFault: 1,
  journalLapFault: 2,
  ringCrowdedFault: 4,
  /** Pages handed to a batch since reset; a probe's counter. */
  builtWord: 15,
  generationWord: 16,
  frameFlagsWord: 17,
  /** Nonzero on a frame whose journal range was overwritten before it was read. */
  lappedWord: 18,
  /** The clock the last rotation started at, and the frames one is allowed to finish in. */
  fireClockWord: 19,
  marginWord: 20,
  /** Latest rotation any page's feedback window ends at: no rotation starts once none could select a page. */
  windowCycleWord: 21,
  stateWords: 32,
  ringHeaderWords: 4,
  ringTailWord: 0,
  ringHeadWord: 1,
  /** Words a slot keeps: flags, revision stamp, window clock, window rotation, level + 1, page xyz. */
  slotWords: 8,
  slotFlagsWord: 0,
  slotRevisionWord: 1,
  slotWindowClockWord: 2,
  slotWindowCycleWord: 3,
  slotLevelWord: 4,
  slotPageWord: 5,
  queuedFlag: 1,
  windowFlag: 2,
  seedRing: 0,
  ancestorRing: 1,
  /** `Params.domain.w`. */
  listMode: 0,
  allLeavesMode: 1,
  journalMode: 2,
  /** `Params.frame.z`. */
  globalWindowFlag: 1,
  skipJournalFlag: 2,
  noFeedbackFlag: 4,
  /** Build ancestors although a revision is still landing; a first build owes every page. */
  openAncestorsFlag: 8,
} as const);

const QUEUE = LIVE_SVO_DERIVED_QUEUE;

export const liveSvoDerivedWorklistWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
// \`capacities\` is per level, because the queue's rings are: a level holds at
// most its own grid's pages, so a ring that deep can never refuse an append.
// vec4u rather than array<u32,N> so the element alignment is unambiguously
// legal in the uniform address space. The page table is a hash keyed by level,
// so no per-level table offsets are needed and the level count is bounded only
// by these two arrays.
//   source   x count or journal write-cursor word, y generation word, z first record word, w record stride
//   domain   x finest tree level, y level count, z radiance floor level, w source mode
//   limits   x source capacity, y slot capacity, z window frames, w window rotations
//   frame    x pages this frame may build, y batch capacity, z frame flags, w feedback reach in pages
//   scene    x requested-revision word, y completed-revision word, z first radiance slot, w frames a rotation is assumed to take at worst
struct Params{source:vec4u,domain:vec4u,limits:vec4u,frame:vec4u,scene:vec4u,sections:array<vec4u,${LIVE_SVO_DERIVED_PLANNER_LEVEL_VECTORS}>,capacities:array<vec4u,${LIVE_SVO_DERIVED_PLANNER_LEVEL_VECTORS}>}
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> control:array<u32>;
@group(0) @binding(2) var<storage,read> topology:array<u32>;
@group(0) @binding(3) var pageTable:texture_2d<u32>;
@group(0) @binding(4) var<storage,read_write> slots:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> queue:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read> generationState:array<u32>;
@group(0) @binding(8) var<storage,read_write> batch:array<u32>;
@group(0) @binding(9) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(10) var radianceValidity:texture_storage_2d<r32uint,write>;
const INVALID:u32=0xffffffffu;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
const HEADER_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.headerWords}u;const PLANNER_THREADS:u32=${LIVE_SVO_DERIVED_BUDGET.plannerThreads}u;
const Q_CLOCK:u32=${QUEUE.clockWord}u;const Q_CYCLE:u32=${QUEUE.cycleWord}u;const Q_JOURNAL_BEGIN:u32=${QUEUE.journalBeginWord}u;const Q_JOURNAL_END:u32=${QUEUE.journalEndWord}u;
const Q_JOURNAL_CURSOR:u32=${QUEUE.journalCursorWord}u;const Q_SETTLED:u32=${QUEUE.settledWord}u;const Q_REVISION:u32=${QUEUE.revisionWord}u;const Q_SWEEP_RING:u32=${QUEUE.sweepRingWord}u;
const Q_SWEEP_REMAINING:u32=${QUEUE.sweepRemainingWord}u;const Q_ALLOWANCE:u32=${QUEUE.allowanceWord}u;const Q_FIRE:u32=${QUEUE.fireWord}u;const Q_WINDOW_CLOCK:u32=${QUEUE.windowClockWord}u;
const Q_GLOBAL_CLOCK:u32=${QUEUE.globalClockWord}u;const Q_GLOBAL_CYCLE:u32=${QUEUE.globalCycleWord}u;const Q_OVERFLOW:u32=${QUEUE.overflowWord}u;const Q_BUILT:u32=${QUEUE.builtWord}u;
const Q_GENERATION:u32=${QUEUE.generationWord}u;const Q_FRAME_FLAGS:u32=${QUEUE.frameFlagsWord}u;const Q_LAPPED:u32=${QUEUE.lappedWord}u;const Q_FIRE_CLOCK:u32=${QUEUE.fireClockWord}u;const Q_MARGIN:u32=${QUEUE.marginWord}u;const Q_WINDOW_CYCLE:u32=${QUEUE.windowCycleWord}u;const Q_RINGS:u32=${QUEUE.stateWords}u;const RING_WORDS:u32=${QUEUE.ringHeaderWords}u;
const SLOT_WORDS:u32=${QUEUE.slotWords}u;const QUEUED:u32=${QUEUE.queuedFlag}u;const WINDOW:u32=${QUEUE.windowFlag}u;
const SEED_RING:u32=${QUEUE.seedRing}u;const ANCESTOR_RING:u32=${QUEUE.ancestorRing}u;
const MODE_ALL_LEAVES:u32=${QUEUE.allLeavesMode}u;const MODE_JOURNAL:u32=${QUEUE.journalMode}u;
const GLOBAL_WINDOW:u32=${QUEUE.globalWindowFlag}u;const SKIP_JOURNAL:u32=${QUEUE.skipJournalFlag}u;const NO_FEEDBACK:u32=${QUEUE.noFeedbackFlag}u;const OPEN_ANCESTORS:u32=${QUEUE.openAncestorsFlag}u;
fn levelCapacity(level:u32)->u32{return params.capacities[level/4u][level%4u];}
fn levelSection(level:u32)->u32{return params.sections[level/4u][level%4u];}
${svoNodeMipPageHashWGSL}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn tableSlot(level:u32,p:vec3u)->u32{if(level>=params.domain.y){return INVALID;}return svoNodeMipPageHashFind(pageTable,level,p);}
fn deepestLeaf(globalCell:vec3u)->u32{let brickSize=control[11];if(brickSize==0u||control[0]==0u){return INVALID;}let brick=globalCell/brickSize;var node=0u;var selected=INVALID;
 for(var level=0u;level<=params.domain.x&&node<control[0];level+=1u){let nodeBase=node*8u;let leaf=topology[nodeBase+6u];if(leaf<control[1]&&(topology[nodeBase+7u]&ACTIVE)!=0u){selected=leaf;}if(level==params.domain.x){break;}let bit=params.domain.x-level-1u;let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);let mask=topology[nodeBase+3u]&0xffu;if((mask&(1u<<octant))==0u){break;}node=topology[nodeBase+4u]+countOneBits(mask&((1u<<octant)-1u));}
 return selected;}
fn ringHeader(level:u32,kind:u32)->u32{return Q_RINGS+(level*2u+kind)*RING_WORDS;}
fn ringRecords(level:u32,kind:u32)->u32{return levelSection(level)+kind*levelCapacity(level);}
fn ringPending(header:u32)->u32{return atomicLoad(&queue[header+${QUEUE.ringTailWord}u])-atomicLoad(&queue[header+${QUEUE.ringHeadWord}u]);}
// Append one page to its level's ring, once however many threads ask. The
// QUEUED bit is the whole of the deduplication: a page already waiting is not
// appended again, and \`resolve\` clears the bit as it takes the page, so work that
// arrives after that is a new append and a new build. The key is written
// unconditionally because a slot can change hands between frames, and the page
// built is always the one the hash names now. A ring is as deep as its level, so
// an append is only ever refused when stale slots crowd a full level.
fn appendRing(slot:u32,level:u32,kind:u32)->bool{
 let header=ringHeader(level,kind);let capacity=levelCapacity(level);let position=atomicAdd(&queue[header+${QUEUE.ringTailWord}u],1u);
 if(position-atomicLoad(&queue[header+${QUEUE.ringHeadWord}u])>=capacity){atomicSub(&queue[header+${QUEUE.ringTailWord}u],1u);return false;}
 atomicStore(&queue[ringRecords(level,kind)+position%capacity],slot);return true;
}
fn enqueue(slot:u32,level:u32,page:vec3u,kind:u32,flags:u32)->bool{
 let base=slot*SLOT_WORDS;
 atomicStore(&slots[base+${QUEUE.slotLevelWord}u],level+1u);atomicStore(&slots[base+${QUEUE.slotPageWord}u],page.x);atomicStore(&slots[base+${QUEUE.slotPageWord + 1}u],page.y);atomicStore(&slots[base+${QUEUE.slotPageWord + 2}u],page.z);
 if((atomicOr(&slots[base],flags)&QUEUED)!=0u){return false;}
 if(appendRing(slot,level,kind)){return true;}
 // Crowded out by slots released while they waited. The level's other ring
 // takes the page — early for an ancestor, late for a seed, never lost.
 if(appendRing(slot,level,1u-kind)){atomicOr(&queue[Q_OVERFLOW],${QUEUE.ringCrowdedFault}u);return true;}
 atomicAnd(&slots[base],~QUEUED);atomicOr(&queue[Q_OVERFLOW],${QUEUE.ringOverflowFault}u);return false;
}
// "Awaiting build": resident, and holding nothing a cone may read. Radiance
// validity exists only for slots of the radiance pool.
fn markAwaiting(slot:u32){
 textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(0u));
 if(slot>=params.scene.z){textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(0u));}
}
// The page a leaf's voxels are published in: the first level at or above its
// own that the plan holds a page at. Everything above it is reached by
// \`resolve\`, which queues a page's parent as it builds the page — so a parent is
// always built after the child that changed it, whatever order work arrives in.
fn seedPage(firstLevel:u32,firstPage:vec3u,mode:u32,revision:u32){
 var page=firstPage;
 for(var level=firstLevel;level<params.domain.y;level+=1u){
  let slot=tableSlot(level,page);
  if(slot!=INVALID&&slot<params.limits.y){
   let won=enqueue(slot,level,page,SEED_RING,select(QUEUED,QUEUED|WINDOW,level==params.domain.z));
   // A page whose voxels a revision rewrote is awaiting from the frame they
   // land until it is rebuilt — marked once per revision, so a page several
   // chunks touch does not blink between them.
   if(mode==MODE_JOURNAL){if(atomicExchange(&slots[slot*SLOT_WORDS+${QUEUE.slotRevisionWord}u],revision)!=revision){markAwaiting(slot);}}
   else if(mode==MODE_ALL_LEAVES&&won){markAwaiting(slot);}
   return;}
  page>>=vec3u(1u);}
}
fn seedLeaf(leaf:u32,mode:u32,revision:u32){
 // A retired leaf's slot names no node until it is reused; it seeds nothing.
 if(leaf>=control[1]){return;}let leafBase=control[16]+leaf*4u;let node=topology[leafBase];if(node>=control[0]){return;}let nodeBase=node*8u;let leafLevel=topology[nodeBase+2u];if(leafLevel>params.domain.x){return;}let scale=1u<<(params.domain.x-leafLevel);let brickOrigin=morton(topology[nodeBase],topology[nodeBase+1u],leafLevel)*scale;let cellMinimum=brickOrigin*control[11];
${COARSE_LEAF_PAGES === "extent" ? /* wgsl */ ` let cellMaximum=(brickOrigin+vec3u(scale))*control[11];let firstPage=cellMinimum/${SVO_NODE_MIP_LAYOUT.interiorSize}u;let lastPage=(cellMaximum-vec3u(1u))/${SVO_NODE_MIP_LAYOUT.interiorSize}u;
 for(var z=firstPage.z;z<=lastPage.z;z+=1u){for(var y=firstPage.y;y<=lastPage.y;y+=1u){for(var x=firstPage.x;x<=lastPage.x;x+=1u){seedPage(0u,vec3u(x,y,z),mode,revision);}}}` : /* wgsl */ ` // One page per leaf, at the level whose texels are its voxels — the CPU mirror
 // is \`liveSvoLeafPage\`. Expanding a coarse leaf across its extent published the
 // same brickSize^3 values once per finest brick it covered, which is what made
 // the pyramid scale with claim volume instead of with surface area.
 let cells=scale*control[11];var seedLevel=0u;
 if(cells>=${SVO_NODE_MIP_LAYOUT.interiorSize}u){seedLevel=firstTrailingBit(cells)-${Math.log2(SVO_NODE_MIP_LAYOUT.interiorSize)}u;}
 seedPage(seedLevel,cellMinimum/(${SVO_NODE_MIP_LAYOUT.interiorSize}u<<seedLevel),mode,revision);`}
}
// A rebuilt floor page changes what every page within one gather's reach
// receives, and nothing farther in the first bounce. Each of those pages keeps
// its own window, so feedback runs over the neighbourhood of the edit and stops.
fn stampWindows(page:vec3u){
 let clockEnd=atomicLoad(&queue[Q_CLOCK])+params.limits.z;let cycleEnd=atomicLoad(&queue[Q_CYCLE])+params.limits.w;
 atomicMax(&queue[Q_WINDOW_CLOCK],clockEnd);atomicMax(&queue[Q_WINDOW_CYCLE],cycleEnd);
 let reach=vec3u(params.frame.w);let lowest=max(page,reach)-reach;let highest=page+reach;let floorLevel=params.domain.z;
 for(var z=lowest.z;z<=highest.z;z+=1u){for(var y=lowest.y;y<=highest.y;y+=1u){for(var x=lowest.x;x<=highest.x;x+=1u){
  let neighbour=vec3u(x,y,z);let slot=tableSlot(floorLevel,neighbour);if(slot==INVALID||slot>=params.limits.y){continue;}
  let base=slot*SLOT_WORDS;
  atomicStore(&slots[base+${QUEUE.slotLevelWord}u],floorLevel+1u);atomicStore(&slots[base+${QUEUE.slotPageWord}u],x);atomicStore(&slots[base+${QUEUE.slotPageWord + 1}u],y);atomicStore(&slots[base+${QUEUE.slotPageWord + 2}u],z);
  atomicMax(&slots[base+${QUEUE.slotWindowClockWord}u],clockEnd);atomicMax(&slots[base+${QUEUE.slotWindowCycleWord}u],cycleEnd);}}}
}
fn writeRecord(record:u32,slot:u32,level:u32,page:vec3u){batch[record]=slot;
 // One layout at every level now: slot, page coordinate, eight children.
 batch[record+1u]=page.x;batch[record+2u]=page.y;batch[record+3u]=page.z;
 if(level==0u){for(var octant=0u;octant<8u;octant+=1u){let bit=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);batch[record+4u+octant]=deepestLeaf(page*${SVO_NODE_MIP_LAYOUT.interiorSize}u+bit*4u);}}
 else{
  // An absent child page is resolved here, once, rather than read as zero in
  // the reduction: a leaf that publishes at its own coarse level leaves every
  // level beneath it unpaged, and those cells are solid ground, not air.
  // Because *every* leaf publishes a page, an absent child page provably has
  // no leaf inside it, so the leaf covering its centre covers all of it.
  let childScale=1u<<(level-1u);
  for(var child=0u;child<8u;child+=1u){let bit=vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
   let childPage=page*2u+bit;var slotOrLeaf=tableSlot(level-1u,childPage);
   if(slotOrLeaf==INVALID){
    let centre=(childPage*${SVO_NODE_MIP_LAYOUT.interiorSize}u+vec3u(${SVO_NODE_MIP_LAYOUT.interiorSize / 2}u))*childScale;
    let leaf=deepestLeaf(centre);
    if(leaf!=INVALID){slotOrLeaf=${LIVE_SVO_DERIVED_WORKLIST.leafSlotTag}u|leaf;}}
   batch[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+child]=slotOrLeaf;}}
}
// First dispatch of a lighting frame, one thread. Latches everything the rest
// of the frame agrees on — the journal range to consume, whether a revision is
// still landing, the allowance — and decides whether a feedback rotation starts.
@compute @workgroup_size(1) fn prepareFrame(){
 let clock=atomicLoad(&queue[Q_CLOCK])+1u;atomicStore(&queue[Q_CLOCK],clock);
 let generation=generationState[params.source.y];atomicStore(&queue[Q_GENERATION],generation);
 let flags=params.frame.z;atomicStore(&queue[Q_FRAME_FLAGS],flags);
 let requested=source[params.scene.x];let settled=requested==source[params.scene.y];
 let wasSettled=atomicLoad(&queue[Q_SETTLED])!=0u;
 atomicStore(&queue[Q_SETTLED],select(0u,1u,settled));atomicStore(&queue[Q_REVISION],requested+1u);
 // The journal is taken exactly once: the cursor moves past whatever this frame
 // will consume. An initial build lights every leaf, so it skips what is there.
 let write=source[params.source.x];var begin=atomicLoad(&queue[Q_JOURNAL_CURSOR]);var end=begin;var lapped=0u;
 if(generation!=0u){end=write;if((flags&SKIP_JOURNAL)!=0u){begin=write;}
  // More leaves were finalized since the last frame than the journal holds (or
  // its cursor went backwards): the range is gone, so every leaf is dirty.
  if(end-begin>params.limits.x){lapped=1u;atomicOr(&queue[Q_OVERFLOW],${QUEUE.journalLapFault}u);}
  atomicStore(&queue[Q_JOURNAL_CURSOR],write);}
 atomicStore(&queue[Q_JOURNAL_BEGIN],begin);atomicStore(&queue[Q_JOURNAL_END],end);atomicStore(&queue[Q_LAPPED],lapped);
 // Fresh seeds, or a revision that has just finished, restart the sweep at the
 // finest level so every level is reduced after the one beneath it.
 if(end!=begin||(settled&&!wasSettled)){atomicStore(&queue[Q_SWEEP_RING],0u);atomicStore(&queue[Q_SWEEP_REMAINING],0u);}
 // Positions only ever grow; fold them back so they never reach 2^32.
 var pending=0u;
 for(var ring=0u;ring<params.domain.y*2u;ring+=1u){
  let header=Q_RINGS+ring*RING_WORDS;let capacity=levelCapacity(ring/2u);let head=atomicLoad(&queue[header+${QUEUE.ringHeadWord}u]);
  if(head>=capacity){let folded=(head/capacity)*capacity;atomicStore(&queue[header+${QUEUE.ringHeadWord}u],head-folded);atomicStore(&queue[header+${QUEUE.ringTailWord}u],atomicLoad(&queue[header+${QUEUE.ringTailWord}u])-folded);}
  pending+=ringPending(header);}
 // Feedback waits for the build: a rotation starts only when no revision is
 // landing and less than one frame of building is still queued. It must also
 // be able to finish inside its window, which is all the frames the host has
 // promised: a rotation is given as long as the last one took, and the worst
 // case when there is no last one to go by.
 let cycle=atomicLoad(&queue[Q_CYCLE]);var fire=0u;
 let lastFire=atomicLoad(&queue[Q_FIRE_CLOCK]);var margin=params.scene.w;if(lastFire!=0u){margin=min(clock-lastFire,margin);}margin+=1u;
 if(settled&&generation!=0u&&pending<params.frame.x&&(flags&NO_FEEDBACK)==0u
  &&((clock+margin<atomicLoad(&queue[Q_WINDOW_CLOCK])&&cycle<atomicLoad(&queue[Q_WINDOW_CYCLE]))||(clock+margin<atomicLoad(&queue[Q_GLOBAL_CLOCK])&&cycle<atomicLoad(&queue[Q_GLOBAL_CYCLE])))){
  fire=1u;atomicStore(&queue[Q_CYCLE],cycle+1u);atomicStore(&queue[Q_FIRE_CLOCK],clock);atomicStore(&queue[Q_MARGIN],margin);}
 atomicStore(&queue[Q_FIRE],fire);
 // A relight, or the first build, opens the window over every floor page.
 if((flags&GLOBAL_WINDOW)!=0u&&generation!=0u){atomicStore(&queue[Q_GLOBAL_CLOCK],clock+params.limits.z);atomicStore(&queue[Q_GLOBAL_CYCLE],atomicLoad(&queue[Q_CYCLE])+params.limits.w);}
 atomicStore(&queue[Q_ALLOWANCE],params.frame.x);
}
// One feedback rotation: every floor page whose window is open, queued once.
// A page's parents follow from \`resolve\` like any other build's.
@compute @workgroup_size(64) fn selectFeedback(@builtin(global_invocation_id) gid:vec3u){
 if(atomicLoad(&queue[Q_FIRE])==0u){return;}
 let clock=atomicLoad(&queue[Q_CLOCK])+atomicLoad(&queue[Q_MARGIN]);let cycle=atomicLoad(&queue[Q_CYCLE])-1u;let floorLevel=params.domain.z;
 let everywhere=clock<atomicLoad(&queue[Q_GLOBAL_CLOCK])&&cycle<atomicLoad(&queue[Q_GLOBAL_CYCLE]);
 for(var slot=gid.x;slot<params.limits.y;slot+=PLANNER_THREADS){
  let base=slot*SLOT_WORDS;if(atomicLoad(&slots[base+${QUEUE.slotLevelWord}u])!=floorLevel+1u){continue;}
  if(!everywhere&&!(clock<atomicLoad(&slots[base+${QUEUE.slotWindowClockWord}u])&&cycle<atomicLoad(&slots[base+${QUEUE.slotWindowCycleWord}u]))){continue;}
  let page=vec3u(atomicLoad(&slots[base+${QUEUE.slotPageWord}u]),atomicLoad(&slots[base+${QUEUE.slotPageWord + 1}u]),atomicLoad(&slots[base+${QUEUE.slotPageWord + 2}u]));
  // The slot may have been given to another page since its window was stamped.
  if(tableSlot(floorLevel,page)!=slot){continue;}
  enqueue(slot,floorLevel,page,ANCESTOR_RING,QUEUED);}
}
// Seeds: the leaves a source names, each to the page its voxels are published in.
@compute @workgroup_size(64) fn populate(@builtin(global_invocation_id) gid:vec3u){
 if(atomicLoad(&queue[Q_GENERATION])==0u){return;}
 let mode=params.domain.w;let revision=atomicLoad(&queue[Q_REVISION]);var begin=0u;var count=0u;
 // A lapped journal no longer names what changed, so the frame seeds every leaf instead.
 let everyLeaf=mode==MODE_ALL_LEAVES||(mode==MODE_JOURNAL&&atomicLoad(&queue[Q_LAPPED])!=0u);
 if(everyLeaf){count=control[1];}
 else if(mode==MODE_JOURNAL){begin=atomicLoad(&queue[Q_JOURNAL_BEGIN]);count=atomicLoad(&queue[Q_JOURNAL_END])-begin;}
 else{count=min(source[params.source.x],params.limits.x);}
 for(var index=gid.x;index<count;index+=PLANNER_THREADS){
  var leaf=index;
  if(!everyLeaf){
   if(mode==MODE_JOURNAL){leaf=source[params.source.z+((begin+index)%params.limits.x)*params.source.w];}
   else{leaf=source[params.source.z+index*params.source.w];}}
  seedLeaf(leaf,mode,revision);}
}
// One thread per round: choose the ring this round builds from, and how much.
//
// The sweep climbs from the finest level. A ring is visited with a quota — what
// it held on arrival — so pages a solver keeps appending cannot hold the sweep
// on one level. Seed rings are always open; ancestor rings open once the
// revision has landed. Because a batch is as large as a frame's allowance, a
// round either spends the allowance or finishes its ring.
@compute @workgroup_size(1) fn prepareRound(){
 let generation=atomicLoad(&queue[Q_GENERATION]);
 batch[${LIVE_SVO_DERIVED_WORKLIST.countWord}u]=0u;batch[${LIVE_SVO_DERIVED_WORKLIST.generationWord}u]=generation;batch[${LIVE_SVO_DERIVED_WORKLIST.recordOffsetWord}u]=HEADER_WORDS;
 let allowance=atomicLoad(&queue[Q_ALLOWANCE]);if(generation==0u||allowance==0u){return;}
 let settled=atomicLoad(&queue[Q_SETTLED])!=0u||(atomicLoad(&queue[Q_FRAME_FLAGS])&OPEN_ANCESTORS)!=0u;let ringCount=params.domain.y*2u;
 var ring=atomicLoad(&queue[Q_SWEEP_RING])%ringCount;var remaining=atomicLoad(&queue[Q_SWEEP_REMAINING]);
 if((ring&1u)==ANCESTOR_RING&&!settled){remaining=0u;}
 remaining=min(remaining,ringPending(Q_RINGS+ring*RING_WORDS));
 if(remaining==0u){
  for(var step=0u;step<ringCount;step+=1u){
   let candidate=(ring+step)%ringCount;if((candidate&1u)==ANCESTOR_RING&&!settled){continue;}
   let pending=ringPending(Q_RINGS+candidate*RING_WORDS);if(pending>0u){ring=candidate;remaining=pending;break;}}
  if(remaining==0u){atomicStore(&queue[Q_SWEEP_REMAINING],0u);return;}}
 let level=ring/2u;let kind=ring&1u;let header=Q_RINGS+ring*RING_WORDS;let head=atomicLoad(&queue[header+${QUEUE.ringHeadWord}u]);
 let take=min(min(remaining,allowance),params.frame.y);
 batch[${LIVE_SVO_DERIVED_WORKLIST.countWord}u]=take;batch[${LIVE_SVO_DERIVED_WORKLIST.levelWord}u]=level;
 batch[${LIVE_SVO_DERIVED_WORKLIST.ringRecordsWord}u]=ringRecords(level,kind);batch[${LIVE_SVO_DERIVED_WORKLIST.ringHeadWord}u]=head;
 batch[${LIVE_SVO_DERIVED_WORKLIST.ringCapacityWord}u]=levelCapacity(level);batch[${LIVE_SVO_DERIVED_WORKLIST.ringKindWord}u]=kind;
 atomicStore(&queue[header+${QUEUE.ringHeadWord}u],head+take);remaining-=take;
 atomicStore(&queue[Q_SWEEP_REMAINING],remaining);atomicStore(&queue[Q_SWEEP_RING],select(ring,(ring+1u)%ringCount,remaining==0u));
 atomicStore(&queue[Q_ALLOWANCE],allowance-take);atomicStore(&queue[Q_BUILT],atomicLoad(&queue[Q_BUILT])+take);
}
// One thread per record of the batch: turn a queued slot into a build record.
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x;if(index>=min(batch[${LIVE_SVO_DERIVED_WORKLIST.countWord}u],params.frame.y)){return;}
 let record=batch[${LIVE_SVO_DERIVED_WORKLIST.recordOffsetWord}u]+index*RECORD_WORDS;batch[record]=INVALID;
 let level=batch[${LIVE_SVO_DERIVED_WORKLIST.levelWord}u];
 let slot=atomicLoad(&queue[batch[${LIVE_SVO_DERIVED_WORKLIST.ringRecordsWord}u]+(batch[${LIVE_SVO_DERIVED_WORKLIST.ringHeadWord}u]+index)%batch[${LIVE_SVO_DERIVED_WORKLIST.ringCapacityWord}u]]);
 if(slot>=params.limits.y){return;}
 let base=slot*SLOT_WORDS;let flags=atomicAnd(&slots[base],~(QUEUED|WINDOW));if((flags&QUEUED)==0u){return;}
 let keyLevel=atomicLoad(&slots[base+${QUEUE.slotLevelWord}u]);if(keyLevel==0u||keyLevel>params.domain.y){return;}
 let page=vec3u(atomicLoad(&slots[base+${QUEUE.slotPageWord}u]),atomicLoad(&slots[base+${QUEUE.slotPageWord + 1}u]),atomicLoad(&slots[base+${QUEUE.slotPageWord + 2}u]));
 // A slot is revalidated against the hash before anything is built into it:
 // released and given to another page while it waited, it is either that
 // page's build now, or — at another level — queued again where it belongs.
 // The seed ring, because it may be a new page nothing else will light first.
 if(tableSlot(keyLevel-1u,page)!=slot){return;}
 if(keyLevel-1u!=level){enqueue(slot,keyLevel-1u,page,SEED_RING,flags&(QUEUED|WINDOW));return;}
 if(level==params.domain.z&&(flags&WINDOW)!=0u&&(atomicLoad(&queue[Q_FRAME_FLAGS])&GLOBAL_WINDOW)==0u){stampWindows(page);}
 if(level+1u<params.domain.y){
  let parent=page>>vec3u(1u);let parentSlot=tableSlot(level+1u,parent);
  if(parentSlot!=INVALID&&parentSlot<params.limits.y){enqueue(parentSlot,level+1u,parent,ANCESTOR_RING,select(QUEUED,QUEUED|WINDOW,level+1u==params.domain.z));}}
 writeRecord(record,slot,level,page);
}
`;

/**
 * Payload reads that differ between a world with a solver and one without.
 *
 * A `dry` world allocates neither the dynamic geometry/velocity/material lanes
 * nor the two dead channels of the scene geometry lane, so every read of them
 * is compiled out here rather than clamped at runtime: an absent lane's offset
 * is the 256-byte zero page, and `offset + voxel * stride` would walk straight
 * out of it into the first present lane. See SPARSE_BRICK_PAYLOAD_PROFILES.
 *
 * Channel *meaning* is preserved across profiles — only the stride and the
 * index move — so the `full` expansion is character-identical to the shader
 * that shipped before profiles existed, which `tests/webgpu-svo-live-derived-
 * builder.test.ts` pins by matching against the exported constants.
 *
 * A dry world may narrow those two channels further (see
 * {@link SparseBrickSceneGeometryFormat}). That moves the unpack, not the meaning:
 * every surviving arm returns `solidDistance` in metres. A removed 2-byte arm
 * returned per-leaf band units instead, which was tolerable only because
 * `safeNormal` central-differences six samples from one leaf and normalises, so the
 * per-leaf scale cancelled. The `min` against the solver's metres on the `full`
 * profile is exactly why narrowing is refused there — and the reason any future
 * scaled arm must declare its units rather than rely on that cancellation.
 */

/**
 * `solidDistance` under the banded leaf payload's reconstruction rule.
 *
 * The B0.5b gate (`docs/svo-banded-leaf-payload-handoff.md`): the banded layout
 * stores a distance only for the band and the one-cell ring `safeNormal` reads,
 * and reconstructs everything else as the saturated value signed by occupancy.
 * That reconstruction is the design's one lossy semantic — 86.4 % of occupied
 * voxels lose their normal by it — and it can be tested *without* the arena, the
 * masks, the palette or the allocator by changing only what this function
 * returns. If the frame hash holds here, the rest of the layout is engineering; if
 * it moves, no amount of allocator work would have discovered that.
 *
 * The record predicate is the encoder's, expressed the same way: a voxel is
 * stored when it or one of its six axis neighbours is band, with the neighbours
 * clamped inside the leaf exactly as `safeNormal` clamps them. That clamped
 * 6-neighbourhood *is* the encoder's dilation — at a face the clamp folds the
 * outward neighbour onto the voxel itself, which is the same set the encoder
 * marks.
 *
 * The saturation is `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` cell radii of the
 * leaf's *own* voxel, so it matches `encodeBandedLeaf` rather than approximating
 * it, and the finest cell size is baked as a literal because the build shader's
 * uniform block does not carry one.
 */
/**
 * How far a radiance gather starts from the voxel that emits it, in finest cells.
 *
 * `1.5 * scale` unconditionally is what shipped, and at a coarse level that is
 * tens of finest cells — so a voxel metres inside the ground began its visibility
 * march *above* the terrain and took direct sun. That is the root cause of the
 * buried-radiance leak measured at 22.22/255 mean, and it is why fixing the
 * origin is preferable to omitting the radiance downstream of it.
 *
 * The offset cannot simply shrink. A **band** voxel has the surface passing
 * through it, so it needs an offset on the order of its own extent or its gather
 * self-intersects the voxel it started in — that is the self-shadow acne
 * `1.5 * scale` was buying, and it gets worse, not better, at coarse levels.
 *
 * The distinction is not the offset's size but whether there is anywhere outside
 * to move to. A **fully solid** voxel is interior everywhere within it; no offset
 * along a normal can put its origin in air that is legitimately its own. So the
 * rule is not a smaller offset, it is *no* offset for a fully solid voxel, and the
 * shipped offset for every voxel the surface actually touches. Expressed on the
 * fraction rather than on band membership because that is the exact question —
 * "is any part of this voxel not solid" — and it needs no stencil.
 */
function derivedMarchOffsetWGSL(solidOffsetFix: boolean): string {
  if (!solidOffsetFix) return "(1.5*f32(scale))";
  return "(select(1.5*f32(scale),0.0,bandedFractionAt(leaf,sampleLocal)>=1.0))";
}

/**
 * A fully solid receiver is opaque to direct light, decided before any step.
 *
 * `directVisibility` shares the gather march's step schedule, and that schedule is
 * expressed in *receiver* widths so a step from a floor-level block cannot land
 * inside the block that emitted it. That ratio is what stops a gather
 * self-sampling; for occlusion it is precisely backwards, because it guarantees
 * the *first* sample clears the receiver's own block. At a floor level of 3 and a
 * 6.25 mm cell the first sample is 20 finest cells — 12.5 cm — away, and the
 * hero's terrain is 0.14-0.38 m thick, so every sample from a voxel just under
 * the ground is already in open air above it, `1 - coverage` is 1 at every step,
 * and solid earth takes full sun.
 *
 * Rescaling the occlusion march alone does not fix it, and that is measured: the
 * same four steps taken in *finest* cells — 8x shorter here — recover 1.42/255 of
 * a 12.30/255 leak, and add 0.013/255 on top of this guard. The coarse pyramid
 * holds one average coverage per floor-level block and cannot separate "just below
 * the surface" from "just above": a first step short enough to find the ground
 * under a buried voxel reads the receiver's own straddling block for an exposed
 * voxel too, and dims every lit surface by that block's coverage. That is the
 * self-shadow acne the current scaling buys, arriving as a uniform darkening
 * instead of speckle.
 *
 * So the question is answered where the answer actually exists — in the fine
 * fraction, which knows whether *this* voxel has any air in it at all. A fully
 * solid voxel sees no sun, exactly and independently of any step length. A
 * partially solid voxel is by definition one the surface passes through, is
 * entitled to sun, and keeps the shipped march unchanged. Same predicate as the
 * march-offset rule (`bandedFractionAt >= 1.0`), deliberately: two arms that
 * disagree about which voxels are solid would be two definitions of solid.
 */
function derivedSolidDirectOcclusionWGSL(enabled: boolean) {
  return {
    /** Threaded rather than recomputed: the fraction read is a payload fetch. */
    parameter: enabled ? ",receiverSolid:bool" : "",
    guard: enabled ? "if(receiverSolid){return 0.0;}" : "",
    forward: enabled ? ",receiverSolid" : "",
    receiver: enabled ? ",bandedFractionAt(leaf,sampleLocal)>=1.0" : "",
  };
}

function derivedSolidDistanceWGSL(
  lane: { solidDistance: string; sceneCoverage: string },
  cellSize_m: readonly [number, number, number] | undefined,
  // The band helpers on their own: the march-offset rule needs
  // `bandedFractionAt` without the banded reconstruction being on.
  sharedOnly = false,
): string {
  const stored = /* wgsl */ `fn solidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.solidDistance};}`;
  // The band helpers are shared rather than inlined per caller: the banded
  // reconstruction and the march-offset rule ask the same "is this voxel's normal
  // real" question, and defining it twice is how two callers come to disagree
  // about which voxels have one.
  const shared = /* wgsl */ `
fn bandedFractionAt(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.sceneCoverage};}
fn bandedBandAt(leafIndex:u32,local:vec3u)->bool{let f=bandedFractionAt(leafIndex,local);return f>0.0&&f<1.0;}
fn bandedRecordAt(leafIndex:u32,local:vec3u)->bool{
  if(bandedBandAt(leafIndex,local)){return true;}
  let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));
  return bandedBandAt(leafIndex,vec3u(lo.x,local.y,local.z))||bandedBandAt(leafIndex,vec3u(hi.x,local.y,local.z))
    ||bandedBandAt(leafIndex,vec3u(local.x,lo.y,local.z))||bandedBandAt(leafIndex,vec3u(local.x,hi.y,local.z))
    ||bandedBandAt(leafIndex,vec3u(local.x,local.y,lo.z))||bandedBandAt(leafIndex,vec3u(local.x,local.y,hi.z));
}
fn bandedLeafScale(leafIndex:u32)->u32{
  let node=topology[control[16]+leafIndex*4u];let level=topology[node*8u+2u];
  let finest=finestLevel();
  return select(1u,1u<<(finest-level),finest>level);
}`;
  const helpers = sharedOnly ? shared : "";
  if (!cellSize_m) return `${stored}${helpers}`;
  const cell = cellSize_m.map((value) => value.toExponential(9)).join(",");
  return /* wgsl */ `${shared}
const BANDED_FINEST_CELL:vec3f=vec3f(${cell});
fn storedSolidDistance(leafIndex:u32,local:vec3u)->f32{let voxel=leafVoxel(leafIndex,local);return ${lane.solidDistance};}
fn bandedSaturation(leafIndex:u32)->f32{
  return ${SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII}.0*0.5*length(BANDED_FINEST_CELL*f32(bandedLeafScale(leafIndex)));
}
fn solidDistance(leafIndex:u32,local:vec3u)->f32{
  if(bandedRecordAt(leafIndex,local)){return storedSolidDistance(leafIndex,local);}
  let saturation=bandedSaturation(leafIndex);
  return select(saturation,-saturation,bandedFractionAt(leafIndex,local)>0.0);
}`;
}

function derivedLaneAccess(
  profile: SparseBrickPayloadProfileName,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
) {
  const dry = profile === "dry";
  const format = dry ? sceneGeometryFormat : "f32x2";
  const mode = dry ? leafPayloadMode : "dense";
  const stride = dry ? "2u" : "4u";
  const scene = (channel: string) => `payload[params.laneOffsets.z+voxel*${stride}+${channel}]`;
  const packedWord = "payload[sceneGeometryWord(params.laneOffsets.z,voxel)]";
  const narrowedDistance = `sceneDistanceOf(${packedWord},voxel)`;
  const narrowedCoverage = `clamp(sceneFractionOf(${packedWord},voxel),0.,1.)`;
  // Scene identity is not a lane under `banded`, so it is not an expression
  // either: it comes from the shared codec, addressed by the same
  // `bandedLanes` block the voxeliser is handed. The two geometry channels
  // deliberately do *not* move with it — this builder still reads the dense
  // `sceneGeometry` lane, which is the B2 cutover and a separate piece of work.
  const identityCodec = mode === "dense" ? "" : /* wgsl */ `${sparseBrickBandedLeafCodecWGSL({
    occupancyBase: "params.bandedLanes.x", recordMaskBase: "params.bandedLanes.y",
    headerBase: "params.bandedLanes.z", blobBase: "params.bandedLanes.w",
    recordsBase: "0u",
    load: (index) => `payload[${index}]`, mode, records: false,
  })}`;
  const storedSolidDistance = dry
    ? format === "f32x2" ? `bitcast<f32>(${scene("0u")})` : narrowedDistance
    : `min(bitcast<f32>(payload[voxel*4u+1u]),bitcast<f32>(${scene("1u")}))`;
  const storedSceneCoverage = format === "f32x2"
    ? `clamp(bitcast<f32>(${scene(dry ? "1u" : "2u")}),0.,1.)`
    : narrowedCoverage;
  return {
    /** Helper declarations the narrowed formats and the identity decode need in scope. */
    codec: `${sparseBrickSceneGeometryCodecWGSL(format)}${identityCodec}
${sparseBrickSceneIdentityCodecWGSL({
      mode, materialOwnerBase: "params.laneOffsets.w", load: (index) => `payload[${index}]`,
    })}`,
    /** The gradient of this is the only smooth normal the world can offer. */
    solidDistance: storedSolidDistance,
    // `>>> 0` because `0xffff << 16` is -65536 in JS's signed shift, and WGSL has
    // no unary minus on `u32` — the dry expansion failed to compile without it.
    dynamicIdentity: dry ? `${(SPARSE_BRICK_NO_OWNER << 16) >>> 0}u` : "payload[params.laneOffsets.y+voxel]",
    dynamicCoverage: dry ? "0." : "clamp(bitcast<f32>(payload[voxel*4u+2u]),0.,1.)",
    sceneCoverage: storedSceneCoverage,
    fluidFraction: dry ? "0." : "clamp(bitcast<f32>(payload[params.laneOffsets.x+voxel*4u+3u]),0.,1.)",
  };
}

/**
 * Where a page's interior starts, physically.
 *
 * Zero by default (see `SVO_NODE_MIP_LAYOUT`), so this is a rename of the `+1u`
 * that used to be sprinkled through every atlas read. Keeping it named rather
 * than deleting it is what lets `FLUID_SVO_NODE_MIP_APRON=1` put the shell back
 * for an A/B without hunting for literals in six shaders.
 */
const APRON = SVO_NODE_MIP_LAYOUT.apron;
/** Interior-texel clamp that reproduces a replica apron of the declared width. */
const INTERIOR_CLAMP_LOW = (-0.5 * APRON).toFixed(1);
const INTERIOR_CLAMP_HIGH = (1 - 0.5 * APRON).toFixed(1);

/**
 * Opacity lanes this build actually reduces.
 *
 * On a dry world `fluidFraction` is the literal `0.`, so `mean.y` is a mean of
 * zeros and `maximum.y` a maximum of zeros — arithmetic whose only destination
 * is two texture channels the narrow format does not have. Compiling it out is
 * not the saving (that is the format); it is what stops the shader claiming to
 * carry a lane its page cannot hold.
 */
function derivedOpacityLanes(format: GPUTextureFormat) {
  const fluid = svoNodeMipOpacityChannels(format) > 2;
  return {
    fluid,
    /** Base-level store. `solid` and `fluid` must be in scope. */
    storeBase: fluid
      ? "vec4f(solid,select(0.,1.,solid>0.),clamp(fluid,0.,1.),select(0.,1.,fluid>0.))"
      : "vec4f(solid,select(0.,1.,solid>0.),0.,0.)",
    /** Parent-level store. `mean:vec2f` and `maximum:vec2f` must be in scope. */
    storeParent: fluid
      ? "vec4f(mean.x,maximum.x,mean.y,maximum.y)"
      : "vec4f(mean.x,maximum.x,0.,0.)",
  };
}

/**
 * The two uniform lanes that place a radiance page.
 *
 * `radianceAtlas` is `xyz` = the radiance atlas page grid, `w` = the slot the
 * atlas begins at. `radianceScratch` is `xyz` = the compact radiance scratch
 * grid, `w` = the radiance floor level. Both atlases are addressed by the *same*
 * slot number the opacity pyramid uses — page validity, the lighting queue and
 * the black-page certificate all stay in that one index space — so the only
 * thing the floor changes is where a slot lands physically, and whether it
 * lands anywhere at all.
 */
const LIVE_SVO_RADIANCE_PARAMS_FIELDS = "radianceAtlas:vec4u,radianceScratch:vec4u";

/** Requires `params` and `PHYSICAL` to be in scope. */
const liveSvoRadianceAddressingWGSL = /* wgsl */ `
fn radianceFloorLevel()->u32{return params.radianceScratch.w;}
fn radianceSlotResident(slot:u32)->bool{return slot>=params.radianceAtlas.w;}
fn radianceSlotOrigin(slot:u32)->vec3u{
  let pages=params.radianceAtlas.xyz;let index=slot-params.radianceAtlas.w;
  return vec3u(index%pages.x,(index/pages.x)%pages.y,index/(pages.x*pages.y))*PHYSICAL;
}
fn radianceScratchOrigin(recordIndex:u32)->vec3u{
  let pages=params.radianceScratch.xyz;
  return vec3u(recordIndex%pages.x,(recordIndex/pages.x)%pages.y,recordIndex/(pages.x*pages.y))*PHYSICAL;
}
fn radianceLobeDepth()->u32{return params.radianceScratch.z*PHYSICAL;}
`;

/**
 * Deepest active leaf covering one finest-level cell.
 *
 * A copy of the worklist planner's descent, which the radiance floor now needs
 * on the build side too: a floor-level texel spans `2^floor` cells and no longer
 * matches the eight per-octant leaves a base record carries. Requires `control`,
 * `topology`, `ACTIVE`, `INVALID` and a `finestLevel()` accessor in scope.
 */
const liveSvoDeepestLeafWGSL = /* wgsl */ `
fn deepestLeaf(globalCell:vec3u)->u32{
  let brickSize=control[11];if(brickSize==0u||control[0]==0u){return INVALID;}
  let finest=finestLevel();let brick=globalCell/brickSize;var node=0u;var selected=INVALID;
  for(var level=0u;level<=finest&&node<control[0];level+=1u){
    let nodeBase=node*8u;let leaf=topology[nodeBase+6u];
    if(leaf<control[1]&&(topology[nodeBase+7u]&ACTIVE)!=0u){selected=leaf;}
    if(level==finest){break;}
    let bit=finest-level-1u;
    let octant=((brick.x>>bit)&1u)|(((brick.y>>bit)&1u)<<1u)|(((brick.z>>bit)&1u)<<2u);
    let mask=topology[nodeBase+3u]&0xffu;if((mask&(1u<<octant))==0u){break;}
    node=topology[nodeBase+4u]+countOneBits(mask&((1u<<octant)-1u));}
  return selected;}
`;

/**
 * Radiance storage declarations follow the atlas the device actually allocated.
 *
 * The lobe value is `vec3f` everywhere in this shader; the fourth component of
 * the `textureStore` operand is discarded by a three-channel format and was a
 * literal `1.0` in the four-channel one, so the WGSL below is the *same*
 * arithmetic under either declaration — only the stored precision differs.
 */
export function liveSvoDerivedBuildWGSLFor(
  profile: SparseBrickPayloadProfileName = "full",
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  opacityFormat: GPUTextureFormat = SVO_NODE_MIP_OPACITY_STORAGE.wideFormat,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  /**
   * The finest cell size, present only when the banded reconstruction arm is on.
   * `undefined` — the default — emits the shipped `solidDistance` unchanged.
   */
  bandedReconstructionCellSize_m?: readonly [number, number, number],
  /** How the world stores scene identity. Decides which identity decode compiles. */
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
  includeRadiance = true,
): string {
  const lane = derivedLaneAccess(profile, sceneGeometryFormat, leafPayloadMode);
  const solidDistance = derivedSolidDistanceWGSL(lane, bandedReconstructionCellSize_m);

  const opacity = derivedOpacityLanes(opacityFormat);
  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read> emission:array<vec4f>;
@group(0) @binding(5) var opacitySource:texture_3d<f32>;
@group(0) @binding(6) var opacityScratch:texture_storage_3d<${opacityFormat},write>;
@group(0) @binding(7) var radianceSource0:texture_3d<f32>;
@group(0) @binding(8) var radianceSource1:texture_3d<f32>;
@group(0) @binding(9) var radianceSource2:texture_3d<f32>;
@group(0) @binding(10) var radianceSource3:texture_3d<f32>;
@group(0) @binding(11) var radianceScratch:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(12) var<uniform> params:Params;
@group(0) @binding(13) var opacityValidity:texture_2d<u32>;
@group(0) @binding(14) var radianceValidity:texture_2d<u32>;
@group(0) @binding(15) var<storage,read_write> scratchValidity:array<u32>;

const INVALID:u32=0xffffffffu;
const LEAF_SLOT_TAG:u32=${LIVE_SVO_DERIVED_WORKLIST.leafSlotTag}u;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;
const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const TETRA0:vec3f=vec3f(.577350269,.577350269,.577350269);
const TETRA1:vec3f=vec3f(.577350269,-.577350269,-.577350269);
const TETRA2:vec3f=vec3f(-.577350269,.577350269,-.577350269);
const TETRA3:vec3f=vec3f(-.577350269,-.577350269,.577350269);
const BUILD_THREADS:u32=${LIVE_SVO_DERIVED_BUDGET.buildThreads}u;

fn finestLevel()->u32{return params.scratchAtlasPages.w;}
${lane.codec}
${liveSvoRadianceAddressingWGSL}
${liveSvoDeepestLeafWGSL}
fn slotOrigin(slot:u32)->vec3u{
  let pages=params.targetAtlasPages.xyz;
  return vec3u(slot%pages.x,(slot/pages.x)%pages.y,slot/(pages.x*pages.y))*PHYSICAL;
}
fn scratchOrigin(recordIndex:u32)->vec3u{
  let pages=params.scratchAtlasPages.xyz;
  return vec3u(recordIndex%pages.x,(recordIndex/pages.x)%pages.y,recordIndex/(pages.x*pages.y))*PHYSICAL;
}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
// At a zero apron this is the identity, which is the point: the shell it used to
// strip held a clamped copy of the very texel the clamp already selects.
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(${APRON}u),vec3u(INTERIOR-1u+${APRON}u))-${APRON}u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{
  let leafBase=control[16]+leafIndex*4u;
  return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];
}
${solidDistance}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{
  let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));
  let gradient=vec3f(solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
    solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
  return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);
}
/**
 * Opacity of one finest-level cell, read straight out of the leaf that owns it.
 *
 * The same arithmetic the base level performs, minus the descent: the caller
 * already knows the leaf. This is what a reduction reads where the level below
 * holds no page — the interior of a coarse leaf, which publishes one page at
 * its own level and nothing beneath it.
 */
fn leafOpacityAt(leaf:u32,cell:vec3u)->vec4f{
  let sampleLocal=leafLocal(cell,leaf);let voxel=leafVoxel(leaf,sampleLocal);
  let dynamicSolid=${lane.dynamicCoverage};let sceneSolid=${lane.sceneCoverage};
  let fluid=${lane.fluidFraction};let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
  return ${opacity.storeBase};
}
/** The finest cell at the centre of one child-level texel of this record's page. */
fn childCell(record:u32,fine:vec3u)->vec3u{
  let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
  let childScale=1u<<(worklist[2]-1u);
  return (page*INTERIOR*2u+fine)*childScale+vec3u(childScale>>1u);
}
fn childSample(record:u32,childBase:u32,local:vec3u,sampleIndex:u32)->vec4f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+childBase+octant];
  if(child==INVALID){return vec4f(0.);}
  if((child&LEAF_SLOT_TAG)!=0u){return leafOpacityAt(child&~LEAF_SLOT_TAG,childCell(record,fine));}
  return textureLoad(opacitySource,vec3i(slotOrigin(child)+(fine%INTERIOR)+${APRON}u),0);
}
fn childRadiance(record:u32,local:vec3u,sampleIndex:u32,lobe:u32)->vec3f{
  let bit=vec3u(sampleIndex&1u,(sampleIndex>>1u)&1u,(sampleIndex>>2u)&1u);
  let fine=local*2u+bit;let octant=(fine.x/INTERIOR)|((fine.y/INTERIOR)<<1u)|((fine.z/INTERIOR)<<2u);
  let child=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u+octant];
  if(child==INVALID||(child&LEAF_SLOT_TAG)!=0u||!radianceSlotResident(child)){return vec3f(0.);}
  let coordinate=vec3i(radianceSlotOrigin(child)+(fine%INTERIOR)+${APRON}u);
  if(lobe==0u){return textureLoad(radianceSource0,coordinate,0).rgb;}
  if(lobe==1u){return textureLoad(radianceSource1,coordinate,0).rgb;}
  if(lobe==2u){return textureLoad(radianceSource2,coordinate,0).rgb;}
  return textureLoad(radianceSource3,coordinate,0).rgb;
}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){
  let depth=radianceLobeDepth();
  textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
  textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));
}
fn referencedChildrenReady(record:u32,childBase:u32,radianceChildren:bool)->bool{
  for(var childIndex=0u;childIndex<8u;childIndex+=1u){let child=worklist[record+childBase+childIndex];
    // A tagged child names a leaf, not a page: the payload it reads is the
    // publication's own and needs no page to have been built first.
    if(child==INVALID||(child&LEAF_SLOT_TAG)!=0u){continue;}
    if(textureLoad(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),child),0).x==0u){return false;}
    if(radianceChildren&&textureLoad(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),child),0).x==0u){return false;}}
  return true;
}
/**
 * Radiance for one texel of the level the radiance chain is *based* at.
 *
 * Below the floor there is no radiance page to reduce, so this level stops
 * being a reduction and becomes a source. Its texel spans \`2^level\` cells, and
 * the three things it needs come from three different places: the exact mean
 * coverage from the opacity reduction that has just run over its children, the
 * emitting material from a tree descent at the texel's centre cell, and the
 * outward normal from the gradient of that same coverage across the eight
 * children — the iso-surface normal *at the texel's own scale*, which is the
 * one a cone this coarse is asking about. A degenerate gradient (a buried or a
 * uniformly thin block) falls back to the leaf's own signed-distance normal.
 */
fn radianceBaseNormal(record:u32,childBase:u32,local:vec3u,leaf:u32,sampleLocal:vec3u)->vec3f{
  var gradient=vec3f(0.);
  for(var child=0u;child<8u;child+=1u){
    let sign=vec3f(vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u))*2.-1.;
    gradient+=sign*childSample(record,childBase,local,child).x;}
  if(dot(gradient,gradient)>1e-6){return -normalize(gradient);}
  if(leaf<control[1]){return safeNormal(leaf,sampleLocal);}
  return vec3f(0.,1.,0.);
}

fn buildTexel(index:u32){
  let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;
  let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);
  let record=worklist[3]+recordIndex*RECORD_WORDS;let destination=scratchOrigin(recordIndex)+physical;
  // A record the planner declined to resolve builds nothing, and certifies nothing.
  if(worklist[record]==INVALID){if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=0u;}return;}
  let radianceDestination=radianceScratchOrigin(recordIndex)+physical;
  let level=worklist[2];let floorLevel=radianceFloorLevel();
  if(level==0u){
    if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=worklist[1];}let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);let octant=(local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u);let leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+octant];if(leaf>=control[1]){textureStore(opacityScratch,destination,vec4f(0.));if(${includeRadiance}&&floorLevel==0u){writeRadiance(radianceDestination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));}return;}let sampleLocal=leafLocal(page*INTERIOR+local,leaf);
    let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=${lane.dynamicIdentity};let sceneIdentity=sceneIdentityAt(voxel);
    let dynamicSolid=${lane.dynamicCoverage};let sceneSolid=${lane.sceneCoverage};
    ${opacity.fluid ? `let fluid=${lane.fluidFraction};` : ""}let solid=1.-(1.-dynamicSolid)*(1.-sceneSolid);
    textureStore(opacityScratch,destination,${opacity.storeBase});
    if(!${includeRadiance}||floorLevel!=0u){return;}
    let material=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneSolid>=dynamicSolid&&sceneSolid>0.);var emitted=vec3f(0.);
    if(material<arrayLength(&emission)){emitted=max(emission[material].rgb,vec3f(0.));}
    let normal=safeNormal(leaf,sampleLocal);let covered=solid;
    writeRadiance(radianceDestination,emitted*covered*max(0.,dot(normal,TETRA0)),emitted*covered*max(0.,dot(normal,TETRA1)),
      emitted*covered*max(0.,dot(normal,TETRA2)),emitted*covered*max(0.,dot(normal,TETRA3)));
    return;
  }
  let radianceBase=level==floorLevel;
  // One layout at every level: the coordinate every parent now needs to resolve
  // a leaf-tagged child is where the radiance floor always kept it.
  let childBase=${LIVE_SVO_DERIVED_WORKLIST.childSlotWord}u;
  if(all(physical==vec3u(0u))){scratchValidity[recordIndex]=select(0u,worklist[1],referencedChildrenReady(record,childBase,${includeRadiance}&&level>floorLevel));}
  var mean=vec2f(0.);var maximum=vec2f(0.);var r0=vec3f(0.);var r1=vec3f(0.);var r2=vec3f(0.);var r3=vec3f(0.);
  for(var child=0u;child<8u;child+=1u){let value=childSample(record,childBase,local,child);mean+=value.xz/8.;maximum=max(maximum,value.yw);}
  if(${includeRadiance}&&level>floorLevel){
    for(var child=0u;child<8u;child+=1u){
      r0+=childRadiance(record,local,child,0u)/8.;r1+=childRadiance(record,local,child,1u)/8.;
      r2+=childRadiance(record,local,child,2u)/8.;r3+=childRadiance(record,local,child,3u)/8.;}
  }
  textureStore(opacityScratch,destination,${opacity.storeParent});
  if(!${includeRadiance}||level<floorLevel){return;}
  if(radianceBase){
    let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
    let scale=1u<<level;
    let cell=(page*INTERIOR+local)*scale+vec3u(scale>>1u);
    let leaf=deepestLeaf(cell);
    var emitted=vec3f(0.);var sampleLocal=vec3u(0u);
    if(leaf<control[1]){
      sampleLocal=leafLocal(cell,leaf);
      let material=sceneIdentityAt(leafVoxel(leaf,sampleLocal))&0xffffu;
      if(material<arrayLength(&emission)){emitted=max(emission[material].rgb,vec3f(0.));}}
    let normal=radianceBaseNormal(record,childBase,local,leaf,sampleLocal);
    r0=emitted*mean.x*max(0.,dot(normal,TETRA0));r1=emitted*mean.x*max(0.,dot(normal,TETRA1));
    r2=emitted*mean.x*max(0.,dot(normal,TETRA2));r3=emitted*mean.x*max(0.,dot(normal,TETRA3));
  }
  writeRadiance(radianceDestination,r0,r1,r2,r3);
}
// A fixed dispatch: every thread strides the batch's texels, and the batch's
// count — read here, never by the host — is what bounds the work.
@compute @workgroup_size(256)
fn buildPages(@builtin(global_invocation_id) gid:vec3u){
  let total=min(worklist[0],params.limits.x)*PHYSICAL*PHYSICAL*PHYSICAL;
  for(var index=gid.x;index<total;index+=BUILD_THREADS){buildTexel(index);}
}`;
}

/** The `full` expansion. Byte-identical to the pre-profile shader. */
export const liveSvoDerivedBuildWGSL = liveSvoDerivedBuildWGSLFor("full");

/**
 * A deliberately low-rate Jacobi-style diffuse solve. Floor pages read the
 * previous complete radiance atlas and write emission + reflected incident
 * light into compact scratch; their parents are ordinary builds, reduced from
 * the newly published children. The recurrence replaces, rather than adds to,
 * previous radiance, so energy cannot grow merely because more frames have
 * elapsed.
 *
 * It runs inside every round, after the batch's opacity has been published and
 * before its radiance is, and does nothing unless the batch is the radiance
 * floor. So a floor page's first build already holds one bounce, and a feedback
 * rotation is the same build queued again.
 */
export function liveSvoRadianceFeedbackWGSLFor(
  profile: SparseBrickPayloadProfileName = "full",
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  sceneGeometryFormat: SparseBrickSceneGeometryFormat = "f32x2",
  bandedReconstructionCellSize_m?: readonly [number, number, number],
  solidMarchOffsetFix = false,
  solidDirectOcclusion = false,
  /** How the world stores scene identity. Decides which identity decode compiles. */
  leafPayloadMode: SparseBrickLeafPayloadMode = "dense",
): string {
  const lane = derivedLaneAccess(profile, sceneGeometryFormat, leafPayloadMode);
  // The offset rule and the direct-occlusion rule both read the fraction, so the
  // shared helpers must be in scope whenever either is on, not only when banding
  // is.
  const solidDistance = derivedSolidDistanceWGSL(
    lane, bandedReconstructionCellSize_m, solidMarchOffsetFix || solidDirectOcclusion);
  const marchOffsetCells = derivedMarchOffsetWGSL(solidMarchOffsetFix);
  const directOcclusion = derivedSolidDirectOcclusionWGSL(solidDirectOcclusion);

  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
${svoMaterialWGSL}
${svoEnvironmentLightingWGSL}
${svoLightWGSL}
${svoTetrahedralRadianceWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,direct:vec4u,reserved:array<vec4u,3>,mappingOrigin:vec4f,mappingCellSize:vec4f,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var opacitySource:texture_3d<f32>;
@group(0) @binding(5) var radianceSource0:texture_3d<f32>;
@group(0) @binding(6) var radianceSource1:texture_3d<f32>;
@group(0) @binding(7) var radianceSource2:texture_3d<f32>;
@group(0) @binding(8) var radianceSource3:texture_3d<f32>;
@group(0) @binding(9) var radianceScratch:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(10) var<uniform> params:Params;
@group(0) @binding(11) var pageTable:texture_2d<u32>;
@group(0) @binding(12) var<storage,read> materials:array<SvoMaterialRecord>;
@group(0) @binding(13) var<storage,read> environment:array<SvoEnvironmentLightingRecord>;
@group(0) @binding(14) var opacityValidity:texture_2d<u32>;
@group(0) @binding(15) var radianceValidity:texture_2d<u32>;
@group(0) @binding(17) var<storage,read> lights:array<SvoLightRecord>;
@group(0) @binding(18) var atlasSampler:sampler;

const INVALID:u32=0xffffffffu;const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const INTERIOR:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;const PI:f32=3.141592653589793;
const BUILD_THREADS:u32=${LIVE_SVO_DERIVED_BUDGET.buildThreads}u;
fn finestLevel()->u32{return params.scratchAtlasPages.w;}
${lane.codec}
${liveSvoRadianceAddressingWGSL}
${liveSvoDeepestLeafWGSL}
fn slotOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(recordIndex:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(recordIndex%p.x,(recordIndex/p.x)%p.y,recordIndex/(p.x*p.y))*PHYSICAL;}
fn physicalCoordinate(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
// At a zero apron this is the identity, which is the point: the shell it used to
// strip held a clamped copy of the very texel the clamp already selects.
fn interiorCoordinate(physical:vec3u)->vec3u{return clamp(physical,vec3u(${APRON}u),vec3u(INTERIOR-1u+${APRON}u))-${APRON}u;}
fn keyBit(lo:u32,hi:u32,bit:u32)->u32{if(bit>=32u){return(hi>>(bit-32u))&1u;}return(lo>>bit)&1u;}
fn morton(lo:u32,hi:u32,level:u32)->vec3u{var p=vec3u(0u);for(var bit=0u;bit<level;bit+=1u){let s=1u<<bit;p+=vec3u(keyBit(lo,hi,3u*bit),keyBit(lo,hi,3u*bit+1u),keyBit(lo,hi,3u*bit+2u))*s;}return p;}
fn leafLocal(globalCell:vec3u,leaf:u32)->vec3u{let brickSize=control[11];let finest=params.scratchAtlasPages.w;let node=topology[control[16]+leaf*4u];let nodeBase=node*8u;let level=topology[nodeBase+2u];let scale=1u<<(finest-level);let origin=morton(topology[nodeBase],topology[nodeBase+1u],level)*scale*brickSize;return min((globalCell-origin)/scale,vec3u(brickSize-1u));}
fn leafVoxel(leafIndex:u32,local:vec3u)->u32{let leafBase=control[16]+leafIndex*4u;return topology[leafBase+1u]+local.x+local.y*control[11]+local.z*control[11]*control[11];}
${solidDistance}
fn safeNormal(leafIndex:u32,local:vec3u)->vec3f{let lo=max(local,vec3u(1u))-1u;let hi=min(local+1u,vec3u(control[11]-1u));let gradient=vec3f(
 solidDistance(leafIndex,vec3u(hi.x,local.y,local.z))-solidDistance(leafIndex,vec3u(lo.x,local.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,hi.y,local.z))-solidDistance(leafIndex,vec3u(local.x,lo.y,local.z)),
 solidDistance(leafIndex,vec3u(local.x,local.y,hi.z))-solidDistance(leafIndex,vec3u(local.x,local.y,lo.z)));
 return select(vec3f(0.,1.,0.),normalize(gradient),dot(gradient,gradient)>1e-12);}
fn writeRadiance(coordinate:vec3u,a:vec3f,b:vec3f,c:vec3f,d:vec3f){let depth=radianceLobeDepth();
 textureStore(radianceScratch,coordinate,vec4f(a,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth),vec4f(b,1.));
 textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*2u),vec4f(c,1.));textureStore(radianceScratch,coordinate+vec3u(0u,0u,depth*3u),vec4f(d,1.));}
${svoNodeMipPageHashWGSL}
fn tableSlot(level:u32,page:vec3u)->u32{if(level>=params.direct.x){return INVALID;}return svoNodeMipPageHashFind(pageTable,level,page);}
struct SceneSample{coverage:f32,radiance:SvoTetraRadiance,valid:u32}
// Callers must pass a level at or above the radiance floor; below it there is no
// radiance page to gather from, and the opacity there is finer than this solve
// resolves anyway.
//
// A page awaiting its build holds nothing a gather may read, and is treated
// like a page that is not there: it neither occludes nor emits. The one case
// worth separating is a page whose opacity is built and whose radiance is not —
// it occludes, and contributes no light yet. Either way the gather goes on;
// a page that reads its neighbour too early is inside that neighbour's window
// and is gathered again once it is built.
fn sceneSample(positionIn:vec3f,level:u32)->SceneSample{if(any(positionIn<vec3f(0.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let scale=exp2(f32(level));let levelVoxel=positionIn/scale;let pageFloor=floor(levelVoxel/f32(INTERIOR));if(any(pageFloor<vec3f(0.0))||any(pageFloor>=vec3f(2097152.0))){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let page=vec3u(pageFloor);let slot=tableSlot(level,page);if(slot==INVALID||!radianceSlotResident(slot)){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let opacityDimensions=textureDimensions(opacityValidity);let radianceDimensions=textureDimensions(radianceValidity);if(textureLoad(opacityValidity,svoDerivedPageValidityTexel(opacityDimensions,slot),0).x==0u){return SceneSample(0.,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}let radianceBuilt=textureLoad(radianceValidity,svoDerivedPageValidityTexel(radianceDimensions,slot),0).x!=0u;let interiorTexel=clamp(levelVoxel-vec3f(page)*f32(INTERIOR)-vec3f(.5),vec3f(${INTERIOR_CLAMP_LOW}),vec3f(f32(INTERIOR)-${INTERIOR_CLAMP_HIGH}));let opacityUv=(vec3f(slotOrigin(slot))+vec3f(f32(${APRON}u)+.5)+interiorTexel)/vec3f(textureDimensions(opacitySource));let radianceUv=(vec3f(radianceSlotOrigin(slot))+vec3f(f32(${APRON}u)+.5)+interiorTexel)/vec3f(textureDimensions(radianceSource0));if(!radianceBuilt){return SceneSample(textureSampleLevel(opacitySource,atlasSampler,opacityUv,0.0).x,SvoTetraRadiance(vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.)),1u);}return SceneSample(textureSampleLevel(opacitySource,atlasSampler,opacityUv,0.0).x,SvoTetraRadiance(textureSampleLevel(radianceSource0,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource1,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource2,atlasSampler,radianceUv,0.0).rgb,textureSampleLevel(radianceSource3,atlasSampler,radianceUv,0.0).rgb),1u);}
fn hemisphereDirection(normal:vec3f,index:u32)->vec3f{var basis=svoTetraDirection0();if(index==1u){basis=svoTetraDirection1();}else if(index==2u){basis=svoTetraDirection2();}else if(index==3u){basis=svoTetraDirection3();}basis=select(-basis,basis,dot(basis,normal)>=0.0);return normalize(normal*.45+basis);}
// The four gather steps are expressed in *receiver* widths, not in finest
// voxels: the base of the radiance chain is now a floor-level texel spanning
// 2^floor cells, and a 2.5-cell first step from an eight-cell block lands inside
// the block that emitted it. Scaling the distances and shifting the level ladder
// by the same factor keeps every step's ratio of distance to sample width
// exactly what it was when the chain was based at the leaf.
fn incidentAlong(origin:vec3f,normal:vec3f,index:u32)->vec3f{let direction=hemisphereDirection(normal,index);var transmittance=1.0;var incident=vec3f(0.0);
 let floorLevel=radianceFloorLevel();let floorScale=exp2(f32(floorLevel));
 for(var step=0u;step<4u;step+=1u){var distance=2.5;if(step==1u){distance=6.0;}else if(step==2u){distance=18.0;}else if(step==3u){distance=54.0;}let level=min(step+floorLevel,params.direct.x-1u);let sample=sceneSample(origin+direction*distance*floorScale,level);let coverage=clamp(sample.coverage,0.0,1.0);incident+=transmittance*svoTetraRadianceAlong(sample.radiance,-direction);transmittance*=1.0-coverage;if(transmittance<=0.01){break;}}
 return incident+transmittance*svoEnvironmentDiffuseIrradiance(environment[0],normal)/PI;}
fn directVisibility(originCells:vec3f,towardLight:vec3f,maximumDistance_m:f32${directOcclusion.parameter})->f32{${directOcclusion.guard}let cellScale=max(params.mappingCellSize.x,max(params.mappingCellSize.y,params.mappingCellSize.z))*exp2(f32(radianceFloorLevel()));var transmittance=1.0;for(var step=0u;step<4u;step+=1u){var distance_m=cellScale*2.5;if(step==1u){distance_m=cellScale*6.0;}else if(step==2u){distance_m=cellScale*18.0;}else if(step==3u){distance_m=cellScale*54.0;}if(maximumDistance_m>0.0){distance_m=min(distance_m,maximumDistance_m*.9);}let position=originCells+towardLight*distance_m/params.mappingCellSize.xyz;let sample=sceneSample(position,min(step+radianceFloorLevel(),params.direct.x-1u));transmittance*=1.0-clamp(sample.coverage,0.0,1.0);}return clamp(transmittance,0.0,1.0);}
fn directIncident(worldPosition:vec3f,originCells:vec3f,normal:vec3f${directOcclusion.parameter})->vec3f{var result=vec3f(0.0);let count=min(min(params.direct.w,arrayLength(&lights)),4u);for(var index=0u;index<4u;index+=1u){if(index>=count){break;}let light=lights[index];let base=svoLightRadiance(light);var toward=light.directionCone.xyz;var maximumDistance_m=0.0;var attenuation=1.0;if(light.identity.x!=SVO_LIGHT_DIRECTIONAL){let offset=light.positionRange.xyz-worldPosition;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10||(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w)){continue;}maximumDistance_m=sqrt(distanceSquared);toward=offset/maximumDistance_m;let rangeFade=select(1.0,pow(clamp(1.0-maximumDistance_m/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);attenuation=rangeFade/max(1.0,distanceSquared);if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*PI*light.shape.x*light.shape.x;attenuation=rangeFade*area/max(area,distanceSquared);}else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;attenuation=rangeFade*max(dot(normalize(light.directionCone.xyz),-toward),0.0)*area/max(area,distanceSquared);}else if(light.identity.x==SVO_LIGHT_SPOT){attenuation*=svoLightConeFalloff(light,toward);}}let cosine=max(dot(normal,toward),0.0);if(cosine>0.0&&directVisibility(originCells,toward,maximumDistance_m${directOcclusion.forward})>0.0){result+=base*(attenuation*cosine/PI);}}return result;}
/**
 * Outward normal at the scale of one floor-level texel, from its own opacity page.
 *
 * The taps are clamped inside the page rather than allowed to walk into an
 * apron. That is not an approximation of the old behaviour, it *is* the old
 * behaviour: an apron texel held a clamped copy of the edge texel, so a tap into
 * it and a tap clamped to the edge read the same byte. Without the clamp a zero
 * apron would read the adjacent atlas slot, which is a different page of the
 * world.
 */
fn pageGradientNormal(pageOrigin:vec3u,center:vec3u)->vec3f{
 let lo=max(center,vec3u(1u))-vec3u(1u);
 let hi=min(center+vec3u(1u),vec3u(PHYSICAL-1u));
 let gradient=vec3f(
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(hi.x,center.y,center.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(lo.x,center.y,center.z)),0).x,
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,hi.y,center.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,lo.y,center.z)),0).x,
  textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,center.y,hi.z)),0).x-textureLoad(opacitySource,vec3i(pageOrigin+vec3u(center.x,center.y,lo.z)),0).x);
 if(dot(gradient,gradient)<=1e-8){return vec3f(0.);}
 return -normalize(gradient);}
fn feedbackTexel(index:u32){let texels=PHYSICAL*PHYSICAL*PHYSICAL;let recordIndex=index/texels;let physical=physicalCoordinate(index-recordIndex*texels);let local=interiorCoordinate(physical);let record=worklist[3]+recordIndex*RECORD_WORDS;let level=worklist[2];let destination=radianceScratchOrigin(recordIndex)+physical;
 // The page's own opacity was published by this round's copy, so it is read
 // where every other gather will read it. A record the planner declined to
 // resolve names no slot.
 let slot=worklist[record];if(slot==INVALID){return;}let pageOrigin=slotOrigin(slot);let center=local+vec3u(${APRON}u);let coverage=textureLoad(opacitySource,vec3i(pageOrigin+center),0).x;let page=vec3u(worklist[record+1u],worklist[record+2u],worklist[record+3u]);
 // At the floor level one texel spans 2^level cells, so the eight per-octant
 // leaves a base record carries no longer address it; the material comes from a
 // descent at the texel's own centre cell instead.
 let scale=1u<<level;let globalCell=(page*INTERIOR+local)*scale+vec3u(scale>>1u);
 var leaf=worklist[record+${LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord}u+((local.x/4u)|((local.y/4u)<<1u)|((local.z/4u)<<2u))];
 if(level>0u){leaf=deepestLeaf(globalCell);}
 if(coverage<=0.0||leaf>=control[1]){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let sampleLocal=leafLocal(globalCell,leaf);
 let voxel=leafVoxel(leaf,sampleLocal);let dynamicIdentity=${lane.dynamicIdentity};let sceneIdentity=sceneIdentityAt(voxel);let dynamicCoverage=${lane.dynamicCoverage};let sceneCoverage=${lane.sceneCoverage};let materialIndex=select(dynamicIdentity&0xffffu,sceneIdentity&0xffffu,sceneCoverage>=dynamicCoverage&&sceneCoverage>0.);if(materialIndex>=arrayLength(&materials)){writeRadiance(destination,vec3f(0.),vec3f(0.),vec3f(0.),vec3f(0.));return;}let material=materials[materialIndex];let fineNormal=safeNormal(leaf,sampleLocal);let coarseNormal=pageGradientNormal(pageOrigin,center);let normal=select(fineNormal,coarseNormal,level>0u&&dot(coarseNormal,coarseNormal)>0.5);let originCells=vec3f(globalCell)+vec3f(.5)+normal*${marchOffsetCells};let transportAlbedo=clamp(material.baseColorOpacity.rgb*(1.0-clamp(material.surface.x,0.0,1.0)),vec3f(0.0),vec3f(${LIVE_SVO_RADIANCE_FEEDBACK.maximumTransportAlbedo}));var incident=vec3f(0.0);for(var direction=0u;direction<${LIVE_SVO_RADIANCE_FEEDBACK.directionCount}u;direction+=1u){incident+=incidentAlong(originCells,normal,direction)/f32(${LIVE_SVO_RADIANCE_FEEDBACK.directionCount});}let worldPosition=params.mappingOrigin.xyz+(vec3f(globalCell)+vec3f(.5))*params.mappingCellSize.xyz;incident+=directIncident(worldPosition,originCells,normal${directOcclusion.receiver});let outgoing=max(material.emissiveRoughness.rgb,vec3f(0.0))+transportAlbedo*incident;writeRadiance(destination,outgoing*coverage*max(0.,dot(normal,svoTetraDirection0())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection1())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection2())),outgoing*coverage*max(0.,dot(normal,svoTetraDirection3())));
}
// The same fixed dispatch as the build. Only a batch of floor pages gathers.
@compute @workgroup_size(256) fn feedbackPages(@builtin(global_invocation_id) gid:vec3u){
 if(worklist[2]!=radianceFloorLevel()){return;}
 let total=min(worklist[0],params.limits.x)*PHYSICAL*PHYSICAL*PHYSICAL;
 for(var index=gid.x;index<total;index+=BUILD_THREADS){feedbackTexel(index);}
}`;
}

/** The `full` expansion. Byte-identical to the pre-profile shader. */
export const liveSvoRadianceFeedbackWGSL = liveSvoRadianceFeedbackWGSLFor("full");

export function liveSvoDerivedCopyWGSLFor(
  radianceFormat: GPUTextureFormat = LIVE_SVO_RADIANCE_STORAGE.fallbackFormat,
  opacityFormat: GPUTextureFormat = SVO_NODE_MIP_OPACITY_STORAGE.wideFormat,
): string {
  return /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var<storage,read> worklist:array<u32>;
@group(0) @binding(1) var opacityScratch:texture_3d<f32>;
@group(0) @binding(2) var opacityDestination:texture_storage_3d<${opacityFormat},write>;
@group(0) @binding(3) var radianceScratch:texture_3d<f32>;
// Two lobes a dispatch: a stage may write four storage textures, and the
// second pair shares its dispatch with the validity word that publishes the page.
@group(0) @binding(4) var radianceDestinationA:texture_storage_3d<${radianceFormat},write>;
@group(0) @binding(5) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(6) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(7) var<uniform> params:Params;
@group(0) @binding(8) var<storage,read> scratchValidity:array<u32>;
@group(0) @binding(9) var radianceDestinationB:texture_storage_3d<${radianceFormat},write>;
const PHYSICAL:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;const RECORD_WORDS:u32=${LIVE_SVO_DERIVED_WORKLIST.recordWords}u;
const INVALID:u32=0xffffffffu;const BUILD_THREADS:u32=${LIVE_SVO_DERIVED_BUDGET.buildThreads}u;
${liveSvoRadianceAddressingWGSL}
fn targetOrigin(slot:u32)->vec3u{let p=params.targetAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn scratchOrigin(slot:u32)->vec3u{let p=params.scratchAtlasPages.xyz;return vec3u(slot%p.x,(slot/p.x)%p.y,slot/(p.x*p.y))*PHYSICAL;}
fn physical(index:u32)->vec3u{return vec3u(index%PHYSICAL,(index/PHYSICAL)%PHYSICAL,index/(PHYSICAL*PHYSICAL));}
fn batchTexels()->u32{return min(worklist[0],params.limits.x)*PHYSICAL*PHYSICAL*PHYSICAL;}
// Publishing the validity word is what ends a page's "awaiting build": it is
// written with the page's last texels, and only when its children were ready.
@compute @workgroup_size(256) fn copyOpacity(@builtin(global_invocation_id) gid:vec3u){
 let total=batchTexels();let texels=PHYSICAL*PHYSICAL*PHYSICAL;
 for(var index=gid.x;index<total;index+=BUILD_THREADS){
  let recordIndex=index/texels;let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];if(slot==INVALID){continue;}
  let source=vec3i(scratchOrigin(recordIndex)+local);let destination=vec3i(targetOrigin(slot)+local);
  textureStore(opacityDestination,destination,textureLoad(opacityScratch,source,0));if(all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(scratchValidity[recordIndex]));}}}
fn copyRadianceLobes(thread:u32,firstLobe:u32){
 let total=batchTexels();let texels=PHYSICAL*PHYSICAL*PHYSICAL;
 for(var index=thread;index<total;index+=BUILD_THREADS){
  let recordIndex=index/texels;let local=physical(index-recordIndex*texels);let record=worklist[3]+recordIndex*RECORD_WORDS;let slot=worklist[record];if(slot==INVALID){continue;}
  let scratch=radianceScratchOrigin(recordIndex)+local+vec3u(0u,0u,firstLobe*radianceLobeDepth());
  // Below the floor there is no radiance page to publish, but the level still
  // certifies itself valid: an absent page and a black page answer every cone
  // identically, and a page that never goes valid stalls its parents forever.
  if(radianceSlotResident(slot)){let destination=vec3i(radianceSlotOrigin(slot)+local);
   textureStore(radianceDestinationA,destination,textureLoad(radianceScratch,vec3i(scratch),0));
   textureStore(radianceDestinationB,destination,textureLoad(radianceScratch,vec3i(scratch+vec3u(0u,0u,radianceLobeDepth())),0));}
  if(firstLobe==2u&&all(local==vec3u(0u))&&scratchValidity[recordIndex]!=0u){textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(scratchValidity[recordIndex]));}}}
@compute @workgroup_size(256) fn copyRadianceLow(@builtin(global_invocation_id) gid:vec3u){copyRadianceLobes(gid.x,0u);}
@compute @workgroup_size(256) fn copyRadianceHigh(@builtin(global_invocation_id) gid:vec3u){copyRadianceLobes(gid.x,2u);}`;
}

/** The `rgba16float` expansion. Byte-identical to the pre-format-choice shader. */
export const liveSvoDerivedCopyWGSL = liveSvoDerivedCopyWGSLFor();

/**
 * New WebGPU textures are logically zero-initialized. This runtime pass
 * certifies those zero pages against the current live publication before any
 * occupied-page worklist invalidates and overwrites them. It is deliberately
 * independent of scene contents: every address-resident page starts as empty,
 * and ordinary dirty propagation owns every later edit.
 */
export const liveSvoDerivedEmptyInitializationWGSL = /* wgsl */ `
${liveSvoDerivedPageValidityWGSL}
struct Params{targetAtlasPages:vec4u,scratchAtlasPages:vec4u,limits:vec4u,laneOffsets:vec4u,${LIVE_SVO_RADIANCE_PARAMS_FIELDS},bandedLanes:vec4u}
@group(0) @binding(0) var opacityValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(1) var radianceValidity:texture_storage_2d<r32uint,write>;
@group(0) @binding(2) var<storage,read> generationState:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@compute @workgroup_size(256) fn initializeValidEmptyPages(@builtin(global_invocation_id) gid:vec3u){
  let slot=gid.x;if(slot>=params.limits.w){return;}let generation=generationState[params.limits.z];if(generation==0u){return;}
  textureStore(opacityValidity,svoDerivedPageValidityTexel(textureDimensions(opacityValidity),slot),vec4u(generation));
  textureStore(radianceValidity,svoDerivedPageValidityTexel(textureDimensions(radianceValidity),slot),vec4u(generation));
}`;

/** One pipeline and the one bind group it is ever dispatched with. */
interface LiveSvoDerivedStage { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup }

interface LiveSvoDerivedBuilderPipelineState {
  emptyInitialization: LiveSvoDerivedStage;
  build: LiveSvoDerivedStage;
  opacityBuild: LiveSvoDerivedStage;
  feedback?: LiveSvoDerivedStage;
  opacityCopy: LiveSvoDerivedStage;
  /** Lobes 0-1, then lobes 2-3 with the page's validity. */
  radianceCopy: readonly [LiveSvoDerivedStage, LiveSvoDerivedStage];
}

function compactScratchAtlasPages(capacity: number, maximumTextureDimension3D: number): readonly [number, number, number] {
  const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
  const maximumXyPages = Math.floor(maximumTextureDimension3D / physical);
  const maximumZPages = Math.floor(maximumTextureDimension3D / (physical * 4));
  if (maximumXyPages < 1 || maximumZPages < 1 || capacity > maximumXyPages * maximumXyPages * maximumZPages) {
    throw new RangeError("Live SVO derived scratch work budget exceeds the device 3D texture limit");
  }
  const x = Math.min(capacity, maximumXyPages);
  const y = Math.min(Math.ceil(capacity / x), maximumXyPages);
  return [x, y, Math.ceil(capacity / (x * y))];
}

/**
 * GPU-only incremental builder: builds whatever batch the planner has prepared.
 *
 * Every dispatch is direct and the same size on every frame; the batch's count,
 * read in the shader, is what bounds the work.
 */
export class WebGpuLiveSvoDerivedBuilder {
  readonly allocatedBytes: number;
  readonly scratchAtlasPages: readonly [number, number, number];
  /** Sized by the deepest *radiance* level, which the floor makes far shallower. */
  readonly radianceScratchAtlasPages: readonly [number, number, number];
  readonly radianceFloorLevel: number;
  readonly radianceFeedbackEnabled: boolean;
  /** Mirrors the target atlas; scratch, copy destination and WGSL must agree. */
  readonly radianceFormat: GPUTextureFormat;
  /** Mirrors the opacity atlas; scratch, copy destination and WGSL must agree. */
  readonly opacityFormat: GPUTextureFormat;
  private readonly plannedPageCount: number;
  private readonly opacityScratch: GPUTexture;
  private readonly radianceScratch: GPUTexture;
  private readonly scratchValidity: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly feedbackParams?: GPUBuffer;
  private readonly feedbackSampler?: GPUSampler;
  private pipelineState?: LiveSvoDerivedBuilderPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedBuilderOptions) {
    // The two atlases share a *slot* index space — page validity, the lighting
    // queue and every build record name one slot for both — but no longer a
    // shape: the radiance floor gives radiance far fewer physical pages.
    if (options.nodeMips.pageCapacity !== options.radiance.pageCapacity) throw new Error("Live opacity and radiance page capacities must match");
    if (options.nodeMips.atlasPages.reduce((product, value) => product * value, 1) < options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived target atlas does not contain its declared page capacity");
    }
    if (!Number.isInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > SVO_NODE_MIP_MAXIMUM_LEVELS) {
      throw new RangeError(`Live derived level count must be in [1, ${SVO_NODE_MIP_MAXIMUM_LEVELS}]`);
    }
    if (!Number.isSafeInteger(options.plannedPageCount) || options.plannedPageCount < 1
      || options.plannedPageCount > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived planned page count must fit the target page atlas");
    }
    this.plannedPageCount = options.plannedPageCount;
    if (!Number.isInteger(options.generationSource.offsetBytes) || options.generationSource.offsetBytes < 0
      || options.generationSource.offsetBytes % 4 !== 0) {
      throw new RangeError("Live derived generation offset must be a nonnegative u32-aligned byte offset");
    }
    if (!Number.isInteger(options.finestLevel) || options.finestLevel < 0 || options.finestLevel > 21) {
      throw new RangeError("Live derived finest level must be an integer in [0, 21]");
    }
    this.radianceFeedbackEnabled = options.radianceFeedback
      ?? Boolean(options.materialPbr && options.environmentLighting && options.lights);
    if (this.radianceFeedbackEnabled && (!options.materialPbr || !options.environmentLighting || !options.lights)) {
      throw new Error("Live SVO radiance feedback requires material, environment, and light buffers");
    }
    if (!Number.isSafeInteger(options.lightCount ?? 0) || (options.lightCount ?? 0) < 0) {
      throw new RangeError("Live SVO feedback light count must be a nonnegative safe integer");
    }
    if ((options.worldOrigin_m ?? [0, 0, 0]).some((value) => !Number.isFinite(value))
      || (options.cellSize_m ?? [1, 1, 1]).some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new RangeError("Live SVO feedback mapping must have a finite origin and positive cell size");
    }
    // Scratch holds one batch, not one level: the largest level of a wide world
    // is tens of thousands of pages, and staging them all at once is what made
    // this the largest allocation in the lighting path.
    const scratchCapacity = options.batch.capacity;
    if (!Number.isSafeInteger(scratchCapacity) || scratchCapacity < 1 || scratchCapacity > options.nodeMips.pageCapacity) {
      throw new RangeError("Live derived batch capacity must fit the target page atlas");
    }
    const maximumTextureDimension3D = Number(device.limits?.maxTextureDimension3D) || 2_048;
    this.scratchAtlasPages = options.scratchAtlasPages ?? compactScratchAtlasPages(scratchCapacity, maximumTextureDimension3D);
    if (this.scratchAtlasPages.some((value) => !Number.isSafeInteger(value) || value < 1)
      || this.scratchAtlasPages.reduce((product, value) => product * value, 1) < scratchCapacity
      || this.scratchAtlasPages[0] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[1] * SVO_NODE_MIP_LAYOUT.physicalSize > maximumTextureDimension3D
      || this.scratchAtlasPages[2] * SVO_NODE_MIP_LAYOUT.physicalSize * 4 > maximumTextureDimension3D) {
      throw new RangeError("Live derived scratch atlas cannot contain one batch on this device");
    }
    const label = options.label ?? "Live SVO derived content";
    this.radianceFormat = options.radiance.format ?? LIVE_SVO_RADIANCE_STORAGE.fallbackFormat;
    this.opacityFormat = options.nodeMips.format ?? SVO_NODE_MIP_OPACITY_STORAGE.wideFormat;
    this.radianceFloorLevel = options.radianceFloorLevel ?? 0;
    if (!Number.isSafeInteger(this.radianceFloorLevel) || this.radianceFloorLevel < 0
      || this.radianceFloorLevel >= options.levelCount) {
      throw new RangeError("Live derived radiance floor must be a level the page table covers");
    }
    const radianceScratchCapacity = Math.max(1, Math.min(scratchCapacity, options.radianceScratchCapacity ?? scratchCapacity));
    this.radianceScratchAtlasPages = compactScratchAtlasPages(radianceScratchCapacity, maximumTextureDimension3D);
    const scratchTexels = this.scratchAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number];
    const radianceScratchTexels = this.radianceScratchAtlasPages.map((value) => value * SVO_NODE_MIP_LAYOUT.physicalSize) as [number, number, number];
    this.opacityScratch = device.createTexture({ label: `${label} opacity scratch`, size: scratchTexels, dimension: "3d", format: this.opacityFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.radianceScratch = device.createTexture({ label: `${label} radiance scratch`, size: [radianceScratchTexels[0], radianceScratchTexels[1], radianceScratchTexels[2] * 4],
      dimension: "3d", format: this.radianceFormat, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.scratchValidity = device.createBuffer({ label: `${label} scratch page validity`, size: scratchCapacity * 4,
      usage: GPUBufferUsage.STORAGE });
    // The banded lane bases ride at the end of both parameter blocks, so every
    // other field stays exactly where it was.
    const bandedLanes = options.tree.bandedLaneWordOffsets.slice(0, 4);
    this.params = device.createBuffer({ label: `${label} parameters`, size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.nodeMips.atlasPages, 0,
      ...this.scratchAtlasPages, options.finestLevel,
      // `limits.x` bounds the records a dispatch will touch: the batch, which
      // is also exactly what the scratch can hold.
      scratchCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
      options.generationSource.offsetBytes / 4, options.plannedPageCount,
      options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
      options.tree.sceneGeometryOffsetBytes / 4, options.tree.scenePayloadLanes.materialOwnerWords,
      ...options.radiance.atlasPages, options.radiance.slotOffset ?? 0,
      ...this.radianceScratchAtlasPages, this.radianceFloorLevel,
      ...bandedLanes,
    ]));
    if (this.radianceFeedbackEnabled) {
      this.feedbackSampler = device.createSampler({ label: `${label} radiance feedback sampler`,
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge",
        minFilter: "linear", magFilter: "linear", mipmapFilter: "nearest" });
      this.feedbackParams = device.createBuffer({ label: `${label} radiance feedback parameters`, size: 208,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const feedbackData = new ArrayBuffer(208), feedbackWords = new Uint32Array(feedbackData), feedbackFloats = new Float32Array(feedbackData);
      feedbackWords.set([...options.nodeMips.atlasPages, 0], 0);
      feedbackWords.set([...this.scratchAtlasPages, options.finestLevel], 4);
      feedbackWords.set([scratchCapacity, SVO_NODE_MIP_LAYOUT.physicalSize,
        options.generationSource.offsetBytes / 4, options.plannedPageCount], 8);
      feedbackWords.set([options.tree.velocityOffsetBytes / 4, options.tree.materialOwnerOffsetBytes / 4,
        options.tree.sceneGeometryOffsetBytes / 4, options.tree.scenePayloadLanes.materialOwnerWords], 12);
      // Words 20-31 held the dense table's twelve level slabs; the page hash
      // carries its own shape, so they are reserved zeros.
      feedbackWords.set([options.levelCount, 0, 0, options.lightCount ?? 0], 16);
      feedbackFloats.set([...(options.worldOrigin_m ?? [0, 0, 0]), 0], 32);
      feedbackFloats.set([...(options.cellSize_m ?? [1, 1, 1]), 0], 36);
      feedbackWords.set([...options.radiance.atlasPages, options.radiance.slotOffset ?? 0], 40);
      feedbackWords.set([...this.radianceScratchAtlasPages, this.radianceFloorLevel], 44);
      feedbackWords.set(bandedLanes, 48);
      device.queue.writeBuffer(this.feedbackParams, 0, feedbackData);
    }
    const texelCount = scratchTexels[0] * scratchTexels[1] * scratchTexels[2];
    const radianceTexelCount = radianceScratchTexels[0] * radianceScratchTexels[1] * radianceScratchTexels[2];
    this.allocatedBytes = texelCount * svoNodeMipOpacityChannels(this.opacityFormat)
      + radianceTexelCount * 4 * liveSvoRadianceBytesPerTexel(this.radianceFormat)
      + scratchCapacity * 4 + this.params.size + (this.feedbackParams?.size ?? 0);
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived builder");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedBuilderPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived content";
    // The lane set is a property of the tree, so the shader follows it rather
    // than taking a second, separately-authored opinion about what is resident.
    const profile = options.tree.payloadProfile;
    const sceneGeometry = options.tree.sceneGeometryFormat;
    // The B0.5b gate. Off by default, because off is what ships; when on, the
    // only thing that changes anywhere in the world is what `solidDistance`
    // returns outside the banded record set.
    const bandedCellSize = svoBandedReconstructionEnabled() ? options.cellSize_m : undefined;
    // Gated separately from banding: the skip changes the product frame by itself,
    // so it has to stand on its own hash before banding rides on top of it.
    // Root cause rather than omission: gated on its own hash, off by default.
    const solidMarchOffsetFix = svoSolidMarchOffsetFixEnabled();
    // The second mechanism of the same leak — the occlusion march inheriting the
    // gather's receiver-width steps — and gated on its own hash for the same
    // reason: it changes the hero's direct sun by itself.
    const solidDirectOcclusion = svoSolidDirectOcclusionEnabled();
    const buildModule = device.createShaderModule({
      label: `${label} build shader`,
      code: liveSvoDerivedBuildWGSLFor(
        profile, this.radianceFormat, this.opacityFormat, sceneGeometry, bandedCellSize,
        options.tree.leafPayloadMode),
    });
    const feedbackModule = this.radianceFeedbackEnabled
      ? device.createShaderModule({ label: `${label} radiance feedback shader`,
        code: liveSvoRadianceFeedbackWGSLFor(
          profile, this.radianceFormat, sceneGeometry, bandedCellSize, solidMarchOffsetFix,
          solidDirectOcclusion, options.tree.leafPayloadMode) })
      : undefined;
    const copyModule = device.createShaderModule({ label: `${label} copy shader`,
      code: liveSvoDerivedCopyWGSLFor(this.radianceFormat, this.opacityFormat) });
    const emptyInitializationModule = device.createShaderModule({
      label: `${label} valid-empty initialization shader`, code: liveSvoDerivedEmptyInitializationWGSL,
    });
    const [emptyInitializationPipeline, buildPipeline, feedbackPipeline, opacityCopyPipeline, radianceCopyPipelines] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} valid-empty initialization pipeline`, layout: "auto",
        compute: { module: emptyInitializationModule, entryPoint: "initializeValidEmptyPages" } }),
      device.createComputePipelineAsync({ label: `${label} build pipeline`, layout: "auto", compute: { module: buildModule, entryPoint: "buildPages" } }),
      feedbackModule ? device.createComputePipelineAsync({ label: `${label} radiance feedback pipeline`, layout: "auto",
        compute: { module: feedbackModule, entryPoint: "feedbackPages" } }) : Promise.resolve(undefined),
      device.createComputePipelineAsync({ label: `${label} opacity copy pipeline`, layout: "auto", compute: { module: copyModule, entryPoint: "copyOpacity" } }),
      Promise.all(["Low", "High"].map((pair) => device.createComputePipelineAsync({ label: `${label} radiance copy ${pair.toLowerCase()} lobes pipeline`,
        layout: "auto", compute: { module: copyModule, entryPoint: `copyRadiance${pair}` } }))),
    ]);
    const opacityBuildPipeline = await device.createComputePipelineAsync({
      label: `${label} opacity-only pipeline`, layout: "auto",
      compute: { module: device.createShaderModule({ code: liveSvoDerivedBuildWGSLFor(
        profile, this.radianceFormat, this.opacityFormat, sceneGeometry, bandedCellSize,
        options.tree.leafPayloadMode, false) }), entryPoint: "buildPages" },
    });
    if (this.destroyed) throw new Error("Live SVO derived builder was destroyed during pipeline initialization");
    const emptyInitializationBindGroup = device.createBindGroup({
      layout: emptyInitializationPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: options.nodeMips.pageValidity.view },
        { binding: 1, resource: options.radiance.pageValidity.view },
        { binding: 2, resource: { buffer: options.generationSource.buffer } },
        { binding: 3, resource: { buffer: this.params } },
      ],
    });
    const radianceScratchView = this.radianceScratch.createView({ dimension: "3d" });
    const targetRadianceViews = options.radiance.textures.map((texture) => texture.createView({ dimension: "3d" }));
    const materialPbr = options.materialPbr, environmentLighting = options.environmentLighting, lights = options.lights;
    const feedbackParams = this.feedbackParams, feedbackSampler = this.feedbackSampler;
    // One batch, so one bind group a pipeline: nothing is per level any more.
    const batch = { buffer: options.batch.buffer };
    const buildBindGroup = (pipeline: GPUComputePipeline) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } }, { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
      { binding: 2, resource: { buffer: options.tree.payload } }, { binding: 3, resource: batch },
      { binding: 4, resource: { buffer: options.materialEmission } }, { binding: 5, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
      { binding: 6, resource: this.opacityScratch.createView({ dimension: "3d" }) },
      ...targetRadianceViews.map((resource, index) => ({ binding: 7 + index, resource })),
      { binding: 11, resource: radianceScratchView }, { binding: 12, resource: { buffer: this.params } },
      { binding: 13, resource: options.nodeMips.pageValidity.view }, { binding: 14, resource: options.radiance.pageValidity.view },
      { binding: 15, resource: { buffer: this.scratchValidity } },
    ] });
    const feedback = feedbackPipeline && materialPbr && environmentLighting && lights && feedbackParams && feedbackSampler
      ? { pipeline: feedbackPipeline, bindGroup: device.createBindGroup({ layout: feedbackPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
        { binding: 1, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } },
        { binding: 2, resource: { buffer: options.tree.payload } },
        { binding: 3, resource: batch },
        { binding: 4, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
        ...targetRadianceViews.map((resource, index) => ({ binding: 5 + index, resource })),
        { binding: 9, resource: radianceScratchView }, { binding: 10, resource: { buffer: feedbackParams } },
        { binding: 11, resource: options.nodeMips.pageTableTexture.createView() },
        { binding: 12, resource: { buffer: materialPbr } }, { binding: 13, resource: { buffer: environmentLighting } },
        { binding: 14, resource: options.nodeMips.pageValidity.view }, { binding: 15, resource: options.radiance.pageValidity.view },
        { binding: 17, resource: { buffer: lights } },
        { binding: 18, resource: feedbackSampler },
      ] }) } : undefined;
    const opacityCopy = { pipeline: opacityCopyPipeline, bindGroup: device.createBindGroup({ layout: opacityCopyPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: batch }, { binding: 1, resource: this.opacityScratch.createView({ dimension: "3d" }) },
      { binding: 2, resource: options.nodeMips.texture.createView({ dimension: "3d" }) },
      { binding: 5, resource: options.nodeMips.pageValidity.view }, { binding: 7, resource: { buffer: this.params } },
      { binding: 8, resource: { buffer: this.scratchValidity } },
    ] }) };
    const radianceCopyStage = (pair: number): LiveSvoDerivedStage => ({ pipeline: radianceCopyPipelines[pair],
      bindGroup: device.createBindGroup({ layout: radianceCopyPipelines[pair].getBindGroupLayout(0), entries: [
        { binding: 0, resource: batch }, { binding: 3, resource: radianceScratchView },
        { binding: 4, resource: targetRadianceViews[pair * 2] }, { binding: 6, resource: options.radiance.pageValidity.view },
        { binding: 7, resource: { buffer: this.params } }, { binding: 8, resource: { buffer: this.scratchValidity } },
        { binding: 9, resource: targetRadianceViews[pair * 2 + 1] },
      ] }) });
    return {
      emptyInitialization: { pipeline: emptyInitializationPipeline, bindGroup: emptyInitializationBindGroup },
      build: { pipeline: buildPipeline, bindGroup: buildBindGroup(buildPipeline) },
      opacityBuild: { pipeline: opacityBuildPipeline, bindGroup: buildBindGroup(opacityBuildPipeline) },
      feedback, opacityCopy, radianceCopy: [radianceCopyStage(0), radianceCopyStage(1)],
    };
  }

  /**
   * First build only: certify every address-resident page as empty, so a page
   * the build never reaches answers a cone as the air it is.
   */
  encodeCertifyEmpty(pass: GPUComputePassEncoder): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived builder pipelines are not initialized");
    pass.setPipeline(state.emptyInitialization.pipeline); pass.setBindGroup(0, state.emptyInitialization.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.plannedPageCount / 256));
  }

  /**
   * Build the batch the planner has just resolved, and publish it.
   *
   * Opacity is published before the gather so a floor page reads its own new
   * coverage from the atlas; radiance and its validity word are published
   * last. Each dispatch is its own usage scope, so the scratch a dispatch
   * writes is the scratch the next one samples.
   */
  encodeRound(pass: GPUComputePassEncoder, includeRadiance = true): void {
    if (this.destroyed) return;
    if (includeRadiance && this.options.radiance.storageEnabled === false) throw new Error("Radiance dispatch requires allocated radiance storage; rebuild the presentation source first");
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived builder pipelines are not initialized");
    const groups = LIVE_SVO_DERIVED_BUDGET.buildWorkgroups;
    const build = includeRadiance ? state.build : state.opacityBuild;
    pass.setPipeline(build.pipeline); pass.setBindGroup(0, build.bindGroup); pass.dispatchWorkgroups(groups);
    pass.setPipeline(state.opacityCopy.pipeline); pass.setBindGroup(0, state.opacityCopy.bindGroup); pass.dispatchWorkgroups(groups);
    if (!includeRadiance) return;
    if (state.feedback) {
      pass.setPipeline(state.feedback.pipeline); pass.setBindGroup(0, state.feedback.bindGroup); pass.dispatchWorkgroups(groups);
    }
    for (const copy of state.radianceCopy) {
      pass.setPipeline(copy.pipeline); pass.setBindGroup(0, copy.bindGroup); pass.dispatchWorkgroups(groups);
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.opacityScratch.destroy(); this.radianceScratch.destroy(); this.scratchValidity.destroy(); this.params.destroy(); this.feedbackParams?.destroy(); this.destroyed = true;
  }
}

/**
 * What a lighting frame is for. Each is a parameter block written once at
 * build, so choosing one costs a bind group and no upload.
 *
 * - `dirty`: consume the journal and whatever is queued.
 * - `initial`: the queue has just been reset; light every leaf, unbudgeted,
 *   and open the feedback window over the whole scene.
 * - `relight`: `dirty`, and open the window over the whole scene — the lights
 *   or the environment changed, so every floor page's gather is stale.
 * - `dirty-opacity`, `initial-opacity`: the same with radiance switched off,
 *   which starts no feedback rotation.
 */
export type LiveSvoDerivedFrameMode = "dirty" | "initial" | "relight" | "dirty-opacity" | "initial-opacity";

const LIVE_SVO_DERIVED_FRAME_MODES: readonly LiveSvoDerivedFrameMode[] = ["dirty", "initial", "relight", "dirty-opacity", "initial-opacity"];
const LIVE_SVO_DERIVED_UNBUDGETED = 0xffff_ffff;

interface LiveSvoDerivedPlannerPipelineState {
  prepareFramePipeline: GPUComputePipeline;
  prepareFrame: Record<LiveSvoDerivedFrameMode, GPUBindGroup>;
  selectFeedback: LiveSvoDerivedStage;
  populatePipeline: GPUComputePipeline;
  journal: GPUBindGroup;
  allLeaves: GPUBindGroup;
  stepSources: readonly GPUBindGroup[];
  prepareRound: LiveSvoDerivedStage;
  resolve: LiveSvoDerivedStage;
}

/**
 * The lighting queue: which pages are owed a build, and which a round builds.
 *
 * All of it lives on the GPU. Dirty leaves become queued pages (`populate`), a
 * built page queues its parent (`resolve`), and a cursor per ring advances by
 * at most a frame's allowance. The host encodes the same dispatches every
 * lighting frame and learns nothing back; it decides only how many frames to
 * encode, from bounds it already holds.
 */
export class WebGpuLiveSvoDerivedWorklistPlanner {
  /** The batch every round resolves into; the builder binds it once. */
  readonly batch: LiveSvoDerivedGpuBatch;
  readonly allocatedBytes: number;
  /**
   * Rounds a budgeted frame runs. A round moves the sweep one ring, and the
   * chain above a single page is one ring a level, so this many rounds carry a
   * one-voxel edit from its seed to the root in the frame it lands.
   */
  readonly roundsPerFrame: number;
  /** Rounds that drain a first build whatever it holds: every ring, a batch at a time. */
  readonly initialRounds: number;
  private readonly queue: GPUBuffer;
  private readonly slots: GPUBuffer;
  private readonly frameParams: Record<LiveSvoDerivedFrameMode, GPUBuffer>;
  private readonly allLeavesParams: GPUBuffer;
  private readonly stepParams: readonly GPUBuffer[];
  private readonly queueHeaderBytes: number;
  private readonly pageCapacityByLevel: readonly number[];
  private readonly stepLeafCapacity: number;
  private pipelineState?: LiveSvoDerivedPlannerPipelineState;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly options: WebGpuLiveSvoDerivedWorklistPlannerOptions) {
    if (!Number.isInteger(options.levelCount) || options.levelCount < 1 || options.levelCount > SVO_NODE_MIP_MAXIMUM_LEVELS) {
      throw new RangeError(`Live derived level count must be in [1, ${SVO_NODE_MIP_MAXIMUM_LEVELS}]`);
    }
    const requestedCapacity = options.pageCapacityPerLevel;
    const capacities = typeof requestedCapacity === "number"
      ? Array.from({ length: options.levelCount }, () => requestedCapacity)
      : [...requestedCapacity];
    if (capacities.length !== options.levelCount) throw new RangeError("Live derived per-level capacities must name every level");
    if (capacities.some((capacity) => !Number.isInteger(capacity) || capacity < 1)) throw new RangeError("Live derived per-level capacity must be positive");
    this.pageCapacityByLevel = capacities;
    const journal = options.sceneJournal;
    for (const source of [journal, ...options.stepLeafSources]) {
      if (!Number.isInteger(source.recordStrideWords) || source.recordStrideWords < 1) throw new RangeError("Dirty leaf stride must be positive");
      if (!Number.isInteger(source.capacity) || source.capacity < 1) throw new RangeError("Dirty leaf capacity must be positive");
    }
    const offsets = [journal.writeCursorOffsetBytes, journal.recordOffsetBytes, journal.requestedRevisionOffsetBytes,
      journal.completedRevisionOffsetBytes, options.generationSource.offsetBytes,
      ...options.stepLeafSources.flatMap((source) => [source.countOffsetBytes, source.recordOffsetBytes])];
    if (offsets.some((offset) => !Number.isInteger(offset) || offset < 0 || offset % 4 !== 0)) {
      throw new RangeError("Live derived source offsets must be nonnegative u32-aligned byte offsets");
    }
    const levelCount = options.levelCount;
    // One buffer: the state block, a header per ring, then each level's two
    // rings back to back. A record is one slot, so the whole queue is a few
    // megabytes where the per-level worklists it replaces were 48 B a page.
    const sectionOffsetWords: number[] = [];
    let queueWords = QUEUE.stateWords + QUEUE.ringHeaderWords * 2 * levelCount;
    this.queueHeaderBytes = queueWords * 4;
    for (const capacity of capacities) { sectionOffsetWords.push(queueWords); queueWords += capacity * 2; }
    const batchCapacity = Math.min(LIVE_SVO_DERIVED_BUDGET.pagesPerFrame, Math.max(...capacities));
    const batchBytes = (LIVE_SVO_DERIVED_WORKLIST.headerWords + batchCapacity * LIVE_SVO_DERIVED_WORKLIST.recordWords) * 4;
    const label = options.label ?? "Live SVO derived worklists";
    // COPY_SRC for probes only: the fault and built words are never read on the frame path.
    this.queue = device.createBuffer({ label: `${label} queue`, size: queueWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.slots = device.createBuffer({ label: `${label} slot states`, size: options.nodeMips.pageCapacity * QUEUE.slotWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.batch = { capacity: batchCapacity,
      buffer: device.createBuffer({ label: `${label} batch`, size: batchBytes, usage: GPUBufferUsage.STORAGE }) };
    this.roundsPerFrame = levelCount + 2;
    this.initialRounds = capacities.reduce((rounds, capacity) => rounds + Math.ceil(capacity / batchCapacity) + 1, 0);
    this.stepLeafCapacity = options.stepLeafSources.reduce((total, source) => total + source.capacity, 0);
    // Every parameter block is written here, once. Nothing about a frame —
    // not its mode, not a feedback phase — is uploaded while the world runs.
    const settle = options.radianceFeedback
      ? [LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount, LIVE_SVO_RADIANCE_FEEDBACK.settleCycleCount] : [0, 0];
    // A rotation over every floor page and everything above it, at capacity:
    // what a rotation is assumed to take until one has been timed. Half the
    // window at most, so a relight of the largest world still starts.
    const floorLevel = Math.min(options.radianceFloorLevel ?? 0, levelCount - 1);
    const rotationFrames = Math.min(LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount >> 1,
      this.framesToBuild(capacities.slice(floorLevel).reduce((pages, capacity) => pages + capacity, 0)));
    const createParams = (suffix: string, source: readonly number[], capacity: number, mode: number, allowance: number, flags: number) => {
      const params = device.createBuffer({ label: `${label} ${suffix} parameters`, size: LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES / 4);
      words.set([source[0], options.generationSource.offsetBytes / 4, source[1], source[2]], 0);
      words.set([options.finestLevel, levelCount, options.radianceFloorLevel ?? 0, mode], 4);
      words.set([capacity, options.nodeMips.pageCapacity, settle[0], settle[1]], 8);
      words.set([allowance, batchCapacity, flags, LIVE_SVO_DERIVED_BUDGET.feedbackReachPages], 12);
      words.set([journal.requestedRevisionOffsetBytes / 4, journal.completedRevisionOffsetBytes / 4, options.radiance.slotOffset ?? 0, rotationFrames], 16);
      words.set(sectionOffsetWords, 20);
      words.set(capacities, 20 + 4 * LIVE_SVO_DERIVED_PLANNER_LEVEL_VECTORS);
      device.queue.writeBuffer(params, 0, words);
      return params;
    };
    const journalSource = [journal.writeCursorOffsetBytes / 4, journal.recordOffsetBytes / 4, journal.recordStrideWords];
    const frame = (mode: LiveSvoDerivedFrameMode, allowance: number, flags: number) =>
      createParams(`${mode} frame`, journalSource, journal.capacity, QUEUE.journalMode, allowance, flags);
    const budget = LIVE_SVO_DERIVED_BUDGET.pagesPerFrame;
    this.frameParams = {
      dirty: frame("dirty", budget, 0),
      initial: frame("initial", LIVE_SVO_DERIVED_UNBUDGETED, QUEUE.globalWindowFlag | QUEUE.skipJournalFlag | QUEUE.openAncestorsFlag),
      relight: frame("relight", budget, QUEUE.globalWindowFlag),
      "dirty-opacity": frame("dirty-opacity", budget, QUEUE.noFeedbackFlag),
      "initial-opacity": frame("initial-opacity", LIVE_SVO_DERIVED_UNBUDGETED, QUEUE.skipJournalFlag | QUEUE.noFeedbackFlag | QUEUE.openAncestorsFlag),
    };
    this.allLeavesParams = createParams("all-leaves", [0, 0, 1], options.tree.leafCapacity, QUEUE.allLeavesMode, budget, 0);
    this.stepParams = options.stepLeafSources.map((source, index) => createParams(`step source ${index}`,
      [source.countOffsetBytes / 4, source.recordOffsetBytes / 4, source.recordStrideWords], source.capacity, QUEUE.listMode, budget, 0));
    this.allocatedBytes = queueWords * 4 + this.slots.size + batchBytes
      + LIVE_SVO_DERIVED_PLANNER_PARAMS_BYTES * (LIVE_SVO_DERIVED_FRAME_MODES.length + 1 + this.stepParams.length);
  }

  /**
   * Frames that are certain to build `pages` queued pages and everything they
   * queue in turn. A frame either spends its allowance or runs out of rounds,
   * and a round that does not spend the allowance finishes a ring; the sweep
   * passes each ring at most twice before nothing is left — once from where it
   * stood, once from the bottom.
   */
  private framesToBuild(pages: number): number {
    return Math.ceil(pages / LIVE_SVO_DERIVED_BUDGET.pagesPerFrame)
      + Math.ceil(4 * this.pageCapacityByLevel.length / this.roundsPerFrame) + 1;
  }

  /**
   * Lighting frames that are certain to drain the queue once `seedLeaves` more
   * leaves have been journaled, with nothing read back. A leaf seeds one page;
   * a level holds no more pages than its capacity and no more than one a seed;
   * and a page is built at most twice before the queue is empty — where the
   * sweep stood when its seed arrived, and again after the level beneath it.
   */
  drainFrames(seedLeaves: number): number {
    let pages = seedLeaves;
    for (const capacity of this.pageCapacityByLevel) pages += 2 * Math.min(seedLeaves, capacity);
    return this.framesToBuild(pages);
  }

  /** {@link drainFrames} for one solver step's lists at their fullest, on top of `seedLeaves` already journaled. */
  stepDrainFrames(seedLeaves: number): number { return this.drainFrames(seedLeaves + this.stepLeafCapacity); }

  async initializePipelines(): Promise<void> {
    if (this.pipelineState) return;
    if (this.destroyed) throw new Error("Cannot initialize destroyed live SVO derived worklist planner");
    if (!this.pipelineInitialization) {
      this.pipelineInitialization = this.compilePipelineState().then((state) => {
        if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
        this.pipelineState = state;
      });
    }
    try {
      await this.pipelineInitialization;
    } catch (error) {
      this.pipelineInitialization = undefined;
      throw error;
    }
  }

  private async compilePipelineState(): Promise<LiveSvoDerivedPlannerPipelineState> {
    const { device, options } = this;
    const label = options.label ?? "Live SVO derived worklists";
    const plannerModule = device.createShaderModule({ label: `${label} shader`, code: liveSvoDerivedWorklistWGSL });
    const entry = (entryPoint: string) => device.createComputePipelineAsync({ label: `${label} ${entryPoint} pipeline`, layout: "auto",
      compute: { module: plannerModule, entryPoint } });
    const [prepareFramePipeline, selectFeedbackPipeline, populatePipeline, prepareRoundPipeline, resolvePipeline] = await Promise.all([
      entry("prepareFrame"), entry("selectFeedback"), entry("populate"), entry("prepareRound"), entry("resolve"),
    ]);
    if (this.destroyed) throw new Error("Live SVO derived worklist planner was destroyed during pipeline initialization");
    // A pipeline's automatic layout holds only the bindings its entry point
    // reaches, so each group below lists exactly those and no others.
    const control = { binding: 1, resource: { buffer: options.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } };
    const topology = { binding: 2, resource: { buffer: options.tree.topology, offset: options.tree.topologyOffsetBytes } };
    const pageTable = { binding: 3, resource: options.nodeMips.pageTableTexture.createView() };
    const slots = { binding: 4, resource: { buffer: this.slots } };
    const queue = { binding: 5, resource: { buffer: this.queue } };
    const params = (buffer: GPUBuffer) => ({ binding: 6, resource: { buffer } });
    const batch = { binding: 8, resource: { buffer: this.batch.buffer } };
    const journalBuffer = { binding: 0, resource: { buffer: options.sceneJournal.buffer } };
    const prepareFrameGroup = (mode: LiveSvoDerivedFrameMode) => device.createBindGroup({
      layout: prepareFramePipeline.getBindGroupLayout(0), entries: [
        journalBuffer, queue, params(this.frameParams[mode]), { binding: 7, resource: { buffer: options.generationSource.buffer } },
      ] });
    const populateGroup = (source: GPUBuffer, sourceParams: GPUBuffer) => device.createBindGroup({
      layout: populatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: source } }, control, topology, pageTable, slots, queue, params(sourceParams),
        { binding: 9, resource: options.nodeMips.pageValidity.view }, { binding: 10, resource: options.radiance.pageValidity.view },
      ] });
    return {
      prepareFramePipeline,
      prepareFrame: {
        dirty: prepareFrameGroup("dirty"), initial: prepareFrameGroup("initial"), relight: prepareFrameGroup("relight"),
        "dirty-opacity": prepareFrameGroup("dirty-opacity"), "initial-opacity": prepareFrameGroup("initial-opacity"),
      },
      selectFeedback: { pipeline: selectFeedbackPipeline, bindGroup: device.createBindGroup({
        layout: selectFeedbackPipeline.getBindGroupLayout(0), entries: [pageTable, slots, queue, params(this.frameParams.dirty)] }) },
      populatePipeline,
      journal: populateGroup(options.sceneJournal.buffer, this.frameParams.dirty),
      allLeaves: populateGroup(options.tree.control, this.allLeavesParams),
      stepSources: options.stepLeafSources.map((source, index) => populateGroup(source.buffer, this.stepParams[index])),
      prepareRound: { pipeline: prepareRoundPipeline, bindGroup: device.createBindGroup({
        layout: prepareRoundPipeline.getBindGroupLayout(0), entries: [queue, params(this.frameParams.dirty), batch] }) },
      resolve: { pipeline: resolvePipeline, bindGroup: device.createBindGroup({
        layout: resolvePipeline.getBindGroupLayout(0), entries: [control, topology, pageTable, slots, queue, params(this.frameParams.dirty), batch] }) },
    };
  }

  /**
   * Forget everything queued. The frame that follows must be an initial one,
   * which lights every leaf and so owes nothing to what was waiting.
   */
  encodeReset(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    encoder.clearBuffer(this.queue, 0, this.queueHeaderBytes);
    encoder.clearBuffer(this.slots);
  }

  /**
   * Open a lighting frame: latch its state, start a feedback rotation if one is
   * due, and queue the pages whose voxels changed. `stepSources` adds the
   * lists a solver step has just rewritten.
   */
  encodeFrame(pass: GPUComputePassEncoder, mode: LiveSvoDerivedFrameMode, stepSources: boolean): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived worklist planner pipelines are not initialized");
    const groups = LIVE_SVO_DERIVED_BUDGET.plannerWorkgroups;
    pass.setPipeline(state.prepareFramePipeline); pass.setBindGroup(0, state.prepareFrame[mode]); pass.dispatchWorkgroups(1);
    pass.setPipeline(state.selectFeedback.pipeline); pass.setBindGroup(0, state.selectFeedback.bindGroup); pass.dispatchWorkgroups(groups);
    pass.setPipeline(state.populatePipeline);
    if (mode === "initial" || mode === "initial-opacity") {
      pass.setBindGroup(0, state.allLeaves); pass.dispatchWorkgroups(groups);
      return;
    }
    pass.setBindGroup(0, state.journal); pass.dispatchWorkgroups(groups);
    if (!stepSources) return;
    for (const source of state.stepSources) { pass.setBindGroup(0, source); pass.dispatchWorkgroups(groups); }
  }

  /** Take the next batch from the queue and resolve it into build records. */
  encodeRound(pass: GPUComputePassEncoder): void {
    if (this.destroyed) return;
    const state = this.pipelineState;
    if (!state) throw new Error("Live SVO derived worklist planner pipelines are not initialized");
    pass.setPipeline(state.prepareRound.pipeline); pass.setBindGroup(0, state.prepareRound.bindGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(state.resolve.pipeline); pass.setBindGroup(0, state.resolve.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.batch.capacity / 64));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.queue.destroy(); this.slots.destroy(); this.batch.buffer.destroy();
    for (const mode of LIVE_SVO_DERIVED_FRAME_MODES) this.frameParams[mode].destroy();
    this.allLeavesParams.destroy(); this.stepParams.forEach((params) => params.destroy()); this.destroyed = true;
  }
}
