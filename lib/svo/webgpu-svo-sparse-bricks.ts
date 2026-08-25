import type { RigidBodyDescription, SceneDescription } from "../core/model";
import type { RenderFrameSeam } from "../core/render-frame-stages";
import { svoSceneLighting } from "./svo-dry-scene-lighting";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
import {
  buildSvoEnvironmentLighting,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  type SvoEnvironmentLightingRecord,
} from "./svo-environment-lighting";
import {
  buildSvoSceneLights,
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  type SvoLightRecord,
} from "./svo-light-abi";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  SVO_MATERIAL_RECORD_STRIDE_BYTES,
  svoMaterialFunctionIdForEnvironmentProxy,
  svoMaterialFromEnvironmentProxyMaterial,
} from "./svo-material-abi";
import { sceneTerrainSurfaceModel, type SvoTerrainSurfaceModel } from "./svo-terrain-material";
import { sceneCellSizes_m } from "../core/scene-lattice";
import {
  SOLID_WORLD_BRICK_CELLS,
  solidWorldContentStamp,
  solidWorldForScene,
  type SolidWorld,
} from "../core/solid-world";
import {
  SparseBrickOctreeGPU,
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_SCENE_GEOMETRY_FORMATS,
  SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES,
  octreeLiveSceneLeafPayloadMode,
  octreeLiveSceneSceneGeometryFormat,
  packSparseBrickPlan,
  type SparseBrickCoordinate,
  type SparseBrickPlan,
  type SparseBrickPayloadProfileName,
  type SparseBrickPublicationSource,
  type SparseBrickSceneGeometryFormat,
  type SparseBrickLeafPayloadMode,
  type SparseBrickSize
} from "./sparse-brick-octree";
import { planAdaptiveSparseBrickOctreeSteps } from "../core/adaptive-sparse-brick-plan";
import {
  createSvoEnvironmentRefinement,
  svoEnvironmentRefinementMode,
} from "./svo-environment-refinement";
import {
  createSvoEnvironmentCoarsening,
  svoEnvironmentCoarseningPower,
} from "./svo-environment-coarsening";
import type { SvoPrimitiveDescriptor } from "./svo-primitive-abi";
import { planSvoWideFanoutSteps } from "./svo-wide-fanout";
import { WebGPUSvoWideFanout } from "./webgpu-svo-wide-fanout";
import { packSvoCompactHierarchy } from "./svo-compact-hierarchy";
import { WebGpuSvoCompactHierarchy } from "./webgpu-svo-compact-hierarchy";
import {
  SVO_NODE_MIP_LAYOUT,
  svoNodeMipOpacityFormat,
  svoOpacityLevelFloor,
  type SvoNodeMipCoordinate,
} from "./svo-node-mip-pyramid";
import {
  growSvoNodeMipAddressPlan,
  pagesOutsideSvoNodeMipAddressPlan,
  planSvoNodeMipAddressesSteps,
  svoNodeMipDomainDirectPageTableDimensions,
  type SvoNodeMipAddressPlan,
} from "./svo-node-mip-address-plan";
import {
  WEBGPU_SVO_NODE_MIP_LAYOUT,
  WebGpuLiveSvoNodeMipPyramid,
  createWebGpuSvoNodeMipDirectPageTable,
  webGpuSvoNodeMipMaximumPages,
} from "./webgpu-svo-node-mip-pyramid";
import { WebGpuLiveSvoTetrahedralRadiance } from "./webgpu-svo-tetrahedral-radiance";
import {
  LIVE_SVO_RADIANCE_FEEDBACK,
  WebGpuLiveSvoDerivedBuilder,
  WebGpuLiveSvoDerivedWorklistPlanner,
  liveSvoPlanBasePagesSteps,
} from "./webgpu-svo-live-derived-builder";
import { WebGpuSvoBrickOccupancyBuilder } from "./webgpu-svo-brick-occupancy";
import {
  WebGpuSparseBrickTopologyMutator,
  packSparseBrickTopologyMutationWorklist,
  sparseBrickTopologyMutationNodeReserve,
} from "../core/webgpu-sparse-brick-topology-mutation";
import { assertSvoBrickRasterNodeAddressable } from "./webgpu-svo-brick-raster";
import { planSparseSceneDomain } from "../core/sparse-scene-domain";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../core/webgpu-rigid-body";
import { VOXEL_MATERIAL_IDS, materialIdForRigidShape, packVoxelDebugMaterialTable } from "../core/voxel-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives, type EnvironmentProxyPrimitive } from "../core/voxel-environments";
import {
  SparseSceneProxyVoxelizer,
  SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW,
  SPARSE_SCENE_MAINTENANCE_STATE_WORDS,
  sparseScenePrimitiveBounds,
  sparseScenePrimitiveForProxy,
  SPARSE_SCENE_FIELD_PROGRAM_CAPACITY,
  type SparseSceneAxisAlignedBounds,
  type SparseSceneMaintenanceBinding,
  type SparseScenePrimitive,
  type SparseScenePrimitiveUpdate,
  type SparseScenePublication,
} from "../core/webgpu-sparse-scene-proxies";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelSceneRenderSource,
  type SparseVoxelStructuralRenderSource,
} from "../core/webgpu-voxel-debug";
import { GPUFluidBrickResidency, type FluidBrickResidencyStats } from "../core/webgpu-fluid-brick-residency";
import {
  completeCooperativeBuild,
  driveCooperativeBuild,
  type CooperativeBuildOptions,
} from "../core/cooperative-build";
import { PassBroker } from "../core/webgpu-pass-broker";

export interface OctreeSparseBrickWorldOptions {
  brickSize?: SparseBrickSize;
  /** Additional authored-environment subdivision beyond the legacy 2x brick ceiling. */
  environmentBrickRefinementLevels?: number;
  /**
   * Extra octree levels the environment may descend into where a leaf holds
   * more primitives than the hierarchy binds. Zero is the historical plan.
   *
   * Honoured only when `systems.fluid` is false, because a solver brick pins
   * its node at the solver's own level and the domain planner claims the whole
   * container as solver bricks. Two levels take the hero garden's busiest brick
   * from 122 candidates to 64 for 8 % more leaves; the uniform alternative —
   * halving `finestCellSize_m` twice — costs nine times as many.
   */
  environmentRefinementDepth?: number;
  /**
   * The scene's authored solids, exactly, for brick selection only.
   *
   * A primitive claims its whole AABB, and an AABB over-claims: the corners a
   * curved or thin solid never enters are 512 voxels of air and a node-mip page
   * of zeros each. Given the exact distance functions the claim can be narrowed
   * to the bricks geometry can actually reach
   * (`liveSceneReachableBrickCoordinates`). Absent, the claim is the AABB, which
   * is what shipped.
   *
   * Honoured only when `systems.fluid` is false. A solver writes cells the scene
   * geometry says nothing about, so its claim is not a geometric one.
   */
  sceneSolids?: readonly SparseSceneSolidReach[];
  /**
   * The SVO record one authored proxy publishes, for the curvature-driven
   * refinement rule (`FLUID_SVO_REFINEMENT_MODE=surface`).
   *
   * Supplied by the caller for the same reason `sceneSolids` is:
   * `svo-scene-primitives` depends on this module, so importing
   * `svoDescriptorForEnvironmentProxy` here would close a cycle. Without it the
   * surface rule has no field to evaluate and the planner keeps the shipped
   * candidate-density predicate, whatever the lever says.
   */
  environmentDescriptorFor?: (primitive: EnvironmentProxyPrimitive) => SvoPrimitiveDescriptor;
  /**
   * Whether that rule may stop at a node its surface crosses *flatly*, instead
   * of splitting every node the surface reaches.
   *
   * Off is the default and the product's. The exemption is a second-order test,
   * so what it leaves coarse is precisely the low-curvature part of a surface —
   * a mound's cap, a shallow basin — and the primary shades a coarse leaf as
   * axis-aligned voxel faces, which is a visible grid of flat facets against
   * the refined slope around it. See `lib/svo-environment-refinement.ts`.
   */
  environmentPlanarRefinementExemption?: boolean;
  /** Air-side support retained for pressure-topology rebuilds. */
  haloCells?: number;
  /** Keep the independent deep-liquid topology worklist. */
  bulkResidency?: boolean;
  /** Temporally amortized diffuse-radiance feedback; on by default. */
  radianceFeedback?: boolean;
  /** Velocity-swept residency support plus downstream neighbor activation. */
  brickPreActivation?: boolean;
  /**
   * Power-of-two bricks per topology-tile axis. Topology rebuilds operate on
   * tiles of max(brickSize, maximumLeafSize) cells so a pressure leaf can
   * never straddle a partial-rebuild boundary; payload residency remains
   * brick-granular.
   */
  topologyTileBricks?: number;
  /** Retain dry pressure-wall tiles and their grading support for owner pages. */
  includePressureBoundarySupport?: boolean;
  pressureBoundaryTopClosed?: boolean;
  includeWholeDomainPressureSupport?: boolean;
  /** Retain a dry wall tile only when liquid is within reach; see the residency option. */
  fluidGatedBoundarySupport?: boolean;
  /** Lifetime budget of previously absent scene bricks that may be activated in-place. */
  sceneMutationBrickCapacity?: number;
  /**
   * Build the compact hierarchy and the wide-fanout snapshot.
   *
   * Off by default because nothing production runs binds either. Binding 5 —
   * the only binding they reach — is populated only when the dry-scene
   * renderer's `traversalMode` is `compact`, `wide` or `hybrid`, and the
   * shipping primary pins the secondary to `canonical-parametric`, so on every
   * production frame and every raster benchmark arm both structures cost a CPU
   * plan, a GPU encode and resident memory for nothing. The wide-fanout
   * micro-mip buffer alone is `maximumPages * 292 B` and is bound by no shader
   * at all.
   *
   * The lane that measures one of those three traversals sets this; a lane that
   * selects `compact` without it gets no bind group rather than a different
   * shader (`webgpu-svo-dry-scene.ts`, the `compact.status !== "ready"` and
   * `!derivedTraversal` guards).
   */
  derivedTraversalStructures?: boolean;
  /**
   * Called as each planning stage of the constructor begins.
   *
   * The constructor is one uninterrupted synchronous block, and on a refined
   * world it is a long one — 40 s at environment refinement depth 3 on
   * `hero-garden-hose`, against 1.4 s at depth 0. Without this the only thing
   * a caller can say for the whole of it is "allocating", which is what makes
   * a slow build indistinguishable from a wedged one.
   *
   * `completed` counts stages *started*, so the label always names work that is
   * actually happening rather than work that has just finished.
   */
  progress?: OctreeSparseBrickWorldProgress;
  /**
   * Suspend the build instead of running it, for `OctreeSparseBrickWorld.create`.
   *
   * Not a caller-facing lever: a world constructed this way has none of its
   * fields assigned yet and every method on it would fault. `create` is the
   * only thing that should set it, and it takes the suspended build back out
   * in the next statement.
   */
  deferBuild?: boolean;
}

/**
 * The constructor's stage marks, in the order they occur.
 *
 * Kept as a list rather than free-form strings so the count is derived from the
 * stages rather than maintained beside them, and so a caller can size its own
 * progress bar before the build starts.
 *
 * The stages are the ones that were *measured* to cost something at refinement
 * depth 3, and their order here is their order in the constructor. `--cpu-prof`
 * on `hero-garden-hose`, as a share of the 40 s build: brick claim 25 %,
 * octree plan 39 %, node-mip addressing and pyramid 20 %, ground 4 %, and the
 * remainder split across packing and arena creation. Anything cheaper than a
 * percent is folded into the stage it precedes rather than given its own mark,
 * because a progress bar that jumps through five stages in 3 ms is noise.
 */
export const OCTREE_SPARSE_BRICK_WORLD_STAGES = [
  "Select the bricks the scene reaches",
  "Plan the adaptive octree",
  "Pack the octree and allocate its arenas",
  "Plan the node-mip pyramid",
] as const;

export type OctreeSparseBrickWorldStage = typeof OCTREE_SPARSE_BRICK_WORLD_STAGES[number];

/**
 * `stage` is deliberately a plain string rather than the union above: a caller
 * that wraps this build in a larger one — the live SVO scene does, with a solid
 * reach before it and a shader compile after — reports its own stages through
 * the same callback, and narrowing the type would force it to invent a second
 * one that differs only in the strings it admits.
 */
export type OctreeSparseBrickWorldProgress = (stage: {
  stage: string;
  /** Stages started, 0-based; `total` is `OCTREE_SPARSE_BRICK_WORLD_STAGES.length`. */
  completed: number;
  total: number;
}) => void;

export interface OctreeSparseBrickDenseFields {
  levelSet: GPUTexture;
  velocity: GPUTexture;
  solidCells: GPUBuffer;
}

/** Environment terminal leaves are at most 2x the solver brick scale. */
export const ENVIRONMENT_MAXIMUM_COARSENING_POWER = 1;

/** One extra level is the production default; zero reproduces the previous plan. */
export const DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS = 1;

export function environmentMaximumCoarseningPower(
  refinementLevels = DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS,
): number {
  if (!Number.isInteger(refinementLevels) || refinementLevels < 0
    || refinementLevels > ENVIRONMENT_MAXIMUM_COARSENING_POWER) {
    throw new RangeError(`Environment brick refinement must be an integer from 0 to ${ENVIRONMENT_MAXIMUM_COARSENING_POWER}`);
  }
  return ENVIRONMENT_MAXIMUM_COARSENING_POWER - refinementLevels;
}

/** Derive an optional renderer traversal snapshot without taking structural authority. */
export function planOctreeSvoWideFanout(plan: SparseBrickPlan) {
  return completeCooperativeBuild(planOctreeSvoWideFanoutSteps(plan));
}

/** The same plan, offered as slices; see `planSvoWideFanoutSteps` for why. */
export function* planOctreeSvoWideFanoutSteps(plan: SparseBrickPlan) {
  return yield* planSvoWideFanoutSteps({
    sourceGeneration: 1,
    generation: 1,
    maximumDepth: plan.maximumDepth,
    terminals: plan.leaves.map((leaf) => {
      const node = plan.nodes[leaf.nodeIndex];
      return {
        sourceNodeIndex: node.index,
        sourceLeafIndex: leaf.index,
        level: node.level,
        coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z] as const,
      };
    }),
  });
}

export function planOctreeBrickCoordinates(dimensions: readonly [number, number, number], brickSize: SparseBrickSize) {
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Octree brick size must be 4 or 8");
  for (const value of dimensions) if (!Number.isInteger(value) || value < 1) throw new RangeError("Octree field dimensions must be positive integers");
  const brickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const coordinates: SparseBrickCoordinate[] = [];
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let y = 0; y < brickDimensions[1]; y += 1) for (let x = 0; x < brickDimensions[0]; x += 1) coordinates.push({ x, y, z });
  return { brickDimensions, coordinates };
}

/** Root depth must cover declared empty bounds as well as allocated terminals. */
export function sparseSceneOctreeMaximumDepth(
  brickDimensions: readonly [number, number, number],
  coordinates: readonly SparseBrickCoordinate[],
): number {
  for (const [axis, value] of brickDimensions.entries()) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Sparse scene brick dimension ${axis} must be positive`);
  }
  let maximumBrickCoordinate = Math.max(...brickDimensions.map((value) => value - 1));
  for (const coordinate of coordinates) {
    for (const [axis, value] of [coordinate.x, coordinate.y, coordinate.z].entries()) {
      if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Sparse scene brick coordinate ${axis} must be non-negative`);
      maximumBrickCoordinate = Math.max(maximumBrickCoordinate, value);
    }
  }
  return maximumBrickCoordinate === 0 ? 0 : Math.ceil(Math.log2(maximumBrickCoordinate + 1));
}

export const ENVIRONMENT_VOXEL_MATERIAL_BASE = 32;

export const OCTREE_SVO_PBR_MATERIAL_REVISION = 2;
export const OCTREE_SVO_LIGHT_REVISION = 1;
export const OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION = 1;

/**
 * Fixed live-scene arenas. Capacity changes require an explicit new world, never
 * a hidden reallocating bake.
 *
 * 16,384 to match `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` and
 * `SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES`, so the record count that draws is the
 * record count that voxelizes. At 4,096 the ten-times acceptance scene
 * (`hero-garden-hose-x10`, 5,039 records) did not fail an overflow flag and
 * degrade — it threw at world construction, out of `packLiveSvoMaterialEmission`,
 * because the *material* table is sized from this constant and overran first.
 * The handoff plan's capacity table calls that row an "overflow flag"; it never
 * was one, and the cheapest way to make it degrade is to put the ceiling past
 * where any authored scene reaches.
 *
 * Priced: the three arenas this constant sizes are the PBR material table at
 * 96 B/record (16,416 x 96 = 1.58 MB, up from 396 KB), the emission table at
 * 16 B/record (263 KB, up from 66 KB), and the voxelizer's own 48 B record
 * arena plus its 32 B conservative bound (16,384 x 80 = 1.31 MB, up from
 * 328 KB). The dirty-region arenas below scale with it too and are the larger
 * item; see their note.
 */
export const OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY = 16_384;
/**
 * Two regions per moved record — the bound it left and the bound it arrived at.
 *
 * At 32 B a bound plus the 32 B brick-cell box the indexed invalidation
 * dispatches over, this is 2.1 MB of the maintenance arena. It is worth it:
 * coalescing past this capacity collapses every edit into one whole-scene box,
 * which dirties the entire tree.
 */
export const OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY = OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY * 2;
/**
 * Primitives one brick's plan will tolerate before the planner splits it.
 *
 * This is a *shape* decision — a leaf holding more geometry than this gets
 * refined — and it is deliberately not the arena capacity below. Raising the
 * arena to survive a dense scene must not make the planner build a coarser tree
 * for a sparse one, which is exactly what sharing one constant did.
 */
export const OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET = 64;
/**
 * Candidate slots the voxelizer binds per dirty brick.
 *
 * 512, not the planner's 64, because refinement runs out of depth before it runs
 * out of density: on the ten-times acceptance scene the busiest brick binds 442
 * primitives at the finest level the tree has, and there is no split left to
 * make. The surplus past this capacity is *not* image-preserving despite what
 * the handoff plan's capacity table says — `binDirtyBrickCandidates` writes
 * nothing for the losers of the atomic race, so those primitives are absent from
 * the opacity pyramid and the radiance atlas while still drawing in primary
 * visibility. Geometry that keeps its silhouette and stops casting a shadow is
 * a worse artefact than the cost of the slots.
 *
 * Priced: the candidate arena is `leafCapacity x this` words. The hero garden's
 * 587 leaves cost 1.2 MB (up from 150 KB); a ten-times world at 8,000 leaves
 * costs 16 MB. The rebuild pass pays for it too — a brick evaluates every bound
 * candidate at every voxel — but only in the bricks that are genuinely that
 * dense, and that cost is the geometry being there rather than a budget hiding
 * it. `maximumBrickCandidates` on the maintenance state reports the real
 * density, so approaching this ceiling is visible before it is reached.
 */
export const OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 512;
/**
 * Grid cells the record index may address per axis, and in total.
 *
 * One cell per finest brick is the ideal: a dirty brick then reads a list of
 * records that genuinely reach into it. The hero garden's whole domain is
 * 12x4x8 bricks, so that is 384 cells and free; the coarsening below exists only
 * so a very large or very anisotropic domain cannot ask for an unbounded one.
 * Packed coordinates are ten bits an axis, which is the hard per-axis ceiling.
 */
export const OCTREE_LIVE_SCENE_INDEX_MAXIMUM_CELLS = 262_144;
export const OCTREE_LIVE_SCENE_INDEX_MAXIMUM_AXIS = 1_023;

/**
 * Place the record index's lattice over a brick domain.
 *
 * A cell is a whole power-of-two number of finest bricks so the lattice shares
 * the octree's alignment, which is what lets a leaf's cell range be read off its
 * bounds a quarter of a brick inside its faces instead of at an exact face.
 */
export function planOctreeLiveSceneRecordIndex(
  brickDimensions: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
): { dimensions: [number, number, number]; cellExtent_m: [number, number, number]; bricksPerCell: number } {
  let bricksPerCell = 1;
  const cells = (factor: number) => brickDimensions
    .map((bricks) => Math.max(1, Math.ceil(bricks / factor))) as [number, number, number];
  while (
    cells(bricksPerCell).reduce((product, value) => product * value, 1) > OCTREE_LIVE_SCENE_INDEX_MAXIMUM_CELLS
    || cells(bricksPerCell).some((value) => value > OCTREE_LIVE_SCENE_INDEX_MAXIMUM_AXIS)
  ) {
    bricksPerCell *= 2;
  }
  return {
    dimensions: cells(bricksPerCell),
    cellExtent_m: cellSize.map((size) => size * brickSize * bricksPerCell) as [number, number, number],
    bricksPerCell,
  };
}
/**
 * Dirty bricks one encoded frame repairs, before the environment overrides it.
 *
 * A brick is 512 voxels against up to 64 candidates, so this is the knob that
 * bounds a publication's worst frame. Publications converge over frames like the
 * light cache; the first publication of a scene is deliberately exempt, because
 * until it completes there is no scene generation at all to show.
 */
export const OCTREE_LIVE_SCENE_VOXELIZATION_BRICK_BUDGET = 2_048;

/** Per-frame voxelization budget, overridable so a lane can force convergence to take frames. */
export function octreeLiveSceneBrickBudget(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): number {
  const raw = environment?.FLUID_SVO_VOXELIZATION_BRICK_BUDGET;
  if (raw === undefined || raw === "") return OCTREE_LIVE_SCENE_VOXELIZATION_BRICK_BUDGET;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("FLUID_SVO_VOXELIZATION_BRICK_BUDGET must be a non-negative integer");
  }
  return value;
}
/**
 * Which per-voxel lanes a solverless world allocates. `dry` by default.
 *
 * A world with `systems.fluid === false` never writes the dynamic
 * geometry/velocity/material lanes — `encodeFromDenseFields` is unreachable and
 * `encodePublish` only ever zero-fills them — so 36 of its 56 bytes per voxel
 * are dead, and its scene geometry lane carries two channels no writer touches.
 * `dry` drops both, for 56 -> 12 bytes a voxel.
 *
 * It shipped as a lever because the saving is invisible in a CPU test and the
 * regression would not be: every consumer of an absent lane is compiled out
 * rather than clamped, and that set is only provable on a device. It is the
 * default now because the device answered — the hero garden renders frame hash
 * 0xa949ae24 under both profiles, byte-identical, with the arena down from
 * 155 MB to 116 MB and the frame from 32.6 ms to 14.1 ms. The env var stays as
 * the A/B, and `FLUID_SVO_DRY_PAYLOAD_PROFILE=full` is the one-word rollback.
 *
 * Authoring water flips `systems.fluid` and restores the full lane set
 * regardless of this setting.
 */
export function octreeLiveSceneDryPayloadProfile(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): SparseBrickPayloadProfileName {
  const raw = environment?.FLUID_SVO_DRY_PAYLOAD_PROFILE;
  if (raw === undefined || raw === "") return "dry";
  if (raw !== "full" && raw !== "dry") throw new RangeError("FLUID_SVO_DRY_PAYLOAD_PROFILE must be \"full\" or \"dry\"");
  return raw;
}

/**
 * How wide a dry world's two scene-geometry channels are, per voxel.
 *
 * Pruning dead lanes took the dry world to 12 bytes a voxel and left the two
 * live channels at f32. That is the whole arena once refinement is on: at
 * environment depth 3 the hero garden holds 116.3 M voxels, so the geometry lane
 * alone is 930 MB at 8 bytes and the leaf, node and state records together do
 * not reach 1 % of it. Depth 3 fitting in memory is this one string.
 *
 * `f16-unorm8` is the default: 4 bytes a voxel, one word, word-aligned, with the
 * distance still in metres. `=f32x2` is the one-word rollback to what shipped.
 *
 * A third arm, `snorm8-unorm8`, has been **removed**. It was the intended landing
 * point — 2 bytes a voxel — and the device rejected it. Measured
 * on the real hero-garden payload at depth 0, over the 225 247 voxels the
 * renderer actually shades, the angular error `safeNormal` picks up against the
 * f32 arm is:
 *
 *   arm           mean     p50     p90     p99      max   fallbackFlips  saturated
 *   f16          0.008   0.007   0.016   0.030    0.507         5           0
 *   snorm8@1R    6.952   4.033  17.245  45.969  128.959      1841   5 083 681
 *   snorm8@2R    0.386   0.260   0.612   2.557   30.869      3299   4 861 669
 *   snorm8@3R    0.450   0.382   0.851   1.428   18.435      5131   4 658 301
 *   snorm8@4R    0.576   0.502   1.128   1.726   26.970      6988   4 483 103
 *
 * The `saturated` column is the finding, and no band fixes it. Most voxels of a
 * solid scene are deep interior, so 4.5-5.1 M of them sit outside any band
 * narrow enough to quantise usefully; they all clamp to +/-1, their central
 * differences go to zero, and `safeNormal` falls back to `vec3f(0,1,0)` — 3 299
 * outright normal flips at the band this was tuned for, against 5 for f16. The
 * synthetic study that chose 4R only sampled *surface* voxels and so could not
 * see it. f16 is 48x better in the mean at twice the bytes, and it is free of the
 * failure mode rather than further along the same curve.
 *
 * See {@link SparseBrickSceneGeometryFormat} for what each arm stores and
 * {@link SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII} for the band's own derivation.
 *
 * Only a dry world may narrow: the solver's own `solidSignedDistance` is in
 * metres on a lane this one is min'd against, so authoring water restores `f32x2`
 * regardless of this setting.
 *
 * Defined in `lib/sparse-brick-octree.ts` beside the format it selects, and
 * re-exported here because that is where every caller has always imported it
 * from. The renderer needs it too — the dry primary now reads this lane to
 * reconstruct its shading normal — and importing it from the module that owns
 * the codec keeps the shader off this one.
 */
export { octreeLiveSceneSceneGeometryFormat };

/**
 * How a dry world stores a leaf's 512-voxel payload.
 *
 * Defined in `lib/sparse-brick-octree.ts` beside the mode it selects, for the
 * same reason {@link octreeLiveSceneSceneGeometryFormat} is: the renderer resolves
 * it too — the dry primary's identity decode is compiled from the very layout the
 * voxeliser picked — and importing it from the module that owns the codec keeps
 * the shader off this one. Re-exported here because that is where every existing
 * caller imports it from.
 */
export { octreeLiveSceneLeafPayloadMode };

/**
 * Whether a dry world's primitive claim is the AABB or the solid inside it.
 *
 * `reachable` is the default and `aabb` is the one-word rollback, on the same
 * terms as `FLUID_SVO_DRY_PAYLOAD_PROFILE`: the saving is invisible in a CPU
 * test and a regression would not be, so the arm that shipped stays reachable
 * from the command line. Measured on the hero garden at 6.25 mm: 14 836 -> 14 136
 * leaves and 17 220 -> 16 377 node-mip pages, with the frame unchanged to 18
 * pixels of 392 920 at one part in 255.
 */
export function octreeLiveSceneBrickClaim(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): "aabb" | "reachable" {
  const raw = environment?.FLUID_SVO_BRICK_CLAIM;
  if (raw === undefined || raw === "") return "reachable";
  if (raw !== "aabb" && raw !== "reachable") throw new RangeError("FLUID_SVO_BRICK_CLAIM must be \"aabb\" or \"reachable\"");
  return raw;
}

export const OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY = 4_096;
/**
 * Aggregate parameter blocks one live publication may carry.
 *
 * Matched to the renderer's own cluster arena (`SVO_DRY_SCENE_CLUSTER_CAPACITY`)
 * so a scene that draws cannot fail to voxelize for want of a block. Overflow is
 * a loud throw in `publish`, not a silent envelope: an aggregate whose packing
 * did not arrive has no shape, and guessing one draws geometry nobody authored.
 */
export const OCTREE_LIVE_SCENE_CLUSTER_CAPACITY = 4_096;

/**
 * Field-program tapes one live publication may carry.
 *
 * Matched to `SPARSE_SCENE_FIELD_PROGRAM_CAPACITY`, for the same reason the
 * cluster capacity is matched to the renderer's: a scene that draws must not
 * fail to voxelize for want of a block.
 *
 * It defaulted to **zero** until the hero bonsai's crown became a tape, so the
 * `field-program` node kind was reachable through the graph, validated by the
 * ABI, packed by the dry-scene renderer — and rejected outright by the live
 * voxelizer with "publishes 1 field programs but the fixed arena holds 0". A
 * capacity nobody set is a capability nobody could use, and the whole path was
 * proven by tests rather than by any object in a scene.
 */
export const OCTREE_LIVE_SCENE_FIELD_PROGRAM_CAPACITY = SPARSE_SCENE_FIELD_PROGRAM_CAPACITY;
export const OCTREE_LIVE_SCENE_MATERIAL_CAPACITY = ENVIRONMENT_VOXEL_MATERIAL_BASE + OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY;

interface LiveScenePrimitiveState {
  readonly signature: string;
  readonly bounds: SparseSceneAxisAlignedBounds;
}

interface LiveScenePrimitiveEntry {
  readonly key: string;
  readonly primitive: SparseScenePrimitive;
  readonly materialSignature: string;
}

function brickCoordinateKey({ x, y, z }: SparseBrickCoordinate): string { return `${x},${y},${z}`; }

function liveScenePrimitiveSignature(primitive: SparseScenePrimitive): string {
  return JSON.stringify(primitive);
}

function sparseScenePrimitiveForRigidBody(body: RigidBodyDescription, ownerId: number): SparseScenePrimitive {
  const center = [body.position_m.x, body.position_m.y, body.position_m.z] as const;
  const orientation = [body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w] as const;
  const materialId = materialIdForRigidShape(body.shape);
  if (body.shape === "sphere") {
    return { kind: "ellipsoid", center, radii: [body.dimensions_m.x, body.dimensions_m.x, body.dimensions_m.x], materialId, ownerId };
  }
  if (body.shape === "box") {
    return { kind: "box", center, halfExtents: [body.dimensions_m.x / 2, body.dimensions_m.y / 2, body.dimensions_m.z / 2], orientation, materialId, ownerId };
  }
  if (body.shape === "capsule") {
    return { kind: "capsule", center, radius: body.dimensions_m.x, halfLength: body.dimensions_m.y / 2, orientation, materialId, ownerId };
  }
  // A cup's rigid dimensions carry the full height; every render vocabulary
  // downstream carries the half.
  if (body.shape === "cup") {
    return {
      kind: "cup", center, radius: body.dimensions_m.x, halfHeight: body.dimensions_m.y / 2,
      wallThickness: body.dimensions_m.z, orientation, materialId, ownerId,
    };
  }
  return { kind: "cylinder", center, radius: body.dimensions_m.x, halfHeight: body.dimensions_m.y / 2, orientation, materialId, ownerId };
}

/**
 * Page-level topology claims for the canonical SolidWorld image.
 *
 * These bounds allocate no geometry records and never expand pages into host
 * voxels or boxes. The live GPU voxelizer samples the same fraction/SDF page
 * image used by physics; this list only ensures those samples have destination
 * SVO bricks.
 */
function solidWorldPageBounds(scene: SceneDescription,
  world: SolidWorld): SparseSceneAxisAlignedBounds[] {
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m] as const;
  return world.pages.map((page) => {
    const minimum = page.coordinate.map((value, axis) => origin[axis]!
      + value * SOLID_WORLD_BRICK_CELLS * cell[axis]!) as
      [number, number, number];
    const maximum = minimum.map((value, axis) => value
      + SOLID_WORLD_BRICK_CELLS * cell[axis]!) as [number, number, number];
    return { minimum, maximum };
  });
}

function coalesceDirtyRegions(
  regions: readonly SparseSceneAxisAlignedBounds[],
  capacity = OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY,
): SparseSceneAxisAlignedBounds[] {
  if (regions.length <= capacity) return [...regions];
  return [{
    minimum: [
      Math.min(...regions.map(({ minimum }) => minimum[0])),
      Math.min(...regions.map(({ minimum }) => minimum[1])),
      Math.min(...regions.map(({ minimum }) => minimum[2])),
    ],
    maximum: [
      Math.max(...regions.map(({ maximum }) => maximum[0])),
      Math.max(...regions.map(({ maximum }) => maximum[1])),
      Math.max(...regions.map(({ maximum }) => maximum[2])),
    ],
  }];
}

export function liveSceneBrickCoordinatesForRegions(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
): SparseBrickCoordinate[] {
  return completeCooperativeBuild(liveSceneBrickCoordinatesForRegionsSteps(
    regions, worldOrigin, cellSize, brickSize, brickDimensions));
}

/** Bricks per yield offer; a brick here is a floor divide and a map insert. */
const BRICK_SELECTION_YIELD_BATCH = 8192;

/**
 * The same enumeration, offered as slices.
 *
 * Every authored bound is rasterised to the *refined* brick lattice, so this
 * grows with the cube of the refinement scale: at environment refinement depth
 * 3 it is 2.3 s on `hero-garden-hose`, held whole.
 */
export function* liveSceneBrickCoordinatesForRegionsSteps(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
): Generator<unknown, SparseBrickCoordinate[], undefined> {
  const coordinates = new Map<string, SparseBrickCoordinate>();
  let visited = 0;
  for (const region of regions) {
    const minimum = region.minimum.map((value, axis) => Math.floor(
      (value - worldOrigin[axis]) / (cellSize[axis] * brickSize),
    ));
    const maximum = region.maximum.map((value, axis) => Math.floor(
      (value - worldOrigin[axis]) / (cellSize[axis] * brickSize),
    ));
    for (let z = Math.max(0, minimum[2]); z <= Math.min(brickDimensions[2] - 1, maximum[2]); z += 1) {
      for (let y = Math.max(0, minimum[1]); y <= Math.min(brickDimensions[1] - 1, maximum[1]); y += 1) {
        for (let x = Math.max(0, minimum[0]); x <= Math.min(brickDimensions[0] - 1, maximum[0]); x += 1) {
          coordinates.set(`${x},${y},${z}`, { x, y, z });
          if ((visited += 1) % BRICK_SELECTION_YIELD_BATCH === 0) yield;
        }
      }
    }
  }
  return [...coordinates.values()];
}

/**
 * One authored solid, as far as brick selection is concerned.
 *
 * `bounds` must *contain* the solid and `distance_m` must never *over*-report
 * the distance to its surface. Both hold for an `SvoPrimitiveDescriptor` paired
 * with its `coverageBounds.conservative_m`, which is the only set the world
 * builds (`svoScenePrimitiveSolidReach`). A distance the caller cannot compute
 * is reported as `-Infinity`, which reads as "this solid reaches everywhere"
 * and keeps every brick — the safe direction.
 */
export interface SparseSceneSolidReach {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
  readonly distance_m: (x: number, y: number, z: number) => number;
}

/**
 * How far a solid must be from a brick's centre before the brick is certainly
 * empty, in units of the brick and the cell.
 *
 * The voxeliser marks a cell solid when its coverage distance is under one
 * cell radius, and that coverage is the minimum over the cell's centre and its
 * eight corners (`primitiveCellCoverageDistance`,
 * `lib/webgpu-sparse-scene-proxies.ts:1683`). So a voxel centred at `p` can only
 * be solid if some solid's surface is within `2 * cellRadius` of `p` — one
 * radius to reach the corner, one to satisfy the predicate. Every voxel centre
 * in a brick lies within the brick's circumradius of the brick centre, so a
 * solid further than `brickCircumradius + 2 * cellRadius` from the centre
 * cannot mark any of the brick's 512 voxels. Distances are 1-Lipschitz, so the
 * one evaluation at the centre decides the whole brick.
 */
function sparseSceneEmptyBrickMargin(
  cellSize: readonly [number, number, number],
  brickSize: number,
): number {
  const cellRadius = 0.5 * Math.hypot(cellSize[0], cellSize[1], cellSize[2]);
  return brickSize * cellRadius + 2 * cellRadius;
}

/**
 * Drop claimed bricks that no authored solid can reach.
 *
 * A primitive is claimed by its whole AABB, so a curved or thin one claims the
 * corners of a box it never enters: censused on the hero garden at 6.25 mm,
 * **1 981 of 4 433 claimed bricks hold no geometry at all**. Each one is 512
 * voxels of air plus a node-mip page of zeros, and removing it is exactly
 * neutral — an absent page and a resident page of zeros both sample as zero
 * coverage (`dryNodeMipAt` returns `valid = 1` either way,
 * `lib/webgpu-svo-dry-scene.ts:1826`).
 *
 * This is emphatically *not* true of an interior brick, which reads as solid
 * today and would read as a hole if it were dropped. Interiors are left in the
 * claim; making them cheap is a payload question, not a selection one.
 *
 * `pinned` names bricks some other claim already owns — the solver's container
 * and the ground — so the test is only paid where its answer can change the
 * plan.
 */
export function liveSceneReachableBrickCoordinates(
  coordinates: readonly SparseBrickCoordinate[],
  solids: readonly SparseSceneSolidReach[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  pinned?: ReadonlySet<string>,
): SparseBrickCoordinate[] {
  return completeCooperativeBuild(liveSceneReachableBrickCoordinatesSteps(
    coordinates, solids, worldOrigin, cellSize, brickSize, pinned));
}

/**
 * The same narrowing, offered as slices.
 *
 * This is the single longest block a refined scene build had left after the
 * plan itself was sliced, and it was invisible because of where it sits: it was
 * evaluated inside the *argument literal* of the octree-plan call, so it ran
 * after the "Plan the adaptive octree" mark and before the plan generator's
 * first offer. Measured end to end on `hero-garden-hose` at environment
 * refinement depth 3 that is 17.5 s in which nothing yields, against 22.7 s for
 * the whole planning stage — an exact distance function per candidate brick,
 * per overlapping solid.
 *
 * Batched small because the per-item cost is a real SDF evaluation rather than
 * a map insert.
 */
export function* liveSceneReachableBrickCoordinatesSteps(
  coordinates: readonly SparseBrickCoordinate[],
  solids: readonly SparseSceneSolidReach[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  pinned?: ReadonlySet<string>,
): Generator<unknown, SparseBrickCoordinate[], undefined> {
  if (solids.length === 0) return [...coordinates];
  const margin = sparseSceneEmptyBrickMargin(cellSize, brickSize);
  const edge = cellSize.map((value) => value * brickSize);
  const reachable: SparseBrickCoordinate[] = [];
  let visited = 0;
  for (const coordinate of coordinates) {
    // 64 rather than the thousands a flat pass uses: the body is an SDF
    // evaluation per overlapping solid, and the batch has to stay inside a
    // slice at the *worst* per-candidate cost rather than the average one.
    if ((visited += 1) % 64 === 0) yield;
    if (pinned?.has(brickCoordinateKey(coordinate))) { reachable.push(coordinate); continue; }
    const minimum = [coordinate.x, coordinate.y, coordinate.z]
      .map((value, axis) => worldOrigin[axis] + value * edge[axis]);
    const maximum = minimum.map((value, axis) => value + edge[axis]);
    const centre = minimum.map((value, axis) => value + edge[axis] / 2);
    let reached = false;
    for (const solid of solids) {
      if (solid.maximum[0] < minimum[0] - margin || solid.minimum[0] > maximum[0] + margin) continue;
      if (solid.maximum[1] < minimum[1] - margin || solid.minimum[1] > maximum[1] + margin) continue;
      if (solid.maximum[2] < minimum[2] - margin || solid.minimum[2] > maximum[2] + margin) continue;
      if (solid.distance_m(centre[0], centre[1], centre[2]) > margin) continue;
      reached = true;
      break;
    }
    if (reached) reachable.push(coordinate);
  }
  return reachable;
}

/**
 * Missing finest-brick coverage only; existing coarse/fine leaves remain reusable.
 *
 * `covered` may be the flat set of finest-brick keys, or a predicate — which is
 * what a world with coarse leaves passes, because materialising every finest
 * brick a coarse leaf covers is the claim's *volume* and it is the one thing a
 * sub-millimetre lattice cannot afford on the host.
 */
export function liveSceneMissingBrickCoordinates(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
  covered: ReadonlySet<string> | ((coordinate: SparseBrickCoordinate) => boolean),
): SparseBrickCoordinate[] {
  const isCovered = typeof covered === "function"
    ? covered
    : (coordinate: SparseBrickCoordinate) => covered.has(brickCoordinateKey(coordinate));
  return liveSceneBrickCoordinatesForRegions(regions, worldOrigin, cellSize, brickSize, brickDimensions)
    .filter((coordinate) => !isCovered(coordinate));
}

export function liveSvoDenseFinestPages(dimensions: readonly [number, number, number]): SvoNodeMipCoordinate[] {
  const pages: SvoNodeMipCoordinate[] = [];
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    pages.push([x, y, z]);
  }
  return pages;
}

/** Finest 8-cell node-mip page grid for any supported canonical brick size. */
export function liveSvoBasePageDimensions(
  brickDimensions: readonly [number, number, number],
  brickSize: SparseBrickSize,
): readonly [number, number, number] {
  return brickDimensions.map((bricks) => Math.ceil(
    bricks * brickSize / SVO_NODE_MIP_LAYOUT.interiorSize,
  )) as [number, number, number];
}

/** Compact vec4 emission table consumed by the GPU-only live radiance builder. */
function packLiveSvoMaterialEmission(packedRecords: Uint32Array, count: number): Float32Array<ArrayBuffer> {
  const source = new Float32Array(packedRecords.buffer, packedRecords.byteOffset, packedRecords.byteLength / 4);
  const recordWords = SVO_MATERIAL_RECORD_STRIDE_BYTES / 4;
  const result = new Float32Array(OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * 4);
  for (let material = 0; material < count; material += 1) {
    result.set(source.subarray(material * recordWords + 4, material * recordWords + 7), material * 4);
  }
  return result;
}

export interface OctreeSvoPbrMaterialPublicationData {
  packedRecords: Uint32Array<ArrayBuffer>;
  count: number;
  strideBytes: number;
  revision: number;
  contentRevision: string;
  cacheKey: string;
}

const octreeSvoPbrMaterialCache = new Map<string, OctreeSvoPbrMaterialPublicationData>();

/**
 * Dense default table used by the producer and CPU ABI/lifecycle tests.
 *
 * `surfaceModel` is the scene's own declaration, not a per-primitive one, and it
 * is part of the content revision because two scenes with identical proxies and
 * different ground models publish different closures.
 */
export function buildOctreeSvoPbrMaterialPublication(
  revision = OCTREE_SVO_PBR_MATERIAL_REVISION,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[] = [],
  surfaceModel: SvoTerrainSurfaceModel = "garden-terrain",
): OctreeSvoPbrMaterialPublicationData {
  if (!Number.isSafeInteger(revision) || revision < 1 || revision > 0xffff_ffff) {
    throw new RangeError("SVO PBR material publication revision must be a positive uint32");
  }
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    revision,
    surfaceModel,
    environmentPrimitives: environmentPrimitives.map(({ key, ownerIndex, group, tags, material }) => ({ key, ownerIndex, group, tags, material })),
  }));
  const cacheKey = `octree-svo-pbr-material-v1:${contentRevision}`;
  const cached = cachedSvoPublication(octreeSvoPbrMaterialCache, cacheKey);
  if (cached) return cached;
  const records = [
    ...buildDefaultSvoMaterialRecords(revision, { terrainSurface: surfaceModel }),
    ...environmentPrimitives.map((primitive) => {
      if (!Number.isSafeInteger(primitive.ownerIndex) || primitive.ownerIndex < 0) {
        throw new RangeError(`Environment material owner index for ${primitive.key} must be a non-negative safe integer`);
      }
      const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
      if (materialId > 0xffff) throw new RangeError(`Environment material ID for ${primitive.key} does not fit uint16`);
      return svoMaterialFromEnvironmentProxyMaterial(
        materialId,
        primitive.material,
        revision,
        svoMaterialFunctionIdForEnvironmentProxy(primitive, surfaceModel),
      );
    }),
  ];
  const packedRecords = packSvoMaterialTable(records);
  return internSvoPublication(octreeSvoPbrMaterialCache, cacheKey, {
    packedRecords,
    count: packedRecords.byteLength / SVO_MATERIAL_RECORD_STRIDE_BYTES,
    strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES,
    revision,
    contentRevision,
    cacheKey,
  });
}

export interface OctreeSvoLightPublicationData {
  records: readonly SvoLightRecord[];
  packedRecords: Uint32Array<ArrayBuffer>;
  count: number;
  strideBytes: number;
  revision: number;
  omittedFixtureKeys: readonly string[];
  contentRevision: string;
  cacheKey: string;
}

const octreeSvoLightCache = new Map<string, OctreeSvoLightPublicationData>();

/** Build the selected scene/environment's deterministic bounded light table. */
export function buildOctreeSvoLightPublication(
  scene: SceneDescription,
  options: { revision?: number; maximumRecords?: number } = {},
): OctreeSvoLightPublicationData {
  const revision = options.revision ?? OCTREE_SVO_LIGHT_REVISION;
  const lights = buildSvoSceneLights(scene, {
    revision,
    maximumRecords: options.maximumRecords ?? SVO_LIGHT_MAXIMUM_RECORDS,
  });
  const cacheKey = `octree-${lights.cacheKey}`;
  return internSvoPublication(octreeSvoLightCache, cacheKey, {
    records: lights.records,
    packedRecords: lights.packedRecords,
    count: lights.records.length,
    strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES,
    revision: lights.revision,
    omittedFixtureKeys: lights.omittedFixtureKeys,
    contentRevision: lights.contentRevision,
    cacheKey,
  });
}

export interface OctreeSvoEnvironmentLightingPublicationData {
  record: SvoEnvironmentLightingRecord;
  packedRecords: Uint32Array<ArrayBuffer>;
  count: 1;
  strideBytes: number;
  revision: number;
  cacheKey: string;
}

/** Build the selected environment's single image-free lighting record. */
export function buildOctreeSvoEnvironmentLightingPublication(
  scene: Pick<SceneDescription, "environment" | "systems" | "lighting">,
  revision = OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION,
): OctreeSvoEnvironmentLightingPublicationData {
  const lighting = buildSvoEnvironmentLighting(
    scene.environment ?? "default", revision, svoSceneLighting(scene)?.environment);
  return {
    record: lighting.record,
    packedRecords: lighting.packedRecord,
    count: 1,
    strideBytes: SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
    revision: lighting.record.revision,
    cacheKey: lighting.cacheKey,
  };
}

function storageBuffer(device: GPUDevice, label: string, size: number, data?: ArrayBufferView<ArrayBuffer>) {
  const buffer = device.createBuffer({ label, size: Math.max(4, size), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (data && data.byteLength > 0) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** Fields a live scene publication makes valid before fluid payload is attached. */
export const OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS =
  SPARSE_VOXEL_VALID_FIELDS.topology
  | SPARSE_VOXEL_VALID_FIELDS.sceneGeometry
  | SPARSE_VOXEL_VALID_FIELDS.materialOwner;

function structuralPublicationFinalizeShader(sceneMaintenanceStateOffsetWords = 0) { return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@group(0) @binding(1) var<storage, read> sceneMaintenance: array<u32>;
@group(0) @binding(2) var<storage, read> topologyMutation: array<u32>;

// Scene geometry is its own live generation. Fluid coverage is intentionally
// published through a separate renderer channel so evolving physics can never
// overwrite authored scene voxels in-place.
const SCENE_VALID_FIELDS: u32 = ${OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS}u;
const PHYSICS_VALID_FIELDS: u32 = ${
  SPARSE_VOXEL_VALID_FIELDS.dynamicSolid |
  SPARSE_VOXEL_VALID_FIELDS.coarseFluid |
  SPARSE_VOXEL_VALID_FIELDS.velocity
}u;

@compute @workgroup_size(1)
fn finalizePhysics() {
  state[${SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision}] += 1u;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}] += 1u;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}] |= PHYSICS_VALID_FIELDS;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}] += 1u;
}

@compute @workgroup_size(1)
fn finalizeScene() {
  let maintenanceOffset = ${sceneMaintenanceStateOffsetWords}u;
  let requested = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.requestedRevision}u];
  let completed = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision}u];
  let overflow = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.overflowFlags}u];
  // Only the overflows that left a brick holding the previous revision withhold
  // the generation. A per-brick candidate overflow is a detail budget: the brick
  // was rebuilt, just from fewer primitives than overlapped it, and refusing the
  // whole publication for it takes every derived-lighting page down with it.
  // See SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW.
  if (requested == 0u || completed != requested
    || (overflow & ${SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW}u) != 0u
    || topologyMutation[3] != 0u) { return; }
  if (topologyMutation[0] != 0u) { state[${SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision}] += 1u; }
  state[${SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision}] += 1u;
  // Scene publication adds its independently-owned lane; it must never clear
  // already-current fluid/dynamic fields while physics is paused.
  state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}] |= SCENE_VALID_FIELDS;
  // Last: topology, payload maintenance, and the stores above become visible
  // to every render consumer as one live scene generation.
  state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}] += 1u;
}
`; }

/**
 * Transitional GPU bridge: the octree solver remains authoritative while its
 * resident level set, velocity and VOS solid field are published into one
 * sparse-brick ABI for scene inspection and subsequent sparse kernels.
 */
export class OctreeSparseBrickWorld {
  tree!: SparseBrickOctreeGPU;
  /** Narrow two-sided band used by surface and topology scheduling. */
  residency!: GPUFluidBrickResidency;
  /** Full wet-domain topology residency, independent of the surface band. */
  bulkResidency?: GPUFluidBrickResidency;
  sceneSource!: SparseVoxelSceneRenderSource;
  private preActivation!: boolean;

  private device!: GPUDevice;
  private dimensions!: readonly [number, number, number];
  private solverGridOriginCells!: readonly [number, number, number];
  private finestLevel!: number;
  private cellSize!: readonly [number, number, number];
  private source!: SparseBrickPublicationSource;
  private sourceBuffers!: GPUBuffer[];
  private pbrMaterialBuffer!: GPUBuffer;
  private materialEmissionBuffer!: GPUBuffer;
  private lightBuffer!: GPUBuffer;
  private environmentLightingBuffer!: GPUBuffer;
  private baseAllocatedBytes!: number;
  private structuralPublicationState!: GPUBuffer;
  private structuralPhysicsPipeline!: GPUComputePipeline;
  private structuralScenePipeline!: GPUComputePipeline;
  private structuralModule!: GPUShaderModule;
  private structuralPipelineLayout!: GPUPipelineLayout;
  private structuralFinalizeBindGroup!: GPUBindGroup;
  private proxyVoxelizer!: SparseSceneProxyVoxelizer;
  private topologyMutator!: WebGpuSparseBrickTopologyMutator;
  private topologyMutationWorklist!: GPUBuffer;
  private topologyMutationCapacity!: number;
  private sceneBrickDimensions!: readonly [number, number, number];
  private brickSize!: SparseBrickSize;
  private sceneWorldOrigin!: readonly [number, number, number];
  private brickOccupancyBuilder!: WebGpuSvoBrickOccupancyBuilder;
  private wideFanout?: WebGPUSvoWideFanout;
  private compactHierarchy?: WebGpuSvoCompactHierarchy;
  private nodeMipPyramid?: WebGpuLiveSvoNodeMipPyramid;
  private tetrahedralRadiance?: WebGpuLiveSvoTetrahedralRadiance;
  private liveDerivedPlanner?: WebGpuLiveSvoDerivedWorklistPlanner;
  private liveDerivedBuilder?: WebGpuLiveSvoDerivedBuilder;
  /**
   * The addresses derived lighting can reach, which is no longer fixed at the
   * first publication. See `lib/svo-node-mip-address-plan.ts`; the two shapes
   * it can take are "every page in the domain" and "what is occupied plus a
   * bounded reserve to grow into".
   */
  private liveDerivedAddressPlan?: SvoNodeMipAddressPlan;
  private liveDerivedAddressPlanValid = true;
  private liveDerivedFeedbackPhase = 0;
  private liveDerivedFeedbackFramesRemaining = 0;
  private liveDerivedInitial = true;
  /** The scene's one word about what its surfaces are made of; see the ABI. */
  private surfaceModel!: SvoTerrainSurfaceModel;
  private solidWorld!: SolidWorld;
  private solidWorldStamp = "";
  private liveScenePrimitiveStates = new Map<string, LiveScenePrimitiveState>();
  private liveScenePrimitives = new Map<string, LiveScenePrimitiveEntry>();
  /**
   * Which finest bricks a leaf already covers, stored *at each leaf's own level*.
   *
   * This used to be the expansion: every leaf enumerated every finest brick in
   * its extent, so the set was the claim's volume rather than its leaf count —
   * on `hero-garden-hose` at environment refinement depth 3 that is 4.4 M
   * strings against 222 690 leaves, several hundred megabytes of host heap in
   * the one place a scene this fine has none to spare. A brick is covered iff
   * one of its ancestors is here, which is at most `finestLevel + 1` lookups
   * and is asked only about the bricks an *edit* touches.
   */
  private readonly coveredSceneBrickNodes = new Set<string>();
  private readonly pendingTopologyCoordinates = new Map<string, SparseBrickCoordinate>();
  private pendingScenePublication?: SparseScenePublication;
  private pendingTopologyMutation = false;
  private sceneRevision = 0;
  private topologyPublished = false;
  private pipelineInitialization?: Promise<void>;
  private destroyed = false;
  /**
   * The suspended build, when `deferBuild` asked for one.
   *
   * Held for exactly as long as it takes {@link create} to take it, so a world
   * whose build was deferred and never driven is a programming error rather
   * than a silent half-world; every accessor below would fault on the first
   * unassigned field.
   */
  private deferredBuild?: Generator<unknown, void, undefined>;
  /**
   * Whether {@link buildSteps} ran to completion.
   *
   * Set by the last statement of the build and checked by {@link create}, so
   * "the driver ran the generator to the end" is asserted rather than assumed —
   * a driver that returned early would otherwise hand back a world whose later
   * fields are undefined and whose first fault is somewhere in a frame.
   */
  private built = false;

  /**
   * Build the world, holding the thread for its whole duration.
   *
   * Kept for the callers that are already synchronous — the octree solver's own
   * world, the stress-scene harness and the CPU tests. The browser's render
   * worker uses {@link create} instead, which runs the identical body in
   * slices, so there is only ever one definition of what a world *is*.
   */
  constructor(device: GPUDevice, scene: SceneDescription, dimensions: readonly [number, number, number], options: OctreeSparseBrickWorldOptions = {}) {
    const steps = this.buildSteps(device, scene, dimensions, options);
    if (options.deferBuild) { this.deferredBuild = steps; return; }
    completeCooperativeBuild(steps);
  }

  /**
   * Build the world without wedging the thread it runs on.
   *
   * The constructor is a single uninterrupted synchronous block and on a
   * refined world it is a long one — 40 s at environment refinement depth 3 on
   * `hero-garden-hose`. In the render worker that is 40 s in which no `draw`,
   * no scene republication and no *abort* can be serviced, because every one of
   * them arrives on the message queue the build is holding shut. So a slider
   * move during a build could not stop it, and the rebuild it asked for ran
   * only after the superseded one had finished in full.
   *
   * Driving the same body as a generator fixes both. Between slices the thread
   * returns to the event loop, so the worker answers messages while the world
   * is being planned; and the abort signal is read there, so a superseded build
   * stops at the next slice rather than at the next retry-ladder rung.
   *
   * ### What an abandoned build leaves behind
   *
   * Nothing that outlives this call. Every `yield` in `buildSteps` is placed at
   * a point where every GPU object created so far is reachable from a field
   * this instance has already assigned — the tree, the residencies, the
   * voxeliser, the source buffers and the material tables are all assigned in
   * the same statement that creates them, and the node-mip block, whose
   * intermediates are locals until the end, contains no yield at all. So
   * `destroy()` below releases exactly what exists, and it is called here
   * before the rejection propagates. The half-built world is unreachable
   * afterwards: it was never returned, never attached, and nothing appends to
   * it — the next request builds a new one from scratch.
   *
   * ### What is still whole
   *
   * The granularity is only as fine as the coarsest block between two offers,
   * and *where an expression sits* decides which two those are — the reach
   * narrowing was 17.5 s of frozen worker purely because it was an argument of
   * the plan call rather than a statement before it.
   *
   * What is still whole, measured: three sorts — one per plan — at 30 ms in the
   * node-mip address plan, 34 ms in the wide-fanout plan (which only a lane that
   * sets `derivedTraversalStructures` builds at all now) and one per octree
   * level; `packSparseBrickPlan` at 32 ms, which writes the payload lane
   * tables; the ground's own brick claim at 66 ms; and one
   * `bakeProceduralTerrain` at 282 ms on the first build of a procedural
   * ground, content-cached for every build after. A sort is the natural floor
   * here: it cannot be suspended, so the only way to shrink one is to make it
   * cheaper, which is why both plan comparators decorate their Morton key
   * rather than recompute it. Nothing else exceeds ~20 ms.
   */
  static async create(
    device: GPUDevice,
    scene: SceneDescription,
    dimensions: readonly [number, number, number],
    options: OctreeSparseBrickWorldOptions = {},
    interrupt: CooperativeBuildOptions = {},
  ): Promise<OctreeSparseBrickWorld> {
    const world = new OctreeSparseBrickWorld(device, scene, dimensions, { ...options, deferBuild: true });
    const steps = world.deferredBuild;
    world.deferredBuild = undefined;
    if (!steps) throw new Error("Deferred sparse-brick world build was already consumed");
    try {
      await driveCooperativeBuild(steps, interrupt);
      if (!world.built) throw new Error("Sparse-brick world build returned before it finished");
    } catch (error) {
      world.destroy();
      throw error;
    }
    return world;
  }

  /**
   * The build itself.
   *
   * A generator rather than the constructor body it used to be, so the caller
   * decides how often the thread goes back to the event loop. The fields it
   * assigns are therefore no longer `readonly` — TypeScript only admits those
   * assignments from a constructor — and the invariant they carried is now
   * carried by this: nothing outside this method assigns them, and `built`
   * records when the last of them has been.
   */
  private *buildSteps(device: GPUDevice, scene: SceneDescription, dimensions: readonly [number, number, number], options: OctreeSparseBrickWorldOptions = {}): Generator<unknown, void, undefined> {
    this.device = device;
    this.dimensions = dimensions;
    const brickSize = options.brickSize ?? 8;
    this.brickSize = brickSize;
    this.surfaceModel = sceneTerrainSurfaceModel(scene);
    const environmentCatalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
    const environmentPrimitives = environmentProxyPrimitives(environmentCatalog, true);
    const initialSolidWorld = solidWorldForScene(scene);
    this.solidWorld = initialSolidWorld;
    this.solidWorldStamp = solidWorldContentStamp(scene);
    const solidWorldBounds = solidWorldPageBounds(scene, initialSolidWorld);
    // This tree is a sparse presentation consumer, not the fluid solver's
    // address space. Fluid residency claims wet pages as they appear; static
    // geometry claims only the exact voxel boxes compiled from SolidWorld.
    // Claiming the whole logical container here was the render-side OOM path
    // for long, mostly dry tanks.
    const dryWorld = scene.systems?.fluid === false;
    const claimsContainer = false;
    const sceneDomain = planSparseSceneDomain(
      scene, dimensions, brickSize,
      [
        ...environmentPrimitives.map((primitive) => ({ min: primitive.aabb_m.min,
          max: primitive.aabb_m.max })),
        ...solidWorldBounds.map((bounds) => {
          return { min: { x: bounds.minimum[0], y: bounds.minimum[1], z: bounds.minimum[2] },
            max: { x: bounds.maximum[0], y: bounds.maximum[1], z: bounds.maximum[2] } };
        }),
      ],
      {
        conservativePaddingCells: 1,
        worldBounds_m: scene.voxelDomain.bounds_m,
        solverClaim: "none",
      }
    );
    this.solverGridOriginCells = sceneDomain.solverGridOriginCells;
    this.sceneBrickDimensions = sceneDomain.brickDimensions;  // replaced below once the refinement depth is known
    this.sceneWorldOrigin = [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z];
    const solverLevel = sparseSceneOctreeMaximumDepth(sceneDomain.brickDimensions, sceneDomain.coordinates);
    /**
     * Extra octree levels the *environment* may use, below the solver's own.
     *
     * Only on a scene the solver does not own. `planSparseSceneDomain` claims
     * every brick of the container as a solver brick while a solver is present,
     * and a solver brick pins its node at `solverLevel` — so while the
     * simulation is there, there is nowhere for environment geometry to descend
     * into, and the only resolution knob is `finestCellSize_m` applied to the
     * whole domain. On a dry scene there is no such claim, and the render tree
     * is free to spend depth where the geometry actually is.
     */
    const refinementDepth = dryWorld
      ? Math.max(0, Math.trunc(options.environmentRefinementDepth ?? 0))
      : 0;
    const refineScale = 2 ** refinementDepth;
    const maximumDepth = solverLevel + refinementDepth;
    const refinedBrickDimensions = sceneDomain.brickDimensions.map((value) => value * refineScale) as [number, number, number];
    const renderCellSize = sceneDomain.cellSize_m.map((value) => value / refineScale) as [number, number, number];
    const refinedBrickEdge = renderCellSize.map((value) => value * brickSize) as [number, number, number];
    const worldOrigin = [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z] as const;
    /**
     * Announce a stage, by its position in the declared list.
     *
     * Looking the index up rather than counting by hand means a stage cannot be
     * reported out of order or with a stale total, and the list stays the single
     * statement of what this constructor does.
     */
    const reportStage = (stage: OctreeSparseBrickWorldStage): void => options.progress?.({
      stage,
      completed: OCTREE_SPARSE_BRICK_WORLD_STAGES.indexOf(stage),
      total: OCTREE_SPARSE_BRICK_WORLD_STAGES.length,
    });
    /**
     * Node edge in metres at each level, so the hot predicate stops calling
     * `Math.pow` three times per node. At environment refinement depth 3 the
     * planner visits a quarter of a million nodes and this ran once per axis
     * per visit — measurably the single hottest symbol in the whole world build.
     */
    const nodeEdge_m: (readonly number[])[] = [];
    for (let level = 0; level <= maximumDepth; level += 1) {
      const scale = 2 ** (maximumDepth - level);
      nodeEdge_m.push(refinedBrickEdge.map((value) => value * scale));
    }
    /**
     * Primitives whose bounds touch a brick — what the voxeliser bins per leaf.
     *
     * `limit` stops the scan as soon as the answer cannot change. Every caller
     * compares against a threshold rather than using the count, and the whole
     * primitive list is scanned per node, so a dense node used to pay the full
     * catalogue to learn something the first few entries already settled.
     */
    const candidatesInBrick = (level: number, coordinate: SparseBrickCoordinate, limit = Infinity): number => {
      const edge = nodeEdge_m[level];
      const loX = worldOrigin[0] + coordinate.x * edge[0], hiX = loX + edge[0];
      const loY = worldOrigin[1] + coordinate.y * edge[1], hiY = loY + edge[1];
      const loZ = worldOrigin[2] + coordinate.z * edge[2], hiZ = loZ + edge[2];
      let count = 0;
      for (const primitive of environmentPrimitives) {
        const { min, max } = primitive.aabb_m;
        if (max.x < loX || min.x > hiX || max.y < loY || min.y > hiY || max.z < loZ || min.z > hiZ) continue;
        count += 1;
        if (count > limit) return count;
      }
      return count;
    };
    /**
     * The curvature-driven detail rule, when the lever asks for it.
     *
     * Built here rather than at the plan call because it owns the same
     * primitive catalogue `candidatesInBrick` scans and carries a narrowed
     * candidate set down the planner's recursion; see
     * `lib/svo-environment-refinement.ts` for why the crowding count above is a
     * cost control and not the detail rule it was standing in for.
     */
    const environmentDescriptorFor = options.environmentDescriptorFor;
    const surfaceRefinement = refinementDepth > 0 && environmentDescriptorFor
      && svoEnvironmentRefinementMode() === "surface"
      ? createSvoEnvironmentRefinement({
        primitives: environmentPrimitives, descriptorFor: environmentDescriptorFor,
        worldOrigin_m: worldOrigin as readonly [number, number, number],
        nodeEdge_m, brickSize, maximumDepth,
        crowdingTarget: OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
        planarExemption: options.environmentPlanarRefinementExemption === true,
      })
      : undefined;
    /**
     * The set's own resolution, on the one path that never had one.
     *
     * A simulated container pins every brick at `solverLevel`, so a wet scene's
     * only knob was `finestCellSize_m` over the whole domain and an authored
     * floor was drawn at the water's cell however large the floor was. See
     * `lib/svo/svo-environment-coarsening.ts` for the rule and for why the
     * ceiling is `log2(brickSize)`; it is offered here and taken per node.
     *
     * Wet only. A solverless world already has `environmentRefinementDepth`
     * below the lattice and buried-ground coarsening above it, each with its own
     * predicate and its own gate, and `minimumEnvironmentLevel` can only carry
     * one meaning at a time.
     */
    const environmentCoarsening = !dryWorld
      ? createSvoEnvironmentCoarsening({
        primitives: environmentPrimitives,
        worldOrigin_m: worldOrigin as readonly [number, number, number],
        nodeEdge_m, brickSize, maximumDepth,
        crowdingTarget: OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
      })
      : undefined;
    const pinnedBricks = new Set<string>();
    if (refinementDepth === 0) {
      for (const coordinate of sceneDomain.solverBrickCoordinates) pinnedBricks.add(brickCoordinateKey(coordinate));
    }
    const sceneSolids = dryWorld && octreeLiveSceneBrickClaim() === "reachable"
      ? options.sceneSolids ?? [] : [];
    reportStage("Select the bricks the scene reaches");
    yield;
    const primitiveBricks = refinementDepth > 0
      ? yield* liveSceneBrickCoordinatesForRegionsSteps(
        environmentPrimitives.map((primitive) => ({
          minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y, primitive.aabb_m.min.z] as const,
          maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y, primitive.aabb_m.max.z] as const,
        })),
        worldOrigin, renderCellSize, brickSize, refinedBrickDimensions)
      : sceneDomain.proxyBrickCoordinates.slice(0, environmentPrimitives.length).flat();
    const solidWorldBricks = yield* liveSceneBrickCoordinatesForRegionsSteps(
      solidWorldBounds, worldOrigin, renderCellSize, brickSize, refinedBrickDimensions);
    /**
     * The static rigid bodies, which nothing else in the claim accounts for.
     *
     * `stageSceneUpdate` voxelizes the `motion: "static"` bodies alongside the
     * authored proxies — dynamic bodies stay analytic, their pose is
     * solver-owned and a voxel copy would freeze at the authored drop point —
     * but only the proxies reach `planSparseSceneDomain`: a body's bricks
     * existed because the *container's* claim covered them. Once that claim is
     * gone they have to be asked for, and the first publication is the one
     * that cannot create a missing brick, so it has to happen here. Measured
     * on `garden-svo-lighting`: exactly one of its 21 body bricks is claimed by
     * no other producer, and without this it would silently stop being voxelized.
     *
     * Only where the container claim was dropped. Everywhere else these bricks
     * are already covered and adding them would change a shipped plan.
     */
    const rigidBodyBricks = claimsContainer ? [] : yield* liveSceneBrickCoordinatesForRegionsSteps(
      scene.rigidBodies.flatMap((body, ownerId) => body.motion === "static"
        ? [sparseScenePrimitiveBounds(sparseScenePrimitiveForRigidBody(body, ownerId))] : []),
      worldOrigin, renderCellSize, brickSize, refinedBrickDimensions);
    /**
     * What the planner is told the solver owns, which is nothing on a dry world.
     *
     * Two consumers below read this rather than `sceneDomain` directly: the
     * planner, and the fluid brick table's "every solver brick has a finest
     * leaf" assertion — which is a statement about *this* set, not about the
     * refinement depth that used to stand in for it.
     */
    const plannedSolverBricks = refinementDepth > 0 ? [] : sceneDomain.solverBrickCoordinates;
    /**
     * The narrowed primitive claim, computed *before* the plan call rather than
     * inside its argument literal.
     *
     * Where an expression is evaluated is normally a style question. Here it
     * was 17.5 s of frozen worker: an argument literal is evaluated after the
     * stage mark above and before the plan generator's first offer, so this —
     * one exact distance function per candidate brick, per overlapping solid —
     * sat in the one place between two yield points where nothing could
     * interrupt it, and it was the longest such block in the whole build.
     */
    const reachablePrimitiveBricks = yield* liveSceneReachableBrickCoordinatesSteps(
      primitiveBricks, sceneSolids, worldOrigin, renderCellSize, brickSize, pinnedBricks);
    reportStage("Plan the adaptive octree");
    yield;
    // The interruptible form of the same plan. This is the longest block in a
    // refined build by a wide margin, so yielding only *around* it would leave
    // most of the freeze exactly where it was.
    const plan = yield* planAdaptiveSparseBrickOctreeSteps({
      brickSize,
      // A dry scene has no simulation to pin bricks for. Handing the planner the
      // container anyway is what made the tree uniform-depth in practice.
      solverBricks: plannedSolverBricks,
      // Ground bricks are proxy bricks like any other content's. Inside the
      // container they are already claimed as solver bricks and this adds
      // nothing; outside it — where the garden's scenery stands beyond the
      // container footprint — they are the difference between ground that
      // occludes and ground that is merely drawn.
      //
      // The primitive half of the claim is narrowed from each AABB to the
      // bricks its solid can actually reach, which drops the corners of the box
      // it never enters. Those bricks are all air today and an absent page and
      // a resident page of zeros sample identically, so the frame does not move.
      proxyBricks: [
        ...reachablePrimitiveBricks,
        ...solidWorldBricks,
        ...rigidBodyBricks,
      ],
      maximumDepth,
      solverLevel,
      maximumEnvironmentCoarseningPower: Math.min(
        Math.max(
          environmentMaximumCoarseningPower(options.environmentBrickRefinementLevels),
          // Offered, not spent: `environmentCoarsening` is the predicate that
          // decides how much of it each node takes, and a scene whose solids
          // are all finer than the coarse voxel takes none of it.
          environmentCoarsening ? svoEnvironmentCoarseningPower(brickSize) : 0,
        ),
        solverLevel,
      ),
      // Split for primitive crowding or for a non-planar primitive surface.
      // SolidWorld boxes already claim their exact voxel bricks and therefore
      // need no heightfield-shaped refinement arm.
      refineEnvironmentLeaf: surfaceRefinement
        ? (level, coordinate) => surfaceRefinement.refineEnvironmentLeaf(level, coordinate)
        : refinementDepth > 0
          ? (level, coordinate) => (
            candidatesInBrick(level, coordinate, OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET)
              > OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET
          )
          : environmentCoarsening
            ? (level, coordinate) => environmentCoarsening.refineEnvironmentLeaf(level, coordinate)
            : undefined,
    });
    this.finestLevel = plan.maximumDepth;
    this.sceneBrickDimensions = refinedBrickDimensions;
    reportStage("Pack the octree and allocate its arenas");
    yield;
    let coveredLeaves = 0;
    for (const leaf of plan.leaves) {
      const node = plan.nodes[leaf.nodeIndex];
      this.coveredSceneBrickNodes.add(`${node.level}:${brickCoordinateKey(node.coordinate)}`);
      if ((coveredLeaves += 1) % 4096 === 0) yield;
    }
    const packed = packSparseBrickPlan(plan, 1);
    yield;
    const sceneBrickVolume = sceneDomain.brickDimensions.reduce((product, value) => product * value, 1);
    const requestedMutationCapacity = options.sceneMutationBrickCapacity ?? OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY;
    if (!Number.isSafeInteger(requestedMutationCapacity) || requestedMutationCapacity < 1) {
      throw new RangeError("Live scene mutation brick capacity must be a positive safe integer");
    }
    this.topologyMutationCapacity = Math.min(sceneBrickVolume, requestedMutationCapacity);
    const nodeCapacity = Math.max(1, plan.nodes.length
      + sparseBrickTopologyMutationNodeReserve(maximumDepth, this.topologyMutationCapacity));
    const leafCapacity = Math.max(1, plan.leaves.length + this.topologyMutationCapacity);
    // The loud half of the brick raster's node-index ceiling, at the one point
    // where the count first exists. Capacity rather than the planned count on
    // purpose: topology mutation can reach every reserved node, so a world that
    // *could* grow past the mask is already broken, and finding that out at the
    // publication that crosses it is finding out from a scene with holes in it.
    assertSvoBrickRasterNodeAddressable(nodeCapacity, "the live sparse-brick world");
    // `systems.fluid === false` is already the world's dry/wet discriminator
    // (see `refinementDepth` above and the inspection note at :693). Only a dry
    // world may narrow its lanes: a solver writes all five, so the profile is
    // forced back to `full` the moment water is authored, whatever the lever
    // says. That is the whole reversibility contract.
    const payloadProfile: SparseBrickPayloadProfileName =
      scene.systems?.fluid === false ? octreeLiveSceneDryPayloadProfile() : "full";
    const sceneGeometryFormat: SparseBrickSceneGeometryFormat =
      payloadProfile === "dry" ? octreeLiveSceneSceneGeometryFormat() : "f32x2";
    const leafPayloadMode: SparseBrickLeafPayloadMode =
      payloadProfile === "dry" ? octreeLiveSceneLeafPayloadMode() : "dense";
    this.tree = new SparseBrickOctreeGPU(device, {
      brickSize, nodeCapacity, leafCapacity, label: "Octree unified live sparse-brick world",
      payloadProfile, sceneGeometryFormat, leafPayloadMode,
      // Every *reader* of scene identity is banded now. Both dense lanes stay
      // anyway, and for reasons that have nothing to do with the renderer:
      // `encodeBandedLeaves` is the consumer that keeps them alive, because it
      // builds each leaf's palette and record set by reading back the dense
      // identity and geometry words the rebuild pass has just written. Retiring
      // them is a producer change, not a reader cutover — see
      // `SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES`. `tree.payloadProductBytes`
      // remains the arena the same measured scene costs once nothing retains one.
      retainDenseLanes: leafPayloadMode === "banded"
        ? SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES : undefined,
    });
    yield;
    this.brickOccupancyBuilder = new WebGpuSvoBrickOccupancyBuilder(
      device, payloadProfile, this.tree.scenePayloadLanes);
    yield;
    this.topologyMutator = new WebGpuSparseBrickTopologyMutator(device);
    yield;
    this.topologyMutationWorklist = device.createBuffer({
      label: "Live sparse scene topology mutation worklist",
      size: (8 + this.topologyMutationCapacity * 4) * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Both of the next two are for binding 5 alone, which no shipping traversal
    // populates — see `derivedTraversalStructures`. Skipped unless a lane that
    // actually selects compact/wide/hybrid asks for them.
    const derivedTraversalStructures = options.derivedTraversalStructures ?? false;
    this.compactHierarchy = !derivedTraversalStructures || plan.nodes.length === 0 ? undefined : new WebGpuSvoCompactHierarchy(device,
      packSvoCompactHierarchy({ nodes: packed.nodes, leaves: packed.leaves,
        publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length }, 1));
    yield;
    // Renderer traversal snapshots are optional views of the canonical terminal
    // set. A later topology revision rejects them by generation; they never own
    // structure and failure simply omits the capability.
    let wideFanout: WebGPUSvoWideFanout | undefined;
    if (derivedTraversalStructures) try {
      const widePlan = yield* planOctreeSvoWideFanoutSteps(plan);
      wideFanout = new WebGPUSvoWideFanout(device, {
        maximumPages: Math.max(1, widePlan.pages.length),
        maximumDescriptors: Math.max(1, widePlan.descriptorCount),
      });
      const encoder = device.createCommandEncoder({ label: "Publish immutable SVO wide-fanout hierarchy" });
      if (wideFanout.encode(encoder, widePlan) !== "encoded" || !wideFanout.capability()) {
        wideFanout.destroy();
        wideFanout = undefined;
      }
    } catch {
      wideFanout?.destroy();
      wideFanout = undefined;
    }
    this.wideFanout = wideFanout;
    yield;
    const solverOriginBricks = this.solverGridOriginCells.map((value) => value / brickSize);
    if (solverOriginBricks.some((value) => !Number.isInteger(value))) throw new Error("Shared sparse scene origin must align to the fluid brick lattice");
    const leafByCoordinate = new Map<string, number>();
    let indexedLeaves = 0;
    for (const leaf of plan.leaves) {
      const node = plan.nodes[leaf.nodeIndex];
      if (node.level === solverLevel) leafByCoordinate.set(`${leaf.coordinate.x},${leaf.coordinate.y},${leaf.coordinate.z}`, leaf.index);
      if ((indexedLeaves += 1) % 4096 === 0) yield;
    }
    const localBrickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
    const leafIndices = new Uint32Array(localBrickDimensions[0] * localBrickDimensions[1] * localBrickDimensions[2]);
    let mappedBrick = 0;
    for (let z = 0; z < localBrickDimensions[2]; z += 1) for (let y = 0; y < localBrickDimensions[1]; y += 1) for (let x = 0; x < localBrickDimensions[0]; x += 1) {
      const key = `${solverOriginBricks[0] + x},${solverOriginBricks[1] + y},${solverOriginBricks[2] + z}`;
      const leafIndex = leafByCoordinate.get(key);
      // A dry scene publishes no solver bricks at all — whether because it spent
      // its depth on the environment or because it never claimed the container —
      // so there is no leaf to name and nothing that will ever read this table:
      // the fluid frame path is not reached when `systems.fluid` is false.
      // Anywhere the solver *is* present, a missing leaf is still a hard error —
      // that is the statement that solver bricks are never coarsened.
      if (leafIndex === undefined && plannedSolverBricks.length > 0) throw new Error(`Fluid brick ${key} has no finest scene leaf`);
      leafIndices[mappedBrick++] = leafIndex ?? 0;
      if (mappedBrick % 4096 === 0) yield;
    }
    this.residency = new GPUFluidBrickResidency(device, dimensions, sceneDomain.cellSize_m, {
      brickSize, haloCells: options.haloCells ?? 2, retireAfterFrames: 3, leafIndices, leafCapacity: this.tree.leafCapacity,
      topologyTileBricks: options.topologyTileBricks ?? 1,
    });
    yield;
    this.preActivation = options.brickPreActivation ?? true;
    if (options.bulkResidency) {
      // Bulk velocity must remain defined throughout deep liquid, while the
      // surface path wins by visiting only a narrow two-sided band. The two
      // schedulers stay independent so topology support never widens surface
      // redistance back to O(wet volume).
      this.bulkResidency = new GPUFluidBrickResidency(device, dimensions, sceneDomain.cellSize_m, {
        brickSize,
        haloCells: options.haloCells ?? 2,
        retireAfterFrames: 3,
        includeLiquidInterior: true,
        includePressureBoundarySupport: options.includePressureBoundarySupport,
        pressureBoundaryTopClosed: options.pressureBoundaryTopClosed,
        includeWholeDomainPressureSupport: options.includeWholeDomainPressureSupport,
        fluidGatedBoundarySupport: options.fluidGatedBoundarySupport,
        leafIndices,
        leafCapacity: this.tree.leafCapacity,
        topologyTileBricks: options.topologyTileBricks ?? 1,
      });
    }
    yield;

    // No yield until `sourceBuffers` below: these five are locals in between,
    // and a build abandoned here would leave device allocations nothing can
    // reach. The invariant is stated once in `create` and enforced here.
    const counts = storageBuffer(device, "Sparse brick source counts", packed.counts.byteLength, packed.counts);
    const topology = storageBuffer(device, "Sparse brick source topology", packed.topology.byteLength, packed.topology);
    const initialVoxelCount = Math.max(1, plan.voxelCount);
    const geometry = storageBuffer(device, "Sparse brick source geometry", initialVoxelCount * 16);
    const velocity = storageBuffer(device, "Sparse brick source velocity", initialVoxelCount * 16);
    const materialOwners = storageBuffer(device, "Sparse brick source material owners", initialVoxelCount * 4);
    this.sourceBuffers = [counts, topology, geometry, velocity, materialOwners];
    this.source = { counts, topology, geometry, velocity, materialOwners, capacities: {
      nodes: plan.nodes.length, leaves: plan.leaves.length, voxels: plan.voxelCount,
    } };
    yield;

    const pbrMaterials = buildOctreeSvoPbrMaterialPublication(
      OCTREE_SVO_PBR_MATERIAL_REVISION,
      environmentPrimitives,
      this.surfaceModel,
    );
    this.pbrMaterialBuffer = storageBuffer(
      device,
      "Sparse voxel PBR material table",
      OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_STRIDE_BYTES,
      pbrMaterials.packedRecords,
    );
    this.materialEmissionBuffer = storageBuffer(
      device,
      "Live SVO material emission table",
      OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * 16,
      packLiveSvoMaterialEmission(pbrMaterials.packedRecords, pbrMaterials.count),
    );
    const lights = buildOctreeSvoLightPublication(scene);
    this.lightBuffer = storageBuffer(
      device,
      "Sparse voxel authored light table",
      Math.max(SVO_LIGHT_RECORD_STRIDE_BYTES, lights.packedRecords.byteLength),
      lights.packedRecords,
    );
    const environmentLighting = buildOctreeSvoEnvironmentLightingPublication(scene);
    this.environmentLightingBuffer = storageBuffer(
      device,
      "Sparse voxel environment lighting",
      environmentLighting.packedRecords.byteLength,
      environmentLighting.packedRecords,
    );
    yield;
    // The render lattice, which is the solver's divided by the refinement. Every
    // world bound downstream is `coordinate * 2^(maximumDepth - level) * brickSize
    // * cellSize`, so a deeper tree over the same world needs a proportionally
    // finer cell here — a solver leaf at `solverLevel` then measures exactly what
    // it always did. The residency above deliberately keeps the solver's own.
    this.cellSize = renderCellSize;
    this.proxyVoxelizer = new SparseSceneProxyVoxelizer(device, this.tree, {
      cellSize: this.cellSize,
      worldOrigin: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
      finestLevel: plan.maximumDepth,
      primitiveCapacity: OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY,
      dirtyRegionCapacity: OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY,
      dirtyBrickCapacity: this.tree.leafCapacity,
      candidatesPerDirtyBrick: OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
      clusterCapacity: OCTREE_LIVE_SCENE_CLUSTER_CAPACITY,
      fieldProgramCapacity: OCTREE_LIVE_SCENE_FIELD_PROGRAM_CAPACITY,
      solidWorld: initialSolidWorld,
      solidWorldLattice: {
        origin_m: [-0.5 * scene.container.width_m, 0,
          -0.5 * scene.container.depth_m],
        cellSize_m: sceneCellSizes_m(scene),
      },
      // The coarse record index and the per-frame budget. Both are what turn
      // maintenance from "cheap because the scene is small" into something that
      // survives ten times the records: binning stops reading every record, and
      // invalidation stops reading every leaf.
      recordIndex: planOctreeLiveSceneRecordIndex(this.sceneBrickDimensions, this.cellSize, brickSize),
      bricksPerFrameBudget: octreeLiveSceneBrickBudget(),
      label: `${environmentCatalog.environmentId} live scene geometry`,
    });
    yield;
    if (SPARSE_VOXEL_PUBLICATION_STATE.strideBytes !== SPARSE_BRICK_GPU_LAYOUT.publicationStrideBytes) {
      throw new Error("Sparse voxel publication ABI does not match the structural arena");
    }
    this.structuralPublicationState = this.tree.structuralPublication;
    const structuralLayout = device.createBindGroupLayout({
      label: "Sparse voxel structural publication finalizer layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    this.structuralPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [structuralLayout] });
    this.structuralFinalizeBindGroup = device.createBindGroup({
      label: "Sparse voxel structural publication finalizer bindings",
      layout: structuralLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.structuralPublicationState,
            offset: this.tree.structuralPublicationOffsetBytes,
            size: SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
          },
        },
        { binding: 1, resource: { buffer: this.proxyVoxelizer.maintenanceBinding.buffer } },
        { binding: 2, resource: { buffer: this.topologyMutationWorklist } },
      ],
    });
    const publicationBinding = {
      buffer: this.structuralPublicationState,
      offset: this.tree.structuralPublicationOffsetBytes,
      size: SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
    };
    const publicationWord = (word: number) => ({ binding: publicationBinding, word });
    const residencyLayout = sparseVoxelFluidResidencyLayout(this.residency.capacity);
    if (residencyLayout.worklistByteLength !== this.residency.worklistByteLength) {
      throw new Error("Sparse voxel residency ABI does not match the producer worklist allocation");
    }
    yield;
    let nodeMipPyramid: WebGpuLiveSvoNodeMipPyramid | undefined;
    let tetrahedralRadiance: WebGpuLiveSvoTetrahedralRadiance | undefined;
    let liveDerivedPlanner: WebGpuLiveSvoDerivedWorklistPlanner | undefined;
    let liveDerivedBuilder: WebGpuLiveSvoDerivedBuilder | undefined;
    const liveDerivedBasePageDimensions = liveSvoBasePageDimensions(this.sceneBrickDimensions, brickSize);
    const liveDerivedLevelCount = sparseSceneOctreeMaximumDepth(liveDerivedBasePageDimensions, []) + 1;
    const maximumDerivedPages = webGpuSvoNodeMipMaximumPages(device);
    let derivedLighting: NonNullable<SparseVoxelSceneRenderSource["derivedLighting"]> = {
      state: "unavailable",
      reason: "unsupported-level-count",
      detail: `Live SVO derived lighting needs ${liveDerivedLevelCount} mip levels; the runtime supports at most 12`,
      requiredPages: 0,
      capacity: maximumDerivedPages,
    };
    if (liveDerivedLevelCount <= 12) {
      try {
        reportStage("Plan the node-mip pyramid");
        // Where the opacity pyramid's base sits. Anchored to a world size, so
        // the finest opacity texel stays at the reference 6.25 mm however fine
        // the tree is — at the reference leaf this is level 0 and the plan is
        // the one that shipped. See `SVO_OPACITY_LEVEL_FLOOR`.
        const opacityFloorLevel = svoOpacityLevelFloor({
          levelCount: liveDerivedLevelCount,
          cellSize_m: Math.max(...this.cellSize),
        });
        const occupiedPages = yield* liveSvoPlanBasePagesSteps(plan, undefined, opacityFloorLevel);
        // Addresses are planned over the *domain*, not over the first frame's
        // occupancy. A plan that covers every page the domain can hold cannot
        // be invalidated by an edit, which is the whole point: the withdrawal
        // this replaces nulled the pyramid and the radiance atlas silently.
        const addressPlan = yield* planSvoNodeMipAddressesSteps({
          occupiedBasePages: occupiedPages,
          basePageDimensions: liveDerivedBasePageDimensions,
          levelCount: liveDerivedLevelCount,
          addressCapacity: maximumDerivedPages,
          generation: 1,
          // Anchors the radiance floor to a world size, so refining the leaf
          // raises the floor with it and the radiance atlas stops moving.
          cellSize_m: Math.max(...this.cellSize),
          // The seeds above are already floored; the plan carries the level so
          // that a later growth floors the pages an edit activates too.
          opacityFloorLevel,
        });
        this.liveDerivedAddressPlan = addressPlan;
        const mipPlan = addressPlan.plan;
        derivedLighting = {
          state: "unavailable",
          reason: "capacity",
          detail: `Live SVO derived lighting needs ${mipPlan.requestedPageCount} sparse pages; this device can address ${maximumDerivedPages}`,
          requiredPages: mipPlan.requestedPageCount,
          capacity: maximumDerivedPages,
        };
        const direct = createWebGpuSvoNodeMipDirectPageTable(mipPlan, device.limits.maxTextureDimension3D);
        const atlasFits = mipPlan.atlas.texels.every((value) => value <= device.limits.maxTextureDimension3D);
        if (!mipPlan.complete || !direct.ready || !atlasFits || mipPlan.pages.length === 0) {
          throw new RangeError("Live SVO derived-page capacity cannot cover the declared editable domain");
        }
        // Size the direct table for the domain where that is affordable, so a
        // grown plan writes new coordinates into the texture the planner and
        // builder already bind. Where it is not, the plan's own extent is used
        // and a growth that outruns it degrades to the sorted directory — which
        // stays correct, because a re-plan is in the Morton order it searches.
        const domainDirect = svoNodeMipDomainDirectPageTableDimensions(
          liveDerivedBasePageDimensions, liveDerivedLevelCount, opacityFloorLevel);
        const domainDirectFits = domainDirect.every((value) => value <= device.limits.maxTextureDimension3D)
          && domainDirect.reduce((product, value) => product * value, 4) <= WEBGPU_SVO_NODE_MIP_LAYOUT.directPageTableMaximumBytes;
        const directPageTableDimensions = (domainDirectFits
          ? domainDirect.map((value, axis) => Math.max(value, direct.dimensions[axis]))
          : direct.dimensions) as [number, number, number];
        /**
         * The wall. Everything above it in this block — the opacity floor, the
         * base-page seeds, the address plan and the direct page table — is pure
         * CPU and owns nothing, which is why it can be sliced. Everything below
         * creates device resources that stay locals until the four assignments
         * at the end of the block, so a build abandoned past this point would
         * leak an atlas nothing can reach. No yield from here to there.
         */
        yield;
        nodeMipPyramid = new WebGpuLiveSvoNodeMipPyramid(device, {
          pageCapacity: addressPlan.pageCapacity,
          atlasTexels: addressPlan.atlasTexels as [number, number, number],
          directPageTableDimensions,
          // The page follows the payload profile, exactly as the derived
          // builder's lane expansion does: a `dry` tree writes a literal zero
          // into both fluid lanes at every level, so the page that carries them
          // is half constant. See `SVO_NODE_MIP_OPACITY_STORAGE`.
          format: svoNodeMipOpacityFormat({
            dry: this.tree.payloadProfile === "dry",
            features: device.features,
          }),
          label: "Unified live SVO node mips",
        });
        tetrahedralRadiance = new WebGpuLiveSvoTetrahedralRadiance(device, {
          // The slot index space stays the pyramid's — validity, black-page
          // certificates and worklist records all name one slot for both
          // atlases — but the atlas itself only holds the levels at or above
          // the radiance floor, which is where the 512x comes from.
          pageCapacity: addressPlan.pageCapacity,
          atlasTexels: addressPlan.radianceAtlasTexels as [number, number, number],
          atlasPageCapacity: addressPlan.radiancePageCapacity,
          slotOffset: addressPlan.radianceSlotOffset,
          radianceFloorLevel: addressPlan.radianceFloorLevel,
          label: "Unified live SVO radiance",
        });
        const nodeTarget = nodeMipPyramid.prepareGpuUpdate(mipPlan);
        const radianceTarget = tetrahedralRadiance.prepareGpuUpdate(mipPlan, addressPlan.radianceSlotOffset);
        const sceneMaintenance = this.proxyVoxelizer.maintenanceBinding;
        const liveDerivedGenerationSource = {
          buffer: this.structuralPublicationState,
          offsetBytes: this.tree.structuralPublicationOffsetBytes
            + SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration * 4,
        };
        liveDerivedPlanner = new WebGpuLiveSvoDerivedWorklistPlanner(device, {
          tree: this.tree,
          nodeMips: nodeTarget,
          dirtyLeafSources: [
            {
              buffer: sceneMaintenance.buffer,
              countOffsetBytes: sceneMaintenance.stateOffsetBytes
                + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.dirtyBrickCount * 4,
              recordOffsetBytes: sceneMaintenance.dirtyBrickOffsetBytes,
              capacity: sceneMaintenance.dirtyBrickCapacity,
              recordStrideWords: 4,
            },
            {
              buffer: this.residency.worklist,
              countOffsetBytes: SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount * 4,
              recordOffsetBytes: residencyLayout.activeEntryOffsetBytes + 4,
              capacity: this.residency.capacity,
              recordStrideWords: residencyLayout.entryStrideBytes / 4,
            },
            {
              buffer: this.residency.worklist,
              countOffsetBytes: SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount * 4,
              recordOffsetBytes: residencyLayout.retiredEntryOffsetBytes + 4,
              capacity: this.residency.capacity,
              recordStrideWords: residencyLayout.entryStrideBytes / 4,
            },
          ],
          generationSource: liveDerivedGenerationSource,
          levelCount: liveDerivedLevelCount,
          finestLevel: plan.maximumDepth,
          // Every sparse address is retained, but a level can only emit its
          // own resident pages. Avoid sizing all worklists like the hierarchy.
          // Sized from the address plan rather than from today's occupancy,
          // because a grown plan has to fit the worklist it was allocated with.
          // Per level, not the maximum over them: a coarse level's grid is a
          // fraction of the base's and its section is now sized to say so.
          pageCapacityPerLevel: addressPlan.pageCapacityByLevel,
          // The floor level's records carry a page coordinate as well as their
          // children: it is the base of the radiance chain and a parent of the
          // opacity one at the same time.
          radianceFloorLevel: addressPlan.radianceFloorLevel,
          label: "Unified live SVO derived-page planner",
        });
        liveDerivedBuilder = new WebGpuLiveSvoDerivedBuilder(device, {
          tree: this.tree,
          nodeMips: nodeTarget,
          radiance: radianceTarget,
          materialEmission: this.materialEmissionBuffer,
          materialPbr: this.pbrMaterialBuffer,
          environmentLighting: this.environmentLightingBuffer,
          lights: this.lightBuffer,
          lightCount: lights.count,
          worklists: liveDerivedPlanner.worklists,
          generationSource: liveDerivedGenerationSource,
          // Every addressable slot, not just the occupied ones: the empty
          // certification has to reach a reserved slot before a grown plan can
          // hand it to a page, or the first frame after growth reads garbage.
          plannedPageCount: addressPlan.pageCapacity,
          finestLevel: plan.maximumDepth,
          worldOrigin_m: this.sceneWorldOrigin,
          cellSize_m: this.cellSize,
          radianceFeedback: options.radianceFeedback ?? LIVE_SVO_RADIANCE_FEEDBACK.enabledByDefault,
          radianceFloorLevel: addressPlan.radianceFloorLevel,
          // The radiance scratch was sized like the opacity scratch — one 16 kB
          // staging page per level-zero page, as much memory as the atlas it
          // stages into. Above the floor a level holds hundreds of pages.
          radianceScratchCapacity: addressPlan.radiancePageCapacityPerLevel,
          label: "Unified live SVO derived-page builder",
        });
        nodeMipPyramid.acceptGpuUpdate(mipPlan.generation);
        tetrahedralRadiance.acceptGpuUpdate(mipPlan.generation);
        derivedLighting = {
          state: "ready",
          requiredPages: mipPlan.requestedPageCount,
          capacity: maximumDerivedPages,
        };
      } catch (error) {
        liveDerivedBuilder?.destroy();
        liveDerivedPlanner?.destroy();
        nodeMipPyramid?.destroy();
        tetrahedralRadiance?.destroy();
        nodeMipPyramid = undefined;
        tetrahedralRadiance = undefined;
        liveDerivedPlanner = undefined;
        liveDerivedBuilder = undefined;
        if (derivedLighting.requiredPages <= derivedLighting.capacity) {
          derivedLighting = {
            ...derivedLighting,
            reason: "initialization-failed",
            detail: error instanceof Error ? error.message : String(error),
          };
        }
        console.warn("[svo] live derived lighting unavailable; exact visibility will be used", error);
      }
    }
    this.liveDerivedAddressPlanValid = derivedLighting.state === "ready";
    this.nodeMipPyramid = nodeMipPyramid;
    this.tetrahedralRadiance = tetrahedralRadiance;
    this.liveDerivedPlanner = liveDerivedPlanner;
    this.liveDerivedBuilder = liveDerivedBuilder;
    const residencyWorklistBinding = { buffer: this.residency.worklist, size: this.residency.worklistByteLength };
    const residencyWord = (word: number) => ({ binding: residencyWorklistBinding, word });
    const activeResidencyList = {
      count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount),
      entryOffsetBytes: residencyLayout.activeEntryOffsetBytes,
      entryStrideBytes: residencyLayout.entryStrideBytes,
      capacity: this.residency.capacity,
    };
    const structural: SparseVoxelStructuralRenderSource = {
      structure: { buffer: this.tree.structure, size: this.tree.structure.size },
      structureOffsetsWords: {
        control: this.tree.controlOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        publication: this.tree.structuralPublicationOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        nodes: this.tree.nodeOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        leaves: this.tree.leafOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
      },
      control: { buffer: this.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes },
      nodes: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes, size: this.tree.nodeCapacity * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes },
      leaves: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes, size: this.tree.leafCapacity * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes },
      geometry: { buffer: this.tree.geometry, offset: this.tree.geometryOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes },
      // Two channels wide on a `dry` world, four on a `full` one.
      sceneGeometry: { buffer: this.tree.sceneGeometry, offset: this.tree.sceneGeometryOffsetBytes, size: this.tree.sceneGeometryBytes },
      velocity: { buffer: this.tree.velocity, offset: this.tree.velocityOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes },
      materialOwners: { buffer: this.tree.materialOwners, offset: this.tree.materialOwnerOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes },
      // The whole payload arena, not the owner lane's slice. Under `banded` there
      // *is* no owner lane: identity is an occupancy bit, a per-leaf header and a
      // palette entry in four lanes of this one buffer, so a consumer binds the
      // arena once and addresses it through `scenePayloadLanes` — the same shape
      // `structureOffsetsWords` already uses for the structural arena.
      scenePayload: { buffer: this.tree.payload, size: this.tree.payload.size },
      scenePayloadLanes: this.tree.scenePayloadLanes,
      fluidLeafStates: { buffer: this.residency.leafStates, size: this.tree.leafCapacity * Uint32Array.BYTES_PER_ELEMENT },
      fluidResidency: {
        states: { buffer: this.residency.stateBuffer, size: this.residency.capacity * residencyLayout.stateStrideBytes },
        worklist: residencyWorklistBinding,
        domain: {
          originBricks: solverOriginBricks as [number, number, number],
          dimensionsBricks: localBrickDimensions,
        },
        stateStrideBytes: residencyLayout.stateStrideBytes,
        stateBits: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
        active: activeResidencyList,
        core: {
          ...activeResidencyList,
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.coreCount),
          requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core,
        },
        halo: {
          ...activeResidencyList,
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.haloCount),
          requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo,
        },
        retired: {
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount),
          entryOffsetBytes: residencyLayout.retiredEntryOffsetBytes,
          entryStrideBytes: residencyLayout.entryStrideBytes,
          capacity: this.residency.capacity,
        },
        counters: {
          activated: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activatedCount),
        },
        generation: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.generation),
        revision: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
        owner: "GPUFluidBrickResidency",
      },
      capacities: { nodes: this.tree.nodeCapacity, leaves: this.tree.leafCapacity, voxels: this.tree.voxelCapacity },
      strides: {
        control: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes,
        node: SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes,
        leaf: SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes,
        geometry: SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes,
        velocity: SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes,
        materialOwner: SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes,
      },
      domain: {
        worldOrigin_m: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
        cellSize_m: this.cellSize,
        // The *render* lattice, both halves of it. `cellSize_m` is already the
        // refined cell (`this.cellSize` is `renderCellSize`), so publishing the
        // solver's unrefined `sceneDimensionsCells` alongside it described a
        // world 2^depth too small on every axis: at environment refinement
        // depth 1 a consumer baking a replica from this got half the columns
        // and clamped everything past the edge, which is what made
        // `terrain-coverage-solid` judge real surface voxels buried against a
        // clamped edge column (`cell 5,95,250` against an `nz` of 232). The
        // inspection uniform at :1620 already refines by the same factor.
        dimensionsCells: sceneDomain.sceneDimensionsCells.map(
          (cells) => cells * refineScale) as [number, number, number],
        brickSize,
        maximumDepth: plan.maximumDepth,
      },
      publication: {
        state: publicationBinding,
        completeGeneration: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration),
        validFields: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.validFields),
        revisions: {
          topology: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision),
          sceneGeometry: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision),
          dynamicSolid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision),
          coarseFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
          fineFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision),
        },
      },
      fields: {
        topology: { bit: SPARSE_VOXEL_VALID_FIELDS.topology, residency: "all-published-leaves" },
        // Metres under every surviving geometry format. This was a conditional
        // while a cell-band lane existed, and the rule it encoded still holds: units
        // follow the lane's storage format, never the other way round, so any future
        // narrowed lane must report its own units here rather than let a consumer
        // read a scaled value as if it were a distance.
        sceneGeometry: { bit: SPARSE_VOXEL_VALID_FIELDS.sceneGeometry, signedDistance: "negative-inside-metres", distanceQuality: "mixed-exact-approximate", residency: "all-published-leaves" },
        dynamicSolid: { bit: SPARSE_VOXEL_VALID_FIELDS.dynamicSolid, signedDistance: "negative-inside-metres", distanceQuality: "occupancy-estimate", residency: "fluid-resident-leaves" },
        coarseFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.coarseFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric-near-interface", residency: "fluid-resident-leaves" },
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric", residency: "unavailable" },
        velocity: { bit: SPARSE_VOXEL_VALID_FIELDS.velocity, residency: "fluid-resident-leaves" },
        materialOwner: { bit: SPARSE_VOXEL_VALID_FIELDS.materialOwner, residency: "all-published-leaves" },
      },
    };
    this.sceneSource = {
      pbrMaterials: {
        binding: { buffer: this.pbrMaterialBuffer, size: this.pbrMaterialBuffer.size },
        count: pbrMaterials.count,
        strideBytes: pbrMaterials.strideBytes,
        revision: pbrMaterials.revision,
      },
      lights: {
        binding: { buffer: this.lightBuffer, size: lights.packedRecords.byteLength },
        count: lights.count,
        strideBytes: lights.strideBytes,
        revision: lights.revision,
      },
      environmentLighting: {
        binding: { buffer: this.environmentLightingBuffer, size: environmentLighting.packedRecords.byteLength },
        count: environmentLighting.count,
        strideBytes: environmentLighting.strideBytes,
        revision: environmentLighting.revision,
        cacheKey: environmentLighting.cacheKey,
      },
      materialCount: pbrMaterials.count,
      fluidBrickStats: { buffer: this.residency.worklist }, fluidBrickCapacity: this.residency.capacity,
      structural,
      wideFanout: this.wideFanout?.capability(),
      compactHierarchy: this.compactHierarchy?.capability(),
      nodeMipPyramid: this.nodeMipPyramid?.visibleGeneration() && {
        ...this.nodeMipPyramid.visibleGeneration()!,
        worldOrigin_m: this.sceneWorldOrigin,
        worldExtent_m: sceneDomain.sceneDimensionsCells.map((cells, axis) => cells * sceneDomain.cellSize_m[axis]) as [number, number, number],
      },
      tetrahedralRadiance: this.tetrahedralRadiance?.visibleGeneration(),
      derivedLighting,
      derivedRenderAllocationBytes: {
        wideFanout: this.wideFanout?.allocatedBytes ?? 0,
        compactHierarchy: this.compactHierarchy?.allocatedBytes ?? 0,
        nodeMipPyramid: (this.nodeMipPyramid?.allocatedBytes ?? 0)
          + (this.liveDerivedPlanner?.allocatedBytes ?? 0),
        tetrahedralRadiance: (this.tetrahedralRadiance?.allocatedBytes ?? 0)
          + (this.liveDerivedBuilder?.allocatedBytes ?? 0),
      },
      revision: 1
    };
    this.stageSceneUpdate(scene);
    this.baseAllocatedBytes = this.tree.allocatedBytes + this.residency.allocatedBytes
      + (this.bulkResidency?.allocatedBytes ?? 0)
      + this.sourceBuffers.reduce((sum, buffer) => sum + buffer.size, 0)
      + this.pbrMaterialBuffer.size + this.materialEmissionBuffer.size + this.lightBuffer.size + this.environmentLightingBuffer.size
      + this.topologyMutationWorklist.size + this.topologyMutator.allocatedBytes
      + this.proxyVoxelizer.allocatedBytes + (this.wideFanout?.allocatedBytes ?? 0)
      + (this.compactHierarchy?.allocatedBytes ?? 0)
      + (this.nodeMipPyramid?.allocatedBytes ?? 0)
      + (this.tetrahedralRadiance?.allocatedBytes ?? 0)
      + (this.liveDerivedPlanner?.allocatedBytes ?? 0)
      + (this.liveDerivedBuilder?.allocatedBytes ?? 0);
    this.built = true;
  }

  /** Compile this world's structural programs and every owned maintenance pipeline. */
  initializePipelines(): Promise<void> {
    if (this.destroyed) return Promise.reject(new Error("Cannot initialize a destroyed sparse-brick world"));
    this.pipelineInitialization ??= (async () => {
      this.structuralModule = this.device.createShaderModule({
        label: "Sparse voxel structural publication finalizer",
        code: structuralPublicationFinalizeShader(
          this.proxyVoxelizer.maintenanceBinding.stateOffsetBytes / 4),
      });
      const structuralPipelines = Promise.all([
        this.device.createComputePipelineAsync({
          label: "Finalize live sparse voxel physics publication",
          layout: this.structuralPipelineLayout,
          compute: { module: this.structuralModule, entryPoint: "finalizePhysics" },
        }),
        this.device.createComputePipelineAsync({
          label: "Finalize live sparse voxel scene publication",
          layout: this.structuralPipelineLayout,
          compute: { module: this.structuralModule, entryPoint: "finalizeScene" },
        }),
      ] as const);
      const [, compiledStructuralPipelines] = await Promise.all([
        Promise.all([
          this.tree.initializePipelines(),
          this.brickOccupancyBuilder.initializePipelines(),
          this.proxyVoxelizer.initializePipelines(),
          this.topologyMutator.initializePipelines(),
          this.liveDerivedPlanner?.initializePipelines(),
          this.liveDerivedBuilder?.initializePipelines(),
        ]),
        structuralPipelines,
      ]);
      [this.structuralPhysicsPipeline, this.structuralScenePipeline] = compiledStructuralPipelines;
    })().catch((error) => {
      this.pipelineInitialization = undefined;
      throw error;
    });
    return this.pipelineInitialization;
  }

  get allocatedBytes(): number { return this.baseAllocatedBytes; }

  /**
   * Adopt a new metre mapping for the resident render lattice.
   *
   * World scaling deliberately keeps the cell-address lattice intact, so the
   * sparse buffers and their topology remain reusable. The presentation ABI,
   * however, also publishes metres per render cell and the world origin. If
   * those construction-time values survive a re-seed, seeded bricks keep their
   * old size while their authored centres move with the enlarged tank.
   */
  rescaleRenderDomain(scene: SceneDescription): void {
    const structural = this.sceneSource.structural;
    if (!structural) return;
    const dimensions = structural.domain.dimensionsCells;
    const worldOrigin_m: [number, number, number] = [
      -0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m,
    ];
    const cellSize_m: [number, number, number] = [
      scene.container.width_m / dimensions[0],
      scene.container.height_m / dimensions[1],
      scene.container.depth_m / dimensions[2],
    ];
    structural.domain = { ...structural.domain, worldOrigin_m, cellSize_m };
    this.sceneSource.nodeMipPyramid = this.sceneSource.nodeMipPyramid && {
      ...this.sceneSource.nodeMipPyramid,
      worldOrigin_m,
      worldExtent_m: [
        scene.container.width_m, scene.container.height_m, scene.container.depth_m,
      ],
    };
    this.sceneSource.revision += 1;
  }

  /**
   * Stage the newest authored scene as render truth. Multiple editor writes
   * before the next presentation frame coalesce into one publication whose
   * dirty coverage includes every superseded old/new bound.
   */
  stageSceneUpdate(scene: SceneDescription): boolean {
    if (this.destroyed) throw new Error("Cannot update a destroyed sparse-brick world");
    const initialPublication = this.sceneRevision === 0;
    const previousSolidWorld = this.solidWorld;
    const nextSolidWorldStamp = solidWorldContentStamp(scene);
    const solidWorldStampChanged = nextSolidWorldStamp !== this.solidWorldStamp;
    const solidWorldChanged = initialPublication
      || solidWorldStampChanged;
    const nextSolidWorld = solidWorldStampChanged
      ? solidWorldForScene(scene) : previousSolidWorld;
    const previousSolidBounds = initialPublication ? []
      : solidWorldPageBounds(scene, previousSolidWorld);
    const nextSolidBounds = solidWorldChanged
      ? solidWorldPageBounds(scene, nextSolidWorld) : [];
    const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
    const authored = environmentProxyPrimitives(catalog, true);
    const liveEntries = [
      // Only the bodies that cannot move. A dynamic body is solver-owned: the
      // GPU advances its pose every step while the document keeps the authored
      // drop point, so a voxel copy staged from `scene.rigidBodies` freezes at
      // the placement pose the moment the run leaves it — a second, immobile
      // ghost beside the analytic body every lighting path already accounts
      // for (see `visibleOwnership: "analytic-rigid-body"` in the coverage
      // contract, and the analytic rigid-blocker corrections in the dry-scene
      // shading). Static bodies have no live pose to diverge from, and the
      // cone-traced lighting reads their voxels — the garden's stepping stones
      // are the shipped case.
      //
      // `flatMap` rather than filter-then-map: `ownerId` is the body's roster
      // index, and the analytic side (`rigidMotion[hit.ownerId]`) resolves the
      // same index, so a filtered body must not shift its neighbours' ids.
      ...scene.rigidBodies.flatMap((body, ownerId) => body.motion === "static" ? [{
        key: `rigid:${body.id}`,
        primitive: sparseScenePrimitiveForRigidBody(body, ownerId),
        materialSignature: body.shape,
      }] : []),
      ...authored.map((primitive) => ({
        key: primitive.key,
        primitive: sparseScenePrimitiveForProxy(primitive, {
          materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
          ownerId: SCENE_ENVIRONMENT_OWNER_BASE + primitive.ownerIndex,
        }),
        materialSignature: JSON.stringify(primitive.material),
      })),
    ];
    if (liveEntries.length > OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY) {
      throw new RangeError(`Live scene needs ${liveEntries.length} primitives but the fixed arena holds ${OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY}`);
    }
    const nextPrimitives = new Map<string, LiveScenePrimitiveEntry>();
    const nextStates = new Map<string, LiveScenePrimitiveState>();
    const dirtyRegions: SparseSceneAxisAlignedBounds[] = [
      ...(this.pendingScenePublication?.dirtyRegions ?? []),
      ...(solidWorldChanged ? previousSolidBounds : []),
      ...nextSolidBounds,
    ];
    const newBounds: SparseSceneAxisAlignedBounds[] = [...nextSolidBounds];
    for (const entry of liveEntries) {
      const live = entry.primitive;
      nextPrimitives.set(entry.key, entry);
      const state = { signature: `${liveScenePrimitiveSignature(live)}:${entry.materialSignature}`, bounds: sparseScenePrimitiveBounds(live) };
      nextStates.set(entry.key, state);
      const previous = this.liveScenePrimitiveStates.get(entry.key);
      if (!previous || previous.signature !== state.signature) {
        if (previous) dirtyRegions.push(previous.bounds);
        dirtyRegions.push(state.bounds); newBounds.push(state.bounds);
      }
    }
    for (const [key, previous] of this.liveScenePrimitiveStates) {
      if (!nextStates.has(key)) dirtyRegions.push(previous.bounds);
    }
    if (dirtyRegions.length === 0) return false;
    if (solidWorldChanged && !initialPublication) {
      this.proxyVoxelizer.setSolidWorld(nextSolidWorld);
    }
    this.solidWorld = nextSolidWorld;
    this.solidWorldStamp = nextSolidWorldStamp;
    this.liveScenePrimitiveStates = nextStates;
    this.liveScenePrimitives = nextPrimitives;
    const revision = this.advanceSceneRevision();
    const pbrMaterials = buildOctreeSvoPbrMaterialPublication(
      OCTREE_SVO_PBR_MATERIAL_REVISION + revision,
      authored,
      this.surfaceModel,
    );
    if (pbrMaterials.count > OCTREE_LIVE_SCENE_MATERIAL_CAPACITY) {
      throw new RangeError(`Live scene needs ${pbrMaterials.count} materials but the fixed arena holds ${OCTREE_LIVE_SCENE_MATERIAL_CAPACITY}`);
    }
    this.device.queue.writeBuffer(this.pbrMaterialBuffer, 0, pbrMaterials.packedRecords);
    this.device.queue.writeBuffer(this.materialEmissionBuffer, 0,
      packLiveSvoMaterialEmission(pbrMaterials.packedRecords, pbrMaterials.count));
    const publishedSource = (this as unknown as { sceneSource?: SparseVoxelSceneRenderSource }).sceneSource;
    if (publishedSource) {
      publishedSource.pbrMaterials = {
        binding: { buffer: this.pbrMaterialBuffer, size: this.pbrMaterialBuffer.size },
        count: pbrMaterials.count,
        strideBytes: pbrMaterials.strideBytes,
        revision: pbrMaterials.revision,
      };
      publishedSource.materialCount = pbrMaterials.count;
    }
    this.stagePrimitivePublication(revision, dirtyRegions, newBounds, initialPublication);
    return true;
  }

  /** Allocation-free keyed motion/content updates shared by renderer-only and fluid worlds. */
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]): boolean {
    if (this.destroyed) throw new Error("Cannot update a destroyed sparse-brick world");
    if (updates.length === 0) return false;
    const dirtyRegions: SparseSceneAxisAlignedBounds[] = [...(this.pendingScenePublication?.dirtyRegions ?? [])];
    const newBounds: SparseSceneAxisAlignedBounds[] = [];
    let changed = false;
    const seen = new Set<string>();
    for (const update of updates) {
      if (!update.key || seen.has(update.key)) throw new RangeError("Live primitive update keys must be unique and nonempty");
      seen.add(update.key);
      const previousEntry = this.liveScenePrimitives.get(update.key);
      const previousState = this.liveScenePrimitiveStates.get(update.key);
      if (!previousEntry || !previousState) throw new RangeError(`Unknown live primitive key ${update.key}`);
      const signature = `${liveScenePrimitiveSignature(update.primitive)}:${previousEntry.materialSignature}`;
      if (signature === previousState.signature) continue;
      const bounds = sparseScenePrimitiveBounds(update.primitive);
      dirtyRegions.push(previousState.bounds, bounds); newBounds.push(bounds); changed = true;
      this.liveScenePrimitives.set(update.key, { ...previousEntry, primitive: update.primitive });
      this.liveScenePrimitiveStates.set(update.key, { signature, bounds });
    }
    if (!changed) return false;
    const revision = this.advanceSceneRevision();
    this.stagePrimitivePublication(revision, dirtyRegions, newBounds, false);
    return true;
  }

  private advanceSceneRevision(): number {
    this.sceneRevision = this.sceneRevision === 0xffff_ffff ? 1 : this.sceneRevision + 1;
    return this.sceneRevision;
  }

  private stagePrimitivePublication(
    revision: number,
    dirtyRegions: readonly SparseSceneAxisAlignedBounds[],
    newBounds: readonly SparseSceneAxisAlignedBounds[],
    initialPublication: boolean,
  ): void {
    const coalescedDirty = coalesceDirtyRegions(dirtyRegions, initialPublication ? 1 : OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY);
    this.pendingScenePublication = {
      primitives: [...this.liveScenePrimitives.values()].map(({ primitive }) => primitive),
      dirtyRegions: coalescedDirty,
      revision,
      // The publication that brings a scene into existence is never budgeted.
      // Everything downstream reads a scene generation, and that generation only
      // advances when a revision completes; spreading the first one over frames
      // would leave every consumer reading zero, which is the state a black slab
      // across the container footprint is made of. Edits are a different case:
      // there is already a current generation to keep showing while the next
      // converges.
      budgeted: !initialPublication,
    };
    if (!initialPublication) {
      const activatedPages: SvoNodeMipCoordinate[] = [];
      for (const coordinate of liveSceneMissingBrickCoordinates(
        newBounds, this.sceneWorldOrigin, this.cellSize, this.brickSize,
        this.sceneBrickDimensions, (coordinate) => this.sceneBrickCovered(coordinate),
      )) {
        this.pendingTopologyCoordinates.set(brickCoordinateKey(coordinate), coordinate);
        activatedPages.push([coordinate.x, coordinate.y, coordinate.z]
          .map((value) => Math.floor(value * this.brickSize / SVO_NODE_MIP_LAYOUT.interiorSize)) as unknown as SvoNodeMipCoordinate);
      }
      this.growLiveDerivedAddressPlan(activatedPages);
    }
    if (this.pendingTopologyCoordinates.size === 0) {
      this.device.queue.writeBuffer(this.topologyMutationWorklist, 0, new Uint32Array(8));
      this.pendingTopologyMutation = false;
      return;
    }
    const requested = [...this.pendingTopologyCoordinates.values()];
    const packed = packSparseBrickTopologyMutationWorklist(
      requested.slice(0, this.topologyMutationCapacity), revision, this.topologyMutationCapacity,
    );
    packed[0] = requested.length;
    this.device.queue.writeBuffer(this.topologyMutationWorklist, 0, packed);
    this.pendingTopologyMutation = true;
    // Only a topology-changing transaction invalidates topology-keyed wide,
    // compact, and renderer caches. Covered primitive motion leaves it stable.
    this.sceneSource.revision = revision;
  }

  /**
   * Extend the derived address plan to cover pages this edit activated.
   *
   * The predecessor of this method set `liveDerivedAddressPlanValid = false`
   * and nulled the pyramid and the radiance atlas, which drops cone visibility
   * onto exact traversal at roughly 15x the frame cost, and reported it only in
   * a status string. Incremental voxelization activates pages continuously, so
   * that path fires by construction — it had to stop being the answer.
   *
   * A total plan (the ordinary authored room, including the hero garden) never
   * reaches the growth branch: every page of the domain already has a slot.
   * A reserved plan re-plans into its reserve, which renumbers physical slots
   * and therefore costs one full rebuild — hence `liveDerivedInitial`.
   *
   * Withdrawal survives only as the last resort, and now says so out loud.
   */
  /**
   * Whether some leaf already covers this finest brick.
   *
   * Ancestor lookup rather than membership, because coverage is stored at each
   * leaf's own level: a brick under a coarse leaf is covered by that leaf, and
   * nothing had to enumerate the `8^p` bricks between them.
   */
  private sceneBrickCovered(coordinate: SparseBrickCoordinate): boolean {
    for (let level = this.finestLevel; level >= 0; level -= 1) {
      const shift = this.finestLevel - level;
      const key = `${level}:${coordinate.x >>> shift},${coordinate.y >>> shift},${coordinate.z >>> shift}`;
      if (this.coveredSceneBrickNodes.has(key)) return true;
    }
    return false;
  }

  private growLiveDerivedAddressPlan(activatedPages: readonly SvoNodeMipCoordinate[]): void {
    if (activatedPages.length === 0 || !this.liveDerivedAddressPlanValid) return;
    const current = this.liveDerivedAddressPlan;
    if (!current || !this.nodeMipPyramid || !this.tetrahedralRadiance || !this.liveDerivedPlanner) return;
    const outside = pagesOutsideSvoNodeMipAddressPlan(current, activatedPages);
    if (outside.length === 0) return;
    const grown = growSvoNodeMipAddressPlan(current, outside);
    if (grown) {
      this.liveDerivedAddressPlan = grown.plan;
      this.nodeMipPyramid.prepareGpuUpdate(grown.plan.plan);
      this.tetrahedralRadiance.prepareGpuUpdate(grown.plan.plan, grown.plan.radianceSlotOffset);
      this.nodeMipPyramid.acceptGpuUpdate(grown.plan.plan.generation);
      this.tetrahedralRadiance.acceptGpuUpdate(grown.plan.plan.generation);
      this.liveDerivedPlanner.configurePlan(this.nodeMipPyramid.gpuTarget());
      // Renumbered slots move where the radiance atlas begins; the builder holds
      // that offset in a uniform written once at construction.
      this.liveDerivedBuilder?.configureRadianceSlotOffset(grown.plan.radianceSlotOffset);
      // Slots renumbered, so every page's atlas content now belongs to some
      // other page. The next maintenance certifies the whole capacity empty and
      // rebuilds from every live leaf, which is the same work the first
      // publication does and is bounded by the same worklists.
      if (grown.rebuildRequired) this.liveDerivedInitial = true;
      return;
    }
    this.liveDerivedAddressPlanValid = false;
    const detail = `Scene edit activated derived page ${outside[0].join(",")} outside the sparse address plan's`
      + ` ${current.pageCapacity}-page capacity (${current.reservePages} reserved); exact visibility is active`;
    this.sceneSource.derivedLighting = {
      state: "unavailable",
      reason: "address-plan-invalidated",
      detail,
      requiredPages: current.domainPyramidPageCount,
      capacity: this.sceneSource.derivedLighting?.capacity ?? webGpuSvoNodeMipMaximumPages(this.device),
    };
    this.sceneSource.nodeMipPyramid = undefined;
    this.sceneSource.tetrahedralRadiance = undefined;
    // The cheapest withdrawal path used to have no console output at all. It is
    // a ~15x frame-time cliff; it gets the same warning the other one has.
    console.warn(`[svo] live derived address plan exhausted; exact visibility will be used — ${detail}`);
  }

  /**
   * The voxelizer's maintenance arena, so a lane can read the completion
   * contract off the device rather than infer it.
   *
   * `completedRevision != requestedRevision` is the whole statement of
   * "this publication is not observable as current"; a gate that asserts a
   * budget without reading those two words is asserting its own arithmetic.
   */
  get sceneMaintenanceBinding(): SparseSceneMaintenanceBinding { return this.proxyVoxelizer.maintenanceBinding; }

  /** Per-frame voxelization budget state, for the lane that proves a teleport converges. */
  get sceneVoxelizationBudget(): { budget: number; plannedChunks: number; chunksRemaining: number; converging: boolean } {
    return this.proxyVoxelizer.budgetStatus;
  }

  /** What the last publication asked of the coarse record index. */
  get sceneVoxelizationIndex(): {
    enabled: boolean; indexedInvalidation: boolean; regionCells: number; gridItems: number; globalRecords: number;
  } {
    return this.proxyVoxelizer.indexStatus;
  }

  /** Keys of the live primitives this world is currently voxelizing. */
  get liveScenePrimitiveKeys(): readonly string[] { return [...this.liveScenePrimitives.keys()]; }

  /** The live primitive a key names, so an edit lane can move one without reauthoring the scene. */
  liveScenePrimitive(key: string): SparseScenePrimitive | undefined {
    return this.liveScenePrimitives.get(key)?.primitive;
  }

  /** Address-plan shape, for lanes that have to prove the pyramid never withdrew. */
  get liveDerivedAddressPlanStatus(): {
    valid: boolean;
    total: boolean;
    pageCapacity: number;
    activePages: number;
    reservePages: number;
    domainPyramidPages: number;
    basePages: number;
    domainBasePages: number;
  } | undefined {
    const plan = this.liveDerivedAddressPlan;
    if (!plan) return undefined;
    return {
      valid: this.liveDerivedAddressPlanValid,
      total: plan.total,
      pageCapacity: plan.pageCapacity,
      activePages: plan.plan.pages.length,
      reservePages: plan.reservePages,
      domainPyramidPages: plan.domainPyramidPageCount,
      basePages: plan.basePageKeys.size,
      domainBasePages: plan.domainBasePageCount,
    };
  }

  /**
   * Encode bounded maintenance for the latest staged scene revision.
   *
   * This is intentionally callable every presentation frame, including while
   * physics is paused. The scene is always authoritative; topology and voxel
   * payloads are reusable accelerators whose completion generation advances
   * only after the maintenance passes finish.
   */
  encodeSceneMaintenance(
    encoder: GPUCommandEncoder,
    deferDerived = false,
    seam?: RenderFrameSeam<"world">,
  ): boolean {
    if (this.destroyed) return false;
    let encoded = false;
    if (!this.topologyPublished) {
      this.tree.encodePublish(encoder, this.source);
      this.topologyPublished = true;
      encoded = true;
    }
    if (this.pendingTopologyMutation) {
      this.topologyMutator.encode(encoder, this.tree, {
        buffer: this.topologyMutationWorklist,
        capacity: this.topologyMutationCapacity,
      }, {
        maximumDepth: this.finestLevel,
        brickDimensions: this.sceneBrickDimensions,
        generation: this.sceneRevision,
        maximumRequests: this.topologyMutationCapacity,
      });
      if (this.pendingTopologyCoordinates.size <= this.topologyMutationCapacity) {
        // Mutation only ever adds *finest* bricks, so these enter at the finest
        // level whatever level the leaves around them sit at.
        for (const [key] of this.pendingTopologyCoordinates) {
          this.coveredSceneBrickNodes.add(`${this.finestLevel}:${key}`);
        }
      }
      this.pendingTopologyCoordinates.clear();
      this.pendingTopologyMutation = false;
      encoded = true;
    }
    // Publish and mutate are the topology half: they change which bricks exist.
    // Everything after this seam changes what is *in* them.
    seam?.("world-topology-publish");
    if (this.pendingScenePublication) {
      this.proxyVoxelizer.publish(this.pendingScenePublication);
      this.pendingScenePublication = undefined;
    }
    encoded = this.proxyVoxelizer.encodeMaintenance(encoder) || encoded;
    // Diffuse feedback continues for a bounded convergence window after the
    // last source revision. A static room normally reaches this branch after
    // its first publication; stopping immediately freezes phase 0 and exposes
    // its quarter-leaf pattern, while running forever makes a complex hero
    // scene visibly breathe and spends GPU time after convergence.
    if (!encoded) {
      // The settled path, and the one the render panel spends most of its life
      // reporting: nothing was voxelized this frame. Both remaining seams still
      // close, over nothing, so the manifest can say "reached, encoded no pass"
      // — which is a zero, and is what stops a silent row from inheriting the
      // source band's whole fence-partitioned wall.
      seam?.("world-proxy-voxelize");
      seam?.("world-derived-lighting");
      const fed = !deferDerived && this.encodeLiveRadianceFeedback(encoder);
      seam?.("world-radiance-feedback");
      return fed;
    }
    const broker = new PassBroker(encoder);
    const finalizer = broker.compute({ label: "Finalize live sparse voxel scene publication" });
    finalizer.setPipeline(this.structuralScenePipeline);
    finalizer.setBindGroup(0, this.structuralFinalizeBindGroup);
    finalizer.dispatchWorkgroups(1);
    broker.fence("live sparse scene publication finalized");
    seam?.("world-proxy-voxelize");
    if (deferDerived) {
      seam?.("world-derived-lighting");
      seam?.("world-radiance-feedback");
    } else this.encodeLiveDerivedMaintenance(encoder, seam);
    return true;
  }

  private encodeLiveDerivedMaintenance(
    encoder: GPUCommandEncoder,
    seam?: RenderFrameSeam<"world">,
  ): void {
    if (!this.liveDerivedAddressPlanValid || !this.liveDerivedPlanner || !this.liveDerivedBuilder) {
      seam?.("world-derived-lighting");
      seam?.("world-radiance-feedback");
      return;
    }
    const initializeEmpty = this.liveDerivedInitial;
    if (initializeEmpty) this.liveDerivedPlanner.encodeInitial(encoder);
    else this.liveDerivedPlanner.encode(encoder);
    this.liveDerivedBuilder.encode(encoder, initializeEmpty);
    this.liveDerivedFeedbackFramesRemaining = this.liveDerivedBuilder.radianceFeedbackEnabled
      ? LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount : 0;
    seam?.("world-derived-lighting");
    this.encodeLiveRadianceFeedback(encoder);
    seam?.("world-radiance-feedback");
    this.liveDerivedInitial = false;
    const nodeMip = this.nodeMipPyramid?.visibleGeneration();
    const radiance = this.tetrahedralRadiance?.visibleGeneration();
    this.sceneSource.nodeMipPyramid = nodeMip && {
      ...nodeMip,
      worldOrigin_m: this.sceneWorldOrigin,
      worldExtent_m: this.sceneBrickDimensions.map((bricks, axis) => bricks * this.brickSize * this.cellSize[axis]) as [number, number, number],
    };
    this.sceneSource.tetrahedralRadiance = radiance;
  }

  private encodeLiveRadianceFeedback(encoder: GPUCommandEncoder): boolean {
    if (this.liveDerivedFeedbackFramesRemaining <= 0 || !this.liveDerivedAddressPlanValid
      || !this.liveDerivedPlanner || !this.liveDerivedBuilder?.radianceFeedbackEnabled) {
      return false;
    }
    this.liveDerivedPlanner.encodeRadianceFeedback(
      encoder,
      this.liveDerivedFeedbackPhase,
      LIVE_SVO_RADIANCE_FEEDBACK.phaseCount,
    );
    this.liveDerivedBuilder.encodeRadianceFeedback(encoder);
    this.liveDerivedFeedbackPhase = (this.liveDerivedFeedbackPhase + 1) % LIVE_SVO_RADIANCE_FEEDBACK.phaseCount;
    this.liveDerivedFeedbackFramesRemaining -= 1;
    return true;
  }

  encode(encoder: GPUCommandEncoder, fields: OctreeSparseBrickDenseFields, dt_s = 0): void {
    if (this.destroyed) return;
    this.residency.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    this.bulkResidency?.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    this.encodeSceneMaintenance(encoder, true);
    this.tree.encodeFromDenseFields(encoder, {
      levelSet: fields.levelSet.createView(), velocity: fields.velocity.createView(), solidCells: fields.solidCells,
      dimensions: this.dimensions,
      cellSize: this.cellSize,
      fluidMaterialId: VOXEL_MATERIAL_IDS.fluid,
      solidMaterialId: VOXEL_MATERIAL_IDS.terrain,
      gridOriginCells: this.solverGridOriginCells,
      finestLevel: this.finestLevel,
      activeBrickWorklist: this.residency.worklist,
    });
    // One conservative terminal summary covers both independently-owned
    // payload lanes; scene and physics writes never overwrite one another.
    this.brickOccupancyBuilder.encodeFluidResidency(encoder, this.tree, this.residency.worklist);
    const finalizerBroker = new PassBroker(encoder);
    const finalizer = finalizerBroker.compute({ label: "Finalize live sparse voxel physics publication" });
    finalizer.setPipeline(this.structuralPhysicsPipeline);
    finalizer.setBindGroup(0, this.structuralFinalizeBindGroup);
    finalizer.dispatchWorkgroups(1);
    finalizerBroker.fence("live sparse physics publication finalized");
    this.encodeLiveDerivedMaintenance(encoder);
  }

  readResidencyStats(): Promise<FluidBrickResidencyStats> { return this.residency.readStats(); }

  readBulkResidencyStats(): Promise<FluidBrickResidencyStats> | undefined { return this.bulkResidency?.readStats(); }

  /** Full wet-domain worklist, independent of the narrow surface band. */
  get bulkResidencyWorklist(): GPUBuffer | undefined { return this.bulkResidency?.worklist; }

  /** Persistent wet-domain topology scheduler on compact authority. */
  get topologyResidency(): GPUFluidBrickResidency { return this.bulkResidency ?? this.residency; }

  /**
   * Release everything this world owns, whether or not it finished being built.
   *
   * The optional chaining is the whole point: a build superseded mid-way is
   * destroyed from wherever it was suspended, and at every one of `buildSteps`'
   * yield points some suffix of these fields is still unassigned. Reading them
   * unguarded is what would turn an abandoned build into a leaked arena — at
   * refinement depth 3 that is gigabytes of device memory the *replacement*
   * build then has to allocate around. `built` is not consulted here on
   * purpose: a partial world and a whole one are released the same way, and the
   * only difference is how far down this list there is something to release.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.deferredBuild?.return();
    this.deferredBuild = undefined;
    this.tree?.destroy();
    this.residency?.destroy();
    this.bulkResidency?.destroy();
    this.proxyVoxelizer?.destroy();
    this.topologyMutator?.destroy();
    this.topologyMutationWorklist?.destroy();
    this.wideFanout?.destroy();
    this.compactHierarchy?.destroy();
    this.liveDerivedBuilder?.destroy();
    this.liveDerivedPlanner?.destroy();
    this.nodeMipPyramid?.destroy();
    this.tetrahedralRadiance?.destroy();
    for (const buffer of [...(this.sourceBuffers ?? []), this.pbrMaterialBuffer, this.materialEmissionBuffer, this.lightBuffer, this.environmentLightingBuffer]) buffer?.destroy();
  }
}

export const octreeSparseBrickStructuralFinalizeShader = structuralPublicationFinalizeShader();
