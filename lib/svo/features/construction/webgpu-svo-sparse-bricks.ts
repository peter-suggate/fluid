import { assertDualGridAttachmentFits } from "../meshing/dual-grid-capacity";
import { TerrainEditPages, terrainEditSurfacePages, terrainFieldStamp } from "../../../core/live-terrain-overlay";
import type { RigidBodyDescription, SceneDescription } from "../../../core/model";
import type { RenderFrameSeam } from "../../../core/render-frame-stages";
import { svoSceneLighting } from "../lighting-visibility/svo-dry-scene-lighting";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "../scene-publication/svo-publication-cache";
import {
  buildSvoEnvironmentLighting,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  type SvoEnvironmentLightingRecord,
} from "../radiance/svo-environment-lighting";
import {
  buildSvoSceneLights,
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  type SvoLightRecord,
} from "../../contracts/svo-light-abi";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  SVO_MATERIAL_RECORD_STRIDE_BYTES,
  SVO_MATERIAL_RECORD_WORDS,
  svoMaterialFunctionIdForEnvironmentProxy,
  svoMaterialFromEnvironmentProxyMaterial,
  writeSvoEnvironmentMaterialRecord,
  writeSvoMaterialFunctionId,
  writeSvoMaterialRecord,
} from "../../contracts/svo-material-abi";
import { successorEnvironmentCatalog } from "../scene-publication/scene-environment-catalog";
import { uploadChangedWords } from "../scene-publication/svo-table-upload";
import { sceneTerrainSurfaceModel, type SvoTerrainSurfaceModel } from "../materials/svo-terrain-material";
import { sceneCellSizes_m } from "../../../core/scene-lattice";
import { solidWorldChangeBounds, type SolidWorldChangeBounds } from "../../../core/solid-world-change-bounds";
import {
  SOLID_WORLD_BRICK_CELLS,
  SOLID_WORLD_TERRAIN_MATERIAL_ID,
  solidWorldContentStamp,
  solidWorldForScene,
  solidWorldPageCoordinates,
  type SolidWorld,
} from "../../../core/solid-world";
import {
  SparseBrickOctreeGPU,
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_SCENE_GEOMETRY_FORMATS,
  SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES,
  SPARSE_BRICK_LEAF_TERMINAL,
  octreeLiveSceneLeafPayloadMode,
  octreeLiveSceneSceneGeometryFormat,
  packSparseBrickPlan,
  resolveSparseBrickPayloadLayout,
  sparseBrickPayloadProfileForOwnership,
  type SparseBrickCoordinate,
  type SparseBrickPlan,
  type SparseBrickPayloadProfileName,
  type SparseBrickPublicationSource,
  type SparseBrickSceneGeometryFormat,
  type SparseBrickLeafPayloadMode,
  type SparseBrickSize
} from "./sparse-brick-octree";
import { liveTopologyMutationRecords, planLiveTopologyLeafReserve } from "./svo-live-topology-reserve";
import { WEBGPU_SOLID_WORLD_DISPLAY_BUDGET_BYTES } from "../../../core/webgpu-solid-world-pages";
import { planAdaptiveSparseBrickOctreeSteps } from "../../../core/adaptive-sparse-brick-plan";
import {
  createSvoEnvironmentRefinement,
  svoEnvironmentRefinementMode,
} from "./svo-environment-refinement";
import {
  createSvoEnvironmentCoarsening,
  environmentProxyFeatureSize_m,
  SVO_ENVIRONMENT_FEATURE_VOXELS,
  solidWorldTerrainSurfaceCoarseningRegions,
  solidWorldVoxelPatchCoarseningRegions,
  svoEnvironmentCoarseningPower,
} from "./svo-environment-coarsening";
import { svoPrimitiveForRigidBody, type SvoPrimitiveDescriptor } from "../../contracts/svo-primitive-abi";
import {
  buildSvoRenderTerrainFieldSteps,
  createSvoRenderEditPageRefinement,
  createSvoRenderTerrainRefinement,
} from "../scene-publication/svo-render-solid-field";
import { planSvoWideFanoutSteps } from "./svo-wide-fanout";
import { WebGPUSvoWideFanout } from "./webgpu-svo-wide-fanout";
import { packSvoCompactHierarchy } from "./svo-compact-hierarchy";
import { WebGpuSvoCompactHierarchy } from "./webgpu-svo-compact-hierarchy";
import {
  SVO_NODE_MIP_LAYOUT,
  svoNodeMipOpacityFormat,
  svoOpacityLevelFloor,
  type SvoNodeMipCoordinate,
} from "../radiance/svo-node-mip-pyramid";
import {
  SVO_NODE_MIP_ADDRESS_PLAN,
  planSvoNodeMipAddressesSteps,
  svoNodeMipLevelPageDimensions,
  type SvoNodeMipAddressPlan,
  type SvoNodeMipAddressPlanGrowth,
} from "../radiance/svo-node-mip-address-plan";
import {
  WebGpuLiveSvoNodeMipPyramid,
  invalidateLiveSvoDerivedPage,
  webGpuSvoNodeMipMaximumPages,
} from "../radiance/webgpu-svo-node-mip-pyramid";
import { SVO_NODE_MIP_MAXIMUM_LEVELS, svoNodeMipPageHashShape } from "../radiance/svo-node-mip-page-hash";
import { WebGpuLiveSvoTetrahedralRadiance } from "../radiance/webgpu-svo-tetrahedral-radiance";
import {
  LIVE_SVO_RADIANCE_FEEDBACK,
  WebGpuLiveSvoDerivedBuilder,
  WebGpuLiveSvoDerivedWorklistPlanner,
  liveSvoPlanBasePagesSteps,
} from "../radiance/webgpu-svo-live-derived-builder";
import { WebGpuSvoBrickOccupancyBuilder } from "./webgpu-svo-brick-occupancy";
import {
  SPARSE_BRICK_TOPOLOGY_MUTATION,
  SparseBrickTopologyMirror,
  WebGpuSparseBrickTopologyMutator,
  compareSparseBrickMorton,
  sparseBrickTopologyMutationNodeReserve,
} from "../../../core/webgpu-sparse-brick-topology-mutation";
import { assertSvoBrickRasterNodeAddressable } from "../primary-visibility/webgpu-svo-brick-raster";
import { planSparseSceneDomain } from "../../../core/sparse-scene-domain";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../../../core/webgpu-rigid-body";
import { BATH_SOLID_DISTANCE_WGSL } from "../../../core/voxel-bath";
import { sweptFlumeSurfaceWGSL } from "../../../core/swept-flume-surface";
import { VOXEL_MATERIAL_IDS, materialIdForRigidShape, packVoxelDebugMaterialTable } from "../../../core/voxel-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives, type EnvironmentProxyCatalog, type EnvironmentProxyPrimitive } from "../../../core/voxel-environments";
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
} from "../../../core/webgpu-sparse-scene-proxies";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelSceneRenderSource,
  type SparseVoxelStructuralRenderSource,
} from "../../../core/webgpu-voxel-debug";
import { GPUFluidBrickResidency, type FluidBrickResidencyStats } from "../../../core/webgpu-fluid-brick-residency";
import {
  completeCooperativeBuild,
  driveCooperativeBuild,
  type CooperativeBuildOptions,
} from "../../../core/cooperative-build";
import { PassBroker } from "../../../core/webgpu-pass-broker";
import { backdropSeamForScene, compileBackdropField } from "../backdrop/backdrop-field";
import { BACKDROP_CONTENT_CEILING_HEADER_WORDS, BACKDROP_CONTENT_CEILING_TILE_CELLS, BACKDROP_CONTENT_CEILING_TILES, BACKDROP_TERRAIN_HEADER_WORDS, BACKDROP_TERRAIN_TABLE, createBackdropContentCeiling, packBackdropTerrainTable, packContentCeilingTable, planBackdropTileLayout, planBackdropTilesSteps, raiseBackdropContentCeiling, raiseBackdropContentCeilingOver, type BackdropContentCeiling } from "../backdrop/backdrop-terrain-tiles";
import {
  BACKDROP_DETAIL_DUAL_APRON_BRICKS,
  backdropDetailCentreLattice,
  backdropDetailFromPlan,
  backdropDetailVoxelizerWGSL,
  backdropDetailWorldBounds,
  createBackdropDetailClassifier,
} from "../backdrop/backdrop-detail";
import {
  packPlanarBoundaryPatches,
  PLANAR_BOUNDARY_PATCH_BYTES,
} from "../../../core/planar-boundary";
import {
  buildSvoPlanarBoundaryCatalog,
  buildSvoSolidWorldPlanarBoundaryCatalog,
  createSvoPlanarLeafClassifier,
  svoPlanarResidualEnvironmentPrimitives,
  svoPlanarSolidWorldBlockers,
  svoRenderResidual,
  type SvoRenderResidual,
} from "../scene-publication/svo-planar-boundary";

export interface OctreeSparseBrickWorldOptions {
  surfaceContours?: boolean;
  surfaceDualContouring?: boolean;
  surfaceDualMarchingCubes?: boolean;
  buildRenderTerrainGpu?: (cellSize: readonly [number,number,number], materialId: number) => Promise<import("../scene-publication/svo-render-solid-field").SvoRenderTerrainField | undefined>;
  classifyEnvironmentNodesGpu?: (input: import("./webgpu-svo-node-classification").SvoNodeClassificationInput) => Promise<import("../../../core/adaptive-sparse-brick-plan").SparseBrickEnvironmentClassification[]>;
  /** Async renderer startup selection; the synchronous constructor retains the CPU oracle. */
  selectPrimitiveBricksGpu?: (input: import("../primary-visibility/webgpu-svo-brick-selection").SvoGpuBrickSelectionInput) => Promise<import("../../../core/adaptive-sparse-brick-plan").SparseBrickProxyOccupancy>;
  brickSize?: SparseBrickSize;
  /**
   * This world publishes authored scene geometry only; no fluid solver writes
   * its payload lanes. Used by the renderer sidecar for fluid-enabled scenes.
   */
  rendererOnly?: boolean;
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
   * How many of the backdrop's stored detail rings, innermost first, take one
   * level of `environmentRefinementDepth` (`BackdropDetail.refinedRings`).
   * Clamped to the scene's `backdrop.detailRings`, and zero unless the tree
   * has a level below the scene cell. The walk and the outer rings keep their
   * cells, so the cost is paid only where a refined voxel is still pixels wide.
   */
  backdropRefinedRings?: number;
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
   * Honoured for solverless and renderer-only worlds. A solver-owned world
   * writes cells the scene geometry says nothing about, so its claim is not a
   * geometric one.
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
  /**
   * Leaves reserved for scene bricks activated in place: the floor of a display
   * world's reserve, and the whole reserve of a world the solver owns.
   */
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
/** The one pass a lighting frame encodes. A constant, so a frame allocates no descriptor. */
const LIVE_DERIVED_LIGHTING_PASS: GPUComputePassDescriptor = Object.freeze({ label: "Maintain live SVO derived lighting" });

/**
 * Dirty bricks one encoded frame repairs, before the environment overrides it.
 *
 * A brick is 512 voxels against up to 64 candidates, so this is the knob that
 * bounds a publication's worst frame. Publications converge over frames like the
 * light cache. Renderer startup drains the first publication in fenced chunks
 * before attaching it, so no consumer sees an incomplete scene generation.
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
 * A solver-owned world restores the full lane set regardless of this setting;
 * a renderer-only sidecar remains compact even when water is authored.
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

/** Resolve payload ownership independently of whether the document has water. */
export function octreeLiveScenePayloadProfile(
  fluidEnabled: boolean,
  rendererOnly = false,
  environment?: Record<string, string | undefined>,
): SparseBrickPayloadProfileName {
  return sparseBrickPayloadProfileForOwnership(
    fluidEnabled, rendererOnly, octreeLiveSceneDryPayloadProfile(environment),
  );
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

/**
 * Leaves a world reserves for live edits where no budget decides it: all of a
 * solver-owned world's reserve, and the least a display world is given when the
 * budget or the device buys it less. No publication is capped by it: the
 * mutation worklist holds every brick the reserve can come to be asked for.
 */
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
  readonly bounds: SparseSceneAxisAlignedBounds;
}

interface LiveScenePrimitiveEntry {
  readonly key: string;
  readonly primitive: SparseScenePrimitive;
}

function brickCoordinateKey({ x, y, z }: SparseBrickCoordinate): string { return `${x},${y},${z}`; }

/** Structural equality of two authored values, with nothing serialized. */
function sameSceneValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return a !== a && b !== b;
  if (Array.isArray(a) || ArrayBuffer.isView(a)) {
    const left = a as ArrayLike<unknown>, right = b as ArrayLike<unknown>;
    if (Array.isArray(a) !== Array.isArray(b) || left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) if (!sameSceneValue(left[index], right[index])) return false;
    return true;
  }
  if (Array.isArray(b) || ArrayBuffer.isView(b)) return false;
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
  for (const key in left) if (!sameSceneValue(left[key], right[key])) return false;
  for (const key in right) if (left[key] === undefined && right[key] !== undefined) return false;
  return true;
}

/**
 * Whether two authored primitives are the same solid at the same owner: every
 * field but the material, which reaches a table and never a voxel.
 */
function sameAuthoredGeometry(a: EnvironmentProxyPrimitive, b: EnvironmentProxyPrimitive): boolean {
  const left = a as unknown as Record<string, unknown>, right = b as unknown as Record<string, unknown>;
  for (const key in left) if (key !== "material" && !sameSceneValue(left[key], right[key])) return false;
  for (const key in right) if (key !== "material" && left[key] === undefined && right[key] !== undefined) return false;
  return true;
}

/** The bodies a live scene voxelizes, as one comparable value. */
function staticRigidBodiesStamp(bodies: SceneDescription["rigidBodies"]): string {
  return bodies.some((body) => body.motion === "static")
    ? JSON.stringify(bodies.map((body, ownerId) => body.motion === "static" ? sparseScenePrimitiveForRigidBody(body, ownerId) : 0))
    : "";
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
 *
 * A run of pages stacked in one column is claimed as one box: the ground is
 * such a run under every column, so a garden is a few thousand tall boxes
 * rather than tens of thousands of pages. A run's top is the top page's own
 * sum, so the box ends on the very number its last page would have, and every
 * consumer reads the bounds as a union — the bricks they touch, whether one
 * overlaps a node — which a run and its pages share.
 */
function solidWorldPageBounds(scene: SceneDescription,
  world: SolidWorld): SparseSceneAxisAlignedBounds[] {
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m] as const;
  const edge = [SOLID_WORLD_BRICK_CELLS * cell[0], SOLID_WORLD_BRICK_CELLS * cell[1],
    SOLID_WORLD_BRICK_CELLS * cell[2]] as const;
  const bounds: SparseSceneAxisAlignedBounds[] = [];
  // Pages arrive ascending in (z, y, x), so the run open under an x is its
  // column's while z holds; one that is not is closed by the mismatch.
  const open = new Map<number, { z: number; next: number; maximum: [number, number, number] }>();
  for (const [x, y, z] of solidWorldPageCoordinates(world)) {
    const run = open.get(x);
    if (run && run.z === z && run.next === y) {
      run.maximum[1] = origin[1] + y * edge[1] + edge[1];
      run.next = y + 1;
      continue;
    }
    const minimum: [number, number, number] = [origin[0] + x * edge[0], origin[1] + y * edge[1],
      origin[2] + z * edge[2]];
    const maximum: [number, number, number] = [minimum[0] + edge[0], minimum[1] + edge[1], minimum[2] + edge[2]];
    bounds.push({ minimum, maximum });
    open.set(x, { z, next: y + 1, maximum });
  }
  return bounds;
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
  // A brick's index in the lattice names it; first sight keeps its place.
  if (!Number.isSafeInteger(brickDimensions[0] * brickDimensions[1] * brickDimensions[2])) {
    throw new RangeError(`Brick lattice ${brickDimensions.join("x")} has more bricks than an index can name`);
  }
  const coordinates = new Map<number, SparseBrickCoordinate>();
  // Doubles, not integers: an unbounded or empty bound must stay what it is.
  const range = new Float64Array(6);
  let visited = 0;
  for (const region of regions) {
    regionBrickRange(region, worldOrigin, cellSize, brickSize, brickDimensions, range);
    for (let z = range[2]!; z <= range[5]!; z += 1) {
      visited += claimBrickSlab(coordinates, brickDimensions, range, z);
      if (visited >= BRICK_SELECTION_YIELD_BATCH) { visited = 0; yield; }
    }
  }
  return [...coordinates.values()];
}

/**
 * The bricks a bound touches, as `[x0, y0, z0, x1, y1, z1]` inside the lattice.
 *
 * This and the slab below are plain functions because a generator's own body
 * is never optimised: a loop written in one runs several times slower and
 * boxes every number it forms.
 */
function regionBrickRange(
  region: SparseSceneAxisAlignedBounds,
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
  range: Float64Array,
): void {
  for (let axis = 0; axis < 3; axis += 1) {
    range[axis] = Math.max(0, Math.floor((region.minimum[axis]! - worldOrigin[axis]!) / (cellSize[axis]! * brickSize)));
    range[axis + 3] = Math.min(brickDimensions[axis]! - 1,
      Math.floor((region.maximum[axis]! - worldOrigin[axis]!) / (cellSize[axis]! * brickSize)));
  }
}

/** Claim one z slab of a brick range; returns how many bricks it visited. */
function claimBrickSlab(
  coordinates: Map<number, SparseBrickCoordinate>,
  brickDimensions: readonly [number, number, number],
  range: Float64Array,
  z: number,
): number {
  const x0 = range[0]!, y0 = range[1]!, x1 = range[3]!, y1 = range[4]!;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const index = (z * brickDimensions[1] + y) * brickDimensions[0] + x;
      if (!coordinates.has(index)) coordinates.set(index, { x, y, z });
    }
  }
  return y1 < y0 || x1 < x0 ? 0 : (y1 - y0 + 1) * (x1 - x0 + 1);
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

/** Visit every brick of every `[x0, y0, z0, x1, y1, z1]` span, a brick once for each span that holds it. */
function eachSpanBrick(spans: Int32Array, visit: (x: number, y: number, z: number) => void): void {
  for (let base = 0; base < spans.length; base += 6) {
    for (let z = spans[base + 2]!; z <= spans[base + 5]!; z += 1) {
      for (let y = spans[base + 1]!; y <= spans[base + 4]!; y += 1) {
        for (let x = spans[base]!; x <= spans[base + 3]!; x += 1) visit(x, y, z);
      }
    }
  }
}

/**
 * Whether a solid world holds, in the lattice box `[low, high)`, anything the
 * display draws from a leaf. Off terrain that is a solid cell. Over terrain
 * the ground is drawn without leaves, so it is a cell an edit shaped that is
 * solid, or that the ground would have filled: a dig. A region is a whole box
 * of solid.
 */
function solidWorldHoldsContent(world: SolidWorld, low: readonly number[], high: readonly number[]): boolean {
  for (const region of world.regions ?? []) {
    if (region.minimum.every((value, axis) => value < high[axis]!)
      && region.maximumExclusive.every((value, axis) => value > low[axis]!)) return true;
  }
  const B = SOLID_WORLD_BRICK_CELLS, ground = world.ground;
  for (let pz = Math.floor(low[2]! / B); pz * B < high[2]!; pz += 1) {
    for (let py = Math.floor(low[1]! / B); py * B < high[1]!; py += 1) {
      for (let px = Math.floor(low[0]! / B); px * B < high[0]!; px += 1) {
        const index = world.directory.lookup([px, py, pz]);
        const page = index === undefined ? undefined : world.pages[index];
        const authored = page?.authored;
        if (!page || (ground && !authored)) continue;
        const x1 = Math.min(high[0]! - px * B, B), y1 = Math.min(high[1]! - py * B, B), z1 = Math.min(high[2]! - pz * B, B);
        for (let z = Math.max(low[2]! - pz * B, 0); z < z1; z += 1) {
          for (let y = Math.max(low[1]! - py * B, 0); y < y1; y += 1) {
            for (let x = Math.max(low[0]! - px * B, 0); x < x1; x += 1) {
              const cell = x + B * (y + B * z);
              if (!ground) { if (page.solidFraction[cell]! > 0) return true; continue; }
              if ((authored![cell >> 3]! & (1 << (cell & 7))) === 0) continue;
              if (page.solidFraction[cell]! > 0 || py * B + y < ground.solidCells(px * B + x, pz * B + z)) return true;
            }
          }
        }
      }
    }
  }
  return false;
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

/**
 * The table `buildOctreeSvoPbrMaterialPublication` packs, written into a table
 * already sized for the arena, without the publication around it: a staged
 * document needs the words and not their content hash. Returns the record count.
 */
function writeOctreeSvoPbrMaterialTable(
  table: Uint32Array,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[],
  surfaceModel: SvoTerrainSurfaceModel,
): number {
  table.fill(0);
  let count = 0;
  for (const record of buildDefaultSvoMaterialRecords(OCTREE_SVO_PBR_MATERIAL_REVISION, { terrainSurface: surfaceModel })) {
    writeSvoMaterialRecord(table, record);
    count = Math.max(count, record.materialId + 1);
  }
  for (const primitive of environmentPrimitives) {
    if (!Number.isSafeInteger(primitive.ownerIndex) || primitive.ownerIndex < 0) {
      throw new RangeError(`Environment material owner index for ${primitive.key} must be a non-negative safe integer`);
    }
    const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
    if (materialId >= OCTREE_LIVE_SCENE_MATERIAL_CAPACITY) {
      throw new RangeError(`Live scene needs material ${materialId} but the fixed arena holds ${OCTREE_LIVE_SCENE_MATERIAL_CAPACITY}`);
    }
    // A written record names its own slot in its ID word.
    if (table[materialId * SVO_MATERIAL_RECORD_WORDS + 20] !== 0) throw new RangeError(`Duplicate SVO material ID ${materialId}`);
    writeSvoEnvironmentMaterialRecord(table, materialId, primitive.material, OCTREE_SVO_PBR_MATERIAL_REVISION,
      svoMaterialFunctionIdForEnvironmentProxy(primitive, surfaceModel));
    count = Math.max(count, materialId + 1);
  }
  return count;
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

/** The scene maintenance revision `finalizeScene` last published. */
const STRUCTURAL_PUBLISHED_SCENE_REVISION_WORD = SPARSE_VOXEL_PUBLICATION_STATE.publishedSceneRevision;

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
  // A revision is published once, by the frame whose chunk completes it. The
  // chunk frames before it advance a cursor and publish nothing; a frame
  // planned past the last chunk finds the revision already published. Scene
  // revisions strictly increase, so the word never matches a later one.
  if (state[${STRUCTURAL_PUBLISHED_SCENE_REVISION_WORD}u] == completed) { return; }
  state[${STRUCTURAL_PUBLISHED_SCENE_REVISION_WORD}u] = completed;
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
  /** What the last edit that added bricks asked of the plan; see `growLiveDerivedAddressPlan`. */
  private liveDerivedLastGrowth?: SvoNodeMipAddressPlanGrowth;
  /**
   * Lighting frames still owed. The queue, its cursors and the feedback
   * windows live on the GPU and are never read back, so the host only counts:
   * frames certain to drain what was journaled, then the feedback window.
   */
  private liveDerivedFramesRemaining = 0;
  /** How many of those may still change opacity, and so republish the pyramid. */
  private liveDerivedDrainFramesRemaining = 0;
  /** Leaves journaled since the queue was last certain to be empty; an upper bound. */
  private liveDerivedSeedBound = 0;
  /** Leaves one voxelizer frame can journal at most. */
  private liveDerivedChunkSeedBound = 0;
  /** The voxelizer's frame count when lighting last looked. */
  private liveDerivedSceneActivity = 0;
  /** The lights or the sky moved: the next lighting frame opens the window everywhere. */
  private liveDerivedRelight = false;
  private liveDerivedInitial = true;
  /**
   * The published pyramid alternates between these two identical objects,
   * because its identity is the renderer's signal that opacity changed and
   * nothing in it is. No object is made for a publication.
   */
  private liveDerivedPyramidTwins?: [NonNullable<SparseVoxelSceneRenderSource["nodeMipPyramid"]>, NonNullable<SparseVoxelSceneRenderSource["nodeMipPyramid"]>];
  /** The one radiance source this world ever publishes; withdrawn and restored, never remade. */
  private liveDerivedRadianceSource?: SparseVoxelSceneRenderSource["tetrahedralRadiance"];
  private radianceEnabled = true;
  /** The scene's one word about what its surfaces are made of; see the ABI. */
  private surfaceModel!: SvoTerrainSurfaceModel;
  private solidWorld!: SolidWorld;
  private renderSolidWorld!: SolidWorld;
  /** The residual `renderSolidWorld` came from: what the next stroke's residual extends. */
  private renderResidual?: SvoRenderResidual;
  /** The last pair of snapshots compared, so validating and staging one edit compares once. */
  private solidChangeMemo?: { readonly previous: SolidWorld; readonly next: SolidWorld;
    readonly changes: SolidWorldChangeBounds };
  private solidWorldStamp = "";
  private terrainFieldStamp = "";
  /** Catalog expansion input, retained exactly for subsequent live publications. */
  private environmentDetailCellSize_m = 0;
  /**
   * Set only when immutable planar terminals make authored geometry structural:
   * the terrain field, every authored primitive's shape and the static bodies
   * are then baked into macro leaves, and a document that moves one is refused.
   */
  private planarTerminalsBaked = false;
  private liveScenePrimitiveStates = new Map<string, LiveScenePrimitiveState>();
  private liveScenePrimitives = new Map<string, LiveScenePrimitiveEntry>();
  /** `liveScenePrimitives` in publication order, replaced only when one of them is. */
  private liveScenePrimitiveList: readonly SparseScenePrimitive[] = [];
  /**
   * The document the live primitives were last listed from. Staging it again
   * does nothing; a keyed primitive update clears it, so the next document is
   * listed whole.
   */
  private stagedScene?: SceneDescription;
  /** The last document staged, whose parts the next one is compared with. */
  private stagedDocument?: SceneDescription;
  private stagedCatalog?: EnvironmentProxyCatalog;
  /** The staged catalog's primitives, shell faces first: the order the material table indexes. */
  private stagedAuthored: readonly EnvironmentProxyPrimitive[] = [];
  private stagedStaticBodies = "";
  /**
   * Host copies of the appearance tables, which an edit is diffed against.
   * `pbrMaterialStaging` is the table being written; it equals the published
   * copy between stages.
   */
  private pbrMaterialRecords = new Uint32Array(0);
  private pbrMaterialStaging = new Uint32Array(0);
  private materialEmissionRecords = new Uint32Array(0);
  private lightRecords = new Uint32Array(0);
  private environmentLightingRecords = new Uint32Array(0);
  /**
   * The host's exact copy of the live tree's links and its allocator: which
   * finest bricks a leaf covers, which leaves and nodes are free, and the
   * records that take the GPU's tree to the current one. Every question an
   * edit asks of the tree is answered here, by a walk from the root, and
   * nothing is read back.
   */
  private topologyMirror!: SparseBrickTopologyMirror;
  private readonly sampledTerrainNodes = new Map<string, SparseSceneAxisAlignedBounds>();
  /** The stored backdrop rings' box, voxelized by the initial publication. */
  private backdropDetailBounds?: SparseSceneAxisAlignedBounds;
  /** The sun cones' content ceiling, kept on the CPU so an edit can raise its columns in place. */
  private backdropContentCeiling?: BackdropContentCeiling;
  /** The ceiling section's word index within the terrain table. */
  private backdropContentCeilingWord = 0;
  /**
   * Finest bricks an edit activated, by coordinate key. Each holds something
   * of the staged document: the publication that leaves one holding nothing
   * takes its leaf back.
   */
  private readonly releasableTopologyBricks = new Map<string, SparseBrickCoordinate>();
  /** The solid lattice's corner and cell, which place a solid world's pages in the scene. */
  private solidLatticeOrigin!: readonly [number, number, number];
  private solidLatticeCell!: readonly [number, number, number];
  private pendingScenePublication?: SparseScenePublication;
  /** Records in the staged topology batch; none when the GPU's tree is the host's. */
  private pendingTopologyRecords = 0;
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
    const dryWorld = scene.systems?.fluid === false;
    const rendererOnly = options.rendererOnly === true;
    // A renderer-only world is a disposable sidecar: the solver beside it
    // keeps its own lattice and SolidWorld, so the sidecar may descend below
    // it. A world the solver itself draws from cannot (see `maximumDepth`).
    const refinementDepth = dryWorld || rendererOnly
      ? Math.max(0, Math.trunc(options.environmentRefinementDepth ?? 0))
      : 0;
    const refineScale = 2 ** refinementDepth;
    this.environmentDetailCellSize_m = scene.voxelDomain.finestCellSize_m / refineScale;
    const environmentCatalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default", {
      detailCellSize_m: this.environmentDetailCellSize_m,
    });
    const environmentPrimitives = environmentProxyPrimitives(environmentCatalog, true);
    const initialSolidWorld = solidWorldForScene(scene);
    this.solidWorld = initialSolidWorld;
    this.solidWorldStamp = solidWorldContentStamp(scene);
    this.terrainFieldStamp = terrainFieldStamp(scene);
    const solidWorldBounds = solidWorldPageBounds(scene, initialSolidWorld);
    /**
     * The backdrop ground (`scene.backdrop`): render-only terrain beyond the
     * set's footprint, out to the haze horizon. It is not octree content: the
     * renderer walks it as a tiled height field (`backdrop-terrain-tiles.ts`)
     * whose table sits past the dense owner lane. Only on a world the solver
     * does not own, so no solver lattice, SolidWorld or collider ever sees it,
     * and the set's domain is not widened.
     */
    const backdropField = scene.backdrop && (dryWorld || rendererOnly)
      ? compileBackdropField(scene.backdrop, backdropSeamForScene(scene)) : undefined;
    /**
     * The backdrop's stored detail rings (`backdrop-detail.ts`): the walk's
     * inner levels as real octree voxels, so they are lit like the set. The
     * domain widens to hold them — bounds only, no bricks enumerated — padded
     * by one outermost-ring brick, which is how far the centre can move when
     * it snaps to the final origin below.
     */
    const backdropDetailRings = backdropField ? scene.backdrop?.detailRings ?? 0 : 0;
    const backdropRefinedRings = refinementDepth > 0
      ? Math.min(backdropDetailRings, Math.max(0, Math.trunc(options.backdropRefinedRings ?? 0))) : 0;
    const backdropCell_m = sceneCellSizes_m(scene)[0];
    const backdropDetailLattice_m = backdropDetailRings > 0
      ? backdropDetailCentreLattice(backdropCell_m, backdropDetailRings, brickSize) : undefined;
    const backdropDetailDomain = backdropField && backdropDetailRings > 0 ? backdropDetailWorldBounds(backdropField,
      backdropDetailFromPlan(backdropField, planBackdropTileLayout(backdropField, {
        origin_m: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m],
        cellSize_m: backdropCell_m, firstLevel: backdropDetailRings,
      // A dual-grid plan also claims an apron brick past the outer square.
      }), backdropDetailRings), backdropDetailLattice_m! * (options.surfaceDualMarchingCubes
        ? 1 + BACKDROP_DETAIL_DUAL_APRON_BRICKS : 1)) : undefined;
    const authoredWorldBounds = scene.voxelDomain.bounds_m;
    // This tree is a sparse presentation consumer, not the fluid solver's
    // address space. Fluid residency claims wet pages as they appear; static
    // geometry claims only the exact voxel boxes compiled from SolidWorld.
    // Claiming the whole logical container here was the render-side OOM path
    // for long, mostly dry tanks.
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
        worldBounds_m: backdropDetailDomain ? {
          min: { x: Math.min(backdropDetailDomain.min.x, authoredWorldBounds?.min.x ?? Infinity),
            y: Math.min(backdropDetailDomain.min.y, authoredWorldBounds?.min.y ?? Infinity),
            z: Math.min(backdropDetailDomain.min.z, authoredWorldBounds?.min.z ?? Infinity) },
          max: { x: Math.max(backdropDetailDomain.max.x, authoredWorldBounds?.max.x ?? -Infinity),
            y: Math.max(backdropDetailDomain.max.y, authoredWorldBounds?.max.y ?? -Infinity),
            z: Math.max(backdropDetailDomain.max.z, authoredWorldBounds?.max.z ?? -Infinity) },
        } : authoredWorldBounds,
        solverClaim: "none",
        // Per-proxy bricks are read below by the unrefined voxel claim alone.
        // A refined or function-fitted world rasterises the same bounds on
        // its own lattice, the device's selection takes the ranges, and the
        // tree's depth follows from the lattice's dimensions.
        enumerateProxyBricks: refinementDepth === 0 && !options.surfaceDualMarchingCubes
          && !(options.selectPrimitiveBricksGpu && options.sceneSolids?.length && octreeLiveSceneBrickClaim() === "reachable" && (dryWorld || rendererOnly)),
      }
    );
    this.solverGridOriginCells = sceneDomain.solverGridOriginCells;
    this.sceneBrickDimensions = sceneDomain.brickDimensions;  // replaced below once the refinement depth is known
    this.sceneWorldOrigin = [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z];
    const solverLevel = sparseSceneOctreeMaximumDepth(sceneDomain.brickDimensions, sceneDomain.coordinates);
    /**
     * Extra octree levels the *environment* may use, below the solver's own.
     *
     * Only on a tree the solver does not own. `planSparseSceneDomain` claims
     * every brick of the container as a solver brick while a solver is present,
     * and a solver brick pins its node at `solverLevel` — so a world the
     * simulation draws from has nowhere for environment geometry to descend
     * into. A dry scene makes no such claim, and a renderer-only sidecar drops
     * it below (`pinnedBricks`, `plannedSolverBricks`), so either is free to
     * spend depth where the geometry actually is.
     */
    const maximumDepth = solverLevel + refinementDepth;
    const refinedBrickDimensions = sceneDomain.brickDimensions.map((value) => value * refineScale) as [number, number, number];
    const renderCellSize = sceneDomain.cellSize_m.map((value) => value / refineScale) as [number, number, number];
    // The physical lattice can round the requested cell size. Do not feed that
    // rounded value back into catalog expansion: even a one-ULP change can
    // alter procedural geometry and make the same scene fail its immutable
    // planar-topology stamp when it is next published.
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
    let renderTerrain: import("../scene-publication/svo-render-solid-field").SvoRenderTerrainField | undefined;
    // A function-fitted surface needs the continuous terrain whatever the
    // depth: fitted from the solver's voxel SDF instead, a renderer-only
    // sidecar at depth 0 draws the terraces and their sentinel gaps.
    if (refinementDepth > 0 || (dryWorld && (options.surfaceContours === true || options.surfaceDualContouring === true || options.surfaceDualMarchingCubes === true))
      || (rendererOnly && options.surfaceDualMarchingCubes === true)) {
      if (options.buildRenderTerrainGpu) yield options.buildRenderTerrainGpu(renderCellSize, SOLID_WORLD_TERRAIN_MATERIAL_ID)
        .then(field => { renderTerrain = field; });
      renderTerrain ??= yield* buildSvoRenderTerrainFieldSteps(scene, renderCellSize, SOLID_WORLD_TERRAIN_MATERIAL_ID);
    }
    const planarCatalog = buildSvoPlanarBoundaryCatalog(environmentPrimitives, (primitive) => ({
        materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
        ownerId: SCENE_ENVIRONMENT_OWNER_BASE + primitive.ownerIndex,
      }));
    // Never promote editable voxel patches into immutable planar terminals.
    // Static environment surfaces retain their compact planar representation.
    const solidPlanarCatalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene,
      initialSolidWorld.patches, planarCatalog.sources.length,
      { promoteEditablePatches: false });
    const planarSources = [...planarCatalog.sources];
    this.renderResidual = svoRenderResidual(scene, initialSolidWorld);
    const residualSolidWorld = this.renderResidual.residual;
    this.renderSolidWorld = residualSolidWorld;
    const terrainRefinement = renderTerrain
      ? createSvoRenderTerrainRefinement({
        field: renderTerrain,
        worldOrigin_m: worldOrigin,
        renderCellSize_m: renderCellSize,
        refinedBrickDimensions,
        nodeEdge_m,
        brickSize,
        maximumDepth,
      })
      : undefined;
    // A terrain scene draws its edits from the residual's pages, as every live
    // republish does (`setSolidWorld`): refine where their surfaces can lie.
    const editPageLattice = {
      origin_m: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m] as const,
      cellSize_m: sceneCellSizes_m(scene),
    };
    const editSurfacePages = renderTerrain
      ? terrainEditSurfacePages(new TerrainEditPages(renderTerrain, editPageLattice).select(residualSolidWorld))
      : [];
    const solidPatchRefinement = editSurfacePages.length > 0
      ? createSvoRenderEditPageRefinement({
        pages: editSurfacePages,
        lattice: editPageLattice,
        worldOrigin_m: worldOrigin,
        nodeEdge_m,
        maximumDepth,
      })
      : undefined;
    // Shell exclusion is independent of planar promotion: canonical tank walls
    // stay physical-only, while editable fills remain mutable voxel geometry.
    const planarLeafOptions = {
      sources: planarSources,
      blockers: [
        ...environmentPrimitives.map((primitive) => ({
          minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y,
            primitive.aabb_m.min.z] as const,
          maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y,
            primitive.aabb_m.max.z] as const,
          planarSourceIndex: planarCatalog.patchIndexByOwner.get(primitive.ownerIndex),
        })),
        // Terrain lives in pages, so its blockers are the pages of what is
        // drawn: the residual, which no longer holds the physics-only shell.
        ...(scene.terrain
          ? (residualSolidWorld === initialSolidWorld ? solidWorldBounds
            : solidWorldPageBounds(scene, residualSolidWorld)).map((bounds) => ({
            minimum: bounds.minimum,
            maximum: bounds.maximum,
          }))
          : svoPlanarSolidWorldBlockers(scene, initialSolidWorld.patches,
            solidPlanarCatalog)),
        ...scene.rigidBodies.flatMap((body, ownerId) => {
          if (body.motion !== "static") return [];
          const bounds = sparseScenePrimitiveBounds(sparseScenePrimitiveForRigidBody(body, ownerId));
          return [{ minimum: bounds.minimum, maximum: bounds.maximum }];
        }),
      ],
      worldOrigin_m: worldOrigin,
      nodeEdge_m,
    } satisfies import("../scene-publication/svo-planar-boundary").SvoPlanarLeafClassifierOptions;
    const planarLeafClassifier = createSvoPlanarLeafClassifier(planarLeafOptions);
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
        additionalSurfaces: scene.rigidBodies.flatMap((body, ownerId) => {
          if (body.motion !== "static") return [];
          const bounds = sparseScenePrimitiveBounds(sparseScenePrimitiveForRigidBody(body, ownerId));
          return [{
            aabb_m: {
              min: { x: bounds.minimum[0], y: bounds.minimum[1], z: bounds.minimum[2] },
              max: { x: bounds.maximum[0], y: bounds.maximum[1], z: bounds.maximum[2] },
            },
            descriptor: svoPrimitiveForRigidBody(body,
              environmentPrimitives.length + ownerId, ownerId),
          }];
        }),
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
     * `lib/svo/features/construction/svo-environment-coarsening.ts` for the rule and for why the
     * ceiling is `log2(brickSize)`; it is offered here and taken per node.
     *
     * Wet and unrefined only. A solverless world already has
     * `environmentRefinementDepth` below the lattice and buried-ground
     * coarsening above it, each with its own predicate and its own gate, and
     * `minimumEnvironmentLevel` can only carry one meaning at a time — so a
     * refined sidecar takes the refinement and gives up the coarsening.
     */
    const wetCoarsening = !dryWorld && refinementDepth === 0;
    const coarseningRegions = wetCoarsening ? [
      ...solidWorldTerrainSurfaceCoarseningRegions(scene, initialSolidWorld),
      ...solidWorldVoxelPatchCoarseningRegions(scene, residualSolidWorld.patches),
    ] : [];
    const environmentCoarsening = wetCoarsening
      ? createSvoEnvironmentCoarsening({
        primitives: environmentPrimitives,
        regions: coarseningRegions,
        worldOrigin_m: worldOrigin as readonly [number, number, number],
        nodeEdge_m, brickSize, maximumDepth,
        crowdingTarget: OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
      })
      : undefined;
    const pinnedBricks = new Set<string>();
    if (refinementDepth === 0) {
      for (const coordinate of sceneDomain.solverBrickCoordinates) pinnedBricks.add(brickCoordinateKey(coordinate));
    }
    const sceneSolids = (dryWorld || rendererOnly)
      && octreeLiveSceneBrickClaim() === "reachable"
      ? options.sceneSolids ?? [] : [];
    reportStage("Select the bricks the scene reaches");
    yield;
    const gpuSelection = sceneSolids.length > 0 ? options.selectPrimitiveBricksGpu : undefined;
    let gpuOccupancy: import("../../../core/adaptive-sparse-brick-plan").SparseBrickProxyOccupancy | undefined;
    if (gpuSelection) {
      yield gpuSelection({
        regions: environmentPrimitives.map(primitive => ({minimum:[primitive.aabb_m.min.x,primitive.aabb_m.min.y,primitive.aabb_m.min.z],
          maximum:[primitive.aabb_m.max.x,primitive.aabb_m.max.y,primitive.aabb_m.max.z]})),
        brickRanges: refinementDepth === 0 ? sceneDomain.proxyBrickRanges.slice(0, environmentPrimitives.length) : undefined,
        retainedRegions: [...solidWorldBounds, ...scene.rigidBodies.flatMap((body, ownerId) => body.motion === "static"
          ? [sparseScenePrimitiveBounds(sparseScenePrimitiveForRigidBody(body, ownerId))] : [])],
        worldOrigin, cellSize:renderCellSize, brickSize, brickDimensions:refinedBrickDimensions, maximumDepth,
        haloCells: options.surfaceDualMarchingCubes ? 1 : 0,
      }).then(result => { gpuOccupancy = result; });
    }
    const primitiveBricks = gpuOccupancy ? [] : (refinementDepth > 0 || options.surfaceDualMarchingCubes
      ? yield* liveSceneBrickCoordinatesForRegionsSteps(
        environmentPrimitives.map((primitive) => ({
          minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y, primitive.aabb_m.min.z].map(v=>v-(options.surfaceDualMarchingCubes ? Math.max(...renderCellSize) : 0)) as [number,number,number],
          maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y, primitive.aabb_m.max.z].map(v=>v+(options.surfaceDualMarchingCubes ? Math.max(...renderCellSize) : 0)) as [number,number,number],
        })),
        worldOrigin, renderCellSize, brickSize, refinedBrickDimensions)
      : sceneDomain.proxyBrickCoordinates.slice(0, environmentPrimitives.length).flat());
    const solidWorldBricks = gpuOccupancy ? [] : yield* liveSceneBrickCoordinatesForRegionsSteps(
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
    const rigidBodyBricks = claimsContainer || gpuOccupancy ? [] : yield* liveSceneBrickCoordinatesForRegionsSteps(
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
    const reachablePrimitiveBricks = gpuOccupancy ? [] : options.surfaceDualMarchingCubes ? primitiveBricks : (yield* liveSceneReachableBrickCoordinatesSteps(
      primitiveBricks, sceneSolids, worldOrigin, renderCellSize, brickSize, pinnedBricks));
    const gpuNodeClassification = (options.surfaceDualContouring !== true && options.surfaceDualMarchingCubes !== true) && (refinementDepth > 0 || !dryWorld) && !surfaceRefinement
      ? options.classifyEnvironmentNodesGpu : undefined;
    if ((options.surfaceDualContouring || options.surfaceDualMarchingCubes) && gpuOccupancy) {
      // These strategies force occupied leaves to maximumDepth. Their fitted samples
      // and candidate slots can exceed a storage binding before the planner
      // allocates millions of nodes or any payload arenas. Let the existing
      // resolution ladder answer this failure immediately.
      assertDualGridAttachmentFits(gpuOccupancy.keys(maximumDepth), brickSize,
        Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize),
        OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
        Math.min(sceneDomain.brickDimensions.reduce((a, b) => a * b, 1),
          options.sceneMutationBrickCapacity ?? OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY));
    }
    // The backdrop's tile plan, on the final origin. With detail rings its
    // centre snaps to the outermost ring's bricks, so every ring boundary is a
    // node boundary of that ring's leaf level and stored columns are walked
    // columns; the walk starts at the first level not stored.
    if (backdropField && Math.abs(sceneDomain.cellSize_m[0] - backdropCell_m) > 1e-12) {
      throw new RangeError("Backdrop detail lattice does not match the scene domain's cell");
    }
    const backdropTilePlan = backdropField ? yield* planBackdropTilesSteps(backdropField, {
      origin_m: worldOrigin, cellSize_m: sceneDomain.cellSize_m[0],
      firstLevel: backdropDetailRings, centreLattice_m: backdropDetailLattice_m,
    }) : undefined;
    const backdropDetail = backdropField && backdropTilePlan && backdropDetailRings > 0
      ? backdropDetailFromPlan(backdropField, backdropTilePlan, backdropDetailRings, backdropRefinedRings) : undefined;
    const backdropDetailClassifier = backdropField && backdropDetail ? createBackdropDetailClassifier({
      field: backdropField, detail: backdropDetail, worldOrigin_m: worldOrigin, nodeEdge_m,
      // The level whose voxel is the scene cell: the solver's, a refinement
      // depth above the finest.
      solverLevel: maximumDepth - refinementDepth,
      dualGrid: options.surfaceDualMarchingCubes === true,
    }) : undefined;
    if (backdropField && backdropDetail) {
      const box = backdropDetailWorldBounds(backdropField, backdropDetail, options.surfaceDualMarchingCubes
        ? BACKDROP_DETAIL_DUAL_APRON_BRICKS * backdropDetailLattice_m! : 0);
      this.backdropDetailBounds = { minimum: [box.min.x, box.min.y, box.min.z], maximum: [box.max.x, box.max.y, box.max.z] };
    }
    reportStage("Plan the adaptive octree");
    yield;
    // The interruptible form of the same plan. This is the longest block in a
    // refined build by a wide margin, so yielding only *around* it would leave
    // most of the freeze exactly where it was.
    const plan = yield* planAdaptiveSparseBrickOctreeSteps({
      brickSize,
      proxyOccupancy: gpuOccupancy,
      classifyEnvironmentBatch: gpuNodeClassification ? (level, coordinates) => gpuNodeClassification({
        planar: planarLeafOptions, candidateCount: environmentPrimitives.length,
        candidateLimit: OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET, level, coordinates,
        coarsening: wetCoarsening ? {
          resolves_m: Math.max(...nodeEdge_m[level]) / brickSize * SVO_ENVIRONMENT_FEATURE_VOXELS,
          features_m: environmentPrimitives.map(environmentProxyFeatureSize_m),
          regions: coarseningRegions,
        } : undefined,
      }) : undefined,
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
      refineEnvironmentLeaf: (level, coordinate) =>
        ((options.surfaceDualContouring === true || options.surfaceDualMarchingCubes === true) && level < maximumDepth)
        || terrainRefinement?.refineEnvironmentLeaf(level, coordinate)
        || solidPatchRefinement?.refineEnvironmentLeaf(level, coordinate)
        || (!gpuNodeClassification && (planarLeafClassifier.requiresFineVoxelResidual(level, coordinate)
        || (surfaceRefinement
          ? surfaceRefinement.refineEnvironmentLeaf(level, coordinate)
          : refinementDepth > 0
            ? candidatesInBrick(level, coordinate, OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET)
              > OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET
            : environmentCoarsening?.refineEnvironmentLeaf(level, coordinate) ?? false))),
      classifyEnvironmentLeaf: (options.surfaceDualContouring || options.surfaceDualMarchingCubes) ? undefined : planarLeafClassifier,
      // The stored ring leaves come back marked (`supplementalLeaf`), and the
      // classifier's name lets a rebuild over the same set take them as found.
      classifySupplementalNode: backdropDetailClassifier?.classify,
      supplementalContentKey: backdropDetailClassifier?.contentKey,
    });
    // Every node-mip page lies under a leaf, so the leaf boxes bound the
    // content each column can hold (see `BackdropContentCeiling`). Built for
    // every world: sun cones stop at it whether or not there is a backdrop.
    {
      const worldEdge = refinedBrickDimensions.map((count, axis) => count * refinedBrickEdge[axis]!);
      const ceiling = createBackdropContentCeiling([worldOrigin[0], worldOrigin[2]], [worldEdge[0]!, worldEdge[2]!]);
      const planarTops = new Map(planarLeafOptions.sources.map((source) => [source.sourceIndex, source.bounds_m.maximum[1]]));
      // In a function of its own, as every loop over the leaves below is: this
      // is a generator, and a generator's own body is never optimised.
      const raiseOverLeaves = (): void => { for (const leaf of plan.leaves) {
        const node = plan.nodes[leaf.nodeIndex]!;
        const scale = 2 ** (maximumDepth - node.level);
        const { x, y, z } = node.coordinate;
        let top = worldOrigin[1] + (y + 1) * refinedBrickEdge[1] * scale;
        // A planar terminal holds its one source and nothing else, so its
        // content stops at that source's top, not at the leaf box's: a coarse
        // floor leaf is most of a metre of empty air over the floor. One of
        // the leaf's own voxels above that top covers its pages' smear.
        const planarTop = leaf.terminalKind === SPARSE_BRICK_LEAF_TERMINAL.planarBoundary
          ? planarTops.get(leaf.terminalIndex) : undefined;
        if (planarTop !== undefined) {
          top = Math.min(top, planarTop + refinedBrickEdge[1] * scale / brickSize);
        }
        raiseBackdropContentCeilingOver(ceiling,
          worldOrigin[0] + x * refinedBrickEdge[0] * scale, worldOrigin[2] + z * refinedBrickEdge[2] * scale,
          worldOrigin[0] + (x + 1) * refinedBrickEdge[0] * scale, worldOrigin[2] + (z + 1) * refinedBrickEdge[2] * scale, top);
      } };
      raiseOverLeaves();
      this.backdropContentCeiling = ceiling;
    }
    // The backdrop's level-0 voxels continue the set's own terrain lattice
    // (the scene cell, before any environment refinement) from the same origin.
    const backdropTerrainTable = backdropField && backdropTilePlan
      ? packBackdropTerrainTable(backdropField, SOLID_WORLD_TERRAIN_MATERIAL_ID, backdropTilePlan, this.backdropContentCeiling) : undefined;
    this.finestLevel = plan.maximumDepth;
    this.sceneBrickDimensions = refinedBrickDimensions;
    reportStage("Pack the octree and allocate its arenas");
    yield;
    let coveredLeaves = 0;
    // Planned terminals an edit may split: the analytic ones, and the sampled
    // terrain a split resamples.
    const splittableTerminalLeaves = new Set<number>();
    let finestSplittableTerminals = 0;
    let voxelTerminalLeaves = 0, planarTerminalLeaves = 0;
    const coverLeaves = (first: number, end: number): void => { for (let index = first; index < end; index += 1) {
      const leaf = plan.leaves[index]!;
      const node = plan.nodes[leaf.nodeIndex];
      if (leaf.terminalKind === SPARSE_BRICK_LEAF_TERMINAL.voxels) voxelTerminalLeaves += 1;
      if (leaf.terminalKind === SPARSE_BRICK_LEAF_TERMINAL.planarBoundary) {
        planarTerminalLeaves += 1;
        splittableTerminalLeaves.add(leaf.index);
        if (node.level === maximumDepth) finestSplittableTerminals += 1;
      }
      // Stored backdrop rings are neither resampled terrain nor split on edit.
      if (scene.terrain && node.level < maximumDepth && leaf.terminalKind === 0
        && plan.supplementalLeaf[index] === 0) {
        const key = `${node.level}:${brickCoordinateKey(node.coordinate)}`;
        const edge = renderCellSize.map(value => value * brickSize * 2 ** (maximumDepth - node.level));
        this.sampledTerrainNodes.set(key, {
          minimum: edge.map((value, axis) => worldOrigin[axis]! + [node.coordinate.x, node.coordinate.y, node.coordinate.z][axis]! * value) as [number, number, number],
          maximum: edge.map((value, axis) => worldOrigin[axis]! + ([node.coordinate.x, node.coordinate.y, node.coordinate.z][axis]! + 1) * value) as [number, number, number],
        });
        splittableTerminalLeaves.add(leaf.index);
      }
    } };
    while (coveredLeaves < plan.leaves.length) {
      const end = Math.min(plan.leaves.length, coveredLeaves + 4096);
      coverLeaves(coveredLeaves, end);
      if ((coveredLeaves = end) % 4096 === 0) yield;
    }
    const packed = packSparseBrickPlan(plan, 1);
    yield;
    const sceneBrickVolume = sceneDomain.brickDimensions.reduce((product, value) => product * value, 1);
    const requestedMutationCapacity = options.sceneMutationBrickCapacity ?? OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY;
    if (!Number.isSafeInteger(requestedMutationCapacity) || requestedMutationCapacity < 1) {
      throw new RangeError("Live scene mutation brick capacity must be a positive safe integer");
    }
    // The floor of the reserve. What a whole session may add is decided below,
    // once the payload this world stores a leaf in is known.
    const minimumTopologyLeafReserve = Math.min(sceneBrickVolume, requestedMutationCapacity);
    // A renderer-only sidecar never receives solver writes even when the scene
    // document contains water. It can therefore use the same compact lanes as
    // a solverless world. The actual solver-owned octree still forces `full`.
    const payloadProfile = octreeLiveScenePayloadProfile(
      !dryWorld, rendererOnly,
    );
    const sceneGeometryFormat: SparseBrickSceneGeometryFormat =
      payloadProfile === "dry" ? (options.surfaceContours ? "f16-unorm8" : octreeLiveSceneSceneGeometryFormat()) : "f32x2";
    const leafPayloadMode: SparseBrickLeafPayloadMode =
      payloadProfile === "dry" ? octreeLiveSceneLeafPayloadMode() : "dense";
    if (backdropTerrainTable && leafPayloadMode !== "dense") {
      throw new RangeError("A backdrop's terrain table requires the dense leaf payload");
    }
    // A dense world without a backdrop still carries the content ceiling.
    const terrainTable = backdropTerrainTable
      ?? (leafPayloadMode === "dense" && this.backdropContentCeiling ? packContentCeilingTable(this.backdropContentCeiling) : undefined);
    if (terrainTable) this.backdropContentCeilingWord = terrainTable[BACKDROP_TERRAIN_TABLE.ceilingWord]!;
    else this.backdropContentCeiling = undefined;
    // The display reserves what its whole declared domain can come to need, up
    // to a byte budget: the leaves, and with them the lighting pages that many
    // bricks of built surface are lit from, so that the two run out together.
    // A world the solver owns stores seven times the payload a leaf and takes
    // no live edits of its own, so it keeps the floor.
    const plannedFinestLeaves = ((): number => {
      let finest = 0;
      for (const leaf of plan.leaves) if (plan.nodes[leaf.nodeIndex].level === maximumDepth) finest += 1;
      return finest;
    })();
    const derivedReserveBasePages = liveSvoBasePageDimensions(refinedBrickDimensions, brickSize);
    const derivedReserveLevels = sparseSceneOctreeMaximumDepth(derivedReserveBasePages, []) + 1;
    const derivedReserveFloor = derivedReserveLevels <= SVO_NODE_MIP_MAXIMUM_LEVELS
      ? svoOpacityLevelFloor({ levelCount: derivedReserveLevels, cellSize_m: Math.max(...renderCellSize) }) : 0;
    const dualGrid = options.surfaceDualContouring === true || options.surfaceDualMarchingCubes === true;
    const topologyReserve = payloadProfile === "dry" ? planLiveTopologyLeafReserve({
      brickDimensions: refinedBrickDimensions, maximumDepth,
      minimumLeaves: minimumTopologyLeafReserve,
      plannedNodes: plan.nodes.length, plannedLeaves: plan.leaves.length, plannedFinestLeaves,
      splittableTerminals: splittableTerminalLeaves.size, finestSplittableTerminals,
      payloadBytes: (leaves) => resolveSparseBrickPayloadLayout(payloadProfile,
        leaves * brickSize ** 3 + (leafPayloadMode === "dense"
          ? Math.max(terrainTable?.length ?? 0, BACKDROP_TERRAIN_HEADER_WORDS) : 0),
        sceneGeometryFormat, { leafPayloadMode, voxelsPerLeaf: brickSize ** 3,
          retainDenseLanes: leafPayloadMode === "banded" ? SPARSE_BRICK_BANDED_PRODUCER_DENSE_LANES : undefined,
        }).totalBytes,
      storageLimitBytes: Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize),
      candidatesPerBrick: OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
      dualGridVoxelsPerLeaf: dualGrid ? brickSize ** 3 : 0,
      maintenanceFixedBytes: WEBGPU_SOLID_WORLD_DISPLAY_BUDGET_BYTES,
      derived: derivedReserveLevels <= SVO_NODE_MIP_MAXIMUM_LEVELS ? {
        levelPages: svoNodeMipLevelPageDimensions(derivedReserveBasePages, derivedReserveLevels)
          .slice(derivedReserveFloor).map(pages => pages[0] * pages[1] * pages[2]),
        // Eight-cell pages: a four-cell brick is an eighth of one, and so is a
        // page of the level below the floor.
        occupiedBasePages: Math.ceil(plannedFinestLeaves * brickSize ** 3
          / SVO_NODE_MIP_LAYOUT.interiorSize ** 3 / 8 ** derivedReserveFloor),
        bytesPerPage: SVO_NODE_MIP_ADDRESS_PLAN.reservedBytesPerPage,
      } : undefined,
    }) : undefined;
    const topologyLeafReserve = topologyReserve?.leaves ?? minimumTopologyLeafReserve;
    // Every record one batch can come to hold, so that one stroke may spend
    // the whole reserve and take as much back. Nothing else is sized by it.
    this.topologyMutationCapacity = Math.max(1, topologyReserve?.mutationRecords
      ?? liveTopologyMutationRecords(topologyLeafReserve, maximumDepth, finestSplittableTerminals));
    // Absent on a world the solver owns, which keeps the address plan's own reserve.
    const derivedReserveBytes = topologyReserve
      && topologyReserve.derivedPages * SVO_NODE_MIP_ADDRESS_PLAN.reservedBytesPerPage;
    const nodeCapacity = Math.max(1, plan.nodes.length
      + sparseBrickTopologyMutationNodeReserve(maximumDepth, topologyLeafReserve));
    const leafCapacity = Math.max(1, plan.leaves.length + topologyLeafReserve);
    this.topologyMirror = new SparseBrickTopologyMirror({
      maximumDepth, nodeCapacity, leafCapacity, recordCapacity: this.topologyMutationCapacity,
      plan, splittable: leaf => splittableTerminalLeaves.has(leaf),
    });
    this.solidLatticeOrigin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
    this.solidLatticeCell = sceneCellSizes_m(scene);
    // The loud half of the brick raster's node-index ceiling, at the one point
    // where the count first exists. Capacity rather than the planned count on
    // purpose: topology mutation can reach every reserved node, so a world that
    // *could* grow past the mask is already broken, and finding that out at the
    // publication that crosses it is finding out from a scene with holes in it.
    assertSvoBrickRasterNodeAddressable(nodeCapacity, "the live sparse-brick world");
    this.tree = new SparseBrickOctreeGPU(device, {
      brickSize, nodeCapacity, leafCapacity, label: "Octree unified live sparse-brick world",
      // The terrain table's header is always there on a dense world, zeroed
      // when there is no backdrop, so a reader can ask for its level count.
      voxelLaneTailWords: leafPayloadMode === "dense"
        ? Math.max(terrainTable?.length ?? 0, BACKDROP_TERRAIN_HEADER_WORDS) : 0,
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
    if (terrainTable) {
      device.queue.writeBuffer(this.tree.payload,
        this.tree.sceneMaterialOwnerOffsetBytes + this.tree.voxelCapacity * Uint32Array.BYTES_PER_ELEMENT, terrainTable);
    }
    yield;
    this.brickOccupancyBuilder = new WebGpuSvoBrickOccupancyBuilder(
      device, payloadProfile, this.tree.scenePayloadLanes);
    yield;
    this.topologyMutator = new WebGpuSparseBrickTopologyMutator(device);
    yield;
    this.topologyMutationWorklist = device.createBuffer({
      label: "Live sparse scene topology mutation worklist",
      size: (SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords
        + this.topologyMutationCapacity * SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords) * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.topologyMutator.bind(this.tree, { buffer: this.topologyMutationWorklist, capacity: this.topologyMutationCapacity });
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
    const localBrickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
    const leafIndices = new Uint32Array(localBrickDimensions[0] * localBrickDimensions[1] * localBrickDimensions[2]);
    // A solver-level leaf inside the solver's box writes its own slot: the
    // brick's place in the box is its coordinate less the box's origin.
    const leafNamed = new Uint8Array(leafIndices.length);
    let indexedLeaves = 0;
    const nameLeaves = (first: number, end: number): void => { for (let index = first; index < end; index += 1) {
      const leaf = plan.leaves[index]!;
      const node = plan.nodes[leaf.nodeIndex];
      if (node.level === solverLevel) {
        const x = leaf.coordinate.x - solverOriginBricks[0], y = leaf.coordinate.y - solverOriginBricks[1];
        const z = leaf.coordinate.z - solverOriginBricks[2];
        if (x >= 0 && y >= 0 && z >= 0 && x < localBrickDimensions[0] && y < localBrickDimensions[1] && z < localBrickDimensions[2]) {
          const brick = x + localBrickDimensions[0] * (y + localBrickDimensions[1] * z);
          leafIndices[brick] = leaf.index;
          leafNamed[brick] = 1;
        }
      }
    } };
    while (indexedLeaves < plan.leaves.length) {
      const end = Math.min(plan.leaves.length, indexedLeaves + 4096);
      nameLeaves(indexedLeaves, end);
      if ((indexedLeaves = end) % 4096 === 0) yield;
    }
    // A dry scene publishes no solver bricks at all — whether because it spent
    // its depth on the environment or because it never claimed the container —
    // so there is no leaf to name and nothing that will ever read this table:
    // the fluid frame path is not reached when `systems.fluid` is false.
    // Anywhere the solver *is* present, a missing leaf is still a hard error —
    // that is the statement that solver bricks are never coarsened.
    const unnamedBrick = plannedSolverBricks.length > 0 ? leafNamed.indexOf(0) : -1;
    if (unnamedBrick >= 0) {
      const x = unnamedBrick % localBrickDimensions[0], y = Math.floor(unnamedBrick / localBrickDimensions[0]) % localBrickDimensions[1];
      const z = Math.floor(unnamedBrick / (localBrickDimensions[0] * localBrickDimensions[1]));
      throw new Error(`Fluid brick ${solverOriginBricks[0] + x},${solverOriginBricks[1] + y},${solverOriginBricks[2] + z} has no finest scene leaf`);
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
    // Dry publication copies topology only. Its dynamic lanes are absent,
    // so allocating full zero-filled staging fields wastes 36 bytes per voxel
    // and can exceed the device's buffer limit before any scene work starts.
    const initialVoxelCount = this.tree.payloadLayout.lanes.geometry.present
      ? Math.max(1, plan.voxelCount) : 1;
    const geometry = storageBuffer(device, "Sparse brick source geometry", initialVoxelCount * 16);
    const velocity = storageBuffer(device, "Sparse brick source velocity", initialVoxelCount * 16);
    const materialOwners = storageBuffer(device, "Sparse brick source material owners", initialVoxelCount * 4);
    const planarBoundaryRecords = packPlanarBoundaryPatches(planarSources.map(({ patch }) => patch));
    const planarBoundaries = storageBuffer(device, "Accepted SVO planar boundary records",
      Math.max(PLANAR_BOUNDARY_PATCH_BYTES, planarBoundaryRecords.byteLength), planarBoundaryRecords);
    this.sourceBuffers = [counts, topology, geometry, velocity, materialOwners, planarBoundaries];
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
      surfaceContours: options.surfaceContours,
      surfaceDualContouring: options.surfaceDualContouring,
      surfaceDualMarchingCubes: options.surfaceDualMarchingCubes,
      cellSize: this.cellSize,
      worldOrigin: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
      finestLevel: plan.maximumDepth,
      primitiveCapacity: OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY,
      dirtyRegionCapacity: OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY,
      dirtyBrickCapacity: this.tree.leafCapacity,
      candidatesPerDirtyBrick: OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
      clusterCapacity: OCTREE_LIVE_SCENE_CLUSTER_CAPACITY,
      fieldProgramCapacity: OCTREE_LIVE_SCENE_FIELD_PROGRAM_CAPACITY,
      solidWorld: residualSolidWorld,
      solidWorldLattice: {
        origin_m: [-0.5 * scene.container.width_m, 0,
          -0.5 * scene.container.depth_m],
        cellSize_m: sceneCellSizes_m(scene),
      },
      solidSurface: scene.solidSurface?.kind === "swept-flume" ? sweptFlumeSurfaceWGSL(scene.solidSurface)
        : scene.solidSurface && BATH_SOLID_DISTANCE_WGSL,
      renderTerrain,
      backdropDetail: backdropDetail && backdropTerrainTable
        ? backdropDetailVoxelizerWGSL(backdropTerrainTable, backdropDetail, options.surfaceDualMarchingCubes === true) : undefined,
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
      detail: `Live SVO derived lighting needs ${liveDerivedLevelCount} mip levels; the runtime supports at most ${SVO_NODE_MIP_MAXIMUM_LEVELS}`,
      requiredPages: 0,
      capacity: maximumDerivedPages,
    };
    if (liveDerivedLevelCount <= SVO_NODE_MIP_MAXIMUM_LEVELS) {
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
          // The pages the leaf reserve was priced with, so that the two refuse
          // together and a reserve that covers the domain plans totally.
          maximumReserveBytes: derivedReserveBytes,
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
        const atlasFits = mipPlan.atlas.texels.every((value) => value <= device.limits.maxTextureDimension3D);
        if (!mipPlan.complete || !atlasFits || mipPlan.residentPageCount === 0) {
          throw new RangeError("Live SVO derived-page capacity cannot cover the declared editable domain");
        }
        // The page table is a hash sized from the address plan's page capacity
        // — occupancy plus the growth reserve — not from the domain's page grid,
        // so a wide world costs what it holds. Throws past the device's 2D limit.
        svoNodeMipPageHashShape(addressPlan.pageCapacity, device.limits.maxTextureDimension2D);
        /**
         * The wall. Everything above it in this block — the opacity floor, the
         * base-page seeds, the address plan and the page-hash shape — is pure
         * CPU and owns nothing, which is why it can be sliced. Everything below
         * creates device resources that stay locals until the four assignments
         * at the end of the block, so a build abandoned past this point would
         * leak an atlas nothing can reach. No yield from here to there.
         */
        yield;
        nodeMipPyramid = new WebGpuLiveSvoNodeMipPyramid(device, {
          pageCapacity: addressPlan.pageCapacity,
          atlasTexels: addressPlan.atlasTexels as [number, number, number],
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
        const sceneJournal = sceneMaintenance.journal;
        if (!sceneJournal) throw new Error("Live derived lighting needs the voxelizer's dirty journal");
        // A budgeted chunk journals at most its budget; an unbudgeted world can
        // journal every brick it holds in one frame.
        const sceneBrickBudget = octreeLiveSceneBrickBudget();
        this.liveDerivedChunkSeedBound = sceneBrickBudget > 0
          ? Math.min(sceneBrickBudget, sceneMaintenance.dirtyBrickCapacity) : sceneMaintenance.dirtyBrickCapacity;
        const liveDerivedRadianceFeedback = options.radianceFeedback ?? LIVE_SVO_RADIANCE_FEEDBACK.enabledByDefault;
        const liveDerivedGenerationSource = {
          buffer: this.structuralPublicationState,
          offsetBytes: this.tree.structuralPublicationOffsetBytes
            + SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration * 4,
        };
        liveDerivedPlanner = new WebGpuLiveSvoDerivedWorklistPlanner(device, {
          tree: this.tree,
          nodeMips: nodeTarget,
          // The voxelizer's journal, taken with the planner's own cursor: a leaf
          // is lit once however many chunk frames its revision takes, where
          // the revision's dirty list was planned whole on every one of them.
          sceneJournal: {
            buffer: sceneMaintenance.buffer,
            writeCursorOffsetBytes: sceneJournal.cursorOffsetBytes,
            recordOffsetBytes: sceneJournal.recordOffsetBytes,
            recordStrideWords: sceneJournal.recordStrideWords,
            capacity: sceneJournal.capacity,
            requestedRevisionOffsetBytes: sceneMaintenance.stateOffsetBytes
              + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.requestedRevision * 4,
            completedRevisionOffsetBytes: sceneMaintenance.stateOffsetBytes
              + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision * 4,
          },
          stepLeafSources: [
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
          radiance: radianceTarget,
          radianceFeedback: liveDerivedRadianceFeedback,
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
          batch: liveDerivedPlanner.batch,
          levelCount: liveDerivedLevelCount,
          generationSource: liveDerivedGenerationSource,
          // Every addressable slot, not just the occupied ones: the empty
          // certification has to reach a reserved slot before a grown plan can
          // hand it to a page, or the first frame after growth reads garbage.
          plannedPageCount: addressPlan.pageCapacity,
          finestLevel: plan.maximumDepth,
          worldOrigin_m: this.sceneWorldOrigin,
          cellSize_m: this.cellSize,
          radianceFeedback: liveDerivedRadianceFeedback,
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
        console.error("[svo] live derived lighting unavailable; exact visibility will be used", error);
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
    // Counted where the leaves were walked for cover.
    const terminalCounts = { voxel: voxelTerminalLeaves, planarBoundary: planarTerminalLeaves };
    this.planarTerminalsBaked = planarSources.length > 0 || renderTerrain !== undefined;
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
      sceneMaintenance: this.proxyVoxelizer.maintenanceBinding,
      capacities: { nodes: this.tree.nodeCapacity, leaves: this.tree.leafCapacity, voxels: this.tree.voxelCapacity },
      terminalCounts,
      planarBoundaries: {
        records: { buffer: planarBoundaries, size: planarBoundaries.size },
        count: planarSources.length,
        strideBytes: PLANAR_BOUNDARY_PATCH_BYTES,
        generation: 1,
      },
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
    if (this.planarTerminalsBaked) {
      throw new Error("A sparse world with planar terminals must be rebuilt to rescale its render domain");
    }
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
  /**
   * The render copy of one solid snapshot: canonical shell left out, never an
   * editable patch promoted to an immutable planar terminal.
   *
   * An edit asks twice — once to be validated, once to be staged — and the
   * residual is kept per snapshot, so the second answer is the first. A stroke's
   * residual is the published one extended, not the document replayed.
   */
  private renderResidualFor(scene: SceneDescription, world: SolidWorld): SvoRenderResidual {
    return svoRenderResidual(scene, world, this.renderResidual);
  }

  /** What changed between two snapshots, compared once per pair. */
  private solidChangesBetween(scene: SceneDescription, previous: SolidWorld, next: SolidWorld,
    sameTerrain: boolean): SolidWorldChangeBounds {
    const memo = this.solidChangeMemo;
    if (memo?.previous === previous && memo.next === next) return memo.changes;
    const changes = solidWorldChangeBounds(scene, previous, next, { sameTerrain });
    this.solidChangeMemo = { previous, next, changes };
    return changes;
  }

  validateLiveSolidEdit(scene: SceneDescription): void {
    if (this.destroyed) throw new Error("Sparse scene has been destroyed");
    if (terrainFieldStamp(scene) !== this.terrainFieldStamp) {
      throw new Error("Changing terrain heights or its lattice requires rebuilding the scene; voxel overlays remain live.");
    }
    const world = solidWorldForScene(scene);
    // Capacity is the display's, so it is asked of what the display is sent:
    // the residual, not the physical world with its shell still on.
    this.proxyVoxelizer.validateSolidWorld(this.renderResidualFor(scene, world).residual);
    // The ground was just shown to be this world's own, so an appended stroke
    // is compared over its own pages.
    const changes = this.solidChangesBetween(scene, this.solidWorld, world, true);
    const bounds = scene.terrain ? changes.dirtyBounds : changes.addedBounds;
    for (const bound of bounds) {
      if (bound.minimum.some((value, axis) => value < this.sceneWorldOrigin[axis]! - 1e-6)
        || bound.maximum.some((value, axis) => value > this.sceneWorldOrigin[axis]!
          + this.sceneBrickDimensions[axis]! * this.brickSize * this.cellSize[axis]! + 1e-6)) {
        throw new RangeError("Edit exceeds this scene's live world bounds. Use a scene with a larger authored domain.");
      }
    }
    this.assertLiveTopologyCapacity(bounds, [world, this.renderResidualFor(scene, world).residual]);
  }

  /**
   * Bring the world up to a document, at a cost set by what separates it from
   * the last one staged.
   *
   * The same document is nothing. A document whose solids and authored shapes
   * stand where they stood, but whose materials, ground model or lights moved,
   * is byte-range writes into the tables shading already reads: no region is
   * dirtied and no revision is minted. Only a solid that moved produces a
   * revision, and it produces exactly one.
   *
   * Returns whether anything the GPU holds changed.
   */
  stageSceneUpdate(scene: SceneDescription): boolean {
    if (this.destroyed) throw new Error("Cannot update a destroyed sparse-brick world");
    // Accepting an edit and drawing its first frame hand over the same
    // document, and the second has nothing to add.
    if (scene === this.stagedScene) return false;
    const initialPublication = this.sceneRevision === 0;
    const previousSolidWorld = this.solidWorld;
    const nextSolidWorldStamp = solidWorldContentStamp(scene);
    const solidWorldStampChanged = nextSolidWorldStamp !== this.solidWorldStamp;
    const solidWorldChanged = initialPublication
      || solidWorldStampChanged;
    const nextSolidWorld = solidWorldStampChanged
      ? solidWorldForScene(scene) : previousSolidWorld;
    // Staging, unlike validation, is also handed scenes whose ground has moved
    // while their rebuild is in flight; only the same ground makes a stroke's
    // own pages the whole comparison.
    const groundMoved = !initialPublication && terrainFieldStamp(scene) !== this.terrainFieldStamp;
    const sameTerrain = solidWorldChanged && !initialPublication && !groundMoved;
    const changes = solidWorldChanged && !initialPublication
      ? this.solidChangesBetween(scene, previousSolidWorld, nextSolidWorld, sameTerrain)
      : { dirtyBounds: [], addedBounds: [] };
    const nextSolidBounds = initialPublication ? solidWorldPageBounds(scene, nextSolidWorld) : scene.terrain ? changes.dirtyBounds : changes.addedBounds;
    const dirtySolidBounds = initialPublication ? nextSolidBounds : changes.dirtyBounds;
    // The catalog of a document that only recoloured its predecessor shares
    // every untouched primitive with it, so what moved is found by identity and
    // the rest is not read.
    const catalog = successorEnvironmentCatalog(this.stagedDocument, scene, this.environmentDetailCellSize_m || undefined);
    const authored = catalog === this.stagedCatalog ? this.stagedAuthored : environmentProxyPrimitives(catalog, true);
    const stagedAuthored = this.stagedAuthored;
    let authoredMoved = authored.length !== stagedAuthored.length;
    const recoloured: EnvironmentProxyPrimitive[] = [];
    if (!authoredMoved && authored !== stagedAuthored) {
      for (let index = 0; index < authored.length; index += 1) {
        const next = authored[index]!, previous = stagedAuthored[index]!;
        if (next === previous) continue;
        if (!sameAuthoredGeometry(previous, next)) { authoredMoved = true; break; }
        if (!sameSceneValue(previous.material, next.material)) recoloured.push(next);
      }
    }
    const staticBodies = scene.rigidBodies === this.stagedDocument?.rigidBodies
      ? this.stagedStaticBodies : staticRigidBodiesStamp(scene.rigidBodies);
    const primitivesMoved = authoredMoved || staticBodies !== this.stagedStaticBodies;
    // A keyed update left the live list ahead of its document, and the next
    // document is the authority again.
    const relist = primitivesMoved || this.stagedScene === undefined;
    const nextRenderResidual = solidWorldChanged
      ? this.renderResidualFor(scene, nextSolidWorld) : this.renderResidual;
    const nextResidualSolidWorld = nextRenderResidual?.residual ?? this.renderSolidWorld;
    // A cut reveals the rest of that shell slab; Undo hides the whole slab
    // again. Invalidate the presentation delta, not just the edited cells.
    const renderChanges = solidWorldChanged && !initialPublication
      ? this.solidChangesBetween(scene, this.renderSolidWorld, nextResidualSolidWorld, sameTerrain)
      : { dirtyBounds: [], addedBounds: [] };
    if (!initialPublication && this.planarTerminalsBaked && (primitivesMoved || groundMoved)) {
      throw new Error("Authored geometry affecting planar terminals requires a sparse-world rebuild");
    }
    const dirtyRegions: SparseSceneAxisAlignedBounds[] = [
      ...(initialPublication ? authored.map((primitive) => ({
        minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y,
          primitive.aabb_m.min.z] as const,
        maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y,
          primitive.aabb_m.max.z] as const,
      })) : []),
      ...dirtySolidBounds,
      ...renderChanges.dirtyBounds,
      ...(initialPublication && this.backdropDetailBounds ? [this.backdropDetailBounds] : []),
    ];
    const newBounds: SparseSceneAxisAlignedBounds[] = [...nextSolidBounds,
      ...renderChanges.addedBounds];
    const listed = relist ? this.listLivePrimitives(scene, authored, dirtyRegions, newBounds) : undefined;
    if (relist && ENVIRONMENT_VOXEL_MATERIAL_BASE + authored.length > OCTREE_LIVE_SCENE_MATERIAL_CAPACITY) {
      throw new RangeError(`Live scene needs ${ENVIRONMENT_VOXEL_MATERIAL_BASE + authored.length} materials but the fixed arena holds ${OCTREE_LIVE_SCENE_MATERIAL_CAPACITY}`);
    }
    // The bricks the publication activates, found while pricing them. A
    // relisted document's primitives are not the live ones yet, so its bricks
    // are found again once they are.
    let admitted: SparseBrickCoordinate[] | undefined;
    if (dirtyRegions.length > 0) {
      if (!initialPublication) {
        const missing = this.assertLiveTopologyCapacity(newBounds, [nextSolidWorld, nextResidualSolidWorld]);
        if (!listed) admitted = missing;
      }
      if (solidWorldChanged && !initialPublication) {
        this.proxyVoxelizer.setSolidWorld(nextResidualSolidWorld);
      }
      this.solidWorld = nextSolidWorld;
      this.renderSolidWorld = nextResidualSolidWorld;
      this.renderResidual = nextRenderResidual;
      this.solidWorldStamp = nextSolidWorldStamp;
    }
    if (listed) {
      this.liveScenePrimitiveStates = listed.states;
      this.liveScenePrimitives = listed.primitives;
      if (listed.changed) this.liveScenePrimitiveList = [...listed.primitives.values()].map(({ primitive }) => primitive);
    }
    const tablesWritten = this.stageAppearanceTables(scene, authored, recoloured, relist);
    this.stagedScene = scene;
    this.stagedDocument = scene;
    this.stagedCatalog = catalog;
    this.stagedAuthored = authored;
    this.stagedStaticBodies = staticBodies;
    if (dirtyRegions.length === 0) return tablesWritten;
    // Regions an unpublished revision still owes ride on this one. They are
    // read only here, where there is something new to publish with them: a
    // document that changes nothing leaves the pending revision as it stands.
    const pending = this.pendingScenePublication?.dirtyRegions;
    this.stagePrimitivePublication(this.advanceSceneRevision(),
      pending ? [...pending, ...dirtyRegions] : dirtyRegions, newBounds, initialPublication, admitted);
    return true;
  }

  /**
   * The live primitives a document lists, with the bounds of every one that
   * differs from what is listed now pushed onto `dirtyRegions` and `newBounds`.
   */
  private listLivePrimitives(
    scene: SceneDescription,
    authored: readonly EnvironmentProxyPrimitive[],
    dirtyRegions: SparseSceneAxisAlignedBounds[],
    newBounds: SparseSceneAxisAlignedBounds[],
  ): { primitives: Map<string, LiveScenePrimitiveEntry>; states: Map<string, LiveScenePrimitiveState>; changed: boolean } {
    const planarCatalog = buildSvoPlanarBoundaryCatalog(authored, primitive => ({
      materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
      ownerId: SCENE_ENVIRONMENT_OWNER_BASE + primitive.ownerIndex,
    }));
    const residualAuthored = svoPlanarResidualEnvironmentPrimitives(authored, planarCatalog);
    const liveEntries: LiveScenePrimitiveEntry[] = [
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
      }] : []),
      // The exact planar catalogue remains globally visible after a terminal
      // splits. Sampling those planes again would duplicate and dilate floors.
      ...residualAuthored.map((primitive) => ({
        key: primitive.key,
        primitive: sparseScenePrimitiveForProxy(primitive, {
          materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
          ownerId: SCENE_ENVIRONMENT_OWNER_BASE + primitive.ownerIndex,
        }),
      })),
    ];
    if (liveEntries.length > OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY) {
      throw new RangeError(`Live scene needs ${liveEntries.length} primitives but the fixed arena holds ${OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY}`);
    }
    const primitives = new Map<string, LiveScenePrimitiveEntry>();
    const states = new Map<string, LiveScenePrimitiveState>();
    const listedKeys = this.liveScenePrimitives.keys();
    let changed = liveEntries.length !== this.liveScenePrimitives.size;
    for (const entry of liveEntries) {
      primitives.set(entry.key, entry);
      // The same key in the same place is the same record slot downstream.
      if (listedKeys.next().value !== entry.key) changed = true;
      const previous = this.liveScenePrimitives.get(entry.key);
      const previousState = this.liveScenePrimitiveStates.get(entry.key);
      if (previous && previousState && sameSceneValue(previous.primitive, entry.primitive)) {
        primitives.set(entry.key, previous);
        states.set(entry.key, previousState);
        continue;
      }
      const state = { bounds: sparseScenePrimitiveBounds(entry.primitive) };
      states.set(entry.key, state);
      if (previousState) dirtyRegions.push(previousState.bounds);
      dirtyRegions.push(state.bounds); newBounds.push(state.bounds);
      changed = true;
    }
    for (const [key, previous] of this.liveScenePrimitiveStates) {
      if (!states.has(key)) dirtyRegions.push(previous.bounds);
    }
    return { primitives, states, changed };
  }

  /**
   * Bring the tables shading reads up to a document: the PBR table and the
   * emission column the live radiance builder takes from it, the authored
   * lights and the environment record.
   *
   * Every record keeps the slot its owner index names and the revision the
   * table was built at, so a recoloured surface is its own stride of words and
   * the words either side of it are not sent. Returns whether any were.
   *
   * These are the tables derived lighting gathers from, so a write restarts
   * the radiance feedback window: lighting converges on the new values over
   * the bricks already built, and no brick is voxelized again.
   */
  private stageAppearanceTables(
    scene: SceneDescription,
    authored: readonly EnvironmentProxyPrimitive[],
    recoloured: readonly EnvironmentProxyPrimitive[],
    relist: boolean,
  ): boolean {
    const queue = this.device.queue;
    const initial = this.pbrMaterialRecords.length === 0;
    const surfaceModel = sceneTerrainSurfaceModel(scene);
    const staged = this.stagedDocument;
    let written = 0;
    if (initial) {
      // What construction uploaded, which the first document is diffed against.
      this.pbrMaterialRecords = new Uint32Array(OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_WORDS);
      const count = writeOctreeSvoPbrMaterialTable(this.pbrMaterialRecords, authored, this.surfaceModel);
      this.pbrMaterialStaging = this.pbrMaterialRecords.slice();
      const emission = packLiveSvoMaterialEmission(this.pbrMaterialRecords, count);
      this.materialEmissionRecords = new Uint32Array(emission.buffer);
    }
    // The run of records to compare, in material IDs.
    let first = Infinity, end = 0;
    if (relist) {
      const count = writeOctreeSvoPbrMaterialTable(this.pbrMaterialStaging, authored, surfaceModel);
      first = 0; end = OCTREE_LIVE_SCENE_MATERIAL_CAPACITY;
      this.surfaceModel = surfaceModel;
      const source = (this as unknown as { sceneSource?: SparseVoxelSceneRenderSource }).sceneSource;
      if (source?.pbrMaterials && source.pbrMaterials.count !== count) {
        source.pbrMaterials = { ...source.pbrMaterials, count };
        source.materialCount = count;
      }
    } else {
      if (surfaceModel !== this.surfaceModel) {
        // The model chooses closures and nothing else: the built-in records
        // it reaches, and one word of every authored primitive's.
        for (const record of buildDefaultSvoMaterialRecords(OCTREE_SVO_PBR_MATERIAL_REVISION, { terrainSurface: surfaceModel })) {
          writeSvoMaterialRecord(this.pbrMaterialStaging, record);
        }
        first = 0;
        for (const primitive of authored) {
          const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
          writeSvoMaterialFunctionId(this.pbrMaterialStaging, materialId, svoMaterialFunctionIdForEnvironmentProxy(primitive, surfaceModel));
          end = Math.max(end, materialId + 1);
        }
        end = Math.max(end, ENVIRONMENT_VOXEL_MATERIAL_BASE);
        this.surfaceModel = surfaceModel;
      }
      for (const primitive of recoloured) {
        const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
        writeSvoEnvironmentMaterialRecord(this.pbrMaterialStaging, materialId, primitive.material,
          OCTREE_SVO_PBR_MATERIAL_REVISION, svoMaterialFunctionIdForEnvironmentProxy(primitive, surfaceModel));
        first = Math.min(first, materialId); end = Math.max(end, materialId + 1);
      }
    }
    if (end > first) {
      const from = first * SVO_MATERIAL_RECORD_WORDS, to = end * SVO_MATERIAL_RECORD_WORDS;
      const wrote = uploadChangedWords(queue, this.pbrMaterialBuffer, from * 4,
        this.pbrMaterialRecords.subarray(from, to), this.pbrMaterialStaging.subarray(from, to));
      if (wrote > 0) {
        // The emission column is words 4..6 of each record, one vec4 a material.
        const emission = new Uint32Array((end - first) * 4);
        for (let material = first; material < end; material += 1) {
          const record = material * SVO_MATERIAL_RECORD_WORDS + 4;
          emission.set(this.pbrMaterialStaging.subarray(record, record + 3), (material - first) * 4);
        }
        written += wrote + uploadChangedWords(queue, this.materialEmissionBuffer, first * 16,
          this.materialEmissionRecords.subarray(first * 4, end * 4), emission);
      }
    }
    // A light is the sun the document names or an emitting primitive, so only
    // a document that moved one of those is asked for its lights again.
    const lightingMoved = initial || relist || scene.lighting !== staged?.lighting
      || scene.environment !== staged?.environment || scene.systems !== staged?.systems;
    if (lightingMoved || recoloured.length > 0) {
      // Fixtures are read from the document's own catalog, which a world drawn
      // at a finer detail cell has not derived yet.
      successorEnvironmentCatalog(staged, scene);
      const lights = buildOctreeSvoLightPublication(scene).packedRecords;
      if (initial) this.lightRecords = lights.slice();
      // The table and the builder's light count are sized at construction; a
      // lamp lit or put out changes the count and is not carried here.
      else if (lights.length === this.lightRecords.length) {
        written += uploadChangedWords(queue, this.lightBuffer, 0, this.lightRecords, lights);
      }
    }
    if (lightingMoved) {
      const environment = buildOctreeSvoEnvironmentLightingPublication(scene).packedRecords;
      if (initial) this.environmentLightingRecords = environment.slice();
      else written += uploadChangedWords(queue, this.environmentLightingBuffer, 0, this.environmentLightingRecords, environment);
    }
    // Light that moved changes every floor page's radiance and no page's
    // opacity: the next lighting frame opens the feedback window everywhere.
    if (written > 0) this.liveDerivedRelight = true;
    return written > 0;
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
      if (sameSceneValue(previousEntry.primitive, update.primitive)) continue;
      const bounds = sparseScenePrimitiveBounds(update.primitive);
      dirtyRegions.push(previousState.bounds, bounds); newBounds.push(bounds); changed = true;
      this.liveScenePrimitives.set(update.key, { ...previousEntry, primitive: update.primitive });
      this.liveScenePrimitiveStates.set(update.key, { bounds });
    }
    if (!changed) return false;
    this.liveScenePrimitiveList = [...this.liveScenePrimitives.values()].map(({ primitive }) => primitive);
    // The list no longer matches any document, so the next one is listed whole.
    this.stagedScene = undefined;
    const revision = this.advanceSceneRevision();
    this.stagePrimitivePublication(revision, dirtyRegions, newBounds, false);
    return true;
  }

  private advanceSceneRevision(): number {
    this.sceneRevision = this.sceneRevision === 0xffff_ffff ? 1 : this.sceneRevision + 1;
    return this.sceneRevision;
  }

  /**
   * The sampled terrain terminals an edit splits, whole, so that every sibling
   * the split preserves is voxelized again.
   *
   * Only a world built with sampled terrain terminals has any, which a smooth
   * world never is. The 4,096 is a bound on one publication's work, not on any
   * arena: the bricks walked here on the host, then the bricks of the split
   * terminals the voxelizer samples again. A second stroke starts a new count.
   */
  private terrainResamplingBounds(bounds: readonly SparseSceneAxisAlignedBounds[]): SparseSceneAxisAlignedBounds[] {
    const result: SparseSceneAxisAlignedBounds[] = [];
    if (!this.sampledTerrainNodes?.size) return result;
    let touched = 0;
    for (const bound of bounds) {
      touched += bound.minimum.reduce((volume, value, axis) => volume
        * (Math.ceil((bound.maximum[axis]! - value) / (this.cellSize[axis]! * this.brickSize)) + 2), 1);
      if (touched > 4096) throw new RangeError("Terrain edit exceeds the bounded 4096-brick resampling budget. Release and continue in a second stroke.");
    }
    const coordinates = liveSceneMissingBrickCoordinates(bounds, this.sceneWorldOrigin,
      this.cellSize, this.brickSize, this.sceneBrickDimensions, () => false);
    const ancestors = new Set<string>();
    let budget = 0;
    for (const coordinate of coordinates) for (let level = 0; level < this.finestLevel; level++) {
      const shift = this.finestLevel - level;
      const key = `${level}:${coordinate.x >>> shift},${coordinate.y >>> shift},${coordinate.z >>> shift}`;
      if (ancestors.has(key)) continue;
      const ancestor = this.sampledTerrainNodes.get(key);
      if (!ancestor) continue;
      ancestors.add(key);
      budget += ancestor.minimum.reduce((volume, value, axis) => volume
        * Math.ceil((ancestor.maximum[axis]! - value) / (this.cellSize[axis]! * this.brickSize)), 1);
      if (budget > 4096) throw new RangeError("Terrain edit exceeds the bounded 4096-brick resampling budget. Release and continue in a second stroke.");
      result.push(ancestor);
    }
    return result;
  }

  /**
   * Refuse an edit the reserve cannot hold, before anything is taken.
   *
   * `worlds` are the solid worlds of the document being asked about: the
   * staged ones when the document is already staged, the proposal's own when
   * it is only being validated.
   *
   * Returns the bricks the edit activates, which it had to find to price them.
   */
  private assertLiveTopologyCapacity(
    bounds: readonly SparseSceneAxisAlignedBounds[],
    worlds: readonly SolidWorld[] = [this.solidWorld, this.renderSolidWorld],
  ): SparseBrickCoordinate[] {
    this.terrainResamplingBounds(bounds);
    const missing = this.liveBricksToActivate(bounds, worlds);
    // Net demand, from the host's own tree: what the new bricks take, against
    // what is free and what the publication takes back before it activates.
    // The second is counted only by an edit that needs it.
    const demand = this.topologyMirror.demand(missing);
    const free = this.topologyMirror.freeLeaves;
    const reclaimed = demand.leaves <= free ? 0 : this.reclaimableTopologyBricks(worlds);
    if (demand.leaves > free + reclaimed) {
      throw new RangeError(`Edit exceeds the reserved live voxel detail capacity: its ${missing.length} new bricks `
        + `need ${demand.leaves} leaves and this scene has ${free + reclaimed} left. `
        + "Erase or undo detail to free some, or save and reopen the scene to prepare more.");
    }
    // The node arena holds a block for every ancestor of every reserved leaf,
    // so it cannot run out first; a tree that says otherwise is not the tree
    // the reserve was priced for.
    if (demand.blocks > this.topologyMirror.freeBlocks + this.finestLevel * reclaimed) {
      throw new RangeError(`Edit exceeds the reserved live voxel detail capacity: its new bricks need ${demand.blocks} `
        + `node blocks and this scene has ${this.topologyMirror.freeBlocks} left.`);
    }
    // The worklist holds a batch that spends the whole reserve and takes as
    // much back, so only batches no frame has taken up yet can crowd it. A
    // brick taken back writes nine records at most.
    const records = this.topologyMirror.pendingRecords + demand.records + 9 * this.releasableTopologyBricks.size;
    if (records > this.topologyMutationCapacity) {
      throw new RangeError(`Edit exceeds the reserved live voxel detail capacity: ${records} topology records are waiting `
        + `on one publication, which carries ${this.topologyMutationCapacity}. Release and continue in a second stroke.`);
    }
    // Lighting pages are reserved with the leaves, but by the page: bricks
    // scattered one to a parent can run out of pages with leaves to spare.
    const growth = this.planLiveDerivedGrowth(missing.map(coordinate => this.liveDerivedBasePage(coordinate)));
    if (growth && !growth.fits) {
      throw new RangeError(`Edit exceeds the reserved live voxel detail capacity: its new bricks need lighting pages the `
        + `scene did not reserve (${growth.refusal}). Save and reopen the scene to prepare more detail.`);
    }
    return missing;
  }

  /** The finest node-mip page a brick is lit from. */
  private liveDerivedBasePage(coordinate: SparseBrickCoordinate): SvoNodeMipCoordinate {
    const pagesPerBrick = this.brickSize / SVO_NODE_MIP_LAYOUT.interiorSize;
    return [Math.floor(coordinate.x * pagesPerBrick), Math.floor(coordinate.y * pagesPerBrick),
      Math.floor(coordinate.z * pagesPerBrick)];
  }

  private stagePrimitivePublication(
    revision: number,
    dirtyRegions: readonly SparseSceneAxisAlignedBounds[],
    newBounds: readonly SparseSceneAxisAlignedBounds[],
    initialPublication: boolean,
    admitted?: readonly SparseBrickCoordinate[],
  ): void {
    const missing = initialPublication ? [] : admitted ?? this.assertLiveTopologyCapacity(newBounds);
    const coalescedDirty = coalesceDirtyRegions([...dirtyRegions, ...(!initialPublication ? this.terrainResamplingBounds(newBounds) : [])], initialPublication ? 1 : OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY);
    this.pendingScenePublication = {
      // The same array until a primitive in it changes, which is how the
      // voxelizer knows its record tables already hold this list.
      primitives: this.liveScenePrimitiveList,
      dirtyRegions: coalescedDirty,
      revision,
      // A caller that encodes directly needs the first publication atomically.
      // Everything downstream reads a scene generation, and that generation only
      // advances when a revision completes; spreading the first one over frames
      // would leave every consumer reading zero, which is the state a black slab
      // across the container footprint is made of. Edits are a different case:
      // there is already a current generation to keep showing while the next
      // converges. initializeScenePublication instead drains budgeted chunks
      // before exposing the world, keeping the same publication contract.
      budgeted: !initialPublication,
    };
    if (!initialPublication) {
      this.raiseBackdropContentCeiling(newBounds);
      // Take back first, so the leaves, nodes and lighting pages this document
      // gives up are the first it takes.
      this.releaseUnclaimedTopologyBricks(coalescedDirty);
      const activatedPages: SvoNodeMipCoordinate[] = [];
      // Ascending Morton order, and release descending: slots are taken and
      // returned as a stack, so an edit taken back returns its own. What was
      // just taken back held nothing of this document, so the bricks priced
      // above are still the ones to take.
      const coordinates = [...missing].sort(compareSparseBrickMorton);
      for (const coordinate of coordinates) {
        this.topologyMirror.activate(coordinate);
        // A brick under a split terminal is the terminal's own leaf and stays.
        if (this.topologyMirror.releasable(coordinate)) {
          this.releasableTopologyBricks.set(brickCoordinateKey(coordinate), coordinate);
        }
        activatedPages.push([coordinate.x, coordinate.y, coordinate.z]
          .map((value) => Math.floor(value * this.brickSize / SVO_NODE_MIP_LAYOUT.interiorSize)) as unknown as SvoNodeMipCoordinate);
      }
      this.growLiveDerivedAddressPlan(activatedPages);
    }
    if (!this.topologyMirror.hasPendingRecords) {
      this.device.queue.writeBuffer(this.topologyMutationWorklist, 0,
        new Uint32Array(SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords));
      this.pendingTopologyRecords = 0;
      return;
    }
    // The whole batch that takes the GPU's tree to the host's, whatever was
    // staged since the last frame: the preflight admitted no more than the
    // worklist holds, and the packer throws rather than leave a record behind.
    const packed = this.topologyMirror.pack(revision);
    this.device.queue.writeBuffer(this.topologyMutationWorklist, 0, packed);
    this.pendingTopologyRecords = packed[SPARSE_BRICK_TOPOLOGY_MUTATION.countWord]!;
    // Only a topology-changing transaction invalidates topology-keyed wide,
    // compact, and renderer caches. Covered primitive motion leaves it stable.
    this.sceneSource.revision = revision;
  }

  /**
   * Raise the content ceiling over an edit's new boxes and upload only the
   * rows it touched (plus the highest-top word). The initial publication needs
   * none: the plan's leaves already cover it.
   */
  private raiseBackdropContentCeiling(bounds: readonly SparseSceneAxisAlignedBounds[]): void {
    const ceiling = this.backdropContentCeiling;
    if (!ceiling || bounds.length === 0) return;
    let first = Infinity, last = -Infinity;
    for (const bound of bounds) {
      const rows = raiseBackdropContentCeiling(ceiling, bound.minimum, bound.maximum);
      if (rows) { first = Math.min(first, rows[0]); last = Math.max(last, rows[1]); }
    }
    if (first > last) return;
    const tableBytes = this.tree.sceneMaterialOwnerOffsetBytes + this.tree.voxelCapacity * Uint32Array.BYTES_PER_ELEMENT;
    const sectionWord = this.backdropContentCeilingWord;
    this.device.queue.writeBuffer(this.tree.payload, tableBytes + (sectionWord + 3) * 4, new Float32Array([ceiling.highest_m]));
    this.device.queue.writeBuffer(this.tree.payload,
      tableBytes + (sectionWord + BACKDROP_CONTENT_CEILING_HEADER_WORDS + first * ceiling.cells) * 4,
      ceiling.tops.slice(first * ceiling.cells, (last + 1) * ceiling.cells));
    const tileFirst = Math.floor(first / BACKDROP_CONTENT_CEILING_TILE_CELLS), tileLast = Math.floor(last / BACKDROP_CONTENT_CEILING_TILE_CELLS);
    this.device.queue.writeBuffer(this.tree.payload,
      tableBytes + (sectionWord + BACKDROP_CONTENT_CEILING_HEADER_WORDS + ceiling.tops.length + tileFirst * BACKDROP_CONTENT_CEILING_TILES) * 4,
      ceiling.tiles.slice(tileFirst * BACKDROP_CONTENT_CEILING_TILES, (tileLast + 1) * BACKDROP_CONTENT_CEILING_TILES));
  }

  /**
   * Whether some leaf already covers this finest brick: a walk from the root
   * of the host's tree. Analytic coverage cannot store authored voxel detail,
   * so a brick under a terminal an edit can split is not covered.
   */
  private sceneBrickCovered(coordinate: SparseBrickCoordinate): boolean {
    return this.topologyMirror.covered(coordinate);
  }

  /**
   * Whether anything still needs a leaf at this finest brick. A leaf's voxels
   * are a function of the scene within one cell of its box, which is as far as
   * the voxelizer samples for the dual grid, so the brick is weighed with that
   * rim: a brick with nothing in it voxelizes empty and meshes to nothing,
   * while one whose neighbour's surface comes within a cell of its face keeps
   * the leaf that surface is stitched to. The dual grid emits a cell only where
   * all eight of its voxels are in leaves, so that brick is needed on every
   * side, and it is the same test that gives a brick its leaf and takes it
   * back: the edit leaves are a function of the document, not of its history.
   *
   * What claims it is a live primitive whose bounds reach it, or the solid
   * world. Off terrain that is a solid cell. Over terrain the ground is drawn
   * without leaves, so it is a cell an edit shaped that is solid, or that the
   * ground would have filled.
   */
  private liveBrickClaimed(
    coordinate: SparseBrickCoordinate,
    primitives: readonly SparseSceneAxisAlignedBounds[],
    worlds: readonly SolidWorld[] = [this.solidWorld, this.renderSolidWorld],
  ): boolean {
    if (this.liveBrickReaches(coordinate, primitives)) return true;
    const brick = [coordinate.x, coordinate.y, coordinate.z];
    const low = [0, 0, 0], high = [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      const edge = this.brickSize * this.cellSize[axis]!;
      const minimum = this.sceneWorldOrigin[axis]! + brick[axis]! * edge - this.cellSize[axis]!;
      const maximum = minimum + edge + 2 * this.cellSize[axis]!;
      low[axis] = Math.floor((minimum - this.solidLatticeOrigin[axis]!) / this.solidLatticeCell[axis]! + 1e-6);
      high[axis] = Math.ceil((maximum - this.solidLatticeOrigin[axis]!) / this.solidLatticeCell[axis]! - 1e-6);
    }
    for (let index = 0; index < worlds.length; index += 1) {
      if (worlds.indexOf(worlds[index]!) === index && solidWorldHoldsContent(worlds[index]!, low, high)) return true;
    }
    return false;
  }

  /**
   * The finest bricks these bounds reach that have no leaf and hold something
   * of the document `worlds` belong to. Reached as a leaf is claimed, rim and
   * all, so that the brick beside new content gets the leaf its surface is
   * stitched through; a box's corner brick that holds nothing takes none.
   */
  private liveBricksToActivate(
    bounds: readonly SparseSceneAxisAlignedBounds[],
    worlds: readonly SolidWorld[] = [this.solidWorld, this.renderSolidWorld],
  ): SparseBrickCoordinate[] {
    const { spans } = this.liveBrickSpans(bounds);
    const [width, height] = this.sceneBrickDimensions;
    const seen = new Set<number>();
    const missing: SparseBrickCoordinate[] = [];
    eachSpanBrick(spans, (x, y, z) => {
      const index = x + width * (y + height * z);
      if (seen.has(index)) return;
      seen.add(index);
      const coordinate = { x, y, z };
      if (!this.sceneBrickCovered(coordinate)) missing.push(coordinate);
    });
    const primitives = this.livePrimitiveBoundsNear(missing);
    return missing.filter(coordinate => this.liveBrickClaimed(coordinate, primitives, worlds));
  }

  /**
   * Each region as the span of finest bricks whose rimmed box it reaches,
   * `[x0, y0, z0, x1, y1, z1]` a region, and how many bricks the spans hold.
   */
  private liveBrickSpans(regions: readonly SparseSceneAxisAlignedBounds[]): { spans: Int32Array; volume: number } {
    const spans = new Int32Array(regions.length * 6);
    let volume = 0;
    for (let index = 0; index < regions.length; index += 1) {
      const { minimum, maximum } = regions[index]!;
      let cells = 1;
      for (let axis = 0; axis < 3; axis += 1) {
        const cell = this.cellSize[axis]!, edge = this.brickSize * cell, origin = this.sceneWorldOrigin[axis]!;
        const low = Math.max(0, Math.floor((minimum[axis]! - cell - origin) / edge - 1e-6));
        const high = Math.min(this.sceneBrickDimensions[axis]! - 1, Math.floor((maximum[axis]! + cell - origin) / edge + 1e-6));
        spans[index * 6 + axis] = low;
        spans[index * 6 + 3 + axis] = high;
        cells *= Math.max(0, high - low + 1);
      }
      volume += cells;
    }
    return { spans, volume };
  }

  /**
   * The live primitives' bounds that reach the box around these bricks: the
   * only ones that can claim any of them, found once for the whole set.
   */
  private livePrimitiveBoundsNear(bricks: Iterable<SparseBrickCoordinate>): SparseSceneAxisAlignedBounds[] {
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    for (const { x, y, z } of bricks) {
      if (x < low[0]!) low[0] = x;
      if (y < low[1]!) low[1] = y;
      if (z < low[2]!) low[2] = z;
      if (x > high[0]!) high[0] = x;
      if (y > high[1]!) high[1] = y;
      if (z > high[2]!) high[2] = z;
    }
    const near: SparseSceneAxisAlignedBounds[] = [];
    if (low[0] === Infinity) return near;
    for (let axis = 0; axis < 3; axis += 1) {
      const edge = this.brickSize * this.cellSize[axis]!;
      low[axis] = this.sceneWorldOrigin[axis]! + low[axis]! * edge - this.cellSize[axis]!;
      high[axis] = this.sceneWorldOrigin[axis]! + (high[axis]! + 1) * edge + this.cellSize[axis]!;
    }
    for (const { bounds } of this.liveScenePrimitiveStates.values()) {
      const { minimum, maximum } = bounds;
      if (minimum[0] <= high[0]! && minimum[1] <= high[1]! && minimum[2] <= high[2]!
        && maximum[0] >= low[0]! && maximum[1] >= low[1]! && maximum[2] >= low[2]!) near.push(bounds);
    }
    return near;
  }

  /** Whether a finest brick's box, with its one-cell rim, reaches any of the regions. */
  private liveBrickReaches(coordinate: SparseBrickCoordinate, regions: readonly SparseSceneAxisAlignedBounds[]): boolean {
    const origin = this.sceneWorldOrigin, cell = this.cellSize, size = this.brickSize;
    const x0 = origin[0] + coordinate.x * size * cell[0] - cell[0], x1 = x0 + (size + 2) * cell[0];
    const y0 = origin[1] + coordinate.y * size * cell[1] - cell[1], y1 = y0 + (size + 2) * cell[1];
    const z0 = origin[2] + coordinate.z * size * cell[2] - cell[2], z1 = z0 + (size + 2) * cell[2];
    for (let index = 0; index < regions.length; index += 1) {
      const { minimum, maximum } = regions[index]!;
      if (x0 <= maximum[0] && x1 >= minimum[0] && y0 <= maximum[1] && y1 >= minimum[1]
        && z0 <= maximum[2] && z1 >= minimum[2]) return true;
    }
    return false;
  }

  /**
   * How many edit-activated bricks hold nothing of the document `worlds`
   * belong to: the leaves staging that document takes back before it takes any.
   */
  private reclaimableTopologyBricks(worlds: readonly SolidWorld[]): number {
    const primitives = this.livePrimitiveBoundsNear(this.releasableTopologyBricks.values());
    let bricks = 0;
    for (const coordinate of this.releasableTopologyBricks.values()) {
      if (!this.liveBrickClaimed(coordinate, primitives, worlds)) bricks += 1;
    }
    return bricks;
  }

  /**
   * Take the edit-activated bricks the staged document no longer claims out
   * of the tree: each leaf back to the free list, each node left with nothing
   * beneath it after it, and the lighting pages they were lit from back to
   * their reserve. Everything that can change a brick's claim lies in a dirty
   * region, so only the bricks those reach are weighed. Descending Morton
   * order, the reverse of activation, so that an edit taken back whole returns
   * exactly the slots it took.
   *
   * The leaf goes in the publication that empties it, not after: a ray skips a
   * leaf slot the moment it is retired or handed to another brick, the mesh
   * drops what it held for the slot's old brick, and the voxelizer writes a
   * reused slot before anything reads its voxels.
   */
  private releaseUnclaimedTopologyBricks(regions: readonly SparseSceneAxisAlignedBounds[]): void {
    const bricks = this.releasableTopologyBricks;
    if (bricks.size === 0 || regions.length === 0) return;
    const { spans, volume } = this.liveBrickSpans(regions);
    // Whichever is least work: the regions' bricks looked up among the edit's
    // bricks, those looked up among the regions' bricks, or each of them
    // tested against a few large regions.
    const reached: SparseBrickCoordinate[] = [];
    const [width, height] = this.sceneBrickDimensions;
    if (volume < bricks.size) {
      const seen = new Set<string>();
      eachSpanBrick(spans, (x, y, z) => {
        const key = `${x},${y},${z}`, coordinate = bricks.get(key);
        if (coordinate && !seen.has(key)) { seen.add(key); reached.push(coordinate); }
      });
    } else if (volume < bricks.size * regions.length) {
      const cells = new Set<number>();
      eachSpanBrick(spans, (x, y, z) => { cells.add(x + width * (y + height * z)); });
      for (const coordinate of bricks.values()) {
        if (cells.has(coordinate.x + width * (coordinate.y + height * coordinate.z))) reached.push(coordinate);
      }
    } else {
      for (const coordinate of bricks.values()) {
        for (let base = 0; base < spans.length; base += 6) {
          if (coordinate.x >= spans[base]! && coordinate.x <= spans[base + 3]! && coordinate.y >= spans[base + 1]!
            && coordinate.y <= spans[base + 4]! && coordinate.z >= spans[base + 2]! && coordinate.z <= spans[base + 5]!) {
            reached.push(coordinate);
            break;
          }
        }
      }
    }
    const primitives = this.livePrimitiveBoundsNear(reached);
    const unclaimed = reached.filter(coordinate => !this.liveBrickClaimed(coordinate, primitives))
      .sort(compareSparseBrickMorton);
    const released: SparseBrickCoordinate[] = [];
    for (let index = unclaimed.length - 1; index >= 0; index -= 1) {
      const coordinate = unclaimed[index]!;
      if (!this.topologyMirror.release(coordinate)) continue;
      this.releasableTopologyBricks.delete(brickCoordinateKey(coordinate));
      released.push(coordinate);
    }
    // After every leaf is out of the host's tree: a lighting page goes back
    // only when no brick under it is still covered.
    this.releaseLiveDerivedPages(released);
  }

  /**
   * What the pages an edit activates ask of the derived address plan: how many
   * lighting pages they add, and whether the reserve holds them. Nothing when
   * there is no plan to ask.
   *
   * Counted from those pages and the levels above them, so the preflight that
   * refuses an edit and the publication that applies it can each ask.
   */
  private planLiveDerivedGrowth(activatedPages: readonly SvoNodeMipCoordinate[]): SvoNodeMipAddressPlanGrowth | undefined {
    if (activatedPages.length === 0 || !this.liveDerivedAddressPlanValid) return undefined;
    const current = this.liveDerivedAddressPlan;
    if (!current || !this.nodeMipPyramid || !this.tetrahedralRadiance || !this.liveDerivedPlanner) return undefined;
    return current.measureGrowth(activatedPages);
  }

  /**
   * Address the lighting pages this edit activated.
   *
   * Each new page takes a free slot and one entry of the page hash, and those
   * entries are all that is uploaded: no resident page moves, so nothing
   * already built is owed again. The new pages are built by the next
   * maintenance, which reaches them the way it reaches any dirty page — the
   * planner walks every dirty leaf up through the hash, and the brick that
   * brought these pages is dirty.
   *
   * It cannot fail here. `assertLiveTopologyCapacity` asks the same growth of
   * the same pages before anything is reserved, and refuses the edit there.
   */
  private growLiveDerivedAddressPlan(activatedPages: readonly SvoNodeMipCoordinate[]): void {
    if (!this.planLiveDerivedGrowth(activatedPages)) return;
    this.liveDerivedLastGrowth = this.liveDerivedAddressPlan!.grow(activatedPages);
    this.uploadLiveDerivedAddressChanges();
  }

  /**
   * Return the lighting pages of bricks an edit erased, so the reserve they
   * came from can be spent again.
   *
   * Under an opacity floor one page lights several bricks, so a brick's page is
   * returned only once every brick it lights is uncovered: the caller must have
   * already dropped these bricks from the covered set.
   */
  releaseLiveDerivedPages(coordinates: readonly SparseBrickCoordinate[]): number {
    const plan = this.liveDerivedAddressPlan;
    if (!plan || !this.liveDerivedAddressPlanValid || !this.nodeMipPyramid || coordinates.length === 0) return 0;
    const pagesPerBrick = this.brickSize / SVO_NODE_MIP_LAYOUT.interiorSize;
    const bricksPerFloorPage = Math.max(1, Math.round(2 ** plan.opacityFloorLevel / pagesPerBrick));
    const vacated: SvoNodeMipCoordinate[] = [];
    for (const coordinate of coordinates) {
      const first = { x: coordinate.x - coordinate.x % bricksPerFloorPage, y: coordinate.y - coordinate.y % bricksPerFloorPage,
        z: coordinate.z - coordinate.z % bricksPerFloorPage };
      let occupied = false;
      for (let z = 0; z < bricksPerFloorPage && !occupied; z += 1) {
        for (let y = 0; y < bricksPerFloorPage && !occupied; y += 1) {
          for (let x = 0; x < bricksPerFloorPage && !occupied; x += 1) {
            occupied = this.sceneBrickCovered({ x: first.x + x, y: first.y + y, z: first.z + z });
          }
        }
      }
      if (!occupied) vacated.push(this.liveDerivedBasePage(coordinate));
    }
    const released = plan.release(vacated);
    this.uploadLiveDerivedAddressChanges();
    return released;
  }

  /** Carry the address plan's pending changes to the device: one texel per changed hash entry and per new page. */
  private uploadLiveDerivedAddressChanges(): void {
    const plan = this.liveDerivedAddressPlan!;
    const nodeMips = this.nodeMipPyramid!;
    const radianceValidity = this.tetrahedralRadiance?.pageState.validity;
    plan.drainChanges(
      entry => nodeMips.writePageTableEntry(entry),
      slot => {
        invalidateLiveSvoDerivedPage(this.device, nodeMips.pageState.validity, slot);
        if (radianceValidity && slot >= plan.radianceSlotOffset) invalidateLiveSvoDerivedPage(this.device, radianceValidity, slot);
      });
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
    /** Lighting pages an edit may still add; returned pages count again. */
    remainingReservePages: number;
    /** What the last edit that added bricks cost: pages it added, and pages on its chains the builder owes. */
    lastGrowth?: { newPages: number; chainPages: number };
  } | undefined {
    const plan = this.liveDerivedAddressPlan;
    if (!plan) return undefined;
    return {
      valid: this.liveDerivedAddressPlanValid,
      total: plan.total,
      pageCapacity: plan.pageCapacity,
      activePages: plan.residentPageCount,
      reservePages: plan.reservePages,
      domainPyramidPages: plan.domainPyramidPageCount,
      basePages: plan.seedPageCount,
      domainBasePages: plan.domainBasePageCount,
      remainingReservePages: plan.remainingReservePages,
      lastGrowth: this.liveDerivedLastGrowth && {
        newPages: this.liveDerivedLastGrowth.newPages, chainPages: this.liveDerivedLastGrowth.chainPages },
    };
  }

  /** Voxelize a new renderer world before attachment, yielding the GPU between
   * bounded chunks. A single initial rebuild can take seconds on large scenes
   * and reset the browser GPU process even when the allocation fits. */
  async initializeScenePublication(
    signal?: AbortSignal,
    progress?: (completed: number, total: number) => void,
  ): Promise<void> {
    const assertActive = () => {
      if (signal?.aborted || this.destroyed) throw new DOMException("GPU initialization superseded", "AbortError");
    };
    assertActive();
    if (!this.pendingScenePublication) return;
    this.pendingScenePublication = { ...this.pendingScenePublication, budgeted: true };
    let completed = 0;
    do {
      assertActive();
      const encoder = this.device.createCommandEncoder({ label: "Initialize live scene voxelization chunk" });
      // Derived lighting reads the complete first generation. Build it only
      // after the last voxelization chunk has published that generation.
      this.encodeSceneMaintenance(encoder, true);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      assertActive();
      progress?.(++completed, this.proxyVoxelizer.budgetStatus.plannedChunks);
    } while (this.proxyVoxelizer.budgetStatus.converging);
    assertActive();
    const encoder = this.device.createCommandEncoder({ label: "Initialize live scene derived lighting" });
    this.encodeSceneMaintenance(encoder);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    assertActive();
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
    if (this.pendingTopologyRecords > 0) {
      // Two fixed dispatches scatter the staged records, however many bricks
      // they activate or retire. The host's tree is the GPU's from here.
      this.topologyMutator.encode(encoder, this.tree, {
        buffer: this.topologyMutationWorklist,
        capacity: this.topologyMutationCapacity,
      }, { generation: this.sceneRevision, recordCount: this.pendingTopologyRecords });
      this.topologyMirror.commit();
      this.pendingTopologyRecords = 0;
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
    if (!encoded) {
      // The settled path, and the one the render panel spends most of its life
      // reporting: nothing was voxelized this frame. Both remaining seams still
      // close, over nothing, so the manifest can say "reached, encoded no pass"
      // — which is a zero, and is what stops a silent row from inheriting the
      // source band's whole fence-partitioned wall.
      //
      // Lighting may still be owed: the queue drains and diffuse feedback
      // converges over the frames after the last revision, and then stops —
      // running forever makes a complex hero scene visibly breathe.
      seam?.("world-proxy-voxelize");
      if (!deferDerived) return this.encodeLiveDerivedMaintenance(encoder, seam);
      seam?.("world-derived-lighting");
      seam?.("world-radiance-feedback");
      return false;
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

  /** Withdraw radiance immediately; restoring it requires a complete ordered rebuild. */
  setRadianceEnabled(enabled: boolean): void {
    if (enabled === this.radianceEnabled) return;
    this.radianceEnabled = enabled;
    // Withdrawn here and restored by the rebuild's frame: the same object both
    // times, so only the withdrawal and the return rebind anything.
    this.liveDerivedRadianceSource ??= this.sceneSource.tetrahedralRadiance;
    this.sceneSource.tetrahedralRadiance = undefined;
    // The feedback window goes with the radiance; what opacity is owed stays.
    this.liveDerivedFramesRemaining = this.liveDerivedDrainFramesRemaining;
    if (enabled) this.liveDerivedInitial = true;
  }

  /**
   * One lighting frame, if one is owed. Returns whether it encoded anything.
   *
   * Every lighting frame is the same fixed set of direct dispatches in one
   * pass: the planner takes what the voxelizer journaled since it last looked,
   * then `roundsPerFrame` rounds each build at most the frame's remaining
   * allowance from the queue. What is queued, how far each cursor has moved and
   * which pages are inside a feedback window are GPU state the host never
   * reads; it counts frames from bounds it already holds.
   *
   * `physics` is the solver's step: its residency lists are seeded as well, and
   * rewritten every step, so they are taken only here.
   */
  private encodeLiveDerivedMaintenance(
    encoder: GPUCommandEncoder,
    seam?: RenderFrameSeam<"world">,
    physics = false,
  ): boolean {
    const planner = this.liveDerivedPlanner;
    const builder = this.liveDerivedBuilder;
    if (!this.liveDerivedAddressPlanValid || !planner || !builder) {
      seam?.("world-derived-lighting");
      seam?.("world-radiance-feedback");
      return false;
    }
    const radiance = this.radianceEnabled;
    const settleFrames = radiance && builder.radianceFeedbackEnabled ? LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount : 0;
    const initial = this.liveDerivedInitial;
    const activity = this.proxyVoxelizer.maintenanceBinding.activity ?? 0;
    if (initial) {
      // A first build lights every leaf and drains its own queue in one frame,
      // so nothing journaled before it is owed. The window counts it.
      this.liveDerivedSceneActivity = activity;
      this.liveDerivedSeedBound = 0;
      this.liveDerivedDrainFramesRemaining = 1;
      this.liveDerivedFramesRemaining = Math.max(1, settleFrames);
    } else {
      if (activity !== this.liveDerivedSceneActivity) {
        // The voxelizer ran since lighting last looked. Each of its frames
        // journaled at most a chunk, and no more leaves than the revision's
        // dirty regions cover; the window restarts behind the drain.
        const regionCells = this.proxyVoxelizer.indexStatus.regionCells;
        const chunk = regionCells > 0 ? Math.min(regionCells, this.liveDerivedChunkSeedBound) : this.liveDerivedChunkSeedBound;
        this.liveDerivedSeedBound += (activity - this.liveDerivedSceneActivity) * chunk;
        this.liveDerivedSceneActivity = activity;
        this.planLiveDerivedFrames(planner.drainFrames(this.liveDerivedSeedBound), settleFrames);
      }
      if (physics) this.planLiveDerivedFrames(planner.stepDrainFrames(this.liveDerivedSeedBound), settleFrames);
      if (this.liveDerivedRelight) this.planLiveDerivedFrames(0, settleFrames);
    }
    if (this.liveDerivedFramesRemaining <= 0) {
      seam?.("world-derived-lighting");
      seam?.("world-radiance-feedback");
      return false;
    }
    const mode = initial ? (radiance ? "initial" : "initial-opacity")
      : !radiance ? "dirty-opacity" : this.liveDerivedRelight ? "relight" : "dirty";
    if (initial) planner.encodeReset(encoder);
    const pass = encoder.beginComputePass(LIVE_DERIVED_LIGHTING_PASS);
    if (initial) builder.encodeCertifyEmpty(pass);
    planner.encodeFrame(pass, mode, physics);
    const rounds = initial ? planner.initialRounds : planner.roundsPerFrame;
    for (let round = 0; round < rounds; round += 1) {
      planner.encodeRound(pass);
      builder.encodeRound(pass, radiance);
    }
    pass.end();
    // The gather is a dispatch of every round now, not a pass after the build,
    // so the feedback seam closes over nothing and the lighting row carries it.
    seam?.("world-derived-lighting");
    seam?.("world-radiance-feedback");
    this.liveDerivedInitial = false;
    this.liveDerivedRelight = false;
    this.liveDerivedFramesRemaining -= 1;
    if (this.liveDerivedDrainFramesRemaining > 0) {
      this.liveDerivedDrainFramesRemaining -= 1;
      if (this.liveDerivedDrainFramesRemaining === 0) this.liveDerivedSeedBound = 0;
      this.republishLiveDerivedOpacity();
    }
    if (radiance && !this.sceneSource.tetrahedralRadiance) {
      this.sceneSource.tetrahedralRadiance = this.liveDerivedRadianceSource ??= this.tetrahedralRadiance?.visibleGeneration();
    }
    return true;
  }

  /** Owe at least `drain` frames that may change opacity, and `settle` more behind them. */
  private planLiveDerivedFrames(drain: number, settle: number): void {
    this.liveDerivedDrainFramesRemaining = Math.max(this.liveDerivedDrainFramesRemaining, drain);
    this.liveDerivedFramesRemaining = Math.max(this.liveDerivedFramesRemaining, this.liveDerivedDrainFramesRemaining + settle);
  }

  /**
   * Tell the renderer opacity may have changed, without making anything: the
   * published pyramid swaps with its twin. A pyramid someone else published
   * since — a re-seed moves its metres — is paired afresh.
   */
  private republishLiveDerivedOpacity(): void {
    const published = this.sceneSource.nodeMipPyramid;
    if (!published) return;
    let twins = this.liveDerivedPyramidTwins;
    if (!twins || (published !== twins[0] && published !== twins[1])) {
      twins = this.liveDerivedPyramidTwins = [published, { ...published }];
    }
    this.sceneSource.nodeMipPyramid = published === twins[0] ? twins[1] : twins[0];
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
    this.encodeLiveDerivedMaintenance(encoder, undefined, true);
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
