#!/usr/bin/env node
/**
 * What a finer lattice costs the render SVO, before any GPU is asked for it.
 *
 * Every number the dry render path allocates is decided on the CPU, by the same
 * three functions the production path calls in the same order:
 * `createTallCellLayout` (how many cells), `planSparseSceneDomain` (which bricks
 * exist) and `planAdaptiveSparseBrickOctree` (how many leaves and voxels those
 * bricks become). This tool calls exactly those, at a sweep of cell sizes, and
 * prints the allocation each one implies against the device limits it has to fit
 * inside. It is deliberately GPU-free: the point is to know which cell sizes are
 * worth spending a two-minute Dawn render on, and to be able to say *why* a size
 * that fails cannot work rather than only that it did.
 *
 * The byte model mirrors `SparseBrickOctreeGPU` and `OctreeSparseBrickWorld`
 * field by field; if either changes shape this tool is wrong and should be
 * changed with it. See `docs/SVO_FINE_VOXEL_CAPACITY.md`.
 *
 *   node --import tsx tools/svo-fine-voxel-capacity.ts [cellSize_mm ...]
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import { planAdaptiveSparseBrickOctree } from "../lib/core/adaptive-sparse-brick-plan";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import type { SceneDescription } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";
import { planSparseSceneDomain } from "../lib/core/sparse-scene-domain";
import { SPARSE_BRICK_GPU_LAYOUT } from "../lib/svo/sparse-brick-octree";
import { planSvoNodeMipPyramid } from "../lib/svo/svo-node-mip-pyramid";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../lib/svo/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES } from "../lib/svo/svo-primitive-candidates";
import { svoTetrahedralRadianceAtlasBytes } from "../lib/svo/svo-tetrahedral-radiance";
import {
  svoBrickRasterInstanceBytes,
  svoBrickRasterPublicationInstanceOffsetBytes,
  svoRasterCoverageArenaBytes,
  svoRasterCoverageCountAllocationBytes,
  SVO_BRICK_RASTER_CONTRACT,
} from "../lib/svo/webgpu-svo-brick-raster";
import { SVO_SCENE_PRIMITIVE_RASTER_CONTRACT } from "../lib/svo/webgpu-svo-dry-scene";
import { buildSvoScenePrimitives } from "../lib/svo/svo-scene-primitives";
import { svoPrimitiveCandidateBounds } from "../lib/svo/svo-primitive-candidates";
import { createTallCellLayout } from "../lib/core/tall-cell-grid";
import { terrainHeightAt } from "../lib/core/terrain";
import {
  environmentMaximumCoarseningPower,
  OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  liveSvoBasePageDimensions,
  OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY,
  sparseSceneOctreeMaximumDepth,
} from "../lib/svo/webgpu-svo-sparse-bricks";
import { liveSvoPlanBasePages } from "../lib/svo/webgpu-svo-live-derived-builder";
import { svoScenePrimitiveBrickDensity } from "./svo-dry-frame-harness";

/** This machine's Dawn/Metal adapter, as reported by `adapter.limits`. */
const M1_MAX_LIMITS = {
  maxBufferSize: 4_294_967_295,
  maxStorageBufferBindingSize: 4_294_967_295,
  maxTextureDimension2D: 16_384,
  maxTextureDimension3D: 2_048,
  maxComputeWorkgroupsPerDimension: 65_535,
} as const;

const requested = process.argv.slice(2).map(Number).filter((value) => Number.isFinite(value) && value > 0);
const cellSizes_mm = requested.length > 0 ? requested : [25, 15, 12.5, 10, 7.5, 5, 4, 3, 2.5, 2, 1.5, 1];

/**
 * `FLUID_SVO_CAPACITY_SURFACE_ONLY=1` reports the geometry-driven brick count
 * alone and skips `createTallCellLayout`, which is minutes of CPU at a
 * millimetre lattice because it walks every cell to build an initial liquid
 * field a dry scene never reads. The dimensions are then derived rather than
 * planned, by the three lines `createTallCellLayout` uses; `sweep` asserts the
 * derivation against the real thing on every row it does plan, so the two can
 * never quietly disagree.
 */
const surfaceOnly = process.env.FLUID_SVO_CAPACITY_SURFACE_ONLY === "1";

function derivedDimensions(scene: SceneDescription): readonly [number, number, number] {
  const requestedCellSize = scene.voxelDomain.finestCellSize_m;
  const clamp = (extent: number) => Math.min(M1_MAX_LIMITS.maxTextureDimension3D, Math.max(8, Math.round(extent / requestedCellSize)));
  return [clamp(scene.container.width_m), clamp(scene.container.height_m), clamp(scene.container.depth_m)];
}

const mib = (bytes: number) => bytes / 1024 ** 2;

/**
 * How many bricks a *geometry-driven* octree would hold, against the dense cover
 * the live world plans today.
 *
 * The dry scene's surfaces are the sculpted heightfield and the primitives; air
 * above and rock below need no brick to render correctly. Counting the bricks
 * whose cell extent the surface actually crosses is therefore the honest lower
 * bound on what the render SVO would have to allocate if it stopped covering the
 * container, and it is the number that decides whether a fine lattice is a
 * memory question or an architecture question: the dense cover grows as h^-3
 * while this grows as h^-2.
 */
function surfaceBricks(scene: SceneDescription, cells: readonly [number, number, number], brickSize: number): number {
  const container = scene.container;
  const cellSize = [container.width_m / cells[0], container.height_m / cells[1], container.depth_m / cells[2]] as const;
  const bricks = cells.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  // A conservative band of one cell on each side, matching the planner's padding.
  const pad = Math.max(...cellSize);
  const occupied = new Set<number>();
  for (let bz = 0; bz < bricks[2]; bz += 1) {
    for (let bx = 0; bx < bricks[0]; bx += 1) {
      let low = Number.POSITIVE_INFINITY;
      let high = Number.NEGATIVE_INFINITY;
      // Brick corners bound a bilinear heightfield's extremes over the footprint
      // only approximately, so sample every cell centre in the footprint.
      for (let cz = 0; cz < brickSize; cz += 1) {
        for (let cx = 0; cx < brickSize; cx += 1) {
          const x = -container.width_m / 2 + (bx * brickSize + cx + 0.5) * cellSize[0];
          const z = -container.depth_m / 2 + (bz * brickSize + cz + 0.5) * cellSize[2];
          const height = terrainHeightAt(scene.terrain, x, z);
          low = Math.min(low, height);
          high = Math.max(high, height);
        }
      }
      const first = Math.max(0, Math.floor((low - pad) / (cellSize[1] * brickSize)));
      const last = Math.min(bricks[1] - 1, Math.floor((high + pad) / (cellSize[1] * brickSize)));
      for (let by = first; by <= last; by += 1) occupied.add((bz * bricks[1] + by) * bricks[0] + bx);
    }
  }
  for (const descriptor of buildSvoScenePrimitives(scene).descriptors) {
    if (descriptor.kind === "terrain-heightfield") continue;
    const bounds = svoPrimitiveCandidateBounds(descriptor as never);
    const minimum = [bounds.minimum_m.x, bounds.minimum_m.y, bounds.minimum_m.z];
    const maximum = [bounds.maximum_m.x, bounds.maximum_m.y, bounds.maximum_m.z];
    const origin = [-container.width_m / 2, 0, -container.depth_m / 2];
    const first = minimum.map((value, axis) => Math.max(0, Math.floor((value - origin[axis] - pad) / (cellSize[axis] * brickSize))));
    const last = maximum.map((value, axis) => Math.min(bricks[axis] - 1, Math.floor((value - origin[axis] + pad) / (cellSize[axis] * brickSize))));
    for (let bz = first[2]; bz <= last[2]; bz += 1) for (let by = first[1]; by <= last[1]; by += 1) for (let bx = first[0]; bx <= last[0]; bx += 1) {
      occupied.add((bz * bricks[1] + by) * bricks[0] + bx);
    }
  }
  return occupied.size;
}

/**
 * How crowded the busiest brick gets, at a lattice this sweep is only
 * contemplating rather than one the GPU has published.
 *
 * The binning itself is `svoScenePrimitiveBrickDensity` in
 * `./svo-dry-frame-harness`, shared with the render smoke lane so the sweep and
 * the gate cannot disagree about what "busiest brick" means. What stays here is
 * this tool's *hypothetical* lattice: cell size derived from the container and
 * the swept cell count, with the container's own corner as the origin. The
 * smoke lane instead reads the real published `structural.domain`, which is the
 * right frame once a world exists and the wrong one for a sweep over sizes no
 * world has been built at.
 */
function primitivesPerBrick(scene: SceneDescription, cells: readonly [number, number, number], brickSize: number) {
  const container = scene.container;
  const { primitives, maximumPerBrick, overflowedBricks, brickEdge_m } = svoScenePrimitiveBrickDensity(
    buildSvoScenePrimitives(scene).descriptors,
    {
      worldOrigin_m: [-container.width_m / 2, 0, -container.depth_m / 2],
      cellSize_m: [container.width_m / cells[0], container.height_m / cells[1], container.depth_m / cells[2]],
      dimensionsCells: cells as readonly [number, number, number],
      brickSize,
    },
    OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  );
  return { primitives, maximumPerBrick, overflowedBricks, brickEdge_m };
}

function sweep(cellSize_m: number, sceneFactory: () => SceneDescription = () => getScenePreset("hero-garden-hose").create()) {
  // Through the catalog, not the factory: the environment is attached on the
  // way out, and it is the environment's proxy AABBs that decide how far the
  // sparse address space extends past the container.
  const scene: SceneDescription = sceneFactory();
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: cellSize_m };
  const result: Record<string, unknown> = { cellSize_mm: cellSize_m * 1000 };

  const brickSize = scene.voxelDomain.brickSize_cells === 4 ? 4 : 8;
  let dimensions = derivedDimensions(scene);
  if (!surfaceOnly) {
    const layoutStart = performance.now();
    const layout = createTallCellLayout(scene, "balanced", M1_MAX_LIMITS.maxTextureDimension3D, {
      regularLayers: 2, liquidHalo: 0, airHalo: 0,
    });
    result.layout_ms = performance.now() - layoutStart;
    if ([layout.nx, layout.fineNy, layout.nz].some((value, axis) => value !== dimensions[axis])) {
      throw new Error(`Derived lattice ${dimensions} disagrees with createTallCellLayout ${[layout.nx, layout.fineNy, layout.nz]}`);
    }
    dimensions = [layout.nx, layout.fineNy, layout.nz];
  }
  result.solverCells = dimensions;
  result.clampedByTexture3D = dimensions.some((value) => value === M1_MAX_LIMITS.maxTextureDimension3D);
  Object.assign(result, primitivesPerBrick(scene, dimensions, brickSize));
  if (surfaceOnly) {
    const surfaceStart = performance.now();
    result.surfaceBricks = surfaceBricks(scene, dimensions, brickSize);
    result.surface_ms = performance.now() - surfaceStart;
    result.surfaceVoxels = (result.surfaceBricks as number) * brickSize ** 3;
    result.denseBricks = dimensions.reduce((product, value) => product * Math.ceil(value / brickSize), 1);
    return result;
  }
  const environment = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, scene.environment ?? "default"), true);
  const domainStart = performance.now();
  const domain = planSparseSceneDomain(scene, dimensions, brickSize,
    environment.map((primitive) => ({ min: primitive.aabb_m.min, max: primitive.aabb_m.max })),
    { conservativePaddingCells: 1, worldBounds_m: scene.voxelDomain.bounds_m });
  result.domain_ms = performance.now() - domainStart;
  result.sceneCells = domain.sceneDimensionsCells;
  result.brickDimensions = domain.brickDimensions;
  result.solverBricks = domain.solverBrickCoordinates.length;
  result.environmentBricks = domain.environmentBrickCoordinates.length;
  const surfaceStart = performance.now();
  result.surfaceBricks = surfaceBricks(scene, dimensions, brickSize);
  result.surface_ms = performance.now() - surfaceStart;
  result.surfaceVoxels = (result.surfaceBricks as number) * brickSize ** 3;

  const maximumDepth = sparseSceneOctreeMaximumDepth(domain.brickDimensions, domain.coordinates);
  const planStart = performance.now();
  const plan = planAdaptiveSparseBrickOctree({
    brickSize,
    solverBricks: domain.solverBrickCoordinates,
    proxyBricks: domain.proxyBrickCoordinates.flat(),
    maximumDepth,
    maximumEnvironmentCoarseningPower: Math.min(environmentMaximumCoarseningPower(), maximumDepth),
  });
  result.plan_ms = performance.now() - planStart;
  result.maximumDepth = maximumDepth;
  result.nodes = plan.nodes.length;
  result.leaves = plan.leaves.length;
  result.planVoxels = plan.voxelCount;

  const sceneBrickVolume = domain.brickDimensions.reduce((product, value) => product * value, 1);
  const mutationCapacity = Math.min(sceneBrickVolume, OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY);
  const leafCapacity = plan.leaves.length + mutationCapacity;
  const voxelCapacity = leafCapacity * brickSize ** 3;
  result.voxelCapacity = voxelCapacity;

  // SparseBrickOctreeGPU's two arenas, exactly as its constructor sizes them.
  const g = SPARSE_BRICK_GPU_LAYOUT;
  const payloadBytes = voxelCapacity * (2 * g.geometryStrideBytes + g.velocityStrideBytes + 2 * g.materialOwnerStrideBytes);
  const structureBytes = g.topologyOffsetBytes + plan.nodes.length * g.nodeStrideBytes + leafCapacity * g.leafStrideBytes;
  // The CPU staging buffers OctreeSparseBrickWorld uploads the initial plan through.
  const sourceBytes = plan.voxelCount * (g.geometryStrideBytes + g.velocityStrideBytes + g.materialOwnerStrideBytes);
  // WebGPULiveSvoScene's dense fluid-field textures, allocated for a dry scene too.
  const denseTextureBytes = dimensions[0] * dimensions[1] * dimensions[2] * (4 + 16);
  result.payloadMiB = mib(payloadBytes);
  result.structureMiB = mib(structureBytes);
  result.sourceMiB = mib(sourceBytes);
  result.denseTextureMiB = mib(denseTextureBytes);
  result.totalMiB = mib(payloadBytes + structureBytes + sourceBytes + denseTextureBytes);
  result.payloadOverBufferLimit = payloadBytes > M1_MAX_LIMITS.maxBufferSize;
  result.largestBindingMiB = mib(voxelCapacity * g.geometryStrideBytes);
  result.bindingOverLimit = voxelCapacity * g.geometryStrideBytes > M1_MAX_LIMITS.maxStorageBufferBindingSize;

  // The node-mip pyramid, whose withdrawal is silent and costs the whole frame.
  const basePageDimensions = liveSvoBasePageDimensions(domain.brickDimensions, brickSize);
  const levelCount = sparseSceneOctreeMaximumDepth(basePageDimensions, []) + 1;
  result.mipLevelCount = levelCount;
  if (levelCount > 12) {
    result.mipState = "unsupported-level-count";
  } else {
    const pageStart = performance.now();
    const mipPlan = planSvoNodeMipPyramid({
      generation: 1,
      occupiedPages: liveSvoPlanBasePages(plan),
      levelCount,
      capacity: M1_MAX_LIMITS.maxTextureDimension2D,
    });
    result.mip_ms = performance.now() - pageStart;
    result.mipPagesRequested = mipPlan.requestedPageCount;
    result.mipAtlasTexels = mipPlan.atlas.texels;
    result.mipAtlasMiB = mib(mipPlan.allocatedBytes);
    // The tetrahedral radiance atlas shares the pyramid's page plan and its
    // capacity, at 10^3 x 16 B = 16,000 B a physical page. It is a separate
    // allocation of comparable size and section 8 of the handoff names it
    // explicitly, so it is reported beside the pyramid rather than folded in.
    result.radianceAtlasMiB = mib(svoTetrahedralRadianceAtlasBytes(mipPlan));
    result.mipAllocatedBytes = mipPlan.allocatedBytes;
    result.radianceAllocatedBytes = svoTetrahedralRadianceAtlasBytes(mipPlan);
    result.mipState = !mipPlan.complete ? "capacity"
      : mipPlan.atlas.texels.some((value) => value > M1_MAX_LIMITS.maxTextureDimension3D) ? "atlas-texture-limit"
      : "ready";
  }
  return result;
}

/**
 * ---------------------------------------------------------------------------
 * The 10x memory audit (`docs/svo-raster-visibility-handoff.md` section 8)
 * ---------------------------------------------------------------------------
 *
 * "56 B/voxel across lanes, pyramid pages, radiance atlas (16 KB/page), ~190 MB
 * coverage buffer, candidate arena. W4's audit must produce totals before W2
 * commits to a finest-voxel size at 10x."
 *
 * The sweep above answers the *first* question — what one cell size costs the
 * octree — for one scene at one record count. This mode answers section 8's
 * question, which is a different one: what does the whole renderer allocate,
 * across every arena the handoff names, at the acceptance scene's record rungs
 * and at candidate finest-voxel sizes, and which of those configurations do not
 * fit on this machine.
 *
 * Two axes, because the two costs are independent and are routinely confused:
 *
 *   - **record count** (`recordMultiplier` 1 / 8 / 10, spanning exactly
 *     `hero-garden-hose` at 501 records to `hero-garden-hose-x10` at 5,039)
 *     moves the primitive records, the candidate arena and — through the
 *     proxy AABBs that decide which bricks exist — the octree itself;
 *   - **finest cell size** moves the octree as h^-2 to h^-3 and moves nothing
 *     else at all.
 *
 * The viewport-sized arenas move with neither, which is the point worth taking
 * away: W1's shared coverage arena is 325.4 MB at 1600x1240 whatever the scene
 * contains, and it is already the largest single allocation in the renderer at
 * every rung the acceptance scene draws at.
 *
 * `FLUID_SVO_CAPACITY_AUDIT=1 node --import tsx tools/svo-fine-voxel-capacity.ts [cellSize_mm ...]`
 *   FLUID_SVO_CAPACITY_MULTIPLIERS  record rungs (default 1,8,10)
 *   FLUID_SVO_CAPACITY_VIEWPORT     WxH for the viewport arenas (default 1600x1240)
 */
const auditMode = process.env.FLUID_SVO_CAPACITY_AUDIT === "1";
const auditMultipliers = (process.env.FLUID_SVO_CAPACITY_MULTIPLIERS ?? "1,8,10")
  .split(",").map(Number).filter((value) => Number.isFinite(value) && value >= 1);
const [auditWidth, auditHeight] = (process.env.FLUID_SVO_CAPACITY_VIEWPORT ?? "1600x1240")
  .split("x").map(Number);

/**
 * Arenas whose size is decided by the viewport, not by the scene.
 *
 * The coverage arena is one allocation shared by the brick pass and the
 * authored-SDF pass (W1) — `width * height * (candidates + 1) * 4`, where the
 * `+ 1` is the per-pixel nearest-opaque seed the brick resolve publishes for
 * the scene-primitive resolve to reject against. The count buffer beside it
 * carries the overflow tail this workstream added.
 */
function viewportArenas(width: number, height: number) {
  const coverageArenaBytes = svoRasterCoverageArenaBytes(width, height,
    SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.coverageCandidatesPerPixel);
  const coverageCountBytes = svoRasterCoverageCountAllocationBytes(width, height);
  // Raster-primary's four colour attachments plus the depth plane. Attachments
  // are not storage bindings, so they answer to maxTextureDimension2D rather
  // than to maxStorageBufferBindingSize, but they are real device memory.
  const gBufferBytes = width * height * (SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample + 4);
  return {
    coverageArenaBytes,
    coverageCountBytes,
    gBufferBytes,
    total: coverageArenaBytes + coverageCountBytes + gBufferBytes,
  };
}

function auditRow(recordMultiplier: number, cellSize_m: number) {
  const scene = createHeroGardenHoseStressScene({ recordMultiplier });
  const octree = sweep(cellSize_m, () => createHeroGardenHoseStressScene({ recordMultiplier }));
  const recordCount = buildSvoScenePrimitives(scene).descriptors.length;
  const leafCapacity = Math.max(1, Number(octree.voxelCapacity ?? 0) / (scene.voxelDomain.brickSize_cells === 4 ? 64 : 512));
  const view = viewportArenas(auditWidth, auditHeight);

  // Per-record host publications. The candidate arena is a fixed allocation
  // (W0 raised it with the leaf ceiling), so it is charged in full whatever the
  // record count is — that is what "cliff to slope" cost.
  const recordBytes = recordCount * SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
  const candidateArenaBytes = SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES;
  // Brick-raster instance arenas: the sorted publication and the pre-sort
  // candidate list, both one 32-byte record per leaf of published capacity.
  const brickInstanceBytes = svoBrickRasterInstanceBytes(Math.round(leafCapacity));
  const brickRasterBytes = 2 * brickInstanceBytes + svoBrickRasterPublicationInstanceOffsetBytes();

  const octreeBytes = Math.round(Number(octree.payloadMiB ?? 0) * 1024 ** 2)
    + Math.round(Number(octree.structureMiB ?? 0) * 1024 ** 2)
    + Math.round(Number(octree.sourceMiB ?? 0) * 1024 ** 2)
    + Math.round(Number(octree.denseTextureMiB ?? 0) * 1024 ** 2);
  const lightingBytes = Number(octree.mipAllocatedBytes ?? 0) + Number(octree.radianceAllocatedBytes ?? 0);
  const total = octreeBytes + lightingBytes + view.total + recordBytes + candidateArenaBytes + brickRasterBytes;

  // Two different device limits, kept apart because conflating them is how an
  // audit says "fits" about a configuration that cannot be allocated.
  //
  //   * `maxBufferSize` bounds one **allocation**. The octree payload is a
  //     single buffer holding all five per-voxel lanes end to end, so that is
  //     the limit it answers to.
  //   * `maxStorageBufferBindingSize` bounds one **binding**. The lanes inside
  //     the payload are bound as separate sub-ranges, so the geometry lane —
  //     the widest of them at 16 B/voxel — is the one that hits this ceiling.
  //
  // On this adapter both are 4,294,967,295 B, which is why the distinction
  // costs nothing here and would cost the whole verdict on hardware where they
  // differ (they commonly do; 128 MiB and 256 MiB binding limits are ordinary).
  const allocations = ([
    { name: "octree payload buffer (56 B/voxel over five lanes)", bytes: Math.round(Number(octree.payloadMiB ?? 0) * 1024 ** 2), limit: "buffer" },
    { name: "shared coverage candidate arena", bytes: view.coverageArenaBytes, limit: "binding" },
    { name: "octree geometry lane (16 B/voxel)", bytes: Math.round(Number(octree.largestBindingMiB ?? 0) * 1024 ** 2), limit: "binding" },
    { name: "octree CPU staging source", bytes: Math.round(Number(octree.sourceMiB ?? 0) * 1024 ** 2), limit: "buffer" },
    { name: "brick raster instances", bytes: brickRasterBytes, limit: "binding" },
    { name: "primitive candidate arena", bytes: candidateArenaBytes, limit: "binding" },
    { name: "coverage counts + overflow tail", bytes: view.coverageCountBytes, limit: "binding" },
  ] as { name: string; bytes: number; limit: "buffer" | "binding" }[]).sort((left, right) => right.bytes - left.bytes);
  const bindings = allocations;
  const overBinding = allocations.filter(({ bytes, limit }) => bytes > (limit === "buffer"
    ? M1_MAX_LIMITS.maxBufferSize : M1_MAX_LIMITS.maxStorageBufferBindingSize));

  return {
    recordMultiplier,
    recordCount,
    cellSize_mm: cellSize_m * 1000,
    leaves: octree.leaves,
    nodes: octree.nodes,
    voxelCapacity: octree.voxelCapacity,
    maximumPerBrick: octree.maximumPerBrick,
    overflowedBricks: octree.overflowedBricks,
    mipState: octree.mipState,
    mipPagesRequested: octree.mipPagesRequested,
    bytes: {
      octreePayload: Math.round(Number(octree.payloadMiB ?? 0) * 1024 ** 2),
      octreeStructure: Math.round(Number(octree.structureMiB ?? 0) * 1024 ** 2),
      octreeCpuSource: Math.round(Number(octree.sourceMiB ?? 0) * 1024 ** 2),
      denseFluidTextures: Math.round(Number(octree.denseTextureMiB ?? 0) * 1024 ** 2),
      nodeMipAtlas: Number(octree.mipAllocatedBytes ?? 0),
      radianceAtlas: Number(octree.radianceAllocatedBytes ?? 0),
      coverageArena: view.coverageArenaBytes,
      coverageCounts: view.coverageCountBytes,
      gBufferAttachments: view.gBufferBytes,
      primitiveRecords: recordBytes,
      candidateArena: candidateArenaBytes,
      brickRasterInstances: brickRasterBytes,
    },
    totalMiB: Number(mib(total).toFixed(1)),
    largestBinding: bindings[0],
    largestBindingMiB: Number(mib(bindings[0].bytes).toFixed(1)),
    bindingLimitMiB: Number(mib(M1_MAX_LIMITS.maxStorageBufferBindingSize).toFixed(1)),
    fits: overBinding.length === 0 && octree.mipState === "ready",
    doesNotFitBecause: [
      ...overBinding.map(({ name, bytes, limit }) =>
        `${name} needs ${mib(bytes).toFixed(1)} MiB against a ${mib(limit === "buffer"
          ? M1_MAX_LIMITS.maxBufferSize : M1_MAX_LIMITS.maxStorageBufferBindingSize).toFixed(1)} MiB ${limit} limit`),
      ...(octree.mipState === "ready" ? []
        : [`node-mip pyramid state is "${octree.mipState}", not "ready" — cone lighting withdraws to exact traversal`]),
    ],
  };
}

if (auditMode) {
  const auditRows: unknown[] = [];
  for (const millimetres of cellSizes_mm) {
    for (const multiplier of auditMultipliers) {
      const started = performance.now();
      try {
        const row = { ...auditRow(multiplier, millimetres / 1000), audit_ms: Math.round(performance.now() - started) };
        auditRows.push(row);
        console.log(JSON.stringify(row));
      } catch (error) {
        const row = {
          recordMultiplier: multiplier,
          cellSize_mm: millimetres,
          fits: false,
          threw: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        };
        auditRows.push(row);
        console.log(JSON.stringify(row));
      }
    }
  }
  console.log(JSON.stringify({
    record: "svo-10x-memory-audit",
    viewport: { width: auditWidth, height: auditHeight },
    limits: M1_MAX_LIMITS,
    rows: auditRows,
  }));
  process.exit(0);
}

const rows: Record<string, unknown>[] = [];
for (const millimetres of cellSizes_mm) {
  const started = performance.now();
  try {
    const row = sweep(millimetres / 1000);
    row.total_ms = performance.now() - started;
    rows.push(row);
    console.log(JSON.stringify(row));
  } catch (error) {
    const row = {
      cellSize_mm: millimetres,
      total_ms: performance.now() - started,
      threw: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    };
    rows.push(row);
    console.log(JSON.stringify(row));
  }
}
console.log(JSON.stringify({ limits: M1_MAX_LIMITS, rows }, null, 2).slice(0, 0));
