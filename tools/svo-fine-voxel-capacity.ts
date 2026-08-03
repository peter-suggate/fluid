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
import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import type { SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { planSparseSceneDomain } from "../lib/sparse-scene-domain";
import { SPARSE_BRICK_GPU_LAYOUT } from "../lib/sparse-brick-octree";
import { planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { svoPrimitiveCandidateBounds } from "../lib/svo-primitive-candidates";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { terrainHeightAt } from "../lib/terrain";
import {
  environmentMaximumCoarseningPower,
  OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  liveSvoBasePageDimensions,
  OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY,
  sparseSceneOctreeMaximumDepth,
} from "../lib/webgpu-octree-sparse-bricks";
import { liveSvoPlanBasePages } from "../lib/webgpu-svo-live-derived-builder";

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
 * How crowded the busiest brick gets.
 *
 * The live voxelizer keeps `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` primitives
 * per dirty brick and raises `CANDIDATE_OVERFLOW` for the rest, which are then
 * simply not written into that brick's voxels
 * (`lib/webgpu-sparse-scene-proxies.ts`). That makes the ceiling on *generated
 * objects* a density rather than a total: 64 primitives per (brickSize x cell)
 * cube of world, which shrinks by 8x every time the lattice halves. A scene
 * library of parameterised species instantiated freely runs into this long
 * before it runs into the 4 096 record arenas.
 */
function primitivesPerBrick(scene: SceneDescription, cells: readonly [number, number, number], brickSize: number) {
  const container = scene.container;
  const cellSize = [container.width_m / cells[0], container.height_m / cells[1], container.depth_m / cells[2]] as const;
  const bricks = cells.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const origin = [-container.width_m / 2, 0, -container.depth_m / 2];
  const counts = new Map<number, number>();
  let primitives = 0;
  for (const descriptor of buildSvoScenePrimitives(scene).descriptors) {
    if (descriptor.kind === "terrain-heightfield") continue;
    primitives += 1;
    const bounds = svoPrimitiveCandidateBounds(descriptor as never);
    const minimum = [bounds.minimum_m.x, bounds.minimum_m.y, bounds.minimum_m.z];
    const maximum = [bounds.maximum_m.x, bounds.maximum_m.y, bounds.maximum_m.z];
    const first = minimum.map((value, axis) => Math.max(0, Math.floor((value - origin[axis]) / (cellSize[axis] * brickSize))));
    const last = maximum.map((value, axis) => Math.min(bricks[axis] - 1, Math.floor((value - origin[axis]) / (cellSize[axis] * brickSize))));
    for (let bz = first[2]; bz <= last[2]; bz += 1) for (let by = first[1]; by <= last[1]; by += 1) for (let bx = first[0]; bx <= last[0]; bx += 1) {
      const key = (bz * bricks[1] + by) * bricks[0] + bx;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  let maximum = 0;
  let overflowed = 0;
  for (const count of counts.values()) {
    maximum = Math.max(maximum, count);
    if (count > OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK) overflowed += 1;
  }
  return { primitives, maximumPerBrick: maximum, overflowedBricks: overflowed, brickEdge_m: cellSize[0] * brickSize };
}

function sweep(cellSize_m: number) {
  // Through the catalog, not the factory: the environment is attached on the
  // way out, and it is the environment's proxy AABBs that decide how far the
  // sparse address space extends past the container.
  const scene: SceneDescription = getScenePreset("hero-garden-hose").create();
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
    result.mipState = !mipPlan.complete ? "capacity"
      : mipPlan.atlas.texels.some((value) => value > M1_MAX_LIMITS.maxTextureDimension3D) ? "atlas-texture-limit"
      : "ready";
  }
  return result;
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
