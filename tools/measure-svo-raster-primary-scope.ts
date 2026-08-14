#!/usr/bin/env node
/**
 * Phase 0 scoping measurement for raster-assisted primary visibility
 * (docs/SVO_RASTER_PRIMARY_HANDOFF.md).
 *
 * Builds the shipped live SVO scene for a scene, reads the published
 * topology back from the GPU, and reports what an instanced brick raster would
 * actually have to draw:
 *
 * - resident leaf-brick count and the level histogram,
 * - projected brick size in pixels at the benchmark resolution (the LOD-cut
 *   input),
 * - exact per-ray brick-interval counts for a sampled pixel grid — the ray
 *   intervals of octree leaves are disjoint, so this count *is* the raster
 *   overdraw an unordered draw would shade,
 * - the same count restricted to bricks in front of the first *occupied* brick
 *   hit, which is what front-to-back ordering plus depth rejection leaves.
 *
 * Rerun: node --import tsx tools/measure-svo-raster-primary-scope.ts
 * Env: FLUID_SVO_RASTER_SCOPE_SCENE (default garden-svo-lighting),
 *      FLUID_SVO_RASTER_SCOPE_WIDTH / _HEIGHT (default 1500),
 *      FLUID_SVO_RASTER_SCOPE_SAMPLES (per-axis ray samples, default 128),
 *      FLUID_SVO_RASTER_SCOPE_OUT, WEBGPU_NODE_MODULE.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { cameraPosition } from "../lib/core/math";
import { defaultCamera, type CameraState } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";
import { decodeSvoBrickOccupancy } from "../lib/svo/svo-brick-occupancy";
import { buildSvoScenePrimitives } from "../lib/svo/svo-scene-primitives";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const log = (message: string) => process.stderr.write(`${message}\n`);

const sceneId = process.env.FLUID_SVO_RASTER_SCOPE_SCENE ?? "garden-svo-lighting";
const width = Number(process.env.FLUID_SVO_RASTER_SCOPE_WIDTH ?? 1500);
const height = Number(process.env.FLUID_SVO_RASTER_SCOPE_HEIGHT ?? 1500);
const samples = Number(process.env.FLUID_SVO_RASTER_SCOPE_SAMPLES ?? 128);
const outPath = path.resolve(repoRoot, process.env.FLUID_SVO_RASTER_SCOPE_OUT
  ?? `artifacts/render-raster-primary/scope-${sceneId}.json`);
assert.ok(Number.isInteger(width) && width > 0 && Number.isInteger(height) && height > 0,
  "scope width/height must be positive integers");
assert.ok(Number.isInteger(samples) && samples >= 8, "scope samples must be an integer of at least 8");

const modulePath = process.env.WEBGPU_NODE_MODULE ?? path.resolve(repoRoot, "node_modules/webgpu/index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "no Metal adapter");
const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });

const preset = getScenePreset(sceneId);
const scene = preset.create();
const camera: CameraState = { ...defaultCamera, ...preset.camera, target_m: { ...(preset.camera?.target_m ?? defaultCamera.target_m) } };
const solver = await WebGPULiveSvoScene.create(device, scene, "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`));
const source = solver.sparseVoxelSceneSource;
assert.ok(source?.structural, "live SVO scene did not publish a structural scene source");
const structural = source.structural;

async function readBuffer(binding: GPUBufferBinding): Promise<Uint32Array> {
  const size = binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return words;
}

const control = await readBuffer(structural.control);
const nodeCount = control[0];
const leafCount = control[1];
const voxelCount = control[2];
assert.equal(control[12] >>> 0, 0, "topology publication overflowed — refusing to scope a truncated world");
log(`Published topology: ${nodeCount} nodes, ${leafCount} leaves, ${voxelCount} voxels`);

const nodes = await readBuffer(structural.nodes);
const leaves = await readBuffer(structural.leaves);
const materialOwners = await readBuffer(structural.materialOwners);
const NODE_WORDS = structural.strides.node / 4;
const LEAF_WORDS = structural.strides.leaf / 4;

const { worldOrigin_m, cellSize_m, brickSize, maximumDepth } = structural.domain;

function compactMortonBits(value: number): number {
  let compact = value & 0x49249249;
  compact = (compact ^ (compact >>> 2)) & 0xc30c30c3;
  compact = (compact ^ (compact >>> 4)) & 0x0f00f00f;
  compact = (compact ^ (compact >>> 8)) & 0xff0000ff;
  return (compact ^ (compact >>> 16)) & 0x0000ffff;
}

function decodeMorton(low: number, high: number, level: number): [number, number, number] {
  const levelMask = level >= 32 ? 0xffffffff : (1 << level) - 1;
  const lowBits: [number, number, number] = [
    compactMortonBits(low), compactMortonBits(low >>> 1), compactMortonBits(low >>> 2),
  ];
  const highBits: [number, number, number] = [
    compactMortonBits(high >>> 1), compactMortonBits(high >>> 2), compactMortonBits(high),
  ];
  const shift: [number, number, number] = [11, 11, 10];
  return [0, 1, 2].map((axis) => (lowBits[axis] | (highBits[axis] << shift[axis])) & levelMask) as [number, number, number];
}

interface BrickInstance {
  minimum: [number, number, number];
  maximum: [number, number, number];
  /** Occupied sub-AABB from the terminal node's published occupancy word. */
  tightMinimum: [number, number, number];
  tightMaximum: [number, number, number];
  level: number;
  voxelOffset: number;
  occupiedVoxels: number;
  occupancyReady: boolean;
}

// Occupancy matches the production leaf DDA exactly: a payload voxel counts
// only when its owner reaches the analytic-primitive base and is not the
// suppressed open-shell owner (webgpu-svo-dry-scene.ts traceLeafPayload).
const scenePrimitives = buildSvoScenePrimitives(scene);
const ownerBase = scene.rigidBodies.length;
const suppressedOwner = scenePrimitives.openShellOwnerId;
const primitiveCount = scenePrimitives.packedRecords.length > 0
  ? scenePrimitives.packedRecords.length / (scenePrimitives.packedRecords.length > 0 ? 1 : 1) : 0;
void primitiveCount;
const voxelOccupied = (index: number): boolean => {
  if (index >= materialOwners.length) return false;
  const owner = materialOwners[index] >>> 16;
  return owner >= ownerBase && owner !== suppressedOwner;
};

const bricks: BrickInstance[] = [];
const levelHistogram = new Map<number, number>();
let occupiedVoxelTotal = 0;
const voxelsPerBrick = brickSize ** 3;
for (let leafIndex = 0; leafIndex < leafCount; leafIndex += 1) {
  const nodeIndex = leaves[leafIndex * LEAF_WORDS];
  const voxelOffset = leaves[leafIndex * LEAF_WORDS + 1];
  const base = nodeIndex * NODE_WORDS;
  const level = nodes[base + 2];
  const coordinate = decodeMorton(nodes[base], nodes[base + 1], level);
  const scale = (1 << (maximumDepth - level)) * brickSize;
  const minimum = [0, 1, 2].map((axis) => worldOrigin_m[axis] + coordinate[axis] * scale * cellSize_m[axis]) as [number, number, number];
  const maximum = [0, 1, 2].map((axis) => minimum[axis] + scale * cellSize_m[axis]) as [number, number, number];
  let occupiedVoxels = 0;
  for (let voxel = 0; voxel < voxelsPerBrick; voxel += 1) {
    if (voxelOccupied(voxelOffset + voxel)) occupiedVoxels += 1;
  }
  occupiedVoxelTotal += occupiedVoxels;
  levelHistogram.set(level, (levelHistogram.get(level) ?? 0) + 1);
  const summary = decodeSvoBrickOccupancy(nodes[base + 7]);
  const cell = [0, 1, 2].map((axis) => (maximum[axis] - minimum[axis]) / brickSize);
  const tight = summary.ready && summary.occupied;
  bricks.push({
    minimum, maximum, level, voxelOffset, occupiedVoxels, occupancyReady: summary.ready,
    tightMinimum: [0, 1, 2].map((axis) => (tight ? minimum[axis] + summary.minInclusive[axis] * cell[axis] : minimum[axis])) as [number, number, number],
    tightMaximum: [0, 1, 2].map((axis) => (tight ? minimum[axis] + (summary.maxInclusive[axis] + 1) * cell[axis] : maximum[axis])) as [number, number, number],
  });
}
log(`Resident leaf bricks: ${bricks.length}; occupied voxels ${occupiedVoxelTotal} `
  + `(${(100 * occupiedVoxelTotal / Math.max(1, bricks.length * voxelsPerBrick)).toFixed(1)}% brick fill)`);

// --- camera basis, exactly mirroring the dry-scene fragment shader ----------
const position = cameraPosition(camera);
const origin: [number, number, number] = [position.x, position.y, position.z];
const target: [number, number, number] = [camera.target_m.x, camera.target_m.y, camera.target_m.z];
const normalize = (v: [number, number, number]): [number, number, number] => {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
};
const cross = (a: [number, number, number], b: [number, number, number]): [number, number, number] =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const forward = normalize([target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]]);
const right = normalize(cross(forward, [0, 1, 0]));
const up = normalize(cross(right, forward));
const aspect = width / height;
const TAN_HALF = 0.72;

// --- projected size of each brick, in pixels --------------------------------
const projectedPixels: number[] = [];
const projectedProxyPixels: number[] = [];
for (const brick of bricks) {
  projectedPixels.push(projectedSize(brick.minimum, brick.maximum));
  if (brick.occupiedVoxels > 0) projectedProxyPixels.push(projectedSize(brick.tightMinimum, brick.tightMaximum));
}
function projectedSize(boxMinimum: readonly number[], boxMaximum: readonly number[]): number {
  let minimumX = Infinity; let maximumX = -Infinity;
  let minimumY = Infinity; let maximumY = -Infinity;
  let behind = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const p: [number, number, number] = [
      corner & 1 ? boxMaximum[0] : boxMinimum[0],
      corner & 2 ? boxMaximum[1] : boxMinimum[1],
      corner & 4 ? boxMaximum[2] : boxMinimum[2],
    ];
    const relative: [number, number, number] = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    const viewDepth = dot(relative, forward);
    if (viewDepth <= 1e-4) { behind += 1; continue; }
    const ndcX = dot(relative, right) / (viewDepth * aspect * TAN_HALF);
    const ndcY = dot(relative, up) / (viewDepth * TAN_HALF);
    minimumX = Math.min(minimumX, ndcX); maximumX = Math.max(maximumX, ndcX);
    minimumY = Math.min(minimumY, ndcY); maximumY = Math.max(maximumY, ndcY);
  }
  if (behind === 8) return 0;
  if (behind > 0) return Number.POSITIVE_INFINITY;
  return Math.max((maximumX - minimumX) * 0.5 * width, (maximumY - minimumY) * 0.5 * height);
}
const finiteProjected = projectedProxyPixels.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
const quantile = (sorted: number[], fraction: number): number =>
  sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];

// --- exact per-ray brick-interval counts over a sampled pixel grid ----------
// Octree leaves partition space, so a ray's [tEnter, tExit] intervals over the
// resident bricks are disjoint: the count below is exactly the number of
// fragments an instanced brick draw shades for that pixel with no ordering,
// and the "beforeFirstOccupied" count is what survives front-to-back ordering.
const brickMinimum = new Float64Array(bricks.length * 3);
const brickMaximum = new Float64Array(bricks.length * 3);
// The rasterized proxy is the published occupied sub-AABB, not the whole
// brick. Sub-boxes of disjoint boxes stay disjoint, so ordering and depth
// resolution are unaffected while silhouette area drops sharply.
const proxyMinimum = new Float64Array(bricks.length * 3);
const proxyMaximum = new Float64Array(bricks.length * 3);
const brickOccupied = new Uint8Array(bricks.length);
bricks.forEach((brick, index) => {
  for (let axis = 0; axis < 3; axis += 1) {
    brickMinimum[index * 3 + axis] = brick.minimum[axis];
    brickMaximum[index * 3 + axis] = brick.maximum[axis];
    proxyMinimum[index * 3 + axis] = brick.tightMinimum[axis];
    proxyMaximum[index * 3 + axis] = brick.tightMaximum[axis];
  }
  brickOccupied[index] = brick.occupiedVoxels > 0 ? 1 : 0;
});

/**
 * Bounded in-brick DDA mirroring `traceLeafPayload`, minus the exact analytic
 * primitive test: entering an occupied voxel counts as a hit. That makes the
 * hit set marginally larger than production at cell corners, which biases the
 * ordered-overdraw estimate optimistically by well under a brick.
 */
function brickDdaSteps(
  brickIndex: number,
  entry: number,
  exit: number,
  direction: readonly [number, number, number],
): { hit: boolean; steps: number } {
  const minimum = [0, 1, 2].map((axis) => brickMinimum[brickIndex * 3 + axis]);
  const extent = [0, 1, 2].map((axis) => (brickMaximum[brickIndex * 3 + axis] - minimum[axis]) / brickSize);
  const voxelOffset = bricks[brickIndex].voxelOffset;
  let t = Math.max(entry, 0);
  const cell = [0, 1, 2].map((axis) => Math.min(brickSize - 1, Math.max(0,
    Math.floor((origin[axis] + direction[axis] * (t + 1e-5) - minimum[axis]) / extent[axis]))));
  const step = [0, 1, 2].map((axis) => (direction[axis] >= 0 ? 1 : -1));
  const nextT = [0, 1, 2].map((axis) => (Math.abs(direction[axis]) > 1e-9
    ? (minimum[axis] + (cell[axis] + (step[axis] > 0 ? 1 : 0)) * extent[axis] - origin[axis]) / direction[axis]
    : Number.POSITIVE_INFINITY));
  const deltaT = [0, 1, 2].map((axis) => (Math.abs(direction[axis]) > 1e-9
    ? Math.abs(extent[axis] / direction[axis]) : Number.POSITIVE_INFINITY));
  let steps = 0;
  for (let iteration = 0; iteration < 32; iteration += 1) {
    if (cell.some((value) => value < 0 || value >= brickSize) || t > exit) break;
    steps += 1;
    if (voxelOccupied(voxelOffset + cell[0] + cell[1] * brickSize + cell[2] * brickSize * brickSize)) {
      return { hit: true, steps };
    }
    const advance = Math.min(nextT[0], nextT[1], nextT[2]);
    for (let axis = 0; axis < 3; axis += 1) {
      if (nextT[axis] <= advance + 1e-6) { cell[axis] += step[axis]; nextT[axis] += deltaT[axis]; }
    }
    t = advance;
  }
  return { hit: false, steps };
}

const perRayAll: number[] = [];
const perRayOccupied: number[] = [];
const perRayOrdered: number[] = [];
const perRayOrderedOccupied: number[] = [];
const perRaySteps: number[] = [];
const perRayProxy: number[] = [];
const perRayProxyOrdered: number[] = [];
const perRayProxySteps: number[] = [];
let raysWithHit = 0;
const enters = new Float64Array(bricks.length);
const exits = new Float64Array(bricks.length);
const candidates = new Int32Array(bricks.length);
for (let sampleY = 0; sampleY < samples; sampleY += 1) {
  for (let sampleX = 0; sampleX < samples; sampleX += 1) {
    const ndcX = ((sampleX + 0.5) / samples) * 2 - 1;
    const ndcY = ((sampleY + 0.5) / samples) * 2 - 1;
    const direction = normalize([
      forward[0] + right[0] * ndcX * aspect * TAN_HALF + up[0] * ndcY * TAN_HALF,
      forward[1] + right[1] * ndcX * aspect * TAN_HALF + up[1] * ndcY * TAN_HALF,
      forward[2] + right[2] * ndcX * aspect * TAN_HALF + up[2] * ndcY * TAN_HALF,
    ]);
    const inverse: [number, number, number] = [1 / direction[0], 1 / direction[1], 1 / direction[2]];
    let count = 0;
    for (let index = 0; index < bricks.length; index += 1) {
      let enter = 0;
      let exit = Number.POSITIVE_INFINITY;
      for (let axis = 0; axis < 3; axis += 1) {
        const first = (brickMinimum[index * 3 + axis] - origin[axis]) * inverse[axis];
        const second = (brickMaximum[index * 3 + axis] - origin[axis]) * inverse[axis];
        enter = Math.max(enter, Math.min(first, second));
        exit = Math.min(exit, Math.max(first, second));
      }
      if (exit < enter) continue;
      enters[count] = enter;
      exits[count] = exit;
      candidates[count] = index;
      count += 1;
    }
    perRayAll.push(count);
    let occupiedCount = 0;
    for (let index = 0; index < count; index += 1) if (brickOccupied[candidates[index]] === 1) occupiedCount += 1;
    perRayOccupied.push(occupiedCount);
    if (count === 0) { perRayOrdered.push(0); perRayOrderedOccupied.push(0); perRaySteps.push(0); continue; }
    // Front-to-back: shading stops at the first brick whose DDA actually hits.
    const order = [...candidates.subarray(0, count).keys()].sort((a, b) => enters[a] - enters[b]);
    let ordered = 0;
    let orderedOccupied = 0;
    let steps = 0;
    let hitFound = false;
    for (const slot of order) {
      ordered += 1;
      if (brickOccupied[candidates[slot]] === 0) continue;
      orderedOccupied += 1;
      const dda = brickDdaSteps(candidates[slot], enters[slot], exits[slot], direction);
      steps += dda.steps;
      if (dda.hit) { hitFound = true; break; }
    }
    if (hitFound) raysWithHit += 1;
    perRayOrdered.push(ordered);
    perRayOrderedOccupied.push(orderedOccupied);
    perRaySteps.push(steps);

    // Production arm: occupied bricks only, rasterized as their tight occupied
    // sub-AABB, submitted front-to-back.
    let proxyCount = 0;
    for (let index = 0; index < bricks.length; index += 1) {
      if (brickOccupied[index] === 0) continue;
      let enter = 0;
      let exit = Number.POSITIVE_INFINITY;
      for (let axis = 0; axis < 3; axis += 1) {
        const first = (proxyMinimum[index * 3 + axis] - origin[axis]) * inverse[axis];
        const second = (proxyMaximum[index * 3 + axis] - origin[axis]) * inverse[axis];
        enter = Math.max(enter, Math.min(first, second));
        exit = Math.min(exit, Math.max(first, second));
      }
      if (exit < enter) continue;
      enters[proxyCount] = enter;
      exits[proxyCount] = exit;
      candidates[proxyCount] = index;
      proxyCount += 1;
    }
    perRayProxy.push(proxyCount);
    let proxyOrdered = 0;
    let proxySteps = 0;
    for (const slot of [...candidates.subarray(0, proxyCount).keys()].sort((a, b) => enters[a] - enters[b])) {
      proxyOrdered += 1;
      const dda = brickDdaSteps(candidates[slot], enters[slot], exits[slot], direction);
      proxySteps += dda.steps;
      if (dda.hit) break;
    }
    perRayProxyOrdered.push(proxyOrdered);
    perRayProxySteps.push(proxySteps);
  }
}
const sortedAll = [...perRayAll].sort((a, b) => a - b);
const sortedOccupied = [...perRayOccupied].sort((a, b) => a - b);
const sortedOrdered = [...perRayOrdered].sort((a, b) => a - b);
const sortedOrderedOccupied = [...perRayOrderedOccupied].sort((a, b) => a - b);
const sortedSteps = [...perRaySteps].sort((a, b) => a - b);
const sortedProxy = [...perRayProxy].sort((a, b) => a - b);
const sortedProxyOrdered = [...perRayProxyOrdered].sort((a, b) => a - b);
const sortedProxySteps = [...perRayProxySteps].sort((a, b) => a - b);
const mean = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

const report = {
  tool: "measure-svo-raster-primary-scope",
  scene: sceneId,
  resolution: { width, height },
  raySamplesPerAxis: samples,
  topology: {
    nodes: nodeCount,
    leaves: leafCount,
    voxels: voxelCount,
    brickSize,
    maximumDepth,
    occupiedVoxels: occupiedVoxelTotal,
    brickFillFraction: occupiedVoxelTotal / Math.max(1, bricks.length * voxelsPerBrick),
    emptyBricks: bricks.filter((brick) => brick.occupiedVoxels === 0).length,
    levelHistogram: Object.fromEntries([...levelHistogram].sort((a, b) => a[0] - b[0])),
  },
  projectedBrickPixels: {
    mean: mean(finiteProjected),
    p10: quantile(finiteProjected, 0.1),
    median: quantile(finiteProjected, 0.5),
    p90: quantile(finiteProjected, 0.9),
    maximum: finiteProjected[finiteProjected.length - 1] ?? 0,
    belowOnePixel: finiteProjected.filter((value) => value < 1).length,
    belowFourPixels: finiteProjected.filter((value) => value < 4).length,
    note: "occupied bricks, tight occupied sub-AABB proxy — the boxes actually drawn",
    cameraInside: projectedProxyPixels.filter((value) => !Number.isFinite(value)).length,
    fullBrickMedian: quantile(projectedPixels.filter((value) => Number.isFinite(value)).sort((a, b) => a - b), 0.5),
  },
  overdraw: {
    definition: "brick ray-intervals per pixel; octree leaves are disjoint so this is exact raster overdraw",
    raysSampled: perRayAll.length,
    raysWithSurfaceHit: raysWithHit,
    unordered: { mean: mean(perRayAll), median: quantile(sortedAll, 0.5), p90: quantile(sortedAll, 0.9), maximum: sortedAll[sortedAll.length - 1] ?? 0 },
    unorderedOccupiedOnly: { mean: mean(perRayOccupied), median: quantile(sortedOccupied, 0.5), p90: quantile(sortedOccupied, 0.9), maximum: sortedOccupied[sortedOccupied.length - 1] ?? 0 },
    frontToBack: { mean: mean(perRayOrdered), median: quantile(sortedOrdered, 0.5), p90: quantile(sortedOrdered, 0.9), maximum: sortedOrdered[sortedOrdered.length - 1] ?? 0 },
    frontToBackOccupiedOnly: { mean: mean(perRayOrderedOccupied), median: quantile(sortedOrderedOccupied, 0.5), p90: quantile(sortedOrderedOccupied, 0.9), maximum: sortedOrderedOccupied[sortedOrderedOccupied.length - 1] ?? 0 },
    tightProxyUnordered: { mean: mean(perRayProxy), median: quantile(sortedProxy, 0.5), p90: quantile(sortedProxy, 0.9), maximum: sortedProxy[sortedProxy.length - 1] ?? 0 },
    tightProxyFrontToBack: { mean: mean(perRayProxyOrdered), median: quantile(sortedProxyOrdered, 0.5), p90: quantile(sortedProxyOrdered, 0.9), maximum: sortedProxyOrdered[sortedProxyOrdered.length - 1] ?? 0 },
  },
  ddaStepsPerRay: {
    definition: "in-brick voxel steps summed over the bricks a front-to-back draw shades",
    fullBrickProxy: {
      mean: mean(perRaySteps), median: quantile(sortedSteps, 0.5), p90: quantile(sortedSteps, 0.9),
      maximum: sortedSteps[sortedSteps.length - 1] ?? 0,
    },
    tightProxy: {
      mean: mean(perRayProxySteps), median: quantile(sortedProxySteps, 0.5), p90: quantile(sortedProxySteps, 0.9),
      maximum: sortedProxySteps[sortedProxySteps.length - 1] ?? 0,
    },
  },
};
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
log(JSON.stringify(report.overdraw, null, 2));
log(JSON.stringify(report.projectedBrickPixels, null, 2));
log(`Wrote ${path.relative(repoRoot, outPath)}`);
solver.destroy?.();
device.destroy();
