#!/usr/bin/env node
/**
 * What live-scene maintenance costs, per stage, against record count.
 *
 *   npm run benchmark:svo-live-maintenance
 *   FLUID_SVO_MAINTENANCE_RECORDS=501,5010 node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/benchmark-svo-live-maintenance.ts
 *
 * W2 replaced two loops whose cost did not depend on how much of the world
 * changed:
 *
 *   - invalidation ran one thread per *leaf* and tested each against every dirty
 *     region — O(leaves x regions);
 *   - binning ran one thread per (dirty brick, record) pair — O(bricks x records).
 *
 * Both are affordable at the hero garden's 501 records and neither survives ten
 * times that, which is the whole point of the acceptance scene. The replacements
 * are a dirty-region brick lattice descended to its leaf, and a coarse record
 * grid a brick reads one cell of.
 *
 * A refactor that is theoretically sub-linear and measures flat has discharged
 * nothing, so this lane measures it: both paths are still compiled, and it runs
 * each against both record counts on one device, in one lock, with GPU
 * timestamps. The dirty regions are held *identical* across record counts on
 * purpose — the axis under test is the record count, and letting the regions
 * move with it would price two changes as one.
 *
 * Environment
 *   WEBGPU_NODE_MODULE                 path to the Dawn node module
 *   FLUID_SVO_MAINTENANCE_SCENE        scene preset id (default hero-garden-hose)
 *   FLUID_SVO_MAINTENANCE_RECORDS      comma-separated record counts (default 501,5010)
 *   FLUID_SVO_MAINTENANCE_REPEATS      timed publications per configuration (default 33)
 *   FLUID_SVO_MAINTENANCE_MOVED        records moved by the edit-burst scenario (default 64)
 *   FLUID_SVO_MAINTENANCE_OUT          optional JSON report path
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getScenePreset } from "../lib/core/scenes";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import {
  ENVIRONMENT_VOXEL_MATERIAL_BASE,
  planOctreeLiveSceneRecordIndex,
  type OctreeSparseBrickWorld,
} from "../lib/svo/webgpu-svo-sparse-bricks";
import {
  SPARSE_SCENE_CLUSTER_CAPACITY,
  SPARSE_SCENE_MAINTENANCE_STAGES,
  SparseSceneProxyVoxelizer,
  sparseScenePrimitiveBounds,
  sparseScenePrimitiveForProxy,
  type SparseSceneAxisAlignedBounds,
  type SparseScenePrimitive,
  type SparseSceneRecordIndexOptions,
} from "../lib/core/webgpu-sparse-scene-proxies";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";

const scenePresetId = process.env.FLUID_SVO_MAINTENANCE_SCENE ?? "hero-garden-hose";
const requestedRecordCounts = (process.env.FLUID_SVO_MAINTENANCE_RECORDS ?? "501,5010")
  .split(",").map((value) => Number(value.trim())).filter((value) => Number.isSafeInteger(value) && value > 0);
const repeats = Number(process.env.FLUID_SVO_MAINTENANCE_REPEATS ?? 33);
const movedRecords = Number(process.env.FLUID_SVO_MAINTENANCE_MOVED ?? 64);
const outPath = process.env.FLUID_SVO_MAINTENANCE_OUT;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

const log = (message: string) => process.stderr.write(`${message}\n`);

// The benchmark's own arenas. Deliberately larger than the world's so a ten-times
// load can be published against a hero-sized tree without reallocating one.
const PRIMITIVE_CAPACITY = 16_384;
const CLUSTER_CAPACITY = SPARSE_SCENE_CLUSTER_CAPACITY;
/** The record packer's own ceiling on aggregate blocks; see `syntheticRecords`. */
const CLUSTER_LIMIT = SPARSE_SCENE_CLUSTER_CAPACITY;
const DIRTY_REGION_CAPACITY = 4_096;
const QUERY_COUNT = 2 * SPARSE_SCENE_MAINTENANCE_STAGES.length;

const { adapterInfo, device, timestampsSupported, validationErrors } = await createDawnRenderDevice({
  modulePath, label: "SVO live maintenance benchmark", requireTimestampQuery: true,
});
log(`Adapter: ${JSON.stringify(adapterInfo)} timestamps=${timestampsSupported}`);

const scene = getScenePreset(scenePresetId).create();
const environmentId = (scene.environment ?? "default") as Parameters<typeof buildEnvironmentProxyCatalog>[1];
const proxies = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId), true);
const solver = await WebGPULiveSvoScene.create(device, scene, "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`));
const world = (solver as unknown as { world: OctreeSparseBrickWorld }).world;

// One converged publication, so the tree under measurement is the real one.
{
  const encoder = device.createCommandEncoder({ label: "Maintenance benchmark warm publication" });
  solver.encodeSceneMaintenance(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

const domain = solver.sparseVoxelSceneSource!.structural!.domain;
const tree = world.tree;
const brickSize = tree.brickSize;
const worldOrigin = domain.worldOrigin_m as [number, number, number];
const cellSize = domain.cellSize_m as [number, number, number];
const extent = domain.dimensionsCells.map((cells, axis) => cells * cellSize[axis]) as [number, number, number];
const brickDimensions = domain.dimensionsCells.map((cells) => Math.ceil(cells / brickSize)) as [number, number, number];
const recordIndex: SparseSceneRecordIndexOptions = planOctreeLiveSceneRecordIndex(brickDimensions, cellSize, brickSize);

/**
 * A synthetic load of `count` records, built by tiling the authored set.
 *
 * Copies are jittered inside the domain rather than stacked, so the coarse grid
 * is asked a real question: a load that all lands in one cell would flatter the
 * index, and one spread perfectly uniformly would flatter it differently.
 *
 * Aggregates past the arena's 1,024 blocks voxelize as their envelope ellipsoid
 * — the downgrade W2 removed — because the record packer refuses more. It costs
 * this measurement nothing: invalidation and binning read a record's *bounds*,
 * and an aggregate's bounds are exactly its envelope's. It does mean the rebuild
 * column is not comparable across record counts, and that column is not the
 * claim; the aggregate ceiling is a handoff item for W4.
 */
function syntheticRecords(count: number): SparseScenePrimitive[] {
  const authored = proxies.map((proxy, index) => sparseScenePrimitiveForProxy(proxy, {
    materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + index, ownerId: index,
  }));
  const records: SparseScenePrimitive[] = [];
  let aggregates = 0;
  let seed = 0x9e3779b9;
  const next = () => {
    seed = (Math.imul(seed ^ (seed >>> 15), 0x2545f491) + 0x9e3779b9) >>> 0;
    return seed / 0x100000000;
  };
  for (let index = 0; index < count; index += 1) {
    const source = authored[index % authored.length];
    const copy = Math.floor(index / authored.length);
    let record: SparseScenePrimitive = {
      ...source,
      materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + (index % 0xf000),
      ownerId: index % 0xffff,
    };
    if (record.kind === "smooth-union-cluster") {
      if (aggregates < CLUSTER_LIMIT) aggregates += 1;
      else record = { ...record, kind: "ellipsoid", radii: record.lobeRadii } as SparseScenePrimitive;
    }
    if (copy === 0) { records.push(record); continue; }
    records.push({
      ...record,
      center: source.center.map((value, axis) =>
        worldOrigin[axis] + (((value - worldOrigin[axis]) + (next() - 0.5) * extent[axis]) % extent[axis] + extent[axis]) % extent[axis],
      ) as [number, number, number],
    });
  }
  return records;
}

interface Scenario { id: string; regions: SparseSceneAxisAlignedBounds[] }

/**
 * Both scenarios use the *smallest* record set's bounds, so the dirty regions
 * are byte-identical across record counts and only the binning axis moves.
 */
const baselineRecords = syntheticRecords(Math.min(...requestedRecordCounts));
const baselineBounds = baselineRecords.map(sparseScenePrimitiveBounds);

/**
 * One region per bound plus one for where it moved to, which is what
 * `stageLivePrimitiveUpdates` publishes for a moved record.
 *
 * Region count is the axis invalidation actually scales on — the old kernel
 * tested every leaf against every region — and it scales with the *edit*, not
 * with the scene, so it is swept separately from the record count.
 */
function editBurst(moved: number): SparseSceneAxisAlignedBounds[] {
  const stride = Math.max(1, Math.floor(baselineBounds.length / Math.max(1, moved)));
  const regions: SparseSceneAxisAlignedBounds[] = [];
  for (let index = 0; index < baselineBounds.length && regions.length < 2 * moved; index += stride) {
    const bounds = baselineBounds[index];
    const shift = 0.15;
    regions.push(bounds, {
      minimum: bounds.minimum.map((value, axis) => value + shift * cellSize[axis] * brickSize) as [number, number, number],
      maximum: bounds.maximum.map((value, axis) => value + shift * cellSize[axis] * brickSize) as [number, number, number],
    });
  }
  return regions;
}

const scenarios: Scenario[] = [
  { id: `edit-${movedRecords}`, regions: editBurst(movedRecords) },
  { id: `edit-${8 * movedRecords}`, regions: editBurst(8 * movedRecords) },
  {
    id: "whole-scene",
    regions: [{
      minimum: worldOrigin,
      maximum: worldOrigin.map((value, axis) => value + extent[axis]) as [number, number, number],
    }],
  },
];

const querySet = device.createQuerySet({ label: "Maintenance stage timestamps", type: "timestamp", count: QUERY_COUNT });
const resolveBuffer = device.createBuffer({
  label: "Maintenance timestamp resolve", size: QUERY_COUNT * 8,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});
const timestampReadback = device.createBuffer({
  label: "Maintenance timestamp readback", size: QUERY_COUNT * 8,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

/**
 * A high-trimmed mean, not a median and not a plain mean.
 *
 * Both of the obvious statistics are wrong here for opposite reasons. These
 * stages run faster than the adapter's timestamp tick when the index is doing
 * its job, so a *median* of a heavily quantized sample is exactly zero — which
 * reads as free and is not a measurement. A plain *mean* recovers the sub-tick
 * value, and then one warm-up or scheduler outlier among thirty samples moves it
 * by more than the quantity being measured. Dropping the largest few and
 * averaging the rest keeps the averaging and discards the tail.
 */
function trimmedMean(values: readonly number[], dropHighest = 3): number {
  const sorted = [...values].sort((a, b) => a - b).slice(0, Math.max(1, values.length - dropHighest));
  return sorted.length === 0 ? 0 : sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
}

interface Measurement {
  records: number;
  scenario: string;
  mode: "indexed" | "unindexed";
  stages: Record<string, number>;
  stageMaxima: Record<string, number>;
  dirtyBricks: number;
  regionCells: number;
  gridItems: number;
  globalRecords: number;
}

const measurements: Measurement[] = [];
let revision = 0;

async function readWords(buffer: GPUBuffer, offset: number, size: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Maintenance state readback" });
  encoder.copyBufferToBuffer(buffer, offset, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return copy;
}

async function measure(
  records: readonly SparseScenePrimitive[], scenario: Scenario, mode: "indexed" | "unindexed",
): Promise<Measurement> {
  const voxelizer = new SparseSceneProxyVoxelizer(device, tree, {
    cellSize, worldOrigin, finestLevel: domain.maximumDepth,
    primitiveCapacity: PRIMITIVE_CAPACITY,
    dirtyRegionCapacity: DIRTY_REGION_CAPACITY,
    dirtyBrickCapacity: tree.leafCapacity,
    candidatesPerDirtyBrick: 64,
    clusterCapacity: CLUSTER_CAPACITY,
    recordIndex: mode === "indexed" ? recordIndex : undefined,
    label: `Maintenance benchmark ${mode}`,
  });
  await voxelizer.initializePipelines();
  const samples: number[][] = SPARSE_SCENE_MAINTENANCE_STAGES.map(() => []);
  for (let iteration = 0; iteration < repeats + 2; iteration += 1) {
    revision += 1;
    voxelizer.publish({ primitives: records, dirtyRegions: scenario.regions, revision, budgeted: false });
    const encoder = device.createCommandEncoder({ label: `Maintenance ${mode} ${scenario.id}` });
    voxelizer.encodeMaintenance(encoder, { querySet, baseIndex: 0 });
    encoder.resolveQuerySet(querySet, 0, QUERY_COUNT, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, timestampReadback, 0, QUERY_COUNT * 8);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await timestampReadback.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(timestampReadback.getMappedRange().slice(0));
    timestampReadback.unmap();
    if (iteration < 2) continue; // warm-up: first submits pay pipeline residency
    SPARSE_SCENE_MAINTENANCE_STAGES.forEach((_, stage) => {
      const begin = stamps[2 * stage];
      const end = stamps[2 * stage + 1];
      samples[stage].push(end > begin ? Number(end - begin) / 1e6 : 0);
    });
  }
  const binding = voxelizer.maintenanceBinding;
  const stateWords = await readWords(binding.buffer, binding.stateOffsetBytes, 8 * 4);
  const status = voxelizer.indexStatus;
  const stages: Record<string, number> = {};
  const stageMaxima: Record<string, number> = {};
  SPARSE_SCENE_MAINTENANCE_STAGES.forEach((name, stage) => {
    stages[name] = trimmedMean(samples[stage]);
    stageMaxima[name] = Math.max(0, ...samples[stage]);
  });
  voxelizer.destroy();
  return {
    records: records.length, scenario: scenario.id, mode, stages, stageMaxima,
    dirtyBricks: Math.min(stateWords[0], binding.dirtyBrickCapacity),
    regionCells: status.regionCells, gridItems: status.gridItems, globalRecords: status.globalRecords,
  };
}

// Two passes over every configuration, keeping the second. The first is a
// warm-up for the *device*, not for a pipeline: whichever configuration is
// measured first otherwise carries the cost of the run starting, and at these
// magnitudes that cost is larger than the difference under test.
for (let pass = 0; pass < 2; pass += 1) {
  for (const count of requestedRecordCounts) {
    const records = syntheticRecords(count);
    for (const scenario of scenarios) {
      for (const mode of ["unindexed", "indexed"] as const) {
        const result = await measure(records, scenario, mode);
        if (pass === 0) continue;
        measurements.push(result);
        log(`  ${String(count).padStart(6)} records  ${scenario.id.padEnd(12)} regions ${String(scenario.regions.length).padStart(5)} ${mode.padEnd(10)}`
          + `  bricks ${String(result.dirtyBricks).padStart(6)}`
          + `  invalidate ${result.stages.invalidate.toFixed(4)} ms  bin ${result.stages.bin.toFixed(4)} ms`
          + `  rebuild ${result.stages.rebuild.toFixed(3)} ms`);
      }
    }
  }
}

function lookup(records: number, scenario: string, mode: string): Measurement | undefined {
  return measurements.find((entry) => entry.records === records && entry.scenario === scenario && entry.mode === mode);
}

const low = Math.min(...requestedRecordCounts);
const high = Math.max(...requestedRecordCounts);
const scaling = scenarios.map(({ id }) => {
  const growth = (mode: "indexed" | "unindexed", stage: "invalidate" | "bin") => {
    const small = lookup(low, id, mode)?.stages[stage] ?? 0;
    const large = lookup(high, id, mode)?.stages[stage] ?? 0;
    return { small, large, factor: small > 0 ? large / small : 0 };
  };
  return {
    scenario: id,
    recordFactor: high / low,
    invalidate: { unindexed: growth("unindexed", "invalidate"), indexed: growth("indexed", "invalidate") },
    bin: { unindexed: growth("unindexed", "bin"), indexed: growth("indexed", "bin") },
    speedup: {
      invalidate: {
        [low]: (lookup(low, id, "unindexed")?.stages.invalidate ?? 0) / (lookup(low, id, "indexed")?.stages.invalidate || 1),
        [high]: (lookup(high, id, "unindexed")?.stages.invalidate ?? 0) / (lookup(high, id, "indexed")?.stages.invalidate || 1),
      },
      bin: {
        [low]: (lookup(low, id, "unindexed")?.stages.bin ?? 0) / (lookup(low, id, "indexed")?.stages.bin || 1),
        [high]: (lookup(high, id, "unindexed")?.stages.bin ?? 0) / (lookup(high, id, "indexed")?.stages.bin || 1),
      },
    },
  };
});

const report = {
  lane: "svo-live-maintenance-benchmark",
  scene: scenePresetId,
  adapter: adapterInfo,
  tree: { leafCapacity: tree.leafCapacity, brickSize, brickDimensions, maximumDepth: domain.maximumDepth },
  recordIndex,
  repeats,
  measurements,
  scaling,
};
if (outPath) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
  log(`Report written to ${outPath}`);
}
console.log(JSON.stringify(report, null, 2));

querySet.destroy();
resolveBuffer.destroy();
timestampReadback.destroy();
solver.destroy();
device.destroy();
assert.equal(validationErrors.length, 0, validationErrors.join(" | "));
