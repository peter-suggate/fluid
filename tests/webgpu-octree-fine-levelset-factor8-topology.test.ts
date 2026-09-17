import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planFineLevelSetBricks } from "../lib/methods/octree-shared/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "../lib/core/webgpu-pass-broker";
import { planGlobalFineNarrowBandBrickCapacity, resolveGlobalFineBrickCapacity } from
  "../lib/methods/octree-shared/octree-fine-band-capacity";
import { WebGPUFineLevelSetRedistance } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-redistance";
import { planFineLevelSetSummaryLeafLookup } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-summary";
import { WebGPUFineLevelSetLeafSeeds, WebGPUFineLevelSetTopology,
  fineLevelSetLeafSeedWGSL, makeFineLevelSetTopologyWGSL, planFineLevelSetLeafBrickBounds,
  unpackFineLevelSetGPUTopologyControl } from "../lib/methods/octree-shared/webgpu-octree-fine-levelset-topology";


test("Dawn production-width factor-8 topology publishes the complete twelve-ring support within planned capacity", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const brickDimensions = [120, 90, 80] as const;
  const capacityPlan = planGlobalFineNarrowBandBrickCapacity(brickDimensions, 12);
  const maximumResidentBricks = resolveGlobalFineBrickCapacity(
    capacityPlan.maximumResidentBricks, undefined,
    device.limits.maxComputeWorkgroupsPerDimension, 64,
    Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize), 64,
  );
  assert.equal(maximumResidentBricks, 405_000,
    "the production physical-band plan reserves 25 layers plus 50% surface-growth headroom");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [60, 45, 40],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks });
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.initializeEmptyGPUGeneration(1); const next = owner.prepareGPUGeneration(2);

  // One compact FineSeedLeaf per x/y column along the domain's largest planar
  // interface. This stresses the same maximum-area orientation used by the
  // production physical-band capacity planner.
  // At factor eight each leaf covers 2^3 globally keyed bricks; the topology
  // pass must deduplicate these and add the complete production support: the
  // factor-8 default has max(8 backtrace + 1 interpolation, 43 redistance)
  // cells, rounded to eleven B4 rings plus one publication-safety ring.
  const leafCount = 60 * 45; const leafBytes = new ArrayBuffer(leafCount * 64);
  const words = new Uint32Array(leafBytes); const floats = new Float32Array(leafBytes);
  for (let y = 0; y < 45; y += 1) for (let x = 0; x < 60; x += 1) {
    const row = x + 60 * y, base = row * 16;
    words[base] = x; words[base + 1] = y; words[base + 2] = 10; words[base + 3] = 1; words[base + 4] = 2;
    floats[base + 8] = 0; floats[base + 11] = 1;
  }
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const leaves = device.createBuffer({ size: leafBytes.byteLength, usage: storage });
  const rowCount = device.createBuffer({ size: 8, usage: storage });
  device.queue.writeBuffer(leaves, 0, leafBytes); device.queue.writeBuffer(rowCount, 0, new Uint32Array([leafCount]));
  const seeds = new WebGPUFineLevelSetLeafSeeds(device, next);
  const topology = new WebGPUFineLevelSetTopology(device, current, next,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.z-10.5;}");
  const headerReadback = device.createBuffer({ size: 56, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  topology.encode(broker, seeds.encodeFromAllInterfaceLeaves(broker, { buffer: leaves }, { buffer: rowCount }), [], {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 43,
    safetyBrickRings: 1,
  });
  encoder.copyBufferToBuffer(topology.control, 0, headerReadback, 0, 36);
  encoder.copyBufferToBuffer(next.worklist, 0, headerReadback, 36, 20);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await headerReadback.mapAsync(GPUMapMode.READ);
  const header = new Uint32Array(headerReadback.getMappedRange().slice(0)); headerReadback.unmap();
  const control = unpackFineLevelSetGPUTopologyControl(header);
  assert.equal(control.flags, 0, JSON.stringify({ header: [...header], control }));
  assert.equal(control.published, true); assert.equal(control.rolledBack, false);
  assert.equal(control.interfaceBricks, 21_600,
    "the cold checkpoint must retain the exact externally seeded interface-brick population");
  assert.equal(control.desiredBricks, 280_800,
    "a maximum-area x/y factor-8 plane dilated by twelve rings occupies exactly 120 x 90 x 26 bricks");
  assert.equal(control.requiredDesiredBricks, control.desiredBricks);
  assert.equal(control.requiredDesiredBricksExact, true);
  assert.equal(control.dilationBrickRings, 12);
  assert.equal(maximumResidentBricks - control.desiredBricks, 56_700);
  assert.ok(control.desiredBricks <= maximumResidentBricks);
  assert.equal(header[8], control.interfaceSeedBricks);
  assert.equal(header[9], 2, "published worklist must carry generation 2");
  assert.equal(header[10], control.desiredBricks);

  headerReadback.destroy(); topology.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy();
  owner.destroy(); device.destroy();
});
