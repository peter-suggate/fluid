import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { fineLevelSetRedistanceWGSL, WebGPUFineLevelSetRedistance } from
  "../lib/webgpu-octree-fine-levelset-redistance";
import { planFineLevelSetSummaryLeafLookup } from "../lib/webgpu-octree-fine-levelset-summary";
import { WebGPUFineLevelSetLeafSeeds, WebGPUFineLevelSetTopology,
  fineLevelSetLeafSeedWGSL, makeFineLevelSetTopologyWGSL, planFineLevelSetLeafBrickBounds,
  unpackFineLevelSetGPUTopologyControl } from "../lib/webgpu-octree-fine-levelset-topology";

test("factor-8 B4 source contracts map, activate, redistance, summarize, and gate publication", () => {
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [60, 45, 40],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 64 });
  assert.deepEqual(plan.brickDimensions, [120, 90, 80]);
  assert.deepEqual(planFineLevelSetLeafBrickBounds(plan, [17, 12, 9], 1), {
    first: [34, 24, 18], last: [35, 25, 19], bricksPerFinestCell: 2, brickCount: 8,
  });
  assert.match(fineLevelSetLeafSeedWGSL,
    /let first=origin\*params\.header\.x\/params\.header\.y;[\s\S]*last\/=params\.header\.y/);

  const encode = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /fineFactor!==4&&this\.source\.plan\.fineFactor!==8/);
  assert.match(encode, /indirectRun\(this\.requestPipeline[\s\S]*this\.finishActivationPipeline/,
    "factor-8 uses the same bounded march-driven page activation as factor-4");
  assert.match(fineLevelSetRedistanceWGSL,
    /flags\[index\]=select\(0u,state,all\(q<p\.sampleDims\)\)/,
    "new B4 pages clip authority against the doubled factor-8 sample dimensions");

  assert.deepEqual(planFineLevelSetSummaryLeafLookup(plan.brickDimensions,
    plan.finestCellDimensions, [17, 12, 9], 1, plan.samplesPerBrick), {
    level: 1, key: 864_000 + 17 + 60 * (12 + 45 * 9), brickSide: 2,
    expectedBrickCount: 8, expectedSampleCount: 512,
  });
  const topology = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  );
  assert.match(topology,
    /redistanceValid=arrayLength\(&redistanceControl\)>=4u&&redistanceControl\[0\]==0u&&redistanceControl\[2\]>0u&&redistanceControl\[3\]!=0u/,
    "factor-8 topology remains provisional until the full redistance generation commits");

  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const construction = projection.match(/this\.globalFineVelocityPrepass = new WebGPUOctreePowerVelocityPrepass[\s\S]*?this\.globalFineTransportB = new WebGPUFineLevelSetTransport[\s\S]*?\);/)?.[0];
  assert.ok(construction, "production fine-velocity construction block must exist");
  assert.doesNotMatch(construction, /fineFactor\s*===\s*4/,
    "factor-8 must construct the regular/transition face publication rather than bypassing it");
  assert.match(construction,
    /new WebGPUOctreeFaceFastMarch\([\s\S]*?this\.powerFaces\.plan\.faceCapacity/,
    "the factor-8 face marcher must retain the bounded generalized-face capacity");
  assert.match(projection,
    /this\.globalFineFaceFastMarch\.encodePhase\([\s\S]*?powerFaces:\s*this\.powerFaces\.source/,
    "the production factor-8 publication must receive the authoritative power-face source");
});

test("Dawn publishes a production-sized factor-8 interface and one-ring with bounded parallel page initialization", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const maximumResidentBricks = Math.min(65_535, device.limits.maxComputeWorkgroupsPerDimension);
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [60, 45, 40],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks });
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.initializeEmptyGPUGeneration(1); const next = owner.prepareGPUGeneration(2);

  // One compact SurfaceLeaf per x/z column along a horizontal interface.
  // At factor eight each leaf covers 2^3 globally keyed bricks; the topology
  // pass must deduplicate these and add the paper's complete block one-ring.
  const leafCount = 60 * 40; const leafBytes = new ArrayBuffer(leafCount * 64);
  const words = new Uint32Array(leafBytes); const floats = new Float32Array(leafBytes);
  for (let z = 0; z < 40; z += 1) for (let x = 0; x < 60; x += 1) {
    const row = x + 60 * z, base = row * 16;
    words[base] = x; words[base + 1] = 10; words[base + 2] = z; words[base + 3] = 1; words[base + 4] = 2;
    floats[base + 8] = 0; floats[base + 10] = 1;
  }
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const leaves = device.createBuffer({ size: leafBytes.byteLength, usage: storage });
  const rowCount = device.createBuffer({ size: 4, usage: storage });
  device.queue.writeBuffer(leaves, 0, leafBytes); device.queue.writeBuffer(rowCount, 0, new Uint32Array([leafCount]));
  const seeds = new WebGPUFineLevelSetLeafSeeds(device, next);
  const topology = new WebGPUFineLevelSetTopology(device, current, next,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.y-10.5;}");
  const headerReadback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  topology.encode(encoder, seeds.encodeFromAllInterfaceLeaves(encoder, { buffer: leaves }, { buffer: rowCount }));
  encoder.copyBufferToBuffer(topology.control, 0, headerReadback, 0, 32);
  encoder.copyBufferToBuffer(next.worklist, 0, headerReadback, 32, 20);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await headerReadback.mapAsync(GPUMapMode.READ);
  const header = new Uint32Array(headerReadback.getMappedRange().slice(0)); headerReadback.unmap();
  const control = unpackFineLevelSetGPUTopologyControl(header);
  assert.equal(control.flags, 0); assert.equal(control.published, true); assert.equal(control.rolledBack, false);
  assert.ok(control.desiredBricks > leafCount * 4); assert.ok(control.desiredBricks <= maximumResidentBricks);
  assert.equal(header[8], control.desiredBricks); assert.equal(header[9], 2);

  const sampleBytes = control.desiredBricks * plan.samplesPerBrick * 4;
  const phiReadback = device.createBuffer({ size: sampleBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const sampleEncoder = device.createCommandEncoder();
  sampleEncoder.copyBufferToBuffer(next.phi, 0, phiReadback, 0, sampleBytes);
  device.queue.submit([sampleEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await phiReadback.mapAsync(GPUMapMode.READ);
  const phi = new Float32Array(phiReadback.getMappedRange()); let negative = 0, positive = 0;
  for (const value of phi) { assert.ok(Number.isFinite(value)); if (value < 0) negative += 1; else positive += 1; }
  phiReadback.unmap(); assert.ok(negative > 0); assert.ok(positive > 0);

  phiReadback.destroy(); headerReadback.destroy(); topology.destroy(); seeds.destroy(); leaves.destroy(); rowCount.destroy();
  owner.destroy(); device.destroy();
});
