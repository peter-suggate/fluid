import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { WebGPUOctreePowerTopology, OCTREE_POWER_TOPOLOGY_VALID } from "../lib/webgpu-octree-power-topology";
import { WebGPUOctreePowerFaces } from "../lib/webgpu-octree-power-faces";
import { WebGPUOctreePowerVelocity } from "../lib/webgpu-octree-power-velocity";
import {
  OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY,
  WebGPUOctreePowerVelocityPrepass,
  buildPowerTrajectoryQueriesWGSL,
  makePowerVelocityPrepassBuilderWGSL,
  planOctreePowerVelocityChunkCapacity,
  planOctreePowerVelocityPrepass,
} from "../lib/webgpu-octree-power-velocity-prepass";
import {
  fineLevelSetGPUQueryTransportWGSL,
  planFineLevelSetGPUTransport,
  planFineLevelSetGPUTransportPasses,
} from "../lib/webgpu-octree-fine-levelset-transport";

test("trajectory prepass is bounded GPU-only Stage-B work", () => {
  assert.match(buildPowerTrajectoryQueriesWGSL, /buildPowerTrajectoryQueries/);
  assert.doesNotMatch(buildPowerTrajectoryQueriesWGSL, /texture|readback/i);
  assert.deepEqual(planOctreePowerVelocityPrepass(4096, 256), {
    queryCapacity: 4096,
    queryBytes: 196_608,
    vertexVelocityBytes: 4_980_736,
    scratchBytes: 5_177_344,
    samplerBytes: 81_968,
    allocatedBytes: 5_259_376,
  });
});

test("factor-4 small dam break batches paper Section 5 transport within portable limits", () => {
  const queryCapacity = 24 * 18 * 16 * 4 ** 3;
  assert.equal(queryCapacity, 442_368);
  const portableLimits = {
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535,
    minStorageBufferOffsetAlignment: 256,
  };
  const velocityChunkCapacity = planOctreePowerVelocityChunkCapacity(queryCapacity, portableLimits);
  assert.equal(OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY, 65_536);
  assert.equal(velocityChunkCapacity, 65_536);
  assert.deepEqual(planOctreePowerVelocityPrepass(velocityChunkCapacity, 256), {
    queryCapacity: 65_536,
    queryBytes: 3_145_728,
    vertexVelocityBytes: 79_691_776,
    scratchBytes: 82_837_504,
    samplerBytes: 1_310_768,
    allocatedBytes: 84_148_336,
  });

  const previous = planFineLevelSetGPUTransport(queryCapacity, 4_096);
  const current = planFineLevelSetGPUTransport(queryCapacity, velocityChunkCapacity);
  assert.equal(previous.chunkCount, 108);
  assert.equal(current.chunkCount, 7);
  assert.deepEqual(planFineLevelSetGPUTransportPasses(previous, 4), {
    chunkCount: 108,
    segmentCount: 4,
    passesPerSegment: 6,
    passesPerChunk: 26,
    encodedPasses: 2_810,
  });
  assert.deepEqual(planFineLevelSetGPUTransportPasses(current, 4), {
    chunkCount: 7,
    segmentCount: 4,
    passesPerSegment: 6,
    passesPerChunk: 26,
    encodedPasses: 184,
  });
});

test("non-divisible final transport chunk keeps its tail inactive without dropping paper segments", () => {
  const queryCapacity = 24 * 18 * 16 * 4 ** 3;
  const chunkCapacity = 65_536;
  const transport = planFineLevelSetGPUTransport(queryCapacity, chunkCapacity);
  const finalChunkBase = (transport.chunkCount - 1) * chunkCapacity;
  const finalChunkLive = queryCapacity - finalChunkBase;
  const finalChunkInactive = chunkCapacity - finalChunkLive;
  assert.deepEqual({ finalChunkBase, finalChunkLive, finalChunkInactive }, {
    finalChunkBase: 393_216,
    finalChunkLive: 49_152,
    finalChunkInactive: 16_384,
  });
  assert.equal(planFineLevelSetGPUTransportPasses(transport, 4).encodedPasses, 184);

  // Every chunk first zeros its complete fixed-capacity position arena. The
  // activeSample(flat) guard then leaves the non-divisible tail at w=0, and
  // the Stage-B builder rejects that sentinel before any owner/hash query.
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /positions\[local\]=vec4f\(0\);let flat=chunk\.base\+local;let a=activeSample\(flat\);if\(a\.x==INVALID\)\{return;\}/);
  assert.match(makePowerVelocityPrepassBuilderWGSL(),
    /if\(x\.w<=0\.0\)\{word\(qb\+4u,0x20000000u\);[^}]*return;\}/);
});

test("velocity chunk planning respects tighter binding and offset limits", () => {
  const limits = {
    maxStorageBufferBindingSize: 8 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535,
    minStorageBufferOffsetAlignment: 256,
  };
  const capacity = planOctreePowerVelocityChunkCapacity(442_368, limits);
  assert.ok(capacity < OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY);
  assert.equal((capacity * 16) % limits.minStorageBufferOffsetAlignment, 0);
  assert.ok(planOctreePowerVelocityPrepass(capacity, 256).scratchBytes
    <= limits.maxStorageBufferBindingSize);
  assert.ok(planOctreePowerVelocityPrepass(capacity + 16, 256).scratchBytes
    > limits.maxStorageBufferBindingSize);
  assert.equal(planOctreePowerVelocityChunkCapacity(7, limits), 7,
    "a single final batch does not need a following aligned slice");
});

async function runSegmentQueries(segmentCount: 4 | 8): Promise<void> {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const raw = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const topology = new WebGPUOctreePowerTopology(device, 8, catalog);
  const faces = new WebGPUOctreePowerFaces(device, 8, 64, topology.source);
  const velocity = new WebGPUOctreePowerVelocity(device, 8);
  const prepass = new WebGPUOctreePowerVelocityPrepass(device, segmentCount, topology.source, faces.source);
  const entry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metrics = new Uint32Array(32), headers = new Uint32Array(96), values = new Float32Array(32);
  const hash = new Uint32Array(faces.plan.hashCapacity * 4);
  const hashFn = (cell: number, size: number) => { let value = (cell ^ Math.imul(size, 0x9e3779b9)) >>> 0;
    value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb352d) >>> 0;
    value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846ca68b) >>> 0; return (value ^ (value >>> 16)) >>> 0; };
  for (let row = 0; row < 8; row += 1) {
    metrics[row * 4] = entry; metrics[row * 4 + 1] = OCTREE_POWER_TOPOLOGY_VALID;
    headers[row * 12] = row; headers[row * 12 + 3] = 1;
    values.set([(row & 1) + 0.5, ((row >> 1) & 1) + 0.5, ((row >> 2) & 1) + 0.5, 1], row * 4);
    let slot = hashFn(row, 1) & (faces.plan.hashCapacity - 1);
    while (hash[slot * 4] !== 0) slot = (slot + 1) & (faces.plan.hashCapacity - 1);
    hash.set([row + 1, 1, row, 0], slot * 4);
  }
  device.queue.writeBuffer(topology.metrics, 0, metrics); device.queue.writeBuffer(velocity.velocities, 0, values);
  device.queue.writeBuffer(faces.siteIndex, 0, hash);
  const upload = (data: ArrayBufferView<ArrayBuffer>) => { const buffer = device.createBuffer({ size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(buffer, 0, data); return buffer; };
  const headerBuffer = upload(headers), positionData = new Float32Array(segmentCount * 4);
  for (let segment = 0; segment < segmentCount; segment += 1) {
    positionData.set([0.5 + 0.5 * (segment + 1) / segmentCount, 1, 1, 1], segment * 4);
  }
  const positions = upload(positionData), readback = device.createBuffer({ size: segmentCount * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  prepass.encodeFromPositions(encoder, positions, headerBuffer, velocity.velocities,
    { dimensions: [2, 2, 2], physicalCellSize: 1, maximumLeafSize: 1, queryCount: segmentCount });
  encoder.copyBufferToBuffer(prepass.source.results, 0, readback, 0, segmentCount * 16);
  device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
  for (let segment = 0; segment < segmentCount; segment += 1) {
    assert.ok(Math.abs(result[segment * 4] - positionData[segment * 4]) < 1e-5);
    assert.deepEqual(Array.from(result.slice(segment * 4 + 1, segment * 4 + 4)), [1, 1, 1]);
  }
  readback.destroy(); positions.destroy(); headerBuffer.destroy(); prepass.destroy(); velocity.destroy(); faces.destroy(); topology.destroy(); device.destroy();
}

for (const factor of [4, 8] as const) test(`Dawn resolves factor-${factor} segment positions through the Stage-B prepass`, {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, () => runSegmentQueries(factor));

test("Dawn extrapolates a positive-air query from the bounded nearest live power cell", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice();
  const raw = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const topology = new WebGPUOctreePowerTopology(device, 1, catalog);
  const faces = new WebGPUOctreePowerFaces(device, 1, 8, topology.source);
  const velocity = new WebGPUOctreePowerVelocity(device, 1);
  const prepass = new WebGPUOctreePowerVelocityPrepass(device, 1, topology.source, faces.source);
  const entry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metrics = new Uint32Array([entry, OCTREE_POWER_TOPOLOGY_VALID, 0, 0]);
  const headers = new Uint32Array(12); headers[3] = 1;
  const values = new Float32Array([0.5, 0.25, -0.125, 1]);
  const hash = new Uint32Array(faces.plan.hashCapacity * 4);
  let mixed = Math.imul(1, 0x9e3779b9) >>> 0;
  mixed = Math.imul((mixed ^ (mixed >>> 16)) >>> 0, 0x7feb352d) >>> 0;
  mixed = Math.imul((mixed ^ (mixed >>> 15)) >>> 0, 0x846ca68b) >>> 0;
  const slot = (mixed ^ (mixed >>> 16)) & (faces.plan.hashCapacity - 1);
  hash.set([1, 1, 0, 0], slot * 4);
  device.queue.writeBuffer(topology.metrics, 0, metrics);
  device.queue.writeBuffer(velocity.velocities, 0, values);
  device.queue.writeBuffer(faces.siteIndex, 0, hash);
  const upload = (data: ArrayBufferView<ArrayBuffer>) => { const buffer = device.createBuffer({
    size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }); device.queue.writeBuffer(buffer, 0, data); return buffer; };
  const headerBuffer = upload(headers);
  // Cell x=1 deliberately has no containing row; cell x=0 is the bounded
  // nearest live power cell and supplies the extrapolated Stage-B query.
  const positions = upload(new Float32Array([1.5, 0.5, 0.5, 1]));
  const readback = device.createBuffer({ size: 20, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  prepass.encodeFromPositions(encoder, positions, headerBuffer, velocity.velocities, {
    dimensions: [2, 1, 1], physicalCellSize: 1, maximumLeafSize: 1, queryCount: 1,
  });
  encoder.copyBufferToBuffer(prepass.source.results, 0, readback, 0, 16);
  encoder.copyBufferToBuffer(prepass.source.statuses, 0, readback, 16, 4);
  device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap();
  assert.deepEqual(Array.from(new Float32Array(bytes, 0, 4)), Array.from(values));
  assert.notEqual(new Uint32Array(bytes, 16, 1)[0] & 0x1000_0000, 0,
    "nearest-owner extrapolation must be marked in the existing status channel");
  readback.destroy(); positions.destroy(); headerBuffer.destroy();
  prepass.destroy(); velocity.destroy(); faces.destroy(); topology.destroy(); device.destroy();
});
