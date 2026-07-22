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
  assert.match(buildPowerTrajectoryQueriesWGSL, /sampleDirectPowerVelocity/);
  assert.doesNotMatch(buildPowerTrajectoryQueriesWGSL, /texture|readback/i);
  const productionBuilder = makePowerVelocityPrepassBuilderWGSL();
  assert.doesNotMatch(productionBuilder, /nearestOwner|nearestFallback|vertexStart/,
    "missing containing owners must be resolved only by the Section 5 face-band publication");
  assert.match(productionBuilder, /let row=owner\(x\.xyz\);if\(row==0xffffffffu\)/,
    "direct Stage B starts from the exact adaptive owner");
  assert.match(productionBuilder, /let va=neighborVelocity[\s\S]*let vb=neighborVelocity[\s\S]*let vc=neighborVelocity/,
    "transition samples gather only the containing tetrahedron's three neighbor velocities");
  assert.doesNotMatch(productionBuilder, /76u|vertexVelocities/,
    "direct Stage B must not materialize 76 velocity vectors per query");
  assert.doesNotMatch(productionBuilder, /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "recurring direct Stage B must use immutable reads and deterministic reductions, never atomics");
  assert.match(productionBuilder, /var<storage,read>sites:array<SI>/,
    "the cold-built site hash is immutable during recurring transport");
  assert.deepEqual(planOctreePowerVelocityPrepass(4096, 256), {
    queryCapacity: 4096,
    rowCapacity: 1,
    queryBytes: 0,
    vertexVelocityBytes: 0,
    rowDescriptorBytes: 256,
    scratchBytes: 256,
    samplerBytes: 84_080,
    allocatedBytes: 84_336,
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
  assert.equal(OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY, Number.MAX_SAFE_INTEGER);
  assert.equal(velocityChunkCapacity, queryCapacity);
  assert.deepEqual(planOctreePowerVelocityPrepass(velocityChunkCapacity, 256), {
    queryCapacity,
    rowCapacity: 1,
    queryBytes: 0,
    vertexVelocityBytes: 0,
    rowDescriptorBytes: 256,
    scratchBytes: 256,
    samplerBytes: 9_068_656,
    allocatedBytes: 9_068_912,
  });

  const previous = planFineLevelSetGPUTransport(queryCapacity, 4_096);
  const current = planFineLevelSetGPUTransport(queryCapacity, velocityChunkCapacity);
  assert.equal(previous.chunkCount, 108);
  assert.equal(current.chunkCount, 1);
  assert.deepEqual(planFineLevelSetGPUTransportPasses(previous, 4), {
    chunkCount: 108,
    segmentCount: 4,
    passesPerSegment: 5,
    passesPerChunk: 23,
    encodedPasses: 2_487,
  });
  assert.deepEqual(planFineLevelSetGPUTransportPasses(current, 4), {
    chunkCount: 1,
    segmentCount: 4,
    passesPerSegment: 5,
    passesPerChunk: 23,
    encodedPasses: 26,
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
  assert.equal(planFineLevelSetGPUTransportPasses(transport, 4).encodedPasses, 164);

  // Every chunk first zeros its complete fixed-capacity position arena. The
  // activeSample(flat) guard then leaves the non-divisible tail at w=0, and
  // the Stage-B builder rejects that sentinel before any owner/hash query.
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /positions\[local\]=vec4f\(0\);outcomes\[local\]=vec2u\(0u,INVALID\);let flat=chunk\.base\+local;let a=activeSample\(flat\);if\(a\.x==INVALID\)\{return;\}/);
  assert.match(makePowerVelocityPrepassBuilderWGSL(),
    /if\(x\.w<=0\.\)\{results\[i\]=vec4f\(0\.,0\.,0\.,1\.\);statuses\[i\]=VALID\|INACTIVE;/);
});

test("velocity chunk planning respects tighter binding and offset limits", () => {
  const limits = {
    maxStorageBufferBindingSize: 4 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535,
    minStorageBufferOffsetAlignment: 256,
  };
  const capacity = planOctreePowerVelocityChunkCapacity(442_368, limits);
  assert.equal(capacity, 262_144);
  assert.equal((capacity * 16) % limits.minStorageBufferOffsetAlignment, 0);
  assert.ok(capacity * 16 <= limits.maxStorageBufferBindingSize);
  assert.ok((capacity + 16) * 16 > limits.maxStorageBufferBindingSize);
  assert.equal(planOctreePowerVelocityChunkCapacity(7, limits), 7,
    "a single final batch does not need a following aligned slice");
});

async function runSegmentQueries(segmentCount: 4 | 8): Promise<void> {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
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
  prepass.encodeRowDescriptors(encoder, headerBuffer);
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

test("Dawn fails closed when a trajectory query has no containing power owner", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
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
  // Cell x=1 deliberately has no containing row. The paper's air-side value
  // must come from the later Section 5 face-band publication, never cell x=0.
  const positions = upload(new Float32Array([1.5, 0.5, 0.5, 1]));
  const readback = device.createBuffer({ size: 20, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  prepass.encodeRowDescriptors(encoder, headerBuffer);
  prepass.encodeFromPositions(encoder, positions, headerBuffer, velocity.velocities, {
    dimensions: [2, 1, 1], physicalCellSize: 1, maximumLeafSize: 1, queryCount: 1,
  });
  encoder.copyBufferToBuffer(prepass.source.results, 0, readback, 0, 16);
  encoder.copyBufferToBuffer(prepass.source.statuses, 0, readback, 16, 4);
  device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap();
  assert.deepEqual(Array.from(new Float32Array(bytes, 0, 4)), [0, 0, 0, 0]);
  const status = new Uint32Array(bytes, 16, 1)[0];
  assert.equal(status & 0x8000_0000, 0);
  assert.equal(status & 0x1000_0000, 0);
  readback.destroy(); positions.destroy(); headerBuffer.destroy();
  prepass.destroy(); velocity.destroy(); faces.destroy(); topology.destroy(); device.destroy();
});
