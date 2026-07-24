import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
} from "../lib/generated/octree-power-catalog";
import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  OCTREE_POWER_TOPOLOGY_ERROR,
  OCTREE_POWER_REGULAR_DESCRIPTOR,
  WebGPUOctreePowerTopology,
  octreePowerTopologyShader,
  planOctreePowerTopology,
  powerCellSpacingIsotropic,
} from "../lib/webgpu-octree-power-topology";
import { createColdPowerRowPublication } from "./webgpu-octree-power-row-delta-fixture";

function catalogViews() {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

test("power topology planner accounts only compact rows and fixed catalog", () => {
  const catalog = catalogViews();
  const shallow = planOctreePowerTopology(100, catalog);
  const deep = planOctreePowerTopology(100, catalog);
  assert.deepEqual(shallow, deep);
  assert.equal(shallow.entryCount, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.configurationCount);
  assert.equal(shallow.lookupCount, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.descriptorCount);
  assert.equal(shallow.metricBytes, 1_600);
  assert.equal(shallow.catalogBytes, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.byteCount - 26 * 4);
  assert.ok(shallow.allocatedBytes < 16 * 1024 * 1024);
});

test("power topology planner rejects malformed catalog lookup metadata", () => {
  const catalog = catalogViews();
  const lookup = catalog.lookup.slice();
  lookup[3] = lookup[0];
  assert.throws(() => planOctreePowerTopology(1, { ...catalog, lookup }), /lookup is invalid/);
});

test("power authority accepts only physically isotropic finest cells", () => {
  assert.equal(powerCellSpacingIsotropic([1, 1, 1]), true);
  assert.equal(powerCellSpacingIsotropic([1, 1 + 1e-6, 1]), true);
  assert.equal(powerCellSpacingIsotropic([1, 1.001, 1]), false);
  assert.equal(powerCellSpacingIsotropic([1, 0, 1]), false);
});

test("power topology WGSL resolves affected rows and compacts exact publication rows in parallel", () => {
  assert.match(octreePowerTopologyShader, /fn resolveDescriptor/);
  assert.match(octreePowerTopologyShader,
    /geometry==REGULAR_DESCRIPTOR&&boundary==0u\)\{return vec2u\(params\.regularPacked&0xffffu,params\.regularPacked>>16u\)/);
  assert.match(octreePowerTopologyShader, /index<arrayLength\(&sameOrCoarserDirect\)/);
  assert.match(octreePowerTopologyShader, /index<arrayLength\(&sameOrFinerDirect\)/);
  assert.match(octreePowerTopologyShader, /fn resolveBoundaryEntry/);
  assert.match(octreePowerTopologyShader, /transformBoundaryMask/);
  assert.match(octreePowerTopologyShader, /fn deltaAccepted/);
  assert.match(octreePowerTopologyShader, /fn stagePowerTopologyDelta/);
  assert.match(octreePowerTopologyShader, /fn prefixPowerTopologyDelta/);
  assert.match(octreePowerTopologyShader, /fn scatterPowerTopologyDelta/);
  assert.doesNotMatch(octreePowerTopologyShader, /fn commitPowerTopology/);
  assert.match(octreePowerTopologyShader,
    /lookupArena\[publicationBase\(\)\+output\]=row;committedMetrics\[row\]=metrics\[row\]/,
    "row-owned scatter must commit the same changed row without a redundant compact dispatch");
  assert.match(octreePowerTopologyShader,
    /candidate\.resolvedCount!=requestedRows\(\)\|\|candidate\.version!=params\.catalogVersion/);
  assert.doesNotMatch(octreePowerTopologyShader,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "row-owned metric status and singleton publication replace recurring topology atomics");
  assert.match(octreePowerTopologyShader, /let row=affectedRow\(item\)/);
  assert.match(octreePowerTopologyShader, /lookupArena\[statusBase\(\)\+row\]=STATUS_LISTED/);
  assert.match(octreePowerTopologyShader,
    /let listed=\(lookupArena\[statusBase\(\)\+row\]&STATUS_LISTED\)!=0u/);
  assert.doesNotMatch(octreePowerTopologyShader, /fn rowListedAffected/,
    "the exact affected-list proof must not binary-search the list once per live row");
  assert.match(octreePowerTopologyShader,
    /if\(old!=row\)\{metrics\[row\]=metric;status\|=STATUS_PUBLISH;\}/);
  assert.match(octreePowerTopologyShader,
    /commitDispatch=dispatchFor\(published\)/);
  assert.doesNotMatch(octreePowerTopologyShader,
    /carryPowerTopology|summarizePowerTopology|publishPowerTopology|controlArena\.moved/);
  assert.doesNotMatch(octreePowerTopologyShader, /for\(var row=0u;row<requested/);
  assert.match(octreePowerTopologyShader, /workgroupBarrier/);
  assert.match(octreePowerTopologyShader, /PowerRowMetric\(INVALID/);
});

test("power topology resolve dispatch is sourced from the exact row delta", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-power-topology.ts", import.meta.url), "utf8");
  assert.match(source,
    /copyBufferToBuffer\(rowDelta\.rows,\s*\(rowDelta\.controlOffsetWords \+ 12\) \* 4,\s*this\.workDispatch,\s*0,\s*12\)/,
    "row-delta control words 12-14 are the producer-owned affected-row dispatch");
  assert.doesNotMatch(source,
    /copyBufferToBuffer\(this\.control,\s*OCTREE_POWER_TOPOLOGY_CONTROL_BYTES,\s*this\.workDispatch,\s*0,\s*12\)/,
    "topology control byte 32 is retained authority, not a resolve dispatch");
  assert.match(source,
    /copyBufferToBuffer\(this\.control,\s*64,\s*this\.workDispatch,\s*0,\s*12\)/,
    "the topology-owned block dispatch is at control byte 64");
  assert.doesNotMatch(source,
    /copyBufferToBuffer\(this\.control,\s*80,\s*this\.workDispatch,\s*0,\s*12\)/,
    "row-owned scatter must not copy or launch a redundant compact commit dispatch");
});

test("Dawn resolves generated catalog descriptors and rejects misses/anisotropy", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-topology checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  const shaderModule = device.createShaderModule({ code: octreePowerTopologyShader });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const catalog = catalogViews();
  const topology = new WebGPUOctreePowerTopology(device, 4, catalog);
  const descriptors = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const lookupCount = catalog.lookup.length / 3;
  const canonicalDescriptors = new Set(Array.from({ length: lookupCount }, (_, index) => catalog.lookup[index * 3]));
  const nonCanonicalDescriptor = Array.from({ length: 1 << 18 }, (_, descriptor) => descriptor)
    .find((descriptor) => !canonicalDescriptors.has(descriptor))!;
  // Geometry 961's canonical entry has the reflected -X boundary selector in
  // the immutable catalog (entry 1021, canonical mask 32).
  const validDescriptors = [(961 | 0x0100_0000) >>> 0, nonCanonicalDescriptor, OCTREE_POWER_REGULAR_DESCRIPTOR];
  const missingDescriptor = 0x7fff_ffff;
  const cold = createColdPowerRowPublication(device, 4, 4, 1);
  assert.equal(validDescriptors.includes(missingDescriptor), false);
  const run = async (spacing: readonly [number, number, number]) => {
    const readback = device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    topology.encode(broker, descriptors, cold.rowCount, spacing, cold.rowDelta);
    broker.fence();
    encoder.copyBufferToBuffer(topology.control, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(topology.metrics, 0, readback, 32, 64);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
    return bytes;
  };

  device.queue.writeBuffer(descriptors, 0, new Uint32Array(4).fill(OCTREE_POWER_REGULAR_DESCRIPTOR));
  const accepted = await run([1, 1, 1]);
  const acceptedWords = new Uint32Array(accepted), acceptedFloats = new Float32Array(accepted);
  assert.deepEqual([...acceptedWords.slice(0, 5)],
    [0, 0xffff_ffff, 0, 4, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version]);
  const regularPacked = catalog.sameOrFinerDirect[OCTREE_POWER_REGULAR_DESCRIPTOR];
  const regularEntry = regularPacked & 0xffff;
  for (let row = 0; row < 4; row += 1) {
    const offset = 8 + row * 4;
    assert.equal(acceptedWords[offset], regularEntry);
    assert.equal(acceptedWords[offset + 1], (0x8000_0000 | (regularPacked >>> 16)) >>> 0);
    assert.equal(acceptedFloats[offset + 2], catalog.entryVolumes[regularEntry]);
    assert.equal(acceptedWords[offset + 3], 0);
  }

  device.queue.writeBuffer(descriptors, 0, new Uint32Array([
    validDescriptors[0], validDescriptors[1], missingDescriptor, validDescriptors[2],
  ]));
  const resolvedBytes = await run([1, 1, 1]);
  const words = new Uint32Array(resolvedBytes);
  assert.deepEqual([...words.slice(0, 5)], [1, 2, 2, 3, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version]);
  assert.deepEqual([...words.slice(8, 24)], [...acceptedWords.slice(8, 24)],
    "a rejected generation must retain the previous immutable metrics");

  const anisotropic = new Uint32Array(await run([1, 1.1, 1]));
  assert.deepEqual([...anisotropic.slice(0, 5)], [4, 0, OCTREE_POWER_TOPOLOGY_ERROR.anisotropicCell, 0,
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version]);
  assert.deepEqual([...anisotropic.slice(8, 24)], [...acceptedWords.slice(8, 24)],
    "a rejected anisotropic generation must leave immutable authority untouched");
  assert.deepEqual(validationErrors, []);
  cold.destroy(); topology.destroy(); descriptors.destroy();
  device.destroy();
});
