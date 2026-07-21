import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  decodeGeneratedOctreePowerCatalog,
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_POWER_TOPOLOGY_ERROR,
  OCTREE_POWER_TOPOLOGY_BOUNDARY_MASK,
  OCTREE_POWER_TOPOLOGY_VALID,
  WebGPUOctreePowerTopology,
  octreePowerTopologyShader,
  planOctreePowerTopology,
  powerCellSpacingIsotropic,
} from "../lib/webgpu-octree-power-topology";

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
  assert.equal(shallow.catalogBytes, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.byteCount - 24 * 4);
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

test("power topology WGSL exposes direct bounded quotient lookup and fail-closed metrics", () => {
  assert.match(octreePowerTopologyShader, /fn resolveDescriptor/);
  assert.match(octreePowerTopologyShader, /index<arrayLength\(&sameOrCoarserDirect\)/);
  assert.match(octreePowerTopologyShader, /index<arrayLength\(&sameOrFinerDirect\)/);
  assert.match(octreePowerTopologyShader, /fn resolveBoundaryEntry/);
  assert.match(octreePowerTopologyShader, /transformBoundaryMask/);
  assert.match(octreePowerTopologyShader, /fn publishPowerTopology/);
  assert.match(octreePowerTopologyShader, /PowerRowMetric\(INVALID/);
});

test("Dawn resolves generated catalog descriptors and rejects misses/anisotropy", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-topology checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
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
  const validDescriptors = [(catalog.lookup[0] | 0x0700_0000) >>> 0, nonCanonicalDescriptor, catalog.lookup[(lookupCount - 1) * 3]];
  const missingDescriptor = 0x7fff_ffff;
  assert.equal(validDescriptors.includes(missingDescriptor), false);
  device.queue.writeBuffer(descriptors, 0, new Uint32Array([
    validDescriptors[0], validDescriptors[1], missingDescriptor, validDescriptors[2],
  ]));
  const readback = device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  topology.encode(encoder, descriptors, 4, [1, 1, 1]);
  encoder.copyBufferToBuffer(topology.control, 0, readback, 0, 32);
  encoder.copyBufferToBuffer(topology.metrics, 0, readback, 32, 64);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const resolvedBytes = readback.getMappedRange().slice(0); readback.unmap();
  const words = new Uint32Array(resolvedBytes);
  const floats = new Float32Array(resolvedBytes);
  assert.deepEqual([...words.slice(0, 5)], [1, 2, 2, 3, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version]);
  validDescriptors.forEach((descriptor, metricIndex) => {
    const metricRow = metricIndex < 2 ? metricIndex : 3;
    const metricOffset = 8 + metricRow * 4;
    const geometryDescriptor = (descriptor & 0x8000_0000) !== 0 ? (descriptor & 0x8000_01ff) >>> 0 : descriptor & 0x3ffff;
    const packed = (geometryDescriptor & 0x8000_0000) !== 0
      ? catalog.sameOrCoarserDirect[geometryDescriptor & 0x1ff]
      : catalog.sameOrFinerDirect[geometryDescriptor];
    const entry = packed & 0xffff;
    assert.equal(words[metricOffset], entry);
    const boundary = (descriptor >>> 16) & OCTREE_POWER_TOPOLOGY_BOUNDARY_MASK;
    assert.equal(words[metricOffset + 1], (OCTREE_POWER_TOPOLOGY_VALID | boundary | (packed >>> 16)) >>> 0);
    assert.equal(floats[metricOffset + 2], catalog.entryVolumes[entry]);
  });
  assert.deepEqual([...words.slice(16, 20)], [0xffff_ffff, 0, 0, OCTREE_POWER_TOPOLOGY_ERROR.lookupMiss]);

  const anisotropicReadback = device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const anisotropicEncoder = device.createCommandEncoder();
  topology.encode(anisotropicEncoder, descriptors, 4, [1, 1.1, 1]);
  anisotropicEncoder.copyBufferToBuffer(topology.control, 0, anisotropicReadback, 0, 32);
  anisotropicEncoder.copyBufferToBuffer(topology.metrics, 0, anisotropicReadback, 32, 64);
  device.queue.submit([anisotropicEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await anisotropicReadback.mapAsync(GPUMapMode.READ);
  const anisotropic = new Uint32Array(anisotropicReadback.getMappedRange().slice(0)); anisotropicReadback.unmap();
  assert.deepEqual([...anisotropic.slice(0, 5)], [4, 0, OCTREE_POWER_TOPOLOGY_ERROR.anisotropicCell, 0,
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version]);
  for (let row = 0; row < 4; row += 1) {
    assert.deepEqual([...anisotropic.slice(8 + row * 4, 12 + row * 4)],
      [0xffff_ffff, 0, 0, OCTREE_POWER_TOPOLOGY_ERROR.anisotropicCell]);
  }
  assert.deepEqual(validationErrors, []);
  topology.destroy(); descriptors.destroy(); readback.destroy(); anisotropicReadback.destroy(); device.destroy();
});
