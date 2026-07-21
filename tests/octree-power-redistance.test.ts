import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { buildOctreePowerCatalog } from "../lib/octree-power-catalog";
import { sitesForSameOrFinerPowerDescriptor } from "../lib/octree-power-descriptor";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import {
  OCTREE_POWER_REDISTANCE_CONTROL_BYTES,
  OCTREE_POWER_REDISTANCE_VALID,
  WebGPUOctreePowerRedistance,
  octreePowerRedistanceShader,
  planOctreePowerRedistance,
  redistanceOctreePowerCatalogCell,
  unpackOctreePowerRedistanceControl,
} from "../lib/octree-power-redistance";
import { WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";

const close = (actual: number, expected: number, tolerance = 5e-5) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);

function transitionEntry() {
  return buildOctreePowerCatalog([{ descriptor: 0, anchorKey: "0,0,0/2", sites: sitesForSameOrFinerPowerDescriptor(0) }]);
}

function causalPlane(catalog = transitionEntry()) {
  const entry = catalog.entries[0], vertexData = catalog.tetrahedronVertexData;
  const selectors = entry.tetrahedra[0];
  const centroid = [0, 1, 2].map((axis) => selectors.reduce((sum, selector) =>
    sum + vertexData[selector * 4 + axis] / 3, 0));
  const length = Math.hypot(...centroid), gradient = centroid.map((value) => -value / length);
  return { entry, vertexData, magnitudes: Array.from({ length: vertexData.length / 4 }, (_, selector) =>
    10 + gradient.reduce((sum, value, axis) => sum + value * vertexData[selector * 4 + axis], 0)) };
}

test("coarse transition redistance exactly reconstructs a planar signed-distance field", () => {
  assert.deepEqual(planOctreePowerRedistance(3), { queryCapacity: 3, resultBytes: 12, statusBytes: 12, allocatedBytes: 72 });
  const { entry, vertexData, magnitudes } = causalPlane();
  const positive = redistanceOctreePowerCatalogCell(entry, vertexData, magnitudes, 1, 1);
  assert.equal(positive.mode, "tetrahedron"); assert.equal(positive.tetrahedron, 0); close(positive.signedDistance, 10, 1e-10);
  const negative = redistanceOctreePowerCatalogCell(entry, vertexData, magnitudes, 1, -1);
  close(negative.signedDistance, -10, 1e-10);
  const sparse = magnitudes.map((value, index) => index === 0 ? value : undefined);
  assert.equal(redistanceOctreePowerCatalogCell(entry, vertexData, sparse, 1, 1).mode, "nearest");
  assert.throws(() => redistanceOctreePowerCatalogCell(entry, vertexData, sparse.map(() => undefined), 1, 1), /known neighbor/);
});

test("coarse tetrahedral redistance is distinct from uniform fine-grid redistance", () => {
  const catalog = buildOctreePowerCatalog([{ descriptor: 0x3ffff, anchorKey: "0,0,0/2",
    sites: sitesForSameOrFinerPowerDescriptor(0x3ffff) }]); const uniform = catalog.entries[0];
  assert.equal(uniform.uniform, true); assert.equal(uniform.tetrahedra.length, 0);
  assert.throws(() => redistanceOctreePowerCatalogCell(uniform, catalog.tetrahedronVertexData,
    Array(catalog.tetrahedronVertexData.length / 4).fill(0), 1, 1), /transition-only/);
  assert.match(octreePowerRedistanceShader, /Tetrahedral coarse-octree redistance|updatePowerDistance/);
  assert.doesNotMatch(octreePowerRedistanceShader, /fine|page|brick/i);
});

test("Dawn matches the CPU tetrahedral update and counts nearest fallback", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-redistance checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const compilation = await device.createShaderModule({ code: octreePowerRedistanceShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const topology = new WebGPUOctreePowerTopology(device, 1, catalog); const redistance = new WebGPUOctreePowerRedistance(device, 2, topology.source);
  const entry = catalog.sameOrFinerDirect[0] & 0xffff, selectorCount = catalog.tetrahedronVertexData.length / 4;
  const firstTetrahedron = catalog.tetrahedronHeaders[entry * 3];
  const tetrahedronCount = catalog.tetrahedronHeaders[entry * 3 + 1];
  const packed = catalog.tetrahedronData[firstTetrahedron], selectors = [packed & 255, (packed >> 8) & 255, (packed >> 16) & 255];
  const centroid = [0, 1, 2].map((axis) => selectors.reduce((sum, selector) =>
    sum + catalog.tetrahedronVertexData[selector * 4 + axis] / 3, 0));
  const length = Math.hypot(...centroid), gradient = centroid.map((value) => -value / length);
  const magnitudes = new Float32Array(selectorCount * 2); magnitudes.fill(Number.NaN);
  for (let slot = 0; slot < selectorCount; slot += 1) {
    magnitudes[slot] = 10 + gradient.reduce((sum, value, axis) => sum + value * catalog.tetrahedronVertexData[slot * 4 + axis], 0);
  }
  magnitudes[selectorCount] = magnitudes[0];
  const queryBufferData = new ArrayBuffer(2 * 48), words = new Uint32Array(queryBufferData), floats = new Float32Array(queryBufferData);
  for (let query = 0; query < 2; query += 1) {
    const base = query * 12; words.set([0, selectorCount, firstTetrahedron, tetrahedronCount, 0,
      query * selectorCount, selectorCount, query], base); floats.set([1, query === 0 ? -1 : 1, 2e-6, 0], base + 8);
  }
  const upload = (data: ArrayBufferView) => { const buffer = device.createBuffer({ size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer; };
  const queries = upload(new Uint8Array(queryBufferData)), known = upload(magnitudes);
  const readback = device.createBuffer({ size: OCTREE_POWER_REDISTANCE_CONTROL_BYTES + redistance.plan.resultBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); redistance.encode(encoder, queries, known, 2, 31);
  encoder.copyBufferToBuffer(redistance.control, 0, readback, 0, OCTREE_POWER_REDISTANCE_CONTROL_BYTES);
  encoder.copyBufferToBuffer(redistance.results, 0, readback, OCTREE_POWER_REDISTANCE_CONTROL_BYTES, redistance.plan.resultBytes);
  device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0); readback.unmap();
  assert.deepEqual(unpackOctreePowerRedistanceControl(new Uint32Array(result, 0, 8)), {
    flags: OCTREE_POWER_REDISTANCE_VALID, firstError: 0xffff_ffff, queryCount: 2, updatedCount: 2,
    tetrahedronCount: 1, nearestFallbackCount: 1, reserved: 0, generation: 31,
  });
  const distances = new Float32Array(result, OCTREE_POWER_REDISTANCE_CONTROL_BYTES, 2); close(distances[0], -10);
  assert.ok(Number.isFinite(distances[1]) && distances[1] > 0);
  readback.destroy(); queries.destroy(); known.destroy(); redistance.destroy(); topology.destroy(); device.destroy();
});
