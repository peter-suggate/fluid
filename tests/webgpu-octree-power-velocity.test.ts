import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { buildOctreePowerCatalog } from "../lib/octree-power-catalog";
import { sitesForSameOrFinerPowerDescriptor } from "../lib/octree-power-descriptor";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import {
  OCTREE_POWER_VELOCITY_CONTROL_BYTES,
  OCTREE_POWER_VELOCITY_ERROR,
  OCTREE_POWER_VELOCITY_VALID,
  WebGPUOctreePowerVelocity,
  WebGPUOctreePowerVelocitySampler,
  octreePowerVelocityPrepareShader,
  octreePowerVelocityPrepareFromFaceControlShader,
  octreePowerVelocityPublishShader,
  octreePowerVelocityShader,
  octreePowerVelocitySampleShader,
  planOctreePowerVelocity,
  planOctreePowerVelocitySamples,
  sampleOctreePowerCatalogVelocity,
  trilinearOctreePowerVelocity,
  unpackOctreePowerVelocityControl,
  unpackOctreePowerVelocitySampleControl,
  OCTREE_POWER_SAMPLE_CONTROL_BYTES,
  OCTREE_POWER_SAMPLE_VALID,
} from "../lib/webgpu-octree-power-velocity";

const INVALID_ROW = 0xffff_ffff;

function close(actual: number, expected: number, tolerance = 5e-5): void {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`);
}

test("power velocity planner allocates only compact row-scaled output", () => {
  const plan = planOctreePowerVelocity(257);
  assert.deepEqual(plan, { rowCapacity: 257, velocityBytes: 257 * 16, statusBytes: 257 * 4,
    allocatedBytes: 257 * 20 + 64 });
  assert.throws(() => planOctreePowerVelocity(0), /positive integer/);
});

test("power velocity shader contains area-weighted normal equations and guarded publication", () => {
  assert.match(octreePowerVelocityShader, /let w=face\.area/);
  assert.match(octreePowerVelocityShader, /conditionEstimate/);
  assert.match(octreePowerVelocityShader, /determinantFloor/);
  assert.match(octreePowerVelocityShader, /rowStatus\[row\]=FALLBACK/);
  assert.match(octreePowerVelocityPublishShader, /control\.flags=VALID/);
  assert.match(octreePowerVelocityPrepareFromFaceControlShader,
    /faceControl\[8\]!=FACE_VALID\|\|faceControl\[3\]!=0u[\s\S]*faceControl\[7\]!=params\.generation/,
    "GPU-derived Stage A must reject invalid or stale power-face generations");
  assert.match(String((WebGPUOctreePowerVelocity.prototype as unknown as { encodePasses: Function }).encodePasses), /clearBuffer/,
    "Stage A clears vectors and row status before every publication attempt");
  assert.match(octreePowerVelocityShader, /velocityControl\[0\]!=0u\|\|row>=params\.rowCount/,
    "an invalid face-authority prepare prevents reconstruction from repopulating cleared vectors");
  assert.doesNotMatch(octreePowerVelocityShader, /faceAxis|axisSpan/);
  assert.match(octreePowerVelocitySampleShader, /tetraWeights/);
  assert.match(octreePowerVelocitySampleShader, /atomicAdd\(&control\.nearest/);
});

test("Stage-B CPU oracle is exact for structured and transition-linear fields", () => {
  assert.deepEqual(planOctreePowerVelocitySamples(7), {
    queryCapacity: 7, resultBytes: 112, statusBytes: 28, allocatedBytes: 188,
  });
  const corners = Array.from({ length: 8 }, (_, corner) => [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1] as const);
  const uniform = trilinearOctreePowerVelocity(corners, [0.25, 0.5, 0.75]);
  uniform.forEach((value, axis) => close(value, [0.25, 0.5, 0.75][axis], 1e-12));

  const sites = sitesForSameOrFinerPowerDescriptor(0);
  const catalog = buildOctreePowerCatalog([{ descriptor: 0, anchorKey: "0,0,0/2", sites }]);
  const entry = catalog.entries[0]; assert.ok(!entry.uniform && entry.tetrahedra.length > 0);
  const tetrahedron = entry.tetrahedra[0];
  const point = tetrahedron.reduce((sum, selector) => sum.map((value, axis) =>
    value + catalog.tetrahedronVertexData[selector * 4 + axis] / 4) as [number, number, number], [0, 0, 0] as [number, number, number]);
  const field = (position: readonly number[]) => [1 + 2 * position[0], -1 + 3 * position[1], 0.5 - 2 * position[2]] as const;
  const positions = Array.from({ length: catalog.tetrahedronVertexData.length / 4 }, (_, selector) =>
    [...catalog.tetrahedronVertexData.slice(selector * 4, selector * 4 + 3)]);
  const sampled = sampleOctreePowerCatalogVelocity(entry, catalog.tetrahedronVertexData, point, field([0, 0, 0]), positions.map(field));
  assert.equal(sampled.mode, "tetrahedron");
  sampled.velocity.forEach((value, axis) => close(value, field(point)[axis], 1e-10));
  const boundary = sampleOctreePowerCatalogVelocity(entry, catalog.tetrahedronVertexData, [0, 0, 0], field([0, 0, 0]), positions.map(field));
  assert.equal(boundary.tetrahedron, 0, "catalog order is the deterministic shared-boundary tie-break");
  const fallback = sampleOctreePowerCatalogVelocity(entry, catalog.tetrahedronVertexData, [100, 100, 100], field([0, 0, 0]), positions.map(field));
  assert.equal(fallback.mode, "nearest");
});

test("transition barycentric interpolation is continuous across a shared local tetrahedron face", () => {
  const sites = sitesForSameOrFinerPowerDescriptor(0);
  const catalog = buildOctreePowerCatalog([{ descriptor: 0, anchorKey: "0,0,0/2", sites }]); const entry = catalog.entries[0];
  let pair: readonly [readonly [number, number, number], readonly [number, number, number]] | undefined;
  for (let left = 0; left < entry.tetrahedra.length && !pair; left += 1) for (let right = left + 1; right < entry.tetrahedra.length; right += 1) {
    if (entry.tetrahedra[left].filter((selector) => entry.tetrahedra[right].includes(selector)).length === 2) {
      pair = [entry.tetrahedra[left], entry.tetrahedra[right]]; break;
    }
  }
  assert.ok(pair);
  const shared = pair[0].filter((selector) => pair![1].includes(selector));
  const opposite = pair.map((tetrahedron) => tetrahedron.find((selector) => !shared.includes(selector))!);
  const base = [0, 1, 2].map((axis) => shared.reduce((sum, selector) =>
    sum + catalog.tetrahedronVertexData[selector * 4 + axis] / 3, 0));
  const epsilon = 1e-5;
  const points = opposite.map((selector) => base.map((value, axis) =>
    (1 - epsilon) * value + epsilon * catalog.tetrahedronVertexData[selector * 4 + axis]) as [number, number, number]);
  const field = (position: readonly number[]) => [0.25 + position[0], -0.5 + 2 * position[1], 1.5 - position[2]] as const;
  const neighbors = Array.from({ length: catalog.tetrahedronVertexData.length / 4 }, (_, selector) =>
    field([...catalog.tetrahedronVertexData.slice(selector * 4, selector * 4 + 3)]));
  const samples = points.map((point) => sampleOctreePowerCatalogVelocity(entry, catalog.tetrahedronVertexData, point, field([0, 0, 0]), neighbors));
  samples.forEach((sample, side) => {
    assert.equal(sample.mode, "tetrahedron");
    sample.velocity.forEach((value, axis) => close(value, field(points[side])[axis], 1e-10));
  });
  samples[0].velocity.forEach((value, axis) => assert.ok(Math.abs(value - samples[1].velocity[axis]) < 1e-4));
});

test("Dawn reconstructs uniform and general power velocities with deterministic guarded fallback", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-velocity checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  for (const code of [octreePowerVelocityPrepareShader, octreePowerVelocityPrepareFromFaceControlShader,
    octreePowerVelocityShader, octreePowerVelocityPublishShader, octreePowerVelocitySampleShader]) {
    const compilation = await device.createShaderModule({ code }).getCompilationInfo();
    assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  }

  const normals: number[][] = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ];
  const uniformNormalVelocities = [3, -1, 5, -3, -2, 4];
  const rootThird = 1 / Math.sqrt(3);
  const generalNormals = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    .map((normal) => normal.map((value) => value * rootThird));
  normals.push(...generalNormals, [1, 0, 0], [-1, 0, 0]);
  const generalVelocity = [-0.4, 0.8, 1.2];
  const normalVelocities = [...uniformNormalVelocities,
    ...generalNormals.map((normal) => normal.reduce((sum, value, axis) => sum + value * generalVelocity[axis], 0)),
    2, -2];
  const areas = [1, 1, 1, 1, 1, 1, 0.5, 1.25, 2, 3, 1, 1];
  const rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];

  const faceData = new ArrayBuffer(rows.length * 32);
  const faceWords = new Uint32Array(faceData); const faceFloats = new Float32Array(faceData);
  rows.forEach((row, face) => {
    const offset = face * 8;
    faceWords[offset] = row; faceWords[offset + 1] = INVALID_ROW;
    faceFloats[offset + 4] = normalVelocities[face]; faceFloats[offset + 5] = areas[face];
    faceFloats[offset + 6] = 1; faceFloats[offset + 7] = 1;
  });
  const normalData = new Float32Array(rows.length * 4);
  normals.forEach((normal, face) => normalData.set([...normal, 0], face * 4));
  const incidenceRowData = new Uint32Array(4 * 4);
  [0, 6, 10, 12].forEach((offset, row) => { incidenceRowData[row * 4 + 3] = offset; });
  const incidenceData = new Int32Array(rows.length * 2);
  rows.forEach((_row, face) => { incidenceData[face * 2] = face; incidenceData[face * 2 + 1] = 1; });

  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength)
      .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const faces = upload(new Uint8Array(faceData));
  const faceNormals = upload(normalData); const incidenceRows = upload(incidenceRowData); const incidences = upload(incidenceData);
  const velocity = new WebGPUOctreePowerVelocity(device, 3);
  const input = { faces, faceNormals, incidenceRows, incidences };
  const options = { rowCount: 3, faceCount: rows.length, incidenceCount: rows.length,
    maximumIncidencePerRow: 6, generation: 19 };

  const run = async (runOptions = options) => {
    const size = OCTREE_POWER_VELOCITY_CONTROL_BYTES + velocity.plan.velocityBytes;
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); velocity.encode(encoder, input, runOptions);
    encoder.copyBufferToBuffer(velocity.control, 0, readback, 0, OCTREE_POWER_VELOCITY_CONTROL_BYTES);
    encoder.copyBufferToBuffer(velocity.velocities, 0, readback, OCTREE_POWER_VELOCITY_CONTROL_BYTES, velocity.plan.velocityBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0);
    readback.unmap(); readback.destroy(); return result;
  };

  const first = await run();
  const firstControl = unpackOctreePowerVelocityControl(new Uint32Array(first, 0, 8));
  assert.deepEqual(firstControl, {
    flags: OCTREE_POWER_VELOCITY_VALID, firstError: INVALID_ROW, rowCount: 3, faceCount: 12,
    incidenceCount: 12, reconstructedCount: 3, fallbackCount: 1, generation: 19,
  });
  const values = new Float32Array(first, OCTREE_POWER_VELOCITY_CONTROL_BYTES, 12);
  // With outward-oriented negative-side values, this is exactly the ordinary
  // average of the two globally oriented Cartesian face values per component.
  [2, 4, -3].forEach((expected, axis) => close(values[axis], expected));
  assert.equal(values[3], 1);
  generalVelocity.forEach((expected, axis) => close(values[4 + axis], expected));
  assert.equal(values[7], 1);
  assert.deepEqual([...values.slice(8, 12)], [0, 0, 0, 0], "rank-deficient fit must take the counted zero fallback");

  const second = await run();
  assert.deepEqual(new Uint8Array(second), new Uint8Array(first), "repeated reconstruction must be byte deterministic");

  // Production obtains compact counts from the already GPU-authored WP4
  // control record.  No CPU readback may sit between face publication and
  // Stage A reconstruction.
  const faceControl = device.createBuffer({ size: 48,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const faceControlWords = new Uint32Array(12);
  faceControlWords.set([3, rows.length, rows.length]); faceControlWords[7] = 19;
  faceControlWords[8] = OCTREE_POWER_VELOCITY_VALID;
  device.queue.writeBuffer(faceControl, 0, faceControlWords);
  const gpuCountReadback = device.createBuffer({ size: first.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const gpuCountEncoder = device.createCommandEncoder();
  velocity.encodeFromFaceControl(gpuCountEncoder, input, faceControl, {
    maximumIncidencePerRow: 6, generation: 19,
  });
  gpuCountEncoder.copyBufferToBuffer(velocity.control, 0, gpuCountReadback, 0,
    OCTREE_POWER_VELOCITY_CONTROL_BYTES);
  gpuCountEncoder.copyBufferToBuffer(velocity.velocities, 0, gpuCountReadback,
    OCTREE_POWER_VELOCITY_CONTROL_BYTES, velocity.plan.velocityBytes);
  device.queue.submit([gpuCountEncoder.finish()]);
  await gpuCountReadback.mapAsync(GPUMapMode.READ);
  const gpuCountResult = gpuCountReadback.getMappedRange().slice(0);
  gpuCountReadback.unmap(); gpuCountReadback.destroy(); faceControl.destroy();
  assert.deepEqual(new Uint8Array(gpuCountResult), new Uint8Array(first),
    "GPU-derived compact counts must reproduce explicit-count Stage A output");

  // A non-finite resolved normal invalidates the complete generation. Partial
  // row output is deliberately not authoritative when VALID is absent.
  device.queue.writeBuffer(faceNormals, 6 * 16, new Float32Array([Number.NaN]));
  let failed = await run();
  let failedControl = unpackOctreePowerVelocityControl(new Uint32Array(failed, 0, 8));
  assert.equal(failedControl.flags & OCTREE_POWER_VELOCITY_VALID, 0);
  assert.equal(failedControl.flags & OCTREE_POWER_VELOCITY_ERROR.invalidNormal, OCTREE_POWER_VELOCITY_ERROR.invalidNormal);
  assert.equal(failedControl.firstError, 1); assert.equal(failedControl.reconstructedCount, 0);

  // Restore the sample, then ask for one row beyond compact capacity. The GPU
  // reports capacity and suppresses publication without an out-of-bounds read.
  device.queue.writeBuffer(faceNormals, 6 * 16, new Float32Array([generalNormals[0][0]]));
  failed = await run({ ...options, rowCount: 4 });
  failedControl = unpackOctreePowerVelocityControl(new Uint32Array(failed, 0, 8));
  assert.equal(failedControl.flags & OCTREE_POWER_VELOCITY_VALID, 0);
  assert.equal(failedControl.flags & OCTREE_POWER_VELOCITY_ERROR.capacity, OCTREE_POWER_VELOCITY_ERROR.capacity);
  assert.equal(failedControl.reconstructedCount, 0);
  assert.deepEqual(validationErrors, []);

  const catalogBytes = readFileSync(join(process.cwd(), "lib/generated/octree-power-catalog.bin"));
  const catalog = decodeGeneratedOctreePowerCatalog(catalogBytes.buffer.slice(catalogBytes.byteOffset, catalogBytes.byteOffset + catalogBytes.byteLength));
  const topology = new WebGPUOctreePowerTopology(device, 1, catalog);
  const sampler = new WebGPUOctreePowerVelocitySampler(device, 3, topology.source);
  const uniformPacked = catalog.sameOrFinerDirect[0x3ffff], transitionPacked = catalog.sameOrFinerDirect[0];
  const uniformEntry = uniformPacked & 0xffff, transitionEntry = transitionPacked & 0xffff;
  const transitionSelectorCount = catalog.tetrahedronVertexData.length / 4;
  const tetrahedronFirst = catalog.tetrahedronHeaders[transitionEntry * 3];
  const packedTetrahedron = catalog.tetrahedronData[tetrahedronFirst];
  const selectors = [packedTetrahedron & 255, (packedTetrahedron >> 8) & 255, (packedTetrahedron >> 16) & 255];
  const transitionPoint = selectors.reduce((sum, selector) => [
    sum[0] + catalog.tetrahedronVertexData[selector * 4] / 4,
    sum[1] + catalog.tetrahedronVertexData[selector * 4 + 1] / 4,
    sum[2] + catalog.tetrahedronVertexData[selector * 4 + 2] / 4,
  ], [0, 0, 0]);
  const queryData = new ArrayBuffer(3 * 48); const queryWords = new Uint32Array(queryData); const queryFloats = new Float32Array(queryData);
  const setQuery = (index: number, entry: number, start: number, count: number, point: readonly number[]) => {
    const word = index * 12;
    const uniform = catalog.tetrahedronHeaders[entry * 3 + 2] !== 0;
    queryWords.set([0, uniform ? 0 : transitionSelectorCount,
      catalog.tetrahedronHeaders[entry * 3], catalog.tetrahedronHeaders[entry * 3 + 1],
      catalog.tetrahedronHeaders[entry * 3 + 2], start, count, index], word);
    queryFloats.set([...point, 0], word + 8);
  };
  setQuery(0, uniformEntry, 0, 8, [0.25, 0.5, 0.75]);
  setQuery(1, transitionEntry, 8, transitionSelectorCount + 1, transitionPoint);
  setQuery(2, transitionEntry, 8, transitionSelectorCount + 1, [100, 100, 100]);
  const vertexValues = new Float32Array((8 + transitionSelectorCount + 1) * 4);
  for (let corner = 0; corner < 8; corner += 1) vertexValues.set([corner & 1, (corner >> 1) & 1, (corner >> 2) & 1, 1], corner * 4);
  const linear = (position: ArrayLike<number>) => [1 + 2 * position[0], -1 + 3 * position[1], 0.5 - 2 * position[2], 1];
  vertexValues.set(linear([0, 0, 0]), 8 * 4);
  for (let slot = 0; slot < transitionSelectorCount; slot += 1) {
    vertexValues.set(linear(catalog.tetrahedronVertexData.slice(slot * 4, slot * 4 + 3)), (9 + slot) * 4);
  }
  const queries = upload(new Uint8Array(queryData)), sampleVertices = upload(vertexValues);
  const sampleReadback = device.createBuffer({ size: OCTREE_POWER_SAMPLE_CONTROL_BYTES + sampler.plan.resultBytes + sampler.plan.statusBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const sampleEncoder = device.createCommandEncoder(); sampler.encode(sampleEncoder, queries, sampleVertices, 3, 23);
  sampleEncoder.copyBufferToBuffer(sampler.control, 0, sampleReadback, 0, OCTREE_POWER_SAMPLE_CONTROL_BYTES);
  sampleEncoder.copyBufferToBuffer(sampler.results, 0, sampleReadback, OCTREE_POWER_SAMPLE_CONTROL_BYTES, sampler.plan.resultBytes);
  sampleEncoder.copyBufferToBuffer(sampler.statuses, 0, sampleReadback, OCTREE_POWER_SAMPLE_CONTROL_BYTES + sampler.plan.resultBytes, sampler.plan.statusBytes);
  device.queue.submit([sampleEncoder.finish()]); await sampleReadback.mapAsync(GPUMapMode.READ);
  const sampledBytes = sampleReadback.getMappedRange().slice(0); sampleReadback.unmap();
  const sampleControl = unpackOctreePowerVelocitySampleControl(new Uint32Array(sampledBytes, 0, 8));
  assert.deepEqual(sampleControl, { flags: OCTREE_POWER_SAMPLE_VALID, firstError: INVALID_ROW, queryCount: 3,
    interpolatedCount: 3, uniformCount: 1, tetrahedronCount: 1, nearestFallbackCount: 1, generation: 23 });
  const sampledVelocities = new Float32Array(sampledBytes, OCTREE_POWER_SAMPLE_CONTROL_BYTES, 12);
  [0.25, 0.5, 0.75].forEach((expected, axis) => close(sampledVelocities[axis], expected));
  linear(transitionPoint).slice(0, 3).forEach((expected, axis) => close(sampledVelocities[4 + axis], expected));
  assert.deepEqual(validationErrors, []);

  sampleReadback.destroy(); queries.destroy(); sampleVertices.destroy(); sampler.destroy(); topology.destroy();
  velocity.destroy(); faces.destroy(); faceNormals.destroy(); incidenceRows.destroy(); incidences.destroy(); device.destroy();
});
