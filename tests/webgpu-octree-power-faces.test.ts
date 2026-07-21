import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { OCTREE_POWER_CATALOG_FACE_FLOATS } from "../lib/octree-power-catalog";
import { OCTREE_POWER_FACE_RECORD_BYTES } from "../lib/octree-power-operator";
import {
  OCTREE_POWER_FACE_BOUNDARY,
  OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP,
  OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP,
  OCTREE_POWER_FACE_ERROR,
  OCTREE_POWER_FACE_OPEN_BOUNDARY,
  OCTREE_POWER_FACE_QUADRATURE_BYTES,
  OCTREE_POWER_FACE_VALID,
  OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT,
  WebGPUOctreePowerFaces,
  octreePowerClosedBoundaryMask,
  octreePowerBoundaryPhiShader,
  octreePowerFaceShader,
  planOctreePowerFaces,
  unpackOctreePowerFaceControl,
} from "../lib/webgpu-octree-power-faces";
import { OCTREE_POWER_TOPOLOGY_VALID, type OctreePowerTopologySource } from "../lib/webgpu-octree-power-topology";

// The native Metal binding can release a shared Dawn instance while a later
// test in this file is still creating pipelines. Retain wrappers/devices until
// process exit; submitted work and mapped readbacks remain the correctness gate.
const retainedNativeGPUs: GPU[] = [];
const retainedDevices: GPUDevice[] = [];
const octreeProjectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

function catalogViews() {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

test("power-face planner allocates compact live face and incidence stores", () => {
  const plan = planOctreePowerFaces(100, 640, 1_100);
  assert.equal(plan.faceBytes, 640 * OCTREE_POWER_FACE_RECORD_BYTES);
  assert.equal(plan.normalBytes, 640 * 16);
  assert.equal(plan.centroidBytes, 640 * 16);
  assert.equal(plan.quadratureBytes, 640 * OCTREE_POWER_FACE_QUADRATURE_BYTES);
  assert.equal(plan.incidenceBytes, 1_100 * 8);
  assert.equal(plan.workspaceBytes, 101 * 16);
  assert.equal(plan.hashCapacity, 256);
  assert.equal(plan.scanBlockCount, 1);
  assert.equal(plan.boundaryQueryBytes, 640 * 32);
  assert.equal(plan.allocatedBytes, plan.faceBytes + plan.normalBytes + plan.centroidBytes + plan.quadratureBytes + plan.incidenceBytes + plan.workspaceBytes
    + plan.hashBytes + plan.scanBytes + plan.boundaryQueryBytes + 64 + 96);
  assert.throws(() => planOctreePowerFaces(1, 2, 5), /two incidences/);
});

test("power-face boundary policy preserves geometric world identity and scene-open ceiling authority", () => {
  assert.equal(OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP, 47);
  assert.equal(OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP, 63);
  assert.equal(octreePowerClosedBoundaryMask(false), 47, "the dam ceiling (+y) is open");
  assert.equal(octreePowerClosedBoundaryMask(true), 63, "a closed container closes all six world faces");
  assert.match(octreeProjectionSource,
    /closedBoundaryMask:\s*octreePowerClosedBoundaryMask\(this\.scene\.container\.top === "closed"\)/,
    "power-face construction must receive the authored container policy used by the Section 5 face band");

  const compactShader = octreePowerFaceShader.replace(/\s+/g, "");
  assert.match(compactShader, /structParams\{dimensionsRowCount:vec4u,capacitiesGeneration:vec4u,physical:vec4f,boundaryPolicy:vec4u,containerFill:vec4f,phiPolicy:vec4u\}/);
  assert.match(compactShader,
    /letgeometricWorld=world&declared;if\(geometricWorld!=0u\)\{letopen=select\(OPEN_BOUNDARY,0u,\(params\.boundaryPolicy\.x&geometricWorld\)!=0u\);returnBOUNDARY\|open\|\(geometricWorld<<WORLD_BOUNDARY_SHIFT\)/,
    "catalog boundary bits identify world geometry while the scene mask alone selects open versus closed");
  assert.match(compactShader, /returnBOUNDARY\|OPEN_BOUNDARY;/,
    "a missing non-world phase neighbor remains an internal free surface");
  assert.match(compactShader,
    /letworld=\(face\.flags>>WORLD_BOUNDARY_SHIFT\)&63u;if\(world!=0u\).*letboundaryDistance=dot\(geometry\.centroid-rowCenter\(face\.negativeRow\),geometry\.normal\).*face\.inverseDistance=1\.0\/boundaryDistance/s,
    "an open world plane uses exact row-centre-to-boundary distance instead of free-surface theta");
});

test("power free-surface coefficients use actual signed cell-centre phi without repair or floor", () => {
  const geometry = octreePowerFaceShader.replace(/\s+/g, "");
  const sampler = octreePowerBoundaryPhiShader.replace(/\s+/g, "");
  assert.match(geometry,
    /BoundaryPhiQuery\(vec4f\(rowCenter\(face\.negativeRow\),1\.0\),vec4f\(geometry\.neighborCenter,1\.0\)\)/,
    "geometry must publish the actual liquid and absent-air power-cell centres");
  assert.match(sampler, /letliquid=sampleAuthority\(query\.liquidCenter\.xyz\);letair=sampleAuthority\(query\.airCenter\.xyz\)/,
    "both pressure samples must come from the selected signed-distance authority");
  assert.match(sampler, /lettheta=\(-liquid\.x\)\/\(air\.x-liquid\.x\)/,
    "the Ghost Fluid fraction must use the two signed cell-centre values");
  assert.doesNotMatch(sampler, /abs\(liquid\.x\)|abs\(air\.x\)|clamp\([^)]*theta|0\.05/,
    "authoritative power pressure must not repair signs or floor theta");
  assert.match(sampler, /if\(liquid\.y==0\.0\|\|air\.y==0\.0\)\{failBoundary/,
    "missing narrow-band support must reject publication rather than estimate phi");
  assert.match(sampler, /worklist\[1\]!=fineParams\.generation\|\|worklist\[3\]!=1u\|\|worklist\[4\]!=1u/,
    "fine pressure sampling must require the current all-or-nothing GPU publication header");
  assert.match(octreeProjectionSource,
    /const boundaryFine = this\.globalFineBootstrapped\s*\? \(this\.globalFineCurrentIsA \? this\.globalFineSourceA : this\.globalFineSourceB\)\s*:\s*this\.globalFineSourceA/,
    "pressure assembly must bind the current fine generation, with source A used only as the analytic bootstrap binding");
  assert.match(octreeProjectionSource,
    /mode: this\.globalFineBootstrapped \? "fine" as const : "analytic" as const/,
    "only the cold operator may use the authored analytic signed distance");
  const bindings = [...octreePowerBoundaryPhiShader.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
  assert.equal(new Set(bindings).size, 10, "boundary sampling stays at the practical ten-binding ceiling");
});

test("Dawn site hash and block scan scale across blocks with world-boundary identity", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-face scale checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  retainedNativeGPUs.push(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  retainedDevices.push(device);
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  const catalog = catalogViews();
  const packedUniform = catalog.sameOrFinerDirect[0x3_ffff];
  assert.notEqual(packedUniform, 0xffff_ffff);
  const entry = packedUniform & 0xffff;
  const transform = packedUniform >>> 16;
  const header = catalog.entryHeaders.slice(entry * 2, entry * 2 + 2);
  assert.equal(header[1], 6);
  const faceData = catalog.faceData.slice(header[0] * OCTREE_POWER_CATALOG_FACE_FLOATS, (header[0] + header[1]) * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const dimensions: [number, number, number] = [10, 10, 3];
  const rowCount = dimensions[0] * dimensions[1] * dimensions[2];
  const headerWords = new Uint32Array(rowCount * 12);
  const metricWords = new Uint32Array(rowCount * 4);
  const metricFloats = new Float32Array(metricWords.buffer);
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    const row = x + dimensions[0] * (y + dimensions[1] * z);
    headerWords[row * 12] = row; headerWords[row * 12 + 3] = 1;
    let boundary = 0;
    if (x === 0) boundary |= 1; if (y === 0) boundary |= 2; if (z === 0) boundary |= 4;
    if (z === dimensions[2] - 1) boundary |= 8; if (y === dimensions[1] - 1) boundary |= 16;
    if (x === dimensions[0] - 1) boundary |= 32;
    metricWords[row * 4] = 0;
    metricWords[row * 4 + 1] = (OCTREE_POWER_TOPOLOGY_VALID | transform | (boundary << 8)) >>> 0;
    metricFloats[row * 4 + 2] = catalog.entryVolumes[entry];
  }
  const headers = upload(headerWords);
  const metrics = upload(metricWords);
  const entryHeaders = upload(new Uint32Array([0, 6]));
  const facesCatalog = upload(faceData);
  const placeholder = upload(new Uint32Array([0]));
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: rowCount, entryCount: 1, lookupCount: 1, metricBytes: metricWords.byteLength,
      catalogBytes: faceData.byteLength + 8, allocatedBytes: 0 },
    metrics, control: placeholder, catalogEntryHeaders: entryHeaders, catalogVolumes: placeholder,
    catalogFaces: facesCatalog, catalogLookup: placeholder, sameOrFinerDirect: placeholder, sameOrCoarserDirect: placeholder,
  };
  const expectedInterior = 9 * 10 * 3 + 10 * 9 * 3 + 10 * 10 * 2;
  const expectedBoundary = 2 * (10 * 3 + 10 * 3 + 10 * 10);
  const expectedFaces = expectedInterior + expectedBoundary;
  const builder = new WebGPUOctreePowerFaces(device, rowCount, expectedFaces, topology, rowCount * 6);
  assert.equal(builder.plan.scanBlockCount, 2);
  const rowCountBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Uint32Array(rowCountBuffer.getMappedRange())[0] = rowCount; rowCountBuffer.unmap();

  const run = async () => {
    const outputBytes = builder.plan.workspaceBytes + builder.plan.faceBytes + builder.plan.incidenceBytes;
    const readback = device.createBuffer({ size: 64 + outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    builder.encode(encoder, headers, { dimensions, rowCount: rowCountBuffer, generation: 19 });
    let offset = 0;
    encoder.copyBufferToBuffer(builder.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(builder.incidenceOffsets, 0, readback, offset, builder.plan.workspaceBytes); offset += builder.plan.workspaceBytes;
    encoder.copyBufferToBuffer(builder.faces, 0, readback, offset, builder.plan.faceBytes); offset += builder.plan.faceBytes;
    encoder.copyBufferToBuffer(builder.incidence, 0, readback, offset, builder.plan.incidenceBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const result = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy(); return result;
  };
  const first = await run();
  const control = unpackOctreePowerFaceControl(new Uint32Array(first, 0, 16));
  assert.equal(control.flags, 0); assert.equal(control.valid, OCTREE_POWER_FACE_VALID);
  assert.equal(control.rowCount, rowCount); assert.equal(control.faceCount, expectedFaces);
  assert.equal(control.incidenceCount, rowCount * 6); assert.equal(control.boundaryCount, expectedBoundary);
  assert.equal(control.lookupMissCount, expectedBoundary); assert.equal(control.worldBoundaryCount, expectedBoundary);
  assert.ok(control.maximumObservedProbe > 1 && control.maximumObservedProbe <= builder.plan.maximumHashProbes);
  const workspace = new Uint32Array(first, 64, builder.plan.workspaceBytes / 4);
  assert.ok(workspace[255 * 4 + 2] < workspace[256 * 4 + 2]);
  assert.equal(workspace[rowCount * 4 + 2], expectedFaces);
  assert.equal(workspace[rowCount * 4 + 3], rowCount * 6);
  device.queue.writeBuffer(rowCountBuffer, 0, new Uint32Array([0]));
  const empty = unpackOctreePowerFaceControl(new Uint32Array(await run(), 0, 16));
  assert.equal(empty.valid, 0); assert.equal(empty.faceCount, 0); assert.equal(empty.incidenceCount, 0);
  assert.equal(empty.flags & OCTREE_POWER_FACE_ERROR.invalidHeader, OCTREE_POWER_FACE_ERROR.invalidHeader,
    "an empty generation must fail closed instead of publishing valid=true");
  device.queue.writeBuffer(rowCountBuffer, 0, new Uint32Array([rowCount]));
  const faceWordOffset = 64 + builder.plan.workspaceBytes;
  const faceWords = new Uint32Array(first, faceWordOffset, builder.plan.faceBytes / 4);
  let observedBoundary = 0;
  for (let face = 0; face < expectedFaces; face += 1) {
    const word = face * 8;
    if (faceWords[word + 1] === 0xffff_ffff) {
      observedBoundary += 1;
      assert.equal(faceWords[word + 3] & OCTREE_POWER_FACE_BOUNDARY, OCTREE_POWER_FACE_BOUNDARY);
      assert.equal(faceWords[word + 3] & OCTREE_POWER_FACE_OPEN_BOUNDARY, 0);
      assert.notEqual((faceWords[word + 3] >>> OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT) & 63, 0);
    }
  }
  assert.equal(observedBoundary, expectedBoundary);
  const second = await run();
  assert.deepEqual(new Uint8Array(second, 64), new Uint8Array(first, 64), "multi-block public face/CSR output must be stable");

  const hashSite = (cell: number, size: number) => {
    let value = (cell ^ Math.imul(size, 0x9e3779b9)) >>> 0;
    value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb352d) >>> 0;
    value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846ca68b) >>> 0;
    return (value ^ (value >>> 16)) >>> 0;
  };
  const buckets = Array.from({ length: 128 }, () => [] as number[]);
  for (let cell = 0; cell < 8_192; cell += 1) buckets[hashSite(cell, 1) & 127].push(cell);
  const collidingCells = buckets.find((bucket) => bucket.length >= 33)!.slice(0, 33);
  assert.equal(collidingCells.length, 33);
  const collisionHeaders = new Uint32Array(33 * 12);
  collidingCells.forEach((cell, row) => { collisionHeaders[row * 12] = cell; collisionHeaders[row * 12 + 3] = 1; });
  const collisionHeaderBuffer = upload(collisionHeaders);
  const collisionBuilder = new WebGPUOctreePowerFaces(device, 33, 198, topology, 198);
  assert.equal(collisionBuilder.plan.hashCapacity, 128);
  const collisionReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const collisionEncoder = device.createCommandEncoder();
  collisionBuilder.encode(collisionEncoder, collisionHeaderBuffer, { dimensions: [8_192, 1, 1], rowCount: 33 });
  collisionEncoder.copyBufferToBuffer(collisionBuilder.control, 0, collisionReadback, 0, 64);
  device.queue.submit([collisionEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await collisionReadback.mapAsync(GPUMapMode.READ);
  const collisionControl = unpackOctreePowerFaceControl(new Uint32Array(collisionReadback.getMappedRange().slice(0)));
  collisionReadback.unmap();
  assert.equal(collisionControl.faceCount, 0); assert.equal(collisionControl.incidenceCount, 0); assert.equal(collisionControl.valid, 0);
  assert.notEqual(collisionControl.flags & OCTREE_POWER_FACE_ERROR.siteIndex, 0);
  assert.deepEqual(validationErrors, []);
  collisionBuilder.destroy(); collisionHeaderBuffer.destroy(); collisionReadback.destroy();
  builder.destroy(); rowCountBuffer.destroy(); headers.destroy(); metrics.destroy(); entryHeaders.destroy();
  facesCatalog.destroy(); placeholder.destroy();
});

test("power-face WGSL uses count/scan/emit and no atomic public append", () => {
  assert.match(octreePowerFaceShader, /fn countPowerFaces/);
  assert.match(octreePowerFaceShader, /fn scanPowerFaceRows/);
  assert.match(octreePowerFaceShader, /fn scanPowerFaceBlocks/);
  assert.match(octreePowerFaceShader, /fn buildPowerSiteIndex/);
  assert.match(octreePowerFaceShader, /fn emitPowerFaces/);
  assert.match(octreePowerFaceShader, /fn sortPowerIncidenceRows/);
  assert.match(octreePowerFaceShader, /incidence\[begin\+cursor-1u\]\.face>value\.face/,
    "the public CSR must be strictly face-ID ordered before operator assembly");
  assert.match(octreePowerFaceShader, /rows\[row\]\.faceOffset\+localFace/);
  assert.doesNotMatch(octreePowerFaceShader, /atomicAdd\(&control\.faceCount/);
  assert.match(octreePowerFaceShader, /PowerFaceRecord\(row,neighbor,geometryCode,flags,0\.0,geometry\.area,geometry\.inverseDistance,1\.0\)/);
  assert.match(octreePowerFaceShader, /fn buildPowerBoundaryPhiQueries/);
  assert.doesNotMatch(octreePowerFaceShader, /airPhi=abs|liquidPhi\+dot|0\.05/,
    "face geometry must not synthesize or floor a free-surface coefficient");
});

test("Dawn emits stable unique power faces and reciprocal compact incidence", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-face checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  retainedNativeGPUs.push(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  retainedDevices.push(device);
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => {
    validationErrors.push((event as { error: { message: string } }).error.message);
  });
  const compilation = await device.createShaderModule({ code: octreePowerFaceShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);

  const catalog = catalogViews();
  // 18 same-size face/edge neighbors is the uniform catalog descriptor.
  const lookupFor = (descriptor: number) => {
    if (descriptor < catalog.sameOrFinerDirect.length) {
      const packed = catalog.sameOrFinerDirect[descriptor];
      if (packed !== 0xffff_ffff) return { entry: packed & 0xffff, transform: packed >>> 16 };
    }
    for (let offset = 0; offset < catalog.lookup.length; offset += 3) {
      if (catalog.lookup[offset] === descriptor) return { entry: catalog.lookup[offset + 1], transform: catalog.lookup[offset + 2] };
    }
    throw new Error(`Missing catalog descriptor ${descriptor}`);
  };
  const uniform = lookupFor(0x3_ffff);
  const uniformHeader = catalog.entryHeaders.slice(uniform.entry * 2, uniform.entry * 2 + 2);
  const uniformFaces = catalog.faceData.slice(uniformHeader[0] * OCTREE_POWER_CATALOG_FACE_FLOATS, (uniformHeader[0] + uniformHeader[1]) * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const compactFaces = new Float32Array(uniformFaces);
  const compactHeaders = new Uint32Array([0, uniformHeader[1]]);
  const metricWords = new Uint32Array(8);
  const metricFloats = new Float32Array(metricWords.buffer);
  [31, 62].forEach((boundary, row) => {
    metricWords[row * 4] = 0;
    metricWords[row * 4 + 1] = (OCTREE_POWER_TOPOLOGY_VALID | uniform.transform | (boundary << 8)) >>> 0;
    metricFloats[row * 4 + 2] = catalog.entryVolumes[uniform.entry];
  });
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const metrics = upload(metricWords);
  const entryHeaders = upload(compactHeaders);
  const catalogFaces = upload(compactFaces);
  const placeholder = upload(new Uint32Array([0]));
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: 2, entryCount: 1, lookupCount: 1, metricBytes: 32, catalogBytes: compactHeaders.byteLength + compactFaces.byteLength, allocatedBytes: 0 },
    metrics, control: placeholder, catalogEntryHeaders: entryHeaders, catalogVolumes: placeholder,
    catalogFaces, catalogLookup: placeholder, sameOrFinerDirect: placeholder, sameOrCoarserDirect: placeholder,
  };
  const builder = new WebGPUOctreePowerFaces(device, 2, 11, topology, 12);
  const dimensions: [number, number, number] = [2, 1, 1];
  const headerWords = new Uint32Array(2 * 12);
  [0, 1].forEach((cell, row) => { headerWords[row * 12] = cell; headerWords[row * 12 + 3] = 1; });
  const headers = upload(headerWords);

  const readRun = async () => {
    const byteCount = 64 + builder.plan.workspaceBytes + builder.plan.faceBytes
      + builder.plan.incidenceBytes + builder.plan.normalBytes + builder.plan.centroidBytes + builder.plan.quadratureBytes;
    const readback = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    builder.encode(encoder, headers, { dimensions, rowCount: 2, physicalCellSize: 0.25, generation: 7 });
    let offset = 0;
    encoder.copyBufferToBuffer(builder.control, 0, readback, offset, 64); offset += 64;
    encoder.copyBufferToBuffer(builder.source.incidenceOffsets, 0, readback, offset, builder.plan.workspaceBytes); offset += builder.plan.workspaceBytes;
    encoder.copyBufferToBuffer(builder.faces, 0, readback, offset, builder.plan.faceBytes); offset += builder.plan.faceBytes;
    encoder.copyBufferToBuffer(builder.incidence, 0, readback, offset, builder.plan.incidenceBytes); offset += builder.plan.incidenceBytes;
    encoder.copyBufferToBuffer(builder.faceNormals, 0, readback, offset, builder.plan.normalBytes); offset += builder.plan.normalBytes;
    encoder.copyBufferToBuffer(builder.faceCentroids, 0, readback, offset, builder.plan.centroidBytes); offset += builder.plan.centroidBytes;
    encoder.copyBufferToBuffer(builder.faceQuadrature, 0, readback, offset, builder.plan.quadratureBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
    return bytes;
  };

  const first = await readRun();
  const controlWords = new Uint32Array(first, 0, 16);
  const control = unpackOctreePowerFaceControl(controlWords);
  assert.deepEqual(control, {
    rowCount: 2, faceCount: 11, incidenceCount: 12, flags: 0,
    firstInvalid: 0xffff_ffff, invalidCount: 0, boundaryCount: 10, generation: 7, valid: OCTREE_POWER_FACE_VALID,
    lookupMissCount: 10, maximumObservedProbe: control.maximumObservedProbe, worldBoundaryCount: 10,
    firstInvalidSlot: 0xffff_ffff, firstInvalidNeighbor: 0xffff_ffff, firstInvalidDetail: 0,
    firstInvalidRow: 0xffff_ffff,
  });
  assert.ok(control.maximumObservedProbe >= 1 && control.maximumObservedProbe <= builder.plan.maximumHashProbes);
  assert.equal(controlWords[8], OCTREE_POWER_FACE_VALID);
  const workspaceOffset = 64;
  const workspace = new Uint32Array(first, workspaceOffset, builder.plan.workspaceBytes / 4);
  assert.deepEqual([workspace[3], workspace[7], workspace[11]], [0, 6, 12]);
  assert.deepEqual([workspace[0], workspace[4]], [6, 5]);
  assert.deepEqual([workspace[1], workspace[5]], [6, 6]);

  const faceOffset = workspaceOffset + builder.plan.workspaceBytes;
  const faceWords = new Uint32Array(first, faceOffset, builder.plan.faceBytes / 4);
  const faceFloats = new Float32Array(first, faceOffset, builder.plan.faceBytes / 4);
  const keys = new Set<string>();
  let interiorFace = -1;
  for (let face = 0; face < control.faceCount; face += 1) {
    const word = face * 8;
    const negative = faceWords[word];
    const positive = faceWords[word + 1];
    const geometryCode = faceWords[word + 2];
    const flags = faceWords[word + 3];
    assert.ok(negative < 2);
    assert.equal((flags & OCTREE_POWER_FACE_VALID) >>> 0, OCTREE_POWER_FACE_VALID);
    assert.ok(Number.isFinite(faceFloats[word + 5]) && faceFloats[word + 5] > 0);
    assert.ok(Number.isFinite(faceFloats[word + 6]) && faceFloats[word + 6] > 0);
    assert.equal(faceFloats[word + 7], 1);
    const key = `${negative}:${positive}:${geometryCode}`;
    assert.equal(keys.has(key), false); keys.add(key);
    if (positive !== 0xffff_ffff) {
      assert.equal(interiorFace, -1);
      assert.deepEqual([negative, positive], [0, 1]);
      assert.equal(flags & OCTREE_POWER_FACE_BOUNDARY, 0);
      interiorFace = face;
    } else {
      assert.equal(flags & OCTREE_POWER_FACE_BOUNDARY, OCTREE_POWER_FACE_BOUNDARY);
    }
  }
  assert.notEqual(interiorFace, -1);

  const incidenceOffset = faceOffset + builder.plan.faceBytes;
  const incidenceWords = new Uint32Array(first, incidenceOffset, builder.plan.incidenceBytes / 4);
  const normalOffset = incidenceOffset + builder.plan.incidenceBytes;
  const normalFloats = new Float32Array(first, normalOffset, builder.plan.normalBytes / 4);
  const centroidOffset = normalOffset + builder.plan.normalBytes;
  const centroidFloats = new Float32Array(first, centroidOffset, builder.plan.centroidBytes / 4);
  const quadratureOffset = centroidOffset + builder.plan.centroidBytes;
  const quadratureWords = new Uint32Array(first, quadratureOffset, builder.plan.quadratureBytes / 4);
  const quadratureFloats = new Float32Array(first, quadratureOffset, builder.plan.quadratureBytes / 4);
  for (let face = 0; face < control.faceCount; face += 1) {
    const at = face * 4; const length = Math.hypot(normalFloats[at], normalFloats[at + 1], normalFloats[at + 2]);
    assert.ok(Math.abs(length - 1) < 2e-4); assert.equal(normalFloats[at + 3], 0);
    assert.ok(Number.isFinite(centroidFloats[at]) && Number.isFinite(centroidFloats[at + 1])
      && Number.isFinite(centroidFloats[at + 2]));
    assert.equal(centroidFloats[at + 3], 0);
    const quadratureAt = face * (OCTREE_POWER_FACE_QUADRATURE_BYTES / 4);
    assert.deepEqual(Array.from(quadratureFloats.slice(quadratureAt, quadratureAt + 3)),
      Array.from(centroidFloats.slice(at, at + 3)), "polygon quadrature must retain the exact public face centroid");
    assert.ok(Math.abs(quadratureFloats[quadratureAt + 3] - faceFloats[face * 8 + 5])
      <= Math.max(2e-5, faceFloats[face * 8 + 5] * 5e-4));
    assert.ok(new Set(Array.from(quadratureWords.slice(quadratureAt + 4, quadratureAt + 20))).size > 1,
      "actual power-polygon strata must not collapse to the face centroid");
  }
  const incident = Array.from({ length: 2 }, () => [] as { face: number; sign: number }[]);
  for (let row = 0; row < 2; row += 1) {
    const begin = workspace[row * 4 + 3];
    const end = workspace[(row + 1) * 4 + 3];
    for (let index = begin; index < end; index += 1) {
      incident[row].push({ face: incidenceWords[index * 2], sign: new Int32Array(first, incidenceOffset + index * 8 + 4, 1)[0] });
    }
  }
  assert.deepEqual(incident[0].filter((item) => item.face === interiorFace), [{ face: interiorFace, sign: 1 }]);
  assert.deepEqual(incident[1].filter((item) => item.face === interiorFace), [{ face: interiorFace, sign: -1 }]);
  for (const row of incident) for (const item of row) assert.ok(item.face < control.faceCount);

  const second = await readRun();
  assert.deepEqual(new Uint8Array(second), new Uint8Array(first), "repeated GPU rebuild must be byte deterministic");

  const overflow = new WebGPUOctreePowerFaces(device, 2, 1, topology, 2);
  const overflowReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const overflowEncoder = device.createCommandEncoder();
  overflow.encode(overflowEncoder, headers, { dimensions, rowCount: 2 });
  overflowEncoder.copyBufferToBuffer(overflow.control, 0, overflowReadback, 0, 64);
  device.queue.submit([overflowEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await overflowReadback.mapAsync(GPUMapMode.READ);
  const overflowWords = new Uint32Array(overflowReadback.getMappedRange().slice(0)); overflowReadback.unmap();
  assert.equal(overflowWords[1], 0); assert.equal(overflowWords[2], 0);
  assert.equal(overflowWords[3] & OCTREE_POWER_FACE_ERROR.capacity, OCTREE_POWER_FACE_ERROR.capacity);
  assert.equal(overflowWords[8], 0);

  device.queue.writeBuffer(metrics, 5 * 4, new Uint32Array([0]));
  const invalidReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const invalidEncoder = device.createCommandEncoder();
  builder.encode(invalidEncoder, headers, { dimensions, rowCount: 2 });
  invalidEncoder.copyBufferToBuffer(builder.control, 0, invalidReadback, 0, 64);
  device.queue.submit([invalidEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await invalidReadback.mapAsync(GPUMapMode.READ);
  const invalidWords = new Uint32Array(invalidReadback.getMappedRange().slice(0)); invalidReadback.unmap();
  assert.equal(invalidWords[1], 0); assert.equal(invalidWords[2], 0);
  assert.equal(invalidWords[3] & OCTREE_POWER_FACE_ERROR.invalidMetric, OCTREE_POWER_FACE_ERROR.invalidMetric);
  assert.ok(invalidWords[5] >= 1); assert.equal(invalidWords[8], 0);
  assert.deepEqual(validationErrors, []);

  invalidReadback.destroy();
  overflow.destroy(); overflowReadback.destroy(); builder.destroy();
  metrics.destroy(); entryHeaders.destroy(); catalogFaces.destroy(); placeholder.destroy(); headers.destroy();
  for (const retained of retainedDevices) await retained.queue.onSubmittedWorkDone();
  for (const retained of [...retainedDevices].reverse()) retained.destroy();
  retainedDevices.length = 0;
});
