import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { WebGPUOctreeCoarseLevelSet } from "../lib/webgpu-octree-coarse-levelset";
import {
  OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  WebGPUOctreePowerCoarseLevelSet,
  makeOctreePowerCoarseLevelSetSampleWGSL,
  octreePowerCoarseLevelSetShader,
  planOctreePowerCoarseLevelSet,
  unpackOctreePowerCoarseLevelSetControl,
} from "../lib/webgpu-octree-power-coarse-levelset";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { OCTREE_POWER_TOPOLOGY_VALID, WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

test("fine modes retain the exact row-directory ABI while coarse-only opts into a dense complement", () => {
  const fine = planOctreePowerCoarseLevelSet(32);
  assert.equal(fine.sampleDirectoryBytes, OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
    + 32 * OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES);
  const coarseOnly = planOctreePowerCoarseLevelSet(32, 0, 0, 64);
  assert.equal(coarseOnly.sampleDirectoryBytes, OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
    + (32 + 64) * OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES);
});

test("coarse dense sampling is D4 invariant at dyadic face centres", () => {
  const sample = makeOctreePowerCoarseLevelSetSampleWGSL();
  assert.match(sample,
    /halfGrid=\.5\*round\(2\.\*rawGrid\)[\s\S]*select\(rawGrid,halfGrid,abs\(rawGrid-halfGrid\)<=vec3f\(1e-4\)\)/,
    "roundoff-sized half-grid errors must snap to identical mirrored coordinates");
  assert.match(sample,
    /var terms:array<f32,8>[\s\S]*terms\[corner\]=\(wx\*wz\)\*wy\*sample[\s\S]*terms\[j-1u\]<=term[\s\S]*value\+=terms\[i\]/,
    "trilinear corner contributions must fold independently of x/z corner permutation");
});

test("coarse publication snapshots physical power-cell volume with phi authority", () => {
  assert.match(octreePowerCoarseLevelSetShader,
    /struct SampleEntry \{[^}]*row:u32, physicalVolume:f32 \}/);
  assert.match(octreePowerCoarseLevelSetShader,
    /let metric=metrics\[row\][\s\S]*physicalVolume=metric\.volume\*extent\*extent\*extent/,
    "the directory must remain a complete volume snapshot after pressure rows rebuild");
  assert.match(octreePowerCoarseLevelSetShader,
    /rowStatus\[row\]\.physicalVolume=physicalVolume[\s\S]*SampleEntry\([^)]*rowStatus\[row\]\.physicalVolume\)/);
});

test("Dawn runs live-row advection and cell-center fine restriction without readback dependencies", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WP8 GPU checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const compilation = await device.createShaderModule({ code: octreePowerCoarseLevelSetShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const rowCount = 8, topology = new WebGPUOctreePowerTopology(device, rowCount, catalog);
  const coarse = new WebGPUOctreeCoarseLevelSet(device, rowCount);
  const uniformEntry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metricData = new ArrayBuffer(rowCount * 16), metricWords = new Uint32Array(metricData), metricFloats = new Float32Array(metricData);
  for (let row = 0; row < rowCount; row += 1) { metricWords[row * 4] = uniformEntry;
    metricWords[row * 4 + 1] = OCTREE_POWER_TOPOLOGY_VALID; metricFloats[row * 4 + 2] = 1; }
  device.queue.writeBuffer(topology.metrics, 0, metricData);
  const records = new Map(Array.from({ length: rowCount }, (_, row) => { const x = row & 1, phi = x - 0.5;
    return [row, { phi, minimumPhi: phi, maximumPhi: phi,
      flags: OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite, generation: 0, fineSampleCount: 0 }] as const; }));
  coarse.upload(records);
  const headerData = new ArrayBuffer(rowCount * 48), headerWords = new Uint32Array(headerData);
  for (let row = 0; row < rowCount; row += 1) { headerWords[row * 12] = row; headerWords[row * 12 + 3] = 1; }
  const velocities = new Float32Array(rowCount * 4); for (let row = 0; row < rowCount; row += 1) velocities[row * 4 + 3] = 1;
  const upload = (data: ArrayBufferView) => { const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer; };
  const headers = upload(new Uint8Array(headerData));
  const packedVelocities = new Float32Array(rowCount * 8); packedVelocities.set(velocities);
  const velocityBuffer = upload(packedVelocities);
  const rowGeometryWords = new Uint32Array(rowCount * 8);
  for (let bank = 0; bank < 2; bank += 1) for (let row = 0; row < rowCount; row += 1) {
    rowGeometryWords.set([row, 1, 0, row], (bank * rowCount + row) * 4);
  }
  const rowGeometry = upload(rowGeometryWords);
  const structuredControl = upload(new Uint32Array([0, 0xffff_ffff, rowCount, 41, 0, rowCount]));
  const liveRowDispatch = device.createBuffer({ size: 12,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  device.queue.writeBuffer(liveRowDispatch, 0, new Uint32Array([Math.ceil(rowCount / 64), 1, 1]));
  const structured = { control: structuredControl, rowVelocities: velocityBuffer,
    rowGeometry, rowBankStrideWords: rowCount, liveRowDispatch };
  const offsets = upload(new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]));
  const aggregateData = new ArrayBuffer(rowCount * 16), aggregateFloats = new Float32Array(aggregateData);
  const aggregateWords = new Uint32Array(aggregateData);
  aggregateFloats.set([-0.05, -0.05, 0.05]); aggregateWords[3] = 1;
  const contributions = upload(new Uint8Array(aggregateData));
  const schedule = new WebGPUOctreePowerCoarseLevelSet(device, coarse, topology.source);
  const directoryOffset = OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + coarse.plan.recordBytes;
  const readback = device.createBuffer({ size: directoryOffset + schedule.plan.sampleDirectoryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); schedule.encode(new PassBroker(encoder), { headers, structured,
    rowCount, fineCorrection: { rowOffsets: offsets, contributions, contributionCount: rowCount,
      maximumContributionsPerRow: 1, aggregated: true } },
  { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, generation: 41 });
  encoder.copyBufferToBuffer(schedule.control, 0, readback, 0, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  encoder.copyBufferToBuffer(coarse.records, 0, readback, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, coarse.plan.recordBytes);
  encoder.copyBufferToBuffer(schedule.sampleDirectory, 0, readback, directoryOffset, schedule.plan.sampleDirectoryBytes);
  device.queue.submit([encoder.finish()]); schedule.retireSubmittedEncoder(encoder);
  await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0); readback.unmap();
  const decoded = unpackOctreePowerCoarseLevelSetControl(new Uint32Array(result, 0, 16));
  assert.deepEqual(decoded, {
    flags: 0, firstErrorRow: 0xffff_ffff, rowCount, advectedRows: rowCount, uniformUpdates: 0,
    transitionUpdates: 0, representationPasses: 0, correctedRows: 1, interfaceRows: 1,
    contributionCount: rowCount, generation: 41, valid: OCTREE_POWER_COARSE_LEVELSET_VALID,
  });
  assert.equal(decoded.uniformUpdates, 0);
  const output = new Float32Array(result, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, rowCount * 4);
  assert.ok(Math.abs(output[0] + 0.05) < 1e-6); assert.ok(Math.abs(output[1] + 0.05) < 1e-6);
  assert.ok(Math.abs(output[2] - 0.05) < 1e-6);
  const directory = new Uint32Array(result, directoryOffset, schedule.plan.sampleDirectoryBytes / 4);
  assert.deepEqual(Array.from(directory.subarray(0, 7)),
    [OCTREE_POWER_COARSE_LEVELSET_VALID, 41, rowCount, 2, 2, 2, 2]);
  assert.equal(new Float32Array(result, directoryOffset + 28, 1)[0], 1);
  let indexedRows = 0;
  for (let slot = 0; slot < schedule.plan.rowCapacity; slot += 1) {
    const base = 8 + slot * 8;
    if (directory[base] === 0) continue;
    indexedRows += 1;
    assert.equal(directory[base + 5] & (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite),
      OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);
  }
  assert.equal(indexedRows, rowCount, "PUBLISHED implies every requested compact row has a directory key");

  // Bootstrap is a geometry/sign publication. A rank-deficient Stage-A row
  // has w=0, but at dt=0 its velocity is mathematically unused and must not
  // block the first coarse directory. Once time advances, the same input is
  // strictly invalid and publication remains fail-closed.
  device.queue.writeBuffer(velocityBuffer, 0, new Float32Array(rowCount * 4));
  const runVelocityGuard = async (dt: number, generation: number) => {
    device.queue.writeBuffer(structuredControl, 0,
      new Uint32Array([0, 0xffff_ffff, rowCount, generation, 0, rowCount]));
    const guardReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const guardEncoder = device.createCommandEncoder();
    schedule.encode(new PassBroker(guardEncoder), { headers, structured, rowCount },
      { dimensions: [2, 2, 2], physicalCellSize: 1, dt, generation });
    guardEncoder.copyBufferToBuffer(schedule.control, 0, guardReadback, 0,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
    guardEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, guardReadback,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
    device.queue.submit([guardEncoder.finish()]); schedule.retireSubmittedEncoder(guardEncoder);
    await guardReadback.mapAsync(GPUMapMode.READ);
    const bytes = guardReadback.getMappedRange().slice(0); guardReadback.unmap(); guardReadback.destroy();
    return { control: unpackOctreePowerCoarseLevelSetControl(new Uint32Array(bytes, 0, 16)),
      directoryState: new Uint32Array(bytes)[16] };
  };
  const bootstrapWithoutVelocity = await runVelocityGuard(0, 42);
  assert.equal(bootstrapWithoutVelocity.control.flags, 0);
  assert.equal(bootstrapWithoutVelocity.control.valid, OCTREE_POWER_COARSE_LEVELSET_VALID);
  assert.equal(bootstrapWithoutVelocity.directoryState, OCTREE_POWER_COARSE_LEVELSET_VALID);
  const advancedWithoutVelocity = await runVelocityGuard(0.01, 43);
  assert.equal(advancedWithoutVelocity.control.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidVelocity);
  assert.equal(advancedWithoutVelocity.control.valid, 0);
  assert.equal(advancedWithoutVelocity.directoryState, 0);

  coarse.upload(new Map(Array.from({ length: rowCount }, (_, row) => [row, {
    phi: 0, minimumPhi: 0, maximumPhi: 0, flags: 0, generation: 0, fineSampleCount: 0,
  }] as const)));
  const invalidReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const invalidEncoder = device.createCommandEncoder();
  device.queue.writeBuffer(structuredControl, 0,
    new Uint32Array([0, 0xffff_ffff, rowCount, 44, 0, rowCount]));
  schedule.encode(new PassBroker(invalidEncoder), { headers, structured, rowCount },
    { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, generation: 44 });
  invalidEncoder.copyBufferToBuffer(schedule.control, 0, invalidReadback, 0,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  invalidEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, invalidReadback,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
  device.queue.submit([invalidEncoder.finish()]); schedule.retireSubmittedEncoder(invalidEncoder);
  await invalidReadback.mapAsync(GPUMapMode.READ);
  const invalidResult = invalidReadback.getMappedRange().slice(0); invalidReadback.unmap();
  const invalidControl = unpackOctreePowerCoarseLevelSetControl(new Uint32Array(invalidResult, 0, 16));
  assert.equal(invalidControl.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidSource,
    "invalid source records fail closed instead of becoming valid zero phi");
  assert.ok(invalidControl.firstErrorRow < rowCount);
  assert.equal(invalidControl.valid, 0);
  assert.equal(new Uint32Array(invalidResult)[16], 0, "invalid source cannot publish an authoritative directory");
  schedule.destroy(); headers.destroy(); velocityBuffer.destroy(); rowGeometry.destroy(); structuredControl.destroy();
  liveRowDispatch.destroy();
  offsets.destroy(); contributions.destroy();
  readback.destroy(); invalidReadback.destroy(); coarse.destroy(); topology.destroy(); device.destroy();
});
