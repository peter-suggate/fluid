import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  buildOctreeFaceTopologyTransfer,
  decodeOctreeFaceKey,
  decodeOctreeFaceVelocityDiagnostics,
  encodeOctreeFaceKey,
  OCTREE_FACE_TRANSPORT_CFL_BYTES,
  OCTREE_FACE_TRANSPORT_PARAMETER_BYTES,
  octreeFaceTransportShader,
  planOctreeFaceTransport,
  WebGPUOctreeFaceTransport,
} from "../lib/webgpu-octree-face-transport";
import type { OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";

test("adaptive face transport uses compact double-buffered scalar storage", () => {
  const plan = planOctreeFaceTransport(8 * 15_686);
  assert.equal(plan.faceCapacity, 125_488);
  assert.equal(plan.velocityBytes, 125_488 * 4);
  assert.equal(plan.allocatedBytes, 2 * 125_488 * 4 + OCTREE_FACE_TRANSPORT_CFL_BYTES + OCTREE_FACE_TRANSPORT_PARAMETER_BYTES);
  assert.ok(plan.allocatedBytes < 1.1 * 1024 * 1024, "the measured 15.7k-row topology stays near one MiB");
  assert.throws(() => planOctreeFaceTransport(0), /positive/);
});

test("compact face diagnostics decode the GPU four-word ABI", () => {
  const bytes = new ArrayBuffer(OCTREE_FACE_TRANSPORT_CFL_BYTES);
  const floats = new Float32Array(bytes); const words = new Uint32Array(bytes);
  floats[0] = 0.75; floats[1] = 3.5; words[2] = 2; words[3] = 417;
  assert.deepEqual(decodeOctreeFaceVelocityDiagnostics(bytes), {
    maxComponentCfl: 0.75,
    maxSpeed_m_s: 3.5,
    nonFiniteCount: 2,
    transportedFaceCount: 417,
  });
  assert.throws(() => decodeOctreeFaceVelocityDiagnostics(new ArrayBuffer(12)), /require 16 bytes/);
});

test("face transport is hierarchical, 2:1 bounded, force-aware, and CFL-reduced", () => {
  assert.match(octreeFaceTransportShader, /side < 2u/);
  assert.match(octreeFaceTransportShader, /local < count/);
  assert.match(octreeFaceTransportShader, /INCIDENCE_PER_ROW = 48u/);
  assert.match(octreeFaceTransportShader, /support = max\(1\.0, f32\(faceSpan\(candidate\)\)\)/);
  assert.match(octreeFaceTransportShader, /departure = centre - params\.dtAcceleration\.x \* advecting/);
  assert.match(octreeFaceTransportShader, /transported \+ params\.dtAcceleration\.x \* component\(params\.dtAcceleration\.yzw/);
  assert.match(octreeFaceTransportShader, /atomicMax\(&cfl\[0\]/);
  assert.match(octreeFaceTransportShader, /faces\[index\]\.normalVelocity = value/);
});

test("face transport encode explicitly reseeds mutable atomic topology", () => {
  const encode = WebGPUOctreeFaceTransport.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /!this\.initialized\|\|options\.reseedFromMirror/);
  assert.match(encode, /this\.seedPipeline/);
  assert.match(encode, /this\.transportPipeline/);
  assert.match(encode, /this\.active=1-this\.active/);
});

test("topology transfer preserves exact faces, prolongs coarse values, and restricts fine flux", () => {
  const pack = (x: number, y: number, z: number) => x | (y << 10) | (z << 20);
  const xAxis = 0;
  const coarse = { packedOrigin: pack(4, 4, 4), axisSpan: xAxis | (4 << 2), normalVelocity: 3 };
  const fineNext = [
    { packedOrigin: pack(4, 4, 4), axisSpan: xAxis | (2 << 2), normalVelocity: 0 },
    { packedOrigin: pack(4, 6, 4), axisSpan: xAxis | (2 << 2), normalVelocity: 0 },
    { packedOrigin: pack(4, 4, 6), axisSpan: xAxis | (2 << 2), normalVelocity: 0 },
    { packedOrigin: pack(4, 6, 6), axisSpan: xAxis | (2 << 2), normalVelocity: 0 },
  ] as const;
  const prolong = buildOctreeFaceTopologyTransfer([coarse], fineNext);
  assert.deepEqual([...prolong.velocities], [3, 3, 3, 3]);
  assert.ok(prolong.records.every((record) => record.sourceCount === 1 && record.oldFaces[0] === 0));
  const fineOld = fineNext.map((face, index) => ({ ...face, normalVelocity: index + 1 }));
  const restrict = buildOctreeFaceTopologyTransfer(fineOld, [coarse]);
  assert.equal(restrict.records[0].sourceCount, 4);
  assert.deepEqual(restrict.records[0].oldFaces, [0, 1, 2, 3]);
  assert.equal(restrict.velocities[0], 2.5, "four equal-area children conservatively average their normal velocity");
  assert.equal(restrict.packedRecords.length, 6);
  const exact = buildOctreeFaceTopologyTransfer([coarse], [coarse]);
  assert.equal(exact.velocities[0], 3);
  assert.equal(exact.records[0].sourceCount, 1);
});

test("canonical face keys retain origins beyond the legacy ten-bit axes", () => {
  const origin = [65_537, 2_000_000, 0xffff_ff00] as const;
  const words = encodeOctreeFaceKey(origin, 2 | (8 << 2));
  assert.deepEqual(decodeOctreeFaceKey(words), { origin: [...origin], axisSpan: 34 });
  const previous = [{ origin, axisSpan: 34, normalVelocity: 9 }];
  const transfer = buildOctreeFaceTopologyTransfer(previous, previous);
  assert.equal(transfer.velocities[0], 9);
  assert.throws(() => buildOctreeFaceTopologyTransfer([...previous, ...previous], previous), /duplicate canonical/);
});

test("Dawn executes adaptive face advection, gravity, publication, and CFL", async (t) => {
  const runtime = process.env.WEBGPU_NODE_MODULE;
  if (!runtime) { t.skip("set WEBGPU_NODE_MODULE for GPU face-transport checks"); return; }
  let dawn: { create(options: string[]): GPU; globals: Record<string, unknown> };
  try {
    dawn = await import(pathToFileURL(runtime).href) as typeof dawn;
  } catch {
    t.skip("local Dawn runtime is unavailable");
    return;
  }
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const control = device.createBuffer({ size: 16, usage: storage });
  const faces = device.createBuffer({ size: 4 * 32, usage: storage });
  const incidence = device.createBuffer({ size: 49 * 4, usage: storage });
  const parity = device.createBuffer({ size: 16, usage: storage });
  device.queue.writeBuffer(control, 0, new Uint32Array([1, 0, 4, 1]));
  const record = new ArrayBuffer(32);
  const recordU32 = new Uint32Array(record); const recordF32 = new Float32Array(record);
  recordU32.set([0, 0xffffffff, 0, 1, 0, 1 | (1 << 2)]); // one interior free-surface y face, span one
  recordF32[6] = 2; recordF32[7] = 1;
  device.queue.writeBuffer(faces, 0, record);
  const incidenceWords = new Uint32Array(25); incidenceWords[0] = 1; incidenceWords[1] = 0;
  device.queue.writeBuffer(incidence, 0, incidenceWords);
  const source: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 1, faceCapacity: 4, faceBytes: 128, incidenceBytes: 100, allocatedBytes: 260 },
    control, faces, incidence, parity,
  };
  const transport = new WebGPUOctreeFaceTransport(device, source, [1, 1, 1]);
  const encoder = device.createCommandEncoder();
  transport.encode(encoder, { dt: 0.1, acceleration: [0, -10, 0], reseedFromMirror: true });
  const faceReadback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const cflReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, 32);
  encoder.copyBufferToBuffer(transport.cfl, 0, cflReadback, 0, 16);
  device.queue.submit([encoder.finish()]);
  await Promise.all([faceReadback.mapAsync(GPUMapMode.READ), cflReadback.mapAsync(GPUMapMode.READ)]);
  const transported = new Float32Array(faceReadback.getMappedRange().slice(0))[6];
  const cflBytes = cflReadback.getMappedRange().slice(0);
  const cfl = new Float32Array(cflBytes)[0];
  const cflWords = new Uint32Array(cflBytes);
  assert.ok(Math.abs(transported - 1) < 1e-6, `expected transported velocity 1, got ${transported}`);
  assert.ok(Math.abs(cfl - 0.1) < 1e-6);
  assert.equal(cflWords[2], 0);
  assert.equal(cflWords[3], 1);
  const diagnosticPromise = transport.readDiagnostics();
  assert.equal(transport.readDiagnostics(), diagnosticPromise, "concurrent telemetry polls share the pooled compact readback");
  assert.deepEqual(await diagnosticPromise, {
    maxComponentCfl: cfl,
    maxSpeed_m_s: 2,
    nonFiniteCount: 0,
    transportedFaceCount: 1,
  });
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  faceReadback.unmap(); cflReadback.unmap();
  faceReadback.destroy(); cflReadback.destroy(); transport.destroy();
  control.destroy(); faces.destroy(); incidence.destroy(); parity.destroy(); device.destroy();
});
