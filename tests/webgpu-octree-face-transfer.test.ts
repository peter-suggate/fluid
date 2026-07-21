import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES,
  octreeFaceTopologyTransferShader,
  planOctreeFaceTopologyTransfer,
  WebGPUOctreeFaceTopologyTransfer,
} from "../lib/webgpu-octree-face-transfer";
import type { OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";
import { WebGPUOctreeFaceMirror } from "../lib/webgpu-octree-face-mirror";

test("topology transfer uses compact radix-sort storage", () => {
  const plan = planOctreeFaceTopologyTransfer(125_488);
  assert.equal(plan.sortCapacity, 131_072);
  assert.equal(plan.sortPasses, 32);
  assert.equal(plan.previousFaceBytes, 125_488 * 20);
  assert.equal(plan.recordBytes, 0);
  assert.equal(plan.scratchBytes, plan.indexBytes);
  assert.equal(OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES, 32);
  assert.equal(plan.allocatedBytes, 3_599_344);
  const inspected = planOctreeFaceTopologyTransfer(125_488, { retainRecords: true });
  assert.equal(inspected.recordBytes, 125_488 * 24);
  assert.equal(inspected.scratchBytes, inspected.recordBytes);
  assert.equal(inspected.allocatedBytes, 6_086_768);
  assert.throws(() => planOctreeFaceTopologyTransfer(0), /positive/);
});

test("GPU topology transfer has exact, prolongation, restriction, and fail-closed paths", () => {
  assert.match(octreeFaceTopologyTransferShader, /fn radixHistogram/);
  assert.match(octreeFaceTopologyTransferShader, /fn radixPrefix/);
  assert.match(octreeFaceTopologyTransferShader, /fn radixScatter/);
  assert.match(octreeFaceTopologyTransferShader, /struct PreviousFace \{ originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32 \}/);
  assert.match(octreeFaceTopologyTransferShader, /fn findFace/);
  assert.match(octreeFaceTopologyTransferShader, /childCount == 4u/);
  assert.match(octreeFaceTopologyTransferShader, /0\.25 \* \(previousFaces/);
  assert.match(octreeFaceTopologyTransferShader, /parentSpan = span \* 2u/);
  assert.match(octreeFaceTopologyTransferShader, /atomicStore\(&diagnostics\[3\], 1u\)/);
  assert.match(octreeFaceTopologyTransferShader, /publishHash/);
});

test("face publication captures old IDs before rebuild and transfers after deterministic emit", () => {
  const encode = WebGPUOctreeFaceMirror.prototype.encodeTopology.toString().replace(/\s+/g, "");
  const capture = encode.indexOf("this.topologyTransfer?.encodeCapture(encoder)");
  const clear = encode.indexOf("encoder.clearBuffer(this.control");
  const emit = encode.indexOf("this.emitPipeline");
  const transfer = encode.indexOf("this.topologyTransfer?.encodeTransfer(encoder)");
  assert.ok(capture >= 0 && capture < clear);
  assert.ok(emit > clear && transfer > emit);
  const compatibility = WebGPUOctreeFaceMirror.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(compatibility, /this\.encodeTopology\(encoder,rowDispatch\).*this\.encodeRhs\(encoder,rowDispatch,applyRhs\)/);
});

test("Dawn preserves canonical velocities through exact, prolongation, and restriction rebuilds", async (t) => {
  const runtime = process.env.WEBGPU_NODE_MODULE;
  if (!runtime) { t.skip("set WEBGPU_NODE_MODULE for GPU face-transfer checks"); return; }
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
  const capacity = 8;
  const control = device.createBuffer({ size: 16, usage: storage });
  const faces = device.createBuffer({ size: capacity * 32, usage: storage });
  const incidence = device.createBuffer({ size: 4, usage: storage });
  const parity = device.createBuffer({ size: 16, usage: storage });
  const source: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 1, faceCapacity: capacity, faceBytes: capacity * 32, incidenceBytes: 4, allocatedBytes: capacity * 32 + 36 },
    control, faces, incidence, parity,
  };
  const transfer = new WebGPUOctreeFaceTopologyTransfer(device, source, { retainRecords: true });
  const descriptor = (origin: readonly [number, number, number], span: number, velocity: number): ArrayBuffer => {
    const bytes = new ArrayBuffer(32); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([0, 0xffffffff, origin[0], origin[1], origin[2], span << 2]); f32[6] = velocity; f32[7] = span * span; return bytes;
  };
  const writeFaces = (items: readonly ArrayBuffer[]): void => {
    const bytes = new Uint8Array(capacity * 32);
    items.forEach((item, index) => bytes.set(new Uint8Array(item), index * 32));
    device.queue.writeBuffer(faces, 0, bytes);
    device.queue.writeBuffer(control, 0, new Uint32Array([items.length, 0, capacity, 1]));
  };
  // Deliberately scramble both key words so transfer correctness depends on
  // the stable LSD radix order (axisSpan nibbles, then packedOrigin nibbles).
  const oldFaces = [
    descriptor([65_548, 6, 6], 2, 4), descriptor([65_540, 4, 4], 4, 3),
    descriptor([65_548, 4, 4], 2, 1), descriptor([65_537, 1, 1], 1, 7),
    descriptor([65_548, 4, 6], 2, 3), descriptor([65_548, 6, 4], 2, 2),
  ];
  writeFaces(oldFaces);
  const captureEncoder = device.createCommandEncoder(); transfer.encodeCapture(captureEncoder);
  device.queue.submit([captureEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  const nextFaces = [
    descriptor([65_537, 1, 1], 1, -1),
    descriptor([65_540, 4, 4], 2, -1), descriptor([65_540, 6, 4], 2, -1),
    descriptor([65_540, 4, 6], 2, -1), descriptor([65_540, 6, 6], 2, -1),
    descriptor([65_548, 4, 4], 4, -1),
  ];
  writeFaces(nextFaces);
  const encoder = device.createCommandEncoder(); transfer.encodeTransfer(encoder);
  const faceReadback = device.createBuffer({ size: capacity * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const recordReadback = device.createBuffer({ size: capacity * 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const diagnosticReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, capacity * 32);
  encoder.copyBufferToBuffer(transfer.records, 0, recordReadback, 0, capacity * 24);
  encoder.copyBufferToBuffer(transfer.diagnostics, 0, diagnosticReadback, 0, 16);
  device.queue.submit([encoder.finish()]);
  await Promise.all([faceReadback.mapAsync(GPUMapMode.READ), recordReadback.mapAsync(GPUMapMode.READ), diagnosticReadback.mapAsync(GPUMapMode.READ)]);
  const faceValues = new Float32Array(faceReadback.getMappedRange().slice(0));
  assert.deepEqual([0, 1, 2, 3, 4, 5].map((index) => faceValues[index * 8 + 6]), [7, 3, 3, 3, 3, 2.5]);
  const records = new Uint32Array(recordReadback.getMappedRange().slice(0));
  assert.deepEqual([0, 1, 2, 3, 4, 5].map((index) => records[index * 6 + 1]), [1, 1, 1, 1, 1, 4]);
  assert.deepEqual([...new Uint32Array(diagnosticReadback.getMappedRange().slice(0))], [6, 0, 0, 0]);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  faceReadback.unmap(); recordReadback.unmap(); diagnosticReadback.unmap();
  faceReadback.destroy(); recordReadback.destroy(); diagnosticReadback.destroy(); transfer.destroy();
  control.destroy(); faces.destroy(); incidence.destroy(); parity.destroy(); device.destroy();
});
