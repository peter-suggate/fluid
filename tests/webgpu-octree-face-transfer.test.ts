import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES,
  OCTREE_FACE_PREVIOUS_CONTROL_BYTES,
  OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES,
  OCTREE_FACE_PREVIOUS_RECORD_BYTES,
  OCTREE_FACE_PREVIOUS_VALID_OFFSET_BYTES,
  octreeFaceTopologyTransferShader,
  octreeFaceTransferRadixFields,
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
  assert.equal(OCTREE_FACE_PREVIOUS_RECORD_BYTES, 20);
  assert.equal(OCTREE_FACE_PREVIOUS_CONTROL_BYTES, 64);
  assert.equal(OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES, 52);
  assert.equal(OCTREE_FACE_PREVIOUS_VALID_OFFSET_BYTES, 56);
  assert.equal(plan.recordBytes, 0);
  assert.equal(plan.scratchBytes, plan.indexBytes);
  assert.equal(plan.dispatchBytes, 36);
  assert.equal(OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES, 32);
  assert.equal(plan.allocatedBytes, 3_599_428);
  const uiSized = planOctreeFaceTopologyTransfer(165_888, { keyDimensions: [24, 18, 16] });
  assert.equal(uiSized.sortPasses, 8,
    "exact immutable bounds remove the 24 provably-zero high-nibble passes");
  assert.deepEqual(octreeFaceTransferRadixFields([24, 18, 16]), [
    { field: 0, digits: 2 }, { field: 1, digits: 2 },
    { field: 2, digits: 2 }, { field: 3, digits: 2 },
  ]);
  assert.equal(planOctreeFaceTopologyTransfer(1, { keyDimensions: [1_000_000, 1, 1] }).sortPasses, 13,
    "large domains retain every exact origin/span nibble they need");
  const inspected = planOctreeFaceTopologyTransfer(125_488, { retainRecords: true });
  assert.equal(inspected.recordBytes, 125_488 * 24);
  assert.equal(inspected.scratchBytes, inspected.recordBytes);
  assert.equal(inspected.allocatedBytes, 6_086_852);
  assert.throws(() => planOctreeFaceTopologyTransfer(0), /positive/);
});

test("GPU topology transfer has exact, prolongation, restriction, and fail-closed paths", () => {
  assert.match(octreeFaceTopologyTransferShader, /fn radixHistogram/);
  assert.match(octreeFaceTopologyTransferShader, /fn radixPrefix/);
  assert.match(octreeFaceTopologyTransferShader, /fn radixScatter/);
  assert.match(octreeFaceTopologyTransferShader,
    /fn publishSortDispatch[\s\S]*previousControl\[4\] = \(count \+ 255u\) \/ 256u/,
    "the captured GPU count publishes the rounded live radix dispatch");
  assert.match(octreeFaceTopologyTransferShader,
    /fn publishTransferDispatches[\s\S]*max\(oldCount, nextCount\)[\s\S]*previousControl\[10\] = \(nextCount \+ 255u\) \/ 256u/,
    "validation covers both generations while transfer is bounded by the new generation");
  assert.match(octreeFaceTopologyTransferShader,
    /let rounded = previousControl\[4\] \* 256u;[\s\S]*select\(INVALID, gid\.x, gid\.x < count\)/,
    "the final live block must initialize every padding lane to INVALID");
  assert.equal(octreeFaceTopologyTransferShader.match(/block < previousControl\[4\]/g)?.length, 2,
    "prefix totals and offsets must scan only the live block prefix");
  const scatter = octreeFaceTopologyTransferShader.slice(octreeFaceTopologyTransferShader.indexOf("fn radixScatter"),
    octreeFaceTopologyTransferShader.indexOf("fn validateTopology"));
  assert.doesNotMatch(scatter.slice(0, scatter.lastIndexOf("workgroupBarrier")), /return;/,
    "all 256 scatter lanes must reach both workgroup barriers");
  assert.match(octreeFaceTopologyTransferShader, /struct PreviousFace \{ originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32 \}/);
  assert.match(octreeFaceTopologyTransferShader, /fn findFace/);
  assert.match(octreeFaceTopologyTransferShader, /childCount == 4u/);
  assert.match(octreeFaceTopologyTransferShader, /0\.25 \* \(previousFaces/);
  assert.match(octreeFaceTopologyTransferShader, /parentSpan = span \* 2u/);
  assert.match(octreeFaceTopologyTransferShader, /atomicStore\(&diagnostics\[3\], 1u\)/);
  assert.match(octreeFaceTopologyTransferShader, /publishHash/);
});

test("all radix data stages consume one storage-separated GPU-authored live dispatch", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 },
    GPUShaderStage: { COMPUTE: 1 },
  });
  const created: Array<{ label?: string; usage: number; buffer: GPUBuffer }> = [];
  const buffer = (size: number, usage = 31, label?: string) => ({ size, usage, label, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer({ label, size, usage }: { label?: string; size: number; usage: number }) {
      const gpuBuffer = buffer(size, usage, label); created.push({ label, usage, buffer: gpuBuffer }); return gpuBuffer;
    },
    createBindGroupLayout: () => ({}), createShaderModule: () => ({}), createPipelineLayout: () => ({}),
    createComputePipeline: ({ label }: { label: string }) => ({ label }),
    createBindGroup: ({ label }: { label: string }) => ({ label }),
  } as unknown as GPUDevice;
  const source: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 64, faceCapacity: 1_000, faceBytes: 32_000, incidenceBytes: 4, allocatedBytes: 32_004 },
    control: buffer(16), faces: buffer(32_000), incidence: buffer(4), parity: buffer(16),
  };
  const transfer = new WebGPUOctreeFaceTopologyTransfer(device, source);
  const direct: string[] = [], indirect: Array<{ label: string; source: GPUBuffer; offset: number }> = [], copies: unknown[][] = [];
  const bound: Array<{ pipeline: string; group: string }> = [];
  let current = "";
  const encoder = {
    clearBuffer() {},
    copyBufferToBuffer(...args: unknown[]) { copies.push(args); },
    beginComputePass() { return {
      setPipeline(pipeline: { label: string }) { current = pipeline.label; },
      setBindGroup(_index: number, group: { label: string }) { bound.push({ pipeline: current, group: group.label }); },
      dispatchWorkgroups() { direct.push(current); },
      dispatchWorkgroupsIndirect(sourceBuffer: GPUBuffer, offset: number) { indirect.push({ label: current, source: sourceBuffer, offset }); },
      end() {},
    }; },
  } as unknown as GPUCommandEncoder;
  const generation = buffer(64, 31, "Power-face generation control");
  transfer.encodeCapture(encoder, { buffer: generation, offsetBytes: 28 });
  transfer.encodeTransfer(encoder);
  const previous = transfer.previousPublication;
  assert.equal(previous.faceCapacity, transfer.plan.faceCapacity);
  assert.equal(previous.sortCapacity, transfer.plan.sortCapacity);
  assert.ok(copies.some((copy) => copy[0] === generation && copy[1] === 28
    && copy[2] === previous.control && copy[3] === OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES && copy[4] === 4),
  "the previous publication generation must be copied from the GPU authority, not inferred on the host");
  assert.ok(direct.includes("Publish compact previous octree faces"),
    "generation validity publishes only after sorted-key validation");
  const dispatch = created.find((entry) => entry.label === "Previous octree face live radix dispatch")!;
  assert.equal(dispatch.usage, GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
  assert.equal(dispatch.usage & GPUBufferUsage.STORAGE, 0);
  assert.ok(copies.some((copy) => (copy[0] as { label?: string }).label === "Previous octree face control"
    && copy[1] === 16 && copy[2] === dispatch.buffer && copy[4] === 12));
  assert.equal(indirect.length, 4 + 2 * 32);
  assert.ok(indirect.every((item) => item.source === dispatch.buffer));
  assert.equal(indirect.filter((item) => item.offset === 0).length, 2 + 2 * 32,
    "capture, sort initialization, histogram, and scatter consume the old live-count dispatch");
  assert.equal(indirect.filter((item) => item.offset === 12).length, 1);
  assert.equal(indirect.filter((item) => item.offset === 24).length, 1);
  assert.equal(indirect.filter((item) => item.label.startsWith("Histogram previous")).length, 32);
  assert.equal(indirect.filter((item) => item.label.startsWith("Scatter previous")).length, 32);
  assert.equal(indirect.filter((item) => item.label === "Prepare previous octree face sort").length, 1);
  assert.ok(!direct.some((label) => /Capture compact|Prepare previous octree face sort|Histogram previous|Scatter previous|Validate octree|Transfer canonical/.test(label)),
    "no face data stage may dispatch the fixed capacity");
  assert.equal(bound.find((item) => item.pipeline === "Prepare previous octree face sort")?.group,
    "Octree face topology transfer bindings",
    "an even radix plan starts in canonical A and therefore also finishes in canonical A");

  const oddStart = bound.length;
  const odd = new WebGPUOctreeFaceTopologyTransfer(device, source, { keyDimensions: [1_000_000, 16, 16] });
  assert.equal(odd.plan.sortPasses % 2, 1, "the regression domain must exercise odd radix parity");
  odd.encodeTransfer(encoder);
  const oddBindings = bound.slice(oddStart);
  assert.equal(oddBindings.find((item) => item.pipeline === "Prepare previous octree face sort")?.group,
    "Octree face topology transfer swapped radix bindings",
    "an odd radix plan initializes scratch so its last scatter lands in canonical A without a copy");
  assert.ok(oddBindings.filter((item) => item.pipeline.startsWith("Histogram previous"))
    .every((item, index) => item.group === (index % 2 === 0
      ? "Octree face topology transfer swapped radix bindings"
      : "Octree face topology transfer bindings")),
  "odd radix passes must ping-pong from scratch to canonical storage");
  for (const pipeline of ["Validate octree face topology transfer", "Transfer canonical octree face velocities"]) {
    assert.equal(oddBindings.find((item) => item.pipeline === pipeline)?.group,
      "Octree face topology transfer bindings",
      `${pipeline} must consume canonical A regardless of radix parity`);
  }
  odd.destroy();
  transfer.destroy();
});

test("face publication captures old IDs before rebuild and transfers after deterministic emit", () => {
  const encode = WebGPUOctreeFaceMirror.prototype.encodeTopology.toString().replace(/\s+/g, "");
  const capture = encode.indexOf("this.topologyTransfer?.encodeCapture(encoder,previousGeneration)");
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
  const transfer = new WebGPUOctreeFaceTopologyTransfer(device, source, {
    retainRecords: true, keyDimensions: [1_000_000, 16, 16],
  });
  assert.equal(transfer.plan.sortPasses % 2, 1,
    "Dawn transfer coverage must exercise an odd exact-key radix plan");
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
