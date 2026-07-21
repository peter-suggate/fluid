import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { OCTREE_SURFACE_STATE, WebGPUOctreeSurfacePages, octreeSurfacePageShader } from "../lib/webgpu-octree-surface-pages";

test("Dawn executes the complete optional 4-cubed page lifecycle", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for 4-cubed GPU page checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  // Retain the native GPU wrapper for the full test lifetime. Dropping it while
  // adapter/device work is in flight can release Dawn's Metal instance early.
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const leafData = new ArrayBuffer(96);
  const leafU32 = new Uint32Array(leafData), leafF32 = new Float32Array(leafData);
  leafU32.set([0, 1, 0, 0], 0); leafF32.set([-0.25, 1, 0, 0, 0, 0, 0, 0], 4);
  leafU32.set([1, 1, 0, 0], 12); leafF32.set([0.75, 1, 0, 0, 0, 0, 0, 0], 16);
  const make = (data: ArrayBufferView, usage: GPUBufferUsageFlags) => {
    const upload = new Uint8Array(data.byteLength);
    upload.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    const buffer = device.createBuffer({ size: data.byteLength, usage });
    device.queue.writeBuffer(buffer, 0, upload);
    return buffer;
  };
  const leaves = make(new Uint8Array(leafData), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const candidates = make(new Uint32Array([0, OCTREE_SURFACE_STATE.core, 1, OCTREE_SURFACE_STATE.halo]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const candidateControl = make(new Uint32Array([2, 1, 1, 1]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
  const pages = new WebGPUOctreeSurfacePages(device, {
    leaves, candidates: { candidates, countAndDispatch: candidateControl },
  }, 2, [128, 128, 128], [1, 1, 1], { pageResolution: 4, maximumPages: 2, maximumResidentFraction: 1 });
  const readback = device.createBuffer({ size: pages.plan.arenaBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const compilation = await device.createShaderModule({ code: octreeSurfacePageShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const encoder = device.createCommandEncoder();
  pages.encodeLifecycle(encoder); pages.encodeTransport(encoder, 0.1); pages.encodeRedistance(encoder, 4); pages.encodeVolumeCorrection(encoder);
  encoder.copyBufferToBuffer(pages.arena, 0, readback, 0, pages.plan.arenaBytes);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
  const copy = readback.getMappedRange().slice(0); readback.unmap();
  const words = new Uint32Array(copy), floats = new Float32Array(copy);
  assert.equal(pages.plan.pageResolution, 4); assert.equal(pages.plan.samplesPerPage, 64);
  assert.equal(words[3], 0); assert.equal(words[6], 2);
  assert.ok([...floats.slice(pages.plan.phiAOffsetWords, pages.plan.phiAOffsetWords + 128)].every(Number.isFinite));
  // Native Dawn Metal teardown is independently unstable for this allocation;
  // correctness is established after submitted work and mapped readback.
});
