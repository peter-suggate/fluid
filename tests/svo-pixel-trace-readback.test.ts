import assert from "node:assert/strict";
import test from "node:test";

import { StagedReadback } from "../lib/core/staged-readback";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

Object.assign(globalThis, { GPUBufferUsage: { COPY_DST: 8, MAP_READ: 1 }, GPUMapMode: { READ: 1 } });

const device = {
  createBuffer: () => ({ mapAsync: async () => {}, getMappedRange: () => new ArrayBuffer(4), unmap() {}, destroy() {} }),
} as unknown as GPUDevice;

/** One probe half over a real readback, answering with the frame it was copied in. */
function half(name: string, frame: () => number) {
  const readback = new StagedReadback(device, { label: name, size: 4 });
  let copiedIn = -1;
  return {
    get idle() { return readback.idle; },
    get copied() { return readback.state === "copied"; },
    writeRequest() {},
    encodeReadback: (encoder: GPUCommandEncoder) => readback.encode(encoder, () => { copiedIn = frame(); }),
    discard: () => readback.discard(),
    read: (current: () => boolean) => readback.read(() => current()
      ? { stages: [], records: [], name, frame: copiedIn } : undefined),
  };
}

const pass = { setPipeline() {}, setBindGroup() {}, draw() {}, dispatchWorkgroups() {}, end() {} };
const encoder = { beginRenderPass: () => pass, beginComputePass: () => pass } as unknown as GPUCommandEncoder;

test("a raster probe half no lighting half was encoded beside is never merged into a later trace", async () => {
  let frame = 1;
  const lighting = half("lighting", () => frame), brick = half("brick", () => frame);
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    probeRequest: { pixelX: 1, pixelY: 2, token: 1 }, rasterPrimary: true,
    brickProbePipeline: {}, brickProbeBindGroup: {}, brickProbeBuffers: brick, probeBuffers: lighting,
    probeBindGroup: {}, probeTargetView: {}, bindGroup: {}, frameConePrepass: true,
  }) as SparseVoxelDrySceneRenderer;
  const fields = renderer as unknown as Record<string, unknown>;
  const encodeBrick = () => (renderer as unknown as { encodeBrickRasterProbe(encoder: GPUCommandEncoder): void }).encodeBrickRasterProbe(encoder);

  // Frame 1: the raster half encodes beside the primary; the lighting probe is still compiling.
  encodeBrick();
  assert.equal(brick.copied, true);
  assert.equal(renderer.encodePixelTrace(encoder), false);
  assert.equal(brick.idle, true, "the orphaned raster half is discarded with the refusal");

  // Frame 2 has no raster half; frame 3's encode runs without the prepass before the read resolves.
  frame = 2;
  fields.rasterPrimary = false;
  fields.probePipeline = {};
  assert.equal(renderer.encodePixelTrace(encoder), true);
  assert.equal(renderer.pixelTraceCopied, true);
  fields.frameConePrepass = false;
  const trace = await renderer.readPixelTrace() as unknown as { name: string; frame: number; conePrepass?: { active: boolean } };
  assert.equal(trace.name, "lighting");
  assert.equal(trace.frame, 2);
  assert.equal(trace.conePrepass?.active, true, "the prepass flag is the encoded frame's, not the reading frame's");
  assert.equal(renderer.pixelTraceCopied, false);
});

test("an unread raster half from an earlier frame is replaced, not merged", async () => {
  let frame = 1;
  const lighting = half("lighting", () => frame), brick = half("brick", () => frame);
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    probeRequest: { pixelX: 1, pixelY: 2, token: 1 }, rasterPrimary: true,
    brickProbePipeline: {}, brickProbeBindGroup: {}, brickProbeBuffers: brick, probeBuffers: lighting,
    probeBindGroup: {}, probeTargetView: {}, bindGroup: {}, probePipeline: {}, frameConePrepass: false,
  }) as SparseVoxelDrySceneRenderer;
  const encodeBrick = () => (renderer as unknown as { encodeBrickRasterProbe(encoder: GPUCommandEncoder): void }).encodeBrickRasterProbe(encoder);
  // Frame 1's host never asked for the lighting half.
  encodeBrick();
  frame = 2;
  encodeBrick();
  assert.equal(renderer.encodePixelTrace(encoder), true);
  const halves: { name: string; frame: number }[] = [];
  const read = (buffers: ReturnType<typeof half>) => { const original = buffers.read; buffers.read = (current) => original(current).then(value => { if (value) halves.push(value); return value; }); };
  read(lighting); read(brick);
  await renderer.readPixelTrace().catch(() => undefined);
  assert.deepEqual(halves.map(value => value.frame), [2, 2]);
});
