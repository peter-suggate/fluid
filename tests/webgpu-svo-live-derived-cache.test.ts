import assert from "node:assert/strict";
import test from "node:test";
import { planSvoNodeMipPyramid, SVO_NODE_MIP_LAYOUT } from "../lib/svo-node-mip-pyramid";
import { WebGpuLiveSvoDerivedPageState } from "../lib/webgpu-svo-live-derived-cache";
import { WebGpuLiveSvoNodeMipPyramid } from "../lib/webgpu-svo-node-mip-pyramid";
import { WebGpuLiveSvoTetrahedralRadiance } from "../lib/webgpu-svo-tetrahedral-radiance";

class BufferMock {
  destroyed = false;
  constructor(readonly descriptor: GPUBufferDescriptor) {}
  destroy() { this.destroyed = true; }
}
class TextureMock {
  destroyed = false;
  constructor(readonly descriptor: GPUTextureDescriptor) {}
  createView() { return {} as GPUTextureView; }
  destroy() { this.destroyed = true; }
}

function mockDevice() {
  const buffers: BufferMock[] = [], textures: TextureMock[] = [];
  const bufferWrites: unknown[][] = [], textureWrites: unknown[][] = [];
  const device = {
    queue: {
      writeBuffer: (...args: unknown[]) => bufferWrites.push(args),
      writeTexture: (...args: unknown[]) => textureWrites.push(args),
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => { const value = new BufferMock(descriptor); buffers.push(value); return value; },
    createTexture: (descriptor: GPUTextureDescriptor) => { const value = new TextureMock(descriptor); textures.push(value); return value; },
    createSampler: () => ({}),
  } as unknown as GPUDevice;
  return { device, buffers, textures, bufferWrites, textureWrites };
}

function installGpuConstants() {
  const oldBuffer = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const oldTexture = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_DST: 2 } });
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { TEXTURE_BINDING: 1, COPY_DST: 2 } });
  return () => {
    if (oldBuffer) Object.defineProperty(globalThis, "GPUBufferUsage", oldBuffer); else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
    if (oldTexture) Object.defineProperty(globalThis, "GPUTextureUsage", oldTexture); else delete (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage;
  };
}

test("live derived state immediately invalidates dirty pages and publishes only a complete worklist", () => {
  const restore = installGpuConstants();
  try {
    const mock = mockDevice();
    const state = new WebGpuLiveSvoDerivedPageState(mock.device, 8, "test");
    assert.equal(mock.textures.length, 1);
    state.begin({ generation: 1, dirtySlots: [0, 1] });
    assert.equal(state.availability(0), "unavailable");
    state.markComplete(0);
    assert.deepEqual(state.publish(), { published: false, generation: 0, pendingDirtyPages: 1, unavailableDirtyPages: 2 });
    state.markComplete(1);
    assert.equal(state.publish().published, true);
    assert.equal(state.availability(0), "current");
    state.begin({ generation: 2, dirtySlots: [1] });
    assert.equal(state.availability(0), "current", "clean pages remain current while an unrelated delta is pending");
    assert.equal(state.availability(1), "unavailable", "old content is never visible inside a dirty page");
    state.markComplete(1); assert.equal(state.publish().published, true);
    assert.equal(state.availability(0), "last-known-good", "clean page data remains reusable across complete generations");
    assert.equal(mock.textures.length, 1, "updates do not allocate GPU resources");
    state.destroy(); assert.equal(mock.textures[0].destroyed, true);
  } finally { restore(); }
});

test("fixed node-mip atlas stages GPU-built generations without CPU page content", () => {
  const restore = installGpuConstants();
  try {
    const mock = mockDevice();
    const owner = new WebGpuLiveSvoNodeMipPyramid(mock.device, {
      pageCapacity: 4, atlasTexels: [20, 20, 10], directPageTableDimensions: [4, 4, 4],
    });
    const allocations = [mock.buffers.length, mock.textures.length];
    const first = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 1 });
    owner.prepareGpuUpdate(first); owner.acceptGpuUpdate(1);
    assert.equal(owner.visibleGeneration()?.pageValidity.capacity, 4);
    const writesBeforeSecondUpdate = mock.textureWrites.length;
    const second = planSvoNodeMipPyramid({ generation: 2, occupiedPages: [[0, 0, 0]], levelCount: 1 });
    owner.prepareGpuUpdate(second); owner.acceptGpuUpdate(2);
    assert.equal(mock.textureWrites.length, writesBeforeSecondUpdate,
      "same-layout GPU revisions do not upload directory, table, or page content");
    assert.deepEqual([mock.buffers.length, mock.textures.length], allocations);
    assert.equal(mock.textureWrites.some((write) => (write[3] as number[])[0] === SVO_NODE_MIP_LAYOUT.physicalSize), false,
      "live page interiors are never CPU-uploaded");
    owner.destroy(); assert.ok(mock.buffers.every((buffer) => buffer.destroyed)); assert.ok(mock.textures.every((texture) => texture.destroyed));
  } finally { restore(); }
});

test("fixed radiance atlas exposes storage targets without CPU content uploads", () => {
  const restore = installGpuConstants();
  try {
    const mock = mockDevice();
    const owner = new WebGpuLiveSvoTetrahedralRadiance(mock.device, { pageCapacity: 2, atlasTexels: [20, 10, 10] });
    const allocations = [mock.buffers.length, mock.textures.length];
    const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 1 });
    const target = owner.prepareGpuUpdate(plan); owner.acceptGpuUpdate(1);
    assert.equal(target.textures.length, 4);
    assert.ok(mock.textures.slice(0, 4).every((texture) => texture.descriptor.format === "rgba16float"));
    assert.equal(mock.textureWrites.length, 0, "radiance content is never CPU-uploaded");
    assert.deepEqual([mock.buffers.length, mock.textures.length], allocations);
    owner.destroy(); assert.ok(mock.buffers.every((buffer) => buffer.destroyed)); assert.ok(mock.textures.every((texture) => texture.destroyed));
  } finally { restore(); }
});
