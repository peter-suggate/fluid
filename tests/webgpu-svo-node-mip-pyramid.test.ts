import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planSvoNodeMipPyramid, SVO_NODE_MIP_LAYOUT } from "../lib/svo-node-mip-pyramid";
import {
  WEBGPU_SVO_NODE_MIP_LAYOUT,
  WebGpuSvoNodeMipPyramid,
  webgpuSvoNodeMipSamplingValidationWGSL,
} from "../lib/webgpu-svo-node-mip-pyramid";

interface MockTexture { descriptor: GPUTextureDescriptor; destroyed: boolean; createView(): GPUTextureView; destroy(): void }
interface MockBuffer { descriptor: GPUBufferDescriptor; destroyed: boolean; destroy(): void }

function mockDevice() {
  const textures: MockTexture[] = [];
  const buffers: MockBuffer[] = [];
  const textureWrites: Array<{ destination: GPUTexelCopyTextureInfo; data: Uint8Array; layout: GPUTexelCopyBufferLayout; size: GPUExtent3D }> = [];
  const bufferWrites: unknown[][] = [];
  const sampler = {} as GPUSampler;
  const device = {
    queue: {
      writeBuffer: (...args: unknown[]) => bufferWrites.push(args),
      writeTexture: (destination: GPUTexelCopyTextureInfo, data: GPUAllowSharedBufferSource, layout: GPUTexelCopyBufferLayout, size: GPUExtent3D) => {
        textureWrites.push({ destination, data: new Uint8Array(data as ArrayBuffer), layout, size });
      },
    },
    createSampler: () => sampler,
    createTexture: (descriptor: GPUTextureDescriptor) => {
      const value: MockTexture = { descriptor, destroyed: false, createView: () => ({}) as GPUTextureView, destroy() { value.destroyed = true; } };
      textures.push(value); return value as unknown as GPUTexture;
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const value: MockBuffer = { descriptor, destroyed: false, destroy() { value.destroyed = true; } };
      buffers.push(value); return value as unknown as GPUBuffer;
    },
  } as unknown as GPUDevice;
  return { device, textures, buffers, textureWrites, bufferWrites, sampler };
}

test("WebGPU owner uploads directory/pages and atomically swaps complete generations", () => {
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 4, COPY_DST: 2 }, GPUBufferUsage: { STORAGE: 128, COPY_DST: 8 } });
  const mock = mockDevice();
  const owner = new WebGpuSvoNodeMipPyramid(mock.device);
  const first = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 1 });
  owner.beginGeneration(first);
  assert.equal(owner.publish().published, false, "payload is not published before upload");
  owner.uploadInteriorPage(first.pages[0].key, new Uint8Array(8 ** 3 * 4).fill(64));
  const published = owner.publish();
  assert.equal(published.published, true);
  assert.equal(owner.visibleGeneration()?.generation, 1);
  assert.equal(mock.bufferWrites.length, 1);
  assert.equal(mock.textureWrites.length, 2, "sampled directory and page payload are uploaded");
  assert.deepEqual(mock.textureWrites[1].size, [10, 10, 10]);
  assert.equal(mock.textureWrites[1].layout.bytesPerRow, 40);
  assert.equal(mock.textureWrites[1].data.byteLength, SVO_NODE_MIP_LAYOUT.bytesPerPage);
  assert.equal(owner.visibleGeneration()?.directoryTexture, mock.textures[1] as unknown as GPUTexture);

  const second = planSvoNodeMipPyramid({ generation: 2, occupiedPages: [[0, 0, 0]], levelCount: 1 });
  owner.beginGeneration(second);
  assert.equal(owner.telemetry().fallback, "previous-complete-generation");
  assert.equal(owner.publish().published, false);
  assert.equal(owner.visibleGeneration()?.generation, 1);
  owner.uploadInteriorPage(second.pages[0].key, new Uint8Array(8 ** 3 * 4));
  assert.equal(owner.publish().published, true);
  assert.equal(owner.visibleGeneration()?.generation, 2);
  assert.equal(mock.textures[0].destroyed, true);
  assert.equal(mock.textures[1].destroyed, true);
  assert.equal(mock.buffers[0].destroyed, true);
  owner.destroy();
  assert.equal(mock.textures.at(-1)?.destroyed, true);
  assert.equal(mock.buffers.at(-1)?.destroyed, true);
});

test("WebGPU owner publishes an empty complete generation without zero-sized resources", () => {
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 4, COPY_DST: 2 }, GPUBufferUsage: { STORAGE: 128, COPY_DST: 8 } });
  const mock = mockDevice();
  const owner = new WebGpuSvoNodeMipPyramid(mock.device);
  const plan = planSvoNodeMipPyramid({ generation: 7, occupiedPages: [], levelCount: 4 });
  owner.beginGeneration(plan);
  assert.equal(owner.publish().published, true);
  assert.deepEqual(mock.textures[0].descriptor.size, [1, 1, 1]);
  assert.deepEqual(mock.textures[1].descriptor.size, [2, 1]);
  assert.equal(mock.buffers[0].descriptor.size, 32);
  owner.destroy();
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("node-mip binding-free sampling WGSL compiles on WebGPU", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU validation" }, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const module = device.createShaderModule({ code: webgpuSvoNodeMipSamplingValidationWGSL });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    assert.equal(WEBGPU_SVO_NODE_MIP_LAYOUT.format, "rgba8unorm");
  } finally { device.destroy(); }
});
