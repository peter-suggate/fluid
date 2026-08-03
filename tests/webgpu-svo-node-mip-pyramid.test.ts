import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planSvoNodeMipPyramid, SVO_NODE_MIP_LAYOUT } from "../lib/svo-node-mip-pyramid";
import {
  WEBGPU_SVO_NODE_MIP_LAYOUT,
  WebGpuSvoNodeMipPyramid,
  createWebGpuSvoNodeMipDirectPageTable,
  webgpuSvoNodeMipSamplingValidationWGSL,
} from "../lib/webgpu-svo-node-mip-pyramid";

test("direct page table packs level slabs as slot+1 and bounds pathological extents", () => {
  const plan = planSvoNodeMipPyramid({ generation: 3, occupiedPages: [[0, 0, 0], [3, 1, 2]], levelCount: 3 });
  const table = createWebGpuSvoNodeMipDirectPageTable(plan);
  assert.equal(table.ready, true);
  assert.deepEqual(table.dimensions, [4, 2, 6]);
  assert.deepEqual([...table.levelZOffsets.slice(0, 3)], [0, 3, 5]);
  const [width, height] = table.dimensions;
  for (const page of plan.pages) {
    const [x, y, z] = page.key.coordinate;
    const index = ((table.levelZOffsets[page.key.level] + z) * height + y) * width + x;
    assert.equal(table.words[index], page.slot + 1);
  }
  assert.equal(createWebGpuSvoNodeMipDirectPageTable(plan, 2).ready, false,
    "a device-incompatible volume must preserve the compact directory fallback");
});

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
        // Copy the bytes, not the elements: a Uint32Array source through
        // `new Uint8Array(view)` would truncate every word to its low byte and
        // silently make a word-layout assertion untestable.
        const view = data as ArrayBufferView;
        textureWrites.push({ destination, data: new Uint8Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)), layout, size });
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
  assert.equal(mock.textureWrites.length, 3, "sampled directory, direct page table, and page payload are uploaded");
  assert.deepEqual(mock.textureWrites[1].size, [1, 1, 1]);
  assert.equal(mock.textureWrites[1].data[0], 1, "direct-table zero is reserved for non-resident pages");
  assert.deepEqual(mock.textureWrites[2].size, [10, 10, 10]);
  assert.equal(mock.textureWrites[2].layout.bytesPerRow, 40);
  assert.equal(mock.textureWrites[2].data.byteLength, SVO_NODE_MIP_LAYOUT.bytesPerPage);
  assert.equal(owner.visibleGeneration()?.directoryTexture, mock.textures[1] as unknown as GPUTexture);
  assert.equal(owner.visibleGeneration()?.directPageTableTexture, mock.textures[2] as unknown as GPUTexture);

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
  assert.equal(mock.textures[2].destroyed, true);
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
  assert.deepEqual(mock.textures[2].descriptor.size, [1, 1, 1]);
  assert.equal(mock.buffers[0].descriptor.size, 32);
  owner.destroy();
});

test("a directory past the 2D height limit wraps into columns without repacking", () => {
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 4, COPY_DST: 2 }, GPUBufferUsage: { STORAGE: 128, COPY_DST: 8 } });
  const mock = mockDevice();
  // A device that can only address four rows, so six pages must wrap. On real
  // hardware the same arithmetic runs at 16 384 rows, where the hero pond's
  // 5 mm lattice asks for 23 545 pages and used to be refused outright.
  Object.assign(mock.device, { limits: { maxTextureDimension2D: 4 } });
  const owner = new WebGpuSvoNodeMipPyramid(mock.device);
  const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], levelCount: 2 });
  assert.equal(plan.pages.length, 5);
  owner.beginGeneration(plan);
  assert.deepEqual(mock.textures[1].descriptor.size, [4, 3], "two page columns of three rows");
  const directory = mock.textureWrites[0];
  assert.deepEqual(directory.size, [4, 3]);
  assert.equal(directory.layout.bytesPerRow, 2 * SVO_NODE_MIP_LAYOUT.directoryBytesPerPage);
  assert.equal(directory.layout.rowsPerImage, 3);
  // Slot order along the row is what lets the storage buffer and the texture
  // share one array, and it is what keeps the directory sorted by (level,
  // morton) in page-index order for the marcher's binary search.
  const words = new Uint32Array(directory.data.buffer, directory.data.byteOffset, directory.data.byteLength / 4);
  for (const page of plan.pages) {
    const base = page.slot * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage;
    assert.equal(words[base], plan.generation, `page ${page.slot} must sit at linear slot order regardless of the wrap`);
    assert.equal(words[base + 1], page.key.level);
    assert.equal(words[base + 7], page.slot);
  }
  assert.equal(words.length, 2 * 3 * WEBGPU_SVO_NODE_MIP_LAYOUT.directoryWordsPerPage,
    "the unused tail of the last row is zero, which reads back as generation 0 and therefore as a miss");
  owner.destroy();
});

test("the direct page table survives a page count that overflows an argument list", () => {
  // A 3 mm hero lattice plans 108 421 pages. Spreading that many into
  // `Math.max` throws RangeError: Maximum call stack size exceeded, which the
  // world catches and reports as "derived lighting unavailable" — a CPU limit
  // wearing a device limit's clothes.
  const occupiedPages = Array.from({ length: 200_000 }, (_, index) => [index, 0, 0] as const);
  const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages, levelCount: 1 });
  assert.equal(plan.pages.length, 200_000);
  const table = createWebGpuSvoNodeMipDirectPageTable(plan, 262_144);
  assert.equal(table.ready, true);
  assert.deepEqual(table.dimensions, [200_000, 1, 1]);
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
