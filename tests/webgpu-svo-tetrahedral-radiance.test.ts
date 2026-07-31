import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_NODE_MIP_LAYOUT, planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { SVO_TETRAHEDRAL_RADIANCE_LAYOUT } from "../lib/svo-tetrahedral-radiance";
import {
  WEBGPU_SVO_TETRAHEDRAL_RADIANCE_LAYOUT,
  WebGpuSvoTetrahedralRadiance,
  createSvoTetrahedralRadiancePage,
  createSvoTetrahedralRadiancePageWithApron,
} from "../lib/webgpu-svo-tetrahedral-radiance";

const DIRECTIONS = SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount;

function offset(x: number, y: number, z: number, size: number): number {
  return ((z * size + y) * size + x) * DIRECTIONS;
}

function sample(page: Uint32Array, x: number, y: number, z: number): number[] {
  const start = offset(x, y, z, SVO_NODE_MIP_LAYOUT.physicalSize);
  return [...page.slice(start, start + DIRECTIONS)];
}

test("packed radiance pages clamp missing neighbours and source resident same-level aprons", () => {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const interior = new Uint32Array(n ** 3 * DIRECTIONS);
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const start = offset(x, y, z, n);
    for (let direction = 0; direction < DIRECTIONS; direction += 1) {
      interior[start + direction] = 1000 * z + 100 * y + 10 * x + direction;
    }
  }

  const clamped = createSvoTetrahedralRadiancePage(interior);
  assert.equal(clamped.length, 10 ** 3 * DIRECTIONS);
  assert.deepEqual(sample(clamped, 0, 0, 0), [0, 1, 2, 3]);
  assert.deepEqual(sample(clamped, 9, 9, 9), [7770, 7771, 7772, 7773]);
  assert.deepEqual(sample(clamped, 4, 5, 6), [5430, 5431, 5432, 5433]);

  const apron = createSvoTetrahedralRadiancePageWithApron([3, 2, 1], interior, ({ page, texel }) =>
    page[0] === 2 && page[1] === 2 && page[2] === 1 && texel[0] === 7
      ? [0x100, 0x200, 0x300, 0x400]
      : undefined);
  assert.deepEqual(sample(apron, 0, 4, 5), [0x100, 0x200, 0x300, 0x400]);
  assert.deepEqual(sample(apron, 9, 4, 5), [4370, 4371, 4372, 4373],
    "a missing positive-X neighbour clamps to the local interior edge");
});

interface MockTexture {
  descriptor: GPUTextureDescriptor;
  destroyed: boolean;
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface TextureWrite {
  destination: GPUTexelCopyTextureInfo;
  words: Uint32Array;
  layout: GPUTexelCopyBufferLayout;
  size: GPUExtent3D;
}

function mockDevice() {
  const textures: MockTexture[] = [];
  const textureWrites: TextureWrite[] = [];
  const device = {
    queue: {
      writeTexture(
        destination: GPUTexelCopyTextureInfo,
        data: GPUAllowSharedBufferSource,
        layout: GPUTexelCopyBufferLayout,
        size: GPUExtent3D,
      ) {
        const view = data as Uint32Array;
        textureWrites.push({ destination, words: new Uint32Array(view), layout, size });
      },
    },
    createTexture(descriptor: GPUTextureDescriptor) {
      const value: MockTexture = {
        descriptor,
        destroyed: false,
        createView: () => ({}) as GPUTextureView,
        destroy() { value.destroyed = true; },
      };
      textures.push(value);
      return value as unknown as GPUTexture;
    },
  } as unknown as GPUDevice;
  return { device, textures, textureWrites };
}

function installWebGpuUsageConstants(): void {
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 4, COPY_DST: 2 } });
}

test("GPU owner allocates four shared-slot RGB9E5 atlases and deinterleaves physical uploads", () => {
  installWebGpuUsageConstants();
  const mock = mockDevice();
  const owner = new WebGpuSvoTetrahedralRadiance(mock.device);
  const plan = planSvoNodeMipPyramid({
    generation: 11,
    occupiedPages: [[1, 0, 0]],
    levelCount: 1,
    atlasPages: [2, 1, 1],
  });
  owner.beginGeneration(plan);

  assert.equal(mock.textures.length, 4);
  for (const texture of mock.textures) {
    assert.equal(texture.descriptor.format, "rgb9e5ufloat");
    assert.equal(texture.descriptor.dimension, "3d");
    assert.deepEqual(texture.descriptor.size, [20, 10, 10]);
  }

  const physical = new Uint32Array(10 ** 3 * DIRECTIONS);
  for (let texel = 0; texel < 10 ** 3; texel += 1) {
    for (let direction = 0; direction < DIRECTIONS; direction += 1) physical[texel * DIRECTIONS + direction] = texel * 4 + direction;
  }
  owner.uploadPhysicalPage(plan.pages[0].key, physical);
  assert.equal(mock.textureWrites.length, 4);
  mock.textureWrites.forEach((write, direction) => {
    assert.equal(write.destination.texture, mock.textures[direction] as unknown as GPUTexture);
    assert.deepEqual(write.destination.origin, plan.pages[0].atlasTexelOrigin);
    assert.deepEqual(write.size, [10, 10, 10]);
    assert.equal(write.layout.bytesPerRow, 40);
    assert.equal(write.layout.rowsPerImage, 10);
    assert.deepEqual([...write.words.slice(0, 3)], [direction, 4 + direction, 8 + direction]);
    assert.equal(write.words.at(-1), 3996 + direction);
  });
  assert.equal(owner.publish().published, true);
  assert.strictEqual(owner.visibleGeneration()?.plan, plan, "the node-mip topology object is the shared slot authority");
  assert.deepEqual(owner.telemetry(), {
    visibleGeneration: 11,
    candidateGeneration: 0,
    residentPages: 1,
    uploadedPages: 1,
    blackPages: 0,
    allocatedBytes: 32_000,
    fallback: "none",
  });
  owner.destroy();
  assert.ok(mock.textures.every((texture) => texture.destroyed));
});

test("candidate publication is atomic and preserves the prior complete radiance generation as fallback", () => {
  installWebGpuUsageConstants();
  const mock = mockDevice();
  const owner = new WebGpuSvoTetrahedralRadiance(mock.device);
  const first = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 1 });
  const interior = new Uint32Array(8 ** 3 * DIRECTIONS);
  owner.beginGeneration(first);
  assert.deepEqual(owner.publish(), { published: false, reason: "incomplete-payload", visible: undefined });
  owner.uploadInteriorPage(first.pages[0].key, interior);
  assert.equal(owner.publish().published, true);
  const firstTextures = mock.textures.slice(0, 4);

  const second = planSvoNodeMipPyramid({ generation: 2, occupiedPages: [[0, 0, 0], [1, 0, 0]], levelCount: 1 });
  owner.beginGeneration(second);
  assert.equal(owner.telemetry().fallback, "previous-complete-generation");
  assert.equal(owner.telemetry().allocatedBytes, 48_000, "candidate and visible allocations are both reported");
  owner.uploadInteriorPage(second.pages[0].key, interior);
  assert.equal(owner.publish().published, false);
  assert.equal(owner.visibleGeneration()?.generation, 1);
  assert.ok(firstTextures.every((texture) => !texture.destroyed));

  owner.uploadInteriorPage(second.pages[1].key, interior, () => [1, 2, 3, 4]);
  assert.equal(owner.publish().published, true);
  assert.equal(owner.visibleGeneration()?.generation, 2);
  assert.ok(firstTextures.every((texture) => texture.destroyed));

  const stale = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [], levelCount: 1 });
  owner.beginGeneration(stale);
  assert.equal(owner.publish().reason, "generation-order");
  assert.equal(owner.visibleGeneration()?.generation, 2);
  owner.destroy();
  assert.ok(mock.textures.every((texture) => texture.destroyed));
  assert.throws(() => owner.beginGeneration(first), /destroyed/);
});

test("black-certified pages complete publication idempotently without queue writes", () => {
  installWebGpuUsageConstants();
  const mock = mockDevice();
  const owner = new WebGpuSvoTetrahedralRadiance(mock.device);
  const plan = planSvoNodeMipPyramid({ generation: 5, occupiedPages: [[0, 0, 0], [1, 0, 0]], levelCount: 1 });
  owner.beginGeneration(plan);

  owner.certifyBlackPage(plan.pages[0].key);
  owner.certifyBlackPage(plan.pages[0].key);
  assert.throws(() => owner.certifyBlackPage({ generation: 5, level: 0, coordinate: [2, 0, 0] }), /not resident/);
  assert.equal(owner.publish().published, false, "certifying one slot twice must not complete a two-page candidate");
  owner.certifyBlackPage(plan.pages[1].key);
  owner.certifyBlackPage(plan.pages[1].key);

  assert.equal(mock.textureWrites.length, 0);
  assert.equal(owner.telemetry().uploadedPages, 2);
  assert.equal(owner.publish().published, true);
  assert.equal(owner.visibleGeneration()?.generation, 5);
  assert.deepEqual([...owner.visibleGeneration()!.blackSlots], [0, 1]);
  assert.equal(owner.telemetry().blackPages, 2);
  assert.throws(() => owner.certifyBlackPage({ generation: 5, level: 0, coordinate: [2, 0, 0] }),
    /candidate generation/, "published candidates may no longer be mutated");
  owner.destroy();
});

test("an empty complete plan publishes without zero-sized textures", () => {
  installWebGpuUsageConstants();
  const mock = mockDevice();
  const owner = new WebGpuSvoTetrahedralRadiance(mock.device);
  const plan = planSvoNodeMipPyramid({ generation: 7, occupiedPages: [], levelCount: 4 });
  owner.beginGeneration(plan);
  assert.equal(owner.publish().published, true);
  assert.equal(mock.textures.length, 4);
  assert.ok(mock.textures.every((texture) => JSON.stringify(texture.descriptor.size) === "[1,1,1]"));
  owner.destroy();
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("radiance atlas owner uploads and publishes RGB9E5 textures on Metal", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU radiance-atlas validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const owner = new WebGpuSvoTetrahedralRadiance(device);
  try {
    device.pushErrorScope("validation");
    const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0]], levelCount: 1 });
    owner.beginGeneration(plan);
    owner.uploadInteriorPage(plan.pages[0].key, new Uint32Array(8 ** 3 * DIRECTIONS));
    assert.equal(owner.publish().published, true);
    await device.queue.onSubmittedWorkDone();
    assert.equal(await device.popErrorScope(), null);
    assert.equal(WEBGPU_SVO_TETRAHEDRAL_RADIANCE_LAYOUT.format, "rgb9e5ufloat");
  } finally {
    owner.destroy();
    device.destroy();
  }
});
