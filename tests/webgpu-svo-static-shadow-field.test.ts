import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { planSvoStaticShadowField } from "../lib/svo-static-shadow-field";
import {
  WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT,
  WebGpuSvoStaticShadowField,
  packWebGpuSvoStaticShadowDirtySlots,
  packWebGpuSvoStaticShadowProofs,
  webGpuSvoStaticShadowBuildBindGroupLayoutEntries,
  webGpuSvoStaticShadowFieldConsumerWGSL,
  webGpuSvoStaticShadowFieldShader,
  webGpuSvoStaticShadowRenderBindGroupLayoutEntry,
} from "../lib/webgpu-svo-static-shadow-field";
import { getScenePreset } from "../lib/scenes";

function shadowPlan(generation = 5, lightRevision = 9) {
  const nodeMips = planSvoNodeMipPyramid({
    generation,
    occupiedPages: [[0, 0, 0], [1, 0, 0]],
    levelCount: 1,
    atlasPages: [4, 1, 1],
  });
  const lights = buildSvoSceneLights(getScenePreset("hose-tank").create(), {
    revision: lightRevision,
    maximumRecords: 3,
  });
  return { plan: planSvoStaticShadowField(nodeMips, lights), lights };
}

test("proof ABI packs shared slots, light channels, and exact revision stamps", () => {
  const { plan, lights } = shadowPlan();
  const words = packWebGpuSvoStaticShadowProofs(plan, [
    { page: plan.pages[1].key, lightId: lights.records[2].lightId, certificate: "visible" },
    { page: plan.pages[0].keyString, lightId: lights.records[0].lightId, certificate: "mixed" },
  ]);
  assert.deepEqual([...words], [
    plan.pages[1].slot, 2, 1, plan.sourceGeneration, plan.lightRevision,
    plan.pages[0].slot, 0, 3, plan.sourceGeneration, plan.lightRevision,
  ]);
  assert.equal(words.byteLength, 2 * WEBGPU_SVO_STATIC_SHADOW_FIELD_LAYOUT.proofJobBytes);
  assert.throws(() => packWebGpuSvoStaticShadowProofs(plan, [
    { page: plan.pages[0].key, lightId: lights.records[0].lightId, certificate: "visible" },
    { page: plan.pages[0].key, lightId: lights.records[0].lightId, certificate: "occluded" },
  ]), /Duplicate static-shadow proof job/);
});

test("dirty-page ABI is deterministic and rejects aliases", () => {
  const { plan } = shadowPlan();
  assert.deepEqual([...packWebGpuSvoStaticShadowDirtySlots(plan, [plan.pages[1].key, plan.pages[0].keyString])],
    [plan.pages[1].slot, plan.pages[0].slot]);
  assert.throws(() => packWebGpuSvoStaticShadowDirtySlots(plan, [plan.pages[0].key, plan.pages[0].keyString]),
    /Duplicate static-shadow dirty slot/);
});

test("GPU shader clears and invalidates before publishing revision-checked atomic certificates", () => {
  installWebGpuConstants();
  assert.deepEqual(webGpuSvoStaticShadowBuildBindGroupLayoutEntries().map(({ binding, buffer }) => [binding, buffer?.type]), [
    [0, "uniform"], [1, "storage"], [2, "read-only-storage"], [3, "read-only-storage"],
  ]);
  assert.deepEqual(webGpuSvoStaticShadowRenderBindGroupLayoutEntry(17), {
    binding: 17, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" },
  });
  assert.match(webGpuSvoStaticShadowFieldShader, /fn clearStaticShadowField/);
  assert.match(webGpuSvoStaticShadowFieldShader, /fn invalidateStaticShadowPages/);
  assert.match(webGpuSvoStaticShadowFieldShader, /fn publishStaticShadowProofs/);
  assert.match(webGpuSvoStaticShadowFieldShader,
    /proof\.sourceGeneration!=params\.sourceGeneration\|\|proof\.lightRevision!=params\.lightRevision/);
  assert.match(webGpuSvoStaticShadowFieldShader, /atomicStore\(&shadowPages\[slot\]\.visibleMask,0u\)/);
  assert.match(webGpuSvoStaticShadowFieldShader, /atomicOr\(&shadowPages\[proof\.slot\]\.visibleMask,lightBit\)/);
  assert.match(webGpuSvoStaticShadowFieldConsumerWGSL,
    /fieldSourceGeneration!=activeSourceGeneration\|\|fieldLightRevision!=activeLightRevision/);
  assert.match(webGpuSvoStaticShadowFieldConsumerWGSL, /return STATIC_SHADOW_UNKNOWN/);
  assert.doesNotMatch(webGpuSvoStaticShadowFieldShader + webGpuSvoStaticShadowFieldConsumerWGSL,
    /frameIndex|history|random|jitter|camera/);
});

interface MockBuffer {
  descriptor: GPUBufferDescriptor;
  destroyed: boolean;
  destroy(): void;
}

function installWebGpuConstants(): void {
  Object.assign(globalThis, {
    GPUBufferUsage: { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 },
    GPUShaderStage: { FRAGMENT: 2, COMPUTE: 4 },
  });
}

function mockGpu() {
  const buffers: MockBuffer[] = [];
  const writes: Array<{ buffer: GPUBuffer; words: number[] }> = [];
  const dispatches: Array<{ pipeline: string; count: number }> = [];
  const layout = {} as GPUBindGroupLayout;
  const device = {
    queue: {
      writeBuffer(buffer: GPUBuffer, _offset: number, data: Uint32Array) {
        writes.push({ buffer, words: [...data] });
      },
    },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const buffer: MockBuffer = {
        descriptor,
        destroyed: false,
        destroy() { buffer.destroyed = true; },
      };
      buffers.push(buffer);
      return buffer as unknown as GPUBuffer;
    },
    createBindGroupLayout: () => layout,
    createPipelineLayout: () => ({} as GPUPipelineLayout),
    createShaderModule: () => ({} as GPUShaderModule),
    createComputePipeline({ compute }: GPUComputePipelineDescriptor) {
      return { entryPoint: compute.entryPoint } as unknown as GPUComputePipeline;
    },
    createBindGroup: () => ({} as GPUBindGroup),
  } as unknown as GPUDevice;
  const encoder = {
    beginComputePass() {
      let pipeline = "";
      return {
        setBindGroup() {},
        setPipeline(value: GPUComputePipeline) { pipeline = (value as unknown as { entryPoint: string }).entryPoint; },
        dispatchWorkgroups(count: number) { dispatches.push({ pipeline, count }); },
        end() {},
      } as unknown as GPUComputePassEncoder;
    },
  } as unknown as GPUCommandEncoder;
  return { device, encoder, buffers, writes, dispatches };
}

test("GPU owner enforces full invalidation on revision changes and orders same-frame work", () => {
  installWebGpuConstants();
  const mock = mockGpu();
  const owner = new WebGpuSvoStaticShadowField(mock.device, {
    maximumAtlasCapacity: 4,
    maximumProofJobs: 8,
    maximumDirtyPages: 4,
  });
  const first = shadowPlan().plan;
  const view = owner.encodeUpdate(mock.encoder, {
    plan: first,
    fullInvalidate: true,
    proofs: [{ page: first.pages[0].key, lightId: first.lightChannels[0].lightId, certificate: "visible" }],
  });
  assert.deepEqual(mock.dispatches, [
    { pipeline: "clearStaticShadowField", count: 1 },
    { pipeline: "publishStaticShadowProofs", count: 1 },
  ]);
  assert.equal(view.binding.size, first.atlasCapacity * 8);
  assert.equal(mock.writes[0].words[0], first.sourceGeneration);
  assert.equal(mock.writes[0].words[1], first.lightRevision);

  mock.dispatches.length = 0;
  owner.encodeUpdate(mock.encoder, {
    plan: first,
    dirtyPages: [first.pages[0].key],
    proofs: [{ page: first.pages[0].key, lightId: first.lightChannels[0].lightId, certificate: "mixed" }],
  });
  assert.deepEqual(mock.dispatches, [
    { pipeline: "invalidateStaticShadowPages", count: 1 },
    { pipeline: "publishStaticShadowProofs", count: 1 },
  ]);
  const next = shadowPlan(6, 9).plan;
  assert.throws(() => owner.encodeUpdate(mock.encoder, { plan: next }), /requires full invalidation/);
  owner.destroy();
  assert.ok(mock.buffers.every(({ destroyed }) => destroyed));
  assert.throws(() => owner.encodeUpdate(mock.encoder, { plan: first }), /destroyed/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("static-shadow publication and same-frame fallback ABI compile on Dawn", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for Dawn validation",
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
  try {
    const shaderModule = device.createShaderModule({ code: webGpuSvoStaticShadowFieldShader });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    for (const entryPoint of ["clearStaticShadowField", "invalidateStaticShadowPages", "publishStaticShadowProofs"]) {
      device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
    }
    const consumer = device.createShaderModule({
      code: `${webGpuSvoStaticShadowFieldConsumerWGSL}
@compute @workgroup_size(1) fn validateConsumer(){_=resolveStaticShadowCertificate(vec2u(0u),0u,1u,1u,1u,1u);}`,
    });
    const consumerInfo = await consumer.getCompilationInfo();
    assert.deepEqual(consumerInfo.messages.filter(({ type }) => type === "error"), []);

    const { plan } = shadowPlan();
    const owner = new WebGpuSvoStaticShadowField(device, {
      maximumAtlasCapacity: plan.atlasCapacity,
      maximumProofJobs: 8,
      maximumDirtyPages: plan.atlasCapacity,
    });
    const readCertificates = async (encode: (encoder: GPUCommandEncoder) => void) => {
      const readback = device.createBuffer({
        size: plan.atlasCapacity * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encode(encoder);
      encoder.copyBufferToBuffer(owner.buffer, 0, readback, 0, plan.atlasCapacity * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = [...new Uint32Array(readback.getMappedRange().slice(0))];
      readback.unmap();
      readback.destroy();
      return words;
    };
    const page = plan.pages[0];
    const first = await readCertificates((encoder) => {
      owner.encodeUpdate(encoder, {
        plan,
        fullInvalidate: true,
        proofs: [
          { page: page.key, lightId: plan.lightChannels[0].lightId, certificate: "visible" },
          { page: page.key, lightId: plan.lightChannels[1].lightId, certificate: "occluded" },
          { page: page.key, lightId: plan.lightChannels[2].lightId, certificate: "mixed" },
        ],
      });
    });
    assert.deepEqual(first.slice(page.slot * 2, page.slot * 2 + 2), [0b101, 0b110]);

    const second = await readCertificates((encoder) => {
      owner.encodeUpdate(encoder, {
        plan,
        dirtyPages: [page.key],
        proofs: [{ page: page.key, lightId: plan.lightChannels[0].lightId, certificate: "visible" }],
      });
    });
    assert.deepEqual(second.slice(page.slot * 2, page.slot * 2 + 2), [0b001, 0],
      "dirty-page invalidation leaves unpublished lights at 00 for exact same-frame fallback");
    owner.destroy();
  } finally {
    device.destroy();
  }
});
