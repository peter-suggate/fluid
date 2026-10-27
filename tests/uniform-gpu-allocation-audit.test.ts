import assert from "node:assert/strict";
import test from "node:test";
import { auditUniformGPUAllocations } from "../tools/uniform-gpu-allocation-audit";

function fakeDevice() {
  const buffers: GPUBufferDescriptor[] = [];
  const textures: GPUTextureDescriptor[] = [];
  const device = {
    createBuffer(descriptor: GPUBufferDescriptor) {
      buffers.push(descriptor);
      return { label: descriptor.label ?? "", destroy() {} };
    },
    createTexture(descriptor: GPUTextureDescriptor) {
      textures.push(descriptor);
      return { label: descriptor.label ?? "", destroy() {} };
    },
  } as unknown as GPUDevice;
  return { device, buffers, textures };
}

test("allocation cap refuses a buffer before calling the GPU and credits destroyed resources once", () => {
  const raw = fakeDevice();
  const audit = auditUniformGPUAllocations(raw.device, 64);
  const first = audit.device.createBuffer({ label: "first", size: 48, usage: 0 });
  assert.throws(() => audit.device.createBuffer({ size: 32, usage: 0 }), /past the cap/);
  assert.equal(raw.buffers.length, 1);
  assert.equal(audit.snapshot().liveBytes, 48);
  first.destroy(); first.destroy();
  audit.device.createBuffer({ size: 64, usage: 0 });
  assert.equal(audit.snapshot().liveBytes, 64);
  assert.equal(audit.snapshot().peakBytes, 64);
});

test("texture byte rules and caps are checked before allocation, including iterable extents", () => {
  const raw = fakeDevice();
  const audit = auditUniformGPUAllocations(raw.device, 64);
  const descriptor = { format: "rgba32float", size: { width: 2, height: 2 }, usage: 0 } as const;
  assert.throws(() => audit.device.createTexture({ ...descriptor, size: [2, 2, 2] }), /past the cap/);
  assert.throws(() => audit.device.createTexture({ ...descriptor, mipLevelCount: 2 }), /exact byte rule/);
  assert.throws(() => audit.device.createTexture({ ...descriptor, format: "bc1-rgba-unorm" }), /exact byte rule/);
  assert.equal(raw.textures.length, 0);
  const texture = audit.device.createTexture(descriptor);
  assert.equal(audit.snapshot().liveBytes, 64);
  texture.destroy();
  audit.device.createTexture({ ...descriptor, size: [2, 2][Symbol.iterator]() });
  assert.deepEqual(raw.textures[1]!.size, { width: 2, height: 2, depthOrArrayLayers: 1 });
  assert.equal(audit.snapshot().liveBytes, 64);
});
