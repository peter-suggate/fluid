import assert from "node:assert/strict";
import test from "node:test";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

test("large fluid device requests preserve the adapter-supported limits", () => {
  const limits = requiredFluidDeviceLimits({
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
  });
  assert.deepEqual(limits, {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
  });
});
