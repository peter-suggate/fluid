import assert from "node:assert/strict";
import test from "node:test";
import {
  FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE,
  FLUID_REDUCTION_BYTES_PER_LANE,
  requiredFluidDeviceLimits,
  supportsFluidM1MaxReduction,
} from "../lib/webgpu-device-limits";

test("large fluid device requests preserve the adapter-supported limits", () => {
  const limits = requiredFluidDeviceLimits({
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
    maxColorAttachmentBytesPerSample: 128,
  });
  assert.deepEqual(limits, {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
    maxColorAttachmentBytesPerSample: FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE,
  });
});

test("colour-attachment request never exceeds the adapter and never drops below the default", () => {
  const base = {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 1,
    maxBufferSize: 1,
    maxTextureDimension3D: 1,
  };
  assert.equal(requiredFluidDeviceLimits({ ...base, maxColorAttachmentBytesPerSample: 32 })
    .maxColorAttachmentBytesPerSample, 32);
  assert.equal(requiredFluidDeviceLimits({ ...base, maxColorAttachmentBytesPerSample: 40 })
    .maxColorAttachmentBytesPerSample, 40);
  assert.equal(requiredFluidDeviceLimits({ ...base, maxColorAttachmentBytesPerSample: 512 })
    .maxColorAttachmentBytesPerSample, FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE);
});

test("sole 128-lane target reduction is fail-closed on device limits", () => {
  assert.equal(FLUID_REDUCTION_BYTES_PER_LANE, 32);
  assert.equal(supportsFluidM1MaxReduction({
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupStorageSize: 32 * 1024,
  }), true);
  assert.equal(supportsFluidM1MaxReduction({
    maxComputeInvocationsPerWorkgroup: 96,
    maxComputeWorkgroupStorageSize: 32 * 1024,
  }), false);
  assert.equal(supportsFluidM1MaxReduction({
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupStorageSize: 3 * 1024,
  }), false);
});
