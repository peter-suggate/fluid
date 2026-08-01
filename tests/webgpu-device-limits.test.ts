import assert from "node:assert/strict";
import test from "node:test";
import { SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY } from "../lib/svo-static-node-mips";
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
    maxTextureDimension2D: 16384,
    maxTextureDimension3D: 2048,
    maxSampledTexturesPerShaderStage: 16,
    maxColorAttachmentBytesPerSample: 128,
  });
  assert.deepEqual(limits, {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension2D: 16384,
    maxTextureDimension3D: 2048,
    maxColorAttachmentBytesPerSample: FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE,
    maxSampledTexturesPerShaderStage: 16,
  });
});

/**
 * The static node-mip directory is one texture row per page, so this limit is
 * the opacity pyramid's page ceiling. A scene that needs more than the granted
 * value does not get a coarser pyramid, it gets none — and cone lighting then
 * falls back to exact traversal for every shadow and GI ray in the frame.
 * Measured on hose-tank (10361 pages needed, M1 Max/Metal, 1280x720, raster
 * primary): 20 ms/frame at the adapter's 16384, 305 ms/frame at the 8192
 * WebGPU default.
 */
test("node-mip page ceiling requests the adapter value, not the WebGPU default", () => {
  const base = {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 1,
    maxBufferSize: 1,
    maxTextureDimension3D: 1,
    maxSampledTexturesPerShaderStage: 16,
    maxColorAttachmentBytesPerSample: 128,
  };
  assert.equal(
    requiredFluidDeviceLimits({ ...base, maxTextureDimension2D: 16384 }).maxTextureDimension2D,
    16384,
  );
  assert.ok(
    Number(requiredFluidDeviceLimits({ ...base, maxTextureDimension2D: 16384 }).maxTextureDimension2D)
      > SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY,
    "an adapter advertising more than the default page ceiling must have it requested",
  );
  // Adapters that really do cap at the default are still served their own value.
  assert.equal(
    requiredFluidDeviceLimits({ ...base, maxTextureDimension2D: SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY })
      .maxTextureDimension2D,
    SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY,
  );
});

test("colour-attachment request never exceeds the adapter and never drops below the default", () => {
  const base = {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 1,
    maxBufferSize: 1,
    maxTextureDimension2D: 8192,
    maxTextureDimension3D: 1,
    maxSampledTexturesPerShaderStage: 16,
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
