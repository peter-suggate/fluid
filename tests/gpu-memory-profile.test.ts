import assert from "node:assert/strict";
import test from "node:test";
import { GPUMemoryProfile, gpuTexturePayloadBytes } from "../tools/gpu-memory-profile";

test("memory profile counts texture mips, array layers, compressed blocks and shadow targets", () => {
  const base = { width: 8, height: 8, depthOrArrayLayers: 4, dimension: "3d" as const, mipLevelCount: 3, sampleCount: 1 };
  assert.equal(gpuTexturePayloadBytes({ ...base, format: "rgba32float" }), (256 + 32 + 4) * 16);
  assert.equal(gpuTexturePayloadBytes({ ...base, dimension: "2d", format: "bgra8unorm" }), (64 + 16 + 4) * 4 * 4);
  assert.equal(gpuTexturePayloadBytes({ ...base, mipLevelCount: 1, sampleCount: 4, format: "depth32float" }), 256 * 4 * 4);
  assert.equal(gpuTexturePayloadBytes({ ...base, width: 7, height: 5, depthOrArrayLayers: 1, mipLevelCount: 1, format: "bc1-rgba-unorm" }), 32);
  assert.equal(gpuTexturePayloadBytes({ ...base, width: 7, height: 5, depthOrArrayLayers: 1, mipLevelCount: 1, format: "astc-5x4-unorm" }), 64);
});

test("memory profile distinguishes live allocation, resize peak and explicit destruction", () => {
  const raw = { createBuffer: (d: GPUBufferDescriptor) => ({ size: d.size, label: d.label ?? "", destroy() {} }) } as unknown as GPUDevice;
  const profile = new GPUMemoryProfile(raw);
  const old = profile.device.createBuffer({ size: 128, usage: 0, label: "old" });
  const next = profile.device.createBuffer({ size: 256, usage: 0, label: "next" });
  old.destroy(); old.destroy();
  const snapshot = profile.snapshot();
  assert.equal(snapshot.bytes, 256);
  assert.equal(snapshot.peakBytes, 384);
  assert.equal(snapshot.bufferBytes, 256);
  assert.deepEqual(snapshot.resources.map(r => r.label), ["next"]);
  next.destroy();
  assert.equal(profile.allocatedBytes, 0);
});
