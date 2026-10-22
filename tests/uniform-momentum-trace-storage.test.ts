import assert from "node:assert/strict";
import test from "node:test";
import { UniformMixedMomentum, uniformMomentumTraceBinding } from "../lib/methods/uniform/uniform-mixed-momentum";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";

test("momentum borrowing preserves the full trace range and rejects stale reservations", () => {
  const buffer = { size: 256 + 1280 * 16 } as GPUBuffer;
  const binding = uniformMomentumTraceBinding(8, { buffer, offset: 256 });
  assert.equal(binding.buffer, buffer);
  assert.equal(binding.offset, 256);
  assert.equal(binding.size, 1280 * 8);
  assert.equal(uniformMomentumTraceBinding(16, { buffer, offset: 256 }).size, 1280 * 16);
  assert.throws(() => uniformMomentumTraceBinding(17, { buffer, offset: 256 }), /every admitted h owner/);
  assert.throws(() => uniformMomentumTraceBinding(16, binding), /every admitted h owner/,
    "Growing capacity must rebuild the prior bounded binding even when the underlying buffer fits");
  assert.throws(() => uniformMomentumTraceBinding(16, { buffer, offset: 512, size: 1280 * 16 }), /every admitted h owner/);
  assert.throws(() => uniformMomentumTraceBinding(0, undefined), /every admitted h owner/);
});

test("shared momentum never allocates or destroys the caller's trace arena", () => {
  Object.assign(globalThis, { GPUShaderStage: { COMPUTE: 4 }, GPUBufferUsage: { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 } });
  const buffers: Array<{ label: string; size: number; destroyed: number; destroy(): void }> = [];
  const device = {
    features: new Set(),
    createBindGroupLayout: () => ({}),
    createBuffer: (d: GPUBufferDescriptor) => {
      const buffer = { label: d.label ?? "", size: d.size, destroyed: 0, destroy() { this.destroyed++; } };
      buffers.push(buffer); return buffer;
    },
  } as unknown as GPUDevice;
  const ownership = { capacity: { fineTiles: 262144, tiles: 262144 } } as unknown as UniformMixedOwnership;
  const heat = {} as GPUBuffer;
  const owned = new UniformMixedMomentum(device, ownership, heat, false, true);
  const ownedBuffers = [...buffers];
  const shared = new UniformMixedMomentum(device, ownership, heat, false, true, undefined, undefined, "software", "shared");
  const sharedBuffers = buffers.slice(ownedBuffers.length);
  assert.equal(owned.allocatedBytes - shared.allocatedBytes, 335544320);
  assert.equal(ownedBuffers.filter(b => b.label.includes("trace list")).length, 1);
  assert.equal(sharedBuffers.filter(b => b.label.includes("trace list")).length, 0);
  const callerArena = { size: 1 << 30, destroyed: 0, destroy() { this.destroyed++; } };
  assert.equal(uniformMomentumTraceBinding(262144, { buffer: callerArena as unknown as GPUBuffer }).buffer, callerArena);
  shared.destroy();
  assert.equal(callerArena.destroyed, 0);
  assert.ok(sharedBuffers.every(b => b.destroyed === 1));
  assert.ok(ownedBuffers.every(b => b.destroyed === 0));
  owned.destroy();
  assert.ok(ownedBuffers.every(b => b.destroyed === 1));
});
