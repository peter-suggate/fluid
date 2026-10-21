import assert from "node:assert/strict";
import test from "node:test";
import { GPUFluidParticleMotion } from "../lib/core/webgpu-particle-motion";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("display energy survives FLIP scratch reuse without modifying particles", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("particle display energy", async device => {
    // Calm translation, a weak fluctuation, and full-energy liquid. The
    // unused fourth slot must not inherit data from beyond the live prefix.
    const records = new Float32Array(4 * 12);
    for (let i = 0; i < 4; i++) {
      records.set([0, 0, 0, 1, 2 + [0, 0.125, 0.5, 10][i]!, 0, 0, -1, 2, 0, 0, 0], i * 12);
    }
    const particles = device.createBuffer({ size: records.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const count = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(particles, 0, records);
    device.queue.writeBuffer(count, 0, new Uint32Array([3]));
    const motion = new GPUFluidParticleMotion(device, particles, count, 12, 4);
    try {
      await motion.initialize();
      const encoder = device.createCommandEncoder();
      motion.capture(encoder);
      device.queue.submit([encoder.finish()]);
      const unchanged = await readMixedBuffer(device, particles);
      assert.deepEqual(unchanged, records, "display capture cannot change the simulation");
      // The solver replaces the grid mean with deltas / unsupported sentinels.
      for (let i = 0; i < 3; i++) device.queue.writeBuffer(particles, 4 * (i * 12 + 8), new Float32Array([3e38, -5, 9]));
      const energy = await readMixedBuffer(device, motion.energy);
      assert.deepEqual([...energy], [0, 0.015625, 0.25, 0]);
    } finally {
      motion.destroy(); particles.destroy(); count.destroy();
    }
  });
});
