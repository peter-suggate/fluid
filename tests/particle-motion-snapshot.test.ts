import assert from "node:assert/strict";
import test from "node:test";
import { GPUFluidParticleMotion } from "../lib/core/webgpu-particle-motion";
import { ParticleOverlay } from "../lib/core/webgpu-particle-overlay";
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

test("motion renders calm overlap transparent and full energy opaque despite scratch sentinels", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("particle motion opacity", async device => {
    const capacity = 128, records = new Float32Array(capacity * 12);
    for (let i = 0; i < capacity; i++) records.set([0, 0, 0, 1, 2, 0, 0, -1, 3e38, 0, 0, 0], i * 12);
    const particles = device.createBuffer({ size: records.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const energy = device.createBuffer({ size: capacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(particles, 0, records);
    const target = device.createTexture({ size: [32, 32], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 256 * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const overlay = new ParticleOverlay(device, "rgba8unorm");
    try {
      await overlay.initialize();
      overlay.setSource({ buffer: particles, motionEnergy: energy, strideFloats: 12, capacity,
        positionScale_m: [1, 1, 1], radius_m: 0.1,
        grid: { velocityFloat: 8, depthFloat: 7, ballisticFloat: 11, surfaceDepth: 2 } });
      const alphas: number[] = [];
      for (const value of [0, 0.015625, 0.25]) {
        device.queue.writeBuffer(energy, 0, new Float32Array(capacity).fill(value));
        const encoder = device.createCommandEncoder();
        encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
        assert.ok(overlay.encode(encoder, target.createView(), undefined, {
          camera: { position_m: [0, 0, 2], forward: [0, 0, -1], right: [1, 0, 0], up: [0, 1, 0], tanHalfFov: 0.5, aspect: 1 },
          viewportWidth: 32, viewportHeight: 32, container_m: [0, 0, 0], depthNear_m: 0.01, view: "motion",
        }));
        encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [32, 32]);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        alphas.push(new Uint8Array(readback.getMappedRange())[16 * 256 + 16 * 4 + 3]! / 255);
        readback.unmap();
      }
      assert.equal(alphas[0], 0, "calm samples cannot reveal the occupied tile");
      assert.ok(alphas[1]! < 0.02, `weak overlap must stay faint: ${alphas[1]}`);
      assert.ok(alphas[2]! > 0.99, `full-energy liquid must remain opaque: ${alphas[2]}`);
    } finally {
      overlay.destroy(); particles.destroy(); energy.destroy(); target.destroy(); readback.destroy();
    }
  });
});
