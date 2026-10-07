import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { ParticleOverlay, type GPUFluidParticleSource, type ParticleOverlayFrame } from "../lib/core/webgpu-particle-overlay";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("particle records draw as depth-ordered spheres over exactly the live records", { timeout: 90000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "particle-overlay");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const overlay = new ParticleOverlay(device, "rgba8unorm"); await overlay.initialize();
    const size = 64, near = 0.05;
    const target = device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const readback = device.createBuffer({ size: size * 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // A 1 m tank seen square-on from 2 m: 32 pixels per metre at its centre plane.
    const frame: ParticleOverlayFrame = {
      camera: { position_m: [0, 0.5, -2], forward: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0], tanHalfFov: 0.5, aspect: 1 },
      viewportWidth: size, viewportHeight: size, container_m: [1, 1, 1], depthNear_m: near,
    };
    const records = (stride: number, rows: readonly (readonly number[])[]) => {
      const data = new Float32Array(stride * rows.length);
      rows.forEach((row, i) => data.set(row, i * stride));
      const buffer = device!.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device!.queue.writeBuffer(buffer, 0, data); return buffer;
    };
    async function draw(source: GPUFluidParticleSource, sceneDepth?: GPUTextureView, opacity?: number) {
      overlay.setSource(source);
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
      assert.equal(overlay.encode(encoder, target.createView(), sceneDepth, { ...frame, opacity }), true);
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [size, size]); device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const pixels = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
      return (x: number, y: number) => [...pixels.subarray(y * 256 + x * 4, y * 256 + x * 4 + 4)];
    }
    const base = { positionScale_m: [1, 1, 1], radius_m: 0.2 } as const;

    // Tank-local metres, eight floats: a fast sphere nearer the camera, a slow
    // one behind it on the same ray and drawn later, and a retired lane beside them.
    const fixed = records(8, [
      [0.5, 0.5, 0.3, 1, 3, 0, 0, 0],
      [0.5, 0.5, 0.5, 1, 0, 0, 0, 0],
      [0.2, 0.5, 0.5, 0, 0, 0, 0, 0],
    ]);
    const pixel = await draw({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 });
    const centre = pixel(32, 32);
    assert.equal(centre[3], 255, "an opaque sphere covers the centre");
    assert.ok(centre[0]! > centre[2]!, `the nearer, fast sphere wins the depth test whatever the draw order: ${centre}`);
    assert.deepEqual(pixel(22, 32), [0, 0, 0, 0], "a retired record draws nothing");
    assert.deepEqual(pixel(4, 4), [0, 0, 0, 0], "nothing is drawn away from the spheres");
    const edge = pixel(36, 32);
    assert.ok(edge[3] === 255 && edge.slice(0, 3).join() !== centre.slice(0, 3).join(), "the sphere is shaded across its face, not flat");
    assert.ok((await draw({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 }, undefined, 0.5))(32, 32)[3]! < 200, "layer opacity reaches the spheres");

    // Lattice-cell positions in a wider record, and a live prefix counted on the GPU.
    const band = records(12, [
      [2, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [8, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]);
    const count = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const counted: GPUFluidParticleSource = { buffer: band, strideFloats: 12, capacity: 2, positionScale_m: [0.1, 0.1, 0.1], radius_m: 0.05, liveCount: { buffer: count, byteOffset: 0 } };
    device.queue.writeBuffer(count, 0, new Uint32Array([1]));
    let live = await draw(counted);
    assert.ok(live(22, 32)[3] === 255 && live(22, 32)[2]! > live(22, 32)[0]!, "the first live record is a slow sphere at its cell position");
    assert.deepEqual(live(41, 32), [0, 0, 0, 0], "a record past the live count is not drawn");
    device.queue.writeBuffer(count, 0, new Uint32Array([2]));
    live = await draw(counted);
    assert.equal(live(41, 32)[3], 255, "the draw follows the GPU count");
    device.queue.writeBuffer(count, 0, new Uint32Array([1000]));
    live = await draw(counted);
    assert.ok(live(22, 32)[3] === 255 && live(41, 32)[3] === 255, "an overrun count draws the buffer's records and no more");

    // Opaque scenery 1 m from the camera hides the spheres 2 m away; scenery behind them does not.
    const scenery = device.createTexture({ size: [size, size], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const sceneryAt = (distance_m: number) => {
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: scenery.createView(), depthClearValue: near / distance_m, depthLoadOp: "clear", depthStoreOp: "store" } }).end();
      device!.queue.submit([encoder.finish()]);
    };
    sceneryAt(1);
    assert.deepEqual((await draw({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 }, scenery.createView()))(32, 32), [0, 0, 0, 0], "scenery in front hides the spheres");
    sceneryAt(3);
    assert.equal((await draw({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 }, scenery.createView()))(32, 32)[3], 255, "scenery behind leaves them drawn");
    assert.deepEqual(errors, []); overlay.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
