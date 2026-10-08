import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { ParticleOverlay, type GPUFluidParticleSource, type ParticleOverlayFrame } from "../lib/core/webgpu-particle-overlay";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("particle records draw as depth-ordered spheres over exactly the live records", { timeout: 90000 }, async () => {
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
    async function draw(source: GPUFluidParticleSource, sceneDepth?: GPUTextureView, opacity?: number, view?: ParticleOverlayFrame["view"]) {
      overlay.setSource(source);
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
      assert.equal(overlay.encode(encoder, target.createView(), sceneDepth, { ...frame, opacity, view }), true);
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

    // The Motion view: energy is opacity and whiteness, each sample a soft footprint three radii wide that hides no other.
    const moving = await draw({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 }, undefined, undefined, "motion");
    const lit = moving(32, 32);
    assert.ok(lit[3]! > 130 && lit[3]! < 170 && lit[0]! >= lit[3]! - 4 && lit[2]! >= lit[3]! - 4, `one sample at full scale is white and translucent: ${lit}`);
    assert.ok(moving(44, 32)[3]! > 0 && moving(44, 32)[3]! < lit[3]! - 40, `and fades out past its sphere's rim: ${moving(44, 32)}`);
    const slowInFront = records(8, [
      [0.5, 0.5, 0.3, 1, 0, 0, 0, 0],
      [0.5, 0.5, 0.5, 1, 3, 0, 0, 0],
      [-0.3, 0.5, 0.5, 1, 0, 0, 0, 0],
    ]);
    const veiled = await draw({ ...base, buffer: slowInFront, strideFloats: 8, capacity: 3 }, undefined, undefined, "motion");
    assert.ok(veiled(32, 32)[3]! > 130 && veiled(32, 32)[0]! >= veiled(32, 32)[3]! - 4, `a slow sample in front does not hide the fast one behind it: ${veiled(32, 32)}`);
    const still = veiled(6, 32)[3]!;
    assert.ok(still > 0 && still < 8, `a sample at rest is nearly, not fully, transparent: ${veiled(6, 32)}`);
    assert.equal((await draw({ ...base, buffer: slowInFront, strideFloats: 8, capacity: 3 }))(6, 32)[3], 255, "the speed ramp draws the same sample as an opaque sphere");

    // A sample no grid carries is measured by kinetic energy less what falling gives: the same 3 m/s lights one
    // thrown sideways or upward and leaves one that is only dropping as dark as one at rest.
    const energy = async (velocity: readonly number[], gravity_m_s2?: ParticleOverlayFrame["gravity_m_s2"], carried?: readonly number[]) => {
      // Carried by a grid: its depth, the grid's velocity and the ballistic flag follow the sample's own velocity.
      overlay.setSource(carried
        ? { ...base, buffer: records(12, [[0.5, 0.5, 0.5, 1, ...velocity, ...carried]]), strideFloats: 12, capacity: 1,
          grid: { velocityFloat: 8, depthFloat: 7, ballisticFloat: 11, surfaceDepth: 2 } }
        : { ...base, buffer: records(8, [[0.5, 0.5, 0.5, 1, ...velocity, 0]]), strideFloats: 8, capacity: 1 });
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
      assert.equal(overlay.encode(encoder, target.createView(), undefined, { ...frame, view: "motion", gravity_m_s2 }), true);
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [size, size]); device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const alpha = new Uint8Array(readback.getMappedRange())[32 * 256 + 32 * 4 + 3]!; readback.unmap();
      return alpha;
    };
    const down = [0, -9.81, 0] as const, rest = await energy([0, 0, 0], down);
    assert.equal(await energy([0, -3, 0], down), rest, "a sample falling with gravity is as dark as one at rest");
    assert.equal(await energy([0, 3, 0], down), lit[3], "one rising against it is at full scale");
    assert.equal(await energy([3, -4, 0], down), lit[3], "a falling sample keeps the energy of its sideways motion");
    assert.equal(await energy([0, -3, 0]), lit[3], "without gravity every direction counts");
    const half = await energy([Math.SQRT2, 0, 0], down);
    assert.ok(half > rest + 30 && half < lit[3]! - 30, `the scale is energy, half of it at 1.41 m/s: ${half} between ${rest} and ${lit[3]}`);

    // A sample a grid carries is measured against it: the energy of its velocity about the grid's, full scale at 0.5 m/s,
    // counted in full through two position units of depth and not at all by four.
    assert.equal(await energy([3, 0, 0], down, [-1, 3, 0, 0, 0]), rest, "liquid moving with the grid is dark however fast it goes");
    assert.equal(await energy([0, -3, 0], down, [-1, 0, -3, 0, 0]), rest, "falling with it too");
    assert.equal(await energy([3.5, 0, 0], down, [-1, 3, 0, 0, 0]), lit[3], "half a metre a second against the grid is full scale");
    assert.equal(await energy([0, 0, 0], down, [-1, 0.5, 0, 0, 0]), lit[3], "and so is a sample at rest in liquid that moves");
    assert.equal(await energy([3.5, 0, 0], down, [-2, 3, 0, 0, 0]), lit[3], "the surface's share of the band counts in full");
    const deeper = await energy([3.5, 0, 0], down, [-3, 3, 0, 0, 0]);
    assert.ok(deeper > rest + 30 && deeper < lit[3]! - 30, `below it a sample counts for less: ${deeper} between ${rest} and ${lit[3]}`);
    assert.equal(await energy([3.5, 0, 0], down, [-4, 3, 0, 0, 0]), rest, "and at twice that depth for nothing");
    assert.equal(await energy([0, -3, 0], down, [6, 0, 0, 0, 1]), rest, "a ballistic sample has no grid: falling, it is dark");
    assert.equal(await energy([3, 0, 0], down, [6, 0, 0, 0, 1]), lit[3], "thrown sideways, it is at full scale");

    // A solver reorders its records every step, so the picture must not depend on the order: three overlapping
    // spheres at three speeds, drawn in both orders.
    const crowd = [[0.5, 0.5, 0.3, 1, 1, 0, 0, 0], [0.55, 0.5, 0.5, 1, 3, 0, 0, 0], [0.45, 0.55, 0.7, 1, 0.5, 0, 0, 0]];
    const forwards = await draw({ ...base, buffer: records(8, crowd), strideFloats: 8, capacity: 3 }, undefined, undefined, "motion");
    const backwards = await draw({ ...base, buffer: records(8, [...crowd].reverse()), strideFloats: 8, capacity: 3 }, undefined, undefined, "motion");
    let reordered = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) forwards(x, y).forEach((channel, i) => { reordered = Math.max(reordered, Math.abs(channel - backwards(x, y)[i]!)); });
    assert.ok(reordered <= 1, `the Motion view is the same picture in either record order: channels differ by ${reordered}`);

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

    // Simple water in front of the spheres: its front interface a plane facing the camera, no exit behind it.
    const interfaceTarget = () => device!.createTexture({ size: [size, size], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const waterFront = interfaceTarget(), waterNormal = interfaceTarget(), noInterface = interfaceTarget();
    const interfaceAt = (texture: GPUTexture, value: readonly number[]) => {
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", storeOp: "store", clearValue: value }] }).end();
      device!.queue.submit([encoder.finish()]);
    };
    interfaceAt(waterNormal, [0, 0, -1, 1]); interfaceAt(noInterface, [0, 0, 0, 0]);
    const water = { frontPosition: waterFront.createView(), frontNormal: waterNormal.createView(), backPosition: noInterface.createView(),
      rearFrontPosition: noInterface.createView(), rearBackPosition: noInterface.createView() };
    const under = async (surfaceZ: number) => {
      // Every texel names the point on the camera's axis: the liquid in front of the sphere at the centre pixel.
      interfaceAt(waterFront, [0, 0.5, surfaceZ, 1]);
      overlay.setSource({ ...base, buffer: fixed, strideFloats: 8, capacity: 3 });
      const encoder = device!.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
      assert.equal(overlay.encode(encoder, target.createView(), undefined, { ...frame, water }), true);
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [size, size]); device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const pixels = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
      return [...pixels.subarray(32 * 256 + 32 * 4, 32 * 256 + 32 * 4 + 4)];
    };
    // The fast sphere's centre is 1.8 m from the camera and the murk length of a 1 m tank is 9 cm.
    assert.deepEqual(await under(0.5), centre, "a sphere in front of the surface is drawn as it is without water");
    const shallow = await under(-0.5), deep = await under(-0.9);
    assert.ok(shallow[3] === 255 && shallow[2]! > shallow[0]! && shallow[0]! < centre[0]!, `liquid in front turns the orange sphere toward the water's blue: ${shallow}`);
    assert.ok(deep[3] === 255 && deep[0]! < shallow[0]!, `more liquid hides more of it: ${deep} after ${shallow}`);
    assert.deepEqual(await under(-1.5), [0, 0, 0, 0], "a sphere the body hides outright is not drawn");
    assert.deepEqual(errors, []); overlay.destroy();
  } finally { device?.destroy(); }
});
