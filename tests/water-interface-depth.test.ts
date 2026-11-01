import assert from "node:assert/strict";
import test from "node:test";
import { WATER_INTERFACE_DEPTH } from "../lib/core/water-interface-depth";
import { surfaceRasterShader } from "../lib/core/webgpu-water-pipeline";
import { SecondaryParticleRenderPipeline } from "../lib/core/webgpu-secondary-particles";
import { SVO_DRY_SCENE_REVERSED_Z_NEAR_M } from "../lib/svo/features/shading/program";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/svo/features/primary-visibility/webgpu-svo-gbuffer-targets";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("water optical interfaces use the SVO infinite reversed-Z convention", () => {
  assert.equal(WATER_INTERFACE_DEPTH.near_m, SVO_DRY_SCENE_REVERSED_Z_NEAR_M);
  assert.equal(WATER_INTERFACE_DEPTH.format, SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat);
  assert.equal(WATER_INTERFACE_DEPTH.compare, SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare);
  assert.equal(WATER_INTERFACE_DEPTH.clearValue, SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthClearValue);
});

test("water and spray retain complete, ordered coverage beyond the old 50 metre limit", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("distant water interface depth", async device => {
    const size = 16;
    const uniform = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Camera at the origin looking +Z, aperture=1; no peeling or film.
    device.queue.writeBuffer(uniform, 0, new Float32Array([
      size, size, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      48, 24, 48, 0, 0, 0, 0, 0, 192, 96, 192, 0, 0, 0, 0, 0,
    ]));
    const vertices = device.createBuffer({ size: 18 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const particles = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const position = device.createTexture({ size: [size, size], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const normal = device.createTexture({ size: [size, size], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const firstBack = device.createTexture({ size: [size, size], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const depth = device.createTexture({ size: [size, size], format: WATER_INTERFACE_DEPTH.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const readback = device.createBuffer({ size: size * size * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const spray = new SecondaryParticleRenderPipeline(device, uniform);
    try {
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ] });
      const peelLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      ] });
      const module = device.createShaderModule({ code: surfaceRasterShader });
      const pipeline = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout, peelLayout] }),
        vertex: { module, entryPoint: "surfaceVertex" },
        fragment: { module, entryPoint: "surfaceFragment", targets: [{ format: "rgba32float" }, { format: "rgba16float" }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { format: WATER_INTERFACE_DEPTH.format, depthWriteEnabled: true, depthCompare: WATER_INTERFACE_DEPTH.compare },
      });
      const group = device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: vertices } },
      ] });
      const peel = device.createBindGroup({ layout: peelLayout, entries: [{ binding: 0, resource: firstBack.createView() }] });
      await spray.initialize();
      spray.setSource({ buffer: particles, capacity: 1, strideBytes: 64 });
      const render = async (distance: number, sprayDepth?: number) => {
        // Far/near/far draw order catches either compare direction and z ties.
        const points = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
        const data: number[] = [];
        for (const z of [distance + 1, distance, distance + 2]) {
          for (let i = 0; i < points.length; i += 2) data.push(points[i]! * z, points[i + 1]! * z, z, 1, 0, 0, -1, 0);
        }
        device.queue.writeBuffer(vertices, 0, new Float32Array(data));
        if (sprayDepth !== undefined) device.queue.writeBuffer(particles, 0,
          new Float32Array([0, 0, sprayDepth, 10, 0, 0, 0, 0, 0, 0, -1, 10, 1, 0, 1, 1]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [position, normal].map(texture => ({
          view: texture.createView(), loadOp: "clear" as const, storeOp: "store" as const, clearValue: { r: 0, g: 0, b: 0, a: 0 },
        })), depthStencilAttachment: { view: depth.createView(), depthClearValue: WATER_INTERFACE_DEPTH.clearValue,
          depthLoadOp: "clear", depthStoreOp: "store" } });
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.setBindGroup(1, peel); pass.draw(18);
        if (sprayDepth !== undefined) spray.encodeOpticalInterface(pass, "front");
        pass.end();
        encoder.copyTextureToBuffer({ texture: position }, { buffer: readback, bytesPerRow: size * 16 }, [size, size]);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
        return result;
      };
      for (const distance of [1, 49, 50, 51, 60, 100, 1000]) {
        const result = await render(distance);
        for (let pixel = 0; pixel < size * size; pixel++) {
          assert.equal(result[4 * pixel + 3], 1, `coverage at ${distance}m, pixel ${pixel}`);
          assert.ok(Math.abs(result[4 * pixel + 2]! - distance) < distance * 1e-5, `nearest surface at ${distance}m, pixel ${pixel}`);
        }
      }
      const center = 4 * (8 * size + 8);
      const frontSpray = await render(80, 60);
      assert.ok(frontSpray[center + 2]! < 60, "spray beyond 50m must occlude water behind it");
      const backSpray = await render(60, 80);
      assert.ok(Math.abs(backSpray[center + 2]! - 60) < .001, "spray behind the water must not replace its front interface");
    } finally {
      for (const resource of [uniform, vertices, particles, position, normal, firstBack, depth, readback]) resource.destroy();
    }
  });
});
