import assert from "node:assert/strict";
import test from "node:test";
import { waterInterfaceSamplingWGSL } from "../lib/core/water-interface-sampling.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("water interface geometry rejects offscreen exits and preserves covered edge positions", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("water interface capture bounds", async device => {
    const queries = [
      [.25, .25], [.5, .5], [.25, .75], [0, 0], [1, 1],
      [-.001, .5], [1.001, .5], [.5, -.001], [.5, 1.001],
      // Measured projected exits from a live 1200×800 FLIP lake frame.
      [600.54 / 1200, 1118.55 / 800], [602.36 / 1200, 1006.56 / 800],
    ];
    const texture = device.createTexture({ size: [2, 2], format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const points = device.createBuffer({ size: queries.length * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: queries.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeTexture({ texture }, new Float32Array([2, 4, 6, 1, 4, 8, 12, 1, 0, 0, 0, 0, 8, 16, 24, 1]), { bytesPerRow: 32 }, [2, 2]);
      device.queue.writeBuffer(points, 0, new Float32Array(queries.flat()));
      const module = device.createShaderModule({ code: `
        @group(0) @binding(0) var positions:texture_2d<f32>;
        @group(0) @binding(1) var<storage,read> points:array<vec2f>;
        @group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
        ${waterInterfaceSamplingWGSL}
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
          result[id.x]=safePositionSample(positions,points[id.x]);
        }` });
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout, entries: [
        { binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: points } }, { binding: 2, resource: { buffer: output } },
      ] }));
      pass.dispatchWorkgroups(queries.length); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(read.getMappedRange());
      const expected = [[2, 4, 6, 1], [14 / 3, 28 / 3, 14, .75], [0, 0, 0, 0], [2, 4, 6, 1], [8, 16, 24, 1], ...queries.slice(5).map(() => [0, 0, 0, 0])].flat();
      expected.forEach((v, i) => assert.ok(Math.abs(actual[i]! - v) < 1e-6, `query ${Math.floor(i / 4)}, channel ${i % 4}: ${actual[i]} vs ${v}`));
    } finally {
      if (read.mapState === "mapped") read.unmap();
      for (const buffer of [points, output, read]) buffer.destroy(); texture.destroy();
    }
  });
});
