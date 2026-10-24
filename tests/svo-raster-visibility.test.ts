import assert from "node:assert/strict";
import test from "node:test";
import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../lib/svo/contracts/svo-light-abi";
import { rasterVisibilityLayers, rasterVisibilityPackingWGSL, rasterVisibilityTilesWGSL } from "../lib/svo/features/lighting-visibility/raster-light-visibility";

test("visibility allocation covers the existing light sample budget, including GI", () => {
  const kinds = ["directional", "rectangleArea", "point", "spot", "sphereArea"] as const;
  const records = new Uint32Array(kinds.length * SVO_LIGHT_RECORD_WORDS);
  kinds.forEach((kind, i) => { records[i * SVO_LIGHT_RECORD_WORDS + 24] = SVO_LIGHT_KINDS[kind]; });
  assert.equal(rasterVisibilityLayers(records, 8, 2, false), 8);
  assert.equal(rasterVisibilityLayers(records, 8, 2, true), 5);
  assert.equal(rasterVisibilityLayers(records, 4, 2, false), 4);
  assert.equal(rasterVisibilityLayers(records.subarray(0, 56), 8, 2, false), 3);
  assert.equal(rasterVisibilityLayers(undefined, 8, 2, false), 1);
});

test("GPU visibility queue preserves thin receivers and clears after an empty frame", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  await withUniformDevice("visibility tiles", async device => {
    const geometry = device.createTexture({ size: [35, 19], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const queue = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const read = device.createBuffer({ size: 40, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const module = device.createShaderModule({ code: `
@group(1) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
const DRY_MISS=1e20;
struct Lighting { control:vec4u }
const rasterLighting=Lighting(vec4u(0,0,0,3));
fn rasterVisibilityGeometryAt(c:vec2i)->vec4f{return textureLoad(drySplitGeometryRead,c,0);}
${rasterVisibilityTilesWGSL}` });
      const empty = device.createBindGroupLayout({ entries: [] });
      const geometryLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } }] });
      const queueLayout = device.createBindGroupLayout({ entries: [{ binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [empty, geometryLayout, queueLayout] });
      const classify = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "rasterClassifyTiles" } });
      const finalize = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "rasterFinalizeTiles" } });
      const group0 = device.createBindGroup({ layout: empty, entries: [] });
      const group1 = device.createBindGroup({ layout: geometryLayout, entries: [{ binding: 0, resource: geometry.createView() }] });
      const group2 = device.createBindGroup({ layout: queueLayout, entries: [{ binding: 17, resource: { buffer: queue } }] });
      for (const occupied of [true, false]) {
        const pixels = new Float32Array(35 * 19 * 4);
        for (let i = 3; i < pixels.length; i += 4) pixels[i] = 1e20;
        // Even/even thin features never appear in the odd/odd reduced lattice.
        if (occupied) for (const [x, y] of [[0, 0], [16, 0], [34, 18]]) pixels.set([0, 1, 0, 2], (x! + 35 * y!) * 4);
        device.queue.writeTexture({ texture: geometry }, pixels, { bytesPerRow: 35 * 16 }, [35, 19]);
        const encoder = device.createCommandEncoder();encoder.clearBuffer(queue, 0, 4);
        const pass = encoder.beginComputePass();pass.setBindGroup(0, group0);pass.setBindGroup(1, group1);pass.setBindGroup(2, group2);
        pass.setPipeline(classify);pass.dispatchWorkgroups(3, 2);pass.setPipeline(finalize);pass.dispatchWorkgroups(1);pass.end();
        encoder.copyBufferToBuffer(queue, 0, read, 0, 40);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(read.getMappedRange()).slice();read.unmap();
        assert.deepEqual([...words.subarray(0, 4)], occupied ? [3, 3, 1, 1] : [0, 0, 1, 1]);
        if (occupied) assert.deepEqual([...words.subarray(4, 7)].sort(), [0, 1, 5]);
      }
    } finally { geometry.destroy();queue.destroy();read.destroy(); }
  });
});

test("GPU visibility packing retains coloured transmission within RGB10 error", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  await withUniformDevice("visibility RGB10", async device => {
    const output = device.createBuffer({ size: 1025 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `${rasterVisibilityPackingWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=1025u){return;}let x=f32(id.x)/1024.;
  result[id.x]=vec4f(rasterUnpackVisibility(rasterPackVisibility(vec3f(x,1.-x,x*x))),0);
}` });
      const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(17);pass.end();device.queue.submit([encoder.finish()]);
      const values = await readMixedBuffer(device, output);
      for (let i = 0; i <= 1024; i++) {
        const x = i / 1024;
        [x, 1 - x, x * x].forEach((expected, c) => assert.ok(Math.abs(values[i * 4 + c]! - expected) <= 1 / 2046 + 1e-7));
      }
    } finally { output.destroy(); }
  });
});

test("GPU exact-edge dispatch handles zero, partial groups, and more than 65535 groups", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { rasterExactQueueWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  await withUniformDevice("exact visibility queue", async device => {
    const queue = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const empty = device.createBindGroupLayout({ entries: [] });
      const layout = device.createBindGroupLayout({ entries: [{ binding: 20, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
      const module = device.createShaderModule({ code: rasterExactQueueWGSL });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [empty, empty, layout] }), compute: { module, entryPoint: "rasterExactFinalize" } });
      const emptyGroup = device.createBindGroup({ layout: empty, entries: [] });
      const group = device.createBindGroup({ layout, entries: [{ binding: 20, resource: { buffer: queue } }] });
      for (const [count, x, y] of [[0, 0, 1], [1, 1, 1], [64, 1, 1], [65, 2, 1], [65535 * 64 + 1, 65535, 2], [3474 * 1882, 65535, 2], [0, 0, 1]]) {
        device.queue.writeBuffer(queue, 0, new Uint32Array([count!]));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, emptyGroup); pass.setBindGroup(1, emptyGroup); pass.setBindGroup(2, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(queue, 0, read, 0, 16); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        assert.deepEqual([...new Uint32Array(read.getMappedRange())], [count, x, y, 1]); read.unmap();
      }
    } finally { queue.destroy(); read.destroy(); }
  });
});

test("GPU atlas permutation keeps representatives compact and exact receivers collision-free", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { rasterAtlasLayoutWGSL } = await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  await withUniformDevice("compact visibility atlas", async device => {
    const count = 36 * 20, size = count * 8;
    const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const dimensions = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    try {
      const module = device.createShaderModule({ code: `${rasterAtlasLayoutWGSL}
@group(0) @binding(0) var<uniform> dimensions:vec4i;
@group(0) @binding(1) var<storage,read_write> output:array<vec2i>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  let i=i32(id.x);if(i>=dimensions.x*dimensions.y){return;}
  output[i]=rasterCompactAtlasCoordinate(vec2i(i%dimensions.x,i/dimensions.x),dimensions.xy);
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: dimensions } }, { binding: 1, resource: { buffer: output } },
      ] });
      for (const [width, height] of [[1, 1], [1, 7], [6, 1], [2, 2], [5, 7], [35, 19], [36, 20]] as const) {
        device.queue.writeBuffer(dimensions, 0, new Int32Array([width, height, 0, 0]));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(width * height / 64)); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, size); device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        const values = new Int32Array(read.getMappedRange()); const seen = new Set<number>();
        for (let i = 0; i < width * height; i++) {
          const x = values[2 * i]!, y = values[2 * i + 1]!;
          assert.ok(x >= 0 && x < width && y >= 0 && y < height);
          seen.add(x + width * y);
          const sourceX = i % width, sourceY = Math.floor(i / width);
          if (sourceX === Math.min(Math.floor(sourceX / 2) * 2 + 1, width - 1)
            && sourceY === Math.min(Math.floor(sourceY / 2) * 2 + 1, height - 1)) {
            assert.ok(x < Math.ceil(width / 2) && y < Math.ceil(height / 2));
          }
        }
        assert.equal(seen.size, width * height); read.unmap();
      }
    } finally { output.destroy(); read.destroy(); dimensions.destroy(); }
  });
});
