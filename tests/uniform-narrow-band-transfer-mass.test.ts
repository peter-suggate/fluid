import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandExpectedMassWGSL } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("graded mass-conserving seeds blend by mass, including partially covered deep faces", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("NB-FLIP transfer mass", async device => {
    // Four two-unit seeds and eight unit seeds represent the same full cell.
    // Two two-unit seeds cover half a deep face, not all of it. The legacy
    // retire mode still counts its four deep records as four units.
    const depths = [-2, -1.25, -0.5, 0, 0.25, 0.5];
    const source = `@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn main(){
${depths.map((depth, i) => `{
 let depth=${depth.toFixed(2)};
 result[${3*i}]=${narrowBandExpectedMassWGSL("depth", true, true)};
 result[${3*i+1}]=${narrowBandExpectedMassWGSL("depth", false, true)};
 result[${3*i+2}]=${narrowBandExpectedMassWGSL("depth", true, false)};
}`).join("\n")}
}`;
    const output = device.createBuffer({size: depths.length*12, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read = device.createBuffer({size: output.size, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      const module = device.createShaderModule({code: source});
      const pipeline = await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
      encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(read.getMappedRange()).slice();read.unmap();
      assert.deepEqual(Array.from(actual), [8,8,4, 8,8,4, 8,8,8, 4,4,4, 2,2,2, 1,1,1]);
      assert.equal(4/actual[0]!, 0.5, "half the physical mass gives half coverage");
    } finally { read.destroy();output.destroy(); }
  });
});

