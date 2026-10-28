import { withUniformDevice } from "./helpers/uniform-geometric";
import assert from "node:assert/strict";
import test from "node:test";
import { sampledMaterialFilterWGSL } from "../lib/svo/features/materials/svo-sampled-material-filter.wgsl";
import { svoMaterialWGSL } from "../lib/svo/contracts/svo-material-abi";

test("sampled PBR reconstruction preserves linear fields, air boundaries and cell continuity on GPU",
  { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("sampled PBR interpolation", async device => {
    const patterns = [[1, 2], [1, 0], [0, 0], [1, 3]];
    const stride = 20 + 729 * 2, words = new Uint32Array(stride * patterns.length), floats = new Float32Array(words.buffer);
    for (let caseIndex = 0; caseIndex < patterns.length; caseIndex++) {
      const base = caseIndex * stride;
      floats[base + 3] = 1; words.set([16, 8, 8], base + 4); floats[base + 7] = 1;
      words.set([2, 1, 1], base + 8); words[base + 16] = 20;
      for (let z = 0; z < 9; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
        const at = base + 20 + 2 * (x + 9 * (y + 9 * z));
        floats[at] = y - 2.3;
        words[at + 1] = (patterns[caseIndex]![x & 1]! << 16) | 0x8080;
      }
    }
    const locations = [2, 2.25, 2.4999, 2.5, 2.5001, 2.75, 2.9999, 3, 3.0001, -1, 8, 17];
    const queries = new Float32Array(patterns.flatMap((_, i) => locations.flatMap(x => [x, 2.3, 2.7, i * stride])));
    const count = queries.length / 4;
    const data = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const points = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeBuffer(data, 0, words); device.queue.writeBuffer(points, 0, queries);
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
fn loadWord(at:u32)->u32{return data[at];}
${svoMaterialWGSL}
fn materialAt(id:u32)->SvoMaterialRecord{
  if(id==11u){return SvoMaterialRecord(vec4f(.9,.1,.2,1),vec4f(.2,0,.1,.2),vec4f(0,.3,1.5,0),vec4f(0),vec4f(0),vec4u(id,1,0,1));}
  if(id==12u){return SvoMaterialRecord(vec4f(.1,.3,.8,1),vec4f(0,.4,.1,.8),vec4f(1,.9,2,0),vec4f(0),vec4f(0),vec4u(id,1,0,1));}
  return SvoMaterialRecord();
}
fn validMaterial(m:SvoMaterialRecord,id:u32)->bool{return svoMaterialValid(m,id);}
${sampledMaterialFilterWGSL("loadWord", "materialAt", "validMaterial")}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&points)){return;}let p=points[id.x];
  let m=sampledMaterialPbr(u32(p.w),11u,p.xyz);
  result[id.x*3u]=vec4f(m.baseColor,m.roughness);
  result[id.x*3u+1u]=vec4f(m.emissive,m.metallic);
  result[id.x*3u+2u]=vec4f(m.specularF0.x,m.specularWeight,m.weight,0);
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [data, points, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const actual = new Float32Array(read.getMappedRange());
      const a = [.9, .1, .2, .2, .2, 0, .1, 0, .04, .3];
      const b = [.1, .3, .8, .8, 0, .4, .1, 1, 1 / 9, .9];
      for (let query = 0; query < count; query++) {
        const x = queries[query * 4]!, pattern = patterns[Math.floor(query / locations.length)]!;
        const cell = Math.floor(x), f = x - cell;
        const weights = [0, 0];
        if (x >= 0 && x < 8) for (let corner = 0; corner < 2; corner++) {
          const id = pattern[(cell + corner) & 1]!;
          if (id === 1 || id === 2) weights[id - 1]! += corner ? f : 1 - f;
        }
        const total = weights[0]! + weights[1]!;
        for (let channel = 0; channel < 10; channel++) {
          const expected = total ? (weights[0]! * a[channel]! + weights[1]! * b[channel]!) / total : 0;
          assert.ok(Math.abs(actual[query * 12 + channel]! - expected) < 2e-6,
            `query ${query}, channel ${channel}: linear optical field`);
        }
        assert.ok(Math.abs(actual[query * 12 + 10]! - total) < 2e-6, `query ${query}: valid weight`);
      }
      // The old material election jumps here. The reconstructed field crosses
      // the cell centre continuously and meets the next cell at the same value.
      assert.ok(Math.abs(actual[2 * 12]! - actual[4 * 12]!) < .0002);
      assert.ok(Math.abs(actual[6 * 12]! - actual[8 * 12]!) < 2e-6);
    } finally {
      if (read.mapState === "mapped") read.unmap();
      for (const buffer of [read, output, points, data]) buffer.destroy();
    }
  });
});
