import assert from 'node:assert/strict';
import test from 'node:test';
import { meshVolumeWGSL } from '../lib/core/imported-assets/mesh-volume.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';

test('native GPU material sampling preserves weighted votes, ties, empty samples and normals',
  { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('native material sampling', async device => {
    const cases = [[0,0,0,0,0,0,0,0], [7,7,7,7,7,7,7,7], [1,2,3,4,5,6,7,8],
      [0,0,0,0,1,1,1,1], [1,1,2,2,1,1,2,2], [9,9,9,2,2,3,4,0]];
    let random = 42;
    for (let n = 0; n < 32; n++) cases.push(Array.from({ length: 8 }, () => {
      random = (Math.imul(random, 1664525) + 1013904223) >>> 0; return random % 5;
    }));
    const stride = 18 + 729 * 2, words = new Uint32Array(stride * cases.length), floats = new Float32Array(words.buffer);
    for (let n = 0; n < cases.length; n++) {
      const base = n * stride;
      floats[base + 3] = 1; words.set([8,8,8], base + 4); floats[base + 7] = 10;
      words.set([1,1,1], base + 8); words[base + 16] = 18;
      for (let z = 0; z < 9; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
        const index = base + 18 + 2 * (x + 9 * (y + 9 * z));
        floats[index] = x + y + z - 7;
        words[index + 1] = (cases[n]![(x & 1) + 2 * (y & 1) + 4 * (z & 1)]! << 16) | 0x8080;
      }
    }
    const positions = [[.5,.5,.5], [0,0,0], [1,1,1], [.25,.5,.75], [.75,.25,.5]];
    const count = cases.length * positions.length, queries = new Float32Array(count * 4);
    for (let n = 0; n < count; n++) queries.set([...positions[n % positions.length]!.map(v => v + 2), Math.floor(n / positions.length) * stride], n * 4);
    const data = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const points = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeBuffer(data, 0, words); device.queue.writeBuffer(points, 0, queries);
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4u>;
fn loadWord(at:u32)->u32{return data[at];}
${meshVolumeWGSL('sampleVolume', 'loadWord')}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=arrayLength(&points)){return;}let p=points[id.x];let v=sampleVolume(u32(p.w),p.xyz);
 output[id.x]=vec4u(v.material,bitcast<u32>(v.distance),bitcast<u32>(v.normal.z),bitcast<u32>(sampleVolumeDistance(u32(p.w),p.xyz)));
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [data,points,output].map((buffer,binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0,group); pass.dispatchWorkgroups(Math.ceil(count/64)); pass.end();
      encoder.copyBufferToBuffer(output,0,read,0,output.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const actual = new Uint32Array(read.getMappedRange()), values = new Float32Array(actual.buffer);
      for (let n = 0; n < count; n++) {
        const p = positions[n % positions.length]!, materials = cases[Math.floor(n / positions.length)]!;
        // At a cell boundary the sampler moves into the next cell; our fixture
        // alternates IDs by coordinate parity, so account for that permutation.
        const ids = materials.map((_,corner) => materials[corner ^ p.reduce((mask,v,a) => mask | (v === 1 ? 1 << a : 0),0)]!);
        const f = p.map(v => v === 1 ? 0 : v), votes = new Map<number,number>();
        for (let i = 0; i < 8; i++) if (ids[i]) votes.set(ids[i]!, (votes.get(ids[i]!) ?? 0) + f.reduce((w,v,a) => w * (i & (1 << a) ? v : 1-v),1));
        let expected = 0, best = -1;
        for (const id of ids) if (id && votes.get(id)! > best) { expected = id; best = votes.get(id)!; }
        assert.equal(actual[n*4],expected,`case ${n}: material vote`);
        assert.equal(values[n*4+1],p.reduce((a,b)=>a+b,-1),`case ${n}: distance`);
        assert.equal(actual[n*4+1],actual[n*4+3],`case ${n}: scalar and surface distances agree`);
        const hasWeight = [...votes.values()].some(v=>v>0);
        assert.ok(hasWeight ? values[n*4+2]! > .999 : values[n*4+2] === 0,`case ${n}: source normal`);
      }
    } finally { if (read.mapState === 'mapped') read.unmap(); read.destroy(); output.destroy(); points.destroy(); data.destroy(); }
  });
});

test('native cached scalar probes preserve bits across cell, volume, empty-brick and exterior boundaries',
  { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('cached imported distance probes', async device => {
    const stride = 20 + 729 * 2, words = new Uint32Array(stride * 2), floats = new Float32Array(words.buffer);
    for (let asset = 0; asset < 2; asset++) {
      const base = asset * stride;
      floats[base + 3] = 1; words.set([16, 8, 8], base + 4); floats[base + 7] = 10;
      words.set([2, 1, 1], base + 8); words[base + 16] = 20; floats[base + 19] = 12;
      for (let z = 0; z < 9; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
        floats[base + 20 + 2 * (x + 9 * (y + 9 * z))] = Math.sin(x * .31 + y * .71) * .17 + z * .23 + asset * 7;
      }
    }
    const locations = [[2.5, 3.5, 4.5], [2.501, 3.5, 4.5], [2.499, 3.5, 4.5], [2.5, 3.501, 4.5],
      [2.5, 3.499, 4.5], [2.5, 3.5, 4.501], [2.5, 3.5, 4.499],
      [2.9999, 3.5, 4.5], [3.0001, 3.5, 4.5], [3.0002, 3.5, 4.5],
      [8, 3, 4], [9, 3, 4], [-1, 3, 4], [17, 3, 4], [7.9999, 3, 4], [7.9998, 3, 4]];
    const queries = new Float32Array([0, 1, 0].flatMap(asset => locations.flatMap(p => [...p, asset * stride])));
    const count = queries.length / 4;
    const data = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const points = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeBuffer(data, 0, words); device.queue.writeBuffer(points, 0, queries);
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4u>;
var<private> referenceLoads:u32;var<private> cachedLoads:u32;
fn referenceLoad(at:u32)->u32{referenceLoads+=1u;return data[at];}
fn cachedLoad(at:u32)->u32{cachedLoads+=1u;return data[at];}
${meshVolumeWGSL('referenceVolume', 'referenceLoad')}
${meshVolumeWGSL('cachedVolume', 'cachedLoad', true)}
@compute @workgroup_size(1) fn main(){
 for(var i=0u;i<arrayLength(&points);i+=1u){
   referenceLoads=0u;cachedLoads=0u;let p=points[i];
   let reference=referenceVolumeDistance(u32(p.w),p.xyz);let cached=cachedVolumeDistance(u32(p.w),p.xyz);
   output[i]=vec4u(bitcast<u32>(reference),bitcast<u32>(cached),referenceLoads,cachedLoads);
 }
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [data, points, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, read.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const result = new Uint32Array(read.getMappedRange());
      for (let i = 0; i < count; i++) {
        assert.ok(Math.abs(result[i * 4]! - result[i * 4 + 1]!) <= 4, `query ${i}: scalar interpolation stays within four float32 ULPs`);
        if (i % locations.length >= 1 && i % locations.length <= 6) {
          assert.equal(result[i * 4 + 2]! - result[i * 4 + 3]!, 8, 'a neighboring gradient probe reuses all eight scalar corners');
        }
      }
    } finally { if (read.mapState === 'mapped') read.unmap(); read.destroy(); output.destroy(); points.destroy(); data.destroy(); }
  });
});
