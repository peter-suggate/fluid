import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandParticleStride, narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU FLIP state packing preserves age bits and ballistic mode, including negative zero", {skip: !process.env.WEBGPU_NODE_MODULE}, async () => {
 await withUniformDevice("FLIP packed state", async device => {
  // Zero, subnormal, fractional frame increments, arbitrary mantissas, large age.
  const ages = new Uint32Array([0,1,0x3c888889,0x3eaaaaab,0x3ff12345,0x41200000]);
  const output=device.createBuffer({size:ages.length*2*narrowBandParticleStride(false)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try {
   const module=device.createShaderModule({code:`${narrowBandParticleWGSL(false)}
@group(0) @binding(0) var<storage,read_write> records:array<Particle>;
@compute @workgroup_size(1) fn main(){
 let ages=array<u32,${ages.length}>(${[...ages].map(x=>`${x}u`).join(",")});
 for(var i=0u;i<${ages.length}u;i++){for(var mode=0u;mode<2u;mode++){
  var p=Particle();p.before.w=nbParticleState(bitcast<f32>(ages[i]),f32(mode));
  // A storage round trip, and both helpers, must preserve the physical state.
  records[2u*i+mode]=p;
  records[2u*i+mode].velocity.x=nbParticleMode(records[2u*i+mode]);
  records[2u*i+mode].velocity.y=nbParticleAge(records[2u*i+mode]);
 }}
}`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);
   p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}));p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output),bits=new Uint32Array(values.buffer);
   for(let i=0;i<ages.length;i++)for(let mode=0;mode<2;mode++){
    const at=(2*i+mode)*12;assert.equal(values[at+4],mode);assert.equal(bits[at+5],ages[i]);
    assert.equal(bits[at+11],(ages[i]!|(mode<<31))>>>0);
   }
  } finally {output.destroy();}
 });
});
