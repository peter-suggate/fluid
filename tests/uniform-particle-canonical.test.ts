import assert from 'node:assert/strict';
import test from 'node:test';
import { particleCanonicalWGSL } from '../lib/methods/uniform/uniform-particle-canonical.wgsl';
import { narrowBandParticleWGSL, narrowBandParticleStride } from '../lib/methods/uniform/uniform-narrow-band-apic.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';

test('canonical GPU cell sort preserves FLIP/APIC records and ignores append order',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('canonical particle cells',async device=>{
  for(const apic of [false,true]){
   const stride=narrowBandParticleStride(apic),counts=[0,1,8,16,65,257,4096],ranges:number[]=[];
   let total=0;for(const n of counts){ranges.push(total,n);total+=n;}
   const records=Array.from({length:total},(_,i)=>{
    const f=new Float32Array(stride);for(let k=0;k<stride;k++)f[k]=k<3?Math.floor(i/7)%11:(i*17+k*31)%113-56;
    // Coincident positions and duplicate complete records exercise ties.
    return new Uint32Array(f.buffer);
   });
   const less=(a:Uint32Array,b:Uint32Array)=>{for(let k=0;k<stride;k++){const d=a[k]!-b[k]!;if(d)return d;}return 0;};
   const expected=new Uint32Array(total*stride);
   for(let c=0;c<counts.length;c++){const start=ranges[2*c]!,n=counts[c]!;records.slice(start,start+n).sort(less).forEach((r,j)=>expected.set(r,(start+j)*stride));}
   const buffer=device.createBuffer({size:expected.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const cells=device.createBuffer({size:4*ranges.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(cells,0,new Uint32Array(ranges));
   const read=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const module=device.createShaderModule({code:`${narrowBandParticleWGSL(apic)}
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(1) var<storage,read> cells:array<vec2u>;
${particleCanonicalWGSL(apic)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){if(gid.x<arrayLength(&cells)){let c=cells[gid.x];nbCanonicalCell(c.x,c.y);}}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>`${m.lineNum}: ${m.message}`),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}},{binding:1,resource:{buffer:cells}}]});
   try{
    for(const reverse of [false,true]){
     const input=new Uint32Array(expected.length);
     for(let c=0;c<counts.length;c++){const start=ranges[2*c]!,n=counts[c]!;for(let j=0;j<n;j++)input.set(records[start+(reverse?n-1-j:j)]!,(start+j)*stride);}
     device.queue.writeBuffer(buffer,0,input);const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(buffer,0,read,0,read.size);device.queue.submit([e.finish()]);
     await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange()).slice();read.unmap();assert.deepEqual(result,expected,`${apic?'APIC':'FLIP'}, reversed=${reverse}`);
    }
   }finally{read.destroy();cells.destroy();buffer.destroy();}
  }
 });
});
