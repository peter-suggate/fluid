import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {initialUniformPageDomain,UNIFORM_PAGE_DOMAIN_BASE,uniformPageDomainWGSL} from "../lib/methods/uniform/uniform-page-domain";
import {UniformTexturePages} from "../lib/methods/uniform/uniform-texture-pages";
import {UniformPageDomainPublication} from "../lib/methods/uniform/uniform-page-domain-publication";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("GPU accepted catalogue controls reordered, reduced and empty page dispatch and overlay",{timeout:60000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform accepted page publication");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const domain=initialUniformPageDomain([65,33,17],32,true),base=UNIFORM_PAGE_DOMAIN_BASE;
  const accepted=device.createBuffer({size:base*4+domain.words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const visited=device.createBuffer({size:65*33*17*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:visited.size+128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const publication=new UniformPageDomainPublication(device,domain,accepted);await publication.initialize();
  const module=device.createShaderModule({code:`
   @group(0) @binding(0) var<storage,read> activeRegion:array<u32>;
   @group(0) @binding(1) var<storage,read_write> visited:array<atomic<u32>>;
   fn dims()->vec3i{return vec3i(65,33,17);}
   ${uniformPageDomainWGSL(domain)}
   @compute @workgroup_size(4,4,4) fn visit(@builtin(global_invocation_id)g:vec3u){
    let q=pageDomainCell(g);if(any(q<vec3i(0))||any(q>=dims())){return;}
    atomicAdd(&visited[u32(q.x+65*(q.y+33*q.z))],1u);
   }`});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"visit"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:accepted}},{binding:1,resource:{buffer:visited}}]});
  const pages=new UniformTexturePages(device);
  const field=pages.createTexture({size:[65,33,17],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  pages.upload(field,new Float32Array(65*33*17).fill(2));
  const reductions=device.createBuffer({size:40,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const auditModule=device.createShaderModule({code:pages.shader(`
   @group(0) @binding(0) var inputField:texture_3d<f32>;
   @group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,10>;
   @group(0) @binding(29) var<storage,read> activeRegion:array<u32>;
   fn dims()->vec3i{return vec3i(65,33,17);}
   ${uniformPageDomainWGSL(domain)}
   @compute @workgroup_size(4,4,4) fn audit(@builtin(global_invocation_id)g:vec3u){
    if(any(g>=vec3u(65,33,17))){return;}
    let value=textureLoad(inputField,vec3i(g),0).x;
    if(value!=2.0){atomicAdd(&reductions[0],1u);}
   }`,new Map([[0,field]]),true)});
  const auditPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:auditModule,entryPoint:"audit"}});
  const auditGroup=pages.createBindGroup({layout:auditPipeline.getBindGroupLayout(0),entries:[
   {binding:0,resource:pages.view(field)},{binding:9,resource:{buffer:reductions}},{binding:29,resource:{buffer:accepted}},
  ]});
  try{
   for(const count of [domain.count,2,0,domain.count]){
    const words=domain.words.slice();words[8]=count;
    // Deliberately stale header must be replaced by the GPU publisher.
    words.fill(777,0,7);device.queue.writeBuffer(accepted,base*4,words);
    const e=device.createCommandEncoder();publication.encode(e);e.clearBuffer(visited);
    const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroupsIndirect(publication.dispatch,0);p.end();
    e.copyBufferToBuffer(accepted,base*4,output,0,32);
    e.copyBufferToBuffer(publication.view,0,output,32,publication.view.size);
    e.copyBufferToBuffer(visited,0,output,128,visited.size);device.queue.submit([e.finish()]);
    await output.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(output.getMappedRange().slice(0));output.unmap();
    assert.deepEqual(Array.from(actual.slice(0,3)),[count*8,8,8]);
    assert.deepEqual(Array.from(actual.slice(4,7)),[count*9,9,9]);
    assert.equal(actual[12],count);
    const expected=new Uint32Array(65*33*17);const visible=new Set<number>();
    for(let i=0;i<count;i++){
     const slot=words[16+16*domain.capacity+i]!,at=16+16*slot;
     const q=Array.from(words.slice(at,at+3));visible.add(q[0]!+3*q[1]!);
     for(let z=q[2]!*32;z<Math.min(q[2]!*32+32,17);z++)for(let y=q[1]!*32;y<Math.min(q[1]!*32+32,33);y++)for(let x=q[0]!*32;x<Math.min(q[0]!*32+32,65);x++)expected[x+65*(y+33*z)]=1;
    }
    assert.deepEqual(actual.slice(32),expected);
    for(let i=0;i<domain.capacity;i++)assert.equal(actual[16+i],Number(visible.has(i)),"retired overlay flags cleared");
    const auditEncoder=device.createCommandEncoder();auditEncoder.clearBuffer(reductions);
    const ap=auditEncoder.beginComputePass();ap.setPipeline(auditPipeline);ap.setBindGroup(0,auditGroup);ap.dispatchWorkgroups(17,9,5);ap.end();
    auditEncoder.copyBufferToBuffer(reductions,0,output,0,40);device.queue.submit([auditEncoder.finish()]);
    await output.mapAsync(GPUMapMode.READ);const audit=new Uint32Array(output.getMappedRange().slice(0,40));output.unmap();
    const missing=expected.reduce((sum,n)=>sum+Number(n===0),0);
    assert.equal(audit[0],0,"auditing leaves field values unchanged");
    assert.equal(audit[8],missing?1:0,"field binding mask identifies absent-page reads");
    assert.equal(audit[9],missing,"every absent-page tap is counted, and restored membership clears the fault");
   }
   assert.deepEqual(errors,[]);
  }finally{publication.destroy();accepted.destroy();visited.destroy();output.destroy();pages.destroy();field.destroy();reductions.destroy();}
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
