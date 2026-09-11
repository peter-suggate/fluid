import assert from 'node:assert/strict';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock} from '../lib/harness/webgpu-smoke-isolation';
import {compileCM12InterpolationPatch, evaluateCM12InterpolationPatch, packCM12InterpolationPatches, CM12_INTERPOLATION_PATCH_WGSL, type PatchPoint} from '../lib/methods/adaptive-mass/sparse-cm12-interpolation-patch';
import {patchFixtures} from './sparse-cm12-interpolation-patch.test';

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)('compiled patch GPU weights retain affine moments, positivity and collapsed-node identity', async()=>{
 await acquireWebGPUExclusiveLock('dawn-test','compiled-interpolation-patches');
 let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const patches=patchFixtures.map(n=>{const p=compileCM12InterpolationPatch(n);assert.ok(p);return p;});
  const queries:number[][]=[];
  patchFixtures.forEach((nodes,index)=>{
   for(const node of nodes) queries.push([...node,index]);
   for(const t of [1e-5,.1,.5,.9]) for(const u of [.2,.5,.8]) {
    const param=[t,u,.37];
    const w=nodes.map((_,c)=>param.reduce((a,v,k)=>a*(c&(1<<k)?v:1-v),1));
    queries.push(...[[0,1,2].map(k=>Math.fround(nodes.reduce((a,n,i)=>a+w[i]!*n[k]!,0))).concat(index)]);
   }
   queries.push([1000,1000,1000,index]);
  });
  const shader=device.createShaderModule({code:CM12_INTERPOLATION_PATCH_WGSL+`
@group(0)@binding(0)var<storage,read>patches:array<CM12InterpolationPatch>;
@group(0)@binding(1)var<storage,read>queries:array<vec4f>;
@group(0)@binding(2)var<storage,read_write>output:array<vec4f>;
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3u){
 if(id.x>=arrayLength(&queries)){return;}
 let q=queries[id.x];let w=cm12EvaluateInterpolationPatch(patches[u32(q.w)],q.xyz);
 output[3u*id.x]=vec4f(w.weights[0],w.weights[1],w.weights[2],w.weights[3]);
 output[3u*id.x+1u]=vec4f(w.weights[4],w.weights[5],w.weights[6],w.weights[7]);
 output[3u*id.x+2u]=vec4f(f32(w.inside));
}`});
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
  const descriptor=device.createBuffer({size:64*patches.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const input=device.createBuffer({size:16*queries.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:48*queries.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  device.queue.writeBuffer(descriptor,0,packCM12InterpolationPatches(patches));device.queue.writeBuffer(input,0,new Float32Array(queries.flat()));
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[descriptor,input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(queries.length/64));pass.end();
  enc.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);
  const values=new Float32Array(read.getMappedRange());let maximumMomentError=0,maximumWeightError=0;
  queries.forEach((query,i)=>{
   const index=query[3]!,point=query.slice(0,3) as unknown as PatchPoint;
   const expected=evaluateCM12InterpolationPatch(patches[index]!,point);
   assert.equal(values[12*i+8],Number(expected!==null),`query ${i}: containment`);
   if(!expected)return;
   const weights=Array.from(values.subarray(12*i,12*i+8));assert.ok(weights.every(w=>w>=0));
   assert.ok(Math.abs(weights.reduce((a,b)=>a+b)-1)<1e-6);
   weights.forEach((w,k)=>{maximumWeightError=Math.max(maximumWeightError,Math.abs(w-expected[k]!));});
   for(let k=0;k<3;k++) maximumMomentError=Math.max(maximumMomentError,Math.abs(weights.reduce((a,w,c)=>a+w*patchFixtures[index]![c]![k]!,0)-point[k]!));
  });
  assert.ok(maximumWeightError<2e-6,`weight error ${maximumWeightError}`);
  assert.ok(maximumMomentError<1e-5,`moment error ${maximumMomentError}`);
  console.log(JSON.stringify({patches:patches.length,queries:queries.length,maximumWeightError,maximumMomentError,descriptorBytes:descriptor.size}));
  read.unmap();for(const b of [descriptor,input,output,read])b.destroy();
 } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
