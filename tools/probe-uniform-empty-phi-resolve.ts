/** Actual production phi-resolve shader: zero fine tiles, 1024 queued groups.
 * Compare its counted barrier with a uniform-buffer zero-fine early return. */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {uniformLatticeAddress} from '../lib/methods/uniform/uniform-lattice-address';
import {createUniformMixedLayout} from '../lib/methods/uniform/uniform-mixed-layout';
import {UniformMixedOwnership} from '../lib/methods/uniform/uniform-mixed-ownership';
import {UniformDetailStorage} from '../lib/methods/uniform/uniform-detail-fields';
import {UniformMixedPhiResolve} from '../lib/methods/uniform/uniform-mixed-phi-resolve';
import {GPUPassProfile} from './gpu-pass-profile';
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query']}),profile=new GPUPassProfile(raw);
const sources=new WeakMap<GPUShaderModule,string>();let original:GPUComputePipeline|undefined,optimized:GPUComputePipeline|undefined;
const proxy=new Proxy(profile.device,{get(target,key){
 if(key==='createShaderModule')return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const p=await target.createComputePipelineAsync(d);
  if(d.compute.entryPoint==='resolve'){
   const source=sources.get(d.compute.module)!;
   const count=source.includes('umCounts.work.')?'umCounts.work.x':'umCounts.x';
   const guard=` if(${count}==0u){return;}\n`;
   const baseline=source.replace(guard,''),candidate=baseline.replace(' if(umLane==0u){resolveJobs=',`${guard} if(umLane==0u){resolveJobs=`);assert.notEqual(candidate,baseline);
   const compile=async(code:string)=>{const module=target.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);return target.createComputePipelineAsync({...d,compute:{...d.compute,module}});};
   original=source===baseline?p:await compile(baseline);optimized=source===candidate?p:await compile(candidate);

  }return p;
 };
 const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
}});
const device=managedGPUDevice(proxy,{requireWorkerRealm:false}),errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const address=uniformLatticeAddress([32,32,32]),lattice={dimensions:[32,32,32] as [number,number,number],cellSize_m:[1,1,1] as [number,number,number],origin_m:{x:0,y:0,z:0}};
const layout=createUniformMixedLayout(lattice,[],4,undefined,address),ownership=new UniformMixedOwnership(device,layout,false),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
const field=storage.createField('empty resolve test','vertex','r32float',false,7.25),resolveStage=new UniformMixedPhiResolve(device,ownership);await resolveStage.initialize();const group=resolveStage.bind(field);assert.ok(original&&optimized);
const texture=storage.physical(field),bytesPerRow=Math.ceil(texture.width*4/256)*256,bytes=bytesPerRow*texture.height*texture.depthOrArrayLayers;
const read=raw.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
const capture=async()=>{const e=raw.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:read,bytesPerRow,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const w=new Uint32Array(read.getMappedRange().slice(0));read.unmap();return w;};
const before=await capture();
for(const groups of [512,1024]){
 const timing:number[][]=[[],[]];
 const run=(variant:number,batch=1)=>{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,group.group);p.setPipeline(variant?optimized!:original!);for(let i=0;i<batch;i++)p.dispatchWorkgroups(groups);p.end();device.queue.submit([e.finish()]);};
 for(let i=0;i<4;i++){run(0);run(1);}await device.queue.onSubmittedWorkDone();
 for(let round=0;round<12;round++)for(const v of round%2?[1,0]:[0,1]){profile.start();run(v,32);await device.queue.onSubmittedWorkDone();timing[v]!.push((await profile.finish())[0]!.ms/32);}
 const after=await capture();assert.deepEqual(after,before);const median=(v:number[])=>[...v].sort((a,b)=>a-b)[6];console.log(JSON.stringify({groups,fine:0,coarse:layout.coarseTiles.length,baseline_ms:median(timing[0]!),candidate_ms:median(timing[1]!),timing,unchanged:true,errors}));
}
// A real mixed tile exercises the nonempty path with nonconstant canonical
// values; compare every resulting logical vertex after restoring the upload.
const forced=new Uint8Array(address.boxTiles);forced[3+8*(3+8*3)]=1;
const mixed=createUniformMixedLayout(lattice,[],4,forced,address);ownership.update(mixed);
const volume=storage.createField('fixture volume','cell','r32float'),velocity=storage.createField('fixture velocity','face','rgba32float'),velocityScratch=storage.createField('fixture velocity scratch','face','rgba32float');
await storage.prepare(ownership);
const values=Float32Array.from({length:33**3},(_,i)=>{const x=i%33,y=Math.floor(i/33)%33,z=Math.floor(i/1089);return (x*x+7*y*y+13*z*z)*.125;});
storage.upload(field,values);storage.install(mixed,{volume,velocity,velocityScratch});
const mixedGroup=resolveStage.bind(field),results:Float32Array[]=[];
for(const pipeline of [original,optimized]){
 storage.upload(field,values);const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline!);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,mixedGroup.group);p.dispatchWorkgroups(1024);p.end();device.queue.submit([e.finish()]);results.push(await storage.read(field));
}
let changed=0;for(let i=0;i<values.length;i++)if(values[i]!==results[0]![i])changed++;
assert.ok(changed>0);assert.deepEqual(results[1],results[0]);console.log(JSON.stringify({scenario:'mixed',fine:mixed.fineTiles.length,changed,vertices:values.length,bitwiseEqual:true,errors}));
assert.deepEqual(errors,[]);read.destroy();storage.destroy();ownership.destroy();profile.destroy();device.destroy();
