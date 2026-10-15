/** Counter diagnostic: ordinary stationary solver frames, replacing only phi
 * resolve shader addressing. No warm repeated-empty-dispatch microbenchmark.
 * All replacements are counterfactuals valid only for this closed resident page. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createPassEncoderIsolationScratch,isolateComputePassEncoders} from '../lib/harness/webgpu-pass-encoder-isolation';
import {experimentalImmutableWindow} from './experimental-immutable-window';
import {PhiResolveCountProbe} from './phi-resolve-count-probe';
const flag=(name:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3);
const root=flag('root')??'/tmp/fluid-history-main-b1c6e2c5';
const seconds=Number(flag('seconds')??2);assert.ok(seconds>0);
type Layer='baseline'|'immutable'|'header'|'direct';let active:Layer='baseline',frame=0;
const sources=new WeakMap<GPUShaderModule,string>(),variants=new WeakMap<GPUComputePipeline,Record<Layer,GPUComputePipeline>>();
let rewritePages=0;
function specialize(source:string,layer:Layer):string{
 if(layer==='baseline')return source;
 if(layer==='immutable')return experimentalImmutableWindow(source,[0,0,0,1,1,1,63,0xffffffff]);
 let changed=0;
 source=source.replace(/fn (\w*PageWord)\(page:vec3i\)->u32\{\n let world=page\+[\s\S]*?\n\}/g,(body,name:string)=>{
  changed++;if(layer==='direct')return `fn ${name}(page:vec3i)->u32{return select(0xffffffffu,1u,all(page==vec3i(0)));}`;
  const raw=body.match(/let raw=([^;]+);/)![1];return `fn ${name}(page:vec3i)->u32{let world=page;let e=bitcast<vec3u>(world)&vec3u(63u);let raw=${raw};return select(0xffffffffu,raw,all(page==vec3i(0)));}`;
 });
 source=source.replace(/fn (\w*BlockWord)\(b:u32\)->u32\{\n let raw=[\s\S]*?\n\}/g,(body,name:string)=>layer==='direct'?`fn ${name}(b:u32)->u32{return select(0xffffffffu,0u,b==1u);}`:body.replace(/let q=[^;]+;/,'let q=vec3u(raw&63u,(raw>>6u)&63u,(raw>>12u)&63u);'));
 assert.ok(changed,'actual dynamic window addressing is present');rewritePages+=changed;return source;
}
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal','enable-dawn-features=use_user_defined_labels_in_backend']).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups' as const]:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const scratch=createPassEncoderIsolationScratch(raw);
const errors:string[]=[];raw.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const proxy=new Proxy(raw,{get(target,key){
 if(key==='createShaderModule')return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
 if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
  const p=await target.createComputePipelineAsync(d),source=sources.get(d.compute.module);
  if(source?.includes('umResolveLattice')&&['resolve','resolveListed'].includes(d.compute.entryPoint??'')){
   const result:Record<Layer,GPUComputePipeline>={baseline:p,immutable:p,header:p,direct:p};
   for(const layer of ['immutable','header','direct'] as const){
    const code=specialize(source,layer),module=target.createShaderModule({label:`QA phi ${layer}`,code});
    const diagnostics=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.deepEqual(diagnostics,[]);
    result[layer]=await target.createComputePipelineAsync({...d,label:`QA phi ${layer} ${d.compute.entryPoint}`,compute:{...d.compute,module}});
   }
   variants.set(p,result);
  }return p;
 };
 if(key==='createCommandEncoder')return(d?:GPUCommandEncoderDescriptor)=>{
  const e=isolateComputePassEncoders(target.createCommandEncoder(d),scratch);return new Proxy(e,{get(et,ek){
   if(ek==='beginComputePass')return(pd?:GPUComputePassDescriptor)=>{
    const phi=pd?.label==='Uniform mixed phi resolve';const p=et.beginComputePass(phi?{...pd,label:`QA empty phi ${active}`} : pd);
    if(!phi)return p;
    return new Proxy(p,{get(pt,pk){if(pk==='setPipeline')return(value:GPUComputePipeline)=>{pt.setPipeline(variants.get(value)?.[active]??value);};const v=Reflect.get(pt,pk,pt);return typeof v==='function'?v.bind(pt):v;}});
   };
   const v=Reflect.get(et,ek,et);return typeof v==='function'?v.bind(et):v;
  }});
 };
 const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
}});
// Count probe stays outside label rewriting so it sees production pass names.
const counts=new PhiResolveCountProbe(proxy);
const load=(file:string)=>import(pathToFileURL(resolve(root,file)).href);
const [{uniformNarrowBandMethod:method},{resolveMethodValues},{managedGPUDevice}]=await Promise.all([load('lib/methods/uniform/uniform-narrow-band-method.ts'),load('lib/core/method-contract.ts'),load('lib/core/gpu-compilation-manager.ts')]);
const scene=JSON.parse(readFileSync(resolve('artifacts/dam-history-2026-10-15/fluid-stationary-32.json'),'utf8'));scene.duration_s=120;
const values=resolveMethodValues(method,'balanced',{...method.appDefaults,timeStep:'scene'});
const device=managedGPUDevice(counts.device,{requireWorkerRealm:false});
const solver=await method.createSolverAsync(device,scene,'balanced',values,undefined,()=>{});
const advance=async()=>{frame++;assert.ok(solver.advanceTo(frame/60,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);};
try{
 for(let i=0;i<12;i++)await advance();
 const epochs=[];
 for(const layer of ['baseline','immutable','header','direct','direct','header','immutable','baseline'] as const){
  active=layer;counts.enabled=true;await advance();counts.enabled=false;
  const jobs=await counts.read();assert.ok(jobs.length);assert.ok(jobs.every(p=>p.jobs===0),'all phi passes actually empty');
  const start=performance.now(),first=frame;while(performance.now()-start<1000*seconds)await advance();
  const epoch={layer,first,last:frame,frames:frame-first,wall_ms:performance.now()-start,jobs};epochs.push(epoch);console.log(JSON.stringify({phase:'epoch',...epoch}));
 }
 assert.deepEqual(errors,[]);console.log(JSON.stringify({phase:'result',root,rewritePages,frames:frame,epochs,errors}));
 if(flag('report'))writeFileSync(flag('report')!,JSON.stringify({root,rewritePages,epochs,errors},null,2));
}finally{solver.destroy();await raw.queue.onSubmittedWorkDone();scratch.destroy();raw.destroy();}
