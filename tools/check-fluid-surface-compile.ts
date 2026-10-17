/** Capture the real surface module, or cold-compile one of its actual variants.
 * `capture` writes the generated WGSL and constants. `<index> <nonce>` renames
 * that entry point to defeat Metal's shader cache without changing its work.
 * Run with `node --import tsx tools/check-fluid-surface-compile.ts capture`,
 * then `node --import tsx tools/check-fluid-surface-compile.ts 12`.
 */
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { resolveMethodValues } from '../lib/core/method-contract';
import { uniformNarrowBandMethod as method } from '../lib/methods/uniform/uniform-narrow-band-method';
import { createEmptyScene } from '../lib/core/empty-scene';
import { interactionCatalog } from '../lib/interaction/installed';
import { interactionScene } from '../lib/interaction/adapters/scene';

const [mode='capture',nonce=`cold_${Date.now().toString(36)}`]=process.argv.slice(2);
const directory=process.env.FLUID_COMPILE_DIR??'/tmp/fluid-surface-compile';
mkdirSync(directory,{recursive:true});
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
type Variant={entryPoint:string;constants:Record<string,number>};
try {
 if(mode==='capture'){
  const modules=new Set<GPUShaderModule>(),variants:Variant[]=[];
  const shader=raw.createShaderModule.bind(raw);
  raw.createShaderModule=descriptor=>{const module=shader(descriptor);if(descriptor.label==='Uniform mixed surface'){modules.add(module);writeFileSync(`${directory}/surface.wgsl`,descriptor.code);}return module;};
  const managed=managedGPUDevice(raw,{requireWorkerRealm:false});
  const captured=new Error('Surface descriptors captured');
  const device=new Proxy(managed,{get(target,key){
   if(key==='createComputePipelineAsync')return (descriptor:GPUComputePipelineDescriptor)=>{
    if(!modules.has(descriptor.compute.module))return target.createComputePipelineAsync(descriptor);
    variants.push({entryPoint:descriptor.compute.entryPoint!,constants:descriptor.compute.constants??{}});
    // The initializer enumerates every variant before awaiting Promise.all.
    // Stop there: capture never compiles or dispatches a substitute kernel.
    return Promise.reject(captured);
   };
   const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
  }});
  const scene=createEmptyScene({extents_m:{x:2,y:2,z:2},finestCellSize_m:.125});scene.scenery=undefined;scene.systems={fluid:true};scene.container.fillFraction=.5;scene.fluid.initialCondition='tank-fill';scene.fluid.surfaceTension_N_m=0;scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.interaction={schemaVersion:1,entities:[interactionCatalog.createEntity({id:'float',kind:'fluid.crate',transform:{position_m:{x:0,y:.9,z:0},orientation:{x:0,y:0,z:0,w:1}}})],connections:[]};
  const values=resolveMethodValues(method,'balanced',{...method.appDefaults,timeStep:'scene',rigidCoupling:'on'});
  try{await method.createSolverAsync!(device,interactionScene(scene),'balanced',values,undefined,()=>{});assert.fail('Capture must stop before warmup');}
  catch(error){if(error!==captured)throw error;}
  assert.ok(variants.some(v=>v.entryPoint==='advectOwners'));writeFileSync(`${directory}/variants.json`,JSON.stringify(variants,null,2));
  console.log(JSON.stringify(variants.map((v,index)=>({index,...v}))));
 }else{
  const variants=JSON.parse(readFileSync(`${directory}/variants.json`,'utf8')) as Variant[],variant=variants[Number(mode)];assert.ok(variant);assert.match(nonce,/^[a-zA-Z][a-zA-Z0-9_]*$/);
  let code=readFileSync(process.env.FLUID_COMPILE_SOURCE??`${directory}/surface.wgsl`,'utf8');
  const entryPoint=`${variant.entryPoint}_${nonce}`;code=code.replace(new RegExp(`\\b${variant.entryPoint}\\b`,'g'),entryPoint);
  const module=raw.createShaderModule({label:'Surface compile regression',code});
  const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.deepEqual(messages,[]);
  const start=performance.now();console.log(JSON.stringify({phase:'start',index:Number(mode),entryPoint,constants:variant.constants}));
  await raw.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint,constants:variant.constants}});
  const result={index:Number(mode),entryPoint,milliseconds:performance.now()-start};console.log(JSON.stringify(result));writeFileSync(`${directory}/${entryPoint}.json`,JSON.stringify(result,null,2));
 }
 assert.deepEqual(errors,[]);
}finally{raw.destroy();}
