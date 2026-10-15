/** Native GPU ledger: actual production geometric mass and an owner-face energy diagnostic. */
import { createHighResolutionDamBreakScene } from '../lib/core/scenes';
import { createUniformSolver, withUniformDevice, advanceUniform } from '../tests/helpers/uniform-geometric';
import { paceDawnEventPolling } from '../lib/harness/node-dawn-provider';
import { damProbeOrder } from './dam-probe-order';
import { damProbeLedger } from './dam-probe-ledger';
import { readMixedBuffer } from '../tests/helpers/uniform-mixed-native-fields';
const option=(name:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3);
const steps=Number(option('steps')??480),start=Number(option('start')??1),every=Number(option('every')??1);
const values=JSON.parse(option('values')??'{}');
const maxPhi=Number(option('max-phi')??Infinity);
const replay=process.argv.includes('--replay');
const liveVariant=option('live-variant');
const variants=(option('variants')??'nofoot,nosmooth,nowall,linear').split(',');
const recordSteps=(option('record-steps')??'').split(',').map(Number);
const replaceBody=(code:string,name:string,body:string)=>{
 const at=code.indexOf(`fn ${name}(`);if(at<0)throw new Error(`Missing ${name}`);
 const start=code.indexOf('{',at);let end=start+1,depth=1;for(;depth;end++){if(code[end]==='{')depth++;if(code[end]==='}')depth--;}
 return code.slice(0,start+1)+body+code.slice(end-1);
};
paceDawnEventPolling();
await withUniformDevice('dam stage growth',async device=>{
 const modules=new WeakMap<GPUShaderModule,Map<string,GPUShaderModule>>(),pipelines=new WeakMap<GPUComputePipeline,Map<string,GPUComputePipeline>>();
 const source=(!replay&&!liveVariant)?device:new Proxy(device,{get(target,key){
  if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{
   const module=target.createShaderModule(d);
   if(d.code.includes('fn umVertexTrace(')){
    const alternatives=new Map<string,GPUShaderModule>();
    for(const name of variants){let code=d.code;
     if(name==='nofoot')code=replaceBody(code,'umVertexTrace','return umSurfaceTrace(p);');
     else if(name==='nosmooth')code=replaceBody(code,'umVertexSmoothing','return 0.0;');
     else if(name==='nowall')code=replaceBody(code,'umWallContact','return value;');
     else if(name==='linear')code=replaceBody(code,'umCubicPhi','return cell.value;');
     else throw new Error(`Unknown variant ${name}`);
     alternatives.set(name,target.createShaderModule({...d,label:`${d.label} ${name}`,code}));
    }modules.set(module,alternatives);
   }return module;
  };
  if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
   const pipeline=await target.createComputePipelineAsync(d),alternatives=modules.get(d.compute.module);
   if(alternatives){const compiled=new Map<string,GPUComputePipeline>();for(const [name,module] of alternatives)compiled.set(name,await target.createComputePipelineAsync({...d,compute:{...d.compute,module}}));pipelines.set(pipeline,compiled);}
   return pipeline;
  };
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
 const scene=createHighResolutionDamBreakScene();scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const solver=await createUniformSolver(source,scene,values),f=(solver as any).mixedFrame,nb=f.narrowBandFlip;
 const canonicalize=process.argv.includes('--unordered')?undefined:await damProbeOrder(device,nb);
 const order=nb.order.encode.bind(nb.order);nb.order.encode=(e:GPUCommandEncoder,parity:number)=>{order(e,parity);canonicalize?.(e,parity);};
 const ledger=await damProbeLedger(device,f,recordSteps.some(n=>n>0));let step=0,name='head';
 let selected=liveVariant;
 if(replay||liveVariant){const variant=f.surface.variant.bind(f.surface);f.surface.variant=(p:GPUComputePipeline)=>{const original=variant(p);return selected?(pipelines.get(original)?.get(selected)??original):original;};}
 const pending:{name:string;get:ReturnType<typeof ledger.capture>}[]=[];
 const capture=(e:GPUCommandEncoder,name:string,phi=f.fields.phi,v=f.fields.velocity)=>{
  if(step>=start&&step%every===0)pending.push({name,get:ledger.capture(e,phi,v,recordSteps.includes(step)&&['pre-advect','advect','retire','redistance'].includes(name)?`/tmp/fluid-corners-${step}-${name.replace(/\W/g,'_')}.bin`:undefined)});

 };
 const hook=(o:any,key:string,action:(e:GPUCommandEncoder)=>void,before=false)=>{const original=o[key].bind(o);o[key]=(...args:any[])=>{if(before)action(args[0]);const result=original(...args);if(!before)action(args[0]);return result;};};
 hook(f,'encodeRelayoutHead',e=>capture(e,'pre-layout'),true);
 hook(nb,'move',e=>capture(e,'pre-advect'),true);
 for(const k of ['reconstruct','clearAir','retire','redistance'])hook(nb,k,()=>{name=k;},true);
 const resolve=f.phiResolve.encode.bind(f.phiResolve);f.phiResolve.encode=(e:GPUCommandEncoder,g:any,...args:any[])=>{resolve(e,g,...args);capture(e,g===f.phiResolveGroups.scratch?'advect':name,g===f.phiResolveGroups.scratch?f.fields.phiScratch:f.fields.phi);};
 if(replay){const advect=f.surface.encode.bind(f.surface);f.surface.encode=(e:GPUCommandEncoder,g:any)=>{
  if(step>=start&&step%every===0){for(const variant of variants){selected=variant;advect(e,g);resolve(e,f.phiResolveGroups.scratch);capture(e,`advect:${variant}`,f.fields.phiScratch);}}
  selected=liveVariant;advect(e,g);
 };}
 hook(nb,'transfer',e=>capture(e,'momentum',f.fields.phi,f.fields.departure),true);
 hook(nb,'transfer',e=>capture(e,'p2g'));
 hook(f.forces,'encode',e=>capture(e,'forces',f.fields.phi,f.fields.velocityScratch));
 hook(f.split.transfer,'encodeToSimulation',e=>capture(e,'root-pressure'));
 hook(f.band,'encodeSolve',e=>capture(e,'band-pressure'));
 try{
  for(step=1;step<=steps;step++){
   name='head';await advanceUniform(solver,step/60);
   const result=[];for(const p of pending.splice(0)){
    const sample={stage:p.name,...await p.get()};result.push(sample);
    if(!Number.isFinite(sample.mass)||!Number.isFinite(sample.energy)||!Number.isFinite(sample.minPhi)||!Number.isFinite(sample.maxPhi)||Math.max(Math.abs(sample.minPhi),Math.abs(sample.maxPhi))>maxPhi)throw new Error(`Invalid fluid field at step ${step}: ${JSON.stringify(sample)}`);
   }
   if(result.length)console.log(JSON.stringify({step,time:step/60,budget:Array.from(await readMixedBuffer(device,f.surfaceVolume.budget)),stages:result}));
  }
 }finally{ledger.destroy();solver.destroy();}
});
