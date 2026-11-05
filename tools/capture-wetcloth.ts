/** Capture a source fixture through the actual application renderer/physics.
 * Run through fluid-gpu-run.sh check, e.g. --scene=wetcloth-source-pore-test-mid
 * --time=0 --out=artifacts/wetcloth/pore-mid-t0.png. Unvalidated diagnostic
 * captures are not paper-reproduction certificates. */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import '../lib/methods';
import {FluidLabRenderer} from '../lib/core/webgpu-renderer';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {defaultCamera,markSceneRevision} from '../lib/core/model';
import {sceneRenderTuning} from '../lib/core/scene-render-profile';
import {DEFAULT_SVO_LIGHTING_OPTIONS} from '../lib/svo/pipeline/svo-render-options';
import {surfaceMeshAllowsLiveStartup} from '../lib/svo/features/primary-visibility/svo-surface-mesh';
import {getMethod} from '../lib/core/method-registry';
import {resolveMethodValues,type GPUSolverInstance} from '../lib/core/method-contract';
import type {EffectiveRendererStatus} from '../lib/core/renderer-status';
import type {SparseVoxelSceneRenderSource} from '../lib/core/webgpu-voxel-debug';
import {encodeRgbPng} from '../lib/core/png-codec';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {prepareWetClothSource,buildWetClothSourceScene,wetClothValidationDefinitions} from '../lib/cloth/scenes/definitions';
import {wetClothSource} from '../lib/cloth/scenes/sources';
import {sourceCameraForScene} from '../lib/cloth/source-camera';
import {auditUniformGPUAllocations} from './uniform-gpu-allocation-audit';
import {initializeRigidBodies} from '../lib/core/rigid-body';

const arg=(name:string)=>process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3);
const id=arg('scene')??'wetcloth-source-pore-test-mid',time=Number(arg('time')??0),width=Number(arg('width')??800),height=Number(arg('height')??600);
assert.ok(time>=0&&Number.isFinite(time)&&Number.isSafeInteger(width)&&width>0&&Number.isSafeInteger(height)&&height>0);
const output=resolve(arg('out')??`artifacts/wetcloth/${id}-${time}.png`);await mkdir(dirname(output),{recursive:true});
const request:typeof fetch=async url=>{const path=String(url);if(!path.startsWith('/assets/wetcloth/')||path.includes('..'))throw new Error('Unexpected source URL');return new Response(await readFile(resolve('public'+path)));};
await prepareWetClothSource(id,request);const scene=markSceneRevision(buildWetClothSourceScene(id)),source=wetClothSource(id);
const bodies=initializeRigidBodies(scene.rigidBodies);
const cameraView=sourceCameraForScene(scene),camera=cameraView?.camera??{...defaultCamera,target_m:{x:0,y:scene.container.height_m/2,z:0},distance_m:Math.max(scene.container.width_m,scene.container.height_m,scene.container.depth_m)*1.6};
const profile=wetClothValidationDefinitions.find(s=>s.id===id)!.methodProfile!;
const method=getMethod('uniform-narrow-band-flip'),values=resolveMethodValues(method,'high',{...method.appDefaults,...profile.overrides});
const dt=source.dt_s,target=Math.ceil(time/dt-1e-9)*dt,errors:string[]=[],started=performance.now();
let device:GPUDevice|undefined,swap:GPUTexture|undefined,renderer:FluidLabRenderer|undefined,status:EffectiveRendererStatus|undefined,failure:string|undefined;
let lastStatusLog=0,lastFrameLog=0;
const deadline=started+Number(arg('timeout')??1800)*1000;
const turn=()=>new Promise<void>(resolve=>setImmediate(resolve));
const check=()=>{if(failure||errors.length)throw new Error(failure??errors.join('\n'));if(performance.now()>deadline)throw new Error(`Source capture exceeded deadline: ${JSON.stringify({effectiveStatus:status,completedPresentations:renderer?.completedPresentationCount})}`);};
let raf=0;const timers=new Map<number,ReturnType<typeof setTimeout>>();
globalThis.requestAnimationFrame=callback=>{const i=++raf;timers.set(i,setTimeout(()=>{timers.delete(i);callback(performance.now());},0));return i;};
globalThis.cancelAnimationFrame=i=>{clearTimeout(timers.get(i));timers.delete(i);};
const report:Record<string,unknown>={sourceId:id,source:source.source,configuration:source.variant?.kind??'released-input',variant:source.variant,sourceSettings:source.sourceSettings,methodProfile:profile,time_s:target,dt_s:dt,h_m:source.cellSize_m,camera:cameraView??camera,reproductionValidated:false,
 capturePath:'FluidLabRenderer.drawWithDeferredPresentation → production BGRA8 swap chain'};
try{
 paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??'node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]).requestAdapter();assert.ok(adapter);report.adapter=adapter.info;
 Object.defineProperty(navigator,'gpu',{configurable:true,value:{requestAdapter:async()=>({features:adapter.features,limits:adapter.limits,info:adapter.info,
  requestDevice:async(descriptor:GPUDeviceDescriptor)=>{device=managedGPUDevice(auditUniformGPUAllocations(await adapter.requestDevice(descriptor),6*2**30).device,{requireWorkerRealm:false});device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});return device;}}),getPreferredCanvasFormat:()=> 'bgra8unorm'}});
 const canvas={width,height,getContext:()=>({configure:()=>{swap=device!.createTexture({size:[width,height],format:'bgra8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});},getCurrentTexture:()=>swap,unconfigure:()=>{}})} as unknown as OffscreenCanvas;
 renderer=new FluidLabRenderer(canvas,s=>{if(['unavailable','lost','blocked'].includes(s.state))failure=s.label;
  if(performance.now()-lastStatusLog>5000){lastStatusLog=performance.now();console.log(JSON.stringify({phase:'renderer',state:s.state,label:s.label,elapsed_ms:lastStatusLog-started}));}
 },undefined,undefined,undefined,s=>{status=s;report.effectiveStatus=s;
  console.log(JSON.stringify({phase:'effective-renderer',elapsed_ms:performance.now()-started,...s}));
  if(s.state==='failed')failure=`Source renderer failed: ${s.failureReason}: ${s.detail??''}`;
 });
 const internal=renderer as unknown as {gpuFluid?:GPUSolverInstance;solverBuild?:{settled?:Promise<void>};svoDrySceneSource?:SparseVoxelSceneRenderSource};
 const inspectPublication=async()=>{
  const structural=internal.svoDrySceneSource?.structural;if(!structural)return undefined;
  const binding=structural.publication.state,bytes=32,read=device!.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=device!.createCommandEncoder();encoder.copyBufferToBuffer(binding.buffer,binding.offset??0,read,0,bytes);device!.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const result={words:Array.from(new Uint32Array(read.getMappedRange())),capacities:structural.capacities,maintenance:structural.sceneMaintenance?{surfaceVertexOffsetBytes:structural.sceneMaintenance.surfaceVertexOffsetBytes,surfaceVertexKind:structural.sceneMaintenance.surfaceVertexKind}:undefined};
   console.log(JSON.stringify({phase:'publication-diagnostic',...result}));report.publicationDiagnostic=result;return result;
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();}
 };
 const args=(t:number):Parameters<FluidLabRenderer['draw']>=>[t,scene,camera,bodies,undefined,{methodId:method.id,quality:'high',values,simulationEpoch:0,inFlightDepth:1},
  {axis:'off',position:0},'default','full-scene','shaded',DEFAULT_SVO_LIGHTING_OPTIONS,undefined,sceneRenderTuning(scene.renderProfile)];
 const draw=async(t:number)=>{check();const log=performance.now()-lastFrameLog>10000;
  if(log){lastFrameLog=performance.now();console.log(JSON.stringify({phase:'frame-start',elapsed_ms:lastFrameLog-started,effectiveState:status?.state,completedPresentations:renderer!.completedPresentationCount}));}
  const frame=await renderer!.drawWithDeferredPresentation(args(t));
  if(log)console.log(JSON.stringify({phase:'frame-encoded',elapsed_ms:performance.now()-started,presentationSubmitted:frame.presentationSubmitted}));
  await device!.queue.onSubmittedWorkDone();await turn();
  if(log)console.log(JSON.stringify({phase:'frame-complete',elapsed_ms:performance.now()-started,presentationSubmitted:frame.presentationSubmitted,completedPresentations:renderer!.completedPresentationCount,effectiveState:status?.state,surfaceMesh:status?.surfaceMesh}));
  check();return frame;};
 await renderer.initialize();await renderer.setSimulationRunning(false);
 // Optional render pipelines can finish before the draw that starts scene
 // initialization. Drive normal presentation turns until its owner attaches.
 while(!internal.gpuFluid){renderer.draw(...args(0));await internal.solverBuild?.settled;await turn();check();}
 let publicationWait=0;
 for(;;){const frame=await draw(0);if(status?.state==='active'&&surfaceMeshAllowsLiveStartup(status.surfaceMesh)&&frame?.presentationSubmitted&&renderer.completedPresentationCount>0)break;
  if(status?.surfaceMesh?.state==='blocked'&&status.surfaceMesh.fallbackReason==='publication'){
   publicationWait||=performance.now();if(performance.now()-publicationWait>1000){await inspectPublication();throw new Error('Source renderer publication remained incomplete after a fenced initial frame');}
  }
 }
 console.log(JSON.stringify({phase:'initial-presented',elapsed_ms:performance.now()-started}));
 await renderer.setSimulationRunning(true);let completed=0,lastLog=performance.now();
 while(completed<target-dt*.01){await draw(Math.min(target,(Math.round(completed/dt)+1)*dt));completed=internal.gpuFluid.info.completedTime_s??0;
  if(performance.now()-lastLog>10000){lastLog=performance.now();console.log(JSON.stringify({phase:'simulate',completedTime_s:completed,targetTime_s:target}));}}
 await renderer.setSimulationRunning(false);await draw(target);await internal.gpuFluid.awaitFrameCompletion?.();
 assert.ok(swap&&device);const stride=Math.ceil(width*4/256)*256,read=device.createBuffer({size:stride*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  const e=device.createCommandEncoder();e.copyTextureToBuffer({texture:swap},{buffer:read,bytesPerRow:stride},[width,height]);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
  const bgra=new Uint8Array(read.getMappedRange()),rgb=new Uint8Array(width*height*3);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const a=y*stride+4*x,b=3*(y*width+x);rgb.set([bgra[a+2]!,bgra[a+1]!,bgra[a]!],b);}
  await writeFile(output,encodeRgbPng({width,height,rgb}));
 }finally{if(read.mapState==='mapped')read.unmap();read.destroy();}
 Object.assign(report,{info:internal.gpuFluid.info,cloth:internal.gpuFluid.clothSurfaces?.map(s=>({documentId:s.documentId,time_s:s.time_s,tick:s.tick,water_m3:Array.from(s.waterVolume_m3).reduce((a,b)=>a+b,0)})),elapsed_ms:performance.now()-started});
 console.log(output);
}catch(error){report.failure=String(error);throw error;}
finally{await writeFile(output.replace(/\.png$/,'.json'),JSON.stringify(report,null,2));renderer?.destroy();swap?.destroy();device?.destroy();for(const timer of timers.values())clearTimeout(timer);}
