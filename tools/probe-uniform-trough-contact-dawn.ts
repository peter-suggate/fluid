/** Real voxel-trough contact diagnostic. --contact=pinned reproduces the old
 * min-only wall rule; --contact=assign removes the resting-pool safeguard.
 * The default runs production at the UI pressure tolerance for six seconds. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const arm=arg("arm","baseline"),frames=Number(arg("frames","180")),contact=arg("contact","production");
async function read(device:GPUDevice,texture:GPUTexture){
 const components=texture.format==="rgba32float"?4:1,width=texture.width*components,row=Math.ceil(width*4/256)*256;
 const buffer=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
  const source=new Float32Array(buffer.getMappedRange()),result=new Float32Array(width*texture.height*texture.depthOrArrayLayers);
  for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(source.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+width),(z*texture.height+y)*width);
  return result;
 }finally{if(buffer.mapState==="mapped")buffer.unmap();buffer.destroy();}
}

// Wait for a competing local GPU job before importing Dawn; never overlap it.
const leaseStart=performance.now();
for(;;){
 try{await acquireWebGPUExclusiveLock("dawn-probe",`trough wall contact ${arm}`);break;}
 catch(error){
  if(performance.now()-leaseStart>120_000||!String(error).includes("Refusing concurrent GPU execution"))throw error;
  await new Promise(resolve=>setTimeout(resolve,250));
 }
}
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 let patches=0;
 if(contact!=="production"){
  const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=descriptor=>{
   let code=descriptor.code;
   const needle="if(arriving<1e20){result=select(arriving,min(result,arriving),supported);}else if(continued<1e20){result=select(continued,min(result,continued),supported);}";
   if(code.includes(needle)){
    patches++;
    if(contact==="assign")code=code.replace(needle,"if(arriving<1e20){result=arriving;}else if(continued<1e20){result=continued;}");
    else if(contact==="pinned")code=code.replace(needle,"if(arriving<1e20){result=min(result,arriving);}else if(continued<1e20){result=min(result,continued);}");
    else throw new Error(`Unknown contact control ${contact}`);
   }
   return create({...descriptor,code});
  };
 }
 const scene=sceneDocument(getSceneDefinition("uniform-trough-dam-break"));
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{pressureResidualTolerance:Number(arg("tolerance","5"))});
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(contact!=="production")assert.ok(patches>0,"control must patch the compiled contact rule");
 const {nx,ny,nz}=solver.info,h=scene.container.height_m/ny;
 const open=await read(device,solver.denseLevelSetVolumeSource!.openFraction);
 const wall:{index:number;cellIds:number[];x:number;y:number;z:number}[]=[];
 for(let z=1;z<nz;z++)for(let y=1;y<ny;y++)for(let x=1;x<nx;x++){
  let solid=false;const cellIds:number[]=[];
  for(let k=0;k<8;k++){const i=x-1+(k&1)+nx*(y-1+((k>>1)&1)+ny*(z-1+((k>>2)&1)));if(open[i]!<1e-5)solid=true;else cellIds.push(i);}
  if(solid&&cellIds.length)wall.push({index:x+(nx+1)*(y+(ny+1)*z),cellIds,x,y,z});
 }
 const samples:unknown[]=[];
 for(let frame=0;frame<=frames;frame++){
  if(frame){assert.ok(solver.advanceTo(frame/30,[]));await solver.awaitFrameCompletion();}
  if(frame>3&&frame%30&&frame!==frames)continue;
  const phi=await read(device,solver.vertexPhiTexture!),v=await read(device,solver.volumeTexture),velocity=await read(device,solver.velocityTexture),stats=await solver.readStats();
  let mass=0,maxSpeed=0,highBackMass=0,closedMass=0;
  for(let i=0;i<v.length;i++){assert.ok(Number.isFinite(v[i])&&v[i]!>=-1e-5);mass+=v[i]!;if(open[i]!<1e-5)closedMass+=v[i]!;if(v[i]!>1e-5)maxSpeed=Math.max(maxSpeed,Math.hypot(...velocity.subarray(4*i,4*i+3)));const x=i%nx,y=Math.floor(i/nx)%ny;if(x<nx*.3&&y*h>.45)highBackMass+=v[i]!;}
  const regions:Record<string,{wet:number;unsupported:number;deepest:number;unsupportedDeepest:number}>={all:{wet:0,unsupported:0,deepest:0,unsupportedDeepest:0},highBack:{wet:0,unsupported:0,deepest:0,unsupportedDeepest:0}};
  const worst:{vertex:number[];phi:number;volume:number}[]=[];
  for(const w of wall){const p=phi[w.index]!;assert.ok(Number.isFinite(p));if(p>=0)continue;const support=Math.max(...w.cellIds.map(i=>v[i]!));
   for(const name of ["all",...(w.x<nx*.3&&w.y*h>.45?["highBack"]:[])]){const r=regions[name]!;r.wet++;r.deepest=Math.min(r.deepest,p);if(support<.05){r.unsupported++;r.unsupportedDeepest=Math.min(r.unsupportedDeepest,p);}}
   if(w.x<nx*.3&&w.y*h>.45&&support<.05){worst.push({vertex:[w.x,w.y,w.z],phi:p,volume:support});}
  }
  worst.sort((a,b)=>a.phi-b.phi);
  const sample={frame,mass,closedMass,highBackMass,maxSpeed,regions,worst:worst.slice(0,8),residual:stats.uniformCM11aFineResidualInfinity};samples.push(sample);console.log(JSON.stringify({arm,...sample}));
 }
 assert.deepEqual(errors,[]);
 mkdirSync("artifacts/trough-contact",{recursive:true});
 writeFileSync(`artifacts/trough-contact/${arm}.json`,JSON.stringify({arm,contact,values,dimensions:[nx,ny,nz],h,patches,samples},null,2));
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
