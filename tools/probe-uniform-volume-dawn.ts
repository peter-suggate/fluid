import { createUniformReferenceComputeShader } from "../lib/methods/uniform/webgpu-uniform-reference.wgsl";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const arg=(name:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
await acquireWebGPUExclusiveLock("dawn-probe", "uniform-volume");
let device:GPUDevice|undefined, solver:GPUSolverInstance|undefined;
try {
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}), { requireWorkerRealm: false });
  const errors:string[]=[]; device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
  const shader=device.createShaderModule({code:createUniformReferenceComputeShader(true)});
  const messages=(await shader.getCompilationInfo()).messages;
  assert.deepEqual(messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}: ${m.message}`),[]);
  const scene=sceneDocument(getSceneDefinition(arg("scene","minimal-power-dam-break-32")));
  const method=arg("method","uniform-volume")==="adaptive-volume"?adaptiveMassMethod:uniformVolumeMethod;
  const values=resolveMethodValues(method,"balanced",{redistance:arg("redistance","on"),totalSurfaceVolume:arg("total-surface-volume","on")});
  const start=performance.now();
  solver=await method.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{},new AbortController().signal) as WebGPUUniformReferenceSolver;
  await solver.waitForSimulationReady?.();
  console.log(JSON.stringify({method:method.id,phase:"ready",ms:performance.now()-start,dimensions:[solver.info.nx,solver.info.ny,solver.info.nz],bytes:solver.info.allocatedBytes}));
  if(process.argv.includes("--trace"))console.log(JSON.stringify({frame:0,stats:await solver.readStats()}));
  const frames=Number(arg("frames","12")); const times:number[]=[];
  for(let frame=1;frame<=frames;frame++){
    const start=performance.now();while(!solver.advanceTo(frame/30,[])) await new Promise(setImmediate);await solver.awaitFrameCompletion?.();await device.queue.onSubmittedWorkDone();times.push(performance.now()-start);
    if(process.argv.includes("--trace"))console.log(JSON.stringify({frame,stats:await solver.readStats()}));
  }
  assert.deepEqual(errors,[]);
  const stats=await solver.readStats();
  if(method.id==="uniform-volume"){
  const tex=solver.volumeTexture; const row=Math.ceil(tex.width*4/256)*256;
  const raw=device.createBuffer({size:row*tex.height*tex.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:tex},{buffer:raw,bytesPerRow:row,rowsPerImage:tex.height},[tex.width,tex.height,tex.depthOrArrayLayers]);device.queue.submit([encoder.finish()]);
  await raw.mapAsync(GPUMapMode.READ);const data=new Float32Array(raw.getMappedRange());let mass=0,maximum=0,minimum=Infinity;const volumeRows=new Array(tex.height).fill(0);let largestCell:number[]=[];
  for(let z=0;z<tex.depthOrArrayLayers;z++)for(let y=0;y<tex.height;y++)for(let x=0;x<tex.width;x++){const v=data[(z*tex.height+y)*row/4+x]!;mass+=v;volumeRows[y]+=v;if(v>maximum)largestCell=[x,y,z];maximum=Math.max(maximum,v);minimum=Math.min(minimum,v);}
  raw.unmap();raw.destroy();console.log(JSON.stringify({rawMass:mass,maximum,minimum,largestCell,volumeRows,drift:mass/stats.initialVolumeCellSum!-1}));
  const phiTexture=(solver as WebGPUUniformReferenceSolver).vertexPhiTexture!;
  const phiRow=Math.ceil(phiTexture.width*4/256)*256;
  const phiBuffer=device.createBuffer({size:phiRow*phiTexture.height*phiTexture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const pe=device.createCommandEncoder();pe.copyTextureToBuffer({texture:phiTexture},{buffer:phiBuffer,bytesPerRow:phiRow,rowsPerImage:phiTexture.height},[phiTexture.width,phiTexture.height,phiTexture.depthOrArrayLayers]);device.queue.submit([pe.finish()]);
  await phiBuffer.mapAsync(GPUMapMode.READ);const phi=new Float32Array(phiBuffer.getMappedRange());
  const rowLiquid=new Array(tex.height).fill(0);let negativeCells=0;
  for(let z=0;z<tex.depthOrArrayLayers;z++)for(let y=0;y<tex.height;y++)for(let x=0;x<tex.width;x++){
    let centre=0;for(let k=0;k<8;k++)centre+=phi[((z+(k>>2))*phiTexture.height+y+((k>>1)&1))*phiRow/4+x+(k&1)]!/8;
    if(centre<0){negativeCells++;rowLiquid[y]++;}
  }
  phiBuffer.unmap();phiBuffer.destroy();console.log(JSON.stringify({negativeCells,rowLiquid}));
  }
  const sorted=times.slice(Math.min(3,times.length-1)).sort((a,b)=>a-b);
  console.log(JSON.stringify({method:method.id,scene:scene.sceneId,phase:"result",frames,medianMs:sorted[Math.floor(sorted.length/2)],times,stats,info:solver.info},null,2));
} finally { solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock(); }
