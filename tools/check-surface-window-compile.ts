// Device check, compile only: the tree's water extraction module's two window
// entry points as Metal compute pipelines (auto layout; nothing is dispatched).
// Run from the tree: node --import tsx <this>
import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
const W=await import(pathToFileURL(`${process.cwd()}/lib/core/webgpu-water-pipeline.ts`).href);
const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
// The instance is held for the life of the process: a collected one crashes Dawn's event pump.
const gpu=dawn.create([`backend=${process.env.BACKEND??"metal"}`]) as GPU;(globalThis as any).__gpu=gpu;
const adapter=await gpu.requestAdapter();assert.ok(adapter,"no adapter");
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:10}});let errors=0;
device.addEventListener("uncapturederror",(e:any)=>{errors++;console.log("device error: "+e.error.message.slice(0,600));});
const module=device.createShaderModule({code:W.surfaceExtractionShader});
for(const m of (await module.getCompilationInfo()).messages)if(m.type==="error"){errors++;console.log(`${m.lineNum}: ${m.message}`);}
for(const entryPoint of ["collectWindowsMain","extractWindowsMain"]){
 const t=performance.now();
 try{await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}});const elapsed=performance.now()-t;if(elapsed>5000)errors++;console.log(`${elapsed<=5000?"PASS":"FAIL"} ${entryPoint} compiled in ${elapsed.toFixed(0)} ms`);}
 catch(e){errors++;console.log(`FAIL ${entryPoint}: ${String((e as Error).message).slice(0,800)}`);}
}
// Validate the real composed overlay too: it includes the shared address and
// the new page reader, including its slice path, without advancing a solver.
const {gridOverlayShader}=await import("../lib/core/webgpu-grid-overlay");
const overlay=device.createShaderModule({code:gridOverlayShader});
for(const m of (await overlay.getCompilationInfo()).messages)if(m.type==="error"){errors++;console.log(`FAIL overlay ${m.lineNum}: ${m.message}`);}
try{
 const t=performance.now();await device.createRenderPipelineAsync({layout:"auto",vertex:{module:overlay,entryPoint:"vertexMain"},fragment:{module:overlay,entryPoint:"fragmentPages3D",targets:[{format:"rgba8unorm"}]},primitive:{topology:"triangle-list"}});
 console.log(`PASS page overlay compile ${(performance.now()-t).toFixed(0)} ms`);
}catch(e){errors++;console.log(`FAIL overlay: ${(e as Error).message}`);}
device.destroy();process.exit(errors?1:0);
