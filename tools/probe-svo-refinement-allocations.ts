/** CPU-only production allocation census. Never imports Dawn or submits GPU work. */
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { svoSceneryDetailCellSize_m } from "../lib/svo/pipeline/svo-render-tuning";
const depth = Number(process.env.FLUID_PROBE_DEPTH ?? 3);
const sceneId = process.env.FLUID_PROBE_SCENE ?? "hero-garden-hose-x10";
const preset = getScenePreset(sceneId).create();
const scene = sceneDocumentAtLattice(getSceneDefinition(sceneId), {
  cellSize_m: preset.voxelDomain.finestCellSize_m,
  detailCellSize_m: svoSceneryDetailCellSize_m(preset.voxelDomain.finestCellSize_m, {environmentRefinementDepth: depth, fluid: false}),
}).scene;
Object.assign(globalThis, {
  GPUBufferUsage: {MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,INDEX:16,VERTEX:32,UNIFORM:64,STORAGE:128,INDIRECT:256,QUERY_RESOLVE:512},
  GPUTextureUsage: {COPY_SRC:1,COPY_DST:2,TEXTURE_BINDING:4,STORAGE_BINDING:8,RENDER_ATTACHMENT:16},
  GPUShaderStage: {VERTEX:1,FRAGMENT:2,COMPUTE:4},
});
const allocations: {label:string;size:number}[]=[];
const noop=()=>{};
const device = {
 limits: {maxTextureDimension3D:2048,maxTextureDimension2D:16384,maxStorageBufferBindingSize:2**32,maxBufferSize:2**32,maxStorageBuffersPerShaderStage:31},
 features:new Set(),queue:{writeBuffer:noop,writeTexture:noop},
 createBuffer(d:GPUBufferDescriptor){allocations.push({label:d.label??"",size:d.size}); return {size:d.size,destroy:noop};},
 createTexture(){return {createView:()=>({}),destroy:noop};},
 createBindGroup:()=>({}),createBindGroupLayout:()=>({}),createPipelineLayout:()=>({}),
 createSampler:()=>({}), createShaderModule:()=>({}),
} as unknown as GPUDevice;
try {
 // This production planning/allocation entry point does not initialize pipelines.
 const {world}=await (WebGPULiveSvoScene as unknown as {buildWorld:(...args: unknown[])=>Promise<{world:{destroy():void}}>}).buildWorld(device,scene,{environmentRefinementDepth:depth,cpuBrickSelection:true},
  (p:{stage:string})=>console.error(p.stage), {sliceBudget_ms:8});
 console.log(JSON.stringify({sceneId,depth,bytes:allocations.reduce((n,a)=>n+a.size,0),allocations:allocations.sort((a,b)=>b.size-a.size)},null,2));
 world.destroy();
} catch(error) {
 console.log(JSON.stringify({sceneId,depth,error:String(error),allocations},null,2));process.exitCode=1;
}
