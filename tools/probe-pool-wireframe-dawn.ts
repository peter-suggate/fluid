import {losassoMethod} from "../lib/methods/losasso/method";
import {createGlobalFineLevelSetConsumerSource} from "../lib/core/octree-consumer-sampling";
import {WebGPULiveSvoScene} from "../lib/svo/webgpu-live-svo-scene";
import {SparseVoxelDrySceneRenderer} from "../lib/svo/webgpu-svo-dry-scene";
import {buildSvoDrySceneAssembly} from "./svo-dry-frame-harness";
import {sceneDocument} from "../lib/core/scene-definition";
import {environmentIndex} from "../lib/core/environments";
import {getSceneDefinition,createCoarseFirstPoolImpactScene} from "../lib/core/scenes";
import {readFile} from "node:fs/promises";
import {pathToFileURL} from "node:url";
import sharp from "sharp";
import {RasterWaterPipeline} from "../lib/core/webgpu-water-pipeline";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
await acquireWebGPUExclusiveLock("dawn-test","pool-wireframe");
const {create,globals}=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,globals);
const gpu=create(["backend=metal"]);const adapter=await gpu.requestAdapter();const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
try{
const plan=JSON.parse(await readFile("artifacts/pool-gap/source.json","utf8"));
const load=async(name:string)=>{const bytes=await readFile(`artifacts/pool-gap/${name}.bin`);const b=device.createBuffer({size:bytes.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,bytes);return b;};
const metadata=await load("metadata"),samples=await load("samples"),worklist=await load("worklist");
const words=await readFile("artifacts/pool-gap/worklist.bin");
const uniform=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const body=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM});
const width=1024,height=512;const u=new Float32Array(104);u.set([width,height,0,0]);u.set([0,5.8,10,0],4);u.set([0,2,0,0],8);u.set([6.4,4.8,6.4,1.6],12);u.set([11008,.05,0,1],16);u.set([128,96,128,3],20);u.set([0,.5,1,0],24);u[28]=environmentIndex("stage");u[3]=-1;device.queue.writeBuffer(uniform,0,u);
const pipeline=new RasterWaterPipeline(device,"rgba8unorm",uniform,body);await pipeline.initialize();
pipeline.setFluidDomain({origin_m:[-3.2,0,-3.2],cellSize_m:[.05,.05,.05],dimensions:[128,96,128]});
pipeline.setVolume(device.createTexture({size:[1,1,1],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING}),device.createTexture({size:[1,1],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING}));
pipeline.setGlobalFineLevelSet({kind:"global-fine-levelset-sampling",metadata:{buffer:metadata},samples:{buffer:samples},worklist:{buffer:worklist},sampleDimensions:plan.sampleDimensions,brickDimensions:plan.brickDimensions,brickResolution:8,samplesPerBrick:512,pageCapacity:plan.maximumResidentBricks,fineFactor:1,fineCellWidth:.05,domainOrigin:[0,0,0],generation:words.readUInt32LE(0),surfaceMeshRefinement:2});
if(process.env.FLUID_LOSASSO){
 const scene=sceneDocument(getSceneDefinition("coarse-first-pool-impact"));
 const solver=await losassoMethod.createSolverAsync!(device,scene,"balanced",{...losassoMethod.presetFor("balanced"),maximumLeafSize:"8",interfaceRefinementBandCells:4,globalFineLevelSetFactor:"1"},undefined,()=>{});
 await device.queue.onSubmittedWorkDone();
 console.log("actual source",solver.info,solver.coarseLevelSetSource?.kind,!!solver.globalFineLevelSetSource);
 pipeline.setGlobalFineLevelSet(solver.globalFineLevelSetSource?createGlobalFineLevelSetConsumerSource(solver.globalFineLevelSetSource):undefined);
 pipeline.setCoarseLevelSet(solver.coarseLevelSetSource);
}
pipeline.ensureSize(width,height);const output=device.createTexture({size:[width,height],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
let dryRenderer:SparseVoxelDrySceneRenderer|undefined;
if(process.env.FLUID_SVO){
 const scene=sceneDocument(getSceneDefinition("coarse-first-pool-impact"));const live=await WebGPULiveSvoScene.create(device,scene,"balanced",()=>{});
 const maintenance=device.createCommandEncoder();live.encodeSceneMaintenance(maintenance);device.queue.submit([maintenance.finish()]);await device.queue.onSubmittedWorkDone();
 const source=live.sparseVoxelSceneSource!;dryRenderer=new SparseVoxelDrySceneRenderer(device,uniform,body,"rgba16float","raster-primary","bounds","split",0,true,true,true,{surfaceMesh:true});await dryRenderer.initialize();dryRenderer.setSource(source);dryRenderer.publishScene(buildSvoDrySceneAssembly(scene,source).drySceneData);dryRenderer.ensureSize(width,height);
}
const encoder=device.createCommandEncoder();const result=pipeline.encode(encoder,output,128,96,128,false,0,1,dryRenderer ? (e,t)=>dryRenderer!.encode(e,t) : undefined,undefined,true,"clear",true,undefined,true,"wireframe");console.log(result);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
const readback=device.createBuffer({size:width*height*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const copy=device.createCommandEncoder();copy.copyTextureToBuffer({texture:output},{buffer:readback,bytesPerRow:width*4},[width,height]);device.queue.submit([copy.finish()]);await readback.mapAsync(GPUMapMode.READ);await sharp(Buffer.from(readback.getMappedRange()),{raw:{width,height,channels:4}}).png().toFile("artifacts/pool-gap/wire.png");readback.unmap();pipeline.destroy();
}finally{device.destroy();await releaseWebGPUExclusiveLock();}
