/** Full depth-3 rendering probe with allocation limits and a fence after every batch. */
import assert from "node:assert/strict";
import {PerformanceObserver} from "node:perf_hooks";
const gcObserver=new PerformanceObserver(list=>{for(const entry of list.getEntries())if(entry.duration>50)console.log(JSON.stringify({phase:"host-gc",duration_ms:entry.duration}));});
gcObserver.observe({entryTypes:["gc"]});
import {GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import {createDawnRenderDevice,buildSvoDrySceneAssembly,packSvoDryRigidBodies,packSvoDryViewUniforms} from "./svo-dry-frame-harness";
import {WebGPULiveSvoScene} from "../lib/svo/webgpu-live-svo-scene";
import {SparseVoxelDrySceneRenderer} from "../lib/svo/webgpu-svo-dry-scene";
import {getSceneDefinition,getScenePreset} from "../lib/core/scenes";
import {sceneDocumentAtLattice} from "../lib/core/scene-definition";
import {svoSceneryDetailCellSize_m,DEFAULT_SVO_RENDER_TUNING} from "../lib/svo/svo-render-tuning";
import {defaultCamera} from "../lib/core/model";
const depth=Number(process.env.FLUID_PROBE_DEPTH??3), id=process.env.FLUID_PROBE_SCENE??"hero-garden-hose-x10";
const preset=getScenePreset(id),base=preset.create();
const scene=sceneDocumentAtLattice(getSceneDefinition(id),{cellSize_m:base.voxelDomain.finestCellSize_m,
 detailCellSize_m:svoSceneryDetailCellSize_m(base.voxelDomain.finestCellSize_m,{environmentRefinementDepth:depth,fluid:false})}).scene;
scene.surfaceStyle="voxel-flat";
const {device:raw,adapterInfo,validationErrors}=await createDawnRenderDevice();
let allocated=0;const allocations:{label:string;size:number}[]=[];
const device=new Proxy(raw,{get(target,key){
 if(key==='createBuffer')return (d:GPUBufferDescriptor)=>{
  assert.ok(d.size<=1024**3,`Safety stop: single buffer ${d.label}: ${d.size}`);
  assert.ok(allocated+d.size<=3*1024**3,`Safety stop: buffer census exceeds 3 GiB`);
  allocated+=d.size;allocations.push({label:d.label??"",size:d.size});return target.createBuffer(d);
 };
 const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
}});
const started=performance.now();
const world=await WebGPULiveSvoScene.create(device,scene,'balanced',p=>console.log(JSON.stringify({phase:p.label,elapsed_ms:performance.now()-started})),undefined,{environmentRefinementDepth:depth,radianceFeedback:false,cpuBrickSelection:process.env.FLUID_PROBE_CPU_SELECTION==="1"});
const preparation_ms=performance.now()-started;
assert.equal(world.builtRefinementDepth,depth,'No silent refinement downgrade');
assert.ok(allocations.filter(a=>/Sparse brick source (geometry|velocity|material owners)/.test(a.label)).every(a=>a.size<=16));
global.gc?.(); // Release temporary CPU planner objects before measuring GPU submissions.
const readback=device.createBuffer({size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
async function submit(encoder:GPUCommandEncoder, recorder?:GPUPassTimestampRecorder){
 const t=performance.now();const command=encoder.finish();const finished=performance.now();
 device.queue.submit([command]);const submitted=performance.now();await device.queue.onSubmittedWorkDone();
 const completed=performance.now();const elapsed=completed-t;
 const reading=await recorder?.read();if(elapsed>50)console.log(JSON.stringify({phase:"slow-batch",elapsed,finish_ms:finished-t,submit_ms:submitted-finished,wait_ms:completed-submitted,memory:process.memoryUsage(),reading}));
 assert.ok(elapsed<500,`Safety stop: batch wall time ${elapsed.toFixed(1)} ms`);
 assert.deepEqual(validationErrors,[]);return elapsed;
}
let maintenanceBatches=0,maxBatch_ms=0;
for(;maintenanceBatches<10000;maintenanceBatches++){
 const encoder=device.createCommandEncoder();const work=world.encodeSceneMaintenance(encoder);
 maxBatch_ms=Math.max(maxBatch_ms,await submit(encoder));if(!work)break;
 if(maintenanceBatches%100===0)console.log(JSON.stringify({phase:'maintenance',maintenanceBatches,maxBatch_ms}));
}
assert.ok(maintenanceBatches<10000,'maintenance must converge');
const source=world.sparseVoxelSceneSource;assert.ok(source?.structural);
const {drySceneData}=buildSvoDrySceneAssembly(scene,source);
const bodies=packSvoDryRigidBodies(scene),width=Number(process.env.FLUID_PROBE_WIDTH??64),height=Number(process.env.FLUID_PROBE_HEIGHT??64);
const uniforms=device.createBuffer({size:416,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const body=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({scene,camera:{...defaultCamera,...preset.camera},environmentId:scene.environment??'default',info:world.info,bodyCount:bodies.count,width,height}));
device.queue.writeBuffer(body,0,bodies.data);
const renderer=new SparseVoxelDrySceneRenderer(device,uniforms,body,'rgba16float','raster-primary','bounds','split',0,true,true,false,{surfaceMesh:true,singlePassReconstruction:process.env.FLUID_STEADY_SINGLE_PASS!=="0"});
renderer.setLightingOptions({globalIlluminationEnabled:false,coneTracingMode:'off',shadowsEnabled:false,ambientOcclusionEnabled:false});
await renderer.initialize();renderer.setRigidBodyCount(bodies.count);renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,coneLightingScale:1});
renderer.setSource(source);renderer.publishScene(drySceneData);renderer.ensureSize(width,height);
const target=device.createTexture({size:[width,height],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
global.gc?.();
let receipt:number[]=[];let frames=0;
for(;frames<4000;frames++){
 const encoder=device.createCommandEncoder();const recorder=new GPUPassTimestampRecorder(device,128,"Depth-3 mesh probe");
 assert.ok(renderer.encode(recorder.instrument(encoder),target));renderer.copySurfaceMeshDiagnostics(encoder,readback);recorder.resolve(encoder);
 maxBatch_ms=Math.max(maxBatch_ms,await submit(encoder,recorder));await readback.mapAsync(GPUMapMode.READ);receipt=Array.from(new Uint32Array(readback.getMappedRange()));readback.unmap();
 if(frames%100===0)console.log(JSON.stringify({phase:'mesh',frames,cursor:receipt[14],quads:receipt[4],ready:receipt[13],overflow:receipt[5]}));
 if(receipt[13]===1&&receipt[15]===0)break;
}
assert.ok(frames<4000,'mesh must complete');assert.ok(receipt[1]>0,'camera must draw quads');assert.equal(receipt[5],0);
console.log(JSON.stringify({phase:'complete',adapterInfo,depth,allocated,preparation_ms,total_ms:performance.now()-started,maintenanceBatches,frames,maxBatch_ms,receipt,validationErrors}));
if(process.env.FLUID_PROBE_PNG){
 const {writeFramePng}=await import('./write-frame-png');
 const pitch=Math.ceil(width*8/256)*256;
 const pixels=device.createBuffer({size:pitch*height,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:target},{buffer:pixels,bytesPerRow:pitch},{width,height});
 device.queue.submit([encoder.finish()]);await pixels.mapAsync(GPUMapMode.READ);
 const sourceRows=new Uint8Array(pixels.getMappedRange()),packedRows=new Uint32Array(width*height*2),packedBytes=new Uint8Array(packedRows.buffer);
 for(let row=0;row<height;row++)packedBytes.set(sourceRows.subarray(row*pitch,row*pitch+width*8),row*width*8);
 writeFramePng(process.env.FLUID_PROBE_PNG,{width,height,packedRows});pixels.unmap();pixels.destroy();
}
if(process.env.FLUID_STEADY_BENCH === "1") {
 const {benchmarkRasterSteady}=await import("./benchmark-svo-raster-steady");
 await benchmarkRasterSteady(device,renderer,uniforms,{scene,camera:{...defaultCamera,...preset.camera},environmentId:scene.environment??'default',info:world.info,bodyCount:bodies.count,width,height},{body,source});
 assert.deepEqual(validationErrors,[]);
}
renderer.destroy();world.destroy();for(const b of [uniforms,body,readback])b.destroy();target.destroy();device.destroy();gcObserver.disconnect();
