/** Smooth/voxel-face x10 render-budget exploration. Run through run-webgpu-exclusive.ts.
 * Full-frame GPU spans; no pass withholding, solver, or water optical composite.
 * FLUID_EXPLORE_OUT, FLUID_EXPLORE_CYCLES, FLUID_EXPLORE_ARMS, FLUID_EXPLORE_VIEWS select evidence. */
/** Full depth-3 rendering probe with allocation limits and a fence after every batch. */
import assert from "node:assert/strict";
import {PerformanceObserver} from "node:perf_hooks";
const gcObserver=new PerformanceObserver(list=>{for(const entry of list.getEntries())if(entry.duration>50)console.log(JSON.stringify({phase:"host-gc",duration_ms:entry.duration}));});
gcObserver.observe({entryTypes:["gc"]});
import {GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import {createDawnRenderDevice,buildSvoDrySceneAssembly,packSvoDryRigidBodies,packSvoDryViewUniforms} from "./svo-dry-frame-harness";
import {WebGPULiveSvoScene} from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import {createProductionSparseVoxelDrySceneRenderer} from "../lib/core/webgpu-renderer";
import {getSceneDefinition,getScenePreset} from "../lib/core/scenes";
import {sceneDocumentAtLattice} from "../lib/core/scene-definition";
import {svoSceneryDetailCellSize_m,DEFAULT_SVO_RENDER_TUNING,resolveSvoSurfaceTuning} from "../lib/svo/pipeline/svo-render-tuning";
import {defaultCamera} from "../lib/core/model";
import {DEFAULT_SVO_LIGHTING_OPTIONS} from "../lib/svo/pipeline/svo-render-options";
import {SVO_GBUFFER_FLAGS,SVO_GBUFFER_PRODUCERS,svoGBufferProducerOf} from "../lib/svo/contracts/svo-gbuffer";
const depth=Number(process.env.FLUID_PROBE_DEPTH??0), id=process.env.FLUID_PROBE_SCENE??"hero-garden-hose-x10";
const preset=getScenePreset(id),base=preset.create();
const scene=sceneDocumentAtLattice(getSceneDefinition(id),{cellSize_m:base.voxelDomain.finestCellSize_m,
 detailCellSize_m:svoSceneryDetailCellSize_m(base.voxelDomain.finestCellSize_m,{environmentRefinementDepth:depth,fluid:false})}).scene;
const surfaceStyle=process.env.FLUID_EXPLORE_SURFACE_STYLE??"smooth";
assert.ok(surfaceStyle==="smooth"||surfaceStyle==="voxel-flat", "Expected smooth or voxel-flat surface style");
scene.surfaceStyle=surfaceStyle;
const smooth=surfaceStyle==="smooth";
const {device:raw,adapterInfo,validationErrors}=await createDawnRenderDevice();
let allocated=0;const allocations:{label:string;size:number}[]=[];
const renderPipelineLabels:string[]=[];
const device=new Proxy(raw,{get(target,key){
 if(key==='createRenderPipelineAsync')return (d:GPURenderPipelineDescriptor)=>{
  renderPipelineLabels.push(d.label??'');return target.createRenderPipelineAsync(d);
 };
 if(key==='createBuffer')return (d:GPUBufferDescriptor)=>{
  assert.ok(d.size<=1024**3,`Safety stop: single buffer ${d.label}: ${d.size}`);
  assert.ok(allocated+d.size<=3*1024**3,`Safety stop: buffer census exceeds 3 GiB`);
  allocated+=d.size;allocations.push({label:d.label??"",size:d.size});return target.createBuffer(d);
 };
 const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
}});
const started=performance.now();
const world=await WebGPULiveSvoScene.create(device,scene,'balanced',p=>console.log(JSON.stringify({phase:p.label,elapsed_ms:performance.now()-started})),undefined,{environmentRefinementDepth:depth,radianceFeedback:false,surfaceDualMarchingCubes:smooth,cpuBrickSelection:process.env.FLUID_PROBE_CPU_SELECTION==="1"});
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
 // Scene publication includes first-use compilation; it is excluded from steady measurements.
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
const rasterAo=process.env.FLUID_EXPLORE_RASTER_AO==='1';
const renderer=createProductionSparseVoxelDrySceneRenderer(device,uniforms,body,'mesh',false,false,false,rasterAo);
const productionLighting=true;
renderer.setLightingOptions(productionLighting ? {...DEFAULT_SVO_LIGHTING_OPTIONS,coneLightingScale:0.5,coneTracingMode:rasterAo?'raster-ao':'cones'} : {globalIlluminationEnabled:false,coneTracingMode:'off',shadowsEnabled:false,ambientOcclusionEnabled:false});
await renderer.initialize((label,completed,total)=>console.log(JSON.stringify({phase:'pipeline',label,completed,total,elapsed_ms:performance.now()-started})));renderer.setRigidBodyCount(bodies.count);renderer.setRenderTuning(resolveSvoSurfaceTuning(DEFAULT_SVO_RENDER_TUNING,smooth));
renderer.setSource(source);renderer.publishScene(drySceneData);renderer.ensureSize(width,height);
if(productionLighting)await renderer.ensureConeLightingPrepass();
const target=device.createTexture({size:[width,height],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
global.gc?.();
let receipt:number[]=[];let frames=0;
async function assertRasterPixels(pending:boolean){
 const texture=renderer.gBufferTextures?.packedSurface;assert.ok(texture);
 const pitch=Math.ceil(width*16/256)*256;
 const pixels=device.createBuffer({size:pitch*height,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 try{
  const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer:pixels,bytesPerRow:pitch},{width,height});
  device.queue.submit([encoder.finish()]);await pixels.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(pixels.getMappedRange()),counts=Array<number>(8).fill(0);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const flags=(words[y*pitch/4+x*4+3]>>>4)&0xffff;
   if(flags&SVO_GBUFFER_FLAGS.validSurface)counts[svoGBufferProducerOf(flags)]++;
  }
  console.log(JSON.stringify({phase:'producer-counts',counts,receipt}));
  assert.equal(counts[SVO_GBUFFER_PRODUCERS.tracedPrimary],0,'Raster must never substitute traced primary pixels');
  if(pending){
   assert.equal(counts[SVO_GBUFFER_PRODUCERS.rasterBackground],0,'Pending mesh must withhold its background');
   assert.equal(counts[SVO_GBUFFER_PRODUCERS.brickRaster],0,'Pending mesh must withhold partial geometry');
  }else assert.ok(counts[SVO_GBUFFER_PRODUCERS.brickRaster]>0,'Completed frame must contain actual raster mesh pixels');
  console.log(JSON.stringify({phase:'raster-pixel-proof',pending,counts}));
  pixels.unmap();
 }finally{pixels.destroy();}
}
for(;frames<4000;frames++){
 const encoder=device.createCommandEncoder();const recorder=new GPUPassTimestampRecorder(device,128,"Depth-3 mesh probe");
 assert.ok(renderer.encode(recorder.instrument(encoder),target));renderer.copySurfaceMeshDiagnostics(encoder,readback);recorder.resolve(encoder);
 maxBatch_ms=Math.max(maxBatch_ms,await submit(encoder,recorder));await readback.mapAsync(GPUMapMode.READ);receipt=Array.from(new Uint32Array(readback.getMappedRange()));readback.unmap();
 if(frames%100===0)console.log(JSON.stringify({phase:'mesh',frames,cursor:receipt[14],quads:receipt[4],ready:receipt[13],overflow:receipt[5]}));
 // Steady-state probe: startup may show partial raster geometry; wait for full publication.
 if(receipt[13]===1&&receipt[15]===0)break;
}
assert.ok(frames<4000,'mesh must complete');assert.ok(receipt[1]>0,'camera must draw quads');assert.equal(receipt[5],0);
renderer.setIdentityPlanesInspected(true);
for(let i=0;i<4;i++){const e=device.createCommandEncoder();assert.ok(renderer.encode(e,target));await submit(e);}
await assertRasterPixels(false);
renderer.setIdentityPlanesInspected(false);
// Pixel proof above verifies the published mesh; compiled but unused variants are recorded.
console.log(JSON.stringify({phase:'compiled-pipelines',renderPipelineLabels}));
console.log(JSON.stringify({phase:'complete',adapterInfo,depth,allocated,preparation_ms,total_ms:performance.now()-started,maintenanceBatches,frames,maxBatch_ms,receipt,validationErrors}));

const {mkdirSync,writeFileSync,readFileSync}=await import('node:fs');
const {createHash}=await import('node:crypto');
const {writeFramePng}=await import('./write-frame-png');
const directory=process.env.FLUID_EXPLORE_OUT??'artifacts/hero-render-halving-2026-10-04';
mkdirSync(directory,{recursive:true});
const renderWidth=1600,renderHeight=920,cycles=Number(process.env.FLUID_EXPLORE_CYCLES??12);
assert.ok(Number.isInteger(cycles)&&cycles>=3&&cycles<=120);
const output=device.createTexture({size:[renderWidth,renderHeight],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
renderer.ensureSize(renderWidth,renderHeight);
const tuning=resolveSvoSurfaceTuning(DEFAULT_SVO_RENDER_TUNING,smooth);
const arms=[
 {name:rasterAo?'raster-ao':'baseline',scale:0.5,ao:true,shadows:true},
 {name:'no-ao',scale:0.5,ao:false,shadows:true},
 {name:'no-shadows',scale:0.5,ao:true,shadows:false},
 {name:'no-visibility',scale:0.5,ao:false,shadows:false},
 {name:'visibility-off',scale:0.5,ao:false,shadows:false,off:true},
 {name:'quarter',scale:0.25,ao:true,shadows:true},
 {name:'eighth',scale:0.125,ao:true,shadows:true},
 {name:'quarter-bilateral',scale:0.25,ao:true,shadows:true,bilateral:true},
] as const;
const selectedArms=arms.filter(a=>!process.env.FLUID_EXPLORE_ARMS||process.env.FLUID_EXPLORE_ARMS.split(',').includes(a.name));
assert.ok(selectedArms.length);
const baseCamera={...defaultCamera,...preset.camera};
const views=[{name:'hero',camera:baseCamera},{name:'low',camera:{...baseCamera,elevation_rad:0.2,distance_m:baseCamera.distance_m*1.3}}];
const selectedViews=views.filter(v=>!process.env.FLUID_EXPLORE_VIEWS||process.env.FLUID_EXPLORE_VIEWS.split(',').includes(v.name));
const moving=process.env.FLUID_EXPLORE_MOVING==='1';
let frameCamera=baseCamera, motionFrame=0;
assert.ok(selectedViews.length);
const results:unknown[]=[];
const provenance=Object.fromEntries(['lib/svo/features/shading/program.ts','lib/svo/features/lighting-visibility/svo-raster-ao.ts','lib/svo/features/lighting-visibility/webgpu-svo-cone-fanout.ts','lib/svo/pipeline/webgpu-svo-dry-scene.ts','lib/svo/pipeline/svo-render-tuning.ts','lib/core/webgpu-renderer.ts', 'tools/explore-hero-render-budget.ts'].map(p=>[p,createHash('sha256').update(readFileSync(p)).digest('hex')]));
writeFileSync(`${directory}/provenance.json`,JSON.stringify({adapterInfo,depth,rasterAo,scene:id,surfaceStyle:scene.surfaceStyle,width:renderWidth,height:renderHeight,tuning,provenance,mesh:receipt},null,2));
const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
async function frame(measure:boolean){
 if(moving){
  const phase=motionFrame++*.035;
  const camera={...frameCamera,azimuth_rad:frameCamera.azimuth_rad+Math.sin(phase)*.35,
   target_m:{...frameCamera.target_m,x:frameCamera.target_m.x+Math.sin(phase*.7)*.15}};
  device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({scene,camera,cameraMoving:true,environmentId:scene.environment??'default',info:world.info,bodyCount:bodies.count,width:renderWidth,height:renderHeight}));
 }
 const start=performance.now(),encoder=device.createCommandEncoder();
 const recorder=measure?new GPUPassTimestampRecorder(device,128,'Hero render budget'):undefined;
 assert.ok(renderer.encode(recorder?recorder.instrument(encoder):encoder,output));
 recorder?.resolve(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
 const wall_ms=performance.now()-start;
 const reading=await recorder?.read();
 assert.deepEqual(validationErrors,[]);
 return {wall_ms,reading};
}
for(const view of selectedViews){
 frameCamera=view.camera;
 device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({scene,camera:view.camera,environmentId:scene.environment??'default',info:world.info,bodyCount:bodies.count,width:renderWidth,height:renderHeight}));
 for(let repetition=0;repetition<2;repetition++)for(const arm of repetition===0?selectedArms:[...selectedArms].reverse()){
  motionFrame=0;
  renderer.setRenderTuning({...tuning,coneRadianceReconstruction:'bilateral' in arm?'joint-bilateral':'full-res-relight'});
  renderer.setLightingOptions({...DEFAULT_SVO_LIGHTING_OPTIONS,coneLightingScale:arm.scale,ambientOcclusionEnabled:arm.ao,shadowsEnabled:arm.shadows,coneTracingMode:'off' in arm?'off':rasterAo?'raster-ao':'cones'});
  await renderer.ensureConeLightingPrepass();
  for(let i=0;i<12;i++)await frame(false);
  const samples=[];
  for(let i=0;i<cycles;i++)samples.push(await frame(true));
  const walls=[];for(let i=0;i<cycles;i++)walls.push((await frame(false)).wall_ms);
  const result={view:view.name,camera:view.camera,moving,arm,repetition,gpuMedian_ms:median(samples.map(s=>s.reading!.span_ms)),wallMedian_ms:median(walls),samples,walls,latticeActive:renderer.latticeVisibilityActive};
  results.push(result);writeFileSync(`${directory}/results.json`,JSON.stringify(results,null,2));
  console.log(JSON.stringify({phase:'render-budget',view:view.name,arm:arm.name,repetition,gpuMedian_ms:result.gpuMedian_ms,wallMedian_ms:result.wallMedian_ms}));
  if(repetition===0){
   const pitch=renderWidth*8, pixels=device.createBuffer({size:pitch*renderHeight,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:output},{buffer:pixels,bytesPerRow:pitch},[renderWidth,renderHeight]);
   device.queue.submit([encoder.finish()]);await pixels.mapAsync(GPUMapMode.READ);
   const bytes=new Uint8Array(pixels.getMappedRange());
   writeFileSync(`${directory}/${view.name}-${arm.name}.rgba16f`,bytes);
   writeFramePng(`${directory}/${view.name}-${arm.name}.png`,{width:renderWidth,height:renderHeight,packedRows:new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4)});
   pixels.unmap();pixels.destroy();
  }
 }
}
renderer.destroy();world.destroy();for(const b of [uniforms,body,readback])b.destroy();target.destroy();output.destroy();device.destroy();gcObserver.disconnect();
