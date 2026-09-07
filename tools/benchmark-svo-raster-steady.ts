/** Measure the production raster renderer only after its complete mesh is published. */
import assert from "node:assert/strict";
import {writeFile} from "node:fs/promises";
import {GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import {DEFAULT_SVO_RENDER_TUNING} from "../lib/svo/pipeline/svo-render-tuning";
import type {SparseVoxelDrySceneRenderer} from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import {packSvoDryViewUniforms,packSvoDryRigidBodies,buildSvoDrySceneAssembly} from "./svo-dry-frame-harness";

export async function benchmarkRasterSteady(device:GPUDevice, renderer:SparseVoxelDrySceneRenderer,
 uniforms:GPUBuffer, view:Parameters<typeof packSvoDryViewUniforms>[0],
 runtime:{body:GPUBuffer;source:Parameters<typeof buildSvoDrySceneAssembly>[1]}) {
 const width=Number(process.env.FLUID_STEADY_WIDTH??1920),height=Number(process.env.FLUID_STEADY_HEIGHT??1080);
 const target=device.createTexture({size:[width,height],format:"rgba16float",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
 renderer.setRenderTuning(DEFAULT_SVO_RENDER_TUNING);renderer.ensureSize(width,height);
 device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({...view,width,height}));
 const movingScene={...view.scene,rigidBodies:[...view.scene.rigidBodies,{...view.scene.rigidBodies[0]!,id:"benchmark-moving-body",motion:"dynamic" as const,position_m:{x:0,y:0.3,z:0}}]};
 const motion=device.createBuffer({size:12*128,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 const motionWords=new Float32Array(12*32);motionWords[view.bodyCount*32+19]=0.01;
 device.queue.writeBuffer(motion,0,motionWords);
 const readback=device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
 try {
  for(const [scale,reconstruction] of [[0.5,"full-res-relight"]] as const) {
  renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,coneRadianceReconstruction:reconstruction,stableAoSamples:4,movingAoSamples:4,stableAreaLightSamples:2,movingAreaLightSamples:2});
  for(const motionCase of ["stationary","camera","body","camera-and-body"] as const) {
  const cameraMoving=motionCase==="camera"||motionCase==="camera-and-body";
  const bodyMoving=motionCase==="body"||motionCase==="camera-and-body";
  const activeScene=bodyMoving?movingScene:view.scene;
  const roster=packSvoDryRigidBodies(activeScene);
  renderer.publishScene(buildSvoDrySceneAssembly(activeScene,runtime.source).drySceneData);
  renderer.setRigidBodyCount(roster.count);renderer.setRigidMotionSource(bodyMoving?motion:undefined);
  device.queue.writeBuffer(runtime.body,0,roster.data);
  for(const [name,shadows,ao] of [["baseline",true,true]] as const){
   renderer.setLightingOptions({coneLightingScale:scale,globalIlluminationEnabled:false,coneTracingMode:"cones",shadowsEnabled:shadows,ambientOcclusionEnabled:ao});
   await renderer.ensureConeLightingPrepass();
   const samples:number[]=[];const frameSamples:number[]=[];let passReading;
   for(let frame=0;frame<28;frame++){
    const frameStarted=performance.now();
    if(bodyMoving){movingScene.rigidBodies.at(-1)!.position_m.x=0.08*Math.sin(frame*0.15);device.queue.writeBuffer(runtime.body,0,packSvoDryRigidBodies(movingScene).data);}
    device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({...view,scene:activeScene,bodyCount:roster.count,width,height,cameraMoving,
      camera:cameraMoving?{...view.camera,azimuth_rad:view.camera.azimuth_rad+0.25*Math.sin(frame*0.08),elevation_rad:view.camera.elevation_rad+0.1*Math.sin(frame*0.05)}:view.camera}));
    const encoder=device.createCommandEncoder();
    const recorder=frame===27?new GPUPassTimestampRecorder(device,128,"Steady raster attribution"):undefined;
    assert.ok(renderer.encode(recorder?.instrument(encoder)??encoder,target));
    if(frame===27){renderer.copySurfaceMeshDiagnostics(encoder,readback);recorder!.resolve(encoder);}
    const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    const elapsed=performance.now()-started;assert.ok(elapsed<500,`Safety stop: steady frame ${elapsed} ms`);
    if(frame>=4&&frame<27){samples.push(elapsed);frameSamples.push(performance.now()-frameStarted);}
    if(recorder)passReading=await recorder.read();
   }
   await readback.mapAsync(GPUMapMode.READ);const mesh=Array.from(new Uint32Array(readback.getMappedRange()));readback.unmap();
   assert.equal(mesh[13],1,"Steady timing requires a complete mesh");assert.equal(mesh[5],0);assert.equal(mesh[15],0);
   if(motionCase==="stationary") {
    const bytesPerRow=Math.ceil(width*8/256)*256;
    const capture=device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const copy=device.createCommandEncoder();copy.copyTextureToBuffer({texture:target},{buffer:capture,bytesPerRow},[width,height]);
    device.queue.submit([copy.finish()]);await capture.mapAsync(GPUMapMode.READ);
    await writeFile(`/tmp/fluid-steady-${scale}-${reconstruction}.rgba16f`,new Uint8Array(capture.getMappedRange()));capture.unmap();capture.destroy();
   }
   console.log(JSON.stringify({phase:"steady-raster",cameraIndependentQuality:true,name,scale,reconstruction,motionCase,cameraMoving,bodyMoving,width,height,median_ms:median(samples),frameMedian_ms:median(frameSamples),frameP95_ms:[...frameSamples].sort((a,b)=>a-b)[Math.ceil(frameSamples.length*0.95)-1],samples,frameSamples,mesh,passReading}));
  }
 }
 }
 } finally {renderer.setRigidMotionSource(undefined);motion.destroy();target.destroy();readback.destroy();}
}
