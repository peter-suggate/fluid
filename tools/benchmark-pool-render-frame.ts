/** Production rendering with an authored, synthetic dense initial water field.
 * This exercises the dense fallback, not the live solver's mixed surface windows.
 * Forced surface updates are a stress test, not evidence of live water cost.
 * Physics stepping is deliberately outside the render-frame measurement. */
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import type { CameraState, SceneDescription } from '../lib/core/model';
import type { EnvironmentId } from '../lib/core/environments';
import { RasterWaterPipeline } from '../lib/core/webgpu-water-pipeline';
import { initialLiquidVertexPhi } from '../lib/core/initial-liquid-lattice';
import { sceneLatticeDimensions } from '../lib/core/scene-lattice-dimensions';
import { svoSceneLighting } from '../lib/svo/features/lighting-visibility/svo-dry-scene-lighting';
import type { SparseVoxelDrySceneRenderer } from '../lib/svo/pipeline/webgpu-svo-dry-scene';
import { GPUPassTimestampRecorder, type GPUPassTimestampReading } from '../lib/core/performance-trace';
import { packSvoDryViewUniforms } from './svo-dry-frame-harness';

export async function benchmarkPoolRenderFrame(input: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; scene: SceneDescription; camera: CameraState;
  uniformBuffer: GPUBuffer; bodyBuffer: GPUBuffer; bodyCount: number;
  width: number; height: number; environmentId: EnvironmentId; capturePath?: string;
}) {
  const {device,renderer,scene,camera,uniformBuffer,bodyBuffer,width,height}=input;
  const [nx,ny,nz]=sceneLatticeDimensions(scene), c=scene.container;
  const cellSize_m:[number,number,number]=[c.width_m/nx,c.height_m/ny,c.depth_m/nz];
  console.log('Preparing synthetic dense initial water (not the live mixed-window solver source)');
  const phi=initialLiquidVertexPhi(scene,[nx,ny,nz]);
  const field=device.createTexture({label:'Benchmark authored initial liquid phi',dimension:'3d',size:[nx+1,ny+1,nz+1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  device.queue.writeTexture({texture:field},new Float32Array(phi),{bytesPerRow:(nx+1)*4,rowsPerImage:ny+1},[nx+1,ny+1,nz+1]);
  const unusedOpen=device.createTexture({label:'Unused water open fraction',dimension:'3d',size:[1,1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING});
  const target=device.createTexture({label:'Complete pool render',size:[width,height],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const water=new RasterWaterPipeline(device,'rgba8unorm',uniformBuffer,bodyBuffer);
  await water.initialize();water.ensureSize(width,height);water.setSceneHasFluid(true);water.setRigidBodyCount(0);
  water.setFluidDomain({origin_m:[-c.width_m/2,0,-c.depth_m/2],cellSize_m,dimensions:[nx,ny,nz]});
  water.setVolume(field,{vertexPhi:field,contourVertexPhi:true,openFraction:unusedOpen,cellSize_m});
  const lighting=svoSceneLighting(scene);
  water.setSceneOptics({optics:scene.fluid.optics,directional:lighting.directional,grade:lighting.grade,terrain:scene.terrain,container:{width_m:c.width_m,depth_m:c.depth_m}});
  const quantile=(values:number[],q:number)=>[...values].sort((a,b)=>a-b)[Math.min(values.length-1,Math.floor(values.length*q))]!;
  try {
    for(const motion of ['stationary','camera','surface-update','camera-and-surface-update'] as const){
      const readings:GPUPassTimestampReading[]=[];
      for(let frame=0;frame<32;frame++){
        const moving=motion==='camera'||motion==='camera-and-surface-update';
        const updating=motion==='surface-update'||motion==='camera-and-surface-update';
        const view=moving?{...camera,azimuth_rad:camera.azimuth_rad+.12*Math.sin(frame*.13)}:camera;
        device.queue.writeBuffer(uniformBuffer,0,packSvoDryViewUniforms({scene,camera:view,environmentId:input.environmentId,info:{nx,ny,nz},bodyCount:input.bodyCount,width,height,cameraMoving:moving}));
        const encoder=device.createCommandEncoder({label:`Complete pool frame ${motion} ${frame}`});
        const recorder=new GPUPassTimestampRecorder(device,512,'Complete pool render');
        assert.ok(water.encode(recorder.instrument(encoder),target,nx,ny,nz,updating?frame+1:0,
          (commands,output)=>renderer.encode(commands,output),undefined,false,'require-dry-scene',false,undefined,true));
        assert.ok(recorder.resolve(encoder));device.queue.submit([encoder.finish()]);
        const reading=await recorder.read();if(reading&&frame>=8)readings.push(reading);
      }
      assert.ok(readings.length>=20,'complete pool frame timestamps must resolve');
      const labels=[...new Set(readings.flatMap(r=>r.passes.map(p=>p.label)))];
      console.log(`Complete pool render GPU: ${JSON.stringify({motion,resolution:{width,height},scope:'production rendering with synthetic dense water; surface-update forces dense fallback extraction, not live mixed-window solver work; excludes physics',median_ms:quantile(readings.map(r=>r.span_ms),.5),p95_ms:quantile(readings.map(r=>r.span_ms),.95),samples_ms:readings.map(r=>r.span_ms),passes:labels.map(label=>({label,median_ms:quantile(readings.map(r=>r.passes.find(p=>p.label===label)?.duration_ms??0),.5)}))})}`);
      if(motion==='stationary'&&input.capturePath){
        const pitch=Math.ceil(width*4/256)*256;
        const read=device.createBuffer({size:pitch*height,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
        const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:pitch},[width,height]);device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);const raw=new Uint8Array(read.getMappedRange()),rgba=new Uint8Array(width*height*4);
        for(let y=0;y<height;y++)rgba.set(raw.subarray(y*pitch,y*pitch+width*4),y*width*4);
        writeFileSync(input.capturePath,rgba);read.unmap();read.destroy();
      }
    }
  } finally {water.destroy();field.destroy();unusedOpen.destroy();target.destroy();}
}
