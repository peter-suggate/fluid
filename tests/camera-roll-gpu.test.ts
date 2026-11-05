import test from 'node:test';import assert from 'node:assert/strict';
import {cameraApertureShaderLibrary,viewportRay,VIEW_UNIFORM_BYTES,VIEW_UNIFORM_FLOATS} from '../lib/core/webgpu-camera';
import {cameraBasis} from '../lib/core/math';
import type {CameraState} from '../lib/core/model';
import {withUniformDevice} from './helpers/uniform-geometric';
import {clothSurfaceWGSL} from '../lib/cloth/render';
import {surfaceRasterShader,surfaceWireframeShader,compositeShader} from '../lib/core/webgpu-water-pipeline';
import {gridOverlayShader} from '../lib/core/webgpu-grid-overlay';
import {secondaryParticleOpticalShader} from '../lib/core/webgpu-secondary-particles';
import {fluidOnlyRigidSceneShader} from '../lib/core/webgpu-rigid-raymarch';
import {spatialPerformanceWGSL} from '../lib/features/spatial-performance/overlay';
import {drySceneShader} from '../lib/svo/features/shading/program';
import {createSvoBrickRasterCullWGSL} from '../lib/svo/features/primary-visibility/webgpu-svo-brick-raster';
import {createSvoPrimaryEntryPrepassWGSL} from '../lib/svo/features/primary-visibility/webgpu-svo-primary-entry-prepass';
import {svoRigidRasterShader} from '../lib/svo/features/primary-visibility/webgpu-svo-rigid-raster';

test('all production camera shader modules compile the appended orientation ABI',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('camera orientation ABI',async device=>{
  for(const[label,code]of Object.entries({cloth:clothSurfaceWGSL,waterRaster:surfaceRasterShader,waterWire:surfaceWireframeShader,waterComposite:compositeShader,grid:gridOverlayShader,secondary:secondaryParticleOpticalShader,rigid:fluidOnlyRigidSceneShader,performance:spatialPerformanceWGSL,svo:drySceneShader,svoCull:createSvoBrickRasterCullWGSL({reversedZNear_m:.001}),svoEntry:createSvoPrimaryEntryPrepassWGSL({reversedZNear_m:.001}),svoRigid:svoRigidRasterShader})){
   const info=await device.createShaderModule({label,code}).getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error').map(m=>`${m.lineNum}:${m.message}`),[],label);
  }
 });
});
test('actual GPU rolled camera rays match shared CPU picking including polar views and legacy zero basis',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('rolled camera projection',async device=>{
  const code=`struct View{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,padding:array<vec4f,23>,cameraBasisRight:vec4f}
@group(0) @binding(0)var<uniform>u:View;@group(0) @binding(1)var<storage,read_write>out:array<vec4f>;
${cameraApertureShaderLibrary('u')}
@compute @workgroup_size(1)fn main(){let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=cameraRightDirection(f);let up=normalize(cross(r,f));out[0]=vec4f(normalize(f+r*.37*u.viewport.x/u.viewport.y*cameraTanHalfFov()+up*(-.28)*cameraTanHalfFov()),0);out[1]=vec4f(r,0);out[2]=vec4f(up,0);}`;
  const module=device.createShaderModule({code}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),uniform=device.createBuffer({size:VIEW_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:output}}]});
  try{for(const[roll,elevation,legacy]of [[0,.4,true],[.9,.4,false],[-1.3,-.7,false],[.5,Math.PI/2,false],[.8,-Math.PI/2,false]] as const){
   const camera:CameraState={azimuth_rad:.73,elevation_rad:elevation,roll_rad:roll,distance_m:2,target_m:{x:.2,y:.3,z:-.4},tanHalfFov:.5},basis=cameraBasis(camera),words=new Float32Array(VIEW_UNIFORM_FLOATS);words.set([800,600,0,0,basis.position.x,basis.position.y,basis.position.z,.5,.2,.3,-.4,1]);if(!legacy)words.set([basis.right.x,basis.right.y,basis.right.z,0],104);device.queue.writeBuffer(uniform,0,words);
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();e.copyBufferToBuffer(output,0,read,0,48);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()),expected=viewportRay(camera,.37,-.28,4/3).direction;for(const[a,name]of(['x','y','z'] as const).entries()){assert.ok(Math.abs(values[a]!-expected[name])<2e-6);assert.ok(Math.abs(values[4+a]!-basis.right[name])<2e-6);assert.ok(Math.abs(values[8+a]!-basis.up[name])<2e-6);}read.unmap();
  }}finally{if(read.mapState==='mapped')read.unmap();read.destroy();output.destroy();uniform.destroy();}
 });
});
