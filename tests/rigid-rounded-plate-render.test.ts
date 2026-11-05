import test from 'node:test';
import assert from 'node:assert/strict';
import {fluidOnlyRigidSceneShader} from '../lib/core/webgpu-rigid-raymarch';
import {RigidCutRenderAtlas,ROUNDED_PLATE_RENDER_TAG} from '../lib/core/rigid-cut-render';
import {GPU_RIGID_BODY_CAPACITY} from '../lib/core/rigid-capacity';
import {VIEW_UNIFORM_BYTES,VIEW_UNIFORM_FLOATS} from '../lib/core/webgpu-camera';
import {WATER_SCENE_OPTICS_BYTES} from '../lib/core/webgpu-lighting';
import {withUniformDevice} from './helpers/uniform-geometric';
import type {SolidBoxCut} from '../lib/core/solid-cuts';

test('actual fluid-only rigid raster preserves rounded plate rims, local holes, rotation and undo',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('rounded plate display parity',async device=>{
  const atlas=new RigidCutRenderAtlas(device),view=device.createBuffer({size:VIEW_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),bodies=device.createBuffer({size:GPU_RIGID_BODY_CAPACITY*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),optics=device.createBuffer({size:WATER_SCENE_OPTICS_BYTES,usage:GPUBufferUsage.UNIFORM}),receiver=device.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING}),target=device.createTexture({size:[1,1],format:'rgba32float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}),read=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const cut:SolidBoxCut={center_m:{x:0,y:0,z:0},halfExtents_m:{x:.02,y:.02,z:.02},orientation:{x:0,y:0,z:0,w:1}};
  try{
   const module=device.createShaderModule({code:fluidOnlyRigidSceneShader});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},{binding:1,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}}]}),pipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'vertexMain'},fragment:{module,entryPoint:'fragmentMain',targets:[{format:'rgba32float'}]},primitive:{topology:'triangle-list'}}),group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:view}},{binding:1,resource:{buffer:bodies}},{binding:2,resource:{buffer:optics}},{binding:3,resource:receiver.createView()},{binding:4,resource:atlas.view}]});
   const render=async(x:number,cuts:readonly SolidBoxCut[],rotated=false,grazing=false)=>{
    const words=new Float32Array(VIEW_UNIFORM_FLOATS),grazingY=.2+.005*(1-1e-5),ro=grazing?[0,grazingY,.2]:rotated?[.3,-x,0]:[x,.3,0],to=grazing?[0,grazingY,0]:rotated?[.2,-x,0]:[x,.2,0];words.set([1,1,0,0,...ro,.5,...to,1]);words[18]=1;words.set(rotated?[0,-1,0,0]:[1,0,0,0],104);device.queue.writeBuffer(view,0,words);
    const body=new Float32Array(GPU_RIGID_BODY_CAPACITY*16);body.set([...(rotated?[.2,0,0]:[0,.2,0]),.14,.085,.005,.105,ROUNDED_PLATE_RENDER_TAG,...(rotated?[Math.SQRT1_2,0,0,-Math.SQRT1_2]:[1,0,0,0]),.24,.26,.29,0]);device.queue.writeBuffer(bodies,0,body);atlas.setBodies([{cuts}]);
    const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:target.createView(),clearValue:{r:0,g:0,b:0,a:123},loadOp:'clear',storeOp:'store'}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[1,1]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const depth=new Float32Array(read.getMappedRange())[3]!;read.unmap();return depth;
   };
   const before=await render(0,[]);assert.ok(Math.abs(before-.095)<2e-7);
   const outside=await render(.06,[]);assert.equal(await render(0,[cut]),123);assert.equal(await render(.06,[cut]),outside);
   assert.ok(Math.abs(await render(.083,[])-.096)<3e-7,'curved rim retains the authored sphere radius');
   assert.equal(await render(0,[cut],true),123);assert.ok(Math.abs(await render(.06,[cut],true)-outside)<2e-7,'cuts follow the body-local rotated pose');assert.equal(await render(0,[]),before);
   const y=Math.fround(Math.fround(.2+.005*(1-1e-5))-Math.fround(.2)),r=Math.fround(.005),core=Math.fround(Math.fround(.105)-r),expected=Math.fround(.2)-core-Math.sqrt(r*r-y*y);
   assert.ok(Math.abs(await render(0,[],false,true)-expected)<2e-6,'grazing finite cylinder intersection has no raymarch iteration cutoff');
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();target.destroy();receiver.destroy();optics.destroy();bodies.destroy();view.destroy();atlas.destroy();}
 });
});
