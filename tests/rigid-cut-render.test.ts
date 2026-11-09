import test from 'node:test';
import assert from 'node:assert/strict';
import type {CameraState,RigidBodyDescription} from '../lib/core/model';
import {cameraBasis} from '../lib/core/math';
import {viewportRayForPixel,VIEW_UNIFORM_BYTES,VIEW_UNIFORM_FLOATS} from '../lib/core/webgpu-camera';
import {sceneShape} from '../lib/core/scene-shape';
import {RigidCutRenderAtlas} from '../lib/core/rigid-cut-render';
import {intersectCutSphere,intersectRoundedPlate} from '../lib/core/rounded-plate-ray';
import {svoRigidRasterShader,svoRigidRasterInputBindGroupLayoutEntries} from '../lib/svo/features/primary-visibility/webgpu-svo-rigid-raster';
import {SVO_DRY_RIGID_MOTION_UNIFORM_BYTES} from '../lib/svo/pipeline/webgpu-svo-dry-scene';
import {GPU_RIGID_BODY_CAPACITY} from '../lib/core/rigid-capacity';
import {withUniformDevice} from './helpers/uniform-geometric';

test('actual moving-rigid raster shares rounded clamp cuts and nearest visible surfaces',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('editable moving rounded clamp render',async device=>{
  const width=64,height=64,uniform=device.createBuffer({size:VIEW_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),bodies=device.createBuffer({size:64*GPU_RIGID_BODY_CAPACITY,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),motion=device.createBuffer({size:SVO_DRY_RIGID_MOTION_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM});
  const atlas=new RigidCutRenderAtlas(device),textures=(['rgba32uint','rgba16uint','rg32uint'] as const).map(format=>device.createTexture({size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC})),depth=device.createTexture({size:[width,height],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT}),read=device.createBuffer({size:width*height*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:svoRigidRasterShader}),layout=device.createBindGroupLayout({entries:svoRigidRasterInputBindGroupLayoutEntries()}),pipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'rigidRasterVertex'},fragment:{module,entryPoint:'rigidRasterFragment',targets:[{format:'rgba32uint'},{format:'rgba16uint'},{format:'rg32uint'}]},primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthCompare:'greater',depthWriteEnabled:true}});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:bodies}},{binding:14,resource:{buffer:motion}},{binding:15,resource:atlas.view}]});
   for(const cut of [false,true])for(const moved of [false,true]){
    const body:RigidBodyDescription={id:'clamp',name:'Clamp',shape:'rounded-plate',motion:'static',kinematic:true,dimensions_m:{x:.08,y:.003,z:.04},density_kg_m3:1000,
     position_m:{x:moved?.02:0,y:moved?.01:0,z:0},orientation:{x:0,y:Math.sin(.31),z:0,w:Math.cos(.31)},linearVelocity_m_s:{x:0,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},restitution:0,friction:.3,
     cuts:cut?[{center_m:{x:0,y:0,z:0},halfExtents_m:{x:.015,y:.02,z:.08},orientation:{x:0,y:0,z:0,w:1}}]:[]};
    const camera:CameraState={azimuth_rad:.2,elevation_rad:.8,roll_rad:.37,distance_m:.4,target_m:body.position_m,tanHalfFov:.4},basis=cameraBasis(camera),view=new Float32Array(VIEW_UNIFORM_FLOATS);
    view.set([width,height,0,0,basis.position.x,basis.position.y,basis.position.z,.4,body.position_m.x,body.position_m.y,body.position_m.z,1]);view[18]=1;view.set([basis.right.x,basis.right.y,basis.right.z,0],104);device.queue.writeBuffer(uniform,0,view);
    const shape=sceneShape(body.shape),half=shape.renderHalfExtent_m(body.dimensions_m),q=body.orientation;
    device.queue.writeBuffer(bodies,0,new Float32Array([body.position_m.x,body.position_m.y,body.position_m.z,shape.boundingRadius_m(body.dimensions_m),...half,10,q.w,q.x,q.y,q.z,.24,.26,.29,0]));atlas.setBodies([body]);
    const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:textures.map(texture=>({view:texture.createView(),clearValue:[0,0,0,0],loadOp:'clear' as const,storeOp:'store' as const})),depthStencilAttachment:{view:depth.createView(),depthClearValue:0,depthLoadOp:'clear',depthStoreOp:'store'}});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(36,1);pass.end();encoder.copyTextureToBuffer({texture:textures[2]!},{buffer:read,bytesPerRow:width*8},[width,height]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(read.getMappedRange());let hits=0,misses=0;
    for(let y=8;y<56;y+=3)for(let x=8;x<56;x+=3){const ray=viewportRayForPixel(camera,x,y,width,height),expected=intersectRoundedPlate(body,ray,1),actual=values[2*(x+width*y)]!;
     if(expected){hits++;assert.ok(Math.abs(actual-expected.t_m)<3e-6,`${cut}/${moved} pixel ${x},${y}: ${actual} vs ${expected.t_m}`);}else{misses++;assert.equal(actual,0,`hidden uncut surface at ${x},${y}`);}
    }assert.ok(hits>20&&misses>20);read.unmap();
   }
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();for(const t of textures)t.destroy();depth.destroy();atlas.destroy();uniform.destroy();bodies.destroy();motion.destroy();}
 });
});

test('a sphere with local cuts rasters the same nearest visible surface as the shared picking oracle; uncut it keeps its closed form',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('editable moving cut sphere render',async device=>{
  const width=64,height=64,uniform=device.createBuffer({size:VIEW_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),bodies=device.createBuffer({size:64*GPU_RIGID_BODY_CAPACITY,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),motion=device.createBuffer({size:SVO_DRY_RIGID_MOTION_UNIFORM_BYTES,usage:GPUBufferUsage.UNIFORM});
  const atlas=new RigidCutRenderAtlas(device),textures=(['rgba32uint','rgba16uint','rg32uint'] as const).map(format=>device.createTexture({size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC})),depth=device.createTexture({size:[width,height],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT}),read=device.createBuffer({size:width*height*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:svoRigidRasterShader}),layout=device.createBindGroupLayout({entries:svoRigidRasterInputBindGroupLayoutEntries()}),pipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'rigidRasterVertex'},fragment:{module,entryPoint:'rigidRasterFragment',targets:[{format:'rgba32uint'},{format:'rgba16uint'},{format:'rg32uint'}]},primitive:{topology:'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthCompare:'greater',depthWriteEnabled:true}});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:bodies}},{binding:14,resource:{buffer:motion}},{binding:15,resource:atlas.view}]});
   for(const cut of [false,true])for(const moved of [false,true]){
    const body:RigidBodyDescription={id:'ball',name:'Ball',shape:'sphere',motion:'static',dimensions_m:{x:.05,y:.05,z:.05},density_kg_m3:1000,
     position_m:{x:moved?.02:0,y:moved?.01:0,z:0},orientation:{x:0,y:Math.sin(.31),z:0,w:Math.cos(.31)},linearVelocity_m_s:{x:0,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},restitution:0,friction:.3,
     cuts:cut?[{center_m:{x:.04,y:.01,z:0},halfExtents_m:{x:.03,y:.02,z:.08},orientation:{x:0,y:Math.sin(.2),z:0,w:Math.cos(.2)}},{center_m:{x:0,y:.05,z:0},halfExtents_m:{x:.08,y:.025,z:.08},orientation:{x:0,y:0,z:0,w:1}}]:[]};
    const camera:CameraState={azimuth_rad:.2,elevation_rad:.8,roll_rad:.37,distance_m:.4,target_m:body.position_m,tanHalfFov:.4},basis=cameraBasis(camera),view=new Float32Array(VIEW_UNIFORM_FLOATS);
    view.set([width,height,0,0,basis.position.x,basis.position.y,basis.position.z,.4,body.position_m.x,body.position_m.y,body.position_m.z,1]);view[18]=1;view.set([basis.right.x,basis.right.y,basis.right.z,0],104);device.queue.writeBuffer(uniform,0,view);
    const shape=sceneShape(body.shape),half=shape.renderHalfExtent_m(body.dimensions_m),q=body.orientation;
    device.queue.writeBuffer(bodies,0,new Float32Array([body.position_m.x,body.position_m.y,body.position_m.z,shape.boundingRadius_m(body.dimensions_m),...half,0,q.w,q.x,q.y,q.z,.24,.26,.29,0]));atlas.setBodies([body]);
    const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:textures.map(texture=>({view:texture.createView(),clearValue:[0,0,0,0],loadOp:'clear' as const,storeOp:'store' as const})),depthStencilAttachment:{view:depth.createView(),depthClearValue:0,depthLoadOp:'clear',depthStoreOp:'store'}});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(36,1);pass.end();encoder.copyTextureToBuffer({texture:textures[2]!},{buffer:read,bytesPerRow:width*8},[width,height]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(read.getMappedRange());let hits=0,misses=0,carved=0;
    for(let y=8;y<56;y+=3)for(let x=8;x<56;x+=3){const ray=viewportRayForPixel(camera,x,y,width,height),expected=intersectCutSphere(body,ray,1),actual=values[2*(x+width*y)]!;if(expected?.t_m!==intersectCutSphere({...body,cuts:[]},ray,1)?.t_m)carved++;
     if(expected){hits++;assert.ok(Math.abs(actual-expected.t_m)<3e-6,`${cut}/${moved} pixel ${x},${y}: ${actual} vs ${expected.t_m}`);}else{misses++;assert.equal(actual,0,`hidden uncut surface at ${x},${y}`);}
    }assert.ok(hits>20&&misses>20);assert.ok(cut?carved>20:carved===0,`${carved} carved pixels`);read.unmap();
   }
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();for(const t of textures)t.destroy();depth.destroy();atlas.destroy();uniform.destroy();bodies.destroy();motion.destroy();}
 });
});
