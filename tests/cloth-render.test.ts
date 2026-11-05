import test from 'node:test';
import assert from 'node:assert/strict';
import { ClothSurfaceRenderer, packClothSurface } from '../lib/cloth/render';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { pickClothSurface } from '../lib/cloth/geometry';
import type { ClothDocument, ClothSurfacePublication } from '../lib/cloth/types';
import { withUniformDevice } from './helpers/uniform-geometric';

const doc:ClothDocument={schemaVersion:'1.0.0',id:'render-check',materials:[{...DEFAULT_CLOTH_MATERIAL,rodRadius_m:.1,baseColor:[1,.5,.25]}],geometry:{kind:'inline',data:{
 restPositions_m:[-1,0,0,1,0,0,-1,2,0,1,2,0,-1,1,1,1,1,1],triangles:[0,1,2,1,3,2],rodOffsets:[0,2],rodIndices:[4,5]}}};
if(doc.geometry.kind!=='inline')throw new Error('Inline fixture');
const topology=doc.geometry.data;
const surface:ClothSurfacePublication={documentId:doc.id,topology,positions_m:new Float64Array(topology.restPositions_m),waterVolume_m3:new Float64Array(6),saturation:new Float32Array(6),generation:1,tick:0,time_s:0,topologyRevision:0};
test('render buffers preserve physical shell and capsule geometry and use published saturation',()=>{
 const {shells,rods}=packClothSurface(doc,surface);assert.equal(shells.length,6*12);assert.equal(rods.length,12);
 assert.deepEqual([...rods.slice(0,3)],[-1,1,1]);assert.ok(Math.abs(rods[3]!-.1)<1e-7);
 assert.deepEqual([...shells.slice(8,11)],[1,.5,.25]);
 const wet=packClothSurface(doc,{...surface,saturation:new Float32Array(6).fill(1)});
 assert.ok(wet.shells[8]!<shells[8]!);assert.deepEqual([...wet.shells.slice(0,3)],[...shells.slice(0,3)]);
});

test('source preview uses per-vertex water/control-volume colors and group yarn palette without changing geometry',()=>{
 const preview={...doc,appearance:'source-preview' as const},topologyWithGroups={...topology,groups:[0,0,0,0,1,2]},published={...surface,topology:topologyWithGroups,
  waterVolume_m3:Float64Array.from([0,.25,.5,0,0,1]),controlVolume_m3:new Float64Array(6).fill(1),saturation:new Float32Array(6).fill(1)};
 const dry=packClothSurface(doc,surface),data=packClothSurface(preview,published);
 assert.deepEqual([...data.shells.slice(8,12)],[.75,.75,.75,0]);
 assert.deepEqual([...data.shells.slice(20,24)],[.5625,.5625,.8125,0]);
 assert.deepEqual([...data.shells.slice(32,36)],[.375,.375,.875,0]);
 assert.ok(Math.abs(data.rodPlanes[8]!-1/3)<1e-7);assert.deepEqual([...data.rodPlanes.slice(12,15)],[0,0,1]);
 assert.deepEqual([...data.shells.slice(0,8)],[...dry.shells.slice(0,8)]);
 assert.deepEqual([...data.rods.slice(0,8)],[...dry.rods.slice(0,8)]);
 assert.deepEqual([...data.rodPlanes.slice(0,8)],[...dry.rodPlanes.slice(0,8)]);
 assert.throws(()=>packClothSurface(preview,{...published,controlVolume_m3:undefined}),/control volume/);
});

test('source appearance survives shared shell edits and scene save/load/undo',async()=>{
 const {cutClothShellBox}=await import('../lib/cloth/edit'),{cloneScene,defaultScene,serializeScene,parseScene}=await import('../lib/core/model'),{validateClothDocument}=await import('../lib/cloth/topology');
 const preview={...doc,appearance:'source-preview' as const},scene=cloneScene(defaultScene);scene.cloth=[preview];
 const cut=cutClothShellBox(preview,topology,surface.positions_m,{minimum_m:[-.2,.2,-.1],maximum_m:[.2,.8,.1]});
 assert.ok(cut.changed);assert.equal(cut.document.appearance,'source-preview');
 const edited={...scene,cloth:[cut.document]},saved=parseScene(serializeScene(edited));
 assert.equal(saved.cloth![0]!.appearance,'source-preview');assert.deepEqual(saved.cloth![0]!.materials,preview.materials);
 assert.equal(scene.cloth![0],preview,'undo retains the original material and appearance document');
 assert.ok(validateClothDocument({...preview,appearance:'unknown' as never}).includes('Unsupported cloth appearance'));
});

const circularDocument=doc;
const half=(bits:number)=>{const sign=(bits&0x8000)?-1:1,e=(bits>>>10)&31,m=bits&1023;return sign*(e===0?m*2**-24:e===31?Infinity:(1+m/1024)*2**(e-15));};
for(const radiusB_m of [.1,.04])test('GPU cloth depth matches shared nearest-surface query for yarn in front of shell (radiusB='+radiusB_m+')',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const doc={...circularDocument,materials:circularDocument.materials.map(m=>({...m,rodRadiusB_m:radiusB_m}))};
 await withUniformDevice('cloth physical surface',async device=>{
  const uniform=device.createBuffer({size:432,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const view=new Float32Array(108);view.set([32,32,0,0,0,1,3,1,0,1,0,1]);device.queue.writeBuffer(uniform,0,view);
  const renderer=new ClothSurfaceRenderer(device,uniform);await renderer.initialize();renderer.setSurfaces([doc],[surface]);
  const target=device.createTexture({size:[32,32],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const depth=device.createTexture({size:[32,32],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT});
  const read=device.createBuffer({size:256*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
   const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:target.createView(),clearValue:[0,0,0,0],loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:0,depthLoadOp:'clear',depthStoreOp:'store'}});pass.end();
   renderer.encode(encoder,target.createView(),depth.createView());encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[32,32]);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);const pixels=new Uint16Array(read.getMappedRange());
   const hit=pickClothSurface(doc,topology,surface.positions_m,{origin:[0,1,3],direction:[1/32,-1/32,-1]});
   assert.equal(hit?.kind,'rod');const rendered=half(pixels[16*128+16*4+3]!);
   assert.ok(Math.abs(rendered-hit.distance_m)<.004,`${rendered} vs ${hit.distance_m}`);
   assert.ok(rendered>0);
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();target.destroy();depth.destroy();renderer.destroy();uniform.destroy();}
 });
});

for(const radiusB_m of [.1,.04])test('GPU yarn cut cap and open gap match the shared editable geometry (radiusB='+radiusB_m+')',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const doc={...circularDocument,materials:circularDocument.materials.map(m=>({...m,rodRadiusB_m:radiusB_m}))};
 const {cutClothShellBox}=await import('../lib/cloth/edit');
 const cut=cutClothShellBox(doc,topology,surface.positions_m,{minimum_m:[-.2,.5,.5],maximum_m:[.2,1.5,1.5]});
 const published:ClothSurfacePublication={...surface,topology:cut.topology,positions_m:Float64Array.from(cut.positions_m),waterVolume_m3:new Float64Array(cut.positions_m.length/3),saturation:new Float32Array(cut.positions_m.length/3),topologyRevision:1};
 await withUniformDevice('cloth editable cut cap',async device=>{
  const uniform=device.createBuffer({size:432,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const renderer=new ClothSurfaceRenderer(device,uniform);await renderer.initialize();renderer.setSurfaces([cut.document],[published]);
  const target=device.createTexture({size:[32,32],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const depth=device.createTexture({size:[32,32],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT});
  const read=device.createBuffer({size:256*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   for(const view of [
    {eye:[0,1,1],target:[1,1,1],ray:[1,-1/32,1/32],kind:'rod'},
    {eye:[0,1,3],target:[0,1,0],ray:[1/32,-1/32,-1],kind:'triangle'},
   ] as const){
    const values=new Float32Array(108);values.set([32,32,0,0,...view.eye,1,...view.target,1]);device.queue.writeBuffer(uniform,0,values);
    const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:target.createView(),clearValue:[0,0,0,0],loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:0,depthLoadOp:'clear',depthStoreOp:'store'}});pass.end();
    renderer.encode(encoder,target.createView(),depth.createView());encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[32,32]);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const pixels=new Uint16Array(read.getMappedRange()),actual=half(pixels[16*128+16*4+3]!);
    const hit=pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:view.eye,direction:view.ray});assert.equal(hit?.kind,view.kind);
    assert.ok(Math.abs(actual-hit!.distance_m)<.004,`${view.kind}: rendered ${actual}, picked ${hit!.distance_m}`);read.unmap();
   }
  }finally{if(read.mapState==='mapped')read.unmap();read.destroy();target.destroy();depth.destroy();renderer.destroy();uniform.destroy();}
 });
});

for(const radiusB_m of [.1,.04])test('rolled cloth raster and capsule rays agree with nearest editable surfaces across the image (radiusB='+radiusB_m+')',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const doc={...circularDocument,materials:circularDocument.materials.map(m=>({...m,rodRadiusB_m:radiusB_m}))};
 await withUniformDevice('rolled editable cloth surface',async device=>{
  const uniform=device.createBuffer({size:432,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),view=new Float32Array(108);view.set([32,32,0,0,0,1,3,1,0,1,0,1]);
  const renderer=new ClothSurfaceRenderer(device,uniform);await renderer.initialize();renderer.setSurfaces([doc],[surface]);const target=device.createTexture({size:[32,32],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}),depth=device.createTexture({size:[32,32],format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT}),read=device.createBuffer({size:256*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{for(const roll of [.67,Math.PI/2]){const c=Math.cos(roll),s=Math.sin(roll);view.set([c,-s,0,0],104);device.queue.writeBuffer(uniform,0,view);
   const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{view:target.createView(),clearValue:[0,0,0,0],loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:0,depthLoadOp:'clear',depthStoreOp:'store'}});pass.end();renderer.encode(encoder,target.createView(),depth.createView());encoder.copyTextureToBuffer({texture:target},{buffer:read,bytesPerRow:256},[32,32]);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const pixels=new Uint16Array(read.getMappedRange());let hits=0;
   for(let y=8;y<=24;y+=4)for(let x=8;x<=24;x+=4){const nx=2*(x+.5)/32-1,ny=1-2*(y+.5)/32,hit=pickClothSurface(doc,topology,surface.positions_m,{origin:[0,1,3],direction:[c*nx+s*ny,-s*nx+c*ny,-1]}),rendered=half(pixels[y*128+x*4+3]!);if(hit){hits++;assert.ok(Math.abs(rendered-hit.distance_m)<.004,`roll ${roll} pixel ${x},${y}: ${rendered} vs ${hit.distance_m}`);}else assert.equal(rendered,0);}
   assert.ok(hits>0);read.unmap();
  }}finally{if(read.mapState==='mapped')read.unmap();read.destroy();target.destroy();depth.destroy();renderer.destroy();uniform.destroy();}
 });
});
