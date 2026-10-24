import assert from "node:assert/strict";
import test from "node:test";
import { UniformSurfaceWorkControl } from "../lib/methods/uniform/uniform-surface-work-control";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { createUniformMixedLayout, uniformMixedAllCoarseLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformDetailStorage } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformLatticeAddress, uniformTileCoord, uniformTileKey } from "../lib/methods/uniform/uniform-lattice-address";
import { UniformMixedSolid, uniformSolidScratchLayout } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { UNIFORM_SURFACE_SUPPORT_WORDS as S } from "../lib/methods/uniform/uniform-surface-transport.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

// Write the production coarse record ABI directly: the record builder encodes
// a cell as open iff its canonical h-cell fraction is strictly above 1e-5.
// This isolates current support reduction from voxel/terrain construction.
test("production solid support is local, strict, and refreshed each frame",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("local solid support",async device=>{
  const dimensions:[number,number,number]=[64,64,64],address=uniformLatticeAddress(dimensions);
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,undefined,address);
  const ownership=new UniformMixedOwnership(device,layout,true,0),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const phi=storage.createField("solid support phi","vertex","r32float"),texture=storage.physical(phi);
  const params=device.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const scratch=device.createBuffer({size:4*(uniformSolidScratchLayout().cutMapOffsetWords+layout.tiles.length),usage:GPUBufferUsage.STORAGE});
  const bodies=device.createBuffer({size:1536,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const terrain=device.createTexture({size:[1,1],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});
  const solid=new UniformMixedSolid(device,{params,scratch,bodies,terrain,coupledTiles:0,paramsWritten:()=>true},uniformMixedAllCoarseLayout(layout));
  const control=new UniformSurfaceWorkControl(device,ownership,phi,solid);
  const settings=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const record=solid.coarse!;
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> rows:array<vec4f>;
@group(0) @binding(1) var<uniform> setting:vec4u;
@compute @workgroup_size(64) fn fill(@builtin(global_invocation_id) id:vec3u){
 let t=id.x;if(t>=${layout.tiles.length}u){return;}
 var bits=vec4u(65535u);
 // One selected canonical cell (31) crosses the exact production threshold.
 if(t==setting.x&&!(bitcast<f32>(setting.y)>1e-5)){bits.y=32767u;}
 rows[${record.count+layout.tiles.length}u+3u*t]=vec4f(bits);
}`});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"fill"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:record.record}},{binding:1,resource:{buffer:settings}}]});
  const read=device.createBuffer({size:4*64*S*control.regions,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const regionAt=(x:number,y:number,z:number)=>{
   for(let r=0;r<control.regions;r++){
    const t=512*(r>>3)+4*(r&1)+32*((r>>1)&1)+256*((r>>2)&1);
    const c=uniformTileCoord(address,t,[0,0,0]);if(c[0]*4===x&&c[1]*4===y&&c[2]*4===z)return r;
   }throw new Error("region missing");
  };
  const home=regionAt(16,16,16),remote=regionAt(48,48,48),boundary=regionAt(0,0,0);
  const nearTile=uniformTileKey(address,8,4,4),remoteTile=uniformTileKey(address,12,12,12);
  const capture=async(tile:number,fraction:number)=>{
   const f=new Float32Array([fraction]);device.queue.writeBuffer(settings,0,new Uint32Array([tile,new Uint32Array(f.buffer)[0]!,0,0]));
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(layout.tiles.length/64));p.end();
   control.encode(e,"surface");e.copyBufferToBuffer(control.surface,4*control.supportWordOffset,read,0,read.size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange()).slice();read.unmap();
   // Check every rectangle, not just the examples below: no stale clear bit
   // may survive at any old support extent when the solid moves or disappears.
   const solidCoord=tile===0xffffffff?undefined:uniformTileCoord(address,tile,[0,0,0]).map(v=>Math.floor(v/4));
   for(let r=0;r<control.regions;r++){
    const key=512*(r>>3)+4*(r&1)+32*((r>>1)&1)+256*((r>>2)&1);
    const center=uniformTileCoord(address,key,[0,0,0]).map(v=>v/4);
    for(let box=0;box<64;box++){
     let valid=true,contains=true;
     for(let axis=0;axis<3;axis++){
      const code=(box>>(2*axis))&3,low=center[axis]!-((code&1)?1:0),high=center[axis]!+((code&2)?1:0);
      valid=valid&&low>=0&&high<4;
      contains=contains&&!!solidCoord&&solidCoord[axis]!>=low&&solidCoord[axis]!<=high;
     }
     const closed=contains&&!(Math.fround(fraction)>Math.fround(1e-5));
     assert.equal(data[S*(64*r+box)+7],valid?(closed?1:3):0,`region ${r}, support ${box}, fraction ${fraction}`);
    }
   }
   return (r:number,box:number)=>data[S*(64*r+box)+7]!;
  };
  try{
   await control.initialize();device.queue.writeTexture({texture},new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers).fill(1),{bytesPerRow:4*texture.width,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
   control.invalidate();
   device.queue.writeBuffer(ownership.regionVelocity,0,new Uint32Array(6*control.regions).fill(0x80000000));
   let flag=await capture(remoteTile,0);assert.equal(flag(home,0),3,"distant solid leaves home clear despite global solid presence");assert.equal(flag(remote,0),1);
   flag=await capture(nearTile,1e-5);assert.equal(flag(home,0),3);assert.equal(flag(home,2),1,"positive-x support includes the closed canonical cell");assert.equal(flag(remote,0),3,"moving solid clears its previous region");
   flag=await capture(nearTile,1.0001e-5);assert.equal(flag(home,2),3,"strictly above threshold is open");
   flag=await capture(nearTile,0);assert.equal(flag(home,2),1);
   flag=await capture(0xffffffff,1);assert.equal(flag(home,2),3,"removal clears all previously affected summaries next frame");
   assert.equal(flag(boundary,0),3,"stored boundary home remains usable");assert.equal(flag(boundary,1),0,"missing negative page cannot certify clear support");
  }finally{read.destroy();settings.destroy();control.destroy();solid.destroy();params.destroy();scratch.destroy();bodies.destroy();terrain.destroy();storage.destroy();ownership.destroy();}
 });
});
