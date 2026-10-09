import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedForces } from "../lib/methods/uniform/uniform-mixed-forces";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("vector viscosity preserves scalar force evaluation across h/4h layouts and walls",{timeout:120_000},async()=>{
 await withUniformDevice("Vector force parity",async device=>{
  const n=32,tiles=(n/4)**3;
  const lattice={dimensions:[n,n,n] as const,cellSize_m:[.05,.1,.2] as const,origin_m:{x:0,y:0,z:0}};
  const ownership=new UniformMixedOwnership(device,createUniformMixedLayoutFromWidths(lattice,new Uint8Array(tiles).fill(1),[]));
  const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
  const texture=(size:number[],format:GPUTextureFormat)=>{const t=device.createTexture({size,dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});textures.push(t);return t;};
  const buffer=(size:number,uniform=false)=>{const b=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|(uniform?GPUBufferUsage.UNIFORM:0)});buffers.push(b);return b;};
  const advected=texture([n,n,n],"rgba32float"),phi=texture([n+1,n+1,n+1],"r32float"),volume=texture([n,n,n],"r32float"),centerPhi=texture([n,n,n],"r32float"),unitVelocity=texture([n,n,n],"rgba32float"),coarseVelocity=texture([n/4+2,n/4+2,n/4+2],"rgba32float");
  const upload=(t:GPUTexture,components:number,fn:(i:number,c:number)=>number)=>{
   const data=Float32Array.from({length:t.width*t.height*t.depthOrArrayLayers*components},(_,i)=>fn(Math.floor(i/components),i%components));
   device.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*components*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  };
  upload(advected,4,(i,c)=>c===3?0:Math.sin(.13*i+c));upload(unitVelocity,4,(i,c)=>c===3?0:Math.cos(.09*i-2*c));upload(coarseVelocity,4,(i,c)=>c===3?0:Math.sin(.19*i+3*c));
  upload(phi,1,i=>((Math.floor(i/(n+1))%(n+1))-n/2)*.1);upload(centerPhi,1,i=>((Math.floor(i/n)%n)-n/2)*.1);upload(volume,1,i=>i%11===0?.2:0);
  const negative=buffer(3*n*n*4),params=buffer(48,true);device.queue.writeBuffer(negative,0,Float32Array.from({length:3*n*n},(_,i)=>Math.sin(i*.1)));
  const actual=new UniformMixedForces(device,ownership,true),reference=new UniformMixedForces(device,ownership,true,undefined,undefined,false,false);
  try{
   await actual.initialize();await reference.initialize();
   const outputs=[texture([n,n,n],"rgba32float"),texture([n,n,n],"rgba32float")];const walls=[buffer(negative.size),buffer(negative.size)];
   const bind=(index:number)=>({advected,phi,volume,centerPhi,unitVelocity,coarseVelocity,negative,params,output:outputs[index]!,outputNegative:walls[index]!});
   const groups=[actual.bind(bind(0)),reference.bind(bind(1))];
   for(let pattern=0;pattern<6;pattern++){
    const widths=Uint8Array.from({length:tiles},(_,i)=>pattern===0?1:pattern===1?4:pattern===2?(i===0?1:4):pattern===3?(i===0?4:1):((i*13+Math.floor(i/8))%7<pattern-2?1:4));
    ownership.update(createUniformMixedLayoutFromWidths(lattice,widths,[]));
    for(let mode=0;mode<3;mode++){
     device.queue.writeBuffer(params,0,new Float32Array([.05,.1,.2,1/30,-9.81,998.2,mode===0?0:mode===1?.001002:1,0,mode===2?1:0,mode===1?1:0,0,0]));
     const encoder=device.createCommandEncoder();actual.encode(encoder,groups[0]!,false);reference.encode(encoder,groups[1]!,false);device.queue.submit([encoder.finish()]);
     const a=await readMixedTexture(device,outputs[0]!),b=await readMixedTexture(device,outputs[1]!);
     let worst=0;for(let i=0;i<a.length;i++){assert.ok(Number.isFinite(a[i]));worst=Math.max(worst,Math.abs(a[i]!-b[i]!));}
     assert.ok(worst<1e-6,`layout ${pattern}, mode ${mode}: delta ${worst}`);
     assert.deepEqual(await readMixedBuffer(device,walls[0]!),await readMixedBuffer(device,walls[1]!),"negative wall planes");
    }
   }
  }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
 });
});
