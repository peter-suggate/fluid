import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createUniformMixedLayout,MIXED_CELL_MASK} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformPressureBand} from "../lib/methods/uniform/uniform-pressure-band";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("h pressure preserves air interfaces hidden from the 4h pressure grid",{timeout:120000},async()=>{
 let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const N=8,T=2,cells=N**3,key=(x:number,y:number,z:number,n=N)=>x+n*(y+n*z);
  const lattice={dimensions:[N,N,N] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const layout=createUniformMixedLayout(lattice,[]),simulation=new UniformMixedOwnership(d,layout,false),pressure=new UniformMixedOwnership(d,createUniformMixedLayout(lattice,[],4),false);
  const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
  const texture=(n:number,format:GPUTextureFormat)=>{const t=d.createTexture({size:[n,n,n],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
  const buffer=(size:number,uniform=false)=>{const b=d.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
  const vertexPhi=texture(N+1,"r32float"),correction=texture(N,"r32float"),velocity=texture(N,"rgba32float"),copy=texture(N,"rgba32float");
  const phi=buffer(4*cells),negative=buffer(4*3*N*N),coarsePressure=buffer(4*T**3),params=buffer(64,true),presentation=buffer(4*cells);
  d.queue.writeBuffer(params,0,new Float32Array([1,1,1,1/60,1000,1,1/60000,1,5,1,1,1]));
  const band=new UniformPressureBand(d,simulation,pressure,{phi:{buffer:phi},vertexPhi,correction,forced:{velocity,negative},velocity,negative,copy,coarsePressure:{buffer:coarsePressure},params,presentation:{buffer:presentation,word:0}});
  try{
   await band.initialize();
   // An isolated positive vertex creates eight h air centres, while every
   // 4h corner (and hence every 4h pressure centre) remains liquid. Repeat
   // across a tile boundary: air authority cannot depend on tile placement.
   for(const point of [[2,2,2],[4,2,2],[6,6,6]]){
    const vertices=new Float32Array((N+1)**3).fill(-1);vertices[key(point[0]!,point[1]!,point[2]!,N+1)]=8;
    const centres=new Float32Array(cells);
    for(let z=0;z<N;z++)for(let y=0;y<N;y++)for(let x=0;x<N;x++){
     let centre=0;for(let k=0;k<8;k++)centre+=vertices[key(x+(k&1),y+((k>>1)&1),z+((k>>2)&1),N+1)]!*.125;
     const tile=key(x>>2,y>>2,z>>2,T),lane=(x%4)+4*(y%4)+16*(z%4);
     centres[(layout.tiles[tile]!&MIXED_CELL_MASK)+lane]=centre;
    }
    d.queue.writeTexture({texture:vertexPhi},vertices,{bytesPerRow:4*(N+1),rowsPerImage:N+1},[N+1,N+1,N+1]);d.queue.writeBuffer(phi,0,centres);
    const e=d.createCommandEncoder();band.encodePrepare(e);d.queue.submit([e.finish()]);
    const index=new Uint32Array((await readMixedBuffer(d,band.index)).buffer);
    const rows=await readMixedBuffer(d,(band as unknown as {rows:GPUBuffer}).rows),bits=new Uint32Array(rows.buffer);
    assert.equal(index[0],T**3,"surrounding liquid keeps every tile in the pressure band");
    let air=0;
    for(let tile=0;tile<T**3;tile++){
     const slot=index[band.slotMapOffset/4+tile]!-1;
     for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++){
      const lane=x+4*y+16*z,row=slot*64+((x+y+z)&1)*32+(x>>1)+2*y+8*z;
      const expected=centres[(layout.tiles[tile]!&MIXED_CELL_MASK)+lane]!<0;
      assert.equal((bits[2*cells+row]!&0x80000000)!==0,expected,`pressure must preserve the h phase: pocket ${point}, tile ${tile}, cell ${x},${y},${z}`);
      if(!expected){air++;assert.equal(rows[row],0,"air has no liquid pressure RHS");}
     }
    }
    assert.equal(air,8,"exercise an interface entirely invisible to coarse pressure");
   }
   assert.deepEqual(errors,[]);
  }finally{band.destroy();simulation.destroy();pressure.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());}
 }finally{device?.destroy();}
});
