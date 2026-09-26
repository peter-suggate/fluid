import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedRemap} from "../lib/methods/uniform/uniform-mixed-remap";
import {createUniformMixedLayout,mixedCellWidth} from "../lib/methods/uniform/uniform-mixed-layout";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("live remap conserves nonuniform volume and restricts wall release over the complete patch",{timeout:120000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform live remap");
 let device:GPUDevice|undefined;
 const resources:(GPUTexture|GPUBuffer)[]=[];
 let ownership:UniformMixedOwnership|undefined,remap:UniformMixedRemap|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const lattice={dimensions:[8,8,8] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const fine=createUniformMixedLayout(lattice,[]),coarse=createUniformMixedLayout(lattice,[],true,4);
  const mixed=createUniformMixedLayout(lattice,[{id:"half",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:{x:4,y:0,z:0},max_m:{x:8,y:8,z:8}}]);
  const texture=(format:GPUTextureFormat,size:number)=>{const t=device!.createTexture({size:[size,size,size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});resources.push(t);return t;};
  const fields=()=>{const negative=device!.createBuffer({size:3*64*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});resources.push(negative);return {volume:texture("r32float",8),velocity:texture("rgba32float",8),phi:texture("r32float",9),negative};};
  let editing=false;
  const guarded:GPUDevice=new Proxy(device,{get(target,key){
   if(editing&&["createBuffer","createTexture","createShaderModule","createComputePipeline","createComputePipelineAsync"].includes(String(key)))return()=>{throw new Error("live edit allocated or compiled GPU resources");};
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const live=fields(),scratch=fields();ownership=new UniformMixedOwnership(guarded,fine);remap=new UniformMixedRemap(guarded,ownership,live,scratch);await remap.initialize();editing=true;
  const upload=(t:GPUTexture,data:Float32Array<ArrayBuffer>)=>device!.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*4*(t.format==="rgba32float"?4:1),rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  const initial=Float32Array.from({length:512},(_,i)=>(i%17)/19),velocities=new Float32Array(512*4);
  for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
   const p=[x,y,z];let bits=0;
   for(let axis=0;axis<3;axis++){
    // First tangential half is wholly released; the second contains one
    // attached unit patch and must remain attached when restricted.
    const u=(axis+1)%3,v=(axis+2)%3;
    const released=p[u]!<4||p[u]!==4||p[v]!==0;
    if(released&&p[axis]===7)bits|=1<<axis;
    if(released&&p[axis]===0)bits|=1<<(axis+3);
   }
   velocities[4*(x+8*(y+8*z))+3]=bits;
  }
  upload(live.volume,initial);upload(live.velocity,velocities);upload(live.phi,new Float32Array(729).fill(1));
  let expected=initial;
  for(const layout of [coarse,fine,mixed,fine,coarse,fine]){
   const prior=ownership.layout,source=expected;expected=new Float32Array(512);
   const widthAt=(x:number,y:number,z:number)=>mixedCellWidth(prior.tiles[Math.floor(x/4)+2*(Math.floor(y/4)+2*Math.floor(z/4))]!);
   for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
    const w=mixedCellWidth(layout.tiles[Math.floor(x/4)+2*(Math.floor(y/4)+2*Math.floor(z/4))]!);if(x%w||y%w||z%w)continue;
    let sum=0;for(let dz=0;dz<w;dz++)for(let dy=0;dy<w;dy++)for(let dx=0;dx<w;dx++){
     const a=x+dx,b=y+dy,c=z+dz,v=widthAt(a,b,c);sum+=source[a-a%v+8*(b-b%v+8*(c-c%v))]!;
    }expected[x+8*(y+8*z)]=sum/w**3;
   }
   remap.apply(layout);const actual=await readMixedTexture(device,live.volume);assert.deepEqual(errors,[]);
   let mass=0;
   for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
    const w=mixedCellWidth(layout.tiles[Math.floor(x/4)+2*(Math.floor(y/4)+2*Math.floor(z/4))]!);if(x%w||y%w||z%w)continue;
    const at=x+8*(y+8*z);assert.ok(Math.abs(actual[at]!-expected[at]!)<2e-6);mass+=actual[at]!*w**3;
   }
   assert.ok(Math.abs(mass-initial.reduce((a,b)=>a+b,0))<1e-4);
   if(layout===coarse){
    const velocity=await readMixedTexture(device,live.velocity);
    for(let axis=0;axis<3;axis++)for(const sign of [-1,1])for(const tangential of [0,4]){
     const p=[0,0,0];p[axis]=sign>0?7:3;p[(axis+1)%3]=tangential;
     const bits=Math.round(velocity[4*(p[0]!+8*(p[1]!+8*p[2]!))+3]!);
     assert.equal((bits>>(axis+(sign<0?3:0)))&1,tangential===0?1:0,"wall release must not spread from a partial patch");
    }
   }
  }
 }finally{remap?.destroy();ownership?.destroy();resources.forEach(r=>r.destroy());device?.destroy();await releaseWebGPUExclusiveLock();}
});
