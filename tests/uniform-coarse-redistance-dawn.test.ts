import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("coarse redistancing leaves stationary surfaces fixed and resumes after material travel",{timeout:120_000},async()=>{
 let device:GPUDevice|undefined,ownership:UniformMixedOwnership|undefined,surface:UniformMixedSurface|undefined;
 const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const dims=[16,16,16] as const,verts=dims.map(n=>n+1),h=.05;
  const lattice={dimensions:dims,cellSize_m:[h,h,h] as const,origin_m:{x:0,y:0,z:0}};
  const fine=createUniformMixedLayout(lattice,[]),coarse=createUniformMixedLayout(lattice,[],4);
  ownership=new UniformMixedOwnership(d,fine,false);
  const texture=(size:readonly number[],format:GPUTextureFormat)=>{
   const t=d.createTexture({size:[...size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;
  };
  const buffer=(size:number,uniform=false)=>{
   const b=d.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST});buffers.push(b);return b;
  };
  const phi=texture(verts,"r32float"),outputPhi=texture(verts,"r32float"),velocity=texture(dims,"rgba32float"),volume=texture(dims,"r32float");
  const coarseVelocity=texture(dims.map(n=>n/4+2),"rgba32float"),departures=texture(dims,"rgba32float");
  const negative=buffer(4*3*16*16),params=buffer(32,true),count=17**3;
  const evidence=buffer(Math.ceil((ownership.capacity.tiles*4+32)/256)*256+16+4*count);
  surface=new UniformMixedSurface(d,ownership,undefined,undefined,false,true);await surface.initialize();
  const group=surface.bind({phi,outputPhi,velocity,coarseVelocity,volume,negative,departures,params,evidence:{buffer:evidence}});
  const input=new Float32Array(count);
  // With no material travel, repeated timestep maintenance must not move
  // the represented surface. The old per-step rebuild moves it by 0.151h.
  for(let z=0;z<17;z++)for(let y=0;y<17;y++)for(let x=0;x<17;x++)input[x+17*(y+17*z)]=(Math.hypot(x-8,y-8,z-8)-5.5)*h;
  d.queue.writeTexture({texture:phi},input,{bytesPerRow:17*4,rowsPerImage:17},[17,17,17]);
  d.queue.writeTexture({texture:outputPhi},input,{bytesPerRow:17*4,rowsPerImage:17},[17,17,17]);
  ownership.update(coarse);
  d.queue.writeBuffer(params,0,new Float32Array([h,h,h,1/60]));d.queue.writeBuffer(params,16,new Uint32Array([4,1,1,4]));
  const e=d.createCommandEncoder();surface.encode(e,"redistance",group);d.queue.submit([e.finish()]);
  const result=await readMixedTexture(d,outputPhi);let maxMove=0,crossings=0;
  for(let z=0;z<17;z+=4)for(let y=0;y<17;y+=4)for(let x=0;x<17;x+=4){
   const p=[x,y,z],i=x+17*(y+17*z);
   for(let axis=0;axis<3;axis++){
    if(p[axis]===16)continue;const q=[...p];q[axis]!+=4;const j=q[0]!+17*(q[1]!+17*q[2]!);
    const a=input[i]!,b=input[j]!;if(a*b>=0)continue;crossings++;
    const before=-4*a/(b-a),after=-4*result[i]!/(result[j]!-result[i]!);
    maxMove=Math.max(maxMove,Math.abs(after-before));
   }
  }
  for(let z=0;z<16;z+=4)for(let y=0;y<16;y+=4)for(let x=0;x<16;x+=4){
   const corners=Array.from({length:8},(_,k)=>x+4*(k&1)+17*(y+4*((k>>1)&1)+17*(z+4*(k>>2))));
   if(!corners.some(i=>input[i]!<0)||!corners.some(i=>input[i]!>0))continue;
   // Same-sign edge endpoints also shape the trilinear zero set inside a
   // cut cell; an edge-crossing-only preservation mask is insufficient.
   for(const i of corners)assert.equal(result[i],input[i],"cut-cell polynomial changed during redistance");
  }
  assert.ok(crossings>0,"fixture crosses coarse edges");
  assert.ok(maxMove<1e-5,`redistance moved a stationary surface by ${maxMove} h`);
  // Repeated calls alone must not consume the motion budget.
  for(let step=0;step<8;step++){const e=d.createCommandEncoder();surface.encode(e,"redistance",group);d.queue.submit([e.finish()]);}
  const stationary=await readMixedTexture(d,outputPhi);
  assert.equal(stationary[8+17*(4+17*4)],input[8+17*(4+17*4)]);
  // Feed characteristics one h of translation per call. Keep the sampled
  // sphere fixed to isolate scheduling from advection's shape error; the
  // redistance input is identical on each call. After one 4h traversal the
  // normal rebuild must run, so this cannot pass by disabling redistance.
  for(const t of [velocity,coarseVelocity]){
   const values=new Float32Array(t.width*t.height*t.depthOrArrayLayers*4);
   for(let i=0;i<values.length;i+=4)values[i]=1;
   d.queue.writeTexture({texture:t},values,{bytesPerRow:t.width*16,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  }
  d.queue.writeBuffer(params,0,new Float32Array([h,h,h,h]));d.queue.writeBuffer(params,16,new Uint32Array([4,1,0,4]));
  let rebuilt=false;
  for(let step=1;step<=8;step++){
   const e=d.createCommandEncoder();surface.encode(e,"advect",group);surface.encode(e,"redistance",group);d.queue.submit([e.finish()]);
   const state=await readMixedTexture(d,outputPhi);
   const changed=Math.abs(state[8+17*(4+17*4)]!-input[8+17*(4+17*4)]!)>1e-7;
   if(step<4)assert.equal(changed,false,"a fraction of a coarse-cell traversal must not trigger a rebuild");
   rebuilt ||= changed;
  }
  assert.ok(rebuilt,"material travel must still trigger distance rebuilding");
  assert.deepEqual(errors,[]);
 }finally{surface?.destroy();ownership?.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());device?.destroy();}
});
