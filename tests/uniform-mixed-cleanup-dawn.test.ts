import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedCleanup} from "../lib/methods/uniform/uniform-mixed-cleanup";
import {mixedPressureFixture,mixedPressureLayouts} from "./helpers/uniform-mixed-pressure";
import {seamLayout} from "./helpers/uniform-geometric-seam";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed cleanup preserves surface and cluster evidence and accounts for physical discarded mass",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed cleanup");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!]){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,n=d[0]*d[1]*d[2];
   const ownership:UniformMixedOwnership=new UniformMixedOwnership(device,layout);
   const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(pad=0)=>{const t=device!.createTexture({size:d.map(n=>n+pad),dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
   const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
   const a=texture(),b=texture(),phi=texture(1),params=buffer(32,true),reductions=buffer(48);
   const cells=mixedPressureFixture(layout).cells.map(c=>({...c,origin:c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))}));
   const index=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   const upload=(t:GPUTexture,data:Float32Array<ArrayBuffer>)=>device!.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
   try{
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("cleanup allocated fields");};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}});
    const stage:UniformMixedCleanup=new UniformMixedCleanup(borrowed,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const groups=[stage.bind(a,b,phi,params,reductions),stage.bind(b,a,phi,params,reductions)] as const;
    for(const mode of ["off","dust","protected","isolated","evidence","cluster","threshold"] as const){
     const floor=mode==="off"?0:.001,orphan=.02,distance=mode==="protected"?-100:100;
     device.queue.writeBuffer(params,0,new Float32Array([0,0,floor,orphan,0,0,0,0]));
     const values=new Float32Array(n).fill(NaN);
     cells.forEach((c,i)=>{values[index(c.origin)]=mode==="dust"||mode==="protected"?(i%2?.0001:-.0001):mode==="cluster"?.019:mode==="threshold"?.02:i%17===0?.004:mode==="evidence"&&i%17===1?.1:0;});
     upload(a,values);upload(b,new Float32Array(n).fill(NaN));upload(phi,new Float32Array(phi.width*phi.height*phi.depthOrArrayLayers).fill(distance));device.queue.writeBuffer(reductions,0,new Uint32Array(12));
     const expectedCounts=new Uint32Array(12);
     const discard=(i:number,v:number,threshold:number,word:number)=>{expectedCounts[word]!++;expectedCounts[word+1]!+=Math.min(Math.trunc(Math.fround(Math.fround(Math.abs(v)/Math.fround(threshold))*64)),64)*cells[i]!.width**3;return 0;};
     // Independent geometric overlap oracle: integrate neighbouring owners by
     // intersecting boxes, never iterate GPU addresses or duplicate donors.
     const floored=cells.map((c,i)=>{const v=values[index(c.origin)]!;return v!==0&&Math.abs(v)<Math.fround(floor)&&!(v>0&&distance<4*Math.max(...h)*c.width)?discard(i,v,floor,5):v;});
     const expected=floored.map((v,i)=>{
      const c=cells[i]!,w=c.width;
      if(!(floor>0&&orphan>floor&&v>0&&v<Math.fround(orphan))||distance<4*Math.max(...h)*w)return v;
      let mass=0;
      for(let j=0;j<cells.length;j++){
       const other=cells[j]!;let overlap=1;
       for(let axis=0;axis<3;axis++)overlap*=Math.max(0,Math.min(c.origin[axis]!+2*w,other.origin[axis]!+other.width)-Math.max(c.origin[axis]!-w,other.origin[axis]!));
       if(!overlap)continue;if(floored[j]!>=Math.fround(.05))return v;mass+=Math.max(floored[j]!,0)*overlap;
      }
      return mass>=.25*w**3?v:discard(i,v,orphan,10);
     });
     const encoder=device.createCommandEncoder();stage.encode(encoder,groups);device.queue.submit([encoder.finish()]);
     const output=await readMixedTexture(device,a);
     const counts:Uint32Array=new Uint32Array((await readMixedBuffer(device,reductions)).buffer);
     let removed=0;cells.forEach((c,i)=>{assert.equal(output[index(c.origin)],expected[i],`${mode}, width ${c.width}, owner ${i}`);removed+=(Math.abs(values[index(c.origin)]!)-Math.abs(expected[i]!))*c.width**3;});
     assert.deepEqual(counts,expectedCounts,`${mode} physical mass counters`);
     const accounted=counts[6]!*floor/64+counts[11]!*orphan/64;
     const quantization=cells.reduce((sum,c,i)=>sum+(expected[i]===0&&values[index(c.origin)]!==0?c.width**3*Math.max(floor,orphan)/64:0),0);
     assert.ok(accounted<=removed+1e-6&&removed-accounted<=quantization+1e-6,`${mode} loss must be covered by counter quantization`);
     if(mode==="isolated")assert.ok(counts[10]!>0,"orphan cleanup must do work");
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
