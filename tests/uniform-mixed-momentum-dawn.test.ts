import { UniformMixedMomentumCache } from "../lib/methods/uniform/uniform-mixed-momentum-cache";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedMomentum, UNIFORM_MIXED_MOMENTUM_LIMITS } from "../lib/methods/uniform/uniform-mixed-momentum";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed momentum retains native characteristic, phase filtering, wall carry and bounded correction",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed momentum");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const [layoutIndex,layout] of [seamLayout(0,"fine"),seamLayout(0,"coarse"),...mixedPressureLayouts().slice(0,5)].entries()){
   console.log(`momentum layout ${layoutIndex} starting`);
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,n=d[0]*d[1]*d[2],fixture=mixedPressureFixture(layout);
   const ownership=new UniformMixedOwnership(device,layout),textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(format:GPUTextureFormat,size:number[]=[...d])=>{const t=device!.createTexture({size,dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});textures.push(t);return t;};
   const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.push(b);return b;};
   const extended=texture("rgba32float"),physical=texture("rgba32float"),phase=texture("r32float"),volume=texture("r32float"),predicted=texture("rgba32float"),reversed=texture("rgba32float"),output=texture("rgba32float");
   const coarseExtended=texture("rgba32float",d.map(n=>n/4+2)),coarsePhysical=texture("rgba32float",d.map(n=>n/4+2)),coarseWeight=texture("rgba32float",d.map(n=>n/4+2));
   const boundarySize=4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]);
   const negative=buffer(boundarySize),predictedNegative=buffer(boundarySize),reversedNegative=buffer(boundarySize),outputNegative=buffer(boundarySize),params=buffer(32,true);
   const patches:{center:number[];anchor:number[];axis:number;width:number}[]=[];
   const patch=(p:readonly number[],width:number,axis:number)=>patches.push({center:[...p],anchor:p.map((v,a)=>Math.round(v-(a===axis?1:width/2))),axis,width});
   fixture.faces.forEach(f=>patch(f.center.map((v,a)=>v/h[a]!),Math.min(fixture.cells[f.left]!.width,fixture.cells[f.right]!.width),f.axis));
   fixture.cells.forEach(c=>{for(let a=0;a<3;a++)for(const sign of [-1,1]){const p=c.center.map((v,i)=>v/h[i]!);p[a]!+=sign*c.width/2;if(p[a]===0||p[a]===d[a])patch(p,c.width,a);}});
   const at=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   const boundaryAt=(p:readonly number[],a:number)=>a===0?p[1]!+d[1]*p[2]!:a===1?d[1]*d[2]+p[0]!+d[0]*p[2]!:d[1]*d[2]+d[0]*d[2]+p[0]!+d[0]*p[1]!;
   const writeFaces=(t:GPUTexture,b:GPUBuffer,value:(p:number[],a:number)=>number)=>{
    const data=new Float32Array(n*4).fill(NaN),plane=new Float32Array(boundarySize/4).fill(NaN);
    for(const f of patches){if(f.anchor[f.axis]!<0)plane[boundaryAt(f.anchor,f.axis)]=value(f.center,f.axis);else data[4*at(f.anchor)+f.axis]=value(f.center,f.axis);}
    device!.queue.writeTexture({texture:t},data,{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);device!.queue.writeBuffer(b,0,plane);
   };
   const scalar=(t:GPUTexture,value:number)=>{const data=new Float32Array(n).fill(NaN);fixture.cells.forEach(c=>{const origin=c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2));data[at(origin)]=value;});device!.queue.writeTexture({texture:t},data,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},[...d]);};
   try{
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("momentum allocated fields");};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}});
    const stage:UniformMixedMomentum=new UniformMixedMomentum(borrowed,ownership);await stage.initialize();console.log(`momentum layout ${layoutIndex} compiled`);assert.equal(stage.allocatedBytes,0);
    const defaults:UniformMixedMomentum=new UniformMixedMomentum(borrowed,ownership,true);await defaults.initialize();
    const cache:UniformMixedMomentumCache=new UniformMixedMomentumCache(borrowed,ownership);await cache.initialize();
    const cacheGroup=cache.bind({extended,physical,phase,negative,coarseExtended,coarsePhysical,coarseWeight});
    const group=stage.bind({coarseExtended,coarsePhysical,coarseWeight,extended,physical,phase,volume,predicted,reversed,negative,predictedNegative,reversedNegative,output,outputNegative,params});
    const value=(p:readonly number[],a:number)=>.6+.2*a+.03*p[(a+1)%3]!;
    for(const mode of ["endpoint","default","phase","empty","limiter"] as const){
     if(mode==="endpoint"&&layoutIndex>1)continue;
     const dt=.4,flags=new Uint32Array(8);flags[7]=UNIFORM_MIXED_MOMENTUM_LIMITS;flags[5]=mode==="phase"||mode==="empty"?1:0;
     device.queue.writeBuffer(params,0,flags);device.queue.writeBuffer(params,0,new Float32Array([...h,dt]));
     writeFaces(extended,negative,mode==="endpoint"?value:()=>.75);
     writeFaces(physical,negative,mode==="phase"?()=>3:mode==="empty"?()=>0:mode==="endpoint"?value:()=>.75);
     writeFaces(predicted,predictedNegative,()=>.75);writeFaces(reversed,reversedNegative,()=>100);
     scalar(phase,mode==="empty"?0:1);scalar(volume,1);
     for(const entry of mode==="limiter"?["correct"] as const:["semiLagrangian","predict","reverse"] as const){
      device.queue.writeBuffer(params,24,new Uint32Array([{semiLagrangian:0,predict:1,reverse:2,correct:3}[entry]]));
      const encoder=device.createCommandEncoder();cache.encode(encoder,cacheGroup);stage.encode(encoder,group);device.queue.submit([encoder.finish()]);
      const actual=await readMixedTexture(device,output),boundary=await readMixedBuffer(device,outputNegative);
      if(layout.fineTiles.length===layout.tiles.length){
       const count=layout.tiles.length,header=new Uint32Array(16);header[1]=count;header.set([count,1,1],4);
       device.queue.writeBuffer(ownership.support,count*16,header);
       device.queue.writeBuffer(ownership.support,(5*count+16)*4,layout.fineTiles);
       device.queue.writeBuffer(ownership.certifiedDispatch,0,header.subarray(4,12));
       const fast=device.createCommandEncoder();stage.encode(fast,group);device.queue.submit([fast.finish()]);
       const fastValues=await readMixedTexture(device,output),fastBoundary=await readMixedBuffer(device,outputNegative);
       for(const f of patches){
        const original=f.anchor[f.axis]!<0?boundary[boundaryAt(f.anchor,f.axis)]!:actual[4*at(f.anchor)+f.axis]!;
        const certified=f.anchor[f.axis]!<0?fastBoundary[boundaryAt(f.anchor,f.axis)]!:fastValues[4*at(f.anchor)+f.axis]!;
        assert.ok(Math.abs(original-certified)<1e-6,`${mode}/${entry}: certified sampling changed momentum ${original}/${certified}`);
       }
       ownership.update(layout);
      }
      if((mode==="endpoint"||mode==="default")&&entry==="semiLagrangian"){
       const optimized=device.createCommandEncoder();defaults.encode(optimized,group);device.queue.submit([optimized.finish()]);
       const optimizedValues=await readMixedTexture(device,output),optimizedBoundary=await readMixedBuffer(device,outputNegative);
       for(const f of patches){const original=f.anchor[f.axis]!<0?boundary[boundaryAt(f.anchor,f.axis)]!:actual[4*at(f.anchor)+f.axis]!;
        const specialized=f.anchor[f.axis]!<0?optimizedBoundary[boundaryAt(f.anchor,f.axis)]!:optimizedValues[4*at(f.anchor)+f.axis]!;
        assert.ok(Math.abs(original-specialized)<1e-6,"default specialization changed canonical momentum");}
      }
      for(const f of patches){
       const observed=f.anchor[f.axis]!<0?boundary[boundaryAt(f.anchor,f.axis)]!:actual[4*at(f.anchor)+f.axis]!;
       assert.ok(Number.isFinite(observed),`${layoutIndex}/${mode}/${entry}: stale face ${f.anchor}/${f.axis}`);
       let expected=mode==="phase"?3:mode==="empty"?0:.75;
       if(mode==="endpoint"){
        const width=layoutIndex===0?1:4;
        const sample=(p:readonly number[],a:number)=>{const q=p.map((v,i)=>width*(Math.max(i===a?-1:0,Math.min(d[i]!/width-1,v/width-(i===a?1:.5)))+(i===a?1:.5)));return value(q,a);};
        const clamp=(p:number[])=>p.map((v,a)=>Math.max(0,Math.min(d[a]!,v)));
        let p=[...f.center],remaining=dt;const direction=entry==="reverse"?-1:1;
        for(let k=0;k<32&&remaining>1e-7;k++){
         const first=[0,1,2].map(a=>sample(p,a)),rate=Math.max(...first.map((v,a)=>Math.abs(v)/h[a]!))/width,step=Math.min(remaining,1.5/Math.max(rate,1e-6));
         const mid=clamp(p.map((v,a)=>v-.5*direction*step*first[a]!/h[a]!));p=clamp(p.map((v,a)=>v-direction*step*sample(mid,a)/h[a]!));remaining-=step;
        }
        expected=sample(p,f.axis);
        if(f.anchor[f.axis]!<0||f.center[f.axis]===d[f.axis]&&entry!=="semiLagrangian")expected=value(f.center,f.axis);
        if(f.center[f.axis]===d[f.axis]&&entry==="semiLagrangian")expected=Math.min(expected,value(f.center,f.axis));
       }
       assert.ok(Math.abs(observed-expected)<1e-5,`${layoutIndex}/${mode}/${entry}: ${f.anchor}/${f.axis}: ${observed} != ${expected}`);
      }
     }
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
