import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSurfaceGeometry} from "../lib/methods/uniform/uniform-mixed-surface-geometry";
import { UniformMixedForces } from "../lib/methods/uniform/uniform-mixed-forces";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed forces preserve native endpoint viscosity, flat-interface balance and gravity eligibility",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed forces");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const [li,layout] of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!].entries()){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,fixture=mixedPressureFixture(layout),ownership=new UniformMixedOwnership(device,layout);
   const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(format:GPUTextureFormat,pad=0)=>{const t=device!.createTexture({size:d.map(n=>n+pad),dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
   const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
   const velocity=texture("rgba32float"),advected=texture("rgba32float"),phi=texture("r32float",1),volume=texture("r32float"),output=texture("rgba32float");
   const centerPhi=texture("r32float"),targetFill=texture("r32float");
   const bytes=4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]),negative=buffer(bytes),outputNegative=buffer(bytes),params=buffer(48,true);
   const cells=fixture.cells.map(c=>({...c,origin:c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))}));
   const faces:{p:number[];axis:number;width:number;anchor:number[];left:typeof cells[number]}[]=[];
   const put=(p:number[],axis:number,width:number,left:typeof cells[number])=>faces.push({p,axis,width,left,anchor:p.map((v,a)=>Math.round(v-(a===axis?1:width/2)))});
   fixture.faces.forEach(f=>put(f.center.map((v,a)=>v/h[a]!),f.axis,Math.min(cells[f.left]!.width,cells[f.right]!.width),cells[f.left]!));
   cells.forEach(c=>{for(let a=0;a<3;a++)for(const sign of [-1,1]){const p=c.center.map((v,i)=>v/h[i]!);p[a]!+=sign*c.width/2;if(p[a]===0||p[a]===d[a])put(p,a,c.width,c);}});
   const index=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   try{
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("force stage allocated fields");};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}});
    const stage:UniformMixedForces=new UniformMixedForces(borrowed,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const cached=new UniformMixedForces(borrowed,ownership,true);await cached.initialize();
    const geometry=new UniformMixedSurfaceGeometry(borrowed,ownership);await geometry.initialize();
    const geometryGroup=geometry.bind(phi,targetFill,centerPhi);
    const cachedGroup=cached.bind({velocity,advected,phi,volume,negative,output,outputNegative,params,centerPhi});
    const group=stage.bind({velocity,advected,phi,volume,negative,output,outputNegative,params});
    for(const mode of ["gravity","viscosity","capillary","airborne","compressed"] as const){
     if(mode==="viscosity"&&li>1)continue;
     const dt=.03,nu=.001,gravity=mode==="gravity"||mode==="airborne"||mode==="compressed"?-9.81:0;
     device.queue.writeBuffer(params,0,new Float32Array([...h,dt,gravity,1000,mode==="viscosity"?1000*nu:0,mode==="capillary"?.072:0,0,0,1,0]));
     const data=new Float32Array(d[0]*d[1]*d[2]*4).fill(NaN),zero=new Float32Array(data.length).fill(NaN),density=new Float32Array(data.length/4).fill(NaN);
     for(const f of faces)if(f.anchor[f.axis]!>=0){data[4*index(f.anchor)+f.axis]=mode==="viscosity"?.001*f.p[f.axis]!**2:0;zero[4*index(f.anchor)+f.axis]=0;}
     for(const c of cells)density[index(c.origin)]=mode==="airborne"?.2:mode==="compressed"?1.1:0;
     device.queue.writeTexture({texture:velocity},data,{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);device.queue.writeTexture({texture:advected},zero,{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);
     device.queue.writeTexture({texture:volume},density,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},[...d]);device.queue.writeBuffer(negative,0,new Float32Array(bytes/4));
     const pv=new Float32Array((d[0]+1)*(d[1]+1)*(d[2]+1)).fill(NaN);
     for(let z=0;z<=d[2];z++)for(let y=0;y<=d[1];y++)for(let x=0;x<=d[0];x++){
      const p=[x,y,z],owner=cells.filter(c=>c.origin.every((v,a)=>p[a]!>=v&&p[a]!<=v+c.width)).sort((a,b)=>b.width-a.width)[0]!;
      if(owner.origin.every((v,a)=>(p[a]!-v)%owner.width===0))pv[x+(d[0]+1)*(y+(d[1]+1)*z)]=mode==="capillary"?.2*x*h[0]-.4*y*h[1]+.3*z*h[2]:mode==="airborne"||mode==="compressed"?100:-1;
     }
     device.queue.writeTexture({texture:phi},pv,{bytesPerRow:(d[0]+1)*4,rowsPerImage:d[1]+1},d.map(n=>n+1));
     const encoder=device.createCommandEncoder();stage.encode(encoder,group);device.queue.submit([encoder.finish()]);
     const actual=await readMixedTexture(device,output),boundary=await readMixedBuffer(device,outputNegative);
     const optimized=device.createCommandEncoder();geometry.encode(optimized,geometryGroup);cached.encode(optimized,cachedGroup);device.queue.submit([optimized.finish()]);
     const cachedValues=await readMixedTexture(device,output);
     for(const f of faces)if(f.anchor[f.axis]!>=0){const at=4*index(f.anchor)+f.axis;assert.ok(Math.abs(cachedValues[at]!-actual[at]!)<1e-6,"cached force geometry changed canonical velocity");}
     assert.ok(boundary.every(v=>Number.isFinite(v)&&v===0),"negative wall carry changed");
     const airborne=(c:typeof cells[number])=>c.origin.every((v,a)=>v-2*c.width>=0&&v+3*c.width<=d[a]!);
     for(const f of faces)if(f.anchor[f.axis]!>=0){
      const observed=actual[4*index(f.anchor)+f.axis]!;assert.ok(Number.isFinite(observed),`${li}/${mode}: stale face`);
      let expected=0;
      if(mode==="gravity"||mode==="compressed")expected=f.axis===1?gravity*dt:0;
      if(mode==="airborne"){
       const other=cells.find(c=>c.origin[f.axis]===f.p[f.axis]&&c.origin.every((v,a)=>a===f.axis||f.p[a]!>=v&&f.p[a]!<v+c.width));
       expected=f.axis===1&&(airborne(f.left)||other&&airborne(other))?gravity*dt:0;
      }
      if(mode==="viscosity"){
       if(f.p.some((v,a)=>v<2*f.width||v>d[a]!-2*f.width))continue;
       expected=dt*nu*.002/(h[f.axis]!**2);
      }
      assert.ok(Math.abs(observed-expected)<2e-6,`${li}/${mode}: ${f.anchor}/${f.axis}: ${observed} != ${expected}`);
     }
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
