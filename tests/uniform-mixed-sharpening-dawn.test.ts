import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedSharpening } from "../lib/methods/uniform/uniform-mixed-sharpening";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed geometric sharpening conserves physical mass and honors dose, compaction, orphan and dust controls",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed sharpening");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!]){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,n=d[0]*d[1]*d[2],ownership=new UniformMixedOwnership(device,layout);
   const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(pad=0)=>{const t=device!.createTexture({size:d.map(n=>n+pad),dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
   const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
   const a=texture(),b=texture(),phi=texture(1),target=texture(),center=texture(),scratch=buffer(40*n),params=buffer(32,true),reductions=buffer(32);
   const cells=mixedPressureFixture(layout).cells.map(c=>({...c,origin:c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))}));
   const index=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   const upload=(t:GPUTexture,data:Float32Array<ArrayBuffer>)=>device!.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
   try{
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("sharpening allocated fields");};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}});
    const stage:UniformMixedSharpening=new UniformMixedSharpening(borrowed,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const groups=[stage.bind(a,b,phi,target,center,{buffer:scratch},params,reductions),stage.bind(b,a,phi,target,center,{buffer:scratch},params,reductions)];
    for(const mode of ["off","surface","compact","orphan","dust","protected"] as const){
     const dose=mode==="off"||mode==="dust"||mode==="protected"?0:1;
     device.queue.writeBuffer(params,0,new Float32Array([dose,2.1,mode==="dust"||mode==="protected"?.001:0,0,mode==="compact"?1:0,mode==="orphan"?2:0,0,0]));
     const values=new Float32Array(n).fill(NaN),targets=new Float32Array(n).fill(NaN),distances=new Float32Array(n).fill(NaN);
     const phiValue=(p:readonly number[])=>mode==="compact"?-100-.1*p[0]!:mode==="orphan"||mode==="dust"?100:mode==="protected"?-100:0;
     cells.forEach((c,i)=>{
      const at=index(c.origin);values[at]=mode==="dust"||mode==="protected"?.0001:.03+.94*((i*17)%101)/100;
      targets[at]=mode==="compact"?1:mode==="orphan"?0:.5;distances[at]=phiValue(c.origin.map(v=>v+c.width/2));
     });
     upload(a,values);upload(b,new Float32Array(n).fill(NaN));upload(target,targets);upload(center,distances);
     const vertices=new Float32Array(phi.width*phi.height*phi.depthOrArrayLayers);
     for(let z=0;z<=d[2];z++)for(let y=0;y<=d[1];y++)for(let x=0;x<=d[0];x++)vertices[x+phi.width*(y+phi.height*z)]=phiValue([x,y,z]);
     upload(phi,vertices);device.queue.writeBuffer(scratch,0,new Float32Array(scratch.size/4).fill(NaN));device.queue.writeBuffer(reductions,0,new Uint32Array(8));
     const mass=(v:Float32Array)=>cells.reduce((sum,c)=>sum+v[index(c.origin)]!*c.width**3,0),initial=mass(values);
     const encoder=device.createCommandEncoder();stage.encodeGeometry(encoder,groups[0]!);for(let i=0;i<8;i++)stage.encodeSweep(encoder,groups[i%2]!,false);device.queue.submit([encoder.finish()]);
     const output=await readMixedTexture(device,a);let changed=0;
     for(const c of cells){const at=index(c.origin),v=output[at]!;assert.ok(Number.isFinite(v)&&v>=-1e-6&&v<=1+1e-6,`${mode}: invalid fill ${v}`);changed+=Math.abs(v-values[at]!);}
     if(mode==="dust"){
      assert.equal(mass(output),0);const result=await readMixedBuffer(device,reductions);assert.equal(new Uint32Array(result.buffer)[5],cells.length);
     }else{
      assert.ok(Math.abs(mass(output)-initial)<5e-6*Math.max(1,initial),`${mode}: sharpening changed mass`);
      if(mode==="off"||mode==="protected")assert.equal(changed,0);else assert.ok(changed>1e-4,`${mode}: enabled control did no work`);
     }
     if(mode==="surface"){
      const error=(v:Float32Array)=>cells.reduce((sum,c)=>sum+Math.abs(v[index(c.origin)]!-.5)*c.width**3,0);
      assert.ok(error(output)<error(values),"surface sharpening did not approach geometric fill");
     }
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
