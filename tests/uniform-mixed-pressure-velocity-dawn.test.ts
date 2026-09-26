import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureVelocity } from "../lib/methods/uniform/uniform-mixed-pressure-velocity";
import { UniformMixedPressureLevelStage } from "../lib/methods/uniform/uniform-mixed-pressure-stage";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed pressure RHS and projection share canonical fluxes and publish separating-wall release",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed pressure velocity");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),...mixedPressureLayouts()]){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,n=layout.cellCount,storage=uniformMixedPressureStorage(layout),fixture=mixedPressureFixture(layout),ownership=new UniformMixedOwnership(device,layout);
   const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(format:GPUTextureFormat)=>{const t=device!.createTexture({size:[...d],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
   const buffer=(count:number,uniform=false)=>{const b=device!.createBuffer({size:count*4,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
   const velocity=texture("rgba32float"),output=texture("rgba32float"),centerPhi=texture("r32float"),volume=texture("r32float"),correction=texture("r32float");
   const negative=buffer(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]),outputNegative=buffer(negative.size/4),params=buffer(8,true);
   const pressure=buffer(storage.count),rhs=buffer(storage.count),minimum=buffer(storage.count),phi=buffer(n),slopes=buffer(n*4),frozen=buffer(storage.count),result=buffer(storage.count);
   const cells=fixture.cells.map(c=>({...c,origin:c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))}));
   const faces:{axis:number;anchor:number[];width:number;left:number;right:number;sign:number}[]=[];
   fixture.faces.forEach(f=>{const width=Math.min(cells[f.left]!.width,cells[f.right]!.width);faces.push({axis:f.axis,anchor:f.center.map((v,a)=>Math.round(v/h[a]!-(a===f.axis?1:width/2))),width,left:f.left,right:f.right,sign:1});});
   cells.forEach((c,left)=>{for(let axis=0;axis<3;axis++)for(const sign of [-1,1])if(sign<0?c.origin[axis]===0:c.origin[axis]!+c.width===d[axis]){
    const anchor=[...c.origin];anchor[axis]!+=sign<0?-1:c.width-1;faces.push({axis,anchor,width:c.width,left,right:-1,sign});
   }});
   const index=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   const boundaryIndex=(p:readonly number[],a:number)=>a===0?p[1]!+d[1]*p[2]!:a===1?d[1]*d[2]+p[0]!+d[0]*p[2]!:d[1]*d[2]+d[0]*d[2]+p[0]!+d[0]*p[1]!;
   const haloIndex=(f:typeof faces[number])=>{const p=cells[f.left]!.origin.map(v=>v/storage.width),s=d.map(v=>v/storage.width),side=f.sign>0?1:0,a=f.axis;
    return n+(a===0?side*s[1]!*s[2]!+p[1]!+s[1]!*p[2]!:a===1?2*s[1]!*s[2]!+side*s[0]!*s[2]!+p[0]!+s[0]!*p[2]!:2*(s[1]!*s[2]!+s[0]!*s[2]!)+side*s[0]!*s[1]!+p[0]!+s[0]!*p[1]!);};
   try{
    const stage:UniformMixedPressureVelocity=new UniformMixedPressureVelocity(device,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const level=new UniformMixedPressureLevelStage(device,ownership,true,true,{openTop:false});await level.initialize();
    const view=(buffer:GPUBuffer)=>({buffer});
    const rg=stage.bindRhs({velocity,negative:view(negative),phi:view(phi),params,correction,rhs:view(rhs),minimum:view(minimum),pressure:view(pressure)});
    const pg=stage.bindProjection({velocity,negative:view(negative),phi:view(phi),params,pressure:view(pressure),slopes:view(slopes),centerPhi,volume,output,outputNegative:view(outputNegative)});
    const lg=level.bind({pressure:view(pressure),rhs:view(rhs),slopes:view(slopes),frozen:view(frozen),result:view(result),minimum:view(minimum),phi:view(phi)});
    const dt=.1,rho=2,g=[.2,-.3,.1];device.queue.writeBuffer(params,0,new Float32Array([...h,dt,rho,0,0,.01]));device.queue.writeBuffer(phi,0,new Float32Array(n).fill(-1));
    for(const mode of ["affine","release"]){
     const field=new Float32Array(d[0]*d[1]*d[2]*4).fill(NaN),low=new Float32Array(negative.size/4).fill(NaN),p=new Float32Array(storage.count),expectedRhs=new Float64Array(storage.count);
     cells.forEach((c,i)=>{p[i]=mode==="affine"?10+c.center.reduce((sum,v,a)=>sum+g[a]!*v,0):0;});
     for(const f of faces){
      const c=cells[f.left]!,u=mode==="affine"?dt/rho*g[f.axis]!:f.right<0?-f.sign*.25:0;
      if(f.anchor[f.axis]!<0)low[boundaryIndex(c.origin,f.axis)]=u;else field[4*index(f.anchor)+f.axis]=u;
      const area=f.width*f.width*h[(f.axis+1)%3]!*h[(f.axis+2)%3]!,flux=(f.right<0?.5:1)*area*u;
      expectedRhs[f.left]!-=rho/dt*f.sign*flux/c.volume;
      if(f.right>=0)expectedRhs[f.right]!+=rho/dt*flux/cells[f.right]!.volume;
      else{const hi=haloIndex(f);p[hi]=mode==="affine"?p[f.left]!+f.sign*c.width*h[f.axis]!*g[f.axis]!:0;expectedRhs[hi]=rho/dt*f.sign*.5*u/(c.width*h[f.axis]!);}
     }
     device.queue.writeTexture({texture:velocity},field,{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);device.queue.writeBuffer(negative,0,low);
     let encoder=device.createCommandEncoder();stage.encode(encoder,"rhs",rg);device.queue.submit([encoder.finish()]);
     const built=await readMixedBuffer(device,rhs);for(let i=0;i<n;i++)assert.ok(Math.abs(built[i]!-expectedRhs[i]!)<2e-6,`rhs ${mode}/${n}/${i}`);
     for(const f of faces)if(f.right<0)assert.ok(Math.abs(built[haloIndex(f)]!-expectedRhs[haloIndex(f)]!)<2e-6);
     device.queue.writeBuffer(pressure,0,p);encoder=device.createCommandEncoder();level.encode(encoder,"reconstruct",lg);stage.encode(encoder,"project",pg);device.queue.submit([encoder.finish()]);
     const projected=await readMixedTexture(device,output),projectedLow=await readMixedBuffer(device,outputNegative);
     for(const f of faces){
      const observed=f.anchor[f.axis]!<0?projectedLow[boundaryIndex(cells[f.left]!.origin,f.axis)]!:projected[4*index(f.anchor)+f.axis]!;
      const expected=mode==="affine"?0:f.right<0?-f.sign*.25:0;
      assert.ok(Math.abs(observed-expected)<1e-5,`projection ${mode}/${n}/${f.anchor}/${f.axis}: ${observed} != ${expected}`);
      if(f.right<0&&f.sign>0){const bits=projected[4*index(f.anchor)+3]!;assert.equal((bits>>f.axis)&1,mode==="release"?1:0);}
     }
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
