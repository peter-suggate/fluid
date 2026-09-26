import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureLevelStage } from "../lib/methods/uniform/uniform-mixed-pressure-stage";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed pressure retains half-dual wall coefficients and separating constrained halo rows",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed pressure walls");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!])for(const openTop of [false,true]){
   const ownership=new UniformMixedOwnership(device,layout),storage=uniformMixedPressureStorage(layout),n=layout.cellCount,owned:GPUBuffer[]=[];
   const buffer=(count:number)=>{const b=device!.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});owned.push(b);return b;};
   const pressure=buffer(storage.count),slopes=buffer(n*4),rhs=buffer(storage.count),frozen=buffer(storage.count),result=buffer(storage.count),minimum=buffer(storage.count),phi=buffer(n);
   try{
    const b=new Float32Array(storage.count),low=new Float32Array(storage.count).fill(-3.402823e38),expected=new Float32Array(storage.count);
    const h=layout.lattice.cellSize_m,d=layout.lattice.dimensions.map(v=>v/storage.width),valid=new Set<number>();
    const fixture=mixedPressureFixture(layout);
    fixture.cells.forEach((c,i)=>{
     valid.add(i);const origin=c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2));
     for(let axis=0;axis<3;axis++)for(const side of [0,1]){
      if(side===0?origin[axis]!==0:origin[axis]!+c.width!==layout.lattice.dimensions[axis])continue;
      const p=origin.map(v=>v/storage.width);
      const at=n+(axis===0?side*d[1]!*d[2]!+p[1]!+d[1]!*p[2]!:axis===1?2*d[1]!*d[2]!+side*d[0]!*d[2]!+p[0]!+d[0]!*p[2]!:2*(d[1]!*d[2]!+d[0]!*d[2]!)+side*d[0]!*d[1]!+p[0]!+d[0]!*p[1]!);
      assert.ok(!valid.has(at)&&at<storage.count,"halo patches overlap");valid.add(at);low[at]=0;
      const release=axis===1&&side===1;
      const coefficient=.5/(c.width*h[axis]!)**2;
      expected[at]=release?0:1+axis;
      // Independent manufactured wall system: physical p=0. Contact halos
      // own positive pressure, while the ceiling's active bound permits
      // separation. The closed half-dual coefficient is the native 1/2 h².
      b[i]!-=coefficient*expected[at]!;
      b[at]=release?(openTop?0:-coefficient):coefficient*expected[at]!;
     }
    });
    device.queue.writeBuffer(rhs,0,b);device.queue.writeBuffer(minimum,0,low);device.queue.writeBuffer(phi,0,new Float32Array(n).fill(-1));
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("boundary stage allocated a field");};const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;}});
    const stage:UniformMixedPressureLevelStage=new UniformMixedPressureLevelStage(borrowed,ownership,true,true,{openTop});await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const view=(buffer:GPUBuffer)=>({buffer});const group=stage.bind({pressure:view(pressure),slopes:view(slopes),rhs:view(rhs),frozen:view(frozen),result:view(result),minimum:view(minimum),phi:view(phi)});
    // This isolates the local operator, whose low-frequency contact error
    // converges slowly. Live traversal uses the separately tested multigrid
    // schedule; it must never adopt this diagnostic sweep count.
    const encoder=device.createCommandEncoder();for(let i=0;i<8000;i++)stage.encodeSweep(encoder,group);stage.encode(encoder,"measure",group);device.queue.submit([encoder.finish()]);
    const actual=await readMixedBuffer(device,pressure),measured=await readMixedBuffer(device,result);
    for(const at of valid){assert.ok(Number.isFinite(actual[at])&&Math.abs(actual[at]!-expected[at]!)<2e-4,`${openTop}/${n}: row ${at}: ${actual[at]} != ${expected[at]}`);assert.ok(measured[at]!<2e-4,`constrained residual ${at}: ${measured[at]}`);}
   }finally{owned.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
