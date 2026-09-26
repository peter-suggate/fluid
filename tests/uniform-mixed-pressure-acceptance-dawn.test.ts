import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureAcceptance } from "../lib/methods/uniform/uniform-mixed-pressure-acceptance";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed pressure convergence fails closed without changing pressure or permitting recovery",{timeout:120000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed pressure acceptance");let device:GPUDevice|undefined;
 const buffers:GPUBuffer[]=[];let ownership:UniformMixedOwnership|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const layout=seamLayout(0,"coarse");ownership=new UniformMixedOwnership(device,layout);const count=uniformMixedPressureStorage(layout).count;
  const buffer=(size:number,extra=0)=>{const b=device!.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|extra});buffers.push(b);return b;};
  const pressure=buffer(count*4),residual=buffer(count*4),state=buffer(32),params=buffer(16,GPUBufferUsage.UNIFORM);
  device.queue.writeBuffer(params,0,new Float32Array([1,2,0,0]));
  const stage:UniformMixedPressureAcceptance=new UniformMixedPressureAcceptance(device,ownership);await stage.initialize();
  const group=stage.bind({residual:{buffer:residual},state,params});
  // A rejection is terminal, even if a later checkpoint would converge.
  // Each sequence starts a fresh frame; pressure is deliberately not bound.
  for(const norms of [[10,8,12,1],[10,8,1],[3.402823e38,1],[1],[10,NaN,1]]){
   let failed=false,previous=Infinity;
   for(const [i,norm] of norms.entries()){
    device.queue.writeBuffer(pressure,0,new Float32Array(count).fill(i+1));
    device.queue.writeBuffer(residual,0,new Float32Array(count).fill(norm));
    const encoder=device.createCommandEncoder();stage.encode(encoder,group,state,i===0?"initial":"cycle");device.queue.submit([encoder.finish()]);
    const actual=await readMixedBuffer(device,pressure),diagnostics=await readMixedBuffer(device,state);
    assert.ok(actual.every(v=>v===i+1),"checkpoint must never restore or otherwise change pressure");
    failed ||= !Number.isFinite(norm)||norm>=3.402823e38||norm>previous;
    if(!failed)previous=norm;
    const words=new Uint32Array(diagnostics.buffer);
    assert.equal(words[4],+failed,"failure stays latched until a new frame");
    assert.equal(words[5],+(!failed&&norm<=2),"only a valid converged candidate can be published");
    assert.equal(words[7],0,"no recovery work exists");
   }
  }
  assert.deepEqual(errors,[]);
 }finally{buffers.forEach(b=>b.destroy());ownership?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
