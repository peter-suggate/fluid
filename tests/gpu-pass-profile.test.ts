import assert from "node:assert/strict";
import test from "node:test";
import { GPUPassProfile } from "../tools/gpu-pass-profile";

test("split dispatch diagnostics replay pipeline and captured binding state",async()=>{
 const passes:{descriptor:GPUComputePassDescriptor;commands:unknown[][]}[]=[];
 const encoder={beginComputePass(descriptor:GPUComputePassDescriptor){
  const commands:unknown[][]=[];passes.push({descriptor,commands});
  return {setPipeline:(...args:unknown[])=>commands.push(["pipeline",...args]),setBindGroup:(...args:unknown[])=>commands.push(["bind",...args]),dispatchWorkgroups:(...args:unknown[])=>commands.push(["direct",...args]),dispatchWorkgroupsIndirect:(...args:unknown[])=>commands.push(["indirect",...args]),end:()=>commands.push(["end"])};
 }};
 const raw={createQuerySet:()=>({destroy(){}}),createCommandEncoder:()=>encoder,createComputePipelineAsync:async()=>({label:""})} as unknown as GPUDevice;
 const profile=new GPUPassProfile(raw,32,true);profile.start();
 const module={} as GPUShaderModule;
 const first=await profile.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"first"}});
 const second=await profile.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"second"}});
 const group={} as GPUBindGroup,buffer={} as GPUBuffer;
 const pass=profile.device.createCommandEncoder().beginComputePass({label:"stage"});
 const offsets=new Uint32Array([0,256,512]);
 pass.setBindGroup(0,group,offsets,1,1);offsets[1]=1024;
 pass.setPipeline(first);pass.dispatchWorkgroups(2,3);
 pass.setPipeline(second);pass.setBindGroup(1,null);pass.dispatchWorkgroupsIndirect(buffer,16);pass.end();
 assert.equal(passes.length,2);
 assert.deepEqual(passes[0]!.commands,[["pipeline",first],["bind",0,group,[256]],["direct",2,3],["end"]]);
 assert.deepEqual(passes[1]!.commands,[["pipeline",second],["bind",0,group,[256]],["bind",1,null,undefined],["indirect",buffer,16],["end"]]);
 assert.equal(passes[0]!.descriptor.timestampWrites!.beginningOfPassWriteIndex,0);
 assert.equal(passes[1]!.descriptor.timestampWrites!.beginningOfPassWriteIndex,2);
 profile.destroy();
});
