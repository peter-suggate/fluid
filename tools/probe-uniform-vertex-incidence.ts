/** Isolated frozen topology benchmark: same vertices/field, walk vs compiled incidence.
 * node --import tsx tools/probe-uniform-vertex-incidence.ts
 */
import assert from "node:assert/strict";
import {resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {writeFileSync} from "node:fs";
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from "../lib/harness/node-dawn-provider";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {uniformLatticeAddress} from "../lib/methods/uniform/uniform-lattice-address";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import {GPUPassProfile} from "./gpu-pass-profile";
function referenceIncidence(source:string):string {
 const start=source.indexOf(" // Stored tiles already compiled"),end=source.indexOf(" if(stored",start);assert.ok(start>=0&&end>start);
 const walk=source.slice(start,end),first=walk.indexOf(" for(var k="),last=walk.lastIndexOf(" }");
 assert.ok(first>=0&&last>first);return source.slice(0,start)+walk.slice(first,last)+source.slice(end);
}
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query"]}),profile=new GPUPassProfile(raw),device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
for(const scenario of ["fine","coarse","mixed","air-seam"]){
 const dimensions:[number,number,number]=[64,64,64],lattice={dimensions,cellSize_m:[1,1,1] as [number,number,number],origin_m:{x:0,y:0,z:0}};
 const address=uniformLatticeAddress(dimensions,true,{blocks:12,permute:7,...(scenario==="air-seam"?{air:[[1,1,0] as [number,number,number]]}:{})});
 const fine=new Uint8Array(16**3);if(scenario==="mixed"||scenario==="air-seam")for(let i=0;i<fine.length;i++)fine[i]=((i%16+Math.floor(i/16)%16+Math.floor(i/256))%3===0)?1:0;
 const layout=createUniformMixedLayout(lattice,[],scenario==="fine"?1:4,fine,address),ownership=new UniformMixedOwnership(device,layout,false);
 const count=65**3,bytes=count*4,output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const bindLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),bind=device.createBindGroup({layout:bindLayout,entries:[{binding:0,resource:{buffer:output}}]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,bindLayout]});
 const sampling=uniformMixedVertexSamplingSource(),header=uniformMixedTopologyWGSL(layout,0)+`
fn umLoadVertex(p:vec3u)->f32{return f32((p.x*p.x+7u*p.y*p.y+13u*p.z*p.z)%10007u)*0.125;}
`,tail=`
@group(1) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=${count}u){return;}let p=vec3u(i%65u,(i/65u)%65u,i/4225u);result[i]=umVertexValue(p);}`;
 const pipelines:GPUComputePipeline[]=[];
 for(const code of [referenceIncidence(sampling),sampling]){const source=header+code+tail,module=device.createShaderModule({code:source});const failures=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(failures.length){writeFileSync("/tmp/vertex-incidence.wgsl",source);throw new Error(JSON.stringify(failures));}pipelines.push(await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"evaluate"}}));}
 const encode=(candidate:number,replays=1)=>{const e=device.createCommandEncoder(),p=e.beginComputePass({label:String(candidate)});p.setPipeline(pipelines[candidate]!);p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,bind);for(let replay=0;replay<replays;replay++)p.dispatchWorkgroups(Math.ceil(count/64));p.end();device.queue.submit([e.finish()]);};
 const outputs:Uint32Array[]=[];for(let v=0;v<2;v++){encode(v);const b=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=device.createCommandEncoder();e.copyBufferToBuffer(output,0,b,0,bytes);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);outputs.push(new Uint32Array(b.getMappedRange().slice(0)));b.unmap();b.destroy();}
 let mismatches=0;const first:unknown[]=[];for(let i=0;i<count;i++)if(outputs[0]![i]!==outputs[1]![i]){mismatches++;if(first.length<5)first.push({p:[i%65,Math.floor(i/65)%65,Math.floor(i/4225)],a:outputs[0]![i],b:outputs[1]![i]});}
 for(let i=0;i<4;i++){encode(0);encode(1);}await device.queue.onSubmittedWorkDone();
 const timings:number[][]=[[],[]];for(let repeat=0;repeat<12;repeat++)for(const v of repeat%2?[1,0]:[0,1]){profile.start();encode(v,16);await device.queue.onSubmittedWorkDone();timings[v]!.push((await profile.finish())[0]!.ms/16);}
 const median=(v:number[])=>[...v].sort((a,b)=>a-b)[Math.floor(v.length/2)]!;
 console.log(JSON.stringify({scenario,vertices:count,mismatches,first,baseline_ms:median(timings[0]!),candidate_ms:median(timings[1]!),timings,errors}));assert.equal(mismatches,0);ownership.destroy();output.destroy();
}
assert.deepEqual(errors,[]);profile.destroy();device.destroy();
