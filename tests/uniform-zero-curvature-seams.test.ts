import assert from "node:assert/strict";
import test from "node:test";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";
import {uniformLatticeAddress} from "../lib/methods/uniform/uniform-lattice-address";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedForces} from "../lib/methods/uniform/uniform-mixed-forces";
import {uniformCoarseCacheExtent} from "../lib/methods/uniform/uniform-coarse-velocity-cache";
import {uniformMixedTopologyWGSL} from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "../lib/methods/uniform/uniform-mixed-face-dispatch.wgsl";
import {UniformDetailStorage,uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,uniformDetailPick} from "../lib/methods/uniform/uniform-detail-fields";

test("cached zero-curvature seams match the full production force operator",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("zero capillary seam operator",async device=>{
  const dimensions:[number,number,number]=[32,32,32],address=uniformLatticeAddress(dimensions),forced=new Uint8Array(address.boxTiles);forced[73]=1;
  const layout=createUniformMixedLayout({dimensions,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[],4,forced,address);
  const ownership=new UniformMixedOwnership(device,layout,true,1),storage=new UniformDetailStorage(device,address);storage.attach(ownership);
  const phi=storage.createField("seam phi","vertex","r32float"),volume=storage.createField("seam volume","cell","r32float"),centerPhi=storage.createField("seam center","cell","r32float"),curvature=storage.createField("prescribed curvature","cell","r32float");
  const velocity=storage.createField("seam input","face","rgba32float"),unit=storage.createField("seam unit","face","rgba32float"),output=[0,1].map(i=>storage.createField(`seam output ${i}`,"face","rgba32float"));
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const negatives=[0,1,2].map(()=>device.createBuffer({size:4*address.lowFaceKeys,usage}));
  const normals=device.createBuffer({size:UniformMixedForces.normalBytes(ownership.capacity.owners),usage});
  const params=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(params,0,new Float32Array([1,1,1,.01,0,1000,0,.072,0,0,0,0]));
  const mode=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),stats=device.createBuffer({size:8,usage});
  const coarse=device.createTexture({size:uniformCoarseCacheExtent(address),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING});
  const forces=[false,true].map(exit=>new UniformMixedForces(device,ownership,true,undefined,undefined,true,true,false,true,exit));
  const groups=forces.map((force,i)=>force.bind({advected:velocity,phi,volume,centerPhi,coarseVelocity:coarse,unitVelocity:unit,negative:negatives[0]!,output:output[i]!,outputNegative:negatives[i+1]!,params,curvature,normals:{buffer:normals}}));
  const resources=uniformDetailBindLayout(device,{entries:[...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedFaceAddressWGSL+`
@group(1) @binding(0) var centerPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(1) var curvature:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var outputVolume:texture_storage_3d<r32float,write>;
@group(1) @binding(3) var<uniform> mode:vec4f;
@group(1) @binding(4) var<storage,read_write> stats:array<atomic<u32>>;
@compute @workgroup_size(64) fn writeFields(@builtin(global_invocation_id) id:vec3u){
 let owner=umAllOwner(id);if(owner.width==0u){return;}let origin=umOrigin(owner);let p=vec3f(origin);
 let c=dot(p-vec3f(4),vec3f(.15,.2,.1));var k=0.0;
 if(mode.x==1.0){k=.5+.01*p.x;}if(mode.x==2.0){k=-.5-.01*p.x;}
 if(mode.x==3.0){k=select(-1.0,1.0,owner.width==1u);}if(mode.x==4.0){k=mode.y;}
 textureStore(centerPhi,vec3i(origin),vec4f(c));textureStore(curvature,vec3i(origin),vec4f(k));textureStore(outputVolume,vec3i(origin),vec4f(.5));
 for(var axis=0u;axis<3u;axis++){let face=umFace(owner,axis,1,0u);if(face.neighbor.width!=0u){atomicAdd(&stats[u32(face.neighbor.width==owner.width)],1u);}}
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}: ${m.message}`),[]);
  const writer=await uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"writeFields"}});
  const writerGroup=uniformDetailGroup(device,{layout:resources,entries:[{binding:0,resource:centerPhi},{binding:1,resource:curvature},{binding:2,resource:volume},{binding:3,resource:{buffer:mode}},{binding:4,resource:{buffer:stats}}]});
  try{
   await Promise.all(forces.map(f=>f.initialize()));
   for(const scenario of [0,1,2,3,4]){
    device.queue.writeBuffer(mode,0,new Float32Array([scenario,NaN,0,0]));const encoder=device.createCommandEncoder();encoder.clearBuffer(stats);
    const pass=encoder.beginComputePass();pass.setPipeline(uniformDetailPick(writer));pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,writerGroup.group);pass.dispatchWorkgroups(Math.ceil(ownership.capacity.owners/64));pass.end();
    // Prescribed current curvature is intentional: exercise the production face
    // force operator without overwriting it in cacheCurvature.
    forces.forEach((f,i)=>f.encode(encoder,groups[i]!,false));device.queue.submit([encoder.finish()]);
    const reference=await storage.read(output[0]!),actual=await storage.read(output[1]!);assert.deepEqual(actual,reference,`curvature scenario ${scenario}`);
    assert.deepEqual(await readMixedBuffer(device,negatives[2]!),await readMixedBuffer(device,negatives[1]!));
    const counts=new Uint32Array((await readMixedBuffer(device,stats)).buffer);assert.ok(counts[0]!>0&&counts[1]!>0,"fixture covers mixed and equal width faces");
    if(scenario===0)assert.ok(actual.every(v=>v===0),"zero curvature has no capillary contribution");
    if(scenario===1||scenario===2||scenario===3)assert.ok(actual.some(v=>Math.abs(v)>1e-10),"nonzero equal/mixed-width capillary response is retained");
    if(scenario===4)assert.ok(actual.some(Number.isNaN),"nonfinite curvature takes the original operator path");
   }
  }finally{storage.destroy();ownership.destroy();normals.destroy();params.destroy();mode.destroy();stats.destroy();coarse.destroy();negatives.forEach(b=>b.destroy());}
 });
});
