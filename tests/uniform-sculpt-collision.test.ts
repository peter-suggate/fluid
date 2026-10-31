import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { createRigidCouplingOracleScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";
import { GPU_RIGID_EXCHANGE_BYTES } from "../lib/core/webgpu-eulerian";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { narrowBandTerrainContactWGSL } from "../lib/methods/uniform/uniform-particle-contact.wgsl";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("carving updates fluid pressure geometry and particle contact, admits falling water, and supports undo/redo", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
 await withUniformDevice("live carved terrain collision", async source => {
  const device=new Proxy(source,{get(target,key){
   if(key==="createComputePipelineAsync")return async(descriptor:GPUComputePipelineDescriptor)=>{
    try{return await target.createComputePipelineAsync(descriptor);}
    catch(error){const info=await descriptor.compute.module.getCompilationInfo();throw new Error(`${descriptor.label ?? descriptor.compute.entryPoint}: ${info.messages.map(m=>m.message).join("; ")}`,{cause:error});}
   };
   const value=Reflect.get(target,key);return typeof value==="function"?value.bind(target):value;
  }});
  const scene=createEmptyScene({extents_m:{x:3.2,y:3.2,z:3.2},finestCellSize_m:.1});
  scene.systems={...scene.systems,fluid:true};
  scene.scenery=undefined;scene.solidVoxels=[];scene.solidBoundary=undefined;scene.rigidBodies=[];
  scene.terrain={baseHeight_m:1.6,features:[]};scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-.2,y:1.7,z:-.2},max_m:{x:.2,y:2.1,z:.2}}];
   const module=device.createShaderModule({code:`const UM_D=vec3u(32);
fn umFramePage()->vec3i{return vec3i(0);}
fn umWorldSolid(page:vec3i)->bool{return any(page<vec3i(0))||any(page>=vec3i(1));}
${uniformMixedSolidWGSL(0,undefined,false)}
struct Params{hDt:vec4f} const params=Params(vec4f(.1,.1,.1,1.0/60.0));
fn umPointPresent(q:vec3f)->bool{return all(q>=vec3f(0))&&all(q<vec3f(32));}
${narrowBandTerrainContactWGSL}
@group(1) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(){
 let c=vec3i(16,12,16);let q=vec3f(c)+.5;
 result[0]=vec4f(umCellOpen(c),umPressureFaceV(c,0u),select(0.0,1.0,nbParticleBlocked(q)),umTerrainBoundary(q).w);
 let floor=nbTerrainContact(vec3f(16.5,8.5,16.5),vec3f(16.5,7.8,16.5),vec3f(0,-1,0),.25);
 result[1]=vec4f(floor.position,floor.velocity.y);
 let wall=nbTerrainContact(vec3f(19.5,12.5,16.5),vec3f(20.2,12.5,16.5),vec3f(1,0,0),.25);
 result[2]=vec4f(wall.position,wall.velocity.x);
 result[3]=vec4f(umCellOpen(vec3i(5,12,5)),umTerrainBoundary(vec3f(5.5,12.5,5.5)).w,0,0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
  const solver=await createUniformSolver(device,scene);
  const solid=(solver as unknown as {mixedFrame:{solid:UniformMixedSolid}}).mixedFrame.solid;
  const output=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try {
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[solid.bindLayout,layout]}),compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:output}}]});
   const sample=async()=>{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,solid.bindGroup);p.setBindGroup(1,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);return readMixedBuffer(device,output);};
   const before=await sample();assert.equal(before[0],0);assert.equal(before[2],1);
   const carved=sceneWithSolidStroke(scene,[{operation:"clear",affectsScenery:true,minimum:[12,8,12],maximumExclusive:[20,24,20]}]);
   solver.validateLiveSolidEdit(carved);solver.applySceneUniforms(carved);
   const after=await sample();assert.equal(after[0],1,"carved terrain cell opens");assert.equal(after[1],1,"pressure face opens");assert.equal(after[2],0,"particles may occupy the cavity");
   assert.ok(Math.abs(after[5]!-8.25)<.01,`cavity floor contact: ${after[5]}`);
   assert.ok(Math.abs(after[8]!-19.75)<.01,`cavity side contact: ${after[8]}`);
   assert.ok(Math.abs(after[9]!-12.5)<1e-5,"side contact must not teleport particles to the old terrain top");
   assert.equal(after[12],0,"remote terrain stays solid");
   solver.applySceneUniforms(scene);assert.deepEqual(await sample(),before,"undo restores collision immediately");
   solver.applySceneUniforms(carved);assert.deepEqual(await sample(),after,"redo restores collision immediately");
   for(let i=1;i<=24;i++)await advanceUniform(solver,i/60);
   const fields=await readUniformFields(device,solver);let entered=0;
   for(let z=13;z<19;z++)for(let y=9;y<15;y++)for(let x=13;x<19;x++)entered+=fields.density[x+solver.info.nx*(y+solver.info.ny*z)]!;
   assert.ok(entered>1,`water must fall below the original terrain into the cut: ${entered}`);
   t.diagnostic(`water below the original terrain after 24 steps: ${entered.toFixed(3)} cell volumes`);
   // The rigid solver shares the same resident cut field. It must not push a
   // body out of the cavity onto the original heightfield, either.
   const inner=solver as unknown as {activeScratch:GPUBuffer;terrainTexture:GPUTexture};
   const rigidScene={...carved,fluid:{...carved.fluid,gravity_m_s2:{x:0,y:0,z:0}}};
   const exchange=device.createBuffer({size:GPU_RIGID_EXCHANGE_BYTES,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const rigid=new WebGPURigidBodySystem(device,rigidScene,exchange,inner.terrainTexture,[32,32]);
   rigid.setProceduralCollisionSource({buffer:inner.activeScratch,pointerWord:inner.activeScratch.size/4-2,
    sculptCollision:true,origin_m:[-1.6,0,-1.6],cellSize_m:[.1,.1,.1]});
   try{
    for(const task of rigid.initializationTasks())await task.run(new AbortController().signal);
    const body={...createRigidCouplingOracleScene("rigid-hydrostatic").rigidBodies[0]!,motion:"dynamic" as const,
     position_m:{x:0,y:1.25,z:0},dimensions_m:{x:.05,y:.05,z:.05}};
    const contact=async()=>{
     rigid.syncBodies([]);rigid.syncBodies(initializeRigidBodies([body]));
     const encoder=device.createCommandEncoder();rigid.encode(encoder,1/60,.001,1,.1);device.queue.submit([encoder.finish()]);
     return (await rigid.readPoses())![0]!.position_m.y;
    };
    assert.ok(Math.abs(await contact()-1.25)<1e-5,"rigid body fits inside carved terrain");
    solver.applySceneUniforms(scene);assert.ok(await contact()>1.64,"undo restores rigid terrain contact");
    solver.applySceneUniforms(carved);assert.ok(Math.abs(await contact()-1.25)<1e-5,"redo reopens rigid cavity");
   }finally{rigid.destroy();exchange.destroy();}
  } finally {output.destroy();solver.destroy();}
 });
});
