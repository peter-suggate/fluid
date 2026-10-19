import assert from "node:assert/strict";
import test from "node:test";
import {createBackyardPoolPartyScene} from "../lib/core/backyard-pool-party-scene";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {narrowBandRigidContactWGSL} from "../lib/methods/uniform/uniform-particle-contact.wgsl";
import {createUniformSolver,withUniformDevice,readUniformFields} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};

test("water overtaken by a moving rigid surface is displaced and keeps tangential momentum",gpu,async()=>{
 await withUniformDevice("moving rigid particle contact",async device=>{
  const output=device.createBuffer({size:4*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
struct Params{hDt:vec4f} const params=Params(vec4f(.05,.1,.025,1.0/60.0));
fn umSolidTraceWorld(q:vec3f)->vec3f{return vec3f(-1,0,-1)+q*params.hDt.xyz;}
fn umBodyCount()->u32{return 1u;}
fn umBodyNear(i:u32,p:vec3f,margin:f32)->bool{return length(p-vec3f(0,1,0))<=.24+margin;}
fn umBodyDistance(i:u32,p:vec3f)->f32{return length(p-vec3f(0,1,0))-.24;}
fn umBodyVelocity(i:u32,p:vec3f)->vec3f{return vec3f(0,-8,0);}
${narrowBandRigidContactWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) gid:vec3u){
 let points=array<vec3f,4>(vec3f(0,.85,0),vec3f(.2,1,0),vec3f(0,.5,0),vec3f(0,1,0));
 let p=points[gid.x];let q=(p-vec3f(-1,0,-1))/params.hDt.xyz;
 let contact=nbRigidContact(q,vec3f(3,0,2));
 result[2u*gid.x]=vec4f(umSolidTraceWorld(contact.position),1);result[2u*gid.x+1u]=vec4f(contact.velocity,0);
}`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(4);pass.end();device.queue.submit([encoder.finish()]);
   const result=await readMixedBuffer(device,output);
   for(const i of [0,1,3])assert.ok(Math.hypot(result[8*i]!,result[8*i+1]!-1,result[8*i+2]!)>=.24,"overtaken sample must leave the body");
   assert.ok(Math.abs(result[5]!+8)<2e-4,`the descending wall supplies its normal velocity, rather than zero: ${Array.from(result.slice(0,8))}`);
   assert.ok(Math.abs(result[4]!-3)<2e-4&&Math.abs(result[6]!-2)<2e-4,"frictionless contact preserves tangential momentum");
   assert.ok(Math.abs(result[12]!-3)<1e-5,"a separating sample must not be pulled toward the wall");
   assert.ok(Math.abs(result[17]!-.5)<1e-5&&Math.abs(result[21]!)<1e-5,"a sample outside the body is unchanged");
  }finally{output.destroy();}
 });
});

test("pool sphere impact responds to the solved pressure impulse without extra momentum sinks",gpu,async t=>{
 await withUniformDevice("pool sphere pressure impulse",async device=>{
  // Same ball, water and h as the backyard scene; a smaller basin shortens
  // the test. Start just above impact to avoid simulating the board traverse.
  const scene=createBackyardPoolPartyScene();
  scene.container={...scene.container,width_m:3.2,height_m:4.8,depth_m:3.2};scene.solidBoundary=undefined;
  scene.solidVoxels=[[[0,0,0],[64,4,64]],[[0,0,0],[8,40,64]],[[56,0,0],[64,40,64]],[[8,0,0],[56,40,8]],[[8,0,56],[56,40,64]]].map(([minimum,maximumExclusive])=>({operation:"fill",minimum:minimum as [number,number,number],maximumExclusive:maximumExclusive as [number,number,number],materialId:1}));
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-1.2,y:.2,z:-1.2},max_m:{x:1.2,y:1.8,z:1.2}}];
  Object.assign(scene.rigidBodies[0]!,{position_m:{x:0,y:2.2,z:0},linearVelocity_m_s:{x:0,y:-8,z:0}});
  const bodies=initializeRigidBodies(scene.rigidBodies),mass=bodies[0]!.mass_kg,solver=await createUniformSolver(device,scene);
  const inner=solver as unknown as {mixedFrame:UniformMixedFrame;rigidSystem:{stateBuffer:GPUBuffer};rigidExchange:GPUBuffer};
  const dt=1/60;let previous=[0,-8,0],impulseY=0,deepest=2.2;
  try{
   for(let step=1;step<=30;step++){
    assert.ok(solver.advanceTo(step*dt,bodies));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
    const state=await readMixedBuffer(device,inner.rigidSystem.stateBuffer),exchange=new Int32Array((await readMixedBuffer(device,inner.rigidExchange)).buffer);
    for(let axis=0;axis<3;axis++){
     const gravity=axis===1?scene.fluid.gravity_m_s2.y:0;
     const momentumChange=mass*(state[12+axis]!-previous[axis]!-gravity*dt),pressureImpulse=exchange[axis]!*1e-6;
     assert.ok(Math.abs(momentumChange-pressureImpulse)<.002,`frame ${step}, axis ${axis}: ${momentumChange} N s body change vs ${pressureImpulse} N s pressure impulse`);
    }
    previous=Array.from(state.slice(12,15));impulseY+=exchange[1]!*1e-6;deepest=Math.min(deepest,state[1]!);
   }
   assert.ok(impulseY>200,"water must receive a substantial impact reaction");
   assert.ok(deepest<1.37,"the ball must penetrate before buoyancy reverses its motion");
   const fields=await readUniformFields(device,solver);let crest=1.8;
   for(let z=12;z<52;z++)for(let x=12;x<52;x++)for(let y=37;y<56;y++){
    // Measure the zero crossing, not its lower lattice sample: a crown
    // between 2.10 and 2.15 m must not be rounded down to the 2.10 m limit.
    const low=fields.vertex(x,y,z),high=fields.vertex(x,y+1,z);
    if(low<=0&&high>0)crest=Math.max(crest,.05*(y-low/(high-low)));
   }
   t.diagnostic(`pressure impulse ${impulseY.toFixed(2)} N s; minimum centre ${deepest.toFixed(3)} m; crest ${(crest-1.8).toFixed(3)} m above pool`);
   assert.ok(crest>2.1,"the impact must displace a visible crown above the pool");
   assert.ok(inner.mixedFrame.narrowBandFlip!.count>0,"impact must create detailed surface particles");
  }finally{solver.destroy();}
 });
});
