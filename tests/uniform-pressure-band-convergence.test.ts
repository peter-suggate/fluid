import assert from "node:assert/strict";
import test from "node:test";
import {createRigidCouplingOracleScene} from "../lib/core/scenes";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import {UniformPressureBand} from "../lib/methods/uniform/uniform-pressure-band";
import {createUniformSolver,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("band stopping accepts measured zero, rejects large or nonfinite residuals, and resets each frame",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("band residual validity and wake",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.voxelDomain.finestCellSize_m=0.025;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.dynamicViscosity_Pa_s=0;
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  // Exercise the real solver and its real convergence shaders. The middle
  // frames manufacture a large or NaN RHS only in the latter half of the band,
  // after rebase. It is a linear-system test, not physical forcing: suppress
  // its projection so the next frame has the original zero-velocity problem.
  type Launch=number|"tiles"|"cells"|"slots"|"middle"|"coarse";
  const {band}= (solver as unknown as {mixedFrame:{band:{index:GPUBuffer;rows:GPUBuffer;capacity:number;boundPass?:GPUComputePassEncoder;dispatch(pass:GPUComputePassEncoder,name:string,launch:Launch):void}}}).mixedFrame;
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> index:array<u32>;
@group(0) @binding(1) var<storage,read_write> rows:array<f32>;
@group(0) @binding(2) var<uniform> injection:vec4f;
const N:u32=${band.capacity*64}u;
@compute @workgroup_size(64) fn inject(@builtin(global_invocation_id) id:vec3u){
 let c=id.x;if(c>=index[0]*64u||c/64u<index[0]/2u){return;}
 if((bitcast<u32>(rows[2u*N+c])&0x80000000u)!=0u&&rows[N+c]>0.0){rows[c]+=injection.x;}
}`});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"inject"}});
  const injection=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:band.index}},{binding:1,resource:{buffer:band.rows}},{binding:2,resource:{buffer:injection}}]});
  const dispatch=band.dispatch.bind(band);let disturb=false;
  band.dispatch=(pass,name,launch)=>{
   if(disturb&&name==="measureInitial"){
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(band.capacity);
    // Restore all production bindings after the test's injected pipeline.
    band.boundPass=undefined;
   }
   if(disturb&&name==="project")return;
   dispatch(pass,name,launch);
  };
  try{
   for(let step=1;step<=5;step++){
    disturb=step===2||step===4;
    device.queue.writeBuffer(injection,0,new Float32Array([step===4?NaN:1e6,0,0,0]));
    assert.ok(solver.advanceTo(step/60,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    const floats=await readMixedBuffer(device,band.index),words=new Uint32Array(floats.buffer);
    const validity=words[UniformPressureBand.residualValidityWord]!;
    assert.ok(words[0]!>1,"the residual spans multiple workgroups");
    assert.ok(validity&UniformPressureBand.initialResidualBit,"the initial measurement is explicitly valid");
    if(disturb){
     assert.ok(floats[UniformPressureBand.initialResidualWord]!>1,"a late workgroup's large residual prevents initial acceptance");
     assert.ok(validity&1,"ordinary smoothing and restriction must run despite the previous frame converging");
     if(step===4){
      assert.equal(floats[UniformPressureBand.initialResidualWord],Infinity,"nonfinite residuals cannot disappear in a floating max");
      assert.equal(words[6],0,"nonfinite residuals cannot converge");
     }
    }else{
     assert.equal(floats[UniformPressureBand.initialResidualWord],0,"the manufactured resting system has exactly zero residual");
     assert.equal(validity,UniformPressureBand.initialResidualBit,"no restriction or final remeasurement executes after initial acceptance");
     assert.equal(words[4],0,"no cycles are needed");assert.equal(words[6],1,"measured zero converges");
     assert.equal(floats[2],0,"the receipt preserves zero");
    }
   }
  }finally{band.dispatch=dispatch;injection.destroy();solver.destroy();}
 });
});
