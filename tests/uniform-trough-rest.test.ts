import assert from "node:assert/strict";
import test from "node:test";
import type {MethodParamValues} from "../lib/core/method-contract";
import {createUniformTroughScene} from "../lib/core/uniform-trough-scenes";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const cases:{h:number;steps:number;name:string;values:MethodParamValues}[]=[
 {h:0.025,steps:1200,name:"settled h/4h/16h",values:{}},
 {h:0.0125,steps:1200,name:"scene resolution h/4h/16h",values:{}},
 {h:0.0125,steps:60,name:"active pressure",values:{warmResolution:"h",bandExit:"strict",regionTiering:"motion",coldRegions:"fine",coldFrameSkip:"off"}},
];
for(const {h,steps,name,values} of cases)test(`hydrostatic trough, ${name}, h=${h}`,{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice(`hydrostatic trough ${name}`,async device=>{
  // The real flared, curved voxel bath includes shallow shorelines, split
  // solid tiles and both resolution seams. Keep production solver defaults.
  const solver=await createUniformSolver(device,createUniformTroughScene("settled-tank",h),values);
  const initial=await readUniformFields(device,solver);
  const {mixedFrame:frame}=solver as unknown as {mixedFrame:UniformMixedFrame};
  // The h field is frozen in COLD regions; also inspect their 16h velocity
  // term so a constant raw field cannot hide a growing coarse correction.
  const freeze=(frame as unknown as {freeze?:{shape:{regions:number};regions:GPUTexture}}).freeze;
  const emptyRegions=freeze?undefined:device.createTexture({size:[1,1],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING});
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var regions:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> peak:atomic<u32>;
@compute @workgroup_size(64) fn read(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=${freeze?.shape.regions??0}u){return;}
 let size=textureDimensions(regions);let value=abs(textureLoad(regions,vec2i(i32(gid.x%size.x),i32(gid.x/size.x)),0).yzw);
 atomicMax(&peak,bitcast<u32>(max(value.x,max(value.y,value.z))));
}`});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"read"}});
  const coarsePeak=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:(freeze?.regions??emptyRegions!).createView()},{binding:1,resource:{buffer:coarsePeak}}]});
  try{
   for(let step=1;step<=steps;step++){
    await advanceUniform(solver,step/60);
    assert.equal(frame.narrowBandFlip!.count,0,`frame ${step}: a resting shoreline must not seed particles`);
    if(step>12&&step%30!==0&&step!==steps)continue;
    const velocity=await readMixedTexture(device,solver.velocityTexture);let peak=0;
    for(let i=0;i<velocity.length;i++)if(i%4!==3){
     assert.ok(Number.isFinite(velocity[i]),`frame ${step}: nonfinite velocity`);
     // Dry backing texels are not fluid velocity. Check every face incident
     // to the tank's liquid, including its free surface and solid contacts.
     const cell=i>>2,axis=i%4,stride=[1,solver.info.nx,solver.info.nx*solver.info.ny][axis]!;
     if(initial.density[cell]!>1e-5||initial.density[cell+stride]!>1e-5)peak=Math.max(peak,Math.abs(velocity[i]!));
    }
    device.queue.writeBuffer(coarsePeak,0,new Uint32Array(1));
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.max(1,Math.ceil((freeze?.shape.regions??0)/64)));pass.end();device.queue.submit([encoder.finish()]);
    const term=(await readMixedBuffer(device,coarsePeak))[0]!;
    assert.ok(Number.isFinite(term)&&term<0.0001,`frame ${step}: coarse frozen velocity ${term} m/s`);
    if(step===1||step===steps)t.diagnostic(`frame ${step}: peak fluid component ${peak} m/s, coarse term ${term} m/s`);
    assert.ok(peak<0.0001,`frame ${step}: hydrostatic trough developed ${peak} m/s flow`);
   }
  }finally{coarsePeak.destroy();emptyRegions?.destroy();solver.destroy();}
 });
});

test("pressure still releases a dam inside the voxel trough",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("moving trough liquid",async device=>{
  const solver=await createUniformSolver(device,createUniformTroughScene("dam-break",0.025));
  try{
   for(let step=1;step<=15;step++)await advanceUniform(solver,step/60);
   const velocity=await readMixedTexture(device,solver.velocityTexture);let peak=0;
   for(let i=0;i<velocity.length;i++)if(i%4!==3){assert.ok(Number.isFinite(velocity[i]));peak=Math.max(peak,Math.abs(velocity[i]!));}
   assert.ok(peak>0.1,`a real pressure imbalance must move the liquid: ${peak} m/s`);
  }finally{solver.destroy();}
 });
});
