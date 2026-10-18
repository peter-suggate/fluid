import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {narrowBandRedistanceWGSL} from "../lib/methods/uniform/uniform-narrow-band-redistance.wgsl";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";

test("coarse retirement bounds use free surfaces, including surfaces at wall edges",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("coarse retirement wall bound",async device=>{
  const source=narrowBandRedistanceWGSL.slice(narrowBandRedistanceWGSL.indexOf("fn nbMetricZeroOpen"),narrowBandRedistanceWGSL.indexOf("@compute @workgroup_size(64) fn redistanceCoarse"));
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
const NB_DEPTH_A=0u;struct Params{hDt:vec4f} const params=Params(vec4f(0.05));
var<private> fixture:u32;
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umVertexAt(p:vec3u)->u32{return fixture;}
fn umCellOpen(p:vec3i)->f32{return f32(all(p>=vec3i(0))&&all(p<vec3i(32,24,select(16,32,fixture==2u))));}
fn umSampleVertex(p:vec3f)->f32{
 if(p.z>16.0){return 1.6;}
 if(fixture==1u){return 0.05*(p.x-5.0);}
 return 0.05*(p.y-15.25);
}
${source}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){fixture=id.x;nbBoundCoarseMetric(vec3u(4,12,16));}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const buffer=device.createBuffer({size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(buffer,0,new Float32Array([-3.25,-6,-3.25]));
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(3);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,buffer);
   assert.ok(Math.abs(values[0]!+3.25)<1e-6,"solid padding must not shorten the waterline distance");
   assert.ok(Math.abs(values[1]!+1)<1e-6,"a real free surface on the wall edge must still bound the metric");
   assert.ok(Math.abs(values[2]!+4*3.25/(3.25+32))<1e-6,"the same sign change in open cells must remain a surface");
  }finally{buffer.destroy();}
 });
});

test("a detached liquid source preserves the pool and its solid boundary while refining",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice("hydrostatic offset detached source",async device=>{
  for(const [gravity,strict] of [[false,false],[true,false],[true,true]] as const){
   const scene=sceneDocument(getSceneDefinition("hydrostatic-power-large-offset"));
   scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
   if(!gravity)scene.fluid.gravity_m_s2={x:0,y:0,z:0};
   const solver=await createUniformSolver(device,scene,strict?{bandExit:"strict"}:{});
   try{
    for(let step=1;step<=120;step++)await advanceUniform(solver,step/60);
    const initial=await readUniformFields(device,solver),{nx,ny}=solver.info;
    const at=(x:number,y:number,z:number)=>x+(nx+1)*(y+(ny+1)*z);
    const fineBefore=initial.tiles.filter(word=>(word&0x80000000)!==0).length;
    solver.injectLiquidBall({centre_m:{x:0,y:1.05,z:0},radius_m:0.08});
    let maxHeightError=0,maxWallError=0,refined=false;
    for(let step=1;step<=10;step++){
     await advanceUniform(solver,(120+step)/60);
     const field=await readUniformFields(device,solver);
     refined ||= field.tiles.filter(word=>(word&0x80000000)!==0).length>fineBefore;
     for(let z=20;z<=32;z+=4)for(let y=4;y<=16;y+=4)for(let x=0;x<=32;x+=4)
      assert.equal(field.phi[at(x,y,z)],initial.phi[at(x,y,z)],`gravity ${gravity}, source step ${step}: buried ${[x,y,z]}`);
     for(let z=0;z<=16;z++)for(let x=0;x<=32;x++){
      const a=field.vertex(x,15,z),b=field.vertex(x,16,z);
      assert.ok(a<0&&b>0,`source step ${step}: lost pool waterline at ${[x,z]}`);
      maxHeightError=Math.max(maxHeightError,Math.abs(.05*(15+a/(a-b))-.7625));
     }
     // Compare the represented wall zero, not just the waterline. A positive
     // solid sentinel changing magnitude can move this contour at zero speed.
     const wall=(f:typeof field,x:number,y:number)=>{
      for(let z=16;z<20;z++){const a=f.vertex(x,y,z),b=f.vertex(x,y,z+1);if(a<=0&&b>0)return .05*(z+a/(a-b));}
      throw new Error(`missing wall at ${[x,y]}`);
     };
     for(let y=4;y<=14;y++)for(let x=0;x<=32;x++)maxWallError=Math.max(maxWallError,Math.abs(wall(field,x,y)-wall(initial,x,y)));
     // The default band policy accepts up to 0.05/s residual (2.5 mm/s
     // at this h); keep a separate strict solve to distinguish that bounded
     // drift from a geometry change. Both catch the former 5–7 mm dent.
     assert.ok(maxHeightError<(gravity&&!strict ? .00025 : .0001),`gravity ${gravity}, strict ${strict}, source step ${step}: pool error ${maxHeightError} m`);
     assert.ok(maxWallError<.0001,`gravity ${gravity}, source step ${step}: wall error ${maxWallError} m`);
    }
    assert.ok(refined,"the injected curved surface must activate finer ownership");
    const frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;
    assert.ok(frame.narrowBandFlip!.count>0,"the real drop must seed surface particles");
    t.diagnostic(`detached source, gravity ${gravity}, strict ${strict}: waterline ${maxHeightError} m; wall drift ${maxWallError} m`);
   }finally{solver.destroy();}
  }
 });
});

test("catalog hydrostatic offset preserves its waterline and buried padding through retirement",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice("hydrostatic offset retirement",async device=>{
  for(const gravity of [false,true]){
   const scene=sceneDocument(getSceneDefinition("hydrostatic-power-large-offset"));
   scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
   if(!gravity)scene.fluid.gravity_m_s2={x:0,y:0,z:0};
   const solver=await createUniformSolver(device,scene);
   try{
    const initial=await readUniformFields(device,solver),{nx,ny}=solver.info;
    const at=(x:number,y:number,z:number)=>x+(nx+1)*(y+(ny+1)*z);
    const initialVolume=initial.density.reduce((sum,v)=>sum+v,0);
    let maxHeightError=0,maxSpeed=0,retiredFine=false;
    for(let step=1;step<=(gravity?240:5);step++){
     await advanceUniform(solver,step/60);
     if(step>5&&step%20!==0)continue;
     const field=await readUniformFields(device,solver);
     retiredFine ||= field.tiles.every(word=>(word&0x80000000)===0);
     // These corners have no open incident owner. They must never become
     // short-distance samples of the water behind the padded solid wall.
     for(let z=20;z<=32;z+=4)for(let y=4;y<=20;y+=4)for(let x=0;x<=32;x+=4)
      assert.equal(field.phi[at(x,y,z)],initial.phi[at(x,y,z)],`gravity ${gravity}, step ${step}: buried ${[x,y,z]}`);
     for(let z=0;z<=16;z++)for(let x=0;x<=32;x++){
      const a=field.vertex(x,15,z),b=field.vertex(x,16,z);
      assert.ok(a<0&&b>0,`step ${step}: lost waterline at ${[x,z]}`);
      maxHeightError=Math.max(maxHeightError,Math.abs(0.05*(15+a/(a-b))-.7625));
     }
     assert.ok(maxHeightError<0.0001,`step ${step}: hydrostatic height error ${maxHeightError} m`);
     const volume=field.density.reduce((sum,v)=>sum+v,0);
     assert.ok(Math.abs(volume/initialVolume-1)<0.0001,`step ${step}: volume drift ${volume/initialVolume-1}`);
     const velocity=await readMixedTexture(device,solver.velocityTexture);
     for(let i=0;i<velocity.length;i++)if(i%4!==3)maxSpeed=Math.max(maxSpeed,Math.abs(velocity[i]!));
     assert.ok(maxSpeed<0.001,`step ${step}: resting tank speed ${maxSpeed} m/s`);
    }
    if(gravity)assert.ok(retiredFine,"the run must exercise the delayed handoff from fine to coarse surface samples");
    t.diagnostic(`gravity ${gravity}: peak component ${maxSpeed} m/s; waterline error ${maxHeightError} m`);
   }finally{solver.destroy();}
  }
 });
});
