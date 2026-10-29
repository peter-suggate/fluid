import assert from "node:assert/strict";
import test from "node:test";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/features/shading/program";
import { integrateSvoTetrahedralRadianceCone, svoTetrahedralRadianceConeWGSL } from "../lib/svo/features/radiance/svo-tetrahedral-radiance-cone";

function productionBudget(): string {
  const shader = createSvoDrySceneFragmentWGSL(.5,"hybrid","bounds","split");
  const match = /let perConeBudget=([^;]+);/.exec(shader);
  assert.ok(match); return match[1].replace("dry.tuningCounts0.y", "configured");
}
const black = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] as const;
const cone = {origin_m:[0,0,0] as const,direction:[0,0,1] as const,
  aperture_radians:.4,minimumVoxelWidth_m:.125,maximumDistance_m:40};

test("narrow GI cones need the configured per-cone budget to reach distant sources",()=>{
  const old = integrateSvoTetrahedralRadianceCone({...cone,maximumSteps:12},()=>({coverage:0,radiance:black}));
  const current = integrateSvoTetrahedralRadianceCone({...cone,maximumSteps:48},()=>({coverage:0,radiance:black}));
  assert.equal(old.termination,"step-limit");assert.equal(current.termination,"distance");
  assert.ok(current.coneTaps>old.coneTaps);
});

test("GPU production GI budget reaches distant source, reports finite exhaustion, and retains wide-cone early exit",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("GI per-cone configured budget",async device=>{
    const module=device.createShaderModule({code:`
${svoTetrahedralRadianceConeWGSL}
var<private> emitter:bool;
fn svoTetraRadianceConeLoad(q:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
  let coverage=select(0.0,1.0,emitter&&q.distance_m>10.0);
  let c=vec3f(coverage*2.0);
  return SvoTetraRadianceConeSourceSample(coverage,SvoTetraRadiance(c,c,c,c),1u,1u);
}
@group(0) @binding(0) var<storage,read_write> out:array<vec4f,7>;
fn budget(configured:u32,coneCount:u32)->u32{return ${productionBudget()};}
@compute @workgroup_size(1) fn main(){
  for(var i=0u;i<7u;i++){
    var steps=budget(48u,4u);var aperture=.4;emitter=false;
    if(i==1u){steps=12u;}if(i==2u){steps=budget(1u,4u);}
    if(i==3u){aperture=1.05;}if(i==4u){steps=budget(100u,3u);}
    if(i>=5u){emitter=true;}if(i==6u){steps=12u;}
    let result=svoTetraRadianceConeTrace(SvoTetraRadianceConeConfig(vec3f(0),vec3f(0,0,1),aperture,.125,40.0,steps,.995,.0039215686,1u));
    out[i]=vec4f(result.radiance.x,f32(result.coneTaps),f32(result.termination),f32(result.valid));
  }
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const output=device.createBuffer({size:112,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:112,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}));pass.dispatchWorkgroups(1);pass.end();
      encoder.copyBufferToBuffer(output,0,read,0,112);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const data=new Float32Array(read.getMappedRange()).slice();read.unmap();
      assert.equal(data[2],0,"configured narrow cone reaches scene distance");
      assert.equal(data[6],2,"old divided budget exhausts");assert.equal(data[5],12);
      assert.equal(data[10],2,"one-step budget reports exhaustion");assert.equal(data[9],1);
      assert.equal(data[14],0,"wide cone terminates by distance");assert.ok(data[13]<=12,"wide-cone work remains unchanged");
      assert.equal(data[18],0);assert.ok(data[17]<=64,"hard bound retained");
      assert.ok(data[20]>1.9,"distant emitter gathered with configured budget");assert.equal(data[22],1);
      assert.equal(data[24],0,"old divided budget never reaches distant emitter");assert.equal(data[26],2);
      for(let i=0;i<7;i++)assert.equal(data[i*4+3],1,"exhaustion is finite partial data, not invalid-black failure");
    }finally{output.destroy();read.destroy();}
  });
});
