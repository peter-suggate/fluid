import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { buildSvoSceneLights } from "../lib/svo/contracts/svo-light-abi";
import type { SceneryNode } from "../lib/core/scenery-graph";

test("bounded shading keeps the brightest late-authored fixture before dim earlier fixtures",()=>{
  const scene=cloneScene(defaultScene);
  const fixtures:SceneryNode[]=Array.from({length:12},(_,i)=>({kind:"box",id:`lamp-${i}`,tags:["light","point-light"],place:{units:"metres",position:{x:i,y:2,z:0}},halfSize:{x:.1,y:.1,z:.1},material:{colorLinear:[1,1,1],emission:i===11?20:1}}));
  scene.scenery={palettes:{},nodes:[{kind:"terrain-shell",id:"shell",materialModel:"porcelain"},...fixtures]};
  const full=buildSvoSceneLights(scene),limited=buildSvoSceneLights(scene,{maximumRecords:5});
  assert.equal(full.records[0]!.kind,"directional");
  assert.equal(full.records[1]!.intensity,20,"the shader's first8 must include the strong fixture even when the table has room for all12");
  assert.deepEqual(full.records.slice(0,5),limited.records);
  assert.equal(new Set(full.records.map(r=>r.lightId)).size,13,"prioritization must preserve stable owner identities");
  assert.deepEqual(full.omittedFixtureKeys,[]);
});

test("direct sample budget defaults to8, round-trips32, and sizes raster visibility for the whole budget",async()=>{
  const {DEFAULT_SVO_RENDER_TUNING,normalizeSvoRenderTuning}=await import("../lib/svo/pipeline/svo-render-tuning");
  const {rasterVisibilityLayers}=await import("../lib/svo/features/lighting-visibility/raster-light-visibility");
  const {lightingTuningQuery}=await import("../lib/svo/features/lighting-visibility/persistence");
  assert.equal(DEFAULT_SVO_RENDER_TUNING.maximumShadedLights,8);
  assert.equal(normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING,maximumShadedLights:32}).maximumShadedLights,32);
  const records=new Uint32Array(32*28);for(let i=0;i<32;i++)records[i*28+24]=2;
  assert.equal(rasterVisibilityLayers(records,32,1,false),32);
  assert.equal(rasterVisibilityLayers(records,8,1,false),8);
  const query=new URLSearchParams();lightingTuningQuery.write(query,{...DEFAULT_SVO_RENDER_TUNING,maximumShadedLights:32});
  assert.equal(query.get("svoLightSamples"),"32");
  assert.equal(lightingTuningQuery.read(query).maximumShadedLights,32);
  assert.equal(lightingTuningQuery.read(new URLSearchParams()).maximumShadedLights,8);
});

test("GPU production shading loop reaches late light records with32samples and retains8sample default",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {drySceneShader}=await import("../lib/svo/features/shading/program");
  const body=drySceneShader.slice(drySceneShader.indexOf("fn shadeDryOpaque"));
  const loop=body.slice(body.indexOf("  let lightCount="),body.indexOf("  let viewDirection="));
  assert.ok(loop.length>400);
  await withUniformDevice("32light production loop",async device=>{
    const module=device.createShaderModule({code:`
      const SVO_LIGHT_SPHERE_AREA=3u;const SVO_LIGHT_RECTANGLE_AREA=4u;const SVO_LIGHT_SPOT=5u;
      struct Light{identity:vec4u} struct Rig{metadata:vec4u,lights:array<Light,32>} var<private> dryLighting:Rig;
      struct Dry{tuningCounts0:vec4u,tuningCounts1:vec4u} var<private> dry:Dry;
      struct Hit{normal:vec3f,ownerId:u32} const hit=Hit(vec3f(0,1,0),0u);
      struct Sample{valid:u32,towardLight:vec3f,radiance:vec3f,finiteDistance_m:f32}
      fn dryLightSample(light:Light,sample:u32,p:vec3f)->Sample{return Sample(1u,vec3f(0,1,0),vec3f(f32(light.identity.y)),1.0);}
      fn dryLightVisibility(p:vec3f,n:vec3f,o:u32,d:vec3f,t:f32,i:u32)->vec3f{return vec3f(1);}
      fn unifiedLightingInputWithGeometry(a:vec3f,b:vec3f,c:vec3f,d:vec3f,r:vec3f)->vec3f{return r;}
      fn shadeUnifiedSurface(c:f32,l:vec3f)->vec3f{return l;}
      fn shade(budget:u32)->f32{dry.tuningCounts0=vec4u(0,0,budget,1);dry.tuningCounts1=vec4u(1);var direct=vec3f(0);var sampleBudget=0u;let globalIllumination=false;let position=vec3f(0);let geometricNormal=vec3f(0,1,0);let rd=vec3f(0,-1,0);let directClosure=0.0;
        ${loop} return direct.x;}
      @group(0) @binding(0) var<storage,read_write> result:vec2f;
      @compute @workgroup_size(1) fn main(){dryLighting.metadata=vec4u(32,1,0,0);for(var i=0u;i<32u;i++){dryLighting.lights[i]=Light(vec4u(2,i+1u,0,1));}result=vec2f(shade(8),shade(32));}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const out=device.createBuffer({size:8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(out,0,read,0,8);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);assert.deepEqual([...new Float32Array(read.getMappedRange())],[36,528]);read.unmap();}finally{out.destroy();read.destroy();}
  });
});

test("GPU finite lights inside their emitter envelope retain finite shadow segments and unchanged radiance",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {drySceneShader}=await import("../lib/svo/features/shading/program");
  const {svoLightWGSL}=await import("../lib/svo/contracts/svo-light-abi");
  const sampleCode=drySceneShader.slice(drySceneShader.indexOf("struct DryLightSample"),drySceneShader.indexOf("// The static scene is traceStatic alone"));
  assert.ok(sampleCode.length>2000);
  await withUniformDevice("finite emitter endpoint",async device=>{
    const module=device.createShaderModule({code:`const UNIFIED_PI=3.141592653589793;${svoLightWGSL}${sampleCode}
      @group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
      @compute @workgroup_size(1) fn main(){var light:SvoLightRecord;light.positionRange=vec4f(0,.5,0,10);light.directionCone=vec4f(0,-1,0,.1);light.colorIntensity=vec4f(1,2,3,4);light.shape=vec4f(1,.5,0,0);light.identity=vec4u(SVO_LIGHT_POINT,1,0,1);
        let point=dryLightSample(light,0u,vec3f(0));result[0]=vec4f(point.finiteDistance_m,point.radiance.x,f32(point.valid),select(1000.0,point.finiteDistance_m,point.finiteDistance_m>0.0));
        light.identity.x=SVO_LIGHT_SPOT;let spot=dryLightSample(light,0u,vec3f(0));result[1]=vec4f(spot.finiteDistance_m,spot.radiance.x,f32(spot.valid),select(1000.0,spot.finiteDistance_m,spot.finiteDistance_m>0.0));
        light.identity.x=SVO_LIGHT_DIRECTIONAL;let sun=dryLightSample(light,0u,vec3f(0));result[2]=vec4f(sun.finiteDistance_m,sun.radiance.x,f32(sun.valid),select(1000.0,sun.finiteDistance_m,sun.finiteDistance_m>0.0));
        light.identity.x=SVO_LIGHT_POINT;let far=dryLightSample(light,0u,vec3f(0,-4.5,0));result[3]=vec4f(far.finiteDistance_m,far.radiance.x,f32(far.valid),0);}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const out=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(out,0,read,0,64);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const v=new Float32Array(read.getMappedRange()).slice();read.unmap();
      for(const offset of [0,4]){assert.equal(v[offset+2],1);assert.ok(v[offset]!>0&&v[offset]!<1e-5);assert.equal(v[offset],v[offset+3],"finite lights cannot take the directional scene-exit branch");}
      assert.ok(Math.abs(v[1]!-4*.95**2)<1e-6);
      const spotDistance=Math.hypot(.5,.45);assert.ok(Math.abs(v[5]!-4*(1-spotDistance/10)**2)<1e-5);
      assert.equal(v[8],0);assert.equal(v[9],4);assert.equal(v[11],1000);
      assert.equal(v[12],4);assert.ok(Math.abs(v[13]!-.04)<1e-6);
    }finally{out.destroy();read.destroy();}
  });
});
