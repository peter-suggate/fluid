import { RASTER_LIGHTING_PARAM_BYTES } from "../lib/svo/features/lighting-visibility/raster-local-shadows";
import assert from "node:assert/strict";
import test from "node:test";
import { SvoRasterAo, rasterAoConsumerWGSL, rasterAoProducerWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { lightingTuningQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

test("shadow resolution defaults and saved choices remain discrete",()=>{
  assert.equal(DEFAULT_SVO_RENDER_TUNING.rasterShadowResolution,2048);
  for(const value of [2048,4096] as const){
    const tuning={...DEFAULT_SVO_RENDER_TUNING,rasterShadowResolution:value};
    const query=new URLSearchParams();lightingTuningQuery.write(query,tuning);
    assert.equal(lightingTuningQuery.read(query).rasterShadowResolution,value);
    assert.equal(query.has("svoShadowResolution"),value!==2048);
  }
  for(const value of [undefined,0,3000,8192,NaN]){
    assert.equal(normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING,rasterShadowResolution:value as 2048}).rasterShadowResolution,2048);
    assert.equal(lightingTuningQuery.read(new URLSearchParams(`svoShadowResolution=${value}`)).rasterShadowResolution,2048);
  }
});

test("live shadow resolution routes only to shadow attachments",()=>{
  const renderer=Object.create(SparseVoxelDrySceneRenderer.prototype);
  const calls:number[]=[];const scene={},source={},pipeline={};
  Object.assign(renderer,{renderTuning:DEFAULT_SVO_RENDER_TUNING,lightingOptions:DEFAULT_SVO_LIGHTING_OPTIONS,
    scene,source,surfaceMeshPipeline:pipeline,worldGiCacheDirty:false,
    rasterAo:{setShadowResolution:(size:number)=>calls.push(size)},
    writeParams:()=>assert.fail("shadow-only change must not rebuild scene parameters"),
    writeBandParams:()=>assert.fail("shadow-only change must not rebuild geometry")});
  renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,rasterShadowResolution:4096});
  renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,rasterShadowResolution:4096});
  renderer.setRenderTuning(DEFAULT_SVO_RENDER_TUNING);
  assert.deepEqual(calls,[4096,2048]);
  assert.equal(renderer.scene,scene);assert.equal(renderer.source,source);
  assert.equal(renderer.surfaceMeshPipeline,pipeline);assert.equal(renderer.worldGiCacheDirty,false);
});

test("GPU shadow resizing preserves screen targets and native casters, with matching producer dimensions",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const {readMixedBuffer}=await import("./helpers/uniform-mixed-native-fields");
  await withUniformDevice("live shadow resolution",async device=>{
    const raster=new SvoRasterAo(device);
    const internal=raster as unknown as Record<string,any>;
    const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      raster.update([-32,0,-40],[64,8,80],[.25,.25,.25],[0,1,0]);raster.ensureSize(32,24);
      raster.setNativeDrawArguments(new Uint32Array([3,1,0,0,0,1]));
      const stable=Object.fromEntries(["params","cache","ao","aoRange","aoImportance","shadowVisibility","lightVisibility","guide","tileRequests","exactRequests","nativeDrawArguments","shadowGroups"].map(k=>[k,internal[k]]));
      const source=rasterAoConsumerWGSL();
      const declarations=source.slice(source.indexOf("struct RasterSpot"),source.indexOf("@group(2) @binding(21)"));
      const module=device.createShaderModule({code:declarations.replaceAll("@group(2)","@group(0)")+`
@group(0) @binding(1) var<storage,read_write> output:vec4u;
@compute @workgroup_size(1) fn main(){output=vec4u(u32(rasterShadowMapSize()),rasterLighting.control.w&1u,rasterLighting.control.w&2u,rasterLighting.control.x);}`});
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:raster.params,offset:0,size:RASTER_LIGHTING_PARAM_BYTES}},{binding:1,resource:{buffer:output}}]});
      let generation=0;
      for(const size of [2048,4096,2048]){
        const prior=raster.shadow,priorConsumer=raster.consumer;
        raster.setShadowResolution(size);
        assert.equal(raster.shadow.width,size);assert.equal(raster.shadow.height,size);
        if(prior.width!==size){assert.notEqual(raster.shadow,prior);assert.notEqual(raster.consumer,priorConsumer);}
        for(const [key,value] of Object.entries(stable))assert.equal(internal[key],value,`${key} must survive shadow resizing`);
        for(const enabled of [false,true]){
          raster.setVisibilityEnabled(enabled);raster.setVisibilityTilingEnabled(true);
          const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
          const values=new Uint32Array((await readMixedBuffer(device,output)).buffer);
          assert.equal(values[0],size,"caster terrain and clear receive actual attachment width");
          assert.equal(values[1],enabled?1:0,"resolution bits cannot enable cached visibility");
          assert.equal(values[2],enabled?2:0);assert.ok(values[3]!>=generation);generation=values[3]!;
        }
      }
      assert.ok(rasterAoProducerWGSL(true).includes("pixel.xy-vec2f(rasterAtlasTile(layer).xy)"));
    }finally{output.destroy();raster.destroy();}
  });
});
