import assert from "node:assert/strict";
import test from "node:test";
import { ExactDryFrameReuse } from "../lib/svo/features/lighting-visibility/dry-frame-reuse";
import { rasterVisibilityViewsEqual } from "../lib/svo/features/lighting-visibility/raster-visibility-view";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, type SvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";
import { svoFeatureQuery } from "../lib/svo/pipeline/persistence";

test("validated dry visibility reuse defaults on while explicit false survives query round trips",()=>{
  assert.equal(DEFAULT_SVO_RENDER_TUNING.rasterStillViewReuse,true);
  const initial=svoFeatureQuery.read(new URLSearchParams());
  assert.equal(initial.svoRenderTuning.rasterStillViewReuse,true);
  for(const enabled of [false,true]){
    const query=new URLSearchParams("unrelated=kept&svoStillReuse=0");
    svoFeatureQuery.write(query,{...initial,svoRenderTuning:{...initial.svoRenderTuning,rasterStillViewReuse:enabled}});
    assert.equal(query.get("svoStillReuse"),enabled?null:"0");
    assert.equal(query.get("unrelated"),"kept");
    assert.equal(svoFeatureQuery.read(query).svoRenderTuning.rasterStillViewReuse,enabled);
  }
  assert.equal(svoFeatureQuery.read(new URLSearchParams("svoStillReuse=1")).svoRenderTuning.rasterStillViewReuse,true);
});

test("older tuning objects inherit reuse default without overwriting explicit false",()=>{
  const {rasterStillViewReuse:omitted,...older}=DEFAULT_SVO_RENDER_TUNING;
  assert.equal(omitted,true);
  assert.equal(normalizeSvoRenderTuning(older as SvoRenderTuning).rasterStillViewReuse,true);
  assert.equal(normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING,rasterStillViewReuse:false}).rasterStillViewReuse,false);
});

test("dry visibility view key ignores only the four simulation-only lanes",()=>{
  const view=Float32Array.from({length:104},(_,i)=>i+.25);
  assert.ok(rasterVisibilityViewsEqual(view,new Float32Array(view)));
  for(let lane=0;lane<view.length;lane++){
    const changed=new Float32Array(view);changed[lane]=changed[lane]!+1;
    assert.equal(rasterVisibilityViewsEqual(view,changed),[2,19,29,30].includes(lane),`view lane${lane}`);
  }
  assert.equal(rasterVisibilityViewsEqual(view,view.slice(0,32)),false);
});

test("running fluid telemetry keeps the view epoch while camera and body changes invalidate it",()=>{
  const renderer=Object.create(SparseVoxelDrySceneRenderer.prototype);
  let invalidations=0;
  Object.assign(renderer,{visibilityViewRevision:0,visibilityInputsRevision:7,visibilityViewValid:false,
    dryFrameReuse:new ExactDryFrameReuse(),dryFrameReuseEligibility:"disabled",
    renderTuning:{rasterStillViewReuse:true},experiments:{waterShadows:false},
    rasterAo:{invalidateVisibility:()=>invalidations++,visibilityCacheStatistics:{reusedFrames:0,producedFrames:0,lastFrameReused:false}}});
  const view=new Float32Array(104);view[0]=1200;view[1]=800;view[7]=.5;
  renderer.setRasterVisibilityView(view);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewRevision,1);
  const movingWater=new Float32Array(view);movingWater[2]=2;movingWater[19]=4;movingWater[29]=1/60;movingWater[30]=3;
  renderer.setRasterVisibilityView(movingWater);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewRevision,1);
  assert.equal(renderer.rasterVisibilityCacheStatistics.inputsRevision,7,"view reuse never erases scene parameter epochs");
  const camera=new Float32Array(movingWater);camera[4]=1;renderer.setRasterVisibilityView(camera);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewRevision,2);
  const body=new Float32Array(camera);body[18]=1;renderer.setRasterVisibilityView(body);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewRevision,3);
  renderer.setRasterVisibilityView(undefined);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewEligible,false);assert.equal(invalidations,1);
  renderer.setRasterVisibilityView(view);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewEligible,true);
  assert.equal(renderer.rasterVisibilityCacheStatistics.viewRevision,4);
});


test("receiver LOD and band edits invalidate dry reuse and raster visibility on their fast paths",async()=>{
  const {DEFAULT_SVO_RENDER_TUNING}=await import("../lib/svo/pipeline/svo-render-tuning");
  const {DEFAULT_SVO_LIGHTING_OPTIONS}=await import("../lib/svo/pipeline/svo-render-options");
  const renderer=Object.create(SparseVoxelDrySceneRenderer.prototype);
  let invalidations=0,lodWrites=0,bandWrites=0;
  Object.assign(renderer,{renderTuning:DEFAULT_SVO_RENDER_TUNING,lightingOptions:DEFAULT_SVO_LIGHTING_OPTIONS,dryFrameControlRevision:0,
    worldGiCacheDirty:false,rasterAo:{invalidateVisibility:()=>invalidations++},
    writeLodParams:()=>lodWrites++,writeBandParams:()=>bandWrites++});
  renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,surfaceMeshNormalStrength:.5});
  assert.equal(renderer.dryFrameControlRevision,1);assert.equal(invalidations,1);assert.equal(lodWrites,1);assert.equal(renderer.worldGiCacheDirty,false);
  renderer.setRenderTuning(DEFAULT_SVO_RENDER_TUNING);
  const before=invalidations;
  renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,nearFieldBandPixels:16});
  assert.equal(renderer.dryFrameControlRevision,3);assert.equal(invalidations,before+1);assert.equal(bandWrites,1);assert.equal(renderer.worldGiCacheDirty,false);
});
