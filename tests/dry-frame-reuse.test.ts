import assert from "node:assert/strict";
import test from "node:test";
import { ExactDryFrameReuse, canReuseDryFrame, type DryFrameReuseEligibility, type DryFrameReuseKey } from "../lib/svo/features/lighting-visibility/dry-frame-reuse";
const eligible:DryFrameReuseEligibility={enabled:true,viewValid:true,splitReady:true,meshReady:true,derivedSettled:true,hasBodies:false,waterShadows:false,diagnostic:false,evolvingVisibilityCache:false};
function key():DryFrameReuseKey{return {viewRevision:1,parameterRevision:2,controlRevision:1,sourceRevision:3,sceneRevision:4,meshReceipt:"complete:5",sceneMaintenanceActivity:6,derivedContentRevision:7,source:{},scene:{},bindings:{},pipeline:{},optimizedPipeline:{},precomputedPipeline:{},target:{},gBuffer:{}};}
test("exact dry frame reuse requires static settled deterministic inputs",()=>{
 assert.equal(canReuseDryFrame(eligible),true);
 for(const name of Object.keys(eligible) as (keyof DryFrameReuseEligibility)[]){assert.equal(canReuseDryFrame({...eligible,[name]:!eligible[name]}),false,name);}
});
test("exact dry frame reuse invalidates every consumed revision and resource",()=>{
 const base=key();
 for(const field of Object.keys(base) as (keyof DryFrameReuseKey)[]){
  const cache=new ExactDryFrameReuse();assert.equal(cache.matches(base),false);cache.publish(base);assert.equal(cache.matches({...base}),true);
  const old=base[field];const changed=typeof old==="object"?{}:typeof old==="number"?old+1:old+":changed";
  assert.equal(cache.matches({...base,[field]:changed}),false,field);
 }
});
test("unknown eligibility withdraws retained imagery; only completed encode publishes",()=>{
 const cache=new ExactDryFrameReuse(),base=key();cache.publish(base);assert.equal(cache.matches(undefined),false);assert.equal(cache.matches(base),false);
 cache.publish(base);assert.equal(cache.matches({...base,derivedContentRevision:base.derivedContentRevision+1}),false,"same hierarchy generation with a new feedback write invalidates");
 assert.equal(cache.matches(base),true);assert.deepEqual(cache.statistics,{reusedFrames:1,renderedFrames:2,retained:true});
});

test("production tuning fast paths invalidate exact frames for band, LOD, filters and reuse changes",async()=>{
 const {SparseVoxelDrySceneRenderer}=await import("../lib/svo/pipeline/webgpu-svo-dry-scene");
 const {DEFAULT_SVO_RENDER_TUNING}=await import("../lib/svo/pipeline/svo-render-tuning");
 for(const delta of [
  {rasterShadowResolution:4096 as const},
  {nearFieldBandPixels:DEFAULT_SVO_RENDER_TUNING.nearFieldBandPixels+1},
  {surfaceMeshLodPixels:DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels+1},
  {surfaceMeshNormalStrength:DEFAULT_SVO_RENDER_TUNING.surfaceMeshNormalStrength===1?.5:1},
  {surfaceMeshFilteringEnabled:!DEFAULT_SVO_RENDER_TUNING.surfaceMeshFilteringEnabled},
  {rasterStillViewReuse:!DEFAULT_SVO_RENDER_TUNING.rasterStillViewReuse},
 ]){
  const receiver={renderTuning:{...DEFAULT_SVO_RENDER_TUNING},dryFrameControlRevision:0,lightingOptions:{},writeBandParams(){},writeLodParams(){}};
  const tuning={...DEFAULT_SVO_RENDER_TUNING,...delta};
  SparseVoxelDrySceneRenderer.prototype.setRenderTuning.call(receiver as never,tuning);
  assert.equal(receiver.dryFrameControlRevision,1,JSON.stringify(delta));
  SparseVoxelDrySceneRenderer.prototype.setRenderTuning.call(receiver as never,tuning);
  assert.equal(receiver.dryFrameControlRevision,1,"unchanged tuning preserves cache eligibility");
 }
});


test("production view withdrawal invalidates dry target ownership across fluid-only presentation",async()=>{
 const {SparseVoxelDrySceneRenderer}=await import("../lib/svo/pipeline/webgpu-svo-dry-scene");
 const dryFrameReuse=new ExactDryFrameReuse(),base=key();dryFrameReuse.publish(base);
 const receiver={dryFrameReuse,visibilityViewValid:true};
 SparseVoxelDrySceneRenderer.prototype.setRasterVisibilityView.call(receiver as never,undefined);
 assert.equal(receiver.visibilityViewValid,false);assert.equal(dryFrameReuse.matches(base),false);
});
