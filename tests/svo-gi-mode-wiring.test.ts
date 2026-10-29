import assert from "node:assert/strict";
import test from "node:test";
import { createUIStore } from "../lib/core/stores/ui-store";
import { svoFeatureQuery } from "../lib/svo/pipeline/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo/pipeline/svo-render-tuning";
import { RENDER_PIPELINE_NODES, type RenderPipelineContext } from "../lib/svo/pipeline/render-pipeline-graph";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

test("GI enable selects its cone producer while disabling preserves an explicit visibility choice",()=>{
  const store=createUIStore();
  assert.equal(DEFAULT_SVO_RENDER_TUNING.giBounceStrength,1,"enabling GI produces indirect bounce without a second strength override");
  for(const mode of ["raster-ao","exact","off"] as const){
    store.getState().setSvoConeTracingMode(mode);
    store.getState().setSvoGlobalIlluminationEnabled(true);
    assert.equal(store.getState().svoConeTracingMode,"cones");
    assert.equal(store.getState().svoGlobalIlluminationEnabled,true);
    store.getState().setSvoConeTracingMode(mode);
    store.getState().setSvoGlobalIlluminationEnabled(false);
    assert.equal(store.getState().svoConeTracingMode,mode);
  }
});

test("GI mode, cache preference and composition survive a shared URL round trip",()=>{
  const state=svoFeatureQuery.read(new URLSearchParams());
  const requested={...state,svoConeTracingMode:"cones" as const,svoGlobalIlluminationEnabled:true,svoWorldGiCacheEnabled:true,
    svoRenderTuning:{...state.svoRenderTuning,giBounceStrength:1.25,giOcclusionStrength:.4,giEnvironmentStrength:.75,giDirectStrength:1.1,giConeAperture:1.2,giConeCount:3}};
  const query=new URLSearchParams("foreign=retained");
  svoFeatureQuery.write(query,requested);
  assert.equal(query.get("foreign"),"retained");
  assert.deepEqual(svoFeatureQuery.read(query),requested);
  const off={...requested,svoGlobalIlluminationEnabled:false,svoWorldGiCacheEnabled:false};
  svoFeatureQuery.write(query,off);
  assert.deepEqual(svoFeatureQuery.read(query),off);
});

test("backend does not advertise a GI producer in exact, off or raster visibility",()=>{
  for(const mode of ["cones","exact","off","raster-ao"] as const){
    const renderer=Object.create(SparseVoxelDrySceneRenderer.prototype);
    Object.assign(renderer,{lightingOptions:DEFAULT_SVO_LIGHTING_OPTIONS,rasterAo:{},coneScale:1,
      writeVoxelLightCacheParams(){},shadingPath:"inline"});
    renderer.setLightingOptions({...DEFAULT_SVO_LIGHTING_OPTIONS,coneTracingMode:mode,globalIlluminationEnabled:true});
    assert.equal(renderer.lightingOptions.globalIlluminationEnabled,mode==="cones");
  }
});

test("GI graph distinguishes uncached reduced gathering from full-rate inline GI",()=>{
  const node=RENDER_PIPELINE_NODES.find(node=>node.id==="world-gi-cache")!;
  const context={coneTracingMode:"cones",globalIlluminationEnabled:true,worldGiCacheEnabled:false,
    tuning:DEFAULT_SVO_RENDER_TUNING} as RenderPipelineContext;
  assert.equal(node.state(context),"on");
  assert.match(node.chip(context),/uncached gather/);
  const full={...context,worldGiCacheEnabled:true,tuning:{...DEFAULT_SVO_RENDER_TUNING,coneLightingScale:1 as const}};
  assert.equal(node.state(full),"off");
  assert.match(node.chip(full),/inline GI/);
  assert.equal(node.state({...context,coneTracingMode:"exact"}),"off");
  assert.match(node.chip({...context,coneTracingMode:"exact"}),/needs cones/);
});

test("GI capability changes withhold stale split bundles even without a lattice or rate change",()=>{
  for(const scale of [1,.5] as const){
    const renderer=Object.create(SparseVoxelDrySceneRenderer.prototype);
    Object.assign(renderer,{shadingPath:"split",coneScale:scale,conePipelineScale:scale,splitPipelineScale:scale,
      splitPipelineLattice:false,splitPipelineGlobalIllumination:false,
      lightingOptions:{...DEFAULT_SVO_LIGHTING_OPTIONS,coneTracingMode:"cones",globalIlluminationEnabled:true},
      latticeVisibilityRequested:()=>false});
    assert.equal(renderer.requestedBundleStatus().state,"compiling");
    renderer.splitPipelineGlobalIllumination=true;
    assert.equal(renderer.requestedBundleStatus().state,"ready");
    renderer.lightingOptions={...renderer.lightingOptions,globalIlluminationEnabled:false};
    assert.equal(renderer.requestedBundleStatus().state,"compiling");
    renderer.splitPipelineGlobalIllumination=false;
    assert.equal(renderer.requestedBundleStatus().state,"ready");
  }
});
