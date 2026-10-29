import assert from "node:assert/strict";
import test from "node:test";
import {svoFeatureQuery} from "../lib/svo/pipeline/persistence";
import {SVO_PIPELINE_FEATURES} from "../lib/svo/pipeline/composition";
import {SVO_PRESENTATION_FEATURE} from "../lib/svo/features/presentation/definition";
import {DEFAULT_SVO_RENDER_TUNING} from "../lib/svo/pipeline/svo-render-tuning";

test("presentation resolution is installed directly in the RHS Performance slot",()=>{
 assert.ok(SVO_PIPELINE_FEATURES.includes(SVO_PRESENTATION_FEATURE));
 assert.deepEqual(SVO_PRESENTATION_FEATURE.placements,[{slot:"view.performance",control:"resolution",presentation:"compact"}]);
});
test("supersampling and per-cone budget persist without altering defaults",()=>{
 const initial=svoFeatureQuery.read(new URLSearchParams());
 assert.equal(initial.svoRenderTuning.resolutionScale,DEFAULT_SVO_RENDER_TUNING.resolutionScale);
 assert.equal(initial.svoRenderTuning.coneStepBudget,DEFAULT_SVO_RENDER_TUNING.coneStepBudget);
 for(const [resolutionScale,coneStepBudget] of [[2,24],[.35,1],[1.5,48]]){
  const desired={...initial,svoRenderTuning:{...initial.svoRenderTuning,resolutionScale:resolutionScale!,coneStepBudget:coneStepBudget!}};
  const query=new URLSearchParams("unrelated=preserved");svoFeatureQuery.write(query,desired);
  assert.equal(query.get("unrelated"),"preserved");assert.deepEqual(svoFeatureQuery.read(query),desired);
 }
 const invalid=svoFeatureQuery.read(new URLSearchParams("svoResolution=3&svoConeSteps=100"));
 assert.equal(invalid.svoRenderTuning.resolutionScale,DEFAULT_SVO_RENDER_TUNING.resolutionScale);
 assert.equal(invalid.svoRenderTuning.coneStepBudget,DEFAULT_SVO_RENDER_TUNING.coneStepBudget);
});
