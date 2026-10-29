import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createPaneSession } from "../lib/core/session/session";
import { uiFeatureQuery } from "../lib/features/persistence";
import { parseQueryState } from "../lib/core/url-state";
import { DEFAULT_LENS_BLOOM_OPTIONS } from "../lib/core/webgpu-lens-bloom";

test("lens bloom is off by default and presentation settings round trip independently of scene and panes",()=>{
  const a=createPaneSession("a"),b=createPaneSession("b"),scene=a.scene.getState().scene;
  assert.equal(a.ui.getState().lensBloomEnabled,false);
  assert.equal(a.ui.getState().lensBloomStrength,DEFAULT_LENS_BLOOM_OPTIONS.strength);
  a.ui.getState().setLensBloom({lensBloomEnabled:true,lensBloomStrength:.2,lensBloomThreshold:2,lensBloomRadius:.05});
  assert.equal(a.scene.getState().scene,scene);
  assert.equal(b.ui.getState().lensBloomEnabled,false);
  const query=new URLSearchParams();uiFeatureQuery.write(query,a.ui.getState());
  const restored=parseQueryState(`?${query}`).ui;
  assert.equal(restored.lensBloomEnabled,true);assert.equal(restored.lensBloomStrength,.2);
  assert.equal(restored.lensBloomThreshold,2);assert.equal(restored.lensBloomRadius,.05);
  a.ui.getState().setLensBloom({lensBloomEnabled:false});uiFeatureQuery.write(query,a.ui.getState());
  assert.equal(query.has("lensBloom"),false);
  assert.equal(parseQueryState(`?${query}`).ui.lensBloomEnabled,false);
  assert.throws(()=>a.ui.getState().setLensBloom({lensBloomRadius:1}),/Invalid lensBloomRadius/);
});
