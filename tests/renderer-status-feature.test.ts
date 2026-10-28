import assert from "node:assert/strict";
import test from "node:test";
import { resolveEffectiveRendererStatus, type EffectiveRendererConditions } from "../lib/core/renderer-status";

test("a selected optional pass cannot report active while only the base scene encoded",()=>{
  const base:EffectiveRendererConditions={pipelineAvailable:true,sourceAvailable:true,terrainSupported:true,materialsSupported:true,lightingSupported:true,svoEncoded:true};
  assert.equal(resolveEffectiveRendererStatus(base).state,"active");
  assert.deepEqual(resolveEffectiveRendererStatus({...base,requestedFeature:{state:"pending",detail:"Compiling world reflections"}}),{state:"pending",failureReason:"pipeline-compiling",detail:"Compiling world reflections"});
  assert.deepEqual(resolveEffectiveRendererStatus({...base,requestedFeature:{state:"failed",detail:"World reflection shader failed"}}),{state:"failed",failureReason:"pipeline-compile-failure",detail:"World reflection shader failed"});
  assert.equal(resolveEffectiveRendererStatus({...base,requestedFeature:{state:"ready"}}).state,"active");
  assert.equal(resolveEffectiveRendererStatus({...base,required:false,requestedFeature:{state:"pending"}}).state,"not-required");
});
