import assert from "node:assert/strict";
import test from "node:test";

import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

function shaderSection(start: string, end: string): string {
  const startIndex = svoDrySceneShader.indexOf(start);
  const endIndex = svoDrySceneShader.indexOf(end, startIndex + start.length);
  assert.ok(startIndex >= 0 && endIndex > startIndex, `missing shader section ${start}`);
  return svoDrySceneShader.slice(startIndex, endIndex);
}

test("dry primary static payload traversal resumes one selected near-to-far cursor across empty leaves", () => {
  const traceStatic = shaderSection("fn traceStatic(", "struct DryGlassHit");
  assert.match(traceStatic, /var continuation:DryTraversalCursor/);
  assert.match(traceStatic, /dryTraversalCursorBegin\(SvoRay\(ro,minimum,rd,DRY_MISS\),mapping,&continuation\)/);
  assert.match(traceStatic, /dryTraversalCursorNext\(ray,mapping,&continuation\)/);
  assert.doesNotMatch(traceStatic, /dryTraverse\(/);
  assert.match(traceStatic, /dryPrimaryNodeVisits\+=leaf\.visits/);
  assert.match(traceStatic, /dryPrimaryLeafVisits\+=1u/);
  assert.match(traceStatic, /minimum=leaf\.tExit\+max\(1e-5,length\(dry\.mapping\.cellSize\)\*1e-3\)/);
});

test("dry exact visibility reuses the selected static cursor without changing later terrain and glass events", () => {
  const visibility = shaderSection("fn svoVisibilityNext(", "fn dryLightVisibility(");
  assert.match(visibility, /var shadowContinuation:DryTraversalCursor/);
  assert.match(visibility, /dryTraversalCursorBegin\(SvoRay\(ray\.origin_m,cursor,ray\.direction,bestT\),initialShadowMapping,&shadowContinuation\)/);
  assert.match(visibility, /dryTraversalCursorNext\(SvoRay\(ray\.origin_m,cursor,ray\.direction,bestT\),shadowMapping,&shadowContinuation\)/);
  assert.doesNotMatch(visibility, /dryTraverse\(/);
  assert.match(visibility, /if\(terrainEnabled\(\)\)/);
  assert.match(visibility, /traceGlass\(ray\.origin_m,ray\.direction,tMin_m,bestT,false\)/);
  assert.match(visibility, /leafVisits>=remaining\.leafVisits\|\|nodeVisits>=remaining\.nodeVisits/);
});

test("one-shot traversal remains available to dry paths outside static leaf enumeration", () => {
  assert.match(svoDrySceneShader, /fn dryTraverse\(ray:SvoRay,mapping:SvoMapping\)->SvoTraversalHit\{return svoTraverseWithDepthLimit/);
  assert.match(svoDrySceneShader, /struct DryTraversalCursor\{canonical:SvoTraversalContinuation,wide:SvoWideTraversalCursor,useWide:u32\}/);
  assert.match(svoDrySceneShader, /fn dryTraversalCursorNext\([^]*svoWideCursorNext\([^]*svoTraversalContinuationNext\(/);
});
