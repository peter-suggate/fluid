import assert from "node:assert/strict";
import test from "node:test";

import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

test("all static opaque rays use the SVO hierarchy and brick payload", () => {
  const primaryStart = svoDrySceneShader.indexOf("fn traceStatic(");
  const primaryEnd = svoDrySceneShader.indexOf("struct DryGlassHit", primaryStart);
  const primary = svoDrySceneShader.slice(primaryStart, primaryEnd);
  assert.match(primary, /dryTraversalCursorBegin/);
  assert.match(primary, /dryTraversalCursorNext/);
  assert.match(primary, /traceLeafPayload/);
  assert.doesNotMatch(primary, /tracePrimitiveCandidates|dry\.metadata\.x\s*[<>]=?/);

  const visibilityStart = svoDrySceneShader.indexOf("fn svoVisibilityNext(");
  const visibilityEnd = svoDrySceneShader.indexOf("fn dryLightVisibility(", visibilityStart);
  const visibility = svoDrySceneShader.slice(visibilityStart, visibilityEnd);
  assert.match(visibility, /dryTraversalCursorBegin/);
  assert.match(visibility, /dryTraversalCursorNext/);
  assert.match(visibility, /traceLeafPayloadVisibility/);
  assert.doesNotMatch(visibility, /tracePrimitiveCandidates|dry\.metadata\.x\s*[<>]=?\s*\d/);
});
