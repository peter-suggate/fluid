import assert from "node:assert/strict";
import test from "node:test";

import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

function cursorSection(shader: string): string {
  const start = shader.indexOf("struct DryTraversalCursor");
  const end = shader.indexOf("const DRY_MISS", start);
  assert.ok(start >= 0 && end > start, "dry traversal cursor section must exist");
  return shader.slice(start, end);
}

test("hybrid remains the byte-identical default traversal shader", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1, "hybrid"), svoDrySceneShader);
  const cursor = cursorSection(svoDrySceneShader);
  assert.match(cursor, /canonical:SvoTraversalContinuation,wide:SvoWideTraversalCursor,useWide:u32/);
  assert.match(cursor, /svoWideCursorNext/);
  assert.match(cursor, /svoTraversalContinuationNext/);
});

test("canonical experiment removes wide cursor state and traversal functions", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "canonical");
  const cursor = cursorSection(shader);
  assert.match(cursor, /struct DryTraversalCursor\{canonical:SvoTraversalContinuation\}/);
  assert.doesNotMatch(shader, /SvoWideTraversalCursor|svoWideCursorInitialize|svoWideCursorNext/);
});

test("canonical-parametric experiment selects midpoint-plane enumeration without wide state", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "canonical-parametric");
  const cursor = cursorSection(shader);
  assert.match(cursor, /struct DryTraversalCursor\{canonical:SvoTraversalContinuation\}/);
  assert.match(shader, /fn svoParametricSegments\(/);
  assert.match(shader, /parametricSegments\.valid == 0u/);
  assert.doesNotMatch(shader, /SvoWideTraversalCursor|svoWideCursorInitialize|svoWideCursorNext/);
});

test("compact experiment keeps canonical continuation semantics with four aligned hot words", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "compact");
  const cursor = cursorSection(shader);
  assert.match(cursor, /struct DryTraversalCursor\{compact:SvoCompactTraversalContinuation\}/);
  assert.match(shader, /struct SvoCompactNode\{mortonLow:u32,mortonHigh:u32,levelMaskTerminal:u32,linkOrLeaf:u32\}/);
  assert.match(cursor, /svoCompactContinuationNext/);
  assert.match(shader, /fn dryLeafBounds\(nodeIndex:u32\)->mat2x3f\{return svoCompactNodeBounds\(svoCompactNodes\[nodeIndex\],dry\.mapping\);\}/);
  assert.match(shader, /fn traceLeafPayload\([^]*let bounds=dryLeafBounds\(hit\.nodeIndex\)/);
  assert.match(shader, /fn traceLeafPayloadVisibility\([^]*let bounds=dryLeafBounds\(hit\.nodeIndex\)/);
  assert.doesNotMatch(shader, /SvoWideTraversalCursor|svoWideCursorInitialize|svoWideCursorNext/);
});

test("strict-wide experiment removes canonical state and calls from the active cursor", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "wide");
  const cursor = cursorSection(shader);
  assert.match(cursor, /struct DryTraversalCursor\{wide:SvoWideTraversalCursor\}/);
  assert.match(cursor, /svoWideCursorInitialize/);
  assert.match(cursor, /svoWideCursorNext/);
  assert.doesNotMatch(cursor, /canonical:|useWide|svoTraversalContinuationBegin|svoTraversalContinuationNext/);
});

test("invalid traversal variants fail before shader generation", () => {
  assert.throws(
    () => createSvoDrySceneFragmentWGSL(1, "invalid" as never),
    /Unsupported dry-scene traversal mode/,
  );
});
