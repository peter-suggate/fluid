import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../tools/benchmark-svo-dry-frame-gpu.ts", import.meta.url), "utf8");

test("SVO dry-frame benchmark uses the generic one-shot GPU trace", () => {
  assert.match(source, /new GPUPerformanceTraceRecorder\(/);
  assert.match(source, /\[\{ id: "dry-scene", label: "SVO traversal \+ dry shading" \}\]/);
  assert.match(source, /traceRecorder\.boundary\(encoder, `\$\{label\} trace start`\)/);
  assert.match(source, /traceRecorder\.boundary\(encoder, `\$\{label\} trace end`\)/);
  assert.match(source, /traceRecorder\.resolve\(encoder\)/);
  assert.match(source, /samples\.push\(trace\.total_ms\)/);
  assert.doesNotMatch(source, /createQuerySet|resolveQuerySet|timestampWrites|GPU(?:Compute|Render)PassTimestampWrites/);
});
