import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const tools = [
  "benchmark-svo-cone-gpu.ts",
  "benchmark-svo-continuation-gpu.ts",
  "benchmark-svo-wide-fanout-gpu.ts",
  "benchmark-svo-traversal-gpu.ts",
] as const;

const sources = new Map(tools.map((name) => [
  name,
  readFileSync(new URL(`../tools/${name}`, import.meta.url), "utf8"),
]));

test("standalone SVO benchmarks use only the generic GPU performance recorder", () => {
  for (const [name, source] of sources) {
    assert.match(source, /import \{ GPUPerformanceTraceRecorder \} from "\.\.\/lib\/performance-trace"/, name);
    assert.match(source, /new GPUPerformanceTraceRecorder\(/, name);
    assert.match(source, /\.boundary\(/, name);
    assert.match(source, /\.resolve\(/, name);
    assert.match(source, /await recorder\.read\(\)/, name);
    assert.doesNotMatch(source,
      /createQuerySet|resolveQuerySet|timestampWrites|GPU(?:Compute|Render)PassTimestampWrites/,
      name);
  }
});

test("generic benchmark traces continue to feed the established output measurements", () => {
  assert.match(sources.get("benchmark-svo-cone-gpu.ts")!, /gpuMilliseconds:/);
  assert.match(sources.get("benchmark-svo-continuation-gpu.ts")!, /gpuMilliseconds:/);
  assert.match(sources.get("benchmark-svo-wide-fanout-gpu.ts")!, /timing: \{ canonicalMedian_ms:/);
  assert.match(sources.get("benchmark-svo-traversal-gpu.ts")!, /results: rows/);
});
