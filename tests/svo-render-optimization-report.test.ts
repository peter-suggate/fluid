import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../tools/render-svo-optimization-report.ts", import.meta.url), "utf8");

test("SVO optimization report is an offline artifact reducer with separate mode seams", () => {
  assert.match(source, /--control-xctrace=/);
  assert.match(source, /--general-xctrace=/);
  assert.match(source, /--general-benchmark=/);
  assert.match(source, /--static-xctrace=/);
  assert.match(source, /--static-benchmark=/);
  assert.match(source, /--final-xctrace/);
  assert.match(source, /Final matched xctrace: control → selected general path/);
  assert.doesNotMatch(source, /requestAdapter|xctrace\s+record|benchmark-svo-dry-frame-gpu\.ts"\]/);
});

test("report applies the measured-quality policy without hiding exactness", () => {
  assert.match(source, /Experiment decisions/);
  assert.match(source, /at least 5% median savings and at most 0\.05% changed pixels/);
  assert.doesNotMatch(source, /Reject; keep inline/);
  assert.match(source, /canonical-parametric traversal \+ split visibility\/lighting/);
  assert.match(source, /Superseded; keep split relight/);
  assert.match(source, /status: "missing-evidence"/,
    "a clean checkout emits an honest restoration manifest instead of throwing");
  assert.match(source, /Screen-space 64 px is rejected/);
  assert.match(source, /Final same-source, same-resolution fingerprint is exact/);
  assert.match(source, /presented reference PNGs are byte-identical/);
  assert.match(source, /Memory\/work tradeoff/);
  assert.match(source, /Compact 16-byte hierarchy/);
  assert.match(source, /Split visibility\/lighting/);
  assert.match(source, /Uniform brick 4/);
  assert.match(source, /Screen-space termination/);
  assert.match(source, /Static-primary ray coherence/);
});
