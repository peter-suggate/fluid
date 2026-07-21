import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_DERIVED_LOOKUP_REDUCTION_GATE,
  formatSvoDerivedBenchmarkReport,
  parseSvoDerivedBenchmarkArgs,
  runSvoDerivedHierarchyBenchmark,
} from "../tools/benchmark-svo-derived-hierarchy";

test("derived hierarchy benchmark arguments are deterministic and gate only when requested", () => {
  assert.deepEqual(parseSvoDerivedBenchmarkArgs([]), { rays: 4_096, seed: 0x5eeda11, gate: false, json: false });
  assert.deepEqual(parseSvoDerivedBenchmarkArgs(["--rays=32", "--seed", "17", "--gate", "--json"]), {
    rays: 32, seed: 17, gate: true, json: true,
  });
  assert.throws(() => parseSvoDerivedBenchmarkArgs(["--rays=0"]), /positive integer/);
  assert.throws(() => parseSvoDerivedBenchmarkArgs(["--unknown"]), /Unknown benchmark argument/);
});
test("derived hierarchy benchmark reports repeatable visit and memory telemetry", () => {
  const first = runSvoDerivedHierarchyBenchmark({ rays: 48, seed: 1234 });
  const second = runSvoDerivedHierarchyBenchmark({ rays: 48, seed: 1234 });
  assert.deepEqual(second, first);
  assert.equal(first.rayCount, 48);
  assert.equal(first.canonical.failures, 0);
  assert.equal(first.wide.failures, 0);
  assert.deepEqual(first.outputParity, { mismatches: 0, exactMatches: 48 });
  assert.ok(first.canonical.nodeVisits > first.wide.pageVisits);
  assert.ok(first.estimatedLookupReduction >= SVO_DERIVED_LOOKUP_REDUCTION_GATE);
  assert.equal(first.gatePassed, true);
  assert.ok(first.memory.wideBytes > 0 && first.memory.mipPages > 0 && first.memory.mipAllocatedBytes > 0);
  assert.match(formatSvoDerivedBenchmarkReport(first), /estimated lookup reduction: [\d.]+% \(gate >= 35%: PASS\)/);
  assert.match(formatSvoDerivedBenchmarkReport(first), /output parity: 48\/48 exact, mismatches=0/);
  assert.match(formatSvoDerivedBenchmarkReport(first), /mip pages=\d+/);
});
