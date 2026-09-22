import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SPARSE_CM12_DAWN_LANES } from "../tools/sparse-cm12-dawn-regression-manifest";

test("the suite stays discoverable from package scripts, README, and agent guidance", () => {
  const packageJson = JSON.parse(readFileSync(
    new URL("../package.json", import.meta.url), "utf8")) as {
      scripts?: Record<string, string>;
    };
  assert.ok(packageJson.scripts?.["test:dawn:sparse-cm12"]?.trim(),
    "the documented regression command must have an executable package script");
  const expectedCommand = "npm run test:dawn:sparse-cm12";
  assert.match(readFileSync(new URL("../README.md", import.meta.url), "utf8"),
    new RegExp(expectedCommand.replaceAll(" ", "\\s+")));
  assert.match(readFileSync(new URL("../AGENTS.md", import.meta.url), "utf8"),
    new RegExp(expectedCommand.replaceAll(" ", "\\s+")));
});

test("checked-in performance baselines match the executable manifest", () => {
  const baselines = JSON.parse(readFileSync(new URL(
    "../benchmarks/results/sparse-cm12-dawn-regression-baselines.json",
    import.meta.url), "utf8")) as {
      performance: Record<string, { referenceMs: number; maximumMs: number }>;
    };
  for (const lane of SPARSE_CM12_DAWN_LANES) {
    if (lane.kind !== "performance") continue;
    const baseline = baselines.performance[
      lane.referenceBaselineKey];
    assert.ok(baseline, `${lane.id} has no checked-in baseline receipt`);
    assert.equal(lane.referenceMedianAdvanceMs, baseline.referenceMs);
    assert.equal(lane.maximumMedianAdvanceMs, baseline.maximumMs);
  }
});

