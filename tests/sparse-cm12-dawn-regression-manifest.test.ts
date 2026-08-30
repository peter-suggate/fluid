import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import {
  SPARSE_CM12_DAWN_LANES,
  SPARSE_CM12_DAWN_SUITE_BUDGET_MS,
  type SparseCM12DawnCoverage,
} from "../tools/sparse-cm12-dawn-regression-manifest";

const expectedCoverage: readonly SparseCM12DawnCoverage[] = [
  "symmetric-expansion",
  "hydrostatic-stability-adaptivity",
  "mini32-correctness",
  "mini32-performance",
  "mini64-performance",
  "mini64-min8-surface",
  "long-dam-far-wall",
  "tall-cells-hills-far-wall",
  "live-rigid-body-coupling",
  "live-liquid-injection",
  "outside-tank-symmetric-collapse",
];

test("the compact Sparse CM12 Dawn suite retains its complete confidence matrix", () => {
  assert.equal(SPARSE_CM12_DAWN_SUITE_BUDGET_MS, 180_000);
  assert.deepEqual(SPARSE_CM12_DAWN_LANES.map((lane) => lane.coverage).sort(),
    [...expectedCoverage].sort());
  assert.equal(new Set(SPARSE_CM12_DAWN_LANES.map((lane) => lane.id)).size,
    SPARSE_CM12_DAWN_LANES.length, "lane ids must remain unique and selectable");

  for (const lane of SPARSE_CM12_DAWN_LANES) {
    assert.ok(lane.timeoutMs > 0 && lane.timeoutMs < SPARSE_CM12_DAWN_SUITE_BUDGET_MS,
      `${lane.id} must remain individually time bounded`);
    if (lane.kind === "correctness") {
      assert.ok(existsSync(new URL(`../${lane.testFile}`, import.meta.url)),
        `${lane.id} points at a missing Dawn authority`);
    } else {
      assert.equal(lane.brickFineResolution, 8);
      assert.equal(lane.presentationPageResolution, 8);
      assert.ok(lane.measuredFrames >= 12,
        `${lane.id} needs enough hardware samples for a stable median`);
      assert.ok(lane.maximumMedianAdvanceMs > lane.referenceMedianAdvanceMs,
        `${lane.id} ceiling must preserve explicit noise headroom`);
    }
  }
});

test("the suite stays discoverable from package scripts, README, and agent guidance", () => {
  const packageJson = JSON.parse(readFileSync(
    new URL("../package.json", import.meta.url), "utf8")) as {
      scripts?: Record<string, string>;
    };
  assert.equal(packageJson.scripts?.["test:dawn:sparse-cm12"],
    "node --import tsx tools/run-sparse-cm12-dawn-regression-suite.ts");
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
      `${lane.scene}-b${lane.brickFineResolution}-p${lane.presentationPageResolution}`];
    assert.ok(baseline, `${lane.id} has no checked-in baseline receipt`);
    assert.equal(lane.referenceMedianAdvanceMs, baseline.referenceMs);
    assert.equal(lane.maximumMedianAdvanceMs, baseline.maximumMs);
  }
});
