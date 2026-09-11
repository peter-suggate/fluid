import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

/** Explicitly accepted current behavior, separate from the original ideal targets. */
const baseline = JSON.parse(readFileSync(new URL(
  "../../benchmarks/results/sparse-cm12-dawn-behavior-baseline.json", import.meta.url), "utf8")) as {
  metrics: Record<string, { maximum: number; previousMaximum: number }>;
};

export function assertSparseCM12Baseline(metric: string, actual: number): void {
  const limit = baseline.metrics[metric];
  assert.ok(limit, `missing Sparse CM12 baseline for ${metric}`);
  console.log(JSON.stringify({ baselineMetric: metric, actual, maximum: limit.maximum }));
  assert.ok(Number.isFinite(actual) && actual <= limit.maximum,
    `${metric}: ${actual} exceeds accepted baseline ${limit.maximum}`);
}
