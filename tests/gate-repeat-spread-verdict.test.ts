import assert from "node:assert/strict";
import test from "node:test";

import { METRICS, SPREAD_FACTOR, repeatSpreadVerdict, type Metric } from "../tools/gate-repeat-spread-verdict";

/** Every metric's series: instance i holds value(i, frame) for frames 1..frames. */
function series(instances: number, frames: number, value: (metric: Metric, i: number, frame: number) => number) {
  return Object.fromEntries(METRICS.map(m => [m, Array.from({ length: instances }, (_, i) =>
    new Map(Array.from({ length: frames }, (_, f) => [f + 1, value(m, i, f + 1)])))])) as Record<Metric, Map<number, number>[]>;
}
const arms: (0 | 1)[] = [0, 0, 0, 1, 1, 1];
const worst = (report: Record<string, unknown>, m: Metric) => (report[m] as { worst: { ratio: number } }).worst.ratio;

test("bit-identical repeats keep the noise band exactly", () => {
  // Baseline 1000 everywhere; the candidate off by 0.19% passes and by 0.21% fails (0.2% to frame 60).
  for (const [shift, pass] of [[1.9, true], [2.1, false]] as const) {
    const s = series(6, 30, (m, i) => m === "volumeDrift" ? 0 : 1000 + (arms[i] && m === "bandTiles" ? shift : 0));
    const v = repeatSpreadVerdict(arms, s);
    assert.equal(v.pass, pass, `shift ${shift}`);
    assert.ok(Math.abs(worst(v.report, "bandTiles") - shift / 2) < 1e-9);
  }
});

test("a deviation is scored against SPREAD_FACTOR times the largest window spread", () => {
  // Each arm's instances spread 10 apart in the last window only; the arms' means differ by d there.
  const spread = 10, edge = SPREAD_FACTOR * spread;
  for (const [d, pass] of [[edge - 0.5, true], [edge + 0.5, false]] as const) {
    const s = series(6, 30, (m, i, f) => {
      if (m !== "particles") return m === "volumeDrift" ? 0 : 1000;
      const late = f > 20, spreadOf = [0, spread / 2, spread][i % 3]!;
      return 100 + (late ? spreadOf : 0) + (late && arms[i] ? d : 0);
    });
    const v = repeatSpreadVerdict(arms, s);
    assert.equal(v.pass, pass, `d ${d}`);
    assert.ok(Math.abs(worst(v.report, "particles") - d / edge) < 1e-9);
  }
});

test("volume drift takes the band absolute", () => {
  const s = series(4, 20, (m, i) => m === "volumeDrift" ? (i >= 2 ? 0.0025 : 0) : 500);
  const v = repeatSpreadVerdict([0, 0, 1, 1], s);
  assert.equal(v.pass, false);
  assert.ok(Math.abs(worst(v.report, "volumeDrift") - 1.25) < 1e-9);
});

test("missing or non-finite measurements fail, including JSON nulls", () => {
  for (const invalid of [NaN, Infinity, -Infinity, null]) {
    const s = series(6, 20, () => 100);
    s.volumeDrift[0]!.set(10, invalid as number);
    const result = repeatSpreadVerdict(arms, s);
    assert.equal(result.pass, false, String(invalid));
    assert.match((result.report.volumeDrift as { error: string }).error, /finite measurements/);
  }
});

test("empty or disjoint frame sets cannot pass the gate", () => {
  for (const disjoint of [false, true]) {
    const s = series(6, 20, () => 100);
    s.particles[0] = disjoint ? new Map([[21, 100]]) : new Map();
    const result = repeatSpreadVerdict(arms, s);
    assert.equal(result.pass, false);
    assert.match((result.report.particles as { error: string }).error, /No frames/);
  }
});

test("both arms and every instance's series are required", () => {
  assert.equal(repeatSpreadVerdict([], series(0, 20, () => 100)).pass, false);
  assert.equal(repeatSpreadVerdict([0, 0, 0, 0], series(4, 20, () => 100)).pass, false);
  const s = series(6, 20, () => 100);
  s.bandTiles.pop();
  assert.equal(repeatSpreadVerdict(arms, s).pass, false);
});

test("overflowing window statistics cannot pass the gate", () => {
  assert.equal(repeatSpreadVerdict(arms, series(6, 20, () => Number.MAX_VALUE)).pass, false);
});

test("lagged endpoints still score the common measured frames", () => {
  const s = series(6, 20, () => 100);
  for (const m of METRICS) s[m][0]!.delete(20);
  assert.equal(repeatSpreadVerdict(arms, s).pass, true);
});
