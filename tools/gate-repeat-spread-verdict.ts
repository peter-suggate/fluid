/** The verdict of tools/gate-repeat-spread.ts, on 10-frame window means.
 *
 * For each value and window: each instance's mean over the window, the A/A
 * and B/B spreads (max - min of those means within an arm) and the A/B
 * deviation |mean A - mean B|. It passes when every window's A/B deviation
 * is within the tolerance: max(noise band, SPREAD_FACTOR x spread), where
 * spread is the largest A/A or B/B window spread of that value over the
 * whole run and the noise band is the repeat-noise band (relative to the
 * baseline's window mean; volume drift, a ratio, absolute): 0.2% to frame
 * 60, 2% after (speed-of-light-plan-2026-10-20.md section 6). A value whose
 * A/A and B/B spreads are 0 (bit-identical repeats) keeps the noise band
 * alone, exactly.
 *
 * Calibration of SPREAD_FACTOR (2026-10-25; scripts fpr.py, fpr-mc.py and
 * fpr-show.py in the session scratchpad). On seep and letters the
 * instance-to-instance divergence grows through the run, so the largest
 * spread and the largest A/B gap both fall in the last windows and the rule
 * is a one-window test of |mean A - mean B| against c x max(range A,
 * range B), with r instances an arm. Null false-positive rate a metric,
 * Gaussian noise (Monte Carlo):
 *
 *   model                                 c    r=2     r=3     r=4
 *   divergence grows (last window)        1    21.7%   4.8%    1.0%
 *                                         1.66   -     1.0%      -
 *   6 independent windows, constant noise 1    9.9%    0.5%    0.0%
 *
 * c for a 1% rate under growing divergence: 5.6 at r=2 (no useful bound),
 * 1.66 at r=3, 1.0 at r=4. Pseudo A/A (every balanced split of the pooled
 * instances of 61 saved gate runs, the real split excluded), runs failing
 * any metric at c=1: letters 6/22 (r=2), 3/18 (r=3); seep 6/24 (r=2), 3/90
 * (r=3); at c=1.5 and r=3, letters 0/18 and seep 0/90. The cost in power
 * at r=3, c=1.66 (c=1 in brackets): a true shift of 2 sigma (the last
 * window's instance sigma) is caught 15% of runs (45%), 3 sigma 36% (76%),
 * 4 sigma 61% (94%). Hence 3 repeats at least for a registered scene. */

export const METRICS = ["fineTiles", "bandTiles", "bandBricks", "particles", "volumeDrift", "occupancy"] as const;
export type Metric = typeof METRICS[number];
/** The tolerance's multiple of the run's largest A/A or B/B window spread (see the calibration above). */
export const SPREAD_FACTOR = 1.66;
const WINDOW = 10;

/** The verdict over series[metric][instance] (frame -> value), each instance's arm in armOf. */
export function repeatSpreadVerdict(armOf: readonly (0 | 1)[], series: Record<Metric, Map<number, number>[]>) {
  const arms = [0, 1].map(arm => armOf.flatMap((x, i) => x === arm ? [i] : []));
  const band = (frame: number) => frame <= 60 ? 0.002 : 0.02;
  const report: Record<string, unknown> = {}; let pass = true;
  for (const m of METRICS) {
    const frames = [...series[m][0]!.keys()].filter(f => armOf.every((_, i) => series[m][i]!.has(f))).sort((x, y) => x - y);
    const windows = new Map<number, number[]>();
    for (const f of frames) { const w = Math.floor((f - 1) / WINDOW); windows.set(w, [...windows.get(w) ?? [], f]); }
    const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
    const rows = [...windows].map(([w, fs]) => {
      const means = armOf.map((_, i) => mean(fs.map(f => series[m][i]!.get(f)!)));
      const [a, b] = arms.map(arm => arm.map(i => means[i]!)) as [number[], number[]];
      const last = fs.at(-1)!;
      return { frames: `${w * WINDOW + 1}-${(w + 1) * WINDOW}`, aa: Math.max(...a) - Math.min(...a), bb: Math.max(...b) - Math.min(...b), ab: Math.abs(mean(a) - mean(b)), noise: m === "volumeDrift" ? band(last) : band(last) * Math.abs(mean(a)) };
    });
    const spread = Math.max(0, ...rows.map(r => Math.max(r.aa, r.bb)));
    let worst = { frames: "", ratio: 0, ab: 0, tolerance: 0 }, over = 0;
    for (const r of rows) {
      const tolerance = Math.max(r.noise, SPREAD_FACTOR * spread), ratio = r.ab === 0 ? 0 : tolerance === 0 ? Infinity : r.ab / tolerance;
      if (ratio > 1) over++;
      if (ratio > worst.ratio) worst = { frames: r.frames, ratio, ab: r.ab, tolerance };
    }
    pass &&= over === 0;
    report[m] = { windows: rows.length, over, worst, spread, maxAB: Math.max(0, ...rows.map(r => r.ab)) };
  }
  return { pass, report };
}
