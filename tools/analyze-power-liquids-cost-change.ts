import { readFileSync } from "node:fs";

export interface CostChangePoint {
  readonly step: number;
  readonly wall_ms: number;
  readonly bandPages: number;
  readonly dirtyRows: number;
  readonly membershipChanges: number;
  readonly displacement: number;
}

const pearson = (points: readonly CostChangePoint[], pick: (point: CostChangePoint) => number) => {
  if (points.length < 2) return 0;
  const meanX = points.reduce((sum, point) => sum + pick(point), 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.wall_ms, 0) / points.length;
  let xy = 0, xx = 0, yy = 0;
  for (const point of points) {
    const x = pick(point) - meanX, y = point.wall_ms - meanY;
    xy += x * y; xx += x * x; yy += y * y;
  }
  return xx > 0 && yy > 0 ? xy / Math.sqrt(xx * yy) : 0;
};

export function joinCostAndChange(records: readonly Record<string, unknown>[]): CostChangePoint[] {
  const census = records.filter((record) => record.phase === "octree-row-delta-census-sample")
    .map((record) => ({ ...record, sample: Number(record.sample) }))
    .filter((record) => Number.isSafeInteger(record.sample));
  // The executor performs bootstrap publications before the measured advance
  // loop. Census sample ordinals include them; progress `steps` deliberately
  // does not. Infer that stable prefix from the terminal record so windows do
  // not silently join step N to bootstrap-shifted sample N.
  const terminalSteps = Math.max(0, ...records.map((record) => Number(record.steps))
    .filter(Number.isSafeInteger));
  const terminalSample = Math.max(0, ...census.map((record) => record.sample));
  const sampleOffset = Math.max(0, terminalSample - terminalSteps);
  const points: CostChangePoint[] = [];
  for (const progress of records.filter((record) => record.record === "progress")) {
    const end = Number(progress.steps), width = Number(progress.windowSteps);
    if (!Number.isSafeInteger(end) || !Number.isSafeInteger(width) || width < 1) continue;
    const sampleEnd = end + sampleOffset;
    const window = census.filter((record) =>
      record.sample > sampleEnd - width && record.sample <= sampleEnd);
    if (window.length === 0) continue;
    const sum = (pick: (record: Record<string, unknown>) => number) =>
      window.reduce((total, record) => total + pick(record), 0) / window.length;
    const finiteMean = (pick: (record: Record<string, unknown>) => number) => {
      // Fine topology publishes UINT_MAX when the producer has no bounded
      // displacement yet (bootstrap / invalid producer). That is state, not a
      // count, so omit it from this metric without discarding the useful row
      // and membership counters from the same generation.
      const values = window.map(pick).filter((value) =>
        Number.isFinite(value) && value !== 0xffff_ffff);
      return values.length > 0
        ? values.reduce((total, value) => total + value, 0) / values.length
        : Number.NaN;
    };
    points.push({ step: end, wall_ms: Number(progress.windowWallPerStep_ms),
      bandPages: sum((record) => Number((record.fine as Record<string, unknown>)?.support ?? 0)),
      dirtyRows: sum((record) => Number(record.dirty ?? 0)),
      membershipChanges: sum((record) => Number(record.added ?? 0) + Number(record.retired ?? 0)),
      displacement: finiteMean((record) =>
        Number((record.fine as Record<string, unknown>)?.displacement ?? Number.NaN)),
    });
  }
  return points.filter((point) => Object.values(point).every(Number.isFinite));
}

export function analyzeCostAndChange(records: readonly Record<string, unknown>[]) {
  const points = joinCostAndChange(records);
  const correlation = {
    bandPages: pearson(points, (point) => point.bandPages),
    dirtyRows: pearson(points, (point) => point.dirtyRows),
    membershipChanges: pearson(points, (point) => point.membershipChanges),
    displacement: pearson(points, (point) => point.displacement),
  };
  const changeCorrelation = Math.max(Math.abs(correlation.membershipChanges),
    Math.abs(correlation.displacement));
  return { schemaVersion: 1, experiment: "X-8-cost-vs-change", points, correlation,
    decision: Math.abs(correlation.bandPages) > 0.6 && changeCorrelation < 0.3
      ? "prices-existence-not-motion" : "inconclusive" };
}

if (process.argv[1]?.endsWith("analyze-power-liquids-cost-change.ts")) {
  const paths = process.argv.slice(2);
  const parseLines = (text: string) => text.split(/\r?\n/)
    .filter(Boolean).flatMap((line) => { try { return [JSON.parse(line) as Record<string, unknown>]; }
      catch { return []; } });
  if (paths.length === 0) throw new Error(
    "usage: analyze-power-liquids-cost-change.ts LOG.ndjson [...]");
  const records = paths.flatMap((path) => parseLines(readFileSync(path, "utf8")));
  console.log(JSON.stringify(analyzeCostAndChange(records), null, 2));
}
