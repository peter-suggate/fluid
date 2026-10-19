/** Solver-independent spatial accounting. Timings are exclusive wall intervals;
 * weights are an attribution model, never independent timers on GPU invocations. */
export type WorkTier = 0 | 1 | 2;
export interface PerformanceStageRule {
  id: string;
  label: string;
  /** Exact producer-owned trace labels. Never infer stages from a regex. */
  phases: readonly string[];
  features: readonly string[];
}
export interface SpatialPerformancePlugin {
  id: string;
  revision: number;
  stages: readonly PerformanceStageRule[];
}
export interface SpatialWorkCell {
  origin: readonly [number, number, number];
  size: readonly [number, number, number];
  /** Number of actual solver owners in this tile, not geometric h voxels. */
  owners: number;
  tier: WorkTier;
  region: number;
  features: Readonly<Record<string, number>>;
}
export interface SpatialTiming {
  frame: number;
  wall_ms: number;
  phases: readonly { label: string; duration_ms: number }[];
}
export interface CostFit {
  coefficients: number[];
  overhead: number;
  overheadIdentified: boolean;
  validationError: number;
  samples: number;
  validated: boolean;
  /** Predictive accuracy can be established even when constant work cannot identify spatial coefficients. */
  predictable: boolean;
  ranges: readonly (readonly [number, number])[];
}
interface Observation { x: number[]; y: number }
/** Non-negative, scale-conditioned coordinate descent. The intercept is shared
 * launch/scheduling cost; it is never smeared onto every cell. */
export function fitCost(observations: readonly Observation[], features: number): CostFit {
  const train = observations.filter((_, i) => i % 5 !== 4), test = observations.filter((_, i) => i % 5 === 4);
  const ranges = Array.from({ length: features }, (_, j) => [Math.min(...train.map(o => o.x[j] ?? 0)), Math.max(...train.map(o => o.x[j] ?? 0))] as const);
  if (!train.length) return { coefficients: Array(features).fill(0), overhead: 0, overheadIdentified: false, samples: 0, validationError: Infinity, validated: false, predictable: false, ranges: Array.from({ length: features }, () => [0, 0]) };
  const scales = [1, ...ranges.map(r => Math.max(1, r[1]))];
  const rows = train.map(o => [1, ...o.x.map((v, j) => v / scales[j + 1]!)]);
  const coefficients = Array(features + 1).fill(0) as number[], residual = train.map(o => o.y);
  for (let iteration = 0; iteration < 160; iteration++) {
    for (let j = 0; j <= features; j++) {
      let dot = 0, norm = 0;
      for (let i = 0; i < rows.length; i++) { const x = rows[i]![j]!; dot += x * residual[i]!; norm += x * x; }
      if (norm < 1e-12) continue;
      const next = Math.max(0, coefficients[j]! + dot / norm), delta = next - coefficients[j]!;
      coefficients[j] = next;
      for (let i = 0; i < rows.length; i++) residual[i]! -= delta * rows[i]![j]!;
    }
  }
  const c = coefficients.map((v, j) => v / scales[j]!);
  const predict = (o: Observation) => c[0]! + o.x.reduce((s, v, j) => s + v * c[j + 1]!, 0);
  const error = test.reduce((s, o) => s + Math.abs(predict(o) - o.y), 0) / Math.max(1e-9, test.reduce((s, o) => s + o.y, 0));
  // A constant workload cannot distinguish a per-cell cost from fixed overhead.
  const varying = ranges.some(([lo, hi]) => hi > 1.2 * Math.max(lo, 1));
  const predictable = train.length >= 24 && test.length >= 6 && error <= 0.15;
  return { coefficients: c.slice(1), overhead: c[0]!, overheadIdentified: predictable && train.some(o => o.x.every(v => v === 0)), samples: observations.length, validationError: test.length ? error : Infinity,
    validated: predictable && varying, predictable, ranges };
}
export interface SpatialStageAccounting {
  id: string; label: string; total_ms: number; attributed_ms: number; shared_ms: number;
  validationError: number | null; validated: boolean;
}
export interface SpatialPerformanceReport {
  frame: number; capturedAt_ms: number; wall_ms: number; stage_ms: number;
  attributed_ms: number; shared_ms: number; samples: number; cells: number; tiles: number; regions: number;
  tiers: readonly [number, number, number];
  stages: SpatialStageAccounting[];
  status: string;
  evidence: "measured-stages" | "compiled-model";
}
export interface SpatialCostSnapshot {
  cells: readonly SpatialWorkCell[];
  /** milliseconds, one value per tile, for all stages then each stage. */
  costs: Float32Array[];
  report: SpatialPerformanceReport;
}
export class SpatialCostModel {
  private history = new Map<string, Observation[]>();
  private sampleCount = 0;
  private fits = new Map<string, CostFit>();
  constructor(readonly plugin: SpatialPerformancePlugin) {
    const labels = plugin.stages.flatMap(s => s.phases);
    if (new Set(labels).size !== labels.length) throw new Error("Spatial performance stages must own disjoint trace labels");
  }
  /** Compiled mode needs only existing work evidence and one frame-completion
   * clock. Periodic diagnostic frames validate/refit these small dot products. */
  predict(cells: readonly SpatialWorkCell[], frame: number, wall_ms: number): SpatialCostSnapshot {
    const phases = this.plugin.stages.map(rule => {
      const fit = this.fits.get(rule.id);
      const duration_ms = fit ? fit.overhead + rule.features.reduce((sum, f, j) => sum + fit.coefficients[j]! * cells.reduce((n, c) => n + Math.max(0, c.features[f] ?? 0), 0), 0) : 0;
      return { label: rule.phases[0]!, duration_ms };
    });
    return this.account(cells, { frame, wall_ms, phases }, false);
  }
  canPredict(cells: readonly SpatialWorkCell[]): boolean {
    return this.plugin.stages.every(rule => {
      const fit = this.fits.get(rule.id);
      if (!fit?.predictable) return false;
      return rule.features.every((f, j) => {
        const x = cells.reduce((n, c) => n + (c.features[f] ?? 0), 0);
        return x >= fit.ranges[j]![0] * .8 && x <= Math.max(1, fit.ranges[j]![1]) * 1.2;
      });
    });
  }
  profile() { return { plugin: this.plugin.id, revision: this.plugin.revision, samples: this.sampleCount, stages: Object.fromEntries(this.fits) }; }
  account(cells: readonly SpatialWorkCell[], timing: SpatialTiming, observed = true): SpatialCostSnapshot {
    if (!(Number.isFinite(timing.wall_ms) && timing.wall_ms > 0) || timing.phases.some(p => !Number.isFinite(p.duration_ms) || p.duration_ms < -1e-9)) throw new Error("Invalid spatial timing");
    if (cells.some(c => ![...c.origin, ...c.size, ...Object.values(c.features)].every(Number.isFinite)
      || c.size.some(v => v <= 0) || !Number.isInteger(c.owners) || c.owners <= 0
      || ![0, 1, 2].includes(c.tier) || Object.values(c.features).some(v => v < 0))) throw new Error("Invalid spatial work evidence");
    if (observed) this.sampleCount++;
    const costs = Array.from({ length: this.plugin.stages.length + 1 }, () => new Float32Array(cells.length));
    // The trace partition can append a negative rounding remainder (~1e-15 ms).
    // Accept only numerical dust; real negative/invalid intervals fail closed.
    const phases = timing.phases.map(p => ({ ...p, duration_ms: Math.max(0, p.duration_ms) }));
    const gpu = phases.reduce((s, p) => s + p.duration_ms, 0);
    // The CPU and GPU overlap. Do not add them, and do not manufacture more
    // wall time than was observed when timestamp calibration has jitter.
    const wallScale = Math.min(1, timing.wall_ms / Math.max(gpu, 1e-9));
    const stages = this.plugin.stages.map((rule, stage) => {
      const measured = phases.filter(p => rule.phases.includes(p.label)).reduce((s, p) => s + p.duration_ms, 0) * wallScale;
      const x = rule.features.map(f => cells.reduce((s, c) => s + Math.max(0, c.features[f] ?? 0), 0));
      const history = this.history.get(rule.id) ?? []; this.history.set(rule.id, history);
      // Holdout is chronological: every fifth observation is never fitted.
      // Bound both memory and adaptation time after an algorithm/workload change.
      if (observed) { history.push({ x, y: measured }); if (history.length > 80) history.splice(0, 5); }
      const fit = observed ? fitCost(history, rule.features.length) : this.fits.get(rule.id) ?? fitCost([], rule.features.length);
      if (observed) this.fits.set(rule.id, fit);
      const inRange = x.every((v, j) => v >= fit.ranges[j]![0] * 0.8 && v <= Math.max(1, fit.ranges[j]![1]) * 1.2);
      const validated = fit.validated && inRange;
      // A provisional cost is explicitly hatched in the view. It uses work
      // units, not a fabricated microsecond coefficient. Zero work remains zero.
      const coefficients = fit.coefficients.some(c => c > 0) ? fit.coefficients : rule.features.map(() => 1);
      const weights = cells.map(c => rule.features.reduce((s, f, j) => s + Math.max(0, c.features[f] ?? 0) * coefficients[j]!, 0));
      const total = weights.reduce((s, w) => s + w, 0);
      // A good fit over a narrow nonzero workload does not identify a fixed
      // overhead. Require an observed zero-work reference before subtracting it.
      const budget = total > 0 ? Math.max(0, measured - (fit.overheadIdentified ? Math.min(measured, fit.overhead) : 0)) : 0;
      weights.forEach((w, i) => { const value = budget * w / Math.max(total, 1e-30); costs[stage + 1]![i] = value; costs[0]![i]! += value; });
      return { id: rule.id, label: rule.label, total_ms: measured, attributed_ms: budget, shared_ms: measured - budget,
        validationError: Number.isFinite(fit.validationError) ? fit.validationError : null, validated };
    });
    const attributed = stages.reduce((s, p) => s + p.attributed_ms, 0);
    const regions = new Map<number, WorkTier>(); cells.forEach(c => regions.set(c.region, Math.max(regions.get(c.region) ?? 0, c.tier) as WorkTier));
    const tiers: [number, number, number] = [0, 0, 0]; regions.forEach(t => tiers[t]++);
    const validated = stages.filter(s => s.total_ms > 0).every(s => s.validated);
    return { cells, costs, report: { frame: timing.frame, capturedAt_ms: performance.now(), wall_ms: timing.wall_ms, stage_ms: gpu,
      attributed_ms: attributed, shared_ms: Math.max(0, timing.wall_ms - attributed), samples: this.sampleCount,
      evidence: observed ? "measured-stages" : "compiled-model", cells: cells.reduce((s, c) => s + c.owners, 0), tiles: cells.length, regions: regions.size, tiers, stages,
      status: validated ? "Frame-cost model validated; spatial distribution is estimated" : "Calibrating spatial estimates · hatched costs are provisional" } };
  }
}
