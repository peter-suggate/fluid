import type { SpatialCostSnapshot } from "../../core/spatial-performance";
export interface PerformanceView {
  stage: string;
  sampling: "adaptive" | "diagnostic";
  grain: "cell" | "tile" | "region";
  dimension: "slice" | "3d";
  paint: "cost" | "tiers";
}
export const DEFAULT_PERFORMANCE_VIEW: PerformanceView = { stage: "all", sampling: "adaptive", grain: "tile", dimension: "slice", paint: "cost" };
export function readPerformanceView(value: unknown): PerformanceView {
  const v = value as Partial<PerformanceView> | undefined;
  return { sampling: v?.sampling === "diagnostic" ? "diagnostic" : "adaptive", stage: typeof v?.stage === "string" && v.stage.length < 80 ? v.stage : "all",
    grain: v?.grain === "cell" || v?.grain === "region" ? v.grain : "tile",
    dimension: v?.dimension === "3d" ? "3d" : "slice", paint: v?.paint === "tiers" ? "tiers" : "cost" };
}
export const PERFORMANCE_RAMP = ["#182138", "#46448d", "#218eaa", "#77c8a4", "#fee68a"] as const;
/** Render records: origin.xyz / size.x; size.yz / tier / owner count;
 * attributed ms / model validation / reserved.xy. All aggregation is additive. */
export function performanceViewRecords(snapshot: SpatialCostSnapshot, view: PerformanceView): { records: Float32Array<ArrayBuffer>; count: number; entities: number; stage: number; validated: boolean } {
  const index = view.stage === "all" ? -1 : snapshot.report.stages.findIndex(s => s.id === view.stage);
  // Unknown persisted stage names must not silently become "All".
  if (view.stage !== "all" && index < 0) return { records: new Float32Array(), count: 0, entities: 0, stage: -1, validated: false };
  const values = snapshot.costs[index + 1]!;
  const validated = index < 0 ? snapshot.report.stages.filter(s => s.total_ms > 0).every(s => s.validated) : snapshot.report.stages[index]!.validated;
  const rows: number[][] = [];
  if (view.grain === "region") {
    const groups = new Map<number, { low: number[]; high: number[]; tier: number; cost: number }>();
    snapshot.cells.forEach((c, i) => {
      const group = groups.get(c.region);
      if (!group) groups.set(c.region, { low: [...c.origin], high: c.origin.map((v, a) => v + c.size[a]!), tier: c.tier, cost: values[i]! });
      else { c.origin.forEach((v, a) => { group.low[a] = Math.min(group.low[a]!, v); group.high[a] = Math.max(group.high[a]!, v + c.size[a]!); }); group.tier = Math.max(group.tier, c.tier); group.cost += values[i]!; }
    });
    for (const g of groups.values()) rows.push([...g.low, g.high[0]! - g.low[0]!, g.high[1]! - g.low[1]!, g.high[2]! - g.low[2]!, g.tier, 1, g.cost, +validated, 0, 0]);
  } else snapshot.cells.forEach((c, i) => rows.push([...c.origin, ...c.size, c.tier, view.grain === "cell" ? c.owners : 1, values[i]!, +validated, 0, 0]));
  return { records: Float32Array.from(rows.flat()), count: rows.length, entities: view.grain === "cell" ? snapshot.report.cells : rows.length, stage: index, validated };
}
