"use client";
import { RailChoice, RailKey, RailSelect } from "../../../components/rail";
import { useSession } from "../../core/session/session-context";
import { DEFAULT_PERFORMANCE_VIEW, PERFORMANCE_RAMP, type PerformanceView } from "./view";
import type { VisualLayerState } from "../../core/visual-layers";
export function PerformanceControls({ state, onChange }: { state: VisualLayerState; onChange: (s: VisualLayerState) => void }) {
  const session = useSession();
  const report = session.diagnostics(s => s.gpuInfo?.spatialPerformance);
  const stages = session.diagnostics(s => s.gpuInfo?.spatialPerformanceStages);
  const view = state.performance ?? DEFAULT_PERFORMANCE_VIEW;
  const change = (patch: Partial<PerformanceView>) => onChange({ ...state, performance: { ...view, ...patch } });
  return <>
    <RailChoice<PerformanceView["sampling"]> label="Sampling" ariaLabel="Performance sampling" value={view.sampling} onChange={sampling => change({ sampling })}
      options={[{ value: "adaptive", label: "Estimate", hint: "Live estimate: learn from diagnostic observations; use cheap rules only after held-out validation passes." }, { value: "diagnostic", label: "Measure", hint: "Measure + learn: measure stage timings on every sampled frame. Adds diagnostic overhead." }]} />
    <RailSelect<string> label="Stage" ariaLabel="Performance stage" value={view.stage} onChange={stage => change({ stage })}
      options={[{ value: "all", label: "All simulation stages" }, ...((report?.stages ?? stages)?.map(s => ({ value: s.id, label: s.label })) ?? [])]} />
    <RailChoice<PerformanceView["grain"]> label="Grain" ariaLabel="Performance aggregation" value={view.grain} onChange={grain => change({ grain })}
      options={[{ value: "cell", label: "Cells" }, { value: "tile", label: "4h" }, { value: "region", label: "16h" }]} />
    <RailChoice<PerformanceView["paint"]> label="Colour" ariaLabel="Performance colour" value={view.paint} onChange={paint => change({ paint })}
      options={[{ value: "cost", label: "Frame cost" }, { value: "tiers", label: "Tiers" }]} />
  </>;
}
const ms = (value: number) => `${value.toFixed(2)} ms`;
export function PerformanceLegend({ state }: { state: VisualLayerState }) {
  const session = useSession();
  const report = session.diagnostics(s => s.gpuInfo?.spatialPerformance);
  const error = session.diagnostics(s => s.gpuInfo?.spatialPerformanceError);
  const view = state.performance ?? DEFAULT_PERFORMANCE_VIEW;
  const selected = report?.stages.find(s => s.id === view.stage);
  const attributed = selected?.attributed_ms ?? report?.attributed_ms ?? 0;
  const entities = report ? view.grain === "cell" ? report.cells : view.grain === "tile" ? report.tiles : report.regions : 0;
  const meanUs = report && entities ? 1000 * report.wall_ms / entities : 0;
  return <div className="performance-key" aria-label="Spatial performance">
    {view.paint === "cost" && <>
      <div style={{ height: 8, borderRadius: 3, background: `linear-gradient(90deg,${PERFORMANCE_RAMP.join(",")})` }} />
      <div style={{ display: "flex", justifyContent: "space-between" }}><span>0</span><span>¼×</span><span>1×</span><span>4×</span><span>16×+</span></div>
      <span>1× = {meanUs ? `${meanUs.toPrecision(3)} µs` : "mean frame share"} per {view.grain === "cell" ? "cell" : view.grain === "tile" ? "4h tile" : "16h region"} · fixed across stages</span>
    </>}
    <RailKey label="Region work tiers" entries={[{ color: "#4094f0", label: "Cold" }, { color: "#fab842", label: "Warm" }, { color: "#f54761", label: "Hot" }]
      .map((t, i) => ({ ...t, label: report ? `${t.label} ${report.tiers[i]}` : t.label }))} />
    {error ? <span role="status">{error}</span> : !report ? <span role="status">Run the simulation to sample performance.</span> : <>
      <strong>Frame {report.frame} · {ms(report.wall_ms)} wall</strong>
      <span>{selected?.label ?? "All stages"}: {ms(attributed)} mapped · {(100 * attributed / report.wall_ms).toFixed(1)}%</span>
      {selected && <span>Stage total: {ms(selected.total_ms)} · shared: {ms(selected.shared_ms)}</span>}
      <span>Frame shared / unmodelled: {ms(report.shared_ms)}</span>
      {view.paint === "cost" && <span role="status">{(selected ? selected.validated : report.stages.filter(s => s.total_ms > 0).every(s => s.validated)) ? "Estimated spatial allocation · frame model validated" : "Provisional spatial estimates · hatched"}</span>}
    </>}
    <details><summary>{report?.evidence === "compiled-model" ? "Compiled estimate" : "Measured stage totals"} · {report?.samples ?? 0} samples · attribution details</summary>
      <p>{report?.status ?? "Hardware stage timing is required."}{selected?.validationError != null ? ` · holdout error ${(100 * selected.validationError).toFixed(0)}%` : ""}</p>
      <p>Colour shows estimated contribution divided by the average frame share at this granularity. Tier rims are independent. Hatching marks provisional attribution. 3D uses transparent fills and cost-coloured wireframes through the whole volume; overlapping colours blend. Use a slice for precise colour comparison.</p>
      <p>The map is a frozen sample, refreshed asynchronously while running. Wall time includes encoding, GPU execution and queue contention. Overlapping CPU and GPU times are not added. Unmodelled time is kept separate.</p>
      <p>Cell cost is an equal share of its 4h tile; sub-tile differences are unresolved. Held-out validation checks frame totals, not independent cell timings.</p>
    </details>
  </div>;
}
