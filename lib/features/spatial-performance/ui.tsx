"use client";
import { Choice, Select } from "../../../components/ui";
import { useSession } from "../../core/session/session-context";
import { DEFAULT_PERFORMANCE_VIEW, PERFORMANCE_RAMP, type PerformanceView } from "./view";
import type { VisualLayerState } from "../../core/visual-layers";
export function PerformanceControls({ state, onChange, plane }: { state: VisualLayerState; onChange: (s: VisualLayerState) => void; plane: boolean }) {
  const session = useSession();
  const report = session.diagnostics(s => s.gpuInfo?.spatialPerformance);
  const stages = session.diagnostics(s => s.gpuInfo?.spatialPerformanceStages);
  const view = state.performance ?? DEFAULT_PERFORMANCE_VIEW;
  const change = (patch: Partial<PerformanceView>) => onChange({ ...state, performance: { ...view, ...patch } });
  return <div style={{ display: "grid", gap: 6, padding: "6px 8px" }}>
    <Choice<PerformanceView["sampling"]> ariaLabel="Performance sampling" value={view.sampling} onChange={sampling => change({ sampling })}
      options={[{ value: "adaptive", label: "Live estimate", hint: "Learn from diagnostic observations; use cheap rules only after held-out validation passes." }, { value: "diagnostic", label: "Measure + learn", hint: "Measure stage timings on every sampled frame. Adds diagnostic overhead." }]} />
    <Select<string> ariaLabel="Performance stage" value={view.stage} onChange={stage => change({ stage })}
      options={[{ value: "all", label: "All simulation stages" }, ...((report?.stages ?? stages)?.map(s => ({ value: s.id, label: s.label })) ?? [])]} />
    <Choice<PerformanceView["grain"]> ariaLabel="Performance aggregation" value={view.grain} onChange={grain => change({ grain })}
      options={[{ value: "cell", label: "Cells" }, { value: "tile", label: "4h tiles" }, { value: "region", label: "16h regions" }]} />
    <Choice<PerformanceView["paint"]> ariaLabel="Performance colour" value={view.paint} onChange={paint => change({ paint })}
      options={[{ value: "cost", label: "Frame cost" }, { value: "tiers", label: "Work tiers" }]} />
    {plane && <Choice<PerformanceView["dimension"]> ariaLabel="Performance view" value={view.dimension} onChange={dimension => change({ dimension })}
      options={[{ value: "slice", label: "Slice" }, { value: "3d", label: "3D" }]} />}
  </div>;
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
  return <div aria-label="Spatial performance" style={{ display: "grid", gap: 4, width: 300, maxWidth: "100%", fontSize: 11, lineHeight: 1.4, padding: "6px 8px" }}>
    {view.paint === "cost" && <>
      <div style={{ height: 8, borderRadius: 3, background: `linear-gradient(90deg,${PERFORMANCE_RAMP.join(",")})` }} />
      <div style={{ display: "flex", justifyContent: "space-between" }}><span>0</span><span>¼×</span><span>1×</span><span>4×</span><span>16×+</span></div>
      <span>1× = {meanUs ? `${meanUs.toPrecision(3)} µs` : "mean frame share"} per {view.grain === "cell" ? "cell" : view.grain === "tile" ? "4h tile" : "16h region"} · fixed across stages</span>
    </>}
    <div className="toolstrip-legend" aria-label="Region work tiers">
      {[{ color: "#4094f0", label: "Cold" }, { color: "#fab842", label: "Warm" }, { color: "#f54761", label: "Hot" }].map((t, i) => <span key={t.label}><i style={{ background: t.color }} />{t.label}{report ? ` ${report.tiers[i]}` : ""}</span>)}
    </div>
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
