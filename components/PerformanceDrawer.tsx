"use client";

import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";

export function PerformanceDrawer() {
  const snapshot = useDiagnosticsStore((state) => state.performanceSnapshot);
  const history = useDiagnosticsStore((state) => state.performanceHistory);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const setPerformanceOpen = useUIStore((state) => state.setPerformanceOpen);
  const timestampsAvailable = gpuStatus.state === "ready" && Boolean(gpuInfo?.gpuTimings);
  const gpuStages = [
    { key: "layer", label: "Adaptive layer", value: snapshot.gpuLayerConstruction_ms, className: "stage-overhead" },
    { key: "advection", label: "Advection + density", value: snapshot.gpuAdvection_ms, className: "stage-advection" },
    { key: "pressure", label: "Pressure Jacobi", value: snapshot.gpuPressure_ms, className: "stage-pressure" },
    { key: "projection", label: "Projection", value: snapshot.gpuProjection_ms, className: "stage-projection" },
    { key: "rigid", label: "Rigid coupling", value: snapshot.gpuRigid_ms, className: "stage-rigid" },
    { key: "diagnostics", label: "Reductions", value: snapshot.gpuDiagnostics_ms, className: "stage-diagnostics" },
    { key: "overhead", label: "Copies + queue gaps", value: snapshot.gpuOverhead_ms, className: "stage-overhead" },
    { key: "render", label: "Raymarch render", value: snapshot.gpuRender_ms, className: "stage-render" }
  ];
  const cpuOther = Math.max(0, snapshot.cpuFrame_ms - snapshot.cpuPhysicsSubmit_ms - snapshot.cpuDataUpload_ms - snapshot.cpuRenderEncode_ms);
  const cpuStages = [
    { label: "Rigid + CPU oracles", value: snapshot.cpuSimulation_ms },
    { label: "GPU physics encode", value: snapshot.cpuPhysicsSubmit_ms },
    { label: "Buffer uploads", value: snapshot.cpuDataUpload_ms },
    { label: "Render encode + submit", value: snapshot.cpuRenderEncode_ms },
    { label: "Frame orchestration", value: cpuOther }
  ];
  const gpuTotal = gpuStages.reduce((sum, stage) => sum + stage.value, 0), cpuTotal = snapshot.cpuSimulation_ms + snapshot.cpuFrame_ms, budget = 16.67;
  const bottleneck = [...gpuStages].sort((a, b) => b.value - a.value)[0];
  const historyValues = history.map((sample) => ({ gpu: sample.gpuLayerConstruction_ms + sample.gpuAdvection_ms + sample.gpuPressure_ms + sample.gpuProjection_ms + sample.gpuRigid_ms + sample.gpuDiagnostics_ms + sample.gpuOverhead_ms + sample.gpuRender_ms, cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]));
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");
  const gpuTime = (value: number) => !timestampsAvailable ? "—" : value > 0 ? `${value.toFixed(3)} ms` : "< timer resolution";
  const cpuTime = (value: number) => value > 0 ? `${value.toFixed(3)} ms` : "< 0.1 ms";
  return <section id="performance-drawer" className="performance-drawer" aria-label="Performance profiler" data-testid="performance-drawer">
    <header className="performance-header"><div><p className="eyebrow">FRAME PROFILER · LIVE</p><h2>GPU and CPU pipeline contribution</h2></div><div className="performance-summary"><span><small>GPU work</small><strong>{gpuTotal.toFixed(2)} ms</strong></span><span><small>CPU work</small><strong>{cpuTotal.toFixed(2)} ms</strong></span><span><small>Largest GPU stage</small><strong>{bottleneck?.label ?? "—"}</strong></span><span><small>60 Hz budget</small><strong>{budget.toFixed(2)} ms</strong></span></div><button className="icon-button" onClick={() => setPerformanceOpen(false)} aria-label="Close performance profiler">×</button></header>
    <div className="performance-body">
      <section className="performance-lane"><div className="performance-lane-heading"><strong>GPU queue</strong><span>{timestampsAvailable ? "hardware timestamps" : "timestamps unavailable"}</span></div><div className="performance-stack" aria-label={`GPU work ${gpuTotal.toFixed(2)} milliseconds of a 16.67 millisecond frame budget`}>{gpuStages.map((stage) => <i key={stage.key} className={stage.className} style={{ width: `${Math.min(stage.value / budget * 100, 100)}%` }} />)}<b style={{ left: `${Math.min(gpuTotal / budget * 100, 100)}%` }} /></div><div className="performance-rows">{gpuStages.map((stage) => <div className="performance-row" key={stage.key}><span><i className={stage.className} />{stage.label}</span><div><i className={stage.className} style={{ width: `${gpuTotal > 0 ? stage.value / gpuTotal * 100 : 0}%` }} /></div><strong>{gpuTime(stage.value)}</strong><small>{gpuTotal > 0 ? (stage.value / gpuTotal * 100).toFixed(1) : "0.0"}%</small></div>)}</div></section>
      <section className="performance-lane"><div className="performance-lane-heading"><strong>CPU main thread</strong><span>wall-clock instrumentation</span></div><div className="performance-rows cpu-rows">{cpuStages.map((stage) => <div className="performance-row" key={stage.label}><span>{stage.label}</span><div><i style={{ width: `${cpuTotal > 0 ? stage.value / cpuTotal * 100 : 0}%` }} /></div><strong>{cpuTime(stage.value)}</strong><small>{cpuTotal > 0 ? (stage.value / cpuTotal * 100).toFixed(1) : "0.0"}%</small></div>)}</div><div className="performance-history"><div><strong>Recent frames</strong><span><i className="history-gpu" />GPU <i className="history-cpu" />CPU</span></div><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent GPU and CPU timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} /><polyline className="history-gpu" points={points("gpu")} /><polyline className="history-cpu" points={points("cpu")} /></svg><small>0</small><small>{historyMax.toFixed(1)} ms</small></div></section>
    </div>
  </section>;
}
