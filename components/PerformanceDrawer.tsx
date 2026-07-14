"use client";

import { getMethod } from "@/lib/methods";
import { measuredGPUTime_ms, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useUIStore } from "@/lib/stores/ui-store";

type GPUStage = {
  key: string;
  label: string;
  value: number;
  className: string;
  timer: "physics" | "render";
};

export function PerformanceDrawer() {
  const snapshot = useDiagnosticsStore((state) => state.performanceSnapshot);
  const history = useDiagnosticsStore((state) => state.performanceHistory);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const activeMethodId = useMethodStore((state) => state.methodId);
  const activeWaterMode = useUIStore((state) => state.waterRenderMode);
  const setPerformanceOpen = useUIStore((state) => state.setPerformanceOpen);
  const method = getMethod(activeMethodId);
  const contextMatches = snapshot.methodId === activeMethodId && snapshot.waterRenderMode === activeWaterMode;
  const physicsTimed = contextMatches && snapshot.gpuPhysicsTimingAvailable;
  const renderTimed = contextMatches && snapshot.gpuRenderTimingAvailable;
  const rasterized = activeWaterMode === "rasterized";
  const pressureLabel = gpuInfo?.pressureSolver ? `Pressure · ${gpuInfo.pressureSolver}` : "Pressure solve";

  const physicsStages: GPUStage[] = method.backend === "webgpu" ? [
    ...(activeMethodId === "quadtree-tall-cell" || (contextMatches && snapshot.gpuLayerConstruction_ms > 0)
      ? [{ key: "layer", label: "Quadtree construction", value: contextMatches ? snapshot.gpuLayerConstruction_ms : 0, className: "stage-overhead", timer: "physics" as const }]
      : []),
    { key: "advection", label: "Advection + density", value: contextMatches ? snapshot.gpuAdvection_ms : 0, className: "stage-advection", timer: "physics" },
    { key: "pressure", label: pressureLabel, value: contextMatches ? snapshot.gpuPressure_ms : 0, className: "stage-pressure", timer: "physics" },
    { key: "projection", label: "Projection", value: contextMatches ? snapshot.gpuProjection_ms : 0, className: "stage-projection", timer: "physics" },
    { key: "rigid", label: "Rigid coupling", value: contextMatches ? snapshot.gpuRigid_ms : 0, className: "stage-rigid", timer: "physics" },
    { key: "diagnostics", label: "Diagnostics reductions", value: contextMatches ? snapshot.gpuDiagnostics_ms : 0, className: "stage-diagnostics", timer: "physics" },
    { key: "overhead", label: "Physics copies + gaps", value: contextMatches ? snapshot.gpuOverhead_ms : 0, className: "stage-overhead", timer: "physics" }
  ] : [];
  const renderStages: GPUStage[] = rasterized ? [
    { key: "extract", label: "Surface extraction", value: contextMatches ? snapshot.gpuSurfaceExtraction_ms : 0, className: "stage-render", timer: "render" },
    { key: "dry-scene", label: "Dry scene", value: contextMatches ? snapshot.gpuDryScene_ms : 0, className: "stage-render", timer: "render" },
    { key: "interfaces", label: "Front + back interfaces", value: contextMatches ? snapshot.gpuInterfaces_ms : 0, className: "stage-render", timer: "render" },
    { key: "composite", label: "Optical composite", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-render", timer: "render" },
    { key: "upscale", label: "Final upscale", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-render", timer: "render" }
  ] : [
    { key: "ray", label: "Ray march", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-render", timer: "render" },
    { key: "upscale", label: "Final upscale", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-render", timer: "render" }
  ];
  const gpuStages: GPUStage[] = [...physicsStages, ...renderStages];
  const stageTimed = (stage: GPUStage) => stage.timer === "physics" ? physicsTimed : renderTimed;

  const cpuOther = Math.max(0, snapshot.cpuFrame_ms - snapshot.cpuPhysicsSubmit_ms - snapshot.cpuDataUpload_ms - snapshot.cpuRenderEncode_ms);
  const cpuStages = [
    { label: method.backend === "cpu" ? "CPU fluid + rigid solve" : "Rigid + CPU orchestration", value: snapshot.cpuSimulation_ms },
    { label: method.backend === "webgpu" ? "GPU physics encode" : "GPU physics encode (inactive)", value: snapshot.cpuPhysicsSubmit_ms },
    { label: "Buffer uploads", value: snapshot.cpuDataUpload_ms },
    { label: rasterized ? "Raster passes encode + submit" : "Ray pass encode + submit", value: snapshot.cpuRenderEncode_ms },
    { label: "Frame orchestration", value: cpuOther }
  ];
  const gpuTotal = gpuStages.reduce((sum, stage) => sum + (stageTimed(stage) ? stage.value : 0), 0);
  const cpuTotal = snapshot.cpuSimulation_ms + snapshot.cpuFrame_ms;
  const budget = 16.67;
  const measuredStages = gpuStages.filter(stageTimed);
  const bottleneck = [...measuredStages].sort((a, b) => b.value - a.value)[0];
  const matchingHistory = history.filter((sample) => sample.methodId === activeMethodId && sample.waterRenderMode === activeWaterMode);
  const historyValues = matchingHistory.map((sample) => ({ gpu: measuredGPUTime_ms(sample), cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]));
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");
  const gpuTime = (stage: GPUStage) => !stageTimed(stage) ? "—" : stage.value > 0 ? `${stage.value.toFixed(3)} ms` : "< timer resolution";
  const cpuTime = (value: number) => value > 0 ? `${value.toFixed(3)} ms` : "< 0.1 ms";
  const timestampSupported = contextMatches && (snapshot.gpuRenderTimestampSupported || snapshot.gpuPhysicsTimingAvailable);
  const timerDescription = gpuStatus.state !== "ready"
    ? "GPU unavailable"
    : !timestampSupported ? "hardware timestamps unavailable"
    : physicsTimed && renderTimed ? "physics + presentation timestamps"
      : renderTimed ? "presentation timestamps · physics pending"
        : physicsTimed ? "physics timestamps · presentation sampling"
          : "waiting for hardware timestamp sample";

  return <section id="performance-drawer" className="performance-drawer" aria-label="Performance profiler" data-testid="performance-drawer" data-method={activeMethodId} data-water-renderer={activeWaterMode}>
    <header className="performance-header"><div><p className="eyebrow">FRAME PROFILER · LIVE · {method.label}</p><h2>{rasterized ? "Raster-optics" : "Ray-marched"} GPU and CPU contribution</h2></div><div className="performance-summary"><span><small>GPU work</small><strong>{measuredStages.length ? `${gpuTotal.toFixed(2)} ms` : "sampling…"}</strong></span><span><small>CPU work</small><strong>{cpuTotal.toFixed(2)} ms</strong></span><span><small>Largest GPU stage</small><strong>{bottleneck?.label ?? "Awaiting sample"}</strong></span><span><small>60 Hz budget</small><strong>{budget.toFixed(2)} ms</strong></span></div><button className="icon-button" onClick={() => setPerformanceOpen(false)} aria-label="Close performance profiler">×</button></header>
    <div className="performance-body">
      <section className="performance-lane"><div className="performance-lane-heading"><strong>GPU queue · {method.label}</strong><span>{timerDescription}</span></div><div className="performance-stack" aria-label={`Measured GPU work ${gpuTotal.toFixed(2)} milliseconds of a 16.67 millisecond frame budget`}>{gpuStages.map((stage) => <i key={stage.key} className={stage.className} style={{ width: `${stageTimed(stage) ? Math.min(stage.value / budget * 100, 100) : 0}%` }} />)}<b style={{ left: `${Math.min(gpuTotal / budget * 100, 100)}%` }} /></div><div className="performance-rows">{gpuStages.map((stage) => <div className="performance-row" key={stage.key}><span><i className={stage.className} />{stage.label}</span><div><i className={stage.className} style={{ width: `${stageTimed(stage) && gpuTotal > 0 ? stage.value / gpuTotal * 100 : 0}%` }} /></div><strong>{gpuTime(stage)}</strong><small>{stageTimed(stage) && gpuTotal > 0 ? `${(stage.value / gpuTotal * 100).toFixed(1)}%` : "—"}</small></div>)}</div></section>
      <section className="performance-lane"><div className="performance-lane-heading"><strong>CPU main thread · {method.label}</strong><span>wall-clock instrumentation</span></div><div className="performance-rows cpu-rows">{cpuStages.map((stage) => <div className="performance-row" key={stage.label}><span>{stage.label}</span><div><i style={{ width: `${cpuTotal > 0 ? stage.value / cpuTotal * 100 : 0}%` }} /></div><strong>{cpuTime(stage.value)}</strong><small>{cpuTotal > 0 ? (stage.value / cpuTotal * 100).toFixed(1) : "0.0"}%</small></div>)}</div><div className="performance-history"><div><strong>Recent frames · current method only</strong><span><i className="history-gpu" />GPU <i className="history-cpu" />CPU</span></div><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent ${method.label} ${activeWaterMode} timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} /><polyline className="history-gpu" points={points("gpu")} /><polyline className="history-cpu" points={points("cpu")} /></svg><small>0</small><small>{historyMax.toFixed(1)} ms</small></div></section>
    </div>
  </section>;
}
