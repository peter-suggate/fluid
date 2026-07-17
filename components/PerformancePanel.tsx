"use client";

import { useMemo, useState } from "react";
import { getMethod } from "@/lib/methods";
import { measuredGPUTime_ms, useDiagnosticsStore, type PerformanceSnapshot } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

type GPUStage = {
  key: string;
  label: string;
  shortLabel: string;
  value: number;
  className: string;
  timer: "physics" | "render";
  group: "compute" | "graphics" | "transfer";
  description: string;
  reads: string[];
  writes: string[];
  dependsOn: string[];
  sync?: string;
};

const formatMs = (value: number, available = true) => !available ? "—" : value > 0 ? `${value.toFixed(value < 1 ? 3 : 2)} ms` : "< timer resolution";

function averageSnapshots(samples: PerformanceSnapshot[], fallback: PerformanceSnapshot) {
  if (samples.length < 2) return samples[0] ?? fallback;
  const latest = samples[samples.length - 1];
  const averaged: PerformanceSnapshot = { ...latest };
  const writable = averaged as unknown as Record<string, number>;
  for (const [key, value] of Object.entries(latest)) {
    if (typeof value !== "number") continue;
    writable[key] = samples.reduce((sum, sample) => sum + ((sample as unknown as Record<string, number>)[key] ?? 0), 0) / samples.length;
  }
  averaged.gpuPhysicsTimingAvailable = samples.every((sample) => sample.gpuPhysicsTimingAvailable);
  averaged.gpuRenderTimingAvailable = samples.every((sample) => sample.gpuRenderTimingAvailable);
  averaged.gpuRenderTimestampSupported = samples.every((sample) => sample.gpuRenderTimestampSupported);
  averaged.adaptiveRebuildBlockedFrames = latest.adaptiveRebuildBlockedFrames;
  averaged.adaptiveRebuildPending = latest.adaptiveRebuildPending;
  return averaged;
}

/** A frame-oriented CPU/GPU trace assembled from the profiler's live timestamp samples. */
export function PerformancePanel() {
  const liveSnapshot = useDiagnosticsStore((state) => state.performanceSnapshot);
  const history = useDiagnosticsStore((state) => state.performanceHistory);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const activeMethodId = useMethodStore((state) => state.methodId);
  const activeWaterMode = useUIStore((state) => state.waterRenderMode);
  const targetFps = 60;
  const fixedDt_s = useSceneStore((state) => state.scene.numerics.fixedDt_s);
  const maxDt_s = useSceneStore((state) => state.scene.numerics.maxDt_s);
  const observedSimRate = useRuntimeStore((state) => state.simRate);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const [selectedStageKey, setSelectedStageKey] = useState("pressure");
  const [averageWindow, setAverageWindow] = useState(30);

  const method = getMethod(activeMethodId);
  const matchingHistory = useMemo(
    () => history.filter((sample) => sample.methodId === activeMethodId && sample.waterRenderMode === activeWaterMode),
    [history, activeMethodId, activeWaterMode]
  );
  const safeIndex = historyIndex === null ? null : Math.min(historyIndex, Math.max(0, matchingHistory.length - 1));
  const windowEnd = safeIndex ?? matchingHistory.length - 1;
  const windowSamples = matchingHistory.slice(Math.max(0, windowEnd - averageWindow + 1), windowEnd + 1);
  const snapshot = averageSnapshots(windowSamples, liveSnapshot);
  const averagedFrameCount = windowSamples.length;
  const contextMatches = snapshot.methodId === activeMethodId && snapshot.waterRenderMode === activeWaterMode;
  const physicsTimed = contextMatches && snapshot.gpuPhysicsTimingAvailable;
  const renderTimed = contextMatches && snapshot.gpuRenderTimingAvailable;
  const rasterized = activeWaterMode === "rasterized";
  const adaptive = activeMethodId === "quadtree-tall-cell";
  const budget = 1000 / targetFps;
  const pressureLabel = gpuInfo?.pressureSolver
    ? `${adaptive ? "Pressure + projection" : "Pressure solve"} · ${gpuInfo.pressureSolver}`
    : adaptive ? "Pressure + projection" : "Pressure solve";

  const physicsStages: GPUStage[] = method.backend === "webgpu" ? [
    ...(!adaptive && contextMatches && snapshot.gpuLayerConstruction_ms > 0 ? [{
      key: "layer", label: "Quadtree construction", shortLabel: "TREE", value: snapshot.gpuLayerConstruction_ms, className: "stage-overhead", timer: "physics" as const, group: "compute" as const,
      description: "Builds the spatial hierarchy used to constrain work to the active fluid region.", reads: ["signed distance φ", "grid bounds"], writes: ["quadtree layers", "active cells"], dependsOn: ["uploads"]
    }] : []),
    {
      key: "advection", label: "Advection + density", shortLabel: "ADVECT", value: contextMatches ? snapshot.gpuAdvection_ms : 0, className: "stage-advection", timer: "physics", group: "compute",
      description: "Transports the signed-distance and velocity fields through the current flow.", reads: ["velocity u", "signed distance φ", "density"], writes: ["advected φ", "advected u"], dependsOn: ["uploads"]
    },
    {
      key: "pressure", label: pressureLabel, shortLabel: "PRESSURE", value: contextMatches ? snapshot.gpuPressure_ms : 0, className: "stage-pressure", timer: "physics", group: "compute",
      description: "Solves the pressure system that makes the liquid velocity divergence-free.", reads: ["advected u", "solid SDF", "pressure system"], writes: ["pressure p", ...(adaptive ? ["projected u"] : [])], dependsOn: ["advection"], sync: adaptive ? "Projection is fused into this measured stage." : activeMethodId === "tall-cell" ? "Fixed V-cycle count from the method settings; residual is measured after the solve, not used for early exit." : undefined
    },
    ...(!adaptive ? [{
      key: "projection", label: "Velocity projection", shortLabel: "PROJECT", value: contextMatches ? snapshot.gpuProjection_ms : 0, className: "stage-projection", timer: "physics" as const, group: "compute" as const,
      description: "Applies the pressure gradient to the velocity field.", reads: ["pressure p", "advected u"], writes: ["projected u"], dependsOn: ["pressure"]
    }] : []),
    {
      key: "rigid", label: "Rigid coupling", shortLabel: "COUPLE", value: contextMatches ? snapshot.gpuRigid_ms : 0, className: "stage-rigid", timer: "physics", group: "compute",
      description: "Exchanges impulses between the liquid and rigid bodies.", reads: ["projected u", "body transforms"], writes: ["fluid impulses", "body forces"], dependsOn: [adaptive ? "pressure" : "projection"]
    },
    {
      key: "diagnostics", label: "Diagnostics reductions", shortLabel: "REDUCE", value: contextMatches ? snapshot.gpuDiagnostics_ms : 0, className: "stage-diagnostics", timer: "physics", group: "transfer",
      description: "Reduces stability and conservation signals for the diagnostics layer.", reads: ["φ", "u", "pressure p"], writes: ["diagnostic summary"], dependsOn: ["rigid"], sync: "Small summaries may be copied back asynchronously."
    },
    {
      key: "overhead", label: "Physics copies + queue gaps", shortLabel: "GAPS", value: contextMatches ? snapshot.gpuOverhead_ms : 0, className: "stage-overhead", timer: "physics", group: "transfer",
      description: "Measured time between physics timestamp regions: copies, transitions, and unclassified queue work.", reads: ["GPU queue"], writes: ["staging buffers"], dependsOn: ["diagnostics"], sync: "This is measured overhead, not shader occupancy."
    }
  ] : [];

  const renderStages: GPUStage[] = rasterized ? [
    { key: "extract", label: "Surface extraction", shortLabel: "SURFACE", value: contextMatches ? snapshot.gpuSurfaceExtraction_ms : 0, className: "stage-extract", timer: "render", group: "compute", description: "Extracts visible liquid surface geometry from the signed-distance field.", reads: ["signed distance φ", "active cells"], writes: ["surface vertices", "indirect draw args"], dependsOn: ["rigid"] },
    { key: "dry-scene", label: "Dry scene", shortLabel: "SCENE", value: contextMatches ? snapshot.gpuDryScene_ms : 0, className: "stage-scene", timer: "render", group: "graphics", description: "Rasterizes the environment and rigid bodies behind the liquid.", reads: ["body transforms", "environment"], writes: ["scene color", "scene depth"], dependsOn: ["uploads"] },
    { key: "interfaces", label: "Front + back interfaces", shortLabel: "INTERFACE", value: contextMatches ? snapshot.gpuInterfaces_ms : 0, className: "stage-interface", timer: "render", group: "graphics", description: "Captures front and back liquid interfaces for thickness and refraction.", reads: ["surface vertices", "camera"], writes: ["front depth", "back depth", "normals"], dependsOn: ["extract"] },
    { key: "composite", label: "Optical composite", shortLabel: "COMPOSITE", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", description: "Combines refraction, absorption, reflection, and the dry scene.", reads: ["scene color", "front/back depth", "normals"], writes: ["water color"], dependsOn: ["dry-scene", "interfaces"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", description: "Resolves the internal render target into the presentation surface.", reads: ["water color"], writes: ["swapchain"], dependsOn: ["composite"], sync: "Presentation boundary" }
  ] : [
    { key: "ray", label: "Ray march", shortLabel: "RAY MARCH", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", description: "Ray-marches the signed-distance field and shades the liquid in one presentation pass.", reads: ["signed distance φ", "scene", "camera"], writes: ["water color"], dependsOn: ["rigid"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", description: "Resolves the internal render target into the presentation surface.", reads: ["water color"], writes: ["swapchain"], dependsOn: ["ray"], sync: "Presentation boundary" }
  ];
  const gpuStages = [...physicsStages, ...renderStages];
  const stageTimed = (stage: GPUStage) => stage.timer === "physics" ? physicsTimed : renderTimed;
  const measuredStages = gpuStages.filter(stageTimed);
  const gpuTotal = measuredStages.reduce((sum, stage) => sum + stage.value, 0);
  const cpuOther = Math.max(0, snapshot.cpuFrame_ms - snapshot.cpuPhysicsSubmit_ms - snapshot.cpuDataUpload_ms - snapshot.cpuRenderEncode_ms);
  const cpuStages = [
    { key: "simulation", label: method.backend === "cpu" ? "Fluid + rigid solve" : "Rigid bodies + CPU oracle", shortLabel: "SIM", value: snapshot.cpuSimulation_ms, note: method.backend === "cpu" ? "CPU solver" : "Includes the coarse validation solve" },
    { key: "encode", label: "Physics encode", shortLabel: "ENCODE", value: snapshot.cpuPhysicsSubmit_ms, note: "Submit GPU work" },
    { key: "upload", label: "Buffer uploads", shortLabel: "UPLOAD", value: snapshot.cpuDataUpload_ms, note: "CPU → GPU" },
    { key: "render", label: rasterized ? "Render passes encode" : "Ray pass encode", shortLabel: "RENDER", value: snapshot.cpuRenderEncode_ms, note: "Submit presentation" },
    { key: "frame", label: "Frame orchestration", shortLabel: "FRAME", value: cpuOther, note: "Input + scheduling" }
  ];
  const cpuTotal = cpuStages.reduce((sum, stage) => sum + stage.value, 0);
  const physicsPerStep_ms = physicsStages.reduce((sum, stage) => sum + (stageTimed(stage) ? stage.value : 0), 0);
  const renderPerFrame_ms = renderStages.reduce((sum, stage) => sum + (stageTimed(stage) ? stage.value : 0), 0);
  const measuredGPUAdvance_s = gpuInfo?.gpuQueueSimulation_s && gpuInfo.gpuQueueSimulation_s > 0 ? Math.min(maxDt_s, gpuInfo.gpuQueueSimulation_s) : maxDt_s;
  const requiredStepsPerFrame = measuredGPUAdvance_s > 0 ? 1 / (targetFps * measuredGPUAdvance_s) : 0;
  const realtimeDemand_ms = physicsPerStep_ms * requiredStepsPerFrame + renderPerFrame_ms;
  const demandPercent = realtimeDemand_ms / budget * 100;
  const estimatedSimRate = physicsPerStep_ms > 0 ? Math.max(0, (1000 - renderPerFrame_ms * targetFps) * measuredGPUAdvance_s / physicsPerStep_ms) : 0;
  const cpuWindow = windowSamples.map((sample) => sample.cpuSimulation_ms + sample.cpuFrame_ms).sort((a, b) => a - b);
  const cpuP95_ms = cpuWindow.length ? cpuWindow[Math.min(cpuWindow.length - 1, Math.floor(cpuWindow.length * .95))] : cpuTotal;
  const gpuConstrained = demandPercent > 100;
  const cpuSpikeConstrained = cpuP95_ms > budget;
  const unexplainedSlowdown = !gpuConstrained && !cpuSpikeConstrained && observedSimRate !== null && observedSimRate < .95;
  const timelineScale = Math.max(budget, gpuTotal, cpuTotal, 0.01);
  const utilization = Math.min(demandPercent, 100);
  const headroom = budget - realtimeDemand_ms;
  const bottleneck = [...measuredStages].sort((a, b) => b.value - a.value)[0];
  const selectedStage = gpuStages.find((stage) => stage.key === selectedStageKey) ?? gpuStages[0];
  const frameOffset = safeIndex === null ? 0 : Math.max(0, matchingHistory.length - 1 - safeIndex);
  const frameLabel = safeIndex === null ? "LIVE" : frameOffset === 0 ? "LATEST" : `F−${frameOffset}`;
  const sampleLabel = averageWindow === 1 ? "single frame" : `${averagedFrameCount}-frame average`;
  const timerDescription = gpuStatus.state !== "ready" ? "GPU unavailable" : physicsTimed && renderTimed ? "Hardware timestamps · physics + presentation" : renderTimed ? "Presentation timestamps · physics pending" : physicsTimed ? "Physics timestamps · presentation pending" : "Awaiting timestamp sample";
  const gridTicks = Array.from({ length: 5 }, (_, index) => index * timelineScale / 4);
  const historyValues = matchingHistory.map((sample) => ({ gpu: measuredGPUTime_ms(sample), cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]), 0.01);
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");
  let gpuCursor = 0;
  let cpuCursor = 0;

  return <aside id="performance-panel" className="right-panel panel-scroll performance-panel" aria-label="Performance profiler" data-testid="performance-panel" data-method={activeMethodId} data-water-renderer={activeWaterMode}>
    <header className="performance-header">
      <div className="performance-title">
        <div><p className="eyebrow">FRAME GRAPH · {frameLabel} · {sampleLabel}</p><h1>Performance trace</h1></div>
        <span className="trace-source"><i />{timerDescription}</span>
      </div>
      <div className="frame-browser" aria-label="Frame selection">
        <button onClick={() => setHistoryIndex((current) => current === null ? Math.max(0, matchingHistory.length - 2) : Math.max(0, current - 1))} disabled={matchingHistory.length < 2 || safeIndex === 0} aria-label="Previous frame">‹</button>
        <div><strong>{frameLabel}</strong><small>{safeIndex === null ? sampleLabel : `${sampleLabel} · ${frameOffset} behind`}</small></div>
        <button onClick={() => setHistoryIndex((current) => current === null ? null : current >= matchingHistory.length - 2 ? null : current + 1)} disabled={safeIndex === null} aria-label="Next frame">›</button>
        <button className={safeIndex === null ? "active" : ""} onClick={() => setHistoryIndex(null)}>LIVE</button>
      </div>
      <label className="averaging-control"><small>AVERAGING</small><select value={averageWindow} onChange={(event) => setAverageWindow(Number(event.target.value))} aria-label="Timing averaging window"><option value="1">1 frame</option><option value="10">10 frames</option><option value="30">30 frames</option><option value="60">60 frames</option><option value="100">100 frames</option></select></label>
      <button className="panel-close" onClick={() => setRightPanel(null)} aria-label="Close performance profiler">×</button>
    </header>

    <section className={`performance-overview${gpuConstrained ? " over-budget" : ""}`} aria-label="Frame performance summary">
      <div className="budget-gauge" style={{ "--utilization": `${utilization}%` } as React.CSSProperties}>
        <div><strong>{measuredStages.length ? `${demandPercent.toFixed(0)}%` : "—"}</strong><small>RT demand</small></div>
      </div>
      <div className="overview-stat"><small>GPU REALTIME DEMAND</small><strong>{measuredStages.length ? `${realtimeDemand_ms.toFixed(2)} ms / frame` : "sampling…"}</strong><span>{requiredStepsPerFrame.toFixed(2)} advances × {physicsPerStep_ms.toFixed(2)} ms + {renderPerFrame_ms.toFixed(2)} ms render</span></div>
      <div className="overview-stat"><small>CPU WORK</small><strong>{cpuTotal.toFixed(2)} ms avg · {cpuP95_ms.toFixed(2)} p95</strong><span>SIM includes rigid bodies + CPU validation oracle</span></div>
      <div className="overview-stat"><small>CRITICAL STAGE</small><strong>{bottleneck?.label ?? "Awaiting sample"}</strong><span>{bottleneck ? `${(bottleneck.value / Math.max(gpuTotal, .001) * 100).toFixed(0)}% of GPU work` : "No timestamp data"}</span></div>
      <div className="overview-stat"><small>SIMULATION THROUGHPUT</small><strong>{observedSimRate === null ? "measuring…" : `×${observedSimRate.toFixed(2)} observed`}</strong><span>GPU ceiling ≈ ×{estimatedSimRate.toFixed(2)} · target ×1.00</span></div>
    </section>

    <section className={`throughput-explainer${gpuConstrained || cpuSpikeConstrained || unexplainedSlowdown ? " constrained" : ""}`}>
      <div className="throughput-verdict"><small>WHY NOT ×1?</small><strong>{gpuConstrained ? "GPU advance throughput exceeds the frame budget" : cpuSpikeConstrained ? "CPU SIM spikes exceed the frame deadline" : unexplainedSlowdown ? "Visible pass timings do not yet explain the slowdown" : "Measured CPU and GPU work fit the realtime budget"}</strong><span>{gpuConstrained ? `${Math.abs(headroom).toFixed(2)} ms more GPU time is required per presentation frame` : cpuSpikeConstrained ? `CPU p95 is ${cpuP95_ms.toFixed(2)} ms against a ${budget.toFixed(2)} ms deadline` : unexplainedSlowdown ? "The missing time is likely between timestamp regions: queue fences, completion latency, or scheduler gaps" : `${headroom.toFixed(2)} ms estimated GPU headroom`}</span></div>
      <div className="throughput-equation"><span><small>CONTROLLER CLOCK</small><strong>Δt {(fixedDt_s * 1000).toFixed(2)} ms</strong><b>{(1 / Math.max(fixedDt_s, 1e-9)).toFixed(0)} rigid/oracle ticks per s</b></span><i>→</i><span><small>GPU ADVANCE</small><strong>Δt {(measuredGPUAdvance_s * 1000).toFixed(2)} ms</strong><b>capped by max Δt {(maxDt_s * 1000).toFixed(2)} ms</b></span><i>×</i><span><small>PHYSICS COST</small><strong>{physicsPerStep_ms.toFixed(2)} ms / advance</strong><b>pressure runs once per advance</b></span><i>+</i><span><small>GPU DEMAND</small><strong>{(realtimeDemand_ms * targetFps / 1000).toFixed(2)} GPU s/s</strong><b>{demandPercent.toFixed(0)}% including presentation</b></span></div>
      <p><strong>SIM spikes:</strong> this CPU region is the fixed-step controller, rigid-body integration, state publication, and a coarse CPU validation solve every {method.backend === "webgpu" ? (adaptive ? 32 : 4) : 1} solver steps. It is not GPU execution.</p>
    </section>

    <div className="performance-workspace">
      <section className="trace-card timeline-card">
        <header className="trace-card-header"><div><p className="eyebrow">ONE SOLVER STEP + ONE PRESENTATION</p><h2>CPU submission → GPU execution → present</h2></div><div className="timeline-legend"><span><i className="legend-compute" />Compute</span><span><i className="legend-graphics" />Graphics</span><span><i className="legend-transfer" />Transfer / gap</span><span><i className="legend-idle" />Available</span></div></header>
        <div className="timeline-ruler"><span>0</span>{gridTicks.slice(1).map((tick) => <span key={tick}>{tick.toFixed(1)} ms</span>)}</div>
        <div className="timeline-lanes">
          <div className="timeline-lane"><div className="lane-label"><strong>CPU</strong><small>Main thread</small></div><div className="lane-track cpu-track">{cpuStages.map((stage) => { const left = cpuCursor / timelineScale * 100; cpuCursor += stage.value; return <button key={stage.key} className="cpu-block" style={{ left: `${left}%`, width: `${Math.max(.7, stage.value / timelineScale * 100)}%` }} title={`${stage.label} · ${formatMs(stage.value)}`}><span>{stage.shortLabel}</span></button>; })}</div><output>{cpuTotal.toFixed(2)} ms</output></div>
          <div className="timeline-connector"><span>queue.submit</span><i /></div>
          <div className="timeline-lane gpu-lane"><div className="lane-label"><strong>GPU</strong><small>One physics solve</small></div><div className="lane-track gpu-track">{gpuStages.map((stage) => { const left = gpuCursor / timelineScale * 100; const width = stageTimed(stage) ? stage.value / timelineScale * 100 : 0; gpuCursor += stageTimed(stage) ? stage.value : 0; return <button key={stage.key} className={`gpu-block ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} style={{ left: `${left}%`, width: `${Math.max(.7, width)}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value, stageTimed(stage))}`}><span>{stage.shortLabel}</span><b>{formatMs(stage.value, stageTimed(stage))}</b></button>; })}<span className="budget-marker" style={{ left: `${Math.min(100, budget / timelineScale * 100)}%` }} /></div><output>{physicsPerStep_ms.toFixed(2)} ms / solve</output></div>
          <div className="timeline-lane async-lane"><div className="lane-label"><strong>ASYNC</strong><small>{adaptive ? "Topology worker" : "Browser + readback"}</small></div><div className="lane-track">{adaptive ? <span className={`async-block${snapshot.adaptiveRebuildPending ? " active" : ""}`} style={{ width: `${Math.min(100, Math.max(3, snapshot.adaptiveRebuildWall_ms / timelineScale * 100))}%` }}><i />{snapshot.adaptiveRebuildPending ? "REBUILD IN FLIGHT" : "LAST ADAPTIVE REBUILD"}</span> : <span className="async-note"><i />Readbacks resolve without blocking the queue unless consumed by the CPU</span>}</div><output>{adaptive && snapshot.adaptiveRebuildWall_ms ? `${snapshot.adaptiveRebuildWall_ms.toFixed(1)} ms` : "non-blocking"}</output></div>
        </div>
        <footer className="timeline-footnote"><span><i className="sync-mark" />CPU↔GPU synchronization boundary</span><span>The physics regions above repeat {requiredStepsPerFrame.toFixed(2)}× per frame to sustain realtime; shader-core occupancy requires a captured trace.</span></footer>
      </section>

      <section className="trace-card frame-graph-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME GRAPH</p><h2>Pass dependencies & resource flow</h2></div><small>SELECT A PASS TO INSPECT</small></header>
        <div className="frame-graph-flow">
          <div className="graph-section"><span>PHYSICS</span><div>{physicsStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage))}</small></button>{index < physicsStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
          <b className="graph-divider">QUEUE ORDER →</b>
          <div className="graph-section"><span>PRESENTATION</span><div>{renderStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage))}</small></button>{index < renderStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
        </div>
        {selectedStage && <article className="stage-inspector">
          <div className="stage-inspector-title"><span className={selectedStage.className} /><div><small>{selectedStage.group.toUpperCase()} PASS</small><h3>{selectedStage.label}</h3></div><strong>{formatMs(selectedStage.value, stageTimed(selectedStage))}</strong></div>
          <p>{selectedStage.description}</p>
          <div className="resource-columns"><div><small>READS</small>{selectedStage.reads.map((item) => <span key={item}><i>R</i>{item}</span>)}</div><div><small>WRITES</small>{selectedStage.writes.map((item) => <span key={item}><i>W</i>{item}</span>)}</div><div><small>WAITS FOR</small>{selectedStage.dependsOn.map((item) => <span key={item}><i>↳</i>{gpuStages.find((stage) => stage.key === item)?.shortLabel ?? item.toUpperCase()}</span>)}{selectedStage.sync && <span className="sync-detail"><i>⇄</i>{selectedStage.sync}</span>}</div></div>
        </article>}
      </section>

      <section className="trace-card history-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME HISTORY</p><h2>Last {matchingHistory.length} frames</h2></div><div className="timeline-legend"><span><i className="history-gpu" />GPU</span><span><i className="history-cpu" />CPU</span><span><i className="history-budget" />Budget</span></div></header>
        <div className="history-chart"><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} /><polyline className="history-gpu" points={points("gpu")} /><polyline className="history-cpu" points={points("cpu")} />{safeIndex !== null && <line className="history-cursor" x1={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y1="2" x2={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y2="49" />}</svg><span>0</span><span>{historyMax.toFixed(1)} ms</span></div>
        <input type="range" min="0" max={Math.max(0, matchingHistory.length - 1)} value={safeIndex ?? Math.max(0, matchingHistory.length - 1)} onChange={(event) => setHistoryIndex(Number(event.target.value))} disabled={!matchingHistory.length} aria-label="Inspect a frame from history" />
      </section>

      <aside className="trace-card capture-card">
        <div><p className="eyebrow">CAPTURE LAYER</p><h2>Timestamp trace</h2><span className="capture-status"><i />LIVE</span></div>
        <div className="architecture-model" title="Assumed architecture context; timing values remain directly measured"><span><small>ARCH MODEL</small><strong>APPLE M1 MAX</strong></span><span><small>CPU</small><strong>10 cores</strong></span><span><small>GPU</small><strong>32 cores*</strong></span><span><small>MEMORY</small><strong>Unified</strong></span></div>
        <p>This view is driven by WebGPU timestamp queries and CPU wall-clock regions. A direct browser GPU capture can plug into the same frame graph to add hardware occupancy, barriers, and command-level gaps.</p>
        <dl><div><dt>GPU physics</dt><dd>{measuredStages.length ? `${physicsPerStep_ms.toFixed(2)} ms / solve` : "sampling"}</dd></div><div><dt>CPU sync waits</dt><dd>{adaptive ? `${snapshot.adaptiveRebuildBlockedFrames} blocked frames` : "none observed"}</dd></div><div><dt>Trace fidelity</dt><dd>Pass level</dd></div></dl>
        <small className="architecture-note">* Assumed full M1 Max GPU configuration; architecture context only, not a hardware-counter reading.</small>
      </aside>
    </div>
  </aside>;
}
