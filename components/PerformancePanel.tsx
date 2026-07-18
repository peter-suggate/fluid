"use client";

import { useMemo, useState } from "react";
import { getMethod } from "@/lib/methods";
import { measuredGPUTime_ms, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { gpuBatchDepth, gpuInFlightStepLimit } from "@/lib/simulation/gpu-clock";
import { performanceSchedule } from "@/lib/performance-scheduling";
import { averagePerformanceSnapshots, rollingPerformanceSnapshots } from "@/lib/performance-averaging";
import { adaptiveTopologyPerformanceStages, physicsPerformanceStages, type PerformanceStage } from "@/lib/performance-stage-model";

const formatMs = (value: number, available = true, active = true) => !available ? "—" : !active ? "idle" : value > 0 ? `${value.toFixed(value < 1 ? 3 : 2)} ms` : "< timer resolution";

/** A frame-oriented CPU/GPU trace assembled from the profiler's live timestamp samples. */
export function PerformancePanel() {
  const liveSnapshot = useDiagnosticsStore((state) => state.performanceSnapshot);
  const history = useDiagnosticsStore((state) => state.performanceHistory);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const hasRigidBodies = useDiagnosticsStore((state) => state.bodies.length > 0);
  const activeMethodId = useMethodStore((state) => state.methodId);
  const sprayEnabled = useMethodStore((state) => state.methodId === "octree" && state.overrides.octree?.secondaryParticles !== "off" && state.overrides.octree?.secondaryParticles !== false);
  const activeWaterMode = useUIStore((state) => state.waterRenderMode);
  const targetFps = useUIStore((state) => state.targetFps);
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
  const snapshot = averagePerformanceSnapshots(windowSamples, liveSnapshot);
  const averagedFrameCount = windowSamples.length;
  const contextMatches = snapshot.methodId === activeMethodId && snapshot.waterRenderMode === activeWaterMode;
  const physicsTimed = contextMatches && snapshot.gpuPhysicsTimingAvailable;
  const renderTimed = contextMatches && snapshot.gpuRenderTimingAvailable;
  const rasterized = activeWaterMode === "rasterized";
  const adaptive = activeMethodId === "quadtree-tall-cell";
  const budget = 1000 / targetFps;
  const topologyPath = snapshot.adaptiveInlineTopology ? "inline" : "async";
  const physicsStages = method.backend === "webgpu" ? physicsPerformanceStages({ methodId: activeMethodId, snapshot, contextMatches, pressureSolver: gpuInfo?.pressureSolver, topologyPath }) : [];
  const adaptiveTopologyStages = adaptive && topologyPath === "async" ? adaptiveTopologyPerformanceStages({ snapshot, contextMatches }) : [];
  const physicsOutputStage = activeMethodId === "quadtree-tall-cell" ? "surface-update" : activeMethodId === "tall-cell" || activeMethodId === "uniform" ? "projection" : "uploads";

  const renderStages: PerformanceStage[] = rasterized ? [
    { key: "extract", label: "Surface extraction", shortLabel: "SURFACE", value: contextMatches ? snapshot.gpuSurfaceExtraction_ms : 0, className: "stage-extract", timer: "render", group: "compute", active: true, description: "Extracts visible liquid surface geometry from the signed-distance field.", reads: ["signed distance φ", "active cells"], writes: ["surface vertices", "indirect draw args"], dependsOn: [physicsOutputStage] },
    { key: "dry-scene", label: "Dry scene", shortLabel: "SCENE", value: contextMatches ? snapshot.gpuDryScene_ms : 0, className: "stage-scene", timer: "render", group: "graphics", active: true, description: "Rasterizes the environment and rigid bodies behind the liquid.", reads: ["body transforms", "environment"], writes: ["scene color", "scene depth"], dependsOn: ["uploads"] },
    { key: "interfaces", label: "Front + back interfaces", shortLabel: "INTERFACE", value: contextMatches ? snapshot.gpuInterfaces_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: true, description: "Captures front and back liquid interfaces for thickness and refraction.", reads: ["surface vertices", "camera"], writes: ["front depth", "back depth", "normals"], dependsOn: ["extract"] },
    { key: "spray-render", label: "Spray optical interfaces", shortLabel: "SPRAY DRAW", value: contextMatches ? snapshot.gpuSprayRender_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: sprayEnabled, description: "Rasterizes escaped droplets into the same front/back interface buffers used by the resolved water surface.", reads: ["spray particle ring", "camera"], writes: ["front/back depth", "droplet normals"], dependsOn: ["interfaces"] },
    { key: "composite", label: "Optical composite", shortLabel: "COMPOSITE", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", active: true, description: "Combines refraction, absorption, reflection, and the dry scene.", reads: ["scene color", "front/back depth", "normals"], writes: ["water color"], dependsOn: ["dry-scene", "spray-render"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", active: true, description: "Resolves the internal render target into the presentation surface.", reads: ["water color"], writes: ["swapchain"], dependsOn: ["composite"], sync: "Presentation boundary" }
  ] : [
    { key: "ray", label: "Ray march", shortLabel: "RAY MARCH", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", active: true, description: "Ray-marches the signed-distance field and shades the liquid in one presentation pass.", reads: ["signed distance φ", "scene", "camera"], writes: ["water color"], dependsOn: [physicsOutputStage] },
    { key: "spray-render", label: "Spray fallback draw", shortLabel: "SPRAY DRAW", value: contextMatches ? snapshot.gpuSprayRender_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: sprayEnabled, description: "Draws escaped droplets above the ray-marched water fallback.", reads: ["spray particle ring", "camera"], writes: ["water color"], dependsOn: ["ray"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", active: true, description: "Resolves the internal render target into the presentation surface.", reads: ["water color"], writes: ["swapchain"], dependsOn: ["spray-render"], sync: "Presentation boundary" }
  ];
  const gpuStages = [...physicsStages, ...adaptiveTopologyStages, ...renderStages];
  const stageTimed = (stage: PerformanceStage) => stage.timer === "physics" ? physicsTimed : stage.timer === "render" ? renderTimed : contextMatches && snapshot.adaptiveRebuildCompletedCount > 0;
  const measuredStages = [...physicsStages, ...renderStages].filter(stageTimed);
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
  const submissionBatchDepth = method.backend === "webgpu" ? gpuBatchDepth(activeMethodId, fixedDt_s, hasRigidBodies, targetFps) : 1;
  const preparedStepLimit = method.backend === "webgpu" ? gpuInFlightStepLimit(activeMethodId, fixedDt_s, hasRigidBodies, targetFps) : 1;
  const pressureSolvesPerAdvance = activeMethodId === "tall-cell" && gpuInfo?.pressureSolver?.includes("defect correction") ? 2 : 1;
  const schedule = performanceSchedule({
    targetFps,
    gpuAdvance_s: measuredGPUAdvance_s,
    submissionBatchDepth,
    physicsPerAdvance_ms: physicsPerStep_ms,
    renderPerFrame_ms,
    pressureSolvesPerAdvance
  });
  const batchSimulation_ms = schedule.batchSimulation_ms;
  const batchGPU_ms = schedule.batchGPU_ms;
  const submissionEnvelope_ms = batchGPU_ms + renderPerFrame_ms;
  const completionRate = gpuInfo?.gpuCompletionWall_ms && gpuInfo.gpuCompletionSimulation_s
    ? gpuInfo.gpuCompletionSimulation_s * 1000 / gpuInfo.gpuCompletionWall_ms
    : null;
  const realtimeDemand_ms = schedule.gpuDemandPerFrame_ms;
  const demandPercent = schedule.demandPercent;
  const cpuWindow = windowSamples.map((sample) => sample.cpuSimulation_ms + sample.cpuFrame_ms).sort((a, b) => a - b);
  const cpuP95_ms = cpuWindow.length ? cpuWindow[Math.min(cpuWindow.length - 1, Math.floor(cpuWindow.length * .95))] : cpuTotal;
  const gpuConstrained = demandPercent > 100;
  const cpuSpikeConstrained = cpuP95_ms > budget;
  const unexplainedSlowdown = !gpuConstrained && !cpuSpikeConstrained && observedSimRate !== null && observedSimRate < .95;
  const timelineScale = Math.max(budget, submissionEnvelope_ms, cpuTotal, 0.01);
  const headroom = schedule.headroom_ms;
  const selectedStage = gpuStages.find((stage) => stage.key === selectedStageKey) ?? gpuStages[0];
  const frameOffset = safeIndex === null ? 0 : Math.max(0, matchingHistory.length - 1 - safeIndex);
  const frameLabel = safeIndex === null ? "LIVE" : frameOffset === 0 ? "LATEST" : `F−${frameOffset}`;
  const sampleLabel = averageWindow === 1 ? "single frame" : `${averagedFrameCount}-frame rolling average`;
  const timerDescription = gpuStatus.state !== "ready" ? "GPU unavailable" : physicsTimed && renderTimed ? "Hardware timestamps · physics + presentation" : renderTimed ? "Presentation timestamps · physics pending" : physicsTimed ? "Physics timestamps · presentation pending" : "Awaiting timestamp sample";
  const gridTicks = Array.from({ length: 5 }, (_, index) => index * timelineScale / 4);
  const averagedHistory = rollingPerformanceSnapshots(matchingHistory, averageWindow);
  const historyValues = averagedHistory.map((sample) => ({ gpu: measuredGPUTime_ms(sample), cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]), 0.01);
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");
  const physicsDemandPercent = Math.min(100, schedule.physicsPerFrame_ms / budget * 100);
  const renderDemandPercent = Math.min(100 - physicsDemandPercent, schedule.renderPerFrame_ms / budget * 100);
  const observedPressureSolvesPerFrame = observedSimRate === null ? null : schedule.pressureSolvesPerFrame * observedSimRate;
  const advanceDisplay_ms = physicsPerStep_ms > 0 ? physicsPerStep_ms : timelineScale / submissionBatchDepth;
  const cpuTimeline = cpuStages.map((stage, index) => ({ stage, left: cpuStages.slice(0, index).reduce((sum, previous) => sum + previous.value, 0) / timelineScale * 100 }));
  const physicsOffsets = physicsStages.map((_, index) => physicsStages.slice(0, index).reduce((sum, previous) => sum + (stageTimed(previous) ? previous.value : 0), 0));
  const physicsTimeline = Array.from({ length: submissionBatchDepth }, (_, step) => physicsStages.map((stage, index) => ({ stage, step, left: (step * physicsPerStep_ms + physicsOffsets[index]) / timelineScale * 100, width: stageTimed(stage) ? stage.value / timelineScale * 100 : 0 }))).flat();
  const renderTimeline = renderStages.map((stage, index) => ({ stage, left: (batchGPU_ms + renderStages.slice(0, index).reduce((sum, previous) => sum + (stageTimed(previous) ? previous.value : 0), 0)) / timelineScale * 100, width: stageTimed(stage) ? stage.value / timelineScale * 100 : 0 }));

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

    <section className={`performance-overview schedule-overview${gpuConstrained ? " over-budget" : ""}`} aria-label="Frame performance summary">
      <div className="realtime-budget-summary">
        <div className="realtime-budget-title"><span><small>REALTIME GPU LOAD</small><strong>{measuredStages.length ? `${demandPercent.toFixed(0)}%` : "—"}</strong></span><b>{measuredStages.length ? `${realtimeDemand_ms.toFixed(2)} / ${budget.toFixed(2)} ms` : "sampling…"}</b></div>
        <div className="frame-budget-track" aria-label={`${schedule.physicsPerFrame_ms.toFixed(2)} milliseconds physics, ${schedule.renderPerFrame_ms.toFixed(2)} milliseconds presentation, ${Math.max(0, headroom).toFixed(2)} milliseconds headroom`}>
          <span className="budget-physics" style={{ width: `${physicsDemandPercent}%` }} />
          <span className="budget-render" style={{ width: `${renderDemandPercent}%` }} />
          <i />
        </div>
        <div className="frame-budget-labels"><span>0</span><span><b />physics {schedule.physicsPerFrame_ms.toFixed(2)}</span><span><b />render {schedule.renderPerFrame_ms.toFixed(2)}</span><strong>{budget.toFixed(2)} ms deadline</strong></div>
      </div>
      <div className="overview-stat pressure-cadence"><small>PRESSURE CADENCE</small><strong>{schedule.pressureSolvesPerFrame.toFixed(2)} solves / frame</strong><span>{schedule.pressureSolvesPerSecond.toFixed(1)} / s · {schedule.pressureSolvesPerAdvance} per GPU advance</span></div>
      <div className="overview-stat"><small>SUBMISSION PAYLOAD</small><strong>{schedule.pressureSolvesPerBatch} solves / batch</strong><span>{batchSimulation_ms.toFixed(0)} ms simulation · {schedule.realtimeFramesPerBatch.toFixed(2)} RT frames</span></div>
      <div className="overview-stat"><small>OBSERVED COMPLETION</small><strong>{observedPressureSolvesPerFrame === null ? "measuring…" : `${observedPressureSolvesPerFrame.toFixed(2)} solves / frame`}</strong><span>{observedSimRate === null ? "simulation rate sampling" : `×${observedSimRate.toFixed(2)} realtime`} · {completionRate === null ? "queue sampling" : `queue ×${completionRate.toFixed(2)}`}</span></div>
    </section>

    <section className={`schedule-translation${gpuConstrained || cpuSpikeConstrained || unexplainedSlowdown ? " constrained" : ""}`} aria-label="Scheduling model">
      <div className="schedule-verdict"><small>CAPACITY VERDICT</small><strong>{gpuConstrained ? "GPU demand exceeds the presentation budget" : cpuSpikeConstrained ? "CPU p95 exceeds the presentation deadline" : unexplainedSlowdown ? "Timestamped work does not explain the observed slowdown" : `${headroom.toFixed(2)} ms GPU headroom per presentation frame`}</strong><span>{gpuConstrained ? `${Math.abs(headroom).toFixed(2)} ms over budget` : `CPU ${cpuTotal.toFixed(2)} ms avg · ${cpuP95_ms.toFixed(2)} p95`}</span></div>
      <div className="schedule-node"><small>PRESENTATION FRAME</small><strong>{budget.toFixed(2)} ms wall time</strong><span>needs {schedule.advancesPerFrame.toFixed(2)} GPU advances</span><b>{schedule.pressureSolvesPerFrame.toFixed(2)} pressure solves</b></div>
      <i className="schedule-arrow">→</i>
      <div className="schedule-node active"><small>ONE GPU ADVANCE</small><strong>{schedule.gpuAdvance_ms.toFixed(2)} ms simulation</strong><span>costs {physicsPerStep_ms.toFixed(2)} ms on GPU</span><b>{schedule.pressureSolvesPerAdvance} pressure {schedule.pressureSolvesPerAdvance === 1 ? "solve" : "solves"}</b></div>
      <i className="schedule-arrow">×{submissionBatchDepth}</i>
      <div className="schedule-node"><small>ONE SUBMISSION BATCH</small><strong>{batchSimulation_ms.toFixed(2)} ms simulation</strong><span>costs {batchGPU_ms.toFixed(2)} ms on GPU</span><b>{schedule.pressureSolvesPerBatch} solves · spans {schedule.realtimeFramesPerBatch.toFixed(2)} RT frames</b></div>
    </section>

    <div className="performance-workspace">
      <section className="trace-card timeline-card">
        <header className="trace-card-header"><div><p className="eyebrow">GPU ELAPSED TIME · ONE SUBMISSION BURST</p><h2>{submissionBatchDepth} advances are one batch—not one frame</h2></div><div className="timeline-legend"><span><i className="legend-compute" />Compute</span><span><i className="legend-graphics" />Presentation</span><span><i className="legend-transfer" />Transfer / gap</span></div></header>
        <div className="timeline-ruler"><span>0</span>{gridTicks.slice(1).map((tick) => <span key={tick}>{tick.toFixed(1)} ms</span>)}</div>
        <div className="timeline-lanes">
          <div className="timeline-lane"><div className="lane-label"><strong>CPU</strong><small>Main thread</small></div><div className="lane-track cpu-track">{cpuTimeline.map(({ stage, left }) => <button key={stage.key} className="cpu-block" style={{ left: `${left}%`, width: `${Math.max(.7, stage.value / timelineScale * 100)}%` }} title={`${stage.label} · ${formatMs(stage.value)}`}><span>{stage.shortLabel}</span></button>)}</div><output>{cpuTotal.toFixed(2)} ms</output></div>
          <div className="timeline-connector"><span>queue.submit</span><i /></div>
          <div className="timeline-lane advance-lane"><div className="lane-label"><strong>ADVANCES</strong><small>{schedule.pressureSolvesPerAdvance} pressure / advance</small></div><div className="lane-track advance-track">{Array.from({ length: submissionBatchDepth }, (_, step) => <span key={step} style={{ left: `${step * advanceDisplay_ms / timelineScale * 100}%`, width: `${advanceDisplay_ms / timelineScale * 100}%` }}><b>#{step + 1}</b><small>{schedule.gpuAdvance_ms.toFixed(0)} ms sim</small></span>)}{renderPerFrame_ms > 0 && <span className="presentation-window" style={{ left: `${batchGPU_ms / timelineScale * 100}%`, width: `${renderPerFrame_ms / timelineScale * 100}%` }}><b>PRESENT</b></span>}</div><output>{schedule.pressureSolvesPerBatch} solves</output></div>
          <div className="timeline-lane gpu-lane"><div className="lane-label"><strong>GPU PASSES</strong><small>elapsed execution</small></div><div className="lane-track gpu-track">{physicsTimeline.map(({ stage, step, left, width }) => <button key={`${step}:${stage.key}`} className={`gpu-block ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} style={{ left: `${left}%`, width: `${Math.max(.7, width)}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`Advance ${step + 1} · ${stage.label} · ${formatMs(stage.value, stageTimed(stage), stage.active)}`}><span>{stage.shortLabel}</span><b>{formatMs(stage.value, stageTimed(stage), stage.active)}</b></button>)}{renderTimeline.map(({ stage, left, width }) => <button key={`render:${stage.key}`} className={`gpu-block ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} style={{ left: `${left}%`, width: `${Math.max(.7, width)}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value, stageTimed(stage), stage.active)}`}><span>{stage.shortLabel}</span><b>{formatMs(stage.value, stageTimed(stage), stage.active)}</b></button>)}</div><output>{submissionEnvelope_ms.toFixed(2)} ms</output></div>
          <div className="timeline-lane async-lane"><div className="lane-label"><strong>ASYNC</strong><small>{adaptive ? "Topology worker" : "Browser + readback"}</small></div><div className="lane-track">{adaptive ? <span className={`async-block${snapshot.adaptiveRebuildPending ? " active" : ""}`} style={{ width: `${Math.min(100, Math.max(3, snapshot.adaptiveRebuildWall_ms / timelineScale * 100))}%` }}><i />{snapshot.adaptiveRebuildPending ? "REBUILD IN FLIGHT" : "LAST ADAPTIVE REBUILD"}</span> : <span className="async-note"><i />Readbacks resolve without blocking the queue unless consumed by the CPU</span>}</div><output>{adaptive && snapshot.adaptiveRebuildWall_ms ? `${snapshot.adaptiveRebuildWall_ms.toFixed(1)} ms` : "non-blocking"}</output></div>
        </div>
        <footer className="timeline-footnote"><span><i className="sync-mark" />This {submissionEnvelope_ms.toFixed(2)} ms trace carries {schedule.realtimeFramesPerBatch.toFixed(2)} realtime frames of simulation; it does not have to fit inside one frame deadline.</span><span>{preparedStepLimit} advances max in flight · {gpuInfo?.gpuPendingBatches ?? 0} batches pending · {(gpuInfo?.gpuQueueStarved_ms ?? 0).toFixed(2)} ms last host gap</span></footer>
      </section>

      <section className="trace-card frame-graph-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME GRAPH</p><h2>Pass dependencies & resource flow</h2></div><small>SELECT A PASS TO INSPECT</small></header>
        <div className="frame-graph-flow">
          <div className="graph-section"><span>PHYSICS</span><div>{physicsStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < physicsStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
          {adaptiveTopologyStages.length > 0 && <><b className="graph-divider">EVENT-DRIVEN · NOT REPEATED PER ADVANCE</b><div className="graph-section async-graph-section"><span>ADAPTIVE TOPOLOGY</span><div>{adaptiveTopologyStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < adaptiveTopologyStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div></>}
          <b className="graph-divider">QUEUE ORDER →</b>
          <div className="graph-section"><span>PRESENTATION</span><div>{renderStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < renderStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
        </div>
        {selectedStage && <article className="stage-inspector">
          <div className="stage-inspector-title"><span className={selectedStage.className} /><div><small>{selectedStage.group.toUpperCase()} PASS</small><h3>{selectedStage.label}</h3></div><strong>{formatMs(selectedStage.value, stageTimed(selectedStage), selectedStage.active)}</strong></div>
          <p>{selectedStage.description}</p>
          <div className="resource-columns"><div><small>READS</small>{selectedStage.reads.map((item) => <span key={item}><i>R</i>{item}</span>)}</div><div><small>WRITES</small>{selectedStage.writes.map((item) => <span key={item}><i>W</i>{item}</span>)}</div><div><small>WAITS FOR</small>{selectedStage.dependsOn.map((item) => <span key={item}><i>↳</i>{gpuStages.find((stage) => stage.key === item)?.shortLabel ?? item.toUpperCase()}</span>)}{selectedStage.sync && <span className="sync-detail"><i>⇄</i>{selectedStage.sync}</span>}</div></div>
        </article>}
      </section>

      <section className="trace-card history-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME HISTORY · {averageWindow === 1 ? "RAW" : `${averageWindow}-FRAME ROLLING AVERAGE`}</p><h2>Last {matchingHistory.length} frames</h2></div><div className="timeline-legend"><span><i className="history-gpu" />GPU</span><span><i className="history-cpu" />CPU</span><span><i className="history-budget" />Budget</span></div></header>
        <div className="history-chart"><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} /><polyline className="history-gpu" points={points("gpu")} /><polyline className="history-cpu" points={points("cpu")} />{safeIndex !== null && <line className="history-cursor" x1={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y1="2" x2={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y2="49" />}</svg><span>0</span><span>{historyMax.toFixed(1)} ms</span></div>
        <input type="range" min="0" max={Math.max(0, matchingHistory.length - 1)} value={safeIndex ?? Math.max(0, matchingHistory.length - 1)} onChange={(event) => setHistoryIndex(Number(event.target.value))} disabled={!matchingHistory.length} aria-label="Inspect a frame from history" />
      </section>

      <aside className="trace-card capture-card">
        <div><p className="eyebrow">CAPTURE LAYER</p><h2>Timestamp trace</h2><span className="capture-status"><i />LIVE</span></div>
        <div className="architecture-model" title="Assumed architecture context; timing values remain directly measured"><span><small>ARCH MODEL</small><strong>APPLE M1 MAX</strong></span><span><small>CPU</small><strong>10 cores</strong></span><span><small>GPU</small><strong>32 cores*</strong></span><span><small>MEMORY</small><strong>Unified</strong></span></div>
        <p>This view is driven by WebGPU timestamp queries and CPU wall-clock regions. A direct browser GPU capture can plug into the same frame graph to add hardware occupancy, barriers, and command-level gaps.</p>
        <dl><div><dt>GPU physics</dt><dd>{measuredStages.length ? `${physicsPerStep_ms.toFixed(2)} ms / solve` : "sampling"}</dd></div><div><dt>CPU sync waits</dt><dd>{adaptive ? `${snapshot.adaptiveRebuildBlockedFrames} blocked frames` : "none observed"}</dd></div><div><dt>Trace fidelity</dt><dd>{adaptive ? "Pass + rebuild phase" : "Pass level"}</dd></div></dl>
        <small className="architecture-note">* Assumed full M1 Max GPU configuration; architecture context only, not a hardware-counter reading.</small>
      </aside>
    </div>
  </aside>;
}
