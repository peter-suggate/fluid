"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { getMethod } from "@/lib/methods";
import { measuredGPUTime_ms, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { measuredGPUUtilization, performanceSchedule } from "@/lib/performance-scheduling";
import type { CSSProperties } from "react";
import { averagePerformanceSnapshots, rollingPerformanceSnapshots } from "@/lib/performance-averaging";
import { adaptiveTopologyPerformanceStages, physicsPerformanceStages, type PerformanceStage } from "@/lib/performance-stage-model";
import { PRESENTATION_FPS } from "@/lib/frame-pacing";
import { captureTargetForStage, gpuStageCapture, type GPUStageCaptureArtifact, type GPUStageCaptureLane } from "@/lib/gpu-stage-capture";

const formatMs = (value: number, available = true, active = true) => !available ? "—" : !active ? "idle" : value > 0 ? `${value.toFixed(value < 1 ? 3 : 2)} ms` : "< timer resolution";

function CapturePreview({ artifact }: { artifact: GPUStageCaptureArtifact }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = ref.current?.getContext("2d");
    if (!context) return;
    const image = context.createImageData(artifact.previewWidth, artifact.previewHeight);
    image.data.set(artifact.previewRgba);
    context.putImageData(image, 0, 0);
  }, [artifact]);
  return <canvas ref={ref} width={artifact.previewWidth} height={artifact.previewHeight} aria-label={`${artifact.label} ${artifact.selectorLabel} preview`} />;
}

/** A frame-oriented CPU/GPU trace assembled from the profiler's live timestamp samples. */
export function PerformancePanel() {
  const liveSnapshot = useDiagnosticsStore((state) => state.performanceSnapshot);
  const history = useDiagnosticsStore((state) => state.performanceHistory);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const activeMethodId = useMethodStore((state) => state.methodId);
  const sprayEnabled = useMethodStore((state) => state.methodId === "octree" && state.overrides.octree?.secondaryParticles !== "off" && state.overrides.octree?.secondaryParticles !== false);
  const maxDt_s = useSceneStore((state) => state.scene.numerics.maxDt_s);
  const runState = useRuntimeStore((state) => state.runState);
  const observedSimRate = useRuntimeStore((state) => state.simRate);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const [selectedStageKey, setSelectedStageKey] = useState("pressure");
  const [averageWindow, setAverageWindow] = useState(30);
  const captureState = useSyncExternalStore(gpuStageCapture.subscribe, gpuStageCapture.getSnapshot, gpuStageCapture.getServerSnapshot);

  const method = getMethod(activeMethodId);
  const matchingHistory = useMemo(
    () => history.filter((sample) => sample.methodId === activeMethodId && sample.renderTimingContext === liveSnapshot.renderTimingContext),
    [history, activeMethodId, liveSnapshot.renderTimingContext]
  );
  const safeIndex = historyIndex === null ? null : Math.min(historyIndex, Math.max(0, matchingHistory.length - 1));
  const windowEnd = safeIndex ?? matchingHistory.length - 1;
  const windowSamples = matchingHistory.slice(Math.max(0, windowEnd - averageWindow + 1), windowEnd + 1);
  const snapshot = averagePerformanceSnapshots(windowSamples, liveSnapshot);
  const averagedFrameCount = windowSamples.length;
  const contextMatches = snapshot.methodId === activeMethodId && snapshot.renderTimingContext === liveSnapshot.renderTimingContext;
  const physicsTimed = contextMatches && snapshot.gpuPhysicsTimingAvailable;
  const renderTimed = contextMatches && snapshot.gpuRenderTimingAvailable;
  const adaptive = activeMethodId === "quadtree-tall-cell";
  const budget = 1000 / PRESENTATION_FPS;
  const topologyPath = snapshot.adaptiveInlineTopology ? "inline" : "async";
  const physicsStages = method.backend === "webgpu" ? physicsPerformanceStages({ methodId: activeMethodId, snapshot, contextMatches, pressureSolver: gpuInfo?.pressureSolver, topologyPath }) : [];
  const adaptiveTopologyStages = adaptive && topologyPath === "async" ? adaptiveTopologyPerformanceStages({ snapshot, contextMatches }) : [];
  const physicsOutputStage = activeMethodId === "octree" ? "sparse-publication" : activeMethodId === "quadtree-tall-cell" ? "surface-update" : activeMethodId === "tall-cell" || activeMethodId === "uniform" ? "projection" : "uploads";

  const renderStages: PerformanceStage[] = [
    { key: "extract", label: "Surface extraction", shortLabel: "SURFACE", value: contextMatches ? snapshot.gpuSurfaceExtraction_ms : 0, className: "stage-extract", timer: "render", group: "compute", active: true, description: "Extracts visible liquid surface geometry from the signed-distance field.", reads: ["signed distance φ", "active cells"], writes: ["surface vertices", "indirect draw args"], dependsOn: [physicsOutputStage] },
    { key: "dry-scene", label: "Dry scene", shortLabel: "SCENE", value: contextMatches ? snapshot.gpuDryScene_ms : 0, className: "stage-scene", timer: "render", group: "graphics", active: true, description: "Rasterizes the environment and rigid bodies behind the liquid.", reads: ["body transforms", "environment"], writes: ["scene color", "scene depth"], dependsOn: ["uploads"] },
    { key: "svo-temporal", label: "SVO temporal resolve", shortLabel: "TEMPORAL", value: contextMatches ? snapshot.gpuSvoTemporal_ms : 0, className: "stage-scene", timer: "render", group: "graphics", active: snapshot.gpuSvoTemporal_ms > 0, description: "Reprojects and filters the SVO dry target before legacy raster water composition.", reads: ["dry G-buffer", "history"], writes: ["filtered dry color", "history"], dependsOn: ["dry-scene"] },
    { key: "interfaces", label: sprayEnabled ? "Water + spray interfaces" : "Front + back interfaces", shortLabel: "INTERFACES", value: contextMatches ? snapshot.gpuInterfaces_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: true, description: sprayEnabled ? "Captures the resolved surface and active spray ellipsoids together in one front pass and one back pass." : "Captures front and back liquid interfaces for thickness and refraction.", reads: sprayEnabled ? ["surface vertices", "spray particle ring", "camera"] : ["surface vertices", "camera"], writes: ["front depth", "back depth", "normals"], dependsOn: ["extract"] },
    { key: "composite", label: "Optical composite", shortLabel: "COMPOSITE", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", active: true, description: "Combines refraction, absorption, reflection, and the dry scene.", reads: ["scene color", "front/back depth", "normals"], writes: ["water color"], dependsOn: [snapshot.gpuSvoTemporal_ms > 0 ? "svo-temporal" : "dry-scene", "interfaces"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", active: true, description: "Resolves the internal render target into the presentation surface.", reads: ["water color"], writes: ["swapchain"], dependsOn: ["composite"], sync: "Presentation boundary" }
  ];
  const gpuStages = [...physicsStages, ...adaptiveTopologyStages, ...renderStages];
  const stageTimed = (stage: PerformanceStage) => stage.timer === "physics" ? physicsTimed : stage.timer === "render" ? renderTimed : contextMatches && snapshot.adaptiveRebuildCompletedCount > 0;
  const measuredStages = [...physicsStages, ...renderStages].filter(stageTimed);
  const cpuOther = Math.max(0, snapshot.cpuFrame_ms - snapshot.cpuPhysicsSubmit_ms - snapshot.cpuDataUpload_ms - snapshot.cpuRenderEncode_ms);
  const cpuStages = [
    { key: "simulation", label: method.backend === "cpu" ? "Fluid + rigid solve" : "Rigid bodies + CPU oracle", shortLabel: "SIM", value: snapshot.cpuSimulation_ms, note: method.backend === "cpu" ? "CPU solver" : "Includes the coarse validation solve" },
    { key: "encode", label: "Physics encode", shortLabel: "ENCODE", value: snapshot.cpuPhysicsSubmit_ms, note: "Submit GPU work" },
    { key: "upload", label: "Buffer uploads", shortLabel: "UPLOAD", value: snapshot.cpuDataUpload_ms, note: "CPU → GPU" },
    { key: "render", label: "Render passes encode", shortLabel: "RENDER", value: snapshot.cpuRenderEncode_ms, note: "Submit presentation" },
    { key: "frame", label: "Frame orchestration", shortLabel: "FRAME", value: cpuOther, note: "Input + scheduling" }
  ];
  const cpuTotal = cpuStages.reduce((sum, stage) => sum + stage.value, 0);
  const physicsPerStep_ms = physicsStages.reduce((sum, stage) => sum + (stageTimed(stage) ? stage.value : 0), 0);
  const renderPerFrame_ms = renderStages.reduce((sum, stage) => sum + (stageTimed(stage) ? stage.value : 0), 0);
  const measuredGPUAdvance_s = gpuInfo?.gpuQueueSimulation_s && gpuInfo.gpuQueueSimulation_s > 0 ? Math.min(maxDt_s, gpuInfo.gpuQueueSimulation_s) : maxDt_s;
  const measuredBatchSimulation_s = gpuInfo?.gpuBatchSimulation_s && gpuInfo.gpuBatchSimulation_s > 0 ? gpuInfo.gpuBatchSimulation_s : measuredGPUAdvance_s;
  const submissionBatchDepth = method.backend === "webgpu" ? Math.min(64, Math.max(1, Math.round(measuredBatchSimulation_s / Math.max(measuredGPUAdvance_s, 1e-9)))) : 1;
  const pressureSolvesPerAdvance = activeMethodId === "tall-cell" && gpuInfo?.pressureSolver?.includes("defect correction") ? 2 : 1;
  const schedule = performanceSchedule({
    targetFps: PRESENTATION_FPS,
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
  const paused = runState === "paused";
  const measuredUtilization = paused ? { physics: 0, presentation: 0, total: 0 } : measuredGPUUtilization({
    physics_ms: gpuInfo?.gpuStep_ms,
    physicsCompletionInterval_ms: gpuInfo?.gpuCompletionWall_ms,
    presentation_ms: liveSnapshot.gpuRenderTimingAvailable ? liveSnapshot.gpuRender_ms : undefined,
    presentationInterval_ms: gpuInfo?.gpuPresentationWall_ms
  });
  const measuredUtilizationPercent = measuredUtilization ? measuredUtilization.total * 100 : null;
  const realtimeDemand_ms = paused ? 0 : schedule.gpuDemandPerFrame_ms;
  const demandPercent = paused ? 0 : schedule.demandPercent;
  const cpuWindow = windowSamples.map((sample) => sample.cpuSimulation_ms + sample.cpuFrame_ms).sort((a, b) => a - b);
  const cpuP95_ms = cpuWindow.length ? cpuWindow[Math.min(cpuWindow.length - 1, Math.floor(cpuWindow.length * .95))] : cpuTotal;
  const gpuConstrained = demandPercent > 100;
  const cpuSpikeConstrained = !paused && cpuP95_ms > budget;
  const unexplainedSlowdown = !paused && !gpuConstrained && !cpuSpikeConstrained && observedSimRate !== null && observedSimRate < .95;
  const timelineScale = Math.max(budget, submissionEnvelope_ms, cpuTotal, 0.01);
  const headroom = paused ? budget : schedule.headroom_ms;
  const selectedStage = gpuStages.find((stage) => stage.key === selectedStageKey) ?? gpuStages[0];
  const selectedCaptureLane: GPUStageCaptureLane | undefined = selectedStage?.timer === "render" ? "presentation" : selectedStage?.timer === "physics" ? "physics" : undefined;
  const selectedCaptureTarget = selectedStage && selectedCaptureLane ? captureTargetForStage(activeMethodId, selectedCaptureLane, selectedStage.key) : undefined;
  const captureBusy = captureState.phase === "armed" || captureState.phase === "encoding" || captureState.phase === "submitted" || captureState.phase === "reading";
  const captureDisabledReason = !selectedCaptureTarget ? "No typed visual resource is registered for this stage" : !selectedStage?.active ? "This stage is idle in the live pipeline" : selectedCaptureLane === "physics" && paused ? "Resume the simulation to capture a physics boundary" : gpuStatus.state !== "ready" ? "GPU is not ready" : undefined;
  const armSelectedCapture = () => {
    if (!selectedCaptureTarget || !selectedStage || captureDisabledReason) return;
    gpuStageCapture.arm({
      ...selectedCaptureTarget,
      baseline: {
        methodId: activeMethodId,
        renderTimingContext: liveSnapshot.renderTimingContext,
        stage_ms: selectedStage.value,
        gpuTotal_ms: measuredGPUTime_ms(snapshot),
        sampleCount: averagedFrameCount,
      },
    });
  };
  const frameOffset = safeIndex === null ? 0 : Math.max(0, matchingHistory.length - 1 - safeIndex);
  const frameLabel = safeIndex === null ? "LIVE" : frameOffset === 0 ? "LATEST" : `F−${frameOffset}`;
  const sampleLabel = averageWindow === 1 ? "single frame" : `${averagedFrameCount}-frame rolling average`;
  const timerDescription = gpuStatus.state !== "ready" ? "GPU unavailable" : paused && renderTimed ? "Paused · last on-change presentation" : physicsTimed && renderTimed ? "Hardware timestamps · physics + presentation" : renderTimed ? "Presentation timestamps · physics pending" : physicsTimed ? "Physics timestamps · presentation pending" : "Awaiting timestamp sample";
  const gridTicks = Array.from({ length: 5 }, (_, index) => index * timelineScale / 4);
  const averagedHistory = rollingPerformanceSnapshots(matchingHistory, averageWindow);
  const historyValues = averagedHistory.map((sample) => ({ gpu: measuredGPUTime_ms(sample), cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]), 0.01);
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");
  const physicsDemandPercent = paused ? 0 : Math.min(100, schedule.physicsPerFrame_ms / budget * 100);
  const renderDemandPercent = paused ? 0 : Math.min(100 - physicsDemandPercent, schedule.renderPerFrame_ms / budget * 100);
  const observedPressureSolvesPerFrame = paused ? 0 : observedSimRate === null ? null : schedule.pressureSolvesPerFrame * observedSimRate;
  const advanceDisplay_ms = physicsPerStep_ms > 0 ? physicsPerStep_ms : timelineScale / submissionBatchDepth;
  const cpuTimeline = cpuStages.map((stage, index) => ({ stage, left: cpuStages.slice(0, index).reduce((sum, previous) => sum + previous.value, 0) / timelineScale * 100 }));
  const physicsOffsets = physicsStages.map((_, index) => physicsStages.slice(0, index).reduce((sum, previous) => sum + (stageTimed(previous) ? previous.value : 0), 0));
  const physicsTimeline = Array.from({ length: submissionBatchDepth }, (_, step) => physicsStages.map((stage, index) => ({ stage, step, left: (step * physicsPerStep_ms + physicsOffsets[index]) / timelineScale * 100, width: stageTimed(stage) ? stage.value / timelineScale * 100 : 0 }))).flat();
  const renderTimeline = renderStages.map((stage, index) => ({ stage, left: (batchGPU_ms + renderStages.slice(0, index).reduce((sum, previous) => sum + (stageTimed(previous) ? previous.value : 0), 0)) / timelineScale * 100, width: stageTimed(stage) ? stage.value / timelineScale * 100 : 0 }));

  return <aside id="performance-panel" className="right-panel panel-scroll performance-panel" aria-label="Performance profiler" data-testid="performance-panel" data-method={activeMethodId}
    data-render-timing-context={liveSnapshot.renderTimingContext}
    data-render-timing-epoch={liveSnapshot.renderTimingEpoch}
    data-render-timing-sample-id={liveSnapshot.renderTimingSampleId}
    data-render-timestamp-supported={liveSnapshot.gpuRenderTimestampSupported}
    data-render-timing-available={liveSnapshot.gpuRenderTimingAvailable}
    data-render-timing-total-ms={liveSnapshot.gpuRenderTimingAvailable ? liveSnapshot.gpuRender_ms : undefined}>
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
      <div className="budget-gauge" style={{ "--utilization": `${measuredUtilizationPercent ?? 0}%` } as CSSProperties} title="Timestamped GPU work divided by queue-confirmed wall intervals">
        <div><strong>{measuredUtilizationPercent === null ? "—" : `${measuredUtilizationPercent.toFixed(0)}%`}</strong><small>GPU busy</small></div>
      </div>
      <div className="realtime-budget-summary">
        <div className="realtime-budget-title"><span><small>{paused ? "IDLE GPU LOAD" : "REALTIME GPU LOAD"}</small><strong>{measuredStages.length ? `${demandPercent.toFixed(0)}%` : "—"}</strong></span><b>{paused ? renderTimed ? `${renderPerFrame_ms.toFixed(2)} ms on last change` : "awaiting timestamp…" : measuredStages.length ? `${realtimeDemand_ms.toFixed(2)} / ${budget.toFixed(2)} ms` : "sampling…"}</b></div>
        <div className="frame-budget-track" aria-label={`${schedule.physicsPerFrame_ms.toFixed(2)} milliseconds physics, ${schedule.renderPerFrame_ms.toFixed(2)} milliseconds presentation, ${Math.max(0, headroom).toFixed(2)} milliseconds headroom`}>
          <span className="budget-physics" style={{ width: `${physicsDemandPercent}%` }} />
          <span className="budget-render" style={{ width: `${renderDemandPercent}%` }} />
          <i />
        </div>
        <div className="frame-budget-labels"><span>0</span><span><b />physics {paused ? "0.00" : schedule.physicsPerFrame_ms.toFixed(2)}</span><span><b />render {paused ? "on change" : schedule.renderPerFrame_ms.toFixed(2)}</span><strong>{budget.toFixed(2)} ms deadline</strong></div>
      </div>
      <div className="overview-stat"><small>MEASURED UTILIZATION</small><strong>{measuredUtilizationPercent === null ? "sampling…" : `${measuredUtilizationPercent.toFixed(1)}% busy`}</strong><span>{measuredUtilization === null ? "awaiting completion cadence" : paused ? "simulation paused" : `${(measuredUtilization.physics * 100).toFixed(0)}% physics · ${(measuredUtilization.presentation * 100).toFixed(0)}% presentation`}</span></div>
      <div className="overview-stat pressure-cadence"><small>PRESSURE CADENCE</small><strong>{paused ? "0.00" : schedule.pressureSolvesPerFrame.toFixed(2)} solves / frame</strong><span>{paused ? "simulation paused" : `${schedule.pressureSolvesPerSecond.toFixed(1)} / s · ${schedule.pressureSolvesPerAdvance} per GPU advance`}</span></div>
      <div className="overview-stat"><small>OBSERVED COMPLETION</small><strong>{observedPressureSolvesPerFrame === null ? "measuring…" : `${observedPressureSolvesPerFrame.toFixed(2)} solves / frame`}</strong><span>{paused ? "idle · presentation redraws on change" : `${observedSimRate === null ? "simulation rate sampling" : `×${observedSimRate.toFixed(2)} realtime`} · ${completionRate === null ? "queue sampling" : `queue ×${completionRate.toFixed(2)}`}`}</span></div>
    </section>

    <section className={`schedule-translation${gpuConstrained || cpuSpikeConstrained || unexplainedSlowdown ? " constrained" : ""}`} aria-label="Scheduling model">
      <div className="schedule-verdict"><small>CAPACITY VERDICT</small><strong>{paused ? "Paused · viewport renders only when its inputs change" : gpuConstrained ? "GPU demand exceeds the presentation budget" : cpuSpikeConstrained ? "CPU p95 exceeds the presentation deadline" : unexplainedSlowdown ? "Timestamped work does not explain the observed slowdown" : `${headroom.toFixed(2)} ms GPU headroom per presentation frame`}</strong><span>{paused ? renderTimed ? `Last presentation ${renderPerFrame_ms.toFixed(2)} ms on GPU` : "Awaiting presentation timestamp" : gpuConstrained ? `${Math.abs(headroom).toFixed(2)} ms over budget` : `CPU ${cpuTotal.toFixed(2)} ms avg · ${cpuP95_ms.toFixed(2)} p95`}</span></div>
      <div className="schedule-node"><small>PRESENTATION FRAME</small><strong>{budget.toFixed(2)} ms wall time</strong><span>needs {paused ? "0.00" : schedule.advancesPerFrame.toFixed(2)} GPU advances</span><b>{paused ? "0.00" : schedule.pressureSolvesPerFrame.toFixed(2)} pressure solves</b></div>
      <i className="schedule-arrow">→</i>
      <div className="schedule-node active"><small>ONE GPU ADVANCE</small><strong>{schedule.gpuAdvance_ms.toFixed(2)} ms simulation</strong><span>costs {physicsPerStep_ms.toFixed(2)} ms on GPU</span><b>{schedule.pressureSolvesPerAdvance} pressure {schedule.pressureSolvesPerAdvance === 1 ? "solve" : "solves"}</b></div>
      <i className="schedule-arrow">×{submissionBatchDepth}</i>
      <div className="schedule-node"><small>ONE QUEUE SLOT</small><strong>{batchSimulation_ms.toFixed(2)} ms simulation</strong><span>costs {batchGPU_ms.toFixed(2)} ms on GPU</span><b>next slot follows completion or presentation</b></div>
    </section>

    <div className="performance-workspace">
      <section className="trace-card timeline-card">
        <header className="trace-card-header"><div><p className="eyebrow">GPU ELAPSED TIME · COMPLETION-GATED QUEUE</p><h2>Dense rolling advances with presentation boundaries</h2></div><div className="timeline-legend"><span><i className="legend-compute" />Compute</span><span><i className="legend-graphics" />Presentation</span><span><i className="legend-transfer" />Transfer / gap</span></div></header>
        <div className="timeline-ruler"><span>0</span>{gridTicks.slice(1).map((tick) => <span key={tick}>{tick.toFixed(1)} ms</span>)}</div>
        <div className="timeline-lanes">
          <div className="timeline-lane"><div className="lane-label"><strong>CPU</strong><small>Main thread</small></div><div className="lane-track cpu-track">{cpuTimeline.map(({ stage, left }) => <button key={stage.key} className="cpu-block" style={{ left: `${left}%`, width: `${Math.max(.7, stage.value / timelineScale * 100)}%` }} title={`${stage.label} · ${formatMs(stage.value)}`}><span>{stage.shortLabel}</span></button>)}</div><output>{cpuTotal.toFixed(2)} ms</output></div>
          <div className="timeline-connector"><span>queue.submit</span><i /></div>
          <div className="timeline-lane advance-lane"><div className="lane-label"><strong>ADVANCES</strong><small>{schedule.pressureSolvesPerAdvance} pressure / advance</small></div><div className="lane-track advance-track">{Array.from({ length: submissionBatchDepth }, (_, step) => <span key={step} style={{ left: `${step * advanceDisplay_ms / timelineScale * 100}%`, width: `${advanceDisplay_ms / timelineScale * 100}%` }}><b>#{step + 1}</b><small>{schedule.gpuAdvance_ms.toFixed(0)} ms sim</small></span>)}{renderPerFrame_ms > 0 && <span className="presentation-window" style={{ left: `${batchGPU_ms / timelineScale * 100}%`, width: `${renderPerFrame_ms / timelineScale * 100}%` }}><b>PRESENT</b></span>}</div><output>{schedule.pressureSolvesPerBatch} solves</output></div>
          <div className="timeline-lane gpu-lane"><div className="lane-label"><strong>GPU PASSES</strong><small>elapsed execution</small></div><div className="lane-track gpu-track">{physicsTimeline.map(({ stage, step, left, width }) => <button key={`${step}:${stage.key}`} className={`gpu-block ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} style={{ left: `${left}%`, width: `${Math.max(.7, width)}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`Advance ${step + 1} · ${stage.label} · ${formatMs(stage.value, stageTimed(stage), stage.active)}`}><span>{stage.shortLabel}</span><b>{formatMs(stage.value, stageTimed(stage), stage.active)}</b></button>)}{renderTimeline.map(({ stage, left, width }) => <button key={`render:${stage.key}`} className={`gpu-block ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} style={{ left: `${left}%`, width: `${Math.max(.7, width)}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value, stageTimed(stage), stage.active)}`}><span>{stage.shortLabel}</span><b>{formatMs(stage.value, stageTimed(stage), stage.active)}</b></button>)}</div><output>{submissionEnvelope_ms.toFixed(2)} ms</output></div>
          <div className="timeline-lane async-lane"><div className="lane-label"><strong>ASYNC</strong><small>{adaptive ? "Topology worker" : "Browser + readback"}</small></div><div className="lane-track">{adaptive ? <span className={`async-block${snapshot.adaptiveRebuildPending ? " active" : ""}`} style={{ width: `${Math.min(100, Math.max(3, snapshot.adaptiveRebuildWall_ms / timelineScale * 100))}%` }}><i />{snapshot.adaptiveRebuildPending ? "REBUILD IN FLIGHT" : "LAST ADAPTIVE REBUILD"}</span> : <span className="async-note"><i />Readbacks resolve without blocking the queue unless consumed by the CPU</span>}</div><output>{adaptive && snapshot.adaptiveRebuildWall_ms ? `${snapshot.adaptiveRebuildWall_ms.toFixed(1)} ms` : "non-blocking"}</output></div>
        </div>
        <footer className="timeline-footnote"><span><i className="sync-mark" />Post-presentation depth rounds up to the next whole advance, preferring simulation throughput even when that step crosses the 16.67 ms target.</span><span>{gpuInfo?.gpuPendingBatches ?? 0} advances pending · {(gpuInfo?.gpuQueueStarved_ms ?? 0).toFixed(2)} ms last host gap</span></footer>
      </section>

      <section className="trace-card frame-graph-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME GRAPH</p><h2>Pass dependencies & resource flow</h2></div><small>SELECT A PASS TO INSPECT</small></header>
        <div className="frame-graph-flow">
          <div className="graph-section"><span>PHYSICS</span><div>{physicsStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < physicsStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
          {adaptiveTopologyStages.length > 0 && <><b className="graph-divider">EVENT-DRIVEN · NOT REPEATED PER ADVANCE</b><div className="graph-section async-graph-section"><span>ADAPTIVE TOPOLOGY</span><div>{adaptiveTopologyStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < adaptiveTopologyStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div></>}
          <b className="graph-divider">QUEUE ORDER · OVERDUE 60 HZ PRESENTATION RUNS FIRST</b>
          <div className="graph-section"><span>PRESENTATION</span><div>{renderStages.map((stage, index) => <div className="graph-node-wrap" key={stage.key}><button className={`graph-node ${stage.className} ${selectedStage?.key === stage.key ? "selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)}><i /><strong>{stage.shortLabel}</strong><small>{formatMs(stage.value, stageTimed(stage), stage.active)}</small></button>{index < renderStages.length - 1 && <b className="graph-arrow">→</b>}</div>)}</div></div>
        </div>
        {selectedStage && <article className="stage-inspector">
          <div className="stage-inspector-title"><span className={selectedStage.className} /><div><small>{selectedStage.group.toUpperCase()} PASS</small><h3>{selectedStage.label}</h3></div><strong>{formatMs(selectedStage.value, stageTimed(selectedStage), selectedStage.active)}</strong></div>
          <p>{selectedStage.description}</p>
          <div className="resource-columns"><div><small>READS</small>{selectedStage.reads.map((item) => <span key={item}><i>R</i>{item}</span>)}</div><div><small>WRITES</small>{selectedStage.writes.map((item) => <span key={item}><i>W</i>{item}</span>)}</div><div><small>WAITS FOR</small>{selectedStage.dependsOn.map((item) => <span key={item}><i>↳</i>{gpuStages.find((stage) => stage.key === item)?.shortLabel ?? item.toUpperCase()}</span>)}{selectedStage.sync && <span className="sync-detail"><i>⇄</i>{selectedStage.sync}</span>}</div></div>
          <div className="stage-capture-action"><div><small>DIAGNOSTIC RESOURCE</small><strong>{selectedCaptureTarget?.label ?? "Timing and counters only"}</strong><span>{selectedCaptureTarget ? `${selectedCaptureTarget.selectorLabel}${selectedCaptureTarget.units ? ` · ${selectedCaptureTarget.units}` : ""} · centre slice` : captureDisabledReason}</span></div><button onClick={captureBusy ? () => gpuStageCapture.cancel() : armSelectedCapture} disabled={!captureBusy && Boolean(captureDisabledReason)} title={captureDisabledReason}>{captureBusy ? "CANCEL CAPTURE" : "CAPTURE NEXT"}</button></div>
        </article>}
      </section>

      <section className="trace-card history-card">
        <header className="trace-card-header"><div><p className="eyebrow">FRAME HISTORY · {averageWindow === 1 ? "RAW" : `${averageWindow}-FRAME ROLLING AVERAGE`}</p><h2>Last {matchingHistory.length} frames</h2></div><div className="timeline-legend"><span><i className="history-gpu" />GPU</span><span><i className="history-cpu" />CPU</span><span><i className="history-budget" />Budget</span></div></header>
        <div className="history-chart"><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}><line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} /><polyline className="history-gpu" points={points("gpu")} /><polyline className="history-cpu" points={points("cpu")} />{safeIndex !== null && <line className="history-cursor" x1={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y1="2" x2={matchingHistory.length < 2 ? 0 : safeIndex / (matchingHistory.length - 1) * 100} y2="49" />}</svg><span>0</span><span>{historyMax.toFixed(1)} ms</span></div>
        <input type="range" min="0" max={Math.max(0, matchingHistory.length - 1)} value={safeIndex ?? Math.max(0, matchingHistory.length - 1)} onChange={(event) => setHistoryIndex(Number(event.target.value))} disabled={!matchingHistory.length} aria-label="Inspect a frame from history" />
      </section>

      <aside className="trace-card capture-card">
        <div><p className="eyebrow">DIAGNOSTIC CAPTURE · INSTRUMENTED</p><h2>{captureState.artifact?.label ?? selectedCaptureTarget?.label ?? "Stage resource capture"}</h2><span className={`capture-status phase-${captureState.phase}`}><i />{captureState.phase.toUpperCase()}</span></div>
        {captureState.phase === "ready" && captureState.artifact ? <div className="capture-result">
          <CapturePreview artifact={captureState.artifact} />
          <div className="capture-interpretation"><strong>{captureState.artifact.selectorLabel.toUpperCase()} · {captureState.artifact.dimensions.join("×")}</strong><p>{captureState.artifact.interpretation}</p></div>
          <dl><div><dt>Clean stage</dt><dd>{captureState.artifact.baseline.stage_ms === undefined ? "—" : `${captureState.artifact.baseline.stage_ms.toFixed(3)} ms`}</dd></div><div><dt>Range</dt><dd>{Number.isFinite(captureState.artifact.minimum) && Number.isFinite(captureState.artifact.maximum) ? `${captureState.artifact.minimum.toPrecision(3)} → ${captureState.artifact.maximum.toPrecision(3)}` : "Unavailable"}</dd></div><div><dt>Coverage</dt><dd>{captureState.artifact.totalValues.toLocaleString()} values{captureState.artifact.invalidValues > 0 ? ` · ${captureState.artifact.invalidValues.toLocaleString()} invalid` : ""}</dd></div><div><dt>Readback wall</dt><dd>{captureState.artifact.readbackWall_ms.toFixed(1)} ms*</dd></div></dl>
          <small>* Instrumented completion latency, never used as production stage timing. {(captureState.artifact.stagingBytes / 1024).toFixed(0)} KiB staged asynchronously.</small>
        </div> : <>
          <div className="architecture-model"><span><small>SELECTED</small><strong>{selectedStage?.shortLabel ?? "NONE"}</strong></span><span><small>PRODUCT</small><strong>{selectedCaptureTarget ? "SUMMARY" : "UNAVAILABLE"}</strong></span><span><small>PREVIEW</small><strong>{selectedCaptureTarget ? "≤256²" : "—"}</strong></span><span><small>RAW 3D</small><strong>DISABLED</strong></span></div>
          <p>{captureState.message ?? (selectedCaptureTarget ? "Capture the next matching stage boundary. A full-domain GPU summary and bounded centre-slice preview are read back asynchronously." : "Select a stage with a registered texture product. Timing-only stages remain visible in the frame graph.")}</p>
          <dl><div><dt>GPU physics</dt><dd>{measuredStages.length ? `${physicsPerStep_ms.toFixed(2)} ms / solve` : "sampling"}</dd></div><div><dt>CPU sync waits</dt><dd>{adaptive ? `${snapshot.adaptiveRebuildBlockedFrames} blocked frames` : "none observed"}</dd></div><div><dt>Trace fidelity</dt><dd>{adaptive ? "Pass + resource" : "Pass + slice"}</dd></div></dl>
          {captureState.phase === "failed" && <small className="capture-error">{captureState.message}</small>}
        </>}
      </aside>
    </div>
  </aside>;
}
