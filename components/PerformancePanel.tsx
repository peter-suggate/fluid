"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { getMethod } from "@/lib/methods";
import { measuredGPUTime_ms, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { advanceWallBreakdown, measuredGPUUtilization, performanceSchedule } from "@/lib/performance-scheduling";
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

/** Averaged frame usage against the presentation target, with a per-stage drill-down. */
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
  const [selectedStageKey, setSelectedStageKey] = useState("pressure");
  const [averageWindow, setAverageWindow] = useState(30);
  const captureState = useSyncExternalStore(gpuStageCapture.subscribe, gpuStageCapture.getSnapshot, gpuStageCapture.getServerSnapshot);

  const method = getMethod(activeMethodId);
  const matchingHistory = useMemo(
    () => history.filter((sample) => sample.methodId === activeMethodId && sample.renderTimingContext === liveSnapshot.renderTimingContext),
    [history, activeMethodId, liveSnapshot.renderTimingContext]
  );
  const windowSamples = matchingHistory.slice(-averageWindow);
  const snapshot = averagePerformanceSnapshots(windowSamples, liveSnapshot);
  const averagedFrameCount = Math.max(1, windowSamples.length);
  const contextMatches = snapshot.methodId === activeMethodId && snapshot.renderTimingContext === liveSnapshot.renderTimingContext;
  const physicsTimed = contextMatches && snapshot.gpuPhysicsTimingAvailable;
  const renderTimed = contextMatches && snapshot.gpuRenderTimingAvailable;
  const svoRendered = snapshot.effectiveRenderMode === "svo";
  const temporalEnabled = svoRendered && Boolean(snapshot.renderTimingContext?.includes(":temporal-true:"));
  const paused = runState === "paused";
  // This panel's presence requests continuous, completion-gated presentation
  // from the viewport even for a static scene whose simulation stays paused.
  const continuousPausedPresentation = paused;
  const adaptive = activeMethodId === "quadtree-tall-cell";
  const budget = 1000 / PRESENTATION_FPS;
  const topologyPath = snapshot.adaptiveInlineTopology ? "inline" : "async";
  const physicsStages = method.backend === "webgpu" ? physicsPerformanceStages({ methodId: activeMethodId, snapshot, contextMatches, pressureSolver: gpuInfo?.pressureSolver, topologyPath }) : [];
  const adaptiveTopologyStages = adaptive && topologyPath === "async" ? adaptiveTopologyPerformanceStages({ snapshot, contextMatches }) : [];
  const physicsOutputStage = activeMethodId === "octree" ? "sparse-publication" : activeMethodId === "quadtree-tall-cell" ? "surface-update" : activeMethodId === "tall-cell" || activeMethodId === "uniform" ? "projection" : "uploads";

  const renderStages: PerformanceStage[] = [
    { key: "extract", label: "Surface extraction", shortLabel: "SURFACE", value: contextMatches ? snapshot.gpuSurfaceExtraction_ms : 0, className: "stage-extract", timer: "render", group: "compute", active: true, description: "Extracts visible liquid surface geometry from the signed-distance field.", reads: ["signed distance φ", "active cells"], writes: ["surface vertices", "indirect draw args"], dependsOn: [physicsOutputStage] },
    { key: "caustics", label: "Water caustic map", shortLabel: "CAUSTICS", value: contextMatches ? snapshot.gpuCaustics_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: snapshot.gpuCaustics_ms > 0, description: "Updates the light-space water caustic target when the extracted liquid surface changes.", reads: ["surface vertices", "light transform"], writes: ["caustic texture"], dependsOn: ["extract"] },
    { key: "dry-scene", label: svoRendered ? "SVO traversal + dry shading" : "Raster dry scene", shortLabel: svoRendered ? "SVO SCENE" : "RASTER SCENE", value: contextMatches ? snapshot.gpuDryScene_ms : 0, className: "stage-scene", timer: "render", group: "graphics", active: true, description: svoRendered ? "Traverses the sparse voxel octree and exact scene primitives, then evaluates materials, direct visibility, shadows, media, and environment lighting into the dry color and G-buffer targets." : "Rasterizes the environment and rigid bodies behind the liquid.", reads: svoRendered ? ["published SVO", "materials + lights", "rigid primitives", "camera"] : ["body transforms", "environment", "caustic texture"], writes: svoRendered ? ["dry HDR", "surface G-buffer", "identity / media", "depth"] : ["scene color", "scene depth"], dependsOn: [snapshot.gpuCaustics_ms > 0 ? "caustics" : activeMethodId === "octree" ? "sparse-publication" : "uploads"] },
    { key: "svo-temporal", label: "SVO temporal resolve", shortLabel: "TEMPORAL", value: contextMatches ? snapshot.gpuSvoTemporal_ms : 0, className: "stage-scene", timer: "render", group: "graphics", active: temporalEnabled, description: "Reprojects, validates, clamps, and filters the SVO dry target before raster water composition.", reads: ["dry G-buffer", "history", "camera + rigid motion"], writes: ["filtered dry color", "history"], dependsOn: ["dry-scene"] },
    { key: "front-interface", label: sprayEnabled ? "Water + spray front interface" : "Water front interface", shortLabel: "FRONT", value: contextMatches ? snapshot.gpuInterfaceFront_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: true, description: sprayEnabled ? "Draws the liquid mesh and active spray ellipsoids into the nearest front position, normal, and depth attachments in one render pass." : "Draws the nearest liquid position, normal, and depth attachments.", reads: sprayEnabled ? ["surface vertices", "spray particle ring", "camera"] : ["surface vertices", "camera"], writes: ["front position", "front normal", "front depth"], dependsOn: ["extract"] },
    { key: "back-interface", label: sprayEnabled ? "Water + spray back interface" : "Water back interface", shortLabel: "BACK", value: contextMatches ? snapshot.gpuInterfaceBack_ms : 0, className: "stage-interface", timer: "render", group: "graphics", active: true, description: sprayEnabled ? "Draws the liquid mesh and active spray ellipsoids into the far back position, normal, and depth attachments in one render pass." : "Draws the far liquid position, normal, and depth attachments used to reconstruct thickness.", reads: sprayEnabled ? ["surface vertices", "spray particle ring", "camera"] : ["surface vertices", "camera"], writes: ["back position", "back normal", "back depth"], dependsOn: ["front-interface"] },
    { key: "composite", label: "Optical composite", shortLabel: "COMPOSITE", value: contextMatches ? snapshot.gpuOpticalComposite_ms : 0, className: "stage-composite", timer: "render", group: "graphics", active: true, description: "Combines refraction, absorption, reflection, and the dry scene.", reads: ["scene color", "front/back depth", "normals"], writes: ["water color"], dependsOn: [temporalEnabled ? "svo-temporal" : "dry-scene", "back-interface"] },
    { key: "overlays", label: "Inspection + diagnostic overlays", shortLabel: "OVERLAYS", value: contextMatches ? snapshot.gpuOverlays_ms : 0, className: "stage-overhead", timer: "render", group: "graphics", active: snapshot.gpuOverlays_ms > 0, description: "Renders raw-voxel or brick inspection and any enabled grid, technique, or audit overlay after optical composition.", reads: ["sparse scene records", "diagnostic fields", "water color"], writes: ["annotated presentation target"], dependsOn: ["composite"] },
    { key: "upscale", label: "Final upscale", shortLabel: "UPSCALE", value: contextMatches ? snapshot.gpuUpscale_ms : 0, className: "stage-upscale", timer: "render", group: "graphics", active: true, description: "Resolves the internal render target into the presentation surface.", reads: ["water color / inspection target"], writes: ["swapchain"], dependsOn: [snapshot.gpuOverlays_ms > 0 ? "overlays" : "composite"], sync: "Presentation boundary" }
  ];
  const gpuStages = [...physicsStages, ...adaptiveTopologyStages, ...renderStages];
  const stageTimed = (stage: PerformanceStage) => stage.timer === "physics" ? physicsTimed : stage.timer === "render" ? renderTimed : contextMatches && snapshot.adaptiveRebuildCompletedCount > 0;
  const measuredStages = [...physicsStages, ...renderStages].filter(stageTimed);
  const cpuOther = Math.max(0, snapshot.cpuFrame_ms - snapshot.cpuPhysicsSubmit_ms - snapshot.cpuDataUpload_ms - snapshot.cpuRenderEncode_ms);
  const cpuStages = [
    { key: "simulation", label: method.backend === "cpu" ? "Fluid + rigid solve" : "Rigid bodies + CPU oracle", shortLabel: "SIM", value: snapshot.cpuSimulation_ms, note: method.backend === "cpu" ? "CPU solver" : "Includes the coarse validation solve" },
    { key: "encode", label: "Physics encode + submit", shortLabel: "ENCODE", value: snapshot.cpuPhysicsSubmit_ms, note: "Build commands + queue.submit" },
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
  const batchGPU_ms = paused ? 0 : schedule.batchGPU_ms;
  const displayedBatchDepth = paused ? 0 : submissionBatchDepth;
  const submissionEnvelope_ms = batchGPU_ms + renderPerFrame_ms;
  const completionRate = gpuInfo?.gpuCompletionWall_ms && gpuInfo.gpuCompletionSimulation_s
    ? gpuInfo.gpuCompletionSimulation_s * 1000 / gpuInfo.gpuCompletionWall_ms
    : null;
  const measuredUtilization = paused && !continuousPausedPresentation ? { physics: 0, presentation: 0, total: 0 } : measuredGPUUtilization({
    physics_ms: paused ? undefined : gpuInfo?.gpuStep_ms,
    physicsCompletionInterval_ms: paused ? undefined : gpuInfo?.gpuCompletionWall_ms,
    presentation_ms: liveSnapshot.gpuRenderTimingAvailable ? liveSnapshot.gpuRender_ms : undefined,
    presentationInterval_ms: gpuInfo?.gpuPresentationWall_ms
  });
  const measuredUtilizationPercent = measuredUtilization ? measuredUtilization.total * 100 : null;
  const realtimeDemand_ms = paused ? continuousPausedPresentation ? renderPerFrame_ms : 0 : schedule.gpuDemandPerFrame_ms;
  const demandPercent = realtimeDemand_ms / budget * 100;
  const cpuPercent = cpuTotal / budget * 100;
  const frameUsagePercent = Math.max(demandPercent, cpuPercent);
  const overBudget = frameUsagePercent > 100;
  const cpuWindow = windowSamples.map((sample) => sample.cpuSimulation_ms + sample.cpuFrame_ms).sort((a, b) => a - b);
  const cpuP95_ms = cpuWindow.length ? cpuWindow[Math.min(cpuWindow.length - 1, Math.floor(cpuWindow.length * .95))] : cpuTotal;
  const gpuConstrained = demandPercent > 100;
  const cpuSpikeConstrained = !paused && cpuP95_ms > budget;
  const unexplainedSlowdown = !paused && !gpuConstrained && !cpuSpikeConstrained && observedSimRate !== null && observedSimRate < .95;
  const lastAdvance = advanceWallBreakdown(gpuInfo);
  const encodeBreakdown = gpuInfo?.cpuAdvanceEncodeBreakdown;
  const encodeStages = encodeBreakdown ? [
    { key: "setup", label: "SETUP + PARAMS", value: encodeBreakdown.setup_ms },
    { key: "topology", label: "TOPOLOGY COMMANDS", value: encodeBreakdown.topology_ms },
    { key: "pressure", label: "PRESSURE + PROJECT COMMANDS", value: encodeBreakdown.pressureProjection_ms },
    { key: "surface", label: "SURFACE COMMANDS", value: encodeBreakdown.surface_ms },
    { key: "publication", label: "PUBLICATION COMMANDS", value: encodeBreakdown.publication_ms },
    { key: "finalize", label: "FINALIZE + SUBMIT", value: encodeBreakdown.finalize_ms },
  ] : [];
  const headroom = budget - Math.max(realtimeDemand_ms, cpuTotal);
  const heroScale = Math.max(budget * 1.12, realtimeDemand_ms, cpuTotal, .01);
  const budgetTick = budget / heroScale * 100;
  const pausedPresentationNote = renderTimed ? `Last presentation ${renderPerFrame_ms.toFixed(2)} ms on GPU` : "Awaiting presentation timestamp";
  const verdict = paused
    ? gpuConstrained ? "Continuous presentation exceeds the frame target" : "Paused · continuous presentation only"
    : gpuConstrained ? "GPU demand exceeds the frame target"
      : cpuSpikeConstrained ? "CPU p95 exceeds the frame deadline"
        : unexplainedSlowdown ? "Slowdown not explained by timestamped work"
          : measuredStages.length ? `${headroom.toFixed(2)} ms headroom` : "sampling…";

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

  const sampleLabel = averageWindow === 1 ? "SINGLE FRAME" : `${averagedFrameCount}-FRAME AVERAGE`;
  const timestampQueriesSupported = liveSnapshot.gpuRenderTimestampSupported;
  const timerDescription = gpuStatus.state !== "ready" ? "GPU unavailable" : !timestampQueriesSupported ? "GPU timestamps unavailable · CPU + queue fences only" : continuousPausedPresentation && renderTimed ? "Paused simulation · continuous presentation timestamps" : physicsTimed && renderTimed ? "Hardware timestamps · physics + presentation" : renderTimed ? "Presentation timestamps · physics pending" : physicsTimed ? "Physics timestamps · presentation pending" : "Awaiting timestamp sample";
  const averagedHistory = rollingPerformanceSnapshots(matchingHistory, averageWindow);
  const historyValues = averagedHistory.map((sample) => ({ gpu: measuredGPUTime_ms(sample), cpu: sample.cpuSimulation_ms + sample.cpuFrame_ms }));
  const historyMax = Math.max(budget, ...historyValues.flatMap((sample) => [sample.gpu, sample.cpu]), 0.01);
  const points = (key: "gpu" | "cpu") => historyValues.map((sample, index) => `${historyValues.length < 2 ? 0 : index / (historyValues.length - 1) * 100},${48 - Math.min(sample[key] / historyMax, 1) * 44}`).join(" ");

  const stageScale = (stages: PerformanceStage[], floor_ms = 0) => Math.max(...stages.map((stage) => stageTimed(stage) && stage.active ? stage.value : 0), floor_ms, .01);
  const stageRows = (stages: PerformanceStage[], scale = stageScale(stages)) => stages.map((stage) => <button key={stage.key} className={`perf-row${selectedStage?.key === stage.key ? " selected" : ""}`} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value, stageTimed(stage), stage.active)}`}>
    <i className={stage.className} />
    <span>{stage.shortLabel}</span>
    <div className="perf-row-bar">{stageTimed(stage) && stage.active && stage.value > 0 && <b className={stage.className} style={{ width: `${Math.max(1, stage.value / scale * 100)}%` }} />}</div>
    <output>{formatMs(stage.value, stageTimed(stage), stage.active)}</output>
  </button>);
  // A paused manual step often costs far more wall time than its timestamped
  // shaders (driver scheduling, pipeline compilation, fence latency), so the
  // physics list reconciles against the queue-fence wall explicitly.
  const untimedAdvance_ms = paused && lastAdvance && lastAdvance.untimestampedQueue_ms > .05 ? lastAdvance.untimestampedQueue_ms : 0;
  const physicsRowScale = stageScale(physicsStages, untimedAdvance_ms);
  const cpuScale = Math.max(...cpuStages.map((stage) => stage.value), .01);
  const encodeScale = Math.max(...encodeStages.map((stage) => stage.value), .01);

  return <aside id="performance-panel" className="right-panel panel-scroll performance-panel" aria-label="Performance profiler" data-testid="performance-panel" data-method={activeMethodId}
    data-render-timing-context={liveSnapshot.renderTimingContext}
    data-render-timing-epoch={liveSnapshot.renderTimingEpoch}
    data-render-timing-sample-id={liveSnapshot.renderTimingSampleId}
    data-render-timestamp-supported={liveSnapshot.gpuRenderTimestampSupported}
    data-render-timing-available={liveSnapshot.gpuRenderTimingAvailable}
    data-render-timing-total-ms={liveSnapshot.gpuRenderTimingAvailable ? liveSnapshot.gpuRender_ms : undefined}>
    <header className="perf-header">
      <div>
        <p className="eyebrow">LIVE PROFILE · {sampleLabel}</p>
        <h1>Performance trace</h1>
        <span className="perf-source"><i />{timerDescription}</span>
      </div>
      <label className="averaging-control"><small>AVERAGING</small><select value={averageWindow} onChange={(event) => setAverageWindow(Number(event.target.value))} aria-label="Timing averaging window"><option value="1">1 frame</option><option value="10">10 frames</option><option value="30">30 frames</option><option value="60">60 frames</option><option value="100">100 frames</option></select></label>
      <button className="panel-close" onClick={() => setRightPanel(null)} aria-label="Close performance profiler">×</button>
    </header>

    <section className={`perf-hero${overBudget ? " over-budget" : ""}`} aria-label="Averaged frame usage against the presentation target">
      <div className="perf-gauge" style={{ "--usage": `${Math.min(100, Math.max(0, frameUsagePercent))}%` } as CSSProperties} title="Averaged frame demand divided by the presentation deadline">
        <div><strong>{measuredStages.length ? `${frameUsagePercent.toFixed(0)}%` : "—"}</strong><small>OF {PRESENTATION_FPS} HZ FRAME</small></div>
      </div>
      <div className="perf-budget">
        <div className="perf-budget-head"><small>FRAME USAGE VS {PRESENTATION_FPS} HZ TARGET</small><strong>{verdict}</strong></div>
        <div className="perf-budget-row">
          <span>GPU</span>
          <div className="perf-track" aria-label={`${(paused ? 0 : schedule.physicsPerFrame_ms).toFixed(2)} milliseconds physics and ${renderPerFrame_ms.toFixed(2)} milliseconds presentation against a ${budget.toFixed(2)} millisecond deadline`}>
            {!paused && physicsStages.map((stage) => stageTimed(stage) && stage.active && stage.value > 0 && <button key={stage.key} className={`perf-flame ${stage.className}${selectedStage?.key === stage.key ? " selected" : ""}`} style={{ width: `${stage.value * schedule.advancesPerFrame / heroScale * 100}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value)} × ${schedule.advancesPerFrame.toFixed(2)} advances / frame`}><span>{stage.shortLabel}</span></button>)}
            {renderStages.map((stage) => stageTimed(stage) && stage.active && stage.value > 0 && <button key={stage.key} className={`perf-flame ${stage.className}${selectedStage?.key === stage.key ? " selected" : ""}`} style={{ width: `${stage.value / heroScale * 100}%` }} onClick={() => setSelectedStageKey(stage.key)} title={`${stage.label} · ${formatMs(stage.value)}`}><span>{stage.shortLabel}</span></button>)}
            <b className="perf-tick" style={{ left: `${budgetTick}%` }} />
          </div>
          <output>{measuredStages.length ? `${realtimeDemand_ms.toFixed(2)} ms` : "sampling…"}</output>
        </div>
        <div className="perf-budget-row">
          <span>CPU</span>
          <div className="perf-track" aria-label={`${cpuTotal.toFixed(2)} milliseconds main-thread work against a ${budget.toFixed(2)} millisecond deadline`}>
            {cpuStages.map((stage) => stage.value > 0 && <span key={stage.key} className="perf-flame seg-cpu" style={{ width: `${stage.value / heroScale * 100}%` }} title={`${stage.label} · ${formatMs(stage.value)} · ${stage.note}`}><span>{stage.shortLabel}</span></span>)}
            <b className="perf-tick" style={{ left: `${budgetTick}%` }} />
          </div>
          <output>{cpuTotal.toFixed(2)} ms</output>
        </div>
        <div className="perf-budget-key">
          <span><b className="seg-physics" />physics {paused ? "0.00" : schedule.physicsPerFrame_ms.toFixed(2)}</span>
          <span><b className="seg-render" />render {renderPerFrame_ms.toFixed(2)}</span>
          <span><b className="seg-cpu" />cpu {cpuTotal.toFixed(2)}</span>
          <strong>{budget.toFixed(2)} ms deadline</strong>
        </div>
      </div>
    </section>

    <section className="perf-meta" aria-label="Measured scheduling summary">
      <div><small>GPU BUSY</small><strong>{measuredUtilizationPercent === null ? "—" : `${measuredUtilizationPercent.toFixed(0)}%`}</strong><span>{measuredUtilization === null ? "awaiting completion cadence" : paused ? pausedPresentationNote : `${(measuredUtilization.physics * 100).toFixed(0)}% physics · ${(measuredUtilization.presentation * 100).toFixed(0)}% present`}</span></div>
      <div className={gpuConstrained || cpuSpikeConstrained ? "constrained" : undefined}><small>HEADROOM</small><strong>{measuredStages.length ? `${headroom.toFixed(2)} ms` : "—"}</strong><span>{gpuConstrained ? `${Math.abs(headroom).toFixed(2)} ms over the deadline` : cpuSpikeConstrained ? `CPU p95 ${cpuP95_ms.toFixed(2)} ms` : `per frame at ${PRESENTATION_FPS} Hz`}</span></div>
      <div><small>{paused ? "LAST PHYSICS ADVANCE" : "SIM CADENCE"}</small><strong>{paused ? lastAdvance ? `${lastAdvance.wall_ms.toFixed(1)} ms wall` : "no advance sampled" : `${schedule.advancesPerFrame.toFixed(2)} adv / frame`}</strong><span>{paused ? lastAdvance ? `CPU encode ${lastAdvance.encode_ms.toFixed(1)} · queue fence ${lastAdvance.queueFence_ms.toFixed(1)} ms` : "presentation redraws continuously" : `×${displayedBatchDepth} per submission · ${schedule.pressureSolvesPerFrame.toFixed(2)} pressure solves`}</span></div>
      <div><small>OBSERVED RATE</small><strong>{paused ? "—" : observedSimRate === null ? "measuring…" : `×${observedSimRate.toFixed(2)} realtime`}</strong><span>{paused ? "simulation paused" : completionRate === null ? "queue sampling" : `queue completion ×${completionRate.toFixed(2)}`}</span></div>
    </section>

    <section className="perf-history" aria-label="Recent frame history">
      <header><small>LAST {matchingHistory.length} FRAMES{averageWindow > 1 ? ` · ${averageWindow}-FRAME ROLLING AVERAGE` : " · RAW"}</small><div className="perf-legend"><span><i className="history-gpu" />GPU</span><span><i className="history-cpu" />CPU</span><span><i className="history-budget" />Target</span></div></header>
      <svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label={`Recent timing history; vertical scale zero to ${historyMax.toFixed(1)} milliseconds`}>
        <line x1="0" y1={48 - budget / historyMax * 44} x2="100" y2={48 - budget / historyMax * 44} />
        <polyline className="history-gpu" points={points("gpu")} />
        <polyline className="history-cpu" points={points("cpu")} />
      </svg>
      <footer><span>0 ms</span><span>{historyMax.toFixed(1)} ms</span></footer>
    </section>

    <section className="perf-breakdown" aria-label="Per-stage breakdown">
      {physicsStages.length > 0 && <div className="perf-group">
        <header><small>GPU PHYSICS · PER ADVANCE</small><output>{!physicsTimed ? "awaiting timestamps" : untimedAdvance_ms > 0 ? `${physicsPerStep_ms.toFixed(2)} ms shaders · ${(lastAdvance?.queueFence_ms ?? 0).toFixed(1)} ms queue wall` : `${physicsPerStep_ms.toFixed(2)} ms / advance`}</output></header>
        <div className="perf-rows">
          {stageRows(physicsStages, physicsRowScale)}
          {untimedAdvance_ms > 0 && <div className="perf-row perf-row-untimed" title="Queue-fence wall time no shader timestamp covers: driver scheduling, pipeline compilation, bind and barrier overhead, and fence latency. It dominates when each dispatch is smaller than the timer quantum.">
            <i />
            <span>QUEUE + DRIVER</span>
            <div className="perf-row-bar"><b style={{ width: `${Math.max(1, untimedAdvance_ms / physicsRowScale * 100)}%` }} /></div>
            <output>{untimedAdvance_ms.toFixed(1)} ms wall</output>
          </div>}
        </div>
      </div>}
      {adaptiveTopologyStages.length > 0 && <div className="perf-group">
        <header><small>ADAPTIVE TOPOLOGY · EVENT-DRIVEN, NOT PER ADVANCE</small><output>{snapshot.adaptiveRebuildWall_ms ? `${snapshot.adaptiveRebuildWall_ms.toFixed(1)} ms wall${snapshot.adaptiveRebuildPending ? " · in flight" : ""}` : "no rebuild sampled"}</output></header>
        <div className="perf-rows">{stageRows(adaptiveTopologyStages)}</div>
      </div>}
      <div className="perf-group">
        <header><small>GPU PRESENTATION · PER FRAME</small><output>{renderTimed ? `${renderPerFrame_ms.toFixed(2)} ms / frame` : "awaiting timestamps"}</output></header>
        <div className="perf-rows">{stageRows(renderStages)}</div>
        <div className="perf-submission"><span>ONE GPU SUBMISSION · ×{displayedBatchDepth} ADVANCES + PRESENT</span><output>{measuredStages.length ? `${submissionEnvelope_ms.toFixed(2)} ms` : "—"}</output></div>
      </div>
      <div className="perf-group">
        <header><small>CPU · MAIN THREAD</small><output>{cpuTotal.toFixed(2)} ms / frame</output></header>
        <div className="perf-rows">{cpuStages.map((stage) => <div key={stage.key} className="perf-row" title={`${stage.label} · ${stage.note}`}>
          <i className="seg-cpu" />
          <span>{stage.shortLabel}</span>
          <div className="perf-row-bar">{stage.value > 0 && <b className="seg-cpu" style={{ width: `${Math.max(1, stage.value / cpuScale * 100)}%` }} />}</div>
          <output>{formatMs(stage.value)}</output>
        </div>)}</div>
        {lastAdvance && <div className="perf-submission"><span>LAST PHYSICS ADVANCE · CPU encode {lastAdvance.encode_ms.toFixed(1)} ms + queue fence {lastAdvance.queueFence_ms.toFixed(1)} ms · timestamped shaders {lastAdvance.timestampedGPU_ms.toFixed(3)} ms · untimestamped queue/driver {lastAdvance.untimestampedQueue_ms.toFixed(1)} ms</span><output>{lastAdvance.wall_ms.toFixed(1)} ms wall</output></div>}
        {encodeStages.length > 0 && <>
          <header><small>LAST PHYSICS ENCODE · SOLVER ATTRIBUTION</small><output>{encodeStages.reduce((sum, stage) => sum + stage.value, 0).toFixed(1)} ms</output></header>
          <div className="perf-rows">{encodeStages.map((stage) => <div key={stage.key} className="perf-row" title={`${stage.label} · host command construction only`}>
            <i className="seg-cpu" />
            <span>{stage.label}</span>
            <div className="perf-row-bar">{stage.value > 0 && <b className="seg-cpu" style={{ width: `${Math.max(1, stage.value / encodeScale * 100)}%` }} />}</div>
            <output>{formatMs(stage.value)}</output>
          </div>)}</div>
        </>}
      </div>
    </section>

    {selectedStage && <section className="perf-inspector" aria-label="Stage inspector">
      <header>
        <span className={selectedStage.className} />
        <div><small>{selectedStage.group.toUpperCase()} PASS · {selectedStage.timer === "physics" ? "PHYSICS QUEUE" : selectedStage.timer === "render" ? "PRESENTATION" : "ASYNC WORKER"}</small><h2>{selectedStage.label}</h2></div>
        <strong>{formatMs(selectedStage.value, stageTimed(selectedStage), selectedStage.active)}</strong>
      </header>
      <p>{selectedStage.description}</p>
      <div className="perf-io">
        <div><small>READS</small>{selectedStage.reads.map((item) => <span key={item}><i>R</i>{item}</span>)}</div>
        <div><small>WRITES</small>{selectedStage.writes.map((item) => <span key={item}><i>W</i>{item}</span>)}</div>
        <div><small>WAITS FOR</small>{selectedStage.dependsOn.map((item) => <span key={item}><i>↳</i>{gpuStages.find((stage) => stage.key === item)?.shortLabel ?? item.toUpperCase()}</span>)}{selectedStage.sync && <span className="sync-detail"><i>⇄</i>{selectedStage.sync}</span>}</div>
      </div>
      <div className="perf-capture">
        <div className="perf-capture-head">
          <div><small>DIAGNOSTIC RESOURCE</small><strong>{selectedCaptureTarget?.label ?? "Timing and counters only"}</strong><span>{selectedCaptureTarget ? `${selectedCaptureTarget.selectorLabel}${selectedCaptureTarget.units ? ` · ${selectedCaptureTarget.units}` : ""} · centre slice` : captureDisabledReason}</span></div>
          <span className={`perf-phase phase-${captureState.phase}`}><i />{captureState.phase.toUpperCase()}</span>
          <button onClick={captureBusy ? () => gpuStageCapture.cancel() : armSelectedCapture} disabled={!captureBusy && Boolean(captureDisabledReason)} title={captureDisabledReason}>{captureBusy ? "CANCEL" : "CAPTURE NEXT"}</button>
        </div>
        {captureState.phase === "ready" && captureState.artifact ? <div className="perf-capture-result">
          <CapturePreview artifact={captureState.artifact} />
          <div><strong>{captureState.artifact.label.toUpperCase()} · {captureState.artifact.selectorLabel.toUpperCase()} · {captureState.artifact.dimensions.join("×")}</strong><p>{captureState.artifact.interpretation}</p></div>
          <dl>
            <div><dt>Clean stage</dt><dd>{captureState.artifact.baseline.stage_ms === undefined ? "—" : `${captureState.artifact.baseline.stage_ms.toFixed(3)} ms`}</dd></div>
            <div><dt>Range</dt><dd>{Number.isFinite(captureState.artifact.minimum) && Number.isFinite(captureState.artifact.maximum) ? `${captureState.artifact.minimum.toPrecision(3)} → ${captureState.artifact.maximum.toPrecision(3)}` : "Unavailable"}</dd></div>
            <div><dt>Coverage</dt><dd>{captureState.artifact.totalValues.toLocaleString()} values{captureState.artifact.invalidValues > 0 ? ` · ${captureState.artifact.invalidValues.toLocaleString()} invalid` : ""}</dd></div>
            <div><dt>Readback wall</dt><dd>{captureState.artifact.readbackWall_ms.toFixed(1)} ms*</dd></div>
          </dl>
          <small>* Instrumented completion latency, never used as production stage timing. {(captureState.artifact.stagingBytes / 1024).toFixed(0)} KiB staged asynchronously.</small>
        </div> : <p>{captureState.message ?? (selectedCaptureTarget ? "Capture the next matching stage boundary. A full-domain GPU summary and a bounded centre-slice preview are read back asynchronously without blocking the queue." : "This stage has no registered texture product; its timing stays visible in the breakdown above.")}</p>}
        {captureState.phase === "failed" && <small className="perf-capture-error">{captureState.message}</small>}
      </div>
    </section>}
  </aside>;
}
