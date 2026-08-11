"use client";

import { useMemo } from "react";
import { VISUALIZATION_FIELDS } from "@/lib/visualization-catalog";
import type { FieldVisualization } from "@/lib/visualization-registry";
import { PerformanceActivityGrid } from "./PerformanceActivityGrid";
import { PerformanceDials } from "./PerformanceDials";
import {
  averagePerformanceTraces,
  performanceTraceIsExact,
  type PaperPhaseId,
  type PerformanceTrace,
} from "@/lib/performance-trace";
import { performanceActivityFrameHasSettledEvidence } from "@/lib/performance-activity";
import { emptyPerformanceReport, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { getMethod } from "@/lib/methods";
import { simulation } from "@/lib/simulation/controller";
import { usePerformanceActivityStore } from "@/lib/stores/performance-activity-store";
import { useMethodStore } from "@/lib/stores/method-store";
import {
  usePerformanceInstrumentationStore,
  type PerformanceInstrumentationMode,
} from "@/lib/stores/performance-instrumentation-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useUIStore } from "@/lib/stores/ui-store";
import type { GridOverlayConfig, GridOverlayMode } from "@/lib/webgpu-renderer";

const PERFORMANCE_AVERAGE_WINDOW = 30;
const PERFORMANCE_CAPTURE_MODES: readonly {
  mode: PerformanceInstrumentationMode;
  label: string;
  title: string;
}[] = [
  { mode: "off", label: "OFF", title: "Production command graph with measurement work bypassed" },
  { mode: "timeline", label: "TIMELINE", title: "Low-overhead CPU and GPU timestamp timeline using production WGSL" },
  { mode: "activity", label: "DETAILED", title: "Perturbing shader-heartbeat capture with instrumented WGSL and readbacks" },
];
const PHASE_LAYOUT: Readonly<Record<PerformanceTrace["lane"], readonly [PaperPhaseId, string][]>> = {
  "main-thread": [
    ["frame-control", "Frame control + simulation admission"],
    ["scene-upload", "Scene and field uploads"],
    ["command-encoding", "Presentation command encoding"],
    ["other", "Post-submit scheduling + other"],
  ],
  physics: [
    ["coarse-grid", "Adaptive coarse-grid topology"],
    ["power-topology", "Power topology + physical faces"],
    ["velocity-advection", "Velocity transport + body forces"],
    ["pressure-system", "Pressure operator + RHS assembly"],
    ["pressure-solve", "Section 4.3 pressure solve"],
    ["velocity-projection", "Pressure projection"],
    ["velocity-extrapolation", "Velocity extrapolation"],
    ["fine-sdf-advection", "Fine SDF advection"],
    ["fine-sdf-redistance", "Fine SDF redistance"],
    ["adaptive-publication", "Adaptive publication"],
    ["other", "Other measured GPU work"],
  ],
  presentation: [
    ["surface-extraction", "Surface extraction + caustics"],
    ["dry-scene", "Dry scene lighting"],
    ["water-interfaces", "Water interfaces"],
    ["optical-composite", "Optical composite"],
    ["inspection-overlay", "Inspection overlays"],
    ["present", "Final upscale + present"],
    ["other", "Other measured GPU work"],
  ],
};

const SVO_CONE_PRESENTATION_PHASE_LAYOUT: readonly [PaperPhaseId, string][] = [
  ["surface-extraction", "Water surface extraction"],
  ["water-caustics", "Water caustic map"],
  ["svo-cone-lighting", "SVO cone-lighting prepass"],
  ["svo-environment-gi", "SVO environmental GI"],
  ["svo-voxel-light", "SVO voxel light cache"],
  ["svo-primary", "SVO primary visibility"],
  ["svo-rigid", "SVO rigid discovery"],
  ["svo-glass", "SVO thin-glass discovery"],
  ["dry-scene", "Live SVO dry scene"],
  ["water-front-interface", "Water + spray front interface"],
  ["water-back-interface", "Water + spray back interface"],
  ["optical-composite", "Optical composite"],
  ["inspection-overlay", "Inspection overlays"],
  ["present", "Final upscale + present"],
  ["other", "Other measured GPU work"],
] as const;

const isSvoConePresentationTrace = (trace: PerformanceTrace) =>
  trace.lane === "presentation" && /:gi:smooth:/.test(trace.context);

const stabilizePhaseLayout = (trace: PerformanceTrace | undefined) => {
  if (!trace) return undefined;
  // Physics producers already emit stable, command-adjacent semantic labels.
  // Preserve those checkpoints instead of folding distinct structured
  // transport, projection, and publication work into one broad category. The
  // single-phase queue fallback remains normalized below.
  if (trace.lane === "physics" && trace.measurementSource !== "gpu-queue-wall") {
    return trace;
  }
  const durations = new Map<PaperPhaseId, number>();
  for (const phase of trace.phases) {
    // Averaged phases carry per-encode durations plus the fraction of frames
    // that encoded them; the fixed layout wants the expected per-frame cost.
    const expected_ms = phase.duration_ms * (phase.encodedFraction ?? 1);
    durations.set(phase.id, (durations.get(phase.id) ?? 0) + expected_ms);
  }
  const layout = isSvoConePresentationTrace(trace)
    ? SVO_CONE_PRESENTATION_PHASE_LAYOUT
    : PHASE_LAYOUT[trace.lane];
  return {
    ...trace,
    phases: layout.map(([id, label]) => ({
      id,
      label: trace.measurementSource === "gpu-queue-wall" && id === "other"
        ? "GPU queue completion · hardware timestamps unavailable or invalid"
        : label,
      duration_ms: durations.get(id) ?? 0,
    })),
  };
};

const averagedTrace = (traces: PerformanceTrace[], windowSize: number) => {
  const unique = [...new Map(traces.map((trace) => [
    `${trace.domain}\0${trace.lane}\0${trace.context}\0${trace.sampleId}\0${trace.capturedAt_ms}`,
    trace,
  ])).values()];
  const latest = unique.at(-1);
  if (!latest) return undefined;
  const latestHardware = unique.findLast((trace) => trace.measurementSource === "gpu-hardware-timestamp");
  const source = latestHardware && latest.capturedAt_ms - latestHardware.capturedAt_ms <= 2_000
    ? "gpu-hardware-timestamp"
    : latest.measurementSource;
  const window = unique.filter((trace) => trace.measurementSource === source).slice(-windowSize);
  return stabilizePhaseLayout(averagePerformanceTraces(window));
};

/**
 * The observatory's cards are a read of the visualization catalog.
 *
 * Each card's label, description, legend and source live beside the pass that
 * publishes the data — see `lib/visualization-catalog.ts` — so a colour choice
 * and the swatch that explains it cannot drift apart across two files. Hidden
 * fields are modes the overlay can render that no picker offers; they are
 * declared for the shader harness and deliberately have no card.
 */
type PaperView = FieldVisualization & { mode: GridOverlayMode };

const PAPER_VIEWS: readonly PaperView[] = VISUALIZATION_FIELDS
  .filter((field) => !field.hidden) as readonly PaperView[];

export function PerformancePanel() {
  const report = useDiagnosticsStore((state) => state.performanceReport);
  const reports = useDiagnosticsStore((state) => state.performanceReports);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const methodId = useMethodStore((state) => state.methodId);
  const instrumentationEnabled = usePerformanceInstrumentationStore((state) => state.enabled);
  const instrumentationMode = usePerformanceInstrumentationStore((state) => state.mode);
  const setInstrumentationMode = usePerformanceInstrumentationStore((state) => state.setMode);
  const runState = useRuntimeStore((state) => state.runState);
  const activityHistory = usePerformanceActivityStore((state) => state.history);
  const selectedActivityFrameId = usePerformanceActivityStore((state) => state.selectedFrameId);
  const referenceActivityFrameId = usePerformanceActivityStore((state) => state.referenceFrameId);
  const selectActivityFrame = usePerformanceActivityStore((state) => state.selectFrame);
  const pinActivityReference = usePerformanceActivityStore((state) => state.pinReference);
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const lanes = useMemo(() => {
    const matching = reports.filter((candidate) =>
      candidate.methodId === methodId && candidate.context === report.context);
    const methodReports = reports.filter((candidate) => candidate.methodId === methodId);
    const samples = (lane: "cpu" | "physics" | "presentation") => {
      const current = matching.flatMap((sample) => sample[lane] ? [sample[lane]] : []);
      if (current.length > 0 || runState !== "paused") return { traces: current, held: false };
      const held = methodReports.findLast((sample) => sample[lane] !== undefined)?.[lane];
      return { traces: held ? [held] : [], held: held !== undefined };
    };
    const cpuSamples = samples("cpu");
    const physicsSamples = samples("physics");
    const presentationSamples = samples("presentation");
    return {
      cpu: { trace: averagedTrace(cpuSamples.traces, PERFORMANCE_AVERAGE_WINDOW), held: cpuSamples.held },
      physics: { trace: averagedTrace(physicsSamples.traces, PERFORMANCE_AVERAGE_WINDOW), held: physicsSamples.held },
      presentation: { trace: averagedTrace(presentationSamples.traces, PERFORMANCE_AVERAGE_WINDOW), held: presentationSamples.held },
    };
  }, [reports, methodId, report.context, runState]);
  const cpu = lanes.cpu.trace;
  const physics = lanes.physics.trace;
  const presentation = lanes.presentation.trace;
  const selectView = (view: PaperView) => {
    if (overlayMode === view.mode && overlayAxis !== "off") {
      setOverlayAxis("off");
      return;
    }
    setOverlayMode(view.mode);
    // A field card changes the publication, not the user's chosen
    // presentation. Only an inactive overlay needs the card's volume default.
    if (overlayAxis === "off") {
      setOverlayAxis(view.axis);
      if (view.axis === "volume") setOverlaySlice(0.42);
    }
  };
  // Only the views the active method registered: a card offering a publication
  // the solver never produces is a button that draws nothing.
  const supportedModes = new Set(getMethod(methodId).supportedFieldModes ?? []);
  const methodViews = PAPER_VIEWS.filter((view) => supportedModes.has(view.mode));
  const selectedView = methodViews.find((view) => view.mode === overlayMode);
  const volumeCapable = methodId === "octree";
  const traces = [cpu, physics, presentation].filter((trace): trace is PerformanceTrace => trace !== undefined);
  const allExact = traces.length === 3 && traces.every(performanceTraceIsExact);
  const holdingPausedMeasurements = lanes.cpu.held || lanes.physics.held || lanes.presentation.held;
  // Base traces and shader readback arrive independently. Keep the base-only
  // intermediate internal; once logical rows and their verdict arrive, expose
  // the settled evidence even when validation correctly marks it incomplete.
  const settledActivityHistory = useMemo(
    () => activityHistory.filter(performanceActivityFrameHasSettledEvidence),
    [activityHistory],
  );
  const selectedActivityFrame = settledActivityHistory.find((frame) =>
    frame.identity.frameId === selectedActivityFrameId) ?? settledActivityHistory.at(-1);
  const referenceActivityFrame = settledActivityHistory.find((frame) =>
    frame.identity.frameId === referenceActivityFrameId);
  const selectedActivityIndex = selectedActivityFrame
    ? settledActivityHistory.findIndex((frame) => frame.identity.frameId === selectedActivityFrame.identity.frameId)
    : -1;
  const changeInstrumentationMode = (mode: PerformanceInstrumentationMode) => {
    if (mode === instrumentationMode) return;
    const shaderVariantChanged = (instrumentationMode === "activity") !== (mode === "activity");
    setInstrumentationMode(mode);
    const activityStore = usePerformanceActivityStore.getState();
    if (mode !== "activity") activityStore.setEnabled(false);
    else if (activityStore.enabled) activityStore.beginGeneration();
    else activityStore.setEnabled(true);
    useDiagnosticsStore.getState().set({
      performanceReport: emptyPerformanceReport,
      performanceReports: [],
    });
    if (shaderVariantChanged) {
      simulation.reset();
      useRuntimeStore.getState().setNotice(
        mode === "activity"
          ? "Activity profiler enabled · rebuilding heartbeat-capable GPU pipeline variants"
          : "Activity profiler disabled · rebuilding production WGSL with profiling fragments omitted",
      );
    }
  };
  return <aside id="performance-panel" className="right-panel panel-scroll performance-panel performance-v2" aria-label="Performance and paper field observatory" data-testid="performance-panel" data-method={methodId} data-traces-exact={allExact}>
    <header className="performance-panel-header">
      <div><span>POWER LIQUIDS OBSERVATORY</span><h2>Measured work + live fields</h2></div>
      <div className="performance-panel-header-actions">
        <div className="measurement-mode" role="group" aria-label="Performance capture mode">
          <span>CAPTURE</span>
          {PERFORMANCE_CAPTURE_MODES.map(({ mode, label, title }) => <button
            key={mode}
            type="button"
            aria-pressed={instrumentationMode === mode}
            data-mode={mode}
            title={title}
            onClick={() => changeInstrumentationMode(mode)}
          >{label}</button>)}
        </div>
      </div>
    </header>

    <PerformanceDials />

    {instrumentationEnabled ? <>
    <PerformanceActivityGrid
      frame={selectedActivityFrame}
      cpu={instrumentationMode === "timeline" ? cpu : undefined}
      physics={instrumentationMode === "timeline" ? physics : undefined}
      presentation={instrumentationMode === "timeline" ? presentation : undefined}
      captureLabel={selectedActivityIndex >= 0 ? `CAPTURE ${selectedActivityIndex + 1}/${settledActivityHistory.length}` : undefined}
      captureOptions={settledActivityHistory.map((frame, index) => ({
        id: frame.identity.frameId,
        label: `#${index + 1} · ${frame.context}`,
      }))}
      selectedCaptureId={selectedActivityFrame?.identity.frameId}
      onSelectCapture={(frameId) => selectActivityFrame(frameId)}
      referenceFrame={referenceActivityFrame}
      referenceLabel={referenceActivityFrame ? "PINNED REFERENCE" : undefined}
      onSetReference={selectedActivityFrame
        && selectedActivityFrame.identity.frameId !== referenceActivityFrameId
        ? () => pinActivityReference(selectedActivityFrame.identity.frameId)
        : undefined}
      onClearReference={referenceActivityFrame ? () => pinActivityReference(undefined) : undefined}
      statusLabel={instrumentationMode === "activity" && !selectedActivityFrame
        ? "WAITING FOR COMPLETE CAPTURE"
        : holdingPausedMeasurements
          ? "PAUSED · LAST SETTLED CAPTURE"
          : instrumentationMode === "activity" ? "DETAILED ACTIVITY" : "TIMELINE ONLY"}
    />
    </> : <div className="performance-disabled-notice">
      <strong>Running without measurement instrumentation</strong>
      <span>Timestamp queries, stage-boundary encoder breaks, and trace-buffer resolves/readbacks are bypassed. Correctness synchronization remains active. TIMELINE keeps production WGSL and samples stage timestamps; DETAILED recompiles shader-heartbeat variants and is intentionally perturbing.</span>
    </div>}

    <section className="paper-observatory">
      <header><div><h3>Paper field observatory</h3><small>LIVE GPU PUBLICATIONS · NO FIELD READBACK</small></div><span>{methodId === "octree" ? "OCTREE AUTHORITY" : "SELECT OCTREE FOR FULL SET"}</span></header>
      <div className="paper-view-grid">
        {methodViews.map((view) => {
          const active = overlayAxis !== "off" && overlayMode === view.mode;
          return <button key={view.id} className={active ? "active" : ""} onClick={() => selectView(view)} aria-pressed={active}>
            <span>{view.figure ?? "§"}</span><strong>{view.label}</strong><small>{view.description}</small>
          </button>;
        })}
      </div>
      <div className="paper-view-controls">
        <div>
          <span>VIEW PLANE</span>
          <div role="group" aria-label="Paper field view plane">
            {(["x", "y", "z"] as const).map((axis) => <button key={axis} className={overlayAxis === axis ? "active" : ""} onClick={() => setOverlayAxis(axis)}>{axis.toUpperCase()}</button>)}
            <button className={overlayAxis === "volume" ? "active" : ""} disabled={!volumeCapable} onClick={() => setOverlayAxis("volume")}>VOLUME</button>
            <button className={overlayAxis === "off" ? "active" : ""} onClick={() => setOverlayAxis("off")}>HIDE</button>
          </div>
        </div>
        {overlayAxis !== "off" && <label>
          <span>{overlayAxis === "volume" ? "VOLUME OPACITY" : `${overlayAxis.toUpperCase()} SLICE`}</span>
          <input
            type="range"
            min={overlayAxis === "volume" ? 0.05 : 0}
            max={1}
            step={overlayAxis === "volume" ? 0.01 : 0.005}
            value={overlaySlice}
            onChange={(event) => setOverlaySlice(Number(event.currentTarget.value))}
            aria-label={overlayAxis === "volume" ? "Paper field volume opacity" : `Paper field ${overlayAxis} slice position`}
          />
          <output>{Math.round(overlaySlice * 100)}%</output>
        </label>}
      </div>
      <div className="paper-view-inspector" aria-live="polite">
        <header>
          <div><span>ACTIVE FIELD</span><strong>{selectedView?.label ?? overlayMode}</strong></div>
          <code>{overlayMode} · {overlayAxis === "off" ? "hidden" : overlayAxis}</code>
        </header>
        <p>{selectedView?.description ?? "This field was selected in the Render panel and reads the existing GPU publication directly."}</p>
        {selectedView && <>
          <small>SOURCE · {selectedView.source}</small>
          {selectedView.legend && <div className="paper-field-legend" aria-label={`${selectedView.label} legend`}>
            {selectedView.legend.map((entry) => <span key={entry.label}><i style={{ background: entry.swatch }} />{entry.label}</span>)}
          </div>}
        </>}
        <footer>{overlayAxis === "volume"
          ? "Orbit the camera to interrogate the ray-integrated live structure; the slider controls front-to-back opacity."
          : overlayAxis === "off"
            ? "Choose X, Y, Z, or an available volume view to display this field."
            : "Sweep the slider or drag the highlighted slice edge in the viewport. No field is read back to the CPU."}</footer>
      </div>
      <div className="paper-live-stats">
        <div><span>COARSE LEAVES</span><output>{gpuInfo?.quadtreeLeafCount?.toLocaleString() ?? "—"}</output></div>
        <div><span>POWER FACES</span><output>{gpuInfo?.quadtreeFaceCount?.toLocaleString() ?? "—"}</output></div>
        <div><span>FINE PAGES</span><output>{gpuInfo?.globalFineLevelSetLogicalBrickCount?.toLocaleString() ?? "—"}</output></div>
        <div><span>PRESSURE DOFs</span><output>{gpuInfo?.quadtreeLiquidDofCount?.toLocaleString() ?? "—"}</output></div>
        <div><span>ALLOCATED CELLS</span><output>{gpuInfo?.cellCount.toLocaleString() ?? "—"}</output></div>
        <div><span>MAX |∇·u|</span><output>{gpuInfo?.maxDivergenceAfter_s !== undefined ? gpuInfo.maxDivergenceAfter_s.toExponential(2) : "—"}</output></div>
      </div>
    </section>
  </aside>;
}
