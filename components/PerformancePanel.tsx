"use client";

import { useMemo, useState } from "react";
import {
  averagePerformanceTraces,
  performanceTraceAccounting,
  performanceTraceIsExact,
  type PaperPhaseId,
  type PerformanceTrace,
} from "@/lib/performance-trace";
import { isOctreeTechniqueOverlayMode } from "@/lib/octree-technique-debug";
import { emptyPerformanceReport, useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { usePerformanceInstrumentationStore } from "@/lib/stores/performance-instrumentation-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useUIStore } from "@/lib/stores/ui-store";
import type { GridOverlayConfig, GridOverlayMode } from "@/lib/webgpu-renderer";

const formatMs = (value: number) => value < 1 ? `${value.toFixed(3)} ms` : `${value.toFixed(2)} ms`;
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
    ["dry-scene", "Dry scene + temporal lighting"],
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
  ["svo-primary", "SVO traversal + dry shading"],
  ["svo-temporal", "SVO temporal resolve"],
  ["dry-scene", "Raster dry-scene fallback"],
  ["water-front-interface", "Water + spray front interface"],
  ["water-back-interface", "Water + spray back interface"],
  ["optical-composite", "Optical composite"],
  ["inspection-overlay", "Inspection overlays"],
  ["present", "Final upscale + present"],
  ["other", "Other measured GPU work"],
] as const;

const isSvoConePresentationTrace = (trace: PerformanceTrace) =>
  trace.lane === "presentation" && trace.context.includes(":lighting-cone:smooth:svo:");

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
    durations.set(phase.id, (durations.get(phase.id) ?? 0) + phase.duration_ms);
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

const averagedLane = (traces: PerformanceTrace[], windowSize: number) => {
  const unique = [...new Map(traces.map((trace) => [
    `${trace.domain}\0${trace.lane}\0${trace.context}\0${trace.sampleId}\0${trace.capturedAt_ms}`,
    trace,
  ])).values()];
  const latest = unique.at(-1);
  if (!latest) return { trace: undefined, sampleCount: 0 };
  const latestSegmented = unique.findLast((trace) => trace.measurementSource === "gpu-segmented-queue-wall");
  const latestHardware = unique.findLast((trace) => trace.measurementSource === "gpu-hardware-timestamp");
  const source = latestSegmented && latest.capturedAt_ms - latestSegmented.capturedAt_ms <= 10_000
    ? "gpu-segmented-queue-wall"
    : latestHardware && latest.capturedAt_ms - latestHardware.capturedAt_ms <= 2_000
      ? "gpu-hardware-timestamp"
      : latest.measurementSource;
  const window = unique.filter((trace) => trace.measurementSource === source).slice(-windowSize);
  return {
    trace: stabilizePhaseLayout(averagePerformanceTraces(window)),
    sampleCount: window.length,
  };
};

function TraceLane({ trace, title, sampleCount }: { trace?: PerformanceTrace; title: string; sampleCount: number }) {
  if (!trace) {
    return <section className="trace-lane trace-empty"><header><h3>{title}</h3><span>WAITING FOR COMPLETE SAMPLE</span></header></section>;
  }
  const scale = Math.max(trace.total_ms, 0.000001);
  const accounting = performanceTraceAccounting(trace);
  const exact = accounting.exact;
  const sourceLabel = trace.measurementSource === "gpu-hardware-timestamp"
    ? "HARDWARE TIMESTAMPS"
    : trace.measurementSource === "gpu-segmented-queue-wall"
      ? "INTRUSIVE QUEUE CHECKPOINT PROBE"
    : trace.measurementSource === "gpu-queue-wall"
      ? "QUEUE COMPLETION FALLBACK"
      : "ACTIVE CPU WALL";
  const sumLabel = trace.measurementSource === "gpu-hardware-timestamp"
    ? "GPU EXECUTION SUM"
    : trace.measurementSource === "cpu-active-wall"
      ? "ACTIVE WALL SUM"
      : "QUEUE-WALL SUM";
  return <section
    className="trace-lane"
    data-domain={trace.domain}
    data-lane={trace.lane}
    data-observed-total-ms={accounting.observedTotal_ms}
    data-accounted-ms={accounting.accounted_ms}
    data-closure-error-ms={accounting.closureError_ms}
    data-exact={exact}
    data-measurement-source={trace.measurementSource}
  >
    <header>
      <div className="trace-title-block">
        <h3>{title}</h3>
        <div className="trace-meta">
          <span>AVG {sampleCount} · LATEST {trace.sampleId}</span>
          <code title={trace.context}>{trace.context}</code>
          <span>{sourceLabel}</span>
        </div>
      </div>
      <div className="trace-total"><strong>{formatMs(trace.total_ms)}</strong><span>{exact ? sumLabel : "INVALID"}</span></div>
    </header>
    {trace.measurementSource === "gpu-queue-wall" && <p className="trace-source-warning">
      Hardware timestamps were unavailable or invalid for this sample. Total queue wall time is measured exactly; semantic GPU phase attribution requires a valid timestamp sample.
    </p>}
    {trace.measurementSource === "gpu-segmented-queue-wall" && <p className="trace-source-warning trace-source-measured">
      Hardware timestamps were unavailable or invalid. Each row is an intrusive submit-to-queue-callback wall-time checkpoint, not GPU execution time. Submission, synchronization, and browser callback latency make these values upper bounds; use them to locate expensive steps, not to estimate normal simulation throughput.
    </p>}
    <div className="trace-stack" aria-label={`${title}: ${formatMs(trace.total_ms)} total`}>
      {trace.phases.filter((phase) => phase.duration_ms > 0).map((phase, index) => <span
        key={`${phase.id}-${index}`}
        className={`trace-segment phase-${phase.id}`}
        style={{ width: `${phase.duration_ms / scale * 100}%` }}
        title={`${phase.label}: ${formatMs(phase.duration_ms)}`}
      />)}
    </div>
    <div className="trace-rows">
      {trace.phases.map((phase, index) => <div className="trace-row" key={`${phase.id}-${index}`}>
        <i className={`phase-${phase.id}`} />
        <span>{phase.label}</span>
        <div><b className={`phase-${phase.id}`} style={{ width: `${phase.duration_ms > 0 ? Math.max(1, phase.duration_ms / scale * 100) : 0}%` }} /></div>
        <output>{formatMs(phase.duration_ms)}</output>
      </div>)}
    </div>
    <div className="trace-accounting" aria-label={`${title} accounting closure`}>
      <span><small>OBSERVED TOTAL</small><output>{formatMs(accounting.observedTotal_ms)}</output></span>
      <span><small>ACCOUNTED PHASE SUM</small><output>{formatMs(accounting.accounted_ms)}</output></span>
      <span><small>CLOSURE ERROR</small><output>{accounting.closureError_ms === 0 ? "0.000 ms" : formatMs(accounting.closureError_ms)}</output></span>
    </div>
  </section>;
}

type PaperView = {
  id: string;
  figure: string;
  label: string;
  description: string;
  source: string;
  mode: GridOverlayMode;
  axis: GridOverlayConfig["axis"];
  legend: readonly { swatch: string; label: string }[];
};

const PAPER_VIEWS: readonly PaperView[] = [
  {
    id: "layered-grid", figure: "FIG. 6", label: "Fine SDF layer",
    description: "Interface core, redistanced support, and compact coarse authority outside the fine allocation.",
    source: "Sparse fine-band hash, sample flags, and publication controls",
    mode: "fine-band-lifecycle", axis: "z",
    legend: [
      { swatch: "#ff168f", label: "interface core" },
      { swatch: "#15c8db", label: "valid redistanced sample" },
      { swatch: "#10255e", label: "compact coarse authority" },
      { swatch: "#ff1738", label: "failed or stale publication" },
    ],
  },
  {
    id: "coarse-grid", figure: "FIG. 6/8", label: "Adaptive coarse grid",
    description: "Current compact leaf scale and refinement transitions on an exact scene slice.",
    source: "Published octree owner map and compact leaf records",
    mode: "resolution", axis: "z",
    legend: [
      { swatch: "#38adbd", label: "finest compact leaf" },
      { swatch: "#55a8ba", label: "intermediate dyadic leaf" },
      { swatch: "#152e7a", label: "coarsest compact leaf" },
      { swatch: "#ff05b8", label: "missing compact owner" },
    ],
  },
  {
    id: "sdf", figure: "FIG. 6/7", label: "Fine signed distance",
    description: "Factor-m φ values, the zero crossing, and the centered-gradient Eikonal residual.",
    source: "Published Section 5 fine-lattice φ samples",
    mode: "global-fine-phi", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#1973eb,#f5f5e6,#ed7829)", label: "liquid (−) · zero · air (+)" },
      { swatch: "#ffffff", label: "φ = 0 crossing" },
      { swatch: "#f505b8", label: "|∇φ|−1 residual" },
      { swatch: "#ff0610", label: "missing or rejected support" },
    ],
  },
  {
    id: "power-cells", figure: "FIG. 4", label: "Power cells",
    description: "Pressure sites and exact regular/transition power-cell classification.",
    source: "Compact leaves, topology descriptors, and generated power catalog",
    mode: "power-cells", axis: "volume",
    legend: [
      { swatch: "#159578", label: "regular Cartesian power cell" },
      { swatch: "#8c38c7", label: "transition power cell" },
      { swatch: "#f5ba1a", label: "pressure site" },
      { swatch: "#ff0303", label: "invalid descriptor or metric" },
    ],
  },
  {
    id: "power-faces", figure: "FIG. 4", label: "Power face geometry",
    description: "Primal face planes, dual links, stored normals, and boundary faces.",
    source: "Published generalized faces, centroids, normals, and incidences",
    mode: "power-faces", axis: "volume",
    legend: [
      { swatch: "#13cfe8", label: "power-face plane" },
      { swatch: "#8c38c7", label: "dual pressure-site link" },
      { swatch: "#f5ba1a", label: "stored face normal" },
      { swatch: "#ff00aa", label: "incomplete publication" },
    ],
  },
  {
    id: "sparse-pyramid", figure: "FIG. 5", label: "Sparse topology lifecycle",
    description: "Actual active and retired rebuild tiles; this does not pretend the current implementation has the paper's idealized ghost aliases.",
    source: "Live topology rebuild tile worklist",
    mode: "octree-lifecycle", axis: "volume",
    legend: [
      { swatch: "#0a193b", label: "unchanged tile" },
      { swatch: "#16cbdc", label: "active rebuild tile" },
      { swatch: "#ff6812", label: "retired tile" },
      { swatch: "#ff1738", label: "invalid lifecycle publication" },
    ],
  },
  {
    id: "pressure", figure: "§4", label: "Evaluated pressure",
    description: "Affine pressure potential dt·p/ρ reconstructed from the live leaf degrees of freedom.",
    source: "Current pressure leaf field",
    mode: "pressure", axis: "z",
    legend: [{ swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "low → high pressure potential" }],
  },
  {
    id: "velocity", figure: "§5", label: "Evaluated velocity",
    description: "Magnitude of the reconstructed projected velocity, including the extrapolated air band.",
    source: "Current solver velocity publication",
    mode: "speed", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "zero → live maximum speed" },
      { swatch: "#b8deda", label: "bright air · extrapolated band" },
    ],
  },
  {
    id: "projection", figure: "§4.1", label: "Pressure update Δu",
    description: "Magnitude of u after projection minus u before projection, normalized by live maximum speed.",
    source: "Current before/after projection fields",
    mode: "projection", axis: "z",
    legend: [{ swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "small → large |Δu|" }],
  },
  {
    id: "divergence", figure: "EQ. 2", label: "Divergence closure",
    description: "Post-projection divergence; the color scale saturates at |∇·u| Δt = 1.",
    source: "Current projected velocity field",
    mode: "divergence", axis: "z",
    legend: [{ swatch: "linear-gradient(90deg,#1548df,#f5f5f5,#e21a14)", label: "compression (−) · zero · expansion (+)" }],
  },
  {
    id: "extrapolation", figure: "§5", label: "Structured velocity",
    description: "Projected full-vector reconstruction over the direct six-family authority.",
    source: "Live structured rows, family slots, and projected CPT seeds",
    mode: "speed", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "zero → live projected speed" },
      { swatch: "#ff1738", label: "invalid structured publication" },
    ],
  },
  {
    id: "operator", figure: "§6.3", label: "Power operator",
    description: "Largest incident A/d coefficient with publication and reciprocity failures exposed.",
    source: "Published power Laplacian rows and generalized face graph",
    mode: "power-operator", axis: "volume",
    legend: [
      { swatch: "linear-gradient(90deg,#15489a,#18bf8c,#ed2d12)", label: "low → high incident coefficient" },
      { swatch: "#f5ba1a", label: "high diagonal or residual contribution" },
      { swatch: "#ff0303", label: "invalid or asymmetric operator" },
    ],
  },
] as const;

export function PerformancePanel() {
  const report = useDiagnosticsStore((state) => state.performanceReport);
  const reports = useDiagnosticsStore((state) => state.performanceReports);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const methodId = useMethodStore((state) => state.methodId);
  const instrumentationEnabled = usePerformanceInstrumentationStore((state) => state.enabled);
  const setInstrumentationEnabled = usePerformanceInstrumentationStore((state) => state.setEnabled);
  const runState = useRuntimeStore((state) => state.runState);
  const overlayMode = useUIStore((state) => state.gridOverlayMode);
  const overlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const overlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setOverlayMode = useUIStore((state) => state.setGridOverlayMode);
  const setOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const setOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const [windowSize, setWindowSize] = useState(30);
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
      cpu: { ...averagedLane(cpuSamples.traces, windowSize), held: cpuSamples.held },
      physics: { ...averagedLane(physicsSamples.traces, windowSize), held: physicsSamples.held },
      presentation: { ...averagedLane(presentationSamples.traces, windowSize), held: presentationSamples.held },
    };
  }, [reports, methodId, report.context, runState, windowSize]);
  const cpu = lanes.cpu.trace;
  const physics = lanes.physics.trace;
  const presentation = lanes.presentation.trace;
  const selectView = (view: PaperView) => {
    setOverlayMode(view.mode);
    setOverlayAxis(view.axis);
    if (view.axis === "volume") setOverlaySlice(0.42);
  };
  const selectedView = PAPER_VIEWS.find((view) => view.mode === overlayMode);
  const volumeCapable = isOctreeTechniqueOverlayMode(overlayMode) && overlayMode !== "global-fine-phi";
  const traces = [cpu, physics, presentation].filter((trace): trace is PerformanceTrace => trace !== undefined);
  const allExact = traces.length === 3 && traces.every(performanceTraceIsExact);
  const holdingPausedMeasurements = lanes.cpu.held || lanes.physics.held || lanes.presentation.held;
  const toggleInstrumentation = () => {
    const enabled = !instrumentationEnabled;
    setInstrumentationEnabled(enabled);
    useDiagnosticsStore.getState().set({
      performanceReport: emptyPerformanceReport,
      performanceReports: [],
    });
  };

  return <aside id="performance-panel" className="right-panel panel-scroll performance-panel performance-v2" aria-label="Performance and paper field observatory" data-testid="performance-panel" data-method={methodId} data-traces-exact={allExact}>
    <header className="trace-header">
      <div><span>POWER LIQUIDS OBSERVATORY</span><h2>Measured work + live fields</h2></div>
      <div className="trace-header-actions">
        <button
          type="button"
          className={`measurement-toggle ${instrumentationEnabled ? "active" : ""}`}
          role="switch"
          aria-checked={instrumentationEnabled}
          onClick={toggleInstrumentation}
          title="Disable timestamp queries, marker passes, trace readbacks, and measurement-only queue fences"
        >
          <span>MEASUREMENT LOAD</span><b>{instrumentationEnabled ? "ON" : "OFF"}</b><i />
        </button>
      </div>
    </header>

    {instrumentationEnabled ? <><section className="trace-summary">
      <div><small>CPU MAIN THREAD</small><strong>{cpu ? formatMs(cpu.total_ms) : "—"}</strong></div>
      <div><small>GPU PHYSICS</small><strong>{physics ? formatMs(physics.total_ms) : "—"}</strong></div>
      <div><small>GPU PRESENTATION</small><strong>{presentation ? formatMs(presentation.total_ms) : "—"}</strong></div>
      <label><small>AVERAGE</small><select value={windowSize} onChange={(event) => setWindowSize(Number(event.currentTarget.value))}><option value={1}>1 sample</option><option value={10}>10 samples</option><option value={30}>30 samples</option><option value={60}>60 samples</option></select></label>
    </section>

    <div className="trace-notice">
      CPU and GPU are independent ledgers and are never added together. Each lane is one adjacent-boundary partition; phase rows sum exactly to that lane&apos;s measured total.
    </div>
    {holdingPausedMeasurements && <div className="trace-notice">
      PAUSED · Holding the last completed measurements until a fresh sample for this view is available.
    </div>}

    <TraceLane trace={physics} title="GPU · POWER LIQUIDS ADVANCE" sampleCount={lanes.physics.sampleCount} />
    <TraceLane trace={presentation} title="GPU · PRESENTATION" sampleCount={lanes.presentation.sampleCount} />
    <TraceLane trace={cpu} title="CPU · MAIN-THREAD FRAME WORK" sampleCount={lanes.cpu.sampleCount} />
    </> : <div className="trace-disabled-notice">
      <strong>Running without measurement instrumentation</strong>
      <span>Timestamp queries, marker dispatches, forced encoder breaks, trace-buffer resolves/readbacks, and measurement-only queue fences are bypassed. Correctness synchronization remains active.</span>
    </div>}

    <section className="paper-observatory">
      <header><div><h3>Paper field observatory</h3><small>LIVE GPU PUBLICATIONS · NO FIELD READBACK</small></div><span>{methodId === "octree" ? "OCTREE AUTHORITY" : "SELECT OCTREE FOR FULL SET"}</span></header>
      <div className="paper-view-grid">
        {PAPER_VIEWS.map((view) => {
          const active = overlayMode === view.mode;
          return <button key={view.id} className={active ? "active" : ""} onClick={() => selectView(view)} aria-pressed={active}>
            <span>{view.figure}</span><strong>{view.label}</strong><small>{view.description}</small>
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
          <div className="paper-field-legend" aria-label={`${selectedView.label} legend`}>
            {selectedView.legend.map((entry) => <span key={entry.label}><i style={{ background: entry.swatch }} />{entry.label}</span>)}
          </div>
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
