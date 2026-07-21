"use client";

import { useEffect, useLayoutEffect, useState } from "react";
import { getMethod } from "@/lib/methods";
import { defaultCamera } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { startQueryStateSync } from "@/lib/url-state";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { WebGPUViewport } from "./WebGPUViewport";
import { ScenePanel } from "./ScenePanel";
import { SceneConfigPopover } from "./SceneConfigPopover";
import { MethodPanel } from "./MethodPanel";
import { RigidBodyPanel } from "./RigidBodyTray";
import { VisualPanel } from "./VisualPanel";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { PerformancePanel } from "./PerformancePanel";
import { TransportBar } from "./TransportBar";
import { RecordingPlaybackModal } from "./RecordingPlaybackModal";
import type { GPUStatus } from "@/lib/webgpu-renderer";
import { getEnvironmentPreset } from "@/lib/environments";
import { getScenePreset } from "@/lib/scenes";
import { useSceneStore } from "@/lib/stores/scene-store";
import { requestManualGPUStart } from "@/lib/gpu-startup";
import { useSafeBrowserGPUBringup } from "@/lib/use-safe-browser-gpu-bringup";

function GPUInitializationPanel({ status }: { status: Extract<GPUStatus, { state: "initializing" }> }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => { const timer = window.setInterval(() => setNow(performance.now()), 100); return () => window.clearInterval(timer); }, []);
  const completed = status.completed ?? 0, total = status.total ?? 0;
  const elapsed_s = Math.max(0, now - (status.startedAt_ms ?? now)) / 1000;
  const rebuilding = status.kind === "rebuild";
  const phaseCopy: Record<string, string> = {
    planning: "Determining which GPU resources and programs are invalidated.",
    allocation: "Allocating replacement textures and buffers. This stage can briefly occupy the browser's GPU process.",
    "solver-pipelines": "Compiling simulation programs. Input may be delayed while the graphics driver finishes an individual program.",
    "adaptive-topology": "Compiling adaptive topology and pressure programs.",
    "secondary-particles": "Preparing secondary-liquid programs.",
    upload: "Uploading initial simulation fields.",
    warmup: "Submitting the initial scene and waiting for GPU completion.",
    attach: "Atomically attaching the warmed replacement.",
  };
  return <div className={`${rebuilding ? "gpu-build-card" : "gpu-fallback"} gpu-initializing`} role="status" aria-live="polite" aria-busy="true">
    <div className="gpu-build-heading"><i aria-hidden="true" /><strong>{rebuilding ? "Applying simulation settings" : "Starting WebGPU"}</strong></div>
    {status.operation && <p className="gpu-build-operation">{status.operation}</p>}
    <p>{status.label}</p>
    <progress max={Math.max(1, total)} {...(total > 0 ? { value: Math.min(completed, total) } : {})} aria-label="GPU initialization progress" />
    <div className="gpu-progress-summary"><span>{total > 0 ? `${completed} / ${total} stages` : "Planning work…"}</span><span>{elapsed_s.toFixed(1)} s</span></div>
    <p className="gpu-stage-explanation">{phaseCopy[status.phase ?? "planning"] ?? "Preparing the replacement GPU state."}</p>
    {rebuilding && <small>{status.retainingPrevious ? "The previous GPU frame remains visible. " : "The viewport will resume when the replacement is ready. "}Simulation is paused at t = 0 until attachment; another structural change will supersede this build.</small>}
  </div>;
}

export function FluidLab() {
  const safeBringup = useSafeBrowserGPUBringup() === true;
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const methodId = useMethodStore((state) => state.methodId);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const setCamera = useUIStore((state) => state.setCamera);
  const diagnosticsOpen = useUIStore((state) => state.diagnosticsOpen);
  const setDiagnosticsOpen = useUIStore((state) => state.setDiagnosticsOpen);
  const rightPanel = useUIStore((state) => state.rightPanel);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const presetId = useSceneStore((state) => state.presetId);
  const fluidState = useDiagnosticsStore((state) => state.fluidState);
  const method = getMethod(methodId);
  const backend = method.backend === "cpu" ? "cpu-reference" : "webgpu";
  const environment = getEnvironmentPreset(getScenePreset(presetId).background);
  const healthFlags = backend === "webgpu"
    ? [...(gpuInfo?.stabilityFlags ?? []), ...(gpuInfo?.nonFiniteCount ? ["non-finite-values"] : [])]
    : [...(fluidState?.nanCount ? ["non-finite-values"] : []), ...(fluidState && !fluidState.pressureConverged ? ["pressure-not-converged"] : [])];

  useLayoutEffect(() => startQueryStateSync(() => simulation.reset()), []);

  useEffect(() => {
    let frame = 0;
    const tick = (now: number) => { simulation.tick(now); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const setPresetCamera = (preset: "front" | "side" | "top" | "reset") => {
    if (preset === "reset") setCamera(defaultCamera);
    else if (preset === "front") setCamera({ ...defaultCamera, azimuth_rad: 0, elevation_rad: 0.08 });
    else if (preset === "side") setCamera({ ...defaultCamera, azimuth_rad: Math.PI / 2, elevation_rad: 0.08 });
    else setCamera({ ...defaultCamera, azimuth_rad: 0, elevation_rad: 1.34, distance_m: 2.25 });
  };

  return (
    <main className="lab-shell" data-run-state={runState} data-solver-mode="eulerian" data-simulation-time={simulationTime.toFixed(6)} data-body-count={bodies.length} data-right-panel-open={Boolean(rightPanel)} data-right-panel={rightPanel ?? "closed"}>
      <aside className="left-panel panel-scroll">
        <div className="brand"><span className="brand-mark">FL</span><div><strong>Fluid Lab</strong><small>WEBGPU CFD WORKBENCH</small></div></div>
        <ScenePanel />
        <MethodPanel />
      </aside>

      <section className="viewport-shell" aria-busy={gpuStatus.state === "initializing"} data-gpu-transition={gpuStatus.state === "initializing" ? gpuStatus.kind ?? "startup" : gpuStatus.state}>
        <WebGPUViewport />
        <div className="viewport-topline">
          <div className="topline-left">
            <div className={`gpu-badge state-${gpuStatus.state}`}><span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} /><strong>{gpuStatus.state === "ready" ? "WEBGPU" : gpuStatus.state.toUpperCase()}</strong><span>{gpuStatus.label}</span></div>
            {runState === "running" && gpuStatus.state === "ready" && (
              <button
                className={`health-chip${healthFlags.length ? " alert" : ""}`}
                onClick={() => setDiagnosticsOpen(true)}
                title={healthFlags.length ? "Instrumented stability gates are firing — click for live diagnostics" : "All instrumented stability gates clear — click for live diagnostics"}
                data-testid="health-chip"
              >
                <span className={`status-dot ${healthFlags.length ? "warning" : "online"}`} />
                <strong>{healthFlags.length ? "ALERT" : "STABLE"}</strong>
                {healthFlags.length > 0 && <span>{healthFlags.join(" · ")}</span>}
              </button>
            )}
          </div>
          <div className="environment-chip" title={`${environment.name} · fixed by the selected scene`}>
            <span aria-hidden="true">{environment.swatch.map((color) => <i key={color} style={{ background: color }} />)}</span>
            <small>BACKGROUND</small><strong>{environment.shortName}</strong>
          </div>
        </div>
        <div className="physics-stage-badge"><strong>{method.badge}</strong><span>{backend === "webgpu" ? `${method.description}` : "CPU validation oracle active"}</span><small>{backend === "webgpu" ? `${gpuInfo?.cellCount.toLocaleString() ?? "…"} allocated · ${gpuInfo?.activeSampleCount?.toLocaleString() ?? "…"} ${gpuInfo?.gridKind === "octree" ? "live pressure rows" : "active samples"} · f32 · ${gpuInfo?.pressureSolver ?? `${gpuInfo?.pressureIterations ?? "…"} Jacobi`}` : "MAC · binary64 · PCG"}</small></div>
        <div className="axis-widget"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span></div>
        <div className="camera-toolbar" aria-label="Camera controls">
          <button onClick={() => setPresetCamera("reset")}>Reset</button><button onClick={() => setPresetCamera("front")}>Front</button><button onClick={() => setPresetCamera("side")}>Side</button><button onClick={() => setPresetCamera("top")}>Top</button>
          <span>drag body to move · drag to orbit · ⇧ drag pan · wheel zoom</span>
        </div>
        <nav className="utility-panel-tabs" aria-label="Viewport panels">
          <button className={rightPanel === "visual" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "visual" ? null : "visual")} aria-expanded={rightPanel === "visual"} title="Render and debug controls">RENDER</button>
          <button className={rightPanel === "bodies" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "bodies" ? null : "bodies")} aria-expanded={rightPanel === "bodies"} title="Rigid body controls">BODIES</button>
          <button className={diagnosticsOpen ? "active" : ""} onClick={() => setDiagnosticsOpen(!diagnosticsOpen)} aria-expanded={diagnosticsOpen} title="Live diagnostics">DIAG</button>
          <button className={rightPanel === "performance" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "performance" ? null : "performance")} aria-expanded={rightPanel === "performance"} aria-controls="performance-panel" title="Live performance profiler">PERF</button>
        </nav>
        {gpuStatus.state === "initializing" && <GPUInitializationPanel status={gpuStatus} />}
        {gpuStatus.state === "manual" && <div className="gpu-fallback gpu-manual-start" role="status">
          <strong>WebGPU startup paused for safety</strong>
          <p>{safeBringup
            ? "Bounded bring-up permits the authored 384-column dam break, one STEP, then an explicit STOP GPU. Close every Dawn process first."
            : "The dam-break GPU workload will not start until you explicitly allow it."}</p>
          <button type="button" onClick={requestManualGPUStart}>START WEBGPU</button>
          <small>{safeBringup
            ? "This browser can exclude other Fluid Lab tabs, but cannot observe Dawn's local filesystem lease."
            : <>Use <code>gpu=off</code> for UI-only inspection or <code>gpu=on</code> to restore automatic startup.</>}</small>
        </div>}
        {gpuStatus.state === "unavailable" && <div className="gpu-fallback"><strong>3D renderer unavailable</strong><p>{gpuStatus.label}</p><small>The scene editor, serialization, and CPU validation remain available.</small></div>}
      </section>

      {rightPanel === "visual" && <VisualPanel />}
      {rightPanel === "bodies" && <RigidBodyPanel />}
      {diagnosticsOpen && <DiagnosticsPanel />}
      {rightPanel === "performance" && <PerformancePanel />}
      <TransportBar />

      <RecordingPlaybackModal />
      <SceneConfigPopover />
    </main>
  );
}
