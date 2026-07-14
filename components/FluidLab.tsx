"use client";

import { useEffect, useMemo } from "react";
import { runShellValidation } from "@/lib/validation";
import { getMethod } from "@/lib/methods";
import { defaultCamera } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { WebGPUViewport } from "./WebGPUViewport";
import { ScenePanel } from "./ScenePanel";
import { SceneConfigPopover } from "./SceneConfigPopover";
import { MethodPanel } from "./MethodPanel";
import { RigidBodyTray } from "./RigidBodyTray";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { PerformanceDrawer } from "./PerformanceDrawer";
import { ValidationPanel } from "./ValidationPanel";
import { TransportBar } from "./TransportBar";
import { RecordingPlaybackModal } from "./RecordingPlaybackModal";

export function FluidLab() {
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const methodId = useMethodStore((state) => state.methodId);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const view = useUIStore((state) => state.view);
  const setView = useUIStore((state) => state.setView);
  const setCamera = useUIStore((state) => state.setCamera);
  const performanceOpen = useUIStore((state) => state.performanceOpen);
  const validationOpen = useUIStore((state) => state.validationOpen);
  const setValidationOpen = useUIStore((state) => state.setValidationOpen);
  const diagnosticsOpen = useUIStore((state) => state.diagnosticsOpen);
  const setDiagnosticsOpen = useUIStore((state) => state.setDiagnosticsOpen);
  const gridOverlayAxis = useUIStore((state) => state.gridOverlayAxis);
  const setGridOverlayAxis = useUIStore((state) => state.setGridOverlayAxis);
  const gridOverlaySlice = useUIStore((state) => state.gridOverlaySlice);
  const setGridOverlaySlice = useUIStore((state) => state.setGridOverlaySlice);
  const fluidState = useDiagnosticsStore((state) => state.fluidState);
  const validationResults = useMemo(() => runShellValidation(), []);
  const method = getMethod(methodId);
  const backend = method.backend === "cpu" ? "cpu-reference" : "webgpu";
  const scientific = view === "scientific";
  const healthFlags = backend === "webgpu"
    ? [...(gpuInfo?.stabilityFlags ?? []), ...(gpuInfo?.nonFiniteCount ? ["non-finite-values"] : [])]
    : [...(fluidState?.nanCount ? ["non-finite-values"] : []), ...(fluidState && !fluidState.pressureConverged ? ["pressure-not-converged"] : [])];

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
    <main className="lab-shell" data-run-state={runState} data-solver-mode="eulerian" data-simulation-time={simulationTime.toFixed(6)} data-body-count={bodies.length} data-diagnostics-open={diagnosticsOpen}>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">FL</span><div><strong>Fluid Lab</strong><small>WEBGPU CFD WORKBENCH</small></div></div>
        <div className="solver-identity">{method.label}</div>
        <div className="top-actions">
          <button className="quiet-button" onClick={() => setValidationOpen(true)}><span className={`status-dot ${validationResults.every((result) => result.passed) ? "online" : "warning"}`} />Validation</button>
          <button className="quiet-button" title="Download the scene description — configuration only, reloadable via Import" onClick={() => simulation.saveScene()}>Save scene</button>
          <button className="primary-button" title="Download the run manifest — metrics, diagnostics, and performance history for this run" onClick={() => simulation.exportMetrics()}>Export run</button>
        </div>
      </header>

      <aside className="left-panel panel-scroll">
        <ScenePanel />
        <MethodPanel />
      </aside>

      <section className="viewport-shell">
        <WebGPUViewport />
        <div className="viewport-topline">
          <div className="topline-left">
            {scientific && <div className={`gpu-badge state-${gpuStatus.state}`}><span className={`status-dot ${gpuStatus.state === "ready" ? "online" : "warning"}`} /><strong>{gpuStatus.state === "ready" ? "WEBGPU" : gpuStatus.state.toUpperCase()}</strong><span>{gpuStatus.label}</span></div>}
            {scientific && runState === "running" && gpuStatus.state === "ready" && (
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
          <div className="topline-right">
            {view === "scientific" && <div className="grid-overlay-cluster" title="Overlay the solver grid on a cross-section slice (tall cells teal, regular cells outlined with sample dots)">
              {gridOverlayAxis !== "off" && <>
                <input type="range" min={0} max={1} step={0.005} value={gridOverlaySlice} onChange={(event) => setGridOverlaySlice(Number(event.target.value))} aria-label="Grid slice position" />
                <span className="slice-readout">{Math.round(gridOverlaySlice * 100)}%</span>
              </>}
              <div className="segmented">
                <button className={gridOverlayAxis === "off" ? "active" : ""} onClick={() => setGridOverlayAxis("off")}>Grid off</button>
                <button className={gridOverlayAxis === "z" ? "active" : ""} onClick={() => setGridOverlayAxis("z")}>Z slice</button>
                <button className={gridOverlayAxis === "x" ? "active" : ""} onClick={() => setGridOverlayAxis("x")}>X slice</button>
              </div>
            </div>}
            <div className="segmented"><button className={view === "scientific" ? "active" : ""} onClick={() => setView("scientific")}>Scientific</button><button className={view === "presentation" ? "active" : ""} onClick={() => setView("presentation")}>Presentation</button></div>
          </div>
        </div>
        {scientific && <RigidBodyTray />}
        {scientific && <div className="physics-stage-badge"><strong>{method.badge}</strong><span>{backend === "webgpu" ? `${method.description}` : "CPU validation oracle active"}</span><small>{backend === "webgpu" ? `${gpuInfo?.cellCount.toLocaleString() ?? "…"} allocated samples · f32 · ${gpuInfo?.pressureSolver ?? `${gpuInfo?.pressureIterations ?? "…"} Jacobi`}` : "MAC · binary64 · PCG"}</small></div>}
        {view === "scientific" && gridOverlayAxis !== "off" && (() => {
          const gridKind = method.backend === "cpu" ? "uniform" : gpuInfo?.gridKind ?? "uniform";
          const tall = gridKind !== "uniform";
          return <div className="grid-legend" data-testid="grid-legend">
            <strong>{gridKind === "restricted-tall-cell" ? "TALL-CELL GRID" : gridKind === "adaptive-optical-layer" ? "ADAPTIVE-LAYER GRID" : "UNIFORM GRID"} · {gridOverlayAxis.toUpperCase()} SLICE</strong>
            {tall && <span><i className="sw sw-tall" />tall cell · liquid (one per column)</span>}
            {tall && <span><i className="sw sw-tall-dry" />tall cell · air</span>}
            <span><i className="sw sw-wet" />{tall ? "regular cell · liquid" : "cell · liquid"}</span>
            <span><i className="sw sw-air" />{tall ? "regular cell · air" : "cell · air"}</span>
            {tall && <span><i className="sw sw-outside" />above band · not stored</span>}
            <span><i className="sw sw-dot" />stored samples (zoom in)</span>
            <small>drag the bright top edge to sweep the slice</small>
          </div>;
        })()}
        {scientific && <div className="axis-widget"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span></div>}
        <div className="camera-toolbar" aria-label="Camera controls">
          <button onClick={() => setPresetCamera("reset")}>Reset</button><button onClick={() => setPresetCamera("front")}>Front</button><button onClick={() => setPresetCamera("side")}>Side</button><button onClick={() => setPresetCamera("top")}>Top</button>
          {scientific && <span>drag a shape from the tray to add · drag body to move · drag to orbit · ⇧ drag pan · wheel zoom</span>}
        </div>
        {scientific && <button className={`diagnostics-toggle${diagnosticsOpen ? " active" : ""}`} onClick={() => setDiagnosticsOpen(!diagnosticsOpen)} aria-expanded={diagnosticsOpen} title="Toggle live diagnostics">{diagnosticsOpen ? "›" : "‹"} DIAG</button>}
        {gpuStatus.state === "initializing" && <div className="gpu-fallback gpu-initializing"><strong>Initializing WebGPU…</strong><p>{gpuStatus.label}</p><small>The first frame appears once the adapter and solver pipelines are ready.</small></div>}
        {gpuStatus.state === "unavailable" && <div className="gpu-fallback"><strong>3D renderer unavailable</strong><p>{gpuStatus.label}</p><small>The scene editor, serialization, and CPU validation remain available.</small></div>}
      </section>

      {diagnosticsOpen && <DiagnosticsPanel />}
      <TransportBar />

      {performanceOpen && <PerformanceDrawer />}
      {validationOpen && <ValidationPanel results={validationResults} />}
      <RecordingPlaybackModal />
      <SceneConfigPopover />
    </main>
  );
}
