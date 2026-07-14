"use client";

import { useRef } from "react";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

export function TransportBar() {
  const runState = useRuntimeStore((state) => state.runState);
  const setRunState = useRuntimeStore((state) => state.setRunState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const notice = useRuntimeStore((state) => state.notice);
  const noticeTone = useRuntimeStore((state) => state.noticeTone);
  const simRate = useRuntimeStore((state) => state.simRate);
  const maxDt = useSceneStore((state) => state.scene.numerics.maxDt_s);
  const gpuLag = useDiagnosticsStore((state) => state.gpuInfo?.simulationLag_s);
  const performanceOpen = useUIStore((state) => state.performanceOpen);
  const setPerformanceOpen = useUIStore((state) => state.setPerformanceOpen);
  const fileRef = useRef<HTMLInputElement>(null);
  const lagged = simulation.backend === "webgpu" && gpuLag !== undefined && gpuLag > 2 * maxDt;
  const importScene = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    simulation.importScene(file.name, await file.text());
    event.target.value = "";
  };
  return (
    <footer className="transport-bar">
      <div className="transport-controls">
        <button className="transport-main" onClick={() => setRunState(runState === "running" ? "paused" : "running")} aria-label={runState === "running" ? "Pause simulation" : "Play simulation"}>{runState === "running" ? "Ⅱ" : "▶"}</button>
        <button onClick={() => simulation.singleStep()} aria-label="Single fluid clock step">STEP</button>
        <button onClick={() => simulation.reset()}>RESET</button>
        <button className={performanceOpen ? "active" : ""} onClick={() => setPerformanceOpen(!performanceOpen)} aria-expanded={performanceOpen} aria-controls="performance-drawer">PERF</button>
      </div>
      <div className="time-readout">
        <span>t</span><strong>{simulationTime.toFixed(4)}</strong><small>s</small>
        {simRate !== null && <small className="sim-rate" title="Simulated seconds per wall-clock second">×{simRate.toFixed(2)}</small>}
        {lagged && <small className="lag-chip" title="The GPU solve is behind the transport clock; it advances at most one max-dt step per frame. RESET to resynchronize.">GPU −{gpuLag.toFixed(1)} s</small>}
        <span className="continuous-run">CONTINUOUS · ∞</span>
      </div>
      <div className="file-actions">
        <span className={`notice${noticeTone === "warn" ? " warn" : ""}`}>{notice}</span>
        <button onClick={() => { if (!simulation.loadLocalScene()) fileRef.current?.click(); }}>Load</button>
        <button onClick={() => fileRef.current?.click()}>Import</button>
        <input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden />
      </div>
    </footer>
  );
}
