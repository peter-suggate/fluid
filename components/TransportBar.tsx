"use client";

import { useRef, useState } from "react";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useRecordingStore } from "@/lib/stores/recording-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";

function CoupledTimestepSlider({ fixedDt, maxDt, onCommit }: { fixedDt: number; maxDt: number; onCommit: (value_ms: number) => void }) {
  const ratio = Math.max(1, Math.round(maxDt / fixedDt));
  const [draft_ms, setDraft_ms] = useState(fixedDt * 1000);
  return <label title="Adjust the simulation timestep; the GPU cap follows at the locked integer ratio"><span>STEP</span><input type="range" min="0.5" max="33.5" step="0.5" value={draft_ms} onChange={(event) => setDraft_ms(Number(event.currentTarget.value))} onPointerUp={() => onCommit(draft_ms)} onKeyUp={(event) => { if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) onCommit(Number(event.currentTarget.value)); }} onBlur={() => onCommit(draft_ms)} aria-label="Coupled simulation timestep in milliseconds" /><output>{draft_ms.toFixed(1)} ms</output><small>GPU {(draft_ms * ratio).toFixed(1)} ms · LOCK {ratio}×</small></label>;
}

export function TransportBar() {
  const runState = useRuntimeStore((state) => state.runState);
  const setRunState = useRuntimeStore((state) => state.setRunState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const notice = useRuntimeStore((state) => state.notice);
  const noticeTone = useRuntimeStore((state) => state.noticeTone);
  const simRate = useRuntimeStore((state) => state.simRate);
  const maxDt = useSceneStore((state) => state.scene.numerics.maxDt_s);
  const fixedDt = useSceneStore((state) => state.scene.numerics.fixedDt_s);
  const patchNumerics = useSceneStore((state) => state.patchNumerics);
  const gpuLag = useDiagnosticsStore((state) => state.gpuInfo?.simulationLag_s);
  const recordingStatus = useRecordingStore((state) => state.status);
  const recordingStart = useRecordingStore((state) => state.startedAtSimulation_s);
  const recording = useRecordingStore((state) => state.recording);
  const fileRef = useRef<HTMLInputElement>(null);
  const lagged = simulation.backend === "webgpu" && gpuLag !== undefined && gpuLag > 2 * maxDt;
  const stepRatio = Math.max(1, Math.round(maxDt / fixedDt));
  const commitDt = (raw_ms: number) => {
    const seconds = Math.max(0.0005, raw_ms / 1000);
    if (!Number.isFinite(seconds)) return;
    if (Math.abs(seconds - fixedDt) < 1e-9) return;
    const wasRunning = useRuntimeStore.getState().runState === "running";
    patchNumerics({ fixedDt_s: seconds, maxDt_s: seconds * stepRatio });
    simulation.applyAndResetFluid();
    if (wasRunning) useRuntimeStore.getState().setRunState("running");
  };
  const importScene = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    simulation.importScene(file.name, await file.text());
    event.target.value = "";
  };
  const toggleRecording = () => {
    if (recordingStatus === "recording") simulationRecording.stop(simulation.time());
    else {
      setRunState("running");
      simulationRecording.start(simulation.time());
    }
  };
  return (
    <footer className="transport-bar">
      <div className="transport-controls">
        <button className="transport-main" onClick={() => setRunState(runState === "running" ? "paused" : "running")} aria-label={runState === "running" ? "Pause simulation" : "Play simulation"}>{runState === "running" ? "Ⅱ" : "▶"}</button>
        <button onClick={() => simulation.singleStep()} aria-label="Single fluid clock step">STEP</button>
        <button onClick={() => {
          if (recordingStatus === "recording") simulationRecording.stop(simulation.time());
          simulation.reset();
        }}>RESET</button>
        <button
          className={`record-button${recordingStatus === "recording" ? " active" : ""}`}
          onClick={toggleRecording}
          disabled={recordingStatus === "processing"}
          aria-label={recordingStatus === "recording" ? "Stop simulation recording" : "Record simulation video"}
          data-testid="record-simulation"
        >{recordingStatus === "recording" ? "■ STOP" : recordingStatus === "processing" ? "WAIT" : "● REC"}</button>
      </div>
      <div className="time-readout">
        <span>t</span><strong>{simulationTime.toFixed(4)}</strong><small>s</small>
        {simRate !== null && <small className="sim-rate" title="Simulated seconds per wall-clock second">×{simRate.toFixed(2)}</small>}
        {lagged && <small className="lag-chip" title="The GPU solve is behind the transport clock. Uncoupled tall-cell scenes batch up to one display interval; coupled scenes retain one-step impulse ordering. RESET to resynchronize.">GPU −{gpuLag.toFixed(1)} s</small>}
        {recordingStatus === "recording" && recordingStart !== null && <small className="recording-chip"><i />REC {(simulationTime - recordingStart).toFixed(2)} s</small>}
        <div className="transport-timing" aria-label="Simulation timestep controls">
          <CoupledTimestepSlider key={`${fixedDt}-${maxDt}`} fixedDt={fixedDt} maxDt={maxDt} onCommit={commitDt} />
        </div>
        <span className="continuous-run">CONTINUOUS · ∞</span>
      </div>
      <div className="file-actions">
        <span className={`notice${noticeTone === "warn" ? " warn" : ""}`}>{notice}</span>
        {recording && recordingStatus !== "recording" && <button onClick={() => simulationRecording.open()}>Playback</button>}
        <button onClick={() => { if (!simulation.loadLocalScene()) fileRef.current?.click(); }}>Load</button>
        <button onClick={() => fileRef.current?.click()}>Import</button>
        <input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden />
      </div>
    </footer>
  );
}
