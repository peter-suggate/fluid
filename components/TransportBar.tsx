"use client";

import { useRef, useState } from "react";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useRecordingStore } from "@/lib/stores/recording-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

function TimingSlider({ label, unit, value, min, max, step, integer = false, detail, onCommit }: { label: string; unit: string; value: number; min: number; max: number; step: number; integer?: boolean; detail: (value: number) => string; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState(value);
  const [entry, setEntry] = useState(String(value));
  const normalize = (raw: number) => Math.min(max, Math.max(min, integer ? Math.round(raw) : raw));
  const commit = (raw: number) => {
    if (!Number.isFinite(raw)) { setEntry(String(draft)); return; }
    const next = normalize(raw);
    setDraft(next);
    setEntry(String(next));
    onCommit(next);
  };
  const updateFromRange = (raw: number) => { const next = normalize(raw); setDraft(next); setEntry(String(next)); };
  return <label title={`Adjust ${label.toLowerCase()}`}><span>{label}</span><input type="range" min={min} max={max} step={step} value={draft} onChange={(event) => updateFromRange(Number(event.currentTarget.value))} onPointerUp={() => commit(draft)} onKeyUp={(event) => { if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) commit(Number(event.currentTarget.value)); }} aria-label={`${label} slider`} /><span className="timing-entry"><input type="number" min={min} max={max} step={step} inputMode={integer ? "numeric" : "decimal"} value={entry} onChange={(event) => { const raw = event.currentTarget.value; setEntry(raw); const next = Number(raw); if (raw !== "" && Number.isFinite(next)) setDraft(normalize(next)); }} onBlur={() => commit(Number(entry))} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); commit(Number(entry)); } else if (event.key === "Escape") { event.preventDefault(); setEntry(String(value)); setDraft(value); } }} aria-label={`${label} exact value`} /><b>{unit}</b></span><small>{detail(draft)}</small></label>;
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
  const targetFps = useUIStore((state) => state.targetFps);
  const gpuLag = useDiagnosticsStore((state) => state.gpuInfo?.simulationLag_s);
  const recordingStatus = useRecordingStore((state) => state.status);
  const recordingStart = useRecordingStore((state) => state.startedAtSimulation_s);
  const recording = useRecordingStore((state) => state.recording);
  const fileRef = useRef<HTMLInputElement>(null);
  const lagged = simulation.backend === "webgpu" && gpuLag !== undefined && gpuLag > 2 * maxDt;
  const baseRate_hz = 1 / fixedDt;
  const gpuCpuMultiplier = Math.max(1, Math.round(maxDt / fixedDt));
  const commitNumerics = (patch: Parameters<typeof patchNumerics>[0]) => {
    const wasRunning = useRuntimeStore.getState().runState === "running";
    patchNumerics(patch);
    simulation.applyAndResetFluid();
    if (wasRunning) useRuntimeStore.getState().setRunState("running");
  };
  const commitBaseRate = (raw_hz: number) => {
    const rate_hz = Math.max(1, raw_hz);
    const seconds = 1 / rate_hz;
    if (!Number.isFinite(seconds)) return;
    if (Math.abs(seconds - fixedDt) < 1e-9) return;
    commitNumerics({ fixedDt_s: seconds, maxDt_s: seconds * gpuCpuMultiplier });
  };
  const commitMultiplier = (rawMultiplier: number) => {
    const multiplier = Math.max(1, Math.round(rawMultiplier));
    if (!Number.isFinite(multiplier)) return;
    if (Math.abs(multiplier - gpuCpuMultiplier) < 1e-9) return;
    commitNumerics({ maxDt_s: fixedDt * multiplier });
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
          <TimingSlider key={`base-${fixedDt}`} label="BASE RATE" unit="Hz" value={baseRate_hz} min={Math.min(30, baseRate_hz)} max={Math.max(2000, baseRate_hz)} step={0.01} detail={(value) => `${(1000 / value).toFixed(2)} ms`} onCommit={commitBaseRate} />
          <TimingSlider key={`ratio-${fixedDt}-${maxDt}`} label="GPU / CPU" unit="×" value={gpuCpuMultiplier} min={1} max={Math.max(32, gpuCpuMultiplier)} step={1} integer detail={(value) => `GPU ${(fixedDt * value * 1000).toFixed(1)} ms`} onCommit={commitMultiplier} />
        </div>
        <span className="continuous-run" title={`${(1000 / targetFps).toFixed(2)} ms presentation interval · ${(fixedDt * 1000).toFixed(2)} ms numerical substep`}>REALTIME ×1 · {targetFps} FPS</span>
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
