"use client";

import { useRef, useState } from "react";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useRecordingStore } from "@/lib/stores/recording-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { requestManualGPUStop } from "@/lib/gpu-startup";
import { useSafeBrowserGPUBringup } from "@/lib/use-safe-browser-gpu-bringup";
import { planSceneRuntime } from "@/lib/scene-runtime";
import { resourceInteractionGates } from "@/lib/resource-readiness";


export function TransportBar() {
  const runState = useRuntimeStore((state) => state.runState);
  const setRunState = useRuntimeStore((state) => state.setRunState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const notice = useRuntimeStore((state) => state.notice);
  const noticeTone = useRuntimeStore((state) => state.noticeTone);
  const simRate = useRuntimeStore((state) => state.simRate);
  const maxDt = useSceneStore((state) => state.scene.numerics.maxDt_s);
  const fixedDt = useSceneStore((state) => state.scene.numerics.fixedDt_s);
  const rendererOnlyScene = useSceneStore((state) => !planSceneRuntime(state.scene).fluidSolver);
  const gpuLag = useDiagnosticsStore((state) => state.gpuInfo?.simulationLag_s);
  const resourceReadiness = useDiagnosticsStore((state) => state.resourceReadiness);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const methodId = useMethodStore((state) => state.methodId);
  const recordingStatus = useRecordingStore((state) => state.status);
  const recordingStart = useRecordingStore((state) => state.startedAtSimulation_s);
  const recording = useRecordingStore((state) => state.recording);
  const fileRef = useRef<HTMLInputElement>(null);
  const safeBringupPolicy = useSafeBrowserGPUBringup();
  const safeBringup = safeBringupPolicy === true;
  const browserPolicyPending = safeBringupPolicy === null;
  const browserSafetyLocked = safeBringupPolicy !== false;
  const [safeStepRequested, setSafeStepRequested] = useState(false);
  const webgpu = simulation.backend === "webgpu";
  const lagged = webgpu && gpuLag !== undefined && gpuLag > 2 * maxDt;
  const applyingGPUSettings = resourceReadiness.fluid.state === "preparing"
    && Boolean(resourceReadiness.fluid.activity?.operation);
  const interaction = resourceInteractionGates(resourceReadiness, !rendererOnlyScene);
  const initialSceneReady = methodId !== "octree" || (gpuInfo?.initialSparseAuthorityReady === true
    && gpuInfo?.initialRasterSurfaceReady === true);
  const transportLocked = rendererOnlyScene || (webgpu && (!interaction.transportInteractive || !initialSceneReady));
  const safeStepLocked = safeBringup && (safeStepRequested || (gpuInfo?.encodedSteps ?? 0) >= 1);
  const importScene = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    simulation.importScene(file.name, await file.text());
    event.target.value = "";
  };
  const toggleRecording = () => {
    if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
    else {
      setRunState("running");
      simulationRecording.start(simulationTime);
    }
  };
  return (
    <footer className="transport-bar">
      <div className="transport-controls">
        <button disabled={transportLocked || browserSafetyLocked} className="transport-main" onClick={() => setRunState(runState === "running" ? "paused" : "running")} aria-label={browserPolicyPending ? "Browser GPU safety policy is loading" : rendererOnlyScene ? "Fluid simulation is disabled for this renderer validation scene" : safeBringup ? "Continuous play is disabled during bounded GPU bring-up" : transportLocked ? "Simulation controls unlock after the initial GPU scene is ready" : runState === "running" ? "Pause simulation" : "Play simulation"}>{transportLocked || browserPolicyPending ? "…" : runState === "running" ? "Ⅱ" : "▶"}</button>
        <button disabled={browserPolicyPending || transportLocked || safeStepLocked} onClick={() => { if (safeBringup) setSafeStepRequested(true); simulation.singleStep(); }} aria-label={browserPolicyPending ? "Browser GPU safety policy is loading" : transportLocked ? "Single step unavailable until the initial GPU scene is ready" : safeStepLocked ? "The bounded browser GPU step has already been requested" : "Single fluid clock step"}>STEP</button>
        <button disabled={browserSafetyLocked} onClick={() => {
          if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
          simulation.reset();
        }}>RESET</button>
        <button
          className={`record-button${recordingStatus === "recording" ? " active" : ""}`}
          onClick={toggleRecording}
          disabled={recordingStatus === "processing" || transportLocked || browserSafetyLocked}
          aria-label={recordingStatus === "recording" ? "Stop simulation recording" : "Record simulation video"}
          data-testid="record-simulation"
        >{recordingStatus === "recording" ? "■ STOP" : recordingStatus === "processing" ? "WAIT" : "● REC"}</button>
        {safeBringup && <button type="button" className="stop-gpu-button" onClick={requestManualGPUStop}>STOP GPU</button>}
      </div>
      <div className="time-readout">
        <span>t</span><strong>{simulationTime.toFixed(4)}</strong><small>s</small>
        {simRate !== null && <small className="sim-rate" title={webgpu ? "Queue-confirmed simulated seconds completed per wall-clock second" : "Simulated seconds completed per wall-clock second"}>ACTUAL ×{simRate.toFixed(2)}</small>}
        {lagged && <small className="lag-chip" title="Simulation time currently admitted to the bounded GPU feed window.">GPU −{gpuLag.toFixed(1)} s</small>}
        {recordingStatus === "recording" && recordingStart !== null && <small className="recording-chip"><i />REC {(simulationTime - recordingStart).toFixed(2)} s</small>}
        {rendererOnlyScene
          ? <span className="continuous-run" title="This preset runs the live sparse scene renderer without fluid physics.">LIVE SVO · FLUID SOLVER DISABLED</span>
          : webgpu
          ? <span className="continuous-run" title={`Each admitted simulation advance is immediately followed by its presentation · double-buffered pairs · rigid step ${(fixedDt * 1000).toFixed(2)} ms · GPU step cap ${(maxDt * 1000).toFixed(2)} ms`}>SIM + RENDER LOCKSTEP · PRESENT ASAP</span>
          : <span className="continuous-run" title={`CPU reference simulation · present every browser animation frame · fixed step ${(fixedDt * 1000).toFixed(2)} ms`}>CPU REFERENCE · PRESENT ASAP</span>}
      </div>
      <div className="file-actions">
        <span className={`notice${noticeTone === "warn" ? " warn" : ""}`}>{applyingGPUSettings ? `GPU SETTINGS · ${resourceReadiness.fluid.activity?.operation ?? resourceReadiness.fluid.label}` : notice}</span>
        {recording && recordingStatus !== "recording" && <button onClick={() => simulationRecording.open()}>Playback</button>}
        <button disabled={browserSafetyLocked} onClick={() => { if (!simulation.loadLocalScene()) fileRef.current?.click(); }}>Load</button>
        <button disabled={browserSafetyLocked} onClick={() => fileRef.current?.click()}>Import</button>
        <input ref={fileRef} type="file" accept="application/json,.json" onChange={importScene} hidden />
      </div>
    </footer>
  );
}
