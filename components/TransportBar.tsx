"use client";

import { useEffect, useState } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { simulationRecording } from "../lib/core/simulation/recording";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useRecordingStore } from "../lib/core/stores/recording-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useMethodStore } from "../lib/core/stores/method-store";
import { requestManualGPUStop } from "../lib/core/gpu-startup";
import { useSafeBrowserGPUBringup } from "../lib/core/use-safe-browser-gpu-bringup";
import { planSceneRuntime } from "../lib/core/scene-runtime";
import { resourceActivitiesFor, resourceInteractionGates } from "../lib/core/resource-readiness";
import { requiresFencedInitialRasterPresentation } from "../lib/core/gpu-t0-presentation";

/** How long the pointer has to be still before the cluster recedes. */
const POINTER_IDLE_MS = 2600;

/**
 * Whether the pointer has been still long enough for chrome to get out of the
 * way.
 *
 * Window-level rather than per-element because the statement is about the
 * reader, not about this control: a pointer moving anywhere over the scene means
 * someone is working, and the transport should be legible before they reach for
 * it. Keyboard activity counts too — a reader driving the studio from the
 * keyboard is not idle, and fading the only readout of simulation time out from
 * under them would be the same bug in a different input device.
 */
function usePointerIdle(delay_ms = POINTER_IDLE_MS) {
  const [idle, setIdle] = useState(false);
  useEffect(() => {
    let timer = 0;
    const wake = () => {
      setIdle(false);
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setIdle(true), delay_ms);
    };
    wake();
    for (const type of ["pointermove", "pointerdown", "keydown", "wheel"] as const) {
      window.addEventListener(type, wake, { passive: true });
    }
    return () => {
      window.clearTimeout(timer);
      for (const type of ["pointermove", "pointerdown", "keydown", "wheel"] as const) {
        window.removeEventListener(type, wake);
      }
    };
  }, [delay_ms]);
  return idle;
}

/**
 * The clock, and the two gestures that move it.
 *
 * This was a full-width docked footer with three columns: transport, a step-size
 * slider with an ms entry, a rate chip, a lag chip, a lockstep label, and file
 * actions. All of that cost the viewport a permanent 58 px row in order to state
 * things that are either constant (the lockstep contract), diagnostic (the rate
 * and lag chips, which belong beside the timings that explain them), or
 * authored once (the step size, which is now a pipeline-overlay parameter).
 *
 * What is left is what a reader reaches for without being told to: run, advance
 * one step, record, and what time it is. It floats on the scene and recedes when
 * the pointer stops — but only visually. It stays in the DOM, stays focusable,
 * and comes fully back the moment anything inside it takes focus, because a
 * control that disappears from the tab order is a control that is gone.
 */
export function TransportBar() {
  const runState = useRuntimeStore((state) => state.runState);
  const setRunState = useRuntimeStore((state) => state.setRunState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const notice = useRuntimeStore((state) => state.notice);
  const noticeTone = useRuntimeStore((state) => state.noticeTone);
  const rendererOnlyScene = useSceneStore((state) => !planSceneRuntime(state.scene).fluidSolver);
  const resourceReadiness = useDiagnosticsStore((state) => state.resourceReadiness);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const methodId = useMethodStore((state) => state.methodId);
  const recordingStatus = useRecordingStore((state) => state.status);
  const recordingStart = useRecordingStore((state) => state.startedAtSimulation_s);
  const recording = useRecordingStore((state) => state.recording);
  const safeBringupPolicy = useSafeBrowserGPUBringup();
  const safeBringup = safeBringupPolicy === true;
  const browserPolicyPending = safeBringupPolicy === null;
  const browserSafetyLocked = safeBringupPolicy !== false;
  const [safeStepRequested, setSafeStepRequested] = useState(false);
  const idle = usePointerIdle();
  // A resource that declares `blocks: "transport"` says so here rather than in
  // the activity tray: the suspended control and its reason stay together. It is
  // one line of micro text now instead of a card — the controls it suspends are
  // already showing their own disabled state beside it.
  const transportResourceWork = resourceActivitiesFor(resourceReadiness, "transport-inline")[0];
  const interaction = resourceInteractionGates(resourceReadiness, !rendererOnlyScene);
  const initialSceneReady = !requiresFencedInitialRasterPresentation(methodId)
    || (gpuInfo?.initialSparseAuthorityReady === true
      && gpuInfo?.initialRasterSurfaceReady === true);
  const transportLocked = rendererOnlyScene || !interaction.transportInteractive || !initialSceneReady;
  const safeStepLocked = safeBringup && (safeStepRequested || (gpuInfo?.encodedSteps ?? 0) >= 1);
  const toggleRecording = () => {
    if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
    else {
      setRunState("running");
      simulationRecording.start(simulationTime);
    }
  };
  return (
    <div
      className="transport-cluster"
      data-idle={idle ? "true" : "false"}
      data-testid="transport-cluster"
    >
      {/* The studio's only feedback channel — it is what says "Nothing to undo"
          now that the chip's history buttons are gone — so it survives the cut,
          as one ellipsised line that fades with the rest of the cluster. */}
      {notice && <p className={`transport-notice${noticeTone === "warn" ? " warn" : ""}`} title={notice}>{notice}</p>}
      <button
        type="button"
        className="transport-main"
        disabled={transportLocked || browserSafetyLocked}
        onClick={() => setRunState(runState === "running" ? "paused" : "running")}
        aria-label={browserPolicyPending ? "Browser GPU safety policy is loading"
          : rendererOnlyScene ? "Fluid simulation is disabled for this renderer validation scene"
          : safeBringup ? "Continuous play is disabled during bounded GPU bring-up"
          : transportLocked ? "Simulation controls unlock after the initial GPU scene is ready"
          : runState === "running" ? "Pause simulation" : "Play simulation"}
      >{transportLocked || browserPolicyPending ? "…" : runState === "running" ? "Ⅱ" : "▶"}</button>
      <button
        type="button"
        disabled={browserPolicyPending || transportLocked || safeStepLocked}
        onClick={() => { if (safeBringup) setSafeStepRequested(true); simulation.singleStep(); }}
        aria-label={browserPolicyPending ? "Browser GPU safety policy is loading"
          : transportLocked ? "Single step unavailable until the initial GPU scene is ready"
          : safeStepLocked ? "The bounded browser GPU step has already been requested"
          : "Single fluid clock step"}
      >STEP</button>
      {/* Not in the plan's list, and kept anyway: nothing else in the studio can
          put a scene back to t=0, and there is no shortcut for it. It goes to
          the ring with the rest of the tank actions in a later slice. */}
      <button
        type="button"
        disabled={browserSafetyLocked}
        onClick={() => {
          if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
          simulation.reset();
        }}
        aria-label="Reset the simulation to its authored initial state"
      >RESET</button>
      <button
        type="button"
        className={`record-button${recordingStatus === "recording" ? " active" : ""}`}
        onClick={toggleRecording}
        disabled={recordingStatus === "processing" || transportLocked || browserSafetyLocked}
        aria-label={recordingStatus === "recording" ? "Stop simulation recording" : "Record simulation video"}
        data-testid="record-simulation"
      >{recordingStatus === "recording" ? "■ STOP" : recordingStatus === "processing" ? "WAIT" : "● REC"}</button>
      {recording && recordingStatus !== "recording" && <button
        type="button"
        onClick={() => simulationRecording.open()}
        title="Play back the recorded simulation"
      >Playback</button>}
      {safeBringup && <button type="button" className="stop-gpu-button" onClick={requestManualGPUStop}>STOP GPU</button>}
      <output className="transport-time" aria-label="Simulation time in seconds">
        {recordingStatus === "recording" && recordingStart !== null
          ? <i className="transport-recording-dot" aria-hidden="true" /> : null}
        <strong>{simulationTime.toFixed(4)}</strong><small>s</small>
      </output>
      {transportResourceWork && <span
        className="transport-resource-state"
        role="status"
        aria-live="polite"
        title={transportResourceWork.label}
      >
        <i aria-hidden="true" />
        <strong>{transportResourceWork.operation ?? transportResourceWork.label}</strong>
        {transportResourceWork.total > 0 && <small>{Math.min(transportResourceWork.completed, transportResourceWork.total)}/{transportResourceWork.total}</small>}
      </span>}
    </div>
  );
}
