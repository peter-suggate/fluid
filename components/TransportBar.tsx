"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useStore } from "zustand";
import { simulation } from "../lib/core/simulation/controller";
import { simulationRecording } from "../lib/core/simulation/recording";
import { requestManualGPUStop } from "../lib/core/gpu-startup";
import { useSafeBrowserGPUBringup } from "../lib/core/use-safe-browser-gpu-bringup";
import { planSceneRuntime } from "../lib/core/scene-runtime";
import { resourceInteractionGates } from "../lib/core/resource-readiness";
import { transportLockReason, transportReadiness } from "../lib/core/transport-status";
import { effectiveSimulationStep_s, methodPinsSimulationStep } from "../lib/core/simulation-step";
import { SessionProvider, useSession } from "../lib/core/session/session-context";
import type { PaneSession } from "../lib/core/session/session";
import { attachFrameRateReadout } from "../lib/core/frame-rate-readout";
import { InteractiveDocumentActions, useInteractionPane } from "../lib/features/interactive/ui";
import type { InteractionPane } from "../lib/features/interactive/pane";
import { SceneInstrumentTags } from "./PipelineOverlay";
import { ViewportModeToggle } from "./ViewportModeToggle";
import { Stepper } from "./ui";

/** How long the pointer has to be still before the cluster recedes. */
const POINTER_IDLE_MS = 2600;
/** How long a notice stays on the cluster after it was last said. */
const NOTICE_LIFETIME_MS = 6000;
/** The step-size control's range, in milliseconds. */
const STEP_MIN_MS = 1;
const STEP_MAX_MS = 50;

/**
 * The fixed simulation step, as one control on the transport.
 *
 * The clock's step is the one solve parameter a reader changes while watching
 * the scene — to slow a splash down, or to find where a scene goes unstable —
 * so reaching into the pipeline overlay's folded Numerics drawer for it was a
 * detour every time. It sits in the transport's menu rather than on the bar,
 * because almost every run keeps the default. This is the same setter the
 * drawer calls, shown as the *effective* step: when a method pins the clock
 * (the uniform paper profile), the number shown is the one the simulation
 * actually takes, and editing it releases the pin rather than presenting a
 * number that refuses to move.
 */
function StepSizeControl({ disabled }: { readonly disabled: boolean }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const quality = session.method((state) => state.quality);
  const overrides = session.method((state) => state.overrides);
  const method = { methodId, quality, overrides };
  const step_ms = effectiveSimulationStep_s(scene, method) * 1000;
  const pinned = methodPinsSimulationStep(scene, method);
  // Halve and double rather than ±1 ms: the step is a log-scale knob — 4, 8,
  // 16, 33 ms are the settings a reader actually moves between, and reaching
  // 4 ms from 33 one millisecond at a time is not a control. Printed to a tenth
  // at most: the control moves in halvings and doublings, so a second decimal
  // is noise, and it was the character that clipped the field.
  return (
    <span
      className="transport-step"
      role="group"
      aria-label="Simulation step size"
      title={pinned
        ? "The active method pins the clock to this step; editing it releases the method to the scene-authored step."
        : "Fixed simulation step. Rigid bodies and fluid advance on the same step; changes apply live without resetting the clock."}
      data-pinned={pinned ? "true" : "false"}
      data-testid="transport-step"
    >
      <small>dt</small>
      <Stepper
        value={step_ms}
        onChange={(value_ms) => simulation.setStepSize(value_ms / 1000, session.id)}
        min={STEP_MIN_MS} max={STEP_MAX_MS}
        factor={2} grain={0.1}
        unit="ms"
        editable
        disabled={disabled}
        ariaLabel="Simulation step in milliseconds"
        decreaseHint="Halve the simulation step"
        increaseHint="Double the simulation step"
      />
    </span>
  );
}

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
 * Whether the notice has been on screen long enough to have been read.
 *
 * Keyed on the store's said-count rather than on the text, so the same sentence
 * said twice (two resets in a row) starts its clock again instead of staying
 * faded out. The notice keeps its row either way — this only stops it from
 * becoming a permanent caption on the transport, and never moves the buttons.
 */
function useNoticeStale(notice: string, said: number, lifetime_ms = NOTICE_LIFETIME_MS) {
  // What is held is the notice that has expired, not a flag: a flag would have
  // to be cleared from the effect body on every change, and the comparison says
  // the same thing without a second render.
  const [expired, setExpired] = useState<number | null>(null);
  useEffect(() => {
    const timer = window.setTimeout(() => setExpired(said), lifetime_ms);
    return () => window.clearTimeout(timer);
  }, [said, lifetime_ms]);
  return notice !== "" && expired === said;
}

/**
 * The focused pane's presentation rate. The pane's render loop writes the text
 * straight into this element (see `frame-rate-readout.ts`), so the number
 * changing every frame never re-renders the bar.
 */
function FrameRateReadout({ paneId }: { readonly paneId: string }) {
  const ref = useRef<HTMLOutputElement>(null);
  useEffect(() => (ref.current ? attachFrameRateReadout(paneId, ref.current) : undefined), [paneId]);
  return <output
    ref={ref}
    className="transport-fps"
    data-testid="fps-meter"
    aria-label="Presentation frame rate"
    title="WebGPU presentations per second · rolling mean of the latest 5 frame intervals"
  />;
}

/**
 * Why the interactive entities hold the run, when they do.
 *
 * The host already refuses to run while a pane's interaction is not ready.
 * Before, the only explanation was a readiness chip and a list in the
 * interactive HUD at the top of the pane, far from the Play button that did
 * nothing. Now the reason sits next to Play, and only while it applies.
 */
function InteractionHold({ pane }: { readonly pane: InteractionPane }) {
  const readiness = useStore(pane.store, (state) => state.view.readiness);
  const entities = useStore(pane.store, (state) => state.view.document.entities.length);
  if (!entities || readiness.state === "ready") return null;
  return <span
    className="transport-hold"
    data-state={readiness.state}
    role="status"
    title={readiness.reasons.join("\n") || undefined}
    data-testid="interactive-readiness"
  >{readiness.reasons[0] ?? readiness.state}</span>;
}

/**
 * Everything the bar does not need to show all the time.
 *
 * Recording, the step size, the three instruments and the interactive
 * document's file actions are each used now and then, not every session.
 * Before, they took a permanent place in one of three bars. The menu stays
 * open until Escape, a click outside, or the trigger again: an item never
 * closes it, because Import opens a file dialog whose input lives in here.
 */
function TransportMenu({ viewSession, viewPane, recordingControls, stepDisabled }: {
  readonly viewSession: PaneSession;
  readonly stepDisabled: boolean;
  readonly viewPane: InteractionPane | undefined;
  readonly recordingControls: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", outside, true);
    return () => window.removeEventListener("pointerdown", outside, true);
  }, [open]);
  return <span
    ref={root}
    className="transport-menu"
    onKeyDown={(event) => {
      if (event.key !== "Escape" || !open) return;
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      trigger.current?.focus();
    }}
  >
    <button
      ref={trigger}
      type="button"
      className="transport-more"
      aria-haspopup="true"
      aria-expanded={open}
      aria-label="More: recording, step size, instruments, entities"
      title="Recording, step size, instruments and entities"
      data-testid="transport-more"
      onClick={() => setOpen(!open)}
    >⋯</button>
    {open && <div className="transport-menu-panel" role="group" aria-label="More transport controls" data-testid="transport-menu">
      <div className="transport-menu-section" role="group" aria-label="Recording">
        <small>Recording</small>
        {recordingControls}
      </div>
      <div className="transport-menu-section" role="group" aria-label="Step size">
        <small>Step size</small>
        <StepSizeControl disabled={stepDisabled} />
      </div>
      <SessionProvider value={viewSession}>
        <div className="transport-menu-section">
          <small>Instruments</small>
          <SceneInstrumentTags />
        </div>
      </SessionProvider>
      {viewPane && <InteractiveDocumentActions pane={viewPane} />}
    </div>}
  </span>;
}

/**
 * The studio's one bar: the mode, the clock, and what the frame costs.
 *
 * This was a full-width docked footer with three columns: transport, a step-size
 * slider with an ms entry, a rate chip, a lag chip, a lockstep label, and file
 * actions. All of that cost the viewport a permanent 58 px row in order to state
 * things that are either constant (the lockstep contract), diagnostic (the rate
 * and lag chips, which belong beside the timings that explain them), or
 * authored once (the step size).
 *
 * After that it was one of three floating bars: this one, a top-right corner
 * with the LOOK/EDIT toggle, the instrument tags and the frame rate, and an
 * interactive HUD at the top centre with its own Play and Reset. Two Plays, two
 * Resets, and an EDIT chip beside a LOOK toggle that meant different things.
 * Now every verb has one home. The bar holds what is used or read every
 * session: LOOK/EDIT, run, step, reset, the time, the frame rate, and (only
 * while it applies) why the run is held. Everything occasional is in `⋯`.
 *
 * It floats on the scene and recedes when the pointer stops, but only
 * visually. It stays in the DOM, stays focusable, and comes fully back the
 * moment anything inside it takes focus, because a control that disappears
 * from the tab order is a control that is gone.
 */
export function TransportBar({ hostBlockReason, viewSession }: {
  readonly hostBlockReason?: string;
  /** The focused pane: what the mode, the frame rate and the menu's view items are about. */
  readonly viewSession: PaneSession;
}) {
  // The host's realm. This component is mounted once by `CompareHost`, outside
  // pane B's provider, so `useSession()` here is pane A — which *is* the host:
  // `runState` is the transport's own state and `simulationTime` is the host
  // clock's minimum, published into every pane. What the controls write goes
  // back through `simulation`, never into this one session, because a transport
  // gesture is about the experiment and reaches every attached pane.
  const session = useSession();
  const hostPane = useInteractionPane(session.id);
  const viewPane = useInteractionPane(viewSession.id);
  const runState = session.runtime((state) => state.runState);
  const simulationTime = session.runtime((state) => state.simulationTime);
  const notice = session.runtime((state) => state.notice);
  const noticeTone = session.runtime((state) => state.noticeTone);
  const noticeSaid = session.runtime((state) => state.noticeSaid);
  const fluidRequired = session.scene((state) => planSceneRuntime(state.scene).fluidSolver);
  const rendererOnlyScene = session.scene((state) => !planSceneRuntime(state.scene).fluidSolver && !state.scene.interaction?.entities.length);
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const gpuInfo = session.diagnostics((state) => state.gpuInfo);
  const methodId = session.method((state) => state.methodId);
  const recordingStatus = session.recording((state) => state.status);
  const recordingStart = session.recording((state) => state.startedAtSimulation_s);
  const recording = session.recording((state) => state.recording);
  const safeBringupPolicy = useSafeBrowserGPUBringup();
  const noticeStale = useNoticeStale(notice, noticeSaid);
  const safeBringup = safeBringupPolicy === true;
  const browserPolicyPending = safeBringupPolicy === null;
  const browserSafetyLocked = safeBringupPolicy !== false;
  const [safeStepRequested, setSafeStepRequested] = useState(false);
  const idle = usePointerIdle();
  const interaction = resourceInteractionGates(resourceReadiness, fluidRequired);
  // The bar itself never narrates work: loading, compiling and topology states
  // show as progress in the activity tray (see `transportWorkStatus`, rendered
  // by ScenePane). Here they exist only as a lock, stated on the controls.
  const readiness = transportReadiness(gpuInfo, methodId);
  const transportLocked = Boolean(hostBlockReason) || rendererOnlyScene || !interaction.transportInteractive
    || fluidRequired && (!readiness.initialSceneReady || !readiness.simulationReady);
  const transportLockedReason = hostBlockReason ?? transportLockReason(readiness, gpuInfo);
  const safeStepLocked = safeBringup && (safeStepRequested || (gpuInfo?.encodedSteps ?? 0) >= 1);
  // A simulated interaction host (`?interactive=demo`) is not wired to the
  // controller, so it follows the transport by command. The production host
  // follows the controller's run state and needs nothing here.
  const simulatedPanes = [...new Set([hostPane, viewPane])]
    .filter((pane): pane is InteractionPane => Boolean(pane?.binding.hostLabel));
  const setRunState = (next: "running" | "paused") => {
    simulation.setRunState(next);
    for (const pane of simulatedPanes) void pane.transport(next === "running" ? "play" : "pause");
  };
  const toggleRecording = () => {
    if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
    else {
      setRunState("running");
      simulationRecording.start(simulationTime);
    }
  };
  const recordLabel = recordingStatus === "recording" ? "■ Stop recording" : recordingStatus === "processing" ? "Encoding…" : "● Record";
  const recordingControls = <>
    <button
      type="button"
      className={`record-button${recordingStatus === "recording" ? " active" : ""}`}
      onClick={toggleRecording}
      disabled={recordingStatus === "processing" || transportLocked || browserSafetyLocked}
      aria-label={recordingStatus === "recording" ? "Stop simulation recording" : "Record simulation video"}
      data-testid="record-simulation"
    >{recordLabel}</button>
    <button
      type="button"
      disabled={!recording || recordingStatus === "recording"}
      onClick={() => simulationRecording.open()}
      title="Play back the recorded simulation"
    >▸ Play back</button>
  </>;
  return (
    <div
      className="transport-cluster"
      data-idle={idle ? "true" : "false"}
      data-testid="transport-cluster"
    >
      {/* The studio's only feedback channel — it is what says "Nothing to undo"
          now that the chip's history buttons are gone — so it survives the cut,
          as one ellipsised caption floated above the bar (out of flow, so it can
          never resize the cluster) that fades with the rest of the cluster. */}
      {notice && <p
        className={`transport-notice${noticeTone === "warn" ? " warn" : ""}`}
        data-stale={noticeStale ? "true" : "false"}
        title={notice}
      >{notice}</p>}
      {/* First, and fully lit in EDIT: it is the answer to "why did my click
          do nothing", so it cannot itself be something you have to find. */}
      <SessionProvider value={viewSession}><ViewportModeToggle /></SessionProvider>
      <button
        type="button"
        className="transport-main"
        disabled={transportLocked || browserSafetyLocked}
        onClick={() => setRunState(runState === "running" ? "paused" : "running")}
        aria-label={browserPolicyPending ? "Browser GPU safety policy is loading"
          : rendererOnlyScene ? "Fluid simulation is disabled for this renderer validation scene"
          : safeBringup ? "Continuous play is disabled during bounded GPU bring-up"
          : transportLocked ? transportLockedReason
          : runState === "running" ? "Pause simulation" : "Play simulation"}
      >{transportLocked || browserPolicyPending ? "…" : runState === "running" ? "Ⅱ" : "▶"}</button>
      <button
        type="button"
        disabled={browserPolicyPending || transportLocked || safeStepLocked}
        onClick={() => { if (safeBringup) setSafeStepRequested(true); simulation.singleStep(); }}
        aria-label={browserPolicyPending ? "Browser GPU safety policy is loading"
          : transportLocked ? transportLockedReason
          : safeStepLocked ? "The bounded browser GPU step has already been requested"
          : "Single fluid clock step"}
      >STEP</button>
      {/* Nothing else in the studio puts a scene back to t=0, and there is no
          shortcut for it, so it stays on the bar. */}
      <button
        type="button"
        disabled={browserSafetyLocked}
        onClick={() => {
          if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
          simulation.resetAll();
          for (const pane of simulatedPanes) void pane.transport("reset");
        }}
        aria-label="Reset the simulation to its authored initial state"
      >RESET</button>
      {/* Stopping a recording is the one recording verb that cannot wait in a
          menu, so it comes out onto the bar while a recording runs. */}
      {recordingStatus === "recording" && <button
        type="button"
        className="record-button active"
        onClick={toggleRecording}
        aria-label="Stop simulation recording"
      >■ STOP</button>}
      {safeBringup && <button type="button" className="stop-gpu-button" onClick={requestManualGPUStop}>STOP GPU</button>}
      <output className="transport-time" aria-label="Simulation time in seconds">
        {recordingStatus === "recording" && recordingStart !== null
          ? <i className="transport-recording-dot" aria-hidden="true" /> : null}
        <strong>{simulationTime.toFixed(4)}</strong><small>s</small>
      </output>
      {viewPane && <InteractionHold pane={viewPane} />}
      <FrameRateReadout paneId={viewSession.id} />
      <TransportMenu
        viewSession={viewSession}
        viewPane={viewPane}
        recordingControls={recordingControls}
        stepDisabled={browserSafetyLocked || rendererOnlyScene}
      />
    </div>
  );
}
