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
import { useSession } from "../lib/core/session/session-context";
import type { PaneSession } from "../lib/core/session/session";
import { attachFrameRateReadout } from "../lib/core/frame-rate-readout";
import { InteractiveDocumentActions, useInteractionPane } from "../lib/features/interactive/ui";
import type { InteractionPane } from "../lib/features/interactive/pane";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { ActivityIndicator } from "./ActivityIndicator";
import { LoadMeters } from "./LoadMeters";
import { usePaneActivity } from "./pane-activity";

/** How long a notice stays on the card after it was last said. */
const NOTICE_LIFETIME_MS = 6000;

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
 * Everything the card does not need to show all the time.
 *
 * Recording and the interactive document's file actions are each used now and
 * then, not every session. The step size and the instruments used to be here
 * too; they are settings and readings of the run, so they live on the rail
 * with every other one (Simulation, Performance). The menu stays open until
 * Escape, a click outside, or the trigger again: an item never closes it,
 * because Import opens a file dialog whose input lives in here.
 */
function TransportMenu({ viewPane, recordingControls }: {
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
      aria-label="More: recording and entities"
      title="Recording and entities"
      data-testid="transport-more"
      onClick={() => setOpen(!open)}
    >⋯</button>
    {open && <div className="transport-menu-panel" role="group" aria-label="More transport controls" data-testid="transport-menu">
      <div className="transport-menu-section" role="group" aria-label="Recording">
        <small>Recording</small>
        {recordingControls}
      </div>
      {viewPane && <InteractiveDocumentActions pane={viewPane} />}
    </div>}
  </span>;
}

/** Re-renders once a second while `active`, so a time-based estimate keeps moving between reports. */
function useSecondTick(active: boolean) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setTick((tick) => tick + 1), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
}

/**
 * One pane's waiting, under the controls it holds, with the machine's load
 * beside it while there is any — the load is what says whether "still 37 of
 * 124" means busy or stuck.
 */
function PaneWork({ session, tag }: { readonly session: PaneSession; readonly tag?: string }) {
  const activity = usePaneActivity(session);
  const busy = activity.entries.length > 0;
  useSecondTick(busy);
  if (!busy) return null;
  return <div className="transport-work">
    <ActivityIndicator title={activity.title} entries={activity.entries} note={activity.note} tag={tag} />
    <LoadMeters session={session} />
  </div>;
}

/**
 * The studio's one card: the mode, the clock, what the frame costs, and —
 * while there is any — what the run is waiting on.
 *
 * This was a full-width docked footer, then one of three floating bars, then a
 * bar at the bottom centre while its explanation (the activity mark) hung in
 * the top-right corner and the settings stood on a rail down the right edge:
 * three places to look, and the reason Play was disabled 900 px from Play. Now
 * it heads the rail. The right edge is one column: this card, then the marks.
 *
 * The card holds what is used or read every session — run, step, reset, the
 * time, the frame rate, and (only while it applies) why the run is held — and
 * grows a section under them for as long as work holds the scene:
 * its progress, and the load of the machine doing it. Everything occasional is
 * in `⋯`. It never recedes: the clock is the one readout of where the run is.
 *
 * LOOK/EDIT is not on it. Tab switches the mode, and EDIT announces itself —
 * the sculpt rail stands on the left edge only then — so a button restating it
 * was the card's widest control and its least used.
 */
export function TransportBar({ hostBlockReason, viewSession, sessions }: {
  readonly hostBlockReason?: string;
  /** The focused pane: what the frame rate and the menu's entity items are about. */
  readonly viewSession: PaneSession;
  /** Every pane, in order: each states its own waiting, tagged when there is more than one. */
  readonly sessions: readonly PaneSession[];
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
  const [safeStepSpent, setSafeStepSpent] = useState(() => simulation.safeBrowserStepSpent());
  const interaction = resourceInteractionGates(resourceReadiness, fluidRequired);
  // The controls never narrate work: loading, compiling and topology states are
  // the work section's (see `usePaneActivity`). Here they are only a lock,
  // stated on each control.
  const readiness = transportReadiness(gpuInfo, methodId);
  const transportLocked = Boolean(hostBlockReason) || rendererOnlyScene || !interaction.transportInteractive
    || fluidRequired && (!readiness.initialSceneReady || !readiness.simulationReady);
  const transportLockedReason = hostBlockReason ?? transportLockReason(readiness, gpuInfo);
  const safeStepLocked = safeBringup && (safeStepSpent || (gpuInfo?.encodedSteps ?? 0) >= 1);
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
  const playLabel = browserPolicyPending ? "Browser GPU safety policy is loading"
    : rendererOnlyScene ? "Fluid simulation is disabled for this renderer validation scene"
    : safeBringup ? "Continuous play is disabled during bounded GPU bring-up"
    : transportLocked ? transportLockedReason
    : runState === "running" ? "Pause simulation" : "Play simulation";
  const stepLabel = browserPolicyPending ? "Browser GPU safety policy is loading"
    : transportLocked ? transportLockedReason
    : safeStepLocked ? "The bounded browser GPU step has already been requested"
    : "Single fluid clock step";
  return (
    <div className="transport-card" data-testid="transport-cluster">
      <div className="transport-controls">
        <button
          type="button"
          className="transport-main"
          disabled={transportLocked || browserSafetyLocked}
          onClick={() => setRunState(runState === "running" ? "paused" : "running")}
          aria-label={playLabel}
          title={playLabel}
        >{runState === "running" && !transportLocked
          ? <Pause width={15} height={15} strokeWidth={2} aria-hidden />
          : <Play width={15} height={15} strokeWidth={2} aria-hidden />}</button>
        <button
          type="button"
          disabled={browserPolicyPending || transportLocked || safeStepLocked}
          onClick={() => { simulation.singleStep(); setSafeStepSpent(simulation.safeBrowserStepSpent()); }}
          aria-label={stepLabel}
          title={stepLabel}
        ><StepForward width={15} height={15} strokeWidth={1.8} aria-hidden /></button>
        {/* Nothing else in the studio puts a scene back to t=0, and there is no
            shortcut for it, so it stays on the card. */}
        <button
          type="button"
          disabled={browserSafetyLocked}
          onClick={() => {
            if (recordingStatus === "recording") simulationRecording.stop(simulationTime);
            simulation.resetAll();
            for (const pane of simulatedPanes) void pane.transport("reset");
          }}
          aria-label="Reset the simulation to its authored initial state"
          title="Reset to t = 0"
        ><RotateCcw width={14} height={14} strokeWidth={1.8} aria-hidden /></button>
        {/* Stopping a recording is the one recording verb that cannot wait in a
            menu, so it comes out onto the card while a recording runs. */}
        {recordingStatus === "recording" && <button
          type="button"
          className="record-button active"
          onClick={toggleRecording}
          aria-label="Stop simulation recording"
        >■ STOP</button>}
        {safeBringup && <button type="button" className="stop-gpu-button" onClick={requestManualGPUStop}>STOP GPU</button>}
        <TransportMenu viewPane={viewPane} recordingControls={recordingControls} />
      </div>
      <div className="transport-readouts">
        <output className="transport-time" aria-label="Simulation time in seconds">
          {recordingStatus === "recording" && recordingStart !== null
            ? <i className="transport-recording-dot" aria-hidden="true" /> : null}
          <strong>{simulationTime.toFixed(4)}</strong><small>s</small>
        </output>
        {viewPane && <InteractionHold pane={viewPane} />}
        <FrameRateReadout paneId={viewSession.id} />
      </div>
      {sessions.map((pane) => <PaneWork key={pane.id} session={pane}
        tag={sessions.length > 1 ? pane.id.toUpperCase() : undefined} />)}
      {/* The studio's only feedback channel — it is what says "Nothing to undo"
          — so it hangs under the card, out of flow so it can never resize it. */}
      {notice && <p
        className={`transport-notice${noticeTone === "warn" ? " warn" : ""}`}
        data-stale={noticeStale ? "true" : "false"}
        title={notice}
      >{notice}</p>}
    </div>
  );
}
