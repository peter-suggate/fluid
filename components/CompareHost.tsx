"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { shellCompareStore, startCompareSync } from "../lib/core/compare/compare-model";
import { COMPARE_ADOPTIONS, SECOND_PANE_ID } from "../lib/core/compare/compare-mode";
import { createPaneSession, type PaneSession } from "../lib/core/session/session";
import { SessionProvider, useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { useShellStore } from "../lib/core/stores/shell-store";
import { useEditorShortcuts } from "../lib/core/use-editor-shortcuts";
import { CompareDiffStrip } from "./CompareDiffStrip";
import { RecordingPlaybackModal } from "./RecordingPlaybackModal";
import { SceneOverlay } from "./SceneOverlay";
import { ScenePane } from "./ScenePane";
import { TransportBar } from "./TransportBar";

/**
 * The shell, with one pane or two.
 *
 * Single-pane mode is compare mode with one session, so there is exactly one
 * code path: the same `<ScenePane>` under the same provider, and the second
 * column is either there or it is not. Host chrome — the scene chip, the
 * transport, the recorder — is drawn once over the whole shell, because there
 * is one document being authored and one clock.
 *
 * The splitter writes its result at the *end* of the drag and never during it.
 * Every column change reallocates both panes' G-buffers and presentation
 * surfaces, so a live resize would rebuild two renderers per pointer move; what
 * follows the pointer instead is a hairline ghost that costs nothing.
 */

type PaneSessionWindow = Window & {
  /**
   * Retained for the reason the viewport lifecycle and the pane-lease broker
   * are: a Fast Refresh or an RSC program reload re-evaluates this module, and
   * a fresh session would hand pane B a second set of stores while its worker,
   * its device and its lease still belong to the first.
   */
  __fluidLabPaneBSession?: PaneSession;
};

function paneBSession(): PaneSession {
  if (typeof window === "undefined") return createPaneSession(SECOND_PANE_ID);
  const host = window as PaneSessionWindow;
  host.__fluidLabPaneBSession ??= createPaneSession(SECOND_PANE_ID);
  return host.__fluidLabPaneBSession;
}

/** Room for the smallest pane that is still a scene rather than a sliver. */
const MINIMUM_PANE_FRACTION = 0.18;

export function CompareHost() {
  const sessionA = useSession();
  const compare = useShellStore((state) => state.compare);
  const setFocusedPane = useShellStore((state) => state.setFocusedPane);
  const active = compare.active;
  const [sessionB] = useState(paneBSession);
  const [splitFraction, setSplitFraction] = useState(0.5);
  const shellRef = useRef<HTMLElement>(null);
  const ghostRef = useRef<HTMLDivElement>(null);

  // Attached and registered here rather than only in the ring's verb, so a
  // compare *link* restores the pair too: the URL sets the record directly and
  // never goes through `openCompareMode`.
  //
  // The attach is what makes pane B a pane at all. Everything the controller
  // does for a named pane — `setMethod`, the transport, the published clock —
  // resolves the id through `simulation.session(paneId)`, which falls back to
  // pane A's realm for an id it has never been handed. Without this, every
  // control mounted under B would read B's stores and write A's.
  useEffect(() => {
    if (!active) return;
    simulation.attachPaneSession(SECOND_PANE_ID, sessionB);
    simulation.registerPane(SECOND_PANE_ID);
    return () => {
      simulation.unregisterPane(SECOND_PANE_ID);
      simulation.detachPaneSession(SECOND_PANE_ID);
    };
  }, [active, sessionB]);

  // B is A plus the diff, for as long as the mode is open. Started after the
  // registration so the first mirror lands on a pane the host already knows.
  useEffect(() => {
    if (!active) return;
    const sync = startCompareSync(sessionA, sessionB, shellCompareStore(), COMPARE_ADOPTIONS);
    return () => sync.stop();
  }, [active, sessionA, sessionB]);

  const focusA = useCallback(() => setFocusedPane("a"), [setFocusedPane]);
  const focusB = useCallback(() => setFocusedPane(SECOND_PANE_ID), [setFocusedPane]);
  // The keyboard belongs to the pane last pointed at. One listener, not two:
  // these are window shortcuts, and two of them would fire both.
  useEditorShortcuts(active && compare.focusedPane === "b" ? sessionB : sessionA);

  const beginSplitDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const shell = shellRef.current;
    const ghost = ghostRef.current;
    if (!shell || !ghost) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const rect = shell.getBoundingClientRect();
    let fraction = splitFraction;
    const track = (clientX: number) => {
      fraction = Math.min(1 - MINIMUM_PANE_FRACTION, Math.max(MINIMUM_PANE_FRACTION,
        (clientX - rect.left) / Math.max(rect.width, 1)));
      ghost.style.left = `${fraction * 100}%`;
    };
    ghost.dataset.dragging = "true";
    track(event.clientX);
    const move = (moved: PointerEvent) => track(moved.clientX);
    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      delete ghost.dataset.dragging;
      // The one layout write of the whole gesture.
      setSplitFraction(fraction);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  const runState = sessionA.runtime((state) => state.runState);
  const simulationTime = sessionA.runtime((state) => state.simulationTime);
  const bodies = sessionA.diagnostics((state) => state.bodies);
  const shellView = useShellStore((state) => state.view);

  return (
    <main
      ref={shellRef}
      className="lab-shell"
      data-run-state={runState}
      data-solver-mode="eulerian"
      data-simulation-time={simulationTime.toFixed(6)}
      data-body-count={bodies.length}
      data-shell-view={shellView}
      data-compare={active}
      style={active ? {
        gridTemplateColumns: `minmax(0, ${splitFraction}fr) 6px minmax(0, ${1 - splitFraction}fr)`,
      } : undefined}
    >
      <ScenePane paneId="a" tagged={active} focused={!active || compare.focusedPane === "a"} onFocus={focusA} />
      {active && <>
        <div
          className="compare-splitter"
          data-testid="compare-splitter"
          role="separator"
          aria-orientation="vertical"
          aria-label="Pane split"
          onPointerDown={beginSplitDrag}
        />
        <SessionProvider value={sessionB}>
          <ScenePane paneId={SECOND_PANE_ID} tagged focused={compare.focusedPane === "b"} onFocus={focusB} />
        </SessionProvider>
        <div ref={ghostRef} className="compare-splitter-ghost" aria-hidden="true" style={{ left: `${splitFraction * 100}%` }} />
        <CompareDiffStrip a={sessionA} b={sessionB} />
      </>}

      {/* Host chrome: one scene chip, one transport, one recorder. Absolutely
          positioned against the shell rather than a pane, so with one column it
          lands exactly where it always did and with two it stays centred over
          the pair rather than duplicating itself on each. */}
      <div className="viewport-topline">
        <div className="topline-left">
          <SceneOverlay />
        </div>
      </div>
      <TransportBar />
      <RecordingPlaybackModal />
    </main>
  );
}
