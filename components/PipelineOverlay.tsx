"use client";

import { SCENE_INSTRUMENTS } from "../lib/core/scene-instruments";
import { DiagnosticsOverlay } from "./DiagnosticsOverlay";
import { useSession } from "../lib/core/session/session-context";

/**
 * The host the metric cards are drawn in.
 *
 * A pane of frosted glass along one edge, never a page. The cards are
 * readings *about the scene*, and a reading that covers the thing it is a
 * reading of is useless: the picture has to stay visible while a slider moves,
 * because watching the water is how you tell whether the number was worth it.
 * So the pane claims a narrow bounded width, the scene keeps the rest, and the
 * viewport behind and around it stays live — no scrim, no modal trap, no
 * pointer-blocking layer. The panel itself is the only thing that takes a click.
 *
 * The chrome is deliberately quiet. This used to open with an accent eyebrow
 * over a 16px heading over a status strip over a control strip, which is four
 * bands of furniture before the first measurement and reads as a docked panel
 * that happens to be floating. It is one 34px line now — what this is, and the
 * way out — and the contents state their own status on a single band under it.
 *
 * The contents own that status band; this owns the frame, the scroll, the
 * title and the two ways out (the close button and Escape).
 */

export function PipelineOverlay() {
  const session = useSession();
  const overlay = session.ui((state) => state.sceneOverlay);
  const setSceneOverlay = session.ui((state) => state.setSceneOverlay);
  // The two pipelines are rail cards now (Simulation, Performance); ViewRail opens them.
  if (overlay !== "diagnostics") return null;
  const chrome = SCENE_INSTRUMENTS[overlay];
  return <aside
    className="scene-instrument"
    data-testid="scene-instrument"
    data-overlay={overlay}
    aria-label={chrome.label}
    // Escape from inside, where the global shortcut cannot reach: that handler
    // ignores keys typed into a field, which is exactly where the pointer is
    // while somebody is editing a budget in here. Propagation stops so the two
    // paths never both fire and drop the selection on the way out.
    onKeyDown={(event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      setSceneOverlay(null);
    }}
  >
    <header className="scene-instrument-head">
      <h2>{chrome.title}</h2>
      <span>{chrome.eyebrow}</span>
      <button className="panel-close" type="button" aria-label={`Close ${chrome.label.toLowerCase()}`}
        onClick={() => setSceneOverlay(null)}>×</button>
    </header>
    {/* The scroll is the body's, not the pane's: a pane that scrolls itself
        cannot hold a header still without a sticky element re-blurring the
        backdrop under it, and rounded corners clip a scrolling child cleanly. */}
    <div className="scene-instrument-body panel-scroll">
      <DiagnosticsOverlay />
    </div>
  </aside>;
}
