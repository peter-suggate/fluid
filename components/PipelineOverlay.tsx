"use client";

import { SCENE_INSTRUMENTS, SCENE_INSTRUMENT_ORDER } from "../lib/core/scene-instruments";
import { DiagnosticsOverlay } from "./DiagnosticsOverlay";
import { RenderPipelineOverlay } from "../lib/svo/pipeline/ui";
import { SimPipelineOverlay } from "./SimPipelineOverlay";
import { useSession } from "../lib/core/session/session-context";

/**
 * The one host every scene instrument is drawn in.
 *
 * A pane of frosted glass along one edge, never a page. The three things it can
 * hold — the advance pipeline, the frame pipeline, the metric cards — are all
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
 * The three contents own that status band; this owns the frame, the scroll, the
 * title and the two ways out (the close button and Escape).
 */

export function PipelineOverlay() {
  const session = useSession();
  const overlay = session.ui((state) => state.sceneOverlay);
  const setSceneOverlay = session.ui((state) => state.setSceneOverlay);
  if (!overlay) return null;
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
      {overlay === "sim-pipeline" && <SimPipelineOverlay />}
      {overlay === "render-pipeline" && <RenderPipelineOverlay />}
      {overlay === "diagnostics" && <DiagnosticsOverlay />}
    </div>
  </aside>;
}

/**
 * The three instruments as toggles, in the transport's overflow menu.
 *
 * The ring is still the route for everything you do *to* the scene, but an
 * instrument is not a verb on an object — it is a reading of the whole run, so
 * it needs a route that does not depend on a right-click landing on the water.
 * These used to sit permanently beside the frame rate in the top-right corner,
 * which made that corner a third bar. They are occasional, so they live in the
 * menu now. The keys in `SCENE_INSTRUMENTS` do the same job with no chrome, and
 * each toggle shows its key so the menu is where people learn them.
 */
export function SceneInstrumentTags() {
  const session = useSession();
  const overlay = session.ui((state) => state.sceneOverlay);
  const setSceneOverlay = session.ui((state) => state.setSceneOverlay);
  return <div className="transport-menu-instruments" role="group" aria-label="Scene instruments" data-testid="scene-instrument-tags">
    {SCENE_INSTRUMENT_ORDER.map((instrument) => {
      const open = overlay === instrument.id;
      return <button
        key={instrument.id}
        type="button"
        data-testid={`scene-instrument-tag-${instrument.id}`}
        aria-pressed={open}
        aria-label={`${open ? "Close" : "Open"} ${instrument.label.toLowerCase()}`}
        title={`${instrument.hint} · ${instrument.shortcut.toUpperCase()}`}
        onClick={() => setSceneOverlay(open ? null : instrument.id)}
      >{instrument.tag}<kbd>{instrument.shortcut.toUpperCase()}</kbd></button>;
    })}
  </div>;
}
