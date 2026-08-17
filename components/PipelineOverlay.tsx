"use client";

import { useUIStore, type SceneOverlay } from "../lib/core/stores/ui-store";
import { DiagnosticsOverlay } from "./DiagnosticsOverlay";
import { RenderPipelineOverlay } from "./RenderPipelineOverlay";
import { SimPipelineOverlay } from "./SimPipelineOverlay";

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

const OVERLAYS: Readonly<Record<SceneOverlay, {
  readonly eyebrow: string;
  readonly title: string;
  readonly label: string;
}>> = {
  "sim-pipeline": {
    eyebrow: "Simulation",
    title: "Advance pipeline",
    label: "Simulation pipeline",
  },
  "render-pipeline": {
    eyebrow: "Render",
    title: "Frame pipeline",
    label: "Render pipeline",
  },
  diagnostics: {
    eyebrow: "Live",
    title: "Diagnostics",
    label: "Diagnostics",
  },
};

export function PipelineOverlay() {
  const overlay = useUIStore((state) => state.sceneOverlay);
  const setSceneOverlay = useUIStore((state) => state.setSceneOverlay);
  if (!overlay) return null;
  const chrome = OVERLAYS[overlay];
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
