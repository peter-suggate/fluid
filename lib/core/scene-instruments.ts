import type { EditorAction, EditorActionIcon, EditorActionTone } from "./editor-action";
import type { SceneOverlay } from "./stores/ui-store";

/**
 * The three instruments, declared once.
 *
 * An instrument is reachable from four places — the ring on the water, the ring
 * on empty space, a key, and the cluster beside the frame rate — and each of
 * those used to spell out its own name, icon and hint. Two of them had already
 * drifted into copies of the same sentence about the frame graph, which is the
 * failure mode: a wedge and a button that open the same pane should not be able
 * to disagree about what it is called.
 *
 * So the routes compose from here and hold nothing of their own. `Record` over
 * `SceneOverlay` rather than a list, so adding a fourth instrument to the union
 * fails to compile until it has said what it is — the same bargain
 * `SCENE_OVERLAY_KEYS` makes in `url-state.ts`.
 */
export interface SceneInstrument {
  readonly id: SceneOverlay;
  /** Kicker over the title in the pane's header. */
  readonly eyebrow: string;
  /** The pane's heading. */
  readonly title: string;
  /** Accessible name, used wherever the thing is named rather than shown. */
  readonly label: string;
  /** The word on a ring wedge, where the full title does not fit and Title Case is the house style. */
  readonly wedge: string;
  /** Short name in the frame-rate cluster, which is set in the tag font beside the number. */
  readonly tag: string;
  /**
   * The unmodified key that raises it.
   *
   * Chosen clear of `EDITOR_TOOLS` and of the camera framings, and the reason
   * these exist at all: every other route to an instrument starts with a
   * right-click that has to *land on something*, so a scene whose pick is
   * rebuilding — or whose water is off screen — could not be asked what it
   * costs. See `editorToolForShortcut` for the same lookup over tools.
   */
  readonly shortcut: string;
  readonly icon: EditorActionIcon;
  readonly tone: EditorActionTone;
  /** The one sentence saying what is inside, shown under the ring and on hover. */
  readonly hint: string;
}

export const SCENE_INSTRUMENTS: Readonly<Record<SceneOverlay, SceneInstrument>> = Object.freeze({
  "sim-pipeline": {
    id: "sim-pipeline",
    eyebrow: "Simulation",
    title: "Advance pipeline",
    label: "Simulation pipeline",
    wedge: "Pipeline",
    tag: "SIM",
    shortcut: "s",
    icon: "pipeline",
    tone: "fluid",
    hint: "Open the advance pipeline: per-stage GPU cost, gates and solver tuning",
  },
  "render-pipeline": {
    id: "render-pipeline",
    eyebrow: "Render",
    title: "Frame pipeline",
    label: "Render pipeline",
    wedge: "Render",
    tag: "FRAME",
    shortcut: "v",
    icon: "render-pipeline",
    tone: "prop",
    hint: "Open the frame graph: per-pass GPU cost, stage ablation and node tuning",
  },
  diagnostics: {
    id: "diagnostics",
    eyebrow: "Live",
    title: "Diagnostics",
    label: "Diagnostics",
    wedge: "Diagnostics",
    tag: "DIAG",
    shortcut: "h",
    icon: "diagnostics",
    tone: "fluid",
    hint: "Open the live instrument cards: divergence, residual, CFL, mass drift",
  },
} as const satisfies Readonly<Record<SceneOverlay, SceneInstrument>>);

/** Reading order: what the solve costs, what the picture costs, then what the solve is doing. */
export const SCENE_INSTRUMENT_ORDER: readonly SceneInstrument[] = Object.freeze([
  SCENE_INSTRUMENTS["sim-pipeline"],
  SCENE_INSTRUMENTS["render-pipeline"],
  SCENE_INSTRUMENTS.diagnostics,
]);

/**
 * Resolve an unmodified keypress to an instrument.
 *
 * Modified keys belong to the browser and to undo/redo, exactly as they do for
 * tools, so the caller filters those before asking.
 */
export function sceneInstrumentForShortcut(key: string): SceneInstrument | undefined {
  const normalized = key.toLowerCase();
  return SCENE_INSTRUMENT_ORDER.find((instrument) => instrument.shortcut === normalized);
}

/**
 * The wedge that raises an instrument, wherever a ring wants to offer it.
 *
 * The label is the wedge word rather than the title because a wedge is read at
 * a flick — "Pipeline" over "Advance pipeline" — while the hint under the ring
 * carries the sentence, and the key rides on the end of it so the ring is also
 * where the shortcut is learned.
 */
export function sceneInstrumentAction(
  instrument: SceneInstrument,
  overrides?: { readonly label?: string },
): EditorAction {
  return {
    id: instrument.id,
    label: overrides?.label ?? instrument.wedge,
    icon: instrument.icon,
    tone: instrument.tone,
    hint: `${instrument.hint} · ${instrument.shortcut.toUpperCase()}`,
    effect: { kind: "open-overlay", overlay: instrument.id },
  };
}
