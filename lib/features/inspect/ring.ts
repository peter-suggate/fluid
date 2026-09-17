import type { EditorAction, EditorActionEffect } from "../../core/editor-action";

/**
 * The two pointer probes, as ring wedges — composed once for either host.
 *
 * They belong to no *entity*: they belong to the pixel. The ray probe answers
 * "what did the renderer do to draw this dot", which is the same question over
 * a stone, a canopy leaf or the water; the cell probe answers "which pressure
 * unknown lives behind this dot", which is only a question where there is
 * fluid. `lib/core/editor-probe-actions.ts` is the studio's two lines over
 * these, and the lab's deleted `slice-actions.ts` used to write the cell one
 * out again under a third id (`slice-inspect-cell`) with a third hint.
 *
 * The effect is the host's, and here that is not a formality: the studio aims
 * at a *pixel* and the renderer answers with the leaf behind it, while the lab
 * has a cell index already — the ring opened on one. The wedge is the same
 * wedge either way, which is the claim worth keeping.
 */

/** Read the pressure cell behind this point. */
export function cellProbeWedge(options: {
  readonly effect?: EditorActionEffect;
  readonly hint: string;
  readonly enabled?: boolean;
}): EditorAction {
  return {
    id: "inspect-cell",
    label: "Inspect cell",
    icon: "inspect-cell",
    tone: "fluid",
    hint: options.hint,
    ...(options.enabled === undefined ? {} : { enabled: options.enabled }),
    ...(options.effect ? { effect: options.effect } : {}),
  };
}

/**
 * Read the ray behind this pixel.
 *
 * Toned like the frame graph rather than like the thing under the cursor,
 * because that is what it prices: the picture, not the object. The wedge means
 * the same over a stone and over the water, and a wedge that changed colour
 * with whatever it happened to be over would read as a different verb each time.
 */
export function rayProbeWedge(options: {
  readonly effect?: EditorActionEffect;
  readonly hint: string;
  readonly enabled?: boolean;
}): EditorAction {
  return {
    id: "trace-ray",
    label: "Trace ray",
    icon: "trace-ray",
    tone: "prop",
    hint: options.hint,
    ...(options.enabled === undefined ? {} : { enabled: options.enabled }),
    ...(options.effect ? { effect: options.effect } : {}),
  };
}
