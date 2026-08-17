import type { EditorAction, EditorActionTarget } from "./editor-action";

/**
 * The two pointer probes, as ring wedges.
 *
 * They are composed here rather than declared by an entity because they belong
 * to no *entity* — they belong to the pixel. The ray probe answers "what did the
 * renderer do to draw this dot", which is the same question over a stone, a
 * canopy leaf or the water; the cell probe answers "which pressure unknown lives
 * behind this dot", which is only a question where there is fluid. So the ray
 * wedge is offered by everything solid the ring can reach and the cell wedge
 * only by the water, and neither declaration is repeated in the four entity
 * files that carry them.
 *
 * Both are aimed instruments and the aim is the click's, not the pointer's — see
 * `EditorActionTarget.aim`. Passing the target straight through is what keeps
 * that true: an entity never has to know that a probe has an aim at all.
 */

/**
 * Read the ray behind this pixel.
 *
 * Toned like the frame graph rather than like the thing under the cursor,
 * because that is what it prices: the picture, not the object. The wedge means
 * the same over a stone and over the water, and a wedge that changed colour with
 * whatever it happened to be over would read as a different verb each time.
 */
export function rayProbeAction(target: EditorActionTarget): EditorAction {
  return {
    id: "trace-ray",
    label: "Trace ray",
    icon: "trace-ray",
    tone: "prop",
    hint: "Pin the ray behind this pixel and read the work that drew it · orbit to see it in 3D",
    effect: { kind: "probe", probe: "ray", aim: target.aim },
  };
}

/**
 * Read the pressure cell behind this pixel.
 *
 * Toned like the pipeline and the diagnostic cards, the water's other two
 * instruments, because it answers the same kind of question about the same
 * thing. `[` and `]` walk the ray inward from the cell it pins, which is how an
 * interior unknown is reached — the pixel alone can only ever name the nearest
 * leaf, and on a liquid that is a surface cell.
 */
export function cellProbeAction(target: EditorActionTarget): EditorAction {
  return {
    id: "inspect-cell",
    label: "Inspect cell",
    icon: "inspect-cell",
    tone: "fluid",
    hint: "Pin the pressure cell behind this pixel · [ and ] walk the ray in and out",
    effect: { kind: "probe", probe: "cell", aim: target.aim },
  };
}
