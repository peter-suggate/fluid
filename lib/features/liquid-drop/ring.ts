import type { EditorAction, EditorActionEffect } from "../../core/editor-action";

/**
 * Putting liquid into a running world, as ring wedges — for either host.
 *
 * Both pages offer a ball of water at a point, and until now both composed the
 * wedge for it by hand: `lib/core/editor-fluid-body.ts` inside `fluidPlayActions`
 * and `advance-lab/slice-actions.ts` inside its own `waterWedge`, with two ids,
 * two tones written out and two hints. Nothing about "a ball of water goes
 * here" is 3-D, and nothing about it is the studio's.
 *
 * What genuinely differs between the two is the **effect**, and it differs in a
 * way neither host could fake for the other: the studio *arms a gesture*,
 * because a ball there is sized by dragging it out in a perspective view; the
 * lab *lands one now*, at the cell the press was over, because the ring there
 * opened on a cell it already knows. So the effect is the caller's and the
 * wedge around it is not — the same split `entityDeleteWedge` makes.
 *
 * The container is shared too. A root ring holding one wedge per *kind* of
 * intention, each opening the ring of its verbs, is the two-level doctrine; a
 * page that grouped its water verbs differently would be a page where the flick
 * a reader learned in one does not work in the other.
 */

/** The gesture a drag over the water means when a ball is armed. */
export const LIQUID_BALL_GESTURE = "fluid-ball" as const;

/**
 * A ball of liquid, as one wedge.
 *
 * `hint` is the host's because the two promise different things — one says what
 * a drag will do next, the other says something has already happened and the
 * stroke is still armed — and a hint is the sentence under the ring, which has
 * to be true of the page the reader is on.
 */
export function liquidBallWedge(options: {
  readonly effect?: EditorActionEffect;
  readonly label?: string;
  readonly hint: string;
  readonly enabled?: boolean;
}): EditorAction {
  return {
    id: "ball",
    label: options.label ?? "Ball",
    icon: "water-ball",
    tone: "fluid",
    hint: options.hint,
    ...(options.enabled === undefined ? {} : { enabled: options.enabled }),
    ...(options.effect ? { effect: options.effect } : {}),
  };
}

/**
 * The water wedge: one kind of intention, opening the ring of its verbs.
 *
 * The children are the host's, because what can be done to water is not the
 * same in a 3-D document and on a 2-D cut of a running Rust world — the studio
 * paints and erases bricks and aims a hose, none of which the lab has. The
 * slot in the root ring, its word and its tone are not the host's.
 */
export function liquidWedge(children: readonly EditorAction[]): EditorAction {
  return {
    id: "water",
    label: "Water",
    icon: "water-ball",
    tone: "fluid",
    hint: "Add, paint, erase or pour water here",
    children,
  };
}
