"use client";

import { getEditorGesture, type EditorGestureId } from "../lib/core/editor-gesture-catalog";
import { useSession } from "../lib/core/session/session-context";

/**
 * Arming a stroke from a toolstrip row, for whichever host owns the strip.
 *
 * Lifted out of `MakeRows` when the 2-D advance lab grew an EDIT strip of its
 * own. Both hosts want the same two things and neither of them is about the
 * scene document: a row that reports whether its gesture is armed, and a hint
 * taken from the gesture's own declaration rather than paraphrased beside it.
 * Keeping them here means the lab reaches the gesture catalog without also
 * reaching `performEditorAction` — and through it the 3-D `simulation`
 * singleton, which the lab has no world for.
 */

/**
 * Arm state for one stroke, and the toggle that owns it.
 *
 * Written against the store rather than through `performEditorAction`, because
 * the effect union has an `arm` and no disarm: putting a stroke away is not
 * something a wedge can express — a ring closes on the choice, so it never
 * needed to — and it is half of what a row means.
 */
export function useArmedStroke(gesture: EditorGestureId) {
  const session = useSession();
  const armed = session.ui((state) => state.armedGesture) === gesture;
  const setArmedGesture = session.ui((state) => state.setArmedGesture);
  return { armed, toggle: () => setArmedGesture(armed ? undefined : gesture) };
}

/**
 * One line of the gesture's own hint, plus how to put it away.
 *
 * The catalog's hints are written for the chip under an armed mode and run to
 * three clauses; the row's tip clamps at three short lines. Taking the first
 * clause keeps the sentence the author wrote for the stroke rather than a second
 * paraphrase of it that can drift.
 */
export function strokeHint(gesture: EditorGestureId, armed: boolean): string {
  const [first = ""] = getEditorGesture(gesture).hint.split(" · ");
  const sentence = `${first.slice(0, 1).toUpperCase()}${first.slice(1)}.`;
  return armed ? `${sentence} Click the mark again to put it away.` : sentence;
}
