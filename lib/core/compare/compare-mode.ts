"use client";

import { simulation } from "../simulation/controller";
import { resolveSession, type PaneId, type PaneSession } from "../session/session";
import { useShellStore } from "../stores/shell-store";
import {
  promoteCompareDiff,
  shellCompareStore,
  swapComparePanes,
  type CompareAdoptions,
  type CompareSceneAdopt,
} from "./compare-model";

/**
 * Entering and leaving the mode, as four verbs.
 *
 * Kept apart from `compare-model.ts` because these are the compare operations
 * that reach the *host*: opening compare is what puts the transport into
 * lockstep (through `CompareHost`, the pane's one membership owner), keeping
 * or swapping a pane re-seeds it, and the model has no business knowing that.
 */

/** The second pane's id. Pane A is the session and is never registered or dropped. */
export const SECOND_PANE_ID: PaneId = "b";

export function compareModeActive(): boolean {
  return useShellStore.getState().compare.active;
}

/**
 * Open the second pane.
 *
 * Only the record changes here. Pane B joins the host — its realm and its place
 * on the clock, in one `simulation.attachPane` — from `CompareHost`'s effect,
 * the one path a compare *link* takes too. The clock is not otherwise touched:
 * a running scene keeps running and a paused one stays paused, because the mode
 * is a second view of the same experiment rather than a new one. Joining is
 * what puts the host into lockstep — see `PaneClockHost.lockstep`.
 */
export function openCompareMode(): void {
  if (compareModeActive()) return;
  useShellStore.getState().setCompareActive(true);
}

/** Close it, keeping pane A. The diff and the padlocks go with the mode. */
export function closeCompareMode(): void {
  if (!compareModeActive()) return;
  useShellStore.getState().setCompareActive(false);
}

export function toggleCompareMode(): void {
  if (compareModeActive()) closeCompareMode();
  else openCompareMode();
}

/**
 * Adopting a document, rather than merely holding one.
 *
 * The mirror's document write is the same shape of half-measure the method
 * write was: the receiving pane draws the new scene and keeps simulating the
 * old one, because its rigid roster and its solver were built from the document
 * it was started on. `adoptSceneEdit` is the controller's own reconciliation —
 * bodies adopted, and a re-seed only when the edit crosses the tier boundary,
 * which is exactly what the edited pane paid.
 */
export const adoptCompareScene: CompareSceneAdopt = (session, previous) => {
  simulation.adoptSceneEdit(previous, session.id);
};

/**
 * Both adoptions, as the one object every mirroring surface passes.
 *
 * A mirror missing either of these is silently half-applied — the pane looks
 * right and runs something else — so they travel together rather than being
 * remembered one call site at a time.
 */
export const COMPARE_ADOPTIONS: CompareAdoptions = {
  onSceneAdopted: adoptCompareScene,
};

/** "Keep this one" on pane B: A ← A ⊕ diff, then the mode closes. */
export function keepComparePaneB(a: PaneSession = resolveSession()): void {
  promoteCompareDiff(shellCompareStore(), a, COMPARE_ADOPTIONS);
  closeCompareMode();
}

/** "Swap": B's values become A's and A's old values become the diff. */
export function swapComparePanesInPlace(a: PaneSession = resolveSession()): void {
  swapComparePanes(shellCompareStore(), a, COMPARE_ADOPTIONS);
}
