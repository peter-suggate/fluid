"use client";

import { simulation } from "../simulation/controller";
import { resolveSession, type PaneId, type PaneSession } from "../session/session";
import { useShellStore } from "../stores/shell-store";
import {
  promoteCompareDiff,
  shellCompareStore,
  swapComparePanes,
  type CompareAdoptions,
  type CompareMethodSwitch,
  type CompareSceneAdopt,
} from "./compare-model";

/**
 * Entering and leaving the mode, as four verbs.
 *
 * Kept apart from `compare-model.ts` because these are the only compare
 * operations that touch the *host clock*: opening compare is what puts the
 * transport into lockstep, and the model has no business knowing that. Nothing
 * here is reachable from a CPU test, and nothing in the model needs it.
 */

/** The second pane's id. Pane A is the session and is never registered or dropped. */
export const SECOND_PANE_ID: PaneId = "b";

export function compareModeActive(): boolean {
  return useShellStore.getState().compare.active;
}

/**
 * Open the second pane.
 *
 * The clock is not touched beyond the registration: a running scene keeps
 * running and a paused one stays paused, because the mode is a second view of
 * the same experiment rather than a new one. Registering is what puts the host
 * into lockstep — see `PaneClockHost.lockstep`.
 */
export function openCompareMode(sessionB?: PaneSession): void {
  if (compareModeActive()) return;
  useShellStore.getState().setCompareActive(true);
  // The host attaches and registers pane B from its own effect, which is the
  // path a compare *link* takes too. Doing it here as well is deliberate: a
  // caller that already holds the realm should not have to wait a commit for
  // the clock to know about it.
  if (sessionB) simulation.attachPaneSession(SECOND_PANE_ID, sessionB);
  simulation.registerPane(SECOND_PANE_ID);
}

/** Close it, keeping pane A. The diff and the padlocks go with the mode. */
export function closeCompareMode(): void {
  if (!compareModeActive()) return;
  simulation.unregisterPane(SECOND_PANE_ID);
  useShellStore.getState().setCompareActive(false);
}

export function toggleCompareMode(sessionB?: PaneSession): void {
  if (compareModeActive()) closeCompareMode();
  else openCompareMode(sessionB);
}

/**
 * Adopting a solver, rather than merely recording one.
 *
 * The mirror writes a pane's stores directly, which is right for a quality
 * setting and wrong for the method itself: a different solver is different GPU
 * work and a different t=0, and only the controller knows how to rebuild one.
 * The model takes this as a callback so it stays free of the singleton; this is
 * the one implementation, and every surface that mirrors passes it.
 */
export const adoptCompareMethod: CompareMethodSwitch = (session, methodId) => {
  simulation.setMethod(methodId, session.id);
};

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
  onMethodSwitched: adoptCompareMethod,
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
