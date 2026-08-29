import type { RunState } from "../model";

/**
 * The transport is the host's, not a pane's.
 *
 * There is one target clock (`PaneClockHost`) and one transport over it. Play,
 * pause, STEP and RESET are therefore statements about the *experiment*, and
 * every attached pane hears them at the same instant. A gesture that reached
 * only pane A left the product in a state it has no way to describe: pane B's
 * renderer never learned the clock had stopped, so it never took the pause
 * boundary its stats readback rides on, and — the failure that actually kills
 * the mode — it never reported the work it had already admitted to its queue,
 * so the host could rewind the target below an advance pane B had encoded and
 * would then wait forever for a completion no solver will ever re-encode.
 *
 * The rules live here, apart from the controller, for one reason: the
 * controller is the singleton that imports the method registry, the renderer
 * and every store, so nothing in it can be reached from a CPU test. These
 * functions are typed structurally against the half of a pane's realm a
 * transport gesture touches, which `Pick<PaneSession, "id" | "runtime">`
 * satisfies without this module ever naming a session.
 */

/** The transport half of one pane's runtime store. */
export interface TransportRuntimeState {
  readonly runState: RunState;
  setRunState: (runState: RunState) => void;
}

export interface TransportRuntimeStore {
  getState: () => TransportRuntimeState;
}

/** One pane, as the transport sees it. */
export interface TransportPane {
  readonly id: string;
  readonly runtime: TransportRuntimeStore;
}

/**
 * Play/pause: one transport, every pane.
 *
 * Panes already in the asked-for state are skipped, so a fan-out is idempotent
 * and a pane that is re-attached mid-run is not told to do what it is doing.
 */
export function applyHostRunState(panes: Iterable<TransportPane>, runState: RunState): void {
  for (const pane of panes) {
    const runtime = pane.runtime.getState();
    if (runtime.runState === runState) continue;
    runtime.setRunState(runState);
  }
}

/**
 * One pane's share of a reset.
 *
 * `resynchronizing` marks a pane re-seeded as a *consequence* of someone else's
 * reset rather than as the subject of one: its clock has to return to zero with
 * the host's, but nobody asked it to stop, to drop its selection, or to say so
 * in the notice line.
 */
export interface PaneResetStep<Id extends string = string> {
  readonly id: Id;
  readonly resynchronizing: boolean;
}

/**
 * A reset of one pane, as it reaches all of them.
 *
 * `PaneClockHost.reset` returns *every* pane's completion to zero, because
 * there is one clock. A pane whose document runtime and renderer timeline were
 * not re-seeded with it is then stranded above a target that has rewound past
 * it: its solver refuses to re-encode a time it has already submitted, so it
 * can never report the completion the min-completion barrier is waiting for,
 * and the transport dies for both panes. Re-seeding the others is not a
 * courtesy — it is what makes the one-clock claim true.
 *
 * The pane that was asked pauses, as a reset always has. The others keep the
 * run state they were in, so choosing a scene in pane B does not stop pane A.
 */
export function paneResetPlan<Id extends string>(paneIds: Iterable<Id>, paneId: Id): PaneResetStep<Id>[] {
  const plan: PaneResetStep<Id>[] = [{ id: paneId, resynchronizing: false }];
  for (const id of paneIds) {
    if (id === paneId) continue;
    plan.push({ id, resynchronizing: true });
  }
  return plan;
}

/**
 * The transport's RESET: every pane back to its own t = 0, and stopped.
 *
 * Not `paneResetPlan` for each pane in turn — that would restore a pane's run
 * state from a pass that had already paused it. RESET is one gesture with one
 * outcome: the whole experiment at zero, waiting.
 */
export function hostResetPlan<Id extends string>(paneIds: Iterable<Id>): PaneResetStep<Id>[] {
  return [...paneIds].map((id) => ({ id, resynchronizing: false }));
}
