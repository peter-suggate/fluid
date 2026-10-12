import type { SceneDescription } from "../model";
import type { PaneSession } from "../session/session";

/** Ask the runtime to take `next` in place of `base`. Resolves once it has; rejects with the refusal. */
export type SolidEditAcceptance = (next: SceneDescription, base: SceneDescription) => Promise<void>;

export interface QueuedSolidEdit {
  /** The document the edit was authored against: the queue's head when it is submitted. */
  readonly base: SceneDescription;
  readonly next: SceneDescription;
  /**
   * Write `next` to the pane and record whatever the edit records. Called once,
   * when the runtime has accepted the edit and every edit ahead of it has been
   * written, with `base` still the pane's document.
   */
  commit(): void;
}

/**
 * A pane's solid edits, in the order they were made.
 *
 * The runtime answers an edit some time after it is proposed, and the hand does
 * not wait for it: a second stroke can be pressed, drawn and released while the
 * first is still being accepted. So an edit is authored against the *head* —
 * the pane's document with every edit still in flight applied — and is sent to
 * the runtime at once, behind the ones before it. Their answers come back in
 * that order, and each is written to the pane, with its history entry, in that
 * order too.
 *
 * One refusal ends the chain it belongs to. Everything queued behind a refused
 * edit was authored against a document that will never stand, so each is
 * rejected, saying which refusal cancelled it, and the head falls back to the
 * pane's document.
 *
 * `voxelStrokePending` is this queue's: up while a gesture is open or an edit
 * is in flight.
 */
export interface SolidEditQueue {
  /** The document the next edit is authored against. */
  head(): SceneDescription;
  /** A gesture opened. Call the result when it has closed, however it closed. */
  hold(): () => void;
  submit(edit: QueuedSolidEdit, accept: SolidEditAcceptance): Promise<void>;
}

const queues = new WeakMap<PaneSession, SolidEditQueue>();

export function solidEditQueue(session: PaneSession): SolidEditQueue {
  let queue = queues.get(session);
  if (!queue) { queue = createSolidEditQueue(session); queues.set(session, queue); }
  return queue;
}

function createSolidEditQueue(session: PaneSession): SolidEditQueue {
  let holds = 0, inFlight = 0;
  /** The last document proposed, while edits are in flight. */
  let proposed: SceneDescription | undefined;
  /** Replaced when one of its edits is refused; the edits left on the old one are cancelled. */
  let chain: { refusal?: Error } = {};
  let tail: Promise<void> = Promise.resolve();
  const publishPending = () => {
    const pending = holds > 0 || inFlight > 0;
    if (session.ui.getState().voxelStrokePending !== pending) session.ui.setState({ voxelStrokePending: pending });
  };
  const cancelled = (refusal: Error) => new Error(`Cancelled: the edit before it was refused. ${refusal.message}`);
  return {
    head: () => proposed ?? session.scene.getState().scene,
    hold() {
      holds += 1;
      publishPending();
      let open = true;
      return () => {
        if (!open) return;
        open = false;
        holds -= 1;
        publishPending();
      };
    },
    submit(edit, accept) {
      if ((proposed ?? session.scene.getState().scene) !== edit.base) {
        return Promise.reject(new Error("Scene changed during the stroke."));
      }
      const own = chain;
      proposed = edit.next;
      inFlight += 1;
      publishPending();
      // Sent now, behind the edits already in flight; written when its turn comes.
      let accepted: Promise<void>;
      try { accepted = accept(edit.next, edit.base); } catch (error) { accepted = Promise.reject(error); }
      accepted.catch(() => {});
      const turn = tail.then(async () => {
        try { await accepted; } catch (error) { throw own.refusal ? cancelled(own.refusal) : error; }
        if (own.refusal) throw cancelled(own.refusal);
        if (session.scene.getState().scene !== edit.base) throw new Error("Scene changed during the stroke.");
        edit.commit();
      }).catch((error: unknown) => {
        if (!own.refusal) {
          own.refusal = error instanceof Error ? error : new Error(String(error));
          if (chain === own) { chain = {}; proposed = undefined; }
        }
        throw error;
      }).finally(() => {
        inFlight -= 1;
        if (inFlight === 0) proposed = undefined;
        publishPending();
      });
      tail = turn.catch(() => {});
      return turn;
    },
  };
}
