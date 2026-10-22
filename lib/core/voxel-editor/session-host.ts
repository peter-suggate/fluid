import type { SceneDescription } from "../model";
import type { PaneSession } from "../session/session";
import type { SolidWorldVoxelPatch } from "../solid-world";
import type { ToolAction } from "./plugin";
import { solidEditQueue } from "./solid-edit-queue";
import type { ToolHost } from "./transaction";
import { combinedRegion } from "./voxel-grid";

/**
 * A pane's stores as the host of a solid edit.
 *
 * One definition of what "commit" means for this editor, shared by everything
 * that proposes solid patches — a stroke when the pointer lets go, a selection
 * verb when it is chosen — so the two cannot come to differ in what they
 * preflight, what they record or how they fail:
 *
 * - the edit is authored against the pane's queue head (`solid-edit-queue`):
 *   its document with every edit still being accepted applied, so a stroke
 *   pressed before the last one was answered is never turned away.
 * - `publish` hands the proposed document to `validate`, the renderer's live
 *   solid-edit preflight, at once, and writes the scene store when that
 *   accepted it and the edits ahead of it are written. A capacity refusal is
 *   therefore a rejected edit with its own message, never a fault raised later
 *   by a renderer drawing a document it could not hold.
 * - the document the edit was authored against is recorded once, under the
 *   edit's label, in the same step that writes the edit.
 * - `preview` shows a gesture's proposal through the display alone; nothing is
 *   written, recorded or handed to the simulation.
 * - a live gesture publishes accepted samples throughout its drag, recording
 *   the starting document once on finish. Cancel validates a rollback before
 *   releasing the hold and leaves both history stacks unchanged.
 * - `voxelStrokePending` is up from `begin` until the edit has been written or
 *   refused, which is what keeps undo, import and a save out meanwhile.
 */
export function sessionToolHost(session: PaneSession,
  validate: (scene: SceneDescription, base: SceneDescription) => Promise<void>,
  execute?: (action: ToolAction) => Promise<void>,
  preview?: (patches: readonly SolidWorldVoxelPatch[] | undefined, base: SceneDescription) => void): ToolHost {
  const queue = solidEditQueue(session);
  let label: string | undefined;
  let close: (() => void) | undefined;
  let live: { base: SceneDescription; latest: SceneDescription; presetId: string; label: string } | undefined;
  const release = () => { close?.(); close = undefined; label = undefined; live = undefined; };
  const recordLive = () => {
    if (live && live.latest !== live.base) session.history.getState().record({
      label: live.label, scene: live.base, presetId: live.presetId,
    });
  };
  return {
    scene: () => queue.head(),
    publish: (next, base) => {
      const recorded = label;
      const stroke = live;
      if (recorded === undefined) return Promise.reject(new Error("A solid edit was published without having begun."));
      return queue.submit({ base, next, commit: () => {
        const store = session.scene.getState();
        const presetId = store.presetId;
        store.setScene(next);
        if (stroke) stroke.latest = next;
        else session.history.getState().record({ label: recorded, scene: base, presetId });
      } }, validate);
    },
    preview,
    execute,
    select: (region, mode = "replace") => {
      const ui = session.ui.getState();
      const standing = ui.selection?.kind === "voxel-region" ? ui.voxelRegion : undefined;
      // Adding to nothing is selecting; taking from nothing leaves nothing.
      if (mode === "replace" || !standing) { if (mode !== "subtract") ui.selectVoxelRegion(region); return; }
      ui.selectVoxelRegion(combinedRegion(standing, region, mode));
    },
    clipboard: () => session.ui.getState().voxelClipboard,
    begin: (edit, options) => {
      label = edit; close ??= queue.hold();
      if (options?.live) live = { base: queue.head(), latest: queue.head(),
        presetId: session.scene.getState().presetId, label: edit };
    },
    finish: () => { recordLive(); release(); },
    cancel: async () => {
      try {
        // Never replace an unrelated document changed by another editor.
        if (live && live.latest !== live.base && queue.head() === live.latest) {
          const stroke = live;
          await queue.submit({ base: stroke.latest, next: stroke.base,
            commit: () => { session.scene.getState().setScene(stroke.base, stroke.presetId); } }, validate);
        }
      } catch (error) {
        // A lost runtime may refuse rollback. Keep the accepted stroke undoable
        // rather than leaving changed solids with no history entry.
        recordLive();
        throw error;
      } finally { release(); }
    },
  };
}
