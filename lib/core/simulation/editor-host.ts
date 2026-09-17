import type { EditorCommitOptions, EditorHost } from "../editor-host";
import type { MethodParamValue } from "../method-contract";
import type { SceneDescription } from "../model";
import type { PaneId, PaneSession } from "../session/session";
import { resolvedMethodValues } from "../stores/method-store";
import type { SceneDraftSubject } from "../stores/scene-draft-store";
import { simulation } from "./controller";

/**
 * The studio's fulfilment of `EditorHost` — and the only new importer of the
 * `simulation` singleton.
 *
 * Nothing here is new behaviour. Each member is the exact call sequence the UI
 * already made through the singleton, moved behind the seam so a capability
 * module can be rendered by a second host: `commit` is
 * `beginEdit → setScene → commitEdit`, which is the whole of what a
 * `{kind:"scene"}` effect does (`editor-action-runtime.ts`), and `commitPatch`
 * is the two-call form `EntityOptions` has always used.
 *
 * The controller arrives as a parameter with the singleton as its default. That
 * is not dependency injection for its own sake: the order of the three calls in
 * `commit` *is* the contract — a `setScene` before the `beginEdit` would snapshot
 * the post-edit document and silently make the edit un-undoable — and the order
 * is only observable to a test that can watch the calls.
 */
export interface StudioEditorHostController {
  beginEdit(label: string, paneId?: PaneId): void;
  commitEdit(
    patch?: Partial<SceneDescription>,
    options?: { reseed?: boolean },
    paneId?: PaneId,
  ): boolean;
  beginDraft(subject: SceneDraftSubject, label: string, paneId?: PaneId): void;
  commitDraft(
    options?: { announceRebuild?: string; reseed?: boolean },
    paneId?: PaneId,
  ): boolean;
  cancelDraft(paneId?: PaneId): void;
  undo(paneId?: PaneId): boolean;
  redo(paneId?: PaneId): boolean;
  setMethodParam(methodId: string, key: string, value: MethodParamValue, paneId?: PaneId): void;
}

export function studioEditorHost(
  session: PaneSession,
  controller: StudioEditorHostController = simulation,
): EditorHost<SceneDescription, Partial<SceneDescription>> {
  const ui = () => session.ui.getState();
  return {
    id: session.id,
    commit: (label: string, next: SceneDescription, options?: EditorCommitOptions) => {
      controller.beginEdit(label, session.id);
      session.scene.getState().setScene(next);
      controller.commitEdit(undefined, { reseed: options?.reseed }, session.id);
    },
    commitPatch: (label: string, patch: Partial<SceneDescription>, options?: EditorCommitOptions) => {
      controller.beginEdit(label, session.id);
      controller.commitEdit(patch, { reseed: options?.reseed ?? true }, session.id);
    },
    draft: {
      begin: (subject: string, label: string) =>
        controller.beginDraft(subject as SceneDraftSubject, label, session.id),
      commit: (options?: EditorCommitOptions) => {
        controller.commitDraft({
          reseed: options?.reseed,
          announceRebuild: options?.announceRebuild,
        }, session.id);
      },
      cancel: () => controller.cancelDraft(session.id),
    },
    history: {
      undo: () => controller.undo(session.id),
      redo: () => controller.redo(session.id),
    },
    params: {
      get: (key: string) => resolvedMethodValues(session.method.getState())[key],
      set: (key: string, value: MethodParamValue) =>
        controller.setMethodParam(session.method.getState().methodId, key, value, session.id),
    },
    select: (selection, openControls) => {
      ui().select(selection);
      if (openControls) ui().setSelectionControlsOpen(true);
    },
    arm: (gesture) => ui().setArmedGesture(gesture),
    notice: (text: string, tone: "info" | "warn" = "info") =>
      session.runtime.getState().setNotice(text, tone),
  };
}
