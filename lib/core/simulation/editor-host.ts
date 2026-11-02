import type { EditorCommitOptions, EditorHost } from "../editor-host";
import type { MethodParamValue } from "../method-contract";
import type { SceneDescription } from "../model";
import type { PaneSession } from "../session/session";
import { resolvedMethodValues } from "../stores/method-store";
import type { SceneDraftSubject } from "../stores/scene-draft-store";
import { simulation, type SimulationController } from "./controller";

/**
 * The studio's fulfilment of `EditorHost` — and the only new importer of the
 * `simulation` singleton.
 *
 * Nothing here is new behaviour. Each member is the exact call the UI already
 * made through the singleton, moved behind the seam so a capability module can
 * be rendered by a second host: `commit` and `commitPatch` are one `edit`,
 * which is the whole of what a `{kind:"scene"}` effect does
 * (`editor-action-runtime.ts`), and a draft is the gesture handle
 * `beginDraft` returns.
 *
 * The controller arrives as a parameter with the singleton as its default, so
 * a test can watch which pane each call addresses.
 */
export type StudioEditorHostController = Pick<SimulationController, "edit" | "beginDraft" | "undo" | "redo" | "setMethodParam">;

export function studioEditorHost(
  session: PaneSession,
  controller: StudioEditorHostController = simulation,
): EditorHost<SceneDescription, Partial<SceneDescription>> {
  const ui = () => session.ui.getState();
  return {
    id: session.id,
    commit: (label: string, next: SceneDescription, options?: EditorCommitOptions) => {
      controller.edit(label, () => session.scene.getState().setScene(next), { reseed: options?.reseed }, session.id);
    },
    commitPatch: (label: string, patch: Partial<SceneDescription>, options?: EditorCommitOptions) => {
      controller.edit(label, patch, { reseed: options?.reseed ?? true }, session.id);
    },
    draft: {
      begin: (subject: string, label: string) => {
        const gesture = controller.beginDraft(subject as SceneDraftSubject, label, session.id);
        return gesture && {
          // A draft re-seeds unless the caller says otherwise, as a patch does.
          commit: (options?: EditorCommitOptions) => { gesture.commit({ ...options, reseed: options?.reseed ?? true }); },
          cancel: () => gesture.cancel(),
        };
      },
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
