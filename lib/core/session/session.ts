import { createDiagnosticsStore, useDiagnosticsStore, type DiagnosticsStoreHook } from "../stores/diagnostics-store";
import { createEditorHistoryStore, useEditorHistoryStore, type EditorHistoryStoreHook } from "../stores/history-store";
import { createMethodStore, useMethodStore, type MethodStoreHook } from "../stores/method-store";
import { createRecordingStore, useRecordingStore, type RecordingStoreHook } from "../stores/recording-store";
import { createRuntimeStore, useRuntimeStore, type RuntimeStoreHook } from "../stores/runtime-store";
import { createSceneDraftStore, useSceneDraftStore, type SceneDraftStoreHook } from "../stores/scene-draft-store";
import { createSceneStore, useSceneStore, type SceneStoreHook } from "../stores/scene-store";
import { createUIStore, useUIStore, type UIStoreHook } from "../stores/ui-store";

/**
 * One pane's realm.
 *
 * Everything a single simulation pane authors, runs and reports about itself
 * lives in exactly one of these store instances: the scene document and its
 * draft, the method and its overrides, the view state, the transport, the
 * diagnostics it publishes, its undo history, and its recorder. A second pane
 * is a second `PaneSession` with its own set — no code path is duplicated,
 * because single-pane mode is compare mode with one session.
 *
 * What deliberately stays *outside* a session, because it is a property of the
 * page rather than of a pane: `shell-store` (which view the page is showing),
 * `theme-store`, and the `performance-*` stores.
 */
export type PaneId = "a" | "b";

export interface PaneSession {
  /** Which pane this is. `"a"` is the base experiment; `"b"` is A plus a diff. */
  readonly id: PaneId;
  readonly scene: SceneStoreHook;
  readonly method: MethodStoreHook;
  readonly ui: UIStoreHook;
  readonly runtime: RuntimeStoreHook;
  readonly diagnostics: DiagnosticsStoreHook;
  readonly sceneDraft: SceneDraftStoreHook;
  readonly history: EditorHistoryStoreHook;
  readonly recording: RecordingStoreHook;
}

/** A fresh, independent realm. Every store starts at its authored initial state. */
export function createPaneSession(id: PaneId): PaneSession {
  return {
    id,
    scene: createSceneStore(),
    method: createMethodStore(),
    ui: createUIStore(),
    runtime: createRuntimeStore(),
    diagnostics: createDiagnosticsStore(),
    sceneDraft: createSceneDraftStore(),
    history: createEditorHistoryStore(),
    recording: createRecordingStore(),
  };
}

/**
 * Pane A, built from the module-level default store instances.
 *
 * These are the same objects the store modules export, so a caller that has not
 * yet been threaded a session and one that reads `useSession()` in single-pane
 * mode are looking at the same state. That equality is what lets WP0 land as a
 * pure refactor.
 */
export const defaultSession: PaneSession = {
  id: "a",
  scene: useSceneStore,
  method: useMethodStore,
  ui: useUIStore,
  runtime: useRuntimeStore,
  diagnostics: useDiagnosticsStore,
  sceneDraft: useSceneDraftStore,
  history: useEditorHistoryStore,
  recording: useRecordingStore,
};

/**
 * The session for a caller that cannot reach React context.
 *
 * Non-React modules — the URL writer, autosave, the recorder, the editor action
 * runtime — still bind to pane A. Each such call site is a place WP2/WP3 must
 * hand the real session in; routing them all through this one function is what
 * makes them findable.
 */
export function resolveSession(): PaneSession {
  return defaultSession;
}
