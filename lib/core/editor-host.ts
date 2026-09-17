import type { EditorGestureId } from "./editor-gesture-catalog";
import type { EditorSelection } from "./editor-tools";

/**
 * The world a capability module is editing, as that module is allowed to see it.
 *
 * Every UI file in this repo that could not be rendered by a second host was one
 * that imported the module-level `simulation` singleton. That is the whole of
 * the coupling: the rows, the rings and the gestures are already data, and the
 * only studio-specific thing about them is who they hand a commit to. This
 * interface is that "who", reached from the session exactly as the stores are,
 * so one colocated capability module renders in the 3-D studio and in the 2-D
 * advance lab without either host learning about the other.
 *
 * It is capability-scoped rather than one document-shaped `commitEdit`, because
 * the lab has no document: its world is a Rust `World` behind
 * `AdvanceLabController.setRefinementRegions` / `injectLiquid`, both async and
 * both whole-list. Forcing a merge-patch shape there would mean inventing a
 * fake document and a reducer for it. So a host declares what it can honour and
 * a row needing a capability the host lacks does not render — the same bargain
 * `SimulationMethod.capabilities` makes one level up.
 *
 * `Doc` is whatever the host calls a whole document and `Patch` whatever it
 * calls a merge into one. In the studio those are `SceneDescription` and
 * `Partial<SceneDescription>`; in the lab they are the same type, because a
 * whole-list command is the only write it has.
 */
export interface EditorCommitOptions {
  /** The edit moved the lattice; the run restarts from a defined t = 0. */
  readonly reseed?: boolean;
  /** Names the pause a lattice rebuild costs, for the host's own notice. */
  readonly announceRebuild?: string;
}

/** An open gesture's transient document — a drag, before the pointer comes up. */
export interface EditorHostDraft {
  begin(subject: string, label: string): void;
  commit(options?: EditorCommitOptions): void;
  cancel(): void;
}

export interface EditorHostHistory {
  undo(): boolean;
  redo(): boolean;
}

/** Placing liquid, in the host's own frame — the lab has no scene to patch. */
export interface EditorHostLiquid {
  dropAt(centre: readonly number[], radius: number): void;
}

/** Feature-control values, keyed by `FeatureControl.setting`. */
export interface EditorHostParams {
  get(key: string): number | string | boolean | undefined;
  set(key: string, value: number | string | boolean): void;
}

export interface EditorHost<Doc = unknown, Patch = Doc> {
  /**
   * Which world this is. The studio passes its `PaneId` through, so a host
   * built for pane B addresses pane B and nothing else.
   */
  readonly id: string;
  /** A whole document, under one history label. */
  commit(label: string, next: Doc, options?: EditorCommitOptions): void;
  /**
   * A merge patch, under one history label.
   *
   * Absent on a host with no merge to express. A row that has only a patch to
   * offer falls back to reading the whole document and committing it, which is
   * what `commit` is for.
   */
  commitPatch?(label: string, patch: Patch, options?: EditorCommitOptions): void;
  draft?: EditorHostDraft;
  history?: EditorHostHistory;
  liquid?: EditorHostLiquid;
  params?: EditorHostParams;
  select(selection: EditorSelection | undefined, openControls?: boolean): void;
  arm(gesture: EditorGestureId | undefined): void;
  notice(text: string, tone?: "info" | "warn"): void;
}
