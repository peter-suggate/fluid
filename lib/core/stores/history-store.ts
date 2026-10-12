import { create } from "zustand";
import { cloneScene, type SceneDescription } from "../model";

/**
 * Undo/redo over whole scene documents.
 *
 * The scene description is the single source of truth and, its solid patch list
 * aside, is small plain JSON, so a snapshot per edit is cheaper and far more
 * robust than an inverse-command log — every authoring path (viewport gizmos,
 * roster buttons, preset loads, imports) becomes undoable without writing an
 * inverse for each. The patch list is shared between snapshots rather than
 * copied; see `historySceneCopy`.
 *
 * Entries record the document *before* an edit. Undo therefore pops the most
 * recent snapshot and pushes the live document onto the redo stack.
 */

export const EDITOR_HISTORY_LIMIT = 64;

/**
 * Consecutive edits that share a coalesce key inside this window collapse into
 * one entry, so dragging a slider or holding a gesture is a single undo rather
 * than one per pointermove.
 */
export const EDITOR_HISTORY_COALESCE_MS = 700;

export interface EditorHistorySnapshot {
  /** What the *following* edit did, used for the undo/redo notice. */
  readonly label: string;
  readonly scene: SceneDescription;
  readonly presetId: string;
}

export interface EditorHistoryRecordOptions {
  /** Gesture identity, e.g. `body:ball-1:position`. Absent never coalesces. */
  readonly coalesceKey?: string;
  /** Injectable clock; production callers use the default wall clock. */
  readonly now_ms?: number;
}

interface EditorHistoryStore {
  past: EditorHistorySnapshot[];
  future: EditorHistorySnapshot[];
  lastCoalesceKey?: string;
  lastRecordedAt_ms: number;
  record: (snapshot: EditorHistorySnapshot, options?: EditorHistoryRecordOptions) => void;
  undo: (current: EditorHistorySnapshot) => EditorHistorySnapshot | undefined;
  redo: (current: EditorHistorySnapshot) => EditorHistorySnapshot | undefined;
  clear: () => void;
}

/**
 * A copy of a document that history can hold, or hand back to the store: every
 * field cloned but the solid patch list, the terrain and the scenery graph,
 * which are shared.
 *
 * The patch list is never written in place. A stroke makes a new array over
 * the same patches, so the entries either side of it can hold those patches
 * once between them. On a sculpted scene the list is nearly the whole
 * document, and cloning it per entry kept sixty-four copies of every patch.
 *
 * The terrain is not written in place either. A sculpt replaces the grid and
 * its samples (`applyTerrainBrush`), an edit to the described ground replaces
 * the description, and the one writer of `baseHeight_m` and `grid`, the scale
 * repair (`repairSceneForContainer`), writes only the deep copy `cloneScene`
 * has just made for it. A sculpted ground is megabytes of samples, cloned here
 * twice per gesture and held once per entry; shared, every entry either side
 * of an edit that left the ground alone holds the one object, and whatever is
 * keyed by that object (its content stamp, its derived grid) is found again
 * after an undo.
 *
 * The scenery graph is the same case and, on a dressed scene, the larger one:
 * megabytes of nodes that a voxel gesture never touches. Every scenery edit
 * goes through `withSceneryNode`, `withSceneryNodes` or `withoutSceneryNode`,
 * which make a new graph, a new node list and a new node; expansion reads the
 * graph and writes nothing; and the expanded catalog is already kept per graph
 * object (`voxel-environments`), which only holds while the graph is never
 * written.
 */
export function historySceneCopy(scene: SceneDescription): SceneDescription {
  // Emptied and nulled where they stand, so each keeps its place among the fields.
  const copy = cloneScene({ ...scene, solidVoxels: [], ...(scene.terrain && { terrain: null }),
    ...(scene.scenery && { scenery: null }) } as unknown as SceneDescription);
  return { ...copy, solidVoxels: scene.solidVoxels, ...(scene.terrain && { terrain: scene.terrain }),
    ...(scene.scenery && { scenery: scene.scenery }) };
}

function owned(snapshot: EditorHistorySnapshot): EditorHistorySnapshot {
  return { label: snapshot.label, scene: historySceneCopy(snapshot.scene), presetId: snapshot.presetId };
}

export const createEditorHistoryStore = () => create<EditorHistoryStore>((set, get) => ({
  past: [],
  future: [],
  lastCoalesceKey: undefined,
  lastRecordedAt_ms: Number.NEGATIVE_INFINITY,
  record: (snapshot, options = {}) => {
    const now_ms = options.now_ms ?? Date.now();
    const state = get();
    const continuing = options.coalesceKey !== undefined
      && options.coalesceKey === state.lastCoalesceKey
      && state.past.length > 0
      && now_ms - state.lastRecordedAt_ms <= EDITOR_HISTORY_COALESCE_MS;
    // A continuing gesture keeps the snapshot taken before it started; only the
    // freshness stamp moves so a held gesture never expires mid-drag.
    if (continuing) {
      set({ lastRecordedAt_ms: now_ms, future: [] });
      return;
    }
    const past = [...state.past, owned(snapshot)];
    set({
      past: past.length > EDITOR_HISTORY_LIMIT ? past.slice(past.length - EDITOR_HISTORY_LIMIT) : past,
      future: [],
      lastCoalesceKey: options.coalesceKey,
      lastRecordedAt_ms: now_ms,
    });
  },
  undo: (current) => {
    const state = get();
    const entry = state.past[state.past.length - 1];
    if (!entry) return undefined;
    set({
      past: state.past.slice(0, -1),
      future: [...state.future, { ...owned(current), label: entry.label }],
      lastCoalesceKey: undefined,
      lastRecordedAt_ms: Number.NEGATIVE_INFINITY,
    });
    return entry;
  },
  redo: (current) => {
    const state = get();
    const entry = state.future[state.future.length - 1];
    if (!entry) return undefined;
    set({
      past: [...state.past, { ...owned(current), label: entry.label }],
      future: state.future.slice(0, -1),
      lastCoalesceKey: undefined,
      lastRecordedAt_ms: Number.NEGATIVE_INFINITY,
    });
    return entry;
  },
  clear: () => set({ past: [], future: [], lastCoalesceKey: undefined, lastRecordedAt_ms: Number.NEGATIVE_INFINITY }),
}));

export type EditorHistoryStoreHook = ReturnType<typeof createEditorHistoryStore>;

/**
 * The default (pane A) instance.
 *
 * Per-pane instances come from `createPaneSession`; this one is what a tree
 * with no `SessionProvider` mounted reads, and what non-React callers that
 * have not yet been threaded a session resolve to.
 */
export const useEditorHistoryStore = createEditorHistoryStore();
