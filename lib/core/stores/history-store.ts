import { create } from "zustand";
import { cloneScene, type SceneDescription } from "../model";

/**
 * Undo/redo over whole scene documents.
 *
 * The scene description is the single source of truth and is small plain JSON,
 * so a snapshot per edit is cheaper and far more robust than an inverse-command
 * log — every authoring path (viewport gizmos, roster buttons, preset loads,
 * imports) becomes undoable without writing an inverse for each.
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

function owned(snapshot: EditorHistorySnapshot): EditorHistorySnapshot {
  return { label: snapshot.label, scene: cloneScene(snapshot.scene), presetId: snapshot.presetId };
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
