import { canonicalScene, type SceneDescription } from "./model";
import { savedSceneCard } from "./scene-cards";
import type { SceneCard } from "./scene-definition";
import {
  isSceneAutosaveEntry,
  loadSceneFromLibrary,
  saveSceneToLibrary,
  SCENE_AUTOSAVE_ENTRY_ID,
  type SceneLibraryEntry,
  type SceneLibraryStorage,
} from "./scene-library";
import { findSceneDefinition } from "./scenes";
import { useSceneStore } from "./stores/scene-store";
import { useShellStore } from "./stores/shell-store";

/**
 * The working document, kept so that closing the tab is not destructive.
 *
 * Everything a person authors between two explicit saves used to exist only in
 * a store, which made the library a wall between them and what they were doing:
 * *Continue* could offer nothing but scenes they had already thought to name.
 *
 * This writes the live document into the same storage as a saved scene, under a
 * reserved id, because it is the same artifact — it round-trips through
 * `parseScene` and reopens through the same card as any saved entry. What it is
 * not is a scene the reader saved: it never appears on the *My scenes* shelf,
 * and an explicit save is never written over by it.
 */

/** How long the edits must stop before the document is written. */
export const SCENE_AUTOSAVE_DEBOUNCE_MS = 1_000;

/** What an autosave is called when its origin scene is not this build's. */
export const SCENE_AUTOSAVE_NAME = "Untitled scene";

export interface SceneAutosaveDocument {
  readonly scene: SceneDescription;
  readonly presetId: string;
}

/**
 * The name to file the working document under.
 *
 * The scene it was opened from, when this build still has it: *Continue —
 * Garden pond* names the thing the reader was looking at, which a generic
 * "Untitled scene" does not. A starter or a scene from an older build has no
 * definition to borrow a name from.
 */
export function sceneAutosaveName(presetId: string): string {
  return findSceneDefinition(presetId)?.name ?? SCENE_AUTOSAVE_NAME;
}

/**
 * Write the working document to the reserved entry.
 *
 * `replaceId` rather than the name match `saveSceneToLibrary` would otherwise
 * make: the autosave and an explicit save can carry the same name, and this
 * must land on its own entry either way.
 */
export function writeSceneAutosave(
  storage: SceneLibraryStorage | undefined,
  working: SceneAutosaveDocument,
  savedAt_ms: number,
): SceneLibraryEntry {
  return saveSceneToLibrary(storage, sceneAutosaveName(working.presetId), working.scene, working.presetId, {
    savedAt_ms,
    replaceId: SCENE_AUTOSAVE_ENTRY_ID,
  }).entry;
}

export function readSceneAutosave(entries: readonly SceneLibraryEntry[]): SceneLibraryEntry | undefined {
  return entries.find(isSceneAutosaveEntry);
}

export interface SceneResume {
  readonly entry: SceneLibraryEntry;
  readonly card: SceneCard;
  /** Where the reader left off, rather than something they chose to keep. */
  readonly autosaved: boolean;
}

/**
 * What *Continue* offers, or nothing.
 *
 * The working document wins over the newest explicit save even when the save is
 * more recent, because the question the hero answers is "where was I", not
 * "what did I keep". A candidate is only offered once its document has been
 * read: a stored entry is validated on load, so an autosave written by an older
 * schema must degrade to no Continue at all rather than to a hero button that
 * throws — and the reader's own newest save is a better answer than nothing.
 */
export function sceneResume(entries: readonly SceneLibraryEntry[]): SceneResume | undefined {
  const autosave = readSceneAutosave(entries);
  const explicit = entries.filter((entry) => !isSceneAutosaveEntry(entry))
    .sort((left, right) => right.savedAt_ms - left.savedAt_ms);
  for (const entry of autosave ? [autosave, ...explicit] : explicit) {
    try {
      loadSceneFromLibrary(entry);
    } catch {
      continue;
    }
    return { entry, card: savedSceneCard(entry), autosaved: isSceneAutosaveEntry(entry) };
  }
  return undefined;
}

export interface SceneAutosaveOptions {
  readonly storage?: SceneLibraryStorage;
  readonly delay_ms?: number;
  readonly now?: () => number;
}

export interface SceneAutosave {
  /** Note a document change. The write happens once the edits stop. */
  request(working: SceneAutosaveDocument): void;
  /** Write anything outstanding now, for a tab that is going away. */
  flush(): void;
  cancel(): void;
}

/**
 * A debounced writer.
 *
 * A gesture reaches the document many times — the tank resizer patches per
 * pointer-move — and serializing the whole scene on each one would put a
 * `JSON.stringify` of the terrain grid on the drag path. Waiting for the edits
 * to stop costs at most the last second of work and nothing during the drag.
 */
export function createSceneAutosave(options: SceneAutosaveOptions = {}): SceneAutosave {
  const delay_ms = options.delay_ms ?? SCENE_AUTOSAVE_DEBOUNCE_MS;
  const now = options.now ?? Date.now;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: SceneAutosaveDocument | undefined;
  let written: string | undefined;

  const write = () => {
    timer = undefined;
    const working = pending;
    pending = undefined;
    if (!working) return;
    // Selection, panels and the shell all touch stores that reach this; a
    // document that has not actually changed is not worth a write.
    const identity = `${working.presetId}\n${canonicalScene(working.scene)}`;
    if (identity === written) return;
    written = identity;
    writeSceneAutosave(options.storage, working, now());
  };

  return {
    request(working) {
      pending = working;
      if (timer !== undefined) clearTimeout(timer);
      timer = setTimeout(write, delay_ms);
    },
    flush() {
      if (timer !== undefined) clearTimeout(timer);
      write();
    },
    cancel() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      pending = undefined;
    },
  };
}

/**
 * Autosave for as long as the app is up. Returns the stop function.
 *
 * Subscribed to the stores rather than threaded through the controller, as
 * `startQueryStateSync` is, because every path that changes the document
 * already ends in `useSceneStore` and none of them should have to remember
 * this. `studioEntered` is the gate: on a cold load the store holds the default
 * preset that nobody chose, and offering that as *Continue* would be a lie.
 * Entering the studio is itself a trigger, so a scene opened and left alone is
 * still where the reader is.
 */
export function startSceneAutosave(options: SceneAutosaveOptions = {}): () => void {
  const autosave = createSceneAutosave(options);
  const schedule = () => {
    if (!useShellStore.getState().studioEntered) return;
    const { scene, presetId } = useSceneStore.getState();
    autosave.request({ scene, presetId });
  };
  // A closing tab is exactly the case this exists for, and it never runs an
  // unmount; `visibilitychange` is the last callback a browser reliably gives.
  const onHidden = () => { if (document.visibilityState === "hidden") autosave.flush(); };
  const stopScene = useSceneStore.subscribe(schedule);
  const stopShell = useShellStore.subscribe(schedule);
  if (typeof document !== "undefined") document.addEventListener("visibilitychange", onHidden);
  schedule();

  return () => {
    stopScene();
    stopShell();
    if (typeof document !== "undefined") document.removeEventListener("visibilitychange", onHidden);
    autosave.flush();
  };
}
