import { cloneScene, parseScene, serializeScene, type SceneDescription } from "./model";

/**
 * Named local scene storage.
 *
 * Authoring is worthless if a reload discards it, and the URL cannot carry a
 * sculpted terrain grid or a large seed list. Entries are stored as the same
 * serialized JSON the file export writes, so a library entry and a downloaded
 * scene are the same artifact and both round-trip through `parseScene`.
 *
 * The storage handle is injected so the module is testable without a DOM and
 * degrades to a no-op when storage is unavailable (private browsing, quota).
 */

export const SCENE_LIBRARY_STORAGE_KEY = "fluid-lab.scene-library.v1";
export const SCENE_LIBRARY_LIMIT = 64;
export const SCENE_NAME_MAXIMUM_LENGTH = 80;

/**
 * The working document's reserved identity — see `lib/scene-autosave.ts`.
 *
 * It is stored here because it is the same artifact as a saved scene, but it is
 * not one the reader saved: the name match below skips it, so saving explicitly
 * under whatever name the autosave happens to carry writes the reader's own
 * entry instead of adopting this one.
 */
export const SCENE_AUTOSAVE_ENTRY_ID = "autosave";

export interface SceneLibraryEntry {
  readonly id: string;
  readonly name: string;
  /** Epoch milliseconds; supplied by the caller so the module stays pure. */
  readonly savedAt_ms: number;
  readonly presetId: string;
  /** Serialized `SceneDescription`, byte-identical to the file export. */
  readonly scene: string;
}

export interface SceneLibraryStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** `localStorage` when the platform has it, else undefined. */
export function browserSceneLibraryStorage(): SceneLibraryStorage | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage;
  } catch {
    // Storage access throws outright under some privacy settings.
    return undefined;
  }
}

export function normalizeSceneName(name: string): string {
  const trimmed = name.trim().replace(/\s+/g, " ");
  return trimmed.slice(0, SCENE_NAME_MAXIMUM_LENGTH) || "Untitled scene";
}

/** A stable id from the name plus the save time; collisions are disambiguated on write. */
function sceneEntryId(name: string, savedAt_ms: number): string {
  const slug = normalizeSceneName(name).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `${slug || "scene"}-${savedAt_ms.toString(36)}`;
}

function isEntry(value: unknown): value is SceneLibraryEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<SceneLibraryEntry>;
  return typeof entry.id === "string" && typeof entry.name === "string"
    && typeof entry.savedAt_ms === "number" && Number.isFinite(entry.savedAt_ms)
    && typeof entry.presetId === "string" && typeof entry.scene === "string";
}

export function isSceneAutosaveEntry(entry: SceneLibraryEntry): boolean {
  return entry.id === SCENE_AUTOSAVE_ENTRY_ID;
}

/** The entries a reader chose to keep; the autosave is not one of them. */
export function savedSceneEntries(entries: readonly SceneLibraryEntry[]): SceneLibraryEntry[] {
  return entries.filter((entry) => !isSceneAutosaveEntry(entry));
}

/** Malformed storage is treated as empty rather than throwing into the UI. */
export function readSceneLibrary(storage: SceneLibraryStorage | undefined): SceneLibraryEntry[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(SCENE_LIBRARY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isEntry).sort((a, b) => b.savedAt_ms - a.savedAt_ms);
  } catch {
    return [];
  }
}

function writeSceneLibrary(storage: SceneLibraryStorage | undefined, entries: readonly SceneLibraryEntry[]): SceneLibraryEntry[] {
  const bounded = [...entries].sort((a, b) => b.savedAt_ms - a.savedAt_ms).slice(0, SCENE_LIBRARY_LIMIT);
  if (!storage) return bounded;
  try {
    storage.setItem(SCENE_LIBRARY_STORAGE_KEY, JSON.stringify(bounded));
  } catch {
    // A full quota must not lose the caller's in-memory list.
  }
  return bounded;
}

export interface SaveSceneOptions {
  readonly savedAt_ms: number;
  /** Overwrite this entry instead of adding one, for "save over". */
  readonly replaceId?: string;
}

/**
 * Save under a name. Saving the same name twice replaces the earlier entry, so
 * iterating on one scene does not litter the list with near-duplicates.
 */
export function saveSceneToLibrary(
  storage: SceneLibraryStorage | undefined,
  name: string,
  scene: SceneDescription,
  presetId: string,
  options: SaveSceneOptions,
): { entries: SceneLibraryEntry[]; entry: SceneLibraryEntry } {
  const normalized = normalizeSceneName(name);
  const existing = readSceneLibrary(storage);
  const replaced = options.replaceId
    ?? savedSceneEntries(existing).find((entry) => entry.name.toLowerCase() === normalized.toLowerCase())?.id;
  const entry: SceneLibraryEntry = {
    id: replaced ?? sceneEntryId(normalized, options.savedAt_ms),
    name: normalized,
    savedAt_ms: options.savedAt_ms,
    presetId,
    scene: serializeScene(cloneScene(scene)),
  };
  const entries = writeSceneLibrary(storage, [entry, ...existing.filter((candidate) => candidate.id !== entry.id)]);
  return { entries, entry };
}

export function renameSceneInLibrary(
  storage: SceneLibraryStorage | undefined,
  id: string,
  name: string,
): SceneLibraryEntry[] {
  const normalized = normalizeSceneName(name);
  return writeSceneLibrary(storage, readSceneLibrary(storage).map((entry) => entry.id === id ? { ...entry, name: normalized } : entry));
}

export function deleteSceneFromLibrary(storage: SceneLibraryStorage | undefined, id: string): SceneLibraryEntry[] {
  return writeSceneLibrary(storage, readSceneLibrary(storage).filter((entry) => entry.id !== id));
}

/**
 * Parse a stored entry. Validation runs on load rather than on save so an
 * entry written by an older schema surfaces as a clear error instead of
 * corrupting the live scene.
 */
export function loadSceneFromLibrary(entry: SceneLibraryEntry): SceneDescription {
  return parseScene(entry.scene);
}
