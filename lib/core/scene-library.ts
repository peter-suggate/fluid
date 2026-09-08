import { cloneScene, parseScene, type SceneDescription } from "./model";
import type { MethodProfile } from "./method-contract";

/**
 * Named local scene storage.
 *
 * Authoring is worthless if a reload discards it, and the URL cannot carry a
 * sculpted terrain grid or a large seed list. Entries are stored as the same
 * scene JSON the file export writes, so a library entry and a downloaded
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
  /** Compact serialized `SceneDescription`; file exports remain human-readable. */
  readonly scene: string;
  /** Active solver choice at save time. Absent in pre-benchmark entries. */
  readonly methodProfile?: MethodProfile;
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

function isMethodProfile(value: unknown): value is MethodProfile {
  if (!value || typeof value !== "object") return false;
  const profile = value as Partial<MethodProfile>;
  if (typeof profile.methodId !== "string"
    || !["balanced", "high", "ultra"].includes(String(profile.quality))
    || !profile.overrides || typeof profile.overrides !== "object"
    || Array.isArray(profile.overrides)) return false;
  return Object.values(profile.overrides).every((entry) =>
    typeof entry === "string" || typeof entry === "number" || typeof entry === "boolean");
}

function isEntry(value: unknown): value is SceneLibraryEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<SceneLibraryEntry>;
  return typeof entry.id === "string" && typeof entry.name === "string"
    && typeof entry.savedAt_ms === "number" && Number.isFinite(entry.savedAt_ms)
    && typeof entry.presetId === "string" && typeof entry.scene === "string"
    && (entry.methodProfile === undefined || isMethodProfile(entry.methodProfile));
}

export function isSceneAutosaveEntry(entry: SceneLibraryEntry): boolean {
  return entry.id === SCENE_AUTOSAVE_ENTRY_ID;
}

/** The entries a reader chose to keep; the autosave is not one of them. */
export function savedSceneEntries(entries: readonly SceneLibraryEntry[]): SceneLibraryEntry[] {
  return entries.filter((entry) => !isSceneAutosaveEntry(entry));
}

/**
 * Rewrite a profile saved before the adaptive backends became two methods.
 *
 * `isMethodProfile` accepts any string as a method id, so a stored `"octree"`
 * passes validation and reaches the store intact — where it names no
 * registered method, so its overrides land in a bucket nothing ever reads and
 * `getMethod` quietly hands back the default. The reader's saved tuning would
 * be gone with no error anywhere: the scene reopens, on a different solver,
 * looking like it worked. The autosave shares this array, so restoring the
 * working document crosses the same map.
 */
function migratedMethodProfile(profile: MethodProfile): MethodProfile {
  if (profile.methodId !== "octree") return profile;
  const { coarseBackend, ...overrides } = profile.overrides;
  return {
    ...profile,
    methodId: coarseBackend === "power2017" ? "power-liquids" : "losasso",
    overrides,
  };
}

/** Malformed storage is treated as empty rather than throwing into the UI. */
export function readSceneLibrary(storage: SceneLibraryStorage | undefined): SceneLibraryEntry[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(SCENE_LIBRARY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    const entries = decodeLibrary(parsed);
    return entries.filter(isEntry)
      .map((entry) => entry.methodProfile === undefined
        ? entry : { ...entry, methodProfile: migratedMethodProfile(entry.methodProfile) })
      .sort((a, b) => b.savedAt_ms - a.savedAt_ms);
  } catch {
    return [];
  }
}

/** Legacy arrays remain readable. Pooling keeps an explicit save and its
 * identical autosave from paying for the same large terrain document twice. */
function decodeLibrary(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];
  const library = value as { version?: unknown; documents?: unknown; entries?: unknown };
  if (library.version !== 2 || !Array.isArray(library.documents) || !Array.isArray(library.entries)) return [];
  const documents = library.documents;
  return library.entries.flatMap((value) => {
    if (!value || typeof value !== "object") return [];
    const entry = value as { document?: unknown };
    if (typeof entry.document !== "number" || !Number.isInteger(entry.document)
      || typeof documents[entry.document] !== "string") return [];
    const { document, ...metadata } = entry;
    return [{ ...metadata, scene: documents[document] }];
  });
}

function compactSceneDocument(scene: string): string {
  try { return JSON.stringify(JSON.parse(scene)); }
  // Keep old invalid documents intact; validation still reports errors on load.
  catch { return scene; }
}

function writeSceneLibrary(storage: SceneLibraryStorage | undefined, entries: readonly SceneLibraryEntry[]): {
  entries: SceneLibraryEntry[]; persisted: boolean;
} {
  const previous = readSceneLibrary(storage);
  // Never evict a saved scene to make a new save fit.
  if (!storage || entries.length > SCENE_LIBRARY_LIMIT) return { entries: previous, persisted: false };
  const compact = [...entries].sort((a, b) => b.savedAt_ms - a.savedAt_ms)
    .map((entry) => ({ ...entry, scene: compactSceneDocument(entry.scene) }));
  const documents: string[] = [];
  const indices = new Map<string, number>();
  const records = compact.map(({ scene, ...entry }) => {
    let document = indices.get(scene);
    if (document === undefined) {
      document = documents.length;
      indices.set(scene, document);
      documents.push(scene);
    }
    return { ...entry, document };
  });
  try {
    // One atomic replacement: a quota failure leaves every previous byte intact.
    storage.setItem(SCENE_LIBRARY_STORAGE_KEY, JSON.stringify({ version: 2, documents, entries: records }));
    return { entries: compact, persisted: true };
  } catch {
    return { entries: previous, persisted: false };
  }
}

export interface SaveSceneOptions {
  readonly savedAt_ms: number;
  /** Overwrite this entry instead of adding one, for "save over". */
  readonly replaceId?: string;
  /** Active method state, so Continue resumes the selected solver contract. */
  readonly methodProfile?: MethodProfile;
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
): { entries: SceneLibraryEntry[]; entry: SceneLibraryEntry; persisted: boolean } {
  const normalized = normalizeSceneName(name);
  const existing = readSceneLibrary(storage);
  const replaced = options.replaceId
    ?? savedSceneEntries(existing).find((entry) => entry.name.toLowerCase() === normalized.toLowerCase())?.id;
  const entry: SceneLibraryEntry = {
    id: replaced ?? sceneEntryId(normalized, options.savedAt_ms),
    name: normalized,
    savedAt_ms: options.savedAt_ms,
    presetId,
    scene: JSON.stringify(cloneScene(scene)),
    ...(options.methodProfile === undefined ? {} : { methodProfile: options.methodProfile }),
  };
  const result = writeSceneLibrary(storage, [entry, ...existing.filter((candidate) => candidate.id !== entry.id)]);
  return { ...result, entry };
}

/**
 * The given name, numbered if another saved entry already holds it.
 *
 * Names are not identities here — ids are — but `saveSceneToLibrary` resolves
 * "save over" by name, so two entries sharing one would make the next save
 * replace whichever the recency sort happened to put first: the reader's other
 * scene of that name is overwritten with no error anywhere. Saving cannot
 * reach that state (it replaces rather than duplicates), so renaming is the
 * only caller that has to number. `exceptId` is the entry being renamed, which
 * must not collide with itself.
 */
export function uniqueSceneName(
  entries: readonly SceneLibraryEntry[],
  name: string,
  exceptId: string,
): string {
  const normalized = normalizeSceneName(name);
  const taken = new Set(savedSceneEntries(entries)
    .filter((entry) => entry.id !== exceptId)
    .map((entry) => entry.name.toLowerCase()));
  if (!taken.has(normalized.toLowerCase())) return normalized;
  // Bounded by the library itself: with every slot taken the loop cannot run
  // out of candidates before it runs out of entries to collide with.
  for (let suffix = 2; suffix <= SCENE_LIBRARY_LIMIT + 1; suffix += 1) {
    const candidate = normalizeSceneName(`${normalized} ${suffix}`);
    if (!taken.has(candidate.toLowerCase())) return candidate;
  }
  return normalized;
}

/**
 * Rename one saved entry, keeping its id and its save time: a rename is not an
 * edit of the scene, so it must not reorder the shelf or invalidate a preview.
 * The autosave is refused — it is the working document, named by the machine.
 */
export function renameSceneInLibrary(
  storage: SceneLibraryStorage | undefined,
  id: string,
  name: string,
): SceneLibraryEntry[] {
  const entries = readSceneLibrary(storage);
  if (id === SCENE_AUTOSAVE_ENTRY_ID) return entries;
  const normalized = uniqueSceneName(entries, name, id);
  return writeSceneLibrary(storage, entries.map((entry) => entry.id === id ? { ...entry, name: normalized } : entry)).entries;
}

export function deleteSceneFromLibrary(storage: SceneLibraryStorage | undefined, id: string): SceneLibraryEntry[] {
  return writeSceneLibrary(storage, readSceneLibrary(storage).filter((entry) => entry.id !== id)).entries;
}

/**
 * Parse a stored entry. Validation runs on load rather than on save so an
 * entry written by an older schema surfaces as a clear error instead of
 * corrupting the live scene.
 */
export function loadSceneFromLibrary(entry: SceneLibraryEntry): SceneDescription {
  return parseScene(entry.scene);
}
