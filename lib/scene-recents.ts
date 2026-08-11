import type { SceneCard } from "./scene-definition";
import type { SceneLibraryStorage } from "./scene-library";

/**
 * Which cards were actually opened, newest first.
 *
 * The library's sections are organised by audience, but the person who opens
 * the same three oracle lanes every day is not browsing by audience — they are
 * re-finding cards they already know, buried mid-scroll in a disclosed section.
 * Recording opens lets the page offer those cards at the top, as a row of
 * compact cards, instead of making every visit a scroll hunt.
 *
 * Stored as card ids rather than documents: an id that no longer resolves —
 * a deleted save, a scene an older build had — silently drops out of the row
 * rather than becoming an unopenable card. The storage handle is injected for
 * the same reason as `scene-library`: testable without a DOM, a no-op without
 * storage.
 */

export const SCENE_RECENTS_STORAGE_KEY = "fluid-lab.scene-recents.v1";
/** More than the row shows, so deleted saves do not thin the row below it. */
export const SCENE_RECENTS_LIMIT = 16;

export interface RecentSceneOpen {
  readonly cardId: string;
  readonly openedAt_ms: number;
}

function isRecentOpen(value: unknown): value is RecentSceneOpen {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<RecentSceneOpen>;
  return typeof entry.cardId === "string"
    && typeof entry.openedAt_ms === "number" && Number.isFinite(entry.openedAt_ms);
}

/** Malformed storage is treated as empty rather than throwing into the UI. */
export function readSceneRecents(storage: SceneLibraryStorage | undefined): RecentSceneOpen[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(SCENE_RECENTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecentOpen).sort((a, b) => b.openedAt_ms - a.openedAt_ms);
  } catch {
    return [];
  }
}

/**
 * Record an open. Reopening a card moves it to the front rather than adding a
 * duplicate, so the row is the reader's working set, not their click log.
 */
export function recordSceneOpen(
  storage: SceneLibraryStorage | undefined,
  cardId: string,
  openedAt_ms: number,
): RecentSceneOpen[] {
  const entries = [
    { cardId, openedAt_ms },
    ...readSceneRecents(storage).filter((entry) => entry.cardId !== cardId),
  ].slice(0, SCENE_RECENTS_LIMIT);
  if (storage) {
    try {
      storage.setItem(SCENE_RECENTS_STORAGE_KEY, JSON.stringify(entries));
    } catch {
      // A full quota must not lose the caller's in-memory list.
    }
  }
  return entries;
}

/**
 * The recents as openable cards, in open order, unknown ids dropped.
 *
 * Resolution happens against whatever card set the caller has *now*, which is
 * what keeps a stale id harmless: the row can only ever offer cards the library
 * would offer anyway.
 */
export function recentSceneCards(
  recents: readonly RecentSceneOpen[],
  cards: readonly SceneCard[],
  limit: number,
): readonly SceneCard[] {
  const byId = new Map(cards.map((card) => [card.id, card]));
  return recents
    .flatMap((entry) => byId.get(entry.cardId) ?? [])
    .slice(0, limit);
}
