import { SCENE_STARTERS } from "./empty-scene";
import type { SceneDescription } from "./model";
import { sceneIsoGlyph, type SceneIsoGlyph } from "./scene-glyph-iso";
import {
  SCENE_AUDIENCES,
  sceneDefinitionCamera,
  sceneShelves,
  type SceneAudience,
  type SceneCard,
  type SceneShelf,
} from "./scene-definition";
import { loadSceneFromLibrary, savedSceneEntries, type SceneLibraryEntry } from "./scene-library";
import { findSceneDefinition, sceneCatalogCards } from "./scenes";

/**
 * Everything the library can offer, as one list of cards.
 *
 * Three sources used to be three behaviours. `loadPreset` applied the scene's
 * camera and its solver profile and started the clock; `loadNamedScene` applied
 * neither; nothing at all could create an empty document. A grid that shows all
 * three in one row cannot afford that — a card is a promise that clicking it
 * does the same kind of thing whichever shelf it came from.
 *
 * So the differences are carried as data (`SceneCard.source`, `savedAt_ms`) and
 * the behaviour is one function, `SceneCard.open`.
 */

export type SceneSectionId = SceneAudience | "mine";

export interface SceneSection {
  readonly id: SceneSectionId;
  readonly label: string;
  readonly blurb: string;
  /** Collapsed by default: present and findable, not part of the first impression. */
  readonly disclosed: boolean;
  readonly shelves: readonly SceneShelf[];
}

/** Cards for the empty-document starters, which create rather than load. */
export function starterSceneCards(): readonly SceneCard[] {
  return SCENE_STARTERS.map((starter) => ({
    id: `starter:${starter.id}`,
    source: "starter" as const,
    name: starter.name,
    blurb: starter.blurb,
    shelf: "Start from empty",
    open: () => ({ scene: starter.create(), presetId: `starter:${starter.id}` }),
  }));
}

/**
 * One stored entry as a card.
 *
 * A saved document records the scene it was authored from but not a camera, so
 * the framing and the solver profile come from that definition when this build
 * still has it. Dropping the profile is how a saved validation scene used to
 * end up running under whatever method happened to be selected.
 *
 * Separate from the list below because the autosaved working document is a
 * stored entry that the shelf must not offer and *Continue* must still open.
 */
export function savedSceneCard(entry: SceneLibraryEntry): SceneCard {
  const origin = findSceneDefinition(entry.presetId);
  return {
    id: `saved:${entry.id}`,
    source: "saved",
    name: entry.name,
    blurb: origin ? `Saved from ${origin.name}.` : "A scene you saved in this browser.",
    shelf: "My scenes",
    savedAt_ms: entry.savedAt_ms,
    open: () => ({
      scene: loadSceneFromLibrary(entry),
      presetId: entry.presetId,
      camera: origin ? sceneDefinitionCamera(origin) : undefined,
      methodProfile: origin?.methodProfile,
    }),
  };
}

/**
 * Cards for the browser's saved scenes, newest first.
 *
 * The autosave is excluded: it is the *Continue* affordance, and a shelf that
 * also listed it would offer the reader a copy of the scene they are already in,
 * under a name they never chose, beside the save they actually made.
 */
export function savedSceneCards(entries: readonly SceneLibraryEntry[]): readonly SceneCard[] {
  return savedSceneEntries(entries)
    .sort((left, right) => right.savedAt_ms - left.savedAt_ms)
    .map(savedSceneCard);
}

export function allSceneCards(saved: readonly SceneLibraryEntry[]): readonly SceneCard[] {
  return [...starterSceneCards(), ...sceneCatalogCards, ...savedSceneCards(saved)];
}

/**
 * The library, in the order it is read.
 *
 * The starters are deliberately absent: creating is a peer of choosing, so the
 * three room sizes sit in the hero beside *Continue* rather than in a shelf
 * below the scenes. They are still cards, still in `allSceneCards`, and still
 * findable by search — they simply are not a section. The person's own scenes
 * sit above the oracles because those are the two things they are most likely to
 * have come back for. Empty sections are dropped rather than rendered blank — a
 * "My scenes (0)" heading is a reproach, not information.
 */
export function sceneSections(saved: readonly SceneLibraryEntry[]): readonly SceneSection[] {
  const mine = savedSceneCards(saved);
  const audience = (id: SceneAudience): SceneSection | undefined => {
    const entry = SCENE_AUDIENCES.find((candidate) => candidate.id === id);
    if (!entry) return undefined;
    const cards = sceneCatalogCards.filter((card) => card.audience === id);
    if (cards.length === 0) return undefined;
    return { id, label: entry.label, blurb: entry.blurb, disclosed: entry.disclosed, shelves: sceneShelves(cards) };
  };
  const sections: (SceneSection | undefined)[] = [
    audience("explore"),
    audience("study"),
    mine.length === 0 ? undefined : {
      id: "mine",
      label: "My scenes",
      blurb: "Saved in this browser. They export as the same JSON as the file download.",
      disclosed: false,
      shelves: sceneShelves(mine),
    },
    audience("validation"),
  ];
  return sections.filter((section): section is SceneSection => section !== undefined);
}

/**
 * The document a card would open, for drawing its preview.
 *
 * Cached because a card's `open` is a constructor — the garden scenes build a
 * terrain field and expand a scenery graph — and the library draws every card
 * on every keystroke of the search box. Documents are deterministic, so one
 * build per identity is exact rather than merely an optimisation.
 *
 * A saved entry keeps its id when it is re-saved, so its save time is part of
 * the key; otherwise editing and re-saving a scene would keep showing the
 * preview of the version it replaced. A card whose document fails validation
 * answers undefined: the library still lists it, and the failure surfaces as a
 * notice when it is actually opened.
 */
const previewCache = new Map<string, SceneDescription>();

function previewKey(card: SceneCard): string {
  return card.savedAt_ms === undefined ? card.id : `${card.id}@${card.savedAt_ms}`;
}

export function sceneCardPreview(card: SceneCard): SceneDescription | undefined {
  const key = previewKey(card);
  const cached = previewCache.get(key);
  if (cached) return cached;
  try {
    const scene = card.open().scene;
    previewCache.set(key, scene);
    return scene;
  } catch {
    return undefined;
  }
}

/**
 * The thumbnail a card draws, cached on the same identity as its document.
 *
 * The mark samples the ground on a nine-by-seventeen grid, and the library
 * redraws every card on every keystroke of the search box. Both are pure
 * functions of the document, so one build per identity is exact rather than
 * merely an optimisation.
 */
const glyphCache = new Map<string, SceneIsoGlyph>();

export function sceneCardGlyph(card: SceneCard): SceneIsoGlyph | undefined {
  const key = previewKey(card);
  const cached = glyphCache.get(key);
  if (cached) return cached;
  const scene = sceneCardPreview(card);
  if (!scene) return undefined;
  const glyph = sceneIsoGlyph(scene);
  glyphCache.set(key, glyph);
  return glyph;
}

/**
 * Free-text match over what a card actually shows.
 *
 * Deliberately not fuzzy: someone typing "garden" is naming a shelf, and
 * someone typing "drop" is naming an oracle they already know exists. Matching
 * the blurb as well is what makes the second one work without a tag vocabulary
 * nobody would maintain.
 */
export function matchSceneCards(cards: readonly SceneCard[], query: string): readonly SceneCard[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return cards;
  return cards.filter((card) => `${card.name} ${card.shelf} ${card.blurb}`.toLowerCase().includes(needle));
}
