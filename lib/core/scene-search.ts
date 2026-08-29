import type { SceneCard } from "./scene-definition";

/**
 * Which scenes a few typed characters mean.
 *
 * The library's own `matchSceneCards` is a filter: it answers *whether* a card
 * matches and leaves the catalog's editorial order alone, which is right for a
 * page of tiles someone is browsing. A selector popover is the other gesture —
 * the reader already knows the scene's name and is typing it to get there — so
 * the answer has to be *ordered*, and the card they meant has to be the one
 * under the cursor when they press Enter. "dam" must not offer nine oracles
 * before "Dam break".
 *
 * So this module ranks rather than filters, and it is deliberately pure: no
 * store, no storage, no React, and only a *type* from the scene catalog. That
 * is what lets the whole ranking be tested on the CPU against literal cards,
 * with none of the method registry or the scene factories in the graph.
 *
 * Still not fuzzy, for the reason `matchSceneCards` gives: someone typing
 * "garden" is naming a shelf and someone typing "drop" is naming an oracle
 * they know exists. Neither is asking for edit distance, and a fuzzy matcher's
 * false positives are exactly the rows that would sit above the right answer.
 */

/**
 * How well one card answers a needle. Lower is better; every tier is a
 * different *kind* of match rather than a weight, so the order between them is
 * a statement about what the typing meant rather than a tuning constant.
 */
export const SCENE_MATCH = {
  /** The whole name, or the card's id. Nothing outranks having typed it. */
  exact: 0,
  /** The name begins with it: still typing the thing they meant. */
  namePrefix: 1,
  /** A later word of the name begins with it — "break" for "Dam break". */
  wordPrefix: 2,
  /** The shelf begins with it: naming a family rather than a scene. */
  shelfPrefix: 3,
  /** Somewhere inside the name. */
  name: 4,
  /** Somewhere inside the shelf. */
  shelf: 5,
  /** Inside the id, which a reader who came from a URL may well be typing. */
  id: 6,
  /** Only the blurb — the tier that makes "drop" find an oracle by its prose. */
  blurb: 7,
} as const;

export type SceneMatchTier = (typeof SCENE_MATCH)[keyof typeof SCENE_MATCH];

/** The card ids the selector shows under their own heading, before the shelves. */
export const SCENE_SEARCH_RECENT_SHELF = "Recent";

function normalize(value: string): string {
  return value.trim().toLowerCase();
}

/**
 * Whether some word of `haystack` begins with `needle`.
 *
 * Words rather than characters, because a name is a phrase: "break" naming
 * "Dam break" is a reader completing a word they can see, and it is a
 * materially better match than "reak" happening to fall inside it.
 */
function wordPrefixed(haystack: string, needle: string): boolean {
  return haystack.split(/[^a-z0-9]+/).some((word) => word.length > 0 && word.startsWith(needle));
}

/**
 * How well a card answers, or `undefined` when it does not.
 *
 * An empty needle matches everything at the best tier, so a selector opened and
 * not yet typed into lists the catalog in its authored order rather than in
 * whatever order a scoring function happened to produce.
 */
export function sceneMatchTier(card: SceneCard, query: string): SceneMatchTier | undefined {
  const needle = normalize(query);
  if (!needle) return SCENE_MATCH.exact;
  const name = card.name.toLowerCase();
  const shelf = card.shelf.toLowerCase();
  const id = card.id.toLowerCase();
  if (name === needle || id === needle) return SCENE_MATCH.exact;
  if (name.startsWith(needle)) return SCENE_MATCH.namePrefix;
  if (wordPrefixed(name, needle)) return SCENE_MATCH.wordPrefix;
  if (shelf.startsWith(needle)) return SCENE_MATCH.shelfPrefix;
  if (name.includes(needle)) return SCENE_MATCH.name;
  if (shelf.includes(needle)) return SCENE_MATCH.shelf;
  if (id.includes(needle)) return SCENE_MATCH.id;
  if (card.blurb.toLowerCase().includes(needle)) return SCENE_MATCH.blurb;
  return undefined;
}

/**
 * The cards a needle names, best first.
 *
 * Ties keep the order they were handed in, which is the catalog's editorial
 * order — so within one tier the list still reads the way the library reads,
 * and a ranking never quietly overrules an authoring decision it knows nothing
 * about.
 */
export function searchSceneCards(
  cards: readonly SceneCard[],
  query: string,
): readonly SceneCard[] {
  return cards
    .map((card, index) => ({ card, index, tier: sceneMatchTier(card, query) }))
    .filter((hit): hit is { card: SceneCard; index: number; tier: SceneMatchTier } => hit.tier !== undefined)
    .sort((left, right) => left.tier - right.tier || left.index - right.index)
    .map((hit) => hit.card);
}

export interface SceneSearchGroup {
  readonly shelf: string;
  readonly cards: readonly SceneCard[];
}

export interface SceneSearchOptions {
  /**
   * Cards to offer first under their own heading while nothing is typed.
   *
   * Only while nothing is typed: once there is a needle the ranking *is* the
   * ordering, and a recency shelf above it would put a card the reader is not
   * looking at over the one they just described.
   */
  readonly recent?: readonly SceneCard[];
}

/**
 * The rows the popover draws, as headed groups.
 *
 * Grouping happens *after* ranking and never re-sorts across groups, so the
 * first card of the first group is always the best answer — which is what makes
 * Enter-on-open a safe gesture rather than a lottery. A shelf appears in the
 * order its best card did, so typing narrows the list without shuffling the
 * headings under the reader's eyes.
 */
export function sceneSearchGroups(
  cards: readonly SceneCard[],
  query: string,
  options: SceneSearchOptions = {},
): readonly SceneSearchGroup[] {
  const groups: SceneSearchGroup[] = [];
  const recent = normalize(query) ? [] : options.recent ?? [];
  if (recent.length > 0) groups.push({ shelf: SCENE_SEARCH_RECENT_SHELF, cards: recent });
  const shelves: { shelf: string; cards: SceneCard[] }[] = [];
  for (const card of searchSceneCards(cards, query)) {
    const existing = shelves.find((entry) => entry.shelf === card.shelf);
    if (existing) existing.cards.push(card);
    else shelves.push({ shelf: card.shelf, cards: [card] });
  }
  return [...groups, ...shelves];
}

/** Every card in a group list, in the order the keyboard walks them. */
export function sceneSearchOrder(groups: readonly SceneSearchGroup[]): readonly SceneCard[] {
  return groups.flatMap((group) => group.cards);
}

/**
 * The same cards, laid out as the rows a grid of `columns` actually draws.
 *
 * A selector that shows thumbnails is read in two dimensions, so its arrow keys
 * have to move in two — and "up" is not `index - columns`: every group starts a
 * new row, because a shelf of four does not spill its fourth tile under the
 * next shelf's heading. That makes the layout a fact about the *groups* rather
 * than about the flat index, which is why it is computed here beside them and
 * not divided out of a count in the component.
 *
 * Indices are into `sceneSearchOrder`, so the rows and the keyboard's cursor
 * are the same numbers and neither has to be translated into the other.
 */
export function sceneSearchRows(
  groups: readonly SceneSearchGroup[],
  columns: number,
): readonly (readonly number[])[] {
  const rows: number[][] = [];
  let flat = 0;
  for (const group of groups) {
    for (let at = 0; at < group.cards.length; at += columns) {
      const width = Math.min(columns, group.cards.length - at);
      rows.push(Array.from({ length: width }, () => flat++));
    }
  }
  return rows;
}

/** The four keys that move a grid cursor. */
export type SceneGridKey = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";

/**
 * Where an arrow key lands, given the rows and where the cursor is.
 *
 * Left and right walk the flat order, so they cross a shelf heading the way
 * reading does. Up and down hold the column and clamp it, so a ragged last row
 * — three shelves in this catalog have one — catches the cursor at its end
 * instead of swallowing the keystroke. Both axes wrap: a list this shallow is
 * faster to walk round than to walk back through, and wrapping is also what
 * makes Up from the very first tile reach the last one.
 */
export function sceneSearchStep(
  rows: readonly (readonly number[])[],
  active: number,
  key: SceneGridKey,
): number {
  const total = rows.reduce((sum, row) => sum + row.length, 0);
  if (total === 0) return active;
  if (key === "ArrowLeft" || key === "ArrowRight") {
    const delta = key === "ArrowRight" ? 1 : -1;
    return (active + delta + total) % total;
  }
  const row = rows.findIndex((entry) => entry.includes(active));
  if (row < 0) return active;
  const column = rows[row].indexOf(active);
  const next = rows[(row + (key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length];
  return next[Math.min(column, next.length - 1)];
}

/**
 * The span of a card's name the needle matched, for the row to mark.
 *
 * Only the name, and only a literal substring: the point is to show the reader
 * *why* a row is here, and a highlight that claimed a match the ranking did not
 * make would be a lie about the ordering. A card matched only by its blurb or
 * its shelf therefore marks nothing, which is itself the right signal.
 */
export function sceneNameMatch(card: SceneCard, query: string): readonly [number, number] | undefined {
  const needle = normalize(query);
  if (!needle) return undefined;
  const at = card.name.toLowerCase().indexOf(needle);
  return at < 0 ? undefined : [at, at + needle.length];
}
