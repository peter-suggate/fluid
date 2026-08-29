import assert from "node:assert/strict";
import test from "node:test";

import type { SceneCard } from "../lib/core/scene-definition";
import {
  SCENE_MATCH,
  SCENE_SEARCH_RECENT_SHELF,
  sceneMatchTier,
  sceneNameMatch,
  sceneSearchGroups,
  sceneSearchOrder,
  searchSceneCards,
} from "../lib/core/scene-search";

/**
 * Literal cards rather than the catalog.
 *
 * The ranking is a claim about *typing*, not about the scenes this build ships,
 * so binding it to real catalog names would make an editorial rename a red
 * test. It also keeps the whole file free of the scene factories and of the
 * method registry, which is what lets it run as an ordinary CPU test.
 */
function card(partial: Partial<SceneCard> & Pick<SceneCard, "id" | "name">): SceneCard {
  return {
    source: "catalog",
    blurb: "",
    shelf: "Shelf",
    open: () => { throw new Error("not opened in this test"); },
    ...partial,
  };
}

const CARDS: readonly SceneCard[] = [
  card({ id: "dam-break", name: "Dam break", shelf: "Dam breaks", blurb: "The classic collapse." }),
  card({ id: "dam-break-384", name: "Dam break 384", shelf: "Dam breaks", blurb: "A wider column." }),
  card({ id: "hero-garden-hose", name: "Garden hose", shelf: "Garden", blurb: "Water falls into a bowl." }),
  card({ id: "ceiling-slab", name: "Ceiling slab", shelf: "Free-fall contact", blurb: "A slab dropped from rest." }),
  card({ id: "ocean-seiche", name: "Ocean seiche", shelf: "Oceans", blurb: "A basin sloshing end to end." }),
];

test("an empty needle keeps every card in the order it was handed", () => {
  assert.deepEqual(
    searchSceneCards(CARDS, "  ").map((hit) => hit.id),
    CARDS.map((hit) => hit.id));
});

test("a prefix of the name outranks every other kind of match", () => {
  // "dam" prefixes two names and also names their shelf; the shelf-only tier
  // must not float a card above one whose own name starts with the needle.
  const ranked = searchSceneCards(CARDS, "dam").map((hit) => hit.id);
  assert.deepEqual(ranked, ["dam-break", "dam-break-384"]);
  assert.equal(sceneMatchTier(CARDS[0]!, "dam"), SCENE_MATCH.namePrefix);
});

test("a later word of the name is a better match than a substring of it", () => {
  assert.equal(sceneMatchTier(CARDS[0]!, "break"), SCENE_MATCH.wordPrefix);
  assert.equal(sceneMatchTier(CARDS[3]!, "lab"), SCENE_MATCH.name);
  const ranked = searchSceneCards([CARDS[3]!, CARDS[0]!], "b");
  assert.equal(ranked[0]!.id, "dam-break", "the word prefix sorts above the substring");
});

test("the id and the blurb are matched, and rank below the name", () => {
  assert.equal(sceneMatchTier(CARDS[2]!, "hero"), SCENE_MATCH.id);
  assert.equal(sceneMatchTier(CARDS[4]!, "sloshing"), SCENE_MATCH.blurb);
  assert.equal(sceneMatchTier(CARDS[2]!, "no such scene"), undefined);
});

test("an exact name or id beats a prefix of a longer one", () => {
  assert.equal(sceneMatchTier(CARDS[0]!, "Dam break"), SCENE_MATCH.exact);
  assert.equal(sceneMatchTier(CARDS[1]!, "Dam break"), SCENE_MATCH.namePrefix);
  assert.equal(searchSceneCards(CARDS, "dam break")[0]!.id, "dam-break");
  assert.equal(searchSceneCards(CARDS, "ocean-seiche")[0]!.id, "ocean-seiche");
});

test("groups keep the ranking: the first card of the first group is the best answer", () => {
  const groups = sceneSearchGroups(CARDS, "dam");
  assert.deepEqual(groups.map((group) => group.shelf), ["Dam breaks"]);
  assert.equal(sceneSearchOrder(groups)[0]!.id, "dam-break");
});

test("a shelf appears in the order its best card did", () => {
  // "o" reaches an ocean by name prefix and a garden hose only by substring, so
  // the Oceans heading has to come first even though Garden is earlier in the
  // catalog: a heading order that ignored the ranking would put the reader's
  // best answer under the second heading.
  const groups = sceneSearchGroups(CARDS, "o");
  assert.equal(groups[0]!.shelf, "Oceans");
  assert.ok(groups.some((group) => group.shelf === "Garden"));
});

test("recents head the list while nothing is typed, and never once something is", () => {
  const recent = [CARDS[4]!, CARDS[2]!];
  const idle = sceneSearchGroups(CARDS, "", { recent });
  assert.equal(idle[0]!.shelf, SCENE_SEARCH_RECENT_SHELF);
  assert.deepEqual(idle[0]!.cards.map((hit) => hit.id), ["ocean-seiche", "hero-garden-hose"]);
  // Every card is still reachable under its own shelf; the recent row is an
  // offer, not a filter.
  assert.equal(sceneSearchOrder(idle).length, CARDS.length + recent.length);
  const typed = sceneSearchGroups(CARDS, "dam", { recent });
  assert.ok(typed.every((group) => group.shelf !== SCENE_SEARCH_RECENT_SHELF));
});

test("the highlight marks only what the name actually matched", () => {
  assert.deepEqual(sceneNameMatch(CARDS[0]!, "break"), [4, 9]);
  assert.equal(sceneNameMatch(CARDS[0]!, ""), undefined);
  // Matched by its blurb: there is nothing in the name to mark, and claiming a
  // span would misreport why the row is on screen.
  assert.equal(sceneNameMatch(CARDS[4]!, "sloshing"), undefined);
});
