/**
 * The shape lab's landing pad: numbers tuned at `/shape-lab`, in the one form
 * the hero document actually reads.
 *
 * ## What this is
 *
 * Every shape in the garden is already described rather than baked — that is the
 * whole argument of `lib/scenery-generators.ts` — so a tuned specimen is a
 * `SceneryNode`, and a tuned pond is a `PondVesselSpec`. Both are plain JSON by
 * construction, because a document is cloned by JSON round trip and written to
 * `localStorage`. There is therefore nothing to invent here: this file is a place
 * to *put* those two, keyed the way the graph already keys them.
 *
 * The loop it closes:
 *
 *     /shape-lab  ->  copy JSON  ->  paste into HERO_GARDEN_OVERRIDES below
 *                                      |
 *                                      v
 *          the app, the smoke lane, the fidelity gate and the lab itself
 *          all rebuild from it, because they all build the same document
 *
 * Paste replaces: a node override is the **whole node**, not a patch, which is
 * what the lab hands you and what makes "what is actually in effect" a thing you
 * can read rather than compute.
 *
 * ## What it is not
 *
 * Not a second authority on the garden. An override that has settled belongs in
 * the species or in `lib/hero-garden-scene.ts` beside the argument for it —
 * every number in that file carries a paragraph saying what it was measured
 * against, and a value that lands here keeps none of that. So this is a
 * *staging* area: tune here, then promote, then empty it again. An entry that
 * has been here across more than one piece of work is a note nobody wrote.
 *
 * Empty is the normal state, and an empty file changes nothing.
 */
import type { SceneryNode } from "./scenery-graph";
import type { PondVesselSpec } from "./voxel-scenery/pond-vessel";

export interface HeroGardenOverrides {
  /** Bumped only if the shape of this file ever changes. */
  readonly version: 1;
  /**
   * The pond, whole. Replaces `HERO_GARDEN_VESSEL`, which the terrain, the
   * waterline, the stone set and the layout all derive from — so this one entry
   * moves the ground under every specimen, exactly as the lab shows it.
   */
  readonly vessel?: PondVesselSpec;
  /**
   * Scenery nodes by id, whole. Applied to the graph after the layout is
   * composed, so both an authored node (`bonsai`) and a laid-out one
   * (`layout/plant/air-1`) can be replaced by the same rule.
   */
  readonly nodes?: Readonly<Record<string, SceneryNode>>;
}

export const HERO_GARDEN_OVERRIDES: HeroGardenOverrides = {
  version: 1,
};

/** Swap in any node the overrides name, leaving order and everything else alone. */
export function applyHeroGardenNodeOverrides(nodes: readonly SceneryNode[]): readonly SceneryNode[] {
  const overrides = HERO_GARDEN_OVERRIDES.nodes;
  if (!overrides) return nodes;
  return nodes.map((node) => overrides[node.id] ?? node);
}
