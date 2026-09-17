import type { EditorAction } from "../../core/editor-action";
import { regionCapacityRemaining, type RegionSpace } from "./definition";

/**
 * The region's contribution to a ring, for either host.
 *
 * A verb, not an instrument: drawing a box is a thing you *do* at a point, so
 * it belongs on the pie the pointer opens and never on a shelf — see the
 * contextual-capabilities doctrine. The studio composes the same wedge today in
 * `lib/core/editor-fluid-body.ts`; this is that wedge with the scene document
 * lifted out, so the lab's ring can hold the identical one.
 *
 * Capacity is in the hint rather than in a notice, because the answer to "why
 * did nothing happen" has to be readable *before* the click. A full tail
 * disables the wedge instead of hiding it: a capability that vanishes teaches
 * the reader it was never there.
 */
export function regionDrawWedge<Doc, Patch>(
  space: RegionSpace<Doc, Patch>,
  doc: Doc,
): EditorAction {
  const remaining = regionCapacityRemaining(space, doc);
  return {
    id: "region",
    label: "Region",
    icon: "region",
    tone: "region",
    hint: remaining > 0
      ? "Drag a box over the water to cap how finely it is solved there"
      : `All ${space.capacity} refinement boxes are in use — delete one to draw another`,
    enabled: remaining > 0,
    effect: { kind: "arm", gesture: "region-draw" },
  };
}

/**
 * Every wedge a region offers at a point.
 *
 * One entry today, and a list because the ring is composed rather than
 * hard-coded: `entityActionsAt` appends the general `select` and `delete` after
 * whatever a definition contributes, so a region's own verbs go here and the
 * two irreversible ones stay in the one place that guarantees `delete` is last
 * and toned `danger` for every entity alike.
 */
export function regionWedges<Doc, Patch>(
  space: RegionSpace<Doc, Patch>,
  doc: Doc,
): readonly EditorAction[] {
  return [regionDrawWedge(space, doc)];
}
