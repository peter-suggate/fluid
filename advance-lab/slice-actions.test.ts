import assert from "node:assert/strict";
import test from "node:test";
import { actionIsChoosable, type EditorAction } from "../lib/core/editor-action";
import type { SliceOverlayId } from "./lenses";
import {
  sliceActionsAt, sliceHostVerb, sliceRingTitle, type SliceRingContext,
} from "./slice-actions";

/**
 * The lab's ring, pinned by shape.
 *
 * The ring's premise is that a wedge is a direction a hand can learn, which is
 * only true while the wedge is in the same place every time. So these tests are
 * about the *shape* — which wedges, in which order, enabled or not — rather
 * than about labels, and a change that moves a verb between wedges is meant to
 * fail here before anybody's muscle memory finds out.
 */

const AT: readonly [number, number] = [12, 7];

function context(patch: Partial<SliceRingContext> = {}): SliceRingContext {
  return {
    mode: "interact",
    at: AT,
    surfaceView: "shared-rdf",
    surfaceImposed: false,
    overlays: new Set<SliceOverlayId>(),
    overlaysOffered: ["fraction", "normal"],
    capacityLeft: 8,
    ...patch,
  };
}

const ids = (actions: readonly EditorAction[]): readonly string[] =>
  actions.map(action => action.id);

const find = (actions: readonly EditorAction[], id: string): EditorAction => {
  const found = actions.find(action => action.id === id);
  assert.ok(found, `expected a "${id}" wedge`);
  return found;
};

test("LOOK offers the instruments and withholds every verb", () => {
  const actions = sliceActionsAt(context({ mode: "camera" }));
  assert.deepEqual(ids(actions), ["slice-inspect-cell", "slice-visuals"]);
  assert.equal(sliceRingTitle(context({ mode: "camera" })), "Looking at the slice");
  // Pinning a cell reads the water; it never moves it, which is why it is the
  // one wedge LOOK keeps.
  assert.equal(find(actions, "slice-inspect-cell").enabled, true);
});

test("LOOK over a region still offers no region verb", () => {
  const actions = sliceActionsAt(context({ mode: "camera", regionId: "advance-region-1" }));
  assert.deepEqual(ids(actions), ["slice-inspect-cell", "slice-visuals"]);
});

test("EDIT on the water offers the two strokes under one wedge", () => {
  const actions = sliceActionsAt(context());
  assert.deepEqual(ids(actions), ["slice-water", "slice-inspect-cell", "slice-visuals"]);
  const water = find(actions, "slice-water");
  assert.deepEqual((water.children ?? []).map(child => child.id),
    ["slice-drop-ball", "slice-draw-region"]);
  // A category wedge is the direction, never the verb: it carries no effect of
  // its own, and choosing it opens the second level rather than doing something.
  assert.equal(water.effect, undefined);
  assert.equal(actionIsChoosable(water), true);
});

test("EDIT on a region is about that region", () => {
  const actions = sliceActionsAt(context({ regionId: "advance-region-3" }));
  assert.deepEqual(ids(actions),
    ["slice-region-select", "slice-region-remove", "slice-inspect-cell", "slice-visuals"]);
  const select = find(actions, "slice-region-select");
  assert.deepEqual(select.effect, {
    kind: "select",
    selection: { kind: "refinement-region", id: "refinement-region-advance-region-3" },
    openControls: true,
  });
  assert.deepEqual(sliceHostVerb(find(actions, "slice-region-remove").effect!),
    { id: "remove-region", region: "advance-region-3" });
});

test("off the slice the wedges stay, disabled", () => {
  const actions = sliceActionsAt(context({ at: null }));
  // The same three directions, so the ring a reader learned is the ring they get.
  assert.deepEqual(ids(actions), ["slice-water", "slice-inspect-cell", "slice-visuals"]);
  assert.equal(find(actions, "slice-inspect-cell").enabled, false);
  assert.equal(find(actions, "slice-inspect-cell").effect, undefined);
  const water = find(actions, "slice-water").children ?? [];
  assert.equal(water.find(child => child.id === "slice-drop-ball")?.enabled, false);
  // Drawing a box needs no point: the drag brings its own two corners.
  assert.equal(water.find(child => child.id === "slice-draw-region")?.enabled, true);
});

test("a full document offers the box and says why it cannot be drawn", () => {
  const actions = sliceActionsAt(context({ capacityLeft: 0 }));
  const draw = (find(actions, "slice-water").children ?? [])
    .find(child => child.id === "slice-draw-region");
  assert.equal(draw?.enabled, false);
  assert.match(draw?.hint ?? "", /eight/);
});

test("Visuals carries the surfaces and the overlays, and nothing else", () => {
  const visuals = find(sliceActionsAt(context()), "slice-visuals");
  assert.deepEqual((visuals.children ?? []).map(child => child.id), [
    "slice-surface-shared-rdf", "slice-surface-plic",
    "slice-overlay-fraction", "slice-overlay-normal",
  ]);
  // The reading that is already up is not a choice to make.
  assert.equal(visuals.children?.find(c => c.id === "slice-surface-shared-rdf")?.enabled, false);
  assert.equal(visuals.children?.find(c => c.id === "slice-surface-plic")?.enabled, true);
});

test("an imposed surface is stated rather than offered", () => {
  const visuals = find(sliceActionsAt(context({
    surfaceView: "direct-level-set", surfaceImposed: true,
  })), "slice-visuals");
  for (const child of visuals.children ?? []) {
    if (!child.id.startsWith("slice-surface-")) continue;
    assert.equal(child.enabled, false, `${child.id} must not be choosable`);
    assert.match(child.hint ?? "", /publishes/);
  }
});

test("an overlay this transport cannot draw is offered disabled, never dropped", () => {
  const visuals = find(sliceActionsAt(context({ overlaysOffered: ["fraction"] })), "slice-visuals");
  assert.equal(visuals.children?.find(c => c.id === "slice-overlay-fraction")?.enabled, true);
  assert.equal(visuals.children?.find(c => c.id === "slice-overlay-normal")?.enabled, false);
});

test("an overlay that is on says so in its label", () => {
  const visuals = find(sliceActionsAt(context({
    overlays: new Set<SliceOverlayId>(["fraction"]),
  })), "slice-visuals");
  assert.match(visuals.children?.find(c => c.id === "slice-overlay-fraction")?.label ?? "", /off$/);
  assert.doesNotMatch(visuals.children?.find(c => c.id === "slice-overlay-normal")?.label ?? "",
    /off$/);
});

test("the lab composes only arm, select and its own host verbs", () => {
  /* The boundary the `host` arm exists to hold. Anything else would be an
   * effect `performEditorAction` answers by reaching the 3-D simulation
   * singleton, and this page has no world for it to act on. */
  const every: SliceRingContext[] = [
    context({ mode: "camera" }),
    context(),
    context({ at: null }),
    context({ regionId: "advance-region-1" }),
    context({ capacityLeft: 0, surfaceImposed: true, surfaceView: "direct-level-set" }),
  ];
  const seen = new Set<string>();
  const walk = (actions: readonly EditorAction[]): void => {
    for (const action of actions) {
      if (action.effect) {
        seen.add(action.effect.kind);
        assert.ok(["arm", "select", "host"].includes(action.effect.kind),
          `${action.id} carries a "${action.effect.kind}" effect the lab cannot perform`);
        if (action.effect.kind === "host") {
          assert.ok(sliceHostVerb(action.effect),
            `${action.id} carries a host effect the lab's own union does not recognise`);
        }
      }
      walk(action.children ?? []);
    }
  };
  for (const one of every) walk(sliceActionsAt(one));
  assert.deepEqual([...seen].sort(), ["arm", "host", "select"]);
});

test("a host effect is validated on the way out, never cast", () => {
  assert.equal(sliceHostVerb({ kind: "host", id: "drop-ball", payload: undefined }), undefined);
  assert.equal(sliceHostVerb({ kind: "host", id: "drop-ball", payload: { id: "pin-cell", at: AT } }),
    undefined);
  assert.equal(sliceHostVerb({ kind: "arm", gesture: "region-draw" }), undefined);
});
