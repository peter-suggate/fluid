import assert from "node:assert/strict";
import test from "node:test";
import { actionIsChoosable, type EditorAction, type EditorActionEffect } from "../lib/core/editor-action";
import type { EditorHost } from "../lib/core/editor-host";
import type { PaneSession } from "../lib/core/session/session";
import type { EditorSelection } from "../lib/core/editor-tools";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import { labRegionSpace, type LabRegionDocument } from "./lab-region-space";
import {
  labActionPerformer, labHostVerb, labRingActions, labRingTitle, type LabRingContext,
} from "./lab-ring";
import type { SliceOverlayId } from "./lenses";

/**
 * The lab's ring, pinned by shape.
 *
 * The ring's premise is that a wedge is a direction a hand can learn, which is
 * only true while the wedge is in the same place every time. So these tests are
 * about the *shape* — which wedges, in which order, enabled or not — rather
 * than about labels, and a change that moves a verb between wedges is meant to
 * fail here before anybody's muscle memory finds out.
 *
 * Since WP6 the ids in those lists are not this page's: WATER and the ball in
 * it are `lib/features/liquid-drop/ring.ts`, REGION is
 * `lib/features/refinement-region/ring.ts`, EDIT and DELETE are
 * `lib/core/editor-entity-wedges.ts` and INSPECT CELL is
 * `lib/features/inspect/ring.ts`. That a selected box here offers the *same*
 * list as one selected in the studio is pinned in
 * `tests/editor-scene-ring.test.ts`, where both hosts are in scope.
 */

const AT: readonly [number, number] = [12, 7];
const REGION: AdvanceRefinementRegion = {
  id: "advance-region-1",
  minimumFine: [8, 8],
  maximumFine: [24, 24],
  minimumCellWidth: 2,
};
const DOC: LabRegionDocument = { regions: [REGION], nx: 128, ny: 64 };

function context(patch: Partial<LabRingContext> = {}): LabRingContext {
  return {
    mode: "interact",
    at: AT,
    doc: DOC,
    surfaceView: "shared-rdf",
    surfaceImposed: false,
    overlays: new Set<SliceOverlayId>(),
    overlaysOffered: ["fraction", "normal"],
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
  const actions = labRingActions(context({ mode: "camera" }));
  assert.deepEqual(ids(actions), ["inspect-cell", "slice-visuals"]);
  assert.equal(labRingTitle(context({ mode: "camera" })), "Looking at the slice");
});

test("LOOK over a region still offers no region verb", () => {
  const actions = labRingActions(context({ mode: "camera", regionId: REGION.id }));
  assert.deepEqual(ids(actions), ["inspect-cell", "slice-visuals"]);
});

test("EDIT on the water offers the two strokes under one wedge", () => {
  const actions = labRingActions(context());
  assert.deepEqual(ids(actions), ["water", "inspect-cell", "slice-visuals"]);
  assert.deepEqual(ids(find(actions, "water").children ?? []), ["ball", "region"]);
  assert.equal(labRingTitle(context()), "Water");
});

test("EDIT on a region is about that region", () => {
  const actions = labRingActions(context({ regionId: REGION.id }));
  assert.deepEqual(ids(actions), ["select", "delete", "inspect-cell", "slice-visuals"]);
  assert.equal(find(actions, "delete").tone, "danger",
    "the one irreversible wedge is toned the same on every ring in the product");
  const select = find(actions, "select").effect;
  assert.equal(select?.kind, "select");
  assert.equal(select?.kind === "select" && select.selection?.id,
    "refinement-region-advance-region-1");
  assert.equal(labRingTitle(context({ regionId: REGION.id })), "Enforcement region");
});

test("a region the document does not have offers no verbs for it", () => {
  const actions = labRingActions(context({ regionId: "advance-region-9" }));
  assert.deepEqual(ids(actions), ["inspect-cell", "slice-visuals"]);
});

test("off the slice the wedges stay, disabled", () => {
  const actions = labRingActions(context({ at: null }));
  assert.deepEqual(ids(actions), ["water", "inspect-cell", "slice-visuals"]);
  assert.equal(actionIsChoosable(find(actions, "inspect-cell")), false);
  const ball = find(find(actions, "water").children ?? [], "ball");
  assert.equal(actionIsChoosable(ball), false, "a ball needs a cell to land in");
  // The box does not: a drag names its own corners.
  assert.equal(actionIsChoosable(find(find(actions, "water").children ?? [], "region")), true);
  assert.equal(labRingTitle(context({ at: null })), "The slice");
});

test("a full document offers the box and says why it cannot be drawn", () => {
  const full: LabRegionDocument = { ...DOC, regions: Array.from(
    { length: labRegionSpace.capacity },
    (_value, index) => ({ ...REGION, id: `advance-region-${index}` })) };
  const region = find(find(labRingActions(context({ doc: full })), "water").children ?? [],
    "region");
  assert.equal(region.enabled, false);
  assert.match(region.hint ?? "", /delete one/, "the answer is readable before the click");
});

test("Visuals carries the surfaces and the overlays, and nothing else", () => {
  const children = find(labRingActions(context()), "slice-visuals").children ?? [];
  assert.ok(children.every(child =>
    child.id.startsWith("slice-surface-") || child.id.startsWith("slice-overlay-")));
  assert.ok(children.some(child => child.id === "slice-overlay-fraction"));
});

test("an imposed surface is stated rather than offered", () => {
  const children = find(labRingActions(context({
    surfaceImposed: true, surfaceView: "direct-level-set" })), "slice-visuals").children ?? [];
  for (const child of children.filter(one => one.id.startsWith("slice-surface-"))) {
    assert.equal(child.enabled, false);
    assert.match(child.hint ?? "", /publishes/);
  }
});

test("an overlay this transport cannot draw is offered disabled, never dropped", () => {
  const children = find(labRingActions(context({ overlaysOffered: ["fraction"] })),
    "slice-visuals").children ?? [];
  assert.equal(find(children, "slice-overlay-normal").enabled, false);
  assert.equal(find(children, "slice-overlay-fraction").enabled, true);
});

test("an overlay that is on says so in its label", () => {
  const children = find(labRingActions(context({
    overlays: new Set<SliceOverlayId>(["fraction"]) })), "slice-visuals").children ?? [];
  assert.match(find(children, "slice-overlay-fraction").label, /off$/);
  assert.doesNotMatch(find(children, "slice-overlay-normal").label, /off$/);
});

test("the lab composes only arm, select and its own host verbs", () => {
  /* The boundary the `host` arm exists to hold. Anything else would be an
   * effect `performEditorAction` answers by reaching the 3-D simulation
   * singleton, and this page has no world for it to act on. */
  const every: LabRingContext[] = [
    context({ mode: "camera" }),
    context(),
    context({ at: null }),
    context({ regionId: REGION.id }),
    context({ surfaceImposed: true, surfaceView: "direct-level-set" }),
  ];
  const seen = new Set<string>();
  const walk = (actions: readonly EditorAction[]): void => {
    for (const action of actions) {
      if (action.effect) {
        seen.add(action.effect.kind);
        assert.ok(["arm", "select", "host"].includes(action.effect.kind),
          `${action.id} carries a "${action.effect.kind}" effect the lab cannot perform`);
        if (action.effect.kind === "host") {
          assert.ok(labHostVerb(action.effect),
            `${action.id} carries a host effect the lab's own union does not recognise`);
        }
      }
      walk(action.children ?? []);
    }
  };
  for (const one of every) walk(labRingActions(one));
  assert.deepEqual([...seen].sort(), ["arm", "host", "select"]);
});

test("a host effect is validated on the way out, never cast", () => {
  assert.equal(labHostVerb({ kind: "host", id: "drop-ball", payload: undefined }), undefined);
  assert.equal(labHostVerb({ kind: "host", id: "drop-ball", payload: { id: "pin-cell", at: AT } }),
    undefined);
  assert.equal(labHostVerb({ kind: "arm", gesture: "region-draw" }), undefined);
});

/* ---- the performer, over a real `EditorHost` ------------------------- */

interface Note { readonly call: string; readonly payload: unknown }

function recordingHost(log: Note[]): EditorHost<LabRegionDocument, LabRegionDocument> {
  return {
    id: "lab",
    commit: (label, next) => log.push({ call: "commit", payload: { label, next } }),
    liquid: { dropAt: (centre, radius) => log.push({ call: "dropAt", payload: { centre, radius } }) },
    select: (selection: EditorSelection | undefined, openControls?: boolean) =>
      log.push({ call: "select", payload: { selection, openControls } }),
    arm: gesture => log.push({ call: "arm", payload: gesture }),
    notice: () => {},
  };
}

function performer(log: Note[], page: Partial<Parameters<typeof labActionPerformer>[1]> = {}) {
  return labActionPerformer(recordingHost(log), {
    doc: DOC,
    dropRadius_cells: 6,
    pinCell: () => {},
    setSurfaceView: () => {},
    toggleOverlay: () => {},
    ...page,
  });
}

/** The performer takes a session it no longer reads; the host is the world now. */
const SESSION = undefined as unknown as PaneSession;

test("arming puts the selection down first, so handles cannot claim the press", () => {
  const log: Note[] = [];
  performer(log)({ kind: "arm", gesture: "region-draw" }, SESSION);
  assert.deepEqual(log.map(note => note.call), ["select", "arm"]);
  assert.equal(log[0]?.payload && (log[0].payload as { selection: unknown }).selection, undefined);
  assert.equal(log[1]?.payload, "region-draw");
});

test("selecting disarms, and opens the controls the wedge promised", () => {
  const log: Note[] = [];
  const effect: EditorActionEffect = { kind: "select",
    selection: { kind: "refinement-region", id: "refinement-region-advance-region-1" },
    openControls: true };
  performer(log)(effect, SESSION);
  assert.deepEqual(log.map(note => note.call), ["arm", "select"]);
  assert.deepEqual(log[1]?.payload, { selection: effect.kind === "select" ? effect.selection : undefined,
    openControls: true });
});

test("a ball drops through the host's liquid capability and stays armed", () => {
  const log: Note[] = [];
  const ball = find(find(labRingActions(context()), "water").children ?? [], "ball");
  assert.ok(ball.effect);
  performer(log)(ball.effect, SESSION);
  assert.deepEqual(log, [
    { call: "dropAt", payload: { centre: AT, radius: 6 } },
    { call: "arm", payload: "fluid-ball" },
  ]);
});

test("removing a box is a whole-document commit through the shared RegionSpace", () => {
  const log: Note[] = [];
  const remove = find(labRingActions(context({ regionId: REGION.id })), "delete");
  assert.ok(remove.effect);
  performer(log)(remove.effect, SESSION);
  assert.equal(log[0]?.call, "commit");
  const committed = (log[0]?.payload as { next: LabRegionDocument }).next;
  assert.deepEqual(committed.regions, [], "the box is gone from the list the controller takes");
  assert.equal(committed.nx, DOC.nx, "and the lattice it was measured on is untouched");
  assert.equal(log[1]?.call, "select", "the selection it left behind goes down with it");
});

test("the three readings that are genuinely this page's reach the page", () => {
  const seen: string[] = [];
  const run = performer([], {
    pinCell: () => seen.push("pin"),
    setSurfaceView: () => seen.push("surface"),
    toggleOverlay: () => seen.push("overlay"),
  });
  const visuals = find(labRingActions(context()), "slice-visuals").children ?? [];
  run(find(labRingActions(context()), "inspect-cell").effect!, SESSION);
  run(find(visuals, "slice-surface-shared-rdf").effect!, SESSION);
  run(find(visuals, "slice-overlay-fraction").effect!, SESSION);
  assert.deepEqual(seen, ["pin", "surface", "overlay"]);
});
