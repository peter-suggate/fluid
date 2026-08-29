import assert from "node:assert/strict";
import test from "node:test";

// The method registry is installed by side effect, and `session/session.ts`
// reaches it while building a store. Without this the first `createPaneSession`
// throws "No simulation methods are installed" rather than failing a claim.
import "../lib/methods";

import {
  compareDiffRows,
  dropCompareOverride,
  moveCompareOverrideToA,
  promoteCompareDiff,
  startCompareSync,
  swapComparePanes,
  type CompareStateStore,
} from "../lib/core/compare/compare-model";
import {
  COMPARE_ABSENT,
  compareGroupForKey,
  compareQueryEntries,
  INITIAL_COMPARE_STATE,
  parseCompareQuery,
  type CompareState,
} from "../lib/core/compare/compare-query";
import { getMethod } from "../lib/core/method-registry";
import type { SceneDescription } from "../lib/core/model";
import { createBodyDescription } from "../lib/core/rigid-body";
import { createPaneSession, type PaneSession } from "../lib/core/session/session";
import { getScenePreset } from "../lib/core/scenes";
import { createSceneQueryLayerCache, parseQueryState, serializeQueryState } from "../lib/core/url-state";

/**
 * The compare record as a plain object rather than the page's shell store, so
 * one test file's claims cannot leak into another's through a module singleton.
 */
function fakeStore(initial: CompareState = INITIAL_COMPARE_STATE): CompareStateStore {
  let state: CompareState = { ...initial, active: true };
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    setState: (next) => { state = next; for (const listener of [...listeners]) listener(); },
    subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
  };
}

const layers = new WeakMap<PaneSession, ReturnType<typeof createSceneQueryLayerCache>>();

/** The canonical query for one pane — the same serialization the mirror reads. */
function search(session: PaneSession): string {
  let layer = layers.get(session);
  if (!layer) { layer = createSceneQueryLayerCache(); layers.set(session, layer); }
  const scene = session.scene.getState();
  return serializeQueryState(
    "",
    { presetId: scene.presetId, scene: scene.scene },
    session.method.getState(),
    session.ui.getState(),
    { view: "studio" },
    layer({ presetId: scene.presetId, scene: scene.scene }),
  );
}

function panes() {
  return { a: createPaneSession("a"), b: createPaneSession("b") };
}

/** A registered method that is not the one a fresh pane opens on. */
function otherMethodId(session: PaneSession): string {
  const current = session.method.getState().methodId;
  const other = current === "uniform" ? "losasso" : "uniform";
  return other;
}

test("opening compare with an empty diff makes B a copy of A", () => {
  const { a, b } = panes();
  // Something in A that is not the preset default, so "B copies A" is a claim
  // about mirroring rather than about two panes both being untouched.
  a.ui.getState().setGridOverlayAxis("y");
  const sync = startCompareSync(a, b, fakeStore());
  assert.equal(search(b), search(a));
  assert.equal(b.ui.getState().gridOverlayAxis, "y");
  sync.stop();
});

test("changing B's solver records exactly one override", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const other = otherMethodId(b);
  b.method.getState().setMethodId(other);
  assert.deepEqual(Object.keys(store.getState().diff), ["method"]);
  assert.equal(store.getState().diff.method, other);
  assert.equal(a.method.getState().methodId, other === "uniform" ? "adaptive-mass" : a.method.getState().methodId);
  assert.notEqual(a.method.getState().methodId, other);
  sync.stop();
});

test("a solver key is config: it never moves both panes, whatever the padlocks say", () => {
  assert.equal(compareGroupForKey("method"), "config");
  assert.equal(compareGroupForKey("scene.container.width_m"), "config");
  assert.equal(compareGroupForKey("camera.azimuth"), "view");
  assert.equal(compareGroupForKey("gridMode"), "cut");
  assert.equal(compareGroupForKey("overlay"), "instrument");
  assert.equal(compareGroupForKey("quality"), "look");
  assert.equal(compareGroupForKey("svoStage"), "look");
});

test("editing A with an empty diff edits both panes", () => {
  const { a, b } = panes();
  const sync = startCompareSync(a, b, fakeStore());
  const methodId = a.method.getState().methodId;
  const [spec] = [...registeredParams(a)];
  assert.ok(spec, "the default method declares at least one numeric parameter");
  a.method.getState().setParam(methodId, spec.key, spec.value);
  assert.equal(b.method.getState().overrides[methodId]?.[spec.key], spec.value);
  assert.equal(search(b), search(a));
  sync.stop();
});

test("with View unlinked, B's camera is an override", () => {
  const { a, b } = panes();
  const store = fakeStore({ ...INITIAL_COMPARE_STATE, links: { ...INITIAL_COMPARE_STATE.links, view: false } });
  const sync = startCompareSync(a, b, store);
  const before = a.ui.getState().camera;
  b.ui.getState().setCamera((current) => ({ ...current, azimuth_rad: current.azimuth_rad + 0.5 }));
  assert.ok(Object.keys(store.getState().diff).some((key) => key.startsWith("camera.")),
    `expected a camera override, got ${JSON.stringify(store.getState().diff)}`);
  assert.equal(a.ui.getState().camera.azimuth_rad, before.azimuth_rad);
  sync.stop();
});

test("with View linked, B's camera moves A and records nothing", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const target = a.ui.getState().camera.azimuth_rad + 0.5;
  b.ui.getState().setCamera((current) => ({ ...current, azimuth_rad: target }));
  assert.deepEqual(store.getState().diff, {});
  assert.equal(a.ui.getState().camera.azimuth_rad, target);
  assert.equal(b.ui.getState().camera.azimuth_rad, target);
  sync.stop();
});

test("dropping an override returns B to A", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const other = otherMethodId(b);
  const original = a.method.getState().methodId;
  b.method.getState().setMethodId(other);
  dropCompareOverride(store, "method");
  assert.deepEqual(store.getState().diff, {});
  assert.equal(b.method.getState().methodId, original);
  assert.equal(search(b), search(a));
  sync.stop();
});

test("moving an override to A leaves both panes on B's value", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const other = otherMethodId(b);
  b.method.getState().setMethodId(other);
  moveCompareOverrideToA(store, a, "method");
  assert.deepEqual(store.getState().diff, {});
  assert.equal(a.method.getState().methodId, other);
  assert.equal(b.method.getState().methodId, other);
  sync.stop();
});

test("keeping B promotes the diff onto A and empties it", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const other = otherMethodId(b);
  b.method.getState().setMethodId(other);
  promoteCompareDiff(store, a);
  assert.deepEqual(store.getState().diff, {});
  assert.equal(a.method.getState().methodId, other);
  sync.stop();
});

test("swapping inverts the diff", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const original = a.method.getState().methodId;
  const other = otherMethodId(b);
  b.method.getState().setMethodId(other);
  swapComparePanes(store, a);
  assert.equal(a.method.getState().methodId, other);
  // A was on the product default, which the serializer writes as the *absence*
  // of the key — so the inverted diff has to be able to say "absent", or B
  // would silently re-inherit A's new solver instead of taking A's old one.
  assert.equal(store.getState().diff.method, COMPARE_ABSENT);
  assert.equal(b.method.getState().methodId, original);
  sync.stop();
});

test("B is seeded from A even when the diff forks the document", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  b.scene.getState().setScene({
    ...b.scene.getState().scene,
    container: { ...b.scene.getState().scene.container, width_m: b.scene.getState().scene.container.width_m * 1.5 },
  });
  assert.ok(Object.keys(store.getState().diff).includes("scene.container.width_m"),
    `expected a container override, got ${JSON.stringify(store.getState().diff)}`);
  // The seed is never an override: two runs seeded differently are not an A/B.
  assert.equal(b.scene.getState().scene.randomSeed, a.scene.getState().scene.randomSeed);
  assert.equal(store.getState().diff["scene.randomSeed"], undefined);
  sync.stop();
});

test("a mirrored document is adopted; a mirrored parameter is not", () => {
  const { a, b } = panes();
  const adopted: { session: PaneSession; previous: SceneDescription }[] = [];
  const sync = startCompareSync(a, b, fakeStore(), {
    onSceneAdopted: (session, previous) => { adopted.push({ session, previous }); },
  });
  // A parameter is a setting *of* the running solver: the mirror writes B's
  // method store and there is nothing for the controller to reconcile.
  const methodId = a.method.getState().methodId;
  const [spec] = [...registeredParams(a)];
  assert.ok(spec, "the default method declares at least one numeric parameter");
  a.method.getState().setParam(methodId, spec.key, spec.value);
  assert.equal(adopted.length, 0, "a param mirror must not claim to be a document");

  // A body placed in A is a document edit, and the whole point of the callback:
  // B's *running* roster is built from the document B was started on, so
  // without this B draws the sphere and simulates a scene without one.
  const before = b.scene.getState().scene;
  const scene = a.scene.getState().scene;
  const sphere = createBodyDescription("sphere", 1, scene.container.height_m);
  a.scene.getState().patchScene({ rigidBodies: [...scene.rigidBodies, sphere] });
  assert.equal(b.scene.getState().scene.rigidBodies.at(-1)?.id, sphere.id);
  assert.equal(adopted.length, 1);
  assert.equal(adopted[0]!.session, b);
  // The document B *held*, not the one it now holds: the controller decides
  // between a roster adoption and a re-seed by comparing the two.
  assert.equal(adopted[0]!.previous, before);
  sync.stop();
});

test("moving a document override to A adopts it in A", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const adopted: PaneSession[] = [];
  const adopt = { onSceneAdopted: (session: PaneSession) => { adopted.push(session); } };
  const sync = startCompareSync(a, b, store, adopt);
  const scene = b.scene.getState().scene;
  b.scene.getState().patchScene({
    container: { ...scene.container, width_m: scene.container.width_m * 1.5 },
  });
  adopted.length = 0;
  moveCompareOverrideToA(store, a, "scene.container.width_m", adopt);
  // The `⇄` writes A's document, so A is the pane whose solver is now behind.
  assert.ok(adopted.includes(a), "the receiving pane must adopt what it was handed");
  sync.stop();
});

// ---- per-pane scenes -------------------------------------------------------
//
// The coarsest diff the mode can carry, and the one the scene selector writes:
// a pane's `scene` key is its preset id, so choosing another scene in B is an
// ordinary config override and nothing about it is a second document. The
// selector itself goes through the controller (`openSceneCard`), which is a GPU
// path; these drive the store write that path performs, which is the half the
// mirror can see.

/** A second authored scene, cheap to build, that is not the one a pane opens on. */
const SECOND_SCENE_ID = "water-box-tank-fill";

test("choosing another scene in B records the scene key and leaves A alone", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const before = a.scene.getState();
  const preset = getScenePreset(SECOND_SCENE_ID);
  b.scene.getState().setScene(preset.create(), preset.id);

  assert.equal(store.getState().diff.scene, SECOND_SCENE_ID);
  assert.equal(b.scene.getState().presetId, SECOND_SCENE_ID);
  // A is untouched — document *and* identity. A scene chosen for one pane that
  // silently re-opened the other is the exact failure the diff rule exists to
  // prevent, and it is invisible in a screenshot of two similar tanks.
  assert.equal(a.scene.getState().scene, before.scene);
  assert.equal(a.scene.getState().presetId, before.presetId);
  // One key, not a serialized second scene: the two documents are each their
  // own preset's, so nothing else has to be spelled out.
  assert.deepEqual(Object.keys(store.getState().diff), ["scene"]);
  sync.stop();
});

test("opening a scene in A carries it into B while the diff is empty", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const preset = getScenePreset(SECOND_SCENE_ID);
  a.scene.getState().setScene(preset.create(), preset.id);

  assert.equal(b.scene.getState().presetId, SECOND_SCENE_ID);
  // The very object, not a rebuild of it from the address: what crosses the
  // seam has to include the authoring a URL cannot express.
  assert.equal(b.scene.getState().scene, a.scene.getState().scene);
  assert.deepEqual(store.getState().diff, {});
  sync.stop();
});

test("opening a scene in A leaves B on the scene B chose", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const forked = getScenePreset(SECOND_SCENE_ID);
  b.scene.getState().setScene(forked.create(), forked.id);
  const held = b.scene.getState().scene;

  const third = getScenePreset("bounded-pool-transfer");
  a.scene.getState().setScene(third.create(), third.id);

  assert.equal(a.scene.getState().presetId, third.id);
  // B keeps its own. The document may be rebuilt from the address — the diff
  // has forked it, so the mirror has no exact document to hand over — but the
  // scene it names must not move.
  assert.equal(b.scene.getState().presetId, SECOND_SCENE_ID);
  assert.equal(b.scene.getState().scene.sceneId, held.sceneId);
  assert.equal(store.getState().diff.scene, SECOND_SCENE_ID);
  sync.stop();
});

test("moving the scene override to A opens B's scene in A", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const preset = getScenePreset(SECOND_SCENE_ID);
  b.scene.getState().setScene(preset.create(), preset.id);

  // The `\u21c4` on the scene row: an experiment that turned out well is adopted
  // by moving the one key that names it, and for an authored preset that key is
  // the whole document.
  moveCompareOverrideToA(store, a, "scene");
  assert.equal(a.scene.getState().presetId, SECOND_SCENE_ID);
  assert.equal(a.scene.getState().scene.sceneId, preset.create().sceneId);
  assert.equal(store.getState().diff.scene, undefined);
  assert.equal(b.scene.getState().presetId, SECOND_SCENE_ID);
  sync.stop();
});

/**
 * A whole address, both panes: pane A's canonical query with the `b.*` block on
 * the end. The same string `replaceQueryStateUrl` writes (`url-state.ts:824`)
 * and the same one `hydrate` reads back on a reload.
 */
function compareAddress(a: PaneSession, state: CompareState): string {
  const query = new URLSearchParams(search(a));
  for (const [key, value] of compareQueryEntries(state)) query.set(key, value);
  return `?${query.toString()}`;
}

/**
 * A reload, on the CPU: hydrate a fresh pane A from the address exactly as
 * `hydrate` does, restore the compare record with `parseCompareQuery`, and let
 * `startCompareSync`'s opening pass rebuild pane B from the diff.
 */
function reload(address: string): { a: PaneSession; b: PaneSession; store: CompareStateStore; sync: ReturnType<typeof startCompareSync> } {
  const { a, b } = panes();
  const state = parseQueryState(address);
  a.method.setState({ methodId: state.methodId, quality: state.quality, overrides: state.overrides });
  a.scene.getState().setScene(state.scene, state.presetId);
  a.ui.setState(state.ui);
  const restored = parseCompareQuery(address);
  const store = fakeStore(restored);
  return { a, b, store, sync: startCompareSync(a, b, store) };
}

test("two panes on two scenes survive the address", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const preset = getScenePreset(SECOND_SCENE_ID);
  b.scene.getState().setScene(preset.create(), preset.id);
  const address = compareAddress(a, store.getState());
  const openedOn = a.scene.getState().presetId;
  sync.stop();

  // The whole of pane B, in one key. This is the claim the mode's URL rests on:
  // a compare link is an address, not a second document.
  assert.match(address, /[?&]b\.scene=water-box-tank-fill(&|$)/);

  const back = reload(address);
  assert.equal(back.a.scene.getState().presetId, openedOn, "pane A comes back on its own scene");
  assert.equal(back.b.scene.getState().presetId, SECOND_SCENE_ID, "pane B comes back on the scene it was given");
  assert.equal(back.store.getState().diff.scene, SECOND_SCENE_ID);
  // And the document actually is that preset's, not A's under B's name.
  assert.equal(back.b.scene.getState().scene.sceneId, preset.create().sceneId);
  back.sync.stop();
});

test("a scene chosen for A alone survives the address, with B still following", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const preset = getScenePreset(SECOND_SCENE_ID);
  a.scene.getState().setScene(preset.create(), preset.id);
  const address = compareAddress(a, store.getState());
  sync.stop();

  // Nothing forked, so there is no `b.scene` to write: the second pane is
  // restored by the bare flag and rebuilt from A.
  assert.match(address, /[?&]scene=water-box-tank-fill(&|$)/);
  assert.ok(!address.includes("b.scene="), "an unforked B carries no scene key");

  const back = reload(address);
  assert.equal(back.a.scene.getState().presetId, SECOND_SCENE_ID);
  assert.equal(back.b.scene.getState().presetId, SECOND_SCENE_ID);
  assert.deepEqual(back.store.getState().diff, {});
  back.sync.stop();
});

test("a scene's own document deltas ride beside the key they belong to", () => {
  // What the selector writes for a *saved* card: the origin preset id plus the
  // paths that differ from it. Both halves have to reach B, or a saved scene
  // would come back as the catalog scene it was minted from.
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const preset = getScenePreset(SECOND_SCENE_ID);
  const authored = preset.create();
  b.scene.getState().setScene({
    ...authored,
    container: { ...authored.container, fillFraction: 0.42 },
  }, preset.id);
  const address = compareAddress(a, store.getState());
  assert.equal(store.getState().diff.scene, SECOND_SCENE_ID);
  assert.equal(store.getState().diff["scene.container.fillFraction"], "0.42");
  sync.stop();

  const back = reload(address);
  assert.equal(back.b.scene.getState().presetId, SECOND_SCENE_ID);
  assert.equal(back.b.scene.getState().scene.container.fillFraction, 0.42);
  assert.notEqual(back.a.scene.getState().scene.container.fillFraction, 0.42);
  back.sync.stop();
});

test("a starter scene is not URL-representable, in either pane", () => {
  // Said out loud because the selector offers starter cards: `starter:<id>` is
  // not a registered preset, so the address round-trips to the default scene
  // for pane A and — by the same rule, deliberately, rather than by a scheme of
  // B's own — for pane B. See docs/ab-compare-handoff.md.
  const parsed = parseQueryState("?scene=starter:blank");
  assert.notEqual(parsed.presetId, "starter:blank");
  // The `b.*` half is not the thing that drops it: the key survives the block
  // and dies in exactly the same parse pane A's key dies in.
  assert.equal(parseCompareQuery("?b.scene=starter:blank").diff.scene, "starter:blank");
});

test("the diff round-trips through the address as b.* keys", () => {
  const state: CompareState = {
    active: true,
    diff: { method: "uniform", gridMode: COMPARE_ABSENT },
    links: { view: true, cut: false, instrument: true, look: false },
    focusedPane: "b",
  };
  const search = new URLSearchParams(compareQueryEntries(state).map(([key, value]) => [key, value])).toString();
  const parsed = parseCompareQuery(`?${search}`);
  assert.equal(parsed.active, true);
  assert.deepEqual(parsed.diff, state.diff);
  assert.deepEqual(parsed.links, state.links);
  // An empty diff still has to survive a reload, or opening compare and
  // reloading would silently drop the second pane.
  const bare = compareQueryEntries({ ...state, diff: {}, links: INITIAL_COMPARE_STATE.links });
  assert.deepEqual(bare, [["b", "1"]]);
  assert.equal(parseCompareQuery("?b=1").active, true);
  assert.equal(parseCompareQuery("?scene=dam-break").active, false);
});

test("the strip reads one row per override", () => {
  const { a, b } = panes();
  const store = fakeStore();
  const sync = startCompareSync(a, b, store);
  const other = otherMethodId(b);
  b.method.getState().setMethodId(other);
  const rows = compareDiffRows(store.getState(), a, b);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.key, "method");
  assert.equal(rows[0]!.group, "config");
  assert.equal(rows[0]!.valueB, other);
  sync.stop();
});

/** A numeric parameter of the pane's own method, with a value it does not hold. */
function* registeredParams(session: PaneSession) {
  const method = getMethod(session.method.getState().methodId);
  for (const spec of method.params) {
    if (spec.kind === "select") continue;
    yield { key: spec.key, value: (spec.min + spec.max) / 2 };
  }
}
