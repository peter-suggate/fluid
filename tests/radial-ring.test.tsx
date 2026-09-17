import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  RadialRing,
  radialRingBack,
  radialRingKeyIntent,
  radialRingLevel,
  radialRingStep,
} from "../components/RadialRing";
import type { EditorAction } from "../lib/core/editor-action";
import type { RadialMenuState } from "../lib/core/stores/ui-store";

/**
 * The ring, with no host under it.
 *
 * `RadialRing` is now the whole menu minus the binding — the studio's session
 * and the advance lab's each supply their own — so what it owes both hosts is
 * pinned here once: the shape it draws, the two-level walk, and which wedges
 * are allowed to fire. The repo renders React with `renderToStaticMarkup`,
 * which cannot press a wedge, so the drawing is pinned from markup and the walk
 * is driven through the ring's own exported step functions — the same ones the
 * component calls, one line apart from where the state lands.
 */

const leaf = (id: string, label: string, extra: Partial<EditorAction> = {}): EditorAction => ({
  id, label, tone: "fluid", effect: { kind: "arm", gesture: "fluid-ball" }, ...extra,
});

const parent: EditorAction = {
  id: "water", label: "Water", tone: "fluid", icon: "water-ball",
  children: [leaf("ball", "Ball of water"), leaf("sheet", "Sheet")],
};
const disabled: EditorAction = leaf("wipe", "Wipe", { tone: "danger", enabled: false });

const menu: RadialMenuState = { x: 300, y: 200, title: "Tank", actions: [parent, disabled] };

const draw = (state: RadialMenuState, sink: EditorAction[] = []) =>
  renderToStaticMarkup(createElement(RadialRing, {
    menu: state,
    onChoose: (action: EditorAction) => sink.push(action),
    onClose: () => sink.push(leaf("closed", "closed")),
  }));

/**
 * One press, exactly as the component spends it: the step is data, the caller
 * either moves down a level or hands the leaf out. A wedge that answers `none`
 * reaches neither.
 */
function press(state: RadialMenuState, path: readonly number[], index: number, chosen: EditorAction[]): readonly number[] {
  const action = radialRingLevel(state, path)[index];
  assert.ok(action, `no wedge at ${index}`);
  const step = radialRingStep(action, index, path);
  if (step.kind === "descend") return step.path;
  if (step.kind === "choose") chosen.push(step.action);
  return path;
}

test("a root ring draws every wedge it was given", () => {
  const html = draw(menu);
  assert.match(html, /Water/);
  assert.match(html, /Wipe/);
  assert.equal((html.match(/class="radial-wedge/g) ?? []).length, 2);
  // Named above the ring, and the hub is the way out of the root.
  assert.match(html, /aria-label="Tank"/);
  assert.match(html, /aria-label="Close"/);
  // The sub-ring is advertised where it is opened from, not spelled out.
  assert.match(html, /aria-haspopup="menu"/);
  assert.doesNotMatch(html, /Ball of water/);
});

test("a wedge with children descends, and the hub climbs back", () => {
  const chosen: EditorAction[] = [];
  const path = press(menu, [], 0, chosen);
  assert.deepEqual(path, [0]);
  assert.equal(chosen.length, 0, "descending is not choosing");
  assert.deepEqual(radialRingLevel(menu, path).map((action) => action.label), ["Ball of water", "Sheet"]);
  // Drawn, the level the walk resolved is the children and nothing else.
  const html = draw({ ...menu, title: "Water", actions: radialRingLevel(menu, path) });
  assert.match(html, /Ball of water/);
  assert.match(html, /Sheet/);
  assert.doesNotMatch(html, /Wipe/);
  // The hub one level down is a step back to the root, not a close.
  assert.deepEqual(radialRingBack(path), { kind: "level", path: [] });
  assert.deepEqual(radialRingLevel(menu, []).map((action) => action.label), ["Water", "Wipe"]);
});

test("choosing a leaf hands that action to the host", () => {
  const chosen: EditorAction[] = [];
  const path = press(menu, press(menu, [], 0, chosen), 1, chosen);
  assert.deepEqual(path, [0], "a leaf leaves the ring where it was; the host closes it");
  assert.equal(chosen.length, 1);
  assert.equal(chosen[0]?.id, "sheet");
  assert.equal(chosen[0]?.effect?.kind, "arm");
});

test("Escape closes at the root and backs out below it", () => {
  assert.deepEqual(radialRingKeyIntent("Escape", false), { kind: "back" });
  assert.deepEqual(radialRingBack([]), { kind: "close" });
  assert.deepEqual(radialRingBack([0]), { kind: "level", path: [] });
});

test("the keyboard walks the ring and chooses what it lands on", () => {
  for (const key of ["ArrowRight", "ArrowDown"]) assert.deepEqual(radialRingKeyIntent(key, false), { kind: "focus", delta: 1 });
  for (const key of ["ArrowLeft", "ArrowUp"]) assert.deepEqual(radialRingKeyIntent(key, false), { kind: "focus", delta: -1 });
  assert.deepEqual(radialRingKeyIntent("Tab", false), { kind: "focus", delta: 1 });
  assert.deepEqual(radialRingKeyIntent("Tab", true), { kind: "focus", delta: -1 });
  for (const key of ["Enter", " "]) assert.deepEqual(radialRingKeyIntent(key, false), { kind: "choose" });
  // Anything else is left to the page: the ring only swallows what it uses.
  assert.deepEqual(radialRingKeyIntent("a", false), { kind: "ignore" });
});

test("a disabled wedge is drawn in its place but fires nothing", () => {
  const html = draw(menu);
  assert.match(html, /aria-disabled="true"/);
  assert.match(html, /data-choosable="false"/);
  const chosen: EditorAction[] = [];
  assert.deepEqual(press(menu, [], 1, chosen), [], "a disabled wedge does not descend either");
  assert.equal(chosen.length, 0);
  assert.deepEqual(radialRingStep(disabled, 1, []), { kind: "none" });
});

test("a caption with nothing behind it is drawn and inert", () => {
  const bare: EditorAction = { id: "note", label: "Nothing here", tone: "tank" };
  const html = draw({ ...menu, actions: [bare, leaf("ok", "Ok")] });
  assert.match(html, /Nothing here/);
  assert.deepEqual(radialRingStep(bare, 0, []), { kind: "none" });
});

test("an empty level draws nothing at all", () => {
  assert.equal(draw({ ...menu, actions: [] }), "");
});
