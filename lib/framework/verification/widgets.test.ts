import assert from "node:assert/strict";
import test from "node:test";
import { collectWidgets, placementAtPoint, resolveWidgetLayout, type ControlWidget, type WidgetProvider } from "../widgets";

const lever = (id: string, anchor: ControlWidget["placement"]["anchor"] = "bottom", order?: number): ControlWidget => ({
  id, target: `body:${id}`, label: id, placement: { anchor, order },
  channels: [{ kind: "axis1", id: "motor", label: "Motor", min: 0, max: 100, step: 5, value: 100 }],
});
const provider = (widgets: ControlWidget[]): WidgetProvider<void> => ({ id: "test", widgets: () => widgets, input: () => {} });

test("layers scene then user placements over provider defaults, grouped by anchor in order", () => {
  const widgets = collectWidgets([provider([lever("a"), lever("b"), lever("c", "bottom", -1)])], undefined);
  assert.equal(widgets[0]?.provider, "test");
  const scene = { a: { anchor: "bottom-left" as const }, b: { anchor: "bottom-right" as const } };
  const groups = resolveWidgetLayout(widgets, scene, { b: { hidden: true } });
  assert.deepEqual(groups.map(group => [group.anchor, group.widgets.map(widget => widget.id)]), [["bottom-left", ["a"]], ["bottom", ["c"]]]);
  const stacked = resolveWidgetLayout(widgets);
  assert.deepEqual(stacked[0]?.widgets.map(widget => widget.id), ["c", "a", "b"]);
});

test("an exact placement stands alone; a later anchor without an offset drops the earlier offset", () => {
  const widgets = collectWidgets([provider([lever("a"), lever("b")])], undefined);
  const placed = resolveWidgetLayout(widgets, { a: { anchor: "top-right", offset_px: { x: 20, y: 30 } } });
  assert.deepEqual(placed.map(group => [group.anchor, group.offset_px, group.widgets.length]), [["top-right", { x: 20, y: 30 }, 1], ["bottom", undefined, 1]]);
  const moved = resolveWidgetLayout(widgets, { a: { anchor: "top-right", offset_px: { x: 20, y: 30 } } }, { a: { anchor: "left" } });
  assert.equal(moved.find(group => group.widgets[0]?.id === "a")?.offset_px, undefined);
});

test("rejects bad widget data instead of coercing it", () => {
  assert.throws(() => collectWidgets([provider([lever("a"), lever("a")])], undefined), /Duplicate/);
  const bad = { ...lever("a"), channels: [{ kind: "axis1" as const, id: "m", label: "M", min: 1, max: 1, value: 1 }] };
  assert.throws(() => collectWidgets([provider([bad])], undefined), /Invalid range/);
  const rest = { ...lever("a"), channels: [{ kind: "axis1" as const, id: "m", label: "M", min: 0, max: 1, value: 0, rest: 2 }] };
  assert.throws(() => collectWidgets([provider([rest])], undefined), /Rest outside/);
  const widgets = collectWidgets([provider([lever("a")])], undefined);
  assert.throws(() => resolveWidgetLayout(widgets, { a: { anchor: "middle" as never } }), /Unknown anchor/);
  assert.throws(() => resolveWidgetLayout(widgets, { a: { offset_px: { x: NaN, y: 0 } } }), /Non-finite/);
});

test("a drop lands at the nearest region, offset from its anchor", () => {
  const size = { width: 900, height: 600 }, box = { width: 100, height: 200 };
  assert.deepEqual(placementAtPoint({ x: 80, y: 480 }, size, box), { anchor: "bottom-left", offset_px: { x: 30, y: 20 } });
  assert.deepEqual(placementAtPoint({ x: 820, y: 130 }, size, box), { anchor: "top-right", offset_px: { x: 30, y: 30 } });
  assert.deepEqual(placementAtPoint({ x: 470, y: 300 }, size, box), { anchor: "center", offset_px: { x: 20, y: 0 } });
});
