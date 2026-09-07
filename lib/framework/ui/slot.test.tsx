import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { composeFeatures } from "../composition";
import { ComposedFeatureSlot, type FeatureControlViewProps } from "./slot";

const composition = composeFeatures({ features: [{ id: "test", controls: [
  { id: "a", kind: "toggle", label: "A" }, { id: "b", kind: "number", label: "B" },
], placements: [
  { control: "a", slot: "arbitrary-panel", order: 2, presentation: "expanded" },
  { control: "b", slot: "arbitrary-panel", order: 1, presentation: "compact" },
  { control: "a", slot: "toolstrip", presentation: "compact" },
] }] });
const view = ({ control, placement }: FeatureControlViewProps) => createElement("span", null, `${control.label}:${placement.presentation}`);

test("arbitrary slots resolve owner controls, order and representation", () => {
  const views = { "test/a": view, "test/b": view };
  assert.equal(renderToStaticMarkup(createElement(ComposedFeatureSlot, { composition, views, slot: "arbitrary-panel" })),
    "<span>B:compact</span><span>A:expanded</span>");
  assert.equal(renderToStaticMarkup(createElement(ComposedFeatureSlot, { composition, views, slot: "toolstrip" })),
    "<span>A:compact</span>");
});

test("unbound installed controls fail explicitly instead of disappearing", () => {
  assert.throws(() => renderToStaticMarkup(createElement(ComposedFeatureSlot, { composition, views: {}, slot: "toolstrip" })), /Missing feature UI binding: test\/a/);
});
