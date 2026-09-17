import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { gridOverlayVisualizations, VOLUME_LEVELSET_OVERLAY_MODE_CODE } from "../lib/core/grid-overlay-visualizations";
import { fractionBandPaint } from "../lib/core/fluid-fraction-view";
import { pickFieldOverlay } from "../lib/core/field-overlay-pick";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";

test("volume and level-set slice is catalogued for Sparse Geometric with mode code 21", () => {
  const field = gridOverlayVisualizations.find(
    definition => definition.kind === "field" && definition.mode === "volume-levelset",
  );

  if (!field || field.kind !== "field") assert.fail("volume-levelset field is missing");
  assert.equal(field?.label, "Volume + level set");
  assert.equal(field?.axis, "z");
  assert.equal(field.sliceOnly, true);
  assert.equal(field.icon, "surface");
  assert.equal(VOLUME_LEVELSET_OVERLAY_MODE_CODE, 21);
  assert.ok(adaptiveMassMethod.supportedFieldModes?.includes("volume-levelset"));

  /* The legend is read beside the picture, so its V/K swatch has to be the
   * colour the shader actually fills a full cell with — not a hand-kept copy
   * that drifted from it, which is what it was. */
  assert.equal(field.legend?.[0]?.swatch, fractionBandPaint("liquid").swatch,
    "the V/K legend swatch must be the shared fraction view's liquid colour");
  assert.equal(field.scalar?.band(0.25), "dilute",
    "the view carries the shared definition, not just a picture of it");
});

test("selecting the slice-only view from VOL restores its authored plane", () => {
  assert.deepEqual(pickFieldOverlay(
    { mode: "density", axis: "volume" },
    { mode: "volume-levelset", axis: "z", sliceOnly: true },
    true,
    "z",
  ), { mode: "volume-levelset", axis: "z", slice: 0.5 });
});

test("selecting the slice-only view preserves an existing plane", () => {
  assert.deepEqual(pickFieldOverlay(
    { mode: "density", axis: "x" },
    { mode: "volume-levelset", axis: "z", sliceOnly: true },
    true,
    "z",
  ), { mode: "volume-levelset" });
});

test("volume and level-set slice survives the URL", () => {
  const parsed = parseQueryState("?gridMode=volume-levelset&grid=x&gridSlice=0.25");
  assert.equal(parsed.ui.gridOverlayMode, "volume-levelset");
  assert.equal(parsed.ui.gridOverlayAxis, "x");
  assert.equal(parsed.ui.gridOverlaySlice, 0.25);

  const serialized = serializeQueryState("", parsed, parsed, parsed.ui);
  assert.equal(new URLSearchParams(serialized).get("gridMode"), "volume-levelset");
});

test("a direct VOL URL for a slice-only view canonicalizes to the Z midpoint", () => {
  const parsed = parseQueryState("?gridMode=volume-levelset&grid=volume&gridSlice=0.23");
  assert.equal(parsed.ui.gridOverlayMode, "volume-levelset");
  assert.equal(parsed.ui.gridOverlayAxis, "z");
  assert.equal(parsed.ui.gridOverlaySlice, 0.5);

  const serialized = serializeQueryState("", parsed, parsed, parsed.ui);
  const query = new URLSearchParams(serialized);
  assert.equal(query.get("grid"), "z");
  assert.equal(query.get("gridSlice"), null);
});
