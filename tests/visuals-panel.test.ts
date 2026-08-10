import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { VISUALIZATION_FIELDS } from "../lib/visualization-catalog";

const panel = readFileSync(new URL("../components/VisualsPanel.tsx", import.meta.url), "utf8");
const performance = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
const shell = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");

test("Visuals starts with only the Losasso adaptive velocity view", () => {
  const velocity = VISUALIZATION_FIELDS.find((field) => field.mode === "adaptive-velocity-arrows");
  assert.ok(velocity);
  assert.equal(velocity.hidden, true,
    "the dedicated visual must stay out of Performance's generic field picker");
  assert.match(velocity.description, /translucent air-side arrows.*liquid velocity extrapolation/);
  assert.ok(velocity.legend?.some((entry) => /extrapolated liquid velocity in the air band/.test(entry.label)),
    "the legend must not imply the solver models air dynamics");
  assert.match(panel, /const ADAPTIVE_VELOCITY_MODE: GridOverlayMode = "adaptive-velocity-arrows"/);
  assert.match(panel, /VISUALIZATION_FIELDS\.find/);
  assert.doesNotMatch(panel, /VISUALIZATION_FIELDS\.filter|VISUALIZATION_FIELDS\.map/,
    "new catalog entries must not silently enter Visuals");
  assert.match(panel, /aria-label="Adaptive velocity view plane"/);
  assert.match(panel, />\s*VOLUME\s*<\/button>/);
  assert.match(panel, />HIDE<\/button>/);
  assert.match(performance, /\.filter\(\(field\) => !field\.hidden\)/);
});

test("Visuals is a dedicated viewport tab and panel", () => {
  assert.match(shell, />VISUALS<\/button>/);
  assert.match(shell, /rightPanel === "visuals" && <VisualsPanel \/>/);
  assert.match(panel, /data-testid="visuals-panel"/);
});
