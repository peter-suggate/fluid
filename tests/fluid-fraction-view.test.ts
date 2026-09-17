/**
 * One definition of the liquid-fraction view, consumed by two renderers.
 *
 * The claim under test is not that the numbers are right — the lab's own tests
 * pin the arithmetic — but that there is only one set of them. A threshold or a
 * colour that reappears as a literal in the shader, in the legend or in the
 * canvas is the drift this module exists to prevent, and the WGSL is where it
 * is hardest to see: the overlay is a template string, so a retyped `1e-6` in
 * the wrong branch compiles perfectly and simply draws something else.
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  FRACTION_FLOOR, FRACTION_LIQUID_KNEE, FRACTION_OVERFULL, FRACTION_VIEW_BANDS,
  cellFillIsOverCapacity, fractionBand, fractionBandPaint, fractionReadout,
  fractionViewShaderConstants, wgslDisplayColor, wgslFloatLiteral,
} from "../lib/core/fluid-fraction-view";
import { gridOverlayVisualizations } from "../lib/core/grid-overlay-visualizations";
import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";

/** The `fieldMode == 21` arm, which is the fraction view in the 3-D overlay. */
function volumeLevelSetBranch(): string {
  const start = gridOverlayShader.indexOf("} else if (fieldMode == 21) {");
  assert.ok(start > 0, "the overlay no longer has a mode-21 branch to check");
  const end = gridOverlayShader.indexOf("} else if (fieldMode == 10) {", start);
  assert.ok(end > start, "the mode-21 branch no longer ends where this test expects");
  return gridOverlayShader.slice(start, end);
}

test("every V/K lands in exactly one band, at the declared thresholds", () => {
  assert.equal(fractionBand(0), "vacuum");
  assert.equal(fractionBand(FRACTION_FLOOR), "vacuum", "the floor itself is still vacuum");
  assert.equal(fractionBand(FRACTION_FLOOR * 2), "dilute");
  assert.equal(fractionBand(FRACTION_LIQUID_KNEE - 1e-9), "dilute");
  assert.equal(fractionBand(FRACTION_LIQUID_KNEE), "liquid",
    "half a cell is what the projection calls liquid");
  assert.equal(fractionBand(1), "liquid");
  assert.equal(fractionBand(FRACTION_OVERFULL), "liquid", "at the tolerance, not past it");
  assert.equal(fractionBand(1.04), "overfull");
  assert.equal(fractionBand(Number.NaN), "vacuum",
    "an unreadable record must not be drawn as a fault");

  assert.deepEqual(FRACTION_VIEW_BANDS.map(paint => paint.band),
    ["vacuum", "dilute", "liquid", "overfull"],
    "every band a value can land in carries paint");
  assert.equal(new Set(FRACTION_VIEW_BANDS.map(paint => paint.tone)).size,
    FRACTION_VIEW_BANDS.length, "two bands sharing a tone cannot be told apart");
});

test("one overfull threshold answers for the wash, the box and the readout", () => {
  /* The three that disagreed: the lab washed at 1 + 1e-4, boxed at 1 + 1e-6,
   * and the shader hatched at 1.000001. Nothing may reintroduce a fourth. */
  assert.equal(FRACTION_OVERFULL, 1 + FRACTION_FLOOR);
  assert.equal(cellFillIsOverCapacity(FRACTION_OVERFULL), false);
  assert.equal(cellFillIsOverCapacity(1 + 2e-6), true);
  assert.equal(cellFillIsOverCapacity(Number.NaN), false);
  for (const fill of [0, 1e-7, 1e-3, 0.5, 1, 1.000002, 1.04, 3]) {
    assert.equal(cellFillIsOverCapacity(fill), fractionBand(fill) === "overfull",
      `the over-capacity predicate disagreed with the band at ${fill}`);
    assert.equal(fractionReadout(fill) === fill.toFixed(2) && fill > 1,
      fractionBand(fill) === "overfull",
      `the readout kept a different threshold at ${fill}`);
  }
});

test("a WGSL literal always carries a point or an exponent", () => {
  assert.equal(wgslFloatLiteral(1), "1.0", "a bare 1 is an integer in WGSL, not an f32");
  assert.equal(wgslFloatLiteral(0.5), "0.5");
  assert.equal(wgslFloatLiteral(1e-9), "1e-9");
  assert.throws(() => wgslFloatLiteral(Number.NaN), RangeError);
  assert.equal(wgslDisplayColor("#000000"), "vec3f(0.0, 0.0, 0.0)");
  assert.equal(wgslDisplayColor("#ffffff"), "vec3f(1.0, 1.0, 1.0)");
  assert.throws(() => wgslDisplayColor("rgb(1,2,3)"), RangeError);
});

test("the generated shader constants are the shared definition, exactly", () => {
  const source = fractionViewShaderConstants;
  assert.match(source, /const FRACTION_FLOOR: f32 = 0\.000001;/);
  assert.match(source, /const FRACTION_LIQUID_KNEE: f32 = 0\.5;/);
  assert.match(source, /const FRACTION_OVERFULL: f32 = 1\.000001;/);

  /* Every literal the block emits has to be a float to WGSL: an `f32`
   * initialized from an integer literal is a compile error, and no GPU is
   * available here to catch one. */
  for (const [, literal] of source.matchAll(/(?:=\s*|,\s*|\(\s*)(-?\d[\d.e+-]*)/g)) {
    assert.match(literal!, /[.e]/, `${literal} is an integer literal in an f32 context`);
    assert.ok(Number.isFinite(Number(literal!)), `${literal} does not parse as a number`);
  }

  for (const band of ["vacuum", "liquid", "overfull"] as const) {
    const paint = fractionBandPaint(band);
    assert.ok(source.includes(wgslDisplayColor(paint.swatch)),
      `the ${band} band's colour is not the one the shader is given`);
    /* Display space in, display space out: the channels the shader receives
     * are the swatch's own bytes, which is what lets a legend show the
     * pixels rather than an approximation of them. */
    const emitted = wgslDisplayColor(paint.swatch)
      .slice("vec3f(".length, -1).split(", ").map(Number);
    const hex = `#${emitted.map(channel =>
      Math.round(channel * 255).toString(16).padStart(2, "0")).join("")}`;
    assert.equal(hex, paint.swatch.toLowerCase());
  }
});

test("the overlay shader reads the shared constants and keeps no copy of them", () => {
  assert.ok(gridOverlayShader.includes(fractionViewShaderConstants),
    "the generated block is not in the shader, so the shader is not reading it");
  assert.match(gridOverlayShader, /const DENSITY_FLOOR: f32 = FRACTION_FLOOR;/,
    "the surface-density floor must be the fraction floor, not a second copy");
  assert.ok(!/log2\(0\.5 \/ DENSITY_FLOOR\)/.test(gridOverlayShader),
    "the residue ramp still spells the liquid knee as a literal");

  const branch = volumeLevelSetBranch();
  for (const name of ["FRACTION_EMPTY_DISPLAY", "FRACTION_LIQUID_DISPLAY", "FRACTION_OVERFULL"]) {
    assert.ok(branch.includes(name), `the mode-21 branch does not use ${name}`);
  }
  /* The magic this branch used to carry: two `vec3f` colours and the hatch
   * threshold. `0.5` and `1.4` survive as pixel feathers for the contour, which
   * are geometry rather than readings of V/K, so only the thresholds and the
   * colours are barred. */
  assert.ok(!/1\.000001/.test(branch), "the overfull threshold is spelled again in the branch");
  assert.ok(!/1e-6/.test(branch), "a second copy of the fraction floor is in the branch");
  assert.ok(!/sceneColor\(vec3f\(/.test(branch),
    "a colour is still authored inline instead of coming from the shared palette");
  assert.ok(!/\bmix\(ground, water, fraction\).*vec3f/.test(branch));

  /* As far as a text check can go without a GPU: the branch has to be a
   * balanced block, and every shared name it uses has to be declared. */
  const code = branch.replace(/\/\/[^\n]*/g, "");
  for (const [open, close] of [["{", "}"], ["(", ")"]] as const) {
    assert.equal(code.split(open).length, code.split(close).length,
      `the mode-21 branch has unbalanced ${open}${close}`);
  }
  for (const [, name] of gridOverlayShader.matchAll(/\b(FRACTION_[A-Z_]+)\b/g)) {
    assert.ok(new RegExp(`const ${name} *: *(f32|vec3f) *=`).test(gridOverlayShader),
      `${name} is used in the overlay but never declared`);
  }
});

test("the volume + level-set legend is derived from the shared bands", () => {
  const field = gridOverlayVisualizations.find(
    definition => definition.kind === "field" && definition.mode === "volume-levelset");
  if (!field || field.kind !== "field") assert.fail("volume-levelset field is missing");

  assert.equal(field.scalar?.band(1.04), "overfull",
    "the view has to classify a value the way both renderers do");
  assert.equal(field.scalar?.format(1.04), fractionReadout(1.04));
  assert.deepEqual(field.scalar?.bands, FRACTION_VIEW_BANDS);

  const liquid = fractionBandPaint("liquid"), overfull = fractionBandPaint("overfull");
  assert.equal(field.legend?.[0]?.swatch, liquid.swatch,
    "the V/K line must be the colour the shader fills a full cell with");
  assert.equal(field.legend?.[0]?.label, liquid.label);
  assert.equal(field.swatch, liquid.swatch, "the toggle chip is the same colour");
  assert.ok(field.legend?.[2]?.swatch.includes(overfull.swatch),
    "the overcapacity hatch must be hatched in the overfull band's own colour");
  assert.equal(field.legend?.[2]?.label, overfull.label);
  /* Not bands, and deliberately still authored: the zero contour is the level
   * set drawn over the fill, and the lattice is the topology under it. */
  assert.deepEqual(field.legend?.slice(1, 2).concat(field.legend.slice(3))
    .map(entry => entry.label),
  ["φ = 0 — level-set interface", "accepted adaptive grid"]);
});
