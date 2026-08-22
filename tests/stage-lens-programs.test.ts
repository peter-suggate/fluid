import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  STAGE_LENS_UNIFORM_NAME,
  STAGE_LENS_UNIFORM_TYPE,
  stageLensProblems,
  type LensLayerKind,
} from "../lib/core/stage-lens";
import { stageLensProgramWGSL } from "../lib/core/webgpu-stage-lens-overlay";
import {
  SPARSE_CM12_LENSES,
  SPARSE_CM12_STAGE_LENSES,
} from "../lib/methods/adaptive-mass/sparse-cm12-stage-lenses";

/**
 * A lens is a declaration plus a snippet, and neither half is executed until a
 * user opens the flyout on that stage. Everything here is what would otherwise
 * be found by opening it: a phase naming a program that no longer exists, a
 * classifier renamed out from under the layer template that calls it, an id
 * that does not match the mode the URL carries.
 *
 * The naga and Dawn gates compile the same text this asserts the shape of.
 * They answer "does it parse"; these answer "is it the module the framework
 * expects", which a shader that parses can still fail to be.
 */

/** Classifier and placement the framework's layer template calls, per kind. */
const REQUIRED: Readonly<Record<LensLayerKind,
  { readonly classifier: string; readonly placement: string }>> = Object.freeze({
    "row-glyph": { classifier: "lensRow", placement: "lensRowPlacement" },
    "cell-tile": { classifier: "lensCell", placement: "lensCellPlacement" },
    "cell-glyph": { classifier: "lensCellGlyph", placement: "lensCellPlacement" },
    "brick-frame": { classifier: "lensBrick", placement: "lensBrickPlacement" },
  });

const occurrences = (source: string, pattern: RegExp): number =>
  [...source.matchAll(pattern)].length;

test("every Sparse CM12 lens is declared consistently", () => {
  const ids = new Set<string>();
  for (const lens of SPARSE_CM12_LENSES) {
    assert.deepEqual(stageLensProblems(lens), [], `${lens.id} is not drawable`);
    assert.ok(!ids.has(lens.id), `duplicate lens id ${lens.id}`);
    ids.add(lens.id);
    assert.equal(lens.id, `stage-lens/${lens.stage}`,
      "a lens id is its overlay mode; the URL parser derives the stage from it");
    for (const phase of lens.phases) {
      assert.ok(phase.layers.length > 0,
        `${lens.id}: phase "${phase.id}" draws nothing, so the scrubber has a dead stop`);
    }
  }
});

/**
 * The roster is `satisfies Record<StageId, …>` at the declaration site, which
 * makes a missing stage a `tsc` error but says nothing about whether the lens
 * filed under a key is a lens *on that stage*. A mis-filed one arms the wrong
 * stage's taps and draws a confident wrong picture.
 */
test("the stage roster files every lens under its own stage", () => {
  for (const [stage, lens] of Object.entries(SPARSE_CM12_STAGE_LENSES)) {
    if (lens === null) continue;
    assert.equal(lens.stage, stage, `${lens.id} is filed under "${stage}"`);
  }
  const filed = Object.values(SPARSE_CM12_STAGE_LENSES).filter((lens) => lens !== null);
  assert.equal(SPARSE_CM12_LENSES.length, filed.length,
    "the renderer's list must be exactly the roster's non-null entries");
});

test("every lens program composes into a complete WGSL module", () => {
  for (const lens of SPARSE_CM12_LENSES) {
    for (const [name, program] of Object.entries(lens.programs)) {
      const code = stageLensProgramWGSL(lens, name);
      const label = `${lens.id}#${name}`;
      assert.equal(occurrences(code, /\bfn vertexMain\b/g), 1,
        `${label}: exactly one vertex entry point`);
      assert.equal(occurrences(code, /\bfn fragmentMain\b/g), 1,
        `${label}: exactly one fragment entry point`);
      const required = REQUIRED[program.kind];
      // Anchored on the open paren because every prelude also declares
      // `lensCellPlacement` and friends, which `fn lensCell` is a prefix of.
      assert.match(code, new RegExp(`\\bfn ${required.classifier}\\s*\\(`),
        `${label}: a ${program.kind} layer calls ${required.classifier}`);
      assert.match(code, new RegExp(`\\bfn ${required.placement}\\s*\\(`),
        `${label}: a ${program.kind} layer calls ${required.placement}`);
      // Slot zero is reserved rather than declarable: the layer templates read
      // the camera, the slice and the palette through it, so a program that
      // numbered it anything else would compile and draw off-screen.
      assert.match(code, new RegExp(
        `@group\\(0\\) @binding\\(0\\) var<uniform> ${STAGE_LENS_UNIFORM_NAME}:`
        + `${STAGE_LENS_UNIFORM_TYPE};`),
        `${label}: the framework uniform must be @binding(0)`);
      assert.equal(occurrences(code, new RegExp(`struct ${STAGE_LENS_UNIFORM_TYPE}\\b`, "g")), 1,
        `${label}: a publication re-declaring the framework block would not compile`);
    }
  }
});

/**
 * A tap is the only way an intermediate reaches the screen, and the only part
 * of a lens `tsc` cannot reach: `capture()` is typed, so a *wrong* tap name is
 * a compile error, but a tap that is simply never called is not. Its phases
 * then paint magenta on a frame where nothing is wrong.
 */
test("every declared tap is captured in its own stage's encode body", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
  for (const lens of SPARSE_CM12_LENSES) {
    // Stages that tap a lens take it from the encode context: `({ lens }) =>`.
    const sentinel = `stage("${lens.stage}", (`;
    const begin = source.indexOf(sentinel);
    assert.ok(begin >= 0, `the resident does not encode a stage named "${lens.stage}"`);
    const next = source.indexOf('\n    stage("', begin + sentinel.length);
    const block = source.slice(begin, next >= 0 ? next : source.length);
    for (const tap of Object.keys(lens.taps)) {
      assert.match(block, new RegExp(`capture\\("${tap}"\\)`),
        `${lens.id}: tap "${tap}" is declared but never captured in ${lens.stage}`);
    }
  }
});
