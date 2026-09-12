import assert from "node:assert/strict";
import test from "node:test";

import { baseInitialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { fluidSolidWorldForScene } from "../lib/core/solid-world";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickContainingCoordinate,
  sparseBrickSpan,
  sparseCM12InitialActiveBrickKeys,
} from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-volume/method";

const sorted = (values: Iterable<number>): number[] => [...values].sort((a, b) => a - b);

test("long dam generation zero contains exactly the authored material leaves", () => {
  const scene = createSparseCM12LongDamBreakScene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  const options = adaptiveMassSolverOptions(values);
  const dimensions = sceneLatticeDimensions(scene) as [number, number, number];
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    brickFineResolution: options.brickFineResolution,
    solidWorld: fluidSolidWorldForScene(scene),
    maximumMacroSpanBricks: options.maximumMacroSpanBricks,
    surfaceFineRings: options.surfaceFineRings,
    coarseFirstCurvatureTolerance: options.activityPolicy?.coarseFirst
      ? options.activityPolicy.curvatureTolerance : undefined,
    initialSurfaceCoarseningBiasRings:
      options.activityPolicy?.activitySignals ? 1 : 0,
  });

  const sourceMaterialKeys = new Set(atlas.bricks
    .filter((brick) => brick.density.some((density) => density > 0))
    .map((brick) => brick.key));

  // Derive the occupied logical pages from the authored finest-cell geometry,
  // then resolve each page through the atlas independently of the active-key
  // selector. The 80 wet B8 pages are represented by 64 ordinary leaves and
  // two span-2 macro leaves, hence 66 physical leaves.
  const authoredWetPages = new Set<string>();
  const authoredOwnerKeys = new Set<number>();
  const B = atlas.brickFineResolution;
  let authoredWetCells = 0;
  for (let z = 0; z < dimensions[2]; z += 1) {
    for (let y = 0; y < dimensions[1]; y += 1) {
      for (let x = 0; x < dimensions[0]; x += 1) {
        if (baseInitialLiquidFractionAtCell(scene, x, y, z, dimensions) <= 0) continue;
        authoredWetCells += 1;
        const page = [Math.floor(x / B), Math.floor(y / B), Math.floor(z / B)] as const;
        authoredWetPages.add(page.join(","));
        const owner = sparseBrickContainingCoordinate(atlas, page);
        assert.ok(owner, `authored wet page ${page.join(",")} must have a source owner`);
        authoredOwnerKeys.add(owner.key);
      }
    }
  }

  assert.deepEqual(dimensions, [192, 96, 32]);
  assert.equal(authoredWetCells, 40_960);
  assert.equal(authoredWetPages.size, 80);
  const pageCoordinates = [...authoredWetPages].map((key) => key.split(",").map(Number));
  assert.deepEqual(pageCoordinates.reduce((minimum, coordinate) => minimum.map(
    (value, axis) => Math.min(value, coordinate[axis]!),
  ), [Infinity, Infinity, Infinity]), [0, 0, 0]);
  assert.deepEqual(pageCoordinates.reduce((maximum, coordinate) => maximum.map(
    (value, axis) => Math.max(value, coordinate[axis]!),
  ), [-Infinity, -Infinity, -Infinity]), [3, 4, 3]);

  assert.equal(authoredOwnerKeys.size, 66);
  assert.deepEqual(sorted(authoredOwnerKeys), sorted(sourceMaterialKeys),
    "source density must identify exactly every authored wet-page owner");
  const materialLeaves = [...sourceMaterialKeys].map((key) => atlas.directory.get(key)!);
  assert.equal(materialLeaves.filter((brick) => sparseBrickSpan(brick) === 1).length, 64);
  assert.deepEqual(materialLeaves
    .filter((brick) => sparseBrickSpan(brick) === 2)
    .map((brick) => brick.coordinate), [[0, 0, 0], [0, 0, 2]]);

  const activeKeys = sparseCM12InitialActiveBrickKeys(scene, atlas, 2);
  assert.equal(activeKeys.size, 66);
  assert.deepEqual(sorted(activeKeys), sorted(sourceMaterialKeys),
    "generation zero must contain no missing material leaf or extra dry leaf");
});
