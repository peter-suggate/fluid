import assert from "node:assert/strict";
import test from "node:test";

import {
  createTallCellsHillsideDamBreakScene,
  TALL_CELLS_FLOOD_CELL_SIZE_M,
  TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M,
  TALL_CELLS_FLOOD_GRID,
  TALL_CELLS_FLOOD_RESERVOIR_M,
  TALL_CELLS_FLOOD_RUNOUT_M,
  TALL_CELLS_FLOOD_UPHILL_HEIGHT_M,
} from "../lib/core/scenes";
import { sampleSolidWorld, solidWorldForScene,
  SOLID_WORLD_TERRAIN_MATERIAL_ID } from "../lib/core/solid-world";
import { terrainHeightAt } from "../lib/core/terrain";
import "../lib/methods";
import {
  STUDIO_STAGE_DRY_SCENE_LIGHTING,
  svoSceneLighting,
} from "../lib/svo/features/lighting-visibility/svo-dry-scene-lighting";

const [NX, NY, NZ] = TALL_CELLS_FLOOD_GRID;
const WIDTH_M = NX * TALL_CELLS_FLOOD_CELL_SIZE_M;
const HEIGHT_M = NY * TALL_CELLS_FLOOD_CELL_SIZE_M;
const DEPTH_M = NZ * TALL_CELLS_FLOOD_CELL_SIZE_M;

/**
 * The shape of the course, not the expression that produced it.
 *
 * The exponent, the run and the bank height are a look and can be re-authored;
 * what the sim depends on is that the reservoir stands on ground that is level
 * to the bit, that the crest is steep enough to throw the front rather than
 * drain it, that the toe meets the runout tangentially instead of as a step,
 * and that everything stays under the domain lid. Each assertion below is one
 * of those, so re-grading the hillside does not become a test edit but flatten-
 * ing it, kinking it, or overrunning the container does.
 */
test("height samples author a flat shelf, an easing steep drop, channel banks, and a runout", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  const terrain = scene.terrain!;
  const minimumX = -0.5 * WIDTH_M;
  const slopeStart = minimumX + TALL_CELLS_FLOOD_RESERVOIR_M.x;
  const slopeEnd = 0.5 * WIDTH_M - TALL_CELLS_FLOOD_RUNOUT_M;
  const drop = TALL_CELLS_FLOOD_UPHILL_HEIGHT_M - TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M;
  const meanGrade = drop / (slopeEnd - slopeStart);
  const gradeAt = (x: number) => (terrainHeightAt(terrain, x, 0)
    - terrainHeightAt(terrain, x + TALL_CELLS_FLOOD_CELL_SIZE_M, 0))
    / TALL_CELLS_FLOOD_CELL_SIZE_M;
  assert.deepEqual(terrain.grid?.size, { nx: NX + 1, nz: NZ + 1 });
  assert.equal(terrain.grid?.spacing_m, TALL_CELLS_FLOOD_CELL_SIZE_M);

  // Sampled by cell index rather than by accumulating the spacing: a running
  // `x += spacing` drifts a whole cell over 256 of them and would silently stop
  // testing the crest sample, which is the one that has to be exact.
  const sampleX = (cell: number) => minimumX + cell * TALL_CELLS_FLOOD_CELL_SIZE_M;
  const crestCell = Math.round(TALL_CELLS_FLOOD_RESERVOIR_M.x
    / TALL_CELLS_FLOOD_CELL_SIZE_M);
  assert.equal(sampleX(crestCell), slopeStart);

  // The launch shelf is where the block is placed, so it is asserted over the
  // whole reservoir footprint rather than at one sample: a shelf that is level
  // at its ends and sagging in the middle would seat the dam on air.
  for (let cell = 0; cell <= crestCell; cell += 1) {
    for (const z of [-0.5 * DEPTH_M, -0.4, 0, 0.4, 0.5 * DEPTH_M]) {
      assert.equal(terrainHeightAt(terrain, sampleX(cell), z),
        TALL_CELLS_FLOOD_UPHILL_HEIGHT_M,
        `shelf sample (${sampleX(cell)}, ${z}) is not level with the rest of it`);
    }
  }

  // Steep at the crest, gentle at the toe, and never uphill in between.
  assert.ok(gradeAt(slopeStart) > 1,
    `the crest must fall past 45 degrees: got ${gradeAt(slopeStart)}`);
  assert.ok(gradeAt(slopeStart) > 1.8 * meanGrade,
    "a crest no steeper than the mean grade is a straight ramp again");
  assert.ok(gradeAt(slopeEnd - TALL_CELLS_FLOOD_CELL_SIZE_M)
    < 0.1 * gradeAt(slopeStart),
  "the toe must arrive tangent to the runout, not as a step into it");
  let previous = Number.POSITIVE_INFINITY;
  for (let cell = 0; cell <= NX; cell += 1) {
    const height = terrainHeightAt(terrain, sampleX(cell), 0);
    assert.ok(height <= previous + 1e-12,
      `the course climbs again at x = ${sampleX(cell)}`);
    previous = height;
  }

  // Both ends land exactly on their constant, and the runout stays there.
  assert.equal(terrainHeightAt(terrain, slopeEnd, 0),
    TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M);
  assert.equal(terrainHeightAt(terrain, 0.5 * WIDTH_M, 0),
    TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M);
  assert.equal(terrainHeightAt(terrain, 0.5 * (slopeEnd + 0.5 * WIDTH_M), 0),
    TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M);

  // The bank is a crossfall on the hill alone: flat where the water starts and
  // flat where it ends, or the course becomes a flume.
  const midSlope = 0.5 * (slopeStart + slopeEnd);
  assert.ok(terrainHeightAt(terrain, midSlope, 0.5 * DEPTH_M)
    > terrainHeightAt(terrain, midSlope, 0.25 * DEPTH_M));
  assert.ok(terrainHeightAt(terrain, midSlope, 0.25 * DEPTH_M)
    > terrainHeightAt(terrain, midSlope, 0));
  assert.equal(terrainHeightAt(terrain, 0.5 * WIDTH_M, 0.5 * DEPTH_M),
    TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M);

  // The lid has to clear the terrain *and* the reservoir standing on it, or the
  // document is one that `validateScene` rejects rather than one that renders.
  assert.ok(Math.max(...terrain.grid!.heights_m) <= HEIGHT_M);
  assert.ok(TALL_CELLS_FLOOD_UPHILL_HEIGHT_M + TALL_CELLS_FLOOD_RESERVOIR_M.y
    < HEIGHT_M, "the released block must have air above it");
});

test("the scene authors its boundary and terrain in SolidWorld", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  const solids = solidWorldForScene(scene);
  assert.equal(sampleSolidWorld(solids, [-1, NY - 1, Math.floor(NZ / 2)]).solidFraction,
    1, "the hidden vessel still authors its editable low-X wall in SolidWorld");
  assert.equal(sampleSolidWorld(solids, [Math.floor(NX / 2), NY - 1, -1]).solidFraction,
    1, "the hidden vessel still authors its editable low-Z wall in SolidWorld");
  assert.equal(sampleSolidWorld(solids, [Math.floor(NX / 2), NY, Math.floor(NZ / 2)])
    .solidFraction, 1, "the Flood tank has the catalog's closed ceiling");
  assert.equal(sampleSolidWorld(solids, [NX - 1, 0, Math.floor(NZ / 2)]).materialId,
    SOLID_WORLD_TERRAIN_MATERIAL_ID);
  assert.equal(sampleSolidWorld(solids, [NX - 1, NY - 1, Math.floor(NZ / 2)]).solidFraction,
    0, "SolidWorld claims terrain voxels rather than the logical world volume");

});

/**
 * What the rig has to be, rather than what it currently is.
 *
 * The numbers themselves are a look and were solved by measuring the frame, so
 * pinning them here would only make re-grading the scene a test edit. What must
 * not silently come back is the *shape* the stage base has and a landscape does
 * not: a key held down to a fill because a practical was doing the lighting, and
 * a hemisphere that is black in every direction.
 */
test("the hillside overrides every term of the stage's dark-room rig", () => {
  const scene = createTallCellsHillsideDamBreakScene();
  const lighting = svoSceneLighting({ ...scene, environment: "stage" });
  const stage = STUDIO_STAGE_DRY_SCENE_LIGHTING;
  const key = lighting.directional!;
  const environment = lighting.environment!;

  assert.ok(key.intensity! >= 4 * stage.directional.intensity,
    `a sun, not the stage's ${stage.directional.intensity} fill: got ${key.intensity}`);
  assert.ok(key.direction![1] > 0.5 * Math.hypot(...key.direction!),
    "the sun must come from above, so the slope has a lit face and a shaded one");
  assert.ok(key.direction![0] < 0, "and from the side, so the slope is modelled rather than flat");

  const luminance = (color: readonly number[]) =>
    0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
  const sky = luminance(environment.upperRadianceLinear!);
  const ground = luminance(environment.lowerRadianceLinear!);
  assert.ok(sky >= 10 * luminance(stage.environment.upperRadianceLinear),
    `the backdrop is a sky: got ${sky}`);
  assert.ok(sky > 3 * ground, "and it has a gradient in it: the sky must outrun the ground");
  assert.ok(environment.upperRadianceLinear![2] > environment.upperRadianceLinear![0],
    "cool above");
  assert.ok(environment.lowerRadianceLinear![0] > environment.lowerRadianceLinear![2],
    "warm below");

  assert.equal(lighting.grade?.toneCurve, "aces", "the stage's tone curve is the one field kept");
  const balance = lighting.grade!.whiteBalance!;
  assert.ok(balance[0] > balance[2], "and the frame is graded warm, which is what makes it read as dusk");
});
