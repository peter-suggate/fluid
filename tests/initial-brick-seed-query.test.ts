import assert from "node:assert/strict";
import test from "node:test";
import { editorFluidLattice, fluidBrickCenter, fluidBrickIndexAt } from "../lib/editor-fluid";
import { sceneSeedsQuery, withSceneSeedsFromQuery } from "../lib/initial-brick-seed-query";
import { cloneScene, validateScene, type SceneDescription, type Vec3 } from "../lib/model";
import { getScenePreset } from "../lib/scenes";

/** The occupancy a document actually means: which bricks its seeds wet. */
function wetBricks(scene: SceneDescription): string[] {
  const lattice = editorFluidLattice(scene);
  const keys = new Set<string>();
  for (const seed of scene.fluid.initialBrickSeeds_m ?? []) {
    const index = fluidBrickIndexAt(lattice, seed);
    if (index) keys.add(`${index.x}:${index.y}:${index.z}`);
  }
  return [...keys].sort();
}

/** A scene at the lattice the studio paints on, with `bricks` painted into it. */
function painted(cellSize_m: number, bricks: readonly (readonly [number, number, number])[]) {
  const scene = cloneScene(getScenePreset("twin-dam-collision").create());
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: cellSize_m };
  const lattice = editorFluidLattice(scene);
  const seeds: Vec3[] = bricks.map(([x, y, z]) => fluidBrickCenter(lattice, { x, y, z }));
  scene.fluid = { ...scene.fluid, initialBrickSeeds_m: seeds, initialBrickSeedsAdditive: true };
  return scene;
}

/** The two 8 x 4 x 4 brick dams from the twin-dam link that motivated the key. */
function twinDamBricks() {
  const bricks: [number, number, number][] = [];
  for (const originX of [0, 20]) {
    for (let z = originX === 0 ? 0 : 4; z < (originX === 0 ? 4 : 8); z += 1) {
      for (let y = 0; y < 4; y += 1) {
        for (let x = originX; x < originX + 8; x += 1) bricks.push([x, y, z]);
      }
    }
  }
  return bricks;
}

test("a painted document round-trips through the seeds key as the same wet bricks", () => {
  const scene = painted(0.0125, twinDamBricks());
  // Onto the preset's own unpainted document at the same lattice, which is what
  // hydration does: the link carries the paint, not the scene it landed on.
  const target = cloneScene(getScenePreset("twin-dam-collision").create());
  target.voxelDomain = { ...target.voxelDomain, finestCellSize_m: 0.0125 };
  const restored = withSceneSeedsFromQuery(target, sceneSeedsQuery(scene));

  assert.deepEqual(wetBricks(restored), wetBricks(scene));
  assert.equal(restored.fluid.initialBrickSeeds_m?.length, 256);
  assert.deepEqual(validateScene(restored), []);
});

test("the seeds key is orders of magnitude shorter than the seed array it replaces", () => {
  const scene = painted(0.0125, twinDamBricks());
  const encoded = sceneSeedsQuery(scene);
  const asArray = new URLSearchParams([
    ["scene.fluid.initialBrickSeeds_m", JSON.stringify(scene.fluid.initialBrickSeeds_m)],
  ]).toString();

  assert.ok(encoded.length < 250, `expected a compact key, got ${encoded.length} chars`);
  // The array form is what reloaded as an HTTP 431 once the request line passed
  // Node's 16 kB header ceiling; the whole point of the key is that it cannot.
  assert.ok(asArray.length > 40 * encoded.length,
    `expected a large saving, got ${asArray.length} vs ${encoded.length}`);
});

test("dense paint encodes as a bitset and scattered paint as gaps, and both round-trip", () => {
  const dense = painted(0.0125, twinDamBricks());
  const scattered = painted(0.0125, [[0, 0, 0], [13, 3, 7], [27, 1, 4]]);

  assert.equal(sceneSeedsQuery(dense).split("*")[1]?.[0], "b");
  assert.equal(sceneSeedsQuery(scattered).split("*")[1]?.[0], "d");

  for (const scene of [dense, scattered]) {
    const target = painted(0.0125, [[0, 0, 0]]);
    const restored = withSceneSeedsFromQuery(target, sceneSeedsQuery(scene));
    assert.deepEqual(wetBricks(restored), wetBricks(scene));
  }
});

test("an emptied paint drops the field rather than authoring the array validateScene rejects", () => {
  const scene = painted(0.0125, twinDamBricks());
  const cleared = withSceneSeedsFromQuery(scene, "");

  assert.equal(cleared.fluid.initialBrickSeeds_m, undefined);
  assert.deepEqual(validateScene(cleared), []);
  assert.equal(sceneSeedsQuery(cleared), "", "a document with no paint has no key to write");
});

test("a link opened at a finer lattice puts the water back in the same place", () => {
  const coarse = painted(0.05, [[0, 0, 0], [6, 1, 1]]);
  const fine = painted(0.0125, [[0, 0, 0]]);
  const restored = withSceneSeedsFromQuery(fine, sceneSeedsQuery(coarse));

  // Bricks are four times smaller at this lattice, so the same instruction is a
  // different index — but the same physical corner of the tank.
  const near = restored.fluid.initialBrickSeeds_m ?? [];
  assert.equal(near.length, 2);
  for (const [index, source] of (coarse.fluid.initialBrickSeeds_m ?? []).entries()) {
    const brick = editorFluidLattice(coarse).brickSize_m;
    assert.ok(Math.abs(near[index]!.x - source.x) <= brick.x,
      `x drifted more than one coarse brick: ${near[index]!.x} vs ${source.x}`);
    assert.ok(Math.abs(near[index]!.z - source.z) <= brick.z,
      `z drifted more than one coarse brick: ${near[index]!.z} vs ${source.z}`);
  }
  assert.deepEqual(validateScene(restored), []);
});

test("a malformed seeds value leaves the document's own water untouched", () => {
  const scene = painted(0.0125, twinDamBricks());
  for (const raw of ["nonsense", "28_8_8", "28_8_8*b", "28_8_8*q_1_2", "0_0_0*d_1",
    "28_8_8*b_0_0_0_2_2_2_!!!!", "28_8_8*d_1*d_2"]) {
    assert.equal(withSceneSeedsFromQuery(scene, raw), scene, `"${raw}" must be refused whole`);
  }
});

test("paint too large for any link is dropped rather than written past the budget", () => {
  const solid = (bricks: readonly [number, number, number]) => [...Array(bricks[0]).keys()].flatMap((x) =>
    [...Array(bricks[1]).keys()].flatMap((y) => [...Array(bricks[2]).keys()].map((z) => [x, y, z] as const)));

  // A tank painted solid at the studio's own lattice is still a link, which is
  // the case the budget must not catch.
  const whole = sceneSeedsQuery(painted(0.0125, solid([28, 8, 8])));
  assert.ok(whole.length < 700, `a fully painted tank must stay shareable, got ${whole.length} chars`);

  // Fourteen thousand bricks scattered through a 700 x 200 x 200 grid defeat
  // both encodings at once — too many for gaps, too sparse for a bitset — which
  // is the shape the budget exists for.
  const vast = painted(0.00625, solid([56, 16, 16]));
  vast.voxelDomain = { ...vast.voxelDomain, finestCellSize_m: 0.0005 };
  assert.equal(sceneSeedsQuery(vast), "",
    "authoring past the budget belongs to the scene library, not the address bar");
});
