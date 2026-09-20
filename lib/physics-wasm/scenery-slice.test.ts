import "../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createHeroGardenHoseScene } from "../core/hero-garden-scene";
import { scenerySliceFraction, uniformLabSlice } from "./scenery-slice";
import { uniformLabSeed } from "./uniform-controller";
import { uniformLabQuery } from "../../advance-lab/uniform-lab-state";

test("scenery uses curved geometry, unions overlaps, and respects slice depth", () => {
  const scene = createHeroGardenHoseScene();
  scene.container = { ...scene.container, width_m: 1, height_m: 1, depth_m: 1 };
  const ball = { kind: "ellipsoid" as const, id: "ball",
    radius: { x: 0.25, y: 0.25, z: 0.25 },
    place: { units: "metres" as const, position: { x: 0, y: 0.5, z: 0 } },
    material: { colorLinear: [1, 1, 1] as const } };
  const shell = { kind: "terrain-shell" as const, id: "shell", materialModel: "porcelain" as const };
  scene.scenery = { palettes: {}, nodes: [shell, ball] };
  const single = scenerySliceFraction(scene, [16, 16, 16], 0);
  assert.equal(single[8 + 16 * 8], 1);
  assert.equal(single[4 + 16 * 4], 0, "outside sphere, inside its bounding box");
  assert.ok(single.some((value) => value > 0 && value < 1));
  scene.scenery = { palettes: {}, nodes: [shell, ball, { ...ball, id: "duplicate" }] };
  assert.deepEqual(scenerySliceFraction(scene, [16, 16, 16], 0), single);
  assert.ok(scenerySliceFraction(scene, [16, 16, 16], 0.4).every((v) => v === 0));
});

test("garden default slice contains trunk and canopy and removes occupied liquid", () => {
  const scene = createHeroGardenHoseScene({ cellSize_m: 0.025 });
  const seed = uniformLabSeed(scene);
  const [nx, ny] = seed.dimensions;
  const at = (x: number, y: number) => Math.floor((x + scene.container.width_m / 2) / seed.cellSize[0]!)
    + nx! * Math.floor(y / seed.cellSize[1]!);
  assert.ok(seed.capacity[at(0.53, 0.35)]! < 1, "trunk blocks the selected slice");
  assert.ok(seed.capacity.some((v, i) => v < 1 && Math.floor(i / nx!) > ny! * 0.5), "canopy visible");
  assert.ok(seed.volume.every((v, i) => v <= seed.capacity[i]!));
  assert.equal(uniformLabSlice(scene, 48, 100).index, 47);
  assert.equal(uniformLabSlice(scene, 48, -100).index, 0);
  assert.throws(() => uniformLabSlice(scene, 48, NaN), /finite/);
});

test("slice depth persists and invalid URLs fall back to the scene default", () => {
  const state = uniformLabQuery.read(new URLSearchParams("sliceDepth=-0.15"));
  assert.equal(state.sliceDepth_m, -0.15);
  const query = new URLSearchParams();
  uniformLabQuery.write(query, state);
  assert.equal(uniformLabQuery.read(query).sliceDepth_m, -0.15);
  assert.equal(uniformLabQuery.read(new URLSearchParams("sliceDepth=Infinity")).sliceDepth_m, undefined);
});
