/**
 * The presentation's extent (2026-11-19): the renderer's dense surface spans the scene's container with tankCells
 * cells and reads container / tankCells as the cell size. The brick lattice pads the container to whole pages, so the
 * solver hands over the container's cells (setup.presentCells, info.tankCells), not the lattice's. Spanning the lattice
 * squashed reservoir-seep's water toward the floor by 256 / 288 in y: the jet drew 7 cells under its particles. CPU only.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { BrickLayout } from "./brick-abi";
import { brickSceneSetup } from "./brick-scene";
import { WGSL_VOLUME } from "./brick-present";

test("brick presentation: a padded lattice presents the container's cells", () => {
  const seep = sceneDocument(getSceneDefinition("reservoir-seep"));
  const setup = brickSceneSetup(seep, { timeStep: "sixtieth" }).setup!;
  assert.deepEqual(setup.lattice, [512, 288, 64], "the lattice pads y to whole pages");
  assert.deepEqual(setup.presentCells, [512, 256, 64]);
  assert.deepEqual(setup.presentCells, sceneLatticeDimensions(seep), "the container's cells, from the lattice's corner");
  assert.deepEqual(setup.origin_m, [-0.5 * seep.container.width_m, 0, -0.5 * seep.container.depth_m], "the lattice's corner is the container's");
});

test("brick presentation: a lattice that is the container presents itself", () => {
  const tank = sceneDocument(getSceneDefinition("bulk-rest-tank"));
  const cells = sceneLatticeDimensions(tank), setup = brickSceneSetup(tank, { timeStep: "sixtieth" }).setup!;
  // An open top keeps headroom above the container: the renderer's extent is still the container's.
  assert.deepEqual(setup.presentCells ?? setup.lattice, cells);
});

test("brick presentation: presentVolume writes no cell past the presented extent", () => {
  const L = new BrickLayout({ lattice: [32, 32, 32], f: 8, c: 1, b: 1, k: 0, particles: 1 << 10 });
  const code = WGSL_VOLUME(L, [24, 16, 24]);
  assert.match(code, /const NB_PRESENT_CELLS:vec3i=vec3i\(24,16,24\);/);
  assert.ok(code.indexOf("if(any(v>=NB_PRESENT_CELLS)){continue;}") < code.indexOf("textureStore(volume"), "the guard precedes the store");
});
