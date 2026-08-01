import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { fluidBodyHandleById, fluidBodyHandles, fluidBodyBox } from "../lib/editor-fluid-body";
import {
  dragTankExtents,
  tankBox,
  tankBoxForExtents,
  tankHandleIsGrabbable,
  tankLatticeForExtents,
  tankResizeIsStructural,
  tankResizePatch,
  TANK_MINIMUM_CELLS,
} from "../lib/editor-tank";
import { getEditorTool } from "../lib/editor-tools";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("shape mode is an implemented tool with its own shortcut", () => {
  const tool = getEditorTool("shape");
  assert.equal(tool.status, "active");
  assert.equal(tool.shortcut, "s");
});

test("the tank box is the container interior, resting on the floor", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  assert.equal(box.min.y, 0);
  assert.equal(box.max.y, scene.container.height_m);
  assert.equal(box.max.x - box.min.x, scene.container.width_m);
  assert.equal(box.min.x, -box.max.x, "the container is centred on x");
  assert.equal(box.min.z, -box.max.z, "the container is centred on z");
});

test("the floor has no handle, because nothing in the schema can move it", () => {
  const box = tankBox(preset("water-box-dam-break"));
  const grabbable = fluidBodyHandles(box).filter(tankHandleIsGrabbable);
  assert.equal(grabbable.length, 26 - 9, "the nine handles on the -y face are not offered");
  assert.ok(grabbable.every((handle) => handle.sides.y !== "min"));
});

test("a horizontal drag resizes the tank about its centre, moving both walls", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const cell = scene.voxelDomain.finestCellSize_m;
  const handle = fluidBodyHandleById(box, "+00")!;
  const extents = dragTankExtents(scene, handle, { x: box.max.x + 4 * cell, y: 0, z: 0 });
  assert.ok(extents.width_m > scene.container.width_m);
  assert.equal(extents.height_m, scene.container.height_m, "an x handle must not change the height");
  assert.equal(extents.depth_m, scene.container.depth_m);
  const dragged = tankBoxForExtents(extents);
  assert.equal(dragged.min.x, -dragged.max.x, "both walls move, keeping the centre");
});

test("the ceiling handle sets the height from the floor", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const handle = fluidBodyHandleById(box, "0+0")!;
  const extents = dragTankExtents(scene, handle, { x: 0, y: scene.container.height_m * 1.5, z: 0 });
  assert.ok(extents.height_m > scene.container.height_m);
  assert.equal(extents.width_m, scene.container.width_m);
  assert.equal(tankBoxForExtents(extents).min.y, 0);
});

test("tank extents snap to whole cells, so a drag lands on an exact lattice", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const cell = scene.voxelDomain.finestCellSize_m;
  for (const target of [0.37, 1.02, 2.61]) {
    const extents = dragTankExtents(scene, fluidBodyHandleById(box, "+00")!, { x: target, y: 0, z: 0 });
    const cells = extents.width_m / cell;
    assert.ok(Math.abs(cells - Math.round(cells)) < 1e-9, `${extents.width_m} m is not a whole number of cells`);
    assert.deepEqual(tankLatticeForExtents(scene, extents)[0], Math.round(cells));
  }
});

test("a tank cannot be dragged below the lattice floor", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const cell = scene.voxelDomain.finestCellSize_m;
  const extents = dragTankExtents(scene, fluidBodyHandleById(box, "+00")!, { x: 0, y: 0, z: 0 });
  assert.ok(Math.abs(extents.width_m - TANK_MINIMUM_CELLS * cell) < 1e-9);
});

test("resizing the tank repairs what the moved walls invalidated", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  const shrunk = dragTankExtents(scene, fluidBodyHandleById(box, "+++")!,
    { x: 0.2, y: scene.container.height_m * 0.3, z: 0.2 });
  const next = { ...scene, ...tankResizePatch(scene, shrunk) };
  assert.deepEqual(validateScene(next), [], "a resized tank must still be a valid scene");
  const body = fluidBodyBox(next);
  assert.ok(body);
  assert.ok(body.max.x <= next.container.width_m / 2 + 1e-9, "the water must have been brought inside");
  assert.ok(body.max.y <= next.container.height_m + 1e-9);
});

test("a resize that leaves the lattice alone is not announced as structural", () => {
  const scene = preset("water-box-dam-break");
  const before = sceneLatticeDimensions(scene);
  const unchanged = {
    width_m: scene.container.width_m,
    height_m: scene.container.height_m,
    depth_m: scene.container.depth_m,
  };
  assert.equal(tankResizeIsStructural(scene, unchanged), false);
  const grown = { ...unchanged, width_m: unchanged.width_m + 4 * scene.voxelDomain.finestCellSize_m };
  assert.equal(tankResizeIsStructural(scene, grown), true);
  assert.deepEqual(tankLatticeForExtents(scene, grown)[0], before[0] + 4);
});

test("a shape drag previews without touching the document", () => {
  // The cost of this feature is entirely in when the document is written: the
  // water body and the tank both live in the solver's rebuild key, so a
  // per-move write asks the renderer to re-seed at pointer rate. The viewport
  // holds the dragged box in local state and spends one write on release.
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const move = viewport.slice(viewport.indexOf('if (active.action === "shape-handle") {'));
  const moveBody = move.slice(0, move.indexOf("\n    }"));
  assert.match(moveBody, /setShapePreview\(/, "the drag must update the preview");
  assert.doesNotMatch(moveBody, /patchScene|patchContainer|patchFluid|commitEdit|shapeFluidBody|resizeTank/,
    "a shape drag must not write the scene document while the pointer is down");

  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  assert.match(controller, /shapeFluidBody\(box: FluidBodyBox\)/,
    "there must be no per-move variant that a caller could reach for");
  assert.match(controller, /resizeTank\(box: FluidBodyBox\)/);
});
