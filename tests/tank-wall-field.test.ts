import assert from "node:assert/strict";
import test from "node:test";
import {
  pickTankWallCell,
  projectTankWallCellOnSide,
  tankWallRectangleCorners,
  withTankWallRectangle,
} from "../lib/core/editor-tank-wall";
import { cloneScene, defaultScene, validateScene } from "../lib/core/model";
import { entityActionsAt } from "../lib/core/editor-entity-catalog";
import { TANK_SELECTION_ID } from "../lib/core/editor-tank";
import {
  createBoxTankWallField,
  packTankWallField,
  resampleTankWallField,
  tankWallCellIsSolid,
  tankWallOpeningFraction,
  tankWallOpeningCellCount,
  withTankWallCell,
} from "../lib/core/tank-wall-field";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";

test("tank wall runs cut and restore individual solver faces", () => {
  const full = createBoxTankWallField({ x: 6, y: 4, z: 5 });
  const cut = withTankWallCell(full, "left", 2, 1, false);
  assert.equal(tankWallCellIsSolid(cut, "left", 2, 1), false);
  assert.equal(tankWallOpeningCellCount(cut), 1);
  const restored = withTankWallCell(cut, "left", 2, 1, true);
  assert.equal(tankWallCellIsSolid(restored, "left", 2, 1), true);
  assert.equal(tankWallOpeningCellCount(restored), 0);
});

test("resizing resamples openings instead of silently healing the wall", () => {
  let field = createBoxTankWallField({ x: 6, y: 4, z: 5 });
  field = withTankWallCell(field, "right", 2, 1, false);
  const resized = resampleTankWallField(field, { x: 12, y: 8, z: 10 });
  assert.ok(tankWallOpeningCellCount(resized) >= 4);
  assert.equal(tankWallCellIsSolid(resized, "right", 4, 2), false);
});

test("the GPU wall bitset preserves the four face atlases", () => {
  let field = createBoxTankWallField({ x: 6, y: 4, z: 5 });
  field = withTankWallCell(field, "front", 3, 2, false);
  const packed = packTankWallField(field);
  assert.equal(packed[1], 6);
  assert.equal(packed[2], 4);
  assert.equal(packed[3], 5);
  const cell = 3 + 6 * 2;
  const offset = packed[6]!;
  assert.equal((packed[offset + (cell >>> 5)]! >>> (cell & 31)) & 1, 0);
});

test("wall picking addresses the visible side atlas, including holes", () => {
  const scene = cloneScene(defaultScene);
  const left = pickTankWallCell(scene, {
    origin: { x: -2, y: scene.container.height_m * 0.5, z: 0 },
    direction: { x: 1, y: 0, z: 0 },
  });
  assert.ok(left);
  assert.equal(left.side, "left");
  const opened = withTankWallCell(scene.container.wallField, left.side, left.u, left.v, false);
  scene.container.wallField = opened;
  assert.deepEqual(pickTankWallCell(scene, {
    origin: { x: -2, y: scene.container.height_m * 0.5, z: 0 },
    direction: { x: 1, y: 0, z: 0 },
  })?.key, left.key);
});

test("the editor removes one inclusive rectangle and schema 2 requires it", () => {
  const scene = cloneScene(defaultScene);
  scene.container.wallField = withTankWallRectangle(
    scene.container.wallField, "back", 8, 6, 10, 7, false,
  );
  assert.equal(tankWallOpeningCellCount(scene.container.wallField), 6);
  assert.equal(tankWallCellIsSolid(scene.container.wallField, "back", 8, 6), false);
  assert.equal(tankWallCellIsSolid(scene.container.wallField, "back", 10, 7), false);
  assert.equal(tankWallCellIsSolid(scene.container.wallField, "back", 11, 7), true);
  assert.deepEqual(validateScene(scene), []);
  const invalid: unknown = cloneScene(scene);
  delete (invalid as { container: { wallField?: unknown } }).container.wallField;
  assert.ok(validateScene(invalid as typeof scene).some((error) => error.includes("wall field")));
});

test("a reversed wall drag is clamped and stays on its starting face", () => {
  const scene = cloneScene(defaultScene);
  const face = scene.container.wallField.faces.right;
  const dragged = withTankWallRectangle(
    scene.container.wallField, "right", face.uCells + 3, face.vCells + 2, 2, 1, false,
  );
  assert.equal(tankWallOpeningCellCount(dragged), (face.uCells - 2) * (face.vCells - 1));
  assert.equal(tankWallCellIsSolid(dragged, "left", 2, 1), true);

  const projected = projectTankWallCellOnSide(scene, {
    origin: { x: 4, y: scene.container.height_m * 2, z: scene.container.depth_m * 2 },
    direction: { x: -1, y: -0.1, z: -0.1 },
  }, "right");
  assert.ok(projected);
  assert.equal(projected.side, "right");
  assert.equal(projected.u, face.uCells - 1);
  assert.equal(projected.v, face.vCells - 1);
});

test("wall rectangle preview corners follow exact cell boundaries", () => {
  const scene = cloneScene(defaultScene);
  const face = scene.container.wallField.faces.front;
  const [bottomLeft, bottomRight, topRight, topLeft] = tankWallRectangleCorners(scene, {
    side: "front", u0: 2, v0: 3, u1: 4, v1: 5,
  });
  assert.equal(bottomLeft.z, -scene.container.depth_m / 2);
  assert.ok(Math.abs(bottomRight.x - bottomLeft.x - 3 * scene.container.width_m / face.uCells) < 1e-12);
  assert.ok(Math.abs(topLeft.y - bottomLeft.y - 3 * scene.container.height_m / face.vCells) < 1e-12);
  assert.equal(topRight.z, bottomLeft.z);
});

test("the tank radial menu exposes the rectangle cut as a direct action", () => {
  const scene = cloneScene(defaultScene);
  const actions = entityActionsAt({ scene, bodies: [] }, {
    selection: { kind: "tank", id: TANK_SELECTION_ID },
    point_m: { x: scene.container.width_m / 2, y: scene.container.height_m / 2, z: 0 },
  });
  const cut = actions.find((action) => action.id === "cut-wall-opening");
  assert.ok(cut);
  assert.equal(cut.label, "Cut opening");
  assert.equal(cut.children, undefined);
  assert.deepEqual(cut.effect, { kind: "arm", tool: "tank-wall-cut" });
});

test("opening coverage is area weighted across adaptive faces", () => {
  let field = createBoxTankWallField({ x: 8, y: 8, z: 8 });
  field = withTankWallCell(field, "right", 2, 2, false);
  assert.equal(tankWallOpeningFraction(field, "right", 2, 4, 2, 4), 0.25);
  assert.equal(tankWallOpeningFraction(field, "right", 0, 2, 0, 2), 0);
});

test("adaptive topology exposes only authored outer-wall openings to air", () => {
  let wallField = createBoxTankWallField({ x: 8, y: 8, z: 8 });
  wallField = withTankWallCell(wallField, "right", 2, 2, false);
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [{
    key: 0,
    coordinate: [0, 0, 0],
    resolution: 4,
    density: new Float64Array(4 ** 3).fill(1),
    gamma: new Float64Array(4 ** 3).fill(1),
  }], 1, undefined, 8, wallField);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const outerRows = grid.gradientRows.filter((row) =>
    row.kind === "sparse-air" && (row.centerFine[row.axis] === 0
      || row.centerFine[row.axis] === atlas.dimensions[row.axis]));
  assert.equal(outerRows.length, 1);
  assert.equal(outerRows[0]!.axis, 0);
  assert.equal(outerRows[0]!.centerFine[0], 8);
  assert.equal(outerRows[0]!.geometricArea, 1);
  assert.equal(outerRows[0]!.openFraction, 0.25);
  assert.equal(outerRows[0]!.terms[0]!.coefficient, -0.5);
});
