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
import { installSimulationMethods } from "../lib/core/method-registry";
import {
  createBoxTankWallField,
  createOpenTankWallField,
  TANK_WALL_SIDES,
  packTankWallField,
  resampleTankWallField,
  tankWallCellIsSolid,
  tankWallOpeningFraction,
  tankWallOpeningCellCount,
  tankWallSideHasFloorOpening,
  tankWallSideHasOpening,
  withTankWallCell,
} from "../lib/core/tank-wall-field";
import {
  gpuSceneSeedKey,
  gpuSceneUniformKey,
  sceneEditRequiresReset,
  sceneStructuralKey,
} from "../lib/core/webgpu-renderer";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  adaptiveMassFluidDomainForScene,
  embedTankAtlasInFluidDomain,
  residentSupportAtlas,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import {
  packSparseCM12AcceptedTopologyTemplatesForQA,
  sparseCM12TankWallRowOpenFractions,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

installSimulationMethods({
  methods: [adaptiveMassMethod],
  interactive: [adaptiveMassMethod],
  defaultId: adaptiveMassMethod.id,
});

test("tank wall runs cut and restore individual solver faces", () => {
  const full = createBoxTankWallField({ x: 6, y: 4, z: 5 });
  const cut = withTankWallCell(full, "left", 2, 1, false);
  assert.equal(tankWallCellIsSolid(cut, "left", 2, 1), false);
  assert.equal(tankWallOpeningCellCount(cut), 1);
  const restored = withTankWallCell(cut, "left", 2, 1, true);
  assert.equal(tankWallCellIsSolid(restored, "left", 2, 1), true);
  assert.equal(tankWallOpeningCellCount(restored), 0);
});

test("the live topology envelope keeps every side face available", () => {
  const field = createOpenTankWallField({ x: 6, y: 4, z: 5 });
  assert.equal(tankWallOpeningCellCount(field), 2 * 5 * 4 + 2 * 6 * 4);
  assert.ok(TANK_WALL_SIDES.every((side) => field.faces[side].solidRuns.length === 0));
});

test("cutting a wall is a live uniform edit, not a scene reset", () => {
  const before = cloneScene(defaultScene);
  const after = cloneScene(before);
  after.container.wallField = withTankWallCell(
    after.container.wallField, "front", 1, 1, false,
  );
  assert.equal(sceneStructuralKey(after), sceneStructuralKey(before));
  assert.equal(gpuSceneSeedKey(after), gpuSceneSeedKey(before));
  assert.notEqual(gpuSceneUniformKey(after), gpuSceneUniformKey(before));
  assert.equal(sceneEditRequiresReset(before, after, "adaptive-mass"), false);
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

test("a wall drag can extend below the viewport floor and clears row zero", () => {
  const scene = cloneScene(defaultScene);
  const face = scene.container.wallField.faces.right;
  const projected = projectTankWallCellOnSide(scene, {
    origin: { x: 4, y: -2, z: 0 },
    direction: { x: -1, y: 0, z: 0 },
  }, "right");
  assert.ok(projected);
  assert.equal(projected.v, 0);
  const cut = withTankWallRectangle(
    scene.container.wallField, "right", projected.u, 5, projected.u, projected.v, false,
  );
  assert.equal(tankWallCellIsSolid(cut, "right", projected.u, 0), false);
  assert.equal(tankWallSideHasOpening(cut, "right"), true);
  assert.equal(tankWallSideHasFloorOpening(cut, "right"), true);
  assert.equal(tankWallSideHasOpening(cut, "left"), false);
  assert.equal(face.vCells, scene.container.wallField.dimensions.y);
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

test("the resident wall mask closes envelope rows and opens only the cut area", () => {
  const envelope = createOpenTankWallField({ x: 8, y: 8, z: 8 });
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [{
    key: 0,
    coordinate: [0, 0, 0],
    resolution: 4,
    density: new Float64Array(4 ** 3).fill(1),
    gamma: new Float64Array(4 ** 3).fill(1),
  }], 1, undefined, 8, envelope);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const templates = packSparseCM12AcceptedTopologyTemplatesForQA(atlas, grid);
  const authored = withTankWallCell(
    createBoxTankWallField({ x: 8, y: 8, z: 8 }), "right", 2, 2, false,
  );
  const fractions = sparseCM12TankWallRowOpenFractions(
    templates.words, templates.rowCount, atlas.dimensions, atlas.tankWallPlacement, authored,
  );
  assert.ok(fractions.some((value) => value === 0));
  assert.ok(fractions.some((value) => Math.abs(value - 0.25) < 1e-6));
  assert.ok(fractions.some((value) => value === 1));
});

test("Sparse CM12 places a floor opening inside an automatically expanded receiver world", () => {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 0.4,
    height_m: 0.4,
    depth_m: 0.4,
    fillFraction: 0.5,
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.container.wallField = createBoxTankWallField({ x: 8, y: 8, z: 8 });
  scene.container.wallField = withTankWallCell(
    scene.container.wallField, "right", 2, 0, false,
  );

  const domain = adaptiveMassFluidDomainForScene(scene, 8, 0);
  assert.deepEqual(domain.dimensions, [16, 8, 8]);
  assert.deepEqual(domain.tankMinimumFine, [0, 0, 0]);
  assert.deepEqual(domain.tankMaximumFine, [8, 8, 8]);

  const tank = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: domain.tankDimensions,
    brickFineResolution: 8,
    maximumMacroSpanBricks: 1,
  });
  const embedded = embedTankAtlasInFluidDomain(tank, domain);
  const supported = residentSupportAtlas(embedded, "adaptive");
  assert.ok(supported.bricks.some((brick) => brick.coordinate[0] === 1),
    "the opened side must own a real exterior receiver page");

  const grid = buildSparseAtlasCompositeGrid(supported);
  const rightOpeningRows = grid.gradientRows.filter((row) =>
    row.axis === 0 && Math.abs(row.centerFine[0] - 8) < 1e-9
      && row.centerFine[2] >= domain.tankMinimumFine[2]
      && row.centerFine[2] < domain.tankMaximumFine[2]);
  assert.equal(rightOpeningRows.length, 1);
  assert.equal(rightOpeningRows[0]!.centerFine[1], 0.5);
  assert.equal(rightOpeningRows[0]!.centerFine[2], 2.5);
  assert.equal(rightOpeningRows[0]!.terms.length, 2,
    "the opening must connect tank liquid to an exterior cell, not sparse air");
  assert.equal(grid.gradientRows.some((row) =>
    row.axis === 0 && Math.abs(row.centerFine[0]) < 1e-9
      && row.centerFine[2] >= domain.tankMinimumFine[2]
      && row.centerFine[2] < domain.tankMaximumFine[2]), false,
  "the opposite closed wall must remain impermeable inside the larger world");
});
