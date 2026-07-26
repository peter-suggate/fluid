import assert from "node:assert/strict";
import test from "node:test";
import {
  editorFluidLattice,
  eraseFluidBrick,
  fillFractionForHeight,
  fillLevelHandlePosition,
  fillLevelHeight_m,
  fluidBrickCenter,
  fluidBrickIndexAt,
  fluidBrickKey,
  fluidPaintPatch,
  paintFluidBrick,
} from "../lib/editor-fluid";
import { initialFluidBrickContainsCell, INITIAL_FLUID_BRICK_SIZE } from "../lib/initial-fluid";
import { cloneScene, defaultScene, validateScene, type SceneDescription } from "../lib/model";

function scene(): SceneDescription {
  const next = cloneScene(defaultScene);
  next.fluid.initialCondition = "tank-fill";
  delete next.fluid.initialBrickSeeds_m;
  delete next.fluid.initialBrickSeedsAdditive;
  return next;
}

function painted(base: SceneDescription, points: readonly { x: number; y: number; z: number }[]): SceneDescription {
  let current = base;
  for (const point of points) {
    const result = paintFluidBrick(current, point);
    if (result) current = { ...current, fluid: fluidPaintPatch(current, result) };
  }
  return current;
}

test("the editor lattice mirrors the solver's container rounding", () => {
  const base = scene();
  const lattice = editorFluidLattice(base);
  const cell = base.voxelDomain.finestCellSize_m;
  assert.deepEqual([...lattice.dimensions], [
    Math.max(8, Math.round(base.container.width_m / cell)),
    Math.max(8, Math.round(base.container.height_m / cell)),
    Math.max(8, Math.round(base.container.depth_m / cell)),
  ]);
  assert.equal(lattice.bricks.x, Math.ceil(lattice.dimensions[0] / INITIAL_FLUID_BRICK_SIZE));
  assert.ok(lattice.brickSize_m.x > 0 && lattice.brickSize_m.y > 0 && lattice.brickSize_m.z > 0);
});

test("brick centres round-trip to their own index and reject points outside the tank", () => {
  const base = scene();
  const lattice = editorFluidLattice(base);
  for (const index of [{ x: 0, y: 0, z: 0 }, { x: 1, y: 2, z: 1 }, { x: lattice.bricks.x - 1, y: lattice.bricks.y - 1, z: lattice.bricks.z - 1 }]) {
    assert.deepEqual(fluidBrickIndexAt(lattice, fluidBrickCenter(lattice, index)), index, fluidBrickKey(index));
  }
  assert.equal(fluidBrickIndexAt(lattice, { x: 0, y: -1, z: 0 }), undefined);
  assert.equal(fluidBrickIndexAt(lattice, { x: 500, y: 0.1, z: 0 }), undefined);
});

test("painting is idempotent per brick and always additive over the base fill", () => {
  const base = scene();
  const lattice = editorFluidLattice(base);
  const centre = fluidBrickCenter(lattice, { x: 1, y: 1, z: 1 });

  const first = paintFluidBrick(base, centre);
  assert.ok(first);
  const once = { ...base, fluid: fluidPaintPatch(base, first) };
  assert.equal(once.fluid.initialBrickSeeds_m?.length, 1);
  assert.equal(once.fluid.initialBrickSeedsAdditive, true, "painted water adds to the authored fill");

  // Anywhere else inside the same brick is the same brick.
  const offset = { x: centre.x + lattice.brickSize_m.x * 0.2, y: centre.y, z: centre.z - lattice.brickSize_m.z * 0.2 };
  assert.equal(paintFluidBrick(once, offset), undefined, "a stroke inside one brick does not churn the document");
  assert.deepEqual(validateScene(once), []);
});

test("erasing removes a brick and drops the field rather than leaving it empty", () => {
  const base = scene();
  const lattice = editorFluidLattice(base);
  const a = fluidBrickCenter(lattice, { x: 1, y: 1, z: 1 });
  const b = fluidBrickCenter(lattice, { x: 2, y: 1, z: 1 });
  const two = painted(base, [a, b]);
  assert.equal(two.fluid.initialBrickSeeds_m?.length, 2);

  const removedOne = eraseFluidBrick(two, a);
  assert.ok(removedOne);
  assert.equal(removedOne.empty, false);
  const one = { ...two, fluid: fluidPaintPatch(two, removedOne) };
  assert.equal(one.fluid.initialBrickSeeds_m?.length, 1);

  const removedLast = eraseFluidBrick(one, b);
  assert.ok(removedLast);
  assert.equal(removedLast.empty, true);
  const none = { ...one, fluid: fluidPaintPatch(one, removedLast) };
  // validateScene rejects an empty seed array, so the field must be absent.
  assert.equal("initialBrickSeeds_m" in none.fluid, false);
  assert.deepEqual(validateScene(none), []);

  assert.equal(eraseFluidBrick(none, a), undefined, "erasing empty air changes nothing");
});

test("painted seeds wet the brick the solver resolves them to", () => {
  const base = scene();
  const lattice = editorFluidLattice(base);
  const index = { x: 1, y: 1, z: 1 };
  const target = painted(base, [fluidBrickCenter(lattice, index)]);
  const brick = INITIAL_FLUID_BRICK_SIZE;
  // initialFluidBrickContainsCell is the solver-side reader of the same seeds.
  const inside = initialFluidBrickContainsCell(
    target, index.x * brick + 1, index.y * brick + 1, index.z * brick + 1, lattice.dimensions,
  );
  assert.equal(inside, true, "the painted brick reads as wet");
  assert.equal(
    initialFluidBrickContainsCell(target, 1, 1, 1, lattice.dimensions), false,
    "an unpainted brick stays dry",
  );
});

test("the fill handle rides the container corner and maps height to fraction", () => {
  const base = scene();
  base.container.fillFraction = 0.4;
  assert.ok(Math.abs(fillLevelHeight_m(base) - 0.4 * base.container.height_m) < 1e-12);
  const handle = fillLevelHandlePosition(base);
  assert.equal(handle.x, -0.5 * base.container.width_m);
  assert.equal(handle.z, -0.5 * base.container.depth_m);
  assert.ok(Math.abs(handle.y - fillLevelHeight_m(base)) < 1e-12);

  assert.ok(Math.abs(fillFractionForHeight(base, 0.5 * base.container.height_m) - 0.5) < 1e-12);
  assert.equal(fillFractionForHeight(base, -5), 0, "the surface cannot go below the floor");
  assert.equal(fillFractionForHeight(base, 500), 1, "nor above the rim");
});
