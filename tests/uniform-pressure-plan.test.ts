import assert from "node:assert/strict";
import test from "node:test";
import { planUniformCM11aHierarchy, planUniformCM11aWindow } from "../lib/methods/uniform/pressure-plan";

test("odd terminal grids are supported without changing the hierarchy or scene dimensions", () => {
  const plan = planUniformCM11aHierarchy([72, 48, 48]);
  assert.equal(plan.rejection, undefined);
  assert.deepEqual(plan.levelDimensions, [[72,48,48],[36,24,24],[18,12,12],[9,6,6],[9,3,3]]);
  assert.equal(plan.coarsestCells, 275);
  for (const dimensions of [[288,192,192], [19,7,5], [79,3,3]] as const) {
    const result = planUniformCM11aHierarchy(dimensions);
    assert.equal(result.rejection, undefined);
    assert.deepEqual(result.levelDimensions[0], dimensions);
  }
  const window = planUniformCM11aWindow([72,48,48], [0,0,0], [72,48,48]);
  assert.equal(window.hierarchy.rejection, undefined);
  assert.deepEqual(window.capacity, [72,48,48]);
});

test("existing lockstep and semi-coarsened plans retain their levels", () => {
  assert.deepEqual(planUniformCM11aHierarchy([64,64,64]).levelDimensions,
    [[64,64,64],[32,32,32],[16,16,16],[8,8,8],[4,4,4],[2,2,2]]);
  assert.deepEqual(planUniformCM11aHierarchy([128,128,8]).levelDimensions,
    [[128,128,8],[64,64,4],[32,32,2],[16,16,2],[8,8,2],[4,4,2],[2,2,2]]);
  assert.ok(planUniformCM11aHierarchy([72.5,48,48]).rejection);
  assert.ok(planUniformCM11aHierarchy([0,48,48]).rejection);
});
