import assert from "node:assert/strict";
import test from "node:test";
import { planUniformCM11aHierarchy } from "../lib/methods/uniform/pressure-plan";

test("existing lockstep and semi-coarsened plans retain their levels", () => {
  assert.deepEqual(planUniformCM11aHierarchy([64,64,64]).levelDimensions,
    [[64,64,64],[32,32,32],[16,16,16],[8,8,8],[4,4,4],[2,2,2]]);
  assert.deepEqual(planUniformCM11aHierarchy([128,128,8]).levelDimensions,
    [[128,128,8],[64,64,4],[32,32,2],[16,16,2],[8,8,2],[4,4,2],[2,2,2]]);
  assert.ok(planUniformCM11aHierarchy([72.5,48,48]).rejection);
  assert.ok(planUniformCM11aHierarchy([0,48,48]).rejection);
});
