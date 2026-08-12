import assert from "node:assert/strict";
import test from "node:test";
import { createMassConservingFigure9DamBreak } from "../lib/paper-scenarios";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { validateScene } from "../lib/model";

test("CM12 Figure 9 harness uses every published numerical parameter", () => {
  const scene = createMassConservingFigure9DamBreak();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneLatticeDimensions(scene), [128, 128, 64]);
  assert.equal(scene.voxelDomain.finestCellSize_m, 0.05);
  assert.equal(scene.numerics.fixedDt_s, 1 / 30);
  assert.equal(scene.numerics.maxDt_s, 1 / 30);
  assert.deepEqual(scene.fluid.gravity_m_s2, { x: 0, y: -10, z: 0 });
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
});
