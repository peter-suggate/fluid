import "../lib/methods";
import test from "node:test";
import assert from "node:assert/strict";
import { createPaneSession } from "../lib/core/session/session";
import { effectiveSimulationStep_s } from "../lib/core/simulation-step";
import { sharePaneStep } from "../lib/core/simulation/shared-pane-step";

function setup() {
  const a = createPaneSession("a"), b = createPaneSession("b");
  a.method.getState().setParam("adaptive-volume", "timeStep", "scene");
  a.scene.getState().patchNumerics({ fixedDt_s: 1 / 60, maxDt_s: 1 / 60 });
  b.method.getState().setMethodId("uniform");
  const shared = sharePaneStep([a, b]);
  const check = (dt: number) => {
    for (const pane of [a, b]) {
      assert.equal(effectiveSimulationStep_s(pane.scene.getState().scene, pane.method.getState()), dt);
      assert.equal(pane.scene.getState().scene.numerics.fixedDt_s, dt);
      assert.equal(pane.scene.getState().scene.numerics.maxDt_s, dt);
    }
  };
  return { a, b, shared, check };
}

test("shared edits reach both panes and survive scene and solver changes", () => {
  const { a, b, shared, check } = setup();
  try {
    shared.setStepSize(0.012);
    check(0.012);
    b.scene.getState().patchNumerics({ fixedDt_s: 1 / 30, maxDt_s: 1 / 30 });
    b.method.getState().resetParams("uniform");
    a.method.getState().setMethodId("uniform");
    check(0.012);
  } finally { shared.stop(); }
  b.scene.getState().patchNumerics({ fixedDt_s: 0.025 });
  assert.equal(b.scene.getState().scene.numerics.fixedDt_s, 0.025);
});
