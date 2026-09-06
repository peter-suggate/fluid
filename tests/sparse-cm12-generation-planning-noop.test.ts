import assert from "node:assert/strict";
import test from "node:test";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

test("negative preflight and no-op planning preserve pressure feedback and avoid transfer capture", async () => {
  for (const needed of [false, true]) {
    const atlas = { bricks: [], dimensions: [8, 8, 8], brickFineResolution: 8, directory: new Map() };
    const receipt = { executed: 16, encoded: 32 };
    let captures = 0;
    const accepted = {
      async captureGenerationPlanningSource() {
        captures++;
        return { atlas, recordsByKey: new Map(), planned: new Map(), activity: { acceptedSteps: 1 } };
      },
      async captureGenerationTransferSource() { throw new Error("no-op captured transfer rows"); },
    };
    const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
      disposed: false, scene: { fluid: {} }, info: { encodedSteps: 1 }, atlas,
      options: { activityPolicy: { coarseFirst: true, topologyCadenceSteps: 1 } },
      topologyGenerationPolicyDirty: false, topologyGenerationLimits: { maximumSpanBricks: 8 },
      presentation: { allocatedBytes: 0 }, pressureIterationReceipt: receipt,
      sparseRuntime: { generationPlanningRequired: true, acceptedAtlas: atlas, allocatedBytes: 0,
        async needsDetailedGenerationPlanning() { return needed; },
        async prepareResidentGeneration(build: (source: typeof accepted) => Promise<unknown>) { await build(accepted); } },
    });
    solver.scheduleTopologyGeneration(); await solver.waitForTopologyReady();
    assert.equal(solver.info.topologyGenerationError, undefined);
    assert.equal(captures, Number(needed));
    assert.equal(solver.pressureIterationReceipt, receipt);
  }
});
