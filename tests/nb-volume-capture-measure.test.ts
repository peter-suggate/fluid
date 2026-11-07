import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import type { SceneDescription } from "../lib/core/model";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The NB budget's telemetry is one measure in every row. The target is the
// occupancy capture; the rows' volume was the dense fill (occupancy) before
// any probe and phi's measure after, so a still tank "jumped" at step 1 by
// the gap between the two measures. The capture now takes phi's measure of
// the same surface, so the unadvanced row and the first step's row agree,
// and the gap is its own field: drift against the target starts from it.
function stillTank(): SceneDescription {
  const scene = createBackyardPoolPartyScene();
  scene.interaction = undefined;
  scene.solidBoundary = undefined;
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 3.2, height_m: 4.8, depth_m: 3.2 };
  // A floor and four walls (0.05 m cells); the liquid fills the basin to 1.8 m.
  scene.solidVoxels = ([[[0, 0, 0], [64, 4, 64]], [[0, 0, 0], [8, 40, 64]], [[56, 0, 0], [64, 40, 64]], [[8, 0, 0], [56, 40, 8]], [[8, 0, 56], [56, 40, 64]]] as const)
    .map(([minimum, maximumExclusive]) => ({ operation: "fill", minimum: [...minimum], maximumExclusive: [...maximumExclusive], materialId: 1 })) as unknown as SceneDescription["solidVoxels"];
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -1.2, y: 0.2, z: -1.2 }, max_m: { x: 1.2, y: 1.8, z: 1.2 } }];
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 64 };
  return scene;
}

test("NB volume telemetry: a still tank's unadvanced row and first-step row are one measure", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("nb capture measure", async device => {
    const scene = stillTank(), solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
    try {
      // The capture the wet-cloth path takes before any step.
      const target = await frame.initializePhysicalWaterBudget();
      const row0 = { ...(await solver.readStats()) };
      assert.ok(solver.advanceTo(1 / 60, []));
      await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
      const row1 = { ...(await solver.readStats()) };
      for (const [name, row] of [["row 0", row0], ["row 1", row1]] as const) {
        assert.equal(row.volumeTelemetrySource, "narrow-band-phi", `${name} reports phi's measure`);
        assert.equal(row.narrowBandTargetVolume_cells, target, `${name} keeps the occupancy target`);
      }
      const v0 = row0.volumeCellSum!, v1 = row1.volumeCellSum!, gap = row0.narrowBandCaptureMeasureGap_cells!;
      assert.ok(v0 > 0 && Number.isFinite(gap), `row 0 ${v0}, gap ${gap}`);
      // Row 0 is phi's measure of the captured surface: target plus the gap.
      assert.ok(Math.abs(v0 - (target + gap)) <= 1e-6 * target, `row 0 ${v0} against target ${target} + gap ${gap}`);
      assert.equal(row1.narrowBandCaptureMeasureGap_cells, gap, "the capture gap is fixed at capture");
      // One still step in one measure: well under the measure switch's 1.3e-3.
      assert.ok(Math.abs(v1 - v0) <= 1e-4 * v0, `row 1 ${v1} against row 0 ${v0} (${(v1 - v0) / v0})`);
      const drift0 = row0.narrowBandVolumeBudgetDrift!, drift1 = row1.narrowBandVolumeBudgetDrift!;
      assert.ok(Math.abs(drift1 - drift0) <= 1e-4, `budget drift row 0 ${drift0}, row 1 ${drift1}`);
    } finally { solver.destroy(); }
  });
});
