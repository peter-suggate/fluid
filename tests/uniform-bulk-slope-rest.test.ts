import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* BULK16 Stage B (docs/plans/bulk16-plan-2026-10-28.md), the sloped-bed
 * closed rest: the seep reservoir at 1 m with its throat filled, over a
 * terrain ramp rising from 0.05 m at the box's -x face to 0.30 m at the
 * dam's -x face. The bed cuts the liquid's deep tiles; their cells are open
 * above it and buried below. Still water on a static bed must stay still: no
 * fatal, max speed well under the start's terrain-clip transient (HEAD's h
 * path peaks at 0.25 m/s at step 2 and decays), and the volume flat at the
 * clip's offset (-1.34e-3 from step 1). A cut tile that reads a buried
 * corner as air puts a false free surface on the bed: the liquid sliver over
 * it then accelerates at g into the false void (5 m/s at step 1, a fatal by
 * step 4), which this test catches. */

const STEPS = 6;

function sceneOf(cell: number) {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const definition = { ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { heightScale: 1, widthScale: 1, depthScale: 1 }) };
  const { scene } = sceneDocumentAtLattice(definition, { cellSize_m: cell });
  // Closed: the throat and its mouths filled, the passage's wetted start dropped.
  scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
  scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
  // The bed: a terrain grid ramp, 0.05 m at -x rising linearly to 0.30 m at the dam's -x face.
  const c = scene.container, W = c.width_m, D = c.depth_m, nxg = 65, nzg = Math.ceil(D / (W / 64)) + 1;
  const dam = (16 * Math.floor(Math.round(W / cell) / 32) - 16) * cell, heights: number[] = [];
  for (let j = 0; j < nzg; j++) for (let i = 0; i < nxg; i++) heights.push(0.05 + 0.25 * (i * W / (nxg - 1)) / dam);
  scene.terrain = { baseHeight_m: 0, features: [], grid: { kind: "grid", origin_m: { x: -W / 2, z: -D / 2 }, spacing_m: W / (nxg - 1), size: { nx: nxg, nz: nzg }, heights_m: heights } };
  scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 164 };
  scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0;
  scene.duration_s = Math.max(scene.duration_s, STEPS * scene.numerics.fixedDt_s);
  return { scene, values: { ...definition.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0 } as never };
}

test("BULK16 sloped-bed closed rest: still water over a cut bed stays still", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { scene, values } = sceneOf(0.005);
  await withUniformDevice("bulk16 sloped-bed rest", async device => {
    const solver = await createUniformSolver(device, scene, values);
    const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
    try {
      for (let step = 1; step <= STEPS; step++) {
        frame.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []), `step ${step} must be admitted`);
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `step ${step}`);
        const stats = await solver.readStats(), speed = stats.maxSpeed_m_s ?? NaN, drift = stats.volumeDrift ?? NaN;
        assert.ok(speed < 0.5, `step ${step}: max speed ${speed} m/s at rest`);
        assert.ok(Math.abs(drift) < 1.5e-3, `step ${step}: volume drift ${drift}`);
      }
    } finally { solver.destroy(); }
  });
});
