import assert from "node:assert/strict";
import test from "node:test";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_STATUS } from "../lib/methods/uniform/uniform-page-pool";
import { uniformPagePoolRule } from "../lib/methods/uniform/uniform-particle-census";
import { uniformLatticePagedVolume } from "../lib/methods/uniform/uniform-volume-initial";
import { advanceUniform, createUniformSolver, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";

test("the hose-fill pool covers the bath and open-top support at each resolution", () => {
  for (const h of [0.05, 0.025, 0.02, 0.0125]) {
    const scene = createUniformTroughScene("hose-fill", h);
    const { volume } = uniformLatticePagedVolume(scene, UNIFORM_PAGE_POOL_EDGE);
    // The eventual footprint is the entire bath, with the same nozzle's
    // ballistic and closure reach above it, not just the initial source.
    const fullBathBudget = uniformPagePoolRule(scene, volume, [])!;
    assert.ok(scene.fluid.pagePool!.blocks! >= fullBathBudget, `h=${h}: cover ${fullBathBudget} blocks`);
    assert.ok(Number(scene.fluid.tiers?.fineBlocks) >= fullBathBudget - 2, `h=${h}: detail covers the growing footprint`);
    assert.ok(scene.fluid.tiers!.bricks! * 8 >= (fullBathBudget - 2) * 512, `h=${h}: owners cover the detail pages`);
  }
  for (const mode of ["settled-tank", "dam-break"] as const) {
    assert.equal(createUniformTroughScene(mode).fluid.pagePool, undefined);
  }
});

test("the hose-fill trough can spread beyond its initially pinned pages", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("hose-fill trough page growth", async device => {
    const scene = createUniformTroughScene("hose-fill");
    const solver = await createUniformSolver(device, scene);
    try {
      let peakResident = 0;
      const frames = Math.round(scene.duration_s / scene.numerics.fixedDt_s);
      for (let frame = 1; frame <= frames; frame++) {
        await advanceUniform(solver, frame * scene.numerics.fixedDt_s);
        assert.equal(solver.info.uniformMixedRefusedTiles ?? 0, 0, `frame ${frame}: growing water retains requested detail`);
        if (frame % 60 === 0) {
          const address = await readUniformLiveAddress(device, solver);
          peakResident = Math.max(peakResident, address.seed.state[UNIFORM_PAGE_POOL_STATUS.resident]!);
        }
        if (frame % 300 === 0) console.info(`hose-fill trough: completed frame ${frame}/${frames}`);
      }
      assert.ok((await solver.readStats()).volumeCellSum! > 0, "the hose adds liquid");
      assert.ok(peakResident > 8, "the run exercises growth beyond the old eight-slot pool");
      t.diagnostic(`completed ${frames} frames; peak sampled residency ${peakResident}`);
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
