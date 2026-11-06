/** The first V-cycle's contraction on a seam scene with a moving free
 * surface: a 2-cell drop falls onto the BULK16 rest tank, so band rows meet
 * native 4h rows at the seams and the surface rows' ghost faces move. Level 1
 * weighs a fixed-zero face out of an aggregate (a free surface, or a link to
 * a row outside the solve) at the aggregate's centre, half a cell further than
 * its child's (UniformPressureArena pRebase, rFixed): the children's sum
 * over-weighs it by (theta+0.5)/theta. With the sum, the residual after the
 * first cycle averaged 0.12 here (frames 21-40, geometric mean); with rFixed 0.08. */
import assert from "node:assert/strict";
import test from "node:test";

test("the first cycle contracts a seam scene's free-surface rows", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  const [{ withUniformDevice, createUniformSolver }, { createBulkRestTankScene, BULK_REST_TANK_ID }, { getSceneDefinition }] = await Promise.all([
    import("./helpers/uniform-geometric"), import("../lib/core/bulk-rest-tank-scene"), import("../lib/core/scenes")]);
  await import("../lib/methods");
  const steps = 40, from = 20, words = 32, cell = 0.01;
  const scene = createBulkRestTankScene(cell, { pagesX: 8, pagesY: 4, pagesZ: 4 });
  scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
  // tools/probe-tank-pressure-trace.ts --drop=2: 8 cells over the surface, 16 in from the -x/-z walls.
  const box = scene.fluid.initialLiquidVolumes![0]! as { min_m: { x: number; y: number; z: number }; max_m: { x: number; y: number; z: number } };
  const x = box.min_m.x + 16 * cell, z = box.min_m.z + 16 * cell, y = box.max_m.y + 8 * cell;
  scene.fluid.initialLiquidVolumes!.push({ shape: "box", min_m: { x, y, z }, max_m: { x: x + 2 * cell, y: y + 2 * cell, z: z + 2 * cell } });
  const f32 = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0]!;
  await withUniformDevice("pressure seam cycle", async device => {
    const solver = await createUniformSolver(device, scene, getSceneDefinition(BULK_REST_TANK_ID).methodProfile?.overrides);
    const index = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer } } }).mixedFrame.band.index;
    const read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const after: number[] = [];
    try {
      for (let step = 1; step <= steps; step++) {
        assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s));
        await solver.awaitFrameCompletion();
        const e = device.createCommandEncoder(); e.copyBufferToBuffer(index, 0, read, 0, 4 * words); device.queue.submit([e.finish()]);
        await read.mapAsync(GPUMapMode.READ); const h = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
        // Header word 9: the cycle history's residual after the first cycle (relative to the start).
        if (step > from && h[4]! > 0 && f32(h[9]!) > 0) after.push(f32(h[9]!));
        const stats = await solver.readStats();
        assert.ok(Math.abs(stats.volumeDrift ?? 0) < 1e-3, `step ${step} volume drift ${stats.volumeDrift}`);
      }
    } finally { read.destroy(); solver.destroy(); }
    assert.ok(after.length >= (steps - from) / 2, `frames that cycle: ${after.length}`);
    const mean = Math.exp(after.reduce((s, r) => s + Math.log(r), 0) / after.length);
    t.diagnostic(`after one cycle: geometric mean ${mean.toFixed(4)}, frames ${after.map(r => r.toFixed(3))}`);
    assert.ok(mean <= 0.1, `the residual after one cycle (geometric mean over frames ${from + 1}-${steps}) ${mean.toFixed(4)}: ${after.map(r => r.toFixed(3))}`);
  });
});
