/** The page tail's 16h up leg (pUp4) takes its pages red-black on the last
 * tail cycle, so a cycle contracts the BULK16 page rows without flipping their sign. As
 * block Jacobi it flipped them to -0.47..-0.67 of their start, and the warm
 * started frames (init(n+1) = 2 r(n) - r(n-1)) hold any per-cycle ratio
 * under -1/3 at the stop's floor: the rest tank ran a cycle every frame.
 * UniformPressureArena.diagnostics records each BULK16 row's ratio after the
 * first cycle (header row 5: class 7 the largest sign flip, class 0 the
 * largest same-sign ratio) for rows that start above the f32 scale. */
import assert from "node:assert/strict";
import test from "node:test";

test("a cycle contracts the rest tank's BULK16 rows without a sign flip", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const [{ withUniformDevice, createUniformSolver }, { createBulkRestTankScene, BULK_REST_TANK_ID }, { getSceneDefinition }, { UniformPressureArena }] = await Promise.all([
    import("./helpers/uniform-geometric"), import("../lib/core/bulk-rest-tank-scene"), import("../lib/core/scenes"), import("../lib/methods/uniform/uniform-pressure-arena")]);
  await import("../lib/methods");
  const steps = 12, words = 63 + 128;
  const scene = createBulkRestTankScene(0.01, { pagesX: 8, pagesY: 4, pagesZ: 4 });
  scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
  const f32 = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0]!;
  UniformPressureArena.diagnostics = true;
  try {
    await withUniformDevice("pressure tail red-black", async device => {
      const solver = await createUniformSolver(device, scene, getSceneDefinition(BULK_REST_TANK_ID).methodProfile?.overrides);
      const index = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer } } }).mixedFrame.band.index;
      const read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const flips: number[] = [], cycles: number[] = [];
      try {
        for (let step = 1; step <= steps; step++) {
          assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s));
          await solver.awaitFrameCompletion();
          const e = device.createCommandEncoder(); e.copyBufferToBuffer(index, 0, read, 0, 4 * words); device.queue.submit([e.finish()]);
          await read.mapAsync(GPUMapMode.READ); const h = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
          cycles.push(h[4]!);
          // Row 5 class 7: the largest -r1/r0 over the BULK16 rows whose sign flipped (0: none).
          if (h[4]! > 0) flips.push(f32(h[63 + 8 * 5 + 7]!));
          const stats = await solver.readStats();
          assert.ok(Math.abs(stats.volumeDrift ?? 0) < 1e-5, `step ${step} volume drift ${stats.volumeDrift}`);
        }
      } finally { read.destroy(); solver.destroy(); }
      assert.ok(flips.length > 0, `some frame cycles (cycles ${cycles})`);
      assert.ok(Math.max(...flips) < 1 / 3, `the first cycle's largest BULK16 sign flip ${Math.max(...flips)} (cycles ${cycles}, flips ${flips})`);
    });
  } finally { UniformPressureArena.diagnostics = false; }
});
