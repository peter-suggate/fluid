// A ball dropped into a running brick scene (2026-11-19): injectLiquidBall queues a live drop that the next submitted
// step adds, with no re-seed. The dam runs 10 steps, takes a ball in its air, and:
// - the step after the call holds the ball: the presentation's volume rises by the ball's (in cells), within BALL;
// - no live drop is left pending, no FAIL latches, and over the next 20 steps the volume holds within DRIFT of the
//   dam plus the ball;
// - a ball the fixed pools cannot take is refused (false) with nothing queued, and the run goes on.
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { brickDropRefusal, type BrickSceneSetup } from "../lib/methods/narrow-band/brick-scene";
import { brickLiveDrop } from "../lib/methods/narrow-band/brick-source";
import { withUniformDevice } from "./helpers/uniform-geometric";

const BALL = 0.15, DRIFT = Number(process.env.LIVE_DROP_DRIFT ?? 0.05);

test("brick engine: a ball dropped into the running dam is in the field the next step, no re-seed", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick live drop", async device => {
    const definition = getSceneDefinition("minimal-power-dam-break-32"), scene = structuredClone(sceneDocument(definition));
    const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
    const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
    const solver = await createBrickSolver(device, scene, values, quality), e = solver.brickEngine;
    const setup = (solver as unknown as { options: { setup: BrickSceneSetup } }).options.setup, h = setup.h_m;
    const volume = async () => {
      await e.fence();
      const tex = solver.brickPresentation.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
      const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ); const f = new Float32Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
      let total = 0;
      for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) for (let x = 0; x < w; x++) total += f[(z * hgt + y) * bpr / 4 + x]!;
      return total;
    };
    let step = 0;
    const run = async (n: number) => {
      for (let i = 0; i < n; i++) {
        step++;
        assert.ok(solver.advanceTo(step * dt, []), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
      }
    };
    try {
      await run(10);
      const v0 = await volume(), cells = setup.presentCells ?? setup.lattice;
      // In the air above the dam's far side: x at three quarters, y at three quarters, z centred; radius 3 cells.
      const r = 3, centre_m = { x: setup.origin_m[0] + 0.75 * cells[0] * h, y: setup.origin_m[1] + 0.75 * cells[1] * h, z: setup.origin_m[2] + 0.5 * cells[2] * h };
      const ball = (4 / 3) * Math.PI * r ** 3;
      const why = brickDropRefusal(brickLiveDrop({ centre_m, radius_m: r * h }, e.nextDropTime(), { lattice: setup.lattice, ...(setup.box ? { box: setup.box } : {}), h_m: h, origin_m: setup.origin_m }),
        e.receipt?.header, e.layout.cap, e.stages.tiered === true, setup.particleBandDepth, e.layout.window);
      t.diagnostic(`pools f ${e.layout.cap.f} c ${e.layout.cap.c} particles ${e.layout.cap.particles}; receipt step ${e.receipt?.step}; refusal ${why ?? "none"}`);
      assert.equal(solver.injectLiquidBall({ centre_m, radius_m: r * h }), true, `taken (${why})`);
      assert.equal(e.pendingDrops, 1);
      await run(1);
      const v1 = await volume();
      t.diagnostic(`V ${v0.toFixed(1)} -> ${v1.toFixed(1)} cells, the ball ${ball.toFixed(1)}`);
      assert.equal(e.pendingDrops, 0, "added by the next step");
      assert.ok(Math.abs(v1 - v0 - ball) <= BALL * ball, `the step added ${(v1 - v0).toFixed(1)} cells, the ball is ${ball.toFixed(1)}`);
      await run(20);
      const v2 = await volume(), drift = (v2 - (v0 + ball)) / (v0 + ball);
      t.diagnostic(`20 steps on: V ${v2.toFixed(1)}, drift ${drift.toExponential(2)}`);
      assert.ok(Math.abs(drift) <= DRIFT, `drift ${drift}`);
      // A ball as wide as the container: its box claims every brick, past the fixed F pool.
      const whole = { x: setup.origin_m[0] + 0.5 * cells[0] * h, y: setup.origin_m[1] + 0.5 * cells[1] * h, z: setup.origin_m[2] + 0.5 * cells[2] * h };
      assert.equal(solver.injectLiquidBall({ centre_m: whole, radius_m: cells[0] * h }), false, "refused");
      assert.equal(e.pendingDrops, 0, "nothing queued");
      const edit = await solver.editFluid({ operation: "add", shape: "ball", center_m: whole, radius_m: cells[0] * h });
      assert.equal(edit.accepted, false);
      await run(2);
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
