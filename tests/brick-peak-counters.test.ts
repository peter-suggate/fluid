// LANE P OWNS THIS FILE. The brick engine's peak counters bound what each step held (lane P, 2026-11-17).
//
// Pool authoring takes its pools from the measured peaks (H.PEAK_*), so every counter must be at least what the
// pool held. H.PEAK_PARTICLES was never written (0 on a splash that had 5779 particles after a step and latched a pool of
// 1024 inside one); it is now the bank's high-water at the append, the count PCAP is checked against.
//
// rigid-sink at its own h (16^3 in 32^3): the dense sphere enters the tank and splashes. After every step of 16:
// - PEAK_PARTICLES >= both banks' PCOUNT (each was reached by an append);
// - PEAK_F >= LIVE_F, PEAK_C >= LIVE_C, PEAK_K >= LIVE_K.
// At the end, the splash has particles and PEAK_PARTICLES is at least the largest count read after any step.
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { H } from "../lib/methods/narrow-band/brick-abi";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import { withUniformDevice } from "./helpers/uniform-geometric";

const STEPS = 16;

test("brick engine: the peak counters bound every step's pools, particles included (rigid-sink's splash, 16 steps)", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick peak counters", async device => {
    const definition = getSceneDefinition("rigid-sink"), scene = structuredClone(sceneDocument(definition));
    // Pools to spare: the test reads the counters, it does not test the authored pools.
    scene.fluid.brickPools = { f: 125, c: 8, b: 16, k: 64, particles: 65536 };
    const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
    const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
    const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(scene.sceneId!);
    if (!had) enabled.push(scene.sceneId!);
    const solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}).finally(() => { if (!had) enabled.splice(enabled.indexOf(scene.sceneId!), 1); });
    try {
      await solver.waitForSimulationReady?.();
      const e = (solver as unknown as BrickNBFlipSolver).brickEngine, roster = initializeRigidBodies(scene.rigidBodies);
      let seen = 0;
      for (let step = 1; step <= STEPS; step++) {
        assert.ok(solver.advanceTo(step * dt, roster), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
        const h = await e.read("set", 0, 256), p = Math.max(h[H.PCOUNT0]!, h[H.PCOUNT1]!);
        seen = Math.max(seen, p);
        t.diagnostic(`step ${step}: particles ${h[H.PCOUNT0]} / ${h[H.PCOUNT1]}, peak ${h[H.PEAK_PARTICLES]}; F ${h[H.LIVE_F]} (peak ${h[H.PEAK_F]}), K ${h[H.LIVE_K]} (peak ${h[H.PEAK_K]})`);
        assert.ok(h[H.PEAK_PARTICLES]! >= p, `step ${step}: PEAK_PARTICLES ${h[H.PEAK_PARTICLES]} under the bank's ${p}`);
        assert.ok(h[H.PEAK_F]! >= h[H.LIVE_F]! && h[H.PEAK_C]! >= h[H.LIVE_C]! && h[H.PEAK_K]! >= h[H.LIVE_K]!, `step ${step}: a peak under its live count`);
      }
      assert.ok(seen > 0, "the splash has particles");
      const h = await e.read("set", 0, 256);
      assert.ok(h[H.PEAK_PARTICLES]! >= seen, `PEAK_PARTICLES ${h[H.PEAK_PARTICLES]} under the largest count read, ${seen}`);
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
