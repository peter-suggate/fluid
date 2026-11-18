// A lake on terrain boots and steps on the brick engine (2026-11-18). alpine-lake-sunrise's ground cuts the bottom cell
// layer under a shallow lake: the cut bricks there are F at the boot (inside its band) but under the census's band, and
// the boot's census runs before its setMark, so no F record is LIQUID yet. The static need rule took such a brick as dry,
// the boot's commit took its tiles 4h, and the kick's solve latched (0xa3: a cut face into a native row; rowsC: a liquid
// 4h row in a cut brick). The rule now also reads an h tile's F vertices (brick-static-need.wgsl.ts sdFVertex). The
// operator's own latches are the check: no FAIL through the boot and three steps (unfixed: FAIL at frame 0).
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { H } from "../lib/methods/narrow-band/brick-abi";
import { snapshot } from "../lib/methods/narrow-band/brick-readback";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("brick engine: a lake on terrain (alpine-lake-sunrise) boots and steps with its submerged cut bricks h, no latch; save and load are bitwise", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick terrain lake", async device => {
    const definition = getSceneDefinition("alpine-lake-sunrise"), scene = structuredClone(sceneDocument(definition));
    const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
    const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
    const solver = await createBrickSolver(device, scene, values, quality);
    const e = solver.brickEngine;
    try {
      const s0 = await snapshot(e);
      assert.equal(s0.header[H.FAIL], 0, `boot: FAIL ${s0.header[H.FAIL]} info ${s0.header[H.FAIL_INFO]?.toString(16)}`);
      t.diagnostic(`boot: ${s0.bricks.size} F bricks, live K ${s0.header[H.LIVE_K]}`);
      for (let step = 1; step <= 3; step++) {
        assert.ok(solver.advanceTo(step * dt, []), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.();
        const s = await snapshot(e);
        assert.equal(s.header[H.FAIL], 0, `step ${step}: FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]?.toString(16)}`);
        await solver.assertSimulationHealthy?.();
        t.diagnostic(`step ${step}: F ${s.bricks.size}, live K ${s.header[H.LIVE_K]}`);
      }
      // Save and load (AGENTS.md): a saved state put back reads back bit for bit, the ground's columns in "solid"
      // among it, and a step from it keeps them and latches nothing.
      const saved = await e.save();
      e.advance(1); await e.assertHealthy();
      await e.restore(saved);
      const again = await e.save();
      for (const [name, words] of saved.buffers) assert.deepEqual(again.buffers.get(name), words, `restore: buffer ${name}`);
      e.advance(1); await e.assertHealthy();
      assert.deepEqual(new Uint32Array(await e.read("solid")), saved.buffers.get("solid"), "a step keeps the ground's columns");
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
