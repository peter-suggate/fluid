/**
 * LANE F OWNS THIS FILE. The engine's uploads (brick-engine.ts uploadWords) against Dawn's large-write abort.
 *
 * Dawn (dawn.node, Metal) aborts the whole process on a lone writeBuffer over about 1 MB whose size is not an 8-byte
 * multiple (BufferBase::UploadData). Probed on 2026-11-17: 1 MB + 4 bytes writes, and 4 MB + 4, 4 MB + 12, 8 MB + 4
 * and 16 MB + 4 abort. The boot table of a ball over the letters' pool (23986916 bytes) did too.
 * - uploadWords puts 4 MB + 4 bytes at an offset (a 16-byte body, then the tail) and every word reads back.
 * - The scene that aborted (the letters' pool with a ball as initial liquid, its table 5996729 words) boots through
 *   createBrickSolver and steps.
 * GPU only (WEBGPU_NODE_MODULE).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { uploadWords } from "./brick-engine";
import { BRICK_ENGINE_SCENES, brickSceneSetup } from "./brick-scene";
import { createBrickSolver } from "./brick-method";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import type { SceneDescription } from "../../core/model";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };

test("brick uploads (GPU): 4 MB + 4 bytes at an offset, through uploadWords, read back whole", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick upload", async device => {
    const words = (1 << 20) + 1, offset = 3, data = new Uint32Array(words);
    for (let i = 0; i < words; i++) data[i] = (i * 2654435761) >>> 0;
    const size = 4 * (words + offset + 1), U = GPUBufferUsage;
    const b = device.createBuffer({ size, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const r = device.createBuffer({ size, usage: U.COPY_DST | U.MAP_READ });
    try {
      uploadWords(device.queue, b, 4 * offset, data);
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, r, 0, size); device.queue.submit([enc.finish()]);
      await r.mapAsync(GPUMapMode.READ);
      const back = new Uint32Array(r.getMappedRange().slice(0));
      r.unmap();
      let bad = 0;
      for (let i = 0; i < words; i++) if (back[offset + i] !== data[i]) bad++;
      assert.equal(bad, 0, `${bad} of ${words} words differ`);
      assert.equal(back[0], 0); assert.equal(back[offset + words], 0, "nothing past the upload");
    } finally { b.destroy(); r.destroy(); }
  });
});

test("brick uploads (GPU): a ball over the letters' pool as initial liquid boots (its table aborted Dawn) and steps", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const id = "nbflip-figure-8-letters", METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition(id);
  const base = sceneDocument(definition), h = base.container.width_m / 256;
  const scene: SceneDescription = { ...base, fluid: { ...base.fluid, scheduledDrops: [], initialLiquidVolumes: [{ shape: "sphere", center_m: { x: 0, y: 115.2 * h, z: 0 }, radius_m: 14.42 * h }] } };
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const enabled = BRICK_ENGINE_SCENES.includes(id);
  if (!enabled) (BRICK_ENGINE_SCENES as string[]).push(id);
  try {
    const setup = brickSceneSetup(scene, values).setup!, t = setup.initial!;
    const words = 1 + t.pages.length + t.phi.length + (setup.initialVelocity ? 4 + setup.initialVelocity.tiles.length : 0) + 1;
    assert.ok(4 * words > 1 << 22 && (4 * words) % 8 !== 0, `the boot table is ${4 * words} bytes: over 4 MB and not an 8-byte multiple`);
    await withUniformDevice("brick upload scene", async device => {
      const solver = await createBrickSolver(device, scene, values, quality as never);
      try {
        for (let k = 1; k <= 2; k++) { assert.ok(solver.advanceTo(k * setup.dt_s + 1e-12, [])); await solver.assertSimulationHealthy(); }
      } finally { solver.destroy(); }
    });
  } finally {
    if (!enabled) (BRICK_ENGINE_SCENES as string[]).splice(BRICK_ENGINE_SCENES.indexOf(id), 1);
  }
});
