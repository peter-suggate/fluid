// R2's calm demotion (brick-census.wgsl.ts censusDecide) on a fast smooth jet: nbflip-figure-7-pour's falling sheet.
//
// A sheet sliding along itself is calm by every R2 measure (its tile-mean velocity has no second difference, |u.n| is
// about 0, its tile-mean normals barely turn), yet the 4h corners do not hold its shape (the shape trigger: the 4h
// surface misses the h surface by more than the census tolerance). Particles live only in h bricks, so a crossing tile
// R2 demotes loses the sheet's particles and the stream breaks into blobs. 35eb7ee3 dropped the shape trigger from R2's
// exclusion and the pour's upper stream went from 5% of its crossing tiles at 4h to 27% (step 42; 216k stream
// particles to 175k at step 60). This pins the stream at h: at step 42 (0.7 s) at most 15% of the crossing tiles above
// 0.4 of the lattice height are 4h-owned (measured: 4-8% with the shape trigger in the exclusion, 27% without).
// GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { CENSUS } from "./brick-census.wgsl";
import { brickKeyCoord } from "./brick-abi";
import { snapshot } from "./brick-readback";
import type { BrickNBFlipSolver } from "./brick-solver";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const CROSSES = 1 << 28;

test("R2 (GPU): the pour's fast smooth stream stays h (its shape-triggered tiles are not calm-demoted)", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD);
  const definition = getSceneDefinition("nbflip-figure-7-pour"), base = sceneDocument(definition), profile = definition.methodProfile;
  const dt = 1 / 60, scene = { ...base, numerics: { ...base.numerics, fixedDt_s: dt, maxDt_s: dt } };
  const quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  await withUniformDevice("brick census jet", async device => {
    const s = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as BrickNBFlipSolver;
    try {
      await s.waitForSimulationReady?.();
      s.setPresentationEnabled?.(false);
      for (let k = 1; k <= 42; k++) {
        while (!s.advanceTo(k * dt + 1e-12, [])) {
          if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message);
          await new Promise<void>(r => setImmediate(r));
        }
      }
      const e = s.brickEngine, L = e.layout;
      await e.fence(); await e.assertHealthy();
      const snap = await snapshot(e), census = await e.read("census"), yCut = 0.4 * L.lattice[1]!;
      let stream = 0, coarse = 0;
      for (const [key, cs] of snap.pages) {
        const P = brickKeyCoord(key);
        for (let i = 0; i < 512; i++) {
          if (!(census[CENSUS.STRIDE * cs + CENSUS.FIN + i]! & CROSSES)) continue;
          if (4 * (8 * P[1]! + ((i >> 3) & 7)) < yCut) continue;
          stream++;
          if (!((snap.dir[L.dir.cmask + 16 * cs + (i >> 5)]! >>> (i & 31)) & 1)) coarse++;
        }
      }
      console.log(`pour step 42: ${stream} stream crossing tiles, ${coarse} 4h (${(100 * coarse / stream).toFixed(1)}%)`);
      assert.ok(stream > 500, `the stream is there: ${stream} crossing tiles above ${yCut}`);
      assert.ok(coarse <= 0.15 * stream, `${coarse} of the stream's ${stream} crossing tiles are 4h (at most 15%)`);
    } finally { s.destroy(); }
  });
});
