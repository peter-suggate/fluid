// R2's calm demotion (brick-census.wgsl.ts censusDecide) on moving water 4h cannot hold. Particles live only in h
// bricks, so a tile R2 demotes loses its particles; these pin where it must not demote and where it must.
//
// - A fast smooth jet: nbflip-figure-7-pour's falling sheet. A sheet sliding along itself is calm by every tile-mean
//   measure (its tile-mean velocity has no second difference, |u.n| is about 0, its tile-mean normals barely turn), yet
//   the 4h corners do not hold its shape (the shape trigger: the 4h surface misses the h surface by more than the
//   census tolerance). Without the shape trigger in R2's exclusion the upper stream went from 5% of its crossing tiles
//   at 4h to 27% (step 42; 216k stream particles to 175k at step 60). The exclusion holds on a fast tile (R2's speed
//   horizon, 2 ncKeySpeed >= 4 cells a step): at step 42 (0.7 s) at most 15% of the crossing tiles above 0.4 of the
//   lattice height are 4h-owned (measured: 11-12% with the speed horizon, 4-8% with the trigger on every tile, 27%
//   without it).
// - The long dam's thinning layer: sparse-cm12-long-dam-break at 3-3.5 s, a layer a tile or two deep sliding at about
//   0.5 m/s, slow by R2's speed horizon. Its tile means are calm (TC word 15 within budget), but an h tile's per-cell
//   defect (the DEF word, the largest of its 64 cells) reads the h particles' cell-scale noise, about a quarter of |u|,
//   as error, and its 4h corners misfit the layer by about the shape tolerance. With the defect and the shape trigger on
//   every tile (cb56115c) the layer's crossing tiles were 17-18% 4h at 3.5 s and drew bumpy; judged slow, on the tile
//   means alone, 97% (19bcf370's 95-97%, the smoothness the user liked). At step 210 at least 85% of the crossing tiles
//   below 8 cells are 4h.
// - The nozzle: nbflip-figure-7-pour's inflow tiles. Once the plug has filled a nozzle tile it is joined, and R2 read
//   only an unjoined source tile as a source, so 8-11 of the 454 nozzle tiles were 4h at every step (the lip at the
//   nozzle held no particles and drew ragged). A source tile is never calm, joined or not: at step 60 none is 4h.
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
const METHOD = "uniform-narrow-band-flip";

/** The scene's brick solver at fixed dt 1/60, presentation off, stepped to `steps`; check reads the engine. */
async function runScene(sceneId: string, steps: number, check: (s: BrickNBFlipSolver) => Promise<void>) {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const method = getMethod(METHOD);
  const definition = getSceneDefinition(sceneId), base = sceneDocument(definition), profile = definition.methodProfile;
  const dt = 1 / 60, scene = { ...base, numerics: { ...base.numerics, fixedDt_s: dt, maxDt_s: dt } };
  const quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  await withUniformDevice(`brick census ${sceneId}`, async device => {
    const s = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as BrickNBFlipSolver;
    try {
      await s.waitForSimulationReady?.();
      s.setPresentationEnabled?.(false);
      for (let k = 1; k <= steps; k++) {
        while (!s.advanceTo(k * dt + 1e-12, [])) {
          if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message);
          await new Promise<void>(r => setImmediate(r));
        }
      }
      await s.brickEngine.fence(); await s.brickEngine.assertHealthy();
      await check(s);
    } finally { s.destroy(); }
  });
}

test("R2 (GPU): the pour's fast smooth stream stays h (its shape-triggered tiles are not calm-demoted)", GPU, async () => {
  await runScene("nbflip-figure-7-pour", 42, async s => {
    const e = s.brickEngine, L = e.layout;
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
  });
});

test("R2 (GPU): the long dam's slow thinning layer goes 4h (a slow tile is judged on its tile means)", GPU, async () => {
  await runScene("sparse-cm12-long-dam-break", 210, async s => {
    const e = s.brickEngine, L = e.layout, V = L.lattice;
    const snap = await snapshot(e), census = await e.read("census");
    let thin = 0, coarse = 0;
    for (const [key, cs] of snap.pages) {
      const P = brickKeyCoord(key);
      for (let i = 0; i < 512; i++) {
        if (!(census[CENSUS.STRIDE * cs + CENSUS.FIN + i]! & CROSSES)) continue;
        const t = [8 * P[0]! + (i & 7), 8 * P[1]! + ((i >> 3) & 7), 8 * P[2]! + (i >> 6)];
        // The layer: crossing tiles whose cells lie below 8 cells, off the far walls.
        if (4 * t[1]! >= 8 || t.some((c, a) => 4 * c >= V[a]! - 1)) continue;
        thin++;
        if (!((snap.dir[L.dir.cmask + 16 * cs + (i >> 5)]! >>> (i & 31)) & 1)) coarse++;
      }
    }
    console.log(`long dam step 210: ${thin} thin crossing tiles, ${coarse} 4h (${(100 * coarse / thin).toFixed(1)}%)`);
    assert.ok(thin > 200, `the layer is there: ${thin} crossing tiles below 8 cells`);
    assert.ok(coarse >= 0.85 * thin, `${coarse} of the layer's ${thin} crossing tiles are 4h (at least 85%)`);
  });
});

test("R2 (GPU): the pour's nozzle tiles stay h (a joined source tile is still a source)", GPU, async () => {
  await runScene("nbflip-figure-7-pour", 60, async s => {
    const e = s.brickEngine, L = e.layout, dt = 1 / 60;
    const inflow = (s as unknown as { options: { setup: { sources?: { inflow?: { outlet: number[]; radius: number; velocity_mps: number[]; speed: number } } } } }).options.setup.sources?.inflow;
    assert.ok(inflow, "the pour has a nozzle");
    // nbSourceTile (brick-source.wgsl.ts) for a running nozzle at full strength: its plug's distance at the tile centre
    // within half the tile diagonal and two cells.
    const speed = Math.hypot(...inflow.velocity_mps), dir = inflow.velocity_mps.map(v => v / speed), sweep = inflow.speed * dt;
    const plug = (p: number[]) => {
      const d = p.map((x, k) => x - inflow.outlet[k]!), axial = d[0]! * dir[0]! + d[1]! * dir[1]! + d[2]! * dir[2]!;
      const side = Math.hypot(...d.map((x, k) => x - axial * dir[k]!)) - inflow.radius, cap = Math.max(-axial, axial - sweep - 2);
      return cap <= 0 ? side : Math.max(side, cap);
    };
    const snap = await snapshot(e);
    let nozzle = 0; const coarse: string[] = [];
    for (const [key, cs] of snap.pages) {
      const P = brickKeyCoord(key);
      for (let i = 0; i < 512; i++) {
        const t = [8 * P[0]! + (i & 7), 8 * P[1]! + ((i >> 3) & 7), 8 * P[2]! + (i >> 6)];
        if (plug(t.map(c => 4 * c + 2)) > 0.5 * Math.sqrt(48) + 2) continue;
        nozzle++;
        if (!((snap.dir[L.dir.cmask + 16 * cs + (i >> 5)]! >>> (i & 31)) & 1)) coarse.push(t.join(","));
      }
    }
    console.log(`pour step 60: ${nozzle} resident nozzle tiles, ${coarse.length} 4h`);
    assert.ok(nozzle > 100, `the nozzle's tiles are resident: ${nozzle}`);
    assert.equal(coarse.length, 0, `nozzle tiles 4h: ${coarse.slice(0, 8).join(" ")}`);
  });
});
