// R2's calm demotion (brick-census.wgsl.ts censusDecide) on moving water 4h cannot hold. Particles live only in h
// bricks, so a tile R2 demotes loses its particles; these pin two cases 35eb7ee3's tile-mean tests demoted.
//
// - A fast smooth jet: nbflip-figure-7-pour's falling sheet. A sheet sliding along itself is calm by every tile-mean
//   measure (its tile-mean velocity has no second difference, |u.n| is about 0, its tile-mean normals barely turn), yet
//   the 4h corners do not hold its shape (the shape trigger: the 4h surface misses the h surface by more than the
//   census tolerance). Without the shape trigger in R2's exclusion the upper stream went from 5% of its crossing tiles
//   at 4h to 27% (step 42; 216k stream particles to 175k at step 60). At step 42 (0.7 s) at most 15% of the crossing
//   tiles above 0.4 of the lattice height are 4h-owned (measured: 4-8% with the trigger in the exclusion, 27% without).
// - Shear inside a tile: sparse-cm12-long-dam-break's flow. The second difference of tile means (TC word 15) cannot see
//   velocity that varies within a tile, which a 4h tile's trilinear field cannot hold; an h tile's per-cell defect
//   (certFine, the DEF word) does. With TC word 15 alone the run held 36-75k particles at steps 100-300; with the
//   per-cell defect 117-119k (before 35eb7ee3, 111-114k). At step 100 both banks hold at least 90k.
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
import { H, brickKeyCoord } from "./brick-abi";
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

test("R2 (GPU): the long dam's sheared flow stays h (an h tile's per-cell defect keeps it from calm)", GPU, async () => {
  await runScene("sparse-cm12-long-dam-break", 100, async s => {
    const hdr = await s.brickEngine.read("set", 0, 32), counts = [hdr[H.PCOUNT0]!, hdr[H.PCOUNT1]!];
    console.log(`long dam step 100: particles ${counts.join(" / ")}`);
    assert.ok(Math.min(...counts) >= 90000, `the flow keeps its particles: ${counts.join(" / ")} (at least 90k)`);
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
