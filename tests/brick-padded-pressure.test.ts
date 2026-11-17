// LANE P OWNS THIS FILE. A padded still tank holds still water with the pressure walls at the box (M3 padding,
// lane P 2026-11-17; pressure-composite.wgsl.ts pressureResolveBox, tests/brick-padded-pressure.ts).
//
// rigid-hydrostatic without its body, the container 1.2 x 0.8 x 1.2 m at h = 0.05: 24 x 16 x 24 cells, the lattice
// 32^3 (E's padded regression's padding), half full. Before the switch its rows met the storage edge's walls 8 cells
// past the container, so the walls were twice or nowhere. Two runs, each 15 steps:
// - tiered as the census leaves it (the still tank's 4h native rows meet the box's x, z and top planes);
// - with the census hold (test only) over the tiles at the box's +x / +z corner and the floor, so h rows meet the box
//   planes and their seams to the native rows lie along them.
// Each step: assertPaddedPressure (no row past the box, every box face of a liquid h row WALL and no other, peak |U|
// within the solve's bound 2 MG_TOLERANCE h), and the box observer's volume (ff563656, over layout.box) within 1e-3
// of step 1's (its step-0 read of the boot state sits low, lane F).
// The peak's rise to ~1e-3 m/s by step 2 and step 15, decaying to ~5e-5 by step 40, is the tank's own settling: an
// unpadded 32^3 tank at the same depth runs the same series within 15% (lane P, 2026-11-17).
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { solidVoxelsOnMovedLattice } from "../lib/core/scene-lattice";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import type { BrickCensusHold } from "../lib/methods/narrow-band/brick-census.wgsl";
import { boxGeometry } from "../tools/nbflip-observables";
import { createNBFlipBoxObserver } from "../tools/nbflip-observe";
import { assertPaddedPressure } from "./brick-padded-pressure";
import { withUniformDevice } from "./helpers/uniform-geometric";

const STEPS = 15, H_M = 0.05, DRIFT = 1e-3;
const RUNS: { readonly name: string; readonly hold?: BrickCensusHold }[] = [
  { name: "tiered" },
  { name: "held at the box corner", hold: { lo: [3, 0, 3], hi: [6, 2, 6] } },
];

for (const run of RUNS) {
  test(`brick engine: a padded still tank (24x16x24 in 32^3) holds still with the pressure walls at the box, ${run.name}, 15 steps`, { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
    await withUniformDevice("brick padded pressure", async device => {
      const definition = getSceneDefinition("rigid-hydrostatic"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
      scene.container = { ...scene.container, width_m: 24 * H_M, height_m: 16 * H_M, depth_m: 24 * H_M, fillFraction: 0.5 };
      scene.voxelDomain.finestCellSize_m = H_M; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
      scene.fluid.brickPools = { f: 64, c: 8, b: 16, k: 0, particles: 65536 }; scene.rigidBodies = [];
      const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
      const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
      const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
      const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(scene.sceneId!);
      if (!had) enabled.push(scene.sceneId!);
      const solver = await createBrickSolver(device, scene, values, quality, run.hold ? { censusHold: run.hold } : {}).finally(() => { if (!had) enabled.splice(enabled.indexOf(scene.sceneId!), 1); });
      let observer: Awaited<ReturnType<typeof createNBFlipBoxObserver>> | undefined;
      try {
        const e = solver.brickEngine;
        assert.deepEqual([...e.layout.lattice], [32, 32, 32]); assert.deepEqual([...e.layout.box], [24, 16, 24]);
        observer = await createNBFlipBoxObserver(device, solver, scene, boxGeometry(scene, "paddleboats"));
        let hRows = 0, v1 = NaN;
        for (let step = 1; step <= STEPS; step++) {
          assert.ok(solver.advanceTo(step * dt, []), `step ${step} admitted`);
          await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
          const m = await assertPaddedPressure(e, H_M, { step, still: true });
          const v = (await observer.observe()).total; if (step === 1) v1 = v;
          const drift = Math.abs(v - v1) / v1;
          hRows = Math.max(hRows, m.hRows);
          t.diagnostic(`step ${step}: ${m.hRows} h rows (${m.boxFaces} box faces), ${m.nativeRows} native rows, peak |U| ${m.peak.toExponential(2)} m/s (walls ${m.wallPeak.toExponential(2)}) at ${m.where}, drift ${drift.toExponential(2)}`);
          assert.ok(drift <= DRIFT, `step ${step}: volume drift ${drift} from step 1`);
        }
        if (run.hold) assert.ok(hRows >= 512, `the hold puts h rows on the box planes (${hRows})`);
      } finally { observer?.destroy(); solver.destroy(); }
    }, ["subgroups"]);
  });
}
