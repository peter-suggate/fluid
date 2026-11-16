/**
 * The memory-must-scale check on the real stage set (lane E): the high-resolution dam's liquid box in a container 4x
 * wider in x (512 x 128 x 128 cells, whole pages), tiered, with lane F's stages and lane P's composite solver.
 * Test-only: no UI scene. brick-engine.test.ts checks the same rule on the bring-up stages.
 *
 * - Memory follows the pools: at the same pools the two lattices allocate the same buffers, apart from the
 *   directory's page window (one word a 32^3 page).
 * - Work follows the fluid: until the front reaches the narrow box's far wall (x = 128, about step 12) the two runs
 *   hold the same fluid, and their residency and lists agree. After it, the wide run's residency grows with its
 *   spreading sheet, and no brick is resident far past the liquid's front: the empty far end of the box costs nothing.
 *
 * Measured (progress-lane-e.md 2026-11-16, ~/code/me/fluid-scratch/lane-e/wide): step 10 liveF 2243 / 2244, F_ACT
 * 1906 / 1899, F_LIQ 366 / 362; by step 40 the wide sheet holds 12764 live F against 4500 (F_LIQ 1797 against 1234).
 * The excess is the reach class (6957 bricks at step 45 against 458): the flow box is isotropic, (max BOUND_F + g dt)
 * dt / h = 21.8 cells at step 40 (8 m/s spray), so a fast thin sheet is held 3 bricks deep above and below. The step
 * costs 14-21 ms a thousand live F bricks in both boxes.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { H, LIST, brickKeyCoord, type BrickCapacities } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions } from "./brick-engine";
import { brickSceneSetup } from "./brick-scene";
import { fluidStagesAt } from "./brick-fluid-stages";
import { FT, fluidLayout } from "./fluid-grid.wgsl";
import { compositeMGPCG } from "./brick-pressure-solver";
import { snapshot } from "./brick-readback";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };

/** The dam's engine options, the container `wide` x wider in x (the same liquid box in cells), at `pools`. */
function damOptions(wide: number, pools?: Partial<BrickCapacities>): BrickEngineOptions {
  const METHOD = "uniform-narrow-band-flip";
  const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const { setup, gaps } = brickSceneSetup(scene, values);
  assert.ok(setup, gaps.join("; "));
  return {
    lattice: [wide * setup.lattice[0], setup.lattice[1], setup.lattice[2]], h_m: setup.h_m, dt_s: setup.dt_s, gravity_mps2: setup.gravity_mps2,
    capacities: { ...setup.capacities, ...pools }, liquid: setup.liquid, solver: compositeMGPCG(), stages: fluidStagesAt(true), values,
    fluid: { density_kg_m3: setup.density_kg_m3, viscosity_Pa_s: setup.viscosity_Pa_s, surfaceTension_N_m: setup.surfaceTension_N_m }, label: `dam-x${wide}`,
  };
}

/** The header's counts after step k (its bank: PCOUNT[k & 1]). */
const counts = (h: Uint32Array, k: number) => ({ liveF: h[H.LIVE_F]!, fAct: h[H.LIST_COUNT + LIST.F_ACT]!, fLiq: h[H.LIST_COUNT + LIST.F_LIQ]!, cAll: h[H.LIST_COUNT + LIST.C_ALL]!, particles: h[H.PCOUNT0 + (k & 1)]! });

test("brick wide dam (GPU): the same pools on a 4x-wider lattice allocate the same buffers, apart from the page window", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick wide dam bytes", async device => {
    const at = async (wide: number) => { const e = await BrickEngine.create(device, damOptions(wide)); const r = { bytes: e.bytes, dir: e.layout.dir.words }; e.destroy(); return r; };
    const a = await at(1), b = await at(4);
    console.log(`wide dam bytes: ${a.bytes} (128^3) and ${b.bytes} (512 x 128^2); directory ${a.dir} and ${b.dir} words`);
    assert.equal(b.bytes - a.bytes, 4 * (b.dir - a.dir), "only the directory's page window grows with the lattice");
  });
});

test("brick wide dam (GPU): residency and lists follow the fluid, not the box", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick wide dam", async device => {
    const head = async (e: BrickEngine) => { const h = await e.read("set", 0, 256); assert.equal(h[H.FAIL], 0, `${e.options.label}: FAIL ${h[H.FAIL]} info 0x${h[H.FAIL_INFO]!.toString(16)}`); return h; };
    // The same fluid while the front is short of x = 128. The engine is not bitwise reproducible run to run, and need
    // not be: the set's lists and slots are built by atomic appends in arrival order (setLists, setClaimBricks, pSeed,
    // pMove, pmgCoarsenTop's row bricks and coarse rows), and every kernel is deterministic on the same input (lane F
    // 2026-11-16, ~/code/me/fluid-scratch/lane-f/tools/probe-det.mts). Sums folded in list order still move in the last
    // bits: the CG scalars' fold (pressure-mgpcg.ts R_ZF + g, g a position in pmgCoarsenTop's T_RB list) and the coarse
    // levels' assembly. The splash is chaotic, so by step 6 one particle's speed, the max BOUND_F, reads 2.08 or 2.67 m/s
    // and from step 7 the narrow runs' residency spreads past 1% even at the same ceil(flow) (step 9 liveF 1335-1376 at
    // ceil flow 8, 36 runs). So steps 1-6, identical in every run, compare tightly against the narrow runs, and steps
    // 7-10 are not compared: a band around the narrow runs flaked (step 9 liveF wide 1401 against narrow 1338-1352),
    // and against their median the wide run came within 5.1% of it (10 runs), too close to any bound that still
    // tells the chaos from a defect. A leak into the empty far end costs hundreds of bricks, and the reach asserts at
    // step 40 below hold it.
    const early = 6, narrowRuns = 5;
    const n: ReturnType<typeof counts>[][] = Array.from({ length: early }, () => []);
    for (let r = 0; r < narrowRuns; r++) {
      const narrow = await BrickEngine.create(device, damOptions(1));
      try { for (let k = 1; k <= early; k++) { narrow.advance(1); n[k - 1]!.push(counts(await head(narrow), k)); } } finally { narrow.destroy(); }
    }
    console.log(`wide dam narrow runs, liveF by step: ${n.map(v => v.map(x => x.liveF).join("/")).join(" ")}`);
    const near = (a: number, b: number) => Math.abs(a - b) <= Math.max(2, 0.01 * b);
    // The wide sheet's measured peak by step 40 (12764 F, 316 C, 1.87M particles), with a margin.
    const wide = await BrickEngine.create(device, damOptions(4, { f: 17000, c: 425, particles: 3_500_000 }));
    try {
      for (let k = 1; k <= 40; k++) {
        wide.advance(1);
        const w = counts(await head(wide), k);
        if (k <= early) {
          const runs = n[k - 1]!;
          for (const key of ["fLiq", "liveF", "fAct", "cAll"] as const) {
            const xs = runs.map(x => x[key]);
            assert.ok(xs.some(x => near(w[key], x)), `step ${k} ${key}: wide ${w[key]} against narrow ${xs.join("/")}`);
          }
        }
      }
      // No brick far past the fluid: past the furthest brick holding liquid or particles, F reaches no further than
      // the residency rule's box (the flow (max BOUND_F + g dt) dt / h, 9 covering every class's margin past the fluid
      // (R3: the h ring 4, the particles flow + 4, the seeds 6 past a heat tile) and a brick's rounding),
      // and pages no further than the page sources' flow + 20 and a page's rounding.
      const s = await snapshot(wide), slotKey = new Map([...s.bricks].map(([key, slot]) => [slot, key])), L = wide.layout, o = wide.options;
      const xOf = (slots: number[]) => Math.max(...slots.map(slot => brickKeyCoord(slotKey.get(slot)!)[0]));
      const fluidX = Math.max(xOf(s.list(LIST.F_LIQ)), xOf(s.list(LIST.F_PART)));
      const liveX = Math.max(...[...s.bricks.keys()].map(key => brickKeyCoord(key)[0]));
      const pagesX = Math.max(...[...s.pages.keys()].map(key => brickKeyCoord(key)[0]));
      const bound = Math.max(...[...s.bricks.values()].map(slot => new Float32Array(new Uint32Array([s.set[L.set.boundF + slot]!]).buffer)[0]!));
      const flow = (bound + Math.hypot(...o.gravity_mps2) * o.dt_s) * o.dt_s / o.h_m;
      console.log(`wide dam step 40: ${JSON.stringify(counts(s.header, 40))}; fluid to brick x ${fluidX}, live F to ${liveX}, pages to ${pagesX} `
        + `(of ${Math.floor(o.lattice[0] / 8) + 1} bricks, ${o.lattice[0] / 32} pages); flow ${flow.toFixed(1)} cells`);
      assert.ok(8 * liveX <= 8 * (fluidX + 1) + Math.ceil(flow) + 9 + 8, `live F bricks reach x ${liveX}, the fluid ${fluidX} (flow ${flow})`);
      assert.ok(32 * pagesX <= 8 * (fluidX + 1) + Math.ceil(flow) + 20 + 32, `pages reach x page ${pagesX}, the fluid brick ${fluidX} (flow ${flow})`);
      // levelset-coarse's lcTileFresh: no dilated FT_DIST has its low 13 bits all ones (lInit's fresh word; lDilate
      // sets the support bit 12 only with an occupied distance of at most 3), so the sentinel names a fresh slot only.
      const F = fluidLayout(L, { tiered: true }), fcell = new Uint32Array(await wide.read("fcell"));
      let sentinel = 0;
      for (const slot of s.list(LIST.F_ALL)) for (let t = 0; t < 8; t++) if ((fcell[F.fcell.TILE + FT.WORDS * (8 * slot + t) + FT.DIST]! & 0x1fff) === 0x1fff) sentinel++;
      assert.equal(sentinel, 0, "a dilated FT_DIST never reads lInit's fresh sentinel");
    } finally { wide.destroy(); }
  });
});
