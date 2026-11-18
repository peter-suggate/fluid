// LANE E OWNS THIS FILE. The census's static need rule (brick-static-need.wgsl.ts).
//
// CPU: the rule's text is emitted only with the static mask, tiered, and only then does the census drop its
// static-K latches (censusBegin's LIVE_K, censusFine's 0x63000004). The TRIG-only census never promotes.
//
// GPU: a still tank (32 x 64 x 32 cells, tiered, liquid y < 17) under two solid blocks over x, z 8..16: a shelf at
// y 19..23, just over the surface, whose cut bricks hold the surface band; and a block at y 40..48, dry and far above it.
// Then a deep tank (liquid y < 48) over a block on its floor (y 0..4, x, z 20..28), 44 cells under the surface: the
// boot's census runs before its setMark, so that brick is wet by its F vertices alone (alpine-lake's ground and
// sphere-jet's padded corner, 2026-11-18: their tiles went 4h at the boot's commit and the kick's solve latched).
// Neither meets the liquid: a liquid row in a static solid cell latches 0xa2 in the operator until phiSolid and pSolid
// keep liquid out of solid cells (E, next). 30 steps (under the 60-step probe). Each step must hold:
// - no FAIL latch (before the rule, the census latched any tiered static cut, 0x63000000 / 0x63000004);
// - every tile of a resident cut brick that is wet is h-owned (the operator's need: the old band certificate);
// - a cut brick 16 or more cells above the surface is never h (the rule is liquid-conditional, as the old one is:
//   a dry cut brick far from liquid stays 4h or is not resident).
// The rule must promote something (its count) at least once.
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { solidVoxelsOnMovedLattice } from "../../core/scene-lattice";
import { BrickLayout, CMASK_WORDS, F_CLASS, H, LIST, NONE, REC, SEAM, brickKeyCoord } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions, type Vec3 } from "./brick-engine";
import { brickCensusKernels, brickCensusSchedule, brickTrigCensusSchedule } from "./brick-census.wgsl";
import { brickInitialLiquid, brickInitialVelocity, bootParticleCount } from "./brick-initial";
import { brickStaticMask, brickStaticMaskWalls } from "./brick-static.wgsl";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import { snapshot } from "./brick-readback";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const METHOD = "uniform-narrow-band-flip";

test("brick static need: the rule's kernels and the census's static latches follow the mask, tiered", () => {
  const L = new BrickLayout({ lattice: [64, 64, 64], f: 64, c: 8, b: 16, k: 16, particles: 1024 }), walls = brickStaticMaskWalls([64, 64, 64], false);
  const g1 = (k: number) => ({ indirect: k });
  const labels = (s: { kernel: { label: string } }[]) => s.map(p => p.kernel.label);
  const plain = brickCensusKernels(L, { tiered: true }), statics = brickCensusKernels(L, { tiered: true, statics: walls });
  assert.equal("censusStaticActive" in plain, false);
  assert.ok(statics.censusStaticActive && statics.censusStaticPromote);
  assert.equal("censusStaticActive" in brickCensusKernels(L, { tiered: false, statics: walls }), false, "all-h: no rule (every tile is h)");
  // The static-K latches go with the rule, and only then.
  assert.match(plain.censusBegin.code, /nbSet\[NB_H_LIVE_K\]/); assert.doesNotMatch(statics.censusBegin.code, /nbSet\[NB_H_LIVE_K\]/);
  const bodies = { frame_m: [0, 0, 0] as Vec3, h_m: 1 / 32 };
  assert.match(brickCensusKernels(L, { tiered: true, bodies }).censusFine.code, /0x63000004u/);
  assert.doesNotMatch(brickCensusKernels(L, { tiered: true, bodies, statics: walls }).censusFine.code, /0x63000004u/);
  // After decide, in order; the TRIG-only census has none.
  const s = labels(brickCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, true, true, false, undefined, undefined, walls));
  assert.deepEqual(s.slice(-3), ["censusDecide", "censusStaticActive", "censusStaticPromote"]);
  assert.ok(!labels(brickCensusSchedule(L, g1)).includes("censusStaticActive"));
  const trig = brickTrigCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, true, undefined, undefined, walls);
  assert.ok(!labels(trig).some(l => l.startsWith("censusStatic")), "the TRIG-only census promotes none");
  assert.doesNotMatch(trig[0]!.kernel.code, /nbSet\[NB_H_LIVE_K\]/, "but its censusBegin drops the static-K latch (the boot's cut K slots)");
  // Without the mask the census's text is unchanged.
  assert.equal(brickCensusKernels(L, { tiered: true }).censusDecide.code, brickCensusKernels(L, { tiered: true, statics: undefined }).censusDecide.code);
});

/** The still tank with the shelf and the block: the engine's options built directly (the scene's voxel gap aside: this is the rule's
 * own regression, ahead of pSolid and phiSolid). The old engine's t = 0 φ excludes the solids (brickInitialPhi). */
const SHELF_SOLIDS = [{ lo: [8, 19, 8] as Vec3, hi: [16, 23, 16] as Vec3 }, { lo: [8, 40, 8] as Vec3, hi: [16, 48, 16] as Vec3 }];
function shelfTank(surface = 17, solids = SHELF_SOLIDS): { options: BrickEngineOptions; surface: number; solids: { lo: Vec3; hi: Vec3 }[] } {
  const definition = getSceneDefinition("minimal-power-dam-break-32"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
  const h = 0.025;
  scene.container = { ...scene.container, width_m: 32 * h, height_m: 64 * h, depth_m: 32 * h, fillFraction: 0 };
  scene.voxelDomain.finestCellSize_m = h; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
  scene.solidVoxels = [...scene.solidVoxels, ...solids.map(b => ({ operation: "fill" as const, minimum: [b.lo[0], b.lo[1], b.lo[2]] as [number, number, number], maximumExclusive: [b.hi[0], b.hi[1], b.hi[2]] as [number, number, number] }))];
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -16 * h, y: 0, z: -16 * h }, max_m: { x: 16 * h, y: surface * h, z: 16 * h } }];
  const method = getMethod(METHOD), profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const lattice: Vec3 = [32, 64, 32], g = scene.fluid.gravity_m_s2, band = 3;
  const o = { lattice, h_m: h, dt_s: 1 / 60, gravity_mps2: [g.x, g.y, g.z] as Vec3 };
  const start = brickInitialLiquid(scene, { ...o, farPhi: 64 });
  assert.ok(start.initial, "the solids force the old engine's φ table");
  const mask = brickStaticMask(scene);
  assert.ok(mask, "the solids are a static mask");
  const seeds = bootParticleCount({ lattice, farPhi: 64, liquid: start.liquid, initial: start.initial }, start.bricks, band);
  return {
    surface, solids,
    options: {
      ...o, farPhi: 64, capacities: { f: 4 * 8 * 4 * 4, c: 2 * 2 * 2 * 4, b: 16, k: 64, particles: Math.max(3 * seeds, 65536) }, liquid: start.liquid, initial: start.initial,
      initialVelocity: brickInitialVelocity(scene, lattice), solver: compositeMGPCG(), stages: fluidStagesAt(true), values, particles: { bandDepth: band },
      fluid: { density_kg_m3: scene.fluid.density_kg_m3, viscosity_Pa_s: scene.fluid.dynamicViscosity_Pa_s, surfaceTension_N_m: scene.fluid.surfaceTension_N_m },
      staticMask: { words: mask.words, walls: brickStaticMaskWalls(lattice, false) }, label: "static-need shelf",
    },
  };
}

for (const [name, surface, solids] of [["a still tank under a solid shelf", 17, SHELF_SOLIDS], ["a deep tank over a floor block", 48, [{ lo: [20, 0, 20] as Vec3, hi: [28, 4, 28] as Vec3 }]]] as const)
test(`brick static need (GPU): ${name} keeps its wet cut bricks h from the boot and a dry block's 4h, no latch, 30 steps`, GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { options } = shelfTank(surface, [...solids]);
  await withUniformDevice("brick static need", async device => {
    const e = await BrickEngine.create(device, options), L = e.layout;
    { const s0 = await snapshot(e); assert.equal(s0.header[H.FAIL], 0, `boot: FAIL ${s0.header[H.FAIL]} info ${s0.header[H.FAIL_INFO]?.toString(16)}`); t.diagnostic(`boot: ${s0.bricks.size} F bricks, live K ${s0.header[H.LIVE_K]}`); }
    const cut = (b: Vec3) => {
      const inCol = (c: readonly number[]) => solids.some(b => c.every((v, a) => v >= b.lo[a]! && v < b.hi[a]!));
      for (let l = 0; l < 512; l++) {
        const c = [8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)];
        if (inCol(c) || [0, 1, 2].some(a => (c[a] === 8 * b[a]!) && inCol(c.map((v, k) => v - (k === a ? 1 : 0))))) return true;
      }
      return false;
    };
    let promotedEver = 0, wetCutChecked = 0;
    try {
      for (let step = 0; step <= 30; step++) {
        if (step > 0) e.advance(1);
        const s = await snapshot(e), need = await e.read("staticNeed");
        assert.equal(s.header[H.FAIL], 0, `step ${step}: FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]?.toString(16)}, live K ${s.header[H.LIVE_K]}, promoted ${need[4 * L.cap.c]}`);
        const promoted = need[4 * L.cap.c]!; promotedEver = Math.max(promotedEver, promoted);
        let wet = 0, dryHigh = 0;
        for (const [key, slot] of s.bricks) {
          const b = brickKeyCoord(key) as Vec3, cls = s.dir[L.dir.frec + 32 * slot + REC.CLASS]!;
          if (!cut(b)) continue;
          const cs = s.dir[L.dir.page + (b[0] >> 2) + (L.lattice[0] / 32) * ((b[1] >> 2) + (L.lattice[1] / 32) * (b[2] >> 2))]!;
          assert.ok(cs < SEAM && cs !== NONE, `brick ${b}: its page is resident`);
          const cm = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
          const tiles = [0, 1, 2, 3, 4, 5, 6, 7].map(j => { const T = [2 * b[0] + (j & 1), 2 * b[1] + ((j >> 1) & 1), 2 * b[2] + (j >> 2)].map(v => v & 7); const i = T[0]! | (T[1]! << 3) | (T[2]! << 6); return ((cm[i >> 5]! >>> (i & 31)) & 1) !== 0; });
          if ((cls & (F_CLASS.LIQUID | F_CLASS.PARTICLES)) !== 0) { wet++; assert.ok(tiles.every(x => x), `step ${step}: wet cut brick ${b} has a 4h tile (${tiles})`); }
          if (8 * b[1] >= surface + 16) { dryHigh++; assert.ok(!tiles.some(x => x), `step ${step}: dry cut brick ${b}, ${8 * b[1] - surface} cells above the surface, is h`); }
        }
        wetCutChecked += wet;
        if (step % 10 === 0) t.diagnostic(`step ${step}: promoted ${promoted} tiles, wet cut bricks ${wet}, dry high cut bricks resident ${dryHigh}, live F ${s.header[H.LIVE_F]} K ${s.header[H.LIVE_K]}`);
      }
      assert.ok(promotedEver > 0, "the rule promoted tiles");
      assert.ok(wetCutChecked > 0, "wet cut bricks were checked");
    } finally { e.destroy(); }
  }, ["subgroups"]);
});
