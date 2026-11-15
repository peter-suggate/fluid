// The real D on the engine (lane E, brick-region-heat.wgsl.ts): the tiered dam and a still reservoir on lane F's
// tiered stages and lane P's composite MGPCG.
//
// CPU: the head runs regionHeat and regionSpread between censusBegin and censusCoarse (which takes the head's deep),
// and the certificate (certCoarse, certFine, certError) after the transfer; without P's PDELTA the heat binds no pmgC.
//
// GPU (WEBGPU_NODE_MODULE):
// - the tiered dam, 40 steps: no latch; every resident region's tier is 0, 1 or 2, its heat finite in [0, 2], the
//   spread's COLD count at least the tier-0 regions after the claims (they only promote), and no deep region next to
//   a non-COLD one;
// - a still reservoir (the seep-like deep state: a box at rest, its floor regions 32 cells below the surface): deep
//   regions appear, the census keeps their tiles' CC_DECIDED and CC_STATIC words, and no region next to a non-COLD
//   one is deep;
// - save and restore: a restored engine's buffers equal the saved words, and the steps after a restore reproduce the
//   heat, tier and deep words of the same steps run straight on.
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { BrickEngine, type BrickEngineState } from "./brick-engine";
import { CENSUS, REGION_H, RG } from "./brick-census.wgsl";
import { brickRegionHeatSchedule } from "./brick-region-heat.wgsl";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import { brickSceneSetup, type BrickSceneSetup } from "./brick-scene";
import { snapshot } from "./brick-readback";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const f32 = (w: number) => new Float32Array(new Uint32Array([w]).buffer)[0]!;

test("region heat: without P's PDELTA the heat binds no pmgC; the head and certificate placements", () => {
  const { setup } = damSetup();
  const L = new BrickLayout({ lattice: setup.lattice, ...setup.capacities });
  const s = brickRegionHeatSchedule(L, {});
  assert.deepEqual(s.head.map(p => p.kernel.label), ["regionHeat", "regionSpread"]);
  assert.deepEqual(s.cert.map(p => p.kernel.label), ["certCoarse", "certFine", "certError"]);
  assert.ok(!s.kernels.regionHeat.bindings.some(b => b.name === "pmgC"));
  const withP = brickRegionHeatSchedule(L, { pdelta: 0 });
  assert.ok(withP.kernels.regionHeat.bindings.some(b => b.name === "pmgC"));
});

function damSetup(): { setup: BrickSceneSetup; values: Record<string, unknown> } {
  const METHOD = "uniform-narrow-band-flip";
  const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  (scene.fluid as { brickPools: { f: number } }).brickPools = { ...scene.fluid.brickPools!, f: 4913 };
  const { setup, gaps } = brickSceneSetup(scene, values);
  assert.ok(setup, gaps.join("; "));
  return { setup, values: values as Record<string, unknown> };
}

async function engineFor(device: GPUDevice, setup: BrickSceneSetup, values: Record<string, unknown>) {
  return BrickEngine.create(device, {
    lattice: setup.lattice, h_m: setup.h_m, dt_s: setup.dt_s, gravity_mps2: setup.gravity_mps2, capacities: setup.capacities, liquid: setup.liquid,
    solver: compositeMGPCG(), stages: fluidStagesAt(true), values, label: setup.sceneId,
    fluid: { density_kg_m3: setup.density_kg_m3, viscosity_Pa_s: setup.viscosity_Pa_s, surfaceTension_N_m: setup.surfaceTension_N_m },
  });
}

/** Every resident region's words (page key, octant) inside the lattice. Pages beyond it (the reach ring across a
 * closed wall) hold no lattice tile: their regions are none of the heat's (rgOf: NB_NONE), as the old lattice's. */
async function regions(e: BrickEngine) {
  const s = await snapshot(e), census = await e.read("census"), out = new Map<string, { heat: number; tier: number; deep: number; bits: number; pending: number; words: number[] }>();
  const N = e.layout.lattice;
  for (const [key, cs] of s.pages) for (let r = 0; r < 8; r++) {
    const R = [2 * (key & 1023) + (r & 1), 2 * ((key >>> 10) & 1023) + ((r >> 1) & 1), 2 * ((key >>> 20) & 1023) + ((r >> 2) & 1)];
    if (R.some((v, a) => 16 * v >= N[a]!)) continue;
    const at = CENSUS.STRIDE * cs + CENSUS.REG + RG.WORDS * r;
    out.set(`${key}.${r}`, { heat: f32(census[at + RG.HEAT]!), tier: census[at + RG.TIER]!, deep: census[at + RG.DEEP]!, bits: census[at + RG.BITS]!, pending: census[at + RG.PENDING]!, words: [...census.subarray(at, at + RG.WORDS)] });
  }
  return { s, census, regions: out };
}

/** The deep rule after the claims (certError's deep pass, the last writer): a deep region and its 27 stored
 * neighbours are tier 0. */
function checkDeep(rs: Map<string, { tier: number; deep: number; words: number[] }>) {
  const coord = (key: number, r: number) => [2 * (key & 1023) + (r & 1), 2 * ((key >>> 10) & 1023) + ((r >> 1) & 1), 2 * ((key >>> 20) & 1023) + ((r >> 2) & 1)];
  const byCoord = new Map<string, { tier: number; words: number[]; id: string }>();
  for (const [id, w] of rs) { const [k, r] = id.split(".").map(Number); byCoord.set(coord(k!, r!).join(","), { tier: w.tier, words: w.words, id }); }
  for (const [id, w] of rs) {
    if (w.deep !== 1) continue;
    const [k, r] = id.split(".").map(Number), c = coord(k!, r!);
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const n = byCoord.get([c[0]! + dx, c[1]! + dy, c[2]! + dz].join(","));
      assert.ok(n === undefined || n.tier === 0, `deep region ${id} [${w.words.map(x => x.toString(16))}] has a tier ${n?.tier} neighbour ${n?.id} (${dx},${dy},${dz}) [${n?.words.map(x => x.toString(16))}]`);
    }
  }
}

test("region heat (GPU): the tiered dam runs 40 steps without a latch, every region's tier and heat in range", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("region heat dam", async device => {
    const { setup, values } = damSetup();
    const e = await engineFor(device, setup, values);
    const tally: string[] = [];
    for (let step = 0; step < 40; step += 10) {
      e.advance(10); await e.assertHealthy();
      const { s, census, regions: rs } = await regions(e), h = s.header;
      const tiers = [0, 0, 0];
      for (const [id, w] of rs) {
        assert.ok(w.tier <= 2, `${id}: tier ${w.tier}`);
        assert.ok(Number.isFinite(w.heat) && w.heat >= 0 && w.heat <= 2, `${id}: heat ${w.heat}`);
        tiers[w.tier]!++;
      }
      // HEAT_COLD is the spread's count: the claims and the transfer promote regions after it, never cool them.
      assert.ok(h[REGION_H.HEAT_COLD]! >= tiers[0]!, `HEAT_COLD ${h[REGION_H.HEAT_COLD]} below the ${tiers[0]} tier-0 regions`);
      checkDeep(rs);
      assert.equal(h[REGION_H.HEAT_FORCED], 1);
      // The deep tiles' kept decisions: band (DEC set) or not.
      let deepTiles = 0, deepBand = 0;
      for (const [id, w] of rs) {
        if (w.deep !== 1) continue;
        const [k, r] = id.split(".").map(Number), cs = s.pages.get(k!)!;
        for (let i = 0; i < 512; i++) {
          if ((((i >> 2) & 1) | (((i >> 5) & 1) << 1) | (((i >> 8) & 1) << 2)) !== r) continue;
          deepTiles++; if ((census[CENSUS.STRIDE * cs + CENSUS.DEC + (i >> 5)]! >>> (i & 31)) & 1) deepBand++;
        }
      }
      tally.push(`step ${step + 10}: ${s.bricks.size} F, ${deepTiles} deep tiles (${deepBand} kept band), ${rs.size} regions, tiers ${tiers.join("/")}, deep ${h[REGION_H.HEAT_DEEP]}, extent ${h[REGION_H.STILL_EXTENT]}, travel ${f32(h[REGION_H.STILL_TRAVEL]!).toExponential(2)}`);
    }
    console.log(`region heat dam: ${tally.join("; ")}`);
    e.destroy();
  });
});

test("region heat (GPU): a still reservoir goes deep, the census keeps its decided words, and save/restore reproduces the heat", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("region heat reservoir", async device => {
    const { setup: dam, values } = damSetup();
    // A box at rest across the dam's lattice, its surface at 40 cells: the floor's regions are deep once the surface cools.
    const N = dam.lattice;
    const setup: BrickSceneSetup = { ...dam, liquid: [{ lo: [-8, -8, -8], hi: [N[0] + 8, 40, N[2] + 8] }] };
    const e = await engineFor(device, setup, values);
    e.advance(50); await e.assertHealthy();
    const a = await regions(e);
    const deep = [...a.regions.entries()].filter(([, w]) => w.deep === 1);
    console.log(`region heat reservoir: ${a.regions.size} regions, deep ${deep.length} (H ${a.s.header[REGION_H.HEAT_DEEP]}), cold ${a.s.header[REGION_H.HEAT_COLD]}`);
    assert.ok(deep.length > 0, "a still reservoir has deep regions");
    assert.equal(a.s.header[REGION_H.HEAT_DEEP], deep.length, "HEAT_DEEP counts the lattice's deep regions (no claims since the head)");
    checkDeep(a.regions);
    // The census keeps a deep tile's decided words: one more head with the band re-decided must leave them.
    const before = await e.read("census");
    e.advance(1); await e.assertHealthy();
    const after = await e.read("census"), s1 = await snapshot(e);
    let kept = 0, checked = 0;
    for (const [id] of deep) {
      const [k, r] = id.split(".").map(Number), cs = s1.pages.get(k!);
      if (cs === undefined) continue;
      const at = CENSUS.STRIDE * cs + CENSUS.REG + RG.WORDS * r!;
      if (after[at + RG.DEEP] !== 1 || before[at + RG.DEEP] !== 1) continue;
      for (let lane = 0; lane < 2; lane++) {
        const d = CENSUS.STRIDE * cs + CENSUS.DEC + 2 * r! + lane, st = CENSUS.STRIDE * cs + CENSUS.STAT + 2 * r! + lane;
        checked++; if (after[d] === before[d] && after[st] === before[st]) kept++;
      }
    }
    console.log(`region heat reservoir: kept ${kept} of ${checked} deep decided words`);
    assert.ok(checked > 0); assert.equal(kept, checked);
    // Save and restore.
    const saved: BrickEngineState = await e.save();
    e.advance(6); await e.assertHealthy();
    const straight = await regions(e);
    await e.restore(saved);
    for (const name of ["census", "set", "dir", "vel", "phi"]) assert.deepEqual(await e.read(name), saved.buffers.get(name), `restored ${name}`);
    assert.equal(e.stepsSubmitted, saved.steps);
    e.advance(6); await e.assertHealthy();
    const replay = await regions(e);
    let tierDiff = 0, heatDiff = 0;
    for (const [id, w] of straight.regions) {
      const v = replay.regions.get(id);
      assert.ok(v, `region ${id} after the restore`);
      if (v.tier !== w.tier || v.deep !== w.deep) tierDiff++;
      if (v.heat !== w.heat) heatDiff++;
    }
    console.log(`region heat save/restore: ${straight.regions.size} regions, tier/deep differ ${tierDiff}, heat differs ${heatDiff}`);
    assert.equal(replay.regions.size, straight.regions.size);
    assert.equal(tierDiff, 0); assert.equal(heatDiff, 0);
    e.destroy();
  });
});
