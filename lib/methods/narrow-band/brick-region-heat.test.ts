// The real D on the engine (lane E, brick-region-heat.wgsl.ts): the tiered dam and a still reservoir on lane F's
// tiered stages and lane P's composite MGPCG.
//
// CPU: the head runs regionHeat and regionSpread between censusBegin and regionDeep (the head's deep and replay mode),
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
//   heat, tier and deep words of the same steps run straight on;
// - replay on the reservoir: under gravity every deep region holds its stamp and is demoted (the projection's
//   residual moves its faces' bits each step); without gravity (velocity exactly at rest) deep regions replay
//   (censusFine and censusCoarse skip their workgroups) with their stored words standing, a fresh classification of
//   the same state has the same keys and WANT, a stored key moved (an h tile's, a 4h tile's) demotes its region,
//   which takes the key afresh, and a restored step replays the same.
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout, LIST } from "./brick-abi";
import { BrickEngine, type BrickEngineState } from "./brick-engine";
import { CENSUS, REGION_H, RG, brickCensusSchedule } from "./brick-census.wgsl";
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
  const g1 = (k: number) => ({ indirect: k });
  assert.deepEqual(brickCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, true, true, true).slice(0, 4).map(p => [p.kernel.label, p.w ?? 0]),
    [["censusBegin", 0], ["regionDeep", 0], ["censusFine", 1], ["censusCoarse", 1]]);
  assert.ok(!brickCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, true, true, false).some(p => p.kernel.label === "regionDeep"));
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

/** The certificate's packed copy (CENSUS.DEF, certError's tileErrorOf), on every certified region (RG_CKEY its page
 * key + 1; a page claimed after the last certificate has none): each lattice tile carries the page key + 1 and its TC
 * width (f32); its deformation is 0 with error 0 (a reader of class 0) or at least its span (max(4h gradient, span)),
 * and saturated where the error is; the region's error is the max of its tiles'. */
function checkCertificateCopy(s: Awaited<ReturnType<typeof snapshot>>, census: Uint32Array, N: readonly number[]) {
  const SAT = new Float32Array([3.402823e38])[0]!, bits = (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0]!;
  const tally = { tiles: 0, cleared: 0, saturated: 0, regions: 0 };
  for (const [key, cs] of s.pages) for (let r = 0; r < 8; r++) {
    const at = CENSUS.STRIDE * cs + CENSUS.REG + RG.WORDS * r;
    if (census[at + RG.CKEY] !== key + 1) continue;
    tally.regions++;
    let errorMax = 0;
    for (let l = 0; l < 64; l++) {
      const i = ((l & 3) + 4 * (r & 1)) | ((((l >> 2) & 3) + 4 * ((r >> 1) & 1)) << 3) | (((l >> 4) + 4 * (r >> 2)) << 6);
      const t = [8 * (key & 1023) + (i & 7), 8 * ((key >>> 10) & 1023) + ((i >> 3) & 7), 8 * ((key >>> 20) & 1023) + (i >> 6)];
      if (t.some((v, a) => 4 * v >= N[a]!)) continue;
      const d = CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i, o = CENSUS.STRIDE * cs + CENSUS.TC + 16 * i;
      const deformation = f32(census[d]!), error = f32(census[d + 1]!), span = f32(census[o + 6]!), id = `tile ${t} (page ${key}.${r})`;
      assert.equal(census[d + 2], key + 1, `${id}: DEF's key`);
      assert.equal(census[d + 3], bits(census[o + 5]!), `${id}: DEF's width is TC's (${census[o + 5]})`);
      assert.ok((deformation === 0 && error === 0) || deformation >= span, `${id}: deformation ${deformation} below the span ${span}`);
      if (!(error < 3e38)) assert.equal(deformation, SAT, `${id}: error ${error} saturates the deformation`);
      tally.tiles++; if (deformation === 0 && error === 0) tally.cleared++; if (deformation === SAT) tally.saturated++;
      errorMax = Math.max(errorMax, census[d + 1]!);
    }
    assert.equal(errorMax, census[at + RG.ERROR], `page ${key}.${r}: the region's error is its tiles' max`);
  }
  assert.ok(tally.regions > 0 && tally.tiles > 0, "certified tiles");
  return tally;
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
      const def = checkCertificateCopy(s, census, e.layout.lattice);
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
      tally.push(`step ${step + 10}: DEF ${JSON.stringify(def)}, ${s.bricks.size} F, ${deepTiles} deep tiles (${deepBand} kept band), ${rs.size} regions, tiers ${tiers.join("/")}, deep ${h[REGION_H.HEAT_DEEP]}, extent ${h[REGION_H.STILL_EXTENT]}, travel ${f32(h[REGION_H.STILL_TRAVEL]!).toExponential(2)}`);
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

test("census replay (GPU): under gravity a still reservoir's deep regions hold their stamps and demote on the velocity's moved keys; without it they replay their stored words, a moved key demotes its region, and save/restore carries the mode", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("census replay reservoir", async device => {
   for (const g of [9.81, 0]) {
    const { setup: dam, values } = damSetup();
    const N = dam.lattice;
    const setup: BrickSceneSetup = { ...dam, gravity_mps2: g ? dam.gravity_mps2 : [0, 0, 0], liquid: [{ lo: [-8, -8, -8], hi: [N[0] + 8, 40, N[2] + 8] }] };
    const e = await engineFor(device, setup, values);
    e.advance(50); await e.assertHealthy();
    const saved: BrickEngineState = await e.save(), census0 = saved.buffers.get("census")!;
    const S = CENSUS.STRIDE, reg = (cs: number, r: number) => S * cs + CENSUS.REG + RG.WORDS * r;
    const tileOf = (r: number, l: number) => ((l & 3) + 4 * (r & 1)) | ((((l >> 2) & 3) + 4 * ((r >> 1) & 1)) << 3) | (((l >> 4) + 4 * (r >> 2)) << 6);
    const runFrom = async (state: BrickEngineState) => {
      await e.restore(state); e.advance(1); await e.assertHealthy();
      return { census: await e.read("census"), s: await snapshot(e) };
    };
    const wantOf = (s: Awaited<ReturnType<typeof snapshot>>) => [...s.pages.values()].sort((a, b) => a - b).map(cs => [...s.set.subarray(e.layout.set.want + 16 * cs, e.layout.set.want + 16 * cs + 16)]);
    const cmask = (cs: number, i: number, dir: Uint32Array) => (dir[e.layout.dir.cmask + 16 * cs + (i >> 5)]! >>> (i & 31)) & 1;
    const A = await runFrom(saved);
    // A region's coordinate (16-cell units) and whether it holds a lattice tile (regions(): a page past the lattice
    // holds none, rgInLattice).
    const keyOf = new Map([...A.s.pages].map(([key, cs]) => [cs, key]));
    const rgCoord = (cs: number, r: number) => { const key = keyOf.get(cs)!; return [2 * (key & 1023) + (r & 1), 2 * ((key >>> 10) & 1023) + ((r >> 1) & 1), 2 * ((key >>> 20) & 1023) + ((r >> 2) & 1)]; };
    const inLattice = (cs: number, r: number) => rgCoord(cs, r).every((v, a) => 16 * v < N[a]!);
    const replayed: [number, number][] = [];
    let hTiles = 0, slabs = 0;
    for (const cs of A.s.pages.values()) {
      for (let r = 0; r < 8; r++) if (A.census[reg(cs, r) + RG.MODE] === 1) {
        replayed.push([cs, r]);
        for (let l = 0; l < 64; l++) hTiles += cmask(cs, tileOf(r, l), A.s.dir);
      }
      for (let z = 0; z < 8; z++) if ([0, 1, 2, 3].every(q => A.census[reg(cs, 4 * (z >> 2) + q) + RG.MODE] === 1)) slabs++;
    }
    console.log(`census replay, g = ${g}: ${replayed.length} replayed regions of ${8 * A.s.pages.size} (deep ${A.s.header[REGION_H.HEAT_DEEP]}); censusFine skips ${hTiles} h-tile workgroups, censusCoarse ${slabs} slab workgroups`);
    // Why a deep region classifies: its stamp broke (deep epoch, page, h bits or pages), or a key moved (the
    // velocity's bits since the last classification; the largest move in m/s).
    const ov = (k: number) => { const b = k & 0x80000000 ? k & 0x7fffffff : ~k >>> 0; return f32(b); };
    let deepRegions = 0, stampHeld = 0, keyMoved = 0, moveMax = 0;
    for (const cs of A.s.pages.values()) for (let r = 0; r < 8; r++) {
      const at = reg(cs, r);
      if (!inLattice(cs, r) || A.census[at + RG.DEEP] !== 1 || A.census[at + RG.MODE] === 1) continue;
      deepRegions++;
      if (census0[at + RG.DEEP] === 1 && census0[at + RG.STAMP] === census0[at + RG.DEEP_EPOCH]! + 1 && A.census[at + RG.DEEP_EPOCH] === census0[at + RG.DEEP_EPOCH]
        && [RG.GEN, RG.OWN, RG.OWN + 1, RG.PAGES].every(w => A.census[at + w] === census0[at + w])) stampHeld++;
      let moved = false;
      for (let l = 0; l < 64; l++) for (let k = 0; k < 6; k++) {
        const w = S * cs + CENSUS.KEYS + 6 * tileOf(r, l) + k, a = census0[w]!, b = A.census[w]!;
        if (a !== b) { moved = true; const d = Math.abs(ov(a) - ov(b)); if (Number.isFinite(d)) moveMax = Math.max(moveMax, d); }
      }
      if (moved) keyMoved++;
    }
    console.log(`census replay, g = ${g}: ${deepRegions} deep regions classify: stamp held ${stampHeld}, a key moved ${keyMoved} (largest move ${moveMax.toExponential(2)} m/s)`);
    if (g) {
      // The projection's residual moves every submerged face's bits each step: every deep region in the reservoir is
      // demoted. The still air the residency keeps above it (the page sources' 20 cells past the 4h liquid,
      // brick-residency.wgsl.ts) has no moving key, so a deep region there replays: every replayed region lies wholly
      // above the surface.
      const surface = setup.liquid[0]!.hi[1]!;
      assert.ok(deepRegions > 0); assert.equal(stampHeld, deepRegions); assert.equal(keyMoved, deepRegions);
      for (const [cs, r] of replayed) assert.ok(16 * rgCoord(cs, r)[1]! >= surface, `replayed region ${rgCoord(cs, r)} is below the surface (${surface})`);
      e.destroy(); continue;
    }
    assert.ok(replayed.length > 0, "a still reservoir replays");
    // A replayed tile's stored words stand, and it publishes RCROSS and RGAP's words before the importance.
    for (const [cs, r] of replayed) {
      assert.equal(A.census[reg(cs, r) + RG.DEEP], 1);
      for (let l = 0; l < 64; l++) {
        const i = tileOf(r, l), o = S * cs;
        for (const [w, n] of [[CENSUS.KEYS, 6], [CENSUS.W0, 1], [CENSUS.W1, 1], [CENSUS.NEAR, 1], [CENSUS.RCROSS, 2], [CENSUS.RGAP, 1]] as const)
          for (let k = 0; k < n; k++) assert.equal(A.census[o + w + n * i + k], census0[o + w + n * i + k], `replayed tile ${cs}.${i} word ${w}`);
      }
    }
    // Fresh: every stamp cleared, every region classifies. The keys are the replay's; WANT is the same.
    const fresh = new Uint32Array(census0);
    for (const cs of A.s.pages.values()) for (let r = 0; r < 8; r++) fresh[reg(cs, r) + RG.STAMP] = 0;
    const B = await runFrom({ ...saved, buffers: new Map([...saved.buffers, ["census", fresh]]) });
    let differ = 0, words = 0;
    for (const [cs, r] of replayed) {
      assert.equal(B.census[reg(cs, r) + RG.MODE], 2, "a cleared stamp classifies");
      for (let l = 0; l < 64; l++) {
        const i = tileOf(r, l), o = S * cs;
        for (let k = 0; k < 6; k++) assert.equal(B.census[o + CENSUS.KEYS + 6 * i + k], A.census[o + CENSUS.KEYS + 6 * i + k], `fresh keys of tile ${cs}.${i}`);
        for (const [w, n] of [[CENSUS.W0, 1], [CENSUS.W1, 1], [CENSUS.FIN, 1], [CENSUS.CROSS, 2], [CENSUS.GAP, 1], [CENSUS.NEAR, 1]] as const)
          for (let k = 0; k < n; k++) { words++; if (B.census[o + w + n * i + k] !== A.census[o + w + n * i + k]) differ++; }
      }
    }
    console.log(`census replay: against a fresh classification, ${differ} of ${words} replayed record words differ`);
    assert.deepEqual(wantOf(B.s), wantOf(A.s), "WANT with replay is the fresh classification's");
    // Demotion: a replayed region with one tile's stored key moved (an h tile's when one replays, and a 4h tile's)
    // classifies, and its keys are fresh again; the other replayed regions replay.
    const pick = (h: number) => {
      for (const [cs, r] of replayed) for (let l = 0; l < 64; l++) if (cmask(cs, tileOf(r, l), A.s.dir) === h) return { cs, r, i: tileOf(r, l) };
      return undefined;
    };
    const moved = [pick(0), pick(1)].filter(p => p !== undefined).filter((p, k, a) => a.findIndex(q => q.cs === p.cs && q.r === p.r) === k);
    const bent = new Uint32Array(census0);
    for (const p of moved) bent[S * p.cs + CENSUS.KEYS + 6 * p.i] ^= 1;
    const D = await runFrom({ ...saved, buffers: new Map([...saved.buffers, ["census", bent]]) });
    for (const p of moved) {
      assert.equal(D.census[reg(p.cs, p.r) + RG.MODE], 2, `the moved key's region ${p.cs}.${p.r} (${cmask(p.cs, p.i, A.s.dir) ? "h" : "4h"}) is demoted`);
      assert.equal(D.census[S * p.cs + CENSUS.KEYS + 6 * p.i], A.census[S * p.cs + CENSUS.KEYS + 6 * p.i], "its classification takes the key afresh");
    }
    const others = replayed.filter(([cs, r]) => !moved.some(p => p.cs === cs && p.r === r));
    assert.ok(others.every(([cs, r]) => D.census[reg(cs, r) + RG.MODE] === 1), "the other regions replay");
    console.log(`census replay: demoted ${moved.map(p => `${p.cs}.${p.r} (${cmask(p.cs, p.i, A.s.dir) ? "h" : "4h"})`).join(", ")}; ${others.length} others replay`);
    // Save/restore across replay: the restored step replays the same regions with the same words and WANT.
    const C = await runFrom(saved);
    for (const cs of A.s.pages.values()) for (let r = 0; r < 8; r++) assert.equal(C.census[reg(cs, r) + RG.MODE], A.census[reg(cs, r) + RG.MODE]);
    for (const [cs, r] of replayed) for (let l = 0; l < 64; l++) {
      const i = tileOf(r, l), o = S * cs;
      for (const [w, n] of [[CENSUS.KEYS, 6], [CENSUS.W1, 1], [CENSUS.FIN, 1], [CENSUS.CROSS, 2], [CENSUS.GAP, 1]] as const)
        for (let k = 0; k < n; k++) assert.equal(C.census[o + w + n * i + k], A.census[o + w + n * i + k]);
    }
    assert.deepEqual(wantOf(C.s), wantOf(A.s));
    e.destroy();
   }
  });
});
