// LANE E OWNS THIS FILE. pSolid and phiSolid (brick-solid-walk.wgsl.ts): the static solid in the move and the level set.
//
// CPU: the text transforms apply to F's kernels exactly where the statics are bound, and leave them unchanged
// otherwise. pMove defers and stays inside the ring; pSolid carries the walk, the start's retirement and the split's
// test; pSeed refuses a blocked site; lAdvect skips NEAR_SOLID bricks; phiSolid runs over F_SOLID with the walk and the
// burial. pSolid runs after pMove, and phiSolid after lAdvect. setMark keeps NEAR_SOLID, the head's stepEnd clears the
// queue, and solidStamp marks NEAR_SOLID.
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { BrickLayout, LIST } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
import { validateBrickStages, type BrickStageContext, type BrickStages } from "./brick-stages";
import { brickSetKernels } from "./brick-set.wgsl";
import { brickSolidStampKernels } from "./brick-solid-stamp.wgsl";
import { brickStaticMaskWalls } from "./brick-static.wgsl";
import { ARGS } from "./brick-abi";
import "../index";

function context(statics?: boolean): BrickStageContext {
  const method = getMethod("uniform-narrow-band-flip");
  const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
  const layout = new BrickLayout({ lattice: [32, 32, 32], f: 200, c: 16, b: 16, k: 16, particles: 1 << 16 });
  return { layout, dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values, ...(statics === undefined ? {} : { statics }) } as BrickStageContext;
}
const placements = (s: BrickStages) => [...s.move, ...s.rebin, ...s.levelSet, ...s.velocity, ...s.tail, ...s.boot, ...(s.kick?.force ?? []), ...(s.kick?.after ?? []), ...(s.kick?.seed ?? [])];
const code = (s: BrickStages, label: string) => { const p = placements(s).find(p => p.kernel.label === label); assert.ok(p, label); return p.kernel.code; };

test("brick solid walk: pSolid and phiSolid join F's stage set only with statics, after pMove and lAdvect", () => {
  for (const tiered of [false, true]) {
    const plain = fluidStagesAt(tiered)(context()), off = fluidStagesAt(tiered)(context(false)), on = fluidStagesAt(tiered)(context(true));
    validateBrickStages(on);
    // Without statics, every kernel's text is the dam's.
    assert.deepEqual(placements(off).map(p => p.kernel.code), placements(plain).map(p => p.kernel.code), `tiered ${tiered}: statics false is the plain set`);
    assert.ok(!placements(plain).some(p => /pSolid|phiSolid|psClosed/.test(p.kernel.code)), "the plain set has no solid text");
    const move = on.move.map(p => p.kernel.label);
    assert.equal(move[move.indexOf("pMove") + 1], "pSolid", "pSolid follows pMove");
    const solid = on.move.find(p => p.kernel.label === "pSolid")!, pmove = on.move.find(p => p.kernel.label === "pMove")!;
    assert.deepEqual(solid.dispatch, pmove.dispatch, "pSolid dispatches over the records' arguments, as pMove");
    const ls = on.levelSet.map(p => p.kernel.label);
    assert.equal(ls[ls.indexOf("lAdvect") + 1], "phiSolid", "phiSolid follows lAdvect");
    assert.deepEqual(on.levelSet.find(p => p.kernel.label === "phiSolid")!.dispatch, { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 }, "phiSolid runs over F_ALL (its NEAR_SOLID bricks and the far vertices)");
    // Both bind pMove's buffers alone (8 storage buffers, the limit).
    assert.deepEqual(placements(on).find(p => p.kernel.label === "pSolid")!.kernel.bindings, placements(on).find(p => p.kernel.label === "pMove")!.kernel.bindings);
    // pMove skips a NEAR_SOLID brick's records and keeps the ring; pSolid takes exactly those, and walks them. The
    // boot's filing (cfg.z = 1) is pMove's alone.
    const m = code(on, "pMove"), s = code(on, "pSolid");
    const skip = " if(psNearSolid(fgSlot(pCellOf(p.position.xyz)))){return;}";
    assert.ok(m.includes(skip)); assert.ok(m.includes("+nbSampleRadius(m)>=8.0){fpart[FP_RANK+2u*i]=PS_FAR;return;}"), "a record that travels the ring is pSolid's");
    assert.ok(m.indexOf("if(cfg.z==1u){pFile(i,p,false);return;}") < m.indexOf(skip), "the boot rebin files before the skip");
    assert.ok(s.includes(" if(i>=n||cfg.z==1u){return;}\n if(fpart[FP_RANK+2u*i]!=PS_FAR){let p0=parts[pAt(src,i)];if(!nbLive(p0)||!psNearSolid(fgSlot(pCellOf(p0.position.xyz)))){return;}}\n fpart[FP_RANK+2u*i]=NB_NONE;"), "pSolid returns before it writes another record's rank, and clears PS_FAR");
    assert.ok(!s.includes(skip) && !s.includes("=PS_FAR;return;"));
    assert.doesNotMatch(m, /psWalk\(q/, "pMove does not inline the walk");
    assert.match(s, /fn pSolid\(/); assert.doesNotMatch(s, /fn pMove\(/);
    assert.match(s, /let endpoint=psWalk\(q,boxed,nbSampleRadius\(m\),&p\.velocity\);/);
    assert.match(s, /if\(psBlocked\(q\)\)\{return;\}/);
    assert.match(s, /&&!psBlocked\(nbfSplitA\(endpoint,m,along\)\)&&!psBlocked\(nbfSplitB\(endpoint,m,along\)\);/);
    assert.match(code(on, "pSeed"), /nbfSeedSiteOk\(distance,psBlocked\(site\),/);
    // lAdvect skips the NEAR_SOLID bricks; phiSolid walks the trace and keeps a buried vertex.
    const a = code(on, "lAdvect"), p = code(on, "phiSolid");
    assert.match(a, /if\(psNearSolid\(slot\)\)\{return;\}/); assert.doesNotMatch(a, /psWalkPhi\(foot/);
    assert.ok(a.includes(">=8.0){psFar=true;return 0.0;}") && a.includes("if(psFar){phi[NB_PHI_F_PHI1+NB_STRIDE*slot+l]=PS_PHI_FAR;continue;}"), "lAdvect leaves a trace that leaves the ring to phiSolid");
    assert.ok(p.includes("if(!near&&phi[NB_PHI_F_PHI1+NB_STRIDE*slot+l]!=PS_PHI_FAR){continue;}") && !p.includes("psFar=true"), "phiSolid takes the far vertices and walks them");
    assert.match(p, /fn phiSolid\(/); assert.match(p, /let near=psNearSolid\(slot\);/);
    assert.match(p, /let traced=psWalkPhi\(foot,nbfSurfaceTrace\(foot,dt,h,wall\)\);/);
    assert.match(p, /if\(lVertexIn\(v\)&&!psBuried\(v\)\)\{value=lAdvected\(v,dt,h\);\}/);
    assert.doesNotMatch(p, /psNearSolid\(slot\)\)\{return;\}/, "phiSolid advects its bricks");
  }
});

test("brick solid walk: setMark keeps NEAR_SOLID, solidStamp marks it", () => {
  const L = context().layout;
  const plain = brickSetKernels(L, { tiered: true }), on = brickSetKernels(L, { tiered: true, statics: true });
  assert.match(on.setMark.code, /&\(NB_CLASS_CUT_STATIC\|NB_CLASS_NEAR_SOLID\);/);
  assert.doesNotMatch(plain.setMark.code, /NB_CLASS_NEAR_SOLID\)/);
  assert.equal(on.stepEnd.code, plain.stepEnd.code);
  assert.equal(brickSetKernels(L, { tiered: true, statics: false }).setMark.code, plain.setMark.code);
  const stamp = brickSolidStampKernels(L, { walls: brickStaticMaskWalls([32, 32, 32], false) }).solidStamp.code;
  assert.match(stamp, /l<13824u/); assert.match(stamp, /atomicOr\(&dir\[r\+NB_REC_CLASS\],NB_CLASS_NEAR_SOLID\)/);
  assert.ok(stamp.indexOf("NB_CLASS_NEAR_SOLID") < stamp.indexOf("if(h.w!=0u){return;}"), "a body brick is marked before P's skip");
});

// GPU: a 32 x 64 x 32 tank (tiered) with voxel blocks, the engine's options built directly (the scenes' static mask
// path is the voxel gates' work). Each step must hold: no FAIL latch; no live particle in a closed cell (a mask voxel);
// no vertex whose eight cells are all mask voxels reads liquid (φ < 0). Over the run, live records must sit in NEAR_SOLID
// bricks (pSolid's).
// 1. The shelf: the static-need tank (liquid y < 17, a shelf at y 19..23 over x, z 8..16, a dry block at y 40..48). The
//    liquid does not meet a solid: its NEAR_SOLID bricks' records are walked without a hit.
// 2. The ledge: a blob (x, z 8..16, y 30..38) falls onto a block (x, z 4..20, y 0..12) and spreads over it. Its records
//    hit the block's top and stop at it.
// Until P's 0xa2 fix (a static cell with capacity <= 1e-5 is never a row) lands, a contact may latch 0xa2 in the
// operator; the ledge reports it as P's and stops there.
const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V3 = [number, number, number];
async function tankRun(t: { diagnostic(s: string): void }, label: string, solids: { lo: V3; hi: V3 }[], liquid: { lo: V3; hi: V3 }[], steps: number, contact: boolean, resting = contact, velocity?: { x: number; y: number; z: number }) {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { sceneDocument } = await import("../../core/scene-definition");
  const { getSceneDefinition } = await import("../../core/scenes");
  const { solidVoxelsOnMovedLattice } = await import("../../core/scene-lattice");
  const { solidOccupancyMaskVoxel } = await import("../../core/solid-occupancy-mask");
  const { H, REC, F_CLASS, brickKey, brickKeyCoord } = await import("./brick-abi");
  const { BrickEngine } = await import("./brick-engine");
  const { brickInitialLiquid, brickInitialVelocity, bootParticleCount } = await import("./brick-initial");
  const { brickStaticMask } = await import("./brick-static.wgsl");
  const { compositeMGPCG } = await import("./brick-pressure-solver");
  const { snapshot, readChannel } = await import("./brick-readback");
  const definition = getSceneDefinition("minimal-power-dam-break-32"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
  const h = 0.025, lattice: V3 = [32, 64, 32];
  scene.container = { ...scene.container, width_m: 32 * h, height_m: 64 * h, depth_m: 32 * h, fillFraction: 0 };
  scene.voxelDomain.finestCellSize_m = h; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
  scene.solidVoxels = [...scene.solidVoxels, ...solids.map(b => ({ operation: "fill" as const, minimum: b.lo, maximumExclusive: b.hi }))];
  const m = (c: V3) => ({ x: (c[0] - 16) * h, y: c[1] * h, z: (c[2] - 16) * h });
  scene.fluid.initialLiquidVolumes = liquid.map(b => ({ shape: "box" as const, min_m: m(b.lo), max_m: m(b.hi) }));
  if (velocity) scene.fluid.initialVelocity_m_s = velocity;
  const method = getMethod("uniform-narrow-band-flip"), values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
  const g = scene.fluid.gravity_m_s2, o = { lattice, h_m: h, dt_s: 1 / 60, gravity_mps2: [g.x, g.y, g.z] as V3 };
  const start = brickInitialLiquid(scene, { ...o, farPhi: 64 }), mask = brickStaticMask(scene);
  assert.ok(start.initial && mask, "the solids force the old engine's φ table and a static mask");
  const seeds = bootParticleCount({ lattice, farPhi: 64, liquid: start.liquid, initial: start.initial }, start.bricks, 3);
  const walls = brickStaticMaskWalls(lattice, false);
  const voxel = (c: readonly number[]) => c.every((v, a) => v >= 0 && v < lattice[a]!) && solidOccupancyMaskVoxel(mask.words, c[0]!, c[1]!, c[2]!);
  const closed = (c: readonly number[]) => !c.every((v, a) => v >= 0 && v < lattice[a]!) || voxel(c);
  await withUniformDevice(label, async device => {
    const e = await BrickEngine.create(device, {
      ...o, farPhi: 64, capacities: { f: 4 * 8 * 4 * 4, c: 2 * 2 * 1 * 4, b: 16, k: 64, particles: Math.max(3 * seeds, 65536) }, liquid: start.liquid, initial: start.initial,
      initialVelocity: brickInitialVelocity(scene, lattice), solver: compositeMGPCG(), stages: fluidStagesAt(true), values, particles: { bandDepth: 3 },
      fluid: { density_kg_m3: scene.fluid.density_kg_m3, viscosity_Pa_s: scene.fluid.dynamicViscosity_Pa_s, surfaceTension_N_m: scene.fluid.surfaceTension_N_m },
      staticMask: { words: mask.words, walls }, label,
    });
    const L = e.layout, S = L.particleStride, P = L.cap.particles;
    let deferred = 0, hits = 0, buriedChecked = 0;
    try {
      for (let step = 1; step <= steps; step++) {
        e.advance(1);
        const s = await snapshot(e), fail = s.header[H.FAIL]!, info = s.header[H.FAIL_INFO]!;
        if (contact && fail !== 0 && (info >>> 24) === 0xa2) { t.diagnostic(`step ${step}: 0xa2 (info 0x${info.toString(16)}), P's fix (b) pending; stopping`); break; }
        assert.equal(fail, 0, `step ${step}: FAIL ${fail} info 0x${info.toString(16)}`);
        const near = new Set<number>();
        for (const [key, slot] of s.bricks) if ((s.dir[L.dir.frec + 32 * slot + REC.CLASS]! & F_CLASS.NEAR_SOLID) !== 0) near.add(key);
        const parts = new Float32Array((await e.read("parts")).buffer), bank = step & 1, count = Math.min(s.header[bank ? H.PCOUNT1 : H.PCOUNT0]!, P);
        let inside = 0, onTop = 0, live = 0, queued = 0;
        for (let i = 0; i < count; i++) {
          const r = S * (bank * P + i); if (!(parts[r + 3]! > 0)) continue;
          live++;
          const c = [0, 1, 2].map(a => Math.floor(parts[r + a]!));
          if (near.has(brickKey(c[0]! >> 3, c[1]! >> 3, c[2]! >> 3))) queued++;
          if (closed(c)) { if (inside++ < 3) t.diagnostic(`step ${step}: a record at ${[0, 1, 2].map(a => parts[r + a]!.toFixed(3))} in a closed cell`); }
          if (voxel([c[0]!, c[1]! - 1, c[2]!])) onTop++;
        }
        hits = Math.max(hits, onTop); deferred = Math.max(deferred, queued);
        assert.equal(inside, 0, `step ${step}: ${inside} of ${live} records in closed cells`);
        const phi = await readChannel(e, "phi", "PHI");
        let wet = 0;
        for (const [key, slot] of s.bricks) {
          const b = brickKeyCoord(key).map(v => 8 * v), f = phi(slot);
          for (let l = 0; l < 512; l++) {
            const v = [b[0]! + (l & 7), b[1]! + ((l >> 3) & 7), b[2]! + (l >> 6)];
            if (![0, 1, 2, 3, 4, 5, 6, 7].every(k => voxel([v[0]! - (k & 1), v[1]! - ((k >> 1) & 1), v[2]! - (k >> 2)]))) continue;
            buriedChecked++; if (f[l]! < 0) wet++;
          }
        }
        assert.equal(wet, 0, `step ${step}: ${wet} buried vertices read liquid`);
        if (step % 10 === 0) t.diagnostic(`step ${step}: ${live} records, ${queued} in NEAR_SOLID bricks (${near.size}), resting on a voxel ${onTop}, live F ${s.header[H.LIVE_F]} K ${s.header[H.LIVE_K]}`);
      }
      t.diagnostic(`peak records in NEAR_SOLID bricks ${deferred}, peak records on a voxel ${hits}, buried vertices checked ${buriedChecked}`);
      assert.ok(deferred > 0, "pSolid moved records");
      assert.ok(buriedChecked > 0, "buried vertices were checked");
      if (resting) assert.ok(hits > 0, "records came to rest on the block");
    } finally { e.destroy(); }
  }, ["subgroups"]);
}

test("brick solid walk (GPU): the shelf tank's records near the shelf are walked, none in a solid, 30 steps", GPU, async t => {
  await tankRun(t, "solid walk shelf", [{ lo: [8, 19, 8], hi: [16, 23, 16] }, { lo: [8, 40, 8], hi: [16, 48, 16] }], [{ lo: [0, 0, 0], hi: [32, 17, 32] }], 30, false);
});

test("brick solid walk (GPU): a blob falls onto a ledge and its records stop at the ledge's top, 40 steps", GPU, async t => {
  await tankRun(t, "solid walk ledge", [{ lo: [4, 0, 4], hi: [20, 12, 20] }], [{ lo: [8, 30, 8], hi: [16, 38, 16] }], 40, true);
});

// 3. The column (P's 0xa2 gate): a column from the floor through the surface (x, z 8..16, y 0..24) stands in the
// still tank. Its faces meet the liquid from the first step: until P's fix (b) a contact row may latch 0xa2.
test("brick solid walk (GPU): a column standing through a still tank keeps its records out, 30 steps", GPU, async t => {
  await tankRun(t, "solid walk column", [{ lo: [8, 0, 8], hi: [16, 24, 16] }], [{ lo: [0, 0, 0], hi: [32, 17, 32] }], 30, true, false);
});

// 4. The hand-off: a blob thrown at 15 m/s (10 cells a step) from bricks 16 or more cells clear of a block, into it.
// Its records and level-set traces leave the ring outside the NEAR_SOLID bricks: pMove and lAdvect leave them to pSolid
// and phiSolid (PS_FAR, PS_PHI_FAR), so none crosses into the block (a contact: 0xa2 may stop it until P's (b)).
test("brick solid walk (GPU): a blob thrown across the ring at 15 m/s is walked, none in the block, 8 steps", GPU, async t => {
  await tankRun(t, "solid walk ring", [{ lo: [16, 0, 0], hi: [32, 32, 32] }], [{ lo: [0, 8, 12], hi: [6, 14, 20] }], 8, true, false, { x: 15, y: 0, z: 0 });
});
