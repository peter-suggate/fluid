// The tiered brick set (brick-abi.md §2.5 and §5.4, brick-set.wgsl.ts and brick-residency.wgsl.ts with `tiered`)
// on a frozen state: after a tier commit, the head's census and residency keep and request F bricks (ownership and
// particle records, and WANT's rings) and C pages (every stage's records) as the CPU residency oracle says, bricks
// entered in a step start 4h, entering pages start as air, and setMarkPages counts LIQUID_TILES. GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_STRIDE, CMASK_WORDS, FLAG, H, LIST, PHI_AIR, REC, brickKeyCoord, phiFar } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { snapshot } from "./brick-readback";
import { residencyOracle } from "./brick-oracle";
import { brickSetKernels } from "./brick-set.wgsl";
import { bringUpStages } from "./brick-stages";
import { brickTransferKernels } from "./brick-transfer.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V3 = [number, number, number];
const N = 64;
/** The engine's dt, h and gravity (BrickEngine.create below), for the residency oracle's flow term. */
const RING = { dt: 1 / 60, h: 1 / N, gravity: [0, -9.81, 0] as [number, number, number] };
const key = (b: V3) => ((b[0] & 1023) | ((b[1] & 1023) << 10) | ((b[2] & 1023) << 20)) >>> 0;
const tileOf = (i: number): V3 => [i & 7, (i >> 3) & 7, i >> 6];
/** h tiles after the commit: block A (tiles x 0..5, y 4..5, z 0..5) and block B (tiles x 14..15, y 7, z 14): core bricks A x 0..2, y 2, z 0..2 and B (7, 3, 7). */
const wantH = (T: V3) => (T[0] <= 5 && T[1] >= 4 && T[1] <= 5 && T[2] <= 5) || (T[0] >= 14 && T[1] === 7 && T[2] === 14);
/** C φ (h units): liquid below 26 cells, so 4h tiles up to y = 6 hold a liquid corner and their reach (7.17 cells) crosses into page row y = 1. */
const cPhi = (T: V3) => 4 * T[1] - 26;

test("tiered brick set (GPU): the head's residency keeps and requests F and C as the oracle says, entries start 4h and air", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick set tiers", async device => {
    const stages = (c: Parameters<typeof bringUpStages>[0]) => ({ ...bringUpStages(c), tiered: true });
    const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: 1 / N, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 600, c: 40, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [N + 1, 20.3, N + 1] }], solver: L => bringUpCG(L), stages });
    const L = e.layout, s0 = await snapshot(e);
    assert.equal(s0.header[H.FAIL], 0);
    // ---- The frozen state: WANT committed (setCommitMask), C φ on every page, U = 0 (setInit).
    const pages0 = new Map<number, number>(s0.pages);
    for (const [pk, cs] of pages0) {
      const P = brickKeyCoord(pk) as unknown as V3, want = new Uint32Array(CMASK_WORDS), phi = new Float32Array(512);
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3;
        if (T.every(v => 4 * v < N) && wantH(T)) want[i >> 5]! |= 1 << (i & 31);
        phi[i] = cPhi(T);
      }
      e.write("set", L.set.want + CMASK_WORDS * cs, want);
      e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, phi);
    }
    const listC = { indirect: 12 * LIST.C_ALL };
    await e.run([{ kernel: brickTransferKernels(L).setCommitMask, dispatch: listC }]);
    // The class pass of the committed state (part 2's setMark): the head's brick sources are its h-owned tiles.
    await e.run(e.schedule.part2.filter(p => p.kernel.label === "setMark"));
    // The head up to its lists: census, residency (seed, request, claims, leaves), records, init, lists.
    const before = await snapshot(e), part1 = e.schedule.part1, cBefore = { phi: await e.read("phi"), vel: await e.read("vel") };
    const head = part1.slice(0, part1.findIndex(p => p.kernel.label === "censusWant"));
    assert.deepEqual(head.slice(-2).map(p => p.kernel.label), ["setLists", "setArgs"]);
    await e.run(head);
    const s = await snapshot(e);
    assert.equal(s.header[H.FAIL], 0, `FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]}`);
    // ---- The oracle (resMark's local per-axis reach from the ACTIVE bricks, plus the census's band rings): the set and its radius.
    const want = residencyOracle(L, before, s.set, true, RING, cBefore);
    assert.equal(s.header[H.RES_RADIUS], want.radius, "RES_RADIUS");
    assert.deepEqual([...s.bricks.keys()].sort((a, b) => a - b), [...want.bricks].sort((a, b) => a - b), "F residency");
    assert.deepEqual([...s.pages.keys()].sort((a, b) => a - b), [...want.pages].sort((a, b) => a - b), "C residency");
    const entered = [...want.bricks].filter(k => !s0.bricks.has(k));
    assert.ok(entered.length > 0 && [...want.pages].some(k => !pages0.has(k)), "bricks and pages enter");
    // C_LIQ: part 2's setMarkPages on the new set.
    await e.run(e.schedule.part2.filter(p => p.kernel.label === "setMarkPages"));
    const liq = await snapshot(e);
    // Masks: committed pages keep WANT; entered pages and bricks hold no h tile.
    for (const [pk, cs] of s.pages) {
      const m = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
      const P = brickKeyCoord(pk) as unknown as V3;
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3;
        const h = ((m[i >> 5]! >>> (i & 31)) & 1) === 1;
        assert.equal(h, pages0.has(pk) && T.every(v => 4 * v < N) && wantH(T), `CMASK page ${P} tile ${tileOf(i)}`);
      }
    }
    for (const k of entered) assert.equal(s.dir[L.dir.frec + 32 * s.bricks.get(k)! + REC.MASK]! & 0xff, 0, "an entered brick starts 4h");
    // LIQUID_TILES: 4h-owned tiles in the lattice with a liquid corner (corners in non-resident pages read air).
    const cVertex = (T: V3) => {
      if (T.some(v => v < 0 || 4 * v > N)) return phiFar([N, N, N]);
      const pk = key(T.map(v => v >> 3) as V3);
      return pages0.has(pk) ? Math.fround(cPhi(T)) : phiFar([N, N, N]);
    };
    for (const [pk, cs] of pages0) {
      if (!liq.pages.has(pk)) continue;
      const P = brickKeyCoord(pk) as unknown as V3;
      let n = 0;
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3;
        if (!T.every(v => 4 * v < N) || wantH(T)) continue;
        let m = PHI_AIR;
        for (let q = 0; q < 8; q++) m = Math.min(m, cVertex([T[0] + (q & 1), T[1] + ((q >> 1) & 1), T[2] + (q >> 2)]));
        if (m < 0) n++;
      }
      assert.equal(liq.dir[L.dir.crec + 32 * cs + REC.C_LIQUID_TILES], n, `LIQUID_TILES page ${P}`);
    }
    // Entered pages start as air: φ AIR, every velocity 0, FLAGS open on in-lattice tiles.
    for (const [pk, cs] of s.pages) {
      if (pages0.has(pk)) continue;
      const P = brickKeyCoord(pk) as unknown as V3;
      const phi = new Float32Array((await e.read("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, 512)).buffer);
      const flags = await e.read("phi", L.base("phi", "C", "FLAGS") + BRICK_STRIDE * cs, 512);
      const u = new Float32Array((await e.read("vel", L.base("vel", "C", "US1") + BRICK_STRIDE * cs, 512)).buffer);
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3;
        assert.equal(phi[i], PHI_AIR); assert.equal(u[i], 0);
        assert.equal(flags[i], T.every(v => 4 * v < N) ? FLAG.OPEN : 0);
      }
    }
    console.log(`tiered set: ${s.bricks.size} F (${entered.length} entered, ${s0.bricks.size + entered.length - s.bricks.size} left), ${s.pages.size} C (${s.pages.size - pages0.size} entered), claimed F ${s.header[H.RES_CLAIMED_F]} C ${s.header[H.RES_CLAIMED_C]}`);
  });
});

test("tiered boot (GPU): the census commits without a remap; a column's edges are h, its still interior 4h", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick tiered boot", async device => {
    const stages = (c: Parameters<typeof bringUpStages>[0]) => ({ ...bringUpStages(c), tiered: true });
    const hi: V3 = [30.3, 40.3, 30.3];
    const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: 1 / N, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 600, c: 40, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi }], solver: L => bringUpCG(L), stages });
    const L = e.layout, s = await snapshot(e);
    assert.equal(s.header[H.FAIL], 0, `FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]}`);
    let nH = 0, n4 = 0, edgeH = 0, deepH = 0;
    for (const [pk, cs] of s.pages) {
      const P = brickKeyCoord(pk) as unknown as V3, m = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
      const phi = new Float32Array((await e.read("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, 512)).buffer);
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3, c = T.map(v => 4 * v) as V3;
        if (T.every(v => 4 * v <= N)) {
          // bootPhi's C vertex: the box distance at h vertex 4T, as F holds it.
          const q = c.map((v, a) => Math.abs(v - 0.5 * (hi[a]! - 1)) - 0.5 * (hi[a]! + 1));
          const d = Math.min(Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(...q), 0), PHI_AIR);
          assert.ok(Math.abs(phi[i]! - d) <= 1e-5 * Math.max(1, Math.abs(d)), `C φ at ${T}: ${phi[i]} vs ${d}`);
        }
        if (!T.every(v => 4 * v < N)) continue;
        const h = ((m[i >> 5]! >>> (i & 31)) & 1) === 1;
        if (h) nH++; else n4++;
        const inside = c.every((v, a) => v + 4 <= hi[a]! - 8);
        if (h && inside) deepH++;
        if (h && T[0] === 7 && T[2] === 7 && T[1] < 9) edgeH++;
      }
    }
    console.log(`tiered boot: ${nH} h tiles, ${n4} 4h; edge column h ${edgeH}; deep h ${deepH}; census crossing ${s.header[H.CENSUS_CROSSING]} required ${s.header[H.CENSUS_REQUIRED]} band ${s.header[H.CENSUS_BAND]} clamped ${s.header[H.CENSUS_CLAMPED]} triggered ${s.header[H.CENSUS_TRIGGERED]}`);
    assert.ok(nH > 0 && n4 > 0, "both tiers at boot");
    assert.equal(deepH, 0, "the still interior (more than 8 cells deep) is 4h");
    // The boot claimed every band tile's ring, so nothing is clamped at 4h.
    for (const [pk, cs] of s.pages) {
      const P = brickKeyCoord(pk) as unknown as V3, want = s.set.subarray(L.set.want + CMASK_WORDS * cs, L.set.want + CMASK_WORDS * (cs + 1));
      const m = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
      assert.deepEqual([...m], [...want], `page ${P}: CMASK = WANT (the census's band, unclamped)`);
    }
    // Part 1's head again on the unchanged state: the same decisions, so the transfer lists no change.
    const cS = { phi: await e.read("phi"), vel: await e.read("vel") };
    await e.run(e.schedule.part1);
    const s2 = await snapshot(e);
    assert.equal(s2.header[H.FAIL], 0, `FAIL ${s2.header[H.FAIL]} info ${s2.header[H.FAIL_INFO]}`);
    for (const [pk, cs] of s.pages) {
      const a = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1)), b = s2.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
      assert.deepEqual([...b], [...a], `page ${brickKeyCoord(pk)}: a still column keeps its tiers`);
    }
    // The head's residency replaces the boot's reach set with the residency oracle's (the ring of every band tile kept).
    const res = residencyOracle(L, s, s2.set, true, RING, cS);
    assert.deepEqual([...s2.bricks.keys()].sort((a, b) => a - b), [...res.bricks].sort((a, b) => a - b), "F residency after the head");
    assert.deepEqual([...s2.pages.keys()].sort((a, b) => a - b), [...res.pages].sort((a, b) => a - b), "C residency after the head");
    console.log(`tiered head residency: boot ${s.bricks.size} F -> ${s2.bricks.size} F, ${s.pages.size} C -> ${s2.pages.size} C`);
    console.log(`tiered head: listed ${s2.header[H.N_TLIST_RUN]} (neighbour-only listing), clamped ${s2.header[H.CENSUS_CLAMPED]}, cand ${s2.header[H.CENSUS_CAND]}`);
  });
});

test("all-h brick set (GPU): every tile of a resident page is h-owned, so an absent brick is air, never stale C", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick all-h cmask", async device => {
    // Lane P's finding (2026-11-15): with CMASK per brick, an all-h page's absent bricks read 4h, i.e. bootPhi's
    // C vertices, which nothing updates all-h. A column that fills a page only partly leaves such bricks.
    const hi: V3 = [30.3, 40.3, 30.3];
    const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: 1 / N, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 600, c: 40, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi }], solver: L => bringUpCG(L), stages: bringUpStages });
    const L = e.layout, s = await snapshot(e);
    assert.equal(s.header[H.FAIL], 0, `FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]}`);
    let absent = 0;
    for (const [pk, cs] of s.pages) {
      const P = brickKeyCoord(pk) as unknown as V3, m = s.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
      assert.ok(m.every(w => w === 0xffffffff), `page ${P}: CMASK all-ones`);
      for (let bl = 0; bl < 64; bl++) {
        const b: V3 = [4 * P[0]! + (bl & 3), 4 * P[1]! + ((bl >> 2) & 3), 4 * P[2]! + (bl >> 4)];
        if (!s.bricks.has(key(b))) absent++;
      }
    }
    assert.ok(absent > 0, "the column leaves absent bricks in its pages");
    // One full step: the census classifies the absent bricks' tiles as air (no TIER failure) and CMASK stays.
    e.advance(1); await e.fence(); const s2 = await snapshot(e);
    assert.equal(s2.header[H.FAIL], 0, `FAIL ${s2.header[H.FAIL]} info ${s2.header[H.FAIL_INFO]}`);
    for (const [, cs] of s2.pages) assert.ok(s2.dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1)).every(w => w === 0xffffffff));
  });
});
