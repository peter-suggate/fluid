// The tiered brick set (brick-abi.md §2.5, brick-set.wgsl.ts with `tiered`) on a frozen state: after a tier
// commit, setMark keeps F bricks by core and ring only (no bins here), setMarkPages keeps the pages that 4h
// liquid reaches and lists the missing ones as page candidates, bricks entered in a step start 4h, and
// entering pages start as air. GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_STRIDE, CMASK_WORDS, FLAG, H, LIST, PHI_AIR, REC, brickKeyCoord, phiFar } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { snapshot } from "./brick-readback";
import { brickSetKernels } from "./brick-set.wgsl";
import { bringUpStages } from "./brick-stages";
import { brickTransferKernels } from "./brick-transfer.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V3 = [number, number, number];
const N = 64;
const key = (b: V3) => ((b[0] & 1023) | ((b[1] & 1023) << 10) | ((b[2] & 1023) << 20)) >>> 0;
const tileOf = (i: number): V3 => [i & 7, (i >> 3) & 7, i >> 6];
/** h tiles after the commit: block A (tiles x 0..5, y 4..5, z 0..5) and block B (tiles x 14..15, y 7, z 14): core bricks A x 0..2, y 2, z 0..2 and B (7, 3, 7). */
const wantH = (T: V3) => (T[0] <= 5 && T[1] >= 4 && T[1] <= 5 && T[2] <= 5) || (T[0] >= 14 && T[1] === 7 && T[2] === 14);
/** C φ (h units): liquid below 26 cells, so 4h tiles up to y = 6 hold a liquid corner and their reach (7.17 cells) crosses into page row y = 1. */
const cPhi = (T: V3) => 4 * T[1] - 26;

test("tiered brick set (GPU): core + ring keep F, 4h liquid keeps and requests pages, entries start 4h and air", GPU, async () => {
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
    const part2 = e.schedule.part2, setArgs = brickSetKernels(L, { tiered: true }).setArgs.label;
    const update = part2.slice(0, part2.findIndex(p => p.kernel.label === setArgs) + 1);
    assert.deepEqual(update.slice(0, 2).map(p => p.kernel.label), ["setMark", "setMarkPages"]);
    await e.run(update);
    const s = await snapshot(e);
    assert.equal(s.header[H.FAIL], 0, `FAIL ${s.header[H.FAIL]} info ${s.header[H.FAIL_INFO]}`);
    // ---- The oracle.
    const nb = Math.floor(N / 8) + 1, np = Math.ceil(nb / 4);
    const coreA = (b: V3) => b[0] <= 2 && b[1] === 2 && b[2] <= 2, coreB = (b: V3) => b[0] === 7 && b[1] === 3 && b[2] === 7;
    const ring = (core: (b: V3) => boolean, b: V3) => {
      for (let d = 0; d < 27; d++) { const n = [b[0] + (d % 3) - 1, b[1] + (Math.floor(d / 3) % 3) - 1, b[2] + Math.floor(d / 9) - 1] as V3; if (n.every(v => v >= 0 && v < nb) && core(n)) return true; }
      return false;
    };
    const kept = new Set<number>();
    for (let z = 0; z < nb; z++) for (let y = 0; y < nb; y++) for (let x = 0; x < nb; x++) if (ring(coreA, [x, y, z]) || ring(coreB, [x, y, z])) kept.add(key([x, y, z]));
    assert.deepEqual([...s.bricks.keys()].sort((a, b) => a - b), [...kept].sort((a, b) => a - b), "F residency is core + ring");
    const entered = [...kept].filter(k => !s0.bricks.has(k));
    assert.equal(entered.length, 9, "the ring of B above row 3 enters");
    // Pages: every page row y = 0 page stays (4h liquid, or a neighbour's reach), row y = 1 enters (page candidates and B's ring).
    const wantPages = new Set<number>();
    for (let z = 0; z < np; z++) for (let x = 0; x < np; x++) for (const y of [0, 1]) wantPages.add(key([x, y, z]));
    assert.deepEqual([...s.pages.keys()].sort((a, b) => a - b), [...wantPages].sort((a, b) => a - b), "C residency");
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
      const P = brickKeyCoord(pk) as unknown as V3;
      let n = 0;
      for (let i = 0; i < 512; i++) {
        const T = tileOf(i).map((t, a) => 8 * P[a]! + t) as V3;
        if (!T.every(v => 4 * v < N) || wantH(T)) continue;
        let m = PHI_AIR;
        for (let q = 0; q < 8; q++) m = Math.min(m, cVertex([T[0] + (q & 1), T[1] + ((q >> 1) & 1), T[2] + (q >> 2)]));
        if (m < 0) n++;
      }
      assert.equal(s.dir[L.dir.crec + 32 * cs + REC.C_LIQUID_TILES], n, `LIQUID_TILES page ${P}`);
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
    console.log(`tiered set: ${s.bricks.size} F (${entered.length} entered, ${s0.bricks.size + entered.length - s.bricks.size} left), ${s.pages.size} C (${s.pages.size - pages0.size} entered)`);
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
  });
});
