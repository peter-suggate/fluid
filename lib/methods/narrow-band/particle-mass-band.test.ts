/**
 * LANE F OWNS THIS FILE. The deep class's mass band (lane F, 2026-11-17): pMove's cell-mass cut and pSeed's refill
 * agree on one cell's mass (8) where the refill seeds graded records (centre depth <= -1.5). CPU only.
 *
 * The step's refill (pSeedCellMask) raises a deep cell below 8 to its goal (nbSeedTarget 4 records of mass 2, under the
 * 8-mass cap); the move's cut (Ferstl et al. 2016 §3.3) retires the records of a cell's run past its bound. With the
 * bound at NB_CELL_MASS (16) the band was one-sided: a moving interior's shuffle leaves some cells under 8 (refilled)
 * and some over 8 (kept), so the mean mass ratchets up, and the level set's union follows it (fill-256-800 +25% at rest,
 * fill-256-6400 +15%, nbflip-stair-cube +10%, cm12-figure-7 +3% where the old engine has none of it).
 * - The kernel: pMove cuts at pCellMassBound(deep), 8 in the deep class and 16 above it.
 * - The oracle (Float64): a periodic 16^3 deep block, every record moving to a random neighbour with probability p each
 *   step (mass-preserving mixing), then the refill and the cut read the moved state. The band at 8 holds the mass to
 *   0.5% over 40 steps; the old bound at 16 gains over 20%.
 * - The boundary cases: an empty deep cell refills to 8; a full one takes nothing; a cell over 8 is cut to 8, keeping
 *   its run's first records.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { fluidLayout } from "./fluid-grid.wgsl";
import { particleKernels } from "./particles-kernels.wgsl";

test("mass band: pMove cuts a deep-class cell at one cell's mass (8), shallower cells at NB_CELL_MASS", () => {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  for (const tiered of [false, true]) {
    const ks = Object.values(particleKernels(L, fluidLayout(L, { tiered }))) as unknown as { label: string; code: string }[];
    const move = ks.find(k => k?.label === "pMove")!.code;
    assert.ok(move.includes("fn pCellMassBound(deep:f32)->f32{return select(NB_CELL_MASS,8.0,deep<= -1.5);}"));
    assert.ok(move.includes("var held=0.0;let bound=pCellMassBound(deep);"));
    assert.ok(move.includes("if(held>0.0&&held+m>bound){end=j;break;}"));
    // The cut loop over the cell's run is the bound's (nbfCellMassAdd, the old parity helper, keeps NB_CELL_MASS).
    assert.ok(!/for\(var j=start;j<run\.y;j\+\+\)\{[^\n]*held\+m>NB_CELL_MASS/.test(move));
  }
});

// ---- The oracle. A cell's records are their masses (1/8 cell), in run order.
const MASS = 2, GOAL = 4;
const held = (recs: readonly number[]) => recs.reduce((t, m) => t + m, 0);
/** pSeedCellMask's step refill in a deep cell: none at the goal's count or one cell's mass; else sites n0..GOAL-1 while
 * held + (j - n0) MASS < 8. Returns the masses seeded. */
function refill(recs: readonly number[]): number[] {
  const n0 = recs.length, h = held(recs); if (n0 >= GOAL || h >= 8) return [];
  const add: number[] = []; for (let j = n0; j < GOAL; j++) if (h + (j - n0) * MASS < 8) add.push(MASS); return add;
}
/** pMove's cut: the run's records kept while held + m <= bound (the first always). Returns the records cut. */
function cut(recs: readonly number[], bound: number): number {
  let h = 0, keep = 0; for (const m of recs) { if (h > 0 && h + m > bound) break; h += m; keep++; } return recs.length - keep;
}
function shuffle(bound: number, p: number, steps: number): number {
  const N = 16, at = (x: number, y: number, z: number) => ((x + N) % N) + N * (((y + N) % N) + N * ((z + N) % N));
  let s = 12345; const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
  let g: number[][] = Array.from({ length: N ** 3 }, () => [2, 2, 2, 2]);
  const start = g.reduce((t, c) => t + held(c), 0);
  for (let step = 0; step < steps; step++) {
    const moved: number[][] = g.map(() => []);
    g.forEach((c, i) => {
      const x = i % N, y = Math.floor(i / N) % N, z = Math.floor(i / (N * N));
      for (const m of c) {
        if (rnd() < p) { const d = () => Math.floor(rnd() * 3) - 1; moved[at(x + d(), y + d(), z + d())]!.push(m); } else moved[i]!.push(m);
      }
    });
    const adds = moved.map(refill), cuts = moved.map(c => cut(c, bound));
    g = moved.map((c, i) => [...c.slice(0, c.length - cuts[i]!), ...adds[i]!]);
  }
  return g.reduce((t, c) => t + held(c), 0) / start - 1;
}

test("mass band (oracle): a shuffled deep region keeps its mass with the cut at 8, and ratchets up with 16", () => {
  for (const p of [0.1, 0.3]) {
    const band = shuffle(8, p, 40), old = shuffle(16, p, 40);
    assert.ok(Math.abs(band) < 0.005, `p ${p}: the band at 8 drifts ${(100 * band).toFixed(2)}%`);
    assert.ok(old > 0.2, `p ${p}: the old cut at 16 gains ${(100 * old).toFixed(2)}% (the ratchet this replaces)`);
  }
});

test("mass band (oracle): the boundary cases", () => {
  assert.deepEqual(refill([]), [2, 2, 2, 2], "an empty deep cell refills to one cell's mass");
  assert.deepEqual(refill([2, 2, 2, 2]), [], "a full cell takes nothing");
  assert.deepEqual(refill([2, 2]), [2, 2], "a half cell refills to 8");
  assert.equal(cut([2, 2, 2, 2], 8), 0, "a full cell keeps its records");
  assert.equal(cut([2, 2, 2, 2, 2, 2], 8), 2, "a cell over 8 is cut to 8, its run's first records kept");
  assert.equal(cut([8, 2], 8), 1);
  assert.equal(cut([2, 2, 2, 2, 2, 2], 16), 0, "the old bound kept it");
});
