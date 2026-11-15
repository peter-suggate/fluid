/** lDilate's absent tiles (tiered): a tile no F slot holds is a source by its eight C corners (lDilateC, DIL4), in
 * place of lTiles' and lClear's flags over its 64 cells. The rule is exact only if, for every tile whose h vertices
 * are its 4h interpolant (foVertexPhi, umVertexFrom4's f32 arithmetic), the corner rule gives lTiles' air seed and
 * occupancy and lClear's crossing cells. That is checked here over random and degenerate corners. The schedule
 * test checks that each round's lDilateC runs after its sources' writers and before the round's x pass. */
import test from "node:test";
import assert from "node:assert/strict";
import { BrickLayout } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
import { fluidLayout } from "./fluid-grid.wgsl";

const f32 = Math.fround;

/** foVertexPhi at local vertex (i, j, k) (0..4) of a 4h tile with corners c (corner order x + 2y + 4z), in f32. */
const vertexPhi = (c: readonly number[], i: number, j: number, k: number) => {
  const t = [i, j, k].map(x => f32(x / 4)), values = new Array<number>(8).fill(0);
  for (let n = 0; n < 8; n++) {
    const w = [0, 1, 2].map(a => ((n >> a) & 1) ? t[a]! : f32(1 - t[a]!));
    const weight = f32(f32(w[0]! * w[1]!) * w[2]!);
    if (weight > 0) values[n] = f32(weight * c[n]!);
  }
  const s = (a: number, b: number) => f32(values[a]! + values[b]!);
  return f32(f32(s(0, 5) + s(1, 4)) + f32(s(2, 7) + s(3, 6)));
};

/** lTiles' air seed and occupancy, and lClear's crossing (raw values: its clear raises only positives), over the
 * tile's 64 cells. */
const cellRule = (c: readonly number[]) => {
  const v: number[][][] = [];
  for (let i = 0; i <= 4; i++) { v[i] = []; for (let j = 0; j <= 4; j++) { v[i]![j] = []; for (let k = 0; k <= 4; k++) v[i]![j]![k] = vertexPhi(c, i, j, k); } }
  let seed = false, occupied = false, crossing = false;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) {
    let low = Infinity, high = -Infinity;
    for (let n = 0; n < 8; n++) { const x = v[i + (n & 1)]![j + ((n >> 1) & 1)]![k + (n >> 2)]!; low = Math.min(low, x); high = Math.max(high, x); }
    if (low <= 0) seed = true;
    if (low < 0) occupied = true;
    if (low <= 0 && high >= 0 && low < high) crossing = true;
  }
  return { seed, occupied, crossing };
};

/** lDilateC's corner rule (dSource). */
const cornerRule = (c: readonly number[]) => {
  const le = c.some(x => x <= 0), lt = c.some(x => x < 0), ge = c.some(x => x >= 0), zero = c.every(x => x === 0);
  return { seed: le, occupied: lt, crossing: le && ge && !zero };
};

test("lDilateC: a 4h tile's corner rule gives lTiles' and lClear's flags over its cells", () => {
  let seed = 0x2545f491;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; };
  const pick = () => {
    const r = random();
    if (r < 0.15) return 0;
    if (r < 0.2) return random() < 0.5 ? -0 : 16;
    if (r < 0.3) return f32((random() - 0.5) * 1e-3);
    return f32((random() - 0.5) * 40);
  };
  const cases: number[][] = [
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, -1], [-1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 0], [-2, 2, 2, -2, 2, -2, -2, 2],
    [f32(1e-6), 1, 1, 1, 1, 1, 1, -f32(1e-6)], [16, 16, 16, 16, 16, 16, 16, -0],
  ];
  // A third mixed, a third with no corner under 0, a third with none over 0 (zeros and tiny values in each).
  for (let n = 0; n < 6000; n++) { const side = n % 3; cases.push(Array.from({ length: 8 }, () => { const x = pick(); return side === 0 ? x : side === 1 ? Math.abs(x) : -Math.abs(x); })); }
  let crossings = 0;
  for (const c of cases) {
    const want = cellRule(c), got = cornerRule(c);
    assert.deepEqual(got, want, `corners ${c.join(",")}`);
    if (want.crossing) crossings++;
  }
  assert.ok(crossings > 2000 && crossings < cases.length - 500, `${crossings} crossing tiles of ${cases.length}`);
});

test("lDilateC: each round's runs after its sources' writers and before the round's x pass (tiered only)", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  assert.equal(fluidLayout(L, { tiered: false }).fcell.LIQ + 16 * 600, fluidLayout(L, { tiered: false }).fcell.DIL4);
  assert.equal(fluidLayout(L, { tiered: true }).fcell.words - fluidLayout(L, { tiered: true }).fcell.DIL4, 512 * 40);
  for (const tiered of [false, true]) {
    const stages = fluidStagesAt(tiered)({ layout: L, values: { volumeControlSeconds: 0 }, fluid: { viscosity_Pa_s: 0, density_kg_m3: 1000, surfaceTension_N_m: 0 } } as never) as unknown as Record<string, unknown>;
    type P = { kernel: { label: string; code: string }; z?: number; w?: number };
    for (const name of ["levelSet", "boot"]) {
      const list = stages[name] as P[];
      const at = (label: string, z?: number, w?: number) => list.findIndex(p => p.kernel.label === label && (z === undefined || (p.z ?? 0) === z) && (w === undefined || (p.w ?? 0) === w));
      const readers = list.filter(p => p.kernel.code.includes("FC_DIL4+")).map(p => p.kernel.label);
      if (!tiered) { assert.equal(at("lDilateC"), -1, `${name}: all h runs no lDilateC`); continue; }
      assert.deepEqual([...new Set(readers)], ["lDilateC", "lDilate"], `${name}: DIL4's writer and reader`);
      const c1 = at("lDilateC", 0, 1), c2 = at("lDilateC", 0, 2);
      assert.ok(at("lTiles") < c1 && c1 < at("lDilate", 0, 1), `${name}: round 1 lDilateC after lTiles, before the x pass`);
      assert.ok(at("lClear") < c2 && at("lcClear") < c2 && c2 < at("lDilate", 0, 2), `${name}: round 2 lDilateC after lClear and lcClear, before the x pass`);
      // Nothing between a round's lDilateC and its last pass writes φ or the tile flags.
      for (const [from, to] of [[c1, at("lDilate", 2, 1)], [c2, at("lDilate", 2, 2)]] as const)
        for (const p of list.slice(from + 1, to)) assert.equal(p.kernel.label, "lDilate", `${name}: ${p.kernel.label} inside a round`);
    }
  }
});

test("fcell binds read-only, with plain loads, in every kernel that writes none of it", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  for (const tiered of [false, true]) {
    const stages = fluidStagesAt(tiered)({ layout: L, values: { volumeControlSeconds: 0 }, fluid: { viscosity_Pa_s: 0, density_kg_m3: 1000, surfaceTension_N_m: 0 } } as never) as unknown as Record<string, unknown>;
    type K = { label: string; code: string; bindings: { name: string; kind: string; type: string }[] };
    const kernels = new Map<string, K>();
    const walk = (x: unknown): void => { if (Array.isArray(x)) x.forEach(walk); else if (x && typeof x === "object") { const k = (x as { kernel?: K }).kernel; if (k) kernels.set(k.label, k); else Object.values(x).forEach(walk); } };
    walk(stages);
    for (const k of kernels.values()) {
      const b = k.bindings.find(x => x.name === "fcell"); if (!b) continue;
      const writes = /fcStore|atomic\w*\(&fcell\[/.test(k.code);
      assert.equal(b.kind, writes ? "rw" : "ro", `${k.label} (tiered ${tiered})`);
      if (!writes) assert.ok(b.type === "array<u32>" && !k.code.includes("&fcell["), `${k.label}: plain loads`);
    }
    for (const label of ["uAdvect", "lAdvect", "uExtend", "pSeed"]) assert.equal(kernels.get(label)?.bindings.find(x => x.name === "fcell")?.kind, "ro", `${label} (tiered ${tiered})`);
  }
});
