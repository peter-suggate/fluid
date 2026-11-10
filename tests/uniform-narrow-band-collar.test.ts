import assert from "node:assert/strict";
import test from "node:test";
import {
  NARROW_BAND_DEEP_SITES, narrowBandCollarArrival, narrowBandCollarArrivalWGSL, narrowBandCollarCheck, narrowBandCollarDensity,
  narrowBandCollarDensityWGSL, narrowBandCollarSpread, narrowBandCollarSpreadWGSL, narrowBandCollarTargets, narrowBandCorner,
  narrowBandSeedSite, narrowBandSeedTarget, narrowBandSeedTargetWGSL, type CollarTileGrid, type NarrowBandCollarDensity,
} from "../lib/methods/uniform/uniform-narrow-band-collar";
import { narrowBandActivityWGSL, narrowBandActivityWGSLFor } from "../lib/methods/uniform/uniform-narrow-band-activity.wgsl";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { narrowBandFlipValues, uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { UNIFORM_STAGE_IMPORTANCE } from "../lib/methods/uniform/uniform-stage-grids";

// The text the solver compiled before collarSpread/collarDensity (HEAD 112163d3).
const OLD_SPREAD_LOOP = `for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
   let value=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+umTileAt(vec3u(q))]));
   heat=max(heat,value*select(0.5,1.0,x==0&&y==0&&z==0));
  }}}`;
const OLD_ARRIVAL = "if(arrived>0.0){heat=min(heat,max(arrived,1.0));}";

// A deterministic generator (mulberry32).
function random(seed: number): () => number {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const N: [number, number, number] = [7, 6, 5];
const at = (x: number, y: number, z: number) => x + N[0] * (y + N[1] * z);
function grid(seed: number): CollarTileGrid & { heat: Float64Array; wet: boolean[]; present: boolean[] } {
  const r = random(seed), count = N[0] * N[1] * N[2];
  const heat = new Float64Array(count), wet: boolean[] = [], present: boolean[] = [];
  for (let i = 0; i < count; i++) {
    present.push(r() > 0.05); wet.push(r() > 0.4);
    // Triggered tiles start at 2 and cool; most tiles are cold.
    const u = r(); heat[i] = u < 0.15 ? 2 : u < 0.3 ? 2 * r() : 0;
  }
  return { n: N, heat, wet, present };
}
function neighbours(x: number, y: number, z: number, f: (x: number, y: number, z: number) => void): void {
  for (let c = -1; c <= 1; c++) for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
    const p = [x + a, y + b, z + c] as const;
    if (p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] < N[0] && p[1] < N[1] && p[2] < N[2]) f(...p);
  }
}

test("collarSpread and collarDensity parse their values, default to today's, and fail on unknown ones", () => {
  assert.equal(narrowBandCollarSpread(undefined), "neighbours");
  assert.equal(narrowBandCollarSpread("liquid"), "liquid");
  assert.equal(narrowBandCollarDensity(undefined), "band");
  assert.equal(narrowBandCollarDensity("deep"), "deep");
  for (const bad of ["bogus", "", 1, true, null, "Liquid"]) {
    assert.throws(() => narrowBandCollarSpread(bad), /collarSpread must be one of neighbours, liquid/);
    assert.throws(() => narrowBandCollarDensity(bad), /collarDensity must be one of band, deep/);
    assert.throws(() => uniformNearSettled({ collarSpread: bad as string }), /collarSpread/);
    assert.throws(() => uniformNearSettled({ collarDensity: bad as string }), /collarDensity/);
  }
  const settled = uniformNearSettled({});
  assert.equal(settled.collarSpread, "neighbours");
  assert.equal(settled.collarDensity, "band");
  assert.equal(uniformNearSettled({ collarSpread: "liquid", collarDensity: "deep" }).collarDensity, "deep");
  // Deep samples carry two units: only crowdedCells "merge" (the default) holds mass.
  assert.throws(() => uniformNearSettled({ collarDensity: "deep", crowdedCells: "retire" }), /crowdedCells "merge"/);
  assert.throws(() => narrowBandCollarCheck("deep", "retire"), /crowdedCells "merge"/);
  assert.doesNotThrow(() => narrowBandCollarCheck("band", "retire"));
  const values = narrowBandFlipValues({});
  assert.equal(values.collarSpread, "neighbours");
  assert.equal(values.collarDensity, "band");
  for (const [key, options] of [["collarSpread", ["neighbours", "liquid"]], ["collarDensity", ["band", "deep"]]] as const) {
    const param = uniformNarrowBandMethod.params.find(p => p.key === key);
    assert.ok(param && param.kind === "select" && param.update === "solver", key);
    assert.deepEqual(param.kind === "select" ? param.options.map(o => o.value) : [], options);
  }
});

test("the default collar values compile today's WGSL exactly", () => {
  assert.equal(narrowBandCollarSpreadWGSL("neighbours", UNIFORM_STAGE_IMPORTANCE.wet), OLD_SPREAD_LOOP);
  assert.equal(narrowBandActivityWGSL, narrowBandActivityWGSLFor("neighbours"));
  assert.ok(narrowBandActivityWGSL.includes(`  ${OLD_SPREAD_LOOP}\n  let old=`));
  assert.equal(narrowBandCollarArrivalWGSL("neighbours"), OLD_ARRIVAL);
  assert.equal(narrowBandSeedTargetWGSL("band", "d", "heat", "nbSourceCell(c)"), "nbSeedTarget(d)");
  assert.equal(narrowBandCollarDensityWGSL("band"), "");
  // The variants read the census's wet bit, and the deep target is defined.
  const liquid = narrowBandActivityWGSLFor("liquid");
  assert.ok(liquid.includes(`&${UNIFORM_STAGE_IMPORTANCE.wet >>> 0}u)!=0u`) && liquid.includes("for(var z=-reach;z<=reach;z++)"));
  assert.ok(!narrowBandCollarArrivalWGSL("liquid").includes("if(arrived>0.0)"));
  assert.ok(narrowBandCollarDensityWGSL("deep").includes("fn nbCollarSeedTarget(heat:f32,depth:f32,source:bool)->u32"));
  assert.equal(narrowBandSeedTargetWGSL("deep", "d", "heat", "nbSourceCell(c)"), "nbCollarSeedTarget(heat,d,nbSourceCell(c))");
});

test("collarSpread neighbours is today's spread: the most of a tile's heat and half of each present neighbour's", () => {
  for (let seed = 1; seed <= 20; seed++) {
    const g = grid(seed), targets = narrowBandCollarTargets(g, "neighbours");
    for (let z = 0; z < N[2]; z++) for (let y = 0; y < N[1]; y++) for (let x = 0; x < N[0]; x++) {
      const i = at(x, y, z);
      if (!g.present[i]) { assert.equal(targets[i], 0); continue; }
      let heat = 0;
      neighbours(x, y, z, (a, b, c) => { const j = at(a, b, c); if (g.present[j]) heat = Math.max(heat, g.heat[j]! * (j === i ? 1 : 0.5)); });
      assert.equal(targets[i], heat);
    }
  }
  // The arrival cap: a grid-supported sample arriving where the target is zero keeps its heat (the gap "liquid" closes).
  assert.equal(narrowBandCollarArrival("neighbours", 2, 0), 2);
  assert.equal(narrowBandCollarArrival("neighbours", 2, 0.5), 1);
  assert.equal(narrowBandCollarArrival("neighbours", 2, 1.6), 1.6);
});

test("collarSpread liquid: every wet tile keeps today's target and dry tiles take no neighbour heat", () => {
  for (let seed = 1; seed <= 20; seed++) {
    const g = grid(seed), old = narrowBandCollarTargets(g, "neighbours"), next = narrowBandCollarTargets(g, "liquid");
    for (let i = 0; i < old.length; i++) {
      if (!g.present[i]) continue;
      // (i) a wet tile beside a hot one is in the collar exactly as before: seeded.
      if (g.wet[i]) assert.equal(next[i], old[i]);
      // (ii) a dry tile holds only its own heat: a cold dry tile beside a hot one gets none.
      else assert.equal(next[i], g.heat[i]);
    }
    // (i) every wet present neighbour of a heated tile takes at least half its heat.
    for (let z = 0; z < N[2]; z++) for (let y = 0; y < N[1]; y++) for (let x = 0; x < N[0]; x++) {
      const i = at(x, y, z); if (!g.present[i] || g.heat[i]! <= 0) continue;
      neighbours(x, y, z, (a, b, c) => { const j = at(a, b, c); if (g.present[j] && g.wet[j]) assert.ok(next[j]! >= 0.5 * g.heat[i]!); });
    }
  }
});

test("collarSpread liquid: no grid-supported sample ends a step with erasure heat outside the heated tiles", () => {
  // A tile's target exceeds one only through its own heat (a neighbour gives at most half of 2), so "heated" is own heat over one:
  // those tiles are seeded (target > 0), and so are their wet neighbours. A sample may travel any number of tiles in a step.
  for (let seed = 1; seed <= 20; seed++) {
    const g = grid(seed), targets = narrowBandCollarTargets(g, "liquid"), r = random(1000 + seed);
    for (let i = 0; i < targets.length; i++) if (targets[i]! > 1) assert.ok(g.heat[i]! > 1);
    for (let k = 0; k < 2000; k++) {
      const heat = 2 * r(), tile = Math.floor(r() * targets.length), arrived = targets[tile]!;
      const after = narrowBandCollarArrival("liquid", heat, arrived);
      assert.ok(after <= heat);
      if (after > 1) { assert.ok(arrived >= after && g.heat[tile]! > 1, "erasure heat only where the tile is hot, and so seeded"); }
      // Support heat (<= 1) is kept, so a sample arriving in a seeded collar tile still supports the transfer.
      if (arrived > 0) assert.equal(after, Math.min(heat, Math.max(arrived, 1)));
      else assert.equal(after, Math.min(heat, 1));
    }
  }
});

// A flat or tilted surface; depth = signed distance (negative in liquid), in cells. Sites as seedSite places them
// (vec3f(c) + 0.5 corner + 0.5 jitter), the gate as seedCell/seedCells apply it, the retire test as the move does.
interface Plane { readonly n: [number, number, number]; readonly offset: number }
const depthAt = (s: Plane, p: readonly number[]) => s.n[0] * p[0]! + s.n[1] * p[1]! + s.n[2] * p[2]! - s.offset;
function seedColumn(s: Plane, bandDepth: number, density: NarrowBandCollarDensity, heat: number, jitter: () => number) {
  const out: { depth: number; mass: number; cell: number[] }[] = [];let gated = 0;
  for (let z = 0; z < 3; z++) for (let x = 0; x < 3; x++) for (let y = -12; y < 4; y++) {
    const c = [x, y, z], d = depthAt(s, [x + 0.5, y + 0.5, z + 0.5]);
    const target = narrowBandSeedTarget(d, true, density, heat);
    const sites = Array.from({ length: target }, (_, j) => {
      const k = narrowBandSeedSite(j, target), o = narrowBandCorner(k);
      return [x + 0.5 * o[0] + 0.5 * jitter(), y + 0.5 * o[1] + 0.5 * jitter(), z + 0.5 * o[2] + 0.5 * jitter()];
    });
    const accepted = sites.map(p => depthAt(s, p)).filter(dd => !(dd > 0 || dd < -bandDepth));
    if (d > 0.9 || d < -bandDepth - 0.5) { gated += accepted.length; continue; }
    for (const dd of accepted) out.push({ depth: dd, mass: 8 / target, cell: c });
  }
  return { seeds: out, gated };
}

test("fractional particleBandDepth: monotone seed counts, no seed past the band, seed and retire boundaries agree", () => {
  const depths = [2, 2.25, 2.5, 2.75, 3, 3.5, 4];
  const planes: Plane[] = [{ n: [0, 1, 0], offset: 0 }, { n: [0, 1, 0], offset: 0.37 }, { n: [0, 1, 0], offset: 0.5 }];
  const r = random(7);
  for (let k = 0; k < 6; k++) { const a = [r() - 0.5, 1, r() - 0.5], l = Math.hypot(...a); planes.push({ n: [a[0]! / l, a[1]! / l, a[2]! / l], offset: r() }); }
  for (const density of ["band", "deep"] as const) for (const heat of [2, 1]) for (const [p, plane] of planes.entries()) {
    let previous = -1;
    for (const bandDepth of depths) {
      // The same jitter for every depth: the depth decides which sites stay, nothing else.
      const { seeds, gated } = seedColumn(plane, bandDepth, density, heat, random(100 + p));
      assert.ok(seeds.length >= previous, `${density} heat ${heat} plane ${p}: ${seeds.length} seeds at depth ${bandDepth} after ${previous}`);
      previous = seeds.length;
      for (const s of seeds) {
        assert.ok(s.depth <= 0 && s.depth >= -bandDepth, "seeded inside the band");
        // The move retires a sample past the band (depth < -NB_BAND_DEPTH): no seed is retired where it lies.
        assert.ok(!(s.depth < -bandDepth));
      }
      // The cell gate is exact for a level surface: it never drops a site the site test keeps.
      if (plane.n[1] === 1) assert.equal(gated, 0, `level plane ${p} depth ${bandDepth}`);
    }
  }
});

test("collarDensity deep: a collar cell holds the band's eight units in four even-site samples", () => {
  for (const depth of [-0.2, -0.9, -1.4, -1.6, -2.5]) {
    // Hot tiles keep the band's density; the collar (heat in (0, 1]) takes four at every depth, sources keep the band's.
    assert.equal(narrowBandSeedTarget(depth, true, "deep", 2), narrowBandSeedTarget(depth, true, "band", 2));
    assert.equal(narrowBandSeedTarget(depth, true, "deep", 1), 4);
    assert.equal(narrowBandSeedTarget(depth, true, "deep", 0.3), 4);
    assert.equal(narrowBandSeedTarget(depth, true, "deep", 1, true), narrowBandSeedTarget(depth, true, "band", 1));
    for (const density of ["band", "deep"] as const) for (const heat of [2, 1]) {
      const target = narrowBandSeedTarget(depth, true, density, heat);
      assert.equal(target * (8 / target), 8, "a full cell is eight units in either density");
    }
  }
  // The deep sites are the even corners of the cell's 2x2x2 lattice: each half-cell beside a face holds two of them,
  // so a sample stands within half a cell of every face of a collar cell, as in the band's deep cells.
  for (const k of NARROW_BAND_DEEP_SITES) { const o = narrowBandCorner(k); assert.equal((o[0] + o[1] + o[2]) % 2, 0); }
  for (let axis = 0; axis < 3; axis++) for (const side of [0, 1]) {
    assert.equal(NARROW_BAND_DEEP_SITES.filter(k => narrowBandCorner(k)[axis] === side).length, 2);
  }
  // nbFinish splits a sample of mass m in a cell under its seed target when m * target > 8: a deep collar seed
  // (2 units, target 4) does not split where it was seeded; in a hot shallow cell (target 8) it splits back to band density.
  const splits = (mass: number, target: number, held: number) => held < target && mass * target > 8;
  assert.equal(splits(2, narrowBandSeedTarget(-0.4, true, "deep", 1), 0), false);
  assert.equal(splits(2, narrowBandSeedTarget(-2.4, true, "deep", 1), 0), false);
  assert.equal(splits(2, narrowBandSeedTarget(-0.4, true, "deep", 2), 3), true);
  assert.equal(splits(2, narrowBandSeedTarget(-2.4, true, "deep", 2), 3), false);
  // A shallow collar cell seeds every site above the surface test either way; deep halves its samples there.
  const level: Plane = { n: [0, 1, 0], offset: 0 };
  const band = seedColumn(level, 3, "band", 1, random(3)).seeds, deep = seedColumn(level, 3, "deep", 1, random(3)).seeds;
  const mass = (s: typeof band) => s.reduce((m, x) => m + x.mass, 0);
  assert.equal(mass(band), mass(deep));
  assert.ok(deep.length < band.length);
});
