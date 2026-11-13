import assert from "node:assert/strict";
import test from "node:test";
import {
  NARROW_BAND_SURFACE_SAMPLES, narrowBandCoarseSite, narrowBandLatticeCorner, narrowBandMassBits, narrowBandSampleRadii,
  narrowBandSampleRadiusWGSL, narrowBandSeedPlan, narrowBandSiteAxis, narrowBandSurfaceSampleScale, narrowBandSurfaceSamples,
  narrowBandSurfaceSamplesCheck, narrowBandSurfaceSite, narrowBandUnionShape,
} from "../lib/methods/uniform/uniform-narrow-band-sites";
import {
  NARROW_BAND_CENTRED_BOUND_WGSL, NARROW_BAND_SEED_DEPTH, NARROW_BAND_SURFACE_RADIUS, NARROW_BAND_WALL_BOUND_WGSL,
  narrowBandCentredBoundWGSL, narrowBandExpectedMassWGSL, narrowBandFineTransferSource, narrowBandParticleSurfaceWGSL,
  narrowBandParticleSurfaceWGSLFor, narrowBandSurfaceWGSL, narrowBandSurfaceWGSLFor, narrowBandTransferPropertiesSource, narrowBandWallBoundWGSL,
} from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import { narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { narrowBandSparseWGSL, narrowBandSparseWGSLFor } from "../lib/methods/uniform/uniform-narrow-band-sparse.wgsl";
import { particleSlabWGSL } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { particleBrickDepositWGSL } from "../lib/methods/uniform/uniform-particle-brick.wgsl";
import { narrowBandSeedRankShape, narrowBandSeedRankWGSL, narrowBandSeparationReach } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";

// The radius table the solver compiled before surfaceSamples (HEAD f111c8b3).
const OLD_RADIUS = `const NB_SAMPLE_RADIUS=array<f32,9>(0.0,0.25,0.31498,0.36056,0.39685,0.42750,0.45428,0.47823,0.5);
fn nbSampleRadius(m:f32)->f32{return NB_SAMPLE_RADIUS[min(u32(m),8u)];}`;

test("surfaceSamples parses 8 and 64 and throws on anything else", () => {
  assert.equal(narrowBandSurfaceSamples(undefined), 8);
  for (const v of [8, "8"]) assert.equal(narrowBandSurfaceSamples(v), 8);
  for (const v of [64, "64"]) assert.equal(narrowBandSurfaceSamples(v), 64);
  for (const v of [27, "27", 16, 0, "x", "", null, 8.5, "08x"]) assert.throws(() => narrowBandSurfaceSamples(v), /surfaceSamples must be one of 8, 64/);
  narrowBandSurfaceSamplesCheck(8, "retire");
  narrowBandSurfaceSamplesCheck(64, "merge");
  assert.throws(() => narrowBandSurfaceSamplesCheck(64, "retire"), /needs crowdedCells "merge"/);
});

test("the method's rail select and the near-settled values carry surfaceSamples", () => {
  const param = uniformNarrowBandMethod.params.find(p => p.key === "surfaceSamples");
  assert.ok(param && param.kind === "select");
  assert.equal(param.default, "8");
  assert.equal(param.update, "solver");
  assert.deepEqual(param.options.map(o => o.value), NARROW_BAND_SURFACE_SAMPLES.map(String));
  assert.equal(uniformNearSettled({}).surfaceSamples, 8);
  assert.equal(uniformNearSettled({ surfaceSamples: "64", crowdedCells: "merge" }).surfaceSamples, 64);
  assert.throws(() => uniformNearSettled({ surfaceSamples: "64", crowdedCells: "retire" }), /needs crowdedCells "merge"/);
  assert.throws(() => uniformNearSettled({ surfaceSamples: "27" }), /surfaceSamples must be one of/);
});

test("the surface layer's n^3 sites are distinct and inside their cell, jittered anywhere in their sub-cell", () => {
  let seed = 7;
  const random = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  for (const samples of NARROW_BAND_SURFACE_SAMPLES) {
    const n = narrowBandSiteAxis(samples);
    assert.equal(n ** 3, samples);
    for (const jitter of [[0, 0, 0], [0.999999, 0.999999, 0.999999], [0.5, 0.5, 0.5]] as const) {
      const cells = new Set<string>();
      for (let k = 0; k < samples; k++) {
        const j = jitter[0] === 0.5 ? [random(), random(), random()] as const : jitter;
        const p = narrowBandSurfaceSite(k, samples, j);
        for (const x of p) assert.ok(x >= 0 && x < 1, `site ${k} of ${samples} at ${p} leaves its cell`);
        cells.add(p.map(x => Math.floor(x * n)).join(","));
        assert.deepEqual(p.map(x => Math.floor(x * n)), narrowBandLatticeCorner(k, n));
      }
      assert.equal(cells.size, samples);
    }
  }
  // The deep band's coarse sites: all eight, or the even four.
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 7].map(j => narrowBandCoarseSite(j, 8)), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual([0, 1, 2, 3].map(j => narrowBandCoarseSite(j, 4)), [0, 3, 5, 6]);
});

test("a full cell holds S units at every depth: the seed plan and its masses", () => {
  for (const samples of NARROW_BAND_SURFACE_SAMPLES) for (const graded of [false, true]) for (const depth of [-0.5, -1.4, -1.5, -2, -3.5]) {
    const plan = narrowBandSeedPlan(depth, graded, samples);
    assert.equal(plan.target * plan.mass, samples);
    assert.ok(Number.isInteger(plan.mass) && plan.mass >= 1 && plan.mass <= samples);
    assert.equal(plan.fine, plan.target === samples);
    if (samples === 64) assert.equal(plan.fine, depth > -1.5);
    if (depth <= -1.5) assert.equal(plan.target, graded ? 4 : 8);
  }
  assert.deepEqual(narrowBandSeedPlan(-1, false, 64), { target: 64, mass: 1, fine: true });
  assert.deepEqual(narrowBandSeedPlan(-2, false, 64), { target: 8, mass: 8, fine: false });
  assert.deepEqual(narrowBandSeedPlan(-2, true, 64), { target: 4, mass: 16, fine: false });
  assert.deepEqual(narrowBandSeedPlan(-1, false, 64, true), { target: 4, mass: 16, fine: false });
  assert.deepEqual(narrowBandSeedPlan(-1, true, 8), { target: 8, mass: 1, fine: true });
});

test("radii cbrt(m)/(2n): unit samples 1/n apart, a whole cell 1/2 at any S, the old table at 8", () => {
  assert.equal(narrowBandSampleRadiusWGSL(8), OLD_RADIUS);
  for (const samples of NARROW_BAND_SURFACE_SAMPLES) {
    const r = narrowBandSampleRadii(samples), n = narrowBandSiteAxis(samples);
    assert.equal(r.length, samples + 1);
    assert.equal(r[0], 0);
    assert.ok(Math.abs(2 * r[1]! - 1 / n) < 1e-12);
    assert.ok(Math.abs(r[samples]! - 0.5) < 1e-12);
    for (let m = 1; m <= samples; m++) assert.ok(r[m]! > r[m - 1]!);
    const wgsl = narrowBandSampleRadiusWGSL(samples);
    assert.match(wgsl, new RegExp(`array<f32,${samples + 1}>`));
    assert.match(wgsl, new RegExp(`min\\(u32\\(m\\),${samples}u\\)`));
    const table = wgsl.slice(wgsl.indexOf("(", wgsl.indexOf(">")) + 1, wgsl.indexOf(")")).split(",").map(Number);
    table.forEach((v, m) => assert.ok(Math.abs(v - r[m]!) < 1e-5, `radius ${m} of ${samples}: ${v} against ${r[m]}`));
  }
  assert.equal(narrowBandSeparationReach("pairs", true, 64), 1);
  assert.equal(narrowBandSeparationReach("pairs", true, 8), 1);
  assert.equal(narrowBandSeparationReach("pairs", false, 8), 0.5);
  assert.equal(narrowBandSeparationReach("off", true, 64), 0);
});

test("the records scale by the surface layer's share and the mass bits by log2 S", () => {
  for (const depth of [2, 2.5, 3, 4]) assert.equal(narrowBandSurfaceSampleScale(8, depth), 1);
  assert.equal(narrowBandSurfaceSampleScale(64, 2), 8);
  assert.ok(Math.abs(narrowBandSurfaceSampleScale(64, 3) - (2 * 64 + 8) / 24) < 1e-12);
  assert.equal(narrowBandSurfaceSampleScale(64, 4), 4.5);
  assert.equal(narrowBandMassBits(8), 3);
  assert.equal(narrowBandMassBits(64), 6);
  // The seed rank keys a cell's eight octants under 64.
  assert.deepEqual(narrowBandSeedRankShape(8 * 32768, 512), { bits: 7, levels: 3, keyBits: 18 });
  assert.throws(() => narrowBandSeedRankShape(8 * 2 ** 28, 1 << 20), /exceed the seed list's 30-bit key/);
});

test("the union keeps its surface where a flat seeded layer puts it: the shape at n = 2 is the solver's", () => {
  const two = narrowBandUnionShape(2), four = narrowBandUnionShape(4);
  assert.equal(two.radius, NARROW_BAND_SURFACE_RADIUS);
  assert.equal(two.seedDepth, NARROW_BAND_SEED_DEPTH);
  assert.equal(two.offset2, 0.125);
  assert.equal(two.spacing2, 0.25);
  assert.equal(four.radius, 0.75);
  assert.equal(four.offset2, 1 / 32);
  assert.equal(four.spacing2, 1 / 16);
  assert.ok(Math.abs(four.seedDepth - Math.sqrt(0.75 ** 2 - 1 / 32)) < 1e-15);
  const fine = narrowBandParticleSurfaceWGSLFor(4);
  assert.match(fine, /const NB_SURFACE_RADIUS=0\.75;/);
  assert.match(fine, /nearest2-0\.03125/);
  assert.match(fine, /nearest2<=0\.0625/);
  assert.match(fine, /min\(nbBinCount\(umCellAt\(vec3u\(c\)\)\),64u\)/);
  assert.match(fine, /32\.0\*f32\(open\)/);
});

test("every generator emits the solver's own text at eight a cell", () => {
  for (const apic of [false, true]) for (const merge of [false, true]) {
    assert.equal(narrowBandParticleWGSL(apic, merge, 3), narrowBandParticleWGSL(apic, merge));
    assert.equal(particleBrickDepositWGSL(apic, { joins: 1, allocation: 2 }, "bins", merge, 1, 3), particleBrickDepositWGSL(apic, { joins: 1, allocation: 2 }, "bins", merge, 1));
    for (const graded of [false, true]) {
      assert.equal(narrowBandExpectedMassWGSL("d", graded, merge, 8), narrowBandExpectedMassWGSL("d", graded, merge));
      assert.equal(narrowBandTransferPropertiesSource(graded, "bins", merge, 8), narrowBandTransferPropertiesSource(graded, "bins", merge));
      assert.equal(narrowBandFineTransferSource(true, apic, graded, merge, 8), narrowBandFineTransferSource(true, apic, graded, merge));
    }
  }
  assert.equal(particleSlabWGSL(1000, 100, 4, 900, 3), particleSlabWGSL(1000, 100, 4, 900));
  assert.equal(narrowBandSparseWGSLFor(8), narrowBandSparseWGSL);
  assert.equal(narrowBandCentredBoundWGSL(8), NARROW_BAND_CENTRED_BOUND_WGSL);
  assert.equal(narrowBandWallBoundWGSL(8), NARROW_BAND_WALL_BOUND_WGSL);
  assert.equal(narrowBandParticleSurfaceWGSLFor(2), narrowBandParticleSurfaceWGSL);
  assert.equal(narrowBandSurfaceWGSLFor(8, false), narrowBandSurfaceWGSL);
  assert.equal(narrowBandSurfaceWGSLFor(8, true), narrowBandSurfaceWGSL.replace(NARROW_BAND_CENTRED_BOUND_WGSL, NARROW_BAND_WALL_BOUND_WGSL));
  const rank = narrowBandSeedRankShape(32768, 512);
  assert.equal(narrowBandSeedRankWGSL(rank, 5000, 8), narrowBandSeedRankWGSL(rank, 5000));
  // The old expected mass and its graded form.
  assert.equal(narrowBandExpectedMassWGSL("d", true, false), "max(1.0,select(8.0,4.0,d< -1.0)*clamp(0.5-d,0.0,1.0))");
  assert.equal(narrowBandExpectedMassWGSL("d", true, true), "max(1.0,8.0*clamp(0.5-d,0.0,1.0))");
});

test("sixty-four a cell keeps a sample's mass in six bits, clear of the age and the seed sentinel", () => {
  const flip = narrowBandParticleWGSL(false, true, 6);
  assert.match(flip, /&0x7fffffc0u/);
  assert.match(flip, /&63u\)\+1u\)/);
  assert.match(flip, /&0xffffffc0u\)\|\(\(u32\(mass\)-1u\)&63u\)/);
  const slab = particleSlabWGSL(1000, 100, 4, 900, 6), first = /NB_SEED_FIRST:u32=0x([0-9a-f]+)u/.exec(slab);
  assert.ok(first);
  assert.equal(Number.parseInt(first[1]!, 16) & 63, 0);
  assert.equal(Number.parseInt(first[1]!, 16), 0x7f7fffc0);
  assert.match(particleBrickDepositWGSL(false, { joins: 1, allocation: 2 }, "bins", true, 1, 6), /&0x7fffffc0u\)\|\(u32\(nbParticleMass\(s\)\)-1u\)/);
  assert.equal(narrowBandExpectedMassWGSL("d", true, true, 64), "max(8.0,64.0*clamp(0.5-d,0.0,1.0))");
  assert.match(narrowBandSparseWGSLFor(64), /fn nbSparseAge\(i:u32,p:vec3f,age:f32,mass:f32\)/);
  assert.match(narrowBandSparseWGSLFor(64), /f32\(params\.live\.y\)\*8\.0\/mass/);
  const rank = narrowBandSeedRankWGSL(narrowBandSeedRankShape(8 * 32768, 512), 5000, 64);
  assert.match(rank, /nbSeedLimit\(\)/);
  assert.match(rank, /<<3u\)\|\(links\[nbSeedList\(\)\+nbSeedLimit\(\)\+slot\]&7u\)/);
});
