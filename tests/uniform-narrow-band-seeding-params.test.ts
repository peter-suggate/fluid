import assert from "node:assert/strict";
import test from "node:test";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { narrowBandFlipValues, uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { narrowBandRetireCounting } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { UNIFORM_PRESSURE_ARENA_SCHEDULE } from "../lib/methods/uniform/uniform-pressure-arena";
import { narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { particleSlabWGSL } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { UNIFORM_REGION_HEAT_STATE_WORDS, UNIFORM_REGION_HEAT_STILL_H, UNIFORM_REGION_HEAT_STILL_ROUNDOFF } from "../lib/methods/uniform/uniform-region-heat";
import { NARROW_BAND_CENTRED_BOUND_WGSL, NARROW_BAND_WALL_BOUND_WGSL, narrowBandSurfaceWGSL } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";

test("seedScoring and outerSeeding default to scoring every surface tile and reseeding uncovered layers", () => {
  assert.equal(uniformNearSettled({}).seedScoring, "every");
  assert.equal(uniformNearSettled({ seedScoring: "bogus" }).seedScoring, "every");
  assert.equal(uniformNearSettled({ seedScoring: "unshaped" }).seedScoring, "unshaped");
  assert.equal(uniformNearSettled({}).outerSeeding, "uncovered");
  assert.equal(uniformNearSettled({ outerSeeding: "bogus" }).outerSeeding, "uncovered");
  assert.equal(uniformNearSettled({ outerSeeding: "retain" }).outerSeeding, "retain");
  const values = narrowBandFlipValues({});
  assert.equal(values.seedScoring, "every");
  assert.equal(values.outerSeeding, "uncovered");
  for (const key of ["seedScoring", "outerSeeding"]) {
    const param = uniformNarrowBandMethod.params.find(p => p.key === key);
    assert.ok(param && param.kind === "select" && param.update === "solver", `${key} is a solver-restarting select`);
  }
});

test("retirement counting is a diagnostic that is off unless a probe turns it on", () => {
  assert.equal(narrowBandRetireCounting.on, false);
});

test("heatExpiry and sampleBound default to retiring only held liquid and the bound against walls", () => {
  assert.equal(uniformNearSettled({}).heatExpiry, "held");
  assert.equal(uniformNearSettled({ heatExpiry: "anywhere" }).heatExpiry, "anywhere");
  assert.equal(uniformNearSettled({}).sampleBound, "walls");
  assert.equal(uniformNearSettled({ sampleBound: "centred" }).sampleBound, "centred");
  const values = narrowBandFlipValues({});
  assert.equal(values.heatExpiry, "held");
  assert.equal(values.sampleBound, "walls");
});

test("the wall bound replaces the centred one once, and keeps its expression where all eight cells are open", () => {
  assert.equal(narrowBandSurfaceWGSL.split(NARROW_BAND_CENTRED_BOUND_WGSL).length - 1, 1, "particleSurfaceFinish holds the centred bound once");
  const centred = "particle=max(particle,min(bulk,1.0-f32(held)/(4.0*f32(open))));";
  assert.ok(NARROW_BAND_CENTRED_BOUND_WGSL.includes(`if(open>0u){${centred}}`));
  assert.ok(NARROW_BAND_WALL_BOUND_WGSL.includes(`if(open==8u){${centred}}`));
  assert.match(NARROW_BAND_WALL_BOUND_WGSL, /bound=select\(-sparse,1\.0-max\(low,high\)-sparse,denseOpen\);/);
});

test("nativeSmoothing defaults to composite, relaxing the native rows with the band each cycle", () => {
  assert.equal(uniformNearSettled({}).nativeSmoothing, "composite");
  assert.equal(uniformNearSettled({ nativeSmoothing: "bogus" }).nativeSmoothing, "composite");
  assert.equal(uniformNearSettled({ nativeSmoothing: "coarse" }).nativeSmoothing, "coarse");
  assert.equal(UNIFORM_PRESSURE_ARENA_SCHEDULE.nativeSweeps, 0, "the schedule's own default is the level 2 correction alone; the frame sets it");
  assert.equal(narrowBandFlipValues({}).nativeSmoothing, "composite");
  const param = uniformNarrowBandMethod.params.find(p => p.key === "nativeSmoothing");
  assert.ok(param && param.kind === "select" && param.update === "solver");
});

test("coldSurface still holds a surface COLD only within round-off, and is the default", () => {
  assert.equal(uniformNearSettled({}).coldSurface, "still");
  assert.equal(uniformNearSettled({ coldSurface: "warm" }).coldSurface, "warm");
  assert.equal(uniformNearSettled({ coldSurface: "still" }).coldSurface, "still");
  assert.equal(uniformNearSettled({ coldSurface: "budget" }).coldSurface, "budget");
  assert.equal(uniformNearSettled({ coldSurface: "bogus" }).coldSurface, "still");
  const param = uniformNarrowBandMethod.params.find(p => p.key === "coldSurface");
  assert.ok(param && param.kind === "select" && param.default === "still" && param.options.some(o => o.value === "still"));
  assert.equal(UNIFORM_REGION_HEAT_STILL_ROUNDOFF, 256);
  assert.equal(UNIFORM_REGION_HEAT_STILL_H, 0.01);
  assert.equal(UNIFORM_REGION_HEAT_STATE_WORDS, 19);
});

test("crowdedCells defaults to merging the excess, and the merge carries mass in the age word's low bits", () => {
  assert.equal(uniformNearSettled({}).crowdedCells, "merge");
  assert.equal(uniformNearSettled({ crowdedCells: "bogus" }).crowdedCells, "merge");
  assert.equal(uniformNearSettled({ crowdedCells: "retire" }).crowdedCells, "retire");
  assert.equal(narrowBandFlipValues({}).crowdedCells, "merge");
  const param = uniformNarrowBandMethod.params.find(p => p.key === "crowdedCells");
  assert.ok(param && param.kind === "select" && param.update === "solver");
  const retire = narrowBandParticleWGSL(false);
  assert.ok(retire.includes("fn nbParticleMass(p:Particle)->f32{return 1.0;}"));
  assert.ok(retire.includes("&0x7fffffffu") && !retire.includes("0x7ffffff8u"), "retire keeps every age bit");
  const merge = narrowBandParticleWGSL(false, true);
  assert.ok(merge.includes("(bitcast<u32>(p.before.w)&7u)+1u"), "a FLIP sample's mass is its low three bits plus one");
  assert.ok(merge.includes("&0x7ffffff8u"), "ages leave the mass bits");
  const seed = /const NB_SEED_FIRST:u32=(0x[0-9a-f]+)u;/.exec(particleSlabWGSL(1024, 4096));
  assert.ok(seed);
  assert.equal(Number(seed[1]) & 7, 0, "a seed's sentinel age holds mass 1");
});

test("sample separation, whole-cell inner reseeding and uncovered activation are the defaults", () => {
  const settled = uniformNearSettled({});
  assert.equal(settled.sampleSeparation, "pairs");
  assert.equal(settled.innerReseed, "cells");
  assert.equal(settled.activationSeeding, "uncovered");
  assert.equal(settled.particleDensity, "graded");
  assert.equal(uniformNearSettled({ sampleSeparation: "bogus" }).sampleSeparation, "pairs");
  assert.equal(uniformNearSettled({ sampleSeparation: "off" }).sampleSeparation, "off");
  assert.equal(uniformNearSettled({ innerReseed: "sites" }).innerReseed, "sites");
  assert.equal(uniformNearSettled({ activationSeeding: "band" }).activationSeeding, "band");
  assert.equal(uniformNearSettled({ particleDensity: "uniform" }).particleDensity, "uniform");
  const values = narrowBandFlipValues({});
  assert.equal(values.sampleSeparation, "pairs");
  assert.equal(values.innerReseed, "cells");
  assert.equal(values.activationSeeding, "uncovered");
  assert.equal(values.particleDensity, "graded");
  for (const [key, value] of [["sampleSeparation", "pairs"], ["innerReseed", "cells"], ["activationSeeding", "uncovered"], ["particleDensity", "graded"]] as const) {
    const param = uniformNarrowBandMethod.params.find(p => p.key === key);
    assert.ok(param && param.kind === "select" && param.update === "solver" && param.default === value, `${key} is a solver-restarting select defaulting to ${value}`);
  }
});
