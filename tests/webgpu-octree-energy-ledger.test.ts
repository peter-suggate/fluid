import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decodeOctreeEnergyLedger,
  OCTREE_ENERGY_LEDGER_RECORD_BYTES,
  OCTREE_ENERGY_LEDGER_STAGES,
  OCTREE_ENERGY_LEDGER_VALID,
  octreeEnergyLedgerWGSL,
  planOctreeEnergyLedger,
} from "../lib/webgpu-octree-energy-ledger";

test("energy-ledger plan is a fixed bounded ring with reusable reduction scratch", () => {
  const plan = planOctreeEnergyLedger(65, 129, 2, 256);
  assert.equal(plan.facePartialCount, 2);
  assert.equal(plan.finePartialCount, 3);
  assert.equal(plan.scratchBytes, 3 * 16);
  assert.equal(plan.commonSupportBytes, 129 * 12);
  assert.equal(plan.recordCount, 2 * OCTREE_ENERGY_LEDGER_STAGES.length);
  assert.equal(plan.recordBytes, plan.recordCount * OCTREE_ENERGY_LEDGER_RECORD_BYTES);
  assert.equal(plan.parameterStride, 256);
  assert.throws(() => planOctreeEnergyLedger(0, 1), RangeError);
  assert.throws(() => planOctreeEnergyLedger(1, 1, 0), RangeError);
});

test("energy-ledger decoder orders retained ring records and rejects stale slots", () => {
  const capacity = 2, stageCount = OCTREE_ENERGY_LEDGER_STAGES.length;
  const control = new Uint32Array([1, 3, capacity, stageCount]);
  const bytes = new ArrayBuffer(capacity * stageCount * OCTREE_ENERGY_LEDGER_RECORD_BYTES);
  const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
  const write = (slot: number, stage: number, step: number, kind: 0 | 1 | 2, value: number) => {
    const base = (slot * stageCount + stage) * 8;
    words.set([step, 100 + step, stage, OCTREE_ENERGY_LEDGER_VALID | kind], base);
    floats[base + 4] = value; floats[base + 5] = kind ? 0.25 : 0;
    words[base + 6] = 7; words[base + 7] = 0;
  };
  write(0, 0, 2, 0, 4.5);
  write(1, 6, 1, 1, 8.25);
  write(1, 8, 1, 2, 7.5);
  write(1, 7, 0, 1, 99); // Valid bit alone cannot resurrect an overwritten step.
  const decoded = decodeOctreeEnergyLedger(control.buffer, bytes);
  assert.equal(decoded.totalSteps, 3);
  assert.deepEqual(decoded.records.map(({ step, stage, kind, value }) => ({ step, stage, kind, value })), [
    { step: 1, stage: "preFineTransport", kind: "residentFinePotential", value: 8.25 },
    { step: 1, stage: "preFineTopologyCommon", kind: "commonFinePotential", value: 7.5 },
    { step: 2, stage: "oldFaceCapture", kind: "faceMetricKinetic", value: 4.5 },
  ]);
});

test("GPU ledger uses the pressure metric and a volume-weighted fine potential moment", () => {
  assert.match(octreeEnergyLedgerWGSL,
    /\.5\*f\.area\/\(f\.openFraction\*f\.inverseDistance\)\*f\.normalVelocity\*f\.normalVelocity/);
  assert.match(octreeEnergyLedgerWGSL, /energy=-dot\(p\.gravity,position\)\*volume/);
  assert.match(octreeEnergyLedgerWGSL, /atomicMax\(&control\[1\],p\.step\+1u\)/);
  assert.match(octreeEnergyLedgerWGSL, /if\(a\.x==0xffffffffu\)\{invalid=1u;/,
    "a malformed live worklist entry must not disappear from the ledger");
  assert.match(octreeEnergyLedgerWGSL, /\(sampleFlags\[index\]&1u\)==0u\)\{invalid=1u;/,
    "an invalid in-domain fine sample must not disappear from the ledger");
  assert.match(octreeEnergyLedgerWGSL,
    /captureFineCommonCandidates[\s\S]*commonKeys\[flat\]=key;commonLocalOrTarget\[flat\]=a\.y;commonPrePhi\[flat\]=value/);
  assert.match(octreeEnergyLedgerWGSL,
    /buildFineCommonSupport[\s\S]*lookupTargetBrick\(key\)[\s\S]*commonLocalOrTarget\[flat\]=index/);
  assert.match(octreeEnergyLedgerWGSL,
    /reduceFineCommonPotentialPartials[\s\S]*var value=commonPrePhi\[flat\][\s\S]*value=phi\[index\]/);
});

test("authoritative stage hooks are ordered and remain opt-in", () => {
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  const productionStart = octree.indexOf("private encodePowerAssemblyMirror");
  const positions = [
    "oldFaceCapture", "postRemap", "postGravity", "postSolidConstraint",
    "postProjection", "postFaceBandPublication",
  ].map((stage) => octree.indexOf(`\"${stage}\"`, productionStart));
  positions.forEach((position) => assert.ok(position >= 0));
  positions.slice(1).forEach((position, index) => assert.ok(position > positions[index]));
  assert.match(octree, /if \(this\.energyLedgerRequested && this\.powerPolicy\.authoritative/);
  assert.match(octree,
    /const oldPowerGeneration = this\.powerGeneration;[\s\S]*"oldFaceCapture", oldPowerGeneration/);
  assert.match(uniform, /energyLedger: options\.octree\.energyLedger/);
  assert.match(smoke, /FLUID_POWER_ENERGY_LEDGER === "1"/);
  assert.match(smoke, /await \(solver as GPUSolverInstance\)\.readPowerEnergyLedger\?\.\(\)/);
});

test("fine ledger brackets transport, topology, redistance, and correction", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  for (const stage of ["preFineTransport", "postFineTransport", "postFineTopology",
    "postFineRedistance", "postFineVolumeCorrection"] as const) {
    assert.match(source, new RegExp(`encodeFinePotential\\(encoder, \\"${stage}\\"`));
  }
  assert.match(source,
    /preFineTransport[\s\S]*transport\.encode[\s\S]*postFineTransport[\s\S]*publicationTopology\.encode[\s\S]*postFineTopology[\s\S]*publicationRedistance\.encode[\s\S]*postFineRedistance[\s\S]*publicationVolume\?\.encode[\s\S]*postFineVolumeCorrection/);
  assert.match(source,
    /postFineTransport[\s\S]*encodeFineCommonCapture[\s\S]*publicationTopology\.encode[\s\S]*encodeFineCommonTopologyPair[\s\S]*publicationRedistance\.encode[\s\S]*postFineRedistanceCommon/,
    "the immutable old sample capture must precede topology and its frozen support must survive redistance");
});
