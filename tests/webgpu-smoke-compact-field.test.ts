import assert from "node:assert/strict";
import test from "node:test";
import { FINE_LEVELSET_INVALID, FINE_LEVELSET_SAMPLE_FLAGS, planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";
import { compactOctreeFieldEvidenceIsAcceptable, reconstructCompactOctreeOccupancyField,
  compactOctreePublicationHeaderEvidence, type CompactOctreeFieldSnapshot } from "../tools/webgpu-smoke-compact-field";

const generation = 3;
const plan = planFineLevelSetBricks({
  domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1], finestCellWidth: 1,
  fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2,
});

function coarseHash(cell: number, size: number) {
  let value = (cell ^ Math.imul(size, 0x9e37_79b9)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb_352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846c_a68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

function setFloat(words: Uint32Array, index: number, value: number) {
  new Float32Array(words.buffer, words.byteOffset + index * 4, 1)[0] = value;
}

function snapshot(): CompactOctreeFieldSnapshot {
  const capacity = 4, coarseDirectory = new Uint32Array(8 + capacity * 8);
  coarseDirectory.set([OCTREE_POWER_COARSE_LEVELSET_VALID, generation, capacity, 1, 2, 1, 1], 0);
  setFloat(coarseDirectory, 7, 1);
  const cell = 0, size = 1, slot = coarseHash(cell, size) & (capacity - 1), base = 8 + slot * 8;
  coarseDirectory[base] = cell + 1; coarseDirectory[base + 1] = size;
  setFloat(coarseDirectory, base + 2, -0.5);
  setFloat(coarseDirectory, base + 3, -0.5);
  setFloat(coarseDirectory, base + 4, -0.5);
  coarseDirectory[base + 5] = OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite;
  return {
    plan, generation,
    hash: new Uint32Array(plan.hashCapacity * 2).fill(FINE_LEVELSET_INVALID),
    metadata: new Uint32Array(plan.maximumResidentBricks * 10),
    flags: new Uint32Array(plan.maximumResidentBricks * plan.samplesPerBrick),
    phi: new Float32Array(plan.maximumResidentBricks * plan.samplesPerBrick),
    worklist: new Uint32Array(5 + plan.maximumResidentBricks), coarseDirectory,
    coarseControl: new Uint32Array([0, 0xffff_ffff, 1, 1, 0, 0, 0, 8, 0, 1, 0, generation, OCTREE_POWER_COARSE_LEVELSET_VALID, 0, 0, 0]),
    topologyControl: new Uint32Array([0, 1, 1, 1, 1, 0, 1, 0]),
  };
}

test("compact smoke reconstruction returns a real spatial field from coarse leaves and positive-air complement", () => {
  const result = reconstructCompactOctreeOccupancyField(snapshot(), [2, 1, 1]);
  assert.deepEqual([...result.field], [1, 0]);
  assert.equal(result.fineSamples, 0);
  assert.equal(result.coarseSamples, 2 * 4 ** 3);
  assert.equal(result.positiveAirSamples, 4 ** 3);
});

test("current valid fine page overrides compact coarse phi for its base cell", () => {
  const current = snapshot();
  const key = 0, physicalId = 0;
  const slot = (Math.imul(key ^ (key >>> 16), 0x9e37_79b1) >>> 0) & (plan.hashCapacity - 1);
  current.hash[slot * 2] = key; current.hash[slot * 2 + 1] = physicalId;
  current.metadata.set([physicalId, key, generation], physicalId * 10);
  current.worklist.set([1, generation, 1, 1, 1, physicalId]);
  current.flags.fill(FINE_LEVELSET_SAMPLE_FLAGS.valid, 0, plan.samplesPerBrick);
  current.phi.fill(0.5, 0, plan.samplesPerBrick);
  const result = reconstructCompactOctreeOccupancyField(current, [2, 1, 1]);
  assert.deepEqual([...result.field], [0, 0]);
  assert.equal(result.fineSamples, 4 ** 3);
  assert.equal(result.coarseSamples, 4 ** 3);
});

test("required compact acceptance rejects a plausible coarse-only field", () => {
  const coarseOnly = snapshot();
  // Publish an otherwise current, clean, signed fine page but deliberately do
  // not place it in the fine hash. Reconstruction is therefore spatially
  // plausible and entirely coarse-backed; the required evidence must reject it.
  coarseOnly.metadata.set([0, 0, generation], 0);
  coarseOnly.worklist.set([1, generation, 1, 1, 1, 0]);
  coarseOnly.flags.fill(FINE_LEVELSET_SAMPLE_FLAGS.valid, 0, plan.samplesPerBrick);
  coarseOnly.phi.fill(-0.5, 0, plan.samplesPerBrick / 2);
  coarseOnly.phi.fill(0.5, plan.samplesPerBrick / 2, plan.samplesPerBrick);
  const result = reconstructCompactOctreeOccupancyField(coarseOnly, [2, 1, 1]);
  assert.equal(result.fineSamples, 0);
  assert.equal(result.coarseSamples, 2 * 4 ** 3);
  assert.equal(result.publicationValid, true);
  assert.equal(result.negativeValidSamples, plan.samplesPerBrick / 2);
  assert.equal(result.positiveValidSamples, plan.samplesPerBrick / 2);
  assert.equal(result.downstreamFinalizeReason, 0);
  assert.equal(compactOctreeFieldEvidenceIsAcceptable(result), false);
});

test("required compact validation rejects and exposes a downstream publication reason", () => {
  const volumeControl = new Uint32Array(16); volumeControl[0] = 2;
  const rejected: CompactOctreeFieldSnapshot = { ...snapshot(),
    transportControl: new Uint32Array([0, 0, 42, 0, 7, 4, 3, 3]),
    redistanceControl: new Uint32Array([0, 1, 14, 1]), volumeControl,
    faceBandControl: new Uint32Array([
      0, 0xffff_ffff, 41, 82, 164, generation, 0x8000_0000, 4, 2, 82, 0, 41, 7, 0, 0, 0,
    ]),
    faceBandTransitionControl: new Uint32Array([
      4, 17, 41, 9, 33, 0, 0, 36,
    ]),
    faceBandTransientPowerControl: new Uint32Array([
      8, 19, 41, 1_230, 301, 300, 40, generation, 0, 0, 0, 0, 0, 0, 0, 0,
    ]),
    faceBandPointFieldControl: new Uint32Array([
      8, 21, 41, generation, 40, 0, 7, 17,
    ]),
    faceBandPowerPublicationControl: new Uint32Array([
      0, 0xffff_ffff, 82, 17, 17, 17, generation, 9, 0x8000_0000, 0, 0, 0, 0, 0, 0, 0,
    ]),
    powerVelocityControl: new Uint32Array([0x8000_0000, 0xffff_ffff, 41, 82, 164, 41, 0, 9]),
    powerVelocitySampleControl: new Uint32Array([0x8000_0000, 0xffff_ffff, 42, 42, 20, 22, 0, 9]),
  };
  rejected.coarseDirectory[1] = generation - 1;
  rejected.coarseControl![11] = generation - 1;
  rejected.topologyControl!.set([16, 1, 1, 1, 1, 1, 1, 2]);
  assert.throws(() => reconstructCompactOctreeOccupancyField(rejected, [2, 1, 1]),
    /coarse\/fine generation mismatch/,
    "a rejected transaction must retain the prior mesh rather than reconstruct a new render publication");
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).downstreamFinalizeReason, 2);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).transportControl,
    [0, 0, 42, 0, 7, 4, 3, 3]);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).redistanceControl, [0, 1, 14, 1]);
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).volumeControl?.length, 16);
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).faceBandControl?.length, 16);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).faceBandTransitionControl,
    [4, 17, 41, 9, 33, 0, 0, 36]);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).faceBandTransientPowerControl,
    [8, 19, 41, 1_230, 301, 300, 40, generation, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).faceBandPointFieldControl,
    [8, 21, 41, generation, 40, 0, 7, 17]);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).faceBandPowerPublicationControl,
    [0, 0xffff_ffff, 82, 17, 17, 17, generation, 9, 0x8000_0000, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).powerVelocityControl?.length, 8);
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).powerVelocitySampleControl?.length, 8);
});

test("compact smoke reconstruction rejects a stale coarse/fine generation pair", () => {
  const stale = snapshot();
  stale.coarseDirectory[1] = generation - 1;
  assert.throws(() => reconstructCompactOctreeOccupancyField(stale, [2, 1, 1]), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /coarse\/fine generation mismatch/);
    assert.match(error.message, /"fineGeneration":3/);
    assert.match(error.message, /"coarseState":2147483648/);
    assert.match(error.message, /"coarseGeneration":2/);
    assert.match(error.message, /"coarseControlGeneration":3/);
    assert.match(error.message, /"coarseControlValid":2147483648/);
    assert.match(error.message, /"topologyPublished":1/);
    return true;
  });
});

test("compact smoke reconstruction rejects every rollback epoch", () => {
  const rolledBack = snapshot();
  rolledBack.coarseDirectory[1] = generation - 1;
  rolledBack.coarseControl![11] = generation - 1;
  rolledBack.topologyControl!.set([16, 1, 1, 1, 1, 1, 1, 2]);
  const key = 0, physicalId = 0;
  const slot = (Math.imul(key ^ (key >>> 16), 0x9e37_79b1) >>> 0) & (plan.hashCapacity - 1);
  rolledBack.hash[slot * 2] = key; rolledBack.hash[slot * 2 + 1] = physicalId;
  rolledBack.metadata.set([physicalId, key, generation], physicalId * 10);
  rolledBack.worklist.set([1, generation, 1, 1, 1, physicalId]);
  rolledBack.flags.fill(FINE_LEVELSET_SAMPLE_FLAGS.valid, 0, plan.samplesPerBrick);
  rolledBack.phi.fill(-0.5, 0, plan.samplesPerBrick / 2);
  rolledBack.phi.fill(0.5, plan.samplesPerBrick / 2, plan.samplesPerBrick);
  assert.throws(() => reconstructCompactOctreeOccupancyField(rolledBack, [2, 1, 1]),
    /coarse\/fine generation mismatch/,
    "rollback evidence is diagnostic-only and cannot create a new render publication");

  for (const invalid of [
    { topology: new Uint32Array([16, 1, 1, 1, 1, 1, 1, 0]), coarse: generation - 1 },
    { topology: new Uint32Array([17, 1, 1, 1, 1, 1, 1, 2]), coarse: generation - 1 },
    { topology: new Uint32Array([16, 1, 1, 1, 1, 1, 1, 2]), coarse: generation - 2 },
  ]) {
    const stale = { ...rolledBack, topologyControl: invalid.topology,
      coarseDirectory: rolledBack.coarseDirectory.slice() };
    stale.coarseDirectory[1] = invalid.coarse;
    assert.throws(() => reconstructCompactOctreeOccupancyField(stale, [2, 1, 1]),
      /coarse\/fine generation mismatch/);
  }
});

test("compact smoke reconstruction rejects a coarse epoch without current-slot proof", () => {
  const missing = { ...snapshot(), topologyControl: undefined };
  assert.throws(() => reconstructCompactOctreeOccupancyField(missing, [2, 1, 1]),
    /coarse\/fine generation mismatch/);
});

test("compact smoke reconstruction distinguishes invalid coarse state from generation mismatch", () => {
  const invalid = snapshot();
  invalid.coarseDirectory[0] = 0;
  assert.throws(() => reconstructCompactOctreeOccupancyField(invalid, [2, 1, 1]), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /coarse publication is not valid/);
    assert.match(error.message, /"fineGeneration":3/);
    assert.match(error.message, /"coarseState":0/);
    assert.match(error.message, /"coarseGeneration":3/);
    return true;
  });
});

test("compact smoke reconstruction rejects a stale GPU fine worklist against matching host/coarse tags", () => {
  const stale = snapshot();
  stale.worklist.set([1, generation - 1, 1, 1, 1, 0]);
  assert.throws(() => reconstructCompactOctreeOccupancyField(stale, [2, 1, 1]), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /fine publication is not valid\/current/);
    assert.match(error.message, /"fineGeneration":3/);
    assert.match(error.message, /"worklistGeneration":2/);
    assert.match(error.message, /"coarseGeneration":3/);
    return true;
  });
});

test("rejected publication evidence retains coarse failure and fine topology controls", () => {
  const rejected = snapshot();
  rejected.coarseDirectory[0] = 0;
  rejected.coarseControl!.set([32, 17, 41, 40], 0);
  rejected.topologyControl!.set([16, 5, 12, 11, 0, 1, 7, 2], 0);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected), {
    fineGeneration: 3,
    worklistActivePages: 0, worklistGeneration: 0, worklistInitialized: 0, worklistPublished: 0,
    coarseState: 0, coarseGeneration: 3, coarseHashCapacity: 4, coarseMaximumLeafSize: 1,
    coarseControlFlags: 32, coarseControlFirstErrorRow: 17, coarseControlRowCount: 41,
    coarseControlAdvectedRows: 40, coarseControlCorrectedRows: 0, coarseControlInterfaceRows: 1,
    coarseControlContributionCount: 0, coarseControlGeneration: 3,
    coarseControlValid: OCTREE_POWER_COARSE_LEVELSET_VALID,
    topologyFlags: 16, topologyInterfaceBricks: 5, topologyDesiredBricks: 12,
    topologyActivatedBricks: 11, topologyPublished: 0, topologyRolledBack: 1,
    topologyCapacityOrDilation: 7, downstreamFinalizeReason: 2,
  });
});
