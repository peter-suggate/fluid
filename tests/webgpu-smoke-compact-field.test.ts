import assert from "node:assert/strict";
import test from "node:test";
import { FINE_LEVELSET_SAMPLE_FLAGS, planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";
import { compactOctreeFieldEvidenceIsAcceptable, reconstructCompactOctreeOccupancyField,
  compactOctreePublicationHeaderEvidence, type CompactOctreeFieldSnapshot } from "../tools/webgpu-smoke-compact-field";

const generation = 3;
const plan = planFineLevelSetBricks({
  domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1], finestCellWidth: 1,
  fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2,
});

function setFloat(words: Uint32Array, index: number, value: number) {
  new Float32Array(words.buffer, words.byteOffset + index * 4, 1)[0] = value;
}

function snapshot(): CompactOctreeFieldSnapshot {
  const capacity = 4, coarseDirectory = new Uint32Array(8 + capacity * 8);
  coarseDirectory.set([OCTREE_POWER_COARSE_LEVELSET_VALID, generation, 1, 1, 2, 1, 1], 0);
  setFloat(coarseDirectory, 7, 1);
  const cell = 0, size = 1, base = 8;
  coarseDirectory[base] = cell + 1; coarseDirectory[base + 1] = size;
  setFloat(coarseDirectory, base + 2, -0.5);
  setFloat(coarseDirectory, base + 3, -0.5);
  setFloat(coarseDirectory, base + 4, -0.5);
  coarseDirectory[base + 5] = OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite;
  return {
    plan, generation,
    metadata: new Uint32Array(plan.maximumResidentBricks * 10),
    flags: new Uint32Array(plan.maximumResidentBricks * plan.samplesPerBrick),
    phi: new Float32Array(plan.maximumResidentBricks * plan.samplesPerBrick),
    worklist: new Uint32Array(7 + plan.maximumResidentBricks + plan.logicalBrickCount
      + (plan.includeHalo27 ? 27 * plan.maximumResidentBricks : 0)), coarseDirectory,
    coarseControl: new Uint32Array([0, 0xffff_ffff, 1, 1, 0, 0, 8, 0, 1, 0, generation, OCTREE_POWER_COARSE_LEVELSET_VALID, 0, 0, 0, 0]),
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
  current.metadata.set([physicalId, key, generation], physicalId * 10);
  current.worklist.set([generation, 1, plan.maximumResidentBricks, 3, 1, 1, 1, physicalId]);
  current.worklist[7 + plan.maximumResidentBricks + key] = physicalId;
  current.flags.fill(FINE_LEVELSET_SAMPLE_FLAGS.valid, 0, plan.samplesPerBrick);
  current.phi.fill(0.5, 0, plan.samplesPerBrick);
  const result = reconstructCompactOctreeOccupancyField(current, [2, 1, 1]);
  assert.deepEqual([...result.field], [0, 0]);
  assert.equal(result.fineSamples, 4 ** 3);
  assert.equal(result.coarseSamples, 4 ** 3);
});

test("required compact acceptance rejects a plausible coarse-only field", () => {
  const coarseOnly = snapshot();
  // A spatially plausible compact-coarse field is not sufficient evidence
  // when no current sorted fine directory has been published.
  const result = reconstructCompactOctreeOccupancyField(coarseOnly, [2, 1, 1]);
  assert.equal(result.fineSamples, 0);
  assert.equal(result.coarseSamples, 2 * 4 ** 3);
  assert.equal(result.publicationValid, false);
  assert.equal(result.negativeValidSamples, 0);
  assert.equal(result.positiveValidSamples, 0);
  assert.equal(result.downstreamFinalizeReason, 0);
  assert.equal(compactOctreeFieldEvidenceIsAcceptable(result), false);
});

test("required compact validation rejects and exposes a downstream publication reason", () => {
  const volumeControl = new Uint32Array(16); volumeControl[0] = 2;
  const rejected: CompactOctreeFieldSnapshot = { ...snapshot(),
    transportControl: new Uint32Array([0, 0, 42, 0, 7, 4, 3, 3]),
    redistanceControl: new Uint32Array([
      2_388, 26_436_288, 5_804, 0, 4, 13_337, 188_108, 4_096, 4_072,
    ]),
    volumeControl,
  };
  rejected.coarseDirectory[1] = generation - 1;
  rejected.coarseControl![10] = generation - 1;
  rejected.topologyControl!.set([16, 1, 1, 1, 1, 1, 1, 2]);
  const reconstruction = reconstructCompactOctreeOccupancyField(rejected, [2, 1, 1]);
  assert.equal(reconstruction.publicationValid, false);
  assert.equal(compactOctreeFieldEvidenceIsAcceptable(reconstruction), false,
    "a rollback still requires a complete current fine-SPGrid publication");
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).downstreamFinalizeReason, 2);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected).transportControl,
    [0, 0, 42, 0, 7, 4, 3, 3]);
  const redistanceEvidence = compactOctreePublicationHeaderEvidence(rejected);
  assert.deepEqual(redistanceEvidence.redistanceControl,
    [2_388, 26_436_288, 5_804, 0, 4, 13_337, 188_108, 4_096, 4_072]);
  assert.equal(redistanceEvidence.redistanceUnresolvedCells, 2_388);
  assert.equal(redistanceEvidence.redistanceMaximumResidualScaled, 26_436_288);
  assert.equal(redistanceEvidence.redistanceSeedCount, 5_804);
  assert.equal(redistanceEvidence.redistanceCommitted, false);
  assert.equal(redistanceEvidence.redistanceFlags, 4);
  assert.equal(redistanceEvidence.redistanceFirstError, 13_337);
  assert.equal(redistanceEvidence.redistanceAcceptedCells, 188_108);
  assert.equal(redistanceEvidence.redistanceInitialPages, 4_096);
  assert.equal(redistanceEvidence.redistanceFinalPages, 4_072);
  assert.equal(compactOctreePublicationHeaderEvidence(rejected).volumeControl?.length, 16);
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

test("compact smoke reconstruction accepts an explicit retained octree with a current fine SPGrid", () => {
  const rolledBack = snapshot();
  rolledBack.coarseDirectory[1] = generation - 1;
  rolledBack.coarseControl![10] = generation - 1;
  rolledBack.topologyControl!.set([16, 1, 1, 1, 1, 1, 1, 2]);
  const key = 0, physicalId = 0;
  rolledBack.metadata.set([physicalId, key, generation], physicalId * 10);
  rolledBack.worklist.set([generation, 1, plan.maximumResidentBricks, 3, 1, 1, 1, physicalId]);
  rolledBack.worklist[7 + plan.maximumResidentBricks + key] = physicalId;
  rolledBack.flags.fill(FINE_LEVELSET_SAMPLE_FLAGS.valid, 0, plan.samplesPerBrick);
  rolledBack.phi.fill(-0.5, 0, plan.samplesPerBrick / 2);
  rolledBack.phi.fill(0.5, plan.samplesPerBrick / 2, plan.samplesPerBrick);
  // Aanjaneya et al. 2017 Section 5
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) explicitly uses a
  // background octree and a separately rebuilt fine SPGrid. The rollback is
  // provenance for retaining the valid octree, not a reason to hide the fine
  // interface that advanced independently.
  const retained = reconstructCompactOctreeOccupancyField(rolledBack, [2, 1, 1]);
  assert.equal(retained.coarseGeneration, generation - 1);
  assert.equal(retained.retainedCoarseAuthority, true);
  assert.equal(compactOctreeFieldEvidenceIsAcceptable(retained), true);

  for (const invalid of [
    { topology: new Uint32Array([16, 1, 1, 1, 1, 1, 1, 0]), coarse: generation - 1 },
    { topology: new Uint32Array([0x20, 1, 1, 1, 1, 1, 1, 2]), coarse: generation - 1 },
    { topology: new Uint32Array([16, 1, 1, 1, 1, 1, 1, 0x10]), coarse: generation - 2 },
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
  stale.worklist.set([generation - 1, 1, plan.maximumResidentBricks, 3, 1, 1, 1, 0]);
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
  const rejected: CompactOctreeFieldSnapshot = {
    ...snapshot(),
    mgpcgControl: new Uint32Array([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 17, 0x7fc0_0000, 0, 0, 0]),
  };
  rejected.coarseDirectory[0] = 0;
  rejected.coarseControl!.set([32, 17, 41, 40], 0);
  rejected.topologyControl!.set([16, 5, 12, 11, 0, 1, 7, 2], 0);
  assert.deepEqual(compactOctreePublicationHeaderEvidence(rejected), {
    fineGeneration: 3,
    worklistActivePages: 0, worklistGeneration: 0, worklistInitialized: 0, worklistPublished: 0,
    coarseState: 0, coarseGeneration: 3, coarseRowCount: 1, coarseMaximumLeafSize: 1,
    coarseControlFlags: 32, coarseControlFirstErrorRow: 17, coarseControlRowCount: 41,
    coarseControlAdvectedRows: 40, coarseControlCorrectedRows: 0, coarseControlInterfaceRows: 1,
    coarseControlContributionCount: 0, coarseControlGeneration: 3,
    coarseControlValid: OCTREE_POWER_COARSE_LEVELSET_VALID,
    topologyFlags: 16, topologyInterfaceBricks: 5, topologyDesiredBricks: 12,
    topologyActivatedBricks: 11, topologyPublished: 0, topologyRolledBack: 1,
    topologyCapacityOrDilation: 7, downstreamFinalizeReason: 2,
    mgpcgControl: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 17, 0x7fc0_0000, 0, 0, 0],
  });
});
