import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_PHASE1_BIT_EXACT_SCHEMA,
  assertSparseCM12Phase1ArmBitExact,
  assertSparseCM12Phase1PacketChurn,
  assertSparseCM12Phase1TransportReceipt,
  sparseCM12Phase1CanonicalJSON,
  sparseCM12Phase1FieldReceipt,
  sparseCM12Phase1TopologyReceipt,
  sparseCM12Phase1TrajectorySha256,
  type SparseCM12Phase1ArmReceipt,
} from "../tools/sparse-cm12-phase1-bit-exact-receipts";

const fields = (bits = 0x3f80_0000) => {
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = bits;
  const value = new Float32Array(buffer);
  return { density: value, gamma: value, velocity: value,
    pressure: value, divergence: value };
};

const hash = "0".repeat(64);
const transport = (overrides: Record<string, number | string> = {}) => ({
  stencilCellsSha256: hash, stencilWeightBitsSha256: hash,
  departureBitsSha256: hash, betaFixedSha256: hash,
  deficitDensityFixedSha256: hash, deficitGammaFixedSha256: hash,
  massReceiptSha256: hash, packetSha256: hash, publishedPacketSha256: hash,
  packetCount: 1,
  packetCellCount: 64, duplicateScatterCellCount: 0, omittedAcceptedCellCount: 0,
  acceptedTopologyGeneration: 1, transportTopologyGeneration: 1,
  frameGeneration: 1, packetGeneration: 1, publishedPacketGeneration: 1,
  effectiveVelocityTopologyGeneration: 1, effectiveVelocityFrameGeneration: 1,
  ...overrides,
});

const arm = (bits = 0x3f80_0000): SparseCM12Phase1ArmReceipt => {
  const checkpoint = {
    step: 0,
    fields: sparseCM12Phase1FieldReceipt(fields(bits)),
    scalarResult: { phase: 1, fault: 0, executedWork: 7, topologyGeneration: 1 },
    frameControl: { phase: 1, fault: 0, acceptedGeneration: 1 },
    scalarResultSha256: "scalar",
    frameControlSha256: "frame",
    acceptedTopologySha256: "topology",
    acceptedLeafStateSha256: "accepted",
    acceptedTopologyRecords: 1,
    topologyPreparationScheduled: 0,
    transport: transport(),
  };
  return {
    schema: SPARSE_CM12_PHASE1_BIT_EXACT_SCHEMA,
    scene: "mini64",
    experiment: "baseline",
    dimensions: [64, 64, 64],
    brickFineResolution: 8,
    presentationPageResolution: 8,
    dt_s: 0.004,
    checkpoints: [checkpoint],
    trajectorySha256: sparseCM12Phase1TrajectorySha256([checkpoint]),
  };
};

test("Phase-1 field receipt hashes raw Float32 payload bits", () => {
  const canonicalNaN = sparseCM12Phase1FieldReceipt(fields(0x7fc0_0000));
  const alternateNaN = sparseCM12Phase1FieldReceipt(fields(0x7fc0_0001));
  assert.notEqual(canonicalNaN.density.sha256, alternateNaN.density.sha256);
});

test("Phase-1 topology receipt is key ordered and includes candidate churn", () => {
  const records = [{ key: 9, active: true, acceptedResolution: 8,
    candidateResolution: 4, candidateStatus: 1, candidateEpoch: 3,
    topologyPreparationScheduled: true, topologyPreparationEpoch: 3 },
  { key: 2, active: false, acceptedResolution: 1,
    candidateResolution: 1, candidateStatus: 0, candidateEpoch: 0,
    topologyPreparationScheduled: false, topologyPreparationEpoch: 0 }];
  const forward = sparseCM12Phase1TopologyReceipt(records);
  const reverse = sparseCM12Phase1TopologyReceipt([...records].reverse());
  assert.deepEqual(forward, reverse);
  assert.equal(forward.topologyPreparationScheduled, 1);
  assert.notEqual(forward.acceptedTopologySha256,
    sparseCM12Phase1TopologyReceipt([{ ...records[0]!, candidateResolution: 2 },
      records[1]!]).acceptedTopologySha256);
});

test("Phase-1 arm comparison reports the first raw-bit field mismatch", () => {
  assert.doesNotThrow(() => assertSparseCM12Phase1ArmBitExact(arm(), arm()));
  assert.throws(() => assertSparseCM12Phase1ArmBitExact(arm(), arm(0x3f80_0001)),
    /step 0: density raw-bit receipt changed/);
});

test("Phase-1 canonical JSON is independent of object insertion order", () => {
  assert.equal(sparseCM12Phase1CanonicalJSON({ z: 1, a: { d: 2, b: 3 } }),
    sparseCM12Phase1CanonicalJSON({ a: { b: 3, d: 2 }, z: 1 }));
});

test("Phase-1 transport receipt rejects packet overlap, omission, and stale products", () => {
  assert.doesNotThrow(() => assertSparseCM12Phase1TransportReceipt(
    transport(), "fixture"));
  assert.throws(() => assertSparseCM12Phase1TransportReceipt(
    transport({ duplicateScatterCellCount: 1 }), "fixture"), /more than one packet/);
  assert.throws(() => assertSparseCM12Phase1TransportReceipt(
    transport({ omittedAcceptedCellCount: 1 }), "fixture"), /missing from packets/);
  assert.throws(() => assertSparseCM12Phase1TransportReceipt(
    transport({ publishedPacketGeneration: 0 }), "fixture"), /packets are stale/);
  assert.throws(() => assertSparseCM12Phase1TransportReceipt(
    transport({ effectiveVelocityTopologyGeneration: 0 }), "fixture"), /stale topology/);
  assert.throws(() => assertSparseCM12Phase1TransportReceipt(
    transport({ effectiveVelocityFrameGeneration: 0 }), "fixture"), /not the VEX product/);
});

test("Phase-1 arm comparison catches compensating transport changes", () => {
  const baseline = arm();
  const checkpoint = baseline.checkpoints[0]!;
  const candidate: SparseCM12Phase1ArmReceipt = { ...baseline,
    checkpoints: [{ ...checkpoint, transport: {
      ...checkpoint.transport, packetSha256: "1".repeat(64),
    } }] };
  assert.throws(() => assertSparseCM12Phase1ArmBitExact(baseline, candidate),
    /transport intermediate receipt changed/);
});

test("Phase-1 packet churn requires same-checkpoint republish after accepted rerung", () => {
  const initial = arm();
  const first = initial.checkpoints[0]!;
  const changed = { ...first, step: 1, acceptedLeafStateSha256: "changed",
    acceptedTopologySha256: "full-changed", transport: {
      ...first.transport, acceptedTopologyGeneration: 2, publishedPacketGeneration: 2,
      effectiveVelocityTopologyGeneration: 2, frameGeneration: 2,
      effectiveVelocityFrameGeneration: 2, publishedPacketSha256: "2".repeat(64),
    } };
  const churn: SparseCM12Phase1ArmReceipt = { ...initial,
    checkpoints: [first, changed], trajectorySha256: "trajectory" };
  assert.doesNotThrow(() => assertSparseCM12Phase1PacketChurn(churn));
  const stale: SparseCM12Phase1ArmReceipt = { ...churn,
    checkpoints: [first, { ...changed, transport: {
      ...changed.transport, publishedPacketGeneration: 1,
    } }] };
  assert.throws(() => assertSparseCM12Phase1PacketChurn(stale), /packets are stale/);
});
