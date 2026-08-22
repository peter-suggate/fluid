import assert from "node:assert/strict";
import { createHash } from "node:crypto";

export const SPARSE_CM12_PHASE1_BIT_EXACT_SCHEMA =
  "fluid.sparse-cm12.phase1-bit-exact.v2" as const;

export type SparseCM12Phase1OracleScene =
  "ocean-seiche" | "mini64" | "long-dam" | "symmetric-expansion";

export interface SparseCM12Phase1HashReceipt {
  readonly elements: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface SparseCM12Phase1FieldReceipt {
  readonly density: SparseCM12Phase1HashReceipt;
  readonly gamma: SparseCM12Phase1HashReceipt;
  readonly velocity: SparseCM12Phase1HashReceipt;
  readonly pressure: SparseCM12Phase1HashReceipt;
  readonly divergence: SparseCM12Phase1HashReceipt;
  readonly combinedSha256: string;
}

export interface SparseCM12Phase1CheckpointReceipt {
  readonly step: number;
  readonly fields: SparseCM12Phase1FieldReceipt;
  /** FSM1 is the producer-authored final-scalar packet-mask receipt. */
  readonly scalarResult: Readonly<Record<string, unknown>>;
  readonly frameControl: Readonly<Record<string, unknown>>;
  readonly scalarResultSha256: string;
  readonly frameControlSha256: string;
  readonly acceptedTopologySha256: string;
  /** Active + accepted rung only; candidate planning cannot change this hash. */
  readonly acceptedLeafStateSha256: string;
  readonly acceptedTopologyRecords: number;
  readonly topologyPreparationScheduled: number;
  /**
   * Transport-internal raw-bit receipt. Final fields alone cannot detect a
   * compensating stencil/scatter change, so Phase 1 requires every member.
   */
  readonly transport: SparseCM12Phase1TransportReceipt;
}

export interface SparseCM12Phase1TransportReceipt {
  readonly stencilCellsSha256: string;
  readonly stencilWeightBitsSha256: string;
  readonly departureBitsSha256: string;
  readonly betaFixedSha256: string;
  readonly deficitDensityFixedSha256: string;
  readonly deficitGammaFixedSha256: string;
  readonly massReceiptSha256: string;
  readonly packetSha256: string;
  readonly publishedPacketSha256: string;
  readonly packetCount: number;
  readonly packetCellCount: number;
  readonly duplicateScatterCellCount: number;
  readonly omittedAcceptedCellCount: number;
  readonly acceptedTopologyGeneration: number;
  readonly transportTopologyGeneration: number;
  readonly frameGeneration: number;
  readonly packetGeneration: number;
  readonly publishedPacketGeneration: number;
  readonly effectiveVelocityTopologyGeneration: number;
  readonly effectiveVelocityFrameGeneration: number;
}

export interface SparseCM12Phase1ArmReceipt {
  readonly schema: typeof SPARSE_CM12_PHASE1_BIT_EXACT_SCHEMA;
  readonly scene: SparseCM12Phase1OracleScene;
  readonly experiment: string;
  readonly dimensions: readonly [number, number, number];
  readonly brickFineResolution: 8 | 16;
  readonly presentationPageResolution: 8 | 16;
  readonly dt_s: number;
  readonly checkpoints: readonly SparseCM12Phase1CheckpointReceipt[];
  readonly trajectorySha256: string;
}

export interface SparseCM12Phase1DiagnosticFields {
  readonly density: Float32Array;
  readonly gamma: Float32Array;
  readonly velocity: Float32Array;
  readonly pressure: Float32Array;
  readonly divergence: Float32Array;
}

export interface SparseCM12Phase1TopologyRecord {
  readonly key: number;
  readonly active: boolean;
  readonly acceptedResolution: number;
  readonly candidateResolution: number;
  readonly candidateStatus: number;
  readonly candidateEpoch: number;
  readonly topologyPreparationScheduled: boolean;
  readonly topologyPreparationEpoch: number;
}

const bytesOf = (view: ArrayBufferView): Uint8Array =>
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

export function sparseCM12Phase1Sha256(view: ArrayBufferView | string): string {
  return createHash("sha256").update(view instanceof Object && typeof view !== "string"
    ? bytesOf(view) : view).digest("hex");
}

function hashReceipt(view: ArrayBufferView): SparseCM12Phase1HashReceipt {
  return {
    elements: "length" in view ? Number(view.length) : view.byteLength,
    bytes: view.byteLength,
    sha256: sparseCM12Phase1Sha256(view),
  };
}

/** Stable JSON is used only for integer control receipts; physical floats are raw-byte hashed. */
export function sparseCM12Phase1CanonicalJSON(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((item) => sparseCM12Phase1CanonicalJSON(item)).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${sparseCM12Phase1CanonicalJSON(record[key])}`).join(",")}}`;
}

export function sparseCM12Phase1FieldReceipt(
  fields: SparseCM12Phase1DiagnosticFields,
): SparseCM12Phase1FieldReceipt {
  const combined = createHash("sha256");
  const receipt = {} as Record<keyof SparseCM12Phase1DiagnosticFields,
    SparseCM12Phase1HashReceipt>;
  for (const name of ["density", "gamma", "velocity", "pressure", "divergence"] as const) {
    const view = fields[name];
    receipt[name] = hashReceipt(view);
    combined.update(name); combined.update("\0"); combined.update(bytesOf(view));
  }
  return { ...receipt, combinedSha256: combined.digest("hex") };
}

export function sparseCM12Phase1TopologyReceipt(
  records: readonly SparseCM12Phase1TopologyRecord[],
): Pick<SparseCM12Phase1CheckpointReceipt, "acceptedTopologySha256"
  | "acceptedLeafStateSha256" | "acceptedTopologyRecords"
  | "topologyPreparationScheduled"> {
  const ordered = [...records].sort((left, right) => left.key - right.key);
  const words = new Uint32Array(8 * ordered.length);
  let scheduled = 0;
  const acceptedWords = new Uint32Array(3 * ordered.length);
  for (let index = 0; index < ordered.length; index += 1) {
    const record = ordered[index]!;
    const base = 8 * index;
    words.set([record.key, record.active ? 1 : 0, record.acceptedResolution,
      record.candidateResolution, record.candidateStatus, record.candidateEpoch,
      record.topologyPreparationScheduled ? 1 : 0,
      record.topologyPreparationEpoch], base);
    scheduled += record.topologyPreparationScheduled ? 1 : 0;
    acceptedWords.set([record.key, record.active ? 1 : 0,
      record.acceptedResolution], 3 * index);
  }
  return {
    acceptedTopologySha256: sparseCM12Phase1Sha256(words),
    acceptedLeafStateSha256: sparseCM12Phase1Sha256(acceptedWords),
    acceptedTopologyRecords: ordered.length,
    topologyPreparationScheduled: scheduled,
  };
}

export function sparseCM12Phase1TrajectorySha256(
  checkpoints: readonly SparseCM12Phase1CheckpointReceipt[],
): string {
  return sparseCM12Phase1Sha256(sparseCM12Phase1CanonicalJSON(checkpoints.map((checkpoint) => ({
    step: checkpoint.step,
    fields: checkpoint.fields,
    scalarResultSha256: checkpoint.scalarResultSha256,
    frameControlSha256: checkpoint.frameControlSha256,
    acceptedTopologySha256: checkpoint.acceptedTopologySha256,
    acceptedLeafStateSha256: checkpoint.acceptedLeafStateSha256,
    transport: checkpoint.transport,
  }))));
}

/** Mini/long-dam lifecycle gate: every accepted rerung is visible in packets immediately. */
export function assertSparseCM12Phase1PacketChurn(
  arm: SparseCM12Phase1ArmReceipt,
  minimumAcceptedChanges = 1,
): void {
  assert.ok(arm.scene === "mini64" || arm.scene === "long-dam",
    "packet churn is required on mini64 or long-dam");
  let acceptedChanges = 0;
  for (let index = 0; index < arm.checkpoints.length; index += 1) {
    const checkpoint = arm.checkpoints[index]!;
    assertSparseCM12Phase1TransportReceipt(checkpoint.transport,
      `${arm.scene} step ${checkpoint.step}`);
    if (index === 0) continue;
    const previous = arm.checkpoints[index - 1]!;
    if (checkpoint.acceptedLeafStateSha256 === previous.acceptedLeafStateSha256) continue;
    acceptedChanges += 1;
    assert.notEqual(checkpoint.transport.publishedPacketSha256,
      previous.transport.publishedPacketSha256,
      `${arm.scene} step ${checkpoint.step}: accepted rerung did not republish packets`);
    assert.equal(checkpoint.transport.acceptedTopologyGeneration,
      checkpoint.transport.publishedPacketGeneration,
      `${arm.scene} step ${checkpoint.step}: rerung packets have one-frame latency`);
  }
  assert.ok(acceptedChanges >= minimumAcceptedChanges,
    `${arm.scene} exercised ${acceptedChanges}/${minimumAcceptedChanges} accepted topology changes`);
}

export function assertSparseCM12Phase1OracleCoverage(reports: readonly {
  readonly scene: SparseCM12Phase1OracleScene;
  readonly baseline: SparseCM12Phase1ArmReceipt;
  readonly candidate: SparseCM12Phase1ArmReceipt;
}[]): void {
  const expected = ["ocean-seiche", "mini64", "long-dam", "symmetric-expansion"] as const;
  assert.deepEqual([...reports.map((report) => report.scene)].sort(), [...expected].sort(),
    "Phase-1 acceptance requires ocean, mini64, long-dam, and symmetric expansion");
  const minimumCheckpoints: Record<SparseCM12Phase1OracleScene, number> = {
    "ocean-seiche": 4, mini64: 6, "long-dam": 6, "symmetric-expansion": 21,
  };
  for (const report of reports) {
    assert.equal(report.baseline.scene, report.scene);
    assert.equal(report.candidate.scene, report.scene);
    assert.ok(report.baseline.checkpoints.length >= minimumCheckpoints[report.scene],
      `${report.scene} baseline is shorter than its canonical Phase-1 lane`);
    assert.ok(report.candidate.checkpoints.length >= minimumCheckpoints[report.scene],
      `${report.scene} candidate is shorter than its canonical Phase-1 lane`);
    assertSparseCM12Phase1ArmBitExact(report.baseline, report.candidate);
    if (report.scene === "mini64" || report.scene === "long-dam") {
      assertSparseCM12Phase1PacketChurn(report.baseline);
      assertSparseCM12Phase1PacketChurn(report.candidate);
    }
  }
}

const receiptHash = (value: unknown, label: string): void => {
  assert.equal(typeof value, "string", `${label} is missing`);
  assert.match(value as string, /^[0-9a-f]{64}$/, `${label} is not a SHA-256 digest`);
};

/** Validate exact packet/scatter coverage and the zero-frame publication receipt. */
export function assertSparseCM12Phase1TransportReceipt(
  receipt: SparseCM12Phase1TransportReceipt,
  context: string,
): void {
  for (const field of ["stencilCellsSha256", "stencilWeightBitsSha256",
    "departureBitsSha256", "betaFixedSha256", "deficitDensityFixedSha256",
    "deficitGammaFixedSha256", "massReceiptSha256", "packetSha256",
    "publishedPacketSha256"] as const) {
    receiptHash(receipt[field], `${context}: ${field}`);
  }
  for (const field of ["packetCount", "packetCellCount",
    "duplicateScatterCellCount", "omittedAcceptedCellCount",
    "acceptedTopologyGeneration", "transportTopologyGeneration", "packetGeneration",
    "publishedPacketGeneration",
    "frameGeneration", "effectiveVelocityTopologyGeneration",
    "effectiveVelocityFrameGeneration"] as const) {
    assert.ok(Number.isSafeInteger(receipt[field]) && receipt[field] >= 0,
      `${context}: ${field} must be a non-negative integer`);
  }
  assert.equal(receipt.duplicateScatterCellCount, 0,
    `${context}: a transport cell is owned by more than one packet`);
  assert.equal(receipt.omittedAcceptedCellCount, 0,
    `${context}: accepted transport cells are missing from packets`);
  assert.equal(receipt.packetGeneration, receipt.transportTopologyGeneration,
    `${context}: executed packets do not match the transport topology`);
  assert.equal(receipt.publishedPacketGeneration, receipt.acceptedTopologyGeneration,
    `${context}: packets are stale after the topology publication point`);
  assert.equal(receipt.effectiveVelocityTopologyGeneration,
    receipt.acceptedTopologyGeneration,
    `${context}: effective-velocity plane has stale topology`);
  assert.equal(receipt.effectiveVelocityFrameGeneration, receipt.frameGeneration,
    `${context}: effective-velocity plane is not the VEX product for this frame`);
}

/** Throws at the first changed checkpoint and names the changed authority. */
export function assertSparseCM12Phase1ArmBitExact(
  baseline: SparseCM12Phase1ArmReceipt,
  candidate: SparseCM12Phase1ArmReceipt,
): void {
  assert.equal(candidate.schema, baseline.schema, "oracle schema changed");
  assert.equal(candidate.scene, baseline.scene, "oracle scene changed");
  assert.deepEqual(candidate.dimensions, baseline.dimensions, "oracle dimensions changed");
  assert.equal(candidate.brickFineResolution, baseline.brickFineResolution,
    "brick-fine profile changed");
  assert.equal(candidate.presentationPageResolution, baseline.presentationPageResolution,
    "presentation-page profile changed");
  assert.equal(candidate.dt_s, baseline.dt_s, "oracle timestep changed");
  assert.equal(candidate.checkpoints.length, baseline.checkpoints.length,
    "checkpoint count changed");
  for (let index = 0; index < baseline.checkpoints.length; index += 1) {
    const expected = baseline.checkpoints[index]!;
    const actual = candidate.checkpoints[index]!;
    assert.equal(actual.step, expected.step, `checkpoint ${index} step changed`);
    assertSparseCM12Phase1TransportReceipt(expected.transport,
      `baseline step ${expected.step}`);
    assertSparseCM12Phase1TransportReceipt(actual.transport,
      `candidate step ${actual.step}`);
    assert.equal(expected.transport.frameGeneration,
      expected.frameControl.acceptedGeneration,
      `baseline step ${expected.step}: plane frame generation differs from FCA1`);
    assert.equal(actual.transport.frameGeneration,
      actual.frameControl.acceptedGeneration,
      `candidate step ${actual.step}: plane frame generation differs from FCA1`);
    assert.equal(expected.transport.transportTopologyGeneration,
      expected.scalarResult.topologyGeneration,
      `baseline step ${expected.step}: packet topology generation differs from FSM1`);
    assert.equal(actual.transport.transportTopologyGeneration,
      actual.scalarResult.topologyGeneration,
      `candidate step ${actual.step}: packet topology generation differs from FSM1`);
    for (const field of ["density", "gamma", "velocity", "pressure", "divergence",
      "combinedSha256"] as const) {
      assert.deepEqual(actual.fields[field], expected.fields[field],
        `step ${expected.step}: ${field} raw-bit receipt changed`);
    }
    assert.deepEqual(actual.scalarResult, expected.scalarResult,
      `step ${expected.step}: FSM1 final-scalar packet-mask receipt changed`);
    assert.deepEqual(actual.frameControl, expected.frameControl,
      `step ${expected.step}: FCA1 frame receipt changed`);
    assert.equal(actual.acceptedTopologySha256, expected.acceptedTopologySha256,
      `step ${expected.step}: accepted topology receipt changed`);
    assert.equal(actual.acceptedLeafStateSha256, expected.acceptedLeafStateSha256,
      `step ${expected.step}: accepted active/rung state changed`);
    assert.deepEqual(actual.transport, expected.transport,
      `step ${expected.step}: transport intermediate receipt changed`);
  }
  assert.equal(candidate.trajectorySha256, baseline.trajectorySha256,
    "combined Phase-1 trajectory receipt changed");
}
