import assert from "node:assert/strict";
import test from "node:test";
import {
  StructuredStepSnapshotRing,
  structuredAuthorityStepHealth,
  type StructuredStepSnapshotSources,
  type StructuredStepSnapshotStamp,
} from "../lib/structured-step-snapshot";
import { STRUCTURED_GENERATION_AUDIT_SNAPSHOT } from "../lib/structured-authority-audit";

/** Byte-backed stand-ins: copyBufferToBuffer moves real bytes so decode paths
 * run against the exact ABI, and mapAsync resolves like a settled queue. */
class FakeBuffer {
  readonly bytes: Uint8Array;
  mapState: "unmapped" | "mapped" = "unmapped";
  destroyed = false;
  constructor(size: number, words?: number[]) {
    this.bytes = new Uint8Array(size);
    if (words) this.bytes.set(new Uint8Array(new Uint32Array(words).buffer));
  }
  async mapAsync() { this.mapState = "mapped"; }
  getMappedRange() { return this.bytes.buffer.slice(this.bytes.byteOffset, this.bytes.byteOffset + this.bytes.byteLength); }
  unmap() { this.mapState = "unmapped"; }
  destroy() { this.destroyed = true; }
}

const fakeDevice = () => {
  const created: FakeBuffer[] = [];
  return {
    created,
    device: {
      createBuffer: (descriptor: { size: number }) => {
        const buffer = new FakeBuffer(descriptor.size);
        created.push(buffer);
        return buffer as unknown as GPUBuffer;
      },
    } as unknown as GPUDevice,
  };
};

const fakeEncoder = () => ({
  copyBufferToBuffer(source: unknown, sourceOffset: number, target: unknown, targetOffset: number, bytes: number) {
    const from = source as FakeBuffer, to = target as FakeBuffer;
    to.bytes.set(from.bytes.subarray(sourceOffset, sourceOffset + bytes), targetOffset);
  },
}) as unknown as GPUCommandEncoder;

const sources = (fineGeneration: number, epoch: number, boundaryEpoch = epoch): StructuredStepSnapshotSources => ({
  structuredVelocityControl: new FakeBuffer(24, [0, 0xffff_ffff, 21, epoch, 1, 73]) as unknown as GPUBuffer,
  structuredBoundaryControl: new FakeBuffer(28, [0, 0xffff_ffff, 21, 73, boundaryEpoch, 1, boundaryEpoch]) as unknown as GPUBuffer,
  fineWorklist: new FakeBuffer(28, [fineGeneration, 40, 64, 3, 1, 1, 1]) as unknown as GPUBuffer,
});

const stamp = (step: number): StructuredStepSnapshotStamp => ({
  step, dt_s: 0.004, submittedTime_s: step * 0.004, hostFineGeneration: 0,
});

test("a record decodes the exact words the step's encoder copied", async () => {
  const { device } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 3);
  assert.ok(ring.encode(fakeEncoder(), sources(8, 7), stamp(6)));
  const record = await ring.readLatest();
  assert.ok(record);
  assert.equal(record.stamp.step, 6);
  assert.equal(record.snapshot.structured.epoch, 7);
  assert.equal(record.snapshot.boundary.publishedEpoch, 7);
  assert.equal(record.snapshot.fineHeader[0], 8);
  const health = structuredAuthorityStepHealth(record);
  assert.equal(health.acceptedEpoch, 7);
  assert.equal(health.publishedFineGeneration, 8);
  assert.equal(health.authorityLagSteps, 0, "fine = epoch + 1 is current");
  assert.equal(health.receiptValid, true);
});

test("whole-step authority lag is exact and pipeline-independent", async () => {
  const { device } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 3);
  ring.encode(fakeEncoder(), sources(10, 7), stamp(9));
  const record = await ring.readLatest();
  assert.ok(record);
  const health = structuredAuthorityStepHealth(record);
  assert.equal(health.authorityLagSteps, 2, "two steps ran without adopting a new epoch");
  assert.equal(health.receiptValid, true, "stale-but-valid is the damping mode, not a collapse");
});

test("an incoherent boundary epoch invalidates the receipt", async () => {
  const { device } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 3);
  ring.encode(fakeEncoder(), sources(8, 7, 6), stamp(6));
  const record = await ring.readLatest();
  assert.ok(record);
  assert.equal(structuredAuthorityStepHealth(record).receiptValid, false);
});

test("readLatest returns the freshest record and slots recycle", async () => {
  const { device } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 2);
  ring.encode(fakeEncoder(), sources(8, 7), stamp(6));
  ring.encode(fakeEncoder(), sources(9, 8), stamp(7));
  ring.encode(fakeEncoder(), sources(10, 9), stamp(8));
  const record = await ring.readLatest();
  assert.equal(record?.stamp.step, 8, "the stalest slot was overwritten, the freshest read");
  ring.encode(fakeEncoder(), sources(11, 10), stamp(9));
  const next = await ring.readLatest();
  assert.equal(next?.stamp.step, 9);
});

test("readLatest with nothing encoded resolves undefined", async () => {
  const { device } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 2);
  assert.equal(await ring.readLatest(), undefined);
});

test("destroy releases the slot buffers", () => {
  const { device, created } = fakeDevice();
  const ring = new StructuredStepSnapshotRing(device, 2);
  ring.destroy();
  assert.ok(created.every((buffer) => buffer.destroyed));
  assert.equal(ring.encode(fakeEncoder(), sources(8, 7), stamp(1)), false);
});

test("the ring's record layout is the harness audit ABI", () => {
  // Derived from the segments rather than restated. The record has grown once
  // already (P0.4 appended the topology-epoch, SPGrid level-delta, air-support
  // and fine-transport verdicts), and restating the total meant a legitimate
  // append read as a regression while the property that actually matters went
  // unchecked: the segments must tile the record exactly. A gap wastes copy
  // bandwidth on every audited step; an overlap silently corrupts a
  // neighbouring subsystem's verdict words.
  const layout = STRUCTURED_GENERATION_AUDIT_SNAPSHOT as unknown as Record<string, number>;
  const segments = Object.keys(layout)
    .filter((key) => key.endsWith("OffsetBytes"))
    .map((key) => ({
      name: key.slice(0, -"OffsetBytes".length),
      offset: layout[key]!,
      bytes: layout[`${key.slice(0, -"OffsetBytes".length)}Bytes`]!,
    }))
    .sort((a, b) => a.offset - b.offset);
  assert.ok(segments.length >= 6, "every audited authority must publish a segment");
  let cursor = 0;
  for (const segment of segments) {
    assert.equal(typeof segment.bytes, "number", `${segment.name} declares no width`);
    assert.equal(segment.offset, cursor,
      `${segment.name} must start where the previous segment ends`);
    cursor += segment.bytes;
  }
  assert.equal(STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes, cursor,
    "the stride must be exactly the packed segments");
  assert.equal(STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes, 808);
});
