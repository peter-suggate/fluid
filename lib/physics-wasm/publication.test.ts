import assert from "node:assert/strict";
import test from "node:test";
import { decodePhysicsPublication, PhysicsPlane } from "./publication";
import type { PhysicsPublication } from "./protocol";

interface Entry { id: number; kind: number; offset: number; count: number }

function encoded(entries: Entry[], totalBytes = 384): { publication: PhysicsPublication; released(): number } {
  const buffer = new ArrayBuffer(totalBytes), bytes = new Uint8Array(buffer), view = new DataView(buffer);
  bytes.set(new TextEncoder().encode("FLUIDCPU"));
  const revision = { schemaVersion: 1, dimension: 2 as const, runEpoch: 1, commandSequence: 1,
    frame: 0, time: 0, injections: 0, topologyGeneration: 0,
    fieldRevision: 0, surfaceRevision: 0, memoryEpoch: 0 };
  const metadata = new TextEncoder().encode(JSON.stringify({ revision }));
  const metadataOffset = 32 + 16 * entries.length;
  for (const [index, value] of [1, totalBytes, entries.length, metadataOffset, metadata.length, 0].entries()) {
    view.setUint32(8 + 4 * index, value, true);
  }
  bytes.set(metadata, metadataOffset);
  for (const [index, entry] of entries.entries()) {
    const at = 32 + 16 * index;
    view.setUint32(at, entry.id, true); view.setUint32(at + 4, entry.kind, true);
    view.setUint32(at + 8, entry.offset, true); view.setUint32(at + 12, entry.count, true);
  }
  let releases = 0;
  return {
    publication: {
      id: 0,
      revision,
      bytes,
      release: () => { releases++; },
    },
    released: () => releases,
  };
}

test("decoder preserves signed zero and NaN payloads in zero-copy f32 views", () => {
  const fixture = encoded([{ id: PhysicsPlane.Density, kind: 1, offset: 256, count: 2 }]);
  const words = new Uint32Array(fixture.publication.bytes.buffer, 256, 2);
  words.set([0x80000000, 0x7fc00042]);
  const decoded = decodePhysicsPublication(fixture.publication);
  const density = decoded.plane(PhysicsPlane.Density);
  assert.ok(density instanceof Float32Array);
  assert.deepEqual([...new Uint32Array(density.buffer, density.byteOffset, density.length)],
    [0x80000000, 0x7fc00042]);
  decoded.release();
  assert.equal(fixture.released(), 1);
  assert.throws(() => decoded.plane(PhysicsPlane.Density), /released/);
});

test("decoder rejects corrupt, misaligned, duplicate, and wrongly typed planes", () => {
  assert.throws(() => decodePhysicsPublication(encoded([
    { id: PhysicsPlane.Density, kind: 1, offset: 512, count: 1 },
  ]).publication), /range is invalid/);
  assert.throws(() => decodePhysicsPublication(encoded([
    { id: PhysicsPlane.Density, kind: 1, offset: 257, count: 1 },
  ]).publication), /64-byte aligned/);
  assert.throws(() => decodePhysicsPublication(encoded([
    { id: PhysicsPlane.Density, kind: 1, offset: 256, count: 1 },
    { id: PhysicsPlane.Density, kind: 1, offset: 256, count: 1 },
  ]).publication), /Duplicate physics plane/);
  assert.throws(() => decodePhysicsPublication(encoded([
    { id: PhysicsPlane.PressureMember, kind: 1, offset: 256, count: 1 },
  ]).publication), /expected 2/);
});

test("decoder rejects a header whose declared byte length is not exact", () => {
  const fixture = encoded([]);
  new DataView(fixture.publication.bytes.buffer).setUint32(12, 383, true);
  assert.throws(() => decodePhysicsPublication(fixture.publication), /length does not match/);
});

test("decoder rejects metadata from a different world revision", () => {
  const fixture = encoded([]);
  const inconsistent = { ...fixture.publication,
    revision: { ...fixture.publication.revision, commandSequence: 2 } };
  assert.throws(() => decodePhysicsPublication(inconsistent), /disagrees on commandSequence/);
});
