import assert from "node:assert/strict";
import test from "node:test";
import { createAdvanceView } from "./advance-view";
import { decodePhysicsPublication, PhysicsPlane } from "./publication";
import type { PhysicsPublication, PhysicsRevision } from "./protocol";

type Value = Float32Array | Uint8Array | string;
const align64 = (value: number): number => (value + 63) & ~63;

function publication(values: ReadonlyMap<number, Value>, includeGraph = true): PhysicsPublication {
  const revision: PhysicsRevision = { schemaVersion: 1, dimension: 2, runEpoch: 3,
    commandSequence: 4, frame: 2, time: 0.1, injections: 1, topologyGeneration: 9,
    fieldRevision: 2, surfaceRevision: 2, memoryEpoch: 1 };
  const graph = { schemaVersion: 1, dimension: 2, dimensions: [1, 1, 1], topologyGeneration: 9,
    cells: [{ id: 0, minimum: [0, 0, 0], maximum: [1, 1, 1], center: [.5, .5, .5],
      widths: [1, 1, 1], measure: 1, brickKey: 17 }],
    rows: [{ id: 0, axis: 0, kind: "intraBrick", center: [0, .5, .5], measure: 1, terms: [] },
      { id: 1, axis: 1, kind: "intraBrick", center: [.5, 0, .5], measure: 1, terms: [] }],
    subfaces: [{ id: 0, rowId: 0, axis: 0, center: [0, .5, .5], measure: 1 },
      { id: 1, rowId: 1, axis: 1, center: [.5, 0, .5], measure: 1 }],
    bricks: [{ id: 0, key: 17, coordinate: [0, 0, 0], spanBricks: 1, resolution: 1, active: true }] };
  const entries = new Map(values);
  if (includeGraph) entries.set(PhysicsPlane.GraphJson, JSON.stringify(graph));
  const metadataBytes = new TextEncoder().encode(JSON.stringify({ revision,
    receipt: { ...revision, microsteps: 2, maxVelocity: 3, drift: .01,
      lastInjection: { accepted: true } },
    scene: { dimensions: [1, 1, 1], cellSizeM: .25,
      frame: { sourceDimensions: [1, 1, 1], centerZ: 0, centerCellZ: 0, originX: -.125, originY: 0 },
      hasStaticWorld: true, hasInflow: false, hasRigidBodies: false },
    surface: { exactAreaFine: .5 }, tracersEnabled: true }));
  const directoryEnd = 32 + 16 * entries.size;
  let cursor = align64(directoryEnd + metadataBytes.length);
  const records = [...entries].map(([id, value]) => {
    const bytes = typeof value === "string" ? new TextEncoder().encode(value)
      : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    const kind = typeof value === "string" ? 3 : value instanceof Float32Array ? 1 : 2;
    const record = { id, kind, offset: cursor, count: typeof value === "string" ? bytes.length : value.length, bytes };
    cursor = align64(cursor + bytes.length); return record;
  });
  const buffer = new ArrayBuffer(cursor), bytes = new Uint8Array(buffer), data = new DataView(buffer);
  bytes.set(new TextEncoder().encode("FLUIDCPU"));
  [1, cursor, records.length, directoryEnd, metadataBytes.length, 0]
    .forEach((value, index) => data.setUint32(8 + index * 4, value, true));
  bytes.set(metadataBytes, directoryEnd);
  records.forEach((record, index) => {
    const at = 32 + 16 * index;
    data.setUint32(at, record.id, true); data.setUint32(at + 4, record.kind, true);
    data.setUint32(at + 8, record.offset, true); data.setUint32(at + 12, record.count, true);
    bytes.set(record.bytes, record.offset);
  });
  return { id: 1, revision, bytes, release() {} };
}

const f = (...values: number[]): Float32Array => Float32Array.from(values);
function planes(): Map<number, Value> {
  return new Map<number, Value>([
    [PhysicsPlane.Density, f(.5)], [PhysicsPlane.Capacity, f(1)],
    [PhysicsPlane.Pressure, f(7)], [PhysicsPlane.PressureRhs, f(-2)],
    [PhysicsPlane.ExtensionDepth, Uint8Array.of(3)], [PhysicsPlane.InterfaceNormal, f(0, 1)],
    [PhysicsPlane.InterfaceOffset, f(.5)], [PhysicsPlane.FaceVelocity, f(2, 3)],
    [PhysicsPlane.LimitedFlux, f(1, 2)], [PhysicsPlane.HighFlux, f(1, 3)],
    [PhysicsPlane.DensityBefore, f(.25)], [PhysicsPlane.CapacityFine, f(.75)],
    [PhysicsPlane.MaterialFine, f(42)], [PhysicsPlane.BrickResolutionBefore, Uint8Array.of(2)],
    [PhysicsPlane.BrickActivity, f(.6)], [PhysicsPlane.VelocityXBeforePressure, f(4, 5)],
    [PhysicsPlane.VelocityYBeforePressure, f(6, 7)], [PhysicsPlane.Tracers, f(.25, .75, .5, 1)],
    [PhysicsPlane.RdfVertices, f(-1, 1, 1, -1)], [PhysicsPlane.RdfSegments, f(0, .5, 1, .5)],
  ] as [number, Value][]);
}

test("Advance view owns and reflects the complete Wasm publication", () => {
  const decoded = decodePhysicsPublication(publication(planes()));
  const view = createAdvanceView(decoded, undefined, { id: "fixture", label: "Fixture" });
  decoded.release();
  assert.equal(view.lattice.cells[0]?.fill, .5);
  assert.deepEqual([...view.faceVelocityXFine], [2, 0]);
  assert.deepEqual([...view.faceVelocityYFine], [0, -3]);
  assert.deepEqual([...view.fluxLimitedXFine], [0, 0]);
  assert.deepEqual([...view.fluxLimitedYFine], [0, 1]);
  assert.deepEqual(view.markers, [{ x: .25, y: .25, alive: true }]);
  assert.equal(view.capacityFine[0], .75);
  assert.equal(view.materialFine[0], 42);
  assert.equal(view.scene.hasStaticWorld, true);
});

test("Advance view reuses cold graph and rejects a malformed hot plane", () => {
  const first = decodePhysicsPublication(publication(planes()));
  const firstView = createAdvanceView(first); first.release();
  const next = decodePhysicsPublication(publication(planes(), false));
  assert.equal(createAdvanceView(next, firstView.graph).graph, firstView.graph); next.release();
  const corrupt = planes(); corrupt.set(PhysicsPlane.Density, f(.2, .3));
  const bad = decodePhysicsPublication(publication(corrupt));
  assert.throws(() => createAdvanceView(bad), /density plane has 2 values; expected 1/); bad.release();
});
