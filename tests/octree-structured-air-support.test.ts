import assert from "node:assert/strict";
import test from "node:test";

import {
  STRUCTURED_AIR_SUPPORT_ARENA_MAGIC,
  STRUCTURED_AIR_SUPPORT_ARENA_VERSION,
  STRUCTURED_AIR_SUPPORT_CONTROL_WORDS,
  STRUCTURED_AIR_SUPPORT_INVALID,
  STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS,
  STRUCTURED_AIR_SUPPORT_RECORD_FLAGS,
  STRUCTURED_AIR_SUPPORT_RECORD_WORDS,
  buildStructuredAirSupport,
  findStructuredAirSupportRecord,
  packStructuredAirSupportArena,
  planStructuredAirSupportArena,
  type StructuredAirSupportTopologyCell,
} from "../lib/octree-structured-air-support";

const dimensions = [16, 16, 16] as const;
const regularMissingVectorAnchor = { origin: [11, 1, 1] as const, size: 1 };
const transition7949Owner = { origin: [3, 3, 7] as const, size: 1 };
const transitionPositiveAir = { origin: [3, 3, 8] as const, size: 1 };

function acceptedTopology(): StructuredAirSupportTopologyCell[] {
  const cells: StructuredAirSupportTopologyCell[] = [];
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) {
    for (let x = 0; x < dimensions[0]; x += 1) {
      const isRegularAnchor = x === 11 && y === 1 && z === 1;
      const isTransition = x === 3 && y === 3 && z === 7;
      cells.push({ origin: [x, y, z], size: 1,
        caseId: isTransition ? 7_949 : 0, transform: isTransition ? 38 : 0,
        regular: !isTransition,
        ...(isRegularAnchor ? { compactRow: 17 } : isTransition ? { compactRow: 29 } : {}) });
    }
  }
  return cells;
}

test("support closure contains the exact regular and case-7949 positive-air demands", () => {
  const plan = buildStructuredAirSupport({ dimensions, topologyComplete: true,
    topologyCells: acceptedTopology(),
    interfaceRows: [regularMissingVectorAnchor, transition7949Owner],
    transitionSelectorCells: [transitionPositiveAir],
    fineBandCells: [transitionPositiveAir],
    regularInterpolationRows: [regularMissingVectorAnchor],
    extensionLayers: 6, capacity: 4_096, generation: 23 });

  const regular = findStructuredAirSupportRecord(plan, regularMissingVectorAnchor);
  assert.equal(regular?.compactRow, 17);
  assert.ok((regular!.flags & STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.interfaceSource) !== 0);
  assert.ok((regular!.flags & STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.regularInterpolationAnchor) !== 0);
  for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const record = findStructuredAirSupportRecord(plan,
        { origin: [11 + dx, 1 + dy, 1 + dz], size: 1 });
      assert.ok(record, `regular cube support ${dx},${dy},${dz} must exist`);
    }
  }
  const transition = findStructuredAirSupportRecord(plan, transition7949Owner);
  assert.equal(transition?.caseId, 7_949);
  assert.equal(transition?.transform, 38);
  const air = findStructuredAirSupportRecord(plan, transitionPositiveAir);
  assert.equal(air?.compactRow, undefined, "positive air must not acquire a momentum DOF");
  assert.ok((air!.flags & STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.transitionSelector) !== 0);
  assert.ok((air!.flags & STRUCTURED_AIR_SUPPORT_RECORD_FLAGS.fineBandDemand) !== 0);
});

test("packed arena groups immutable identities and invalid destination vectors", () => {
  const plan = buildStructuredAirSupport({ dimensions, topologyComplete: true,
    topologyCells: acceptedTopology(), interfaceRows: [transition7949Owner],
    transitionSelectorCells: [transitionPositiveAir], extensionLayers: 0,
    capacity: 8, generation: 9 });
  const packed = packStructuredAirSupportArena(plan), layout = plan.layout;
  assert.deepEqual([...packed.slice(0, 13)], [
    STRUCTURED_AIR_SUPPORT_ARENA_MAGIC, STRUCTURED_AIR_SUPPORT_ARENA_VERSION,
    9, 2, 8, 3, 0, STRUCTURED_AIR_SUPPORT_CONTROL_WORDS,
    layout.vectorOffsetWords, layout.allocatedWords,
    STRUCTURED_AIR_SUPPORT_OWNED_FACE_SLOTS, STRUCTURED_AIR_SUPPORT_RECORD_WORDS, 4,
  ]);
  const transitionIndex = plan.records.findIndex((record) => record.caseId === 7_949);
  const transitionBase = layout.recordOffsetWords + transitionIndex * STRUCTURED_AIR_SUPPORT_RECORD_WORDS;
  assert.deepEqual([...packed.slice(transitionBase, transitionBase + 5)], [3, 3, 7, 1, 7_949]);
  assert.equal(packed[transitionBase + 6], 29);
  assert.ok(packed.slice(layout.vectorOffsetWords, layout.vectorOffsetWords + 8 * 4)
    .every((word) => word === 0), "valid zero velocity must remain distinguishable from initial invalid vectors");
  assert.ok(packed.slice(layout.recordOffsetWords + plan.records.length * STRUCTURED_AIR_SUPPORT_RECORD_WORDS,
    layout.vectorOffsetWords).every((word) => word === STRUCTURED_AIR_SUPPORT_INVALID));
});

test("support publication fails closed on missing stencil identity and capacity overflow", () => {
  const missing = acceptedTopology().filter((cell) =>
    !(cell.origin[0] === 12 && cell.origin[1] === 1 && cell.origin[2] === 1));
  assert.throws(() => buildStructuredAirSupport({ dimensions, topologyComplete: true,
    topologyCells: missing, interfaceRows: [regularMissingVectorAnchor],
    regularInterpolationRows: [regularMissingVectorAnchor], extensionLayers: 0,
    capacity: 64, generation: 1 }), /3x3x3 stencil.*absent/);
  assert.throws(() => buildStructuredAirSupport({ dimensions, topologyComplete: true,
    topologyCells: acceptedTopology(), interfaceRows: [regularMissingVectorAnchor],
    regularInterpolationRows: [regularMissingVectorAnchor], extensionLayers: 1,
    capacity: 26, generation: 1 }), /closure .* exceeds capacity/);
});

test("support publication rejects transform codes outside the catalog cube symmetries", () => {
  const malformed = acceptedTopology();
  malformed[0] = { ...malformed[0]!, transform: 48 };
  assert.throws(() => buildStructuredAirSupport({ dimensions, topologyComplete: true,
    topologyCells: malformed, interfaceRows: [regularMissingVectorAnchor], extensionLayers: 0,
    capacity: 64, generation: 1 }), /transform must name a cube symmetry/);
});

test("physical exterior is excluded from a boundary cube support stencil", () => {
  const localDimensions = [4, 4, 4] as const;
  const topology: StructuredAirSupportTopologyCell[] = [];
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) {
    topology.push({ origin: [x, y, z], size: 1, caseId: 0, transform: 0,
      regular: true, ...(x === 0 && y === 0 && z === 0 ? { compactRow: 0 } : {}) });
  }
  const plan = buildStructuredAirSupport({ dimensions: localDimensions, topologyComplete: true,
    topologyCells: topology, interfaceRows: [{ origin: [0, 0, 0], size: 1 }],
    regularInterpolationRows: [{ origin: [0, 0, 0], size: 1 }], extensionLayers: 0,
    capacity: 8, generation: 1 });
  assert.equal(plan.records.length, 8, "only the in-domain octant of the 3x3x3 stencil is support");
  assert.ok(plan.records.every((record) => record.origin.every((coordinate) => coordinate >= 0)));
});

test("arena sizing is aligned and reserves no hidden per-face momentum authority", () => {
  const layout = planStructuredAirSupportArena(5);
  assert.equal(layout.recordOffsetWords, STRUCTURED_AIR_SUPPORT_CONTROL_WORDS);
  assert.equal(layout.vectorOffsetWords,
    STRUCTURED_AIR_SUPPORT_CONTROL_WORDS + 5 * STRUCTURED_AIR_SUPPORT_RECORD_WORDS);
  assert.equal(layout.allocatedBytes % 256, 0);
  assert.equal(STRUCTURED_AIR_SUPPORT_INVALID, 0xffff_ffff);
});
