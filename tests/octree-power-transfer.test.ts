import assert from "node:assert/strict";
import test from "node:test";

import {
  buildOctreePowerTopologyTransfer,
  compareOctreePowerTransferKeys,
  createOctreePowerFaceTransferKey,
  createOctreePowerSiteTransferKey,
  packOctreePowerTransferKeys,
  type OctreePowerTransferFace,
  type OctreePowerTransferKey,
} from "../lib/octree-power-transfer";

const aggregate: OctreePowerTransferKey = [0x8000_0000, 4, 5, 0xffff_ffff];
const key = (id: number): OctreePowerTransferKey => [0, 0, 0, id];
const face = (
  id: number,
  area: number,
  normalVelocity: number,
  overrides: Partial<OctreePowerTransferFace> = {},
): OctreePowerTransferFace => ({
  key: key(id), aggregateKey: aggregate, area, normalVelocity,
  centroid: [id, 0, 0], normal: [1, 0, 0], boundary: false,
  ...overrides,
});

const close = (actual: number, expected: number, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);

test("power transfer keys have canonical fixed 128-bit u32 tuple semantics", () => {
  const large = createOctreePowerSiteTransferKey([1023, 511, 513], 32);
  const small = createOctreePowerSiteTransferKey([1, 2, 3], 2);
  const forward = createOctreePowerFaceTransferKey(small, large, 0xabcd);
  const reverse = createOctreePowerFaceTransferKey(large, small, 0xabcd);
  assert.deepEqual(forward.key, reverse.key);
  assert.equal(forward.orientation, 1);
  assert.equal(reverse.orientation, -1);
  assert.equal(forward.key[2], (1 | (5 << 6) | (0xabcd << 16)) >>> 0);
  assert.equal(forward.key[3], 0);

  const boundary = createOctreePowerFaceTransferKey(large, undefined, 17, 0xf123_4567);
  assert.equal(boundary.key[1], 0xffff_ffff);
  assert.equal((boundary.key[2] >>> 12) & 1, 1);
  assert.equal(boundary.key[3], 0xf123_4567);
  assert.deepEqual([...packOctreePowerTransferKeys([aggregate, boundary.key])], [...aggregate, ...boundary.key]);
  assert.equal(compareOctreePowerTransferKeys(aggregate, key(1)), 1, "comparison must be unsigned, not signed Int32");
  assert.throws(() => createOctreePowerSiteTransferKey([1024, 0, 0], 1), /\[0, 1023\]/);
  assert.throws(() => createOctreePowerFaceTransferKey(small, large, 0x1_0000), /16-bit/);
});

test("unchanged power topology transfers velocity without arithmetic", () => {
  const velocity = Math.fround(1.0000001192092896);
  const boundaryKey = [7, 0xffff_ffff, 1 << 12, 9] as const;
  const previous = [face(7, 2.5, velocity, { key: boundaryKey, boundary: true })];
  const next = [face(7, 2.5, -99, { key: boundaryKey, boundary: true })];
  const transfer = buildOctreePowerTopologyTransfer(previous, next, {
    sampleVelocity: () => { throw new Error("exact transfer must not sample"); },
  });
  assert.equal(transfer.velocities[0], velocity);
  assert.deepEqual(transfer.records[0], {
    newFace: 0, mode: "exact", oldFaces: [0], inheritedFlux: 2.5 * velocity, detailFlux: 0,
  });
  assert.equal(transfer.diagnostics.exactFaceCount, 1);
  assert.equal(transfer.diagnostics.traceBackFaceCount, 0);
  assert.equal(transfer.diagnostics.oldBoundaryFlux, transfer.diagnostics.newBoundaryFlux);
});

test("many-to-one restriction preserves area-integrated flux for unequal faces", () => {
  const previous = [face(1, 1, 2), face(2, 3, 4)];
  const next = [face(9, 5, 0)];
  const transfer = buildOctreePowerTopologyTransfer(previous, next);
  assert.equal(transfer.records[0].mode, "restriction");
  assert.deepEqual(transfer.records[0].oldFaces, [0, 1]);
  close(transfer.velocities[0], 14 / 5);
  close(next[0].area * transfer.velocities[0], 1 * 2 + 3 * 4);
});

test("one-to-many prolongation preserves parent flux and adds zero-net detail", () => {
  const previous = [face(4, 4, 3)];
  const next = [
    face(10, 1, 0, { centroid: [1, 0, 0] }),
    face(11, 3, 0, { centroid: [5, 0, 0] }),
  ];
  const transfer = buildOctreePowerTopologyTransfer(previous, next, {
    sampleVelocity: ([x]) => [x, 0, 0],
  });
  assert.deepEqual(transfer.records.map((record) => record.mode), ["prolongation", "prolongation"]);
  assert.deepEqual(transfer.velocities, [0, 4]);
  const parentFlux = previous[0].area * previous[0].normalVelocity;
  const childFlux = next.reduce((sum, child, index) => sum + child.area * transfer.velocities[index], 0);
  close(childFlux, parentFlux);
  const parentMeanOnChildren = parentFlux / next.reduce((sum, child) => sum + child.area, 0);
  close(next.reduce((sum, child, index) => sum + child.area * (transfer.velocities[index] - parentMeanOnChildren), 0), 0);

  const refined = next.map((child, index) => ({ ...child, normalVelocity: transfer.velocities[index] }));
  const roundTrip = buildOctreePowerTopologyTransfer(refined, previous);
  close(roundTrip.velocities[0], 3);
});

test("genuinely new connectivity uses finite trace-back projection", () => {
  const next = [face(77, 2, 0, { aggregateKey: undefined, centroid: [2, 1, 0], normal: [0, 4, 0] })];
  const rotating = ([x, y]: readonly [number, number, number]) => [-y, x, 0] as const;
  const transfer = buildOctreePowerTopologyTransfer([], next, { dt: 0.5, sampleVelocity: rotating });
  // v(2,1)=(-1,2), departure=(2.5,0), v(departure)=(0,2.5), projected on +y.
  close(transfer.velocities[0], 2.5);
  assert.equal(transfer.records[0].mode, "trace-back");
  assert.equal(transfer.diagnostics.traceBackFaceCount, 1);

  const fallback = buildOctreePowerTopologyTransfer([], next, { fallbackVelocity: [1, 7, 3] });
  assert.equal(fallback.velocities[0], 7);
  assert.ok(Number.isFinite(fallback.velocities[0]));
  assert.throws(() => buildOctreePowerTopologyTransfer([], next, { sampleVelocity: () => [Number.NaN, 0, 0] }), /finite/);
});

test("ambiguous face identities and malformed geometry fail closed", () => {
  assert.throws(() => buildOctreePowerTopologyTransfer([face(1, 1, 0), face(1, 1, 0)], []), /Duplicate previous/);
  assert.throws(() => buildOctreePowerTopologyTransfer([], [face(1, 0, 0)]), /area must be finite and positive/);
  assert.throws(() => buildOctreePowerTopologyTransfer([], [face(1, 1, 0, { normal: [0, 0, 0] })]), /non-zero/);
  assert.throws(() => buildOctreePowerTopologyTransfer([], [face(1, 1, 0)], { dt: -1 }), /non-negative/);
});
