import assert from "node:assert/strict";
import test from "node:test";

import {
  applyFactorOneDenseSixFace,
  buildFactorOneDensePressureHierarchy,
  factorOneDenseAggregateChildren,
  factorOneDenseAggregateParent,
  factorOneDenseCoordinate,
  factorOneDenseSlot,
  FACTOR_ONE_DENSE_CELL_ROLE,
  FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS,
  FACTOR_ONE_DENSE_GPU_PHYSICAL_PLAN_VERSION,
  planFactorOneDenseGpuPhysicalImage,
  planFactorOneDensePressureHierarchy,
  prolongFactorOneDenseAggregate,
  restrictFactorOneDenseAggregate,
  validateFactorOneDensePressureHierarchy,
  type FactorOneDenseCoordinate,
} from "../lib/factor-one-dense-pressure-hierarchy";

const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  assert.equal(a.length, b.length);
  let result = 0;
  for (let index = 0; index < a.length; index += 1) result += a[index]! * b[index]!;
  return result;
};

test("factor-1 dense plan publishes exact x-fast mini bases and invertible slots", () => {
  const plan = planFactorOneDensePressureHierarchy([16, 16, 16]);
  assert.deepEqual(plan.levelDimensions, [
    [16, 16, 16], [8, 8, 8], [4, 4, 4], [2, 2, 2], [1, 1, 1],
  ]);
  assert.deepEqual(plan.levelVolumes, [4096, 512, 64, 8, 1]);
  assert.deepEqual(plan.levelBases, [0, 4096, 4608, 4672, 4680]);
  assert.equal(plan.totalSlots, 4681);
  for (let level = 0; level < plan.levelCount; level += 1) {
    const dimensions = plan.levelDimensions[level]!;
    for (let z = 0; z < dimensions[2]; z += 1) {
      for (let y = 0; y < dimensions[1]; y += 1) {
        for (let x = 0; x < dimensions[0]; x += 1) {
          const coordinate = [x, y, z] as const;
          assert.deepEqual(factorOneDenseCoordinate(
            plan, level, factorOneDenseSlot(plan, level, coordinate),
          ), coordinate);
        }
      }
    }
  }
});

test("odd dimensions ceil-halve without aliasing and expose only in-bounds children", () => {
  const plan = planFactorOneDensePressureHierarchy([5, 3, 2]);
  assert.deepEqual(plan.levelDimensions, [[5, 3, 2], [3, 2, 1], [2, 1, 1], [1, 1, 1]]);
  assert.deepEqual(plan.levelVolumes, [30, 6, 2, 1]);
  assert.deepEqual(plan.levelBases, [0, 30, 36, 38]);
  assert.equal(plan.totalSlots, 39);

  const edgeParent = factorOneDenseSlot(plan, 1, [2, 1, 0]);
  const children = factorOneDenseAggregateChildren(plan, 0, edgeParent);
  assert.deepEqual(children.map((slot) => factorOneDenseCoordinate(plan, 0, slot)), [
    [4, 2, 0], [4, 2, 1],
  ]);
  for (const child of children) {
    assert.equal(factorOneDenseAggregateParent(plan, 0, child), edgeParent);
  }
});

test("role merging is seed-order independent and owner invariants fail closed", () => {
  const seeds = [
    { level: 0, coordinate: [0, 0, 0] as const, role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 4 },
    { level: 0, coordinate: [0, 0, 0] as const, role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 4 },
    { level: 0, coordinate: [4, 2, 1] as const, role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly },
    { level: 1, coordinate: [1, 1, 0] as const, role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 7 },
  ];
  const a = buildFactorOneDensePressureHierarchy([5, 3, 2], seeds);
  const b = buildFactorOneDensePressureHierarchy([5, 3, 2], [...seeds].reverse());
  assert.deepEqual(a.flags, b.flags);
  assert.deepEqual(a.owners, b.owners);
  const merged = factorOneDenseSlot(a.plan, 0, [0, 0, 0]);
  assert.equal(a.flags[merged], FACTOR_ONE_DENSE_CELL_ROLE.active);
  assert.equal(a.owners[merged], 5);
  validateFactorOneDensePressureHierarchy(a);

  assert.throws(() => buildFactorOneDensePressureHierarchy([2, 2, 2], [
    { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 1 },
    { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 2 },
  ]), /conflicting owners/);
  assert.throws(() => buildFactorOneDensePressureHierarchy([2, 2, 2], [
    { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly, owner: 1 },
  ]), /cannot carry an owner/);

  const invalid = {
    plan: a.plan,
    flags: a.flags.slice(),
    owners: a.owners.slice(),
    occupiedLocalIndices: a.occupiedLocalIndices.map((worklist) => worklist.slice()),
  };
  invalid.owners[factorOneDenseSlot(a.plan, 0, [4, 2, 1])] = 9;
  assert.throws(() => validateFactorOneDensePressureHierarchy(invalid), /role and owner disagree/);
});

test("occupied-local worklists reproduce the captured mini census in sorted x-fast order", () => {
  const plan = planFactorOneDensePressureHierarchy([16, 16, 16]);
  const capturedCounts = [1475, 214, 35, 8, 1] as const;
  const seeds = capturedCounts.slice(0, -1).flatMap((count, level) =>
    Array.from({ length: count }, (_, local) => ({
      level,
      coordinate: factorOneDenseCoordinate(plan, level, plan.levelBases[level]! + local),
      role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly,
    })));
  const hierarchy = buildFactorOneDensePressureHierarchy([16, 16, 16], seeds);
  assert.deepEqual(
    hierarchy.occupiedLocalIndices.map((worklist) => worklist.length),
    capturedCounts,
  );
  hierarchy.occupiedLocalIndices.forEach((worklist, level) => {
    const sortedUnique = [...new Set(worklist)].sort((left, right) => left - right);
    assert.deepEqual([...worklist], sortedUnique);
    for (const local of worklist) {
      const globalSlot = hierarchy.plan.levelBases[level]! + local;
      assert.notEqual(hierarchy.flags[globalSlot], FACTOR_ONE_DENSE_CELL_ROLE.absent);
      assert.equal(
        factorOneDenseSlot(
          hierarchy.plan,
          level,
          factorOneDenseCoordinate(hierarchy.plan, level, globalSlot),
        ),
        globalSlot,
        "compact work selection must preserve the direct x-fast channel address",
      );
    }
  });
  assert.equal(
    hierarchy.occupiedLocalIndices.reduce((sum, worklist) => sum + worklist.length, 0),
    1733,
  );
});

test("versioned x-fast SoA physical plan is aligned, disjoint, and exactly accounted", () => {
  const logical = planFactorOneDensePressureHierarchy([16, 16, 16]);
  const physical = planFactorOneDenseGpuPhysicalImage(logical);
  assert.equal(physical.version, FACTOR_ONE_DENSE_GPU_PHYSICAL_PLAN_VERSION);
  assert.equal(physical.stability, "prototype-not-frozen");
  assert.equal(physical.storageOrder, "x-fast-soa");
  assert.equal(physical.channelAlignmentElements,
    FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS);
  assert.equal(physical.channelAlignmentBytes, 256);
  assert.equal(physical.channelStrideElements, 4736);
  assert.equal(physical.channelStrideBytes, 18_944);

  assert.equal(physical.channels.length, 9);
  for (const channel of physical.channels) {
    assert.equal(channel.directAddressed, true);
    assert.equal(channel.payloadElements, logical.totalSlots);
    assert.equal(channel.allocatedElements, physical.channelStrideElements);
    for (const slot of [0, 1, 4095, 4096, logical.totalSlots - 1]) {
      assert.ok(slot < logical.totalSlots);
      assert.equal(
        channel.offsetBytes + slot * physical.scalarBytes,
        channel.offsetBytes + factorOneDenseSlot(
          logical,
          logical.levelBases.findLastIndex((base) => base <= slot),
          factorOneDenseCoordinate(
            logical,
            logical.levelBases.findLastIndex((base) => base <= slot),
            slot,
          ),
        ) * physical.scalarBytes,
      );
    }
  }

  let expectedOffset = 0;
  for (const region of physical.regions) {
    assert.equal(region.offsetBytes, expectedOffset);
    assert.equal(region.offsetBytes % physical.channelAlignmentBytes, 0);
    assert.equal(region.allocatedBytes % physical.channelAlignmentBytes, 0);
    assert.equal(region.payloadBytes, region.payloadElements * physical.scalarBytes);
    assert.ok(region.payloadBytes <= region.allocatedBytes);
    expectedOffset += region.allocatedBytes;
  }
  assert.equal(expectedOffset, physical.totalBytes);

  assert.deepEqual(physical.worklistLevels.map((level) => level.capacityElements),
    logical.levelVolumes);
  for (const level of physical.worklistLevels) {
    assert.equal(
      level.acceptedIndicesOffsetBytes,
      physical.acceptedOccupiedIndices.offsetBytes
        + logical.levelBases[level.level]! * physical.scalarBytes,
    );
    assert.equal(
      level.candidateIndicesOffsetBytes,
      physical.candidateOccupiedIndices.offsetBytes
        + logical.levelBases[level.level]! * physical.scalarBytes,
    );
    assert.equal(level.acceptedDispatchOffsetBytes, level.acceptedCountOffsetBytes + 4);
    assert.equal(level.candidateDispatchOffsetBytes, level.candidateCountOffsetBytes + 4);
  }

  assert.equal(physical.channels.reduce((sum, channel) => sum + channel.payloadBytes, 0)
    + physical.spectralUpper.payloadBytes, 168_536);
  assert.equal(
    physical.acceptedOccupiedIndices.payloadBytes
      + physical.candidateOccupiedIndices.payloadBytes
      + physical.acceptedOccupiedCounts.payloadBytes
      + physical.candidateOccupiedCounts.payloadBytes,
    37_608,
  );
  assert.equal(physical.publicationControl.payloadBytes, 32);
  assert.equal(physical.candidateSpectralUpper.payloadBytes, 20);
  assert.equal(physical.payloadBytes, 206_196);
  assert.equal(physical.paddingBytes, 3_468);
  assert.equal(physical.totalBytes, 209_664);
  assert.equal([1475, 214, 35, 8, 1].reduce((sum, count) => sum + count, 0) * 4, 6_932,
    "one captured occupied-index publication has 6.77 KiB of live entries");
});

test("aggregate restriction and prolongation are exact transposes on an odd sparse image", () => {
  const hierarchy = buildFactorOneDensePressureHierarchy([5, 3, 2], [
    { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 0 },
    { level: 0, coordinate: [1, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 3 },
    { level: 0, coordinate: [4, 2, 1], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 2 },
    { level: 0, coordinate: [3, 2, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly },
  ]);
  const fineLevel = 0;
  const fine = Float64Array.from(
    { length: hierarchy.plan.levelVolumes[fineLevel]! },
    (_, index) => (index * 17 % 23) / 7 - 1,
  );
  const coarse = Float64Array.from(
    { length: hierarchy.plan.levelVolumes[fineLevel + 1]! },
    (_, index) => (index * 11 % 19) / 5 + 0.25,
  );
  const restricted = restrictFactorOneDenseAggregate(hierarchy, fineLevel, fine);
  const prolonged = prolongFactorOneDenseAggregate(hierarchy, fineLevel, coarse);
  assert.equal(dot(restricted, coarse), dot(fine, prolonged));

  const parent = factorOneDenseSlot(hierarchy.plan, 1, [0, 0, 0]);
  const expected = factorOneDenseAggregateChildren(hierarchy.plan, 0, parent)
    .filter((slot) => hierarchy.flags[slot] !== 0)
    .reduce((sum, slot) => sum + fine[slot]!, 0);
  assert.equal(restricted[parent - hierarchy.plan.levelBases[1]!], expected);
});

test("matrix-free six-face apply equals an independently assembled sparse matrix", () => {
  const hierarchy = buildFactorOneDensePressureHierarchy([5, 4, 3], [
    { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 0 },
    { level: 0, coordinate: [1, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 4 },
    { level: 0, coordinate: [2, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly },
    { level: 0, coordinate: [2, 1, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 2 },
    { level: 0, coordinate: [4, 3, 2], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 8 },
    { level: 0, coordinate: [4, 2, 2], role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 8 },
  ]);
  const level = 0, base = hierarchy.plan.levelBases[level]!;
  const volume = hierarchy.plan.levelVolumes[level]!, coefficient = 0.375;
  const input = Float64Array.from({ length: volume }, (_, index) => (index * 13 % 29) / 9 - 0.7);
  const diagonal = new Float64Array(volume);
  for (let local = 0; local < volume; local += 1) {
    diagonal[local] = hierarchy.flags[base + local] === 0 ? 0 : 2.25 + (local % 5) * 0.125;
  }
  const actual = applyFactorOneDenseSixFace(hierarchy, level, input, diagonal, coefficient);

  // Independent reference: compact occupied coordinates, assemble an explicit
  // sparse matrix, then multiply in compact-column order.
  const occupied: Array<{ local: number; coordinate: FactorOneDenseCoordinate }> = [];
  const byCoordinate = new Map<string, number>();
  for (let local = 0; local < volume; local += 1) if (hierarchy.flags[base + local] !== 0) {
    const coordinate = factorOneDenseCoordinate(hierarchy.plan, level, base + local);
    byCoordinate.set(coordinate.join(","), occupied.length);
    occupied.push({ local, coordinate });
  }
  const matrix = Array.from({ length: occupied.length }, () =>
    new Float64Array(occupied.length));
  const directions = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0],
    [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ] as const;
  occupied.forEach((cell, row) => {
    matrix[row]![row] = diagonal[cell.local]!;
    for (const direction of directions) {
      const key = [
        cell.coordinate[0] + direction[0],
        cell.coordinate[1] + direction[1],
        cell.coordinate[2] + direction[2],
      ].join(",");
      const column = byCoordinate.get(key);
      if (column !== undefined) matrix[row]![column] -= coefficient;
    }
  });
  for (let row = 0; row < matrix.length; row += 1) {
    for (let column = 0; column < matrix.length; column += 1) {
      assert.equal(matrix[row]![column], matrix[column]![row]);
    }
    let expected = 0;
    for (let column = 0; column < matrix.length; column += 1) {
      expected += matrix[row]![column]! * input[occupied[column]!.local]!;
    }
    assert.ok(Math.abs(actual[occupied[row]!.local]! - expected) < 1e-12);
  }
  for (let local = 0; local < volume; local += 1) {
    if (hierarchy.flags[base + local] === 0) assert.equal(actual[local], 0);
  }
});
