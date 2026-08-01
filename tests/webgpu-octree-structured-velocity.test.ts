import assert from "node:assert/strict";
import test from "node:test";
import {
  
  buildResolvedPowerRowsPublication,
  planResolvedPowerRows,
} from "../lib/webgpu-octree-power-resolved-rows";
import { packOctreePowerRowTemplateSlot } from "../lib/octree-power-catalog";
import {
  OCTREE_STRUCTURED_VELOCITY_CLASSES,
  OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE,
  advectStructuredVelocityDestinationOwned,
  projectStructuredVelocity,
  publishStructuredVelocityFamilies,
  reconstructStructuredRowVelocity,
  structuredVelocityFamilyHandlesForRow,
  structuredVelocityDivergence,
  type StructuredVelocityCatalogSource,
} from "../lib/webgpu-octree-structured-velocity";
import { unpackOctreeWorksetHeader } from "../lib/webgpu-octree-worksets";

const INVALID = OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE;
const noFamilyHandles = [INVALID, INVALID, INVALID, INVALID, INVALID, INVALID];

function fixtureCatalog(): StructuredVelocityCatalogSource {
  const slots = 12;
  const rowTemplateHeaders = new Uint32Array([
    0, 6, 0, 6,
    6, 6, 0, 6,
  ]);
  const rowTemplateSlots = new Uint32Array(slots);
  const rowTemplateData = new Float32Array(slots * 3);
  const reconstructionData = new Float32Array(slots * 3);
  const faceData = new Float32Array(slots * 12);
  const normals = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ] as const;
  for (let entry = 0; entry < 2; entry += 1) {
    for (let local = 0; local < 6; local += 1) {
      const slot = entry * 6 + local;
      const normalIndex = entry === 0 ? local : (local === 0 ? 1 : local === 1 ? 0 : local);
      const normal = normals[normalIndex]!;
      const axis = normal.findIndex((value) => value !== 0);
      const orientation = normal[axis]! > 0 ? 1 : 0;
      rowTemplateSlots[slot] = packOctreePowerRowTemplateSlot({
        neighborSelector: local === 0 ? 0 : 0xff,
        family: axis,
        orientation,
        transferRelation: local === 0 ? 0 : 3,
        dynamicBoundarySlot: local,
        worldBoundary: local !== 0,
      });
      rowTemplateData.set([1, 1, 1], slot * 3);
      reconstructionData.set(normal.map((value) => 0.5 * value), slot * 3);
      faceData.set(normal, slot * 12 + 8);
    }
  }
  return { rowTemplateHeaders, rowTemplateSlots, rowTemplateData, reconstructionData, faceData };
}

function resolved(epoch = 7, swapped = false) {
  const plan = planResolvedPowerRows(4, 6);
  return buildResolvedPowerRowsPublication(plan, epoch, [
    {
      row: 0, caseId: 1, catalogEntry: swapped ? 1 : 0, physicalPage: 0, localCell: 0,
      volume: 1, scale: 1, neighborRows: [1], neighborCoefficients: [1],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
    {
      row: 1, caseId: 1, catalogEntry: swapped ? 0 : 1, physicalPage: 0, localCell: 1,
      volume: 1, scale: 1, neighborRows: [0], neighborCoefficients: [1],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
  ]);
}

const constant = [2, -3, 5] as const;
const normals0 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const;
const dot = (left: readonly number[], right: readonly number[]) =>
  left.reduce((sum, value, axis) => sum + value * right[axis]!, 0);

function publication(epoch = 7) {
  return publishStructuredVelocityFamilies(resolved(epoch), fixtureCatalog(), [
    {
      row: 0,
      neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID],
      normalVelocities: normals0.map((normal) => dot(constant, normal)),
    },
    {
      row: 1,
      neighborRows: [0, INVALID, INVALID, INVALID, INVALID, INVALID],
      normalVelocities: [dot(constant, normals0[1]), dot(constant, normals0[0]),
        ...normals0.slice(2).map((normal) => dot(constant, normal))],
    },
  ]);
}

test("six family SoAs canonically share reciprocal slots without scatter", () => {
  const result = publication();
  assert.equal(result.slotCount, 11, "the internal face is represented once");
  assert.equal(result.familyOffsets.length, 7);
  for (let family = 0; family < 6; family += 1) {
    assert.ok(result.familyOffsets[family]! <= result.familyOffsets[family + 1]!);
  }
  const row0 = result.rowSlotHandles[0]!;
  const row1 = result.rowSlotHandles[result.maximumCaseSlots]!;
  assert.equal(row0, row1, "both incidences address one canonical scalar");
  assert.deepEqual([result.rowSlotSigns[0], result.rowSlotSigns[result.maximumCaseSlots]], [1, -1]);
  assert.equal(result.neighborRows[row0], 1);
  assert.equal(result.ownerRows[row0], 0);
  assert.equal(result.coefficients[row0], 1);
  assert.equal(result.boundaryFractions[row0], 1);
  assert.equal(result.orientation[row0]! & 7, 0);
  assert.equal(result.orientation[row0]! >>> 3 & 7, 1);
  const sixHandles = structuredVelocityFamilyHandlesForRow(result, 0);
  assert.equal(sixHandles.length, 6);
  assert.equal(result.rowFamilySlotHandles[sixHandles[0]! + 1], row0);
  assert.equal(result.rowFamilySlotCounts[0], 2);
  const classCounts = OCTREE_STRUCTURED_VELOCITY_CLASSES.map((name) =>
    unpackOctreeWorksetHeader(result.familyWorksets[name]).count);
  assert.deepEqual(classCounts, [0, 0, 11, 0]);
  assert.equal(new Set(result.slotStableKeys).size, result.slotCount);
});

test("publication order is deterministic and family worksets remain disjoint", () => {
  const source = resolved();
  const catalog = fixtureCatalog();
  const inputs = [
    { row: 0, neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID] },
    { row: 1, neighborRows: [0, INVALID, INVALID, INVALID, INVALID, INVALID] },
  ] as const;
  const forward = publishStructuredVelocityFamilies(source, catalog, inputs);
  const reverse = publishStructuredVelocityFamilies(source, catalog, [...inputs].reverse());
  for (const key of ["familyOffsets", "ownerRows", "neighborRows", "orientation", "coefficients",
    "inverseDistances", "boundaryFractions", "rowFamilyHandles", "rowFamilySlotHandles",
    "rowFamilySlotCounts", "rowSlotHandles", "rowSlotSigns", "rowCatalogSlots"] as const) {
    assert.deepEqual([...forward[key]], [...reverse[key]], `${key} must not depend on publisher input order`);
  }
  const handles = OCTREE_STRUCTURED_VELOCITY_CLASSES.flatMap((name) => {
    const workset = forward.familyWorksets[name];
    const count = unpackOctreeWorksetHeader(workset).count;
    return [...workset.slice(7, 7 + count)];
  });
  assert.equal(handles.length, forward.slotCount);
  assert.equal(new Set(handles).size, forward.slotCount);
});

test("catalog pseudoinverse reproduces constants and fixed gathers preserve zero divergence", () => {
  const result = publication();
  const catalog = fixtureCatalog();
  for (const row of [0, 1]) {
    const reconstructed = reconstructStructuredRowVelocity(result, catalog, row);
    reconstructed.forEach((value, axis) => assert.ok(Math.abs(value - constant[axis]!) < 1e-6));
    assert.ok(Math.abs(structuredVelocityDivergence(result, row, 1)) < 1e-6);
  }
});

test("projection and advection are unique destination writes and affine-exact", () => {
  const result = publication();
  const projected = projectStructuredVelocity(result, new Float32Array([2, 5]), 0.25);
  const shared = result.rowSlotHandles[0]!;
  assert.equal(projected[shared], Math.fround(result.values[shared]! - 0.75));
  for (let handle = 0; handle < result.slotCount; handle += 1) {
    if (handle !== shared) assert.equal(projected[handle], result.values[handle]);
  }
  const centroid = (handle: number) => [0.125 * handle, -0.25 * handle, 0.5 * handle] as const;
  const affine = (handle: number) => {
    const point = centroid(handle);
    return [1 + point[0] - 2 * point[1], -2 + 0.5 * point[1] + point[2],
      3 - point[0] + 0.25 * point[2]] as const;
  };
  const advected = advectStructuredVelocityDestinationOwned(result, affine);
  for (let handle = 0; handle < result.slotCount; handle += 1) {
    assert.equal(advected[handle], Math.fround(dot(affine(handle), [
      result.normalX[handle]!, result.normalY[handle]!, result.normalZ[handle]!,
    ])));
  }
});

test("unchanged values carry bit-exactly through an old-to-new row remap", () => {
  const first = publication(7);
  first.values.forEach((_, handle) => { first.values[handle] = Math.fround(handle * 0.125 - 0.75); });
  const next = publishStructuredVelocityFamilies(resolved(8, true), fixtureCatalog(), [
    { row: 0, neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID] },
    { row: 1, neighborRows: [0, INVALID, INVALID, INVALID, INVALID, INVALID] },
  ], { previous: first, oldToNewRows: new Uint32Array([1, 0, INVALID, INVALID]) });
  const oldByRemappedKey = new Map<string, number>();
  first.slotStableKeys.forEach((key, handle) => {
    const parts = key.split(":");
    const remapped = parts[0] === "i" ? "i:0:1"
      : `b:${parts[1] === "0" ? 1 : 0}:${parts.slice(2).join(":")}`;
    oldByRemappedKey.set(remapped, handle);
  });
  next.slotStableKeys.forEach((key, handle) => {
    const old = oldByRemappedKey.get(key);
    assert.notEqual(old, undefined);
    assert.equal(new Uint32Array(next.values.buffer)[handle], new Uint32Array(first.values.buffer)[old!]);
  });
});

test("publisher rejects missing reciprocity and coefficient asymmetry instead of falling back", () => {
  const catalog = fixtureCatalog();
  assert.throws(() => publishStructuredVelocityFamilies(resolved(), catalog, [
    { row: 0, neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID] },
    { row: 1, neighborRows: [INVALID, INVALID, INVALID, INVALID, INVALID, INVALID] },
  ]), /boundary selector and neighbor disagree|two incidences/);
  const asymmetric = buildResolvedPowerRowsPublication(planResolvedPowerRows(4, 6), 7, [
    {
      row: 0, caseId: 1, catalogEntry: 0, physicalPage: 0, localCell: 0,
      volume: 1, scale: 1, neighborRows: [1], neighborCoefficients: [1],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
    {
      row: 1, caseId: 1, catalogEntry: 1, physicalPage: 0, localCell: 1,
      volume: 1, scale: 1, neighborRows: [0], neighborCoefficients: [1 + 2 ** -20],
      velocityFamilyHandles: noFamilyHandles, transition: false, physicalBoundary: true,
    },
  ]);
  assert.throws(() => publishStructuredVelocityFamilies(asymmetric, catalog, [
    { row: 0, neighborRows: [1, INVALID, INVALID, INVALID, INVALID, INVALID] },
    { row: 1, neighborRows: [0, INVALID, INVALID, INVALID, INVALID, INVALID] },
  ]), /asymmetric/);
});
