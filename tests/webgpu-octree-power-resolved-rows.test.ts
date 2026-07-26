import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_POWER_ROW_CLASS,
  buildResolvedPowerRowsPublication,
  classifyExclusivePowerGrading,
  planResolvedPowerRows,
  validateResolvedPowerReciprocity,
} from "../lib/webgpu-octree-power-resolved-rows";
import { OCTREE_WORKSET_INVALID_ID, unpackOctreeWorksetHeader } from "../lib/webgpu-octree-worksets";

const handles = [0, 1, 2, OCTREE_WORKSET_INVALID_ID,
  OCTREE_WORKSET_INVALID_ID, OCTREE_WORKSET_INVALID_ID];

test("exclusive grading rejects a mixed finer/coarser 1-ring before case assignment", () => {
  assert.equal(classifyExclusivePowerGrading(3, [3, 4, 3]), "same-or-finer");
  assert.equal(classifyExclusivePowerGrading(3, [2, 3, 3]), "same-or-coarser");
  assert.equal(classifyExclusivePowerGrading(3, [3, 3]), "same");
  assert.throws(() => classifyExclusivePowerGrading(3, [2, 4]), /mixes/);
  assert.throws(() => classifyExclusivePowerGrading(3, [5]), /2:1/);
});

test("resolved rows publish four disjoint worksets plus diagnostic invalid rows", () => {
  const plan = planResolvedPowerRows(8, 6);
  const publication = buildResolvedPowerRowsPublication(plan, 9, [
    { row: 0, caseId: 0, catalogEntry: 0, physicalPage: 0, localCell: 0,
      volume: 1, scale: 1, neighborRows: [1], neighborCoefficients: [2],
      velocityFamilyHandles: handles, transition: false, physicalBoundary: false },
    { row: 1, caseId: 3, catalogEntry: 7, physicalPage: 0, localCell: 1,
      volume: 0.75, scale: 1, neighborRows: [0], neighborCoefficients: [2],
      velocityFamilyHandles: handles, transition: true, physicalBoundary: false },
    { row: 2, caseId: 0, catalogEntry: 0, physicalPage: 0, localCell: 2,
      volume: 0.5, scale: 1, neighborRows: [3], neighborCoefficients: [1],
      velocityFamilyHandles: handles, transition: false, physicalBoundary: true,
      dynamicBoundaryFractions: [0.5] },
    { row: 3, caseId: 4, catalogEntry: 8, physicalPage: 0, localCell: 3,
      volume: 0.5, scale: 1, neighborRows: [2], neighborCoefficients: [1],
      velocityFamilyHandles: handles, transition: true, physicalBoundary: true },
    { row: 4, caseId: 5, catalogEntry: 9, physicalPage: 0, localCell: 4,
      volume: 1, scale: 1, neighborRows: [], neighborCoefficients: [],
      velocityFamilyHandles: handles, transition: true, physicalBoundary: false, valid: false },
  ]);
  const headers = [publication.regularInteriorRows, publication.transitionInteriorRows,
    publication.physicalBoundaryRows, publication.transitionBoundaryRows, publication.invalidRows]
    .map(unpackOctreeWorksetHeader);
  assert.deepEqual(headers.map((header) => header.count), [1, 1, 1, 1, 1]);
  assert.deepEqual(publication.rows.map((row) => row.rowClass), [
    OCTREE_POWER_ROW_CLASS.regularInterior, OCTREE_POWER_ROW_CLASS.transitionInterior,
    OCTREE_POWER_ROW_CLASS.physicalBoundary, OCTREE_POWER_ROW_CLASS.transitionBoundary,
    OCTREE_POWER_ROW_CLASS.invalid,
  ]);
  assert.deepEqual(validateResolvedPowerReciprocity(publication), []);
  assert.equal(publication.packedRows.length, plan.rowCapacity * plan.rowStrideWords);
});

test("empty row classes emit zero indirect work", () => {
  const plan = planResolvedPowerRows(2, 6);
  const publication = buildResolvedPowerRowsPublication(plan, 1, [{
    row: 0, caseId: 0, catalogEntry: 0, physicalPage: 0, localCell: 0,
    volume: 1, scale: 1, neighborRows: [], neighborCoefficients: [],
    velocityFamilyHandles: handles, transition: false, physicalBoundary: false,
  }]);
  assert.deepEqual(unpackOctreeWorksetHeader(publication.transitionBoundaryRows).dispatch, [0, 1, 1]);
});

test("resolved rows fail closed on sparse prefixes and asymmetric coefficients", () => {
  const plan = planResolvedPowerRows(4, 6);
  assert.throws(() => buildResolvedPowerRowsPublication(plan, 1, [{
    row: 1, caseId: 0, catalogEntry: 0, physicalPage: 0, localCell: 0,
    volume: 1, scale: 1, neighborRows: [], neighborCoefficients: [],
    velocityFamilyHandles: handles, transition: false, physicalBoundary: false,
  }]), /dense live prefix/);
  const publication = buildResolvedPowerRowsPublication(plan, 1, [
    { row: 0, caseId: 0, catalogEntry: 0, physicalPage: 0, localCell: 0,
      volume: 1, scale: 1, neighborRows: [1], neighborCoefficients: [1],
      velocityFamilyHandles: handles, transition: false, physicalBoundary: false },
    { row: 1, caseId: 1, catalogEntry: 1, physicalPage: 0, localCell: 1,
      volume: 1, scale: 1, neighborRows: [0], neighborCoefficients: [1 + 2 ** -20],
      velocityFamilyHandles: handles, transition: true, physicalBoundary: false },
  ]);
  assert.match(validateResolvedPowerReciprocity(publication).join("; "), /coefficient bits differ/);
});
