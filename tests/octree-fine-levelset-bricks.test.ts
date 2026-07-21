import assert from "node:assert/strict";
import test from "node:test";
import {
  FINE_LEVELSET_INVALID,
  FineLevelSetBrickOracle,
  fineLevelSetAddressAtCoordinate,
  fineLevelSetAddressAtPosition,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
  unpackFineLevelSetBrickKey,
} from "../lib/octree-fine-levelset-bricks";

function plan(fineFactor: 4 | 8 = 4) {
  return planFineLevelSetBricks({
    domainOrigin: [10, -2, 5], finestCellDimensions: [8, 8, 8], finestCellWidth: 2,
    fineFactor, brickResolution: 4, maximumResidentBricks: 64,
  });
}

test("factor-4/factor-8 plans expose exact global-lattice and four-channel memory", () => {
  const factor4 = plan(4);
  const factor8 = plan(8);
  assert.deepEqual(factor4.sampleDimensions, [32, 32, 32]);
  assert.deepEqual(factor4.brickDimensions, [8, 8, 8]);
  assert.equal(factor4.fineCellWidth, 0.5);
  assert.equal(factor4.samplesPerBrick, 64);
  assert.equal(factor4.payloadBytesPerBrick, 64 * 16);
  assert.deepEqual(factor8.sampleDimensions, [64, 64, 64]);
  assert.equal(factor8.fineCellWidth, 0.25);
  assert.equal(factor8.payloadBytesPerBrick, factor4.payloadBytesPerBrick);
  assert.equal(factor4.pageTableBytes, 2 * factor4.hashCapacity * 8);
});

test("packed brick keys are range checked, invertible, and independent of leaf rows", () => {
  const configured = plan();
  const key = packFineLevelSetBrickKey(configured, [3, 4, 5]);
  assert.deepEqual(unpackFineLevelSetBrickKey(configured, key), [3, 4, 5]);
  const position = [10 + 7.2 * 0.5, -2 + 9.1 * 0.5, 5 + 11.8 * 0.5] as const;
  const beforeOctreeRebuild = fineLevelSetAddressAtPosition(configured, position);
  // No octree row/generation is an input: rebuilding row ownership cannot alter the address.
  const afterOctreeRebuild = fineLevelSetAddressAtPosition(configured, position);
  assert.deepEqual(afterOctreeRebuild, beforeOctreeRebuild);
  assert.deepEqual(beforeOctreeRebuild?.fineCoord, [7, 9, 11]);
  assert.throws(() => packFineLevelSetBrickKey(configured, [-1, 0, 0]), /outside/);
  assert.throws(() => packFineLevelSetBrickKey(configured, [8, 0, 0]), /outside/);
  assert.throws(() => unpackFineLevelSetBrickKey(configured, FINE_LEVELSET_INVALID), /outside/);
  assert.equal(fineLevelSetAddressAtPosition(configured, [9.99, 0, 6]), undefined);
  assert.throws(() => planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [65_536, 65_536, 2], finestCellWidth: 1,
    fineFactor: 8, brickResolution: 4, maximumResidentBricks: 1,
  }), /32-bit key/);
});

test("interface publication deterministically builds a six-neighbor ring and caches physical IDs", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  const centerKey = packFineLevelSetBrickKey(configured, [3, 3, 3]);
  const publication = oracle.publishInterfaceAndRing([centerKey, centerKey], ([x]) => x - 14);
  assert.equal(publication.generation, 1);
  assert.equal(publication.interfaceKeys.length, 1);
  assert.equal(publication.desiredKeys.length, 7);
  assert.deepEqual([...publication.desiredKeys], [...publication.desiredKeys].sort((a, b) => a - b));
  const center = oracle.pageForKey(centerKey); assert.ok(center);
  for (let direction = 0; direction < 6; direction += 1) {
    assert.notEqual(center.neighborIds[direction], FINE_LEVELSET_INVALID);
  }
  const minusX = oracle.pageForKey(packFineLevelSetBrickKey(configured, [2, 3, 3])); assert.ok(minusX);
  assert.equal(minusX.neighborIds[1], center.physicalId);
  assert.equal(publication.activePhysicalIds.length, 7);
});

test("two generations reuse stable keys, retire old-only pages, and initialize new pages from coarse phi", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  const first = packFineLevelSetBrickKey(configured, [2, 2, 2]);
  const second = packFineLevelSetBrickKey(configured, [3, 2, 2]);
  oracle.publishInterfaceAndRing([first], () => -2);
  const stablePageId = oracle.pageForKey(second)?.physicalId; assert.notEqual(stablePageId, undefined);
  const next = oracle.publishInterfaceAndRing([second], () => 7);
  assert.equal(next.generation, 2);
  assert.equal(oracle.pageForKey(second)?.physicalId, stablePageId);
  assert.ok(next.reusedPages > 0);
  assert.ok(next.retiredPages > 0);
  const newlyActivated = [...next.desiredKeys]
    .map((key) => oracle.pageForKey(key)!)
    .find((page) => page.phi[0] === 7);
  assert.ok(newlyActivated);
  assert.ok(newlyActivated.phi.every(Number.isFinite));
});

test("missing bricks use explicit coarse phi and interface discovery scans residents only", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  const key = packFineLevelSetBrickKey(configured, [1, 1, 1]);
  oracle.publishInterfaceAndRing([key], () => 3);
  const page = oracle.pageForKey(key); assert.ok(page);
  page.phi[21] = -1;
  page.phi[22] = 1;
  assert.deepEqual([...oracle.detectInterfaceKeys()], [key]);
  page.phi[0] = -1;
  const residentAddress = fineLevelSetAddressAtCoordinate(configured, [4, 4, 4]);
  const residentPosition = residentAddress.fineCoord.map((q, axis) => configured.domainOrigin[axis] + (q + 0.1) * configured.fineCellWidth) as [number, number, number];
  assert.equal(oracle.sampleOrCoarse(residentPosition, () => 99), -1);
  const missingPosition = [configured.domainOrigin[0] + 30.1 * configured.fineCellWidth,
    configured.domainOrigin[1] + 30.1 * configured.fineCellWidth,
    configured.domainOrigin[2] + 30.1 * configured.fineCellWidth] as const;
  assert.equal(oracle.sampleOrCoarse(missingPosition, () => 99), 99);
  assert.equal(oracle.sampleOrCoarse([0, 0, 0], () => -99), -99);
});

test("interface discovery detects a sign change exactly across cached brick neighbors", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  const leftKey = packFineLevelSetBrickKey(configured, [2, 2, 2]);
  oracle.publishInterfaceAndRing([leftKey], () => 1);
  const rightKey = packFineLevelSetBrickKey(configured, [3, 2, 2]);
  const left = oracle.pageForKey(leftKey); const right = oracle.pageForKey(rightKey);
  assert.ok(left); assert.ok(right);
  left.phi.fill(-1);
  right.phi.fill(1);
  // Other ring pages remain positive, so the required pair is a subset even
  // though left also crosses its other resident page boundaries.
  const detected = new Set(oracle.detectInterfaceKeys());
  assert.ok(detected.has(leftKey));
  assert.ok(detected.has(rightKey));
  assert.equal(left.neighborIds[1], right.physicalId);
});

test("memory accounting separates active payload, hash/worklists, and fragmentation", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(configured, [0, 0, 0])], () => 1);
  const memory = oracle.memoryAccounting();
  assert.equal(memory.residentBricks, 4);
  assert.equal(memory.activePayloadBytes, 4 * configured.payloadBytesPerBrick);
  assert.equal(memory.fragmentationBytes, configured.payloadCapacityBytes - memory.activePayloadBytes);
  assert.equal(memory.pageTableBytes, configured.pageTableBytes);
  assert.equal(memory.worklistBytes, configured.worklistBytes);
  assert.equal(memory.allocatedBytes, configured.allocatedBytes);
  const gpu = oracle.exportGPUGeneration();
  assert.equal(gpu.hashPairs.length, configured.hashCapacity * 2);
  assert.equal(gpu.metadataWords.length, configured.maximumResidentBricks * 10);
  assert.equal(gpu.worklistWords[0], 4);
});

test("publication fails closed before mutating the current generation", () => {
  const configured = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 3,
  });
  const oracle = new FineLevelSetBrickOracle(configured);
  assert.throws(() => oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(configured, [0, 0, 0])], () => 1), /capacity/);
  assert.equal(oracle.generation, 0);
  assert.equal(oracle.residentBrickCount, 0);
});

test("coarse initialization failure cannot partially publish or retag reused pages", () => {
  const configured = plan();
  const oracle = new FineLevelSetBrickOracle(configured);
  const firstKey = packFineLevelSetBrickKey(configured, [2, 2, 2]);
  oracle.publishInterfaceAndRing([firstKey], () => 1);
  const firstPage = oracle.pageForKey(firstKey); assert.ok(firstPage);
  assert.throws(() => oracle.publishInterfaceAndRing(
    [packFineLevelSetBrickKey(configured, [3, 2, 2])],
    () => { throw new Error("coarse unavailable"); },
  ), /coarse unavailable/);
  assert.equal(oracle.generation, 1);
  assert.equal(oracle.pageForKey(firstKey), firstPage);
  assert.equal(firstPage.generation, 1);
});
