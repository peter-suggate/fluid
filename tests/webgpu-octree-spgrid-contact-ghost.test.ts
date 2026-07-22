import assert from "node:assert/strict";
import test from "node:test";
import {
  SPGRID_CELL_FLAG,
  buildSPGridContactLevelOracle,
  type SPGridContactOracle,
  type SPGridLeafOracle,
} from "../lib/webgpu-octree-spgrid-vcycle";

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let result = 0;
  for (let index = 0; index < a.length; index += 1) result += a[index] * b[index];
  return result;
};

const multiply = (matrix: ArrayLike<number>, size: number, vector: ArrayLike<number>) => {
  const result = new Float64Array(size);
  for (let row = 0; row < size; row += 1) for (let column = 0; column < size; column += 1) {
    result[row] += matrix[row * size + column] * vector[column];
  }
  return result;
};

const leaves: readonly SPGridLeafOracle[] = [
  { origin: [0, 0, 0], size: 2 },
  { origin: [2, 0, 0], size: 1 }, // face neighbour, lower quadrant
  { origin: [2, 1, 1], size: 1 }, // face neighbour, upper quadrant
  { origin: [2, 2, 0], size: 1 }, // power-face edge neighbour
];
const contacts: readonly SPGridContactOracle[] = [
  { negative: 0, positive: 1, coefficient: 1.25 },
  { negative: 0, positive: 2, coefficient: 0.75 },
  { negative: 0, positive: 3, coefficient: 0.5 },
];

test("Section 4.2 spawns distinct size-2 contact ghosts for face quadrants and an edge T-junction", () => {
  const level = buildSPGridContactLevelOracle(leaves, contacts, 0);
  const coarseAliases = level.owners.map((owner, slot) => ({ owner, slot })).filter(({ owner }) => owner === 0);
  assert.equal(coarseAliases.length, 3, "a coarse leaf must not collapse all contacts to its origin ghost");
  assert.ok(coarseAliases.every(({ slot }) => (level.flags[slot] & SPGRID_CELL_FLAG.ghost) !== 0));
  assert.deepEqual(coarseAliases.map(({ slot }) => level.coordinates[slot]).sort(), [[1, 0, 0], [1, 1, 0], [1, 1, 1]]);

  const endpointDelta = (owner: number) => {
    const fineSlot = level.owners.indexOf(owner), fine = level.coordinates[fineSlot];
    const coarse = coarseAliases.reduce((best, candidate) => {
      const distance = level.coordinates[candidate.slot].reduce((sum, value, axis) => sum + Math.abs(value - fine[axis]), 0);
      const bestDistance = level.coordinates[best.slot].reduce((sum, value, axis) => sum + Math.abs(value - fine[axis]), 0);
      return distance < bestDistance ? candidate : best;
    });
    const a = level.coordinates[coarse.slot], b = fine;
    return a.map((value, axis) => Math.abs(value - b[axis])).reduce((sum, value) => sum + Number(value !== 0), 0);
  };
  assert.equal(endpointDelta(1), 1, "ordinary face contact is axis adjacent");
  assert.equal(endpointDelta(3), 2, "power contact across an octree edge retains both offset axes");
});

test("GhostValuePropagate and GhostValueAccumulate are exact adjoints for all contact aliases", () => {
  const level = buildSPGridContactLevelOracle(leaves, contacts, 0), leafValues = new Float64Array([0.3, -1.2, 2.1, 0.7]);
  const slotValues = Float64Array.from({ length: level.coordinates.length }, (_, slot) => Math.sin(0.4 + 0.7 * slot));
  const propagated = new Float64Array(level.coordinates.length);
  for (let slot = 0; slot < propagated.length; slot += 1) for (let leaf = 0; leaf < leaves.length; leaf += 1) {
    propagated[slot] += level.propagate[slot * leaves.length + leaf] * leafValues[leaf];
  }
  const accumulated = new Float64Array(leaves.length);
  for (let leaf = 0; leaf < leaves.length; leaf += 1) for (let slot = 0; slot < slotValues.length; slot += 1) {
    accumulated[leaf] += level.accumulate[leaf * slotValues.length + slot] * slotValues[slot];
  }
  assert.ok(Math.abs(dot(propagated, slotValues) - dot(leafValues, accumulated)) < 1e-14);
  for (const { slot } of level.owners.map((owner, slot) => ({ owner, slot })).filter(({ owner }) => owner === 0)) {
    assert.equal(propagated[slot], leafValues[0], "propagation copies the same coarse value to every fine contact ghost");
  }
});

test("assembled face-and-edge T-junction operator is symmetric and strictly positive", () => {
  const level = buildSPGridContactLevelOracle(leaves, contacts, 0, [0.4, 0.2, 0.3, 0.5]);
  const size = leaves.length, operator = level.assembledOperator;
  for (let row = 0; row < size; row += 1) for (let column = 0; column < size; column += 1) {
    assert.ok(Math.abs(operator[row * size + column] - operator[column * size + row]) < 1e-14);
  }
  for (let sample = 0; sample < 16; sample += 1) {
    const value = Float64Array.from({ length: size }, (_, index) => Math.sin((sample + 0.25) * (index + 0.6)));
    const product = multiply(operator, size, value);
    assert.ok(dot(value, product) > 1e-10);
  }
  assert.equal(operator[0 * size + 1], -contacts[0].coefficient);
  assert.equal(operator[0 * size + 3], -contacts[2].coefficient, "edge-neighbour off diagonal must not become diagonal-only");
});
