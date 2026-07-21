import assert from "node:assert/strict";
import test from "node:test";
import {
  buildOctreeFaceIncidence,
  buildCanonicalOctreeFaceFragments,
  OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS,
  octreeLeafNetFluxes,
  restrictOctreeFaceVelocity,
  type OctreeFaceLeaf,
} from "../lib/octree-face-fragments";

const coarseFineLeaves: OctreeFaceLeaf[] = [
  { id: 0, origin: [0, 0, 0], size: 2 },
  { id: 1, origin: [2, 0, 0], size: 1 },
  { id: 2, origin: [2, 1, 0], size: 1 },
  { id: 3, origin: [2, 0, 1], size: 1 },
  { id: 4, origin: [2, 1, 1], size: 1 },
];

test("canonical octree faces split a 2:1 interface into four unique fragments", () => {
  const fragments = buildCanonicalOctreeFaceFragments(coarseFineLeaves, [3, 2, 2]);
  const seam = fragments.filter((fragment) => fragment.axis === 0
    && fragment.origin[0] === 2
    && fragment.negativeLeaf === 0
    && fragment.positiveLeaf !== null);
  assert.equal(OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS, 4);
  assert.equal(seam.length, 4);
  assert.deepEqual(seam.map((fragment) => fragment.positiveLeaf).sort(), [1, 2, 3, 4]);
  assert.ok(seam.every((fragment) => fragment.span === 1 && fragment.areaFineFaces === 1));
});

test("coarse restriction preserves exactly the sum of fine fragment fluxes", () => {
  const fragments = buildCanonicalOctreeFaceFragments(coarseFineLeaves, [3, 2, 2]);
  const seam = fragments.filter((fragment) => fragment.axis === 0 && fragment.origin[0] === 2 && fragment.negativeLeaf === 0);
  const velocities = [1, 2, 3, 4];
  const restricted = restrictOctreeFaceVelocity(seam, velocities);
  assert.equal(restricted, 2.5);
  assert.equal(restricted * 4, velocities.reduce((sum, velocity) => sum + velocity, 0));
});

test("one canonical fragment contributes equal and opposite internal flux", () => {
  const fragments = buildCanonicalOctreeFaceFragments(coarseFineLeaves, [3, 2, 2]);
  const seamIndices = fragments.flatMap((fragment, index) => fragment.axis === 0
    && fragment.origin[0] === 2
    && fragment.negativeLeaf === 0 ? [index] : []);
  const velocities = fragments.map(() => 0);
  seamIndices.forEach((fragmentIndex, index) => { velocities[fragmentIndex] = index + 1; });
  const net = octreeLeafNetFluxes(coarseFineLeaves, fragments, velocities);
  assert.equal(net.get(0), 10);
  assert.deepEqual([net.get(1), net.get(2), net.get(3), net.get(4)].sort((a, b) => a! - b!), [-4, -3, -2, -1]);
  assert.equal([...net.values()].reduce((sum, flux) => sum + flux, 0), 0);
});

test("row incidence reuses one signed face index on both sides of an interface", () => {
  const fragments = buildCanonicalOctreeFaceFragments(coarseFineLeaves, [3, 2, 2]);
  const incidence = buildOctreeFaceIncidence(coarseFineLeaves, fragments);
  const seam = fragments.filter((fragment) => fragment.axis === 0 && fragment.origin[0] === 2 && fragment.negativeLeaf === 0);
  const coarseEntries = Array.from(incidence.fragmentIndices.slice(incidence.offsets[0], incidence.offsets[1]));
  for (const fragment of seam) {
    assert.ok(coarseEntries.includes(fragment.index));
    const fineRow = coarseFineLeaves.findIndex((leaf) => leaf.id === fragment.positiveLeaf);
    const begin = incidence.offsets[fineRow];
    const end = incidence.offsets[fineRow + 1];
    const local = Array.from(incidence.fragmentIndices.slice(begin, end)).indexOf(fragment.index);
    assert.notEqual(local, -1);
    assert.equal(incidence.signs[begin + local], -1);
  }
  const coarseSigns = seam.map((fragment) => {
    const local = coarseEntries.indexOf(fragment.index);
    return incidence.signs[incidence.offsets[0] + local];
  });
  assert.deepEqual(coarseSigns, [1, 1, 1, 1]);
});

test("face construction rejects an unbalanced 4:1 transition", () => {
  const leaves: OctreeFaceLeaf[] = [{ id: 0, origin: [0, 0, 0], size: 4 }];
  let id = 1;
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1) {
    leaves.push({ id, origin: [4, y, z], size: 1 }); id += 1;
  }
  assert.throws(() => buildCanonicalOctreeFaceFragments(leaves, [5, 4, 4]), /four-fragment bound|2:1 balance/);
});

test("face construction rejects gaps and overlapping leaves", () => {
  assert.throws(() => buildCanonicalOctreeFaceFragments([{ id: 0, origin: [0, 0, 0], size: 1 }], [2, 1, 1]), /do not cover/);
  assert.throws(() => buildCanonicalOctreeFaceFragments([
    { id: 0, origin: [0, 0, 0], size: 2 },
    { id: 1, origin: [1, 0, 0], size: 1 },
  ], [2, 2, 2]), /overlaps/);
});
