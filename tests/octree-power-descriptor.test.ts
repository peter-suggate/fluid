import assert from "node:assert/strict";
import test from "node:test";
import { constructOctreePowerCell } from "../lib/octree-power-geometry";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_FINER_MASK,
  buildCanonicalSameOrFinerPowerDescriptorLookup,
  canonicalizeSameOrFinerPowerDescriptor,
  decodeSameOrFinerPowerDescriptor,
  decodeSameOrCoarserPowerDescriptor,
  encodeSameOrFinerPowerDescriptor,
  encodeSameOrCoarserPowerDescriptor,
  enumerateCanonicalSameOrFinerPowerDescriptors,
  sitesForSameOrFinerPowerDescriptor,
  sitesForSameOrCoarserPowerDescriptor,
  transformSameOrFinerPowerDescriptor,
} from "../lib/octree-power-descriptor";
import { OCTREE_CUBE_TRANSFORMS } from "../lib/octree-power-topology";

test("18-bit descriptor covers six face and twelve edge directions", () => {
  assert.equal(OCTREE_POWER_NEIGHBOR_DIRECTIONS.length, 18);
  assert.equal(OCTREE_POWER_NEIGHBOR_DIRECTIONS.filter((direction) => direction.filter(Boolean).length === 1).length, 6);
  assert.equal(OCTREE_POWER_NEIGHBOR_DIRECTIONS.filter((direction) => direction.filter(Boolean).length === 2).length, 12);
  const states = OCTREE_POWER_NEIGHBOR_DIRECTIONS.map((_, index) => index % 3 === 0);
  assert.deepEqual(decodeSameOrFinerPowerDescriptor(encodeSameOrFinerPowerDescriptor(states)), states);
});

test("nine-bit same-or-coarser case reconstructs parity-constrained coarse neighbors", () => {
  const descriptor = encodeSameOrCoarserPowerDescriptor({
    child: [0, 1, 0], coarseNeighbors: [true, false, true, false, true, false],
  });
  assert.deepEqual(decodeSameOrCoarserPowerDescriptor(descriptor), {
    child: [0, 1, 0], coarseNeighbors: [true, false, true, false, true, false],
  });
  const sites = sitesForSameOrCoarserPowerDescriptor(descriptor);
  assert.equal(sites.filter((site) => site.size === 2).length, 3);
  const anchor = sites.find((site) => site.key === "anchor")!;
  const cell = constructOctreePowerCell(anchor, sites);
  assert.ok(cell.volume > 0 && cell.faces.length >= 6);
});

test("all 512 same-or-coarser descriptors reconstruct bounded positive cells", () => {
  for (let child = 0; child < 8; child += 1) for (let mask = 0; mask < 64; mask += 1) {
    const descriptor = encodeSameOrCoarserPowerDescriptor({
      child: [child & 1, (child >> 1) & 1, (child >> 2) & 1] as [0 | 1, 0 | 1, 0 | 1],
      coarseNeighbors: [0, 1, 2, 3, 4, 5].map((bit) => (mask & (1 << bit)) !== 0) as [boolean, boolean, boolean, boolean, boolean, boolean],
    });
    const sites = sitesForSameOrCoarserPowerDescriptor(descriptor);
    const cell = constructOctreePowerCell(sites.find((site) => site.key === "anchor")!, sites);
    assert.ok(cell.volume > 0);
  }
});

test("descriptor reconstruction reaches the paper's 18 same and 48 finer neighbors", () => {
  assert.equal(sitesForSameOrFinerPowerDescriptor(OCTREE_POWER_SAME_OR_FINER_MASK).length, 19);
  assert.equal(sitesForSameOrFinerPowerDescriptor(0).length, 49);
  for (const descriptor of [0, 1, 0x15555, 0x2aaaa, OCTREE_POWER_SAME_OR_FINER_MASK]) {
    const sites = sitesForSameOrFinerPowerDescriptor(descriptor);
    const anchor = sites.find((site) => site.origin.every((value) => value === 0) && site.size === 2)!;
    const cell = constructOctreePowerCell(anchor, sites);
    assert.ok(cell.volume > 0 && cell.faces.length >= 6);
  }
});

test("cube transforms preserve descriptors and canonical representatives", () => {
  for (const descriptor of [0, 1, 0x12345, 0x2aaaa, OCTREE_POWER_SAME_OR_FINER_MASK]) {
    const canonical = canonicalizeSameOrFinerPowerDescriptor(descriptor).descriptor;
    for (const transform of OCTREE_CUBE_TRANSFORMS) {
      const transformed = transformSameOrFinerPowerDescriptor(descriptor, transform);
      assert.equal(canonicalizeSameOrFinerPowerDescriptor(transformed).descriptor, canonical);
    }
  }
});

test("exhaustive same-or-finer symmetry quotient is stable and complete", () => {
  const canonical = enumerateCanonicalSameOrFinerPowerDescriptors();
  assert.equal(canonical.length, 6_456);
  assert.equal(canonical[0], 0);
  assert.equal(canonical.at(-1), OCTREE_POWER_SAME_OR_FINER_MASK);
  const set = new Set(canonical);
  const lookup = buildCanonicalSameOrFinerPowerDescriptorLookup();
  for (let descriptor = 0; descriptor <= OCTREE_POWER_SAME_OR_FINER_MASK; descriptor += 1) {
    assert.ok(set.has(lookup[descriptor]));
  }
});
