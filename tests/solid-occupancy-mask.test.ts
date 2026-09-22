import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneLatticeDimensions, solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene, type SolidWorld,
  type SolidWorldVoxelPatch } from "../lib/core/solid-world";

/** The per-cell sampler the mask replaced: the definition of a correct mask. */
function sampledMask(world: SolidWorld, [nx, ny, nz]: readonly [number, number, number]): Uint32Array {
  const sx = nx + 2, sy = ny + 2, sz = nz + 2;
  const words = new Uint32Array(4 + Math.ceil(sx * sy * sz / 32));
  words.set([0x53565731, sx, sy, sz]);
  for (let z = -1; z <= nz; z += 1) for (let y = -1; y <= ny; y += 1) for (let x = -1; x <= nx; x += 1) {
    if (sampleSolidWorld(world, [x, y, z]).solidFraction <= 0) continue;
    const index = (x + 1) + sx * ((y + 1) + sy * (z + 1));
    words[4 + (index >>> 5)]! |= 1 << (index & 31);
  }
  return words;
}

test("a page-driven solid mask matches the per-cell sampler through a stroke history", () => {
  let scene = createEmptyScene({ extents_m: { x: 1.3, y: .9, z: 1.1 }, finestCellSize_m: .05 });
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  const dimensions = sceneLatticeDimensions(scene);
  const mask = new SolidOccupancyMask(dimensions);
  assert.equal(mask.update(solidWorldForScene(scene))?.wordCount, mask.words.length);
  assert.deepEqual(mask.words, sampledMask(solidWorldForScene(scene), dimensions));
  const strokes: SolidWorldVoxelPatch[][] = [
    [{ operation: "fill", minimum: [3, 1, 3], maximumExclusive: [12, 7, 9] }],
    // Clears a whole page of the fill above, so that page leaves the world.
    [{ operation: "clear", minimum: [0, 0, 0], maximumExclusive: [16, 8, 16] }],
    [{ operation: "fill", minimum: [-4, -4, -4], maximumExclusive: [2, 40, 2] },
      { operation: "clear", minimum: [5, 0, 5], maximumExclusive: [6, 3, 6] }],
    [{ operation: "fill", minimum: [20, 10, 17], maximumExclusive: [40, 30, 30] }],
  ];
  for (const stroke of strokes) {
    scene = sceneWithSolidStroke(scene, stroke);
    const before = mask.words.slice();
    const dirty = mask.update(solidWorldForScene(scene));
    assert.deepEqual(mask.words, sampledMask(solidWorldForScene(scene), dimensions));
    // Everything outside the reported range is what the GPU keeps, so it must not have moved.
    const first = dirty?.firstWord ?? 0, end = first + (dirty?.wordCount ?? 0);
    before.forEach((word, index) => { if (index < first || index >= end) assert.equal(mask.words[index], word); });
    assert.ok((dirty?.wordCount ?? 0) < mask.words.length, "a stroke must not rewrite the whole mask");
  }
  assert.equal(mask.update(solidWorldForScene(scene)), undefined);
  // Regions override pages in order, as the sampler applies them.
  const world = { ...solidWorldForScene(scene), regions: [
    { operation: "fill", minimum: [-9, 2, -9], maximumExclusive: [99, 3, 99] },
    { operation: "clear", minimum: [4, 0, 4], maximumExclusive: [9, 9, 9] }] as SolidWorldVoxelPatch[] };
  mask.update(world);
  assert.deepEqual(mask.words, sampledMask(world, dimensions));
  scene = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [4, 1, 4], maximumExclusive: [10, 4, 10] }]);
  const next = { ...solidWorldForScene(scene), regions: world.regions };
  assert.ok(mask.update(next)!.wordCount < mask.words.length);
  assert.deepEqual(mask.words, sampledMask(next, dimensions));
});
