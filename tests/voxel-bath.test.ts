import assert from "node:assert/strict";
import test from "node:test";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { bathInteriorContains } from "../lib/core/voxel-bath";
import { createSolidWorld, sampleSolidWorld } from "../lib/core/solid-world";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { uniformInitialVolume } from "../lib/methods/uniform/uniform-volume-initial";
import { validateScene } from "../lib/core/model";

test("curved bath has a sealed base and sides at both authoring resolutions", () => {
  for (const h of [0.05, 0.025]) {
    const scene = createUniformTroughScene("hose-fill", h);
    assert.deepEqual(validateScene(scene), []);
    const [nx, ny, nz] = sceneLatticeDimensions(scene);
    const world = createSolidWorld(scene.solidVoxels);
    const inside = (x: number, y: number, z: number) => bathInteriorContains(
      (x + 0.5) * h - 1.6, (y + 0.5) * h, (z + 0.5) * h - 0.6);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      if (!inside(x, y, z)) continue;
      assert.equal(sampleSolidWorld(world, [x, y, z]).solidFraction, 0);
      for (const [dx, dy, dz] of [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]) {
        if (y + dy >= ny || inside(x + dx, y + dy, z + dz)) continue;
        assert.equal(sampleSolidWorld(world, [x + dx, y + dy, z + dz]).solidFraction, 1,
          `unsealed cavity at ${[x, y, z]} towards ${[dx, dy, dz]} with h=${h}`);
      }
    }
    assert.equal(bathInteriorContains(0, 0.1, 0.45), false);
    assert.equal(bathInteriorContains(0, 1.1, 0.45), true);
    assert.equal(bathInteriorContains(1.5, 1.1, 0.5), false);
  }
});

test("bath seeds remain inside the cavity and hose starts empty on the doubled grid", () => {
  const totals: number[] = [];
  for (const mode of ["dam-break", "settled-tank", "hose-fill"] as const) {
    const scene = createUniformTroughScene(mode, 0.025);
    const dims = sceneLatticeDimensions(scene);
    assert.deepEqual(dims, [128, 48, 48]);
    const { volume, initial } = uniformInitialVolume(scene, dims);
    totals.push(initial);
    for (let z = 0; z < 48; z++) for (let y = 0; y < 48; y++) for (let x = 0; x < 128; x++) {
      if (volume[x + 128 * (y + 48 * z)] > 0) {
        assert.ok(bathInteriorContains((x + 0.5) * 0.025 - 1.6,
          (y + 0.5) * 0.025, (z + 0.5) * 0.025 - 0.6));
      }
    }
  }
  assert.ok(totals[0]! > 0);
  // The generic volume seeder uses eight samples per partially filled cell.
  assert.ok(Math.abs(totals[0]! - totals[1]!) / totals[0]! < 0.005);
  assert.equal(totals[2], 0);
});
