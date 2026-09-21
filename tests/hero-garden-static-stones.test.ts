import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";

test("garden stepping stones are five voxel-solid components, with no rigid bodies", () => {
  const scene = sceneDocument(getSceneDefinition("hero-garden-hose"));
  assert.equal(scene.rigidBodies.length, 0);
  const world = solidWorldForScene(scene);
  const cells = new Set<string>();
  for (const patch of scene.solidVoxels) {
    assert.equal(patch.operation, "fill");
    for (let z=patch.minimum[2]; z<patch.maximumExclusive[2]; z++)
      for (let y=patch.minimum[1]; y<patch.maximumExclusive[1]; y++)
        for (let x=patch.minimum[0]; x<patch.maximumExclusive[0]; x++) {
          cells.add(`${x},${y},${z}`);
          assert.ok(sampleSolidWorld(world, [x,y,z]).solidFraction > 0);
        }
  }
  assert.ok(cells.size > 0);
  // Five separate stones must survive voxelization at the production lattice.
  let components = 0;
  while (cells.size) {
    const queue = [cells.values().next().value!]; cells.delete(queue[0]!); components++;
    while (queue.length) {
      const p = queue.pop()!.split(",").map(Number);
      for (let axis=0; axis<3; axis++) for (const side of [-1,1]) {
        const q=[...p]; q[axis]!+=side; const key=q.join(",");
        if(cells.delete(key)) queue.push(key);
      }
    }
  }
  assert.equal(components, 5);
});
