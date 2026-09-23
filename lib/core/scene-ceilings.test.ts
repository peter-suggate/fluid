import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "./scenes";
import { sceneDocument } from "./scene-definition";
import { sceneLatticeDimensions } from "./scene-lattice";

test("each pool-impact resolution publishes a physical ceiling on its own lattice", () => {
  for (const id of ["coarse-first-pool-impact", "coarse-first-pool-impact-half",
    "coarse-first-pool-impact-quarter"]) {
    const scene = sceneDocument(getSceneDefinition(id));
    const [nx, ny, nz] = sceneLatticeDimensions(scene);
    assert.ok(scene.solidVoxels.some(p => p.operation === "fill"
      && p.minimum[0] <= 0 && p.minimum[1] === ny && p.minimum[2] <= 0
      && p.maximumExclusive[0] >= nx && p.maximumExclusive[1] > ny
      && p.maximumExclusive[2] >= nz), `${id}: missing solid ceiling`);
  }
});

test("settled-tank drop bodies start below its closed lid and above the water", () => {
  const scene = sceneDocument(getSceneDefinition("water-box-tank-fill"));
  assert.equal(scene.rigidBodies.length, 2);
  const waterline = scene.container.height_m * scene.container.fillFraction;
  for (const body of scene.rigidBodies) {
    const extent = Math.max(body.dimensions_m.x, body.dimensions_m.y, body.dimensions_m.z);
    assert.ok(body.position_m.y + extent < scene.container.height_m, body.id);
    assert.ok(body.position_m.y - extent > waterline, body.id);
  }
});
