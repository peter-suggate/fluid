import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { buildSvoSceneGlass } from "../lib/svo/svo-scene-glass";
import { svoDryRasterGlassRecordRange } from "../lib/svo/webgpu-svo-dry-scene";

test("closed tank floor and ceiling form one compositor-owned pane range", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "stage";
  scene.container.top = "closed";
  const glass = buildSvoSceneGlass(scene, { environmentId: "stage" });
  const container = glass.metadata.filter(({ role }) =>
    role === "container-pane" || role === "container-top");

  assert.deepEqual(container.map(({ side }) => side), ["floor", "ceiling"]);
  assert.equal(container[1]!.paneId, container[0]!.paneId + 1,
    "the dry renderer can only exclude compositor ownership as one contiguous pane-ID range");

  const range = svoDryRasterGlassRecordRange(
    glass.packedRecords, container[0]!.paneId, container.length,
  );
  assert.equal(range.firstRecord, container.length);
  assert.equal(range.recordCount, glass.descriptors.length - container.length,
    "neither the floor nor ceiling may enter dry primary depth ahead of water");
});

test("an open tank publishes only its compositor-owned floor", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "stage";
  scene.container.top = "open";
  const glass = buildSvoSceneGlass(scene, { environmentId: "stage" });
  const container = glass.metadata.filter(({ role }) =>
    role === "container-pane" || role === "container-top");

  assert.deepEqual(container.map(({ side }) => side), ["floor"]);
  const range = svoDryRasterGlassRecordRange(
    glass.packedRecords, container[0]!.paneId, container.length,
  );
  assert.equal(range.firstRecord, 1);
  assert.equal(range.recordCount, glass.descriptors.length - 1);
});
