import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoSceneGlass, SVO_SCENE_GLASS_MAXIMUM_PANES } from "../lib/svo-scene-glass";
import { SVO_THIN_GLASS_RECORD_STRIDE_BYTES, unpackSvoThinGlassPanes } from "../lib/svo-thin-glass";
import { SPARSE_BRICK_NO_OWNER } from "../lib/sparse-brick-octree";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import { intersectSvoRayAabb } from "../lib/webgpu-svo-traversal";

test("every shipped environment has explicit container and authored glazing coverage", () => {
  const scene = cloneScene(defaultScene);
  scene.container.top = "open";
  const expectedEnvironmentPanes = new Map([
    ["conservatory", 6], ["night-lab", 1],
  ]);
  for (const environmentId of environmentIds) {
    const glass = buildSvoSceneGlass(scene, { environmentId });
    const expectedContainer = environmentId === "garden" ? 0 : 5;
    assert.equal(glass.containerPaneIndices.length, expectedContainer, `${environmentId} container coverage`);
    assert.equal(glass.environmentPaneIndices.length, expectedEnvironmentPanes.get(environmentId) ?? 0, `${environmentId} authored glazing coverage`);
    assert.equal(glass.descriptors.length, expectedContainer + (expectedEnvironmentPanes.get(environmentId) ?? 0));
    assert.equal(glass.packedRecords.byteLength, glass.descriptors.length * SVO_THIN_GLASS_RECORD_STRIDE_BYTES);
    assert.equal(glass.containerPolicy, environmentId === "garden" ? "absent-open-environment" : "thin-glass-vessel");
  }
});

test("closed container top is a stable sixth pane and garden remains vessel-free", () => {
  const scene = cloneScene(defaultScene);
  scene.container.top = "closed";
  const closed = buildSvoSceneGlass(scene, { environmentId: "default" });
  assert.equal(closed.containerPaneIndices.length, 6);
  assert.equal(closed.containerTopPaneIndex, 5);
  assert.equal(closed.metadata[5].role, "container-top");
  assert.equal(closed.metadata[5].side, "ceiling");
  assert.equal(closed.descriptors[5].paneId, 0x1005);
  const garden = buildSvoSceneGlass(scene, { environmentId: "garden" });
  assert.equal(garden.descriptors.length, 0);
  assert.equal(garden.containerTopPaneIndex, undefined);
});

test("container glass preserves the existing stable material/no-owner identity", () => {
  const scene = cloneScene(defaultScene);
  const glass = buildSvoSceneGlass(scene, { environmentId: "default" });
  assert.ok(glass.descriptors.every(({ materialId }) => materialId === VOXEL_MATERIAL_IDS.containerGlass));
  assert.ok(glass.descriptors.every(({ ownerId }) => ownerId === SPARSE_BRICK_NO_OWNER));
  const unpacked = unpackSvoThinGlassPanes(glass.packedRecords);
  assert.deepEqual(unpacked.map(({ paneId }) => paneId), [0x1000, 0x1001, 0x1002, 0x1003, 0x1004]);
  assert.ok(unpacked.every(({ materialId, ownerId }) => materialId === 1 && ownerId === 0xffff));
});

test("conservatory panes follow frame bays and lab glazing uses an analytic wall opening", () => {
  const scene = cloneScene(defaultScene);
  const conservatory = buildSvoSceneGlass(scene, { environmentId: "conservatory" });
  const keys = conservatory.environmentPaneIndices.map((index) => conservatory.metadata[index].key);
  assert.deepEqual(keys, [
    "conservatory/glazing/pane-left-low",
    "conservatory/glazing/pane-left-middle",
    "conservatory/glazing/pane-left-high",
    "conservatory/glazing/pane-right-low",
    "conservatory/glazing/pane-right-middle",
    "conservatory/glazing/pane-right-high",
  ]);
  const lab = buildSvoSceneGlass(scene, { environmentId: "night-lab" });
  const window = lab.metadata[lab.environmentPaneIndices[0]];
  assert.equal(window.key, "night-lab/window/city-glazing");
  assert.equal(window.opaqueCutoutKey, undefined);
});

test("catalog glass-like entries which are not finite dielectric panes remain explicit", () => {
  const scene = cloneScene(defaultScene);
  const expected = {
    conservatory: ["curved-emissive-volume", "curved-emissive-volume", "curved-emissive-volume"],
    "night-lab": ["curved-emissive-volume", "emissive-display"],
    "research-station": ["emissive-display", "emissive-display", "procedural-circular-glazing"],
  } as const;
  for (const environmentId of environmentIds) {
    const reasons = buildSvoSceneGlass(scene, { environmentId }).unsupportedEntries.map(({ reason }) => reason);
    assert.deepEqual(reasons, expected[environmentId as keyof typeof expected] ?? [], environmentId);
  }
});

test("conservative pane bounds are candidates for normal and grazing rays", () => {
  const scene = cloneScene(defaultScene);
  const glass = buildSvoSceneGlass(scene, { environmentId: "default", cellSize_m: [0.02, 0.04, 0.08] });
  const front = glass.metadata.find(({ side }) => side === "front");
  assert.ok(front);
  const normal = intersectSvoRayAabb({ origin: [0, scene.container.height_m / 2, -scene.container.depth_m], direction: [0, 0, 1] }, {
    minimum: front!.bounds.conservative_m.minimum,
    maximum: front!.bounds.conservative_m.maximum,
  });
  const grazing = intersectSvoRayAabb({
    origin: [-scene.container.width_m, scene.container.height_m / 2, -scene.container.depth_m / 2 - 0.02],
    direction: [1, 0, 0.01],
  }, { minimum: front!.bounds.conservative_m.minimum, maximum: front!.bounds.conservative_m.maximum });
  assert.ok(normal);
  assert.ok(grazing);
  assert.equal(front?.bounds.analyticFinalHitRequired, true);
  assert.equal(front?.bounds.maximumRefinementIterations, 6);
});

test("static revision and upload cache key are deterministic and content-sensitive", () => {
  const scene = cloneScene(defaultScene);
  const first = buildSvoSceneGlass(scene, { environmentId: "conservatory", cellSize_m: 0.02 });
  const second = buildSvoSceneGlass(cloneScene(scene), { environmentId: "conservatory", cellSize_m: [0.02, 0.02, 0.02] });
  assert.equal(first.staticRevision, second.staticRevision);
  assert.equal(first.cacheKey, second.cacheKey);
  scene.container.width_m += 0.1;
  const resized = buildSvoSceneGlass(scene, { environmentId: "conservatory", cellSize_m: 0.02 });
  assert.notEqual(resized.staticRevision, first.staticRevision);
  assert.notEqual(resized.cacheKey, first.cacheKey);
  const differentCell = buildSvoSceneGlass(scene, { environmentId: "conservatory", cellSize_m: 0.01 });
  assert.notEqual(differentCell.staticRevision, resized.staticRevision);
});

test("scene glass authoring enforces its fixed record bound", () => {
  const scene = cloneScene(defaultScene);
  assert.equal(SVO_SCENE_GLASS_MAXIMUM_PANES, 256);
  assert.throws(() => buildSvoSceneGlass(scene, { environmentId: "conservatory", maximumPanes: 10 }), /record limit/);
  assert.throws(() => buildSvoSceneGlass(scene, { maximumPanes: 0 }), /integer from 1/);
});
