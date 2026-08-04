import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { unpackSvoPrimitiveRecords } from "../lib/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import {
  SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES,
  buildSvoScenePrimitives,
  svoScenePrimitivesFromEnvironmentCatalog,
} from "../lib/svo-scene-primitives";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";

test("static environment bridge preserves sparse voxel owner/material identity", () => {
  const scene = cloneScene(defaultScene);
  const catalog = buildEnvironmentProxyCatalog(scene, "night-lab");
  const source = svoScenePrimitivesFromEnvironmentCatalog(scene, catalog);
  const proxies = environmentProxyPrimitives(catalog);
  const unpacked = unpackSvoPrimitiveRecords(source.packedRecords);

  assert.equal(source.environmentId, "night-lab");
  assert.equal(source.descriptors.length, proxies.length);
  assert.equal(source.metadata.length, proxies.length);
  assert.equal(unpacked.length, proxies.length);

  for (let primitiveIndex = 0; primitiveIndex < proxies.length; primitiveIndex += 1) {
    const proxy = proxies[primitiveIndex];
    const metadata = source.metadata[primitiveIndex];
    const expectedOwner = scene.rigidBodies.length + proxy.ownerIndex;
    const expectedMaterial = ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex;
    assert.equal(metadata.primitiveIndex, primitiveIndex);
    assert.equal(metadata.environmentOwnerIndex, proxy.ownerIndex);
    assert.equal(metadata.ownerId, expectedOwner);
    assert.equal(metadata.materialId, expectedMaterial);
    assert.equal(metadata.key, proxy.key);
    assert.deepEqual(metadata.material, proxy.material);
    assert.equal(source.primitiveIndexByOwnerId.get(expectedOwner), primitiveIndex);
    assert.equal(source.primitiveIndexByMaterialId.get(expectedMaterial), primitiveIndex);
    assert.equal(unpacked[primitiveIndex].primitiveId, expectedOwner);
    assert.equal(unpacked[primitiveIndex].ownerId, expectedOwner);
    assert.equal(unpacked[primitiveIndex].materialId, expectedMaterial);
  }
});

test("bridge converts each proxy shape into the aligned implicit ABI", () => {
  const scene = cloneScene(defaultScene);
  const catalog = buildEnvironmentProxyCatalog(scene, "conservatory");
  const source = svoScenePrimitivesFromEnvironmentCatalog(scene, catalog);
  const proxies = environmentProxyPrimitives(catalog);
  const byKey = new Map(source.metadata.map((metadata) => [metadata.key, source.descriptors[metadata.primitiveIndex]]));

  const boxProxy = proxies.find((proxy) => proxy.kind === "box");
  assert.ok(boxProxy?.kind === "box");
  const box = byKey.get(boxProxy.key);
  assert.ok(box?.kind === "box");
  assert.deepEqual(box.center_m, boxProxy.center_m);
  assert.deepEqual(box.halfExtents_m, boxProxy.halfSize_m);

  const cylinderProxy = proxies.find((proxy) => proxy.kind === "cylinder");
  assert.ok(cylinderProxy?.kind === "cylinder");
  const cylinder = byKey.get(cylinderProxy.key);
  assert.ok(cylinder?.kind === "cylinder");
  assert.equal(cylinder.radius_m, cylinderProxy.radius_m);
  assert.equal(cylinder.halfHeight_m, cylinderProxy.halfHeight_m);

  const ellipsoidProxy = proxies.find((proxy) => proxy.kind === "ellipsoid");
  assert.ok(ellipsoidProxy?.kind === "ellipsoid");
  const ellipsoid = byKey.get(ellipsoidProxy.key);
  assert.ok(ellipsoid?.kind === "ellipsoid");
  assert.deepEqual(ellipsoid.radii_m, ellipsoidProxy.radius_m);
});

test("stepping-stone fillets reach the renderer as one rounded SDF per plate", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const source = svoScenePrimitivesFromEnvironmentCatalog(scene, catalog);
  const proxies = environmentProxyPrimitives(catalog);
  const treads = proxies.filter(({ key }) => /stone\/path\/step-\d+\/tread$/.test(key));
  assert.equal(treads.length, 5);
  assert.ok(treads.every((proxy) => proxy.kind === "cylinder" && (proxy.edgeRadius_m ?? 0) > 0));
  for (const tread of treads) {
    const metadata = source.metadata.find(({ key }) => key === tread.key)!;
    const descriptor = source.descriptors[metadata.primitiveIndex];
    assert.equal(descriptor.kind, "rounded-cylinder", `${tread.key} lost its fillet at the render bridge`);
  }
  assert.ok(!proxies.some(({ key }) => /stone\/path\/step-\d+\/crown$/.test(key)),
    "a second crown owner would restore the sub-pixel overlap ring");
});

test("front room shell remains modelled but is identified as the interior-view skip owner", () => {
  const scene = cloneScene(defaultScene);
  const source = buildSvoScenePrimitives(scene, { environmentId: "night-lab" });
  const open = source.metadata.filter((metadata) => metadata.openShell);
  assert.equal(open.length, 1);
  assert.ok(open[0].key.endsWith("/shell/wall-front"));
  assert.equal(open[0].shell, true);
  assert.equal(source.openShellOwnerId, scene.rigidBodies.length + 8);
  assert.deepEqual(source.skipOwnerIds, [source.openShellOwnerId]);
  assert.equal(source.primitiveIndexByOwnerId.get(source.openShellOwnerId!), open[0].primitiveIndex);

  const withoutShell = buildSvoScenePrimitives(scene, { environmentId: "night-lab", includeShell: false });
  assert.equal(withoutShell.descriptors.length, 105);
  assert.equal(withoutShell.openShellOwnerId, undefined);
  assert.deepEqual(withoutShell.skipOwnerIds, []);
  assert.ok(withoutShell.metadata.every((metadata) => !metadata.shell));
  assert.equal(withoutShell.metadata[0].environmentOwnerIndex, 9, "prop owner IDs retain the full-catalog convention");
});

test("night-lab back wall is four stable analytic boxes around the thin-glass opening", () => {
  const scene = cloneScene(defaultScene);
  const source = buildSvoScenePrimitives(scene, { environmentId: "night-lab" });
  const backWall = source.metadata.filter(({ key }) => key.startsWith("night-lab/shell/wall-back"));
  assert.deepEqual(backWall.map(({ key }) => key), [
    "night-lab/shell/wall-back-left",
    "night-lab/shell/wall-back-right",
    "night-lab/shell/wall-back-bottom",
    "night-lab/shell/wall-back-top",
  ]);
  assert.deepEqual(backWall.map(({ environmentOwnerIndex }) => environmentOwnerIndex), [4, 5, 6, 7]);
  assert.equal(new Set(backWall.map(({ materialId }) => materialId)).size, 4);
  assert.ok(backWall.every(({ sourceKind, tags }) => sourceKind === "box" && tags.includes("window-cutout")));
});

test("studio-room and garden catalogs report presentation support explicitly", () => {
  const scene = cloneScene(defaultScene);
  const studio = buildSvoScenePrimitives(scene, { environmentId: "default" });
  assert.equal(studio.descriptors.length, 7, "the six-face white room has only its overhead light");
  assert.equal(studio.metadata.filter(({ shell }) => shell).length, 6);
  assert.deepEqual(studio.metadata.filter(({ shell }) => !shell).map(({ key }) => key), ["default/light/softbox"]);
  assert.ok(studio.metadata.find(({ ownerId }) => ownerId === studio.openShellOwnerId)?.key.endsWith("/shell/wall-front"));
  assert.equal(studio.requiresRasterTerrainFallback, false);
  assert.deepEqual(studio.unsupportedSources, []);

  scene.terrain = { baseHeight_m: 0.2, features: [] };
  const garden = buildSvoScenePrimitives(scene, { environmentId: "garden" });
  assert.equal(garden.descriptors.length, 122, "garden props remain directly representable");
  assert.ok(garden.descriptors.every((descriptor) => descriptor.kind !== "terrain-heightfield"));
  assert.equal(garden.requiresRasterTerrainFallback, false);
  assert.deepEqual(garden.unsupportedSources, []);
  assert.deepEqual(garden.analyticTerrain, {
    kind: "terrain-heightfield",
    materialId: VOXEL_MATERIAL_IDS.terrain,
    normalEpsilon_m: 0.02,
  });
});

test("scene convenience API follows the selected environment and enforces a hard record bound", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "conservatory";
  const selected = buildSvoScenePrimitives(scene);
  assert.equal(selected.environmentId, "conservatory");
  // 16 384 since W0 of the raster-visibility program raised it from 4 096 for
  // the `hero-garden-hose-x10` acceptance scene: 5 039 records did not draw at
  // all under the old ceiling. Pinned to the candidate arena's own leaf count
  // rather than to a literal, because the two are deliberately the same number
  // — a scene that clears this bound must not then be refused by the BVH.
  assert.equal(SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES, SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES);
  assert.equal(SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES, 16_384);
  assert.throws(
    () => buildSvoScenePrimitives(scene, { maximumPrimitives: selected.descriptors.length - 1 }),
    /exceeding the .* record limit/,
  );
  assert.throws(() => buildSvoScenePrimitives(scene, { maximumPrimitives: 0 }), /positive safe integer/);
});

test("static primitive publication reuses packed records and invalidates on authored geometry", () => {
  const scene = cloneScene(defaultScene);
  const first = buildSvoScenePrimitives(scene, { environmentId: "night-lab" });
  const repeated = buildSvoScenePrimitives(cloneScene(scene), { environmentId: "night-lab" });
  assert.strictEqual(repeated, first);
  assert.strictEqual(repeated.packedRecords, first.packedRecords);
  assert.equal(repeated.cacheKey, first.cacheKey);

  scene.container.width_m += 0.1;
  const resized = buildSvoScenePrimitives(scene, { environmentId: "night-lab" });
  assert.notEqual(resized.cacheKey, first.cacheKey);
  assert.notStrictEqual(resized.packedRecords, first.packedRecords);
});

test("subcell props retain exact analytic shapes plus conservative audit bounds", () => {
  const scene = cloneScene(defaultScene);
  for (const [environmentId, keys] of [
    ["conservatory", ["conservatory/pendant-1/cord"]],
    ["night-lab", ["night-lab/desk/lower-shelf", "night-lab/counter/keyboard"]],
    ["bathhouse", ["bathhouse/lantern-left/cord", "bathhouse/lantern-right/cord"]],
    ["research-station", ["research-station/observation-port/backing"]],
  ] as const) {
    const publication = buildSvoScenePrimitives(scene, { environmentId });
    for (const key of keys) {
      const metadata = publication.metadata.find((entry) => entry.key === key);
      assert.ok(metadata, key);
      assert.equal(metadata.coverageBounds.policy, "conservative-subcell", key);
      assert.ok(metadata.coverageBounds.subcellAxes.length > 0, key);
      for (const axis of metadata.coverageBounds.subcellAxes) {
        const conservativeWidth = metadata.coverageBounds.conservative_m.max[axis]
          - metadata.coverageBounds.conservative_m.min[axis];
        assert.ok(conservativeWidth >= scene.voxelDomain.finestCellSize_m, `${key}/${axis}`);
      }
    }
  }
});
