import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import { unpackSvoPrimitiveRecords } from "../lib/svo-primitive-abi";
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
  assert.equal(withoutShell.descriptors.length, 32);
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

test("floor-only and garden catalogs report presentation support explicitly", () => {
  const scene = cloneScene(defaultScene);
  const floor = buildSvoScenePrimitives(scene, { environmentId: "default" });
  assert.equal(floor.descriptors.length, 1);
  assert.equal(floor.metadata[0].shell, true);
  assert.equal(floor.openShellOwnerId, undefined);
  assert.equal(floor.requiresRasterTerrainFallback, false);
  assert.deepEqual(floor.unsupportedSources, []);

  scene.terrain = { baseHeight_m: 0.2, features: [] };
  const garden = buildSvoScenePrimitives(scene, { environmentId: "garden" });
  assert.equal(garden.descriptors.length, 22, "garden props remain directly representable");
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
  assert.equal(SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES, 4_096);
  assert.throws(
    () => buildSvoScenePrimitives(scene, { maximumPrimitives: selected.descriptors.length - 1 }),
    /exceeding the .* record limit/,
  );
  assert.throws(() => buildSvoScenePrimitives(scene, { maximumPrimitives: 0 }), /positive safe integer/);
});
