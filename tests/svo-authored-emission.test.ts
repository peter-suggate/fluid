import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoEnvironmentCoverage } from "../lib/svo-scene-coverage";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { unpackSvoMaterialRecord } from "../lib/svo-material-abi";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import {
  buildOctreeSvoPbrMaterialPublication,
  ENVIRONMENT_VOXEL_MATERIAL_BASE,
} from "../lib/webgpu-octree-sparse-bricks";

const close = (actual: number, expected: number, label: string) => {
  assert.ok(Math.abs(actual - expected) <= 1e-6, `${label}: ${actual} != ${expected}`);
};

test("every authored emissive owner has a matching physical light or an explicit surface-only exception", () => {
  for (const environmentId of environmentIds) {
    const scene = cloneScene(defaultScene);
    scene.environment = environmentId;
    const proxies = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId), true);
    const lights = buildSvoSceneLights(scene, { environmentId, revision: 17 });
    const repeated = buildSvoSceneLights(scene, { environmentId, revision: 17 });
    assert.deepEqual(repeated, lights, `${environmentId} light IDs and selection are deterministic`);
    assert.equal(lights.omittedFixtureKeys.length, 0, `${environmentId} authored fixtures fit the production table`);
    const lightByKey = new Map(lights.records.map((light) => [light.sourceKey, light]));
    const materials = buildOctreeSvoPbrMaterialPublication(17, proxies);
    const coverage = new Map(buildSvoEnvironmentCoverage(scene, environmentId).entries.map((entry) => [entry.key, entry]));

    for (const proxy of proxies.filter(({ material }) => material.emission > 0)) {
      const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex;
      assert.ok(materialId < materials.count, `${proxy.key} material remains in range`);
      const material = unpackSvoMaterialRecord(materials.packedRecords, materialId);
      assert.equal(material.materialId, materialId);
      assert.equal(material.revision, 17);
      assert.ok([...material.baseColorLinear, ...material.emissiveLinear].every(Number.isFinite), `${proxy.key} material is finite`);
      assert.ok(material.baseColorLinear.some((channel) => channel > 0), `${proxy.key} cannot publish a black closure`);
      proxy.material.colorLinear.forEach((channel, index) => close(
        material.emissiveLinear[index],
        channel * proxy.material.emission,
        `${proxy.key} emissive channel ${index}`,
      ));

      const light = lightByKey.get(proxy.key);
      if (!light) {
        assert.ok(proxy.tags.includes("emissive-surface-only"), `${proxy.key} needs an explicit no-light exception`);
        assert.equal(coverage.get(proxy.key)?.lightingOwnership, "emissive-surface-only");
        assert.equal(coverage.get(proxy.key)?.reason, "documented-low-power-emissive-surface");
        continue;
      }
      assert.ok(proxy.tags.includes("light"), `${proxy.key} physical light is explicitly authored`);
      assert.equal(light.lightId, proxy.ownerIndex + 2);
      assert.equal(light.ownerId, scene.rigidBodies.length + proxy.ownerIndex);
      assert.equal(light.revision, 17);
      assert.deepEqual(light.position_m, [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z]);
      assert.deepEqual(light.colorLinear, proxy.material.colorLinear);
      assert.equal(light.intensity, proxy.material.emission);
      assert.ok(light.range_m > 0 && Number.isFinite(light.range_m));
      assert.equal(coverage.get(proxy.key)?.lightingOwnership, "svo-area-light");
    }
  }
});

test("night-lab and station screens emit into the room while ceiling fixtures emit downward", () => {
  const expected = new Map([
    ["night-lab/counter/monitor-screen", [0, 0, 1]],
    ["night-lab/fixtures/troffer-left-1", [0, -1, 0]],
    ["night-lab/fixtures/troffer-right-2", [0, -1, 0]],
    ["research-station/console-left/monitor", [0, 0, 1]],
    ["research-station/console-right/monitor", [0, 0, 1]],
  ] as const);
  for (const environmentId of ["night-lab", "research-station"] as const) {
    const scene = cloneScene(defaultScene);
    scene.environment = environmentId;
    const lights = new Map(buildSvoSceneLights(scene, { environmentId }).records.map((light) => [light.sourceKey, light]));
    for (const [key, direction] of expected) if (key.startsWith(environmentId)) {
      assert.deepEqual(lights.get(key)?.direction, direction, key);
    }
  }
});

test("research-station observation port has stable analytic frame, backing, and finite thin-pane coverage", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "research-station";
  const first = buildSvoScenePrimitives(scene, { environmentId: "research-station" });
  const second = buildSvoScenePrimitives(scene, { environmentId: "research-station" });
  assert.deepEqual(second.metadata, first.metadata);
  for (const suffix of ["backing", "frame-left", "frame-right", "frame-bottom", "frame-top"]) {
    const metadata = first.metadata.find(({ key }) => key === `research-station/observation-port/${suffix}`);
    assert.ok(metadata, suffix);
    assert.ok(Number.isSafeInteger(metadata.ownerId) && Number.isSafeInteger(metadata.materialId));
  }
  const glass = buildSvoSceneGlass(scene, { environmentId: "research-station" });
  const pane = glass.metadata.find(({ key }) => key === "research-station/observation-port/glazing");
  assert.ok(pane);
  assert.equal(pane.role, "environment-glazing");
  assert.ok(Object.values(pane.bounds.exact_m).flatMap((value) => Object.values(value)).every(Number.isFinite));
  assert.ok(glass.unsupportedEntries.some(({ reason }) => reason === "procedural-circular-glazing"),
    "curved circular transmission remains an explicit unsupported raster-shell detail");
  const coverage = buildSvoEnvironmentCoverage(scene, "research-station");
  assert.equal(coverage.entries.find(({ key }) => key === pane.key)?.status, "complete");
});
