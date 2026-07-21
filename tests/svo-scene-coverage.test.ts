import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { scenePresets } from "../lib/scenes";
import {
  buildSvoEnvironmentCoverage,
  buildSvoShippedSceneCoverage,
  canonicalSvoSceneCoverage,
  SVO_SCENE_COVERAGE_VERSION,
} from "../lib/svo-scene-coverage";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import { materialIdForRigidShape } from "../lib/voxel-scene";

test("coverage report deterministically includes every shipped environment and preset", () => {
  const first = buildSvoShippedSceneCoverage();
  const second = buildSvoShippedSceneCoverage();
  assert.equal(first.version, SVO_SCENE_COVERAGE_VERSION);
  assert.deepEqual(first.environments.map(({ environmentId }) => environmentId), environmentIds);
  assert.deepEqual(first.presets.map(({ presetId }) => presetId), scenePresets.map(({ id }) => id));
  assert.equal(canonicalSvoSceneCoverage(first), canonicalSvoSceneCoverage(second));
  for (const environment of first.environments) {
    assert.equal(new Set(environment.entries.map(({ key }) => key)).size, environment.entries.length,
      `${environment.environmentId} coverage keys remain one-to-one`);
    assert.equal(
      environment.summary.complete + environment.summary.degraded + environment.summary.unsupported,
      environment.entries.length,
    );
  }
});

test("every authored analytic primitive has visible, collision, lighting, owner, and material coverage", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
    const proxies = environmentProxyPrimitives(catalog, true);
    const report = buildSvoEnvironmentCoverage(scene, environmentId);
    const entries = new Map(report.entries.map((entry) => [entry.key, entry]));
    for (const proxy of proxies) {
      const entry = entries.get(proxy.key);
      assert.ok(entry, `${proxy.key} is audited`);
      assert.equal(entry!.materialId, ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex);
      assert.equal(entry!.ownerId, scene.rigidBodies.length + proxy.ownerIndex);
      assert.equal(entry!.collisionOwnership, "solver-environment-proxy");
      assert.notEqual(entry!.visibleOwnership, "raster-only-procedural");
      assert.ok(entry!.lightingOwnership);
    }
  }
});

test("finite panes and terrain are complete while unsupported optical/procedural details fail closed", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const report = buildSvoEnvironmentCoverage(scene, environmentId);
    const entries = new Map(report.entries.map((entry) => [entry.key, entry]));
    for (const pane of buildSvoSceneGlass(scene, { environmentId }).metadata) {
      const entry = entries.get(pane.key);
      assert.ok(entry, `${pane.key} is audited`);
      assert.equal(entry!.status, "complete");
      assert.ok(entry!.visibleOwnership === "thin-glass" || entry!.visibleOwnership === "thick-glass");
    }
  }

  const lab = buildSvoEnvironmentCoverage(scene, "night-lab");
  const cityWindow = lab.entries.find(({ key }) => key === "night-lab/window/city-glazing");
  assert.equal(cityWindow?.status, "complete");
  assert.ok(lab.entries.filter(({ key }) => key.includes("shell/wall-back-")).every(({ status }) => status === "complete"));
  assert.equal(lab.entries.find(({ key }) => key === "night-lab/counter/monitor-screen")?.reason, "emissive-display");

  const station = buildSvoEnvironmentCoverage(scene, "research-station");
  const portholes = station.entries.find(({ key }) => key.endsWith("procedural-portholes"));
  assert.equal(portholes?.status, "complete");
  assert.equal(portholes?.visibleOwnership, "thick-glass");
  assert.equal(portholes?.reason, "procedural-circular-glazing");

  const garden = buildSvoEnvironmentCoverage(scene, "garden");
  const terrain = garden.entries.find(({ category }) => category === "terrain");
  assert.equal(terrain?.status, "complete");
  assert.equal(terrain?.visibleOwnership, "analytic-terrain");
  assert.equal(terrain?.collisionOwnership, "solver-terrain-heightfield");
});

test("preset rigid bodies retain solver ownership and stable direct material IDs", () => {
  const report = buildSvoShippedSceneCoverage();
  for (const preset of report.presets) {
    const scene = scenePresets.find(({ id }) => id === preset.presetId)!.create();
    assert.equal(preset.rigidBodies.length, scene.rigidBodies.length);
    preset.rigidBodies.forEach((entry, ownerId) => {
      assert.equal(entry.status, "complete");
      assert.equal(entry.visibleOwnership, "analytic-rigid-body");
      assert.equal(entry.collisionOwnership, "solver-rigid-body");
      assert.equal(entry.ownerId, ownerId);
      assert.equal(entry.materialId, materialIdForRigidShape(scene.rigidBodies[ownerId].shape));
    });
  }
});

test("default-camera audit prioritizes subcell and thin authored props", () => {
  const scene = cloneScene(defaultScene);
  const lab = buildSvoEnvironmentCoverage(scene, "night-lab");
  for (const key of [
    "night-lab/counter/monitor-screen",
    "night-lab/counter/keyboard",
    "night-lab/desk-lamp/stem",
    "night-lab/desk/lower-shelf",
  ]) assert.equal(lab.entries.find((entry) => entry.key === key)?.defaultCameraPriority, true, key);
  const bathhouse = buildSvoEnvironmentCoverage(scene, "bathhouse");
  for (const key of ["bathhouse/lantern-left/cord", "bathhouse/lantern-right/cord"]) {
    assert.equal(bathhouse.entries.find((entry) => entry.key === key)?.defaultCameraPriority, true, key);
  }
});

test("coverage publications cache by content and expose subcell collision proxy ownership", () => {
  const scene = cloneScene(defaultScene);
  const first = buildSvoEnvironmentCoverage(scene, "night-lab");
  const repeated = buildSvoEnvironmentCoverage(cloneScene(scene), "night-lab");
  assert.strictEqual(repeated, first);
  assert.equal(repeated.cacheKey, first.cacheKey);

  const keyboard = first.entries.find(({ key }) => key === "night-lab/counter/keyboard");
  assert.equal(keyboard?.visibleOwnership, "analytic-primitive");
  assert.equal(keyboard?.collisionOwnership, "solver-environment-proxy");
  assert.ok(keyboard?.materialId !== undefined && keyboard.ownerId !== undefined);
  assert.equal(keyboard?.boundsPolicy, "conservative-subcell");
  assert.ok(keyboard?.subcellAxes?.includes("y"));
  assert.ok(keyboard?.collisionProxyBounds_m);

  scene.container.width_m += 0.1;
  const resized = buildSvoEnvironmentCoverage(scene, "night-lab");
  assert.notEqual(resized.cacheKey, first.cacheKey);

  const shippedFirst = buildSvoShippedSceneCoverage();
  const shippedSecond = buildSvoShippedSceneCoverage();
  assert.strictEqual(shippedSecond, shippedFirst);
  assert.equal(shippedSecond.cacheKey, shippedFirst.cacheKey);
});
