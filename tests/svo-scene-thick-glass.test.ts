import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoEnvironmentCoverage } from "../lib/svo-scene-coverage";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";

test("authored curved glass catalog is bounded, deterministic, and renderer-bound", () => {
  const scene = cloneScene(defaultScene);
  const expected = new Map([
    ["conservatory", 3],
    ["night-lab", 1],
    ["research-station", 1],
  ]);
  for (const environmentId of environmentIds) {
    const first = buildSvoSceneThickGlass(scene, { environmentId, revision: 12 });
    const repeated = buildSvoSceneThickGlass(cloneScene(scene), { environmentId, revision: 12 });
    assert.equal(first.descriptors.length, expected.get(environmentId) ?? 0, environmentId);
    assert.strictEqual(repeated, first);
    assert.strictEqual(repeated.packedRecords, first.packedRecords);
    assert.ok(first.metadata.every(({ productionBinding }) => productionBinding === "renderer-uniform-binder"));
  }
});

test("lamp globes preserve the source analytic owner/material identity", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of ["conservatory", "night-lab"] as const) {
    const proxies = new Map(environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId), true).map((proxy) => [proxy.key, proxy]));
    const publication = buildSvoSceneThickGlass(scene, { environmentId, revision: 7 });
    for (const metadata of publication.metadata) {
      const proxy = proxies.get(metadata.sourceKey);
      assert.ok(proxy, metadata.sourceKey);
      assert.equal(metadata.ownerId, scene.rigidBodies.length + proxy!.ownerIndex);
      assert.equal(metadata.materialId, ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy!.ownerIndex);
      assert.equal(metadata.shape, "sphere");
    }
    const coverage = buildSvoEnvironmentCoverage(scene, environmentId);
    for (const metadata of publication.metadata) {
      const entry = coverage.entries.find(({ key }) => key === metadata.sourceKey);
      assert.equal(entry?.status, "degraded", "opaque production fallback remains explicit");
      assert.equal(entry?.plannedThickGlassId, metadata.glassId);
      assert.equal(entry?.plannedThickGlassContract, "analytic-thick-glass-unbound");
    }
  }
});

test("station observation lens has stable elliptical identity and explicit replacement seams", () => {
  const scene = cloneScene(defaultScene);
  const publication = buildSvoSceneThickGlass(scene, { environmentId: "research-station", revision: 5 });
  assert.equal(publication.metadata.length, 1);
  const lens = publication.metadata[0];
  assert.equal(lens.key, "research-station/observation-port/thick-lens");
  assert.equal(lens.shape, "ellipsoid");
  assert.equal(lens.replacesThinPaneKey, "research-station/observation-port/glazing");
  assert.equal(lens.replacesUnsupportedKey, "research-station/shell/procedural-portholes");
  assert.ok(lens.bounds_m.maximum[0] > lens.bounds_m.minimum[0]);

  const coverage = buildSvoEnvironmentCoverage(scene, "research-station");
  const curvedPort = coverage.entries.find(({ key }) => key === "research-station/shell/procedural-portholes");
  assert.equal(curvedPort?.status, "unsupported", "production remains fail-closed until its binder lands");
  assert.equal(curvedPort?.plannedThickGlassId, lens.glassId);
  assert.equal(curvedPort?.plannedThickGlassContract, "analytic-thick-glass-unbound");
  assert.equal(coverage.summary.thickGlassVolumes, 1);
});

test("curved glass revisions and capacity invalidate deterministically", () => {
  const scene = cloneScene(defaultScene);
  const first = buildSvoSceneThickGlass(scene, { environmentId: "conservatory", revision: 2 });
  const advanced = buildSvoSceneThickGlass(scene, { environmentId: "conservatory", revision: 3 });
  assert.notEqual(advanced.cacheKey, first.cacheKey);
  assert.notStrictEqual(advanced.packedRecords, first.packedRecords);
  assert.throws(() => buildSvoSceneThickGlass(scene, { environmentId: "conservatory", maximumVolumes: 2 }), /record limit/);
});
