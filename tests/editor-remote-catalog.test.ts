import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneryEntity } from "../lib/core/editor-scenery";
import { adoptEnvironmentProxyCatalog, buildEnvironmentProxyCatalog, cachedEnvironmentProxyCatalog,
  environmentCatalogPending, useRemoteEnvironmentCatalogs, markEnvironmentCatalogRemote, reuseEnvironmentProxyCatalog } from "../lib/core/voxel-environments";

test("remote scenery picks never expand cold geometry and exact worker catalog survives document clones", () => {
  const workerScene = cloneScene(defaultScene);
  const catalog = buildEnvironmentProxyCatalog(workerScene, workerScene.environment ?? "default");
  const document = structuredClone(workerScene);
  const ray = { origin: { x: 0, y: 2, z: 2 }, direction: { x: 0, y: -1, z: -1 } };
  markEnvironmentCatalogRemote(document);
  assert.equal(environmentCatalogPending(document), true);
  assert.equal(sceneryEntity.pick?.({ scene: document, bodies: [] }, ray), undefined);
  assert.equal(cachedEnvironmentProxyCatalog(document), undefined, "input must not create a procedural catalog");
  const transferred = structuredClone(catalog);
  adoptEnvironmentProxyCatalog(document, transferred);
  assert.equal(environmentCatalogPending(document), false);
  assert.deepEqual(sceneryEntity.pick?.({ scene: document, bodies: [] }, ray),
    sceneryEntity.pick?.({ scene: workerScene, bodies: [] }, ray));
  const next = structuredClone(document);
  next.solidVoxels.push({ operation: "fill", minimum: [1, 1, 1], maximumExclusive: [2, 2, 2] });
  reuseEnvironmentProxyCatalog(document, next);
  assert.equal(cachedEnvironmentProxyCatalog(next), transferred, "voxel-only changes preserve exact scenery geometry");
  const changed = structuredClone(document); changed.container.width_m *= 2;
  reuseEnvironmentProxyCatalog(document, changed);
  assert.equal(cachedEnvironmentProxyCatalog(changed), undefined, "changed geometry must not inherit stale pick bounds");
});


test("newly reopened documents cannot expand scenery before the first worker draw", () => {
  useRemoteEnvironmentCatalogs();
  const reopened = structuredClone(defaultScene);
  assert.equal(environmentCatalogPending(reopened), true);
  assert.equal(sceneryEntity.pick?.({ scene: reopened, bodies: [] },
    { origin: { x: 0, y: 2, z: 2 }, direction: { x: 0, y: -1, z: -1 } }), undefined);
  assert.equal(cachedEnvironmentProxyCatalog(reopened), undefined);
});
