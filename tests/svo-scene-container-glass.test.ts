import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { buildSvoSceneGlass } from "../lib/svo/svo-scene-glass";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/webgpu-svo-dry-scene";
import { liveSvoDerivedBuildWGSLFor } from "../lib/svo/webgpu-svo-live-derived-builder";

test("the pane arena contains authored environment glazing only", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "conservatory";
  scene.scenery = undefined;
  const glass = buildSvoSceneGlass(scene, { environmentId: "conservatory" });

  assert.ok(glass.descriptors.length > 0);
  assert.equal(glass.environmentPaneIndices.length, glass.descriptors.length);
  assert.ok(glass.metadata.every(({ role }) => role === "environment-glazing"));

  const before = glass.cacheKey;
  scene.solidVoxels.push({ operation: "clear", minimum: [-1, 0, 0],
    maximumExclusive: [0, 1, 1] });
  assert.equal(buildSvoSceneGlass(scene, { environmentId: "conservatory" }).cacheKey, before,
    "SolidWorld edits must not expand into or invalidate the authored-pane arena");
});

test("thin dielectric SVO hits shade and transmit from their material record", () => {
  const shader = createSvoDrySceneFragmentWGSL();

  assert.match(shader, /fn dryMaterialThinDielectric/);
  assert.match(shader, /fn shadeDryThinDielectric/);
  assert.match(shader, /dryTraceBeyondThinWall/);
  assert.match(shader, /let cellExit=min\(nextT\.x,min\(nextT\.y,nextT\.z\)\)/);
  assert.match(shader, /opaque=payload\.opaque!=0u;glassTransmission=payload\.transmittance/);
});

test("derived cone opacity excludes SolidWorld glass", () => {
  const shader = liveSvoDerivedBuildWGSLFor();
  assert.match(shader, /Container glass remains structural geometry, but it is not an opacity/);
  assert.match(shader, /let sceneSolid=select\(sceneCoverage,0\.,\(sceneIdentity&0xffffu\)==1u\)/);
});
