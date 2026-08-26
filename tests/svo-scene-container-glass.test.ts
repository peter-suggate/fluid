import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneDocument } from "../lib/core/scene-definition";
import { SCENE_CATALOG } from "../lib/core/scenes";
import { VOXEL_MATERIAL_IDS } from "../lib/core/voxel-scene";
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
  assert.match(shader, /var<private> drySurfaceOcclusionDepth_m:f32/);
  assert.match(shader, /var depth=drySurfaceOcclusionDepth_m/);
  assert.doesNotMatch(shader, /fn dryOcclusionDepth_m/,
    "water-sort depth must reuse the glass shading traversal");
  assert.match(shader, /let cellExit=min\(nextT\.x,min\(nextT\.y,nextT\.z\)\)/);
  assert.match(shader, /opaque=payload\.opaque!=0u;glassTransmission=payload\.transmittance/);
});

test("garden scene documents contain terrain without a glass container", () => {
  const gardenDefinitions = SCENE_CATALOG.filter(({ environment }) => environment === "garden");
  assert.ok(gardenDefinitions.length > 0);

  for (const definition of gardenDefinitions) {
    const scene = sceneDocument(definition);
    assert.ok(scene.terrain, `${definition.id} must retain its generated terrain vessel`);
    assert.ok(scene.solidVoxels.every((patch) => patch.operation !== "fill"
      || (patch.materialId ?? VOXEL_MATERIAL_IDS.containerGlass)
        !== VOXEL_MATERIAL_IDS.containerGlass),
    `${definition.id} must not compile a glass tank into SolidWorld`);
    assert.equal(buildSvoSceneGlass(scene, { environmentId: definition.environment })
      .descriptors.length, 0, `${definition.id} must not publish analytic glazing`);
  }
});

test("derived cone opacity excludes SolidWorld glass", () => {
  const shader = liveSvoDerivedBuildWGSLFor();
  assert.match(shader, /Container glass remains structural geometry, but it is not an opacity/);
  assert.match(shader, /let sceneSolid=select\(sceneCoverage,0\.,\(sceneIdentity&0xffffu\)==1u\)/);
});
