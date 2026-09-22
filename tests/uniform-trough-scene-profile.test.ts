import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getSceneDefinition } from "../lib/core/scenes";
import { createMethodStore } from "../lib/core/stores/method-store";
import { sceneDocument } from "../lib/core/scene-definition";
import { buildVesselOutlineGeometry } from "../lib/core/vessel-outline";
import { createSolidWorld, sampleSolidWorld } from "../lib/core/solid-world";
import { buildSvoSolidWorldPlanarBoundaryCatalog, svoPlanarResidualSolidWorld } from "../lib/svo/features/scene-publication/svo-planar-boundary";
import { createSvoEnvironmentCoarsening, solidWorldVoxelPatchCoarseningRegions } from "../lib/svo/features/construction/svo-environment-coarsening";

for (const mode of ["dam-break", "settled-tank", "hose-fill"] as const) {
  test(`voxel trough ${mode} publishes opaque voxel walls without a tank outline`, () => {
    const scene = sceneDocument(getSceneDefinition(`uniform-trough-${mode}`));
    assert.equal(buildVesselOutlineGeometry(scene), undefined);
    const world = createSolidWorld(scene.solidVoxels);
    const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches, 0,
      { promoteEditablePatches: false });
    const residual = svoPlanarResidualSolidWorld(world, catalog);
    const refinement = createSvoEnvironmentCoarsening({
      primitives: [], regions: solidWorldVoxelPatchCoarseningRegions(scene, residual.patches),
      worldOrigin_m: [-2, 0, -1.6], nodeEdge_m: [[0.8, 0.8, 0.8], [0.4, 0.4, 0.4]],
      brickSize: 8, maximumDepth: 1, crowdingTarget: 16,
    });
    // Above the stage, all four thin walls still require 5 cm render voxels.
    // A 10 cm cell centre can miss the complete 5 cm slab.
    for (const coordinate of [{ x: 0, y: 1, z: 2 }, { x: 4, y: 1, z: 2 },
      { x: 2, y: 1, z: 1 }, { x: 2, y: 1, z: 2 }]) {
      assert.equal(refinement.refineEnvironmentLeaf(0, coordinate), true);
    }
    for (const point of [[32, 0, 12], [-1, 12, 12], [64, 12, 12],
      [32, 12, -1], [32, 12, 24]] as const) {
      assert.equal(sampleSolidWorld(residual, point).materialId, 17);
    }
    // Even an old document retaining the default presentation cannot replace
    // these opaque walls with the glass tank's outline.
    delete scene.container.vessel;
    assert.equal(buildVesselOutlineGeometry(scene), undefined);
  });
  test(`voxel trough ${mode} can seed its profile when opened`, () => {
    const profile = getSceneDefinition(`uniform-trough-${mode}`).methodProfile;
    assert.ok(profile);
    const store = createMethodStore();
    // Scene selection uses seedProfile, including registry/parameter validation.
    assert.doesNotThrow(() => store.getState().seedProfile(profile));
    assert.equal(profile.methodId, "uniform-volume");
    assert.deepEqual(store.getState().overrides["uniform-volume"], {});
  });
}
