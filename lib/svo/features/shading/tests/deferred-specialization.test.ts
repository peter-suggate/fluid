import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../../../../core/model";
import { buildDefaultSvoMaterialRecords, packSvoMaterialTable } from "../../../contracts/svo-material-abi";
import type { SparseVoxelDrySceneData } from "../../../contracts/scene-publication";
import { canUseOpaqueDirectionalCones, publishOpaqueSurfaceCapability } from "../deferred-specialization";
import { createSvoDrySceneFragmentWGSL } from "../program";
import { shadingExperimentSource } from "../../../../../tools/svo-shading-experiment";

const options = { coneMode: "cones", hierarchyReady: true, globalIllumination: false, reconstruction: "full-res-relight" };
function publication(): SparseVoxelDrySceneData {
  const lights = new Uint32Array(28); lights[24] = 1;
  return { opaqueSurfaceOnly: true, lightRecords: lights } as SparseVoxelDrySceneData;
}

test("live changes cannot reuse an unsupported specialized closure", () => {
  const scene = publication();
  assert.equal(canUseOpaqueDirectionalCones(scene, options), true);
  for (const patch of [
    { coneMode: "exact" }, { coneMode: "off" }, { hierarchyReady: false },
    { globalIllumination: true }, { reconstruction: "joint-bilateral" },
  ]) assert.equal(canUseOpaqueDirectionalCones(scene, { ...options, ...patch }), false);
  for (const patch of [
    { opaqueSurfaceOnly: undefined }, { opaqueSurfaceOnly: false },
    { glassRecords: new Uint32Array(20) }, { thickGlassRecords: new Uint32Array(32) },
    { lightRecords: new Uint32Array(56) }, { lightRecords: new Uint32Array(28) },
    { lightRecords: undefined },
  ]) assert.equal(canUseOpaqueDirectionalCones({ ...scene, ...patch }, options), false);
  scene.lightRecords![24] = 2;
  assert.equal(canUseOpaqueDirectionalCones(scene, options), false);
  scene.lightRecords![24] = 1;
  assert.equal(canUseOpaqueDirectionalCones(scene, options), true);
});

test("opaque proof follows solid edits, primitive materials and material flags", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [];
  const materials = packSvoMaterialTable(buildDefaultSvoMaterialRecords(1));
  const primitives = new Uint32Array(16); primitives[7] = 2;
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), true);
  scene.solidVoxels.push({ operation: "fill", minimum: [0,0,0], maximumExclusive: [1,1,1] });
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), false, "default filled voxels are glass");
  scene.solidVoxels = [{ operation: "fill", minimum: [0,0,0], maximumExclusive: [1,1,1], materialId: 2 }];
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), true);
  primitives[7] = 1;
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), false);
  primitives[7] = 65535;
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), false);
  primitives[7] = 2; materials[2 * 24 + 23] = 7;
  assert.equal(publishOpaqueSurfaceCapability(scene, materials, primitives), false);
});

test("production specialized source matches the benchmarked closure", () => {
  const generate = (fast: boolean) => createSvoDrySceneFragmentWGSL(.5, "raster-primary", "off", "split", 0, false, true, false, true,
    { surfaceMesh: true, voxelLightCache: false, globalIlluminationAbsent: true, opaqueDirectionalCones: fast });
  const normalize = (source: string) => source.replace(/\/\/[^\n]*/g, "").replace(/\s+/g, "");
  const reference = shadingExperimentSource(generate(false), "cone-only-opaque-one-light-visibility-guide");
  assert.equal(normalize(generate(true)), normalize(reference));
});
