import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import {
  SVO_MATERIAL_FUNCTION_IDS,
  svoMaterialFunctionIdForEnvironmentProxy,
  unpackSvoMaterialRecord,
} from "../lib/svo-material-abi";
import {
  SVO_PROCEDURAL_MATERIAL_POLICIES,
  SVO_PROCEDURAL_VARIATION_ACTIVE,
  evaluateSvoProceduralMaterial,
  sampleSvoProceduralNoise,
  svoProceduralHashCell,
  svoProceduralMaterialWGSL,
} from "../lib/svo-procedural-material";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import {
  buildOctreeSvoPbrMaterialPublication,
  ENVIRONMENT_VOXEL_MATERIAL_BASE,
  OCTREE_SVO_PBR_MATERIAL_REVISION,
} from "../lib/webgpu-octree-sparse-bricks";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

test("material-function IDs and seeds are stable, unique ABI values", () => {
  assert.deepEqual(SVO_MATERIAL_FUNCTION_IDS, {
    none: 0,
    gardenTerrain: 1,
    architecturalSurface: 2,
    wood: 3,
    stone: 4,
    foliage: 5,
    ceramic: 6,
    brushedMetal: 7,
    organic: 8,
  });
  assert.equal(new Set(SVO_PROCEDURAL_MATERIAL_POLICIES.map(({ functionId }) => functionId)).size,
    SVO_PROCEDURAL_MATERIAL_POLICIES.length);
  assert.equal(new Set(SVO_PROCEDURAL_MATERIAL_POLICIES.map(({ seed }) => seed)).size,
    SVO_PROCEDURAL_MATERIAL_POLICIES.length);
  assert.equal(OCTREE_SVO_PBR_MATERIAL_REVISION, 2, "new material functions invalidate revision-one publications");
});

test("authored semantic groups select deterministic functions without changing identity", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId), true);
    const publication = buildOctreeSvoPbrMaterialPublication(OCTREE_SVO_PBR_MATERIAL_REVISION, primitives);
    for (const primitive of primitives) {
      const expectedFunction = svoMaterialFunctionIdForEnvironmentProxy(primitive);
      const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
      const record = unpackSvoMaterialRecord(publication.packedRecords, materialId);
      assert.equal(record.materialId, materialId, primitive.key);
      assert.equal(record.materialFunctionId, expectedFunction, primitive.key);
      if (primitive.tags.includes("shell")) {
        assert.equal(expectedFunction, SVO_MATERIAL_FUNCTION_IDS.architecturalSurface, primitive.key);
      }
    }
  }

  const garden = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, "garden"), true);
  assert.ok(garden.every((primitive) => svoMaterialFunctionIdForEnvironmentProxy(primitive) !== SVO_MATERIAL_FUNCTION_IDS.none),
    "tree, foliage, mushrooms, and pebbles all retain bounded procedural variation");
  assert.ok(garden.filter(({ group }) => group === "leaf-foliage").every((primitive) =>
    svoMaterialFunctionIdForEnvironmentProxy(primitive) === SVO_MATERIAL_FUNCTION_IDS.foliage));
  assert.ok(garden.filter(({ group }) => group === "stone-pebble").every((primitive) =>
    svoMaterialFunctionIdForEnvironmentProxy(primitive) === SVO_MATERIAL_FUNCTION_IDS.stone));
});

test("CPU seeded noise is repeatable, bounded, and continuous across world-space seams", () => {
  assert.equal(svoProceduralHashCell([-7, 11, 23], 0x243f_6a88), 0.5717669129371643);
  assert.equal(sampleSvoProceduralNoise(
    { x: 0.137, y: -0.241, z: 1.091 }, [2.25, 2.25, 2.25], 0x243f_6a88,
  ), 0.5882431864738464);

  for (const policy of SVO_PROCEDURAL_MATERIAL_POLICIES) {
    const position = { x: 0.137, y: -0.241, z: 1.091 };
    const first = evaluateSvoProceduralMaterial(policy.functionId, [0.5, 0.4, 0.3], 0.6, position);
    const repeated = evaluateSvoProceduralMaterial(policy.functionId, [0.5, 0.4, 0.3], 0.6, position);
    assert.deepEqual(repeated, first, policy.key);
    assert.equal(first.variationFlags, (SVO_PROCEDURAL_VARIATION_ACTIVE | policy.functionId) >>> 0);
    assert.ok(first.baseColorLinear.every((channel) => Number.isFinite(channel) && channel >= 0 && channel <= 1));
    assert.ok(first.roughness >= 0.04 && first.roughness <= 1);

    const cellBoundary = 1 / policy.frequency_mInv[0];
    const left = evaluateSvoProceduralMaterial(policy.functionId, [0.5, 0.4, 0.3], 0.6, {
      x: cellBoundary - 1e-4, y: 0.23, z: -0.17,
    });
    const right = evaluateSvoProceduralMaterial(policy.functionId, [0.5, 0.4, 0.3], 0.6, {
      x: cellBoundary + 1e-4, y: 0.23, z: -0.17,
    });
    for (let channel = 0; channel < 3; channel += 1) {
      assert.ok(Math.abs(left.baseColorLinear[channel] - right.baseColorLinear[channel]) < 1e-5,
        `${policy.key} color must remain continuous at the shared cell/primitive seam`);
    }
    assert.ok(Math.abs(left.roughness - right.roughness) < 1e-5,
      `${policy.key} roughness must remain continuous at the shared cell/primitive seam`);
  }

  const identity = evaluateSvoProceduralMaterial(SVO_MATERIAL_FUNCTION_IDS.none, [0.2, 0.4, 0.6], 0.7, { x: 1, y: 2, z: 3 });
  assert.deepEqual(identity, { baseColorLinear: [0.2, 0.4, 0.6], roughness: 0.7, variationFlags: 0 });
});

test("WGSL is generated from the CPU policy table and stays binding-free", () => {
  assert.doesNotMatch(svoProceduralMaterialWGSL, /@group|@binding|texture_/);
  assert.match(svoProceduralMaterialWGSL, /fn svoProceduralHashCell/);
  assert.match(svoProceduralMaterialWGSL, /linear\*linear\*\(vec3f\(3\.0\)-2\.0\*linear\)/,
    "cubic interpolation, not primitive-local UVs, removes procedural seams");
  for (const policy of SVO_PROCEDURAL_MATERIAL_POLICIES) {
    assert.match(svoProceduralMaterialWGSL, new RegExp(`functionId==${policy.functionId}u`), policy.key);
    assert.match(svoProceduralMaterialWGSL, new RegExp(`seed=${policy.seed}u`), policy.key);
    assert.match(svoProceduralMaterialWGSL, new RegExp(`colorAmplitude=${policy.colorAmplitude.toFixed(8)}`), policy.key);
    assert.match(svoProceduralMaterialWGSL, new RegExp(`roughnessAmplitude=${policy.roughnessAmplitude.toFixed(8)}`), policy.key);
  }
});

test("direct SVO PBR applies procedural base/roughness after identity validation and preserves emission", () => {
  assert.match(svoDrySceneShader, /let procedural=svoProceduralMaterial\(material\.identity\.z,base,roughness,position\)/);
  assert.match(svoDrySceneShader, /base=procedural\.baseColorLinear;roughness=procedural\.roughness;variationFlags=procedural\.variationFlags/);
  assert.match(svoDrySceneShader, /material\.emissiveRoughness\.xyz\+selectedEmission/,
    "procedural functions cannot amplify or replace authored emission");
  assert.match(svoDrySceneShader, /if\(terrainPolicyValid\)[^]*else\{let procedural=/,
    "the exact garden terrain policy remains authoritative and is never double-varied");
});
