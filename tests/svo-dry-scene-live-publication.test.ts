import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { environmentIds, type EnvironmentId } from "../lib/environments";
import type { SceneDescription } from "../lib/model";
import { scenePresets } from "../lib/scenes";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { buildSvoPrimitiveCandidates } from "../lib/svo-primitive-candidates";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  svoMaterialFromEnvironmentProxyMaterial,
  svoMaterialFunctionIdForEnvironmentProxy,
} from "../lib/svo-material-abi";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildSvoTerrainMaterial } from "../lib/svo-terrain-material";
import { sceneHasTerrain } from "../lib/terrain";
import {
  buildOctreeSvoEnvironmentLightingPublication,
  buildOctreeSvoLightPublication,
  buildOctreeSvoPbrMaterialPublication,
  OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS,
  octreeSparseBrickStructuralFinalizeShader,
} from "../lib/webgpu-octree-sparse-bricks";
import {
  SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW,
  SPARSE_SCENE_MAINTENANCE_OVERFLOW,
  sparseSceneRevisionIncomplete,
} from "../lib/webgpu-sparse-scene-proxies";
import {
  buildSparseVoxelDrySceneLightingMirrors,
  canEncodeSparseVoxelDryScene,
  SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import { SPARSE_VOXEL_VALID_FIELDS, type SparseVoxelSceneRenderSource } from "../lib/webgpu-voxel-debug";

const octreeSource = readFileSync(fileURLToPath(new URL("../lib/webgpu-octree.ts", import.meta.url)), "utf8");
const sparseWorldSource = readFileSync(
  fileURLToPath(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url)),
  "utf8",
);

/**
 * Regression guard for live publication ordering: analytic primitives become
 * renderable immediately, while the SVO is consumed only after its scene
 * generation and terminal lifecycle are current.
 */

test("the dry renderer's required fields are the shared ABI bits, not a hand-copied literal", () => {
  assert.equal(
    SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    SPARSE_VOXEL_VALID_FIELDS.topology
    | SPARSE_VOXEL_VALID_FIELDS.sceneGeometry
    | SPARSE_VOXEL_VALID_FIELDS.materialOwner,
  );
  assert.match(svoDrySceneShader, new RegExp(`const REQUIRED_FIELDS:u32 = ${SVO_DRY_SCENE_REQUIRED_VALID_FIELDS}u;`),
    "the shader mask must be generated from the exported constant so the two cannot drift");
});

test("a live-scene publication satisfies every field an SVO ray needs", () => {
  assert.equal(
    OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS & SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    "the live producer must publish topology, scene geometry and material owners",
  );
  // Fluid, velocity and dynamic solids are deliberately absent: the octree lane
  // presents water through raster extraction, and a consumer that needs those
  // must reject the generation rather than read a buffer nobody filled.
  for (const absent of ["dynamicSolid", "coarseFluid", "fineFluid", "velocity"] as const) {
    assert.equal(OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS & SPARSE_VOXEL_VALID_FIELDS[absent], 0,
      `${absent} has no scene payload and must not be claimed valid`);
  }
});

test("the live finalizer is overflow-gated and closes with the completion fence", () => {
  const liveEntry = octreeSparseBrickStructuralFinalizeShader.slice(
    octreeSparseBrickStructuralFinalizeShader.indexOf("fn finalizeScene()"),
  );
  assert.ok(liveEntry, "the finalizer must expose a live-scene entry point");
  // Gated on the overflows that mean a brick still holds the previous revision,
  // not on the flag word. A per-brick candidate overflow rebuilt that brick with
  // fewer primitives than overlapped it; withholding the generation for it took
  // every derived-lighting page down and rendered the whole octree domain as
  // full shadow. See SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW.
  assert.match(liveEntry, new RegExp(
    `completed != requested\\s*\\|\\| \\(overflow & ${SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW}u\\) != 0u\\s*\\|\\| topologyMutation\\[3\\] != 0u`));
  assert.equal(sparseSceneRevisionIncomplete(SPARSE_SCENE_MAINTENANCE_OVERFLOW.candidates), false);
  assert.match(liveEntry, /state\[1\] \|= SCENE_VALID_FIELDS;/);
  assert.match(liveEntry, /state\[2\] \+= 1u;/);
  assert.match(liveEntry, /state\[3\] \+= 1u;/);
  const validFieldsAt = liveEntry.indexOf("state[1] |= SCENE_VALID_FIELDS;");
  const generationAt = liveEntry.indexOf("state[0] +=");
  assert.ok(generationAt > validFieldsAt,
    "completeGeneration is the fence: it must be written after every field it certifies");
  assert.match(sparseWorldSource, new RegExp(`const SCENE_VALID_FIELDS: u32 = \\$\\{OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS\\}u;`),
    "the WGSL mask must be generated from the exported constant");
});

test("the octree render world invokes live scene maintenance", () => {
  const publication = octreeSource.slice(
    octreeSource.indexOf("encodeSparseBrickWorld(encoder:"),
    octreeSource.indexOf("\n  destroy()", octreeSource.indexOf("encodeSparseBrickWorld(encoder:")),
  );
  assert.ok(publication, "the octree projection must expose a render-world publication");
  assert.match(publication, /this\.sparseBrickWorld\?\.encodeSceneMaintenance\(encoder\)/);
});

test("live maintenance mutates topology, repairs dirty payload, then finalizes", () => {
  const encodeLive = sparseWorldSource.slice(
    sparseWorldSource.indexOf("encodeSceneMaintenance(encoder: GPUCommandEncoder, deferDerived = false)"),
    sparseWorldSource.indexOf("\n  encode(encoder: GPUCommandEncoder"),
  );
  const publishAt = encodeLive.indexOf("this.tree.encodePublish(encoder, this.source)");
  const mutateAt = encodeLive.indexOf("this.topologyMutator.encode");
  const proxiesAt = encodeLive.indexOf("this.proxyVoxelizer.encodeMaintenance(encoder)");
  const finalizeAt = encodeLive.indexOf("this.structuralScenePipeline");
  assert.ok(publishAt >= 0 && mutateAt > publishAt && proxiesAt > mutateAt && finalizeAt > proxiesAt);
  assert.doesNotMatch(encodeLive, /encodeFromDenseFields|fields\.levelSet|fields\.velocity/,
    "presentation-only maintenance must not require a physics step");
  assert.match(sparseWorldSource, /sceneGeometry: \{ buffer: this\.tree\.sceneGeometry/);
  assert.match(sparseWorldSource, /geometry: \{ buffer: this\.tree\.geometry/);
});

/**
 * The catalogs grew from ~19 to 70-114 authored props per environment. Sweep
 * every shipped pairing so a catalog that outgrows a renderer capacity is a
 * test failure rather than a black viewport.
 */
function structuralSourceFor(scene: SceneDescription): SparseVoxelSceneRenderSource {
  const binding = { buffer: {} as GPUBuffer };
  const pbrMaterials = buildOctreeSvoPbrMaterialPublication();
  const lights = buildOctreeSvoLightPublication(scene);
  const environmentLighting = buildOctreeSvoEnvironmentLightingPublication(scene);
  const field = (bit: number) => ({ bit, residency: "all-published-leaves" as const });
  return {
    materialCount: pbrMaterials.count,
    pbrMaterials: { binding, count: pbrMaterials.count, strideBytes: pbrMaterials.strideBytes, revision: pbrMaterials.revision },
    lights: { binding, count: lights.count, strideBytes: lights.strideBytes, revision: lights.revision },
    environmentLighting: {
      binding,
      count: 1,
      strideBytes: environmentLighting.strideBytes,
      revision: environmentLighting.revision,
      cacheKey: environmentLighting.cacheKey,
    },
    revision: 1,
    structural: {
      fields: {
        topology: field(SPARSE_VOXEL_VALID_FIELDS.topology),
        sceneGeometry: field(SPARSE_VOXEL_VALID_FIELDS.sceneGeometry),
        dynamicSolid: field(SPARSE_VOXEL_VALID_FIELDS.dynamicSolid),
        coarseFluid: field(SPARSE_VOXEL_VALID_FIELDS.coarseFluid),
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, residency: "unavailable" },
        velocity: field(SPARSE_VOXEL_VALID_FIELDS.velocity),
        materialOwner: field(SPARSE_VOXEL_VALID_FIELDS.materialOwner),
      },
    } as unknown as SparseVoxelSceneRenderSource["structural"],
  };
}

function drySceneDataFor(scene: SceneDescription): SparseVoxelDrySceneData {
  const scenePrimitives = buildSvoScenePrimitives(scene);
  const sceneGlass = buildSvoSceneGlass(scene);
  const sceneThickGlass = buildSvoSceneThickGlass(scene);
  // An environment may contribute analytic terrain to a preset that authored
  // none; the terrain material is keyed off the authored description.
  const terrainMaterial = scenePrimitives.analyticTerrain && sceneHasTerrain(scene) ? buildSvoTerrainMaterial(scene) : undefined;
  const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
  const revision = 1;
  const primitiveCandidates = scenePrimitives.primitiveCandidates
    ?? buildSvoPrimitiveCandidates(
      scenePrimitives.descriptors as Parameters<typeof buildSvoPrimitiveCandidates>[0],
      { skippedOwnerId: scenePrimitives.openShellOwnerId },
    );
  const materialRecords = packSvoMaterialTable([
    ...buildDefaultSvoMaterialRecords(revision),
    ...scenePrimitives.metadata.map((primitive) => svoMaterialFromEnvironmentProxyMaterial(
      primitive.materialId,
      primitive.material,
      revision,
      svoMaterialFunctionIdForEnvironmentProxy(primitive),
    )),
  ]);
  return {
    renderRevision: revision,
    primitiveRecords: scenePrimitives.packedRecords,
    primitiveCandidates,
    materialRecords,
    materialRevision: revision,
    ownerBase: scene.rigidBodies.length,
    skippedOwnerId: scenePrimitives.openShellOwnerId,
    terrainMaterialId: scenePrimitives.analyticTerrain?.materialId,
    terrainMaterialMetadata: terrainMaterial?.packedMetadata,
    terrainMaterialCacheKey: terrainMaterial?.cacheKey,
    glassRecords: sceneGlass.packedRecords,
    glassCacheKey: sceneGlass.cacheKey,
    thickGlassRecords: sceneThickGlass.packedRecords,
    thickGlassRevision: sceneThickGlass.revision,
    thickGlassCacheKey: sceneThickGlass.cacheKey,
    primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass[0]?.paneId,
    primaryCompositeOwnedGlassPaneCount: compositorOwnedGlass.length,
    ...buildSparseVoxelDrySceneLightingMirrors(scene, revision),
  };
}

test("every shipped preset renders through the dry-scene contract in every environment", () => {
  const failures: string[] = [];
  for (const preset of scenePresets) {
    for (const environment of environmentIds as readonly EnvironmentId[]) {
      const scene: SceneDescription = { ...preset.create(), environment };
      const source = structuralSourceFor(scene);
      const drySceneData = drySceneDataFor(scene);
      const scenePrimitives = buildSvoScenePrimitives(scene);
      if (scenePrimitives.requiresRasterTerrainFallback) failures.push(`${preset.id}/${environment}: terrain falls back to raster`);
      if (scenePrimitives.unsupportedSources.length > 0) {
        failures.push(`${preset.id}/${environment}: unsupported sources ${scenePrimitives.unsupportedSources.join(",")}`);
      }
      if (!canEncodeSparseVoxelDryScene(source, drySceneData)) failures.push(`${preset.id}/${environment}: dry-scene contract rejected`);
    }
  }
  assert.deepEqual(failures, []);
});
