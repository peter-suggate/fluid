import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { environmentIds, type EnvironmentId } from "../lib/environments";
import type { SceneDescription } from "../lib/model";
import { scenePresets } from "../lib/scenes";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildSvoTerrainMaterial } from "../lib/svo-terrain-material";
import { sceneHasTerrain } from "../lib/terrain";
import {
  buildOctreeSvoEnvironmentLightingPublication,
  buildOctreeSvoLightPublication,
  buildOctreeSvoPbrMaterialPublication,
  OCTREE_SPARSE_BRICK_STATIC_VALID_FIELDS,
  octreeSparseBrickStructuralFinalizeShader,
} from "../lib/webgpu-octree-sparse-bricks";
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
 * Regression guard for a frame that renders analytic glass and rigid bodies
 * over a black void: `traceStatic` refuses to trace until the producer has
 * published topology, static geometry and material owners, and the octree lane
 * allocated its sparse world without ever finalizing that publication. Nothing
 * in the CPU-side contract was false — only the GPU fence was never written.
 */

test("the dry renderer's required fields are the shared ABI bits, not a hand-copied literal", () => {
  assert.equal(
    SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    SPARSE_VOXEL_VALID_FIELDS.topology
    | SPARSE_VOXEL_VALID_FIELDS.staticGeometry
    | SPARSE_VOXEL_VALID_FIELDS.materialOwner,
  );
  assert.match(svoDrySceneShader, new RegExp(`const REQUIRED_FIELDS:u32 = ${SVO_DRY_SCENE_REQUIRED_VALID_FIELDS}u;`),
    "the shader mask must be generated from the exported constant so the two cannot drift");
});

test("a static-world publication satisfies every field a static primary ray needs", () => {
  assert.equal(
    OCTREE_SPARSE_BRICK_STATIC_VALID_FIELDS & SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    SVO_DRY_SCENE_REQUIRED_VALID_FIELDS,
    "a producer without a dense fluid payload must still publish topology, static geometry and material owners",
  );
  // Fluid, velocity and dynamic solids are deliberately absent: the octree lane
  // presents water through raster extraction, and a consumer that needs those
  // must reject the generation rather than read a buffer nobody filled.
  for (const absent of ["dynamicSolid", "coarseFluid", "fineFluid", "velocity"] as const) {
    assert.equal(OCTREE_SPARSE_BRICK_STATIC_VALID_FIELDS & SPARSE_VOXEL_VALID_FIELDS[absent], 0,
      `${absent} has no static payload and must not be claimed valid`);
  }
});

test("the static finalize entry point publishes the field mask and closes with the completion fence", () => {
  const staticEntry = octreeSparseBrickStructuralFinalizeShader.slice(
    octreeSparseBrickStructuralFinalizeShader.indexOf("fn finalizeStatic()"),
  );
  assert.ok(staticEntry, "the finalizer must expose a static-world entry point");
  assert.match(staticEntry, /state\[1\] = STATIC_VALID_FIELDS;/);
  assert.match(staticEntry, /state\[2\] = 1u;/, "topology revision must match the node-mip generation the renderer pins");
  assert.match(staticEntry, /state\[3\] = 1u;/, "static geometry revision must be published");
  const validFieldsAt = staticEntry.indexOf("state[1] = STATIC_VALID_FIELDS;");
  const generationAt = staticEntry.indexOf("state[0] +=");
  assert.ok(generationAt > validFieldsAt,
    "completeGeneration is the fence: it must be written after every field it certifies");
  assert.match(sparseWorldSource, new RegExp(`const STATIC_VALID_FIELDS: u32 = \\$\\{OCTREE_SPARSE_BRICK_STATIC_VALID_FIELDS\\}u;`),
    "the WGSL mask must be generated from the exported constant");
});

test("the octree render-world publication encodes the static structural publication", () => {
  const publication = octreeSource.slice(
    octreeSource.indexOf("encodeSparseBrickWorld(encoder:"),
    octreeSource.indexOf("\n  destroy()", octreeSource.indexOf("encodeSparseBrickWorld(encoder:")),
  );
  assert.ok(publication, "the octree projection must expose a render-world publication");
  assert.match(publication, /this\.sparseBrickWorld\?\.encodeStaticPublication\(encoder\)/,
    "without this call the structural publication fence stays zero and every static primary ray misses");
});

test("the static publication encodes topology, authored proxies, and the finalize pass exactly once", () => {
  const encodeStatic = sparseWorldSource.slice(
    sparseWorldSource.indexOf("encodeStaticPublication(encoder: GPUCommandEncoder)"),
    sparseWorldSource.indexOf("\n  encode(encoder: GPUCommandEncoder"),
  );
  assert.ok(encodeStatic, "the sparse world must expose a dense-field-free static publication");
  assert.match(encodeStatic, /if \(this\.destroyed \|\| this\.published\) return false;/,
    "the publication is idempotent so a per-frame caller pays nothing after bring-up");
  const publishAt = encodeStatic.indexOf("this.tree.encodePublish(encoder, this.source)");
  const proxiesAt = encodeStatic.indexOf("this.proxyVoxelizer.encode(encoder)");
  const finalizeAt = encodeStatic.indexOf("this.structuralStaticPipeline");
  assert.ok(publishAt >= 0 && proxiesAt > publishAt && finalizeAt > proxiesAt,
    "topology must publish before the proxies write into it, and both before the finalize fence");
  assert.doesNotMatch(encodeStatic, /encodeFromDenseFields|fields\.levelSet|fields\.velocity/,
    "the static publication must not depend on dense fluid fields the octree lane has released");
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
        staticGeometry: field(SPARSE_VOXEL_VALID_FIELDS.staticGeometry),
        dynamicSolid: field(SPARSE_VOXEL_VALID_FIELDS.dynamicSolid),
        coarseFluid: field(SPARSE_VOXEL_VALID_FIELDS.coarseFluid),
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, residency: "unavailable" },
        velocity: field(SPARSE_VOXEL_VALID_FIELDS.velocity),
        materialOwner: field(SPARSE_VOXEL_VALID_FIELDS.materialOwner),
      },
    } as SparseVoxelSceneRenderSource["structural"],
  };
}

function drySceneDataFor(scene: SceneDescription, source: SparseVoxelSceneRenderSource): SparseVoxelDrySceneData {
  const scenePrimitives = buildSvoScenePrimitives(scene);
  const sceneGlass = buildSvoSceneGlass(scene);
  const sceneThickGlass = buildSvoSceneThickGlass(scene);
  // An environment may contribute analytic terrain to a preset that authored
  // none; the terrain material is keyed off the authored description.
  const terrainMaterial = scenePrimitives.analyticTerrain && sceneHasTerrain(scene) ? buildSvoTerrainMaterial(scene) : undefined;
  const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
  return {
    primitiveRecords: scenePrimitives.packedRecords,
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
    ...buildSparseVoxelDrySceneLightingMirrors(scene, source),
  };
}

test("every shipped preset renders through the dry-scene contract in every environment", () => {
  const failures: string[] = [];
  for (const preset of scenePresets) {
    for (const environment of environmentIds as readonly EnvironmentId[]) {
      const scene: SceneDescription = { ...preset.create(), environment };
      const source = structuralSourceFor(scene);
      const drySceneData = drySceneDataFor(scene, source);
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
