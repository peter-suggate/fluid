import { sceneUsesFlatVoxelNormals, type SceneDescription } from "../../../core/model";
import { solidWorldContentStamp } from "../../../core/solid-world";
import type { EnvironmentProxyCatalog } from "../../../core/voxel-environments";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../../../core/webgpu-rigid-body";
import type { SparseVoxelDrySceneData } from "../../contracts/scene-publication";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  SVO_MATERIAL_RECORD_WORDS,
  svoMaterialFromEnvironmentProxyMaterial,
  svoMaterialFunctionIdForEnvironmentProxy,
  writeSvoEnvironmentMaterialRecord,
  writeSvoMaterialFunctionId,
  writeSvoMaterialRecord,
  type SvoMaterialRecord,
} from "../../contracts/svo-material-abi";
import { buildSparseVoxelDrySceneLightingMirrors } from "../../pipeline/webgpu-svo-dry-scene";
import { sceneSvoGroundPlane, sceneTerrainSurfaceModel, type SvoGroundPlane, type SvoTerrainSurfaceModel } from "../materials/svo-terrain-material";
import { publishOpaqueSurfaceCapability } from "../shading/deferred-specialization";
import { successorEnvironmentCatalog } from "./scene-environment-catalog";
import { buildSvoPrimitiveCandidates, type SvoPrimitiveCandidatePublication } from "./svo-primitive-candidates";
import { buildSvoScenePrimitives, type SvoEnvironmentPrimitiveMetadata, type SvoScenePrimitiveBuild } from "./svo-scene-primitives";

/**
 * The revision every material and light record of a render publication carries.
 *
 * A record's revision says which table it belongs to and the shader checks it
 * against the table's own, so a revision that moved with every publication
 * restamped every record. One that stands lets a changed material be one
 * record's bytes; what changed is found by comparing the tables.
 */
const RENDER_SCENE_TABLE_REVISION = 1;

type LightingMirrors = NonNullable<ReturnType<typeof buildSparseVoxelDrySceneLightingMirrors>>;

/**
 * What separates a publication from the one it followed: nothing the dry scene
 * reads, only what shading reads from tables, or a solid.
 */
export type SvoRenderSceneChange = "none" | "appearance" | "geometry";

/** One document's dry-scene publication, kept so that its successor is derived from it. */
export interface SvoRenderScenePublication {
  readonly scene: SceneDescription;
  /** The sparse world's revision when the document was published. */
  readonly worldRevision: number;
  readonly catalog: EnvironmentProxyCatalog;
  readonly primitives: SvoScenePrimitiveBuild;
  readonly terrainSurface: SvoTerrainSurfaceModel;
  readonly lighting: LightingMirrors;
  readonly solidStamp: string;
  /** Everything of `SparseVoxelDrySceneData` but the render revision, which the renderer counts. */
  readonly data: Omit<SparseVoxelDrySceneData, "renderRevision">;
  readonly change: SvoRenderSceneChange;
}

/** The candidate index of a primitive list, by the list: a recoloured build shares its predecessor's. */
const candidateIndices = new WeakMap<object, { readonly skippedOwnerId: number | undefined; readonly candidates: SvoPrimitiveCandidatePublication }>();

function primitiveCandidates(primitives: SvoScenePrimitiveBuild): SvoPrimitiveCandidatePublication {
  if (primitives.primitiveCandidates) return primitives.primitiveCandidates;
  const held = candidateIndices.get(primitives.descriptors);
  if (held && held.skippedOwnerId === primitives.openShellOwnerId) return held.candidates;
  const candidates = buildSvoPrimitiveCandidates(primitives.descriptors as Parameters<typeof buildSvoPrimitiveCandidates>[0], {
    skippedOwnerId: primitives.openShellOwnerId,
  });
  candidateIndices.set(primitives.descriptors, { skippedOwnerId: primitives.openShellOwnerId, candidates });
  return candidates;
}

function environmentMaterialRecord(primitive: SvoEnvironmentPrimitiveMetadata, terrainSurface: SvoTerrainSurfaceModel): SvoMaterialRecord {
  return svoMaterialFromEnvironmentProxyMaterial(
    primitive.materialId,
    primitive.material,
    RENDER_SCENE_TABLE_REVISION,
    // The same word that chose the ground's closure chooses every prop's:
    // a porcelain scene is one fired material throughout, not a porcelain
    // floor with granite on it.
    svoMaterialFunctionIdForEnvironmentProxy(primitive, terrainSurface),
  );
}

/**
 * The complete material table, as the previous one wherever nothing it reads
 * moved.
 *
 * A table is direct-index and a primitive keeps its slot for as long as the
 * list keeps its order, so a primitive whose metadata is another object is one
 * record written over a copy. A list of another length or a slot that moved
 * is a table packed from nothing.
 */
function materialTable(
  previous: SvoRenderScenePublication | undefined,
  primitives: SvoScenePrimitiveBuild,
  terrainSurface: SvoTerrainSurfaceModel,
): Uint32Array<ArrayBuffer> {
  const metadata = primitives.metadata;
  const published = previous?.primitives.metadata;
  if (previous && published && published.length === metadata.length) {
    const held = previous.data.materialRecords;
    const sameSurface = previous.terrainSurface === terrainSurface;
    if (published === metadata && sameSurface) return held;
    let table: Uint32Array<ArrayBuffer> | undefined = held.slice();
    let moved = !sameSurface;
    // Another surface model is other closures: the built-in records it
    // reaches, and one word of every primitive's whose record otherwise stands.
    if (!sameSurface) {
      for (const record of buildDefaultSvoMaterialRecords(RENDER_SCENE_TABLE_REVISION, { terrainSurface })) writeSvoMaterialRecord(table, record);
    }
    for (let index = 0; index < metadata.length; index += 1) {
      const entry = metadata[index]!;
      if (entry.materialId !== published[index]!.materialId) { table = undefined; break; }
      if (entry === published[index]) {
        if (!sameSurface) writeSvoMaterialFunctionId(table, entry.materialId, svoMaterialFunctionIdForEnvironmentProxy(entry, terrainSurface));
        continue;
      }
      writeSvoEnvironmentMaterialRecord(table, entry.materialId, entry.material, RENDER_SCENE_TABLE_REVISION,
        svoMaterialFunctionIdForEnvironmentProxy(entry, terrainSurface));
      const offset = entry.materialId * SVO_MATERIAL_RECORD_WORDS;
      for (let word = offset; !moved && word < offset + SVO_MATERIAL_RECORD_WORDS; word += 1) moved = table[word] !== held[word];
    }
    if (table) return moved ? table : held;
  }
  return packSvoMaterialTable([
    ...buildDefaultSvoMaterialRecords(RENDER_SCENE_TABLE_REVISION, { terrainSurface }),
    ...metadata.map((primitive) => environmentMaterialRecord(primitive, terrainSurface)),
  ]);
}

function sameWords(a: Uint32Array | undefined, b: Uint32Array | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

/**
 * The light and environment-lighting mirrors, as the previous ones whenever
 * they say the same thing.
 *
 * A fixture is a catalog primitive that emits, so the same catalog under the
 * same authored lighting is the same publication and nothing is walked.
 */
function lightingMirrors(
  previous: SvoRenderScenePublication | undefined,
  scene: SceneDescription,
  catalog: EnvironmentProxyCatalog,
): LightingMirrors | undefined {
  if (previous && previous.catalog === catalog && previous.scene.lighting === scene.lighting
    && previous.scene.environment === scene.environment && previous.scene.systems === scene.systems) return previous.lighting;
  const lighting = buildSparseVoxelDrySceneLightingMirrors(scene, RENDER_SCENE_TABLE_REVISION);
  if (!lighting || !previous) return lighting;
  const held = previous.lighting;
  return sameWords(lighting.lightRecords, held.lightRecords)
    && sameWords(lighting.environmentLightingRecord, held.environmentLightingRecord)
    && lighting.environmentLightingCacheKey === held.environmentLightingCacheKey ? held : lighting;
}

function sameGroundPlane(a: SvoGroundPlane | undefined, b: SvoGroundPlane | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.height_m === b.height_m && a.hazeDistance_m === b.hazeDistance_m
    && a.backdropContentRadius_m === b.backdropContentRadius_m
    && a.footprint_m.every((value, index) => value === b.footprint_m[index]);
}

/**
 * The dry-scene publication of a document, derived from the one before it.
 *
 * Every part that says what it said before is the previous publication's own
 * array, so the renderer's arenas find an unchanged region by identity and a
 * changed one by comparing words. A document that changed nothing the dry
 * scene reads comes back as `none` and is not published at all.
 *
 * `worldRevision` is the sparse world's after the document was staged into it:
 * the world mints one for every change to a voxel and for nothing else, which
 * is what makes it the authority on whether a solid moved.
 *
 * Undefined when the document's lighting cannot be mirrored.
 */
export function svoRenderScenePublication(
  scene: SceneDescription,
  worldRevision: number,
  previous: SvoRenderScenePublication | undefined,
): SvoRenderScenePublication | undefined {
  if (previous && previous.scene === scene && previous.worldRevision === worldRevision) {
    return previous.change === "none" ? previous : { ...previous, change: "none" };
  }
  const catalog = successorEnvironmentCatalog(previous?.scene, scene);
  const primitives = buildSvoScenePrimitives(scene);
  const candidates = primitiveCandidates(primitives);
  const terrainSurface = sceneTerrainSurfaceModel(scene);
  const materialRecords = materialTable(previous, primitives, terrainSurface);
  const lighting = lightingMirrors(previous, scene, catalog);
  if (!lighting) return undefined;
  const solidStamp = solidWorldContentStamp(scene);
  const held = previous?.data;
  const flatVoxelNormals = sceneUsesFlatVoxelNormals(scene);
  const geometryMoved = !previous || !held || previous.worldRevision !== worldRevision || previous.solidStamp !== solidStamp
    || primitives.packedRecords !== held.primitiveRecords || candidates !== held.primitiveCandidates
    || primitives.clusterBlocks !== held.clusterBlocks || primitives.fieldProgramBlocks !== held.fieldProgramBlocks
    || flatVoxelNormals !== held.flatVoxelNormals;
  // The plane is levelled from the ground's own heights, sampled around its
  // rim; a document that shares the ground and placed the same solids has the
  // plane it had.
  const ground = held && previous.scene.terrain === scene.terrain && previous.scene.container === scene.container
    && previous.scene.backdrop === scene.backdrop && previous.scene.environment === scene.environment
    && primitives.descriptors === previous.primitives.descriptors
    ? held.groundPlane : sceneSvoGroundPlane(scene);
  const groundPlane = held && sameGroundPlane(ground, held.groundPlane) ? held.groundPlane : ground;
  // The capability reads the solids' materials, the bodies' and the table.
  const opaqueSurfaceOnly = held && !geometryMoved && materialRecords === held.materialRecords
    && previous.scene.rigidBodies === scene.rigidBodies
    ? held.opaqueSurfaceOnly
    : publishOpaqueSurfaceCapability(scene, materialRecords, primitives.packedRecords);
  const change: SvoRenderSceneChange = geometryMoved ? "geometry"
    : materialRecords === held.materialRecords && lighting === previous.lighting && groundPlane === held.groundPlane
      && opaqueSurfaceOnly === held.opaqueSurfaceOnly && primitives.openShellOwnerId === held.skippedOwnerId ? "none" : "appearance";
  return {
    scene, worldRevision, catalog, primitives, terrainSurface, lighting, solidStamp, change,
    data: change === "none" ? held! : {
      voxelGeometryRevision: geometryMoved ? ((held?.voxelGeometryRevision ?? 0) % 0xffff_fffe) + 1 : held.voxelGeometryRevision,
      primitiveRecords: primitives.packedRecords,
      primitiveCandidates: candidates,
      materialRecords,
      opaqueSurfaceOnly,
      materialRevision: RENDER_SCENE_TABLE_REVISION,
      ownerBase: SCENE_ENVIRONMENT_OWNER_BASE,
      skippedOwnerId: primitives.openShellOwnerId,
      // An aggregate's packing does not fit in a 64-byte record, so the record
      // carries a reference and the numbers live in the arena. Publication
      // order assigns the references, so these arrive already in agreement
      // with the records beside them.
      clusterBlocks: primitives.clusterBlocks,
      // The same door again, for the tape kind. A `field-program` record whose
      // block never arrives resolves to zeroes, which the shader reads as "not
      // resolved" and draws as nothing — the aggregate's failure one level up.
      fieldProgramBlocks: primitives.fieldProgramBlocks,
      ...lighting,
      flatVoxelNormals,
      groundPlane,
    },
  };
}
