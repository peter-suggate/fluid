import type { EnvironmentId } from "../core/environments";
import type { SceneDescription } from "../core/model";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../core/webgpu-rigid-body";
import { swayedPrimitiveDescriptor, type EnvironmentProxySway } from "../core/scenery-sway";
import type { SvoFieldProgram } from "./svo-field-program";
import {
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveDescriptor,
  type SvoSmoothUnionClusterPacking,
} from "./svo-primitive-abi";
import {
  buildSvoPrimitiveCandidates,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  type SvoPrimitiveCandidatePublication,
} from "./svo-primitive-candidates";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
  type EnvironmentProxyCatalog,
  type EnvironmentProxyMaterial,
  type EnvironmentProxyPrimitive,
} from "../core/voxel-environments";
import { VOXEL_MATERIAL_IDS } from "../core/voxel-scene";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE, type SparseSceneSolidReach } from "./webgpu-svo-sparse-bricks";
// The arena layout an aggregate's word-13 reference points into. Imported
// rather than restated because a reference that does not name the region the
// renderer uploads resolves to nothing, and a cluster that resolves to nothing
// draws nothing — with no error anywhere.
import {
  packSvoDrySceneClusters,
  packSvoDrySceneFieldPrograms,
  svoDrySceneClusterReference,
  svoDrySceneFieldProgramReference,
} from "./webgpu-svo-dry-scene";

/**
 * Defensive ceiling on what one environment may publish.
 *
 * Raised from 4 096 to match `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` for the
 * 10x acceptance scene (`hero-garden-hose-x10`,
 * `docs/svo-raster-visibility-handoff.md` W0): the hero garden publishes 501
 * records and its 10x densification publishes 5 039, and this is the *first* of
 * the two 4 096s that fired — a `RangeError` from the throw below, caught as
 * `console.error` + `{state: "blocked"}` in `lib/webgpu-renderer.ts:2020`, so
 * the scene did not draw at all.
 *
 * It costs nothing on its own; it is a bound on a number the candidate arena
 * has already reserved space for. Kept equal to the candidate ceiling rather
 * than merely below it, so a scene that gets past this one cannot then be
 * rejected by the BVH for the same reason one line later.
 */
export const SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES = SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES;
export const SVO_SCENE_PRIMITIVE_VERSION = "1" as const;

const scenePrimitiveCache = new Map<string, SvoScenePrimitiveBuild>();

export interface SvoScenePrimitiveBuildOptions {
  environmentId?: EnvironmentId;
  includeShell?: boolean;
  maximumPrimitives?: number;
  /** Nominal coverage width used only for conservative subcell audit bounds. */
  coverageCellSize_m?: number;
}

export interface SvoPrimitiveCoverageBounds {
  exact_m: EnvironmentProxyPrimitive["aabb_m"];
  conservative_m: EnvironmentProxyPrimitive["aabb_m"];
  subcellAxes: readonly ("x" | "y" | "z")[];
  policy: "exact" | "conservative-subcell";
}

export interface SvoEnvironmentPrimitiveMetadata {
  /** Dense index into descriptors and packedRecords. */
  primitiveIndex: number;
  /** Original dense owner index from EnvironmentProxyCatalog. */
  environmentOwnerIndex: number;
  /** Sparse voxel convention: rigid body count plus environment owner index. */
  ownerId: number;
  /** Sparse voxel convention: ENVIRONMENT_VOXEL_MATERIAL_BASE plus owner index. */
  materialId: number;
  key: string;
  group: string;
  tags: readonly string[];
  sourceKind: EnvironmentProxyPrimitive["kind"];
  material: EnvironmentProxyMaterial;
  shell: boolean;
  /** Front room shell is retained in the model but skipped for interior presentation. */
  openShell: boolean;
  /** Authored gust motion; the renderer re-poses this record every frame. */
  sway?: EnvironmentProxySway;
  /** Audit-only bounds; they do not alter rigid-body or solver collision physics. */
  coverageBounds: SvoPrimitiveCoverageBounds;
}

export interface SvoUnsupportedStaticSource {
  kind: "terrain-heightfield";
  fallback: "raster-terrain";
  materialId: number;
  reason: string;
}

export interface SvoAnalyticTerrainSource {
  kind: "terrain-heightfield";
  materialId: number;
  /** Matches the central-difference normal policy in lib/terrain.ts. */
  normalEpsilon_m: number;
}

export interface SvoScenePrimitiveBuild {
  environmentId: EnvironmentId;
  descriptors: readonly SvoPrimitiveDescriptor[];
  packedRecords: Uint32Array<ArrayBuffer>;
  /** Aggregate packings in publication order; index `i` is reference `i`. */
  clusterPackings: readonly SvoSmoothUnionClusterPacking[];
  /** The same, packed for the scene arena's cluster region. Absent means no aggregates. */
  clusterBlocks?: Uint32Array<ArrayBuffer>;
  /** Field-program tapes in publication order; index `i` is reference `i`. */
  fieldPrograms: readonly SvoFieldProgram[];
  /** The same, packed for the scene arena's tape region. Absent means no tapes. */
  fieldProgramBlocks?: Uint32Array<ArrayBuffer>;
  /** @deprecated Offline audit index; the renderer always uses SVO payload traversal. */
  primitiveCandidates?: SvoPrimitiveCandidatePublication;
  metadata: readonly SvoEnvironmentPrimitiveMetadata[];
  primitiveIndexByOwnerId: ReadonlyMap<number, number>;
  primitiveIndexByMaterialId: ReadonlyMap<number, number>;
  /** Owner IDs the SVO dry-scene hit loop should ignore for an interior view. */
  skipOwnerIds: readonly number[];
  openShellOwnerId?: number;
  unsupportedSources: readonly SvoUnsupportedStaticSource[];
  requiresRasterTerrainFallback: boolean;
  /** Analytic heightfield consumed directly from the packed scene uniforms. */
  analyticTerrain?: SvoAnalyticTerrainSource;
  /** Content hash over packed records, identities, bounds policy, and terrain. */
  contentRevision: string;
  /** Versioned identity used to reuse immutable packed publication records. */
  cacheKey: string;
}

function positiveSafeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function environmentIdentity(primitive: EnvironmentProxyPrimitive) {
  const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
  const ownerId = SCENE_ENVIRONMENT_OWNER_BASE + primitive.ownerIndex;
  if (materialId > 0xffff) throw new RangeError(`Environment material ID for ${primitive.key} does not fit uint16`);
  // 0xffff is the established no-owner sentinel, so authored owners stop at 0xfffe.
  if (ownerId >= 0xffff) throw new RangeError(`Environment owner ID for ${primitive.key} collides with the no-owner sentinel`);
  return { materialId, ownerId };
}

/**
 * The scene-global owner id the SVO publishes for one catalog proxy.
 *
 * This is the number a shader reads back off a hit, and it is *not* the
 * catalog's own dense index: the rigid bodies own the first
 * `SCENE_ENVIRONMENT_OWNER_BASE` ids and the environment's owners follow them.
 * Exported so that anything addressing a traced surface — the editor's hover
 * rim, for one — asks for the numbering rather than reproducing the offset.
 */
export function svoOwnerIdForEnvironmentProxy(primitive: EnvironmentProxyPrimitive): number {
  return environmentIdentity(primitive).ownerId;
}

/**
 * The SVO record one catalog proxy publishes.
 *
 * Exported because the editor's scenery picking has to intersect *exactly* the
 * geometry the renderer traces. Rebuilding the mapping beside the picker would
 * be a second definition of every scenery shape, and the first cone whose
 * radius convention drifted would make the cursor lie.
 */
export function svoDescriptorForEnvironmentProxy(primitive: EnvironmentProxyPrimitive): SvoPrimitiveDescriptor {
  return descriptorForProxy(primitive);
}

function descriptorForProxy(
  primitive: EnvironmentProxyPrimitive,
  clusterIndex = 0,
  fieldProgramIndex = 0,
): SvoPrimitiveDescriptor {
  const identity = environmentIdentity(primitive);
  // Primitive ID follows the scene-global owner ID. It is stable for the same
  // catalog, and remains distinct from the packed index.
  const base = {
    primitiveId: identity.ownerId,
    materialId: identity.materialId,
    ownerId: identity.ownerId,
    center_m: { ...primitive.center_m },
    // Undefined stays undefined: canonicalisation supplies the identity
    // rotation, so axis-aligned scenery packs exactly as it always has.
    orientation: primitive.orientation ? { ...primitive.orientation } : undefined,
  };
  if (primitive.kind === "box") {
    return { ...base, kind: "box", halfExtents_m: { ...primitive.halfSize_m } };
  }
  if (primitive.kind === "cylinder") {
    return primitive.edgeRadius_m && primitive.edgeRadius_m > 0
      ? { ...base, kind: "rounded-cylinder", radius_m: primitive.radius_m, halfHeight_m: primitive.halfHeight_m, edgeRadius_m: primitive.edgeRadius_m }
      : { ...base, kind: "cylinder", radius_m: primitive.radius_m, halfHeight_m: primitive.halfHeight_m };
  }
  if (primitive.kind === "capsule") {
    return { ...base, kind: "capsule", radius_m: primitive.radius_m, segmentHalfLength_m: primitive.halfLength_m };
  }
  if (primitive.kind === "torus") {
    return { ...base, kind: "torus", majorRadius_m: primitive.majorRadius_m, minorRadius_m: primitive.minorRadius_m };
  }
  if (primitive.kind === "cone") {
    return {
      ...base, kind: primitive.roundedEnds ? "round-cone" : "cone",
      baseRadius_m: primitive.baseRadius_m, topRadius_m: primitive.topRadius_m, halfHeight_m: primitive.halfHeight_m,
    };
  }
  if (primitive.kind === "cluster") {
    // Both halves: the packing travels on the descriptor so a caller holding
    // only this — the editor's picker does — can evaluate the exact surface the
    // renderer draws, and the reference names where the same numbers live in
    // the scene arena for the shader that has no descriptor at all.
    return {
      ...base, kind: "smooth-union-cluster",
      lobeRadii_m: { ...primitive.radius_m },
      clusterReference: svoDrySceneClusterReference(clusterIndex),
      packing: primitive.packing,
    };
  }
  // Both halves again, for the same reason and against a second arena: a tape
  // that lost its warp draws a smooth box, which is exactly the frame the kind
  // exists to stop producing.
  if (primitive.kind === "field-program") {
    return {
      ...base, kind: "field-program",
      envelopeRadii_m: { ...primitive.halfExtent_m },
      fieldProgramReference: svoDrySceneFieldProgramReference(fieldProgramIndex),
      program: primitive.program,
    };
  }
  return { ...base, kind: "ellipsoid", radii_m: { ...primitive.radius_m } };
}

function frontOpenShell(primitive: EnvironmentProxyPrimitive): boolean {
  return primitive.tags.includes("shell")
    && primitive.tags.includes("wall")
    && primitive.key.endsWith("/shell/wall-front");
}

function coverageBounds(primitive: EnvironmentProxyPrimitive, minimumWidth_m: number): SvoPrimitiveCoverageBounds {
  const axes = ["x", "y", "z"] as const;
  const exact_m = {
    min: { ...primitive.aabb_m.min },
    max: { ...primitive.aabb_m.max },
  };
  if (!(minimumWidth_m > 0)) return { exact_m, conservative_m: exact_m, subcellAxes: [], policy: "exact" };
  const center = primitive.center_m;
  const halfMinimum = 0.5 * minimumWidth_m;
  const subcellAxes = axes.filter((axis) => primitive.aabb_m.max[axis] - primitive.aabb_m.min[axis] < minimumWidth_m);
  const conservative_m = {
    min: {
      x: Math.min(primitive.aabb_m.min.x, center.x - halfMinimum),
      y: Math.min(primitive.aabb_m.min.y, center.y - halfMinimum),
      z: Math.min(primitive.aabb_m.min.z, center.z - halfMinimum),
    },
    max: {
      x: Math.max(primitive.aabb_m.max.x, center.x + halfMinimum),
      y: Math.max(primitive.aabb_m.max.y, center.y + halfMinimum),
      z: Math.max(primitive.aabb_m.max.z, center.z + halfMinimum),
    },
  };
  return { exact_m, conservative_m, subcellAxes, policy: subcellAxes.length > 0 ? "conservative-subcell" : "exact" };
}

/**
 * Convert one existing deterministic environment catalog into bounded SVO
 * primitive records without changing sparse-voxel material or owner identity.
 */
export function svoScenePrimitivesFromEnvironmentCatalog(
  catalog: EnvironmentProxyCatalog,
  options: Omit<SvoScenePrimitiveBuildOptions, "environmentId"> = {},
): SvoScenePrimitiveBuild {
  const includeShell = options.includeShell ?? true;
  const maximumPrimitives = positiveSafeInteger(
    options.maximumPrimitives ?? SVO_SCENE_DEFAULT_MAXIMUM_PRIMITIVES,
    "Maximum SVO scene primitives",
  );
  const primitives = environmentProxyPrimitives(catalog, includeShell);
  if (primitives.length > maximumPrimitives) {
    throw new RangeError(`Environment ${catalog.environmentId} needs ${primitives.length} SVO primitives, exceeding the ${maximumPrimitives} record limit`);
  }
  const coverageCellSize_m = options.coverageCellSize_m ?? 0;
  if (!Number.isFinite(coverageCellSize_m) || coverageCellSize_m < 0) {
    throw new RangeError("SVO scene primitive coverage cell size must be finite and non-negative");
  }
  const analyticTerrain: SvoAnalyticTerrainSource | undefined = catalog.shell.kind === "terrain-heightfield" ? {
    kind: "terrain-heightfield",
    materialId: VOXEL_MATERIAL_IDS.terrain,
    normalEpsilon_m: 0.02,
  } : undefined;
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    environmentId: catalog.environmentId,
    coverageCellSize_m,
    primitives,
    analyticTerrain,
  }));
  const cacheKey = `svo-scene-primitives-v${SVO_SCENE_PRIMITIVE_VERSION}:${catalog.environmentId}:${contentRevision}`;
  const cached = cachedSvoPublication(scenePrimitiveCache, cacheKey);
  if (cached) return cached;

  const descriptors: SvoPrimitiveDescriptor[] = [];
  const metadata: SvoEnvironmentPrimitiveMetadata[] = [];
  const primitiveIndexByOwnerId = new Map<number, number>();
  const primitiveIndexByMaterialId = new Map<number, number>();
  // Publication order, which is what makes block `i` the one reference `i`
  // names. Expansion is depth-first in document order, so two builds of the
  // same graph assign the same references.
  const clusterPackings: SvoSmoothUnionClusterPacking[] = [];
  // The second arena, assigned by the same rule and for the same reason.
  const fieldPrograms: SvoFieldProgram[] = [];
  let openShellOwnerId: number | undefined;

  const fieldProgramIndexByKey = new Map<string, number>();
  for (const primitive of primitives) {
    const primitiveIndex = descriptors.length;
    const { materialId, ownerId } = environmentIdentity(primitive);
    if (primitiveIndexByOwnerId.has(ownerId)) throw new Error(`Duplicate environment owner ID ${ownerId}`);
    if (primitiveIndexByMaterialId.has(materialId)) throw new Error(`Duplicate environment material ID ${materialId}`);
    const openShell = frontOpenShell(primitive);
    if (openShell && openShellOwnerId !== undefined) throw new Error("Environment catalog contains multiple front/open shell owners");
    if (openShell) openShellOwnerId = ownerId;

    // Field-program tapes are **shared**, not one per record.
    //
    // The arena holds `SVO_DRY_SCENE_FIELD_PROGRAM_CAPACITY` blocks and a record
    // addresses one by index, so nothing ever required the mapping to be
    // injective — but it was, and that made the arena a ceiling on *records*
    // rather than on distinct shapes. A dense crown is the case that found it:
    // a canopy can be a thousand masses drawn from a handful of tapes, and a
    // quantised shape family can ask for eight
    // distinct programs however many masses hang off them.
    //
    // Keyed on the serialised tape, which is the whole of what a block holds, so
    // two records share a block exactly when the shader would read identical
    // words out of it.
    const programKey = primitive.kind === "field-program" ? JSON.stringify(primitive.program) : undefined;
    const sharedIndex = programKey === undefined ? undefined : fieldProgramIndexByKey.get(programKey);
    const descriptor = descriptorForProxy(
      primitive, clusterPackings.length, sharedIndex ?? fieldPrograms.length,
    );
    if (descriptor.kind === "smooth-union-cluster" && descriptor.packing) clusterPackings.push(descriptor.packing);
    if (descriptor.kind === "field-program" && descriptor.program && sharedIndex === undefined) {
      if (programKey !== undefined) fieldProgramIndexByKey.set(programKey, fieldPrograms.length);
      fieldPrograms.push(descriptor.program);
    }
    descriptors.push(descriptor);
    metadata.push({
      primitiveIndex,
      environmentOwnerIndex: primitive.ownerIndex,
      ownerId,
      materialId,
      key: primitive.key,
      group: primitive.group,
      tags: [...primitive.tags],
      sourceKind: primitive.kind,
      material: {
        colorLinear: [...primitive.material.colorLinear],
        emission: primitive.material.emission,
        roughness: primitive.material.roughness,
      },
      shell: primitive.tags.includes("shell"),
      openShell,
      sway: primitive.sway,
      coverageBounds: coverageBounds(primitive, coverageCellSize_m),
    });
    primitiveIndexByOwnerId.set(ownerId, primitiveIndex);
    primitiveIndexByMaterialId.set(materialId, primitiveIndex);
  }

  const unsupportedSources: SvoUnsupportedStaticSource[] = [];
  const primitiveCandidates = descriptors.length <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES
    && descriptors.every(({ kind }) => kind !== "terrain-heightfield")
    ? buildSvoPrimitiveCandidates(descriptors as SvoFinitePrimitiveDescriptor[], { skippedOwnerId: openShellOwnerId })
    : undefined;

  const packedRecords = packSvoPrimitiveRecords(descriptors);
  return internSvoPublication(scenePrimitiveCache, cacheKey, {
    environmentId: catalog.environmentId,
    descriptors,
    packedRecords,
    clusterPackings,
    clusterBlocks: clusterPackings.length > 0 ? packSvoDrySceneClusters(clusterPackings) : undefined,
    fieldPrograms,
    fieldProgramBlocks: fieldPrograms.length > 0 ? packSvoDrySceneFieldPrograms(fieldPrograms) : undefined,
    primitiveCandidates,
    metadata,
    primitiveIndexByOwnerId,
    primitiveIndexByMaterialId,
    skipOwnerIds: openShellOwnerId === undefined ? [] : [openShellOwnerId],
    openShellOwnerId,
    unsupportedSources,
    requiresRasterTerrainFallback: unsupportedSources.length > 0,
    analyticTerrain,
    contentRevision,
    cacheKey,
  });
}

/**
 * The contiguous span of primitive records the renderer re-poses each frame.
 *
 * A span rather than a list of indices: authored motion belongs to whole
 * objects, an object's parts are emitted together, and one `writeBuffer` over
 * the span costs less than a scatter even when it carries a few unchanged records
 * along with it. Nothing outside the span is touched, and nothing about the
 * span's identity, material, or dimensions ever changes — only its transform.
 * Sparse maintenance independently consumes the keyed old/new geometry bounds.
 */
export interface SvoScenePrimitiveAnimation {
  /** Index of the first record in the span, in `descriptors` order. */
  readonly firstPrimitiveIndex: number;
  /** Authored rest pose of every record in the span, animated or not. */
  readonly restDescriptors: readonly SvoPrimitiveDescriptor[];
  /** Aligned with `restDescriptors`; undefined leaves that record at rest. */
  readonly sway: readonly (EnvironmentProxySway | undefined)[];
}

/** Undefined when the environment authored no motion, which is the usual case. */
export function svoScenePrimitiveAnimation(build: SvoScenePrimitiveBuild): SvoScenePrimitiveAnimation | undefined {
  const animated = build.metadata.filter(({ sway }) => sway);
  if (animated.length === 0) return undefined;
  const first = Math.min(...animated.map(({ primitiveIndex }) => primitiveIndex));
  const last = Math.max(...animated.map(({ primitiveIndex }) => primitiveIndex));
  return {
    firstPrimitiveIndex: first,
    restDescriptors: build.descriptors.slice(first, last + 1),
    sway: build.metadata.slice(first, last + 1).map(({ sway }) => sway),
  };
}

/** Records for the animated span at one presentation time, ready to upload. */
export function packSvoScenePrimitiveAnimation(animation: SvoScenePrimitiveAnimation, time_s: number): Uint32Array<ArrayBuffer> {
  return packSvoPrimitiveRecords(animation.restDescriptors.map((descriptor, index) => {
    const sway = animation.sway[index];
    return sway ? swayedPrimitiveDescriptor(descriptor, sway, time_s) : descriptor;
  }));
}

/**
 * The build's solids as brick selection sees them: a containing box and an
 * exact distance.
 *
 * The pairing is the point. A brick claim is made from boxes, and a box
 * over-claims — the corners a cone or a floret never enters are still 512
 * voxels of air and a node-mip page of zeros. The exact distance is what turns
 * that box back into the solid, and it is already here: `sampleSvoPrimitive` is
 * the CPU mirror of the same evaluation the voxeliser runs on the device, so
 * the two cannot disagree about which bricks hold geometry.
 *
 * A descriptor this evaluator cannot resolve — a terrain heightfield, whose
 * resolver is not part of the build — reports `-Infinity` rather than throwing,
 * which reads as "reaches everywhere" and keeps every brick it bounds. Being
 * wrong in that direction costs allocation; being wrong in the other costs
 * geometry.
 */
export function svoScenePrimitiveSolidReach(build: SvoScenePrimitiveBuild): SparseSceneSolidReach[] {
  const clusterResolver = (reference: number) => build.clusterPackings[reference];
  return build.metadata.map((entry) => {
    const descriptor = build.descriptors[entry.primitiveIndex];
    const { min, max } = entry.coverageBounds.conservative_m;
    return {
      minimum: [min.x, min.y, min.z] as const,
      maximum: [max.x, max.y, max.z] as const,
      distance_m: (x: number, y: number, z: number): number => {
        try {
          const sample = sampleSvoPrimitive(descriptor, { x, y, z }, undefined, clusterResolver);
          return Number.isFinite(sample.signedDistance_m) ? sample.signedDistance_m : Number.NEGATIVE_INFINITY;
        } catch {
          return Number.NEGATIVE_INFINITY;
        }
      },
    };
  });
}

/** Build the selected scene environment catalog and convert it in one call. */
export function buildSvoScenePrimitives(
  scene: SceneDescription,
  options: SvoScenePrimitiveBuildOptions = {},
): SvoScenePrimitiveBuild {
  const environmentId = options.environmentId ?? scene.environment ?? "default";
  const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
  return svoScenePrimitivesFromEnvironmentCatalog(catalog, {
    ...options,
    // Match the default-camera acceptance audit: features below 1.5 finest
    // cells retain conservative coverage even when cell-centre sampling moves.
    coverageCellSize_m: options.coverageCellSize_m ?? 1.5 * scene.voxelDomain.finestCellSize_m,
  });
}
