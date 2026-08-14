import {
  SVO_PORCELAIN_TERRAIN_BASE_COLOR_LINEAR,
  SVO_PORCELAIN_TERRAIN_ROUGHNESS,
  type SvoTerrainSurfaceModel,
} from "./svo-terrain-material";
import { GLASS_OPTICS, WATER_OPTICS, type LinearRgb } from "../core/webgpu-lighting";
import type { SceneryySurface } from "../core/scenery-graph";
import type { EnvironmentProxyMaterial, EnvironmentProxyPrimitive } from "../core/voxel-environments";
import { VOXEL_MATERIAL_IDS, VOXEL_MATERIALS, type VoxelMaterial } from "../core/voxel-scene";

/** Six host-shareable 16-byte lanes. Stable material IDs remain direct indices. */
export const SVO_MATERIAL_RECORD_STRIDE_BYTES = 96;
export const SVO_MATERIAL_RECORD_WORDS = SVO_MATERIAL_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;

export const SVO_MATERIAL_FLAGS = Object.freeze({
  opaque: 1 << 0,
  dielectric: 1 << 1,
  thinWall: 1 << 2,
} as const);

/**
 * Stable ABI values; append only, because a published material record carries
 * the integer and a cached publication outlives this file.
 *
 * `gardenTerrain` is the odd one here: it is the lawn closure, chosen by
 * `svoMaterialFromVoxelMaterial` from the scene's terrain shell, and no
 * environment proxy can ever select it — a tide line belongs to ground, not to
 * a bench. `plaster` is chosen from the same shell and *also* by every proxy in
 * a porcelain scene, which is the point of that model: one fired surface, so one
 * closure. Everything else is picked semantically by
 * `svoMaterialFunctionIdForEnvironmentProxy` below.
 */
export const SVO_MATERIAL_FUNCTION_IDS = Object.freeze({
  none: 0,
  gardenTerrain: 1,
  architecturalSurface: 2,
  wood: 3,
  stone: 4,
  foliage: 5,
  ceramic: 6,
  brushedMetal: 7,
  organic: 8,
  plaster: 9,
} as const);

/**
 * What each authored surface name resolves to.
 *
 * A total `Record`, so a name added to `SCENERY_SURFACE_IDS` does not compile
 * until it has a closure here. That is the whole reason the two lists are
 * allowed to live in different files: the scene schema cannot import the
 * material ABI without a cycle, and this table is what stops them drifting.
 *
 * `architectural` is spelled without its `-surface` suffix on the authoring
 * side because a scene author names a *kind of surface*, not a function.
 */
export const SVO_MATERIAL_FUNCTION_ID_BY_SURFACE: Readonly<Record<SceneryySurface, number>> = Object.freeze({
  none: SVO_MATERIAL_FUNCTION_IDS.none,
  architectural: SVO_MATERIAL_FUNCTION_IDS.architecturalSurface,
  wood: SVO_MATERIAL_FUNCTION_IDS.wood,
  stone: SVO_MATERIAL_FUNCTION_IDS.stone,
  foliage: SVO_MATERIAL_FUNCTION_IDS.foliage,
  ceramic: SVO_MATERIAL_FUNCTION_IDS.ceramic,
  "brushed-metal": SVO_MATERIAL_FUNCTION_IDS.brushedMetal,
  organic: SVO_MATERIAL_FUNCTION_IDS.organic,
  plaster: SVO_MATERIAL_FUNCTION_IDS.plaster,
});

/**
 * Stable semantic selection; it depends only on authored material, group and
 * tags, never on publication order.
 *
 * An authored `material.surface` wins outright. Everything below it is the
 * legacy inference — a regular expression over the object's own name — kept
 * for the sets that have not declared one. It is worth being explicit about
 * why that is a fallback and not the design: under it, a generator could not
 * rename its own parts without restyling them, which is why
 * `lib/voxel-scenery/stone-set.ts` still carries the comment "Nothing here may
 * be named 'mushroom'". A node that says what it is made of has no such
 * coupling, and the `shell` tag stays authoritative either way because a wall
 * is architectural by virtue of being a wall.
 *
 * `surfaceModel` sits above all of that, and it is the scene's word rather than
 * the object's. A porcelain scene is one in which every surface is the *same*
 * fired white material — the ground, the rim it is set into, the boulders bedded
 * on it, the tree standing in it — so the granite speckle a boulder would
 * otherwise inherit from its own name is not a refinement of that set, it is a
 * contradiction of it. Rather than teach every generator to author
 * `surface: "plaster"` (and to keep authoring it), the scene says once what it
 * is made of, in the same place it already says what its ground is made of:
 * `{ kind: "terrain-shell", materialModel: "porcelain" }`. The default is the
 * garden model, so a scene that says nothing resolves exactly as before.
 *
 * Note that this selects a *closure*, never a colour. A porcelain scene's one
 * chromatic object — the hero garden's slate-teal hose — keeps its authored
 * albedo and merely stops being grained, which is the whole of what was wanted.
 */
export function svoMaterialFunctionIdForEnvironmentProxy(
  primitive: Pick<EnvironmentProxyPrimitive, "group" | "tags"> & {
    readonly material?: Pick<EnvironmentProxyMaterial, "surface">;
  },
  surfaceModel: SvoTerrainSurfaceModel = "garden-terrain",
): number {
  if (surfaceModel === "porcelain") return SVO_MATERIAL_FUNCTION_IDS.plaster;
  const surface = primitive.material?.surface;
  if (surface !== undefined) return SVO_MATERIAL_FUNCTION_ID_BY_SURFACE[surface];
  const semantic = `${primitive.group} ${primitive.tags.join(" ")}`;
  if (primitive.tags.includes("shell")) return SVO_MATERIAL_FUNCTION_IDS.architecturalSurface;
  if (/leaf|foliage|hedge|flower|fruit|canopy|lilypad|reed/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.foliage;
  if (/wood|cedar|bench|stool|bucket|tree|trunk|duckboard|batten/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.wood;
  if (/stone|column|plinth|pebble|limestone|coping|parapet|curb|kerb|monolith/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.stone;
  if (/pot|planter|ceramic|clay|tile|porcelain|bridge/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.ceramic;
  if (/steel|metal|pipe|frame|fixture|instrument|console|monitor|hull|rib|stringer|coaming|gauge|gantry|mast|rack|conduit|bollard|dewar|equipment-case/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.brushedMetal;
  if (/mushroom|organic|hose|rope|cable|cord|floor-mat|linen|towel|cloth/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.organic;
  // Flat painted joinery: counters, boards, targets and panels are sprayed or
  // laminated, so they take the same fine architectural grain as a wall rather
  // than a material grain of their own.
  if (/counter|board|panel|target/.test(semantic)) return SVO_MATERIAL_FUNCTION_IDS.architecturalSurface;
  // Everything still here is deliberately flat: calibration wedges (whose
  // whole job is unmodulated albedo), softboxes, glass, and the
  // void behind a porthole. Procedural grain on any of those would be a lie.
  return SVO_MATERIAL_FUNCTION_IDS.none;
}

export interface SvoMaterialRecord {
  materialId: number;
  revision: number;
  materialFunctionId: number;
  flags: number;
  baseColorLinear: LinearRgb;
  opacity: number;
  emissiveLinear: LinearRgb;
  roughness: number;
  metallic: number;
  specularWeight: number;
  indexOfRefraction: number;
  transmission: number;
  absorption_mInv: LinearRgb;
  scattering_mInv: number;
  scatteringColorLinear: LinearRgb;
  scatteringAnisotropy: number;
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function unit(value: number, label: string): number {
  const result = finite(value, label);
  if (result < 0 || result > 1) throw new RangeError(`${label} must be from zero to one`);
  return result;
}

function nonNegative(value: number, label: string): number {
  const result = finite(value, label);
  if (result < 0) throw new RangeError(`${label} must be non-negative`);
  return result;
}

function color(value: LinearRgb, label: string, bounded = true): [number, number, number] {
  if (value.length !== 3) throw new RangeError(`${label} must contain three channels`);
  return value.map((channel, index) => bounded
    ? unit(channel, `${label}[${index}]`)
    : nonNegative(channel, `${label}[${index}]`)) as [number, number, number];
}

export function canonicalSvoMaterialRecord(input: SvoMaterialRecord): SvoMaterialRecord {
  if (!Number.isSafeInteger(input.materialId) || input.materialId < 0 || input.materialId > 0xffff) {
    throw new RangeError("SVO material ID must be an unsigned 16-bit integer");
  }
  for (const [value, label] of [
    [input.revision, "revision"],
    [input.materialFunctionId, "material function ID"],
    [input.flags, "flags"],
  ] as const) if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`SVO material ${label} must be an unsigned 32-bit integer`);
  }
  const ior = finite(input.indexOfRefraction, "SVO material IOR");
  if (ior < 1 || ior > 4) throw new RangeError("SVO material IOR must be from one to four");
  const anisotropy = finite(input.scatteringAnisotropy, "SVO material scattering anisotropy");
  if (anisotropy <= -1 || anisotropy >= 1) throw new RangeError("SVO material scattering anisotropy must be strictly between -1 and 1");
  return Object.freeze({
    ...input,
    baseColorLinear: color(input.baseColorLinear, "SVO material base color"),
    opacity: unit(input.opacity, "SVO material opacity"),
    emissiveLinear: color(input.emissiveLinear, "SVO material emission", false),
    roughness: Math.max(0.04, unit(input.roughness, "SVO material roughness")),
    metallic: unit(input.metallic, "SVO material metallic"),
    specularWeight: unit(input.specularWeight, "SVO material specular weight"),
    indexOfRefraction: ior,
    transmission: unit(input.transmission, "SVO material transmission"),
    absorption_mInv: color(input.absorption_mInv, "SVO material absorption", false),
    scattering_mInv: nonNegative(input.scattering_mInv, "SVO material scattering"),
    scatteringColorLinear: color(input.scatteringColorLinear, "SVO material scattering color"),
    scatteringAnisotropy: anisotropy,
  });
}

/**
 * Which closure the terrain material record selects.
 *
 * The default is the garden lawn. `porcelain` swaps it for the fired-plaster
 * grain and a white albedo — the same override the sculpted-vessel scenes want,
 * expressed once here rather than at each of the two sites that assemble a
 * material table.
 *
 * It used to swap the lawn for the *flat* function, which is why the hero pond
 * shipped with no surface at all: the vessel is most of that frame, and one
 * albedo across it reads as untextured geometry rather than as fired clay.
 */
export interface SvoVoxelMaterialOptions {
  readonly terrainSurface?: "garden-terrain" | "porcelain";
}

export function svoMaterialFromVoxelMaterial(
  material: VoxelMaterial,
  revision = 1,
  options: SvoVoxelMaterialOptions = {},
): SvoMaterialRecord {
  const isWater = material.id === VOXEL_MATERIAL_IDS.fluid;
  const isThinGlass = material.closure === "thin-dielectric";
  const isTerrain = material.id === VOXEL_MATERIAL_IDS.terrain;
  const isPorcelainTerrain = isTerrain && options.terrainSurface === "porcelain";
  return canonicalSvoMaterialRecord({
    materialId: material.id,
    revision,
    // Three arms, not two, because terrain is the only built-in material with a
    // choice of closure and porcelain is a different closure rather than the
    // lack of one. The lawn's height-banded liner/soil/grass classifier would be
    // a lie on a fired vessel — but so was the flat function this used to fall
    // through to, which is why a scene whose whole subject is a white basin came
    // out with an unmodulated basin.
    materialFunctionId: !isTerrain
      ? SVO_MATERIAL_FUNCTION_IDS.none
      : isPorcelainTerrain
        ? SVO_MATERIAL_FUNCTION_IDS.plaster
        : SVO_MATERIAL_FUNCTION_IDS.gardenTerrain,
    flags: material.closure === "opaque"
      ? SVO_MATERIAL_FLAGS.opaque
      : SVO_MATERIAL_FLAGS.dielectric | (isThinGlass ? SVO_MATERIAL_FLAGS.thinWall : 0),
    baseColorLinear: isPorcelainTerrain ? SVO_PORCELAIN_TERRAIN_BASE_COLOR_LINEAR : material.baseColorLinear,
    opacity: isThinGlass ? 0.24 : 1,
    emissiveLinear: material.emissiveLinear,
    roughness: isPorcelainTerrain ? SVO_PORCELAIN_TERRAIN_ROUGHNESS : material.roughness,
    metallic: material.metallic,
    specularWeight: 1,
    indexOfRefraction: material.ior,
    transmission: material.transmission,
    absorption_mInv: isWater ? WATER_OPTICS.absorption : [0, 0, 0],
    scattering_mInv: isWater ? Math.max(...WATER_OPTICS.scatter) : 0,
    scatteringColorLinear: isWater ? WATER_OPTICS.scatter : (isThinGlass ? GLASS_OPTICS.tint : [0, 0, 0]),
    scatteringAnisotropy: 0,
  });
}

/**
 * Convert the canonical authored-environment material into the same direct-index
 * PBR closure used by the production dry renderer. Environment IDs deliberately
 * remain assigned by the caller because they share the sparse scene's stable
 * owner-index convention rather than the built-in voxel-material enum.
 */
export function svoMaterialFromEnvironmentProxyMaterial(
  materialId: number,
  material: EnvironmentProxyMaterial,
  revision = 1,
  materialFunctionId: number = SVO_MATERIAL_FUNCTION_IDS.none,
): SvoMaterialRecord {
  return canonicalSvoMaterialRecord({
    materialId,
    revision,
    materialFunctionId,
    flags: SVO_MATERIAL_FLAGS.opaque,
    baseColorLinear: material.colorLinear,
    opacity: 1,
    emissiveLinear: material.colorLinear.map((channel) => channel * material.emission) as [number, number, number],
    roughness: material.roughness,
    metallic: 0,
    specularWeight: 1,
    indexOfRefraction: 1.5,
    transmission: 0,
    absorption_mInv: [0, 0, 0],
    scattering_mInv: 0,
    scatteringColorLinear: [0, 0, 0],
    scatteringAnisotropy: 0,
  });
}

export function buildDefaultSvoMaterialRecords(
  revision = 1,
  options: SvoVoxelMaterialOptions = {},
): readonly SvoMaterialRecord[] {
  return VOXEL_MATERIALS.map((material) => svoMaterialFromVoxelMaterial(material, revision, options));
}

/** Dense direct-index table. Slot zero is empty and unassigned slots are inert. */
export function packSvoMaterialTable(records: readonly SvoMaterialRecord[]): Uint32Array<ArrayBuffer> {
  const canonical = records.map(canonicalSvoMaterialRecord);
  const ids = new Set<number>();
  let maximumId = 0;
  for (const record of canonical) {
    if (record.materialId === 0) throw new RangeError("SVO material slot zero is reserved for empty space");
    if (ids.has(record.materialId)) throw new RangeError(`Duplicate SVO material ID ${record.materialId}`);
    ids.add(record.materialId);
    maximumId = Math.max(maximumId, record.materialId);
  }
  const buffer = new ArrayBuffer((maximumId + 1) * SVO_MATERIAL_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  for (const record of canonical) {
    const offset = record.materialId * SVO_MATERIAL_RECORD_WORDS;
    floats.set([...record.baseColorLinear, record.opacity], offset);
    floats.set([...record.emissiveLinear, record.roughness], offset + 4);
    floats.set([record.metallic, record.specularWeight, record.indexOfRefraction, record.transmission], offset + 8);
    floats.set([...record.absorption_mInv, record.scattering_mInv], offset + 12);
    floats.set([...record.scatteringColorLinear, record.scatteringAnisotropy], offset + 16);
    words.set([record.materialId, record.revision, record.materialFunctionId, record.flags], offset + 20);
  }
  return words;
}

export function unpackSvoMaterialRecord(table: Uint32Array, materialId: number): SvoMaterialRecord {
  if (!Number.isSafeInteger(materialId) || materialId < 0) throw new RangeError("SVO material index must be non-negative");
  const offset = materialId * SVO_MATERIAL_RECORD_WORDS;
  if (offset + SVO_MATERIAL_RECORD_WORDS > table.length) throw new RangeError("SVO material index exceeds the packed table");
  const floats = new Float32Array(table.buffer, table.byteOffset, table.byteLength / 4);
  return canonicalSvoMaterialRecord({
    baseColorLinear: [floats[offset], floats[offset + 1], floats[offset + 2]], opacity: floats[offset + 3],
    emissiveLinear: [floats[offset + 4], floats[offset + 5], floats[offset + 6]], roughness: floats[offset + 7],
    metallic: floats[offset + 8], specularWeight: floats[offset + 9], indexOfRefraction: floats[offset + 10], transmission: floats[offset + 11],
    absorption_mInv: [floats[offset + 12], floats[offset + 13], floats[offset + 14]], scattering_mInv: floats[offset + 15],
    scatteringColorLinear: [floats[offset + 16], floats[offset + 17], floats[offset + 18]], scatteringAnisotropy: floats[offset + 19],
    materialId: table[offset + 20], revision: table[offset + 21], materialFunctionId: table[offset + 22], flags: table[offset + 23],
  });
}

/**
 * The shader's copy of the function IDs, generated rather than transcribed.
 *
 * Hand-written, it was a second table that had to be remembered: the porcelain
 * ground shipped for a release with no grain at all partly because adding an ID
 * meant editing two lists and a policy row, and it is exactly the sort of edit
 * that gets two of three. `camelCase` maps to `SCREAMING_SNAKE` verbatim, so
 * the emitted text is byte-identical to the block this replaced.
 */
const wgslMaterialFunctionConstants = Object.entries(SVO_MATERIAL_FUNCTION_IDS)
  .map(([key, id]) => `const SVO_MATERIAL_FUNCTION_${key.replace(/[A-Z]/g, (letter) => `_${letter}`).toUpperCase()}:u32=${id}u;`)
  .join("\n");

export const svoMaterialWGSL = /* wgsl */ `
struct SvoMaterialRecord {
  baseColorOpacity:vec4f,
  emissiveRoughness:vec4f,
  surface:vec4f,
  absorptionScattering:vec4f,
  scatteringColorAnisotropy:vec4f,
  identity:vec4u,
}
const SVO_MATERIAL_FLAG_OPAQUE:u32=1u;
const SVO_MATERIAL_FLAG_DIELECTRIC:u32=2u;
const SVO_MATERIAL_FLAG_THIN_WALL:u32=4u;
${wgslMaterialFunctionConstants}
fn svoMaterialValid(material:SvoMaterialRecord,index:u32)->bool{
  return material.identity.x==index&&index!=0u&&material.identity.w!=0u;
}
fn svoMaterialDielectricF0(material:SvoMaterialRecord)->f32{
  let ior=clamp(material.surface.z,1.0,4.0);let ratio=(ior-1.0)/(ior+1.0);return ratio*ratio;
}
`;
