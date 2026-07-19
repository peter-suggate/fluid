import { GLASS_OPTICS, WATER_OPTICS, type LinearRgb } from "./webgpu-lighting";
import type { EnvironmentProxyMaterial } from "./voxel-environments";
import { VOXEL_MATERIAL_IDS, VOXEL_MATERIALS, type VoxelMaterial } from "./voxel-scene";

/** Six host-shareable 16-byte lanes. Stable material IDs remain direct indices. */
export const SVO_MATERIAL_RECORD_STRIDE_BYTES = 96;
export const SVO_MATERIAL_RECORD_WORDS = SVO_MATERIAL_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;

export const SVO_MATERIAL_FLAGS = Object.freeze({
  opaque: 1 << 0,
  dielectric: 1 << 1,
  thinWall: 1 << 2,
} as const);

export const SVO_MATERIAL_FUNCTION_IDS = Object.freeze({
  none: 0,
  gardenTerrain: 1,
} as const);

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

export function svoMaterialFromVoxelMaterial(material: VoxelMaterial, revision = 1): SvoMaterialRecord {
  const isWater = material.id === VOXEL_MATERIAL_IDS.fluid;
  const isThinGlass = material.closure === "thin-dielectric";
  return canonicalSvoMaterialRecord({
    materialId: material.id,
    revision,
    materialFunctionId: material.id === VOXEL_MATERIAL_IDS.terrain
      ? SVO_MATERIAL_FUNCTION_IDS.gardenTerrain
      : SVO_MATERIAL_FUNCTION_IDS.none,
    flags: material.closure === "opaque"
      ? SVO_MATERIAL_FLAGS.opaque
      : SVO_MATERIAL_FLAGS.dielectric | (isThinGlass ? SVO_MATERIAL_FLAGS.thinWall : 0),
    baseColorLinear: material.baseColorLinear,
    opacity: isThinGlass ? 0.24 : 1,
    emissiveLinear: material.emissiveLinear,
    roughness: material.roughness,
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
): SvoMaterialRecord {
  return canonicalSvoMaterialRecord({
    materialId,
    revision,
    materialFunctionId: SVO_MATERIAL_FUNCTION_IDS.none,
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

export function buildDefaultSvoMaterialRecords(revision = 1): readonly SvoMaterialRecord[] {
  return VOXEL_MATERIALS.map((material) => svoMaterialFromVoxelMaterial(material, revision));
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
const SVO_MATERIAL_FUNCTION_NONE:u32=0u;
const SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN:u32=1u;
fn svoMaterialValid(material:SvoMaterialRecord,index:u32)->bool{
  return material.identity.x==index&&index!=0u&&material.identity.w!=0u;
}
fn svoMaterialDielectricF0(material:SvoMaterialRecord)->f32{
  let ior=clamp(material.surface.z,1.0,4.0);let ratio=(ior-1.0)/(ior+1.0);return ratio*ratio;
}
`;
