import type { SceneDescription } from "./model";
import { VOXEL_MATERIAL_IDS, type LinearRgb } from "./voxel-scene";
import type { SvoVec3 } from "./webgpu-svo-traversal";

/** Version of the binding-free garden terrain material contract. */
export const SVO_TERRAIN_MATERIAL_VERSION = 1;
export const SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES = 16;

/**
 * Stable sub-region identities. The sparse material identity remains terrain
 * (2); these IDs only select the procedural closure within that material.
 */
export const SVO_TERRAIN_REGION_IDS = Object.freeze({
  pondLinerRock: 0,
  pondEdgeSoil: 1,
  grass: 2,
} as const);

export const SVO_TERRAIN_VARIATION_FLAGS = Object.freeze({
  pebble: 1,
  mowStripe: 2,
  clover: 4,
  daisy: 8,
} as const);

/** Exact scene-linear constants from `gardenGroundMaterial` in the raster shader (porcelain-garden monochrome palette). */
export const SVO_GARDEN_TERRAIN_PALETTE = Object.freeze({
  linerDarkLinear: [0.135, 0.13, 0.125] as LinearRgb,
  pebbleLinear: [0.44, 0.435, 0.42] as LinearRgb,
  soilLinear: [0.56, 0.55, 0.52] as LinearRgb,
  grassDarkLinear: [0.46, 0.455, 0.435] as LinearRgb,
  grassLightLinear: [0.66, 0.65, 0.62] as LinearRgb,
  cloverLinear: [0.58, 0.575, 0.55] as LinearRgb,
  daisyLinear: [0.95, 0.94, 0.90] as LinearRgb,
});

export interface SvoTerrainMaterialMetadata {
  baseHeight_m: number;
  waterline_m: number;
  materialId: number;
  policyVersion: number;
}

export interface SvoTerrainRegionWeights {
  pondLinerRock: number;
  pondEdgeSoil: number;
  grass: number;
}

export interface SvoTerrainMaterialSample {
  colorLinear: LinearRgb;
  /** Color before the raster hollow self-occlusion multiplier. */
  unoccludedColorLinear: LinearRgb;
  materialId: number;
  regionId: number;
  regionWeights: SvoTerrainRegionWeights;
  variationFlags: number;
  hollowOcclusion: number;
  /** 0 is level and 1 is vertical. It does not alter raster material ownership. */
  slope: number;
}

export interface SvoTerrainMaterialBuild {
  metadata: SvoTerrainMaterialMetadata;
  packedMetadata: Uint32Array<ArrayBuffer>;
  staticRevision: string;
  cacheKey: string;
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function fract(value: number): number {
  return value - Math.floor(value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** WGSL-compatible smoothstep, including the raster shader's reversed pebble edges. */
function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = clamp01((value - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function mix(left: number, right: number, weight: number): number {
  return left * (1 - weight) + right * weight;
}

function mixRgb(left: LinearRgb, right: LinearRgb, weight: number): LinearRgb {
  return [
    mix(left[0], right[0], weight),
    mix(left[1], right[1], weight),
    mix(left[2], right[2], weight),
  ];
}

/** Exact deterministic `envHash21` CPU mirror used by the raster environment. */
export function svoTerrainVariationHash21(point: readonly [number, number]): number {
  finite(point[0], "Terrain variation X");
  finite(point[1], "Terrain variation Z");
  return fract(Math.sin(point[0] * 127.1 + point[1] * 311.7) * 43758.5453);
}

function floor2(x: number, z: number, scale: number): [number, number] {
  return [Math.floor(x * scale), Math.floor(z * scale)];
}

function add2(value: readonly [number, number], offset: number): [number, number] {
  return [value[0] + offset, value[1] + offset];
}

function categoricalRegion(metadata: SvoTerrainMaterialMetadata, y: number): number {
  // The source is continuously blended. These ownership edges name its exact
  // plateaus: the lower smoothstep edge still belongs to the liner, while the
  // upper lawn edge (where lawn weight is exactly one) belongs to grass.
  if (y >= metadata.baseHeight_m - 0.008) return SVO_TERRAIN_REGION_IDS.grass;
  if (y > metadata.waterline_m - 0.02) return SVO_TERRAIN_REGION_IDS.pondEdgeSoil;
  return SVO_TERRAIN_REGION_IDS.pondLinerRock;
}

/**
 * Exact CPU mirror of the current raster `gardenGroundMaterial` function.
 * Position is world-space. Normal only reports slope for downstream PBR; the
 * raster classifier itself is height-only, so slope never changes ownership.
 */
export function sampleSvoTerrainMaterial(
  metadata: SvoTerrainMaterialMetadata,
  point_m: SvoVec3,
  normal: SvoVec3 = [0, 1, 0],
): SvoTerrainMaterialSample {
  const canonical = canonicalSvoTerrainMaterialMetadata(metadata);
  point_m.forEach((value) => finite(value, "Terrain material position"));
  normal.forEach((value) => finite(value, "Terrain material normal"));
  const normalLength = Math.hypot(...normal);
  if (!(normalLength > 1e-12)) throw new RangeError("Terrain material normal must be non-zero");

  const x = point_m[0];
  const y = point_m[1];
  const z = point_m[2];
  const cell = floor2(x, z, 26);
  const jitterX = svoTerrainVariationHash21(cell) - 0.5;
  const jitterZ = svoTerrainVariationHash21(add2(cell, 19.7)) - 0.5;
  const pebbleX = fract(x * 26) - 0.5 - jitterX * 0.55;
  const pebbleZ = fract(z * 26) - 0.5 - jitterZ * 0.55;
  const pebbleDistance = Math.hypot(pebbleX, pebbleZ);
  const pebbleTone = 0.55 + 0.45 * svoTerrainVariationHash21(add2(cell, 7.3));
  const pebbleWeight = smoothstep(0.44, 0.18, pebbleDistance);
  const pebbleColor = SVO_GARDEN_TERRAIN_PALETTE.pebbleLinear.map(
    (channel) => channel * pebbleTone,
  ) as [number, number, number];
  const liner = mixRgb(SVO_GARDEN_TERRAIN_PALETTE.linerDarkLinear, pebbleColor, pebbleWeight);

  const soilCell = floor2(x, z, 40);
  const soilScale = 0.9 + 0.2 * svoTerrainVariationHash21(soilCell);
  const soil = SVO_GARDEN_TERRAIN_PALETTE.soilLinear.map(
    (channel) => channel * soilScale,
  ) as [number, number, number];

  const stripe = 0.5 + 0.5 * Math.sin((x * 0.9 + z * 0.35) * 4.4);
  const grassCell = floor2(x, z, 90);
  const grassWeight = 0.5 * stripe + 0.5 * svoTerrainVariationHash21(grassCell);
  let grass = mixRgb(SVO_GARDEN_TERRAIN_PALETTE.grassDarkLinear, SVO_GARDEN_TERRAIN_PALETTE.grassLightLinear, grassWeight);
  const clover = svoTerrainVariationHash21(floor2(x, z, 14)) >= 0.962 ? 1 : 0;
  grass = mixRgb(grass, SVO_GARDEN_TERRAIN_PALETTE.cloverLinear, clover * 0.55);
  const daisy = svoTerrainVariationHash21(add2(floor2(x, z, 24), 3.1)) >= 0.986 ? 1 : 0;
  grass = mixRgb(grass, SVO_GARDEN_TERRAIN_PALETTE.daisyLinear, daisy * 0.85);

  const soilBlend = smoothstep(canonical.waterline_m - 0.02, canonical.waterline_m + 0.04, y);
  const lawnBlend = smoothstep(canonical.baseHeight_m - 0.05, canonical.baseHeight_m - 0.008, y);
  const underlay = mixRgb(liner, soil, soilBlend);
  const unoccludedColorLinear = mixRgb(underlay, grass, lawnBlend);
  const hollow = smoothstep(0, Math.max(canonical.baseHeight_m, 1e-3), y);
  const hollowOcclusion = 0.38 + 0.62 * hollow * hollow;
  const colorLinear = unoccludedColorLinear.map(
    (channel) => channel * hollowOcclusion,
  ) as [number, number, number];

  let variationFlags = SVO_TERRAIN_VARIATION_FLAGS.mowStripe;
  if (pebbleWeight >= 0.5) variationFlags |= SVO_TERRAIN_VARIATION_FLAGS.pebble;
  if (clover > 0) variationFlags |= SVO_TERRAIN_VARIATION_FLAGS.clover;
  if (daisy > 0) variationFlags |= SVO_TERRAIN_VARIATION_FLAGS.daisy;
  return {
    colorLinear,
    unoccludedColorLinear,
    materialId: canonical.materialId,
    regionId: categoricalRegion(canonical, y),
    regionWeights: {
      pondLinerRock: (1 - lawnBlend) * (1 - soilBlend),
      pondEdgeSoil: (1 - lawnBlend) * soilBlend,
      grass: lawnBlend,
    },
    variationFlags,
    hollowOcclusion,
    slope: 1 - clamp01(normal[1] / normalLength),
  };
}

export function canonicalSvoTerrainMaterialMetadata(input: SvoTerrainMaterialMetadata): SvoTerrainMaterialMetadata {
  const baseHeight_m = finite(input.baseHeight_m, "Terrain material base height");
  const waterline_m = finite(input.waterline_m, "Terrain material waterline");
  if (baseHeight_m < 0 || waterline_m < 0) throw new RangeError("Terrain material heights must be non-negative");
  if (!Number.isInteger(input.materialId) || input.materialId < 1 || input.materialId > 0xffff) {
    throw new RangeError("Terrain material ID must be a nonzero uint16");
  }
  if (input.policyVersion !== SVO_TERRAIN_MATERIAL_VERSION) {
    throw new RangeError(`Unsupported terrain material policy version ${input.policyVersion}`);
  }
  return { baseHeight_m, waterline_m, materialId: input.materialId, policyVersion: input.policyVersion };
}

/** One host-shareable vec4 containing the only scene-dependent material data. */
export function packSvoTerrainMaterialMetadata(input: SvoTerrainMaterialMetadata): Uint32Array<ArrayBuffer> {
  const metadata = canonicalSvoTerrainMaterialMetadata(input);
  const buffer = new ArrayBuffer(SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  floats[0] = metadata.baseHeight_m;
  floats[1] = metadata.waterline_m;
  words[2] = metadata.materialId;
  words[3] = metadata.policyVersion;
  return words;
}

export function unpackSvoTerrainMaterialMetadata(packed: Uint32Array): SvoTerrainMaterialMetadata {
  if (packed.byteLength !== SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES) {
    throw new RangeError("Packed terrain material metadata must contain exactly one 16-byte record");
  }
  const words = new Uint32Array(packed);
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  return canonicalSvoTerrainMaterialMetadata({
    baseHeight_m: floats[0],
    waterline_m: floats[1],
    materialId: words[2],
    policyVersion: words[3],
  });
}

function fnvStep(hash: number, value: number): number {
  return Math.imul((hash ^ value) >>> 0, 0x01000193) >>> 0;
}

function staticRevision(words: Uint32Array): string {
  let hash = 0x811c9dc5;
  for (const word of words) for (const shift of [0, 8, 16, 24]) hash = fnvStep(hash, (word >>> shift) & 0xff);
  return hash.toString(16).padStart(8, "0");
}

/** Build garden metadata from the same scene fields uploaded to raster uniforms. */
export function buildSvoTerrainMaterial(scene: Pick<SceneDescription, "terrain" | "container">): SvoTerrainMaterialBuild {
  if (!scene.terrain) throw new Error("SVO terrain material requires an authored terrain description");
  const metadata = canonicalSvoTerrainMaterialMetadata({
    baseHeight_m: scene.terrain.baseHeight_m,
    waterline_m: scene.container.height_m * scene.container.fillFraction,
    materialId: VOXEL_MATERIAL_IDS.terrain,
    policyVersion: SVO_TERRAIN_MATERIAL_VERSION,
  });
  const packedMetadata = packSvoTerrainMaterialMetadata(metadata);
  const revision = staticRevision(packedMetadata);
  return {
    metadata,
    packedMetadata,
    staticRevision: revision,
    cacheKey: `svo-terrain-material-v${SVO_TERRAIN_MATERIAL_VERSION}:${revision}`,
  };
}

/** Binding-free GPU mirror of the raster garden terrain material. */
export const svoTerrainMaterialWGSL = /* wgsl */ `
const SVO_TERRAIN_REGION_POND_LINER_ROCK:u32=0u;const SVO_TERRAIN_REGION_POND_EDGE_SOIL:u32=1u;const SVO_TERRAIN_REGION_GRASS:u32=2u;
const SVO_TERRAIN_VARIATION_PEBBLE:u32=1u;const SVO_TERRAIN_VARIATION_MOW_STRIPE:u32=2u;const SVO_TERRAIN_VARIATION_CLOVER:u32=4u;const SVO_TERRAIN_VARIATION_DAISY:u32=8u;
struct SvoTerrainMaterialMetadata{baseHeight_m:f32,waterline_m:f32,materialId:u32,policyVersion:u32}
struct SvoTerrainMaterialSample{colorLinear:vec3f,materialId:u32,unoccludedColorLinear:vec3f,regionId:u32,regionWeights:vec3f,variationFlags:u32,hollowOcclusion:f32,slope:f32,_padding:vec2f}
fn svoTerrainHash21(p:vec2f)->f32{return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453);}
fn svoTerrainCategoricalRegion(metadata:SvoTerrainMaterialMetadata,y:f32)->u32{if(y>=metadata.baseHeight_m-.008){return SVO_TERRAIN_REGION_GRASS;}if(y>metadata.waterline_m-.02){return SVO_TERRAIN_REGION_POND_EDGE_SOIL;}return SVO_TERRAIN_REGION_POND_LINER_ROCK;}
fn svoTerrainMaterial(metadata:SvoTerrainMaterialMetadata,p:vec3f,normalIn:vec3f)->SvoTerrainMaterialSample{
  let cell=floor(p.xz*26.0);let jitter=vec2f(svoTerrainHash21(cell),svoTerrainHash21(cell+19.7))-.5;let pebbleDistance=length(fract(p.xz*26.0)-.5-jitter*.55);let pebbleTone=.55+.45*svoTerrainHash21(cell+7.3);let pebbleWeight=smoothstep(.44,.18,pebbleDistance);let liner=mix(vec3f(.135,.13,.125),vec3f(.44,.435,.42)*pebbleTone,pebbleWeight);
  let soil=vec3f(.56,.55,.52)*(.9+.2*svoTerrainHash21(floor(p.xz*40.0)));let stripe=.5+.5*sin((p.x*.9+p.z*.35)*4.4);var grass=mix(vec3f(.46,.455,.435),vec3f(.66,.65,.62),.5*stripe+.5*svoTerrainHash21(floor(p.xz*90.0)));let clover=step(.962,svoTerrainHash21(floor(p.xz*14.0)));grass=mix(grass,vec3f(.58,.575,.55),clover*.55);let daisy=step(.986,svoTerrainHash21(floor(p.xz*24.0)+3.1));grass=mix(grass,vec3f(.95,.94,.90),daisy*.85);
  let soilBlend=smoothstep(metadata.waterline_m-.02,metadata.waterline_m+.04,p.y);let lawnBlend=smoothstep(metadata.baseHeight_m-.05,metadata.baseHeight_m-.008,p.y);let underlay=mix(liner,soil,soilBlend);let unoccluded=mix(underlay,grass,lawnBlend);let hollow=smoothstep(0.0,max(metadata.baseHeight_m,1e-3),p.y);let occlusion=.38+.62*hollow*hollow;var flags=SVO_TERRAIN_VARIATION_MOW_STRIPE;if(pebbleWeight>=.5){flags|=SVO_TERRAIN_VARIATION_PEBBLE;}if(clover>0.0){flags|=SVO_TERRAIN_VARIATION_CLOVER;}if(daisy>0.0){flags|=SVO_TERRAIN_VARIATION_DAISY;}let normal=normalize(normalIn);let weights=vec3f((1.0-lawnBlend)*(1.0-soilBlend),(1.0-lawnBlend)*soilBlend,lawnBlend);
  return SvoTerrainMaterialSample(unoccluded*occlusion,metadata.materialId,unoccluded,svoTerrainCategoricalRegion(metadata,p.y),weights,flags,occlusion,1.0-clamp(normal.y,0.0,1.0),vec2f(0.0));
}
`;
