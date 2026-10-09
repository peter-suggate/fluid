import type { SceneDescription } from "../../../core/model";
import { walkSceneryNodes } from "../../../core/scenery-graph";
import { terrainHeightAt, terrainWorldContainer } from "../../../core/terrain";
import { VOXEL_MATERIAL_IDS, type LinearRgb } from "../../../core/voxel-scene";
import type { SvoVec3 } from "../primary-visibility/webgpu-svo-traversal";

/**
 * Which closure the ground wears.
 *
 * `garden-terrain` is everything below in this file. `porcelain` is not its
 * absence but its alternative: the terrain material record carries the
 * `plaster` procedural function instead, no metadata is published, and the
 * shader's terrain-policy branch is therefore never taken — that branch tests
 * for the garden function specifically, so the two closures cannot both run and
 * the ground can never be double-varied.
 *
 * It *was* an absence, and the flat function ID it selected is why the hero
 * pond rendered with no surface: a vessel filling most of the frame at one
 * unmodulated albedo. Not publishing metadata is the part worth keeping; not
 * publishing a material was the mistake.
 *
 * Selecting it is a scene decision, made once by the terrain shell node, so the
 * ground the water rests in and the ground the camera sees cannot disagree
 * about which one it is.
 */
export type SvoTerrainSurfaceModel = "garden-terrain" | "porcelain";

/**
 * Unglazed white porcelain, in scene-linear.
 *
 * It was 0.80, held below paper white on the argument that a ground plane fills
 * most of the frame and an albedo that high turns every bounce into a second
 * light source. That argument is correct and it is describing the *reference*:
 * a high-key porcelain garden is exactly a set in which the floor is a second
 * light source, and holding the largest surface an eighth of a stop under the
 * props standing on it is what made the frame read as concrete rather than as
 * fired clay. The props are authored between 0.62 and 0.92 on their palettes, so
 * 0.90 puts the ground at the top of the set's own band instead of below it.
 *
 * Still faintly warm — the 0.015 spread between the channels is what separates
 * fired clay from a render's default grey, and it survives at any level.
 *
 * Both are the *base* the `plaster` procedural policy modulates, not the final
 * surface. Since that policy carries no colour amplitude, the albedo here is the
 * albedo shaded: the only thing the closure perturbs is the sheen.
 */
export const SVO_PORCELAIN_TERRAIN_BASE_COLOR_LINEAR: LinearRgb = [0.90, 0.891, 0.874];
export const SVO_PORCELAIN_TERRAIN_ROUGHNESS = 0.62;

/** The ground's closure, as declared by the scene's terrain shell. */
export function sceneTerrainSurfaceModel(
  scene: Pick<SceneDescription, "scenery">,
): SvoTerrainSurfaceModel {
  for (const { node } of walkSceneryNodes(scene.scenery?.nodes ?? [])) {
    if (node.kind === "terrain-shell") return node.materialModel === "porcelain" ? "porcelain" : "garden-terrain";
  }
  return "garden-terrain";
}

/**
 * The ground beyond the set, as the dry-scene shader needs it.
 *
 * A terrain shell is a voxelised slab over the container's footprint, and past
 * its edge there used to be nothing but the sky gradient: the set read as a
 * tile floating in a studio sweep, its side walls in plain view. The plane is a
 * shading-time answer to that — one ray/plane intersection per pixel, no
 * geometry, no G-buffer surface, no visibility — so it costs the same at any
 * distance and cannot drift from a camera that moves every frame.
 *
 * `height_m` is where the slab's top meets its edge, so the two surfaces are
 * flush at the seam. `footprint_m` is the slab's plan AABB (minX, minZ, maxX,
 * maxZ); the plane only overrides a voxel hit when it crosses outside it, which
 * is what hides the side walls without touching the pond. `hazeDistance_m` is
 * the e-folding distance of the fade into the horizon, measured from the
 * footprint's edge and stated in footprints rather than metres so a set scaled
 * up keeps the same atmosphere.
 */
export interface SvoGroundPlane {
  height_m: number;
  footprint_m: readonly [number, number, number, number];
  hazeDistance_m: number;
  /**
   * For a scene with `scene.backdrop`: the footprint distance out to which the
   * renderer traces the backdrop ground (`svoBackdropContentRadius`). Positive,
   * it retires the plane: the ground is the tiled terrain walk
   * (`backdrop-terrain-tiles.ts`), terrain beyond the footprint takes the
   * haze, and a ray that misses the ground below the horizon sees the horizon
   * colour. Absent on every other scene, which keeps the plane exactly as it was.
   */
  backdropContentRadius_m?: number;
}

/** Haze e-folding distance, in multiples of the footprint's larger plan extent. */
export const SVO_GROUND_PLANE_HAZE_FOOTPRINTS = 1.25;

/**
 * Haze transmittance at which the backdrop ground stops being traced.
 *
 * The hills fade into the horizon colour as `1 - exp(-d / hazeDistance)`, with
 * d the distance from the footprint. Past the distance where the transmittance
 * drops below 1/256, the ground is the horizon colour to within an 8-bit step,
 * so a ray that has not met it by then is drawn as the horizon. The terrain
 * tiles reach exactly that far (`backdropTerrainReach`); the outer radius only
 * ends the valley's rise, never the ground.
 */
export const SVO_BACKDROP_HAZE_CUTOFF_TRANSMITTANCE = 1 / 256;

/** Footprint distance out to which the backdrop ground is traced: where its haze leaves 1/256 of it. */
export function svoBackdropContentRadius(hazeDistance_m: number): number {
  if (!(hazeDistance_m > 0)) throw new RangeError("Backdrop terrain reach needs a positive haze distance");
  return hazeDistance_m * Math.log(1 / SVO_BACKDROP_HAZE_CUTOFF_TRANSMITTANCE);
}
const GROUND_PLANE_PERIMETER_SAMPLES_PER_SIDE = 32;

/**
 * The analytic ground plane for a garden scene with a terrain shell, or undefined.
 *
 * The height is the median of the slab's own top sampled round the footprint's
 * perimeter — the same clamp `buildSvoRenderTerrainFieldSteps` applies, at the
 * same container-centred coordinates — rather than `baseHeight_m`, because a
 * sculpted or generated ground supersedes that number and only the perimeter
 * decides where the seam is. The median rather than the mean so that a mound or
 * a pond that reaches the edge in one place does not lift or sink the whole
 * horizon.
 */
export function sceneSvoGroundPlane(
  scene: Pick<SceneDescription, "scenery" | "terrain" | "container" | "environment"> & Partial<Pick<SceneDescription, "backdrop">>,
): SvoGroundPlane | undefined {
  // The open-air garden is the one set whose ground is meant to run to the
  // horizon. A sculpted slab elsewhere (the hillside dam) has an uneven edge
  // that a single level plane would half-bury.
  if (scene.environment !== "garden") return undefined;
  const terrain = scene.terrain;
  if (!terrain) return undefined;
  let shell = false;
  for (const { node } of walkSceneryNodes(scene.scenery?.nodes ?? [])) if (node.kind === "terrain-shell") { shell = true; break; }
  if (!shell) return undefined;
  const { width_m, depth_m, height_m } = terrainWorldContainer(scene);
  const minX = -0.5 * width_m, minZ = -0.5 * depth_m, maxX = 0.5 * width_m, maxZ = 0.5 * depth_m;
  const heights: number[] = [];
  const top = (x: number, z: number) => heights.push(Math.min(height_m, Math.max(0, terrainHeightAt(terrain, x, z))));
  // Half a sample in from every corner, so no sample sits exactly on an edge the
  // voxeliser's cell centres never reach.
  for (let index = 0; index < GROUND_PLANE_PERIMETER_SAMPLES_PER_SIDE; index += 1) {
    const f = (index + 0.5) / GROUND_PLANE_PERIMETER_SAMPLES_PER_SIDE;
    const x = minX + f * width_m, z = minZ + f * depth_m;
    top(x, minZ); top(x, maxZ); top(minX, z); top(maxX, z);
  }
  heights.sort((a, b) => a - b);
  const middle = heights.length >> 1;
  const plane_m = heights.length % 2 === 0 ? 0.5 * (heights[middle - 1] + heights[middle]) : heights[middle];
  if (!Number.isFinite(plane_m)) throw new RangeError("Terrain shell ground-plane height is not finite");
  // A backdrop owns the atmosphere: its ground and the horizon past it fade
  // with one haze, so the horizon is continuous where the traced ground ends.
  const hazeDistance_m = scene.backdrop
    ? scene.backdrop.hazeDistance_m
    : SVO_GROUND_PLANE_HAZE_FOOTPRINTS * Math.max(width_m, depth_m);
  if (!(hazeDistance_m > 0)) throw new RangeError("Ground-plane haze distance must be positive");
  return {
    height_m: plane_m,
    footprint_m: [minX, minZ, maxX, maxZ],
    hazeDistance_m,
    ...(scene.backdrop ? { backdropContentRadius_m: svoBackdropContentRadius(hazeDistance_m) } : {}),
  };
}

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
  contentRevision: string;
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

function contentRevision(words: Uint32Array): string {
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
  const revision = contentRevision(packedMetadata);
  return {
    metadata,
    packedMetadata,
    contentRevision: revision,
    cacheKey: `svo-terrain-material-v${SVO_TERRAIN_MATERIAL_VERSION}:${revision}`,
  };
}

/**
 * The per-pixel WGSL colour policy that used to live here is deleted.
 *
 * It shaded the ground from its *world position* — the purest form of the
 * analytic dependency the render path no longer has. Surface colour comes from
 * the published PBR material record by index now. What survives is the CPU side:
 * `sceneTerrainSurfaceModel` still decides which ground closure a scene has, and
 * `buildSvoTerrainMaterial` still packs the metadata the uniform reserves space
 * for.
 */
