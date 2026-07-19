import assert from "node:assert/strict";
import test from "node:test";

import { GARDEN_GRASS_M, GARDEN_WATERLINE_M, gardenPoolTerrain } from "../lib/garden-scene";
import { cloneScene, defaultScene } from "../lib/model";
import {
  buildSvoTerrainMaterial,
  packSvoTerrainMaterialMetadata,
  sampleSvoTerrainMaterial,
  SVO_GARDEN_TERRAIN_PALETTE,
  SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES,
  SVO_TERRAIN_REGION_IDS,
  SVO_TERRAIN_VARIATION_FLAGS,
  svoTerrainMaterialWGSL,
  svoTerrainVariationHash21,
  unpackSvoTerrainMaterialMetadata,
  type SvoTerrainMaterialMetadata,
} from "../lib/svo-terrain-material";
import { VOXEL_MATERIAL_IDS, voxelMaterial } from "../lib/voxel-scene";

const metadata: SvoTerrainMaterialMetadata = {
  baseHeight_m: GARDEN_GRASS_M,
  waterline_m: GARDEN_WATERLINE_M,
  materialId: VOXEL_MATERIAL_IDS.terrain,
  policyVersion: 1,
};

function fract(value: number): number { return value - Math.floor(value); }
function clamp01(value: number): number { return Math.max(0, Math.min(1, value)); }
function rasterSmoothstep(a: number, b: number, x: number): number { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); }
function rasterHash(x: number, z: number): number { return fract(Math.sin(x * 127.1 + z * 311.7) * 43758.5453); }
function rasterMix(a: readonly number[], b: readonly number[], t: number): [number, number, number] {
  return [0, 1, 2].map((i) => a[i] * (1 - t) + b[i] * t) as [number, number, number];
}

/** Independent literal transcription of raster `gardenGroundMaterial`. */
function rasterGardenGroundMaterial(p: readonly [number, number, number]): [number, number, number] {
  const [x, y, z] = p;
  const cell = [Math.floor(x * 26), Math.floor(z * 26)] as const;
  const jitter = [rasterHash(cell[0], cell[1]) - 0.5, rasterHash(cell[0] + 19.7, cell[1] + 19.7) - 0.5];
  const pebbleDistance = Math.hypot(fract(x * 26) - 0.5 - jitter[0] * 0.55, fract(z * 26) - 0.5 - jitter[1] * 0.55);
  const pebbleTone = 0.55 + 0.45 * rasterHash(cell[0] + 7.3, cell[1] + 7.3);
  const liner = rasterMix([0.135, 0.13, 0.125], [0.44 * pebbleTone, 0.435 * pebbleTone, 0.42 * pebbleTone], rasterSmoothstep(0.44, 0.18, pebbleDistance));
  const soilCell = [Math.floor(x * 40), Math.floor(z * 40)] as const;
  const soilScale = 0.9 + 0.2 * rasterHash(soilCell[0], soilCell[1]);
  const soil = [0.56 * soilScale, 0.55 * soilScale, 0.52 * soilScale];
  const stripe = 0.5 + 0.5 * Math.sin((x * 0.9 + z * 0.35) * 4.4);
  const grassCell = [Math.floor(x * 90), Math.floor(z * 90)] as const;
  let grass = rasterMix([0.46, 0.455, 0.435], [0.66, 0.65, 0.62], 0.5 * stripe + 0.5 * rasterHash(grassCell[0], grassCell[1]));
  const cloverCell = [Math.floor(x * 14), Math.floor(z * 14)] as const;
  const clover = rasterHash(cloverCell[0], cloverCell[1]) >= 0.962 ? 1 : 0;
  grass = rasterMix(grass, [0.58, 0.575, 0.55], clover * 0.55);
  const daisyCell = [Math.floor(x * 24), Math.floor(z * 24)] as const;
  const daisy = rasterHash(daisyCell[0] + 3.1, daisyCell[1] + 3.1) >= 0.986 ? 1 : 0;
  grass = rasterMix(grass, [0.95, 0.94, 0.90], daisy * 0.85);
  const soilBand = rasterSmoothstep(metadata.waterline_m - 0.02, metadata.waterline_m + 0.04, y);
  let color = rasterMix(liner, soil, soilBand);
  color = rasterMix(color, grass, rasterSmoothstep(metadata.baseHeight_m - 0.05, metadata.baseHeight_m - 0.008, y));
  const hollow = rasterSmoothstep(0, Math.max(metadata.baseHeight_m, 1e-3), y);
  return color.map((channel) => channel * (0.38 + 0.62 * hollow * hollow)) as [number, number, number];
}

test("packed metadata is minimal, stable, and sourced from garden scene uniforms", () => {
  const scene = cloneScene(defaultScene);
  scene.terrain = gardenPoolTerrain();
  scene.container.height_m = 1;
  scene.container.fillFraction = GARDEN_WATERLINE_M;
  const build = buildSvoTerrainMaterial(scene);
  assert.equal(build.packedMetadata.byteLength, SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES);
  const unpacked = unpackSvoTerrainMaterialMetadata(build.packedMetadata);
  assert.ok(Math.abs(unpacked.baseHeight_m - metadata.baseHeight_m) < 1e-7);
  assert.ok(Math.abs(unpacked.waterline_m - metadata.waterline_m) < 1e-7);
  assert.equal(unpacked.materialId, metadata.materialId);
  assert.equal(unpacked.policyVersion, metadata.policyVersion);
  assert.equal(build.metadata.materialId, VOXEL_MATERIAL_IDS.terrain);
  assert.match(build.cacheKey, /^svo-terrain-material-v1:[0-9a-f]{8}$/);
  assert.equal(build.staticRevision, buildSvoTerrainMaterial(cloneScene(scene)).staticRevision);
  scene.container.fillFraction += 0.01;
  assert.notEqual(buildSvoTerrainMaterial(scene).staticRevision, build.staticRevision);
  assert.throws(() => buildSvoTerrainMaterial({ ...scene, terrain: undefined }), /authored terrain/);
  assert.throws(() => packSvoTerrainMaterialMetadata({ ...metadata, materialId: 0 }), /nonzero uint16/);
});

test("procedural samples match the raster garden classifier across world-space regions", () => {
  const points: Array<readonly [number, number, number]> = [];
  for (const y of [0.04, 0.2, 0.325, 0.326, 0.345, 0.35, 0.372, 0.38, 0.48]) {
    for (const [x, z] of [[-0.63, -0.17], [0.11, 0.29], [0.72, -0.44], [1.2, 0.83]]) points.push([x, y, z]);
  }
  for (const point of points) {
    const actual = sampleSvoTerrainMaterial(metadata, point).colorLinear;
    const expected = rasterGardenGroundMaterial(point);
    actual.forEach((channel, index) => assert.ok(Math.abs(channel - expected[index]) < 1e-12, `${point.join(",")} channel ${index}`));
  }
});

test("height ownership names exact raster plateaus with deterministic edge ties", () => {
  const at = (y: number) => sampleSvoTerrainMaterial(metadata, [0.17, y, -0.23]);
  assert.equal(at(metadata.waterline_m - 0.02).regionId, SVO_TERRAIN_REGION_IDS.pondLinerRock, "lower smoothstep edge remains exact liner");
  assert.equal(at(metadata.waterline_m - 0.02 + 1e-9).regionId, SVO_TERRAIN_REGION_IDS.pondEdgeSoil);
  assert.equal(at(metadata.baseHeight_m - 0.008 - 1e-9).regionId, SVO_TERRAIN_REGION_IDS.pondEdgeSoil);
  assert.equal(at(metadata.baseHeight_m - 0.008).regionId, SVO_TERRAIN_REGION_IDS.grass, "upper lawn edge is exact grass");
  for (const y of [0.1, 0.34, 0.36, 0.4]) {
    const weights = at(y).regionWeights;
    assert.ok(Math.abs(weights.pondLinerRock + weights.pondEdgeSoil + weights.grass - 1) < 1e-12);
  }
});

test("slope is reported for PBR but cannot change raster region or procedural color", () => {
  const flat = sampleSvoTerrainMaterial(metadata, [0.2, 0.36, -0.1], [0, 1, 0]);
  const steep = sampleSvoTerrainMaterial(metadata, [0.2, 0.36, -0.1], [1, 0.1, 0]);
  assert.equal(flat.slope, 0);
  assert.ok(steep.slope > 0.9);
  assert.equal(steep.regionId, flat.regionId);
  assert.deepEqual(steep.colorLinear, flat.colorLinear);
});

test("fixed raster seeds preserve pebble, mow, clover, and daisy variation deterministically", () => {
  const first = sampleSvoTerrainMaterial(metadata, [0.413, 0.4, -0.277]);
  const second = sampleSvoTerrainMaterial(metadata, [0.413, 0.4, -0.277]);
  assert.deepEqual(second, first);
  assert.equal(svoTerrainVariationHash21([12, -3]), svoTerrainVariationHash21([12, -3]));
  assert.ok((first.variationFlags & SVO_TERRAIN_VARIATION_FLAGS.mowStripe) !== 0);
  assert.notDeepEqual(first.colorLinear, sampleSvoTerrainMaterial(metadata, [0.424, 0.4, -0.266]).colorLinear);
});

test("terrain palette is the established raster/voxel palette, not a second authored palette", () => {
  const terrain = voxelMaterial(VOXEL_MATERIAL_IDS.terrain);
  assert.deepEqual(SVO_GARDEN_TERRAIN_PALETTE.grassDarkLinear, terrain.terrainPalette?.lawnDarkLinear);
  assert.deepEqual(SVO_GARDEN_TERRAIN_PALETTE.grassLightLinear, terrain.terrainPalette?.lawnLightLinear);
  assert.deepEqual(SVO_GARDEN_TERRAIN_PALETTE.soilLinear, terrain.terrainPalette?.sandLinear);
});

test("WGSL contract is binding-free and retains exact raster world-space rules", () => {
  assert.match(svoTerrainMaterialWGSL, /struct SvoTerrainMaterialMetadata/);
  assert.match(svoTerrainMaterialWGSL, /fn svoTerrainMaterial/);
  assert.match(svoTerrainMaterialWGSL, /floor\(p\.xz\*26\.0\)/);
  assert.match(svoTerrainMaterialWGSL, /smoothstep\(metadata\.waterline_m-\.02,metadata\.waterline_m\+\.04,p\.y\)/);
  assert.match(svoTerrainMaterialWGSL, /smoothstep\(metadata\.baseHeight_m-\.05,metadata\.baseHeight_m-\.008,p\.y\)/);
  assert.match(svoTerrainMaterialWGSL, /step\(\.962/);
  assert.match(svoTerrainMaterialWGSL, /step\(\.986/);
  assert.doesNotMatch(svoTerrainMaterialWGSL, /@group|@binding/);
});
