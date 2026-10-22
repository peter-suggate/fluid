import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultScene, type SceneDescription } from "../lib/core/model";
import { classifySceneryChange, sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { VOXEL_MATERIAL_IDS, VOXEL_MATERIALS } from "../lib/core/voxel-scene";
import { buildDefaultSvoMaterialRecords, packSvoMaterialTable, SVO_MATERIAL_FLAGS,
  SVO_MATERIAL_FUNCTION_IDS, SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";
import { forestTerrainMaterialWGSL, sampleForestTerrainMaterialId, sceneForestTerrainMaterial,
  sceneTerrainSurfaceModel, SVO_FOREST_TERRAIN_MATERIAL_IDS as ids,
  SVO_FOREST_TERRAIN_PALETTE as palette, SVO_PORCELAIN_TERRAIN_BASE_COLOR_LINEAR,
  type SvoTerrainSurfaceModel } from "../lib/svo/features/materials/svo-terrain-material";

function sceneWithSurface(model: SvoTerrainSurfaceModel, waterline_m?: number): SceneDescription {
  return { ...defaultScene, container: { ...defaultScene.container, height_m: 12, fillFraction: .25 },
    terrain: { baseHeight_m: 1, features: [] },
    scenery: { palettes: {}, nodes: [{ kind: "group", id: "land", children: [
      { kind: "terrain-shell", id: "terrain", materialModel: model, ...(waterline_m === undefined ? {} : { waterline_m }) },
    ] }] } };
}

test("forest terrain publishes four opaque PBR slots without changing water, rigid, garden, or porcelain materials", () => {
  const original = buildDefaultSvoMaterialRecords(7);
  const forest = buildDefaultSvoMaterialRecords(7, { terrainSurface: "forest-terrain" });
  const garden = buildDefaultSvoMaterialRecords(7, { terrainSurface: "garden-terrain" });
  assert.deepEqual(garden, original);
  for (const record of original) {
    if (record.materialId !== VOXEL_MATERIAL_IDS.terrain) {
      assert.deepEqual(forest.find(value => value.materialId === record.materialId), record);
    }
  }
  for (const [id, color] of [[ids.moss, palette.mossLinear], [ids.soil, palette.soilLinear],
    [ids.rock, palette.rockLinear], [ids.wetRock, palette.wetRockLinear]] as const) {
    const record = forest.find(value => value.materialId === id)!;
    assert.ok(record, `material ${id} is published`);
    assert.deepEqual(record.baseColorLinear, color);
    assert.equal(record.flags, SVO_MATERIAL_FLAGS.opaque);
    assert.equal(record.materialFunctionId, SVO_MATERIAL_FUNCTION_IDS.none,
      "forest cannot also receive the garden's porcelain color policy");
    assert.equal(record.revision, 7);
    assert.equal(record.transmission, 0);
  }
  assert.ok(forest.find(value => value.materialId === ids.wetRock)!.roughness
    < forest.find(value => value.materialId === ids.rock)!.roughness);
  const packed = packSvoMaterialTable(forest);
  for (const id of Object.values(ids)) assert.equal(packed[id * SVO_MATERIAL_RECORD_WORDS + 20], id);
  assert.equal(new Set(forest.map(value => value.materialId)).size, forest.length);
  assert.equal(VOXEL_MATERIALS.some(value => new Set<number>([ids.soil, ids.rock, ids.wetRock]).has(value.id)), false,
    "render terrain bands do not add physical material identities");
  const porcelain = buildDefaultSvoMaterialRecords(7, { terrainSurface: "porcelain" });
  assert.equal(porcelain.length, original.length);
  const porcelainGround = porcelain.find(value => value.materialId === ids.moss)!;
  assert.deepEqual(porcelainGround.baseColorLinear, SVO_PORCELAIN_TERRAIN_BASE_COLOR_LINEAR);
  assert.equal(porcelainGround.materialFunctionId, SVO_MATERIAL_FUNCTION_IDS.plaster);
});

test("forest shell metadata uses the authored waterline for volume-filled ponds and falls back to container fill", () => {
  const authored = sceneWithSurface("forest-terrain", 3.1);
  assert.equal(sceneTerrainSurfaceModel(authored), "forest-terrain");
  assert.deepEqual(sceneForestTerrainMaterial(authored), { baseHeight_m: 1, waterline_m: 3.1 });
  assert.deepEqual(sceneForestTerrainMaterial(sceneWithSurface("forest-terrain")), { baseHeight_m: 1, waterline_m: 3 });
  for (const model of ["garden-terrain", "porcelain"] as const) {
    assert.equal(sceneTerrainSurfaceModel(sceneWithSurface(model)), model);
    assert.equal(sceneForestTerrainMaterial(sceneWithSurface(model)), undefined);
  }
  assert.equal(sceneForestTerrainMaterial({ ...authored, scenery: undefined }), undefined);
  assert.throws(() => sceneForestTerrainMaterial(sceneWithSurface("forest-terrain", NaN)), /finite/);
  assert.throws(() => forestTerrainMaterialWGSL({ baseHeight_m: 0, waterline_m: Infinity }), /finite/);
});

test("forest surface bands distinguish submerged stone, cliff faces, shoreline soil, and mossy level ground", () => {
  const metadata = { baseHeight_m: 1, waterline_m: 3.1 };
  // Points stay clear of the deliberately irregular band transitions.
  for (const [x, z] of [[0, 0], [7.4, -2.9], [-10.1, 14.2]]) {
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 2.6, z], [0, 1, 0]), ids.wetRock);
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 15, z], [1, 0, 0]), ids.rock);
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 3.15, z], [0, 1, 0]), ids.soil);
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 8, z], [0, 1, 0]), ids.moss);
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 8, z], [0, 8, 0]), ids.moss,
      "classification depends on slope, not normal magnitude");
    assert.equal(sampleForestTerrainMaterialId(metadata, [x, 2.6, z], [1, 0, 0]), ids.wetRock,
      "submerged cliff rock remains wet");
  }
  assert.equal(sampleForestTerrainMaterialId(metadata, [0, 8, 0], [Math.sqrt(1 - .75 ** 2), .75, 0]), ids.soil);
  assert.equal(sampleForestTerrainMaterialId({ baseHeight_m: 6, waterline_m: 3.1 }, [0, 5, 0], [0, 1, 0]), ids.soil);
});

test("forest band selection invalidates voxel construction in both directions while garden closure edits stay appearance-only", () => {
  const garden = sceneWithSurface("garden-terrain"), forest = sceneWithSurface("forest-terrain");
  const porcelain = sceneWithSurface("porcelain"), movedShore = sceneWithSurface("forest-terrain", 4);
  for (const [a, b] of [[garden, forest], [forest, garden], [porcelain, forest], [forest, porcelain], [forest, movedShore]]) {
    assert.equal(classifySceneryChange(a.scenery, b.scenery).kind, "geometry");
    assert.notEqual(sceneryConstructionKey(a), sceneryConstructionKey(b));
  }
  assert.equal(classifySceneryChange(garden.scenery, porcelain.scenery).kind, "appearance");
  assert.equal(sceneryConstructionKey(garden), sceneryConstructionKey(porcelain));
  const clone = structuredClone(forest);
  assert.equal(classifySceneryChange(forest.scenery, clone.scenery).kind, "same");
  assert.equal(sceneryConstructionKey(forest), sceneryConstructionKey(clone));
});
