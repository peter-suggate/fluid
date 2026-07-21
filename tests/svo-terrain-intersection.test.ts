import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gardenPoolTerrain, GARDEN_CONTAINER } from "../lib/garden-scene";
import { terrainHeightAt, terrainNormalAt } from "../lib/terrain";
import {
  intersectSvoTerrainHeightfield,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  SVO_TERRAIN_FALLBACK_REFINEMENTS,
  SVO_TERRAIN_FALLBACK_STEPS,
  SVO_TERRAIN_FAST_BRACKET_STEPS,
  SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS,
  SVO_TERRAIN_FAST_REFINEMENTS,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";

const terrain = gardenPoolTerrain();
const sceneScale = Math.max(GARDEN_CONTAINER.width_m, GARDEN_CONTAINER.height_m, GARDEN_CONTAINER.depth_m);

test("bounded terrain intersection mirrors the authored garden height and normal", () => {
  const x = -0.2, z = 0.1;
  const height = terrainHeightAt(terrain, x, z);
  const hit = intersectSvoTerrainHeightfield(terrain, { x, y: 1.4, z }, { x: 0, y: -1, z: 0 }, sceneScale);
  assert.ok(hit);
  assert.ok(Math.abs(hit.position_m.y - height) < 5e-4, `${hit.position_m.y} should resolve ${height}`);
  assert.ok(Math.abs(hit.t_m - (1.4 - height)) < 5e-4);
  assert.equal(hit.solver, "fast");
  assert.ok(hit.heightEvaluations <= SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS, `${hit.heightEvaluations} terrain evaluations exceeded the fast-path budget`);
  const expectedNormal = terrainNormalAt(terrain, x, z, 0.02);
  assert.ok(Math.hypot(hit.normal.x - expectedNormal.x, hit.normal.y - expectedNormal.y, hit.normal.z - expectedNormal.z) < 1e-8);
});

test("terrain intersection handles inside exits, exact grazing, and robust misses", () => {
  const flatX = 2.2, flatZ = 1.8;
  const flatHeight = terrainHeightAt(terrain, flatX, flatZ);
  const inside = intersectSvoTerrainHeightfield(terrain, { x: flatX, y: flatHeight - 0.1, z: flatZ }, { x: 0, y: 1, z: 0 }, sceneScale);
  assert.ok(inside);
  assert.ok(Math.abs(inside.position_m.y - flatHeight) < 5e-4);

  const grazing = intersectSvoTerrainHeightfield(terrain, { x: flatX, y: flatHeight, z: flatZ }, { x: 1, y: 0, z: 0 }, sceneScale);
  assert.ok(grazing);
  assert.equal(grazing.t_m, 0.005);
  assert.equal(grazing.solver, "fallback");

  assert.equal(intersectSvoTerrainHeightfield(terrain, { x: flatX, y: flatHeight + 0.01, z: flatZ }, { x: 1, y: 0, z: 0 }, sceneScale), undefined);
  assert.equal(intersectSvoTerrainHeightfield(terrain, { x: flatX, y: flatHeight, z: flatZ }, { x: 0, y: 0, z: 0 }, sceneScale), undefined);
  assert.equal(intersectSvoTerrainHeightfield(undefined, { x: 0, y: 1, z: 0 }, { x: 0, y: -1, z: 0 }, sceneScale), undefined);
});

test("ordinary angled rays keep the nearest root within the fast work cap", () => {
  const origin = { x: 0, y: 1.4, z: 1.4 };
  const direction = { x: 0.2, y: -0.5, z: -0.3 };
  const hit = intersectSvoTerrainHeightfield(terrain, origin, direction, sceneScale);
  assert.ok(hit);
  assert.equal(hit.solver, "fast");
  assert.ok(hit.heightEvaluations <= SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS);
  assert.ok(Math.abs(hit.position_m.y - terrainHeightAt(terrain, hit.position_m.x, hit.position_m.z)) <= 1e-4);

  const length = Math.hypot(direction.x, direction.y, direction.z);
  const rd = { x: direction.x / length, y: direction.y / length, z: direction.z / length };
  for (let t = 0.005; t < hit.t_m - 0.002; t += 0.002) {
    const point = { x: origin.x + rd.x * t, y: origin.y + rd.y * t, z: origin.z + rd.z * t };
    assert.ok(point.y - terrainHeightAt(terrain, point.x, point.z) > 0, `unexpected earlier terrain crossing at ${t}`);
  }
});

test("WGSL terrain path is bounded, refined, and retains terrain material identity", () => {
  assert.match(svoDrySceneShader, /fn terrainHeightAt\(x:f32,z:f32\)->f32/);
  assert.equal(SVO_TERRAIN_FAST_BRACKET_STEPS + SVO_TERRAIN_FAST_REFINEMENTS + 1 + 4, SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS);
  assert.equal(SVO_TERRAIN_FAST_MAX_HEIGHT_EVALUATIONS, 12);
  assert.equal(SVO_TERRAIN_FALLBACK_STEPS, 20);
  assert.equal(SVO_TERRAIN_FALLBACK_REFINEMENTS, 8);
  assert.match(svoDrySceneShader, /for\(var bracket=1;bracket<=2;bracket\+=1\)/);
  assert.match(svoDrySceneShader, /for\(var refinement=0;refinement<5;refinement\+=1\)/);
  assert.match(svoDrySceneShader, /for\(var iteration=1;iteration<=20;iteration\+=1\)/);
  assert.match(svoDrySceneShader, /closestField<=0\.0005/);
  assert.match(svoDrySceneShader, /DryHit\(t,terrainNormalAt\(point\.xz\),dry\.terrain\.x,DRY_OWNER_NONE,SVO_FEATURE_TERRAIN,DRY_GBUFFER_FIELD_TERRAIN,DRY_GBUFFER_MOTION_STATIC,1u,0\.0/);
  assert.match(svoDrySceneShader, /let terrain=traceTerrain\(ro,rd\);if\(terrain\.t<hit\.t\)\{hit=terrain;\}/);

  const source = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
  assert.deepEqual(SVO_DRY_SCENE_PARAMS_LAYOUT, {
    sizeBytes: 256, terrainWordOffset: 24, terrainMaterialWordOffset: 28, materialPublicationWordOffset: 32,
    primitiveCandidateWordOffset: 36, nodeMipWordOffset: 40, nodeMipAtlasWordOffset: 44,
    wideFanoutWordOffset: 48, nodeMipLevelStartWordOffset: 52,
  });
  assert.match(source, /size: SVO_DRY_SCENE_PARAMS_LAYOUT\.sizeBytes/);
  assert.match(source, /new ArrayBuffer\(SVO_DRY_SCENE_PARAMS_LAYOUT\.sizeBytes\)/);
  assert.match(source, /words\.set\(\[scene\.terrainMaterialId \?\? 0xffff_ffff, \(scene\.glassRecords\?\.byteLength \?\? 0\) \/ SVO_THIN_GLASS_RECORD_STRIDE_BYTES, scene\.primaryCompositeOwnedGlassPaneIdBase \?\? 0xffff_ffff, scene\.primaryCompositeOwnedGlassPaneCount \?\? 0\], SVO_DRY_SCENE_PARAMS_LAYOUT\.terrainWordOffset\)/);
});
