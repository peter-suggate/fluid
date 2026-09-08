import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { packTerrainOverlay, terrainFieldStamp, terrainOverlayPatches } from "../lib/core/live-terrain-overlay";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { toolValues } from "../lib/core/voxel-editor/plugin";

function terrainScene() {
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
  scene.terrain = { baseHeight_m: .2, features: [] };
  return scene;
}
test("hero-garden-hose-x10 offers all solid plugins and explains its intentionally disabled fluid", () => {
  const scene = sceneDocument(getSceneDefinition("hero-garden-hose-x10"));
  assert.ok(scene.terrain);
  assert.equal(scene.systems?.fluid, false);
  for (const plugin of voxelTools.tools) {
    const unavailable = plugin.unavailable({ scene, methodId: "adaptive-mass" });
    if (plugin.execution === "release") assert.match(unavailable ?? "", /Enable water from Scene/);
    else assert.equal(unavailable, undefined, plugin.id);
  }
});
test("terrain targeting edits its authoritative surface and preserves untouched pages and terrain recipe", () => {
  const base = terrainScene();
  const before = solidWorldForScene(base);
  const ray = { origin: { x: .075, y: .7, z: .075 }, direction: { x: 0, y: -1, z: 0 } };
  for (const id of ["build", "carve"] as const) {
    const plugin = voxelTools.get(id)!;
    const result = plugin.begin({ scene: base, ray, values: toolValues(plugin, { size: 1, depth: 1 }, base) })!.update(ray)!;
    const edited = sceneWithSolidStroke(base, result.patches);
    assert.equal(edited.terrain, base.terrain);
    assert.equal(terrainFieldStamp(edited), terrainFieldStamp(base));
    const at = result.patches[0]!.minimum;
    assert.equal(sampleSolidWorld(solidWorldForScene(edited), at).solidFraction, id === "build" ? 1 : 0);
    assert.equal(sampleSolidWorld(before, at).solidFraction, id === "build" ? 0 : 1);
    assert.ok(solidWorldForScene(edited).pages.some(page => before.pages.includes(page)), "untouched terrain pages remain shared");
  }
});
test("ordered terrain overlay preserves fill/clear precedence and Undo truncates its program", () => {
  const base = terrainScene();
  const filled = sceneWithSolidStroke(base, [{ operation: "fill", minimum: [8, 4, 8], maximumExclusive: [10, 6, 10], materialId: 2 }]);
  const carved = sceneWithSolidStroke(filled, [{ operation: "clear", minimum: [8, 3, 8], maximumExclusive: [9, 6, 9] }]);
  const lattice = { origin_m: [-.4, 0, -.4] as const, cellSize_m: [.05, .05, .05] as const };
  const packed = packTerrainOverlay(terrainOverlayPatches(solidWorldForScene(carved), lattice), 4096);
  const words = new Uint32Array(packed), floats = new Float32Array(packed);
  assert.equal(words[words.length - 5], 0, "the last operation clears");
  assert.ok(Math.abs(floats[floats.length - 7]! - .15) < 1e-7);
  assert.equal(sampleSolidWorld(solidWorldForScene(carved), [8, 3, 8]).solidFraction, 0);
  const undo = packTerrainOverlay(terrainOverlayPatches(solidWorldForScene(filled), lattice), 4096);
  assert.equal(undo.byteLength, packed.byteLength - 32);
  assert.deepEqual(new Uint8Array(undo), new Uint8Array(packed, 0, undo.byteLength));
  assert.throws(() => packTerrainOverlay(terrainOverlayPatches(solidWorldForScene(carved), lattice), 1), /capacity reached/);
  assert.notEqual(terrainFieldStamp({ ...base, terrain: { baseHeight_m: .25, features: [] } }), terrainFieldStamp(base));
});

test("terrain overlays use renderer uniform publication without replacing the live scene", async () => {
  await import("../lib/methods");
  const { FluidLabRenderer, gpuSceneSolverKey } = await import("../lib/core/webgpu-renderer");
  const { sceneryConstructionKey } = await import("../lib/core/scenery-construction-key");
  const scene = terrainScene();
  const config = { methodId: "adaptive-mass", quality: "balanced", values: {} } as import("../lib/core/webgpu-renderer").SimulationRunConfig;
  const staged: typeof scene[] = [];
  const source = { stageSceneUpdate(next: typeof scene) { staged.push(next); }, info: {} };
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  Object.assign(renderer, { device: {}, gpuFluid: source,
    gpuFluidKey: `${gpuSceneSolverKey(scene, config)}:presentation-full-scene:scenery-${sceneryConstructionKey(scene)}`,
    appliedSceneUniformKey: "before", beginGPUFluidInitialization() { assert.fail("Terrain overlay rebuilt the live scene"); } });
  const access = renderer as unknown as { currentGPUFluid(next: typeof scene, runConfig: typeof config, mode: string): unknown };
  const filled = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [8, 4, 8], maximumExclusive: [9, 5, 9] }]);
  const cleared = sceneWithSolidStroke(filled, [{ operation: "clear", minimum: [8, 3, 8], maximumExclusive: [9, 5, 9] }]);
  for (const document of [scene, filled, cleared, scene]) {
    assert.equal(access.currentGPUFluid(document, config, "full-scene"), source);
  }
  assert.deepEqual(staged, [scene, filled, cleared, scene]);
});
