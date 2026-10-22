import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";
import { svoRenderScenePublication } from "../lib/svo/features/scene-publication/svo-render-scene-publication";

test("voxel publication retains terrain closure without walking an unchanged scenery graph", () => {
  let kindReads = 0;
  const node = { get kind() { kindReads++; return "terrain-shell" as const; },
    id: "ground", materialModel: "garden-terrain" as const };
  const base = { ...createEmptyScene(), scenery: { palettes: {}, nodes: [node] } };
  const first = svoRenderScenePublication(base, 1, undefined)!;
  assert.ok(first); assert.equal(first.terrainSurface, "garden-terrain");
  const painted = sceneWithSolidStroke(base, [{ operation: "fill", minimum: [4, 4, 4], maximumExclusive: [5, 5, 5] }]);
  kindReads = 0;
  const next = svoRenderScenePublication(painted, 2, first)!;
  assert.equal(kindReads, 0, "a solid-only edit must not traverse authored nodes again");
  assert.equal(next.terrainSurface, first.terrainSurface);
  assert.equal(next.data.materialRecords, first.data.materialRecords);
  const porcelain = { ...painted, scenery: { palettes: {}, nodes: [{ ...node, materialModel: "porcelain" as const }] } };
  const changed = svoRenderScenePublication(porcelain, 2, next)!;
  assert.equal(changed.terrainSurface, "porcelain", "an actual scenery change re-evaluates the closure");
  assert.notEqual(changed.data.materialRecords, next.data.materialRecords);
});

test("solid-only world staging bypasses appearance work but light and material changes still reach it", () => {
  const base = createEmptyScene();
  const state = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    stagedDocument: base, pbrMaterialRecords: new Uint32Array(24),
    device: { get queue() { throw new Error("appearance work"); } },
  });
  const painted = sceneWithSolidStroke(base, [{ operation: "fill", minimum: [4, 4, 4], maximumExclusive: [5, 5, 5] }]);
  assert.equal(state.stageAppearanceTables(painted, [], [], false), false);
  assert.throws(() => state.stageAppearanceTables({ ...painted, lighting: { ...base.lighting } }, [], [], false), /appearance work/);
  assert.throws(() => state.stageAppearanceTables({ ...painted, scenery: { ...base.scenery! } }, [], [], false), /appearance work/);
  assert.throws(() => state.stageAppearanceTables(painted, [], [{}], false), /appearance work/);
  assert.throws(() => state.stageAppearanceTables(painted, [], [], true), /appearance work/);
});
