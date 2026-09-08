import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { intersectAuthoredTerrain, terrainHeightAt, terrainSampleShape } from "../lib/core/terrain";

test("cold procedural authoring rays never materialize a grid and hit the sampled heightfield", () => {
  const scene = sceneDocument(getSceneDefinition("hero-garden-hose-x10"));
  assert.ok(scene.terrain?.procedural);
  const terrain = { ...scene.terrain, procedural: { ...scene.terrain.procedural, spacing_m: .00001 } };
  const shape = terrainSampleShape(terrain)!;
  assert.ok(shape.nx * shape.nz > 1e10, "materializing this fixture would exceed the terrain allocation limit");
  for (const [x, z] of [[0, 0], [.3, .2], [-.4, .25]]) {
    const height = terrainHeightAt(terrain, x!, z!);
    const hit = intersectAuthoredTerrain(terrain, { x: x!, y: height + 1, z: z! }, { x: 0, y: -1, z: 0 }, 4);
    assert.ok(hit);
    assert.ok(hit.heightEvaluations <= 64, "point work is bounded independently of authored grid size");
    assert.ok(Math.abs(hit.position_m.y - height) < 1e-4,
      "the cold authoring intersection resolves the same bilinear procedural field");
    assert.ok(Object.values(hit.normal).every(Number.isFinite));
  }
});
