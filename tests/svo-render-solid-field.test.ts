import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getSceneDefinition } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { sceneDocument } from "../lib/core/scene-definition";
import { solidWorldContentStamp, solidWorldForScene } from "../lib/core/solid-world";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import {
  DEFAULT_SVO_RENDER_TUNING,
  normalizeSvoRenderTuning,
} from "../lib/svo/svo-render-tuning";
import {
  buildSvoRenderTerrainFieldSteps,
  createSvoRenderTerrainRefinement,
} from "../lib/svo/svo-render-solid-field";

const complete = <T>(steps: Generator<unknown, T, undefined>): T => {
  for (;;) {
    const next = steps.next();
    if (next.done) return next.value;
  }
};

test("render refinement leaves the garden physics lattice and SolidWorld unchanged", () => {
  const scene = sceneDocument(getSceneDefinition("garden-svo-lighting"));
  const domain = structuredClone(scene.voxelDomain);
  const stamp = solidWorldContentStamp(scene);
  const world = solidWorldForScene(scene);
  const pageCoordinates = world.pages.map((page) => [...page.coordinate]);

  const tuning = normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    environmentRefinementDepth: 3,
  });

  assert.equal(tuning.environmentRefinementDepth, 3);
  assert.deepEqual(scene.voxelDomain, domain);
  assert.equal(solidWorldContentStamp(scene), stamp);
  assert.deepEqual(solidWorldForScene(scene).pages.map((page) => [...page.coordinate]),
    pageCoordinates);
});

test("render refinement round-trips independently of the scene lattice", () => {
  const parsed = parseQueryState("?scene=garden-svo-lighting&svoRefinementDepth=3");
  assert.equal(parsed.ui.svoRenderTuning.environmentRefinementDepth, 3);
  const finestCellSize_m = parsed.scene.voxelDomain.finestCellSize_m;
  const serialized = serializeQueryState("", {
    presetId: parsed.presetId,
    scene: parsed.scene,
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides,
  }, parsed.ui, { view: "studio" });
  assert.equal(new URLSearchParams(serialized).get("svoRefinementDepth"), "3");
  const reparsed = parseQueryState(`?${serialized}`);
  assert.equal(reparsed.ui.svoRenderTuning.environmentRefinementDepth, 3);
  assert.equal(reparsed.scene.voxelDomain.finestCellSize_m, finestCellSize_m);
});

test("render terrain samples and topology follow the render lattice", () => {
  const scene = {
    container: { width_m: 1, height_m: 1, depth_m: 1 },
    terrain: { baseHeight_m: 0.4, features: [] },
    voxelDomain: { finestCellSize_m: 0.25, brickSize_cells: 8 },
    solidVoxels: [],
  } as unknown as Pick<SceneDescription, "container" | "terrain" | "voxelDomain" | "solidVoxels">;
  const cell = [0.125, 0.125, 0.125] as const;
  const field = complete(buildSvoRenderTerrainFieldSteps(scene, cell, 2));
  assert.ok(field);
  assert.deepEqual(field.dimensions, [8, 8]);
  assert.equal(field.heights_m.length, 64);
  assert.ok(field.heights_m.every((height) => Math.abs(height - 0.4) < 1e-6));

  const refinement = createSvoRenderTerrainRefinement({
    field,
    worldOrigin_m: [-0.5, 0, -0.5],
    renderCellSize_m: cell,
    refinedBrickDimensions: [4, 4, 4],
    nodeEdge_m: [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    brickSize: 2,
    maximumDepth: 2,
  });
  assert.equal(refinement.refineEnvironmentLeaf(0, { x: 0, y: 0, z: 0 }), true);
  assert.equal(refinement.refineEnvironmentLeaf(1, { x: 0, y: 0, z: 0 }), true);
  assert.equal(refinement.refineEnvironmentLeaf(1, { x: 0, y: 2, z: 0 }), false);
  assert.equal(refinement.refineEnvironmentLeaf(2, { x: 0, y: 1, z: 0 }), false);
});
