import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";
import { sceneLatticeCellCount, sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";
import { terrainHeightAt } from "../lib/core/terrain";
import { sceneSvoGroundPlane } from "../lib/svo/features/materials/svo-terrain-material";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";
import { svoPondTerrainProgram } from "../lib/svo/features/scene-publication/webgpu-svo-render-terrain";

const pond = () => sceneDocument(getSceneDefinition("hero-garden-hose-x10"));
function original() {
  const scene = createHeroGardenHoseStressScene({ water: true, detailCellSize_m: 0.003125 });
  scene.environment = "garden";
  return scene;
}

test("fitted pond increases water resolution fourfold while preserving the set and waterline", () => {
  const scene = pond(), before = original();
  assert.equal(scene.voxelDomain.finestCellSize_m, before.voxelDomain.finestCellSize_m / 4);
  assert.deepEqual(sceneLatticeDimensions(scene), [384, 256, 288]);
  assert.equal(sceneLatticeCellCount(scene) * 3, sceneLatticeCellCount(before) * 64,
    "the fitted tank uses one third of the full garden domain at this resolution");
  assert.equal(scene.container.height_m, 0.8);
  assert.equal(scene.container.height_m * scene.container.fillFraction,
    before.container.height_m * before.container.fillFraction);
  assert.deepEqual(scene.fluid, before.fluid);
  assert.deepEqual(scene.terrain, before.terrain);
  assert.deepEqual(scene.scenery, before.scenery);
  assert.deepEqual(scene.backdrop, before.backdrop);
  assert.deepEqual(buildEnvironmentProxyCatalog(scene, "garden"),
    buildEnvironmentProxyCatalog(before, "garden"));
  assert.deepEqual(sceneSvoGroundPlane(scene), sceneSvoGroundPlane(before));
  const { width_m: width, depth_m: depth, height_m: height } = scene.container;
  const level = height * scene.container.fillFraction;
  for (let i = 0; i <= 100; i++) {
    const x = width * (i / 100 - 0.5), z = depth * (i / 100 - 0.5);
    for (const [px, pz] of [[x, -depth / 2], [x, depth / 2], [-width / 2, z], [width / 2, z]]) {
      assert.ok(terrainHeightAt(scene.terrain!, px!, pz!) > level + 0.025,
        "the tank boundary stays beyond the wet shoreline");
    }
  }
  const inflow = scene.fluid.inflow!;
  assert.ok(Math.abs(inflow.center_m.x) + inflow.radius_m + inflow.length_m < width / 2);
  assert.ok(Math.abs(inflow.center_m.z) + inflow.radius_m + inflow.length_m < depth / 2);
  assert.ok(inflow.center_m.y + inflow.radius_m < height);
});

test("fitted tank retains full-size render terrain and solid garden banks", () => {
  const scene = pond(), before = original();
  const render = (source: typeof scene) => {
    const steps = buildSvoRenderTerrainFieldSteps(source, [.003125, .003125, .003125], 2);
    let next = steps.next();
    while (!next.done) next = steps.next();
    return next.value!;
  };
  const field = render(scene), prior = render(before);
  assert.deepEqual(field.origin_m, prior.origin_m);
  assert.deepEqual(field.dimensions, [576, 384]);
  assert.deepEqual(field.heights_m, prior.heights_m);
  const gpu = svoPondTerrainProgram(scene, [.003125, .003125, .003125])!;
  assert.deepEqual([gpu.nx, gpu.nz], field.dimensions);
  const world = solidWorldForScene(scene);
  assert.ok(sceneLatticeDimensions(scene).every(n => n % 4 === 0),
    "the Uniform solver requires axes divisible by four");
  for (const [x, z] of [[-.8, 0], [.8, 0], [0, -.55], [0, .55]]) {
    const q = [Math.floor((x! + .6) / .003125), 16, Math.floor((z! + .45) / .003125)] as const;
    assert.equal(sampleSolidWorld(world, q).solidFraction, 1, "banks outside the fluid tank remain solid");
  }
});
