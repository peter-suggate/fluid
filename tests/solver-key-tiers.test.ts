import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { getMethod, resolveMethodValues } from "../lib/methods";
import {
  gpuSceneSeedKey,
  gpuSceneSolverKey,
  gpuSceneStructuralKey,
  gpuSceneUniformKey,
  type SimulationRunConfig,
} from "../lib/webgpu-renderer";

const config: SimulationRunConfig = {
  methodId: "octree",
  quality: "balanced",
  values: resolveMethodValues(getMethod("octree"), "balanced", {}),
  simulationEpoch: 0,
};

function edited(mutate: (scene: SceneDescription) => void): SceneDescription {
  const scene = cloneScene(defaultScene);
  mutate(scene);
  return scene;
}

test("terrain is part of the solver identity", () => {
  // Previously absent from the key entirely: a terrain edit was rendered but
  // never reached the solver, which is exactly what the terrain handles edit.
  const flat = cloneScene(defaultScene);
  const sculpted = edited((scene) => {
    scene.terrain = { baseHeight_m: 0.2, features: [{ kind: "mound", center_m: { x: 0, z: 0 }, radius_m: { x: 0.2, z: 0.2 }, amount_m: 0.1 }] };
  });
  assert.notEqual(gpuSceneSeedKey(flat), gpuSceneSeedKey(sculpted));
  assert.notEqual(gpuSceneSolverKey(flat, config), gpuSceneSolverKey(sculpted, config));
});

test("each authoring input lands in exactly one tier", () => {
  const base = cloneScene(defaultScene);
  const structural = gpuSceneStructuralKey(base, config);
  const seed = gpuSceneSeedKey(base);
  const uniform = gpuSceneUniformKey(base);

  const cases: ReadonlyArray<{ label: string; tier: "structural" | "seed" | "uniform"; mutate: (scene: SceneDescription) => void }> = [
    { label: "container width", tier: "structural", mutate: (scene) => { scene.container.width_m += 0.3; } },
    { label: "container top", tier: "structural", mutate: (scene) => { scene.container.top = scene.container.top === "open" ? "closed" : "open"; } },
    { label: "wall mode", tier: "structural", mutate: (scene) => { scene.container.fluidWallMode = scene.container.fluidWallMode === "no-slip" ? "free-slip" : "no-slip"; } },
    { label: "voxel domain", tier: "structural", mutate: (scene) => { scene.voxelDomain.finestCellSize_m *= 0.5; } },
    { label: "fill fraction", tier: "seed", mutate: (scene) => { scene.container.fillFraction = 0.5 * (scene.container.fillFraction + 1); } },
    { label: "rigid bodies", tier: "seed", mutate: (scene) => { scene.rigidBodies = []; } },
    { label: "initial condition", tier: "seed", mutate: (scene) => { scene.fluid.initialCondition = scene.fluid.initialCondition === "dam-break" ? "tank-fill" : "dam-break"; } },
    { label: "brick seeds", tier: "seed", mutate: (scene) => { scene.fluid.initialBrickSeeds_m = [{ x: 0, y: 0.1, z: 0 }]; } },
    { label: "seed additivity", tier: "seed", mutate: (scene) => { scene.fluid.initialBrickSeedsAdditive = true; } },
    { label: "terrain", tier: "seed", mutate: (scene) => { scene.terrain = { baseHeight_m: 0.1, features: [] }; } },
    { label: "density", tier: "uniform", mutate: (scene) => { scene.fluid.density_kg_m3 += 10; } },
    { label: "viscosity", tier: "uniform", mutate: (scene) => { scene.fluid.dynamicViscosity_Pa_s += 0.01; } },
    { label: "surface tension", tier: "uniform", mutate: (scene) => { scene.fluid.surfaceTension_N_m += 0.01; } },
    { label: "gravity", tier: "uniform", mutate: (scene) => { scene.fluid.gravity_m_s2.y *= 0.5; } },
  ];

  for (const { label, tier, mutate } of cases) {
    const next = edited(mutate);
    const changed = {
      structural: gpuSceneStructuralKey(next, config) !== structural,
      seed: gpuSceneSeedKey(next) !== seed,
      uniform: gpuSceneUniformKey(next) !== uniform,
    };
    assert.equal(changed[tier], true, `${label} must change the ${tier} tier`);
    for (const other of ["structural", "seed", "uniform"] as const) {
      if (other !== tier) assert.equal(changed[other], false, `${label} must not leak into the ${other} tier`);
    }
    // Structural and seed changes rebuild; scalars are adopted hot instead.
    const rebuilds = gpuSceneSolverKey(next, config) !== gpuSceneSolverKey(cloneScene(defaultScene), config);
    assert.equal(rebuilds, tier !== "uniform", `${label} must ${tier === "uniform" ? "not " : ""}rebuild`);
  }
});

test("every GPU solver can adopt scene scalars, so no method silently ignores them", () => {
  // The renderer drops the uniform tier from the rebuild key. That is only
  // safe while each solver class implements the hot path; a method without it
  // falls back to rebuilding, but these two are the whole GPU roster.
  for (const path of ["../lib/webgpu-uniform-eulerian.ts", "../lib/webgpu-eulerian.ts", "../lib/webgpu-octree.ts"]) {
    const source = readFileSync(new URL(path, import.meta.url), "utf8");
    assert.match(source, /applySceneUniforms\(scene: SceneDescription\)/, `${path} must adopt scene scalars`);
  }
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /this\.gpuFluid\.applySceneUniforms\(scene\)/);
  assert.match(renderer, /beginGPUFluidInitialization\(scene, config, rebuildKey\)/,
    "a method without the hot path must rebuild rather than ignore the edit");
});

test("render-only authoring never reaches the solver identity", () => {
  const base = cloneScene(defaultScene);
  // Props extend the render domain only; placing scenery must not rebuild.
  const withProps = edited((scene) => {
    scene.props = [{ id: "p", name: "P", shape: "box", position_m: { x: 0, y: 0.1, z: 0 }, halfSize_m: { x: 0.05, y: 0.05, z: 0.05 }, colorLinear: [0.4, 0.4, 0.4] }];
  });
  assert.equal(gpuSceneSeedKey(withProps), gpuSceneSeedKey(base));
  assert.equal(gpuSceneUniformKey(withProps), gpuSceneUniformKey(base));
});

test("the epoch still identifies one replacement per reset", () => {
  const scene = cloneScene(defaultScene);
  assert.notEqual(gpuSceneSolverKey(scene, config), gpuSceneSolverKey(scene, { ...config, simulationEpoch: 1 }));
  assert.equal(gpuSceneStructuralKey(scene, config), gpuSceneStructuralKey(scene, { ...config, simulationEpoch: 1 }),
    "the epoch is a reset identity, not a structural property");
});

test("the generated power catalog is decoded once per module, and a failure is not cached", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(source, /let generatedOctreePowerCatalog: Promise<GeneratedOctreePowerCatalogViews> \| undefined/,
    "the 14 MB device-independent constant must be memoized across solver builds");
  assert.match(source, /generatedOctreePowerCatalog \?\?= readGeneratedOctreePowerCatalog\(\)/);
  assert.match(source, /generatedOctreePowerCatalog = undefined; throw error/,
    "a transient load failure must not poison the session");
});

test("a seed-tier change re-seeds the live solver instead of rebuilding it", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const solver = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const surface = readFileSync(new URL("../lib/webgpu-quadtree-builder.ts", import.meta.url), "utf8");

  assert.match(renderer, /tryReseedGPUFluid\(scene,config,key\)\)return undefined/,
    "the warm path must be attempted before beginGPUFluidInitialization");
  assert.match(renderer, /gpuSceneStructuralKey\(scene,config\)!==this\.attachedStructuralKey\)return false/,
    "a structural change must never be answered by a re-seed");
  assert.match(renderer, /this\.gpuFluid!==solver\|\|this\.gpuFluidGeneration!==generation/,
    "a solver replaced mid-flight must not be promoted to the new key");

  // The re-seed must reuse allocations: it re-runs the fenced phases rather
  // than constructing anything.
  assert.match(solver, /async reseed\(scene: SceneDescription\): Promise<boolean>/);
  assert.match(solver, /await this\.publishInitialSparseScene\(\)/);
  assert.doesNotMatch(solver.slice(solver.indexOf("async reseed"), solver.indexOf("async reseed") + 1800), /new WebGPUOctreeProjection|createTexture|createBuffer/,
    "a warm re-seed must not allocate");
  assert.match(projection, /reseed\(scene: SceneDescription\): boolean/);
  assert.match(surface, /reseedLevelSet\(device: GPUDevice, phi: Float32Array\): boolean/);

  // Failing closed is the contract: the caller falls back to a full rebuild.
  assert.match(solver, /if \(!this\.octreeProjection\) return false/);
  assert.match(solver, /if \(!this\.octreeProjection\.reseed\(scene\)\) return false/);
});
