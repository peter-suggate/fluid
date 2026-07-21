import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { parseScene, serializeScene, validateScene } from "../lib/model";
import { getScenePreset, scenePresets } from "../lib/scenes";
import { planSceneRuntime } from "../lib/scene-runtime";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { buildEnvironmentProxyCatalog } from "../lib/voxel-environments";
import { buildOctreeSvoEnvironmentLightingPublication } from "../lib/webgpu-octree-sparse-bricks";
import { canInitializeGPUSceneSource, gpuSceneSolverKey, type SimulationRunConfig } from "../lib/webgpu-renderer";

test("garden SVO lighting preset is a valid fluid-free static scene", () => {
  const preset = getScenePreset("garden-svo-lighting");
  const scene = preset.create();

  assert.equal(preset.group, "Garden");
  assert.equal(scene.sceneId, "garden-svo-lighting-study");
  assert.equal(scene.systems?.fluid, false);
  const runtimePlan = planSceneRuntime(scene, { methodId: "octree", renderMode: "svo" });
  assert.equal(runtimePlan.staticWorld, true);
  assert.equal(runtimePlan.fluidSolver, false);
  assert.equal(runtimePlan.rigidCoupling, false);
  assert.equal(runtimePlan.waterPresentation, false);
  assert.equal(runtimePlan.sparseVoxelPresentation, true);
  assert.equal(runtimePlan.readiness.fluidAuthority.state, "not-required");
  assert.equal(runtimePlan.readiness.transport.state, "not-required");
  assert.equal(scene.environment, "garden");
  assert.equal(scene.container.fillFraction, 0);
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.025, brickSize_cells: 8 });
  assert.ok(scene.terrain);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.equal(scene.rigidBodies.some(({ id }) => id === "garden-cork-ball"), false);
  assert.ok(scene.rigidBodies.length >= 3);
  assert.ok(scene.rigidBodies.every(({ motion }) => motion === "static"));
  assert.deepEqual(validateScene(scene), []);

  const roundTrip = parseScene(serializeScene(scene));
  assert.equal(roundTrip.systems?.fluid, false);
  assert.deepEqual(roundTrip.voxelDomain, scene.voxelDomain);
  assert.deepEqual(roundTrip.lighting, scene.lighting);

  roundTrip.lighting = { directional: { intensity: -1 }, environment: { diffuseScale: Number.NaN } };
  assert.ok(validateScene(roundTrip).includes("Scene directional-light intensity must be non-negative and finite"));
  assert.ok(validateScene(roundTrip).includes("Scene environment diffuse scale must be non-negative and finite"));
});

test("garden lighting study authors a bounded warm point light on visible lamppost geometry", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const lantern = catalog.primitives.find(({ key }) => key === "garden/lamppost/lantern");
  assert.ok(lantern, "the garden previously had no lamppost; its emitter must now be real scene geometry");
  assert.ok(catalog.primitives.some(({ key }) => key === "garden/lamppost/pole"));
  assert.ok(catalog.primitives.some(({ key }) => key === "garden/lamppost/cap"));
  assert.ok(lantern.tags.includes("point-light"));

  const lights = buildSvoSceneLights(scene);
  const point = lights.records.find(({ sourceKey }) => sourceKey === lantern.key);
  const directional = lights.records[0];
  assert.ok(point);
  assert.equal(point.kind, "point");
  assert.ok(point.position_m.every((value, index) => Math.abs(value - [0.96, 1.43, 0.72][index]) < 1e-12));
  assert.equal(point.range_m, 4.5, "finite range bounds shadow traversal and distant energy");
  assert.equal(point.radius_m, 0.18,
    "point visibility ends at the authored lantern globe instead of its center");
  assert.deepEqual(point.colorLinear, [1, 0.48, 0.19]);
  assert.equal(point.intensity, 11);
  assert.equal(directional.kind, "directional");
  assert.equal(directional.intensity, 0.09);
  const luminance = (color: readonly [number, number, number]) => .2126 * color[0] + .7152 * color[1] + .0722 * color[2];
  assert.ok(point.intensity * luminance(point.colorLinear) > 60 * directional.intensity * luminance(directional.colorLinear),
    "the local warm fixture, not the residual directional fill, must carry the composition");

  const environment = buildOctreeSvoEnvironmentLightingPublication(scene);
  assert.equal(environment.record.diffuseScale, 0.12);
  assert.equal(environment.record.specularScale, 0.25);

  const ordinaryGarden = buildEnvironmentProxyCatalog(getScenePreset("garden-pond").create(), "garden");
  assert.equal(ordinaryGarden.primitives.some(({ key }) => key.includes("/lamppost/")), false,
    "the authored fixture belongs to the lighting study, not every garden simulation");
});

test("garden lighting scene rebuilds its complete lattice from scene voxel controls", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const fine = createTallCellLayout(scene, "balanced");
  assert.deepEqual([fine.nx, fine.fineNy, fine.nz], [120, 40, 88]);

  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.05, brickSize_cells: 4 };
  const switched = createTallCellLayout(scene, "ultra");
  assert.deepEqual([switched.nx, switched.fineNy, switched.nz], [60, 20, 44]);
  assert.equal(scene.voxelDomain.brickSize_cells, 4);
  assert.deepEqual(validateScene(scene), []);

  const staticSource = readFileSync(new URL("../lib/webgpu-static-svo-scene.ts", import.meta.url), "utf8");
  assert.match(staticSource, /brickSize: scene\.voxelDomain\.brickSize_cells/,
    "changing the dry lighting scene leaf size must reach the allocated sparse world");
});

test("every authored scene declares one scene-level voxel authority", () => {
  for (const preset of scenePresets) {
    const scene = preset.create();
    assert.ok(scene.voxelDomain.finestCellSize_m > 0, preset.id);
    assert.ok(scene.voxelDomain.brickSize_cells === 4 || scene.voxelDomain.brickSize_cells === 8, preset.id);
    assert.deepEqual(validateScene(scene), [], preset.id);
  }
});

test("existing garden presets retain ordinary fluid execution", () => {
  for (const id of ["garden-pond", "garden-dam-break", "garden-hose"]) {
    const scene = getScenePreset(id).create();
    assert.equal(planSceneRuntime(scene).fluidSolver, true, id);
    assert.ok(scene.container.fillFraction > 0, id);
  }
});

test("4-cubed leaves fail fast for wet scenes instead of degrading fluid ownership", () => {
  const wet = getScenePreset("garden-pond").create();
  wet.voxelDomain = { ...wet.voxelDomain, brickSize_cells: 4 };
  assert.ok(validateScene(wet).includes("Fluid-enabled scenes require 8-cell voxel bricks"));

  const dry = getScenePreset("garden-svo-lighting").create();
  dry.voxelDomain = { ...dry.voxelDomain, brickSize_cells: 4 };
  assert.deepEqual(validateScene(dry), []);
});

test("static SVO startup bypasses the simulation solver and t=0 raster gate", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const staticSource = readFileSync(new URL("../lib/webgpu-static-svo-scene.ts", import.meta.url), "utf8");

  assert.match(renderer, /planSceneRuntime\(scene,\{methodId:config\.methodId\}\)\.fluidSolver/);
  assert.match(renderer, /WebGPUStaticSvoScene\.create/);
  assert.match(renderer, /backend === "webgpu" \|\| !sceneRuntime\.fluidSolver[^]*this\.currentGPUFluid\(scene, config, time_s\)/,
    "fluid-disabled scenes must request their renderer-owned GPU source even under the CPU reference method");
  assert.match(renderer, /if\(!canInitializeGPUSceneSource\(scene,config\.methodId\)\)return/,
    "fluid-enabled scenes without a GPU solver factory must remain fail-closed");
  assert.doesNotMatch(renderer, /if\(staticRenderScene\)this\.onStatus\(\{state:"ready",label:"Static SVO renderer ready"/,
    "sparse-world attachment must not declare the garden visible before the dry renderer presents");
  assert.match(renderer, /this\.pendingStaticSvoPresentation=\{solver,solverGeneration:this\.gpuFluidGeneration,requestGeneration:generation,startedAt_ms,attached:false,submitted:false\}/,
    "static startup must open a separate presentation gate");
  assert.match(renderer, /pendingStaticSvo\.attached[^]*pendingStaticSvo\.solver === readyGPUFluid[^]*svoEncoded/,
    "the first static presentation must require an attached source and successful SVO encoding");
  assert.match(renderer, /queue\.onSubmittedWorkDone\(\)\.then\([^]*settleStaticSvoPresentation\(initialStaticSvoSubmission\)/,
    "ready must be published only after the first sparse garden frame completes");
  assert.doesNotMatch(staticSource, /WebGPUUniformEulerianSolver/);
  assert.match(staticSource, /fluid authority intentionally bypassed/);
  assert.match(staticSource, /new OctreeSparseBrickWorld/);
  assert.match(staticSource, /emptyPhi\.fill/);
});

test("fluid-disabled static GPU sources initialize independently of the selected solver method", () => {
  const staticScene = getScenePreset("garden-svo-lighting").create();
  const fluidScene = getScenePreset("garden-pond").create();

  assert.equal(canInitializeGPUSceneSource(staticScene, "cpu-reference"), true,
    "renderer-owned static SVO construction must not require a fluid solver factory");
  assert.equal(canInitializeGPUSceneSource(fluidScene, "cpu-reference"), false,
    "a fluid-enabled scene must not enter GPU construction without a GPU solver factory");
  assert.equal(canInitializeGPUSceneSource(staticScene, "octree"), true);
  assert.equal(canInitializeGPUSceneSource(fluidScene, "octree"), true);
});

test("GPU scene rebuild identity includes captured container and owner-layout inputs", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const config: SimulationRunConfig = { methodId: "octree", quality: "balanced", values: {}, simulationEpoch: 4 };
  const baseline = gpuSceneSolverKey(scene, config);

  scene.container.top = scene.container.top === "open" ? "closed" : "open";
  const changedTop = gpuSceneSolverKey(scene, config);
  assert.notEqual(changedTop, baseline, "container top is captured by sparse-world construction");

  scene.rigidBodies = scene.rigidBodies.slice(1);
  assert.notEqual(gpuSceneSolverKey(scene, config), changedTop,
    "rigid roster changes environment owner offsets and must rebuild the dry-scene source");

  const changedBodies = gpuSceneSolverKey(scene, config);
  scene.lighting = { ...scene.lighting, environment: { ...scene.lighting?.environment, diffuseScale: 0.25 } };
  assert.notEqual(gpuSceneSolverKey(scene, config), changedBodies,
    "authored lighting changes must rebuild matching GPU publications");
});

test("static SVO scenes lazily expose raw-voxel inspection records", () => {
  const staticSource = readFileSync(new URL("../lib/webgpu-static-svo-scene.ts", import.meta.url), "utf8");

  assert.match(staticSource, /get sparseVoxelRenderSource\(\)/,
    "the renderer's existing inspection attachment must work for fluid-free scenes");
  assert.match(staticSource, /this\.world\.ensureInspectionSource\(\)/,
    "raw records should be derived from the same authoritative static octree");
  assert.match(staticSource, /worldBytes - this\.accountedWorldBytes/,
    "lazy inspection allocation must be reflected in renderer telemetry");

  const constructorStart = staticSource.indexOf("private constructor(");
  const getterStart = staticSource.indexOf("get sparseVoxelRenderSource()", constructorStart);
  assert.doesNotMatch(staticSource.slice(constructorStart, getterStart), /ensureInspectionSource\(\)/,
    "normal smooth SVO startup must not allocate capacity-sized debug records");
});
