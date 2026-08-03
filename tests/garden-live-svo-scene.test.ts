import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { parseScene, serializeScene, validateScene } from "../lib/model";
import { getSceneDefinition, getScenePreset, scenePresets } from "../lib/scenes";
import { planSceneRuntime } from "../lib/scene-runtime";
import { planSparseSceneDomain } from "../lib/sparse-scene-domain";
import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import {
  buildOctreeSvoEnvironmentLightingPublication,
  environmentMaximumCoarseningPower,
  liveSvoBasePageDimensions,
  liveSceneBrickCoordinatesForRegions,
  liveSvoDenseFinestPages,
  sparseSceneOctreeMaximumDepth,
} from "../lib/webgpu-octree-sparse-bricks";
import { liveSvoPlanBasePages } from "../lib/webgpu-svo-live-derived-builder";
import { liveSvoRenderBrickSize } from "../lib/webgpu-live-svo-scene";
import { canInitializeGPUSceneSource, gpuSceneSolverKey, type SimulationRunConfig } from "../lib/webgpu-renderer";

test("garden SVO lighting preset is a valid fluid-free fluid-free scene", () => {
  const preset = getScenePreset("garden-svo-lighting");
  const scene = preset.create();

  // A study, not an oracle and not a scene to explore — it exists to validate
  // mip-cone lighting. The shelf is a display label; the audience is the claim.
  assert.equal(getSceneDefinition("garden-svo-lighting").audience, "study");
  assert.equal(scene.sceneId, "garden-svo-lighting-study");
  assert.equal(scene.systems?.fluid, false);
  const runtimePlan = planSceneRuntime(scene);
  assert.equal(runtimePlan.liveScene, true);
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

  const liveSource = readFileSync(new URL("../lib/webgpu-live-svo-scene.ts", import.meta.url), "utf8");
  assert.match(liveSource, /brickSize: liveSvoRenderBrickSize\(scene, options\)/,
    "changing the dry lighting scene leaf size must reach the allocated sparse world");
  assert.doesNotMatch(liveSource, /staticLightingBrickSize|buildSvoStatic/,
    "live render topology must not retain a second baked lighting lattice");
});

test("garden dense live-derived plan certifies every level and keeps edits local", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const fine = createTallCellLayout(scene, "balanced");
  const solverDimensions = [fine.nx, fine.fineNy, fine.nz] as const;
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const proxies = environmentProxyPrimitives(catalog, true);
  const domain = planSparseSceneDomain(
    scene, solverDimensions, scene.voxelDomain.brickSize_cells,
    proxies.map(({ aabb_m }) => ({ min: aabb_m.min, max: aabb_m.max })),
    { conservativePaddingCells: 1, worldBounds_m: scene.voxelDomain.bounds_m },
  );
  assert.deepEqual(domain.brickDimensions, [23, 12, 14]);
  const finestPages = liveSvoDenseFinestPages(domain.brickDimensions);
  const plan = planSvoNodeMipPyramid({ generation: 1, occupiedPages: finestPages, levelCount: 6 });
  assert.deepEqual(Array.from({ length: 6 }, (_, level) => plan.pages.filter((page) => page.key.level === level).length),
    [3_864, 504, 72, 12, 2, 1]);
  assert.equal(plan.pages.length, 4_455);
  assert.equal(new Set(plan.pages.map(({ slot }) => slot)).size, plan.pages.length,
    "the one-time runtime certification can cover every planned slot exactly once");

  const lantern = proxies.find(({ key }) => key === "garden/lamppost/lantern"); assert.ok(lantern);
  const asTuple = ({ x, y, z }: { x: number; y: number; z: number }) => [x, y, z] as [number, number, number];
  const oldBounds = { minimum: asTuple(lantern.aabb_m.min), maximum: asTuple(lantern.aabb_m.max) };
  const shift = .4;
  const newBounds = {
    minimum: [oldBounds.minimum[0] + shift, oldBounds.minimum[1], oldBounds.minimum[2]] as [number, number, number],
    maximum: [oldBounds.maximum[0] + shift, oldBounds.maximum[1], oldBounds.maximum[2]] as [number, number, number],
  };
  const dirtyBricks = liveSceneBrickCoordinatesForRegions(
    [oldBounds, newBounds], asTuple(domain.worldOrigin_m), domain.cellSize_m,
    scene.voxelDomain.brickSize_cells, domain.brickDimensions,
  );
  assert.ok(dirtyBricks.length > 0 && dirtyBricks.length < finestPages.length / 100,
    "a spatial edit dirties only its old/new brick neighborhood");
  const plannedKeys = new Set(plan.pages.map(({ key }) => `${key.level}:${key.coordinate.join(",")}`));
  for (const brick of dirtyBricks) for (let level = 0; level < 6; level += 1) {
    const coordinate = [brick.x >> level, brick.y >> level, brick.z >> level];
    assert.ok(plannedKeys.has(`${level}:${coordinate.join(",")}`),
      "each dirty old/new page has an address-resident ancestor for incremental rebuild");
  }
});

test("hose live-derived hierarchy follows sparse leaves and fits the M1 page ceiling", () => {
  const scene = getScenePreset("hose-tank").create();
  const fine = createTallCellLayout(scene, "balanced");
  const dimensions = [fine.nx, fine.fineNy, fine.nz] as const;
  const proxies = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, scene.environment ?? "default"), true);
  const domain = planSparseSceneDomain(
    scene, dimensions, scene.voxelDomain.brickSize_cells,
    proxies.map(({ aabb_m }) => ({ min: aabb_m.min, max: aabb_m.max })),
    { conservativePaddingCells: 1, worldBounds_m: scene.voxelDomain.bounds_m },
  );
  const maximumDepth = sparseSceneOctreeMaximumDepth(domain.brickDimensions, domain.coordinates);
  const topology = planAdaptiveSparseBrickOctree({
    brickSize: scene.voxelDomain.brickSize_cells,
    solverBricks: domain.solverBrickCoordinates,
    proxyBricks: domain.proxyBrickCoordinates.flat(),
    maximumDepth,
    maximumEnvironmentCoarseningPower: Math.min(environmentMaximumCoarseningPower(), maximumDepth),
  });
  const baseDimensions = liveSvoBasePageDimensions(domain.brickDimensions, topology.brickSize);
  const levelCount = sparseSceneOctreeMaximumDepth(baseDimensions, []) + 1;
  const sparse = planSvoNodeMipPyramid({ generation: 1, occupiedPages: liveSvoPlanBasePages(topology), levelCount });
  const dense = planSvoNodeMipPyramid({ generation: 1, occupiedPages: liveSvoDenseFinestPages(baseDimensions), levelCount });

  assert.equal(sparse.pages.length, 14_303);
  assert.ok(sparse.pages.length <= 16_384, "the complete sparse hierarchy fits the M1 sampled-directory height");
  assert.ok(dense.pages.length > 16_384, "allocating the empty bounding box would incorrectly disable cones");
  assert.equal(sparse.complete, true);
});

test("live SVO render brick overrides do not mutate a fluid scene's solver contract", () => {
  const scene = getScenePreset("hose-tank").create();
  assert.equal(scene.voxelDomain.brickSize_cells, 8);
  assert.ok(validateScene(scene).every((message) => !message.includes("4-cell voxel bricks")));

  assert.equal(liveSvoRenderBrickSize(scene), 8);
  assert.equal(liveSvoRenderBrickSize(scene, { renderBrickSize: 4 }), 4);
  assert.equal(scene.voxelDomain.brickSize_cells, 8,
    "the benchmark override is a renderer construction input, not an authored scene mutation");
  assert.deepEqual(validateScene(scene), []);
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

test("live SVO startup bypasses the simulation solver and t=0 raster gate", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const liveSource = readFileSync(new URL("../lib/webgpu-live-svo-scene.ts", import.meta.url), "utf8");

  assert.match(renderer, /planSceneRuntime\(scene\)\.fluidSolver/);
  assert.match(renderer, /WebGPULiveSvoScene\.create/);
  // The scene argument is `this.simulationScene ?? scene`: a direct-manipulation
  // draft may be presented while the solver stays pinned to the committed scene
  // (see scene-draft-store). What this asserts is unchanged — the request is
  // still made for fluid-disabled scenes under the CPU reference method.
  assert.match(renderer, /backend === "webgpu" \|\| !sceneRuntime\.fluidSolver[^]*this\.currentGPUFluid\(this\.simulationScene \?\? scene, svoSceneConfig\)/,
    "fluid-disabled scenes must request their renderer-owned GPU source even under the CPU reference method");
  assert.match(renderer, /svoEnvironmentBrickRefinementLevels: activeSvoTuning\.environmentBrickRefinementLevels/,
    "render tuning must participate in sparse-world rebuild identity");
  assert.match(renderer, /if\(!canInitializeGPUSceneSource\(scene,config\.methodId\)\)return/,
    "fluid-enabled scenes without a GPU solver factory must remain fail-closed");
  assert.doesNotMatch(renderer, /if\(rendererOnlyScene\)this\.onStatus\(\{state:"ready",label:"Live SVO renderer ready"/,
    "sparse-world attachment must not declare the garden visible before the dry renderer presents");
  assert.match(renderer, /new PendingLiveSvoPresentation\([\s\S]*solver, source, this\.gpuFluidGeneration, requestGeneration, startedAt_ms\)/,
    "a published sparse source must open a separate presentation gate");
  assert.match(renderer, /private attachSparsePresentationSource\([^]*const source = solver\.sparseVoxelSceneSource;[^]*this\.svoDryScenePipeline\?\.setSource\(source\);[^]*new PendingLiveSvoPresentation/,
    "fluid-backed and renderer-only sparse sources must share the attach/frame completion lifecycle");
  assert.match(renderer, /class PendingLiveSvoPresentation[^]*attach\(\): boolean[^]*submit\(\): boolean/,
    "fenced presentation phases must be represented by monotonic transitions instead of writable booleans");
  assert.doesNotMatch(renderer, /pendingLiveSvoPresentation\s*=\s*\{[^}]*attached:/,
    "the sparse presentation gate must not be restricted to renderer-only scenes");
  assert.match(renderer, /state:"ready",label:"Live sparse scene source ready",adapter:this\.adapterName,resource:liveSvoSceneResourcePlugin/,
    "renderer-only source attachment must explicitly close its own resource activity");
  assert.match(renderer, /pendingLiveSvo\.attached[^]*pendingLiveSvo\.solver === readyGPUFluid[^]*svoEncoded/,
    "the first live presentation must require an attached source and successful SVO encoding");
  assert.match(renderer, /queue\.onSubmittedWorkDone\(\)\.then\([^]*settleLiveSvoPresentation\(initialLiveSvoSubmission\)/,
    "ready must be published only after the first sparse garden frame completes");
  assert.match(renderer, /state: "ready", label: "Live SVO renderer ready", adapter: this\.adapterName, resource: svoPresentationResourcePlugin/,
    "the first sparse frame must explicitly close the presentation resource activity");
  assert.doesNotMatch(liveSource, /WebGPUUniformEulerianSolver/);
  assert.match(liveSource, /fluid authority intentionally absent/);
  assert.match(liveSource, /new OctreeSparseBrickWorld/);
  assert.doesNotMatch(liveSource, /emptyPhi|onSubmittedWorkDone/,
    "renderer-only startup must not bake or fence scene content");
});

test("fluid-disabled live GPU sources initialize independently of the selected solver method", () => {
  const liveScene = getScenePreset("garden-svo-lighting").create();
  const fluidScene = getScenePreset("garden-pond").create();

  assert.equal(canInitializeGPUSceneSource(liveScene, "cpu-reference"), true,
    "renderer-owned live SVO construction must not require a fluid solver factory");
  assert.equal(canInitializeGPUSceneSource(fluidScene, "cpu-reference"), false,
    "a fluid-enabled scene must not enter GPU construction without a GPU solver factory");
  assert.equal(canInitializeGPUSceneSource(liveScene, "octree"), true);
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
  assert.equal(gpuSceneSolverKey(scene, config), changedBodies,
    "authored lighting is hot-published and must not rebuild the shared SVO arenas");
});

test("live SVO scenes lazily expose raw-voxel inspection records", () => {
  const liveSource = readFileSync(new URL("../lib/webgpu-live-svo-scene.ts", import.meta.url), "utf8");

  assert.match(liveSource, /get sparseVoxelRenderSource\(\)/,
    "the renderer's existing inspection attachment must work for fluid-free scenes");
  assert.match(liveSource, /this\.world\.ensureInspectionSource\(\)/,
    "raw records should be derived from the same authoritative static octree");
  assert.match(liveSource, /worldBytes - this\.accountedWorldBytes/,
    "lazy inspection allocation must be reflected in renderer telemetry");

  const constructorStart = liveSource.indexOf("private constructor(");
  const getterStart = liveSource.indexOf("get sparseVoxelRenderSource()", constructorStart);
  assert.doesNotMatch(liveSource.slice(constructorStart, getterStart), /ensureInspectionSource\(\)/,
    "normal smooth SVO startup must not allocate capacity-sized debug records");
});
