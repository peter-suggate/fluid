import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { getScenePreset } from "../lib/scenes";
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

test("each authoring input reaches exactly the tiers that must answer it", () => {
  const base = cloneScene(defaultScene);
  const structural = gpuSceneStructuralKey(base, config);
  const seed = gpuSceneSeedKey(base);
  const uniform = gpuSceneUniformKey(base);

  // Most inputs still land in one tier. The container extents are the
  // exception, and deliberately so: the structural tier keys the lattice in
  // *cells*, while the seed tier keys the extents in metres. An extent edit on
  // its own moves both — the lattice grows and each cell is a different size —
  // so it rebuilds. Scaling the world moves the extent and the cell size
  // together, which leaves the lattice exactly where it was and reaches the
  // seed tier alone. That is the whole reason a world scale is a re-seed
  // rather than a rebuild, so it is asserted here beside the tiers it rests on.
  const cases: ReadonlyArray<{ label: string; tiers: readonly ("structural" | "seed" | "uniform")[]; mutate: (scene: SceneDescription) => void }> = [
    { label: "container width", tiers: ["structural", "seed"], mutate: (scene) => { scene.container.width_m += 0.3; } },
    {
      label: "world scale",
      tiers: ["seed"],
      mutate: (scene) => {
        scene.container.width_m *= 2; scene.container.height_m *= 2; scene.container.depth_m *= 2;
        scene.voxelDomain.finestCellSize_m *= 2;
      },
    },
    { label: "container top", tiers: ["structural"], mutate: (scene) => { scene.container.top = scene.container.top === "open" ? "closed" : "open"; } },
    { label: "wall mode", tiers: ["structural"], mutate: (scene) => { scene.container.fluidWallMode = scene.container.fluidWallMode === "no-slip" ? "free-slip" : "no-slip"; } },
    { label: "depth boundary", tiers: ["structural"], mutate: (scene) => { scene.container.depthBoundary = "symmetry"; } },
    { label: "voxel domain", tiers: ["structural"], mutate: (scene) => { scene.voxelDomain.finestCellSize_m *= 0.5; } },
    { label: "fill fraction", tiers: ["seed"], mutate: (scene) => { scene.container.fillFraction = 0.5 * (scene.container.fillFraction + 1); } },
    // Emptying the roster crosses the one boundary that is genuinely
    // allocation-shaped — solids present or absent — and so moves both tiers.
    { label: "rigid bodies", tiers: ["seed", "uniform"], mutate: (scene) => { scene.rigidBodies = []; } },
    { label: "initial condition", tiers: ["seed"], mutate: (scene) => { scene.fluid.initialCondition = scene.fluid.initialCondition === "dam-break" ? "tank-fill" : "dam-break"; } },
    { label: "dam origin", tiers: ["seed"], mutate: (scene) => { scene.fluid.initialDamBreakOrigin_m = { x: 0.1, y: 0, z: 0 }; } },
    { label: "brick seeds", tiers: ["seed"], mutate: (scene) => { scene.fluid.initialBrickSeeds_m = [{ x: 0, y: 0.1, z: 0 }]; } },
    { label: "seed additivity", tiers: ["seed"], mutate: (scene) => { scene.fluid.initialBrickSeedsAdditive = true; } },
    { label: "analytic liquid volume", tiers: ["seed"], mutate: (scene) => {
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.5, z: 0 }, radius_m: 0.1 }];
    } },
    { label: "terrain", tiers: ["seed"], mutate: (scene) => { scene.terrain = { baseHeight_m: 0.1, features: [] }; } },
    { label: "density", tiers: ["uniform"], mutate: (scene) => { scene.fluid.density_kg_m3 += 10; } },
    { label: "viscosity", tiers: ["uniform"], mutate: (scene) => { scene.fluid.dynamicViscosity_Pa_s += 0.01; } },
    { label: "surface tension", tiers: ["uniform"], mutate: (scene) => { scene.fluid.surfaceTension_N_m += 0.01; } },
    { label: "gravity", tiers: ["uniform"], mutate: (scene) => { scene.fluid.gravity_m_s2.y *= 0.5; } },
    { label: "shared timestep", tiers: ["uniform"], mutate: (scene) => { scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.006; } },
  ];

  for (const { label, tiers, mutate } of cases) {
    const next = edited(mutate);
    const changed = {
      structural: gpuSceneStructuralKey(next, config) !== structural,
      seed: gpuSceneSeedKey(next) !== seed,
      uniform: gpuSceneUniformKey(next) !== uniform,
    };
    for (const tier of ["structural", "seed", "uniform"] as const) {
      assert.equal(changed[tier], tiers.includes(tier),
        `${label} must ${tiers.includes(tier) ? "change" : "not change"} the ${tier} tier`);
    }
    // Structural and seed changes rebuild or re-seed; scalars are adopted hot.
    const rebuilds = gpuSceneSolverKey(next, config) !== gpuSceneSolverKey(cloneScene(defaultScene), config);
    assert.equal(rebuilds, !(tiers.length === 1 && tiers[0] === "uniform"), `${label} must ${tiers[0] === "uniform" ? "not " : ""}rebuild`);
  }
});

test("aiming the hose is a params write; resizing it is a re-seed", () => {
  // The nozzle used to be one JSON blob in the seed tier, so nudging the arrow
  // restarted the simulation. What an allocation is actually sized from is the
  // volume the hose will deliver — `fluidFootprint` integrates radius, speed
  // and the timing window — and never where it points.
  const base = getScenePreset("hose-tank").create();
  const inflow = base.fluid.inflow;
  assert.ok(inflow, "this fixture must actually carry a nozzle");
  const withInflow = (patch: Partial<NonNullable<SceneDescription["fluid"]["inflow"]>>) =>
    ({ ...base, fluid: { ...base.fluid, inflow: { ...inflow, ...patch } } });

  const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
  const aimedAt = (x: number, y: number, z: number) => {
    const norm = speed / Math.hypot(x, y, z);
    return { velocity_m_s: { x: x * norm, y: y * norm, z: z * norm } };
  };
  const hot: ReadonlyArray<{ label: string; scene: SceneDescription }> = [
    { label: "moved", scene: withInflow({ center_m: { ...inflow.center_m, x: inflow.center_m.x + 0.05 } }) },
    { label: "re-aimed", scene: withInflow(aimedAt(0, -1, 0.4)) },
    { label: "lengthened", scene: withInflow({ length_m: inflow.length_m * 1.5 }) },
  ];
  for (const { label, scene } of hot) {
    assert.equal(gpuSceneSeedKey(scene), gpuSceneSeedKey(base), `a ${label} hose must not re-seed`);
    assert.notEqual(gpuSceneUniformKey(scene), gpuSceneUniformKey(base), `a ${label} hose must reach the solver`);
    assert.equal(gpuSceneSolverKey(scene, config), gpuSceneSolverKey(base, config),
      `a ${label} hose must not rebuild`);
  }

  const budgeted: ReadonlyArray<{ label: string; scene: SceneDescription }> = [
    { label: "widened", scene: withInflow({ radius_m: inflow.radius_m * 1.5 }) },
    { label: "sped up", scene: withInflow({ velocity_m_s: { x: 0, y: -2 * speed, z: 0 } }) },
    { label: "re-timed", scene: withInflow({ end_s: inflow.end_s + 1 }) },
    { label: "removed", scene: { ...base, fluid: { ...base.fluid, inflow: undefined } } },
  ];
  for (const { label, scene } of budgeted) {
    assert.notEqual(gpuSceneSeedKey(scene), gpuSceneSeedKey(base),
      `a ${label} hose changes the volume the arena is budgeted for`);
  }

  // Turning the hose at constant speed delivers the same water, so it must not
  // move the budget even though the velocity vector changed.
  const turned = withInflow(aimedAt(1, 0, 0));
  assert.equal(gpuSceneSeedKey(turned), gpuSceneSeedKey(base));
});

test("editing a body that already exists is a buffer write", () => {
  // Every step writes the whole roster through `syncBodies` and re-encodes the
  // solid vertex SDF from that buffer, so what a body *is* was never what an
  // allocation was shaped by — only how many there are, whether any move, and
  // the owner numbering the render source derives from the count.
  const base = getScenePreset("dam-break-boxes").create();
  assert.ok(base.rigidBodies.length > 0, "this fixture must already carry solids");
  const replacingFirst = (patch: Partial<(typeof base.rigidBodies)[number]>) =>
    ({ ...base, rigidBodies: [{ ...base.rigidBodies[0], ...patch }, ...base.rigidBodies.slice(1)] });

  const hot: ReadonlyArray<{ label: string; scene: SceneDescription }> = [
    { label: "repositioned", scene: replacingFirst({ position_m: { x: 0.2, y: 0.3, z: 0 } }) },
    { label: "reshaped", scene: replacingFirst({ dimensions_m: { x: 0.2, y: 0.2, z: 0.2 } }) },
    { label: "re-materialed", scene: replacingFirst({ friction: 0.1, restitution: 0.9 }) },
    { label: "made denser", scene: replacingFirst({ density_kg_m3: 2000 }) },
  ];
  for (const { label, scene } of hot) {
    assert.equal(gpuSceneSeedKey(scene), gpuSceneSeedKey(base), `a ${label} body must not re-seed`);
    assert.notEqual(gpuSceneUniformKey(scene), gpuSceneUniformKey(base), "it must still reach the solver");
    assert.equal(gpuSceneSolverKey(scene, config), gpuSceneSolverKey(base, config));
  }

  // The three facts that are not about a body but about the roster, asserted as
  // such rather than left implicit.
  const added = { ...base, rigidBodies: [...base.rigidBodies, { ...base.rigidBodies[0], id: "dropped-in" }] };
  assert.notEqual(gpuSceneSeedKey(added), gpuSceneSeedKey(base),
    "the count numbers every environment proxy after the bodies, so adding one renumbers the render source");
  assert.notEqual(gpuSceneSeedKey({ ...base, rigidBodies: [] }), gpuSceneSeedKey(base),
    "removing the last body releases the dense solid field");
  const pinned = { ...base, rigidBodies: base.rigidBodies.map((body) => ({ ...body, motion: "static" as const })) };
  assert.notEqual(gpuSceneSeedKey(pinned), gpuSceneSeedKey(base),
    "pinning every body drops the moving-immersed-boundary iteration");
});

test("a solver that adopts a moved nozzle rederives the boundary it derived from it", () => {
  // The step params carry an outlet centre and an aperture scale that are
  // computed on the host from the nozzle's placement. Swapping the scene alone
  // would move the arrow the user is dragging and leave the water coming out of
  // where it used to be — which reads as the aim being ignored, not as a stale
  // derivation.
  const solver = readFileSync(new URL("../lib/webgpu-octree-eulerian.ts", import.meta.url), "utf8");
  const adopt = solver.slice(solver.indexOf("applySceneUniforms(scene: SceneDescription)"));
  assert.match(adopt.slice(0, 600), /this\.inflowBoundary = scene\.fluid\.inflow/);
});

test("a world scale is answered by a re-seed, never a rebuild", () => {
  // `tryReseedGPUFluid` gates the warm path on the structural key alone, so
  // this equality is the single fact that keeps a world scale off the ~350
  // pipeline rebuild path.
  const base = cloneScene(defaultScene);
  const scaled = edited((scene) => {
    scene.container.width_m *= 2; scene.container.height_m *= 2; scene.container.depth_m *= 2;
    scene.voxelDomain.finestCellSize_m *= 2;
  });
  assert.equal(gpuSceneStructuralKey(scaled, config), gpuSceneStructuralKey(base, config));
  assert.notEqual(gpuSceneSolverKey(scaled, config), gpuSceneSolverKey(base, config),
    "the solver must still be replaced-or-re-seeded: it is simulating a different physical size");
});

test("render-only authoring never reaches the solver identity", () => {
  const base = cloneScene(defaultScene);
  // Scenery extends the render domain only; placing a prop must not rebuild.
  const withProp = edited((scene) => {
    scene.scenery = {
      palettes: {},
      nodes: [
        {
          kind: "room-shell", id: "shell", materialModel: "room",
          floor: { colorLinear: [.6, .6, .6] },
          wall: { colorLinear: [.55, .55, .55] },
          ceiling: { colorLinear: [.65, .65, .65] },
        },
        {
          kind: "box", id: "box-1", place: { units: "metres", position: { x: 0, y: 0.1, z: 0 } },
          halfSize: { x: 0.05, y: 0.05, z: 0.05 }, material: { colorLinear: [0.4, 0.4, 0.4] },
        },
      ],
    };
  });
  assert.equal(gpuSceneSeedKey(withProp), gpuSceneSeedKey(base));
  assert.equal(gpuSceneUniformKey(withProp), gpuSceneUniformKey(base));
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
  const solver = readFileSync(new URL("../lib/webgpu-octree-eulerian.ts", import.meta.url), "utf8");
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const surface = readFileSync(new URL("../lib/webgpu-quadtree-builder.ts", import.meta.url), "utf8");

  assert.match(renderer, /tryReseedGPUFluid\(scene,config,key,presentationMode\)\)return undefined/,
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
