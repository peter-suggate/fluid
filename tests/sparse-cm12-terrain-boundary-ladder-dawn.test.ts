import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene, type SceneDescription, validateScene } from
  "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  terrainCellSolidFraction,
  terrainColumnHeights,
  type TerrainDescription,
} from "../lib/core/terrain";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";

const CELL_M = 0.05;
const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

interface TerrainBoundaryRung {
  readonly name: string;
  readonly scene: SceneDescription;
  readonly steps: number;
  readonly dt_s: number;
  readonly timeStep: "paper" | "scene";
  readonly maximumMassDrift: number;
}

function flatTerrain(height_m: number): TerrainDescription {
  return { baseHeight_m: height_m, features: [] };
}

function singleStepTerrain(nx: number, nz: number): TerrainDescription {
  const heights_m: number[] = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    heights_m.push(x < nx / 2 ? CELL_M : 3 * CELL_M);
  }
  return {
    baseHeight_m: CELL_M,
    features: [],
    grid: {
      kind: "grid",
      // Put authored nodes at solver-column centres. The bake therefore has a
      // genuinely discontinuous one-column step, rather than a bilinear ramp.
      origin_m: { x: (-nx / 2 + 0.5) * CELL_M, z: (-nz / 2 + 0.5) * CELL_M },
      spacing_m: CELL_M,
      size: { nx, nz },
      heights_m,
    },
  };
}

function terrainScene(
  name: string,
  nx: number,
  terrain: TerrainDescription,
  initial: "hydrostatic" | "left-release",
  gravity = true,
  dt_s = CM12_PAPER_DT_S,
): SceneDescription {
  // VEX2 currently requires a complete 2x2x2 B8 neighbourhood. This 16^3
  // lattice is therefore the smallest construction-valid Sparse CM12 rung.
  const ny = 16, nz = 16;
  const scene = cloneScene(defaultScene);
  scene.sceneId = `sparse-cm12-terrain-ladder-${name}`;
  scene.container = {
    ...scene.container,
    width_m: nx * CELL_M,
    height_m: ny * CELL_M,
    depth_m: nz * CELL_M,
    fillFraction: 0.75,
    top: "open",
    fluidWallMode: "free-slip",
    vessel: "none",
  };
  scene.voxelDomain = { finestCellSize_m: CELL_M, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: CELL_M };
  scene.terrain = terrain;
  scene.rigidBodies = [];
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  if (!gravity) scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  if (initial === "hydrostatic") {
    scene.fluid.initialCondition = "tank-fill";
    delete scene.fluid.initialDamBreakDimensions_m;
    delete scene.fluid.initialDamBreakOrigin_m;
  } else {
    scene.fluid.initialCondition = "dam-break";
    scene.fluid.initialDamBreakDimensions_m = {
      x: 0.5 * nx * CELL_M,
      y: 5 * CELL_M,
      z: nz * CELL_M,
    };
    scene.fluid.initialDamBreakOrigin_m = { x: 0, y: CELL_M, z: 0 };
    scene.container.fillFraction = scene.fluid.initialDamBreakDimensions_m.x
      * scene.fluid.initialDamBreakDimensions_m.y
      * scene.fluid.initialDamBreakDimensions_m.z
      / (scene.container.width_m * scene.container.height_m
        * scene.container.depth_m);
  }
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
  return scene;
}

export function sparseCM12TerrainBoundaryLadder(): readonly TerrainBoundaryRung[] {
  const noTerrainControl = terrainScene("no-terrain-control", 16,
    flatTerrain(0), "hydrostatic", true, 0.004);
  delete noTerrainControl.terrain;
  const noTerrainRelease = terrainScene("no-terrain-release", 16,
    flatTerrain(0), "left-release", true, 0.004);
  delete noTerrainRelease.terrain;
  return [
    {
      name: "minimum-valid-aligned-flat-quiescent",
      scene: terrainScene("flat-aligned", 16, flatTerrain(CELL_M), "hydrostatic",
        false),
      steps: 8,
      dt_s: CM12_PAPER_DT_S,
      timeStep: "paper",
      maximumMassDrift: 0.05,
    },
    {
      name: "minimum-valid-fractional-flat-quiescent",
      scene: terrainScene("flat-fractional", 16, flatTerrain(1.5 * CELL_M),
        "hydrostatic", false),
      steps: 8,
      dt_s: CM12_PAPER_DT_S,
      timeStep: "paper",
      maximumMassDrift: 0.05,
    },
    {
      name: "no-terrain-hydrostatic-control",
      scene: noTerrainControl,
      steps: 30,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.05,
    },
    {
      name: "aligned-flat-hydrostatic",
      scene: terrainScene("flat-hydrostatic", 16, flatTerrain(CELL_M),
        "hydrostatic", true, 0.004),
      steps: 30,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.05,
    },
    {
      name: "fractional-flat-hydrostatic",
      scene: terrainScene("fractional-flat-hydrostatic", 16,
        flatTerrain(1.5 * CELL_M), "hydrostatic", true, 0.004),
      steps: 30,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.1,
    },
    {
      name: "two-brick-single-step-hydrostatic",
      scene: terrainScene("single-step-static", 16, singleStepTerrain(16, 16),
        "hydrostatic", true, 0.004),
      steps: 30,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.1,
    },
    {
      name: "no-terrain-release-control",
      scene: noTerrainRelease,
      steps: 100,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.15,
    },
    {
      name: "two-brick-single-step-release",
      scene: terrainScene("single-step-release", 16, singleStepTerrain(16, 16),
        "left-release", true, 0.004),
      steps: 100,
      dt_s: 0.004,
      timeStep: "scene",
      maximumMassDrift: 0.15,
    },
  ];
}

test("terrain boundary ladder has analytic flat, fractional, and step capacities", () => {
  const rungs = sparseCM12TerrainBoundaryLadder();
  const aligned = rungs.find((rung) => rung.name === "aligned-flat-hydrostatic")!;
  const fractional = rungs.find((rung) =>
    rung.name === "fractional-flat-hydrostatic")!;
  const stepped = rungs.find((rung) =>
    rung.name === "two-brick-single-step-hydrostatic")!;
  const dimensions = sceneLatticeDimensions(aligned.scene);
  assert.deepEqual(dimensions, [16, 16, 16],
    "16^3 is the smallest complete 2x2x2 B8 construction rung");

  const alignedHeights = terrainColumnHeights(aligned.scene, 16, 16);
  assert.ok(alignedHeights.every((height) => Math.abs(height - CELL_M) < 1e-7));
  assert.equal(1 - terrainCellSolidFraction(CELL_M, 0, CELL_M), 0);
  assert.equal(1 - terrainCellSolidFraction(CELL_M, CELL_M, CELL_M), 1);

  const fractionalHeights = terrainColumnHeights(fractional.scene, 16, 16);
  assert.ok(fractionalHeights.every((height) =>
    Math.abs(height - 1.5 * CELL_M) < 1e-7));
  assert.ok(Math.abs((1 - terrainCellSolidFraction(
    1.5 * CELL_M, CELL_M, CELL_M)) - 0.5) < 1e-12);

  const stepHeights = terrainColumnHeights(stepped.scene, 16, 16);
  for (let z = 0; z < 16; z += 1) for (let x = 0; x < 16; x += 1) {
    const expected = x < 8 ? CELL_M : 3 * CELL_M;
    assert.ok(Math.abs(stepHeights[x + 16 * z]! - expected) < 1e-7);
  }

  for (const rung of [aligned, fractional, stepped]) {
    const atlas = initializeSparseBrickAtlasFromScene(rung.scene, {
      finestDimensions: sceneLatticeDimensions(rung.scene),
      brickFineResolution: 8,
    });
    assert.ok(atlas.bricks.filter((brick) => brick.coordinate[1] === 0)
      .every((brick) => brick.resolution === 8),
    `${rung.name}: every terrain-intersecting authored brick must start fine`);
  }
});

interface BoundaryMetrics {
  readonly mass: number;
  readonly capacityExcess: number;
  readonly closedCellMass: number;
  readonly solidSamplePenetration: number;
  readonly deepestInterfacePenetrationFine: number;
  readonly bottomResolutions: readonly number[];
}

async function boundaryMetrics(
  solver: WebGPUAdaptiveMassSolver,
  scene: SceneDescription,
): Promise<BoundaryMetrics> {
  const dimensions = sceneLatticeDimensions(scene);
  const [nx, ny, nz] = dimensions;
  const fields = await solver.readDiagnosticFields();
  const activity = await solver.readGPUActivityPolicy();
  // This is the exact scalar used by `presentationPhiAt`: compact Sparse CM12
  // publishes buffer-native pages, while the generic 3D texture is only 1^3.
  const heights = terrainColumnHeights(scene, nx, nz);
  const phi = Float32Array.from(fields.density, (rho, at) => {
    const x = at % nx, yz = Math.floor(at / nx);
    const y = yz % ny, z = Math.floor(yz / ny);
    const liquidPhi = (0.5 - Math.min(1, Math.max(0,
      rho / Math.max(fields.solidOpenFraction[at]!, 1e-6)))) * 4 * CELL_M;
    const terrainPhi = scene.terrain
      ? heights[x + nx * z]! - (y + 0.5) * CELL_M
      : Number.NEGATIVE_INFINITY;
    return Math.max(liquidPhi, terrainPhi);
  });
  let mass = 0, capacityExcess = 0, closedCellMass = 0;
  let solidSamplePenetration = 0, deepestInterfacePenetrationFine = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const terrainFine = heights[x + nx * z]! / CELL_M;
    for (let y = 0; y < ny; y += 1) {
      const at = x + nx * (y + ny * z);
      const rho = fields.density[at]!, open = fields.solidOpenFraction[at]!;
      mass += rho;
      if (open < 1 - 1e-6) capacityExcess += Math.max(0, rho - open);
      if (open <= 1e-6) closedCellMass += rho;
      if (y + 0.5 < terrainFine - 1e-5 && phi[at]! < 0) {
        solidSamplePenetration += -phi[at]!;
      }
    }
    // Locate the lowest reconstructed solid-to-liquid crossing. This measures
    // the rendered boundary itself, including fractional cut cells where rho
    // can satisfy V_i while a cell-centred liquid level set still cuts below H.
    for (let y = 0; y + 1 < ny; y += 1) {
      const lower = phi[x + nx * (y + ny * z)]!;
      const upper = phi[x + nx * (y + 1 + ny * z)]!;
      if (!(lower >= 0 && upper < 0)) continue;
      const crossingFine = y + 0.5 + lower / Math.max(lower - upper, 1e-12);
      deepestInterfacePenetrationFine = Math.max(deepestInterfacePenetrationFine,
        terrainFine - crossingFine);
      break;
    }
  }
  return {
    mass, capacityExcess, closedCellMass, solidSamplePenetration,
    deepestInterfacePenetrationFine,
    bottomResolutions: activity.bricks.filter((brick) =>
      brick.active && brick.coordinate[1] === 0).map((brick) =>
      brick.acceptedResolution),
  };
}

dawnTest("Sparse CM12 terrain boundary ladder is impermeable",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-terrain-boundary-ladder-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const selectedRung = process.env.FLUID_TERRAIN_LADDER_RUNG;
      const rungs = sparseCM12TerrainBoundaryLadder().filter((rung) =>
        !selectedRung || rung.name.includes(selectedRung));
      assert.ok(rungs.length > 0, `unknown terrain ladder rung ${selectedRung}`);
      for (const rung of rungs) {
        assert.deepEqual(validateScene(rung.scene), [], rung.name);
        const solver = await WebGPUAdaptiveMassSolver.createAsync(
          device, rung.scene, "balanced", undefined, {
            resolutionMode: "adaptive",
            brickFineResolution: 8,
            timeStep: rung.timeStep,
          }, () => {},
        );
        try {
          await solver.waitForSimulationReady();
          const initial = await boundaryMetrics(solver, rung.scene);
          const requestedSteps = Number(process.env.FLUID_TERRAIN_LADDER_STEPS);
          const steps = Number.isSafeInteger(requestedSteps) && requestedSteps > 0
            ? requestedSteps : rung.steps;
          const checkpoints: Array<{ step: number; metrics: BoundaryMetrics }> = [
            { step: 0, metrics: initial },
          ];
          for (let step = 1; step <= steps; step += 1) {
            assert.equal(solver.advanceTo(step * rung.dt_s, []), true,
              `${rung.name}: advance ${step}`);
            if (step <= 4 || step === 8 || step === 16 || step === steps) {
              await device.queue.onSubmittedWorkDone();
              checkpoints.push({
                step,
                metrics: await boundaryMetrics(solver, rung.scene),
              });
            }
          }
          await device.queue.onSubmittedWorkDone();
          const final = checkpoints.at(-1)!.metrics;
          const evidence = JSON.stringify(checkpoints);
          assert.ok(final.capacityExcess <= 1e-5,
            `${rung.name}: rho exceeded V_i by ${final.capacityExcess}; ${evidence}`);
          assert.ok(final.closedCellMass <= 1e-6,
            `${rung.name}: closed terrain retained ${final.closedCellMass} mass; ${evidence}`);
          assert.ok(final.solidSamplePenetration <= 1e-6,
            `${rung.name}: presentation field is liquid inside analytic solid; ${evidence}`);
          assert.ok(final.deepestInterfacePenetrationFine <= 0.1,
            `${rung.name}: interface penetrated terrain by `
              + `${final.deepestInterfacePenetrationFine} fine cells; ${evidence}`);
          if (rung.scene.terrain) {
            assert.ok(final.bottomResolutions.every((resolution) => resolution === 8),
              `${rung.name}: terrain boundary demoted; ${evidence}`);
          }
          assert.ok(Math.abs(final.mass - initial.mass) <= rung.maximumMassDrift,
            `${rung.name}: mass drifted by ${final.mass - initial.mass} fine cells; `
              + evidence);
        } finally {
          solver.destroy();
          await device.queue.onSubmittedWorkDone();
        }
      }
      assert.deepEqual(validationErrors, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
