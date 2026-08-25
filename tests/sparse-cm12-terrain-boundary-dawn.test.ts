import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene, validateScene } from "../lib/core/model";
import {
  createTallCellsHillsideDamBreakScene,
  TALL_CELLS_FLOOD_GRID,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function tallCellsMetrics(fields: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>) {
  const [nx, ny, nz] = TALL_CELLS_FLOOD_GRID;
  const massByX = new Float64Array(nx);
  let mass = 0, closedMass = 0, capacityExcess = 0;
  let cutCapacityExcess = 0, fullCapacityExcess = 0, maximumDensity = 0;
  let maximumDensityCell: readonly number[] = [-1, -1, -1];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const at = x + nx * (y + ny * z);
      const rho = Math.max(0, fields.density[at]!);
      const open = fields.solidOpenFraction[at]!;
      mass += rho; massByX[x] += rho;
      if (rho > maximumDensity) {
        maximumDensity = rho; maximumDensityCell = [x, y, z];
      }
      if (open <= 1e-6) closedMass += rho;
      const excess = Math.max(0, rho - open);
      capacityExcess += excess;
      if (open < 1 - 1e-6) cutCapacityExcess += excess;
      else fullCapacityExcess += excess;
    }
  }
  const quantileFront = (fraction: number) => {
    const target = fraction * mass;
    let cumulative = 0;
    for (let x = 0; x < nx; x += 1) {
      cumulative += massByX[x]!;
      if (cumulative >= target) return x;
    }
    return nx - 1;
  };
  let thresholdFront = -1;
  for (let x = 0; x < nx; x += 1) {
    if (massByX[x]! >= 0.01) thresholdFront = x;
  }
  return { mass, closedMass, capacityExcess, cutCapacityExcess,
    fullCapacityExcess, maximumDensity, maximumDensityCell, thresholdFront,
    front99: quantileFront(0.99), front999: quantileFront(0.999) };
}

function terrainCutCellDamScene() {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "sparse-cm12-terrain-cut-cell-dam";
  scene.container = {
    ...scene.container,
    width_m: 0.8,
    height_m: 0.8,
    depth_m: 0.8,
    fillFraction: 0.4 * 0.3 * 0.4 / (0.8 ** 3),
    top: "open",
    fluidWallMode: "free-slip",
    vessel: "none",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: 0.05 };
  // Deliberately halfway through a finest cell: V_i must contain both zero and
  // partial capacities, rather than degenerating to an aligned binary floor.
  scene.terrain = {
    baseHeight_m: 0.125,
    features: [],
  };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { x: 0.4, y: 0.3, z: 0.4 };
  scene.fluid.initialDamBreakOrigin_m = { x: 0, y: 0.125, z: 0 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  scene.rigidBodies = [];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
  return scene;
}

dawnTest("Sparse CM12 couples terrain voxels through CM12 cut-cell capacities",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-terrain-boundary-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = process.env.FLUID_SCENE === "tall-cells-hillside-dam-break"
        ? createTallCellsHillsideDamBreakScene()
        : terrainCutCellDamScene();
      assert.deepEqual(validateScene(scene), []);
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          timeStep: "paper",
        }, () => {},
      );
      await solver.waitForSimulationReady();
      const tallCells = process.env.FLUID_SCENE === "tall-cells-hillside-dam-break";
      const initialTallMetrics = tallCells
        ? tallCellsMetrics(await solver.readDiagnosticFields()) : undefined;
      // The exact Tall Cells rung must survive long enough for the released
      // front to leave its authored reservoir and exercise dynamic world pages
      // on the slope; the focused 16^3 cut-cell probe stays intentionally tiny.
      const requestedSteps = Number(process.env.FLUID_TERRAIN_STEPS);
      const steps = Number.isSafeInteger(requestedSteps) && requestedSteps > 0
        ? requestedSteps : tallCells ? 30 : 8;
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 2 === 0) await device.queue.onSubmittedWorkDone();
        if (tallCells && process.env.FLUID_TERRAIN_TRACE === "1"
          && (step <= 4 || step % 10 === 0 || step === steps)) {
          await device.queue.onSubmittedWorkDone();
          const metrics = tallCellsMetrics(await solver.readDiagnosticFields());
          const activity = await solver.readGPUActivityPolicy();
          const active = activity.bricks.filter((brick) => brick.active);
          const activeX = active.reduce((maximum, brick) =>
            Math.max(maximum, brick.coordinate[0]), -1);
          process.stderr.write(`${JSON.stringify({ step, ...metrics,
            relativeMass: metrics.mass / initialTallMetrics!.mass,
            activeX, activeBricks: active.length,
            topologyGeneration: activity.acceptedTopologyGeneration,
            newlyActivated: activity.newlyActivatedBrickCount,
            faultFlags: activity.faultFlags })}\n`);
        }
      }
      await device.queue.onSubmittedWorkDone();

      const fields = await solver.readDiagnosticFields();
      let partialCells = 0, closedCells = 0, maximumClosedDensity = 0;
      let fluidMass = 0;
      for (let cell = 0; cell < fields.density.length; cell += 1) {
        const open = fields.solidOpenFraction[cell]!;
        const density = fields.density[cell]!;
        assert.ok(Number.isFinite(open) && open >= 0 && open <= 1);
        assert.ok(Number.isFinite(density) && density >= 0);
        if (open > 1e-6 && open < 1 - 1e-6) partialCells += 1;
        if (open <= 1e-6) {
          closedCells += 1;
          maximumClosedDensity = Math.max(maximumClosedDensity, density);
        }
        fluidMass += density;
      }
      if (!tallCells) assert.ok(partialCells > 0,
        "the half-cell terrain height must publish partial CM12 capacities");
      assert.ok(closedCells > 0,
        "resident bricks crossing the terrain must publish fully solid cells");
      assert.ok(maximumClosedDensity <= 1e-6,
        `terrain-solid cells retained fluid density ${maximumClosedDensity}`);
      assert.ok(fluidMass > 1,
        "the terrain boundary must retain a material fluid body above it");
      const frame = await solver.sparseWorldTrace.readFrameControlQA();
      assert.equal(frame.fault, 0);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
