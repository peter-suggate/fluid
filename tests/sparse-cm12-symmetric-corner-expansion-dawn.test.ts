import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const SYMMETRY_STEPS = 8;
const DENSITY_D4_LIMIT = 6e-3;
const VELOCITY_D4_LIMIT_M_S = 5e-3;
const PRESSURE_D4_LIMIT_PA = 5;

function scalarD4Error(field: ArrayLike<number>, nx: number, ny: number,
  nz: number): number {
  let maximum = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const source = field[x + nx * (y + ny * z)]!;
      for (const [tx, tz] of [[nx - 1 - x, z], [x, nz - 1 - z], [z, x]]) {
        maximum = Math.max(maximum,
          Math.abs(source - field[tx! + nx * (y + ny * tz!)]!));
      }
    }
  }
  return maximum;
}

function velocityD4Error(field: ArrayLike<number>, nx: number, ny: number,
  nz: number): number {
  let maximum = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const sourceAt = 4 * (x + nx * (y + ny * z));
      const source = [field[sourceAt]!, field[sourceAt + 1]!, field[sourceAt + 2]!];
      const comparisons = [
        { coordinate: [nx - 1 - x, z], expected: [-source[0]!, source[1]!, source[2]!] },
        { coordinate: [x, nz - 1 - z], expected: [source[0]!, source[1]!, -source[2]!] },
        { coordinate: [z, x], expected: [source[2]!, source[1]!, source[0]!] },
      ] as const;
      for (const comparison of comparisons) {
        const targetAt = 4 * (comparison.coordinate[0]
          + nx * (y + ny * comparison.coordinate[1]));
        for (let component = 0; component < 3; component += 1) {
          maximum = Math.max(maximum,
            Math.abs(comparison.expected[component]! - field[targetAt + component]!));
        }
      }
    }
  }
  return maximum;
}

dawnTest("symmetric expansion allocates and wets sparse corner tiles",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-symmetric-corner-expansion-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache",
      ]);
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

      const horizontalCells = 32;
      const scene = createSymmetricExpansionScene();
      scene.voxelDomain.finestCellSize_m = scene.container.width_m / horizontalCells;
      const brickSize = scene.voxelDomain.brickSize_cells;
      const brickDimensions = [horizontalCells / brickSize,
        horizontalCells / (2 * brickSize), horizontalCells / brickSize] as const;
      scene.fluid.initialBrickSeeds_m = [];
      for (let z = brickDimensions[2] / 4; z < 3 * brickDimensions[2] / 4; z += 1)
        for (let y = 0; y < brickDimensions[1] / 2; y += 1)
          for (let x = brickDimensions[0] / 4; x < 3 * brickDimensions[0] / 4; x += 1) {
            scene.fluid.initialBrickSeeds_m.push({
              x: -0.5 * scene.container.width_m
                + (x + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
              y: (y + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
              z: -0.5 * scene.container.depth_m
                + (z + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
            });
          }
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;

      solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          presentationPageResolution: 8,
          timeStep: "paper",
          pressureIterations: 64,
        }, () => {},
      );
      await solver.waitForSimulationReady();

      const horizontalCorner = (coordinate: readonly number[]) =>
        (coordinate[0] === 0 || coordinate[0] === brickDimensions[0] - 1)
        && (coordinate[2] === 0 || coordinate[2] === brickDimensions[2] - 1);
      const initial = await solver.readGPUActivityPolicy();
      assert.equal(initial.bricks.filter((brick) => brick.active).length, 4);
      assert.equal(initial.bricks.filter((brick) =>
        brick.active && horizontalCorner(brick.coordinate)).length, 0,
      "corner tiles must begin absent rather than hiding a preallocated apron");

      for (let step = 1; step <= SYMMETRY_STEPS; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        if (step !== 1 && step !== 3) continue;
        const activity = await solver.readGPUActivityPolicy();
        const corners = activity.bricks.filter((brick) =>
          brick.active && horizontalCorner(brick.coordinate));
        if (step === 1) {
          assert.equal(activity.bricks.filter((brick) => brick.active).length, 16,
            "the first demand wave must publish the symmetric face-neighbour ring");
          assert.equal(corners.length, 0,
            "diagonal corners must not bypass demand-led frontier propagation");
        } else {
          assert.equal(corners.length, 8,
            "the complete horizontal corner orbit must publish by step 3");
          assert.ok(corners.every((brick) => brick.acceptedResolution === 8),
            "each corner must activate a complete B8 frontier tile");
        }
      }

      const [fields, finalActivity] = await Promise.all([
        solver.readDiagnosticFields(true), solver.readGPUActivityPolicy(),
      ]);
      const density = fields.density;
      const topology = new Uint8Array(horizontalCells * horizontalCells / 2
        * horizontalCells);
      for (const brick of finalActivity.bricks.filter((candidate) => candidate.active)) {
        for (let z = 0; z < brickSize; z += 1)
          for (let y = 0; y < brickSize; y += 1)
            for (let x = 0; x < brickSize; x += 1) {
              const qx = brickSize * brick.coordinate[0] + x;
              const qy = brickSize * brick.coordinate[1] + y;
              const qz = brickSize * brick.coordinate[2] + z;
              topology[qx + horizontalCells * (qy + horizontalCells / 2 * qz)] =
                brick.acceptedResolution;
            }
      }
      let cornerMass = 0;
      let totalMass = 0;
      let maximumDensity = 0;
      for (let z = 0; z < horizontalCells; z += 1)
        for (let y = 0; y < horizontalCells / 2; y += 1)
          for (let x = 0; x < horizontalCells; x += 1) {
            const rho = density[x + horizontalCells * (y + horizontalCells / 2 * z)]!;
            totalMass += rho;
            maximumDensity = Math.max(maximumDensity, rho);
            if (!horizontalCorner([Math.floor(x / brickSize), Math.floor(y / brickSize),
              Math.floor(z / brickSize)])) continue;
            cornerMass += rho;
          }
      const initialMass = (scene.fluid.initialBrickSeeds_m?.length ?? 0) * brickSize ** 3;
      assert.ok(Math.abs(totalMass - initialMass) / initialMass <= 3e-3,
        `symmetric expansion lost fluid mass: ${totalMass}/${initialMass}`);
      assert.ok(maximumDensity <= 2.5,
        `conserved mass collapsed into rho=${maximumDensity}, shrinking visible volume`);
      assert.ok(cornerMass > 1e-3,
        `allocated corner tiles must accept transported liquid; measured ${cornerMass}`);
      assert.ok(scalarD4Error(fields.density, 32, 16, 32) <= DENSITY_D4_LIMIT,
        "expanded density must retain horizontal D4 symmetry");
      assert.equal(scalarD4Error(topology, 32, 16, 32), 0,
        "expanded accepted topology must retain exact horizontal D4 symmetry");
      assert.ok(velocityD4Error(fields.velocity, 32, 16, 32)
        <= VELOCITY_D4_LIMIT_M_S,
        "expanded velocity must retain horizontal D4 symmetry");
      assert.ok(scalarD4Error(fields.pressure, 32, 16, 32)
        <= PRESSURE_D4_LIMIT_PA,
        "expanded pressure must retain horizontal D4 symmetry");
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
