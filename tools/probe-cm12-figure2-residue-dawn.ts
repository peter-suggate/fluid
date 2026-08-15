/** Exact Dawn reproduction of the Sparse CM12 Figure 2 residue silhouette. */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createCm12Figure2 } from "../lib/core/cm12-paper-scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

type Dimensions = readonly [number, number, number];

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
};

const steps = Number(argument("steps") ?? 58);
if (!Number.isInteger(steps) || steps < 1) throw new RangeError("steps must be a positive integer");
const dt_s = 1 / 30;

async function readDensity(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
): Promise<Float32Array> {
  const [nx, ny, nz] = dimensions;
  const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const buffer = device.createBuffer({
    label: "CM12 Figure 2 residue density readback",
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture },
      { buffer, bytesPerRow, rowsPerImage: ny },
      dimensions,
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(buffer.getMappedRange());
    const output = new Float32Array(nx * ny * nz);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      output.set(
        source.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx),
        nx * (y + ny * z),
      );
    }
    return output;
  } finally {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
  }
}

function summarizePublishedDensity(density: Float32Array) {
  const thresholds = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.5];
  let mass = 0;
  for (const rho of density) mass += rho;
  return {
    mass_cells: mass,
    maximum: density.reduce((maximum, value) => Math.max(maximum, value), 0),
    bands: thresholds.map((threshold) => {
      let cells = 0, bandMass = 0;
      for (const rho of density) if (rho > threshold) {
        cells += 1;
        bandMass += rho;
      }
      return { threshold, cells, mass_cells: bandMass };
    }),
    hypotheticalExtinction: [1e-5, 1e-4, 1e-3, 1e-2].map((threshold) => {
      let removedCells = 0, removedMass = 0;
      for (const rho of density) if (rho > 0 && rho <= threshold) {
        removedCells += 1;
        removedMass += rho;
      }
      return { threshold, removedCells, removedMass_cells: removedMass,
        removedMassFraction: removedMass / Math.max(mass, Number.MIN_VALUE) };
    }),
  };
}

await acquireWebGPUExclusiveLock("dawn-acceptance", "tools/probe-cm12-figure2-residue-dawn.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  try {
    const scene = createCm12Figure2();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      timeStep: "paper",
      resolutionMode: "all-fine",
    });
    const solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    try {
      const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as Dimensions;
      assert.deepEqual(dimensions, [128, 128, 8]);
      await device.queue.onSubmittedWorkDone();
      const initialDensity = summarizePublishedDensity(
        await readDensity(device, solver.volumeTexture, dimensions),
      );
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * dt_s, []), true);
      }
      await device.queue.onSubmittedWorkDone();
      const density = await readDensity(device, solver.volumeTexture, dimensions);
      const publishedDensity = summarizePublishedDensity(density);
      const activity = await solver.readGPUActivityPolicy();
      const brickWidth = 8;
      const brickDimensions = dimensions.map((value) =>
        value / brickWidth) as unknown as Dimensions;
      const residueBricks: Array<Record<string, unknown>> = [];
      const rows: string[] = [];
      let rawMassFromBrickMeans = 0;
      let hiddenActiveBricks = 0;
      for (const brick of activity.bricks) rawMassFromBrickMeans += brick.meanDensity * brick.resolution ** 3;
      for (let by = brickDimensions[1] - 1; by >= 0; by -= 1) {
        let row = "";
        for (let bx = 0; bx < brickDimensions[0]; bx += 1) {
          const brick = activity.bricks.find((candidate) => candidate.coordinate[0] === bx
            && candidate.coordinate[1] === by && candidate.coordinate[2] === 0)!;
          let maximum = 0, mass = 0, cellsAbove1e5 = 0, cellsAbove1e4 = 0;
          for (let z = 0; z < brickWidth; z += 1) for (let y = 0; y < brickWidth; y += 1) {
            for (let x = 0; x < brickWidth; x += 1) {
              const gx = brickWidth * bx + x, gy = brickWidth * by + y;
              const rho = density[gx + dimensions[0] * (gy + dimensions[1] * z)]!;
              maximum = Math.max(maximum, rho); mass += rho;
              cellsAbove1e5 += rho > 1e-5 ? 1 : 0;
              cellsAbove1e4 += rho > 1e-4 ? 1 : 0;
            }
          }
          const published = maximum > 0;
          if (brick.active && !published) hiddenActiveBricks += 1;
          const character = !brick.active ? " " : maximum >= 0.5 ? "L"
            : maximum >= 0.05 ? "s" : maximum > 1e-4 ? "d"
              : maximum > 1e-5 ? "r" : ".";
          row += character;
          if (published && maximum < 0.05) residueBricks.push({
            coordinate: brick.coordinate,
            maximumDensity: maximum,
            publishedMass_cells: mass,
            rawMassFromMean_cells: brick.meanDensity * brick.resolution ** 3,
            cellsAbove1e5,
            cellsAbove1e4,
            scoreByte: brick.scoreByte,
            reasons: brick.reasons,
            supportMask: brick.supportMask,
          });
        }
        rows.push(row);
      }
      console.log(JSON.stringify({
        scene: "cm12-figure-2",
        method: "adaptive-mass",
        resolutionMode: "all-fine",
        timeStep: "paper",
        steps,
        time_s: steps * dt_s,
        dimensions,
        legend: "L max rho>=.5; s >=.05; d >1e-4; r >1e-5; . active but unpublished; space inactive",
        brickMapTopToBottom: rows,
        activity: {
          acceptedSteps: activity.acceptedSteps,
          activeBricks: activity.bricks.filter((brick) => brick.active).length,
          occupiedReasonBricks: activity.bricks.filter((brick) => (brick.reasons & 64) !== 0).length,
          hiddenActiveBricks,
          rawMassFromBrickMeans_cells: rawMassFromBrickMeans,
          relativeRawMassDrift: (rawMassFromBrickMeans - initialDensity.mass_cells)
            / Math.max(initialDensity.mass_cells, Number.MIN_VALUE),
        },
        initialDensity,
        publishedDensity,
        residueBricks,
        validationErrors,
      }, null, 2));
    } finally {
      solver.destroy();
    }
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
