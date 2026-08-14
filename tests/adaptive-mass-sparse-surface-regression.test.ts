import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { cloneScene, defaultScene } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from
  "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { summarizeScalarField } from "../lib/harness/webgpu-smoke-scenarios";
import {
  createSparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { conditionSparseAtlasSurface } from
  "../lib/methods/adaptive-mass/sparse-atlas-surface-conditioning";
import { adaptiveMassPresentationDimensionsForScene } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";

function brick(
  key: number,
  coordinate: readonly [number, number, number],
  resolution: 4 | 8,
  density: (x: number, y: number, z: number) => number,
): SparseAdaptiveMassBrick {
  const values = Float64Array.from({ length: resolution ** 3 }, (_, index) => {
    const x = index % resolution;
    const yz = Math.floor(index / resolution);
    const y = yz % resolution, z = Math.floor(yz / resolution);
    return density(x, y, z);
  });
  return {
    key,
    coordinate,
    resolution,
    density: values,
    gamma: Float64Array.from(values, (_value, index) => 0.9 + 0.2 * (index % 7) / 6),
  };
}

test("Sparse CM12 preserves the requested 96x64x64 x4 lattice", () => {
  const scene = cloneScene(defaultScene);
  scene.voxelDomain.finestCellSize_m = 0.0125;
  assert.deepEqual(adaptiveMassPresentationDimensionsForScene(scene), [96, 64, 64]);
});

test("resident-row surface conditioning is conservative", () => {
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick(0, [0, 0, 0], 8, (x, y, z) =>
      Math.max(0, Math.min(1, 1.1 - 0.08 * x - 0.1 * y - 0.06 * z))),
    brick(1, [1, 0, 0], 4, (x, y, z) =>
      Math.max(0, Math.min(1, 0.62 - 0.12 * x - 0.08 * y - 0.05 * z))),
  ]);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const density = Float64Array.from(grid.cells, (cell) => cell.density);
  const gamma = Float64Array.from(grid.cells, (cell) => cell.gamma);
  const result = conditionSparseAtlasSurface(grid, { density, gamma });
  assert.ok(result.edgeCount > 0);
  assert.ok(result.gammaPairUpdates > 0);
  assert.ok(result.removedIntegratedMass > 0);
  assert.ok(result.massAbsoluteError < 1e-10, `${result.massAbsoluteError}`);
  assert.ok(result.gammaIntegralAbsoluteError < 1e-10,
    `${result.gammaIntegralAbsoluteError}`);
  assert.ok(Array.from(result.fields.density).every((value) =>
    Number.isFinite(value) && value >= -1e-12));
});

test("Dawn water-box has no frozen presentation shards after five seconds", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  usePerformanceInstrumentationStore.getState().setEnabled(false);
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu },
  });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) =>
    validationErrors.push(event.error.message));
  const scene = getScenePreset("water-box-dam-break").create();
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device,
    scene,
    "balanced",
    resolveMethodValues(adaptiveMassMethod, "balanced", {}),
    undefined,
    () => {},
  );
  const dimensions: [number, number, number] = [
    solver.info.nx, solver.info.ny, solver.info.nz,
  ];
  const readScalarTexture = async (texture: GPUTexture): Promise<Float32Array> => {
    const [nx, ny, nz] = dimensions;
    const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
    const readback = device.createBuffer({
      size: bytesPerRow * ny * nz,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: readback, bytesPerRow, rowsPerImage: ny },
        dimensions,
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Float32Array(readback.getMappedRange());
      const output = new Float32Array(nx * ny * nz);
      const stride = bytesPerRow / 4;
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        output.set(
          mapped.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx),
          nx * (y + ny * z),
        );
      }
      return output;
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  };
  try {
    assert.deepEqual(dimensions, [24, 16, 16]);
    const dt_s = 1 / 30;
    for (let step = 1; step <= 160; step += 1) {
      let retries = 0;
      while (!solver.advanceTo(step * dt_s, [])) {
        assert.ok(++retries <= 10, `advance ${step} remained rejected`);
        await new Promise(setImmediate);
      }
    }
    await device.queue.onSubmittedWorkDone();
    const density = await readScalarTexture(solver.volumeTexture);
    const phi = await readScalarTexture(solver.surfaceFieldTexture!);
    let signMismatchCount = 0;
    for (let index = 0; index < density.length; index += 1) {
      if ((density[index] >= 0.5) !== (phi[index] < 0)) signMismatchCount += 1;
    }
    const phiSummary = summarizeScalarField(
      Float32Array.from(phi, (value) => value < 0 ? 1 : 0),
      ...dimensions,
    );
    assert.equal(signMismatchCount, 0);
    assert.equal(phiSummary.componentCount, 1);
    assert.ok(Math.abs(solver.info.representedVolumeDrift ?? Infinity) < 1e-8);
    assert.deepEqual(validationErrors, []);
  } finally {
    solver.destroy();
    await device.queue.onSubmittedWorkDone();
    device.destroy();
  }
});
