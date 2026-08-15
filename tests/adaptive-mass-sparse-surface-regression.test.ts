import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
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

test("Dawn all-fine Figure 2 clears the tall wall residue conservatively", {
  skip: !process.env.WEBGPU_NODE_MODULE,
  timeout: 30_000,
}, () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath, [
    "--import", "tsx", "tools/probe-cm12-figure2-residue-dawn.ts", "--steps=98",
  ], {
    cwd: root,
    encoding: "utf8",
    timeout: 25_000,
    killSignal: "SIGKILL",
    env: {
      ...process.env,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    },
  });
  assert.equal(child.status, 0,
    `Figure 2 residue probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = JSON.parse(child.stdout) as {
    steps: number;
    time_s: number;
    activity: {
      activeBricks: number;
      relativeRawMassDrift: number;
    };
    residueBricks: Array<{
      coordinate: [number, number, number];
      maximumDensity: number;
    }>;
    validationErrors: string[];
  };
  const upperWallResidue = result.residueBricks.filter(({ coordinate: [x, y] }) =>
    (x <= 2 || x >= 13) && y >= 7);
  const centralTrail = result.residueBricks.filter(({ coordinate: [x, y] }) =>
    x >= 6 && x <= 9 && y >= 7);
  assert.equal(result.steps, 98);
  assert.equal(result.time_s, 98 / 30);
  assert.deepEqual(upperWallResidue, [],
    "the abandoned splash still publishes tall side-wall residue");
  assert.ok(centralTrail.every(({ maximumDensity }) => maximumDensity < 1e-4),
    `the central trail exceeds the microscopic residue band: ${JSON.stringify(centralTrail)}`);
  assert.ok(result.activity.activeBricks <= 115,
    `the cleaned splash retained ${result.activity.activeBricks} active bricks`);
  assert.ok(Math.abs(result.activity.relativeRawMassDrift) < 2e-4,
    `Algorithm 2 mass drift was ${result.activity.relativeRawMassDrift}`);
  assert.deepEqual(result.validationErrors, []);
});

test("Dawn all-coarse mini dam vacates the upper walls without corner amplification", {
  skip: !process.env.WEBGPU_NODE_MODULE,
  timeout: 30_000,
}, () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath, [
    "--import", "tsx", "tools/probe-cm12-mini-residual-dawn.ts",
    "--seconds=5.333333333333333", "--sparse-resolution=all-coarse",
    "--uniform-resolution=matched",
  ], {
    cwd: root,
    encoding: "utf8",
    timeout: 25_000,
    killSignal: "SIGKILL",
    env: {
      ...process.env,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    },
  });
  assert.equal(child.status, 0,
    `mini32 all-coarse probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = JSON.parse(child.stdout) as {
    grids: { sparse: number[]; uniform: number[] };
    sparse: {
      presentationInterpolation: {
        relativeL1: number;
        maximumAbsolute: number;
        coarseSamples: number;
      };
      upperWallFilm: {
        upperWallMaximum: number;
        upperWallLiquidCells: number;
        upperCornerLiquidCells: number;
      };
      maximumDensity: number;
      relativeMassDrift: number;
      fineBricks: number;
      coarseBricks: number;
    };
    validationErrors: string[];
  };
  assert.deepEqual(result.grids, { sparse: [32, 32, 32], uniform: [16, 16, 16] });
  assert.equal(result.sparse.fineBricks, 0);
  assert.equal(result.sparse.coarseBricks, 64);
  // Empty unsupported bricks may now retire after sharpening returns their
  // residue to the front. Every still-published coarse brick contributes one
  // complete 8^3 presentation block.
  assert.ok(result.sparse.presentationInterpolation.coarseSamples > 0);
  assert.ok(result.sparse.presentationInterpolation.coarseSamples <= 32 ** 3);
  assert.equal(result.sparse.presentationInterpolation.coarseSamples % (8 ** 3), 0);
  assert.ok(result.sparse.presentationInterpolation.relativeL1 < 1e-6,
    `all-coarse presentation used discontinuous leaf values: ${result.sparse.presentationInterpolation.relativeL1}`);
  assert.ok(result.sparse.presentationInterpolation.maximumAbsolute < 1e-6,
    `all-coarse presentation interpolation error was ${result.sparse.presentationInterpolation.maximumAbsolute}`);
  assert.ok(Math.abs(result.sparse.relativeMassDrift) < 2e-4,
    `mass drifted by ${result.sparse.relativeMassDrift}`);
  // CM12 Sec. 3.7 permits small rho>1 excursions and removes them gradually;
  // this ceiling catches an impact instability without forbidding that term.
  assert.ok(result.sparse.maximumDensity <= 1.2,
    `corner impact amplified density to ${result.sparse.maximumDensity}`);
  assert.ok(result.sparse.upperWallFilm.upperWallMaximum < 0.5,
    `upper-wall presentation remained liquid at rho=${result.sparse.upperWallFilm.upperWallMaximum}`);
  assert.equal(result.sparse.upperWallFilm.upperWallLiquidCells, 0);
  assert.equal(result.sparse.upperWallFilm.upperCornerLiquidCells, 0);
  assert.deepEqual(result.validationErrors, []);
});

test("Dawn adaptive presentation reconstructs every live coarse owner", {
  skip: !process.env.WEBGPU_NODE_MODULE,
  timeout: 30_000,
}, () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath, [
    "--import", "tsx", "tools/probe-cm12-mini-residual-dawn.ts",
    "--seconds=5.333333333333333", "--sparse-resolution=adaptive",
    "--uniform-resolution=fine",
  ], {
    cwd: root,
    encoding: "utf8",
    timeout: 25_000,
    killSignal: "SIGKILL",
    env: {
      ...process.env,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    },
  });
  assert.equal(child.status, 0,
    `mini32 adaptive probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = JSON.parse(child.stdout) as {
    sparse: {
      fineBricks: number;
      coarseBricks: number;
      presentationInterpolation: {
        relativeL1: number;
        maximumAbsolute: number;
        coarseSamples: number;
      };
      upperWallFilm: {
        upperWallLiquidCells: number;
        upperCornerLiquidCells: number;
      };
    };
    validationErrors: string[];
  };
  assert.ok(result.sparse.fineBricks > 0, "adaptive probe had no fine bricks");
  assert.ok(result.sparse.coarseBricks > 0, "adaptive probe had no coarse bricks");
  assert.ok(result.sparse.presentationInterpolation.coarseSamples > 0,
    "adaptive publication did not identify any live coarse presentation owners");
  assert.ok(result.sparse.presentationInterpolation.relativeL1 < 1e-6,
    `adaptive coarse presentation used discontinuous leaf values: ${result.sparse.presentationInterpolation.relativeL1}`);
  assert.ok(result.sparse.presentationInterpolation.maximumAbsolute < 1e-6,
    `adaptive coarse presentation interpolation error was ${result.sparse.presentationInterpolation.maximumAbsolute}`);
  assert.equal(result.sparse.upperWallFilm.upperWallLiquidCells, 0);
  assert.equal(result.sparse.upperWallFilm.upperCornerLiquidCells, 0);
  assert.deepEqual(result.validationErrors, []);
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
    const phi = await readScalarTexture(solver.surfaceFieldTexture!);
    const phiSummary = summarizeScalarField(
      Float32Array.from(phi, (value) => value < 0 ? 1 : 0),
      ...dimensions,
    );
    // Coarse rho is cell-centred authority. Its trilinear presentation samples
    // need not have the same sign as the repeated leaf value at every finest
    // cell; CM12 renders the interpolated rho=.5 contour, not that repetition.
    assert.equal(phiSummary.componentCount, 1, JSON.stringify(phiSummary));
    assert.ok(Math.abs(solver.info.representedVolumeDrift ?? Infinity) < 1e-8);
    assert.deepEqual(validationErrors, []);
  } finally {
    solver.destroy();
    await device.queue.onSubmittedWorkDone();
    device.destroy();
  }
});
