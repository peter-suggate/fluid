/**
 * Operator-parity probe for CPU-built adaptive-mass conservative transport.
 *
 * This does not construct characteristics or CM12 rows on GPU. It verifies
 * that WebGPU executes the CPU authority's sparse receiver rows and paired
 * persistent nextGamma state correctly for every frozen M1 seam orientation.
 *
 * Run directly (the tool owns the repository-wide Dawn lock):
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js WEBGPU_BACKEND=metal \
 *     node --import tsx tools/probe-adaptive-mass-two-tile-transport-gpu.ts
 */
import { pathToFileURL } from "node:url";
import {
  applyTwoTileConservativeTransport,
  buildTwoTileConservativeTransportOperator,
  integratedScalar,
  type TwoTileConservativeOperator,
} from "../lib/methods/adaptive-mass/two-tile-conservative-transport";
import {
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileResolution,
} from "../lib/methods/adaptive-mass/two-tile-composite-grid";
import { WebGPUTwoTileConservativeTransport } from
  "../lib/methods/adaptive-mass/webgpu-two-tile-conservative-transport";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

interface FieldError {
  readonly maximumAbsoluteError: number;
  readonly maximumScaledError: number;
  readonly maximumExpectedMagnitude: number;
}

interface DensityProbeReceipt {
  readonly name: string;
  readonly density: FieldError;
  readonly gamma: FieldError;
  readonly gpuMassAbsoluteError: number;
}

interface VariantReceipt {
  readonly axis: CompositeAxis;
  readonly negativeResolution: TwoTileResolution;
  readonly positiveResolution: TwoTileResolution;
  readonly direction: -1 | 1;
  readonly gammaScenario: "persistent-nonuniform";
  readonly receiverCount: number;
  readonly coefficientCount: number;
  readonly maximumRowLength: number;
  readonly probes: readonly DensityProbeReceipt[];
  readonly maximumDensityAbsoluteError: number;
  readonly maximumGammaAbsoluteError: number;
  readonly maximumGpuMassAbsoluteError: number;
  readonly tolerance: number;
  readonly passed: boolean;
}

const fieldError = (expected: ArrayLike<number>, actual: ArrayLike<number>): FieldError => {
  let maximumAbsoluteError = 0;
  let maximumExpectedMagnitude = 0;
  for (let index = 0; index < expected.length; index += 1) {
    maximumAbsoluteError = Math.max(maximumAbsoluteError,
      Math.abs(expected[index] - actual[index]));
    maximumExpectedMagnitude = Math.max(maximumExpectedMagnitude, Math.abs(expected[index]));
  }
  return {
    maximumAbsoluteError,
    maximumScaledError: maximumAbsoluteError / Math.max(1, maximumExpectedMagnitude),
    maximumExpectedMagnitude,
  };
};

const makeOperator = (
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  direction: -1 | 1,
): TwoTileConservativeOperator => {
  const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
  const sourceGamma = Float64Array.from(grid.cells, (cell) => {
    const [x, y, z] = cell.center;
    return 0.72 + 0.16 * Math.sin(0.31 + 1.13 * x - 0.77 * y + 0.49 * z)
      + 0.035 * cell.tile;
  });
  return buildTwoTileConservativeTransportOperator({
    axis,
    negativeResolution,
    positiveResolution,
    displacement: direction * 0.75 / Math.max(negativeResolution, positiveResolution),
    sourceGamma,
  });
};

const densityProbes = (
  operator: TwoTileConservativeOperator,
): readonly (readonly [string, Float32Array])[] => [
  // A normalized constant is density == persistent gamma, not density == one.
  ["persistent-gamma-normalized-constant", Float32Array.from(operator.sourceGamma)],
  ["smooth-positive-density", Float32Array.from(operator.grid.cells, (cell) => {
    const [x, y, z] = cell.center;
    const normalized = 0.48 + 0.21 * Math.sin(0.43 + 2.1 * x - 1.4 * y + 0.8 * z);
    return Math.fround(operator.sourceGamma[cell.id] * normalized);
  })],
  ["deterministic-cell-density", Float32Array.from(operator.grid.cells, (cell) => {
    const hash = ((cell.id * 747_796_405 + 2_891_336_453) >>> 0) / 0xffff_ffff;
    return Math.fround(0.08 + 0.84 * hash);
  })],
];

const executeVariant = async (
  device: GPUDevice,
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  direction: -1 | 1,
): Promise<VariantReceipt> => {
  const cpuOperator = makeOperator(axis, negativeResolution, positiveResolution, direction);
  const gpuOperator = await WebGPUTwoTileConservativeTransport.create(device, cpuOperator);
  const valueBytes = gpuOperator.receiverCount * Float32Array.BYTES_PER_ELEMENT;
  const sourceDensity = device.createBuffer({
    label: "Adaptive mass transport probe source density",
    size: valueBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const destinationDensity = device.createBuffer({
    label: "Adaptive mass transport probe destination density",
    size: valueBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const destinationGamma = device.createBuffer({
    label: "Adaptive mass transport probe destination gamma",
    size: valueBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "Adaptive mass transport probe density and gamma readback",
    size: 2 * valueBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const probes: DensityProbeReceipt[] = [];
  try {
    for (const [name, density] of densityProbes(cpuOperator)) {
      const cpuAfter = applyTwoTileConservativeTransport(cpuOperator, {
        density: Float64Array.from(density),
        gamma: cpuOperator.sourceGamma,
      });
      device.queue.writeBuffer(sourceDensity, 0, density.buffer as ArrayBuffer,
        density.byteOffset, density.byteLength);
      const encoder = device.createCommandEncoder({
        label: `Adaptive mass transport parity ${name}`,
      });
      gpuOperator.encode(encoder, sourceDensity, destinationDensity, destinationGamma);
      encoder.copyBufferToBuffer(destinationDensity, 0, readback, 0, valueBytes);
      encoder.copyBufferToBuffer(destinationGamma, 0, readback, valueBytes, valueBytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const gpuDensity = new Float32Array(mapped, 0, gpuOperator.receiverCount).slice();
      const gpuGamma = new Float32Array(mapped, valueBytes, gpuOperator.receiverCount).slice();
      readback.unmap();
      probes.push({
        name,
        density: fieldError(cpuAfter.density, gpuDensity),
        gamma: fieldError(cpuAfter.gamma, gpuGamma),
        gpuMassAbsoluteError: Math.abs(
          integratedScalar(cpuOperator.grid, gpuDensity)
            - integratedScalar(cpuOperator.grid, density),
        ),
      });
    }
  } finally {
    sourceDensity.destroy();
    destinationDensity.destroy();
    destinationGamma.destroy();
    readback.destroy();
    gpuOperator.destroy();
  }

  const maximumDensityAbsoluteError = Math.max(
    ...probes.map((probe) => probe.density.maximumAbsoluteError),
  );
  const maximumGammaAbsoluteError = Math.max(
    ...probes.map((probe) => probe.gamma.maximumAbsoluteError),
  );
  const maximumGpuMassAbsoluteError = Math.max(
    ...probes.map((probe) => probe.gpuMassAbsoluteError),
  );
  const maximumMagnitude = Math.max(1, ...probes.flatMap((probe) => [
    probe.density.maximumExpectedMagnitude,
    probe.gamma.maximumExpectedMagnitude,
  ]));
  const tolerance = 2e-5 * maximumMagnitude;
  return {
    axis,
    negativeResolution,
    positiveResolution,
    direction,
    gammaScenario: "persistent-nonuniform",
    receiverCount: gpuOperator.receiverCount,
    coefficientCount: gpuOperator.coefficientCount,
    maximumRowLength: Math.max(...cpuOperator.rows.map((row) => row.length)),
    probes,
    maximumDensityAbsoluteError,
    maximumGammaAbsoluteError,
    maximumGpuMassAbsoluteError,
    tolerance,
    passed: maximumDensityAbsoluteError <= tolerance
      && maximumGammaAbsoluteError <= tolerance
      && maximumGpuMassAbsoluteError <= tolerance,
  };
};

await acquireWebGPUExclusiveLock(
  "dawn-probe",
  "tools/probe-adaptive-mass-two-tile-transport-gpu.ts",
);
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const backend = process.env.WEBGPU_BACKEND ?? "metal";
  const gpu = dawn.create([`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error(`No Dawn WebGPU adapter is available for backend ${backend}`);
  const device = await adapter.requestDevice();
  try {
    const variants: VariantReceipt[] = [];
    for (const axis of [0, 1, 2] as const) {
      for (const [negativeResolution, positiveResolution] of
        [[8, 8], [4, 4], [8, 4], [4, 8]] as const) {
        for (const direction of [-1, 1] as const) {
          variants.push(await executeVariant(
            device,
            axis,
            negativeResolution,
            positiveResolution,
            direction,
          ));
        }
      }
    }
    const passed = variants.every((variant) => variant.passed);
    const adapterInfo = (adapter as GPUAdapter & { readonly info?: GPUAdapterInfo }).info;
    const receipt = {
      passed,
      scope: "operator parity from CPU-built rows; GPU trace construction is not exercised",
      backend,
      adapter: adapterInfo ? {
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        device: adapterInfo.device,
        description: adapterInfo.description,
      } : undefined,
      variantCount: variants.length,
      densityProbeCount: variants.reduce((sum, variant) => sum + variant.probes.length, 0),
      maximumDensityAbsoluteError: Math.max(
        ...variants.map((variant) => variant.maximumDensityAbsoluteError),
      ),
      maximumGammaAbsoluteError: Math.max(
        ...variants.map((variant) => variant.maximumGammaAbsoluteError),
      ),
      maximumGpuMassAbsoluteError: Math.max(
        ...variants.map((variant) => variant.maximumGpuMassAbsoluteError),
      ),
      variants,
    };
    console.log(JSON.stringify(receipt, null, 2));
    if (!passed) process.exitCode = 1;
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
