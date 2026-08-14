/**
 * Executable WebGPU parity probe for the first adaptive-mass pressure seam.
 *
 * Run directly (the tool owns the repository-wide Dawn lock):
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/probe-adaptive-mass-two-tile-pressure-gpu.ts
 */
import { pathToFileURL } from "node:url";
import {
  applyCompositePressureOperator,
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
} from "../lib/methods/adaptive-mass/two-tile-composite-grid";
import { WebGPUTwoTilePressureOperator } from
  "../lib/methods/adaptive-mass/webgpu-two-tile-pressure-operator";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

interface ProbeError {
  readonly name: string;
  readonly maximumAbsoluteError: number;
  readonly maximumRelativeError: number;
  readonly maximumExpectedMagnitude: number;
}

interface VariantResult {
  readonly axis: CompositeAxis;
  readonly negativeResolution: TwoTileResolution;
  readonly positiveResolution: TwoTileResolution;
  readonly cellCount: number;
  readonly gradientRowCount: number;
  readonly termCount: number;
  readonly incidenceCount: number;
  readonly probes: readonly ProbeError[];
  readonly maximumAbsoluteError: number;
  readonly maximumRelativeError: number;
  readonly tolerance: number;
  readonly passed: boolean;
}

const pressureProbes = (grid: TwoTileCompositeGrid): readonly (readonly [string, Float32Array])[] => [
  ["constant", new Float32Array(grid.cells.length).fill(Math.fround(2.75))],
  ["smooth-world-field", Float32Array.from(grid.cells, (cell) => {
    const [x, y, z] = cell.center;
    return Math.fround(Math.sin(0.37 + 0.91 * x) + 0.31 * y * y - 0.17 * z + 0.07 * x * z);
  })],
  ["deterministic-cell-field", Float32Array.from(grid.cells, (cell) => {
    const [x, y, z] = cell.center;
    const hash = ((cell.id * 747_796_405 + 2_891_336_453) >>> 0) / 0xffff_ffff;
    return Math.fround(0.6 * Math.sin(4.1 * x - 2.3 * y + 1.7 * z) + 0.4 * hash);
  })],
];

const compare = (
  name: string,
  expected: ArrayLike<number>,
  actual: ArrayLike<number>,
): ProbeError => {
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  let maximumExpectedMagnitude = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const magnitude = Math.abs(expected[index]);
    const error = Math.abs(actual[index] - expected[index]);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    maximumExpectedMagnitude = Math.max(maximumExpectedMagnitude, magnitude);
    maximumRelativeError = Math.max(maximumRelativeError, error / Math.max(1e-7, magnitude));
  }
  return { name, maximumAbsoluteError, maximumRelativeError, maximumExpectedMagnitude };
};

const executeVariant = async (
  device: GPUDevice,
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
): Promise<VariantResult> => {
  const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
  const operator = await WebGPUTwoTilePressureOperator.create(device, grid);
  const valueBytes = grid.cells.length * Float32Array.BYTES_PER_ELEMENT;
  const input = device.createBuffer({
    label: "Adaptive mass two-tile probe pressure",
    size: valueBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const output = device.createBuffer({
    label: "Adaptive mass two-tile probe operator output",
    size: valueBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "Adaptive mass two-tile probe readback",
    size: valueBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const errors: ProbeError[] = [];
  try {
    for (const [name, pressure] of pressureProbes(grid)) {
      const expected = applyCompositePressureOperator(grid, pressure);
      device.queue.writeBuffer(input, 0, pressure.buffer as ArrayBuffer,
        pressure.byteOffset, pressure.byteLength);
      const encoder = device.createCommandEncoder({ label: `Adaptive mass ${name} parity apply` });
      operator.encode(encoder, input, output);
      encoder.copyBufferToBuffer(output, 0, readback, 0, valueBytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange()).slice();
      readback.unmap();
      errors.push(compare(name, expected, actual));
    }
  } finally {
    input.destroy();
    output.destroy();
    readback.destroy();
    operator.destroy();
  }

  const maximumAbsoluteError = Math.max(...errors.map((probe) => probe.maximumAbsoluteError));
  const maximumRelativeError = Math.max(...errors.map((probe) => probe.maximumRelativeError));
  const maximumExpectedMagnitude = Math.max(...errors.map((probe) => probe.maximumExpectedMagnitude));
  // CPU evaluation is f64 while WebGPU stores the topology, input, and sums as
  // f32. This is an accuracy threshold, not a bitwise identity requirement.
  const tolerance = 5e-5 * Math.max(1, maximumExpectedMagnitude);
  return {
    axis,
    negativeResolution,
    positiveResolution,
    cellCount: operator.cellCount,
    gradientRowCount: operator.gradientRowCount,
    termCount: operator.termCount,
    incidenceCount: operator.incidenceCount,
    probes: errors,
    maximumAbsoluteError,
    maximumRelativeError,
    tolerance,
    passed: maximumAbsoluteError <= tolerance,
  };
};

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-adaptive-mass-two-tile-pressure-gpu.ts");
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
    const results: VariantResult[] = [];
    for (const axis of [0, 1, 2] as const) {
      for (const [negativeResolution, positiveResolution] of
        [[8, 8], [4, 4], [8, 4], [4, 8]] as const) {
        results.push(await executeVariant(device, axis, negativeResolution, positiveResolution));
      }
    }
    const passed = results.every((result) => result.passed);
    const adapterInfo = (adapter as GPUAdapter & { readonly info?: GPUAdapterInfo }).info;
    console.log(JSON.stringify({
      passed,
      backend,
      adapter: adapterInfo ? {
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        device: adapterInfo.device,
        description: adapterInfo.description,
      } : undefined,
      variantCount: results.length,
      maximumAbsoluteError: Math.max(...results.map((result) => result.maximumAbsoluteError)),
      results,
    }, null, 2));
    if (!passed) process.exitCode = 1;
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
