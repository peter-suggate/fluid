/**
 * Standalone native-Dawn regression lanes for aggressive Sparse CM12 work.
 *
 * These are intentionally not unit tests. Each lane owns an isolated process
 * and the repository-wide WebGPU lease. Missing publications, malformed JSON,
 * skipped work, and non-finite measurements all fail closed.
 *
 * Development gate (normally 2-5 minutes on Apple M1 Max):
 *   node --import tsx tools/run-sparse-cm12-temporal-regressions.ts
 *
 * Print the exact isolated commands without running Dawn:
 *   node --import tsx tools/run-sparse-cm12-temporal-regressions.ts --emit-commands
 */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineStage,
} from "../lib/core/fluid-pipeline";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import {
  createMinimalPowerDamBreak64Scene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from
  "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  gpuCompilationManagerFor,
  invalidateGPUCompilationManager,
} from "../lib/core/gpu-compilation-manager";
import {
  acquireWebGPUExclusiveLock,
  readWebGPUExclusiveLockHolder,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  createProcessRetainedDawnGPU,
  type NodeDawnProvider,
} from "../lib/harness/node-dawn-provider";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import {
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG as FPP_FLAG,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER as FPP_HEADER,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  SPARSE_CM12_FRAME_CONTROL_INVALID,
  SPARSE_CM12_FRAME_CONTROL_PHASE,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { SPARSE_CM12_FINAL_SCALAR_MASK_PHASE } from
  "../lib/methods/adaptive-mass/sparse-cm12-final-scalar-packet-masks";
import { SPARSE_CM12_VELOCITY_EXTENSION_HEADER as VEX_HEADER } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SELF = fileURLToPath(import.meta.url);
const NODE = process.execPath;
const DEFAULT_DAWN_MODULE = `${ROOT}/node_modules/webgpu/index.js`;
const DEFAULT_DAM_PERFORMANCE_BASELINE = fileURLToPath(new URL(
  "../artifacts/sparse-cm12-dam-front64-performance-baseline.json", import.meta.url,
));
const TARGET_BRICK_FINE_RESOLUTION = 16;
const TARGET_PRESENTATION_PAGE_RESOLUTION = 16;
const CANONICAL_PHYSICS_STEPS = 60;
const SHORT_SMOKE_STEPS = 5;
const MINIMUM_CANONICAL_DURATION_S = 2;
const DAM_BRICKS_PER_AXIS = 64 / TARGET_BRICK_FINE_RESOLUTION;
// Freeze the lane's authored controls here. Importing the implementation's
// current defaults would let a solver edit silently change the regression
// setup at the same time as the code under test.
const DAM_ACTIVITY_POLICY = Object.freeze({
  activitySignals: true,
  finestTravelCells: 1,
  fourTravelCells: 0.5,
  twoTravelCells: 0.25,
  thinFeatureCells: 2,
  thinFeatureDensity: 0,
  residencyDensity: 0.005,
  residencyMassFineCells: 1,
  surfaceDensityMinimum: 0.05,
  surfaceDensityMaximum: 0.95,
  detailTolerance: 0.08,
  frontLookaheadSteps: 4,
  topologyCadenceSteps: 1,
  prepareBricksPerFrame: 64,
  promoteEpochs: 2,
  demoteEpochs: 1,
  promoteScore: 160 / 255,
  demoteScore: 96 / 255,
  emergencyScore: 224 / 255,
});

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new RangeError(`--${name} requires a value`);
  }
  return value;
};

const hasFlag = (name: string): boolean => process.argv.includes(`--${name}`);

async function readBufferBinding(device: GPUDevice, binding: GPUBufferBinding,
  byteLength: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0,
      readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Uint32Array(readback.getMappedRange()).slice();
    return copy;
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

const positiveInteger = (name: string, fallback: number): number => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer; received ${value}`);
  }
  return value;
};

const physicsLaneSteps = (name: "dam-steps" | "symmetry-steps"): number => {
  const shortSmoke = hasFlag("short-smoke");
  const minimum = shortSmoke ? SHORT_SMOKE_STEPS : CANONICAL_PHYSICS_STEPS;
  const steps = positiveInteger(name, minimum);
  if (steps < minimum) {
    throw new RangeError(`${name} must be at least ${minimum}${shortSmoke
      ? " in explicit short-smoke mode" : " for the canonical >=2.0 s physics gate; use --short-smoke for an explicitly non-canonical smoke"}`);
  }
  return steps;
};

const checkpointSteps = (steps: number): readonly number[] => {
  const stride = Math.round(0.5 / CM12_PAPER_DT_S);
  return Array.from({ length: steps }, (_, index) => index + 1).filter((step) =>
    step === 1 || step === steps || step % stride === 0);
};

type ActivityBrick = Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]
>>["bricks"][number];

const brickKey = (coordinate: readonly number[]) => coordinate.join(",");

function resolutionHistogram(
  bricks: readonly ActivityBrick[],
  field: "acceptedResolution" | "candidateResolution",
) {
  return Object.fromEntries([1, 2, 4, 8, 16].map((resolution) => [resolution,
    bricks.filter((brick) => brick[field] === resolution).length]));
}

function deeplySubmergedBricks(bricks: readonly ActivityBrick[]): readonly ActivityBrick[] {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  return bricks.filter((brick) => brick.active && (brick.reasons & 64) !== 0
    && directions.every((direction) => {
      const coordinate = brick.coordinate.map((value, axis) =>
        value + direction[axis]!) as [number, number, number];
      if (coordinate.some((value) => value < 0 || value >= DAM_BRICKS_PER_AXIS)) return true;
      const neighbor = byCoordinate.get(brickKey(coordinate));
      return neighbor?.active === true && neighbor.meanDensity >= 0.5;
    }));
}

function collectTwoToOneFailures(
  failures: string[],
  bricks: readonly ActivityBrick[],
  field: "acceptedResolution" | "candidateResolution",
): void {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  for (const brick of bricks) for (let axis = 0; axis < 3; axis += 1) {
    const coordinate = [...brick.coordinate] as [number, number, number];
    coordinate[axis] += 1;
    const neighbor = byCoordinate.get(brickKey(coordinate));
    if (!neighbor) continue;
    const low = Math.min(brick[field], neighbor[field]);
    const high = Math.max(brick[field], neighbor[field]);
    if (high > 2 * low) failures.push(
      `${field} violates 2:1 at ${brickKey(brick.coordinate)}/${brickKey(coordinate)}`,
    );
  }
}

function collectSubmergedCandidateFailures(
  failures: string[],
  bricks: readonly ActivityBrick[],
): void {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  for (const brick of bricks.filter((candidate) => candidate.planReasons === 2048)) {
    if (brick.candidateResolution === brick.acceptedResolution) continue;
    let minimum = 1;
    for (const direction of directions) {
      const coordinate = brick.coordinate.map((value, axis) =>
        value + direction[axis]!) as [number, number, number];
      const neighbor = byCoordinate.get(brickKey(coordinate));
      if (neighbor) minimum = Math.max(minimum,
        Math.max(neighbor.candidateResolution, neighbor.acceptedResolution) / 2);
    }
    if (brick.candidateResolution !== minimum) failures.push(
      `submerged brick ${brickKey(brick.coordinate)} retained resolution ${
        brick.candidateResolution}; expected ${minimum}`,
    );
  }
}

function summarize(values: Float32Array) {
  let nonFinite = 0, minimum = Infinity, maximum = -Infinity, maximumAbsolute = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) { nonFinite += 1; continue; }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
  }
  return { count: values.length, nonFinite, minimum, maximum, maximumAbsolute };
}

function sha256Bytes(values: ArrayBufferView): string {
  return createHash("sha256").update(new Uint8Array(
    values.buffer, values.byteOffset, values.byteLength,
  )).digest("hex");
}

function bitExactFieldReceipt(fields: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readDiagnosticFields"]
>>) {
  return {
    densitySha256: sha256Bytes(fields.density),
    velocitySha256: sha256Bytes(fields.velocity),
    pressureSha256: sha256Bytes(fields.pressure),
    divergenceSha256: sha256Bytes(fields.divergence),
  };
}

function fieldByteDifference(a: Float32Array, b: Float32Array) {
  const av = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bv = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let count = 0;const first: Array<{ index: number; actual: number; oracle: number }> = [];
  for (let index = 0; index < av.length; index += 1) if (av[index] !== bv[index]) {
    count += 1;
    if (first.length < 8) first.push({ index, actual: av[index]!, oracle: bv[index]! });
  }
  return { count, first };
}

type BitExactFieldReceipt = ReturnType<typeof bitExactFieldReceipt>;
const BIT_EXACT_PHYSICAL_FIELDS = ["densitySha256", "velocitySha256",
  "pressureSha256", "divergenceSha256"] as const satisfies readonly (
    keyof BitExactFieldReceipt)[];

async function readDamPhysicalReference(path: string, steps: number): Promise<{
  readonly initial: BitExactFieldReceipt;
  readonly trajectory: readonly BitExactFieldReceipt[];
}> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const lanes = parsed.lanes as Record<string, unknown> | undefined;
  const report = (lanes?.["dam-front"] ?? parsed) as Record<string, unknown>;
  const configuration = report.configuration as Record<string, unknown> | undefined;
  if (report.scene !== "minimal-power-dam-break-64"
    || report.steps !== steps
    || configuration?.brickFineResolution !== TARGET_BRICK_FINE_RESOLUTION
    || configuration?.presentationPageResolution !== TARGET_PRESENTATION_PAGE_RESOLUTION) {
    throw new Error("physical reference scene, step count, or B16/P16 configuration differs");
  }
  const initial = (report.initial as Record<string, unknown> | undefined)?.bitExact;
  const trajectory = report.trajectory;
  if (!initial || !Array.isArray(trajectory) || trajectory.length !== steps) {
    throw new Error("physical reference has no complete bit-exact trajectory");
  }
  const requireReceipt = (value: unknown, location: string): BitExactFieldReceipt => {
    const receipt = value as Record<string, unknown> | undefined;
    for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
      if (typeof receipt?.[field] !== "string" || receipt[field].length !== 64) {
        throw new Error(`physical reference ${location}.${field} is missing or malformed`);
      }
    }
    return receipt as unknown as BitExactFieldReceipt;
  };
  return {
    initial: requireReceipt(initial, "initial"),
    trajectory: trajectory.map((entry, index) => requireReceipt(
      ((entry as Record<string, unknown>).bitExact as Record<string, unknown> | undefined)
        ?.adaptive,
      `trajectory[${index}]`,
    )),
  };
}

function densityFrontX(density: Float32Array, threshold: number): number {
  let front = -1;
  for (let z = 0; z < 64; z += 1) for (let y = 0; y < 64; y += 1)
    for (let x = 0; x < 64; x += 1)
      if (density[x + 64 * (y + 64 * z)]! > threshold) front = Math.max(front, x);
  return front;
}

function densityReceipt(density: Float32Array) {
  let mass = 0, maximum = 0, momentX = 0;
  for (let z = 0; z < 64; z += 1) for (let y = 0; y < 64; y += 1)
    for (let x = 0; x < 64; x += 1) {
      const rho = Math.max(0, density[x + 64 * (y + 64 * z)]!);
      mass += rho;
      maximum = Math.max(maximum, rho);
      momentX += rho * (x + 0.5) / 64;
    }
  return {
    mass,
    maximum,
    centerOfMassX: momentX / Math.max(mass, Number.MIN_VALUE),
    front: {
      trace: densityFrontX(density, 1e-3),
      surface: densityFrontX(density, 0.05),
      liquid: densityFrontX(density, 0.5),
    },
  };
}

function relativeDensityL1(reference: Float32Array, candidate: Float32Array): number {
  let difference = 0, scale = 0;
  for (let index = 0; index < reference.length; index += 1) {
    difference += Math.abs(candidate[index]! - reference[index]!);
    scale += Math.abs(reference[index]!);
  }
  return difference / Math.max(scale, Number.MIN_VALUE);
}

function connectivityReceipt(density: Float32Array) {
  const count = 64 ** 3;
  const visited = new Uint8Array(count);
  const queue = new Uint32Array(count);
  let totalMass = 0, occupiedCells = 0, componentCount = 0;
  for (let index = 0; index < count; index += 1) {
    totalMass += Math.max(0, density[index]!);
    if (density[index]! > 1e-5) occupiedCells += 1;
  }
  let dominantMass = 0, dominantCells = 0;
  for (let seed = 0; seed < count; seed += 1) {
    if (visited[seed] !== 0 || density[seed]! <= 1e-5) continue;
    componentCount += 1;
    visited[seed] = 1;
    let head = 0, tail = 0, componentMass = 0, componentCells = 0;
    queue[tail++] = seed;
    while (head < tail) {
      const index = queue[head++]!;
      componentMass += Math.max(0, density[index]!);
      componentCells += 1;
      const x = index % 64;
      const y = Math.floor(index / 64) % 64;
      const z = Math.floor(index / (64 * 64));
      const visit = (next: number) => {
        if (visited[next] === 0 && density[next]! > 1e-5) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      };
      if (x > 0) visit(index - 1);
      if (x < 63) visit(index + 1);
      if (y > 0) visit(index - 64);
      if (y < 63) visit(index + 64);
      if (z > 0) visit(index - 64 * 64);
      if (z < 63) visit(index + 64 * 64);
    }
    if (componentMass > dominantMass) {
      dominantMass = componentMass;
      dominantCells = componentCells;
    }
  }
  return {
    occupiedCells,
    componentCount,
    dominantCells,
    dominantMassFraction: dominantMass / Math.max(totalMass, Number.MIN_VALUE),
  };
}

type Dimensions = readonly [number, number, number];

function scalarD4(field: Float32Array, dimensions: Dimensions) {
  const [nx, ny, nz] = dimensions;
  let nonFinite = 0, maximumAbsoluteError = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) for (const target of [
      [nx - 1 - x, y, z], [x, y, nz - 1 - z], [z, y, x],
    ] as const) {
      const source = field[x + nx * (y + ny * z)]!;
      const actual = field[target[0] + nx * (target[1] + ny * target[2])]!;
      if (!Number.isFinite(source) || !Number.isFinite(actual)) nonFinite += 1;
      else maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(actual - source));
    }
  return { nonFinite, maximumAbsoluteError };
}

function velocityD4(field: Float32Array, dimensions: Dimensions) {
  const [nx, ny, nz] = dimensions;
  let nonFinite = 0, maximumAbsoluteError = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) for (const transform of [0, 1, 2] as const) {
      const target: readonly [number, number, number] = transform === 0
        ? [nx - 1 - x, y, z] : transform === 1
          ? [x, y, nz - 1 - z] : [z, y, x];
      const sourceCell = x + nx * (y + ny * z);
      const targetCell = target[0] + nx * (target[1] + ny * target[2]);
      for (let component = 0; component < 3; component += 1) {
        const targetComponent = transform === 2
          ? component === 0 ? 2 : component === 2 ? 0 : 1 : component;
        const source = field[4 * sourceCell + component]!;
        const expected = transform === 0 && component === 0
          ? -source : transform === 1 && component === 2 ? -source : source;
        const actual = field[4 * targetCell + targetComponent]!;
        if (!Number.isFinite(expected) || !Number.isFinite(actual)) nonFinite += 1;
        else maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(actual - expected));
      }
    }
  return { nonFinite, maximumAbsoluteError };
}

function genericConnectivityReceipt(density: Float32Array, dimensions: Dimensions) {
  const [nx, ny, nz] = dimensions;
  const count = nx * ny * nz;
  const visited = new Uint8Array(count);
  const queue = new Uint32Array(count);
  let totalMass = 0, componentCount = 0, dominantMass = 0;
  for (let index = 0; index < count; index += 1)
    totalMass += Math.max(0, density[index]!);
  for (let seed = 0; seed < count; seed += 1) {
    if (visited[seed] !== 0 || density[seed]! <= 1e-5) continue;
    componentCount += 1;
    visited[seed] = 1;
    let head = 0, tail = 0, mass = 0;
    queue[tail++] = seed;
    while (head < tail) {
      const index = queue[head++]!;
      mass += Math.max(0, density[index]!);
      const x = index % nx, y = Math.floor(index / nx) % ny;
      const z = Math.floor(index / (nx * ny));
      const visit = (next: number) => {
        if (visited[next] === 0 && density[next]! > 1e-5) {
          visited[next] = 1; queue[tail++] = next;
        }
      };
      if (x > 0) visit(index - 1);
      if (x + 1 < nx) visit(index + 1);
      if (y > 0) visit(index - nx);
      if (y + 1 < ny) visit(index + nx);
      if (z > 0) visit(index - nx * ny);
      if (z + 1 < nz) visit(index + nx * ny);
    }
    dominantMass = Math.max(dominantMass, mass);
  }
  return { componentCount,
    dominantMassFraction: dominantMass / Math.max(totalMass, Number.MIN_VALUE) };
}

function expect(failures: string[], condition: boolean, message: string): void {
  if (!condition) failures.push(message);
}

const median = (values: readonly number[]): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor((ordered.length - 1) / 2)]!;
};

const percentile = (values: readonly number[], quantile: number): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(quantile * ordered.length) - 1]!;
};

function timingDistribution(values: readonly number[]) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("hardware timing distribution is empty, non-finite, or negative");
  }
  return {
    count: values.length,
    samples_ms: values.map((value) => Number(value.toFixed(4))),
    minimum_ms: Number(Math.min(...values).toFixed(4)),
    median_ms: Number(median(values).toFixed(4)),
    p95_ms: Number(percentile(values, 0.95).toFixed(4)),
    maximum_ms: Number(Math.max(...values).toFixed(4)),
  };
}

type DamPerformancePhysicsStep = Readonly<{
  step: number;
  density: ReturnType<typeof densityReceipt>;
  connectivity: ReturnType<typeof connectivityReceipt>;
  fields: Readonly<Record<"density" | "velocity" | "pressure" | "divergence",
    ReturnType<typeof summarize>>>;
  pressure: Readonly<Record<string, number | boolean | undefined>>;
  topology: Readonly<Record<string, number | undefined>>;
}>;

interface DamPerformancePhysicsReceipt {
  readonly initialDensity: ReturnType<typeof densityReceipt>;
  readonly steps: readonly DamPerformancePhysicsStep[];
}

type StagePathKind = "incremental" | "global" | "exclusive";

/**
 * Command-encoding observer used only by the standalone performance lane.
 * It wraps native objects without adding GPU commands and records which
 * indirect authority each resident stage dispatched. The accepted snapshot
 * is global work; explicitly temporal/dirty/frontier/local work is
 * incremental. Seeing both in one stage is the forbidden fallback pattern.
 */
class DamStagePathObserver {
  private readonly pipelineLabels = new WeakMap<object, string>();
  private readonly bufferLabels = new WeakMap<object, string>();
  private observations: Array<{
    stage: string;
    pipeline: string;
    dispatch: "direct" | "indirect";
    indirectBuffer?: string;
  }> = [];

  wrap(device: GPUDevice): GPUDevice {
    const pipelineLabels = this.pipelineLabels;
    const bufferLabels = this.bufferLabels;
    const wrapEncoder = this.wrapEncoder.bind(this);
    return new Proxy(device, {
      get(target, property) {
        if (property === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
          const buffer = target.createBuffer(descriptor);
          bufferLabels.set(buffer, descriptor.label ?? "unlabelled buffer");
          return buffer;
        };
        if (property === "createComputePipeline") {
          return (descriptor: GPUComputePipelineDescriptor) => {
            const pipeline = target.createComputePipeline(descriptor);
            pipelineLabels.set(pipeline, descriptor.label ?? "unlabelled pipeline");
            return pipeline;
          };
        }
        if (property === "createComputePipelineAsync") {
          return async (descriptor: GPUComputePipelineDescriptor) => {
            const pipeline = await target.createComputePipelineAsync(descriptor);
            pipelineLabels.set(pipeline, descriptor.label ?? "unlabelled pipeline");
            return pipeline;
          };
        }
        if (property === "createCommandEncoder") {
          return (descriptor?: GPUCommandEncoderDescriptor) =>
            wrapEncoder(target.createCommandEncoder(descriptor));
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUDevice;
  }

  beginFrame(): void { this.observations = []; }

  finishFrame(): Readonly<Record<string, {
    readonly modes: readonly StagePathKind[];
    readonly dispatchCount: number;
    readonly pipelines: readonly string[];
  }>> {
    const stages = new Map<string, typeof this.observations>();
    for (const observation of this.observations) {
      const values = stages.get(observation.stage) ?? [];
      values.push(observation);
      stages.set(observation.stage, values);
    }
    return Object.fromEntries([...stages].map(([stage, observations]) => {
      const modes = new Set<StagePathKind>();
      for (const observation of observations) {
        const authority = `${observation.pipeline} ${observation.indirectBuffer ?? ""}`;
        if (/(incremental|temporal|dirty|frontier|local)/i.test(authority)) {
          modes.add("incremental");
        }
        if (/accepted indirect dispatch snapshot|\b(global|full|dense)\b/i.test(authority)) {
          modes.add("global");
        }
      }
      if (modes.size === 0) modes.add("exclusive");
      return [stage, {
        modes: [...modes],
        dispatchCount: observations.length,
        pipelines: [...new Set(observations.map((observation) => observation.pipeline))],
      }];
    }));
  }

  private wrapEncoder(encoder: GPUCommandEncoder): GPUCommandEncoder {
    const wrapPass = this.wrapPass.bind(this);
    return new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") {
          return (descriptor?: GPUComputePassDescriptor) => {
            const pass = target.beginComputePass(descriptor);
            const prefix = "Sparse CM12 resident ";
            const stage = descriptor?.label?.startsWith(prefix)
              ? descriptor.label.slice(prefix.length) : undefined;
            return stage ? wrapPass(pass, stage) : pass;
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  private wrapPass(pass: GPUComputePassEncoder, stage: string): GPUComputePassEncoder {
    const pipelineLabels = this.pipelineLabels;
    const bufferLabels = this.bufferLabels;
    const observations = this.observations;
    let pipeline = "unlabelled pipeline";
    return new Proxy(pass, {
      get(target, property) {
        if (property === "setPipeline") return (value: GPUComputePipeline) => {
          pipeline = pipelineLabels.get(value) ?? value.label ?? "unlabelled pipeline";
          target.setPipeline(value);
        };
        if (property === "dispatchWorkgroups") return (x: number, y?: number, z?: number) => {
          observations.push({ stage, pipeline, dispatch: "direct" });
          target.dispatchWorkgroups(x, y, z);
        };
        if (property === "dispatchWorkgroupsIndirect") {
          return (buffer: GPUBuffer, offset: number) => {
            observations.push({
              stage, pipeline, dispatch: "indirect",
              indirectBuffer: bufferLabels.get(buffer) ?? buffer.label ?? "unlabelled buffer",
            });
            target.dispatchWorkgroupsIndirect(buffer, offset);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUComputePassEncoder;
  }
}

const DAM_PERFORMANCE_PHYSICS_EQUIVALENCE = Object.freeze({
  maximumRelativeMassDelta: 1e-6,
  maximumCenterOfMassXDelta: 1e-6,
  // Five post-warmup Metal replays varied by one threshold-straddling liquid
  // cell and its two/three incident active rows. These absolute count bounds
  // are 0.002% of the 50k/150k worksets; larger topology changes still fail.
  maximumPressureCellCountDelta: 1,
  maximumPressureActiveRowCountDelta: 3,
  // Observed maxima were 4.40e-5 density and 3.15e-4 m/s velocity. Roughly
  // twofold envelopes remain orders below the independent 6% density-shape,
  // 0.2% mass, 0.75/s divergence, and exact-front gates.
  fieldMaximum: Object.freeze({
    density: Object.freeze({ absolute: 1e-4, relative: 1e-5 }),
    velocity: Object.freeze({ absolute: 5e-4, relative: 5e-5 }),
    pressure: Object.freeze({ absolute: 1e-1, relative: 1e-5 }),
  }),
});

interface DamPhysicsEquivalenceCheck {
  readonly path: string;
  readonly comparison: "exact" | "absolute" | "relative" | "absolute+relative";
  readonly reference: number;
  readonly candidate: number;
  readonly delta: number;
  readonly tolerance: number;
  readonly passed: boolean;
}

interface DamPhysicsEquivalenceResult {
  readonly equivalent: boolean;
  readonly checks: readonly DamPhysicsEquivalenceCheck[];
  readonly differences: readonly DamPhysicsEquivalenceCheck[];
}

function comparePhysicsReceipts(
  reference: DamPerformancePhysicsReceipt,
  candidate: DamPerformancePhysicsReceipt,
): DamPhysicsEquivalenceResult {
  const checks: DamPhysicsEquivalenceCheck[] = [];
  const record = (
    path: string,
    comparison: DamPhysicsEquivalenceCheck["comparison"],
    left: number,
    right: number,
    delta: number,
    tolerance: number,
  ) => checks.push({ path, comparison, reference: left, candidate: right,
    delta, tolerance, passed: Number.isFinite(delta) && delta <= tolerance });
  const relativeDelta = (left: number, right: number) =>
    Math.abs(left - right) / Math.max(Math.abs(left), Math.abs(right), Number.MIN_VALUE);
  const exact = (path: string, left: number, right: number) =>
    record(path, "exact", left, right, Math.abs(left - right), 0);
  const relative = (path: string, left: number, right: number, tolerance: number) =>
    record(path, "relative", left, right, relativeDelta(left, right), tolerance);
  const absolute = (path: string, left: number, right: number, tolerance: number) =>
    record(path, "absolute", left, right, Math.abs(left - right), tolerance);
  const maximum = (
    path: string,
    left: number,
    right: number,
    tolerance: { readonly absolute: number; readonly relative: number },
  ) => {
    const delta = Math.abs(left - right);
    const limit = tolerance.absolute
      + tolerance.relative * Math.max(Math.abs(left), Math.abs(right));
    record(path, "absolute+relative", left, right, delta, limit);
  };
  const density = (
    path: string,
    left: ReturnType<typeof densityReceipt>,
    right: ReturnType<typeof densityReceipt>,
  ) => {
    relative(`${path}.mass`, left.mass, right.mass,
      DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.maximumRelativeMassDelta);
    absolute(`${path}.centerOfMassX`, left.centerOfMassX, right.centerOfMassX,
      DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.maximumCenterOfMassXDelta);
    exact(`${path}.front.trace`, left.front.trace, right.front.trace);
    exact(`${path}.front.surface`, left.front.surface, right.front.surface);
    exact(`${path}.front.liquid`, left.front.liquid, right.front.liquid);
    maximum(`${path}.maximum`, left.maximum, right.maximum,
      DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.fieldMaximum.density);
  };
  density("initialDensity", reference.initialDensity, candidate.initialDensity);
  exact("steps.length", reference.steps.length, candidate.steps.length);
  const exactPressureKeys = ["iterations", "iterationsExecuted"] as const;
  const exactTopologyKeys = ["generation", "prepared", "committed", "deferred",
    "acceptedCells", "acceptedRows", "deeplySubmerged"] as const;
  for (let index = 0; index < Math.min(reference.steps.length, candidate.steps.length); index += 1) {
    const left = reference.steps[index]!;
    const right = candidate.steps[index]!;
    const path = `steps[${index}]`;
    exact(`${path}.step`, left.step, right.step);
    density(`${path}.density`, left.density, right.density);
    exact(`${path}.connectivity.componentCount`, left.connectivity.componentCount,
      right.connectivity.componentCount);
    for (const key of exactPressureKeys) {
      exact(`${path}.pressure.${key}`, Number(left.pressure[key]), Number(right.pressure[key]));
    }
    absolute(`${path}.pressure.cells`, Number(left.pressure.cells), Number(right.pressure.cells),
      DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.maximumPressureCellCountDelta);
    absolute(`${path}.pressure.activeRows`, Number(left.pressure.activeRows),
      Number(right.pressure.activeRows),
      DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.maximumPressureActiveRowCountDelta);
    for (const key of exactTopologyKeys) {
      exact(`${path}.topology.${key}`, Number(left.topology[key]), Number(right.topology[key]));
    }
    for (const field of ["density", "velocity", "pressure"] as const) {
      maximum(`${path}.fields.${field}.maximumAbsolute`,
        left.fields[field].maximumAbsolute, right.fields[field].maximumAbsolute,
        DAM_PERFORMANCE_PHYSICS_EQUIVALENCE.fieldMaximum[field]);
    }
  }
  const differences = checks.filter((check) => !check.passed);
  return { equivalent: differences.length === 0, checks, differences };
}

function parseDamPerformanceBaseline(value: unknown): {
  nonPressureP95_ms: number;
  physicsReference: DamPerformancePhysicsReceipt;
} {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("dam performance baseline must be an object");
  }
  const baseline = value as Record<string, unknown>;
  if (baseline.kind !== "sparse-cm12-dam-front64-performance" || baseline.version !== 1) {
    throw new Error("dam performance baseline has an unsupported kind or version");
  }
  const configuration = baseline.configuration as Record<string, unknown> | undefined;
  if (configuration?.brickFineResolution !== TARGET_BRICK_FINE_RESOLUTION
    || configuration.presentationPageResolution !== TARGET_PRESENTATION_PAGE_RESOLUTION
    || baseline.steps !== 5) {
    throw new Error("dam performance baseline is not the fixed dam64 B16/P16 five-step lane");
  }
  const nonPressure = baseline.nonPressure as Record<string, unknown> | undefined;
  const nonPressureP95_ms = nonPressure?.p95_ms;
  if (typeof nonPressureP95_ms !== "number" || !Number.isFinite(nonPressureP95_ms)
    || nonPressureP95_ms <= 0) {
    throw new Error("dam performance baseline has no positive finite non-pressure p95");
  }
  const physicsReference = baseline.physicsReference as DamPerformancePhysicsReceipt | undefined;
  if (!physicsReference || !Array.isArray(physicsReference.steps)
    || physicsReference.steps.length !== 5) {
    throw new Error("dam performance baseline has no complete five-step physics reference");
  }
  return { nonPressureP95_ms, physicsReference };
}

async function runDamFrontLane(): Promise<void> {
  const steps = physicsLaneSteps("dam-steps");
  const physicalReferencePath = argument("physical-reference");
  const physicalReference = physicalReferencePath
    ? await readDamPhysicalReference(physicalReferencePath, steps) : undefined;
  const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
  await acquireWebGPUExclusiveLock("dawn-acceptance",
    "tools/run-sparse-cm12-temporal-regressions.ts:dam-front");
  let gpu: GPU | undefined;
  let device: GPUDevice | undefined;
  let adaptive: WebGPUAdaptiveMassSolver | undefined;
  let allFine: WebGPUAdaptiveMassSolver | undefined;
  let pressureOracle: WebGPUAdaptiveMassSolver | undefined;
  let authorityOracle: WebGPUAdaptiveMassSolver | undefined;
  const pressureOraclePaired = hasFlag("pressure-oracle-paired");
  const authorityOraclePaired = hasFlag("fca-authority-oracle-paired");
  const failures: string[] = [];
  try {
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error(`No Dawn adapter is available for backend ${backend}`);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");
    const scene = createMinimalPowerDamBreak64Scene();
    const solverOptions = (resolutionMode: "adaptive" | "all-fine") => ({
        resolutionMode,
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
        surfaceFineRings: resolutionMode === "adaptive" ? 8 : 1,
        activityPolicy: {
          ...DAM_ACTIVITY_POLICY,
          topologyCadenceSteps: resolutionMode === "adaptive" ? 1 : 64,
        },
        timeStep: "paper",
      } as const);
    const createSolver = (resolutionMode: "adaptive" | "all-fine") =>
      WebGPUAdaptiveMassSolver.createAsync(device!, scene, "balanced", undefined,
        solverOptions(resolutionMode), () => {});
    // Dawn's native ProcessEvents runner is instance-scoped. Serial QA solver
    // construction prevents hundreds of manager-owned pipeline promises from
    // racing across multiple resident owners during peak allocation.
    adaptive = await createSolver("adaptive");
    allFine = await createSolver("all-fine");
    if (pressureOraclePaired) {
      pressureOracle = await WebGPUAdaptiveMassSolver.createPressureRefreshOracleForQA(
        device, scene, "balanced", undefined, solverOptions("adaptive"), () => {});
    }
    if (authorityOraclePaired) {
      authorityOracle = await WebGPUAdaptiveMassSolver.createLegacyHostAuthorityOracleForQA(
        device, scene, "balanced", undefined, solverOptions("adaptive"), () => {});
    }
    const [initialAdaptiveFields, initialAllFineFields] = await Promise.all([
      adaptive.readDiagnosticFields(), allFine.readDiagnosticFields(),
    ]);
    const initial = {
      adaptive: densityReceipt(initialAdaptiveFields.density),
      allFine: densityReceipt(initialAllFineFields.density),
      adaptiveBitExact: bitExactFieldReceipt(initialAdaptiveFields),
      allFineBitExact: bitExactFieldReceipt(initialAllFineFields),
    };
    if (pressureOracle) {
      const oracleInitial = bitExactFieldReceipt(await pressureOracle.readDiagnosticFields());
      for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
        expect(failures, oracleInitial[field] === initial.adaptiveBitExact[field],
          `pressure oracle initial physical mismatch in ${field.replace("Sha256", "")}`);
      }
    }
    if (authorityOracle) {
      const oracleInitial = bitExactFieldReceipt(await authorityOracle.readDiagnosticFields());
      for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
        expect(failures, oracleInitial[field] === initial.adaptiveBitExact[field],
          `FCA authority oracle initial physical mismatch in ${field.replace("Sha256", "")}`);
      }
    }
    const initialActivity = await adaptive.readGPUActivityPolicy();
    const initialStats = await adaptive.readStats();
    if (physicalReference) for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
      expect(failures, initial.adaptiveBitExact[field] === physicalReference.initial[field],
        `initial physical byte mismatch in ${field.replace("Sha256", "")}`);
    }
    expect(failures, initialAdaptiveFields.density.length === 64 ** 3,
      `initial density publication has ${initialAdaptiveFields.density.length} cells, expected ${64 ** 3}`);
    expect(failures, summarize(initialAdaptiveFields.density).nonFinite === 0,
      "initial adaptive density contains non-finite values");
    expect(failures, initial.adaptive.front.surface === 39,
      `dam front starts at x=${initial.adaptive.front.surface}, expected x=39`);
    expect(failures, relativeDensityL1(initialAllFineFields.density,
      initialAdaptiveFields.density) === 0,
    "adaptive and all-fine controls do not start from the same density field");

    const trajectory: Array<Record<string, unknown>> = [];
    const checkpointStepSet = new Set(checkpointSteps(steps));
    const presentationControl = adaptive.globalFineLevelSetSource?.presentationControl;
    if (!presentationControl) throw new Error(
      "production Sparse CM12 did not expose the FPP1 presentation control receipt",
    );
    for (let step = 1; step <= steps; step += 1) {
      const time_s = step * CM12_PAPER_DT_S;
      expect(failures, adaptive.advanceTo(time_s, []),
        `step ${step}: adaptive advance did not encode`);
      expect(failures, allFine.advanceTo(time_s, []),
        `step ${step}: all-fine advance did not encode`);
      if (pressureOracle) expect(failures, pressureOracle.advanceTo(time_s, []),
        `step ${step}: pressure oracle advance did not encode`);
      if (authorityOracle) expect(failures, authorityOracle.advanceTo(time_s, []),
        `step ${step}: FCA authority oracle advance did not encode`);
      await device.queue.onSubmittedWorkDone();
      const [adaptiveFields, allFineFields, activity, stats, allFineStats,
        adaptivePressureMembership, allFinePressureMembership,
        adaptiveFca, allFineFca,
        oracleFields, oracleStats, oracleMembership, authorityOracleFields,
        fppHeader, adaptiveSaw, authorityOracleSaw,
        authorityOracleFca,
        authorityOracleActivity, authorityOracleStats, adaptiveVex, allFineVex,
        authorityOracleVex]
        = await Promise.all([
        adaptive.readDiagnosticFields(), allFine.readDiagnosticFields(),
        adaptive.readGPUActivityPolicy(), adaptive.readStats(), allFine.readStats(),
        adaptive.readPressureCanonicalMembershipQA(),
        allFine.readPressureCanonicalMembershipQA(),
        adaptive.readFrameControlQA(), allFine.readFrameControlQA(),
        pressureOracle?.readDiagnosticFields(), pressureOracle?.readStats(),
        pressureOracle?.readPressureCanonicalMembershipQA(),
        authorityOracle?.readDiagnosticFields(),
        checkpointStepSet.has(step)
          ? readBufferBinding(device, presentationControl,
            4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS)
          : Promise.resolve(undefined),
        adaptive.readFinalScalarMaskHeaderQA(), authorityOracle?.readFinalScalarMaskHeaderQA(),
        authorityOracle?.readFrameControlQA(), authorityOracle?.readGPUActivityPolicy(),
        authorityOracle?.readStats(),
        adaptive.readVelocityExtensionQA(), allFine.readVelocityExtensionQA(),
        authorityOracle?.readVelocityExtensionQA(),
      ]);
      for (const [name, saw] of [
        ["adaptive", adaptiveSaw],
        ...(authorityOracleSaw
          ? [["immutable oracle", authorityOracleSaw] as const] : []),
      ] as const) {
        expect(failures, saw.phase === 2 && saw.fault === 0
          && saw.firstFaultPacket === 0xffff_ffff,
        `step ${step}: ${name} FSM1 phase/fault/packet ${saw.phase}/${saw.fault}/${
          saw.firstFaultPacket}`);
        expect(failures, saw.generation === step + 1,
        `step ${step}: ${name} FSM1 generation ${saw.generation}`);
      }
      for (const [name, fca] of [["adaptive", adaptiveFca], ["all-fine", allFineFca]] as const) {
        expect(failures, fca.phase === SPARSE_CM12_FRAME_CONTROL_PHASE.accepted
          && fca.fault === 0 && fca.firstFaultOwner === SPARSE_CM12_FRAME_CONTROL_INVALID,
        `step ${step}: ${name} FCA1 phase/fault/owner ${fca.phase}/${fca.fault}/${
          fca.firstFaultOwner}`);
        expect(failures, fca.acceptedGeneration === step + 1
          && fca.candidateGeneration === fca.acceptedGeneration
          && fca.sealedGeneration === fca.acceptedGeneration,
        `step ${step}: ${name} FCA1 generation accepted/candidate/sealed ${
          fca.acceptedGeneration}/${fca.candidateGeneration}/${fca.sealedGeneration}`);
        expect(failures, fca.scalarParity === (step & 1) && fca.faceParity === (step & 1),
          `step ${step}: ${name} FCA1 scalar/face parity ${fca.scalarParity}/${fca.faceParity}`);
        expect(failures, fca.coverage === (SPARSE_CM12_FRAME_CONTROL_COVERAGE.authority
          | SPARSE_CM12_FRAME_CONTROL_COVERAGE.output)
          && fca.committedFrames === step,
        `step ${step}: ${name} FCA1 coverage/commits ${fca.coverage}/${fca.committedFrames}`);
      }
      if (authorityOracleFca) {
        expect(failures, authorityOracleFca.phase === SPARSE_CM12_FRAME_CONTROL_PHASE.accepted
          && authorityOracleFca.fault === 0
          && authorityOracleFca.firstFaultOwner === SPARSE_CM12_FRAME_CONTROL_INVALID,
        `step ${step}: immutable oracle FCA1 phase/fault/owner ${authorityOracleFca.phase}/${
          authorityOracleFca.fault}/${authorityOracleFca.firstFaultOwner}`);
        expect(failures, authorityOracleFca.acceptedGeneration === step + 1
          && authorityOracleFca.candidateGeneration === authorityOracleFca.acceptedGeneration
          && authorityOracleFca.sealedGeneration === authorityOracleFca.acceptedGeneration,
        `step ${step}: immutable oracle FCA1 generation accepted/candidate/sealed ${
          authorityOracleFca.acceptedGeneration}/${authorityOracleFca.candidateGeneration}/${
          authorityOracleFca.sealedGeneration}`);
      }
      let fppReceipt: Record<string, number> | undefined;
      if (fppHeader) {
        const h = FPP_HEADER;const flags = fppHeader[h.flags]!;
        expect(failures, fppHeader[h.magic] === SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC
          && fppHeader[h.version] === SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION,
        `step ${step}: FPP1 header is unavailable or incompatible`);
        expect(failures, fppHeader[h.faultCode] === 0
          && fppHeader[h.coverageFaultCount] === 0
          && fppHeader[h.omittedPageCount] === 0,
        `step ${step}: FPP1 fault=${fppHeader[h.faultCode]}, coverage=${
          fppHeader[h.coverageFaultCount]}, omitted=${fppHeader[h.omittedPageCount]}`);
        expect(failures, (flags & (FPP_FLAG.localFaults | FPP_FLAG.globalFault)) === 0
          && (flags & FPP_FLAG.executionComplete) !== 0,
        `step ${step}: FPP1 did not publish a clean execution-complete receipt (${flags})`);
        expect(failures, fppHeader[h.dirtyPageCount] === fppHeader[h.executedPageCount]
          && fppHeader[h.executedPageCount] === fppHeader[h.publishedPageCount],
        `step ${step}: FPP1 scheduled/executed/published mismatch ${
          fppHeader[h.dirtyPageCount]}/${fppHeader[h.executedPageCount]}/${
          fppHeader[h.publishedPageCount]}`);
        expect(failures, fppHeader[h.acceptedGeneration] === fppHeader[h.generationReceipt],
          `step ${step}: FPP1 accepted/receipt generation mismatch`);
        fppReceipt = {
          scheduled: fppHeader[h.dirtyPageCount]!, executed: fppHeader[h.executedPageCount]!,
          published: fppHeader[h.publishedPageCount]!, omitted: fppHeader[h.omittedPageCount]!,
          coverageFaults: fppHeader[h.coverageFaultCount]!, faultCode: fppHeader[h.faultCode]!,
          flags, acceptedGeneration: fppHeader[h.acceptedGeneration]!,
          generationReceipt: fppHeader[h.generationReceipt]!,
        };
      }
      collectTwoToOneFailures(failures, activity.bricks, "acceptedResolution");
      collectTwoToOneFailures(failures, activity.bricks, "candidateResolution");
      collectSubmergedCandidateFailures(failures, activity.bricks);
      const adaptiveDensity = densityReceipt(adaptiveFields.density);
      const allFineDensity = densityReceipt(allFineFields.density);
      const fieldReceipts = {
        density: summarize(adaptiveFields.density),
        velocity: summarize(adaptiveFields.velocity),
        pressure: summarize(adaptiveFields.pressure),
        divergence: summarize(adaptiveFields.divergence),
      };
      for (const [name, receipt] of Object.entries(fieldReceipts)) {
        expect(failures, receipt.count > 0 && receipt.nonFinite === 0,
          `step ${step}: ${name} publication is empty or non-finite`);
      }
      const relativeResidual = stats.pressureRelativeResidual ?? stats.pressureResidual;
      const connectivity = connectivityReceipt(adaptiveFields.density);
      let oracleReceipt: Record<string, unknown> | undefined;
      if (oracleFields && oracleStats && oracleMembership) {
        const localBits = bitExactFieldReceipt(adaptiveFields);
        const oracleBits = bitExactFieldReceipt(oracleFields);
        for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
          expect(failures, localBits[field] === oracleBits[field],
            `step ${step}: pressure oracle physical mismatch in ${field.replace("Sha256", "")}`);
        }
        for (const [domain, local, oracle] of [
          ["cell", adaptivePressureMembership.cell, oracleMembership.cell],
          ["row", adaptivePressureMembership.row, oracleMembership.row],
        ] as const) {
          expect(failures, local.totalCount === oracle.totalCount
            && local.activeBitsSha256 === oracle.activeBitsSha256,
          `step ${step}: pressure oracle ${domain} membership mismatch`);
        }
        for (const field of ["thetaSha256", "coefficientSha256", "rhsSha256"] as const) {
          expect(failures, adaptivePressureMembership[field] === oracleMembership[field],
            `step ${step}: pressure oracle ${field.replace("Sha256", "")} mismatch`);
        }
        const firstDifferentWord = (left: Uint32Array, right: Uint32Array) => {
          const count = Math.min(left.length, right.length);
          for (let id = 0; id < count; id += 1) if (left[id] !== right[id]) return id;
          return left.length === right.length ? -1 : count;
        };
        const edgeMismatch = firstDifferentWord(
          adaptivePressureMembership.qaRaw.coefficientBits,
          oracleMembership.qaRaw.coefficientBits,
        );
        expect(failures, edgeMismatch < 0,
          `step ${step}: PCF full-oracle first edge ${edgeMismatch} local/oracle bits ${
            edgeMismatch < 0 ? "equal" : `${adaptivePressureMembership.qaRaw.coefficientBits[
              edgeMismatch]}/${oracleMembership.qaRaw.coefficientBits[edgeMismatch]}`}`);
        for (const [family, local, oracle] of [
          ["aggregate edge", adaptivePressureMembership.qaRaw.aggregateEdgeBits,
            oracleMembership.qaRaw.aggregateEdgeBits],
          ["brick diagonal", adaptivePressureMembership.qaRaw.brickDiagonalBits,
            oracleMembership.qaRaw.brickDiagonalBits],
          ["hierarchy edge", adaptivePressureMembership.qaRaw.hierarchyEdgeBits,
            oracleMembership.qaRaw.hierarchyEdgeBits],
          ["hierarchy diagonal", adaptivePressureMembership.qaRaw.hierarchyDiagonalBits,
            oracleMembership.qaRaw.hierarchyDiagonalBits],
        ] as const) {
          const mismatch = firstDifferentWord(local, oracle);
          expect(failures, mismatch < 0,
            `step ${step}: PCF full-oracle first ${family} ${mismatch} local/oracle bits ${
              mismatch < 0 ? "equal" : `${local[mismatch]}/${oracle[mismatch]}`}`);
        }
        for (const [bank, local, oracle] of [
          ["A", adaptivePressureMembership.qaRaw.faceABits,
            oracleMembership.qaRaw.faceABits],
          ["B", adaptivePressureMembership.qaRaw.faceBBits,
            oracleMembership.qaRaw.faceBBits],
        ] as const) {
          const mismatch = firstDifferentWord(local, oracle);
          expect(failures, mismatch < 0,
            `step ${step}: FPA projected face bank ${bank} first row ${mismatch} local/oracle bits ${
              mismatch < 0 ? "equal" : `${local[mismatch]}/${oracle[mismatch]}`}`);
        }
        oracleReceipt = {
          bitExact: oracleBits,
          membership: oracleMembership,
          cells: oracleStats.adaptivePressureCellCount,
          activeRows: oracleStats.adaptivePressureActiveRowCount,
          topologyGeneration: oracleStats.adaptiveTopologyShadowGeneration,
          topologyDiagnostic: {
            localGeneration: stats.adaptiveTopologyShadowGeneration,
            oracleGeneration: oracleStats.adaptiveTopologyShadowGeneration,
            localAcceptedCells: stats.adaptiveAcceptedCellCount,
            oracleAcceptedCells: oracleStats.adaptiveAcceptedCellCount,
            localAcceptedRows: stats.adaptiveAcceptedRowCount,
            oracleAcceptedRows: oracleStats.adaptiveAcceptedRowCount,
          },
        };
      }
      let authorityOracleReceipt: BitExactFieldReceipt | undefined;
      const vexLifecycleReceipt = (name: string,
        vex: NonNullable<typeof adaptiveVex>): Record<string, unknown> => {
        const header = vex.header;const invalid = 0xffff_ffff;
        const completed = header[VEX_HEADER.sourceFrameGeneration]!;
        expect(failures, header[VEX_HEADER.faultCount] === 0
          && header[VEX_HEADER.firstFaultCell] === invalid
          && header[VEX_HEADER.firstFaultDepth] === invalid,
        `step ${step}: ${name} VEX2 fault/first ${
          header[VEX_HEADER.faultCount]}/${
          header[VEX_HEADER.firstFaultCell]}/${header[VEX_HEADER.firstFaultDepth]}`);
        expect(failures, completed === step,
          `step ${step}: ${name} VEX2 completed generation ${completed}`);
        return {
          generation: completed,
          topologyGeneration: header[VEX_HEADER.topologyGeneration],
          packetCapacity: header[VEX_HEADER.packetCapacity],
          dispatchPacketCount: vex.dispatchPacketCount,
          validCells: header[VEX_HEADER.validCellCount],
          emptyPackets: header[VEX_HEADER.emptyPacketCount],
          faults: header[VEX_HEADER.faultCount],
          firstFaultCell: header[VEX_HEADER.firstFaultCell],
          firstFaultDepth: header[VEX_HEADER.firstFaultDepth],
        };
      };
      let velocityExtensionReceipt: Record<string, unknown> =
        vexLifecycleReceipt("adaptive", adaptiveVex);
      const allFineVelocityExtensionReceipt = vexLifecycleReceipt("all-fine", allFineVex);
      const authorityOracleVelocityExtensionReceipt = authorityOracleVex
        ? vexLifecycleReceipt("immutable oracle", authorityOracleVex) : undefined;
      if (adaptiveVex && authorityOracleVex) {
        let firstCell = -1;let firstLane = -1;
        const invalid = 0xffff_ffff;
        for (let cell = 0; cell < adaptiveVex.acceptedDepth.length && firstCell < 0; cell += 1) {
          if (adaptiveVex.acceptedDepth[cell] === invalid) continue;
          for (let lane = 0; lane < 4; lane += 1) {
            if (adaptiveVex.velocityBits[4 * cell + lane]
                !== authorityOracleVex.velocityBits[4 * cell + lane]) {
              firstCell = cell;firstLane = lane;break;
            }
          }
        }
        const value = (words: Uint32Array, index: number) => index < 0 ? undefined : words[index];
        const ah = adaptiveVex.header;
        velocityExtensionReceipt = {
          ...velocityExtensionReceipt,
          productionVelocitySha256: sha256Bytes(adaptiveVex.velocityBits),
          legacyVelocitySha256: sha256Bytes(authorityOracleVex.velocityBits),
          firstDifferentCell: firstCell,
          firstDifferentLane: firstLane,
          productionBits: value(adaptiveVex.velocityBits, 4 * firstCell + firstLane),
          legacyBits: value(authorityOracleVex.velocityBits, 4 * firstCell + firstLane),
          acceptedDepth: value(adaptiveVex.acceptedDepth, firstCell),
        };
        expect(failures, firstCell < 0,
          `step ${step}: VEX2 output differs from immutable eight-sweep oracle at cell ${
            firstCell} lane ${firstLane}`);
      }
      if (authorityOracleFields) {
        authorityOracleReceipt = bitExactFieldReceipt(authorityOracleFields);
        const localBits = bitExactFieldReceipt(adaptiveFields);
        for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
          expect(failures, localBits[field] === authorityOracleReceipt[field],
          `step ${step}: FCA authority oracle physical mismatch in ${
              field.replace("Sha256", "")}`);
        }
      }
      expect(failures, fieldReceipts.pressure.maximumAbsolute > 1e-6,
        `step ${step}: pressure publication is identically zero`);
      expect(failures, (stats.pressureIterationsExecuted ?? stats.pressureIterations ?? 0) > 0,
        `step ${step}: pressure solve executed no iterations`);
      expect(failures, relativeResidual !== undefined && Number.isFinite(relativeResidual)
        && relativeResidual <= 5e-3,
      `step ${step}: pressure relative residual ${relativeResidual ?? "missing"} exceeds 5e-3`);
      // Two clean-HEAD Metal captures peak at 0.661-0.679 /s on step five.
      // Keep a narrow deterministic margin while remaining orders of magnitude
      // below the 236-407 /s signature of broken temporal pressure scheduling.
      expect(failures, fieldReceipts.divergence.maximumAbsolute <= 7.5e-1,
        `step ${step}: post-projection divergence ${fieldReceipts.divergence.maximumAbsolute} exceeds 7.5e-1 /s`);
      expect(failures, (stats.adaptivePressureCellCount ?? 0) > 0,
        `step ${step}: pressure-cell compaction receipt is missing or empty`);
      expect(failures, (stats.adaptivePressureActiveRowCount ?? 0) > 0,
        `step ${step}: pressure-active row receipt is missing or empty`);
      expect(failures, (stats.adaptivePressureActiveRowCount ?? Infinity)
        <= (stats.adaptiveAcceptedRowCount ?? -1),
      `step ${step}: pressure-active rows exceed accepted topology rows`);
      expect(failures, Number.isFinite(stats.adaptiveTopologyPreparedBrickCount)
        && Number.isFinite(stats.adaptiveTopologyCommittedBrickCount),
      `step ${step}: prepared/committed topology receipts are missing`);
      expect(failures, connectivity.dominantMassFraction >= 0.98,
        `step ${step}: dominant liquid body contains only ${
          connectivity.dominantMassFraction} of represented mass`);
      const activeBricks = activity.bricks.filter((brick) => brick.active);
      trajectory.push({
        step,
        time_s,
        adaptive: adaptiveDensity,
        allFine: allFineDensity,
        bitExact: {
          adaptive: bitExactFieldReceipt(adaptiveFields),
          allFine: bitExactFieldReceipt(allFineFields),
        },
        densityRelativeL1: relativeDensityL1(allFineFields.density, adaptiveFields.density),
        fields: fieldReceipts,
        connectivity,
        frameControl: adaptiveFca,
        finalScalarMasks: adaptiveSaw,
        ...(authorityOracleSaw ? { finalScalarMasksOracle: authorityOracleSaw } : {}),
        ...(authorityOracleFca ? { frameControlOracle: authorityOracleFca } : {}),
        velocityExtension: velocityExtensionReceipt,
        allFineVelocityExtension: allFineVelocityExtensionReceipt,
        ...(authorityOracleVelocityExtensionReceipt
          ? { authorityOracleVelocityExtension: authorityOracleVelocityExtensionReceipt } : {}),
        ...(authorityOracleReceipt ? { authorityOracle: {
          ...authorityOracleReceipt,
          differences: {
            density: fieldByteDifference(adaptiveFields.density, authorityOracleFields!.density),
            velocity: fieldByteDifference(adaptiveFields.velocity, authorityOracleFields!.velocity),
            pressure: fieldByteDifference(adaptiveFields.pressure, authorityOracleFields!.pressure),
            divergence: fieldByteDifference(adaptiveFields.divergence,
              authorityOracleFields!.divergence),
          },
        } } : {}),
        pressure: {
          cells: stats.adaptivePressureCellCount,
          activeRows: stats.adaptivePressureActiveRowCount,
          iterations: stats.pressureIterations,
          iterationsExecuted: stats.pressureIterationsExecuted,
          relativeResidual,
          residualDrift: stats.pressureResidualDrift,
          membership: adaptivePressureMembership,
          topologyRepair: stats.adaptivePressureTopologyRepair,
        },
        allFinePressure: {
          cells: allFineStats.adaptivePressureCellCount,
          activeRows: allFineStats.adaptivePressureActiveRowCount,
          iterations: allFineStats.pressureIterations,
          iterationsExecuted: allFineStats.pressureIterationsExecuted,
          relativeResidual: allFineStats.pressureRelativeResidual ?? allFineStats.pressureResidual,
          residualDrift: allFineStats.pressureResidualDrift,
          membership: allFinePressureMembership,
        },
        ...(oracleReceipt ? { pressureOracle: oracleReceipt } : {}),
        ...(fppReceipt ? { presentation: fppReceipt } : {}),
        topology: {
          activeMaximumFineCellX: Math.max(...activeBricks.map(
            (brick) => TARGET_BRICK_FINE_RESOLUTION * (brick.coordinate[0] + 1) - 1)),
          prepared: stats.adaptiveTopologyPreparedBrickCount,
          committed: stats.adaptiveTopologyCommittedBrickCount,
          deferred: stats.adaptiveTopologyDeferredBrickCount,
          generation: stats.adaptiveTopologyShadowGeneration,
          acceptedCells: stats.adaptiveAcceptedCellCount,
          acceptedRows: stats.adaptiveAcceptedRowCount,
          accepted: resolutionHistogram(activity.bricks, "acceptedResolution"),
          candidate: resolutionHistogram(activity.bricks, "candidateResolution"),
          deeplySubmerged: deeplySubmergedBricks(activity.bricks).length,
          aggressiveSubmerged: activity.bricks.filter(
            (brick) => brick.planReasons === 2048).length,
          bricks: activity.bricks.map((brick) => ({
            coordinate: brick.coordinate,
            active: brick.active,
            acceptedResolution: brick.acceptedResolution,
            candidateResolution: brick.candidateResolution,
          })),
        },
        ...(authorityOracleActivity && authorityOracleStats ? { authorityOracleTopology: {
          generation: authorityOracleStats.adaptiveTopologyShadowGeneration,
          acceptedCells: authorityOracleStats.adaptiveAcceptedCellCount,
          acceptedRows: authorityOracleStats.adaptiveAcceptedRowCount,
          accepted: resolutionHistogram(authorityOracleActivity.bricks, "acceptedResolution"),
          candidate: resolutionHistogram(authorityOracleActivity.bricks, "candidateResolution"),
          bricks: authorityOracleActivity.bricks.map((brick) => ({
            coordinate: brick.coordinate,
            active: brick.active,
            acceptedResolution: brick.acceptedResolution,
            candidateResolution: brick.candidateResolution,
          })),
        } } : {}),
      });
      if (physicalReference) {
        const candidate = bitExactFieldReceipt(adaptiveFields);
        const reference = physicalReference.trajectory[step - 1]!;
        for (const field of BIT_EXACT_PHYSICAL_FIELDS) {
          expect(failures, candidate[field] === reference[field],
            `step ${step}: physical byte mismatch in ${field.replace("Sha256", "")}`);
        }
      }
    }

    const final = trajectory.at(-1)! as {
      adaptive: ReturnType<typeof densityReceipt>;
      densityRelativeL1: number;
      topology: {
        activeMaximumFineCellX: number;
        prepared: number;
        committed: number;
        generation: number;
        acceptedCells: number;
        deeplySubmerged: number;
      };
    };
    const maximumMassDrift = Math.max(...trajectory.flatMap((entry) => {
      const sample = entry as { adaptive: ReturnType<typeof densityReceipt>;
        allFine: ReturnType<typeof densityReceipt> };
      return [Math.abs(sample.adaptive.mass - initial.adaptive.mass) / initial.adaptive.mass,
        Math.abs(sample.allFine.mass - initial.allFine.mass) / initial.allFine.mass];
    }));
    expect(failures, maximumMassDrift <= 2e-3,
      `maximum mass drift ${maximumMassDrift} exceeds 0.2%`);
    expect(failures, final.adaptive.front.surface >= 56,
      `surface front stopped at x=${final.adaptive.front.surface}; expected at least x=56`);
    expect(failures, final.densityRelativeL1 <= 0.06,
      `final adaptive/all-fine density relative L1 ${final.densityRelativeL1} exceeds 0.06`);
    expect(failures, trajectory.every((entry, index) => index === 0
      || (entry as { adaptive: ReturnType<typeof densityReceipt> }).adaptive.front.surface
        >= (trajectory[index - 1] as { adaptive: ReturnType<typeof densityReceipt> })
          .adaptive.front.surface),
    "surface front retreated during release");
    expect(failures, trajectory.every((entry) => {
      const sample = entry as { adaptive: ReturnType<typeof densityReceipt>;
        allFine: ReturnType<typeof densityReceipt> };
      return Math.abs(sample.adaptive.front.surface - sample.allFine.front.surface) <= 1
        && Math.abs(sample.adaptive.front.liquid - sample.allFine.front.liquid) <= 1;
    }), "adaptive and all-fine fronts disagree by more than one fine cell");
    expect(failures, trajectory.every((entry, index) => index === 0
      || (entry as { topology: { generation: number } }).topology.generation
        >= (trajectory[index - 1] as { topology: { generation: number } }).topology.generation),
    "topology generation regressed during the moving front");
    expect(failures,
      (trajectory.at(-1) as { topology: { generation: number } }).topology.generation
        > (trajectory[0] as { topology: { generation: number } }).topology.generation,
    "the moving-front lane exercised no accepted topology generation change");
    expect(failures, trajectory.every((entry) => {
      const topology = (entry as { topology: { prepared: number; committed: number } }).topology;
      return topology.prepared === topology.committed;
    }), "a prepared topology transition failed its conservation commit");
    expect(failures, final.topology.deeplySubmerged > 0,
      "final topology has no deeply submerged bulk region");
    expect(failures, trajectory.some((entry) =>
      (entry as { topology: { aggressiveSubmerged: number } })
        .topology.aggressiveSubmerged > 0),
    "aggressive submerged coarsening was not exercised");
    expect(failures, final.topology.acceptedCells
      < (initialStats.adaptiveAcceptedCellCount ?? Infinity),
    "submerged coarsening did not reduce accepted pressure-cell work");
    expect(failures, final.topology.activeMaximumFineCellX >= final.adaptive.front.trace,
      `trace front x=${final.adaptive.front.trace} escaped residency x=${
        final.topology.activeMaximumFineCellX}`);

    const finalStats = await adaptive.readStats();
    expect(failures, (finalStats.adaptiveAcceptedSameLevelCoarseRowCount ?? 0) > 0,
      "final topology has no same-level coarse pressure rows");
    expect(failures, (finalStats.adaptiveAcceptedMixedSeamRowCount ?? 0) > 0,
      "final topology has no mixed-resolution pressure rows");
    const scopedError = await device.popErrorScope();
    if (scopedError) validationErrors.push(scopedError.message);
    for (const message of validationErrors) failures.push(`WebGPU validation error: ${message}`);
    const report = {
      passed: failures.length === 0,
      lane: "dam-front-64-cubed",
      scene: scene.sceneId,
      backend,
      grid: [64, 64, 64],
      configuration: {
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
      },
      steps,
      dt_s: CM12_PAPER_DT_S,
      simulatedDuration_s: steps * CM12_PAPER_DT_S,
      runMode: hasFlag("short-smoke") ? "short-smoke" : "canonical",
      pressureOraclePaired,
      authorityOraclePaired,
      physicalReference: physicalReferencePath ? {
        path: physicalReferencePath,
        bitExact: failures.every((failure) => !failure.includes("physical byte mismatch")),
      } : undefined,
      thresholds: {
        maximumMassDrift: 2e-3,
        minimumSurfaceFrontX: 56,
        maximumFrontDisagreementCells: 1,
        maximumDensityRelativeL1: 0.06,
        maximumPressureRelativeResidual: 5e-3,
        maximumPostProjectionDivergence_s: 7.5e-1,
        minimumDominantBodyMassFraction: 0.98,
      },
      initial: {
        density: initial.adaptive,
        bitExact: initial.adaptiveBitExact,
        acceptedCells: initialStats.adaptiveAcceptedCellCount,
        accepted: resolutionHistogram(initialActivity.bricks, "acceptedResolution"),
      },
      maximumMassDrift,
      failures,
      trajectory,
      checkpointSteps: [...checkpointStepSet],
      checkpoints: trajectory.filter((entry) => checkpointStepSet.has(Number(entry.step))),
    };
    console.log(JSON.stringify(report, null, 2));
    if (failures.length > 0) {
      throw new Error(`64^3 dam-front regression failed (${failures.length} gates)`);
    }
  } finally {
    adaptive?.destroy();
    allFine?.destroy();
    pressureOracle?.destroy();
    authorityOracle?.destroy();
    if (device) {
      const compiler = gpuCompilationManagerFor(device);
      invalidateGPUCompilationManager(device, "dam-front QA lane complete");
      await compiler.whenIdle();
      try { await device.queue.onSubmittedWorkDone(); } catch { /* Device fault already reported. */ }
      device.destroy();
      // Keep the Dawn instance strongly reachable while its final native
      // ProcessEvents callback observes device retirement.
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
    gpu = undefined;
    await releaseWebGPUExclusiveLock();
  }
}

async function waitForDamHardwareTrace(
  solver: WebGPUAdaptiveMassSolver,
  expectedTime_s: number,
): Promise<{ trace: PerformanceTrace; stats: Awaited<ReturnType<typeof solver.readStats>> }> {
  let stats = await solver.readStats();
  const identity = stats.physicsCaptureIdentity;
  if (!identity || identity.context !== `adaptive-mass:sim-${expectedTime_s.toFixed(6)}`) {
    throw new Error(`step ${expectedTime_s / CM12_PAPER_DT_S}: hardware capture was not authored`);
  }
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const trace = stats.physicsTrace;
    if (trace?.sampleId === identity.sampleId && trace.context === identity.context) {
      if (trace.measurementSource !== "gpu-hardware-timestamp") {
        throw new Error(`step ${expectedTime_s / CM12_PAPER_DT_S}: measurement fell back to ${
          trace.measurementSource ?? "unknown"}`);
      }
      const accounted_ms = trace.phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
      if (Math.abs(accounted_ms - trace.total_ms) > 1e-6) {
        throw new Error(`step ${expectedTime_s / CM12_PAPER_DT_S}: hardware partition closure is ${
          accounted_ms - trace.total_ms} ms`);
      }
      return { trace, stats };
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
    stats = await solver.readStats();
  }
  throw new Error(`step ${expectedTime_s / CM12_PAPER_DT_S}: hardware trace did not resolve`);
}

async function runDamFrontPerformanceLane(): Promise<void> {
  const steps = 5;
  const replays = positiveInteger("performance-replays", 5);
  if (replays < 3) throw new RangeError("performance-replays must be at least 3");
  const captureGap_ms = positiveInteger("performance-capture-gap-ms", 110);
  const topologyCadenceSteps = positiveInteger("topology-cadence", 1);
  if (captureGap_ms < 100) {
    throw new RangeError("performance-capture-gap-ms must be at least the 100 ms trace cadence");
  }
  const recordPath = argument("record-performance-baseline");
  const baselinePath = argument("performance-baseline")
    ?? (recordPath ? undefined : DEFAULT_DAM_PERFORMANCE_BASELINE);
  if (recordPath && argument("performance-baseline")) {
    throw new RangeError("record-performance-baseline and performance-baseline are mutually exclusive");
  }
  let baseline: ReturnType<typeof parseDamPerformanceBaseline> | undefined;
  if (baselinePath) {
    let contents: string;
    try {
      contents = await readFile(baselinePath, "utf8");
    } catch (error) {
      throw new Error(`dam performance baseline is required and unreadable: ${baselinePath}`,
        { cause: error });
    }
    baseline = parseDamPerformanceBaseline(JSON.parse(contents) as unknown);
  }

  const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
  const failures: string[] = [];
  const nonPressureSamples: number[] = [];
  const nonPressureByStep = Array.from({ length: steps }, () => [] as number[]);
  const stageSamples = new Map<string, number[]>();
  const stageSamplesByStep = new Map<string, number[][]>();
  const pathReceipts: Array<Record<string, unknown>> = [];
  const physicsReplays: DamPerformancePhysicsReceipt[] = [];
  const topologyHeavyReplays: boolean[] = [];
  const validationErrors: string[] = [];
  const pathObserver = new DamStagePathObserver();
  let device: GPUDevice | undefined;
  usePerformanceInstrumentationStore.getState().setMode("off");
  await acquireWebGPUExclusiveLock("dawn-acceptance",
    "tools/run-sparse-cm12-temporal-regressions.ts:dam-front-performance");
  try {
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error(`No Dawn adapter is available for backend ${backend}`);
    if (!adapter.features.has("timestamp-query")) {
      throw new Error("dam performance lane requires hardware timestamp queries");
    }
    device = await adapter.requestDevice({
      requiredFeatures: ["timestamp-query" as GPUFeatureName],
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");
    const observedDevice = pathObserver.wrap(device);
    const solverArguments = () => [
      observedDevice, createMinimalPowerDamBreak64Scene(), "balanced" as const, undefined, {
        resolutionMode: "adaptive",
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
        surfaceFineRings: 8,
        activityPolicy: { ...DAM_ACTIVITY_POLICY, topologyCadenceSteps },
        timeStep: "paper" as const,
      }, () => {},
    ] as const;
    const createDamSolver = () =>
      WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(...solverArguments());

    // One complete replay pays Dawn/Metal's deferred shader compilation on
    // this exact device and pipeline variant. Instrumentation remains off, so
    // none of its five advances can enter a timing distribution.
    const warmupSolver = await createDamSolver();
    try {
      for (let step = 1; step <= steps; step += 1) {
        if (!warmupSolver.advanceTo(step * CM12_PAPER_DT_S, [])) {
          throw new Error(`unmeasured warmup step ${step} did not encode`);
        }
        await device.queue.onSubmittedWorkDone();
      }
      await warmupSolver.readStats();
    } finally {
      warmupSolver.destroy();
    }
    usePerformanceInstrumentationStore.getState().setMode("timeline");

    for (let replay = 0; replay < replays; replay += 1) {
      const solver = await createDamSolver();
      try {
        // Construction, uploads, and this initial QA publication are setup;
        // none is admitted to a performance sample.
        const initialFields = await solver.readDiagnosticFields();
        const initialDensity = densityReceipt(initialFields.density);
        expect(failures, initialFields.density.length === 64 ** 3
          && summarize(initialFields.density).nonFinite === 0,
        `replay ${replay + 1}: initial density publication is incomplete`);
        expect(failures, initialDensity.front.surface === 39,
          `replay ${replay + 1}: initial surface front is ${initialDensity.front.surface}, expected 39`);
        const physicsSteps: DamPerformancePhysicsStep[] = [];
        const generations: number[] = [];
        let committedTopology = false;
        for (let step = 1; step <= steps; step += 1) {
          await new Promise((resolve) => setTimeout(resolve, captureGap_ms));
          const time_s = step * CM12_PAPER_DT_S;
          pathObserver.beginFrame();
          if (!solver.advanceTo(time_s, [])) {
            throw new Error(`replay ${replay + 1} step ${step}: advance did not encode`);
          }
          const observedPaths = pathObserver.finishFrame();
          const { trace, stats } = await waitForDamHardwareTrace(solver, time_s);
          const nonPressure_ms = trace.phases.reduce((sum, phase) =>
            sum + (phase.id === "pressure-solve" ? 0 : phase.duration_ms), 0);
          nonPressureSamples.push(nonPressure_ms);
          nonPressureByStep[step - 1]!.push(nonPressure_ms);
          const costs = fluidPipelinePhaseCosts(trace);
          for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
            if (stage.costInsideStage) continue;
            const measurement = measureFluidPipelineStage(
              stage, ADAPTIVE_MASS_FLUID_PIPELINE.stages, costs, trace.total_ms, "on");
            if (measurement.duration_ms !== undefined) {
              const samples = stageSamples.get(stage.id) ?? [];
              samples.push(measurement.duration_ms);
              stageSamples.set(stage.id, samples);
              const perStep = stageSamplesByStep.get(stage.id)
                ?? Array.from({ length: steps }, () => [] as number[]);
              perStep[step - 1]!.push(measurement.duration_ms);
              stageSamplesByStep.set(stage.id, perStep);
            }
            const kinds = observedPaths[stage.id]?.modes ?? [];
            expect(failures, !(kinds.includes("incremental") && kinds.includes("global")),
              `replay ${replay + 1} step ${step}: ${stage.id} ran incremental and global work`);
          }
          pathReceipts.push({ replay: replay + 1, step, stages: observedPaths });

          // QA readbacks happen after the captured advance and therefore do
          // not contaminate its hardware stage distribution.
          const fields = await solver.readDiagnosticFields();
          const fieldReceipts = {
            density: summarize(fields.density), velocity: summarize(fields.velocity),
            pressure: summarize(fields.pressure), divergence: summarize(fields.divergence),
          };
          const activity = await solver.readGPUActivityPolicy();
          const density = densityReceipt(fields.density);
          const connectivity = connectivityReceipt(fields.density);
          const relativeResidual = stats.pressureRelativeResidual ?? stats.pressureResidual;
          const generation = stats.adaptiveTopologyShadowGeneration;
          if (typeof generation === "number") generations.push(generation);
          committedTopology ||= (stats.adaptiveTopologyCommittedBrickCount ?? 0) > 0;
          const physicsStep: DamPerformancePhysicsStep = {
            step,
            density,
            connectivity,
            fields: fieldReceipts,
            pressure: {
              cells: stats.adaptivePressureCellCount,
              activeRows: stats.adaptivePressureActiveRowCount,
              iterations: stats.pressureIterations,
              iterationsExecuted: stats.pressureIterationsExecuted,
              relativeResidual,
              residualDrift: stats.pressureResidualDrift,
            },
            topology: {
              generation,
              prepared: stats.adaptiveTopologyPreparedBrickCount,
              committed: stats.adaptiveTopologyCommittedBrickCount,
              deferred: stats.adaptiveTopologyDeferredBrickCount,
              acceptedCells: stats.adaptiveAcceptedCellCount,
              acceptedRows: stats.adaptiveAcceptedRowCount,
              deeplySubmerged: deeplySubmergedBricks(activity.bricks).length,
            },
          };
          physicsSteps.push(physicsStep);
          expect(failures, fieldReceipts.pressure.maximumAbsolute > 1e-6,
            `replay ${replay + 1} step ${step}: pressure publication is zero`);
          expect(failures, fieldReceipts.divergence.maximumAbsolute <= 7.5e-1,
            `replay ${replay + 1} step ${step}: divergence physics gate changed`);
          expect(failures, relativeResidual !== undefined && Number.isFinite(relativeResidual)
            && relativeResidual <= 5e-3,
          `replay ${replay + 1} step ${step}: pressure residual physics gate changed`);
          expect(failures, connectivity.dominantMassFraction >= 0.98,
            `replay ${replay + 1} step ${step}: connectivity physics gate changed`);
          expect(failures, (stats.adaptivePressureCellCount ?? 0) > 0
            && (stats.adaptivePressureActiveRowCount ?? 0) > 0,
          `replay ${replay + 1} step ${step}: pressure topology receipt is empty`);
          expect(failures, (stats.pressureIterationsExecuted ?? stats.pressureIterations ?? 0) > 0,
            `replay ${replay + 1} step ${step}: pressure solve executed no iterations`);
        }
        const physics = { initialDensity, steps: physicsSteps };
        physicsReplays.push(physics);
        const topologyHeavy = committedTopology && generations.length === steps
          && generations.at(-1)! > generations[0]!;
        topologyHeavyReplays.push(topologyHeavy);
        expect(failures, topologyHeavy,
          `replay ${replay + 1}: five-step window did not exercise changing topology`);
        const initialMass = initialDensity.mass;
        const maximumMassDrift = Math.max(...physicsSteps.map((receipt) =>
          Math.abs(receipt.density.mass - initialMass) / initialMass));
        expect(failures, maximumMassDrift <= 2e-3,
          `replay ${replay + 1}: mass drift ${maximumMassDrift} exceeds dam gate`);
        expect(failures, physicsSteps.at(-1)!.density.front.surface >= 56,
          `replay ${replay + 1}: final surface front failed the dam gate`);
        expect(failures, physicsSteps.every((receipt, index) => index === 0
          || receipt.density.front.surface >= physicsSteps[index - 1]!.density.front.surface),
        `replay ${replay + 1}: surface front retreated during release`);
      } finally {
        solver.destroy();
      }
    }

    const physicsReference = physicsReplays[0]!;
    const replayEquivalence = physicsReplays.map((receipt) =>
      comparePhysicsReceipts(physicsReference, receipt));
    for (let replay = 1; replay < physicsReplays.length; replay += 1) {
      for (const difference of replayEquivalence[replay]!.differences) failures.push(
        `replay ${replay + 1} physics equivalence ${difference.path}: ${
          difference.candidate} vs ${difference.reference}; ${difference.comparison} delta ${
          difference.delta} exceeds ${difference.tolerance}`,
      );
    }
    const baselineEquivalence = baseline
      ? comparePhysicsReceipts(baseline.physicsReference, physicsReference) : undefined;
    if (baseline) {
      for (const difference of baselineEquivalence!.differences) failures.push(
        `baseline physics equivalence ${difference.path}: ${difference.candidate} vs ${
          difference.reference}; ${difference.comparison} delta ${difference.delta} exceeds ${
          difference.tolerance}`,
      );
    }
    const nonPressureP95_ms = percentile(nonPressureSamples, 0.95);
    const nonPressure = timingDistribution(nonPressureSamples);
    const allowedP95_ms = baseline
      ? Math.max(baseline.nonPressureP95_ms * 1.05, baseline.nonPressureP95_ms + 0.5)
      : undefined;
    if (allowedP95_ms !== undefined) {
      expect(failures, nonPressureP95_ms <= allowedP95_ms,
        `combined non-pressure p95 ${nonPressureP95_ms} ms exceeds ${allowedP95_ms} ms`);
    }
    const scopedError = await device.popErrorScope();
    if (scopedError) validationErrors.push(scopedError.message);
    for (const message of validationErrors) failures.push(`WebGPU validation error: ${message}`);
    const stages = Object.fromEntries([...stageSamples].map(([stageId, values]) => [stageId, {
      ...timingDistribution(values),
      perStep: stageSamplesByStep.get(stageId)!.map((stepValues, index) => ({
        step: index + 1, ...timingDistribution(stepValues),
      })),
    }]));
    const report = {
      kind: "sparse-cm12-dam-front64-performance",
      version: 1,
      passed: failures.length === 0,
      lane: "dam-front-performance",
      scene: "minimal-power-dam-break-64",
      backend,
      grid: [64, 64, 64],
      configuration: {
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
        transport: "compiled-topology",
        topologyCadenceSteps,
        measurementSource: "gpu-hardware-timestamp",
        setupExcluded: true,
      },
      warmup: {
        replays: 1,
        steps,
        measurement: "disabled",
        sameDeviceAsMeasuredReplays: true,
      },
      steps,
      replays,
      dt_s: CM12_PAPER_DT_S,
      nonPressure: {
        ...nonPressure,
        scope: "per-frame sum of every hardware GPU phase except pressure-solve",
        perStep: nonPressureByStep.map((values, index) => ({
          step: index + 1, ...timingDistribution(values),
        })),
      },
      stages,
      pathExclusivity: {
        passed: !failures.some((failure) => failure.includes("incremental and global")),
        rule: "a stage may run incremental or global work in one frame, never both",
        samples: pathReceipts,
      },
      topologyHeavyReplays,
      physicsReference,
      physicsEquivalenceThresholds: DAM_PERFORMANCE_PHYSICS_EQUIVALENCE,
      physicsReplayEquality: replayEquivalence.map((result) => result.equivalent),
      physicsReplayEquivalence: replayEquivalence,
      baseline: baseline ? {
        path: baselinePath,
        nonPressureP95_ms: baseline.nonPressureP95_ms,
        allowedP95_ms: Number(allowedP95_ms!.toFixed(4)),
        rule: "candidate p95 <= max(baseline * 1.05, baseline + 0.5 ms)",
        physicsUnchanged: baselineEquivalence!.equivalent,
        physicsDifferences: baselineEquivalence!.differences,
      } : { recording: recordPath },
      failures,
      validationErrors,
    };
    console.log(JSON.stringify(report, null, 2));
    const allowFailingExperimentRecord = hasFlag("allow-failing-experiment-record");
    if (recordPath && (failures.length === 0 || allowFailingExperimentRecord)) {
      await writeFile(recordPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    }
    if (recordPath && failures.length > 0 && !allowFailingExperimentRecord) {
      throw new Error("refusing to record a failing dam performance baseline");
    }
    if (failures.length > 0 && !allowFailingExperimentRecord) {
      throw new Error(`dam-front performance regression failed (${failures.length} gates)`);
    }
  } finally {
    device?.destroy();
    usePerformanceInstrumentationStore.getState().setMode("off");
    await releaseWebGPUExclusiveLock();
  }
}

async function runWeakenedSymmetryLane(): Promise<void> {
  const steps = physicsLaneSteps("symmetry-steps");
  const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
  const dimensions = [32, 16, 32] as const;
  const thresholds = {
    maximumMassDrift: 5e-3,
    minimumDominantBodyMassFraction: 0.95,
    maximumDensityD4Error: 5e-3,
    maximumVelocityD4Error_m_s: 2e-3,
    maximumPressureD4Error: 1,
    // A one-level mismatch is tolerated by this deliberately weak tripwire;
    // the dam lane remains authoritative for topology conservation/coverage.
    maximumTopologyD4Error: 1,
    maximumPressureRelativeResidual: 1e-5,
    maximumPostProjectionDivergence_s: 1e-3,
  } as const;
  await acquireWebGPUExclusiveLock("dawn-acceptance",
    "tools/run-sparse-cm12-temporal-regressions.ts:symmetric-expansion-weakened");
  let gpu: GPU | undefined;
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const failures: string[] = [];
  try {
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error(`No Dawn adapter is available for backend ${backend}`);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });
    device.pushErrorScope("validation");

    const scene = createSymmetricExpansionScene();
    scene.voxelDomain.finestCellSize_m = scene.container.width_m / dimensions[0];
    const brickSize = scene.voxelDomain.brickSize_cells;
    const brickGrid = dimensions.map((value) => value / brickSize) as
      [number, number, number];
    scene.fluid.initialBrickSeeds_m = [];
    for (let bz = brickGrid[2] / 4; bz < 3 * brickGrid[2] / 4; bz += 1)
      for (let by = 0; by < brickGrid[1] / 2; by += 1)
        for (let bx = brickGrid[0] / 4; bx < 3 * brickGrid[0] / 4; bx += 1) {
          scene.fluid.initialBrickSeeds_m.push({
            x: -0.5 * scene.container.width_m
              + (bx + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
            y: (by + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
            z: -0.5 * scene.container.depth_m
              + (bz + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
          });
        }
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device, scene, "balanced", undefined,
      { resolutionMode: "adaptive",
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
        timeStep: "paper" },
      () => {},
    );
    expect(failures, solver.info.nx === dimensions[0]
      && solver.info.ny === dimensions[1] && solver.info.nz === dimensions[2],
    `symmetric grid is ${solver.info.nx}x${solver.info.ny}x${solver.info.nz}, expected ${
      dimensions.join("x")}`);
    const initialFields = await solver.readDiagnosticFields();
    const initialSummary = summarize(initialFields.density);
    const initialMass = initialFields.density.reduce((sum, value) => sum + Math.max(0, value), 0);
    expect(failures, initialSummary.count === dimensions[0] * dimensions[1] * dimensions[2]
      && initialSummary.nonFinite === 0 && initialMass > 0,
    "initial symmetric density publication is missing, non-finite, or empty");

    const checkpoints: Array<Record<string, unknown>> = [];
    let previousGeneration: number | undefined;
    let previousFrameControlGeneration = 1;
    let vexFailureCaptureAttempted = false;
    let vexFirstFailure: Record<string, unknown> | undefined;
    for (let step = 1; step <= steps; step += 1) {
      const target_s = step * CM12_PAPER_DT_S;
      expect(failures, solver.advanceTo(target_s, []),
        `step ${step}: symmetric advance did not encode`);
      await device.queue.onSubmittedWorkDone();
      const [fields, activity, stats, frameControl, scalarResult] = await Promise.all([
        solver.readDiagnosticFields(), solver.readGPUActivityPolicy(), solver.readStats(),
        solver.readFrameControlQA(), solver.readFinalScalarMaskHeaderQA(),
      ]);
      const expectedGeneration = step + 1;
      const frameControlStalled = frameControl.acceptedGeneration
        <= previousFrameControlGeneration;
      const frameControlValid = frameControl.phase === SPARSE_CM12_FRAME_CONTROL_PHASE.accepted
        && frameControl.fault === 0 && !frameControlStalled
        && frameControl.acceptedGeneration === expectedGeneration
        && frameControl.candidateGeneration === expectedGeneration
        && frameControl.sealedGeneration === expectedGeneration;
      const scalarResultValid = scalarResult.phase
          === SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.published
        && scalarResult.fault === 0
        && scalarResult.generation === expectedGeneration;
      expect(failures, frameControlValid,
        `step ${step}: FCA1 phase/fault/accepted/candidate/sealed/stalled ${
          frameControl.phase}/${frameControl.fault}/${frameControl.acceptedGeneration}/${
          frameControl.candidateGeneration}/${frameControl.sealedGeneration}/${
          Number(frameControlStalled)}`);
      expect(failures, scalarResultValid,
        `step ${step}: FSM1 phase/fault/generation ${scalarResult.phase}/${
          scalarResult.fault}/${scalarResult.generation}`);
      if ((!frameControlValid || !scalarResultValid) && !vexFailureCaptureAttempted) {
        vexFailureCaptureAttempted = true;
        try {
          const vex = await solver.readVelocityExtensionQA();
          const header = vex.header;
          const cell = header[VEX_HEADER.firstFaultCell]!;
          const cellValid = cell !== SPARSE_CM12_FRAME_CONTROL_INVALID
            && cell < vex.acceptedDepth.length;
          vexFirstFailure = {
            step,
            frameControl,
            scalarResult,
            header: {
              sourceFrameGeneration: header[VEX_HEADER.sourceFrameGeneration],
              topologyGeneration: header[VEX_HEADER.topologyGeneration],
              packetCapacity: header[VEX_HEADER.packetCapacity],
              dispatchPacketCount: vex.dispatchPacketCount,
              validCellCount: header[VEX_HEADER.validCellCount],
              emptyPacketCount: header[VEX_HEADER.emptyPacketCount],
              faultCount: header[VEX_HEADER.faultCount],
              firstFaultCell: cell,
              firstFaultDepth: header[VEX_HEADER.firstFaultDepth],
            },
            firstCell: cellValid ? {
              acceptedDepth: vex.acceptedDepth[cell],
              acceptedVelocityBits: [...vex.velocityBits.slice(4 * cell, 4 * cell + 4)],
            } : undefined,
          };
        } catch (error) {
          vexFirstFailure = { step, frameControl, scalarResult,
            captureError: error instanceof Error ? error.message : String(error) };
          failures.push(`step ${step}: failed to capture first VEX2 fault detail`);
        }
      }
      previousFrameControlGeneration = frameControl.acceptedGeneration;
      const densitySummary = summarize(fields.density);
      const velocitySummary = summarize(fields.velocity);
      const pressureSummary = summarize(fields.pressure);
      const divergenceSummary = summarize(fields.divergence);
      const densitySymmetry = scalarD4(fields.density, dimensions);
      const velocitySymmetry = velocityD4(fields.velocity, dimensions);
      const pressureSymmetry = scalarD4(fields.pressure, dimensions);
      const topologyField = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
      for (const brick of activity.bricks) {
        const scale = TARGET_BRICK_FINE_RESOLUTION / brick.acceptedResolution;
        for (let z = 0; z < TARGET_BRICK_FINE_RESOLUTION; z += 1)
          for (let y = 0; y < TARGET_BRICK_FINE_RESOLUTION; y += 1)
            for (let x = 0; x < TARGET_BRICK_FINE_RESOLUTION; x += 1) {
            const qx = TARGET_BRICK_FINE_RESOLUTION * brick.coordinate[0] + x;
            const qy = TARGET_BRICK_FINE_RESOLUTION * brick.coordinate[1] + y;
            const qz = TARGET_BRICK_FINE_RESOLUTION * brick.coordinate[2] + z;
            if (qx < dimensions[0] && qy < dimensions[1] && qz < dimensions[2])
              topologyField[qx + dimensions[0] * (qy + dimensions[1] * qz)] = scale;
          }
      }
      const topologySymmetry = scalarD4(topologyField, dimensions);
      const mass = fields.density.reduce((sum, value) => sum + Math.max(0, value), 0);
      let normalizedDensityChange = 0;
      for (let index = 0; index < fields.density.length; index += 1)
        normalizedDensityChange += Math.abs(fields.density[index]! - initialFields.density[index]!);
      normalizedDensityChange /= initialMass;
      const connectivity = genericConnectivityReceipt(fields.density, dimensions);
      const relativeResidual = stats.pressureRelativeResidual ?? stats.pressureResidual;
      for (const [name, summary] of Object.entries({ density: densitySummary,
        velocity: velocitySummary, pressure: pressureSummary, divergence: divergenceSummary })) {
        expect(failures, summary.count > 0 && summary.nonFinite === 0,
          `step ${step}: ${name} publication is absent or non-finite`);
      }
      expect(failures, Math.abs(mass - initialMass) / initialMass
        <= thresholds.maximumMassDrift,
      `step ${step}: mass drift exceeds ${thresholds.maximumMassDrift}`);
      expect(failures, connectivity.dominantMassFraction
        >= thresholds.minimumDominantBodyMassFraction,
      `step ${step}: dominant body fraction ${connectivity.dominantMassFraction} is below ${
        thresholds.minimumDominantBodyMassFraction}`);
      expect(failures, densitySymmetry.maximumAbsoluteError
        <= thresholds.maximumDensityD4Error,
      `step ${step}: weakened density D4 threshold failed (${densitySymmetry.maximumAbsoluteError})`);
      expect(failures, velocitySymmetry.maximumAbsoluteError
        <= thresholds.maximumVelocityD4Error_m_s,
      `step ${step}: weakened velocity D4 threshold failed (${velocitySymmetry.maximumAbsoluteError})`);
      expect(failures, pressureSymmetry.maximumAbsoluteError
        <= thresholds.maximumPressureD4Error,
      `step ${step}: weakened pressure D4 threshold failed (${pressureSymmetry.maximumAbsoluteError})`);
      expect(failures, topologySymmetry.maximumAbsoluteError
        <= thresholds.maximumTopologyD4Error,
      `step ${step}: weakened topology D4 threshold failed (${topologySymmetry.maximumAbsoluteError})`);
      expect(failures, pressureSummary.maximumAbsolute > 1e-6,
        `step ${step}: pressure publication is identically zero`);
      expect(failures, (stats.pressureIterationsExecuted ?? stats.pressureIterations ?? 0) > 0,
        `step ${step}: pressure solve executed no iterations`);
      expect(failures, relativeResidual !== undefined && Number.isFinite(relativeResidual)
        && relativeResidual <= thresholds.maximumPressureRelativeResidual,
      `step ${step}: pressure relative residual ${relativeResidual ?? "missing"} exceeds ${
        thresholds.maximumPressureRelativeResidual}`);
      expect(failures, divergenceSummary.maximumAbsolute
        <= thresholds.maximumPostProjectionDivergence_s,
      `step ${step}: divergence ${divergenceSummary.maximumAbsolute} exceeds ${
        thresholds.maximumPostProjectionDivergence_s}`);
      // The first topology publication can expose its row census one stats
      // read before the compact cell census, while the nonzero pressure field
      // and executed solve above already prove that the cell workset ran.
      expect(failures, (stats.adaptivePressureActiveRowCount ?? 0) > 0,
      `step ${step}: pressure-row topology receipt is missing or empty`);
      expect(failures, stats.adaptiveTopologyPreparedBrickCount
        !== undefined && stats.adaptiveTopologyCommittedBrickCount !== undefined
        && stats.adaptiveTopologyPreparedBrickCount
          === stats.adaptiveTopologyCommittedBrickCount,
      `step ${step}: prepared and committed topology blast radii differ`);
      const generation = stats.adaptiveTopologyShadowGeneration;
      expect(failures, generation !== undefined && (previousGeneration === undefined
        || generation >= previousGeneration),
      `step ${step}: topology generation regressed or is missing`);
      previousGeneration = generation;
      checkpoints.push({
        step,
        time_s: target_s,
        bitExact: bitExactFieldReceipt(fields),
        mass,
        relativeMassDrift: (mass - initialMass) / initialMass,
        normalizedDensityChange,
        connectivity,
        fields: { density: densitySummary, velocity: velocitySummary,
          pressure: pressureSummary, divergence: divergenceSummary },
        symmetry: { density: densitySymmetry, velocity: velocitySymmetry,
          pressure: pressureSymmetry, topology: topologySymmetry },
        pressure: {
          cells: stats.adaptivePressureCellCount,
          activeRows: stats.adaptivePressureActiveRowCount,
          iterations: stats.pressureIterations,
          iterationsExecuted: stats.pressureIterationsExecuted,
          relativeResidual,
        },
        topology: {
          generation,
          prepared: stats.adaptiveTopologyPreparedBrickCount,
          committed: stats.adaptiveTopologyCommittedBrickCount,
          deferred: stats.adaptiveTopologyDeferredBrickCount,
          mixedSeams: stats.adaptiveMixedSeamFaceCount,
        },
        authority: { frameControl, scalarResult },
      });
    }
    const maximum = (read: (checkpoint: Record<string, unknown>) => number) =>
      Math.max(...checkpoints.map(read));
    expect(failures, maximum((checkpoint) => Number(checkpoint.normalizedDensityChange)) >= 1e-5,
      "symmetric liquid did not measurably evolve");
    const scopedError = await device.popErrorScope();
    if (scopedError) validationErrors.push(scopedError.message);
    for (const message of validationErrors) failures.push(`WebGPU validation error: ${message}`);
    const report = {
      passed: failures.length === 0,
      lane: "symmetric-expansion-weakened",
      scene: scene.sceneId,
      backend,
      grid: dimensions,
      configuration: {
        brickFineResolution: TARGET_BRICK_FINE_RESOLUTION,
        presentationPageResolution: TARGET_PRESENTATION_PAGE_RESOLUTION,
      },
      steps,
      dt_s: CM12_PAPER_DT_S,
      simulatedDuration_s: steps * CM12_PAPER_DT_S,
      runMode: hasFlag("short-smoke") ? "short-smoke" : "canonical",
      initialBitExact: bitExactFieldReceipt(initialFields),
      thresholds,
      maximumDensityD4Error: maximum((checkpoint) => Number(
        (checkpoint.symmetry as Record<string, { maximumAbsoluteError: number }>).density
          .maximumAbsoluteError)),
      maximumVelocityD4Error_m_s: maximum((checkpoint) => Number(
        (checkpoint.symmetry as Record<string, { maximumAbsoluteError: number }>).velocity
          .maximumAbsoluteError)),
      maximumPressureD4Error: maximum((checkpoint) => Number(
        (checkpoint.symmetry as Record<string, { maximumAbsoluteError: number }>).pressure
          .maximumAbsoluteError)),
      maximumTopologyD4Error: maximum((checkpoint) => Number(
        (checkpoint.symmetry as Record<string, { maximumAbsoluteError: number }>).topology
          .maximumAbsoluteError)),
      maximumPressureRelativeResidual: maximum((checkpoint) => Number(
        (checkpoint.pressure as Record<string, number>).relativeResidual)),
      maximumPostProjectionDivergence_s: maximum((checkpoint) => Number(
        (checkpoint.fields as Record<string, { maximumAbsolute: number }>).divergence
          .maximumAbsolute)),
      authority: {
        rule: "FCA1 and FSM1 publish exactly one successor per encoded step",
        vexCapturePolicy: "one full QA snapshot on the first FCA1/FSM1 failure only",
        vexFirstFailure,
      },
      failures,
      checkpointSteps: checkpointSteps(steps),
      checkpoints,
    };
    console.log(JSON.stringify(report, null, 2));
    if (failures.length > 0) throw new Error(
      `weakened symmetric-expansion regression failed (${failures.length} gates)`,
    );
  } finally {
    solver?.destroy();
    if (device) {
      const compiler = gpuCompilationManagerFor(device);
      invalidateGPUCompilationManager(device, "weakened symmetry QA lane complete");
      await compiler.whenIdle();
      try { await device.queue.onSubmittedWorkDone(); } catch { /* Device fault already reported. */ }
      device.destroy();
      // Keep Dawn reachable until its final ProcessEvents callback observes
      // device retirement, matching the bounded dam lifecycle.
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
    gpu = undefined;
    await releaseWebGPUExclusiveLock();
  }
}

interface CommandReceipt {
  readonly lane: "dam-front" | "dam-front-performance" | "symmetric-expansion-weakened";
  readonly argv: readonly string[];
  readonly expectedRuntime: string;
  readonly timeoutMs: number;
}

class LaneCommandError extends Error {
  constructor(message: string, readonly report?: Record<string, unknown>) {
    super(message);
    this.name = "LaneCommandError";
  }
}

function commands(): readonly CommandReceipt[] {
  const common = ["--import", "tsx"];
  const backend = argument("backend");
  const shortSmoke = hasFlag("short-smoke");
  const pressureOraclePaired = hasFlag("pressure-oracle-paired");
  const authorityOraclePaired = hasFlag("fca-authority-oracle-paired");
  const damSteps = String(physicsLaneSteps("dam-steps"));
  const symmetrySteps = String(physicsLaneSteps("symmetry-steps"));
  const performanceReplays = argument("performance-replays") ?? "5";
  const performanceCaptureGap = argument("performance-capture-gap-ms") ?? "110";
  const physicalReference = argument("physical-reference");
  const recordPerformanceBaseline = argument("record-performance-baseline");
  const performanceBaseline = argument("performance-baseline")
    ?? DEFAULT_DAM_PERFORMANCE_BASELINE;
  return [{
    lane: "dam-front",
    argv: [NODE, ...common, SELF, "--internal-dam-front", `--dam-steps=${damSteps}`,
      ...(shortSmoke ? ["--short-smoke"] : []),
      ...(pressureOraclePaired ? ["--pressure-oracle-paired"] : []),
      ...(authorityOraclePaired ? ["--fca-authority-oracle-paired"] : []),
      ...(physicalReference ? [`--physical-reference=${physicalReference}`] : []),
      ...(backend ? [`--backend=${backend}`] : [])],
    expectedRuntime: shortSmoke
      ? "roughly 90-180 seconds on Apple M1 Max; hard timeout 300 seconds"
      : "roughly 18-36 minutes on Apple M1 Max; hard timeout 40 minutes",
    timeoutMs: shortSmoke ? 300_000 : 2_400_000,
  }, {
    lane: "symmetric-expansion-weakened",
    argv: [NODE, ...common, SELF, "--internal-symmetric-expansion-weakened",
      `--symmetry-steps=${symmetrySteps}`,
      ...(shortSmoke ? ["--short-smoke"] : []),
      ...(backend ? [`--backend=${backend}`] : [])],
    expectedRuntime: shortSmoke
      ? "roughly 30-90 seconds on Apple M1 Max; hard timeout 240 seconds"
      : "roughly 6-18 minutes on Apple M1 Max; hard timeout 20 minutes",
    timeoutMs: shortSmoke ? 240_000 : 1_200_000,
  }, {
    lane: "dam-front-performance",
    argv: [NODE, ...common, SELF, "--internal-dam-front-performance",
      `--performance-replays=${performanceReplays}`,
      `--performance-capture-gap-ms=${performanceCaptureGap}`,
      ...(recordPerformanceBaseline
        ? [`--record-performance-baseline=${recordPerformanceBaseline}`]
        : [`--performance-baseline=${performanceBaseline}`]),
      ...(backend ? [`--backend=${backend}`] : [])],
    expectedRuntime: "roughly 6-12 minutes on Apple M1 Max; hard timeout 900 seconds",
    timeoutMs: 900_000,
  }];
}

function printHelp(): void {
  console.log(`Sparse CM12 standalone GPU regression lanes

Usage:
  node --import tsx tools/run-sparse-cm12-temporal-regressions.ts [options]

Options:
  --help, -h                         Print this help and exit without acquiring WebGPU
  --emit-commands                    Print isolated child commands; run no GPU work
  --lane=all                         Run all three lanes (default)
  --lane=dam-front                   Run the 64^3 dam-front lane only
  --lane=dam-front-performance       Run repeated hardware-timestamped dam64 replays
  --lane=symmetric-expansion-weakened
                                     Run the weakened-symmetry lane only
  --dam-steps=N                      Dam steps, minimum/default 60 (2.0 seconds)
  --symmetry-steps=N                 Symmetry steps, minimum/default 60 (2.0 seconds)
  --short-smoke                      Explicit non-canonical five-step smoke mode
  --physical-reference=PATH          Require byte-identical dam physical fields
  --performance-replays=N            Independent five-step dam replays, minimum 3 (default 5)
  --performance-capture-gap-ms=N     Delay between timestamp captures (default 110)
  --performance-baseline=PATH        Compare performance and exact physics receipts
  --record-performance-baseline=PATH Record a clean baseline instead of comparing
  --backend=NAME                     Dawn backend (default metal)
  --out=PATH                         Write the combined JSON receipt

These are executable Dawn lanes, not node:test unit tests. Physics thresholds
are fixed in this tool and failures never select a fallback path.`);
}

async function runCommand(receipt: CommandReceipt): Promise<Record<string, unknown>> {
  const [executable, ...argv] = receipt.argv;
  const timeoutMs = receipt.timeoutMs;
  const child = spawn(executable!, argv, {
    cwd: ROOT,
    env: {
      ...process.env,
      WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "", stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => { stdout += chunk; });
  child.stderr.on("data", (chunk: string) => { stderr += chunk; });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGKILL");
  }, timeoutMs);
  let result: { code: number | null; signal: NodeJS.Signals | null };
  try {
    result = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => resolve({ code, signal }));
    });
  } finally {
    clearTimeout(timer);
    // A SIGKILL cannot run the child's async finally. Remove only a dead lease
    // stamped by this exact child; a new or unrelated GPU owner is untouchable.
    const holder = await readWebGPUExclusiveLockHolder();
    if (child.pid !== undefined && holder?.owner?.pid === child.pid && !holder.alive) {
      await releaseWebGPUExclusiveLock();
    }
  }
  if (timedOut) throw new Error(`${receipt.lane} exceeded its ${timeoutMs} ms hard timeout`);
  const firstBrace = stdout.indexOf("{");
  if (firstBrace < 0) {
    throw new Error(`${receipt.lane} exited with code ${result.code ?? "none"}`
      + `${result.signal ? ` (${result.signal})` : ""} and emitted no JSON receipt\n`
      + `${stderr.trim()}\n${stdout.trim()}`);
  }
  let report: Record<string, unknown>;
  try {
    report = JSON.parse(stdout.slice(firstBrace)) as Record<string, unknown>;
  } catch (error) {
    throw new Error(`${receipt.lane} emitted malformed JSON\n${stdout.trim()}`, { cause: error });
  }
  if (result.code !== 0) {
    throw new LaneCommandError(`${receipt.lane} exited with code ${result.code ?? "none"}`
      + `${result.signal ? ` (${result.signal})` : ""}\n${stderr.trim()}`, report);
  }
  if (report.passed !== true) {
    throw new LaneCommandError(`${receipt.lane} did not publish passed=true`, report);
  }
  if (!Array.isArray(report.grid) || report.grid.length !== 3) {
    throw new Error(`${receipt.lane} omitted its three-dimensional grid receipt`);
  }
  if (!Number.isSafeInteger(report.steps) || Number(report.steps) < 1) {
    throw new Error(`${receipt.lane} omitted its executed-step receipt`);
  }
  const simulatedDuration_s = Number(report.simulatedDuration_s);
  const expectedRunMode = receipt.argv.includes("--short-smoke") ? "short-smoke" : "canonical";
  if (receipt.lane !== "dam-front-performance" && report.runMode !== expectedRunMode) {
    throw new Error(`${receipt.lane} published runMode=${String(report.runMode)}; expected ${
      expectedRunMode}`);
  }
  if (receipt.lane !== "dam-front-performance"
    && (!(simulatedDuration_s > 0) || !Number.isFinite(simulatedDuration_s))) {
    throw new Error(`${receipt.lane} omitted its simulated-duration receipt`);
  }
  if (receipt.lane !== "dam-front-performance" && report.runMode === "canonical"
    && simulatedDuration_s < MINIMUM_CANONICAL_DURATION_S) {
    throw new Error(`${receipt.lane} canonical receipt covers only ${simulatedDuration_s} s`);
  }
  const checkpointSchedule = report.checkpointSteps;
  if (receipt.lane !== "dam-front-performance"
    && (!Array.isArray(checkpointSchedule) || checkpointSchedule.length < 2)) {
    throw new Error(`${receipt.lane} omitted its trajectory checkpoint schedule`);
  }
  if (receipt.lane === "dam-front") {
    if (!Array.isArray(report.trajectory) || report.trajectory.length !== report.steps) {
      throw new Error("dam-front trajectory is absent or incomplete");
    }
    if (!Array.isArray(report.checkpoints)
      || !Array.isArray(checkpointSchedule)
      || report.checkpoints.length !== checkpointSchedule.length) {
      throw new Error("dam-front checkpoint receipts are absent or incomplete");
    }
    if (typeof report.maximumMassDrift !== "number" || !Number.isFinite(report.maximumMassDrift)) {
      throw new Error("dam-front mass receipt is absent or non-finite");
    }
  } else if (receipt.lane === "symmetric-expansion-weakened") {
    if (!Array.isArray(report.checkpoints) || report.checkpoints.length !== report.steps) {
      throw new Error("symmetric-expansion checkpoint trajectory is absent or incomplete");
    }
    for (const key of ["maximumDensityD4Error", "maximumVelocityD4Error_m_s",
      "maximumPressureD4Error", "maximumTopologyD4Error",
      "maximumPressureRelativeResidual", "maximumPostProjectionDivergence_s"]) {
      if (typeof report[key] !== "number" || !Number.isFinite(report[key])) {
        throw new Error(`symmetric-expansion receipt ${key} is absent or non-finite`);
      }
    }
  } else {
    if (report.kind !== "sparse-cm12-dam-front64-performance" || report.version !== 1) {
      throw new Error("dam-front performance receipt has an unsupported kind or version");
    }
    const nonPressure = report.nonPressure as Record<string, unknown> | undefined;
    if (typeof nonPressure?.p95_ms !== "number" || !Number.isFinite(nonPressure.p95_ms)) {
      throw new Error("dam-front performance receipt omitted combined non-pressure p95");
    }
    const replayEquality = report.physicsReplayEquality;
    if (!Array.isArray(replayEquality) || replayEquality.some((equal) => equal !== true)) {
      throw new Error("dam-front performance receipt did not preserve physics across replays");
    }
  }
  return report;
}

if (hasFlag("help") || process.argv.includes("-h")) {
  printHelp();
} else if (hasFlag("internal-dam-front")) {
  await runDamFrontLane();
} else if (hasFlag("internal-dam-front-performance")) {
  await runDamFrontPerformanceLane();
} else if (hasFlag("internal-symmetric-expansion-weakened")) {
  await runWeakenedSymmetryLane();
} else {
  const requested = argument("lane") ?? "all";
  if (!["all", "dam-front", "dam-front-performance",
    "symmetric-expansion-weakened"].includes(requested)) {
    throw new RangeError("lane must be all, dam-front, dam-front-performance, "
      + "or symmetric-expansion-weakened");
  }
  const selected = commands().filter((receipt) => requested === "all" || receipt.lane === requested);
  if (hasFlag("emit-commands")) {
    console.log(JSON.stringify({ phase: "sparse-cm12-temporal-regression-commands",
      lanes: selected.map((receipt) => ({ ...receipt, command: receipt.argv.join(" ") })) }, null, 2));
  } else {
    const startedAt = new Date().toISOString();
    const laneReports: Record<string, unknown> = {};
    const outputPath = argument("out");
    for (const receipt of selected) {
      try {
        laneReports[receipt.lane] = await runCommand(receipt);
      } catch (error) {
        if (error instanceof LaneCommandError && error.report) {
          laneReports[receipt.lane] = error.report;
        }
        const failed = {
          passed: false,
          phase: "sparse-cm12-temporal-regressions",
          startedAt,
          completedAt: new Date().toISOString(),
          failedLane: receipt.lane,
          failure: error instanceof Error ? error.message : String(error),
          lanes: laneReports,
        };
        if (outputPath) await writeFile(outputPath, `${JSON.stringify(failed, null, 2)}\n`, "utf8");
        throw error;
      }
    }
    const combined = {
      passed: true,
      phase: "sparse-cm12-temporal-regressions",
      startedAt,
      completedAt: new Date().toISOString(),
      lanes: laneReports,
    };
    if (outputPath) await writeFile(outputPath, `${JSON.stringify(combined, null, 2)}\n`, "utf8");
    console.log(JSON.stringify(combined, null, 2));
  }
}
