/**
 * Native-Dawn acceptance gate for adaptive-mass symmetric expansion.
 *
 * This runner intentionally does not pass absent publications as zero fields.
 * The adaptive solver must publish density, level set, ownership, collocated
 * velocity, pressure, and post-projection divergence before this gate can pass.
 *
 * Fast development lane (20 steps):
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-adaptive-mass-symmetric-expansion-dawn.ts
 *
 * Canonical scene lane (250 steps):
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-adaptive-mass-symmetric-expansion-dawn.ts --steps=250
 */
import { pathToFileURL } from "node:url";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  type AdaptiveMassStepTelemetry,
  WebGPUAdaptiveMassSolver,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

type Dimensions = readonly [number, number, number];
type ScalarFieldName = "density" | "levelSet" | "topology" | "pressure" | "divergence";

interface FieldSummary {
  readonly count: number;
  readonly nonFiniteCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly maximumAbsolute: number;
  readonly sum: number;
}

interface SymmetrySummary {
  readonly comparedValues: number;
  readonly nonFiniteCount: number;
  readonly maximumAbsoluteError: number;
  readonly worst?: {
    readonly transform: "reflect-x" | "reflect-z" | "swap-xz";
    readonly source: readonly [number, number, number];
    readonly target: readonly [number, number, number];
    readonly component?: number;
    readonly expected: number;
    readonly actual: number;
  };
}

interface Checkpoint {
  readonly step: number;
  readonly time_s: number;
  readonly mass_cells: number;
  readonly relativeMassDrift: number;
  /** Finest-lattice L1 change divided by the authored initial mass. */
  readonly normalizedL1DensityChange: number;
  readonly density: FieldSummary;
  readonly levelSet: FieldSummary;
  readonly levelSetOwnerPhaseMismatchCount: number;
  readonly velocity?: FieldSummary;
  readonly pressure?: FieldSummary;
  readonly divergence?: FieldSummary;
  readonly symmetry: Readonly<Partial<Record<ScalarFieldName | "velocity", SymmetrySummary>>>;
  readonly dominantBodyMassFraction: number;
  readonly pressureRelativeResidual?: number;
  readonly maximumPostProjectionDivergence_s?: number;
  readonly statsMaximumPostProjectionDivergence_s?: number;
  readonly maximumAbsoluteVerticalVelocity_m_s?: number;
  readonly maximumCfl?: number;
  readonly kineticEnergyBeforeFineUnits?: number;
  readonly kineticEnergyAfterFineUnits?: number;
  readonly projectionKineticEnergyBeforeFineUnits?: number;
  readonly projectionKineticEnergyAfterFineUnits?: number;
  readonly inactiveFaceCount?: number;
  readonly maximumInactiveFaceSpeedBefore_m_s?: number;
  readonly maximumInactiveFaceSpeedAfter_m_s?: number;
  readonly maximumMixedSeamDivergence_s?: number;
  readonly pressureIterations?: number;
  readonly adaptiveMixedSeamFaceCount?: number;
  readonly adaptiveResolutionTopologyEpoch?: boolean;
  readonly adaptiveResolutionPromotedBrickCount?: number;
  readonly adaptiveResolutionDemotedBrickCount?: number;
  readonly adaptiveResolutionDeferredPromotionCount?: number;
  readonly adaptiveFineBrickCount?: number;
  readonly adaptiveCoarseBrickCount?: number;
  readonly residentOwnerScales: readonly number[];
  readonly encodedSteps?: number;
  readonly submittedTime_s?: number;
  readonly simulatedTime_s?: number;
  readonly completedTime_s?: number;
  readonly hostFluidAuthority?: string;
  readonly hostSimulationSizedWorkItems?: number;
}

const DENSITY_SYMMETRY_LIMIT = 1e-3;
const VELOCITY_SYMMETRY_LIMIT_M_S = 1e-4;
const PRESSURE_SYMMETRY_LIMIT = 0.25;
const PRESSURE_RELATIVE_RESIDUAL_LIMIT = 1e-8;
const POST_PROJECTION_DIVERGENCE_LIMIT_S = 1e-5;
const MASS_RELATIVE_ERROR_LIMIT = 1e-3;
const MINIMUM_FINAL_NORMALIZED_L1_DENSITY_CHANGE = 1e-5;
const DIVERGENCE_PUBLICATION_ABSOLUTE_AGREEMENT_S = 1e-8;
const DIVERGENCE_PUBLICATION_RELATIVE_AGREEMENT = 1e-5;
const MINIMUM_DOMINANT_BODY_MASS_FRACTION = 0.98;
const MAXIMUM_DENSITY = 2.5;

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const positiveInteger = (name: string, fallback: number): number => {
  const environment = `FLUID_ADAPTIVE_SYMMETRY_${name.toUpperCase().replaceAll("-", "_")}`;
  const value = Number(argument(name) ?? process.env[environment] ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer; received ${value}`);
  }
  return value;
};

const positiveNumber = (name: string, fallback: number): number => {
  const environment = `FLUID_ADAPTIVE_SYMMETRY_${name.toUpperCase().replaceAll("-", "_")}`;
  const value = Number(argument(name) ?? process.env[environment] ?? fallback);
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and positive; received ${value}`);
  }
  return value;
};

function summarize(field: ArrayLike<number>): FieldSummary {
  let nonFiniteCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let maximumAbsolute = 0;
  let sum = 0;
  let correction = 0;
  for (let index = 0; index < field.length; index += 1) {
    const value = Number(field[index]);
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1;
      continue;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return {
    count: field.length,
    nonFiniteCount,
    minimum: field.length === nonFiniteCount ? Number.NaN : minimum,
    maximum: field.length === nonFiniteCount ? Number.NaN : maximum,
    maximumAbsolute,
    sum,
  };
}

function scalarD4(field: ArrayLike<number>, dimensions: Dimensions): SymmetrySummary {
  const [nx, ny, nz] = dimensions;
  const transforms = ["reflect-x", "reflect-z", "swap-xz"] as const;
  let comparedValues = 0;
  let nonFiniteCount = 0;
  let maximumAbsoluteError = 0;
  let worst: SymmetrySummary["worst"];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) for (const transform of transforms) {
      const target: [number, number, number] = transform === "reflect-x"
        ? [nx - 1 - x, y, z]
        : transform === "reflect-z" ? [x, y, nz - 1 - z] : [z, y, x];
      const sourceIndex = x + nx * (y + ny * z);
      const targetIndex = target[0] + nx * (target[1] + ny * target[2]);
      const expected = Number(field[sourceIndex]);
      const actual = Number(field[targetIndex]);
      comparedValues += 1;
      if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
        nonFiniteCount += 1;
        continue;
      }
      const error = Math.abs(actual - expected);
      if (error > maximumAbsoluteError) {
        maximumAbsoluteError = error;
        worst = { transform, source: [x, y, z], target, expected, actual };
      }
    }
  }
  return { comparedValues, nonFiniteCount, maximumAbsoluteError, worst };
}

function velocityD4(field: ArrayLike<number>, dimensions: Dimensions): SymmetrySummary {
  const [nx, ny, nz] = dimensions;
  const transforms = ["reflect-x", "reflect-z", "swap-xz"] as const;
  let comparedValues = 0;
  let nonFiniteCount = 0;
  let maximumAbsoluteError = 0;
  let worst: SymmetrySummary["worst"];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) for (const transform of transforms) {
      const target: [number, number, number] = transform === "reflect-x"
        ? [nx - 1 - x, y, z]
        : transform === "reflect-z" ? [x, y, nz - 1 - z] : [z, y, x];
      const sourceCell = x + nx * (y + ny * z);
      const targetCell = target[0] + nx * (target[1] + ny * target[2]);
      for (let component = 0; component < 3; component += 1) {
        const targetComponent = transform === "swap-xz"
          ? component === 0 ? 2 : component === 2 ? 0 : 1
          : component;
        const source = Number(field[3 * sourceCell + component]);
        const expected = transform === "reflect-x" && component === 0
          ? -source : transform === "reflect-z" && component === 2 ? -source : source;
        const actual = Number(field[3 * targetCell + targetComponent]);
        comparedValues += 1;
        if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
          nonFiniteCount += 1;
          continue;
        }
        const error = Math.abs(actual - expected);
        if (error > maximumAbsoluteError) {
          maximumAbsoluteError = error;
          worst = { transform, source: [x, y, z], target, component, expected, actual };
        }
      }
    }
  }
  return { comparedValues, nonFiniteCount, maximumAbsoluteError, worst };
}

async function readTexture(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
  channels: 1 | 2 | 4,
): Promise<Float32Array> {
  const [nx, ny, nz] = dimensions;
  const rowBytes = nx * channels * 4;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const size = bytesPerRow * ny * nz;
  const readback = device.createBuffer({
    label: `Adaptive symmetry ${channels}-channel texture readback`,
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "Adaptive symmetry texture readback" });
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readback, bytesPerRow, rowsPerImage: ny },
      [nx, ny, nz],
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(readback.getMappedRange());
    const output = new Float32Array(nx * ny * nz * channels);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      const source = stride * (y + ny * z);
      const target = channels * nx * (y + ny * z);
      output.set(mapped.subarray(source, source + channels * nx), target);
    }
    return output;
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function readOwnerWords(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: Dimensions,
): Promise<Uint32Array> {
  const [nx, ny, nz] = dimensions;
  const rowBytes = nx * 2 * 4;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const readback = device.createBuffer({
    label: "Adaptive symmetry ownership readback",
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "Adaptive symmetry ownership readback" });
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readback, bytesPerRow, rowsPerImage: ny },
      [nx, ny, nz],
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint32Array(readback.getMappedRange());
    const output = new Uint32Array(nx * ny * nz * 2);
    const stride = bytesPerRow / 4;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      const source = stride * (y + ny * z);
      const target = 2 * nx * (y + ny * z);
      output.set(mapped.subarray(source, source + 2 * nx), target);
    }
    return output;
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

/** Reduce packed absolute owner keys to the resolution/occupancy scalar that
 * is invariant under reflection. Also reject keys that do not own the fine
 * cell at which they were published. */
function topologyFromOwners(
  owners: Uint32Array,
  dimensions: Dimensions,
): { readonly field: Float32Array; readonly invalidOwnershipCount: number } {
  const [nx, ny, nz] = dimensions;
  const field = new Float32Array(nx * ny * nz);
  let invalidOwnershipCount = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const index = x + nx * (y + ny * z);
      const first = owners[2 * index] >>> 0;
      const second = owners[2 * index + 1] >>> 0;
      const wide = (second & 0x8000_0000) !== 0;
      const lowerX = first & (wide ? 0x7ff : 0x3ff);
      const lowerZ = (first >>> (wide ? 11 : 10)) & (wide ? 0x7ff : 0x3ff);
      const scale = wide ? 2 ** ((first >>> 22) & 0xf) : (first >>> 20) & 0x3ff;
      const lowerY = second & (wide ? 0x7ff : 0x3ff);
      const upperY = wide ? lowerY + scale : (second >>> 10) & 0x3ff;
      field[index] = scale;
      if (scale < 1 || upperY !== lowerY + scale
        || x < lowerX || x >= lowerX + scale
        || y < lowerY || y >= upperY
        || z < lowerZ || z >= lowerZ + scale) invalidOwnershipCount += 1;
    }
  }
  return { field, invalidOwnershipCount };
}

/**
 * Check the render-only phi reconstruction against the conservative density
 * authority at the resolution of each published owner. Fine owners have one
 * presentation sample. Coarse owners may reconstruct a smooth zero crossing
 * among their 2x2x2 children, while preserving the authoritative leaf mean.
 */
function levelSetOwnerPhaseMismatchCount(
  density: ArrayLike<number>,
  levelSet: ArrayLike<number>,
  owners: Uint32Array,
): number {
  let mismatches = 0;
  const coarseOwners = new Map<string, { density: number; phi: number; samples: number }>();
  for (let cell = 0; cell < density.length; cell += 1) {
    const first = owners[2 * cell] >>> 0;
    const second = owners[2 * cell + 1] >>> 0;
    const scale = (second & 0x8000_0000) !== 0
      ? 2 ** ((first >>> 22) & 0xf)
      : (first >>> 20) & 0x3ff;
    if (scale === 1) {
      if ((Number(density[cell]) > 0.5) !== (Number(levelSet[cell]) < 0)) mismatches += 1;
      continue;
    }
    if (scale !== 2) continue;
    const key = `${first}:${second}`;
    const aggregate = coarseOwners.get(key) ?? { density: 0, phi: 0, samples: 0 };
    aggregate.density += Number(density[cell]);
    aggregate.phi += Number(levelSet[cell]);
    aggregate.samples += 1;
    coarseOwners.set(key, aggregate);
  }
  for (const owner of coarseOwners.values()) {
    if ((owner.density / owner.samples > 0.5) !== (owner.phi / owner.samples < 0)) {
      mismatches += 1;
    }
  }
  return mismatches;
}

function dominantBodyMassFraction(density: ArrayLike<number>, dimensions: Dimensions): number {
  const [nx, ny, nz] = dimensions;
  const count = nx * ny * nz;
  const visited = new Uint8Array(count);
  let totalMass = 0;
  for (let index = 0; index < count; index += 1) totalMass += Math.max(0, Number(density[index]));
  if (!(totalMass > 0)) return 0;
  let dominantMass = 0;
  const queue = new Uint32Array(count);
  for (let seed = 0; seed < count; seed += 1) {
    if (visited[seed] !== 0 || !(Number(density[seed]) > 1e-5)) continue;
    visited[seed] = 1;
    let head = 0, tail = 0, componentMass = 0;
    queue[tail++] = seed;
    while (head < tail) {
      const index = queue[head++];
      componentMass += Math.max(0, Number(density[index]));
      const x = index % nx;
      const y = Math.floor(index / nx) % ny;
      const z = Math.floor(index / (nx * ny));
      const visit = (next: number) => {
        if (visited[next] === 0 && Number(density[next]) > 1e-5) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      };
      if (x > 0) visit(index - 1);
      if (x + 1 < nx) visit(index + 1);
      if (y > 0) visit(index - nx);
      if (y + 1 < ny) visit(index + nx);
      if (z > 0) visit(index - nx * ny);
      if (z + 1 < nz) visit(index + nx * ny);
    }
    dominantMass = Math.max(dominantMass, componentMass);
  }
  return dominantMass / totalMass;
}

function expect(
  failures: string[],
  condition: boolean,
  message: string,
): void {
  if (!condition) failures.push(message);
}

function requirePublications(solver: GPUSolverInstance): string[] {
  const missing: string[] = [];
  if (!solver.surfaceFieldTexture) missing.push("surfaceFieldTexture (level set)");
  if (!solver.gridCellTexture) missing.push("gridCellTexture (adaptive ownership)");
  if (!solver.velocityTexture) missing.push("velocityTexture (collocated rgba32float velocity)");
  if (!solver.gridPressureTexture) missing.push("gridPressureTexture (r32float pressure)");
  if (!solver.gridDivergenceTexture) missing.push("gridDivergenceTexture (r32float post-projection divergence)");
  return missing;
}

const steps = positiveInteger("steps", 20);
const horizontalGrid = positiveInteger("grid", 32);
if (horizontalGrid % 32 !== 0) {
  throw new RangeError("grid must be a multiple of 32 so the symmetric brick body scales exactly");
}
const resolutionModeArgument = argument("resolution-mode") ?? "adaptive";
if (resolutionModeArgument !== "adaptive" && resolutionModeArgument !== "all-fine") {
  throw new RangeError("resolution-mode must be adaptive or all-fine");
}
const resolutionMode = resolutionModeArgument as "adaptive" | "all-fine";
const postProjectionDivergenceLimit_s = resolutionMode === "adaptive"
  ? POST_PROJECTION_DIVERGENCE_LIMIT_S
  : Math.max(POST_PROJECTION_DIVERGENCE_LIMIT_S,
    5e-5 * (horizontalGrid / 64) ** 2);
const backend = argument("backend") ?? process.env.WEBGPU_BACKEND
  ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? `${process.cwd()}/node_modules/webgpu/index.js`;

await acquireWebGPUExclusiveLock(
  "dawn-acceptance",
  "tools/run-adaptive-mass-symmetric-expansion-dawn.ts",
);
try {
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error(`No Dawn WebGPU adapter is available for backend ${backend}`);
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();
    validationErrors.push(event.error.message);
  });
  device.pushErrorScope("validation");

  const scene = createSymmetricExpansionScene();
  scene.voxelDomain.finestCellSize_m = scene.container.width_m / horizontalGrid;
  const verticalGrid = horizontalGrid / 2;
  const brickSize = scene.voxelDomain.brickSize_cells;
  const brickGrid = [horizontalGrid / brickSize, verticalGrid / brickSize,
    horizontalGrid / brickSize] as const;
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
  const dt_s = positiveNumber(
    "dt",
    scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s,
  );
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
  const expectedInitialMass_cells = (scene.fluid.initialBrickSeeds_m?.length ?? 0)
    * scene.voxelDomain.brickSize_cells ** 3;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const failures: string[] = [];
  const checkpoints: Checkpoint[] = [];
  let initialDensity: Float32Array | undefined;
  try {
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device,
      scene,
      "balanced",
      undefined,
      {
        resolutionMode,
        seamAxis: "x",
        fineSide: "negative",
        fineTileResolution: 8,
        coarseTileResolution: 4,
      },
      () => {},
    );
    const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
    expect(failures, dimensions[0] === horizontalGrid
      && dimensions[1] === verticalGrid && dimensions[2] === horizontalGrid,
    `grid must be exactly ${horizontalGrid}x${verticalGrid}x${horizontalGrid}; observed ${dimensions.join("x")}`);
    const missingPublications = requirePublications(solver);
    for (const publication of missingPublications) failures.push(`missing required publication: ${publication}`);

    const capture = async (step: number): Promise<Checkpoint> => {
      await device.queue.onSubmittedWorkDone();
      const stats = await solver!.readStats();
      const adaptiveStats = stats as typeof stats & AdaptiveMassStepTelemetry;
      const density = await readTexture(device, solver!.volumeTexture, dimensions, 1);
      if (step === 0 && initialDensity === undefined) initialDensity = density.slice();
      const levelSet = solver!.surfaceFieldTexture
        ? await readTexture(device, solver!.surfaceFieldTexture, dimensions, 1) : undefined;
      const owners = solver!.gridCellTexture
        ? await readOwnerWords(device, solver!.gridCellTexture, dimensions) : undefined;
      const velocityRgba = solver!.velocityTexture
        ? await readTexture(device, solver!.velocityTexture, dimensions, 4) : undefined;
      const pressure = solver!.gridPressureTexture
        ? await readTexture(device, solver!.gridPressureTexture, dimensions, 1) : undefined;
      const divergence = solver!.gridDivergenceTexture
        ? await readTexture(device, solver!.gridDivergenceTexture, dimensions, 1) : undefined;
      const velocity = velocityRgba && Float32Array.from(
        { length: dimensions[0] * dimensions[1] * dimensions[2] * 3 },
        (_, index) => velocityRgba[4 * Math.floor(index / 3) + index % 3],
      );
      const topology = owners ? topologyFromOwners(owners, dimensions) : undefined;
      const residentOwnerScales = topology
        ? [...new Set(topology.field)].filter((scale) => scale === 1 || scale === 2)
          .sort((left, right) => left - right)
        : [];
      if (topology) expect(failures, topology.invalidOwnershipCount === 0,
        `step ${step}: ${topology.invalidOwnershipCount} ownership keys do not contain their published cells`);
      const mass_cells = summarize(density).sum;
      let densityL1Change_cells = 0;
      if (initialDensity) for (let cell = 0; cell < density.length; cell += 1) {
        densityL1Change_cells += Math.abs(density[cell] - initialDensity[cell]);
      }
      const pressureRelativeResidual = stats.pressureRelativeResidual ?? stats.pressureResidual;
      const maximumPostProjectionDivergence_s = divergence
        ? summarize(divergence).maximumAbsolute : undefined;
      const levelSetOwnerPhaseMismatches = levelSet && owners
        ? levelSetOwnerPhaseMismatchCount(density, levelSet, owners) : 0;
      let maximumAbsoluteVerticalVelocity_m_s: number | undefined;
      if (velocity) {
        maximumAbsoluteVerticalVelocity_m_s = 0;
        for (let cell = 0; cell < velocity.length / 3; cell += 1) {
          maximumAbsoluteVerticalVelocity_m_s = Math.max(
            maximumAbsoluteVerticalVelocity_m_s,
            Math.abs(velocity[3 * cell + 1]),
          );
        }
      }
      return {
        step,
        time_s: step * dt_s,
        mass_cells,
        relativeMassDrift: (mass_cells - expectedInitialMass_cells)
          / Math.max(1, expectedInitialMass_cells),
        normalizedL1DensityChange: densityL1Change_cells
          / Math.max(1, expectedInitialMass_cells),
        density: summarize(density),
        levelSet: summarize(levelSet ?? []),
        levelSetOwnerPhaseMismatchCount: levelSetOwnerPhaseMismatches,
        velocity: velocity ? summarize(velocity) : undefined,
        pressure: pressure ? summarize(pressure) : undefined,
        divergence: divergence ? summarize(divergence) : undefined,
        symmetry: {
          density: scalarD4(density, dimensions),
          topology: topology ? scalarD4(topology.field, dimensions) : undefined,
          velocity: velocity ? velocityD4(velocity, dimensions) : undefined,
          pressure: pressure ? scalarD4(pressure, dimensions) : undefined,
          divergence: divergence ? scalarD4(divergence, dimensions) : undefined,
        },
        dominantBodyMassFraction: dominantBodyMassFraction(density, dimensions),
        pressureRelativeResidual,
        maximumPostProjectionDivergence_s,
        statsMaximumPostProjectionDivergence_s: stats.maxDivergenceAfter_s,
        maximumAbsoluteVerticalVelocity_m_s,
        maximumCfl: stats.maxComponentCfl,
        kineticEnergyBeforeFineUnits:
          adaptiveStats.adaptiveKineticEnergyBeforeFineUnits,
        kineticEnergyAfterFineUnits:
          adaptiveStats.adaptiveKineticEnergyAfterFineUnits,
        projectionKineticEnergyBeforeFineUnits:
          adaptiveStats.adaptiveProjectionKineticEnergyBeforeFineUnits,
        projectionKineticEnergyAfterFineUnits:
          adaptiveStats.adaptiveProjectionKineticEnergyAfterFineUnits,
        inactiveFaceCount: adaptiveStats.adaptiveInactiveFaceCount,
        maximumInactiveFaceSpeedBefore_m_s:
          adaptiveStats.adaptiveMaximumInactiveFaceSpeedBefore_m_s,
        maximumInactiveFaceSpeedAfter_m_s:
          adaptiveStats.adaptiveMaximumInactiveFaceSpeedAfter_m_s,
        maximumMixedSeamDivergence_s:
          adaptiveStats.adaptiveMaximumMixedSeamDivergence_s,
        pressureIterations: stats.pressureIterations,
        adaptiveMixedSeamFaceCount: (stats as typeof stats & {
          readonly adaptiveMixedSeamFaceCount?: number;
        }).adaptiveMixedSeamFaceCount,
        adaptiveResolutionTopologyEpoch: stats.adaptiveResolutionTopologyEpoch,
        adaptiveResolutionPromotedBrickCount:
          stats.adaptiveResolutionPromotedBrickCount,
        adaptiveResolutionDemotedBrickCount:
          stats.adaptiveResolutionDemotedBrickCount,
        adaptiveResolutionDeferredPromotionCount:
          stats.adaptiveResolutionDeferredPromotionCount,
        adaptiveFineBrickCount: stats.adaptiveFineBrickCount,
        adaptiveCoarseBrickCount: stats.adaptiveCoarseBrickCount,
        residentOwnerScales,
        encodedSteps: stats.encodedSteps,
        submittedTime_s: stats.submittedTime_s,
        simulatedTime_s: stats.simulatedTime_s,
        completedTime_s: stats.completedTime_s,
        hostFluidAuthority: stats.hostFluidAuthority,
        hostSimulationSizedWorkItems: stats.hostSimulationSizedWorkItems,
      };
    };

    checkpoints.push(await capture(0));
    const initialRelativeMassError = Math.abs(
      (checkpoints[0]!.mass_cells - expectedInitialMass_cells) / Math.max(1, expectedInitialMass_cells),
    );
    expect(failures, expectedInitialMass_cells > 0,
      "symmetric-expansion authored mass could not be derived from its brick seeds");
    expect(failures, initialRelativeMassError <= MASS_RELATIVE_ERROR_LIMIT,
      `initial mass relative error ${initialRelativeMassError} exceeds ${MASS_RELATIVE_ERROR_LIMIT}`);
    const initial = checkpoints[0]!;
    expect(failures, initial.density.nonFiniteCount === 0
      && initial.density.minimum >= -1e-6 && initial.density.maximum <= MAXIMUM_DENSITY,
    `initial density is non-finite or outside [-1e-6, ${MAXIMUM_DENSITY}]`);
    expect(failures, initial.levelSet.count > 0 && initial.levelSet.nonFiniteCount === 0,
      "initial level-set publication is absent or non-finite");
    expect(failures, initial.levelSetOwnerPhaseMismatchCount === 0,
      `initial level-set phase disagrees with density in ${initial.levelSetOwnerPhaseMismatchCount} owners`);
    expect(failures, initial.symmetry.density?.maximumAbsoluteError === 0,
      `initial density is not exactly D4 symmetric (error ${initial.symmetry.density?.maximumAbsoluteError ?? "missing"})`);
    expect(failures, initial.symmetry.topology?.maximumAbsoluteError === 0,
      `initial topology is not exactly D4 symmetric (error ${initial.symmetry.topology?.maximumAbsoluteError ?? "missing"})`);
    expect(failures, initial.symmetry.velocity?.maximumAbsoluteError === 0,
      `initial velocity is not exactly D4 symmetric (error ${initial.symmetry.velocity?.maximumAbsoluteError ?? "missing"})`);
    expect(failures, initial.symmetry.pressure?.maximumAbsoluteError === 0,
      `initial pressure is not exactly D4 symmetric (error ${initial.symmetry.pressure?.maximumAbsoluteError ?? "missing"})`);
    expect(failures, initial.dominantBodyMassFraction >= MINIMUM_DOMINANT_BODY_MASS_FRACTION,
      `initial dominant connected body fraction ${initial.dominantBodyMassFraction} is below ${MINIMUM_DOMINANT_BODY_MASS_FRACTION}`);
    expect(failures, initial.encodedSteps === 0
      && initial.submittedTime_s === 0 && initial.simulatedTime_s === 0
      && initial.completedTime_s === 0,
    "initial exact step/time publication is not zero");

    for (let step = 1; step <= steps; step += 1) {
      const target_s = step * dt_s;
      const advanced = solver.advanceTo(target_s, []);
      expect(failures, advanced, `step ${step}: advanceTo(${target_s}) did not encode exactly one step`);
      const checkpoint = await capture(step);
      checkpoints.push(checkpoint);
      expect(failures, Math.abs(checkpoint.relativeMassDrift) <= MASS_RELATIVE_ERROR_LIMIT,
        `step ${step}: mass drift ${checkpoint.relativeMassDrift} exceeds ${MASS_RELATIVE_ERROR_LIMIT}`);
      expect(failures, checkpoint.density.nonFiniteCount === 0
        && checkpoint.density.minimum >= -1e-6
        && checkpoint.density.maximum <= MAXIMUM_DENSITY,
      `step ${step}: density is non-finite or outside [-1e-6, ${MAXIMUM_DENSITY}]`);
      const domainDiagonal_m = Math.hypot(
        scene.container.width_m, scene.container.height_m, scene.container.depth_m,
      );
      expect(failures, checkpoint.levelSet.count > 0
        && checkpoint.levelSet.nonFiniteCount === 0
        && checkpoint.levelSet.maximumAbsolute <= 2 * domainDiagonal_m,
      `step ${step}: level set is absent, non-finite, or exceeds twice the domain diagonal`);
      expect(failures, checkpoint.levelSetOwnerPhaseMismatchCount === 0,
        `step ${step}: level-set phase disagrees with density in ${checkpoint.levelSetOwnerPhaseMismatchCount} owners`);
      expect(failures, checkpoint.symmetry.density !== undefined
        && checkpoint.symmetry.density.nonFiniteCount === 0
        && checkpoint.symmetry.density.maximumAbsoluteError <= DENSITY_SYMMETRY_LIMIT,
      `step ${step}: density D4 error ${checkpoint.symmetry.density?.maximumAbsoluteError ?? "missing"} exceeds ${DENSITY_SYMMETRY_LIMIT}`);
      expect(failures, checkpoint.symmetry.topology !== undefined
        && checkpoint.symmetry.topology.nonFiniteCount === 0
        && checkpoint.symmetry.topology.maximumAbsoluteError === 0,
      `step ${step}: topology is missing or not exactly D4 symmetric`);
      expect(failures, checkpoint.velocity !== undefined
        && checkpoint.velocity.nonFiniteCount === 0
        && checkpoint.velocity.maximumAbsolute <= 4 * solver.info.cellSize_m / dt_s,
      `step ${step}: velocity is missing, non-finite, or exceeds the 4-cell CFL bound`);
      expect(failures, checkpoint.symmetry.velocity !== undefined
        && checkpoint.symmetry.velocity.nonFiniteCount === 0
        && checkpoint.symmetry.velocity.maximumAbsoluteError <= VELOCITY_SYMMETRY_LIMIT_M_S,
      `step ${step}: velocity D4 error ${checkpoint.symmetry.velocity?.maximumAbsoluteError ?? "missing"} exceeds ${VELOCITY_SYMMETRY_LIMIT_M_S}`);
      expect(failures, checkpoint.maximumAbsoluteVerticalVelocity_m_s !== undefined
        && checkpoint.maximumAbsoluteVerticalVelocity_m_s > 1e-6,
      `step ${step}: force-evolved vertical velocity is missing or identically zero`);
      expect(failures, checkpoint.pressure !== undefined
        && checkpoint.pressure.nonFiniteCount === 0
        && checkpoint.pressure.maximumAbsolute <= 1e6,
      `step ${step}: pressure is missing, non-finite, or exceeds 1 MPa`);
      expect(failures, checkpoint.symmetry.pressure !== undefined
        && checkpoint.symmetry.pressure.nonFiniteCount === 0
        && checkpoint.symmetry.pressure.maximumAbsoluteError <= PRESSURE_SYMMETRY_LIMIT,
      `step ${step}: pressure D4 error ${checkpoint.symmetry.pressure?.maximumAbsoluteError ?? "missing"} exceeds ${PRESSURE_SYMMETRY_LIMIT}`);
      expect(failures, checkpoint.pressure !== undefined
        && checkpoint.pressure.maximumAbsolute > 1e-6,
      `step ${step}: projected pressure is missing or identically zero`);
      expect(failures, checkpoint.pressureIterations !== undefined
        && checkpoint.pressureIterations > 0,
      `step ${step}: pressureIterations is ${checkpoint.pressureIterations ?? "missing"}; no iterative projection was executed`);
      expect(failures, resolutionMode !== "adaptive" || (checkpoint.pressureRelativeResidual !== undefined
        && Number.isFinite(checkpoint.pressureRelativeResidual)
        && checkpoint.pressureRelativeResidual <= PRESSURE_RELATIVE_RESIDUAL_LIMIT),
      `step ${step}: pressure relative residual ${checkpoint.pressureRelativeResidual ?? "missing"} exceeds ${PRESSURE_RELATIVE_RESIDUAL_LIMIT}`);
      expect(failures, checkpoint.divergence !== undefined
        && checkpoint.divergence.nonFiniteCount === 0,
      `step ${step}: post-projection divergence publication is missing or non-finite`);
      expect(failures, checkpoint.maximumPostProjectionDivergence_s !== undefined
        && checkpoint.maximumPostProjectionDivergence_s <= postProjectionDivergenceLimit_s,
      `step ${step}: post-projection divergence ${checkpoint.maximumPostProjectionDivergence_s ?? "missing"} exceeds ${postProjectionDivergenceLimit_s}`);
      expect(failures, resolutionMode !== "adaptive"
        || checkpoint.maximumInactiveFaceSpeedAfter_m_s === 0,
        `step ${step}: pressure-inactive faces retained ${checkpoint.maximumInactiveFaceSpeedAfter_m_s ?? "missing"} m/s after projection`);
      expect(failures, resolutionMode !== "adaptive"
        || (checkpoint.maximumMixedSeamDivergence_s !== undefined
        && checkpoint.maximumMixedSeamDivergence_s <= POST_PROJECTION_DIVERGENCE_LIMIT_S),
      `step ${step}: mixed-seam divergence ${checkpoint.maximumMixedSeamDivergence_s ?? "missing"} exceeds ${POST_PROJECTION_DIVERGENCE_LIMIT_S}`);
      const divergenceAgreementTolerance = DIVERGENCE_PUBLICATION_ABSOLUTE_AGREEMENT_S
        + DIVERGENCE_PUBLICATION_RELATIVE_AGREEMENT * Math.max(
          Math.abs(checkpoint.maximumPostProjectionDivergence_s ?? Number.POSITIVE_INFINITY),
          Math.abs(checkpoint.statsMaximumPostProjectionDivergence_s ?? Number.POSITIVE_INFINITY),
        );
      expect(failures, resolutionMode !== "adaptive"
        || (checkpoint.statsMaximumPostProjectionDivergence_s !== undefined
        && Number.isFinite(checkpoint.statsMaximumPostProjectionDivergence_s)
        && checkpoint.maximumPostProjectionDivergence_s !== undefined
        && Math.abs(checkpoint.statsMaximumPostProjectionDivergence_s
          - checkpoint.maximumPostProjectionDivergence_s) <= divergenceAgreementTolerance),
      `step ${step}: max-divergence stats ${checkpoint.statsMaximumPostProjectionDivergence_s ?? "missing"} disagree with texture ${checkpoint.maximumPostProjectionDivergence_s ?? "missing"}`);
      expect(failures, checkpoint.dominantBodyMassFraction >= MINIMUM_DOMINANT_BODY_MASS_FRACTION,
      `step ${step}: dominant connected body fraction ${checkpoint.dominantBodyMassFraction} is below ${MINIMUM_DOMINANT_BODY_MASS_FRACTION}`);
      expect(failures, checkpoint.encodedSteps === step,
        `step ${step}: encodedSteps is ${checkpoint.encodedSteps ?? "missing"}`);
      expect(failures, checkpoint.hostFluidAuthority === "gpu-resident",
        `step ${step}: host fluid authority is ${checkpoint.hostFluidAuthority ?? "missing"}`);
      expect(failures, checkpoint.hostSimulationSizedWorkItems === 0,
        `step ${step}: host scheduled ${checkpoint.hostSimulationSizedWorkItems ?? "missing"} simulation-sized work items`);
      for (const [clock, actual] of [
        ["submittedTime_s", checkpoint.submittedTime_s],
        ["simulatedTime_s", checkpoint.simulatedTime_s],
        ["completedTime_s", checkpoint.completedTime_s],
      ] as const) {
        expect(failures, actual !== undefined && Math.abs(actual - target_s) <= 1e-12,
          `step ${step}: ${clock} is ${actual ?? "missing"}; expected exactly ${target_s}`);
      }
    }

    await device.queue.onSubmittedWorkDone();
    const finalCheckpoint = checkpoints.at(-1)!;
    expect(failures,
      finalCheckpoint.normalizedL1DensityChange >= MINIMUM_FINAL_NORMALIZED_L1_DENSITY_CHANGE,
      `final normalized L1 density change ${finalCheckpoint.normalizedL1DensityChange} is below ${MINIMUM_FINAL_NORMALIZED_L1_DENSITY_CHANGE}; the liquid did not visibly evolve`);
    const maximumMixedSeamFaceCount = Math.max(...checkpoints.map(
      (checkpoint) => checkpoint.adaptiveMixedSeamFaceCount ?? 0));
    const maximumPromotedBrickCount = Math.max(...checkpoints.map(
      (checkpoint) => checkpoint.adaptiveResolutionPromotedBrickCount ?? 0));
    const maximumDemotedBrickCount = Math.max(...checkpoints.map(
      (checkpoint) => checkpoint.adaptiveResolutionDemotedBrickCount ?? 0));
    const maximumDeferredPromotionCount = Math.max(...checkpoints.map(
      (checkpoint) => checkpoint.adaptiveResolutionDeferredPromotionCount ?? 0));
    expect(failures, resolutionMode !== "adaptive" || maximumMixedSeamFaceCount > 0,
      "no evolved step exercised a live 4^3/8^3 mixed-resolution seam");
    expect(failures, resolutionMode !== "adaptive" || maximumDeferredPromotionCount === 0,
      `${maximumDeferredPromotionCount} eligible promotions were deferred`);
    expect(failures, resolutionMode === "all-fine"
      ? finalCheckpoint.residentOwnerScales.length === 1
        && finalCheckpoint.residentOwnerScales[0] === 1
      : finalCheckpoint.residentOwnerScales.includes(1)
        && finalCheckpoint.residentOwnerScales.includes(2),
    `final resident topology scales ${finalCheckpoint.residentOwnerScales.join(",") || "missing"} do not include both fine (1) and coarse (2) leaves`);
    const scopedError = await device.popErrorScope();
    if (scopedError) validationErrors.push(scopedError.message);
    const finalStats = await solver.readStats();
    if (finalStats.gpuValidationError) validationErrors.push(finalStats.gpuValidationError);
    for (const message of validationErrors) failures.push(`WebGPU validation error: ${message}`);

    const maximum = (select: (checkpoint: Checkpoint) => number | undefined): number => {
      const values = checkpoints.map(select).filter(
        (value): value is number => value !== undefined && Number.isFinite(value),
      );
      return values.length > 0 ? Math.max(...values) : Number.NaN;
    };
    const report = {
      passed: failures.length === 0,
      scenario: scene.sceneId,
      method: "adaptive-mass",
      resolutionMode,
      backend,
      adapter: (adapter as GPUAdapter & { readonly info?: GPUAdapterInfo }).info,
      grid: dimensions,
      steps,
      dt_s,
      exactTargetTime_s: steps * dt_s,
      expectedInitialMass_cells,
      observedInitialMass_cells: checkpoints[0]!.mass_cells,
      maximumAbsoluteRelativeMassDrift: maximum((sample) => Math.abs(sample.relativeMassDrift)),
      finalNormalizedL1DensityChange: finalCheckpoint.normalizedL1DensityChange,
      maximumDensityD4Error: maximum((sample) => sample.symmetry.density?.maximumAbsoluteError),
      maximumVelocityD4Error_m_s: maximum((sample) => sample.symmetry.velocity?.maximumAbsoluteError),
      maximumPressureD4Error: maximum((sample) => sample.symmetry.pressure?.maximumAbsoluteError),
      maximumTopologyD4Error: maximum((sample) => sample.symmetry.topology?.maximumAbsoluteError),
      maximumPressureRelativeResidual: maximum((sample) => sample.pressureRelativeResidual),
      maximumPostProjectionDivergence_s: maximum((sample) => sample.maximumPostProjectionDivergence_s),
      maximumMixedSeamDivergence_s: maximum(
        (sample) => sample.maximumMixedSeamDivergence_s,
      ),
      maximumInactiveFaceSpeedBefore_m_s: maximum(
        (sample) => sample.maximumInactiveFaceSpeedBefore_m_s,
      ),
      maximumInactiveFaceSpeedAfter_m_s: maximum(
        (sample) => sample.maximumInactiveFaceSpeedAfter_m_s,
      ),
      maximumCfl: maximum((sample) => sample.maximumCfl),
      maximumKineticEnergyAfterFineUnits: maximum(
        (sample) => sample.kineticEnergyAfterFineUnits,
      ),
      maximumProjectionKineticEnergyAfterFineUnits: maximum(
        (sample) => sample.projectionKineticEnergyAfterFineUnits,
      ),
      maximumAdaptiveMixedSeamFaceCount: maximumMixedSeamFaceCount,
      maximumAdaptivePromotedBrickCount: maximumPromotedBrickCount,
      maximumAdaptiveDemotedBrickCount: maximumDemotedBrickCount,
      maximumAdaptiveDeferredPromotionCount: maximumDeferredPromotionCount,
      minimumDominantBodyMassFraction: Math.min(...checkpoints.map(
        (sample) => sample.dominantBodyMassFraction)),
      requiredThresholds: {
        densityD4: DENSITY_SYMMETRY_LIMIT,
        velocityD4_m_s: VELOCITY_SYMMETRY_LIMIT_M_S,
        pressureD4: PRESSURE_SYMMETRY_LIMIT,
        topologyD4: 0,
        pressureRelativeResidual: PRESSURE_RELATIVE_RESIDUAL_LIMIT,
        postProjectionDivergence_s: postProjectionDivergenceLimit_s,
        massRelativeError: MASS_RELATIVE_ERROR_LIMIT,
        minimumFinalNormalizedL1DensityChange: MINIMUM_FINAL_NORMALIZED_L1_DENSITY_CHANGE,
        divergencePublicationAbsoluteAgreement_s:
          DIVERGENCE_PUBLICATION_ABSOLUTE_AGREEMENT_S,
        divergencePublicationRelativeAgreement: DIVERGENCE_PUBLICATION_RELATIVE_AGREEMENT,
        dominantBodyMassFraction: MINIMUM_DOMINANT_BODY_MASS_FRACTION,
      },
      validationErrors,
      failures,
      finalCheckpoint: checkpoints.at(-1),
    };
    console.log(JSON.stringify(report, null, 2));
    if (failures.length > 0) {
      throw new Error(`Adaptive-mass symmetric-expansion acceptance failed (${failures.length} gates)`);
    }
  } finally {
    solver?.destroy();
    // If construction/capture threw before the normal pop, drain the scope.
    await device.popErrorScope().catch(() => null);
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
