/** Exact uniform-translation probe for production Sparse Geometric volume. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  createGeometricUniformTranslationScene,
  GEOMETRIC_TRANSLATION_BLOCK,
  GEOMETRIC_TRANSLATION_CELL_SIZE_M,
  GEOMETRIC_TRANSLATION_DETACHED_BLOCK,
  GEOMETRIC_TRANSLATION_DT_S,
  GEOMETRIC_TRANSLATION_DURATION_S,
  GEOMETRIC_TRANSLATION_SPEED_M_S,
} from "../lib/core/geometric-translation-scene";
import { SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { fluidExecutionDeviceFeatures } from "../lib/core/gpu-startup";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager, managedGPUDevice } from
  "../lib/core/gpu-compilation-manager";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod as adaptiveVolumeMethod } from
  "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { createGeometricDamStageEnergy } from "./geometric-dam-stage-energy";
import { fingerprintSparseCM12RepositorySources } from
  "./sparse-cm12-source-content-fingerprint";

const argument = (name: string, fallback = "") => process.argv.slice(2)
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const detached = process.argv.includes("--detached");
const transportDetail = process.argv.includes("--transport-detail");
if (process.argv.includes("--help")) {
  console.log(`Sparse Geometric exact uniform-translation probe
node --import tsx tools/probe-geometric-uniform-translation-dawn.ts [options]
  --dt=${GEOMETRIC_TRANSLATION_DT_S}       Outer timestep in seconds
  --speed=${GEOMETRIC_TRANSLATION_SPEED_M_S}             Uniform +X velocity in m/s
  --duration=${GEOMETRIC_TRANSLATION_DURATION_S}        Simulated duration in seconds
  --output=PATH        Atomic JSON receipt (alias: --out)
  --detached           Use the 4x4x8 detached block instead of the default 4x8x8 plug
  --transport-detail   Capture selected accepted cell planes, physical subface fluxes and z=4 density slices

The default 16x8x8 width-one scene has CFL=2 and runs three 1/60 s steps.
Before wall contact its exact Euler solution is the unchanged box translated
at 6 m/s. The criterion compares accepted cell-integrated volume fractions,
volume, cell-quadrature centre of mass, volume-weighted velocity and kinetic
energy to that manufactured solution. A failed criterion is still written.`);
  process.exit(0);
}

const dt_s = Number(argument("dt", String(GEOMETRIC_TRANSLATION_DT_S)));
const speed_m_s = Number(argument("speed", String(GEOMETRIC_TRANSLATION_SPEED_M_S)));
const duration_s = Number(argument("duration", String(GEOMETRIC_TRANSLATION_DURATION_S)));
const outputPath = resolve(argument("output", argument("out",
  "artifacts/analytic-motion/geometric-uniform-translation.json")));
assert.ok(Number.isFinite(dt_s) && dt_s > 0, "--dt must be finite and positive");
assert.ok(Number.isFinite(speed_m_s) && speed_m_s > 0, "--speed must be finite and positive");
assert.ok(Number.isFinite(duration_s) && duration_s > 0, "--duration must be finite and positive");

const CELL_SIZE_M = GEOMETRIC_TRANSLATION_CELL_SIZE_M;
const DIMENSIONS = [16, 8, 8] as const;
const F32_EPSILON = 2 ** -23;
const EXTENSIVE_RELATIVE_LIMIT = 16 * F32_EPSILON;
const EXECUTED_DT_RELATIVE_LIMIT = 8 * F32_EPSILON;
const VOLUME_FIELD_RELATIVE_L1_LIMIT = 1e-3;
const CENTER_ERROR_CELL_FRACTION_LIMIT = 1e-3;
const VELOCITY_RELATIVE_ERROR_LIMIT = 1e-4;
const KINETIC_RELATIVE_ERROR_LIMIT = 1e-4;
const block = detached ? GEOMETRIC_TRANSLATION_DETACHED_BLOCK : GEOMETRIC_TRANSLATION_BLOCK;
const expectedVolume_m3 = block.dimensions_m.x * block.dimensions_m.y
  * block.dimensions_m.z;
assert.ok(block.origin_m.x + block.dimensions_m.x + speed_m_s * duration_s < 0.8,
  "requested translation reaches the downstream wall");

function intervalOverlapFraction(cell: number, minimum_m: number, maximum_m: number): number {
  const lower_m = cell * CELL_SIZE_M, upper_m = lower_m + CELL_SIZE_M;
  return Math.max(0, Math.min(upper_m, maximum_m) - Math.max(lower_m, minimum_m))
    / CELL_SIZE_M;
}

function expectedDensityAt(time_s: number): Float64Array {
  const result = new Float64Array(DIMENSIONS[0] * DIMENSIONS[1] * DIMENSIONS[2]);
  const minimum = { x: block.origin_m.x + speed_m_s * time_s,
    y: block.origin_m.y, z: block.origin_m.z };
  const maximum = { x: minimum.x + block.dimensions_m.x,
    y: minimum.y + block.dimensions_m.y, z: minimum.z + block.dimensions_m.z };
  for (let z = 0; z < DIMENSIONS[2]; z += 1) for (let y = 0; y < DIMENSIONS[1]; y += 1)
    for (let x = 0; x < DIMENSIONS[0]; x += 1) {
      result[x + DIMENSIONS[0] * (y + DIMENSIONS[1] * z)] =
        intervalOverlapFraction(x, minimum.x, maximum.x)
        * intervalOverlapFraction(y, minimum.y, maximum.y)
        * intervalOverlapFraction(z, minimum.z, maximum.z);
    }
  return result;
}

interface FieldLike {
  readonly density: ArrayLike<number>;
  readonly velocity?: ArrayLike<number>;
}

function summarizeField(field: FieldLike, waterDensity: number) {
  const cellVolume_m3 = CELL_SIZE_M ** 3;
  let signedVolume_m3 = 0, positiveVolume_m3 = 0, kinetic_J = 0;
  const firstMoment_m4 = [0, 0, 0], momentum_m4_s = [0, 0, 0];
  const supportMinimum = [Infinity, Infinity, Infinity];
  const supportMaximum = [-Infinity, -Infinity, -Infinity];
  const yzColumnVolume_m3 = new Array<number>(DIMENSIONS[1] * DIMENSIONS[2]).fill(0);
  for (let z = 0; z < DIMENSIONS[2]; z += 1) for (let y = 0; y < DIMENSIONS[1]; y += 1)
    for (let x = 0; x < DIMENSIONS[0]; x += 1) {
      const cell = x + DIMENSIONS[0] * (y + DIMENSIONS[1] * z);
      const fraction = Number(field.density[cell]);
      const signedVolume = fraction * cellVolume_m3;
      signedVolume_m3 += signedVolume;
      yzColumnVolume_m3[y + DIMENSIONS[1] * z] += signedVolume;
      if (!(fraction > 0)) continue;
      positiveVolume_m3 += signedVolume;
      const center = [(x + 0.5) * CELL_SIZE_M, (y + 0.5) * CELL_SIZE_M,
        (z + 0.5) * CELL_SIZE_M];
      for (let axis = 0; axis < 3; axis += 1) firstMoment_m4[axis]! += signedVolume * center[axis]!;
      for (let axis = 0; axis < 3; axis += 1) {
        supportMinimum[axis] = Math.min(supportMinimum[axis]!, center[axis]! - 0.5 * CELL_SIZE_M);
        supportMaximum[axis] = Math.max(supportMaximum[axis]!, center[axis]! + 0.5 * CELL_SIZE_M);
      }
      if (field.velocity) {
        let squared = 0;
        for (let axis = 0; axis < 3; axis += 1) {
          const velocity = Number(field.velocity[4 * cell + axis]);
          momentum_m4_s[axis]! += signedVolume * velocity;
          squared += velocity * velocity;
        }
        kinetic_J += 0.5 * waterDensity * signedVolume * squared;
      }
    }
  return {
    signedVolume_m3, positiveVolume_m3,
    centerOfVolumeLocal_m: positiveVolume_m3 > 0
      ? firstMoment_m4.map(value => value / positiveVolume_m3) : null,
    volumeWeightedVelocity_m_s: field.velocity && positiveVolume_m3 > 0
      ? momentum_m4_s.map(value => value / positiveVolume_m3) : null,
    kinetic_J: field.velocity ? kinetic_J : undefined,
    positiveSupportBoundsLocal_m: positiveVolume_m3 > 0
      ? { minimum: supportMinimum, maximum: supportMaximum } : null,
    yzColumnVolume_m3,
  };
}

function maximumAbsoluteDifference(left: ArrayLike<number>, right: ArrayLike<number>): number {
  assert.equal(left.length, right.length);
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(Number(left[index]) - Number(right[index])));
  }
  return maximum;
}

function relativeVolumeL1(left: ArrayLike<number>, right: ArrayLike<number>): number {
  assert.equal(left.length, right.length);
  let error_m3 = 0;
  for (let index = 0; index < left.length; index += 1) {
    error_m3 += Math.abs(Number(left[index]) - Number(right[index])) * CELL_SIZE_M ** 3;
  }
  return error_m3 / expectedVolume_m3;
}

function expectedVelocityField(cellCount: number): Float64Array {
  const result = new Float64Array(4 * cellCount);
  for (let cell = 0; cell < cellCount; cell += 1) result[4 * cell] = speed_m_s;
  return result;
}

type DiagnosticFields = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>;
type ActivityPolicy = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>;
type SelectedCell = Readonly<{ x: number; y: number; z: number; cellId: number;
  centerFine: readonly [number, number, number]; widthsFine: readonly [number, number, number] }>;
type InterfaceSnapshot = Readonly<{ acceptedDensity: number; reconstructionFill: number;
  acceptedAmountFine3: number; plane: Readonly<{ normal: readonly [number, number, number];
    offsetFine: number; valid: boolean }> }>;

function selectedAuthoredTransportCells(solver: WebGPUAdaptiveMassSolver,
  activity: ActivityPolicy): SelectedCell[] {
  const source = solver.fieldSnapshotSourceForQA;
  const words = source.templateWords;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const result: SelectedCell[] = [];
  for (let x = 2; x <= 8; x += 1) {
    let found: SelectedCell | undefined;
    for (const brick of activity.bricks) {
      if (!brick.active || brick.leafId >= words[13]!) continue;
      const level = Math.log2(brick.acceptedResolution);
      const range = words[11]! + 2 * (4 * brick.leafId + level);
      for (let cellId = words[range]!; cellId < words[range]! + words[range + 1]!; cellId += 1) {
        const at = words[6]! + 8 * cellId;
        if (floats[at] === x + 0.5 && floats[at + 1] === 2.5 && floats[at + 2] === 4.5) {
          found = { x, y: 2, z: 4, cellId,
            centerFine: [floats[at]!, floats[at + 1]!, floats[at + 2]!],
            widthsFine: [floats[at + 4]!, floats[at + 5]!, floats[at + 6]!] };
          break;
        }
      }
      if (found) break;
    }
    assert.ok(found, `transport-detail cell (${x},2,4) is not an active authored cell`);
    result.push(found);
  }
  return result;
}

async function readInterfaceSnapshots(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  fields: DiagnosticFields, cells: readonly SelectedCell[]): Promise<Map<number, InterfaceSnapshot>> {
  const source = solver.fieldSnapshotSourceForQA;
  const readback = device.createBuffer({ label: "Uniform translation interface-plane QA",
    size: 16 * cells.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    cells.forEach((cell, index) => encoder.copyBufferToBuffer(source.state,
      4 * (source.layout.geometricInterfacePlanes + 4 * cell.cellId), readback, 16 * index, 16));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange());
    return new Map(cells.map((cell, index) => {
      const at = 4 * index;
      const normal = [values[at]!, values[at + 1]!, values[at + 2]!] as const;
      const dense = cell.x + DIMENSIONS[0] * (cell.y + DIMENSIONS[1] * cell.z);
      const acceptedDensity = fields.density[dense]!;
      const open = fields.solidOpenFraction[dense]!;
      return [cell.cellId, { acceptedDensity,
        reconstructionFill: acceptedDensity / Math.max(open, 1e-8),
        acceptedAmountFine3: acceptedDensity
          * cell.widthsFine[0] * cell.widthsFine[1] * cell.widthsFine[2],
        plane: { normal, offsetFine: values[at + 3]!,
          valid: normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2 > 0.5 } }] as const;
    }));
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

function densitySliceZ4(fields: DiagnosticFields, origin_m: readonly number[]) {
  return { zIndex: 4, centerFineZ: 4.5,
    vesselCenterZ_m: 4.5 * CELL_SIZE_M,
    worldCenterZ_m: origin_m[2]! + 4.5 * CELL_SIZE_M,
    rows: Array.from({ length: DIMENSIONS[1] }, (_, y) => ({ y,
      cells: Array.from({ length: DIMENSIONS[0] }, (_, x) => {
        const at = x + DIMENSIONS[0] * (y + DIMENSIONS[1] * 4);
        return { x, y, z: 4, centerFine: [x + 0.5, y + 0.5, 4.5],
          vesselCenter_m: [(x + 0.5) * CELL_SIZE_M, (y + 0.5) * CELL_SIZE_M,
            4.5 * CELL_SIZE_M],
          worldCenter_m: [origin_m[0]! + (x + 0.5) * CELL_SIZE_M,
            origin_m[1]! + (y + 0.5) * CELL_SIZE_M,
            origin_m[2]! + 4.5 * CELL_SIZE_M],
          acceptedDensity: fields.density[at]! };
      }) })),
  };
}

const root = fileURLToPath(new URL("..", import.meta.url));
const sourceFingerprint = await fingerprintSparseCM12RepositorySources(root);
const profile = SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE;
assert.equal(profile.methodId, "adaptive-volume");
assert.equal(adaptiveVolumeMethod.id, "adaptive-volume");
const values = resolveMethodValues(adaptiveVolumeMethod, profile.quality,
  { ...profile.overrides, timeStep: "scene" });
const scene = createGeometricUniformTranslationScene({ speed_m_s, dt_s, duration_s, detached });

const report: Record<string, unknown> & { frames: Array<Record<string, unknown>> } = {
  probe: "geometric-uniform-translation-dawn", status: "initializing", passed: false,
  passScope: "exact cell-integrated translated volume fraction, extensive volume, cell-quadrature centre, accepted velocity/kinetic energy, physical bounds, all-fine topology and source stability",
  implementation: { methodId: adaptiveVolumeMethod.id,
    sourceFingerprint, commit: execFileSync("git", ["rev-parse", "HEAD"],
      { cwd: root, encoding: "utf8" }).trim() },
  configuration: { dt_s, speed_m_s, duration_s, cfl: speed_m_s * dt_s / CELL_SIZE_M,
    detached, transportDetail, dimensions: DIMENSIONS, cellSize_m: CELL_SIZE_M,
    block, expectedVolume_m3, expectedMass_kg: scene.fluid.density_kg_m3 * expectedVolume_m3,
    expectedKinetic_J: 0.5 * scene.fluid.density_kg_m3 * expectedVolume_m3 * speed_m_s ** 2,
    methodProfile: profile, resolvedMethodValues: values },
  criteria: { volumeFieldRelativeL1Error: VOLUME_FIELD_RELATIVE_L1_LIMIT,
    centerErrorCellFraction: CENTER_ERROR_CELL_FRACTION_LIMIT,
    velocityRelativeError: VELOCITY_RELATIVE_ERROR_LIMIT,
    kineticRelativeError: KINETIC_RELATIVE_ERROR_LIMIT,
    extensiveRelativeRoundoff: EXTENSIVE_RELATIVE_LIMIT,
    executedDtRelativeRoundoff: EXECUTED_DT_RELATIVE_LIMIT,
    origin: "Analytic accuracy budgets and separate f32 volume-roundoff bounds selected before execution; no empirical baseline" },
  frames: [],
};

async function checkpoint(): Promise<void> {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(report, null, 2)}\n`);
  await rename(temporary, outputPath);
}

let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
let stageObserver: Awaited<ReturnType<typeof createGeometricDamStageEnergy>> | undefined;
let selectedTransportCells: SelectedCell[] | undefined;
let previousInterfaceSnapshots: Map<number, InterfaceSnapshot> | undefined;
let analyticalFailure = false, runtimeFailure = false;
const validationErrors: string[] = [];
await checkpoint();
await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-geometric-uniform-translation-dawn.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(resolve(modulePath)).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn,
    [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn adapter unavailable");
  const rawDevice = await adapter.requestDevice({
    requiredFeatures: fluidExecutionDeviceFeatures(adapter.features),
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  device = managedGPUDevice(rawDevice, { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
  device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));
  solver = await adaptiveVolumeMethod.createSolverAsync!(device, scene, profile.quality,
    values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady(); await solver.waitForTopologyReady();
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], DIMENSIONS);
  solver.setTopologyFrozen(true);
  const initialActivity = await solver.readGPUActivityPolicy();
  const initialTopologyGeneration = solver.info.topologyGenerationCount;
  assert.ok(initialActivity.bricks.filter(brick => brick.active)
    .every(brick => brick.acceptedResolution === 8), "initial topology is not all-fine width one");
  report.initialTopology = { topologyGenerationCount: initialTopologyGeneration,
    activeResolutionCounts: { "8": initialActivity.bricks.filter(brick => brick.active).length } };
  const initialFields = await solver.readDiagnosticFields(true);
  const initialPhysical = await solver.readAcceptedGeometricVolumeQA();
  const initialExpectedDensity = expectedDensityAt(0);
  const initialObserved = summarizeField(initialFields, scene.fluid.density_kg_m3);
  const initialExpected = summarizeField({ density: initialExpectedDensity,
    velocity: expectedVelocityField(initialExpectedDensity.length) }, scene.fluid.density_kg_m3);
  const initialRelativeVolumeL1 = relativeVolumeL1(initialFields.density, initialExpectedDensity);
  const initialDenseGlobalDifference_m3 = initialObserved.signedVolume_m3
    - initialPhysical.volume_m3;
  const initialVelocityError_m_s = initialObserved.volumeWeightedVelocity_m_s
    ? Math.hypot(initialObserved.volumeWeightedVelocity_m_s[0]! - speed_m_s,
      initialObserved.volumeWeightedVelocity_m_s[1]!,
      initialObserved.volumeWeightedVelocity_m_s[2]!) : Infinity;
  const initialKineticError_J = (initialObserved.kinetic_J ?? NaN)
    - (initialExpected.kinetic_J ?? NaN);
  const initialPassed = initialRelativeVolumeL1 <= VOLUME_FIELD_RELATIVE_L1_LIMIT
    && Math.abs(initialObserved.signedVolume_m3 - expectedVolume_m3)
      <= EXTENSIVE_RELATIVE_LIMIT * expectedVolume_m3
    && Math.abs(initialDenseGlobalDifference_m3)
      <= EXTENSIVE_RELATIVE_LIMIT * expectedVolume_m3
    && initialVelocityError_m_s <= VELOCITY_RELATIVE_ERROR_LIMIT * speed_m_s
    && Math.abs(initialKineticError_J)
      <= KINETIC_RELATIVE_ERROR_LIMIT * (initialExpected.kinetic_J ?? Infinity)
    && initialPhysical.invalidCells === 0 && initialPhysical.nonfiniteDynamicsCells === 0
    && initialPhysical.outsideAuthoredVolumeFine3 === 0;
  analyticalFailure ||= !initialPassed;
  report.initial = { acceptedTime_s: 0, observed: initialObserved, expected: initialExpected,
    physical: initialPhysical, relativeVolumeL1: initialRelativeVolumeL1,
    denseGlobalDifference_m3: initialDenseGlobalDifference_m3,
    velocityError_m_s: initialVelocityError_m_s, kineticError_J: initialKineticError_J,
    checks: { exactInjectedCellIntegratedVolume: initialRelativeVolumeL1
      <= VOLUME_FIELD_RELATIVE_L1_LIMIT,
    extensiveVolume: Math.abs(initialObserved.signedVolume_m3 - expectedVolume_m3)
      <= EXTENSIVE_RELATIVE_LIMIT * expectedVolume_m3,
    denseCoversAcceptedAuthority: Math.abs(initialDenseGlobalDifference_m3)
      <= EXTENSIVE_RELATIVE_LIMIT * expectedVolume_m3,
    volumeWeightedVelocity: initialVelocityError_m_s
      <= VELOCITY_RELATIVE_ERROR_LIMIT * speed_m_s,
    kineticEnergy: Math.abs(initialKineticError_J)
      <= KINETIC_RELATIVE_ERROR_LIMIT * (initialExpected.kinetic_J ?? Infinity),
    acceptedBounds: initialPhysical.invalidCells === 0
      && initialPhysical.nonfiniteDynamicsCells === 0
      && initialPhysical.outsideAuthoredVolumeFine3 === 0 }, passed: initialPassed };
  if (transportDetail) {
    selectedTransportCells = selectedAuthoredTransportCells(solver, initialActivity);
    previousInterfaceSnapshots = await readInterfaceSnapshots(device, solver,
      initialFields, selectedTransportCells);
    report.initialTransportDetail = {
      scope: "published interface cache before the first completed transport frame",
      selectedCells: selectedTransportCells.map(cell => ({ ...cell,
        interface: previousInterfaceSnapshots!.get(cell.cellId) })),
      densitySlice: densitySliceZ4(initialFields, solver.fluidDomain.origin_m),
    };
  }
  report.status = "running";
  await checkpoint();

  let acceptedTime_s = 0, executedPhysicalTime_s = 0, frameIndex = 0;
  while (acceptedTime_s < duration_s - 1e-10) {
    const targetTime_s = Math.min(duration_s, acceptedTime_s + dt_s);
    for (;;) {
      await solver.waitForTopologyReady();
      stageObserver?.dispose();
      stageObserver = await createGeometricDamStageEnergy(device, solver, CELL_SIZE_M,
        scene.fluid.density_kg_m3, { x: speed_m_s, y: 0, z: 0 });
      stageObserver.arm();
      if (solver.advanceTo(targetTime_s, [])) break;
      stageObserver.dispose(); stageObserver = undefined;
      await new Promise<void>(done => setImmediate(done));
    }
    await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
    const prefixStageEnergy = await stageObserver.read();
    stageObserver.dispose(); stageObserver = undefined;
    const fields = await solver.readDiagnosticFields(true);
    const physical = await solver.readAcceptedGeometricVolumeQA();
    const transport = await solver.readGeometricVolumeTransportReceiptQA();
    const stats = await solver.readStats();
    const activity = await solver.readGPUActivityPolicy();
    assert.ok(stats.completedTime_s !== undefined, "solver omitted accepted clock");
    const previousAcceptedTime_s = acceptedTime_s;
    acceptedTime_s = stats.completedTime_s; frameIndex += 1;
    const requestedFrameDt_s = acceptedTime_s - previousAcceptedTime_s;
    const executedFrameDt_s = transport.executedPhysicalDt_s;
    executedPhysicalTime_s += executedFrameDt_s;
    const executedFrameDtError_s = executedFrameDt_s - requestedFrameDt_s;
    const acceptedMinusExecutedTime_s = acceptedTime_s - executedPhysicalTime_s;
    const cumulativeExecutedClockLimit_s = EXECUTED_DT_RELATIVE_LIMIT * acceptedTime_s;
    const expectedDensity = expectedDensityAt(acceptedTime_s);
    const expectedDensityAtExecutedTime = expectedDensityAt(executedPhysicalTime_s);
    const observed = summarizeField(fields, scene.fluid.density_kg_m3);
    const expected = summarizeField({ density: expectedDensity,
      velocity: expectedVelocityField(expectedDensity.length) }, scene.fluid.density_kg_m3);
    const expectedAtExecutedPhysicalTime = summarizeField({ density: expectedDensityAtExecutedTime,
      velocity: expectedVelocityField(expectedDensityAtExecutedTime.length) },
    scene.fluid.density_kg_m3);
    const densityMaximumAbsoluteError = maximumAbsoluteDifference(fields.density, expectedDensity);
    const volumeFieldRelativeL1Error = relativeVolumeL1(fields.density, expectedDensity);
    const volumeFieldRelativeL1ErrorAtExecutedPhysicalTime = relativeVolumeL1(
      fields.density, expectedDensityAtExecutedTime);
    const volumeError_m3 = observed.signedVolume_m3 - expectedVolume_m3;
    const globalDenseDifference_m3 = observed.signedVolume_m3 - physical.volume_m3;
    const volumeScale = Math.max(expectedVolume_m3, CELL_SIZE_M ** 3);
    const centerError_m = observed.centerOfVolumeLocal_m && expected.centerOfVolumeLocal_m
      ? Math.hypot(...observed.centerOfVolumeLocal_m.map((value, axis) =>
        value - expected.centerOfVolumeLocal_m![axis]!)) : Infinity;
    const centerErrorAtExecutedPhysicalTime_m = observed.centerOfVolumeLocal_m
      && expectedAtExecutedPhysicalTime.centerOfVolumeLocal_m
      ? Math.hypot(...observed.centerOfVolumeLocal_m.map((value, axis) =>
        value - expectedAtExecutedPhysicalTime.centerOfVolumeLocal_m![axis]!)) : Infinity;
    const velocityError_m_s = observed.volumeWeightedVelocity_m_s
      ? Math.hypot(observed.volumeWeightedVelocity_m_s[0]! - speed_m_s,
        observed.volumeWeightedVelocity_m_s[1]!, observed.volumeWeightedVelocity_m_s[2]!) : Infinity;
    const kineticError_J = (observed.kinetic_J ?? NaN) - (expected.kinetic_J ?? NaN);
    const yzColumnMaximumVolumeError_m3 = maximumAbsoluteDifference(
      observed.yzColumnVolume_m3, expected.yzColumnVolume_m3);
    const activeResolutionCounts = Object.fromEntries([...new Set(activity.bricks
      .filter(brick => brick.active).map(brick => brick.acceptedResolution))]
      .sort((a, b) => a - b).map(resolution => [String(resolution), activity.bricks
        .filter(brick => brick.active && brick.acceptedResolution === resolution).length]));
    let frameTransportDetail: Record<string, unknown> | undefined;
    if (transportDetail) {
      assert.ok(selectedTransportCells && previousInterfaceSnapshots);
      const currentInterfaceSnapshots = await readInterfaceSnapshots(device, solver,
        fields, selectedTransportCells);
      const cells = [];
      for (const cell of selectedTransportCells) {
        const rowQA = await solver.readAcceptedGeometricCellRowsQA(cell.cellId);
        const current = currentInterfaceSnapshots.get(cell.cellId)!;
        const before = previousInterfaceSnapshots.get(cell.cellId)!;
        cells.push({ ...cell,
          acceptedStateAfterFrame: { density: current.acceptedDensity,
            amountFine3: current.acceptedAmountFine3,
            fullCellVolumeFine3: rowQA.volumeFine3,
            transportCurrentVolumeScratchFine3: rowQA.physicalState.currentVolumeFine3,
            transportLowVolumeScratchFine3: rowQA.physicalState.lowVolumeFine3 },
          interfaceCacheBeforeFrame: before,
          interfaceCacheAfterFrame: current,
          physicalSubfaceFluxesFromLastCompletedMicrostep: rowQA.physicalSubfaces.map(face => ({
            face: face.face, rowId: face.rowId, ownNegative: face.ownNegative,
            areaFine2: face.areaFine2, lowFluxFine3: face.lowFluxFine3,
            highFluxFine3: face.highFluxFine3, limitedFluxFine3: face.limitedFluxFine3,
            signedSweepFine3: face.signedSweepFine3,
            residualFluxFine3: face.residualFluxFine3,
            negative: face.negative, positive: face.positive,
          })),
          incidenceRows: rowQA.rows,
        });
      }
      frameTransportDetail = {
        epochSemantics: "Fluxes are from the last completed volume microstep. interfaceCacheBeforeFrame is the previously published plane; for the requested one-microstep frame it is the plane available to these fluxes. interfaceCacheAfterFrame is reconstructed from the newly accepted scalar and is input to the next frame.",
        completedTransportSubsteps: transport.executedSubsteps,
        selectedCellLine: { y: 2, z: 4, xInclusive: [2, 8] },
        cells,
        densitySlice: densitySliceZ4(fields, solver.fluidDomain.origin_m),
      };
      previousInterfaceSnapshots = currentInterfaceSnapshots;
    }
    const checks = {
      exactCellIntegratedDensity: volumeFieldRelativeL1Error <= VOLUME_FIELD_RELATIVE_L1_LIMIT,
      exactCellIntegratedDensityAtExecutedPhysicalTime:
        volumeFieldRelativeL1ErrorAtExecutedPhysicalTime <= VOLUME_FIELD_RELATIVE_L1_LIMIT,
      extensiveVolume: Math.abs(volumeError_m3) <= EXTENSIVE_RELATIVE_LIMIT * volumeScale,
      denseCoversAcceptedAuthority: Math.abs(globalDenseDifference_m3)
        <= EXTENSIVE_RELATIVE_LIMIT * volumeScale,
      cellQuadratureCenter: centerError_m <= CENTER_ERROR_CELL_FRACTION_LIMIT * CELL_SIZE_M,
      cellQuadratureCenterAtExecutedPhysicalTime:
        centerErrorAtExecutedPhysicalTime_m <= CENTER_ERROR_CELL_FRACTION_LIMIT * CELL_SIZE_M,
      volumeWeightedVelocity: velocityError_m_s
        <= VELOCITY_RELATIVE_ERROR_LIMIT * Math.max(speed_m_s, 1),
      kineticEnergy: Math.abs(kineticError_J)
        <= KINETIC_RELATIVE_ERROR_LIMIT * Math.max(expected.kinetic_J ?? 0, 1),
      yzColumnVolume: yzColumnMaximumVolumeError_m3
        <= VOLUME_FIELD_RELATIVE_L1_LIMIT * Math.max(block.dimensions_m.x * CELL_SIZE_M ** 2,
          CELL_SIZE_M ** 3),
      acceptedBounds: physical.invalidCells === 0 && physical.nonfiniteDynamicsCells === 0
        && physical.outsideAuthoredVolumeFine3 === 0,
      allFineTopology: activity.bricks.filter(brick => brick.active)
        .every(brick => brick.acceptedResolution === 8),
      topologyUnchanged: stats.topologyGenerationCount === initialTopologyGeneration,
      transportComplete: transport.fault === 0 && transport.transportCompleted,
      publishedClockMatchesRequest: Math.abs(acceptedTime_s - targetTime_s) <= 1e-12,
      executedDtMatchesPublishedAdvance: Math.abs(executedFrameDtError_s)
        <= EXECUTED_DT_RELATIVE_LIMIT * requestedFrameDt_s,
      cumulativeExecutedClockMatchesPublishedClock:
        Math.abs(acceptedMinusExecutedTime_s) <= cumulativeExecutedClockLimit_s,
      validationClean: validationErrors.length === 0,
    };
    const passed = Object.values(checks).every(Boolean);
    analyticalFailure ||= !passed;
    report.frames.push({ frameIndex, requestedTime_s: targetTime_s, acceptedTime_s,
      expectedBoxAtAcceptedTimeLocal_m: { minimum: {
        x: block.origin_m.x + speed_m_s * acceptedTime_s,
        y: block.origin_m.y, z: block.origin_m.z }, maximum: {
        x: block.origin_m.x + block.dimensions_m.x + speed_m_s * acceptedTime_s,
        y: block.origin_m.y + block.dimensions_m.y,
        z: block.origin_m.z + block.dimensions_m.z } },
      expectedBoxAtExecutedPhysicalTimeLocal_m: { minimum: {
        x: block.origin_m.x + speed_m_s * executedPhysicalTime_s,
        y: block.origin_m.y, z: block.origin_m.z }, maximum: {
        x: block.origin_m.x + block.dimensions_m.x + speed_m_s * executedPhysicalTime_s,
        y: block.origin_m.y + block.dimensions_m.y,
        z: block.origin_m.z + block.dimensions_m.z } },
      clocks: { requestedFrameDt_s, executedFrameDt_s, executedFrameDtError_s,
        acceptedTime_s, executedPhysicalTime_s, acceptedMinusExecutedTime_s,
        cumulativeExecutedClockLimit_s },
      observed, expectedAtAcceptedTime: expected,
      expectedAtExecutedPhysicalTime,
      errors: { densityMaximumAbsoluteError, volumeFieldRelativeL1Error,
        volumeFieldRelativeL1ErrorAtExecutedPhysicalTime, volumeError_m3,
        globalDenseDifference_m3, centerError_m, velocityError_m_s, kineticError_J,
        centerErrorAtExecutedPhysicalTime_m, yzColumnMaximumVolumeError_m3 },
      stageEnergy: { prefix: prefixStageEnergy,
        afterConservativeTransport: {
          scope: "accepted post-frame state; no later frame stage changes fluid volume or velocity",
          mass_kg: scene.fluid.density_kg_m3 * observed.signedVolume_m3,
          volumeWeightedVelocity_m_s: observed.volumeWeightedVelocity_m_s,
          kinetic_J: observed.kinetic_J,
        } },
      physical, transport, activeResolutionCounts,
      ...(frameTransportDetail ? { transportDetail: frameTransportDetail } : {}),
      stats: { completedTime_s: stats.completedTime_s, encodedSteps: stats.encodedSteps,
        topologyGenerationCount: stats.topologyGenerationCount }, checks, passed });
    report.lastAcceptedTime_s = acceptedTime_s;
    report.lastExecutedPhysicalTime_s = executedPhysicalTime_s;
    await checkpoint();
    console.error(JSON.stringify({ probe: report.probe, frameIndex, acceptedTime_s, passed,
      densityMaximumAbsoluteError, volumeFieldRelativeL1Error, volumeError_m3, velocityError_m_s }));
  }
  report.status = analyticalFailure ? "failed-analytic-criterion" : "complete";
  report.validationErrors = validationErrors;
} catch (error) {
  runtimeFailure = true; report.status = "failed-runtime";
  report.failure = error instanceof Error ? error.stack : String(error);
  report.validationErrors = validationErrors;
} finally {
  stageObserver?.dispose(); stageObserver = undefined;
  if (device) {
    const manager = gpuCompilationManagerFor(device);
    try {
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
      solver?.destroy(); solver = undefined;
      invalidateGPUCompilationManager(device, "uniform-translation probe complete");
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
    } finally { solver?.destroy(); device.destroy(); }
  }
  await releaseWebGPUExclusiveLock();
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(root);
  report.sourceUnchanged = (report.sourceFingerprintAfter as { sha256: string }).sha256
    === sourceFingerprint.sha256;
  report.passed = !runtimeFailure && !analyticalFailure && report.sourceUnchanged === true;
  await checkpoint();
}

console.log(JSON.stringify(report, null, 2));
if (report.passed !== true) process.exitCode = 1;
