/** Analytical dry-bed dam benchmark for the production Sparse Geometric path.
 * Run serially under the repository Dawn lease. This is a measurement probe:
 * it records reference errors and arrival brackets without result-fitted gates.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  createSparseGeometricRitterDamBreakScene,
  SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE,
} from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
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
const hasFlag = (name: string) => process.argv.includes(`--${name}`);

if (hasFlag("help")) {
  console.log(`Sparse Geometric Ritter dry-bed dam benchmark
node --import tsx tools/probe-adaptive-volume-dam-front-dawn.ts [options]
  --dt=0.03333333333333333  Requested production outer step, seconds
  --duration=0.7             Simulated duration; 0.7 stays pre-impact
  --cell-size=0.0125         Finest lattice cell size, metres
  --fixed-cell-width=1       Optional whole-domain fixed width (1,2,4,8 fine cells)
  --stage-energy             Capture optional pre-transport stage energy taps
  --timeout-ms=180000        Per initialization/admission/completion deadline
  --out=PATH                 Atomic JSON checkpoint after initialization and every frame

The default run uses the scene's production adaptive-volume balanced profile.
Depth is reconstructed conservatively from accepted physical V, and the dense
sum is checked against the independent accepted-volume GPU reduction. Ritter
profiles are averaged over the same x bins. No empirical pass ceilings are
applied to profile error, front position, arrival time, or energy.`);
  process.exit(0);
}

const dt_s = Number(argument("dt", String(1 / 30)));
const duration_s = Number(argument("duration", "0.7"));
const cellSize_m = Number(argument("cell-size", "0.0125"));
const fixedCellWidthText = argument("fixed-cell-width");
const fixedCellWidth = fixedCellWidthText === "" ? undefined : Number(fixedCellWidthText);
const stageEnergyEnabled = hasFlag("stage-energy");
const outputPath = argument("out");
const timeout_ms = Number(argument("timeout-ms", "180000"));
assert.ok(Number.isFinite(dt_s) && dt_s > 0, "--dt must be finite and positive");
assert.ok(Number.isFinite(duration_s) && duration_s > 0, "--duration must be finite and positive");
assert.ok(Number.isFinite(cellSize_m) && cellSize_m > 0, "--cell-size must be finite and positive");
assert.ok(fixedCellWidth === undefined || [1, 2, 4, 8].includes(fixedCellWidth),
  "--fixed-cell-width must be one of 1,2,4,8");
assert.ok(Number.isFinite(timeout_ms) && timeout_ms > 0, "--timeout-ms must be finite and positive");

const GRAVITY_M_S2 = 9.81;
const HEAD_M = 0.1;
const CHANNEL_WIDTH_M = 0.1;
const RESERVOIR_LENGTH_M = 2.4;
const DOWNSTREAM_LENGTH_M = 1.6;
const INITIAL_VOLUME_M3 = RESERVOIR_LENGTH_M * HEAD_M * CHANNEL_WIDTH_M;
const C0_M_S = Math.sqrt(GRAVITY_M_S2 * HEAD_M);
const TIP_WALL_TIME_S = DOWNSTREAM_LENGTH_M / (2 * C0_M_S);
const THRESHOLDS = [0.01, 0.05, 0.1] as const;
const SENSOR_X_M = [0.4, 0.8, 1.2, 1.6] as const;
const F32_EPSILON = 2 ** -23;

function fanDepthPrimitive(x: number, time_s: number): number {
  const a = 2 * C0_M_S - x / time_s;
  return -time_s * a ** 3 / (27 * GRAVITY_M_S2);
}

function fanDischargePrimitive(x: number, time_s: number): number {
  const a = 2 * C0_M_S - x / time_s;
  return -2 * time_s * (C0_M_S * a ** 3 - a ** 4 / 4) / (27 * GRAVITY_M_S2);
}

function analyticBin(x0: number, x1: number, time_s: number) {
  if (!(x1 > x0)) throw new RangeError("analytic bin must have positive width");
  if (!(time_s > 0)) {
    const wet = Math.max(0, Math.min(x1, 0) - x0);
    return { depth_m: HEAD_M * wet / (x1 - x0), discharge_m2_s: 0 };
  }
  const rear = -C0_M_S * time_s;
  const tip = 2 * C0_M_S * time_s;
  let depthIntegral = HEAD_M * Math.max(0, Math.min(x1, rear) - x0);
  let dischargeIntegral = 0;
  const fan0 = Math.max(x0, rear), fan1 = Math.min(x1, tip);
  if (fan1 > fan0) {
    depthIntegral += fanDepthPrimitive(fan1, time_s) - fanDepthPrimitive(fan0, time_s);
    dischargeIntegral += fanDischargePrimitive(fan1, time_s)
      - fanDischargePrimitive(fan0, time_s);
  }
  return { depth_m: depthIntegral / (x1 - x0),
    discharge_m2_s: dischargeIntegral / (x1 - x0) };
}

interface Crossing {
  readonly bracket_m: readonly [number, number] | null;
  readonly interpolated_m: number | null;
  readonly detachedSegments: readonly (readonly [number, number])[];
}

function thresholdCrossing(profile: readonly number[], centers_m: readonly number[],
  threshold_m: number): Crossing {
  let crossingIndex = -1;
  for (let index = 1; index < profile.length; index += 1) {
    if (profile[index - 1]! >= threshold_m && profile[index]! < threshold_m) {
      crossingIndex = index;
      break;
    }
  }
  if (crossingIndex < 0) return { bracket_m: null, interpolated_m: null,
    detachedSegments: [] };
  const left = crossingIndex - 1, right = crossingIndex;
  const denominator = profile[left]! - profile[right]!;
  const alpha = denominator > 0 ? (profile[left]! - threshold_m) / denominator : 0;
  const interpolated_m = centers_m[left]!
    + alpha * (centers_m[right]! - centers_m[left]!);
  const detachedSegments: Array<readonly [number, number]> = [];
  for (let index = right + 1; index < profile.length;) {
    while (index < profile.length && profile[index]! < threshold_m) index += 1;
    if (index >= profile.length) break;
    const first = index;
    while (index + 1 < profile.length && profile[index + 1]! >= threshold_m) index += 1;
    detachedSegments.push([centers_m[first]! - 0.5 * cellSize_m,
      centers_m[index]! + 0.5 * cellSize_m]);
    index += 1;
  }
  return { bracket_m: [centers_m[left]!, centers_m[right]!], interpolated_m,
    detachedSegments };
}

function interpolateProfile(profile: readonly number[], centers_m: readonly number[], x_m: number): number {
  if (x_m <= centers_m[0]!) return profile[0]!;
  if (x_m >= centers_m[centers_m.length - 1]!) return profile[profile.length - 1]!;
  const right = Math.ceil((x_m - centers_m[0]!) / cellSize_m);
  const left = right - 1;
  const alpha = (x_m - centers_m[left]!) / (centers_m[right]! - centers_m[left]!);
  return profile[left]! * (1 - alpha) + profile[right]! * alpha;
}

function analyticProfile(time_s: number, nx: number) {
  const depth_m = new Array<number>(nx), discharge_m2_s = new Array<number>(nx);
  const centers_m = new Array<number>(nx);
  for (let x = 0; x < nx; x += 1) {
    const lower = x * cellSize_m - RESERVOIR_LENGTH_M;
    const value = analyticBin(lower, lower + cellSize_m, time_s);
    centers_m[x] = lower + 0.5 * cellSize_m;
    depth_m[x] = value.depth_m;
    discharge_m2_s[x] = value.discharge_m2_s;
  }
  return { centers_m, depth_m, discharge_m2_s };
}

function analyticSensorArrival(sensorX_m: number, eta: number, nx: number): number {
  const threshold_m = eta * HEAD_M;
  const sample = (time_s: number) => {
    const profile = analyticProfile(time_s, nx);
    return interpolateProfile(profile.depth_m, profile.centers_m, sensorX_m);
  };
  let lower = 0, upper = 2;
  while (sample(upper) < threshold_m) upper *= 2;
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const middle = 0.5 * (lower + upper);
    if (sample(middle) >= threshold_m) upper = middle;
    else lower = middle;
  }
  return 0.5 * (lower + upper);
}

type DiagnosticFields = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>;

function observeFields(fields: DiagnosticFields, nx: number, ny: number, nz: number,
  waterDensity: number) {
  assert.equal(fields.density.length, nx * ny * nz);
  const depth_m = new Array<number>(nx).fill(0);
  const positiveDepth_m = new Array<number>(nx).fill(0);
  const discharge_m2_s = new Array<number>(nx).fill(0);
  const positiveDischarge_m2_s = new Array<number>(nx).fill(0);
  const pressureSupportedPositiveVolume_m3 = new Array<number>(nx).fill(0);
  const positiveVolume_m3 = new Array<number>(nx).fill(0);
  const positiveVerticalMomentum_m4_s = new Array<number>(nx).fill(0);
  let signedVolume_m3 = 0, excludedNegativeVolume_m3 = 0;
  let cellMeanKineticLowerBound_J = 0;
  let cellCentroidPotentialEstimate_J = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) {
      const cell = x + nx * (y + ny * z);
      const rho = fields.density[cell]!;
      const volume_m3 = rho * cellSize_m ** 3;
      signedVolume_m3 += volume_m3;
      depth_m[x] += cellSize_m * rho / nz;
      discharge_m2_s[x] += cellSize_m * rho * fields.velocity[4 * cell]! / nz;
      if (rho < 0) { excludedNegativeVolume_m3 -= volume_m3; continue; }
      positiveVolume_m3[x] += volume_m3;
      positiveVerticalMomentum_m4_s[x] += volume_m3 * fields.velocity[4 * cell + 1]!;
      if (fields.pressureDiagonal[cell]! > 0) {
        pressureSupportedPositiveVolume_m3[x] += volume_m3;
      }
      positiveDepth_m[x] += cellSize_m * rho / nz;
      positiveDischarge_m2_s[x] += cellSize_m * rho * fields.velocity[4 * cell]! / nz;
      const vx = fields.velocity[4 * cell]!, vy = fields.velocity[4 * cell + 1]!;
      const vz = fields.velocity[4 * cell + 2]!;
      cellMeanKineticLowerBound_J += 0.5 * waterDensity * volume_m3
        * (vx * vx + vy * vy + vz * vz);
      cellCentroidPotentialEstimate_J += waterDensity * GRAVITY_M_S2 * volume_m3
        * ((y + 0.5) * cellSize_m);
    }
  let shallowWaterKinetic_J = 0, shallowWaterPotential_J = 0;
  for (let x = 0; x < nx; x += 1) {
    const h = positiveDepth_m[x]!;
    const meanVelocity = h > 0 ? positiveDischarge_m2_s[x]! / h : 0;
    shallowWaterKinetic_J += 0.5 * waterDensity * CHANNEL_WIDTH_M
      * cellSize_m * h * meanVelocity ** 2;
    shallowWaterPotential_J += 0.5 * waterDensity * GRAVITY_M_S2
      * CHANNEL_WIDTH_M * cellSize_m * h ** 2;
  }
  const pressureSupportedPositiveVolumeFraction = positiveVolume_m3.map((volume, x) =>
    volume > 0 ? pressureSupportedPositiveVolume_m3[x]! / volume : null);
  const positiveVolumeWeightedVerticalVelocity_m_s = positiveVolume_m3.map((volume, x) =>
    volume > 0 ? positiveVerticalMomentum_m4_s[x]! / volume : null);
  const cellMinusShallowWaterKinetic_J = cellMeanKineticLowerBound_J
    - shallowWaterKinetic_J;
  const centroidMinusShallowWaterPotential_J = cellCentroidPotentialEstimate_J
    - shallowWaterPotential_J;
  return { depth_m, positiveDepth_m, discharge_m2_s, positiveDischarge_m2_s,
    pressureSupportedPositiveVolumeFraction,
    positiveVolumeWeightedVerticalVelocity_m_s,
    signedVolume_m3, excludedNegativeVolume_m3,
    energy: {
      cellMeanKineticLowerBound_J,
      cellMeanKineticScope: "resolved accepted-cell mean velocity; unresolved subcell variance excluded",
      cellCentroidPotentialEstimate_J,
      cellCentroidPotentialScope: "accepted V at cell-centroid height; not an exact PLIC centroid integral",
      shallowWaterKinetic_J, shallowWaterPotential_J,
      shallowWaterScope: "positive-volume depth and discharge; depth-averaged horizontal velocity",
      cellMinusShallowWaterKinetic_J,
      cellMinusShallowWaterKineticScope: "resolved cell-mean kinetic lower bound minus depth-averaged horizontal shallow-water kinetic energy; includes resolved vertical/transverse motion and within-column horizontal velocity variation",
      centroidMinusShallowWaterPotential_J,
      centroidMinusShallowWaterPotentialScope: "cell-centroid gravitational potential estimate minus hydrostatic shallow-water potential from column depth; limited by unresolved PLIC vertical centroid",
    } };
}

function profileErrors(observed: readonly number[], reference: readonly number[]) {
  let l1 = 0, l2 = 0, maximum = 0;
  for (let index = 0; index < observed.length; index += 1) {
    const error = observed[index]! - reference[index]!;
    l1 += Math.abs(error) * cellSize_m;
    l2 += error * error * cellSize_m;
    maximum = Math.max(maximum, Math.abs(error));
  }
  return { l1Integral: l1, l2IntegralRoot: Math.sqrt(l2), maximumAbsolute: maximum };
}

function analyticEnergy(profile: ReturnType<typeof analyticProfile>, time_s: number,
  waterDensity: number) {
  let binAveragedKinetic_J = 0, binAveragedPotential_J = 0;
  for (let x = 0; x < profile.depth_m.length; x += 1) {
    const h = profile.depth_m[x]!, q = profile.discharge_m2_s[x]!;
    binAveragedKinetic_J += 0.5 * waterDensity * CHANNEL_WIDTH_M * cellSize_m
      * (h > 0 ? q * q / h : 0);
    binAveragedPotential_J += 0.5 * waterDensity * GRAVITY_M_S2
      * CHANNEL_WIDTH_M * cellSize_m * h * h;
  }
  const initial_J = 0.5 * waterDensity * CHANNEL_WIDTH_M * GRAVITY_M_S2
    * HEAD_M ** 2 * RESERVOIR_LENGTH_M;
  const continuumKinetic_J = 0.2 * waterDensity * CHANNEL_WIDTH_M * GRAVITY_M_S2
    * HEAD_M ** 2 * C0_M_S * time_s;
  return { binAveragedKinetic_J, binAveragedPotential_J,
    continuum: { initial_J, kinetic_J: continuumKinetic_J,
      potential_J: initial_J - continuumKinetic_J },
    scope: "unreflected Ritter solution; bin quadrature and continuum totals" };
}

const root = fileURLToPath(new URL("..", import.meta.url));
const sourceFingerprint = await fingerprintSparseCM12RepositorySources(root);
const scene = createSparseGeometricRitterDamBreakScene();
scene.duration_s = duration_s;
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
if (cellSize_m !== scene.voxelDomain.finestCellSize_m) {
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: cellSize_m };
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
}
if (fixedCellWidth !== undefined) scene.fluid.refinementRegions = [{
  id: `ritter-fixed-cell-width-${fixedCellWidth}`,
  rule: "minimum-cell-size", minimumCellSize_cells: fixedCellWidth,
  maximumCellSize_cells: fixedCellWidth,
  min_m: { x: -2, y: 0, z: -0.05 }, max_m: { x: 2, y: 0.2, z: 0.05 },
}];
const dimensions = [scene.container.width_m, scene.container.height_m,
  scene.container.depth_m].map(value => Math.round(value / cellSize_m));
assert.ok(dimensions.every((value, axis) => Math.abs(value * cellSize_m
  - [scene.container.width_m, scene.container.height_m, scene.container.depth_m][axis]!) < 1e-10),
"--cell-size must divide every scene dimension exactly");
const [nx, ny, nz] = dimensions as [number, number, number];
const profile = SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE;
assert.equal(profile.methodId, "adaptive-volume");
assert.equal(adaptiveVolumeMethod.id, "adaptive-volume");
const methodValues = resolveMethodValues(adaptiveVolumeMethod, profile.quality,
  { ...profile.overrides, timeStep: "scene" });
const analyticArrivals = Object.fromEntries(SENSOR_X_M.flatMap(sensor => THRESHOLDS.map(eta => {
  const observedBin_s = analyticSensorArrival(sensor, eta, nx);
  return [`x${sensor}:eta${eta}`, {
    sensorX_m: sensor, eta, observedBin_s,
    pointSolution_s: sensor / ((2 - 3 * Math.sqrt(eta)) * C0_M_S),
    classification: sensor <= 0.8 && eta >= 0.05 ? "primary-pre-impact"
      : observedBin_s < TIP_WALL_TIME_S ? "pre-impact-tail-sensitivity"
        : "post-impact-unbounded-reference-only",
  }];
})));

const report: Record<string, unknown> & { frames: Array<Record<string, unknown>> } = {
  probe: "adaptive-volume-dam-front-dawn", status: "initializing", passed: false,
  passScope: "run completion, accepted volume/dynamics validity, diagnostic coverage and source stability; no analytic-fidelity acceptance threshold",
  implementation: {
    methodId: adaptiveVolumeMethod.id,
    solver: "lib/methods/adaptive-volume/webgpu-adaptive-mass-solver.ts",
    observation: "accepted density V/full-cell-volume from readDiagnosticFields(true)",
    sourceFingerprint,
    sourceFingerprintInterpretation: "scope includes lib, tools, and build manifests; a changed hash can reflect diagnostic-only probe/helper edits as well as production changes",
    commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
  },
  configuration: { dt_s, duration_s, cellSize_m, dimensions, fixedCellWidth, timeout_ms,
    topologyMode: fixedCellWidth === undefined ? "production-adaptive" : "fixed-width-frozen-diagnostic",
    stageEnergyEnabled, methodProfile: profile, resolvedMethodValues: methodValues },
  reference: { model: "Ritter inviscid hydrostatic shallow-water dry-bed release",
    head_m: HEAD_M, channelWidth_m: CHANNEL_WIDTH_M, reservoirLength_m: RESERVOIR_LENGTH_M,
    downstreamLength_m: DOWNSTREAM_LENGTH_M, gravity_m_s2: GRAVITY_M_S2,
    c0_m_s: C0_M_S, initialVolume_m3: INITIAL_VOLUME_M3,
    tipWallTime_s: TIP_WALL_TIME_S, analyticValidity: `time_s < ${TIP_WALL_TIME_S}`,
    thresholds: THRESHOLDS, sensorsX_m: SENSOR_X_M, analyticArrivals },
  frames: [], arrivals: {},
};

async function checkpoint(): Promise<void> {
  if (!outputPath) return;
  const destination = resolve(outputPath), temporary = `${destination}.tmp-${process.pid}`;
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(temporary, `${JSON.stringify(report, null, 2)}\n`);
  await rename(temporary, destination);
}

function heartbeat(label: string, details: () => Record<string, unknown>): () => void {
  const started = performance.now();
  console.error(JSON.stringify({ probe: report.probe, phase: label, elapsed_s: 0, ...details() }));
  const timer = setInterval(() => console.error(JSON.stringify({ probe: report.probe,
    phase: label, elapsed_s: (performance.now() - started) / 1000, ...details() })), 15_000);
  timer.unref();
  return () => clearInterval(timer);
}

async function waitWithHeartbeat<T>(label: string, promise: Promise<T>,
  details: () => Record<string, unknown>): Promise<T> {
  const stop = heartbeat(label, details);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([promise, new Promise<T>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error(`${label} exceeded ${timeout_ms} ms`)), timeout_ms);
    })]);
  } finally { if (timeout) clearTimeout(timeout); stop(); }
}

type Arrival = { previousTime_s: number; previousDepth_m: number;
  bracket_s?: readonly [number, number]; interpolated_s?: number };
const arrivals = new Map<string, Arrival>();
for (const sensor of SENSOR_X_M) for (const eta of THRESHOLDS) arrivals.set(
  `x${sensor}:eta${eta}`, { previousTime_s: 0, previousDepth_m: 0 });

let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
let activeStageObserver: Awaited<ReturnType<typeof createGeometricDamStageEnergy>> | undefined;
let anyFailure = false;
const validationErrors: string[] = [];
await checkpoint();
await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-adaptive-volume-dam-front-dawn.ts");
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
  device = managedGPUDevice(rawDevice, {
    requireWorkerRealm: false,
    maximumConcurrentBundles: 1,
  });
  device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));
  const stopConstruction = heartbeat("creating-solver", () => ({
    compilation: device ? gpuCompilationManagerFor(device).snapshot() : undefined,
  }));
  try {
    solver = await waitWithHeartbeat("solver-construction",
      adaptiveVolumeMethod.createSolverAsync!(device, scene, profile.quality,
        methodValues, undefined, () => {}) as Promise<WebGPUAdaptiveMassSolver>,
      () => ({ compilation: device ? gpuCompilationManagerFor(device).snapshot() : undefined }));
    await solver.waitForSimulationReady();
    await solver.waitForTopologyReady();
  } finally { stopConstruction(); }
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
  if (fixedCellWidth !== undefined) solver.setTopologyFrozen(true);

  const waterDensity = scene.fluid.density_kg_m3;
  const initialFields = await solver.readDiagnosticFields(true);
  const initialPhysical = await solver.readAcceptedGeometricVolumeQA();
  const initialObservation = observeFields(initialFields, nx, ny, nz, waterDensity);
  const initialAgreement = initialObservation.signedVolume_m3 - initialPhysical.volume_m3;
  const initialTolerance = 16 * F32_EPSILON * Math.max(INITIAL_VOLUME_M3,
    Math.abs(initialObservation.signedVolume_m3), Math.abs(initialPhysical.volume_m3));
  assert.equal(initialPhysical.invalidCells, 0, "initial accepted volume bounds");
  assert.equal(initialPhysical.nonfiniteDynamicsCells, 0, "initial accepted dynamics finite");
  assert.equal(initialPhysical.outsideAuthoredVolumeFine3, 0, "initial volume outside authored flume");
  assert.ok(Math.abs(initialAgreement) <= initialTolerance, "initial dense/global accepted volume mismatch");
  assert.ok(Math.abs(initialPhysical.volume_m3 - INITIAL_VOLUME_M3) <= initialTolerance,
    "initial accepted volume differs from authored 0.024 m3");
  report.initial = { physical: initialPhysical, observation: initialObservation,
    denseMinusGlobalVolume_m3: initialAgreement, agreementTolerance_m3: initialTolerance };
  report.status = "running";
  await checkpoint();

  let acceptedTime_s = 0, frameIndex = 0;
  while (acceptedTime_s < duration_s - 1e-10) {
    const targetTime_s = Math.min(duration_s, acceptedTime_s + dt_s);
    const admissionStarted = performance.now();
    const stopAdmission = heartbeat("frame-admission", () => ({ frameIndex: frameIndex + 1,
      acceptedTime_s, targetTime_s, framePending: solver?.framePending,
      topologyGenerationPending: solver?.info.topologyGenerationPending,
      compilation: device ? gpuCompilationManagerFor(device).snapshot() : undefined }));
    try {
      for (;;) {
        assert.ok(performance.now() - admissionStarted < timeout_ms,
          `frame-admission exceeded ${timeout_ms} ms`);
        await solver.waitForTopologyReady();
        if (stageEnergyEnabled) {
          activeStageObserver?.dispose();
          activeStageObserver = await createGeometricDamStageEnergy(device, solver, cellSize_m, waterDensity);
          activeStageObserver.arm();
        }
        if (solver.advanceTo(targetTime_s, [])) break;
        activeStageObserver?.dispose(); activeStageObserver = undefined;
        await new Promise<void>(done => setImmediate(done));
      }
    } finally { stopAdmission(); }
    await waitWithHeartbeat("frame-completion", solver.awaitFrameCompletion(), () => ({
      frameIndex: frameIndex + 1, acceptedTime_s, targetTime_s,
      framePending: solver?.framePending,
    }));
    await device.queue.onSubmittedWorkDone();
    const stageEnergy = activeStageObserver ? await activeStageObserver.read() : undefined;
    activeStageObserver?.dispose(); activeStageObserver = undefined;
    const fields = await solver.readDiagnosticFields(true);
    const physical = await solver.readAcceptedGeometricVolumeQA();
    const transport = await solver.readGeometricVolumeTransportReceiptQA();
    const stats = await solver.readStats();
    const activity = await solver.readGPUActivityPolicy();
    assert.ok(stats.completedTime_s !== undefined, "solver omitted completed accepted time");
    acceptedTime_s = stats.completedTime_s;
    frameIndex += 1;
    const observed = observeFields(fields, nx, ny, nz, waterDensity);
    const reference = analyticProfile(acceptedTime_s, nx);
    const denseMinusGlobalVolume_m3 = observed.signedVolume_m3 - physical.volume_m3;
    const agreementTolerance_m3 = 16 * F32_EPSILON * Math.max(INITIAL_VOLUME_M3,
      Math.abs(observed.signedVolume_m3), Math.abs(physical.volume_m3));
    assert.equal(physical.invalidCells, 0, "accepted volume bounds");
    assert.equal(physical.nonfiniteDynamicsCells, 0, "accepted dynamics finite");
    assert.equal(physical.outsideAuthoredVolumeFine3, 0, "accepted volume outside authored flume");
    assert.ok(Math.abs(denseMinusGlobalVolume_m3) <= agreementTolerance_m3,
      "dense/global accepted volume mismatch");
    assert.ok(Number.isFinite(acceptedTime_s) && acceptedTime_s >= targetTime_s - 1e-9,
      "solver did not publish requested accepted time");
    assert.deepEqual(validationErrors, []);
    const observedFronts = Object.fromEntries(THRESHOLDS.map(eta => [String(eta),
      thresholdCrossing(observed.depth_m, reference.centers_m, eta * HEAD_M)]));
    const analyticFronts = Object.fromEntries(THRESHOLDS.map(eta => [String(eta),
      thresholdCrossing(reference.depth_m, reference.centers_m, eta * HEAD_M)]));
    const sensors = Object.fromEntries(SENSOR_X_M.map(sensor => [String(sensor), {
      sensorX_m: sensor,
      sampleKind: sensor === DOWNSTREAM_LENGTH_M ? "downstream-wall-adjacent-column" : "linear-between-column-centres",
      observedDepth_m: interpolateProfile(observed.depth_m, reference.centers_m, sensor),
      observedDischarge_m2_s: interpolateProfile(observed.discharge_m2_s, reference.centers_m, sensor),
      analyticDepth_m: interpolateProfile(reference.depth_m, reference.centers_m, sensor),
      analyticDischarge_m2_s: interpolateProfile(reference.discharge_m2_s, reference.centers_m, sensor),
    }]));
    for (const sensor of SENSOR_X_M) for (const eta of THRESHOLDS) {
      const key = `x${sensor}:eta${eta}`, arrival = arrivals.get(key)!;
      const depth = (sensors[String(sensor)] as { observedDepth_m: number }).observedDepth_m;
      const threshold = eta * HEAD_M;
      if (!arrival.bracket_s && arrival.previousDepth_m < threshold && depth >= threshold) {
        const alpha = (threshold - arrival.previousDepth_m) / (depth - arrival.previousDepth_m);
        arrival.bracket_s = [arrival.previousTime_s, acceptedTime_s];
        arrival.interpolated_s = arrival.previousTime_s + alpha * (acceptedTime_s - arrival.previousTime_s);
      }
      arrival.previousTime_s = acceptedTime_s; arrival.previousDepth_m = depth;
    }
    const resolutionDistribution = Object.fromEntries([...new Set(activity.bricks
      .filter(brick => brick.active).map(brick => brick.acceptedResolution))].sort((a, b) => a - b)
      .map(resolution => [String(resolution), activity.bricks.filter(brick => brick.active
        && brick.acceptedResolution === resolution).length]));
    const frame = {
      frameIndex, requestedTime_s: targetTime_s, acceptedTime_s,
      admissionWall_ms: performance.now() - admissionStarted,
      analyticApplicability: acceptedTime_s < TIP_WALL_TIME_S
        ? "unreflected-pre-impact" : "post-impact-diagnostic-only",
      physical, denseSignedVolume_m3: observed.signedVolume_m3,
      denseMinusGlobalVolume_m3, agreementTolerance_m3,
      excludedNegativeVolume_m3: observed.excludedNegativeVolume_m3,
      columns: { xCenterReference_m: reference.centers_m,
        acceptedDepth_m: observed.depth_m,
        acceptedDischarge_m2_s: observed.discharge_m2_s,
        positiveWeightedDepth_m: observed.positiveDepth_m,
        positiveWeightedDischarge_m2_s: observed.positiveDischarge_m2_s,
        pressureSupportedPositiveVolumeFraction:
          observed.pressureSupportedPositiveVolumeFraction,
        pressureSupportedPositiveVolumeFractionScope:
          "positive accepted V whose raw pressureDiagonal is greater than zero, divided by all positive accepted V in the column; null for a dry column",
        positiveVolumeWeightedVerticalVelocity_m_s:
          observed.positiveVolumeWeightedVerticalVelocity_m_s,
        positiveVolumeWeightedVerticalVelocityScope:
          "accepted positive-V weighted cell-mean vertical velocity; null for a dry column",
        analyticBinDepth_m: reference.depth_m,
        analyticBinDischarge_m2_s: reference.discharge_m2_s },
      profileErrors: { depth: profileErrors(observed.depth_m, reference.depth_m),
        discharge: profileErrors(observed.discharge_m2_s, reference.discharge_m2_s) },
      fronts: { observed: observedFronts, analyticSameBinObservation: analyticFronts },
      sensors, energy: { observed: observed.energy,
        analytic: analyticEnergy(reference, acceptedTime_s, waterDensity), stageEnergy },
      transport, resolutionDistribution,
      stats: { encodedSteps: stats.encodedSteps,
        topologyGenerationCount: stats.topologyGenerationCount,
        submittedTime_s: stats.submittedTime_s, completedTime_s: stats.completedTime_s },
    };
    report.frames.push(frame);
    report.arrivals = Object.fromEntries([...arrivals].map(([key, value]) => [key, {
      sensorX_m: Number(key.slice(1, key.indexOf(":"))),
      eta: Number(key.slice(key.indexOf("eta") + 3)),
      observedBracket_s: value.bracket_s ?? null,
      observedInterpolated_s: value.interpolated_s ?? null,
      observedArrivalApplicability: value.interpolated_s === undefined ? "not-observed"
        : value.interpolated_s < TIP_WALL_TIME_S ? "unreflected-pre-impact"
          : "post-impact-diagnostic-only",
      referenceComparisonValid: value.interpolated_s !== undefined
        && value.interpolated_s < TIP_WALL_TIME_S
        && (analyticArrivals[key] as { classification: string }).classification
          !== "post-impact-unbounded-reference-only",
      analytic: analyticArrivals[key],
    }]));
    report.lastAcceptedTime_s = acceptedTime_s;
    await checkpoint();
    console.error(JSON.stringify({ probe: report.probe, frameIndex, acceptedTime_s,
      requestedTime_s: targetTime_s, topologyGenerationCount: stats.topologyGenerationCount }));
  }
  report.status = "complete";
  report.validationErrors = validationErrors;
  report.passed = true;
} catch (error) {
  anyFailure = true;
  report.status = "failed";
  report.failure = error instanceof Error ? error.stack : String(error);
  report.validationErrors = validationErrors;
  if (solver) {
    try { report.failureTransport = await solver.readGeometricVolumeTransportReceiptQA(); }
    catch { /* retain the original failure */ }
  }
} finally {
  activeStageObserver?.dispose(); activeStageObserver = undefined;
  if (device) {
    const manager = gpuCompilationManagerFor(device);
    try {
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
      solver?.destroy(); solver = undefined;
      invalidateGPUCompilationManager(device, "dam-front probe complete");
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
    } finally { solver?.destroy(); device.destroy(); }
  }
  await releaseWebGPUExclusiveLock();
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(root);
  report.sourceUnchanged = (report.sourceFingerprintAfter as { sha256: string }).sha256
    === sourceFingerprint.sha256;
  report.passed = report.passed === true && report.sourceUnchanged === true && !anyFailure;
  await checkpoint();
}

const output = JSON.stringify(report, null, 2);
console.log(output);
assert.equal((report.sourceFingerprintAfter as { sha256: string }).sha256,
  sourceFingerprint.sha256, "source changed during benchmark");
if (anyFailure) process.exitCode = 1;
