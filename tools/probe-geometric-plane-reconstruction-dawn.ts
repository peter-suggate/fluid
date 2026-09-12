/** Production Sparse Geometric interface reconstruction against exact plane volumes. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createGeometricUniformTranslationScene } from
  "../lib/core/geometric-translation-scene";
import { fluidExecutionDeviceFeatures } from "../lib/core/gpu-startup";
import { resolveMethodValues } from "../lib/core/method-contract";
import { SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE } from "../lib/core/scenes";
import { writeGPUBufferView } from "../lib/core/webgpu-buffer-upload";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager, managedGPUDevice } from
  "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod as adaptiveVolumeMethod } from
  "../lib/methods/adaptive-volume/method";
import { rdfCellCentreValue, type RdfPlaneSample3 } from
  "../lib/methods/adaptive-volume/geometric-rdf-reference";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { fingerprintSparseCM12RepositorySources } from
  "./sparse-cm12-source-content-fingerprint";

const argument = (name: string, fallback = "") => process.argv.slice(2)
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (process.argv.includes("--help")) {
  console.log(`Sparse Geometric production interface reconstruction probe
node --import tsx tools/probe-geometric-plane-reconstruction-dawn.ts [options]
  --dimension=2       Run z-extruded 2D planes (2 or 3)
  --out=PATH          Atomic JSON receipt

The probe seeds exact CPU-integrated plane fractions into both production
scalar banks, advances zero-flow physics through pressure-topology, and reads
the production published interface cache at that stage boundary.`);
  process.exit(0);
}

const dimension = Number(argument("dimension", "2"));
assert.ok(dimension === 2 || dimension === 3, "--dimension must be 2 or 3");
const outputPath = resolve(argument("out", argument("output",
  `artifacts/analytic-motion/geometric-plane-reconstruction-${dimension}d.json`)));
const DT_S = 1e-6;
const DIMENSIONS = [16, 8, 8] as const;
const MAX_ANGLE_ERROR_DEG = 0.1;
const MAX_CENTRAL_VOLUME_MISMATCH = 1e-5;
const PARTIAL_EPSILON = 1e-5;

type Vec3 = readonly [number, number, number];
interface PlaneCase {
  readonly id: string;
  readonly normal: Vec3;
  readonly anchorFine: Vec3;
}
interface AuthoredCell {
  readonly cellId: number;
  readonly centerFine: Vec3;
  readonly widthsFine: Vec3;
}

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(...value);
  assert.ok(length > 0);
  return [value[0] / length, value[1] / length, value[2] / length];
}

const cases: readonly PlaneCase[] = dimension === 2 ? [
  { id: "axis-x-offset", normal: [1, 0, 0], anchorFine: [8.25, 4, 4] },
  { id: "slope-2-1", normal: [2, 1, 0], anchorFine: [8, 4, 4] },
  { id: "slope-2-1-positive-offset", normal: [2, 1, 0], anchorFine: [8.3, 4.15, 4] },
  { id: "slope-1-2", normal: [1, 2, 0], anchorFine: [7.8, 4.2, 4] },
  { id: "diagonal", normal: [1, 1, 0], anchorFine: [8, 4, 4] },
  { id: "diagonal-negative-offset", normal: [1, 1, 0], anchorFine: [7.7, 3.9, 4] },
  { id: "signed-x", normal: [-2, 1, 0], anchorFine: [8.1, 3.85, 4] },
  { id: "signed-xy", normal: [-1, -1, 0], anchorFine: [7.9, 4.1, 4] },
] : [
  { id: "slope-2-1-1", normal: [2, 1, 1], anchorFine: [8, 4, 4] },
  { id: "slope-2-1-1-positive-offset", normal: [2, 1, 1], anchorFine: [8.25, 4.1, 4.15] },
  { id: "slope-1-2-3", normal: [1, 2, 3], anchorFine: [7.85, 4.1, 4.2] },
  { id: "signed-x", normal: [-2, 1, 1], anchorFine: [8.15, 3.9, 4.1] },
  { id: "signed-y", normal: [1, -2, 3], anchorFine: [7.9, 4.2, 3.85] },
  { id: "signed-xyz", normal: [-1, -2, -3], anchorFine: [8.1, 3.85, 4.15] },
];

/** Exact CDF of a weighted sum of independent unit uniforms. This is an
 * independent inclusion-exclusion integral of the box/halfspace overlap. */
function boxHalfspaceFraction(normal: Vec3, offsetFine: number, widthsFine: Vec3): number {
  const spans = normal.map((value, axis) => Math.abs(value) * widthsFine[axis]!)
    .filter(value => value > 1e-14);
  if (spans.length === 0) return Number(offsetFine >= 0);
  const target = offsetFine + 0.5 * spans.reduce((sum, value) => sum + value, 0);
  const factorial = spans.length === 1 ? 1 : spans.length === 2 ? 2 : 6;
  const denominator = factorial * spans.reduce((product, value) => product * value, 1);
  let numerator = 0;
  for (let mask = 0; mask < 1 << spans.length; mask += 1) {
    let shifted = target, bits = 0;
    for (let axis = 0; axis < spans.length; axis += 1) if ((mask & (1 << axis)) !== 0) {
      shifted -= spans[axis]!; bits += 1;
    }
    numerator += (bits & 1 ? -1 : 1) * Math.max(0, shifted) ** spans.length;
  }
  return Math.max(0, Math.min(1, numerator / denominator));
}

function authoredCellsForActivity(solver: WebGPUAdaptiveMassSolver,
  activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>): AuthoredCell[] {
  const source = solver.fieldSnapshotSourceForQA;
  const words = source.templateWords;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const cells: AuthoredCell[] = [];
  for (const brick of activity.bricks) {
    if (!brick.active || brick.leafId >= words[13]!) continue;
    assert.equal(brick.acceptedResolution, 8, "reconstruction probe requires all-fine leaves");
    const range = words[11]! + 2 * (4 * brick.leafId + 3);
    for (let cellId = words[range]!; cellId < words[range]! + words[range + 1]!; cellId += 1) {
      const at = words[6]! + 8 * cellId;
      cells.push({ cellId,
        centerFine: [floats[at]!, floats[at + 1]!, floats[at + 2]!],
        widthsFine: [floats[at + 4]!, floats[at + 5]!, floats[at + 6]!] });
    }
  }
  cells.sort((left, right) => left.cellId - right.cellId);
  assert.equal(new Set(cells.map(cell => cell.cellId)).size, cells.length);
  assert.equal(cells.length, DIMENSIONS[0] * DIMENSIONS[1] * DIMENSIONS[2],
    "full all-fine authored lattice must remain active");
  return cells;
}

function angleDegrees(left: Vec3, right: Vec3): number {
  const a = normalize(left), b = normalize(right);
  const cosine = Math.max(-1, Math.min(1,
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  return Math.acos(cosine) * 180 / Math.PI;
}

function solveSymmetric3(matrix: readonly [number, number, number, number, number, number],
  rhs: Vec3): Vec3 {
  const [mxx, mxy, mxz, myy, myz, mzz] = matrix;
  const determinant = mxx * (myy * mzz - myz * myz)
    - mxy * (mxy * mzz - myz * mxz) + mxz * (mxy * myz - myy * mxz);
  if (Math.abs(determinant) <= 1e-10) return [
    mxx > 1e-10 ? rhs[0] / mxx : 0,
    myy > 1e-10 ? rhs[1] / myy : 0,
    mzz > 1e-10 ? rhs[2] / mzz : 0,
  ];
  return [
    (rhs[0] * (myy * mzz - myz * myz) - mxy * (rhs[1] * mzz - myz * rhs[2])
      + mxz * (rhs[1] * myz - myy * rhs[2])) / determinant,
    (mxx * (rhs[1] * mzz - myz * rhs[2]) - rhs[0] * (mxy * mzz - myz * mxz)
      + mxz * (mxy * rhs[2] - rhs[1] * mxz)) / determinant,
    (mxx * (myy * rhs[2] - rhs[1] * myz) - mxy * (mxy * rhs[2] - rhs[1] * mxz)
      + rhs[0] * (mxy * myz - myy * mxz)) / determinant,
  ];
}

const root = fileURLToPath(new URL("..", import.meta.url));
const sourceFingerprint = await fingerprintSparseCM12RepositorySources(root);
const profile = SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE;
const values = resolveMethodValues(adaptiveVolumeMethod, profile.quality,
  { ...profile.overrides, timeStep: "scene" });
const scene = createGeometricUniformTranslationScene({ speed_m_s: 1, dt_s: DT_S,
  duration_s: DT_S * cases.length });
scene.sceneId = `geometric-plane-reconstruction-${dimension}d`;
scene.container.fillFraction = 1;
scene.fluid.initialCondition = "tank-fill";
delete scene.fluid.initialDamBreakOrigin_m;
delete scene.fluid.initialDamBreakDimensions_m;
scene.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };

const report: Record<string, unknown> & { cases: Array<Record<string, unknown>> } = {
  probe: "geometric-plane-reconstruction-dawn", status: "initializing", passed: false,
  dimension, stage: "presentation-publication",
  method: { methodId: adaptiveVolumeMethod.id, quality: profile.quality,
    profile, resolvedValues: values },
  criteria: { maximumNormalAngleError_deg: MAX_ANGLE_ERROR_DEG,
    maximumCentralVolumeMismatch: MAX_CENTRAL_VOLUME_MISMATCH,
    origin: "predeclared analytic planar reconstruction requirements" },
  integrationOracle: "double-precision inclusion-exclusion CDF of a weighted sum of uniforms",
  sourceFingerprint,
  commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
  cases: [],
};

async function checkpoint() {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(report, null, 2)}\n`);
  await rename(temporary, outputPath);
}

let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
let runtimeFailure = false, analyticFailure = false;
const validationErrors: string[] = [];
await checkpoint();
await acquireWebGPUExclusiveLock("dawn-probe", "geometric-plane-reconstruction");
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
  solver.setTopologyFrozen(true);
  const activity = await solver.readGPUActivityPolicy();
  const cells = authoredCellsForActivity(solver, activity);
  const source = solver.fieldSnapshotSourceForQA;
  const republishRdf = async () => {
    solver!.applyRuntimeValues({ ...values, presentationSurface: "plic",
      presentationColumnHeight: "off" });
    await solver!.assertSimulationHealthy();
    solver!.applyRuntimeValues({ ...values, presentationSurface: "rdf",
      presentationColumnHeight: "off" });
    await solver!.assertSimulationHealthy();
  };
  const encodeCapture = (capture: GPUBuffer) => {
    const encoder = device!.createCommandEncoder({ label: "RDF presentation cache capture" });
    encoder.copyBufferToBuffer(source.state,
      4 * source.layout.geometricInterfacePlanes, capture, 0, 16 * source.cellCapacity);
    encoder.copyBufferToBuffer(source.state, 4 * source.layout.densityA,
      capture, 16 * source.cellCapacity, 4 * source.cellCapacity);
    encoder.copyBufferToBuffer(source.state, 4 * source.layout.densityB,
      capture, 20 * source.cellCapacity, 4 * source.cellCapacity);
    encoder.copyBufferToBuffer(source.state, 4 * source.layout.geometricInterfaceRdf,
      capture, 24 * source.cellCapacity, 16 * source.cellCapacity);
    device!.queue.submit([encoder.finish()]);
  };
  const expectedDensity = new Float32Array(source.cellCapacity);
  report.topology = { dimensions: DIMENSIONS, cellCapacity: source.cellCapacity,
    activeAuthoredCells: cells.length,
    topologyGenerationCount: solver.info.topologyGenerationCount };
  report.status = "running";
  await checkpoint();

  for (let caseIndex = 0; caseIndex < cases.length; caseIndex += 1) {
    const specification = cases[caseIndex]!;
    const normal = normalize(specification.normal);
    const globalOffsetFine = normal[0] * specification.anchorFine[0]
      + normal[1] * specification.anchorFine[1]
      + normal[2] * specification.anchorFine[2];
    expectedDensity.fill(0);
    for (const cell of cells) {
      const localOffset = globalOffsetFine - (normal[0] * cell.centerFine[0]
        + normal[1] * cell.centerFine[1] + normal[2] * cell.centerFine[2]);
      expectedDensity[cell.cellId] = boxHalfspaceFraction(normal, localOffset, cell.widthsFine);
    }
    writeGPUBufferView(device.queue, source.state, 4 * source.layout.densityA, expectedDensity);
    writeGPUBufferView(device.queue, source.state, 4 * source.layout.densityB, expectedDensity);

    const floatCount = 10 * source.cellCapacity;
    const capture = device.createBuffer({ label: `Plane reconstruction ${specification.id}`,
      size: 4 * floatCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    await republishRdf();
    encodeCapture(capture);

    try {
      await capture.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(capture.getMappedRange());
      const densityAAt = 4 * source.cellCapacity;
      const densityBAt = 5 * source.cellCapacity;
      const rdfAt = 6 * source.cellCapacity;
      let densityBankMaximumError = 0;
      for (const cell of cells) densityBankMaximumError = Math.max(densityBankMaximumError,
        Math.abs(values[densityAAt + cell.cellId]! - expectedDensity[cell.cellId]!),
        Math.abs(values[densityBAt + cell.cellId]! - expectedDensity[cell.cellId]!));
      const partialCells = cells.filter(cell => {
        const fill = expectedDensity[cell.cellId]!;
        const [x, y, z] = cell.centerFine.map(value => Math.floor(value));
        return fill > PARTIAL_EPSILON && fill < 1 - PARTIAL_EPSILON
          && x > 0 && x < DIMENSIONS[0] - 1 && y > 0 && y < DIMENSIONS[1] - 1
          && z > 0 && z < DIMENSIONS[2] - 1 && (dimension === 3 || z === 4);
      });
      assert.ok(partialCells.length > 0, `${specification.id} has no interior partial cells`);
      let maximumAngleError_deg = 0, maximumCentralVolumeMismatch = 0;
      let maximumOffsetErrorFine = 0, maximumNeighborVolumePredictionError = 0;
      let maximumRdfValueErrorFine = 0, maximumRdfGradientError = 0;
      const cellAt = new Map(cells.map(cell => [cell.centerFine.join(":"), cell]));
      const cellReceipts = partialCells.map(cell => {
        const at = 4 * cell.cellId;
        const reconstructedNormal = [values[at]!, values[at + 1]!, values[at + 2]!] as Vec3;
        const reconstructedOffsetFine = values[at + 3]!;
        const valid = reconstructedNormal[0] ** 2 + reconstructedNormal[1] ** 2
          + reconstructedNormal[2] ** 2 > 0.5;
        const expectedOffsetFine = globalOffsetFine - (normal[0] * cell.centerFine[0]
          + normal[1] * cell.centerFine[1] + normal[2] * cell.centerFine[2]);
        const angleError_deg = valid ? angleDegrees(reconstructedNormal, normal) : Infinity;
        const reconstructedCentralFraction = valid ? boxHalfspaceFraction(
          reconstructedNormal, reconstructedOffsetFine, cell.widthsFine) : NaN;
        const centralVolumeMismatch = Math.abs(reconstructedCentralFraction
          - expectedDensity[cell.cellId]!);
        const neighbors = cells.filter(other => other.cellId !== cell.cellId
          && Math.abs(other.centerFine[0] - cell.centerFine[0]) <= 1
          && Math.abs(other.centerFine[1] - cell.centerFine[1]) <= 1
          && Math.abs(other.centerFine[2] - cell.centerFine[2]) <= (dimension === 3 ? 1 : 0));
        const neighborPredictions = neighbors.map(neighbor => {
          const neighborOffset = reconstructedOffsetFine
            - reconstructedNormal[0] * (neighbor.centerFine[0] - cell.centerFine[0])
            - reconstructedNormal[1] * (neighbor.centerFine[1] - cell.centerFine[1])
            - reconstructedNormal[2] * (neighbor.centerFine[2] - cell.centerFine[2]);
          const predictedFraction = valid ? boxHalfspaceFraction(
            reconstructedNormal, neighborOffset, neighbor.widthsFine) : NaN;
          return { cellId: neighbor.cellId, centerFine: neighbor.centerFine,
            expectedFraction: expectedDensity[neighbor.cellId]!, predictedFraction,
            absoluteError: Math.abs(predictedFraction - expectedDensity[neighbor.cellId]!) };
        });
        const neighborMaximumError = Math.max(0,
          ...neighborPredictions.map(neighbor => neighbor.absoluteError));
        maximumAngleError_deg = Math.max(maximumAngleError_deg, angleError_deg);
        maximumCentralVolumeMismatch = Math.max(maximumCentralVolumeMismatch,
          centralVolumeMismatch);
        maximumOffsetErrorFine = Math.max(maximumOffsetErrorFine,
          Math.abs(reconstructedOffsetFine - expectedOffsetFine));
        maximumNeighborVolumePredictionError = Math.max(maximumNeighborVolumePredictionError,
          neighborMaximumError);
        // Scheufler/Roenby's point-neighbour stencil contains every cell that
        // shares a vertex with the destination: 26 neighbours on this uniform
        // lattice. The production corner/octant walk deduplicates this set.
        const pointNeighbours = neighbors;
        const rdfSources = [cell, ...pointNeighbours].flatMap(candidate => {
          const candidateAt = 4 * candidate.cellId;
          const candidateNormal = [values[candidateAt]!, values[candidateAt + 1]!,
            values[candidateAt + 2]!] as Vec3;
          if (Math.hypot(...candidateNormal) <= 0.5) return [];
          return [{ center: candidate.centerFine, widths: candidate.widthsFine,
            normal: candidateNormal, offset: values[candidateAt + 3]! } satisfies RdfPlaneSample3];
        });
        const ownSource = rdfSources.find(candidate => candidate.center === cell.centerFine);
        const expectedRdf = rdfCellCentreValue(cell.centerFine, rdfSources, ownSource)
          ?? (0.5 - expectedDensity[cell.cellId]!) * 4;
        const gpuRdf = values[rdfAt + 4 * cell.cellId + 3]!;
        const rdfValueErrorFine = Math.abs(gpuRdf - expectedRdf);
        maximumRdfValueErrorFine = Math.max(maximumRdfValueErrorFine, rdfValueErrorFine);
        const gradientMatrix = [0, 0, 0, 0, 0, 0] as
          [number, number, number, number, number, number];
        const gradientRhs = [0, 0, 0] as [number, number, number];
        for (const other of pointNeighbours) {
          const delta = other.centerFine.map((value, axis) => value - cell.centerFine[axis]!) as
            unknown as Vec3;
          const otherRdf = values[rdfAt + 4 * other.cellId + 3]!;
          const weight = 1 / Math.max(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2, 1e-12);
          gradientMatrix[0] += weight * delta[0] * delta[0];
          gradientMatrix[1] += weight * delta[0] * delta[1];
          gradientMatrix[2] += weight * delta[0] * delta[2];
          gradientMatrix[3] += weight * delta[1] * delta[1];
          gradientMatrix[4] += weight * delta[1] * delta[2];
          gradientMatrix[5] += weight * delta[2] * delta[2];
          for (let axis = 0; axis < 3; axis += 1)
            gradientRhs[axis] += weight * (otherRdf - gpuRdf) * delta[axis]!;
        }
        const expectedGradient = solveSymmetric3(gradientMatrix, gradientRhs);
        const gpuGradient = [values[rdfAt + 4 * cell.cellId]!,
          values[rdfAt + 4 * cell.cellId + 1]!,
          values[rdfAt + 4 * cell.cellId + 2]!] as Vec3;
        const rdfGradientError = Math.hypot(...gpuGradient.map((value, axis) =>
          value - expectedGradient[axis]!));
        maximumRdfGradientError = Math.max(maximumRdfGradientError, rdfGradientError);
        return { cellId: cell.cellId, centerFine: cell.centerFine,
          widthsFine: cell.widthsFine, seededFraction: expectedDensity[cell.cellId]!,
          expectedPlane: { normal, offsetFine: expectedOffsetFine },
          reconstructedPlane: { normal: reconstructedNormal, offsetFine: reconstructedOffsetFine,
            valid }, angleError_deg, reconstructedCentralFraction, centralVolumeMismatch,
          offsetErrorFine: reconstructedOffsetFine - expectedOffsetFine,
          neighborMaximumError, neighborPredictions,
          rdf: { gpuValue: gpuRdf, expectedValue: expectedRdf, rdfValueErrorFine,
            gpuGradient, expectedGradient, rdfGradientError } };
      });
      const checks = { densityBanksUnchanged: densityBankMaximumError === 0,
        normalAngle: maximumAngleError_deg < MAX_ANGLE_ERROR_DEG,
        centralVolumeFit: maximumCentralVolumeMismatch < MAX_CENTRAL_VOLUME_MISMATCH,
        rdfValue: maximumRdfValueErrorFine < 2e-5,
        rdfGradient: maximumRdfGradientError < 2e-5,
        validationClean: validationErrors.length === 0 };
      const passed = Object.values(checks).every(Boolean);
      analyticFailure ||= !passed;
      report.cases.push({ id: specification.id, input: { rawNormal: specification.normal,
        normal, anchorFine: specification.anchorFine, globalOffsetFine },
      partialCellCount: partialCells.length, densityBankMaximumError,
      maximumAngleError_deg, maximumCentralVolumeMismatch, maximumOffsetErrorFine,
      maximumNeighborVolumePredictionError, maximumRdfValueErrorFine,
      maximumRdfGradientError,
      neighborPredictionScope: dimension === 2 ? "same-z 3x3 cell patch" : "3x3x3 cell patch",
      cellReceipts, checks, passed });
    } finally {
      if (capture.mapState === "mapped") capture.unmap();
      capture.destroy();
    }
    await checkpoint();
    console.error(JSON.stringify({ probe: report.probe, case: specification.id,
      passed: report.cases.at(-1)?.passed }));
  }

  // A face-only stencil cannot see this interface: the sole reconstructed
  // plane is one corner away from the destination. The paper point-neighbour
  // stencil must publish that diagonal plane's signed distance instead of the
  // destination cell's fill fallback.
  {
    const byCentre = new Map(cells.map(cell => [cell.centerFine.join(":"), cell]));
    const target = byCentre.get("7.5:3.5:3.5");
    const diagonal = byCentre.get("8.5:4.5:4.5");
    const gradientSupport = byCentre.get("9.5:4.5:4.5");
    assert.ok(target && diagonal && gradientSupport, "diagonal RDF fixture cells are authored");
    expectedDensity.fill(0);
    expectedDensity[diagonal.cellId] = 0.25;
    expectedDensity[gradientSupport.cellId] = 1;
    writeGPUBufferView(device.queue, source.state, 4 * source.layout.densityA, expectedDensity);
    writeGPUBufferView(device.queue, source.state, 4 * source.layout.densityB, expectedDensity);

    const capture = device.createBuffer({ label: "RDF diagonal-only point neighbour",
      size: 40 * source.cellCapacity,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    await republishRdf();
    encodeCapture(capture);
    try {
      await capture.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(capture.getMappedRange());
      const rdfAt = 6 * source.cellCapacity;
      const diagonalAt = 4 * diagonal.cellId;
      const diagonalNormal = [values[diagonalAt]!, values[diagonalAt + 1]!,
        values[diagonalAt + 2]!] as Vec3;
      assert.ok(Math.hypot(...diagonalNormal) > 0.5,
        "the diagonal source must reconstruct a valid plane");
      const diagonalPlane = { center: diagonal.centerFine, widths: diagonal.widthsFine,
        normal: diagonalNormal, offset: values[diagonalAt + 3]! } satisfies RdfPlaneSample3;
      const expectedRdf = rdfCellCentreValue(target.centerFine, [diagonalPlane]);
      assert.notEqual(expectedRdf, null);
      const gpuRdf = values[rdfAt + 4 * target.cellId + 3]!;
      const fillFallback = 2 * Math.min(...target.widthsFine);
      const faceDeltas: readonly Vec3[] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0],
        [0, 1, 0], [0, 0, -1], [0, 0, 1]];
      const validFacePlaneCount = faceDeltas.reduce((count, delta) => {
        const neighbor = byCentre.get(target.centerFine.map((value, axis) =>
          value + delta[axis]!).join(":"));
        if (!neighbor) return count;
        const at = 4 * neighbor.cellId;
        return count + Number(Math.hypot(values[at]!, values[at + 1]!, values[at + 2]!) > 0.5);
      }, 0);
      const absoluteError = Math.abs(gpuRdf - expectedRdf!);
      const checks = { noFacePlane: validFacePlaneCount === 0,
        diagonalContribution: Math.abs(gpuRdf - fillFallback) > 1e-3,
        pointNeighbourValue: absoluteError < 2e-5 };
      const passed = Object.values(checks).every(Boolean);
      analyticFailure ||= !passed;
      report.cases.push({ id: "diagonal-only-point-neighbour",
        target: target.centerFine, diagonal: diagonal.centerFine,
        diagonalPlane, validFacePlaneCount, gpuRdf, expectedRdf, fillFallback,
        absoluteError, checks, passed });
      console.error(JSON.stringify({ probe: report.probe,
        case: "diagonal-only-point-neighbour", passed }));
    } finally {
      if (capture.mapState === "mapped") capture.unmap(); capture.destroy();
    }
    await checkpoint();
  }
  report.status = analyticFailure ? "failed-analytic-criterion" : "complete";
} catch (error) {
  runtimeFailure = true;
  report.status = "failed-runtime";
  report.failure = error instanceof Error ? error.stack : String(error);
} finally {
  solver?.setStageCaptureForQA(undefined);
  if (device) {
    const manager = gpuCompilationManagerFor(device);
    try {
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
      solver?.destroy(); solver = undefined;
      invalidateGPUCompilationManager(device, "plane reconstruction probe complete");
      await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
    } finally { solver?.destroy(); device.destroy(); }
  }
  await releaseWebGPUExclusiveLock();
  report.validationErrors = validationErrors;
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(root);
  report.sourceUnchanged = (report.sourceFingerprintAfter as { sha256: string }).sha256
    === sourceFingerprint.sha256;
  report.passed = !runtimeFailure && !analyticFailure && report.sourceUnchanged === true;
  await checkpoint();
}

console.log(JSON.stringify(report, null, 2));
if (report.passed !== true) process.exitCode = 1;
