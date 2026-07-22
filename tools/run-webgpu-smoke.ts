import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { tallCellMethod } from "../lib/methods/tall-cell";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
import type { OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";
import { uniformMethod } from "../lib/methods/uniform";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { octreeMethod } from "../lib/methods/octree";
import { maximumFluidScale } from "../lib/quadtree-tall-cell-grid";
import { damBreakFractions } from "../lib/initial-fluid";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "../lib/rigid-body";
import type { SceneDescription } from "../lib/model";
import { createSingleTallCellProbeControlLayout, createSingleTallCellProbeLayout, createTallCellLayout, tallCellSettings, type SingleTallCellProbeOptions } from "../lib/tall-cell-grid";
import { WebGPUEulerianSolver, type GPUEulerianInfo, type GPUQuality } from "../lib/webgpu-eulerian";
import { summarizeDriftOscillation } from "../lib/tall-cell-diagnostics";
import { VOXEL_MATERIAL_IDS, voxelMaterial } from "../lib/voxel-scene";
import { SPARSE_VOXEL_DEBUG_RECORD_STRIDE, SparseVoxelDebugRenderer, type SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { activeCubeCapacity, RasterWaterPipeline, surfaceVertexCapacity,
  type WaterSurfaceGeometrySource } from "../lib/webgpu-water-pipeline";
import { createGlobalFineLevelSetConsumerSource, createUnifiedOctreeConsumerSource } from "../lib/octree-consumer-sampling";
import { unpackOctreeFaceBandControl } from "../lib/webgpu-octree-face-fast-march";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";
import { environmentIndex } from "../lib/environments";
import { OCTREE_POWER_FACE_RECORD_BYTES, OCTREE_POWER_INVALID_ROW } from "../lib/octree-power-operator";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, sceneHasTerrain } from "../lib/terrain";
import { compactOctreeFieldEvidenceIsAcceptable, compactOctreePublicationHeaderEvidence,
  reconstructCompactOctreeOccupancyField,
  type CompactOctreeFieldEvidence } from "./webgpu-smoke-compact-field";
import { decodeOctreeMGPCGDiagnostics, octreeMGPCGDiagnosticsAreAcceptable,
  type OctreeMGPCGDiagnostics } from "./webgpu-smoke-pressure";
import { compactPowerFaceIntegratedFlux, compactPowerFaceMetricKineticEnergy } from "./webgpu-smoke-power-diagnostics";
import { OCTREE_POWER_VELOCITY_VALID } from "../lib/webgpu-octree-power-velocity";
import { compareVelocityFields, DAM_BREAK_VELOCITY_PARITY_LIMITS, rasterizeCompactPowerCellVelocities,
  velocityParityFailures, type CompactVelocityRaster, type VelocityParityMetrics } from "./webgpu-smoke-velocity-parity";
import { narrowVerticalSlitMetrics, type NarrowVerticalSlitMetrics } from "./raster-slit-metrics";
import { viewportFailureIndicator } from "../lib/viewport-failure-diagnostics";
import {
  compareScalarFields,
  compareSingleTallCellNeighborhood,
  createSmokeScenario,
  isSmokeScenarioId,
  minimumOceanFarHalfDisturbanceCells,
  smokeScenarioIds,
  summarizeScalarField,
  summarizeTallCellActivity,
  type ScalarFieldSummary,
  type TallCellActivitySummary,
  type SmokeScenarioId
} from "./webgpu-smoke-scenarios";

// SAFETY (2026-07-20): two browser WebGPU attempts triggered machine-wide
// WindowServer/AGX watchdog failures. Native Dawn and browser WebGPU workloads
// must remain mutually exclusive until that driver fault is localized.
console.error("SAFETY: close every browser WebGPU tab before running Dawn. Never run this smoke and browser GPU validation concurrently.");

const modulePath = process.env.WEBGPU_NODE_MODULE;
const webgpuModule = modulePath ? await import(pathToFileURL(modulePath).href) : await import("webgpu");
const { create, globals } = webgpuModule as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
// Dawn exposes a Web Worker-compatible global in Node. The adaptive solver
// must still select its worker_threads transport here: Dawn's worker wrapper
// does not preserve typed-array inputs used by the topology packer.
Reflect.deleteProperty(globalThis, "Worker");
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const COMPUTE_SENTINEL_WORD = 0x4f43_5452;

/**
 * Prove that this adapter executes and returns compute results before paying
 * the much larger octree construction cost. Shader validation and successful
 * submission alone are insufficient: a poisoned backend has historically
 * accepted both while returning an all-zero readback.
 */
async function assertComputeSentinel(device: GPUDevice): Promise<void> {
  const output = device.createBuffer({
    label: "WebGPU smoke compute sentinel",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "WebGPU smoke compute sentinel readback",
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const shaderModule = device.createShaderModule({
      label: "WebGPU smoke compute sentinel",
      code: `
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn sentinel() { output[0] = 0x4f435452u; }
`,
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "WebGPU smoke compute sentinel",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "sentinel" },
    });
    const bindGroup = device.createBindGroup({
      label: "WebGPU smoke compute sentinel",
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }],
    });
    const encoder = device.createCommandEncoder({ label: "WebGPU smoke compute sentinel" });
    const pass = encoder.beginComputePass({ label: "WebGPU smoke compute sentinel" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const observed = new Uint32Array(readback.getMappedRange())[0];
    readback.unmap();
    if (observed !== COMPUTE_SENTINEL_WORD) {
      throw new Error(`WebGPU compute sentinel returned 0x${observed.toString(16).padStart(8, "0")} instead of 0x${COMPUTE_SENTINEL_WORD.toString(16)}; stop before collecting simulation evidence`);
    }
    console.log(JSON.stringify({ phase: "compute-sentinel", value: observed, passed: true }));
  } finally {
    output.destroy();
    readback.destroy();
  }
}

const availableMethods = [tallCellMethod, quadtreeTallCellMethod, octreeMethod, uniformMethod];
const methodFilter = process.env.FLUID_METHOD?.split(",").map((value) => value.trim()).filter(Boolean);
const methods = availableMethods.filter((method) => !methodFilter || methodFilter.includes(method.id));
if (methods.length === 0 || (methodFilter && methodFilter.length !== methods.length)) throw new Error(`Unknown FLUID_METHOD=${process.env.FLUID_METHOD}; expected a comma list of tall-cell, quadtree-tall-cell, octree, or uniform`);

const qualityValue = process.env.FLUID_QUALITY ?? "balanced";
if (!["balanced", "high", "ultra"].includes(qualityValue)) throw new Error(`Unknown FLUID_QUALITY=${qualityValue}`);
const quality = qualityValue as GPUQuality;
const targetOverride = process.env.FLUID_TARGET_S === undefined ? undefined : Number(process.env.FLUID_TARGET_S);
const maxDtOverride = process.env.FLUID_MAX_DT === undefined ? undefined : Number(process.env.FLUID_MAX_DT);
const exactStepCount = process.env.FLUID_EXPECT_EXACT_STEPS === undefined ? undefined : Number(process.env.FLUID_EXPECT_EXACT_STEPS);
const minimumPeakSpeed_m_s = process.env.FLUID_MIN_PEAK_SPEED_M_S === undefined ? undefined : Number(process.env.FLUID_MIN_PEAK_SPEED_M_S);
const minimumDamSpread_m = process.env.FLUID_MIN_DAM_SPREAD_M === undefined ? undefined : Number(process.env.FLUID_MIN_DAM_SPREAD_M);
if (maxDtOverride !== undefined && (!Number.isFinite(maxDtOverride) || maxDtOverride <= 0)) throw new Error("FLUID_MAX_DT must be positive and finite");
if (exactStepCount !== undefined && (!Number.isInteger(exactStepCount) || exactStepCount < 1)) throw new Error("FLUID_EXPECT_EXACT_STEPS must be a positive integer");
if (minimumPeakSpeed_m_s !== undefined && (!Number.isFinite(minimumPeakSpeed_m_s) || minimumPeakSpeed_m_s <= 0)) throw new Error("FLUID_MIN_PEAK_SPEED_M_S must be positive and finite");
if (minimumDamSpread_m !== undefined && (!Number.isFinite(minimumDamSpread_m) || minimumDamSpread_m <= 0)) throw new Error("FLUID_MIN_DAM_SPREAD_M must be positive and finite");
if (exactStepCount !== undefined && maxDtOverride === undefined) throw new Error("FLUID_EXPECT_EXACT_STEPS requires FLUID_MAX_DT so submitted/completed time is unambiguous");
const reportEvery = Number(process.env.FLUID_REPORT_EVERY ?? 0);
const includeFinalFieldStats = process.env.FLUID_FIELD_STATS !== "0";
/** Timing-only mode keeps solver/control/timestamp readbacks while omitting
 * compact cubic reconstruction and scene quality gates. The reconstruction
 * is not part of simulationWall_ms and can independently reject a measurable
 * run when an upstream publication generation is stale. */
const performanceProfileRequested = process.env.FLUID_PERFORMANCE_PROFILE === "1";
const requireSpatialField = process.env.FLUID_REQUIRE_SPATIAL_FIELD === "1";
const runCPUOracle = process.env.FLUID_CPU_ORACLE !== "0";
const cpuMaximumCells = Number(process.env.FLUID_CPU_MAX_CELLS ?? 250_000);
const cpuMarkerSamplesPerAxis = Number(process.env.FLUID_CPU_MARKERS_PER_AXIS ?? 1);
const oracleStepsOverride = process.env.FLUID_ORACLE_STEPS === undefined ? undefined : Number(process.env.FLUID_ORACLE_STEPS);
const pressureCyclesOverride = process.env.FLUID_PRESSURE_CYCLES === undefined ? undefined : Number(process.env.FLUID_PRESSURE_CYCLES);
const pressureWarmStartOverride = process.env.FLUID_PRESSURE_WARM_START === undefined ? undefined : process.env.FLUID_PRESSURE_WARM_START !== "0";
const quadtreeMegakernelOverride = process.env.FLUID_QUADTREE_MEGAKERNEL === undefined ? undefined : process.env.FLUID_QUADTREE_MEGAKERNEL !== "0";
const quadtreePressureSolverOverride = process.env.FLUID_QUADTREE_PRESSURE_SOLVER;
if (quadtreePressureSolverOverride !== undefined && quadtreePressureSolverOverride !== "chebyshev" && quadtreePressureSolverOverride !== "pcg") throw new Error("FLUID_QUADTREE_PRESSURE_SOLVER must be chebyshev or pcg");
const remeshIntervalOverride = process.env.FLUID_REMESH_INTERVAL === undefined ? undefined : Number(process.env.FLUID_REMESH_INTERVAL);
const regularLayersOverride = process.env.FLUID_REGULAR_LAYERS === undefined ? undefined : Number(process.env.FLUID_REGULAR_LAYERS);
const voxelCellSizeOverride = (() => {
  if (process.env.FLUID_VOXEL_CELL_SIZE === undefined) return undefined;
  const value = Number(process.env.FLUID_VOXEL_CELL_SIZE);
  if (!Number.isFinite(value) || value <= 0) throw new Error("FLUID_VOXEL_CELL_SIZE must be positive and finite");
  return value;
})();
const expectedGridOverride = (() => {
  const raw = process.env.FLUID_EXPECT_GRID;
  if (raw === undefined) return undefined;
  const values = raw.split(",").map(Number);
  if (values.length !== 3 || values.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("FLUID_EXPECT_GRID must be three comma-separated positive integers (for example 24,18,16)");
  }
  return values as [number, number, number];
})();
const maximumNeighborDeltaOverride = process.env.FLUID_MAX_NEIGHBOR_DELTA === undefined ? undefined : Number(process.env.FLUID_MAX_NEIGHBOR_DELTA);
const maximumTallHeightOverride = process.env.FLUID_MAX_TALL_HEIGHT === undefined ? undefined : Number(process.env.FLUID_MAX_TALL_HEIGHT);
const adaptivityOverride = process.env.FLUID_ADAPTIVITY === undefined ? undefined : Number(process.env.FLUID_ADAPTIVITY);
const opticalDepthOverride = process.env.FLUID_OPTICAL_DEPTH_FRACTION === undefined ? undefined : Number(process.env.FLUID_OPTICAL_DEPTH_FRACTION);
const opticalLayerModeOverride = process.env.FLUID_OPTICAL_LAYER_MODE;
if (opticalLayerModeOverride !== undefined && opticalLayerModeOverride !== "fixed" && opticalLayerModeOverride !== "adaptive-motion") throw new Error("FLUID_OPTICAL_LAYER_MODE must be fixed or adaptive-motion");
const opticalAlphaOverride = process.env.FLUID_OPTICAL_ALPHA === undefined ? undefined : Number(process.env.FLUID_OPTICAL_ALPHA);
const deepSpeedGradientOverride = process.env.FLUID_SIZING_DEEP_SPEED === undefined ? undefined : Number(process.env.FLUID_SIZING_DEEP_SPEED);
const rebuildTopologyOverride = process.env.FLUID_REBUILD_TOPOLOGY === undefined ? undefined : process.env.FLUID_REBUILD_TOPOLOGY !== "0";
const maximumLeafSizeOverride = process.env.FLUID_MAXIMUM_LEAF_SIZE === undefined ? undefined : Number(process.env.FLUID_MAXIMUM_LEAF_SIZE);
const octreeInterfaceBandOverride = process.env.FLUID_OCTREE_INTERFACE_BAND === undefined
  ? undefined : Number(process.env.FLUID_OCTREE_INTERFACE_BAND);
if (octreeInterfaceBandOverride !== undefined
  && (!Number.isInteger(octreeInterfaceBandOverride) || octreeInterfaceBandOverride < 0)) {
  throw new Error("FLUID_OCTREE_INTERFACE_BAND must be a non-negative integer");
}
const octreeGlobalFineFactorOverride = process.env.FLUID_OCTREE_GLOBAL_FINE_FACTOR;
if (octreeGlobalFineFactorOverride !== undefined && !["off", "4", "8"].includes(octreeGlobalFineFactorOverride)) {
  throw new Error("FLUID_OCTREE_GLOBAL_FINE_FACTOR must be off, 4, or 8");
}
const powerGenerationAuditRequested = process.env.FLUID_POWER_GENERATION_AUDIT === "1";
const powerGenerationAuditLog = process.env.FLUID_POWER_GENERATION_AUDIT_LOG !== "0";
const powerStageAuditLog = process.env.FLUID_POWER_STAGE_AUDIT === "1";
const powerAuditEverySteps = Number(process.env.FLUID_POWER_AUDIT_EVERY_STEPS ?? 1);
if (!Number.isSafeInteger(powerAuditEverySteps) || powerAuditEverySteps < 1) {
  throw new Error("FLUID_POWER_AUDIT_EVERY_STEPS must be a positive integer");
}
const powerBoundaryQueryAuditKeys = (process.env.FLUID_POWER_BOUNDARY_QUERY_AUDIT_KEYS ?? "")
  .split(",").filter(Boolean).map(Number);
if (powerBoundaryQueryAuditKeys.some((key) => !Number.isSafeInteger(key) || key < 0 || key >= 0xffff_ffff)) {
  throw new Error("FLUID_POWER_BOUNDARY_QUERY_AUDIT_KEYS must contain comma-separated unsigned brick keys");
}
const topologyTransitionAuditLog = process.env.FLUID_TOPOLOGY_TRANSITION_AUDIT === "1";
const quadtreeStaleStepsOverride = process.env.FLUID_QUADTREE_STALE_STEPS === undefined ? undefined : Number(process.env.FLUID_QUADTREE_STALE_STEPS);
const quadtreeInlineRebuildOverride = process.env.FLUID_QUADTREE_INLINE === undefined ? undefined : process.env.FLUID_QUADTREE_INLINE !== "0";
const quadtreePreconditionerOverride = process.env.FLUID_QUADTREE_PRECONDITIONER;
if (quadtreePreconditionerOverride !== undefined && !["ic0", "blockic", "jacobi", "line", "poly", "mg"].includes(quadtreePreconditionerOverride)) throw new Error("FLUID_QUADTREE_PRECONDITIONER must be ic0, blockic, jacobi, line, poly, or mg");
const quadtreeDebrisCullingOverride = process.env.FLUID_QUADTREE_DEBRIS_CULLING === undefined ? undefined : process.env.FLUID_QUADTREE_DEBRIS_CULLING !== "0";
const quadtreeVofReconciliationOverride = process.env.FLUID_QUADTREE_VOF_RECONCILIATION === undefined ? undefined : process.env.FLUID_QUADTREE_VOF_RECONCILIATION !== "0";
const pressurePhaseTimings = process.env.FLUID_PRESSURE_PHASE_TIMINGS === "1";
const polynomialDegreeOverride = process.env.FLUID_POLYNOMIAL_DEGREE === undefined ? undefined : Number(process.env.FLUID_POLYNOMIAL_DEGREE);
const velocityTransportOverride = process.env.FLUID_VELOCITY_TRANSPORT;
const sharpeningOverride = process.env.FLUID_SHARPENING === undefined ? undefined : process.env.FLUID_SHARPENING !== "0";
const volumeControlOverride = process.env.FLUID_VOLUME_CONTROL === undefined ? undefined : process.env.FLUID_VOLUME_CONTROL !== "0";
const referenceVolumeScaleOverride = process.env.FLUID_REFERENCE_VOLUME_SCALE === undefined ? undefined : Number(process.env.FLUID_REFERENCE_VOLUME_SCALE);
const hierarchyOverride = process.env.FLUID_HIERARCHY === undefined ? undefined : process.env.FLUID_HIERARCHY !== "0";
const checkpointEvery_s = Number(process.env.FLUID_CHECKPOINT_EVERY_S ?? 0);
const stabilityEnvelopeRequested = process.env.FLUID_STABILITY_ENVELOPE === "1";
const energyEverySteps = Number(process.env.FLUID_ENERGY_EVERY_STEPS ?? 0);
const settlingGateRequested = process.env.FLUID_SETTLING_GATE === "1";
const sparseStatsRequested = process.env.FLUID_SPARSE_STATS === "1";
const rasterCheckpointRequested = process.env.FLUID_RASTER_CHECKPOINTS === "1";
const globalFineGenerationTransitionRequested = process.env.FLUID_GLOBAL_FINE_GENERATION_TRANSITION === "1";
// Publication-transition acceptance needs the existing bounded renderer
// counter readback so it can distinguish global fine/coarse authority from an
// adaptive or retained presentation fallback. This is QA-only and adds no
// shader bindings or simulation readback.
if (globalFineGenerationTransitionRequested) process.env.FLUID_WATER_DIAGNOSTICS = "1";
// The CPU-side level-set/velocity reconstruction has a small positive
// equilibrium drift even for the uniform oracle.  The default is more than
// six times the measured uniform 10 s noise floor (1.61e-4 /s on 2026-07-16)
// while remaining almost nine times below the reproduced tall-cell growth.
const settlingNormalizedSlopeEpsilon = Number(process.env.FLUID_SETTLING_NORMALIZED_SLOPE_EPSILON ?? 1e-3);
if (!Number.isInteger(energyEverySteps) || energyEverySteps < 0) throw new Error("FLUID_ENERGY_EVERY_STEPS must be a non-negative integer");
if (settlingGateRequested && energyEverySteps === 0) throw new Error("FLUID_SETTLING_GATE=1 requires FLUID_ENERGY_EVERY_STEPS > 0");
if (!Number.isFinite(settlingNormalizedSlopeEpsilon) || settlingNormalizedSlopeEpsilon < 0) throw new Error("FLUID_SETTLING_NORMALIZED_SLOPE_EPSILON must be a non-negative finite number");
if (referenceVolumeScaleOverride !== undefined && (!Number.isFinite(referenceVolumeScaleOverride) || referenceVolumeScaleOverride <= 0)) throw new Error("FLUID_REFERENCE_VOLUME_SCALE must be a positive finite number");
const singleTallCellSupportRadius = process.env.FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS === undefined
  ? 0 : Number(process.env.FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS);
const singleTallCellProbe: SingleTallCellProbeOptions | undefined = (() => {
  const raw = process.env.FLUID_SINGLE_TALL_CELL;
  if (!raw) return undefined;
  const values = raw.split(",").map(Number);
  if (values.some((value) => !Number.isFinite(value)) || (values.length !== 1 && values.length !== 3)) {
    throw new Error("FLUID_SINGLE_TALL_CELL must be HEIGHT or X,Z,HEIGHT");
  }
  if (!Number.isFinite(singleTallCellSupportRadius) || singleTallCellSupportRadius < 0) {
    throw new Error("FLUID_SINGLE_TALL_CELL_SUPPORT_RADIUS must be a non-negative number");
  }
  return values.length === 1
    ? { height: values[0], supportRadius: singleTallCellSupportRadius }
    : { x: values[0], z: values[1], height: values[2], supportRadius: singleTallCellSupportRadius };
})();

function selectedScenarios(): SmokeScenarioId[] {
  const selection = process.env.FLUID_SCENE ?? "hose-tank";
  if (selection === "all") return [...smokeScenarioIds];
  const ids = selection.split(",").map((value) => value.trim()).filter(Boolean);
  for (const id of ids) if (!isSmokeScenarioId(id)) throw new Error(`Unknown FLUID_SCENE=${id}; expected all or ${smokeScenarioIds.join(", ")}`);
  return ids as SmokeScenarioId[];
}

function applySceneOverrides(scene: SceneDescription): SceneDescription {
  if (maxDtOverride !== undefined) scene.numerics.maxDt_s = maxDtOverride;
  return scene;
}

async function readFloatTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * (y + height * z), width);
    output.set(row, width * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

async function readBufferBinding(device: GPUDevice, binding: GPUBufferBinding, byteLength: number) {
  const alignedLength = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const readback = device.createBuffer({ size: alignedLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Sparse voxel smoke readback" });
  encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, readback, 0, alignedLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(byteLength);
  bytes.set(new Uint8Array(readback.getMappedRange(0, alignedLength)).subarray(0, byteLength));
  readback.unmap(); readback.destroy();
  return bytes;
}

interface OctreeFaceMirrorDiagnostics {
  faceCount: number;
  faceCapacity: number;
  overflow: boolean;
  maximumIncidence: number;
  incidenceOverflowCount: number;
  comparedRows: number;
  mismatchedRows: number;
  maximumAbsoluteRhsError: number;
  maximumReferenceRhs: number;
  projectionComparedFaces: number;
  projectionMismatchedFaces: number;
  maximumAbsoluteProjectedVelocityError: number;
  maximumReferenceProjectedVelocity: number;
  maximumProjectedDivergence_s?: number;
  rmsProjectedDivergence_s?: number;
  maximumProjectedDivergenceStep?: number;
  rmsProjectedDivergenceStep?: number;
  projectedDivergenceRows?: number;
  projectedDivergenceNonFiniteRows?: number;
  topologyTransferTransferred?: number;
  topologyTransferInitialized?: number;
  topologyTransferInvalid?: number;
  topologyTransferFailed?: boolean;
  topologyTransferKeyHash?: number;
  topologyTransferVelocityHash?: number;
  topologyTransferHashedFaces?: number;
}

interface OctreePowerFaceTransferDiagnostics {
  previousFaceCount: number;
  valid: boolean;
  flags: number;
  generation: number;
  exactFaces: number;
  fallbackFaces: number;
  sourceFlags: number;
}
interface OctreePowerFaceDiagnostics {
  rowCount: number; faceCount: number; incidenceCount: number; flags: number;
  firstInvalid: number; invalidCount: number; boundaryCount: number; generation: number;
  valid: boolean; lookupMissCount: number; maximumObservedProbe: number; worldBoundaryCount: number;
  firstInvalidSlot: number; firstInvalidNeighbor: number; firstInvalidDetail: number;
  firstInvalidRow?: number;
  firstInvalidPad3?: number;
  firstInvalidPair?: Array<{ row: number; cell: number; size: number; topologyCode: number; transformAndFlags: number; gradient: number[] }>;
}

async function readOctreeFaceMirrorDiagnostics(device: GPUDevice, source: OctreeFaceMirrorSource, dt_s = 1): Promise<OctreeFaceMirrorDiagnostics> {
  const projectionParityBuffer = source.projectionParity ?? source.parity;
  const projectionParityOffset = source.projectionParityOffset ?? 0;
  const [controlBytes, parityBytes, projectionBytes, divergenceBytes, transferBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.control }, 24),
    readBufferBinding(device, { buffer: source.parity }, 16),
    readBufferBinding(device, { buffer: projectionParityBuffer, offset: projectionParityOffset }, 16),
    source.projectedDivergence
      ? readBufferBinding(device, { buffer: source.projectedDivergence, offset: source.projectedDivergenceOffset ?? 0 }, 16)
      : undefined,
    source.topologyTransferDiagnostics
      ? readBufferBinding(device, { buffer: source.topologyTransferDiagnostics }, 32)
      : undefined,
  ]);
  const control = new Uint32Array(controlBytes.buffer, controlBytes.byteOffset, 6);
  const parity = new Uint32Array(parityBytes.buffer, parityBytes.byteOffset, 4);
  const parityFloats = new Float32Array(parity.buffer, parity.byteOffset, 2);
  const projection = new Uint32Array(projectionBytes.buffer, projectionBytes.byteOffset, 4);
  const projectionFloats = new Float32Array(projection.buffer, projection.byteOffset, 2);
  const divergence = divergenceBytes ? new Uint32Array(divergenceBytes.buffer, divergenceBytes.byteOffset, 4) : undefined;
  const divergenceFloats = divergenceBytes ? new Float32Array(divergenceBytes.buffer, divergenceBytes.byteOffset, 4) : undefined;
  const maximumProjectedDivergence_s = divergenceFloats?.[0];
  const rmsProjectedDivergence_s = divergence && divergenceFloats
    ? Math.sqrt(Math.max(0, divergenceFloats[3]) / Math.max(1, divergence[1]))
    : undefined;
  const transfer = transferBytes ? new Uint32Array(transferBytes.buffer, transferBytes.byteOffset, 8) : undefined;
  return {
    faceCount: control[0],
    faceCapacity: control[2],
    overflow: control[1] !== 0,
    maximumIncidence: control[4],
    incidenceOverflowCount: control[5],
    maximumAbsoluteRhsError: parityFloats[0],
    maximumReferenceRhs: parityFloats[1],
    mismatchedRows: parity[2],
    comparedRows: parity[3],
    maximumAbsoluteProjectedVelocityError: projectionFloats[0],
    maximumReferenceProjectedVelocity: projectionFloats[1],
    projectionMismatchedFaces: projection[2],
    projectionComparedFaces: projection[3],
    maximumProjectedDivergence_s,
    rmsProjectedDivergence_s,
    maximumProjectedDivergenceStep: maximumProjectedDivergence_s === undefined ? undefined : maximumProjectedDivergence_s * dt_s,
    rmsProjectedDivergenceStep: rmsProjectedDivergence_s === undefined ? undefined : rmsProjectedDivergence_s * dt_s,
    projectedDivergenceRows: divergence?.[1],
    projectedDivergenceNonFiniteRows: divergence?.[2],
    topologyTransferTransferred: transfer?.[0],
    topologyTransferInitialized: transfer?.[1],
    topologyTransferInvalid: transfer?.[2],
    topologyTransferFailed: transfer ? transfer[3] !== 0 : undefined,
    topologyTransferKeyHash: transfer?.[4],
    topologyTransferVelocityHash: transfer?.[5],
    topologyTransferHashedFaces: transfer?.[6],
  };
}

interface SparseVoxelSmokeStats {
  voxelCount: number;
  brickCount: number;
  activeVoxelCount: number;
  activeBrickCount: number;
  fluidVoxelCount: number;
  environmentVoxelCount: number;
  materialVoxelCounts: Record<string, number>;
  nonFiniteRecordCount: number;
  invalidMaterialCount: number;
  fluidColorLinear: number[];
  uiRawVoxelRenderWall_ms: number;
  uiBrickGridRenderWall_ms: number;
  fluidBrickCapacity?: number;
  fluidBrickResidentCount?: number;
  fluidBrickCoreCount?: number;
  fluidBrickHaloCount?: number;
  fluidBrickActivatedCount?: number;
  fluidBrickRetiredCount?: number;
  fluidBrickGeneration?: number;
  fluidBrickCoreOrigins_m?: number[][];
  fluidBrickHaloOrigins_m?: number[][];
  sourceBrickFluidVoxelCount?: number;
  sourceBrickResidency?: "core" | "halo" | "vacant";
}

interface FluidBrickSnapshot { resident: number; core: number; halo: number; generation: number }
interface WorldBounds { min: [number, number, number]; max: [number, number, number] }

function initialSeedBrickBounds(scene: SceneDescription, dimensions: readonly [number, number, number], brickSize = 8): WorldBounds | undefined {
  const seed = scene.fluid.initialBrickSeeds_m?.[0];
  if (!seed) return undefined;
  const minimum: [number, number, number] = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const extent: [number, number, number] = [scene.container.width_m, scene.container.height_m, scene.container.depth_m];
  const point = [seed.x, seed.y, seed.z];
  const start = point.map((value, axis) => {
    const cell = Math.max(0, Math.min(dimensions[axis] - 1, Math.floor((value - minimum[axis]) * dimensions[axis] / extent[axis])));
    return Math.floor(cell / brickSize) * brickSize;
  });
  return {
    min: start.map((cell, axis) => minimum[axis] + cell * extent[axis] / dimensions[axis]) as [number, number, number],
    max: start.map((cell, axis) => minimum[axis] + Math.min(dimensions[axis], cell + brickSize) * extent[axis] / dimensions[axis]) as [number, number, number]
  };
}

async function readFluidBrickSnapshot(device: GPUDevice, source: SparseVoxelRenderSource): Promise<FluidBrickSnapshot | undefined> {
  if (!source.fluidBrickStats) return undefined;
  const words = new Uint32Array((await readBufferBinding(device, source.fluidBrickStats, 64)).buffer);
  return { resident: words[0], core: words[8], halo: words[9], generation: words[15] };
}

async function smokeRenderSparseVoxelDebugModes(device: GPUDevice, source: SparseVoxelRenderSource) {
  const main = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
  await main.initialize();
  main.setSource(source);
  const color = device.createTexture({ size: [320, 180], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const depth = device.createTexture({ size: [320, 180], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const matrix = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const renderMode = async (mode: "raw-voxels" | "brick-grid") => {
    const started = performance.now();
    const encoder = device.createCommandEncoder({ label: `Sparse voxel ${mode} WebGPU smoke` });
    main.encode(encoder, {
      mode,
      colorTarget: color.createView(), depthTarget: depth.createView(),
      colorLoadOp: "clear", depthLoadOp: "clear",
      viewProjection: matrix, cameraPosition: [0, 0, 4],
      containerBounds: { min: [-1, 0, -1], max: [1, 2, 1] },
      containerClosedTop: false
    });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  };
  const uiRawVoxelRenderWall_ms = await renderMode("raw-voxels");
  const uiBrickGridRenderWall_ms = await renderMode("brick-grid");
  main.destroy(); color.destroy(); depth.destroy();
  return { uiRawVoxelRenderWall_ms, uiBrickGridRenderWall_ms };
}

async function readSparseVoxelStats(device: GPUDevice, source: SparseVoxelRenderSource, sourceBrick?: WorldBounds): Promise<SparseVoxelSmokeStats> {
  const voxelCount = Math.min(new Uint32Array((await readBufferBinding(device, source.voxelCount, 4)).buffer)[0], source.voxelCapacity);
  const brickCount = Math.min(new Uint32Array((await readBufferBinding(device, source.brickCount, 4)).buffer)[0], source.brickCapacity);
  const voxelBytes = await readBufferBinding(device, source.voxelRecords, voxelCount * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
  const brickBytes = await readBufferBinding(device, source.brickRecords, brickCount * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
  const materialBytes = await readBufferBinding(device, source.materials, source.materialCount * 32);
  const voxelFloats = new Float32Array(voxelBytes.buffer), voxelWords = new Uint32Array(voxelBytes.buffer);
  const brickWords = new Uint32Array(brickBytes.buffer), brickFloats = new Float32Array(brickBytes.buffer), materialFloats = new Float32Array(materialBytes.buffer);
  let activeVoxelCount = 0, activeBrickCount = 0, fluidVoxelCount = 0, environmentVoxelCount = 0, nonFiniteRecordCount = 0, invalidMaterialCount = 0;
  let sourceBrickFluidVoxelCount = 0;
  const fluidBrickCoreOrigins_m: number[][] = [], fluidBrickHaloOrigins_m: number[][] = [];
  const materialVoxelCounts: Record<string, number> = {};
  for (let index = 0; index < voxelCount; index += 1) {
    const word = index * 12, material = voxelWords[word + 8], flags = voxelWords[word + 9];
    if ((flags & 1) === 0) continue;
    activeVoxelCount += 1;
    materialVoxelCounts[String(material)] = (materialVoxelCounts[String(material)] ?? 0) + 1;
    if (material === VOXEL_MATERIAL_IDS.fluid) {
      fluidVoxelCount += 1;
      const centre = [voxelFloats[word] + 0.5 * voxelFloats[word + 4], voxelFloats[word + 1] + 0.5 * voxelFloats[word + 5], voxelFloats[word + 2] + 0.5 * voxelFloats[word + 6]];
      if (sourceBrick && centre.every((value, axis) => value >= sourceBrick.min[axis] - 1e-6 && value < sourceBrick.max[axis] - 1e-6)) sourceBrickFluidVoxelCount += 1;
    }
    if (material >= ENVIRONMENT_VOXEL_MATERIAL_BASE) environmentVoxelCount += 1;
    if (material >= source.materialCount) invalidMaterialCount += 1;
    if (![...voxelFloats.slice(word, word + 3), ...voxelFloats.slice(word + 4, word + 7)].every(Number.isFinite)
      || voxelFloats[word + 4] <= 0 || voxelFloats[word + 5] <= 0 || voxelFloats[word + 6] <= 0) nonFiniteRecordCount += 1;
  }
  for (let index = 0; index < brickCount; index += 1) {
    const word = index * 12, flags = brickWords[word + 9];
    if ((flags & 1) !== 0) activeBrickCount += 1;
    const origin = () => Array.from(brickFloats.slice(word, word + 3));
    if ((flags & 2) !== 0) fluidBrickCoreOrigins_m.push(origin());
    else if ((flags & 4) !== 0) fluidBrickHaloOrigins_m.push(origin());
  }
  const colorOffset = VOXEL_MATERIAL_IDS.fluid * 8;
  const debugRenderTimings = await smokeRenderSparseVoxelDebugModes(device, source);
  const fluidBrickWords = source.fluidBrickStats
    ? new Uint32Array((await readBufferBinding(device, source.fluidBrickStats, 64)).buffer)
    : undefined;
  return {
    voxelCount, brickCount, activeVoxelCount, activeBrickCount, fluidVoxelCount, environmentVoxelCount, materialVoxelCounts,
    nonFiniteRecordCount, invalidMaterialCount,
    fluidColorLinear: Array.from(materialFloats.slice(colorOffset, colorOffset + 3)),
    ...debugRenderTimings,
    ...(fluidBrickWords ? {
      fluidBrickCapacity: source.fluidBrickCapacity,
      fluidBrickResidentCount: fluidBrickWords[0], fluidBrickCoreCount: fluidBrickWords[8], fluidBrickHaloCount: fluidBrickWords[9],
      fluidBrickActivatedCount: fluidBrickWords[10], fluidBrickRetiredCount: fluidBrickWords[11], fluidBrickGeneration: fluidBrickWords[15],
      fluidBrickCoreOrigins_m, fluidBrickHaloOrigins_m,
      sourceBrickFluidVoxelCount,
      sourceBrickResidency: fluidBrickCoreOrigins_m.some((origin) => origin.every((value, axis) => Math.abs(value - (sourceBrick?.min[axis] ?? Infinity)) <= 1e-5))
        ? "core" as const
        : fluidBrickHaloOrigins_m.some((origin) => origin.every((value, axis) => Math.abs(value - (sourceBrick?.min[axis] ?? Infinity)) <= 1e-5))
          ? "halo" as const
          : "vacant" as const,
    } : {})
  };
}

interface HybridPresentationSmokeStats {
  initializeWall_ms: number;
  frameWall_ms: number;
  bodyCount: number;
  width: number;
  height: number;
  frontInterfacePixels: number;
  backInterfacePixels: number;
  pairedInterfacePixels: number;
  frontOnlyInterfacePixels: number;
  backOnlyInterfacePixels: number;
  /** First isolated screen-space witnesses where a back crossing has no front crossing. */
  backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
  /** World-space back-interface positions for the corresponding witnesses. */
  backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
  frontInterfaceHash: number;
  backInterfaceHash: number;
  rendererValidationErrorCount: number;
  rendererUncapturedErrorCount: number;
  surfaceGeometrySource?: WaterSurfaceGeometrySource;
  globalFineAuthorityLatch?: number;
  globalFineCrossingPublished?: boolean;
  presentationFallbackActive?: boolean;
  vertexCount?: number;
  activeCubeCount?: number;
  vertexAllocator?: number;
  vertexCapacity?: number;
  activeCubeCapacity?: number;
  narrowVerticalSlits: NarrowVerticalSlitMetrics;
  /** Front-facing pixels lying on a side-wall cap within 0.4 fine cells of each x/z corner. */
  wallCornerCapPixels?: readonly [number, number, number, number];
  /** Pixels on the two exposed vertical dam faces next to their shared +x/+z corner. */
  damExposedCornerCapPixels?: readonly [number, number];
  frontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  reverseView?: {
    frontInterfacePixels: number;
    backInterfacePixels: number;
    pairedInterfacePixels: number;
    frontOnlyInterfacePixels: number;
    backOnlyInterfacePixels: number;
    backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
    backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
    frontInterfaceHash: number;
    backInterfaceHash: number;
    narrowVerticalSlits: NarrowVerticalSlitMetrics;
    wallCornerCapPixels?: readonly [number, number, number, number];
    damExposedCornerCapPixels?: readonly [number, number];
    frontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  };
  globalFineAuthorityTransition?: {
    validGeneration: number;
    unpublishedGeneration: number;
    cleanFineCoarseRequired: true;
    retainedGeometrySource?: WaterSurfaceGeometrySource;
    retainedFrontInterfacePixels: number;
    retainedBackInterfacePixels: number;
    retainedFrontInterfaceHash: number;
    retainedBackInterfaceHash: number;
    retainedFrontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  };
}

interface GlobalFineGenerationDiagnostics {
  generation: number;
  worklistGeneration?: number;
  generationSlot: number;
  activePages: number;
  configuredBrickCapacity: number;
  taggedMetadataPages: number;
  malformedActivePages: number;
  validSamples: number;
  finiteValidSamples: number;
  negativeValidSamples: number;
  positiveValidSamples: number;
  publicationValid: boolean;
  residentPayloadBytes: number;
  payloadCapacityBytes: number;
  payloadFragmentationBytes: number;
  pageHashBytes: number;
  pageMetadataBytes: number;
  pageWorklistBytes: number;
  diagnosticReadbackBytes: number;
  coarseState?: number;
  coarseGeneration?: number;
  coarseHashCapacity?: number;
  coarseMaximumLeafSize?: number;
  coarseEntryCount?: number;
  coarseNegativeEntries?: number;
  coarsePositiveEntries?: number;
  coarseInterfaceEntries?: number;
  coarseMalformedEntries?: number;
  seedCount?: number;
  seedFlags?: number;
  topologyFlags?: number;
  topologyInterfaceBricks?: number;
  topologyDesiredBricks?: number;
  topologyRequiredDesiredBricks?: number;
  topologyRequiredDesiredBricksExact?: boolean;
  topologyDilationBrickRings?: number;
  topologyActivatedBricks?: number;
  topologyPublished?: boolean;
  topologyRolledBack?: boolean;
  /** Downstream finalization reason mask: topology=1, redistance=2, volume=4, transport=8. */
  topologyFinalizeReason?: number;
  phiBitXor: number;
  phiBitSum: number;
  phiSum: number;
  phiAbsSum: number;
  transportDepartureOutsideBand?: number;
  transportNonfiniteVelocity?: number;
  transportProcessed?: number;
  transportCommitted?: boolean;
  transportExtrapolatedVelocity?: number;
  transportMaximumDisplacementFineCells?: number;
  transportFaceBandUnavailable?: number;
  transportVelocityUnavailable?: number;
  redistanceUnresolvedCells?: number;
  redistanceMaximumResidualScaled?: number;
  redistanceSeedCount?: number;
  redistanceCommitted?: boolean;
  volumeFlags?: number;
  volumeInitialized?: boolean;
  volumeSamples?: number;
  volumeReference?: number;
  volumeCurrent?: number;
  volumeInterfaceArea?: number;
  volumeCorrection?: number;
  volumeCorrected?: boolean;
  volumeCoarse?: number;
  volumeFine?: number;
  volumeReplacedCoarse?: number;
  volumeCoarseRows?: number;
  volumeUnowned?: number;
  volumeExpectedAir?: number;
  volumeLookupFailures?: number;
  volumeStaleOwners?: number;
  volumeGeneration?: number;
  powerVelocityFlags?: number;
  powerVelocityRows?: number;
  powerVelocityReconstructed?: number;
  powerVelocityFallbacks?: number;
  powerProjectionControl?: readonly number[];
  faceBandFlags?: number;
  faceBandFirstError?: number;
  faceBandRows?: number;
  faceBandFaces?: number;
  faceBandIncidences?: number;
  faceBandGeneration?: number;
  faceBandValid?: boolean;
  faceBandMaximumDepth?: number;
  faceBandSeeds?: number;
  faceBandAccepted?: number;
  faceBandUnresolved?: number;
  faceBandSampleFailures?: number;
  faceBandRowCapacity?: number;
  faceBandFaceCapacity?: number;
  probedPages?: Array<{
    key: number;
    hashPhysicalId?: number;
    hashFound: boolean;
    metadataKey?: number;
    metadataGeneration?: number;
    metadataMatchesGeneration: boolean;
    inPublishedWorklist: boolean;
    validSamples: number;
    finiteValidSamples: number;
    requiredCenterSamples?: Array<{ local: number; flags: number; phi: number | null }>;
    requiredCenterValid?: boolean;
    requiredCenterFinite?: boolean;
  }>;
  probedCoarseRecords?: Array<{
    cell: number; found: boolean; lookupSize: number; phi: number;
    minimumPhi: number; maximumPhi: number; flags: number;
  }>;
}

function fineTrilinearBrickKeysAtPosition(
  source: WebGPUFineLevelSetBrickSource,
  position: readonly [number, number, number],
): number[] {
  const plan = source.plan;
  const lattice = position.map((value, axis) =>
    (value - plan.domainOrigin[axis]) / plan.fineCellWidth - 0.5) as [number, number, number];
  if (lattice.some((value) => !Number.isFinite(value))) return [];
  const base = lattice.map(Math.floor) as [number, number, number];
  const keys = new Set<number>();
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    const q: [number, number, number] = [base[0] + x, base[1] + y, base[2] + z];
    if (q.some((value, axis) => value < 0 || value >= plan.sampleDimensions[axis])) continue;
    const brick = q.map((value) => Math.floor(value / plan.brickResolution)) as [number, number, number];
    keys.add(brick[0] + plan.brickDimensions[0] * (brick[1] + plan.brickDimensions[1] * brick[2]));
  }
  return [...keys].sort((a, b) => a - b);
}

async function readGlobalFineGenerationDiagnostics(
  device: GPUDevice,
  solver: GPUSolverInstance,
  probeBrickKeys: readonly number[] = [],
  sourceOverride?: WebGPUFineLevelSetBrickSource,
): Promise<GlobalFineGenerationDiagnostics | undefined> {
  const source = sourceOverride ?? solver.globalFineLevelSetSource;
  if (!source) return undefined;
  const transportControl = (solver as GPUSolverInstance & { globalFineTransportControl?: GPUBuffer })
    .globalFineTransportControl;
  const redistanceControl = solver.globalFineRedistanceControl;
  const volumeControl = solver.globalFineVolumeControl;
  const powerVelocityControl = (solver as GPUSolverInstance & { globalFinePowerVelocityControl?: GPUBuffer })
    .globalFinePowerVelocityControl;
  const powerProjectionControl = (solver as GPUSolverInstance & { globalFinePowerProjectionControl?: GPUBuffer })
    .globalFinePowerProjectionControl;
  const faceBandControl = solver.globalFineFaceBandControl;
  const faceBandPlan = solver.globalFineFaceBandPlan;
  const pageCapacity = source.plan.maximumResidentBricks;
  const samplesPerBrick = source.plan.samplesPerBrick;
  const [worklistBytes, metadataBytes, flagBytes, phiBytes, hashBytes, coarseBytes, seedBytes, topologyBytes,
    transportBytes, redistanceBytes, volumeBytes, powerVelocityBytes, powerProjectionBytes, faceBandBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, (5 + pageCapacity) * 4),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 40),
    readBufferBinding(device, { buffer: source.flags }, pageCapacity * samplesPerBrick * 4),
    readBufferBinding(device, { buffer: source.phi }, pageCapacity * samplesPerBrick * 4),
    probeBrickKeys.length > 0
      ? readBufferBinding(device, { buffer: source.hash }, source.plan.hashCapacity * 8)
      : Promise.resolve(undefined),
    source.coarsePhiDirectory
      ? readBufferBinding(device, { buffer: source.coarsePhiDirectory },
        32 + (source.coarsePhiHashCapacity ?? 0) * 32)
      : Promise.resolve(undefined),
    source.seedControl ? readBufferBinding(device, { buffer: source.seedControl }, 8) : Promise.resolve(undefined),
    source.topologyControl
      ? readBufferBinding(device, { buffer: source.topologyControl }, 32)
      : Promise.resolve(undefined),
    transportControl ? readBufferBinding(device, { buffer: transportControl }, 64) : Promise.resolve(undefined),
    redistanceControl ? readBufferBinding(device, { buffer: redistanceControl }, 16) : Promise.resolve(undefined),
    volumeControl ? readBufferBinding(device, { buffer: volumeControl }, 64) : Promise.resolve(undefined),
    powerVelocityControl ? readBufferBinding(device, { buffer: powerVelocityControl }, 32) : Promise.resolve(undefined),
    powerProjectionControl ? readBufferBinding(device, { buffer: powerProjectionControl }, 64) : Promise.resolve(undefined),
    faceBandControl ? readBufferBinding(device, { buffer: faceBandControl }, 64) : Promise.resolve(undefined),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const flags = new Uint32Array(flagBytes.buffer, flagBytes.byteOffset, flagBytes.byteLength / 4);
  const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const phiBits = new Uint32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const activePages = Math.min(worklist[0], pageCapacity);
  let taggedMetadataPages = 0, malformedActivePages = 0;
  let validSamples = 0, finiteValidSamples = 0, negativeValidSamples = 0, positiveValidSamples = 0;
  let phiBitXor = 0, phiBitSum = 0, phiSum = 0, phiAbsSum = 0;
  for (let id = 0; id < pageCapacity; id += 1) if (metadata[id * 10 + 2] === source.generation) taggedMetadataPages += 1;
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[5 + work];
    if (id >= pageCapacity || metadata[id * 10 + 2] !== source.generation || metadata[id * 10] !== id) {
      malformedActivePages += 1; continue;
    }
    for (let local = 0; local < samplesPerBrick; local += 1) {
      const index = id * samplesPerBrick + local;
      if ((flags[index] & 1) === 0) continue;
      validSamples += 1;
      const value = phi[index];
      if (!Number.isFinite(value)) continue;
      finiteValidSamples += 1;
      const logicalSample = (Math.imul(metadata[id * 10 + 1], samplesPerBrick) + local) >>> 0;
      let mixed = Math.imul((phiBits[index] ^ logicalSample) >>> 0, 0x7feb_352d) >>> 0;
      mixed = Math.imul((mixed ^ (mixed >>> 15)) >>> 0, 0x846c_a68b) >>> 0;
      mixed = (mixed ^ (mixed >>> 16)) >>> 0;
      phiBitXor = (phiBitXor ^ mixed) >>> 0; phiBitSum = (phiBitSum + mixed) >>> 0;
      phiSum += value; phiAbsSum += Math.abs(value);
      if (value < 0) negativeValidSamples += 1; else positiveValidSamples += 1;
    }
  }
  const probedPages: GlobalFineGenerationDiagnostics["probedPages"] = [];
  if (hashBytes && probeBrickKeys.length > 0) {
    const hash = new Uint32Array(hashBytes.buffer, hashBytes.byteOffset, hashBytes.byteLength / 4);
    const publishedIds = new Set<number>();
    for (let work = 0; work < activePages; work += 1) publishedIds.add(worklist[5 + work]);
    const lookup = (key: number) => {
      const start = Math.imul((key ^ (key >>> 16)) >>> 0, 0x9e37_79b1) >>> 0
        & (source.plan.hashCapacity - 1);
      for (let probe = 0; probe < Math.min(32, source.plan.maximumHashProbes); probe += 1) {
        const slot = (start + probe) & (source.plan.hashCapacity - 1), stored = hash[slot * 2];
        if (stored === key) return hash[slot * 2 + 1];
        if (stored === 0xffff_ffff) return undefined;
      }
      return undefined;
    };
    const requiredLocals = source.plan.fineFactor === 4 && source.plan.brickResolution === 4
      ? [21, 22, 25, 26, 37, 38, 41, 42] : undefined;
    for (const key of probeBrickKeys) {
      const id = lookup(key); const validId = id !== undefined && id < pageCapacity;
      let pageValid = 0, pageFinite = 0;
      if (validId) for (let local = 0; local < samplesPerBrick; local += 1) {
        const index = id * samplesPerBrick + local;
        if ((flags[index] & 1) === 0) continue;
        pageValid += 1; if (Number.isFinite(phi[index])) pageFinite += 1;
      }
      const requiredCenterSamples = validId && requiredLocals
        ? requiredLocals.map((local) => {
          const index = id * samplesPerBrick + local, value = phi[index];
          return { local, flags: flags[index], phi: Number.isFinite(value) ? value : null };
        }) : undefined;
      probedPages.push({
        key, hashPhysicalId: id, hashFound: validId,
        metadataKey: validId ? metadata[id * 10 + 1] : undefined,
        metadataGeneration: validId ? metadata[id * 10 + 2] : undefined,
        metadataMatchesGeneration: validId && metadata[id * 10] === id
          && metadata[id * 10 + 1] === key && metadata[id * 10 + 2] === source.generation,
        inPublishedWorklist: validId && publishedIds.has(id),
        validSamples: pageValid, finiteValidSamples: pageFinite,
        requiredCenterSamples,
        requiredCenterValid: requiredCenterSamples?.every((sample) => (sample.flags & 1) !== 0),
        requiredCenterFinite: requiredCenterSamples?.every((sample) => sample.phi !== null),
      });
    }
  }
  const coarse = coarseBytes
    ? new Uint32Array(coarseBytes.buffer, coarseBytes.byteOffset, coarseBytes.byteLength / 4)
    : undefined;
  let coarseEntryCount = 0, coarseNegativeEntries = 0, coarsePositiveEntries = 0;
  let coarseInterfaceEntries = 0, coarseMalformedEntries = 0;
  if (coarse) for (let slot = 0; slot < (coarse.length - 8) / 8; slot += 1) {
    const base = 8 + slot * 8;
    if (coarse[base] === 0) continue;
    coarseEntryCount += 1;
    const values = new Float32Array(coarse.buffer, coarse.byteOffset + (base + 2) * 4, 3);
    const [phiValue, minimumPhi, maximumPhi] = values;
    if (!Number.isFinite(phiValue) || !Number.isFinite(minimumPhi) || !Number.isFinite(maximumPhi)
      || minimumPhi > phiValue || phiValue > maximumPhi || (coarse[base + 5] & 9) !== 9) {
      coarseMalformedEntries += 1; continue;
    }
    if (minimumPhi < 0) coarseNegativeEntries += 1; else coarsePositiveEntries += 1;
    if (minimumPhi <= 0 && maximumPhi >= 0) coarseInterfaceEntries += 1;
  }
  const probedCoarseRecords: GlobalFineGenerationDiagnostics["probedCoarseRecords"] = [];
  if (coarse && probeBrickKeys.length > 0) {
    const capacity = coarse[2], maximumLeaf = coarse[3], dimensions = [coarse[4], coarse[5], coarse[6]];
    const physicalCellSize = new Float32Array(coarse.buffer, coarse.byteOffset + 7 * 4, 1)[0];
    const hash = (cell: number, size: number) => {
      let value = (cell ^ Math.imul(size, 0x9e37_79b9)) >>> 0;
      value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb_352d) >>> 0;
      value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846c_a68b) >>> 0;
      return (value ^ (value >>> 16)) >>> 0;
    };
    for (const requested of probeBrickKeys) {
      const q = [requested % dimensions[0], Math.floor(requested / dimensions[0]) % dimensions[1],
        Math.floor(requested / (dimensions[0] * dimensions[1]))];
      let foundBase = -1, lookupSize = 0;
      for (let size = 1; size <= maximumLeaf; size *= 2) {
        const origin = q.map((value) => Math.floor(value / size) * size);
        const cell = origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
        let slot = hash(cell, size) & (capacity - 1);
        for (let probe = 0; probe < Math.min(32, capacity); probe += 1) {
          const base = 8 + slot * 8, observed = coarse[base];
          if (observed === 0) break;
          if (observed === cell + 1 && coarse[base + 1] === size) { foundBase = base; lookupSize = size; break; }
          slot = (slot + 1) & (capacity - 1);
        }
        if (foundBase >= 0) break;
      }
      if (foundBase >= 0) {
        const values = new Float32Array(coarse.buffer, coarse.byteOffset + (foundBase + 2) * 4, 3);
        probedCoarseRecords.push({ cell: requested, found: true, lookupSize,
          phi: values[0], minimumPhi: values[1], maximumPhi: values[2], flags: coarse[foundBase + 5] });
      } else {
        const air = physicalCellSize * maximumLeaf;
        probedCoarseRecords.push({ cell: requested, found: false, lookupSize: 0,
          phi: air, minimumPhi: air, maximumPhi: air, flags: 0 });
      }
    }
  }
  const seed = seedBytes
    ? new Uint32Array(seedBytes.buffer, seedBytes.byteOffset, seedBytes.byteLength / 4)
    : undefined;
  const topology = topologyBytes
    ? new Uint32Array(topologyBytes.buffer, topologyBytes.byteOffset, topologyBytes.byteLength / 4)
    : undefined;
  const transport = transportBytes
    ? new Uint32Array(transportBytes.buffer, transportBytes.byteOffset, transportBytes.byteLength / 4)
    : undefined;
  const redistance = redistanceBytes
    ? new Uint32Array(redistanceBytes.buffer, redistanceBytes.byteOffset, redistanceBytes.byteLength / 4)
    : undefined;
  const volume = volumeBytes
    ? new Uint32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  const volumeFloats = volumeBytes
    ? new Float32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  const powerVelocity = powerVelocityBytes
    ? new Uint32Array(powerVelocityBytes.buffer, powerVelocityBytes.byteOffset, powerVelocityBytes.byteLength / 4)
    : undefined;
  const powerProjection = powerProjectionBytes
    ? new Uint32Array(powerProjectionBytes.buffer, powerProjectionBytes.byteOffset, powerProjectionBytes.byteLength / 4)
    : undefined;
  const faceBand = faceBandBytes
    ? unpackOctreeFaceBandControl(new Uint32Array(faceBandBytes.buffer, faceBandBytes.byteOffset,
      faceBandBytes.byteLength / 4))
    : undefined;
  return {
    generation: source.generation, worklistGeneration: worklist[1], generationSlot: source.generationSlot, activePages,
    configuredBrickCapacity: pageCapacity,
    taggedMetadataPages, malformedActivePages, validSamples, finiteValidSamples,
    negativeValidSamples, positiveValidSamples, phiBitXor, phiBitSum, phiSum, phiAbsSum,
    residentPayloadBytes: activePages * source.plan.payloadBytesPerBrick,
    payloadCapacityBytes: source.plan.payloadCapacityBytes,
    payloadFragmentationBytes: (pageCapacity - activePages) * source.plan.payloadBytesPerBrick,
    pageHashBytes: source.plan.hashCapacity * 8,
    pageMetadataBytes: pageCapacity * 40,
    pageWorklistBytes: (5 + pageCapacity) * 4,
    diagnosticReadbackBytes: [worklistBytes, metadataBytes, flagBytes, phiBytes, hashBytes, coarseBytes, seedBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes, powerVelocityBytes, powerProjectionBytes,
      faceBandBytes]
      .reduce((sum, bytes) => sum + (bytes?.byteLength ?? 0), 0),
    publicationValid: worklist[1] === source.generation && worklist[3] === 1 && worklist[4] === 1
      && activePages > 0 && taggedMetadataPages >= activePages && malformedActivePages === 0
      && validSamples > 0 && finiteValidSamples === validSamples,
    ...(probedPages.length > 0 ? { probedPages } : {}),
    ...(probedCoarseRecords.length > 0 ? { probedCoarseRecords } : {}),
    ...(coarse ? { coarseState: coarse[0], coarseGeneration: coarse[1],
      coarseHashCapacity: coarse[2], coarseMaximumLeafSize: coarse[3], coarseEntryCount,
      coarseNegativeEntries, coarsePositiveEntries, coarseInterfaceEntries, coarseMalformedEntries } : {}),
    ...(seed ? { seedCount: seed[0], seedFlags: seed[1] } : {}),
    ...(topology ? { topologyFlags: topology[0], topologyInterfaceBricks: topology[1],
      topologyDesiredBricks: topology[2], topologyActivatedBricks: topology[3],
      topologyPublished: topology[4] !== 0, topologyRolledBack: topology[5] !== 0,
      topologyFinalizeReason: topology[7],
      topologyRequiredDesiredBricks: (topology[0] & 1) !== 0 ? topology[6] : topology[2],
      topologyRequiredDesiredBricksExact: (topology[0] & 1) === 0,
      topologyDilationBrickRings: topology[0] === 0 ? topology[6] : 0 } : {}),
    ...(transport ? { transportDepartureOutsideBand: transport[0], transportNonfiniteVelocity: transport[1],
      transportProcessed: transport[2], transportCommitted: transport[3] !== 0,
      transportExtrapolatedVelocity: transport[4],
      transportMaximumDisplacementFineCells: transport[5], transportFaceBandUnavailable: transport[6],
      transportVelocityUnavailable: transport[7] } : {}),
    ...(redistance ? { redistanceUnresolvedCells: redistance[0],
      redistanceMaximumResidualScaled: redistance[1], redistanceSeedCount: redistance[2],
      redistanceCommitted: redistance[3] !== 0 } : {}),
    ...(volume && volumeFloats ? { volumeFlags: volume[0], volumeInitialized: volume[1] !== 0,
      volumeSamples: volume[2], volumeReference: volumeFloats[3], volumeCurrent: volumeFloats[4],
      volumeInterfaceArea: volumeFloats[5], volumeCorrection: volumeFloats[6],
      volumeCorrected: volume[7] !== 0, volumeCoarse: volumeFloats[8], volumeFine: volumeFloats[9],
      volumeReplacedCoarse: volumeFloats[10], volumeCoarseRows: volume[11], volumeUnowned: volume[12],
      volumeExpectedAir: volume[12], volumeGeneration: volume[13],
      volumeLookupFailures: volume[14], volumeStaleOwners: volume[15] } : {}),
    ...(powerVelocity ? { powerVelocityFlags: powerVelocity[0], powerVelocityRows: powerVelocity[2],
      powerVelocityReconstructed: powerVelocity[5], powerVelocityFallbacks: powerVelocity[6] } : {}),
    ...(powerProjection ? { powerProjectionControl: Array.from(powerProjection) } : {}),
    ...(faceBand ? { faceBandFlags: faceBand.flags, faceBandFirstError: faceBand.firstError,
      faceBandRows: faceBand.rowCount, faceBandFaces: faceBand.faceCount,
      faceBandIncidences: faceBand.incidenceCount, faceBandGeneration: faceBand.generation,
      faceBandValid: faceBand.valid, faceBandMaximumDepth: faceBand.maximumDepth,
      faceBandSeeds: faceBand.seedCount, faceBandAccepted: faceBand.acceptedCount,
      faceBandUnresolved: faceBand.unresolvedCount, faceBandSampleFailures: faceBand.sampleFailures,
      faceBandRowCapacity: faceBandPlan?.rowCapacity, faceBandFaceCapacity: faceBandPlan?.faceCapacity } : {}),
  };
}

function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function hybridPresentationBodies(scene: SceneDescription, bodies: RigidBodyState[]): RigidBodyState[] {
  if (bodies.length > 0) return bodies;
  const scale = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  return initializeRigidBodies([{
    id: "hybrid-render-smoke-body", name: "Hybrid render smoke body", shape: "box",
    dimensions_m: { x: 0.18 * scale, y: 0.22 * scale, z: 0.16 * scale }, density_kg_m3: 700,
    position_m: { x: 0.18 * scene.container.width_m, y: 0.36 * scene.container.height_m, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.5, motion: "static"
  }]);
}

async function smokeRenderHybridPresentation(
  device: GPUDevice,
  solver: GPUSolverInstance,
  scene: SceneDescription,
  bodies: RigidBodyState[],
  verifyGlobalFineAuthorityTransition = false,
): Promise<HybridPresentationSmokeStats> {
  // Match the UI's practical pixel density closely enough that a one-pixel
  // slit there cannot disappear through smoke-test undersampling.
  const width = 640, height = 360;
  const uniformBuffer = device.createBuffer({ label: "Hybrid presentation smoke uniforms", size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bodyBuffer = device.createBuffer({ label: "Hybrid presentation smoke bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createTexture({ label: "Hybrid presentation smoke output", size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const presentationBodies = hybridPresentationBodies(scene, bodies).slice(0, 12);
  const span = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  const packed = new Float32Array(100);
  packed.set([width, height, solver.info.submittedTime_s ?? 0, 0], 0);
  packed.set([1.55 * span, 1.12 * span, 1.72 * span, 0], 4);
  packed.set([0, 0.38 * scene.container.height_m, 0, 0], 8);
  packed.set([scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction], 12);
  packed.set([0, scene.voxelDomain.finestCellSize_m, presentationBodies.length, 0], 16);
  packed.set([solver.info.nx, solver.info.ny, solver.info.nz, solver.info.gridKind === "restricted-tall-cell" ? 2 : solver.info.gridKind === "quadtree-tall-cell" || solver.info.gridKind === "octree" ? 3 : 1], 20);
  packed.set([0, 0.5, 0, 0], 24);
  packed.set([environmentIndex(scene.environment ?? "default"), solver.info.lastDt_s ?? 0, solver.info.maxSpeed_m_s ?? 0, 0], 28);
  if (sceneHasTerrain(scene) && scene.terrain) {
    const features = scene.terrain.features.slice(0, MAX_TERRAIN_FEATURES);
    packed.set([1, scene.terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
    features.forEach((feature, index) => {
      packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
      packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
    });
  }
  device.queue.writeBuffer(uniformBuffer, 0, packed);
  const bodyData = new Float32Array(12 * 16);
  const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
  const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
  presentationBodies.forEach((body, index) => {
    const offset = index * 16, d = body.description.dimensions_m;
    const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
    const color = palette[shapeIndex[body.description.shape]];
    bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
    bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
    bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
    bodyData.set([color[0], color[1], color[2], 0], offset + 12);
  });
  device.queue.writeBuffer(bodyBuffer, 0, bodyData);
  const pipeline = new RasterWaterPipeline(device, "rgba8unorm", uniformBuffer, bodyBuffer);
  const uncapturedRendererErrors: string[] = [];
  const onUncapturedRendererError = (event: Event) => {
    const error = (event as Event & { error?: GPUError }).error;
    uncapturedRendererErrors.push(error?.message ?? "unknown uncaptured WebGPU renderer error");
  };
  device.addEventListener("uncapturederror", onUncapturedRendererError);
  device.pushErrorScope("validation");
  let rendererValidationScopeActive = true;
  try {
    const initializeStarted = performance.now();
    await pipeline.initialize();
    const initializeWall_ms = performance.now() - initializeStarted;
    const adaptiveOctree = solver.adaptiveFaceMirrorSource && solver.adaptiveFaceVelocitySource && solver.adaptiveSurfacePageSource
      ? createUnifiedOctreeConsumerSource(solver.adaptiveFaceMirrorSource, solver.adaptiveSurfacePageSource)
      : undefined;
    const globalFineLevelSet = solver.globalFineLevelSetSource
      ? createGlobalFineLevelSetConsumerSource(solver.globalFineLevelSetSource)
      : undefined;
    if (verifyGlobalFineAuthorityTransition && !globalFineLevelSet) {
      throw new Error("Global-fine authority transition requested without a published source");
    }
    pipeline.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture, solver.columnBaseTexture);
    pipeline.setAdaptiveOctree(adaptiveOctree);
    pipeline.setGlobalFineLevelSet(globalFineLevelSet);
    pipeline.ensureSize(width, height);
    const capture = async (label: string, revision: number) => {
      const frameStarted = performance.now();
      const encoder = device.createCommandEncoder({ label });
      const encoded = pipeline.encode(
        encoder, output.createView(), solver.info.nx, solver.info.ny, solver.info.nz,
        solver.info.gridKind === "restricted-tall-cell", solver.info.maximumNeighborDelta ?? 0,
        revision
      );
      if (!encoded) throw new Error("Hybrid presentation pipeline did not encode a frame");
      const interfaceCapture = pipeline.diagnosticCaptureTexture("interface-positions");
      if (!interfaceCapture) throw new Error("Hybrid presentation did not expose its front interface target");
      const backInterfaceCapture = pipeline.diagnosticCaptureTexture("back-interface-positions");
      if (!backInterfaceCapture) throw new Error("Hybrid presentation did not expose its back interface target");
      const interfaceBytesPerRow = Math.ceil(width * 8 / 256) * 256;
      const interfacePlaneBytes = interfaceBytesPerRow * height;
      const interfaceReadback = device.createBuffer({ size: 2 * interfacePlaneBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        encoder.copyTextureToBuffer({ texture: interfaceCapture.texture }, { buffer: interfaceReadback, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        encoder.copyTextureToBuffer({ texture: backInterfaceCapture.texture }, { buffer: interfaceReadback, offset: interfacePlaneBytes, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        device.queue.submit([encoder.finish()]);
        const presentationDiagnostics = await pipeline.completeAdaptiveDiagnostics();
        await device.queue.onSubmittedWorkDone();
        await interfaceReadback.mapAsync(GPUMapMode.READ);
        const interfaceWords = new Uint16Array(interfaceReadback.getMappedRange());
        const interfaceRowWords = interfaceBytesPerRow / 2;
        let frontInterfacePixels = 0, backInterfacePixels = 0, pairedInterfacePixels = 0;
        let frontOnlyInterfacePixels = 0, backOnlyInterfacePixels = 0;
        const backOnlyInterfaceLocations: [number, number][] = [];
        const backOnlyInterfacePositions_m: [number, number, number][] = [];
        let frontInterfaceHash = 0x811c_9dc5, backInterfaceHash = 0x811c_9dc5;
        const fold = (hash: number, value: number) => Math.imul((hash ^ value) >>> 0, 0x0100_0193) >>> 0;
        const frontMinimum: [number, number, number] = [Infinity, Infinity, Infinity];
        const frontMaximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        const wallCornerCapPixels: [number, number, number, number] = [0, 0, 0, 0];
        const damExposedCornerCapPixels: [number, number] = [0, 0];
        const fineCellWidth = globalFineLevelSet?.fineCellWidth ?? scene.voxelDomain.finestCellSize_m;
        const wallPlaneTolerance = Math.max(5e-4, 0.08 * fineCellWidth);
        const cornerTangentialBand = 0.4 * fineCellWidth;
        const dam = damBreakFractions(scene.container.fillFraction);
        const damMaximum = [
          -0.5 * scene.container.width_m + dam.width * scene.container.width_m,
          dam.height * scene.container.height_m,
          -0.5 * scene.container.depth_m + dam.depth * scene.container.depth_m,
        ] as const;
        const wallCorners = [
          [-0.5 * scene.container.width_m, -0.5 * scene.container.depth_m],
          [-0.5 * scene.container.width_m, 0.5 * scene.container.depth_m],
          [0.5 * scene.container.width_m, -0.5 * scene.container.depth_m],
          [0.5 * scene.container.width_m, 0.5 * scene.container.depth_m],
        ] as const;
        const frontMask = new Uint8Array(width * height);
        for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
          const at = y * interfaceRowWords + x * 4;
          const backAt = interfacePlaneBytes / 2 + at;
          for (let channel = 0; channel < 4; channel += 1) {
            frontInterfaceHash = fold(frontInterfaceHash, interfaceWords[at + channel]);
            backInterfaceHash = fold(backInterfaceHash, interfaceWords[backAt + channel]);
          }
          const frontPresent = interfaceWords[at + 3] !== 0;
          const backPresent = interfaceWords[backAt + 3] !== 0;
          if (frontPresent) {
            frontInterfacePixels += 1;
            frontMask[x + y * width] = 1;
            const px = decodeFloat16(interfaceWords[at]);
            const py = decodeFloat16(interfaceWords[at + 1]);
            const pz = decodeFloat16(interfaceWords[at + 2]);
            for (let corner = 0; corner < wallCorners.length; corner += 1) {
              const dx = Math.abs(px - wallCorners[corner][0]);
              const dz = Math.abs(pz - wallCorners[corner][1]);
              if ((dx <= wallPlaneTolerance && dz <= cornerTangentialBand)
                || (dz <= wallPlaneTolerance && dx <= cornerTangentialBand)) wallCornerCapPixels[corner] += 1;
            }
            if (scene.fluid.initialCondition === "dam-break"
              && py >= fineCellWidth && py <= damMaximum[1] - fineCellWidth) {
              const damDx = Math.abs(px - damMaximum[0]);
              const damDz = Math.abs(pz - damMaximum[2]);
              if (damDx <= wallPlaneTolerance && damDz <= cornerTangentialBand) damExposedCornerCapPixels[0] += 1;
              if (damDz <= wallPlaneTolerance && damDx <= cornerTangentialBand) damExposedCornerCapPixels[1] += 1;
            }
            for (let axis = 0; axis < 3; axis += 1) {
              const value = decodeFloat16(interfaceWords[at + axis]);
              frontMinimum[axis] = Math.min(frontMinimum[axis], value);
              frontMaximum[axis] = Math.max(frontMaximum[axis], value);
            }
          }
          if (backPresent) backInterfacePixels += 1;
          if (frontPresent && backPresent) pairedInterfacePixels += 1;
          else if (frontPresent) frontOnlyInterfacePixels += 1;
          else if (backPresent) {
            backOnlyInterfacePixels += 1;
            if (backOnlyInterfaceLocations.length < 16) {
              backOnlyInterfaceLocations.push([x, y]);
              backOnlyInterfacePositions_m.push([
                decodeFloat16(interfaceWords[backAt]),
                decodeFloat16(interfaceWords[backAt + 1]),
                decodeFloat16(interfaceWords[backAt + 2]),
              ]);
            }
          }
        }
        const narrowVerticalSlits = narrowVerticalSlitMetrics(frontMask, width, height);
        interfaceReadback.unmap();
        return { initializeWall_ms, frameWall_ms: performance.now() - frameStarted,
          bodyCount: presentationBodies.length, width, height, frontInterfacePixels, backInterfacePixels,
          pairedInterfacePixels, frontOnlyInterfacePixels, backOnlyInterfacePixels,
          backOnlyInterfaceLocations, backOnlyInterfacePositions_m,
          frontInterfaceHash, backInterfaceHash,
          narrowVerticalSlits,
          wallCornerCapPixels,
          damExposedCornerCapPixels,
          ...(presentationDiagnostics ? {
            surfaceGeometrySource: presentationDiagnostics.surfaceGeometrySource,
            globalFineAuthorityLatch: presentationDiagnostics.globalFineAuthorityLatch,
            globalFineCrossingPublished: presentationDiagnostics.globalFineCrossingPublished,
            presentationFallbackActive: presentationDiagnostics.presentationFallbackActive,
            vertexCount: presentationDiagnostics.vertexCount,
            activeCubeCount: presentationDiagnostics.activeCubeCount,
            vertexAllocator: presentationDiagnostics.vertexAllocator,
            vertexCapacity: surfaceVertexCapacity(...(globalFineLevelSet?.sampleDimensions
              ?? [solver.info.nx, solver.info.ny, solver.info.nz])),
            activeCubeCapacity: activeCubeCapacity(surfaceVertexCapacity(...(globalFineLevelSet?.sampleDimensions
              ?? [solver.info.nx, solver.info.ny, solver.info.nz]))),
          } : {}),
          ...(frontInterfacePixels > 0 ? { frontInterfaceBounds_m: [frontMinimum, frontMaximum] as const } : {}) };
      } finally {
        interfaceReadback.destroy();
      }
    };
    const revision = solver.info.encodedSteps ?? 0;
    const validA = await capture("Hybrid smooth WebGPU smoke", revision);
    let globalFineAuthorityTransition: HybridPresentationSmokeStats["globalFineAuthorityTransition"];
    if (verifyGlobalFineAuthorityTransition && globalFineLevelSet) {
      const unpublishedGeneration = globalFineLevelSet.generation + 1;
      pipeline.setGlobalFineLevelSet({ ...globalFineLevelSet, generation: unpublishedGeneration });
      const invalidB = await capture("Unpublished global-fine generation retention smoke", revision + 1);
      globalFineAuthorityTransition = {
        validGeneration: globalFineLevelSet.generation, unpublishedGeneration, cleanFineCoarseRequired: true,
        retainedGeometrySource: invalidB.surfaceGeometrySource,
        retainedFrontInterfacePixels: invalidB.frontInterfacePixels,
        retainedBackInterfacePixels: invalidB.backInterfacePixels,
        retainedFrontInterfaceHash: invalidB.frontInterfaceHash,
        retainedBackInterfaceHash: invalidB.backInterfaceHash,
        ...(invalidB.frontInterfaceBounds_m ? { retainedFrontInterfaceBounds_m: invalidB.frontInterfaceBounds_m } : {}),
      };
      pipeline.setGlobalFineLevelSet(globalFineLevelSet);
    }
    // Exercise the opposite camera hemisphere as a distinct closure oracle.
    // Missing rear sheets and one-sided winding can be invisible from the
    // default camera even when the scalar and pressure publications are valid.
    packed.set([-1.55 * span, 1.12 * span, -1.72 * span, 0], 4);
    device.queue.writeBuffer(uniformBuffer, 0, packed);
    const reverse = await capture("Hybrid reverse-view closure smoke", revision + 2);
    if (process.env.FLUID_WATER_DIAGNOSTICS === "1") {
      await new Promise((resolve) => setTimeout(resolve, 25));
      console.info(JSON.stringify({ phase: "hybrid-water-diagnostics", ...pipeline.adaptiveRenderDiagnostics }));
    }
    await device.queue.onSubmittedWorkDone();
    const rendererValidationError = await device.popErrorScope();
    rendererValidationScopeActive = false;
    await Promise.resolve();
    const rendererErrors = [
      ...(rendererValidationError ? [rendererValidationError.message] : []),
      ...uncapturedRendererErrors,
    ];
    if (rendererErrors.length > 0) {
      throw new Error(`RasterWaterPipeline production validation failed:\n${rendererErrors.join("\n")}`);
    }
    return { ...validA, reverseView: {
      frontInterfacePixels: reverse.frontInterfacePixels,
      backInterfacePixels: reverse.backInterfacePixels,
      pairedInterfacePixels: reverse.pairedInterfacePixels,
      frontOnlyInterfacePixels: reverse.frontOnlyInterfacePixels,
      backOnlyInterfacePixels: reverse.backOnlyInterfacePixels,
      backOnlyInterfaceLocations: reverse.backOnlyInterfaceLocations,
      backOnlyInterfacePositions_m: reverse.backOnlyInterfacePositions_m,
      frontInterfaceHash: reverse.frontInterfaceHash,
      backInterfaceHash: reverse.backInterfaceHash,
      narrowVerticalSlits: reverse.narrowVerticalSlits,
      wallCornerCapPixels: reverse.wallCornerCapPixels,
      damExposedCornerCapPixels: reverse.damExposedCornerCapPixels,
      ...(reverse.frontInterfaceBounds_m ? { frontInterfaceBounds_m: reverse.frontInterfaceBounds_m } : {}),
    }, rendererValidationErrorCount: 0, rendererUncapturedErrorCount: 0,
      ...(globalFineAuthorityTransition ? { globalFineAuthorityTransition } : {}) };
  } finally {
    if (rendererValidationScopeActive) await device.popErrorScope().catch(() => null);
    device.removeEventListener("uncapturederror", onUncapturedRendererError);
    pipeline.destroy(); output.destroy(); uniformBuffer.destroy(); bodyBuffer.destroy();
  }
}

interface VelocityStageSummary {
  maximum: number;
  liquidMaximum: number;
  location: number[];
  component: number;
  nonFiniteCount: number;
  kineticEnergyProxy: number;
  maximumComponentCfl: number;
  maximumLiquidDivergence_s: number;
  rmsLiquidDivergence_s: number;
}

function gravitationalPotentialEnergyProxy(
  volume: ArrayLike<number>,
  width: number,
  height: number,
  depth: number,
  spacing: { x: number; y: number; z: number },
  gravity: { x: number; y: number; z: number }
) {
  const cellVolume = spacing.x * spacing.y * spacing.z;
  let energy = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const alpha = Math.max(0, Math.min(1, volume[x + width * (y + height * z)]));
    const position = {
      x: (x + 0.5 - width / 2) * spacing.x,
      y: (y + 0.5) * spacing.y,
      z: (z + 0.5 - depth / 2) * spacing.z
    };
    energy -= alpha * (gravity.x * position.x + gravity.y * position.y + gravity.z * position.z) * cellVolume;
  }
  return energy;
}

async function readRgbaTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 16 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth * 4);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + height * z), width * 4);
    output.set(row, width * 4 * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

function summarizeVelocityField(
  velocity: Float32Array,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number,
  divergenceStencil: "backward" | "centered"
): VelocityStageSummary {
  let maximum = 0, liquidMaximum = 0, location = [0, 0, 0], component = 0, nonFiniteCount = 0;
  let kineticEnergyProxy = 0, maximumComponentCfl = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const index = x + width * (y + height * z), value = velocity[3 * index + axis];
      if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
      const speed = Math.abs(value); if (speed > maximum) { maximum = speed; location = [x, y, z]; component = axis; }
      if (volume[index] > 0 && speed > liquidMaximum) liquidMaximum = speed;
      maximumComponentCfl = Math.max(maximumComponentCfl, speed * dt_s / [spacing.x, spacing.y, spacing.z][axis]);
      kineticEnergyProxy += 0.5 * Math.max(0, Math.min(1, volume[index])) * value * value * spacing.x * spacing.y * spacing.z;
    }
  }
  let maximumLiquidDivergence_s = 0, divergenceSquared = 0, liquidCells = 0;
  const at = (x: number, y: number, z: number, axis: number) => velocity[3 * (x + width * (y + height * z)) + axis];
  // Mirror the collocated solver's `centeredFaceVelocity`: the face value is
  // the average of the two adjacent cell centers, and a face whose neighbor
  // is outside the domain carries zero velocity.
  const centered = (x: number, y: number, z: number, axis: number) => {
    const limit = [width, height, depth][axis];
    const coordinate = [x, y, z][axis];
    const own = at(x, y, z, axis);
    const facePlus = coordinate + 1 < limit ? 0.5 * (own + at(axis === 0 ? x + 1 : x, axis === 1 ? y + 1 : y, axis === 2 ? z + 1 : z, axis)) : 0;
    const faceMinus = coordinate > 0 ? 0.5 * (own + at(axis === 0 ? x - 1 : x, axis === 1 ? y - 1 : y, axis === 2 ? z - 1 : z, axis)) : 0;
    return (facePlus - faceMinus) / [spacing.x, spacing.y, spacing.z][axis];
  };
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = x + width * (y + height * z);
    if (!(volume[index] > 1e-4)) continue;
    const divergence = divergenceStencil === "centered"
      ? centered(x, y, z, 0) + centered(x, y, z, 1) + centered(x, y, z, 2)
      : (at(x, y, z, 0) - (x > 0 ? at(x - 1, y, z, 0) : 0)) / spacing.x
        + (at(x, y, z, 1) - (y > 0 ? at(x, y - 1, z, 1) : 0)) / spacing.y
        + (at(x, y, z, 2) - (z > 0 ? at(x, y, z - 1, 2) : 0)) / spacing.z;
    if (!Number.isFinite(divergence)) { nonFiniteCount += 1; continue; }
    maximumLiquidDivergence_s = Math.max(maximumLiquidDivergence_s, Math.abs(divergence));
    divergenceSquared += divergence * divergence; liquidCells += 1;
  }
  return {
    maximum, liquidMaximum, location, component, nonFiniteCount, kineticEnergyProxy, maximumComponentCfl,
    maximumLiquidDivergence_s,
    rmsLiquidDivergence_s: Math.sqrt(divergenceSquared / Math.max(1, liquidCells))
  };
}

async function readVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing = { x: 1, y: 1, z: 1 },
  dt_s = 0
): Promise<VelocityStageSummary> {
  const velocity = await readVelocityField3D(device, texture, width, height, depth);
  return summarizeVelocityField(velocity, width, height, depth, volume, spacing, dt_s, "backward");
}

async function readVelocityField3D(
  device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number,
) {
  const raw = await readRgbaTexture3D(device, texture, width, height, depth);
  const velocity = new Float32Array(width * height * depth * 3);
  for (let index = 0; index < width * height * depth; index += 1) {
    velocity[3 * index] = raw[4 * index];
    velocity[3 * index + 1] = raw[4 * index + 1];
    velocity[3 * index + 2] = raw[4 * index + 2];
  }
  return velocity;
}

/**
 * Reconstruct the cubic velocity field from a packed restricted tall-cell
 * texture (rows 0/1 are the tall endpoint samples; interior rows interpolate
 * linearly between them per paper Eq 5, mirroring `validVelocityCell`) and
 * summarize it with the solver's own centered collocated divergence.
 */
async function readTallVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number
): Promise<VelocityStageSummary> {
  const velocity = await readTallVelocityField3D(device, texture, nx, storedNy, nz, fineNy, bases);
  return summarizeVelocityField(velocity, nx, fineNy, nz, volume, spacing, dt_s, "backward");
}

async function readTallVelocityField3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>
) {
  const raw = await readRgbaTexture3D(device, texture, nx, storedNy, nz);
  const velocity = new Float32Array(nx * fineNy * nz * 3);
  const packedAt = (x: number, packedY: number, z: number, axis: number) => raw[4 * (x + nx * (packedY + storedNy * z)) + axis];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (let y = 0; y < fineNy; y += 1) {
      const index = 3 * (x + nx * (y + fineNy * z));
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, 0, z, axis) * (1 - t) + packedAt(x, 1, z, axis) * t;
      } else {
        const packedY = 2 + y - base;
        if (packedY >= 2 && packedY < storedNy) for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, packedY, z, axis);
      }
    }
  }
  return velocity;
}

async function readCompactOctreeVelocityField3D(
  device: GPUDevice,
  solver: GPUSolverInstance,
  dimensions: readonly [number, number, number],
): Promise<(CompactVelocityRaster & {
  publicationValid: boolean;
  rowCount: number;
  reconstructedRows: number;
}) | undefined> {
  const controlBuffer = solver.globalFinePowerVelocityControl;
  const headerBuffer = solver.powerLeafHeaders;
  const velocityBuffer = solver.powerCellVelocityBuffer;
  if (!controlBuffer || !headerBuffer || !velocityBuffer) return undefined;
  const controlBytes = await readBufferBinding(device, { buffer: controlBuffer }, 32);
  const control = new Uint32Array(controlBytes.buffer, controlBytes.byteOffset, 8);
  const rowCount = control[2] >>> 0, reconstructedRows = control[5] >>> 0;
  if (rowCount === 0 || rowCount * 48 > headerBuffer.size || rowCount * 16 > velocityBuffer.size) return undefined;
  const [headerBytes, velocityBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headerBuffer }, rowCount * 48),
    readBufferBinding(device, { buffer: velocityBuffer }, rowCount * 16),
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const velocities = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset, rowCount * 4);
  return {
    ...rasterizeCompactPowerCellVelocities(headers, velocities, rowCount, dimensions),
    publicationValid: control[0] === OCTREE_POWER_VELOCITY_VALID && reconstructedRows === rowCount,
    rowCount,
    reconstructedRows,
  };
}

function velocityDifferenceMagnitude(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length || left.length % 3 !== 0) throw new Error("Velocity fields must share xyz dimensions");
  const difference = new Float32Array(left.length / 3);
  for (let index = 0; index < difference.length; index += 1) {
    difference[index] = Math.hypot(left[3 * index] - right[3 * index], left[3 * index + 1] - right[3 * index + 1], left[3 * index + 2] - right[3 * index + 2]);
  }
  return difference;
}

async function readFloatTexture2D(device: GPUDevice, texture: GPUTexture, width: number, height: number) {
  const components = texture.format === "rg32float" ? 2 : 1;
  const bytesPerRow = Math.ceil(width * components * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * y, width * components);
    for (let x = 0; x < width; x += 1) output[x + width * y] = row[components * x];
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

function inspectColumnBases(bases: ArrayLike<number>, nx: number, nz: number, fineNy: number, regularLayers: number, maximumDelta: number) {
  const histogram: Record<string, number> = {}, violations: Array<{ a: [number, number, number]; b: [number, number, number]; delta: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height = Math.round(bases[x + nx * z]);
    histogram[height] = (histogram[height] ?? 0) + 1;
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherHeight = Math.round(bases[otherX + nx * otherZ]), delta = Math.abs(height - otherHeight);
      if (delta > maximumDelta && violations.length < 12) violations.push({ a: [x, z, height], b: [otherX, otherZ, otherHeight], delta });
    }
  }
  return { ...summarizeTallCellActivity(bases, fineNy, regularLayers, nx, nz), histogram, violations };
}

function inspectTallVolumeGaps(packed: ArrayLike<number>, bases: ArrayLike<number>, nx: number, storedNy: number, nz: number, fineNy: number, maximumDelta = Infinity) {
  let dryTallColumns = 0, dryTallWithWetRegularAbove = 0, mixedEndpointColumns = 0, wetBandCeilingColumns = 0, unexcusedDeltaViolations = 0;
  const phiAt = (x: number, y: number, z: number) => {
    const base = Math.round(bases[x + nx * z]);
    if (y < base && base > 0) {
      const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
      const bottom = packed[x + nx * storedNy * z];
      const top = packed[x + nx * (1 + storedNy * z)];
      return bottom + (top - bottom) * t;
    }
    const packedY = 2 + y - base;
    return packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : Infinity;
  };
  // Eq. 10 is an unconditional restriction on neighboring band bases now
  // that the signed-distance remap can move the interface without VOF
  // representability floors.
  if (Number.isFinite(maximumDelta)) for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherBase = Math.round(bases[otherX + nx * otherZ]);
      if (Math.abs(base - otherBase) > maximumDelta) unexcusedDeltaViolations += 1;
    }
  }
  const examples: Array<{ x: number; z: number; base: number; bottom: number; top: number; lowestWetWorldY: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    const ceilingWorldY = base + storedNy - 3;
    if (ceilingWorldY < fineNy - 1 && phiAt(x, ceilingWorldY, z) <= 0) wetBandCeilingColumns += 1;
    if (base < 2) continue;
    const bottom = packed[x + nx * storedNy * z];
    const top = packed[x + nx * (1 + storedNy * z)];
    if ((bottom <= 0) !== (top <= 0)) mixedEndpointColumns += 1;
    if (bottom <= 0 || top <= 0) continue;
    dryTallColumns += 1;
    let lowestWetWorldY = -1;
    for (let y = base; y < fineNy; y += 1) if (phiAt(x, y, z) <= 0) {
      lowestWetWorldY = y;
      break;
    }
    if (lowestWetWorldY < 0) continue;
    dryTallWithWetRegularAbove += 1;
    if (examples.length < 12) examples.push({ x, z, base, bottom, top, lowestWetWorldY });
  }
  return { dryTallColumns, dryTallWithWetRegularAbove, mixedEndpointColumns, wetBandCeilingColumns, unexcusedDeltaViolations, examples };
}

interface CubicVolumeFieldReadback {
  field: Float32Array;
  summary: ScalarFieldSummary;
  compactFieldEvidence?: CompactOctreeFieldEvidence;
  tallCellActivity?: TallCellActivitySummary;
  tallVolumeGaps?: ReturnType<typeof inspectTallVolumeGaps>;
}

/**
 * QA forensics: decode the packed owner lattice around the Section-5
 * transition failure cell so descriptor/adjacency disagreements can be traced
 * to the exact stored owner words instead of inferred from control words.
 */
async function dumpOwnerLatticeForensics(
  device: GPUDevice, solver: GPUSolverInstance, dims: readonly [number, number, number],
  transitionControl?: Uint32Array,
): Promise<void> {
  const debug = solver.ownerLatticeDebug;
  if (!debug || !transitionControl || transitionControl.length < 32) return;
  const failureCell = transitionControl[18] >>> 0;
  if (failureCell === 0xffff_ffff) return;
  const [nx, ny, nz] = debug.dimensions ?? dims;
  const centre = [failureCell % nx, Math.floor(failureCell / nx) % ny, Math.floor(failureCell / (nx * ny))];
  const words = new Uint32Array((await readBufferBinding(device, { buffer: debug.buffer }, debug.buffer.size)).buffer);
  const canonical = (q: readonly number[]) => {
    let size = Math.min(debug.maximumLeafSize, 8);
    for (;;) {
      const origin = q.map((v) => Math.floor(v / size) * size);
      if (origin.every((v, a) => v + size <= dims[a])) return { origin, size, source: "canonical" };
      if (size === 1) return { origin: q, size: 0, source: "invalid" };
      size >>= 1;
    }
  };
  const decodeWord = (word: number, q: readonly number[]) => {
    if ((word & 0x8000_0000) !== 0) return { origin: [...q], size: 1, source: "word" };
    const exponent = word & 7;
    if (exponent === 0 || exponent > 5) return canonical(q);
    const size = 1 << exponent;
    return { origin: q.map((v) => Math.floor(v / size) * size), size, source: "word" };
  };
  const ownerAt = (q: readonly number[]) => {
    const dense = q[0] + nx * (q[1] + ny * q[2]);
    if (!debug.paged || words.length <= 15 || words[15] !== 0x4f57_4e52) {
      return { ...decodeWord(words[dense] ?? 0, q), word: words[dense] ?? 0 };
    }
    const freeListOffset = words[5];
    if (freeListOffset <= 16 || ((freeListOffset - 16) & 1) !== 0) return { ...canonical(q), word: -1 };
    const hashCapacity = (freeListOffset - 16) / 2;
    const bd = [Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8)];
    const b = q.map((v) => Math.floor(v / 8));
    const logical = b[0] + b[1] * bd[0] + b[2] * bd[0] * bd[1];
    let slot = (Math.imul(logical, 0x9e3779b1) >>> 0) % hashCapacity;
    let encoded = 0;
    for (let probe = 0; probe < hashCapacity; probe += 1) {
      const observed = words[16 + slot];
      if (observed === logical + 1) { encoded = words[16 + hashCapacity + slot]; break; }
      if (observed === 0) break;
      slot = slot + 1 === hashCapacity ? 0 : slot + 1;
    }
    if (encoded === 0 || encoded === 0xffff_ffff) return { ...canonical(q), word: 0, page: "absent" };
    const local = q.map((v) => v % 8);
    const at = words[6] + (encoded - 1) * 512 + local[0] + local[1] * 8 + local[2] * 64;
    const word = words[at] ?? 0;
    if (word === 0 || word === 0xffff_ffff) return { ...canonical(q), word, page: encoded - 1 };
    return { ...decodeWord(word, q), word, page: encoded - 1 };
  };
  const cells: Record<string, unknown>[] = [];
  for (let dz = -2; dz <= 2; dz += 1) for (let dy = -2; dy <= 3; dy += 1) for (let dx = -2; dx <= 2; dx += 1) {
    const q = [centre[0] + dx, centre[1] + dy, centre[2] + dz];
    if (q.some((v, a) => v < 0 || v >= dims[a])) continue;
    const owner = ownerAt(q);
    cells.push({ q, ...owner, word: typeof owner.word === "number" ? owner.word >>> 0 : owner.word });
  }
  console.error(JSON.stringify({ phase: "owner-lattice-forensics", failureCell, centre,
    paged: debug.paged, maximumLeafSize: debug.maximumLeafSize,
    arenaHeader: Array.from(words.slice(0, 16)), cells }));
}

/**
 * QA forensics: verify the packed owner lattice encodes a partition. Every
 * decoded leaf's cells must all decode to that same leaf; a zero word inside
 * a paged block that also holds written words is an overlap by construction.
 */
async function auditOwnerLatticeConsistency(
  device: GPUDevice, solver: GPUSolverInstance, dims: readonly [number, number, number],
): Promise<Record<string, unknown>[]> {
  const debug = solver.ownerLatticeDebug;
  if (!debug) return [];
  const [nx, ny, nz] = dims;
  const words = new Uint32Array((await readBufferBinding(device, { buffer: debug.buffer }, debug.buffer.size)).buffer);
  const paged = debug.paged && words.length > 15 && words[15] === 0x4f57_4e52;
  const rawWord = (q: readonly number[]): number | undefined => {
    const dense = q[0] + nx * (q[1] + ny * q[2]);
    if (!paged) return words[dense] ?? 0;
    const freeListOffset = words[5];
    if (freeListOffset <= 16 || ((freeListOffset - 16) & 1) !== 0) return undefined;
    const hashCapacity = (freeListOffset - 16) / 2;
    const bd = [Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8)];
    const b = q.map((v) => Math.floor(v / 8));
    const logical = b[0] + b[1] * bd[0] + b[2] * bd[0] * bd[1];
    let slot = (Math.imul(logical, 0x9e3779b1) >>> 0) % hashCapacity;
    for (let probe = 0; probe < hashCapacity; probe += 1) {
      const observed = words[16 + slot];
      if (observed === logical + 1) {
        const encoded = words[16 + hashCapacity + slot];
        if (encoded === 0 || encoded === 0xffff_ffff) return undefined;
        const local = q.map((v) => v % 8);
        return words[words[6] + (encoded - 1) * 512 + local[0] + local[1] * 8 + local[2] * 64] ?? 0;
      }
      if (observed === 0) return undefined;
      slot = slot + 1 === hashCapacity ? 0 : slot + 1;
    }
    return undefined;
  };
  const leafOf = (word: number, q: readonly number[]) => {
    if ((word & 0x8000_0000) !== 0) return { origin: [...q], size: 1 };
    const exponent = word & 7;
    if (exponent >= 1 && exponent <= 5) {
      const size = 1 << exponent;
      return { origin: q.map((v) => Math.floor(v / size) * size), size };
    }
    // zero / invalid: paged decodes canonical (maxLeaf), dense decodes fine.
    if (!paged) return { origin: [...q], size: 1 };
    let size = Math.min(debug.maximumLeafSize, 8);
    for (;;) {
      const origin = q.map((v) => Math.floor(v / size) * size);
      if (origin.every((v, a) => v + size <= dims[a])) return { origin, size };
      size >>= 1;
    }
  };
  const issues: Record<string, unknown>[] = [];
  for (let z = 0; z < nz && issues.length < 64; z += 1) for (let y = 0; y < ny && issues.length < 64; y += 1) for (let x = 0; x < nx; x += 1) {
    const q = [x, y, z];
    const word = rawWord(q);
    if (word === undefined) continue;
    const leaf = leafOf(word, q);
    if (leaf.origin.some((v, a) => v + leaf.size > dims[a])) {
      issues.push({ q, word: word >>> 0, issue: "leaf-exceeds-domain", leaf }); continue;
    }
    if (leaf.size > 1 && (x !== leaf.origin[0] || y !== leaf.origin[1] || z !== leaf.origin[2])) continue;
    for (let dz = 0; dz < leaf.size; dz += 1) for (let dy = 0; dy < leaf.size; dy += 1) for (let dx = 0; dx < leaf.size; dx += 1) {
      const partner = [leaf.origin[0] + dx, leaf.origin[1] + dy, leaf.origin[2] + dz];
      const partnerWord = rawWord(partner);
      const partnerLeaf = partnerWord === undefined
        ? leafOf(0, partner) : leafOf(partnerWord, partner);
      if (partnerLeaf.size !== leaf.size || partnerLeaf.origin.some((v, a) => v !== leaf.origin[a])) {
        issues.push({ q, word: word >>> 0, leaf, partner, partnerWord: partnerWord === undefined ? "absent" : partnerWord >>> 0, partnerLeaf });
        if (issues.length >= 64) return issues;
      }
    }
  }
  return issues;
}

async function readCubicVolumeField(device: GPUDevice, solver: GPUSolverInstance): Promise<CubicVolumeFieldReadback> {
  const { nx, ny, nz, storedNy, gridKind } = solver.info;
  const compactPaged = solver.info.gridKind === "octree" && (
    Boolean(solver.adaptiveSurfacePageSource) || (solver.info.adaptiveSurfacePageCapacity ?? 0) > 0
  );
  if (compactPaged) {
    const source = solver.globalFineLevelSetSource;
    if (!source?.coarsePhiDirectory || !source.coarsePhiHashCapacity) {
      if (requireSpatialField) {
        throw new Error("Compact octree QA field requires a published global-fine source and compact-coarse fallback");
      }
      // Legacy compact-only smoke cases do not request cross-method spatial
      // acceptance. Keep their reduction summary while making the exact
      // comparison harness fail closed via FLUID_REQUIRE_SPATIAL_FIELD=1.
      const cellSum = solver.info.volumeCellSum ?? solver.info.initialVolumeCellSum ?? 0;
      const occupied = Math.max(0, Math.min(nx * ny * nz, Math.round(cellSum)));
      return { field: new Float32Array(0), summary: {
        minimum: 0, maximum: 1, cellSum, wetCells: occupied, mixedCells: solver.info.phiInterfaceCellCount ?? 0,
        excessCells: 0, meanColumnAmount: cellSum / Math.max(1, nx * nz), columnAmountStdDev: 0,
        componentCount: occupied > 0 ? 1 : 0, largestComponent: occupied, interfaceFaceCount: 0,
        enclosedAirComponentCount: 0, enclosedAirCells: 0, centroidCells: null,
      } };
    }
    const sampleWords = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    const [hashBytes, metadataBytes, flagBytes, phiBytes, worklistBytes, coarseBytes, coarseControlBytes,
      fineRestrictionBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes, faceBandBytes,
      faceBandTransitionBytes, faceBandTransientPowerBytes, faceBandPointFieldBytes, faceBandPowerPublicationBytes,
      powerVelocityBytes, powerProjectionBytes, powerVelocitySampleBytes, mgpcgBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.hash }, source.plan.hashCapacity * 8),
      readBufferBinding(device, { buffer: source.metadata }, source.plan.maximumResidentBricks * 40),
      readBufferBinding(device, { buffer: source.flags }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.phi }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.worklist }, (5 + source.plan.maximumResidentBricks) * 4),
      readBufferBinding(device, { buffer: source.coarsePhiDirectory }, 32 + source.coarsePhiHashCapacity * 32),
      solver.globalFineCoarseLevelSetControl
        ? readBufferBinding(device, { buffer: solver.globalFineCoarseLevelSetControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRestrictionControl
        ? readBufferBinding(device, { buffer: solver.globalFineRestrictionControl }, 32)
        : Promise.resolve(undefined),
      source.topologyControl
        ? readBufferBinding(device, { buffer: source.topologyControl }, 32)
        : Promise.resolve(undefined),
      solver.globalFineTransportControl
        ? readBufferBinding(device, { buffer: solver.globalFineTransportControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRedistanceControl
        ? readBufferBinding(device, { buffer: solver.globalFineRedistanceControl }, 16)
        : Promise.resolve(undefined),
      solver.globalFineVolumeControl
        ? readBufferBinding(device, { buffer: solver.globalFineVolumeControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineFaceBandControl
        ? readBufferBinding(device, { buffer: solver.globalFineFaceBandControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineFaceBandTransitionControl
        ? readBufferBinding(device, { buffer: solver.globalFineFaceBandTransitionControl }, 160)
        : Promise.resolve(undefined),
      solver.globalFineFaceBandTransientPowerControl
        ? readBufferBinding(device, { buffer: solver.globalFineFaceBandTransientPowerControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineFaceBandPointFieldControl
        ? readBufferBinding(device, { buffer: solver.globalFineFaceBandPointFieldControl }, 32)
        : Promise.resolve(undefined),
      solver.globalFineFaceBandPowerPublicationControl
        ? readBufferBinding(device, { buffer: solver.globalFineFaceBandPowerPublicationControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFinePowerVelocityControl
        ? readBufferBinding(device, { buffer: solver.globalFinePowerVelocityControl }, 32)
        : Promise.resolve(undefined),
      solver.globalFinePowerProjectionControl
        ? readBufferBinding(device, { buffer: solver.globalFinePowerProjectionControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFinePowerVelocitySampleControl
        ? readBufferBinding(device, { buffer: solver.globalFinePowerVelocitySampleControl }, 32)
        : Promise.resolve(undefined),
      (solver as GPUSolverInstance & { mgpcgControl?: GPUBuffer }).mgpcgControl
        ? readBufferBinding(device, { buffer: (solver as GPUSolverInstance & { mgpcgControl: GPUBuffer }).mgpcgControl }, 64)
        : Promise.resolve(undefined),
    ]);
    const compactSnapshot = {
      plan: source.plan,
      generation: source.generation,
      hash: new Uint32Array(hashBytes.buffer, hashBytes.byteOffset, hashBytes.byteLength / 4),
      metadata: new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4),
      flags: new Uint32Array(flagBytes.buffer, flagBytes.byteOffset, flagBytes.byteLength / 4),
      phi: new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4),
      worklist: new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4),
      coarseDirectory: new Uint32Array(coarseBytes.buffer, coarseBytes.byteOffset, coarseBytes.byteLength / 4),
      ...(coarseControlBytes ? { coarseControl: new Uint32Array(coarseControlBytes.buffer,
        coarseControlBytes.byteOffset, coarseControlBytes.byteLength / 4) } : {}),
      ...(fineRestrictionBytes ? { fineRestrictionControl: new Uint32Array(fineRestrictionBytes.buffer,
        fineRestrictionBytes.byteOffset, fineRestrictionBytes.byteLength / 4) } : {}),
      ...(topologyBytes ? { topologyControl: new Uint32Array(topologyBytes.buffer, topologyBytes.byteOffset,
        topologyBytes.byteLength / 4) } : {}),
      ...(transportBytes ? { transportControl: new Uint32Array(transportBytes.buffer, transportBytes.byteOffset,
        transportBytes.byteLength / 4) } : {}),
      ...(redistanceBytes ? { redistanceControl: new Uint32Array(redistanceBytes.buffer, redistanceBytes.byteOffset,
        redistanceBytes.byteLength / 4) } : {}),
      ...(volumeBytes ? { volumeControl: new Uint32Array(volumeBytes.buffer, volumeBytes.byteOffset,
        volumeBytes.byteLength / 4) } : {}),
      ...(faceBandBytes ? { faceBandControl: new Uint32Array(faceBandBytes.buffer, faceBandBytes.byteOffset,
        faceBandBytes.byteLength / 4) } : {}),
      ...(faceBandTransitionBytes ? { faceBandTransitionControl: new Uint32Array(
        faceBandTransitionBytes.buffer, faceBandTransitionBytes.byteOffset,
        faceBandTransitionBytes.byteLength / 4) } : {}),
      ...(faceBandTransientPowerBytes ? { faceBandTransientPowerControl: new Uint32Array(
        faceBandTransientPowerBytes.buffer, faceBandTransientPowerBytes.byteOffset,
        faceBandTransientPowerBytes.byteLength / 4) } : {}),
      ...(faceBandPointFieldBytes ? { faceBandPointFieldControl: new Uint32Array(
        faceBandPointFieldBytes.buffer, faceBandPointFieldBytes.byteOffset,
        faceBandPointFieldBytes.byteLength / 4) } : {}),
      ...(faceBandPowerPublicationBytes ? { faceBandPowerPublicationControl: new Uint32Array(
        faceBandPowerPublicationBytes.buffer, faceBandPowerPublicationBytes.byteOffset,
        faceBandPowerPublicationBytes.byteLength / 4) } : {}),
      ...(powerVelocityBytes ? { powerVelocityControl: new Uint32Array(powerVelocityBytes.buffer,
        powerVelocityBytes.byteOffset, powerVelocityBytes.byteLength / 4) } : {}),
      ...(powerProjectionBytes ? { powerProjectionControl: new Uint32Array(powerProjectionBytes.buffer,
        powerProjectionBytes.byteOffset, powerProjectionBytes.byteLength / 4) } : {}),
      ...(powerVelocitySampleBytes ? { powerVelocitySampleControl: new Uint32Array(
        powerVelocitySampleBytes.buffer, powerVelocitySampleBytes.byteOffset,
        powerVelocitySampleBytes.byteLength / 4) } : {}),
      ...(mgpcgBytes ? { mgpcgControl: new Uint32Array(mgpcgBytes.buffer,
        mgpcgBytes.byteOffset, mgpcgBytes.byteLength / 4) } : {}),
    };
    let reconstructed: ReturnType<typeof reconstructCompactOctreeOccupancyField>;
    try {
      reconstructed = reconstructCompactOctreeOccupancyField(compactSnapshot, [nx, ny, nz]);
    } catch (error) {
      console.error(JSON.stringify({ phase: "compact-octree-field-publication-rejected", grid: [nx, ny, nz],
        ...compactOctreePublicationHeaderEvidence(compactSnapshot),
        error: error instanceof Error ? error.message : String(error) }));
      const transition = compactSnapshot.faceBandTransitionControl;
      const failureIndex = transition?.[16];
      const failureStage = transition?.[17];
      const disconnected = failureStage === 7 && failureIndex !== undefined
        ? await (solver as GPUSolverInstance & {
          readGlobalFineDisconnectedFaceFailure?: (index: number) => Promise<unknown>;
        }).readGlobalFineDisconnectedFaceFailure?.(failureIndex)
        : undefined;
      if (disconnected) console.error(JSON.stringify({ phase: "face-band-disconnected-forensics", disconnected }));
      await dumpOwnerLatticeForensics(device, solver, [nx, ny, nz], compactSnapshot.faceBandTransitionControl);
      throw error;
    }
    const { field, ...compactFieldEvidence } = reconstructed;
    // Keep the complete publication transaction visible even when compact
    // reconstruction succeeds. Section 5 face-band failures are downstream
    // of the fine/coarse field gate, so hiding these headers on the successful
    // path discards the exact transition tier/detail that must remain
    // fail-closed (in particular the first exact-owner mismatch record).
    console.log(JSON.stringify({ phase: "compact-octree-field-readback", grid: [nx, ny, nz],
      ...compactOctreePublicationHeaderEvidence(compactSnapshot), ...compactFieldEvidence }));
    return { field, summary: summarizeScalarField(field, nx, ny, nz), compactFieldEvidence };
  }
  const levelSet = solver.info.surfaceField === "levelset";
  const packed = await readFloatTexture3D(device, levelSet ? solver.surfaceFieldTexture ?? solver.volumeTexture : solver.volumeTexture, nx, storedNy, nz);
  let bases = new Float32Array(nx * nz);
  if (gridKind === "restricted-tall-cell") bases = await readFloatTexture2D(device, solver.columnBaseTexture, nx, nz);
  const field = new Float32Array(nx * ny * nz);
  const h = solver.info.cellSize_m;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (gridKind !== "restricted-tall-cell") {
      const value = packed[index(x, y, z)];
      field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / (4 * h))) : value;
    }
    else {
      const base = Math.round(bases[x + nx * z]);
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        const bottom = packed[x + nx * storedNy * z];
        const top = packed[x + nx * (1 + storedNy * z)];
        const value = bottom + (top - bottom) * t;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : Math.max(0, value);
      } else {
        const packedY = 2 + y - base;
        const value = packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : 5 * h;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : value;
      }
    }
  }
  return {
    field,
    summary: summarizeScalarField(field, nx, ny, nz),
    tallCellActivity: gridKind === "restricted-tall-cell" ? summarizeTallCellActivity(bases, ny, solver.info.regularLayers, nx, nz) : undefined,
    tallVolumeGaps: gridKind === "restricted-tall-cell" ? inspectTallVolumeGaps(packed, bases, nx, storedNy, nz, ny, solver.info.maximumNeighborDelta) : undefined
  };
}

interface GPUSmokeResult {
  method: string;
  info: GPUEulerianInfo;
  grid: [number, number, number];
  matchedField: Float32Array;
  matchedSummary: ScalarFieldSummary;
  compactFieldEvidence?: CompactOctreeFieldEvidence;
  matchedTallCellActivity?: TallCellActivitySummary;
  finalSummary?: ScalarFieldSummary;
  finalTallCellActivity?: TallCellActivitySummary;
  finalTallVolumeGaps?: ReturnType<typeof inspectTallVolumeGaps>;
  validationErrors: string[];
  construction_ms: number;
  runtime_ms: number;
  /** Solver-loop wall time excluding deliberate full-field QA readbacks. */
  simulationWall_ms: number;
  steps: number;
  /** Accepted steps whose live power topology, faces, transfer, and MGPCG publication passed the generation audit. */
  powerGenerationAuditedSteps: number;
  /** Historical transition telemetry for diagnosing when adaptivity was lost.
   * It must never substitute for a live final-topology assertion. */
  powerTransitionWitness?: PowerTransitionWitness;
  velocitySummary?: VelocityStageSummary;
  /** Final collocated cubic velocity and exactly aligned occupancy for the
   * dam-break octree/tall-cell parity gate. */
  velocityParityField?: Float32Array;
  velocityParityVolume?: Float32Array;
  compactVelocityRaster?: Omit<CompactVelocityRaster, "field"> & {
    publicationValid: boolean;
    rowCount: number;
    reconstructedRows: number;
  };
  initialFluidBrickStats?: FluidBrickSnapshot;
  sparseVoxelStats?: SparseVoxelSmokeStats;
  hybridPresentationStats?: HybridPresentationSmokeStats;
  initialGlobalFineGeneration?: GlobalFineGenerationDiagnostics;
  initialGlobalFineRaster?: HybridPresentationSmokeStats;
  finalGlobalFineGeneration?: GlobalFineGenerationDiagnostics;
  finalGlobalFineRaster?: HybridPresentationSmokeStats;
  octreeFaceMirrorDiagnostics?: OctreeFaceMirrorDiagnostics;
  octreePowerFaceTransferDiagnostics?: OctreePowerFaceTransferDiagnostics;
  octreePowerFaceDiagnostics?: OctreePowerFaceDiagnostics;
  octreePowerTopologyDiagnostics?: OctreePowerTopologyDiagnostics;
  octreeMGPCGDiagnostics?: OctreeMGPCGDiagnostics;
  powerProjectionDiagnostics?: PowerProjectionDiagnostics;
  stabilityEnvelope?: StabilityEnvelope;
  energyTrace: MechanicalEnergySample[];
  checkpoints: Array<{
    time_s: number;
    field: Float32Array;
    summary: ScalarFieldSummary;
    /** GPU raster result sampled for QA only; never feeds the simulation. */
    raster?: HybridPresentationSmokeStats;
    globalFineGeneration?: GlobalFineGenerationDiagnostics;
    preProjectionVelocity?: Float32Array;
    postProjectionVelocity?: Float32Array;
  }>;
}

interface OctreePowerTopologyDiagnostics {
  descriptor: { rowCount: number; validCount: number; errorCount: number; firstInvalid: number; flags: number;
    sameOrFinerCount: number; sameOrCoarserCount: number; generation: number };
  topology: { invalidCount: number; firstInvalid: number; flags: number; resolvedCount: number; version: number };
  firstInvalidRow?: { row: number; descriptor: number; topologyCode: number; transformAndFlags: number;
    volume: number; reserved: number; cell: number; size: number;
    ownerNeighborhood?: Array<{ direction: number; probe: [number, number, number]; origin: [number, number, number]; size: number; invalid: boolean }> };
}

interface PowerProjectionDiagnostics {
  rowCount: number;
  faceCount: number;
  leafSizeHistogram: Record<string, number>;
  transitionSizePairHistogram: Record<string, number>;
  maximumTransitionSizeRatio: number;
  transitionFaceCount: number;
  obliqueTransitionFaceCount: number;
  maximumTransitionNormalVelocity_m_s: number;
  topology: {
    validRowCount: number;
    invalidRowCount: number;
    rowsWithTetrahedra: number;
    transitionRowCount: number;
    transitionTetrahedronCount: number;
  };
  geometry: { invalidFaceCount: number; maximumNormalLengthError: number };
  incidence: { incidenceCount: number; invalidEntryCount: number; reciprocityFailureCount: number };
  pressureRows: {
    finiteRowCount: number;
    invalidRowCount: number;
    entryCount: number;
    reciprocityFailureCount: number;
    /** PCG solves A q = -storedFluxRhs. */
    relativeResidual: number;
    /** Residual obtained by substituting q=dt*|g|*(H-y) into the published rows. */
    analyticRelativeResidual: number;
    /** A*q_analytic plus an independently reconstructed dt*g.n face-flux RHS. */
    analyticGravityRelativeResidual: number;
    /** Difference between assembled RHS and independent dt*g.n face fluxes. */
    gravityRhsRelativeError: number;
    analyticInteriorRelativeResidual: number;
    analyticOpenBoundaryRelativeResidual: number;
    analyticProjectionMaximum_m_s: number;
    worstAnalyticFace: { face: number; negative: number; positive: number; flags: number; area: number;
      inverseDistance: number; openFraction: number; normal: [number, number, number]; predicted: number;
      pressureGradient: number; remaining: number };
    worstAnalyticRow: { row: number; cell: number; size: number; centerY: number; residual: number;
      rhs: number; gravityRhs: number; analyticApplied: number; diagonal: number; openBoundary: boolean };
  };
  velocityReconstruction?: {
    flags: number;
    rowCount: number;
    faceCount: number;
    incidenceCount: number;
    reconstructedCount: number;
    fallbackCount: number;
  };
  operator?: {
    flags: number;
    firstError: number;
    rowCount: number;
    faceCount: number;
    incidenceCount: number;
    entryCount: number;
    projectedCount: number;
  };
  pressurePotential: {
    dt_s: number;
    relativeL2Error: number;
    maximumAbsoluteError_m2_s: number;
    maximumAbsolutePressureError_Pa: number;
    maximumExpected_m2_s: number;
    maximumObserved_m2_s: number;
  };
  maximumSpeed_m_s: number;
  maximumDivergence_s: number;
  /** Final generalized-face velocity norm in the pressure operator's H metric. */
  metricKineticEnergyProxy: number;
  volumeDrift: number;
}

interface PowerTransitionWitness {
  step: number;
  generation: number;
  leafSizeHistogram: Record<string, number>;
  transitionFaceCount: number;
  obliqueTransitionFaceCount: number;
  maximumTransitionSizeRatio: number;
  transitionRowCount: number;
  transitionTetrahedronCount: number;
}

async function readPowerTransitionWitness(
  device: GPUDevice,
  solver: GPUSolverInstance,
  step: number,
  generation: number,
  rowCount: number,
  faceCount: number,
): Promise<PowerTransitionWitness | undefined> {
  const headersBuffer = solver.powerLeafHeaders;
  const debug = solver.octreeTechniqueDebugSource;
  if (!headersBuffer || !debug || rowCount === 0 || faceCount === 0) return undefined;
  const [headerBytes, faceBytes, normalBytes, metricBytes, tetraHeaderBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headersBuffer }, rowCount * 48),
    readBufferBinding(device, debug.powerFaces, faceCount * OCTREE_POWER_FACE_RECORD_BYTES),
    readBufferBinding(device, debug.faceNormals, faceCount * 16),
    readBufferBinding(device, debug.topologyMetrics, rowCount * 16),
    readBufferBinding(device, debug.tetrahedronHeaders, debug.tetrahedronHeaders.buffer.size),
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const faces = new Uint32Array(faceBytes.buffer, faceBytes.byteOffset, faceCount * 8);
  const normals = new Float32Array(normalBytes.buffer, normalBytes.byteOffset, faceCount * 4);
  const metrics = new Uint32Array(metricBytes.buffer, metricBytes.byteOffset, rowCount * 4);
  const tetraHeaders = new Uint32Array(tetraHeaderBytes.buffer, tetraHeaderBytes.byteOffset,
    tetraHeaderBytes.byteLength / 4);
  const leafSizeHistogram: Record<string, number> = {};
  for (let row = 0; row < rowCount; row += 1) {
    const size = headers[row * 12 + 3];
    leafSizeHistogram[String(size)] = (leafSizeHistogram[String(size)] ?? 0) + 1;
  }
  let transitionFaceCount = 0, obliqueTransitionFaceCount = 0, maximumTransitionSizeRatio = 1;
  for (let face = 0; face < faceCount; face += 1) {
    const negative = faces[face * 8], positive = faces[face * 8 + 1];
    if (negative >= rowCount || positive >= rowCount) continue;
    const negativeSize = headers[negative * 12 + 3], positiveSize = headers[positive * 12 + 3];
    if (negativeSize === positiveSize) continue;
    transitionFaceCount += 1;
    maximumTransitionSizeRatio = Math.max(maximumTransitionSizeRatio,
      Math.max(negativeSize, positiveSize) / Math.max(1, Math.min(negativeSize, positiveSize)));
    const dominant = Math.max(Math.abs(normals[face * 4]), Math.abs(normals[face * 4 + 1]),
      Math.abs(normals[face * 4 + 2]));
    if (dominant < 1 - 1e-5) obliqueTransitionFaceCount += 1;
  }
  let transitionRowCount = 0, transitionTetrahedronCount = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const topologyCode = metrics[row * 4], topologyFlags = metrics[row * 4 + 1];
    const tetraOffset = topologyCode * 3;
    if (topologyCode === OCTREE_POWER_INVALID_ROW || tetraOffset + 2 >= tetraHeaders.length
      || (topologyFlags & 0x8000_0000) === 0) continue;
    const tetraCount = tetraHeaders[tetraOffset + 1];
    if (tetraCount > 0 && (tetraHeaders[tetraOffset + 2] & 1) === 0) {
      transitionRowCount += 1;
      transitionTetrahedronCount += tetraCount;
    }
  }
  if ((leafSizeHistogram["1"] ?? 0) === 0 || (leafSizeHistogram["2"] ?? 0) === 0
    || transitionFaceCount === 0 || obliqueTransitionFaceCount === 0
    || transitionRowCount === 0 || transitionTetrahedronCount === 0) return undefined;
  return { step, generation, leafSizeHistogram, transitionFaceCount, obliqueTransitionFaceCount,
    maximumTransitionSizeRatio, transitionRowCount, transitionTetrahedronCount };
}

async function readPowerProjectionDiagnostics(
  device: GPUDevice,
  solver: GPUSolverInstance,
  scene: SceneDescription,
  faceDiagnostics: OctreePowerFaceDiagnostics | undefined,
): Promise<PowerProjectionDiagnostics | undefined> {
  const rowCount = faceDiagnostics?.rowCount ?? 0;
  const faceCount = faceDiagnostics?.faceCount ?? 0;
  const headersBuffer = solver.powerLeafHeaders;
  const entriesBuffer = solver.powerLeafEntries;
  const pressureBuffer = solver.powerPressureBuffer;
  const debug = solver.octreeTechniqueDebugSource;
  if (rowCount === 0 || faceCount === 0 || !headersBuffer || !entriesBuffer || !pressureBuffer || !debug) return undefined;
  const structuralSolver = solver as GPUSolverInstance & { powerOperatorControl?: GPUBuffer };
  const incidenceCount = faceDiagnostics?.incidenceCount ?? 0;
  const [headerBytes, pressureBytes, faceBytes, normalBytes, centroidBytes, metricBytes, tetraHeaderBytes,
    incidenceRowBytes, incidenceBytes, velocityControlBytes, operatorControlBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headersBuffer }, rowCount * 48),
    readBufferBinding(device, { buffer: pressureBuffer }, rowCount * 4),
    readBufferBinding(device, debug.powerFaces, faceCount * OCTREE_POWER_FACE_RECORD_BYTES),
    readBufferBinding(device, debug.faceNormals, faceCount * 16),
    readBufferBinding(device, debug.faceCentroids, faceCount * 16),
    readBufferBinding(device, debug.topologyMetrics, rowCount * 16),
    readBufferBinding(device, debug.tetrahedronHeaders, debug.tetrahedronHeaders.buffer.size),
    readBufferBinding(device, debug.incidenceRows, (rowCount + 1) * 16),
    readBufferBinding(device, debug.incidence, incidenceCount * 8),
    solver.globalFinePowerVelocityControl
      ? readBufferBinding(device, { buffer: solver.globalFinePowerVelocityControl }, 32) : Promise.resolve(undefined),
    structuralSolver.powerOperatorControl
      ? readBufferBinding(device, { buffer: structuralSolver.powerOperatorControl }, 64) : Promise.resolve(undefined),
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const headerFloats = new Float32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const pressure = new Float32Array(pressureBytes.buffer, pressureBytes.byteOffset, rowCount);
  const faces = new Uint32Array(faceBytes.buffer, faceBytes.byteOffset, faceCount * 8);
  const faceFloats = new Float32Array(faceBytes.buffer, faceBytes.byteOffset, faceCount * 8);
  const normals = new Float32Array(normalBytes.buffer, normalBytes.byteOffset, faceCount * 4);
  const centroids = new Float32Array(centroidBytes.buffer, centroidBytes.byteOffset, faceCount * 4);
  const metrics = new Uint32Array(metricBytes.buffer, metricBytes.byteOffset, rowCount * 4);
  const metricFloats = new Float32Array(metricBytes.buffer, metricBytes.byteOffset, rowCount * 4);
  const tetraHeaders = new Uint32Array(tetraHeaderBytes.buffer, tetraHeaderBytes.byteOffset,
    tetraHeaderBytes.byteLength / 4);
  const incidenceRows = new Uint32Array(incidenceRowBytes.buffer, incidenceRowBytes.byteOffset,
    (rowCount + 1) * 4);
  const incidences = new Uint32Array(incidenceBytes.buffer, incidenceBytes.byteOffset, incidenceCount * 2);
  const incidenceSigns = new Int32Array(incidenceBytes.buffer, incidenceBytes.byteOffset, incidenceCount * 2);
  const histogram: Record<string, number> = {};
  const sizes = new Uint32Array(rowCount);
  const dt_s = solver.info.lastDt_s ?? scene.numerics.maxDt_s;
  const h = scene.container.height_m / solver.info.ny;
  const surfaceY = scene.container.fillFraction * scene.container.height_m;
  const gravity = Math.abs(scene.fluid.gravity_m_s2.y);
  let squaredError = 0, squaredReference = 0, maximumAbsoluteError = 0;
  let maximumExpected = 0, maximumObserved = 0;
  const expectedPressure = new Float64Array(rowCount);
  let finitePressureRows = 0, invalidPressureRows = 0, entryCount = 0;
  let validTopologyRows = 0, invalidTopologyRows = 0, rowsWithTetrahedra = 0;
  let transitionRowCount = 0, transitionTetrahedronCount = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const cell = headers[row * 12], size = headers[row * 12 + 3];
    sizes[row] = size;
    histogram[String(size)] = (histogram[String(size)] ?? 0) + 1;
    const y = Math.floor(cell / solver.info.nx) % solver.info.ny;
    const centerY = (y + 0.5 * size) * h;
    const expected = dt_s * gravity * Math.max(0, surfaceY - centerY);
    expectedPressure[row] = expected;
    const observed = pressure[row];
    const error = observed - expected;
    const weight = metricFloats[row * 4 + 2] * size ** 3;
    squaredError += weight * error * error;
    squaredReference += weight * expected * expected;
    maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(error));
    maximumExpected = Math.max(maximumExpected, expected);
    maximumObserved = Math.max(maximumObserved, Math.abs(observed));
    const diagonal = headerFloats[row * 12 + 4], rhs = headerFloats[row * 12 + 5];
    const start = headers[row * 12 + 1], count = headers[row * 12 + 2];
    entryCount = Math.max(entryCount, start + count);
    if (Number.isFinite(observed) && Number.isFinite(diagonal) && diagonal > 0 && Number.isFinite(rhs)) {
      finitePressureRows += 1;
    } else invalidPressureRows += 1;
    const topologyCode = metrics[row * 4], topologyFlags = metrics[row * 4 + 1];
    const tetraOffset = topologyCode * 3;
    const tetraCount = topologyCode !== OCTREE_POWER_INVALID_ROW && tetraOffset + 2 < tetraHeaders.length
      ? tetraHeaders[tetraOffset + 1] : 0;
    if ((topologyFlags & 0x8000_0000) !== 0 && tetraCount > 0) {
      validTopologyRows += 1; rowsWithTetrahedra += 1;
      if ((tetraHeaders[tetraOffset + 2] & 1) === 0) {
        transitionRowCount += 1; transitionTetrahedronCount += tetraCount;
      }
    } else invalidTopologyRows += 1;
  }
  const entryBytes = entryCount > 0
    ? await readBufferBinding(device, { buffer: entriesBuffer }, entryCount * 8) : new Uint8Array();
  const entries = new Uint32Array(entryBytes.buffer, entryBytes.byteOffset, entryCount * 2);
  const entryFloats = new Float32Array(entryBytes.buffer, entryBytes.byteOffset, entryCount * 2);
  const pressureEntryMap = new Map<string, number>();
  let pressureEntryInvalid = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const start = headers[row * 12 + 1], count = headers[row * 12 + 2], seen = new Set<number>();
    for (let local = 0; local < count; local += 1) {
      const index = start + local, neighbor = entries[index * 2], coefficient = entryFloats[index * 2 + 1];
      if (neighbor >= rowCount || neighbor === row || seen.has(neighbor)
        || !Number.isFinite(coefficient) || coefficient <= 0) pressureEntryInvalid += 1;
      else { seen.add(neighbor); pressureEntryMap.set(`${row}:${neighbor}`, coefficient); }
    }
  }
  let pressureReciprocityFailures = pressureEntryInvalid;
  for (const [key, coefficient] of pressureEntryMap) {
    const [row, neighbor] = key.split(":");
    const reverse = pressureEntryMap.get(`${neighbor}:${row}`);
    if (reverse === undefined || Math.abs(reverse - coefficient) > 2e-4 * Math.max(1, Math.abs(coefficient))) {
      pressureReciprocityFailures += 1;
    }
  }
  const gravityRhs = new Float64Array(rowCount);
  const hasOpenBoundary = new Uint8Array(rowCount);
  let analyticProjectionMaximum = 0;
  let worstAnalyticFace = { face: 0, negative: 0, positive: 0, flags: 0, area: 0,
    inverseDistance: 0, openFraction: 0, normal: [0, 0, 0] as [number, number, number],
    predicted: 0, pressureGradient: 0, remaining: 0 };
  for (let face = 0; face < faceCount; face += 1) {
    const negative = faces[face * 8], positive = faces[face * 8 + 1], flags = faces[face * 8 + 3];
    const area = faceFloats[face * 8 + 5];
    if (negative >= rowCount || !Number.isFinite(area)) continue;
    const closed = positive === OCTREE_POWER_INVALID_ROW && (flags & 1) !== 0 && (flags & 2) === 0;
    if (positive === OCTREE_POWER_INVALID_ROW && (flags & 2) !== 0) hasOpenBoundary[negative] = 1;
    const velocity = closed ? 0 : dt_s * (
      scene.fluid.gravity_m_s2.x * normals[face * 4]
      + scene.fluid.gravity_m_s2.y * normals[face * 4 + 1]
      + scene.fluid.gravity_m_s2.z * normals[face * 4 + 2]
    );
    const flux = area * velocity;
    gravityRhs[negative] += flux;
    if (positive !== OCTREE_POWER_INVALID_ROW && positive < rowCount) gravityRhs[positive] -= flux;
    const inverseDistance = faceFloats[face * 8 + 6], openFraction = faceFloats[face * 8 + 7];
    const pressureGradient = closed ? 0 : ((positive === OCTREE_POWER_INVALID_ROW ? 0 : expectedPressure[positive])
      - expectedPressure[negative]) * inverseDistance * openFraction;
    const remaining = velocity - pressureGradient;
    if (Math.abs(remaining) > analyticProjectionMaximum) {
      analyticProjectionMaximum = Math.abs(remaining);
      worstAnalyticFace = { face, negative, positive, flags, area, inverseDistance, openFraction,
        normal: [normals[face * 4], normals[face * 4 + 1], normals[face * 4 + 2]],
        predicted: velocity, pressureGradient, remaining };
    }
  }
  let residualSquared = 0, analyticResidualSquared = 0, analyticGravityResidualSquared = 0;
  let gravityRhsErrorSquared = 0, rhsSquared = 0, gravityRhsSquared = 0;
  let analyticInteriorResidualSquared = 0, analyticOpenBoundaryResidualSquared = 0;
  let worstAnalyticRow = { row: 0, cell: 0, size: 0, centerY: 0, residual: 0,
    rhs: 0, gravityRhs: 0, analyticApplied: 0, diagonal: 0, openBoundary: false };
  for (let row = 0; row < rowCount; row += 1) {
    let applied = headerFloats[row * 12 + 4] * pressure[row];
    let analyticApplied = headerFloats[row * 12 + 4] * expectedPressure[row];
    const start = headers[row * 12 + 1], count = headers[row * 12 + 2];
    for (let local = 0; local < count; local += 1) {
      const index = start + local, neighbor = entries[index * 2];
      if (neighbor < rowCount) {
        const coefficient = entryFloats[index * 2 + 1];
        applied -= coefficient * pressure[neighbor];
        analyticApplied -= coefficient * expectedPressure[neighbor];
      }
    }
    // emitPowerRows stores the integrated predicted flux. Projection adds
    // A*q to that flux, so both MGPCG and the paper-facing equation solve
    // A*q = -storedFluxRhs.
    const rhs = headerFloats[row * 12 + 5], residual = applied + rhs;
    const analyticResidual = analyticApplied + rhs;
    const analyticGravityResidual = analyticApplied + gravityRhs[row];
    residualSquared += residual * residual;
    analyticResidualSquared += analyticResidual * analyticResidual;
    analyticGravityResidualSquared += analyticGravityResidual * analyticGravityResidual;
    if (hasOpenBoundary[row] !== 0) analyticOpenBoundaryResidualSquared += analyticGravityResidual ** 2;
    else analyticInteriorResidualSquared += analyticGravityResidual ** 2;
    if (Math.abs(analyticGravityResidual) > Math.abs(worstAnalyticRow.residual)) {
      const cell = headers[row * 12], size = headers[row * 12 + 3];
      const y = Math.floor(cell / solver.info.nx) % solver.info.ny;
      worstAnalyticRow = { row, cell, size, centerY: (y + 0.5 * size) * h,
        residual: analyticGravityResidual, rhs, gravityRhs: gravityRhs[row], analyticApplied,
        diagonal: headerFloats[row * 12 + 4], openBoundary: hasOpenBoundary[row] !== 0 };
    }
    gravityRhsErrorSquared += (rhs - gravityRhs[row]) ** 2;
    rhsSquared += rhs * rhs;
    gravityRhsSquared += gravityRhs[row] ** 2;
  }
  let transitionFaceCount = 0, obliqueTransitionFaceCount = 0, invalidFaceCount = 0;
  const transitionSizePairHistogram: Record<string, number> = {};
  let maximumTransitionSizeRatio = 1;
  let maximumNormalLengthError = 0, maximumTransitionNormalVelocity = 0, maximumFaceNormalVelocity = 0;
  let metricKineticEnergyProxy = 0;
  const faceIncidenceCount = new Uint32Array(faceCount), faceIncidenceSignSum = new Int32Array(faceCount);
  const integratedFlux = new Float64Array(rowCount);
  let invalidIncidenceEntries = 0;
  for (let face = 0; face < faceCount; face += 1) {
    const negative = faces[face * 8], positive = faces[face * 8 + 1];
    const area = faceFloats[face * 8 + 5], inverseDistance = faceFloats[face * 8 + 6];
    const openFraction = faceFloats[face * 8 + 7], normalVelocity = faceFloats[face * 8 + 4];
    const nx = normals[face * 4], ny = normals[face * 4 + 1], nz = normals[face * 4 + 2];
    const normalLengthError = Math.abs(Math.hypot(nx, ny, nz) - 1);
    maximumNormalLengthError = Math.max(maximumNormalLengthError, normalLengthError);
    maximumFaceNormalVelocity = Math.max(maximumFaceNormalVelocity, Math.abs(normalVelocity));
    const centroidFinite = Number.isFinite(centroids[face * 4]) && Number.isFinite(centroids[face * 4 + 1])
      && Number.isFinite(centroids[face * 4 + 2]);
    if (negative >= rowCount || (positive !== OCTREE_POWER_INVALID_ROW && positive >= rowCount)
      || !Number.isFinite(area) || area <= 0 || !Number.isFinite(inverseDistance) || inverseDistance <= 0
      || !Number.isFinite(openFraction) || openFraction < 0 || openFraction > 1
      || !Number.isFinite(normalVelocity) || !centroidFinite
      || !Number.isFinite(normalLengthError) || normalLengthError > 2e-4) {
      invalidFaceCount += 1;
    }
    if (negative < rowCount && Number.isFinite(area) && Number.isFinite(openFraction) && Number.isFinite(normalVelocity)) {
      // The solid-face pass has already folded the aperture into the compact
      // normal velocity. Match emitPowerRows/computePowerDivergence by using
      // the power-face area once; multiplying by openFraction here again
      // understates the projected flux residual on cut faces.
      const flux = compactPowerFaceIntegratedFlux(area, normalVelocity);
      integratedFlux[negative] += flux;
      if (positive !== OCTREE_POWER_INVALID_ROW && positive < rowCount) integratedFlux[positive] -= flux;
    }
    const metricEnergy = compactPowerFaceMetricKineticEnergy(area, inverseDistance, openFraction, normalVelocity);
    metricKineticEnergyProxy = Number.isFinite(metricKineticEnergyProxy) && Number.isFinite(metricEnergy)
      ? metricKineticEnergyProxy + metricEnergy : Infinity;
    if (negative >= rowCount || positive === OCTREE_POWER_INVALID_ROW || positive >= rowCount
      || sizes[negative] === sizes[positive]) continue;
    transitionFaceCount += 1;
    const smaller = Math.min(sizes[negative], sizes[positive]);
    const larger = Math.max(sizes[negative], sizes[positive]);
    const pair = `${smaller}:${larger}`;
    transitionSizePairHistogram[pair] = (transitionSizePairHistogram[pair] ?? 0) + 1;
    maximumTransitionSizeRatio = Math.max(maximumTransitionSizeRatio, larger / Math.max(1, smaller));
    maximumTransitionNormalVelocity = Math.max(maximumTransitionNormalVelocity, Math.abs(normalVelocity));
    const dominant = Math.max(Math.abs(nx), Math.abs(ny), Math.abs(nz));
    if (dominant < 1 - 1e-5) obliqueTransitionFaceCount += 1;
  }
  for (let row = 0; row < rowCount; row += 1) {
    const count = incidenceRows[row * 4 + 1], start = incidenceRows[row * 4 + 3];
    if (start > incidenceCount || count > incidenceCount - start) { invalidIncidenceEntries += 1; continue; }
    for (let local = 0; local < count; local += 1) {
      const index = start + local, face = incidences[index * 2], sign = incidenceSigns[index * 2 + 1];
      if (face >= faceCount || (sign !== 1 && sign !== -1)) { invalidIncidenceEntries += 1; continue; }
      const expected = faces[face * 8] === row ? 1 : faces[face * 8 + 1] === row ? -1 : 0;
      if (sign !== expected) { invalidIncidenceEntries += 1; continue; }
      faceIncidenceCount[face] += 1; faceIncidenceSignSum[face] += sign;
    }
  }
  let incidenceReciprocityFailures = 0;
  for (let face = 0; face < faceCount; face += 1) {
    const boundary = faces[face * 8 + 1] === OCTREE_POWER_INVALID_ROW;
    if (faceIncidenceCount[face] !== (boundary ? 1 : 2)
      || faceIncidenceSignSum[face] !== (boundary ? 1 : 0)) incidenceReciprocityFailures += 1;
  }
  let maximumPowerDivergence = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const size = sizes[row], physicalVolume = metricFloats[row * 4 + 2] * (size * h) ** 3;
    const divergence = integratedFlux[row] / physicalVolume;
    maximumPowerDivergence = Math.max(maximumPowerDivergence,
      Number.isFinite(divergence) ? Math.abs(divergence) : Infinity);
  }
  const velocityControl = velocityControlBytes
    ? new Uint32Array(velocityControlBytes.buffer, velocityControlBytes.byteOffset, 8) : undefined;
  const operatorControl = operatorControlBytes
    ? new Uint32Array(operatorControlBytes.buffer, operatorControlBytes.byteOffset, 16) : undefined;
  return {
    rowCount, faceCount, leafSizeHistogram: histogram, transitionSizePairHistogram,
    maximumTransitionSizeRatio, transitionFaceCount, obliqueTransitionFaceCount,
    maximumTransitionNormalVelocity_m_s: maximumTransitionNormalVelocity,
    topology: { validRowCount: validTopologyRows, invalidRowCount: invalidTopologyRows,
      rowsWithTetrahedra, transitionRowCount, transitionTetrahedronCount },
    geometry: { invalidFaceCount, maximumNormalLengthError },
    incidence: { incidenceCount, invalidEntryCount: invalidIncidenceEntries,
      reciprocityFailureCount: incidenceReciprocityFailures },
    pressureRows: { finiteRowCount: finitePressureRows, invalidRowCount: invalidPressureRows,
      entryCount, reciprocityFailureCount: pressureReciprocityFailures,
      relativeResidual: Math.sqrt(residualSquared / Math.max(rhsSquared, 1e-30)),
      analyticRelativeResidual: Math.sqrt(analyticResidualSquared / Math.max(rhsSquared, 1e-30)),
      analyticGravityRelativeResidual: Math.sqrt(analyticGravityResidualSquared / Math.max(gravityRhsSquared, 1e-30)),
      gravityRhsRelativeError: Math.sqrt(gravityRhsErrorSquared / Math.max(gravityRhsSquared, 1e-30)),
      analyticInteriorRelativeResidual: Math.sqrt(analyticInteriorResidualSquared / Math.max(gravityRhsSquared, 1e-30)),
      analyticOpenBoundaryRelativeResidual: Math.sqrt(analyticOpenBoundaryResidualSquared / Math.max(gravityRhsSquared, 1e-30)),
      analyticProjectionMaximum_m_s: analyticProjectionMaximum, worstAnalyticFace,
      worstAnalyticRow },
    velocityReconstruction: velocityControl ? { flags: velocityControl[0], rowCount: velocityControl[2],
      faceCount: velocityControl[3], incidenceCount: velocityControl[4], reconstructedCount: velocityControl[5],
      fallbackCount: velocityControl[6] } : undefined,
    operator: operatorControl ? { flags: operatorControl[0], firstError: operatorControl[1],
      rowCount: operatorControl[2], faceCount: operatorControl[3], incidenceCount: operatorControl[4],
      entryCount: operatorControl[5], projectedCount: operatorControl[6] } : undefined,
    pressurePotential: {
      dt_s,
      relativeL2Error: Math.sqrt(squaredError / Math.max(squaredReference, 1e-30)),
      maximumAbsoluteError_m2_s: maximumAbsoluteError,
      maximumAbsolutePressureError_Pa: maximumAbsoluteError * scene.fluid.density_kg_m3 / Math.max(dt_s, 1e-30),
      maximumExpected_m2_s: maximumExpected,
      maximumObserved_m2_s: maximumObserved,
    },
    maximumSpeed_m_s: maximumFaceNormalVelocity,
    maximumDivergence_s: maximumPowerDivergence,
    metricKineticEnergyProxy,
    volumeDrift: solver.info.volumeDrift ?? Infinity,
  };
}

interface MechanicalEnergySample {
  time_s: number;
  gravitationalPotentialEnergyProxy: number;
  preProjectionKineticEnergyProxy: number;
  postProjectionKineticEnergyProxy: number;
  preProjectionMechanicalEnergyProxy: number;
  postProjectionMechanicalEnergyProxy: number;
  projectionEnergyDelta: number;
  sampledIntervalEnergyDelta: number;
  preProjectionMaximumDivergence_s: number;
  postProjectionMaximumDivergence_s: number;
  maximumDivergenceRatio: number;
  preProjectionRmsDivergence_s: number;
  postProjectionRmsDivergence_s: number;
  rmsDivergenceRatio: number;
  pressureResidual: number;
  pressureRelativeResidual: number;
  exactVolumeDrift: number;
}

function energyTraceSummary(samples: MechanicalEnergySample[]) {
  if (samples.length === 0) return undefined;
  const initial = samples[0].postProjectionMechanicalEnergyProxy;
  const endTime = samples.at(-1)?.time_s ?? 0;
  const middle = samples.filter((sample) => sample.time_s >= 0.2 * endTime && sample.time_s <= 0.4 * endTime);
  const late = samples.filter((sample) => sample.time_s >= 0.8 * endTime);
  const maximumKinetic = (values: MechanicalEnergySample[]) => Math.max(0, ...values.map((sample) => sample.postProjectionKineticEnergyProxy));
  const regression = samples.filter((sample) => sample.time_s >= 0.5 * endTime);
  const meanTime = regression.reduce((sum, sample) => sum + sample.time_s, 0) / Math.max(1, regression.length);
  const meanEnergy = regression.reduce((sum, sample) => sum + sample.postProjectionMechanicalEnergyProxy, 0) / Math.max(1, regression.length);
  const denominator = regression.reduce((sum, sample) => sum + (sample.time_s - meanTime) ** 2, 0);
  const slope = denominator > 0
    ? regression.reduce((sum, sample) => sum + (sample.time_s - meanTime) * (sample.postProjectionMechanicalEnergyProxy - meanEnergy), 0) / denominator
    : 0;
  const middleKineticEnvelope = maximumKinetic(middle);
  const lateKineticEnvelope = maximumKinetic(late);
  const netProjectionEnergyDelta = samples.reduce((sum, sample) => sum + sample.projectionEnergyDelta, 0);
  const cumulativePositiveProjectionEnergyGain = samples.reduce((sum, sample) => sum + Math.max(0, sample.projectionEnergyDelta), 0);
  const driftOscillation = summarizeDriftOscillation(samples.map((sample) => sample.exactVolumeDrift));
  return {
    initialMechanicalEnergyProxy: initial,
    maximumMechanicalEnergyRatio: Math.max(...samples.map((sample) => sample.postProjectionMechanicalEnergyProxy / Math.max(initial, 1e-30))),
    maximumSampledExactVolumeDrift: Math.max(...samples.map((sample) => Math.abs(sample.exactVolumeDrift))),
    finalSampledExactVolumeDrift: Math.abs(samples.at(-1)?.exactVolumeDrift ?? Infinity),
    maximumProjectionEnergyGain: Math.max(0, ...samples.map((sample) => sample.projectionEnergyDelta)),
    netProjectionEnergyDelta,
    normalizedNetProjectionEnergyDelta: netProjectionEnergyDelta / Math.max(initial, 1e-30),
    cumulativePositiveProjectionEnergyGain,
    normalizedCumulativePositiveProjectionEnergyGain: cumulativePositiveProjectionEnergyGain / Math.max(initial, 1e-30),
    maximumProjectionRmsDivergenceRatio: Math.max(...samples.map((sample) => sample.rmsDivergenceRatio)),
    projectionAmplifiedRmsDivergenceSamples: samples.filter((sample) => sample.rmsDivergenceRatio > 1.05).length,
    middleKineticEnvelope,
    lateKineticEnvelope,
    lateToMiddleKineticEnvelopeRatio: lateKineticEnvelope / Math.max(middleKineticEnvelope, 1e-30),
    lateMechanicalEnergySlopePerSecond: slope,
    normalizedLateMechanicalEnergySlopePerSecond: slope / Math.max(initial, 1e-30),
    ...driftOscillation
  };
}

interface StabilityEnvelope {
  peakLiquidSpeed_m_s: number;
  peakComponentCfl: number;
  peakKineticEnergyProxy: number;
  maximumProjectionEnergyRatio: number;
  maximumPressureRelativeResidual: number;
  maximumProjectedVariationalResidual: number;
  maximumExactVolumeDrift: number;
  maximumLevelSetMismatchFraction: number;
  maximumComponentCount: number;
  minimumDominantComponentFraction: number;
  nonFiniteVelocityCount: number;
  invalidVolumeSampleCount: number;
  sampledSteps: number;
}

function referenceVolumeCells(info: GPUEulerianInfo) {
  return info.surfaceField === "levelset"
    ? info.referenceLiquidVolume_cells ?? info.initialVolumeCellSum ?? 0
    : info.initialVolumeCellSum ?? 0;
}

function reportResult(scenario: SmokeScenarioId, result: GPUSmokeResult) {
  const info = result.info;
  console.log(JSON.stringify({
    scenario, method: result.method, phase: "result", construction_ms: Math.round(result.construction_ms), runtime_ms: Math.round(result.runtime_ms), simulationWall_ms: Math.round(result.simulationWall_ms), steps: result.steps,
    powerGenerationAuditedSteps: result.powerGenerationAuditedSteps,
    powerTransitionWitness: result.powerTransitionWitness,
    cpuAdvanceEncodeBreakdown: result.info.cpuAdvanceEncodeBreakdown,
    simulatedTime_s: info.simulatedTime_s, submittedTime_s: info.submittedTime_s, completedTime_s: info.completedTime_s,
    grid: [info.nx, info.storedNy, info.nz], cubicGrid: result.grid,
    allocatedBytes: info.allocatedBytes,
    encodedSteps: info.encodedSteps, gridKind: info.gridKind, compressionRatio: info.compressionRatio,
    activeCompressionRatio: info.activeCompressionRatio, activeSampleCount: info.activeSampleCount,
    quadtreeMaximumFluidScale: info.quadtreeMaximumFluidScale,
    quadtreeLevelSetMismatchFraction: info.quadtreeLevelSetMismatchFraction,
    quadtreeCulledDebrisCells: info.quadtreeCulledDebrisCells,
    quadtreeVelocityClampCount: info.quadtreeVelocityClampCount,
    quadtreeVofReconciliationActive: info.quadtreeVofReconciliationActive,
    quadtreeTopologyStaleSteps: info.quadtreeTopologyStaleSteps,
    quadtreeRebuildCadenceSteps: info.quadtreeRebuildCadenceSteps,
    quadtreeRebuildCompletedCount: info.quadtreeRebuildCompletedCount,
    quadtreeRebuildBlockedFrames: info.quadtreeRebuildBlockedFrames,
    quadtreePressureIterationsUsed: info.quadtreePressureIterationsUsed,
    quadtreeMLSProjectionRowCount: info.quadtreeMLSProjectionRowCount,
    quadtreePressureIterationBudget: info.quadtreePressureIterationBudget,
    quadtreePressureIterationHardBudget: info.quadtreePressureIterationHardBudget,
    pressureRowCapacity: info.pressureRowCapacity,
    pressureEntryCapacity: info.pressureEntryCapacity,
    pressureRequiredRows: info.pressureRequiredRows,
    pressureRequiredEntries: info.pressureRequiredEntries,
    pressureCapacityOverflow: info.pressureCapacityOverflow,
    powerDiagramProjection: info.powerDiagramProjection,
    powerDiagramReady: info.powerDiagramReady,
    powerDiagramAuthoritative: info.powerDiagramAuthoritative,
    powerDiagramFallbackReason: info.powerDiagramFallbackReason,
    powerDiagramAllocatedBytes: info.powerDiagramAllocatedBytes,
    globalFineLevelSetAllocatedBytes: info.globalFineLevelSetAllocatedBytes,
    globalFineLevelSetResidentBrickCapacity: info.globalFineLevelSetResidentBrickCapacity,
    globalFineLevelSetLogicalBrickCount: info.globalFineLevelSetLogicalBrickCount,
    globalFineLevelSetEnabled: info.globalFineLevelSetEnabled,
    globalFineLevelSetFactor: info.globalFineLevelSetFactor,
    frontierListCapacity: info.frontierListCapacity,
    frontierRequiredLeaves: info.frontierRequiredLeaves,
    frontierCapacityOverflow: info.frontierCapacityOverflow,
    quadtreePressureConverged: info.quadtreePressureConverged,
    quadtreeFactorLevelCount: info.quadtreeFactorLevelCount,
    quadtreeMultigridLevelCount: info.quadtreeMultigridLevelCount,
    quadtreeMultigridCoarsestDofs: info.quadtreeMultigridCoarsestDofs,
    quadtreeCPUTopologyPack_ms: info.quadtreeCPUTopologyPack_ms,
    quadtreeGPUSparsePack_ms: info.quadtreeGPUSparsePack_ms,
    quadtreeCPUQuadtreeDecode_ms: info.quadtreeCPUQuadtreeDecode_ms,
    quadtreeCPUTallGrid_ms: info.quadtreeCPUTallGrid_ms,
    quadtreeCPUVariationalAssembly_ms: info.quadtreeCPUVariationalAssembly_ms,
    quadtreeCPUSystemPack_ms: info.quadtreeCPUSystemPack_ms,
    quadtreeCPUICFactorization_ms: info.quadtreeCPUICFactorization_ms,
    quadtreeCPUResourceUpload_ms: info.quadtreeCPUResourceUpload_ms,
    quadtreeTopologyReadbackBytes: info.quadtreeTopologyReadbackBytes,
    quadtreePressurePhaseTimings: info.quadtreePressurePhaseTimings,
    initialVolumeCellSum: info.initialVolumeCellSum, volumeCellSum: info.volumeCellSum,
    representedVolumeCellSum: info.representedVolumeCellSum, volumeDrift: info.volumeDrift,
    representedVolumeDrift: info.representedVolumeDrift, rawVolumeDrift: info.rawVolumeDrift,
    volumeCorrectionNormalSpeed_cells_s: info.volumeCorrectionNormalSpeed_cells_s, volumeCorrectionDivergenceRate_s: info.volumeCorrectionDivergenceRate_s, phiInterfaceCellCount: info.phiInterfaceCellCount, front_m: info.front_m,
    maxSpeed_m_s: info.maxSpeed_m_s, maxComponentCfl: info.maxComponentCfl,
    adaptiveFaceTransportedCount: info.adaptiveFaceTransportedCount,
    adaptiveSurfacePageCapacity: info.adaptiveSurfacePageCapacity,
    adaptiveSurfaceActivePages: info.adaptiveSurfaceActivePages,
    adaptiveSurfaceCandidatePages: info.adaptiveSurfaceCandidatePages,
    adaptiveSurfaceOverflow: info.adaptiveSurfaceOverflow,
    adaptiveSurfaceOverflowCode: info.adaptiveSurfaceOverflowCode,
    pagedPhiDifferentialSamples: info.pagedPhiDifferentialSamples,
    pagedPhiDifferentialComparedSamples: info.pagedPhiDifferentialComparedSamples,
    pagedPhiDifferentialMaxAbs: info.pagedPhiDifferentialMaxAbs,
    pagedPhiDifferentialMeanAbs: info.pagedPhiDifferentialMeanAbs,
    pagedPhiDifferentialSignMismatches: info.pagedPhiDifferentialSignMismatches,
    pagedPhiDifferentialHashMisses: info.pagedPhiDifferentialHashMisses,
    pagedPhiDifferentialAffineFallbacks: info.pagedPhiDifferentialAffineFallbacks,
    pagedPhiDifferentialMaxCell: info.pagedPhiDifferentialMaxCell,
    pagedPhiDifferentialMaxDensePhi: info.pagedPhiDifferentialMaxDensePhi,
    pagedPhiDifferentialMaxPagedPhi: info.pagedPhiDifferentialMaxPagedPhi,
    maxDivergenceBefore_s: info.maxDivergenceBefore_s,
    maxDivergenceAfter_s: info.maxDivergenceAfter_s, pressureRelativeResidual: info.pressureRelativeResidual,
    pressureResidual: info.pressureResidual,
    nonFiniteCount: info.nonFiniteCount, stabilityFlags: info.stabilityFlags, gpuTimings: info.gpuTimings,
    matchedFieldStats: result.matchedSummary, volumeFieldStats: result.finalSummary,
    compactFieldEvidence: result.compactFieldEvidence,
    matchedTallCellActivity: result.matchedTallCellActivity, finalTallCellActivity: result.finalTallCellActivity,
    finalTallVolumeGaps: result.finalTallVolumeGaps,
    velocitySummary: result.velocitySummary, initialFluidBrickStats: result.initialFluidBrickStats,
    sparseVoxelStats: result.sparseVoxelStats, hybridPresentationStats: result.hybridPresentationStats,
    initialGlobalFineGeneration: result.initialGlobalFineGeneration,
    initialGlobalFineRaster: result.initialGlobalFineRaster,
    finalGlobalFineGeneration: result.finalGlobalFineGeneration,
    finalGlobalFineRaster: result.finalGlobalFineRaster,
    globalFineGenerationCheckpoints: result.checkpoints.map(({ time_s, globalFineGeneration, raster }) => ({
      time_s, globalFineGeneration,
      raster: raster ? { frontInterfacePixels: raster.frontInterfacePixels, backInterfacePixels: raster.backInterfacePixels,
        pairedInterfacePixels: raster.pairedInterfacePixels,
        frontOnlyInterfacePixels: raster.frontOnlyInterfacePixels,
        backOnlyInterfacePixels: raster.backOnlyInterfacePixels,
        narrowVerticalSlits: raster.narrowVerticalSlits,
        reverseView: raster.reverseView,
        vertexCount: raster.vertexCount, vertexAllocator: raster.vertexAllocator,
        vertexCapacity: raster.vertexCapacity, activeCubeCount: raster.activeCubeCount,
        activeCubeCapacity: raster.activeCubeCapacity,
        frontInterfaceHash: raster.frontInterfaceHash, backInterfaceHash: raster.backInterfaceHash,
        frontInterfaceBounds_m: raster.frontInterfaceBounds_m,
        surfaceGeometrySource: raster.surfaceGeometrySource,
        globalFineAuthorityLatch: raster.globalFineAuthorityLatch,
        globalFineCrossingPublished: raster.globalFineCrossingPublished,
        presentationFallbackActive: raster.presentationFallbackActive,
        globalFineAuthorityTransition: raster.globalFineAuthorityTransition } : undefined,
    })),
    octreeFaceMirrorDiagnostics: result.octreeFaceMirrorDiagnostics,
    octreePowerFaceTransferDiagnostics: result.octreePowerFaceTransferDiagnostics,
    octreePowerFaceDiagnostics: result.octreePowerFaceDiagnostics,
    octreePowerTopologyDiagnostics: result.octreePowerTopologyDiagnostics,
    octreeMGPCGDiagnostics: result.octreeMGPCGDiagnostics,
    powerProjectionDiagnostics: result.powerProjectionDiagnostics,
    stabilityEnvelope: result.stabilityEnvelope,
    energyTraceSummary: energyTraceSummary(result.energyTrace),
    validationErrors: result.validationErrors
  }));
}

async function runGPU(
  scenarioId: SmokeScenarioId,
  method: SimulationMethod,
  target_s: number,
  oracleSteps: number
): Promise<GPUSmokeResult> {
  const scenario = createSmokeScenario(scenarioId), scene = applySceneOverrides(scenario.scene);
  const authoredProfile = scenario.methodProfile?.methodId === method.id
    ? scenario.methodProfile : undefined;
  // Validation presets are authored once for the UI. Native Dawn starts from
  // that exact quality/profile and only then applies explicitly requested
  // diagnostic overrides; do not maintain a second implicit solver preset in
  // package.json.
  const solverQuality = authoredProfile?.quality ?? quality;
  // Validation comparisons author the exact same scene lattice on every backend.
  if (voxelCellSizeOverride !== undefined) scene.voxelDomain.finestCellSize_m = voxelCellSizeOverride;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
  // FLUID_DISABLE_TIMESTAMPS=1 drops the timestamp-query feature: under Dawn
  // node the tall-cell solver's first-step projection pass does not execute
  // when timestamp writes are attached (see docs/TALL_CELL_STABILITY.md), so
  // correctness audits run without GPU stage timings.
  const requiredFeatures: GPUFeatureName[] = [
    ...(adapter.features.has("timestamp-query") && process.env.FLUID_DISABLE_TIMESTAMPS !== "1" ? ["timestamp-query" as GPUFeatureName] : []),
    ...optionalFluidDeviceFeatures(adapter.features),
  ];
  const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  await assertComputeSentinel(device);
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => { lost = info; });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const instrumentedDevice = new Proxy(device, {
    get(target, property) {
      if (property === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => {
        const started = performance.now(), result = target.createComputePipeline(descriptor);
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "pipeline", entryPoint: descriptor.compute.entryPoint, elapsed_ms: Math.round(performance.now() - started) }));
        return result;
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    }
  }) as GPUDevice;
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const constructionStarted = performance.now();
  const values = method.presetFor(solverQuality);
  if (authoredProfile) Object.assign(values, authoredProfile.overrides);
  if (method.id === "tall-cell" && pressureCyclesOverride !== undefined) values.pressureCycles = pressureCyclesOverride;
  if (method.id === "tall-cell" && pressureWarmStartOverride !== undefined) values.pressureWarmStart = pressureWarmStartOverride ? "on" : "off";
  if ((method.id === "quadtree-tall-cell" || method.id === "octree") && pressureCyclesOverride !== undefined) values.pressureIterations = pressureCyclesOverride;
  if (method.id === "quadtree-tall-cell" && pressureWarmStartOverride !== undefined) values.pressureWarmStart = pressureWarmStartOverride ? "on" : "off";
  if (method.id === "quadtree-tall-cell" && quadtreeMegakernelOverride !== undefined) values.megakernelSolve = quadtreeMegakernelOverride;
  if (method.id === "quadtree-tall-cell" && quadtreePressureSolverOverride !== undefined) values.pressureSolver = quadtreePressureSolverOverride;
  if (method.id === "quadtree-tall-cell" && adaptivityOverride !== undefined) values.adaptivityStrength = adaptivityOverride;
  if (method.id === "quadtree-tall-cell" && opticalDepthOverride !== undefined) values.opticalDepthFraction = opticalDepthOverride;
  if (method.id === "quadtree-tall-cell" && opticalLayerModeOverride !== undefined) values.opticalLayerMode = opticalLayerModeOverride;
  if (method.id === "quadtree-tall-cell" && opticalAlphaOverride !== undefined) values.opticalAlpha = opticalAlphaOverride;
  if (method.id === "quadtree-tall-cell" && deepSpeedGradientOverride !== undefined) values.deepSpeedGradientScale = deepSpeedGradientOverride;
  if (method.id === "quadtree-tall-cell" && rebuildTopologyOverride !== undefined) values.rebuildTopology = rebuildTopologyOverride;
  if (method.id === "quadtree-tall-cell" && maximumLeafSizeOverride !== undefined) values.maximumLeafSize = maximumLeafSizeOverride;
  // The ocean scene exists to demonstrate 32-cubed coarse leaves in deep calm
  // water; scenes cannot carry method parameters, so the harness requests the
  // raised cap here. FLUID_MAXIMUM_LEAF_SIZE still wins below for A/B runs.
  if (method.id === "octree" && scenarioId === "ocean-seiche") values.maximumLeafSize = 32;
  if (method.id === "octree" && maximumLeafSizeOverride !== undefined) values.maximumLeafSize = maximumLeafSizeOverride;
  if (method.id === "octree" && octreeInterfaceBandOverride !== undefined) {
    values.interfaceRefinementBandCells = octreeInterfaceBandOverride;
  }
  if (method.id === "octree" && octreeGlobalFineFactorOverride !== undefined) values.globalFineLevelSetFactor = octreeGlobalFineFactorOverride;
  if (method.id === "quadtree-tall-cell" && quadtreePreconditionerOverride !== undefined) values.preconditioner = quadtreePreconditionerOverride;
  if (method.id === "quadtree-tall-cell" && quadtreeStaleStepsOverride !== undefined) values.topologyStaleSteps = quadtreeStaleStepsOverride;
  if (method.id === "quadtree-tall-cell" && quadtreeInlineRebuildOverride !== undefined) values.inlineRebuild = quadtreeInlineRebuildOverride;
  if (method.id === "quadtree-tall-cell" && quadtreeDebrisCullingOverride !== undefined) values.debrisCulling = quadtreeDebrisCullingOverride;
  if (method.id === "quadtree-tall-cell" && quadtreeVofReconciliationOverride !== undefined) values.vofReconciliation = quadtreeVofReconciliationOverride ? "on" : "off";
  if (method.id === "quadtree-tall-cell" && pressurePhaseTimings) values.debugPressureTimings = true;
  if (method.id === "quadtree-tall-cell" && polynomialDegreeOverride !== undefined) values.polynomialDegree = polynomialDegreeOverride;
  if (method.id === "tall-cell" && remeshIntervalOverride !== undefined) values.remeshInterval = remeshIntervalOverride;
  if (method.id === "tall-cell" && regularLayersOverride !== undefined) values.regularLayers = regularLayersOverride;
  if (method.id === "tall-cell" && maximumNeighborDeltaOverride !== undefined) values.maximumNeighborDelta = maximumNeighborDeltaOverride;
  if (method.id === "tall-cell" && maximumTallHeightOverride !== undefined) values.maximumTallHeight = maximumTallHeightOverride;
  if ((method.id === "tall-cell" || method.id === "uniform") && velocityTransportOverride !== undefined) values.velocityTransport = velocityTransportOverride;
  if ((method.id === "tall-cell" || method.id === "uniform") && sharpeningOverride !== undefined) values.densitySharpening = sharpeningOverride ? "on" : "off";
  if (method.id === "tall-cell" && volumeControlOverride !== undefined) values.volumeControl = volumeControlOverride ? "on" : "off";
  if (method.id === "tall-cell" && referenceVolumeScaleOverride !== undefined) values.referenceVolumeScale = referenceVolumeScaleOverride;
  if (method.id === "tall-cell" && hierarchyOverride !== undefined) values.hierarchicalExtrapolation = hierarchyOverride ? "on" : "off";
  const probeLayout = singleTallCellProbe && (method.id === "tall-cell" || method.id === "uniform")
    ? method.id === "tall-cell"
      ? createSingleTallCellProbeLayout(scene, solverQuality, device.limits.maxTextureDimension3D, singleTallCellProbe)
      : createSingleTallCellProbeControlLayout(scene, solverQuality, device.limits.maxTextureDimension3D, singleTallCellProbe)
    : undefined;
  const resultMethod = singleTallCellProbe && method.id === "uniform" ? "tall-cell-control" : method.id;
  const solver = probeLayout
    ? new WebGPUEulerianSolver(instrumentedDevice, scene, solverQuality, undefined, {
      layoutOverride: probeLayout,
      pressureCycles: typeof values.pressureCycles === "number" ? values.pressureCycles : 2,
      pressureWarmStart: values.pressureWarmStart !== "off",
      velocityTransport: values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack",
      volumeControl: values.volumeControl !== "off",
      referenceVolumeScale: typeof values.referenceVolumeScale === "number" ? values.referenceVolumeScale : undefined,
      hierarchicalExtrapolation: values.hierarchicalExtrapolation !== "off"
    })
    : method.id === "octree" && method.createSolverAsync
      // The power catalog and fenced t=0 sparse authority are initialization
      // tasks in the production browser path. Dawn must use the same async
      // constructor even when authority came from an authored UI profile
      // instead of a command-line override.
      ? await method.createSolverAsync(instrumentedDevice, scene, solverQuality, values, undefined, (progress) => {
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
          record: "solver-initialization", ...progress }));
      })
      : method.createSolver!(instrumentedDevice, scene, solverQuality, values);
  const construction_ms = performance.now() - constructionStarted;
  const actualGrid: [number, number, number] = [solver.info.nx, solver.info.ny, solver.info.nz];
  if (expectedGridOverride && actualGrid.some((value, axis) => value !== expectedGridOverride[axis])) {
    throw new Error(`${scenarioId}/${resultMethod} constructed ${actualGrid.join("x")} instead of FLUID_EXPECT_GRID=${expectedGridOverride.join("x")}; refusing to step a mismatched comparison`);
  }
  console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod, phase: "constructed",
    construction_ms: Math.round(construction_ms), quality: solverQuality,
    authoredMethodProfile: authoredProfile,
    grid: [solver.info.nx, solver.info.storedNy, solver.info.nz],
    cubicGrid: [solver.info.nx, solver.info.ny, solver.info.nz] }));
  /** Match the renderer's admission boundary. A profiled octree advance
   * submits its phases asynchronously; queue.onSubmittedWorkDone() alone can
   * resolve between those submissions and expose a torn generation. */
  const awaitAdvanceCompletion = async () => {
    const pending = (solver as GPUSolverInstance).pendingAdvanceCompletion;
    if (pending) await pending;
    await device.queue.onSubmittedWorkDone();
  };
  // Raw voxel/brick records are a lazy inspection product. Merely reading the
  // getter allocates their large publication arenas, so production timing and
  // memory runs must not request them unless the explicit sparse audit is on.
  const sparseSource = sparseStatsRequested
    ? (solver as GPUSolverInstance).sparseVoxelRenderSource
    : undefined;
  const seedBrickBounds = initialSeedBrickBounds(scene, [solver.info.nx, solver.info.ny, solver.info.nz]);
  const initialFluidBrickStats = sparseStatsRequested && sparseSource
    ? await readFluidBrickSnapshot(device, sparseSource)
    : undefined;
  const initialGlobalFineGeneration = globalFineGenerationTransitionRequested && method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  const initialGlobalFineRaster = globalFineGenerationTransitionRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies, true) : undefined;
  if (initialGlobalFineRaster) {
    // Emit the pre-step renderer evidence immediately.  A later simulation
    // transaction may deliberately reject and roll back, but that must not
    // hide whether reset-time global-fine rasterization was already visible.
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "initial-global-fine-raster", ...initialGlobalFineRaster }));
  }
  if (powerGenerationAuditRequested && method.id === "octree") {
    const initialAudit = solver as GPUSolverInstance & {
      adaptiveSurfaceCandidateControl?: GPUBuffer; adaptiveSurfaceLeaves?: GPUBuffer;
    };
    if (initialAudit.adaptiveSurfaceCandidateControl && initialAudit.adaptiveSurfaceLeaves) {
      const [candidateBytes, leafBytes] = await Promise.all([
        readBufferBinding(device, { buffer: initialAudit.adaptiveSurfaceCandidateControl }, 32),
        readBufferBinding(device, { buffer: initialAudit.adaptiveSurfaceLeaves }, initialAudit.adaptiveSurfaceLeaves.size),
      ]);
      const candidates = new Uint32Array(candidateBytes.buffer, candidateBytes.byteOffset, 8);
      const leaves = new Uint32Array(leafBytes.buffer, leafBytes.byteOffset, leafBytes.byteLength / 4);
      const leafFloats = new Float32Array(leafBytes.buffer, leafBytes.byteOffset, leafBytes.byteLength / 4);
      let live = 0, core = 0, halo = 0, minimumPhi = Infinity, maximumPhi = -Infinity;
      for (let row = 0; row + 15 < leaves.length; row += 16) {
        const flags = leaves[row + 4];
        if ((flags & 32) !== 0) live += 1;
        if ((flags & 2) !== 0) core += 1;
        if ((flags & 4) !== 0) halo += 1;
        if ((flags & 32) !== 0) {
          minimumPhi = Math.min(minimumPhi, leafFloats[row + 8]);
          maximumPhi = Math.max(maximumPhi, leafFloats[row + 8]);
        }
      }
      console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
        phase: "initial-surface-candidate-audit", control: Array.from(candidates),
        leaves: { live, core, halo, minimumPhi, maximumPhi } }));
    }
  }
  const runStarted = performance.now();
  let steps = 0, samplingWall_ms = 0, matched: Awaited<ReturnType<typeof readCubicVolumeField>> | undefined;
  // The perturbed cadence remains exclusive to the quadtree dam-break
  // regression; FLUID_STABILITY_ENVELOPE=1 collects the same envelope for any
  // scenario/method at the scene's fixed cadence.
  const perturbCadence = scenarioId === "dam-break-ui" && method.id === "quadtree-tall-cell";
  const collectStabilityEnvelope = perturbCadence || stabilityEnvelopeRequested;
  const stabilityEnvelope: StabilityEnvelope | undefined = collectStabilityEnvelope ? {
    peakLiquidSpeed_m_s: 0, peakComponentCfl: 0, peakKineticEnergyProxy: 0,
    maximumProjectionEnergyRatio: 0, maximumPressureRelativeResidual: 0,
    maximumProjectedVariationalResidual: 0, maximumExactVolumeDrift: 0, maximumLevelSetMismatchFraction: 0,
    maximumComponentCount: 0, minimumDominantComponentFraction: 1,
    nonFiniteVelocityCount: 0, invalidVolumeSampleCount: 0, sampledSteps: 0
  } : undefined;
  // Compact-octree spatial QA reconstructs occupancy on the finest cubic
  // lattice, whereas the conservative controller integrates adaptive cell
  // volumes. Compare this estimator with its own accepted reset-time field;
  // mixing the two baselines manufactures drift even when both are stable.
  const initialExact = !performanceProfileRequested && method.id === "octree" && (collectStabilityEnvelope || energyEverySteps > 0)
    ? await readCubicVolumeField(device, solver) : undefined;
  const spatialExactReference = initialExact?.summary.cellSum;
  // The UI uses a fixed cadence.  The long regression deliberately perturbs
  // that cadence while respecting maxDt so topology transfer and projection
  // are exercised with genuinely different timestep sizes.
  const regressionDtPattern = [0.004, 0.0035, 0.0025, 0.004];
  const checkpoints: GPUSmokeResult["checkpoints"] = [];
  const energyTrace: MechanicalEnergySample[] = [];
  let previousSampledMechanicalEnergy = 0;
  if (energyEverySteps > 0) {
    await device.queue.onSubmittedWorkDone();
    const initial = initialExact ?? await readCubicVolumeField(device, solver);
    const spacing = {
      x: scene.container.width_m / solver.info.nx,
      y: scene.container.height_m / solver.info.ny,
      z: scene.container.depth_m / solver.info.nz
    };
    const potential = gravitationalPotentialEnergyProxy(initial.field, solver.info.nx, solver.info.ny, solver.info.nz, spacing, scene.fluid.gravity_m_s2);
    const exactReference = spatialExactReference ?? referenceVolumeCells(solver.info);
    const sample: MechanicalEnergySample = {
      time_s: 0,
      gravitationalPotentialEnergyProxy: potential,
      preProjectionKineticEnergyProxy: 0,
      postProjectionKineticEnergyProxy: 0,
      preProjectionMechanicalEnergyProxy: potential,
      postProjectionMechanicalEnergyProxy: potential,
      projectionEnergyDelta: 0,
      sampledIntervalEnergyDelta: 0,
      preProjectionMaximumDivergence_s: 0,
      postProjectionMaximumDivergence_s: 0,
      maximumDivergenceRatio: 0,
      preProjectionRmsDivergence_s: 0,
      postProjectionRmsDivergence_s: 0,
      rmsDivergenceRatio: 0,
      pressureResidual: 0,
      pressureRelativeResidual: 0,
      exactVolumeDrift: (initial.summary.cellSum - exactReference) / Math.max(1, Math.abs(exactReference))
    };
    energyTrace.push(sample);
    previousSampledMechanicalEnergy = potential;
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod, phase: "energy", ...sample }));
  }
  let nextCheckpoint_s = checkpointEvery_s;
  let previousAuditedPowerGeneration = 0;
  let powerGenerationAuditedSteps = 0;
  let powerTransitionWitness: PowerTransitionWitness | undefined;
  let topologyTransitionDeepCell: number | undefined;
  while ((solver.info.submittedTime_s ?? 0) + 1e-9 < target_s) {
    const stepDt = perturbCadence
      ? Math.min(scene.numerics.maxDt_s, regressionDtPattern[steps % regressionDtPattern.length])
      : scene.numerics.maxDt_s;
    const requestedTime = Math.min(target_s, (solver.info.submittedTime_s ?? 0) + stepDt);
    const accepted = solver.advanceTo(requestedTime, bodies);
    if (!accepted) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    steps += 1;
    const auditThisPowerStep = steps % powerAuditEverySteps === 0 || requestedTime + 1e-9 >= target_s;
    if (powerGenerationAuditRequested && auditThisPowerStep && method.id === "octree") {
      // Every thirtieth octree advance may be submitted as serial profiler
      // phases. A queue fence taken before those asynchronous submissions are
      // enqueued observes the preceding valid generation, not the accepted
      // step. Match the renderer's lifecycle contract before auditing it.
      await awaitAdvanceCompletion();
      const audited = solver as GPUSolverInstance & {
        powerFaceControl?: GPUBuffer; powerFaceSource?: { faces: GPUBuffer; faceNormals: GPUBuffer; faceCentroids: GPUBuffer };
        powerFaceTransferControl?: GPUBuffer; powerFaceSiteIndex?: GPUBuffer;
        powerFaceSeedControl?: GPUBuffer; powerFaceAdvectionControl?: GPUBuffer; powerSolidFaceControl?: GPUBuffer;
        powerOperatorControl?: GPUBuffer; mgpcgControl?: GPUBuffer;
        powerDescriptorControl?: GPUBuffer; powerTopologyControl?: GPUBuffer;
        globalFineCoarseLevelSetControl?: GPUBuffer;
        globalFinePowerVelocityControl?: GPUBuffer;
        globalFineSummaryDirectory?: GPUBuffer;
        powerLeafHeaders?: GPUBuffer; powerDescriptorRows?: GPUBuffer; powerTopologyMetrics?: GPUBuffer;
        powerLeafFrontier?: GPUBuffer; topologyTileWorklist?: GPUBuffer;
        powerOwnerArena?: GPUBuffer;
        powerCatalogEntryHeaders?: GPUBuffer; powerCatalogFaces?: GPUBuffer;
        adaptiveSurfaceCandidateControl?: GPUBuffer;
        powerBoundaryPhiQueries?: GPUBuffer;
        powerBoundaryFineSource?: { generation: number; generationSlot: 0 | 1 };
        powerBoundaryFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
      };
      const axisSource = (solver as GPUSolverInstance).adaptiveFaceMirrorSource;
      if (!audited.powerFaceControl || !audited.powerFaceTransferControl || !audited.powerFaceSeedControl || !audited.powerOperatorControl
        || !audited.powerDescriptorControl || !audited.powerTopologyControl) {
        const missing = (["powerFaceControl", "powerFaceTransferControl", "powerFaceSeedControl", "powerOperatorControl",
          "powerDescriptorControl", "powerTopologyControl"] as const).filter((name) => !audited[name]);
        throw new Error(`power generation audit step ${steps} is missing authoritative control buffers: ${missing.join(", ")}; `
          + `ready=${solver.info.powerDiagramReady} authoritative=${solver.info.powerDiagramAuthoritative} `
          + `fallback=${solver.info.powerDiagramFallbackReason ?? "none"}`);
      }
      const [faceBytes, transferBytes, seedBytes, advectionBytes, solidBytes, operatorBytes,
        descriptorBytes, topologyBytes, axisBytes, coarsePhiBytes, powerVelocityBytes] = await Promise.all([
        readBufferBinding(device, { buffer: audited.powerFaceControl }, 64),
        readBufferBinding(device, { buffer: audited.powerFaceTransferControl }, 32),
        readBufferBinding(device, { buffer: audited.powerFaceSeedControl }, 64),
        audited.powerFaceAdvectionControl
          ? readBufferBinding(device, { buffer: audited.powerFaceAdvectionControl }, 64) : Promise.resolve(undefined),
        audited.powerSolidFaceControl
          ? readBufferBinding(device, { buffer: audited.powerSolidFaceControl }, 64) : Promise.resolve(undefined),
        readBufferBinding(device, { buffer: audited.powerOperatorControl }, 64),
        readBufferBinding(device, { buffer: audited.powerDescriptorControl }, 32),
        readBufferBinding(device, { buffer: audited.powerTopologyControl }, 32),
        axisSource ? readBufferBinding(device, { buffer: axisSource.control }, 24) : Promise.resolve(undefined),
        audited.globalFineCoarseLevelSetControl
          ? readBufferBinding(device, { buffer: audited.globalFineCoarseLevelSetControl }, 64)
          : Promise.resolve(undefined),
        audited.globalFinePowerVelocityControl
          ? readBufferBinding(device, { buffer: audited.globalFinePowerVelocityControl }, 32)
          : Promise.resolve(undefined),
      ]);
      const face = new Uint32Array(faceBytes.buffer, faceBytes.byteOffset, 16);
      const transfer = new Uint32Array(transferBytes.buffer, transferBytes.byteOffset, 8);
      const seed = new Uint32Array(seedBytes.buffer, seedBytes.byteOffset, 16);
      const advection = advectionBytes
        ? new Uint32Array(advectionBytes.buffer, advectionBytes.byteOffset, 16) : undefined;
      const solid = solidBytes ? new Uint32Array(solidBytes.buffer, solidBytes.byteOffset, 16) : undefined;
      const operator = new Uint32Array(operatorBytes.buffer, operatorBytes.byteOffset, 16);
      const mgpcgBytes = audited.mgpcgControl
        ? await readBufferBinding(device, { buffer: audited.mgpcgControl }, 64) : undefined;
      const mgpcg = mgpcgBytes ? new Uint32Array(mgpcgBytes.buffer, mgpcgBytes.byteOffset, 16) : undefined;
      const mgpcgDiagnostics = mgpcg ? decodeOctreeMGPCGDiagnostics(mgpcg) : undefined;
      const descriptor = new Uint32Array(descriptorBytes.buffer, descriptorBytes.byteOffset, 8);
      const topology = new Uint32Array(topologyBytes.buffer, topologyBytes.byteOffset, 8);
      const axis = axisBytes ? new Uint32Array(axisBytes.buffer, axisBytes.byteOffset, 6) : undefined;
      const coarsePhi = coarsePhiBytes
        ? new Uint32Array(coarsePhiBytes.buffer, coarsePhiBytes.byteOffset, 16) : undefined;
      const powerVelocity = powerVelocityBytes
        ? new Uint32Array(powerVelocityBytes.buffer, powerVelocityBytes.byteOffset, 8) : undefined;
      const floatBits = (word: number) => new Float32Array(new Uint32Array([word]).buffer)[0];
      const faceGeometryFailure = (face[3] & 8) !== 0;
      const faceNeighborFailure = (face[3] & (16 | 64)) !== 0;
      const audit = {
        step: steps,
        requestedTime, stepDt, submittedTime: solver.info.submittedTime_s,
        axis: axis ? { faceCount: axis[0], overflow: axis[1] !== 0, faceCapacity: axis[2], rowCapacity: axis[3],
          maximumIncidence: axis[4], incidenceOverflowAppends: axis[5] } : undefined,
        faces: { rowCount: face[0], faceCount: face[1], incidenceCount: face[2], flags: face[3],
          firstInvalid: face[4], invalidCount: face[5], generation: face[7], valid: face[8] === 0x8000_0000,
          lookupMissCount: face[9], firstInvalidSlot: face[12],
          firstInvalidNeighbor: faceNeighborFailure ? face[13] : 0xffff_ffff, firstInvalidDetail: face[14],
          firstInvalidRow: face[4], firstInvalidPad3: face[15],
          firstInvalidLiquidPhi: faceGeometryFailure ? floatBits(face[13]) : undefined,
          firstInvalidAirPhi: faceGeometryFailure ? floatBits(face[15]) : undefined },
        transfer: { previousFaceCount: transfer[0], valid: transfer[1] === 0x8000_0000,
          flags: transfer[2], generation: transfer[3], exactFaces: transfer[4], fallbackFaces: transfer[5],
          maximumFaceSpeed: floatBits(transfer[6]), sourceFlags: transfer[7] },
        oldMeshAdvection: advection ? { flags: advection[0], firstError: advection[1], oldRowCount: advection[2],
          advectedFaces: advection[3], generation: advection[4], oldGeneration: advection[5],
          valid: advection[6] === 0x8000_0000, coldFallback: advection[7] !== 0, targetFaceCount: advection[8],
          attemptFlags: advection[9], attemptFirstError: advection[10], attemptAdvectedFaces: advection[11],
          attemptValid: advection[12] === 0x8000_0000,
          firstInterpolationStage: advection[13] === 0xffff_ffff ? undefined : advection[13] & 3,
          firstInterpolationFace: advection[13] === 0xffff_ffff ? undefined : Math.floor(advection[13] / 4),
          firstInterpolationReason: advection[14] === 0xffff_ffff ? undefined : advection[14] & 15,
          firstInterpolationReasonFace: advection[14] === 0xffff_ffff ? undefined : Math.floor(advection[14] / 16),
          retainedBandFailureReason: advection[15] === 0xffff_ffff ? undefined : advection[15] & 0xff,
          retainedBandFailureAnchor: advection[15] === 0xffff_ffff ? undefined : advection[15] >>> 8 }
          : undefined,
        solidFaces: solid ? { flags: solid[0], processedFaces: solid[1], cutFaces: solid[2],
          occupiedSamples: solid[3], faceCount: solid[4], rigidBodyCount: solid[5], terrainEnabled: solid[6] !== 0,
          valid: solid[7] === 0x8000_0000, vertexGeneration: solid[14], rollbackGeneration: solid[15] }
          : undefined,
        velocityStages: { axisRowInputMaximum: floatBits(seed[9]), axisToPowerSeedMaximum: floatBits(seed[8]),
          projectionInputMaximum: floatBits(operator[8]), projectionOutputMaximum: floatBits(operator[9]),
          projectedRowMaximum: floatBits(seed[10]), powerToAxisOutputMaximum: floatBits(seed[11]),
          seedFlags: seed[12], seedFirstError: seed[13], seedProcessedCount: seed[14], seedValid: seed[15] === 0x8000_0000,
          reverseFlags: seed[0], reverseFirstError: seed[1], reverseProcessedCount: seed[4], reverseValid: seed[6] === 0x8000_0000,
          operatorFlags: operator[0], operatorFirstError: operator[1], operatorProjectedCount: operator[6],
          mgpcgFlags: mgpcgDiagnostics?.flags, mgpcgConverged: mgpcgDiagnostics?.converged,
          mgpcgIterations: mgpcgDiagnostics?.iterations, mgpcgRows: mgpcgDiagnostics?.rows,
          mgpcgResidualSquared: mgpcgDiagnostics?.residualSquared,
          mgpcgRhsSquared: mgpcgDiagnostics?.rhsSquared,
          mgpcgRelativeResidualSquared: mgpcgDiagnostics?.relativeResidualSquared,
          mgpcgFirstFailureStage: mgpcg?.[10], mgpcgFirstFailureRow: mgpcg?.[11],
          mgpcgFirstFailureValue: mgpcg ? floatBits(mgpcg[12]) : undefined,
          operatorAssemblyFlags: operator[10], operatorAssemblyFirstError: operator[11], operatorEntryCount: operator[5],
          operatorRowCount: operator[2], operatorFaceCount: operator[3], operatorIncidenceCount: operator[4] },
        descriptor: { rowCount: descriptor[0], validCount: descriptor[1], errorCount: descriptor[2],
          firstInvalid: descriptor[3], flags: descriptor[4], generation: descriptor[7] },
        topology: { invalidCount: topology[0], firstInvalid: topology[1], flags: topology[2],
          resolvedCount: topology[3], version: topology[4] },
        coarsePhi: coarsePhi ? { flags: coarsePhi[0], firstError: coarsePhi[1], rowCount: coarsePhi[2],
          advected: coarsePhi[3], uniformUpdates: coarsePhi[4], transitionUpdates: coarsePhi[5],
          nearestFallbacks: coarsePhi[6], passes: coarsePhi[7], correctedRows: coarsePhi[8],
          interfaceRows: coarsePhi[9], generation: coarsePhi[11], valid: coarsePhi[12] === 0x8000_0000 }
          : undefined,
        powerVelocity: powerVelocity ? { flags: powerVelocity[0], firstError: powerVelocity[1],
          rowCount: powerVelocity[2], faceCount: powerVelocity[3], incidenceCount: powerVelocity[4],
          reconstructedCount: powerVelocity[5], fallbackCount: powerVelocity[6], generation: powerVelocity[7] }
          : undefined,
      };
      if (topologyTransitionAuditLog) {
        const transitionHeaders = audited.powerLeafHeaders;
        const transitionFrontier = audited.powerLeafFrontier;
        const transitionTiles = audited.topologyTileWorklist;
        if (!transitionHeaders || !transitionFrontier || !transitionTiles) {
          throw new Error("topology transition audit requires compact headers, frontier, and tile worklist buffers");
        }
        const headerBytes = await readBufferBinding(device, { buffer: transitionHeaders }, audit.faces.rowCount * 48);
        const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength / 4);
        if (topologyTransitionDeepCell === undefined) {
          let bestY = Infinity, bestCell: number | undefined;
          for (let row = 0; row < audit.faces.rowCount; row += 1) {
            const cell = headers[row * 12], y = Math.floor(cell / solver.info.nx) % solver.info.ny;
            if (y < bestY) { bestY = y; bestCell = cell; }
          }
          topologyTransitionDeepCell = bestCell;
        }
        const frontierBytes = await readBufferBinding(device, { buffer: transitionFrontier }, transitionFrontier.size);
        const frontier = new Uint32Array(frontierBytes.buffer, frontierBytes.byteOffset, frontierBytes.byteLength / 4);
        const listCapacity = solver.info.frontierListCapacity ?? 0;
        const current = frontier[2], frontierCount = Math.min(frontier[current], listCapacity);
        let frontierContainsDeepCell = false;
        for (let slot = 0; slot < frontierCount; slot += 1) {
          if (frontier[4 + current * listCapacity + slot] === topologyTransitionDeepCell) {
            frontierContainsDeepCell = true; break;
          }
        }
        const tileBytes = await readBufferBinding(device, { buffer: transitionTiles }, transitionTiles.size);
        const tiles = new Uint32Array(tileBytes.buffer, tileBytes.byteOffset, tileBytes.byteLength / 4);
        const tileCapacity = (tiles.length - 16) / 2;
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "topology-transition-audit",
          step: steps, generation: audit.faces.generation, deepCell: topologyTransitionDeepCell,
          frontierContainsDeepCell, frontierCount, activeTopologyTiles: tiles[0], retiredTopologyTiles: tiles[4],
          topologyGeneration: tiles[15], tileCapacity }));
      }
      if (powerGenerationAuditLog) {
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "power-generation-audit", ...audit }));
      }
      if (powerStageAuditLog) {
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "power-stage-audit",
          step: steps, generation: audit.faces.generation, ...audit.velocityStages,
          topologyTransferMaximum: audit.transfer.maximumFaceSpeed }));
      }
      if (process.env.FLUID_OWNER_AUDIT === "1") {
        const issues = await auditOwnerLatticeConsistency(device, solver, [solver.info.nx, solver.info.ny, solver.info.nz]);
        if (issues.length > 0) {
          console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "owner-lattice-audit",
            step: steps, generation: audit.faces.generation, issues: issues.slice(0, 12), issueCount: issues.length }));
        }
      }
      if (powerGenerationAuditLog) {
        const fine = await readGlobalFineGenerationDiagnostics(device, solver, powerBoundaryQueryAuditKeys);
        const fineSource = (solver as GPUSolverInstance).globalFineLevelSetSource;
        const queryBytes = powerBoundaryQueryAuditKeys.length > 0 && audited.powerBoundaryPhiQueries
          && audit.faces.faceCount > 0
          ? await readBufferBinding(device, { buffer: audited.powerBoundaryPhiQueries }, audit.faces.faceCount * 32)
          : undefined;
        const queryMatches: Array<{
          face: number; liquidCenter: [number, number, number]; airCenter: [number, number, number];
          liquidKeys: number[]; airKeys: number[];
        }> = [];
        const pressureRows: Array<{ cell: number; row: number; size: number }> = [];
        const facesTouchingTargets: Array<{
          face: number; negativeRow: number; positiveRow: number; negativeCell: number;
          positiveCell?: number; slot: number; flags: number;
        }> = [];
        const ownerTopology: Array<{ cell: number; origin: [number, number, number]; size: number;
          invalid: boolean; packed?: number }> = [];
        const legacyPhiSamples: Array<{ cell: number; phi: number | null }> = [];
        if (queryBytes && fineSource) {
          const queries = new Float32Array(queryBytes.buffer, queryBytes.byteOffset, queryBytes.byteLength / 4);
          const targets = new Set(powerBoundaryQueryAuditKeys);
          for (let faceIndex = 0; faceIndex < audit.faces.faceCount; faceIndex += 1) {
            const base = faceIndex * 8;
            if (queries[base + 3] === 0 || queries[base + 7] === 0) continue;
            const liquidCenter: [number, number, number] = [queries[base], queries[base + 1], queries[base + 2]];
            const airCenter: [number, number, number] = [queries[base + 4], queries[base + 5], queries[base + 6]];
            const liquidKeys = fineTrilinearBrickKeysAtPosition(fineSource, liquidCenter);
            const airKeys = fineTrilinearBrickKeysAtPosition(fineSource, airCenter);
            if (![...liquidKeys, ...airKeys].some((key) => targets.has(key))) continue;
            queryMatches.push({ face: faceIndex, liquidCenter, airCenter, liquidKeys, airKeys });
          }
        }
        const technique = (solver as GPUSolverInstance).octreeTechniqueDebugSource;
        if (powerBoundaryQueryAuditKeys.length > 0 && technique && audit.faces.rowCount > 0
          && audit.faces.faceCount > 0) {
          const [headerBytes, recordBytes] = await Promise.all([
            readBufferBinding(device, technique.leafHeaders, audit.faces.rowCount * 48),
            readBufferBinding(device, technique.powerFaces, audit.faces.faceCount * 32),
          ]);
          const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength / 4);
          const records = new Uint32Array(recordBytes.buffer, recordBytes.byteOffset, recordBytes.byteLength / 4);
          const targets = new Set(powerBoundaryQueryAuditKeys), targetRows = new Set<number>();
          for (let row = 0; row < audit.faces.rowCount; row += 1) {
            const cell = headers[row * 12];
            if (!targets.has(cell)) continue;
            pressureRows.push({ cell, row, size: headers[row * 12 + 3] }); targetRows.add(row);
          }
          for (let faceIndex = 0; faceIndex < audit.faces.faceCount; faceIndex += 1) {
            const base = faceIndex * 8, negativeRow = records[base], positiveRow = records[base + 1];
            if (!targetRows.has(negativeRow) && !targetRows.has(positiveRow)) continue;
            facesTouchingTargets.push({ face: faceIndex, negativeRow, positiveRow,
              negativeCell: headers[negativeRow * 12],
              positiveCell: positiveRow < audit.faces.rowCount ? headers[positiveRow * 12] : undefined,
              slot: records[base + 2] & 0xffff, flags: records[base + 3] });
          }
        }
        if (powerBoundaryQueryAuditKeys.length > 0 && audited.powerOwnerArena) {
          const arenaBytes = await readBufferBinding(device, { buffer: audited.powerOwnerArena }, audited.powerOwnerArena.size);
          const arena = new Uint32Array(arenaBytes.buffer, arenaBytes.byteOffset, arenaBytes.byteLength / 4);
          const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
          const maximumLeaf = solver.info.quadtreeMaximumFluidScale ?? 16;
          const canonical = (cell: [number, number, number]) => {
            let size = Math.min(maximumLeaf, 8);
            let origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
            while (size > 1 && origin.some((value, axis) => value + size > dimensions[axis])) {
              size /= 2; origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
            }
            return { origin, size, invalid: false };
          };
          for (const cellIndex of powerBoundaryQueryAuditKeys) {
            const cell: [number, number, number] = [cellIndex % dimensions[0],
              Math.floor(cellIndex / dimensions[0]) % dimensions[1],
              Math.floor(cellIndex / (dimensions[0] * dimensions[1]))];
            if (arena.length <= 15 || arena[15] !== 0x4f57_4e52) {
              ownerTopology.push({ cell: cellIndex, ...canonical(cell), invalid: true }); continue;
            }
            const brickDimensions = dimensions.map((value) => Math.ceil(value / 8));
            const brick = cell.map((value) => Math.floor(value / 8));
            const logical = brick[0] + brick[1] * brickDimensions[0] + brick[2] * brickDimensions[0] * brickDimensions[1];
            const hashCapacity = (arena[5] - 16) / 2;
            let slot = (Math.imul(logical, 0x9e37_79b1) >>> 0) % hashCapacity, encoded = 0;
            for (let probe = 0; probe < hashCapacity; probe += 1) {
              const observed = arena[16 + slot];
              if (observed === logical + 1) { encoded = arena[16 + hashCapacity + slot]; break; }
              if (observed === 0) break; slot = slot + 1 === hashCapacity ? 0 : slot + 1;
            }
            if (encoded === 0) { ownerTopology.push({ cell: cellIndex, ...canonical(cell) }); continue; }
            if (encoded === 0xffff_ffff || encoded > arena[3]) {
              ownerTopology.push({ cell: cellIndex, ...canonical(cell), invalid: true }); continue;
            }
            const local = (cell[0] & 7) + 8 * ((cell[1] & 7) + 8 * (cell[2] & 7));
            const packed = arena[arena[6] + (encoded - 1) * 512 + local];
            if (packed === 0 || packed === 0xffff_ffff) {
              ownerTopology.push({ cell: cellIndex, ...canonical(cell), packed }); continue;
            }
            if (packed === 0x8000_0000) {
              ownerTopology.push({ cell: cellIndex, origin: cell, size: 1, invalid: false, packed }); continue;
            }
            const exponent = packed & 7, size = 1 << exponent;
            const origin = [((packed >>> 3) & 511) << exponent, ((packed >>> 12) & 511) << exponent,
              ((packed >>> 21) & 511) << exponent] as [number, number, number];
            ownerTopology.push({ cell: cellIndex, origin, size, packed,
              invalid: (packed & 0xc000_0000) !== 0 || exponent < 1 || exponent > 5
                || origin.some((value, axis) => cell[axis] < value || cell[axis] >= value + size) });
          }
        }
        const diagnosticSolver = solver as GPUSolverInstance;
        if (powerBoundaryQueryAuditKeys.length > 0 && diagnosticSolver.surfaceFieldTexture) {
          const densePhi = await readFloatTexture3D(device, diagnosticSolver.surfaceFieldTexture,
            solver.info.nx, solver.info.ny, solver.info.nz);
          for (const cell of powerBoundaryQueryAuditKeys) {
            const value = densePhi[cell]; legacyPhiSamples.push({ cell, phi: Number.isFinite(value) ? value : null });
          }
        }
        const summaryBytes = audited.globalFineSummaryDirectory
          ? await readBufferBinding(device, { buffer: audited.globalFineSummaryDirectory },
            powerBoundaryQueryAuditKeys.length > 0 ? audited.globalFineSummaryDirectory.size : 64)
          : undefined;
        const candidateBytes = audited.adaptiveSurfaceCandidateControl
          ? await readBufferBinding(device, { buffer: audited.adaptiveSurfaceCandidateControl }, 32) : undefined;
        const summary = summaryBytes
          ? new Uint32Array(summaryBytes.buffer, summaryBytes.byteOffset, summaryBytes.byteLength / 4) : undefined;
        const candidates = candidateBytes
          ? new Uint32Array(candidateBytes.buffer, candidateBytes.byteOffset, 8) : undefined;
        const fineSummaries: Array<{
          key: number; found: boolean; complete: boolean; coarseAuthority: boolean; centerValid: boolean;
          minimumPhi?: number; maximumPhi?: number; minimumAbsolutePhi?: number; centerPhi?: number;
          sampleCount?: number; brickCount?: number; flags?: number;
        }> = [];
        if (summary && powerBoundaryQueryAuditKeys.length > 0 && summary.length >= 16) {
          const orderedFloat = (word: number) => {
            const mask = (word & 0x8000_0000) === 0 ? 0xffff_ffff : 0x8000_0000;
            return new Float32Array(new Uint32Array([(word ^ mask) >>> 0]).buffer)[0];
          };
          const summaryHash = (key: number) => {
            let value = Math.imul(key, 0x9e37_79b1) >>> 0;
            value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb_352d) >>> 0;
            return (value ^ (value >>> 15)) >>> 0;
          };
          for (const key of powerBoundaryQueryAuditKeys) {
            let base = -1, slot: number = summaryHash(key) & (summary[3] - 1);
            for (let probe = 0; probe < Math.min(summary[8], 32); probe += 1) {
              const candidate = 16 + slot * 8;
              if (candidate + 7 >= summary.length || summary[candidate] === 0xffff_ffff) break;
              if (summary[candidate] === key) { base = candidate; break; }
              slot = (slot + 1) & (summary[3] - 1);
            }
            if (base < 0) { fineSummaries.push({ key, found: false, complete: false,
              coarseAuthority: false, centerValid: false }); continue; }
            const flags = summary[base + 6], sampleCount = summary[base + 4], brickCount = summary[base + 5];
            const fineComplete = brickCount === 1 && sampleCount === summary[11]
              && (summary[11] === 64 || summary[11] === 512);
            fineSummaries.push({ key, found: true,
              complete: (flags & 0x8000_0000) !== 0 || fineComplete,
              coarseAuthority: (flags & 0x8000_0000) !== 0,
              centerValid: fineComplete && Number.isFinite(new Float32Array(
                new Uint32Array([summary[base + 7]]).buffer)[0]),
              minimumPhi: orderedFloat(summary[base + 1]), maximumPhi: orderedFloat(summary[base + 2]),
              minimumAbsolutePhi: new Float32Array(new Uint32Array([summary[base + 3]]).buffer)[0],
              centerPhi: new Float32Array(new Uint32Array([summary[base + 7]]).buffer)[0],
              sampleCount, brickCount, flags,
            });
          }
        }
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "power-source-audit", step: steps,
          compactionRows: audit.faces.rowCount,
          fineTopology: fine ? { flags: fine.topologyFlags, published: fine.topologyPublished,
            rolledBack: fine.topologyRolledBack, finalizeReason: fine.topologyFinalizeReason,
            desiredBricks: fine.topologyDesiredBricks, activePages: fine.activePages } : undefined,
          coarse: fine ? { generation: fine.coarseGeneration, state: fine.coarseState,
            entries: fine.coarseEntryCount, negative: fine.coarseNegativeEntries, positive: fine.coarsePositiveEntries,
            interfaces: fine.coarseInterfaceEntries, malformed: fine.coarseMalformedEntries } : undefined,
          summary: summary ? { flags: summary[0], generation: summary[1], entries: summary[2],
            hashCapacity: summary[3], dimensions: [summary[4], summary[5], summary[6]], maximumLevel: summary[7],
            maximumProbes: summary[8], state: summary[9], hierarchyKeys: summary[10] } : undefined,
          surfaceCandidates: candidates ? { count: candidates[0], dispatch: [candidates[1], candidates[2], candidates[3]],
            generation: candidates[4], published: candidates[5], error: candidates[6], capacity: candidates[7] }
            : undefined }));
        if (powerBoundaryQueryAuditKeys.length > 0) {
          console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
            phase: "power-boundary-query-audit", step: steps, requestedTime,
            powerGeneration: audit.faces.generation, powerFaceCount: audit.faces.faceCount,
            sampledFineSource: audited.powerBoundaryFineSource,
            currentFineSource: fine ? { configuredGeneration: fine.generation,
              worklistGeneration: fine.worklistGeneration, generationSlot: fine.generationSlot,
              publicationValid: fine.publicationValid } : undefined,
            requestedBrickKeys: powerBoundaryQueryAuditKeys,
            probedPages: fine?.probedPages, coarseRecords: fine?.probedCoarseRecords,
            fineSummaryGeneration: summary?.[1], fineSummaries,
            ownerTopology, legacyPhiSamples, pressureRows, facesTouchingTargets, queryMatches }));
        }
      }
      const failure = !audit.faces.valid || audit.faces.flags !== 0 || audit.faces.rowCount === 0
        || audit.faces.faceCount === 0 || audit.faces.incidenceCount === 0 || audit.faces.invalidCount !== 0
        || audit.descriptor.rowCount !== audit.faces.rowCount
        || audit.descriptor.validCount !== audit.descriptor.rowCount || audit.descriptor.errorCount !== 0
        || audit.descriptor.flags !== 0 || audit.topology.invalidCount !== 0 || audit.topology.flags !== 0
        || audit.topology.resolvedCount !== audit.descriptor.rowCount
        || !audit.transfer.valid || audit.transfer.flags !== 0 || audit.transfer.sourceFlags !== 0
        || !audit.velocityStages.seedValid || audit.velocityStages.seedFlags !== 0
        || audit.velocityStages.operatorFlags !== 0xc000_0000
        || !octreeMGPCGDiagnosticsAreAcceptable(mgpcgDiagnostics)
        || (octreeGlobalFineFactorOverride === "off"
          && (!audit.coarsePhi?.valid || audit.coarsePhi.flags !== 0
            || audit.coarsePhi.advected !== audit.coarsePhi.rowCount))
        || audit.faces.generation <= previousAuditedPowerGeneration;
      if (failure) {
        let oldMeshFailure: unknown;
        const oldMeshFace = audit.oldMeshAdvection?.firstInterpolationFace;
        if (oldMeshFace !== undefined && audited.powerFaceSource
          && oldMeshFace < audit.faces.faceCount) {
          const [recordBytes, normalBytes, centroidBytes] = await Promise.all([
            readBufferBinding(device, { buffer: audited.powerFaceSource.faces,
              offset: oldMeshFace * 32, size: 32 }, 32),
            readBufferBinding(device, { buffer: audited.powerFaceSource.faceNormals,
              offset: oldMeshFace * 16, size: 16 }, 16),
            readBufferBinding(device, { buffer: audited.powerFaceSource.faceCentroids,
              offset: oldMeshFace * 16, size: 16 }, 16),
          ]);
          const recordWords = new Uint32Array(recordBytes.buffer, recordBytes.byteOffset, 8);
          const recordFloats = new Float32Array(recordBytes.buffer, recordBytes.byteOffset, 8);
          oldMeshFailure = { face: oldMeshFace, stage: audit.oldMeshAdvection?.firstInterpolationStage,
            negativeRow: recordWords[0], positiveRow: recordWords[1], geometryCode: recordWords[2],
            flags: recordWords[3], normalVelocity: recordFloats[4], area: recordFloats[5],
            inverseDistance: recordFloats[6], openFraction: recordFloats[7],
            normal: Array.from(new Float32Array(normalBytes.buffer, normalBytes.byteOffset, 4)),
            centroid: Array.from(new Float32Array(centroidBytes.buffer, centroidBytes.byteOffset, 4)) };
        }
        let duplicateSite: { slot: number; cellPlusOne?: number; size?: number; row?: number; published?: number;
          firstInvalidRow: number; firstInvalidHeader?: { cell: number; size: number };
          matchingHeaderRows: number[] } | undefined;
        let reciprocalFace: unknown;
        let invalidGeometry: Record<string, unknown> | undefined;
        let axisIncidence: unknown;
        const coarseFailureReader = solver as GPUSolverInstance & {
          readPowerCoarseFailureRow?: (row: number) => Promise<unknown>;
        };
        const coarsePhiFailure = audit.coarsePhi && !audit.coarsePhi.valid
          && audit.coarsePhi.firstError < audit.coarsePhi.rowCount
          && coarseFailureReader.readPowerCoarseFailureRow
          ? await coarseFailureReader.readPowerCoarseFailureRow(audit.coarsePhi.firstError)
          : undefined;
        const bandFailureReader = solver as GPUSolverInstance & {
          readGlobalFineBandRowFailure?: (row: number) => Promise<unknown>;
        };
        const retainedBandFailure = audit.oldMeshAdvection?.retainedBandFailureAnchor !== undefined
          && audit.oldMeshAdvection.retainedBandFailureAnchor < 0xffff_ffff
          && bandFailureReader.readGlobalFineBandRowFailure
          ? await bandFailureReader.readGlobalFineBandRowFailure(
            audit.oldMeshAdvection.retainedBandFailureAnchor)
          : undefined;
        const slot = audit.faces.firstInvalidSlot;
        // firstInvalid is the ordered failure key. pad3 is detail storage used
        // by later face/boundary stages and is not an authoritative row ID.
        const firstInvalidRow = audit.faces.firstInvalid;
        const internal = audited as typeof audited & { powerFaces?: { siteIndex?: GPUBuffer }; leafHeaders?: GPUBuffer };
        const siteIndex = audited.powerFaceSiteIndex ?? internal.powerFaces?.siteIndex;
        const leafHeaders = audited.powerLeafHeaders ?? internal.leafHeaders;
        if (axisSource && audit.velocityStages.seedFlags === 4
          && audit.velocityStages.seedFirstError < (axis?.[3] ?? 0xffff_ffff)) {
          const row = audit.velocityStages.seedFirstError;
          const countBytes = await readBufferBinding(device, { buffer: axisSource.incidence, offset: row * 4, size: 4 }, 4);
          const count = new Uint32Array(countBytes.buffer, countBytes.byteOffset, 1)[0];
          const boundedCount = Math.min(count, 48);
          const indexOffset = ((axis![3] + row * 48) * 4);
          const indexBytes = boundedCount > 0
            ? await readBufferBinding(device, { buffer: axisSource.incidence, offset: indexOffset, size: boundedCount * 4 }, boundedCount * 4)
            : undefined;
          const indices = indexBytes ? Array.from(new Uint32Array(indexBytes.buffer, indexBytes.byteOffset, boundedCount)) : [];
          const records = await Promise.all(indices.map(async (faceIndex) => {
            if (faceIndex >= axis![0] || faceIndex * 24 + 24 > axisSource.faces.size) return { faceIndex, invalid: true };
            const bytes = await readBufferBinding(device, { buffer: axisSource.faces, offset: faceIndex * 24, size: 24 }, 24);
            const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 6);
            const floats = new Float32Array(bytes.buffer, bytes.byteOffset, 6);
            return { faceIndex, negativeRow: words[0], positiveRow: words[1], packedOrigin: words[2], axisSpan: words[3],
              normalVelocity: floats[4], area: floats[5], incident: words[0] === row || words[1] === row };
          }));
          axisIncidence = { row, count, records };
        }
        if (leafHeaders) {
          const headerBytes = await readBufferBinding(device, { buffer: leafHeaders }, audit.faces.rowCount * 48);
          const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength / 4);
          const firstInvalidHeader = firstInvalidRow < audit.faces.rowCount
            ? { cell: headers[firstInvalidRow * 12], size: headers[firstInvalidRow * 12 + 3] } : undefined;
          const matchingHeaderRows: number[] = [];
          if (firstInvalidHeader) for (let row = 0; row < audit.faces.rowCount; row += 1) {
            if (headers[row * 12] === firstInvalidHeader.cell && headers[row * 12 + 3] === firstInvalidHeader.size) {
              matchingHeaderRows.push(row);
            }
          }
          if ((audit.faces.flags & 64) !== 0) {
            duplicateSite = { slot, firstInvalidRow, firstInvalidHeader, matchingHeaderRows };
          }
          if ((audit.faces.flags & 64) !== 0 && siteIndex && slot !== 0xffff_ffff && slot * 16 + 16 <= siteIndex.size) {
            const siteBytes = await readBufferBinding(device, { buffer: siteIndex, offset: slot * 16, size: 16 }, 16);
            const site = new Uint32Array(siteBytes.buffer, siteBytes.byteOffset, 4);
            Object.assign(duplicateSite!, { cellPlusOne: site[0], size: site[1], row: site[2], published: site[3] });
          }
          if ((audit.faces.flags & 8) !== 0 && firstInvalidHeader
            && slot !== 0xffff_ffff && audited.powerTopologyMetrics
            && audited.powerCatalogEntryHeaders && audited.powerCatalogFaces) {
            const metricBytes = await readBufferBinding(device,
              { buffer: audited.powerTopologyMetrics, offset: firstInvalidRow * 16, size: 16 }, 16);
            const metricWords = new Uint32Array(metricBytes.buffer, metricBytes.byteOffset, 4);
            const metricFloats = new Float32Array(metricBytes.buffer, metricBytes.byteOffset, 4);
            const topologyCode = metricWords[0], transformAndFlags = metricWords[1];
            if (topologyCode * 8 + 8 <= audited.powerCatalogEntryHeaders.size) {
              const entryBytes = await readBufferBinding(device,
                { buffer: audited.powerCatalogEntryHeaders, offset: topologyCode * 8, size: 8 }, 8);
              const entry = new Uint32Array(entryBytes.buffer, entryBytes.byteOffset, 2);
              if (slot < entry[1] && (entry[0] + slot) * 48 + 48 <= audited.powerCatalogFaces.size) {
                const rawBytes = await readBufferBinding(device,
                  { buffer: audited.powerCatalogFaces, offset: (entry[0] + slot) * 48, size: 48 }, 48);
                const raw = Array.from(new Float32Array(rawBytes.buffer, rawBytes.byteOffset, 12));
                const transform = transformAndFlags & 63;
                const inverse = (value: number[]) => {
                  const signed = value.map((component, axis) => component * ((transform & (1 << axis)) !== 0 ? -1 : 1));
                  const permutations = [[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]];
                  return permutations[Math.floor(transform / 8) % 6].map((axis) => signed[axis]);
                };
                const origin = [firstInvalidHeader.cell % solver.info.nx,
                  Math.floor(firstInvalidHeader.cell / solver.info.nx) % solver.info.ny,
                  Math.floor(firstInvalidHeader.cell / (solver.info.nx * solver.info.ny))];
                const spacing = scene.container.width_m / solver.info.nx;
                const size_m = firstInvalidHeader.size * spacing;
                const center = origin.map((component) => (component + 0.5 * firstInvalidHeader.size) * spacing);
                const neighborOffset = inverse(raw.slice(0, 3));
                const centroidOffset = inverse(raw.slice(5, 8));
                const unnormalizedNormal = inverse(raw.slice(8, 11));
                const normalLength = Math.hypot(...unnormalizedNormal);
                invalidGeometry = {
                  row: firstInvalidRow, header: firstInvalidHeader, topologyCode, transformAndFlags,
                  metricVolume: metricFloats[2], slot, entry: { firstFace: entry[0], faceCount: entry[1] }, raw,
                  reconstructed: {
                    neighborCenter: center.map((component, axis) => component + size_m * neighborOffset[axis]),
                    neighborSize: size_m * raw[3],
                    centroid: center.map((component, axis) => component + size_m * centroidOffset[axis]),
                    area: size_m * size_m * raw[4],
                    unnormalizedNormal, normalLength,
                    normal: unnormalizedNormal.map((component) => component / normalLength),
                    inverseDistance: raw[11] / size_m,
                  },
                };
                const sampledFine = audited.powerBoundaryFineLevelSetSource;
                if (sampledFine) {
                  const reconstructed = invalidGeometry.reconstructed as {
                    neighborCenter: [number, number, number];
                  };
                  const liquidCenter = center as [number, number, number];
                  const airCenter = reconstructed.neighborCenter;
                  const probeKeys = [...new Set([
                    ...fineTrilinearBrickKeysAtPosition(sampledFine, liquidCenter),
                    ...fineTrilinearBrickKeysAtPosition(sampledFine, airCenter),
                  ])];
                  const sampledFineDiagnostics = await readGlobalFineGenerationDiagnostics(
                    device, solver, probeKeys, sampledFine,
                  );
                  invalidGeometry.sampledFine = sampledFineDiagnostics ? {
                    generation: sampledFineDiagnostics.generation,
                    generationSlot: sampledFineDiagnostics.generationSlot,
                    publicationValid: sampledFineDiagnostics.publicationValid,
                    liquidCenter, airCenter, probeKeys,
                    pages: sampledFineDiagnostics.probedPages,
                  } : undefined;
                }
                if (audited.globalFineSummaryDirectory) {
                  const summaryBytes = await readBufferBinding(device,
                    { buffer: audited.globalFineSummaryDirectory }, audited.globalFineSummaryDirectory.size);
                  const summary = new Uint32Array(summaryBytes.buffer, summaryBytes.byteOffset, summaryBytes.byteLength / 4);
                  const baseDims = [summary[4], summary[5], summary[6]];
                  const ratios = baseDims.map((value, axis) => value / [solver.info.nx, solver.info.ny, solver.info.nz][axis]);
                  const bricksPerCell = ratios[0];
                  const brickSide = firstInvalidHeader.size * bricksPerCell;
                  let remaining = brickSide, levelOffset = 0;
                  let levelDims = [...baseDims];
                  while (remaining > 1) {
                    levelOffset += levelDims[0] * levelDims[1] * levelDims[2];
                    levelDims = levelDims.map((value) => Math.ceil(value / 2));
                    remaining /= 2;
                  }
                  const coordinate = origin.map((value) => value * bricksPerCell / brickSide);
                  const key = levelOffset + coordinate[0] + levelDims[0] * (coordinate[1] + levelDims[1] * coordinate[2]);
                  const hashCapacity = summary[3], maximumProbes = Math.min(summary[8], 32);
                  const hash = (value: number) => {
                    value = Math.imul(value, 0x9e3779b1) >>> 0;
                    value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
                    return (value ^ (value >>> 15)) >>> 0;
                  };
                  let entryOffset: number | undefined;
                  if (hashCapacity > 0 && (hashCapacity & (hashCapacity - 1)) === 0) {
                    let hashSlot = hash(key) & (hashCapacity - 1);
                    for (let probe = 0; probe < maximumProbes; probe += 1) {
                      const base = 16 + hashSlot * 8;
                      if (summary[base] === 0xffff_ffff) break;
                      if (summary[base] === key) { entryOffset = base; break; }
                      hashSlot = (hashSlot + 1) & (hashCapacity - 1);
                    }
                  }
                  const floatWord = (word: number) => new Float32Array(new Uint32Array([word]).buffer)[0];
                  const orderedFloat = (word: number) => floatWord(word ^ ((word & 0x8000_0000) === 0 ? 0xffff_ffff : 0x8000_0000));
                  const expectedBricks = brickSide ** 3;
                  const entryData = entryOffset === undefined ? undefined : {
                    minimumPhi: orderedFloat(summary[entryOffset + 1]),
                    maximumPhi: orderedFloat(summary[entryOffset + 2]),
                    minimumAbsolutePhi: floatWord(summary[entryOffset + 3]),
                    sampleCount: summary[entryOffset + 4], brickCount: summary[entryOffset + 5],
                    flags: summary[entryOffset + 6], authority: summary[entryOffset + 7],
                    complete: (summary[entryOffset + 7] & 1) !== 0
                      || (summary[entryOffset + 5] === expectedBricks && summary[entryOffset + 4] === expectedBricks * 64),
                  };
                  invalidGeometry.fineSummary = {
                    header: Array.from(summary.slice(0, 11)), baseDims, ratios, brickSide, key,
                    expectedBricks, found: entryOffset !== undefined, entry: entryData,
                  };
                }
              }
            }
          }
          const neighborRow = audit.faces.firstInvalidNeighbor;
          if ((audit.faces.flags & 16) !== 0 && firstInvalidRow < audit.faces.rowCount && neighborRow < audit.faces.rowCount
            && audited.powerTopologyMetrics && audited.powerCatalogEntryHeaders && audited.powerCatalogFaces) {
            const readMetric = async (row: number) => {
              const bytes = await readBufferBinding(device,
                { buffer: audited.powerTopologyMetrics!, offset: row * 16, size: 16 }, 16);
              const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
              return { topologyCode: words[0], transformAndFlags: words[1] };
            };
            const [rowMetric, neighborMetric] = await Promise.all([readMetric(firstInvalidRow), readMetric(neighborRow)]);
            const readDescriptor = async (row: number) => {
              if (!audited.powerDescriptorRows) return undefined;
              const bytes = await readBufferBinding(device,
                { buffer: audited.powerDescriptorRows, offset: row * 4, size: 4 }, 4);
              return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
            };
            const [rowDescriptor, neighborDescriptor] = await Promise.all([
              readDescriptor(firstInvalidRow), readDescriptor(neighborRow),
            ]);
            const readEntry = async (topologyCode: number) => {
              if (topologyCode === 0xffff_ffff || topologyCode * 8 + 8 > audited.powerCatalogEntryHeaders!.size) {
                return { firstFace: 0, faceCount: 0 };
              }
              const bytes = await readBufferBinding(device,
                { buffer: audited.powerCatalogEntryHeaders!, offset: topologyCode * 8, size: 8 }, 8);
              const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 2);
              return { firstFace: words[0], faceCount: words[1] };
            };
            const [rowEntry, neighborEntry] = await Promise.all([
              readEntry(rowMetric.topologyCode), readEntry(neighborMetric.topologyCode),
            ]);
            const readFaces = async (firstFace: number, faceCount: number) => {
              if (faceCount === 0) return new Float32Array();
              const bytes = await readBufferBinding(device, { buffer: audited.powerCatalogFaces!,
                offset: firstFace * 48, size: faceCount * 48 }, faceCount * 48);
              return new Float32Array(bytes.buffer, bytes.byteOffset, faceCount * 12);
            };
            const [rowFaces, neighborFaces] = await Promise.all([
              readFaces(rowEntry.firstFace, rowEntry.faceCount),
              readFaces(neighborEntry.firstFace, neighborEntry.faceCount),
            ]);
            const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
            const header = (row: number) => ({ cell: headers[row * 12], size: headers[row * 12 + 3] });
            const origin = ({ cell }: { cell: number }) => [cell % dimensions[0],
              Math.floor(cell / dimensions[0]) % dimensions[1], Math.floor(cell / (dimensions[0] * dimensions[1]))];
            const inverseTransform = (value: number[], code: number) => {
              const q = value.map((component, axis) => component * (((code & (1 << axis)) !== 0) ? -1 : 1));
              const permutations = [[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]];
              return permutations[Math.floor(code / 8) % 6].map((axis) => q[axis]);
            };
            const reconstruct = (faces: Float32Array, selector: number, anchor: { cell: number; size: number }, transform: number) => {
              const raw = Array.from(faces.slice(selector * 12, selector * 12 + 12));
              const anchorOrigin = origin(anchor);
              const anchorCenter = anchorOrigin.map((component) => component + 0.5 * anchor.size);
              const offset = inverseTransform(raw.slice(0, 3), transform & 63);
              return { selector, raw, neighborCenter: anchorCenter.map((component, axis) => component + anchor.size * offset[axis]),
                neighborSize: anchor.size * raw[3] };
            };
            const rowHeader = header(firstInvalidRow), neighborHeader = header(neighborRow);
            const rowCenter = origin(rowHeader).map((component) => component + 0.5 * rowHeader.size);
            const neighborCenter = origin(neighborHeader).map((component) => component + 0.5 * neighborHeader.size);
            const forward = reconstruct(rowFaces, audit.faces.firstInvalidSlot, rowHeader, rowMetric.transformAndFlags);
            const candidates = Array.from({ length: neighborEntry.faceCount }, (_, selector) => {
              const candidate = reconstruct(neighborFaces, selector, neighborHeader, neighborMetric.transformAndFlags);
              return { ...candidate,
                centerError: Math.max(...candidate.neighborCenter.map((component, axis) => Math.abs(component - rowCenter[axis]))),
                sizeError: Math.abs(candidate.neighborSize - rowHeader.size) };
            }).sort((a, b) => (a.centerError + a.sizeError) - (b.centerError + b.sizeError));
            const lookupSite = (targetHeader: { cell: number; size: number }, words: Uint32Array) => {
              let value = (targetHeader.cell ^ Math.imul(targetHeader.size, 0x9e3779b9)) >>> 0;
              value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
              value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
              let hashSlot = (value ^ (value >>> 16)) & (words.length / 4 - 1);
              for (let probe = 0; probe < 32; probe += 1) {
                const at = hashSlot * 4;
                if (words[at] === 0) return undefined;
                if (words[at] === targetHeader.cell + 1 && words[at + 1] === targetHeader.size) {
                  return { slot: hashSlot, cellPlusOne: words[at], size: words[at + 1], row: words[at + 2], published: words[at + 3] };
                }
                hashSlot = (hashSlot + 1) & (words.length / 4 - 1);
              }
              return undefined;
            };
            let siteEntries: unknown;
            if (siteIndex) {
              const bytes = await readBufferBinding(device, { buffer: siteIndex }, siteIndex.size);
              const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
              siteEntries = { row: lookupSite(rowHeader, words), neighbor: lookupSite(neighborHeader, words) };
            }
            reciprocalFace = { row: { index: firstInvalidRow, header: rowHeader, center: rowCenter,
              descriptor: rowDescriptor, ...rowMetric, entry: rowEntry },
              neighbor: { index: neighborRow, header: neighborHeader, center: neighborCenter,
                descriptor: neighborDescriptor, ...neighborMetric, entry: neighborEntry },
              forward, closestReverseCandidates: candidates.slice(0, 4), siteEntries };
          }
        }
        // Decode the same stable stage/detail/location representation consumed
        // by the ordinary viewport. Raw control words remain alongside it for
        // low-level diagnosis, but Dawn may not silently classify a UI-visible
        // rejection differently.
        const uiFailure = viewportFailureIndicator({ ...await solver.readStats() }, undefined, scene);
        throw new Error(`power generation audit failed at step ${steps}: ${JSON.stringify({ uiFailure, ...audit, oldMeshFailure, retainedBandFailure, coarsePhiFailure, duplicateSite, invalidGeometry, reciprocalFace, axisIncidence,
          diagnosticBuffers: { siteIndex: Boolean(siteIndex), siteIndexSize: siteIndex?.size, leafHeaders: Boolean(leafHeaders) } })}`);
      }
      // Retain the first cross-scale generation only to diagnose when a live
      // topology was lost. Aanjaneya et al. has no Section 6.4 smoothing rule,
      // and this historical record cannot satisfy final-state acceptance.
      if (scenarioId === "minimal-power-dam-break" && !powerTransitionWitness) {
        powerTransitionWitness = await readPowerTransitionWitness(device, solver, steps,
          audit.faces.generation, audit.faces.rowCount, audit.faces.faceCount);
      }
      previousAuditedPowerGeneration = audit.faces.generation;
      powerGenerationAuditedSteps += 1;
    }
    if (steps === oracleSteps) {
      const samplingStartedAt = performance.now();
      await awaitAdvanceCompletion();
      solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, solver.info.submittedTime_s ?? 0);
      solver.info.simulatedTime_s = solver.info.submittedTime_s;
      if (exactStepCount !== undefined) await solver.readStats();
      if (!collectStabilityEnvelope && !performanceProfileRequested) matched = await readCubicVolumeField(device, solver);
      samplingWall_ms += performance.now() - samplingStartedAt;
    }
    if (steps % 30 === 0) await awaitAdvanceCompletion();
    const shouldReport = reportEvery > 0 && steps % reportEvery === 0;
    const shouldSampleEnergy = energyEverySteps > 0 && steps % energyEverySteps === 0;
    if (shouldReport || shouldSampleEnergy || collectStabilityEnvelope) {
      const samplingStartedAt = performance.now();
      await awaitAdvanceCompletion();
      solver.info.simulatedTime_s = solver.info.submittedTime_s;
      const sample = await solver.readStats();
      const isRestrictedTall = sample.gridKind === "restricted-tall-cell";
      let tallCellActivity: ReturnType<typeof inspectColumnBases> | undefined, tallVolumeGaps: ReturnType<typeof inspectTallVolumeGaps> | undefined;
      let bases: Float32Array | undefined;
      if (isRestrictedTall) {
        bases = await readFloatTexture2D(device, solver.columnBaseTexture, sample.nx, sample.nz);
        tallCellActivity = inspectColumnBases(bases, sample.nx, sample.nz, sample.ny, sample.regularLayers, sample.maximumNeighborDelta);
        tallVolumeGaps = inspectTallVolumeGaps(await readFloatTexture3D(device, solver.volumeTexture, sample.nx, sample.storedNy, sample.nz), bases, sample.nx, sample.storedNy, sample.nz, sample.ny, sample.maximumNeighborDelta);
      }
      const exact = await readCubicVolumeField(device, solver);
      if (steps === oracleSteps) matched = exact;
      const stagedSolver = solver as GPUSolverInstance & {
        preProjectionVelocityTexture?: GPUTexture;
        velocityTexture?: GPUTexture;
      };
      // U3 deliberately shrinks the legacy dense velocity textures to 1x1x1.
      // Once the compact face source exists, attempting a cubic readback is
      // both semantically wrong and a WebGPU validation error.
      const compactFaceVelocity = stagedSolver.adaptiveFaceVelocitySource !== undefined;
      const spacing = {
        x: scene.container.width_m / sample.nx,
        y: scene.container.height_m / sample.ny,
        z: scene.container.depth_m / sample.nz
      };
      const readStagedVelocity = (texture: GPUTexture | undefined) => {
        if (!texture || compactFaceVelocity) return Promise.resolve(undefined);
        return isRestrictedTall && bases
          ? readTallVelocityTexture3D(device, texture, sample.nx, sample.storedNy, sample.nz, sample.ny, bases, exact.field, spacing, stepDt)
          : readVelocityTexture3D(device, texture, sample.nx, sample.ny, sample.nz, exact.field, spacing, stepDt);
      };
      const preProjectionVelocity = await readStagedVelocity(stagedSolver.preProjectionVelocityTexture);
      const postProjectionVelocity = await readStagedVelocity(stagedSolver.velocityTexture);
      // §14 audit checklist: an extremum without a location cannot separate an
      // interface, tall endpoint, wall, or remesh artifact. Locations arrive
      // in world (cubic) coordinates from the reduction kernels.
      const classifyLocation = (location?: { x: number; y: number; z: number }) => {
        if (!location || !bases) return undefined;
        const base = Math.round(bases[location.x + sample.nx * location.z] ?? 0);
        const region = base > 0 && location.y < base
          ? (location.y === 0 ? "tall-bottom-endpoint" : location.y >= base - 1 ? "tall-top-endpoint" : "tall-interior")
          : "band";
        const wall = location.x === 0 || location.x === sample.nx - 1 || location.z === 0 || location.z === sample.nz - 1 || location.y === 0 || location.y === sample.ny - 1;
        return { ...location, base, region, wall };
      };
      const extrema = isRestrictedTall ? {
        maxSpeed: classifyLocation(sample.maxSpeedLocation),
        maxAirSpeed: classifyLocation(sample.maxAirSpeedLocation),
        divergenceBefore: classifyLocation(sample.maxDivergenceBeforeLocation),
        divergenceAfter: classifyLocation(sample.maxDivergenceAfterLocation),
        pressure: classifyLocation(sample.maxPressureLocation),
        pressureResidual: classifyLocation(sample.maxPressureResidualLocation)
      } : undefined;
      const exactReference = spatialExactReference ?? referenceVolumeCells(sample);
      const exactVolumeDrift = (exact.summary.cellSum - exactReference) / Math.max(1, Math.abs(exactReference));
      if (stabilityEnvelope && (!Number.isFinite(exact.summary.minimum)
        || !Number.isFinite(exact.summary.maximum) || !Number.isFinite(exact.summary.cellSum)
        || exact.summary.minimum < -0.01 || exact.summary.maximum > 1.5
        || exact.summary.cellSum <= 1 || exact.summary.cellSum >= exact.field.length - 1)) {
        stabilityEnvelope.invalidVolumeSampleCount += 1;
      }
      if (shouldSampleEnergy && preProjectionVelocity && postProjectionVelocity) {
        const potential = gravitationalPotentialEnergyProxy(exact.field, sample.nx, sample.ny, sample.nz, spacing, scene.fluid.gravity_m_s2);
        const preMechanical = preProjectionVelocity.kineticEnergyProxy + potential;
        const postMechanical = postProjectionVelocity.kineticEnergyProxy + potential;
        const energySample: MechanicalEnergySample = {
          time_s: sample.simulatedTime_s ?? solver.info.submittedTime_s ?? 0,
          gravitationalPotentialEnergyProxy: potential,
          preProjectionKineticEnergyProxy: preProjectionVelocity.kineticEnergyProxy,
          postProjectionKineticEnergyProxy: postProjectionVelocity.kineticEnergyProxy,
          preProjectionMechanicalEnergyProxy: preMechanical,
          postProjectionMechanicalEnergyProxy: postMechanical,
          projectionEnergyDelta: postProjectionVelocity.kineticEnergyProxy - preProjectionVelocity.kineticEnergyProxy,
          sampledIntervalEnergyDelta: postMechanical - previousSampledMechanicalEnergy,
          preProjectionMaximumDivergence_s: preProjectionVelocity.maximumLiquidDivergence_s,
          postProjectionMaximumDivergence_s: postProjectionVelocity.maximumLiquidDivergence_s,
          maximumDivergenceRatio: postProjectionVelocity.maximumLiquidDivergence_s / Math.max(preProjectionVelocity.maximumLiquidDivergence_s, 1e-30),
          preProjectionRmsDivergence_s: preProjectionVelocity.rmsLiquidDivergence_s,
          postProjectionRmsDivergence_s: postProjectionVelocity.rmsLiquidDivergence_s,
          rmsDivergenceRatio: postProjectionVelocity.rmsLiquidDivergence_s / Math.max(preProjectionVelocity.rmsLiquidDivergence_s, 1e-30),
          pressureResidual: sample.pressureResidual ?? 0,
          pressureRelativeResidual: sample.pressureRelativeResidual ?? 0,
          exactVolumeDrift
        };
        energyTrace.push(energySample);
        previousSampledMechanicalEnergy = postMechanical;
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod, phase: "energy", ...energySample }));
      }
      if (stabilityEnvelope && preProjectionVelocity && postProjectionVelocity) {
        const dominantFraction = exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1;
        stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(stabilityEnvelope.peakLiquidSpeed_m_s, postProjectionVelocity.liquidMaximum);
        stabilityEnvelope.peakComponentCfl = Math.max(stabilityEnvelope.peakComponentCfl, postProjectionVelocity.maximumComponentCfl);
        stabilityEnvelope.peakKineticEnergyProxy = Math.max(stabilityEnvelope.peakKineticEnergyProxy, postProjectionVelocity.kineticEnergyProxy);
        // Floor the denominator at a meaningful fraction of slosh-scale
        // energy: the startup steps legitimately create the velocity field
        // from near-zero KE (pre ~1e-4), and a ratio on that denominator
        // measures noise, not amplification.
        stabilityEnvelope.maximumProjectionEnergyRatio = Math.max(stabilityEnvelope.maximumProjectionEnergyRatio, postProjectionVelocity.kineticEnergyProxy / Math.max(preProjectionVelocity.kineticEnergyProxy, 0.01));
        // The quadtree residual readback is asynchronous, so the first steps
        // can legitimately sample before any residual exists; later steps
        // without one still fail hard.
        stabilityEnvelope.maximumPressureRelativeResidual = Math.max(stabilityEnvelope.maximumPressureRelativeResidual, sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(stabilityEnvelope.maximumProjectedVariationalResidual, sample.pressureResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumExactVolumeDrift = Math.max(stabilityEnvelope.maximumExactVolumeDrift, Math.abs(exactVolumeDrift));
        stabilityEnvelope.maximumLevelSetMismatchFraction = Math.max(stabilityEnvelope.maximumLevelSetMismatchFraction, sample.quadtreeLevelSetMismatchFraction ?? 0);
        stabilityEnvelope.maximumComponentCount = Math.max(stabilityEnvelope.maximumComponentCount, exact.summary.componentCount);
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(stabilityEnvelope.minimumDominantComponentFraction, dominantFraction);
        stabilityEnvelope.nonFiniteVelocityCount += preProjectionVelocity.nonFiniteCount + postProjectionVelocity.nonFiniteCount;
        stabilityEnvelope.sampledSteps += 1;
      }
      if (stabilityEnvelope && compactFaceVelocity) {
        const dominantFraction = exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1;
        stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(stabilityEnvelope.peakLiquidSpeed_m_s, sample.maxSpeed_m_s ?? 0);
        stabilityEnvelope.peakComponentCfl = Math.max(stabilityEnvelope.peakComponentCfl, sample.maxComponentCfl ?? 0);
        stabilityEnvelope.maximumExactVolumeDrift = Math.max(stabilityEnvelope.maximumExactVolumeDrift, Math.abs(exactVolumeDrift));
        stabilityEnvelope.maximumComponentCount = Math.max(stabilityEnvelope.maximumComponentCount, exact.summary.componentCount);
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(stabilityEnvelope.minimumDominantComponentFraction, dominantFraction);
        stabilityEnvelope.nonFiniteVelocityCount += sample.nonFiniteCount ?? 0;
        stabilityEnvelope.maximumPressureRelativeResidual = Math.max(stabilityEnvelope.maximumPressureRelativeResidual,
          sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(stabilityEnvelope.maximumProjectedVariationalResidual,
          sample.pressureResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.sampledSteps += 1;
      }
      if (shouldReport) console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "running", steps, simulatedTime_s: sample.simulatedTime_s, dt_s: stepDt, preProjectionVelocity, postProjectionVelocity, maxSpeed_m_s: sample.maxSpeed_m_s, maxAirSpeed_m_s: sample.maxAirSpeed_m_s, maxDivergenceBefore_s: sample.maxDivergenceBefore_s, maxDivergenceAfter_s: sample.maxDivergenceAfter_s, pressureRelativeResidual: sample.pressureRelativeResidual, pressureIterationsUsed: sample.quadtreePressureIterationsUsed, pressureIterationBudget: sample.quadtreePressureIterationBudget, pressureIterationHardBudget: sample.quadtreePressureIterationHardBudget, pressureConverged: sample.quadtreePressureConverged, velocityClampCount: sample.quadtreeVelocityClampCount, factorLevelCount: sample.quadtreeFactorLevelCount, pressurePhaseTimings: sample.quadtreePressurePhaseTimings, maxComponentCfl: sample.maxComponentCfl, representedVolumeDrift: sample.representedVolumeDrift, volumeCorrectionNormalSpeed_cells_s: sample.volumeCorrectionNormalSpeed_cells_s, volumeCorrectionDivergenceRate_s: sample.volumeCorrectionDivergenceRate_s, phiInterfaceCellCount: sample.phiInterfaceCellCount, exactVolumeCellSum: exact.summary.cellSum, exactVolumeDrift, componentCount: exact.summary.componentCount, dominantComponentFraction: exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1, quadtree: sample.gridKind === "quadtree-tall-cell" ? { opticalLayerMode: sample.quadtreeOpticalLayerMode, opticalAlpha: sample.quadtreeOpticalAlpha, opticalMinimumCells: sample.quadtreeOpticalMinimumCells, opticalMaximumCells: sample.quadtreeOpticalMaximumCells, leafCount: sample.quadtreeLeafCount, pressureSampleCount: sample.quadtreePressureSampleCount, liquidDofCount: sample.quadtreeLiquidDofCount, faceCount: sample.quadtreeFaceCount, tallSegmentCount: sample.quadtreeTallSegmentCount, ghostFaceCount: sample.quadtreeGhostFaceCount, maximumNeighborRatio: sample.quadtreeMaximumNeighborRatio, maximumFluidScale: sample.quadtreeMaximumFluidScale, levelSetMismatchFraction: sample.quadtreeLevelSetMismatchFraction } : undefined, stabilityFlags: sample.stabilityFlags, extrema, tallCellActivity, tallVolumeGaps }));
      samplingWall_ms += performance.now() - samplingStartedAt;
    }
    if (checkpointEvery_s > 0 && (solver.info.submittedTime_s ?? 0) + 1e-9 >= nextCheckpoint_s) {
      const samplingStartedAt = performance.now();
      await awaitAdvanceCompletion();
      const cubic = steps === oracleSteps && matched ? matched : await readCubicVolumeField(device, solver);
      let preProjectionVelocity: Float32Array | undefined, postProjectionVelocity: Float32Array | undefined;
      if (singleTallCellProbe && solver.info.gridKind === "restricted-tall-cell") {
        const bases = await readFloatTexture2D(device, solver.columnBaseTexture, solver.info.nx, solver.info.nz);
        const staged = solver as GPUSolverInstance & { preProjectionVelocityTexture?: GPUTexture; velocityTexture?: GPUTexture };
        if (staged.preProjectionVelocityTexture) preProjectionVelocity = await readTallVelocityField3D(device, staged.preProjectionVelocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
        if (staged.velocityTexture) postProjectionVelocity = await readTallVelocityField3D(device, staged.velocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
      }
      const raster = rasterCheckpointRequested && method.id === "octree"
        ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies,
          globalFineGenerationTransitionRequested)
        : undefined;
      const globalFineGeneration = globalFineGenerationTransitionRequested && method.id === "octree"
        ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
      checkpoints.push({ time_s: solver.info.submittedTime_s ?? 0, field: cubic.field, summary: cubic.summary,
        raster, globalFineGeneration, preProjectionVelocity, postProjectionVelocity });
      while (nextCheckpoint_s <= (solver.info.submittedTime_s ?? 0) + 1e-9) nextCheckpoint_s += checkpointEvery_s;
      samplingWall_ms += performance.now() - samplingStartedAt;
    }
    if (lost) throw new Error(`${method.id} device lost: ${lost.message || lost.reason}`);
  }
  const simulationWall_ms = Math.max(0, performance.now() - runStarted - samplingWall_ms);
  await awaitAdvanceCompletion();
  solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, solver.info.submittedTime_s ?? 0);
  solver.info.simulatedTime_s = solver.info.submittedTime_s;
  const info = { ...await solver.readStats() };
  if (info.gpuValidationError && !validationErrors.includes(info.gpuValidationError)) {
    validationErrors.push(info.gpuValidationError);
  }
  matched ??= performanceProfileRequested
    ? { field: new Float32Array(info.nx * info.ny * info.nz),
      summary: summarizeScalarField(new Float32Array(info.nx * info.ny * info.nz), info.nx, info.ny, info.nz) }
    : await readCubicVolumeField(device, solver);
  const final = includeFinalFieldStats && steps !== oracleSteps ? await readCubicVolumeField(device, solver) : matched;
  const finalSolver = solver as GPUSolverInstance & { velocityTexture?: GPUTexture; powerFaceTransferControl?: GPUBuffer;
    powerFaceControl?: GPUBuffer; powerDescriptorControl?: GPUBuffer; powerTopologyControl?: GPUBuffer;
    powerDescriptorRows?: GPUBuffer; powerTopologyMetrics?: GPUBuffer; powerLeafHeaders?: GPUBuffer; powerOwnerArena?: GPUBuffer;
    mgpcgControl?: GPUBuffer };
  const velocityTexture = finalSolver.velocityTexture;
  const finalSpacing = {
    x: scene.container.width_m / info.nx,
    y: scene.container.height_m / info.ny,
    z: scene.container.depth_m / info.nz
  };
  const velocitySummary = velocityTexture && final && !finalSolver.adaptiveFaceVelocitySource
    ? (info.gridKind === "restricted-tall-cell"
      ? await readTallVelocityTexture3D(device, velocityTexture, info.nx, info.storedNy, info.nz, info.ny, await readFloatTexture2D(device, solver.columnBaseTexture, info.nx, info.nz), final.field, finalSpacing, scene.numerics.maxDt_s)
      : await readVelocityTexture3D(device, velocityTexture, info.nx, info.ny, info.nz, final.field, finalSpacing, scene.numerics.maxDt_s))
    : undefined;
  let velocityParityField: Float32Array | undefined;
  let compactVelocityRaster: GPUSmokeResult["compactVelocityRaster"];
  if (scenarioId === "dam-break-ui" && final) {
    if (method.id === "tall-cell" && velocityTexture && info.gridKind === "restricted-tall-cell") {
      const bases = await readFloatTexture2D(device, solver.columnBaseTexture, info.nx, info.nz);
      velocityParityField = await readTallVelocityField3D(
        device, velocityTexture, info.nx, info.storedNy, info.nz, info.ny, bases,
      );
    } else if (method.id === "octree") {
      const compact = await readCompactOctreeVelocityField3D(device, solver, [info.nx, info.ny, info.nz]);
      if (compact) {
        const { field, ...evidence } = compact;
        velocityParityField = field;
        compactVelocityRaster = evidence;
      }
    }
  }
  const hybridPresentationStats = sparseStatsRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies)
    : undefined;
  const finalGlobalFineGeneration = globalFineGenerationTransitionRequested && method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  const finalGlobalFineRaster = globalFineGenerationTransitionRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies, true) : undefined;
  const sparseVoxelStats = sparseStatsRequested && sparseSource
    ? await readSparseVoxelStats(device, sparseSource, seedBrickBounds)
    : undefined;
  const faceMirrorSource = (solver as GPUSolverInstance).adaptiveFaceMirrorSource;
  const octreeFaceMirrorDiagnostics = faceMirrorSource
    ? await readOctreeFaceMirrorDiagnostics(device, faceMirrorSource, info.lastDt_s ?? scene.numerics.maxDt_s)
    : undefined;
  const powerTransferBytes = finalSolver.powerFaceTransferControl
    ? await readBufferBinding(device, { buffer: finalSolver.powerFaceTransferControl }, 32)
    : undefined;
  const powerTransferWords = powerTransferBytes
    ? new Uint32Array(powerTransferBytes.buffer, powerTransferBytes.byteOffset, 8)
    : undefined;
  const octreePowerFaceTransferDiagnostics: OctreePowerFaceTransferDiagnostics | undefined = powerTransferWords ? {
    previousFaceCount: powerTransferWords[0], valid: powerTransferWords[1] === 0x8000_0000,
    flags: powerTransferWords[2], generation: powerTransferWords[3],
    exactFaces: powerTransferWords[4], fallbackFaces: powerTransferWords[5], sourceFlags: powerTransferWords[7],
  } : undefined;
  const powerFaceBytes = finalSolver.powerFaceControl
    ? await readBufferBinding(device, { buffer: finalSolver.powerFaceControl }, 64) : undefined;
  const powerFaceWords = powerFaceBytes ? new Uint32Array(powerFaceBytes.buffer, powerFaceBytes.byteOffset, 16) : undefined;
  const powerFaceNeighborFailure = powerFaceWords ? (powerFaceWords[3] & (16 | 64)) !== 0 : false;
  const octreePowerFaceDiagnostics: OctreePowerFaceDiagnostics | undefined = powerFaceWords ? {
    rowCount: powerFaceWords[0], faceCount: powerFaceWords[1], incidenceCount: powerFaceWords[2], flags: powerFaceWords[3],
    firstInvalid: powerFaceWords[4], invalidCount: powerFaceWords[5], boundaryCount: powerFaceWords[6], generation: powerFaceWords[7],
    valid: powerFaceWords[8] === 0x8000_0000, lookupMissCount: powerFaceWords[9], maximumObservedProbe: powerFaceWords[10],
    worldBoundaryCount: powerFaceWords[11],
    firstInvalidSlot: powerFaceWords[12],
    firstInvalidNeighbor: powerFaceNeighborFailure ? powerFaceWords[13] : 0xffff_ffff,
    firstInvalidDetail: powerFaceWords[14],
    firstInvalidRow: powerFaceWords[4], firstInvalidPad3: powerFaceWords[15],
  } : undefined;
  if (octreePowerFaceDiagnostics && !octreePowerFaceDiagnostics.valid && finalSolver.powerLeafHeaders
    && finalSolver.powerTopologyMetrics && octreePowerFaceDiagnostics.firstInvalid !== 0xffff_ffff) {
    octreePowerFaceDiagnostics.firstInvalidPair = [];
    for (const row of [octreePowerFaceDiagnostics.firstInvalid, octreePowerFaceDiagnostics.firstInvalidNeighbor]
      .filter((candidate) => candidate !== 0xffff_ffff)) {
      const header = await readBufferBinding(device, { buffer: finalSolver.powerLeafHeaders, offset: row * 48, size: 48 }, 48);
      const metric = await readBufferBinding(device, { buffer: finalSolver.powerTopologyMetrics, offset: row * 16, size: 16 }, 16);
      const hw = new Uint32Array(header.buffer, header.byteOffset, 12), mw = new Uint32Array(metric.buffer, metric.byteOffset, 4);
      const hf = new Float32Array(header.buffer, header.byteOffset, 12);
      octreePowerFaceDiagnostics.firstInvalidPair.push({ row, cell: hw[0], size: hw[3], topologyCode: mw[0], transformAndFlags: mw[1], gradient: [...hf.slice(8, 12)] });
    }
  }
  const descriptorControlBytes = finalSolver.powerDescriptorControl
    ? await readBufferBinding(device, { buffer: finalSolver.powerDescriptorControl }, 32) : undefined;
  const topologyControlBytes = finalSolver.powerTopologyControl
    ? await readBufferBinding(device, { buffer: finalSolver.powerTopologyControl }, 32) : undefined;
  const descriptorControlWords = descriptorControlBytes
    ? new Uint32Array(descriptorControlBytes.buffer, descriptorControlBytes.byteOffset, 8) : undefined;
  const topologyControlWords = topologyControlBytes
    ? new Uint32Array(topologyControlBytes.buffer, topologyControlBytes.byteOffset, 8) : undefined;
  let octreePowerTopologyDiagnostics: OctreePowerTopologyDiagnostics | undefined;
  if (descriptorControlWords && topologyControlWords) {
    const descriptorFirstInvalid = descriptorControlWords[3];
    const topologyFirstInvalid = topologyControlWords[1];
    const firstInvalid = Math.min(descriptorFirstInvalid, topologyFirstInvalid);
    let firstInvalidRow: OctreePowerTopologyDiagnostics["firstInvalidRow"];
    if (firstInvalid !== 0xffff_ffff && firstInvalid < descriptorControlWords[0]
      && finalSolver.powerDescriptorRows && finalSolver.powerTopologyMetrics && finalSolver.powerLeafHeaders) {
      const descriptorBytes = await readBufferBinding(device,
        { buffer: finalSolver.powerDescriptorRows, offset: firstInvalid * 4, size: 4 }, 4);
      const metricBytes = await readBufferBinding(device,
        { buffer: finalSolver.powerTopologyMetrics, offset: firstInvalid * 16, size: 16 }, 16);
      const headerBytes = await readBufferBinding(device,
        { buffer: finalSolver.powerLeafHeaders, offset: firstInvalid * 48, size: 48 }, 48);
      const metricWords = new Uint32Array(metricBytes.buffer, metricBytes.byteOffset, 4);
      const metricFloats = new Float32Array(metricBytes.buffer, metricBytes.byteOffset, 4);
      const headerWords = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, 12);
      let ownerNeighborhood: NonNullable<OctreePowerTopologyDiagnostics["firstInvalidRow"]>["ownerNeighborhood"];
      if (finalSolver.powerOwnerArena) {
        const arenaBytes = await readBufferBinding(device, { buffer: finalSolver.powerOwnerArena }, finalSolver.powerOwnerArena.size);
        const arena = new Uint32Array(arenaBytes.buffer, arenaBytes.byteOffset, arenaBytes.byteLength / 4);
        const dimensions = [info.nx, info.ny, info.nz] as const;
        const maximumLeaf = info.quadtreeMaximumFluidScale ?? 16;
        const canonical = (cell: [number, number, number]) => {
          let size = Math.min(maximumLeaf, 8);
          let origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
          while (size > 1 && origin.some((value, axis) => value + size > dimensions[axis])) {
            size /= 2; origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
          }
          return { origin, size, invalid: false };
        };
        const ownerAt = (cell: [number, number, number]) => {
          if (arena.length <= 15 || arena[15] !== 0x4f57_4e52) return { ...canonical(cell), invalid: true };
          const brickDimensions = dimensions.map((value) => Math.ceil(value / 8));
          const brick = cell.map((value) => Math.floor(value / 8));
          const logical = brick[0] + brick[1] * brickDimensions[0] + brick[2] * brickDimensions[0] * brickDimensions[1];
          const hashCapacity = (arena[5] - 16) / 2; let slot = Math.imul(logical, 0x9e37_79b1) >>> 0; slot %= hashCapacity;
          let encoded = 0;
          for (let probe = 0; probe < hashCapacity; probe += 1) {
            const observed = arena[16 + slot]; if (observed === logical + 1) { encoded = arena[16 + hashCapacity + slot]; break; }
            if (observed === 0) break; slot = slot + 1 === hashCapacity ? 0 : slot + 1;
          }
          if (encoded === 0) return canonical(cell);
          if (encoded === 0xffff_ffff || encoded > arena[3]) return { ...canonical(cell), invalid: true };
          const local = (cell[0] & 7) + 8 * ((cell[1] & 7) + 8 * (cell[2] & 7));
          const word = arena[arena[6] + (encoded - 1) * 512 + local]; if (word === 0) return canonical(cell);
          if (word === 0x8000_0000) return { origin: cell, size: 1, invalid: false };
          const exponent = word & 7; const size = 1 << exponent;
          const origin = [((word >>> 3) & 511) << exponent, ((word >>> 12) & 511) << exponent,
            ((word >>> 21) & 511) << exponent] as [number, number, number];
          const invalid = (word & 0xc000_0000) !== 0 || exponent < 1 || exponent > 5
            || origin.some((value, axis) => cell[axis] < value || cell[axis] >= value + size);
          return { origin, size, invalid };
        };
        const origin = [headerWords[0] % info.nx, Math.floor(headerWords[0] / info.nx) % info.ny,
          Math.floor(headerWords[0] / (info.nx * info.ny))] as [number, number, number];
        const directions = [[0,0,0],[-1,0,0],[0,-1,0],[0,0,-1],[0,0,1],[0,1,0],[1,0,0],
          [-1,-1,0],[-1,0,-1],[-1,0,1],[-1,1,0],[0,-1,-1],[0,-1,1],[0,1,-1],[0,1,1],[1,-1,0],[1,0,-1],[1,0,1],[1,1,0]];
        ownerNeighborhood = directions.map((direction, bit) => {
          const probe = bit === 0 ? origin : direction.map((component, axis) => component < 0 ? origin[axis] - 1
            : component > 0 ? origin[axis] + headerWords[3] : origin[axis] + Math.floor(headerWords[3] / 2)) as [number, number, number];
          return { direction: bit - 1, probe, ...ownerAt(probe) };
        });
      }
      firstInvalidRow = { row: firstInvalid,
        descriptor: new Uint32Array(descriptorBytes.buffer, descriptorBytes.byteOffset, 1)[0],
        topologyCode: metricWords[0], transformAndFlags: metricWords[1], volume: metricFloats[2],
        reserved: metricWords[3], cell: headerWords[0], size: headerWords[3], ownerNeighborhood };
    }
    octreePowerTopologyDiagnostics = {
      descriptor: { rowCount: descriptorControlWords[0], validCount: descriptorControlWords[1],
        errorCount: descriptorControlWords[2], firstInvalid: descriptorControlWords[3], flags: descriptorControlWords[4],
        sameOrFinerCount: descriptorControlWords[5], sameOrCoarserCount: descriptorControlWords[6], generation: descriptorControlWords[7] },
      topology: { invalidCount: topologyControlWords[0], firstInvalid: topologyControlWords[1], flags: topologyControlWords[2],
        resolvedCount: topologyControlWords[3], version: topologyControlWords[4] }, firstInvalidRow,
    };
  }
  const mgpcgControlBytes = finalSolver.mgpcgControl
    ? await readBufferBinding(device, { buffer: finalSolver.mgpcgControl }, 64)
    : undefined;
  const octreeMGPCGDiagnostics = mgpcgControlBytes
    ? decodeOctreeMGPCGDiagnostics(new Uint32Array(mgpcgControlBytes.buffer, mgpcgControlBytes.byteOffset, 16))
    : undefined;
  const powerValidationScenario = scenarioId === "hydrostatic-power-two-level"
    || scenarioId === "hydrostatic-power-large-offset"
    || scenarioId === "minimal-power-dam-break";
  const powerProjectionDiagnostics = powerValidationScenario && method.id === "octree"
    ? await readPowerProjectionDiagnostics(device, solver, scene, octreePowerFaceDiagnostics)
    : undefined;
  // The compact face mirror predates the authoritative power projection and
  // can legitimately contain the transferred pre-projection zero field on
  // the first dam-break step.  For paper-path QA, publish motion telemetry
  // from the generalized power faces that actually drive Section 5.
  if (powerProjectionDiagnostics) {
    const powerSpeed_m_s = powerProjectionDiagnostics.maximumSpeed_m_s;
    const finestCellSize_m = scene.voxelDomain?.finestCellSize_m
      ?? Math.min(scene.container.width_m / info.nx, scene.container.height_m / info.ny,
        scene.container.depth_m / info.nz);
    const powerCfl = powerSpeed_m_s * powerProjectionDiagnostics.pressurePotential.dt_s
      / finestCellSize_m;
    info.maxSpeed_m_s = powerSpeed_m_s;
    info.maxComponentCfl = powerCfl;
    if (stabilityEnvelope) {
      stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(stabilityEnvelope.peakLiquidSpeed_m_s, powerSpeed_m_s);
      stabilityEnvelope.peakComponentCfl = Math.max(stabilityEnvelope.peakComponentCfl, powerCfl);
    }
  }
  await device.queue.onSubmittedWorkDone();
  const result: GPUSmokeResult = {
    method: resultMethod, info, grid: [info.nx, info.ny, info.nz], matchedField: matched.field,
    matchedSummary: matched.summary, compactFieldEvidence: matched.compactFieldEvidence,
    matchedTallCellActivity: matched.tallCellActivity,
    finalSummary: final?.summary, finalTallCellActivity: final?.tallCellActivity,
    finalTallVolumeGaps: final?.tallVolumeGaps, validationErrors,
    construction_ms, runtime_ms: performance.now() - runStarted, simulationWall_ms, steps,
    powerGenerationAuditedSteps, powerTransitionWitness, velocitySummary,
    velocityParityField, velocityParityVolume: final?.field, compactVelocityRaster,
    initialFluidBrickStats, sparseVoxelStats, hybridPresentationStats,
    initialGlobalFineGeneration, initialGlobalFineRaster, finalGlobalFineGeneration, finalGlobalFineRaster,
    octreeFaceMirrorDiagnostics,
    octreePowerFaceTransferDiagnostics, octreePowerFaceDiagnostics, octreePowerTopologyDiagnostics,
    octreeMGPCGDiagnostics, powerProjectionDiagnostics,
    stabilityEnvelope, energyTrace, checkpoints
  };
  reportResult(scenarioId, result);
  solver.destroy(); device.destroy();
  return result;
}

function cpuField(solver: EulerianFluidSolver) {
  const field = new Float32Array(solver.fluid.length);
  for (let index = 0; index < field.length; index += 1) field[index] = solver.fluid[index] ? 1 : 0;
  return field;
}

function runMatchedCPUOracle(scenarioId: SmokeScenarioId, grid: [number, number, number], oracleSteps: number) {
  const cellCount = grid[0] * grid[1] * grid[2];
  if (cpuMaximumCells > 0 && cellCount > cpuMaximumCells) {
    console.log(JSON.stringify({ scenario: scenarioId, method: "cpu-reference", phase: "oracle-skipped", cubicGrid: grid, cellCount, reason: `exact grid exceeds FLUID_CPU_MAX_CELLS=${cpuMaximumCells}; set 0 for unlimited` }));
    return undefined;
  }
  const scene = applySceneOverrides(createSmokeScenario(scenarioId).scene), started = performance.now();
  const solver = new EulerianFluidSolver(scene, { dimensions: { nx: grid[0], ny: grid[1], nz: grid[2] }, markerSamplesPerAxis: cpuMarkerSamplesPerAxis });
  for (let step = 0; step < oracleSteps; step += 1) solver.step(scene.numerics.maxDt_s);
  const field = cpuField(solver), summary = summarizeScalarField(field, ...grid);
  console.log(JSON.stringify({
    scenario: scenarioId, method: "cpu-reference", phase: "oracle", precision: "binary64", cubicGrid: grid,
    markerSamplesPerAxis: cpuMarkerSamplesPerAxis, oracleSteps, runtime_ms: Math.round(performance.now() - started),
    diagnostics: solver.diagnostics, fieldStats: summary
  }));
  return { field, summary, diagnostics: solver.diagnostics };
}

function damBreakVelocityParityMetrics(results: GPUSmokeResult[]): VelocityParityMetrics | undefined {
  const octree = results.find((result) => result.method === "octree");
  const tall = results.find((result) => result.method === "tall-cell");
  if (!octree || !tall || !octree.grid.every((value, axis) => value === tall.grid[axis])
    || !octree.velocityParityField || !tall.velocityParityField
    || !octree.velocityParityVolume || !tall.velocityParityVolume) return undefined;
  return compareVelocityFields(
    octree.velocityParityField, tall.velocityParityField,
    octree.velocityParityVolume, tall.velocityParityVolume,
  );
}

function invariantFailures(scenarioId: SmokeScenarioId, results: GPUSmokeResult[]) {
  const failures: string[] = [];
  const fail = (condition: boolean, message: string) => { if (!condition) failures.push(`${scenarioId}: ${message}`); };
  const coarseOnlyOctreeRequested = octreeGlobalFineFactorOverride === "off";
  for (const result of results) {
    if (exactStepCount !== undefined) {
      const expectedTime_s = exactStepCount * maxDtOverride!;
      fail(result.steps === exactStepCount,
        `${result.method} accepted ${result.steps} outer steps; expected exactly ${exactStepCount}`);
      fail(result.info.encodedSteps === exactStepCount,
        `${result.method} encoded ${result.info.encodedSteps ?? "unknown"} steps; expected exactly ${exactStepCount}`);
      fail(Math.abs((result.info.submittedTime_s ?? -Infinity) - expectedTime_s) <= 1e-9,
        `${result.method} submitted time ${result.info.submittedTime_s} differs from exact checkpoint ${expectedTime_s}`);
      fail(Math.abs((result.info.completedTime_s ?? -Infinity) - expectedTime_s) <= 1e-9,
        `${result.method} completed time ${result.info.completedTime_s} differs from fenced checkpoint ${expectedTime_s}`);
    }
    const exhaustivePowerScenario = scenarioId === "hydrostatic-power-two-level"
      || scenarioId === "minimal-power-dam-break";
    if (exhaustivePowerScenario && powerGenerationAuditRequested && result.method === "octree") {
      if (exactStepCount === undefined) {
        fail(result.steps >= 50, `octree completed only ${result.steps} steps; exhaustive power validation requires at least 50`);
      }
      fail(result.powerGenerationAuditedSteps === result.steps,
        `octree audited ${result.powerGenerationAuditedSteps} of ${result.steps} power generations`);
      const envelope = result.stabilityEnvelope;
      fail((envelope?.sampledSteps ?? 0) === result.steps,
        `octree sampled pressure and volume on ${envelope?.sampledSteps ?? 0} of ${result.steps} steps`);
      if (!coarseOnlyOctreeRequested) {
        fail((envelope?.invalidVolumeSampleCount ?? Infinity) === 0,
          `octree produced ${envelope?.invalidVolumeSampleCount ?? "unknown"} invalid per-step volume fields`);
      }
      fail((envelope?.nonFiniteVelocityCount ?? Infinity) === 0,
        `octree encountered ${envelope?.nonFiniteVelocityCount ?? "unknown"} non-finite velocities`);
      fail((envelope?.maximumPressureRelativeResidual ?? Infinity) <= 1e-4,
        `octree per-step pressure residual peaked at ${envelope?.maximumPressureRelativeResidual}`);
      if (!coarseOnlyOctreeRequested) {
        const maximumVolumeDrift = scenarioId === "hydrostatic-power-two-level" ? 1e-4 : 0.01;
        fail((envelope?.maximumExactVolumeDrift ?? Infinity) <= maximumVolumeDrift,
          `octree per-step volume drift peaked at ${envelope?.maximumExactVolumeDrift}; limit ${maximumVolumeDrift}`);
      }
    }
    if (requireSpatialField) {
      fail(result.matchedField.length === result.grid[0] * result.grid[1] * result.grid[2],
        `${result.method} did not publish a full spatial QA field`);
      fail(result.matchedSummary.cellSum > 1 && result.matchedSummary.cellSum < result.matchedField.length - 1,
        `${result.method} spatial QA field is not a meaningful partially wet domain (sum ${result.matchedSummary.cellSum})`);
      if (result.method === "octree") {
        fail(result.compactFieldEvidence !== undefined
          && compactOctreeFieldEvidenceIsAcceptable(result.compactFieldEvidence),
        `octree spatial QA field lacks current mixed fine/coarse publication evidence: ${JSON.stringify(result.compactFieldEvidence)}`);
      }
    }
    if (minimumPeakSpeed_m_s !== undefined) {
      const observedMotionSpeed_m_s = result.stabilityEnvelope?.peakLiquidSpeed_m_s
        ?? result.powerProjectionDiagnostics?.maximumSpeed_m_s ?? result.info.maxSpeed_m_s ?? 0;
      fail(observedMotionSpeed_m_s >= minimumPeakSpeed_m_s,
        `${result.method} observed motion speed ${observedMotionSpeed_m_s} m/s is below ${minimumPeakSpeed_m_s} m/s`);
    }
    fail(result.validationErrors.length === 0, `${result.method} WebGPU validation errors: ${result.validationErrors.join("; ")}`);
    fail((result.info.nonFiniteCount ?? 0) === 0, `${result.method} reported ${result.info.nonFiniteCount} non-finite values`);
    fail(Number.isFinite(result.info.maxSpeed_m_s ?? NaN), `${result.method} max speed is not finite`);
    const compactFaceTransport = result.method === "octree"
      && (result.info.adaptiveFaceTransportedCount ?? 0) > 0;
    if (compactFaceTransport) {
      const faces = result.octreeFaceMirrorDiagnostics;
      fail(faces !== undefined, "octree compact transport did not publish face diagnostics");
      fail((faces?.faceCount ?? 0) > 0, "octree compact transport published zero faces");
      fail((faces?.comparedRows ?? 0) > 0, "octree compact transport assembled zero adaptive pressure rows");
      if ((result.info.adaptiveSurfacePageCapacity ?? 0) > 0) {
        fail((faces?.projectionComparedFaces ?? 0) === 0,
          "paged octree authority must disable obsolete dense projection parity");
      } else fail((faces?.projectionComparedFaces ?? 0) > 0, "octree compact transport projected zero adaptive faces");
      fail(faces?.overflow === false, "octree compact transport overflowed its face arena");
      fail((result.info.adaptiveFaceTransportedCount ?? 0) > 0, "octree compact transport processed zero faces");
      fail(result.info.adaptiveFaceTransportedCount === faces?.faceCount,
        `octree compact transport processed ${result.info.adaptiveFaceTransportedCount} of ${faces?.faceCount} faces`);
      fail(Number.isFinite(result.info.maxComponentCfl ?? NaN)
        && ((scenarioId === "hydrostatic-power-two-level" || scenarioId === "hydrostatic-power-large-offset")
          || (result.info.maxComponentCfl ?? 0) > 0),
        `octree compact transport reported invalid or zero CFL ${result.info.maxComponentCfl}`);
      if (result.info.adaptiveSurfacePageCapacity !== undefined) {
        fail(result.info.adaptiveSurfaceOverflow === false,
          `octree adaptive surface overflowed code ${result.info.adaptiveSurfaceOverflowCode} (${result.info.adaptiveSurfaceCandidatePages} candidates for ${result.info.adaptiveSurfacePageCapacity} pages)`);
        fail((result.info.adaptiveSurfaceActivePages ?? 0) > 0, "octree adaptive surface published zero active pages");
      }
    }
    if (scenarioId === "hose-tank" || scenarioId === "sphere-jet") {
      fail((result.info.volumeCellSum ?? -Infinity) >= (result.info.initialVolumeCellSum ?? 0) * 0.99, `${result.method} inflow scene lost more than 1% of its initial represented volume`);
      // A working inflow moves fluid; a projection that treats injected liquid
      // as air freezes the whole field at numerical zero while volume grows.
      // Before ~0.3 s the stream is still sub-threshold and max liquid speed
      // measures ambient equilibrium noise, so the gate only applies once the
      // jet has had time to establish.
      if ((result.info.simulatedTime_s ?? 0) >= 0.3) fail((result.info.maxSpeed_m_s ?? 0) >= 0.01, `${result.method} inflow scene is frozen: max speed ${result.info.maxSpeed_m_s} m/s`);
    }
    else {
      // The independently transported level set has a larger release/slosh
      // excursion than conservative VOF. The dedicated 10 s settling gate
      // still requires the tall path to finish within 1%.
      const representedVolumeLimit = scenarioId === "dam-break-ui" && result.method === "tall-cell" ? 0.02 : 0.01;
      fail(Math.abs(result.info.representedVolumeDrift ?? Infinity) <= representedVolumeLimit,
        `${result.method} represented-volume drift ${result.info.representedVolumeDrift} exceeds ${representedVolumeLimit * 100}%`);
    }
    fail(result.matchedSummary.minimum >= -0.01, `${result.method} volume minimum ${result.matchedSummary.minimum} is below -0.01`);
    // Stored density above one is deliberate temporary mass on both paths
    // (sharpening deposits, tall remap residuals) and drains through the
    // correction divergence; the bound only catches runaway accumulation.
    const maximumStoredDensity = 1.5;
    fail(result.matchedSummary.maximum <= maximumStoredDensity, `${result.method} volume maximum ${result.matchedSummary.maximum} exceeds ${maximumStoredDensity}`);
    if (result.finalSummary) {
      fail(result.finalSummary.minimum >= -0.01, `${result.method} final volume minimum ${result.finalSummary.minimum} is below -0.01`);
      fail(result.finalSummary.maximum <= maximumStoredDensity, `${result.method} final volume maximum ${result.finalSummary.maximum} exceeds ${maximumStoredDensity}`);
    }
    if (settlingGateRequested) {
      const energy = energyTraceSummary(result.energyTrace);
      fail(energy !== undefined, `${result.method} did not produce a mechanical-energy trace`);
      if (energy) {
        // The signed-distance occupancy proxy swings during the violent
        // release as interface area changes. Settling correctness concerns
        // the final sampled volume; the maximum remains reported for diagnosis.
        fail(energy.finalSampledExactVolumeDrift <= 0.01,
          `${result.method} final sampled exact-volume drift reached ${energy.finalSampledExactVolumeDrift}`);
        fail(energy.normalizedNetProjectionEnergyDelta <= 0.01,
          `${result.method} pressure projections added ${energy.normalizedNetProjectionEnergyDelta} of the initial mechanical energy`);
        fail(energy.normalizedLateMechanicalEnergySlopePerSecond <= settlingNormalizedSlopeEpsilon,
          `${result.method} late mechanical-energy slope ${energy.lateMechanicalEnergySlopePerSecond}/s (${energy.normalizedLateMechanicalEnergySlopePerSecond} of initial energy/s) exceeds the ${settlingNormalizedSlopeEpsilon} normalized proxy-noise allowance`);
        fail(energy.lateToMiddleKineticEnvelopeRatio <= 1,
          `${result.method} late kinetic-energy envelope is ${energy.lateToMiddleKineticEnvelopeRatio} times its middle-window envelope`);
        if (scenarioId === "dam-break-ui") {
          fail(energy.driftSignChanges <= 3,
            `${result.method} late volume drift changed direction ${energy.driftSignChanges} times after median smoothing`);
          fail(energy.latePeakToPeakDrift <= 0.005,
            `${result.method} late peak-to-peak volume drift ${energy.latePeakToPeakDrift} exceeds 0.5%`);
        }
      }
    }
  }
  if (results.length > 1) {
    const [first, ...rest] = results;
    for (const result of rest) fail(result.grid.every((value, axis) => value === first.grid[axis]), `${result.method} cubic grid ${result.grid} differs from ${first.method} ${first.grid}`);
  }
  const tall = results.find((result) => result.method === "tall-cell");
  if (tall?.finalTallVolumeGaps?.unexcusedDeltaViolations !== undefined) fail(tall.finalTallVolumeGaps.unexcusedDeltaViolations === 0, `tall-cell has ${tall.finalTallVolumeGaps.unexcusedDeltaViolations} adjacent base deltas beyond ${tall.info.maximumNeighborDelta}`);
  else if (tall?.finalTallCellActivity?.maximumAdjacentDelta !== undefined) fail(tall.finalTallCellActivity.maximumAdjacentDelta <= tall.info.maximumNeighborDelta, `tall-cell adjacent base delta ${tall.finalTallCellActivity.maximumAdjacentDelta} exceeds ${tall.info.maximumNeighborDelta}`);
  if (scenarioId === "dam-break-boxes" && tall) {
    fail(tall.info.gridKind === "restricted-tall-cell", `tall-cell dam break used ${tall.info.gridKind} instead of the restricted backend`);
    fail((tall.finalTallCellActivity?.tallColumns ?? 0) > 0, "tall-cell dam break has no allocated tall columns");
    fail((tall.finalTallCellActivity?.ordinaryColumns ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallCellActivity?.ordinaryColumns ?? "unknown"} incomplete base-zero columns`);
    fail((tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? "unknown"} dry tall columns underneath wet regular cells`);
    const tallReference = referenceVolumeCells(tall.info);
    const exactVolumeDrift = tall.finalSummary
      ? (tall.finalSummary.cellSum - tallReference) / Math.max(1, Math.abs(tallReference))
      : tall.info.representedVolumeDrift ?? Infinity;
    fail(Math.abs(exactVolumeDrift) <= 0.01, `tall-cell dam break exact volume drift ${exactVolumeDrift} exceeds 1%`);
  }
  if (scenarioId === "dam-break-ui" && tall) {
    // Gates for the tall-cell dam break with genuinely active tall cells
    // (test:webgpu:dam-tall-active). Thresholds calibrated 2026-07-15 against
    // the post-fix baseline (docs/TALL_CELL_STABILITY.md): KE ratio peaks at
    // 1.67 in the release transient (the endpoint-wetness defect produced >8
    // and non-finite blow-up), peak CFL 10.9 in splash transients (uniform
    // 1.7 — a known remaining gap, gated as a regression backstop).
    fail(tall.info.gridKind === "restricted-tall-cell", `tall-cell dam break used ${tall.info.gridKind} instead of the restricted backend`);
    fail((tall.finalTallCellActivity?.tallColumns ?? 0) > 0, "tall-cell dam break has no allocated tall columns");
    const envelope = tall.stabilityEnvelope;
    if (envelope) {
      fail(envelope.nonFiniteVelocityCount === 0, `tall-cell dam break produced ${envelope.nonFiniteVelocityCount} non-finite staged velocities`);
      fail(envelope.maximumProjectionEnergyRatio <= 2.0, `tall-cell projection amplified kinetic energy by ${envelope.maximumProjectionEnergyRatio}`);
      // Splash chaos gives large run-to-run variance (10.9-21.7 observed on
      // identical configs); the backstop only needs to catch the divergent
      // regime, which reached 1e29 before the 2026-07-15 fixes.
      fail(envelope.peakComponentCfl <= 32, `tall-cell dam break peak CFL ${envelope.peakComponentCfl} exceeds the 32-cell backstop`);
      // The level-set occupancy reconstruction can swing during the release
      // transient as interface area explodes; the general final-volume gate
      // above remains 1%. Keep a broad transient backstop for catastrophic
      // gain/loss without conflating this proxy excursion with settled drift.
      fail(envelope.maximumExactVolumeDrift <= 0.15, `tall-cell dam break transient exact-volume proxy drift peaked at ${envelope.maximumExactVolumeDrift}`);
    }
    if ((tall.info.simulatedTime_s ?? 0) >= 1.5) fail((tall.info.front_m ?? -Infinity) > 0.3, `tall-cell dam break front did not cross the tank: ${tall.info.front_m} m`);
    fail((tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? Infinity) === 0, `tall-cell dam break has ${tall.finalTallVolumeGaps?.dryTallWithWetRegularAbove ?? "unknown"} dry tall columns underneath wet regular cells`);
    const uniformPair = results.find((result) => result.method === "uniform");
    if (uniformPair && tall.checkpoints.length > 0 && uniformPair.checkpoints.length > 0) {
      const pairCount = Math.min(tall.checkpoints.length, uniformPair.checkpoints.length);
      let minimumIoU = 1, finalIoU = 1;
      for (let index = 0; index < pairCount; index += 1) {
        const iou = compareScalarFields(tall.checkpoints[index].field, uniformPair.checkpoints[index].field, ...tall.grid).wetIntersectionOverUnion;
        minimumIoU = Math.min(minimumIoU, iou); finalIoU = iou;
      }
      // The minimal-tall control (bases pinned at 2) bottoms out at 0.37 IoU
      // against uniform through the chaotic slosh; deep tall cells must stay
      // within that envelope rather than match uniform cell-for-cell.
      fail(minimumIoU >= 0.35, `tall-cell wet-IoU vs uniform fell to ${minimumIoU} (minimal-tall control floor is 0.37)`);
      fail(finalIoU >= 0.4, `tall-cell final wet-IoU vs uniform is ${finalIoU}`);
    }
    // The level set should retain a comparably narrow transition even though
    // it deliberately uses semi-Lagrangian advection and periodic reinit.
    if (uniformPair?.finalSummary && tall.finalSummary) {
      const mixedFraction = (summary: ScalarFieldSummary) => summary.wetCells > 0 ? summary.mixedCells / summary.wetCells : 0;
      fail(mixedFraction(tall.finalSummary) <= mixedFraction(uniformPair.finalSummary) * 2 + 0.05,
        `tall-cell mixed-cell fraction ${mixedFraction(tall.finalSummary)} exceeds twice uniform's ${mixedFraction(uniformPair.finalSummary)}`);
    }
  }
  if ((scenarioId === "settled-tank" || scenarioId === "deep-water") && tall) {
    fail((tall.info.stabilityFlags?.length ?? 0) === 0, `tall-cell equilibrium flags: ${tall.info.stabilityFlags?.join(", ")}`);
    const tallReference = referenceVolumeCells(tall.info);
    const exactVolumeDrift = tall.finalSummary
      ? (tall.finalSummary.cellSum - tallReference) / Math.max(1, Math.abs(tallReference))
      : tall.info.representedVolumeDrift ?? Infinity;
    fail(Math.abs(exactVolumeDrift) <= 0.01, `tall-cell equilibrium exact volume drift ${exactVolumeDrift} exceeds 1%`);
    fail((tall.finalSummary?.componentCount ?? 1) === 1, `tall-cell equilibrium split into ${tall.finalSummary?.componentCount} components`);
  }
  if (scenarioId === "deep-water" && tall) fail((tall.info.compressionRatio ?? 1) < 0.5, `tall-cell compression ratio ${tall.info.compressionRatio} is not below 0.5`);
  const quadtree = results.find((result) => result.method === "quadtree-tall-cell");
  if (quadtree) {
    fail(quadtree.info.gridKind === "quadtree-tall-cell", `quadtree method reported ${quadtree.info.gridKind}`);
    fail((quadtree.info.quadtreeMaximumNeighborRatio ?? Infinity) <= 2, `quadtree neighbor ratio ${quadtree.info.quadtreeMaximumNeighborRatio} exceeds 2:1`);
    fail((quadtree.info.quadtreeLeafCount ?? 0) > 0, "quadtree has no leaves");
    fail((quadtree.info.quadtreeGhostFaceCount ?? 0) > 0, "quadtree has no corrected inner ghost faces");
    const residualAccepted = (quadtree.info.pressureRelativeResidual ?? Infinity) <= 1e-4 || (quadtree.info.pressureResidual ?? Infinity) <= 1e-5;
    fail(residualAccepted, `quadtree PCG residual relative=${quadtree.info.pressureRelativeResidual} rms=${quadtree.info.pressureResidual} exceeds the relative target and f32 absolute floor`);
    // Level-set/VOF disagreement is diagnostic: the paper-aligned solver does
    // not optimize it during healthy operation. Represented phi volume and
    // geometric parity below are the acceptance signals.
    fail((quadtree.info.quadtreeMaximumFluidScale ?? Infinity) <= maximumFluidScale, `quadtree free-surface scale ${quadtree.info.quadtreeMaximumFluidScale} escaped the ${maximumFluidScale} ghost-fluid ceiling`);
    if (scenarioId === "settled-tank" || scenarioId === "deep-water") {
      const exactVolumeDrift = quadtree.finalSummary
        ? (quadtree.finalSummary.cellSum - (quadtree.info.initialVolumeCellSum ?? 0)) / Math.max(1, Math.abs(quadtree.info.initialVolumeCellSum ?? 0))
        : quadtree.info.representedVolumeDrift ?? Infinity;
      fail(Math.abs(exactVolumeDrift) <= 1e-3, `quadtree equilibrium exact volume drift ${exactVolumeDrift} exceeds 0.1%`);
      fail((quadtree.finalSummary?.componentCount ?? 1) === 1, `quadtree equilibrium split into ${quadtree.finalSummary?.componentCount} components`);
      fail((quadtree.velocitySummary?.liquidMaximum ?? Infinity) <= 0.05, `quadtree equilibrium liquid velocity ${quadtree.velocitySummary?.liquidMaximum} m/s exceeds 0.05 m/s`);
    }
    if (scenarioId === "deep-water") fail((quadtree.info.compressionRatio ?? 1) < 0.5, `quadtree deep-water compression ratio ${quadtree.info.compressionRatio} is not below 0.5`);
    if (scenarioId === "dam-break-ui") {
      const envelope = quadtree.stabilityEnvelope;
      fail((quadtree.info.simulatedTime_s ?? 0) >= 0.2 - 1e-9, `quadtree dam-break regression reached only ${quadtree.info.simulatedTime_s} s`);
      fail(quadtree.info.quadtreeRebuildCadenceSteps === 1, `quadtree dam-break rebuild cadence was ${quadtree.info.quadtreeRebuildCadenceSteps}, not Algorithm 1's every-step cadence`);
      const staleWindow = quadtree.info.quadtreeTopologyStaleLimit ?? 2;
      // Stale window 0 is the fully GPU-resident inline path: every step
      // regenerates the topology (Algorithm 1), minus a short asynchronous
      // warmup before the first resident pack exists.
      const expectedRebuilds = staleWindow === 0 ? Math.ceil(0.9 * quadtree.steps) : Math.floor((quadtree.steps - 1) / Math.max(1, staleWindow + 1));
      fail((quadtree.info.quadtreeRebuildCompletedCount ?? 0) >= expectedRebuilds, `quadtree completed ${quadtree.info.quadtreeRebuildCompletedCount} rebuilds; stale-limit ${staleWindow} requires at least ${expectedRebuilds}`);
      fail((quadtree.info.quadtreeRebuildBlockedFrames ?? Infinity) === 0, `quadtree rebuild blocked ${(quadtree.info.quadtreeRebuildBlockedFrames ?? Infinity)} frame attempts`);
      const wallPerStep_ms = quadtree.simulationWall_ms / Math.max(1, quadtree.steps), gpuPerStep_ms = quadtree.info.gpuTimings?.total_ms ?? 0;
      fail(gpuPerStep_ms > 0 && wallPerStep_ms <= 2 * gpuPerStep_ms, `quadtree wall ${wallPerStep_ms.toFixed(2)} ms/step exceeds 2x GPU ${gpuPerStep_ms.toFixed(2)} ms/step`);
      fail((envelope?.sampledSteps ?? 0) === quadtree.steps, `quadtree dam-break sampled ${envelope?.sampledSteps} of ${quadtree.steps} steps`);
      fail((envelope?.nonFiniteVelocityCount ?? Infinity) === 0, `quadtree dam-break encountered ${envelope?.nonFiniteVelocityCount} non-finite staged velocities`);
      fail((envelope?.peakLiquidSpeed_m_s ?? Infinity) <= 5, `quadtree dam-break peak liquid speed ${envelope?.peakLiquidSpeed_m_s} m/s exceeds 5 m/s`);
      fail((envelope?.peakComponentCfl ?? Infinity) <= 1, `quadtree dam-break peak CFL ${envelope?.peakComponentCfl} exceeds one cell`);
      fail((envelope?.maximumProjectionEnergyRatio ?? Infinity) <= 1.1, `quadtree pressure projection amplified kinetic energy by ${envelope?.maximumProjectionEnergyRatio}`);
      // The 1e-4 relative-residual limit is this regression's float32 QA
      // policy. A topology transition is not allowed to weaken it.
      fail((envelope?.maximumPressureRelativeResidual ?? Infinity) <= 1e-4, `quadtree dam-break pressure residual peaked at ${envelope?.maximumPressureRelativeResidual}`);
      fail((envelope?.maximumExactVolumeDrift ?? Infinity) <= 0.02, `quadtree dam-break level-set volume drift peaked at ${envelope?.maximumExactVolumeDrift}`);
      fail((quadtree.info.compressionRatio ?? Infinity) <= 0.25, `quadtree dam-break compression ratio ${quadtree.info.compressionRatio} exceeds the 0.25 adaptivity budget`);
      fail((envelope?.minimumDominantComponentFraction ?? -Infinity) >= 0.995, `quadtree dam-break dominant component fell to ${envelope?.minimumDominantComponentFraction}`);
      fail((quadtree.finalSummary?.componentCount ?? Infinity) <= 10, `quadtree dam-break ended with ${quadtree.finalSummary?.componentCount} disconnected level-set components`);
      fail((quadtree.info.front_m ?? -Infinity) > -0.005, `quadtree dam-break front did not progress: ${quadtree.info.front_m} m`);
      const uniform = results.find((result) => result.method === "uniform"), uniformPeak = uniform?.stabilityEnvelope?.peakKineticEnergyProxy ?? 0;
      if (uniform && uniform.checkpoints.length > 0 && quadtree.checkpoints.length > 0) {
        const pairCount = Math.min(quadtree.checkpoints.length, uniform.checkpoints.length);
        for (let index = 0; index < pairCount; index += 1) {
          const iou = compareScalarFields(quadtree.checkpoints[index].field, uniform.checkpoints[index].field, ...quadtree.grid).wetIntersectionOverUnion;
          console.log(JSON.stringify({ scenario: scenarioId, method: "quadtree-tall-cell", phase: "iou-vs-uniform", time_s: quadtree.checkpoints[index].time_s, wetIntersectionOverUnion: iou }));
        }
      }
      if ((quadtree.info.simulatedTime_s ?? 0) >= 0.5) fail((envelope?.peakKineticEnergyProxy ?? 0) >= 0.40, `quadtree peak kinetic-energy proxy ${envelope?.peakKineticEnergyProxy} is below 0.40`);
      if (uniformPeak > 1e-9) fail((envelope?.peakKineticEnergyProxy ?? 0) / uniformPeak >= 0.8, `quadtree/uniform peak kinetic-energy ratio ${(envelope?.peakKineticEnergyProxy ?? 0) / uniformPeak} is below 0.8`);
      if (tall && quadtree.grid.every((value, axis) => value === tall.grid[axis])) {
        const comparison = compareScalarFields(quadtree.finalSummary ? quadtree.checkpoints.at(-1)?.field ?? quadtree.matchedField : quadtree.matchedField, tall.finalSummary ? tall.checkpoints.at(-1)?.field ?? tall.matchedField : tall.matchedField, ...quadtree.grid);
        fail(comparison.wetIntersectionOverUnion >= 0.60, `quadtree dam-break wet-IoU ${comparison.wetIntersectionOverUnion} is below the 0.60 tall-cell parity floor`);
        fail(comparison.centroidDistanceCells === null || comparison.centroidDistanceCells <= 6, `quadtree dam-break centroid differs from tall-cell by ${comparison.centroidDistanceCells} cells`);
        for (const checkpoint of quadtree.checkpoints.filter(({ time_s }) => time_s >= 1 - 1e-6)) {
          const reference = tall.checkpoints.find(({ time_s }) => Math.abs(time_s - checkpoint.time_s) <= 0.01);
          if (!reference) continue;
          const checkpointComparison = compareScalarFields(checkpoint.field, reference.field, ...quadtree.grid);
          fail(checkpointComparison.wetIntersectionOverUnion >= 0.60, `quadtree dam-break wet-IoU ${checkpointComparison.wetIntersectionOverUnion} at t=${checkpoint.time_s.toFixed(2)} s is below 0.60`);
        }
      }
    }
  }
  const octree = results.find((result) => result.method === "octree");
  if (octree) {
    fail(octree.info.gridKind === "octree", `octree method reported ${octree.info.gridKind}`);
    fail((octree.info.quadtreeMaximumNeighborRatio ?? Infinity) <= 2, `octree neighbor ratio ${octree.info.quadtreeMaximumNeighborRatio} exceeds 2:1`);
    fail((octree.info.quadtreeTopologyReadbackBytes ?? Infinity) === 0,
      `octree simulation topology performed ${octree.info.quadtreeTopologyReadbackBytes ?? "unknown"} CPU readback bytes`);
    {
      fail(octree.info.powerDiagramProjection === "authoritative",
        `octree requested authoritative power projection but reported ${octree.info.powerDiagramProjection ?? "unknown"}`);
      fail(octree.info.powerDiagramReady === true && octree.info.powerDiagramAuthoritative === true,
        `octree authoritative power projection did not publish: ${octree.info.powerDiagramFallbackReason ?? "no reason reported"}`);
      const powerFaces = octree.octreePowerFaceDiagnostics;
      fail(powerFaces?.valid === true && (powerFaces.faceCount ?? 0) > 0,
        `octree authoritative power faces did not publish a nonzero valid generation: ${JSON.stringify(powerFaces)}`);
      const powerTopology = octree.octreePowerTopologyDiagnostics;
      fail(powerTopology?.descriptor.errorCount === 0 && powerTopology.topology.invalidCount === 0,
        `octree authoritative power topology is invalid: ${JSON.stringify(powerTopology)}`);
      fail(octree.info.pressureSolver?.includes("Section 4.3 hybrid") === true,
        `octree authoritative power projection selected the wrong pressure solver: ${octree.info.pressureSolver ?? "unknown"}`);
      fail(octreeMGPCGDiagnosticsAreAcceptable(octree.octreeMGPCGDiagnostics),
        `octree authoritative Section 4.3 MGPCG missed the float32 QA relative-residual limit 1e-4: ${JSON.stringify(octree.octreeMGPCGDiagnostics)}`);
    }
    if (octreeGlobalFineFactorOverride === "4" || octreeGlobalFineFactorOverride === "8") {
      const expectedFactor = Number(octreeGlobalFineFactorOverride);
      fail(octree.info.globalFineLevelSetEnabled === true,
        "octree requested the global fine level set but did not expose a live GPU source");
      fail(octree.info.globalFineLevelSetFactor === expectedFactor,
        `octree global fine factor ${octree.info.globalFineLevelSetFactor ?? "unknown"} differs from requested ${expectedFactor}`);
    }
    if (coarseOnlyOctreeRequested) {
      fail(octree.info.globalFineLevelSetEnabled === false,
        "octree requested coarse-only mode but reported a live global fine level set");
      fail(octree.info.globalFineLevelSetAllocatedBytes === 0,
        `octree coarse-only mode allocated ${octree.info.globalFineLevelSetAllocatedBytes ?? "unknown"} global-fine bytes`);
    }
    const powerValidationScenario = scenarioId === "hydrostatic-power-two-level"
      || scenarioId === "hydrostatic-power-large-offset"
      || scenarioId === "minimal-power-dam-break";
    if (powerValidationScenario) {
      const power = octree.powerProjectionDiagnostics;
      const expectedGrid = scenarioId === "hydrostatic-power-large-offset"
        ? [32, 24, 16] : [16, 16, 16];
      fail(octree.grid.every((value, axis) => value === expectedGrid[axis]),
        `power validation grid ${octree.grid.join("x")} is not the intended ${expectedGrid.join("x")} domain`);
      fail(!!power, "power projection diagnostics were not published");
      const sizes = power?.leafSizeHistogram ?? {};
      const expectedLeafSizes = ["1", "2"];
      const dynamicTransitionWitness = scenarioId === "minimal-power-dam-break"
        ? octree.powerTransitionWitness : undefined;
      const requireTransitionWitness = scenarioId !== "minimal-power-dam-break" || powerGenerationAuditRequested;
      if (requireTransitionWitness) {
        fail(expectedLeafSizes.every((size) => (sizes[size] ?? 0) > 0),
          `power run did not contain every expected liquid leaf size ${expectedLeafSizes.join(",")}: ${JSON.stringify({ final: sizes, witness: dynamicTransitionWitness })}`);
      }
      fail(Object.keys(sizes).every((size) => expectedLeafSizes.includes(size)),
        `power grid contained an unexpected leaf size: ${JSON.stringify(sizes)}`);
      const transitionFaceCount = power?.transitionFaceCount ?? 0;
      const obliqueTransitionFaceCount = power?.obliqueTransitionFaceCount ?? 0;
      const maximumTransitionSizeRatio = power?.maximumTransitionSizeRatio ?? Infinity;
      if (requireTransitionWitness) {
        fail(transitionFaceCount > 0,
          "power run did not publish a cross-scale generalized face");
        fail(obliqueTransitionFaceCount > 0,
          "power run did not publish an oblique cross-scale power face");
      }
      fail(maximumTransitionSizeRatio <= 2,
        `power grid transition size ratio ${maximumTransitionSizeRatio} exceeds 2:1`);
      fail((octree.info.quadtreeMaximumNeighborRatio ?? Infinity) <= 2,
        `hydrostatic octree violated 2:1 balance: ${octree.info.quadtreeMaximumNeighborRatio}`);
      if (!coarseOnlyOctreeRequested) {
        fail(octree.info.globalFineFaceBandTransitionValid === true
        && (octree.info.globalFineFaceBandTransitionRows ?? 0) > 0
        && (octree.info.globalFineFaceBandTransitionAdjacencyCount ?? 0) > 0,
      `hydrostatic Section 5 Delaunay face band did not publish live transition adjacency: ${JSON.stringify({
        valid: octree.info.globalFineFaceBandTransitionValid,
        rows: octree.info.globalFineFaceBandTransitionRows,
        adjacency: octree.info.globalFineFaceBandTransitionAdjacencyCount,
      })}`);
      fail(octree.info.globalFineFaceBandValid === true
        && octree.info.globalFineFaceBandFlags === 0
        && (octree.info.globalFineFaceBandFaceCount ?? 0) > 0
        && octree.info.globalFineFaceBandAcceptedCount === octree.info.globalFineFaceBandFaceCount
        && octree.info.globalFineFaceBandUnresolvedCount === 0
        && octree.info.globalFineFaceBandSampleFailures === 0
        && octree.info.globalFineFaceBandCoarsePhiFailures === 0,
      `hydrostatic Section 5 face fast march is incomplete: ${JSON.stringify({
        valid: octree.info.globalFineFaceBandValid,
        flags: octree.info.globalFineFaceBandFlags,
        faces: octree.info.globalFineFaceBandFaceCount,
        accepted: octree.info.globalFineFaceBandAcceptedCount,
        unresolved: octree.info.globalFineFaceBandUnresolvedCount,
        sampleFailures: octree.info.globalFineFaceBandSampleFailures,
        coarsePhiFailures: octree.info.globalFineFaceBandCoarsePhiFailures,
      })}`);
      fail(octree.info.globalFineFaceBandTransientPowerValid === true
        && octree.info.globalFineFaceBandTransientPowerFlags === 0
        && (octree.info.globalFineFaceBandTransientPowerRows ?? 0) > 0
        && octree.info.globalFineFaceBandTransientPowerEmitted
          === octree.info.globalFineFaceBandTransientPowerSampled
        && octree.info.globalFineFaceBandTransientPowerValidated
          === octree.info.globalFineFaceBandTransientPowerRows,
      `hydrostatic Section 5 transient power graph is incomplete: ${JSON.stringify({
        valid: octree.info.globalFineFaceBandTransientPowerValid,
        flags: octree.info.globalFineFaceBandTransientPowerFlags,
        rows: octree.info.globalFineFaceBandTransientPowerRows,
        emitted: octree.info.globalFineFaceBandTransientPowerEmitted,
        sampled: octree.info.globalFineFaceBandTransientPowerSampled,
        validated: octree.info.globalFineFaceBandTransientPowerValidated,
      })}`);
      fail(octree.info.globalFineFaceBandPointFieldValid === true
        && octree.info.globalFineFaceBandPointFieldFlags === 0
        && (octree.info.globalFineFaceBandPointFieldRows ?? 0) > 0
        && octree.info.globalFineFaceBandPointFieldSolved === octree.info.globalFineFaceBandPointFieldRows,
      `hydrostatic Section 5 point field is incomplete: ${JSON.stringify({
        valid: octree.info.globalFineFaceBandPointFieldValid,
        flags: octree.info.globalFineFaceBandPointFieldFlags,
        rows: octree.info.globalFineFaceBandPointFieldRows,
        solved: octree.info.globalFineFaceBandPointFieldSolved,
      })}`);
      fail(octree.info.globalFineFaceBandPowerPublicationValid === true
        && octree.info.globalFineFaceBandPowerPublicationFlags === 0
        && octree.info.globalFineFaceBandPowerPublicationFaces === power?.faceCount
        && (octree.info.globalFineFaceBandPowerPublicationTargets ?? 0) > 0
        && octree.info.globalFineFaceBandPowerPublicationInterpolated
          === octree.info.globalFineFaceBandPowerPublicationTargets
        && octree.info.globalFineFaceBandPowerPublicationCommitted
          === octree.info.globalFineFaceBandPowerPublicationTargets
        && octree.info.globalFineFaceBandPowerGeneration === octree.octreePowerFaceDiagnostics?.generation,
        `hydrostatic Section 5 final power publication is incomplete: ${JSON.stringify({
        valid: octree.info.globalFineFaceBandPowerPublicationValid,
        flags: octree.info.globalFineFaceBandPowerPublicationFlags,
        faces: octree.info.globalFineFaceBandPowerPublicationFaces,
        targets: octree.info.globalFineFaceBandPowerPublicationTargets,
        interpolated: octree.info.globalFineFaceBandPowerPublicationInterpolated,
        committed: octree.info.globalFineFaceBandPowerPublicationCommitted,
        powerGeneration: octree.info.globalFineFaceBandPowerGeneration,
      })}`);
      }
      const transitionRowCount = power?.topology.transitionRowCount ?? 0;
      const transitionTetrahedronCount = power?.topology.transitionTetrahedronCount ?? 0;
      fail(power?.topology.invalidRowCount === 0
        && power.topology.validRowCount === power.rowCount
        && power.topology.rowsWithTetrahedra === power.rowCount
        && (!requireTransitionWitness || (transitionRowCount > 0 && transitionTetrahedronCount > 0)),
      `live Delaunay topology is incomplete: ${JSON.stringify({ final: power?.topology, witness: dynamicTransitionWitness })}`);
      fail(power?.geometry.invalidFaceCount === 0
        && power.geometry.maximumNormalLengthError <= 2e-4,
      `power-face geometry is invalid: ${JSON.stringify(power?.geometry)}`);
      fail(power?.incidence.invalidEntryCount === 0
        && power.incidence.reciprocityFailureCount === 0,
      `signed face incidence is not reciprocal: ${JSON.stringify(power?.incidence)}`);
      fail(power?.pressureRows.invalidRowCount === 0
        && power.pressureRows.finiteRowCount === power.rowCount
        && power.pressureRows.entryCount > 0
        && power.pressureRows.reciprocityFailureCount === 0
        && power.pressureRows.relativeResidual <= 5e-3,
      `pressure CSR is invalid or does not satisfy A*q=-storedFluxRhs: ${JSON.stringify(power?.pressureRows)}`);
      fail(power?.velocityReconstruction?.flags === 0x8000_0000
        && power.velocityReconstruction.rowCount === power.rowCount
        && power.velocityReconstruction.faceCount === power.faceCount
        && power.velocityReconstruction.incidenceCount === power.incidence.incidenceCount
        && power.velocityReconstruction.reconstructedCount === power.rowCount
        && power.velocityReconstruction.fallbackCount === 0,
      `power velocity reconstruction is invalid: ${JSON.stringify(power?.velocityReconstruction)}`);
      fail(power?.operator?.flags === 0xc000_0000
        && power.operator.firstError === OCTREE_POWER_INVALID_ROW
        && power.operator.rowCount === power.rowCount
        && power.operator.faceCount === power.faceCount
        && power.operator.incidenceCount === power.incidence.incidenceCount
        && power.operator.entryCount > 0
        && power.operator.projectedCount === power.faceCount,
      `generalized pressure operator is invalid: ${JSON.stringify(power?.operator)}`);
      const hydrostaticScenario = scenarioId === "hydrostatic-power-two-level"
        || scenarioId === "hydrostatic-power-large-offset";
      if (hydrostaticScenario) {
        // At t=0 the flat-interface analytic potential must satisfy every row.
        // After surface transport/redistancing, ghost-fluid open rows use the
        // evolved interface distance, so the original flat q is no longer an
        // exact boundary-row solution. The interior analytic residual remains
        // independent, while the solved residual and pressure-potential gates
        // below continue to validate every row, including the open boundary.
        const analyticReferenceResidual = octree.steps > 1
          ? power?.pressureRows.analyticInteriorRelativeResidual
          : power?.pressureRows.analyticGravityRelativeResidual;
        fail((analyticReferenceResidual ?? Infinity) <= 0.02,
        `analytic hydrostatic q does not satisfy the ${octree.steps > 1 ? "interior" : "complete"} power operator with an independent gravity RHS: ${JSON.stringify(power?.pressureRows)}`);
        fail((power?.pressureRows.gravityRhsRelativeError ?? Infinity) <= 2e-4,
        `assembled power RHS differs from independent dt*g.n face fluxes: ${JSON.stringify(power?.pressureRows)}`);
        fail((power?.pressurePotential.relativeL2Error ?? Infinity) <= 0.02,
          `hydrostatic pressure-potential relative L2 error ${power?.pressurePotential.relativeL2Error} exceeds 2%`);
        fail((power?.pressurePotential.maximumAbsoluteError_m2_s ?? Infinity)
          <= 0.02 * (power?.pressurePotential.maximumExpected_m2_s ?? 0),
        `hydrostatic pressure-potential L-infinity error ${power?.pressurePotential.maximumAbsoluteError_m2_s} exceeds 2% of ${power?.pressurePotential.maximumExpected_m2_s}`);
        fail((power?.maximumTransitionNormalVelocity_m_s ?? Infinity) <= 0.002,
          `hydrostatic transition faces generated ${power?.maximumTransitionNormalVelocity_m_s} m/s (limit 0.002)`);
        fail((power?.maximumSpeed_m_s ?? Infinity) <= 0.002,
          `hydrostatic final power faces generated ${power?.maximumSpeed_m_s} m/s (limit 0.002)`);
        fail((power?.maximumDivergence_s ?? Infinity) <= 0.02,
          `hydrostatic final power-cell divergence ${power?.maximumDivergence_s} 1/s exceeds 0.02`);
        fail(Math.abs(power?.volumeDrift ?? Infinity) <= 1e-4,
          `hydrostatic rest volume drift ${power?.volumeDrift} exceeds 1e-4`);
      } else if (scenarioId === "minimal-power-dam-break") {
        const dt_s = power?.pressurePotential.dt_s ?? Infinity;
        const envelope = octree.stabilityEnvelope;
        const finestCellSize_m = createSmokeScenario(scenarioId).scene.voxelDomain?.finestCellSize_m ?? 0;
        const powerCfl = (power?.maximumSpeed_m_s ?? Infinity) * dt_s / finestCellSize_m;
        fail(Number.isFinite(power?.maximumSpeed_m_s) && (power?.maximumSpeed_m_s ?? 0) > 1e-3,
          `minimal dam release is frozen: authoritative power speed ${power?.maximumSpeed_m_s} m/s`);
        fail(Number.isFinite(powerCfl) && powerCfl > 0 && powerCfl <= 1,
          `minimal dam authoritative power CFL ${powerCfl} is outside (0, 1]`);
        // Paper Eq. (3) is the variational residual [grad]^T[V A][F grad]p
        // - [grad]^T[V A]u*. The legacy full-cell divergence reduction omits
        // those face-volume, open-area, and liquid-fraction weights and is not
        // a valid incompressibility gate for moving free-surface cut cells.
        if (powerGenerationAuditRequested) {
          fail((envelope?.maximumProjectedVariationalResidual ?? Infinity) <= 1e-6,
            `minimal dam Eq. (3)-form projected residual ${envelope?.maximumProjectedVariationalResidual} exceeds the 1e-6 QA gate`);
        }
        fail(Math.abs(power?.volumeDrift ?? Infinity) <= 0.01,
          `minimal dam final compact volume drift ${power?.volumeDrift} exceeds 1%`);
        // The UI-parity motion smoke deliberately performs no per-step field
        // readbacks. Apply transient gates only when their envelope was
        // requested; the exhaustive Dawn command above requests and checks it.
        if (envelope) {
          if (!coarseOnlyOctreeRequested) {
            fail(envelope.maximumExactVolumeDrift <= 0.01,
              `minimal dam transient exact-volume drift ${envelope.maximumExactVolumeDrift} exceeds 1%`);
          }
          fail(envelope.maximumComponentCount === 1
            && envelope.minimumDominantComponentFraction >= 0.98,
          `minimal dam liquid disconnected: ${JSON.stringify({
            maximumComponentCount: envelope.maximumComponentCount,
            minimumDominantComponentFraction: envelope.minimumDominantComponentFraction,
          })}`);
        }
        if (!coarseOnlyOctreeRequested) {
          const generation = octree.finalGlobalFineGeneration;
          fail(generation?.transportCommitted === true
          && (generation.transportProcessed ?? 0) > 0
          && generation.transportDepartureOutsideBand === 0
          && generation.transportNonfiniteVelocity === 0
          && generation.transportFaceBandUnavailable === 0
          && generation.transportVelocityUnavailable === 0
          && (generation.transportMaximumDisplacementFineCells ?? Infinity) <= 1,
        `minimal dam Section 5 transport is incomplete: ${JSON.stringify({
          committed: generation?.transportCommitted,
          processed: generation?.transportProcessed,
          departureOutsideBand: generation?.transportDepartureOutsideBand,
          nonfiniteVelocity: generation?.transportNonfiniteVelocity,
          faceBandUnavailable: generation?.transportFaceBandUnavailable,
          velocityUnavailable: generation?.transportVelocityUnavailable,
          maximumDisplacementFineCells: generation?.transportMaximumDisplacementFineCells,
        })}`);
          fail(generation?.topologyRolledBack === false && generation?.topologyFinalizeReason === 0,
          `minimal dam final topology publication rolled back: ${JSON.stringify({
            rolledBack: generation?.topologyRolledBack,
            finalizeReason: generation?.topologyFinalizeReason,
          })}`);
        }
      }
    }
    if (globalFineGenerationTransitionRequested) {
      const container = createSmokeScenario(scenarioId).scene.container;
      const assertAuthoritativeRaster = (label: string, publishedGeneration: number | undefined,
        observed: HybridPresentationSmokeStats | undefined, requireInitialDamCornerCaps = false) => {
        const bounds = observed?.frontInterfaceBounds_m;
        const boundsFinite = bounds !== undefined && bounds.flat(2).every(Number.isFinite);
        const tolerance = Math.max(container.width_m, container.height_m, container.depth_m) * 1e-4;
        const boundsInsideTank = boundsFinite && bounds![0][0] >= -0.5 * container.width_m - tolerance
          && bounds![1][0] <= 0.5 * container.width_m + tolerance
          && bounds![0][1] >= -tolerance && bounds![1][1] <= container.height_m + tolerance
          && bounds![0][2] >= -0.5 * container.depth_m - tolerance
          && bounds![1][2] <= 0.5 * container.depth_m + tolerance;
        const reverse = observed?.reverseView;
        // This extraction is the liquid/air zero set, not a wall-capped closed
        // solid. As the dam wets the floor and container walls, legitimate
        // front rays increasingly have no second liquid/air crossing; a fixed
        // front/back Jaccard threshold therefore measures contact area, not
        // missing triangles. Depth peeling still gives a strict invariant:
        // every back hit must have a front hit on that ray in both views.
        fail((observed?.frontInterfacePixels ?? 0) > 0 && (observed?.backInterfacePixels ?? 0) > 0,
          `${label} did not rasterize both front and back liquid/air crossings: ${JSON.stringify(observed)}`);
        fail(observed?.backOnlyInterfacePixels === 0,
          `${label} depth peeling exposed back crossings without front crossings: ${JSON.stringify(observed)}`);
        fail((reverse?.frontInterfacePixels ?? 0) > 0 && (reverse?.backInterfacePixels ?? 0) > 0
          && reverse?.backOnlyInterfacePixels === 0,
        `${label} reverse depth peeling exposed back crossings without front crossings: ${JSON.stringify(reverse)}`);
        if (scenarioId === "minimal-power-dam-break") {
          fail(observed?.narrowVerticalSlits.count === 0,
            `${label} contains narrow vertical surface slits: ${JSON.stringify(observed?.narrowVerticalSlits)}`);
          fail(reverse?.narrowVerticalSlits.count === 0,
            `${label} reverse view contains narrow vertical surface slits: ${JSON.stringify(reverse?.narrowVerticalSlits)}`);
          if (requireInitialDamCornerCaps) fail((reverse?.wallCornerCapPixels?.[0] ?? 0) >= 8,
            `${label} has a chamfered -x/-z reservoir corner instead of two wall-owned caps: ${JSON.stringify(reverse?.wallCornerCapPixels)}`);
          if (requireInitialDamCornerCaps) fail((observed?.damExposedCornerCapPixels?.[0] ?? 0) >= 4
            && (observed?.damExposedCornerCapPixels?.[1] ?? 0) >= 4,
          `${label} is missing one or both exposed +x/+z dam-corner faces: ${JSON.stringify(observed?.damExposedCornerCapPixels)}`);
        }
        fail(observed?.surfaceGeometrySource === "global-fine-coarse"
          && observed.globalFineCrossingPublished === true
          && observed.presentationFallbackActive === false
          && (observed.globalFineAuthorityLatch ?? 0) !== 0
          && (observed.vertexCount ?? 0) > 0
          && observed.vertexAllocator === observed.vertexCount
          && (observed.activeCubeCount ?? 0) > 0
          && (observed.vertexCount ?? Infinity) <= (observed.vertexCapacity ?? -Infinity)
          && (observed.activeCubeCount ?? Infinity) <= (observed.activeCubeCapacity ?? -Infinity),
        `${label} was not extracted from the clean global fine/coarse publication: ${JSON.stringify(observed)}`);
        fail(boundsInsideTank, `${label} produced non-finite or out-of-tank interface bounds: ${JSON.stringify(bounds)}`);
        const transition = observed?.globalFineAuthorityTransition;
        fail(transition?.cleanFineCoarseRequired === true
          && transition.validGeneration === publishedGeneration
          && transition.retainedGeometrySource === "retained-previous",
        `${label} did not reject unpublished B and retain clean A: ${JSON.stringify(transition)}`);
        fail(transition?.retainedFrontInterfacePixels === observed?.frontInterfacePixels
          && transition?.retainedBackInterfacePixels === observed?.backInterfacePixels
          && transition?.retainedFrontInterfaceHash === observed?.frontInterfaceHash
          && transition?.retainedBackInterfaceHash === observed?.backInterfaceHash,
        `${label} changed raster content after the unpublished-B probe: ${JSON.stringify(transition)}`);
      };
      const initialGeneration = octree.initialGlobalFineGeneration;
      fail(initialGeneration?.publicationValid === true && (initialGeneration.generation ?? 0) > 0,
        `octree t=0 global-fine generation is not published: ${JSON.stringify(initialGeneration)}`);
      assertAuthoritativeRaster("octree t=0 raster", initialGeneration?.generation, octree.initialGlobalFineRaster,
        scenarioId === "minimal-power-dam-break");
      const generation = octree.finalGlobalFineGeneration;
      const raster = octree.finalGlobalFineRaster;
      fail(generation?.publicationValid === true && (generation.generation ?? 0) > 0 && (generation.activePages ?? 0) > 0,
        `octree final global-fine generation is not a nonempty published generation: ${JSON.stringify(generation)}`);
      fail((generation?.validSamples ?? 0) > 0
        && generation?.finiteValidSamples === generation?.validSamples
        && (generation?.negativeValidSamples ?? 0) > 0
        && (generation?.positiveValidSamples ?? 0) > 0,
        `octree final global-fine generation does not contain a finite signed interface: ${JSON.stringify(generation)}`);
      assertAuthoritativeRaster("octree final raster", generation?.generation, raster);
      if (scenarioId === "minimal-power-dam-break" && minimumDamSpread_m !== undefined) {
        const initialBounds = octree.initialGlobalFineRaster?.frontInterfaceBounds_m;
        const finalBounds = raster?.frontInterfaceBounds_m;
        const lateralSpread_m = initialBounds && finalBounds
          ? Math.max(finalBounds[1][0] - initialBounds[1][0], finalBounds[1][2] - initialBounds[1][2])
          : -Infinity;
        fail(lateralSpread_m >= minimumDamSpread_m,
          `minimal dam interface spread ${lateralSpread_m} m is below ${minimumDamSpread_m} m: ${JSON.stringify({ initialBounds, finalBounds })}`);
      }
      if (rasterCheckpointRequested) for (const checkpoint of octree.checkpoints) {
        const checkpointGeneration = checkpoint.globalFineGeneration;
        fail(checkpointGeneration?.publicationValid === true
          && checkpointGeneration.coarseState === 0x8000_0000
          && checkpointGeneration.coarseGeneration === checkpointGeneration.generation
          && checkpointGeneration.topologyRolledBack === false
          && checkpointGeneration.topologyFinalizeReason === 0,
        `octree raster checkpoint at t=${checkpoint.time_s} is not a clean current fine/coarse publication: ${JSON.stringify(checkpointGeneration)}`);
        assertAuthoritativeRaster(`octree raster checkpoint at t=${checkpoint.time_s}`,
          checkpointGeneration?.generation, checkpoint.raster);
      }
    }
    if (sparseStatsRequested) {
      const sparse = octree.sparseVoxelStats;
      const hybrid = octree.hybridPresentationStats;
      const expectedFluidColor = voxelMaterial(VOXEL_MATERIAL_IDS.fluid).baseColorLinear;
      fail(!!sparse, "octree did not expose its sparse voxel render publication");
      fail((sparse?.voxelCount ?? 0) > 0 && (sparse?.brickCount ?? 0) > 0, `octree sparse publication has no records: ${JSON.stringify(sparse)}`);
      fail((sparse?.activeVoxelCount ?? 0) > 0 && (sparse?.activeBrickCount ?? 0) > 0, `octree sparse publication has no active geometry: ${JSON.stringify(sparse)}`);
      fail((sparse?.fluidVoxelCount ?? 0) > 0, "octree sparse publication contains no fluid material voxels");
      fail((sparse?.environmentVoxelCount ?? 0) > 0, "octree sparse publication contains no environment geometry voxels");
      if (scenarioId === "sphere-jet") {
        const sphereVoxels = sparse?.materialVoxelCounts[String(VOXEL_MATERIAL_IDS.sphere)] ?? 0;
        fail(sphereVoxels > 0 && sphereVoxels < 10_000,
          `sphere proxy ownership leaked into empty sparse cells: ${sphereVoxels} sphere voxels`);
      }
      fail((sparse?.nonFiniteRecordCount ?? Infinity) === 0, `octree sparse publication contains ${sparse?.nonFiniteRecordCount} invalid spatial records`);
      fail((sparse?.invalidMaterialCount ?? Infinity) === 0, `octree sparse publication contains ${sparse?.invalidMaterialCount} invalid material IDs`);
      fail(Number.isFinite(sparse?.uiRawVoxelRenderWall_ms) && (sparse?.uiRawVoxelRenderWall_ms ?? 0) > 0,
        `octree sparse raw-voxel WebGPU smoke did not complete: ${sparse?.uiRawVoxelRenderWall_ms}`);
      fail(Number.isFinite(sparse?.uiBrickGridRenderWall_ms) && (sparse?.uiBrickGridRenderWall_ms ?? 0) > 0,
        `octree sparse brick-grid WebGPU smoke did not complete: ${sparse?.uiBrickGridRenderWall_ms}`);
      fail(!!hybrid && hybrid.bodyCount > 0 && Number.isFinite(hybrid.frameWall_ms) && hybrid.frameWall_ms > 0,
        `octree hybrid smooth WebGPU smoke did not complete: ${JSON.stringify(hybrid)}`);
      fail((hybrid?.frontInterfacePixels ?? 0) > 0,
        `octree hybrid smooth WebGPU smoke rasterized no water interfaces: ${JSON.stringify(hybrid)}`);
      fail(expectedFluidColor.every((value, index) => Math.abs(value - (sparse?.fluidColorLinear[index] ?? Infinity)) <= 1e-6), `octree sparse fluid color ${sparse?.fluidColorLinear} differs from authored linear color ${expectedFluidColor}`);
      if (scenarioId === "garden-dam-break") {
        fail(octree.initialFluidBrickStats?.core === 1,
          `garden dam break started with ${octree.initialFluidBrickStats?.core ?? "unknown"} core fluid bricks instead of one`);
        fail((sparse?.fluidBrickResidentCount ?? Infinity) < (sparse?.fluidBrickCapacity ?? 0),
          `garden dam break resident set filled its ${sparse?.fluidBrickCapacity ?? "unknown"}-brick capacity`);
        if ((octree.info.simulatedTime_s ?? 0) >= 1 - 1e-9) {
          fail((sparse?.fluidBrickCoreCount ?? 0) > 1,
            `garden dam break did not migrate beyond its initial core brick: ${JSON.stringify(sparse?.fluidBrickCoreOrigins_m)}`);
          fail(sparse?.sourceBrickFluidVoxelCount === 0,
            `garden dam break left ${sparse?.sourceBrickFluidVoxelCount ?? "unknown"} liquid voxels in its original brick`);
          fail(sparse?.sourceBrickResidency !== "core",
            "garden dam break original brick remained a core fluid allocation");
        }
      }
    }
    if (scenarioId === "dam-break-ui") {
      const envelope = octree.stabilityEnvelope;
      // Compact volume authority is the clean, current global-fine controller.
      // The reconstructed cubic field is a spatial/raster QA proxy and must
      // not be mislabeled as the physical volume integral.
      const finalDrift = octree.info.volumeDrift ?? Infinity;
      fail((envelope?.nonFiniteVelocityCount ?? Infinity) === 0, `octree dam break encountered ${envelope?.nonFiniteVelocityCount} non-finite staged velocities`);
      if ((octree.info.adaptiveFaceTransportedCount ?? 0) === 0) {
        fail((envelope?.maximumProjectionEnergyRatio ?? Infinity) <= 1.1, `octree pressure projection amplified kinetic energy by ${envelope?.maximumProjectionEnergyRatio}`);
      }
      fail((envelope?.peakComponentCfl ?? Infinity) <= 3, `octree dam-break peak CFL ${envelope?.peakComponentCfl} exceeds the three-cell backstop`);
      fail((envelope?.maximumPressureRelativeResidual ?? Infinity) <= 1e-4,
        `octree dam-break pressure residual peaked at ${envelope?.maximumPressureRelativeResidual}`);
      fail((envelope?.maximumExactVolumeDrift ?? Infinity) <= 0.01, `octree dam-break level-set volume drift peaked at ${envelope?.maximumExactVolumeDrift}`);
      fail(Math.abs(finalDrift) <= 0.01, `octree dam-break final level-set volume drift ${finalDrift} exceeds 1%`);
      fail((envelope?.minimumDominantComponentFraction ?? -Infinity) >= 0.98, `octree dam-break dominant component fell to ${envelope?.minimumDominantComponentFraction}`);
      if (rasterCheckpointRequested) {
        fail(octree.checkpoints.length > 0, "octree dam break produced no raster checkpoints");
        for (const checkpoint of octree.checkpoints) {
          fail((checkpoint.raster?.frontInterfacePixels ?? 0) > 0,
            `octree fine-level-set raster produced no visible interface at t=${checkpoint.time_s.toFixed(2)} s: ${JSON.stringify(checkpoint.raster)}`);
          fail((checkpoint.raster?.backInterfacePixels ?? 0) > 0,
            `octree fine-level-set raster produced no exit interface at t=${checkpoint.time_s.toFixed(2)} s: ${JSON.stringify(checkpoint.raster)}`);
        }
      }
      const initialInterfaceFaces = octree.matchedSummary.interfaceFaceCount;
      const impact = octree.checkpoints.reduce<typeof octree.checkpoints[number] | undefined>((best, sample) => {
        if (sample.time_s < 0.9 || sample.time_s > 1.3) return best;
        return !best || Math.abs(sample.time_s - 1.1) < Math.abs(best.time_s - 1.1) ? sample : best;
      }, undefined);
      if (impact && initialInterfaceFaces > 0) {
        fail(impact.summary.interfaceFaceCount <= 6 * initialInterfaceFaces, `octree dam-break interface topology expanded to ${impact.summary.interfaceFaceCount} faces from ${initialInterfaceFaces} near impact`);
        fail(impact.summary.enclosedAirCells <= 8, `octree dam-break formed ${impact.summary.enclosedAirCells} enclosed air cells near impact`);
      }
      if (tall && octree.grid.every((value, axis) => value === tall.grid[axis])) {
        {
          const octreePeak = envelope?.peakLiquidSpeed_m_s ?? Infinity;
          const tallPeak = tall.stabilityEnvelope?.peakLiquidSpeed_m_s ?? 0;
          const peakRatio = octreePeak / Math.max(tallPeak, 1e-9);
          fail(Number.isFinite(peakRatio) && peakRatio >= 0.5 && peakRatio <= 2,
            `octree authoritative power peak-speed ratio ${peakRatio} (${octreePeak} / ${tallPeak} m/s) is outside [0.5, 2] versus tall-cell`);
        }
        const compactVelocity = octree.compactVelocityRaster;
        const cubicCellCount = octree.grid[0] * octree.grid[1] * octree.grid[2];
        fail(!!octree.velocityParityField && !!octree.velocityParityVolume,
          "octree did not publish its final compact power-cell velocity parity readback");
        fail(!!tall.velocityParityField && !!tall.velocityParityVolume,
          "tall-cell did not publish its final collocated velocity parity readback");
        fail(compactVelocity?.publicationValid === true,
          `octree compact velocity publication was invalid: ${JSON.stringify(compactVelocity)}`);
        fail(compactVelocity?.coveredCells === cubicCellCount && compactVelocity.overlapCells === 0
          && compactVelocity.invalidRows === 0,
        `octree compact velocity raster did not form a clean ${cubicCellCount}-cell partition: ${JSON.stringify(compactVelocity)}`);
        const velocityMetrics = damBreakVelocityParityMetrics(results);
        fail(velocityMetrics !== undefined, "octree/tall-cell final velocity parity metrics were unavailable");
        if (velocityMetrics) for (const failure of velocityParityFailures(velocityMetrics)) {
          fail(false, `octree/tall-cell ${failure} failed declared limits ${JSON.stringify(DAM_BREAK_VELOCITY_PARITY_LIMITS)}; metrics ${JSON.stringify(velocityMetrics)}`);
        }
        const finalComparison = compareScalarFields(
          octree.finalSummary ? octree.checkpoints.at(-1)?.field ?? octree.matchedField : octree.matchedField,
          tall.finalSummary ? tall.checkpoints.at(-1)?.field ?? tall.matchedField : tall.matchedField,
          ...octree.grid
        );
        fail(finalComparison.wetIntersectionOverUnion >= 0.60, `octree dam-break final wet-IoU ${finalComparison.wetIntersectionOverUnion} is below the 0.60 tall-cell parity floor`);
        fail(finalComparison.centroidDistanceCells === null || finalComparison.centroidDistanceCells <= 6, `octree dam-break final centroid differs from tall-cell by ${finalComparison.centroidDistanceCells} cells`);
        for (const checkpoint of octree.checkpoints) {
          const reference = tall.checkpoints.find(({ time_s }) => Math.abs(time_s - checkpoint.time_s) <= 0.01);
          if (!reference) continue;
          const comparison = compareScalarFields(checkpoint.field, reference.field, ...octree.grid);
          fail(comparison.wetIntersectionOverUnion >= 0.60, `octree dam-break wet-IoU ${comparison.wetIntersectionOverUnion} at t=${checkpoint.time_s.toFixed(2)} s is below 0.60`);
          fail(comparison.centroidDistanceCells === null || comparison.centroidDistanceCells <= 6, `octree dam-break centroid differs from tall-cell by ${comparison.centroidDistanceCells} cells at t=${checkpoint.time_s.toFixed(2)} s`);
        }
      }
    }
  }
  if (scenarioId === "brick-quad-dam-break") {
    // The scene's whole point is cross-brick transport: the domain must be
    // exactly four 8-cubed fluid bricks (2x2 in x/z at one brick of height),
    // and the single seeded quadrant must wet all four brick columns.
    for (const result of results) {
      fail(result.grid[0] === 16 && result.grid[1] === 8 && result.grid[2] === 16,
        `${result.method} grid ${result.grid.join("x")} is not the intended 16x8x16 four-brick domain`);
      const [nx, ny, nz] = result.grid;
      const wetBrickColumns = (field: Float32Array) => {
        const wet = new Set<string>();
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
          if (field[x + nx * (y + ny * z)] >= 0.5) wet.add(`${Math.floor(x / 8)},${Math.floor(z / 8)}`);
        }
        return wet;
      };
      if (result.checkpoints.length > 0) {
        const first = result.checkpoints[0];
        fail(wetBrickColumns(first.field).size >= 2,
          `${result.method} water had not crossed a brick boundary by t=${first.time_s.toFixed(2)} s`);
        const everWet = new Set<string>();
        for (const checkpoint of result.checkpoints) for (const key of wetBrickColumns(checkpoint.field)) everWet.add(key);
        fail(everWet.size === 4, `${result.method} wet only ${everWet.size} of 4 brick columns (${[...everWet].sort().join(" | ")})`);
        fail(result.checkpoints.some((checkpoint) => wetBrickColumns(checkpoint.field).has("1,1")),
          `${result.method} water never reached the far (+x/+z) brick quadrant opposite the seed`);
      }
      console.log(JSON.stringify({
        scenario: scenarioId, method: result.method, phase: "brick-quad-coverage", front_m: result.info.front_m,
        checkpoints: result.checkpoints.map((checkpoint) => ({ time_s: checkpoint.time_s, wetBrickColumns: [...wetBrickColumns(checkpoint.field)].sort() }))
      }));
    }
    if (octree && sparseStatsRequested) {
      // The full-height column places the initial phi zero crossing on the
      // brick faces, so the seeded brick starts as a surface-band (halo)
      // residency rather than a core one; what matters is that the band is
      // resident from the start and that by the end the spread interface is a
      // core crossing in more than one brick.
      fail((octree.initialFluidBrickStats?.resident ?? 0) >= 1,
        `brick-quad dam break started with ${octree.initialFluidBrickStats?.resident ?? "unknown"} resident fluid bricks`);
      fail((octree.sparseVoxelStats?.fluidBrickResidentCount ?? 0) > 1,
        `brick-quad dam break ended with ${octree.sparseVoxelStats?.fluidBrickResidentCount ?? "unknown"} resident fluid bricks; cross-brick flow must keep more than one resident`);
      fail((octree.sparseVoxelStats?.fluidBrickCoreCount ?? 0) >= 2,
        `brick-quad dam break ended with ${octree.sparseVoxelStats?.fluidBrickCoreCount ?? "unknown"} core fluid bricks; the spread interface must cross more than one brick`);
    }
  }
  if (scenarioId === "ocean-seiche") {
    // The scene's whole point is a long gravity wave traversing a wide calm
    // tank: verify the exact intended grid and that the surface disturbance
    // released at the -x wall visibly crosses into the far half of the tank,
    // and log the surface-height profile time series as propagation evidence.
    const scene = createSmokeScenario(scenarioId).scene;
    const cellHeight_m = scene.container.height_m;
    for (const result of results) {
      fail(result.grid[0] === 320 && result.grid[1] === 96 && result.grid[2] === 80,
        `${result.method} grid ${result.grid.join("x")} is not the intended 320x96x80 ocean domain`);
      const [nx, ny, nz] = result.grid;
      const columnHeights = (field: Float32Array) => {
        const heights = new Float64Array(nx);
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
          heights[x] += field[x + nx * (y + ny * z)];
        }
        for (let x = 0; x < nx; x += 1) heights[x] /= nz;
        return heights;
      };
      const xWorld = (x: number) => -0.5 * scene.container.width_m + (x + 0.5) * scene.container.width_m / nx;
      const stationCount = 12;
      const stations = Array.from({ length: stationCount }, (_, i) => Math.min(nx - 1, Math.round((i + 0.5) * nx / stationCount)));
      const baselineHeight_cells = scene.container.fillFraction * ny;
      const minimumFarHalfDisturbance_cells = minimumOceanFarHalfDisturbanceCells(scene.container.width_m);
      let crestReach_m = -Infinity;
      let farHalfDisturbance_cells = 0;
      const series = result.checkpoints.map((checkpoint) => {
        const heights = columnHeights(checkpoint.field);
        let crestX = 0;
        for (let x = 1; x < nx; x += 1) {
          if (heights[x] > heights[crestX]) crestX = x;
          if (xWorld(x) > 0) farHalfDisturbance_cells = Math.max(farHalfDisturbance_cells, Math.abs(heights[x] - baselineHeight_cells));
        }
        crestReach_m = Math.max(crestReach_m, xWorld(crestX));
        return {
          time_s: checkpoint.time_s,
          crestX_m: Number(xWorld(crestX).toFixed(3)),
          crestHeight_cells: Number(heights[crestX].toFixed(2)),
          stationHeights_cells: stations.map((x) => Number(heights[x].toFixed(2)))
        };
      });
      console.log(JSON.stringify({
        scenario: scenarioId, method: result.method, phase: "ocean-wave-profile",
        baselineHeight_cells, cellHeight_m: cellHeight_m / ny,
        minimumFarHalfDisturbance_cells: Number(minimumFarHalfDisturbance_cells.toFixed(3)),
        farHalfDisturbance_cells: Number(farHalfDisturbance_cells.toFixed(3)),
        stationX_m: stations.map((x) => Number(xWorld(x).toFixed(3))), checkpoints: series
      }));
      if (result.checkpoints.length >= 3) {
        // A dispersive/reflected wave can retain its global tallest crest near
        // the release wall even after the leading disturbance has crossed the
        // tank. Gate the actual far-half signal instead of the argmax crest.
        fail(farHalfDisturbance_cells >= minimumFarHalfDisturbance_cells,
          `${result.method} far-half surface disturbance reached only ${farHalfDisturbance_cells.toFixed(3)} cells (required ${minimumFarHalfDisturbance_cells.toFixed(3)}; global crest max x ${crestReach_m.toFixed(3)} m)`);
      }
    }
  }
  return failures;
}

const failures: string[] = [];
try {
  for (const scenarioId of selectedScenarios()) {
    const scenario = createSmokeScenario(scenarioId);
    applySceneOverrides(scenario.scene);
    const oracleSteps = Math.max(1, Math.round(oracleStepsOverride ?? scenario.oracleSteps));
    const target_s = Math.max(targetOverride ?? scenario.target_s, oracleSteps * scenario.scene.numerics.maxDt_s);
    console.log(JSON.stringify({ scenario: scenarioId, phase: "scenario", description: scenario.description, target_s, oracleSteps, quality, methods: methods.map((method) => method.id), cpuOracle: runCPUOracle }));
    if (methods.some((method) => method.id === "tall-cell")) {
      const layout = singleTallCellProbe
        ? createSingleTallCellProbeLayout(scenario.scene, quality, 2048, singleTallCellProbe)
        : createTallCellLayout(scenario.scene, quality, 2048,
          regularLayersOverride === undefined && maximumNeighborDeltaOverride === undefined ? undefined : {
          ...(regularLayersOverride === undefined ? {} : { regularLayers: regularLayersOverride }),
          ...(maximumNeighborDeltaOverride === undefined ? {} : { maximumNeighborDelta: maximumNeighborDeltaOverride })
        });
      console.log(JSON.stringify({
        scenario: scenarioId, phase: "interrogation", interrogation: "tall-cell-activity", stage: "planned",
        cubicGrid: [layout.nx, layout.fineNy, layout.nz], storedNy: layout.packedNy,
        requestedRegularLayers: regularLayersOverride ?? tallCellSettings[quality].regularLayers,
        effectiveRegularLayers: layout.settings.regularLayers, compressionRatio: layout.compressionRatio,
        activeCompressionRatio: layout.activeCompressionRatio, activeSampleCount: layout.activeSampleCount,
        planning: layout.planning,
        activity: summarizeTallCellActivity(layout.columnBases, layout.fineNy, layout.settings.regularLayers, layout.nx, layout.nz),
        singleTallCellProbe: layout.singleTallCellProbe
      }));
    }
    const results: GPUSmokeResult[] = [];
    for (const method of methods) results.push(await runGPU(scenarioId, method, target_s, oracleSteps));
    const velocityParity = scenarioId === "dam-break-ui" ? damBreakVelocityParityMetrics(results) : undefined;
    const velocityParityOctree = results.find((result) => result.method === "octree");
    const velocityParityTall = results.find((result) => result.method === "tall-cell");
    if (scenarioId === "dam-break-ui" && velocityParityOctree && velocityParityTall
      && velocityParityOctree.grid.every((value, axis) => value === velocityParityTall.grid[axis])) {
      console.log(JSON.stringify({
        scenario: scenarioId,
        phase: "velocity-parity",
        candidate: "octree",
        reference: "tall-cell",
        metrics: velocityParity,
        limits: DAM_BREAK_VELOCITY_PARITY_LIMITS,
        failures: velocityParity ? velocityParityFailures(velocityParity) : ["velocity parity readback unavailable"],
        compactRaster: velocityParityOctree.compactVelocityRaster,
      }));
    }
    if (!performanceProfileRequested) failures.push(...invariantFailures(scenarioId, results));

    const tallResult = results.find((result) => result.method === "tall-cell"), uniformResult = results.find((result) => result.method === (singleTallCellProbe ? "tall-cell-control" : "uniform"));
    if (tallResult && uniformResult) {
      const ratio = (uniformValue?:number,tallValue?:number) => typeof uniformValue==="number"&&typeof tallValue==="number"&&Number.isFinite(uniformValue)&&Number.isFinite(tallValue)&&tallValue>0 ? uniformValue/tallValue : null;
      const stages = ["advection_ms","pressure_ms","projection_ms","rigidCoupling_ms","diagnostics_ms"] as const;
      const gpuStageSpeedups = Object.fromEntries(stages.map((stage) => [stage,ratio(uniformResult.info.gpuTimings?.[stage],tallResult.info.gpuTimings?.[stage])]));
      console.log(JSON.stringify({
        scenario:scenarioId,phase:"performance-comparison",baseline:"uniform",candidate:"tall-cell",
        tallBackend:tallResult.info.gridKind,wallRuntimeSpeedup:ratio(uniformResult.runtime_ms,tallResult.runtime_ms),
        constructionSpeedup:ratio(uniformResult.construction_ms,tallResult.construction_ms),gpuStageSpeedups,
        activeSampleReduction:1-(tallResult.info.activeSampleCount??tallResult.info.cellCount)/(uniformResult.info.activeSampleCount??uniformResult.info.cellCount),
        properties:{
          tallRepresentedVolumeDrift:tallResult.info.representedVolumeDrift,uniformRepresentedVolumeDrift:uniformResult.info.representedVolumeDrift,
          representedVolumeDriftDelta:(tallResult.info.representedVolumeDrift??0)-(uniformResult.info.representedVolumeDrift??0),
          tallNonFiniteCount:tallResult.info.nonFiniteCount,uniformNonFiniteCount:uniformResult.info.nonFiniteCount,
          tallPressureRelativeResidual:tallResult.info.pressureRelativeResidual,uniformPressureRelativeResidual:uniformResult.info.pressureRelativeResidual
        }
      }));
      if (singleTallCellProbe) {
        const layout = createSingleTallCellProbeLayout(scenario.scene, quality, 2048, singleTallCellProbe);
        const probe = layout.singleTallCellProbe!;
        console.log(JSON.stringify({
          scenario: scenarioId, phase: "single-tall-cell-difference", control: "restricted-cubic-limit", candidate: "one-tall-cell",
          probe, time_s: tallResult.info.simulatedTime_s,
          global: compareScalarFields(tallResult.finalSummary ? tallResult.checkpoints.at(-1)?.field ?? tallResult.matchedField : tallResult.matchedField, uniformResult.finalSummary ? uniformResult.checkpoints.at(-1)?.field ?? uniformResult.matchedField : uniformResult.matchedField, ...tallResult.grid),
          locality: compareSingleTallCellNeighborhood(tallResult.matchedField, uniformResult.matchedField, ...tallResult.grid, probe.x, probe.z),
          velocity: { tall: tallResult.velocitySummary, uniform: uniformResult.velocitySummary }
        }));
        const pairCount = Math.min(tallResult.checkpoints.length, uniformResult.checkpoints.length);
        for (let index = 0; index < pairCount; index += 1) {
          const tallCheckpoint = tallResult.checkpoints[index], uniformCheckpoint = uniformResult.checkpoints[index];
          const profileTop = Math.min(tallResult.grid[1], probe.height + 4);
          const probeProfile = (field: Float32Array) => Array.from({ length: profileTop }, (_, y) => field[probe.x + tallResult.grid[0] * (y + tallResult.grid[1] * probe.z)]);
          const velocityLocality = (left?: Float32Array, right?: Float32Array) => {
            if (!left || !right) return undefined;
            const magnitude = velocityDifferenceMagnitude(left, right);
            return compareSingleTallCellNeighborhood(magnitude, new Float32Array(magnitude.length), ...tallResult.grid, probe.x, probe.z);
          };
          console.log(JSON.stringify({
            scenario: scenarioId, phase: "single-tall-cell-checkpoint", time_s: tallCheckpoint.time_s, probe,
            global: compareScalarFields(tallCheckpoint.field, uniformCheckpoint.field, ...tallResult.grid),
            locality: compareSingleTallCellNeighborhood(tallCheckpoint.field, uniformCheckpoint.field, ...tallResult.grid, probe.x, probe.z),
            probeVolumeProfile: { candidate: probeProfile(tallCheckpoint.field), control: probeProfile(uniformCheckpoint.field) },
            velocityBeforeProjection: velocityLocality(tallCheckpoint.preProjectionVelocity, uniformCheckpoint.preProjectionVelocity),
            velocityAfterProjection: velocityLocality(tallCheckpoint.postProjectionVelocity, uniformCheckpoint.postProjectionVelocity)
          }));
        }
      }
    }

    const grid = results[0].grid;
    const cpu = runCPUOracle ? runMatchedCPUOracle(scenarioId, grid, oracleSteps) : undefined;
    for (let left = 0; left < results.length; left += 1) for (let right = left + 1; right < results.length; right += 1) {
      console.log(JSON.stringify({ scenario: scenarioId, phase: "discrepancy", left: results[left].method, right: results[right].method, oracleSteps, metrics: compareScalarFields(results[left].matchedField, results[right].matchedField, ...grid) }));
      // The devolution-over-time curve: field discrepancy at every checkpoint,
      // matched by index (all methods advance on the same fixed cadence).
      const pairCount = Math.min(results[left].checkpoints.length, results[right].checkpoints.length);
      for (let index = 0; index < pairCount; index += 1) {
        const a = results[left].checkpoints[index], b = results[right].checkpoints[index];
        console.log(JSON.stringify({
          scenario: scenarioId, phase: "checkpoint-comparison", left: results[left].method, right: results[right].method,
          time_s: a.time_s, leftVolumeCellSum: a.summary.cellSum, rightVolumeCellSum: b.summary.cellSum,
          leftComponentCount: a.summary.componentCount, rightComponentCount: b.summary.componentCount,
          metrics: compareScalarFields(a.field, b.field, ...grid)
        }));
      }
    }
    if (cpu) for (const result of results) {
      console.log(JSON.stringify({ scenario: scenarioId, phase: "discrepancy", left: result.method, right: "cpu-reference", oracleSteps, metrics: compareScalarFields(result.matchedField, cpu.field, ...grid) }));
    }
    console.log(JSON.stringify({ scenario: scenarioId, phase: "scenario-complete",
      performanceProfile: performanceProfileRequested,
      passedInvariants: performanceProfileRequested ? undefined : invariantFailures(scenarioId, results).length === 0,
      qualityGates: performanceProfileRequested ? "skipped" : "evaluated" }));
  }
} finally {
  Reflect.deleteProperty(globalThis, "navigator");
}

if (failures.length > 0) throw new Error(`WebGPU smoke invariant failures:\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
